"""The renderer: rasterise, shade, composite.

Pipeline order, which is the period-correct one rather than the path-traced one:

    supersample -> rasterise opaque z-buffer -> reconstruct fragment attributes
    -> evaluate node graph per material -> resolve closure to a reflectance
    model -> light it -> ray-traced reflection/refraction -> A-buffer
    transparency -> fog -> downsample

Everything here is bpy-free so the whole thing can be exercised headlessly.
"""

import os
import copy
import math

import numpy as np

from . import lights as LI
from . import shadowmask as SM
from . import stats as ST
from . import mathx as M
from . import raster
from . import reyes as REYES            # R251 C119 (MAT-B)
from . import shading as SH
from . import combine as CB          # R251 material pack (MAT-A)
from .bvh import BVH
from .nodeeval import Closure, GraphEvaluator, ShadeContext, to_color, to_value
from .texture import Texture

EMPTY = -1


# ------------------------------------------------------------------ camera


#: R251 C058: the Y-shear caps, ZDoom's (32 degrees up, 56 down); the
#: engines' own constants, never a dial
YSHEAR_UP_DEG, YSHEAR_DOWN_DEG = 32.0, 56.0


def camera_basis(camera):
    """The camera's rigid float32 world matrix (R:34-44's own two
    branches, moved here so the cylinder sky and `camera_matrices` read
    ONE basis): the identity at z = 8 for no camera, else the matrix
    with object scale stripped."""
    if camera is None or camera.matrix_world is None:
        mw = np.eye(4, dtype=np.float32)
        mw[2, 3] = 8.0
        return mw
    # R205: strip object scale from the camera basis. A camera under
    # a scaled parent (the .3DS/FBX $$$DUMMY rigs) composes the same
    # picture but multiplies every camera-space DEPTH by 1/S -- fog,
    # clip and DoF all read wrong distances. Blender ignores camera
    # scale; so does Halcyon now. Rigid matrices pass through
    # bitwise untouched.
    return M.rigid_camera_matrix(camera.matrix_world)


def camera_yaw(mw):
    """The heading of a camera basis (radians about world Z); the level
    camera and the cylinder sky read ONE function. Blender: -Z looks,
    +X right, +Y up. Straight up or down, the right axis carries the
    heading."""
    f = -mw[:3, 2]
    r = mw[:3, 0]
    if math.hypot(float(f[0]), float(f[1])) > 1e-6:
        return math.atan2(float(f[1]), float(f[0]))
    return math.atan2(float(r[0]), -float(r[1]))


def clip_jitter(st, proj, rw, rh):
    """The clip-space translation of a pass: the float32 4x4 J the
    accumulation jitter and the stereo eye ride, or None when the pass
    carries neither (the J block's own lines, R251: moved here so the
    cylinder sky reads the same J).

    The accumulation pass's subpixel offset, as a clip-space
    translation: J @ vp shifts the whole projection by a fraction of
    a pixel, exactly what the OpenGL accumulation buffer's
    glTranslate jitter did. A stereo eye rides the same matrix:
    its off-axis frustum IS a clip-space x translation, sized so
    the convergence plane lands at zero parallax.
    """
    jit = getattr(st, '_accum_jitter', None)
    ster = getattr(st, '_stereo', None)
    if jit is None and ster is None:
        return None
    J = np.eye(4, dtype=np.float32)
    if jit is not None:
        J[0, 3] = 2.0 * float(jit[0]) / rw
        J[1, 3] = 2.0 * float(jit[1]) / rh
    if ster is not None:
        s_off, conv = ster
        J[0, 3] += float(proj[0, 0]) * float(s_off) / max(float(conv),
                                                          1e-4)
    return J


def yshear_active(camera, st):
    """R251 C058: does this frame render its pitch as a Y-shear?
    Perspective cameras only; never inside a panorama strip."""
    return bool(getattr(st, 'camera_yshear', False)) and camera is not None \
        and str(getattr(camera, 'type', 'PERSP')) == 'PERSP' \
        and not getattr(st, '_pano_strip', False)


def level_camera(mw):
    """R251 C058: the camera basis made LEVEL and unrolled, and the
    pitch it had (radians, + looking up). A basis already level and
    unrolled (|f_z| < 1e-7 and |r_z| < 1e-7) passes through bitwise --
    that branch decides whether an existing scene's pixels can move
    under the flag. Position untouched."""
    f = -mw[:3, 2]
    r = mw[:3, 0]
    if abs(float(f[2])) < 1e-7 and abs(float(r[2])) < 1e-7:
        return mw, 0.0
    fz = min(max(float(f[2]), -1.0), 1.0)
    theta = math.asin(fz)
    yaw = camera_yaw(mw)
    cy, sy = math.cos(yaw), math.sin(yaw)
    out = mw.copy()
    out[:3, 0] = np.array([sy, -cy, 0.0], np.float32)   # right' = f' x Z
    out[:3, 1] = np.array([0.0, 0.0, 1.0], np.float32)  # up' = world Z
    out[:3, 2] = np.array([-cy, -sy, 0.0], np.float32)  # -forward'
    return out, theta


def camera_matrices(camera, width, height):
    """(view, proj, viewproj, eye) from a Camera.

    R251: a camera flagged `_yshear` (set by render() from
    `camera_yshear`) renders LEVEL and slides the projection centre by
    focal x tan(pitch) -- Heretic / Hexen / Build's Y-shear (C058); a
    camera carrying `_lens = (lx, ly, d_f)` (the lens-pass road, C098)
    has its window sheared so the focus plane at d_f stays put while
    the eye moves by (lx, ly) across the aperture. Level first, then the
    slide, then the lens shear; `eye` stays the centre eye.
    """
    mw = camera_basis(camera)
    theta = 0.0
    if getattr(camera, '_yshear', False):
        mw, theta = level_camera(mw)
    view = np.linalg.inv(mw).astype(np.float32)
    eye = mw[:3, 3].astype(np.float32)
    if camera is not None and camera.projection is not None:
        proj = np.asarray(camera.projection, np.float32)
    else:
        near = camera.clip_start if camera else 0.1
        far = camera.clip_end if camera else 1000.0
        aspect = width / max(height, 1)
        if camera is not None and camera.type == 'ORTHO':
            half = (camera.ortho_scale or 6.0) * 0.5
            proj = LI._ortho(half, near, far)
            proj[0, 0] = 1.0 / (half * aspect)
            proj[1, 1] = 1.0 / half
        else:
            lens = camera.lens if camera else 50.0
            sensor = camera.sensor if camera else 36.0
            fov_x = 2.0 * np.arctan(sensor * 0.5 / max(lens, 1e-3))
            fov_y = 2.0 * np.arctan(np.tan(fov_x * 0.5) / max(aspect, 1e-6))
            proj = LI._persp(fov_y, aspect, near, far)
    if theta != 0.0:
        # C058: the NDC slide, rounded once. clip_y' = clip_y + s*z_v;
        # with clip_w = -z_v (row 3 is (0,0,-1,0)): ndc_y' = ndc_y - s.
        # proj.copy() is load-bearing: np.asarray returns the camera's
        # OWN float32 array, and the slide must never accumulate on it
        tc = min(max(theta, -math.radians(YSHEAR_DOWN_DEG)),
                 math.radians(YSHEAR_UP_DEG))
        s = np.float32(float(proj[1, 1]) * math.tan(tc))
        proj = proj.copy()
        proj[1, 2] = np.float32(proj[1, 2] + s)
    lens_pt = getattr(camera, '_lens', None)
    if lens_pt is not None:
        # C098: the accumulation buffer's lens shear (Haeberli & Akeley
        # 1990, Appendix B): a view-space point at depth d moves to
        # x' = x + lx*(d/d_f - 1), so a vertex ON the focus plane d_f
        # does not move; the eye returned stays the CENTRE eye
        lx, ly, d_f = lens_pt
        S = np.eye(4, dtype=np.float32)
        S[0, 2] = np.float32(-lx / d_f)
        S[0, 3] = np.float32(-lx)
        S[1, 2] = np.float32(-ly / d_f)
        S[1, 3] = np.float32(-ly)
        proj = (proj @ S).astype(np.float32)
    return view, proj, (proj @ view).astype(np.float32), eye


def pixel_footprint(camera, proj, height):
    """How much of the world one rendered pixel covers.

    Returns (radians, metres): the angle a pixel subtends for a perspective
    camera, or the constant width it covers for an orthographic one. The
    infinite water plane needs this to know which waves are small enough that
    drawing them would only alias -- and it was reading a hard-coded guess of
    0.002 rad, which is roughly four times coarser than a 480-row frame at a
    normal lens and is why the waves stopped being drawn so close in.

    `height` is the *rendered* height, so supersampling is already in it: a
    frame rendered at 4x resolves waves a quarter the size, which it should.
    """
    if camera is not None and getattr(camera, 'type', '') == 'ORTHO':
        return 0.0, float(getattr(camera, 'ortho_scale', 6.0)) / max(height, 1)
    f = abs(float(proj[1, 1])) if proj is not None else 1.0
    fov_y = 2.0 * np.arctan(1.0 / max(f, 1e-6))
    return float(fov_y) / max(height, 1), 0.0


# ---------------------------------------------------------------- textures


_TEX_CACHE = {}

#: R170: rasterised G-buffers by content key (mesh fingerprint + exact
#: view-projection bytes + every raster dial). An idle viewport refine
#: and a repeated F12 re-used ~220ms of identical raster arrays per
#: frame; a hit restores them as copies in a few milliseconds. Small on
#: purpose: three entries covers draft+refine+F12 sizes of one scene.
#: R180: whether the LAST in-process frame wanted GPU shading, whether it
#: got it, and why not -- so the ENGINE can put the verdict in Blender's
#: own UI. The field lost 38 minutes to a frame that silently shaded on
#: the CPU at 16x supersample: the reason was one console line nobody had
#: open. A refusal that expensive belongs in the interface.
LAST_GPU_VERDICT = {'wanted': False, 'engaged': False, 'why': ''}

_GBUF_CACHE = {}
_GBUF_STATS = {'hits': 0, 'misses': 0, 'bytes': 0}
#: R251 C001: the supersample factors the N64 coverage skip was named for
#: (once each, so a viewport redraw never spams the console)
_N64_SKIP_SAID = set()
#: R172: the raster cache is bounded by BYTES, not by a slot count. The
#: field's draft storm inserted a new key per jittered draft and the old
#: 4-slot LRU evicted the refine and F12 entries between uses
#: ('MISS(evicted)' on an unchanged view). Viewport entries are a few MB;
#: F12 supersampled entries tens of MB -- a byte budget holds many of the
#: former without letting the latter hoard memory. Count backstop stays.
_GBUF_BUDGET_BYTES = 256 * 1024 * 1024
_GBUF_CAP = 12


def _gbuf_entry_bytes(ent):
    n = 0
    for v in ent.values():
        if v is not None and hasattr(v, 'nbytes'):
            n += int(v.nbytes)
    return n


def _gbuf_cache_pop(key):
    ent = _GBUF_CACHE.pop(key, None)
    if ent is not None:
        _GBUF_STATS['bytes'] = max(
            _GBUF_STATS['bytes'] - _gbuf_entry_bytes(ent), 0)
    return ent


def clear_caches():
    _TEX_CACHE.clear()
    LI.clear_shadow_cache()


def prepare_textures(scene, settings):
    """ImageBuffers -> sampling-ready Textures, with the era's limits applied.

    Mip building and colour quantisation are not free, and neither depends on
    anything that changes between frames, so the result is cached per image and
    per the settings that affect it.
    """
    out = {}
    sig = (settings.tex_filter, settings.tex_wrap_default, settings.tex_max_size,
           settings.tex_quantize, settings.tex_mipmap, settings.color_management,
           settings.input_gamma_naive,
           # R251 texture pack: prep-time storage laws (content changes)
           str(getattr(settings, 'tex_format', 'NONE')),
           str(getattr(settings, 'tex_tmem_format', 'OFF')),
           str(getattr(settings, 'tex_compress', 'NONE')),
           bool(getattr(settings, 'tex_colorkey', False)))
    images = getattr(scene, 'images', None) or {}
    for key, buf in images.items():
        if buf is None:
            continue
        px = getattr(buf, 'pixels', None)
        if px is None:
            continue
        ckey = (key, id(px), px.shape, sig)
        cached = _TEX_CACHE.get(ckey)
        if cached is not None:
            out[key] = cached
            continue
        tex = Texture(px, name=getattr(buf, 'name', ''),
                      colorspace=getattr(buf, 'colorspace', 'sRGB'),
                      wrap=settings.tex_wrap_default, filt=settings.tex_filter)
        # R251: "encoded" is the one fact the storage laws need -- True
        # when the pixels ARE the file's bytes (no decode ran); a storage
        # law on a decoded image re-encodes, quantises and decodes with
        # the pipeline's own converters (core/texture.py)
        encoded = not (settings.color_management != 'NONE'
                       and not settings.input_gamma_naive)
        if not encoded:
            tex.to_linear()
        if settings.tex_max_size:
            tex.clamp_size(settings.tex_max_size)
        if settings.tex_quantize:
            tex.quantize(settings.tex_quantize)
        # R251 prep-time laws, in this fixed order: a TMEM format OWNS
        # the texel format (C013 over C074), block compression composes
        # after either (C024: the Xbox stored DXT1 of a 565 source), the
        # chroma-key prep and the summed-area table are wave 2's. A
        # texel format with no alpha plane converts its cut-outs to the
        # key INSIDE the law (keyed=): the prep after it would find none
        if sig[8] != 'OFF':
            tex.fit_tmem(sig[8], bool(settings.tex_mipmap), encoded)
        elif sig[7] != 'NONE':
            tex.store_format(sig[7], encoded, keyed=bool(sig[10]))
        if sig[9] != 'NONE':
            tex.block_compress(sig[9], encoded)
        if sig[10] and hasattr(tex, 'colorkey_prepare'):
            tex.colorkey_prepare()
        if settings.tex_filter == 'SUMMED_AREA' and hasattr(tex, 'build_sat'):
            tex.build_sat()
        if settings.tex_mipmap:
            tex.build_mips()
        # the CONTENT laws (everything but filter and wrap) as one tag:
        # the driver's upload keys carry it, so a texture prepared under
        # another law is another upload by construction (A12.13)
        tex.prep = sig[2:]
        if len(_TEX_CACHE) > 32:
            _TEX_CACHE.clear()
        _TEX_CACHE[ckey] = tex
        out[key] = tex
    # the BI texture engine's noise tables, as a prepared texture. The
    # GPU compiler's raw sampler (hal_bitex_tab) looks this key up at
    # plan time and the driver uploads it like any pass texture -- and
    # for want of this entry, EVERY material carrying a Blender Internal
    # texture node refused its GPU pass ("engine table texture
    # '__bitex_tables__' is not among the prepared textures") and fell
    # back to the CPU, silently, on every GPU-device frame since the
    # node existed. 32 KB, built once, cached forever.
    tab = _TEX_CACHE.get('__bitex_tables__')
    if tab is None:
        from .bitex_tables import table_pixels
        tab = Texture(table_pixels(), name='__bitex_tables__',
                      colorspace='Non-Color', wrap='EXTEND',
                      filt='NEAREST')
        _TEX_CACHE['__bitex_tables__'] = tab
    out['__bitex_tables__'] = tab
    # R251 C023 (MAT-B): the PowerVR2 (S,R) angle tables, the SR Bump
    # node's two raw data textures (hal_sr_tab: SIDX / SINS / COSS /
    # COS256 in one row; hal_sr_atan: the 256x256 azimuth table), built
    # once, cached forever, bound like the BI tables
    srt = _TEX_CACHE.get('__sr_tables__')
    if srt is None:
        from .srbump_tables import table_pixels as _sr_table_pixels
        srt = Texture(_sr_table_pixels(), name='__sr_tables__',
                      colorspace='Non-Color', wrap='EXTEND',
                      filt='NEAREST')
        _TEX_CACHE['__sr_tables__'] = srt
    out['__sr_tables__'] = srt
    sra = _TEX_CACHE.get('__sr_atan__')
    if sra is None:
        from .srbump_tables import atan_pixels as _sr_atan_pixels
        sra = Texture(_sr_atan_pixels(), name='__sr_atan__',
                      colorspace='Non-Color', wrap='EXTEND',
                      filt='NEAREST')
        _TEX_CACHE['__sr_atan__'] = sra
    out['__sr_atan__'] = sra
    return out


# ----------------------------------------------------------- closure resolve


def closure_to_surface(cl, ctx, settings, material=None):
    """Collapse a node-graph closure into one reflectance model + parameters.

    Cycles-style closures are additive lobes; a 1990s renderer has one shader
    with a diffuse and a specular term. This is the honest translation: sum the
    weighted lobes into those slots and pick the model the tree implies.
    """
    n = ctx.n
    surf = SH.Surface(n)
    model = None
    if material is not None:
        # R239: the SDF face road resolves its frame from the material
        # (light_surface reads this; -1 = no material at hand)
        surf.material_index = int(getattr(material, 'index', -1))
        surf.diffuse[:] = np.asarray(material.diffuse, np.float32)[None, :]
        surf.specular[:] = np.asarray(material.specular, np.float32)[None, :]
        surf.glossiness[:] = material.glossiness
        surf.specular_level[:] = material.specular_level
        surf.diffuse_level[:] = material.diffuse_level
        surf.ambient[:] = material.ambient_level
        surf.opacity[:] = material.opacity
        _thr_pt = None
        if getattr(material, 'has_alpha', False) or \
                str(getattr(material, 'alpha_mode', 'BLEND')) in ('CLIP', 'CLIP_BLEND'):
            from .scene import clip_road as _clip_road
            _thr_pt = _clip_road(material)[0]
        if _thr_pt is not None or \
                str(getattr(material, 'alpha_mode', 'BLEND')) in ('CLIP', 'CLIP_BLEND'):
            # R211/R213 punch-through: the law below forces hard 0/1
            # alpha -- an identity on the provably-binary chains the
            # auto road promotes, the requested semantics under CLIP
            surf.alpha_clip[:] = float(
                _thr_pt if _thr_pt is not None
                else getattr(material, 'alpha_clip', 0.5))
        # R251 C031 (PS2 AFAIL): the two-pass mode reaches the alpha law
        # PER FRAGMENT (the vertex/face roads batch across materials)
        surf.alpha_soft[:] = 1.0 if str(getattr(material, 'alpha_mode', 'BLEND')) == 'CLIP_BLEND' else 0.0
        surf.ior[:] = material.ior
        surf.ray_ior[:] = material.ior   # one slider, both meanings here
        surf.roughness[:] = material.roughness
        surf.metallic[:] = material.metallic
        surf.anisotropy[:] = material.anisotropy
        surf.aniso_rot[:] = material.aniso_rotation
        surf.reflect[:] = material.reflect_level
        surf.emission[:] = np.asarray(material.emission, np.float32)[None, :] * \
            material.emission_level
        model = material.model

    if not isinstance(cl, Closure) or not cl.items:
        chosen = model or settings.default_model
        if settings.force_model != 'NONE':
            chosen = settings.force_model
        surf.model = chosen
        return surf, chosen, None

    diff = np.zeros((n, 3), np.float32)
    diff_w = np.zeros(n, np.float32)
    spec = np.zeros((n, 3), np.float32)
    spec_w = np.zeros(n, np.float32)
    emis = np.zeros((n, 3), np.float32)
    transp = np.zeros(n, np.float32)
    refr = np.zeros(n, np.float32)
    refr_col = np.zeros((n, 3), np.float32)
    rough_acc = np.zeros(n, np.float32)
    rough_w = np.zeros(n, np.float32)
    normal = None
    halcyon = []          # (weight, params) -- EVERY master lobe, weighted.
    # It was one slot, last-wins, weight DISCARDED: mixing two Halcyon
    # Shaders through a Mix Shader showed only the second whatever the Fac
    # said, and mixing one with any BSDF ignored the BSDF entirely -- "the
    # Mix Shader node doesn't work", said the field, about the one node
    # every converted material in this engine flows through.
    gloss_model = None
    diff_model = None

    for kind, w, p in cl.items:
        w = np.clip(np.asarray(w, np.float32).reshape(-1), 0.0, None)
        if w.shape[0] != n:
            w = np.broadcast_to(w, (n,)).copy()
        col = p.get('color')
        rgb = to_color(col, n)[:, :3] if col is not None else np.ones((n, 3), np.float32)
        if p.get('normal') is not None and kind != 'HALCYON':
            normal = p['normal']
        if kind == 'HALCYON':
            halcyon.append((w, p))
            continue
        if kind == 'DIFFUSE':
            diff += rgb * w[:, None]
            diff_w += w
            diff_model = p.get('model', diff_model)
            r = p.get('roughness')
            if r is not None:
                rough_acc += to_value(r, n) * w
                rough_w += w
        elif kind == 'GLOSSY':
            spec += rgb * w[:, None]
            spec_w += w
            gloss_model = p.get('model', gloss_model)
            r = p.get('roughness')
            if r is not None:
                rough_acc += to_value(r, n) * w
                rough_w += w
            if p.get('metallic') is not None:
                surf.metallic = to_value(p['metallic'], n)
            if p.get('anisotropy') is not None:
                surf.anisotropy = to_value(p['anisotropy'], n)
            if p.get('rotation') is not None:
                surf.aniso_rot = to_value(p['rotation'], n)
        elif kind == 'EMISSION':
            st = p.get('strength')
            emis += rgb * w[:, None] * (to_value(st, n)[:, None] if st is not None else 1.0)
        elif kind == 'TRANSPARENT':
            transp += w
        elif kind in ('GLASS', 'REFRACTION'):
            refr += w
            refr_col += rgb * w[:, None]
            spec += rgb * w[:, None] * 0.5
            spec_w += w * 0.5
            if p.get('ior') is not None:
                surf.ior = to_value(p['ior'], n)
            gloss_model = gloss_model or 'COOK_TORRANCE'
        elif kind == 'TRANSLUCENT':
            surf.translucency = np.maximum(surf.translucency, w)
            diff += rgb * w[:, None] * 0.5
            diff_w += w * 0.5
        elif kind == 'HOLDOUT':
            transp += w

    if halcyon:
        # Every master lobe contributes BY ITS WEIGHT: the parameters blend
        # in material space -- which is exactly how the fixed-function era
        # mixed looks, attribute by attribute -- so Mix Shader between two
        # Halcyon Shaders lerps them by Fac (per pixel when Fac is driven),
        # and a chain of mixes converges instead of last-wins. A single
        # full-weight lobe reduces to multiplying by 1.0: every existing
        # master material shades bit-identically.
        plain_transl = surf.translucency
        wsum = np.zeros(n, np.float32)
        for hwl, _p in halcyon:
            wsum += hwl
        hw = np.clip(wsum, 0.0, 1.0)
        norm_w = np.maximum(wsum, np.float32(1e-6))

        def hmix(key, default, colour=False):
            acc = np.zeros((n, 3), np.float32) if colour \
                else np.zeros(n, np.float32)
            for hwl, hp in halcyon:
                v = hp.get(key)
                v = default if v is None else v
                s = to_color(v, n)[:, :3] if colour else to_value(v, n)
                f = hwl / norm_w
                acc = acc + (s * f[:, None] if colour else s * f)
            return acc

        surf.diffuse = hmix('color', (0.8, 0.8, 0.8), True)
        # R242: ADDITIVE master lobes (Add Shader, the Max Shellac and
        # Composite roads) SUM their colours -- the parameters still
        # blend by share, but a weight total past 1 scales the colour
        # by that total, exactly the colour-chain sum the GPU emits for
        # an Add. A mix (weights summing to 1) multiplies by nothing,
        # so every existing frame holds bitwise.
        over = wsum > np.float32(1.000001)
        if np.any(over):
            surf.diffuse = surf.diffuse * np.where(over, wsum, 1.0)[:, None]
        surf.diffuse_level = hmix('diffuse_level', 1.0)
        surf.specular = hmix('spec_color', (1.0, 1.0, 1.0), True)
        surf.specular_level = hmix('spec_level', 0.5)
        surf.glossiness = np.maximum(hmix('glossiness', 25.0), 0.5)
        surf.roughness = np.clip(hmix('roughness', 0.3), 0.0, 1.0)
        surf.ambient = hmix('ambient', 1.0)
        surf.emission = hmix('emission', (0.0, 0.0, 0.0), True)
        surf.opacity = np.clip(hmix('opacity', 1.0), 0.0, 1.0)
        surf.ior = np.maximum(hmix('ior', 1.45), 1.0)
        surf.anisotropy = hmix('anisotropy', 0.0)
        surf.aniso_rot = hmix('rotation', 0.0)
        surf.metallic = hmix('metallic', 0.0)
        surf.soften = hmix('soften', 0.0)
        surf.reflect = hmix('reflect', 0.0)
        surf.translucency = hmix('translucency', 0.0)
        surf.toon_size = hmix('toon_size', 0.5)
        surf.toon_smooth = hmix('toon_smooth', 0.05)
        if any(hp.get('toon_steps') is not None for _w, hp in halcyon):
            surf.toon_steps = hmix('toon_steps', 2.0)
        for key, attr, knd, dflt in (
                ('fresnel', 'fresnel', 'v', 0.0),
                ('fresnel_power', 'fresnel_power', 'v', 3.0),
                ('fresnel_color', 'fresnel_color', 'c', (1, 1, 1)),
                ('fresnel_blend', 'fresnel_blend', 'v', 0.0),
                ('rim', 'rim', 'v', 0.0),
                ('rim_power', 'rim_power', 'v', 3.0),
                ('rim_color', 'rim_color', 'c', (1, 1, 1)),
                ('rim_blend', 'rim_blend', 'v', 0.0),
                ('matcap', 'matcap', 'c', (0, 0, 0)),
                ('matcap_blend', 'matcap_blend', 'v', 0.0),
                ('matcap_mode', 'matcap_mode', 'v', 0.0),
                ('reflect_color', 'reflect_color', 'c', (1, 1, 1)),
                ('edge_opacity', 'edge_opacity', 'v', 1.0),
                ('backface_color', 'backface_color', 'c', (0, 0, 0)),
                ('backface_mix', 'backface_mix', 'v', 0.0),
                ('sheen', 'sheen', 'v', 0.0),
                ('sheen_color', 'sheen_color', 'c', (1, 1, 1)),
                ('sheen_roughness', 'sheen_roughness', 'v', 0.3),
                # R251 lighting: the period finish dials (F006, F019-F021)
                ('fog_burn', 'fog_burn', 'v', 0.0),
                ('fog_bias', 'fog_bias', 'v', 0.0),
                ('fog_bank', 'fog_bank', 'v', 0.0),
                ('brilliance', 'brilliance', 'v', 1.0),
                ('crand', 'crand', 'v', 0.0),
                ('pov_metallic', 'pov_metallic', 'v', 0.0),
                ('refraction', 'refraction', 'v', 1.0),
                ('toon_size2', 'toon_size2', 'v', 0.5),
                ('toon_smooth2', 'toon_smooth2', 'v', 0.1),
                # R243: the Max Multi-Layer's second highlight and the
                # Translucent shader's colour
                ('spec_color2', 'specular2', 'c', (0.9, 0.9, 0.9)),
                ('spec_level2', 'specular_level2', 'v', 0.0),
                ('glossiness2', 'glossiness2', 'v', 25.0),
                ('anisotropy2', 'anisotropy2', 'v', 0.0),
                ('rotation2', 'aniso_rot2', 'v', 0.0),
                ('translucent_color', 'translucent_color', 'c', (0, 0, 0)),
                ('bi_fresnel', 'bi_fresnel', 'v', 0.1),
                ('bi_fresnel_fac', 'bi_fresnel_fac', 'v', 0.5),
                ('bi_slope', 'bi_slope', 'v', 0.1),
                # the BI panel round
                ('bi_transp_fresnel', 'bi_transp_fresnel', 'v', 0.0),
                ('bi_transp_blend', 'bi_transp_blend', 'v', 1.25),
                ('bi_spectra', 'bi_spectra', 'v', 0.0),
                ('bi_mir_fresnel', 'bi_mir_fresnel', 'v', 0.0),
                ('bi_mir_blend', 'bi_mir_blend', 'v', 1.25),
                ('ray_ior', 'ray_ior', 'v', 1.45),
                ('bi_ray_filter', 'bi_ray_filter', 'v', 1.0),
                ('bi_cubic', 'bi_cubic', 'v', 0.0),
                ('bi_tangent', 'bi_tangent', 'v', 0.0),
                ('anime_shadow1', 'anime_shadow1', 'c',
                 (0.62, 0.44, 0.48)),
                ('anime_shadow2', 'anime_shadow2', 'c',
                 (0.38, 0.26, 0.38)),
                ('anime_th1', 'anime_th1', 'v', 0.5),
                ('anime_soft1', 'anime_soft1', 'v', 0.04),
                ('anime_th2', 'anime_th2', 'v', 0.22),
                ('anime_soft2', 'anime_soft2', 'v', 0.04),
                ('anime_bias', 'anime_bias', 'v', 0.0),
                ('anime_tones', 'anime_tones', 'v', 2.0),
                ('anime_spec_size', 'anime_spec_size', 'v', 0.12),
                ('anime_sharp', 'anime_sharp', 'v', 0.05),
                ('anime_mask', 'anime_mask', 'v', 1.0),
                ('anime_gain', 'anime_gain', 'v', 1.0),
                ('anime_ramp_row', 'anime_ramp_row', 'v', 0.0),
                # R229: the 80s additions
                ('anime_shine', 'anime_shine', 'v', 0.0),
                ('anime_shine_color', 'anime_shine_color', 'c', (1, 1, 1)),
                ('anime_shine_h', 'anime_shine_h', 'v', 0.78),
                ('anime_shine_w', 'anime_shine_w', 'v', 0.06),
                ('anime_shine_wave', 'anime_shine_wave', 'v', 0.03),
                ('anime_shine_waves', 'anime_shine_waves', 'v', 6.0),
                ('anime_shine_soft', 'anime_shine_soft', 'v', 0.01),
                ('anime_shine_second', 'anime_shine_second', 'v', 0.0),
                # R241: the hair pass
                ('anime_shine_shape', 'anime_shine_shape', 'v', 0.0),
                ('anime_shine_angle', 'anime_shine_angle', 'v', 0.0),
                ('anime_shine_follow', 'anime_shine_follow', 'v', 0.0),
                ('anime_shine_color2', 'anime_shine_color2', 'c',
                 (1, 1, 1)),
                ('anime_air', 'anime_air', 'v', 0.0),
                ('anime_air_color', 'anime_air_color', 'c',
                 (0.82, 0.62, 0.62)),
                ('anime_air_width', 'anime_air_width', 'v', 0.35),
                ('anime_air_side', 'anime_air_side', 'v', 0.0),
                # the cartoon/paint master (R228)
                ('cartoon_shadow', 'cartoon_shadow', 'c',
                 (0.55, 0.45, 0.62)),
                ('cartoon_amount', 'cartoon_amount', 'v', 1.0),
                ('cartoon_th', 'cartoon_th', 'v', 0.5),
                ('cartoon_soft', 'cartoon_soft', 'v', 0.02),
                ('cartoon_smooth', 'cartoon_smooth', 'v', 0.0),
                ('cartoon_hl_color', 'cartoon_hl_color', 'c', (1, 1, 1)),
                ('cartoon_hl_size', 'cartoon_hl_size', 'v', 0.0),
                ('cartoon_hl_soft', 'cartoon_hl_soft', 'v', 0.02),
                ('cartoon_mode', 'cartoon_mode', 'v', 0.0),
                ('cartoon_lamp', 'cartoon_lamp', 'v', 0.0),
                # R238: the cel's light
                ('cel_light', 'cel_light', 'v', 0.0),
                ('cel_dir', 'cel_dir', 'c', (0.0, 0.0, 1.0)),
                ('cel_ss', 'cel_ss', 'v', 0.0),
                ('cel_ss_len', 'cel_ss_len', 'v', 24.0),
                ('cel_rim_mode', 'cel_rim_mode', 'v', 0.0),
                ('cel_rim_width', 'cel_rim_width', 'v', 4.0),
                ('cel_rim_side', 'cel_rim_side', 'v', 0.0),
                ('cel_shape', 'cel_shape', 'v', 0.0),
                ('shadow_receive', 'shadow_receive', 'v', 1.0),
                ('cast_only', 'cast_only', 'v', 0.0),
                ('shadows_only', 'shadows_only', 'v', 0.0),
                ('use_mist', 'use_mist', 'v', 1.0),
                # R252: the Console Emulation Shader's light-loop fields
                ('fixed_shade', 'fixed_shade', 'v', 0.0),
                ('light_limit', 'light_limit', 'v', 0.0),
                ('axis_viewer', 'axis_viewer', 'v', 0.0),
                ('prelit', 'prelit', 'c', (1.0, 1.0, 1.0)),
                ('prelit_mode', 'prelit_mode', 'v', 0.0),
                ('gx_diff_fn', 'gx_diff_fn', 'v', 0.0),
                ('gx_attn_fn', 'gx_attn_fn', 'v', 0.0),
                ('sun_clamp', 'sun_clamp', 'v', 1.0),
                ('alpha_steps', 'alpha_steps', 'v', 0.0)):
            if any(hp.get(key) is not None for _w, hp in halcyon):
                setattr(surf, attr, hmix(key, dflt, knd == 'c'))
        # a master shader's refraction bends through its one IOR slider;
        # only the BI node carries a separate ray IOR. Copy rather than
        # default so a master glass at IOR 1.6 keeps bending at 1.6.
        if not any(hp.get('ray_ior') is not None for _w, hp in halcyon):
            surf.ray_ior = np.asarray(surf.ior, np.float32).copy()
        # the non-numeric extras (ramp specs, light group) ride one
        # python object; the HEAVIEST closure names it, like the model
        if any(hp.get('bi_extras') is not None for _w, hp in halcyon):
            surf.bi = max(halcyon,
                          key=lambda t: float(np.mean(t[0])))[1].get(
                              'bi_extras')
        # R221: the baked Shadow Ramp rides the same one-object idiom
        if any(hp.get('anime_extras') is not None for _w, hp in halcyon):
            surf.anime_ramp = max(
                halcyon, key=lambda t: float(np.mean(t[0])))[1].get(
                    'anime_extras')
        # the model cannot blend: the HEAVIEST lobe names it, deterministic
        heavy = max(halcyon, key=lambda t: float(np.mean(t[0])))
        model = heavy[1].get('model', model)
        carriers = [(hwl, hp) for hwl, hp in halcyon
                    if hp.get('normal') is not None]
        if carriers:
            _w_, p = max(carriers, key=lambda t: float(np.mean(t[0])))
            normal = p['normal']
            # Bump Strength scales how far the supplied normal is allowed to
            # bend away from the surface it sits on. Done here rather than in
            # the node graph because the geometric normal is only known once
            # the closure has been collapsed against a fragment.
            bs = p.get('bump_strength')
            if bs is not None:
                k = to_value(bs, n)[:, None]
                if np.any(np.abs(k - 1.0) > 1e-4):
                    geo = M.normalize(np.asarray(ctx.N, np.float32))
                    normal = geo + (M.normalize(np.asarray(normal, np.float32))
                                    - geo) * k
        # R242: Max's Faceted flag on the heaviest lobe -- the stored
        # face normal replaces the shading normal, bump and all (the
        # GPU substitutes hal_triaux after its bend, the same order)
        if heavy[1].get('faceted'):
            normal = np.asarray(ctx.Ng, np.float32)
        # a mix with PLAIN lobes (a master blended against a raw BSDF):
        # albedos blend by relative weight, levels sum toward 1, the era
        # terms that only a master carries fade with its share, and the
        # transparent side eats into opacity exactly as Fac says
        others = np.any(diff_w > 1e-6) or np.any(spec_w > 1e-6) or \
            np.any(transp > 1e-6) or np.any(emis != 0.0) or \
            np.any(refr > 1e-6) or np.any(plain_transl > 1e-6)
        if others:
            if np.any(diff_w > 1e-6):
                t = diff_w / np.maximum(hw + diff_w, 1e-6)
                pd = diff / np.maximum(diff_w, 1e-6)[:, None]
                surf.diffuse = surf.diffuse * (1.0 - t)[:, None] \
                    + pd * t[:, None]
                surf.diffuse_level = np.clip(
                    surf.diffuse_level * hw + diff_w, 0.0, 1.0)
            else:
                surf.diffuse_level = surf.diffuse_level * hw
            if np.any(spec_w > 1e-6):
                t = spec_w / np.maximum(hw + spec_w, 1e-6)
                ps = spec / np.maximum(spec_w, 1e-6)[:, None]
                surf.specular = surf.specular * (1.0 - t)[:, None] \
                    + ps * t[:, None]
                surf.specular_level = np.clip(
                    surf.specular_level * hw + spec_w, 0.0, 1.0)
            else:
                surf.specular_level = surf.specular_level * hw
            if np.any(rough_w > 1e-6):
                t = rough_w / np.maximum(hw + rough_w, 1e-6)
                rp = np.clip(rough_acc / np.maximum(rough_w, 1e-6), 0.0, 1.0)
                gp = np.minimum(np.maximum(
                    2.0 / np.maximum(rp ** 4, 1e-5) - 2.0, 0.5), 8192.0)
                surf.roughness = np.clip(
                    surf.roughness * (1.0 - t) + rp * t, 0.0, 1.0)
                surf.glossiness = surf.glossiness * (1.0 - t) + gp * t
            surf.emission = surf.emission * hw[:, None] + emis
            surf.opacity = np.clip(
                1.0 - transp - (1.0 - surf.opacity) * hw, 0.0, 1.0)
            surf.reflect = np.clip(surf.reflect * hw + refr, 0.0, 1.0)
            surf.translucency = np.maximum(surf.translucency * hw,
                                           plain_transl)
            surf.ambient = 1.0 + (surf.ambient - 1.0) * hw
            surf.edge_opacity = 1.0 + (surf.edge_opacity - 1.0) * hw
            for a_ in ('fresnel', 'rim', 'matcap_blend', 'sheen',
                       'backface_mix'):
                setattr(surf, a_, getattr(surf, a_) * hw)
            # a plain gloss side heavier than every master lobe also
            # outvotes the model
            if np.any(spec_w > 1e-6) and \
                    float(np.mean(spec_w)) > float(np.mean(hw)) and \
                    gloss_model is not None:
                model = gloss_model
    else:
        w = np.maximum(diff_w, 1e-6)
        if np.any(diff_w > 1e-6):
            surf.diffuse = diff / w[:, None]
            surf.diffuse_level = np.clip(diff_w, 0.0, 1.0)
        if np.any(spec_w > 1e-6):
            surf.specular = spec / np.maximum(spec_w, 1e-6)[:, None]
            surf.specular_level = np.clip(spec_w, 0.0, 1.0)
        else:
            surf.specular_level = np.zeros(n, np.float32)
        if np.any(rough_w > 1e-6):
            r = np.clip(rough_acc / np.maximum(rough_w, 1e-6), 0.0, 1.0)
            surf.roughness = r
            # Blender roughness -> a Phong/Blinn exponent, the classic mapping
            surf.glossiness = np.maximum(2.0 / np.maximum(r * r * r * r, 1e-5) - 2.0,
                                         0.5).astype(np.float32)
            surf.glossiness = np.minimum(surf.glossiness, 8192.0)
        surf.emission = emis
        surf.opacity = np.clip(1.0 - transp, 0.0, 1.0)
        surf.reflect = np.clip(refr, 0.0, 1.0)
        if model is None:
            model = gloss_model or diff_model

    if settings.force_model != 'NONE':
        model = settings.force_model
    if model is None:
        model = settings.default_model
    surf.model = model
    return surf, model, normal


# ----------------------------------------------------------------- lighting


def _anime_smooth(e0, e1, x):
    """smoothstep with running edges, written out so the GLSL twin is
    the same three operations."""
    t = np.clip((x - e0) / np.maximum(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _anime_ramp_sample(lut, u, v):
    """R221: the baked ramp LUT, endpoint-mapped bilinear on both axes
    -- u=0 lands exactly on the first texel, u=1 on the last, so the
    ramp's painted ends ARE the shadow floor and the lit ceiling. The
    GPU writes out the same arithmetic on the same uploaded texels."""
    h, w = lut.shape[:2]
    fx = np.clip(u, 0.0, 1.0) * np.float32(w - 1)
    fy = np.clip(v, 0.0, 1.0) * np.float32(h - 1)
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    tx = (fx - x0).astype(np.float32)[:, None]
    ty = (fy - y0).astype(np.float32)[:, None]
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    c00 = lut[y0, x0]
    c10 = lut[y0, x1]
    c01 = lut[y1, x0]
    c11 = lut[y1, x1]
    top = c00 + (c10 - c00) * tx
    bot = c01 + (c11 - c01) * tx
    return top + (bot - top) * ty


def _anime_lamp(surf, wrap, vis, rad, N, L, V, affect_diffuse,
                affect_specular):
    """One lamp of the cel model (R218): the shadow term lives INSIDE
    the band input, so a cast shadow pushes a pixel into its tone band
    instead of darkening it -- a shadow is a COLOUR here. The tint
    multiplies the base per lamp; the stepped highlight gates on the
    wrapped half-vector. Both come back pre-divided by pi so lamp
    energies mean what they mean on every other model."""
    inv_pi = np.float32(1.0 / np.pi)
    rad = rad * surf.anime_gain[:, None]
    x = np.clip(wrap + surf.anime_bias, 0.0, 1.0) * vis
    _extras = getattr(surf, 'anime_ramp', None)
    face = _extras.get('face') if isinstance(_extras, dict) else None
    if isinstance(face, dict) and face.get('on') \
            and face.get('frame') is not None:
        # R239: the SDF face shadow -- the anime face's terminator is
        # DRAWN in a map, not found on the normals. The map's field
        # against the light's horizontal angle about the face's own
        # frame replaces the lambert wrap (bias included: the map IS
        # the authored bias); the side mirrors across the face's
        # centre line; the cast and screen shadows still multiply.
        fwd, up, rgt = face['frame']
        L3 = np.asarray(L, np.float32)
        if L3.ndim == 1:
            L3 = L3[None, :]
        Lh = L3 - up[None, :] * (L3 @ up)[:, None]
        ln = np.sqrt((Lh * Lh).sum(1))
        Lh = Lh / np.maximum(ln, np.float32(1e-9))[:, None]
        ct = np.clip(Lh @ fwd, -1.0, 1.0)
        t_f = np.arccos(ct).astype(np.float32) * np.float32(1.0 / np.pi)
        side_f = Lh @ rgt
        val = np.where(side_f >= 0.0, face['val'][:, 0],
                       face['val'][:, 1]).astype(np.float32)
        x = (np.clip(np.float32(0.5) + (val - t_f), 0.0, 1.0)
             * vis).astype(np.float32)
    ramp = getattr(surf, 'anime_ramp', None)
    if isinstance(ramp, dict) and ramp.get('ramp') is not None:
        # R221: the ramp shading road -- the tint IS the baked ramp,
        # sampled by the light term: shadow side, transition and lit
        # side all painted in the texture, exactly the games' own
        # convention. The tone sliders stand down while a ramp is
        # connected (the node's tooltip says so).
        tint = _anime_ramp_sample(ramp['ramp']['lut'], x,
                                  surf.anime_ramp_row)
    else:
        b1 = _anime_smooth(surf.anime_th1 - surf.anime_soft1,
                           surf.anime_th1 + surf.anime_soft1, x)
        b2 = _anime_smooth(surf.anime_th2 - surf.anime_soft2,
                           surf.anime_th2 + surf.anime_soft2, x)
        t3 = np.clip(surf.anime_tones - 2.0, 0.0, 1.0)
        shade3 = surf.anime_shadow2 + \
            (surf.anime_shadow1 - surf.anime_shadow2) * b2[:, None]
        shade = surf.anime_shadow1 + \
            (shade3 - surf.anime_shadow1) * t3[:, None]
        tint = shade + (1.0 - shade) * b1[:, None]
    if np.any(surf.anime_air > 1e-6):
        # R229: the airbrush gradation -- the cel painter's soft tone
        # against the hard band edge. LIT side: the tint multiplies
        # toward the airbrush colour just above the first threshold,
        # fading out toward full light; SHADOW side: the shadow tone
        # blends toward the airbrush colour approaching the edge from
        # below. Same three operations on the GPU.
        tint = _anime_airbrush(surf, tint, x)
    dcontrib = np.zeros((surf.n, 3), np.float32)
    scontrib = np.zeros((surf.n, 3), np.float32)
    if affect_diffuse:
        dcontrib = (surf.diffuse * tint
                    * surf.diffuse_level[:, None] * rad) * inv_pi
    if affect_specular:
        h = M.normalize(L + V)
        sndh = np.clip(M.dot(N, h) * np.float32(0.5) + np.float32(0.5),
                       0.0, 1.0)
        edge = 1.0 - np.clip(surf.anime_spec_size, 0.0, 1.0)
        gate = _anime_smooth(edge - surf.anime_sharp,
                             edge + surf.anime_sharp, sndh)
        sp = gate * surf.anime_mask * vis
        scontrib = (sp[:, None] * surf.specular
                    * surf.specular_level[:, None] * rad) * inv_pi
    return dcontrib, scontrib



def cartoon_smooth_normal(N, ctx, amount, shape=None, V=None):
    """R228: the inker's simplification -- the shading normal mixed
    toward the normal of a SPHERE around the object's bounding-box
    centre, so a face's shadow terminator sweeps as one clean shape
    instead of following every bump. Reads the same per-object bounds
    Generated coordinates use (both devices carry them); a context
    without object ids (a traced hit, a vertex corner) keeps its
    normal. R238 adds the shape: 1 an upright CYLINDER through the
    bounds (the radial normal with no z: the terminator runs straight
    down a limb), 2 the CAMERA (the normal bent toward the viewer: the
    surface reads as one plane facing the camera -- the anime face)."""
    if not np.any(amount > 1e-6):
        return N
    k = np.clip(amount, 0.0, 1.0)[:, None].astype(np.float32)
    code = np.rint(np.asarray(shape, np.float32)).astype(np.int32) \
        if shape is not None else np.zeros(1, np.int32)
    if np.all(code == 2) and V is not None:
        return M.normalize(N + (np.asarray(V, np.float32) - N) * k)
    bounds = getattr(ctx, 'obj_bounds', None)
    oi = getattr(ctx, 'object_index_raw', None)
    if bounds is None or oi is None or ctx.P is None:
        if V is not None and np.any(code == 2):
            tgt = np.where((code == 2)[:, None] if code.size > 1
                           else np.full((N.shape[0], 1), bool(code[0] == 2)),
                           np.asarray(V, np.float32), N)
            return M.normalize(N + (tgt - N) * k)
        return N
    lo, span = bounds
    idx = np.clip(np.asarray(oi, np.int64), 0, lo.shape[0] - 1)
    centre = lo[idx] + span[idx] * np.float32(0.5)
    rel = np.asarray(ctx.P, np.float32) - centre
    sph = M.normalize(rel)
    if np.any(code == 1):
        flat = rel.copy()
        flat[:, 2] = 0.0
        cyl = M.normalize(flat)
        sel = (code == 1)[:, None] if code.size > 1 else \
            np.full((N.shape[0], 1), bool(code[0] == 1))
        sph = np.where(sel, cyl, sph)
    if np.any(code == 2) and V is not None:
        sel = (code == 2)[:, None] if code.size > 1 else \
            np.full((N.shape[0], 1), bool(code[0] == 2))
        sph = np.where(sel, np.asarray(V, np.float32), sph)
    return M.normalize(N + (sph - N) * k)


def _cartoon_compose(surf, lit, hl, rad_acc):
    """R228: the paint composition after the lamp loop. `lit` is the
    strongest lamp's lit term (max over lamps of wrap * vis), `hl` the
    strongest highlight gate, `rad_acc` the summed lamp radiance times
    the lit term over pi (the Lamp Influence energy).

    Written out so the GLSL twin is the same operations: the shadow
    tone by mode (0 transparent cel: paint x mix(1, shadow, amount);
    1 painted: mix(paint, shadow, amount); 2 none), the LIT paint
    modulated by Lamp Influence (the shadow tone is a colour and stays
    one -- the lamps' energy only ever reaches the lit side), the lit
    step between them, the highlight painted OVER the result, then the
    diffuse level."""
    paint = surf.diffuse
    t = _anime_smooth(surf.cartoon_th - surf.cartoon_soft,
                      surf.cartoon_th + surf.cartoon_soft, lit)[:, None]
    amt = np.clip(surf.cartoon_amount, 0.0, 1.0)[:, None]
    mode = np.rint(surf.cartoon_mode).astype(np.int32)
    transparent = paint * (1.0 + (surf.cartoon_shadow - 1.0) * amt)
    painted = paint + (surf.cartoon_shadow - paint) * amt
    shade = np.where((mode == 1)[:, None], painted, transparent)
    shade = np.where((mode >= 2)[:, None], paint, shade)
    li = np.clip(surf.cartoon_lamp, 0.0, 1.0)[:, None]
    lit_col = paint * (1.0 - li) + paint * rad_acc * li
    base = shade + (lit_col - shade) * t
    if np.any(surf.anime_air > 1e-6):
        # R238: the airbrush gradation against the paint's shadow edge,
        # the Anime Shader's own operations on the cartoon's threshold
        base = _anime_airbrush(surf, base, lit, th=surf.cartoon_th)
    hk = np.clip(hl, 0.0, 1.0)[:, None]
    base = base + (surf.cartoon_hl_color - base) * hk
    return (base * surf.diffuse_level[:, None]).astype(np.float32)


def _anime_airbrush(surf, tint, x, th=None):
    """R229: the airbrush gradation on one lamp's tint. `x` is the
    band input (wrap + bias, times vis). Side code 0 LIT, 1 SHADOW,
    2 BOTH. Written out so the GLSL twin is the same operations.
    R238: `th` names the edge (the cartoon's threshold when the paint
    master airbrushes); the Anime Shader's first threshold otherwise."""
    amt = np.clip(surf.anime_air, 0.0, 1.0)
    width = np.maximum(surf.anime_air_width, 1e-4)
    th = surf.anime_th1 if th is None else th
    side = np.rint(surf.anime_air_side)
    ac = surf.anime_air_color
    # lit side: 1 at the edge, 0 a `width` above it, only above the edge
    g_lit = (1.0 - _anime_smooth(th, th + width, x)) \
        * _anime_smooth(th - np.float32(1e-4), th, x)
    # shadow side: 0 deep in shadow, 1 at the edge, only below the edge
    g_sh = _anime_smooth(th - width, th, x) \
        * (1.0 - _anime_smooth(th, th + np.float32(1e-4), x))
    lit_on = (side != 1).astype(np.float32)
    sh_on = (side != 0).astype(np.float32)
    k_lit = (amt * g_lit * lit_on)[:, None]
    k_sh = (amt * g_sh * sh_on)[:, None]
    out = tint * (1.0 + (ac - 1.0) * k_lit)
    out = out + (ac - out) * k_sh
    return out.astype(np.float32)


def _anime_hair_shine(out, surf, ctx, N, V, key_z=None):
    """R229: the hair shine band -- the 80s "angel ring": a band of
    the shine colour painted across the object at a fraction of its
    height, its edge waving around the object, on the upward- and
    camera-facing surface, light-independent (it was painted on the
    cel at a position, not lit). A thinner second band can sit below.
    Reads the same per-object bounds Generated coordinates use; a
    context without them (a vertex corner, a traced hit off the
    G-buffer) paints nothing. R241 (the hair pass): the wave has a
    SHAPE menu and a phase (Hair Shine Angle); Hair Shine Follow
    lifts the band by the key's world height (`key_z`, the caller's
    directional key: a scene sun or hemi, or the material's fixed
    key; None -- a positional key -- leaves the band painted where it
    is); the second band wears its own tint over the shine colour.
    R241 also runs it for the CARTOON master -- the highlight cel
    over the paint."""
    if not np.any(surf.anime_shine > 1e-6):
        return out
    gen = getattr(ctx, 'generated', None)
    bounds = getattr(ctx, 'obj_bounds', None)
    oi = getattr(ctx, 'object_index_raw', None)
    if gen is None or bounds is None or oi is None or ctx.P is None:
        return out
    lo, span = bounds
    idx = np.clip(np.asarray(oi, np.int64), 0, lo.shape[0] - 1)
    centre = lo[idx] + span[idx] * np.float32(0.5)
    P = np.asarray(ctx.P, np.float32)
    az = np.arctan2(P[:, 1] - centre[:, 1], P[:, 0] - centre[:, 0]) \
        .astype(np.float32)
    t = np.asarray(gen, np.float32)[:, 2]
    amt = np.clip(surf.anime_shine, 0.0, 1.0)
    w = np.maximum(surf.anime_shine_w, 1e-4) * np.float32(0.5)
    s = np.maximum(surf.anime_shine_soft, 1e-4)
    # R241: the hair pass. The wave's ARGUMENT gains a phase (Hair
    # Shine Angle turns the teeth around the head), its SHAPE is the
    # menu's (each a period-matched wave in -1..1, code 0 the 1.72
    # sine bitwise), and the band's HEIGHT rides the key's world
    # height by Hair Shine Follow (the ring is redrawn shifted when
    # the key moves, as a per-cut drawing would be). Every new dial
    # at zero reproduces the 1.83 arithmetic exactly.
    arg = (surf.anime_shine_waves * az
           + surf.anime_shine_angle * np.float32(np.pi / 180.0)) \
        .astype(np.float32)
    shape = int(round(float(np.asarray(surf.anime_shine_shape,
                                       np.float32).reshape(-1)[0])))
    if shape == 1:                                   # zigzag: triangle
        q = arg * np.float32(1.0 / (2.0 * np.pi)) - np.float32(0.25)
        wav = (np.float32(4.0) * np.abs(q - np.floor(q)
                                        - np.float32(0.5))
               - np.float32(1.0)).astype(np.float32)
    elif shape == 2:                                 # scallop: arcs
        wav = (np.float32(1.0)
               - np.float32(2.0) * np.abs(np.sin(arg * np.float32(0.5)))
               ).astype(np.float32)
    elif shape == 3:                                 # step: square
        wav = np.where(np.sin(arg) >= 0.0, np.float32(1.0),
                       np.float32(-1.0)).astype(np.float32)
    else:                                            # smooth: the sine
        wav = np.sin(arg).astype(np.float32)
    hh = surf.anime_shine_h
    if key_z is not None:
        hh = hh + surf.anime_shine_follow * np.float32(0.35 * key_z)
    h0 = hh + surf.anime_shine_wave * wav
    band1 = _anime_smooth(h0 - w - s, h0 - w + s, t) \
        * (1.0 - _anime_smooth(h0 + w - s, h0 + w + s, t))
    second = surf.anime_shine_second
    h2 = h0 - second
    w2 = w * np.float32(0.5)
    band2 = _anime_smooth(h2 - w2 - s, h2 - w2 + s, t) \
        * (1.0 - _anime_smooth(h2 + w2 - s, h2 + w2 + s, t)) \
        * np.float32(0.85)
    band2m = np.where(second > 1e-6, band2, 0.0)
    band = np.maximum(band1, band2m)
    facing = _anime_smooth(np.float32(0.0), np.float32(0.35), M.dot(N, V))
    # the ring circles the head: sides and top carry it in full, only
    # the underside fades it out
    upward = _anime_smooth(np.float32(-0.3), np.float32(0.1), N[:, 2])
    k = (amt * band * facing * upward)[:, None]
    # R241: where the SECOND band owns the pixel, its own tint
    # multiplies the shine colour (white = the 1.83 colour bitwise)
    col = np.where((band2m > band1)[:, None],
                   surf.anime_shine_color * surf.anime_shine_color2,
                   surf.anime_shine_color).astype(np.float32)
    # painted OVER: out*(1-k) + shine*k, so a full band is the shine
    # colour bitwise whatever lay beneath (a 10x lamp included)
    return (out * (1.0 - k) + col * k).astype(np.float32)


def light_surface(surf, model, ctx, scene, settings, bvh=None, rng=None,
                  active_lights=None, extras=None, suppress_spec=False,
                  sss=None):
    """Direct lighting + ambient + emission for a batch of points.

    `extras`, when a dict, receives the BI panel round's side channels:
    'spec_acc' (the accumulated specular colour, for spectra),
    'only_shadow' ((accum, count) of per-light shadow, for Shadows
    Only). Ramp specs and the light group ride `surf.bi`.

    `suppress_spec` masks the specular lobe out of the result -- the
    SSS pre-pass's combinedflag &= ~SCE_PASS_SPEC. `sss`, when a dict
    ({'sampled': (n,3), 'texfac': f}), REPLACES the accumulated
    diffuse with the scatter-tree sample shaped by the material colour
    -- the shade_lamp_loop block, verbatim."""
    # R251 (MAT-A C034): the DS toon pair IS the DS light unit
    model = SH.LOBE_ALIAS.get(model, model)
    n = ctx.n
    # R253: the light split -- shade_batch asks for it (extras
    # ['want_split'] = the wanted pass names, or True for all) at the
    # opaque frame's camera fragments. Every split array is a SEPARATE
    # accumulator fed the same arrays `out` receives; no arithmetic on
    # `out` moves, so the beauty stays bitwise with the split off or on
    _wsn = extras.get('want_split') if extras is not None else None
    want_split = bool(_wsn)
    sp_names = set(_wsn) if isinstance(_wsn, (tuple, list, set, frozenset)) \
        else None
    sp_ao = None
    N = M.normalize(ctx.N)
    V = -M.normalize(ctx.I)
    if settings.two_sided_lighting:
        flip = M.dot(N, V) < 0.0
        N = np.where(flip[:, None], -N, N)
    # R251 F011: the reflectance models' viewer -- the true eye vector,
    # or ONE camera axis for the whole frame (OpenGL 1.1's infinite
    # viewer, the Sega Model boards' R.z, the DS's line of sight): the
    # camera's +Z row of the CPU's own view matrix (inv(mw), R:44), the
    # world-space direction toward a viewer at infinity. Only
    # SH.evaluate reads Vs; the flip, the Hemi override, the cheats,
    # sheen and the cel loops keep the true V.
    if settings.two_sided_lighting:
        flip = M.dot(N, V) < 0.0
        N = np.where(flip[:, None], -N, N)
    # R251 (LIGHT-B1 F011 / LIGHT-B2 F016-F018): the view vector the
    # reflectance models take -- the true eye, or ONE camera axis for the
    # whole frame (OpenGL 1.1's infinite viewer, the Sega boards' R.z,
    # the DS's (0,0,-1) line of sight). The console light units force
    # the axis inside themselves; the flip above, the Hemi override, the
    # silhouette cheats and sheen keep the true eye.
    Vs = V
    if str(getattr(settings, 'specular_viewer', 'PIXEL')) == 'AXIS' \
            or model in getattr(SH, 'AXIS_MODELS', ()) \
            or np.any(surf.axis_viewer > 0.5):
        # (R252: a Console node's Local Viewer off asks for the axis per
        # material -- OpenGL's default, Direct3D's D3DRS_LOCALVIEWER FALSE)
        _vm = getattr(ctx, 'view_matrix', None)
        if _vm is not None:
            Vs = np.broadcast_to(
                M.normalize(np.asarray(_vm, np.float32)[2:3, :3])[0][None, :],
                V.shape).astype(np.float32)

    if model in ('CONSTANT', 'WIREFRAME'):
        if want_split:
            # R253: no light evaluated -- the colour is the diffuse, the
            # shadow and AO read open
            extras['split'] = _split_unlit(
                surf, surf.diffuse * surf.diffuse_level[:, None], n,
                sp_names)
        return surf.diffuse * surf.diffuse_level[:, None] + surf.emission
    if np.any(surf.fixed_shade > 0.5):
        # R252: fixed shading -- no light evaluated (Model 3's fixed
        # shading bit, the PS1's raw-texture bit, a GS / N64 decal, the
        # PSP's replace, RenderWare's unlit prelit geometry): the colour
        # shown as it is, times the prelight where the mode says so, plus
        # emission. At the corner road the LIGHT half is white, so the
        # combine sees unity (the texel exactly itself through every
        # machine's modulate) or the prelight alone
        base = surf.diffuse * surf.diffuse_level[:, None]
        pre = np.where((surf.prelit_mode > 0.5)[:, None], surf.prelit, 1.0)
        fixed = base * pre + surf.emission
        if np.all(surf.fixed_shade > 0.5):
            if want_split:
                # R253: fixed shading reports its shown colour (texel x
                # prelight) as Diffuse, so the identity holds
                extras['split'] = _split_unlit(surf, base * pre, n,
                                               sp_names, color=base)
            return fixed.astype(np.float32)
        fixed_mask = surf.fixed_shade > 0.5
    else:
        fixed_mask = None
    # R238: the cel's light -- the material's key (0 the scene's lamps,
    # 1 a key fixed to the camera, 2 a key fixed to the world), the
    # screen shadow and the depth rim read from the frame's cel field
    # at this batch's own pixels, the smoothing shape
    cel_mode = 0
    cel_ss_term = None
    cel_rim = None
    cel_key = None
    cel_L = None
    vis_acc = None
    if model in ('CARTOON', 'ANIME'):
        from . import celfield as CF
        from . import lines as LN
        # R228: the inker's shadow-shape smoothing, before any lamp
        # (R229: the Anime Shader shares the road; R238 the shape)
        N = cartoon_smooth_normal(N, ctx, surf.cartoon_smooth,
                                  shape=surf.cel_shape, V=V)
        # R239: the SDF face road's frame -- resolved once per batch
        # from the first object wearing the material (the games' own
        # arrangement: a face material is one head's)
        extras_r = getattr(surf, 'anime_ramp', None)
        face_r = extras_r.get('face') if isinstance(extras_r, dict) \
            else None
        if isinstance(face_r, dict) and face_r.get('on') \
                and face_r.get('frame') is None \
                and not face_r.get('resolved'):
            from . import nodeeval as _NE
            mi_r = int(getattr(surf, 'material_index', -1))
            if mi_r >= 0:
                face_r['frame'] = _NE.face_frame(
                    scene, mi_r, face_r['fwd_axis'], face_r['up_axis'])
            face_r['resolved'] = True
        cel_mode = int(round(float(np.asarray(surf.cel_light,
                                              np.float32).reshape(-1)[0])))
        ss_px, rim_px = CF.lookup(getattr(ctx, 'cel_field', None),
                                  getattr(ctx, 'spx', None),
                                  getattr(ctx, 'spy', None),
                                  getattr(ctx, 'depth', None))
        if ss_px is not None and np.any(surf.cel_ss > 1e-6):
            cel_ss_term = (1.0 - ss_px * np.clip(surf.cel_ss, 0.0, 1.0)
                           ).astype(np.float32)
        if rim_px is not None:
            cel_rim = rim_px
        if cel_mode > 0:
            d0 = np.asarray(surf.cel_dir, np.float32).reshape(-1, 3)[0]
            cel_L = np.broadcast_to(
                CF.world_key(cel_mode, d0, getattr(scene, 'camera', None)),
                (n, 3)).astype(np.float32).copy()
            vis_acc = np.full(n, -1.0, np.float32)
        else:
            cel_key = LN.key_light(scene)
    # R241: the hair pass's key height -- Hair Shine Follow lifts the
    # band by the key's world z. Directional keys only: the material's
    # fixed key, or a scene SUN / HEMI key lamp; a positional key
    # leaves the band painted where it is (the tooltip says so).
    hair_key_z = None
    if model in ('CARTOON', 'ANIME') \
            and np.any(surf.anime_shine > 1e-6) \
            and np.any(surf.anime_shine_follow > 1e-6):
        if cel_mode > 0 and cel_L is not None:
            hair_key_z = float(np.asarray(cel_L, np.float32)
                               .reshape(-1, 3)[0, 2])
        else:
            _kl = cel_key
            if _kl is not None and str(getattr(_kl, 'type', '')).upper() \
                    in ('SUN', 'HEMI'):
                _kd = np.asarray(getattr(_kl, 'direction', (0, 0, -1)),
                                 np.float32)
                _kn = float(np.linalg.norm(_kd))
                if _kn > 1e-9:
                    hair_key_z = float(-_kd[2] / _kn)
    if model == 'CARTOON':
        cart_lit = np.zeros(n, np.float32)
        cart_hl = np.zeros(n, np.float32)
        cart_rad = np.zeros((n, 3), np.float32)

    # Lambertian BRDF normalisation. Light energy arrives in Blender's watt-based
    # units, and Cycles divides reflected radiance by pi; without this every
    # surface renders pi times too bright and clips to white, which hides the
    # material colour entirely. Applied to both lobes so the diffuse/specular
    # balance -- and the period-correct unnormalised highlight shape -- is
    # untouched. This is a units conversion, not a change to the models.
    inv_pi = np.float32(1.0 / np.pi)

    spx = getattr(ctx, 'spx', None)
    spy = getattr(ctx, 'spy', None)
    have_id = spx is not None and spy is not None

    if getattr(settings, 'radiosity', False) and bvh is not None:
        # the Radiosity checkbox: gathered ambient replaces the flat
        # term outright, and plain AO with it -- the gather is
        # occlusion-aware by construction (blocked sky IS the darkening,
        # and what blocks it lends its colour instead). Frame pixels
        # read the interpolated FIELD when spacing > 1; ray hits and
        # vertex corners (no pixel identity, or no field) gather fully.
        field = getattr(ctx, 'radiosity_field', None)
        if field is not None and ctx.px is not None:
            # SCREEN pixels (frame and transparent layers) read the
            # cache at their own pixel; traced hits have no place in a
            # screen-space cache and gather fully, identity intact
            irr = radiosity_lookup(
                field, getattr(settings, 'radiosity_spacing', 1),
                ctx.px, ctx.py, LI.ambient_light(scene, settings))
        else:
            irr = radiosity_gather(ctx.P, N, bvh, scene, settings, rng,
                                   sample_xy=(spx, spy) if have_id
                                   else None)
        out = surf.diffuse * surf.ambient[:, None] * irr
    else:
        if isinstance(model, str) and model.startswith('BI_MATRIX_'):
            # verbatim shade_lamp_loop (R155): BI's ambient is a FLAT
            # add -- shr->combined += amb * WORLD ambient -- never
            # multiplied by the diffuse colour. That rule covers the
            # WORLD pool only: the engine's own Global Ambient has no
            # BI counterpart and stays diffuse-tinted, or the default
            # (0.05, 0.05, 0.06) lifts every imported black with a
            # blue-leaning floor (the R156 field report).
            eng_col, wrld_col = LI.ambient_light_split(scene, settings)
            out = (surf.diffuse * surf.ambient[:, None]
                   * eng_col[None, :]
                   + np.broadcast_to(wrld_col[None, :], (n, 3))
                   * surf.ambient[:, None])
        else:
            amb_col = LI.ambient_light(scene, settings)
            out = surf.diffuse * surf.ambient[:, None] * amb_col[None, :]
        if settings.ambient_occlusion and bvh is not None:
            # R253: the factor is kept for the AO pass (ONE call, the
            # same product `out` always took)
            _aof = ambient_occlusion(ctx.P, N, bvh, settings, rng,
                                     sample_xy=(spx, spy) if have_id
                                     else None)
            out *= _aof[:, None]
            sp_ao = _aof

    # the sheen lobe's falloff, computed once rather than per light
    sheen_exp = None
    if np.any(surf.sheen > 1e-4):
        r = np.clip(surf.sheen_roughness, 0.0, 1.0)
        sheen_exp = 1.0 + (1.0 - r) * 15.0
        edge_vn = np.clip(1.0 - np.abs(M.dot(N, V)), 0.0, 1.0)

    lights = active_lights if active_lights is not None else \
        LI.select_lights(scene.lights, settings)
    # R252: the machine's light limit, per material (the GTE's three
    # rows, the DS's four lights, the RSP's seven): the first N of the
    # scene's selection, in scene order -- the games' own register order
    _ll = int(np.max(np.asarray(surf.light_limit, np.float32))) \
        if surf.n else 0
    if _ll > 0 and len(lights) > _ll:
        lights = list(lights)[:_ll]
    if np.any(surf.prelit_mode > 0.5):
        # R252: RenderWare's rpGEOMETRYPRELIT -- the prelight vertex
        # colour ADDED to the computed light before the clamp (never
        # multiplied by the material: the material colour and the texel
        # come after, at the combine)
        out = out + np.where((surf.prelit_mode > 0.5)[:, None],
                             surf.prelit, 0.0).astype(np.float32)
    clamp = float(settings.light_clamp)
    # R253: everything before the first lamp IS the Ambient pass (the
    # ambient term, radiosity, AO, the prelight); the lamp sums and the
    # Shadow pass's own accum / ir (2.79 shade_only_shadow's shape: the
    # dark fraction averaged over the lamps that shadow, 1.0 where the
    # lamp would not light the point -- zero extra rays) live in `sp`
    _spl = None
    if want_split:
        _spl = {'amb': out.copy(), 'diff': np.zeros((n, 3), np.float32),
                'spec': np.zeros((n, 3), np.float32), 'light': {},
                'sh_acc': np.zeros(n, np.float32), 'sh_ir': 0.0,
                'lamps': sp_names is None
                or any(k.startswith('Light') for k in sp_names)}

    def _sp_add(li_, d_, s_):
        # one lamp's (diffuse, specular) as `out` receives them (after
        # vis, shadow colour, lit_mask; BEFORE the per-lamp clamp)
        _spl['diff'] += d_
        if s_ is not None:
            _spl['spec'] += s_
        if _spl['lamps'] and li_ is not None and li_ < LIGHT_PASS_SLOTS:
            slot = _spl['light'].get(li_)
            if slot is None:
                slot = _spl['light'][li_] = np.zeros((n, 3), np.float32)
            slot += d_
            if s_ is not None:
                slot += s_

    def _sp_shadows(light_):
        smode_ = light_.shadow if settings.shadow_default == 'PER_LIGHT' \
            else settings.shadow_default
        return bool(settings.shadows) and light_.shadow != 'NONE' \
            and smode_ != 'NONE' \
            and getattr(light_, 'type', '') != 'HEMI' \
            and not getattr(light_, 'screen_spot', False)

    # ---- the BI panel round's per-material machinery
    bi_ex = surf.bi
    ramp_dif = getattr(bi_ex, 'ramp_dif', None) if bi_ex else None
    ramp_spec = getattr(bi_ex, 'ramp_spec', None) if bi_ex else None
    group = getattr(bi_ex, 'light_group', None) if bi_ex else None
    excl_lights = getattr(scene, 'exclusive_lights', None)
    receive_off = np.any(surf.shadow_receive < 0.5)
    want_only = extras is not None and np.any(surf.shadows_only > 0.5)
    want_spec_acc = extras is not None and (
        np.any(surf.bi_spectra > 0.0)
        or (ramp_spec is not None and ramp_spec.get('input') == 'RESULT'))
    # RESULT ramps read the whole accumulated colour, so those materials
    # keep diffuse and specular apart and clamp once at the end (BI had
    # no per-light clamp at all)
    # R164: BI's world Exposure -- wrld_exposure_correct runs on the
    # accumulated DIFFUSE ("has no spec!") and on SPEC separately,
    # BEFORE ambient/emit join combined, and never in the SSS pre-pass
    # (the C gates on !R.sss_points; suppress_spec marks exactly that)
    _wrld = getattr(scene, 'world', None)
    w_exp = float(getattr(_wrld, 'exposure', 0.0) or 0.0) \
        if _wrld is not None else 0.0
    w_rng = float(getattr(_wrld, 'exposure_range', 1.0) or 1.0) \
        if _wrld is not None else 1.0
    exposure_on = (w_exp != 0.0 or w_rng != 1.0) and not suppress_spec

    # SSS REPLACES the accumulated diffuse after all lights, exactly
    # shade_lamp_loop -- so an SSS material (and an exposed world)
    # rides the same separated bookkeeping the RESULT ramps use
    track_result = (
        (ramp_dif is not None and ramp_dif.get('input') == 'RESULT')
        or (ramp_spec is not None and ramp_spec.get('input') == 'RESULT')
        or (sss is not None) or exposure_on
        # R251 (MAT-A): the period combiners' specular carry
        or (extras is not None and bool(extras.get('want_spec'))))
    diff_acc = np.zeros((n, 3), np.float32) if track_result else None
    spec_acc = np.zeros((n, 3), np.float32) \
        if (track_result or want_spec_acc) else None
    only_accum = np.zeros(n, np.float32) if want_only else None
    only_ir = 0.0
    ndv_g = M.dot(N, V) if (ramp_dif is not None
                            or ramp_spec is not None) else None

    def _band(spec_r, fac):
        from .bitex import colorband_eval
        return colorband_eval(spec_r['stops'], np.asarray(fac, np.float32),
                              int(spec_r.get('ipotype', 0)))

    obj_idx = getattr(ctx, 'object_index_raw', None)

    # R164: shade_one_light's phongcorr (the shadow-bias terminator
    # fix), three branches verbatim: HEMI/AREA pass untouched; RAYBIAS
    # on a smooth face against the object's Auto Smooth threshold;
    # else Material.sbias against shadowed lamps. The face R_SMOOTH
    # flag rides a geometric proxy here (interpolated normal differs
    # from the face normal), noted divergence for flat-but-bumped
    # faces under RAYBIAS.
    pc_sbias = float(getattr(bi_ex, 'sbias', 0.0) or 0.0) \
        if bi_ex else 0.0
    pc_raybias = bool(getattr(bi_ex, 'raybias', False)) if bi_ex else False
    pc_wanted = settings.shadows and (pc_sbias != 0.0 or pc_raybias)
    pc_smooth = None
    pc_thresh = None
    if pc_wanted and pc_raybias:
        ng = M.normalize(np.asarray(ctx.Ng, np.float32))
        pc_smooth = np.abs(M.dot(ng, N)) < np.float32(1.0 - 1e-6)
        if obj_idx is not None and getattr(scene, 'objects', None):
            sres = np.array([float(getattr(o, 'smoothresh', 0.0) or 0.0)
                             for o in scene.objects], np.float32)
            pc_thresh = sres[np.clip(np.asarray(obj_idx, np.int64), 0,
                                     len(sres) - 1)]
        else:
            pc_thresh = np.zeros(n, np.float32)

    def _pcurve(inp, th):
        # (inp - t)/(inp*(1 - t)) above the threshold, 0 below -- the
        # C's exact shape (guarded against the degenerate t >= 1)
        th = np.minimum(th, np.float32(1.0 - 1e-6))
        num = inp - th
        den = inp * (1.0 - th)
        return np.where(inp > th,
                        num / np.where(np.abs(den) > 1e-20, den, 1.0),
                        0.0).astype(np.float32)

    # R251 (MAT-A C057): the house key lamp (lines.key_light: the first
    # non-ambient lamp in scene order), compared by identity in the loop
    sh_key = None
    if extras is not None and extras.get('want_key_lit'):
        from . import lines as _LN
        sh_key = _LN.key_light(scene)
    for li, light in enumerate(lights):
        lname = getattr(light, 'name', None)
        if group is not None:
            # the material's Light Group: only its lamps light this
            if lname not in group:
                continue
        elif excl_lights and lname in excl_lights:
            # a lamp claimed by some material's EXCLUSIVE group lights
            # nothing else
            continue
        if want_split and cel_mode == 0 and _sp_shadows(light):
            # R253: this lamp counts in the Shadow pass's average (its
            # dark fraction joins below wherever `vis` is read)
            _spl['sh_ir'] += 1.0
        lit_mask = None
        excl = getattr(light, 'exclude_objects', None)
        if excl and obj_idx is not None:
            inside = np.isin(obj_idx, np.asarray(list(excl), np.int32))
            lit_mask = inside if light.exclude_mode == 'ONLY' else ~inside
            if not lit_mask.any():
                continue
        if cel_mode > 0:
            # R238: under a fixed key the scene's lamps only CAST their
            # shadows: the strongest visibility among the casters that
            # reach this object is the key's own (no caster: lit).
            # R251 F015: a screen spot casts no shadow -- skipped
            if not LI.casts_shadow(light, settings) or \
                    getattr(light, 'screen_spot', False):
                continue
            L_c, _rad_c, dist_c = LI.sample(light, ctx.P, settings)
            vis_c = LI.visibility(light, ctx.P, N, L_c, dist_c, settings,
                                  bvh, rng,
                                  sample_xy=(spx, spy, li) if have_id
                                  else None,
                                  mask=lit_mask)
            if lit_mask is not None:
                vis_c = np.where(lit_mask, vis_c, -1.0).astype(np.float32)
            vis_acc = np.maximum(vis_acc, vis_c)
            continue
        if getattr(light, 'screen_spot', False) and \
                getattr(light, 'type', '') == 'SPOT':
            # R251 F015: Sega Model 3's viewport spotlight -- an ellipse
            # pinned to the SCREEN at the lamp's projected position, a
            # depth window from the near clip over |Falloff End|, its
            # lobe added to the diffuse at the lamp's Energy/4pi (the
            # same radiance scale every Halcyon lamp uses) -- no cone,
            # no N.L, no shadow. The lamp lives on the screen: a point
            # with no pixel (ray hits, transparent layers, vertex
            # corners) gets nothing (lighting.md section 0, A10). ONE
            # association on both roads (A50): ((diffuse*level)*colE)
            # *lobe*inv_pi. The fog lobe colE*(en*el) accumulates in
            # ctx.spot_fog for core/fog.py (LIGHT-A1's fog_spot).
            px_ = getattr(ctx, 'px', None)
            py_ = getattr(ctx, 'py', None)
            dep_ = getattr(ctx, 'depth', None)
            if px_ is None or py_ is None or dep_ is None:
                continue
            _ss_vp = camera_matrices(scene.camera, ctx.width, ctx.height)[2]
            _ss = LI.screen_spot_params(light, _ss_vp, ctx.width, ctx.height,
                                        scene.camera)
            ss_en, ss_el, ss_lobe = LI.screen_spot_lobe(px_, py_, dep_, *_ss)
            _ss_sign = -1.0 if light.negative else 1.0
            ss_col = np.asarray(light.color, np.float32) * np.float32(
                _ss_sign * float(light.energy) / (4.0 * np.pi))
            ss_t = (surf.diffuse * surf.diffuse_level[:, None]) * ss_col[None, :]
            ss_t = ss_t * ss_lobe[:, None]
            ss_t = ss_t * inv_pi
            if lit_mask is not None:
                ss_t = ss_t * lit_mask[:, None]
            out += ss_t
            if want_split:
                _sp_add(li, ss_t, None)                        # R253
            ss_fog = ss_col[None, :] * (ss_en * ss_el)[:, None]
            if lit_mask is not None:
                ss_fog = ss_fog * lit_mask[:, None]
            _sf = getattr(ctx, 'spot_fog', None)
            ctx.spot_fog = ss_fog.astype(np.float32) if _sf is None \
                else (_sf + ss_fog).astype(np.float32)
            continue
        L, rad, dist = LI.sample(light, ctx.P, settings)
        ndl = M.dot(N, L)
        if not np.any(ndl > 0.0) and not np.any(surf.translucency > 0) \
                and model not in ('ANIME', 'CARTOON') \
                and not (model == 'MAX_TRANSLUCENT'
                         and np.any(surf.translucent_color > 0)) \
                and getattr(light, 'type', '') not in ('HEMI', 'AREA'):
            # the hemi WRAP lights faces the dot product writes off --
            # and an area lamp's Stokes energy can be positive where
            # the centre dot is not (the rectangle extends past the
            # point's horizon), so it never takes this shortcut
            continue
        # the shaders run BEFORE the shadow query, because their own
        # gates -- the C's gates, transcribed -- decide which samples
        # can receive ANY contribution from this lamp. A shadow ray
        # for a sample whose diffuse AND specular are both zero
        # multiplies nothing: shade_one_light's Blinn returns 0 below
        # nl=0.01, CookTorr/Phong below nh=0, Lambert below inp=0, so
        # the whole back side of a character never needed its five
        # sun-shadow rays. Exact by construction: the mask IS "the
        # result reads vis here", not a heuristic. Shadows Only
        # accumulation (only_accum) reads vis at every sample, so its
        # presence disables the skip for the batch.
        area_nd = area_nd_back = None
        if getattr(light, 'type', '') == 'AREA':
            # BI's area lamp: the Stokes form factor becomes the
            # diffuse shader's input and multiplies the specular --
            # size, orientation and distance all speak through it
            if np.any(surf.translucency > 0):
                area_nd, area_nd_back = LI.area_inp(light, ctx.P, N,
                                                    want_back=True)
            else:
                area_nd = LI.area_inp(light, ctx.P, N)
        # R251 (LIGHT-B2): every model's own view term takes Vs -- the
        # true eye, or the frame's camera axis for the console light
        # units (SH.AXIS_MODELS) and under Specular Viewer AXIS
        dif, spec = SH.evaluate(model, surf, N, L, Vs,
                                area_ndl=area_nd,
                                area_ndl_back=area_nd_back)
        # R243: a Max shader whose diffuse carries its own colour (the
        # Oren-Nayar pair, Translucent) returns it as (N, 3); the scalar
        # every other road reads is its mean, and the contribution
        # takes the colour as given instead of the diffuse socket
        if model == 'GX_LIGHT' and \
                (getattr(light, 'type', '') not in ('SUN', 'HEMI')
                 or np.any(surf.gx_attn_fn > 0.5)):
            # R251 (LIGHT-B2): GX_AF_SPEC lit from directional lights
            # only (GX_InitSpecularDir) -- a point or spot lamp adds no
            # highlight on the GameCube. R252: nor does a channel whose
            # attenuation function is GX_AF_SPOT or GX_AF_NONE (the
            # attenuation unit is busy with the cone, or off)
            spec = np.zeros_like(spec)
        elif model == 'DS_FIXED':
            # R251 (LIGHT-B2): the DS's table highlight, evaluated HERE
            # on both roads (the GPU's per-light block reads hal_dstab
            # right after its evaluate call): the unnormalised half
            # vector of the light and the fixed line of sight, squared,
            # through the material's 128-entry shininess table
            spec = (SH.ds_spec(N, L, Vs, surf.glossiness)[:, None]
                    * surf.specular).astype(np.float32)
        dif_rgb = None
        if np.ndim(dif) == 2:
            dif_rgb = dif
            dif = dif.mean(axis=1).astype(np.float32)
        if model not in SH.MAX_MODELS:
            # R251 (LIGHT-B2): POV-Ray's finish dials, in trace.cpp's
            # order -- brilliance on the diffuse cosine (skipped at 1.0),
            # crand's per-lamp, per-pixel grain off the direct diffuse,
            # metallic's rational Fresnel tint on the highlight. The Max
            # shaders return a coloured diffuse and keep their own laws
            # (inert there, on both roads); the Hemi override below
            # replaces both lobes afterwards, so a Hemi lamp is inert too
            dif = SH.apply_brilliance(dif, surf.brilliance)
            if have_id and np.any(surf.crand > 0.0):
                z = 977 + 131 * li + \
                    7919 * int(getattr(settings, 'seed', 0) or 0)
                if getattr(settings, 'crand_per_frame', False):
                    z += 1013 * int(getattr(ctx, 'frame', 0) or 0)
                # the hash reads only the low 31 bits of its salt, so
                # the masked salt IS the int32-wrapped salt's hash (the
                # GLSL carries the same masked literal)
                z = int(np.int64(z) & 0x7fffffff)
                from . import patterns as PT
                h_cr = PT.sample_u(spx, spy, z)
                dif = np.maximum(dif - surf.crand * h_cr,
                                 np.float32(0.0)).astype(np.float32)
            if np.any(surf.pov_metallic > 0.0):
                spec = SH.apply_pov_metallic(spec, ndl, surf.pov_metallic,
                                             surf.diffuse)
        if area_nd is not None:
            # shade_one_light's area lamp correction: specfac *= inp
            spec = spec * area_nd[:, None]
        need = None
        if only_accum is None and getattr(light, 'type', '') != 'HEMI':
            need = (dif != 0.0) | (spec != 0.0).any(axis=1)
            if lit_mask is not None:
                need &= lit_mask         # excluded objects contribute 0
            if not need.any():
                continue
        vis = LI.visibility(light, ctx.P, N, L, dist, settings, bvh, rng,
                            sample_xy=(spx, spy, li) if have_id else None,
                            mask=need)
        if want_split and _sp_shadows(light):
            # R253: the Shadow pass reads the vis this lamp already
            # traced -- 1.0 (no darkening) where `need` was False, so the
            # existing skip logic and the beauty are untouched
            _dk = (np.float32(1.0) - vis).astype(np.float32)
            if need is not None:
                _dk = np.where(need, _dk, np.float32(0.0))
            if lit_mask is not None:
                _dk = _dk * lit_mask
            _spl['sh_acc'] += _dk
        if only_accum is not None:
            smode = light.shadow if settings.shadow_default == 'PER_LIGHT' \
                else settings.shadow_default
            if settings.shadows and light.shadow != 'NONE' and \
                    smode != 'NONE':
                dark = 1.0 - vis
                if lit_mask is not None:
                    dark = dark * lit_mask
                only_accum += dark
                only_ir += 1.0
        if receive_off:
            # Shadow > Receive off: shadows never darken this material
            vis = np.where(surf.shadow_receive > 0.5, vis, 1.0)
        if cel_ss_term is not None and light is cel_key:
            # R238: the screen shadow rides the key lamp's visibility
            vis = (vis * cel_ss_term).astype(np.float32)
        if getattr(light, 'only_shadow', False):
            # R251 F014: Blender Internal's LA_ONLYSHADOW (2.79
            # shadeoutput.c `shr->shad -= ...`): the lamp adds nothing
            # and subtracts the plain diffuse it would have given where
            # its shadow falls -- unshadowed minus shadowed under the
            # shadow-colour rule `vis + shcol*(1 - vis)`, so the dark
            # term is (1 - vis)*(1 - shcol) per channel. No specular,
            # no ramp, no terminator bias, no per-light clamp (the
            # subtraction is negative; Negative lamps have no floor
            # either). Nothing at all when the lobe flags give no
            # diffuse. The GPU block associates identically:
            # (((dif*col*level)*rad)*inv_pi)*dark.
            if light.affect_diffuse and not light.specular_only:
                lamp_os_rad = -rad if light.negative else rad
                if dif_rgb is not None:
                    lamp_os = (dif_rgb * surf.diffuse_level[:, None]) \
                        * lamp_os_rad
                else:
                    lamp_os = (dif[:, None] * surf.diffuse
                               * surf.diffuse_level[:, None]) * lamp_os_rad
                lamp_os = lamp_os * inv_pi
                lamp_dark = (np.float32(1.0) - vis)[:, None]
                _oshc = np.asarray(getattr(light, 'shadow_color',
                                           (0.0, 0.0, 0.0)), np.float32)
                if float(_oshc.max()) > 0.0:
                    lamp_dark = lamp_dark * (np.float32(1.0)
                                             - _oshc[None, :])
                lamp_os = lamp_os * lamp_dark
                if lit_mask is not None:
                    lamp_os = lamp_os * lit_mask[:, None]
                if track_result:
                    diff_acc -= lamp_os
                else:
                    out -= lamp_os
                if want_split:
                    _sp_add(li, -lamp_os, None)                # R253
            continue
        if sh_key is not None and light is sh_key:
            # R251 (MAT-A C057): the key lamp's lit fraction, captured
            # after `vis` is final (absent when the key never gets here)
            extras['key_lit'] = (vis * np.maximum(ndl, 0.0)).astype(
                np.float32)
        if not np.any(vis > 0.0):
            continue
        if getattr(light, 'type', '') == 'HEMI':
            # BI's Hemi lamp REPLACES the shaders: diffuse is the
            # 0.5+0.5*N.L wrap, specular a wrapped half-vector pow --
            # 2.79 shade_one_light's LA_HEMI branches, both lobes
            ndl_h = M.dot(N, L)
            dif = (0.5 * ndl_h + 0.5).astype(np.float32)
            h_h = M.normalize(L + V)
            t_h = 0.5 * M.dot(N, h_h) + 0.5
            # verbatim (R155): t = spec(t, shi->har) -- the integer-bit
            # spec() table, not a float pow
            sp_h = SH.bi_spec_pow(t_h, surf.glossiness)
            spec = sp_h[:, None] * surf.specular
        if light.negative:
            rad = -rad
        if model == 'CARTOON':
            # R228: the paint keeps the STRONGEST lamp's verdict -- a
            # region is lit if any lamp reaches it -- and the lamp's
            # energy only matters through Lamp Influence
            x_raw = np.clip(dif, 0.0, 1.0) * vis
            if lit_mask is not None:
                x_raw = x_raw * lit_mask
            x = x_raw if (light.affect_diffuse
                          and not light.specular_only) \
                else np.zeros_like(x_raw)
            cart_lit = np.maximum(cart_lit, x)
            cart_rad += rad * x[:, None] * inv_pi
            if light.affect_specular and not light.diffuse_only \
                    and not suppress_spec:
                # the painted dot: gated on the wrapped half-vector,
                # and confined to THIS lamp's own lit step (a
                # highlight sits inside the lit region of the form,
                # never in the shadow tone; a specular-only lamp keeps
                # its own verdict for the gate)
                h_c = M.normalize(L + V)
                sndh = np.clip(M.dot(N, h_c) * np.float32(0.5)
                               + np.float32(0.5), 0.0, 1.0)
                # Highlight Size is the dot's RADIUS: the gate opens at
                # N.H > 1 - size^2/2, so the dot's angular radius grows
                # about linearly with the dial (0.3 ~ 17 degrees, 0.5 ~
                # 29, 1.0 a 60-degree cap -- never the whole hemisphere)
                # -- in wrapped units that edge is 1 - size^2/4
                hs_c = np.clip(surf.cartoon_hl_size, 0.0, 1.0)
                edge_c = 1.0 - np.float32(0.25) * hs_c * hs_c
                gate = _anime_smooth(edge_c - surf.cartoon_hl_soft,
                                     edge_c + surf.cartoon_hl_soft, sndh)
                gate = np.where(surf.cartoon_hl_size > 1e-6, gate, 0.0)
                t_l = _anime_smooth(surf.cartoon_th - surf.cartoon_soft,
                                    surf.cartoon_th + surf.cartoon_soft,
                                    x_raw)
                cart_hl = np.maximum(cart_hl, gate * t_l)
            continue
        if model == 'ANIME':
            # the cel composition: bands, tints and the stepped
            # highlight, with vis folded into the band input; the
            # generic tail multiplies are bypassed below
            dcontrib, scontrib = _anime_lamp(
                surf, dif, vis, rad, N, L, V,
                light.affect_diffuse and not light.specular_only,
                light.affect_specular and not light.diffuse_only)
            if suppress_spec:
                scontrib = np.zeros_like(dcontrib)
            if lit_mask is not None:
                dcontrib *= lit_mask[:, None]
                scontrib *= lit_mask[:, None]
            if want_split:
                _sp_add(li, dcontrib, scontrib)                # R253
            if track_result:
                diff_acc += dcontrib
                spec_acc += scontrib
            else:
                contrib = dcontrib + scontrib
                if clamp > 0.0:
                    contrib = np.minimum(contrib, clamp)
                out += contrib
                if spec_acc is not None:
                    spec_acc += scontrib
            continue
        dcontrib = np.zeros((n, 3), np.float32)
        scontrib = np.zeros((n, 3), np.float32)
        if light.affect_diffuse and not light.specular_only:
            dcol = surf.diffuse
            if ramp_dif is not None and ramp_dif.get('input') != 'RESULT':
                inp = ramp_dif.get('input', 'SHADER')
                if inp == 'ENERGY':
                    lum = (0.3 * rad[:, 0] + 0.58 * rad[:, 1] +
                           0.11 * rad[:, 2])
                    fac_in = dif * vis * lum
                elif inp == 'NORMAL':
                    # +N.V: facing reads the band's RIGHT end, grazing
                    # its LEFT -- pinned by the field's own 2.79 files
                    # (edge-darkening bands: black alpha at pos 0).
                    # fresnel_fac is sign-symmetric and never pinned it
                    fac_in = ndv_g
                else:                       # SHADER
                    fac_in = dif
                band = _band(ramp_dif, fac_in)
                dcol = SH.bi_ramp_blend(
                    ramp_dif.get('blend', 'MIX'), surf.diffuse,
                    band[:, 3] * float(ramp_dif.get('factor', 1.0)),
                    band[:, :3])
            if dif_rgb is not None:
                dcontrib = (dif_rgb * surf.diffuse_level[:, None]) * rad
            else:
                dcontrib = (dif[:, None] * dcol *
                            surf.diffuse_level[:, None]) * rad
        if light.affect_specular and not light.diffuse_only:
            sp = spec
            if ramp_spec is not None and ramp_spec.get('input') != 'RESULT':
                # recover the shader scalar from evaluate's coloured
                # return (spec = scalar * specular colour); a black
                # specular colour has no recoverable scalar and stays
                # dark, ramp or no ramp
                mx = surf.specular.max(axis=1)
                idx = surf.specular.argmax(axis=1)
                sr = spec[np.arange(n), idx]
                sfac = np.where(mx > 1e-9, sr / np.maximum(mx, 1e-9), 0.0)
                inp = ramp_spec.get('input', 'SHADER')
                if inp == 'ENERGY':
                    lum = (0.3 * rad[:, 0] + 0.58 * rad[:, 1] +
                           0.11 * rad[:, 2])
                    fac_in = sfac * vis * lum
                elif inp == 'NORMAL':
                    # +N.V: facing reads the band's RIGHT end, grazing
                    # its LEFT -- pinned by the field's own 2.79 files
                    # (edge-darkening bands: black alpha at pos 0).
                    # fresnel_fac is sign-symmetric and never pinned it
                    fac_in = ndv_g
                else:                       # SHADER
                    fac_in = sfac
                band = _band(ramp_spec, fac_in)
                scol = SH.bi_ramp_blend(
                    ramp_spec.get('blend', 'MIX'), surf.specular,
                    band[:, 3] * float(ramp_spec.get('factor', 1.0)),
                    band[:, :3])
                sp = sfac[:, None] * scol
            if not settings.specular_in_gamma:
                sp = np.power(np.maximum(sp, 0.0), 2.2)
            if model in SH.LEVEL_FREE_MODELS:
                # R243: Strauss has no Specular Level in Max, Multi-Layer
                # applies its two levels inside -- the loop scales by nothing
                scontrib = sp * rad
            else:
                scontrib = sp * surf.specular_level[:, None] * rad
            if sheen_exp is not None:
                # velvet: light scattered back at grazing angles, so the lobe
                # lives at the silhouette and vanishes face-on. It still needs
                # a light -- unlike the rim term, which is the cheat version of
                # the same look and needs none.
                sh = (np.power(edge_vn, sheen_exp) *
                      np.maximum(ndl, 0.0) * surf.sheen)
                scontrib = scontrib + surf.sheen_color * sh[:, None] * rad
        # phongcorr multiplies the diffuse contribution only (i = is *
        # phongcorr); the ramps read the PRE-correction shader value
        # (add_to_diffuse's third argument is `is`), which fac_in
        # above already did
        if pc_wanted and light.type not in ('HEMI', 'AREA'):
            smode = light.shadow if settings.shadow_default == 'PER_LIGHT' \
                else settings.shadow_default
            lamp_shadowed = smode in ('MAP', 'RAY') \
                and light.shadow != 'NONE'
            pc = np.ones(n, np.float32)
            done = np.zeros(n, bool)
            if pc_raybias and lamp_shadowed and smode == 'RAY' \
                    and pc_smooth is not None:
                m = pc_smooth
                if np.any(m):
                    pc = np.where(m, _pcurve(ndl, pc_thresh), pc)
                    done = m
            if pc_sbias != 0.0 and lamp_shadowed:
                m2 = ~done
                if np.any(m2):
                    pc = np.where(
                        m2, _pcurve(ndl, np.float32(pc_sbias)), pc)
            # MA_SHADOW is per material: pixels not receiving shadows
            # keep phongcorr = 1, exactly the C's outer gate
            pc = np.where(surf.shadow_receive > 0.5, pc, 1.0)
            dcontrib *= pc[:, None]
        # R164: the lamp's SHADOW COLOUR -- shade_one_light adds
        # lashdw*(i_noshad - i) back to the DIFFUSE (never the spec):
        # equivalently the diffuse sees vis + shadow_color*(1 - vis)
        # per channel. Black keeps the classic full shadow.
        _shcol = np.asarray(getattr(light, 'shadow_color',
                                    (0.0, 0.0, 0.0)), np.float32)
        if float(_shcol.max()) > 0.0:
            dcontrib *= inv_pi * (vis[:, None]
                                  + _shcol[None, :]
                                  * (1.0 - vis[:, None]))
        else:
            dcontrib *= inv_pi * vis[:, None]
        scontrib *= inv_pi * vis[:, None]
        if suppress_spec:
            # the SSS pre-pass: combinedflag &= ~SCE_PASS_SPEC -- the
            # lobe never reaches the stored point colour
            scontrib = np.zeros_like(dcontrib)
        if lit_mask is not None:
            dcontrib *= lit_mask[:, None]
            scontrib *= lit_mask[:, None]
        if want_split:
            _sp_add(li, dcontrib, scontrib)                    # R253
        if track_result:
            diff_acc += dcontrib
            spec_acc += scontrib
        else:
            contrib = dcontrib + scontrib
            if clamp > 0.0:
                contrib = np.minimum(contrib, clamp)
            out += contrib
            if spec_acc is not None:
                spec_acc += scontrib

    if track_result:
        if ramp_dif is not None and ramp_dif.get('input') == 'RESULT':
            fac_in = (0.3 * diff_acc[:, 0] + 0.58 * diff_acc[:, 1] +
                      0.11 * diff_acc[:, 2])
            band = _band(ramp_dif, fac_in)
            diff_acc = SH.bi_ramp_blend(
                ramp_dif.get('blend', 'MIX'), diff_acc,
                band[:, 3] * float(ramp_dif.get('factor', 1.0)),
                band[:, :3])
        if ramp_spec is not None and ramp_spec.get('input') == 'RESULT':
            fac_in = (0.3 * spec_acc[:, 0] + 0.58 * spec_acc[:, 1] +
                      0.11 * spec_acc[:, 2])
            band = _band(ramp_spec, fac_in)
            spec_acc = SH.bi_ramp_blend(
                ramp_spec.get('blend', 'MIX'), spec_acc,
                band[:, 3] * float(ramp_spec.get('factor', 1.0)),
                band[:, :3])
        if sss is not None:
            # shade_lamp_loop's SSS block, verbatim: the scattered
            # radiance replaces the accumulated diffuse, shaped by the
            # material colour per sss_texfac (shr->col = the textured
            # colour, invalpha = 1/col[3])
            sampled = np.asarray(sss['sampled'], np.float32)
            texfac = float(sss.get('texfac', 0.0))
            alpha = np.asarray(surf.opacity, np.float32)
            invalpha = np.where(alpha > np.finfo(np.float32).eps,
                                1.0 / np.maximum(alpha, 1e-30), 1.0)
            if texfac == 0.0:
                col = surf.diffuse * invalpha[:, None]
            elif texfac == 1.0:
                col = np.broadcast_to(
                    invalpha[:, None], (n, 3)).astype(np.float32)
            else:
                col = np.power(
                    np.maximum(surf.diffuse * invalpha[:, None], 0.0),
                    np.float32(1.0 - texfac))
            diff_acc = sampled * col
        if exposure_on:
            # wrld_exposure_correct, letters from the original 2003
            # commit (unchanged through 2.79, removed 2.8):
            #   linfac = 1 + pow(2*exp + 0.5, -10)
            #   logfac = log((linfac-1)/linfac) / range
            #   col    = linfac * (1 - exp(col * logfac))
            # The C corrects the DIFFUSE total ("has no spec!") and
            # SPEC separately, AFTER the SSS replacement, BEFORE
            # ambient and emit join -- exactly this spot.
            linfac = np.float32(1.0 + (2.0 * w_exp + 0.5) ** -10.0)
            logfac = np.float32(
                np.log((linfac - 1.0) / linfac)
                / (w_rng if abs(w_rng) > 1e-6 else 1e-6))
            diff_acc = (linfac
                        * (1.0 - np.exp(diff_acc * logfac))).astype(
                            np.float32)
            spec_acc = (linfac
                        * (1.0 - np.exp(spec_acc * logfac))).astype(
                            np.float32)
        # R251 (MAT-A): a carrying model's corner EXCLUDES the specular
        # (it rides the corner alpha and is added after the texel)
        light_part = diff_acc \
            if (extras is not None and extras.get('want_spec')) \
            else diff_acc + spec_acc
        if want_split:
            # R253: RESULT ramps, SSS and the world exposure rewrote the
            # sums -- the split reports what `out` receives
            _spl['diff'] = np.asarray(diff_acc, np.float32)
            _spl['spec'] = np.asarray(spec_acc, np.float32)
        if clamp > 0.0:
            light_part = np.minimum(light_part, clamp)
        out += light_part

    if extras is not None:
        if spec_acc is not None:
            extras['spec_acc'] = spec_acc
        if only_accum is not None:
            extras['only_shadow'] = (only_accum, only_ir)

    if cel_mode > 0:
        # R238: the fixed key lights the cel once, at the energy of the
        # paint (a radiance of pi: the lit tone is the colour as
        # painted), through the strongest caster's visibility and the
        # screen shadow -- the same composition a lamp gets
        vis_key = np.where(vis_acc >= 0.0, vis_acc, 1.0).astype(np.float32)
        if cel_ss_term is not None:
            vis_key = (vis_key * cel_ss_term).astype(np.float32)
        wrap_k = np.clip(M.dot(N, cel_L) * np.float32(0.5) + np.float32(0.5),
                         0.0, 1.0).astype(np.float32)
        rad_k = np.full((n, 3), np.float32(np.pi), np.float32)
        if model == 'ANIME':
            dcontrib, scontrib = _anime_lamp(surf, wrap_k, vis_key, rad_k,
                                             N, cel_L, V, True,
                                             not suppress_spec)
            if want_split:
                _sp_add(None, dcontrib, scontrib)              # R253
            if track_result:
                diff_acc += dcontrib
                spec_acc += scontrib
            else:
                contrib = dcontrib + scontrib
                if clamp > 0.0:
                    contrib = np.minimum(contrib, clamp)
                out += contrib
                if spec_acc is not None:
                    spec_acc += scontrib
        else:
            x_k = (wrap_k * vis_key).astype(np.float32)
            cart_lit = x_k
            cart_rad = (rad_k * x_k[:, None] * inv_pi).astype(np.float32)
            if not suppress_spec:
                h_k = M.normalize(cel_L + V)
                sndh_k = np.clip(M.dot(N, h_k) * np.float32(0.5)
                                 + np.float32(0.5), 0.0, 1.0)
                hs_k = np.clip(surf.cartoon_hl_size, 0.0, 1.0)
                edge_k = 1.0 - np.float32(0.25) * hs_k * hs_k
                gate_k = _anime_smooth(edge_k - surf.cartoon_hl_soft,
                                       edge_k + surf.cartoon_hl_soft, sndh_k)
                gate_k = np.where(surf.cartoon_hl_size > 1e-6, gate_k, 0.0)
                t_k = _anime_smooth(surf.cartoon_th - surf.cartoon_soft,
                                    surf.cartoon_th + surf.cartoon_soft, x_k)
                cart_hl = (gate_k * t_k).astype(np.float32)
    if model == 'CARTOON':
        # R228: paint replaces the whole accumulation -- no ambient, no
        # lamp energy (unless Lamp Influence asks), one shadow tone
        out = _cartoon_compose(surf, cart_lit, cart_hl, cart_rad)
        # R241: the highlight cel over the paint -- the same hair
        # shine band the Anime Shader wears (inert until its sockets
        # say otherwise, so every older cartoon is bitwise)
        out = _anime_hair_shine(out, surf, ctx, N, V, hair_key_z)
    if model == 'ANIME':
        # R229: the hair shine band, painted OVER the banded result
        # (before emission and the silhouette cheats, like the paint)
        out = _anime_hair_shine(out, surf, ctx, N, V, hair_key_z)
    if fixed_mask is not None:
        # R252: a batch mixing fixed-shaded and lit fragments (a Mix
        # Shader between two console materials) keeps each its own
        out = np.where(fixed_mask[:, None], fixed, out).astype(np.float32)
    if model == 'GX_LIGHT':
        # R251 (LIGHT-B2): the GX vertex unit's saturated 8-bit lit
        # colour (per corner at the Gouraud rate the model is meant for)
        out = SH.quantize_lit(out, 255)
    elif model == 'DS_FIXED':
        # R251 (LIGHT-B2): the DS light unit's 5-bit saturation, once
        # (GBATEK: the vertex colour saturates at 31)
        out = SH.quantize_lit(out, 31)
    if settings.clamp_specular:
        out = np.minimum(out, 64.0)
    if want_split:
        # R253: the non-separable tails -- the cartoon paint and the cel
        # bands (and their hair shine), a fixed-shade mix, the GX / DS
        # saturation -- REPLACE the sum: Diffuse then carries whatever is
        # neither ambient nor specular (cel paint reports as diffuse),
        # so Diffuse + Spec + Ambient + Emit still equals the beauty
        if model in ('CARTOON', 'ANIME', 'GX_LIGHT', 'DS_FIXED') \
                or fixed_mask is not None:
            _spl['diff'] = (out - _spl['amb'] - _spl['spec']).astype(np.float32)
        extras['split'] = _split_pack(surf, _spl, sp_ao, n, sp_names)
    out = out + surf.emission
    return apply_surface_effects(out, surf, N, V, rim_field=cel_rim)


def _split_pack(surf, sp, sp_ao, n, names):
    """R253: the light split as pass name -> (n, 3) float32, for the
    names asked (None = all): the ambient snapshot, the lamp sums, the
    emission, the Shadow average (1 lit, 0 dark; 1.0 when no lamp
    shadows), the AO factor (white when AO is off), the material colour
    and the per-lamp slots (Light00.. zeros where no lamp landed)."""
    def want(k):
        return names is None or k in names
    ones3 = None
    out = {}
    if want('Diffuse'):
        out['Diffuse'] = sp['diff']
    if want('Spec'):
        out['Spec'] = sp['spec']
    if want('Ambient'):
        out['Ambient'] = sp['amb']
    if want('Emit'):
        out['Emit'] = np.broadcast_to(
            np.asarray(surf.emission, np.float32), (n, 3))
    if want('Shadow'):
        ir = float(sp['sh_ir'])
        if ir > 0.0:
            sh = (np.float32(1.0) - sp['sh_acc'] / np.float32(ir))
            sh = np.clip(sh, 0.0, 1.0).astype(np.float32)
            out['Shadow'] = np.broadcast_to(sh[:, None], (n, 3))
        else:
            ones3 = np.ones((n, 3), np.float32)
            out['Shadow'] = ones3
    if want('AO'):
        if sp_ao is not None:
            out['AO'] = np.broadcast_to(
                np.asarray(sp_ao, np.float32)[:, None], (n, 3))
        else:
            ones3 = np.ones((n, 3), np.float32) if ones3 is None else ones3
            out['AO'] = ones3
    if want('Color'):
        out['Color'] = (surf.diffuse
                        * surf.diffuse_level[:, None]).astype(np.float32)
    if sp['lamps']:
        zeros3 = None
        for i in range(LIGHT_PASS_SLOTS):
            k = f'Light{i:02d}'
            if not want(k):
                continue
            slot = sp['light'].get(i)
            if slot is None:
                zeros3 = np.zeros((n, 3), np.float32) \
                    if zeros3 is None else zeros3
                slot = zeros3
            out[k] = slot
    return out


def _split_unlit(surf, base, n, names, color=None):
    """R253: the split of a surface no lamp touched (CONSTANT /
    WIREFRAME, fixed shading): the shown colour as Diffuse, nothing
    specular or ambient, the emission as Emit, shadow and AO open."""
    sp = {'amb': np.zeros((n, 3), np.float32),
          'diff': np.asarray(base, np.float32),
          'spec': np.zeros((n, 3), np.float32), 'light': {},
          'sh_acc': np.zeros(n, np.float32), 'sh_ir': 0.0,
          'lamps': names is None or any(k.startswith('Light')
                                        for k in names)}
    out = _split_pack(surf, sp, None, n, names)
    if color is not None and 'Color' in out:
        out['Color'] = np.asarray(color, np.float32)
    return out


def _blend_layer(out, color, f, mode):
    """One silhouette layer onto the lit result, by its blend mode.

    Mode 0 is Add -- the only behaviour that existed before 1.38, kept
    bit for bit on its own fast path. 1 Mix, 2 Multiply, 3 Screen. The
    factor is clamped for the bounded modes and left free for Add,
    exactly as the original addition left it free.
    """
    fcol = color * f[:, None]
    mi = np.rint(np.asarray(mode, np.float32)).astype(np.int32) \
        if mode is not None else np.zeros(1, np.int32)
    if not np.any(mi):
        return out + fcol
    fc = np.clip(f, 0.0, 1.0)[:, None]
    res = out + fcol
    if np.any(mi == 1):
        res = np.where((mi == 1)[:, None],
                       out * (1.0 - fc) + color * fc, res)
    if np.any(mi == 2):
        res = np.where((mi == 2)[:, None],
                       out * (1.0 - fc * (1.0 - color)), res)
    if np.any(mi == 3):
        res = np.where((mi == 3)[:, None],
                       1.0 - (1.0 - out) * (1.0 - np.clip(fcol, 0.0, 1.0)),
                       res)
    return res


def apply_surface_effects(out, surf, N, V, rim_field=None):
    """Fresnel, rim, matcap, reflection tint and backface override.

    These sit outside the reflectance model on purpose: they are the artistic
    cheats every package of the era offered on top of whichever shader you
    picked, so they behave the same on Lambert as on Cook-Torrance.
    R238: `rim_field` is the cel field's depth rim at this batch's pixels;
    a cel material whose Rim mode is SCREEN takes it in place of the
    Fresnel power.
    """
    facing = np.clip(np.abs(M.dot(N, V)), 0.0, 1.0)
    edge = 1.0 - facing

    if np.any(surf.fresnel > 1e-4):
        f = np.power(edge, np.maximum(surf.fresnel_power, 0.01)) * surf.fresnel
        out = _blend_layer(out, surf.fresnel_color, f * surf.specular_level,
                           getattr(surf, 'fresnel_blend', None))

    if np.any(surf.rim > 1e-4):
        r = np.power(edge, np.maximum(surf.rim_power, 0.01)) * surf.rim
        screen = getattr(surf, 'cel_rim_mode', None)
        if screen is not None and np.any(screen > 0.5):
            rf = np.zeros_like(r) if rim_field is None \
                else np.asarray(rim_field, np.float32) * surf.rim
            r = np.where(screen > 0.5, rf, r).astype(np.float32)
        out = _blend_layer(out, surf.rim_color, r,
                           getattr(surf, 'rim_blend', None))

    if np.any(surf.matcap_blend > 1e-4):
        k = np.clip(surf.matcap_blend, 0.0, 1.0)
        mm = getattr(surf, 'matcap_mode', None)
        mi = np.rint(np.asarray(mm, np.float32)).astype(np.int32) \
            if mm is not None else np.zeros(1, np.int32)
        kc = k[:, None]
        out2 = out * (1.0 - kc) + surf.matcap * kc     # 0: Mix, the original
        if np.any(mi == 1):
            out2 = np.where((mi == 1)[:, None], out + surf.matcap * kc, out2)
        if np.any(mi == 2):
            out2 = np.where((mi == 2)[:, None],
                            out * (1.0 - kc * (1.0 - surf.matcap)), out2)
        if np.any(mi == 3):
            out2 = np.where((mi == 3)[:, None],
                            1.0 - (1.0 - out) *
                            (1.0 - np.clip(surf.matcap * kc, 0.0, 1.0)), out2)
        out = out2

    if np.any(surf.backface_mix > 1e-4) and surf.backfacing is not None:
        k = (np.clip(surf.backface_mix, 0.0, 1.0) *
             np.clip(surf.backfacing, 0.0, 1.0))[:, None]
        out = out * (1.0 - k) + surf.backface_color * k
    return out


def radiosity_albedos(scene):
    """The per-material bleed colours, as one (M,3) float32 table.

    The colour a surface LENDS its neighbours is its flat diffuse -- the
    material's own field, which graph materials carry as their display
    colour. Reading the textured, per-pixel albedo at every gather hit is
    beyond both devices equally, and the era's radiosity previews used
    flat patch colours anyway. Built once per scene and cached on it; the
    GPU bakes THIS table into its selector so both devices bleed the same
    numbers.
    """
    cached = getattr(scene, '_radiosity_albedo', None)
    if cached is not None and cached.shape[0] == len(scene.materials):
        return cached
    table = np.array([tuple(getattr(m, 'diffuse', (0.8, 0.8, 0.8)))[:3]
                      for m in scene.materials] or [(0.8, 0.8, 0.8)],
                     np.float32)
    scene._radiosity_albedo = table
    return table


#: the radiosity gather's hash-stream salt -- distinct from lights (131*li),
#: the AO pass (8389), the AO node (6151) and reflection blur (10009)
RADIOSITY_SALT = 9973


def radiosity_field(job, gbuf, settings, scene, bvh):
    """The interpolated gather: irradiance at every Nth pixel, as a grid.

    LightWave's shipping radiosity was exactly this -- evaluate sparsely,
    blend between the points -- and it is what makes the feature usable at
    real resolutions: spacing 2 casts a quarter of the rays, 4 a
    sixteenth. The grid lives on the RENDER-resolution pixel lattice
    (grid point (gx, gy) at pixel (gx*N, gy*N)); each point gathers at
    the FIRST COVERED pixel of its NxN block in row-major order, with
    that pixel's own deterministic sampling identity -- so the GPU's grid
    pass, walking the same blocks in the same order, draws the same rays
    and lands the same numbers. A block with no coverage is INVALID and
    carries zero weight at lookup; a pixel whose four corners are all
    invalid falls back to the flat ambient colour.

    Returns (Gh, Gw, 4) float32 -- rgb irradiance + validity -- or None
    when spacing is 1 (the full-rate path, byte-for-byte the 1.25.95
    behaviour).
    """
    N = max(int(getattr(settings, 'radiosity_spacing', 1)), 1)
    if N <= 1 or bvh is None:
        return None
    mask = gbuf.mask()
    rh, rw = mask.shape
    Gw = (rw + N - 1) // N
    Gh = (rh + N - 1) // N
    # pad coverage to whole blocks, then order each block row-major
    pad = np.zeros((Gh * N, Gw * N), bool)
    pad[:rh, :rw] = mask
    blocks = pad.reshape(Gh, N, Gw, N).transpose(0, 2, 1, 3) \
                .reshape(Gh, Gw, N * N)
    valid = blocks.any(axis=2)
    first = blocks.argmax(axis=2)          # first covered, row-major
    gy, gx = np.nonzero(valid)
    dy, dx = np.divmod(first[gy, gx], N)
    spy = gy * N + dy
    spx = gx * N + dx
    field = np.zeros((Gh, Gw, 4), np.float32)
    if gy.size:
        tri = gbuf.tri[spy, spx]
        bary = gbuf.bary[spy, spx]
        ctx = job.context(tri, bary, px=spx, py=spy)
        Nrm = M.normalize(ctx.N)
        irr = radiosity_gather(ctx.P, Nrm, bvh, scene, settings,
                               sample_xy=(spx.astype(np.int64),
                                          spy.astype(np.int64)))
        field[gy, gx, :3] = irr
        field[gy, gx, 3] = 1.0
    return field


def radiosity_lookup(field, spacing, px, py, ambient):
    """Bilinear over the grid, validity-weighted; all-invalid -> ambient.

    Pure float32 arithmetic in a fixed order: the GLSL twin runs the same
    expressions over the same texels, so both devices blend identically.
    """
    N = np.float32(max(int(spacing), 1))
    Gh, Gw = field.shape[:2]
    fx = np.asarray(px, np.float32) / N
    fy = np.asarray(py, np.float32) / N
    gx0 = np.minimum(np.floor(fx), Gw - 1).astype(np.int32)
    gy0 = np.minimum(np.floor(fy), Gh - 1).astype(np.int32)
    gx1 = np.minimum(gx0 + 1, Gw - 1)
    gy1 = np.minimum(gy0 + 1, Gh - 1)
    tx = np.clip(fx - gx0, 0.0, 1.0).astype(np.float32)
    ty = np.clip(fy - gy0, 0.0, 1.0).astype(np.float32)
    c00 = field[gy0, gx0]
    c10 = field[gy0, gx1]
    c01 = field[gy1, gx0]
    c11 = field[gy1, gx1]
    w00 = (1.0 - tx) * (1.0 - ty) * c00[:, 3]
    w10 = tx * (1.0 - ty) * c10[:, 3]
    w01 = (1.0 - tx) * ty * c01[:, 3]
    w11 = tx * ty * c11[:, 3]
    total = w00 + w10 + w01 + w11
    rgb = (c00[:, :3] * w00[:, None] + c10[:, :3] * w10[:, None] +
           c01[:, :3] * w01[:, None] + c11[:, :3] * w11[:, None])
    amb = np.asarray(ambient, np.float32)
    safe = np.maximum(total, np.float32(1e-6))
    out = np.where((total > 1e-6)[:, None], rgb / safe[:, None],
                   amb[None, :])
    return out.astype(np.float32)


def radiosity_gather(P, N, bvh, scene, settings, rng=None, sample_xy=None):
    """One-bounce gathered ambient: the era's Radiosity checkbox.

    Cosine-weighted hemisphere rays from the same deterministic streams
    the AO pass draws (its own salt): a ray that reaches the sky within
    `radiosity_distance` returns the scene's ambient colour -- exactly
    what the flat ambient term stood for -- and a ray that lands on a
    surface returns that surface's flat diffuse scaled by
    `radiosity_intensity` and a linear falloff over the gather distance:
    colour bleed. Plain AO is superseded while this is on, because the
    gather is occlusion-aware by construction (blocked sky IS the
    darkening).
    """
    from . import lights as LI
    n = P.shape[0]
    samples = max(int(settings.radiosity_samples), 1)
    dist = max(float(settings.radiosity_distance), 1e-4)
    intensity = float(settings.radiosity_intensity)
    amb = np.asarray(LI.ambient_light(scene, settings), np.float32)
    albedo = radiosity_albedos(scene)
    mat_index = scene.mesh.mat_index if scene.mesh is not None and \
        getattr(scene.mesh, 'mat_index', None) is not None else None
    t, b = M.orthonormal_basis(N)
    origin = P + N * max(settings.ray_bias, 1e-4)
    gather = np.zeros((n, 3), np.float32)
    tmax = np.full(n, dist, np.float32)

    def one_dir(u1, ca, sa):
        r = np.sqrt(u1)
        return M.normalize(t * (r * ca)[:, None] + b * (r * sa)[:, None] +
                           N * np.sqrt(np.maximum(np.float32(1.0) - u1,
                                                  np.float32(0.0)))[:, None])

    def gather_dir(d):
        tid, th, _u, _v = bvh.intersect(origin, d, tmax)
        hit = (tid >= 0) & (th <= dist)
        out = np.where(hit[:, None], np.float32(0.0), amb[None, :])
        if np.any(hit):
            idx = np.nonzero(hit)[0]
            mi = mat_index[tid[idx]] if mat_index is not None else \
                np.zeros(idx.size, np.int32)
            mi = np.clip(mi, 0, albedo.shape[0] - 1)
            fall = np.clip(1.0 - th[idx] / np.float32(dist), 0.0, 1.0)
            out[idx] = albedo[mi] * (fall * np.float32(intensity))[:, None]
        return out.astype(np.float32)

    if sample_xy is not None:
        from . import patterns as PT
        spx, spy = sample_xy
        seed = int(getattr(settings, 'seed', 0) or 0)
        for k in range(samples):
            z = 2 * k + RADIOSITY_SALT + 7919 * seed
            u1 = PT.sample_u(spx, spy, z)
            ca, sa = PT.sample_circle(PT.sample_u(spx, spy, z + 1))
            gather += gather_dir(one_dir(u1, ca, sa))
    else:
        rng = rng or np.random.default_rng(settings.seed)
        for _ in range(samples):
            u1 = rng.random(n).astype(np.float32)
            th_ = (2.0 * np.pi * rng.random(n)).astype(np.float32)
            gather += gather_dir(one_dir(u1, np.cos(th_), np.sin(th_)))
    return gather / np.float32(samples)


def ambient_occlusion(P, N, bvh, settings, rng=None, sample_xy=None):
    """1 = open sky, falling toward 0 in creases, over `ao_distance`.

    With `sample_xy` (integer pixel identity), the cosine-weighted
    hemisphere directions are a pure function of (pixel, sample, seed):
    the same picture whatever the batch order or thread count, and
    exactly reproducible by the deferred pass -- hash draws for the
    radius, the shared unit-circle table for the angle (a driver rounds
    sin/cos differently, and an occlusion ray is a cliff). Without an
    identity, the legacy sequential stream still runs.
    """
    n = P.shape[0]
    samples = max(int(settings.ao_samples), 1)
    t, b = M.orthonormal_basis(N)
    occ = np.zeros(n, np.float32)
    dist = float(settings.ao_distance)
    origin = P + N * max(settings.ray_bias, 1e-4)
    if sample_xy is not None:
        from . import patterns as PT
        spx, spy = sample_xy
        seed = int(getattr(settings, 'seed', 0) or 0)
        for k in range(samples):
            z = 2 * k + 8389 + 7919 * seed
            u1 = PT.sample_u(spx, spy, z)
            ca, sa = PT.sample_circle(PT.sample_u(spx, spy, z + 1))
            r = np.sqrt(u1)
            d = M.normalize(t * (r * ca)[:, None] +
                            b * (r * sa)[:, None] +
                            N * np.sqrt(np.maximum(np.float32(1.0) - u1,
                                                   np.float32(0.0)))[:, None])
            occ += bvh.occluded(origin, d,
                                np.full(n, dist, np.float32)).astype(
                                    np.float32)
        ao = 1.0 - (occ / samples) * float(settings.ao_intensity)
        return np.clip(ao, 0.0, 1.0)
    rng = rng or np.random.default_rng(settings.seed)
    for _ in range(samples):
        u1 = rng.random(n).astype(np.float32)
        u2 = rng.random(n).astype(np.float32)
        r = np.sqrt(u1)
        th = 2.0 * np.pi * u2
        d = M.normalize(t * (r * np.cos(th))[:, None] +
                        b * (r * np.sin(th))[:, None] +
                        N * np.sqrt(np.maximum(1.0 - u1, 0.0))[:, None])
        occ += bvh.occluded(origin, d, np.full(n, dist, np.float32)).astype(np.float32)
    ao = 1.0 - (occ / samples) * float(settings.ao_intensity)
    return np.clip(ao, 0.0, 1.0)


def shade_closure_flat(cl, ctx):
    """Shader-to-RGB: evaluate a closure as a colour. Legal for a rasteriser."""
    scene = getattr(ctx, 'scene', None)
    settings = ctx.settings
    if scene is None or settings is None:
        col = np.zeros((ctx.n, 4), np.float32)
        col[:, 3] = 1.0
        return col
    surf, model, nrm = closure_to_surface(cl, ctx, settings)
    if nrm is not None:
        ctx = _with_normal(ctx, nrm)
    rgb = light_surface(surf, model, ctx, scene, settings,
                        getattr(ctx, 'bvh', None))
    return np.concatenate([rgb, surf.opacity[:, None]], axis=1).astype(np.float32)


def _with_normal(ctx, nrm):
    import copy
    c = copy.copy(ctx)
    c.N = M.normalize(nrm)
    return c


# ---------------------------------------------------------- world / horizon


def _make_ground_light(scene, st, bvh, lts):
    """R204: per-point irradiance callback for the infinite ground plane.

    "The infinite floors don't react to lighting either." The analytic
    plane is background, not geometry, so no light ever reached it. This
    closure is how it reaches it now: for the plane's hit points it sums
    the ambient pool and every lamp -- Lambert against the plane's
    straight-up normal (the wrap term for hemis, the Stokes form factor
    for area lamps, the classic falloffs via `lights.sample`) -- and
    shadows each lamp with the SAME machinery geometry uses
    (`lights.visibility`: ray-traced against the BVH or looked up in the
    lamp's shadow map), so an object standing on the floor finally casts
    onto it. Determinism: soft-shadow jitter takes its pixel identity
    from the QUANTISED WORLD POSITION -- a pure function of the point,
    so the same frame is the same picture whatever the batch order,
    chunking or device, exactly as the doctrine demands.
    """
    from . import lights as LI
    amb = LI.ambient_light(scene, st)
    # the same watts-to-radiance conversion light_surface applies to
    # every lobe -- without it the floor renders pi times brighter than
    # the geometry standing on it (measured, not guessed: a grey mesh
    # plane and a grey ground at the same height must match)
    inv_pi = np.float32(1.0 / np.pi)

    def fn(P):
        n = P.shape[0]
        P = np.asarray(P, np.float32)
        up = np.zeros((n, 3), np.float32)
        up[:, 2] = 1.0
        acc = np.broadcast_to(amb[None, :].astype(np.float32),
                              (n, 3)).copy()
        # world-position pixel identity for deterministic soft shadows
        spx = np.floor(P[:, 0].astype(np.float64) * 64.0).astype(np.int64)
        spy = np.floor(P[:, 1].astype(np.float64) * 64.0).astype(np.int64)
        for li, light in enumerate(lts):
            try:
                L, rad, dist = LI.sample(light, P, st)
                if light.type == 'HEMI':
                    # BI hemis wrap: 0.5 + 0.5 * N.L, never negative
                    ndl = np.clip(0.5 + 0.5 * L[:, 2], 0.0, 1.0)
                elif light.type == 'AREA':
                    ndl = np.clip(np.asarray(
                        LI.area_inp(light, P, up), np.float32), 0.0, None)
                else:
                    ndl = np.clip(L[:, 2], 0.0, 1.0)
                face = ndl > 0.0
                if not face.any():
                    continue
                vis = LI.visibility(light, P, up, L, dist, st, bvh,
                                    sample_xy=(spx, spy, li), mask=face)
                acc += rad * (ndl * vis * inv_pi)[:, None]
            except Exception:                                   # noqa: BLE001
                continue
        return acc
    return fn


def _stash_ground_light(scene, st, bvh):
    """Hand the ground plane the scene's lighting for this frame.

    Refreshed every frame (lamps move); cleared to None first so a stale
    closure from an earlier frame can never light a later one. A scene
    with no lamps hands no callback at all -- there is nothing for the
    floor to react to, and the picture stays exactly what it always was.
    """
    world = getattr(scene, 'world', None)
    if world is None:
        return
    try:
        world._ground_light = None
        if not getattr(world, 'ground_plane', False):
            return
        if str(getattr(world, 'ground_mode', 'SOLID')) == 'OCEAN':
            return
        if float(getattr(world, 'ground_lighting', 1.0) or 0.0) <= 0.0:
            return
        from . import lights as LI
        lts = LI.select_lights(getattr(scene, 'lights', []) or [], st)
        if not lts:
            return
        world._ground_light = _make_ground_light(scene, st, bvh, lts)
    except Exception:                                           # noqa: BLE001
        world._ground_light = None


def world_color(scene, settings, dirs, textures, n=None, eye=None):
    """Background radiance along `dirs` (N,3)."""
    world = scene.world
    n = dirs.shape[0] if n is None else n
    if world is None:
        return np.zeros((n, 3), np.float32)

    # An explicit sky mode wins over the node tree. Blender worlds always have a
    # node tree, so without this the Halcyon sky settings could never take
    # effect -- which is exactly what "the sky doesn't work" looked like.
    from . import sky as SKY
    time = getattr(scene, 'time', 0.0)

    def _ground(col):
        # applied after whichever sky mode ran, node tree included: an infinite
        # floor has nothing to do with how the sky above it is coloured
        if getattr(world, 'ground_plane', False) and eye is not None:
            return SKY.ground_plane(world, dirs, col.astype(np.float32), eye,
                                    time, textures)
        return col

    chosen = SKY.evaluate(world, dirs, textures, eye=eye, time=time)
    if chosen is not None:
        return chosen            # evaluate() already applied the ground

    if world.graph:
        ctx = ShadeContext(n)
        ctx.I = M.normalize(dirs)
        ctx.N = -ctx.I
        ctx.P = M.normalize(dirs) * 1e6
        ctx.generated = M.normalize(dirs)
        ctx.settings = settings
        ev = GraphEvaluator(world.graph, ctx, textures, {})
        cl, _ = ev.evaluate_surface()
        if isinstance(cl, Closure) and cl.items:
            acc = np.zeros((n, 3), np.float32)
            for kind, w, p in cl.items:
                col = to_color(p.get('color'), n)[:, :3]
                st = p.get('strength')
                s = to_value(st, n)[:, None] if st is not None else 1.0
                wv = np.asarray(w, np.float32).reshape(-1)
                if wv.shape[0] != n:
                    wv = np.broadcast_to(wv, (n,))
                acc += col * s * wv[:, None]
            return _ground(acc)
    if world.env_image is not None:
        tex = (textures.get(getattr(world.env_image, 'name', None)) or
               textures.get('world_env'))
        if tex is not None:
            from .texture import env_equirect_uv, env_sphere_uv
            d = M.normalize(dirs)
            u, v = (env_sphere_uv(d) if world.env_mapping == 'MIRRORBALL'
                    else env_equirect_uv(d))
            return _ground(tex.sample(u, v, filt='BILINEAR',
                                      wrap='EXTEND')[:, :3])
    if world.sky_blend:
        d = M.normalize(dirs)
        t = np.clip(d[:, 2] * 0.5 + 0.5, 0.0, 1.0)[:, None]
        hor = np.asarray(world.horizon, np.float32)[None, :]
        zen = np.asarray(world.zenith, np.float32)[None, :]
        return _ground((hor + (zen - hor) * t).astype(np.float32))
    return _ground(np.broadcast_to(np.asarray(world.color, np.float32)[None, :],
                                   (n, 3)).copy())


# ------------------------------------------------------------------- fog


def apply_fog(rgb, depth, settings, scene, vertex_rate=False, P=None,
              ctx=None, surf=None):
    """Distance fog. `fog_vertex` evaluates it per vertex and interpolates,
    which is how fixed-function hardware did it -- and it shows, because the
    fog band follows the tessellation rather than the surface.

    `fog_height` scales the fog by world height: full below fog_height_top,
    thinning exponentially above it -- the layered ground mist the
    sixth-generation consoles drew (fog volumes on the GameCube, VU-computed
    height fog on the PS2). Needs `P`; without it the fog stays pure
    distance fog.

    R251: the arithmetic lives in core/fog.py. A frame with no new dial
    (every 1.89.0 field set) runs `fog.legacy`, the 1.89.0 body moved
    verbatim -- bitwise by construction; a period dial (the GTE cue, a
    hardware table, z-buffer depth) takes `fog.apply`, whose GLSL twin
    `hal_fog` mirrors it statement for statement. `vertex_rate` is kept
    in the signature for callers and read by nothing (it was dead in
    1.89.0 too: called nowhere)."""
    if not settings.fog:
        return rgb
    from . import fog as _FOG
    if _FOG.extended(settings, surf):
        return _FOG.apply(rgb, depth, settings, scene, P, ctx, surf)
    return _FOG.legacy(rgb, depth, settings, scene, P)


def fog_for_points(job, tri_idx, bary, rgb, px=None, py=None, surf=None,
                   spot_fog=None):
    """apply_fog for externally-shaded points, exactly as shade_batch fogs.

    Fog is SEPARABLE: a lerp toward the fog colour by a factor of geometry
    alone (view depth and world height), independent of shading. So the
    deferred pass shades on the driver, and the CPU's OWN apply_fog runs
    on the readback with the same P and the same ctx.depth formula --
    the two devices agree by construction instead of by a GLSL twin of
    four fog modes, a quantised vertex emulation and a height layer.

    R251: most fogged materials now fog INSIDE the deferred pass
    (`hal_fog`, gpu/material.FOG_GLSL); this road serves the materials
    the planner named `fog_cpu` (traced / env composites that land after
    the readback), the reflection hits and the transparent layers.
    `px, py` are the screen pixels where the points ARE pixels (None for
    hits and corners: the Voodoo dither then adds 0, the GC range adjust
    takes k = 1, a backdrop target takes the world along the ray);
    `surf` the per-material fog dials (F006: burn-through, bias, bank --
    `_fog_readback` builds them from `binds['fog_mat']`); `spot_fog` the
    screen-spot lobe (F015).

    The caller decides WHICH points: pixel-rate surfaces fog here;
    vertex-rate materials must NOT (their fog is already inside the
    CPU-lit corner values, at the corner rate, exactly as the era's
    hardware fogged per vertex).
    """
    st = job.settings
    if not getattr(st, 'fog', False) or tri_idx.size == 0:
        return rgb
    P = job.attributes(tri_idx, bary, None)[0]
    # the sequential float32 chain shade_batch's ctx.depth carries (R251
    # A1: the GLSL twin computes exactly these three products and two sums)
    dP = np.asarray(P, np.float32) - np.asarray(job.eye, np.float32)[None, :]
    r = np.asarray(job.view, np.float32)[2, :3]
    dz = dP[:, 0] * r[0]
    dz += dP[:, 1] * r[1]
    dz += dP[:, 2] * r[2]
    depth = np.abs(dz).astype(np.float32)
    from types import SimpleNamespace
    eye = np.asarray(job.eye, np.float32)
    if spot_fog is None and px is not None and \
            float(getattr(st, 'fog_spot', 0.0) or 0.0) > 0.0:
        # R251 LIGHT-A2 (F015's fog half, A52): the refusal road's points
        # ARE pixels -- the screen spotlights' lobe is recomputed here,
        # exactly light_surface's per-batch sum
        from . import fog as _FOGS
        spot_fog = _FOGS.spot_fog_of(job, px, py, depth)
    ctx = SimpleNamespace(
        tri=tri_idx, bary=bary, px=px, py=py, spx=px, spy=py,
        scene=job.scene, camera_pos=eye, view_matrix=job.view,
        settings=st, P=P, depth=depth,
        I=M.normalize(np.asarray(P, np.float32) - eye[None, :]),
        is_camera_ray=(px is not None), width=job.width, height=job.height,
        backdrop=getattr(job, 'backdrop', None), spot_fog=spot_fog,
        # R251 LIGHT-A2 (F008): the world's textures for the direction
        # form, and the job itself so a missing backdrop is built once
        textures=getattr(job, 'textures', None), job=job)
    return apply_fog(np.asarray(rgb, np.float32), depth, st, job.scene,
                     P=P, ctx=ctx, surf=surf)


# --------------------------------------------------------- fragment shading


class ShadeJob:
    """Everything needed to shade an arbitrary set of surface points."""

    def __init__(self, scene, settings, textures, bvh, view, eye, width, height):
        self.scene = scene
        self.settings = settings
        self.textures = textures
        self.bvh = bvh
        self.view = view
        self.eye = eye
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(settings.seed)
        self.lights = LI.select_lights(scene.lights, settings)
        self.unsupported = set()
        self._obj_matrices = None
        self._bounds = None
        self._obj_bounds = None
        #: R253: the per-object OBJECT-space texture space (lo, span)
        #: Generated coordinates span; see object_generated_frame()
        self._obj_gen = None
        self._obj_gen_identity = True
        #: (mat_index, bump-node id) -> (gx, gy) full-frame gradient grids,
        #: filled by _shade_all for materials whose chunking would otherwise
        #: cut n_bump's screen gradients mid-material
        self.bump_fields = {}
        #: mat_index -> (ScatterTree, params) filled by _sss_prepare;
        #: sss_prepass marks the point-collection render (spec masked
        #: out of combined, no tree sampling -- sample_sss returns 0
        #: in 2.79's preprocess too)
        self.sss_trees = {}
        self.sss_prepass = False
        #: R253: the light-split sink -- pass name -> float32 (height,
        #: width, 3) at render resolution, allocated by render() for
        #: exactly the wanted light passes (None otherwise), written by
        #: shade_batch at camera fragments while `pass_sink_armed` is set
        #: (the opaque frame pass only: a transparent layer over a pixel
        #: must not overwrite the surface the pixel's Combined shows)
        self.pass_sink = None
        self.pass_sink_armed = False

    def object_bounds(self):
        """Per-object bounding boxes, for Generated texture coordinates.

        Blender normalises Generated coordinates over each object's own bounding
        box. Normalising over the whole scene instead makes every procedural
        texture on a normal-sized object sample a tiny patch of its own space
        and come out flat -- which is exactly what a big ground plane in the
        scene used to do to everything else.
        """
        if self._obj_bounds is not None:
            return self._obj_bounds
        mesh = self.scene.mesh
        n_obj = max(len(self.scene.objects), 1)
        lo = np.zeros((n_obj, 3), np.float32)
        hi = np.ones((n_obj, 3), np.float32)
        if mesh is not None and mesh.verts is not None and mesh.verts.size:
            if mesh.obj_index is not None and mesh.tris is not None:
                # a vertex belongs to whichever object owns its triangles
                vert_obj = np.zeros(mesh.verts.shape[0], np.int32)
                vert_obj[mesh.tris.reshape(-1)] = np.repeat(mesh.obj_index, 3)
                for i in range(n_obj):
                    sel = vert_obj == i
                    if sel.any():
                        lo[i] = mesh.verts[sel].min(0)
                        hi[i] = mesh.verts[sel].max(0)
            else:
                lo[:] = mesh.verts.min(0)
                hi[:] = mesh.verts.max(0)
        self._obj_bounds = (lo, np.maximum(hi - lo, 1e-6))
        return self._obj_bounds

    def object_generated_frame(self):
        """R253: the box Generated coordinates are measured in, per
        object, in the object's OWN space: (lo, span) float32 (n_obj, 3).

        Blender's Generated output is the mesh's texture space -- the
        object-space bounding box (auto texspace) or the manual Texture
        Space -- so it never changes with the object transform.
        object_bounds() above measures the WORLD box, which rotation,
        scale and any animated motion re-shape: that is the scroll the
        user saw on a moving object. Per object: the exported
        ObjectInfo.gen_bounds when the export supplied one; else, at an
        identity matrix, the world rows of object_bounds() (local ==
        world there, so old scenes are bitwise); else that object's
        world vertices taken back through its inverse matrix (the same
        chain the fragments use) and boxed. The span keeps Halcyon's
        max(hi - lo, 1e-6) clamp, NOT Blender's size-1 rule for a flat
        axis, so a flat ground plane with a 3-D Noise renders as before.
        Cached, and prewarm() builds it before any worker starts.
        """
        if self._obj_gen is not None:
            return self._obj_gen
        lo_w, span_w = self.object_bounds()
        n_obj = lo_w.shape[0]
        # the world rows verbatim (the same bits, not lo + span - lo)
        # for every object that keeps them: identity, no mesh, no verts
        lo = lo_w.copy()
        span = span_w.copy()
        objs = self.scene.objects or []
        mesh = self.scene.mesh
        inv = self.object_matrices()
        vert_obj = None
        eye = np.eye(4, dtype=np.float32)
        for i in range(min(n_obj, len(objs))):
            o = objs[i]
            gb = getattr(o, 'gen_bounds', None)
            if gb is not None:
                g_lo = np.asarray(gb[0], np.float32).reshape(3)
                g_hi = np.asarray(gb[1], np.float32).reshape(3)
                lo[i] = g_lo
                span[i] = np.maximum(g_hi - g_lo, 1e-6)
                continue
            mw = getattr(o, 'matrix_world', None)
            if mw is None or np.array_equal(np.asarray(mw, np.float32), eye):
                continue                     # identity: local == world
            if mesh is None or mesh.verts is None or not mesh.verts.size:
                continue
            if vert_obj is None:
                vert_obj = np.zeros(mesh.verts.shape[0], np.int32)
                if mesh.obj_index is not None and mesh.tris is not None:
                    # a vertex belongs to whichever object owns its
                    # triangles (object_bounds' rule)
                    vert_obj[mesh.tris.reshape(-1)] = np.repeat(mesh.obj_index, 3)
            sel = vert_obj == i
            if not sel.any():
                continue
            local = M.object_space_points(mesh.verts[sel],
                                          inv[min(i, inv.shape[0] - 1)])
            g_lo = local.min(0).astype(np.float32)
            g_hi = local.max(0).astype(np.float32)
            lo[i] = g_lo
            span[i] = np.maximum(g_hi - g_lo, 1e-6)
        self._obj_gen = (lo.astype(np.float32), span.astype(np.float32))
        # every inverse the identity (an untransformed scene, every
        # hand-built one): the chain would return P bit for bit, so the
        # fragments skip it
        self._obj_gen_identity = bool(np.all(inv == eye[None]))
        return self._obj_gen

    def _object_space(self, P, obj_idx):
        """R253: the world points P (n,3) back into their own objects'
        frames, one object at a time through its single (4,4) inverse --
        the same elementwise float32 chain as a per-fragment gather
        (bitwise) with no (n,4,4) temporary (a 262144-fragment chunk's
        would be 16 MB per worker). P itself, untouched, when every
        matrix is the identity."""
        self.object_generated_frame()
        if self._obj_gen_identity:
            return P
        inv = self.object_matrices()
        oi = np.clip(np.asarray(obj_idx, np.int64), 0, inv.shape[0] - 1)
        ids = np.unique(oi)
        if ids.size == 1:
            return M.object_space_points(P, inv[int(ids[0])])
        Po = np.empty((P.shape[0], 3), np.float32)
        for i in ids:
            sel = oi == i
            Po[sel] = M.object_space_points(P[sel], inv[int(i)])
        return Po

    def object_matrices(self):
        """Per-object inverse world matrices (n_obj, 4, 4), for object-space
        texture coordinates -- built once, shared by the CPU contexts and
        baked into the GPU material (R243)."""
        if self._obj_matrices is None:
            objs = self.scene.objects
            if objs:
                mats = []
                for o in objs:
                    m = o.matrix_world
                    mats.append(np.linalg.inv(np.asarray(m, np.float32))
                                if m is not None else np.eye(4, dtype=np.float32))
                self._obj_matrices = np.stack(mats)
            else:
                self._obj_matrices = np.eye(4, dtype=np.float32)[None]
        return self._obj_matrices

    def prewarm(self):
        """Build the lazily-cached tables before any worker thread starts.

        Populating them from several threads at once is a benign race in CPython
        but wastes the work; doing it up front also keeps the workers pure
        readers of shared state, which is what keeps the split safe.
        """
        mesh = self.scene.mesh
        if mesh is not None and mesh.verts is not None and mesh.verts.size:
            if self._bounds is None:
                self._bounds = (mesh.verts.min(0), mesh.verts.max(0))
        self.object_bounds()
        if self._obj_matrices is None and self.scene.objects:
            mats = []
            for o in self.scene.objects:
                m = o.matrix_world
                mats.append(np.linalg.inv(np.asarray(m, np.float32))
                            if m is not None else np.eye(4, dtype=np.float32))
            self._obj_matrices = np.stack(mats)
        # R253: the object-space Generated frame (reads the matrices)
        self.object_generated_frame()

    # ..................................................... attribute fetch
    def attributes(self, tri_idx, bary, bary_lin=None, need=None):
        """Interpolated surface attributes; `need` (a set of 'P', 'N',
        'Ng', 'uv', 'uv2', 'vcol') skips what a caller will not read --
        the punch-through alpha pass (R212) interpolates two fields
        instead of six. None = everything, exactly as before."""
        mesh = self.scene.mesh
        tris = mesh.tris
        want = (lambda k: need is None or k in need)
        P = raster.fetch(mesh.verts, tris, tri_idx, bary) \
            if want('P') else None
        Ns = Ng = None
        if want('N') or want('Ng'):
            smooth = mesh.smooth[tri_idx] if mesh.smooth is not None \
                else None
            if mesh.normals is not None:
                Ns = raster.fetch(mesh.normals, tris, tri_idx, bary)
            else:
                Ns = mesh.face_normals[tri_idx]
            Ng = mesh.face_normals[tri_idx] \
                if mesh.face_normals is not None else Ns
            if smooth is not None:
                Ns = np.where(smooth[:, None], Ns, Ng)
            Ns = M.normalize(Ns)
            Ng = M.normalize(Ng)
        ub = bary_lin if (bary_lin is not None and not self.settings.tex_perspective) \
            else bary
        uv = None
        if want('uv'):
            uv = raster.fetch(mesh.uvs, tris, tri_idx, ub) \
                if mesh.uvs is not None \
                else np.zeros((tri_idx.size, 2), np.float32)
        uv2 = None
        if want('uv2'):
            uv2 = raster.fetch(mesh.uvs2, tris, tri_idx, ub) \
                if getattr(mesh, 'uvs2', None) is not None \
                else (uv if uv is not None
                      else np.zeros((tri_idx.size, 2), np.float32))
        col = None
        if want('vcol'):
            col = raster.fetch(mesh.colors, tris, tri_idx, bary) \
                if mesh.colors is not None \
                else np.ones((tri_idx.size, 4), np.float32)
        return P, Ns, Ng, uv, uv2, col

    def _screen_projection(self):
        """(sx, sy, ws) per VERTEX: the projected screen positions and the
        clip w, cached per job (the accumulation jitter is a pure
        translation, which gradients cannot see). `uv_screen_gradients`
        and C079's `tri_lod` read the same projection."""
        cache = getattr(self, '_sgrad_cache', None)
        if cache is None:
            mesh = self.scene.mesh
            _v, _p, vp, _e = camera_matrices(self.scene.camera, self.width,
                                             self.height)
            verts = np.asarray(mesh.verts, np.float32)
            clip = np.concatenate(
                [verts, np.ones((verts.shape[0], 1), np.float32)],
                axis=1) @ vp.T
            w = clip[:, 3]
            ws = np.where(np.abs(w) < 1e-9, np.float32(1e-9), w)
            sx = (clip[:, 0] / ws * 0.5 + 0.5) * self.width
            sy = (clip[:, 1] / ws * 0.5 + 0.5) * self.height
            cache = (sx.astype(np.float32), sy.astype(np.float32),
                     ws.astype(np.float32))
            self._sgrad_cache = cache
        return cache

    def tri_lod(self, w, h, bias=0.0):
        """One mip level per TRIANGLE for a (w, h) texture: 0.5 * log2(texel
        area / screen area) + bias, float64 once per frame -> float32,
        cached per (w, h, bias). Both devices read this table (R251 C079:
        the Riva 128 / Verite per-polygon level, the Voodoo's lodbase)."""
        key = (int(w), int(h), float(bias))
        cache = getattr(self, '_tri_lod_cache', None)
        if cache is None:
            cache = self._tri_lod_cache = {}
        hit = cache.get(key)
        if hit is not None:
            return hit
        mesh = self.scene.mesh
        sx, sy, _ws = self._screen_projection()
        tris = mesh.tris
        px_ = sx[tris].astype(np.float64)
        py_ = sy[tris].astype(np.float64)                                      # (N,3)
        a_pix = 0.5 * np.abs((px_[:, 1] - px_[:, 0]) * (py_[:, 2] - py_[:, 0])
                             - (py_[:, 1] - py_[:, 0]) * (px_[:, 2] - px_[:, 0]))
        uvt = (mesh.uvs[tris] if mesh.uvs is not None
               else np.zeros(tris.shape + (2,), np.float32)).astype(np.float64)   # (N,3,2)
        a_tex = 0.5 * np.abs((uvt[:, 1, 0] - uvt[:, 0, 0]) * (uvt[:, 2, 1] - uvt[:, 0, 1])
                             - (uvt[:, 1, 1] - uvt[:, 0, 1]) * (uvt[:, 2, 0] - uvt[:, 0, 0])) \
            * float(w) * float(h)
        lod = 0.5 * np.log2(np.maximum(a_tex, 1e-12) / np.maximum(a_pix, 1e-6)) + float(bias)
        out = lod.astype(np.float32)
        cache[key] = out
        return out

    def uv_screen_gradients(self, tri_idx, bary, uv):
        """Analytic per-pixel screen derivatives of the interpolated UV.

        (du/dx, du/dy) and (dv/dx, dv/dy) in UV units per output pixel --
        what a hardware rasteriser reads off its 2x2 quads, computed
        exactly instead: for perspective-correct interpolation over a
        triangle with screen-affine barycentrics L_i (constant gradients
        g_i) and clip w_i per vertex,

            grad(uv) = W * sum_i (uv_i - uv) * g_i / w_i,   W = sum(bary*w)

        Analytic beats quad differences here for three reasons the
        engine already cares about: no seams at triangle edges, a pure
        function of (tri, bary) so chunking and threading cannot change
        it, and it works for A-buffer fragments the same as for opaque
        pixels. Ray hits have no screen footprint and never call this.
        The projection is cached per job; the accumulation jitter is a
        pure translation, which gradients cannot see.
        """
        mesh = self.scene.mesh
        sx, sy, w = self._screen_projection()
        tris = mesh.tris[tri_idx]                              # (N,3)
        p = np.stack([np.stack([sx[tris[:, i]], sy[tris[:, i]]], 1)
                      for i in range(3)], axis=1)              # (N,3,2)
        e1 = p[:, 1] - p[:, 0]
        e2 = p[:, 2] - p[:, 0]
        D = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        D = np.where(np.abs(D) < 1e-9, np.float32(1e-9), D)
        g1 = np.stack([e2[:, 1], -e2[:, 0]], 1) / D[:, None]
        g2 = np.stack([-e1[:, 1], e1[:, 0]], 1) / D[:, None]
        g0 = -g1 - g2                                          # (N,2) each
        wi = w[tris]                                           # (N,3)
        Wp = (bary * wi).sum(axis=1)                           # (N,)
        uvi = mesh.uvs[tris] if mesh.uvs is not None \
            else np.zeros(tris.shape + (2,), np.float32)       # (N,3,2)
        du = np.zeros((tri_idx.size, 2), np.float32)
        dv = np.zeros((tri_idx.size, 2), np.float32)
        for i, g in enumerate((g0, g1, g2)):
            f = (g / wi[:, i, None]) * Wp[:, None]             # (N,2)
            du += (uvi[:, i, 0] - uv[:, 0])[:, None] * f
            dv += (uvi[:, i, 1] - uv[:, 1])[:, None] * f
        return du.astype(np.float32), dv.astype(np.float32)

    def wire_fields(self, tri_idx, bary):
        """For the Wireframe node: (distance to the nearest triangle edge
        in WORLD units, world units per output PIXEL), per fragment.

        The edge distance is exact point-to-segment-line geometry on the
        fragment's own triangle. The per-pixel scale reuses the screen
        gradient machinery (1.25.80): world position is perspective-
        correct in the same barycentrics as UV, so |dP/dx| falls out of
        the identical formula -- a pure function of (tri, bary), so
        chunks, threads and devices cannot disagree about where a wire
        sits.
        """
        mesh = self.scene.mesh
        tris = mesh.tris[tri_idx]                              # (N,3)
        corners = np.asarray(mesh.verts, np.float32)[tris]     # (N,3,3)
        P = (corners * bary[:, :, None]).sum(axis=1)           # (N,3)
        dmin = None
        for a, b in ((0, 1), (1, 2), (2, 0)):
            A = corners[:, a]
            E = corners[:, b] - A
            L2 = np.maximum((E * E).sum(1), np.float32(1e-12))
            X = np.cross(P - A, E)
            d = np.sqrt(np.maximum((X * X).sum(1), 0.0) / L2)
            dmin = d if dmin is None else np.minimum(dmin, d)
        # world-per-pixel via the cached screen projection: dP/dscreen =
        # sum_i corner_i * f_i with the SAME perspective factors the UV
        # gradients use
        self.uv_screen_gradients(tri_idx[:1], bary[:1],
                                 np.zeros((1, 2), np.float32)) \
            if getattr(self, '_sgrad_cache', None) is None else None
        sx, sy, w = self._sgrad_cache
        p = np.stack([np.stack([sx[tris[:, i]], sy[tris[:, i]]], 1)
                      for i in range(3)], axis=1)              # (N,3,2)
        e1 = p[:, 1] - p[:, 0]
        e2 = p[:, 2] - p[:, 0]
        D = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        D = np.where(np.abs(D) < 1e-9, np.float32(1e-9), D)
        g1 = np.stack([e2[:, 1], -e2[:, 0]], 1) / D[:, None]
        g2 = np.stack([-e1[:, 1], e1[:, 0]], 1) / D[:, None]
        g0 = -g1 - g2
        wi = w[tris]
        Wp = (bary * wi).sum(axis=1)
        dPdx = np.zeros((tri_idx.size, 3), np.float32)
        dPdy = np.zeros((tri_idx.size, 3), np.float32)
        for i, g in enumerate((g0, g1, g2)):
            f = (g / wi[:, i, None]) * Wp[:, None]             # (N,2)
            delta = corners[:, i] - P                          # (N,3)
            dPdx += delta * f[:, 0:1]
            dPdy += delta * f[:, 1:2]
        wpp = 0.5 * (np.sqrt((dPdx * dPdx).sum(1))
                     + np.sqrt((dPdy * dPdy).sum(1)))
        return dmin.astype(np.float32), wpp.astype(np.float32)

    def view_depth(self, P):
        """|view z| of the points P (= 1/|Q|, the GS's Q): the sequential
        float32 chain the GLSL fog twin computes -- three products, two
        sums, no matmul kernel -- bitwise the matmul on this NumPy, a
        construction on any. R251 A1 (fog) and C022 (TEX-2: the
        PlayStation 2 LOD field reads it on the same attributes() output
        the CPU's context hands the sampler)."""
        _dP = np.asarray(P, np.float32) - self.eye[None, :]
        _r = np.asarray(self.view, np.float32)[2, :3]
        _dz = _dP[:, 0] * _r[0]
        _dz += _dP[:, 1] * _r[1]
        _dz += _dP[:, 2] * _r[2]
        return np.abs(_dz).astype(np.float32)

    def context(self, tri_idx, bary, px=None, py=None, front=None, bary_lin=None,
                ray_depth=0, is_camera=True, need=None):
        mesh = self.scene.mesh
        n = tri_idx.size
        if need is not None:
            # attribute closures: I and depth derive from P, generated
            # from P and the object bounds
            need = set(need)
            if 'I' in need or 'generated' in need or 'depth' in need:
                need.add('P')
        P, Ns, Ng, uv, uv2, col = self.attributes(tri_idx, bary,
                                                  bary_lin, need=need)
        ctx = ShadeContext(n)
        ctx.P = P
        ctx.N = Ns
        ctx.Ng = Ng
        ctx.uv = uv
        ctx.uv2 = uv2
        ctx.vcol = col
        # BI stripped the vertex-colour material modes when the mesh had
        # no colour layer (convertblender.c); the ones-filled default
        # above must never read as painted white
        ctx.has_vcol = mesh.colors is not None
        # the UV Map node resolves layers BY NAME: both named sets are
        # reachable as attributes, so 'uv:<name>' answers with the
        # right layer instead of silently falling back to the active one
        names = getattr(mesh, 'uv_names', None) or ()
        if len(names) > 0 and names[0] and uv is not None:
            ctx.attributes['uv:' + names[0]] = uv
        if len(names) > 1 and names[1] and uv2 is not None:
            ctx.attributes['uv:' + names[1]] = uv2
        # R246: the colour layer by its name too (the Color Attribute
        # node reads 'col:<name>'; the GPU answers the same name)
        cname = getattr(mesh, 'color_name', None)
        if cname and col is not None:
            ctx.attributes['col:' + cname] = col
        ctx.I = M.normalize(P - self.eye[None, :]) \
            if P is not None else None
        ctx.px = px
        ctx.py = py
        ctx.width = self.width
        ctx.height = self.height
        ctx.tri = tri_idx
        ctx.camera_pos = self.eye
        ctx.settings = self.settings
        ctx.time = self.scene.time
        ctx.frame = self.scene.frame
        ctx.ray_depth = ray_depth
        ctx.is_camera_ray = is_camera
        ctx.scene = self.scene
        ctx.bvh = self.bvh
        if front is not None:
            ctx.backfacing = (~front).astype(np.float32)
        if P is not None:
            # R251 (A1): the sequential float32 chain the GLSL fog twin
            # computes; C022 (TEX-2): ONE function, read by the GPU's
            # hal_lodq field builder too
            ctx.depth = self.view_depth(P)
        else:
            ctx.depth = None
        from .texture import footprint_wanted
        if px is not None and uv is not None and footprint_wanted(self.settings):
            # the mip footprint: screen points only (a ray hit has no
            # pixel footprint and samples the top level, as the era did)
            ctx.duv, ctx.dvv = self.uv_screen_gradients(tri_idx, bary, uv)
        obj_idx = mesh.obj_index[tri_idx] if mesh.obj_index is not None else \
            np.zeros(n, np.int32)
        self._fill_object(ctx, obj_idx)
        ctx.object_index_raw = obj_idx
        if P is not None:
            lo, span = self.object_bounds()
            oi_c = np.clip(obj_idx, 0, lo.shape[0] - 1)
            if str(getattr(self.settings, 'generated_space', 'OBJECT')) == 'WORLD':
                # the pre-1.92 road, verbatim: the WORLD box, which
                # scrolls under a turning or scaling object
                ctx.generated = ((P - lo[oi_c]) / span[oi_c]).astype(np.float32)
            else:
                # R253: Blender's rule -- the point taken back into the
                # object's own frame (the chain the GPU's hal_obj_rN
                # dots run, bitwise in the simulator) and measured over
                # the mesh's OWN box, so a moving object carries its
                # procedural textures with it. At identity Po is P and
                # the frame is the world rows: the same bits as before.
                glo, gspan = self.object_generated_frame()
                Po = self._object_space(P, obj_idx)
                ctx.generated = ((Po - glo[oi_c]) / gspan[oi_c]).astype(np.float32)
            # R228: the cartoon's shape smoothing reads the same bounds
            # (the WORLD box: it measures a direction from the world
            # point to the world centre, correct as-is -- R253)
            ctx.obj_bounds = (lo, span)
        else:
            ctx.generated = None
        ctx.random = _hash1(tri_idx.astype(np.float32))
        ctx.bump_fields = getattr(self, 'bump_fields', None)
        lazy = getattr(self, 'radiosity_lazy', None)
        ctx.radiosity_field = lazy() if lazy is not None else None
        # R238: the cel field (screen shadow, depth rim), lazily too
        cel_lazy = getattr(self, 'cel_lazy', None)
        ctx.cel_field = cel_lazy() if cel_lazy is not None else None
        # the deterministic-sampling identity: the screen pixel where one
        # exists. Traced hits overwrite these with the pixel that spawned
        # their ray (ShadeJob.shade's sample_xy), so a hit's soft shadows
        # and AO draw the same streams either device would.
        ctx.spx = np.asarray(px, np.int64) if px is not None else None
        ctx.spy = np.asarray(py, np.int64) if py is not None else None
        ctx.view_matrix = self.view
        # R251 F015: the Model 3 screen spotlights' fog lobe, colE*(en*el)
        # summed by light_surface (None = no screen spot reached this
        # batch); core/fog.py adds fog_spot * spot_fog to its blend colour
        ctx.spot_fog = None
        # R251 LIGHT-A2 (F008): the backdrop fog target and the world's
        # textures for the direction form (core/fog.py reads them)
        ctx.backdrop = getattr(self, 'backdrop', None)
        ctx.textures = self.textures
        ctx.job = self
        # the Wireframe node's inputs, supplied lazily: nothing is computed
        # unless a graph actually asks (the duv/dvv lesson of 1.25.80 --
        # give the reference its inputs, and give them only when read)
        _job, _tri, _bary = self, tri_idx, bary
        ctx.wire_fields = lambda: _job.wire_fields(_tri, _bary)
        # R251 C079 (TEX-2): the per-triangle mip table, lazily too --
        # n_tex_image reads it under the footprint gate with lod_source
        # TRIANGLE; the GPU's hal_lod_WxH field is the same table
        ctx.tri_lod = lambda w, h, bias=0.0: _job.tri_lod(w, h, bias)
        return ctx

    def _fill_object(self, ctx, obj_idx):
        objs = self.scene.objects
        if not objs:
            return
        n = ctx.n
        locs = np.array([o.location for o in objs], np.float32)
        cols = np.array([o.color for o in objs], np.float32)
        idxs = np.array([o.index for o in objs], np.float32)
        rnds = np.array([o.random for o in objs], np.float32)
        oi = np.clip(obj_idx, 0, len(objs) - 1)
        ctx.object_loc = locs[oi]
        ctx.object_color = cols[oi]
        ctx.object_index = idxs[oi]
        ctx.object_random = rnds[oi]
        if self._obj_matrices is None:
            mats = []
            for o in objs:
                m = o.matrix_world
                mats.append(np.linalg.inv(np.asarray(m, np.float32))
                            if m is not None else np.eye(4, dtype=np.float32))
            self._obj_matrices = np.stack(mats)
        ctx._obj_mats = self._obj_matrices
        ctx._obj_idx = oi

    # .......................................................... the shading
    def shade(self, tri_idx, bary, px=None, py=None, front=None, bary_lin=None,
              ray_depth=0, is_camera=True, rng=None, sample_xy=None):
        """RGBA for a set of surface samples, batched by material.

        `sample_xy` is (spx, spy): the SAMPLING identity for surface
        points that have no screen pixel of their own -- a traced hit
        carries the pixel that spawned its ray, so its soft shadows and
        ambient occlusion draw the same deterministic streams the
        primary surface drew.
        """
        n = tri_idx.size
        out = np.zeros((n, 4), np.float32)
        if n == 0:
            return out
        mesh = self.scene.mesh
        mat_idx = mesh.mat_index[tri_idx] if mesh.mat_index is not None else \
            np.zeros(n, np.int32)
        spx, spy = sample_xy if sample_xy is not None else (None, None)
        for mi in np.unique(mat_idx):
            sel = np.nonzero(mat_idx == mi)[0]
            mat = self.scene.materials[int(mi)] if int(mi) < len(self.scene.materials) \
                else None
            sub = self.context(tri_idx[sel], bary[sel],
                               px[sel] if px is not None else None,
                               py[sel] if py is not None else None,
                               front[sel] if front is not None else None,
                               bary_lin[sel] if bary_lin is not None else None,
                               ray_depth, is_camera)
            if spx is not None:
                sub.spx = np.asarray(spx, np.int64)[sel]
                sub.spy = np.asarray(spy, np.int64)[sel]
            out[sel] = self.shade_batch(sub, mat, ray_depth, rng)
        return out

    def shade_batch(self, ctx, mat, ray_depth=0, rng=None):
        st = self.settings
        n = ctx.n
        cl = None
        discard = None
        disp = None
        if self.sss_prepass and getattr(ctx, 'backfacing', None) \
                is not None:
            # the pre-pass shades the BACK layer with flipped normals,
            # exactly shade_sample_sss's shade_input_flip_normals --
            # unconditionally, before the graph reads any texco
            bf = np.asarray(ctx.backfacing) > 0.5
            if np.any(bf):
                ctx.N = np.where(bf[:, None], -ctx.N, ctx.N)
                ctx.Ng = np.where(bf[:, None], -ctx.Ng, ctx.Ng)
        if mat is not None and mat.graph:
            ev = GraphEvaluator(mat.graph, ctx, self.textures, mat.programs)
            cl, disp = ev.evaluate_surface()
            self.unsupported.update(ev.unsupported)
            discard = ev.cache.get('__discard')
        if mat is not None and mat.shadeless:
            surf, model, nrm = closure_to_surface(cl, ctx, st, mat)
            rgb = surf.diffuse * surf.diffuse_level[:, None] + surf.emission
            return np.concatenate([rgb, surf.opacity[:, None]], 1).astype(np.float32)

        surf, model, nrm = closure_to_surface(cl, ctx, st, mat)
        # Gouraud/flat split (see _shade_interpolated): hardware of the
        # period interpolated the LIGHTING between vertices and still
        # sampled the TEXTURE at every pixel -- `texel x vertex colour`,
        # the MODULATE combiner. ALBEDO returns the per-pixel half,
        # LIGHT shades the lighting half over a white surface so the two
        # multiply back together with the texture at full resolution.
        rate_mode = getattr(self, 'rate_mode', None)
        if rate_mode == 'ALBEDO':
            alb = np.asarray(surf.diffuse, np.float32)
            _op = np.clip(surf.opacity, 0.0, 1.0)
            if np.any(surf.alpha_steps > 0.5):
                # R252: the polygon alpha on the machine's step count
                # (Model 3's 32 translucency levels), half up
                _as = np.maximum(surf.alpha_steps, 1.0)
                _op = np.where(surf.alpha_steps > 0.5,
                               np.floor(_op * _as + 0.5) / _as, _op)
            return np.concatenate(
                [alb, _op[:, None]], axis=1).astype(np.float32)
        if rate_mode == 'LIGHT':
            surf.diffuse = np.ones_like(np.asarray(surf.diffuse, np.float32))
        if nrm is not None:
            ctx.N = M.normalize(nrm)
        if st.normal_source == 'FACE':
            ctx.N = ctx.Ng
        if disp is not None and st.displacement_scale > 0.0:
            d = np.asarray(disp, np.float32)
            h = d if d.ndim == 1 else (d[:, 2] if d.shape[1] >= 3
                                       else d.mean(axis=1))
            bumped = bump_from_height(ctx, h, st.displacement_scale)
            if bumped is not None:
                ctx.N = bumped
                surf.tangent, surf.bitangent = M.orthonormal_basis(ctx.N)
        surf.tangent, surf.bitangent = M.orthonormal_basis(ctx.N)
        # the context knows which fragments face away; the surface never did,
        # so the backface override had nothing to key off
        if getattr(ctx, 'backfacing', None) is not None:
            surf.backfacing = np.asarray(ctx.backfacing, np.float32)
        extras = {} if (np.any(surf.bi_spectra > 0.0)
                        or np.any(surf.shadows_only > 0.5)) else None
        # R253: the light split is asked for at the opaque frame's own
        # camera fragments only -- never a ray hit (px None), a layer, a
        # vertex corner or the SSS point pass. A non-None extras dict is
        # neutral by itself: every reader keys off a surf flag or a key
        # only written when its own flag asked (want_only, want_spec_acc,
        # want_spec, want_key_lit, spec_acc / only_shadow below)
        want_split = (self.pass_sink is not None
                      and getattr(self, 'pass_sink_armed', False)
                      and ray_depth == 0
                      and bool(getattr(ctx, 'is_camera_ray', True))
                      and ctx.px is not None and not self.sss_prepass)
        if want_split:
            extras = dict(extras or {})
            extras['want_split'] = True
        if rate_mode == 'LIGHT':
            # R251 (MAT-A): a refused period item shades as its fallback
            # (DS_FIXED / FLAT) from here on; the carriers ask for the
            # specular (and the key lamp's cosine) apart
            model = CB.effective_model(model, mat, st)
            ex = CB.light_extras(model, mat)
            if ex:
                extras = dict(extras or {})
                extras.update(ex)
        # the SSS main-pass sample: shi->co was CAMERA space, so the
        # tree is queried there; nothing samples during the pre-pass
        # (2.79's sample_sss returns 0 before the tree exists)
        sss_arg = None
        if self.sss_trees and mat is not None and not self.sss_prepass:
            entry = self.sss_trees.get(int(getattr(mat, 'index', -1)))
            if entry is not None:
                tree, sparams = entry
                vm = np.asarray(self.view, np.float32)
                p_cam = ctx.P @ vm[:3, :3].T + vm[:3, 3]
                sss_arg = {'sampled': tree.sample(p_cam),
                           'texfac': sparams['texfac']}
        rgb = light_surface(surf, model, ctx, self.scene, st, self.bvh,
                            rng if rng is not None else self.rng, self.lights,
                            extras=extras, suppress_spec=self.sss_prepass,
                            sss=sss_arg)
        if want_split and extras is not None and 'split' in extras:
            # R253: the split lands in the sink at this batch's pixels
            # (the sink holds exactly the wanted names; the rest is
            # dropped here, never computed twice)
            _sp = extras['split']
            for _name, _buf in self.pass_sink.items():
                _v = _sp.get(_name)
                if _v is not None:
                    _buf[ctx.py, ctx.px] = np.asarray(_v, np.float32)
        light_alpha = None
        if rate_mode == 'LIGHT' and CB.wants_spec(model):
            # R251 (MAT-A): the carried channel, read BEFORE fog (the DS
            # index and the D3D carry are unfogged; the PS2 carry takes
            # the corner's fog transmittance from core/fog.factor)
            _fog_f = None
            if st.fog and CB.fogs_carry(model):
                from . import fog as _FOGM
                _fog_f = _FOGM.factor(ctx.depth, st, self.scene, ctx.P,
                                      ctx, surf)
            light_alpha = CB.light_alpha(model, rgb, extras, fog=_fog_f)

        if np.any(surf.bi_mir_fresnel != 0.0):
            # Mirror > Fresnel: 2.79 scales ray_mirror by fresnel_fac
            # (view, vn, Blend, Fresnel) before any ray leaves. view is
            # eye->surface, so the argument is -N.V; a Blend above 1
            # turns the gradient the classic way (grazing reflects).
            ndv = M.dot(M.normalize(ctx.N), -M.normalize(ctx.I))
            mirf = SH.bi_fresnel_fac(-ndv, surf.bi_mir_blend,
                                     surf.bi_mir_fresnel)
            surf.reflect = (surf.reflect *
                            np.clip(mirf, 0.0, 1.0)).astype(np.float32)

        if st.raytrace and ray_depth < st.ray_depth:
            rgb = self._add_raytraced(rgb, surf, ctx, ray_depth)
        elif st.env_reflection and np.any(surf.reflect > 1e-4):
            V = -M.normalize(ctx.I)
            R = M.reflect(-V, M.normalize(ctx.N))
            env = world_color(self.scene, st, R, self.textures, n,
                              eye=self.eye)
            rgb = rgb + env * surf.reflect[:, None] * surf.specular * \
                surf.reflect_color

        if np.any(surf.use_mist < 0.5):
            # Options > Use Mist off: the material ignores the fog
            fogged = apply_fog(rgb.copy(), ctx.depth, st, self.scene,
                               P=ctx.P, ctx=ctx, surf=surf)
            rgb = np.where((surf.use_mist > 0.5)[:, None], fogged, rgb)
        else:
            rgb = apply_fog(rgb, ctx.depth, st, self.scene, P=ctx.P,
                            ctx=ctx, surf=surf)
        alpha = np.clip(surf.opacity, 0.0, 1.0)
        if np.any(surf.bi_transp_fresnel != 0.0):
            # Transparency > Fresnel REPLACES the alpha slider outright,
            # exactly 2.79's shade_lamp_loop
            ndv = M.dot(M.normalize(ctx.N), -M.normalize(ctx.I))
            fr = SH.bi_fresnel_fac(-ndv, surf.bi_transp_blend,
                                   surf.bi_transp_fresnel)
            alpha = np.where(surf.bi_transp_fresnel != 0.0,
                             np.clip(fr, 0.0, 1.0), alpha)
        if extras is not None and np.any(surf.bi_spectra > 0.0) and \
                'spec_acc' in extras:
            # Transparency > Specular: highlights turn opaque --
            # t = spectra * max(spec), alpha = (1-t)*alpha + t
            t = np.clip(extras['spec_acc'].max(axis=1) * surf.bi_spectra,
                        0.0, 1.0)
            alpha = (1.0 - t) * alpha + t
        if st.alpha_threshold > 0.0:
            # a hard cutoff rather than a blend: cheaper, and what hardware
            # without an alpha unit actually did with cut-out textures
            alpha = np.where(alpha >= st.alpha_threshold, alpha, 0.0)
        if np.any(np.abs(surf.edge_opacity - 1.0) > 1e-4):
            facing = np.clip(np.abs(M.dot(M.normalize(ctx.N),
                                          -M.normalize(ctx.I))), 0.0, 1.0)
            t = np.power(1.0 - facing, np.maximum(surf.fresnel_power, 0.01))
            alpha = np.clip(alpha * (1.0 - t) + surf.edge_opacity * t, 0.0, 1.0)
        if np.any(surf.alpha_clip >= 0.0):
            # R211 punch-through: a CLIP material is either fully there
            # or fully absent -- the era's cut-out alpha test, and the
            # SAME hard law on every road (camera pass, blend layers,
            # ray hits), so whichever road drew the surface agrees on
            # where its holes are
            # R251 C031 (PS2 GS AFAIL / GX alpha compare): a Clip+Blend
            # material keeps its SUB-threshold alpha (the blend half,
            # drawn a second time as a layer with no depth write) and
            # forces 1.0 above it (the opaque half, promoted in the
            # z-pass exactly as CLIP); `surf.alpha_soft` carries the mode
            # PER FRAGMENT because the vertex/face roads batch across
            # materials. CLIP keeps its hard 0/1.
            cm = surf.alpha_clip >= 0.0
            alpha = np.where(
                cm,
                np.where(alpha >= np.maximum(surf.alpha_clip, 1e-6),
                         1.0, np.where(surf.alpha_soft > 0.5, alpha, 0.0)),
                alpha).astype(np.float32)
        if st.transparency == 'STIPPLE' and ctx.px is not None:
            # keep or drop each pixel outright against an ordered threshold --
            # no blending, exactly as hardware without an alpha unit managed it
            from .dither import ORDERED, threshold_map
            kind = str(getattr(st, 'stipple_pattern', 'BAYER4') or '')
            if kind == 'N64_NOISE':
                # R251 C015 (N64 RDP dither_alpha_en): the compare is
                # the RDP's integer one -- round(alpha * 255) against a
                # FRESH 8-bit random per (pixel, frame, seed), a FULL-
                # FRAME map (never tiled) built once per (H, W, frame,
                # seed) and shared by every shading chunk (a double
                # build in two threads is bitwise the same map, never a
                # lock). a8 > r8 exactly as angrylion's alpha_compare;
                # a tie DROPS. The GPU uploads this very array
                frame_n = int(getattr(self.scene, 'frame', 1) or 1)
                seed_n = int(getattr(st, 'seed', 0) or 0)
                nkey = (int(self.height), int(self.width), frame_n,
                        seed_n)
                ncache = getattr(self, '_r251_noise_maps', None)
                if ncache is None:
                    ncache = self._r251_noise_maps = {}
                tm = ncache.get(nkey)
                if tm is None:
                    tm = threshold_map('N64_NOISE', nkey[0], nkey[1],
                                       frame=frame_n, seed=seed_n)
                    ncache.clear()
                    ncache[nkey] = tm
                thr = tm[np.asarray(ctx.py), np.asarray(ctx.px)]
                a8 = np.round(np.asarray(alpha, np.float32)
                              * np.float32(255))
                alpha = np.where(a8 > thr, 1.0, 0.0)
                kind = None
            elif ORDERED.get(kind) is None:
                # R251: `stipple_pattern` is a free string, so a preset
                # dict or a script can carry an error-diffusion kind past
                # the enum, and Blender hands the exporter '' for a
                # stored number no item carries (a scene saved before
                # 1.90.0 with Screen Door at None). Bayer 4x4, by name,
                # once per job (the GPU's `_build_stipple` takes the
                # same fallback so both devices bake the same map)
                if not getattr(self, '_r251_stipple_said', False):
                    print(f'[Halcyon] transparency: stipple pattern '
                          f'{kind!r} is not an ordered map; Bayer 4x4 '
                          'used (an error-diffusion kind, or a scene '
                          'saved before 1.90.0 with Screen Door at None, '
                          'reads back as this)')
                    self._r251_stipple_said = True
                kind = 'BAYER4'
            if kind is not None:
                tm = threshold_map(kind, 64, 64)
                thr = tm[np.asarray(ctx.py) % tm.shape[0],
                         np.asarray(ctx.px) % tm.shape[1]]
                alpha = np.where(alpha > np.asarray(thr, np.float32),
                                 1.0, 0.0)
        elif st.transparency == 'NONE':
            alpha = np.ones_like(alpha)
        if extras is not None and 'only_shadow' in extras and \
                np.any(surf.shadows_only > 0.5):
            # Shadow > Shadows Only: the classic catcher. The material
            # renders black at the mean of its lamps' shadow, exactly
            # 2.79's shade_only_shadow: alpha = mat_alpha * accum/ir
            accum, ir = extras['only_shadow']
            catch = np.clip(surf.opacity, 0.0, 1.0) * \
                (accum / ir if ir > 0.0 else np.zeros_like(accum))
            only = surf.shadows_only > 0.5
            alpha = np.where(only, np.clip(catch, 0.0, 1.0), alpha)
            rgb = np.where(only[:, None], 0.0, rgb).astype(np.float32)
        if np.any(surf.cast_only > 0.5) and \
                bool(getattr(ctx, 'is_camera_ray', True)):
            # Shadow > Cast Only: invisible to the CAMERA, while shadow
            # maps and secondary rays keep seeing the geometry
            alpha = np.where(surf.cast_only > 0.5, 0.0, alpha)
        if discard is not None:
            alpha = np.where(np.asarray(discard).reshape(-1)[:alpha.size], 0.0, alpha)
        # R164: MA_OBCOLOR modulates the WHOLE combined at the very
        # end of shade_lamp_loop (after emit and spec joined), alpha
        # under transparency -- and an SSS material with its tree
        # built SKIPS the modulation, the C's own sss_pass_done gate
        bi_x = getattr(surf, 'bi', None)
        # sss_pass_done quirk kept: an SSS-flagged material skips the
        # modulation once its tree exists AND when the scene switch is
        # off (done = baking || !R_SSS || tree) -- but never in its
        # own pre-pass, where the tree is not built yet
        _skip_obcol = (bi_sss_params(mat) is not None) and (
            sss_arg is not None
            or (not getattr(self.settings, 'sss', True)
                and not self.sss_prepass))
        if bi_x is not None and getattr(bi_x, 'use_obcolor', False) \
                and not _skip_obcol:
            oi = getattr(ctx, 'object_index_raw', None)
            if oi is not None and getattr(self.scene, 'objects', None):
                cols = np.array([tuple(getattr(o, 'color',
                                               (1, 1, 1, 1)))[:4]
                                 for o in self.scene.objects],
                                np.float32)
                oc = cols[np.clip(np.asarray(oi, np.int64), 0,
                                  len(cols) - 1)]
                oc[:, 3] = np.clip(oc[:, 3], 0.0, 1.0)
                rgb = rgb * oc[:, :3]
                if bool(getattr(mat, 'graph', None)) and any(
                        nd.get('props', {}).get('use_transparency')
                        for nd in mat.graph.get('nodes', {}).values()
                        if nd.get('bl_idname')
                        == 'HALCYON_BIMaterialNode'):
                    alpha = alpha * oc[:, 3]
        if rate_mode == 'LIGHT':
            # R251 (MAT-A): the corner quantiser (the lit colour saturated
            # and quantised to the machine's depth, AFTER fog, shadows-only
            # and object colour) and the carried channel in the alpha
            rgb_q = CB.corner_rgb(model, rgb)
            if rgb_q is not None:
                rgb = rgb_q
            if light_alpha is not None:
                alpha = light_alpha
        # R251 C126 (Blender 2.4x Env): an Env-hole material shows the
        # WORLD along the view ray (the same evaluator the background
        # uses, ground plane included) at alpha 0 -- opaque for the
        # z-test, nothing behind it composited; after every alpha law
        # so it holds in every transparency mode, and at ray hits too
        # (a mirror sees the sky through the hole, as 2.4x's Env did).
        # Per FRAGMENT through the material index (the vertex/face
        # roads batch across materials)
        _hole_mats = getattr(self, '_r251_env_hole_mats', None)
        if _hole_mats is None:
            _hole_mats = self._r251_env_hole_mats = np.array(
                [str(getattr(m, 'blend_mode', 'INHERIT')) == 'ENV_HOLE'
                 for m in (self.scene.materials or [])] or [False], bool)
        if bool(_hole_mats.any()):
            _tri_h = getattr(ctx, 'tri', None)
            _mi_h = getattr(self.scene.mesh, 'mat_index', None)
            if _tri_h is not None and _mi_h is not None:
                hole = _hole_mats[np.clip(_mi_h[np.asarray(_tri_h)], 0,
                                          _hole_mats.size - 1)]
            elif mat is not None:
                hole = np.full(alpha.shape[0], str(getattr(
                    mat, 'blend_mode', 'INHERIT')) == 'ENV_HOLE', bool)
            else:
                hole = None
            if hole is not None and bool(hole.any()):
                wc = world_color(self.scene, st,
                                 M.normalize(np.asarray(ctx.I)[hole]),
                                 self.textures, int(hole.sum()),
                                 eye=self.eye)
                rgb = np.asarray(rgb, np.float32).copy()
                rgb[hole] = np.asarray(wc, np.float32)
                alpha = np.where(hole, 0.0, alpha)
        return np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32)

    def _add_raytraced(self, rgb, surf, ctx, ray_depth):
        """Secondary rays, traced only for the fragments that actually want them.

        This used to test `np.any(...)` and then trace for the whole batch: one
        transparent fragment anywhere in a chunk of a quarter of a million meant
        a refraction ray for every one of them, recursively to the ray depth. A
        scene with a single sheet of glass paid for refracting the entire frame.
        """
        st = self.settings
        if self.bvh is None:
            return rgb
        N = M.normalize(ctx.N)
        V = -M.normalize(ctx.I)
        spx = getattr(ctx, 'spx', None)
        spy = getattr(ctx, 'spy', None)

        def _sxy(sel):
            if spx is None or spy is None:
                return None
            return (np.asarray(spx, np.int64)[sel],
                    np.asarray(spy, np.int64)[sel])

        if st.ray_reflection:
            want = np.nonzero(surf.reflect > 1e-4)[0]
            if want.size:
                R = M.reflect(-V[want], N[want])
                blur = float(getattr(st, 'reflection_blur', 0.0))
                bsamples = max(int(getattr(st, 'reflection_blur_samples',
                                           1)), 1)
                if blur > 1e-3 and bsamples >= 1:
                    # blurry reflections: average jittered rays in a cone
                    # around the mirror direction -- LightWave's
                    # Reflection Blurring, MAX's raytrace blur. The
                    # jitter draws from the deterministic streams (own
                    # salt), so the picture is batch- and thread-
                    # invariant like every other sampled effect here.
                    hit = self._blurred_reflection(
                        ctx.P[want] + N[want] * st.ray_bias, R, N[want],
                        blur, bsamples, ray_depth, _sxy(want))
                else:
                    hit = self.trace(ctx.P[want] + N[want] * st.ray_bias,
                                     R, ray_depth + 1, sample_xy=_sxy(want))
                rgb[want] += (hit * surf.reflect[want, None] *
                              surf.specular[want] * surf.reflect_color[want])

        if st.ray_refraction:
            want = np.nonzero(surf.opacity < 0.999)[0]
            if want.size:
                Nw, Vw = N[want], V[want]
                # the RAY IOR: a master shader's one slider, or the BI
                # node's own transparency IOR (Blinn's Refr is spectral,
                # not refractive, on that node)
                ior = np.maximum(surf.ray_ior[want], 1e-3)
                eta = np.where(M.dot(Nw, Vw) < 0, ior, 1.0 / ior)
                T = M.refract(-Vw, Nw, eta)
                bad = (T * T).sum(1) < 1e-9
                T = np.where(bad[:, None], M.reflect(-Vw, Nw), T)
                hit = self.trace(ctx.P[want] - Nw * st.ray_bias, T,
                                 ray_depth + 1, sample_xy=_sxy(want))
                k = ((1.0 - np.clip(surf.opacity[want], 0.0, 1.0)) *
                     np.clip(surf.refraction[want], 0.0, 1.0))[:, None]
                # BI's Filter slider: 0 passes light untinted, 1 tints
                # by the material colour (the master's old fixed look)
                f = np.clip(surf.bi_ray_filter[want], 0.0, 1.0)[:, None]
                tint = 1.0 + f * (surf.diffuse[want] - 1.0)
                rgb[want] = rgb[want] * (1.0 - k) + hit * k * tint
        return rgb

    #: reflection blur's hash-stream salt -- distinct from lights (131*li),
    #: AO (8389), the AO node (6151) and the radiosity gather (9973)
    BLUR_SALT = 10009

    def _blurred_reflection(self, origin, R, N, blur_deg, samples,
                            ray_depth, sxy):
        """Average `samples` rays jittered in a cone of `blur_deg` around
        the mirror direction R. Uniform disk in the tangent plane of R,
        radius tan(half-angle): the standard cone jitter of the era's
        blurry-reflection implementations. Rays bent below the surface
        are folded back to the mirror direction rather than traced into
        the object."""
        from . import patterns as PT
        n = origin.shape[0]
        half = np.float32(np.tan(np.radians(max(blur_deg, 0.0)) * 0.5))
        t, b = M.orthonormal_basis(R)
        seed = int(getattr(self.settings, 'seed', 0) or 0)
        acc = np.zeros((n, 3), np.float32)
        for k in range(max(samples, 1)):
            if sxy is not None:
                z = 2 * k + self.BLUR_SALT + 7919 * seed
                u1 = PT.sample_u(sxy[0], sxy[1], z)
                ca, sa = PT.sample_circle(PT.sample_u(sxy[0], sxy[1],
                                                      z + 1))
            else:
                u1 = self.rng.random(n).astype(np.float32)
                th = (2.0 * np.pi * self.rng.random(n)).astype(np.float32)
                ca, sa = np.cos(th), np.sin(th)
            r = np.sqrt(u1) * half
            d = M.normalize(R + t * (r * ca)[:, None] + b * (r * sa)[:, None])
            below = M.dot(d, N) <= 0.0
            d = np.where(below[:, None], R, d)
            acc += self.trace(origin, d, ray_depth + 1, sample_xy=sxy)
        return acc / np.float32(max(samples, 1))

    def _volume_mats(self):
        """R223: the frame's volume-container material indices, once."""
        vm = getattr(self, '_vol_mats_cache', None)
        if vm is None:
            from .volume import material_is_volume
            vm = frozenset(
                i for i, m in enumerate(self.scene.materials or [])
                if material_is_volume(m))
            self._vol_mats_cache = vm
        return vm

    def trace(self, origin, dirs, ray_depth, sample_xy=None):
        """Shade whatever a secondary ray hits (background if nothing).

        `sample_xy` carries the spawning pixels' identity, so the hit's
        deterministic sampling (soft shadows, AO) follows the ray."""
        n = origin.shape[0]
        if self.bvh is None or ray_depth > self.settings.ray_depth:
            return world_color(self.scene, self.settings, dirs, self.textures, n)
        tmax = np.full(n, 1e30, np.float32)
        tid, t, u, v = self.bvh.intersect(origin, dirs, tmax)
        # R223: traced rays pass THROUGH volume containers -- a fog box
        # in a mirror used to reflect as an empty black surface. The
        # era's mirrors never saw the volume fog either, so the ray
        # steps past the container and takes whatever stands behind it
        # (bounded, and free when the scene has no containers).
        vol_mats = self._volume_mats()
        if vol_mats and self.scene.mesh.mat_index is not None:
            org = None
            vlist = np.fromiter(vol_mats, np.int32)
            for _ in range(8):
                hitm = tid >= 0
                if not hitm.any():
                    break
                redo = hitm.copy()
                redo[hitm] = np.isin(
                    self.scene.mesh.mat_index[tid[hitm]], vlist)
                if not redo.any():
                    break
                if org is None:
                    org = np.array(origin, np.float32, copy=True)
                ri = np.nonzero(redo)[0]
                org[ri] = org[ri] + dirs[ri] * (t[ri][:, None] + 1e-4)
                tid2, t2, u2, v2 = self.bvh.intersect(
                    org[ri], dirs[ri],
                    np.full(ri.size, 1e30, np.float32))
                tid[ri], t[ri], u[ri], v[ri] = tid2, t2, u2, v2
        hit = tid >= 0
        out = np.zeros((n, 3), np.float32)
        miss = np.nonzero(~hit)[0]
        if miss.size:
            out[miss] = world_color(self.scene, self.settings, dirs[miss],
                                    self.textures, miss.size, eye=self.eye)
        if not np.any(hit):
            return out
        idx = np.nonzero(hit)[0]
        bary = np.stack([1.0 - u[idx] - v[idx], u[idx], v[idx]], axis=1).astype(np.float32)
        col = self.shade(tid[idx].astype(np.int32), bary, ray_depth=ray_depth,
                         is_camera=False,
                         sample_xy=(sample_xy[0][idx], sample_xy[1][idx])
                         if sample_xy is not None else None)
        out[idx] = col[:, :3]
        return out


def bump_from_height(ctx, height, strength=1.0):
    """Perturb the normal from a height field, using screen derivatives
    normalised to WORLD slope.

    The node graph's Displacement output was computed and thrown away. Actually
    displacing geometry means tessellating it, which no 1990s scanline renderer
    did either -- they turned the height into a normal perturbation and called
    it bump mapping, which is what this does.

    The gradient is height-per-WORLD-UNIT, not height-per-pixel: the
    raw pixel difference halves every time the resolution doubles, so
    the same material bumped twice as deep in a draft as in the refine
    (and deeper again at F12) -- the field's 'better bump' report. The
    per-pixel world footprint comes from the same neighbour grid the
    heights use, so the normalisation costs one more gather; Blender
    Internal differentiated in texture space with a fixed step and was
    resolution-independent for the same reason.
    """
    if ctx.px is None or ctx.py is None or ctx.width is None:
        return None
    h = np.asarray(height, np.float32).reshape(-1)
    if h.size != ctx.n or float(h.std()) < 1e-9:
        return None
    px = np.asarray(ctx.px, np.int64)
    py = np.asarray(ctx.py, np.int64)
    w, hh = int(ctx.width), int(ctx.height)
    grid = np.zeros((hh, w), np.float32)
    seen = np.zeros((hh, w), bool)
    grid[py, px] = h
    seen[py, px] = True
    pgrid = np.zeros((hh, w, 3), np.float32)
    pgrid[py, px] = np.asarray(ctx.P, np.float32)
    # one-sided differences where the neighbour is missing, so silhouettes do
    # not invent a gradient out of empty space
    right = np.zeros_like(grid)
    right[:, :-1] = grid[:, 1:]
    ok_r = np.zeros_like(seen)
    ok_r[:, :-1] = seen[:, 1:]
    up = np.zeros_like(grid)
    up[:-1, :] = grid[1:, :]
    ok_u = np.zeros_like(seen)
    ok_u[:-1, :] = seen[1:, :]
    pright = np.zeros_like(pgrid)
    pright[:, :-1] = pgrid[:, 1:]
    pup = np.zeros_like(pgrid)
    pup[:-1, :] = pgrid[1:, :]
    # the world size of one pixel step, from the SAME neighbours the
    # height differences use; a missing neighbour keeps step 0 and the
    # gradient stays 0 there exactly as before
    step_r = np.linalg.norm(np.where(ok_r[:, :, None],
                                     pright - pgrid, 0.0),
                            axis=2)[py, px]
    step_u = np.linalg.norm(np.where(ok_u[:, :, None],
                                     pup - pgrid, 0.0),
                            axis=2)[py, px]
    dx = np.where(ok_r, right - grid, 0.0)[py, px] \
        / np.maximum(step_r, 1e-6)
    dy = np.where(ok_u, up - grid, 0.0)[py, px] \
        / np.maximum(step_u, 1e-6)
    dx = np.where(step_r > 1e-6, dx, 0.0)
    dy = np.where(step_u > 1e-6, dy, 0.0)

    N = M.normalize(ctx.N)
    t, b = M.orthonormal_basis(N)
    # 0.5: the world-slope scale that keeps a strength-1.0 bump in the
    # same visual class the old per-pixel 8.0 gave at the proven demo
    # framing (~0.06 world units per pixel there, 8 x 0.06 = 0.5) --
    # the LOOK at the golden sizes is preserved, and now it holds at
    # every other size too
    k = float(strength) * 0.5
    return M.normalize(N - t * (dx * k)[:, None] - b * (dy * k)[:, None])


def _hash1(x):
    h = np.sin(x * 12.9898) * 43758.5453
    return (h - np.floor(h)).astype(np.float32)


# ---------------------------------------------------- shading-rate dispatch


def shade_vertex_rate(job, tri_subset, rate, st=None):
    """Gouraud / flat: shade at vertices or face centres, interpolate after.

    Historically these are *shading rates*, not reflectance models -- the same
    Phong maths evaluated less often. Doing it properly is what gives the
    faceted, banded look rather than a fake approximation of it.
    """
    mesh = job.scene.mesh
    tris = mesh.tris[tri_subset]
    if rate == 'FACE':
        bary = CB.face_bary(job, tri_subset, st)
        idx = tri_subset.astype(np.int32)
        col = (_shade_chunked(job, idx, bary, None, None, None, None, st)
               if st is not None else job.shade(idx, bary))
        return col, None
    verts = np.unique(tris.reshape(-1))
    lookup = np.full(mesh.verts.shape[0], -1, np.int64)
    lookup[verts] = np.arange(verts.size)
    owner = np.zeros(verts.size, np.int32)
    corner = np.zeros(verts.size, np.int32)
    for c in range(3):
        vi = lookup[tris[:, c]]
        owner[vi] = tri_subset
        corner[vi] = c
    bary = np.zeros((verts.size, 3), np.float32)
    bary[np.arange(verts.size), corner] = 1.0
    col = (_shade_chunked(job, owner, bary, None, None, None, None, st)
           if st is not None else job.shade(owner, bary))
    return col, lookup


# --------------------------------------------------------------- main entry


def bi_node_props(mat):
    """The BI material node's props dict from a material's graph, or None."""
    graph = getattr(mat, 'graph', None) if mat is not None else None
    if not graph:
        return None
    for node in graph.get('nodes', {}).values():
        if node.get('bl_idname') == 'HALCYON_BIMaterialNode':
            return node.get('props', {})
    return None


def bi_sss_params(mat):
    """The BI node's Subsurface Scattering panel, or None when off."""
    props = bi_node_props(mat)
    if not props or not props.get('sss_enable'):
        return None
    return {'scale': float(props.get('sss_scale', 0.1) or 0.1),
            'radius': tuple(props.get('sss_radius', (1.0, 1.0, 1.0))),
            'color': tuple(props.get('sss_color', (1.0, 1.0, 1.0))),
            'ior': float(props.get('sss_ior', 1.3)),
            'error': float(props.get('sss_error', 0.05)),
            'colfac': float(props.get('sss_colfac', 1.0)),
            'texfac': float(props.get('sss_texfac', 0.0)),
            'front': float(props.get('sss_front', 1.0)),
            'back': float(props.get('sss_back', 1.0))}


def _sss_prepare(scene, st, job, view, W, H):
    """BI's SSS pre-pass, per material: sss_create_tree_mat's shape.

    2.79 re-rendered the scene once per SSS material with OSA off,
    z-buffering the material's OWN faces into a front layer (nearest)
    and a back layer (farthest), shaded each covered pixel with
    specular masked out of combined, and stored (position, colour,
    pixel area x alpha) points -- back points with NEGATED area. The
    octree over those points answers the main pass. Same here: the
    front/back layers come from two subset rasterisations (the back
    one through a flipped-z projection), the shading reuses the exact
    frame machinery in prepass mode, the pixel areas are
    shade_input_calc_viewco's own derivative construction, and
    everything lives in CAMERA space, where shi->co lived."""
    from . import sss as SSS
    mesh = scene.mesh
    if mesh is None or mesh.tris is None or not mesh.tris.size:
        return
    if not getattr(st, 'sss', True):
        return
    mats = {}
    for i, m in enumerate(scene.materials or ()):
        p = bi_sss_params(m)
        if p is not None:
            mats[i] = p
    if not mats:
        return
    from . import raster as RA
    _view, proj, vp, _eye = camera_matrices(scene.camera, W, H)
    ortho = str(getattr(scene.camera, 'type', 'PERSP')) == 'ORTHO'
    flip = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32) @ vp
    vm = np.asarray(view, np.float32)
    with ST.track('SSS preprocessing'):
        for mi, params in mats.items():
            sel = np.nonzero(mesh.mat_index == mi)[0] \
                if mesh.mat_index is not None else \
                np.arange(mesh.tris.shape[0])
            if sel.size == 0:
                continue
            gf = RA.GBuffer(W, H)
            RA.rasterize(mesh.verts, mesh.tris, vp, W, H, subset=sel,
                         gbuf=gf)
            gb = RA.GBuffer(W, H)
            RA.rasterize(mesh.verts, mesh.tris, flip, W, H, subset=sel,
                         gbuf=gb)
            yy, xx = np.mgrid[0:H, 0:W]
            covf = gf.tri >= 0
            # the tile pass skips a back sample identical to the front
            # one (a single-sided face is one point, not two)
            covb = (gb.tri >= 0) & ~(covf & (gb.tri == gf.tri))
            parts = []
            for gbuf, cov, is_back in ((gf, covf, False),
                                       (gb, covb, True)):
                if not np.any(cov):
                    continue
                tri = gbuf.tri[cov].astype(np.int64)
                bary = gbuf.bary[cov]
                px = xx[cov].astype(np.int64)
                py = yy[cov].astype(np.int64)
                tv = mesh.verts[mesh.tris[tri]]
                nrm = np.cross(tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0])
                cent = tv.mean(axis=1)
                facing = ((job.eye[None, :] - cent) * nrm).sum(axis=1) \
                    > 0.0
                job.sss_prepass = True
                try:
                    out = job.shade(tri, bary, px, py, front=facing,
                                    sample_xy=(px, py))
                finally:
                    job.sss_prepass = False
                P = (bary[:, :, None] * tv).sum(axis=1)
                P_cam = P @ vm[:3, :3].T + vm[:3, 3]
                v1_cam = tv[:, 0] @ vm[:3, :3].T + vm[:3, 3]
                n_cam = nrm @ vm[:3, :3].T
                nl = np.linalg.norm(n_cam, axis=1, keepdims=True)
                n_cam = n_cam / np.maximum(nl, 1e-20)
                area = SSS.pixel_areas(P_cam, n_cam, v1_cam,
                                       px + 0.5, py + 0.5, proj, W, H,
                                       ortho)
                area = area * np.clip(out[:, 3], 0.0, None)
                if is_back:
                    area = -area
                parts.append((P_cam.astype(np.float32),
                              out[:, :3].astype(np.float32),
                              area.astype(np.float32)))
            if not parts:
                continue
            co = np.concatenate([p[0] for p in parts])
            colr = np.concatenate([p[1] for p in parts])
            ar = np.concatenate([p[2] for p in parts])
            ss3 = SSS.settings_for(params)
            tree = SSS.ScatterTree(ss3, params['scale'],
                                   params['error'], co, colr, ar)
            job.sss_trees[mi] = (tree, params)
            # the GPU twin: the tree as a raw data texture (with the
            # world->camera rows embedded, so the pass source stays
            # camera-independent and the plan cache never needs the
            # camera). Registered among the prepared textures under a
            # reserved key -- the frame pass binds it exactly as the
            # BI noise tables bind.
            try:
                from .texture import Texture as _Tex
                packed = tree.pack_gpu(vm[:3, :4].ravel())
                job.textures[f'__sss_tree_{mi}__'] = _Tex(
                    packed, name=f'__sss_tree_{mi}__',
                    colorspace='Non-Color', filt='NEAREST',
                    wrap='EXTEND')
            except Exception:                                   # noqa: BLE001
                import traceback
                traceback.print_exc()


def collect_exclusive_lights(scene):
    """Lamp names claimed by any material's EXCLUSIVE light group.

    BI's Exclusive flag: a lamp in such a group lights ONLY materials
    naming that group. Collected once per render onto the scene, read
    by every light loop."""
    names = set()
    for mat in (scene.materials or []):
        props = bi_node_props(mat)
        if props and props.get('light_group_exclusive') and \
                props.get('light_group_lights'):
            names.update(props['light_group_lights'])
    scene.exclusive_lights = frozenset(names) if names else None


def _caster_keep_tri(scene, mesh):
    """Per-triangle caster mask, or None when everything casts.

    Three voices agree per triangle: the OBJECT's Visibility > Shadow
    toggle, the BI node's Shadow > Cast, and (R208) the plain
    Material.cast_shadow flag -- which fur-shell materials turn off so
    a pelt does not blanket its own body in cast shadow. One mask
    serves shadow maps (the casters are simply left out of the bake)
    and ray shadows (the BVH's any-hit skips the triangles by cast
    filter), so the two shadow roads finally agree about who casts.
    """
    keep_tri = None
    if mesh is not None and mesh.obj_index is not None and scene.objects:
        keep = np.array([o.cast_shadow for o in scene.objects], bool)
        if not keep.all():
            keep_tri = keep[np.clip(mesh.obj_index, 0,
                                    len(scene.objects) - 1)]
    if mesh is not None and mesh.mat_index is not None and scene.materials:
        from .volume import material_is_volume as _miv
        mkeep = []
        any_off = False
        for m in scene.materials:
            props = bi_node_props(m)
            on = bool(props.get('shadow_cast', True)) if props else True
            on = on and bool(getattr(m, 'cast_shadow', True))
            # R222: a volume container's bound never casts -- the fog
            # inside scatters light, the box around it is not geometry
            on = on and not _miv(m)
            mkeep.append(on)
            any_off = any_off or not on
        if any_off:
            mt = np.array(mkeep, bool)[np.clip(mesh.mat_index, 0,
                                               len(scene.materials) - 1)]
            keep_tri = mt if keep_tri is None else (keep_tri & mt)
    if keep_tri is not None and keep_tri.all():
        return None
    return keep_tri


def _build_shadows(scene, st, mesh):
    if not (st.shadows and st.shadow_default in ('MAP', 'PER_LIGHT')):
        return
    keep_tri = _caster_keep_tri(scene, mesh)
    cast = None
    if keep_tri is not None and not keep_tri.all():
        cast = np.nonzero(keep_tri)[0]
    LI.build_shadow_maps(scene, st, cast)


def shaft_sources(scene, settings, vp):
    """Screen positions of lights that scatter, for the light-shaft pass."""
    out = []
    for light in scene.lights:
        vol = float(getattr(light, 'volumetric', 0.0))
        if vol <= 0.0:
            continue
        if light.type == 'SUN':
            # a directional light has no position: its shafts converge on the
            # vanishing point, which is the direction projected with w = 0
            d = M.normalize(np.asarray(light.direction, np.float32))
            clip = np.append(-d, 0.0).astype(np.float32) @ vp.T
        else:
            pos = np.asarray(light.position, np.float32)
            clip = np.append(pos, 1.0).astype(np.float32) @ vp.T
        if clip[3] <= 1e-6:
            continue                       # behind the camera
        ndc = clip[:3] / clip[3]
        # a source outside the frame still throws shafts into it, so the bound
        # is generous; only rule out lights nowhere near the view
        if abs(ndc[0]) > 6.0 or abs(ndc[1]) > 6.0:
            continue
        out.append(((float(ndc[0]), float(ndc[1])), vol))
    return out


def _synth_caustic_cookies(scene):
    """Bake the animated pool-light web into each caustic lamp's cookie.

    The pattern is procedural but the DELIVERY is the ordinary cookie
    road: a small periodic tile rendered by the CPU this frame and set
    as the lamp's cookie image, so the CPU shader and the GPU's cookie
    sampler read the SAME texels -- parity by construction, no second
    GLSL implementation of the light path. A real image cookie on the
    lamp always wins; the tile is deterministic in (frame time, scale,
    speed), and seamless, so a SUN tiles it across the world without a
    join.
    """
    lights = getattr(scene, 'lights', None) or ()
    if not any(float(getattr(l, 'caustics', 0.0)) > 0.0 for l in lights):
        return
    from . import patterns as PT
    from .scene import ImageBuffer
    t = float(getattr(scene, 'time', 0.0))
    for light in lights:
        c = float(getattr(light, 'caustics', 0.0))
        kind = str(getattr(light, 'type', '')).upper()
        if c <= 0.0 or kind not in ('SPOT', 'SUN'):
            continue
        if getattr(light, 'cookie', None) is not None and \
                not getattr(light, '_caustic_cookie', False):
            continue                     # a real image cookie wins
        size = 256
        scale = max(float(getattr(light, 'caustics_scale', 4.0)), 0.05)
        speed = float(getattr(light, 'caustics_speed', 1.0))
        per = 8
        ax = (np.arange(size, dtype=np.float32) + 0.5) / size
        if kind == 'SPOT':
            # `scale` cells across the cone's image
            u = ax[None, :] * min(scale, float(per))
            v = ax[:, None] * min(scale, float(per))
        else:
            # one tile spans the full periodic lattice -- seamless
            u = ax[None, :] * per
            v = ax[:, None] * per
        uu = np.broadcast_to(u, (size, size))
        vv = np.broadcast_to(v, (size, size))
        web = PT.caustic_web(uu, vv, time=t * speed, period=per)
        # energy-neutral-ish: mean stays near 1, crests overdrive
        val = np.maximum(1.0 + (web - 0.35) * 1.6 * min(c, 4.0), 0.0)
        px = np.repeat(val[:, :, None], 3, axis=2).astype(np.float32)
        light.cookie = ImageBuffer(name=f'__caustics_{light.name}',
                                   pixels=px, colorspace='Non-Color')
        light.cookie_strength = 1.0
        if kind == 'SUN':
            light.cookie_scale = scale   # world units per tile
        light._caustic_cookie = True
        light._cookie_tex = None         # the CPU sampler re-reads it


#: the fixed tap pattern a flare's visibility is sampled with: a 13-point
#: disc, deterministic, the same on every device
_FLARE_TAPS = ((0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
               (0.7, 0.7), (-0.7, 0.7), (0.7, -0.7), (-0.7, -0.7),
               (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4))


def _flare_sources(scene, st, gbuf, vp):
    """Screen anchors for the per-lamp lens flares, with visibility.

    Every Video Post flare of the era faded as its source slipped
    behind geometry -- that behaviour IS the feature, and it is what
    the image-space post flare (which chases any bright pixel) cannot
    do. The source's screen disc is sampled against the z-buffer in
    NDC -- the same space both were projected through, so no
    reconstruction and no unit traps -- and the covered fraction scales
    every element. A SUN anchors at its vanishing point and is visible
    only through sky.
    """
    out = []
    lights = getattr(scene, 'lights', None) or ()
    if not any(float(getattr(l, 'flare', 0.0)) > 0.0 for l in lights):
        return out
    h, w = gbuf.depth.shape
    for light in lights:
        fi = float(getattr(light, 'flare', 0.0))
        if fi <= 0.0:
            continue
        if str(getattr(light, 'type', '')).upper() == 'SUN':
            d = M.normalize(np.asarray(light.direction, np.float32))
            clip = np.append(-d, 0.0).astype(np.float32) @ vp.T
            z_l = np.inf                 # at infinity: only sky shows it
        else:
            pos = np.asarray(light.position, np.float32)
            clip = np.append(pos, 1.0).astype(np.float32) @ vp.T
            z_l = None
        if clip[3] <= 1e-6:
            continue                     # behind the camera
        ndc = clip[:3] / clip[3]
        if abs(ndc[0]) > 1.4 or abs(ndc[1]) > 1.4:
            continue                     # far enough off-frame to matter not
        if z_l is None:
            z_l = float(ndc[2])
        px = (float(ndc[0]) * 0.5 + 0.5) * (w - 1)
        py = (float(ndc[1]) * 0.5 + 0.5) * (h - 1)
        rad = max(2.0, 0.008 * min(w, h)
                  * float(getattr(light, 'flare_scale', 1.0)))
        vis = 0
        for ox, oy in _FLARE_TAPS:
            x = int(round(px + ox * rad))
            y = int(round(py + oy * rad))
            if x < 0 or x >= w or y < 0 or y >= h:
                vis += 1                 # off-frame taps count as open sky
                continue
            zs = float(gbuf.depth[y, x])
            if not np.isfinite(zs):
                vis += 1                 # sky
            elif np.isfinite(z_l) and z_l <= zs + 1e-7:
                vis += 1                 # the lamp is nearer than the surface
        frac = vis / float(len(_FLARE_TAPS))
        if frac <= 0.0:
            continue
        out.append({'x': float(ndc[0]), 'y': float(ndc[1]),
                    'color': tuple(float(c) for c in light.color),
                    'intensity': fi * frac,
                    'scale': float(getattr(light, 'flare_scale', 1.0)),
                    'streaks': int(getattr(light, 'flare_streaks', 6)),
                    'rings': int(getattr(light, 'flare_rings', 1)),
                    'ghosts': int(getattr(light, 'flare_ghosts', 6))})
    return out


def _shadow_coarseness_note(scene, gbuf, proj, rh):
    """One line when shadow texels dwarf output pixels -- the high-res trap.

    A 512 map that reads perfectly at 640x480 turns every contact shadow
    into a blocky fringe at 1920+ -- the field read it as "a faint
    wireframe on all objects", and no setting they tried moved it because
    lights carrying their own Map Size ignore the render slider. The note
    names the coarse lights, the ratio, and both roads out.
    """
    try:
        mesh = scene.mesh
        cam = scene.camera
        if mesh is None or mesh.verts is None or not mesh.verts.size \
                or cam is None:
            return
        centre = mesh.verts.mean(axis=0)
        eye = np.asarray(getattr(cam, 'position', (0.0, 0.0, 0.0)),
                         np.float32)
        dist_cam = float(np.linalg.norm(centre - eye))
        ang, ortho_w = pixel_footprint(cam, proj, rh)
        px_world = ortho_w if ang == 0.0 else ang * max(dist_cam, 1e-6)
        if px_world <= 0.0:
            return
        coarse = []
        for light in (scene.lights or ()):
            sm = getattr(light, 'shadow_map', None)
            if sm is None:
                continue
            origin = np.asarray(getattr(sm, 'origin', (0, 0, 0)),
                                np.float32)
            d = float(np.linalg.norm(centre - origin))
            texel = float(np.asarray(sm.texel_size(
                np.asarray([d], np.float32)))[0])
            ratio = texel / px_world
            if ratio > 3.0:
                own = int(getattr(light, 'shadow_map_size', 0) or 0)
                size = int(getattr(sm, 'size', 0) or 0)
                if not size:                     # a cube map wraps its faces
                    faces = getattr(sm, 'faces', None)
                    size = int(getattr(faces[0], 'size', 0)) if faces else 0
                coarse.append((getattr(light, 'name', '?') or '?',
                               size, ratio, own > 0))
        if not coarse:
            return
        named = ', '.join(f"'{n}' ({size} map, ~{r:.0f} px/texel"
                          + (', per-light size set' if own else '')
                          + ')'
                          for n, size, r, own in coarse[:4])
        overridden = any(own for _n, _s, _r, own in coarse)
        road = ('raise the light\'s own Map Size (it overrides the render '
                'setting)' if overridden else
                'raise Shadow Map Size in the render settings, or the '
                "light's own Map Size")
        print(f'[Halcyon] shadows: one shadow texel spans several output '
              f'pixels at this resolution -- contact shadows go blocky. '
              f'{named}; {road}')
    except Exception:                                           # noqa: BLE001
        pass


def snap_grid(st):
    """Screen-space vertex snap in pixels, from BOTH settings that promise it.

    Fixed-Point Subpixel and PS1 Vertex Snap land on the same machinery: a
    grid the projected vertex is rounded onto before the fill. Subpixel
    Precision spent its whole life unread -- two presets set INTEGER and
    nothing changed (found by the settings audit). The coarser of the two
    requests wins. INTEGER and FIXED_1 share the whole-pixel grid: the one
    PS1 behaviour not reproduced is truncation rather than rounding, a
    constant half-pixel phase this raster does not carry.
    """
    sub = {'FIXED_4': 0.25, 'FIXED_1': 1.0, 'INTEGER': 1.0}.get(
        str(getattr(st, 'subpixel_precision', 'FLOAT')), 0.0)
    vs = float(st.vertex_snap_grid) if st.vertex_snap else 0.0
    return max(vs, sub)


def _apply_material_override(scene, st):
    """Material Override: the scene with every material a plain matte.

    A shallow scene copy whose material list is rebuilt as graphless
    Lambert surfaces in the override colour -- same names, same indices,
    so meshes, passes and per-material machinery are untouched.
    Geometry, lights and shadows stay real; textures and shading graphs
    are simply not there, which is what a clay test render is. Both
    devices see the SAME plain materials, so CPU/GPU parity holds by
    construction (a graphless constant material is the GPU plan's
    easiest case).
    """
    mode = str(getattr(st, 'material_override', 'NONE') or 'NONE').upper()
    if mode == 'NONE':
        return scene
    import copy as _copy
    col = tuple(float(c) for c in
                (getattr(st, 'override_color', None) or (0.7, 0.7, 0.7)))
    from .scene import Material as _Mat
    out = _copy.copy(scene)
    out.materials = [
        _Mat(name=getattr(m, 'name', f'mat{i}'),
             index=getattr(m, 'index', i), model='LAMBERT',
             diffuse=col, specular_level=0.0)
        for i, m in enumerate(scene.materials or [])]
    return out


# ---------------------------------------------------------------- R253:
# the render region (Blender's Ctrl+B border, Output > Format > Render
# Region). One mechanism for F12 and the viewport: a RECT carried on the
# settings as Blender's own fractions, applied here as "full-frame
# G-buffer, shading masked to the rect plus a context ring, zeros
# outside, full-frame return shape". The post chain then runs over the
# full-frame canvas, so every pattern stage (CRT mask, Bayer tiles,
# interlace rows, N64 / noise hashes, codec blocks) keeps its full-frame
# anchoring with no per-stage change, and the crop happens LAST (the
# engine for F12, the viewport worker for the rendered view).

def region_pixels(st, W, H):
    """The render region in OUTPUT pixels, (x0, y0, x1, y1), or None.

    Blender's own rounding (render/intern/pipeline.cc: disprect.xmin =
    border_min_x * winx, truncated) on the min edge; the max edge is
    CEILED so a Pixel Scale render rect (computed at render resolution)
    is always a superset of the output crop (computed at output
    resolution). At least one pixel each way, clamped to the frame.
    None when the border is off, empty, or covers the whole frame -- so
    an old scene never enters the region code at all."""
    if not bool(getattr(st, 'use_border', False)):
        return None
    W = max(int(W), 1)
    H = max(int(H), 1)
    try:
        mnx = float(getattr(st, 'border_min_x', 0.0))
        mny = float(getattr(st, 'border_min_y', 0.0))
        mxx = float(getattr(st, 'border_max_x', 1.0))
        mxy = float(getattr(st, 'border_max_y', 1.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (mnx, mny, mxx, mxy)):
        return None
    mnx, mny = min(max(mnx, 0.0), 1.0), min(max(mny, 0.0), 1.0)
    mxx, mxy = min(max(mxx, 0.0), 1.0), min(max(mxy, 0.0), 1.0)
    if mxx <= mnx or mxy <= mny:
        return None
    x0 = min(int(mnx * W), W - 1)
    y0 = min(int(mny * H), H - 1)
    x1 = min(max(int(math.ceil(mxx * W)), x0 + 1), W)
    y1 = min(max(int(math.ceil(mxy * H)), y0 + 1), H)
    if x0 == 0 and y0 == 0 and x1 == W and y1 == H:
        return None
    return (x0, y0, x1, y1)


def _region_keep(rect, rw, rh, ss, reach):
    """The 2-D twin of the band `keep` row mask: a (rh, rw) bool mask of
    the rect at INTERNAL resolution (rect * ss), grown by `reach` pixels
    on every side -- the context ring the neighbour-reading stages (ink,
    bump pre-passes, the interpolated radiosity grid, the edge tent, the
    adaptive contrast window, the Fuzz / Thin Wall blends) read past the
    rect, exactly as the band scissor keeps its context rows."""
    x0, y0, x1, y1 = rect
    s = max(int(ss), 1)
    r = max(int(reach), 0)
    keep = np.zeros((int(rh), int(rw)), bool)
    keep[max(y0 * s - r, 0):min(y1 * s + r, int(rh)),
         max(x0 * s - r, 0):min(x1 * s + r, int(rw))] = True
    return keep


def _region_box(keep):
    """(x, y, w, h) of a keep mask's true pixels -- the GPU scissor box
    (device.draw_many's read_region takes this shape)."""
    ys = np.nonzero(keep.any(axis=1))[0]
    xs = np.nonzero(keep.any(axis=0))[0]
    if ys.size == 0 or xs.size == 0:
        return (0, 0, 0, 0)
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1 - int(xs[0]),
            int(ys[-1]) + 1 - int(ys[0]))


def _apply_region(out, rect, y_off=0):
    """Zero (RGBA 0,0,0,0) every output pixel outside the rect. `y_off`
    is the row the array starts at (a band's first output row), so a
    pooled band of a region frame zeroes the same pixels the whole frame
    would. The array is edited in place when it is writable."""
    if rect is None or out is None:
        return out
    x0, y0, x1, y1 = rect
    h, w = out.shape[:2]
    ya = min(max(y0 - int(y_off), 0), h)
    yb = min(max(y1 - int(y_off), 0), h)
    xa = min(max(x0, 0), w)
    xb = min(max(x1, 0), w)
    try:
        out = np.ascontiguousarray(out, np.float32)
        if not out.flags.writeable:
            out = out.copy()
    except Exception:                                           # noqa: BLE001
        out = np.array(out, np.float32)
    if ya > 0:
        out[:ya] = 0.0
    if yb < h:
        out[yb:] = 0.0
    if xa > 0:
        out[ya:yb, :xa] = 0.0
    if xb < w:
        out[ya:yb, xb:] = 0.0
    return out


def _region_reach(scene, st, ink_reach, has_bump):
    """How far past the rect the frame's stages read, in internal pixels:
    the band scissor's own context rules (ink, bump, the interpolated
    radiosity grid) plus one ring for the edge tent (_edge_smooth), two
    for the adaptive contrast window (_adaptive_mask: 'within two
    pixels of a geometric flag'). The Fuzz / Thin Wall blends have no
    finite ring (render() renders them whole and crops, by name)."""
    reach = int(ink_reach or 0)
    if has_bump:
        reach += 1
    rad_n = int(getattr(st, 'radiosity_spacing', 1) or 1)
    if getattr(st, 'radiosity', False) and rad_n > 1:
        reach += 2 * rad_n
    mode = str(getattr(st, 'aa_mode', 'NONE'))
    if mode == 'EDGE':
        reach += 1
    elif mode == 'ADAPTIVE':
        reach += 2
    return reach


def _region_refuse(st, why):
    """The region's refusals, printed once per reason (the period
    refusal doctrine: by name, never silent)."""
    key = str(why)
    said = getattr(st, '_period_refusals', None)
    if isinstance(said, set):
        if key in said:
            return
        said.add(key)
    print(f'[Halcyon] render region: {why}')


def render(scene, settings=None, progress=None, band=None):
    """Render `scene`. Returns a linear (H,W,4) float32 image.

    `band` is an optional (y0, y1) range of output rows; only those rows are
    shaded and returned. Used by the worker pool to split a frame across
    processes.

    Row 0 is the BOTTOM of the picture: the rasteriser maps NDC y = -1 to row 0.
    That matches Blender's render-result buffer, so the engine hands it over
    without flipping. Anything writing a PNG (PIL, most image libraries) treats
    row 0 as the top and must flip first.
    """
    st = settings or scene.settings
    st._period_refusals = set()      # R251 (MAT-A): CB.refuse prints once
    # R251 C012: the console vertex formats, on the persistent scene
    # (idempotent: quantize_mesh returns the same object for the same
    # dial and the original for NONE)
    scene.mesh = raster.quantize_mesh(
        scene.mesh, str(getattr(st, 'vertex_quantize', 'NONE')),
        float(getattr(st, 'vertex_units', 64.0)))
    scene.last_gel = None       # R251 C134: rebuilt below, beside last_depth
    scene = _apply_material_override(scene, st)
    collect_exclusive_lights(scene)
    mesh = scene.mesh
    W = max(int(st.resolution_x), 1)
    H = max(int(st.resolution_y), 1)
    # sugar sockets become real nodes here, before ANY consumer reads a
    # graph: the CPU evaluator, the GPU plan and the emitter must all see
    # the same desugared tree or the plan's constancy scan lies
    from .nodeeval import desugar_master_bump
    for _m in (scene.materials or ()):
        desugar_master_bump(getattr(_m, 'graph', None))
    if getattr(scene, 'world', None) is not None:
        desugar_master_bump(getattr(scene.world, 'graph', None))
    _synth_caustic_cookies(scene)
    # R253: the render region in output pixels, or None (the identity --
    # every old scene takes exactly the pre-R253 paths below)
    rect = region_pixels(st, W, H)
    if rect is not None and composite_reads_neighbours(scene, st):
        # the Fuzz / Thin Wall blends read the FINISHED frame's
        # neighbouring rows, layer upon layer (a Spectre behind a Spectre
        # reads a row that itself read a row), so no finite context ring
        # holds -- the worker-pool gate refuses them for the same reason.
        # The frame renders whole and the rect is cut after, by name
        _region_refuse(st, 'the Fuzz / Thin Wall blend reads neighbouring '
                           'pixels of the finished frame; the frame renders '
                           'whole and is cropped')
        st_w = st.copy()
        st_w.use_border = False
        st_w._period_refusals = st._period_refusals
        for _k in ('_viewport', '_viewport_stats', '_keep_gpu_frame',
                   '_stereo', '_pano_strip', '_accum_jitter', '_lens_pass',
                   '_refine_mask'):
            if hasattr(st, _k):
                setattr(st_w, _k, getattr(st, _k))
        out_w = render(scene, st_w, progress, band)
        try:
            st._frame_gpu_shaded = bool(getattr(st_w, '_frame_gpu_shaded',
                                                False))
            st._last_coverage = getattr(st_w, '_last_coverage', None)
        except Exception:                                       # noqa: BLE001
            pass
        # the crop below is a CPU edit of the whole frame: a resident GPU
        # frame of the whole picture is released by name, never kept
        from ..gpu import frame as _FRw
        _FRw.edited(st_w, 'render region', 'rendered whole and cropped')
        _FRw.release(st_w)
        return _apply_region(out_w, rect,
                             y_off=max(band[0], 0) if band is not None else 0)
    # PANO outranks stereo: QTVR panoramas were mono deliverables, and an
    # off-axis frustum shift has no honest meaning on a stitched cylinder
    if scene.camera is not None and \
            str(getattr(scene.camera, 'type', 'PERSP')) == 'PANO' and \
            not getattr(st, '_pano_strip', False) and band is None and \
            not getattr(st, '_viewport', False):
        if rect is not None:
            # R253: a strip of the drum has no rect meaning (the region
            # is a planar window); the panorama renders whole and the
            # rect is cut after the stitch, by name
            _region_refuse(st, 'the panorama camera renders whole and '
                               'is cropped after the stitch')
            st_w = st.copy()
            st_w.use_border = False
            st_w._period_refusals = st._period_refusals
            return _apply_region(_render_panorama(scene, st_w, progress),
                                 rect)
        return _render_panorama(scene, st, progress)
    # R251 C125: Blender 2.4's Pano + Xparts -- N yawed planar strips
    # butted together; a PANO camera (the true cylinder) wins above,
    # stereo / Y-shear stay off inside the strips, the AA roads and the
    # lens passes run inside each strip
    if int(getattr(st, 'pano_parts', 1)) > 1 and scene.camera is not None \
            and str(getattr(scene.camera, 'type', 'PERSP')) == 'PERSP' \
            and not getattr(st, '_pano_strip', False) and band is None \
            and not getattr(st, '_viewport', False):
        if rect is not None:
            _region_refuse(st, 'the panorama camera renders whole and '
                               'is cropped after the stitch')
            st_w = st.copy()
            st_w.use_border = False
            st_w._period_refusals = st._period_refusals
            return _apply_region(_render_pano_parts(scene, st_w, progress),
                                 rect)
        return _render_pano_parts(scene, st, progress)
    if str(getattr(st, 'stereo_mode', 'NONE')) != 'NONE' and \
            getattr(st, '_stereo', None) is None and band is None and \
            not getattr(st, '_pano_strip', False) and \
            not getattr(st, '_viewport', False):
        return _render_stereo(scene, st, progress)
    if str(st.aa_mode) == 'ACCUMULATE' and int(st.aa_samples) > 1 \
            and getattr(st, '_accum_jitter', None) is None:
        return _render_accumulated(scene, st, progress, band)
    # R251 C098: lens passes are an F12 road: the viewport shows NO depth
    # of field under Lens Passes (K full frames per draft and refine
    # would be the lag the rules forbid; the post blur belongs to the
    # POST method). No _pano_strip term: the shear is a per-strip camera
    # matrix and composes inside each Pano Parts strip as ACCUMULATE does
    if bool(st.dof) and str(getattr(st, 'dof_method', 'POST')) == 'LENS_ACCUMULATE' \
            and not getattr(st, '_lens_pass', False) and scene.camera is not None \
            and str(getattr(scene.camera, 'type', 'PERSP')) == 'PERSP' \
            and not getattr(st, '_viewport', False):
        return _render_lens_accumulated(scene, st, progress, band)
    LAST_GPU_VERDICT.update(wanted=False, engaged=False, why='')
    # R250: the frame's residency record; a reused settings object
    # (motion blur re-renders on the same one) may still carry the last
    # frame's handle -- released here, never leaked
    from ..gpu import frame as _FR
    _FR.begin_frame(st)
    # R249/R250: the flag every later GPU gate reads (post, ink) starts
    # each frame false -- and BEFORE the geometry-less early return
    # below, which used to leave a reused settings object with the
    # previous frame's verdict
    st._frame_gpu_shaded = False
    # R251 (C092): the coverage plane starts every frame absent, before
    # the geometry-less early return (a reused settings object must not
    # carry the previous frame's plane)
    scene.last_coverage = None
    st._last_coverage = None
    ss = 1
    if st.aa_mode == 'SUPERSAMPLE':
        ss = max(int(np.round(np.sqrt(max(st.aa_samples, 1)))), 1)
    rw, rh = W * ss, H * ss

    # R251 C058: the Y-shear flag lives on the bpy-free Camera, rewritten
    # every frame (a stereo eye or an AA pass re-derives it from the
    # settings itself; st.copy() drops private flags, the camera keeps it)
    if scene.camera is not None:
        scene.camera._yshear = yshear_active(scene.camera, st)
    view, proj, vp, eye = camera_matrices(scene.camera, rw, rh)
    # the accumulation jitter and the stereo eye's window shift, as ONE
    # clip-space translation (clip_jitter: the J block's own lines)
    J = clip_jitter(st, proj, rw, rh)
    if J is not None:
        proj = (J @ proj).astype(np.float32)
        vp = (J @ vp).astype(np.float32)
    # the water needs to know how big a pixel is before it can decide which
    # waves are too small to draw
    if getattr(scene, 'world', None) is not None:
        try:
            pa, pw = pixel_footprint(scene.camera, proj, rh)
            scene.world._pixel_angle = pa
            scene.world._pixel_width = pw
        except Exception:                                       # noqa: BLE001
            pass
    with ST.track('prepare textures'):
        textures = prepare_textures(scene, st)

    # R221: bake any Anime Shader's linked Shadow Ramp into its LUT
    # here, once, before shading -- the lamp loop samples it and the
    # GPU uploads the same texels (cache-keyed, so an unchanged ramp
    # costs one fingerprint)
    from .nodeeval import bake_anime_ramp
    for _m in (scene.materials or ()):
        g = getattr(_m, 'graph', None)
        if g and 'HALCYON_AnimeShaderNode' in str(g.get('nodes', {})):
            try:
                bake_anime_ramp(g, textures, st)
            except Exception:                                   # noqa: BLE001
                pass

    need_bvh = st.raytrace or st.ambient_occlusion or \
        getattr(st, 'radiosity', False) or \
        (st.shadows and st.shadow_default == 'RAY') or \
        _volume_occlusion_wanted(scene, st)
    bvh = None
    if need_bvh and mesh is not None and mesh.tris is not None and mesh.tris.size:
        with ST.track('build BVH'):
            bvh = _cached_bvh(scene, mesh)

    with ST.track('shadow maps'):
        _build_shadows(scene, st, mesh)
    # R208: the caster mask travels to the RAY shadow road too (the
    # maps already honour it at bake time). visibility() reads it off
    # the settings and hands it to the BVH's any-hit as a cast filter,
    # so a fur pelt with Cast Shadows off stops blanketing its own
    # body. Refreshed every frame; None when everything casts.
    st._shadow_cast_tri = _caster_keep_tri(
        scene, mesh) if st.shadows and st.ray_shadows else None
    # R251 shadow pack: the mask pack (planar polygons, DC modifier
    # volumes, DS shadow polygons) is baked after the G-buffer is final
    st._mask_pack = st._mask_params = None
    # R204: the infinite floor answers the scene's lamps now -- built
    # after the BVH and the shadow maps so its cast shadows can use
    # whichever of the two each lamp uses
    _stash_ground_light(scene, st, bvh)
    if False:
        pass

    if progress:
        progress(0.05, 'Rasterising')

    gbuf = raster.GBuffer(rw, rh)
    subdiv_px = int(getattr(st, 'tex_affine_subdiv', 0) or 0) \
        if not st.tex_perspective else 0
    # the camera raster's near-plane epsilon: a real dial now (the
    # settings audit found it registered, ranged, tooltipped and read
    # by NOTHING -- the rasterisers ran their 1e-5 default regardless)
    near_eps = max(float(getattr(st, 'clip_near_epsilon', 1e-5) or 1e-5),
                   1e-6)
    # ---- R251 raster options (RAST-A1): every new rasteriser dial in
    # ONE object, built once, carried through the camera-space raster
    # call sites and the kernel road, keyed into the G-buffer cache.
    # The W encodings read the camera's own depth mapping z = A + B / w
    # (near and far DERIVED from the projection matrix -- Blender's
    # calc_matrix_camera and Halcyon's _persp alike -- so the clip
    # planes, the matrix and the encoding are one truth)
    _A = float(-proj[2, 2])
    _B = float(proj[2, 3])
    _is_ortho = str(getattr(scene.camera, 'type', 'PERSP')) == 'ORTHO'
    _wnear = -_B / (1.0 + _A) if (1.0 + _A) != 0.0 else 0.0
    _wfar = _B / (1.0 - _A) if (1.0 - _A) != 0.0 else 0.0
    _enc = str(getattr(st, 'depth_encoding', 'LINEAR') or 'LINEAR')
    if _enc not in raster.DEPTH_ENCODINGS:
        _enc = 'LINEAR'
    if _enc in raster.W_ENCODINGS and _is_ortho:
        # an orthographic camera has w = 1 everywhere: the fallback by
        # name, BEFORE either road, so the raster never sees the case
        print(f'[Halcyon] depth encoding {_enc} needs a perspective '
              'camera: LINEAR depth this frame')
        _enc = 'LINEAR'
    _painters = str(getattr(st, 'depth_sort', 'ZBUFFER')) == 'PAINTERS'
    if _painters and _enc != 'LINEAR':
        # a painter's key is a distance in units or a bucket, not NDC
        # z: an encoding would clamp every polygon to the top code
        print(f"[Halcyon] depth encoding {_enc} is a z-buffer rule; "
              "Painter's sort keeps LINEAR this frame")
        _enc = 'LINEAR'
    _ot_on = _painters and \
        str(getattr(st, 'painters_key', 'CENTROID')) == 'ORDERING_TABLE'
    # R251 C127: REYES jitter -- fresh per (frame, seed); inert where
    # EDGE / ADAPTIVE / ACCUMULATE run their own sample logic (their
    # sub-passes render at aa_mode NONE with _accum_jitter set)
    _jit = None
    if str(getattr(st, 'aa_sample_pattern', 'GRID')) == 'JITTER' and \
            str(st.aa_mode) in ('NONE', 'SUPERSAMPLE') and \
            getattr(st, '_accum_jitter', None) is None:
        _jit = (int(getattr(scene, 'frame', 1) or 1),
                int(getattr(st, 'seed', 0) or 0))
    # R251 C001: the RDP's coverage plane, at 1 sample per pixel only
    # (the VI filter reads the coverage at output size): a named no-op
    _cvg_on = bool(getattr(st, 'n64_coverage_aa', False))
    if _cvg_on and ss > 1:
        if int(ss) not in _N64_SKIP_SAID:
            _N64_SKIP_SAID.add(int(ss))
            print('[Halcyon] N64 coverage AA needs 1 sample per pixel (the '
                  'VI filter reads the coverage at output size): skipped '
                  'this frame')
        _cvg_on = False
    opts = raster.RasterOpts(
        pixel_shift=raster.pixel_shift_of(st),
        reject=str(getattr(st, 'near_clip_mode', 'CLIP')) == 'REJECT',
        size_limit=(1023 * ss, 511 * ss) if _ot_on else None,
        jitter=_jit,
        cvg=_cvg_on,
        enc=_enc,
        wparams=raster.WParams(near=np.float32(_wnear),
                               far=np.float32(_wfar),
                               A=np.float32(_A), B=np.float32(_B),
                               is_ortho=_is_ortho))
    st._raster_opts = opts          # the punch-through raster reads it
    # R251 C038: the DS rear-plane depth bitmap -- the World's Z pass as
    # the frame's own z-buffer clear, in the frame's encoding and bits
    _wd = getattr(scene.world, 'backdrop_depth', None) \
        if scene.world is not None else None
    if _wd is not None:
        opts.clear = raster.backdrop_clear(
            _wd, getattr(scene.world, 'backdrop_offset', (0, 0)), rw, rh,
            opts, int(st.depth_precision), ss=ss)
        gbuf.clear_depth(opts.clear[0], opts.clear[1], opts.enc)
    if not st.tex_perspective:
        gbuf.alloc_linear()
    frags = raster.FragmentList() if st.transparency in ('SORTED', 'ABUFFER') else None

    job = ShadeJob(scene, st, textures, bvh, view, eye, rw, rh)
    # R250: the sky pass draws the camera rays the background pass draws:
    # the same (jittered, stereo-shifted) view-projection and the same
    # supersample factor ride on the job for gpu/shade and gpu/sky
    job.vp = vp
    job.ss = ss
    # R253: the light-split sink -- one (rh, rw, 3) float32 plane per
    # wanted light pass at RENDER resolution (the cost the Per-Lamp
    # tooltip names), Shadow and AO starting white (an uncovered or
    # unlit pixel reads 'lit / open'). Whole-frame roads only: a band
    # (the pool skips pass frames anyway) and the viewport allocate none
    _env_rgb = None
    _lpn = light_pass_names(st) if (band is None and not getattr(
        st, '_viewport', False)) else ()
    if _lpn:
        job.pass_sink = {
            _pn: np.full((rh, rw, 3),
                         np.float32(1.0 if _pn in ('Shadow', 'AO') else 0.0),
                         np.float32)
            for _pn in _lpn}
    if bool(getattr(st, 'fog', False)) and \
            str(getattr(st, 'fog_color_source', 'FIXED')) == 'BACKDROP':
        # R251 LIGHT-A2 (F008, A30): the backdrop at EVERY pixel, built
        # once before either road shades, so both read one array
        from . import sky as _SKYB
        with ST.track('backdrop (fog target)'):
            job.backdrop = _SKYB.backdrop_rgb(scene, st, rw, rh, vp, eye,
                                              textures, ss=ss)

    if mesh is None or mesh.tris is None or mesh.tris.size == 0:
        # R253: a region frame evaluates the sky inside the rect only
        # (per-pixel, so the rect's values are the whole frame's); the
        # halos / weather below draw whole and the resolve zeroes outside
        img = _background_image(scene, st, rw, rh, vp, eye,
                                _region_keep(rect, rw, rh, ss, 0)
                                if rect is not None else None, textures)
        if band is None:
            # R194/R195 field finds: this early path returned before
            # ANY of the whole-frame extras were computed. A scene of
            # nothing but halo objects drew no halos; a scene of
            # nothing but a sky, a sun and a camera -- the natural way
            # to test a lens flare -- could never flare, because
            # last_flares was never set. An all-far depth buffer is
            # the honest z-test for a geometry-less frame (every tap
            # is open sky), and the same buffer serves the halo splat
            g_empty = raster.GBuffer(rw, rh)
            if getattr(scene, 'halos', None):
                with ST.track('halos'):
                    img = _draw_halos(img, scene, st, g_empty,
                                      view, proj, rw, rh,
                                      sel_mask=getattr(st, '_refine_mask',
                                                       None))
            # R200: a sky-only scene can rain too -- the same early
            # path that once ate the halos and the flares
            wd_w = getattr(scene, 'world', None)
            if wd_w is not None and \
                    str(getattr(wd_w, 'weather', 'NONE')
                        or 'NONE') != 'NONE' and \
                    not getattr(st, '_pano_strip', False):
                from . import sky as SKY
                with ST.track('weather'):
                    img = SKY.weather_overlay(
                        img, wd_w, rw, rh,
                        time=float(getattr(scene, 'time', 0.0)),
                        out_wh=(W, H),
                        sel_mask=getattr(st, '_refine_mask', None))
            scene.last_shafts = shaft_sources(scene, st, vp)
            scene.last_flares = _flare_sources(scene, st, g_empty, vp)
        if band is not None:
            y0, y1 = max(band[0], 0), min(band[1], H)
            return _apply_region(
                _resolve(img[y0 * ss:y1 * ss], W, y1 - y0, ss, st),
                rect, y_off=y0)
        return _apply_region(_resolve(img, W, H, ss, st), rect)

    opaque, transparent = _split_by_alpha(scene, mesh, st)
    snap = snap_grid(st)
    cull = 'BACK' if st.backface_cull else 'NONE'

    # BI subsurface scattering: the per-material point-cloud pre-pass
    # (2.79's make_sss_tree), before any beauty pixel shades. Runs at
    # the BASE resolution with no supersampling, exactly the OSA-off
    # pre-render sss_create_tree_mat performed.
    try:
        _sss_prepare(scene, st, job, view, W, H)
    except Exception:                                           # noqa: BLE001
        import traceback
        traceback.print_exc()
        job.sss_trees = {}

    # when only a band is wanted, the rasteriser is told so: otherwise every
    # worker in a pool rasterises the whole mesh for its own slice
    scissor = None
    has_bump = any('ShaderNodeBump' in
                   str((getattr(m, 'graph', None) or {}).get('nodes', {}))
                   for m in getattr(scene, 'materials', ()) or ())
    # R227: the ink's REACH in rows. A band's scissor culls the
    # triangles outside its rows, and an outline seeded in the row
    # just past the band used to be missing from the band's edge rows
    # -- a seam between every pair of pooled bands under a thick line,
    # there since the outline pass shipped (found by the style pack's
    # band pin). The scissor keeps that many context rows now, exactly
    # as the bump road keeps its one
    ink_reach = _ink_reach_rows(scene, st) \
        if (getattr(st, 'outline', False) or _ink_forced(scene)) else 0
    # R233: the painted background road's strokes, blur and setback
    # read past a band the same way (core/gouache.py names the rows)
    from . import gouache as GOU
    _gou_on = GOU.on(scene, st)
    if _gou_on:
        ink_reach += GOU.reach_rows(scene, st, ss)
    # R238: the cel field's march and rim read past a band the same way
    from . import celfield as _CFR
    if _CFR.field_on(scene, st):
        ink_reach += _CFR.reach_rows(scene, st, rh)
    if band is not None:
        scissor = (max(band[0], 0) * ss, min(band[1], H) * ss)
        if ink_reach:
            scissor = (max(scissor[0] - ink_reach, 0),
                       min(scissor[1] + ink_reach, rh))
        if has_bump:
            # one CONTEXT row past the band's top: n_bump differences
            # toward the +y neighbour, and without that row a band's
            # last shaded row flattened its waves -- the same seam the
            # chunk fix killed, band edition. The scissor culls whole
            # triangles, so one extra row keeps the neighbour coverage
            # complete; shading stays band-masked below.
            scissor = (scissor[0], min(scissor[1] + 1, rh))
        rad_n = int(getattr(st, 'radiosity_spacing', 1) or 1)
        if getattr(st, 'radiosity', False) and rad_n > 1:
            # the interpolated gather's grid blocks must be COMPLETE
            # inside a band, or a band's grid point could pick a
            # different source pixel than the whole frame's and seam.
            # A band pixel reads grid rows up to 2N-1 beyond itself
            # (the far corner of the next grid point's block), so the
            # scissor grows by 2N each way; the extra rows rasterise
            # and are never shaded (band masking below is untouched).
            scissor = (max(scissor[0] - 2 * rad_n, 0),
                       min(scissor[1] + 2 * rad_n, rh))
    # R253: the region's 2-D keep mask -- the rect at internal
    # resolution plus the context ring the band scissor's own rules
    # state (core/render._region_reach). The OPAQUE raster stays
    # whole-frame on both devices (the G-buffer cache is keyed on the
    # raster's inputs, never the rect, so a cached full frame serves a
    # region frame and a region frame can never poison a full one); the
    # see-through raster and composite take the rect's rows as a band
    # takes its own (`t_scissor`), the proven per-row context road
    _keep2 = None
    _region_box_px = None
    t_scissor = scissor
    if rect is not None:
        _keep2 = _region_keep(rect, rw, rh, ss,
                              _region_reach(scene, st, ink_reach, has_bump))
        _region_box_px = _region_box(_keep2)
        if band is None:
            t_scissor = (_region_box_px[1],
                         _region_box_px[1] + _region_box_px[3])
    # the GPU layer passes (gpu/shade.shade_fragments_frame) read the
    # box off the job: each depth rank's scissor becomes bbox AND rect
    job.region = _region_box_px

    flat_depth = None
    if st.depth_sort == 'PAINTERS':
        flat_depth = polygon_depths(
            mesh, view, eye, st.painters_key,
            ot=(int(st.ot_length), float(st.ot_far))
            if str(st.painters_key) == 'ORDERING_TABLE' else None)

    want_overdraw = st.debug_pass == 'OVERDRAW'
    # ---- the G-buffer cache (R170): a redraw with an unchanged camera,
    # mesh and raster settings re-rasterised ~220ms of identical arrays
    # on every viewport refine (85% of an idle refine) and every
    # repeated F12. The key is CONTENT -- mesh fingerprint, the exact
    # view-projection bytes, sizes, and every dial the rasteriser reads
    # -- so an orbit, a deform or a settings change misses honestly.
    # Cached: the plain z-buffer case only (no Painter's, no A-buffer
    # fragments, no overdraw census, no bands); restored by copy into
    # the fresh GBuffer, never shared.
    # (frags is NOT a condition: the transparent subset rasterises in
    # its own later stage -- the main raster's G-buffer covers the
    # opaque subset only, which the key fingerprints via `_sub`)
    gbuf_ckey = None
    # R171: the field's viewport missed this cache on every refine while
    # the headless twin hit -- so a miss now NAMES the key component (or
    # the eligibility gate) that broke it, and the viewport split line
    # prints it. One pasted `gbuf MISS(vp)` beats a round of guessing.
    if band is not None:
        _GBUF_STATS['last'] = 'off(band)'
    elif flat_depth is not None:
        _GBUF_STATS['last'] = 'off(painters)'
    elif want_overdraw:
        _GBUF_STATS['last'] = 'off(overdraw)'
    else:
        try:
            from ..gpu.shade import _mesh_key as _gk
            _sub = 'all' if opaque is None else \
                (int(opaque.size), int(opaque[::17].astype(np.int64).sum()))
            gbuf_ckey = (_gk(mesh), np.asarray(vp, np.float32).tobytes(),
                         rw, rh, cull, float(snap),
                         int(st.depth_precision), int(subdiv_px),
                         float(near_eps), gbuf.bary_lin is not None, _sub,
                         opts.key())
        except Exception:                                       # noqa: BLE001
            gbuf_ckey = None
            _GBUF_STATS['last'] = 'off(keyfail)'
    _gb_hit = _GBUF_CACHE.get(gbuf_ckey) if gbuf_ckey is not None else None
    rastered_from_cache = False
    if _gb_hit is not None:
        with ST.track('rasterise (cached)'):
            _GBUF_CACHE[gbuf_ckey] = _GBUF_CACHE.pop(gbuf_ckey)  # LRU touch
            gbuf.tri[:] = _gb_hit['tri']
            gbuf.bary[:] = _gb_hit['bary']
            gbuf.depth[:] = _gb_hit['depth']
            gbuf.zndc[:] = _gb_hit['zndc']
            gbuf.front[:] = _gb_hit['front']
            if _gb_hit.get('zkey') is not None:
                gbuf.alloc_zkey()[:] = _gb_hit['zkey']
            if _gb_hit.get('cvg') is not None:
                # R251 C001: without this a HIT (the second F12, every
                # viewport redraw) would hand the VI an all-full plane
                gbuf.alloc_cvg()[:] = _gb_hit['cvg']
            if _gb_hit.get('bary_lin') is not None and \
                    gbuf.bary_lin is not None:
                gbuf.bary_lin[:] = _gb_hit['bary_lin']
            _GBUF_STATS['hits'] += 1
            _GBUF_STATS['last'] = 'HIT'
            # keep the per-resolution last-seen key current, so the NEXT
            # miss diffs against what actually rendered last
            _GBUF_STATS.setdefault('lastkey', {})[(rw, rh)] = gbuf_ckey
        rastered_from_cache = True

    # The compute rasteriser: the CPU's own fill rules on the GPU, measured
    # at ZERO differing pixels on hardware. Strictly opt-in and strictly
    # qualified -- anything it does not reproduce rasterises on the CPU
    # exactly as before, with the reason printed. Whole-frame only.
    rastered_on_gpu = False
    _gate_why = None
    if not rastered_from_cache and \
            str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
            getattr(st, 'gpu_raster', False):
        # R251: the two refusals left are printed by name (Painter's
        # now rides the kernel through the flat depth, C004)
        _gate_why = 'a banded frame (workers own their rows)' \
            if band is not None else \
            ('the OVERDRAW census is a sequential count'
             if want_overdraw else None)
        if _gate_why is not None:
            print(f'[Halcyon GPU] rasterising on the CPU: {_gate_why}')
    if not rastered_from_cache and \
            str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
            getattr(st, 'gpu_raster', False) and _gate_why is None:
        # affine frames run the kernel's lin variant (a third image
        # carries the screen-linear barycentrics, fill()'s bary_lin);
        # quantised-depth and snapped frames run with the RASTER TIE
        # REFERRAL: the kernel marks every decision inside a
        # cross-device noise window (a depth within an ulp-wobble of a
        # quantisation boundary, coincident surfaces on shared steps, a
        # pixel centre on a snapped edge) and the marked pixels are
        # replayed with the CPU fill's own arithmetic -- the ray
        # referral's cure, at the raster. A frame whose fragile pixels
        # exceed the referral budget falls back whole, with the count
        # in the printed reason.
        from ..gpu import craster as _craster
        with ST.track('rasterise (GPU)'):
            # the GPU crossings live at the device boundary now
            # (gpu/device.py _main): the CPU halves of this call
            # never block the interface, the driver halves cross
            # as millisecond bursts
            if bool(getattr(opts, 'cvg', False)) and gbuf.cvg is None:
                # C001: the kernel writes the coverage plane only into a
                # plane that EXISTS (craster.raster_into_gbuffer's
                # want_cvg); the CPU fill allocates its own. For want of
                # this line a GPU-rastered N64 frame had no coverage on
                # the driver: the VI filter was skipped on the GPU device
                # (5512 of 6912 px wrong), and the plane-less G-buffer
                # was cached for the next CPU frame of the same key
                gbuf.alloc_cvg()
            try:
                ok_r, why_r = _craster.raster_into_gbuffer(
                    mesh, vp, rw, rh, gbuf, cull=cull, snap=snap,
                    depth_bits=st.depth_precision, subset=opaque,
                    subdiv_px=subdiv_px, near_eps=near_eps,
                    flat_depth=flat_depth, opts=opts)
            except Exception as exc:                            # noqa: BLE001
                ok_r, why_r = False, str(exc)
        if ok_r:
            rastered_on_gpu = True
        else:
            print(f'[Halcyon GPU] rasterising on the CPU: {why_r}')
    if not rastered_from_cache and not rastered_on_gpu:
        with ST.track('rasterise'):
            raster.rasterize(mesh.verts, mesh.tris, vp, rw, rh, cull=cull,
                             snap=snap, depth_bits=st.depth_precision,
                             subset=opaque, gbuf=gbuf,
                             count_overdraw=want_overdraw,
                             flat_depth=flat_depth, scissor=scissor,
                             batched=False if want_overdraw else None,
                             subdiv_px=subdiv_px, near_eps=near_eps,
                             opts=opts)
    if gbuf_ckey is not None and not rastered_from_cache:
        # store copies -- the live gbuf is written by later stages
        # (gpu_alpha, wireframe) and must never alias the cache
        _GBUF_STATS['misses'] += 1
        # name WHICH component moved since the last key seen at this
        # resolution (draft and refine sizes tracked apart)
        _prev = _GBUF_STATS.setdefault('lastkey', {}).get((rw, rh))
        if _prev is None:
            _GBUF_STATS['last'] = 'MISS(first)'
        else:
            _names = ('mesh', 'vp', 'rw', 'rh', 'cull', 'snap', 'depth',
                      'subdiv', 'eps', 'lin', 'subset', 'opts')
            _diff = [n for n, a, b in zip(_names, _prev, gbuf_ckey)
                     if a != b]
            if 'vp' in _diff:
                # R172: name WHICH matrix elements moved and by how
                # much. The field pastes MISS(vp) on an untouched view;
                # 'm03+2ulp' is float jitter to hunt down, 'm03+1.2e-1'
                # is a real move (the user navigating) -- one word
                # settles which
                try:
                    _a = np.frombuffer(_prev[1],
                                       np.float32).reshape(4, 4)
                    _b = np.frombuffer(gbuf_ckey[1],
                                       np.float32).reshape(4, 4)
                    _rr, _cc = np.nonzero(_a != _b)
                    _els = []
                    for _r0, _c0 in list(zip(_rr, _cc))[:4]:
                        _d = float(_b[_r0, _c0]) - float(_a[_r0, _c0])
                        _sp0 = float(np.spacing(np.abs(
                            _a[_r0, _c0]) + np.float32(1e-30)))
                        _ul = abs(_d) / max(_sp0, 1e-45)
                        _sgn = '+' if _d >= 0 else '-'
                        _els.append(
                            f'm{_r0}{_c0}{_sgn}'
                            + (f'{_ul:.0f}ulp' if _ul < 1000
                               else f'{abs(_d):.1e}'))
                    if len(_rr) > 4:
                        _els.append(f'+{len(_rr) - 4}')
                    _diff[_diff.index('vp')] = \
                        'vp[' + ','.join(_els) + ']'
                except Exception:                   # noqa: BLE001
                    pass
            _GBUF_STATS['last'] = \
                'MISS(' + ('+'.join(_diff) or 'evicted') + ')'
        _GBUF_STATS['lastkey'][(rw, rh)] = gbuf_ckey
        while len(_GBUF_STATS['lastkey']) > 8:      # resize churn bound
            _GBUF_STATS['lastkey'].pop(
                next(iter(_GBUF_STATS['lastkey'])))
        _ent = {
            'tri': gbuf.tri.copy(), 'bary': gbuf.bary.copy(),
            'depth': gbuf.depth.copy(), 'zndc': gbuf.zndc.copy(),
            'front': gbuf.front.copy(),
            'zkey': None if gbuf.zkey is None else gbuf.zkey.copy(),
            'cvg': None if gbuf.cvg is None else gbuf.cvg.copy(),
            'bary_lin': None if gbuf.bary_lin is None
            else gbuf.bary_lin.copy()}
        _nb = _gbuf_entry_bytes(_ent)
        if not _GBUF_CACHE:
            _GBUF_STATS['bytes'] = 0    # resync after an outside clear()
        while _GBUF_CACHE and (
                len(_GBUF_CACHE) >= _GBUF_CAP
                or _GBUF_STATS['bytes'] + _nb > _GBUF_BUDGET_BYTES):
            _gbuf_cache_pop(next(iter(_GBUF_CACHE)))
        _GBUF_CACHE[gbuf_ckey] = _ent
        _GBUF_STATS['bytes'] += _nb

    # ---- R211 punch-through: CLIP materials resolve their alpha here,
    # against the freshly rastered opaque depth, and the survivors JOIN
    # the depth-buffered pass -- shaded once, never layered. The blend
    # road below only ever sees what stayed translucent. (Runs after
    # the g-buffer cache stores its clean opaque copy: promotion is
    # deterministic and re-applies on every hit.)
    if frags is not None and transparent is not None and transparent.size:
        clip_sub, transparent, _clip_plans = _clip_partition(
            scene, mesh, st, transparent,
            affine=gbuf.bary_lin is not None)
        if clip_sub is not None and clip_sub.size:
            with ST.track('punch-through'):
                _promote_clip(job, gbuf, vp, st, clip_sub, _clip_plans,
                              snap, flat_depth, scissor, subdiv_px,
                              near_eps, ckey=gbuf_ckey)

    # R251 shadow pack: the shadow-mask bake -- AFTER the R211 promotion
    # (it rewrites gbuf.tri/zndc at every promoted CLIP pixel) and never
    # after the _GBUF_CACHE store alone. One caster raster per PLANAR
    # lamp, one fragment capture per authored volume, on BOTH roads
    # (the beauty G-buffer is bitwise the CPU's on the GPU road too);
    # (None, None) when no planar lamp and no volume exists
    from .shadowmask import build as _mask_build
    _pl = st.shadows and (st.shadow_default == 'PLANAR' or (
        st.shadow_default == 'PER_LIGHT' and any(
            getattr(l, 'shadow', 'MAP') == 'PLANAR' for l in scene.lights)))
    _kt = _caster_keep_tri(scene, mesh) if _pl else None
    with ST.track('shadow mask bake'):
        st._mask_pack, st._mask_params = _mask_build(
            scene, st, gbuf, vp, snap, near_eps,
            None if _kt is None else np.nonzero(_kt)[0],
            scissor=scissor, view=view, eye=eye)

    if band is None and not getattr(st, '_viewport', False):
        # the two numbers behind "the depth is screwed up": what the
        # z-buffer can resolve on THIS frame, and which surfaces were
        # taken out of the depth-buffered pass altogether. Viewport frames
        # skip the report -- at several drafts a second it is not an
        # instrument any more, it is a firehose; F12 still prints it
        _rep = depth_report(proj, gbuf, st.depth_precision,
                            getattr(st, 'depth_sort', 'ZBUFFER'))
        if _rep:
            print(_rep)
        if getattr(st, 'fog', False):
            # R205: when the whole subject sits past Fog End, the frame
            # comes out as flat fog colour -- say so with the numbers,
            # instead of leaving "the fog is not right" to be diagnosed
            # from a beige rectangle
            _note = fog_coverage_note(proj, gbuf, st)
            if _note:
                print(_note)
        if rastered_on_gpu:
            # the split behind the breakdown's one 'rasterise (GPU)'
            # number: at high resolutions the interesting question is
            # WHICH half of the road got slow, and a single bucket
            # cannot answer it
            from ..gpu.craster import LAST_RASTER as _LR
            if _LR:
                print('[Halcyon GPU] raster split: '
                      + ', '.join(f'{k[:-3].replace("_", "+")} '
                                  f'{v:.1f} ms'
                                  for k, v in _LR.items()))
        _shadow_coarseness_note(scene, gbuf, proj, rh)
        _sp = LAST_SPLIT
        if _sp.get('see_through'):
            named = '; '.join(f'{n}: {w}' for n, w
                              in list(_sp['reasons'].items())[:6])
            more = len(_sp['reasons']) - 6
            print(f"[Halcyon] transparency: {_sp['see_through']} of "
                  f"{_sp['materials']} materials are see-through "
                  f"({_sp['tris_see_through']} of {_sp['tris']} "
                  f'triangles) -- {named}'
                  + (f' (+{more} more)' if more > 0 else ''))
            if opaque is not None and opaque.size == 0:
                print('[Halcyon] transparency: NOTHING is in the '
                      'depth-buffered pass -- every surface stacks as '
                      'A-buffer layers instead, so solid geometry can '
                      'show its own back faces. If these materials are '
                      'meant to be solid, set their blend mode to '
                      'Opaque (or Transparency to None) and the frame '
                      'goes back through the z-buffer')
        _cl = LAST_CLIP
        if _cl.get('materials'):
            print('[Halcyon] punch-through: '
                  f"{_cl['materials']} material(s) resolved "
                  f"{_fmt_frags(_cl.get('fragments', 0))} fragments in "
                  'the z-pass '
                  f"({_fmt_frags(_cl.get('kept', 0))} kept, "
                  f"{_fmt_frags(_cl.get('promoted', 0))} pixels won) -- "
                  'those layers never reach the blend road'
                  + (f"; cache {_cl['cache']}"
                     if _cl.get('cache') in ('HIT', 'MISS') else ''))
        for _nm in (_cl.get('auto') or ()):
            print(f"[Halcyon] punch-through: '{_nm}' has provably "
                  'binary alpha (its chain ends in a comparison) -- '
                  'promoted off the blend road automatically; the '
                  'picture is identical, the layers and their cap '
                  'are not paid')
        for _nm, _why in (_cl.get('refused') or {}).items():
            print(f"[Halcyon] punch-through: '{_nm}' stays on the "
                  f'blend road -- {_why}')

    if progress:
        progress(0.35, 'Shading')

    covered = gbuf.mask()
    if band is not None:
        keep = np.zeros(rh, bool)
        keep[max(band[0], 0) * ss:min(band[1], H) * ss] = True
        if ink_reach and (str(getattr(st, 'ink_color_mode', 'FIXED')
                                  ).upper() == 'FILL'
                          or _scene_has_iro(scene)
                          or _paint_reach_rows(st) > 0
                          or _gou_on
                          or getattr(st, 'outline_tone', False)):
            # a From Fill line (R229: or an Iro-Trace material's) reads
            # the SHADED colour of the surface pixel that owns it, which
            # may sit in the context rows: shade those too, so the
            # band's line is the frame's
            keep[max(max(band[0], 0) * ss - ink_reach, 0):
                 min(min(band[1], H) * ss + ink_reach, rh)] = True
        covered &= keep[:, None]
    _rm = getattr(st, '_refine_mask', None)
    if _rm is not None:
        # an adaptive-AA refine pass: shade ONLY the flagged edge
        # pixels. The raster above still ran whole-frame (the jittered
        # projection decides anew which pixels geometry covers) and the
        # background below still fills every uncovered pixel (a masked
        # sky pixel needs its correctly-jittered sky) -- what the mask
        # cuts is the expensive part, per-fragment shading. Everything
        # this pass computes outside the mask is discarded by the
        # accumulation in _adaptive_refine, so restricting `covered`
        # here changes no masked pixel's value
        covered &= _rm
    if _keep2 is not None:
        # R253: the region -- shade the rect and its context ring only;
        # every pixel inside is a pure function of the same G-buffer and
        # lamps the whole frame shades from, so the rect's values are
        # the whole frame's (test_r253_region pins it bitwise)
        covered &= _keep2
    _bg_un = (~covered) & (keep[:, None] if band is not None else True)
    if _keep2 is not None:
        _bg_un = _bg_un & _keep2
    if _rm is not None:
        # a refine pass keeps only its flagged pixels; sky computed
        # anywhere else is discarded by the accumulation, so it is not
        # computed. The values at flagged sky pixels are per-pixel
        # independent and therefore identical to a full evaluation --
        # this took the field's sky bucket from a full evaluation per
        # pass to a sliver
        _bg_un = _bg_un & _rm
    _gpu_road = str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' \
        and bool(st.gpu_shading) and band is None

    def _cpu_sky():
        with ST.track('background / sky'):
            return _background_image(scene, st, rw, rh, vp, eye, _bg_un,
                                     textures, ss=ss)

    # R250: on the GPU road the sky is drawn in the shading burst (the
    # readback is then the whole frame) and the CPU evaluates it only
    # when that pass refused, AFTER shading -- order-free, because the
    # shading never writes an uncovered pixel and the sky never a covered
    # one. The CPU road evaluates it here, exactly as before
    img = None if _gpu_road else _cpu_sky()
    # (volumetric light beams composite AFTER shading now -- they are in
    # front of geometry, and drawing them here had the shading loop's
    # img[py, px] assignment overwrite every beam pixel that crossed a
    # surface: the field's beams only ever showed against the sky)

    py, px = np.nonzero(covered)
    # (R250: st._frame_gpu_shaded is reset at the top of the frame now,
    # before the geometry-less early return)
    if py.size:
        # Deferred GPU shading: the G-buffer just rasterised is shaded in a
        # full-screen pass per material, on the same mechanism the post
        # the interpolated radiosity field, LAZILY: only actual CPU
        # shading (full frames, bands, routed layers, the A-buffer)
        # builds it -- a fully-GPU frame never pays the CPU grid, and
        # the probe never shades through light_surface at all. The
        # builder is deterministic, so whichever caller gets there
        # first computes the same numbers any other would.
        if getattr(st, 'radiosity', False) and bvh is not None and \
                int(getattr(st, 'radiosity_spacing', 1) or 1) > 1:
            import threading as _threading
            _rad_lock = _threading.RLock()
            _rad_box = {}

            def _lazy_field(job=job, gbuf=gbuf):
                # REENTRANT with a building sentinel: the builder's own
                # context() call comes back through here on the same
                # thread, and must see "no field yet", not a deadlock
                with _rad_lock:
                    if 'f' in _rad_box:
                        return _rad_box['f']
                    if _rad_box.get('busy'):
                        return None
                    _rad_box['busy'] = True
                    try:
                        _rad_box['f'] = radiosity_field(job, gbuf, st,
                                                        scene, bvh)
                    finally:
                        _rad_box['busy'] = False
                    return _rad_box['f']

            job.radiosity_lazy = _lazy_field
        # R238: the cel field -- the screen shadow and the depth rim of
        # the cel materials, one whole-frame pass over the G-buffer's own
        # depth, computed on first use (the GPU road builds it before its
        # passes and uploads it; a frame with no cel dial reading it
        # never pays). Deterministic: whoever gets there first computes
        # the same numbers any other would.
        from . import celfield as _CF
        if _CF.field_on(scene, st):
            import threading as _cthreading
            _cel_lock = _cthreading.RLock()
            _cel_box = {}

            def _lazy_cel(job=job, gbuf=gbuf, view=view, proj=proj, vp=vp):
                with _cel_lock:
                    if 'f' in _cel_box:
                        return _cel_box['f']
                    if _cel_box.get('busy'):
                        return None
                    _cel_box['busy'] = True
                    try:
                        with ST.track('cel field'):
                            _cel_box['f'] = _CF.compute(
                                scene, mesh, gbuf, view, proj, vp,
                                getattr(scene, 'camera', None), st)
                    finally:
                        _cel_box['busy'] = False
                    return _cel_box['f']

            job.cel_lazy = _lazy_cel
        # stages run on. Strictly opt-in, and strictly qualified -- a frame
        # using anything the GLSL does not reproduce shades on the CPU
        # exactly as before, with the reason printed rather than guessed
        # around. Whole-frame only: a worker band re-splits the arithmetic
        # the GPU would do in one pass anyway.
        shaded_on_gpu = False
        if str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
                st.gpu_shading and band is None:
            # Adaptive refine passes (_rm set) take this branch too --
            # 1.38.0 gated them onto the CPU, reasoning that a
            # full-screen pass over a few percent of a frame was
            # overhead. The field measured the truth: the CPU pays for
            # rays, bump fields and texture footprints PER PIXEL at
            # ~130us each while the GPU's whole-frame pass is seconds
            # flat -- three CPU refine passes cost 306s of a 317s
            # frame. The full-screen draw shades everything and the
            # masked assignment below keeps only the flagged pixels;
            # base and refines now also shade on the SAME device
            LAST_GPU_VERDICT.update(wanted=True, engaged=False, why='')
            from ..gpu import shade as _gpu_shade
            with ST.track('shade (GPU)'):
                try:
                    # R253: a region frame scissors the material passes
                    # and the readback to the rect's box (pure transport,
                    # the per-rank scissor road); the readback comes back
                    # zero-padded to the frame
                    got, why = _gpu_shade.shade_frame(
                        job, gbuf, region=_region_box_px)
                except Exception as exc:                        # noqa: BLE001
                    got, why = None, str(exc)
            if got is None:
                LAST_GPU_VERDICT['why'] = str(why or '')
                print(f'[Halcyon GPU] shading on the CPU: {why}')
                try:
                    # the field's missing WHY: five GPU-device sessions
                    # showed CPU shading inside GPU frames and the log
                    # never said what refused. Now it does
                    from .. import fault_note
                    fault_note(f'GPU plan refused: {why}'[:220],
                               key='plan-no', limit=3)
                except Exception:                               # noqa: BLE001
                    pass
            else:
                ga = getattr(gbuf, 'gpu_alpha', None)
                _stip = st.transparency == 'STIPPLE' and ga is not None
                _res = getattr(gbuf, 'gpu_frame', None)
                if getattr(gbuf, 'gpu_sky', False):
                    # R250: the sky drew in the burst -- the readback IS
                    # the frame. rgb whole (the sky in every uncovered
                    # pixel, the materials in every covered one); the
                    # alpha plane is the frame's own law: covered 1.0 --
                    # or the Screen Door's decoded 0/1 bit -- uncovered
                    # the film's (0 under a transparent film). No
                    # per-pixel copy: that copy was a bucket of its own
                    _cov_all = gbuf.mask()
                    _bg_a = np.float32(0.0 if getattr(st, 'film_transparent',
                                                       False) else 1.0)
                    _rgba = getattr(gbuf, 'gpu_frame_rgba', None)
                    if _rgba is not None and _rgba.shape[0] == rh and \
                            _rgba.shape[1] == rw and _rgba.shape[2] == 4:
                        img = _rgba
                    else:
                        img = np.empty((rh, rw, 4), np.float32)
                        img[:, :, :3] = got
                    if _stip:
                        img[:, :, 3] = np.where(_cov_all,
                                                ga.astype(np.float32), _bg_a)
                    else:
                        img[:, :, 3] = np.where(_cov_all, np.float32(1.0),
                                                _bg_a)
                    if _res is not None:
                        if img is _rgba and not _stip:
                            # the target on the GPU equals img byte for
                            # byte: it stays there for the ink, the
                            # resolve and the post chain
                            _FR.install(st, _res, 'shade')
                        else:
                            _res.release()
                else:
                    # the sky pass refused by name: the CPU draws the sky
                    # now (after the shading; the pixels are the same)
                    if _res is not None:
                        _res.release()
                    if img is None:
                        img = _cpu_sky()
                    img[py, px, :3] = got[py, px]
                    if _stip:
                        # the Screen Door: rgb is the full shaded colour
                        # and the ordered 0/1 pattern rides the alpha --
                        # exactly the (rgb shaded, alpha stippled) split
                        # _shade writes on the CPU
                        img[py, px, 3] = ga[py, px].astype(np.float32)
                    else:
                        img[py, px, 3] = 1.0
                    _why_sky = str(getattr(gbuf, 'gpu_sky_why', '') or '')
                    if _why_sky:
                        from ..gpu import sky as _GSKY
                        _GSKY._warn(_why_sky)
                shaded_on_gpu = True
                LAST_GPU_VERDICT['engaged'] = True
                try:
                    from .. import fault_note
                    fault_note('GPU shading engaged', key='plan-yes',
                               limit=3)
                except Exception:                               # noqa: BLE001
                    pass
                if band is None and _rm is None and \
                        not getattr(st, '_viewport', False):
                    # the split behind the one 'shade (GPU)' number --
                    # (refine passes shade through here too but stay
                    # quiet: three extra splits per frame would drown
                    # the one that describes the base picture) --
                    # a 44-second shade bucket cannot be attacked until
                    # plan, upload, draw, sweeps and composite each own
                    # their milliseconds, and the pass count is printed
                    # (an M-material frame pays M full-screen passes)
                    from ..gpu.shade import LAST_TIMINGS as _LT
                    if _LT:
                        keys = ('plan_ms', 'pack_upload_ms',
                                'draw_read_ms', 'reflect_ms',
                                'ray_build_ms', 'composite_ms')
                        parts = ', '.join(
                            f"{k[:-3].replace('_', '+')} "
                            f'{float(_LT.get(k, 0.0)):.1f} ms'
                            for k in keys if _LT.get(k))
                        if float(_LT.get('composite_ms', 0.0)) > 1000.0:
                            # R182/R183: a composite over a second names
                            # its measured parts; 'other' is the hunt
                            # list, and the height pre-passes own their
                            # milliseconds (cpu count called out)
                            _pw = str(_LT.get('prepass_whys', '') or '')
                            parts += (
                                ' (composite: own '
                                f"{float(_LT.get('c_own_ms', 0.0)):.0f}"
                                ' + env '
                                f"{float(_LT.get('c_env_ms', 0.0)):.0f}"
                                ' + fog '
                                f"{float(_LT.get('c_fog_ms', 0.0)):.0f}"
                                ' + prepass '
                                f"{float(_LT.get('prepass_ms', 0.0)):.0f}"
                                f" [{int(_LT.get('prepass_n', 0))}x, "
                                f"{int(_LT.get('prepass_cpu', 0))} cpu"
                                + (f' -- {_pw}' if _pw and
                                   int(_LT.get('prepass_cpu', 0)) else '')
                                + ']'
                                ' + other '
                                f"{float(_LT.get('c_other_ms', 0.0)):.0f}"
                                ')')
                        if _LT.get('burst_draw_ms') or \
                                _LT.get('burst_read_ms'):
                            # R175b: inside the burst crossing --
                            # submission overhead vs the readback wait
                            # (the GPU actually executing)
                            parts += (
                                ' (burst: submit '
                                f"{float(_LT.get('burst_draw_ms', 0.0)):.0f}"
                                ' + gpu/read '
                                f"{float(_LT.get('burst_read_ms', 0.0)):.0f}"
                                ')')
                        if _LT.get('compile_ms'):
                            # cold frames are COMPILATION, and the line
                            # says so instead of hiding it in composite
                            parts += (
                                f"; shader compile "
                                f"{int(_LT.get('compile_n', 0))}x "
                                f"{float(_LT.get('compile_ms', 0.0)):.0f}"
                                ' ms (first frame per plan; cached '
                                'after)')
                        print(f'[Halcyon GPU] shade split: {parts}; '
                              f"{int(_LT.get('passes', 0))} material "
                              f'pass(es)')
                        # R250: the sky's verdict and the frame's
                        # residency, on the same line the field reads
                        _sky_m = int(_LT.get('sky', -1))
                        _sky_names = {0: 'transparent film (zeros)',
                                      1: 'flat colour', 2: 'sky blend',
                                      3: 'environment image',
                                      4: 'gradient', 5: 'bands', 6: 'HDRI',
                                      # R251 / R253: named, not numbered
                                      7: 'cylinder sky',
                                      8: 'gradient backdrop',
                                      9: 'cube map'}
                        if _sky_m >= 0:
                            print('[Halcyon GPU] sky: '
                                  f"{_sky_names.get(_sky_m, _sky_m)} drawn "
                                  'in the shading burst; frame '
                                  + ('kept on the GPU'
                                     if _LT.get('resident') else
                                     'read back (' + (
                                         'Screen Door alpha'
                                         if st.transparency == 'STIPPLE'
                                         else 'a CPU composite edits it')
                                     + ')'))
                        if _LT.get('reflect_ms'):
                            print(
                                '[Halcyon GPU] reflect split: trace '
                                f"{float(_LT.get('reflect_trace_ms', 0.0)):.1f} ms, "
                                'secondary draws '
                                f"{float(_LT.get('reflect_draw_ms', 0.0)):.1f} ms, "
                                'sky along misses '
                                f"{float(_LT.get('reflect_env_ms', 0.0)):.1f} ms; "
                                f"{int(_LT.get('reflect_levels', 0))} level(s), "
                                f"{int(_LT.get('reflect_skips', 0))} all-miss "
                                'level(s) skipped, '
                                f"{int(_LT.get('reflect_rays', 0))} ray(s)")
        # what post.process gates its GPU stages on: a frame whose
        # shading ran on the CPU keeps its post on the CPU too. The GPU
        # post pass over a CPU-resident frame was always upload +
        # readback overhead for nothing -- and in the field it was the
        # ONLY GPU work in five sessions that ended in a silent device
        # loss right after the frame parked
        st._frame_gpu_shaded = shaded_on_gpu
        if not shaded_on_gpu:
            if img is None:
                img = _cpu_sky()
            if getattr(st, 'radiosity', False) and bvh is not None and \
                    band is None and not getattr(st, '_viewport', False):
                # the cost, named before it is paid: the field's first
                # radiosity frame took 118 seconds because one refused
                # material put the WHOLE frame -- and the gather with
                # it -- on the CPU. Say what is about to happen and
                # which sliders own the price.
                _n = max(int(getattr(st, 'radiosity_spacing', 1) or 1), 1)
                rays = int(np.count_nonzero(gbuf.mask())) * \
                    max(int(st.radiosity_samples), 1) // (_n * _n)
                amount = f'~{rays / 1e6:.1f}M' if rays >= 1e6 else \
                    f'~{rays / 1e3:.0f}K'
                print(f'[Halcyon] radiosity gathers on the CPU: '
                      f'{amount} rays this frame. The GPU device runs '
                      f'the gather in-shader (~40x); Gather Samples and '
                      f'Gather Distance set the CPU cost')
            tri_idx = gbuf.tri[py, px]
            bary = gbuf.bary[py, px]
            blin = gbuf.bary_lin[py, px] if gbuf.bary_lin is not None else None
            front = gbuf.front[py, px]
            if (band is not None or _rm is not None) and has_bump:
                # a band shades only its rows, but n_bump's gradients
                # need the CONTEXT row the extended scissor rasterised:
                # hand the field builder the G-buffer's FULL coverage,
                # so a band's gradients equal the whole frame's. An
                # adaptive refine pass has the same shape -- a shaded
                # SUBSET whose gradients must equal the full frame's --
                # so it hands over full coverage for the same reason
                fpy, fpx = np.nonzero(gbuf.mask())
                job.bump_field_source = (
                    gbuf.tri[fpy, fpx], gbuf.bary[fpy, fpx], fpx, fpy,
                    gbuf.front[fpy, fpx],
                    gbuf.bary_lin[fpy, fpx]
                    if gbuf.bary_lin is not None else None)
            job.pass_sink_armed = True       # R253: the opaque frame pass
            with ST.track('shade'):
                col = _shade_all(job, tri_idx, bary, px, py, front, blin, st,
                                 progress=progress)
            job.pass_sink_armed = False
            img[py, px, :3] = col[:, :3]
            img[py, px, 3] = np.maximum(img[py, px, 3], col[:, 3])

    if img is None:
        # nothing was covered: the whole frame is sky, drawn on the CPU
        img = _cpu_sky()

    # R251 shadow pack: the mask applied over the finished opaque frame,
    # the same float32 numpy rule on both roads (the frame leaves the GPU
    # for it, as under fog)
    img = SM.apply_after_readback(img, job, st)

    if rect is not None:
        # R253: the scissored target holds the rect AND its context ring
        # shaded; the ring is cut on the CPU side only (_apply_region at
        # the exit), so a resident frame would feed the GPU ink, resolve
        # and post chain pixels the CPU road zeroes -- a device-parity
        # hole in every non-local stage. Released by name: the post chain
        # uploads the zero-padded frame once and _post_warn says why
        _FR.edited(st, 'render region', 'the context ring is cut on the CPU')

    if 'Env' in wanted_passes(st):
        # R253: the Env pass -- the sky where the camera saw it (on the
        # GPU road the readback's sky pixels, the measured twin), taken
        # BEFORE the wires, the ink, the halos and the lights draw over it
        _env_rgb = np.where(gbuf.mask()[:, :, None], np.float32(0.0),
                            img[:, :, :3]).astype(np.float32)

    if _wire_active(scene, st, gbuf):
        _FR.edited(st, 'wireframe')
    with ST.track('wireframe'):
        img = apply_wireframe(job, gbuf, img, st, vp, eye, textures)

    if _gou_on:
        _FR.edited(st, 'painted backgrounds')
        # R233: the painted background road -- strokes over the
        # Background materials and the setback's softness, before the
        # ink so the cels' lines land crisp over the painting. One CPU
        # pass from the shared G-buffer on either device road
        with ST.track('painted backgrounds'):
            img = GOU.apply(scene, gbuf, img, st, vp, view, eye,
                            frame=int(getattr(scene, 'frame', 0) or 0),
                            seed=int(getattr(st, 'seed', 0) or 0),
                            ss=float(gbuf.tri.shape[0])
                            / float(max(int(st.resolution_y), 1)))

    if getattr(st, 'outline', False) or _ink_forced(scene):
        with ST.track('outline'):
            img = apply_outline(scene, gbuf, img, st, vp, proj=proj,
                                eye=eye)

    with ST.track('ground plane over geometry'):
        img = _plane_over_geometry(img, scene, st, job, gbuf, eye, textures)

    if getattr(scene, 'halos', None) and band is None:
        # exactly BI's tile order: solid z-buffer, then halos against
        # it, then the transparent faces composite OVER the halos.
        # Never on a band: a band's buffer starts at its own y and the
        # frame-coordinate splats would land shifted (the pool skips
        # halo scenes; this guard keeps a direct banded call honest)
        _FR.edited(st, 'halos')
        with ST.track('halos'):
            img = _draw_halos(img, scene, st, gbuf, view, proj, rw, rh,
                              sel_mask=_rm)

    if transparent is not None and transparent.size and frags is not None:
        if progress:
            progress(0.65, 'Transparency')
        # R251 C126 (Blender 2.4x Zoffs): one raster call per distinct Z
        # Offset among the see-through materials, nearer = smaller depth,
        # applied at COLLECTION (the keep test against the opaque depth
        # runs on the offset z, so a decal a little behind its surface
        # is captured); every offset zero is ONE call with z_offset 0.0
        _zo = np.array([np.float32(getattr(m, 'z_offset', 0.0))
                        for m in (scene.materials or [None])], np.float32)
        zo_all = (_zo[np.clip(mesh.mat_index[transparent], 0, _zo.size - 1)]
                  if mesh.mat_index is not None
                  else np.zeros(transparent.size, np.float32))
        # the raster's depth is the buffer's own (ndc z), so the slider
        # converts through the camera's clip range exactly as 2.4x's
        # zbuf did (polygon_offset = zoffs * INT_MAX / (clipend - clipsta))
        _zo_scale = zoffs_scale(scene.camera)
        for zo in np.unique(zo_all):
            raster.rasterize(mesh.verts, mesh.tris, vp, rw, rh, cull='NONE', snap=snap,
                             depth_bits=st.depth_precision,
                             subset=transparent[zo_all == zo],
                             gbuf=gbuf, frags=frags, depth_write=False,
                             flat_depth=flat_depth, scissor=t_scissor,
                             subdiv_px=subdiv_px, near_eps=near_eps,
                             opts=opts,
                             z_offset=(-float(np.float32(zo) * _zo_scale)
                                       if zo != 0.0 else 0.0))
        _FR.edited(st, 'transparency')
        with ST.track('transparency'):
            img = _composite_abuffer(job, frags, gbuf, img, st, band=band,
                                     vp=vp, snap=snap, rows=t_scissor)
    elif transparent is not None and transparent.size:
        raster.rasterize(mesh.verts, mesh.tris, vp, rw, rh, cull=cull, snap=snap,
                         depth_bits=st.depth_precision, subset=transparent,
                         gbuf=gbuf, flat_depth=flat_depth, scissor=t_scissor,
                         subdiv_px=subdiv_px, near_eps=near_eps,
                         opts=opts)

    if str(getattr(st, 'framebuffer', 'NONE')) != 'NONE' and not (
            transparent is not None and transparent.size and frags is not None):
        # R251: a 16-bit buffer truncates the opaque frame too -- with
        # nothing see-through the composite never ran, so pack it here
        _FR.edited(st, 'framebuffer format')
        with ST.track('transparency'):
            img = framebuffer_pack_frame(img, st, rows=t_scissor)

    if _beams_active(scene, st):
        _FR.edited(st, 'volumetric lights')
    with ST.track('volumetric lights'):
        img = _light_volumes(img, scene, st, gbuf, vp, eye, rw, rh, bvh,
                             sel_mask=_rm)

    with ST.track('volume containers'):
        # R222: real marched volumes -- containers composite over the
        # finished frame, depth-clipped, on BOTH devices (the outline
        # doctrine: one CPU pass over the shared buffers). R225: a
        # pooled BAND marches only its own rows -- every pixel's march
        # is a pure function of the pixel (whole-lane shadow draws per
        # (volume, step, lamp)), so the band's values are the whole
        # frame's, and N workers no longer march the frame N times
        from . import volume as VOL
        if VOL.scene_volumes(scene):
            _FR.edited(st, 'volume containers')
        _vol_sel = _rm
        if band is not None:
            _bm = np.zeros((rh, rw), bool)
            _bm[keep] = True
            _vol_sel = _bm if _rm is None else (_bm & _rm)
        if _keep2 is not None:
            # R253: the region marches its rect (and ring) only, the
            # band's own per-pixel rule
            _vol_sel = _keep2 if _vol_sel is None else (_vol_sel & _keep2)
        img = VOL.march(img, scene, st, gbuf, vp, eye, rw, rh, bvh,
                        sel_mask=_vol_sel, textures=textures, view=view)

    if band is None and _rm is None and getattr(job, 'unsupported', None):
        # R226: the CPU road's evaluator failures, BY NAME. The GPU road
        # refuses a material whose chain the evaluator cannot run and
        # says so; the CPU road collected the same notes and printed
        # NOTHING -- a node that raised shaded a plausible pass-through
        # in silence. Once per frame, never per band or refine pass
        msg = ('node evaluator: ' + '; '.join(sorted(job.unsupported)))
        print('[Halcyon] ' + msg)
        try:
            lst = getattr(scene, 'unsupported', None)
            if isinstance(lst, list) and msg not in lst:
                lst.append(msg)
        except Exception:                                       # noqa: BLE001
            pass

    if band is None and str(st.debug_pass) == 'BEAUTY' and \
            not getattr(st, '_pano_strip', False):
        # R200: the Weather overlay -- in front of geometry, halos,
        # beams and sky alike (the lens flares composite later still,
        # in post: a flare lives IN the lens, rain in the world).
        # Never on a band (frame-coordinate splats; the engine pool
        # skips weather scenes) and never per pano strip -- the
        # stitcher draws it once on the finished panorama
        wd_w = getattr(scene, 'world', None)
        if wd_w is not None and \
                str(getattr(wd_w, 'weather', 'NONE') or 'NONE') != 'NONE':
            from . import sky as SKY
            _FR.edited(st, 'weather')
            with ST.track('weather'):
                img = SKY.weather_overlay(
                    img, wd_w, rw, rh,
                    time=float(getattr(scene, 'time', 0.0)),
                    out_wh=(W, H), sel_mask=_rm)

    if st.debug_pass != 'BEAUTY':
        _FR.edited(st, 'debug pass')
        img = _debug_pass(job, gbuf, img, st)

    if progress:
        progress(0.85, 'Resolving')
    # hand the post chain what it needs for defocus and shafts
    # Extra passes come off the same G-buffer the beauty image did, before
    # anything downsamples or quantises it.
    depth_m = None
    _wp = wanted_passes(st)
    if st.dof or 'Depth' in _wp or 'Mist' in _wp or \
            str(st.aa_mode) in ('EDGE', 'ADAPTIVE'):
        with ST.track('linear depth'):
            depth_m = linear_depth(job, gbuf, eye)
    scene.last_passes = build_aux_passes(job, gbuf, st, depth_m,
                                         env_rgb=_env_rgb,
                                         sink=job.pass_sink, ss=ss,
                                         out_wh=(W, H))
    if scene.last_passes and ss > 1:
        # data, not colour: averaging a normal or an object index across
        # samples produces a value that was never on any surface, so the
        # top-left sample of each output pixel is taken instead
        # (R253: the COLOUR passes -- Env and the light split -- were
        # resolved through the beauty's own filter in build_aux_passes
        # and are at output size already; DATA_PASSES keep this rule)
        scene.last_passes = {k: (v[::ss, ::ss] if k in DATA_PASSES else v)
                             for k, v in scene.last_passes.items()}

    # R251 C001: the coverage plane for the post chain's VI stage (1
    # sample per pixel, else None -- the skip was named above)
    scene.last_cvg = gbuf.cvg if (getattr(opts, 'cvg', False) and ss == 1
                                  and gbuf.cvg is not None) else None
    scene.last_depth = None
    if st.dof and depth_m is not None:
        # metres, so Focus Distance in the UI is the distance it says it is.
        # It was normalised device depth, which put every focus value the
        # slider allows far behind the whole scene and blurred the lot.
        scene.last_depth = depth_m[::ss, ::ss] if ss > 1 else depth_m
    # R251 (C092): the coverage plane Super Black floors by -- the
    # G-buffer's own triangle mask, ANY sample of a supersampled pixel
    # counting as covered. Built only when the stage is on (the
    # reduction is a pass over every sample)
    scene.last_coverage = None
    if bool(getattr(st, 'super_black', False)):
        try:
            _cm = gbuf.mask()
            if ss > 1:
                _cm = _cm.reshape(_cm.shape[0] // ss, ss,
                                  _cm.shape[1] // ss, ss).any(axis=(1, 3))
            scene.last_coverage = np.ascontiguousarray(_cm)
        except Exception:                                   # noqa: BLE001
            scene.last_coverage = None
    st._last_coverage = scene.last_coverage
    # R251 C134: the material plane the optical printer's mattes are cut
    # from (an F12 stage: no plane for a pooled band or the viewport)
    scene.last_gel = None
    if getattr(st, 'matte_glow', False) and band is None \
            and not getattr(st, '_viewport', False):
        from . import signal_codec as _SC
        scene.last_gel = _SC.gel_plane(gbuf, job.scene.mesh,
                                       scene.materials, ss)
    scene.last_shafts = shaft_sources(scene, st, vp)
    scene.last_flares = _flare_sources(scene, st, gbuf, vp)

    if str(st.aa_mode) == 'EDGE':
        # the console flicker filter: find polygon edges on the id buffer
        # (and depth creases past the threshold) and soften ONLY those
        # pixels with a separable 1-2-1 tent -- the Dreamcast/PS2-era
        # "edge antialias" that smoothed silhouettes without paying for a
        # supersampled frame
        _FR.edited(st, 'edge smooth')
        with ST.track('edge smooth'):
            img = _edge_smooth(img, gbuf, st, depth_m)

    if str(st.aa_mode) == 'ADAPTIVE' and int(st.aa_samples) > 1 and \
            band is None and _rm is None and \
            getattr(st, '_accum_jitter', None) is None and \
            not getattr(st, '_viewport', False):
        # the Bryce/POV-Ray second pass: the frame above is the base
        # render, and only its EDGE pixels get re-sampled. Guards: a
        # refine pass must never recurse (_rm), a jittered pass is
        # already someone's sub-sample (_accum_jitter), a pooled band
        # cannot see across its seam (band -- the engine skips the pool
        # for this mode), and the viewport forces aa NONE anyway
        _FR.edited(st, 'adaptive AA')
        with ST.track('adaptive AA'):
            img = _adaptive_refine(scene, st, img, gbuf, mesh, depth_m,
                                   progress)

    with ST.track('resolve / downsample'):
        if band is not None:
            y0 = max(band[0], 0)
            y1 = min(band[1], H)
            return _apply_region(
                _resolve(img[y0 * ss:y1 * ss], W, y1 - y0, ss, st),
                rect, y_off=y0)
        out = None
        if ss > 1 and _FR.current(st) is not None:
            # R250: the supersample filter drawn on the GPU over the
            # resident frame; the readback is the OUTPUT frame, a quarter
            # of the bytes at ss 2, and the output target stays resident
            # for the post chain
            with ST.track('resolve (GPU)'):
                got_r, why_r = _FR.resolve(st, W, H, ss)
            if got_r is not None:
                out = got_r
            else:
                _FR.edited(st, 'resolve', str(why_r))
        if out is None:
            if ss <= 1 and getattr(st, 'aa_clamp_samples', False):
                # R251 C094: the single pass clips too, on the CPU --
                # the resident frame would feed the post chain unclipped
                _FR.edited(st, 'sample clamp at 1 sample')
            out = _resolve(img, W, H, ss, st)
    if rect is not None:
        # R253: the region's contract at the frame's exit -- zeros (RGBA
        # 0,0,0,0) outside the rect, the full-frame shape kept so the
        # post chain's patterns stay anchored to the frame's origin; the
        # context ring shaded above is cut here (the GPU target that held
        # it was released by name after the readback)
        out = _apply_region(out, rect)
    if not getattr(st, '_keep_gpu_frame', False):
        # a caller that wants the frame kept on the GPU for its post
        # chain says so and releases it itself, in a finally; every
        # other caller (the suite, a script) can never leak a target
        _FR.release(st)
    return out


def bake_lightmap(scene, st, obj_index=None, size=512, mode='COMBINED',
                  margin=4, progress=None):
    """R192: bake lighting into UV space -- the era's lightmap road.

    Rasterises the object's active-UV layout through the engine's own
    rasteriser (UVs as an orthographic vertex position: the seams,
    coverage rules and barycentrics are exactly the frame's), then
    shades every covered texel's WORLD position with the ordinary CPU
    shading core -- same lamps, same shadow maps and rays, same
    material graphs, same deterministic per-texel sampling streams --
    with the view pinned along the surface normal, exactly what 2.79's
    Full Render bake did (a lightmap must not freeze one camera's
    speculars into the wall). AO mode bakes the ambient-occlusion
    term alone, white where open. `margin` texels of edge dilation
    keep bilinear lookups from bleeding the empty background in.

    Returns ((size, size, 4) float32 bottom-row-first, None) or
    (None, why-string). Pure core -- the operator wraps it.
    """
    mesh = scene.mesh
    if mesh is None or mesh.tris is None or not len(mesh.tris):
        return None, 'the scene has no geometry to bake'
    if mesh.uvs is None:
        return None, 'the object has no UV layer to bake into'
    if obj_index is not None and mesh.obj_index is not None:
        sel = np.nonzero(mesh.obj_index == int(obj_index))[0]
        if sel.size == 0:
            return None, 'the object has no faces in the scene'
    else:
        sel = None
    size = max(int(size), 8)

    from .nodeeval import desugar_master_bump
    for _m in (scene.materials or ()):
        desugar_master_bump(getattr(_m, 'graph', None))
    collect_exclusive_lights(scene)
    textures = prepare_textures(scene, st)
    _build_shadows(scene, st, mesh)
    bvh = None
    try:
        bvh = _cached_bvh(scene, mesh)
    except Exception:                                           # noqa: BLE001
        bvh = None

    # the UV layout as geometry: u,v in [0,1] -> NDC through identity
    uv = np.asarray(mesh.uvs, np.float32)
    verts_uv = np.zeros((uv.shape[0], 3), np.float32)
    verts_uv[:, 0] = uv[:, 0] * 2.0 - 1.0
    verts_uv[:, 1] = uv[:, 1] * 2.0 - 1.0
    g = raster.GBuffer(size, size)
    raster.rasterize(verts_uv, mesh.tris, np.eye(4, dtype=np.float32),
                     size, size, cull='NONE', gbuf=g, subset=sel)
    covered = g.tri >= 0
    if not covered.any():
        return None, 'no UV-mapped faces landed inside the 0..1 tile'

    ys, xs = np.nonzero(covered)
    tri_idx = g.tri[covered].astype(np.int64)
    bary = g.bary[covered]
    if progress:
        progress(0.2, f'Baking {tri_idx.size} texels')

    eye = np.zeros(3, np.float32)
    if scene.camera is not None and scene.camera.matrix_world is not None:
        eye = np.asarray(scene.camera.matrix_world,
                         np.float32)[:3, 3].copy()
    job = ShadeJob(scene, st, textures, bvh,
                   np.eye(4, dtype=np.float32), eye, size, size)
    out = np.zeros((size, size, 4), np.float32)
    flat = np.zeros((tri_idx.size, 4), np.float32)
    mat_idx = mesh.mat_index[tri_idx] if mesh.mat_index is not None \
        else np.zeros(tri_idx.size, np.int32)
    px_f = xs.astype(np.float32)
    py_f = ys.astype(np.float32)
    done = 0
    for mi in np.unique(mat_idx):
        s2 = np.nonzero(mat_idx == mi)[0]
        mat = scene.materials[int(mi)] \
            if int(mi) < len(scene.materials) else None
        sub = job.context(tri_idx[s2], bary[s2], px_f[s2], py_f[s2],
                          np.ones(s2.size, bool), None, 0, True)
        sub.spx = xs[s2].astype(np.int64)
        sub.spy = ys[s2].astype(np.int64)
        # the bake view: straight down the normal, per texel
        Nn = M.normalize(sub.N)
        sub.I = (-Nn).astype(np.float32)
        if mode == 'AO':
            if bvh is None:
                occ = np.ones(s2.size, np.float32)
            else:
                occ = ambient_occlusion(sub.P, Nn, bvh, st,
                                        rng=job.rng,
                                        sample_xy=(sub.spx, sub.spy))
            flat[s2, 0] = flat[s2, 1] = flat[s2, 2] = occ
            flat[s2, 3] = 1.0
        else:
            col = job.shade_batch(sub, mat)
            flat[s2] = col
            flat[s2, 3] = 1.0
        done += s2.size
        if progress:
            progress(0.2 + 0.7 * done / max(tri_idx.size, 1), 'Baking')
    out[ys, xs] = flat

    # seam margin: grow the island edges outward so bilinear samples
    # never read the void
    cov = covered.copy()
    for _ in range(max(int(margin), 0)):
        grow = np.zeros_like(cov)
        acc = np.zeros_like(out)
        cnt = np.zeros((size, size), np.float32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                src = cov[max(-dy, 0):size - max(dy, 0),
                          max(-dx, 0):size - max(dx, 0)]
                dsty = slice(max(dy, 0), size - max(-dy, 0))
                dstx = slice(max(dx, 0), size - max(-dx, 0))
                srcy = slice(max(-dy, 0), size - max(dy, 0))
                srcx = slice(max(-dx, 0), size - max(dx, 0))
                acc[dsty, dstx] += np.where(src[:, :, None],
                                            out[srcy, srcx], 0.0)
                cnt[dsty, dstx] += src
                grow[dsty, dstx] |= src
        fill = grow & ~cov & (cnt > 0)
        out[fill] = acc[fill] / cnt[fill][:, None]
        cov |= fill
    if progress:
        progress(1.0, 'Baked')
    return out, None


def _rot_axis3(axis, ang):
    """Rodrigues: the 3x3 rotation of `ang` radians about a unit axis."""
    ax = np.asarray(axis, np.float32)
    c, s = float(np.cos(ang)), float(np.sin(ang))
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], np.float32)
    return (np.eye(3, dtype=np.float32) * c + s * K
            + (1.0 - c) * np.outer(ax, ax)).astype(np.float32)


def _render_panorama(scene, st, progress):
    """The QTVR cylinder: a full 360-degree panorama from one eye point.

    A cylindrical projection cannot come out of a linear rasteriser, so
    the drum is rendered as rotated planar STRIPS -- all sharing the
    same eye, which keeps every view-dependent term (speculars,
    fresnel, reflections) continuous across the joins -- and each strip
    is then resampled onto the cylinder with the exact planar-to-
    cylindrical mapping, bilinear, deterministic. Sixteen strips of
    22.5 degrees, rendered with ~35% horizontal overscan so the
    resample never reads outside its strip.

    The output maps column to azimuth linearly across 360 degrees and
    row to tan(elevation) through the camera's own vertical field.
    Screen-space extras that assume one planar view -- light shafts,
    lens flares, DoF and the aux passes -- are cleared rather than
    stitched wrong; a panorama is a backdrop deliverable, exactly as
    QTVR was.
    """
    import copy as _copy
    W = max(int(st.resolution_x), 1)
    H = max(int(st.resolution_y), 1)
    cam = scene.camera
    mw = np.asarray(cam.matrix_world, np.float32)
    up = mw[:3, 1] / max(float(np.linalg.norm(mw[:3, 1])), 1e-9)
    lens = float(getattr(cam, 'lens', 50.0) or 50.0)
    sensor = float(getattr(cam, 'sensor', 36.0) or 36.0)
    fov_x = 2.0 * np.arctan(sensor * 0.5 / max(lens, 1e-3))
    aspect = W / max(H, 1)
    tan_vh = float(np.tan(fov_x * 0.5) / max(aspect, 1e-6))
    near = float(getattr(cam, 'clip_start', 0.1) or 0.1)
    far = float(getattr(cam, 'clip_end', 1000.0) or 1000.0)

    n_strips = 16
    hstrip = 2.0 * np.pi / n_strips
    tan_hh = float(np.tan(hstrip * 0.5)) * 1.10
    tan_vs = tan_vh / float(np.cos(hstrip * 0.5)) * 1.04
    ws = int(np.ceil(W / n_strips * 1.35))
    hs = int(np.ceil(H * 1.15))
    cols_per = W / float(n_strips)

    # borrow the raster's own _persp so the strips live on exactly the
    # clip conventions every other camera in the engine uses:
    # proj[1,1] = 1/tan_vs and proj[0,0] = 1/tan_hh
    proj = LI._persp(2.0 * np.arctan(tan_vs), tan_hh / tan_vs, near, far)

    out = np.zeros((H, W, 4), np.float32)
    jj = (np.arange(H, dtype=np.float32) + 0.5) / H * 2.0 - 1.0
    tanel = jj * tan_vh
    _gpu_flags = []
    for k in range(n_strips):
        if progress:
            progress(0.05 + 0.9 * (k / n_strips),
                     f'Panorama strip {k + 1}/{n_strips}')
        phi_c = (k + 0.5) * hstrip - np.pi
        cam2 = _copy.copy(cam)
        m2 = mw.copy()
        m2[:3, :3] = _rot_axis3(up, -phi_c) @ mw[:3, :3]
        cam2.matrix_world = m2
        cam2.projection = proj.copy()
        cam2.type = 'PERSP'
        sc2 = _copy.copy(scene)
        sc2.camera = cam2
        st2 = st.copy()
        st2.resolution_x = ws
        st2.resolution_y = hs
        st2._pano_strip = True
        frame = render(sc2, st2, None, band=None)
        _gpu_flags.append(bool(getattr(st2, '_frame_gpu_shaded', False)))
        fh, fw = frame.shape[:2]

        i0 = int(np.floor(k * cols_per))
        i1 = int(np.floor((k + 1) * cols_per)) if k + 1 < n_strips else W
        cols = np.arange(i0, i1, dtype=np.float32)
        phi = (cols + 0.5) / W * (2.0 * np.pi) - np.pi
        dphi = phi - phi_c
        sx = np.tan(dphi) / tan_hh                       # strip ndc x
        sec = 1.0 / np.cos(dphi)
        sy = (tanel[:, None] * sec[None, :]) / tan_vs    # strip ndc y
        px = (sx * 0.5 + 0.5) * fw - 0.5
        py = (sy * 0.5 + 0.5) * fh - 0.5
        pxg = np.broadcast_to(px[None, :], sy.shape)
        x0 = np.clip(np.floor(pxg), 0, fw - 2).astype(np.int32)
        y0 = np.clip(np.floor(py), 0, fh - 2).astype(np.int32)
        tx = np.clip(pxg - x0, 0.0, 1.0).astype(np.float32)[:, :, None]
        ty = np.clip(py - y0, 0.0, 1.0).astype(np.float32)[:, :, None]
        c00 = frame[y0, x0]
        c10 = frame[y0, x0 + 1]
        c01 = frame[y0 + 1, x0]
        c11 = frame[y0 + 1, x0 + 1]
        out[:, i0:i1] = (c00 * (1 - tx) * (1 - ty) + c10 * tx * (1 - ty)
                         + c01 * (1 - tx) * ty + c11 * tx * ty)

    # planar screen-space extras cannot be stitched honestly
    scene.last_passes = None
    scene.last_depth = None
    scene.last_coverage = None      # R251 (C092): no honest stitched plane
    st._last_coverage = None
    scene.last_shafts = []
    scene.last_flares = []
    out = out.astype(np.float32)
    _propagate_gpu_flag(st, _gpu_flags, 'panorama strips')
    # R200: weather falls ONCE across the finished panorama -- each
    # strip skipped it (st2._pano_strip), or sixteen copies of the
    # same storm would tile the sweep
    wd_w = getattr(scene, 'world', None)
    if wd_w is not None and \
            str(getattr(wd_w, 'weather', 'NONE') or 'NONE') != 'NONE':
        from . import sky as SKY
        with ST.track('weather'):
            out = SKY.weather_overlay(
                out, wd_w, W, H,
                time=float(getattr(scene, 'time', 0.0)), out_wh=(W, H))
    return out


def _render_stereo(scene, st, progress):
    """Two eyes, one picture: the era's stereo pair modes.

    Parallel cameras separated by Eye Distance along the camera's own
    right axis, each with the off-axis frustum shift that puts the
    Convergence distance at zero parallax -- translate-and-shift, the
    method with no vertical parallax and no keystone, which is why the
    stereo photography of the period used it. ANAGLYPH combines the
    classic way (left eye's red, right eye's green and blue -- view
    with red/cyan glasses); SBS packs squeezed left|right half-frames;
    CROSS swaps them for the cross-eyed viewer. The auxiliary passes
    and shafts keep the LEFT eye's, the reference eye.
    """
    import copy as _copy
    d = float(getattr(st, 'stereo_eye_distance', 0.065))
    conv = max(float(getattr(st, 'stereo_convergence', 8.0)), 1e-3)
    mw = np.asarray(scene.camera.matrix_world, np.float32)
    right = mw[:3, 0] / max(float(np.linalg.norm(mw[:3, 0])), 1e-9)
    _stereo_flags = []
    _stereo_cov = []                # R251 (C092): each eye's coverage plane

    def eye_frame(sign, prog):
        cam2 = _copy.copy(scene.camera)
        mw2 = mw.copy()
        mw2[:3, 3] = mw2[:3, 3] + right * (sign * d * 0.5)
        cam2.matrix_world = mw2
        sc2 = _copy.copy(scene)
        sc2.camera = cam2
        st2 = st.copy()
        # the eye moved +s along right, so straight-ahead points slide
        # -x in its view; the window shifts by +p00*s/conv to put
        # Convergence back at ndc x = 0 (clip_jitter's own term). R251:
        # the sign was inverted since R190 -- the eyes DIVERGED (a
        # convergence-plane point landed at +/-p00*d/(2*conv) instead
        # of 0); fixed, and every pair with a non-zero eye distance moves
        st2._stereo = (sign * d * 0.5, conv)
        frame = render(sc2, st2, prog, band=None)
        _stereo_flags.append(bool(getattr(st2, '_frame_gpu_shaded', False)))
        _stereo_cov.append(getattr(st2, '_last_coverage', None))
        # aux passes land on the shallow copy's shared attributes; the
        # LEFT eye (rendered first) is the reference, so its values are
        # captured before the right eye overwrites them
        return frame, (getattr(sc2, 'last_passes', None),
                       getattr(sc2, 'last_depth', None),
                       getattr(sc2, 'last_shafts', None),
                       getattr(sc2, 'last_flares', None))

    if getattr(st, 'stereo_parallax_layers', False):
        # R251 C016: the Virtual Boy's VIP never projected two cameras --
        # every world (layer) carried ONE integer parallax and was drawn
        # at x - P into the left buffer and x + P into the right. One
        # centre render (IndexOB + Depth forced at OUTPUT resolution),
        # then each object is a flat card shifted by the whole-pixel
        # disparity the two-camera road would give its centroid depth;
        # the sky is the farthest layer, at the cap. Every op after the
        # render is integer indexing: both devices produce the same
        # bytes by construction.
        st2 = st.copy()
        st2.pass_object_index = True
        st2.pass_depth = True
        # st.copy() drops the private flags and stereo_mode is still
        # set: without this the stereo gate would re-enter this road
        # forever. J[0, 3] += p00 * 0.0 / conv == 0.0, so J is the exact
        # identity and (J @ proj) reproduces proj bitwise
        st2._stereo = (0.0, conv)
        sc2 = _copy.copy(scene)
        centre = render(sc2, st2, progress, band=None)
        _stereo_flags.append(bool(getattr(st2, '_frame_gpu_shaded', False)))
        passes = getattr(sc2, 'last_passes', None) or {}
        H, W = centre.shape[:2]
        Pmax = int(np.clip(int(getattr(st, 'stereo_parallax_max', 16)), 1, 64))
        mesh = scene.mesh
        if 'IndexOB' in passes and 'Depth' in passes and mesh is not None \
                and mesh.verts is not None and mesh.tris is not None \
                and mesh.tris.size and passes['IndexOB'].shape[:2] == (H, W):
            ids = passes['IndexOB'][:, :, 0].astype(np.int64)
            # build_aux_passes' convention for no geometry
            uncovered = passes['Depth'][:, :, 0] >= 1e10
            # per-object centroid view depth through the same flagged
            # camera (the Y-shear composes)
            view, proj, _vp, _eye = camera_matrices(scene.camera, W, H)
            verts = np.asarray(mesh.verts, np.float32)
            ones = np.ones((verts.shape[0], 1), np.float32)
            verts_v = (np.concatenate([verts, ones], 1) @ view.T)[:, :3]
            zv = -verts_v[:, 2].astype(np.float64)
            oi_tri = (np.asarray(mesh.obj_index, np.int64)
                      if getattr(mesh, 'obj_index', None) is not None
                      else np.zeros(len(mesh.tris), np.int64))
            n_obj = int(oi_tri.max()) + 1
            # one pass, no Python loop over objects: every triangle
            # corner counted, so a shared vertex weighs once per triangle
            # it sits on (the rule, stated)
            oi = np.repeat(oi_tri, 3)
            zc = zv[np.asarray(mesh.tris).ravel()]
            cnt = np.bincount(oi, minlength=n_obj)
            z_o = np.where(cnt > 0,
                           np.bincount(oi, weights=zc, minlength=n_obj)
                           / np.maximum(cnt, 1), conv)
            # the two-camera road's disparity at that depth, in OUTPUT
            # pixels, whole: D(z) = (W/2) * p00 * d * (1/conv - 1/z);
            # P = round_half_even(D/2) (np.round: the tie rule); nearer
            # than Convergence -> negative (the VB's sign)
            D = 0.5 * W * float(proj[0, 0]) * d * \
                (1.0 / conv - 1.0 / np.maximum(z_o, 1e-6))
            P_o = np.clip(np.round(D * 0.5), -Pmax, Pmax).astype(np.int64)
            # the sky is the farthest layer, at the cap
            Pmap = np.where(uncovered, Pmax, P_o[np.clip(ids, 0, n_obj - 1)])
        else:
            # no geometry: one layer, the sky, at the cap
            Pmap = np.full((H, W), Pmax, np.int64)
        left = parallax_shift(centre, Pmap, -1)
        right_f = parallax_shift(centre, Pmap, +1)
        # the centre's aux, minus the two passes forced here unless the
        # user asked for them
        want = wanted_passes(st)
        kept = {k: v for k, v in passes.items() if k in want}
        (scene.last_passes, scene.last_depth, scene.last_shafts,
         scene.last_flares) = (kept if kept else None,
                               getattr(sc2, 'last_depth', None),
                               getattr(sc2, 'last_shafts', None),
                               getattr(sc2, 'last_flares', None))
        _propagate_gpu_flag(st, _stereo_flags, 'parallax layers')
        # R251 (C092): each eye's plane is the centre's, shifted by the
        # very gather that shifted the picture, then packed as it is
        _cc = getattr(st2, '_last_coverage', None)
        if _cc is not None and _cc.shape == Pmap.shape:
            _stereo_cov = [parallax_shift(_cc, Pmap, -1),
                           parallax_shift(_cc, Pmap, +1)]
        else:
            _stereo_cov = [None, None]
        scene.last_coverage = _pack_stereo_coverage(
            _stereo_cov, str(getattr(st, 'stereo_mode', 'NONE')), W)
        st._last_coverage = scene.last_coverage
        return _pack_stereo(left, right_f, str(getattr(st, 'stereo_mode', 'NONE')))

    left, aux = eye_frame(-1.0, progress)
    right_f, _aux_r = eye_frame(+1.0, None)
    (scene.last_passes, scene.last_depth, scene.last_shafts,
     scene.last_flares) = aux
    _propagate_gpu_flag(st, _stereo_flags, 'stereo eyes')
    # R251 (C092): the coverage plane packed exactly as the picture is
    scene.last_coverage = _pack_stereo_coverage(
        _stereo_cov, str(getattr(st, 'stereo_mode', 'NONE')), left.shape[1])
    st._last_coverage = scene.last_coverage
    return _pack_stereo(left, right_f, str(getattr(st, 'stereo_mode', 'NONE')))


def parallax_shift(centre, Pmap, sign):
    """R251 C016: one eye of the Virtual Boy pair -- every pixel of the
    centre frame moved horizontally by its layer's whole-pixel parallax
    (`Pmap`, int64 (H, W)); sign -1 is the left buffer (x - P), +1 the
    right (x + P). Layers are gathered nearest first (ascending P) and
    the first claim wins; a pixel no layer claims (the far plane hidden
    behind a shifted near card -- the VIP drew its background world
    there, Halcyon has no picture behind an object) keeps the centre
    frame's own pixel, never a zero. Rows are independent, so the
    gather is order-free per row. `sign` is an INTEGER: a float sign
    would make the source index float64."""
    sign = int(sign)
    H, W = centre.shape[:2]
    out = centre.copy()
    claimed = np.zeros((H, W), bool)
    X = np.arange(W)[None, :]
    rows = np.arange(H)[:, None]
    for P in np.unique(Pmap):                  # ascending = nearest first
        P = int(P)
        # left: the pixel X shows the centre's X + P (a near card, P < 0,
        # moves RIGHT in the left eye)
        src = X - sign * P
        ok = (src >= 0) & (src < W)
        srcc = np.clip(src, 0, W - 1)
        take = ok & (Pmap[rows, srcc] == P) & ~claimed
        out[take] = centre[rows, srcc][take]
        claimed |= take
    return out


def _or_coverage(planes):
    """R251 (C092): the coverage of an accumulated frame -- a pixel any
    pass covered is covered (the any-sample law). None when a pass has
    no plane or the shapes disagree."""
    cov = None
    for c in planes:
        if c is None or (cov is not None and c.shape != cov.shape):
            return None
        cov = c.copy() if cov is None else (cov | c)
    return cov


def _pack_stereo_coverage(planes, mode, width):
    """R251 (C092): the two eyes' coverage planes packed exactly as
    _pack_stereo packs the picture: ANAGLYPH is covL | covR (a pixel
    covered in either eye shows geometry in some channel); SBS / CROSS
    take every other column of each eye, the seam padded `edge`. None
    when either eye has no plane."""
    if len(planes) != 2 or planes[0] is None or planes[1] is None \
            or planes[0].shape != planes[1].shape:
        return None
    covL, covR = planes
    if mode == 'ANAGLYPH':
        return covL | covR
    lh_c = covL[:, ::2]
    rh_c = covR[:, ::2]
    w_half = min(lh_c.shape[1], rh_c.shape[1])
    lh_c, rh_c = lh_c[:, :w_half], rh_c[:, :w_half]
    cov = np.concatenate((rh_c, lh_c) if mode == 'CROSS' else (lh_c, rh_c),
                         axis=1)
    if cov.shape[1] < width:                    # odd width: pad the seam
        cov = np.pad(cov, ((0, 0), (0, width - cov.shape[1])), mode='edge')
    return np.ascontiguousarray(cov[:, :width])


def _pack_stereo(left, right_f, mode):
    """The stereo pair packed for viewing: the 1.89.0 tail of
    _render_stereo, factored so the two-camera road and the parallax
    layers (C016) share it byte for byte. ANAGLYPH combines the classic
    way (left eye's red, right eye's green and blue); SBS packs squeezed
    left|right half-frames; CROSS swaps them for the cross-eyed viewer."""
    if mode == 'ANAGLYPH':
        out = right_f.copy()
        out[:, :, 0] = left[:, :, 0]
        out[:, :, 3] = np.maximum(left[:, :, 3], right_f[:, :, 3])
        return out.astype(np.float32)
    # squeezed halves: every other column, nearest-neighbour -- the
    # period pack, deterministic
    lh = left[:, ::2, :]
    rh = right_f[:, ::2, :]
    w_half = min(lh.shape[1], rh.shape[1])
    lh, rh = lh[:, :w_half], rh[:, :w_half]
    pair = (rh, lh) if mode == 'CROSS' else (lh, rh)
    out = np.concatenate(pair, axis=1)
    if out.shape[1] < left.shape[1]:            # odd width: pad the seam
        out = np.pad(out, ((0, 0), (0, left.shape[1] - out.shape[1]),
                           (0, 0)), mode='edge')
    return out[:, :left.shape[1]].astype(np.float32)


def _render_pano_parts(scene, st, progress):
    """R251 C125: Blender 2.4's Pano + Xparts -- N planar strips, the
    camera yawed by one strip's horizontal field between them, butted
    together unblended (straight lines kink at every seam: the tell of
    the era's QuickTime VR cylinders; `initrender.c`: "pano is fake
    parts"). The strip angle is the camera's own horizontal field fitted
    across the STRIP width (the `LI._persp` fallback at aspect ws/H,
    `cam2.projection = None`), which is what makes the strips abut on a
    shared clip plane by construction; the last strip is cropped when N
    does not divide the width. Each strip is a whole render() on
    whichever device; the concatenation is integer slicing, so both
    roads produce the same bytes by construction.
    """
    n = int(np.clip(int(st.pano_parts), 1, 16))
    W = max(int(st.resolution_x), 1)
    H = max(int(st.resolution_y), 1)
    ws = int(math.ceil(W / n))       # 2.4: SizeX per part; n*ws >= W, cropped
    cam = scene.camera
    mw = M.rigid_camera_matrix(np.asarray(cam.matrix_world, np.float32))
    up = mw[:3, 1] / max(float(np.linalg.norm(mw[:3, 1])), 1e-9)
    lens = float(getattr(cam, 'lens', 50.0) or 50.0)
    sensor = float(getattr(cam, 'sensor', 36.0) or 36.0)
    # camera_matrices' own fov_x, across the STRIP width (2.4's
    # 2*atan(16/lens) with a 32 mm sensor)
    phi = 2.0 * math.atan(sensor * 0.5 / max(lens, 1e-3))
    out = np.zeros((H, W, 4), np.float32)
    flags = []
    for i in range(n):
        if progress:
            progress(0.05 + 0.9 * (i / n), f'Pano part {i + 1}/{n}')
        a_i = (i - (n - 1) * 0.5) * phi          # azimuth of strip i, leftmost first
        cam2 = copy.copy(cam)
        m2 = mw.copy()
        # _render_panorama's sign: azimuth to the right = negative
        # rotation about up
        m2[:3, :3] = _rot_axis3(up, -a_i) @ mw[:3, :3]
        cam2.matrix_world = m2
        cam2.projection = None
        cam2.type = 'PERSP'
        sc2 = copy.copy(scene)
        sc2.camera = cam2
        st2 = st.copy()
        st2.resolution_x = ws
        st2.resolution_y = H
        # the flag every strip-hostile gate already reads (stereo,
        # weather, Y-shear)
        st2._pano_strip = True
        # a dataclass FIELD, so it survives every st.copy() below this
        # road: a strip's AA passes can never re-enter this gate (without
        # it ACCUMULATE inside a strip would split the strip into strips
        # of strips down to width 1, forever)
        st2.pano_parts = 1
        frame = render(sc2, st2, progress if i == 0 else None, band=None)
        flags.append(bool(getattr(st2, '_frame_gpu_shaded', False)))
        x0 = i * ws
        x1 = min((i + 1) * ws, W)
        out[:, x0:x1] = frame[:, :x1 - x0]
    # planar extras cannot be stitched honestly (the PANO rule)
    scene.last_passes = None
    scene.last_depth = None
    scene.last_coverage = None      # R251 (C092): the PANO rule
    st._last_coverage = None
    scene.last_shafts = []
    scene.last_flares = []
    _propagate_gpu_flag(st, flags, 'pano parts')
    # weather falls ONCE across the finished frame -- each strip skipped
    # it (st2._pano_strip), exactly as the true panorama does
    wd_w = getattr(scene, 'world', None)
    if wd_w is not None and \
            str(getattr(wd_w, 'weather', 'NONE') or 'NONE') != 'NONE':
        from . import sky as SKY
        with ST.track('weather'):
            out = SKY.weather_overlay(
                out, wd_w, W, H,
                time=float(getattr(scene, 'time', 0.0)), out_wh=(W, H))
    return out


def _lens_points(st, R_l):
    """R251 C098: the lens points of one lens-sampled DOF frame, in
    scene units across the aperture of radius R_l, rounded once to
    float32. HALTON_DISC: the accumulation-buffer AA's own Halton
    sequence mapped to the unit disc (Shirley-Chiu concentric map; the
    1990 paper's 23-point table was never published, this stands in).
    MAX_SPIRAL: pass 0 at the lens centre (3ds Max's Use Original
    Location), then Halcyon's golden-angle spiral over Max's documented
    multi-pass dials (Sample Radius 1.0, Sample Bias 0.5) -- Autodesk
    documents the dials, not its point pattern."""
    K = int(np.clip(int(getattr(st, 'dof_lens_samples', 23)), 2, 64))
    pts = []
    if str(getattr(st, 'dof_lens_pattern', 'HALTON_DISC')) == 'MAX_SPIRAL':
        for k in range(K):
            if k == 0:
                pts.append((0.0, 0.0))
                continue
            # Sample Radius 1.0, Sample Bias 0.5 (Max's defaults, constants)
            rho = R_l * ((k + 0.5) / K) ** (0.5 + 0.5)
            a = k * 2.399963229728653                      # the golden angle
            pts.append((rho * math.cos(a), rho * math.sin(a)))
    else:
        for k in range(1, K + 1):
            a = 2.0 * _halton(k, 2) - 1.0                  # the AA road's sequence
            b = 2.0 * _halton(k, 3) - 1.0
            if a == 0.0 and b == 0.0:
                r, ph = 0.0, 0.0
            elif a * a > b * b:
                r, ph = a, (math.pi / 4.0) * (b / a)
            else:
                r, ph = b, (math.pi / 2.0) - (math.pi / 4.0) * (a / b)
            pts.append((R_l * r * math.cos(ph), R_l * r * math.sin(ph)))
    return [(float(np.float32(x)), float(np.float32(y))) for x, y in pts]


def _render_lens_accumulated(scene, st, progress, band):
    """R251 C098: depth of field by sampling the LENS -- REYES lens
    sampling by the SGI accumulation buffer's method (Haeberli & Akeley
    1990, Appendix B) and 3ds Max's multi-pass DOF: the frame rendered
    once per lens point with the eye moved across the aperture and the
    window sheared so the focus plane stays put (camera_matrices reads
    the per-pass camera's `_lens`), then averaged in float64 in pass
    order as the accumulation AA does. Exact occlusion, the lens pattern
    as the bokeh, K full renders. The lens diameter is the camera's
    focal length over its f-number (RiDepthOfField's lensdiameter), the
    focus is `dof_focus`. Each pass is a whole render() on whichever
    device; the mean is the CPU's: both roads produce the same bytes by
    construction. An F12 road: the gate excludes the viewport."""
    cam = scene.camera
    # RiDepthOfField: lensdiameter = focallength / fstop (the CAMERA's
    # f-number), scene units = metres; the radius is half of it
    R_l = (float(getattr(cam, 'lens', 50.0) or 50.0) / 1000.0) / \
        (2.0 * max(float(getattr(cam, 'fstop', 2.8) or 2.8), 1e-3))
    d_f = max(float(st.dof_focus), 1e-3)
    acc = None
    flags = []
    aux = None
    _lens_cov = []                  # R251 (C092): every pass's coverage plane
    pts = _lens_points(st, R_l)
    for k, (lx, ly) in enumerate(pts):
        if progress:
            progress(0.05 + 0.9 * (k / len(pts)), f'Lens pass {k + 1}/{len(pts)}')
        # no _yshear copy: render()'s flag line rewrites scene.camera's
        # _yshear inside every pass before camera_matrices reads it
        cam2 = copy.copy(cam)
        cam2._lens = (lx, ly, d_f)
        sc2 = copy.copy(scene)
        sc2.camera = cam2
        st2 = st.copy()
        st2._lens_pass = True
        # st.copy() drops the private flags; the outer roads' state (an
        # AA pass's jitter, a stereo eye, a pano strip) must ride along
        for flag in ('_accum_jitter', '_stereo', '_viewport', '_refine_mask',
                     '_pano_strip'):
            if hasattr(st, flag):
                setattr(st2, flag, getattr(st, flag))
        frame = render(sc2, st2, progress if k == 0 else None, band=band)
        flags.append(bool(getattr(st2, '_frame_gpu_shaded', False)))
        _lens_cov.append(getattr(st2, '_last_coverage', None))
        if k == 0:
            # pass 0's extras (the centre point under MAX_SPIRAL)
            aux = (getattr(sc2, 'last_passes', None),
                   getattr(sc2, 'last_depth', None),
                   getattr(sc2, 'last_shafts', None),
                   getattr(sc2, 'last_flares', None))
            _gel0 = getattr(sc2, 'last_gel', None)      # R251 C134
        # the accumulation road's own sum, in pass order
        acc = frame.astype(np.float64) if acc is None else acc + frame
    (scene.last_passes, scene.last_depth, scene.last_shafts,
     scene.last_flares) = aux
    scene.last_gel = _gel0                              # R251 C134
    _propagate_gpu_flag(st, flags, 'lens passes')
    scene.last_coverage = _or_coverage(_lens_cov)     # R251 (C092)
    st._last_coverage = scene.last_coverage
    return (acc / float(len(flags))).astype(np.float32)


def shutter_steps(st):
    """R251 C090: how many sub-frames the engine's motion blur renders --
    Blur Steps, clamped to 32 under Max Object Motion Blur (Max's own
    Samples cap, printed), doubled under LightWave's Dithered blur (the
    row-parity split needs a slice per parity)."""
    n = max(int(st.motion_steps), 2)
    mode = str(getattr(st, 'motion_blur_mode', 'MEAN'))
    if mode == 'MAX_SLICES' and n > 32:
        print(f"[Halcyon] motion blur: Blur Steps {n} clamped to 32 under "
              "Max Object Motion Blur (Max's own Samples cap)")
        n = 32
    return 2 * n if mode == 'LW_FIELD' else n


def shutter_weight(st, k, S, H, W, frame_no=0, seed=0):
    """R251 C090: the (H, W) float64 weight of slice k of S -- a pure
    function of (x, y, k, S, frame, seed), so the engine streams the
    combine with ONE slice resident. MEAN: every weight 1.0. LW_FIELD:
    even rows take the even slices, odd rows the odd (row 0 = bottom;
    LightWave 5.6-7's Dithered Motion Blur, the field-rendering split).
    MAX_SLICES: each pixel averages only Samples (M) of the S slices,
    the first pick o = floor(h01 * S) from the GRAIN hash family and the
    rest (o + floor(j*S/M)) mod S, j < M -- tested per slice by the
    arithmetic rule "a multiple of S lies in [r*M, r*M + M)" with
    r = (k - o) mod S; Max 3+'s multi-pass Dither Strength then weights
    each picked slice 1 +/- d by an 8x8 ordered pattern phased per slice,
    thresholds at the Bayer cell CENTRES (k + 0.5)/64, so every weight is
    at least 1 - d*63/64 >= 1/64 and no pixel can lose every slice."""
    # R:7883's own form: core/render.py binds neither at module level
    from . import film as FILM
    from .dither import BAYER8       # bayer(8) is ALREADY k/64 as float32
    mode = str(getattr(st, 'motion_blur_mode', 'MEAN'))
    if mode == 'MEAN':
        return np.ones((H, W), np.float64)
    yy, xx = np.mgrid[0:H, 0:W]
    if mode == 'LW_FIELD':
        return ((yy & 1) == (k & 1)).astype(np.float64)
    M_ = int(np.clip(int(getattr(st, 'motion_samples', 10)), 1, S))
    h = FILM._hash_u32_raw(xx.astype(np.uint32), yy.astype(np.uint32),
                           int(frame_no) * 1000003 + int(seed))
    o = (h.astype(np.int64) * S) >> 24          # floor(h01 * S): the first slice
    r = (k - o) % S                             # k's offset from the first pick
    sel = ((-(r * M_)) % S) < M_                # k is one of the M picks
    d = float(np.clip(float(getattr(st, 'motion_dither', 0.0)), 0.0, 1.0))
    if d > 0.0:
        cell = max(int(getattr(st, 'motion_dither_tile', 32)) // 8, 1)
        B = BAYER8[((xx // cell) + k) % 8, (yy // cell) % 8].astype(np.float64) \
            + (0.5 / 64.0)
        return sel * (1.0 + d * (2.0 * B - 1.0))
    return sel.astype(np.float64)


def shutter_combine(frames, st, frame_no=0, seed=0):
    """R251 C090: `frames` are the S sub-frame renders in time order
    ((H, W, 4) float32). Returns the (H, W, 4) float32 frame -- the batch
    form of the engine's streamed loop (for the tests): the same
    k-ordered float64 sums, bitwise. MEAN is the 1.89.0 expression
    verbatim; MAX_SLICES with Samples == S and Dither 0 reproduces it
    bitwise (every weight 1.0, wsum == S exactly)."""
    S = len(frames)
    H, W = frames[0].shape[:2]
    if str(getattr(st, 'motion_blur_mode', 'MEAN')) == 'MEAN':
        acc = frames[0].astype(np.float64)
        for k in range(1, S):
            acc = acc + frames[k]
        return (acc / float(S)).astype(np.float32)
    acc = np.zeros((H, W, 4), np.float64)
    wsum = np.zeros((H, W), np.float64)
    for k in range(S):
        w = shutter_weight(st, k, S, H, W, frame_no, seed)
        acc = acc + frames[k] * w[:, :, None]
        wsum = wsum + w
    return (acc / wsum[:, :, None]).astype(np.float32)


def _halton(i, b):
    """The radical-inverse sequence: deterministic, well-spread offsets."""
    f, r = 1.0, 0.0
    while i > 0:
        f /= b
        r += f * (i % b)
        i //= b
    return r


def _render_accumulated(scene, st, progress, band):
    """The accumulation buffer: N whole frames at subpixel offsets, averaged.

    Exactly what the OpenGL accumulation buffer (and the 3dfx T-buffer) did
    for antialiasing: render the full frame N times, each time with the
    projection nudged by a fraction of a pixel, and average. Halton offsets
    make every run of the same scene the same picture. Each pass is a
    complete render at output resolution, so the cost is N frames -- the
    era's price, honestly paid.
    """
    n = int(np.clip(int(st.aa_samples), 2, 64))
    acc = None
    _accum_flags = []
    _accum_cov = []                 # R251 (C092): every pass's coverage plane
    for k in range(n):
        st2 = st.copy()
        st2.aa_mode = 'NONE'
        st2._accum_jitter = (_halton(k + 1, 2) - 0.5,
                             _halton(k + 1, 3) - 0.5)
        # R251: st.copy() drops the private flags; a stereo eye's or a
        # pano strip's pass must keep reading them (a pair per pass
        # since R190 otherwise; at defaults both read None / False)
        st2._stereo = getattr(st, '_stereo', None)
        st2._pano_strip = getattr(st, '_pano_strip', False)
        frame = render(scene, st2, progress if k == 0 else None, band=band)
        _accum_flags.append(bool(getattr(st2, '_frame_gpu_shaded', False)))
        _accum_cov.append(getattr(st2, '_last_coverage', None))
        acc = frame.astype(np.float64) if acc is None else acc + frame
    _propagate_gpu_flag(st, _accum_flags, 'accumulation passes')
    # R251 (C092): any sample counts as covered -- the passes' planes OR-ed
    scene.last_coverage = _or_coverage(_accum_cov)
    st._last_coverage = scene.last_coverage
    return (acc / float(n)).astype(np.float32)


#: what the last adaptive-AA frame decided, for the console line and the
#: tests: pixels flagged, pixels total, passes run (0 pixels = base
#: frame shipped as-is), and the boolean mask itself
LAST_ADAPTIVE = {}

#: the fixed colour-contrast criterion, in linear luma between
#: neighbours. POV-Ray's +A default is 0.3 over 0..3 summed channels;
#: this is the same order of strictness expressed on one channel
_ADAPTIVE_CONTRAST = 0.06


def _adaptive_mask(img, gbuf, mesh, st, depth_m):
    """The pixels the adaptive refine passes re-sample.

    The geometric criteria flag both pixels of a differing pair (the
    _edge_smooth convention -- a stair-step aliases on both sides):

    - OBJECT-id silhouettes -- object ids, NOT triangle ids: a dense
      mesh is all triangle edges inside a surface that does not alias
      at all, and flagging them would refine the whole frame
    - the sky boundary (covered against uncovered)
    - depth CURVATURE past Edge Depth Threshold: the second
      difference, not the first. The field's first frame flagged 21%
      of the picture, and most of it was smooth floor seen at a graze
      -- a surface running away from the camera has a large depth
      SLOPE at every pixel, which the old step test read as one long
      crease. Its curvature is ~zero; a real step or fold spikes it.
      Same dial, same scene units, honest meaning
    - colour contrast at a FIXED threshold (the POV-Ray +A criterion,
      one number, never a dial), but only within two pixels of a
      geometric flag. Contrast in the open field is texture detail,
      and texture detail is already anti-aliased by the mip chain --
      re-rendering it three times sharpens nothing. At the edges it
      still does its job: it catches the soft side of a silhouette,
      glints and seams the id test half-missed
    """
    tri = gbuf.tri
    if mesh is not None and getattr(mesh, 'obj_index', None) is not None:
        ids = np.where(tri >= 0, mesh.obj_index[np.clip(tri, 0, None)], -1)
    else:
        ids = np.where(tri >= 0, 0, -1)
    geo = np.zeros(tri.shape, bool)
    dif = ids[:, 1:] != ids[:, :-1]
    geo[:, 1:] |= dif
    geo[:, :-1] |= dif
    dif = ids[1:, :] != ids[:-1, :]
    geo[1:, :] |= dif
    geo[:-1, :] |= dif
    thr = float(getattr(st, 'aa_edge_threshold', 0.1))
    if thr > 0.0:
        src = depth_m if depth_m is not None else gbuf.depth
        d = np.nan_to_num(src, nan=1.0, posinf=1.0, neginf=-1.0)
        cx = np.abs(d[:, :-2] - 2.0 * d[:, 1:-1] + d[:, 2:]) > thr
        geo[:, 1:-1] |= cx
        cy = np.abs(d[:-2, :] - 2.0 * d[1:-1, :] + d[2:, :]) > thr
        geo[1:-1, :] |= cy
    # the band the contrast criterion is allowed to work in: the
    # geometric flags grown by two pixels each way
    band = geo.copy()
    for _ in range(2):
        grown = band.copy()
        grown[1:] |= band[:-1]
        grown[:-1] |= band[1:]
        grown[:, 1:] |= band[:, :-1]
        grown[:, :-1] |= band[:, 1:]
        band = grown
    lum = (0.2126 * img[..., 0] + 0.7152 * img[..., 1]
           + 0.0722 * img[..., 2])
    lum = np.nan_to_num(lum, nan=0.0, posinf=1.0, neginf=0.0)
    cont = np.zeros(tri.shape, bool)
    dd = np.abs(lum[:, 1:] - lum[:, :-1]) > _ADAPTIVE_CONTRAST
    cont[:, 1:] |= dd
    cont[:, :-1] |= dd
    dd = np.abs(lum[1:, :] - lum[:-1, :]) > _ADAPTIVE_CONTRAST
    cont[1:, :] |= dd
    cont[:-1, :] |= dd
    return geo | (cont & band)


def _adaptive_refine(scene, st, img, gbuf, mesh, depth_m, progress):
    """The Bryce anti-aliasing pass: re-render ONLY the edge pixels.

    What the field remembered of Bryce -- render the picture, then a
    line sweeps down anti-aliasing it -- and what POV-Ray's +A did on
    every trace: the base frame is sample 1, taken at the pixel
    centre, and passes 2..n re-render at Halton subpixel offsets with
    `_refine_mask` set, so the rasteriser still runs whole-frame (the
    jitter re-decides coverage) but SHADING -- the cost -- runs only
    at the flagged pixels. Flagged pixels average all n samples in
    float64; every other pixel keeps the base frame's value bitwise.

    Deterministic by construction: the mask is a pure function of the
    base frame and the offsets are Halton, so the same scene renders
    the same picture on every run. The refine passes shade on the SAME
    device the settings chose -- 1.38.0 forced them onto the CPU on
    the theory that a full-screen pass over a few percent of a frame
    was overhead, and the field's first frame billed 306 of its 317
    seconds to exactly that theory. The base frame's GPU verdict is
    preserved for the engine's report either way.
    """
    n = int(np.clip(int(st.aa_samples), 2, 64))
    mask = _adaptive_mask(img, gbuf, mesh, st, depth_m)
    count = int(np.count_nonzero(mask))
    total = int(mask.size)
    LAST_ADAPTIVE.clear()
    LAST_ADAPTIVE.update(pixels=count, total=total,
                         passes=(n - 1) if count else 0, mask=mask)
    if not count:
        print('[Halcyon] adaptive AA: no edge pixels flagged; the base '
              'frame ships as-is')
        return img
    print(f'[Halcyon] adaptive AA: {count} of {total} pixels flagged '
          f'({100.0 * count / max(total, 1):.1f}%), {n - 1} refine '
          f'pass(es) for Samples {n}')
    # the sub-renders overwrite scene.last_passes/depth/shafts (each is
    # a complete render) and reset LAST_GPU_VERDICT at entry; the BASE
    # frame's are the real ones, so they are put back afterwards
    _saved = (getattr(scene, 'last_passes', None),
              getattr(scene, 'last_depth', None),
              getattr(scene, 'last_shafts', None),
              getattr(scene, 'last_flares', None))
    _verdict = dict(LAST_GPU_VERDICT)
    # R251 (C092): the BASE frame's coverage plane, put back like the rest
    _saved_cov = getattr(scene, 'last_coverage', None)
    _saved_gel = getattr(scene, 'last_gel', None)       # R251 C134
    acc = img[mask].astype(np.float64)
    # R253: the COLOUR passes (Env, the light split) average over the
    # same samples the flagged pixels do, so Diffuse + Spec + Ambient +
    # Emit keeps summing to the refined beauty; the data passes keep
    # the base frame's values (a refined normal is no normal)
    _base_p = _saved[0] or {}
    _acc_p = {k: v[mask].astype(np.float64) for k, v in _base_p.items()
              if k not in DATA_PASSES and v.shape[:2] == mask.shape}
    try:
        for k in range(1, n):
            if progress:
                progress(0.85 + 0.13 * (k / float(n - 1)),
                         f'Anti-aliasing pass {k}/{n - 1} '
                         f'({count} edge pixels)')
            st2 = st.copy()
            st2.aa_mode = 'NONE'
            st2._accum_jitter = (_halton(k, 2) - 0.5,
                                 _halton(k, 3) - 0.5)
            st2._refine_mask = mask
            st2._stereo = getattr(st, '_stereo', None)      # R251: carried
            st2._pano_strip = getattr(st, '_pano_strip', False)
            acc += render(scene, st2, None, band=None)[mask]
            _rp = getattr(scene, 'last_passes', None) or {}
            for _k in list(_acc_p.keys()):
                if _k in _rp and _rp[_k].shape == _base_p[_k].shape:
                    _acc_p[_k] += _rp[_k][mask]
                else:
                    del _acc_p[_k]          # a refine without it: base kept
    finally:
        (scene.last_passes, scene.last_depth, scene.last_shafts,
         scene.last_flares) = _saved
        LAST_GPU_VERDICT.clear()
        LAST_GPU_VERDICT.update(_verdict)
        scene.last_coverage = _saved_cov
        st._last_coverage = _saved_cov
        scene.last_gel = _saved_gel                     # R251 C134
    out = img.copy()
    out[mask] = (acc / float(n)).astype(np.float32)
    if _acc_p:
        refined = dict(_base_p)
        for _k, _v in _acc_p.items():
            buf = _base_p[_k].copy()
            buf[mask] = (_v / float(n)).astype(np.float32)
            refined[_k] = buf
        scene.last_passes = refined
    return out


def _edge_smooth(img, gbuf, st, depth_m=None):
    """Soften id-buffer edges (and deep depth creases) with a 1-2-1 tent.

    The crease test runs on depth in SCENE UNITS. It used to compare the
    raw device depth, where a whole scene spans a few thousandths and the
    threshold's 0..1 slider could never land anywhere useful -- the same
    trap the DoF focus slider fell into before it was given metres (found
    by the settings audit).
    """
    tri = gbuf.tri
    edge = np.zeros(tri.shape, bool)
    dif = tri[:, 1:] != tri[:, :-1]
    edge[:, 1:] |= dif
    edge[:, :-1] |= dif
    dif = tri[1:, :] != tri[:-1, :]
    edge[1:, :] |= dif
    edge[:-1, :] |= dif
    thr = float(getattr(st, 'aa_edge_threshold', 0.1))
    if thr > 0.0:
        src = depth_m if depth_m is not None else gbuf.depth
        d = np.nan_to_num(src, nan=1.0, posinf=1.0, neginf=-1.0)
        dd = np.abs(d[:, 1:] - d[:, :-1]) > thr
        edge[:, 1:] |= dd
        edge[:, :-1] |= dd
        dd = np.abs(d[1:, :] - d[:-1, :]) > thr
        edge[1:, :] |= dd
        edge[:-1, :] |= dd
    if not edge.any():
        return img
    pad = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='edge')
    horiz = (pad[1:-1, :-2] + 2.0 * pad[1:-1, 1:-1] + pad[1:-1, 2:]) * 0.25
    pad = np.pad(horiz, ((1, 1), (0, 0), (0, 0)), mode='edge')
    tent = (pad[:-2] + 2.0 * pad[1:-1] + pad[2:]) * 0.25
    out = img.copy()
    out[edge] = tent[edge]
    return out.astype(np.float32)


def material_model(mat, settings):
    """The model a material will resolve to, without evaluating its graph.

    Only needs to be right for materials that name a model explicitly; graphs
    whose model falls out of a closure fall back to the scene shading rate.
    """
    if settings.force_model != 'NONE':
        return settings.force_model
    if mat is None:
        return settings.default_model
    # A material-level override is authoritative even when a node tree exists,
    # because Blender materials always have one. Missing this meant a material
    # set to Wireframe in the Halcyon panel shaded as flat colour and never got
    # its edges carved out.
    if getattr(mat, 'use_override', False) and getattr(mat, 'model', None):
        return mat.model
    if getattr(mat, 'graph', None):
        for node in mat.graph.get('nodes', {}).values():
            if node.get('bl_idname') == 'HALCYON_ShaderNode':
                return node.get('props', {}).get('model', settings.default_model)
            if node.get('bl_idname') == 'HALCYON_ConsoleShaderNode':
                # R252: the Console Emulation Shader resolves its machine
                # and shader type to an engine model
                from .console import model_of
                return model_of(node.get('props', {}))
            if node.get('bl_idname') == 'HALCYON_BIMaterialNode':
                return bi_matrix_model(node.get('props', {}))
            if node.get('bl_idname') in ('HALCYON_MaxStandardNode',
                                         'HALCYON_MaxRaytraceNode'):
                # R243: Max's material nodes name their model by shader
                # type (Wire takes the wireframe road)
                from ..nodes.max_nodes import MAX_MODEL_FOR
                pr = node.get('props', {})
                if pr.get('wire'):
                    return 'WIREFRAME'
                return MAX_MODEL_FOR.get(str(pr.get('shader_type', 'BLINN')),
                                         'MAX_BLINN')
        return getattr(mat, 'model', None) if getattr(mat, 'model', None) else None
    return getattr(mat, 'model', None) or settings.default_model


def bi_matrix_model(props):
    """The BI material node's props -> its packed model string.

    Shadeless wins outright (BI's toggle bypassed every lamp); otherwise
    'BI_MATRIX_{d}_{s}' carries the two menu choices in DNA order.
    """
    from .shading import BI_DIFF_ORDER, BI_SPEC_ORDER
    if props.get('shadeless'):
        return 'CONSTANT'
    d = props.get('diff_shader', 'LAMBERT')
    s = props.get('spec_shader', 'COOKTORR')
    di = BI_DIFF_ORDER.index(d) if d in BI_DIFF_ORDER else 0
    si = BI_SPEC_ORDER.index(s) if s in BI_SPEC_ORDER else 0
    return f'BI_MATRIX_{di}_{si}'


def material_wire_size(mat, default=1.0):
    """The wire width for a material, from its node if it has one.

    Same reasoning as `material_model`: for a material shaded as Wireframe by a
    Halcyon Shader node, the node is where the user set it, so the node is
    where it is read from.
    """
    if mat is None:
        return default
    graph = getattr(mat, 'graph', None)
    if graph:
        for node in graph.get('nodes', {}).values():
            if node.get('bl_idname') == 'HALCYON_ShaderNode':
                v = node.get('props', {}).get('wire_size')
                if v is not None:
                    return max(float(v), 0.05)
    return max(float(getattr(mat, 'wire_size', default) or default), 0.05)


RATE_FOR_MODEL = {'GOURAUD': 'VERTEX', 'FLAT': 'FACE', 'WIREFRAME': 'PIXEL',
                  # R251 (LIGHT-B2): the Sega boards' own rates --
                  # Model 2 one luma per polygon, Model 3 per vertex
                  **SH.RATE_FOR_MODEL}


#: BVH trees by mesh CONTENT, surviving across exports. Every F12
#: re-exports a fresh Scene, so a cache stored on the scene object -- as
#: this one was until 1.25.82 -- could never hit: the field paid its
#: 0.76 s tree build on every render of an UNCHANGED mesh, while the
#: cache's own docstring said "identity is useless across exports,
#: content is not" and then keyed on identity anyway. Small LRU: trees
#: are mesh-sized, so a handful is plenty.
_BVH_CACHE = {}
_BVH_CACHE_CAP = 4


def _cached_bvh(scene, mesh):
    """The mesh's BVH, rebuilt only when the mesh CONTENT changed.

    The key is a strided content fingerprint of the vertices AND the
    triangle indices (a re-topologised mesh with the same vertex sums
    must rebuild), the same idiom every GPU upload cache uses. The store
    is module-level so it survives the fresh Scene each export creates
    -- an F12 of an unchanged scene, an orbit, and most animation frames
    all skip the build.
    """
    v = mesh.verts
    t = mesh.tris
    stride = max(1, v.shape[0] // 512)
    tstride = max(1, t.shape[0] // 512)
    key = (int(v.shape[0]), int(t.shape[0]),
           round(float(v[::stride].sum()), 3),
           round(float(np.abs(v[::stride]).sum()), 3),
           int(t[::tstride].astype(np.int64).sum()))
    hit = _BVH_CACHE.get(key)
    if hit is not None:
        return hit
    # R182: THROUGH the disk-cache road, not around it. This site built
    # BVH() directly, so the R177 cross-session cache -- wrapped around
    # make_bvh -- never saw a single F12: the field rebuilt the same
    # 2.5 s tree every session while the cache sat empty, and the
    # instrument line never printed because this path never spoke.
    bvh = BVH.cached(mesh.verts, mesh.tris)
    if len(_BVH_CACHE) >= _BVH_CACHE_CAP:
        _BVH_CACHE.pop(next(iter(_BVH_CACHE)))
    _BVH_CACHE[key] = bvh
    return bvh


def _bump_height_fields(job, mat, mi, tri_m, bary_m, px_m, py_m, front_m,
                        blin_m, chunkn):
    """(mi, node_id) -> (gx, gy): whole-material bump gradients, pre-passed.

    `n_bump` differences its height chain toward the +x/+y neighbours,
    gated on the neighbour being shaded in the same batch -- so when a
    material is too covered for one batch, its gradients used to cut at
    every chunk boundary. This is the CPU's own height PRE-PASS, the
    same idea the deferred pass proved: evaluate the height chain over
    the material's FULL frame pixels (in bounded chunks -- heights are
    per-pixel independent, so chunking here cannot cut anything),
    scatter to a frame grid exactly as `_screen_grad` would, difference
    ONCE, and let every shading chunk gather its own pixels. Bitwise the
    same arithmetic as a whole-material batch: float32 grid, the same
    forward differences, the same validity, the same gather.
    """
    from .nodeeval import VALUE, GraphEvaluator
    graph = getattr(mat, 'graph', None)
    nodes = (graph or {}).get('nodes', {})
    bumps = [n for n in nodes.values()
             if n.get('bl_idname') == 'ShaderNodeBump']
    if not bumps:
        return {}
    H, W = job.height, job.width
    fields = {}
    for node in bumps:
        img = np.zeros((H, W), np.float32)
        valid = np.zeros((H, W), bool)
        for s in range(0, int(tri_m.size), int(chunkn)):
            e = min(s + int(chunkn), int(tri_m.size))
            ctx = job.context(tri_m[s:e], bary_m[s:e], px_m[s:e],
                              py_m[s:e], front_m[s:e]
                              if front_m is not None else None,
                              blin_m[s:e] if blin_m is not None else None,
                              0, True)
            ev = GraphEvaluator(graph, ctx, job.textures,
                                getattr(mat, 'programs', None))
            h = np.asarray(ev.input(node, 'Height', VALUE),
                           np.float32).reshape(-1)
            img[py_m[s:e], px_m[s:e]] = h
            valid[py_m[s:e], px_m[s:e]] = True
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, :-1] = np.where(valid[:, 1:] & valid[:, :-1],
                              img[:, 1:] - img[:, :-1], 0.0)
        gy[:-1, :] = np.where(valid[1:, :] & valid[:-1, :],
                              img[1:, :] - img[:-1, :], 0.0)
        fields[(int(mi), node.get('id'))] = (gx, gy)
    return fields


def _shade_all(job, tri_idx, bary, px, py, front, blin, st, progress=None):
    """Dispatch fragments to the right shading rate, per material.

    Gouraud and flat are shading *rates*, not reflectance models, so a material
    that asks for either is shaded at vertex or face frequency rather than
    having its lighting faked at pixel frequency.
    """
    scene = job.scene
    mesh = scene.mesh
    n = tri_idx.size
    out = np.zeros((n, 4), np.float32)
    mat_idx = mesh.mat_index[tri_idx] if mesh.mat_index is not None else \
        np.zeros(n, np.int32)
    rates = {}
    for i, mat in enumerate(scene.materials):
        model = material_model(mat, st)
        rates[i] = (CB.rate_for_model(model, st, mat)
                    or RATE_FOR_MODEL.get(model, st.shading_rate)) if model \
            else st.shading_rate
    per_frag = np.array([rates.get(int(m), st.shading_rate) for m in
                         range(max(len(scene.materials), 1))], dtype=object)
    frag_rate = per_frag[np.clip(mat_idx, 0, per_frag.size - 1)]
    for rate in ('PIXEL', 'VERTEX', 'FACE'):
        sel = np.nonzero(frag_rate == rate)[0]
        if sel.size == 0:
            continue
        if rate == 'PIXEL':
            # ONE MATERIAL PER CHUNKED CALL. n_bump's neighbour validity
            # is "shaded in the same batch", so a chunk boundary used to
            # cut a material's screen gradients mid-frame: one row of the
            # field's water shaded with flattened waves where fragment
            # 79917 of a 480x360 frame happened to land -- the same row
            # on every machine whose settings chunked there, and a row
            # the GPU (whole-material pre-pass) correctly did NOT
            # flatten. Shading per material makes the gradients a
            # function of the picture, not of the chunk size: any
            # material that fits MAX_CHUNK shades in one batch, and the
            # deferred plan refuses Bump materials too covered to (they
            # would still cut). The memory bound chunking exists for is
            # untouched -- chunks are still capped, just never across a
            # material's interior unless the material alone exceeds the
            # cap.
            sel_mats = mat_idx[sel]
            done = 0
            total = int(sel.size)
            workers = resolve_threads(st)
            # R251 C119 (MAT-B): the REYES micropolygon grid, once per
            # frame (None at rate 0: the snap is never called); the
            # snapped barycentrics feed BOTH the bump pre-pass and the
            # shade, so a Bump material samples its heights where it
            # shades
            reyes_grid = REYES.grid_for(job)
            for m in np.unique(sel_mats):
                sm = sel[sel_mats == m]
                nm = int(sm.size)
                bary_s = REYES.snap(bary[sm], tri_idx[sm], reyes_grid) \
                    if reyes_grid is not None else bary[sm]
                # a material too covered for ONE batch would have its
                # n_bump gradients cut at every chunk boundary -- so its
                # height chains render to whole-material gradient fields
                # FIRST (chunked themselves: heights are per-pixel
                # independent), and every shading chunk gathers from
                # those. Materials that fit one batch never pay this.
                mchunk = int(min(max(int(np.ceil(nm / max(workers * 4, 1))),
                                     MIN_CHUNK), MAX_CHUNK))
                mat = job.scene.materials[int(m)] \
                    if int(m) < len(job.scene.materials) else None
                src = getattr(job, 'bump_field_source', None)
                if src is not None and mat is not None \
                        and getattr(mat, 'graph', None) \
                        and int(m) not in {k[0] for k in job.bump_fields}:
                    # banded shading: the fields come from the G-buffer's
                    # FULL coverage (context row included), so the band's
                    # gradients equal the whole frame's
                    ftri, fbary, fpx, fpy, ffront, fblin = src
                    fm = job.scene.mesh.mat_index[ftri] \
                        if job.scene.mesh.mat_index is not None \
                        else np.zeros(ftri.size, np.int32)
                    fs = np.nonzero(fm == m)[0]
                    if fs.size:
                        fchunk = int(min(max(
                            int(np.ceil(fs.size / max(workers * 4, 1))),
                            MIN_CHUNK), MAX_CHUNK))
                        job.bump_fields.update(_bump_height_fields(
                            job, mat, int(m), ftri[fs],
                            REYES.snap(fbary[fs], ftri[fs], reyes_grid)
                            if reyes_grid is not None else fbary[fs],
                            fpx[fs], fpy[fs],
                            ffront[fs] if ffront is not None else None,
                            fblin[fs] if fblin is not None else None,
                            fchunk))
                elif mchunk < nm and mat is not None \
                        and getattr(mat, 'graph', None):
                    job.bump_fields.update(_bump_height_fields(
                        job, mat, int(m), tri_idx[sm], bary_s, px[sm],
                        py[sm], front[sm] if front is not None else None,
                        blin[sm] if blin is not None else None, mchunk))
                if progress is not None:
                    def _prog(v, msg, _d=done, _nm=nm):
                        # remap this material's local shade fraction into
                        # the frame's global one, keeping the bar (and the
                        # preview's abort ticks) monotonic
                        local = min(max((v - 0.35) / 0.30, 0.0), 1.0)
                        progress(0.35 + 0.30 * ((_d + local * _nm)
                                                / max(total, 1)), msg)
                else:
                    _prog = None
                out[sm] = _shade_chunked(job, tri_idx[sm], bary_s,
                                         px[sm], py[sm], front[sm],
                                         blin[sm] if blin is not None
                                         else None,
                                         st, progress=_prog)
                done += nm
        else:
            out[sel] = _shade_interpolated(
                job, tri_idx[sel], bary[sel], rate, st,
                px[sel] if px is not None else None,
                py[sel] if py is not None else None,
                front[sel] if front is not None else None,
                blin[sel] if blin is not None else None)
    return out


MIN_CHUNK = 16384            # below this, per-chunk overhead dominates
SMALL_CHUNK = 2048           # ...but an idle core costs more than that overhead
MAX_CHUNK = 262144           # above this, one chunk's temporaries get large


def resolve_threads(settings):
    n = int(getattr(settings, 'threads', 0) or 0)
    if n <= 0:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 1
    return max(1, min(int(n), 64))


def _shade_chunked(job, tri_idx, bary, px, py, front, blin, st, progress=None):
    """Shade in bounded chunks, across threads when there are cores to use.

    `progress` ticks between chunks. Beyond a better progress bar, it is
    what makes a viewport render ABORTABLE mid-shade: the preview's tick
    raises when a newer view supersedes this one, and before this the
    abort could only land between whole stages -- which for shade meant
    after the expensive part had already run to completion.

    Deferred shading makes every fragment independent, so this is a clean split.
    The workers only touch bpy-free code and read-only shared state, and NumPy
    drops the GIL for the array work, so the threads do real parallel work
    rather than taking turns.

    Chunking matters on its own even single-threaded: a 3440x1440 frame at 4x
    supersampling is 79 million fragments, and building the full shading context
    for all of them at once would need tens of gigabytes. Bounded chunks make
    the peak memory a function of chunk size and thread count, not resolution.
    """
    n = int(tri_idx.size)
    out = np.zeros((n, 4), np.float32)
    if n == 0:
        return out

    def slice_of(a, s, e):
        return None if a is None else a[s:e]

    workers = resolve_threads(st)
    chunk = int(np.ceil(n / max(workers * 4, 1)))
    chunk = int(min(max(chunk, MIN_CHUNK), MAX_CHUNK))
    # Subdividing below this floor to give every worker a chunk was tried and
    # measured *worse* on a 20-core machine -- 1.00x became 0.87x -- because
    # the shading path is not actually thread-limited by chunk count. See the
    # threading note in the README: NumPy releases the interpreter lock only
    # for large operations, and the node evaluator is dominated by Python
    # dispatch between small ones. More chunks bought only more overhead.
    starts = list(range(0, n, chunk))

    if workers == 1 or len(starts) == 1:
        for s in starts:
            if progress and s:
                progress(0.35 + 0.30 * (s / n), 'Shading')
            e = min(s + chunk, n)
            out[s:e] = job.shade(tri_idx[s:e], bary[s:e], slice_of(px, s, e),
                                 slice_of(py, s, e), slice_of(front, s, e),
                                 slice_of(blin, s, e))
        return out

    # Threads are not permitted in a Blender extension, so chunks run in
    # sequence here and real parallelism comes from the worker processes
    # instead. The chunking still earns its place: it is what bounds peak
    # memory, which is otherwise a function of resolution.
    job.prewarm()
    seed = int(getattr(st, 'seed', 0) or 0)
    for i, s in enumerate(starts):
        if progress and s:
            progress(0.35 + 0.30 * (s / n), 'Shading')
        e = min(s + chunk, n)
        out[s:e] = job.shade(tri_idx[s:e], bary[s:e], slice_of(px, s, e),
                             slice_of(py, s, e), slice_of(front, s, e),
                             slice_of(blin, s, e),
                             rng=np.random.default_rng(seed + i * 7919))
    return out


def _shade_interpolated(job, tri_idx, bary, rate, st=None,
                        px=None, py=None, front=None, blin=None):
    """Shade at vertex or face rate, then interpolate to the fragments.

    THE LIGHTING is what these rates interpolate -- not the texture.
    Hardware that shaded per vertex still sampled the texture at every
    pixel and multiplied: `texel x vertex colour`, the MODULATE
    combiner every fixed-function pipeline of the era implemented, and
    the reason a Gouraud-shaded PlayStation model shows soft banded
    light over a SHARP texture. Halcyon evaluated the whole material
    at the vertices instead, texture included, so a textured model
    came out smeared across its own triangles -- the field photographed
    exactly that on a 1209-triangle character whose preset selects
    Gouraud. So: lighting over a WHITE surface at the vertex or face
    rate, interpolated; albedo and alpha at the PIXEL rate; multiplied.

    An untextured material is unaffected -- its albedo is one colour,
    so multiplying it back in reproduces the old result exactly. The
    banding and the missed highlights, which are the point of a
    shading RATE, are untouched: they live in the lighting term.

    The vertex and face passes went through job.shade() directly, which meant
    they never used the thread pool -- so every preset with a Gouraud or flat
    shading rate, which is most of the console and home-computer ones, rendered
    single-threaded no matter what the thread count said.
    """
    mesh = job.scene.mesh
    uniq = np.unique(tri_idx)
    saved = getattr(job, 'rate_mode', None)
    try:
        job.rate_mode = 'LIGHT'
        col, lookup = shade_vertex_rate(job, uniq, rate, st)
    finally:
        job.rate_mode = saved
    if rate == 'FACE':
        order = np.searchsorted(uniq, tri_idx)
        light = col[order]
        c0 = c1 = c2 = None
    else:
        tris = mesh.tris[tri_idx]
        c0 = col[lookup[tris[:, 0]]]
        c1 = col[lookup[tris[:, 1]]]
        c2 = col[lookup[tris[:, 2]]]
        light = (c0 * bary[:, 0:1] + c1 * bary[:, 1:2]
                 + c2 * bary[:, 2:3]).astype(np.float32)
    try:
        job.rate_mode = 'ALBEDO'
        alb = _shade_chunked(job, tri_idx, bary, px, py, front, blin, st) \
            if st is not None else \
            job.shade(tri_idx, bary, px, py, front, blin)
    finally:
        job.rate_mode = saved
    out = np.empty_like(light)
    # R251 (MAT-A): the period combiners' hook -- None unless a period
    # model is present, and then the plain product on every other pixel
    period = CB.recombine(job, tri_idx, light, c0, c1, c2, bary, alb, px,
                          py, st)
    out[:, :3] = light[:, :3] * alb[:, :3] if period is None else period
    # alpha comes from the PIXEL pass: a cut-out texture's edge is the
    # one thing that must never be interpolated between vertices
    out[:, 3] = alb[:, 3]
    if job.pass_sink is not None and getattr(job, 'pass_sink_armed', False) \
            and px is not None:
        # R253: the corners shaded with no pixel of their own (px None in
        # shade_vertex_rate), so the split cannot be interpolated per
        # lobe: a vertex- or face-rate material reports its WHOLE lit
        # colour as Diffuse (and as Light00), its albedo as Color,
        # nothing specular / ambient / emissive, shadow and AO open
        for _name, _buf in job.pass_sink.items():
            if _name in ('Diffuse', 'Light00'):
                _buf[py, px] = out[:, :3]
            elif _name == 'Color':
                _buf[py, px] = alb[:, :3]
            elif _name in ('Shadow', 'AO'):
                _buf[py, px] = np.float32(1.0)
            else:
                _buf[py, px] = np.float32(0.0)
    return out


def vertex_light_corners(job, mi, rate, st):
    """(T*3, 4) float32: each triangle corner's lighting over WHITE.

    The LIGHT half of the R66 Gouraud split, packed for the deferred
    pass: for every triangle of material `mi`, the three corners carry
    the full CPU lighting result -- shadows, rays, env, the silhouette
    cheats, everything shade_batch runs -- evaluated over a white
    surface at the vertex (or once per face, packed to all three
    corners equal). The GPU pass interpolates these by the G-buffer's
    own barycentrics and multiplies by its per-pixel albedo, so the
    corner VALUES are the CPU's own numbers and the seam is the
    interpolation arithmetic alone. Rows of triangles that belong to
    other materials stay zero; the pass's keep test never fetches them.
    """
    mesh = job.scene.mesh
    T = int(mesh.tris.shape[0])
    out = np.zeros((T * 3, 4), np.float32)
    if mesh.mat_index is None:
        sel = np.arange(T, dtype=np.int64)
    else:
        sel = np.nonzero(np.asarray(mesh.mat_index) == mi)[0]
    if sel.size == 0:
        return out
    saved = getattr(job, 'rate_mode', None)
    try:
        job.rate_mode = 'LIGHT'
        col, lookup = shade_vertex_rate(job, sel, rate, st)
    finally:
        job.rate_mode = saved
    if rate == 'FACE':
        CB.pack_face_corners(out, sel, col, job, st, mi)
    else:
        tris = mesh.tris[sel]
        for c in range(3):
            out[sel * 3 + c] = col[lookup[tris[:, c]]]
    return out


def polygon_depths(mesh, view, eye, mode='CENTROID', ot=None):
    """One depth per triangle, for Painter's algorithm.

    Painter's does not compare fragments, it compares whole polygons: the one
    whose chosen depth is nearest wins the pixel outright. Giving every fragment
    of a triangle that single depth turns the existing z-buffer into exactly
    that comparison, and reproduces the algorithm's real failures -- surfaces
    that interpenetrate meet along a polygon edge instead of their true
    intersection, and a large polygon can be occluded by a small nearer one it
    actually passes in front of.
    """
    v = mesh.verts[mesh.tris]                       # (T, 3, 3)
    # depth of each corner, then reduce -- taking the min or max of the
    # positions instead would pick a corner per axis and mean nothing
    rel = v - eye[None, None, :]
    per_vertex = np.abs(rel @ view[:3, :3].T)[:, :, 2]
    if mode == 'NEAREST':
        depth = per_vertex.min(axis=1)
    elif mode == 'FARTHEST':
        depth = per_vertex.max(axis=1)
    elif mode == 'ORDERING_TABLE' and ot is not None:
        # R251 C004: the PS1's AVSZ3 average floored into one of L
        # integer buckets to `far`; +inf is "not drawn" (a polygon past
        # the table never writes: the fill's test fails against inf and
        # its tie branch against EMPTY). The bucket IS the flat depth,
        # so equal buckets resolve to the lowest id -- the PS1's
        # first-added-on-top, since Halcyon's ids are submission order.
        # float32, one op per statement
        L = max(int(ot[0]), 2)
        far = max(float(ot[1]), 1e-6)
        z_avg = per_vertex.mean(axis=1).astype(np.float32)
        bucket = z_avg * np.float32(L)
        bucket = bucket / np.float32(far)
        bucket = np.floor(bucket)
        depth = np.where((bucket <= 0.0) | (bucket >= np.float32(L)),
                         np.float32(np.inf), bucket).astype(np.float32)
    else:
        depth = per_vertex.mean(axis=1)
    # matches the z-buffer's convention: smaller is nearer
    return depth.astype(np.float32)


#: how the last frame classified its materials: which went to the
#: A-buffer and WHY. A surface in the transparent pass is rasterised
#: with cull NONE and no depth write, so its back faces and everything
#: behind it stack as depth layers -- correct for glass, and a very
#: convincing "the depth is broken" for a solid character whose
#: materials were merely FLAGGED see-through. The printed line names
#: the flag, so the field never has to guess which pass a surface is in.
LAST_SPLIT = {}


def _split_by_alpha(scene, mesh, st=None):
    """Triangle indices for the opaque and the transparent passes.

    Opaque and Screen Door do not use a separate pass at all: their geometry
    belongs in the depth-buffered pass with everything else. Splitting it out
    and then never shading it is what made both modes render nothing.
    """
    LAST_SPLIT.clear()
    if mesh.mat_index is None:
        return None, np.zeros(0, np.int32)
    from .scene import material_is_absent as _absent
    from .scene import material_see_through as _see
    from .volume import material_is_volume as _miv
    if st is not None and st.transparency in ('NONE', 'STIPPLE'):
        # R222: even with transparency off, a volume container's
        # triangles stay out of the raster -- they are the marcher's
        # bound, never a surface. R247: so does a CLIP material whose
        # constant alpha is below its threshold -- no fragment of it
        # exists in any mode (an outline shell left flat on the body
        # z-fought it into black speckles under these two modes)
        vmask = np.array([_miv(m) or _absent(m) for m in scene.materials]
                         or [False], bool)
        gone = {str(getattr(m, 'name', None) or f'material {i}'):
                'constant alpha below the threshold'
                for i, m in enumerate(scene.materials)
                if _absent(m) and not _miv(m)}
        if gone:
            LAST_CLIP.clear()
            LAST_CLIP.update(absent=gone, refused={}, materials=0, auto=[],
                             fragments=0, evaluated=0, kept=0, promoted=0)
        if vmask.any():
            mi0 = np.clip(mesh.mat_index, 0, vmask.size - 1)
            keep = ~vmask[mi0]
            LAST_SPLIT.update(tris_volume=int((~keep).sum()))
            return np.nonzero(keep)[0].astype(np.int32), \
                np.zeros(0, np.int32)
        return None, np.zeros(0, np.int32)
    see_through = np.zeros(max(len(scene.materials), 1), bool)
    is_vol = np.zeros(max(len(scene.materials), 1), bool)
    reasons = {}
    absent = {}
    for i, m in enumerate(scene.materials):
        # R251: ONE predicate (core/scene.py `material_see_through`),
        # shared with the GPU layer loop, so the layer plan and this
        # split can never disagree on which material holds fragments:
        # a constant opacity below 1, export.py's `_alpha_reason`
        # evidence (a transparent node, or which Alpha socket -- a vague
        # reason is what let a whole mis-flagged character hide for
        # three rounds behind "flagged on export"), a per-material Blend
        # Mode (a PS1 additive glow at opacity 1.0 must reach the
        # transparent pass), the PS2 two-pass Clip+Blend; an Env hole is
        # opaque here
        why = _see(m)
        see_through[i] = why is not None
        if why is not None:
            reasons[str(getattr(m, 'name', None) or f'material {i}')] = why
        # R222: a volume container draws NO surface -- its triangles are
        # only the marcher's bound, in neither raster pass
        if _miv(m):
            is_vol[i] = True
            reasons[str(getattr(m, 'name', None) or f'material {i}')] = \
                'volume container (marched, no surface)'
        elif _absent(m):
            # R247: a punch-through with a constant alpha below its
            # threshold has no fragment: in neither pass, the same as
            # the clip stage would have resolved it -- without the
            # stage's evaluation
            is_vol[i] = True
            reasons[str(getattr(m, 'name', None) or f'material {i}')] = \
                'Alpha Mode Clip at a constant alpha below the threshold (no surface)'
            absent[str(getattr(m, 'name', None) or f'material {i}')] = \
                'constant alpha below the threshold'
    mi = np.clip(mesh.mat_index, 0, see_through.size - 1)
    t = see_through[mi]
    vol_t = is_vol[mi]
    LAST_SPLIT.update(reasons=reasons, materials=int(see_through.size),
                      see_through=int(see_through.sum()),
                      tris=int(mesh.mat_index.size),
                      tris_see_through=int(t.sum()),
                      tris_volume=int(vol_t.sum()))
    if absent:
        # the clip instrument tells the truth for this frame even when
        # the stage never runs: nothing kept, these resolved by absence
        LAST_CLIP.clear()
        LAST_CLIP.update(absent=absent, refused={}, materials=0, auto=[],
                         fragments=0, evaluated=0, kept=0, promoted=0)
    return (np.nonzero(~t & ~vol_t)[0].astype(np.int32),
            np.nonzero(t & ~vol_t)[0].astype(np.int32))


#: R211: what the last frame's punch-through stage did -- materials on
#: the clip road, refusals by name, fragment counts, pixels promoted.
#: The report line reads from it; tests hold it so the fast road can
#: never silently be the slow one.
LAST_CLIP = {}


def _clip_partition(scene, mesh, st, transparent, affine=False):
    """Split the see-through subset into punch-through and blend tris.

    A CLIP material whose alpha is one evaluable chain (scene.clip_socket)
    leaves the blend road entirely: its triangles resolve visibility in
    `_promote_clip` and join the depth-buffered pass. Anything CLIP that
    cannot lift its alpha out stays on the blend road -- by name -- where
    the shading law still delivers the hard 0/1. Returns
    (clip_tris_or_None, blend_tris, plans).
    """
    from .scene import clip_road
    absent = LAST_CLIP.get('absent')
    LAST_CLIP.clear()
    if absent:
        # R247: the materials the split resolved by absence stay named
        LAST_CLIP['absent'] = absent
    if transparent is None or transparent.size == 0:
        return None, transparent, {}
    plans, refused, auto = {}, {}, []
    for i, m in enumerate(scene.materials):
        name = str(getattr(m, 'name', None) or f'material {i}')
        thr, kind, nd, extra, note = clip_road(m)
        if thr is None:
            # a CLIP material that cannot lift its alpha refuses BY
            # NAME; a plain Blend material is simply on its own road
            if str(getattr(m, 'alpha_mode', 'BLEND')) == 'CLIP':
                refused[name] = str(note)
            continue
        if affine:
            refused[name] = ('affine texturing shades layers from '
                             'screen-linear coordinates the promoted '
                             'pass does not carry')
            continue
        plans[i] = (kind, nd, extra, float(thr))
        if note == 'binary alpha, detected':
            # R213: a Blend material whose alpha chain provably yields
            # only 0 or 1 -- the blend road and the z-buffer give the
            # IDENTICAL picture, so it takes the fast road unasked.
            # Old scenes' fur materials are exactly this
            auto.append(name)
    LAST_CLIP.update(refused=refused, materials=len(plans), auto=auto)
    if not plans:
        return None, transparent, plans
    mi = np.clip(mesh.mat_index[transparent], 0,
                 max(len(scene.materials), 1) - 1)
    is_clip = np.isin(mi, np.array(sorted(plans), np.int64))
    # R251 C031 (PS2 AFAIL): a Clip+Blend material's triangles resolve
    # their opaque half in the z-pass AND stay on the blend road for
    # the sub-threshold remainder (drawn a second time, no depth write)
    is_cb = np.isin(mi, np.array(
        [i for i in plans
         if str(getattr(scene.materials[i], 'alpha_mode', 'BLEND'))
         == 'CLIP_BLEND'], np.int64))
    clip_sub = transparent[is_clip]
    blend_sub = transparent[~is_clip | is_cb]
    return (clip_sub if clip_sub.size else None), blend_sub, plans


def _chain_attr_needs(graph, node, sockname):
    """Which context attributes an alpha chain can touch, statically.

    Walks the nodes reachable from `node`'s `sockname` input. Every
    node type in the whitelist declares what it reads; ONE unknown node
    returns None, which means "build the full context" -- lean is an
    optimisation, never a semantics. The fur chain (TexCoord ▸ UV,
    Fur Tufts, Hair Info, Greater Than) resolves to {uv, vcol}: two
    interpolations instead of six plus a view-vector normalise."""
    nodes = (graph or {}).get('nodes', {}) or {}
    pure = {
        'ShaderNodeMath', 'ShaderNodeVectorMath', 'ShaderNodeValToRGB',
        'ShaderNodeRGB', 'ShaderNodeValue', 'ShaderNodeMixRGB',
        'ShaderNodeMix', 'ShaderNodeInvert', 'ShaderNodeGamma',
        'ShaderNodeBrightContrast', 'ShaderNodeHueSaturation',
        'ShaderNodeRGBToBW', 'ShaderNodeSeparateColor',
        'ShaderNodeCombineColor', 'ShaderNodeSeparateRGB',
        'ShaderNodeCombineRGB', 'ShaderNodeSeparateXYZ',
        'ShaderNodeCombineXYZ', 'ShaderNodeSeparateHSV',
        'ShaderNodeCombineHSV', 'ShaderNodeMapRange', 'ShaderNodeClamp',
        'ShaderNodeMapping', 'ShaderNodeFloatCurve',
        'ShaderNodeRGBCurve', 'ShaderNodeVectorCurve',
    }
    # the 26 pattern nodes that read only their Vector (or, unlinked,
    # the generated coordinates) -- Matcap reads N/I and is NOT here
    pat = {f'HALCYON_{k}Node' for k in (
        'Marble', 'Wood', 'Granite', 'Dents', 'Crackle', 'Plasma',
        'Ripples', 'Starfield', 'Weave', 'Scratches', 'Tiles', 'Spiral',
        'Noise', 'Caustics', 'Water', 'Gradient', 'Cells', 'Static',
        'FurTufts', 'Bozo', 'Agate', 'Leopard', 'Onion', 'Bumps',
        'Wrinkles', 'Brick')}
    texco = ({'generated'}, {'N'}, {'uv'}, {'P'}, {'P'}, {'P'},
             {'N', 'I'})
    needs = set()
    seen = set()

    def walk(nid, out_idx):
        if (nid, out_idx) in seen:
            return True
        seen.add((nid, out_idx))
        nd = nodes.get(nid)
        if nd is None:
            return False
        bid = nd.get('bl_idname', '')
        if bid == 'ShaderNodeTexCoord':
            if not (0 <= out_idx < len(texco)):
                return False
            needs.update(texco[out_idx])
            return True
        if bid == 'ShaderNodeHairInfo':
            needs.add('vcol')
            return True
        if bid == 'ShaderNodeVertexColor':
            needs.add('vcol')
            return True
        if bid == 'ShaderNodeUVMap':
            needs.update(('uv', 'uv2'))
            return True
        if bid in pat:
            vec_linked = any(sk.get('name') == 'Vector' and sk.get('link')
                             for sk in nd.get('inputs', ()))
            if not vec_linked:
                needs.add('generated')
        elif bid not in pure:
            return False
        for sk in nd.get('inputs', ()):
            link = sk.get('link')
            if link and not walk(link[0], int(link[1])
                                 if len(link) > 1 else 0):
                return False
        return True

    for sk in (node or {}).get('inputs', ()):
        if sk.get('name') != sockname:
            continue
        link = sk.get('link')
        if not link:
            return set()
        return needs if walk(link[0], int(link[1])
                             if len(link) > 1 else 0) else None
    return None


#: R212: the punch-through memo. A viewport refine re-renders an
#: UNCHANGED frame; recomputing an identical resolve for it was most of
#: the field's "still crippled". Keyed on the same content the G-buffer
#: cache trusts, plus the clip side's own fingerprint and the frame.
_CLIP_CACHE = {}
_CLIP_CAP = 6


def _clip_fingerprint(scene, clip_tris, plans):
    parts = [int(clip_tris.size),
             int(clip_tris[::257].astype(np.int64).sum())]
    for mi in sorted(plans):
        kind, nd, extra, thr = plans[mi]
        mat = scene.materials[mi]
        if kind == 'const':
            sig = float(extra)
        else:
            g = getattr(mat, 'graph', None) or {}
            nodes = g.get('nodes', {}) or {}
            bits = []
            stack = [(nd.get('id'), None)]
            seen = set()
            while stack:
                nid, _o = stack.pop()
                if nid in seen or nid not in nodes:
                    continue
                seen.add(nid)
                d = nodes[nid]
                bits.append((nid, d.get('bl_idname', ''),
                             repr(sorted((d.get('props') or {}).items())),
                             tuple((sk.get('name'),
                                    repr(sk.get('default')),
                                    tuple(sk.get('link') or ()))
                                   for sk in d.get('inputs', ()))))
                for sk in d.get('inputs', ()):
                    if sk.get('link'):
                        stack.append((sk['link'][0], None))
            sig = hash(repr(sorted(bits)))
        parts.append((mi, kind, float(thr), sig))
    return tuple(parts)


def _promote_clip(job, gbuf, vp, st, clip_tris, plans, snap, flat_depth,
                  scissor, subdiv_px, near_eps, ckey=None):
    """Resolve CLIP visibility with alpha-tested rasterisation (R212).

    The 1.55.0 road collected every clip fragment into a list, sorted
    millions to keep thousands, and built full shading contexts to
    evaluate a two-attribute chain -- the field was right to call it a
    calculation problem. Now the test runs INSIDE the raster, the way
    the era's hardware ran it: clip triangles rasterise near-first in
    depth-ordered chunks against a live z-buffer seeded with the
    tolerant limit of the opaque depth, each candidate fragment
    evaluates ONLY its alpha chain over ONLY the attributes that chain
    can read, survivors write z immediately -- so everything behind an
    already-solid tuft dies at the depth test unevaluated. No fragment
    lists, no global sort. The winner per pixel is the same one the old
    road picked (nearest solid, ties to the lowest triangle id): the
    z-buffer IS that rule. And because an unchanged frame resolves to
    the same answer, the whole result is memoised: an idle viewport
    refine replays six scatters instead of re-resolving a pelt.
    """
    from .nodeeval import VALUE, GraphEvaluator
    mesh = job.scene.mesh
    key = None
    if ckey is not None:
        key = (ckey, _clip_fingerprint(job.scene, clip_tris, plans),
               int(job.scene.frame), float(job.scene.time))
        hit = _CLIP_CACHE.get(key)
        if hit is not None:
            _CLIP_CACHE[key] = _CLIP_CACHE.pop(key)      # LRU touch
            wy, wx = hit['wy'], hit['wx']
            gbuf.depth[wy, wx] = hit['depth']
            gbuf.zndc[wy, wx] = hit['depth']
            gbuf.tri[wy, wx] = hit['tri']
            gbuf.bary[wy, wx] = hit['bary']
            gbuf.front[wy, wx] = hit['front']
            LAST_CLIP.update(hit['stats'])
            LAST_CLIP['cache'] = 'HIT'
            return
    LAST_CLIP['cache'] = 'MISS' if ckey is not None else 'off'

    # the clip depth pass: its own buffer, seeded at the TOLERANT limit
    # of the opaque depth so coplanar shell-meets-skin contacts promote
    # exactly as a blend layer would have composited
    cg = raster.GBuffer(gbuf.width, gbuf.height)
    _ro = getattr(st, '_raster_opts', None)
    if _ro is not None and getattr(_ro, 'clear', None) is not None:
        # R251 C038: CLIP fragments z-test against the backdrop too
        cg.clear_depth(_ro.clear[0], _ro.clear[1], _ro.enc)
    cg.depth[:] = raster.abuf_depth_limit(gbuf.depth)
    cg.zndc[:] = cg.depth

    thr_of = np.zeros(max(len(job.scene.materials), 1), np.float32)
    for mi, (_k, _n, _x, t_) in plans.items():
        thr_of[mi] = max(float(t_), 1e-6)
    lean = {}
    for mi, (kind, nd, extra, _t) in plans.items():
        if kind == 'node':
            mat = job.scene.materials[mi]
            lean[mi] = _chain_attr_needs(getattr(mat, 'graph', None),
                                         nd, str(extra))
    stats = {'offered': 0, 'evaluated': 0}
    mat_index = mesh.mat_index

    def frag_test(tri_ids, px, py, bary, front):
        keep = np.zeros(tri_ids.size, bool)
        stats['offered'] += int(tri_ids.size)
        mat_f = mat_index[tri_ids] if mat_index is not None \
            else np.zeros(tri_ids.size, np.int32)
        for mi, (kind, nd, extra, _t) in plans.items():
            idx = np.nonzero(mat_f == mi)[0]
            if idx.size == 0:
                continue
            if kind == 'const':
                a = np.full(idx.size, float(extra), np.float32)
            else:
                mat = job.scene.materials[mi]
                graph = getattr(mat, 'graph', None)
                a = np.empty(idx.size, np.float32)
                stats['evaluated'] += int(idx.size)
                for s0 in range(0, int(idx.size), int(MAX_CHUNK)):
                    e0 = min(s0 + int(MAX_CHUNK), int(idx.size))
                    sub = idx[s0:e0]
                    val = None
                    if lean.get(mi) is not None:
                        # the evaluator NEVER raises out of a node: it
                        # records the error and falls back -- so a lean
                        # context that starved a node shows up in
                        # ev.errors, not as an exception. Check, and
                        # redo with everything. Loud fallback, never
                        # silent wrong pixels.
                        try:
                            ctx = job.context(tri_ids[sub], bary[sub],
                                              px[sub], py[sub],
                                              front[sub], None, 0, True,
                                              need=lean[mi])
                            ev = GraphEvaluator(
                                graph, ctx, job.textures,
                                getattr(mat, 'programs', None))
                            val = np.asarray(
                                ev.input(nd, str(extra), VALUE),
                                np.float32).reshape(-1)
                            if ev.errors:
                                val = None
                        except Exception:               # noqa: BLE001
                            val = None
                        if val is None:
                            lean[mi] = None
                    if val is None:
                        ctx = job.context(tri_ids[sub], bary[sub],
                                          px[sub], py[sub], front[sub],
                                          None, 0, True)
                        ev = GraphEvaluator(graph, ctx, job.textures,
                                           getattr(mat, 'programs',
                                                   None))
                        val = np.asarray(
                            ev.input(nd, str(extra), VALUE),
                            np.float32).reshape(-1)
                    a[s0:e0] = val
            if st.alpha_threshold > 0.0:
                # the global hard cutoff runs FIRST in the shading law;
                # promoted visibility must agree with the shaded alpha
                a = np.where(a >= st.alpha_threshold, a, 0.0)
            keep[idx] = a >= thr_of[mi]
        return keep

    # near-first depth-ordered chunks: the clip z tightens between
    # chunks, so far fragments die before evaluation. Order affects
    # only HOW MUCH is evaluated -- the z-buffer's tie rule makes the
    # winner order-free, chunking included
    cent = mesh.verts[mesh.tris[clip_tris]].mean(axis=1)
    wrow = vp[3, :3]
    zkey = cent @ wrow + vp[3, 3]
    order = np.argsort(zkey, kind='stable')
    n_chunks = int(min(24, max(1, order.size // 4096)))
    for ch in np.array_split(order, n_chunks):
        if ch.size == 0:
            continue
        raster.rasterize(mesh.verts, mesh.tris, vp, job.width,
                         job.height, cull='NONE', snap=snap,
                         depth_bits=st.depth_precision,
                         subset=clip_tris[ch], gbuf=cg,
                         depth_write=True, flat_depth=flat_depth,
                         scissor=scissor, subdiv_px=subdiv_px,
                         near_eps=near_eps, frag_test=frag_test,
                         opts=getattr(st, '_raster_opts', None))

    win = cg.mask()
    n_win = int(win.sum())
    LAST_CLIP.update(fragments=stats['offered'],
                     evaluated=stats['evaluated'],
                     kept=n_win, promoted=n_win)
    if n_win == 0:
        return
    wy, wx = np.nonzero(win)
    gbuf.depth[wy, wx] = cg.depth[wy, wx]
    gbuf.zndc[wy, wx] = cg.depth[wy, wx]
    gbuf.tri[wy, wx] = cg.tri[wy, wx]
    gbuf.bary[wy, wx] = cg.bary[wy, wx]
    gbuf.front[wy, wx] = cg.front[wy, wx]
    if key is not None:
        while len(_CLIP_CACHE) >= _CLIP_CAP:
            _CLIP_CACHE.pop(next(iter(_CLIP_CACHE)))
        _CLIP_CACHE[key] = {
            'wy': wy.copy(), 'wx': wx.copy(),
            'depth': cg.depth[wy, wx].copy(),
            'tri': cg.tri[wy, wx].copy(),
            'bary': cg.bary[wy, wx].copy(),
            'front': cg.front[wy, wx].copy(),
            'stats': {'fragments': stats['offered'],
                      'evaluated': stats['evaluated'],
                      'kept': n_win, 'promoted': n_win}}

def depth_report(proj, gbuf, depth_bits, depth_sort='ZBUFFER'):
    """What this frame's z-buffer can actually resolve, in world units.

    "The depth is wrong" is a picture; this is the number behind it. The
    near and far planes come back out of the projection matrix, the
    frame's own covered depths say where the subject sits, and the
    N-bit grid step converts to the smallest world separation two
    surfaces can have and still be told apart THERE. Depth in a
    perspective frame is hyperbolic: resolution falls with the SQUARE
    of distance, so a near plane set very close spends the whole buffer
    on empty air in front of the subject -- the classic cause of
    surfaces tearing through each other at a normal bit depth. Returns
    a one-line string, or None if the frame has no covered pixels.
    """
    cov = gbuf.tri >= 0
    if proj is None or not cov.any():
        return None
    if str(depth_sort).upper() == 'PAINTERS':
        # Painter's does not store ndc depth at all: every fragment of a
        # polygon carries that polygon's single VIEW distance, so the
        # buffer holds distances, not ndc values. Reading them as ndc is
        # what made this line announce that the field's frame sat at its
        # projection's depth asymptote -- it was reporting a scene 4.5
        # to 5.5 units from the camera.
        d = gbuf.depth[cov].astype(np.float64)
        d = d[np.isfinite(d)]
        rng = f'{d.min():.4g}..{d.max():.4g}' if d.size else 'empty'
        return ('[Halcyon] depth: Painter\'s algorithm -- ONE depth per '
                f'polygon (view distance {rng}), no per-pixel z-buffer, '
                f'so the {int(depth_bits)}-bit setting does not apply. '
                'Polygons that interpenetrate meet along an edge '
                'instead of their true intersection, and a large '
                'polygon can be hidden by a small nearer one: that is '
                'the algorithm, not a fault. Depth Method -> Z-Buffer '
                'resolves per pixel')
    p = np.asarray(proj, np.float64)
    a, b = float(p[2, 2]), float(p[2, 3])
    denom_n, denom_f = a - 1.0, a + 1.0
    if abs(denom_n) < 1e-12 or abs(denom_f) < 1e-12:
        return None                     # orthographic: depth is linear
    near, far = b / denom_n, b / denom_f
    if not (0.0 < near < far):
        return None
    z = gbuf.zndc[cov].astype(np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    zlo, zhi = float(z.min()), float(z.max())
    span = far - near
    head = (f'[Halcyon] depth: {int(depth_bits)}-bit z-buffer, clip '
            f'{near:.4g}..{far:.4g}, ndc z {zlo:.6f}..{zhi:.6f}')
    # A covered pixel at or past ndc z = 1 sits AT or BEYOND the far
    # plane. Far clipping is off by design -- period renderers drew it
    # -- but no distance can be recovered from such a value, and the
    # first version of this line clamped the vanishing denominator and
    # reported the subject at 2e+14 world units away. An instrument
    # that lies is worse than no instrument: it must say "I cannot
    # measure this, and here is why".
    # ndc z reaches 1 exactly AT the far plane and approaches
    # (f+n)/(f-n) as distance runs to infinity, so a value past 1 is
    # past the far clip (drawn anyway -- period renderers did) and a
    # value at the asymptote carries no recoverable distance at all:
    # in float32 that is what an enormous scene scale collapses to.
    denom = (far + near) - z * span
    good = denom > 1e-9
    past = int((z > 1.0).sum())
    if not good.any():
        return (head + '; every covered pixel sits at this '
                'projection\'s depth asymptote, where NO distance can '
                'be recovered -- the geometry is effectively infinitely '
                'far for this clip range, so the far clip needs to come '
                'in (or the scene scale down) before depth means '
                'anything')
    dist = 2.0 * far * near / denom[good]
    d_near, d_med = float(dist.min()), float(np.median(dist))
    steps = float((1 << int(max(2, min(int(depth_bits), 32)))) - 1)
    step_ndc = 2.0 / steps

    def res_at(d):
        # d(ndc z)/d(distance) = 2*f*n / ((f-n) * d^2)
        return step_ndc * span * d * d / (2.0 * far * near)

    tail = ''
    if past:
        tail += (f'; {100.0 * past / z.size:.0f}% of covered pixels are '
                 'PAST the far clip plane (drawn anyway, but their '
                 'depth is outside the range the buffer was set up for)')
    if not good.all():
        tail += (f'; {100.0 * float((~good).sum()) / good.size:.0f}% sit '
                 'at the depth asymptote and cannot be measured at all')
    return (head + f'; the subject sits at {d_near:.3g}..{d_med:.3g}, '
            f'where the buffer resolves {res_at(d_near):.3g}..'
            f'{res_at(d_med):.3g} world units -- surfaces closer '
            'together than that cannot be told apart' + tail)


def fog_coverage_note(proj, gbuf, settings):
    """One console line when the fog swallows the frame.

    R205: a scene whose geometry all sits past Fog End renders as a
    flat sheet of fog colour -- correct arithmetic, useless picture,
    and from the output alone indistinguishable from a broken renderer
    (the field diagnosed it as "the fog is not right"). This names the
    numbers: how much of the covered frame is at FULL fog, and where
    Fog End sits against the subject. Distance-mode fog only (LINEAR /
    TABLE16 read Start/End; the EXP modes have no end to compare).
    Returns the line, or None when there is nothing to warn about.
    """
    mode = str(getattr(settings, 'fog_mode', 'LINEAR'))
    if mode not in ('LINEAR', 'TABLE16', 'GTE_1Z'):
        return None
    cov = gbuf.tri >= 0
    if proj is None or not cov.any():
        return None
    from . import fog as _FOG
    if _FOG.depth_of(settings) == 'Z':
        # F007: under Fog Depth Z, Start and End are post-projection
        # depths in 0..1. Compared in scene units this cried PURE fog
        # at any End <= 1, and said nothing while a scene-unit Start
        # fogged nothing at all -- ask in the domain the fog reads.
        # Direct3D's z' IS the raster's NDC z remapped, (z + 1) / 2,
        # under a perspective and an orthographic camera alike
        zf = gbuf.zndc[cov].astype(np.float64)
        zf = zf[np.isfinite(zf)]
        if zf.size == 0:
            return None
        return _fog_coverage_note_z((zf + 1.0) * 0.5, settings)
    p = np.asarray(proj, np.float64)
    a, b = float(p[2, 2]), float(p[2, 3])
    if abs(a - 1.0) < 1e-12 or abs(a + 1.0) < 1e-12:
        return None                     # orthographic
    near, far = b / (a - 1.0), b / (a + 1.0)
    if not (0.0 < near < far):
        return None
    z = gbuf.zndc[cov].astype(np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    denom = (far + near) - z * (far - near)
    good = denom > 1e-9
    end = float(getattr(settings, 'fog_end', 0.0))
    if end <= 0.0:
        return None
    dist = np.where(good, 2.0 * far * near / np.maximum(denom, 1e-9),
                    np.inf)
    frac = float((dist >= end).mean())
    if frac < 0.9:
        return None
    d_near = float(dist[np.isfinite(dist)].min()) \
        if np.isfinite(dist).any() else float('inf')
    return ('[Halcyon] fog: '
            f'{100.0 * frac:.0f}% of the covered frame sits at or past '
            f'Fog End ({end:.4g}) and renders as PURE fog colour -- the '
            f'nearest surface is already {d_near:.4g} away. If the '
            'viewport looks right and this render does not, or the '
            'numbers above look several times too large, check the '
            'camera for scaled parents (now stripped automatically) or '
            'move Fog Start/End out to the scene\'s real distances')


def _fog_coverage_note_z(zd, settings):
    """`fog_coverage_note` under Fog Depth Z (F007): Direct3D's
    post-projection depth of every covered pixel (`zd`, 0..1)
    against Fog Start and End, which are 0..1 depths there. Two
    lines: nothing reaches Fog Start (a switched-on fog that fogs
    nothing -- what scene-unit Start/End do under Z), or the frame
    sits past Fog End. Returns the line or None."""
    lo, hi = float(zd.min()), float(zd.max())
    start = float(getattr(settings, 'fog_start', 0.0))
    end = float(getattr(settings, 'fog_end', 0.0))
    span = ('Fog Depth is Z, so Fog Start and End are post-projection '
            "depths in 0..1, and this frame's surfaces span "
            f'{lo:.4f}..{hi:.4f}')
    fix = ('put Fog Start/End inside that span, or set Fog Depth to W to '
           'give them in scene units')
    if start >= hi:
        return (f'[Halcyon] fog: NOTHING is fogged -- {span}: Fog Start '
                f'({start:.4g}) lies past every one of them; {fix}')
    frac = float((zd >= end).mean()) if end > 0.0 else 0.0
    if frac >= 0.9:
        return (f'[Halcyon] fog: {100.0 * frac:.0f}% of the covered frame '
                f'sits at or past Fog End ({end:.4g}) and renders as PURE '
                f'fog colour -- {span}; {fix}')
    return None


#: the volumetric marchers by lamp type. SUN and HEMI are deliberately
#: absent: an infinite directional beam has no apex to glow from, and
#: 2.79 gave neither a halo
_VOLUME_KINDS = ('SPOT', 'POINT', 'AREA')


def _volume_occlusion_wanted(scene, st):
    """True when any drawn beam asked to be shadowed (needs the BVH)."""
    if not getattr(st, 'spot_cones', False):
        return False
    return any(float(getattr(l, 'volumetric', 0.0)) > 0.0
               and getattr(l, 'volumetric_occlusion', False)
               for l in (getattr(scene, 'lights', None) or ())
               if str(getattr(l, 'type', '')).upper() in _VOLUME_KINDS)


def _halo_noise2(u, v, ti, seed):
    """One slice of lattice value noise: corners wang-hashed, smooth
    lerp. Pure function of (u, v, integer time slice, seed)."""
    from .patterns import _wang01p
    x0 = np.floor(u)
    y0 = np.floor(v)
    fx = u - x0
    fy = v - y0
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)

    def corner(dx, dy):
        hh = ((x0 + dx).astype(np.int64) * 73856093
              ^ (y0 + dy).astype(np.int64) * 19349663
              ^ np.int64(ti) * 83492791
              ^ np.int64(seed) * 15485863)
        return _wang01p((hh & 0xFFFFFFFF).astype(np.uint32))

    n00, n10 = corner(0, 0), corner(1, 0)
    n01, n11 = corner(0, 1), corner(1, 1)
    return ((n00 * (1 - fx) + n10 * fx) * (1 - fy)
            + (n01 * (1 - fx) + n11 * fx) * fy)


def _halo_fbm(u, v, t, seed):
    """Two-octave animated value noise: two integer time slices
    cross-faded, so it writhes continuously and deterministically."""
    tf = float(np.floor(t))
    tt = float(t) - tf

    def one(ti):
        return (_halo_noise2(u, v, ti, seed)
                + 0.5 * _halo_noise2(u * 2.13 + 7.7, v * 2.13 + 3.1,
                                     ti, seed + 11)) / 1.5

    return one(int(tf)) * (1.0 - tt) + one(int(tf) + 1) * tt


def _halo_hsv_mat(dh, s_scale, v_scale):
    """A 3x3 colour matrix: hue rotation about the grey axis, then
    saturation (lerp to luma) and value scaling -- the array-sized
    twin of the per-colour colorsys jitter."""
    ang = dh * 2.0 * np.pi
    c, s = np.cos(ang), np.sin(ang)
    ax = 1.0 / np.sqrt(3.0)
    K = np.array([[0, -ax, ax], [ax, 0, -ax], [-ax, ax, 0]], np.float64)
    rot = (np.eye(3) * c + s * K
           + (1.0 - c) * np.full((3, 3), 1.0 / 3.0))
    luma = np.array([[0.299, 0.587, 0.114]] * 3, np.float64)
    sat = luma + (np.eye(3) - luma) * s_scale
    return (rot @ sat * v_scale).astype(np.float32)


def _halo_image_tex(spec):
    """The halo's image as a Texture, cached on the image buffer."""
    img = spec.get('image')
    if img is None:
        return None
    tex = getattr(img, '_halo_tex', None)
    if tex is None:
        from .texture import Texture
        px = getattr(img, 'pixels', img)
        tex = Texture(np.asarray(px, np.float32), name='halo',
                      colorspace='Non-Color', wrap='EXTEND',
                      filt='BILINEAR')
        try:
            img._halo_tex = tex
        except Exception:                                       # noqa: BLE001
            pass
    return tex


def _halo_hash01(idx, salt):
    """Per-halo deterministic random in [0,1): Wang mix of the seed.

    The same hash family the caustics use, so a halo's flicker, pulse
    phase and colour jitter are pure functions of (seed, salt) --
    identical across runs, devices and batch orders."""
    from .patterns import _wang01p
    u = (np.asarray(idx, np.int64) & 0xFFFFFFFF).astype(np.uint32)
    return _wang01p(u ^ np.uint32(salt))


def _halo_speed(spec, key):
    """An animation-speed dial read the honest way.

    R200: every animated halo effect gained its own Speed multiplier
    on the master Animation Speed. A missing key means 1.0 (old spec
    dicts keep their exact pixels: x * 1.0 is exact in IEEE), and a
    key that is PRESENT keeps its value even at 0.0 -- speed zero
    means FROZEN, which the old `or 1.0` idiom silently turned back
    into full speed."""
    v = spec.get(key)
    return 1.0 if v is None else float(v)


def _halo_falloff(kind, d, inner):
    """R253: the Falloff menu's curves, brightness in [0,1] from the
    squared shape metric d in [0,1] (1 at the rim). 'BI' never reaches
    here (the hardness ladder stays verbatim in _draw_halos).

    GAUSSIAN is Reeves' additive puff normalised to reach zero at the
    rim; LINEAR and QUADRATIC the point-sprite era's fades; HARD the
    flat sprite dot; RING_ONLY dark inside `inner`, then a linear fade
    out -- a hollow shockwave."""
    d = np.minimum(np.asarray(d, np.float32), 1.0)
    r = np.sqrt(d)
    if kind == 'GAUSSIAN':
        k = float(np.exp(-4.5))
        out = (np.exp(-4.5 * d) - k) / (1.0 - k)
    elif kind == 'LINEAR':
        out = 1.0 - r
    elif kind == 'QUADRATIC':
        out = (1.0 - r) * (1.0 - r)
    elif kind == 'HARD':
        out = np.where(d < 1.0, 1.0, 0.0)
    elif kind == 'RING_ONLY':
        inner = min(max(float(inner), 0.0), 0.99)
        out = np.where(r >= inner,
                       1.0 - (r - inner) / max(1.0 - inner, 1e-6), 0.0)
    else:
        out = np.where(d < 1.0, 1.0 - d, 0.0)
    # exactly zero at and past the rim, whatever float32 made of the
    # curve's last step (the window corners past the circle keep the
    # caller's `inside` mask besides)
    out = np.where(d < 1.0, out, 0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _halo_stretch(fx, fy, stretch, ang):
    """R253: the anamorphic streak -- shrink the halo's frame along a
    screen angle so the glow reads `stretch` times longer along it.
    Independent of Rotation/Spin (applied after them), so a turning
    star keeps a fixed horizontal smear."""
    c, s = float(np.cos(ang)), float(np.sin(ang))
    u = c * fx + s * fy
    v = -s * fx + c * fy
    u = u / np.float32(stretch)
    return c * u - s * v, s * u + c * v


def _halo_blend(sub_rgb, sub_a, crgb, ca, lv, mode, addfac):
    """R253: one glow's composite over the frame window, in place.

    ADD_SLIDER is addalphaAddfacFloat verbatim -- the destination
    weight slides with Add from alpha-over (0) to pure addition (1);
    ALPHA and ADDITIVE are its two ends (the same two statements, so
    Add 0 / Add 1 match them bit for bit); SCREEN is the separable
    1 - (1 - a)(1 - b) of the PDF blend modes, which never exceeds
    white (the colour clamps to [0,1] here and only here)."""
    if mode == 'SCREEN':
        c = np.clip(crgb, 0.0, 1.0)
        sub_rgb[lv] = sub_rgb[lv] + c[lv] - sub_rgb[lv] * c[lv]
        sub_a[lv] = sub_a[lv] + ca[lv] - sub_a[lv] * ca[lv]
        return
    if mode == 'ALPHA':
        addfac = 0.0
    elif mode == 'ADDITIVE':
        addfac = 1.0
    mfac = (1.0 - ca * (1.0 - addfac)).astype(np.float32)
    sub_rgb[lv] = (mfac[lv, None] * sub_rgb[lv] + crgb[lv])
    sub_a[lv] = mfac[lv] * sub_a[lv] + ca[lv]


def _draw_halos(img, scene, st, gbuf, view, proj, w, h, sel_mask=None):
    """BI's halo materials: every vertex a depth-tested billboard glow.

    The transcription of the 2.79 pipeline, function for function:
    make_render_halos (a halo per vertex per halo-material slot, seed
    running from the material's own), project_renderdata (the radius
    measured by displacing the VIEW-space position one hasize along x
    and reprojecting; the depth-soften range zd by displacing it one
    hasize along -z), verghalo (painter's order, far to near), and
    shadeHaloFloat itself -- hardness ladder, rings and lines off
    hashvectf at the material seed, the star-points pinch, Extreme
    Alpha's squared alpha, the soft/plain depth softening -- blended
    by addalphaAddfacFloat, where Add slides the destination weight
    from alpha-over (0) to pure addition (1).

    Differences owned by name: depth runs in float32 NDC rather than
    the C's 24-bit ints (same formulas, more digits); the C read
    hashvectf[ofs%768 + 1] one float past the table for some seeds --
    undefined memory in the C, a deterministic wrap here; mist and
    halo textures are not applied (the engine shades no surface mist
    either); Flare on a halo material draws no extra flare (the
    per-LAMP lens flare kit covers flares); `puno` scales by the
    world-frame facing term (the C mixed an object-space normal into
    a view-space dot -- frame soup this transcription declines to
    reproduce). Shaded halos take the per-halo constant lamp sum of
    render_lighting_halo -- constant per halo in the C too, since
    every lamp term reads only the halo's own centre.
    """
    halos = getattr(scene, 'halos', None)
    if not halos:
        return img
    from .bitex_tables import HASHVECTF
    hv = HASHVECTF
    hvn = hv.shape[0]
    mats = scene.materials or ()
    vm = np.asarray(view, np.float32)
    pm = np.asarray(proj, np.float32)
    persp = abs(float(pm[3, 2])) > 0.5

    def _zdist(zn):
        """haloZtoDist: NDC z back to camera-plane distance."""
        if persp:
            den = zn + float(pm[2, 2])
            den = np.where(np.abs(den) < 1e-12, 1e-12, den)
            return float(pm[2, 3]) / den
        return (float(pm[2, 3]) - zn) / min(float(pm[2, 2]), -1e-12)

    # ---- gather every halo point with its projection ----------------
    drawn = []
    eye_w = np.linalg.inv(vm)[:3, 3]
    for grp in halos:
        mi = int(grp.get('mat', -1))
        m = mats[mi] if 0 <= mi < len(mats) else None
        spec = getattr(m, 'halo', None) if m is not None else None
        if not spec:
            continue
        pos = np.asarray(grp.get('pos'), np.float32)
        if pos is None or pos.size == 0:
            continue
        pos = pos.reshape(-1, 3)
        n = pos.shape[0]
        base_size = float(spec.get('size', 0.5))
        sizes = grp.get('sizes')
        sizes = (np.asarray(sizes, np.float32).reshape(-1)
                 if sizes is not None else np.full(n, base_size, np.float32))
        seeds = grp.get('seeds')
        seeds = (np.asarray(seeds, np.int64).reshape(-1)
                 if seeds is not None
                 else int(spec.get('seed', 0)) + np.arange(n, dtype=np.int64))
        if bool(spec.get('animate_seed')):
            # R253: the seed walks with the frame -- rings, lines,
            # jitter and pulse phases all re-roll per frame
            seeds = (seeds + int(getattr(scene, 'frame', 0))) % 256
        if bool(spec.get('puno')) and grp.get('normals') is not None:
            # MA_HALOPUNO: only rear-facing verts glow, scaled by the
            # facing cosine to the fourth
            nor = np.asarray(grp['normals'], np.float32).reshape(-1, 3)
            vdir = pos - eye_w[None, :]
            vdir /= np.maximum(np.linalg.norm(vdir, axis=1), 1e-9)[:, None]
            zn_f = (nor * vdir).sum(axis=1)
            sizes = np.where(zn_f >= 0.0, 0.0,
                             sizes * zn_f * zn_f * zn_f * zn_f)
        pulse = float(spec.get('pulse', 0.0) or 0.0)
        if pulse > 0.0:
            # R194: each halo breathes on its own phase (hashed off its
            # seed), so a cloud shimmers instead of throbbing in sync.
            # R200: Pulse Speed scales the master clock for this effect
            # alone (x1 is bit-for-bit the single-clock road), and a
            # speed of 0 now really FREEZES instead of snapping to 1
            spd = _halo_speed(spec, 'anim_speed') \
                * _halo_speed(spec, 'pulse_speed')
            t_s = float(getattr(scene, 'time', 0.0))
            ph = _halo_hash01(seeds, 0x51ED2701)
            sizes = (sizes * (1.0 + 0.5 * pulse * np.sin(
                2.0 * np.pi * (t_s * spd + ph)))).astype(np.float32)
        vco = pos @ vm[:3, :3].T + vm[:3, 3]
        hoco = vco @ pm[:3, :3].T + pm[:3, 3]
        ww = vco @ pm[3, :3].T + pm[3, 3]
        ok = (ww > 1e-6) & (sizes != 0.0)
        if not ok.any():
            continue
        xs = 0.5 * w * (1.0 + hoco[:, 0] / np.where(ok, ww, 1.0))
        ys = 0.5 * h * (1.0 + hoco[:, 1] / np.where(ok, ww, 1.0))
        zs = hoco[:, 2] / np.where(ok, ww, 1.0)
        # "we clip halos less critical": centre within twice the frame
        ok &= (np.abs(hoco[:, 0] / np.where(ok, ww, 1.0)) < 2.0) & \
              (np.abs(hoco[:, 1] / np.where(ok, ww, 1.0)) < 2.0)
        # radius: displace one hasize along view x, reproject
        v2 = vco.copy()
        v2[:, 0] += sizes
        h2 = v2 @ pm[:3, :3].T + pm[:3, 3]
        w2 = v2 @ pm[3, :3].T + pm[3, 3]
        rad = np.abs(xs - 0.5 * w * (1.0 + h2[:, 0] /
                                     np.where(np.abs(w2) > 1e-6, w2, 1.0)))
        # depth range: displace one hasize away, reproject
        v3 = vco.copy()
        v3[:, 2] -= sizes
        h3 = v3 @ pm[:3, :3].T + pm[:3, 3]
        w3 = v3 @ pm[3, :3].T + pm[3, 3]
        zd = np.abs(zs - h3[:, 2] / np.where(np.abs(w3) > 1e-6, w3, 1.0))
        ok &= rad > 1e-6
        # R253: the camera distance the Fade Near/Far sockets read --
        # view-space depth forward of the lens (the radial distance
        # under an orthographic camera, whose z is the sort key only)
        dcam = -vco[:, 2] if persp else np.linalg.norm(vco, axis=1)
        idx = np.nonzero(ok)[0]
        for i in idx:
            drawn.append((float(zs[i]), mi, float(xs[i]), float(ys[i]),
                          float(rad[i]), float(zd[i]), int(seeds[i]),
                          float(sizes[i]), np.asarray(pos[i]),
                          float(dcam[i])))
    if not drawn:
        return img
    # verghalo: descending zs -- far halos first, near composite over
    drawn.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))

    zbuf = gbuf.zndc
    out = img
    shaded_cache = {}
    # R197 field find: the line/ring windows were RAW PIXELS -- 2.79's
    # own convention -- so a 1080p render drew proportionally thinner
    # hairlines than a 480p one, and a supersampled frame thinner
    # still. The windows now scale with frame height against the era's
    # own 480-line reference: the same scene keeps the same look at
    # any resolution and under any supersample, and a 480-line render
    # is bit-for-bit the old picture. Floored so tiny previews keep
    # their lines
    wscale = max(float(h) / 480.0, 0.5)
    _gauss_k = float(np.exp(-4.5))
    for zs, mi, xs, ys, rad, zd, seed, hasize, wpos, dcam in drawn:
        spec = mats[mi].halo
        alpha0 = float(np.clip(spec.get('alpha', 1.0), 0.0, 1.0))
        flick = float(spec.get('flicker', 0.0) or 0.0)
        if flick > 0.0:
            # R194: per-frame per-halo brightness jitter -- the 90s
            # sparkle. Deterministic in (seed, frame). R200: Flicker
            # Speed rescales the frame counter (x1 is floor(frame),
            # the old integer bit for bit; 0.5 re-rolls every other
            # frame, 2 rolls twice as fast on fractional frames)
            fs_f = _halo_speed(spec, 'flicker_speed')
            fr_i = int(np.floor(
                float(getattr(scene, 'frame', 0)) * fs_f))
            h01 = float(_halo_hash01(np.array([seed * 1013 + fr_i],
                                              np.int64), 0xA511E9B3)[0])
            alpha0 *= max(1.0 - flick * h01, 0.0)
        # R253: the camera-distance fade (Far above Near turns it on;
        # the 0/0 default is a no-op by construction)
        fn_ = float(spec.get('fade_near', 0.0) or 0.0)
        ff_ = float(spec.get('fade_far', 0.0) or 0.0)
        if ff_ > fn_:
            alpha0 *= float(np.clip((ff_ - dcam) / (ff_ - fn_), 0.0, 1.0))
        if alpha0 == 0.0:
            continue
        # R197: the halo's own frame -- static Rotation plus Spin share
        # one turn, and Aspect squashes the rotated x (a turned ellipse
        # when both are set). Coverage, rings, gradient and every shape
        # metric live in these coordinates; at the defaults they ARE
        # the plain pixel offsets, bit for bit
        rot_b = float(spec.get('rotation', 0.0) or 0.0)
        spin = float(spec.get('spin', 0.0) or 0.0)
        aspect = min(max(float(spec.get('aspect', 1.0) or 1.0), 0.05),
                     20.0)
        # R253: the anamorphic Stretch (a screen-angle streak that is
        # independent of Rotation/Spin) and the outer Glow both need a
        # wider pixel window than the core; the window and the halo's
        # own frame are closures so the core at the defaults computes
        # EXACTLY the pre-R253 arrays (same operations, same order)
        stretch = min(max(float(spec.get('stretch', 1.0) or 1.0), 0.05),
                      20.0)
        over = str(spec.get('depth_mode', 'ZBUFFER') or 'ZBUFFER') == 'OVER'
        ext = rad * max(aspect, 1.0)
        if stretch != 1.0:
            ext = ext * max(stretch, 1.0)
        if spin != 0.0 or rot_b != 0.0:
            ang_s = rot_b
            if spin != 0.0:
                t_s = float(getattr(scene, 'time', 0.0))
                ph_s = float(_halo_hash01(np.array([seed], np.int64),
                                          0x7F4A7C15)[0])
                ang_s = ang_s + spin * t_s + ph_s * 2.0 * np.pi
            ca_s, sa_s = float(np.cos(ang_s)), float(np.sin(ang_s))
        else:
            ca_s = sa_s = None

        def _window(extent):
            wx0 = max(int(np.floor(xs - extent)), 0)
            wx1 = min(int(np.ceil(xs + extent)) + 1, w)
            wy0 = max(int(np.floor(ys - extent)), 0)
            wy1 = min(int(np.ceil(ys + extent)) + 1, h)
            if wx0 >= wx1 or wy0 >= wy1:
                return None
            wxn = np.arange(wx0, wx1, dtype=np.float32)[None, :] \
                - np.float32(xs)
            wyn = np.arange(wy0, wy1, dtype=np.float32)[:, None] \
                - np.float32(ys)
            return wx0, wx1, wy0, wy1, wxn, wyn

        def _frame(wxn, wyn):
            if ca_s is not None:
                fx = ca_s * wxn + sa_s * wyn
                fy = -sa_s * wxn + ca_s * wyn
            else:
                fx, fy = wxn, wyn
            if stretch != 1.0:
                fx, fy = _halo_stretch(fx, fy, stretch, float(
                    spec.get('stretch_angle', 0.0) or 0.0))
            fx = fx / np.float32(aspect) if aspect != 1.0 else fx
            return fx, fy

        win = _window(ext)
        if win is None:
            continue
        x0, x1, y0, y1, xn, yn = win
        xa, ya = _frame(xn, yn)
        distsq = xa * xa + ya * ya
        radsq = rad * rad
        shape = str(spec.get('shape', 'DISC') or 'DISC')
        img_rgb = None
        img_a = None
        if shape == 'IMAGE':
            tex_i = _halo_image_tex(spec)
            if tex_i is None:
                shape = 'DISC'
            else:
                # R198: the image IS the halo -- its alpha the shaped
                # intensity, its colours the glow, exactly the role
                # 2.79's HaloTex played. Coverage is the image square
                # the FULL grid: xa/ya are broadcast axes (1,W)/(H,1)
                # and sampling them raw collapses one dimension into
                # a stripe
                uu, vv = np.broadcast_arrays(
                    (xa / max(rad, 1e-12)) * 0.5 + 0.5,
                    (ya / max(rad, 1e-12)) * 0.5 + 0.5)
                flat = tex_i.sample(
                    np.clip(uu, 0.0, 1.0).ravel().astype(np.float32),
                    np.clip(vv, 0.0, 1.0).ravel().astype(np.float32),
                    filt='BILINEAR', wrap='EXTEND')
                rgba_i = np.asarray(flat, np.float32).reshape(
                    uu.shape + (flat.shape[-1],))
                out_i = (uu < 0.0) | (uu > 1.0) | (vv < 0.0) | (vv > 1.0)
                a_ch = rgba_i[..., 3] if rgba_i.shape[-1] > 3 \
                    else np.ones(uu.shape, np.float32)
                img_a = np.where(out_i, 0.0, a_ch).astype(np.float32)
                img_rgb = rgba_i[..., :3]
        if shape == 'IMAGE':
            inside = np.maximum(np.abs(xa), np.abs(ya)) < rad
        else:
            inside = distsq < radsq
        if sel_mask is not None:
            inside &= sel_mask[y0:y1, x0:x1]
        if not inside.any():
            continue
        zz = zbuf[y0:y1, x0:x1]
        soft = bool(spec.get('soft'))
        if not soft and not over:
            inside &= zz > zs
            if not inside.any():
                continue
        alpha = np.full(distsq.shape, alpha0, np.float32)
        if soft:
            # MA_HALO_SOFT: how much of the halo's depth is in front
            seg = hasize * np.sqrt(np.maximum(
                1.0 - distsq / max(radsq, 1e-12), 0.0))
            depth2 = 2.0 * seg
            dfz = _zdist(zz) - _zdist(np.float32(zs))
            soften = np.where(dfz < seg,
                              (seg + dfz) / np.maximum(depth2, 1e-12),
                              1.0)
            alpha *= np.clip(soften, 0.0, 1.0)
            inside &= depth2 > 1e-12
            inside &= alpha > 0.0
        elif not over:
            # the old softening: geometry within zd behind the centre
            # (R253: Over Everything skips it with the z-test)
            zde = max(zd, 1e-12)
            t_soft = (zz - zs) / zde
            near_geo = zs > (zz - zd)
            alpha = np.where(near_geo,
                             alpha * np.sqrt(np.sqrt(
                                 np.maximum(t_soft, 0.0))),
                             alpha)
        if not inside.any():
            continue
        radist = np.sqrt(distsq)
        ringf = np.zeros_like(distsq)
        ringc = int(spec.get('rings', 0) or 0)
        rw_ = max(float(spec.get('ring_width', 1.0) or 1.0), 1e-3) * wscale
        if ringc:
            even_r = bool(spec.get('rings_even'))
            ofs = seed
            for k_r in range(ringc):
                if even_r:
                    # R198: evenly spaced crisp circles -- shockwaves
                    fac = np.abs(1.0 * (rad * ((k_r + 1.0)
                                               / (ringc + 1.0))
                                        - radist)) / rw_
                else:
                    r0 = hv[ofs % hvn]
                    r1 = hv[(ofs % hvn + 1) % hvn]
                    fac = np.abs(r1 * (rad * abs(r0) - radist)) / rw_
                ringf += np.where(fac < 1.0, 1.0 - fac, 0.0)
                ofs += 2
        # clamped into the ladder's domain: every value at or past 1
        # lands on zero brightness either way (1 - min(dist,1) = 0 and
        # the rungs are monotone on [0,1]), and the aspect-extended
        # rect corners otherwise push sin past pi into sqrt(negative)
        dist = np.minimum(distsq / max(radsq, 1e-12), 1.0)
        if shape == 'RING':
            # HA_FLARECIRC, verbatim: the centre pushed to the rim
            dist = 0.5 + np.abs(dist - 0.5)
        elif shape == 'HEX':
            hxm = np.maximum(np.abs(ya),
                             np.abs(xa) * 0.8660254
                             + np.abs(ya) * 0.5) / max(rad, 1e-12)
            # clamped into the ladder's own [0,1] domain -- the disc
            # metric never left it, these metrics can at the corners
            dist = np.minimum(hxm * hxm, 1.0)
        elif shape == 'DIAMOND':
            dmm = (np.abs(xa) + np.abs(ya)) / max(rad, 1e-12)
            dist = np.minimum(dmm * dmm, 1.0)
        elif shape in ('TRIANGLE', 'PENTAGON', 'OCTAGON'):
            # the regular-polygon radial metric: vertices on the halo
            # circle, edges pulled in by cos of the half-segment
            nsides = {'TRIANGLE': 3, 'PENTAGON': 5, 'OCTAGON': 8}[shape]
            seg2 = 2.0 * np.pi / nsides
            th = np.arctan2(ya, xa)
            rr = np.cos(np.mod(th, seg2) - seg2 * 0.5) \
                / np.cos(seg2 * 0.5)
            pmm = radist * rr / max(rad, 1e-12)
            dist = np.minimum(pmm * pmm, 1.0)
        elif shape == 'CROSS':
            # a plus-sign glow: bright along both axes, tapering to
            # nothing at the halo edge
            cmm = (np.minimum(np.abs(xa), np.abs(ya)) * 2.2
                   + np.maximum(np.abs(xa), np.abs(ya))) \
                / max(rad, 1e-12)
            dist = np.minimum(cmm * cmm, 1.0)
        elif shape == 'SQUARE':
            sqm = np.maximum(np.abs(xa), np.abs(ya)) / max(rad, 1e-12)
            dist = np.minimum(sqm * sqm, 1.0)
        elif shape == 'STAR':
            # a solid five-point star: the boundary radius runs from
            # the outer points to an inner waist, linear in angle
            seg5 = 2.0 * np.pi / 5.0
            thf = np.abs(np.mod(np.arctan2(ya, xa) + np.pi * 0.5,
                                seg5) - seg5 * 0.5) / (seg5 * 0.5)
            bound = 0.38 + (1.0 - 0.38) * (1.0 - thf)
            stm = radist / (max(rad, 1e-12) * bound)
            dist = np.minimum(stm * stm, 1.0)
        elif shape == 'HEART':
            # the classic implicit heart, radial falloff inside it
            hx = xa / (max(rad, 1e-12) * 0.82)
            hy = ya / (max(rad, 1e-12) * 0.82) + 0.32
            f_h = ((hx * hx + hy * hy - 1.0) ** 3
                   - hx * hx * hy * hy * hy)
            dist = np.where(f_h < 0.0,
                            np.minimum((radist / max(rad, 1e-12)) ** 2
                                       * 0.8, 1.0), 1.0)
        # R253: the Falloff menu -- 'BI' is the ladder verbatim; the
        # other curves read the same shape metric
        fo_ = str(spec.get('falloff', 'BI') or 'BI')
        if fo_ == 'BI':
            _hd = spec.get('hardness', 50)
            hard = int(_hd) if _hd is not None else 50
            if hard >= 30:
                dist = np.sqrt(dist)
                if hard >= 40:
                    dist = np.sin(dist * (np.pi * 0.5))
                    if hard >= 50:
                        dist = np.sqrt(dist)
            elif hard < 20:
                dist = dist * dist
            dist = np.where(dist < 1.0, 1.0 - dist, 0.0)
        else:
            dist = _halo_falloff(fo_, dist,
                                 float(spec.get('ring_inner', 0.6)))
        if img_a is not None:
            # image authority: its alpha replaces the shaped falloff
            # (the hardness ladder is the disc's law, not the image's)
            dist = img_a
        noise_amt = float(spec.get('noise', 0.0) or 0.0)
        if noise_amt > 0.0:
            # R198: the energy-blast modulator -- animated value noise
            # carving and boosting the shaped core; composes with
            # every shape and the image
            nsc = max(float(spec.get('noise_scale', 4.0) or 4.0), 0.2)
            spd_n = _halo_speed(spec, 'anim_speed') \
                * _halo_speed(spec, 'noise_speed')
            t_n = float(getattr(scene, 'time', 0.0)) * spd_n
            nse = _halo_fbm(xa / max(rad, 1e-12) * nsc + 17.31,
                            ya / max(rad, 1e-12) * nsc + 9.77,
                            t_n, seed)
            dist = np.clip(dist * (1.0 - noise_amt
                                   + noise_amt * (0.35 + 1.3 * nse)),
                           0.0, 1.3).astype(np.float32)
        linef = np.zeros_like(distsq)
        linec = int(spec.get('lines', 0) or 0)
        lw_ = max(float(spec.get('line_width', 1.0) or 1.0), 1e-3) * wscale
        if linec:
            ofs = seed
            for _ in range(linec):
                r0 = hv[ofs % hvn]
                r1 = hv[(ofs % hvn + 1) % hvn]
                fac = np.abs(xa * r0 + ya * r1) / lw_
                linef += np.where(fac < 1.0, 1.0 - fac, 0.0)
                ofs += 3
            linef = linef * dist
        starpoints = int(spec.get('star_points', 0) or 0)
        if starpoints:
            angle = np.arctan2(ya, xa) * (1.0 + 0.25 * starpoints)
            co = np.cos(angle)
            si = np.sin(angle)
            ang2 = (co * xa + si * ya) * (co * ya - si * xa)
            ster = np.abs(ang2)
            big = ster > 1.0
            ster2 = np.where(big, rad / np.maximum(ster, 1e-12), 1.0)
            dist = np.where(big & (ster2 < 1.0),
                            dist * np.sqrt(np.maximum(ster2, 0.0)), dist)
        raysc = int(spec.get('rays', 0) or 0)
        rayf = None
        if raysc:
            # R198: EVEN rays -- the symmetric starburst the hashed
            # Lines can't make; sharper exponent, thinner spikes.
            # R200: their own accumulator, so Ray Colour can differ
            # from Line Colour (unset, it follows the lines -- old
            # scenes keep their look)
            sharp = max(float(spec.get('ray_sharp', 8.0) or 8.0), 0.5)
            th_r = np.arctan2(ya, xa)
            rayf = (np.power(np.abs(np.cos(th_r * raysc * 0.5)), sharp)
                    * np.maximum(1.0 - radist / max(rad, 1e-12), 0.0))
        boltc = int(spec.get('bolts', 0) or 0)
        boltf = None
        if boltc:
            # R198: electric arcs -- each bolt is an angular path that
            # wiggles with radius and RE-STRIKES eight times per
            # animation second (the angle rehashes), pure function of
            # (seed, bolt, time). R200: their own accumulator and
            # colour (following the lines while unset), and Bolt
            # Speed scales strike rate and writhe together
            bw_ = max(float(spec.get('bolt_width', 1.0) or 1.0),
                      0.05) * wscale
            spd_b = _halo_speed(spec, 'anim_speed') \
                * _halo_speed(spec, 'bolt_speed')
            t_b = float(getattr(scene, 'time', 0.0)) * spd_b
            fr_b = int(np.floor(t_b * 8.0))
            th_p = np.arctan2(ya, xa)
            rr_n = radist / max(rad, 1e-12)
            taper_b = np.maximum(1.0 - rr_n, 0.0)
            boltf = np.zeros_like(distsq)
            for b in range(boltc):
                a0 = float(_halo_hash01(
                    np.array([seed * 131 + b * 7919 + fr_b * 104729],
                             np.int64), 0x8DA6B343)[0]) * 2.0 * np.pi
                wig = (_halo_fbm(rr_n * 6.0 + b * 3.7,
                                 np.full_like(rr_n, b * 1.3),
                                 t_b * 2.0, seed + b) - 0.5) * 0.9
                dth = np.mod(th_p - a0 - wig + np.pi,
                             2.0 * np.pi) - np.pi
                arc = np.maximum(1.0 - np.abs(dth) * radist
                                 / (bw_ * 2.0), 0.0)
                boltf = boltf + arc * taper_b
        # rays and bolts glow where the core may not (a ring's hollow
        # centre); classic lines carry dist and change nothing here.
        # The summed field is the coverage the single-accumulator
        # road used -- the live mask and the alpha add read the SUM,
        # only the colour adds split per element
        totf = linef
        if rayf is not None:
            totf = totf + rayf
        if boltf is not None:
            totf = totf + boltf
        live = inside & ((dist > 0.00001) | (totf > 0.00001))
        if not live.any():
            continue
        dist = dist * alpha
        ringf = ringf * dist
        linef = linef * alpha
        if rayf is not None:
            rayf = rayf * alpha
        if boltf is not None:
            boltf = boltf * alpha
        totf = totf * alpha
        col_r = np.asarray(spec.get('color', (0.8, 0.8, 0.8)), np.float32)
        col2_r = np.asarray(spec.get('color2', (0.0, 0.0, 0.0)),
                            np.float32)
        rh_ = float(spec.get('rand_hue', 0.0) or 0.0)
        rs2 = float(spec.get('rand_sat', 0.0) or 0.0)
        rv_ = float(spec.get('rand_val', 0.0) or 0.0)
        # R200: the master Hue/Sat/Val Shift dials -- one deliberate,
        # keyframable re-tint of EVERY coloured option (body, gradient
        # end, ramp, image, rings, lines, rays, bolts). The per-halo
        # random jitter (R194) folds into the same transform, and now
        # scatters every coloured option too, so a confetti halo's
        # trim follows its body. At the defaults (0 / 1 / 1, no
        # jitter) no colour is touched -- bit for bit the old road
        hs_ = float(spec.get('hue_shift', 0.0) or 0.0)
        _ss = spec.get('sat_shift')
        ss_m = 1.0 if _ss is None else max(float(_ss), 0.0)
        _vs = spec.get('val_shift')
        vs_m = 1.0 if _vs is None else max(float(_vs), 0.0)
        dh_a = hs_
        s_a = ss_m
        v_a = vs_m
        if rh_ > 0.0 or rs2 > 0.0 or rv_ > 0.0:
            j1 = float(_halo_hash01(np.array([seed], np.int64),
                                    0x2545F491)[0])
            j2 = float(_halo_hash01(np.array([seed], np.int64),
                                    0x9E3779B9)[0])
            j3 = float(_halo_hash01(np.array([seed], np.int64),
                                    0x85EBCA6B)[0])
            dh_a = dh_a + (j1 - 0.5) * rh_
            s_a = s_a * (1.0 + (j2 - 0.5) * 2.0 * rs2)
            v_a = v_a * max(1.0 + (j3 - 0.5) * 2.0 * rv_, 0.0)
        adj_on = dh_a != 0.0 or s_a != 1.0 or v_a != 1.0

        def _adj(c):
            # scalar colours ride colorsys; images and ramp LUTs take
            # the matrix twin (_halo_hsv_mat) with the SAME dh/s/v
            if not adj_on:
                return c
            import colorsys as _cs
            hh, ss, vv = _cs.rgb_to_hsv(float(c[0]), float(c[1]),
                                        float(c[2]))
            hh = (hh + dh_a) % 1.0
            ss = min(max(ss * s_a, 0.0), 1.0)
            vv = max(vv * v_a, 0.0)
            return np.asarray(_cs.hsv_to_rgb(hh, ss, vv), np.float32)
        col_r = _adj(col_r)
        col2_r = _adj(col2_r)
        lit_v = None
        xalpha = bool(spec.get('xalpha'))
        ca = dist * dist if xalpha else dist
        ramp = spec.get('ramp')
        gtype = str(spec.get('gradient_type', 'RADIAL') or 'RADIAL')

        def _grad_t():
            # R199: the gradient sweep family. Coordinates are the
            # halo's OWN (rotated, aspected) frame, so Rotation turns
            # a linear gradient with the halo. Adjustable noise
            # wobbles the coordinate -- turbulent bands; ANGULAR
            # wraps its wobble so the wheel stays seamless
            if gtype == 'ANGULAR':
                t_g = (np.arctan2(ya, xa) / (2.0 * np.pi) + 0.5)
            elif gtype == 'HORIZONTAL':
                t_g = np.clip((xa / max(rad, 1e-12)) * 0.5 + 0.5,
                              0.0, 1.0)
            elif gtype == 'VERTICAL':
                t_g = np.clip((ya / max(rad, 1e-12)) * 0.5 + 0.5,
                              0.0, 1.0)
            elif gtype == 'DIAGONAL':
                t_g = np.clip(((xa + ya)
                               / (max(rad, 1e-12) * 1.4142135))
                              * 0.5 + 0.5, 0.0, 1.0)
            else:                              # RADIAL: centre out
                t_g = np.clip(radist / max(rad, 1e-12), 0.0, 1.0)
            gn = float(spec.get('gradient_noise', 0.0) or 0.0)
            if gn > 0.0:
                nsc_g = max(float(spec.get('noise_scale', 4.0)
                                  or 4.0), 0.2)
                spd_g = _halo_speed(spec, 'anim_speed') \
                    * _halo_speed(spec, 'grad_noise_speed')
                t_t = float(getattr(scene, 'time', 0.0)) * spd_g
                wob = _halo_fbm(xa / max(rad, 1e-12) * nsc_g + 31.7,
                                ya / max(rad, 1e-12) * nsc_g + 5.9,
                                t_t, seed + 77)
                if gtype == 'ANGULAR':
                    t_g = np.mod(t_g + (wob - 0.5) * gn, 1.0)
                else:
                    t_g = np.clip(t_g + (wob - 0.5) * gn, 0.0, 1.0)
            return t_g.astype(np.float32)

        if img_rgb is not None:
            # the image's own colours, shifted and jittered by the
            # matrix twin of the per-colour HSV transform
            rgbv = img_rgb
            if adj_on:
                m3 = _halo_hsv_mat(dh_a, s_a, v_a)
                rgbv = np.clip(rgbv @ m3.T, 0.0, None)
            crgb = dist[:, :, None] * rgbv
        elif ramp is not None and bool(spec.get('gradient')):
            # R198: the colour ramp -- RADIAL rides the radius,
            # ANGULAR sweeps the full turn (conic)
            lut = np.asarray(ramp, np.float32).reshape(-1, 4)[:, :3]
            if adj_on:
                m3 = _halo_hsv_mat(dh_a, s_a, v_a)
                lut = np.clip(lut @ m3.T, 0.0, None)
            tt1 = _grad_t()
            fpos = np.clip(tt1, 0.0, 1.0) * (len(lut) - 1)
            i0 = np.floor(fpos).astype(np.int32)
            i1 = np.minimum(i0 + 1, len(lut) - 1)
            fr2 = (fpos - i0)[:, :, None]
            crgb = dist[:, :, None] * (lut[i0] * (1.0 - fr2)
                                       + lut[i1] * fr2)
        elif bool(spec.get('gradient')):
            # R194: centre colour to edge colour, through whichever
            # sweep the gradient type asks (R199)
            tt = _grad_t()[:, :, None]
            crgb = dist[:, :, None] * (col_r[None, None, :] * (1.0 - tt)
                                       + col2_r[None, None, :] * tt)
        else:
            crgb = dist[:, :, None] * col_r[None, None, :]
        if bool(spec.get('shaded')) and (scene.lights or ()):
            # render_lighting_halo: every term reads the halo centre
            # only, so the lamp sum is one number per halo (vn is the
            # zero vector for a plain halo: inp = 1 - |0| = 1)
            key = tuple(np.round(wpos, 5))
            lit = shaded_cache.get(key)
            if lit is None:
                p1 = wpos.reshape(1, 3).astype(np.float32)
                acc = np.zeros(3, np.float32)
                for lgt in scene.lights:
                    if getattr(lgt, 'ambient_only', False):
                        continue
                    try:
                        _L, radl, _d = LI.sample(lgt, p1, st)
                    except Exception:
                        continue
                    # inp = 1 - |vn.lv| with vn the zero vector -> 1;
                    # a HEMI's 0.5*i+0.5 also lands on 1
                    acc += np.maximum(radl[0], 0.0)
                lit = np.maximum(acc, 0.0)
                shaded_cache[key] = lit
            crgb = crgb * lit[None, None, :]
            lit_v = lit
        # R200: each element's colour add is its own (rays and bolts
        # no longer ride the line colour once theirs is set; unset
        # keys fall back to it, keeping old scenes). The alpha add
        # reads the SUMMED field -- exactly what the single
        # accumulator fed it, so translucency is unchanged
        lc = int(spec.get('lines', 0) or 0)
        if np.any(totf != 0.0):
            lcol = _adj(np.asarray(spec.get('line_color',
                                            (1.0, 1.0, 1.0)),
                                   np.float32))
            if lc and np.any(linef != 0.0):
                crgb += linef[:, :, None] * lcol[None, None, :]
            if rayf is not None and np.any(rayf != 0.0):
                rcv = spec.get('ray_color')
                ray_col = lcol if rcv is None \
                    else _adj(np.asarray(rcv, np.float32))
                crgb += rayf[:, :, None] * ray_col[None, None, :]
            if boltf is not None and np.any(boltf != 0.0):
                bcv = spec.get('bolt_color')
                bolt_col = lcol if bcv is None \
                    else _adj(np.asarray(bcv, np.float32))
                crgb += boltf[:, :, None] * bolt_col[None, None, :]
            ca = ca + (totf * totf if xalpha else totf)
        if ringc and np.any(ringf != 0.0):
            rcol = _adj(np.asarray(spec.get('ring_color',
                                            (1.0, 1.0, 1.0)),
                                   np.float32))
            crgb += ringf[:, :, None] * rcol[None, None, :]
            ca = ca + (ringf * ringf if xalpha else ringf)
        ca = np.minimum(ca, 1.0).astype(np.float32)
        crgb = crgb.astype(np.float32)
        # addalphaAddfacFloat: dest weight slides with Add (R253: or
        # one of the fixed Blend modes -- the helper is the old two
        # statements verbatim under ADD_SLIDER)
        addfac = float(np.clip(spec.get('add', 0.0), 0.0, 1.0))
        bmode = str(spec.get('blend', 'ADD_SLIDER') or 'ADD_SLIDER')
        gs_ = min(max(float(spec.get('glow_size', 0.0) or 0.0), 0.0), 8.0)
        if gs_ > 0.0:
            # R253: the outer glow -- a second, wider Gaussian layer
            # under the core, the sprite engines' bloom. Its own
            # window, the same frame, the same depth rule (z-tested
            # against its own pixels unless Over Everything), the
            # same blend; composited BEFORE the core so the core
            # sits on top of it
            gwin = _window(ext * gs_)
            if gwin is not None:
                gx0, gx1, gy0, gy1, gxn, gyn = gwin
                gxa, gya = _frame(gxn, gyn)
                dg = (gxa * gxa + gya * gya) \
                    / max((rad * gs_) * (rad * gs_), 1e-12)
                g_in = dg < 1.0
                if sel_mask is not None:
                    g_in &= sel_mask[gy0:gy1, gx0:gx1]
                if not over:
                    g_in &= zbuf[gy0:gy1, gx0:gx1] > zs
                if g_in.any():
                    _gs = spec.get('glow_strength')
                    gstr = 1.0 if _gs is None else max(float(_gs), 0.0)
                    ag = (alpha0 * gstr
                          * np.clip((np.exp(-4.5 * np.minimum(dg, 1.0))
                                     - _gauss_k) / (1.0 - _gauss_k),
                                    0.0, 1.0)).astype(np.float32)
                    gcv = spec.get('glow_color')
                    gcol = col_r if gcv is None \
                        else _adj(np.asarray(gcv, np.float32))
                    if lit_v is not None:
                        gcol = gcol * lit_v
                    g_live = g_in & (ag > 0.00001)
                    if g_live.any():
                        ag = np.minimum(ag * ag if xalpha else ag, 1.0)
                        g_rgb = (ag[:, :, None]
                                 * gcol[None, None, :]).astype(np.float32)
                        gsub = out[gy0:gy1, gx0:gx1]
                        _halo_blend(gsub[:, :, :3], gsub[:, :, 3], g_rgb,
                                    ag.astype(np.float32), g_live, bmode,
                                    addfac)
        lv = live
        sub = out[y0:y1, x0:x1]
        sub_rgb = sub[:, :, :3]
        sub_a = sub[:, :, 3]
        _halo_blend(sub_rgb, sub_a, crgb, ca, lv, bmode, addfac)
    return out


def _light_volumes(img, scene, st, gbuf, vp, eye, w, h, bvh=None,
                   sel_mask=None):
    """Composite every volumetric lamp's beam over the finished frame.

    This used to run BEFORE shading, where the shading loop's
    img[py, px] assignment overwrote every beam pixel that crossed
    geometry -- beams only ever survived against the sky. It also
    clipped the beam against the raw NDC z (a number in [0.97, 0.99]
    on the field scene) as if it were metres, cutting every beam a
    hair from the camera wherever geometry stood. Now it runs over the
    finished frame, and the per-pixel distance is the surface point
    reconstructed through the inverse view-projection -- the exact
    distance, at the z-buffer's own precision.

    SPOT marches its cone, POINT a bounded sphere glow, AREA a
    soft-edged slab beam; a lamp with Beam Occlusion set traces each
    march sample back to the lamp through the BVH, which is what makes
    a beam STOP at a mesh in its way. `sel_mask` (an adaptive refine
    pass) restricts everything to the flagged pixels -- the values
    there are identical to a full evaluation, per-pixel independence
    again.
    """
    if not getattr(st, 'spot_cones', False):
        return img
    lights = [l for l in (getattr(scene, 'lights', None) or ())
              if str(getattr(l, 'type', '')).upper() in _VOLUME_KINDS
              and float(getattr(l, 'volumetric', 0.0)) > 0.0]
    if not lights:
        return img
    from . import cones as CONES
    inv = np.linalg.inv(np.asarray(vp, np.float64)).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    nx = (xx.ravel().astype(np.float32) + 0.5) / w * 2.0 - 1.0
    ny = (yy.ravel().astype(np.float32) + 0.5) / h * 2.0 - 1.0
    zz = gbuf.depth.reshape(-1).astype(np.float32)
    sel = None
    if sel_mask is not None:
        sel = np.asarray(sel_mask, bool).reshape(-1)
        if not sel.any():
            return img
        nx, ny, zz = nx[sel], ny[sel], zz[sel]
    npx = nx.size
    one = np.ones(npx, np.float32)
    far = np.stack([nx, ny, one, one], axis=1) @ inv.T
    far = far[:, :3] / np.where(np.abs(far[:, 3:4]) < 1e-9, 1e-9,
                                far[:, 3:4])
    dirs = M.normalize(far - eye[None, :])
    covered = np.isfinite(zz)
    dist = np.full(npx, np.inf, np.float32)
    if covered.any():
        hit_h = np.stack([nx[covered], ny[covered], zz[covered],
                          one[covered]], axis=1) @ inv.T
        hit = hit_h[:, :3] / np.where(np.abs(hit_h[:, 3:4]) < 1e-9, 1e-9,
                                      hit_h[:, 3:4])
        dist[covered] = np.linalg.norm(hit - eye[None, :], axis=1)

    reach = float(getattr(st, 'spot_cone_reach', 64.0))
    marchers = {'SPOT': CONES.spot_cone, 'POINT': CONES.point_glow,
                'AREA': CONES.area_beam}
    add = np.zeros((npx, 3), np.float32)
    for light in lights:
        kind = str(getattr(light, 'type', '')).upper()
        occ = None
        if getattr(light, 'volumetric_occlusion', False) and bvh is not None:
            lpos = np.asarray(light.position, np.float32)

            def occ(p, live, lpos=lpos):
                out = np.zeros(p.shape[0], bool)
                if not live.any():
                    return out
                delta = lpos[None, :] - p[live]
                dl = np.sqrt(np.maximum((delta * delta).sum(axis=1), 1e-12))
                dl_dir = delta / dl[:, None]
                out[live] = bvh.occluded(p[live] + dl_dir * 1e-3, dl_dir,
                                         dl * (1.0 - 1e-3))
                return out
        reach_l = reach
        if str(getattr(light, 'decay', '')) == 'CUSTOM':
            # Custom Range's end is the lamp's own idea of how far it
            # reaches; the beam honours it rather than growing a dial
            reach_l = min(reach, max(float(getattr(light, 'decay_end',
                                                   reach)), 1e-3))
        scatter = marchers[kind](
            eye, dirs, dist, light,
            samples=int(getattr(st, 'spot_cone_samples', 12)),
            density=float(getattr(st, 'spot_cone_density', 1.0))
            * float(light.volumetric),
            falloff=float(getattr(st, 'spot_cone_falloff', 2.0)),
            max_distance=reach_l, occlude=occ)
        if scatter.any():
            col = np.asarray(light.color, np.float32)[None, :]
            add += scatter[:, None] * col * float(light.energy) / np.pi
    if sel is None:
        img[:, :, :3] += add.reshape(h, w, 3)
    else:
        flat = img.reshape(-1, 4)
        flat[sel, :3] += add
    return img


def _plane_over_geometry(img, scene, st, job, gbuf, eye, textures):
    """The world's analytic ground/ocean plane, in FRONT of what it hides.

    The plane used to exist only in the background pass, which made it a
    BACKDROP: any geometry, however far below the water line, drew fully
    in front of it. A boat sat ON the ocean like a sticker, a character
    could never wade, and the floor of every Bryce scene was something
    objects hovered against rather than stood in. Bryce's water was real
    scene geometry and hid what sank; this pass gives the analytic plane
    the same authority.

    The test is exact and cheap: with the camera above the plane, a
    surface point below it is FURTHER along its own view ray than the
    plane crossing, always -- so coverage is just "is the shaded point
    under the plane". Those pixels are re-evaluated through the same
    world path the background uses (ocean, chequer, haze, cloud shadows,
    everything -- one source of truth), and the ocean lets the drowned
    geometry show through its own Transparency: the deeper the point
    sits, the more the water column absorbs it, until only water is
    left. Deterministic per pixel, so worker bands and adaptive-AA
    refine passes reproduce it bit for bit.

    Known edges, stated rather than hidden: the plane still writes no
    depth or object id (it is a background-family surface, as it always
    was), a transparent surface above the water over a drowned opaque
    one composites after this and stays visible, and a camera at or
    below the plane keeps the old backdrop-only behaviour. A
    transparent film skips the plane entirely -- the 2.79 contract says
    the world does not draw there at all.
    """
    w = getattr(scene, 'world', None)
    if w is None or not getattr(w, 'ground_plane', False) or eye is None:
        return img
    if str(getattr(w, 'mode', 'NODES')) == 'NODES':
        return img            # the analytic plane belongs to the sky models
    if getattr(st, 'film_transparent', False):
        return img
    ey = np.asarray(eye, np.float32)
    gh = float(getattr(w, 'ground_height', 0.0))
    if not (float(ey[2]) > gh):
        return img            # at or under the plane: backdrop only, as before
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    if py.size == 0:
        return img
    ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py)
    P = np.asarray(ctx.P, np.float32)
    under = P[:, 2] < gh
    if not under.any():
        return img
    py, px, P = py[under], px[under], P[under]
    delta = P - ey[None, :]
    dist_geo = np.linalg.norm(delta, axis=1)
    dist_geo = np.maximum(dist_geo, 1e-6)
    dirs = delta / dist_geo[:, None]
    # eye above, point below: the ray's z step is strictly negative and
    # the plane crossing sits strictly inside the segment
    t = (gh - float(ey[2])) / np.minimum(dirs[:, 2], -1e-9)
    t = np.clip(t, 0.0, dist_geo)
    plane_col = world_color(scene, st, dirs, textures or {}, dirs.shape[0],
                            eye=ey)
    out = img[py, px, :3]
    through = None
    if str(getattr(w, 'ground_mode', 'SOLID')) == 'OCEAN':
        trans = float(np.clip(getattr(w, 'ocean_transparency', 0.25),
                              0.0, 1.0))
        if trans > 0.0:
            # what the surface transmits (down the normal, mirror at a
            # graze) times what the water column lets survive -- a fixed
            # absorption length, because Bryce's water swallowed things
            # fast and a second dial for it would be a pointless option
            facing_down = np.clip(-dirs[:, 2], 0.0, 1.0)
            path = np.maximum(dist_geo - t, 0.0)
            absorb = np.exp(-path / 4.0).astype(np.float32)
            through = (trans * facing_down * absorb).astype(np.float32)
    if through is None:
        img[py, px, :3] = plane_col
    else:
        img[py, px, :3] = plane_col * (1.0 - through)[:, None] + \
            out * through[:, None]
    img[py, px, 3] = np.maximum(img[py, px, 3], 1.0)
    return img


def _background_image(scene, st, w, h, vp, eye, uncovered=None,
                      textures=None, ss=1, force_world=False):
    """World colour through the camera rays that miss geometry.

    Shading only the uncovered pixels matters: on a full-frame scene this is
    nearly the whole cost of the background pass, and it is entirely wasted
    work when something is drawn over the top of it.

    `force_world` evaluates the world even under a transparent film --
    the WIREFRAME model's see-through pixels are a Halcyon feature with
    its own contract (the world shows behind the wires, alpha marks the
    wire itself), not part of the 2.79 frame-fill contract below.
    """
    img = np.zeros((h, w, 4), np.float32)
    # An opaque film is the default; Blender's Film > Transparent (and the
    # matching Halcyon toggle) is what makes the background punch through.
    if getattr(st, 'film_transparent', False) and not force_world:
        # 2.79's transparent film (R_ALPHAPREMUL) carries NO sky at all:
        # the uncovered pixels are premultiplied (0,0,0,0), full stop.
        # The old path spent a quarter of the field's 3.1s GPU frame
        # evaluating a world colour into the rgb of pixels whose alpha
        # is zero -- values 2.79 never wrote, that no viewer shows, and
        # that leaked into whatever post read rgb regardless of alpha
        # (a glow could bloom an invisible sky). Zeros are both the
        # exact 2.79 contract and free. The AA resolve stays premul-
        # correct: edge pixels blend toward coverage-weighted colour,
        # which is what a premultiplied frame means.
        return img
    bg_alpha = 0.0 if getattr(st, 'film_transparent', False) else 1.0

    # Supersampling multiplies the number of background rays by ss squared, and
    # a sky is smooth almost everywhere -- at 4x that is sixteen evaluations of
    # a procedural sky or an HDRI lookup for one output pixel. Evaluating it at
    # output resolution and expanding costs a barely visible amount of detail on
    # a sun disc and saves the other fifteen sixteenths.
    if ss > 1 and getattr(st, 'fast_background', True):
        lw, lh = max(w // ss, 1), max(h // ss, 1)
        low_mask = None
        if uncovered is not None:
            # only the low-res blocks the mask can ever READ get evaluated:
            # a block is needed iff any of its output pixels is uncovered,
            # with the edge-padded bands folding into the last row/column
            # exactly as the pad reads them. The values of every pixel the
            # caller receives are bit-identical to evaluating the whole low
            # buffer -- the sky is per-pixel independent -- and on a frame
            # that is mostly geometry, most of the world evaluation (the
            # expensive part of a rich sky) simply never runs
            um = np.broadcast_to(uncovered, (h, w))
            core = um[:lh * ss, :lw * ss].reshape(lh, ss, lw, ss).any((1, 3))
            low_mask = core
            if lw * ss < w and um[:, lw * ss:].any():
                low_mask = low_mask.copy()
                low_mask[:, -1] |= um[:lh * ss, lw * ss:].any(axis=1) \
                    .reshape(lh, ss).any(1)
            if lh * ss < h and um[lh * ss:, :].any():
                if low_mask is core:
                    low_mask = low_mask.copy()
                low_mask[-1, :] |= um[lh * ss:, :lw * ss].any(axis=0) \
                    .reshape(lw, ss).any(1)
                if lw * ss < w and um[lh * ss:, lw * ss:].any():
                    low_mask[-1, -1] = True
            if not low_mask.any():
                return img
        low = _background_image(scene, st, lw, lh, vp, eye, low_mask,
                                textures, ss=1, force_world=force_world)
        big = np.repeat(np.repeat(low, ss, axis=0), ss, axis=1)
        if big.shape[0] < h or big.shape[1] < w:
            big = np.pad(big, ((0, max(h - big.shape[0], 0)),
                               (0, max(w - big.shape[1], 0)), (0, 0)), mode='edge')
        big = big[:h, :w]
        if uncovered is None:
            return big.copy()
        mask = np.broadcast_to(uncovered, (h, w))
        img[mask] = big[mask]
        return img
    if uncovered is None:
        yy, xx = np.mgrid[0:h, 0:w]
        yy = yy.ravel()
        xx = xx.ravel()
    else:
        yy, xx = np.nonzero(uncovered)
        if yy.size == 0:
            return img
    img[yy, xx, 3] = bg_alpha
    inv = np.linalg.inv(vp).astype(np.float32)
    nx = (xx.astype(np.float32) + 0.5) / w * 2.0 - 1.0
    ny = (yy.astype(np.float32) + 0.5) / h * 2.0 - 1.0
    pts = np.stack([nx, ny, np.ones(nx.size, np.float32),
                    np.ones(nx.size, np.float32)], axis=1)
    world = pts @ inv.T
    world = world[:, :3] / np.where(np.abs(world[:, 3:4]) < 1e-9, 1e-9, world[:, 3:4])
    dirs = M.normalize(world - eye[None, :])
    # R251 (sky-camera C056): Doom's cylinder sky is a function of the
    # OUTPUT pixel, not of the ray: the per-column angle table and the
    # 1:1 rows are built ONCE at the output size from the frame's own
    # projection and J (jitter, stereo window), the same bytes
    # gpu/sky.py uploads. Reflections never come here (they evaluate
    # world_color along a direction and see the flat colour)
    from . import sky as SKY
    wd = getattr(scene, 'world', None)
    total = dirs.shape[0]
    if wd is not None and str(getattr(wd, 'mode', 'NODES')) == 'CYLINDER':
        tex = SKY.env_texture(wd, textures or {})
        Wo, Ho = max(w // ss, 1), max(h // ss, 1)
        proj_c, jx, jy, yaw = SKY.cylinder_inputs(scene, st, Wo, Ho)
        if tex is not None:
            cols = SKY.cylinder_pixels(wd, yaw, proj_c, w, h, ss, xx, yy,
                                       tex, jx, jy)
        else:
            # no image: the solid colour then strength, hdri()'s own rule
            cols = (SKY.solid(wd, dirs)
                    * float(getattr(wd, 'strength', 1.0))).astype(np.float32)
        if getattr(wd, 'ground_plane', False):
            cols = SKY.ground_plane(wd, dirs, cols, eye,
                                    getattr(scene, 'time', 0.0), textures)
        img[yy, xx, :3] = cols
    # the background is a large independent job and was the third biggest
    # stage on a real frame, so it is chunked like the shading
    elif total > 4 * MIN_CHUNK:
        # chunked to bound memory on a large frame, not for parallelism
        cols = np.empty((total, 3), np.float32)
        step = MAX_CHUNK
        for lo in range(0, total, step):
            hi = min(lo + step, total)
            cols[lo:hi] = world_color(scene, st, dirs[lo:hi], textures or {},
                                      hi - lo, eye=eye)
        img[yy, xx, :3] = cols
    else:
        img[yy, xx, :3] = world_color(scene, st, dirs, textures or {}, total,
                                      eye=eye)
    # R251 (sky-camera C048): the Mode 7 floor, drawn by OUTPUT pixel over
    # whichever sky ran (the analytic ground_plane() leaves a MODE7 world
    # untouched), from per-row integer registers gpu/sky.py uploads
    if wd is not None and getattr(wd, 'ground_plane', False) and \
            str(getattr(wd, 'ground_mode', 'SOLID')) == 'MODE7':
        img[yy, xx, :3] = SKY.mode7_floor(wd, inv, eye, w, h, ss, xx, yy,
                                          img[yy, xx, :3])
    if getattr(st, 'fog', False) and \
            str(getattr(st, 'fog_mode', 'LINEAR')) == 'GROUND':
        # R251 LIGHT-A2 (F009): POV's fog_type 2 fogs a ray that hits
        # nothing by its elevation (the closed form); gpu/sky.py's
        # hal_ground_fog_sky is the twin, on the same ray
        from . import fog as _FOG
        img[yy, xx, :3] = _FOG.ground_fog_sky(img[yy, xx, :3], dirs, st,
                                              eye)
    return img


def _bump_materials_of(job, tri):
    """{mi: material} for fragment materials carrying a Bump node."""
    mesh = job.scene.mesh
    if mesh.mat_index is None or tri.size == 0:
        return {}
    out = {}
    for mi in np.unique(mesh.mat_index[tri]):
        mi = int(mi)
        mat = job.scene.materials[mi] \
            if mi < len(job.scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        nodes = (graph or {}).get('nodes', {})
        if any(n.get('bl_idname') == 'ShaderNodeBump'
               for n in nodes.values()):
            out[mi] = mat
    return out


def _shade_fragments_cpu(job, tri, bary, px, py, front, rank, st):
    """The A-buffer fragments' CPU shading, scheduling-invariant.

    A Bump node's screen gradients must be a function of the SURFACE,
    not of the batch layout -- and for transparent fragments the surface
    is the LAYER: one fragment per pixel per rank. Shading all ranks in
    one mixed array made `_screen_grad`'s scatter collide (front and
    back faces at the same pixel, last write winning by sort order) and
    let every chunk boundary cut the waves: 539 of 1914 fragments moved
    with the chunk size in the repro scene, by up to 2.98. So when any
    fragment's material carries a Bump node, the fragments shade RANK
    BY RANK with whole-material gradient fields built from each rank's
    own fragments -- `_bump_height_fields`, the same pre-pass the opaque
    frame runs, applied per layer. Frames without a Bump material take
    the old single call: nothing else reads screen gradients here, and
    per-fragment shading is proven chunking-invariant.
    """
    bump_mats = _bump_materials_of(job, tri)
    if not bump_mats:
        return _shade_chunked(job, tri, bary, px, py, front, None, st)
    from .nodeeval import VALUE, GraphEvaluator
    mesh = job.scene.mesh
    mat_f = mesh.mat_index[tri] if mesh.mat_index is not None \
        else np.zeros(tri.size, np.int32)

    # heights are per-fragment pure, so each material's chain evaluates
    # ONCE over ALL its fragments (chunked for memory); each rank then
    # scatters ITS OWN subset and differences on the frame grid -- the
    # same field definition, at a fraction of the evaluator cost the
    # first per-rank version paid (the field's 33-second frame grew to
    # 43 on exactly that; the waves' noise chain was re-running once
    # per depth layer)
    H, W = job.height, job.width
    heights = {}
    mat_idx_of = {}
    for mi, mat in bump_mats.items():
        idx = np.nonzero(mat_f == mi)[0]
        if idx.size == 0:
            continue
        mat_idx_of[mi] = idx
        graph = getattr(mat, 'graph', None)
        nodes = (graph or {}).get('nodes', {})
        for node in nodes.values():
            if node.get('bl_idname') != 'ShaderNodeBump':
                continue
            hv = np.empty(idx.size, np.float32)
            for s in range(0, int(idx.size), int(MAX_CHUNK)):
                e = min(s + int(MAX_CHUNK), int(idx.size))
                sub = idx[s:e]
                ctx = job.context(tri[sub], bary[sub], px[sub], py[sub],
                                  front[sub] if front is not None
                                  else None, None, 0, True)
                ev = GraphEvaluator(graph, ctx, job.textures,
                                    getattr(mat, 'programs', None))
                hv[s:e] = np.asarray(ev.input(node, 'Height', VALUE),
                                     np.float32).reshape(-1)
            heights[(mi, node.get('id'))] = hv

    col = np.zeros((tri.size, 4), np.float32)
    stash = dict(getattr(job, 'bump_fields', {}) or {})
    try:
        top = int(rank.max()) if rank.size else -1
        # one stable sort finds every layer; sixteen `rank == r` scans
        # over millions of fragments used to. Stable argsort keeps equal
        # ranks in original index order, so each slice is bit-identical
        # to nonzero's ascending indices.
        rorder = np.argsort(rank, kind='stable')
        rbounds = np.searchsorted(rank[rorder], np.arange(top + 2))
        for r in range(top + 1):
            sel = rorder[rbounds[r]:rbounds[r + 1]]
            if sel.size == 0:
                continue
            fields = dict(stash)
            for (mi, node_id), hv in heights.items():
                idx = mat_idx_of[mi]
                m = rank[idx] == r
                if not m.any():
                    continue
                img = np.zeros((H, W), np.float32)
                valid = np.zeros((H, W), bool)
                img[py[idx[m]], px[idx[m]]] = hv[m]
                valid[py[idx[m]], px[idx[m]]] = True
                gx = np.zeros_like(img)
                gy = np.zeros_like(img)
                gx[:, :-1] = np.where(valid[:, 1:] & valid[:, :-1],
                                      img[:, 1:] - img[:, :-1], 0.0)
                gy[:-1, :] = np.where(valid[1:, :] & valid[:-1, :],
                                      img[1:, :] - img[:-1, :], 0.0)
                fields[(mi, node_id)] = (gx, gy)
            job.bump_fields = fields
            col[sel] = _shade_chunked(job, tri[sel], bary[sel], px[sel],
                                      py[sel],
                                      front[sel] if front is not None
                                      else None, None, st)
    finally:
        job.bump_fields = stash
    return col


# ---------------------------------------------------------------- R251
# The transparency pack (1.90.0): the blend unit's fixed equations, the
# framebuffer format at the blend, the DS composite rules and the Doom
# fuzz, all in ONE composite skeleton that runs on the CPU on both roads
# (CAP 'abuffer' NEVER). Every rule that quantises a layer colour refuses
# the GPU LAYER passes by name inside the routing block, so its twin is
# CPU vs CPU by construction; the ALPHA road under framebuffer NONE keeps
# 1.89.0's float over byte for byte.

#: the composite's one table of equation ids -- properties.BLEND_EQUATION
#: carries the SAME numbers explicitly (Blender stores the integer in the
#: .blend, so a positional list would renumber every saved scene)
MODE_INDEX = {'ALPHA': 0, 'PS1_AVG': 1, 'PS1_ADD': 2, 'PS1_SUB': 3,
              'PS1_QUARTER': 4, 'SATURN_HALF': 5, 'SATURN_SHADOW': 6,
              'SATURN_HALF_LUM': 7, 'THREEDO_SUB': 8, 'THREEDO_XOR': 9,
              'SNES_ADD': 10, 'SNES_SUB': 11, 'SNES_ADD_HALF': 12,
              'SNES_SUB_HALF': 13, 'GBA': 14, 'DS': 15, 'FUZZ': 16,
              'THIN_WALL': 17, 'IMAGINE_FOG': 18}
#: every equation that reads F's rgb: `to8(F) >> 3` turns a one-ulp
#: simulator/driver difference at a to8 boundary into a whole lattice
#: level, so these refuse the GPU layers by name. ALPHA keeps today's
#: layer twin; SATURN_SHADOW and FUZZ read only `a > 0` (a baked constant)
MODES_READ_F = frozenset(MODE_INDEX.values()) - {0, 6, 16}
#: the one-sub-screen chips: only the layer composited LAST at a pixel
#: blends, every layer beneath it is drawn opaque (SNES colour math has
#: one sub screen; the GBA blends the top two layers)
MODES_ONE_SUB = frozenset({10, 11, 12, 13, 14})
#: id's r_draw.c fuzzoffset, in rows (+1 = one row up, -1 = one row down)
FUZZ_T = np.array([+1, -1, +1, -1, +1, +1, -1, +1, +1, -1, +1, +1, +1, -1,
                   +1, +1, +1, -1, -1, -1, -1, +1, -1, -1, +1, +1, +1, +1,
                   -1, +1, -1, +1, +1, -1, -1, +1, +1, -1, -1, -1, -1, +1,
                   +1, +1, +1, -1, +1, +1, -1, +1], np.int32)


def _to8(c):
    """float rgb/alpha -> 8-bit int32, np.round half-to-even (the named
    tie rule of every integer equation below)."""
    return np.round(np.clip(c, 0.0, 1.0) * np.float32(255)).astype(np.int32)


def _from8(k):
    """8-bit int32 -> float32, one float32 division."""
    return k.astype(np.float32) / np.float32(255)


def _mode_table(scene, st):
    """(mode_of_mat (M,) int32, the global mode): each material's equation
    id -- its own Blend Mode, or the render's Blend Equation at INHERIT;
    an unknown name reads as ALPHA."""
    g = str(getattr(st, 'blend_equation', 'ALPHA') or 'ALPHA')
    g_id = MODE_INDEX.get(g, 0)
    mats = scene.materials or []
    ids = []
    for m in mats:
        mm = str(getattr(m, 'blend_mode', 'INHERIT') or 'INHERIT')
        ids.append(MODE_INDEX.get(mm, 0) if mm != 'INHERIT' else g_id)
    return np.array(ids or [g_id], np.int32), g


def zoffs_scale(camera):
    """R251 C126 (Blender 2.4x Zoffs): the factor that turns a material's
    Z Offset (scene units) into the depth buffer's own units. The raster
    tests and stores ndc z, so a constant offset must live there -- as
    2.4x's zbuf.c did, `polygon_offset = zoffs * 0x7FFFFFFF /
    (clipend - clipsta)`: the offset divided by the camera's clip range.
    Exact scene units under an orthographic camera (ndc z is linear
    there); under perspective the world-space effect grows with
    distance, exactly 2.4x's behaviour, disclosed. One float32."""
    near = float(getattr(camera, 'clip_start', 0.1) or 0.1)
    far = float(getattr(camera, 'clip_end', 1000.0) or 1000.0)
    return np.float32(1.0 / max(far - near, 1e-6))


def composite_reads_neighbours(scene, st):
    """True when the transparent composite reads a NEIGHBOURING pixel of
    the finished frame (Fuzz: the row above/below; Thin Wall: the jogged
    background) -- which a pooled row band never holds at its seam, so
    the engine's worker-pool gate skips the pool by name for it."""
    if str(getattr(st, 'transparency', 'NONE')) not in ('SORTED', 'ABUFFER'):
        return False
    if str(getattr(st, 'blend_equation', 'ALPHA')) in ('FUZZ', 'THIN_WALL'):
        return True
    return any(str(getattr(m, 'blend_mode', 'INHERIT')) in ('FUZZ', 'THIN_WALL')
               for m in (scene.materials or ()))


class _Framebuffer:
    """The framebuffer format the frame is written INTO (C018 + C070):
    PS2 PSMCT16, GameCube RGBA6, 3dfx Voodoo RGB565 -- every write
    truncates, every blend reads the truncated value back. NONE is the
    identity (the SAME array back, no copy). `rows` = the internal row
    range a banded worker owns; the dither is indexed by the FRAME row,
    so a banded frame packs each pixel exactly once and bitwise the
    whole frame. The frame is kept as the format's EXPANDED read-back
    value at all times (every expansion is injective)."""

    def __init__(self, st, width, height, rows=None):
        from . import dither as DI
        fmt = str(getattr(st, 'framebuffer', 'NONE') or 'NONE')
        if fmt not in DI.FB_FORMATS:
            print(f'[Halcyon] transparency: framebuffer format {fmt!r} is '
                  'not one of the formats; the frame stays 24-bit float '
                  '(NONE), by name')
            fmt = 'NONE'
        self.fmt = fmt
        self.on = fmt != 'NONE'
        self.dither = bool(getattr(st, 'fb_dither', True))
        self.subtract = bool(getattr(st, 'fb_dither_subtract', True))
        self.width = int(width)
        self.height = int(height)
        self.rows = rows
        self._DI = DI

    def enter(self, img):
        if not self.on:
            return img
        r0, r1 = (0, img.shape[0]) if self.rows is None else \
            (max(int(self.rows[0]), 0), min(int(self.rows[1]), img.shape[0]))
        if r1 <= r0:
            return img
        ys, xs = np.mgrid[r0:r1, 0:img.shape[1]]
        c8 = _to8(img[r0:r1, :, :3])
        img[r0:r1, :, :3] = _from8(
            self._DI.fb_pack8(c8, xs, ys, self.fmt, self.dither))
        return img

    def leave(self, img):
        return img

    def read(self, rgb, xx, yy):
        """The destination a blend SEES at (xx, yy): identity under NONE,
        the format's read-back (the Voodoo's dither subtraction) else."""
        if not self.on:
            return rgb
        return _from8(self._DI.fb_read8(_to8(rgb), xx, yy, self.fmt,
                                        self.dither, self.subtract))

    def over(self, F, a, B):
        """ALPHA: 1.89.0's float over exactly (R:7606) under NONE; the
        8-bit integer over of every 16-bit-era blend unit else."""
        if not self.on:
            aa = a[:, None]
            return F * aa + B * (1.0 - aa)
        a8 = _to8(a)
        return _from8(self._DI.fb_over8(_to8(F), a8, _to8(B)))

    def write(self, out, xx, yy):
        if not self.on:
            return out
        return _from8(self._DI.fb_pack8(_to8(out), xx, yy, self.fmt,
                                        self.dither))


def framebuffer_pack_frame(img, st, rows=None):
    """The whole-frame pack for a frame the composite never ran on (no
    see-through fragments): the same truncation the composite's entry
    applies, so a 16-bit buffer's opaque frame is on its lattice too."""
    fb = _Framebuffer(st, img.shape[1], img.shape[0], rows=rows)
    return fb.leave(fb.enter(img))


def _layer_order(rank_s, st):
    """The layers in composite order: DEPTH draws the farthest kept layer
    first (R:7599, unchanged); the DS orders (Y_SORT / SUBMISSION) draw
    the first-sorted layer first."""
    top = int(rank_s[-1])
    if str(getattr(st, 'translucent_order', 'DEPTH')) == 'DEPTH':
        return range(top, -1, -1)
    return range(0, top + 1)


def _thin_wall_jog(frag, job, st, width, height):
    """R251 C095 (3ds Max Thin Wall Refraction): the screen-space jog of
    each fragment -- (qx, qy) int64, the frame pixel the pane shows. The
    shading normal is rebuilt from the mesh exactly as shading used it
    (corner normals under smooth, the face normal under flat, the
    normal_source override, the two-sided flip toward the eye), taken
    into view space by three explicit float32 sums in a fixed order
    (never a BLAS kernel's), scaled by Thickness Offset x (IOR - 1) x
    16 px at 480 lines (Halcyon's own calibration of Max's unpublished
    constant, disclosed) and rounded half-to-even (np.round: a jog of
    exactly n + 0.5 px lands on the even pixel, named), edge-clamped."""
    mesh = job.scene.mesh
    t = frag['tri']
    bary = np.asarray(frag['bary'], np.float32)
    w0, w1, w2 = bary[:, 0], bary[:, 1], bary[:, 2]
    n_c = np.asarray(mesh.normals, np.float32)[mesh.tris[t]]
    n = w0[:, None] * n_c[:, 0] + w1[:, None] * n_c[:, 1] \
        + w2[:, None] * n_c[:, 2]
    sm = np.ones(t.size, bool) if mesh.smooth is None \
        else np.asarray(mesh.smooth, bool)[t].copy()
    ns = str(getattr(st, 'normal_source', 'AUTO'))
    if ns == 'FACE':
        sm[:] = False
    elif ns == 'SMOOTH':
        sm[:] = True
    n = np.where(sm[:, None], n,
                 np.asarray(mesh.face_normals, np.float32)[t])
    n = n / np.maximum(np.sqrt((n * n).sum(1, keepdims=True)),
                       np.float32(1e-20))
    front = frag.get('front')
    if front is not None:
        n = np.where(np.asarray(front, bool)[:, None], n, -n)
    n = n.astype(np.float32)
    V = np.asarray(job.view, np.float32)[:3, :3]
    n_vx = (n[:, 0] * V[0, 0] + n[:, 1] * V[0, 1]) + n[:, 2] * V[0, 2]
    n_vy = (n[:, 0] * V[1, 0] + n[:, 1] * V[1, 1]) + n[:, 2] * V[1, 2]
    mats = job.scene.materials or []
    thick_of = np.array([np.float32(getattr(m, 'thin_wall_offset', 0.5))
                         for m in mats] or [0.5], np.float32)
    ior_of = np.array([np.float32(getattr(m, 'ior', 1.45))
                       for m in mats] or [1.45], np.float32)
    mf = np.clip(np.asarray(frag['mat_f']), 0, thick_of.size - 1)
    k = ((thick_of[mf] * (ior_of[mf] - np.float32(1.0)))
         * np.float32(16.0)) * np.float32(height / 480.0)
    ox = k * n_vx
    oy = k * n_vy
    xx = np.asarray(frag['xx'], np.float32)
    yy = np.asarray(frag['yy'], np.float32)
    qx = np.clip(np.round(xx + ox), 0, width - 1).astype(np.int64)
    qy = np.clip(np.round(yy + oy), 0, height - 1).astype(np.int64)
    return qx, qy


def _view_depth(job, P):
    """|z| of world points P along the view axis, scene units -- the
    sequential float32 chain shade_batch's ctx.depth carries (three
    products, two sums, no matmul kernel)."""
    dP = np.asarray(P, np.float32) - np.asarray(job.eye, np.float32)[None, :]
    r = np.asarray(job.view, np.float32)[2, :3]
    dz = dP[:, 0] * r[0]
    dz += dP[:, 1] * r[1]
    dz += dP[:, 2] * r[2]
    return np.abs(dz).astype(np.float32)


def _imagine_fog_spans(mode, pix, mat_f, tri, bary, px, py, front, gbuf,
                       job):
    """R251 C101 (Imagine Fog Length): per fog fragment, the opacity of
    its (pixel, material) group and whether it is the group's NEAREST
    fragment (the one that carries the fog). Returns (a_fog (N,)
    float32, first_fog (N,) bool) in the composite's fragment order.

    The thickness is measured along the VIEW AXIS in scene units from
    the fragments' own surface points (the raster's depth is the
    buffer's ndc z, so both the fragments and the opaque surface behind
    them are taken to view depth through the same float32 chain
    ctx.depth uses); the successor of a fog fragment is found among the
    FOG MATERIAL'S OWN fragments at the pixel (a pane or the Box
    between a front and its back would otherwise orphan the back), a
    winding count per (pixel, material) in depth order makes nested or
    interpenetrating shells of one material count their union ONCE and
    shells one behind the other sum; an unmatched back face (open mesh)
    adds nothing, an unmatched front spans to the opaque surface (the
    camera's clip end where nothing is behind). Enter / leave is the
    face normal against the eye ray (geometric, never the winding flag);
    a shared-edge pixel two coplanar triangles both rasterise counts
    once. One float32 divide by Fog Length, one min: opacity = min(1,
    D / L); L <= 0 saturates."""
    n = pix.size
    a_fog = np.zeros(n, np.float32)
    first_fog = np.zeros(n, bool)
    fi = np.nonzero(mode == 18)[0]
    if fi.size == 0:
        return a_fog, first_fog
    P_f = job.attributes(tri[fi], bary[fi], None, need={'P'})[0]
    dz_all = _view_depth(job, P_f)
    far = np.float32(getattr(job.scene.camera, 'clip_end', 1000.0) or 1000.0)
    oz_all = np.full(fi.size, far, np.float32)
    otri = gbuf.tri[py[fi], px[fi]]
    cov = otri >= 0
    if bool(cov.any()):
        P_o = job.attributes(otri[cov], gbuf.bary[py[fi][cov], px[fi][cov]],
                             None, need={'P'})[0]
        oz_all[cov] = _view_depth(job, P_o)
    # enter / leave by the GEOMETRIC facing (the face normal against the
    # eye ray), never the raster's winding flag: a mesh wound inward
    # would otherwise leave through its front face
    mesh = job.scene.mesh
    fn = np.asarray(mesh.face_normals, np.float32)[tri[fi]]
    to_eye = np.asarray(job.eye, np.float32)[None, :] - np.asarray(P_f, np.float32)
    enter_all = (fn[:, 0] * to_eye[:, 0] + fn[:, 1] * to_eye[:, 1]
                 + fn[:, 2] * to_eye[:, 2]) > 0.0
    o2 = np.lexsort((dz_all, mat_f[fi], pix[fi]))
    fi, dz, oz, frz = fi[o2], dz_all[o2], oz_all[o2], enter_all[o2]
    pz, mz = pix[fi], mat_f[fi]
    same = (pz[1:] == pz[:-1]) & (mz[1:] == mz[:-1])
    # a pixel on an edge two coplanar triangles share is rasterised by
    # BOTH (the raster's coverage wobble window): the second fragment,
    # within the A-buffer's own depth tolerance of the first and facing
    # the same way, is the same surface and counts once, named
    dup = np.zeros(fi.size, bool)
    if fi.size > 1:
        tol = np.abs(dz[:-1]) * raster.ABUF_DEPTH_TOL_REL + raster.ABUF_DEPTH_TOL_ABS
        dup[1:] = same & (frz[1:] == frz[:-1]) & (np.abs(dz[1:] - dz[:-1]) <= tol)
    z_next = np.append(np.where(same, dz[1:], oz[:-1]), oz[-1:])
    start = np.r_[True, ~same]
    gid = np.cumsum(start) - 1
    g0 = np.nonzero(start)[0][gid]
    w = np.where(dup, 0, np.where(frz, 1, -1)).astype(np.int32)
    cs = np.cumsum(w)
    w = cs - (cs[g0] - w[g0])
    inside = w > 0
    span = np.where(inside, np.maximum(z_next - dz, np.float32(0.0)),
                    np.float32(0.0)).astype(np.float32)
    D = np.zeros(int(gid[-1]) + 1, np.float32)
    np.add.at(D, gid[inside], span[inside])
    mats = job.scene.materials or []
    L_of = np.array([np.float32(getattr(m, 'fog_length', 1.0))
                     for m in mats] or [1.0], np.float32)
    L = L_of[np.clip(mz, 0, L_of.size - 1)]
    a_g = np.where(L > 0.0, np.minimum(np.float32(1.0), D[gid] / L),
                   np.float32(1.0)).astype(np.float32)
    a_fog[fi] = a_g
    first_fog[fi[start]] = True
    return a_fog, first_fog


def _fixed_blend(m_id, F, a, B, xx, yy, last, img, fb, job, st, height,
                 frag=None):
    """One fixed equation over its fragments: (out rgb float32, cov,
    alpha_max-or-None). Every integer step int32; `to8` half-to-even;
    `>>` arithmetic. `drawn = a > 0`: an alpha chain that resolved to
    exactly 0 draws nothing -- no other alpha weighting exists on these
    chips (GBA / DS weight by their own fixed-point alpha). `frag` is
    the per-fragment mesh data the software items read (tri, bary,
    front, mat_f, xx, yy; the Imagine fog's a_fog / first_fog)."""
    n = a.size
    drawn = a > 0.0
    if m_id == 17:
        # THIN_WALL (3ds Max Thin Wall Refraction): the frame BENEATH
        # this layer read at the jogged pixel, then the opacity over it.
        # The PANE is the jog: at opacity 0 a clear window still bends
        # the view (Max's invisibility dial is Thickness Offset, not
        # opacity), so every fragment is drawn. Read before the write
        # (order-free within the layer)
        qx, qy = _thin_wall_jog(frag, job, st, img.shape[1], height)
        behind = fb.read(img[qy, qx, :3], qx, qy)
        out = fb.over(F, a, behind)
        return out.astype(np.float32), a.copy(), None
    if m_id == 18:
        # IMAGINE_FOG (Imagine Fog Length): the NEAREST fog fragment of
        # the material at the pixel carries the whole fog -- its shaded
        # colour (the entry point's colour chain) over B at the span
        # opacity; every other fragment of the material there is not
        # drawn. Unlit by construction under the Constant model
        af = np.asarray(frag['a_fog'], np.float32)
        first = np.asarray(frag['first_fog'], bool)
        out = np.where(first[:, None], fb.over(F, af, B), B)
        cov = np.where(first, af, np.float32(0.0)).astype(np.float32)
        return out.astype(np.float32), cov, None
    if m_id == 16:
        # FUZZ (Doom Spectre): the frame one row up or down, at 26/32,
        # the surface's own colour never appears; per (x, y, frame, seed)
        from . import film as _film
        frame = int(getattr(job.scene, 'frame', 1) or 1)
        start = (_film._hash_u32_raw(xx, np.full_like(xx, frame),
                                     int(getattr(st, 'seed', 0)))
                 % np.uint32(50)).astype(np.int32)
        k = (yy.astype(np.int32) + start) % 50
        sy = np.clip(yy + FUZZ_T[k], 1, max(height - 2, 0))
        src = fb.read(img[sy, xx, :3], xx, sy)
        out = np.where(drawn[:, None], src * np.float32(0.8125), B)
        return out.astype(np.float32), \
            np.where(drawn, np.float32(1.0), np.float32(0.0)), None
    if m_id == 15:
        # DS: (F*(A+1) + B*(31-A)) / 32 on 6-bit channels, A = 5-bit
        # polygon alpha; A 0 draws nothing, A 31 replaces; alpha = max
        a5 = np.clip(np.round(a * np.float32(31)), 0, 31).astype(np.int32)
        f6 = _to8(F) >> 2
        b6 = _to8(B) >> 2
        a5c = a5[:, None]
        o = (f6 * (a5c + 1) + b6 * (31 - a5c)) >> 5
        o = np.where(a5c == 31, f6, o)
        exp = _from8((o << 2) | (o >> 4))
        drawn = a5 > 0
        out = np.where(drawn[:, None], exp, B)
        amax = a5.astype(np.float32) / np.float32(31)
        return out.astype(np.float32), np.zeros(n, np.float32), amax
    if m_id in MODES_ONE_SUB:
        # SNES colour math / GBA BLDALPHA: one sub screen -- the layer
        # composited LAST blends, every layer beneath it REPLACES
        f = _to8(F) >> 3
        b = _to8(B) >> 3
        if m_id == 10:
            o = np.minimum(31, b + f)
        elif m_id == 11:
            o = np.maximum(0, b - f)
        elif m_id == 12:
            o = (b + f) >> 1
        elif m_id == 13:
            o = np.maximum(0, b - f) >> 1
        else:
            eva = np.clip(np.round(a * np.float32(16)), 0, 16).astype(np.int32)
            evb = 16 - eva
            o = np.minimum(31, (f * eva[:, None] + b * evb[:, None]) >> 4)
        blended = _from8((o << 3) | (o >> 2))
        out = np.where(last[:, None], blended, F)
        out = np.where(drawn[:, None], out, B)
        return out.astype(np.float32), \
            np.where(drawn, np.float32(1.0), np.float32(0.0)), None
    # PS1 / Saturn VDP1 / 3DO PIXC: 5-bit of the 8-bit values, one
    # integer expression, zero-fill expand (the 15-bit VRAM read)
    f = _to8(F) >> 3
    b = _to8(B) >> 3
    if m_id == 1:
        o = (b + f) >> 1
    elif m_id == 2:
        o = np.minimum(31, b + f)
    elif m_id == 3:
        o = np.maximum(0, b - f)
    elif m_id == 4:
        o = np.minimum(31, b + (f >> 2))
    elif m_id == 5:
        o = (b >> 1) + (f >> 1)
    elif m_id == 6:
        o = b >> 1
    elif m_id == 7:
        o = f >> 1
    elif m_id == 8:
        o = np.clip(f - b, 0, 31)
    elif m_id == 9:
        o = f ^ b
    else:
        return fb.over(F, a, B), a.copy(), None
    out = np.where(drawn[:, None], _from8(o << 3), B)
    return out.astype(np.float32), \
        np.where(drawn, np.float32(1.0), np.float32(0.0)), None


#: how the last GPU-gated A-buffer frame routed its depth layers --
#: per-layer fragment counts, the threshold, and who shaded what. The
#: printed routing line reads from it; the tests assert on it so a
#: "hybrid" run can never silently be a pure one.
LAST_ROUTING = {}


def _fmt_frags(n):
    """1234567 -> '1.2M': the routing line is read off a console."""
    n = int(n)
    if n >= 1000000:
        return f'{n / 1e6:.1f}M'
    if n >= 1000:
        return f'{n / 1e3:.1f}k'
    return str(n)


def _composite_abuffer(job, frags, gbuf, img, st, band=None, vp=None,
                       snap=0.0, rows=None):
    """True A-buffer: shade every fragment, sort per pixel, composite.

    Two things used to make this the slowest stage in the renderer by a wide
    margin:

    The shading called job.shade directly, so every transparent fragment in the
    frame was shaded on one thread no matter what the thread count said. It goes
    through the chunked pool now, like the opaque pass.

    The compositing walked depth layers with `rank == r`, which is a full scan
    and a full gather over *every* fragment in the frame for *every* layer. A
    pixel a hundred deep in glass therefore cost a hundred passes over millions
    of fragments. Sorting by layer first makes each layer a contiguous slice, so
    the whole composite is one pass in total.

    R251 (the transparency pack): the framebuffer format wraps every exit
    (the format is the frame's, not the fragment count's); the sort takes
    the DS orders; each fragment blends by its material's equation (the
    render's Blend Equation at INHERIT); a fixed equation refuses the GPU
    layers by name in the routing block; the alpha plane takes the DS's
    max and the DS depth write / same-ID-once rules run per layer.
    """
    W, H = gbuf.width, gbuf.height
    fb = _Framebuffer(st, W, H, rows=rows)
    img = fb.enter(img)
    px, py, tri, depth, bary, front = frags.finish()
    if px.size == 0:
        return fb.leave(img)
    opaque_z = gbuf.depth[py, px]
    # the SAME tolerant limit the collection used (raster.abuf_depth_limit):
    # a coplanar contact must not flip on which rasteriser rounded the
    # opaque depth's last ULP
    keep = depth <= raster.abuf_depth_limit(opaque_z)
    if not np.any(keep):
        return fb.leave(img)
    px, py, tri, depth, bary, front, opaque_z = (
        a[keep] for a in (px, py, tri, depth, bary, front, opaque_z))

    mesh = job.scene.mesh
    # R251 C126 (Blender 2.4x Zoffs / ZInvert): per material, the Z
    # Offset the raster ALREADY applied at collection (the ABUFFER
    # `depth` carries it; Sorted Blend's centroid key takes it here)
    # and the Invert Z flag (the material's fragments sort far-first,
    # in front of every ordinary fragment -- Depth order only)
    _mats0 = job.scene.materials or []
    zo_of_mat = np.array([np.float32(getattr(m, 'z_offset', 0.0))
                          for m in _mats0] or [0.0], np.float32)
    zi_of_mat = np.array([bool(getattr(m, 'z_invert', False))
                          for m in _mats0] or [False], bool)
    if mesh.mat_index is not None:
        mat_f0 = np.clip(mesh.mat_index[tri], 0, zo_of_mat.size - 1)
    else:
        mat_f0 = np.zeros(tri.size, np.int32)
    order_mode = str(getattr(st, 'translucent_order', 'DEPTH') or 'DEPTH')
    if order_mode == 'Y_SORT' and vp is None:
        print('[Halcyon] transparency: Y-sort needs the projection (vp '
              'missing); composited by depth')
        order_mode = 'DEPTH'
    pix = py.astype(np.int64) * W + px
    if order_mode == 'Y_SORT':
        # the DS auto-sort: polygons with the lower bottom screen row
        # first, then the lower top row, then submission -- never depth.
        # The rows are the raster's own projection (the snapped rows an
        # INTEGER preset rasterised with); a corner behind the eye has a
        # meaningless row and is clamped to the frame (the DS clipped
        # such a polygon and sorted the clipped rows: the named stand-in)
        _c, screen, _iw, _z = raster.project(mesh.verts, vp, W, H, snap=snap)
        row = np.clip(np.floor(screen[:, 1]), 0, H - 1).astype(np.int64)
        rows3 = row[mesh.tris]
        ytop = (H - 1) - rows3.max(axis=1)
        ybot = (H - 1) - rows3.min(axis=1)
        order = np.lexsort((tri, ytop[tri], ybot[tri], pix))
    elif order_mode == 'SUBMISSION':
        order = np.lexsort((tri, pix))
    else:
        # Sorted Blend orders whole polygons by their centroid, which is
        # what a renderer without per-fragment lists could manage -- and
        # it shows the classic sorting errors where surfaces
        # interpenetrate. A-Buffer sorts every fragment on its own depth
        # and is correct through any arrangement.
        if st.transparency == 'SORTED':
            cent = mesh.verts[mesh.tris].mean(axis=1)
            view_z = np.abs((cent - job.eye[None, :])
                            @ job.view[:3, :3].T)[:, 2]
            key = view_z[tri].astype(np.float32)
            if bool(np.any(zo_of_mat != 0.0)):
                # Zoffs on the centroid key: one float32 subtract
                key = key - zo_of_mat[mat_f0]
        else:
            key = depth
        if bool(np.any(zi_of_mat)):
            # ZInvert: the material's fragments take a negated key, so
            # they sort in front of every ordinary fragment at the
            # pixel, far-first among themselves -- the far wall is
            # composited LAST (on top). The opaque z-test kept the
            # TRUE ordering (`keep` read `depth`, not this key)
            key = np.where(zi_of_mat[mat_f0], -key, key).astype(np.float32)
        if np.unique(zo_of_mat[mat_f0]).size > 1:
            # several offsets = several raster calls concatenated, so the
            # list order among equal-depth fragments would be CALL order:
            # the lower triangle id takes the lower rank (composited
            # later, on top), the raster's own tie rule, named. With ONE
            # offset value the stable sort over the raster's own append
            # order stays (the identity pin needs it)
            order = np.lexsort((tri, key, pix))
        else:
            order = np.lexsort((key, pix))
    pix = pix[order]
    px = px[order]
    py = py[order]
    tri = tri[order]
    bary = bary[order]
    front = front[order]
    depth = depth[order]
    opaque_z = opaque_z[order]

    grp_start = np.zeros(pix.size, np.int64)
    new_group = np.nonzero(pix[1:] != pix[:-1])[0] + 1
    grp_start[new_group] = new_group
    np.maximum.accumulate(grp_start, out=grp_start)
    rank = np.arange(pix.size, dtype=np.int64) - grp_start

    limit = int(getattr(st, 'max_transparent_layers', 0) or 0)
    if limit > 0 and rank.size and int(rank.max()) >= limit:
        within = rank < limit
        # a silent truncation reads as a rendering bug: the dropped
        # layers simply are not drawn, so wherever nothing opaque sits
        # behind them the BACKGROUND shows through -- black holes in
        # the middle of solid-looking geometry. Say it, with the number
        # and the setting that fixes it.
        cut = int((~within).sum())
        hit = int(np.unique(py[~within].astype(np.int64) * gbuf.width
                            + px[~within]).size)
        print(f'[Halcyon] transparency: the {limit}-layer cap dropped '
              f'{cut} fragments at {hit} pixels -- those layers are '
              'not drawn, and where nothing opaque sits behind them '
              'the background shows through. Raise Max Transparent '
              'Layers to draw them')
        px, py, tri, bary, front, rank, pix, depth, opaque_z = (
            a[within] for a in (px, py, tri, bary, front, rank, pix,
                                depth, opaque_z))
        if px.size == 0:
            return fb.leave(img)

    # R251: the per-fragment equation and the GPU-layer refusal by name
    mode_of_mat, _g = _mode_table(job.scene, st)
    if mesh.mat_index is not None:
        mat_f = np.clip(mesh.mat_index[tri], 0, mode_of_mat.size - 1)
    else:
        mat_f = np.zeros(tri.size, np.int32)
    mode = mode_of_mat[mat_f]
    # R251 C101 (Imagine Fog Length): the fog spans need the A-buffer's
    # per-fragment depths -- under Sorted Blend the key is the centroid
    # and front/back pairing is meaningless, so the fog material blends
    # as Alpha Over, said once per frame per material
    a_fog = first_fog = None
    if bool(np.any(mode == 18)):
        if str(getattr(st, 'transparency', 'NONE')) == 'SORTED':
            _mats_f = job.scene.materials or []
            for _mi in sorted(set(int(v) for v in np.unique(mat_f[mode == 18]))):
                _nm = str(getattr(_mats_f[_mi], 'name', None) or f'material {_mi}') \
                    if 0 <= _mi < len(_mats_f) else f'material {_mi}'
                print('[Halcyon] transparency: Imagine Fog Length needs '
                      'A-Buffer (Sorted Blend orders whole polygons by '
                      f"centroid): '{_nm}' blends as Alpha Over")
            mode = np.where(mode == 18, 0, mode).astype(np.int32)
        else:
            a_fog, first_fog = _imagine_fog_spans(
                mode, pix, mat_f, tri, bary, px, py, front, gbuf, job)
    why_cpu = None
    if fb.on:
        why_cpu = (f'the {fb.fmt} framebuffer quantises every layer '
                   'colour to 5/6 bits and the layer shading holds 6e-3 '
                   'across devices (TR:19224-19228); the layers shade on '
                   'the CPU, by name')
    else:
        used = sorted(k for k, v in MODE_INDEX.items()
                      if v in MODES_READ_F and bool(np.any(mode == v)))
        if used:
            why_cpu = (f"the {', '.join(used)} blend equation quantises "
                       'the layer colour to 5/6 bits and the layer '
                       'shading holds 6e-3 across devices '
                       '(TR:19224-19228); the layers shade on the CPU, '
                       'by name')

    # Deferred GPU shading of the layers themselves: each depth layer's
    # fragments become an ids texture and every transparent material's
    # LAYER pass (real alpha out) draws it, exactly the opaque frame's
    # mechanism. Same gate as the opaque pass -- opt-in, whole-frame only
    # -- and any refusal keeps this shading on the CPU, with the reason
    # printed. On the field frame this stage was 25.7 s of 33.7.
    #
    # RANK ROUTING: the driver pays full-frame FIXED costs per layer --
    # a draw and a readback cover every pixel whether the layer holds
    # three fragments or a million -- while the CPU path pays per
    # FRAGMENT. The 1.25.59 field split proved it: skipping absent
    # materials' passes moved nothing, because the cost was never the
    # material count, it was sixteen full-frame round trips. So each
    # layer goes to whichever path is cheaper for IT: layers below
    # `layer_gpu_min_frac` of the frame's pixels shade on the proven
    # per-rank CPU path. Routing is by WHOLE layer -- a rank is never
    # split -- so both paths build their per-rank fields from complete
    # layers and each fragment's colour is exactly what the pure run
    # would have given it. The routing prints with per-layer counts:
    # the field names its own distribution.
    col = None
    LAST_ROUTING.clear()
    if str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
            getattr(st, 'gpu_shading', False) and band is None:
        from ..gpu import shade as _gpu_shade
        counts = np.bincount(rank, minlength=int(rank.max()) + 1)
        frac = float(getattr(st, 'layer_gpu_min_frac', 0.0) or 0.0)
        thresh = max(1, int(round(frac * gbuf.width * gbuf.height))) \
            if frac > 0.0 else 1
        dense = counts >= thresh
        gsel = dense[rank]
        n_gpu_r = int((dense & (counts > 0)).sum())
        n_cpu_r = int((~dense & (counts > 0)).sum())
        LAST_ROUTING.update(
            counts=[int(c) for c in counts], thresh=int(thresh),
            gpu_ranks=n_gpu_r, cpu_ranks=n_cpu_r,
            gpu_frags=int(gsel.sum()),
            cpu_frags=int(rank.size - int(gsel.sum())))
        if why_cpu is not None:
            # R251: a fixed equation or a framebuffer format reads F's
            # rgb through a 5/6-bit lattice -- a refusal BY NAME, the
            # layer counts kept in the record; `col` stays None
            LAST_ROUTING['refused'] = why_cpu
            print('[Halcyon GPU] transparent layers on the CPU: '
                  f'{why_cpu}')
        elif not gsel.any():
            # every layer sits below the break-even: nothing for the
            # driver. That is a ROUTE, not a refusal -- say so calmly,
            # and only when a driver was actually there to be skipped
            from ..gpu import device as _gdev
            ok, _dwhy = _gdev.probe()
            if ok:
                print('[Halcyon GPU] layer routing: all '
                      f'{n_cpu_r} layers below the GPU break-even '
                      f'({thresh} fragments); shaded on the CPU '
                      '(routed, not refused)')
        else:
            with ST.track('transparency shading (GPU)'):
                try:
                    got, why = _gpu_shade.shade_fragments_frame(
                        job, gbuf, tri[gsel], bary[gsel], px[gsel],
                        py[gsel], rank[gsel])
                except Exception as exc:                        # noqa: BLE001
                    got, why = None, str(exc)
            if got is None:
                # the partition was recorded above but never ACTED on:
                # say so in the record, or a "hybrid" that quietly fell
                # back whole would look like a mix to the tests
                LAST_ROUTING['refused'] = str(why)
                print('[Halcyon GPU] transparent layers on the CPU: '
                      f'{why}')
            else:
                col = np.zeros((rank.size, 4), np.float32)
                col[gsel] = got
                cpu_s = 0.0
                if n_cpu_r:
                    import time as _time
                    csel = ~gsel
                    t0 = _time.perf_counter()
                    with ST.track('transparency shading (routed CPU)'):
                        col[csel] = _shade_fragments_cpu(
                            job, tri[csel], bary[csel], px[csel],
                            py[csel], front[csel], rank[csel], st)
                    cpu_s = _time.perf_counter() - t0
                LAST_ROUTING['cpu_s'] = cpu_s
                lt = dict(getattr(_gpu_shade, 'LAST_LAYER_TIMINGS', {})
                          or {})
                if lt:
                    # the split that names the next perf target: where
                    # the layer stage's seconds actually went. Buckets
                    # are disjoint and `other` is the printed remainder
                    # -- a cost the line refuses to hide.
                    wait = max(lt.get('wall_ms', 0.0)
                               - lt.get('exec_ms', 0.0), 0.0)
                    scz = 100.0 * lt.get('scissor_px', 0.0) \
                        / max(lt.get('frame_px', 0.0), 1.0)
                    print('[Halcyon GPU] transparent split: '
                          f"plan {lt.get('plan_ms', 0.0) / 1e3:.1f}s, "
                          'compile '
                          f"{lt.get('compile_ms', 0.0) / 1e3:.1f}s, "
                          'uploads '
                          f"{lt.get('upload_ms', 0.0) / 1e3:.1f}s "
                          f"({lt.get('upload_mb', 0.0):.0f} MB), "
                          f"draws {lt.get('draw_ms', 0.0) / 1e3:.1f}s, "
                          'reads+sync '
                          f"{lt.get('read_ms', 0.0) / 1e3:.1f}s, "
                          f"sweeps {lt.get('sweep_ms', 0.0) / 1e3:.1f}s "
                          '(of which CPU ray build '
                          f"{lt.get('ray_build_ms', 0.0) / 1e3:.1f}s), "
                          f"other {lt.get('other_ms', 0.0) / 1e3:.1f}s "
                          f"of {lt.get('total_ms', 0.0) / 1e3:.1f}s; "
                          f'scissor {scz:.0f}% of full frames; '
                          f"{lt.get('ranks', 0)} of "
                          f'{n_gpu_r + n_cpu_r} layers on the GPU; '
                          f"marshal {lt.get('crossings', 0)} crossings, "
                          f'{wait / 1e3:.1f}s waiting')
                print('[Halcyon GPU] layer routing: GPU '
                      f"{n_gpu_r} layers "
                      f"({_fmt_frags(LAST_ROUTING['gpu_frags'])}), CPU "
                      f"{n_cpu_r} layers "
                      f"({_fmt_frags(LAST_ROUTING['cpu_frags'])}, "
                      f'{cpu_s:.1f}s); per-layer frags '
                      + ' '.join(_fmt_frags(c) for c in counts))
    if col is None:
        with ST.track('transparency shading'):
            col = _shade_fragments_cpu(job, tri, bary, px, py, front,
                                       rank, st)
    # R251 C031 (PS2 AFAIL): a Clip+Blend fragment at or above its Clip
    # Threshold is the OPAQUE half seen again through the transparent
    # pass (same triangle, same depth, kept by the tolerant limit): it
    # was promoted into the z-buffer and shaded there, so it draws
    # nothing here. Decided on the PRE-quantised alpha (alpha_bits 1
    # would round a sub-threshold 0.6 to 1.0 and drop the blend half)
    # A material the clip stage REFUSED by name (affine texturing, an
    # unliftable chain) was never promoted: it stays whole on this road,
    # where the law already forces 1.0 above the threshold as CLIP does
    _cb_refused = LAST_CLIP.get('refused') or {}
    _cb_thr = np.full(mode_of_mat.size, -1.0, np.float32)
    for _i, _m in enumerate(job.scene.materials or []):
        if str(getattr(_m, 'alpha_mode', 'BLEND')) == 'CLIP_BLEND' and \
                str(getattr(_m, 'name', None) or f'material {_i}') \
                not in _cb_refused:
            _cb_thr[_i] = max(float(getattr(_m, 'alpha_clip', 0.5)), 1e-6)
    _cb_f = _cb_thr[mat_f]
    is_opaque_half = (_cb_f >= 0.0) & (col[:, 3] >= _cb_f)
    if st.alpha_bits < 8:
        levels = float(2 ** max(st.alpha_bits, 1) - 1)
        col[:, 3] = np.round(col[:, 3] * levels) / levels

    # group by layer so each one is a contiguous run, then composite the
    # farthest kept layer first (DEPTH) or the first-sorted first (the
    # DS orders)
    layer_order = np.argsort(rank, kind='stable')
    rank_s = rank[layer_order]
    bounds = np.searchsorted(rank_s, np.arange(int(rank_s[-1]) + 2))
    n_at = np.bincount(pix, minlength=W * H)        # kept per pixel, after the cap
    by_depth = order_mode == 'DEPTH'
    depth_write = bool(getattr(st, 'translucent_depth_write', False))
    zw = np.full(W * H, np.inf, np.float32) if depth_write else None
    any_ds = bool(np.any(mode == 15))
    last_id = np.full(W * H, -1, np.int32) if any_ds else None
    if any_ds:
        oid = (mesh.obj_index[tri].astype(np.int32)
               if getattr(mesh, 'obj_index', None) is not None
               else np.zeros(tri.size, np.int32))
    for r in _layer_order(rank_s, st):
        lo, hi = bounds[r], bounds[r + 1]
        if hi <= lo:
            continue
        sel = layer_order[lo:hi]
        yy, xx = py[sel], px[sel]
        pp = pix[sel]
        msel = mode[sel]
        a = np.clip(col[sel, 3], 0.0, 1.0).astype(np.float32)
        if bool(np.any(is_opaque_half[sel])):
            a = np.where(is_opaque_half[sel], np.float32(0.0), a)
        if zw is not None:
            # DS POLYGON_ATTR bit 11: a drawn translucent fragment writes
            # depth; a later one behind it at that pixel is dropped
            # (strict LESS: a tie loses, named). A fragment whose alpha
            # chain resolved to 0 draws nothing and writes nothing.
            ok = depth[sel] < zw[pp]
            a = np.where(ok, a, np.float32(0.0)).astype(np.float32)
            wr = ok & (a > 0.0)
            zw[pp[wr]] = depth[sel][wr]
        if last_id is not None:
            # DS same-ID once: a pixel already holding this translucent
            # polygon ID (Halcyon's object index) is never blended twice
            ds = msel == 15
            skip = ds & (oid[sel] == last_id[pp])
            a = np.where(skip, np.float32(0.0), a).astype(np.float32)
        if msel.max() == 0 and msel.min() == 0:
            # the ALPHA road: 1.89.0's over byte for byte under NONE
            aa = a[:, None]
            if fb.on:
                B = fb.read(img[yy, xx, :3], xx, yy)
                img[yy, xx, :3] = fb.write(fb.over(col[sel, :3], a, B), xx, yy)
            else:
                img[yy, xx, :3] = col[sel, :3] * aa + img[yy, xx, :3] * (1.0 - aa)
            img[yy, xx, 3] = np.clip(img[yy, xx, 3] + aa[:, 0], 0.0, 1.0)
            continue
        last = (rank[sel] == 0) if by_depth else (rank[sel] == n_at[pp] - 1)
        F = col[sel, :3]
        B = fb.read(img[yy, xx, :3], xx, yy)
        out = np.empty_like(F)
        cov = a.copy()
        amax = None
        for m_id in np.unique(msel):
            k = msel == m_id
            if m_id == 0:
                out[k] = fb.over(F[k], a[k], B[k])
            else:
                frag_k = None
                if int(m_id) in (17, 18):
                    # the software items read the fragment's mesh data
                    sk = sel[k]
                    frag_k = dict(tri=tri[sk], bary=bary[sk],
                                  front=front[sk], mat_f=mat_f[sk],
                                  xx=xx[k], yy=yy[k])
                    if a_fog is not None:
                        frag_k['a_fog'] = a_fog[sk]
                        frag_k['first_fog'] = first_fog[sk]
                o, c, am = _fixed_blend(int(m_id), F[k], a[k], B[k], xx[k],
                                        yy[k], last[k], img, fb, job, st, H,
                                        frag=frag_k)
                out[k] = o
                cov[k] = c
                if am is not None:
                    if amax is None:
                        amax = np.full(a.size, -1.0, np.float32)
                    amax[k] = am
        img[yy, xx, :3] = fb.write(out, xx, yy)
        new_a = np.clip(img[yy, xx, 3] + cov, 0.0, 1.0)
        if amax is not None:
            # DS destination alpha = max(Ap, Af)
            kd = amax >= 0.0
            new_a[kd] = np.maximum(img[yy, xx, 3][kd], amax[kd])
        img[yy, xx, 3] = new_a
        if last_id is not None:
            drew = ds & (a > 0.0)
            last_id[pp[drew]] = oid[sel][drew]
    return fb.leave(img)


def edge_factor(gbuf, width=1.0):
    """Screen-space distance to the nearest triangle edge, in pixels.

    Barycentrics fall to zero at the edges, so dividing by their screen-space
    derivative converts them into a width that stays constant however far away
    the triangle is -- the standard barycentric wireframe, done properly rather
    than by thresholding raw barycentrics.

    The derivative is the whole difficulty. A central difference needs the
    pixels on *both* sides to belong to the same triangle, and on anything
    denser than a cube that is rarely true: the derivative collapses to zero,
    the distance goes to infinity, and the wire comes out as a scatter of dots
    with most of it missing. One-sided differences are used instead -- a pixel
    needs only one neighbour along each axis, in either direction -- which is
    the difference between a wireframe and a dotted line.

    A pixel with no same-triangle neighbour on either axis is a triangle about
    one pixel across. There is no derivative to measure and no interior to
    speak of, so it is treated as being *on* the edge rather than infinitely
    far from it. That is what a wireframe of a dense mesh looks like, and the
    alternative was those triangles vanishing entirely.
    """
    b = gbuf.bary
    tri = gbuf.tri

    def one_sided(axis):
        # neighbouring pixels along `axis` that share a triangle
        if axis == 1:
            same = tri[:, 1:] == tri[:, :-1]
            diff = b[:, 1:] - b[:, :-1]
        else:
            same = tri[1:, :] == tri[:-1, :]
            diff = b[1:, :] - b[:-1, :]
        d = np.zeros_like(b)
        have = np.zeros(tri.shape, bool)
        if axis == 1:
            d[:, :-1] = np.where(same[:, :, None], diff, 0.0)   # forward
            have[:, :-1] = same
            back = np.zeros_like(b)
            back[:, 1:] = np.where(same[:, :, None], diff, 0.0)  # backward
            hb = np.zeros(tri.shape, bool)
            hb[:, 1:] = same
        else:
            d[:-1, :] = np.where(same[:, :, None], diff, 0.0)
            have[:-1, :] = same
            back = np.zeros_like(b)
            back[1:, :] = np.where(same[:, :, None], diff, 0.0)
            hb = np.zeros(tri.shape, bool)
            hb[1:, :] = same
        # prefer the forward difference, fall back to the backward one
        out = np.where(have[:, :, None], d, back)
        return out, (have | hb)

    dx, has_x = one_sided(1)
    dy, has_y = one_sided(0)
    fw = np.abs(dx) + np.abs(dy)
    measurable = has_x | has_y
    dist = b / np.maximum(fw, 1e-6)
    dist = dist.min(axis=2)
    return np.where(measurable, dist, 0.0)


def edge_distance_exact(gbuf, mesh, vp, snap=0.0, pixel_shift=0.0):
    """Distance in pixels from each covered pixel to its triangle's nearest edge.

    Computed from the triangle's own projected vertices rather than from a
    finite difference of the barycentrics. The difference matters more than it
    sounds: a derivative needs neighbouring pixels that belong to the same
    triangle, and on a mesh whose triangles are a few pixels across there are
    none -- which is how a wireframe ends up as a scatter of dots, or as
    nothing at all.

    For a point with linear barycentrics l0,l1,l2 in a triangle of screen area
    A, the distance to the edge opposite corner i is |l_i| * 2A / |e_i|. Exact,
    at any triangle size, at any resolution, with no neighbours involved.

    Returns None if it cannot be computed, so the caller can fall back.
    """
    if mesh is None or mesh.tris is None or mesh.verts is None:
        return None
    from . import raster as _raster
    try:
        _clip, screen, _invw, _z = _raster.project(
            np.asarray(mesh.verts, np.float32), vp, gbuf.width, gbuf.height,
            snap=snap, pixel_shift=pixel_shift)
    except Exception:                                           # noqa: BLE001
        return None

    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    out = np.full((gbuf.height, gbuf.width), 1e9, np.float32)
    if py.size == 0:
        return out
    tri = np.asarray(mesh.tris, np.int32)[gbuf.tri[py, px]]
    p0 = screen[tri[:, 0]]
    p1 = screen[tri[:, 1]]
    p2 = screen[tri[:, 2]]
    qx = px.astype(np.float32) + 0.5
    qy = py.astype(np.float32) + 0.5

    area2 = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
             - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    safe = np.where(np.abs(area2) < 1e-9, 1e-9, area2)
    l0 = ((p1[:, 1] - p2[:, 1]) * (qx - p2[:, 0])
          + (p2[:, 0] - p1[:, 0]) * (qy - p2[:, 1])) / safe
    l1 = ((p2[:, 1] - p0[:, 1]) * (qx - p2[:, 0])
          + (p0[:, 0] - p2[:, 0]) * (qy - p2[:, 1])) / safe
    l2 = 1.0 - l0 - l1

    e0 = np.linalg.norm(p1 - p2, axis=1)
    e1 = np.linalg.norm(p2 - p0, axis=1)
    e2 = np.linalg.norm(p0 - p1, axis=1)
    a2 = np.abs(area2)
    d0 = np.abs(l0) * a2 / np.maximum(e0, 1e-6)
    d1 = np.abs(l1) * a2 / np.maximum(e1, 1e-6)
    d2 = np.abs(l2) * a2 / np.maximum(e2, 1e-6)
    d = np.minimum(np.minimum(d0, d1), d2)
    # a triangle straddling the near plane projects a corner to nonsense; those
    # keep the fallback rather than inventing a distance
    bad = ~np.isfinite(d)
    d = np.where(bad, 0.0, d)
    out[py, px] = d.astype(np.float32)
    return out


def marked_edge_distance(gbuf, mesh, vp, snap=0.0, pixel_shift=0.0):
    """R220: distance in pixels from each covered pixel to the nearest
    MARKED edge of its own triangle -- Freestyle marks, Sharp, creases,
    carried per triangle as `MeshData.ink_tri_mask` (bit k = the edge
    opposite corner k). The same exact projected-vertex arithmetic as
    edge_distance_exact, restricted to the marked slots; hidden-line
    removal comes free, because only the z-winning triangle's own pixels
    can ever measure small. Returns (H,W) float32, 1e9 where no marked
    edge is near, or None when the mesh carries no marks."""
    if mesh is None or mesh.tris is None or mesh.verts is None or \
            getattr(mesh, 'ink_tri_mask', None) is None:
        return None
    from . import raster as _raster
    try:
        _clip, screen, _invw, _z = _raster.project(
            np.asarray(mesh.verts, np.float32), vp, gbuf.width, gbuf.height,
            snap=snap, pixel_shift=pixel_shift)
    except Exception:                                           # noqa: BLE001
        return None
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    out = np.full((gbuf.height, gbuf.width), 1e9, np.float32)
    if py.size == 0:
        return out
    tid = gbuf.tri[py, px]
    mbits = np.asarray(mesh.ink_tri_mask, np.uint8)[tid]
    hit = mbits != 0
    if not hit.any():
        return out
    py, px, tid, mbits = py[hit], px[hit], tid[hit], mbits[hit]
    tri = np.asarray(mesh.tris, np.int32)[tid]
    p0 = screen[tri[:, 0]]
    p1 = screen[tri[:, 1]]
    p2 = screen[tri[:, 2]]
    qx = px.astype(np.float32) + 0.5
    qy = py.astype(np.float32) + 0.5
    area2 = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
             - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    safe = np.where(np.abs(area2) < 1e-9, 1e-9, area2)
    l0 = ((p1[:, 1] - p2[:, 1]) * (qx - p2[:, 0])
          + (p2[:, 0] - p1[:, 0]) * (qy - p2[:, 1])) / safe
    l1 = ((p2[:, 1] - p0[:, 1]) * (qx - p2[:, 0])
          + (p0[:, 0] - p2[:, 0]) * (qy - p2[:, 1])) / safe
    l2 = 1.0 - l0 - l1
    a2 = np.abs(area2)
    d0 = np.abs(l0) * a2 / np.maximum(np.linalg.norm(p1 - p2, axis=1), 1e-6)
    d1 = np.abs(l1) * a2 / np.maximum(np.linalg.norm(p2 - p0, axis=1), 1e-6)
    d2 = np.abs(l2) * a2 / np.maximum(np.linalg.norm(p0 - p1, axis=1), 1e-6)
    big = np.float32(1e9)
    d = np.minimum(np.minimum(
        np.where(mbits & 1, d0, big),
        np.where(mbits & 2, d1, big)),
        np.where(mbits & 4, d2, big))
    d = np.where(np.isfinite(d), d, 0.0)
    out[py, px] = d.astype(np.float32)
    return out


def crease_edges(gbuf, mesh, angle_deg=25.0):
    """Pixels on a silhouette or on an edge where the surface turns.

    A wireframe of every triangle edge is what these renderers drew, and on a
    dense mesh at 320x240 it is also a solid fill: once triangles are a couple
    of pixels across, every pixel is within a pixel of an edge. That is
    arithmetic rather than a bug, and no width setting escapes it.

    This is the way out. An edge is kept only where the surface genuinely
    turns -- against the background, against another object, or across a
    face-normal difference wider than `angle_deg`. The result is one pixel
    wide whatever the triangle count behind it.
    """
    cov = gbuf.mask()
    out = np.zeros(cov.shape, bool)
    if not cov.any():
        return out
    tri = gbuf.tri
    fn = getattr(mesh, 'face_normals', None)
    obj = getattr(mesh, 'obj_index', None)
    cos_lim = float(np.cos(np.radians(max(min(angle_deg, 179.0), 0.0))))

    normals = None
    if fn is not None:
        fn = np.asarray(fn, np.float32)
        normals = np.zeros(cov.shape + (3,), np.float32)
        normals[cov] = fn[tri[cov]]

    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(None, -dy or None), slice(None, -dx or None))
        b = (slice(dy, None), slice(dx, None))
        both = cov[a] & cov[b]
        # one side covered and the other not: a silhouette
        edge = cov[a] ^ cov[b]
        diff = both & (tri[a] != tri[b])
        if obj is not None and diff.any():
            oi = np.asarray(obj, np.int32)
            diff_obj = diff & (oi[tri[a]] != oi[tri[b]])
        else:
            diff_obj = np.zeros_like(diff)
        turn = np.zeros_like(diff)
        if normals is not None and diff.any():
            dot = (normals[a] * normals[b]).sum(axis=2)
            turn = diff & (dot < cos_lim)
        hit = edge | diff_obj | turn
        out[a] |= hit
        out[b] |= hit
    return out & cov


def _thicken(mask, radius):
    """Grow a boolean mask by `radius` pixels, so a wire can be more than one."""
    r = int(max(round(radius - 1.0), 0))
    if r <= 0:
        return mask
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def _ink_forced(scene):
    """R220: True when any material says Always Ink -- the outline pass
    then runs even with the global switch off."""
    return any(str(getattr(m, 'ink_mode', 'INHERIT')).upper() == 'ON'
               for m in (getattr(scene, 'materials', None) or ()))


def _scene_has_iro(scene):
    """R229: whether any material inks in its own colour (Iro-Trace)."""
    from . import ink as INK
    return any(m[0] == 'IRO' for m in INK.scene_iro(scene))


def INK_EYE(vp):
    """R234: the camera position from the view-projection, for a
    caller without the eye (core/ink._eye_from_vp)."""
    from . import ink as INK
    return INK._eye_from_vp(vp)


def _paint_reach_rows(st):
    """R230: how many internal rows the paint stages (misregistration,
    bleed) read past a band."""
    from . import film as FILM
    ss = float(max(int(getattr(st, 'aa_samples', 1) or 1), 1)) * 4.0
    return FILM.paint_reach(st, ss)


def _ink_reach_rows(scene, st):
    """R227: how many rows past a band an ink line can reach -- the
    context the band's scissor (and, for From Fill, its shading) must
    keep so a pooled band draws the whole frame's line. R230: the paint
    stages' reach rides on top."""
    from . import ink as INK
    mats = getattr(scene, 'materials', None) or []
    g_width = int(np.clip(getattr(st, 'outline_width', 1), 1, 8))
    widths = [int(np.clip(int(getattr(m, 'ink_width', 0) or 0) or g_width,
                          1, 8)) for m in mats] + [g_width]
    w_max = float(max(widths))
    if INK.styled_on(st, scene):
        ref = int(getattr(st, 'ink_reference_height', 0) or 0)
        rs = 1.0
        if ref > 0:
            rs = float(int(st.resolution_y) * max(int(getattr(
                st, 'aa_samples', 1) or 1), 1) * 4) / float(ref)
            # generous: whatever the internal height turns out to be,
            # a reach a few rows too long only costs a few rows
        h_max = max(w_max * rs - 0.5, 0.5) \
            * (1.0 + float(np.clip(getattr(st, 'ink_taper', 0.0), 0, 1))) \
            * (1.0 + float(np.clip(getattr(st, 'ink_shadow_side', 0.0),
                                   0, 1))) \
            * (1.0 + float(np.clip(getattr(st, 'ink_weight_noise', 0.0),
                                   0, 1))) \
            * (1.0 + float(np.clip(getattr(st, 'ink_roughness', 0.0),
                                   0, 1))) \
            + float(max(getattr(st, 'ink_drift', 0.0), 0.0)) * rs
        # R239: the vertex-colour line control can double a width
        if any(str(getattr(m, 'ink_vc', 'OFF') or 'OFF').upper() != 'OFF'
               for m in mats):
            h_max = h_max * 2.0
        if str(getattr(st, 'ink_texture', 'SOLID')).upper() == 'CHARCOAL':
            h_max = h_max * 1.1 + 1.0
        # R231: a stroke end past the band thins the line inside it over
        # End Length, and the scissor's own cut must not read as an end
        # inside the band: the reach covers the whole taper
        if float(getattr(st, 'ink_end_taper', 0.0)) > 0.0:
            h_max += float(max(getattr(st, 'ink_end_length', 12.0), 1.0)) \
                * rs
        spread = float(max(getattr(st, 'ink_pencil_spread', 1.5), 0.0)) \
            if str(getattr(st, 'ink_style', 'CLEAN')).upper() == 'PENCIL' \
            else 0.0
        # R234: the pressure's boost, the isophote's walk, the stroke
        # road's chain operations (core/lines.py names its rows)
        h_max *= 1.0 + 1.5 * float(np.clip(getattr(st, 'ink_pressure', 0.0),
                                           0, 1))
        reach = h_max + float(max(getattr(st, 'ink_boil', 0.0), 0.0)) \
            + spread + 3.0
        if float(getattr(st, 'ink_isophote', 0.0)) > 0.0:
            reach += float(max(getattr(st, 'ink_isophote_range', 24.0),
                               1.0)) * rs + 2.0
        from . import lines as LN
        reach += LN.road_rows(st, rs) + LN.source_rows(st, rs)
        return int(np.ceil(reach)) + _paint_reach_rows(st)
    # the mask road: a seed dilated w-1 each way, marked ink w wide.
    # R234: the new line sources read past a band on this road too
    from . import lines as LN
    rs_m = 1.0
    ref_m = int(getattr(st, 'ink_reference_height', 0) or 0)
    if ref_m > 0:
        rs_m = float(int(st.resolution_y) * max(int(getattr(
            st, 'aa_samples', 1) or 1), 1) * 4) / float(ref_m)
    return int(w_max) + 1 + int(np.ceil(LN.source_rows(st, rs_m))) \
        + _paint_reach_rows(st)


def ink_tables(scene, st):
    """R249: the per-material ink tables every road reads, read once.

    Per material: ink on (Always, or Inherit under a global line) and
    Never, the width (its own or the global), the colour (its own, the
    Anime Shader's Custom line colour, or the global) and the Iro-Trace
    darken; plus the sky class, which inherits the globals. R233: an
    inheriting Background painting reads as Never. The arrays carry the
    sky class as their last entry; 'iro' is None when no material inks
    in its own colour.
    """
    from . import ink as INK
    mats = getattr(scene, 'materials', None) or []
    opacity = float(np.clip(getattr(st, 'outline_opacity', 1.0), 0.0, 1.0))
    g_width = int(np.clip(getattr(st, 'outline_width', 1), 1, 8))
    g_col = np.asarray(getattr(st, 'outline_color', (0.0, 0.0, 0.0)),
                       np.float32)
    over_sky = bool(getattr(st, 'outline_over_sky', True))
    g_on = bool(getattr(st, 'outline', False))
    modes = [str(getattr(m, 'ink_mode', 'INHERIT') or 'INHERIT').upper()
             for m in mats]
    modes = ['OFF' if (md == 'INHERIT' and str(getattr(
        m, 'paint_mode', 'CEL')).upper() == 'BACKGROUND') else md
        for md, m in zip(modes, mats)]
    m_on = [True if md == 'ON' else False if md == 'OFF' else g_on
            for md in modes]
    m_w = [int(np.clip(int(getattr(m, 'ink_width', 0) or 0) or g_width,
                       1, 8)) for m in mats]
    m_col = [tuple(np.asarray(getattr(m, 'ink_color', (0, 0, 0)),
                              np.float32))
             if getattr(m, 'ink_use_color', False) else tuple(g_col)
             for m in mats]
    # R229: the Anime Shader's Line Colour menu -- CUSTOM is a per-
    # material constant (both roads), IRO the surface's own colour per
    # pixel (the style road; styled_on(st, scene) routes there)
    iro_rows = INK.scene_iro(scene)
    for i_m, (imode, icol, _idk) in enumerate(iro_rows):
        if imode == 'CUSTOM' and icol is not None:
            m_col[i_m] = tuple(np.asarray(icol, np.float32))
    iro_lut = np.array([idk if imode == 'IRO' else -1.0
                        for imode, _c, idk in iro_rows] + [-1.0], np.float32)
    return {'mats': mats, 'opacity': opacity, 'g_width': g_width,
            'g_color': g_col, 'over_sky': over_sky, 'g_on': g_on,
            'modes': modes, 'm_on': m_on, 'm_w': m_w, 'm_col': m_col,
            'iro_rows': iro_rows,
            'on': np.array(m_on + [g_on], bool),
            'off': np.array([md_ == 'OFF' for md_ in modes] + [False], bool),
            'width': np.array(m_w + [g_width], np.float32),
            'color': np.array(m_col + [tuple(g_col)], np.float32),
            'iro': iro_lut if np.any(iro_lut >= 0.0) else None}


def apply_outline(scene, gbuf, img, st, vp=None, proj=None, eye=None):
    """Ink cartoon outlines from the G-buffer, before the post chain.

    The line sources are the buffers the raster already filled -- object
    ids, material ids, depth, face normals -- compared against their
    4-neighbourhood, so the ink lands exactly on visible boundaries and
    creases. It runs at the INTERNAL resolution: a supersampled frame
    anti-aliases its own ink on the way down, which is how the era's
    software got clean cartoon lines without a line renderer.

    R220 grew it two ways, both neutral at defaults. Marked Edges draws
    the edges the ARTIST marked (Freestyle marks, Sharp, creases) as
    interior ink, by exact per-pixel distance to the pixel's own
    triangle's marked edges -- hidden-line removal is the z-buffer's own
    verdict. And every material now has a say: Ink mode (inherit /
    always / never), its own ink colour, its own width. A scene where
    every material inherits takes the pre-R220 single-class road,
    bitwise.

    Computed on the CPU from the same G-buffer on either device, so the
    picture cannot differ between them by construction. R249: a frame
    the GPU shaded draws its line on the GPU too (gpu/ink.py -- the
    same seeds, the same distance transform, the same style maths, as
    fragment passes); the dials that stay on the CPU refuse by name and
    the CPU road below runs exactly as before.
    """
    mesh = scene.mesh
    tri = gbuf.tri
    cov = tri >= 0
    if not cov.any():
        return img
    safe = np.where(cov, tri, 0)
    # R230: the paint before the ink -- misregistration and bleed move
    # and soften the painted colour; the lines below land crisp where
    # the drawing put them. Internal-resolution amplitudes scale by the
    # supersample factor so the dials mean output pixels.
    from . import film as FILM
    from ..gpu import frame as _FR
    if float(getattr(st, 'film_misregister', 0.0)) > 0.0 or \
            float(getattr(st, 'film_bleed', 0.0)) > 1e-3:
        _FR.edited(st, 'painted line', 'misregister / bleed')
        _ss = float(tri.shape[0]) / float(max(int(st.resolution_y), 1))
        img = FILM.misregister_and_bleed(
            img, st, frame=int(getattr(scene, 'frame', 0) or 0),
            seed=int(getattr(st, 'seed', 0) or 0), ss=_ss)
    # the per-material tables (on / Never, width, colour, Iro-Trace)
    # and the globals, read once for every road
    tables = ink_tables(scene, st)
    opacity = tables['opacity']
    g_width = tables['g_width']
    g_col = tables['g_color']
    over_sky = tables['over_sky']
    mats = tables['mats']
    g_on = tables['g_on']
    modes = tables['modes']
    m_on = tables['m_on']
    m_w = tables['m_w']
    m_col = tables['m_col']
    iro_rows = tables['iro_rows']
    from . import ink as INK

    md = None
    if getattr(st, 'outline_marked', False) and vp is not None and \
            getattr(mesh, 'ink_tri_mask', None) is not None:
        md = marked_edge_distance(gbuf, mesh, vp, snap=snap_grid(st),
                                  pixel_shift=raster.pixel_shift_of(st))

    if str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
            getattr(st, '_frame_gpu_shaded', False):
        # R249: the GPU road -- the frame was shaded on the GPU, so the
        # device is live and the G-buffer is already up there; the line
        # is drawn by fragment passes and the CPU never extracts a seed.
        # Every refusal is a reason, printed once, and the CPU road runs
        from ..gpu import ink as GINK
        _keep = {}
        with ST.track('outline (GPU)'):
            try:
                # R250: the frame the shading left on the GPU is read in
                # place (no upload); the ink's output target becomes the
                # frame that stays there for the resolve and the post
                got, why = GINK.apply(scene, gbuf, img, st, tables, md=md,
                                      vp=vp, proj=proj, eye=eye,
                                      frame=_FR.current(st), keep=_keep)
            except Exception as exc:                            # noqa: BLE001
                got, why = None, f'{type(exc).__name__}: {exc}'
        if got is not None:
            _out_t = _keep.get('out')
            if _out_t is not None:
                _FR.install(st, _FR.Resident(_out_t, gbuf.width, gbuf.height,
                                             'ink'), 'ink')
            return got
        GINK._warn(str(why))
        _FR.edited(st, 'outline', 'the line drew on the CPU')
        try:
            from .. import fault_note
            fault_note(f'GPU ink refused: {why}'[:220], key='ink-no', limit=3)
        except Exception:                                       # noqa: BLE001
            pass

    # R227: the seeds keep their KIND -- silhouette (object, depth and
    # sky boundaries) apart from interior (material breaks, creases) --
    # for the style road's thick-outer / thin-inner line; the mask roads
    # read their union, bitwise what they always read
    edge_sil = np.zeros(tri.shape, bool)
    edge_int = np.zeros(tri.shape, bool)

    def neigh_diff(plane, differs):
        """OR `differs(a, b)` over the 4-neighbourhood into `edge`."""
        e = np.zeros(tri.shape, bool)
        e[:, 1:] |= differs(plane[:, 1:], plane[:, :-1])
        e[:, :-1] |= differs(plane[:, :-1], plane[:, 1:])
        e[1:, :] |= differs(plane[1:, :], plane[:-1, :])
        e[:-1, :] |= differs(plane[:-1, :], plane[1:, :])
        return e

    dmap = np.where(cov, gbuf.depth, 1e12).astype(np.float32)
    if getattr(st, 'outline_objects', True) and \
            mesh.obj_index is not None:
        omap = np.where(cov, mesh.obj_index[safe], -1)
        edge_sil |= neigh_diff(omap, lambda a, b: a != b)
    if getattr(st, 'outline_materials', False) and \
            mesh.mat_index is not None:
        mmap = np.where(cov, mesh.mat_index[safe], -1)
        edge_int |= neigh_diff(mmap, lambda a, b: a != b)
    if getattr(st, 'outline_depth', True):
        thr = max(float(getattr(st, 'outline_depth_threshold', 0.02)), 1e-5)

        def depth_break(a, b):
            near = np.minimum(np.abs(a), np.abs(b))
            return (a < 1e11) & (a < b) & \
                ((b - a) > thr * np.maximum(near, 1e-4))

        edge_sil |= neigh_diff(dmap, depth_break)
    if getattr(st, 'outline_normals', True) and \
            mesh.face_normals is not None:
        fn = mesh.face_normals[safe]
        fn = np.where(cov[:, :, None], fn, 0.0).astype(np.float32)
        cos_lim = np.float32(np.cos(np.radians(
            float(getattr(st, 'outline_normal_angle', 60.0)))))
        # R249: the same test, four times cheaper -- the dot as three
        # channel products summed in NumPy's own order ((x + y) + z, the
        # order .sum(-1) reduces three elements in), the non-zero test
        # computed once per pixel instead of once per neighbour pair.
        # 193 ms of a 720p frame's 514 ms outline bucket were this
        # reduction over a three-wide last axis
        fx, fy, fz = fn[:, :, 0], fn[:, :, 1], fn[:, :, 2]
        nz = (np.abs(fx) + np.abs(fy) + np.abs(fz)) > 0

        def crease_planes(sl_a, sl_b):
            d = (fx[sl_a] * fx[sl_b] + fy[sl_a] * fy[sl_b]) \
                + fz[sl_a] * fz[sl_b]
            return (d < cos_lim) & nz[sl_a] & nz[sl_b]

        e_n = np.zeros(tri.shape, bool)
        e_n[:, 1:] |= crease_planes((slice(None), slice(1, None)),
                                    (slice(None), slice(None, -1)))
        e_n[:, :-1] |= crease_planes((slice(None), slice(None, -1)),
                                     (slice(None), slice(1, None)))
        e_n[1:, :] |= crease_planes((slice(1, None), slice(None)),
                                    (slice(None, -1), slice(None)))
        e_n[:-1, :] |= crease_planes((slice(None, -1), slice(None)),
                                     (slice(1, None), slice(None)))
        edge_int |= e_n
    # R234: three more interior sources (core/lines.py) -- form lines
    # (the valleys of the facing), shadow lines (the terminator at the
    # Shadow Level) and tone lines (flow-guided DoG of the shaded
    # frame). Each keeps off the silhouette's own neighbourhood; each
    # is a plain seed class, so both roads draw it
    want_form = bool(getattr(st, 'outline_form', False))
    want_shadow = bool(getattr(st, 'outline_shadow', False))
    want_tone = bool(getattr(st, 'outline_tone', False))
    if want_form or want_shadow or want_tone:
        from . import lines as LN
        _rs = 1.0
        _ref = int(getattr(st, 'ink_reference_height', 0) or 0)
        if _ref > 0:
            _rs = float(tri.shape[0]) / float(_ref)
        _N = LN.surface_normals(mesh, gbuf, cov)
        _P = LN.surface_points(mesh, gbuf, cov)
        _omap = np.where(cov, mesh.obj_index[safe], -1) \
            if mesh.obj_index is not None else np.where(cov, 0, -1)
        if want_form:
            _eye = np.asarray(eye, np.float32) if eye is not None else \
                INK_EYE(vp)
            _ndv = LN.ndv_field(mesh, gbuf, cov, _eye, N=_N, P=_P) \
                .reshape(tri.shape)
            _thr = float(max(getattr(st, 'outline_form_threshold', 0.3),
                             0.0)) * 0.01 / (_rs * _rs)
            _sig = 2.0 * _rs
            edge_int |= LN.form_seeds(
                _ndv, cov, LN.dilate8(edge_sil, int(np.ceil(_sig))),
                _sig, _thr, omap=_omap)
        if want_shadow:
            _ndl = LN.ndl_field(scene, mesh, gbuf, cov, N=_N, P=_P) \
                .reshape(tri.shape)
            _lvl = float(np.clip(getattr(st, 'outline_shadow_level', 0.1),
                                 -1.0, 1.0))
            edge_int |= LN.shadow_seeds(_ndl, cov, _omap, _lvl,
                                        LN.dilate8(edge_sil, 1))
        if want_tone:
            _tthr = float(max(getattr(st, 'outline_tone_threshold', 0.15),
                              0.0)) * 0.11
            edge_int |= LN.tone_seeds(img, cov,
                                      LN.dilate8(edge_sil, max(int(_rs), 1)),
                                      1.2 * _rs, _tthr, omap=_omap)
    edge = edge_sil | edge_int
    if not getattr(st, 'outline_over_sky', True):
        edge &= cov

    def _dilate(mask, steps):
        for _ in range(steps):
            grown = mask.copy()
            grown[:, 1:] |= mask[:, :-1]
            grown[:, :-1] |= mask[:, 1:]
            grown[1:, :] |= mask[:-1, :]
            grown[:-1, :] |= mask[1:, :]
            mask = grown
        return mask

    if INK.styled_on(st, scene):
        # R227: the style road -- distance fields, planes of width and
        # colour, the brush / pencil / boil / gradient dials. Opt-in by
        # any dial (R229: or an Iro-Trace material); the mask roads
        # below are untouched
        n_m = len(mats)
        pmat = np.where(cov, mesh.mat_index[safe]
                        if mesh.mat_index is not None else 0, n_m)
        pmat = np.clip(pmat, 0, n_m)
        on_lut = np.array(m_on + [g_on], bool)
        on_plane = on_lut[pmat]
        off_lut = np.array([md_ == 'OFF' for md_ in modes] + [False], bool)
        # the silhouette seed is ONE-SIDED: the nearer pixel of every
        # boundary (the object in front owns its line; the sky never
        # seeds), so the distance field's feature is always a surface
        # pixel with a material
        sil_seed = np.zeros(tri.shape, bool)
        sil_own = edge_sil & cov & on_plane
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = (slice(max(dy, 0), tri.shape[0] + min(dy, 0)),
                 slice(max(dx, 0), tri.shape[1] + min(dx, 0)))
            b = (slice(max(-dy, 0), tri.shape[0] + min(-dy, 0)),
                 slice(max(-dx, 0), tri.shape[1] + min(-dx, 0)))
            nearer = dmap[a] <= dmap[b]
            sil_seed[a] |= sil_own[a] & (nearer | ~cov[b])
        int_seed = edge_int & cov & on_plane
        md_on = None
        if md is not None:
            md_on = np.where(on_plane, md, np.float32(1e9))
        # R239: the Guilty Gear line control -- ASW's own convention on
        # the mesh's vertex colours, read at the seed's OWN pixel (the
        # pixel that owns the line): ALPHA times the width (0.5 the
        # width as set, 0 erases the seed outright so the neighbouring
        # line flows past, 1 doubles); BLUE holds interior and marked
        # lines back until the surface turns toward its silhouette
        # (their hull's depth push, read as facing) -- silhouette seeds
        # are exempt, exactly as a pushed-back hull still rims the
        # silhouette. A mesh without a colour layer controls nothing.
        vc_a = None
        vc_lut = np.array(
            [str(getattr(m, 'ink_vc', 'OFF') or 'OFF').upper() == 'ARCSYS'
             for m in mats] + [False])
        if vc_lut.any() and getattr(mesh, 'colors', None) is not None:
            on_vc = vc_lut[pmat] & cov
            if on_vc.any():
                HH, WW = cov.shape
                vidx = np.nonzero(on_vc.reshape(-1))[0]
                t_vc = gbuf.tri.reshape(-1)[vidx]
                t_vc = np.where(t_vc >= 0, t_vc, 0)
                tv_vc = mesh.tris[t_vc]
                b_vc = gbuf.bary.reshape(-1, 3)[vidx]
                C_vc = np.asarray(mesh.colors, np.float32)
                a_v = (C_vc[tv_vc[:, 0], 3] * b_vc[:, 0]
                       + C_vc[tv_vc[:, 1], 3] * b_vc[:, 1]
                       + C_vc[tv_vc[:, 2], 3] * b_vc[:, 2])
                blu_v = (C_vc[tv_vc[:, 0], 2] * b_vc[:, 0]
                         + C_vc[tv_vc[:, 1], 2] * b_vc[:, 1]
                         + C_vc[tv_vc[:, 2], 2] * b_vc[:, 2])
                vc_a = np.ones(HH * WW, np.float32)
                vc_a[vidx] = np.clip(a_v, 0.0, 1.0) * np.float32(2.0)
                erase = np.zeros(HH * WW, bool)
                erase[vidx] = a_v <= np.float32(0.05)
                er2 = erase.reshape(HH, WW)
                sil_seed &= ~er2
                int_seed &= ~er2
                if md_on is not None:
                    md_on = np.where(er2, np.float32(1e9), md_on)
                held = vidx[blu_v > np.float32(1e-3)]
                if held.size and eye is not None:
                    from . import lines as _LN2
                    P_h, N_h = _LN2.surface_attrs(mesh, gbuf, held)
                    V_h = np.asarray(eye, np.float32)[None, :] - P_h
                    lnv = np.sqrt((V_h * V_h).sum(1))
                    V_h = V_h / np.maximum(lnv, np.float32(1e-9))[:, None]
                    ndv_h = np.abs((N_h * V_h).sum(1))
                    hb = np.clip(blu_v[blu_v > np.float32(1e-3)], 0.0, 1.0)
                    gate = np.zeros(HH * WW, bool)
                    gate[held] = ndv_h > (np.float32(1.0) - hb)
                    g2 = gate.reshape(HH, WW)
                    int_seed &= ~g2
                    if md_on is not None:
                        md_on = np.where(g2, np.float32(1e9), md_on)
        iro_lut = np.array([idk if imode == 'IRO' else -1.0
                            for imode, _c, idk in iro_rows] + [-1.0],
                           np.float32)
        plane = {'cov': cov, 'pmat': pmat, 'on': on_lut,
                 'off': off_lut[pmat],
                 'iro': iro_lut if np.any(iro_lut >= 0.0) else None,
                 'vc_a': vc_a,
                 'width': np.array(m_w + [g_width], np.float32),
                 'color': np.array(m_col + [tuple(g_col)], np.float32),
                 'g_color': g_col, 'opacity': opacity, 'over_sky': over_sky}
        return INK.apply(scene, gbuf, img, st, (sil_seed, int_seed, md_on),
                         plane, vp=vp, proj=proj, eye=eye)

    plain = md is None and all(
        md_ == 'INHERIT' for md_ in modes) and not any(
        getattr(m, 'ink_use_color', False) for m in mats) and not any(
        int(getattr(m, 'ink_width', 0) or 0) for m in mats) and not any(
        imode == 'CUSTOM' for imode, _c, _d in iro_rows)
    if plain:
        # the pre-R220 single-class road, verbatim -- every material
        # inherits and no marked ink is in play, so the classed walk
        # below would reproduce these exact operations anyway
        edge = _dilate(edge, g_width - 1)
        if opacity <= 0.0 or not edge.any():
            return img
        m = edge & (cov if not over_sky else np.ones_like(edge))
        img[m, :3] = img[m, :3] * (1.0 - opacity) + g_col[None, :] * opacity
        img[m, 3] = np.maximum(img[m, 3], opacity if over_sky else
                               img[m, 3])
        return img

    if opacity <= 0.0:
        return img
    # ---- the classed road: each pixel's WINNING material decides its
    # ink (on/off, width, colour); the sky class inherits the globals
    n_m = len(mats)
    pmat = np.where(cov, mesh.mat_index[safe]
                    if mesh.mat_index is not None else 0, n_m)
    on_lut = np.array(m_on + [g_on], bool)
    w_lut = np.array(m_w + [g_width], np.int32)
    off_lut = np.array([md_ == 'OFF' for md_ in modes] + [False], bool)
    col_lut = np.array(m_col + [tuple(g_col)], np.float32)
    pmat = np.clip(pmat, 0, n_m)
    off_plane = off_lut[pmat]
    # R229: a SILHOUETTE seed pair belongs to its NEARER side -- the
    # surface the line is drawn around owns both pixels' class (width
    # and colour), exactly as the style road's one-sided seed does.
    # Until now the far side of every silhouette (the sky, the floor
    # behind a ball) seeded ITS OWN class, so a per-material coloured
    # or widened line was half the global line on its outer side.
    # Interior seeds (material breaks, creases) keep their own pixel's
    # class: each material inks its side of a shared edge.
    owner = pmat
    if edge_sil.any():
        best = dmap.copy()
        own = pmat.copy()
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            a = (slice(max(dy, 0), dmap.shape[0] + min(dy, 0)),
                 slice(max(dx, 0), dmap.shape[1] + min(dx, 0)))
            b = (slice(max(-dy, 0), dmap.shape[0] + min(-dy, 0)),
                 slice(max(-dx, 0), dmap.shape[1] + min(-dx, 0)))
            nearer = dmap[b] < best[a]
            best[a] = np.where(nearer, dmap[b], best[a])
            own[a] = np.where(nearer, pmat[b], own[a])
        owner = np.where(edge_sil, own, pmat)

    paint = np.zeros(edge.shape, bool)
    col_plane = np.zeros(edge.shape + (3,), np.float32)
    classes = {}
    for i in range(n_m + 1):
        if not on_lut[i]:
            continue
        classes.setdefault((int(w_lut[i]), tuple(col_lut[i])),
                           []).append(i)
    for (w_c, col_c), ids in sorted(classes.items()):
        in_class = np.isin(owner, ids)
        seed = edge & in_class
        mask_c = _dilate(seed, w_c - 1) if seed.any() else \
            np.zeros_like(seed)
        if md is not None:
            # marked interior ink: already width-shaped by distance,
            # matched to the dilation convention (w -> 2w-1 pixels)
            mask_c |= in_class & (md < np.float32(w_c - 0.5))
        if not mask_c.any():
            continue
        mask_c &= ~off_plane
        if not over_sky:
            mask_c &= cov
        paint |= mask_c
        col_plane[mask_c] = np.asarray(col_c, np.float32)
    if not paint.any():
        return img
    img[paint, :3] = img[paint, :3] * (1.0 - opacity) \
        + col_plane[paint] * opacity
    img[paint, 3] = np.maximum(img[paint, 3], opacity if over_sky else
                               img[paint, 3])
    return img


def _wire_active(scene, st, gbuf):
    """Whether apply_wireframe will edit the frame (its own early exits)."""
    try:
        if not gbuf.mask().any():
            return False
        if getattr(st, 'render_wire', False):
            return True
        return any(material_model(m, st) == 'WIREFRAME'
                   for m in scene.materials)
    except Exception:                                           # noqa: BLE001
        return True


def _beams_active(scene, st):
    """Whether _light_volumes will edit the frame (its own early exits)."""
    if not getattr(st, 'spot_cones', False):
        return False
    return any(str(getattr(l, 'type', '')).upper() in _VOLUME_KINDS
               and float(getattr(l, 'volumetric', 0.0)) > 0.0
               for l in (getattr(scene, 'lights', None) or ()))


def _propagate_gpu_flag(st, flags, what):
    """R250: a frame stitched from sub-frames (panorama strips, stereo
    eyes, accumulation passes) carries the post chain's device truth
    only if EVERY sub-frame shaded on the GPU; a frame that did not says
    so once instead of running its post on the CPU in silence."""
    ok = bool(flags) and all(bool(f) for f in flags)
    st._frame_gpu_shaded = ok
    if not ok and str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU':
        n_cpu = sum(1 for f in flags if not f)
        _gpu_flag_warn(f'post on the CPU: {n_cpu} of {len(flags)} {what} '
                       'shaded on the CPU')


_GPU_FLAG_WARNED = set()


def _gpu_flag_warn(msg):
    if msg not in _GPU_FLAG_WARNED:
        _GPU_FLAG_WARNED.add(msg)
        print(f'[Halcyon GPU] {msg}')


def apply_wireframe(job, gbuf, img, st, vp, eye, textures):
    """Draw the Wireframe shading model, and the global wire overlay."""
    scene = job.scene
    mesh = scene.mesh
    cov = gbuf.mask()
    if not cov.any():
        return img
    wire_mats = {i for i, m in enumerate(scene.materials)
                 if material_model(m, st) == 'WIREFRAME'}
    if not wire_mats and not st.render_wire:
        # Nothing to carve. If a *surface* nonetheless shaded as Wireframe the
        # user is looking at a flat unlit fill and has no way to know why, so
        # the mismatch is recorded rather than left silent.
        scene.wireframe_note = None
        return img
    scene.wireframe_note = sorted(wire_mats)
    overlay = bool(st.render_wire)
    # R251 (RAST-B): the two wire roads that replace the depth-tested
    # overlay -- C063 Elite's rule (either face front, no depth test, a
    # dot past wire_dot_distance) and C055 the vector monitor beam (a
    # phosphor stroke per feature edge, additive, no depth test). Both
    # draw on the CPU on either device: the wire road released the
    # resident frame by name ('wireframe') before this call. A material
    # shading as WIREFRAME still carves below, without the overlay.
    if st.render_wire and str(getattr(st, 'wire_mode', 'ALL')) == 'ELITE':
        img = raster.draw_elite_wire(img, mesh, vp, eye, st, snap=snap_grid(st))
        overlay = False
    if st.render_wire and str(getattr(st, 'wire_mode', 'ALL')) == 'BEAM':
        img = raster.draw_beam_wire(img, mesh, job.view, vp, eye, st,
                                    rh=gbuf.height, snap=snap_grid(st))
        overlay = False
    if not wire_mats and not overlay:
        return img
    crease = None
    if str(getattr(st, 'wire_mode', 'ALL')) == 'CREASE':
        crease = crease_edges(gbuf, mesh, float(getattr(st, 'wire_angle', 25.0)))
        dist = None
    else:
        dist = edge_distance_exact(gbuf, mesh, vp, snap=snap_grid(st),
                                   pixel_shift=raster.pixel_shift_of(st))
        if dist is None:
            dist = edge_factor(gbuf)
    py, px = np.nonzero(cov)
    d = dist[py, px] if dist is not None else None
    if overlay:
        w = max(float(st.wire_width), 0.1)
        on = (d < w) if d is not None else \
            _thicken(crease, w)[py, px]
        col = np.asarray(st.wire_color, np.float32)
        yy, xx = py[on], px[on]
        img[yy, xx, :3] = col[None, :]
        img[yy, xx, 3] = 1.0
        if not getattr(st, '_viewport', False) and yy.size:
            # the overlay says so when it draws: at high resolutions a
            # one-pixel wire is a FAINT line, and the field spent two
            # rounds hunting one with the checkbox quietly on
            print(f'[Halcyon] wireframe overlay: inked {yy.size} pixels '
                  f'({st.wire_mode}, width {w:g}) -- the Wireframe '
                  f"panel's Wireframe Overlay checkbox turns it off")
    if wire_mats:
        mat_idx = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
            else np.zeros(py.size, np.int32)
        is_wire = np.isin(mat_idx, list(wire_mats))
        if is_wire.any():
            widths = np.array([material_wire_size(m)
                               for m in scene.materials] or [1.0], np.float32)
            w = widths[np.clip(mat_idx, 0, widths.size - 1)]
            if d is not None:
                off = is_wire & (d >= w)
            else:
                thick = _thicken(crease, float(np.max(w)))
                off = is_wire & ~thick[py, px]
            if off.any():
                yy, xx = py[off], px[off]
                # these pixels see straight through the surface, so they get the
                # world behind them -- shaded here because the background pass
                # deliberately skipped every covered pixel
                clear = np.zeros((gbuf.height, gbuf.width), bool)
                clear[yy, xx] = True
                behind = _background_image(scene, st, gbuf.width, gbuf.height,
                                           vp, eye, clear, textures,
                                           force_world=True)
                img[yy, xx, :3] = behind[yy, xx, :3]
                img[yy, xx, 3] = 0.0 if getattr(st, 'film_transparent', False) \
                    else 1.0
    return img


def _debug_pass(job, gbuf, img, st):
    mesh = job.scene.mesh
    h, w = gbuf.height, gbuf.width
    out = np.zeros((h, w, 4), np.float32)
    out[:, :, 3] = 1.0
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    mode = st.debug_pass
    if mode == 'OVERDRAW':
        od = gbuf.overdraw.astype(np.float32)
        lo, hi = float(od.min()), float(od.max())
        if hi <= lo:
            # uniform coverage: show the count rather than one flat colour
            out[:, :, :3] = _heat(np.full_like(od, 0.0 if hi <= 0 else 0.5))[:, :, :3]
            return out
        out[:, :, :3] = _heat((od - lo) / (hi - lo))[:, :, :3]
        return out
    if py.size == 0:
        return out
    tri = gbuf.tri[py, px]
    bary = gbuf.bary[py, px]
    if mode == 'DEPTH':
        d = gbuf.depth[py, px]
        finite = np.isfinite(d)
        lo, hi = (d[finite].min(), d[finite].max()) if finite.any() else (0.0, 1.0)
        v = 1.0 - np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[py, px, :3] = v[:, None]
    elif mode == 'NORMAL':
        ctx = job.context(tri, bary, px, py)
        out[py, px, :3] = ctx.N * 0.5 + 0.5
    elif mode == 'UV':
        ctx = job.context(tri, bary, px, py)
        out[py, px, 0] = ctx.uv[:, 0] % 1.0
        out[py, px, 1] = ctx.uv[:, 1] % 1.0
    elif mode == 'MATID':
        mi = mesh.mat_index[tri] if mesh.mat_index is not None else tri * 0
        out[py, px, :3] = _idcolor(mi)
    elif mode == 'WIREFRAME':
        edge = (bary.min(axis=1) < 0.02).astype(np.float32)
        out[py, px, :3] = edge[:, None]
    else:
        return img
    return out


def linear_depth(job, gbuf, eye):
    """Distance from the camera, in scene units, with NaN where nothing was hit.

    `gbuf.depth` is normalised device depth: it runs 0..1 and crowds almost
    everything into the last few thousandths, which is exactly what a depth
    buffer is for and exactly wrong for anything that wants a distance. A Z
    pass in NDC is unusable in a comp, and a depth-of-field focus expressed in
    metres compared against it is not comparing anything.
    """
    h, w = gbuf.height, gbuf.width
    out = np.full((h, w), np.nan, np.float32)
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    if py.size == 0:
        return out
    ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py)
    out[py, px] = np.linalg.norm(ctx.P - np.asarray(eye, np.float32)[None, :],
                                 axis=1)
    return out


def wanted_passes(st):
    """Which extra passes this render is being asked for.

    Written out one setting at a time rather than looped over a table of
    strings, so that a reader -- and the test that hunts for controls nothing
    reads -- can see each one actually being used.
    """
    names = []
    if getattr(st, 'pass_depth', False):
        names.append('Depth')
    if getattr(st, 'pass_normal', False):
        names.append('Normal')
    if getattr(st, 'pass_position', False):
        names.append('Position')
    if getattr(st, 'pass_uv', False):
        names.append('UV')
    if getattr(st, 'pass_object_index', False):
        names.append('IndexOB')
    if getattr(st, 'pass_material_index', False):
        names.append('IndexMA')
    # R253: the frame passes (G-buffer / frame by-products, both roads
    # bitwise by construction) and the BI light split (light_surface's
    # own side accumulators; the GPU plan refuses them by name)
    if getattr(st, 'pass_mist', False):
        names.append('Mist')
    if getattr(st, 'pass_environment', False):
        names.append('Env')
    if getattr(st, 'pass_beauty', False):
        names.append('Beauty')
    if getattr(st, 'pass_diffuse', False):
        names.append('Diffuse')
    if getattr(st, 'pass_specular', False):
        names.append('Spec')
    if getattr(st, 'pass_ambient', False):
        names.append('Ambient')
    if getattr(st, 'pass_emission', False):
        names.append('Emit')
    if getattr(st, 'pass_shadow', False):
        names.append('Shadow')
    if getattr(st, 'pass_ao', False):
        names.append('AO')
    if getattr(st, 'pass_color', False):
        names.append('Color')
    if getattr(st, 'pass_lights', False):
        names.extend(f'Light{i:02d}' for i in range(LIGHT_PASS_SLOTS))
    return tuple(names)


# R253: the BI light split -- Blender Internal 2.79's RenderResult names
# (render_result.c RE_PASSNAME_DIFFUSE 'Diffuse', _SPEC 'Spec', _SHADOW,
# _AO, _EMIT, _RGBA 'Color'; Ambient is Halcyon's own, BI folded it into
# Combined) -- every one read off light_surface's own accumulators, never
# the frame's arithmetic
LIGHT_SPLIT_PASSES = ('Diffuse', 'Spec', 'Ambient', 'Emit', 'Shadow', 'AO',
                      'Color')
#: fixed so update_render_passes needs no scene knowledge: Light00..Light07
#: in lamp order, the first eight selected lamps
LIGHT_PASS_SLOTS = 8
#: passes that are DATA (top-left sample under supersampling, never
#: filtered); everything else wanted is COLOUR and resolves through the
#: CPU's own AA kernel exactly as the beauty does
DATA_PASSES = ('Depth', 'Normal', 'Position', 'UV', 'IndexOB', 'IndexMA',
               'Mist')


def light_pass_names(st):
    """The wanted light-component passes (split names + LightNN), in
    wanted order -- the ONE predicate render(), the GPU plan's refusal,
    the capability table and the engine share. Empty when none is on."""
    return tuple(n for n in wanted_passes(st)
                 if n in LIGHT_SPLIT_PASSES or n.startswith('Light'))


def build_aux_passes(job, gbuf, st, depth_m=None, env_rgb=None, sink=None,
                     ss=1, out_wh=None):
    """Raw data buffers for the compositor, straight off the G-buffer.

    These are *data*, so nothing here is display-transformed, dithered or
    filtered: a depth in metres stays a depth in metres, and an object index
    stays an integer. Uncovered pixels get the conventions Blender's own
    engines use -- 1e10 for depth, zero for everything else -- so a Z pass
    composites the same way a Cycles one does.

    R253: `env_rgb` is the frame's rgb at uncovered pixels (the Env pass,
    snapshotted by render() before any composite edits the frame), `sink`
    the ShadeJob's light-split sink (pass name -> (rh, rw, 3) at RENDER
    resolution), `ss` / `out_wh` the supersample factor and the output
    size. The COLOUR passes (Env and the split) are resolved here through
    _resolve -- the CPU's own AA kernel, the same one the beauty takes --
    so Diffuse + Spec + Ambient + Emit keeps summing to the beauty after
    the filter; the DATA passes are returned at render resolution and
    render() takes their top-left sample. 'Beauty' is the engine's (the
    frame it hands to post), never built here.
    """
    names = wanted_passes(st)
    if not names:
        return {}
    mesh = job.scene.mesh
    h, w = gbuf.height, gbuf.width
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    out = {}
    ctx = None
    if py.size and ({'Normal', 'Position', 'UV'} & set(names)):
        ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py)
    ss = max(int(ss or 1), 1)
    if out_wh is None:
        out_wh = (w // ss, h // ss)
    W_o, H_o = int(out_wh[0]), int(out_wh[1])

    def _colour(buf3):
        # a colour pass takes the beauty's own filter (a 4-channel
        # shape is what _resolve reads; the alpha lane is discarded)
        buf4 = np.concatenate(
            [np.asarray(buf3, np.float32),
             np.ones((buf3.shape[0], buf3.shape[1], 1), np.float32)],
            axis=2)
        return np.ascontiguousarray(
            _resolve(buf4, W_o, H_o, ss, st)[:, :, :3]).astype(np.float32)

    for name in names:
        if name == 'Beauty':
            continue                      # the engine's: the frame itself
        if name == 'Depth':
            d = depth_m if depth_m is not None else linear_depth(
                job, gbuf, getattr(job, 'eye', (0.0, 0.0, 0.0)))
            d = np.where(np.isfinite(d), d, 1e10)
            out[name] = d[:, :, None].astype(np.float32)
            continue
        if name == 'Mist':
            # R253: BI's mist pass (shadeoutput.c mistfactor) on the scene
            # Fog curve -- core/fog.legacy_curve, the same LINEAR / EXP /
            # EXP2 / TABLE16 curves the fog reads, on the camera DISTANCE
            # (legacy_import maps BI mist onto exactly these dials). It
            # reads the curve, not the Fog toggle: 0 at the near edge,
            # 1 at the sky, whether or not fog is drawn
            from . import fog as _FOGP
            d = depth_m if depth_m is not None else linear_depth(
                job, gbuf, getattr(job, 'eye', (0.0, 0.0, 0.0)))
            fin = np.isfinite(d)
            d32 = np.where(fin, d, 0.0).astype(np.float32)
            with np.errstate(all='ignore'):
                curve = np.asarray(_FOGP.legacy_curve(d32, st), np.float32)
            mist = np.clip(np.float32(1.0) - curve, 0.0, 1.0)
            mist = np.where(fin, mist, np.float32(1.0)).astype(np.float32)
            out[name] = mist[:, :, None]
            continue
        if name == 'Env':
            # R253: the world where the camera saw it -- the frame at
            # uncovered pixels (the sky pass's own pixels on either road),
            # black under geometry. A COLOUR pass: filtered like the beauty
            if env_rgb is None:
                env_rgb = np.zeros((h, w, 3), np.float32)
            out[name] = _colour(np.where(cov[:, :, None], np.float32(0.0),
                                         np.asarray(env_rgb, np.float32)))
            continue
        if name in LIGHT_SPLIT_PASSES or name.startswith('Light'):
            # R253: the light split, read from the ShadeJob's sink (filled
            # by shade_batch at every camera fragment; absent on a road
            # that allocated none -- a band, the viewport)
            if sink is not None and name in sink:
                out[name] = _colour(sink[name])
            continue
        chans = 1 if name in ('IndexOB', 'IndexMA') else 3
        buf = np.zeros((h, w, chans), np.float32)
        if py.size:
            if name == 'Normal':
                buf[py, px] = ctx.N
            elif name == 'Position':
                buf[py, px] = ctx.P
            elif name == 'UV':
                buf[py, px, 0] = ctx.uv[:, 0]
                buf[py, px, 1] = ctx.uv[:, 1]
                buf[py, px, 2] = 1.0
            elif name == 'IndexMA':
                mi = (mesh.mat_index[gbuf.tri[py, px]]
                      if mesh.mat_index is not None else np.zeros(py.size))
                buf[py, px, 0] = np.asarray(mi, np.float32)
            elif name == 'IndexOB':
                oi = (mesh.obj_index[gbuf.tri[py, px]]
                      if mesh.obj_index is not None else np.zeros(py.size))
                buf[py, px, 0] = np.asarray(oi, np.float32)
        out[name] = buf
    return out


def _heat(t):
    t = np.clip(t, 0, 1)[..., None]
    a = np.array([0.0, 0.0, 0.3], np.float32)
    b = np.array([1.0, 0.2, 0.0], np.float32)
    c = np.array([1.0, 1.0, 0.6], np.float32)
    lo = a + (b - a) * np.clip(t * 2, 0, 1)
    hi = b + (c - b) * np.clip(t * 2 - 1, 0, 1)
    return np.where(t < 0.5, lo, hi)


def _idcolor(i):
    i = i.astype(np.float32)
    return np.stack([_hash1(i * 1.7 + 0.3), _hash1(i * 3.1 + 1.9),
                     _hash1(i * 5.3 + 4.1)], axis=1)


# ----------------------------------------------------------- AA / downsample

FILTERS = {
    'BOX': lambda x: np.ones_like(x),
    'TRIANGLE': lambda x: np.maximum(1.0 - np.abs(x), 0.0),
    'GAUSS': lambda x: np.exp(-2.0 * x * x),
    'CATROM': lambda x: _catrom(x),
    'MITCHELL': lambda x: _mitchell(x),
}


def _catrom(x):
    a = np.abs(x)
    a2 = a * a
    a3 = a2 * a
    w = np.where(a < 1.0, 1.5 * a3 - 2.5 * a2 + 1.0,
                 np.where(a < 2.0, -0.5 * a3 + 2.5 * a2 - 4.0 * a + 2.0, 0.0))
    return w


def _mitchell(x, b=1 / 3, c=1 / 3):
    a = np.abs(x)
    a2 = a * a
    a3 = a2 * a
    w1 = ((12 - 9 * b - 6 * c) * a3 + (-18 + 12 * b + 6 * c) * a2 + (6 - 2 * b)) / 6
    w2 = ((-b - 6 * c) * a3 + (6 * b + 30 * c) * a2 + (-12 * b - 48 * c) * a +
          (8 * b + 24 * c)) / 6
    return np.where(a < 1, w1, np.where(a < 2, w2, 0.0))


def _resolve(img, W, H, ss, st):
    # R251 (RAST-B): C094 LightWave's Limit Dynamic Range clips every
    # sample at 1.0 before the filter (the single pass too: LightWave
    # clipped it); C122 Blender 2.41's gamma-2 OSA blend squares the
    # samples through the table and square-roots the sum, inert at one
    # sample as Blender's do_gamma required OSA. Both twins live in
    # gpu/stages.resolve_source and are drawn by gpu/frame.resolve.
    clamp = bool(getattr(st, 'aa_clamp_samples', False))
    if ss <= 1:
        if clamp:
            return raster.clamp_samples(img)
        return img
    fn = FILTERS.get(st.aa_filter, FILTERS['BOX'])
    off = (np.arange(ss, dtype=np.float32) + 0.5) / ss - 0.5
    wx = fn(off * 2.0 * st.aa_filter_width)
    wy = wx
    kern = np.outer(wy, wx).astype(np.float32)
    s = kern.sum()
    kern = kern / (s if abs(s) > 1e-8 else 1.0)
    tile = img[:H * ss, :W * ss].reshape(H, ss, W, ss, 4)
    if clamp:
        tile = raster.clamp_samples(tile)
    if getattr(st, 'aa_gamma_blend', False):
        return raster.gamma2_blend(tile, kern)
    return np.einsum('hiwjc,ij->hwc', tile, kern).astype(np.float32)
