"""R251 texture pack tests (TEX-1, wave 1: the plumbing of texture.md 0.1,
then C074 texel formats, C013 TMEM budget, C024 block compression, C080
coarse bilinear fraction, C111 POV normalised distance; TEX-2 appends the
wave-2 sections to this module).

Every heading's "twin bitwise" means (texture.md 0.0): bitwise at the
SAMPLER (`sampler_run`, 500 off-grid uvs, four wraps) and on the frame's
OWN inputs (`frame_sampler_twin`: the real uv / footprint / depth / pixel
of every covered floor pixel through the emitted sampler in the
simulator), `d == 0.0`; the deferred FRAME twin (`twin()`) proves the
plumbing at the existing lighting-ulp bar `< 6e-3` (measured: NEAREST
5.96e-6 today), printing the max so a regression is visible.

    "C:/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe" -m halcyon.tests.test_r251_texture
"""
import importlib
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import base_settings, _prev_engine
from .scenebuild import demo_scene, checker_image
from . import featurematrix as FM
from ..core import render as R
from ..core import raster as CRg
from ..core import texture as TX
from ..core import nodeeval as NE
from ..core import post as PO
from ..core.settings import RenderSettings
from ..gpu import material as MAT
from ..gpu import shade as GSH
from ..shaders.compiler import try_compile
from ..presets.library import PRESETS

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


PY = "C:/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe"
W, H = 96, 72
WRAPS = ('REPEAT', 'EXTEND', 'CLIP', 'MIRROR')
#: the six settings the identity pins render with every R251 field at its
#: default: the filters the EXISTING presets exercise (A0.2)
PINS = [
    ('NEAREST', {}),
    ('BILINEAR', {'tex_filter': 'BILINEAR'}),
    ('TRILINEAR + mips', {'tex_filter': 'TRILINEAR', 'tex_mipmap': True}),
    ('TRILINEAR + aniso 4 + bias -1', {'tex_filter': 'TRILINEAR', 'tex_mipmap': True,
                                       'tex_aniso': 4, 'tex_mip_bias': -1.0}),
    ('N64 3-point + mips', {'tex_filter': 'N64_3POINT', 'tex_mipmap': True}),
    ('input_gamma_naive', {'input_gamma_naive': True}),
]
#: the eight wave-2 dials: plumbed through sample_opts, inert in pass 1
WAVE2_FIELDS = ('tex_clamp_mode', 'tex_colorkey', 'tex_colorkey_range',
                'tex_mip_select', 'tex_lod_source', 'tex_lod_k', 'tex_lod_l',
                'tex_lod_sharpen')


# ------------------------------------------------------------------ helpers
def scene(key='textured', **kw):
    st = base_settings(W, H, transparency='NONE', **kw)
    st.use_processes = False
    return FM.SCENES[key](st), st


def cpu_frame(key='textured', **kw):
    sc, st = scene(key, **kw)
    return np.asarray(R.render(sc, st))


def rig(sc, st):
    """The TR:20084-20091 rig up to the ShadeJob."""
    R._GBUF_CACHE.clear() if hasattr(R, '_GBUF_CACHE') else None
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, None, view, eye, W, H)
    return g, job


def twin(st, sc):
    """(gpu_img, cpu_img, gbuf, passes, why): the deferred frame twin."""
    cpu = np.asarray(R.render(sc, st))
    g, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    if p is None:
        return None, cpu, g, None, why
    out, hit = GSH.simulate(job, g)
    return out, cpu, g, p, hit


def frame_bar(label, st, sc, bar=6e-3, allow_px=0):
    """The deferred frame at the lighting bar. `allow_px` (wave 2): the
    number of covered pixels allowed past the bar where a one-texel box
    edge sits within a uv ULP of a texel boundary (the frame pass
    interpolates uv in GLSL, the CPU in numpy: a pre-existing class, the
    frame sampler twin proves the sampler on the CPU's own uv)."""
    out, cpu, g, p, why = twin(st, sc)
    check(f'{label}: the deferred pass is not refused', p is not None,
          '' if p is not None else str(why))
    if p is None:
        return None
    check(f'{label}: the passes simulate', out is not None,
          '' if out is not None else str(why))
    if out is None:
        return None
    yy, xx = np.nonzero(g.tri >= 0)
    d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
    over = int((d >= bar).sum())
    if allow_px:
        check(f'{label}: the GPU frame is the CPU frame at the deferred bar on all but at most '
              f'{allow_px} texel-edge pixel(s) (a uv ULP of the frame pass flips a one-texel box)',
              over <= allow_px, f'{over} px past {bar:.0e}; max {float(d.max()):.2e} over {yy.size} px')
    else:
        check(f'{label}: the GPU frame is the CPU frame at the deferred bar',
              float(d.max()) < bar, f'max {float(d.max()):.2e} over {yy.size} px')
    return out


def sampler_run(src, px, uv, extra=None, vuv=None):
    """One emitted sampler through the simulator (TR:11017-11031); `vUV` is
    declared as a uniform because the frame pass declares `in vec2 vUV;`
    and the fake device rewrites it so (A0.9)."""
    n = int(uv.shape[0])
    full = src + """
uniform vec2 hal_uv_in;
uniform vec2 vUV;
out vec4 Color;
void main() { Color = hal_sample_hal_tex0(hal_uv_in); }
"""
    prog, err = try_compile(full, 'GLSL')
    if prog is None:
        return None, str(err)
    uni = {'hal_tex0': TX.Texture(px, colorspace='Non-Color', filt='NEAREST',
                                  wrap='EXTEND'),
           'hal_uv_in': np.asarray(uv, np.float32),
           'vUV': (np.asarray(vuv, np.float32) if vuv is not None
                   else np.full((n, 2), 0.5, np.float32))}
    for k, v in (extra or {}).items():
        uni[k] = v
    got = prog.run(uni, {}, n)[0]['Color']
    return np.asarray(got, np.float32), ''


def frame_sampler_twin(label, st, sc, mat=0, node='tex', bar=0.0):
    """The bitwise proof on the frame's OWN inputs: no lighting in the loop;
    the real footprint, depth, dither pixel and triangle of every covered
    pixel of material `mat` through the sampler the plan would emit."""
    g, job = rig(sc, st)
    mesh = sc.mesh
    cov = (g.tri >= 0) & (mesh.mat_index[np.maximum(g.tri, 0)] == mat)
    yy, xx = np.nonzero(cov)
    tri = g.tri[yy, xx]
    bary = g.bary[yy, xx]
    ctx = job.context(tri, bary, xx, yy)
    graph = sc.materials[mat].graph
    ev = NE.GraphEvaluator(graph, ctx, job.textures, None)
    want = np.asarray(NE.n_tex_image(ev, graph['nodes'][node])['Color'], np.float32)
    key = graph['nodes'][node]['props']['image']
    tex = job.textures[key]
    consts = {'tex_filter': st.tex_filter, 'tex_mipmap': st.tex_mipmap,
              'tex_aniso': st.tex_aniso, 'tex_mip_bias': st.tex_mip_bias}
    for f in ('tex_frac_bits',) + WAVE2_FIELDS:
        consts[f] = getattr(st, f)
    opts = MAT._tex_opts(consts)
    filt = MAT.resolve_tex_filter(graph['nodes'][node]['props'].get('interpolation', 'Linear'),
                                  st.tex_filter)
    # wave 2: the node's own extension (textured_extend is an Extend node)
    wrap = {'REPEAT': 'REPEAT', 'EXTEND': 'EXTEND', 'CLIP': 'CLIP', 'MIRROR': 'MIRROR'}.get(
        graph['nodes'][node]['props'].get('extension', 'REPEAT'), 'REPEAT')
    extra = {}
    vuv = None
    fp = filt in MAT.FOOTPRINT_TEX_FILTERS or (
        filt in MAT.PYRAMID_FILTERS and (opts['lod_sharpen'] or opts['mip_select'] != 'FILTER'))
    if fp:
        lod_key = MAT._lod_field_key(opts, tex)
        src = 'uniform sampler2D hal_uvgrad;\n'
        if filt == 'SUMMED_AREA':
            src += 'uniform sampler2D hal_recip256;\n'
            rec = np.zeros((1, 256, 4), np.float32)
            rec[0, :, 0] = TX.RECIP256
            extra['hal_recip256'] = TX.Texture(rec, colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')
        if lod_key is not None:
            src += f'uniform sampler2D {MAT._lod_uniform(lod_key)};\n'
            extra[MAT._lod_uniform(lod_key)] = TX.Texture(
                GSH._lod_field(job, g, st, lod_key), colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')
        src += MAT._footprint_sampler(
            'hal_tex0', tex, filt, wrap, opts, float(st.tex_mip_bias or 0.0), lod_key)
        extra['hal_uvgrad'] = TX.Texture(GSH._uvgrad_field(job, g), colorspace='Non-Color',
                                         filt='NEAREST', wrap='EXTEND')
        vuv = np.stack([(xx + 0.5) / W, (yy + 0.5) / H], axis=1).astype(np.float32)
        px = MAT.sat_atlas(tex) if filt == 'SUMMED_AREA' else MAT.mip_atlas(tex)[0]
    else:
        src = MAT._texture_sampler('hal_tex0', tex, MAT.NOFOOTPRINT_FILTER.get(filt, filt),
                                   wrap, opts)
        px = tex.pixels
    got, err = sampler_run(src, px, ctx.uv, extra, vuv)
    if got is None:
        check(f'{label}: the frame sampler compiles', False, err)
        return
    d = float(np.abs(got - want).max())
    if bar > 0.0:
        check(f'{label}: the emitted sampler is Texture.sample on the frame' + "'s own uv / footprint / "
              f'pixel inputs within {bar:.1e} (the pre-existing trilinear level blend: the CPU' + "'s "
              "frac is float64 rounded once, 1.89.0's code)", d <= bar, f'max {d:.2e} over {yy.size} px')
        return
    check(f'{label}: the emitted sampler is bitwise Texture.sample on the frame' + "'s own "
          'uv / footprint / pixel inputs', d == 0.0, f'max {d:.2e} over {yy.size} px')


def prev():
    return _prev_engine('halcyon-1.89.0.zip')


def _post(img, st):
    return PO.process(np.asarray(img)[:, :, :3], st, frame=1, seed=st.seed)


def _ncc_table_of(px):
    e8 = np.stack([TX._k(px[:, :, 0], 8), TX._k(px[:, :, 1], 8), TX._k(px[:, :, 2], 8)],
                  axis=-1)
    return TX._ncc_fit(e8)


# ------------------------------------------------------------ 0: identity
def test_r251_identity():
    """The 1.89.0 zip is beside the package (loudly), and with EVERY R251
    field at its default the current tree renders the six PINS bitwise the
    1.89.0 engine, render AND post.process (CPU road)."""
    RP = prev()
    check('the 1.89.0 zip is beside the package', RP is not None,
          '' if RP is not None else 'halcyon-1.89.0.zip missing: the identity pin did NOT run')
    st0 = RenderSettings()
    for f in ('tex_format', 'tex_tmem_format', 'tex_compress', 'tex_frac_bits') + WAVE2_FIELDS:
        check(f'RenderSettings.{f} exists with its neutral default',
              hasattr(st0, f) and TX.sample_opts(st0) == TX.SAMPLE_OPTS_NEUTRAL, str(getattr(st0, f, None)))
    check('sample_opts of a default RenderSettings IS the neutral dict',
          TX.sample_opts(st0) == TX.SAMPLE_OPTS_NEUTRAL)
    for filt in ('NEAREST', 'BILINEAR', 'N64_3POINT', 'POV_NORMDIST'):
        old = MAT._texture_sampler('hal_tex0', TX.Texture(checker_image(), colorspace='Non-Color'), filt, 'REPEAT')
        new = MAT._texture_sampler('hal_tex0', TX.Texture(checker_image(), colorspace='Non-Color'), filt, 'REPEAT',
                                   dict(TX.SAMPLE_OPTS_NEUTRAL))
        check(f'the {filt} sampler at neutral opts is the opts=None text byte for byte', old == new)
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')
    for label, kw in PINS:
        sc, st = scene('textured', **kw)
        now = np.asarray(R.render(sc, st))
        sc2, st2 = scene('textured', **kw)
        old = np.asarray(RP.render(sc2, st2))
        same = now.shape == old.shape and bool(np.array_equal(now, old))
        check(f'identity at defaults: the textured frame under {label} renders bitwise the '
              "1.89.0 engine", same,
              f'max {float(np.abs(now - old).max()) if now.shape == old.shape else "shape"}')
        np_ = _post(now, st)
        op_ = prev_post.process(old[:, :, :3], st2, frame=1, seed=st2.seed)
        check(f'identity at defaults: render + post.process under {label} is bitwise the '
              "1.89.0 engine's", np_.shape == op_.shape and bool(np.array_equal(np_, op_)),
              f'max {float(np.abs(np_ - op_).max()) if np_.shape == op_.shape else "shape"}')
    # pass 2 (TEX-2 landed): the wave-2 dials ACT. On 'textured' (a REPEAT
    # node, an opaque checker, a minified floor) four of the five are
    # neutral for a stated reason each -- GL_CLAMP touches Extend only, the
    # chroma key needs a hole or black, sharpen needs magnification, GS_Q
    # under BILINEAR + FILTER reads level 0 as the GS did without a mip
    # select -- and NEAREST_LEVEL moves the picture; their A/B scenes are in
    # the feature tests below (wave 1's pass-1 inertness block inverted)
    base = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True)
    for f, v, moves in (('tex_mip_select', 'NEAREST_LEVEL', True), ('tex_lod_source', 'GS_Q', False),
                        ('tex_lod_sharpen', True, False), ('tex_clamp_mode', 'GL_CLAMP', False),
                        ('tex_colorkey', True, False)):
        img = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, **{f: v})
        check(f"pass 2: {f}={v} {'moves' if moves else 'is neutral on (stated reason)'} the plain textured "
              'frame under BILINEAR + mips', bool(np.array_equal(base, img)) != moves)
    # ...and the GPU no longer refuses their level roads: every road plans
    for filt, f, v in (('BILINEAR', 'tex_mip_select', 'NEAREST_LEVEL'),
                       ('TRILINEAR', 'tex_lod_source', 'GS_Q'),    # the GS rule needs Trilinear or a mip select
                       ('BILINEAR', 'tex_lod_sharpen', True)):
        sc, st = scene('textured', tex_filter=filt, tex_mipmap=True, **{f: v})
        _o, _c, _g, p, why = twin(st, sc)
        check(f'pass 2: the {filt} {f}={v} footprint road plans on the GPU (no refusal)',
              p is not None, str(why) if p is None else '')
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='GS_Q')
    _o, _c, _g, p, why = twin(st, sc)
    check('BILINEAR + GS_Q without a mip select wants no footprint (sample_opts) and is not refused',
          p is not None, str(why) if p is None else '')


# ------------------------------------------------------------------ C074
def test_texel_format_laws():
    """C074: the texel-format laws on a 256x1 grey ramp (values m/255) and a
    seeded 64x64 colour noise image."""
    ramp = np.zeros((1, 256, 4), np.float32)
    ramp[0, :, :3] = (np.arange(256, dtype=np.float32) / np.float32(255))[:, None]
    ramp[0, :, 3] = np.arange(256, dtype=np.float32) / np.float32(255)
    rng = np.random.default_rng(74)
    noise = rng.random((64, 64, 4), np.float32)
    noise[:, :, 3] = 1.0

    def fmt(px, f, encoded=True):
        t = TX.Texture(px.copy(), colorspace='Non-Color')
        t.store_format(f, encoded)
        return t.pixels

    p = fmt(ramp, 'RGB565')
    check('RGB565 leaves at most 32 / 64 / 32 distinct R / G / B values on the ramp',
          len(np.unique(p[0, :, 0])) <= 32 and len(np.unique(p[0, :, 1])) <= 64
          and len(np.unique(p[0, :, 2])) <= 32,
          f'{len(np.unique(p[0, :, 0]))} {len(np.unique(p[0, :, 1]))} {len(np.unique(p[0, :, 2]))}')
    check('RGB565 is monotone non-decreasing along the ramp',
          bool((np.diff(p[0, :, 0]) >= 0).all() and (np.diff(p[0, :, 1]) >= 0).all()))
    check('RGB565 quantised values are the bit-replicated bytes over 255',
          all(int(round(float(v) * 255)) in {TX._repl(k, 5) for k in range(32)} for v in np.unique(p[0, :, 0])))
    a = fmt(ramp, 'ARGB1555')[0, :, 3]
    lo = ramp[0, :, 3] < 0.5
    check('ARGB1555 alpha is exactly {0, 1} and flips at one half (0.499 -> 0, 0.5 -> 1)',
          bool(np.array_equal(a[lo], np.zeros(lo.sum(), np.float32))
               and np.array_equal(a[~lo], np.ones((~lo).sum(), np.float32))
               and set(np.unique(a).tolist()) <= {0.0, 1.0}))
    p = fmt(ramp, 'ARGB4444')
    check('ARGB4444 has exactly 16 levels per channel on the ramp',
          all(len(np.unique(p[0, :, c])) == 16 for c in range(4)),
          str([len(np.unique(p[0, :, c])) for c in range(4)]))
    p = fmt(noise, 'I8')
    check('I8 gives r == g == b == a per texel bitwise (grey that feeds alpha)',
          bool(np.array_equal(p[..., 0], p[..., 1]) and np.array_equal(p[..., 1], p[..., 2])
               and np.array_equal(p[..., 2], p[..., 3])))
    lum = TX._luma601(noise[:, :, :3])
    hole = TX._k(lum, 4) == 15
    p = fmt(noise, 'I4_MODEL2')
    check('I4_MODEL2 sets alpha 0 exactly where the luma rounds to 15 and 1 elsewhere',
          bool(hole.any()) and bool((p[..., 3][hole] == 0.0).all() and (p[..., 3][~hole] == 1.0).all()),
          f'{int(hole.sum())} hole texels')
    check('I4 keeps every texel opaque (Model 3 luma without the marker)',
          bool((fmt(noise, 'I4')[..., 3] == 1.0).all()))
    p = fmt(ramp, 'A3I5')
    check('A3I5 leaves at most 32 distinct greys and exactly 8 distinct alphas on the ramp',
          len(np.unique(p[0, :, 0])) <= 32 and len(np.unique(p[0, :, 3])) == 8,
          f'{len(np.unique(p[0, :, 0]))} greys {len(np.unique(p[0, :, 3]))} alphas')
    p = fmt(ramp, 'A5I3')
    check('A5I3 leaves at most 8 distinct greys and exactly 32 distinct alphas on the ramp',
          len(np.unique(p[0, :, 0])) <= 8 and len(np.unique(p[0, :, 3])) == 32,
          f'{len(np.unique(p[0, :, 0]))} greys {len(np.unique(p[0, :, 3]))} alphas')
    rgb8, yi, table = _ncc_table_of(noise)
    p = fmt(noise, 'YIQ422')
    dec = np.rint(p[..., :3] * 255).astype(np.int64).reshape(-1, 3)
    members = {tuple(r) for r in table.tolist()}
    check("YIQ422's decoded texels are all members of the 256-entry decode table (bitwise)",
          all(tuple(r) in members for r in dec.tolist()) and bool(np.array_equal(dec, rgb8.reshape(-1, 3))))
    check('the NCC y-index image takes at most 16 values', len(np.unique(yi)) <= 16, str(len(np.unique(yi))))
    check('AYIQ8422 keeps an 8-bit alpha byte (the ramp alpha survives at 256 levels)',
          len(np.unique(fmt(ramp, 'AYIQ8422')[0, :, 3])) == 256)
    check('A8 is white with the image alpha', bool((fmt(ramp, 'A8')[0, :, :3] == 1.0).all())
          and len(np.unique(fmt(ramp, 'A8')[0, :, 3])) == 256)
    # the linear-road round trip (0.1 B's claim, colour items)
    from ..core.mathx import srgb_to_linear
    bytes_ = (rng.integers(0, 256, (16, 16, 4)).astype(np.float32) / np.float32(255))
    bytes_[:, :, 3] = 1.0
    lin = bytes_.copy()
    lin[:, :, :3] = srgb_to_linear(np.ascontiguousarray(bytes_[:, :, :3]))
    a_ = fmt(lin, 'RGB565', encoded=False)
    b_ = fmt(bytes_, 'RGB565', encoded=True)
    b_[:, :, :3] = srgb_to_linear(np.ascontiguousarray(b_[:, :, :3]))
    check('the linear road round-trips: store_format on decoded pixels equals decoding the '
          'quantised bytes, bitwise (no byte sits within 1e-7 of a tie)',
          bool(np.array_equal(a_, b_)), f'max {float(np.abs(a_ - b_).max()):.2e}')
    two = fmt(noise, 'YIQ422')
    check('the NCC fit is deterministic (two calls bitwise equal)', bool(np.array_equal(two, fmt(noise, 'YIQ422'))))
    check('an unknown texel format raises by name',
          _raises(lambda: TX.Texture(noise).store_format('RGB999', True)))


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def test_texel_format_moves_the_picture():
    """C074 A/B on the CPU road."""
    base = cpu_frame('textured')
    img = cpu_frame('textured', tex_format='RGB332')
    check("'tex_format' RGB332 changes the textured frame (A/B)", not np.array_equal(base, img))
    i8 = cpu_frame('textured', tex_format='I8')
    check('I8 keeps the textured frame mean within 10% of NONE (a grey of the same luma)',
          abs(float(i8[..., :3].mean()) - float(base[..., :3].mean())) < 0.1 * float(base[..., :3].mean()),
          f'{float(i8[..., :3].mean()):.4f} vs {float(base[..., :3].mean()):.4f}')
    check('I8 is not the NONE frame', not np.array_equal(base, i8))
    sc, st = scene('textured_white', tex_format='I4_MODEL2')
    t = R.prepare_textures(sc, st)['checker']
    check("on 'textured_white' I4_MODEL2 gives alpha 0 on the white square's texels (the 0xF hole)",
          bool((t.pixels[0:8, 0:8, 3] == 0.0).all()) and bool((t.pixels[8:, 8:, 3] == 1.0).all()))
    sc, st = scene('textured_white', tex_format='I4')
    t = R.prepare_textures(sc, st)['checker']
    check("on 'textured_white' I4 keeps the white square opaque", bool((t.pixels[..., 3] == 1.0).all()))
    check('the prepared texture carries its format and prep tag',
          t.format == 'I4' and t.prep is not None and 'I4' in t.prep)
    p = PRESETS.get('SEGA_MODEL2')
    check('the SEGA_MODEL2 preset exists (merged with lighting F017 by the integrator, #34)', p is not None)
    if p is not None:
        s = p['settings']
        known = {f.name for f in __import__('dataclasses').fields(RenderSettings)}
        check('every SEGA_MODEL2 key is a real setting', not (set(s) - known), str(set(s) - known))
        check("SEGA_MODEL2 stores I4_MODEL2 luma textures at FACE rate over a stipple",
              s.get('tex_format') == 'I4_MODEL2' and s.get('shading_rate') == 'FACE'
              and s.get('transparency') == 'STIPPLE')
        check('SEGA_MODEL2 sits on a shelf the library knows',
              p.get('category') in ('CONSOLE', 'ARCADE'), str(p.get('category')))
        st = base_settings(64, 48)
        st.apply(dict(s))
        st.resolution_x, st.resolution_y = 64, 48
        st.aa_mode, st.aa_samples, st.use_processes = 'NONE', 1, False
        img = np.asarray(R.render(demo_scene(st, with_texture=True), st))
        check('the SEGA_MODEL2 preset renders', np.isfinite(img).all() and float(img.max()) > 0.01)
    check('the VOODOO preset uploads at RGB565', PRESETS['VOODOO']['settings'].get('tex_format') == 'RGB565')


def test_texel_format_gpu_twin():
    """C074 on the GPU: no refusal under every item, the frame at the deferred
    bar and the sampler bitwise (one prepared array, 1.89.0's sampler)."""
    for f in ('RGB565', 'ARGB4444', 'I4_MODEL2', 'A3I5', 'YIQ422'):
        sc, st = scene('textured', tex_filter='NEAREST', tex_format=f)
        frame_bar(f'texel format {f}', st, sc)
        frame_sampler_twin(f'texel format {f}', st, sc)


# ------------------------------------------------------------------ C013
def test_tmem_budget_sizes():
    """C013: the size table from a seeded 128x128 RGBA noise image, the
    colour counts, the alpha laws, the mip sum rule and the precedence."""
    rng = np.random.default_rng(13)
    px = rng.random((128, 128, 4), np.float32)
    px[:, :, 3] = 1.0
    want = {'RGBA16': (32, 32), 'RGBA32': (32, 32), 'CI8': (32, 32), 'CI4': (64, 64),
            'I4': (64, 64), 'IA4': (64, 64), 'IA16': (32, 32), 'IA8': (64, 64), 'I8': (64, 64)}
    for f, (w, h) in want.items():
        t = TX.Texture(px.copy(), colorspace='Non-Color')
        t.fit_tmem(f, False, True)
        check(f'a 128x128 source under {f} fits TMEM at {w}x{h}',
              (t.width, t.height) == (w, h), f'{t.width}x{t.height}')
        if f in ('CI4', 'CI8'):
            n = 16 if f == 'CI4' else 256
            rgb8 = np.rint(t.pixels[..., :3].reshape(-1, 3) * 255).astype(np.int64)
            check(f'{f} holds at most {n} colours', len(np.unique(rgb8, axis=0)) <= n, str(len(np.unique(rgb8, axis=0))))
        if f == 'I4':
            check("I4's alpha IS its intensity bitwise (the N64 I formats read alpha from intensity)",
                  bool(np.array_equal(t.pixels[..., 3], t.pixels[..., 0])))
        if f == 'IA4':
            check("IA4's alpha is {0, 1} and its grey has at most 8 levels",
                  set(np.unique(t.pixels[..., 3]).tolist()) <= {0.0, 1.0} and len(np.unique(t.pixels[..., 0])) <= 8)
    t = TX.Texture(px.copy(), colorspace='Non-Color')
    t.fit_tmem('RGBA16', True, True)
    check('with mips RGBA16 stays 32x32 (2744 bytes for the 32-1 chain; the sum rule never enlarges)',
          (t.width, t.height) == (32, 32) and TX.tmem_bytes(32, 32, 'RGBA16', True) == 2744,
          f'{t.width}x{t.height} {TX.tmem_bytes(32, 32, "RGBA16", True)}')
    t = TX.Texture(np.ones((1, 1, 4), np.float32), colorspace='Non-Color')
    t.fit_tmem('RGBA32', False, True)
    check('a 1x1 image is untouched by the budget', (t.width, t.height) == (1, 1))
    t = TX.Texture(px[:32, :64].copy(), colorspace='Non-Color')
    t.fit_tmem('RGBA16', False, True)
    check("a 64x32 source under RGBA16 stays 64x32 (2048 texels x 2 bytes = 4096, the manual's example)",
          (t.width, t.height) == (64, 32), f'{t.width}x{t.height}')
    check('the byte rule pads rows to 64-bit words',
          TX.tmem_bytes(1, 1, 'I4') == 8 and TX.tmem_bytes(64, 64, 'CI4') == 2048
          and TX.tmem_budget('CI4') == 2048 and TX.tmem_budget('I4') == 4096)
    # precedence (A2.1): TMEM owns the texel format
    sc, st = scene('textured', tex_tmem_format='CI4', tex_format='RGB565')
    a_ = R.prepare_textures(sc, st)['checker'].pixels
    sc, st = scene('textured', tex_tmem_format='CI4', tex_format='NONE')
    b_ = R.prepare_textures(sc, st)['checker'].pixels
    check('a TMEM format owns the texel format: CI4 + RGB565 prepares bitwise CI4 + NONE',
          bool(np.array_equal(a_, b_)))
    base = cpu_frame('textured')
    img = cpu_frame('textured', tex_tmem_format='CI4')
    check("'tex_tmem_format' CI4 changes the textured frame (0.9 is not 5551-representable)",
          not np.array_equal(base, img))
    check('the N64 preset takes CI4', PRESETS['N64']['settings'].get('tex_tmem_format') == 'CI4')


def test_tmem_gpu_twin():
    for f in ('RGBA16', 'CI4'):
        sc, st = scene('textured', tex_filter='NEAREST', tex_tmem_format=f)
        frame_bar(f'TMEM {f}', st, sc)
        frame_sampler_twin(f'TMEM {f}', st, sc)


# ------------------------------------------------------------------ C024
def test_dxt1_block_laws():
    rng = np.random.default_rng(24)
    px = rng.random((64, 64, 4), np.float32)
    px[:, :, 3] = 1.0

    def comp(img, mode):
        t = TX.Texture(img.copy(), colorspace='Non-Color')
        t.block_compress(mode, True)
        return t.pixels

    for mode in ('DXT1', 'DXT1_NV2A', 'CMPR_GC'):
        dec = comp(px, mode)
        blocks = dec[..., :3].reshape(16, 4, 16, 4, 3).transpose(0, 2, 1, 3, 4).reshape(256, 16, 3)
        counts = [len(np.unique(np.rint(b * 255).astype(np.int64), axis=0)) for b in blocks]
        check(f'every 4x4 block of the {mode} decode has at most 4 distinct colours (block law)',
              max(counts) <= 4, str(max(counts)))
    uni = np.zeros((8, 8, 4), np.float32)
    uni[..., 3] = 1.0
    uni[..., :3] = np.array([231, 235, 231], np.float32) / 255
    check('a uniform 5:6:5-representable colour (231, 235, 231) round-trips bitwise under DXT1',
          bool(np.array_equal(comp(uni, 'DXT1'), uni)))
    uni[..., :3] = np.float32(230.0 / 255)
    check('a uniform colour that is not 5:6:5-representable (230, 230, 230) does not round-trip',
          not np.array_equal(comp(uni, 'DXT1'), uni))
    hand = np.zeros((4, 4, 4), np.float32)
    hand[..., 3] = 1.0
    hand[0, 1, :3] = 1.0
    hand[1, :, :3] = 0.333
    hand[2, :, :3] = 0.666
    want = {'DXT1': {(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)},
            'DXT1_NV2A': {(0, 0, 0), (82, 85, 82), (165, 170, 165), (255, 255, 255)},
            'CMPR_GC': {(0, 0, 0), (95, 95, 95), (159, 159, 159), (255, 255, 255)}}
    for mode, pal in want.items():
        got = {tuple(r) for r in np.rint(comp(hand, mode)[..., :3].reshape(-1, 3) * 255).astype(np.int64).tolist()}
        check(f'the (0,0,0)/(255,255,255) hand block decodes its interpolants to {sorted(pal)[1:3]} under {mode}',
              got == pal, str(sorted(got)))
    tr = hand.copy()
    tr[3, 3, 3] = 0.2
    dec = comp(tr, 'DXT1')
    check('a block holding one texel of alpha 0.2 takes the three-colour mode: that texel is (0,0,0,0) and its neighbours keep alpha 1',
          bool((dec[3, 3] == 0.0).all()) and bool((dec[..., 3].reshape(-1)[:-1] == 1.0).all()))
    err = float(np.abs(comp(px, 'DXT1')[..., :3] - px[..., :3]).mean())
    check('DXT1 is lossy on i.i.d. texel noise: 0 < mean abs error < 0.25 (measured 0.1955; the min/max-luma '
          "endpoints cannot follow chroma noise, so the spec's 0.12 is a structured-image number)",
          0.0 < err < 0.25, f'{err:.4f}')
    blk = FM.SCENES['textured_blocks'](base_settings()).images['checker'].pixels
    errb = float(np.abs(comp(blk, 'DXT1')[..., :3] - blk[..., :3]).mean())
    check("DXT1 on the structured 'textured_blocks' image: 0 < mean abs error < 0.12", 0.0 < errb < 0.12, f'{errb:.4f}')
    check('DXT1 is deterministic (two calls bitwise equal)', bool(np.array_equal(comp(px, 'DXT1'), comp(px, 'DXT1'))))
    # the straddling scene (A12.2): an interpolant fires on 'textured_blocks'
    # (its checker carries a two-axis ramp: a plain two-colour checker decodes
    # to its endpoints alone -- measured, so the builder deviates from the spec)
    dec = comp(blk, 'DXT1')
    ends = {tuple(r) for r in np.rint(TX.Texture(blk.copy()).store_format('RGB565', True).pixels[..., :3].reshape(-1, 3) * 255).astype(np.int64).tolist()}
    got = {tuple(r) for r in np.rint(dec[..., :3].reshape(-1, 3) * 255).astype(np.int64).tolist()}
    check("on 'textured_blocks' DXT1 decodes a texel that is neither endpoint colour (an interpolant fired)",
          bool(got - ends), str(sorted(got - ends)[:3]))
    check('an unknown compression raises by name', _raises(lambda: TX.Texture(px).block_compress('LZW', True)))


def test_vq_codebook_laws():
    rng = np.random.default_rng(240)
    px = rng.random((64, 64, 4), np.float32)
    px[:, :, 3] = 1.0

    def vq(img):
        t = TX.Texture(img.copy(), colorspace='Non-Color')
        t.block_compress('VQ_DC', True)
        return t.pixels

    dec = vq(px)
    blocks = dec.reshape(32, 2, 32, 2, 4).transpose(0, 2, 1, 3, 4).reshape(1024, 16)
    n = len(np.unique(np.rint(blocks * 255).astype(np.int64), axis=0))
    check('the VQ decode of the noise image has at most 256 distinct 2x2 blocks', n <= 256, str(n))
    rgb8 = np.rint(dec[..., :3] * 255).astype(np.int64)
    r5 = {TX._repl(k, 5) for k in range(32)}
    g6 = {TX._repl(k, 6) for k in range(64)}
    check("an opaque image's decoded texels are all 5:6:5-representable bytes",
          set(np.unique(rgb8[..., 0]).tolist()) <= r5 and set(np.unique(rgb8[..., 1]).tolist()) <= g6
          and set(np.unique(rgb8[..., 2]).tolist()) <= r5)
    chk = checker_image()
    q565 = TX.Texture(chk.copy(), colorspace='Non-Color').store_format('RGB565', True).pixels
    check('a 2-colour checker decodes bitwise to its own 565 quantisation (LBG converges on two codes)',
          bool(np.array_equal(vq(chk), q565)), f'max {float(np.abs(vq(chk) - q565).max()):.2e}')
    check('VQ is deterministic (two runs bitwise equal)', bool(np.array_equal(dec, vq(px))))
    one = np.zeros((2, 2, 4), np.float32)
    one[..., 3] = 1.0
    one[..., :3] = np.array([[[10, 200, 30], [255, 0, 128]], [[64, 64, 64], [1, 2, 3]]], np.float32) / 255
    q = TX.Texture(one.copy(), colorspace='Non-Color').store_format('RGB565', True).pixels
    check('a 2x2 image (one block) is its own 565 quantisation', bool(np.array_equal(vq(one), q)))
    blk = FM.SCENES['textured_blocks'](base_settings()).images['checker'].pixels
    d2 = vq(blk)
    b2 = d2.reshape(32, 2, 32, 2, 4).transpose(0, 2, 1, 3, 4).reshape(1024, 16)
    check("on 'textured_blocks' VQ holds more than two distinct 2x2 blocks (the codebook is used)",
          len(np.unique(np.rint(b2 * 255).astype(np.int64), axis=0)) > 2)
    al = px.copy()
    al[:8, :8, 3] = 0.0
    da = vq(al)
    check('a non-opaque image takes the 4444 codebook (16 alpha levels at most, holes kept)',
          len(np.unique(np.rint(da[..., 3] * 255).astype(np.int64))) <= 16 and float(da[:8, :8, 3].max()) < 0.5)
    base = cpu_frame('textured_blocks')
    for m in ('DXT1', 'VQ_DC'):
        img = cpu_frame('textured_blocks', tex_compress=m)
        check(f"'tex_compress' {m} changes the 'textured_blocks' frame (A/B)", not np.array_equal(base, img))
    for k, m in (('XBOX', 'DXT1_NV2A'), ('GAMECUBE', 'CMPR_GC'), ('DREAMCAST', 'VQ_DC')):
        check(f'the {k} preset bakes {m}', PRESETS[k]['settings'].get('tex_compress') == m)
    # composition: compression after the texel format (the Xbox stored DXT1 of a 565 source)
    sc, st = scene('textured_blocks', tex_format='RGB565', tex_compress='DXT1')
    t = R.prepare_textures(sc, st)['checker']
    check('block compression composes after the texel format (both tags on the prepared texture)',
          t.format == 'RGB565' and t.compress == 'DXT1')


def test_compress_gpu_twin():
    for m in ('DXT1', 'DXT1_NV2A', 'CMPR_GC', 'VQ_DC'):
        sc, st = scene('textured_blocks', tex_filter='NEAREST', tex_compress=m)
        frame_bar(f'block compression {m}', st, sc)
        if m in ('DXT1', 'VQ_DC'):
            frame_sampler_twin(f'block compression {m}', st, sc)


# ------------------------------------------------------------------ C080
def test_frac_bits_laws():
    """C080: a 2x1 texture (texels 0 and 1) sampled between the centres --
    the endpoint 0.75 is texel 1's own centre (tx = 0 on c00 = 1.0), a 17th
    value no interior u reaches, hence endpoint=False (B4.1)."""
    px = np.zeros((1, 2, 4), np.float32)
    px[0, 1, :] = 1.0
    px[0, 0, 3] = 1.0
    t = TX.Texture(px, colorspace='Non-Color')
    u = np.linspace(0.25, 0.75, 1000, endpoint=False).astype(np.float32)
    v = np.full(1000, 0.5, np.float32)

    def s(bits):
        o = dict(TX.SAMPLE_OPTS_NEUTRAL, frac_bits=bits)
        return t.sample(u, v, filt='BILINEAR', wrap='EXTEND', opts=o)[:, 0]

    f0, f4, f8 = s(0), s(4), s(8)
    check('BITS_4 yields exactly 16 distinct blend values between two texels', len(np.unique(f4)) == 16, str(len(np.unique(f4))))
    check('BITS_8 yields exactly 256 distinct blend values', len(np.unique(f8)) == 256, str(len(np.unique(f8))))
    check('FLOAT yields more than 256', len(np.unique(f0)) > 256, str(len(np.unique(f0))))
    check('each 4-bit value is <= the float value and > float - 1/16 (the floor law)',
          bool((f4 <= f0).all() and (f4 > f0 - 1.0 / 16).all()))
    check('each 8-bit value is <= the float value and > float - 1/256',
          bool((f8 <= f0).all() and (f8 > f0 - 1.0 / 256).all()))
    check('the quantised sequence is monotone non-decreasing in u', bool((np.diff(f4) >= 0).all() and (np.diff(f8) >= 0).all()))
    check('opts=None is FLOAT bitwise', bool(np.array_equal(t.sample(u, v, filt='BILINEAR', wrap='EXTEND'), t.sample(u, v, filt='BILINEAR', wrap='EXTEND', opts=dict(TX.SAMPLE_OPTS_NEUTRAL)))))
    check('sample_opts maps the enum to 0 / 4 / 8 bits',
          [TX.sample_opts(base_settings(tex_frac_bits=x))['frac_bits'] for x in ('FLOAT', 'BITS_4', 'BITS_8')] == [0, 4, 8])
    # 3-point keeps the RDP's own arithmetic
    o4 = dict(TX.SAMPLE_OPTS_NEUTRAL, frac_bits=4)
    check('3-Point is untouched by the fraction bits',
          bool(np.array_equal(t.sample(u, v, filt='N64_3POINT', wrap='EXTEND'), t.sample(u, v, filt='N64_3POINT', wrap='EXTEND', opts=o4))))
    # the volume BOX road takes the dial too (A12.6)
    n = 200
    ctx = NE.ShadeContext(n)
    ctx.is_volume = True
    rng = np.random.default_rng(80)
    ctx.generated = rng.random((n, 3), np.float32)
    node = {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
            'props': {'image': 'img', 'interpolation': 'Linear', 'projection': 'BOX'},
            'inputs': [{'name': 'Vector', 'type': 'VECTOR', 'default': [0, 0, 0], 'link': None}],
            'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]}
    graph = {'output': 'out', 'nodes': {'tex': node}}
    img = TX.Texture(checker_image(size=8, squares=2), colorspace='Non-Color')
    outs = {}
    for bits in ('FLOAT', 'BITS_4'):
        ctx.settings = base_settings(tex_filter='BILINEAR', tex_frac_bits=bits)
        ev = NE.GraphEvaluator(graph, ctx, {'img': img}, None)
        outs[bits] = NE.n_tex_image(ev, node)['Color']
    check("a volume's BOX-projected solid texture takes the fraction bits (BITS_4 differs from FLOAT)",
          not np.array_equal(outs['FLOAT'], outs['BITS_4']))
    base = cpu_frame('textured_mag', tex_filter='BILINEAR')
    img4 = cpu_frame('textured_mag', tex_filter='BILINEAR', tex_frac_bits='BITS_4')
    check("'tex_frac_bits' BITS_4 changes the magnified bilinear frame (A/B)", not np.array_equal(base, img4))
    check('the VOODOO preset keeps four bits', PRESETS['VOODOO']['settings'].get('tex_frac_bits') == 'BITS_4')


def test_frac_bits_gpu_twin():
    rng = np.random.default_rng(11)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = TX.Texture(px, colorspace='Non-Color')
    uv = (rng.random((500, 2)).astype(np.float32) * 4.0 - 1.5)
    for bits in (4, 8):
        o = dict(TX.SAMPLE_OPTS_NEUTRAL, frac_bits=bits)
        for wrap in WRAPS:
            got, err = sampler_run(MAT._texture_sampler('hal_tex0', tex, 'BILINEAR', wrap, o), px, uv)
            if got is None:
                check(f'BILINEAR/{wrap} BITS_{bits} sampler compiles', False, err)
                continue
            want = tex.sample(uv[:, 0], uv[:, 1], filt='BILINEAR', wrap=wrap, opts=o)
            d = float(np.abs(got - want).max())
            check(f'BILINEAR/{wrap} with {bits} fraction bits is bitwise Texture.sample off the grid '
                  "(the CPU's lerp order, no mix())", d == 0.0, f'max {d:.2e}')
    src = MAT._texture_sampler('hal_tex0', tex, 'BILINEAR', 'REPEAT', dict(TX.SAMPLE_OPTS_NEUTRAL, frac_bits=4))
    check('the 4-bit sampler carries the Voodoo quantisation lines', 'float txq = floor(tx * 16.0);' in src and 'tx = txq * 0.0625;' in src)
    sc, st = scene('textured_mag', tex_filter='BILINEAR', tex_frac_bits='BITS_4')
    frame_bar('BILINEAR + BITS_4 on textured_mag', st, sc)
    frame_sampler_twin('BILINEAR + BITS_4 on textured_mag', st, sc)
    sc, st = scene('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_frac_bits='BITS_8')
    frame_bar('TRILINEAR + mips + BITS_8 on textured', st, sc)
    frame_sampler_twin('TRILINEAR + mips + BITS_8 on textured', st, sc, bar=1.2e-7)
    sc, st = scene('textured', tex_filter='TRILINEAR', tex_mipmap=True)
    frame_sampler_twin('TRILINEAR + mips + FLOAT on textured (the same ulp without the pack)', st, sc, bar=1.2e-7)


# ------------------------------------------------------------------ C111
def test_normdist_laws():
    rng = np.random.default_rng(111)
    px = rng.random((4, 4, 4), np.float32)
    px[0, 0, :] = 0.0
    px[3, 3, :] = 1.0
    t = TX.Texture(px, colorspace='Non-Color')
    yy, xx = np.mgrid[0:4, 0:4]
    u = ((xx.reshape(-1) + 0.5) / 4).astype(np.float32)
    v = ((yy.reshape(-1) + 0.5) / 4).astype(np.float32)
    got = t.sample(u, v, filt='POV_NORMDIST', wrap='REPEAT')
    want = px[yy.reshape(-1), xx.reshape(-1)]
    check('at every texel centre the output equals that texel within 1e-6 (the cusp)',
          float(np.abs(got - want).max()) < 1e-6, f'max {float(np.abs(got - want).max()):.2e}')
    check('a texel of exactly 1 comes back bitwise at its centre (w3 ~ 1e12 absorbs the other three weights)',
          bool((got[15] == 1.0).all()))
    check("a texel of exactly 0 comes back within 1e-6 at its centre (the neighbours' share is ~1e-12, not 0: "
          "the spec's bitwise claim holds only for 1)", float(np.abs(got[0]).max()) < 1e-6,
          f'{float(np.abs(got[0]).max()):.1e}')
    const = TX.Texture(np.full((4, 4, 4), 0.37, np.float32), colorspace='Non-Color')
    uv = rng.random((500, 2)).astype(np.float32) * 4 - 1.5
    c = const.sample(uv[:, 0], uv[:, 1], filt='POV_NORMDIST', wrap='REPEAT')
    check('a constant texture returns the constant within one float32 ulp (the weights normalise; '
          'float32 multiplication does not distribute over the four-term sum, measured 5.96e-8)',
          float(np.abs(c - np.float32(0.37)).max()) <= 1.2e-7, f'max {float(np.abs(c - np.float32(0.37)).max()):.2e}')
    step = np.zeros((1, 8, 4), np.float32)
    step[0, 4:, :] = 1.0
    step[0, :, 3] = 1.0
    ts = TX.Texture(step, colorspace='Non-Color')
    us = np.linspace(0.0, 1.0, 200).astype(np.float32)
    vs = np.full(200, 0.5, np.float32)
    nd = ts.sample(us, vs, filt='POV_NORMDIST', wrap='EXTEND')[:, 0]
    bl = ts.sample(us, vs, filt='BILINEAR', wrap='EXTEND')[:, 0]
    stp = (us >= 0.5).astype(np.float32)
    check('on a 1-D step the profile is monotone non-decreasing', bool((np.diff(nd) >= -1e-7).all()))
    check('normalised distance is sharper than bilinear at every sample (|nd - step| <= |bilinear - step|)',
          bool((np.abs(nd - stp) <= np.abs(bl - stp) + 1e-6).all()))
    nd2 = ts.sample((1.0 - us).astype(np.float32), vs, filt='POV_NORMDIST', wrap='EXTEND')[:, 0]
    check('mirror symmetry: f(u) + f(1 - u) == 1 within 1e-6 on the step', float(np.abs(nd + nd2 - 1.0).max()) < 1e-6)
    base = cpu_frame('textured')
    bil = cpu_frame('textured', tex_filter='BILINEAR')
    pov = cpu_frame('textured', tex_filter='POV_NORMDIST')
    check("'tex_filter' POV_NORMDIST differs from NEAREST and from BILINEAR (A/B)",
          not np.array_equal(base, pov) and not np.array_equal(bil, pov))
    check('the POV_NORMDIST item is in the properties enum',
          any(i[0] == 'POV_NORMDIST' for i in _tex_filter_items()))
    check('the level dials are neutral for a filter without a pyramid (rule 2)',
          TX.sample_opts(base_settings(tex_filter='POV_NORMDIST', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL'))['mip_select'] == 'FILTER')
    check('POV_NORMDIST never wants a footprint', not TX.footprint_wanted(base_settings(tex_filter='POV_NORMDIST', tex_mipmap=True)))


def _tex_filter_items():
    try:
        from . import fakebpy
        fakebpy.install()
        from .. import properties as PR
        return PR.TEX_FILTER
    except Exception:                                           # noqa: BLE001
        traceback.print_exc()
        return []


def test_normdist_gpu_twin():
    rng = np.random.default_rng(11)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = TX.Texture(px, colorspace='Non-Color')
    uv = (rng.random((500, 2)).astype(np.float32) * 4.0 - 1.5)
    for wrap in WRAPS:
        got, err = sampler_run(MAT._texture_sampler('hal_tex0', tex, 'POV_NORMDIST', wrap), px, uv)
        if got is None:
            check(f'POV_NORMDIST/{wrap} sampler compiles', False, err)
            continue
        want = tex.sample(uv[:, 0], uv[:, 1], filt='POV_NORMDIST', wrap=wrap)
        d = float(np.abs(got - want).max())
        check(f'POV_NORMDIST/{wrap} is bitwise Texture.sample off the grid in the simulator', d == 0.0, f'max {d:.2e}')
    check('POV_NORMDIST is a supported deferred filter', 'POV_NORMDIST' in MAT.SUPPORTED_TEX_FILTERS)
    sc, st = scene('textured', tex_filter='POV_NORMDIST')
    frame_bar('POV_NORMDIST on textured', st, sc)
    frame_sampler_twin('POV_NORMDIST on textured', st, sc)
    check('an unlisted filter raises in _texture_sampler instead of falling through to bilinear',
          _raises(lambda: MAT._texture_sampler('hal_tex0', tex, 'SUMMED_AREA', 'REPEAT')))


# ------------------------------------------------------------ plumbing laws
def test_plumbing_laws():
    """0.1: the prep tag, the upload key, the derivative gate and the
    sample_opts rules, on both roads."""
    sc, st = scene('textured', tex_format='RGB565')
    t = R.prepare_textures(sc, st)['checker']
    sc2, st2 = scene('textured', tex_format='ARGB4444')
    t2 = R.prepare_textures(sc2, st2)['checker']
    check('two content laws give two prep tags (the upload keys differ by construction)', t.prep != t2.prep)
    st3 = base_settings(tex_filter='BILINEAR')
    check('a filter without a footprint never wants derivatives', not TX.footprint_wanted(st3))
    check('TRILINEAR wants the footprint as today', TX.footprint_wanted(base_settings(tex_filter='TRILINEAR')))
    st4 = base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL')
    check('a level road wants the footprint (rule 1 with mips on)', TX.footprint_wanted(st4))
    st5 = base_settings(tex_filter='BILINEAR', tex_mipmap=False, tex_mip_select='NEAREST_LEVEL')
    check('without Mipmaps the level dials are neutral (rule 1)',
          TX.sample_opts(st5)['mip_select'] == 'FILTER' and not TX.footprint_wanted(st5))
    st6 = base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_aniso=4, tex_lod_source='GS_Q')
    check('anisotropy is off under a level road (rule 3)', TX.sample_opts(st6)['aniso'] == 1)
    st7 = base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_aniso=4)
    check('anisotropy is tex_aniso on the derivative road', TX.sample_opts(st7)['aniso'] == 4)
    st8 = base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_mip_select='BLEND')
    check('TRILINEAR + BLEND reads as FILTER (one picture, one dial position, A9.2)', TX.sample_opts(st8)['mip_select'] == 'FILTER')
    st9 = base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE')
    check('under TRIANGLE, FILTER reads as NEAREST_LEVEL (rule 4)', TX.sample_opts(st9)['mip_select'] == 'NEAREST_LEVEL')
    consts = {k: getattr(st6, k) for k in MAT._OPT_KEYS}
    check('_tex_opts on the consts IS sample_opts on the settings', MAT._tex_opts(consts) == TX.sample_opts(st6))
    # the plan signature carries the tex_* keys (a gate the plan reads)
    import inspect
    src = inspect.getsource(GSH._plan_sig) if hasattr(GSH, '_plan_sig') else open(GSH.__file__, encoding='utf8').read()
    check('every tex_* field is in the plan signature',
          all(f"'{k}'" in src for k in ('tex_format', 'tex_tmem_format', 'tex_compress', 'tex_frac_bits') + WAVE2_FIELDS))
    # the coded-shader Ctx carries opts
    from ..shaders.builtins import Ctx
    check("the coded-shader Ctx carries the pack's opts", hasattr(Ctx(), 'opts'))
    check('the feature matrix carries the three new scenes',
          all(k in FM.SCENES for k in ('textured_white', 'textured_blocks', 'textured_mag')))
    rows = {r[0] for r in FM.ROWS}
    check('the feature matrix carries the pack rows',
          {'texel format RGB565 (Glide)', 'N64 TMEM CI4 budget', 'VQ codebook (Dreamcast)',
           'bilinear 4-bit fraction (Voodoo1)', 'texture POV normalised distance'} <= rows)
    # the bump height pre-pass refuses a level road by name (B0.3)
    src_m = open(MAT.__file__, encoding='utf8').read()
    check('the height pre-pass refuses every footprint filter and level road to the CPU by name',
          'footprint in a height chain' in src_m)


# ---- wave 2 ----
# R251 TEX-2: C077, C083, C072, C079, C022, C008, C088 (appended after wave
# 1's tests; the helpers above gained the Extend wrap, the LOD-field /
# SAT-atlas / hal_recip256 uniforms and frame_bar's allow_px in wave 2)

#: the two-level blends (BLEND, the GS trilinear) carry 1.89.0's own
#: pre-existing 1 ulp: _sample_trilinear's `frac = lod - l0` is float64,
#: blended in float64 and rounded once; the GLSL rounds twice (TEX-1
#: deviation 4, measured 5.96e-8 at FLOAT too). Fixing it would move the
#: 1.89.0 trilinear picture (the identity pin), so the blends hold this bar.
BLEND_ULP = 1.2e-7


def _sat_atlas_extra(tex):
    rec = np.zeros((1, 256, 4), np.float32)
    rec[0, :, 0] = TX.RECIP256
    return {'hal_recip256': TX.Texture(rec, colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')}


def _tex_field(arr):
    return TX.Texture(np.asarray(arr, np.float32), colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')


def _floor_ctx(sc, st, mat=0):
    """(job, gbuf, xx, yy, ctx) of the covered pixels of material `mat`."""
    g, job = rig(sc, st)
    mesh = sc.mesh
    cov = (g.tri >= 0) & (mesh.mat_index[np.maximum(g.tri, 0)] == mat)
    yy, xx = np.nonzero(cov)
    ctx = job.context(g.tri[yy, xx], g.bary[yy, xx], xx, yy)
    return job, g, xx, yy, ctx


def pass_compiles(p):
    """C088 / A12.10: every pass of a plan compiles through the fake
    device with the driver's OWN CreateInfo spec shape (G:4545-4550), so a
    uniform the source declares but `all_samplers` does not list refuses
    by name (FD:135-139) -- hal_recip256 declared once per pass included."""
    from . import fakedevice as FD
    dev = FD.Device()
    for mat_id, name, src, binds in p:
        spec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris']
                + list(binds.get('samplers', ())),
                'floats': ['hal_attr_side', 'hal_slot_count', 'hal_tri_side']
                + list(binds.get('frame_uniforms', ())),
                'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
        prog, err = dev.compile_dynamic(str(name), src, spec)
        check(f"pass '{name}' compiles against the driver's spec (every declared uniform is bound)",
              prog is not None, str(err))


# ------------------------------------------------------------------ C077
def test_gl_clamp_laws():
    """C077: GL_CLAMP clamps the COORDINATE, so the bilinear taps at
    s*w - 0.5 still straddle the edge and read the (0,0,0,0) border: the
    half-texel seam. NEAREST is unaffected (GL 1.1 special-cases s = 1)."""
    t = TX.Texture(np.ones((4, 4, 4), np.float32), colorspace='Non-Color')
    o = dict(TX.SAMPLE_OPTS_NEUTRAL, clamp_mode='GL_CLAMP')
    e = dict(TX.SAMPLE_OPTS_NEUTRAL)

    def f(u, filt, oo, wrap='EXTEND'):
        return t.sample(np.array([u], np.float32), np.array([0.5], np.float32), filt=filt, wrap=wrap, opts=oo)[0]
    check('BILINEAR at u = 1.0 under GL_CLAMP is exactly 0.5 in every channel, alpha included (50% border)',
          bool((f(1.0, 'BILINEAR', o) == 0.5).all()), str(f(1.0, 'BILINEAR', o)))
    check('...and 1.0 under Clamp to Edge', bool((f(1.0, 'BILINEAR', e) == 1.0).all()))
    check('at the last texel centre both modes return 1.0',
          bool((f(1 - 0.5 / 4, 'BILINEAR', o) == 1.0).all() and (f(1 - 0.5 / 4, 'BILINEAR', e) == 1.0).all()))
    check('a quarter texel in from the edge GL_CLAMP returns 0.75 (the seam is half a texel wide)',
          bool((f(1 - 0.25 / 4, 'BILINEAR', o) == 0.75).all()), str(f(1 - 0.25 / 4, 'BILINEAR', o)))
    check('at u = 0.0 GL_CLAMP returns 0.5 (the low edge)', bool((f(0.0, 'BILINEAR', o) == 0.5).all()))
    check('the interior is untouched (u = 0.5 -> 1.0)', bool((f(0.5, 'BILINEAR', o) == 1.0).all()))
    check('NEAREST at u = 1.0 returns 1.0 under both modes (unaffected)',
          bool((f(1.0, 'NEAREST', o) == 1.0).all() and (f(1.0, 'NEAREST', e) == 1.0).all()))
    check('3-Point at u = 1.0 under GL_CLAMP returns 0.5', bool((f(1.0, 'N64_3POINT', o) == 0.5).all()))
    rng = np.random.default_rng(77)
    px = rng.random((5, 7, 4)).astype(np.float32)
    tr = TX.Texture(px, colorspace='Non-Color')
    uv = rng.random((300, 2)).astype(np.float32) * 4 - 1.5
    check('a REPEAT sample is identical under both modes (the mode touches Extend only)',
          bool(np.array_equal(tr.sample(uv[:, 0], uv[:, 1], filt='BILINEAR', wrap='REPEAT', opts=o),
                              tr.sample(uv[:, 0], uv[:, 1], filt='BILINEAR', wrap='REPEAT', opts=e))))
    check('sample_opts reads the clamp mode',
          TX.sample_opts(base_settings(tex_clamp_mode='GL_CLAMP'))['clamp_mode'] == 'GL_CLAMP')
    # A/B on the Extend scene: the seam pulls toward black, and only near the edge
    base = cpu_frame('textured_extend', tex_filter='BILINEAR')
    gl = cpu_frame('textured_extend', tex_filter='BILINEAR', tex_clamp_mode='GL_CLAMP')
    diff = np.abs(base[:, :, :3] - gl[:, :, :3]).max(axis=2) > 0
    check("'tex_clamp_mode' GL_CLAMP changes the textured_extend frame (A/B)", bool(diff.any()),
          f'{int(diff.sum())} px')
    if diff.any():
        check('the mean over the differing pixels is LOWER under GL_CLAMP (the border pulls toward black)',
              float(gl[:, :, :3][diff].mean()) < float(base[:, :, :3][diff].mean()),
              f'{float(gl[:, :, :3][diff].mean()):.4f} vs {float(base[:, :, :3][diff].mean()):.4f}')
        sc, st = scene('textured_extend', tex_filter='BILINEAR')
        _job, g, xx, yy, ctx = _floor_ctx(sc, st)
        floor = np.zeros(diff.shape, bool)
        floor[yy, xx] = True
        check('every differing pixel is a floor pixel', bool((diff & ~floor).sum() == 0),
              f'{int((diff & ~floor).sum())} off the floor')
        sel = diff[yy, xx]
        u, v = ctx.uv[sel, 0], ctx.uv[sel, 1]
        # the [0,1]^2 edge in uv: within one texel (1/8) of it, or past it
        # (Clamp to Edge repeats the last texel there, GL_CLAMP blends 50%)
        edge = np.minimum(np.minimum(np.abs(u), np.abs(1 - u)), np.minimum(np.abs(v), np.abs(1 - v)))
        outside = (u < 0) | (u > 1) | (v < 0) | (v > 1)
        check('the differing pixels lie within one texel of the texture edge (a seam, not the whole floor)',
              bool(((edge <= 0.125 + 1e-3) | outside).all()) and int(sel.sum()) < int(0.5 * yy.size),
              f'{int(sel.sum())} of {yy.size} floor px; farthest {float(edge[~outside].max()) if (~outside).any() else 0:.3f} uv')


def test_gl_clamp_gpu_twin():
    rng = np.random.default_rng(78)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = TX.Texture(px, colorspace='Non-Color')
    uv = (rng.random((500, 2)).astype(np.float32) * 4.0 - 1.5)
    o = dict(TX.SAMPLE_OPTS_NEUTRAL, clamp_mode='GL_CLAMP')
    for filt in ('BILINEAR', 'N64_3POINT', 'NEAREST', 'POV_NORMDIST'):
        src = MAT._texture_sampler('hal_tex0', tex, filt, 'EXTEND', o)
        got, err = sampler_run(src, px, uv)
        if got is None:
            check(f'{filt}/EXTEND + GL_CLAMP sampler compiles', False, err)
            continue
        want = tex.sample(uv[:, 0], uv[:, 1], filt=filt, wrap='EXTEND', opts=o)
        d = float(np.abs(got - want).max())
        check(f'{filt}/EXTEND + GL_CLAMP is bitwise Texture.sample off the grid (the border taps, '
              "the CPU's lerp order)", d == 0.0, f'max {d:.2e}')
    src = MAT._texture_sampler('hal_tex0', tex, 'BILINEAR', 'EXTEND', o)
    check('the BILINEAR GL_CLAMP sampler clamps the coordinate and selects the border',
          'float s = clamp(uv.x, 0.0, 1.0);' in src and '? c10 : vec4(0.0);' in src)
    sc, st = scene('textured_extend', tex_filter='BILINEAR', tex_clamp_mode='GL_CLAMP')
    frame_bar('BILINEAR + GL_CLAMP on textured_extend', st, sc)
    frame_sampler_twin('BILINEAR + GL_CLAMP on textured_extend', st, sc)
    sc, st = scene('textured_extend', tex_filter='TRILINEAR', tex_mipmap=True, tex_clamp_mode='GL_CLAMP')
    frame_bar('TRILINEAR + mips + GL_CLAMP on textured_extend', st, sc)
    frame_sampler_twin('TRILINEAR + mips + GL_CLAMP on textured_extend', st, sc, bar=BLEND_ULP)


# ------------------------------------------------------------------ C083
def _keyed_image():
    px = checker_image(size=16, squares=4)
    yy, xx = np.mgrid[0:16, 0:16]
    px[(xx - 7.5) ** 2 + (yy - 7.5) ** 2 < 16.0, 3] = 0.0
    return px


def test_colorkey_laws():
    """C083: alpha cut-outs are stored as OPAQUE black at prep; the key is
    tested AFTER the filter, so the blended fringe survives (dark), and
    the verdict discards -- a survivor keeps its own alpha."""
    from ..core.mathx import srgb_to_linear
    px = _keyed_image()
    hole = px[:, :, 3] < 0.5
    t = TX.Texture(px.copy(), colorspace='Non-Color').colorkey_prepare()
    check('every hole texel is exactly (0, 0, 0, 1) after prep',
          bool((t.pixels[hole] == np.array([0, 0, 0, 1], np.float32)).all()))
    check('every other texel is untouched bitwise', bool(np.array_equal(t.pixels[~hole], px[~hole])))
    o = dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True)
    yy, xx = np.nonzero(hole)
    u = ((xx + 0.5) / 16).astype(np.float32)
    v = ((yy + 0.5) / 16).astype(np.float32)
    s = t.sample(u, v, filt='BILINEAR', wrap='REPEAT', opts=o)
    check('BILINEAR at every hole centre gives alpha 0 and rgb 0', bool((s == 0.0).all()))
    yy2, xx2 = np.nonzero(~hole)
    s2 = t.sample(((xx2 + 0.5) / 16).astype(np.float32), ((yy2 + 0.5) / 16).astype(np.float32),
                  filt='BILINEAR', wrap='REPEAT', opts=o)
    check('at every lit texel centre alpha is 1 and the colour is the source colour',
          bool((s2[:, 3] == 1.0).all() and np.array_equal(s2[:, :3], px[yy2, xx2, :3])))
    # a hole texel (7, 4) and its lit neighbour (7, 3): the midpoint blends half the neighbour
    assert hole[7, 4] and not hole[7, 3], (hole[7, 4], hole[7, 3])
    mid = t.sample(np.array([(3.5 + 0.5) / 16], np.float32), np.array([7.5 / 16], np.float32),
                   filt='BILINEAR', wrap='REPEAT', opts=o)[0]
    nb = px[7, 3, :3]
    check('at the midpoint between a hole and a lit neighbour the fringe SURVIVES (alpha 1) at half '
          'the neighbour (the dark fringe -- the mechanism)',
          mid[3] == 1.0 and bool(np.allclose(mid[:3], 0.5 * nb, atol=1e-6)), str(mid))
    px2 = px.copy()
    px2[1, 1, 3] = 0.75
    t2 = TX.Texture(px2, colorspace='Non-Color').colorkey_prepare()
    s3 = t2.sample(np.array([1.5 / 16], np.float32), np.array([1.5 / 16], np.float32),
                   filt='NEAREST', wrap='REPEAT', opts=o)[0]
    check("a texel authored at alpha 0.75 keeps 0.75 through prep and through a surviving sample (A6.2)",
          t2.pixels[1, 1, 3] == 0.75 and s3[3] == 0.75, str(s3))
    rng = np.random.default_rng(83)
    uv = rng.random((2000, 2)).astype(np.float32)
    counts = []
    for r in (0, 8, 64, 255):
        sr = t.sample(uv[:, 0], uv[:, 1], filt='BILINEAR', wrap='REPEAT',
                      opts=dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True, colorkey_range=r))
        counts.append(int((sr[:, 3] == 0.0).sum()))
    check('with range 255 everything is keyed', counts[-1] == 2000, str(counts))
    check('the keyed count is non-decreasing in the range (0, 8, 64, 255)',
          all(a <= b for a, b in zip(counts, counts[1:])), str(counts))
    check('colorkey_threshold(0, True) equals srgb_to_linear(0.5/255) bitwise',
          TX.colorkey_threshold(0, True) == np.float32(srgb_to_linear(np.array([0.5 / 255], np.float32))[0]))
    check('colorkey_threshold(0, False) is half a byte', TX.colorkey_threshold(0, False) == np.float32(0.5 / 255))
    check('sample_opts reads the key and its range',
          TX.sample_opts(base_settings(tex_colorkey=True, tex_colorkey_range=9))['colorkey_range'] == 9)
    # the A/B scene: the image's Alpha WIRED, the keyed floor composites over the world (SORTED)
    def ab_frame(**kw):
        st = base_settings(W, H, **kw)
        st.use_processes = False
        return np.asarray(R.render(FM.SCENES['textured_keyed_alpha'](st), st))
    a0 = ab_frame(tex_filter='BILINEAR')
    a1 = ab_frame(tex_filter='BILINEAR', tex_colorkey=True)
    a2 = ab_frame(tex_filter='BILINEAR', tex_colorkey=True, tex_colorkey_range=64)
    check("'tex_colorkey' True changes the textured_keyed_alpha frame (A/B)", not np.array_equal(a0, a1))
    check("'tex_colorkey_range' 64 differs from 0 with the key on (A/B)", not np.array_equal(a1, a2))
    # prep moves the opaque picture too: the fringe is dark where the disc was
    k0 = cpu_frame('textured_keyed', tex_filter='BILINEAR')
    k1 = cpu_frame('textured_keyed', tex_filter='BILINEAR', tex_colorkey=True)
    check('on the opaque keyed scene the prep half moves the picture (blackened holes)', not np.array_equal(k0, k1))


def test_colorkey_gpu_twin():
    rng = np.random.default_rng(84)
    uv = (rng.random((500, 2)).astype(np.float32) * 4.0 - 1.5)
    px = _keyed_image()
    for cs in ('Non-Color', 'Linear'):
        tex = TX.Texture(px.copy(), colorspace=cs).colorkey_prepare()
        for filt, o in (('NEAREST', dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True)),
                        ('BILINEAR', dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True, frac_bits=4)),
                        ('BILINEAR', dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True, frac_bits=4, colorkey_range=64))):
            src = MAT._texture_sampler('hal_tex0', tex, filt, 'REPEAT', o)
            got, err = sampler_run(src, tex.pixels, uv)
            if got is None:
                check(f'{filt} colorkey sampler compiles ({cs})', False, err)
                continue
            want = tex.sample(uv[:, 0], uv[:, 1], filt=filt, wrap='REPEAT', opts=o)
            d = float(np.abs(got - want).max())
            check(f"{filt} + chroma key (range {o['colorkey_range']}, {cs}) is bitwise Texture.sample on all four "
                  'channels (alpha = the verdict)', d == 0.0 and int((want[:, 3] == 0).sum()) > 0,
                  f'max {d:.2e}, {int((want[:, 3] == 0).sum())} keyed')
    src = MAT._texture_sampler('hal_tex0', TX.Texture(px, colorspace='Linear'), 'NEAREST', 'REPEAT',
                               dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True))
    check('the sampler carries the decoded half-byte threshold', '{ c.a = 0.0; }' in src
          and MAT._f(TX.colorkey_threshold(0, True)) in src)
    sc, st = scene('textured_keyed', tex_filter='NEAREST', tex_colorkey=True)
    frame_bar('NEAREST + chroma key on textured_keyed', st, sc)
    frame_sampler_twin('NEAREST + chroma key on textured_keyed', st, sc)
    sc, st = scene('textured_keyed', tex_filter='BILINEAR', tex_frac_bits='BITS_4', tex_colorkey=True)
    frame_bar('BILINEAR + BITS_4 + chroma key on textured_keyed (the FM row)', st, sc)
    frame_sampler_twin('BILINEAR + BITS_4 + chroma key on textured_keyed', st, sc)


# ------------------------------------------------------------------ C072
def _level_texture():
    """A four-level pyramid whose levels are the constants 0.1, 0.2, 0.3,
    0.4: the sampled value NAMES the level."""
    t = TX.Texture(np.full((8, 8, 4), 0.1, np.float32), colorspace='Non-Color')
    t.mips = [np.full((8, 8, 4), 0.1, np.float32), np.full((4, 4, 4), 0.2, np.float32),
              np.full((2, 2, 4), 0.3, np.float32), np.full((1, 1, 4), 0.4, np.float32)]
    return t


def test_mip_select_laws():
    """C072: NEAREST_LEVEL is GL 1.1's ceil(lod + 1/2) - 1 (a tie goes
    DOWN), DITHER_VOODOO is MAME's 8.8 lod + (bayer << 4) >> 8 on the
    screen pixel, BLEND is the RDP's lod_frac lerp over the filter's taps
    -- and BILINEAR + BLEND IS Trilinear."""
    t = _level_texture()
    n = 64
    u = np.linspace(0.05, 0.95, n).astype(np.float32)
    v = np.full(n, 0.5, np.float32)
    d0 = np.zeros((n, 2), np.float32)

    def pick(sel, lod, px=None, py=None, filt='BILINEAR'):
        o = dict(TX.SAMPLE_OPTS_NEUTRAL, mip_select=sel)
        return t.sample(u, v, filt=filt, wrap='REPEAT', lod=np.full(n, lod, np.float32), duv=d0, dvv=d0,
                        opts=o, px=px, py=py)[:, 0]
    check('NEAREST_LEVEL at lod = log2(3) picks level 2 (ceil(2.085) - 1)',
          bool(np.allclose(pick('NEAREST_LEVEL', np.log2(3.0)), 0.3)))
    check('NEAREST_LEVEL at lod = 1.5 picks level 1: the GL tie goes DOWN (named)',
          bool(np.allclose(pick('NEAREST_LEVEL', 1.5), 0.2)))
    check('NEAREST_LEVEL at lod = 1.5001 picks level 2', bool(np.allclose(pick('NEAREST_LEVEL', 1.5001), 0.3)))
    pyy, pxx = np.mgrid[0:4, 0:4]
    o = dict(TX.SAMPLE_OPTS_NEUTRAL, mip_select='DITHER_VOODOO')
    got = t.sample(np.full(16, 0.5, np.float32), np.full(16, 0.5, np.float32), filt='BILINEAR', wrap='REPEAT',
                   lod=np.full(16, 1.5, np.float32), duv=np.zeros((16, 2), np.float32),
                   dvv=np.zeros((16, 2), np.float32), opts=o, px=pxx.ravel(), py=pyy.ravel())[:, 0]
    check('DITHER_VOODOO at lod = 1.5 over one 4x4 block yields levels {1, 2}, eight pixels each '
          '(lod8 = 384; + 16 d >= 512 for d >= 8)',
          int(np.isclose(got, 0.2).sum()) == 8 and int(np.isclose(got, 0.3).sum()) == 8, str(np.round(got, 2)))
    from ..core import dither as DI
    bayer = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32)
    check('BAYER4_16 is the ordinary 4x4 Bayer matrix as integers 0..15 (DI.BAYER4 * 16)',
          bool(np.array_equal(TX.BAYER4_16, bayer) and np.array_equal(TX.BAYER4_16, np.round(DI.BAYER4 * 16))))
    # the bit formula of the GLSL twin, cell by cell
    bits = np.zeros((4, 4), np.float32)
    for by in range(4):
        for bx in range(4):
            xy = bx ^ by
            bits[by, bx] = ((xy & 1) << 3) + ((by & 1) << 2) + (((xy >> 1) & 1) << 1) + ((by >> 1) & 1)
    check("the GLSL twin's bit formula equals DI.BAYER4 * 16 on all 16 cells", bool(np.array_equal(bits, TX.BAYER4_16)))
    # BLEND with N64_3POINT on a real image
    px = checker_image(size=16, squares=4)
    tc = TX.Texture(px, colorspace='Non-Color')
    mips = tc.build_mips()
    rng = np.random.default_rng(72)
    uv = rng.random((500, 2)).astype(np.float32) * 3 - 1
    ob = dict(TX.SAMPLE_OPTS_NEUTRAL, mip_select='BLEND')
    dz = np.zeros((500, 2), np.float32)
    b1 = tc.sample(uv[:, 0], uv[:, 1], filt='N64_3POINT', wrap='REPEAT', lod=np.full(500, 1.0, np.float32),
                   duv=dz, dvv=dz, opts=ob)
    check('BLEND with N64_3POINT at lod = 1.0 equals _sample_3point(mips[1]) bitwise',
          bool(np.array_equal(b1, tc._sample_3point(mips[1], uv[:, 0], uv[:, 1], 'REPEAT'))))
    b15 = tc.sample(uv[:, 0], uv[:, 1], filt='N64_3POINT', wrap='REPEAT', lod=np.full(500, 1.5, np.float32),
                    duv=dz, dvv=dz, opts=ob)
    l1 = tc._sample_3point(mips[1], uv[:, 0], uv[:, 1], 'REPEAT')
    l2 = tc._sample_3point(mips[2], uv[:, 0], uv[:, 1], 'REPEAT')
    check('at lod = 1.5 the blend lies per channel between the level-1 and level-2 samples (lerp law)',
          bool((b15 >= np.minimum(l1, l2) - 1e-6).all() and (b15 <= np.maximum(l1, l2) + 1e-6).all()))
    lods = (rng.random(500) * 3).astype(np.float32)
    tri = tc.sample(uv[:, 0], uv[:, 1], filt='TRILINEAR', wrap='REPEAT', lod=lods, duv=dz, dvv=dz,
                    opts=dict(TX.SAMPLE_OPTS_NEUTRAL))
    bil = tc.sample(uv[:, 0], uv[:, 1], filt='BILINEAR', wrap='REPEAT', lod=lods, duv=dz, dvv=dz, opts=ob)
    check('BILINEAR + BLEND equals TRILINEAR + FILTER bitwise on 500 samples (A9.2, structural)',
          bool(np.array_equal(tri, bil)))
    check('sample_opts with Mipmaps off reports FILTER whatever the dial says (rule 1)',
          TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=False, tex_mip_select='DITHER_VOODOO'))['mip_select'] == 'FILTER')
    # coarser levels are smoother: the NEAREST_LEVEL variance over the bias
    uvg = rng.random((4000, 2)).astype(np.float32)
    dd = np.full((4000, 2), 0.3 / 16, np.float32)
    var = []
    for bias in (0.0, 1.0, 2.0, 3.0):
        s = tc.sample(uvg[:, 0], uvg[:, 1], filt='BILINEAR', wrap='REPEAT', duv=dd, dvv=dd, bias=bias,
                      opts=dict(TX.SAMPLE_OPTS_NEUTRAL, mip_select='NEAREST_LEVEL'))
        var.append(float(s[:, :3].var()))
    check("NEAREST_LEVEL's sampled-image variance is non-increasing over tex_mip_bias 0, 1, 2, 3",
          all(a >= b - 1e-9 for a, b in zip(var, var[1:])), str(np.round(var, 5)))
    # A/B on the demo floor
    n0 = cpu_frame('textured', tex_filter='N64_3POINT', tex_mipmap=True)
    nb = cpu_frame('textured', tex_filter='N64_3POINT', tex_mipmap=True, tex_mip_select='BLEND')
    check("'tex_mip_select' BLEND differs from FILTER under N64_3POINT + mips (A/B)", not np.array_equal(n0, nb))
    f0 = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True)
    fd = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='DITHER_VOODOO')
    fn = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL')
    check("DITHER_VOODOO and NEAREST_LEVEL differ from FILTER and from each other under BILINEAR + mips (A/B)",
          not np.array_equal(f0, fd) and not np.array_equal(f0, fn) and not np.array_equal(fd, fn))
    check('the N64 preset blends its 3-point levels', PRESETS['N64']['settings'].get('tex_mip_select') == 'BLEND')
    check('the VOODOO preset dithers the level pick', PRESETS['VOODOO']['settings'].get('tex_mip_select') == 'DITHER_VOODOO')


def _run_level_sampler(tex, filt, wrap, opts, bias, lod_key, uv, lodfield, uvgrad, vuv):
    """The level-road sampler through the simulator with a 4x4 LOD field
    and a 4x4 derivative field (per-cell inputs), vUV picking the cell."""
    uname = MAT._lod_uniform(lod_key)
    src = ('uniform sampler2D hal_uvgrad;\n' + f'uniform sampler2D {uname};\n'
           + MAT._footprint_sampler('hal_tex0', tex, filt, wrap, opts, bias, lod_key))
    extra = {'hal_uvgrad': _tex_field(uvgrad), uname: _tex_field(lodfield)}
    return sampler_run(src, MAT.mip_atlas(tex)[0], uv, extra, vuv)


def test_mip_select_gpu_twin():
    rng = np.random.default_rng(720)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = TX.Texture(px, colorspace='Non-Color')
    tex.build_mips()
    n = 512
    uv = (rng.random((n, 2)).astype(np.float32) * 4.0 - 1.5)
    i = np.arange(n)
    pxx = i % 4
    pyy = (i // 4) % 4
    vuv = np.stack([(pxx + 0.5) / 4, (pyy + 0.5) / 4], axis=1).astype(np.float32)
    lodf = np.zeros((4, 4, 4), np.float32)
    lodf[:, :, 0] = np.where((np.arange(16).reshape(4, 4) % 2) == 0, np.float32(np.log2(3.0)), np.float32(1.5))
    lodf[1, 2, 0] = 2.5
    lodf[3, 0, 0] = 0.25
    ug = np.zeros((4, 4, 4), np.float32)
    ug[:, :, 0] = 0.02
    ug[:, :, 3] = 0.03
    lod_key = (13, 9)
    for filt in ('BILINEAR', 'NEAREST', 'N64_3POINT'):
        for sel in ('BLEND', 'NEAREST_LEVEL', 'DITHER_VOODOO'):
            o = dict(TX.SAMPLE_OPTS_NEUTRAL, mip_select=sel)
            got, err = _run_level_sampler(tex, filt, 'REPEAT', o, 0.0, lod_key, uv, lodf, ug, vuv)
            if got is None:
                check(f'{filt} x {sel} sampler compiles', False, err)
                continue
            want = tex.sample(uv[:, 0], uv[:, 1], filt=filt, wrap='REPEAT', lod=lodf[pyy, pxx, 0],
                              duv=ug[pyy, pxx, :2], dvv=ug[pyy, pxx, 2:], opts=o, px=pxx, py=pyy)
            d = float(np.abs(got - want).max())
            bar = BLEND_ULP if sel == 'BLEND' else 0.0
            check(f'{filt} x {sel} is {"bitwise" if bar == 0.0 else "within the pre-existing blend ulp of"} '
                  'Texture.sample over 512 uvs and all 16 dither cells', d <= bar, f'max {d:.2e}')
    src = MAT._footprint_sampler('hal_tex0', tex, 'BILINEAR', 'REPEAT',
                                 dict(TX.SAMPLE_OPTS_NEUTRAL, mip_select='DITHER_VOODOO'), 0.0, lod_key)
    check('the Voodoo sampler carries the 8.8 fixed lod line and reads the CPU-decided field',
          'float lod8 = floor(l * 256.0);' in src and 'texelFetch(hal_lod_13x9, gp, 0).r' in src)
    # the derivative-road LOD field IS compute_lod on the context's derivatives
    for persp in (True, False):
        sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL',
                       tex_perspective=persp)
        g, job = rig(sc, st)
        cov = g.tri >= 0
        yy, xx = np.nonzero(cov)
        bl = g.bary_lin[yy, xx] if g.bary_lin is not None else None
        ctx = job.context(g.tri[yy, xx], g.bary[yy, xx], xx, yy, bary_lin=bl)
        want = TX.compute_lod(ctx.duv, ctx.dvv, 64, 64, float(st.tex_mip_bias))
        field = GSH._lod_field(job, g, st, (64, 64))
        check(f'_lod_field under DERIVATIVE equals compute_lod on the context at every covered pixel, bitwise '
              f'(tex_perspective={persp}{"" if persp else ": the affine uv, A12.5"})',
              bool(np.array_equal(field[yy, xx, 0], want.astype(np.float32))),
              f'max {float(np.abs(field[yy, xx, 0] - want).max()):.2e}')
        check(f'_lod_field is cached on the job (tex_perspective={persp})', GSH._lod_field(job, g, st, (64, 64)) is field)
    sc, st = scene('textured', tex_filter='N64_3POINT', tex_mipmap=True, tex_mip_select='BLEND')
    frame_bar('N64_3POINT + mips + BLEND on textured', st, sc)
    frame_sampler_twin('N64_3POINT + mips + BLEND on textured', st, sc, bar=BLEND_ULP)
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='DITHER_VOODOO')
    frame_bar('BILINEAR + mips + DITHER_VOODOO on textured', st, sc)
    frame_sampler_twin('BILINEAR + mips + DITHER_VOODOO on textured', st, sc)
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL')
    frame_bar('BILINEAR + mips + NEAREST_LEVEL on textured', st, sc)
    frame_sampler_twin('BILINEAR + mips + NEAREST_LEVEL on textured', st, sc)


# ------------------------------------------------------------------ C079
def test_tri_lod_laws():
    """C079: one level per TRIANGLE, 0.5 * log2(texel area / screen area)
    + bias -- on a stub job whose projection is a 64x64-pixel quad."""
    import types
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    uvs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)

    def stub(size):
        sx = np.array([10, 10 + size, 10 + size, 10], np.float32)
        sy = np.array([5, 5, 5 + size, 5 + size], np.float32)
        j = types.SimpleNamespace(scene=types.SimpleNamespace(mesh=types.SimpleNamespace(tris=tris, uvs=uvs)))
        j._screen_projection = lambda: (sx, sy, np.ones(4, np.float32))
        return j
    T64 = R.ShadeJob.tri_lod
    l = T64(stub(64), 64, 64)
    check('a 64x64 texture over a 64x64-pixel quad has lod 0.0 on both triangles (one texel per pixel)',
          bool(np.abs(l).max() < 1e-4), str(l))
    check('half the size gives 1.0, double gives -1.0 (0.5 * log2 of a 4x ratio)',
          bool(np.abs(T64(stub(32), 64, 64) - 1.0).max() < 1e-4 and np.abs(T64(stub(128), 64, 64) + 1.0).max() < 1e-4))
    check('bias adds bias exactly', bool(np.array_equal(T64(stub(64), 64, 64, 0.75), (l + np.float32(0.75)).astype(np.float32))))
    check('doubling the texture width adds 0.5', bool(np.abs(T64(stub(64), 128, 64) - 0.5).max() < 1e-5))
    j = stub(64)
    check('the table is cached per (w, h, bias): two calls return the same array',
          T64(j, 64, 64) is T64(j, 64, 64) and T64(j, 64, 64) is not T64(j, 64, 64, 1.0))
    # the demo floor: two triangles with the same UV scale but NOT the same
    # screen area (perspective: the far triangle projects smaller), so the
    # table holds two finite levels, the far one coarser -- the spec's
    # "agree within 1e-4" assumed equal screen areas (deviation, measured
    # 0.31 / 1.98); the band law below is the one the machine drew
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE')
    g, job = rig(sc, st)
    tl = job.tri_lod(64, 64, 0.0)
    floor_tris = np.unique(g.tri[(g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 0)])
    check("the floor's two triangles carry two finite levels, the smaller-on-screen one coarser",
          floor_tris.size == 2 and bool(np.isfinite(tl[floor_tris]).all())
          and float(tl[floor_tris].max()) - float(tl[floor_tris].min()) > 0.1, str(tl[floor_tris]))
    field = GSH._lod_field(job, g, st, (64, 64))
    yy, xx = np.nonzero((g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 0))
    lv = np.clip(-np.floor(-(np.clip(field[yy, xx, 0], 0, 6) + 0.5)) - 1, 0, 6)
    check('every floor pixel under TRIANGLE reads one of at most two levels (the band law)',
          np.unique(lv).size <= 2, str(np.unique(lv)))
    o = TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE'))
    ob = TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE', tex_mip_select='BLEND'))
    od = TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE', tex_mip_select='DITHER_VOODOO'))
    check('rule (4): under TRIANGLE + mips FILTER and BLEND read as NEAREST_LEVEL and DITHER_VOODOO stays',
          o['mip_select'] == 'NEAREST_LEVEL' and ob['mip_select'] == 'NEAREST_LEVEL' and od['mip_select'] == 'DITHER_VOODOO')
    rng = np.random.default_rng(79)
    px = checker_image(size=16, squares=4)
    tc = TX.Texture(px, colorspace='Non-Color')
    uv = rng.random((500, 2)).astype(np.float32) * 3 - 1
    dz = np.zeros((500, 2), np.float32)
    lods = np.full(500, 1.3, np.float32)
    ot = TX.sample_opts(base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE'))
    a = tc.sample(uv[:, 0], uv[:, 1], filt='TRILINEAR', wrap='REPEAT', lod=lods, duv=dz, dvv=dz, opts=ot)
    b = tc.sample(uv[:, 0], uv[:, 1], filt='BILINEAR', wrap='REPEAT', lod=lods, duv=dz, dvv=dz, opts=o)
    check('TRIANGLE + TRILINEAR + FILTER samples bitwise TRIANGLE + BILINEAR + NEAREST_LEVEL (one level, A12.4)',
          bool(np.array_equal(a, b)))
    f0 = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL')
    f1 = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL', tex_lod_source='TRIANGLE')
    check("'tex_lod_source' TRIANGLE differs from DERIVATIVE under BILINEAR + mips + NEAREST_LEVEL (A/B)",
          not np.array_equal(f0, f1))
    check('the VOODOO preset takes the per-triangle level', PRESETS['VOODOO']['settings'].get('tex_lod_source') == 'TRIANGLE')


def test_tri_lod_gpu_twin():
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE')
    g, job = rig(sc, st)
    cov = g.tri >= 0
    field = GSH._lod_field(job, g, st, (64, 64))
    check('_lod_field under TRIANGLE equals job.tri_lod gathered by the G-buffer triangle, bitwise',
          bool(np.array_equal(field[cov, 0], job.tri_lod(64, 64, 0.0)[g.tri[cov]])))
    sc, st = scene('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='TRIANGLE')
    frame_bar('TRILINEAR + mips + TRIANGLE on textured', st, sc)
    frame_sampler_twin('TRILINEAR + mips + TRIANGLE on textured (one bilinear level)', st, sc)
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='DITHER_VOODOO',
                   tex_lod_source='TRIANGLE')
    frame_bar('BILINEAR + mips + DITHER_VOODOO + TRIANGLE on textured (the VOODOO pair)', st, sc)
    frame_sampler_twin('BILINEAR + mips + DITHER_VOODOO + TRIANGLE on textured', st, sc)


# ------------------------------------------------------------------ C022
def test_gs_lod_laws():
    """C022: the GS TEX1 rule on the float32 bit pattern of the view depth."""
    g16 = TX.gs_lod16
    d = np.array([1.0, 2.0, 3.0, 0.5], np.float32)
    check('gs_lod16 at depth 1, 2, 3, 0.5 with K = 0 gives 0, 1, 1.5, -1 levels',
          bool(np.array_equal(g16(d, 0, 0), np.array([0.0, 1.0, 1.5, -1.0], np.float32))), str(g16(d, 0, 0)))
    check('L = 1 doubles (3.0 -> 3.0 levels at K = 0)', float(g16(np.array([3.0], np.float32), 0, 1)[0]) == 3.0)
    check('K = -1.0 (k16 = -16) subtracts one level', float(g16(np.array([1.0], np.float32), -16, 0)[0]) == -1.0)
    check('K = 0.03125 quantises to 0 and 0.09375 to 2/16 (half-to-even, named)',
          TX.sample_opts(base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_k=0.03125))['lod_k16'] == 0
          and TX.sample_opts(base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_k=0.09375))['lod_k16'] == 2)
    band = np.linspace(3.0, 3.125, 100, endpoint=False).astype(np.float32)
    check('the lod is a step function of depth: 100 depths inside [3.0, 3.125) give one value',
          np.unique(g16(band, 0, 0)).size == 1)
    ramp = np.linspace(0.1, 50.0, 5000).astype(np.float32)
    check('the sequence over increasing depths is non-decreasing (monotonic)', bool((np.diff(g16(ramp, 0, 0)) >= 0).all()))
    px = checker_image(size=16, squares=4)
    tc = TX.Texture(px, colorspace='Non-Color')
    rng = np.random.default_rng(22)
    uv = rng.random((400, 2)).astype(np.float32)
    dep = np.full(400, 6.3, np.float32)
    o = TX.sample_opts(base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='GS_Q', tex_lod_k=-2.0))
    da = np.full((400, 2), 0.01, np.float32)
    db = np.full((400, 2), 0.2, np.float32)
    sa = tc.sample(uv[:, 0], uv[:, 1], filt='TRILINEAR', wrap='REPEAT', lod=TX.compute_lod(da, da, 16, 16), duv=da, dvv=da, depth=dep, opts=o)
    sb = tc.sample(uv[:, 0], uv[:, 1], filt='TRILINEAR', wrap='REPEAT', lod=TX.compute_lod(db, db, 16, 16), duv=db, dvv=db, depth=dep, opts=o)
    check('angle independence: GS_Q samples bitwise the same texels for two derivative sets at one depth',
          bool(np.array_equal(sa, sb)))
    check('sample_opts under GS_Q reports aniso == 1 whatever tex_aniso says (rule 3)',
          TX.sample_opts(base_settings(tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='GS_Q', tex_aniso=8))['aniso'] == 1)
    f0 = cpu_frame('textured', tex_filter='TRILINEAR', tex_mipmap=True)
    f1 = cpu_frame('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='GS_Q', tex_lod_k=-2.0)
    check("'tex_lod_source' GS_Q (K = -2) differs from DERIVATIVE under TRILINEAR + mips (A/B)", not np.array_equal(f0, f1))
    f2 = cpu_frame('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='GS_Q', tex_lod_k=-3.0)
    f3 = cpu_frame('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='GS_Q', tex_lod_k=-2.0, tex_lod_l=1)
    check("'tex_lod_k' -3 and 'tex_lod_l' 1 each move the GS picture (A/B)",
          not np.array_equal(f1, f2) and not np.array_equal(f1, f3))
    b0 = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL')
    b1 = cpu_frame('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL', tex_lod_source='GS_Q', tex_lod_k=-2.0)
    check('GS_Q differs from DERIVATIVE under BILINEAR + mips + NEAREST_LEVEL too (A/B)', not np.array_equal(b0, b1))
    ps2 = PRESETS['PS2']['settings']
    check('the PS2 preset takes the GS rule: GS_Q, K = -2, L = 0, one nearest level, no tex_mip_bias',
          ps2.get('tex_lod_source') == 'GS_Q' and ps2.get('tex_lod_k') == -2.0 and ps2.get('tex_lod_l') == 0
          and ps2.get('tex_mip_select') == 'NEAREST_LEVEL' and 'tex_mip_bias' not in ps2)


def test_gs_lod_gpu_twin():
    sc, st = scene('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source='GS_Q', tex_lod_k=-2.0)
    g, job = rig(sc, st)
    cov = g.tri >= 0
    yy, xx = np.nonzero(cov)
    ctx = job.context(g.tri[yy, xx], g.bary[yy, xx], xx, yy)
    o = TX.sample_opts(st)
    field = GSH._lod_field(job, g, st, (0, 0))
    check('_lod_field (0, 0) equals gs_lod16(ctx.depth) at every covered pixel, bitwise',
          bool(np.array_equal(field[yy, xx, 0], TX.gs_lod16(ctx.depth, o['lod_k16'], o['lod_l']))))
    frame_bar('TRILINEAR + mips + GS_Q (K = -2) on textured', st, sc)
    frame_sampler_twin('TRILINEAR + mips + GS_Q (K = -2) on textured', st, sc, bar=BLEND_ULP)
    sc, st = scene('textured', tex_filter='BILINEAR', tex_mipmap=True, tex_mip_select='NEAREST_LEVEL',
                   tex_lod_source='GS_Q', tex_lod_k=-2.0)
    frame_bar('BILINEAR + mips + NEAREST_LEVEL + GS_Q on textured', st, sc)
    frame_sampler_twin('BILINEAR + mips + NEAREST_LEVEL + GS_Q on textured', st, sc)
    # the fake device: the hal_lodq upload is a delta of ONE over the same frame under DERIVATIVE
    from . import fakedevice as FD
    ups = {}
    for src_name in ('DERIVATIVE', 'GS_Q'):
        sc, st = scene('textured', tex_filter='TRILINEAR', tex_mipmap=True, tex_lod_source=src_name, tex_lod_k=-2.0)
        st.render_device = 'GPU'
        g, job = rig(sc, st)
        GSH._PLAN_CACHE.clear()
        with FD.installed() as dev:
            img, why = GSH.shade_frame(job, g)
            check(f'the fake device shades the {src_name} frame', img is not None, str(why))
            ups[src_name] = dev.calls['upload']
    check('the hal_lodq field is ONE upload more than the same frame under DERIVATIVE (A12.14)',
          ups['GS_Q'] - ups['DERIVATIVE'] == 1, str(ups))


# ------------------------------------------------------------------ C008
def test_sharpen_laws():
    """C008: G_TD_SHARPEN -- under magnification level 0 is extrapolated
    AWAY from level 1 with a negative fraction and the 9-bit clamp
    saturates the overshoot."""
    check('clamp9: 0, 255, 300, 400, 408, -10, -200 -> 0, 255, 255, 0, 0, 0, 255 (the RDP wraps)',
          list(TX.clamp9(np.array([0, 255, 300, 400, 408, -10, -200]))) == [0, 255, 255, 0, 0, 0, 255])
    yy, xx = np.mgrid[0:8, 0:8]
    chk = np.zeros((8, 8, 4), np.float32)
    chk[:, :, :3] = ((xx + yy) & 1)[:, :, None]
    chk[:, :, 3] = 1.0
    t = TX.Texture(chk, colorspace='Non-Color')
    t.mips = [chk, np.full((4, 4, 4), 0.5, np.float32)]
    o = TX.sample_opts(base_settings(tex_filter='NEAREST', tex_mipmap=True, tex_lod_sharpen=True))
    u = ((xx.ravel() + 0.5) / 8).astype(np.float32)
    v = ((yy.ravel() + 0.5) / 8).astype(np.float32)

    def dm(m, n=64):
        d = np.zeros((n, 2), np.float32)
        d[:, 0] = m / 8
        return d
    s = t.sample(u, v, filt='NEAREST', wrap='REPEAT', duv=dm(0.5), dvv=dm(0.5), opts=o)
    ones = ((xx + yy) & 1).ravel() == 1
    check('NEAREST at texel centres with m = 0.5: the 1-texels saturate to 1.0 (v = 1.25 -> 319 -> 255)',
          bool((s[ones, 0] == 1.0).all()), str(np.unique(s[ones, 0])))
    check('...and the 0-texels to 0.0 (v = -0.25 -> -64 -> 0)', bool((s[~ones, 0] == 0.0).all()), str(np.unique(s[~ones, 0])))
    ob = TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_lod_sharpen=True))
    mid = t.sample(np.array([1.0 / 8], np.float32), np.array([0.5 / 8], np.float32), filt='BILINEAR', wrap='REPEAT',
                   duv=dm(0.5, 1), dvv=dm(0.5, 1), opts=ob)[0, 0]
    check('a BILINEAR midpoint (0.5 at both levels) is a fixed point up to the 8-bit tail (127.5 -> 128 -> 128/255)',
          abs(float(mid) - 0.5) <= 0.5 / 255 + 1e-7, str(mid))
    t2 = TX.Texture(np.full((8, 8, 4), 0.75, np.float32), colorspace='Non-Color')
    t2.mips = [np.full((8, 8, 4), 0.75, np.float32), np.full((4, 4, 4), 0.5, np.float32)]
    vals = []
    for m in (0.0, 0.25, 0.5, 0.75, 0.999):
        vals.append(float(t2.sample(np.array([0.3], np.float32), np.array([0.3], np.float32), filt='NEAREST',
                                    wrap='REPEAT', duv=dm(m, 1), dvv=dm(m, 1), opts=o)[0, 0]))
    check('a texel of 0.75 over 0.5 sharpens to 1.0 at m = 0 and within 1/255 of 0.75 at m = 0.999 (none at 1:1)',
          vals[0] == 1.0 and abs(vals[-1] - 0.75) <= 1.0 / 255, str(vals))
    check('monotone in m over 0, 0.25, 0.5, 0.75, 0.999', all(a >= b for a, b in zip(vals, vals[1:])), str(vals))
    plain = t2.sample(np.array([0.3], np.float32), np.array([0.3], np.float32), filt='NEAREST', wrap='REPEAT',
                      duv=dm(1.5, 1), dvv=dm(1.5, 1), opts=dict(TX.SAMPLE_OPTS_NEUTRAL))
    sharp = t2.sample(np.array([0.3], np.float32), np.array([0.3], np.float32), filt='NEAREST', wrap='REPEAT',
                      duv=dm(1.5, 1), dvv=dm(1.5, 1), opts=o)
    check('m >= 1 returns the un-sharpened sample bitwise', bool(np.array_equal(plain, sharp)))
    t1 = TX.Texture(np.full((1, 1, 4), 0.3, np.float32), colorspace='Non-Color')
    s1 = t1.sample(np.array([0.3], np.float32), np.array([0.3], np.float32), filt='NEAREST', wrap='REPEAT',
                   duv=dm(0.2, 1), dvv=dm(0.2, 1), opts=o)
    check('a 1x1 texture is untouched (no level 1)', bool((s1 == np.float32(0.3)).all()))
    check('sample_opts with Mipmaps off reports lod_sharpen False (rule 1)',
          not TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=False, tex_lod_sharpen=True))['lod_sharpen'])
    f0 = cpu_frame('textured_mag', tex_filter='BILINEAR', tex_mipmap=True)
    f1 = cpu_frame('textured_mag', tex_filter='BILINEAR', tex_mipmap=True, tex_lod_sharpen=True)
    check("'tex_lod_sharpen' changes the magnified BILINEAR + mips frame (A/B)", not np.array_equal(f0, f1))
    sc, st = scene('textured_mag', tex_filter='BILINEAR', tex_mipmap=True)
    _job, g, xx, yy, _ctx = _floor_ctx(sc, st)
    check('the standard deviation of the floor pixels rises under sharpen (extrapolation raises contrast)',
          float(f1[yy, xx, :3].std()) >= float(f0[yy, xx, :3].std()),
          f'{float(f1[yy, xx, :3].std()):.4f} vs {float(f0[yy, xx, :3].std()):.4f}')
    check("the N64 preset names Sharpen in its note", 'Sharpen' in PRESETS['N64']['note'])


def test_sharpen_gpu_twin():
    rng = np.random.default_rng(8)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = TX.Texture(px, colorspace='Non-Color')
    tex.build_mips()
    n = 500
    uv = (rng.random((n, 2)).astype(np.float32) * 4.0 - 1.5)
    o = TX.sample_opts(base_settings(tex_filter='BILINEAR', tex_mipmap=True, tex_lod_sharpen=True))
    for m in (0.5, 0.9, 1.5):
        ug = np.zeros((1, 1, 4), np.float32)
        ug[0, 0, 0] = m / 13          # |du/dx| * W = m
        ug[0, 0, 3] = 0.3 * m / 9
        lodf = np.zeros((1, 1, 4), np.float32)
        for filt in ('NEAREST', 'BILINEAR', 'N64_3POINT'):
            got, err = _run_level_sampler(tex, filt, 'REPEAT', o, 0.0, (13, 9), uv, lodf, ug, None)
            if got is None:
                check(f'{filt} sharpen sampler compiles (m = {m})', False, err)
                continue
            duv = np.tile(ug[0, 0, :2], (n, 1))
            dvv = np.tile(ug[0, 0, 2:], (n, 1))
            want = tex.sample(uv[:, 0], uv[:, 1], filt=filt, wrap='REPEAT', duv=duv, dvv=dvv, opts=o)
            d = float(np.abs(got - want).max())
            check(f'{filt} + sharpen at m = {m} is bitwise Texture.sample over 500 uvs', d == 0.0, f'max {d:.2e}')
    src = MAT._footprint_sampler('hal_tex0', tex, 'N64_3POINT', 'REPEAT', o, 0.0, (13, 9))
    check('the sharpen sampler carries the 9-bit clamp and roundEven', 'float hal_clamp9_' in src and 'roundEven(v255)' in src)
    sc, st = scene('textured_mag', tex_filter='N64_3POINT', tex_mipmap=True, tex_lod_sharpen=True)
    frame_bar('N64_3POINT + mips + sharpen on textured_mag', st, sc)
    frame_sampler_twin('N64_3POINT + mips + sharpen on textured_mag', st, sc)


# ------------------------------------------------------------------ C088
def test_sat_laws():
    """C088: Crow's summed-area box -- the integer table, the clamped
    rectangle, the continuous mirror, the 1:1 law, the 2^32 window."""
    rng = np.random.default_rng(88)
    px = rng.random((64, 64, 4)).astype(np.float32)
    t = TX.Texture(px, colorspace='Non-Color')
    hi, lo = t.build_sat()
    T16 = np.clip(np.round(px * np.float32(65535.0)), 0, 65535).astype(np.uint64)
    S = T16.cumsum(axis=0).cumsum(axis=1) & np.uint64(0xFFFFFFFF)
    check('hi * 65536 + lo reconstructs the cumsum modulo 2^32 exactly (integer check)',
          bool(np.array_equal(hi.astype(np.uint64) * np.uint64(65536) + lo.astype(np.uint64), S)))
    o = dict(TX.SAMPLE_OPTS_NEUTRAL)
    uv = rng.random((500, 2)).astype(np.float32) * 4 - 1.5
    for wrap in WRAPS:
        a = t.sample(uv[:, 0], uv[:, 1], filt='SUMMED_AREA', wrap=wrap, opts=o)
        b = t.sample(uv[:, 0], uv[:, 1], filt='NEAREST', wrap=wrap, opts=o)
        check(f'without a footprint SUMMED_AREA/{wrap} == NEAREST bitwise (a box of one IS the nearest texel)',
              bool(np.array_equal(a, b)))
    # hu = hv = 2.5: the box against the float64 mean of T16 over the clamped rectangle
    d25 = np.full((14, 2), 2.5 / 64, np.float32)          # (|du|+|dv|) * 0.5 * 64 = 2.5
    pts = [(10.3, 20.7), (33.5, 17.5), (50.2, 40.9), (7.8, 60.1), (44.4, 44.4), (21.0, 9.0), (58.6, 12.2),
           (30.1, 55.5), (15.5, 35.5), (47.9, 26.3), (0.4, 0.4), (63.6, 0.3), (0.2, 63.7), (63.8, 63.9)]
    u = np.array([p[0] / 64 for p in pts], np.float32)
    v = np.array([p[1] / 64 for p in pts], np.float32)
    got = t.sample(u, v, filt='SUMMED_AREA', wrap='EXTEND', duv=d25, dvv=d25, opts=o)
    worst = 0.0
    for k, (uc, vc) in enumerate(pts):
        x0 = max(int(np.floor(uc - 2.5)), 0)
        x1 = min(int(np.ceil(uc + 2.5)) - 1, 63)
        y0 = max(int(np.floor(vc - 2.5)), 0)
        y1 = min(int(np.ceil(vc + 2.5)) - 1, 63)
        mean = T16[y0:y1 + 1, x0:x1 + 1].astype(np.float64).mean(axis=(0, 1)) / 65535.0
        worst = max(worst, float(np.abs(got[k] - mean).max()))
    check('at ten interior and four edge/corner points the box equals the float64 mean of T16 over the '
          'clamped rectangle within 1e-5 (the zero row/column, the edge clamp)', worst < 1e-5, f'max {worst:.2e}')
    # the continuous mirror: at u = 1 + 0.3/w with hu = 0.5 the box is NEAREST's texel w-1 (B8.1)
    d05 = np.zeros((1, 2), np.float32)
    um = np.array([1 + 0.3 / 64], np.float32)
    vm = np.array([32.5 / 64], np.float32)          # a texel centre in v: the box is one texel high
    ms = t.sample(um, vm, filt='SUMMED_AREA', wrap='MIRROR', duv=d05, dvv=d05, opts=o)
    mn = t.sample(um, vm, filt='NEAREST', wrap='MIRROR', opts=o)
    check('under MIRROR at u = 1 + 0.3/w with hu = 0.5 the box is the texel NEAREST reads (w-1)',
          bool(np.abs(ms - mn).max() < 2e-5), f'max {float(np.abs(ms - mn).max()):.2e}')
    uu = (1.05 + rng.random(200) * 0.85).astype(np.float32)
    vv = (0.2 + rng.random(200) * 0.6).astype(np.float32)
    dmir = (rng.random((200, 2)) * 1.5 / 64).astype(np.float32)
    m1 = t.sample(uu, vv, filt='SUMMED_AREA', wrap='MIRROR', duv=dmir, dvv=dmir, opts=o)
    m2 = t.sample((2.0 - uu).astype(np.float32), vv, filt='SUMMED_AREA', wrap='MIRROR', duv=dmir, dvv=dmir, opts=o)
    check('a mirrored box is the mirror image of its unmirrored twin (u -> 2 - u), bitwise',
          bool(np.array_equal(m1, m2)), f'max {float(np.abs(m1 - m2).max()):.2e}')
    chk = TX.Texture(checker_image(), colorspace='Non-Color')
    uvg = rng.random((3000, 2)).astype(np.float32)
    dg = np.full((3000, 2), 0.6 / 64, np.float32)
    var = [float(chk.sample(uvg[:, 0], uvg[:, 1], filt='SUMMED_AREA', wrap='REPEAT', duv=dg, dvv=dg, bias=b, opts=o)[:, :3].var())
           for b in (0.0, 1.0, 2.0, 3.0)]
    check("on the checker the sampled image's variance is non-increasing over tex_mip_bias 0, 1, 2, 3",
          all(a >= b - 1e-9 for a, b in zip(var, var[1:])), str(np.round(var, 5)))
    # the 2^32 window: a 512-wide texture, hu = 1000 clamps to 127
    big = rng.random((8, 512, 4)).astype(np.float32)
    tb = TX.Texture(big, colorspace='Non-Color')
    T16b = np.clip(np.round(big * np.float32(65535.0)), 0, 65535).astype(np.uint64)
    dbig = np.full((2, 2), 1000.0 / 512, np.float32)        # (|du|+|dv|) * 0.5 * 512 = 1000 -> 127
    gb = tb.sample(np.array([0.5, 256.5 / 512], np.float32), np.array([4.5 / 8, 4.5 / 8], np.float32),
                   filt='SUMMED_AREA', wrap='EXTEND', duv=dbig, dvv=np.zeros((2, 2), np.float32), opts=o)
    e1 = T16b[4:5, 129:383].astype(np.float64).mean(axis=(0, 1)) / 65535.0     # uc = 256: [129, 382], 254 texels
    e2 = T16b[4:5, 129:384].astype(np.float64).mean(axis=(0, 1)) / 65535.0     # uc = 256.5: [129, 383], 255 texels
    check('hu = 1000 clamps to 127: at uc = 256 the box is [129, 382] (254 texels), at uc = 256.5 [129, 383] (255)',
          float(np.abs(gb[0] - e1).max()) < 1e-5 and float(np.abs(gb[1] - e2).max()) < 1e-5,
          f'{float(np.abs(gb[0] - e1).max()):.2e} {float(np.abs(gb[1] - e2).max()):.2e}')
    # the 1:1 law (A12.9)
    yy, xx = np.mgrid[0:64, 0:64]
    uc = ((xx.ravel() + 0.5) / 64).astype(np.float32)
    vc = ((yy.ravel() + 0.5) / 64).astype(np.float32)
    dz = np.zeros((4096, 2), np.float32)
    one = t.sample(uc, vc, filt='SUMMED_AREA', wrap='REPEAT', duv=dz, dvv=dz, opts=o)
    check('at every texel centre with hu = hv = 0.5 the box is ONE texel and the output is T16/65535 (within 1e-6)',
          float(np.abs(one - T16[yy.ravel(), xx.ravel()].astype(np.float64) / 65535.0).max()) < 1e-6)
    u3 = ((xx.ravel()[65:] + 0.3) / 64).astype(np.float32)
    v3 = ((yy.ravel()[65:] + 0.5) / 64).astype(np.float32)
    two = t.sample(u3, v3, filt='SUMMED_AREA', wrap='REPEAT', duv=dz[65:], dvv=dz[65:], opts=o)
    ii = xx.ravel()[65:]
    jj = yy.ravel()[65:]
    sel = ii >= 1
    exp2 = (T16[jj[sel], ii[sel] - 1].astype(np.float64) + T16[jj[sel], ii[sel]].astype(np.float64)) / 2 / 65535.0
    check('at u = (i + 0.3)/w it reads texels i-1 and i (the box crosses one edge)',
          float(np.abs(two[sel] - exp2).max()) < 1e-6)
    const = TX.Texture(np.full((16, 16, 4), 0.37, np.float32), colorspace='Non-Color')
    cs = const.sample(uv[:, 0], uv[:, 1], filt='SUMMED_AREA', wrap='REPEAT', duv=np.full((500, 2), 0.1, np.float32),
                      dvv=np.full((500, 2), 0.07, np.float32), opts=o)
    check('a constant texture returns the constant within 2/65535', float(np.abs(cs - np.float32(0.37)).max()) < 2.0 / 65535)
    f0 = cpu_frame('textured')
    f1 = cpu_frame('textured', tex_filter='SUMMED_AREA')
    f2 = cpu_frame('textured', tex_filter='TRILINEAR', tex_mipmap=True)
    check("'tex_filter' SUMMED_AREA differs from NEAREST and from TRILINEAR + mips (A/B)",
          not np.array_equal(f0, f1) and not np.array_equal(f1, f2))
    check('SUMMED_AREA wants the footprint and has no pyramid (rule 2)',
          TX.footprint_wanted(base_settings(tex_filter='SUMMED_AREA'))
          and TX.sample_opts(base_settings(tex_filter='SUMMED_AREA', tex_mipmap=True, tex_mip_select='BLEND'))['mip_select'] == 'FILTER')
    check('the SUMMED_AREA item is in the properties enum', any(i[0] == 'SUMMED_AREA' for i in _tex_filter_items()))
    check('STUDIO_R4 and MAX_R2 name Summed Area in their notes',
          'Summed Area' in PRESETS['STUDIO_R4']['note'] and 'Summed Area' in PRESETS['MAX_R2']['note'])


def _two_sat_scene(st):
    """The demo textured floor with TWO images mixed in one material."""
    sc = FM.SCENES['textured'](st)
    px2 = checker_image(size=32, squares=4, a=(0.2, 0.8, 0.3), b=(0.9, 0.3, 0.2))
    from ..core.scene import ImageBuffer
    sc.images['checker2'] = ImageBuffer(name='checker2', pixels=px2)
    nodes = sc.materials[0].graph['nodes']
    nodes['tex2'] = {'id': 'tex2', 'bl_idname': 'ShaderNodeTexImage',
                     'props': {'image': 'checker2', 'interpolation': 'Closest'},
                     'inputs': [{'name': 'Vector', 'type': 'VECTOR', 'default': [0, 0, 0], 'link': None}],
                     'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]}
    nodes['mix'] = {'id': 'mix', 'bl_idname': 'ShaderNodeMixRGB', 'props': {'blend_type': 'MIX'},
                    'inputs': [{'name': 'Fac', 'type': 'VALUE', 'default': 0.5, 'link': None},
                               {'name': 'Color1', 'type': 'RGBA', 'default': [1, 1, 1, 1], 'link': ['tex', 0]},
                               {'name': 'Color2', 'type': 'RGBA', 'default': [1, 1, 1, 1], 'link': ['tex2', 0]}],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'}]}
    nodes['bsdf']['inputs'][0]['link'] = ['mix', 0]
    return sc


def test_sat_gpu_twin():
    rng = np.random.default_rng(880)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = TX.Texture(px, colorspace='Non-Color')
    n = 500
    uv = (rng.random((n, 2)).astype(np.float32) * 4.0 - 1.5)
    o = dict(TX.SAMPLE_OPTS_NEUTRAL)
    atlas = MAT.sat_atlas(tex)
    check('the SAT atlas is (H, 2W, 4): hi left, lo right', atlas.shape == (13, 18, 4)
          and np.array_equal(atlas[:, :9], tex.sat[0]) and np.array_equal(atlas[:, 9:], tex.sat[1]))
    for gx in ((0.0, 0.0, 0.0, 0.0), (0.02, 0.0, 0.0, 0.02), (0.1, 0.05, 0.05, 0.1)):
        ug = np.zeros((1, 1, 4), np.float32)
        ug[0, 0] = gx
        for wrap in WRAPS:
            for bias in (0.0, 1.5):
                src = ('uniform sampler2D hal_uvgrad;\nuniform sampler2D hal_recip256;\n'
                       + MAT._sat_sampler('hal_tex0', tex, wrap, o, bias))
                extra = dict(_sat_atlas_extra(tex), hal_uvgrad=_tex_field(ug))
                got, err = sampler_run(src, atlas, uv, extra, np.zeros((n, 2), np.float32))
                if got is None:
                    check(f'SUMMED_AREA/{wrap} sampler compiles', False, err)
                    continue
                want = tex.sample(uv[:, 0], uv[:, 1], filt='SUMMED_AREA', wrap=wrap, bias=bias,
                                  duv=np.tile(np.asarray(gx[:2], np.float32), (n, 1)),
                                  dvv=np.tile(np.asarray(gx[2:], np.float32), (n, 1)), opts=o)
                d = float(np.abs(got - want).max())
                check(f'SUMMED_AREA/{wrap} g={gx} bias {bias} is bitwise Texture.sample over 500 uvs (integer table, '
                      'hal_recip256, no division)', d == 0.0, f'max {d:.2e}')
    check('SUMMED_AREA is a supported footprint filter with NEAREST as its no-footprint road',
          'SUMMED_AREA' in MAT.SUPPORTED_TEX_FILTERS and 'SUMMED_AREA' in MAT.FOOTPRINT_TEX_FILTERS
          and MAT.NOFOOTPRINT_FILTER['SUMMED_AREA'] == 'NEAREST')
    sc, st = scene('textured', tex_filter='SUMMED_AREA')
    out = frame_bar('SUMMED_AREA on textured (1:1 boxes)', st, sc, allow_px=1)
    frame_sampler_twin('SUMMED_AREA on textured', st, sc)
    sc, st = scene('textured', tex_filter='SUMMED_AREA', tex_mip_bias=2.0)
    frame_bar('SUMMED_AREA + bias 2 on textured', st, sc)
    frame_sampler_twin('SUMMED_AREA + bias 2 on textured', st, sc)
    # two SAT images in one material: hal_recip256 declared ONCE, every uniform bound (A12.1, A12.10)
    st = base_settings(W, H, transparency='NONE', tex_filter='SUMMED_AREA')
    st.use_processes = False
    sc = _two_sat_scene(st)
    out, cpu, g, p, why = twin(st, sc)
    check('a material with two SAT-filtered images plans', p is not None, str(why))
    if p is not None:
        srcs = [s for _m, _n, s, _b in p]
        floor_src = next((s for s in srcs if 'hal_satfetch_' in s), '')
        check('hal_recip256 is declared once in the pass that samples two SAT images',
              floor_src.count('uniform sampler2D hal_recip256;') == 1 and floor_src.count('hal_satfetch_') >= 2)
        pass_compiles(p)
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
        check('the two-SAT frame is the CPU frame at the deferred bar (up to one texel-edge pixel)',
              int((d >= 6e-3).sum()) <= 1, f'max {float(d.max()):.2e}')
    # the refusal by name: the layer passes have no footprint
    sc, st = FM.build('sorted glass layers')
    st.tex_filter = 'SUMMED_AREA'
    st.use_processes = False
    donor = FM.SCENES['textured'](st)
    sc.materials[1].graph = donor.materials[0].graph
    sc.images.update(donor.images)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, st.resolution_x, st.resolution_y)
    opq, _trn = R._split_by_alpha(sc, sc.mesh, st)
    g = CRg.GBuffer(st.resolution_x, st.resolution_y)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, st.resolution_x, st.resolution_y, subset=opq, gbuf=g,
                  depth_bits=st.depth_precision)
    R._build_shadows(sc, st, sc.mesh)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye, st.resolution_x, st.resolution_y)
    GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, g)
    lwhy = str((atl or {}).get('__layers_why', ''))
    check('the SUMMED_AREA footprint refuses the layer passes BY NAME and the opaque frame still plans',
          passes is not None and 'layer passes' in lwhy, f'{why} / {lwhy}')


# ------------------------------------------------------ wave-2 plumbing
def test_wave2_plumbing():
    """The central edits of wave 2: the enum tables, the UI rows, the FM
    scenes and rows, the presets, the emission site."""
    try:
        from . import fakebpy
        fakebpy.install()
        from .. import properties as PR
        check('TEX_CLAMP_MODE / TEX_MIP_SELECT / TEX_LOD_SOURCE are enum tables of their fields',
              PR.ENUMS.get('tex_clamp_mode') is PR.TEX_CLAMP_MODE and PR.ENUMS.get('tex_mip_select') is PR.TEX_MIP_SELECT
              and PR.ENUMS.get('tex_lod_source') is PR.TEX_LOD_SOURCE)
        check('the item lists carry the machines', [i[0] for i in PR.TEX_MIP_SELECT] == ['FILTER', 'BLEND', 'NEAREST_LEVEL', 'DITHER_VOODOO']
              and [i[0] for i in PR.TEX_LOD_SOURCE] == ['DERIVATIVE', 'GS_Q', 'TRIANGLE']
              and [i[0] for i in PR.TEX_CLAMP_MODE] == ['EDGE', 'GL_CLAMP'])
        check('SUMMED_AREA is the LAST tex_filter item (positional numbering)', PR.TEX_FILTER[-1][0] == 'SUMMED_AREA')
        check('tex_mip_bias says where it is live and inert',
              'Summed Area' in PR.DESCRIPTIONS['tex_mip_bias'] and 'PlayStation 2' in PR.DESCRIPTIONS['tex_mip_bias'])
        for f in WAVE2_FIELDS:
            check(f'{f} carries a tooltip of at least 40 characters naming the mechanism', len(PR.DESCRIPTIONS.get(f, '')) >= 40)
    except Exception:                                           # noqa: BLE001
        traceback.print_exc()
        check('properties import under fakebpy', False)
    import os
    ui_src = open(os.path.join(os.path.dirname(MAT.__file__), '..', 'ui.py'), encoding='utf8').read()
    for f in WAVE2_FIELDS:
        check(f'the Textures panel draws {f}', f"prop(hs, '{f}')" in ui_src
              and f"# sub.prop(hs, '{f}')" not in ui_src and f"# col.prop(hs, '{f}')" not in ui_src)
    check('the feature matrix carries the wave-2 scenes',
          all(k in FM.SCENES for k in ('textured_extend', 'textured_keyed', 'textured_keyed_alpha')))
    rows = {r[0] for r in FM.ROWS}
    want_rows = {'GL_CLAMP border seam (OpenGL 1.1)', 'chroma key after filtering (Glide)', 'N64 3-point mip blend',
                 'nearest mip level (GL 1.1)', 'Voodoo1 LOD dither', 'per-polygon mip level (Riva 128)',
                 'PS2 LOD from Q, nearest level', 'PS2 LOD from Q, trilinear', 'N64 sharpen (G_TD_SHARPEN)',
                 'texture Summed Area (3DS/Max)', 'Summed Area footprint x4'}
    check('the feature matrix carries the eleven wave-2 rows', want_rows <= rows, str(sorted(want_rows - rows)))
    for key in sorted(want_rows):
        sc, st = FM.build(key)
        st.use_processes = False
        ok = True
        try:
            img = np.asarray(R.render(sc, st))
            ok = img.shape[0] == st.resolution_y and np.isfinite(img).all()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            ok = False
        check(f"FM row '{key}' renders a finite frame", ok)
    src_m = open(MAT.__file__, encoding='utf8').read()
    check('the emission site no longer refuses the level roads by name (the samplers are in)',
          'is not in the deferred pass yet\')' not in src_m.split('def _assemble_height_pass')[-1].split('lod_key = _lod_field_key')[1][:600])
    v = PRESETS['VOODOO']['settings']
    check('the VOODOO preset: DITHER_VOODOO + TRIANGLE + chroma key, GL_CLAMP named in the note',
          v.get('tex_mip_select') == 'DITHER_VOODOO' and v.get('tex_lod_source') == 'TRIANGLE' and v.get('tex_colorkey') is True
          and 'GL_CLAMP' in PRESETS['VOODOO']['note'])
    check('VOODOO2 inherits the VOODOO roads', PRESETS['VOODOO2']['settings'].get('tex_mip_select') == 'DITHER_VOODOO')
    check('the wave-2 scenes render under the VOODOO preset (chroma key + dithered per-polygon level)',
          np.isfinite(cpu_frame('textured_keyed', preset=None) if False else
                      cpu_frame('textured_keyed', tex_filter='BILINEAR', tex_mipmap=True, tex_frac_bits='BITS_4',
                                tex_format='RGB565', tex_colorkey=True, tex_mip_select='DITHER_VOODOO',
                                tex_lod_source='TRIANGLE')).all())
    check('an unknown level road raises in _footprint_sampler instead of emitting a wrong sampler',
          _raises(lambda: MAT._footprint_sampler('hal_tex0', TX.Texture(checker_image(), colorspace='Non-Color'),
                                                 'CUBIC', 'REPEAT', dict(TX.SAMPLE_OPTS_NEUTRAL), 0.0, (64, 64))))


# ------------------------------------------------------------------ C083 x C074
def test_colorkey_alphaless_formats():
    """1.90.0 fix pass 3: under a texel format with NO alpha plane the
    cut-out is the only record of the hole and the storage law drops it, so
    the conversion to the key colour happens inside the law. Before this the
    VOODOO / VOODOO2 presets (RGB565 + chroma key) keyed nothing: 0 pixels
    of the keyed-alpha scene moved."""
    px = _keyed_image()
    hole = px[:, :, 3] < 0.5
    black = np.array([0, 0, 0, 1], np.float32)
    for fmt in ('RGB565', 'RGB332', 'RGB5550', 'YIQ422', 'I4'):
        for encoded in (True, False):
            t = TX.Texture(px.copy(), colorspace='Non-Color')
            t.store_format(fmt, encoded, keyed=True)
            stored = t.pixels.copy()
            t.colorkey_prepare()
            u = TX.Texture(px.copy(), colorspace='Non-Color')
            u.store_format(fmt, encoded)
            check(f'{fmt} keyed (encoded {encoded}): every hole texel is stored as exactly (0, 0, 0, 1), every '
                  'other texel is bitwise the unkeyed law, and colorkey_prepare then changes nothing',
                  bool((stored[hole] == black).all()) and bool(np.array_equal(stored[~hole], u.pixels[~hole]))
                  and bool(np.array_equal(t.pixels, stored)) and not bool((u.pixels[hole] == black).all()))
    for fmt in ('ARGB1555', 'ARGB4444', 'RGBA8888', 'A8', 'AI44', 'AI88', 'I8', 'I4_MODEL2', 'A3I5', 'A5I3',
                'AYIQ8422', 'IA31', 'I4A'):
        a = TX.Texture(px.copy(), colorspace='Non-Color')
        a.store_format(fmt, True, keyed=True)
        b = TX.Texture(px.copy(), colorspace='Non-Color')
        b.store_format(fmt, True)
        check(f'{fmt}: a format with an alpha plane of its own is bitwise unchanged by the keyed flag',
              bool(np.array_equal(a.pixels, b.pixels)))
    # through the pipeline's own prep, at the VOODOO preset's texture laws
    v = PRESETS['VOODOO']['settings']
    laws = {k: v[k] for k in ('tex_filter', 'tex_mipmap', 'tex_format', 'tex_frac_bits', 'tex_mip_select',
                              'tex_lod_source', 'tex_colorkey')}
    check("the VOODOO preset's texture laws are RGB565 with the chroma key on",
          laws['tex_format'] == 'RGB565' and laws['tex_colorkey'] is True, str(laws))
    R.clear_caches()
    sc, st = scene('textured_keyed', **dict(laws, tex_mipmap=False))
    tex = R.prepare_textures(sc, st)['checker']
    check('prepare_textures under RGB565 + chroma key stores every cut-out texel as the key colour',
          bool((tex.pixels[hole] == black).all()), f'{int((tex.pixels[hole] == black).all(axis=1).sum())} of '
          f'{int(hole.sum())}')
    yy, xx = np.nonzero(hole)
    s = tex.sample(((xx + 0.5) / 16).astype(np.float32), ((yy + 0.5) / 16).astype(np.float32),
                   filt='BILINEAR', wrap='REPEAT', opts=dict(TX.SAMPLE_OPTS_NEUTRAL, colorkey=True))
    check('...and the sample-time test keys every hole centre (alpha 0)', bool((s[:, 3] == 0.0).all()))

    def ab(**kw):
        R.clear_caches()
        st2 = base_settings(W, H, **kw)
        st2.use_processes = False
        return np.asarray(R.render(FM.SCENES['textured_keyed_alpha'](st2), st2))
    for name in ('VOODOO', 'VOODOO2'):
        pv = PRESETS[name]['settings']
        lw = {k: pv[k] for k in laws}
        moved = int((np.abs(ab(**lw) - ab(**dict(lw, tex_colorkey=False))).max(axis=2) > 0).sum())
        check(f"the {name} preset's chroma key moves the keyed-alpha frame (it moved 0 pixels before the fix)",
              moved > 0, f'{moved} of {W * H} px')
    # the GPU twin at the preset's FULL texture laws (mips, the dithered
    # per-polygon level, 5:6:5, the key) -- after showing it is not
    # vacuous: on the opaque keyed scene the key's prep blackens the disc
    R.clear_caches()
    k_off = cpu_frame('textured_keyed', **dict(laws, tex_colorkey=False))
    R.clear_caches()
    k_on = cpu_frame('textured_keyed', **laws)
    moved = int((np.abs(k_on - k_off).max(axis=2) > 0).sum())
    check("on the opaque keyed scene the VOODOO laws' key moves the frame (blackened holes; 0 pixels "
          'before the fix), so the twin below compares a keyed picture', moved > 0, f'{moved} of {W * H} px')
    R.clear_caches()
    sc, st = scene('textured_keyed', **laws)
    frame_bar('the VOODOO texel laws (BILINEAR + BITS_4 + mips + DITHER_VOODOO + TRIANGLE + RGB565 + '
              'chroma key) on textured_keyed', st, sc)
    frame_sampler_twin('the VOODOO texel laws on textured_keyed', st, sc)
    R.clear_caches()


def test_fix3_preset_keys():
    """1.90.0 fix pass 3: the keys three handovers listed for another
    pack's preset, which the merge dropped. One is in (RAST-A1's
    pixel_center on D3D_RETAIL_1997). Two stay out, each with a measured
    reason the review of the pass found: one global 5:5:5 texel format on
    POWERVR_PCX2 turns every alpha cut-out opaque, and Fog Depth Z on
    D3D_RETAIL_1997 makes the Fog switch fog nothing at scene-unit
    Start/End (test_r251_lighting_fog.test_f007_zfog_advisory)."""
    def preset_frame(name, key='textured', fill_alpha=False, **over):
        R.clear_caches()
        st = base_settings(W, H, **dict(PRESETS[name]['settings'], **over))
        st.resolution_x, st.resolution_y = W, H
        st.use_processes = False
        sc = FM.SCENES[key](st)
        if fill_alpha:
            px = np.array(sc.images['checker'].pixels, np.float32, copy=True)
            px[:, :, 3] = 1.0
            sc.images['checker'].pixels = px
        return np.asarray(R.render(sc, st))
    d = PRESETS['D3D_RETAIL_1997']['settings']
    check("D3D_RETAIL_1997 samples at Direct3D's integer pixel centre", d.get('pixel_center') == 'INTEGER_D3D')
    a = preset_frame('D3D_RETAIL_1997')
    b = preset_frame('D3D_RETAIL_1997', pixel_center='HALF')
    moved = int((np.abs(a - b).max(axis=2) > 0).sum())
    check('...and the key acts: the preset frame differs from the same preset at the HALF centre',
          bool(np.isfinite(a).all()) and moved > 0, f'{moved} of {W * H} px')
    check('D3D_RETAIL_1997 leaves Fog Depth at W (under Z its Fog switch fogs nothing at scene-unit Start/End: '
          'left to the user, disclosed)', 'fog_depth' not in d)
    p = PRESETS['POWERVR_PCX2']['settings']
    check('POWERVR_PCX2 leaves the texel format alone (the card kept 4:4:4:4 for textures with alpha; one '
          'global 5:5:5 would drop every cut-out)', 'tex_format' not in p)
    cut = preset_frame('POWERVR_PCX2', 'textured_keyed_alpha')
    full = preset_frame('POWERVR_PCX2', 'textured_keyed_alpha', fill_alpha=True)
    n_cut = int((np.abs(cut - full).max(axis=2) > 0).sum())
    cut5 = preset_frame('POWERVR_PCX2', 'textured_keyed_alpha', tex_format='RGB5550')
    full5 = preset_frame('POWERVR_PCX2', 'textured_keyed_alpha', fill_alpha=True, tex_format='RGB5550')
    n_cut5 = int((np.abs(cut5 - full5).max(axis=2) > 0).sum())
    check('the reason, measured: as shipped the preset shows the cut-out disc; under tex_format RGB5550 the '
          'frame is bitwise the alpha-filled one (the cut-out gone, nothing printed)',
          n_cut > 0 and n_cut5 == 0, f'{n_cut} px cut as shipped, {n_cut5} under RGB5550')
    R.clear_caches()


def main():
    utf8_console()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    # the identity pin first, then the features in the entry's order
    order = ['test_r251_identity', 'test_plumbing_laws',
             'test_texel_format_laws', 'test_texel_format_moves_the_picture', 'test_texel_format_gpu_twin',
             'test_tmem_budget_sizes', 'test_tmem_gpu_twin',
             'test_dxt1_block_laws', 'test_vq_codebook_laws', 'test_compress_gpu_twin',
             'test_frac_bits_laws', 'test_frac_bits_gpu_twin',
             'test_normdist_laws', 'test_normdist_gpu_twin',
             # ---- wave 2 (TEX-2), in the entry's order
             'test_wave2_plumbing',
             'test_gl_clamp_laws', 'test_gl_clamp_gpu_twin',
             'test_colorkey_laws', 'test_colorkey_gpu_twin',
             'test_colorkey_alphaless_formats', 'test_fix3_preset_keys',
             'test_mip_select_laws', 'test_mip_select_gpu_twin',
             'test_tri_lod_laws', 'test_tri_lod_gpu_twin',
             'test_gs_lod_laws', 'test_gs_lod_gpu_twin',
             'test_sharpen_laws', 'test_sharpen_gpu_twin',
             'test_sat_laws', 'test_sat_gpu_twin']
    names = [n for n in order if n in dict(tests)] + [n for n, _f in tests if n not in order]
    for n in names:
        print(f'--- {n}')
        try:
            globals()[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print(f'\nR251 TEXTURE: {len(FAILS)} failure(s)')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
