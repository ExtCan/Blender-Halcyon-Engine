"""R254: the Spot / Area / Sun / Hemi options and the per-lamp effects.

Every check below fails on the tree before this round and passes now (the
module runs against the old tree too: the helpers fall back to the calls
it had), or pins something that must NOT have moved.

    (a) what must not have moved: Hotspot 0, Volumetric 0, Caustics 0
    (b) Hotspot is the Blender Internal cone's inner angle, a value on
        both roads
    (c) the visible beam's unit: Volumetric 0..4 is a working range at
        Blender's stock lamp powers for Spot, Point and Area
    (d) the beam follows the lamp's cone law (GX Ring 2 out to its zero)
    (e) a Point lamp's glow is the integral at any sample count
    (f) an Area lamp's hidden Custom Range does not cut its beam
    (g) light shafts: the march is the old sums at a fraction of the
        cost, the rendered viewport draws them (a proxy march, anchors
        through the camera-frame crop), the worker pool hands them over
    (h) a Hemi's shafts, flare and caustics follow its direction
    (i) caustics: Caustic Scale above 8 on a Spot, the lamp keeps its
        light
    (j) the cookie's plan signature: Spot Size, every texel
    (k) a Screen Spotlight honours Specular Only
    (l) the Light panel says what each dial needs
    (m) the changed emitter shapes under strict GLSL
    (n) the options nobody had measured

    python -m halcyon.tests.test_r254_spot_beam
"""
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import plane, cube, _mesh_concat, look_at_matrix
from .test_r251_lighting_lamps import _run_glsl
from ..core import cones as CONES
from ..core import lights as LI
from ..core import mathx as M
from ..core import post as POST
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import (Camera, ImageBuffer, Light, Material, ObjectInfo,
                          Scene, World)
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
W, H = 128, 96
BAR = 6e-3
CAM_POS = (0.0, -7.5, 6.5)
CAM_AIM = (0.0, 0.6, 0.0)
#: a camera that looks up at a white sky with the sun ahead of it
FACE_POS = (0.0, -7.5, 1.2)
FACE_AIM = (0.0, 0.6, 2.0)


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def note(text):
    print(f'  note {text}')


# ------------------------------------------------------------------ the rig


def _settings(w=W, h=H, **kw):
    st = base_settings(w, h, shadow_default='PER_LIGHT',
                       global_ambient=(0.0, 0.0, 0.0))
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _aim(pos, at):
    d = np.asarray(at, np.float64) - np.asarray(pos, np.float64)
    return tuple(float(v) for v in d / np.linalg.norm(d))


def _lamp(kind='SPOT', pos=(0.0, 2.5, 4.0), at=(0.0, 0.0, 0.0), **kw):
    base = dict(type=kind, name='L', position=tuple(pos),
                direction=_aim(pos, at), color=(1.0, 1.0, 1.0),
                energy=1000.0, shadow='NONE', radius=0.0)
    if kind in ('SUN', 'HEMI'):
        base['energy'] = 2.0
    if kind == 'SPOT':
        base.update(spot_size=1.0, spot_blend=0.15)
    f = np.asarray(base['direction'], np.float64)
    x = np.cross((0.0, 0.0, 1.0), -f)
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0])
    x = x / np.linalg.norm(x)
    y = np.cross(-f, x)
    base.update(frame_x=tuple(x), frame_y=tuple(y), area_x=tuple(x),
                area_y=tuple(y))
    base.update(kw)
    return Light(**base)


def _scene(lights, st, box=False, cam_pos=CAM_POS, cam_aim=CAM_AIM,
           world=(0.0, 0.0, 0.0), floor=14.0, mats=None):
    if isinstance(lights, Light):
        lights = [lights]
    mats = mats or [
        Material(name='Floor', index=0, model='LAMBERT',
                 diffuse=(0.8, 0.8, 0.8), specular_level=0.0,
                 ambient_level=0.0),
        Material(name='Box', index=1, model='LAMBERT',
                 diffuse=(0.8, 0.3, 0.2), specular_level=0.0,
                 ambient_level=0.0)]
    parts = [plane(z=0.0, size=floor, mat=0, obj=0)]
    objs = [ObjectInfo(name='Floor', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=f32))]
    if box:
        parts.append(cube(centre=(0.0, 0.0, 1.6), size=1.2, mat=1, obj=1))
        objs.append(ObjectInfo(name='Box', index=1, location=(0, 0, 1.6),
                               matrix_world=np.eye(4, dtype=f32)))
    mesh = _mesh_concat(parts)
    mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
    cam = Camera(matrix_world=look_at_matrix(cam_pos, cam_aim), lens=35.0,
                 sensor=36.0, clip_start=0.1, clip_end=200.0)
    sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=list(lights),
               camera=cam, world=World(color=tuple(world),
                                       ambient=(0.0, 0.0, 0.0)),
               settings=st)
    sc.images = {}
    return sc


TOP = dict(cam_pos=(0.0, -0.001, 12.0), cam_aim=(0.0, 0.0, 0.0))


def _cpu(lights, st=None, sc=None, time=None, **scene_kw):
    st = st or _settings()
    LI.clear_shadow_cache()
    sc = sc or _scene(lights, st, **scene_kw)
    if time is not None:
        sc.time = time
    return np.asarray(R.render(sc, st))[..., :3].astype(np.float64), sc


def _gpu(lights, st=None, sc=None, time=None, clear=True, **scene_kw):
    """(frame, plan-cache hit) of the deferred pass through the GLSL
    simulator twin; `clear=False` leaves the plan cache as the second
    frame of a session finds it. None when the plan refuses."""
    st = st or _settings()
    LI.clear_shadow_cache()
    sc = sc or _scene(lights, st, **scene_kw)
    if time is not None:
        sc.time = time
    R.render(sc, st)                 # maps and caustic cookies, as F12
    w, h = int(st.resolution_x), int(st.resolution_y)
    view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = CRg.GBuffer(w, h)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                  depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    if clear:
        GSH._PLAN_CACHE.clear()
    n0 = len(GSH._PLAN_CACHE)
    p, why, atl = GSH.plan_frame(job, g)
    if p is None:
        note(f'the GPU plan refused: {why}')
        return None, False
    hit = (len(GSH._PLAN_CACHE) == n0) and not clear
    out, miss = GSH.simulate(job, g, p, atl)
    if out is None:
        note(f'the GPU simulator refused: {miss}')
        return None, hit
    return np.asarray(out)[..., :3].astype(np.float64), hit


def _gpu_full(lights, st=None, sc=None, time=None, **scene_kw):
    """The WHOLE GPU road (render_device GPU through tests/fakedevice):
    (frame, engaged)."""
    from . import fakedevice
    st = st or _settings()
    st.render_device = 'GPU'
    LI.clear_shadow_cache()
    sc = sc or _scene(lights, st, **scene_kw)
    if time is not None:
        sc.time = time
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed():
        img = np.asarray(R.render(sc, st))
    return img[..., :3].astype(np.float64), \
        bool(getattr(st, '_frame_gpu_shaded', False))


def _roads(label, lights, st_kw=None, bar=BAR, time=None, **scene_kw):
    c, _sc = _cpu(lights, _settings(**(st_kw or {})), time=time, **scene_kw)
    g, _hit = _gpu(lights, _settings(**(st_kw or {})), time=time, **scene_kw)
    if g is None:
        check(f'{label}: the deferred pass plans and simulates', False)
        return c, None
    d = float(np.abs(c - g).max())
    check(f'{label}: the GPU twin is the CPU frame within the deferred bar',
          d < bar, f'max |CPU - GPU| {d:.2e} (frame peak {float(c.max()):.3f})')
    return c, g


def _lum(a):
    return np.asarray(a)[..., :3].mean(axis=-1)


def _beam(lamp_kw, kind='SPOT', vol=1.0, st_kw=None, **scene_kw):
    """The in-air beam alone: the frame with the lamp's Volumetric minus
    the frame without it, the render's switch on."""
    st_kw = dict(st_kw or {})
    a, _s = _cpu(_lamp(kind, **lamp_kw), _settings(spot_cones=True, **st_kw),
                 **scene_kw)
    b, _s = _cpu(_lamp(kind, volumetric=vol, **lamp_kw),
                 _settings(spot_cones=True, **st_kw), **scene_kw)
    return _lum(b - a), a, b


def _checker(n=64, cells=8):
    y, x = np.mgrid[0:n, 0:n]
    c = (((x * cells) // n + (y * cells) // n) % 2).astype(f32)
    return np.repeat(c[:, :, None], 3, axis=2)


def _asym(n=64):
    a = np.ones((n, n, 3), f32)
    a[:, : n // 4] = 0.0
    a[3 * n // 4:, 3 * n // 4:] = 0.0
    return a


def _cookie(px):
    return ImageBuffer(name='ck', pixels=np.asarray(px, f32),
                       colorspace='Non-Color')


def _band(light, spotsi):
    fn = getattr(LI, 'spot_blend_band', None)
    return fn(light, spotsi) if fn else (1.0 - spotsi) * float(light.spot_blend)


def _consts_literal():
    return {'light_texels': False, 'falloff_default': 'INVERSE_SQUARE',
            'cookies': {}, 'light_links': {}, 'resolution': (W, H),
            'specular_in_gamma': True, 'light_clamp': 0.0,
            'shadow_default': 'PER_LIGHT'}


def _block(light, consts=None):
    return GM._one_light_source(0, light, consts or _consts_literal())


def _cone_dirs(n=2001, span=0.9):
    """L (surface -> light) for n angles 0..span off a -Z axis; the cone
    test's cosine is cos(theta)."""
    th = np.linspace(0.0, span, n).astype(f32)
    L = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    return L.astype(f32), np.cos(th).astype(f32), th


def _cone_spot(**kw):
    base = dict(type='SPOT', name='cone', position=(0.0, 0.0, 0.0),
                direction=(0.0, 0.0, -1.0), spot_size=1.0, spot_blend=0.2,
                shadow='NONE')
    base.update(kw)
    return Light(**base)


# ---------------------------------------------- (a) what must not have moved


def _old_spot_blender(light, inpr):
    """The cone body before this round, verbatim."""
    spotsi = np.float32(np.cos(float(light.spot_size) * 0.5))
    spotbl = np.float32((1.0 - spotsi) * float(light.spot_blend))
    t = inpr - spotsi
    if spotbl != 0.0:
        i = np.clip(t / spotbl, 0.0, 1.0)
        soft = np.where(t < spotbl, 3.0 * i * i - 2.0 * i * i * i, 1.0)
    else:
        soft = np.ones_like(inpr)
    fac = np.where(inpr <= spotsi, 0.0, soft * inpr)
    return fac.astype(np.float32)


def test_a_defaults_have_not_moved():
    """Hotspot 0, Volumetric 0, Caustics 0, no cookie: the statements
    each site evaluated before, bitwise (the frames against the previous
    tree are in docs-dev/r254/spot-beam.md: 158 of 158 hashes)."""
    L, cosang, _th = _cone_dirs()
    for blend in (0.0, 0.15, 0.2, 1.0):
        for size in (0.3, 1.0, 2.6):
            lt = _cone_spot(spot_blend=blend, spot_size=size)
            if not np.array_equal(LI.spot_falloff(lt, L),
                                  _old_spot_blender(lt, cosang)):
                check(f'Hotspot 0 (Blend {blend}, size {size}): the cone is '
                      'the old body bitwise', False)
                break
        else:
            continue
        break
    else:
        check('Hotspot 0: the Blender Internal cone is the old body bitwise '
              'for twelve Blend x Spot Size pairs', True)
    lt = _cone_spot(spot_size=1.1, spot_blend=0.37)
    si64 = float(np.cos(float(lt.spot_size) * 0.5))
    tex = GM.pack_light_texels([lt])
    check('Hotspot 0: the packed band (hal_lights texel 2.w) is the '
          'float32 of (1 - spotsi) * Blend, as before',
          tex[0, 2, 3] == np.float32((1.0 - si64) * float(lt.spot_blend)))
    bare = _cone_spot(spot_size=1.1, spot_blend=0.37)
    del bare.__dict__['hotspot']
    check('Hotspot 0: the emitted light block is the block of a lamp '
          'without the field', _block(lt) == _block(bare))
    lt_m = _cone_spot(spot_size=1.1, spot_blend=0.37)
    cos_half = float(np.cos(0.55))
    c = np.linspace(cos_half - 0.05, 1.0, 301)
    inner = cos_half + (1.0 - cos_half) * np.clip(0.37, 0.0, 0.999)
    ang = np.clip((c - cos_half) / np.maximum(inner - cos_half, 1e-6),
                  0.0, 1.0)
    want = ang * ang * (3.0 - 2.0 * ang)
    if hasattr(CONES, 'beam_angular'):
        check('Hotspot 0, Blender Internal law: the beam\'s angular term, '
              'its edge and its cut are the old statements bitwise',
              np.array_equal(CONES.beam_angular(lt_m, 'BLENDER', c, cos_half,
                                                CONES.beam_edge(lt_m)), want)
              and CONES.beam_edge(lt_m) == 0.37
              and CONES.beam_cos_out(lt_m, cos_half) == cos_half)
    # frames: the switches at rest are the frame without the option
    plain, _s = _cpu(_lamp('SPOT'), box=True)
    for label, lamp, st_kw in (
            ('Volumetric 0 with the render\'s beam switch on',
             _lamp('SPOT'), dict(spot_cones=True)),
            ('Volumetric 1 with the switch off',
             _lamp('SPOT', volumetric=1.0), {}),
            ('Caustic Scale and Speed with Caustics 0',
             _lamp('SPOT', caustics_scale=32.0, caustics_speed=3.0), {})):
        f, _s = _cpu(lamp, _settings(**st_kw), box=True)
        check(f'{label}: the frame is the plain lamp\'s bitwise',
              np.array_equal(f, plain))
    for law in ('GL11', 'GX_FLAT', 'GX_COS', 'GX_RING1'):
        a = LI.spot_falloff(_cone_spot(spot_law=law), L)
        b = LI.spot_falloff(_cone_spot(spot_law=law, hotspot=0.6), L)
        if not np.array_equal(a, b):
            check(f'{law}: Hotspot is still not read (no inner angle)', False)
            break
    else:
        check('the OpenGL and GX cone laws still do not read Hotspot '
              '(they have no inner angle; the panel no longer draws it)',
              True)


# -------------------------------------------------------------- (b) hotspot


def test_b_hotspot_is_the_inner_angle():
    """hotspot-dead-01: under the default cone law Hotspot moved no pixel
    on any road. It is the inner angle now: full strength inside it, the
    smoothstep out to the Spot Size, a value on both roads."""
    L, cosang, th = _cone_dirs()
    size = 1.0
    base = LI.spot_falloff(_cone_spot(), L)
    worst = ''
    for hot in (0.2, 0.5, 0.8):
        f = LI.spot_falloff(_cone_spot(hotspot=hot), L)
        inside = th < hot * 0.5 - 2e-3
        outside = th > size * 0.5 + 2e-3
        band = (th > hot * 0.5 + 2e-3) & (th < size * 0.5 - 2e-3)
        soft = f / np.maximum(cosang, 1e-9)
        ok = bool(np.all(np.abs(soft[inside] - 1.0) < 1e-6)) \
            and bool(np.all(f[outside] == 0.0)) \
            and bool(np.all(soft[band] < 1.0)) \
            and bool(np.all(soft[band] > 0.0)) \
            and bool(np.all(np.diff(soft[band]) <= 1e-6))
        if not ok:
            worst = f'hotspot {hot}'
        # the band's own law: the smoothstep over cos(hot/2) - cos(size/2)
        cf = np.cos(size * 0.5)
        bw = np.cos(hot * 0.5) - cf
        i = np.clip((cosang.astype(np.float64) - cf) / bw, 0.0, 1.0)
        want = np.where(cosang > cf, (3 * i * i - 2 * i * i * i) * cosang, 0.0)
        if float(np.abs(f - want).max()) > 2e-5:
            worst = f'hotspot {hot}: off the smoothstep by ' \
                    f'{float(np.abs(f - want).max()):.2e}'
    check('Blender Internal + Hotspot: exactly cos inside the inner angle, '
          'the smoothstep across cos(hotspot/2) - cos(size/2), zero past '
          'the Spot Size (three angles)', not worst, worst)
    f_mid = LI.spot_falloff(_cone_spot(hotspot=0.5), L)
    check('Hotspot moves the default cone (it changed no pixel before)',
          float(np.abs(f_mid - base).max()) > 0.2,
          f'max change {float(np.abs(f_mid - base).max()):.3f}')
    hard = LI.spot_falloff(_cone_spot(spot_blend=0.0), L)
    for hot in (1.0, 1.7, 3.1):
        if not np.array_equal(LI.spot_falloff(_cone_spot(hotspot=hot), L),
                              hard):
            check('a Hotspot at or past the cone is the hard edge', False,
                  f'hotspot {hot}')
            break
    else:
        check('a Hotspot at or past the Spot Size is the hard-edged cone '
              '(Blend 0) bitwise', True)
    # the value road: the CPU's float32 band IS the texel
    lt = _cone_spot(hotspot=0.5)
    si32 = np.float32(np.cos(0.5))
    tex = GM.pack_light_texels([lt])
    check('the CPU\'s band and hal_lights texel 2.w are the same float32',
          tex[0, 2, 3] == np.float32(_band(lt, si32))
          and float(tex[0, 2, 3]) > 0.05,
          f'{float(tex[0, 2, 3]):.7f}')
    # the emitted statements, standalone, against the CPU law
    lines = _block(lt)
    i0 = next((k for k, ln in enumerate(lines) if 'float spot_t' in ln), None)
    i1 = next((k for k, ln in enumerate(lines)
               if 'spot_f = spot_s * cosang' in ln), None)
    if i0 is None or i1 is None:
        check('the Hotspot lamp emits the blend block', False)
    else:
        body = lines[i0:i1 + 1] + ['    }']
        src = ('uniform float cosang;\nout vec4 Color;\nvoid main()\n{\n'
               + '\n'.join(body)
               + '\n    Color = vec4(spot_f, 0.0, 0.0, 1.0);\n}\n')
        got, err = _run_glsl(src, cosang.size, cosang=cosang)
        if got is None:
            check('the Hotspot cone GLSL compiles standalone', False, str(err))
        else:
            d = float(np.abs(got[:, 0] - LI.spot_falloff(lt, L)).max())
            check('the emitted Hotspot cone equals the CPU law within 2e-6',
                  d < 2e-6, f'd {d:.3g}')
    # frames on both roads
    for hot in (0.3, 0.7, 1.4):
        _roads(f'a spot with Hotspot {hot}', _lamp('SPOT', hotspot=hot),
               box=True)
    a, _s = _cpu(_lamp('SPOT'), box=True)
    b, _s = _cpu(_lamp('SPOT', hotspot=0.3), box=True)
    check('Hotspot 0.3 moves the rendered pool',
          float(np.abs(a - b).max()) > 0.05 * float(a.max()),
          f'{float(np.abs(a - b).max() / a.max()):.3f} of peak')
    # plan cache: a drag is a HIT, opening or closing the band a MISS
    s0 = GSH._light_sig(_cone_spot(hotspot=0.3))
    check('dragging Hotspot is a plan-cache HIT (a texel), as is taking it '
          'from 0 while a Blend band exists',
          s0 == GSH._light_sig(_cone_spot(hotspot=0.6))
          and s0 == GSH._light_sig(_cone_spot()))
    check('a Hotspot dragged onto the cone edge closes the band: structure '
          '(the hard-edge block), a MISS',
          s0 != GSH._light_sig(_cone_spot(hotspot=1.2)))
    st = _settings()
    sc = _scene(_lamp('SPOT', hotspot=0.3), st, box=True)
    g1, _h = _gpu(None, st, sc=sc)
    sc.lights[0].hotspot = 0.7
    g2, hit = _gpu(None, st, sc=sc, clear=False)
    c2, _s = _cpu(None, st, sc=sc)
    check('...and the warm plan with the new texel is the CPU frame',
          g2 is not None and hit and float(np.abs(g2 - c2).max()) < BAR
          and float(np.abs(g2 - g1).max()) > 0.01,
          f'hit={hit}' + ('' if g2 is None else
                          f' max |CPU - GPU| {float(np.abs(g2 - c2).max()):.2e}'))
    # POV keeps its own reading of the same dial
    pov = LI.spot_falloff(_cone_spot(spot_law='POV', hotspot=0.5), L)
    check('POV still reads Hotspot as its radius (flat 1 inside it)',
          bool(np.all(pov[th < 0.25 - 2e-3] == 1.0)))


# ---------------------------------------------------- (c) the beam's unit


def test_c_beam_unit():
    """beam-calibration-01 / fx-vol-beam-scale: the composite multiplied
    the marched scatter by energy / pi -- 632 times the radiance scale a
    spot or point lamp lights its surfaces with -- so a beam at the
    slider's unit value was a solid white wedge at any stock power."""
    spot_kw = dict(pos=(0.0, 2.5, 4.0), at=(0.0, 0.0, 0.0), energy=1000.0)
    res = {}
    for vol in (1.0, 4.0):
        beam, a, b = _beam(spot_kw, vol=vol, box=True)
        m = beam > 1e-6
        res[vol] = (float(np.percentile(beam[m], 95)), float(beam[m].mean()),
                    100.0 * float((_lum(b) > 1.0).mean()))
    lit = _lum(a)
    lit_med = float(np.median(lit[lit > 0.02]))
    check('a 1000 W spot at Volumetric 1: 95% of the beam sits under the '
          'lit floor\'s own level, and under white',
          res[1.0][0] < min(1.0, 3.0 * lit_med),
          f'beam p95 {res[1.0][0]:.3f}, lit-surface median {lit_med:.3f}')
    check('...and Volumetric 4, the slider\'s top, is a strong beam, not a '
          'white frame', 0.3 < res[4.0][0] and res[4.0][2] < 10.0,
          f'beam p95 {res[4.0][0]:.3f}, {res[4.0][2]:.1f}% of the frame '
          'over white')
    pt_kw = dict(pos=(2.5, -1.5, 5.0), at=(0.0, 0.0, 0.8), energy=1000.0)
    beam, a, b = _beam(pt_kw, kind='POINT', vol=1.0, box=True)
    over = 100.0 * float((_lum(b) > 1.0).mean())
    check('a 1000 W point lamp at Volumetric 1 is a haze, not a white '
          'frame (it was 100% white)', over < 25.0
          and 0.05 < float(np.median(beam)) < 0.6,
          f'{over:.1f}% over white, glow median {float(np.median(beam)):.3f}')
    # the unit itself
    if hasattr(CONES, 'beam_gain'):
        k = 64.0 * np.pi ** 3
        sp = Light(type='SPOT', energy=370.0)
        pt = Light(type='POINT', energy=370.0)
        ar = Light(type='AREA', energy=370.0, decay_end=25.0)
        ar.area_size = (1.0, 1.0)
        check('spot and point: energy / (64 pi^3) -- the lamp\'s own '
              'energy / (4 pi), the isotropic phase 1 / (4 pi), Density x '
              'Volumetric in 1 / (4 pi) per world unit',
              abs(CONES.beam_gain(sp) - 370.0 / k) < 1e-12
              and CONES.beam_gain(sp) == CONES.beam_gain(pt))
        check('area: energy * Distance^2 / (32 pi^3) -- the form factor\'s '
              '2 D^2 far field on the same scale',
              abs(CONES.beam_gain(ar) - 370.0 * 625.0 / (32.0 * np.pi ** 3))
              < 1e-9)
        sp.style_gain = 0.5
        check('the light style\'s gain rides the beam',
              abs(CONES.beam_gain(sp) - 185.0 / k) < 1e-12)
    # an Area lamp's beam follows its surfaces: the same lamp expressed
    # with Distance 25 or Distance sqrt(pi/2) (Blender's watts) is the
    # same picture, beam included
    d2 = float(np.sqrt(np.pi / 2.0))
    e1 = 0.5
    e2 = e1 * (25.0 / d2) ** 2
    akw = dict(pos=(0.0, 2.5, 4.0), at=(0.0, 0.0, 0.0),
               area_size=(1.0, 1.0))
    b1, a1, f1 = _beam(dict(akw, energy=e1, decay_end=25.0), kind='AREA',
                       vol=2.0)
    b2, a2, f2 = _beam(dict(akw, energy=e2, decay_end=d2), kind='AREA',
                       vol=2.0)
    check('an Area lamp at Distance 25 and the same lamp at Distance 1.2533 '
          'with 398x the Power light the same surfaces',
          float(np.abs(a1 - a2).max()) < 2e-3 * float(a1.max()),
          f'{float(np.abs(a1 - a2).max() / a1.max()):.1e} of peak')
    check('...and draw the same beam (it was 398x apart: the beam read the '
          'Power and not the Distance its surfaces are scaled by)',
          float(b1.max()) > 1e-3
          and float(np.abs(b1 - b2).max()) < 2e-3 * float(b1.max()),
          f'beam peaks {float(b1.max()):.4f} / {float(b2.max()):.4f}')
    check('the Area beam at Volumetric 2 is visible beside a floor exposed '
          'near 1 (0.02 .. 1.5 of the floor\'s peak)',
          0.02 * float(_lum(a1).max()) < float(b1.max())
          < 1.5 * float(_lum(a1).max()),
          f'beam peak {float(b1.max()):.3f}, floor peak '
          f'{float(_lum(a1).max()):.3f}')
    # one lamp, one volumetric effect: where the beam is drawn the lamp
    # is not ALSO a screen-space shaft source (the two shared the slider
    # unseen while a usable beam sat at 0.01; at the new unit's 1 the
    # shafts smeared banded streaks below the lamp's own pool)
    srcs = {}
    for kind in ('SPOT', 'POINT', 'AREA', 'SUN', 'HEMI'):
        for cones in (False, True):
            st_s = _settings(spot_cones=cones)
            sc_s = _scene(_lamp(kind, pos=(0.0, 6.0, 6.0),
                                at=(0.0, -2.0, 0.0), volumetric=1.0), st_s,
                          cam_pos=FACE_POS, cam_aim=FACE_AIM)
            vp_s = R.camera_matrices(sc_s.camera, W, H)[2]
            srcs[(kind, cones)] = len(R.shaft_sources(sc_s, st_s, vp_s))
    check('a Spot, Point or Area lamp throws Light Shafts while its beam '
          'is not drawn, and the beam alone while it is; a Sun or a hemi '
          'always throws shafts',
          all(srcs[(k, False)] == 1 for k in ('SPOT', 'POINT', 'AREA',
                                              'SUN', 'HEMI'))
          and all(srcs[(k, True)] == 0 for k in ('SPOT', 'POINT', 'AREA'))
          and srcs[('SUN', True)] == 1 and srcs[('HEMI', True)] == 1,
          str({f'{k} beams {"on" if c else "off"}': v
               for (k, c), v in srcs.items()}))
    # one composite for both devices
    st_kw = dict(spot_cones=True)
    c, _s = _cpu(_lamp('SPOT', volumetric=1.0, **spot_kw),
                 _settings(**st_kw), box=True)
    g, eng = _gpu_full(_lamp('SPOT', volumetric=1.0, **spot_kw),
                       _settings(**st_kw), box=True)
    check('the whole GPU road draws the same beam',
          eng and float(np.abs(c - g).max()) < BAR,
          f'engaged={eng}, max |CPU - GPU| {float(np.abs(c - g).max()):.2e}')


# ------------------------------------------------ (d) the beam and the law


def _across_lamp(**kw):
    pos = np.array([-3.0, 0.0, 3.5])
    aim = np.array([2.0, 0.5, 0.0])
    base = dict(pos=tuple(pos), at=tuple(aim), energy=300.0,
                spot_size=float(np.radians(40.0)), spot_blend=0.15)
    base.update(kw)
    return base


def test_d_beam_follows_the_cone_law():
    """beam-ignores-law-01: the visible beam followed Spot Size and Blend
    only -- the Cone Law, the Hotspot and the exponent shaped the pool
    and left the beam alone."""
    base, _a, _b = _beam(_across_lamp())
    rows = []
    for name, kw, lo, hi in (
            ('POV', dict(spot_law='POV'), 0.3, 0.8),
            ('POV + Hotspot 30 deg', dict(spot_law='POV',
                                          hotspot=float(np.radians(30))),
             0.6, 0.95),
            ('GL11 (hard rim)', dict(spot_law='GL11'), 1.01, 1.3),
            ('GX Ring 1 (hollow)', dict(spot_law='GX_RING1'), 0.4, 0.95),
            ('Blender Internal + Hotspot 10 deg',
             dict(hotspot=float(np.radians(10))), 0.3, 0.9)):
        b, _a, _f = _beam(_across_lamp(**kw))
        r = float(b.sum() / base.sum())
        rows.append((name, r, lo < r < hi and float(np.abs(b - base).max())
                     > 0.02 * float(base.max())))
    check('the beam\'s light follows the cone law and the Hotspot (sum '
          'against the Blender Internal beam; every row was 1.000)',
          all(ok for _n, _r, ok in rows),
          ', '.join(f'{n} {r:.3f}' for n, r, _ok in rows))
    ring2, _a, _f = _beam(_across_lamp(spot_law='GX_RING2'))
    col = ring2[:, W // 2]
    litc = np.nonzero(col > 1e-6 * ring2.max())[0]
    edge = float(max(col[litc[0]], col[litc[-1]]) / col.max()) \
        if litc.size else 1.0
    wide = int((ring2 > 1e-6 * ring2.max()).sum())
    check('GX Ring 2 is marched out to its own zero: the beam is wider '
          'than the cone and fades instead of ending in a cut rim',
          wide > 1.1 * int((base > 1e-6 * base.max()).sum()) and edge < 0.2,
          f'{wide} px against {int((base > 1e-6 * base.max()).sum())}, '
          f'edge / peak {edge:.3f}')
    # the brute-force twin, per law
    n = 2500
    rng = np.random.default_rng(7)
    origin = np.array([[0.0, 1.0, 6.0]], f32)
    d = rng.normal(size=(n, 3)).astype(f32)
    d[:, 2] = -np.abs(d[:, 2]) - 0.3
    rays = M.normalize(d)
    far = np.full(n, np.inf, f32)
    bad = []
    for law, kw in (('BLENDER', {}), ('BLENDER', dict(hotspot=0.5)),
                    ('POV', dict(hotspot=0.5)),
                    ('POV', dict(spot_exponent=8.0)), ('GL11', {}),
                    ('GX_COS', {}), ('GX_RING1', {}), ('GX_RING2', {})):
        lt = Light(type='SPOT', position=(0.0, 3.0, 0.0),
                   direction=(0.0, -1.0, 0.0), spot_size=0.9, spot_blend=0.2,
                   volumetric=1.0, spot_law=law, **kw)
        fast = CONES.spot_cone(origin, rays, far, lt, samples=1024)
        ref = CONES.reference(origin, rays, far, lt, samples=4096)
        lit = ref > 1e-5
        rel = np.abs(fast[lit] - ref[lit]) / np.maximum(ref[lit], 1e-6)
        if not lit.any() or float(np.median(rel)) > 1e-2 \
                or float(((fast > 1e-6) == lit).mean()) < 0.98:
            bad.append(f'{law} {kw}: median rel '
                       f'{float(np.median(rel)) if lit.any() else -1:.4f}')
    check('the analytic cone agrees with the brute-force march under every '
          'law (eight cases, median relative error under 1%)', not bad,
          '; '.join(bad))
    blend1, _a, _f = _beam(_across_lamp(spot_blend=1.0))
    pov1, _a, _f = _beam(_across_lamp(spot_law='POV', spot_blend=1.0))
    pov0, _a, _f = _beam(_across_lamp(spot_law='POV'))
    check('under POV the beam no longer follows Blend (the pool never '
          'did): Blend 0.15 and 1.0 are one beam',
          np.array_equal(pov1, pov0)
          and float(np.abs(blend1 - base).max()) > 0.0)


# --------------------------------------------- (e) the point lamp's glow


def test_e_point_glow_is_the_integral():
    """The verifiers' "beam dials on Point beams": twelve evenly spaced
    midpoints through a 64-unit sphere stepped over the 1 / d^2 peak --
    the glow was twice the true integral at 12 samples and a third of it
    at 2, drawn as rings."""
    origin = np.array([[0.0, -7.5, 6.5]], f32)
    rng = np.random.default_rng(3)
    d = rng.normal(size=(1500, 3)).astype(f32)
    d[:, 1] = np.abs(d[:, 1]) + 0.5
    rays = M.normalize(d)
    far = np.full(rays.shape[0], np.inf, f32)
    lt = Light(type='POINT', position=(1.5, 1.0, 3.0), volumetric=1.0)
    ref = CONES.point_glow(origin, rays, far, lt, samples=4096)
    got = {n_: float(CONES.point_glow(origin, rays, far, lt,
                                      samples=n_).sum() / ref.sum())
           for n_ in (2, 12, 64)}
    check('a point lamp\'s glow at 2 / 12 / 64 samples is the 4096-sample '
          'sum within 4% / 1% / 1% (the count used to decide the '
          'brightness)', abs(got[2] - 1.0) < 0.04
          and abs(got[12] - 1.0) < 0.01 and abs(got[64] - 1.0) < 0.01,
          ', '.join(f'{n_}: x{v:.3f}' for n_, v in got.items()))
    # against the closed form for falloff 2: the line integral of
    # (1 - d^2 / R^2) / d^2 through the sphere
    Rr = CONES.DEFAULT_REACH
    oc = origin[0].astype(np.float64) - np.asarray(lt.position, np.float64)
    r64 = rays.astype(np.float64)
    b = r64 @ oc
    perp = np.sqrt(np.maximum(oc @ oc - b * b, 1e-12))
    sq = np.sqrt(np.maximum(Rr * Rr - perp * perp, 0.0))
    t0 = np.maximum(-b - sq, 0.0)
    t1 = -b + sq
    exact = (np.arctan((t1 + b) / perp) - np.arctan((t0 + b) / perp)) / perp \
        - (t1 - t0) / (Rr * Rr)
    got12 = CONES.point_glow(origin, rays, far, lt, samples=12)
    rel = np.abs(got12 - exact) / np.maximum(exact, 1e-9)
    check('...and per ray it is the closed-form integral of (1 - d^2/R^2) '
          '/ d^2 within 1%', float(rel.max()) < 1e-2,
          f'max {float(rel.max()):.2e}')
    blocked = CONES.point_glow(origin, rays, far, lt, samples=12,
                               occlude=lambda p, live: np.ones(p.shape[0],
                                                               bool))
    check('Beam Occlusion still reaches every sample',
          float(blocked.max()) == 0.0)


# ------------------------------------------ (f) the area beam's hidden range


def test_f_area_beam_ignores_the_hidden_falloff_menu():
    """The verifiers' "Area Distance while the hidden Falloff menu is
    Custom": an Area lamp draws no Falloff menu and never attenuates, but
    a Custom Range left on it (a converted Point lamp) cut its beam at
    the panel's Distance."""
    akw = dict(pos=(0.0, 2.5, 4.0), at=(0.0, 0.0, 0.0), energy=1.0,
               area_size=(1.5, 1.5), decay_end=3.0)
    b_def, _a, _f = _beam(dict(akw, decay='DEFAULT'), kind='AREA')
    b_cus, _a, _f = _beam(dict(akw, decay='CUSTOM'), kind='AREA')
    check('an Area lamp\'s beam is the same under a hidden Custom Range '
          '(it was cut at Distance 3)', float(b_def.sum()) > 0.0
          and np.array_equal(b_def, b_cus),
          f'{int((b_def > 1e-7).sum())} px / {int((b_cus > 1e-7).sum())} px')
    pkw = dict(pos=(1.5, 1.0, 3.0), at=(0.0, 0.0, 0.0), energy=300.0,
               decay_end=3.0)
    p_def, _a, _f = _beam(dict(pkw, decay='DEFAULT'), kind='POINT')
    p_cus, _a, _f = _beam(dict(pkw, decay='CUSTOM'), kind='POINT')
    check('a Point lamp keeps the rule: Custom Range\'s end is its beam\'s '
          'reach', int((p_cus > 1e-7).sum()) < 0.6 * int((p_def > 1e-7).sum()),
          f'{int((p_cus > 1e-7).sum())} px against '
          f'{int((p_def > 1e-7).sum())}')


# ---------------------------------------------------------- (g) the shafts


def _old_resample(img, xs, ys):
    h, w = img.shape[:2]
    x = np.clip(xs, 0, w - 1)
    y = np.clip(ys, 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


def _old_light_shafts(rgb, st, sources):
    """The stage before this round, verbatim."""
    if not sources:
        return rgb
    h, w = rgb.shape[:2]
    lum = M.luminance(rgb)
    bright = rgb * (lum > float(st.shaft_threshold))[:, :, None]
    if not np.any(bright):
        return rgb
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    total = np.zeros_like(rgb)
    for (sx, sy), strength in sources:
        if strength <= 0.0:
            continue
        cx = (sx * 0.5 + 0.5) * (w - 1)
        cy = (sy * 0.5 + 0.5) * (h - 1)
        acc = np.zeros_like(rgb)
        weight = 0.0
        decay = float(st.shaft_decay)
        amp = 1.0
        steps = max(int(st.shaft_samples), 1)
        for i in range(steps):
            t = (i + 1) / steps * float(st.shaft_length)
            px = xx + (cx - xx) * t
            py = yy + (cy - yy) * t
            acc += _old_resample(bright, px, py) * amp
            weight += amp
            amp *= decay
        total += acc / max(weight, 1e-6) * strength
    return (rgb + total).astype(np.float32)


def _sun_ahead(**kw):
    return _lamp('SUN', pos=(0.0, 6.0, 6.0), at=(0.0, -2.0, 0.0), energy=3.0,
                 **kw)


def test_g_shaft_march_is_the_old_sums():
    """fx-shafts-post-cost: every lamp with Volumetric above zero adds a
    full-frame march to the post chain. The march is the same sums in
    the same order (F12 is bitwise what it was) without the 2-D fancy
    indexing, visiting only the pixels that can receive light, the steps
    on a few threads."""
    st = _settings()
    rng = np.random.default_rng(3)
    yy, xx = np.mgrid[0:240, 0:320].astype(f32)
    pool = (0.25 + 0.9 * np.exp(-(((xx - 200) / 50) ** 2
                                  + ((yy - 150) / 30) ** 2)))
    cases = [('noise', rng.random((120, 160, 3)).astype(f32) * 1.2),
             ('a lamp pool', np.repeat(pool[:, :, None], 3, 2).astype(f32)),
             ('a bright edge column', np.concatenate(
                 [np.full((96, 9, 3), 1.5, f32), np.zeros((96, 119, 3), f32)],
                 1)),
             ('negative pixels', rng.random((96, 128, 3)).astype(f32) * 2.4
              - 0.9),
             ('a large frame (threads)', rng.random((300, 400, 3)).astype(f32)
              * 1.2)]
    bad = []
    for name, img in cases:
        for src in ([((0.3, 1.2), 1.0), ((-0.5, 0.4), 0.5)],
                    [((2.5, -3.0), 1.0)], [((0.0, 0.0), 0.7)]):
            if not np.array_equal(POST.light_shafts(img, st, src),
                                  _old_light_shafts(img, st, src)):
                bad.append(f'{name} {src[0][0]}')
    st2 = _settings(shaft_length=1.0, shaft_samples=7, shaft_decay=0.6)
    if not np.array_equal(
            POST.light_shafts(cases[0][1], st2, [((0.3, 1.2), 1.0)]),
            _old_light_shafts(cases[0][1], st2, [((0.3, 1.2), 1.0)])):
        bad.append('length 1.0')
    check('light_shafts is bitwise the old stage on five frames x three '
          'source sets, and at Shaft Length 1', not bad, ', '.join(bad))
    calls = {'n': 0}
    real = POST._resample

    def counting(*a, **k):
        calls['n'] += 1
        return real(*a, **k)
    POST._resample = counting
    try:
        POST.light_shafts(cases[0][1], st, [((0.3, 1.2), 1.0)])
    finally:
        POST._resample = real
    check('the full-size march no longer runs the 2-D fancy-indexed '
          'resample (24 whole-frame calls per source before)',
          calls['n'] == 0, f'{calls["n"]} calls')
    if hasattr(POST, '_shaft_window'):
        # the window is where the cost goes: a compact bright region is
        # visited, the rest of the frame is not
        seen = []
        real_step = POST._shaft_step

        def spy(flat, w, h, xx_, yy_, cx, cy, t):
            seen.append(xx_.size)
            return real_step(flat, w, h, xx_, yy_, cx, cy, t)
        POST._shaft_step = spy
        try:
            POST.light_shafts(cases[1][1], st, [((0.3, 1.2), 1.0)])
        finally:
            POST._shaft_step = real_step
        whole = 240 * 320 * len(seen)
        check('a lamp pool\'s march visits a fraction of the frame',
              0 < sum(seen) < 0.5 * whole,
              f'{sum(seen)} of {whole} pixel-steps')
    # the proxy (the viewport's march)
    try:
        full = POST.light_shafts(cases[1][1], st, [((0.3, 1.2), 1.0)])
        prox = POST.light_shafts(cases[1][1], st, [((0.3, 1.2), 1.0)],
                                 proxy_pixels=6000)
    except TypeError:
        check('light_shafts takes a proxy pixel budget', False)
        return
    add = full - cases[1][1]
    dd = np.abs(prox - full)
    check('a proxy march of 6000 px on a 76800 px frame keeps the shafts: '
          'mean error under 2% of their peak',
          float(dd.mean()) < 0.02 * float(add.max())
          and float(np.abs(prox - cases[1][1]).max()) > 0.5 * float(add.max()),
          f'mean {float(dd.mean()):.4f}, max {float(dd.max()):.3f}, shafts '
          f'add up to {float(add.max()):.3f}')
    check('a budget at or above the frame is the full march bitwise',
          np.array_equal(POST.light_shafts(cases[0][1], st,
                                           [((0.3, 1.2), 1.0)],
                                           proxy_pixels=10 ** 7),
                         POST.light_shafts(cases[0][1], st,
                                           [((0.3, 1.2), 1.0)])))


def _viewport_refine(sc, st, w, h, rect=None, draft=False):
    from ..preview import Viewport, shape_settings
    shape_settings(st, w, h)
    view, proj, _v, _e = R.camera_matrices(sc.camera, w, h)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(f32),
                 projection=proj.astype(f32), type='PERSP')
    vp = Viewport()
    vp.set_scene(sc, st)
    vp.abort = False
    key = vp._key(cam, w, h, rect)
    vp._render(None, sc, st, cam, w, h, key, vp.version, draft, rect)
    return vp.frame, cam


def test_g_shafts_in_the_rendered_viewport():
    """fx-vol-shafts-viewport: F12 hands the post chain the shaft sources
    the render computed; the viewport's post call passed the flares only,
    so a Sun's Volumetric drew nothing in the rendered viewport."""
    from .. import preview as PV
    w, h = 160, 120
    proxy = getattr(PV, 'SHAFT_PROXY_PX', 0)

    def rig(vol):
        st = _settings(w, h)
        st.preview_scale = 1
        st.render_device = 'CPU'
        sc = _scene(_sun_ahead(volumetric=vol), st, cam_pos=FACE_POS,
                    cam_aim=FACE_AIM, world=(1.0, 1.0, 1.0))
        return sc, st

    sc, st = rig(1.0)
    frame, cam = _viewport_refine(sc, st, w, h)
    sc0, st0 = rig(0.0)
    plain, _c = _viewport_refine(sc0, st0, w, h)
    check('the viewport parks a refine', frame is not None
          and plain is not None)
    if frame is None or plain is None:
        return
    moved = float(np.abs(frame[..., :3] - plain[..., :3]).mean())
    check('a Sun\'s Volumetric moves the rendered viewport\'s refine (it '
          'changed no pixel)', moved > 0.02, f'mean change {moved:.4f}')
    # the reference: the same render through F12's post arguments, with
    # the viewport's proxy budget
    sc.camera = cam
    ref_st = st.copy()
    ref_st._viewport = True
    img = R.render(sc, ref_st)
    kw = dict(frame=getattr(sc, 'frame', 0),
              cvg=getattr(sc, 'last_cvg', None), coverage=None,
              seed=ref_st.seed, target_size=(w, h),
              flare_sources=getattr(sc, 'last_flares', None))
    try:
        ref = POST.process(img, ref_st, shaft_sources=sc.last_shafts,
                           shaft_proxy=proxy, **kw)
    except TypeError:
        ref = POST.process(img, ref_st, shaft_sources=sc.last_shafts, **kw)
    check('the refine IS render() + post with the render\'s shaft sources '
          '(the proxy march)', frame.shape == ref.shape
          and np.array_equal(frame, ref))
    full = POST.process(img, ref_st, shaft_sources=sc.last_shafts, **kw)
    err = float(np.abs(np.asarray(full)[..., :3]
                       - frame[..., :3]).mean())
    check('...and close to the full-size march F12 runs (8-bit steps)',
          err < 0.02, f'mean |viewport - F12 post| {err:.4f}')
    # a draft keeps motion fluid: no shafts
    sc_d, st_d = rig(1.0)
    draft, _c = _viewport_refine(sc_d, st_d, w, h, draft=True)
    sc_d0, st_d0 = rig(0.0)
    draft0, _c = _viewport_refine(sc_d0, st_d0, w, h, draft=True)
    check('a draft frame carries no shafts (it exists to keep motion fluid)',
          draft is not None and draft0 is not None
          and draft.shape == draft0.shape and np.array_equal(draft, draft0))
    # the camera-frame crop: the anchors ride through it
    if not hasattr(PV, 'crop_shaft_sources'):
        check('the shaft anchors are carried through the camera-frame crop',
              False)
        return
    W0, H0 = 200, 150
    rect = (40, 30, 140, 105)
    src = [((0.3, 1.2), 1.0), ((-0.5, 0.4), 0.5), ((0.0, 0.0), 0.25)]
    mv = PV.crop_shaft_sources(src, (W0, H0), rect)
    bw, bh = rect[2] - rect[0], rect[3] - rect[1]
    worst = 0.0
    for ((x, y), amt), ((mx, my), amt2) in zip(src, mv):
        full_px = ((x * 0.5 + 0.5) * (W0 - 1), (y * 0.5 + 0.5) * (H0 - 1))
        crop_px = ((mx * 0.5 + 0.5) * (bw - 1), (my * 0.5 + 0.5) * (bh - 1))
        worst = max(worst, abs(crop_px[0] - (full_px[0] - rect[0])),
                    abs(crop_px[1] - (full_px[1] - rect[1])),
                    abs(amt - amt2))
    check('an anchor through the crop is the same screen point: post\'s '
          'pixel in the cropped frame = the region\'s pixel - the crop '
          'origin', worst < 1e-9, f'worst {worst:.2e} px')
    sc, st = rig(1.0)
    vrect = (0.25, 0.25, 0.75, 0.75, True)
    crop_frame, cam = _viewport_refine(sc, st, w, h, rect=vrect)
    sc.camera = cam
    ref_st = st.copy()
    ref_st._viewport = True
    ref_st.use_border = True
    ref_st.border_min_x, ref_st.border_min_y = 0.25, 0.25
    ref_st.border_max_x, ref_st.border_max_y = 0.75, 0.75
    img = R.render(sc, ref_st)
    x0, y0, x1, y1 = R.region_pixels(ref_st, w, h)
    crop = np.ascontiguousarray(img[y0:y1, x0:x1])
    ref = POST.process(crop, ref_st, frame=getattr(sc, 'frame', 0), cvg=None,
                       coverage=None, seed=ref_st.seed,
                       target_size=(x1 - x0, y1 - y0),
                       flare_sources=getattr(sc, 'last_flares', None),
                       shaft_sources=PV.crop_shaft_sources(
                           sc.last_shafts, (w, h), (x0, y0, x1, y1)),
                       shaft_proxy=proxy)
    noshaft = POST.process(crop, ref_st, frame=getattr(sc, 'frame', 0),
                           cvg=None, coverage=None, seed=ref_st.seed,
                           target_size=(x1 - x0, y1 - y0),
                           flare_sources=getattr(sc, 'last_flares', None))
    check('camera-frame mode: the parked frame is post over the cropped '
          'render with the anchors moved, and it carries shafts',
          crop_frame is not None and crop_frame.shape == ref.shape
          and np.array_equal(crop_frame, ref)
          and float(np.abs(np.asarray(ref) - np.asarray(noshaft)).mean())
          > 0.01)
    # the whole frame's shafts and the cropped frame's agree where they
    # overlap: the anchor did not slide (same picture within the march's
    # own edge handling)
    whole_st = st.copy()
    whole_st._viewport = True
    whole = R.render(sc, whole_st)
    wpost = POST.process(whole, whole_st, frame=getattr(sc, 'frame', 0),
                         cvg=None, coverage=None, seed=whole_st.seed,
                         target_size=(w, h),
                         flare_sources=getattr(sc, 'last_flares', None),
                         shaft_sources=sc.last_shafts)
    inner = np.asarray(wpost)[y0 + 6:y1 - 6, x0 + 6:x1 - 6, :3]
    cin = np.asarray(crop_frame)[6:-6, 6:-6, :3]
    check('...at the place the whole region\'s shafts fall (mean '
          'difference under 0.03 inside the crop)',
          float(np.abs(inner - cin).mean()) < 0.03,
          f'{float(np.abs(inner - cin).mean()):.4f}')


def test_g_shafts_survive_the_worker_pool():
    """fx-vol-shafts-pool: the pooled road returns pixels only, so
    scene.last_shafts was never set and F12 with Worker Processes dropped
    every lamp's shafts."""
    from ..core import parallel as P
    st = _settings(320, 240)
    sc = _scene(_sun_ahead(volumetric=1.0), st, cam_pos=FACE_POS,
                cam_aim=FACE_AIM, world=(1.0, 1.0, 1.0))
    ref = np.asarray(R.render(sc, st))
    want = list(sc.last_shafts)
    sc.last_shafts = None
    try:
        img, err = P.render_parallel(sc, st, workers=2, scene_key='r254')
    finally:
        P.shutdown()
    if img is None:
        note(f'the worker pool is unavailable here ({err}): the pooled '
             'shaft sources were not exercised')
        return
    check('the pooled frame is the in-process frame before post',
          np.array_equal(np.asarray(img), ref))
    got = getattr(sc, 'last_shafts', None)
    check('the pooled road hands over the shaft sources the in-process '
          'render computes', bool(want) and got is not None
          and len(got) == len(want)
          and all(abs(a[0][0] - b[0][0]) < 1e-6 and abs(a[0][1] - b[0][1])
                  < 1e-6 and a[1] == b[1] for a, b in zip(got, want)),
          f'{got} against {want}')


# ------------------------------------------------------------- (h) the hemi


def test_h_hemi_follows_its_direction():
    """fx-hemi-anchor, fx-hemi-caustics: a Hemi is a Sun with the
    Hemisphere flag -- directional -- but its shafts and flare were
    anchored on the lamp OBJECT, and its Caustics sliders did nothing."""
    st = _settings()
    dirv = np.array([0.0, -8.0, -6.0])
    anchors = {}
    for kind in ('SUN', 'HEMI'):
        for pos in ((0.0, 6.0, 6.0), (3.0, 2.0, 3.0)):
            lt = _lamp(kind, pos=pos, at=tuple(np.asarray(pos) + dirv),
                       volumetric=1.0, flare=1.0)
            sc = _scene(lt, st, cam_pos=FACE_POS, cam_aim=FACE_AIM,
                        world=(1.0, 1.0, 1.0))
            vp = R.camera_matrices(sc.camera, W, H)[2]
            g = CRg.GBuffer(W, H)
            CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)
            sh = R.shaft_sources(sc, st, vp)
            fl = R._flare_sources(sc, st, g, vp)
            anchors[(kind, pos)] = (
                [tuple(round(v, 5) for v in s[0]) for s in sh],
                [(round(f['x'], 5), round(f['y'], 5)) for f in fl])
    sun = anchors[('SUN', (0.0, 6.0, 6.0))]
    check('a Sun\'s shaft anchor is its place in the sky (one source)',
          len(sun[0]) == 1)
    check('a Hemi\'s shaft and flare anchors are its Sun\'s, wherever the '
          'lamp object sits (they followed the object)',
          all(anchors[k] == sun for k in anchors),
          '; '.join(f'{k[0]} at {k[1]}: {v[0]}' for k, v in anchors.items()))
    # caustics on a hemi
    c0, _s = _cpu(_lamp('HEMI', pos=(0, 6, 6)), time=0.5, **TOP)
    c1, sc1 = _cpu(_lamp('HEMI', pos=(0, 6, 6), caustics=1.0,
                         caustics_scale=3.0), time=0.5, **TOP)
    changed = int((np.abs(c1 - c0).max(axis=2) > 1.0 / 255.0).sum())
    check('Caustics on a Hemi draws the web (0 px before)',
          changed > 0.3 * W * H, f'{changed} of {W * H} px')
    check('...tiled at Caustic Scale world units, as a Sun\'s',
          getattr(sc1.lights[0], 'cookie', None) is not None
          and float(sc1.lights[0].cookie_scale) == 3.0)
    _roads('a caustic Hemi', _lamp('HEMI', pos=(0, 6, 6), caustics=1.0,
                                   caustics_scale=3.0), time=0.5, **TOP)


# --------------------------------------------------------- (i) the caustics


def _caustic_tile(kind, c, scale, t=0.5, speed=1.0):
    lt = _lamp(kind, pos=(0, 0, 6), caustics=c, caustics_scale=scale,
               caustics_speed=speed)
    sc = _scene(lt, _settings())
    sc.time = t
    R._synth_caustic_cookies(sc)
    return np.asarray(lt.cookie.pixels, np.float64), lt


def test_i_caustics_scale_and_energy():
    """fx-caustics-scale-spot: Caustic Scale 8..64 moved no pixel on a
    Spot (the Sun's seamless 8-cell period clamped it).
    fx-caustics-dims-lamp: the web pivoted on 0.35 with a mean of 0.17,
    so Caustics 1.0 threw away 29% of the lamp's light and 2.0 half."""
    from ..core import patterns as PT
    frames = {}
    for sc_ in (4.0, 8.0, 16.0, 32.0, 64.0):
        frames[sc_], _s = _cpu(_lamp('SPOT', pos=(0, 0, 6), energy=600.0,
                                     caustics=1.0, caustics_scale=sc_),
                               time=0.5, **TOP)
    steps = [int((np.abs(frames[a] - frames[b]).max(axis=2)
                  > 1.0 / 255.0).sum())
             for a, b in ((8.0, 16.0), (16.0, 32.0), (32.0, 64.0))]
    check('Caustic Scale 8 -> 16 -> 32 -> 64 each move a spot\'s web (0 px '
          'each before)', min(steps) > 500, str(steps))
    # at 8 and under the web is the one it always was
    px4, _l = _caustic_tile('SPOT', 1.0, 4.0)
    ax = (np.arange(256, dtype=f32) + 0.5) / 256
    uu = np.broadcast_to(ax[None, :] * 4.0, (256, 256))
    vv = np.broadcast_to(ax[:, None] * 4.0, (256, 256))
    web = PT.caustic_web(uu, vv, time=0.5, period=8)
    old = np.maximum(1.0 + (web - 0.35) * 1.6 * 1.0, 0.0)
    ratio = px4[:, :, 0] / np.maximum(old, 1e-9)
    live = old > 1e-3
    check('at Scale 4 the tile is the old web times one constant (the '
          'pattern did not move)', px4.shape == (256, 256, 3)
          and float(ratio[live].max() - ratio[live].min()) < 1e-5,
          f'ratio {float(ratio[live].min()):.5f} .. '
          f'{float(ratio[live].max()):.5f}')
    # above 8: one lattice, so the web zooms
    px16, _l = _caustic_tile('SPOT', 1.0, 16.0)
    px32, _l = _caustic_tile('SPOT', 1.0, 32.0)
    a16 = px16[:, :, 0]
    h16 = a16.shape[0]
    q32 = px32[:px32.shape[0] // 2, :px32.shape[1] // 2, 0]
    k = h16 // q32.shape[0]
    a16d = a16.reshape(q32.shape[0], k, q32.shape[1], k).mean(axis=(1, 3))
    same = float(np.abs(a16d - q32).mean())
    other, _l = _caustic_tile('SPOT', 1.0, 16.0, t=40.0)
    unrelated = float(np.abs(other[:, :, 0] - a16).mean())
    check('Scale 16 is the first quarter of Scale 32\'s web (one lattice: '
          'the slider zooms, it does not re-seed)',
          same < 0.4 * unrelated,
          f'mean difference {same:.3f} against {unrelated:.3f} for an '
          'unrelated web')
    check('the tile keeps 32 texels a cell up to 512 px',
          px16.shape[0] == 512 and px32.shape[0] == 512)
    _roads('a caustic spot at Scale 64',
           _lamp('SPOT', pos=(0, 0, 6), energy=600.0, caustics=1.0,
                 caustics_scale=64.0), time=0.5, **TOP)
    # the lamp keeps its light
    means = {}
    for c in (0.5, 1.0, 2.0, 4.0):
        means[c] = [float(_caustic_tile('SUN', c, 5.0, t=t)[0].mean())
                    for t in (0.0, 0.5, 3.0, 7.0)]
    worst = max(abs(m - 1.0) for v in means.values() for m in v)
    check('a Sun\'s seamless tile averages 1 at Caustics 0.5 / 1 / 2 / 4 '
          'and four times (it was 0.86 / 0.71 / 0.49 / 0.64)',
          worst < 0.02, ', '.join(f'{c}: {np.mean(v):.3f}'
                                  for c, v in means.items()))
    base, _s = _cpu(_lamp('SUN', pos=(0, 0.001, 6)), time=0.5, **TOP)
    lit, _s = _cpu(_lamp('SUN', pos=(0, 0.001, 6), caustics=1.0,
                         caustics_scale=3.0), time=0.5, **TOP)
    r = float(_lum(lit).mean() / _lum(base).mean())
    check('a floor under a caustic Sun keeps its mean luminance (several '
          'tiles in view; it lost 30%)', abs(r - 1.0) < 0.03, f'{r:.4f}')
    check('...while the web is still there: crests over 2x, cells under '
          '0.8x', float(_lum(lit).max() / _lum(base).max()) > 2.0
          and float(np.percentile(_lum(lit) / np.maximum(_lum(base), 1e-9),
                                  20)) < 0.8)
    if hasattr(R, 'caustic_mean'):
        check('the divisor is one constant per strength (nothing breathes '
              'as the web writhes): 0.857 / 0.715 / 0.493 / 0.637',
              all(abs(R.caustic_mean(c) - m) < 2e-3 for c, m in
                  ((0.5, 0.8574), (1.0, 0.7149), (2.0, 0.4926),
                   (4.0, 0.6370))),
              ', '.join(f'{R.caustic_mean(c):.4f}' for c in (0.5, 1, 2, 4)))


# ------------------------------------------------------ (j) cookie signature


def test_j_cookie_signatures():
    """fx-gpu-cookie-spotsize-stale: a SPOT's cookie lookup bakes
    2 tan(spot_size/2), but the Spot Size was in the plan signature only
    for a MAP-shadowed lamp. fx-gpu-cookie-checksum-stale: the image was
    fingerprinted by the sum of every 7th texel."""
    ck = _checker(64, 8)

    def spot(size, **kw):
        base = dict(pos=(0, 0, 6), energy=600.0, spot_size=size,
                    cookie=_cookie(ck))
        base.update(kw)
        return _lamp('SPOT', **base)
    check('Spot Size is in a cookie spot\'s signature without a baked map',
          GSH._light_sig(spot(1.2)) != GSH._light_sig(spot(0.7)))
    check('a plain spot\'s Spot Size drag stays a plan-cache HIT',
          GSH._light_sig(_lamp('SPOT', spot_size=1.2))
          == GSH._light_sig(_lamp('SPOT', spot_size=0.7)))
    check('a cookie spot\'s Power and position edits stay HITs',
          GSH._light_sig(spot(1.2))
          == GSH._light_sig(spot(1.2, energy=50.0, position=(1.0, 0.0, 5.0))))
    for label, mk, t in (
            ('cookie spot, no shadow', lambda s: spot(s), None),
            ('cookie spot with Radius 0.1', lambda s: spot(s, radius=0.1),
             None),
            ('caustic spot', lambda s: _lamp(
                'SPOT', pos=(0, 0, 6), energy=600.0, spot_size=s,
                caustics=1.0), 0.5)):
        st = _settings()
        _gpu(mk(1.2), st, time=t, **TOP)
        g, hit = _gpu(mk(0.7), _settings(), time=t, clear=False, **TOP)
        c, _s = _cpu(mk(0.7), _settings(), time=t, **TOP)
        check(f'{label}: narrowing the cone with the plan cache warm is the '
              'CPU frame (it kept the old projection scale)',
              g is not None and float(np.abs(g - c).max()) < BAR,
              'refused' if g is None else
              f'hit={hit}, max |CPU - GPU| {float(np.abs(g - c).max()):.2e}')
    # every texel is fingerprinted
    img = _asym(64)
    flip = img[:, ::-1].copy()
    dab = img.copy()
    dab[30:33, 37:40] = 0.0          # between the every-7th samples
    la = _lamp('SUN', cookie=_cookie(img))
    check('a mirrored 64 px slide changes the cookie signature (the '
          'every-7th lattice maps onto itself: 63 = 9 x 7)',
          GSH._cookie_sig(la)
          != GSH._cookie_sig(_lamp('SUN', cookie=_cookie(flip))))
    check('a 3x3 dab between the lattice samples changes it too',
          GSH._cookie_sig(la)
          != GSH._cookie_sig(_lamp('SUN', cookie=_cookie(dab))))
    check('the same pixels in a new array are the same signature',
          GSH._cookie_sig(la)
          == GSH._cookie_sig(_lamp('SUN', cookie=_cookie(img.copy()))))
    for kind, kw in (('SPOT', dict(pos=(0, 0, 6), energy=600.0)),
                     ('SUN', dict(pos=(0, 6, 6), cookie_scale=6.0)),
                     ('AREA', dict(pos=(0, 0, 3), energy=1.0,
                                   area_size=(4.0, 4.0)))):
        _gpu(_lamp(kind, cookie=_cookie(img), **kw), _settings(), **TOP)
        g, hit = _gpu(_lamp(kind, cookie=_cookie(flip), **kw), _settings(),
                      clear=False, **TOP)
        c, _s = _cpu(_lamp(kind, cookie=_cookie(flip), **kw), _settings(),
                     **TOP)
        check(f'{kind}: a mirrored slide with the plan cache warm is the CPU '
              'frame (the device kept the old image)',
              g is not None and float(np.abs(g - c).max()) < BAR,
              'refused' if g is None else
              f'hit={hit}, max |CPU - GPU| {float(np.abs(g - c).max()):.2e}')
    # the CPU road's own caches: the same lamp, its array edited in place
    px = _asym(64)
    lt = _lamp('SPOT', pos=(0, 0, 6), energy=600.0, radius=0.05,
               cookie=_cookie(px))
    st = _settings()
    sc = _scene(lt, st, **TOP)
    a = np.asarray(R.render(sc, st))[..., :3].astype(np.float64)
    px[:] = px[:, ::-1].copy()
    lt.cookie.pixels[:] = px
    b = np.asarray(R.render(sc, st))[..., :3].astype(np.float64)
    ref, _s = _cpu(_lamp('SPOT', pos=(0, 0, 6), energy=600.0, radius=0.05,
                         cookie=_cookie(_asym(64)[:, ::-1].copy())),
                   _settings(), **TOP)
    check('the CPU road notices a slide mirrored IN PLACE on the next frame '
          '(the blurred preparation was served stale)',
          float(np.abs(a - b).max()) > 0.05 and np.array_equal(b, ref),
          f'frame moved {float(np.abs(a - b).max()):.3f}, against a fresh '
          f'lamp {float(np.abs(b - ref).max()):.2e}')


# ----------------------------------------------- (k) the screen spotlight


def test_k_screen_spot_honours_specular_only():
    """screen-spot-specular-only-01: the Screen Spotlight's lobe is
    diffuse light, added on its own road before the lamp's diffuse gates
    -- Specular Only left the ellipse untouched."""
    kw = dict(pos=(0.0, 2.5, 4.0), energy=600.0)
    plain, _s = _cpu(_lamp('SPOT', screen_spot=True, **kw), box=True)
    for flag in ('specular_only', 'affect_diffuse'):
        v = (flag == 'specular_only')
        f, _s = _cpu(_lamp('SPOT', screen_spot=True, **dict(kw, **{flag: v})),
                     box=True)
        check(f'a Screen Spotlight with {flag}={v} adds no diffuse light on '
              'Lambert surfaces (the frame did not change)',
              float(plain.sum()) > 1.0 and float(np.abs(f).max()) == 0.0,
              f'lit sum {float(plain.sum()):.1f} -> {float(f.sum()):.1f}')
    lt = _lamp('SPOT', screen_spot=True, specular_only=True, **kw)
    c, _s = _cpu(lt, box=True)
    g, eng = _gpu_full(_lamp('SPOT', screen_spot=True, specular_only=True,
                             **kw), box=True)
    check('the whole GPU road agrees', eng
          and float(np.abs(c - g).max()) < BAR,
          f'engaged={eng}, max {float(np.abs(c - g).max()):.2e}')
    consts = _consts_literal()
    consts.update(light_texels=True, fogtab=True)
    on = '\n'.join(GM._one_light_source(
        0, _lamp('SPOT', screen_spot=True, **kw), consts))
    off = '\n'.join(GM._one_light_source(
        0, _lamp('SPOT', screen_spot=True, specular_only=True, **kw), consts))
    check('the emitted block drops only the surface add: the fog lobe (the '
          'lamp in the air) stays',
          'total += hal_sst;' in on and 'total += hal_sst;' not in off
          and 'hal_spotfog += ss_fog;' in off
          and off.replace('    hal_spotfog', '    total += hal_sst;\n'
                          '    hal_spotfog', 1) == on)
    a, _s = _cpu(_lamp('SPOT', screen_spot=True, diffuse_only=True, **kw),
                 box=True)
    check('Diffuse Only leaves the ellipse as it was', np.array_equal(a,
                                                                       plain))


# ------------------------------------------------------------ (l) the panel


class _Layout:
    """A recording layout: (property, label) rows and the note texts."""

    def __init__(self, rec):
        self.rec = rec
        self.active = True
        self.use_property_split = False

    def prop(self, data, name, **kw):
        self.rec.append((name, kw.get('text')))

    def column(self, **kw):
        return _Layout(self.rec)

    row = column
    box = column

    def separator(self, **kw):
        pass

    def label(self, **kw):
        self.rec.append(('label', str(kw.get('text', ''))))

    def template_ID(self, data, name, **kw):
        self.rec.append((name, None))

    def operator(self, idn, **kw):
        self.rec.append(('op:' + idn, None))
        return types.SimpleNamespace(scope=None)


def _draw(UI, defaults, kind, cones=None, **fields):
    hs = types.SimpleNamespace(**defaults)
    hs.cookie = None
    for k, v in fields.items():
        setattr(hs, k, v)
    light = types.SimpleNamespace(type=kind, color=(1, 1, 1), energy=10.0,
                                  halcyon=hs, shadow_soft_size=0.0)
    scene = None if cones is None else types.SimpleNamespace(
        halcyon=types.SimpleNamespace(spot_cones=cones,
                                      light_falloff_default='INVERSE_SQUARE'))
    rec = []
    panel = UI.HALCYON_PT_light()
    panel.layout = _Layout(rec)
    UI.HALCYON_PT_light.draw(panel, types.SimpleNamespace(
        engine=UI.ENGINE, light=light, scene=scene, object=None))
    return rec


def _labels(rec):
    return [t for n, t in rec if n == 'label']


def test_l_panel_says_what_each_dial_needs():
    """volumetric-needs-switch-01 / fx-vol-cones-switch,
    fx-vol-shafts-conditions, blend-dead-nonbi-01, hotspot-dead-01's
    panel half, fx-cookie-area-blur-hidden."""
    from . import fakebpy
    fakebpy.install()
    from .. import properties as PR
    from .. import ui as UI
    ann = PR.HalcyonLightSettings.__annotations__
    defaults = {k: (tuple(v.kw.get('default')) if isinstance(
        v.kw.get('default'), (tuple, list)) else v.kw.get('default'))
        for k, v in ann.items()}
    # the beam switch, where the slider is
    for kind in ('SPOT', 'POINT', 'AREA'):
        off = _draw(UI, defaults, kind, cones=False, volumetric=0.5)
        on = _draw(UI, defaults, kind, cones=True, volumetric=0.5)
        zero = _draw(UI, defaults, kind, cones=False, volumetric=0.0)
        check(f'{kind}: Volumetric above zero with the render\'s switch off '
              'draws the reason AND the switch in the lamp panel',
              any(n == 'spot_cones' for n, _t in off)
              and any('No beam' in t for t in _labels(off))
              and not any(n == 'spot_cones' for n, _t in on)
              and not any(n == 'spot_cones' for n, _t in zero))
    sun = _draw(UI, defaults, 'SUN', cones=False, volumetric=0.5)
    check('a Sun never draws the beam switch (it has no beam)',
          not any(n == 'spot_cones' for n, _t in sun))
    check('the render panel is titled for what it gates: Volumetric Lights',
          UI.HALCYON_PT_spot_cones.bl_label == 'Volumetric Lights')
    desc = PR.DESCRIPTIONS
    check('the switch\'s tooltip names Spot, Point and Area lights, and '
          'the Density tooltip its unit',
          'Point' in desc['spot_cones'] and 'Area' in desc['spot_cones']
          and '4 pi' in desc['spot_cone_density'].replace('\n', ' '))
    vt = str(ann['volumetric'].kw.get('description'))
    check('the Volumetric tooltip names the render switch, says when the '
          'value is the Light Shafts instead and what a Sun needs',
          'Volumetric Lights' in vt and 'Shaft Threshold' in vt
          and 'switch off' in vt and 'ahead of the camera' in vt
          and len(vt) >= 40)
    check('Beam Density has a range (it had none: -1e6 .. 1e6)',
          PR.RANGES.get('spot_cone_density') is not None
          and PR.RANGES['spot_cone_density'][0] == 0.0
          and PR.RANGES['spot_cone_density'][1] >= 700.0)
    # the Sun's shaft status agrees with the render's own bounds
    if hasattr(UI, 'shaft_anchor_status'):
        st = _settings()
        sc = _scene(_sun_ahead(), st, cam_pos=FACE_POS, cam_aim=FACE_AIM)
        vp = R.camera_matrices(sc.camera, W, H)[2]
        rng = np.random.default_rng(5)
        wrong = 0
        n_on = 0
        for _ in range(400):
            d = rng.normal(size=3)
            d = d / np.linalg.norm(d)
            sc.lights[0].direction = tuple(float(v) for v in d)
            sc.lights[0].volumetric = 1.0
            src = R.shaft_sources(sc, st, vp)
            clip = np.append(-d, 0.0).astype(f32) @ vp.T
            co = None if abs(clip[3]) < 1e-9 else (
                float(clip[0] / clip[3]) * 0.5 + 0.5,
                float(clip[1] / clip[3]) * 0.5 + 0.5, float(clip[3]))
            status = UI.shaft_anchor_status(co)
            n_on += bool(src)
            if (status[1] == 'CHECKMARK') != bool(src):
                wrong += 1
        check('the Sun\'s Light Shafts status line agrees with '
              'render.shaft_sources on 400 random sun directions',
              wrong == 0 and 20 < n_on < 380, f'{wrong} disagree, {n_on} '
              'with a source')
        check('no camera, no line', UI.shaft_anchor_status(None) is None)
        check('behind the camera the line says so and how to fix it',
              'BEHIND' in UI.shaft_anchor_status((0.5, 0.5, -1.0))[0]
              and 'Tilt' in UI.shaft_anchor_status((0.5, 0.5, -1.0))[2])
    else:
        check('the Sun\'s Light Shafts get a status line', False)
    # the cone rows
    drawn = {}
    for law in ('BLENDER', 'POV', 'GL11', 'GX_FLAT', 'GX_RING2'):
        rec = _draw(UI, defaults, 'SPOT', spot_law=law)
        drawn[law] = ({n for n, _t in rec if n in ('hotspot',
                                                   'spot_exponent')},
                      _labels(rec))
    check('Hotspot is drawn under the laws that read it (Blender Internal, '
          'POV), the exponent under OpenGL and POV, neither under GX',
          drawn['BLENDER'][0] == {'hotspot'}
          and drawn['POV'][0] == {'hotspot', 'spot_exponent'}
          and drawn['GL11'][0] == {'spot_exponent'}
          and drawn['GX_FLAT'][0] == set() and drawn['GX_RING2'][0] == set(),
          str({k: sorted(v[0]) for k, v in drawn.items()}))
    check('every law but Blender Internal says Blend is not read; POV says '
          'what sets the edge instead',
          all(any('Blend' in t and 'not read' in t for t in drawn[law][1])
              for law in ('POV', 'GL11', 'GX_FLAT', 'GX_RING2'))
          and any('Hotspot sets' in t for t in drawn['POV'][1])
          and not any('Blend' in t for t in drawn['BLENDER'][1]))
    rec = _draw(UI, defaults, 'SPOT', hotspot=0.4)
    check('Blender Internal with a Hotspot says it replaces Blend',
          any('Hotspot replaces Blend' in t for t in _labels(rec)))
    rec = _draw(UI, defaults, 'SPOT', screen_spot=True)
    check('a Screen Spotlight says its cone dials are not read',
          any('Screen Spotlight' in t and 'not read' in t
              for t in _labels(rec)))
    ht = str(ann['hotspot'].kw.get('description'))
    check('the Hotspot tooltip says what it does under each law',
          'Blender Internal' in ht and 'replaces Blend' in ht
          and 'POV' in ht and len(ht) >= 40)
    # the cookie's blur
    hs_ck = dict(cookie_strength=1.0)
    rec = _draw(UI, defaults, 'AREA')
    check('an Area lamp draws its projection\'s blur (Blender draws no '
          'Radius for it)', ('shadow_soft_size', 'Projection Blur') in rec)
    for kind in ('POINT', 'SPOT', 'SUN'):
        if any(n == 'shadow_soft_size' for n, _t in
               _draw(UI, defaults, kind)):
            check(f'{kind}: no second Radius row', False)
    notes = {}
    for kind in ('POINT', 'SPOT', 'SUN', 'AREA'):
        hs = types.SimpleNamespace(**defaults)
        hs.cookie = object()
        light = types.SimpleNamespace(type=kind, color=(1, 1, 1),
                                      energy=10.0, halcyon=hs,
                                      shadow_soft_size=0.0)
        rec = []
        panel = UI.HALCYON_PT_light()
        panel.layout = _Layout(rec)
        UI.HALCYON_PT_light.draw(panel, types.SimpleNamespace(
            engine=UI.ENGINE, light=light, scene=None, object=None))
        notes[kind] = [t for t in _labels(rec) if 'softens' in t]
    check('the cookie note names the slider each kind has: Radius, Angle, '
          'Projection Blur',
          all(len(v) == 1 for v in notes.values())
          and 'Radius' in notes['POINT'][0] and 'Radius' in notes['SPOT'][0]
          and 'Angle' in notes['SUN'][0] and 'Radius' not in notes['SUN'][0]
          and 'Projection Blur' in notes['AREA'][0], str(notes))
    ct = str(ann['caustics'].kw.get('description'))
    check('the Caustics tooltip covers a hemi and says the lamp keeps its '
          'light', 'hemi' in ct and 'keeps its total' in ct)
    del hs_ck


# ------------------------------------------------------------ (m) strict GLSL


def test_m_strict_glsl_of_the_changed_blocks():
    """The changed emitter shapes through the strict front-end, when
    HALCYON_GLSLANG points at glslang."""
    from . import glsl_strict as GS
    if GS._glslang() is None:
        note('HALCYON_GLSLANG is not set: the strict front-end did not run '
             'on the hotspot / screen-spot blocks')
        return
    kw = dict(pos=(0.0, 2.5, 4.0), energy=600.0)
    cases = [('Hotspot 0.3', _lamp('SPOT', hotspot=0.3)),
             ('Hotspot past the cone', _lamp('SPOT', hotspot=2.0)),
             ('Hotspot + Square', _lamp('SPOT', hotspot=0.3,
                                        spot_square=True)),
             ('screen spot', _lamp('SPOT', screen_spot=True, **kw)),
             ('screen spot, Specular Only', _lamp(
                 'SPOT', screen_spot=True, specular_only=True, **kw)),
             ('cookie spot', _lamp('SPOT', cookie=_cookie(_checker()))),
             ('caustic hemi', _lamp('HEMI', pos=(0, 6, 6), caustics=1.0))]
    bad = []
    n = 0
    for label, lt in cases:
        st = _settings()
        sc = _scene(lt, st, box=True)
        sc.time = 0.5
        R.render(sc, st)
        view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
        g = CRg.GBuffer(W, H)
        CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                      depth_bits=st.depth_precision)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                         eye, W, H)
        GSH._PLAN_CACHE.clear()
        passes, why, _atl = GSH.plan_frame(job, g)
        if passes is None:
            bad.append(f'{label}: refused, {why}')
            continue
        for ps in passes:
            src = next((x for x in ps if isinstance(x, str)
                        and 'void main' in x), None)
            if src is None:
                continue
            n += 1
            ok, msg = GS.compile_source(src)
            if not ok:
                bad.append(f'{label}: {msg[:200]}')
    check(f'{n} planned passes of the changed light blocks compile under '
          'strict GLSL 4.30', not bad and n > 0, '; '.join(bad[:2]))


# --------------------------------------- (n) what nobody had measured before


def test_n_the_unmeasured_options():
    """The verifiers' missed options that came out working, pinned: the
    render's beam dials on Point and Area beams, several effect lamps at
    once on both roads, the beam under a render border, a coloured
    slide."""
    for kind, kw in (('POINT', dict(pos=(1.5, 1.0, 3.0), energy=300.0)),
                     ('AREA', dict(pos=(0.0, 2.5, 4.0), energy=1.0,
                                   area_size=(1.5, 1.5)))):
        b0, _a, _f = _beam(kw, kind=kind)
        b2, _a, _f = _beam(kw, kind=kind, st_kw=dict(spot_cone_density=2.0))
        bz, _a, _f = _beam(kw, kind=kind, st_kw=dict(spot_cone_density=0.0))
        br, _a, _f = _beam(kw, kind=kind, st_kw=dict(spot_cone_reach=2.0))
        bf, _a, _f = _beam(kw, kind=kind, st_kw=dict(spot_cone_falloff=1.0))
        check(f'{kind}: Beam Density is linear (x2, x0), Beam Reach shortens '
              'the beam, Beam Falloff reshapes it',
              abs(float(b2.sum() / b0.sum()) - 2.0) < 1e-3
              and float(np.abs(bz).max()) == 0.0
              and int((br > 1e-7).sum()) < 0.5 * int((b0 > 1e-7).sum())
              and abs(float(bf.sum() / b0.sum()) - 1.0) > 0.2,
              f'x{float(b2.sum() / b0.sum()):.3f}, reach 2: '
              f'{int((br > 1e-7).sum())} of {int((b0 > 1e-7).sum())} px, '
              f'falloff 1: x{float(bf.sum() / b0.sum()):.3f}')

    def lamps():
        ls = [_lamp('SPOT', pos=(0.0, 2.5, 4.0), energy=600.0,
                    cookie=_cookie(_checker()), volumetric=1.0, hotspot=0.4),
              _lamp('SUN', pos=(0, 6, 6), energy=1.0, caustics=1.0,
                    caustics_scale=3.0),
              _lamp('SPOT', pos=(-3.0, -1.0, 4.0), at=(-2, 0, 0),
                    energy=400.0, spot_size=0.8, caustics=1.0,
                    caustics_scale=16.0),
              _lamp('POINT', pos=(2.5, -1.5, 3.0), energy=200.0,
                    volumetric=0.5, flare=0.5),
              _lamp('AREA', pos=(3.0, 2.0, 3.0), at=(1, 0, 0), energy=0.4,
                    area_size=(1.0, 1.0), volumetric=1.0)]
        for i, lt in enumerate(ls):
            lt.name = f'L{i}'
        return ls
    c, _s = _cpu(lamps(), _settings(spot_cones=True), time=0.5, box=True)
    g, eng = _gpu_full(lamps(), _settings(spot_cones=True), time=0.5,
                       box=True)
    c2, _s = _cpu(lamps()[::-1], _settings(spot_cones=True), time=0.5,
                  box=True)
    check('five effect lamps at once (cookie + beam + Hotspot, caustic Sun, '
          'caustic spot at Scale 16, glowing point with a flare, area '
          'slab): the whole GPU road is the CPU frame',
          eng and float(np.abs(c - g).max()) < BAR,
          f'engaged={eng}, max |CPU - GPU| {float(np.abs(c - g).max()):.2e}')
    check('...in either lamp order',
          float(np.abs(c - c2).max()) < 1e-3 * float(c.max()),
          f'{float(np.abs(c - c2).max()):.2e}')
    skw = dict(pos=(0.0, 2.5, 4.0), energy=1000.0, volumetric=1.0)
    full, _s = _cpu(_lamp('SPOT', **skw), _settings(spot_cones=True),
                    box=True)
    stb = _settings(spot_cones=True, use_border=True, border_min_x=0.25,
                    border_min_y=0.25, border_max_x=0.75, border_max_y=0.75)
    reg, _s = _cpu(_lamp('SPOT', **skw), stb, box=True)
    x0, y0, x1, y1 = R.region_pixels(stb, W, H)
    check('under a render border the beam inside the rect is the full '
          'frame\'s bitwise',
          np.array_equal(reg[y0:y1, x0:x1], full[y0:y1, x0:x1]))
    mk = lambda px: _lamp('SPOT', pos=(0, 0, 6), energy=600.0,     # noqa: E731
                          cookie=_cookie(px))
    plain, _s = _cpu(_lamp('SPOT', pos=(0, 0, 6), energy=600.0), **TOP)
    rgb = np.ones((64, 64, 3), f32) * np.array([1.0, 0.5, 0.25], f32)
    tint, _s = _cpu(mk(rgb), **TOP)
    lit = plain[..., 0] > 0.05
    ratios = [float((tint[..., k][lit] / plain[..., k][lit]).mean())
              for k in range(3)]
    rgba = np.ones((64, 64, 4), f32)
    rgba[:, :32, 3] = 0.0
    alpha, _s = _cpu(mk(rgba), **TOP)
    check('a coloured slide tints the light per channel; its alpha is not '
          'read (the tooltip says so)',
          max(abs(r - w) for r, w in zip(ratios, (1.0, 0.5, 0.25))) < 1e-4
          and np.array_equal(alpha, plain), str([round(r, 4) for r in ratios]))
    from . import fakebpy
    fakebpy.install()
    from .. import properties as PR
    ck = str(PR.HalcyonLightSettings.__annotations__['cookie'].kw.get(
        'description'))
    check('...and the cookie tooltip states both', 'alpha' in ck
          and 'multipl' in ck)


def main():
    utf8_console()
    FAILS.clear()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(f'{t.__name__} raised')
    print()
    if FAILS:
        print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS))
        return 1
    print('all R254 spot / beam / effect checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
