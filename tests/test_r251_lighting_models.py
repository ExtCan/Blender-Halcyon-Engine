"""R251 LIGHT-B2 (1.90.0): the console light units and the POV-Ray finish.

F016 GX_LIGHT (GameCube: Lambert + the rational 'shininess' highlight
against the camera axis, Sun-only specular, 8-bit vertex output), F017
SEGA_MODEL2 / SEGA_MODEL3 (fixed viewer, exponents by squaring, on the
corner road -- no GLSL by design), F018 DS_FIXED (the unnormalised half
vector against the fixed line of sight, squared, through a 128-entry
shininess table, 5-bit output), F019 Brilliance, F020 Crand (+ Crand
Flickers per Frame), F021 Metallic (POV).

Every check reads as a sentence of what it proves. The first test pins
the 1.89.0 zip loudly and the identity at defaults bitwise (render AND
post.process); each feature then proves its semantic law, its GPU twin
(d == 0.0 or the stated bar), its refusal by name where it has one, and
the fake-device road where the wave entry asks.

    python -m halcyon.tests.test_r251_lighting_models
"""
import importlib
import sys
import traceback

import numpy as np

from ..core import post as PO
from ..core import raster
from ..core import render as R
from ..core import shading as SH
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..gpu import shade as GSH
from .scenebuild import _mesh_concat, demo_scene, look_at_matrix
from .test_render import _prev_engine, _sk, base_settings, render

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name
          + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


PREV_ZIP = 'halcyon-1.89.0.zip'
W, H = 96, 72


# ------------------------------------------------------------------ rigs

def _post_kw(sc, st):
    return dict(frame=1, seed=st.seed,
                target_size=(st.resolution_x, st.resolution_y),
                allow_resize=False, depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None))


def _gbuffer_for(sc, st):
    w, h = st.resolution_x, st.resolution_y
    view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    R._build_shadows(sc, st, sc.mesh)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    return g, job


def _twin(sc, st):
    """(cpu, sim, d_max, n_bad, passes, atlases, why): the CPU frame against
    the GLSL simulator over the covered pixels (TR's sim_vs_cpu shape)."""
    cpu = np.asarray(R.render(sc, st))
    g, job = _gbuffer_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, g)
    if passes is None:
        return cpu, None, None, None, None, None, why
    img, hit = GSH.simulate(job, g, passes, atl)
    if img is None:
        return cpu, None, None, None, passes, atl, hit
    cov = g.tri >= 0
    d = np.abs(img[cov] - cpu[cov][:, :3])
    return (cpu, img, float(d.max()), int((d.max(axis=1) > 1e-2).sum()),
            passes, atl, None)


def _facing_quad(st, model, diffuse=(1.0, 1.0, 1.0), gloss=25.0,
                 energy=6.0, cam_shift=0.0, light_dir=None, spec_level=0.9,
                 graph=None, lights=None):
    """One quad whose face normal IS the camera's axis (view[2, :3] of
    R.camera_matrices), shaded under one Sun: every pixel has the same N
    and the same axis, so a fixed-viewer highlight is constant across it
    (the law F011 / F016-F018 share). `cam_shift` moves the camera
    sideways along its own right vector with the SAME rotation (the axis
    is unchanged)."""
    w, h = st.resolution_x, st.resolution_y
    mw = look_at_matrix((0.0, -7.0, 2.5), (0.0, 0.0, 1.2))
    if cam_shift:
        mw = mw.copy()
        mw[:3, 3] += mw[:3, 0] * np.float32(cam_shift)
    cam = Camera(matrix_world=mw, lens=42.0, sensor=36.0, clip_start=0.1,
                 clip_end=200.0)
    view, _proj, _vp, eye = R.camera_matrices(cam, w, h)
    axis = np.asarray(view[2, :3], np.float32)
    axis = axis / np.linalg.norm(axis)
    right = np.asarray(view[0, :3], np.float32)
    up = np.asarray(view[1, :3], np.float32)
    centre = np.asarray(look_at_matrix((0.0, -7.0, 2.5), (0.0, 0.0, 1.2))
                        [:3, 3], np.float32) - axis * np.float32(6.0)
    s = 1.6
    V = np.stack([centre - right * s - up * s, centre + right * s - up * s,
                  centre + right * s + up * s, centre - right * s + up * s]
                 ).astype(np.float32)
    N = np.tile(axis[None, :], (4, 1)).astype(np.float32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    mesh = _mesh_concat([(V, N, UV, T, 0, 0)])
    mat = Material(name='Quad', index=0, model=model, diffuse=diffuse,
                   specular=(1.0, 1.0, 1.0), specular_level=spec_level,
                   glossiness=gloss, ambient_level=0.0, graph=graph)
    if light_dir is None:
        # a Sun a little off the axis, so the highlight is neither the
        # peak nor nothing
        light_dir = tuple(float(v) for v in
                          -(axis * 0.9 + up * 0.35 + right * 0.2))
    if lights is None:
        lights = [Light(type='SUN', name='Sun', direction=light_dir,
                        color=(1.0, 1.0, 1.0), energy=energy,
                        shadow='NONE')]
    world = World(mode='SOLID', color=(0.0, 0.0, 0.0), ambient=(0, 0, 0))
    objs = [ObjectInfo(name='Quad', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=np.float32))]
    sc = Scene(mesh=mesh, materials=[mat], objects=objs, lights=lights,
               camera=cam, world=world, settings=st)
    sc.images = {}
    return sc


def _covered(st, sc):
    g, _job = _gbuffer_for(sc, st)
    return g.tri >= 0, g


def _settings(**kw):
    st = base_settings(W, H)
    st.shadows = False
    st.transparency = 'NONE'
    for k, v in kw.items():
        setattr(st, k, v)
    return st


# ---------------------------------------------------------------- tests

def test_a_identity_at_defaults():
    """The 1.89.0 zip is beside the package (loudly), and every default
    frame -- render AND post.process -- is bitwise the previous release's:
    four new models, three new sockets and one new flag change nothing
    until asked."""
    RP = _prev_engine(PREV_ZIP)
    check('the 1.89.0 zip is beside the package (the identity pins run)',
          RP is not None, 'halcyon-1.89.0.zip is missing: the pins below '
          'would silently skip')
    if RP is None:
        return
    prev_post = importlib.import_module(
        RP.__name__.rsplit('.', 1)[0] + '.post')
    for label, kw in (('defaults', {}), ('untextured, no shadows',
                                        {'shadows': False}),
                      ('force PHONG', {'force_model': 'PHONG'}),
                      ('Gouraud rate', {'shading_rate': 'VERTEX'})):
        st = base_settings(W, H, **kw)
        sc = demo_scene(st, with_texture=(label == 'defaults'))
        now = np.asarray(R.render(sc, st))
        st2 = base_settings(W, H, **kw)
        sc2 = demo_scene(st2, with_texture=(label == 'defaults'))
        prev = np.asarray(RP.render(sc2, st2))
        same = now.shape == prev.shape and np.array_equal(now, prev)
        check(f'the CPU frame at {label} renders bitwise the 1.89.0 zip',
              same, f'max {float(np.abs(now - prev).max()) if same or now.shape == prev.shape else "shape"}')
        pnow = np.asarray(PO.process(now, st, **_post_kw(sc, st)))
        pprev = np.asarray(prev_post.process(prev, st2, **_post_kw(sc2, st2)))
        check(f'...and post.process at {label} is bitwise the 1.89.0 zip',
              pnow.shape == pprev.shape and np.array_equal(pnow, pprev))
    # the new Surface fields sit at POV's own "off" values
    s = SH.Surface(4)
    check('a fresh Surface carries brilliance 1, crand 0, metallic 0 (the '
          'dials at their inert values)',
          np.all(s.brilliance == 1.0) and np.all(s.crand == 0.0)
          and np.all(s.pov_metallic == 0.0))


def test_gx_light_model():
    """F016: GX_LIGHT is item 32; its rational highlight peaks at exactly
    1 when N = H, is monotone in N.H and never exceeds 1; Sun lamps only
    add a highlight; the lit colour is an 8-bit integer per channel with
    white exactly 1.0; the simulator twin is d == 0.0 at pixel rate and
    the corner road carries the preset's Gouraud look."""
    names = [m[0] for m in SH.MODEL_ITEMS]
    check('the four console light units are items 32..35, appended (no '
          'saved model index moves)',
          names[32:36] == ['GX_LIGHT', 'SEGA_MODEL2', 'SEGA_MODEL3',
                           'DS_FIXED'], str(names[32:]))
    check('each of the four names its machine and its mechanism in a '
          'substantial description (> 60 chars, the era in the label)',
          all(len(SH.MODEL_ITEMS[k][2]) > 60 and '(' in SH.MODEL_ITEMS[k][1]
              for k in range(32, 36)))
    check('the four force the camera-axis viewer inside themselves '
          '(SH.AXIS_MODELS)',
          SH.AXIS_MODELS == frozenset({'GX_LIGHT', 'SEGA_MODEL2',
                                       'SEGA_MODEL3', 'DS_FIXED'}))

    # --- the pure function
    n = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (64, 1))
    th = np.linspace(0.0, 1.5, 64).astype(np.float32)
    l = np.stack([np.sin(th), np.zeros(64, np.float32), np.cos(th)],
                 1).astype(np.float32)
    v = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (64, 1))
    gloss = np.full(64, 25.0, np.float32)
    sp = SH.gx_spec(np.sum(n * l, axis=1), n, l, v, gloss)
    check('GX: the highlight is EXACTLY 1.0 where N = H (Glossiness 25: '
          '12.5 + (1 - 12.5) * 1 is exact in float32)', sp[0] == 1.0,
          f'{sp[0]!r}')
    check('GX: the highlight is monotone in N.H and never exceeds 1',
          np.all(np.diff(sp) <= 0.0) and float(sp.max()) <= 1.0)
    h = np.maximum(np.sum(n * (l + v) / np.linalg.norm(l + v, axis=1,
                                                       keepdims=True),
                          axis=1), 0.0)
    want = h * h / (12.5 + (1.0 - 12.5) * h * h)
    check('GX: the curve IS (N.H)^2 / (s/2 + (1 - s/2)(N.H)^2), the '
          'attenuation unit\'s ratio of quadratics (within 1e-6 of the '
          'float64 form)', float(np.abs(sp - want).max()) < 1e-6)

    # --- A/B on the demo scene
    gx = render(_settings(force_model='GX_LIGHT'))
    ph = render(_settings(force_model='PHONG'))
    la = render(_settings(force_model='LAMBERT'))
    check('GX_LIGHT differs from PHONG and from LAMBERT on the demo scene',
          float(np.abs(gx - ph).mean()) > 1e-3
          and float(np.abs(gx - la).mean()) > 1e-4)

    # --- the 8-bit vertex output over a white surface
    st = _settings(force_model='GX_LIGHT')
    sc = _facing_quad(st, 'GX_LIGHT', energy=9.0)
    img = np.asarray(R.render(sc, st))
    cov, _g = _covered(st, sc)
    lit = img[cov][:, :3]
    check('GX: over a white surface every lit value times 255 is an '
          'integer (the vertex unit\'s 8-bit output)',
          float(np.abs(lit * 255.0 - np.round(lit * 255.0)).max()) < 1e-4)
    check('GX: a saturated pixel is EXACTLY 1.0 (a division by 255.0, so '
          '255/255 is 1, not 0.99999994)',
          bool(np.any(lit == 1.0)) and float(lit.max()) == 1.0,
          f'max {lit.max()!r}')
    q = SH.quantize_lit(np.array([[1.0, 0.5, 0.0031]], np.float32), 255)
    check('quantize_lit: white stays exactly 1.0, half rounds half-up to '
          '128/255, the tie rule is floor(x + 0.5)',
          q[0, 0] == 1.0 and q[0, 1] == np.float32(128.0 / 255.0)
          and q[0, 2] == np.float32(1.0 / 255.0), str(q))

    # --- specular from directional lights only
    pt = Light(type='POINT', name='P', position=(0.0, -3.0, 4.0),
               color=(1.0, 1.0, 1.0), energy=120.0, shadow='NONE')
    st_p = _settings(force_model='GX_LIGHT')
    sc_p = _facing_quad(st_p, 'GX_LIGHT', lights=[pt])
    gx_p = np.asarray(R.render(sc_p, st_p))
    st_l = _settings(force_model='LAMBERT')
    sc_l = _facing_quad(st_l, 'LAMBERT', lights=[pt])
    la_p = np.asarray(R.render(sc_l, st_l))
    cov_p, _g = _covered(st_p, sc_p)
    want_p = SH.quantize_lit(la_p[cov_p][:, :3], 255)
    check('GX: a scene with only a POINT lamp has no highlight -- its frame '
          'is the LAMBERT frame after the same 8-bit step, bitwise '
          '(GX_AF_SPEC lights from directional lamps only)',
          np.array_equal(gx_p[cov_p][:, :3], want_p),
          f'max {float(np.abs(gx_p[cov_p][:, :3] - want_p).max())}')

    # --- the twins
    st = _settings(force_model='GX_LIGHT')
    cpu, sim, d, nbad, passes, atl, why = _twin(demo_scene(st, False), st)
    check('GX at pixel rate: the simulator twin (hal_gx_spec, the Sun '
          'gate, the axis texel, the 8-bit tail step) is bitwise the CPU '
          '(d == 0.0)', sim is not None and d == 0.0,
          f'd {d} bad {nbad} {why}')
    if atl is not None:
        g, job = _gbuffer_for(demo_scene(st, False), st)
        ft = atl['hal_fogtab'][1]()
        check('the axis texel (hal_fogtab 227) holds the CPU\'s own Vs '
              'bitwise', np.array_equal(ft[0, 227, :3],
                                        SH.axis_viewer(job.view)))
        check('every pass of the GX frame declares and binds hal_fogtab '
              '(the one rule: iff consts[fogtab])',
              all('hal_fogtab' in b.get('samplers', ())
                  and 'uniform sampler2D hal_fogtab;' in s
                  for _m, _n, s, b in passes))
    st_v = _settings(force_model='GX_LIGHT', shading_rate='VERTEX')
    cpu, sim, d, nbad, passes, atl, why = _twin(demo_scene(st_v, False),
                                                st_v)
    check('GX at the Gouraud rate (the GAMECUBE preset\'s): the corner road '
          'carries the 8-bit corner light within the corner bar (6e-3, no '
          'pixel above 1e-2)', sim is not None and d < 6e-3 and nbad == 0,
          f'd {d} bad {nbad} {why}')
    if passes is not None:
        check('...every pass interpolates CPU-lit corners (vlight), none '
              'lights in GLSL', all(bool(b.get('vlight'))
                                    for _m, _n, _s, b in passes))
    pt_st = _settings(force_model='PHONG')
    cpu, sim, d, nbad, passes, atl, why = _twin(demo_scene(pt_st, False),
                                                pt_st)
    check('a PHONG frame declares no hal_fogtab (the rule is off) and its '
          'twin holds the deferred bar', passes is not None
          and not any('hal_fogtab' in b.get('samplers', ())
                      for _m, _n, _s, b in passes) and d < 6e-3, f'{d}')


def test_sega_models():
    """F017: Model 2 squares its axis-reflection cosine 0-3 times and
    Model 3 raises N.L to 8/16/32/64 with the board's gains; both are
    their rates (FACE / VERTEX) on the corner road, a PIXEL-rate request
    refuses by name, and a fixed viewer keeps the highlight still."""
    c = np.array([0.5], np.float32)
    sp2 = SH.sega_model2_spec(c, np.array([5.0], np.float32))
    check('Model 2 at Glossiness 5 (k = 2, the cosine to the 4th) gives '
          'exactly 0.0625 at c = 0.5', sp2[0] == np.float32(0.0625),
          f'{sp2[0]!r}')
    c9 = np.float32(0.9)
    c2 = c9 * c9
    c4 = c2 * c2
    c8 = c4 * c4
    c16 = c8 * c8
    c32 = c16 * c16
    want3 = c32 * np.float32(2.4)
    sp3 = SH.sega_model3_spec(np.array([0.9], np.float32),
                              np.array([25.0], np.float32))
    check('Model 3 at Glossiness 25 (exponent 32, gain 2.4) at N.L 0.9 '
          'equals the explicit float32 squaring chain bitwise',
          sp3[0] == want3, f'{sp3[0]!r} vs {want3!r}')
    ramp = np.linspace(0.0, 1.0, 33).astype(np.float32)
    for g2 in (1.0, 2.0, 4.0, 8.0):
        s2 = SH.sega_model2_spec(ramp, np.full(33, g2, np.float32))
        check(f'Model 2 is monotone in its cosine at Glossiness {g2:g}',
              np.all(np.diff(s2) >= 0.0))
    for g3 in (8.0, 16.0, 32.0, 64.0):
        s3 = SH.sega_model3_spec(ramp, np.full(33, g3, np.float32))
        check(f'Model 3 is monotone in N.L at Glossiness {g3:g}',
              np.all(np.diff(s3) >= 0.0))
    check('the Sega models are their rates: SEGA_MODEL2 FACE, SEGA_MODEL3 '
          'VERTEX (render.RATE_FOR_MODEL) and refuse a pixel pass by name '
          '(gpu/shade.UNSUPPORTED_MODELS)',
          R.RATE_FOR_MODEL.get('SEGA_MODEL2') == 'FACE'
          and R.RATE_FOR_MODEL.get('SEGA_MODEL3') == 'VERTEX'
          and {'SEGA_MODEL2', 'SEGA_MODEL3'} <= set(GSH.UNSUPPORTED_MODELS))

    # --- A/B
    ph = render(_settings(force_model='PHONG'))
    for m in ('SEGA_MODEL2', 'SEGA_MODEL3'):
        im = render(_settings(force_model=m))
        check(f'{m} differs from PHONG on the demo scene',
              float(np.abs(im - ph).mean()) > 1e-3)

    # --- one colour per polygon (FACE) over white
    st = _settings(force_model='SEGA_MODEL2')
    sc = demo_scene(st, with_texture=False)
    for m in sc.materials:
        m.diffuse = (1.0, 1.0, 1.0)
    img = np.asarray(R.render(sc, st))
    cov, g = _covered(st, sc)
    spread = 0.0
    for t in np.unique(g.tri[cov])[:200]:
        px = img[g.tri == t][:, :3]
        spread = max(spread, float(px.max(axis=0).min() - px.min(axis=0).min()) if px.shape[0] else 0.0)
        spread = max(spread, float((px.max(axis=0) - px.min(axis=0)).max()))
    check('a force_model SEGA_MODEL2 frame shades at FACE rate: one colour '
          'per triangle over white, spread 0 bitwise', spread == 0.0,
          f'{spread}')

    # --- the fixed viewer: constant across a camera-facing quad, and
    # unchanged when the camera slides sideways (same axis)
    st_q = _settings(force_model='SEGA_MODEL2')
    sc_q = _facing_quad(st_q, 'SEGA_MODEL2', gloss=4.0)
    q0 = np.asarray(R.render(sc_q, st_q))
    cov_q, _g = _covered(st_q, sc_q)
    lit = q0[cov_q][:, :3]
    check("Model 2's highlight is constant across a camera-facing quad "
          '(every pixel: the same N, L and axis), spread 0.0 bitwise',
          float((lit.max(axis=0) - lit.min(axis=0)).max()) == 0.0)
    sc_s = _facing_quad(st_q, 'SEGA_MODEL2', gloss=4.0, cam_shift=0.8)
    q1 = np.asarray(R.render(sc_s, st_q))
    cov_s, _g = _covered(st_q, sc_s)
    check("...and unchanged when the camera moves sideways with the same "
          'axis (the spec probe bitwise)',
          cov_s.any() and np.array_equal(q0[cov_q][0, :3], q1[cov_s][0, :3]),
          f'{q0[cov_q][0, :3]} vs {q1[cov_s][0, :3] if cov_s.any() else None}')

    # --- the corner-road twins
    for m in ('SEGA_MODEL2', 'SEGA_MODEL3'):
        st = _settings(force_model=m)
        cpu, sim, d, nbad, passes, atl, why = _twin(demo_scene(st, False), st)
        check(f'{m}: the corner road is the machine on both devices -- the '
              'pass interpolates CPU-lit corners within the corner bar '
              '(6e-3, no pixel above 1e-2)',
              sim is not None and d < 6e-3 and nbad == 0
              and all(bool(b.get('vlight')) for _m, _n, _s, b in passes),
              f'd {d} bad {nbad} {why}')

    # --- the refusal by name with the rate pin removed
    saved = dict(R.RATE_FOR_MODEL)
    try:
        R.RATE_FOR_MODEL.pop('SEGA_MODEL2', None)
        st = _settings(force_model='SEGA_MODEL2')
        sc = demo_scene(st, with_texture=False)
        g, job = _gbuffer_for(sc, st)
        GSH._PLAN_CACHE.clear()
        passes, why, _a = GSH.plan_frame(job, g)
        check('with the rate pin removed a PIXEL-rate SEGA_MODEL2 refuses '
              'the GPU BY NAME (the model shades outside the light loop)',
              passes is None and 'SEGA_MODEL2' in str(why)
              and 'outside the light loop' in str(why), str(why))
    finally:
        R.RATE_FOR_MODEL.clear()
        R.RATE_FOR_MODEL.update(saved)
        GSH._PLAN_CACHE.clear()


def test_ds_model():
    """F018: DS_FIXED's highlight is the unnormalised half vector against
    the fixed line of sight, squared, through a 128-entry 8-bit table
    (Glossiness-shaped), the lit colour 5 bits per channel; the twin is
    bitwise in the simulator and the hal_dstab sampler is declared and
    bound ONLY by a DS pass, through the fake device."""
    # --- the pure function
    T = SH.ds_shininess_table(25.0)
    check('the shininess table has 128 integer entries 0..255, monotone, '
          'with (i/128)^(Glossiness/8) as its curve',
          T.shape == (128,) and np.all(T == np.round(T))
          and np.all(np.diff(T) >= 0) and 0 <= T.min() and T.max() <= 255)
    check('Glossiness 8 is a linear table (T[i] = round(255 (i+0.5)/128))',
          np.array_equal(SH.ds_shininess_table(8.0),
                         np.array([int(round(255.0 * ((i + 0.5) / 128.0)))
                                   for i in range(128)], np.float32)))
    n = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (64, 1))
    th = np.linspace(0.0, 1.5, 64).astype(np.float32)
    l = np.stack([np.sin(th), np.zeros(64, np.float32), np.cos(th)],
                 1).astype(np.float32)
    v = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (64, 1))
    sp = SH.ds_spec(n, l, v, np.full(64, 25.0, np.float32))
    check('DS: sp * 256 is an integer for every sample (an 8-bit entry as '
          'a 0.8 fixed value)',
          float(np.abs(sp * 256.0 - np.round(sp * 256.0)).max()) < 1e-5)
    check('DS: sp is non-decreasing in the half-vector strength',
          np.all(np.diff(sp) <= 0.0))
    check('DS: at the peak (L = V = N, |H| = 1) sp equals '
          'T[min(int(1 * 128), 127)] / 256 -- the unnormalised half '
          'vector\'s own length caps the index',
          sp[0] == np.float32(T[127] * 0.00390625), f'{sp[0]!r}')

    # --- A/B, the constant highlight, the 5-bit output
    ph = render(_settings(force_model='PHONG'))
    ds = render(_settings(force_model='DS_FIXED'))
    check('DS_FIXED differs from PHONG on the demo scene',
          float(np.abs(ds - ph).mean()) > 1e-3)
    st_q = _settings(force_model='DS_FIXED')
    sc_q = _facing_quad(st_q, 'DS_FIXED', gloss=25.0, energy=9.0)
    q = np.asarray(R.render(sc_q, st_q))
    cov, _g = _covered(st_q, sc_q)
    lit = q[cov][:, :3]
    check("DS: the highlight is constant across a camera-facing quad "
          '(spread 0.0 bitwise)',
          float((lit.max(axis=0) - lit.min(axis=0)).max()) == 0.0)
    check('DS: over white every lit value times 31 is an integer and a '
          'saturated pixel is EXACTLY 1.0 (the light unit\'s 5-bit '
          'saturation, a division by 31.0)',
          float(np.abs(lit * 31.0 - np.round(lit * 31.0)).max()) < 1e-4
          and float(lit.max()) == 1.0, f'max {lit.max()!r}')

    # --- the simulator twin, the sampler rule
    st = _settings(force_model='DS_FIXED', max_lights=4)
    cpu, sim, d, nbad, passes, atl, why = _twin(demo_scene(st, False), st)
    check('DS at pixel rate: the simulator twin (the owned DS lines, the '
          'hal_dstab row, the 5-bit tail step) is bitwise the CPU (d == '
          '0.0)', sim is not None and d == 0.0, f'd {d} bad {nbad} {why}')
    if passes is not None:
        check('every DS pass declares hal_dstab and records its (material, '
              'Glossiness) row; the plan packs one table per row',
              all('uniform sampler2D hal_dstab;' in s
                  and b.get('dstab') is not None for _m, _n, s, b in passes)
              and 'hal_dstab' in atl
              and atl['hal_dstab'][1]().shape[1:] == (128, 4))
    st_v = _settings(force_model='DS_FIXED', max_lights=4,
                     shading_rate='VERTEX')
    cpu, sim, d, nbad, passes, atl, why = _twin(demo_scene(st_v, False),
                                                st_v)
    check('DS at the Gouraud rate (the NDS preset\'s): the corner road '
          'within the corner bar', sim is not None and d < 6e-3
          and nbad == 0, f'd {d} bad {nbad} {why}')

    # --- a mixed frame: the DS material's pass binds hal_dstab, the PHONG
    # pass in the same frame declares it NOT (B12); the driver road
    # through the fake device equals the simulator
    from . import fakedevice
    st_m = _settings()
    sc_m = demo_scene(st_m, with_texture=False)
    sc_m.materials[1].use_override = True
    sc_m.materials[1].model = 'DS_FIXED'
    cpu_m, sim_m, d_m, nbad_m, passes_m, atl_m, why_m = _twin(sc_m, st_m)
    check('a mixed frame (one DS material beside PHONG and LAMBERT) '
          'simulates within the deferred bar', sim_m is not None
          and d_m < 6e-3 and nbad_m == 0, f'd {d_m} {why_m}')
    if passes_m is not None:
        decl = {mi: ('uniform sampler2D hal_dstab;' in s,
                     'hal_dstab' in b.get('samplers', ()))
                for mi, _n, s, b in passes_m}
        check('the DS pass declares AND binds hal_dstab; the PHONG and '
              'LAMBERT passes in the same frame declare it not',
              decl.get(1) == (True, True)
              and decl.get(0) == (False, False)
              and decl.get(2) == (False, False), str(decl))
    st_g = _settings(render_device='GPU')
    sc_g = demo_scene(st_g, with_texture=False)
    sc_g.materials[1].use_override = True
    sc_g.materials[1].model = 'DS_FIXED'
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        gpu_m = np.asarray(R.render(sc_g, st_g))
        specs = [sh.spec for sh in dev.shaders.values()]
    ds_specs = [sp for sp in specs if 'hal_dstab' in sp.get('samplers', [])]
    check('through the fake device the DS pass compiles with hal_dstab in '
          'its CreateInfo spec (declared == bound, the driver\'s rule) and '
          'the frame matches the CPU within 6e-6',
          len(ds_specs) >= 1 and gpu_m.shape == cpu_m.shape
          and float(np.abs(gpu_m[..., :3] - cpu_m[..., :3]).max()) <= 6e-6,
          f'{len(ds_specs)} DS specs; max '
          f'{float(np.abs(gpu_m[..., :3] - cpu_m[..., :3]).max()) if gpu_m.shape == cpu_m.shape else "shape"}')


# ------------------------------------------------ the POV-Ray finish

def _pov_scene(st, frame=None, **sockets):
    from .featurematrix import _sc_pov_finish
    return _sc_pov_finish(st, frame=frame, **sockets)


def _driven_socket_scene(st, socket):
    """The POV finish scene with `socket` LINKED from a Noise chain -- a
    per-pixel value the material texels cannot carry."""
    sc = _pov_scene(st)
    g = sc.materials[1].graph
    g['nodes']['noise'] = {
        'id': 'noise', 'bl_idname': 'HALCYON_NoiseNode',
        'props': {'kind': 'RIDGED', 'octaves': 3},
        'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                   _sk('Scale', 'VALUE', 4.0),
                   _sk('Lacunarity', 'VALUE', 2.0),
                   _sk('Gain', 'VALUE', 0.5),
                   _sk('Color 1', 'RGBA', [0.1, 0.05, 0.3, 1.0]),
                   _sk('Color 2', 'RGBA', [1.0, 0.9, 0.6, 1.0])],
        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                    {'name': 'Fac', 'type': 'VALUE'}]}
    g['nodes']['bsdf']['inputs'].append(_sk(socket, 'VALUE', 1.0,
                                            ['noise', 1]))
    return sc


def _plan(sc, st):
    g, job = _gbuffer_for(sc, st)
    GSH._PLAN_CACHE.clear()
    return GSH.plan_frame(job, g)


def _sim_probe(src, uniforms, n):
    from ..shaders.compiler import try_compile
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, err
    return prog.run(uniforms, {}, n)[0]['Color'], None


def _prev_pin(label, build_scene, **kw):
    """One bitwise pin against the 1.89.0 engine for a scene builder."""
    RP = _prev_engine(PREV_ZIP)
    if RP is None:
        return
    st = _settings(**kw)
    now = np.asarray(R.render(build_scene(st), st))
    st2 = _settings(**kw)
    prev = np.asarray(RP.render(build_scene(st2), st2))
    check(f'{label}: the master-shader frame renders bitwise the 1.89.0 zip '
          '(the socket at its inert value changes nothing)',
          now.shape == prev.shape and np.array_equal(now, prev),
          f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')


def test_pov_brilliance():
    """F019: Brilliance raises the diffuse cosine to a power, skipped at
    1.0 exactly; above 1 every lit pixel darkens or stays; it is inert on
    the 3ds Max shaders; the pow is bitwise the CPU in the simulator, the
    frame within the deferred bar; a driven socket refuses by name."""
    _prev_pin('Brilliance 1.0', lambda st: _pov_scene(st, **{'Brilliance': 1.0}))
    st = _settings()
    plain = np.asarray(R.render(_pov_scene(st), st))
    means = []
    for b in (1.0, 2.0, 4.0):
        img = np.asarray(R.render(_pov_scene(st, **{'Brilliance': b}), st))
        cov, _g = _covered(st, _pov_scene(st))
        means.append(float(img[cov][:, :3].mean()))
        if b == 2.0:
            d = plain[..., :3] - img[..., :3]
            check('Brilliance 2 darkens every lit pixel or leaves it (frame '
                  '<= plain everywhere) and moves the picture',
                  float(d.min()) >= -1e-6 and float(d.max()) > 1e-2,
                  f'min {d.min()} max {d.max()}')
            check('...and is bitwise the plain frame where the cosine was 0 '
                  'or 1 (some pixels are exactly equal)',
                  bool(np.any(np.all(plain[..., :3] == img[..., :3],
                                     axis=-1) & cov)))
    check('the lit mean decreases monotonically in Brilliance over {1, 2, 4}',
          means[0] > means[1] > means[2], str(means))
    check('Brilliance 1 is exactly the plain frame (the pow is skipped, as '
          'POV skips it)', np.array_equal(
              plain, np.asarray(R.render(_pov_scene(st, **{'Brilliance': 1.0}), st))))
    # the pure function
    d0 = np.linspace(-0.2, 1.0, 64).astype(np.float32)
    b2 = np.full(64, 2.0, np.float32)
    ab = SH.apply_brilliance(d0, b2)
    check('apply_brilliance IS pow(max(cos, 0), B): float32, one op',
          np.array_equal(ab, np.power(np.maximum(d0, 0.0), b2).astype(np.float32)))
    check('...and returns the input object untouched at B == 1 everywhere',
          SH.apply_brilliance(d0, np.ones(64, np.float32)) is d0)
    # inert on the Max shaders, both roads
    st_m = _settings(force_model='MAX_OREN_NAYAR_BLINN')
    a = np.asarray(R.render(_pov_scene(st_m, **{'Brilliance': 2.0}), st_m))
    b = np.asarray(R.render(_pov_scene(st_m, **{'Brilliance': 1.0}), st_m))
    check('a MAX_OREN_NAYAR_BLINN material with Brilliance 2 renders '
          'bitwise the Brilliance-1 frame (inert on the Max shaders)',
          np.array_equal(a, b))
    p, why, _a = _plan(_pov_scene(st_m, **{'Brilliance': 2.0}), st_m)
    check('...and its GPU pass emits no brilliance pow (inert there too)',
          p is not None and not any('pow(max(ds.x, 0.0), s.brilliance)' in s
                                    for _m, _n, s, b_ in p), str(why))
    # the twins
    cpu, sim, d, nbad, passes, atl, why = _twin(
        _pov_scene(st, **{'Brilliance': 2.0}), st)
    check('Brilliance 2: the simulator twin holds the deferred bar (the '
          'master-shader road\'s own 6e-6) and the pass carries the pow on '
          'the diffuse scalar', sim is not None and d < 6e-3 and nbad == 0
          and any('pow(max(ds.x, 0.0), s.brilliance)' in s
                  for _m, _n, s, b_ in passes), f'd {d} {why}')
    got, err = _sim_probe(
        'uniform float x; uniform float b; out vec4 Color;\n'
        'void main() { float y = (b == 1.0) ? x : pow(max(x, 0.0), b);\n'
        '    Color = vec4(y, 0.0, 0.0, 1.0); }',
        {'x': d0, 'b': b2}, 64)
    check('the brilliance statement through the GLSL simulator is bitwise '
          'apply_brilliance (d == 0.0)', got is not None
          and np.array_equal(got[:, 0], ab), str(err))
    # the refusal
    st_r = _settings()
    p, why, _a = _plan(_driven_socket_scene(st_r, 'Brilliance'), st_r)
    check('a Brilliance socket driven by a Noise chain refuses the GPU BY '
          'NAME (brilliance varies across the frame)',
          p is None and 'brilliance varies' in str(why), str(why))


def test_pov_crand():
    """F020: Crand grains the direct diffuse per lamp and per pixel from
    the integer hash -- deterministic, never on the specular, re-rolled
    per frame only under Crand Flickers per Frame; the hash is bitwise
    the CPU's through the simulator; a no-soft-shadow crand material
    compiles under the fake device with hal_circle bound; a huge seed
    plans; a driven socket refuses by name."""
    _prev_pin('Crand 0', lambda st: _pov_scene(st, **{'Crand': 0.0}))
    st = _settings()
    plain = np.asarray(R.render(_pov_scene(st), st))
    g1 = np.asarray(R.render(_pov_scene(st, **{'Crand': 0.3}), st))
    g2 = np.asarray(R.render(_pov_scene(st, **{'Crand': 0.3}), st))
    check('crand is deterministic: two renders are bitwise equal',
          np.array_equal(g1, g2))
    d = plain[..., :3] - g1[..., :3]
    check('the crand frame is <= the plain frame everywhere and moves the '
          'picture', float(d.min()) >= -1e-6 and float(d.max()) > 1e-2,
          f'min {d.min()} max {d.max()}')
    cov, _g = _covered(st, _pov_scene(st))
    check('...and is bitwise the plain frame at some lit-side-free pixels '
          '(the back side and the shadowed side grain nothing)',
          bool(np.any(np.all(plain[..., :3] == g1[..., :3], axis=-1) & cov)))
    # specular only: diffuse black, specular white -> no grain at all
    s0 = np.asarray(R.render(_pov_scene(st, diffuse=(0.0, 0.0, 0.0)), st))
    s1 = np.asarray(R.render(_pov_scene(st, diffuse=(0.0, 0.0, 0.0),
                                        **{'Crand': 0.3}), st))
    check('a material with a black diffuse and a white specular grains '
          'nothing: the crand frame is bitwise the plain one (highlights '
          'untouched)', np.array_equal(s0, s1))
    # per frame
    f1 = np.asarray(R.render(_pov_scene(st, frame=1, **{'Crand': 0.3}), st))
    f2 = np.asarray(R.render(_pov_scene(st, frame=2, **{'Crand': 0.3}), st))
    check('without Crand Flickers per Frame, frames 1 and 2 are bitwise '
          'equal', np.array_equal(f1, f2))
    st_f = _settings(crand_per_frame=True)
    f1 = np.asarray(R.render(_pov_scene(st_f, frame=1, **{'Crand': 0.3}), st_f))
    f2 = np.asarray(R.render(_pov_scene(st_f, frame=2, **{'Crand': 0.3}), st_f))
    check('with Crand Flickers per Frame, frames 1 and 2 differ (POV\'s '
          'flicker on purpose)', not np.array_equal(f1, f2))
    # the twins
    cpu, sim, d, nbad, passes, atl, why = _twin(
        _pov_scene(st, **{'Crand': 0.3}), st)
    check('Crand 0.3: the simulator twin holds the deferred bar and the '
          'pass reads hal_smp_hash3 with the CPU\'s salt (977 + 131 lamp + '
          '7919 seed, wrapped to 31 bits)', sim is not None and d < 6e-3
          and nbad == 0 and any('int hal_cz = 977;' in s
                                for _m, _n, s, b_ in passes), f'd {d} {why}')
    if passes is not None:
        check('a crand material with NO soft shadow, AO or radiosity in the '
              'frame appends the sampling primitives and binds hal_circle '
              '(B13)', any('hal_circle' in b_.get('samplers', ())
                           and 'float hal_smp_hash3(' in s
                           for _m, _n, s, b_ in passes)
              and 'hal_circle' in atl)
    cpu, sim, d, nbad, passes, atl, why = _twin(
        _pov_scene(st_f, frame=3, **{'Crand': 0.3}), st_f)
    check('Crand per frame: the pass declares hal_frame and folds it into '
          'the salt; the twin holds the bar', sim is not None and d < 6e-3
          and nbad == 0 and any('hal_frame' in b_.get('frame_uniforms', ())
                                and 'int(hal_frame) * 1013' in s
                                for _m, _n, s, b_ in passes), f'd {d} {why}')
    # the hash's own bitwise pin through the simulator
    from ..core import patterns as PT
    from ..gpu.material import SAMPLING_GLSL
    from ..core.texture import Texture
    n = 4096
    px = np.tile(np.arange(64, dtype=np.int32), 64)
    py = np.repeat(np.arange(64, dtype=np.int32), 64)
    z = int((977 + 131 * 1 + 7919 * 7) & 0x7fffffff)
    src = SAMPLING_GLSL + (
        'uniform int px; uniform int py; out vec4 Color;\n'
        f'void main() {{ float u = hal_smp_hash3(px, py, {z});\n'
        '    Color = vec4(u, 0.0, 0.0, 1.0); }')
    circ = np.zeros((1, 256, 4), np.float32)
    got, err = _sim_probe(src, {'px': px, 'py': py,
                                'hal_circle': Texture(circ, colorspace='Non-Color',
                                                      filt='NEAREST', wrap='EXTEND')}, n)
    want = PT.sample_u(px, py, z)
    check('hal_smp_hash3 through the simulator is bitwise PT.sample_u over '
          '4096 pixels (d == 0.0)', got is not None
          and np.array_equal(got[:, 0], want.astype(np.float32)), str(err))
    # a huge seed: the literal stays in range (wrapped) and the sim agrees
    st_s = _settings(seed=2 ** 40)
    cpu, sim, d, nbad, passes, atl, why = _twin(
        _pov_scene(st_s, **{'Crand': 0.3}), st_s)
    zz = (977 + 7919 * 2 ** 40) & 0x7fffffff
    check('st.seed = 2**40 plans (the salt literal is wrapped to 31 bits) '
          'and the simulator equals the CPU within the bar',
          sim is not None and d < 6e-3 and nbad == 0
          and any(f'int hal_cz = {zz};' in s for _m, _n, s, b_ in passes),
          f'd {d} {why}')
    # the fake device: the driver road binds hal_circle for the crand pass
    from . import fakedevice
    st_g = _settings(render_device='GPU')
    sc_g = _pov_scene(st_g, **{'Crand': 0.3})
    cpu_g = np.asarray(R.render(_pov_scene(_settings(), **{'Crand': 0.3}),
                                _settings()))
    cpu_s, sim_s, d_s, nbad_s, _p, _a, _w = _twin(
        _pov_scene(_settings(), **{'Crand': 0.3}), _settings())
    cov_s, _g = _covered(_settings(), _pov_scene(_settings()))
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        gpu = np.asarray(R.render(sc_g, st_g))
        specs = [sh.spec for sh in dev.shaders.values()]
    check('through the fake device the crand pass compiles with hal_circle '
          'in its CreateInfo spec, draws bitwise the simulator over the '
          'covered pixels and lands within the material road\'s 1e-5 of '
          'the CPU', any('hal_circle' in sp.get('samplers', []) for sp in specs)
          and gpu.shape == cpu_g.shape and sim_s is not None
          and np.array_equal(gpu[..., :3][cov_s], sim_s[cov_s])
          and float(np.abs(gpu[..., :3] - cpu_g[..., :3]).max()) <= 1e-5,
          f'max {float(np.abs(gpu[..., :3] - cpu_g[..., :3]).max()) if gpu.shape == cpu_g.shape else "shape"}')
    # the refusal
    st_r = _settings()
    p, why, _a = _plan(_driven_socket_scene(st_r, 'Crand'), st_r)
    check('a Crand socket driven by a Noise chain refuses the GPU BY NAME '
          '(crand varies across the frame)',
          p is None and 'crand varies' in str(why), str(why))
    # the setting is drawn and documented
    from . import fakebpy
    fakebpy.install()
    from ..properties import DESCRIPTIONS, LABELS
    check('Crand Flickers per Frame carries its label and a tooltip naming '
          'the mechanism (>= 40 chars)',
          LABELS.get('crand_per_frame') == 'Crand Flickers per Frame'
          and len(DESCRIPTIONS.get('crand_per_frame', '')) >= 40
          and 'frame' in DESCRIPTIONS.get('crand_per_frame', ''))


def test_pov_metallic():
    """F021: Metallic (POV) tints the highlight by POV's rational Fresnel
    of N.L -- pigment-coloured facing the light, the light's colour at
    grazing, F monotone; inert on ANIME; the emitted literals are the
    float32 of the CPU's constants; the curve is bitwise the CPU through
    the simulator; the frame holds the deferred bar; a driven socket
    refuses by name."""
    _prev_pin('Metallic (POV) 0',
              lambda st: _pov_scene(st, **{'Metallic (POV)': 0.0}))
    # the pure function
    ramp = np.linspace(1.0, 0.0, 256).astype(np.float32)      # x: 0 -> 1
    F = SH.pov_metallic_fresnel(ramp)
    check('F is monotone non-decreasing in x = acos(N.L)/(pi/2) on a '
          '256-point ramp and stays in 0..1',
          np.all(np.diff(F) >= 0.0) and F.min() >= 0.0 and F.max() <= 1.0)
    check('F at normal incidence is ~0 (0.014567225/1.2544 - 0.011612903 = '
          '5e-7) and at grazing ~1', float(F[0]) < 1e-5
          and float(F[-1]) > 0.99, f'{F[0]} {F[-1]}')
    spec = np.ones((256, 3), np.float32)
    pig = np.tile(np.array([[1.0, 0.2, 0.1]], np.float32), (256, 1))
    tinted = SH.apply_pov_metallic(spec, ramp, np.ones(256, np.float32), pig)
    check('facing the light (N.L = 1) a white highlight takes the pigment '
          '(G/B within 2e-2 of 0.2 / 0.1)',
          abs(float(tinted[0, 1]) - 0.2) < 2e-2
          and abs(float(tinted[0, 2]) - 0.1) < 2e-2, str(tinted[0]))
    # (the spec's "N.L < 0.05 -> tint within 1e-2 of 1" is wrong by its
    # own numbers: F(x = 0.97) = 0.014567225/0.0225 - 0.0116 = 0.64; F
    # reaches 1 only at x = 1, N.L = 0 -- the law is stated there)
    check('at grazing (N.L = 0, x = 1) the tint is within 1e-2 of 1 (the '
          'light\'s own colour: F = 0.014567225/0.0144 - 0.011612903 -> 1)',
          float(np.abs(tinted[-1] - 1.0).max()) < 1e-2, str(tinted[-1]))
    check('...and the tint is monotone from the pigment to white as N.L '
          'falls (G channel non-decreasing along the ramp)',
          np.all(np.diff(tinted[:, 1]) >= -1e-7))
    # the frame: A/B and the ANIME law
    st = _settings()
    plain = np.asarray(R.render(_pov_scene(st, diffuse=(1.0, 0.2, 0.1)), st))
    met = np.asarray(R.render(_pov_scene(st, diffuse=(1.0, 0.2, 0.1),
                                         **{'Metallic (POV)': 1.0}), st))
    check('Metallic (POV) 1 moves the picture on a coloured pigment',
          float(np.abs(met - plain).max()) > 1e-2)
    st_a = _settings(force_model='ANIME')
    a0 = np.asarray(R.render(_pov_scene(st_a, diffuse=(1.0, 0.2, 0.1)), st_a))
    a1 = np.asarray(R.render(_pov_scene(st_a, diffuse=(1.0, 0.2, 0.1),
                                        **{'Metallic (POV)': 1.0}), st_a))
    check('an ANIME material with the socket at 1.0 renders bitwise the '
          'socket-0 frame (its highlight is painted in the loop: inert)',
          np.array_equal(a0, a1))
    # the twins and the literals
    from ..gpu.material import _f
    cpu, sim, d, nbad, passes, atl, why = _twin(
        _pov_scene(st, diffuse=(1.0, 0.2, 0.1), **{'Metallic (POV)': 1.0}),
        st)
    check('Metallic (POV): the simulator twin holds the deferred bar',
          sim is not None and d < 6e-3 and nbad == 0, f'd {d} {why}')
    if passes is not None:
        src = next(s for m, _n, s, b_ in passes if m == 1)
        blk = src[src.index('float hal_hx'):src.index('ds.yzw = ds.yzw * hal_ht')]
        lits = (_f(np.float32(2.0 / np.pi)), _f(np.float32(0.014567225)),
                _f(np.float32(0.011612903)), _f(np.float32(1.12)))
        check('the emitted metallic block carries the float32 literals of '
              'the CPU\'s constants (_f: 0.636619747, 0.0145672252, '
              '0.0116129033) and none of the hand-copied float64 digits',
              all(l in blk for l in lits) and '0.636619772' not in blk
              and '0.014567225 ' not in blk and '0.011612903 ' not in blk,
              str(lits))
    n = 256
    src = (
        'uniform float ndl; out vec4 Color;\n'
        'void main() {\n'
        '    float hal_hx = clamp(ndl, 0.0, 1.0);\n'
        '    hal_hx = acos(hal_hx);\n'
        f'    hal_hx = hal_hx * {_f(np.float32(2.0 / np.pi))};\n'
        f'    float hal_hxm = hal_hx - {_f(np.float32(1.12))};\n'
        '    hal_hxm = hal_hxm * hal_hxm;\n'
        f'    float hal_hF = {_f(np.float32(0.014567225))} / hal_hxm;\n'
        f'    hal_hF = hal_hF - {_f(np.float32(0.011612903))};\n'
        '    hal_hF = clamp(hal_hF, 0.0, 1.0);\n'
        '    Color = vec4(hal_hF, 0.0, 0.0, 1.0); }')
    got, err = _sim_probe(src, {'ndl': ramp}, n)
    check('the POV Fresnel through the GLSL simulator is bitwise '
          'pov_metallic_fresnel (d == 0.0)', got is not None
          and np.array_equal(got[:, 0], F), str(err))
    # the refusal
    st_r = _settings()
    p, why, _a = _plan(_driven_socket_scene(st_r, 'Metallic (POV)'), st_r)
    check('a Metallic (POV) socket driven by a Noise chain refuses the GPU '
          'BY NAME (pov_metallic varies across the frame)',
          p is None and 'pov_metallic varies' in str(why), str(why))
    # the sockets are documented on the node (the node module needs bpy)
    from . import fakebpy
    fakebpy.install()
    from ..nodes import shader_nodes as SN
    names = [s[1] for s in SN.HALCYON_ShaderNode.SOCKETS]
    check('the three POV finish sockets exist on the master shader with '
          'docs >= 40 chars naming POV-Ray',
          all(nm in names and len(SN.SOCKET_DOCS.get(nm, '')) >= 40
              and 'POV' in SN.SOCKET_DOCS.get(nm, '')
              for nm in ('Brilliance', 'Crand', 'Metallic (POV)')))


def main():
    from . import utf8_console
    utf8_console()
    order = [v for k, v in sorted(globals().items())
             if k.startswith('test_')]
    for fn in order:
        print(fn.__name__)
        try:
            fn()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(fn.__name__)
    print()
    print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS) if FAILS
          else 'all R251 lighting-model tests passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
