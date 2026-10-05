"""R251 lighting pack, slot LIGHT-B1: the LAMP roads.

    F011  fixed camera-axis viewer for highlights (GL 1.1 / Sega Model / DS)
    F014  Only Shadow lamp (Blender Internal LA_ONLYSHADOW)
    F012  spot cone laws (OpenGL 1.1 exponent, POV-Ray hotspot, GX angular)
    F013  decay laws (OpenGL 3-term, POV-Ray fade, GX distance)
    F015  screen-space spotlight ellipse (Sega Model 3)

Run alone:  python -m halcyon.tests.test_r251_lighting_lamps

The first test pins the 1.89.0 zip loudly and the identity at every default
bitwise (render AND post.process); every feature then proves its semantic
A/B law, its GPU twin (the simulator against the CPU: `d == 0.0` for the
standalone laws, the deferred bar on a frame) and its refusal by name.
"""
import importlib
import sys
import traceback

import numpy as np

from ..core import lights as LI
from ..core import mathx as M
from ..core import post as PO
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import Light
from ..gpu import shade as GSH
from . import featurematrix as FM
from .scenebuild import demo_scene
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name
          + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


PREV_ZIP = 'halcyon-1.89.0.zip'
W, H = 96, 72


# ------------------------------------------------------------------ helpers


def _settings(**kw):
    st = base_settings(W, H, shadows=True, transparency='NONE')
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _scene(st, key='demo'):
    if key == 'demo':
        return demo_scene(st, with_texture=False)
    return FM.SCENES[key](st)


def _render(key='demo', **kw):
    st = _settings(**kw)
    sc = _scene(st, key)
    return np.asarray(R.render(sc, st))


def _twin(sc, st):
    """(cpu, sim, covered-mask, why): the deferred pass through the GLSL
    simulator against the CPU frame -- TR's test_gouraud rig."""
    cpu = np.asarray(R.render(sc, st))
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, None, view, eye, W, H)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    if p is None:
        return cpu, None, None, str(why)
    out, _hit = GSH.simulate(job, g)
    if out is None:
        return cpu, None, None, str(_hit)
    return cpu, out, g.tri >= 0, None


def _frame_bar(label, cpu, sim, mask, why, bar=6e-3):
    """The deferred road's own bar (TR:20113-20118)."""
    check(f'{label}: the deferred pass plans and simulates', sim is not None,
          str(why))
    if sim is None:
        return
    yy, xx = np.nonzero(mask)
    d = np.abs(sim[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
    check(f'{label}: the simulator matches the CPU within the deferred bar',
          int((d > 1e-2).sum()) == 0 and float(d.max()) < bar,
          f'{int((d > 1e-2).sum())} of {yy.size} px >0.01, '
          f'max {float(d.max()):.6f}')


def _run_glsl(src, n, **uni):
    """A standalone fragment through Halcyon's own GLSL front-end: every
    uniform is a per-lane array (the TR:4085 stage-harness shape)."""
    from ..shaders.compiler import try_compile
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, err
    u = {}
    for k, v in uni.items():
        a = np.asarray(v, np.float32)
        u[k] = (np.full(n, float(a), np.float32) if a.ndim == 0
                else np.ascontiguousarray(a))
    outs, _d = prog.run(u, {}, n)
    return np.asarray(outs['Color'], np.float32).reshape(n, 4), None


# ------------------------------------------------------- 1. the identity pin


def test_identity_at_defaults():
    """Every new dial of this slot is invisible at its default: the demo
    scene (and the lamp scenes the features touch) render bitwise the
    1.89.0 zip's frame, render AND post.process."""
    RP = _prev_engine(PREV_ZIP)
    check('the 1.89.0 zip is beside the package', RP is not None,
          'halcyon-1.89.0.zip must sit beside halcyon/ for this pin')
    if RP is None:
        return
    prev_post = importlib.import_module(
        RP.__name__.rsplit('.', 1)[0] + '.post')
    cases = [('the demo scene at defaults', 'demo', {}),
             ('the demo scene under PHONG (F011 identity)', 'demo',
              {'default_model': 'PHONG'}),
             ('the spot-gobo scene (F012 identity: BLENDER law, exponent 0)',
              'cookie_spot', {}),
             ('the negative-lamp scene (F013/F014 identity)', 'negative', {}),
             ('the demo scene under an INVERSE_SQUARE default (F013)',
              'demo', {'light_falloff_default': 'INVERSE_SQUARE'}),
             ('the demo scene fogged LINEAR (F015 identity: fog_spot 0)',
              'demo', {'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0,
                       'fog_end': 12.0})]
    for label, key, kw in cases:
        st = _settings(**kw)
        sc = _scene(st, key)
        now = np.asarray(R.render(sc, st))
        st2 = _settings(**kw)
        sc2 = _scene(st2, key)
        prev = np.asarray(RP.render(sc2, st2))
        same = now.shape == prev.shape and bool(np.array_equal(now, prev))
        check(f'{label} renders bitwise the 1.89.0 zip', same,
              f'max {float(np.abs(now - prev).max()):.3g}' if same is False
              and now.shape == prev.shape else '')
    st = _settings()
    sc = _scene(st)
    now = np.asarray(PO.process(R.render(sc, st), st))
    st2 = _settings()
    sc2 = _scene(st2)
    prev = np.asarray(prev_post.process(RP.render(sc2, st2), st2))
    check('the demo scene: render + post.process bitwise the 1.89.0 zip',
          now.shape == prev.shape and bool(np.array_equal(now, prev)))


# ------------------------------------------------------------------ F011


def test_axis_viewer():
    """F011: one camera axis as the viewer of every reflectance model."""
    # A/B on the demo scene under PHONG
    pix = _render(default_model='PHONG', specular_viewer='PIXEL')
    axis = _render(default_model='PHONG', specular_viewer='AXIS')
    check('specular_viewer AXIS changes the PHONG demo frame',
          float(np.abs(axis - pix).max()) > 1e-3,
          f'{float(np.abs(axis - pix).max()):.4f}')
    # the law on a camera-facing quad: every pixel has the same N, L and
    # viewer under AXIS, so the lit colour is one value across the quad
    sc, st = FM.build('fixed camera-axis viewer (GL 1.1 / Sega Model / DS)')
    on = np.asarray(R.render(sc, st))
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    cov = g.tri >= 0
    lit = on[cov][:, :3]
    spread_on = float((lit.max(axis=0) - lit.min(axis=0)).max())
    check('under AXIS the facing quad shades to ONE colour (spread 0.0 '
          'bitwise: same N, L and viewer at every pixel)',
          lit.shape[0] > 100 and spread_on == 0.0,
          f'{lit.shape[0]} px, spread {spread_on:.3g}')
    st.specular_viewer = 'PIXEL'
    sc2 = FM.SCENES['facing_quad'](st)
    off = np.asarray(R.render(sc2, st))
    lit2 = off[cov][:, :3]
    spread_off = float((lit2.max(axis=0) - lit2.min(axis=0)).max())
    check('under PIXEL the highlight slides across the same quad '
          '(spread above 1e-3)', spread_off > 1e-3, f'{spread_off:.4f}')
    # the axis the CPU took equals the texel the GPU reads
    st.specular_viewer = 'AXIS'
    vs_cpu = M.normalize(np.asarray(view, np.float32)[2:3, :3])[0]
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     W, H)
    GSH._PLAN_CACHE.clear()
    p, why, atl = GSH.plan_frame(job, g)
    check('an AXIS frame plans on the GPU with no refusal', p is not None,
          str(why))
    if p is not None:
        ft = (atl or {}).get('hal_fogtab')
        arr = ft[1]() if ft else None
        check('every AXIS pass reads the axis from hal_fogtab texel 227 '
              '(declared and bound)',
              arr is not None and all('hal_fogtab' in (b.get('samplers')
                                                       or ())
                                      for _mi, _n, s, b in p
                                      if 'hal_fogtab' in s))
        if arr is not None:
            vs_gpu = np.asarray(arr[0, 227, :3], np.float32)
            check('the GPU axis texel equals the CPU Vs bitwise',
                  bool(np.array_equal(vs_gpu, vs_cpu)),
                  f'{vs_gpu} vs {vs_cpu}')
        sim, _hit = GSH.simulate(job, g)
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(sim[yy, xx] - on[yy, xx, :3]).max(axis=1)
        check('the AXIS facing-quad frame: simulator == CPU bitwise '
              '(a constant vector read from a texture)',
              float(d.max()) == 0.0, f'max {float(d.max()):.3g}')
    cpu, sim, mask, why = _twin(_scene(_settings(default_model='PHONG',
                                                 specular_viewer='AXIS')),
                                _settings(default_model='PHONG',
                                          specular_viewer='AXIS'))
    _frame_bar('the AXIS demo frame', cpu, sim, mask, why)
    # structure: the viewer expression is in the plan signature
    st_a = _settings(default_model='PHONG', specular_viewer='AXIS')
    st_p = _settings(default_model='PHONG', specular_viewer='PIXEL')
    sc_a = _scene(st_a)
    job_a = R.ShadeJob(sc_a, st_a, R.prepare_textures(sc_a, st_a), None,
                       view, eye, W, H)
    job_p = R.ShadeJob(sc_a, st_p, R.prepare_textures(sc_a, st_p), None,
                       view, eye, W, H)
    check('specular_viewer is plan-signature structure (AXIS and PIXEL '
          'never share a cached plan)',
          GSH._plan_sig(job_a, None) != GSH._plan_sig(job_p, None))


# ------------------------------------------------------------------ F014


def test_only_shadow_lamp():
    """F014: a lamp that lights nothing and subtracts its plain diffuse
    where its shadow falls."""
    def frames(**kw):
        st = _settings(**kw)
        sc = FM.SCENES['only_shadow'](st)
        with_lamp = np.asarray(R.render(sc, st))
        st2 = _settings(**kw)
        sc2 = FM.SCENES['only_shadow'](st2)
        sc2.lights = [l for l in sc2.lights if not l.only_shadow]
        without = np.asarray(R.render(sc2, st2))
        return with_lamp, without, sc

    on, off, sc = frames()
    cov = off[:, :, 3] > 0.5
    d = (on[:, :, :3] - off[:, :, :3])
    check('an only-shadow lamp never brightens a pixel (adds nothing)',
          float(d.max()) <= 1e-6, f'max +{float(d.max()):.3g}')
    darker = (d.max(axis=2) < -1e-4) & cov
    check('and darkens at least 1% of the covered pixels (its shadow)',
          int(darker.sum()) >= 0.01 * int(cov.sum()),
          f'{int(darker.sum())} of {int(cov.sum())}')
    on2, off2, _s = frames(shadows=False)
    check('with shadows off the lamp does nothing (bitwise the frame '
          'without it: the named inert case)',
          bool(np.array_equal(on2, off2)))
    # a white shadow colour subtracts nothing; a specular-only lamp too
    st = _settings()
    sc = FM.SCENES['only_shadow'](st)
    sc.lights[-1].shadow_color = (1.0, 1.0, 1.0)
    white = np.asarray(R.render(sc, st))
    check('a WHITE shadow colour subtracts nothing (bitwise the frame '
          'without the lamp)', bool(np.array_equal(white, off)))
    st = _settings()
    sc = FM.SCENES['only_shadow'](st)
    sc.lights[-1].specular_only = True
    so = np.asarray(R.render(sc, st))
    check('a specular-only only-shadow lamp subtracts nothing (no diffuse '
          'lobe to subtract)', bool(np.array_equal(so, off)))
    # a grey shadow colour halves the darkening
    st = _settings()
    sc = FM.SCENES['only_shadow'](st)
    sc.lights[-1].shadow_color = (0.5, 0.5, 0.5)
    half = np.asarray(R.render(sc, st))
    dh = (half[:, :, :3] - off[:, :, :3])
    check('a 50% grey shadow colour halves the subtraction '
          '(1 - vis)*(1 - shcol)',
          float(np.abs(dh - 0.5 * d).max()) < 2e-6,
          f'{float(np.abs(dh - 0.5 * d).max()):.3g}')
    # the twin: the simulator against the CPU on the row
    st = _settings()
    sc = FM.SCENES['only_shadow'](st)
    cpu, sim, mask, why = _twin(sc, st)
    _frame_bar('the only-shadow row', cpu, sim, mask, why)
    if sim is not None:
        # the subtraction itself is bitwise: sim(with) - sim(without) ==
        # cpu(with) - cpu(without) is the deferred bar's business; the
        # lamp's block emits the only-shadow text and no contribution
        # tail on every pass
        view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
        g = CRg.GBuffer(W, H)
        CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                      depth_bits=st.depth_precision)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                         eye, W, H)
        GSH._PLAN_CACHE.clear()
        p, why2, _a = GSH.plan_frame(job, g)
        check('every pass of the row carries the only-shadow block '
              '(hal_osh) -- no refusal',
              p is not None and all('hal_osh' in s for _mi, _n, s, _b in p),
              str(why2))
    # structure: the flag re-plans
    st = _settings()
    sc = FM.SCENES['only_shadow'](st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     W, H)
    sig_on = GSH._plan_sig(job, None)
    sc.lights[-1].only_shadow = False
    sig_off = GSH._plan_sig(job, None)
    check('only_shadow is light-signature structure (never a cache hit '
          'across the toggle)', sig_on != sig_off)


# ------------------------------------------------------------------ F012


def _spot(law, exponent=0.0, size=1.0, hotspot=0.5):
    return Light(type='SPOT', name='cone', position=(0.0, 0.0, 0.0),
                 direction=(0.0, 0.0, -1.0), spot_size=size,
                 spot_blend=0.2, hotspot=hotspot, spot_law=law,
                 spot_exponent=exponent, shadow='NONE')


def _cone_dirs(n=64, span=1.2):
    """L (surface -> light) for n angles 0..span from the axis: the cone
    test's cosine is cos(theta) exactly."""
    th = np.linspace(0.0, span, n).astype(np.float32)
    L = np.stack([-np.sin(th), np.zeros_like(th), np.cos(th)], 1)
    return L.astype(np.float32), np.cos(th).astype(np.float32), th


def test_spot_laws():
    """F012: the OpenGL 1.1, POV-Ray and GameCube GX cone laws."""
    from ..gpu import material as GM
    L, cosang, th = _cone_dirs()
    edge = 0.5                       # spot_size 1.0 -> half angle 0.5
    inside = th < edge - 1e-3
    outside = th > edge + 1e-3
    # GL11, exponent 0: exactly 1 inside, 0 outside
    f = LI.spot_falloff(_spot('GL11'), L)
    check('GL11 exponent 0 is exactly 1.0 inside the cutoff and 0.0 outside',
          bool(np.all(f[inside] == 1.0)) and bool(np.all(f[outside] == 0.0)))
    f8 = LI.spot_falloff(_spot('GL11', 8.0), L)
    check('GL11 exponent 8 is cos^8 inside (1.0 on the axis, falling) and '
          '0 outside',
          f8[0] == 1.0 and bool(np.all(np.diff(f8[inside]) <= 0))
          and bool(np.all(f8[outside] == 0.0))
          and float(np.abs(f8[inside] - cosang[inside] ** 8).max()) < 1e-5)
    # POV: flat 1 inside the hotspot, Hermite to 0 at the edge
    Lf, cf, thf = _cone_dirs(4001, 0.6)
    fp = LI.spot_falloff(_spot('POV'), Lf)
    hot = thf < 0.25 - 1e-4
    check('POV is exactly 1.0 inside the hotspot (radius = hotspot/2)',
          bool(np.all(fp[hot] == 1.0)))
    band = (thf > 0.25 + 1e-3) & (thf < 0.5 - 1e-3)
    check('POV falls monotonically through the Hermite band to 0 at the '
          'cone edge',
          bool(np.all(np.diff(fp[band]) <= 1e-6))
          and float(fp[(thf > 0.5 - 2e-3) & (thf < 0.5)].min()) < 2e-4
          and bool(np.all(fp[thf > 0.5 + 1e-3] == 0.0)),
          f'min at edge {float(fp[(thf > 0.5 - 2e-3) & (thf < 0.5)].min()):.3g}')
    # zero slope at both ends of the Hermite (finite differences)
    i0 = int(np.searchsorted(thf, 0.25)) + 2
    i1 = int(np.searchsorted(thf, 0.5)) - 3
    im = int(np.searchsorted(thf, 0.375))
    d0 = abs(float(fp[i0 + 1] - fp[i0]))
    d1 = abs(float(fp[i1] - fp[i1 - 1]))
    dm = abs(float(fp[im + 1] - fp[im]))
    check("POV's Hermite has zero slope at both ends (the edge steps are "
          'below a tenth of the middle step)',
          d0 < 0.1 * dm and d1 < 0.1 * dm, f'{d0:.2e} {d1:.2e} mid {dm:.2e}')
    fw = LI.spot_falloff(_spot('POV', hotspot=2.0), L)
    check('a hotspot WIDER than the cone gives the hard edge at the cone '
          '(cr clamped to the cutoff)',
          bool(np.all(fw[inside] == 1.0)) and bool(np.all(fw[outside] == 0.0)))
    # GX
    fr = LI.spot_falloff(_spot('GX_RING1'), L)
    cr, a0, a1, a2 = LI.spot_law_coeffs(_spot('GX_RING1'))
    c = float(np.cos(0.5))
    Dd = (1.0 - c) ** 2
    s64 = (-4.0 * c / Dd) + (4.0 * (1.0 + c) / Dd) + (-4.0 / Dd)
    check('GX_RING1 is dark on the axis (a0 + a1 + a2 = 0 in float64 before '
          'rounding) and peaks between the axis and the edge',
          abs(s64) < 1e-6 and float(fr[0]) < 1e-3
          and float(fr[inside].max()) > 0.5, f'{s64:.2e} axis {float(fr[0]):.2e}')
    fc = LI.spot_falloff(_spot('GX_COS'), L)
    check('GX_COS is 1.0 on the axis and non-increasing outward',
          abs(float(fc[0]) - 1.0) < 1e-5 and bool(np.all(np.diff(fc) <= 1e-6)))
    for law in ('GX_FLAT', 'GX_COS2', 'GX_SHARP'):
        fl = LI.spot_falloff(_spot(law), L)
        check(f'{law} is non-increasing away from the axis and within [0, 1]',
              bool(np.all(np.diff(fl) <= 1e-6)) and float(fl.min()) >= 0.0
              and float(fl.max()) <= 1.0)
    fr2 = LI.spot_falloff(_spot('GX_RING2'), L)
    # libogc's RING2 coefficients give exactly 1 at the cutoff cosine
    # (1 - 2cr^2/D + 4cr^2/D - 2cr^2/D) and 1 - 2 = -1 on the axis: the
    # narrower ring peaks AT the cone edge and fades outside it (the
    # spec's "dark at the edge" wording was wrong; measured, not reasoned)
    i_edge = int(np.argmin(np.abs(th - 0.5)))
    check('GX_RING2 is dark on the axis, brightest at the cone edge and '
          'fades outside it',
          float(fr2[0]) < 1e-3 and abs(float(fr2[i_edge]) - 1.0) < 2e-2
          and bool(np.all(np.diff(fr2[th > 0.55]) <= 1e-6)),
          f'axis {float(fr2[0]):.2e} edge {float(fr2[i_edge]):.3f}')
    # the standalone GLSL twin of every law
    n = cosang.size
    for law, expo, bar in (('GL11', 0.0, 0.0), ('GL11', 8.0, 2e-6),
                           ('POV', 0.0, 0.0), ('POV', 8.0, 2e-6),
                           ('GX_FLAT', 0.0, 0.0), ('GX_COS', 0.0, 0.0),
                           ('GX_COS2', 0.0, 0.0), ('GX_SHARP', 0.0, 0.0),
                           ('GX_RING1', 0.0, 0.0), ('GX_RING2', 0.0, 0.0)):
        lt = _spot(law, expo)
        cr, a0, a1, a2 = LI.spot_law_coeffs(lt)
        si = np.float32(np.cos(float(lt.spot_size) * 0.5))
        body = GM.spot_law_glsl(law, expo != 0.0,
                                {'si': 'u_si', 'exp': 'u_exp', 'cr': 'u_cr',
                                 'a0': 'u_a0', 'a1': 'u_a1', 'a2': 'u_a2'})
        src = ('uniform float cosang;\nuniform float u_si;\n'
               'uniform float u_exp;\nuniform float u_cr;\n'
               'uniform float u_a0;\nuniform float u_a1;\n'
               'uniform float u_a2;\nout vec4 Color;\nvoid main()\n{\n'
               + '\n'.join(body) + '\n    Color = vec4(spot_f, 0.0, 0.0, 1.0);'
               '\n}\n')
        got, err = _run_glsl(src, n, cosang=cosang, u_si=si,
                             u_exp=np.float32(expo), u_cr=cr, u_a0=a0,
                             u_a1=a1, u_a2=a2)
        want = LI.spot_law_factor(lt, cosang)
        if got is None:
            check(f'the {law} cone GLSL compiles standalone', False, str(err))
            continue
        d = float(np.abs(got[:, 0] - want).max())
        check(f'the {law} (exponent {expo:g}) cone GLSL equals the CPU law '
              + ('bitwise (d == 0.0)' if bar == 0.0 else f'within {bar:g}'),
              d == 0.0 if bar == 0.0 else d < bar, f'd {d:.3g}')
    # the frame twin on the three rows, and the A/B against BLENDER
    for key in ('spot law POV (hotspot + Hermite)',
                'spot law OpenGL 1.1 (exponent 8)',
                'spot law GX ring (GameCube)'):
        sc, st = FM.build(key)
        st.transparency = 'NONE'
        cpu, sim, mask, why = _twin(sc, st)
        _frame_bar(f'the row {key!r}', cpu, sim, mask, why)
        sc2, st2 = FM.build(key)
        st2.transparency = 'NONE'
        sc2.lights[-1].spot_law = 'BLENDER'
        sc2.lights[-1].spot_exponent = 0.0
        base = np.asarray(R.render(sc2, st2))
        check(f'the row {key!r} differs from the BLENDER cone (the law '
              'moves the picture)',
              float(np.abs(cpu - base).max()) > 1e-3,
              f'{float(np.abs(cpu - base).max()):.4f}')
    # structure: the law and the exponent flag re-plan
    a = _spot('POV', 0.0)
    b = _spot('POV', 4.0)
    c_ = _spot('GL11', 0.0)
    check('the cone law and a nonzero exponent are light-signature structure',
          GSH._light_sig(a) != GSH._light_sig(b)
          and GSH._light_sig(a) != GSH._light_sig(c_))
    check('but the exponent VALUE is a texel (4 vs 8 share a plan)',
          GSH._light_sig(b) == GSH._light_sig(_spot('POV', 8.0)))
    # refusal: none by design -- a spot-law plan never refuses
    sc, st = FM.build('spot law GX ring (GameCube)')
    st.transparency = 'NONE'
    _c, sim, _m, why = _twin(sc, st)
    check('no cone law refuses the GPU (none is needed: every law is '
          'float32 arithmetic)', sim is not None, str(why))


# ------------------------------------------------------------------ F013


def _point(decay, end=4.0, ld1=0.0, ld2=0.0, rb=0.5):
    return Light(type='POINT', name='fade', position=(0.0, 0.0, 0.0),
                 decay=decay, decay_end=end, decay_ld1=ld1, decay_ld2=ld2,
                 gx_ref_brite=rb, shadow='NONE')


def test_decay_laws():
    """F013: OpenGL's three-term attenuation, POV-Ray's fade laws and the
    GX distance tables."""
    from ..gpu import material as GM
    st = _settings()
    D = 4.0
    dist = np.linspace(0.0, 2.0 * D, 257).astype(np.float32)
    i_D = int(np.argmin(np.abs(dist - D)))
    assert float(dist[i_D]) == D
    for mode in ('POV_FADE_LINEAR', 'POV_FADE_SQUARE'):
        a = LI.attenuate(_point(mode, D), dist, st)
        check(f'{mode} is exactly 2.0 at the lamp and exactly 1.0 at '
              'd = Falloff End (2/1 and 2/(1 + 1))',
              float(a[0]) == 2.0 and float(a[i_D]) == 1.0
              and bool(np.all(np.diff(a) <= 0)),
              f'{float(a[0])} {float(a[i_D])}')
    a = LI.attenuate(_point('GL_3TERM', D), dist, st)
    check('GL_3TERM with both sliders 0 is exactly 1.0 everywhere (GL\'s '
          '(1, 0, 0) default)', bool(np.all(a == 1.0)))
    a = LI.attenuate(_point('GL_3TERM', D, ld1=0.5, ld2=0.1), dist, st)
    want = 1.0 / (1.0 + 0.5 * dist.astype(np.float64)
                  + 0.1 * dist.astype(np.float64) ** 2)
    check('GL_3TERM with sliders is 1/(1 + kl d + kq d^2) (within 1e-6 of '
          'float64) and non-increasing',
          float(np.abs(a - want).max()) < 1e-6
          and bool(np.all(np.diff(a) <= 0)))
    rb = 0.3
    for mode in ('GX_GENTLE', 'GX_MEDIUM', 'GX_STEEP'):
        a = LI.attenuate(_point(mode, D, rb=rb), dist, st)
        ulp = float(np.spacing(np.float32(rb)))
        check(f'{mode} is exactly ref_brite at d = Falloff End (within 1 ulp '
              'of float32(0.3)) and monotone non-increasing from 1.0',
              abs(float(a[i_D]) - float(np.float32(rb))) <= ulp
              and float(a[0]) == 1.0 and bool(np.all(np.diff(a) <= 0)),
              f'at D {float(a[i_D]):.7f}')
    # the scene default reaches the new laws too (DEFAULT -> the setting)
    st_d = _settings(light_falloff_default='POV_FADE_SQUARE')
    a1 = LI.attenuate(_point('DEFAULT', D), dist, st_d)
    a2 = LI.attenuate(_point('POV_FADE_SQUARE', D), dist, st_d)
    check('a lamp on the Scene Default takes the scene\'s new law bitwise',
          bool(np.array_equal(a1, a2)))
    # the standalone GLSL twin of every law
    n = dist.size
    for mode, kw in (('GL_3TERM', dict(ld1=0.5, ld2=0.1)),
                     ('GL_3TERM', {}), ('POV_FADE_LINEAR', {}),
                     ('POV_FADE_SQUARE', {}), ('GX_GENTLE', dict(rb=0.3)),
                     ('GX_MEDIUM', dict(rb=0.3)), ('GX_STEEP', dict(rb=0.3))):
        lt = _point(mode, D, **kw)
        k1, k2 = LI.gx_dist_coeffs(lt, mode)
        body = GM.decay_law_glsl(mode, {'D': 'u_D', 'ld1': 'u_ld1',
                                        'ld2': 'u_ld2', 'k1': 'u_k1',
                                        'k2': 'u_k2'})
        src = ('uniform float dist;\nuniform float u_D;\nuniform float u_ld1;'
               '\nuniform float u_ld2;\nuniform float u_k1;\n'
               'uniform float u_k2;\nout vec4 Color;\nvoid main()\n{\n'
               + '\n'.join(body) + '\n    Color = vec4(att, 0.0, 0.0, 1.0);'
               '\n}\n')
        got, err = _run_glsl(src, n, dist=dist,
                             u_D=np.float32(max(D, LI.EPS)),
                             u_ld1=np.float32(lt.decay_ld1),
                             u_ld2=np.float32(lt.decay_ld2), u_k1=k1,
                             u_k2=k2)
        want = LI.attenuate(lt, dist, st)
        if got is None:
            check(f'the {mode} decay GLSL compiles standalone', False,
                  str(err))
            continue
        d = float(np.abs(got[:, 0] - want).max())
        check(f'the {mode}{" (sliders 0.5/0.1)" if kw.get("ld1") else ""} '
              'decay GLSL equals LI.attenuate bitwise (d == 0.0)', d == 0.0,
              f'd {d:.3g}')
    # the frame twin on the three rows, and the A/B against the default
    for key in ('POV fade_distance / fade_power 2',
                'OpenGL 1.1 three-term attenuation',
                'GX distance attenuation STEEP (GameCube)'):
        sc, st = FM.build(key)
        st.transparency = 'NONE'
        cpu, sim, mask, why = _twin(sc, st)
        _frame_bar(f'the row {key!r}', cpu, sim, mask, why)
        sc2, st2 = FM.build(key)
        st2.transparency = 'NONE'
        # the row's lamp sits on the Scene Default and the row sets the
        # law AS the scene default: the reference is the same lamp under
        # the 1.89.0 default (INVERSE_SQUARE)
        st2.light_falloff_default = 'INVERSE_SQUARE'
        base = np.asarray(R.render(sc2, st2))
        check(f'the row {key!r} differs from the default decay (the law '
              'moves the picture)',
              float(np.abs(cpu - base).max()) > 1e-3,
              f'{float(np.abs(cpu - base).max()):.4f}')
    # the GX coefficients ride the texel road: the packer uploads the
    # same numbers attenuate evaluates, and the cache-hit repack
    # follows the lamp (a ref_brite drag re-uploads, never re-plans)
    sc, st = FM.build('GX distance attenuation STEEP (GameCube)')
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     W, H)
    lights = LI.select_lights(sc.lights, st)
    tex = GM.pack_light_texels(lights, job)
    i_pt = [i for i, l in enumerate(lights) if l.type == 'POINT'][0]
    k1, k2 = LI.gx_dist_coeffs(lights[i_pt], 'GX_STEEP')
    check('the packer uploads the GX (k1, k2) attenuate evaluates, bitwise',
          float(tex[0, i_pt * GM.LIGHT_TEXEL_STRIDE + 5, 1]) == float(k1)
          and float(tex[0, i_pt * GM.LIGHT_TEXEL_STRIDE + 5, 2]) == float(k2)
          and float(k2) > 0.0)
    sig_a = GSH._light_sig(lights[i_pt])
    lights[i_pt].gx_ref_brite = 0.7
    check('Ref Brightness is a texel, not structure (a drag is a cache hit)',
          GSH._light_sig(lights[i_pt]) == sig_a)
    lights[i_pt].decay = 'GX_GENTLE'
    check('the decay law is structure (already in _light_sig)',
          GSH._light_sig(lights[i_pt]) != sig_a)
    # presets: POV lamps do not fade by default; SGI's GL has (1, 0, 0)
    from ..presets.library import PRESETS
    for key in ('POVRAY_31', 'POVRAY_2', 'SGI_INDY'):
        check(f'the {key} preset sets light_falloff_default NONE',
              PRESETS[key]['settings'].get('light_falloff_default') == 'NONE')
    check('the SGI_INDY preset lights with the infinite viewer (AXIS)',
          PRESETS['SGI_INDY']['settings'].get('specular_viewer') == 'AXIS')


# ------------------------------------------------------------------ F015


def test_screen_spot():
    """F015: the Sega Model 3 viewport spotlight -- an ellipse on the
    screen, a depth window, a lobe added to the diffuse and to the fog."""
    from ..gpu import material as GM
    # the pure lobe
    cx, cy, w_e, h_e, s0, aext = (np.float32(48.0), np.float32(36.0),
                                  np.float32(20.0), np.float32(12.0),
                                  np.float32(0.1), np.float32(10.0))
    yy, xx = np.mgrid[0:H, 0:W]
    px = xx.ravel().astype(np.float32)
    py = yy.ravel().astype(np.float32)
    dep = np.full(px.size, 5.0, np.float32)
    en, el, lobe = LI.screen_spot_lobe(px, py, dep, cx, cy, w_e, h_e, s0,
                                       aext)
    lobe2 = lobe.reshape(H, W)
    r2 = ((px + 0.5 - cx) / w_e) ** 2 + ((py + 0.5 - cy) / h_e) ** 2
    check('the lobe is 0 outside the screen ellipse',
          bool(np.all(lobe[r2 >= 1.0] == 0.0)))
    check('and positive inside it', bool(np.all(lobe[r2 < 0.98] > 0.0)))
    i_max = int(np.argmax(lobe))
    check('the lobe is maximal at the centre pixel',
          abs(float(px[i_max]) + 0.5 - float(cx)) <= 0.5
          and abs(float(py[i_max]) + 0.5 - float(cy)) <= 0.5,
          f'at ({px[i_max]}, {py[i_max]})')
    row = lobe2[36, 48:]
    check('the lobe is non-increasing along a ray from the centre',
          bool(np.all(np.diff(row) <= 0.0)))
    check('the lobe is symmetric in x about cx (mirrored integer pixels '
          'bitwise)', bool(np.array_equal(lobe2[:, 48:68], lobe2[:, 47:27:-1])))
    en2, _el2, lobe_near = LI.screen_spot_lobe(px, py, np.full(px.size, 0.05,
                                                              np.float32),
                                               cx, cy, w_e, h_e, s0, aext)
    check('a depth below the start gives 0 (en = 0)',
          bool(np.all(en2 == 0.0)) and bool(np.all(lobe_near == 0.0)))
    _e, _l, lobe_full = LI.screen_spot_lobe(px, py, np.full(px.size, 3.0,
                                                           np.float32),
                                            cx, cy, w_e, h_e, s0, aext)
    _e, _l, lobe_far = LI.screen_spot_lobe(px, py, np.full(px.size, 30.0,
                                                          np.float32),
                                           cx, cy, w_e, h_e, s0, aext)
    check('inside the extent the window is full (lobe == ellipse) and '
          'beyond it the window fades',
          bool(np.array_equal(lobe_full, el))
          and float(lobe_far.max()) < float(lobe_full.max()))
    # the standalone GLSL twin of the lobe
    body = GM.screen_spot_glsl({'cx': 'u_cx', 'cy': 'u_cy', 'w': 'u_w',
                                'h': 'u_h', 'start': 'u_s0',
                                'aext': 'u_aext'})
    src = ('uniform float ss_px;\nuniform float ss_py;\n'
           'uniform float ss_depth;\nuniform float u_cx;\n'
           'uniform float u_cy;\nuniform float u_w;\nuniform float u_h;\n'
           'uniform float u_s0;\nuniform float u_aext;\nout vec4 Color;\n'
           'void main()\n{\n' + '\n'.join(body)
           + '\n    Color = vec4(ss_lobe, ss_en, ss_el, 1.0);\n}\n')
    depv = (np.linspace(0.0, 30.0, px.size) % 30.0).astype(np.float32)
    got, err = _run_glsl(src, px.size, ss_px=px, ss_py=py, ss_depth=depv,
                         u_cx=cx, u_cy=cy, u_w=w_e, u_h=h_e, u_s0=s0,
                         u_aext=aext)
    en3, el3, lobe3 = LI.screen_spot_lobe(px, py, depv, cx, cy, w_e, h_e,
                                          s0, aext)
    check('the lobe GLSL compiles standalone', got is not None, str(err))
    if got is not None:
        d = max(float(np.abs(got[:, 0] - lobe3).max()),
                float(np.abs(got[:, 1] - en3).max()),
                float(np.abs(got[:, 2] - el3).max()))
        check('the lobe GLSL equals LI.screen_spot_lobe bitwise (d == 0.0) '
              'over every pixel and a depth ramp', d == 0.0, f'd {d:.3g}')
    # the CPU frame: every pixel >= the frame without the lamp
    st = _settings()
    sc = FM.SCENES['screen_spot'](st)
    on = np.asarray(R.render(sc, st))
    st2 = _settings()
    sc2 = FM.SCENES['screen_spot'](st2)
    sc2.lights = [l for l in sc2.lights if not l.screen_spot]
    off = np.asarray(R.render(sc2, st2))
    d = on[:, :, :3] - off[:, :, :3]
    check('a screen-spot lamp never darkens a pixel', float(d.min()) >= -1e-6,
          f'min {float(d.min()):.3g}')
    check('and brightens a pool of them (its ellipse on the screen)',
          int((d.max(axis=2) > 1e-3).sum()) > 50
          and int((d.max(axis=2) > 1e-3).sum()) < 0.9 * W * H,
          f'{int((d.max(axis=2) > 1e-3).sum())} px')
    # the lamp lives on the screen: a batch with no pixel gets nothing
    # (ray hits), a batch with pixels accumulates the fog lobe colE*(en*el)
    orig = R.light_surface
    seen = {'hits': 0, 'hits_equal': 0, 'fog_ok': 0, 'fog_bad': 0}

    def wrapped(surf, model, ctx, scene, settings, *a, **k):
        a_out = orig(surf, model, ctx, scene, settings, *a, **k)
        lamps = list(scene.lights)
        if getattr(ctx, 'px', None) is None:
            scene.lights = [l for l in lamps if not l.screen_spot]
            try:
                b_out = orig(surf, model, ctx, scene, settings, *a, **k)
            finally:
                scene.lights = lamps
            seen['hits'] += 1
            seen['hits_equal'] += int(bool(np.array_equal(a_out, b_out)))
        else:
            lamp = [l for l in lamps if l.screen_spot][0]
            vp = R.camera_matrices(scene.camera, ctx.width, ctx.height)[2]
            prm = LI.screen_spot_params(lamp, vp, ctx.width, ctx.height,
                                        scene.camera)
            en_, el_, _lb = LI.screen_spot_lobe(ctx.px, ctx.py, ctx.depth,
                                                *prm)
            colE = np.asarray(lamp.color, np.float32) * np.float32(
                float(lamp.energy) / (4.0 * np.pi))
            want = colE[None, :] * (en_ * el_)[:, None]
            sf = getattr(ctx, 'spot_fog', None)
            ok = sf is not None and bool(np.array_equal(
                sf, want.astype(np.float32)))
            seen['fog_ok' if ok else 'fog_bad'] += 1
        return a_out

    st3 = _settings(raytrace=True, ray_depth=1, ray_reflection=True)
    sc3 = FM.SCENES['screen_spot'](st3)
    sc3.materials[2].reflect_level = 0.5
    R.light_surface = wrapped
    try:
        R.render(sc3, st3)
    finally:
        R.light_surface = orig
    check('ray-hit batches (px None) get nothing from the screen spot '
          '(bitwise the batch without it)',
          seen['hits'] > 0 and seen['hits_equal'] == seen['hits'],
          str(seen))
    check('every pixel batch accumulates the fog lobe colE*(en*el) in '
          'ctx.spot_fog, bitwise', seen['fog_ok'] > 0 and seen['fog_bad'] == 0,
          str(seen))
    # the frame twin on the row, and the GPU structure
    sc, st = FM.build('Model 3 screen-space spotlight')
    st.transparency = 'NONE'
    cpu, sim, mask, why = _twin(sc, st)
    _frame_bar('the screen-spot row', cpu, sim, mask, why)
    if sim is not None:
        view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
        g = CRg.GBuffer(W, H)
        CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                      depth_bits=st.depth_precision)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                         eye, W, H)
        GSH._PLAN_CACHE.clear()
        p, _why, atl = GSH.plan_frame(job, g)
        check('every primary pass carries the screen-spot block and the '
              'fog-lobe accumulator, and binds hal_lights + hal_fogtab',
              p is not None and all(
                  'hal_sst' in s and 'vec3 hal_spotfog' in s
                  and 'hal_lights' in (b.get('samplers') or ())
                  and 'hal_fogtab' in (b.get('samplers') or ())
                  for _mi, _n, s, b in p))
        lights = LI.select_lights(sc.lights, st)
        tex = GM.pack_light_texels(lights, job)
        i_ss = [i for i, l in enumerate(lights) if l.screen_spot][0]
        prm = LI.screen_spot_params(lights[i_ss], vp, W, H, sc.camera)
        b6 = i_ss * GM.LIGHT_TEXEL_STRIDE + 6
        check('the packer uploads the ellipse the CPU loop computes, bitwise',
              bool(np.array_equal(tex[0, b6, :4], np.asarray(prm[:4],
                                                             np.float32)))
              and float(tex[0, b6 + 1, 0]) == float(prm[4])
              and float(tex[0, b6 + 1, 1]) == float(prm[5]))
    consts_min = {'resolution': (float(W), float(H)), 'cookies': {},
                  'light_links': {}, 'falloff_default': 'INVERSE_SQUARE',
                  'specular_viewer': 'PIXEL', 'light_texels': True}
    lamp = FM.SCENES['screen_spot'](_settings()).lights[-1]
    check('a secondary or layer pass emits NO screen-spot block (the lamp '
          'lives on the screen; hits have no pixel)',
          GM._one_light_source(0, lamp, dict(consts_min, __screen_pass=False),
                               bake={}) == []
          and GM._one_light_source(0, lamp, dict(consts_min, __screen_pass=True),
                                   bake={'__cel_key': True}) == []
          and any('hal_sst' in ln for ln in GM._one_light_source(
              0, lamp, dict(consts_min, __screen_pass=True), bake={})))
    sig_on = GSH._light_sig(lamp)
    lamp.screen_spot = False
    check('screen_spot is light-signature structure',
          GSH._light_sig(lamp) != sig_on)
    lamp.screen_spot = True
    # fog_spot: the gate is plan structure; 0 leaves every frame bitwise
    # (the identity test above pins a fogged frame); the lobe's consumer
    # is LIGHT-A1's core/fog.py (hal_fog SPOT32 * hal_spotfog) -- not in
    # this worktree, stated in the handover
    st_a = _settings(fog=True, fog_spot=1.0)
    st_b = _settings(fog=True, fog_spot=0.0)
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    ja = R.ShadeJob(sc, st_a, R.prepare_textures(sc, st_a), None, view, eye,
                    W, H)
    jb = R.ShadeJob(sc, st_b, R.prepare_textures(sc, st_b), None, view, eye,
                    W, H)
    check('fog_spot > 0 is plan-signature structure (the fog reads the lobe)',
          GSH._plan_sig(ja, None) != GSH._plan_sig(jb, None))
    from ..presets.library import PRESETS
    check('no preset flips fog_spot (a lamp-driven dial; presets keep 0)',
          all(float(p['settings'].get('fog_spot', 0.0)) == 0.0
              for p in PRESETS.values()))


# ------------------------------------------------------------------ main


TESTS = [test_identity_at_defaults, test_axis_viewer,
         test_only_shadow_lamp, test_spot_laws, test_decay_laws,
         test_screen_spot]


def main():
    from . import utf8_console
    utf8_console()
    for t in TESTS:
        print(f'\n--- {t.__name__}')
        try:
            t()
        except Exception:                                   # noqa: BLE001
            traceback.print_exc()
            FAILS.append(t.__name__ + ' (exception)')
    print()
    print(f'LIGHT-B1 lamps: {len(FAILS)} failure(s)')
    for f in FAILS:
        print('  FAIL ' + f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
