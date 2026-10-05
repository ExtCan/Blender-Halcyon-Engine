"""R251 post-signal pack, the codec slot (SIG-3: C134, C132, C133).

The optical printer and the two codecs the era's displays showed:
Tron's backlit Kodalith mattes summed on the linear negative, Video
CD's MPEG-1 intra blocks, and RAD's Smacker 4x4 blocks over a frame
palette. Every check reads as a sentence of what it proves.

    python -m halcyon.tests.test_r251_post_codec
"""

import contextlib
import importlib
import io
import sys
import traceback

import numpy as np

from ..core import post as PO
from ..core import render as R
from ..core import signal_codec as SC
from ..core import signal_era as SIG
from ..core import palette as PA
from ..core.settings import RenderSettings
from ..core.texture import Texture
from ..gpu import chain
from ..gpu import chain_codec as CC
from ..gpu import chain_palette as CP
from ..gpu import frame as FR
from ..gpu import shade as GSH
from ..gpu import stages
from ..gpu import stages_codec
from ..gpu import stages_signal
from ..presets.library import apply_preset, PRESETS
from ..shaders.compiler import try_compile
from . import fakedevice
from . import r251_common as RC
from .scenebuild import demo_scene
from .test_render import base_settings, _prev_engine

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


W, H = 96, 72


def settings(**kw):
    st = base_settings(W, H)
    st.shadows = False
    st.transparency = 'NONE'
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def post_kw(sc, st):
    """Exactly the 1.89.0 keywords (the neutrality pin hands this dict to
    the old engine; never `gel=`)."""
    return dict(frame=7, seed=st.seed, target_size=(W, H),
                allow_resize=False,
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None),
                flare_sources=getattr(sc, 'last_flares', None))


def gpu_road(st, mutate=None, with_gel=False, frame=None):
    """Render + post through the fake device, then the CPU chain over the
    SAME frame (only the device switch differs): (out_g, out_c, rec, live).
    `mutate(sc)` edits the scene (the gel); `with_gel` hands the CURRENT
    engine `gel=sc.last_gel` on both roads (the shared `post_kw` never
    carries it: the neutrality pin gives that dict to the 1.89.0 engine)."""
    st.render_device = 'GPU'
    sc = demo_scene(st, with_texture=False)
    if mutate is not None:
        mutate(sc)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    st_c = st.copy()
    st_c.render_device = 'CPU'

    def kw(s):
        d = post_kw(sc, s)
        if frame is not None:
            d['frame'] = int(frame)
        if with_gel:
            d['gel'] = getattr(sc, 'last_gel', None)
        return d

    with fakedevice.installed() as dev:
        st._keep_gpu_frame = True
        try:
            img = R.render(sc, st)
            out = PO.process(img, st, **kw(st))
            rec = dict(PO.LAST_CHAIN)
        finally:
            FR.release(st)
        live = [t for t in dev.targets if not t.freed]
    out_c = PO.process(img, st_c, **kw(st_c))
    return out, out_c, rec, live


def run_stage(name, h, w, uniforms, samplers):
    """One stage through the GLSL simulator at (h, w): (H, W, 4) or
    (None, err)."""
    src = stages.STAGES[name].replace('in vec2 vUV;', 'uniform vec2 vUV;')
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, err
    n = h * w
    yy, xx = np.mgrid[0:h, 0:w]
    u = {'vUV': np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h],
                         1).astype(np.float32)}
    for k, v in samplers.items():
        u[k] = Texture(v, colorspace='Non-Color', filt='NEAREST',
                       wrap='EXTEND')
    for k, v in uniforms.items():
        if isinstance(v, int):
            u[k] = np.full(n, int(v), np.int32)
        elif isinstance(v, (tuple, list)):
            u[k] = np.tile(np.asarray(v, np.float32)[None, :], (n, 1))
        else:
            u[k] = np.full(n, float(v), np.float32)
    outs, _d = prog.run(u, {}, n)
    return np.asarray(outs['Color'], np.float32).reshape(h, w, 4), None


def _rgba(img, a=None):
    img = np.asarray(img, np.float32)
    if a is None:
        a = np.ones(img.shape[:2] + (1,), np.float32)
    return np.ascontiguousarray(np.concatenate([img, a], 2), np.float32)


def _rec_ok(rec, live, d, want, tol=0.0):
    ok = d <= tol and live == []
    if 'stages_has' in want:
        ok &= all(s in (rec.get('stages') or []) for s in want['stages_has'])
    if 'readbacks' in want:
        ok &= len(rec.get('readbacks') or []) == want['readbacks']
    if 'readback_names' in want:
        rb = rec.get('readbacks') or []
        ok &= len(rb) == len(want['readback_names']) and all(
            nm in r[0] for nm, r in zip(want['readback_names'], rb))
    if 'uploads' in want:
        ok &= rec.get('uploads') == want['uploads']
    return ok


def _capture(fn):
    """Run fn() with the once-only sets cleared; returns (result, text)."""
    chain._WARNED.clear()
    SIG._ONCE.clear()
    PO._POST_WARNED.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = fn()
    return got, buf.getvalue()


def _demo_display(**kw):
    """The demo frame as the codecs see it: rendered, display-encoded
    (the frame between the halftone and the quant), (H, W, 3) float32."""
    st = settings(**kw)
    sc = demo_scene(st, with_texture=False)
    img = R.render(sc, st)
    rgb = PO.display_transform(np.asarray(img, np.float32)[:, :, :3], st)
    return np.ascontiguousarray(rgb, np.float32), st


def _gel_ball(sc):
    sc.materials[1].glow_gel = (0.2, 0.9, 1.0)


# ---------------------------------------------------------------- the pin

_CASES = [
    ('the defaults', {}),
    ('the CEL_ANIME_MODERN preset', 'preset'),
    ('gamma 2.2 with film grain 0.3 at size 1.0',
     {'gamma': 2.2, 'film_grain': 0.3, 'film_grain_size': 1.0}),
    ("colour depth '15'", {'color_depth': '15'}),
    ('a BAYER4 dither', {'dither': 'BAYER4'}),
    ('a glow with film grain 0.2 and gate weave 0.5',
     {'glow': True, 'film_grain': 0.2, 'film_weave': 0.5}),
    ('a halftone 0.5', {'film_halftone': 0.5}),
    ('CRT scanlines 0.4 with the composite cable',
     {'crt': True, 'crt_scanlines': 0.4, 'composite': True}),
    ('JPEG artefacts at quality 30', {'jpeg_artifacts': True,
                                      'jpeg_quality': 30}),
]


def _case_settings(kw):
    if kw == 'preset':
        st = base_settings(W, H)
        apply_preset(st, 'CEL_ANIME_MODERN')
        st.resolution_x, st.resolution_y = W, H
        st.aa_samples = 1
        return st
    return settings(**kw)


def test_r251_a00_codec_defaults_are_invisible():
    """The slot's pin: the 1.89.0 zip is beside the package and every
    default leaves every pixel of render AND post bitwise the previous
    release's; a material with a gel changes nothing while the stage is
    off; every `_on` predicate is False on a default settings object;
    every stage function is the identity at defaults; the slot's stages
    merged before ENABLED was derived."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the neutrality pin runs)',
          RP is not None)
    if RP is not None:
        prev_post = importlib.import_module(
            RP.__name__.rsplit('.', 1)[0] + '.post')
        for label, kw in _CASES:
            st = _case_settings(kw)
            sc = demo_scene(st, with_texture=False)
            RC.clear_palette_locks(RP)
            now_r = R.render(sc, st)
            now = PO.process(now_r, st, **post_kw(sc, st))
            st2 = _case_settings(kw)
            sc2 = demo_scene(st2, with_texture=False)
            prev_r = RP.render(sc2, st2)
            prev = prev_post.process(prev_r, st2, **post_kw(sc2, st2))
            same = now.shape == prev.shape and bool(np.array_equal(now, prev))
            check(f'the CPU device, {label}: render bitwise the 1.89.0 '
                  "release's", bool(np.array_equal(now_r, prev_r)))
            check(f'the CPU device, {label}: render + post bitwise the '
                  "1.89.0 release's (the slot's defaults are invisible)",
                  same, f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
        # a gel with the stage off changes nothing: the gelled scene on
        # this engine against the plain scene on the old one
        st = settings()
        sc = demo_scene(st, with_texture=False)
        _gel_ball(sc)
        now_r = R.render(sc, st)
        now = PO.process(now_r, st, **post_kw(sc, st),
                         gel=getattr(sc, 'last_gel', None))
        st2 = settings()
        sc2 = demo_scene(st2, with_texture=False)
        prev = prev_post.process(RP.render(sc2, st2), st2, **post_kw(sc2, st2))
        check('a material with a Glow Gel under matte_glow=False renders '
              'bitwise the 1.89.0 engine without gels, and no gel plane is '
              'built', bool(np.array_equal(now, prev))
              and getattr(sc, 'last_gel', 0) is None)
    st = RenderSettings()
    preds = {'matte_glow': SC.matte_glow_on, 'mpeg1': SC.mpeg1_on,
             'smacker': SC.smacker_on}
    check('no codec / printer predicate is on for a default settings object',
          not any(p(st) for p in preds.values()),
          str({k: p(st) for k, p in preds.items()}))
    rng = np.random.default_rng(251)
    img = rng.random((12, 16, 3)).astype(np.float32)
    gel = rng.random((12, 16, 3)).astype(np.float32)
    fns = {'matte_glow': lambda: SC.matte_glow(img, st, gel=gel),
           'mpeg1_intra': lambda: SC.mpeg1_intra(img, st, 3),
           'smacker': lambda: SC.smacker(img, st, 0)}
    for name, fn in fns.items():
        got = fn()
        check(f'signal_codec.{name} on default settings returns its input '
              'bitwise (the opening guard)',
              got is img or bool(np.array_equal(got, img)))
    S = stages_codec
    check("the slot's stages merged into stages.STAGES before ENABLED was "
          'derived (a stage pasted after ENABLED would refuse silently)',
          set(S.STAGES_CODEC) <= set(stages.ENABLED)
          and set(S.STAGES_CODEC) <= set(stages.STAGES)
          and set(S.INTERFACE_CODEC) == set(S.STAGES_CODEC)
          and set(S.VALIDATION_CODEC) == set(S.STAGES_CODEC),
          str(sorted(set(S.STAGES_CODEC) - set(stages.ENABLED))))
    check('every orchestrated component stage is in SIGNAL_MULTI (the self '
          'test never draws one alone) and SMACKER is not',
          set(S.STAGES_CODEC) - {'SMACKER'} <= set(stages_signal.SIGNAL_MULTI)
          and 'SMACKER' not in stages_signal.SIGNAL_MULTI)
    check('every slot stage is a chain function post.process finds by name',
          all(callable(getattr(chain, n, None))
              for n in ('matte_glow', 'mpeg1', 'smacker')))
    check('the widest stage binds 7 samplers, under the 16 every driver '
          'guarantees (no sampler refusal can fire)',
          max(len(v['samplers']) for v in S.INTERFACE_CODEC.values()) == 7
          and len(S.INTERFACE_CODEC['MATTE_ADD']['samplers']) == 7)
    for name, src in S.STAGES_CODEC.items():
        bad = [w for w in ('texture(', 'log10', 'saturate', 'lerp(', 'frac(',
                           ' % ') if w in src]
        check(f'{name} stays in the accepted GLSL subset (texelFetch only, '
              'no %, no HLSL names)', not bad, str(bad))


# ------------------------------------------------- C134: the matte glow

def _matte_st(**kw):
    base = dict(matte_glow=True, matte_glow_radius=8.0, matte_glow_passes=3,
                matte_glow_exposure=1.0)
    base.update(kw)
    return settings(**base)


def test_r251_c134_matte_glow():
    """Tron's backlit mattes: the laws of a sum of exposures of a binary
    matte in its gel's colour, the engine road's gel plane, the GPU twin
    stage by stage and through the fake device, and the refusals."""
    rng = np.random.default_rng(134)
    rgb = (rng.random((H, W, 3)) * 0.5).astype(np.float32)
    gel = np.zeros((H, W, 3), np.float32)
    gel[34:38, 46:50] = (0.0, 1.0, 1.0)
    st = _matte_st()
    out = SC.matte_glow(rgb, st, gel=gel)
    add = out - rgb
    P = SC.matte_params(st, H)
    check('three passes at radius 8 on 72 lines are sigmas 0.53, 1.07, 2.13 '
          'px (taps of radius 2, 4, 7) at exposures 1/2, 1/4, 1/8',
          P.radii == [2, 4, 7]
          and [float(w) for w in P.w] == [0.5, 0.25, 0.125]
          and float(P.e) == 1.0, f'{P.radii} {[float(w) for w in P.w]}')
    check('every tap set sums to 1 within float32 and is symmetric',
          all(abs(float(t.astype(np.float64).sum()) - 1.0) < 1e-6
              and bool(np.array_equal(t, t[::-1])) for t in P.taps))
    check('exposures only add: the glow is >= 0 everywhere (within the '
          "sum's own rounding)", float(add.min()) >= -1e-6,
          f'{float(add.min())}')
    check('inside the matte the glow is at least the crisp exposure e * gel',
          bool((add[34:38, 46:50, 1:] >= 1.0 - 1e-6).all()))
    row = add[35, :, 1]
    right = row[49:]
    left = row[:47][::-1]
    check("the halo follows the matte: along the centre row the glow never "
          'grows with distance from the square',
          bool((np.diff(right) <= 1e-6).all())
          and bool((np.diff(left) <= 1e-6).all()))
    far = np.ones((H, W), bool)
    far[34 - 7:38 + 7, 46 - 7:50 + 7] = False
    check('pixels farther than the largest radius are bitwise untouched '
          '(the line stays crisp)', bool(np.array_equal(out[far], rgb[far])))
    want = float(P.e) * float(gel[..., 1].sum()) * (1 + 0.5 + 0.25 + 0.125)
    got = float(add[..., 1].astype(np.float64).sum())
    check('the energy added per channel is e * sum(gel) * (1 + 1/2 + 1/4 + '
          '1/8) within 1e-3 (relative)', abs(got - want) <= 1e-3 * want,
          f'{got} vs {want}')
    check('a channel the gel does not carry (red) is bitwise untouched',
          bool(np.array_equal(out[..., 0], rgb[..., 0])))
    out2 = SC.matte_glow(rgb, _matte_st(matte_glow_exposure=2.0), gel=gel)
    d = np.abs((out2 - rgb)[34:38, 46:50, 1:] - 2.0 * add[34:38, 46:50, 1:])
    check('doubling Matte Exposure doubles the addition at the gel pixels '
          'within 1e-6', float(d.max()) <= 1e-6, f'{float(d.max())}')
    one = SC.matte_glow(rgb, _matte_st(matte_glow_passes=1), gel=gel)
    wide = SC.matte_glow(rgb, _matte_st(matte_glow_radius=24.0), gel=gel)
    check('Matte Passes and Matte Radius both move the halo',
          not np.array_equal(one, out) and not np.array_equal(wide, out))
    flat = np.full((H, W, 3), 0.25, np.float32)
    lum_gel = np.zeros((H, W, 3), np.float32)
    lum_gel[10:14, 10:14] = (1.0, 0.0, 0.0)
    bright = flat.copy()
    bright[50:60, 60:80] = 4.0
    g1 = SC.matte_glow(flat, st, gel=lum_gel) - flat
    g2 = SC.matte_glow(bright, st, gel=lum_gel) - bright
    check('the halo follows the MATTE, not the picture: a brighter picture '
          'elsewhere adds no glow of its own (not a luminance bloom)',
          float(np.abs(g1 - g2).max()) <= 1e-6
          and float(np.abs(g2[45:65, 55:85]).max()) == 0.0)
    # no gel plane: the named no-op
    st_on = _matte_st()
    got, text = _capture(lambda: SC.matte_glow(rgb, st_on, gel=None))
    check('with no gel plane the stage returns its input bitwise and the '
          "console names the 'gel plane'",
          got is rgb and 'gel plane' in text
          and 'gel plane' in str(SC.matte_glow_why(rgb, None)), text.strip())
    check('a gel plane of another size is refused by name (never broadcast)',
          'does not match' in str(SC.matte_glow_why(rgb, gel[:10])))
    st_v = _matte_st()
    st_v._viewport = True
    check('the viewport names the printer as an F12 stage',
          'F12' in str(SC.matte_glow_why(rgb, gel, st_v))
          and SC.matte_glow(rgb, st_v, gel=gel) is rgb)

    # ---- the engine road
    st = _matte_st(pass_material_index=True)
    sc = demo_scene(st, with_texture=False)
    _gel_ball(sc)
    img = R.render(sc, st)
    lg = getattr(sc, 'last_gel', None)
    ma = (sc.last_passes or {}).get('IndexMA')
    ok = lg is not None and lg.shape == (H, W, 3) and lg.dtype == np.float32
    check('R.render leaves scene.last_gel, (H, W, 3) float32', ok,
          str(None if lg is None else lg.shape))
    if ok and ma is not None:
        ball = np.asarray(ma, np.float32).reshape(H, W, -1)[..., 0] == 1.0
        cov = np.asarray(img)[..., 3] > 0
        nz = lg.any(axis=2)
        check("the plane's nonzero pixels are exactly the ball's material "
              "pixels, each the gel's own colour",
              bool(np.array_equal(nz, ball & cov)) and int(nz.sum()) > 50
              and bool(np.array_equal(
                  lg[nz], np.tile(np.float32((0.2, 0.9, 1.0)),
                                  (int(nz.sum()), 1)))),
              f'{int(nz.sum())} gel px, {int((ball & cov).sum())} ball px')
        kw = post_kw(sc, st)
        with_gel = PO.process(img, st, **kw, gel=lg)
        st_off = st.copy()
        st_off.matte_glow = False
        plain = PO.process(img, st_off, **kw, gel=lg)
        diff = np.abs(with_gel - plain).max(axis=2) > 0
        ys, xs = np.nonzero(nz)
        R3 = SC.matte_params(st, H).radii[-1]
        near = np.zeros((H, W), bool)
        near[max(ys.min() - R3, 0):ys.max() + R3 + 1,
             max(xs.min() - R3, 0):xs.max() + R3 + 1] = True
        check('post.process(..., gel=scene.last_gel) differs from the frame '
              'without the stage around the ball only',
              bool(diff.any()) and not bool((diff & ~near).any()),
              f'{int(diff.sum())} px moved, {int((diff & ~near).sum())} far')
    st_ss = _matte_st(aa_mode='SUPERSAMPLE', aa_samples=4)
    sc_ss = demo_scene(st_ss, with_texture=False)
    _gel_ball(sc_ss)
    R.render(sc_ss, st_ss)
    check('under 2x2 supersampling the plane is at OUTPUT resolution (the '
          'top-left sample of each pixel, as last_depth)',
          getattr(sc_ss, 'last_gel', None) is not None
          and sc_ss.last_gel.shape == (H, W, 3))
    st_st = _matte_st(stereo_mode='ANAGLYPH')
    sc_st = demo_scene(st_st, with_texture=False)
    _gel_ball(sc_st)
    sc_st.last_gel = np.ones((H, W, 3), np.float32)        # a stale plane
    R.render(sc_st, st_st)
    check('a stereo pair has no single material plane: last_gel is None '
          '(never a stale plane), and the stage is a named no-op',
          getattr(sc_st, 'last_gel', 0) is None)
    sc_nm = demo_scene(st, with_texture=False)
    _gel_ball(sc_nm)
    sc_nm.mesh.mat_index = None
    R._GBUF_CACHE.clear()
    try:
        R.render(sc_nm, _matte_st())
        lg0 = getattr(sc_nm, 'last_gel', None)
        check('a mesh with no material indices gives a BLACK plane (the '
              'stage adds nothing and prints nothing)',
              lg0 is not None and not lg0.any())
    except Exception as exc:                                    # noqa: BLE001
        check('a mesh with no material indices renders', False, repr(exc))
    R._GBUF_CACHE.clear()

    # ---- the GPU twin, stage by stage
    st = _matte_st()
    P = SC.matte_params(st, H)
    gel_r = (rng.random((H, W, 3)) * (rng.random((H, W, 1)) > 0.8)
             ).astype(np.float32)
    for k in range(3):
        taps_img = SC.matte_taps_image(P.taps[k])
        gh, err = run_stage('MATTE_BLUR', H, W,
                            {'resolution': (float(W), float(H)),
                             'radius': int(P.radii[k]), 'dir_x': 1,
                             'dir_y': 0},
                            {'source': _rgba(gel_r), 'taps': taps_img})
        want_h = SC.blur_h(gel_r, P.taps[k])
        dh = float(np.abs(gh[..., :3] - want_h).max()) if gh is not None \
            else float('inf')
        gv, err2 = run_stage('MATTE_BLUR', H, W,
                             {'resolution': (float(W), float(H)),
                              'radius': int(P.radii[k]), 'dir_x': 0,
                              'dir_y': 1},
                             {'source': _rgba(want_h), 'taps': taps_img})
        want_v = SC.blur_v(want_h, P.taps[k])
        dv = float(np.abs(gv[..., :3] - want_v).max()) if gv is not None \
            else float('inf')
        check(f'MATTE_BLUR pass {k + 1} (radius {P.radii[k]}): horizontal '
              'and vertical are bitwise blur_h / blur_v in the simulator',
              dh == 0.0 and dv == 0.0, f'd {dh} {dv} {err or err2 or ""}')
    blurs = [SC.blur_v(SC.blur_h(gel_r, t), t) for t in P.taps]
    alpha = rng.random((H, W, 1)).astype(np.float32)
    binds = {'source': _rgba(rgb, alpha), 'gel': _rgba(gel_r)}
    uni = {'resolution': (float(W), float(H)), 'passes': 3, 'e': float(P.e)}
    for i in range(5):
        binds[f'b{i + 1}'] = _rgba(blurs[i]) if i < 3 else _rgba(gel_r)
        uni[f'w{i + 1}'] = float(P.w[i]) if i < 3 else 0.0
    ga, err = run_stage('MATTE_ADD', H, W, uni, binds)
    want = SC.matte_glow(rgb, st, gel=gel_r)
    da = float(np.abs(ga[..., :3] - want).max()) if ga is not None \
        else float('inf')
    check('MATTE_ADD is bitwise matte_glow in the simulator and carries the '
          "frame's alpha", da == 0.0 and ga is not None
          and bool(np.array_equal(ga[..., 3:], alpha)), f'd {da} {err or ""}')
    check('MATTE_ADD and MATTE_BLUR carry the CLOSE 5e-4 grade (a '
          "driver's FMA is an ulp per tap; the self test reads it)",
          stages.VALIDATION['MATTE_ADD'] == ('CLOSE', 0.0005)
          and stages.VALIDATION['MATTE_BLUR'] == ('CLOSE', 0.0005))

    # ---- the fake-device road (dither NONE: no quant readback)
    for passes in (3, 1, 5):
        st = _matte_st(matte_glow_radius=12.0, matte_glow_passes=passes)
        out_g, out_c, rec, live = gpu_road(st, mutate=_gel_ball,
                                           with_gel=True)
        d = float(np.abs(out_g - out_c).max())
        check(f'the fake-device road, {passes} pass(es): the gel scene is '
              'bitwise the CPU chain, MATTE_GLOW recorded, no CPU readback, '
              'every target freed',
              _rec_ok(rec, live, d, dict(stages_has=['MATTE_GLOW'],
                                         readbacks=0)),
              f'd {d} stages {rec.get("stages")} readbacks '
              f'{rec.get("readbacks")} live {len(live)}')
    st_off = _matte_st(matte_glow=False)
    off_g, _c, _r, _l = gpu_road(st_off, mutate=_gel_ball, with_gel=True)
    check('the glow moves the frame on the GPU road (the gel reached the '
          'chain through the closure\'s keyword)',
          not np.array_equal(off_g, out_g))
    # no gel on the GPU road: the wiring's gate keeps the frame where it
    # is (no readback is spent on a no-op) and names the reason
    st = _matte_st()
    (out_g, out_c, rec, live), text = _capture(
        lambda: gpu_road(st, mutate=_gel_ball, with_gel=False))
    d = float(np.abs(out_g - out_c).max())
    check('with no gel handed to post on the GPU road the stage is the '
          'named no-op on both roads: bitwise, no readback, MATTE_GLOW not '
          'recorded',
          _rec_ok(rec, live, d, dict(readbacks=0))
          and 'MATTE_GLOW' not in (rec.get('stages') or [])
          and 'gel plane' in text, text.strip()[:200])

    # ---- the refusals
    big = _matte_st(matte_glow_radius=64.0, matte_glow_passes=5)
    check("matte_glow_refusal names the 'loop bound' at 4320 lines with "
          'radius 64 and five passes',
          'loop bound' in str(CC.matte_glow_refusal(big, h=4320)),
          str(CC.matte_glow_refusal(big, h=4320)))
    with fakedevice.installed():
        st_g = _matte_st(matte_glow_radius=8.0, matte_glow_passes=5,
                         render_device='GPU')
        check('at 72 lines (and with no h on the 72-tall settings) radius 8 '
              'with five passes may draw (the widest tap radius is 26)',
              CC.matte_glow_refusal(st_g, h=72) is None
              and CC.matte_glow_refusal(st_g) is None
              and SC.matte_params(st_g, H).radii[-1] == 26,
              str(CC.matte_glow_refusal(st_g)))
        fr = chain.Frame(rgb=rgb.copy())
        got, text = _capture(lambda: chain.matte_glow(fr, st_g))
        check('chain.matte_glow with no gel keyword refuses by name',
              got is None and 'MATTE_GLOW on the CPU' in text
              and 'gel' in text, text.strip())
        fr.release()
    st_cpu = _matte_st()
    check('on the CPU device the refusal names the device',
          'CPU' in str(CC.matte_glow_refusal(st_cpu)))
    # the loop-bound refusal through the fake device: the CPU stage runs
    # over the named readback, bitwise
    st = _matte_st(matte_glow_radius=64.0, matte_glow_passes=5,
                   matte_glow_exposure=0.5)
    (out_g, out_c, rec, live), text = _capture(
        lambda: gpu_road(st, mutate=_gel_ball, with_gel=True))
    d = float(np.abs(out_g - out_c).max())
    check('radius 64 with five passes at 72 lines (tap radius 205) refuses '
          'by name on the fake device and the CPU stage runs over one named '
          'readback, bitwise',
          _rec_ok(rec, live, d, dict(readback_names=['matte_glow']))
          and 'MATTE_GLOW' not in (rec.get('stages') or [])
          and 'loop bound' in text,
          f'd {d} {rec.get("readbacks")} {text.strip()[:160]}')

# ---------------------------------------------- C132: MPEG-1 intra blocks

def _probe_false(chain_fn, st, img):
    """The chain function under a device whose probe fails: (got, text)."""
    from ..gpu import device as DEV
    st = st.copy()
    st.render_device = 'GPU'
    with fakedevice.installed():
        DEV.probe = lambda: (False, 'no driver')     # restored by the context
        st._frame_gpu_shaded = True
        fr = chain.Frame(rgb=np.asarray(img, np.float32)[:, :, :3].copy())
        got, text = _capture(lambda: chain_fn(fr, st))
        fr.release()
    return got, text


def test_r251_c132_mpeg1():
    """Video CD's intra road: the quantiser's laws (DC in steps of 8,
    odd-or-zero AC, the GOP pumping the scale), the flat-block law, the
    five GPU passes bitwise the CPU's own intermediates, the padded
    edge, the fake-device road and the refusal."""
    # ---- the flat-block law
    ok = True
    worst = ''
    for qs in (1, 8, 31):
        for col in ((0.2, 0.5, 0.8), (1.0, 0.0, 0.25), (0.5, 0.5, 0.5)):
            flat = np.tile(np.float32(col), (24, 40, 1))
            st = settings(mpeg1=True, mpeg1_qscale=qs, mpeg1_gop=0)
            out = SC.mpeg1_intra(flat, st, 0)
            v8 = SIG.to_u8(flat)
            y = SIG.y601(v8[..., 0], v8[..., 1], v8[..., 2])
            cb = SIG.cb601(v8[..., 0], v8[..., 1], v8[..., 2])
            cr = SIG.cr601(v8[..., 0], v8[..., 1], v8[..., 2])
            Rr, Gg, Bb = SIG.yuv601_decode(y, cb, cr)
            want = SIG.U8[np.stack([Rr, Gg, Bb], 2)]
            if not np.array_equal(out, want):
                ok = False
                worst = f'qs {qs} colour {col}'
    check('a flat frame comes back bitwise its BT.601 round trip at every '
          'scale (the DC in steps of 8 holds a flat block; the AC residue '
          'truncates to zero)', ok, worst)

    # ---- the quantiser's laws
    rng = np.random.default_rng(132)
    X = ((rng.random((64, 64)) - 0.5) * 2000.0).astype(np.float32)
    v = (np.arange(64) & 7)[:, None]
    u = (np.arange(64) & 7)[None, :]
    dc = (v == 0) & (u == 0)
    good = True
    for qs in (1, 3, 8, 12, 31):
        Rq = SC._mpeg1_quant(X, qs, SC.mpeg1_inv(qs))
        ri = Rq.astype(np.int64)
        good &= bool(np.array_equal(ri.astype(np.float32), Rq))
        good &= bool(((ri[dc] % 8) == 0).all())
        ac = ri[~np.broadcast_to(dc, ri.shape)]
        good &= bool(((ac == 0) | ((np.abs(ac) & 1) == 1)).all())
        good &= bool((np.abs(ac) <= 2047).all())
    check('through _mpeg1_quant every AC reconstruction is odd or zero '
          '(mismatch control), every DC a multiple of 8, |R| <= 2047, at '
          'scales 1, 3, 8, 12, 31', good)
    check('the intra matrix is ISO/IEC 11172-2\'s default (8 at DC, 83 at '
          'the last coefficient, symmetric first row / column)',
          SC.MPEG1_INTRA.shape == (8, 8) and int(SC.MPEG1_INTRA[0, 0]) == 8
          and int(SC.MPEG1_INTRA[7, 7]) == 83
          and int(SC.MPEG1_INTRA.sum()) == 2114
          and list(SC.MPEG1_INTRA[0]) == [8, 16, 19, 22, 26, 27, 29, 34])
    D = SC.DCT8.astype(np.float64)
    check('the DCT table is orthonormal within float32',
          float(np.abs(D @ D.T - np.eye(8)).max()) < 1e-6)

    rgb, _st = _demo_display()
    errs = []
    for qs in (2, 8, 31):
        st = settings(mpeg1=True, mpeg1_qscale=qs, mpeg1_gop=0)
        out = SC.mpeg1_intra(rgb, st, 0)
        errs.append(float(np.abs(out - np.clip(rgb, 0, 1)).mean()))
    check('the mean error on the demo frame grows with the scale 2, 8, 31',
          errs[0] < errs[1] < errs[2], str(errs))
    out8 = SC.mpeg1_intra(rgb, settings(mpeg1=True, mpeg1_gop=0), 0)
    check('the recoded frame sits on the byte lattice (U8 values) at the '
          "frame's own size",
          out8.shape == rgb.shape and out8.dtype == np.float32
          and bool(np.array_equal(out8, SIG.U8[SIG.to_u8(out8)])))

    # ---- the GOP law
    st = settings(mpeg1=True, mpeg1_qscale=8, mpeg1_gop=15)
    qi = [SC.mpeg1_qs(st, f) for f in (0, 15, 3, 6)]
    qb = [SC.mpeg1_qs(st, f) for f in (1, 2, 4)]
    st0 = settings(mpeg1=True, mpeg1_qscale=8, mpeg1_gop=0)
    check('at a group of 15 the I and P pictures (frames 0, 15, 3, 6) take '
          'the I scale 8 and the B pictures (1, 2, 4) take (14 * 8 + 5) // 10 '
          '= 11; a group of 0 codes every frame at the I scale',
          qi == [8, 8, 8, 8] and qb == [11, 11, 11]
          and all(SC.mpeg1_qs(st0, f) == 8 for f in range(40)),
          f'{qi} {qb}')
    st31 = settings(mpeg1=True, mpeg1_qscale=31, mpeg1_gop=15)
    check('the B scale is capped at 31', SC.mpeg1_qs(st31, 1) == 31)
    f0 = SC.mpeg1_intra(rgb, st, 0)
    f1 = SC.mpeg1_intra(rgb, st, 1)
    f1b = SC.mpeg1_intra(rgb, st, 1)
    check("frame 1's picture (a B picture at scale 11) differs from frame "
          "0's (I at 8), and the same frame renders bitwise twice",
          not np.array_equal(f0, f1) and bool(np.array_equal(f1, f1b)))

    # ---- the GPU twin, pass by pass, at the padded size
    st = settings(mpeg1=True, mpeg1_qscale=12, mpeg1_gop=0)
    P = SC.mpeg1_params(st, 0, H, W)
    check('the 96x72 frame pads to 96x80 (whole macroblocks)',
          (P.Wp, P.Hp) == (96, 80) and P.qs == 12)
    v8 = np.pad(SIG.to_u8(rgb), ((0, P.Hp - H), (0, P.Wp - W), (0, 0)), 'edge')
    enc, drow, dcolq, irow, out_i = SC._mpeg1_stages(v8, P)
    pad = (float(P.Wp), float(P.Hp))
    dct = SC.mpeg1_dct_image()
    qtab = SC.mpeg1_q_image(P.qs)
    alpha = rng.random((H, W, 1)).astype(np.float32)
    g, err = run_stage('MPEG_ENC', P.Hp, P.Wp,
                       {'resolution': pad, 'src_size': (float(W), float(H))},
                       {'source': _rgba(rgb, alpha)})
    d = float(np.abs(g[..., :3] - enc).max()) if g is not None else float('inf')
    check('MPEG_ENC at the padded size is bitwise the CPU\'s encode',
          d == 0.0, f'd {d} {err or ""}')
    check("the edge law: rows 72..79 of the luma equal row 71 (the clamp is "
          "the FRAME's, not the draw's), and the stage's text carries "
          '`src_size`',
          g is not None and bool(np.array_equal(
              g[H:, :, 0], np.tile(g[H - 1:H, :, 0], (P.Hp - H, 1))))
          and 'src_size' in stages.STAGES['MPEG_ENC']
          and 'ivec2(src_size) - ivec2(1)' in stages.STAGES['MPEG_ENC'])
    check("the encode carries the frame's alpha", g is not None
          and bool(np.array_equal(g[:H, :, 3:], alpha)))
    steps = (('MPEG_DCT_ROW', enc, drow, {'resolution': pad}, {'dct': dct}),
             ('MPEG_DCT_COL_Q', drow, dcolq, {'resolution': pad,
                                              'qs': int(P.qs)},
              {'dct': dct, 'qtab': qtab}),
             ('MPEG_IDCT_ROW', dcolq, irow, {'resolution': pad}, {'dct': dct}))
    for name, src, want, uni, tex in steps:
        binds = {'source': _rgba(src)}
        binds.update(tex)
        g, err = run_stage(name, P.Hp, P.Wp, uni, binds)
        d = float(np.abs(g[..., :3] - want).max()) if g is not None \
            else float('inf')
        check(f"{name} is bitwise the CPU's own intermediate in the "
              'simulator', d == 0.0, f'd {d} {err or ""}')
    g, err = run_stage('MPEG_IDCT_COL_OUT', H, W,
                       {'resolution': (float(W), float(H))},
                       {'source': _rgba(irow), 'dct': dct,
                        'u8lut': SIG.u8_image()})
    want = SC.mpeg1_intra(rgb, st, 0)
    d = float(np.abs(g[..., :3] - want).max()) if g is not None else float('inf')
    check('MPEG_IDCT_COL_OUT drawn at the frame\'s size over the padded '
          'source is bitwise mpeg1_intra (and its bytes the CPU\'s decode)',
          d == 0.0 and bool(np.array_equal(want, SIG.U8[out_i[:H, :W]])),
          f'd {d} {err or ""}')
    check('the five MPEG passes carry the CLOSE 1/255 grade (a driver\'s '
          'FMA can cross one truncation; the self test reads it)',
          all(stages.VALIDATION[n] == ('CLOSE', 0.004)
              for n in CC.MPEG_STAGES))

    # ---- the fake-device road (dither NONE: no quant readback)
    for label, kw, fno in (
            ('scale 12, every frame an I picture',
             dict(mpeg1=True, mpeg1_qscale=12, mpeg1_gop=0), None),
            ('scale 8 in a group of 15, frame 7 (a B picture)',
             dict(mpeg1=True, mpeg1_qscale=8, mpeg1_gop=15), 7),
            ("scale 4 into a 5:6:5 framebuffer", dict(
                mpeg1=True, mpeg1_qscale=4, mpeg1_gop=0, color_depth='16'),
             None)):
        out_g, out_c, rec, live = gpu_road(settings(**kw), frame=fno)
        d = float(np.abs(out_g - out_c).max())
        check(f'the fake-device road, {label}: bitwise the CPU chain, MPEG1 '
              'recorded, no CPU readback, the padded targets freed',
              _rec_ok(rec, live, d, dict(stages_has=['MPEG1'], readbacks=0)),
              f'd {d} stages {rec.get("stages")} readbacks '
              f'{rec.get("readbacks")} live {len(live)}')
    st_off = settings()
    off_g, _c, _r, _l = gpu_road(st_off)
    check('and the recode moves the frame', not np.array_equal(off_g, out_g))

    # ---- the refusal
    with fakedevice.installed():
        check('mpeg1_refusal is None when the stage may draw',
              CC.mpeg1_refusal(settings(mpeg1=True,
                                        render_device='GPU')) is None)
    check('mpeg1_refusal is None with the stage off and names the device on '
          'the CPU device', CC.mpeg1_refusal(settings()) is None
          and 'CPU' in str(CC.mpeg1_refusal(settings(mpeg1=True))))
    got, text = _probe_false(chain.mpeg1, settings(mpeg1=True), rgb)
    check("a device whose probe fails makes the chain print 'MPEG1 on the "
          "CPU: no driver' and return None",
          got is None and 'MPEG1 on the CPU: no driver' in text, text.strip())


# ------------------------------------------------- C133: Smacker blocks

def _cell_palette(seed=133):
    """A 256-entry test palette built ON the inverse colormap's cell
    centres ((q + 0.5) / 64), every entry in its own cell -- so
    lut[cell(pal[i])] == i for every entry. Entries 0 and 1 are the dark
    and the bright grey of the mono block."""
    rng = np.random.default_rng(seed)
    cells = {(12, 12, 12), (51, 51, 51)}
    order = [(12, 12, 12), (51, 51, 51)]
    while len(order) < 256:
        c = tuple(int(v) for v in rng.integers(0, 64, 3))
        if c not in cells:
            cells.add(c)
            order.append(c)
    return ((np.asarray(order, np.float32) + np.float32(0.5))
            / np.float32(64.0)).astype(np.float32)


def _cells_of(pal):
    q = np.clip((SIG.to_u8(pal) * 64) // 255, 0, 63)
    return (q[:, 0] << 12) | (q[:, 1] << 6) | q[:, 2]


def test_r251_c133_smacker():
    """RAD's block grammar: Fill, Mono and Full over the frame palette,
    the tie rules, the palette lock shared with the 8-bit quantiser, the
    GPU twin in the simulator and through the fake device (the one named
    palette readback, none once the lock is warm), and the refusal."""
    rng = np.random.default_rng(133)
    pal = _cell_palette()
    st = settings(smacker=True, smacker_quality=0.5)
    key = ('ADAPTIVE', 256, str(st.palette_method), 0)
    PA.clear_caches()
    PA.cached_adaptive(key, lambda: pal)
    palc, lut = SC.smacker_palette_cached(st, 0)
    pal8 = SIG.to_u8(palc)
    check('the test palette sits on the cube\'s cell centres: every entry '
          'is the nearest entry of its own cell, and the codec reads it '
          "through the palette quantiser's own lock key",
          bool(np.array_equal(lut[_cells_of(palc)], np.arange(256)))
          and bool(np.array_equal(palc, pal)))
    check('the thresholds at quality 0.5 are t_fill 31 and t_full 48 * 32^2; '
          'at quality 1 a block is full as soon as two colours miss it',
          SC.smacker_thresholds(0.5) == (31, 49152)
          and SC.smacker_thresholds(1.0) == (5, 0)
          and SC.smacker_thresholds(0.0) == (56, 196608),
          str(SC.smacker_thresholds(0.5)))
    frame = np.zeros((8, 8, 3), np.float32)
    frame[0:4, 0:4] = (0.30, 0.55, 0.70)                    # flat
    frame[0:2, 4:8] = 12.5 / 64.0                           # dark grey rows
    frame[2:4, 4:8] = 51.5 / 64.0                           # bright grey rows
    frame[4:8, 0:4] = rng.random((4, 4, 3)).astype(np.float32)
    frame[4:8, 4:8] = (0.9, 0.1, 0.1)
    frame[5, 5] = (0.92, 0.1, 0.1)                          # still a fill
    out, kind = SC.smacker_with(frame, palc, lut, 0.5, detail=True)
    f8 = SIG.to_u8(frame)
    mean = (f8[0:4, 0:4].reshape(16, 3).sum(0) + 8) >> 4
    i_mean, _p = SC.nearest8(lut, pal8, mean)
    check('a flat block is a FILL: all sixteen pixels are the nearest entry '
          'of the block\'s mean',
          int(kind[0, 0]) == 0 and int(kind[1, 1]) == 0
          and bool((out[0:4, 0:4] == palc[i_mean]).all()))
    blk = out[0:4, 4:8].reshape(16, 3)
    cols = np.unique(blk, axis=0)
    check('two dark rows under two bright rows are a MONO block: exactly two '
          'colours, the nearest entries of the two halves\' means, each '
          'pixel the nearer of the two',
          int(kind[0, 1]) == 1 and len(cols) == 2
          and bool((out[0:2, 4:8] == palc[0]).all())
          and bool((out[2:4, 4:8] == palc[1]).all()))
    out1, kind1 = SC.smacker_with(frame, palc, lut, 1.0, detail=True)
    i_full, _p = SC.nearest8(lut, pal8, f8[4:8, 0:4])
    check('a random block at quality 1 is FULL: each pixel its own nearest '
          'entry', int(kind1[1, 0]) == 2
          and bool(np.array_equal(out1[4:8, 0:4], palc[i_full])))
    # the mono tie: a pixel equidistant from the two entries takes the DARK
    tie = np.zeros((4, 4, 3), np.float32)
    tie[0:2] = SIG.U8[50]
    tie[2:4] = SIG.U8[205]
    p2 = np.zeros((256, 3), np.float32)
    p2[:] = SIG.U8[np.arange(256)][:, None]
    # a two-entry decision: every cell with a dark red goes to entry 60,
    # every other to entry 200 (the means 60 and 196 land on them)
    lut2 = np.where((np.arange(64 ** 3) >> 12) < 32, 60, 200).astype(np.int32)
    tie[1, 1] = SIG.U8[130]                          # |130-60| == |130-200|
    tie[3, 3] = SIG.U8[130]
    o_t, k_t = SC.smacker_with(tie, p2, lut2, 0.5, detail=True)
    check('a MONO pixel equidistant from the two entries takes the dark '
          'entry (the `<=` rule), in either half',
          int(k_t[0, 0]) == 1 and float(o_t[1, 1, 0]) == float(SIG.U8[60])
          and float(o_t[3, 3, 0]) == float(SIG.U8[60]),
          f'kind {int(k_t[0, 0])} {o_t[1, 1, 0] * 255:.0f} '
          f'{o_t[3, 3, 0] * 255:.0f}')
    # the '8' road on the synthetic palette: the palette maps onto itself
    st8 = settings(smacker=True, color_depth='8', palette_mode='ADAPTIVE',
                   palette_size=256)
    big = np.tile(frame, (3, 4, 1))
    PA.clear_caches()
    PA.cached_adaptive(key, lambda: pal)
    coded = SC.smacker(big, st8, 0)
    again = PO.reduce_depth(coded, st8, 0)
    check("colour depth '8' after the codec leaves the cell-centre palette's "
          'frame bitwise unchanged (the quantiser reads the same locked '
          'palette and every entry wins its own cell)',
          bool(np.array_equal(coded, again)))

    # ---- the demo frame
    rgb, _st = _demo_display()
    PA.clear_caches()
    st = settings(smacker=True)
    out = SC.smacker(rgb, st, st.seed)
    pal_d, lut_d = SC.smacker_palette_cached(st, st.seed)
    pset = {tuple(p) for p in pal_d.tolist()}
    check('every output pixel of the demo frame is an entry of its 256-'
          'colour palette, bitwise',
          out.shape == rgb.shape
          and all(tuple(p) in pset
                  for p in np.unique(out.reshape(-1, 3), axis=0).tolist()),
          f'{len(pset)} entries')
    check('the same (frame, seed) recodes bitwise twice',
          bool(np.array_equal(out, SC.smacker(rgb, st, st.seed))))
    counts = []
    kinds = []
    for q in (0.1, 0.5, 0.9):
        o, kd = SC.smacker_with(rgb, pal_d, lut_d, q, detail=True)
        hb, wb = kd.shape
        b = o[:hb * 4, :wb * 4].reshape(hb, 4, wb, 4, 3).transpose(
            0, 2, 1, 3, 4).reshape(hb * wb, 16, 3)
        counts.append(float(np.mean([len(np.unique(x, axis=0)) for x in b])))
        kinds.append([int((kd == i).sum()) for i in range(3)])
    check('the mean number of distinct colours per block never falls as the '
          'quality rises 0.1, 0.5, 0.9, and the frame holds all three block '
          'kinds at 0.5',
          counts[0] <= counts[1] <= counts[2] and counts[0] < counts[2]
          and all(k > 0 for k in kinds[1]), f'{counts} {kinds}')
    low = SC.smacker_with(rgb, pal_d, lut_d, 0.0)
    check('Smacker Quality moves the picture', not np.array_equal(low, out))
    # the conditional invariance under the '8' quant on a median-cut palette
    st8 = settings(smacker=True, color_depth='8', palette_mode='ADAPTIVE',
                   palette_size=256)
    PA.clear_caches()
    coded = SC.smacker(rgb, st8, st8.seed)
    pal_d, lut_d = SC.smacker_palette_cached(st8, st8.seed)
    q8 = PO.reduce_depth(coded, st8, st8.seed)
    wins = lut_d[_cells_of(pal_d)] == np.arange(pal_d.shape[0])
    idx_of = {tuple(p): i for i, p in enumerate(pal_d.tolist())}
    flat = coded.reshape(-1, 3)
    pix_idx = np.array([idx_of[tuple(p)] for p in flat.tolist()])
    keep = wins[pix_idx]
    check("under colour depth '8' every pixel whose entry wins its own cube "
          'cell is unchanged, and every pixel stays a palette entry (two '
          'entries within 1/64 share a cell only one of them wins)',
          bool(np.array_equal(q8.reshape(-1, 3)[keep], flat[keep]))
          and int(keep.sum()) > flat.shape[0] // 2
          and all(tuple(p) in idx_of
                  for p in np.unique(q8.reshape(-1, 3), axis=0).tolist()),
          f'{int(keep.sum())} of {flat.shape[0]} px on winning entries')

    # ---- the GPU twin in the simulator
    PA.clear_caches()
    st = settings(smacker=True)
    pal_d, lut_d = SC.smacker_palette(st, rgb, st.seed)
    icm_img = PA.icm_index_image(PA.get_inverse_colormap(pal_d))
    pal_img = CP.pal_image(pal_d)
    alpha = rng.random((H, W, 1)).astype(np.float32)
    for q in (0.2, 0.8):
        stq = settings(smacker=True, smacker_quality=q)
        g, err = run_stage('SMACKER', H, W, CC.smacker_uniforms(stq, W, H),
                           {'source': _rgba(rgb, alpha), 'pal': pal_img,
                            'icm': icm_img})
        want = SC.smacker_with(rgb, pal_d, lut_d, q)
        d = float(np.abs(g[..., :3] - want).max()) if g is not None \
            else float('inf')
        check(f'SMACKER at quality {q} is bitwise the CPU in the simulator '
              "and carries the frame's alpha",
              d == 0.0 and g is not None
              and bool(np.array_equal(g[..., 3:], alpha)), f'd {d} {err or ""}')
    odd = rgb[:70, :94]
    g, err = run_stage('SMACKER', 70, 94, CC.smacker_uniforms(st, 94, 70),
                       {'source': _rgba(odd), 'pal': pal_img, 'icm': icm_img})
    want = SC.smacker_with(odd, pal_d, lut_d, 0.5)
    d = float(np.abs(g[..., :3] - want).max()) if g is not None else float('inf')
    check('on a 94x70 frame the edge blocks clamp to the frame on both roads '
          '(np.pad edge == the clamped fetch)', d == 0.0, f'd {d} {err or ""}')
    check('SMACKER carries the EXACT grade (integer reductions, fetched '
          'palette colours)',
          stages.VALIDATION['SMACKER'] == ('EXACT', 0.00001))

    # ---- the fake-device road: cold lock, then warm, back to back
    PA.clear_caches()
    out_g, out_c, rec, live = gpu_road(settings(smacker=True))
    d = float(np.abs(out_g - out_c).max())
    check('the fake-device road from a cold palette lock: bitwise the CPU '
          'chain, SMACKER recorded, ONE readback named SMACKER (the palette '
          'is fitted on the CPU), one upload',
          _rec_ok(rec, live, d, dict(stages_has=['SMACKER'],
                                     readback_names=['SMACKER'], uploads=1)),
          f'd {d} stages {rec.get("stages")} readbacks {rec.get("readbacks")} '
          f'uploads {rec.get("uploads")} live {len(live)}')
    out_g2, out_c2, rec, live = gpu_road(settings(smacker=True))
    d = float(np.abs(out_g2 - out_c2).max())
    check('and immediately again on the same settings and seed (the lock '
          'warm): no readback, bitwise',
          _rec_ok(rec, live, d, dict(stages_has=['SMACKER'], readbacks=0))
          and bool(np.array_equal(out_g2, out_g)),
          f'd {d} readbacks {rec.get("readbacks")}')
    PA.clear_caches()
    out_g, out_c, rec, live = gpu_road(settings(
        smacker=True, smacker_quality=0.3, color_depth='8',
        palette_mode='ADAPTIVE', palette_size=256))
    d = float(np.abs(out_g - out_c).max())
    check("with colour depth '8' behind it the quantiser peeks the codec's "
          'lock: bitwise the CPU chain, the one palette readback, SMACKER '
          'then PALETTE on the GPU',
          _rec_ok(rec, live, d, dict(stages_has=['SMACKER', 'PALETTE'],
                                     readback_names=['SMACKER'])),
          f'd {d} stages {rec.get("stages")} readbacks {rec.get("readbacks")}')
    PA.clear_caches()
    st_nl = settings(smacker=True, palette_lock=False)
    a_g, a_c, rec, live = gpu_road(st_nl)
    b_g, b_c, rec2, live2 = gpu_road(settings(smacker=True,
                                              palette_lock=False))
    check('with Lock Palette off the palette is refitted from every frame: '
          'the named readback every time, bitwise the CPU chain',
          _rec_ok(rec, live, float(np.abs(a_g - a_c).max()),
                  dict(readback_names=['SMACKER']))
          and _rec_ok(rec2, live2, float(np.abs(b_g - b_c).max()),
                      dict(readback_names=['SMACKER'])),
          f'{rec.get("readbacks")} {rec2.get("readbacks")}')
    PA.clear_caches()

    # ---- the refusal
    with fakedevice.installed():
        check('smacker_refusal is None when the stage may draw',
              CC.smacker_refusal(settings(smacker=True,
                                          render_device='GPU')) is None)
    check('smacker_refusal is None with the stage off and names the device '
          'on the CPU device', CC.smacker_refusal(settings()) is None
          and 'CPU' in str(CC.smacker_refusal(settings(smacker=True))))
    got, text = _probe_false(chain.smacker, settings(smacker=True), rgb)
    check("a device whose probe fails makes the chain print 'SMACKER on the "
          "CPU: no driver' and return None (no readback was spent)",
          got is None and 'SMACKER on the CPU: no driver' in text,
          text.strip())


# ------------------------------------------- the wiring around the stages

def _src(rel):
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with io.open(os.path.join(root, *rel.split('/')), encoding='utf8') as fh:
        return fh.read()


def test_r251_codec_wiring():
    """Where the three stages sit in post.process, how the gel plane
    travels (render -> engine -> post -> chain), the panel rows, the
    export line, the tooltips, the presets and the feature-matrix rows."""
    import inspect
    src = inspect.getsource(PO.process)
    a = src.find("_cpu('lamp flares'")
    b = src.find("_gpu('matte_glow', SC.matte_glow, st, gel=gel)")
    c = src.find('if FILM.film_on(st):')
    check('the matte glow is wired on the linear frame after the lamp flares '
          'and before the film stages, its gel a KEYWORD (the closure hands '
          'it to the chain)', 0 <= a < b < c, f'{a} {b} {c}')
    d = src.find("_cpu('halftone'")
    e = src.find("_gpu('mpeg1', SC.mpeg1_intra, st, frame)")
    f = src.find("_gpu('smacker', SC.smacker, st, seed)")
    g = src.find('reduce_depth, st, seed)')
    check('MPEG-1 then Smacker are wired between the halftone and the quant '
          'block (a file is decoded BEFORE the display quantises it)',
          0 <= d < e < f < g, f'{d} {e} {f} {g}')
    sig = inspect.signature(PO.process)
    check('post.process takes the gel plane as the keyword `gel` (default '
          'None)', sig.parameters['gel'].default is None)

    eng = _src('engine.py')
    rnd = _src('core/render.py')
    check("the engine hands post the render's gel plane, skips the worker "
          'pool by name under Matte Glow, and a held frame re-exposes its '
          "key frame's plane",
          "gel=getattr(scene, 'last_gel', None))" in eng
          and 'worker pool skipped: Matte Glow' in eng
          and "scene.last_gel = _held.get('gel')" in eng
          and "_HOLD_CACHE['last']['gel']" in eng)
    check('render() resets the plane first and builds it beside last_depth '
          '(never for a pooled band or the viewport)',
          rnd.find('scene.last_gel = None       # R251 C134')
          < rnd.find('scene = _apply_material_override(scene, st)')
          and 'scene.last_gel = _SC.gel_plane(gbuf, job.scene.mesh,' in rnd
          and rnd.find('scene.last_depth = depth_m[::ss, ::ss]')
          < rnd.find('scene.last_gel = _SC.gel_plane('))
    ex = _src('export.py')
    i0 = ex.find('m.receive_shadow = hs.receive_shadow')
    i1 = ex.find('m.glow_gel = tuple(float(c) for c in')
    i2 = ex.find("m.volume_role = str(getattr(hs, 'volume_role', 'NONE'))")
    check('the exporter writes glow_gel with the surface flags every '
          'material carries (never inside the override branch)',
          0 <= i0 < i1 < i2, f'{i0} {i1} {i2}')
    ui = _src('ui.py')
    j0 = ui.find("row.prop(hs, 'receive_shadow')")
    j1 = ui.find("layout.prop(hs, 'glow_gel')")
    j2 = ui.find("layout.prop(hs, 'use_override')")
    check('the material panel draws the gel beside the shadow flags, above '
          'the override column (never greyed for a node material)',
          0 <= j0 < j1 < j2, f'{j0} {j1} {j2}')
    check('Optical Effects draws the matte rows; the JPEG panel draws the two '
          'codecs and greys its own rows while MPEG-1 is on',
          all(f"'{n}'" in ui for n in (
              'matte_glow', 'matte_glow_radius', 'matte_glow_passes',
              'matte_glow_exposure', 'mpeg1', 'mpeg1_qscale', 'mpeg1_gop',
              'smacker', 'smacker_quality'))
          and 'col.active = hs.jpeg_artifacts and not hs.mpeg1' in ui)
    pr = _src('properties.py')
    names = ('mpeg1', 'mpeg1_qscale', 'mpeg1_gop', 'smacker',
             'smacker_quality', 'matte_glow', 'matte_glow_radius',
             'matte_glow_passes', 'matte_glow_exposure')
    st = RenderSettings()
    check('the nine settings exist with their defaults (MPEG-1 off at scale '
          '8 in a group of 15; Smacker off at 0.5; the matte off at radius '
          '8, three passes, exposure 1)',
          (st.mpeg1, st.mpeg1_qscale, st.mpeg1_gop, st.smacker,
           st.smacker_quality, st.matte_glow, st.matte_glow_radius,
           st.matte_glow_passes, st.matte_glow_exposure)
          == (False, 8, 15, False, 0.5, False, 8.0, 3, 1.0))
    check('every one has a label and a tooltip in properties.py, and the '
          'material carries the Glow Gel colour',
          all(pr.count(f"'{n}': \"") >= 2 for n in names)
          and 'glow_gel: FloatVectorProperty(' in pr)
    from ..core.scene import Material
    check('scene.Material.glow_gel defaults to black (no matte)',
          tuple(Material(name='m', index=0).glow_gel) == (0.0, 0.0, 0.0))

    # ---- the presets
    want = {'VIDEO_CD': ('BROADCAST', 'mpeg1'),
            'SMACKER_FMV': ('WEB', 'smacker'),
            'TRON_1982': ('CEL', 'matte_glow')}
    for key, (cat, field) in want.items():
        p = PRESETS.get(key)
        check(f'the {key} preset sits on the {cat} shelf with a note and '
              f'switches {field} on',
              p is not None and p['category'] == cat and len(p['note']) > 40
              and p['settings'].get(field) is True)
    check('the round keeps at least its 106 presets here (the exact count is '
          "test_render's pin)", len(PRESETS) >= 106, str(len(PRESETS)))
    v = PRESETS['VIDEO_CD']['settings']
    s = PRESETS['SMACKER_FMV']['settings']
    t = PRESETS['TRON_1982']['settings']
    check('VIDEO_CD is 352x240 at 10:11 pixels, scale 10 in a group of 15; '
          'SMACKER_FMV an 8-bit adaptive 256 palette at quality 0.4; '
          'TRON_1982 radius 8, three passes',
          (v['resolution_x'], v['resolution_y'], v['pixel_aspect_x'],
           v['pixel_aspect_y'], v['mpeg1_qscale'], v['mpeg1_gop'])
          == (352, 240, 10.0, 11.0, 10, 15)
          and (s['color_depth'], s['palette_mode'], s['palette_size'],
               s['smacker_quality']) == ('8', 'ADAPTIVE', 256, 0.4)
          and (t['matte_glow_radius'], t['matte_glow_passes']) == (8.0, 3))
    check('CD_ROM_FMV keeps its JPEG road (Cinepak is not simulated)',
          PRESETS['CD_ROM_FMV']['settings'].get('jpeg_artifacts') is True
          and 'smacker' not in PRESETS['CD_ROM_FMV']['settings'])
    for key in want:
        stp = base_settings(W, H)
        apply_preset(stp, key)
        stp.resolution_x, stp.resolution_y = W, H
        stp.aa_samples = 1
        stp.output_scale = 'NONE'
        sc = demo_scene(stp, with_texture=False)
        _gel_ball(sc)
        PA.clear_caches()
        img = R.render(sc, stp)
        kw = dict(frame=1, seed=stp.seed, allow_resize=False,
                  target_size=(W, H), gel=getattr(sc, 'last_gel', None))
        on = PO.process(img, stp, **kw)
        off_st = stp.copy()
        setattr(off_st, want[key][1], False)
        PA.clear_caches()
        off = PO.process(img, off_st, **kw)
        check(f'the {key} preset renders at 96x72 and its stage moves the '
              'frame', on.shape == off.shape and bool(np.isfinite(on).all())
              and not np.array_equal(on, off))
    PA.clear_caches()

    # ---- the feature-matrix rows (the FALLBACK plumbing: with no driver
    # both devices run the CPU code; the twin proof is the gpu_road cases)
    from .featurematrix import ROWS, build
    from .test_render import _matrix_run
    keys = ('Tron matte glow (gel on the ball)',
            'MPEG-1 intra blocks, Video CD',
            'MPEG-1 B picture in a group of 15',
            'Smacker 4x4 blocks over a 256 palette')
    rows = {k: (o, s) for k, o, s in ROWS}
    check("every feature of the slot has its featurematrix row",
          all(k in rows for k in keys), str([k for k in keys if k not in rows]))
    base = None
    for k in keys:
        if k not in rows:
            continue
        o, scn = rows[k]
        PA.clear_caches()
        sc, stc = build(k)
        stc.render_device = 'CPU'
        cpu = _matrix_run(sc, stc)
        sc, stg = build(k)
        stg.render_device = 'GPU'
        gpu = _matrix_run(sc, stg)
        PA.clear_caches()
        _sc, st0 = build(k)
        for name in o:
            setattr(st0, name, getattr(RenderSettings(), name))
        from .featurematrix import SCENES
        base = _matrix_run(SCENES[scn](st0), st0)
        check(f"the row '{k}' renders on both devices bitwise and differs "
              'from the scene without it',
              cpu.shape == gpu.shape and bool(np.array_equal(cpu, gpu))
              and not np.array_equal(cpu, base))
    PA.clear_caches()

    # ---- the self test and the capability table
    sel = _src('selftest.py')
    check('the self test measures SMACKER as a single draw and MPEG1 / '
          'MATTE_GLOW through their orchestrators',
          "'SMACKER': (_CC.smacker_uniforms(st_sm, w, h)," in sel
          and 'chain.mpeg1(_fr, st_o, frame_no=1)' in sel
          and 'chain.matte_glow(_fr, st_o, gel=gel_img)' in sel
          and "gel=getattr(sc, 'last_gel', None)," in sel)
    from ..gpu import capability as CAP
    check('the capability table carries the two rows (signal_codec, '
          'matte_glow)', 'signal_codec' in CAP.FEATURES
          and 'matte_glow' in CAP.FEATURES)


def main():
    from . import utf8_console
    utf8_console()
    tests = sorted(k for k in globals() if k.startswith('test_'))
    for name in tests:
        print(f'-- {name}')
        try:
            globals()[name]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(name + ' (exception)')
    print(f'\n{len(FAILS)} failure(s)' if FAILS else '\nall post-codec checks passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
