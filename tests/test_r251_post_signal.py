"""R251 post-signal pack (SIG-1: C005, C019, C069, C033, C014, C047).

The machine's scan-out stages between the framebuffer and the glass:
the N64 Video Interface's dither filter and gamma, the GameCube copy
filter, the PS2 CRTC blend, the 3dfx "22-bit" scan-out filter, and the
two machine resamplers (3DO 2x cornerweight, GBA Mode 5 affine) after
the final readback. Every check reads as a sentence of what it proves.

    python -m halcyon.tests.test_r251_post_signal
"""

import contextlib
import importlib
import io
import math
import os
import sys
import traceback

import numpy as np

from ..core import post as PO
from ..core import render as R
from ..core import signal_era as SIG
from ..core import palette as PA
from ..core import wear as WEAR
from ..core.settings import RenderSettings
from ..core.texture import Texture
from ..gpu import chain
from ..gpu import chain_signal as CS
from ..gpu import frame as FR
from ..gpu import shade as GSH
from ..gpu import stages
from ..gpu import stages_signal
from ..presets.library import apply_preset, PRESETS
from ..shaders.compiler import try_compile
from . import fakedevice
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


def gpu_road(st, seed_frame=None):
    """Render + post through the fake device, then the CPU chain over the
    SAME frame (only the device switch differs): (out_g, out_c, rec, live).
    `seed_frame`: before EACH process, clear the CRTC state and run one
    CPU post of the same frame at that frame number (C033's road)."""
    st.render_device = 'GPU'
    sc = demo_scene(st, with_texture=False)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    st_c = st.copy()
    st_c.render_device = 'CPU'

    def _seed(img):
        if seed_frame is None:
            return
        SIG.CRTC_STATE.clear()
        kw = post_kw(sc, st_c)
        kw['frame'] = int(seed_frame)
        PO.process(img, st_c, **kw)

    with fakedevice.installed() as dev:
        st._keep_gpu_frame = True
        try:
            img = R.render(sc, st)
            _seed(img)
            out = PO.process(img, st, **post_kw(sc, st))
            rec = dict(PO.LAST_CHAIN)
        finally:
            FR.release(st)
        live = [t for t in dev.targets if not t.freed]
    _seed(img)
    out_c = PO.process(img, st_c, **post_kw(sc, st_c))
    return out, out_c, rec, live


def gpu_road_resize(st):
    """gpu_road with the output free to change shape (the two resamplers)."""
    st.render_device = 'GPU'
    sc = demo_scene(st, with_texture=False)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    kw = dict(frame=7, seed=st.seed, allow_resize=True,
              depth=getattr(sc, 'last_depth', None))
    with fakedevice.installed() as dev:
        st._keep_gpu_frame = True
        try:
            img = R.render(sc, st)
            out = PO.process(img, st, **kw)
            rec = dict(PO.LAST_CHAIN)
        finally:
            FR.release(st)
        live = [t for t in dev.targets if not t.freed]
    st_c = st.copy()
    st_c.render_device = 'CPU'
    out_c = PO.process(img, st_c, **kw)
    return out, out_c, rec, live


def run_stage(name, h, w, uniforms, samplers):
    """One stage through the GLSL simulator: (H, W, 4) or (None, err)."""
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


def _rec_ok(rec, live, d, want, tol=0.0):
    ok = d <= tol and live == []
    if 'resident' in want:
        ok &= rec.get('resident') is want['resident']
    if 'stages' in want:
        ok &= rec.get('stages') == want['stages']
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


def _demo_frame(**kw):
    """The demo frame after the quant block: render + post up to the
    depth, as a (H, W, 3) float32 array (the input every scan-out stage
    sees)."""
    st = settings(**kw)
    sc = demo_scene(st, with_texture=False)
    img = R.render(sc, st)
    rgb = np.asarray(img, np.float32)[:, :, :3]
    rgb = PO.display_transform(rgb, st)
    rgb = PO.reduce_depth(rgb, st, st.seed)
    return np.ascontiguousarray(rgb, np.float32), st


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
]


def _case_settings(kw):
    if kw == 'preset':
        st = base_settings(W, H)
        apply_preset(st, 'CEL_ANIME_MODERN')
        st.resolution_x, st.resolution_y = W, H
        st.aa_samples = 1
        return st
    return settings(**kw)


def test_r251_signal_defaults_are_invisible():
    """The pack pin: the 1.89.0 zip is beside the package and every
    default leaves every pixel of render AND post bitwise the previous
    release's; every `_on` predicate is False on a default settings
    object; every chain-stage function is the identity at defaults; the
    pack's stages merged before ENABLED was derived."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the neutrality pin runs)',
          RP is not None)
    if RP is not None:
        prev_post = importlib.import_module(
            RP.__name__.rsplit('.', 1)[0] + '.post')
        for label, kw in _CASES:
            st = _case_settings(kw)
            sc = demo_scene(st, with_texture=False)
            now = PO.process(R.render(sc, st), st, **post_kw(sc, st))
            st2 = _case_settings(kw)
            sc2 = demo_scene(st2, with_texture=False)
            prev = prev_post.process(RP.render(sc2, st2), st2,
                                     **post_kw(sc2, st2))
            same = now.shape == prev.shape and bool(np.array_equal(now, prev))
            check(f'the CPU device, {label}: render + post bitwise the '
                  "1.89.0 release's (the pack's defaults are invisible)",
                  same, f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    st = RenderSettings()
    preds = {'vi_filter': SIG.vi_filter_on, 'copy_filter': SIG.copy_filter_on,
             'crtc_blend': SIG.crtc_blend_on,
             'video_filter': SIG.video_filter_on}
    check('no scan-out predicate is on for a default settings object',
          not any(p(st) for p in preds.values()),
          str({k: p(st) for k, p in preds.items()}))
    rng = np.random.default_rng(251)
    img = rng.random((12, 16, 3)).astype(np.float32)
    fns = {'vi_filter': lambda: SIG.vi_filter(img, st),
           'copy_filter': lambda: SIG.copy_filter(img, st),
           'crtc_blend': lambda: SIG.crtc_blend(img, st, 3),
           'video_filter': lambda: SIG.video_filter(img, st)}
    for name, fn in fns.items():
        got = fn()
        check(f'signal_era.{name} on default settings returns its input '
              'bitwise (the opening guard)',
              got is img or bool(np.array_equal(got, img)))
    check('the CRTC state stays empty after a default post',
          not SIG.CRTC_STATE)
    check('the pack\'s stages merged into stages.STAGES before ENABLED was '
          'derived (a stage pasted after ENABLED would refuse silently)',
          set(stages_signal.STAGES_SIGNAL) <= set(stages.ENABLED)
          and set(stages_signal.STAGES_SIGNAL) <= set(stages.STAGES)
          and set(stages_signal.INTERFACE_SIGNAL) == set(
              stages_signal.STAGES_SIGNAL)
          and set(stages_signal.VALIDATION_SIGNAL) == set(
              stages_signal.STAGES_SIGNAL),
          str(sorted(set(stages_signal.STAGES_SIGNAL) - set(stages.ENABLED))))
    check('every pack stage is a chain function post.process finds by name',
          all(callable(getattr(chain, n, None))
              for n in ('vi_filter', 'copy_filter', 'crtc_blend',
                        'video_filter')))


def test_r251_signal_refusals_by_name():
    """Every `<name>_refusal(st)` is None when the stage may run and a
    string naming the reason otherwise (the size defaulting from st)."""
    on = {
        'vi_filter': (CS.vi_filter_refusal,
                      settings(color_depth='15', vi_gamma='GAMMA',
                               render_device='GPU')),
        'copy_filter': (CS.copy_filter_refusal,
                        settings(copy_filter='DEFLICKER', render_device='GPU')),
        'crtc_blend': (CS.crtc_blend_refusal,
                       settings(crtc_blend='BG_COLOR', render_device='GPU')),
        'video_filter': (CS.video_filter_refusal,
                         settings(color_depth='16', video_filter='VOODOO1',
                                  render_device='GPU')),
        'video_filter (VOODOO2)': (CS.video_filter_refusal,
                                   settings(color_depth='16',
                                            video_filter='VOODOO2',
                                            render_device='GPU')),
    }
    with fakedevice.installed():
        for name, (fn, st) in on.items():
            check(f'{name}_refusal is None when the stage may draw',
                  fn(st) is None, str(fn(st)))
        cpu = settings(copy_filter='DEFLICKER')
        check('copy_filter_refusal names the CPU device on the CPU device',
              'CPU' in str(CS.copy_filter_refusal(cpu)),
              str(CS.copy_filter_refusal(cpu)))
    check("vi_filter_refusal names the framebuffer at colour depth '24'",
          'framebuffer' in str(CS.vi_filter_refusal(
              settings(color_depth='24', vi_gamma='GAMMA'))))
    check("video_filter_refusal names 5:6:5 at colour depth '15'",
          '5:6:5' in str(CS.video_filter_refusal(
              settings(color_depth='15', video_filter='VOODOO1'))))


def test_r251_signal_featurematrix_rows():
    """Every row key this pack adds exists in featurematrix.ROWS and
    renders on both devices bitwise (the fallback plumbing; the twin
    proof is the gpu_road cases)."""
    from .featurematrix import ROWS, build
    from .test_render import _matrix_run
    keys = ('N64 VI dedither + gamma dither', 'GameCube copy-filter deflicker',
            'PS2 CRTC blend against BGCOLOR', '3dfx 22-bit scan-out filter',
            '3dfx Voodoo2 look-ahead filter',
            '3DO 2x interpolated, blue at 4 bits (fit_to keeps the sites)',
            'GBA Mode 5 affine stretch')
    names = {k for k, _o, _s in ROWS}
    check('every post-signal feature has its featurematrix row',
          all(k in names for k in keys), str([k for k in keys if k not in names]))
    for k in keys:
        if k not in names:
            continue
        sc, st = build(k)
        cpu = _matrix_run(sc, st)
        sc2, st2 = build(k)
        st2.render_device = 'GPU'
        gpu = _matrix_run(sc2, st2)
        sc3, st3 = build(k)
        for f in ('vi_dither_filter', 'vi_gamma', 'copy_filter', 'crtc_blend',
                  'video_filter', 'output_scale'):
            setattr(st3, f, getattr(RenderSettings(), f))
        off = _matrix_run(sc3, st3)
        check(f"row '{k}' falls back bit-exactly without a driver and moves "
              'the picture (not vacuous)',
              cpu.shape == gpu.shape and bool((cpu == gpu).all())
              and not (off.shape == cpu.shape and bool((off == cpu).all())))


# ---------------------------------------------------------------- C005 VI

def test_r251_c005_vi_filter():
    """The N64 Video Interface: the dedither law on a checkerboard, the
    gamma laws (the LUT, the 64-wide window under the root, the
    hardware's own silence at v = 100, the lift at v = 26 and 102),
    determinism, the GPU twin bitwise for six combinations at two depths,
    the resident record with and without a dither, the refusal by name."""
    st0 = RenderSettings()
    rng = np.random.default_rng(5)
    img = rng.random((12, 16, 3)).astype(np.float32)
    check('vi_filter_on is False on default settings and vi_filter returns '
          'its input',
          not SIG.vi_filter_on(st0) and SIG.vi_filter(img, st0) is img)

    # --- the dedither law: a checkerboard of levels k and k+1
    k = 9
    h, w = 48, 64
    yy, xx = np.mgrid[0:h, 0:w]
    lv = ((xx + yy) & 1)
    board = np.where(lv[:, :, None] == 1, (k + 1) / 31.0, k / 31.0).astype(np.float32)
    board = np.repeat(board, 3, axis=2)
    st = settings(color_depth='15', vi_dither_filter=True)
    out = SIG.vi_filter(board, st)
    inner = out[1:-1, 1:-1]
    want = SIG.U8[8 * k + 4]
    check('DITHER_FILTER on a 5:5:5 checkerboard of levels k, k+1 resolves '
          'every interior pixel to the level between (U8[8k + 4])',
          bool((inner == want).all()), f'{float(inner.min())}..{float(inner.max())} vs {want}')
    flat = np.full((h, w, 3), 20 / 31.0, np.float32)
    check('a flat 15-bit frame is bitwise unchanged by the dedither '
          '(equal neighbours contribute 0)',
          bool(np.array_equal(SIG.vi_filter(flat, st),
                              np.full((h, w, 3), SIG.U8[20 << 3], np.float32))))

    # --- the gamma laws. The VI reads the 5:6:5 FIELDS, so a flat frame
    # reaches the gamma stage at a level the expansion can produce: a
    # multiple of 8 on red / blue (k << 3), of 4 on green (k << 2). The
    # spec's v = 26 / 102 / 100 are not such levels (deviation, recorded);
    # the laws run at v = 56 (48 of 64 lift), 40 (23 of 64) and 96 (the
    # hardware's own silence: 78^2 = 6084 < 6144 and 79^2 = 6241 > 6207)
    def _flat565(v):
        f = np.empty((h, w, 3), np.float32)
        f[..., 0] = (v >> 3) / np.float32(31.0)
        f[..., 1] = (v >> 2) / np.float32(63.0)
        f[..., 2] = (v >> 3) / np.float32(31.0)
        return f

    lut = SIG.VI_GAMMA_LUT
    check('VI_GAMMA_LUT is non-decreasing and equals 2*isqrt(64 i) at every i',
          bool((np.diff(lut) >= 0).all())
          and all(int(lut[i]) == 2 * math.isqrt(64 * i) for i in range(256)))
    for v, lifts in ((56, 48), (40, 23), (96, 0)):
        flat = _flat565(v)
        st = settings(color_depth='16', vi_gamma='GAMMA_DITHER')
        out = SIG.to_u8(SIG.vi_filter(flat, st, 7, 3))
        lo = int(lut[v])
        hi = 2 * (math.isqrt(64 * v + 63) - math.isqrt(64 * v))
        d = out - lo
        window = 2 * np.array([math.isqrt(64 * v + j) for j in range(64)])
        n_lift = int((window > 2 * math.isqrt(64 * v)).sum())
        check(f'GAMMA_DITHER at v = {v}: every byte is within the 64-wide '
              f'window under the root (0..{hi}) and exactly {lifts} of 64 hash '
              'values lift it',
              bool((d >= 0).all()) and bool((d <= hi).all()) and n_lift == lifts
              and ((float(out.mean()) > lo) if lifts else bool((d == 0).all())),
              f'mean {float(out.mean()):.3f} vs {lo}; window lifts {n_lift}')
    flat = _flat565(56)
    st = settings(color_depth='16', vi_gamma='GAMMA_DITHER')
    a = SIG.vi_filter(flat, st, 7, 3)
    b = SIG.vi_filter(flat, st, 7, 3)
    c = SIG.vi_filter(flat, st, 8, 3)
    check('GAMMA_DITHER is deterministic per (frame, seed) and frame + 1 differs',
          bool(np.array_equal(a, b)) and not bool(np.array_equal(a, c)))
    flat = _flat565(64)
    st = settings(color_depth='16', vi_gamma='GAMMA')
    check('GAMMA on a flat v = 64 frame gives 2*isqrt(4096) = 128 everywhere',
          bool((SIG.to_u8(SIG.vi_filter(flat, st)) == 128).all()))
    flat = _flat565(96)
    st = settings(color_depth='16', vi_gamma='DITHER_ONLY')
    out = SIG.to_u8(SIG.vi_filter(flat, st, 7, 3))
    check('DITHER_ONLY adds one hashed bit to each channel (bytes 96 or 97, '
          'both present)',
          set(np.unique(out).tolist()) == {96, 97})

    # --- the GPU twin through the simulator
    rgb, st_d = _demo_frame(color_depth='15', dither='BAYER2')
    src = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], 2)
    for depth in ('15', '16'):
        bits = PO.DEPTH_BITS[depth]
        if depth == '16':
            rgb = PA.snap_bits(rgb, 5, 6, 5)
            src = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], 2)
        for dedither in (True, False):
            for mode, gm in (('GAMMA', 1), ('GAMMA_DITHER', 2),
                             ('DITHER_ONLY', 3)):
                st = settings(color_depth=depth, vi_dither_filter=dedither,
                              vi_gamma=mode)
                glut = {1: SIG.vi_gamma_image(), 2: SIG.vi_gdither_image(),
                        3: SIG.u8_image()}[gm]
                got, err = run_stage('VI', H, W, {
                    'resolution': (float(W), float(H)),
                    'levels': tuple(float((1 << b) - 1) for b in bits),
                    'shift_r': 8 - bits[0], 'shift_g': 8 - bits[1],
                    'shift_b': 8 - bits[2], 'dedither': 1 if dedither else 0,
                    'gamma_mode': gm,
                    'key': int(WEAR._sheet_key(7, 3, 41))},
                    {'source': src, 'u8lut': SIG.u8_image(), 'glut': glut})
                ref = SIG.vi_filter(rgb, st, 7, 3)
                d = float(np.abs(got[..., :3] - ref).max()) if got is not None else -1.0
                check(f"VI stage at depth '{depth}', dedither {dedither}, {mode}: "
                      'bitwise signal_era.vi_filter (d == 0.0), alpha through',
                      got is not None and d == 0.0
                      and bool((got[..., 3] == 1.0).all()),
                      str(err) if got is None else f'max {d}')

    # --- the resident record through the fake device
    out_g, out_c, rec, live = gpu_road(settings(
        color_depth='15', dither='NONE', vi_dither_filter=True,
        vi_gamma='GAMMA_DITHER'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check("VI on the resident frame at '15' with no dither: the GPU chain "
          'equals the CPU chain bitwise, VI drawn, 0 readbacks, nothing live',
          _rec_ok(rec, live, d, dict(stages_has=['VI'], readbacks=0)),
          f'max {d}; {rec}; {len(live)} live')
    out_g, out_c, rec, live = gpu_road(settings(
        color_depth='15', dither='BAYER2', vi_dither_filter=True,
        vi_gamma='GAMMA_DITHER'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    # PAL-1's P1: the BAYER dither is the ORDERED draw, so nothing reads
    # back before the VI and the render's frame is inherited (no upload)
    check("VI on the resident frame at '15' with BAYER2: bitwise, the dither "
          'drawn on the ORDERED stage (no readback), VI drawn after it, the '
          "render's frame inherited (no upload), nothing live",
          _rec_ok(rec, live, d, dict(resident=True,
                                     stages=['DISPLAY', 'ORDERED', 'VI'],
                                     readbacks=0, uploads=0)),
          f'max {d}; {rec}; {len(live)} live')

    # --- the refusal by name
    st = settings(color_depth='24', vi_dither_filter=True)
    why = SIG.vi_filter_why(st)
    check("vi_filter_why at colour depth '24' names the framebuffer",
          why is not None and 'framebuffer' in why, str(why))
    got, text = _capture(lambda: SIG.vi_filter_on(st))
    check("vi_filter_on at '24' is False and prints the reason once",
          got is False and 'framebuffer' in text, text.strip())
    frame = rng.random((h, w, 3)).astype(np.float32)
    check("vi_filter at '24' leaves the frame bitwise (a named no-op)",
          bool(np.array_equal(SIG.vi_filter(frame, st), frame)))
    with fakedevice.installed():
        check("CS.vi_filter_refusal is None at '15' with GAMMA on the GPU device",
              CS.vi_filter_refusal(settings(color_depth='15', vi_gamma='GAMMA',
                                            render_device='GPU')) is None)
    p = PRESETS['N64']['settings']
    check('the N64 preset runs the VI de-dither and gamma-dither on a 15-bit '
          'frame with gamma 1.0 (the VI root replaces the display dial)',
          p.get('vi_dither_filter') is True and p.get('vi_gamma') == 'GAMMA_DITHER'
          and p.get('color_depth') == '15' and p.get('gamma') == 1.0)


# ------------------------------------------------------- C019 COPY_FILTER

def test_r251_c019_copy_filter():
    """The GameCube / Wii copy filter: the blur law on one bright row
    (rows above and below at 16/64, the row at 32/64, floored), the AA
    set, a flat frame unchanged (the taps sum to 64), the clamp at the
    top and bottom rows, the overflow law, the GPU twin bitwise for both
    sets, the resident record, the refusal by name through a probe that
    fails."""
    st0 = RenderSettings()
    rng = np.random.default_rng(19)
    img = rng.random((12, 16, 3)).astype(np.float32)
    check('copy_filter_on is False on default settings and copy_filter '
          'returns its input',
          not SIG.copy_filter_on(st0) and SIG.copy_filter(img, st0) is img)
    h, w = 24, 32
    one = np.zeros((h, w, 3), np.float32)
    one[10] = SIG.U8[200]
    for mode, side, mid in (('DEFLICKER', 50, 100), ('DEFLICKER_AA', 37, 125)):
        st = settings(copy_filter=mode)
        out = SIG.to_u8(SIG.copy_filter(one, st))
        check(f'{mode} on one bright row (200 at row 10): rows 9 and 11 are '
              f'exactly {side} and row 10 is {mid} (the 1/64 divide floored)',
              bool((out[9] == side).all()) and bool((out[11] == side).all())
              and bool((out[10] == mid).all()) and bool((out[12] == 0).all())
              and bool((out[8] == 0).all()),
              f'{out[9, 0, 0]} {out[10, 0, 0]} {out[11, 0, 0]}')
        flat = np.full((h, w, 3), SIG.U8[77], np.float32)
        check(f'{mode} leaves a flat frame bitwise unchanged, top and bottom '
              'rows included (the taps sum to 64; GXSetCopyClamp)',
              bool(np.array_equal(SIG.copy_filter(flat, st), flat)))
        white = np.ones((h, w, 3), np.float32)
        check(f'{mode} on a flat 255 frame stays 255 (the min clamp)',
              bool(np.array_equal(SIG.copy_filter(white, st), white)))

    # --- the GPU twin
    rgb, _st = _demo_frame()
    src = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], 2)
    for mode in ('DEFLICKER', 'DEFLICKER_AA'):
        U, M, L = SIG.COPY_TAPS[mode]
        got, err = run_stage('COPY_FILTER', H, W, {
            'resolution': (float(W), float(H)),
            'tap_u': U, 'tap_m': M, 'tap_l': L},
            {'source': src, 'u8lut': SIG.u8_image()})
        ref = SIG.copy_filter(rgb, settings(copy_filter=mode))
        d = float(np.abs(got[..., :3] - ref).max()) if got is not None else -1.0
        moved = float(np.abs(ref - rgb).max())
        check(f'COPY_FILTER stage, {mode}: bitwise signal_era.copy_filter on '
              'the demo frame (d == 0.0), alpha through, not vacuous',
              got is not None and d == 0.0 and moved > 0.0
              and bool((got[..., 3] == 1.0).all()),
              str(err) if got is None else f'max {d}; moved {moved:.4f}')
    out_g, out_c, rec, live = gpu_road(settings(dither='NONE',
                                                copy_filter='DEFLICKER'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check('COPY_FILTER on the resident frame (no dither): the GPU chain equals '
          'the CPU chain bitwise, COPY_FILTER drawn, 0 readbacks, nothing live',
          _rec_ok(rec, live, d, dict(stages_has=['COPY_FILTER'], readbacks=0)),
          f'max {d}; {rec}; {len(live)} live')

    # --- the refusal by name: a probe that fails
    with fakedevice.installed():
        check('copy_filter_refusal is None when the stage may draw',
              CS.copy_filter_refusal(settings(copy_filter='DEFLICKER',
                                              render_device='GPU')) is None)
    st = settings(copy_filter='DEFLICKER')
    sc = demo_scene(st, with_texture=False)
    img = R.render(sc, st)
    st_c = st.copy()
    ref = PO.process(img, st_c, **post_kw(sc, st_c))
    st.render_device = 'GPU'
    from ..gpu import device as DEV
    with fakedevice.installed():
        DEV.probe = lambda: (False, 'no driver')     # restored by the context
        st._frame_gpu_shaded = True
        fr = chain.Frame(rgb=np.asarray(img, np.float32)[:, :, :3])
        got, text = _capture(lambda: chain.copy_filter(fr, st))
        fr.release()
    check('a device whose probe fails makes the chain print '
          "'COPY_FILTER on the CPU: no driver' and return None",
          got is None and 'COPY_FILTER on the CPU: no driver' in text,
          text.strip())
    st.render_device = 'GPU'
    with fakedevice.installed():
        DEV.probe = lambda: (False, 'no driver')
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        st._keep_gpu_frame = True
        try:
            out = PO.process(R.render(sc, st), st, **post_kw(sc, st))
        finally:
            FR.release(st)
    check('and the CPU road then renders the same picture bitwise',
          out.shape == ref.shape and bool(np.array_equal(out, ref)))
    p = PRESETS['GAMECUBE']['settings']
    check('the GAMECUBE preset runs the deflicker copy filter with no edge AA',
          p.get('copy_filter') == 'DEFLICKER' and p.get('aa_mode') == 'NONE'
          and 'aa_edge_threshold' not in p)


# ------------------------------------------------------ C069 VIDEO_FILTER

def test_r251_c069_video_filter():
    """The 3dfx scan-out filter: the dissolve law on a dithered grey
    (both items), the mean laws, the edge law and the floor law (v1),
    the Voodoo2 laws (the capped edge, the one-column step, the lift
    only), threshold 0 = the expansion, a flat frame = its expansion,
    the GPU twins (one VOODOO_LINE draw, the four-draw orchestration,
    VOODOO_LOOK), the refusals by name, the VOODOO preset."""
    st0 = RenderSettings()
    rng = np.random.default_rng(69)
    img = rng.random((12, 16, 3)).astype(np.float32)
    check('video_filter_on is False on default settings and video_filter '
          'returns its input',
          not SIG.video_filter_on(st0) and SIG.video_filter(img, st0) is img)

    # --- the dissolve law on a 565 BAYER4 flat 0.5 grey
    h, w = 24, 64
    grey = np.full((h, w, 3), 0.5, np.float32)
    from ..core import dither as DI
    dith = DI.ordered_bits(grey, (5, 6, 5), 'BAYER4', 1.0).astype(np.float32)
    dith = PA.snap_bits(dith, 5, 6, 5)
    base = SIG.expand565(dith)
    tv0 = np.abs(np.diff(base, axis=1)).sum()
    for item in ('VOODOO1', 'VOODOO2'):
        st = settings(color_depth='16', video_filter=item)
        out = SIG.to_u8(SIG.video_filter(dith, st))
        tv = np.abs(np.diff(out, axis=1)).sum()
        rows_ok = all(np.abs(np.diff(out[y], axis=0)).sum()
                      < np.abs(np.diff(base[y], axis=0)).sum()
                      for y in range(1, h - 1))
        check(f'{item} on a 565 BAYER4 grey dissolves the dither: the sum of '
              '|p[x] - p[x-1]| along every interior row is strictly smaller',
              rows_ok and tv < tv0, f'{int(tv)} < {int(tv0)}')
        if item == 'VOODOO1':
            check('VOODOO1 keeps the row mean within 1 of the expansion\'s '
                  '(the passes halve differences, they do not shift them)',
                  abs(float(out.mean()) - float(base.mean())) <= 1.0,
                  f'{float(out.mean()):.3f} vs {float(base.mean()):.3f}')
        else:
            check('VOODOO2 never lowers the row mean (the rule only lifts)',
                  float(out.mean()) >= float(base.mean()),
                  f'{float(out.mean()):.3f} vs {float(base.mean()):.3f}')

    # --- the v1 edge law and floor law
    xe = 20
    edge = np.zeros((4, w, 3), np.int32)
    edge[:, xe:] = 255
    one = SIG._voodoo_pass(edge, -1, 64)
    check('one left pass over a 0 | 255 edge at cap 64 moves the bright column '
          'to exactly 255 - 32 = 223 and leaves the dark column still',
          bool((one[:, xe] == 223).all()) and bool((one[:, xe - 1] == 0).all()))
    p = edge
    for direction in (-1, -1, -1, 1):
        p = SIG._voodoo_pass(p, direction, 64)
    check('after the four passes both edge columns moved toward each other, '
          'each by at most cap/2 per pass, and the edge did not cross',
          bool((p[:, xe - 1] > 0).all()) and bool((p[:, xe] < 255).all())
          and bool((p[:, xe - 1] < p[:, xe]).all())
          and bool((p[:, xe - 1] <= 4 * 32).all()),
          f'{p[0, xe - 1, 0]} | {p[0, xe, 0]}')
    pair = np.array([[[7, 7, 7], [10, 10, 10]]], np.int32)
    check('the floor law: a pixel 10 with a left neighbour 7 gives '
          '10 + (-3 >> 1) = 8 after one pass',
          int(SIG._voodoo_pass(pair, -1, 64)[0, 1, 0]) == 8)

    # --- the Voodoo2 laws
    check('VOODOO2 leaves a 0 -> 255 edge at cap 64 bitwise (beyond the cap '
          'nothing moves)',
          bool(np.array_equal(SIG._voodoo2_line(edge, 64), edge)))
    step = np.zeros((4, w, 3), np.int32)
    step[:, xe:] = 41                       # k = 5 expanded
    got = SIG._voodoo2_line(step, 64)
    diff = np.argwhere(got[0, :, 0] != step[0, :, 0]).ravel().tolist()
    check('VOODOO2 on a 0 | 41 step changes ONLY the column left of the step, '
          'to _voodoo2_step(0, 41) = 24, and never moves the bright side down',
          diff == [xe - 1] and int(got[0, xe - 1, 0]) == 24
          and bool((got[:, xe:] >= 41).all()), f'{diff} -> {got[0, xe - 1, 0]}')
    check('_voodoo2_step(0, 41) is min((0 + 164)//5 - (0 + 41)//5, 32) = 24',
          int(SIG._voodoo2_step(np.int32(0), np.int32(41), 64, 32)) == 24)

    for item in ('VOODOO1', 'VOODOO2'):
        st = settings(color_depth='16', video_filter=item,
                      video_filter_threshold=0)
        check(f'{item} with threshold 0 returns exactly the 565 expansion',
              bool(np.array_equal(SIG.video_filter(dith, st),
                                  SIG.U8[SIG.expand565(dith)])))
        st = settings(color_depth='16', video_filter=item)
        flat = PA.snap_bits(np.full((h, w, 3), 0.37, np.float32), 5, 6, 5)
        check(f'{item} on a flat frame is bitwise its own expansion',
              bool(np.array_equal(SIG.video_filter(flat, st),
                                  SIG.U8[SIG.expand565(flat)])))

    # --- the GPU twins
    rgb, _st = _demo_frame(color_depth='16', dither='BAYER4')
    src = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], 2)
    got, err = run_stage('VOODOO_LINE', H, W, {
        'resolution': (float(W), float(H)), 'direction': -1, 'cap': 64,
        'expand': 1}, {'source': src, 'u8lut': SIG.u8_image()})
    ref = SIG.U8[np.clip(SIG._voodoo_pass(SIG.expand565(rgb), -1, 64), 0, 255)]
    d = float(np.abs(got[..., :3] - ref).max()) if got is not None else -1.0
    check('one VOODOO_LINE draw (left, cap 64, expanding) is bitwise '
          '_voodoo_pass over expand565 (d == 0.0), alpha through',
          got is not None and d == 0.0 and bool((got[..., 3] == 1.0).all()),
          str(err) if got is None else f'max {d}')
    out_g, out_c, rec, live = gpu_road(settings(
        color_depth='16', dither='BAYER4', video_filter='VOODOO1'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    # PAL-1's P1: the BAYER4 dither is the ORDERED draw, so the four-draw
    # filter follows it on the GPU with no readback and no upload
    check('VOODOO1 on the resident frame: the four-draw orchestration equals '
          'the CPU chain bitwise, recorded as VIDEO_FILTER after the BAYER4 '
          "dither's ORDERED draw (no readback), the render's frame inherited "
          '(no upload), nothing live',
          _rec_ok(rec, live, d, dict(
              resident=True, stages=['DISPLAY', 'ORDERED', 'VIDEO_FILTER'],
              readbacks=0, uploads=0)),
          f'max {d}; {rec}; {len(live)} live')
    got, err = run_stage('VOODOO_LOOK', H, W, {
        'resolution': (float(W), float(H)), 'cap': 64, 'cap32': 32,
        'inv5': float(np.float32(0.2))},
        {'source': src, 'u8lut': SIG.u8_image()})
    ref = SIG.video_filter(rgb, settings(color_depth='16',
                                         video_filter='VOODOO2'))
    d = float(np.abs(got[..., :3] - ref).max()) if got is not None else -1.0
    check('the VOODOO_LOOK stage is bitwise signal_era.video_filter under '
          'VOODOO2 on the BAYER4 frame (d == 0.0), alpha through',
          got is not None and d == 0.0 and bool((got[..., 3] == 1.0).all()),
          str(err) if got is None else f'max {d}')
    out_g, out_c, rec, live = gpu_road(settings(
        color_depth='16', dither='BAYER4', video_filter='VOODOO2'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check('VOODOO2 on the resident frame: bitwise, VOODOO_LOOK drawn after '
          "the BAYER4 dither's ORDERED draw (no readback), the render's frame "
          'inherited (no upload), nothing live',
          _rec_ok(rec, live, d, dict(
              resident=True, stages=['DISPLAY', 'ORDERED', 'VOODOO_LOOK'],
              readbacks=0, uploads=0)),
          f'max {d}; {rec}; {len(live)} live')

    # --- the refusals by name
    why = SIG.video_filter_why(settings(color_depth='15'))
    check("video_filter_why at colour depth '15' names 5:6:5",
          why is not None and '5:6:5' in why, str(why))
    st = settings(color_depth='24', video_filter='VOODOO1')
    got, text = _capture(lambda: SIG.video_filter_on(st))
    check("video_filter_on at '24' with VOODOO1 is False and prints the reason",
          got is False and '5:6:5' in text, text.strip())
    frame = rng.random((h, w, 3)).astype(np.float32)
    check("video_filter at '24' leaves the frame bitwise (a named no-op)",
          bool(np.array_equal(SIG.video_filter(frame, st), frame)))
    with fakedevice.installed():
        for item in ('VOODOO1', 'VOODOO2'):
            check(f"CS.video_filter_refusal is None at '16' under {item}",
                  CS.video_filter_refusal(settings(
                      color_depth='16', video_filter=item,
                      render_device='GPU')) is None)
    p = PRESETS['VOODOO']['settings']
    check("the VOODOO preset runs the Voodoo Graphics scan-out filter on its "
          "16-bit frame",
          p.get('video_filter') == 'VOODOO1' and p.get('color_depth') == '16')


# ------------------------------------------------------- C033 CRTC_BLEND

def test_r251_c033_crtc_blend():
    """The PS2 CRTC PMODE mix: the mix law against BGCOLOR (A = 160, 255,
    0), the persistence law over frames 1..4 (monotone convergence, the
    predecessor named, a re-render of frame 4 says so), the viewport
    law, the key law (alpha, scene, size), the device law (a CPU
    predecessor serves the GPU pass, routing toggles out of the
    fingerprint), the GPU twin bitwise in both modes, the resident
    records (seeded PREVIOUS_FRAME; stateless BG_COLOR), the refusal."""
    st0 = RenderSettings()
    rng = np.random.default_rng(33)
    img = rng.random((12, 16, 3)).astype(np.float32)
    check('crtc_blend_on is False on default settings and crtc_blend returns '
          'its input',
          not SIG.crtc_blend_on(st0) and SIG.crtc_blend(img, st0, 1) is img)

    # --- the mix law against BGCOLOR
    SIG.CRTC_STATE.clear()
    rgb, _st = _demo_frame()
    cur8 = SIG.to_u8(rgb)
    bg8 = SIG.to_u8(np.asarray((0.1, 0.0, 0.2), np.float32)).reshape(1, 1, 3)
    st = settings(crtc_blend='BG_COLOR', crtc_alpha=160,
                  crtc_bg_color=(0.1, 0.0, 0.2))
    out8 = SIG.to_u8(SIG.crtc_blend(rgb, st, 1))
    want = (cur8 * 160 + bg8 * 95) // 255
    check('BG_COLOR at A = 160 gives (cur8*160 + bg8*95) // 255 at every byte',
          bool(np.array_equal(out8, want)))
    st.crtc_alpha = 255
    check('A = 255 shows only the current frame (U8[to_u8(rgb)] bitwise)',
          bool(np.array_equal(SIG.crtc_blend(rgb, st, 1), SIG.U8[cur8])))
    st.crtc_alpha = 0
    check('A = 0 shows only BGCOLOR bitwise',
          bool(np.array_equal(SIG.to_u8(SIG.crtc_blend(rgb, st, 1)),
                              np.broadcast_to(bg8, cur8.shape))))
    check('the stateless BG_COLOR mode leaves the CRTC state empty',
          not SIG.CRTC_STATE)

    # --- the persistence law
    SIG.CRTC_STATE.clear()
    st = settings(crtc_blend='PREVIOUS_FRAME', crtc_alpha=128,
                  crtc_bg_color=(0.0, 0.0, 0.0))
    st._scene_name = 'pin'
    h, w = rgb.shape[:2]
    _p, why = SIG.crtc_rc2(st, 1, h, w)
    check("frame 1 has no predecessor: crtc_rc2 names it and RC2 reads BGCOLOR",
          why is not None and 'predecessor' in why, str(why))
    outs = []
    for f in (1, 2, 3, 4):
        outs.append(SIG.to_u8(SIG.crtc_blend(rgb, st, f)))
    dist = [int(np.abs(o - cur8).astype(np.int64).sum()) for o in outs]
    check('the same static frame at frames 1..4 converges monotonically toward '
          'itself and frame 2 differs from frame 1',
          dist[1] <= dist[0] and dist[2] <= dist[1] and dist[3] <= dist[2]
          and not bool(np.array_equal(outs[0], outs[1])), str(dist))
    check('after frame 3, frame 4 finds its predecessor (RC2 is the CRTC output)',
          SIG.CRTC_STATE.get('frame') == 4 and SIG.crtc_rc2(st, 5, h, w)[1] is None)
    _p, why = SIG.crtc_rc2(st, 4, h, w)
    check('rendering frame 4 again after frame 4 finds no predecessor and says so',
          why is not None and 'predecessor' in why)
    check('CS.crtc_state reports the kept frame and its key',
          CS.crtc_state() is not None and CS.crtc_state()[0] == 4)
    # the viewport law
    before = dict(SIG.CRTC_STATE)
    stv = st.copy()
    stv._scene_name = 'pin'
    stv._viewport = True
    outv = SIG.to_u8(SIG.crtc_blend(rgb, stv, 5))
    check('a _viewport settings copy neither reads nor writes the CRTC state '
          '(it blends against BGCOLOR)',
          SIG.CRTC_STATE == before
          and bool(np.array_equal(outv, (cur8 * 128) // 255)))
    # the key law
    SIG.CRTC_STATE.clear()
    SIG.crtc_blend(rgb, st, 3)
    st_a = st.copy(); st_a._scene_name = 'pin'; st_a.crtc_alpha = 200
    st_s = st.copy(); st_s._scene_name = 'other'
    check('after frame 3, frame 4 with crtc_alpha changed blends against '
          'BGCOLOR and names the predecessor',
          'predecessor' in str(SIG.crtc_rc2(st_a, 4, h, w)[1]))
    check('after frame 3, frame 4 under another scene name names the predecessor',
          'predecessor' in str(SIG.crtc_rc2(st_s, 4, h, w)[1]))
    check('after frame 3, frame 4 at another size names the predecessor',
          'predecessor' in str(SIG.crtc_rc2(st, 4, h + 1, w)[1]))
    check('and frame 4 with the same key finds it',
          SIG.crtc_rc2(st, 4, h, w)[1] is None)
    # the device law
    SIG.CRTC_STATE.clear()
    st_cpu = st.copy(); st_cpu._scene_name = 'pin'; st_cpu.render_device = 'CPU'
    SIG.crtc_blend(rgb, st_cpu, 6)
    st_gpu = st.copy(); st_gpu._scene_name = 'pin'; st_gpu.render_device = 'GPU'
    check("a frame 6 rendered on the CPU device is frame 7's predecessor on "
          'the GPU device (render_device is out of the fingerprint)',
          SIG.crtc_rc2(st_gpu, 7, h, w)[1] is None)
    st_gpu.gpu_post = not st_gpu.gpu_post
    st_gpu.gpu_scissor = not st_gpu.gpu_scissor
    check('and still is with gpu_post and gpu_scissor flipped (routing '
          'toggles are out of the fingerprint)',
          SIG.crtc_rc2(st_gpu, 7, h, w)[1] is None)
    st_gpu.dither = 'BAYER4'
    check('but not with a PICTURE dial changed (the dither)',
          'predecessor' in str(SIG.crtc_rc2(st_gpu, 7, h, w)[1]))

    # --- the GPU twin through the simulator, both modes
    src = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], 2)
    prev = rng.random((h, w, 3)).astype(np.float32)
    prev8 = SIG.to_u8(prev)
    prev_img = np.concatenate([SIG.U8[prev8], np.ones((h, w, 1), np.float32)], 2)
    for mode, label in ((0, 'PREVIOUS_FRAME (a random prev image)'),
                        (1, 'BG_COLOR')):
        got, err = run_stage('CRTC_BLEND', H, W, {
            'resolution': (float(W), float(H)),
            'bg': tuple(float(SIG.U8[int(v)]) for v in bg8[0, 0]),
            'alpha': 160, 'mode': mode,
            'inv255': float(np.float32(1.0 / 255.0))},
            {'source': src, 'u8lut': SIG.u8_image(),
             'prev': prev_img if mode == 0 else SIG.u8_image()})
        other8 = prev8 if mode == 0 else np.broadcast_to(bg8, cur8.shape)
        ref = SIG.U8[(cur8 * 160 + other8 * 95) // 255]
        d = float(np.abs(got[..., :3] - ref).max()) if got is not None else -1.0
        check(f'the CRTC_BLEND stage in {label} mode is bitwise the CPU mix '
              '(d == 0.0; hal_idiv exact), alpha through',
              got is not None and d == 0.0 and bool((got[..., 3] == 1.0).all()),
              str(err) if got is None else f'max {d}')

    # --- the resident records
    SIG.CRTC_STATE.clear()
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', crtc_blend='PREVIOUS_FRAME', crtc_alpha=128), seed_frame=6)
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check('PREVIOUS_FRAME on the resident frame (frame 6 seeded on the CPU before '
          'each pass): the GPU chain equals the CPU chain bitwise, CRTC_BLEND '
          'drawn and read back by name, nothing live',
          _rec_ok(rec, live, d, dict(stages_has=['CRTC_BLEND'],
                                     readback_names=['CRTC_BLEND'])),
          f'max {d}; {rec}; {len(live)} live')
    SIG.CRTC_STATE.clear()
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', crtc_blend='BG_COLOR', crtc_alpha=160,
        crtc_bg_color=(0.1, 0.0, 0.2)))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check('BG_COLOR on the resident frame (no dither): bitwise, CRTC_BLEND drawn, '
          '0 readbacks (the frame stays resident), nothing live',
          _rec_ok(rec, live, d, dict(stages_has=['CRTC_BLEND'], readbacks=0)),
          f'max {d}; {rec}; {len(live)} live')
    SIG.CRTC_STATE.clear()
    with fakedevice.installed():
        check('CS.crtc_blend_refusal is None under BG_COLOR on the GPU device',
              CS.crtc_blend_refusal(settings(crtc_blend='BG_COLOR',
                                             render_device='GPU')) is None)
    check("the PS2 preset leaves the CRTC blend off and its note names it",
          PRESETS['PS2']['settings'].get('crtc_blend', 'NONE') == 'NONE'
          and 'CRTC' in PRESETS['PS2']['note'])


# ------------------------------------------------------- C014 THREEDO_2X

def test_r251_c014_threedo_2x():
    """The 3DO cornerweight doubling: the shape and lattice laws on a
    15-bit BAYER2 demo frame (sites, the floor-average between, blue's
    LSB cleared), a flat frame flat, the picture differs from nearest
    2x, both roads the same shape and bitwise, the debug-pass road, the
    named no-op text, the THREEDO preset."""
    rgb, st = _demo_frame(color_depth='15', dither='BAYER2')
    st.output_scale = 'THREEDO_2X'
    out = SIG.threedo_2x(rgb, st)
    h, w = rgb.shape[:2]
    check('the output is exactly (2H, 2W, 3) float32',
          out.shape == (2 * h, 2 * w, 3) and out.dtype == np.float32)
    k = np.stack([SIG.to_bits(rgb[..., c], 5) for c in range(3)], 2)
    k[..., 2] &= ~1
    sites = np.stack([SIG.to_bits(out[0::2, 0::2, c], 5) for c in range(3)], 2)
    check("out[0::2, 0::2] equals the source with blue's LSB cleared",
          bool(np.array_equal(sites, k)))
    right = k[:, np.minimum(np.arange(w) + 1, w - 1)]
    between = np.stack([SIG.to_bits(out[0::2, 1::2, c], 5) for c in range(3)], 2)
    check('out[0::2, 1::2] at column 2x+1 is the floor-average of source '
          'columns x and x+1 (the last column repeats itself)',
          bool(np.array_equal(between, (k + right) >> 1)))
    flat = PA.snap_bits(np.full((h, w, 3), (0.4, 0.6, 0.7), np.float32), 5, 5, 5)
    fo = SIG.threedo_2x(flat, st)
    fk = np.stack([SIG.to_bits(flat[..., c], 5) for c in range(3)], 2)
    fk[..., 2] &= ~1
    check('a flat 15-bit frame stays flat and bitwise its own value',
          bool((fo == fo[0, 0]).all())
          and bool(np.array_equal(np.stack([SIG.to_bits(fo[..., c], 5)
                                            for c in range(3)], 2)[0, 0],
                                  fk[0, 0])))
    near = np.repeat(np.repeat(rgb, 2, axis=0), 2, axis=1)
    frac = float((np.abs(out - near).max(axis=2) > 1e-6).mean())
    check('the picture differs from nearest 2x at more than 10% of pixels',
          frac > 0.10, f'{frac:.3f}')
    out_g, out_c, rec, live = gpu_road_resize(settings(output_scale='THREEDO_2X',
                                                       color_depth='15'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check('both roads give 144x192 (RGBA: process keeps the alpha plane) and '
          'the same picture bitwise (the doubling runs after the final '
          'readback on both)',
          out_g.shape[:2] == (2 * H, 2 * W) and out_c.shape == out_g.shape
          and d == 0.0 and live == [], f'{out_g.shape} {out_c.shape} max {d}')
    st_d = settings(output_scale='THREEDO_2X', color_depth='15',
                    debug_pass='DEPTH')
    ramp = np.linspace(0.0, 1.0, W * H, dtype=np.float32).reshape(H, W)
    data = np.stack([ramp, ramp, ramp, np.ones((H, W), np.float32)], 2)
    got = PO.process(data, st_d, frame=1, seed=0, allow_resize=True)
    vals = set(np.unique(got[..., 0]).tolist())
    check('a DEPTH pass under THREEDO_2X takes nearest 2x: the shape doubles '
          'and every value is present in the input (no quantisation of data)',
          got.shape[:2] == (2 * H, 2 * W)
          and vals <= set(np.unique(ramp).tolist()), str(got.shape))
    why = SIG.threedo_2x_why(settings(color_depth='24'))
    check("threedo_2x_why at colour depth '24' names 15-bit and is None at '15'",
          why is not None and '15-bit' in why
          and SIG.threedo_2x_why(settings(color_depth='15')) is None, str(why))
    p = PRESETS['THREEDO']['settings']
    check("the THREEDO preset doubles through the display generator "
          "(output_scale THREEDO_2X on a 15-bit frame)",
          p.get('output_scale') == 'THREEDO_2X' and p.get('color_depth') == '15')


# --------------------------------------------------------- C047 GBA_MODE5

def test_r251_c047_gba_mode5():
    """The GBA Mode 5 affine stretch: the register law (240x160 <->
    160x128), the drifting pitch (runs of 1 and 2), no filtering, the
    round trip at 96x72, the 320x240 trim law, both roads the same shape
    and bitwise, the GBA_MODE5 preset renders on both devices."""
    check('gba_source_size(240, 160) is (160, 128)',
          SIG.gba_source_size(240, 160) == (160, 128))
    st = settings(output_scale='GBA_MODE5')
    src = np.zeros((128, 160, 3), np.float32)
    src[:, :, 0] = (np.arange(160, dtype=np.float32) / 255.0)[None, :]
    src[:, :, 1] = (np.arange(128, dtype=np.float32) / 255.0)[:, None]
    out = SIG.gba_stretch(src, st)
    check('a (128, 160, 3) frame stretches to (160, 240, 3)',
          out.shape == (160, 240, 3), str(out.shape))
    col = out[0, :, 0]
    runs = []
    n = 1
    for x in range(1, col.shape[0]):
        if col[x] == col[x - 1]:
            n += 1
        else:
            runs.append(n)
            n = 1
    runs.append(n)
    check('the run lengths of identical output columns are all in {1, 2} and '
          'both occur (the drifting pitch)',
          set(runs) == {1, 2}, str(sorted(set(runs))))
    check('every output pixel equals some source pixel (no filtering)',
          set(np.unique(out[..., 0]).tolist()) <= set(np.unique(src[..., 0]).tolist())
          and set(np.unique(out[..., 1]).tolist()) <= set(np.unique(src[..., 1]).tolist()))
    sw, sh = SIG.gba_source_size(96, 72)
    small = np.random.default_rng(47).random((sh, sw, 3)).astype(np.float32)
    check('the round trip holds at 96x72 (a 64x57 bitmap stretches to 96x72)',
          (sw, sh) == (64, 57) and SIG.gba_stretch(small, st).shape == (72, 96, 3))
    tw, th = SIG.gba_source_size(320, 240)
    big = np.random.default_rng(48).random((th, tw, 3)).astype(np.float32)
    bare = SIG.gba_stretch(big, st)
    trimmed = SIG.gba_stretch(big, st, size=(320, 240))
    fitted = PO.fit_to(bare, (320, 240))
    check('at 320x240 the bitmap is 214x192, the bare stretch is (240, 321, 3) '
          '(one repeated column) and size= / fit_to both give (240, 320, 3) '
          'bitwise each other',
          (tw, th) == (214, 192) and bare.shape == (240, 321, 3)
          and trimmed.shape == (240, 320, 3) and fitted.shape == (240, 320, 3)
          and bool(np.array_equal(trimmed, fitted))
          and bool(np.array_equal(bare[:, 320], bare[:, 319])),
          f'{(tw, th)} {bare.shape}')
    out_g, out_c, rec, live = gpu_road_resize(settings(output_scale='GBA_MODE5'))
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape else float('inf')
    check('both roads give the same stretched shape and the same picture bitwise',
          out_g.shape == out_c.shape and out_g.shape[:2] == (90, 144)
          and d == 0.0 and live == [], f'{out_g.shape} {out_c.shape} max {d}')
    stp = base_settings(160, 128)
    apply_preset(stp, 'GBA_MODE5')
    stp.resolution_x, stp.resolution_y = 160, 128
    stp.aa_samples = 1
    sc = demo_scene(stp, with_texture=False)
    cpu = PO.process(R.render(sc, stp), stp, frame=1, seed=0, allow_resize=True)
    stp2 = stp.copy()
    stp2.render_device = 'GPU'
    sc2 = demo_scene(stp2, with_texture=False)
    gpu = PO.process(R.render(sc2, stp2), stp2, frame=1, seed=0, allow_resize=True)
    check('the GBA_MODE5 preset renders on both devices to (160, 240, 3) from '
          'the 160x128 bitmap, bitwise',
          cpu.shape[:2] == (160, 240) and gpu.shape == cpu.shape
          and bool(np.array_equal(cpu, gpu)), str(cpu.shape))
    p = PRESETS['GBA_MODE5']
    check("the GBA_MODE5 preset is a 240x160 CONSOLE shelf at 15 bits, FLAT, "
          "no dither, no gamma key",
          p['category'] == 'CONSOLE' and p['settings']['resolution_x'] == 240
          and p['settings']['color_depth'] == '15'
          and p['settings']['default_model'] == 'FLAT'
          and 'gamma' not in p['settings'])


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
    print(f'\n{len(FAILS)} failure(s)' if FAILS else '\nall post-signal checks passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
