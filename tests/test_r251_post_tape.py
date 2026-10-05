"""R251 post-signal pack, wave 2 (SIG-2: C131, C129, C130, C128).

What sat between the machine and the glass once the picture left the
machine: the digital formats' chroma siting, the S-Video and RF cables,
the PAL receiver, the analogue tape. Every check reads as a sentence of
what it proves: the semantic law of each mechanism, the GPU twin bitwise
the CPU (d == 0.0) in the simulator and through the fake device, the
refusal by name, the identity at defaults against the 1.89.0 zip.

    python -m halcyon.tests.test_r251_post_tape
"""

import importlib
import sys
import traceback

import numpy as np

from ..core import post as PO
from ..core import render as R
from ..core import signal_era as SIG
from ..core import signal_tape as SIGT
from ..core.settings import RenderSettings
from ..gpu import chain
from ..gpu import chain_tape as CT
from ..gpu import frame as FR
from ..gpu import shade as GSH
from ..gpu import stages
from ..gpu import stages_tape
from ..presets.library import apply_preset, PRESETS
from . import fakedevice
from .r251_common import clear_palette_locks
from .scenebuild import demo_scene
from .test_render import base_settings, _prev_engine
# the pack's shared helpers, written once in the wave-1 module (pure
# functions: the device switch, the simulator draw, the record reader)
from .test_r251_post_signal import (W, H, settings, post_kw, gpu_road,
                                    run_stage, _rec_ok, _capture, _demo_frame,
                                    _CASES, _case_settings)

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


#: every field this slot adds, with its default
FIELDS = ('chroma_format', 'chroma_upsample', 'signal', 'rf_bandwidth',
          'rf_beat', 'rf_snow', 'rf_ghost', 'rf_ghost_delay', 'pal_decoder',
          'pal_phase_error', 'pal_crawl', 'tape', 'tape_generations',
          'tape_noise', 'tape_head_switch', 'tape_dropouts')


def _rgba(rgb):
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], 2)


def _dmax(a, b):
    if a is None or b is None or a.shape != b.shape:
        return float('inf')
    return float(np.abs(a - b).max())


def _probe_false(st, call):
    """Run `call(frame)` on a fake device whose probe fails: (got, text)."""
    from ..gpu import device as DEV
    st.render_device = 'GPU'
    rng = np.random.default_rng(5)
    img = rng.random((H, W, 3)).astype(np.float32)
    with fakedevice.installed():
        DEV.probe = lambda: (False, 'no driver')     # restored by the context
        st._frame_gpu_shaded = True
        fr = chain.Frame(rgb=img)
        got, text = _capture(lambda: call(fr))
        fr.release()
    return got, text


def _ycc(rgb):
    """The frame's integer BT.601 planes (Y, Cb, Cr)."""
    v8 = SIG.to_u8(rgb)
    a = (v8[..., 0], v8[..., 1], v8[..., 2])
    return SIG.y601(*a), SIG.cb601(*a), SIG.cr601(*a)


# ---------------------------------------------------------------- the pin

def test_r251_00_tape_defaults_are_invisible():
    """The slot's pin: the 1.89.0 zip is beside the package and every
    default leaves every pixel of render AND post bitwise the previous
    release's; every `_on` predicate is False on a default settings
    object; every chain-stage function is the identity at defaults; the
    stages merged before ENABLED was derived."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the neutrality pin runs)',
          RP is not None)
    if RP is not None:
        prev_post = importlib.import_module(
            RP.__name__.rsplit('.', 1)[0] + '.post')
        for label, kw in _CASES:
            clear_palette_locks(RP)
            st = _case_settings(kw)
            sc = demo_scene(st, with_texture=False)
            now = PO.process(R.render(sc, st), st, **post_kw(sc, st))
            st2 = _case_settings(kw)
            sc2 = demo_scene(st2, with_texture=False)
            prev = prev_post.process(RP.render(sc2, st2), st2,
                                     **post_kw(sc2, st2))
            same = now.shape == prev.shape and bool(np.array_equal(now, prev))
            check(f'the CPU device, {label}: render + post bitwise the '
                  "1.89.0 release's (the tape / cable / PAL / chroma "
                  'defaults are invisible)', same,
                  f'max {_dmax(now, prev)}')
    st = RenderSettings()
    want = {'chroma_format': 'NONE', 'chroma_upsample': 'HOLD',
            'signal': 'RGB', 'rf_bandwidth': 3.2, 'rf_beat': 0.0,
            'rf_snow': 0.0, 'rf_ghost': 0.0, 'rf_ghost_delay': 1.0,
            'pal_decoder': 'NONE', 'pal_phase_error': 0.0, 'pal_crawl': 0.0,
            'tape': 'NONE', 'tape_generations': 1, 'tape_noise': 0.02,
            'tape_head_switch': True, 'tape_dropouts': 0.0}
    check('the sixteen new settings fields exist with the defaults the '
          'handover lists', set(want) == set(FIELDS) and all(
              getattr(st, k, None) == v for k, v in want.items()),
          str({k: getattr(st, k, None) for k in want
               if getattr(st, k, None) != want[k]}))
    preds = {'chroma_site': SIGT.chroma_site_on, 'xfb': SIGT.xfb_on,
             'cable_chroma': SIGT.cable_chroma_on,
             'cable_rf': SIGT.cable_rf_on, 'pal_decode': SIGT.pal_decode_on,
             'tape': SIGT.tape_on}
    check('no tape / cable / PAL / chroma predicate is on for a default '
          'settings object', not any(p(st) for p in preds.values()),
          str({k: p(st) for k, p in preds.items()}))
    rng = np.random.default_rng(251)
    img = rng.random((12, 16, 3)).astype(np.float32)
    fns = {'chroma_site': lambda: SIGT.chroma_site(img, st),
           'svideo': lambda: SIGT.svideo(img, st),
           'rf_modulate': lambda: SIGT.rf_modulate(img, st, 3, 1),
           'pal_decode': lambda: SIGT.pal_decode(img, st, 3),
           'tape_path': lambda: SIGT.tape_path(img, st, 3, 1)}
    for name, fn in fns.items():
        check(f'signal_tape.{name} on default settings returns its input '
              '(the opening guard)', fn() is img)
    check("the slot's stages merged into stages.STAGES before ENABLED was "
          'derived (a stage pasted after ENABLED would refuse silently)',
          set(stages_tape.STAGES_TAPE) <= set(stages.ENABLED)
          and set(stages_tape.STAGES_TAPE) <= set(stages.STAGES)
          and set(stages_tape.INTERFACE_TAPE) == set(stages_tape.STAGES_TAPE)
          and set(stages_tape.VALIDATION_TAPE) == set(stages_tape.STAGES_TAPE),
          str(sorted(set(stages_tape.STAGES_TAPE) - set(stages.ENABLED))))
    check('every stage is a chain function post.process finds by name',
          all(callable(getattr(chain, n, None))
              for n in ('chroma_site', 'tape', 'cable_chroma', 'pal_decode',
                        'cable_rf')))
    from ..gpu import stages_signal
    check("the tape's two component stages are named in SIGNAL_MULTI (the "
          'self test measures them only through chain.tape)',
          'TAPE_FIR' in stages_signal.SIGNAL_MULTI
          and 'TAPE_OUT' in stages_signal.SIGNAL_MULTI)
    for name, body in stages_tape.STAGES_TAPE.items():
        bad = [t for t in ('texture(', 'log10', 'saturate(', 'lerp(',
                           'frac(', ' % ', ' / ') if t in body]
        check(f'{name}: the body stays in the accepted GLSL subset (no '
              'texture(), no int / int, no %, no HLSL names)', not bad,
              str(bad))


def test_r251_01_tape_featurematrix_rows():
    """Every row this slot adds exists in featurematrix.ROWS, renders on
    both devices bitwise without a driver (the fallback plumbing; the
    twin proof is the gpu_road cases) and moves the picture."""
    from .featurematrix import ROWS, build
    from .test_render import _matrix_run
    keys = ('DV 4:1:1 chroma, held', 'MPEG-1 4:2:0 chroma, linear',
            'GameCube XFB 4:2:2', 'S-Video cable',
            'RF modulator over composite', 'PAL delay-line decoder + crawl',
            'Simple PAL, Hanover bars', 'VHS tape, third generation')
    names = {k for k, _o, _s in ROWS}
    check('every SIG-2 feature has its featurematrix row (8)',
          all(k in names for k in keys),
          str([k for k in keys if k not in names]))
    d0 = RenderSettings()
    for k in keys:
        if k not in names:
            continue
        sc, st = build(k)
        cpu = _matrix_run(sc, st)
        sc2, st2 = build(k)
        st2.render_device = 'GPU'
        gpu = _matrix_run(sc2, st2)
        sc3, st3 = build(k)
        for f in FIELDS:
            setattr(st3, f, getattr(d0, f))
        off = _matrix_run(sc3, st3)
        check(f"row '{k}' falls back bit-exactly without a driver and moves "
              'the picture (not vacuous)',
              cpu.shape == gpu.shape and bool((cpu == gpu).all())
              and not (off.shape == cpu.shape and bool((off == cpu).all())))


# ---------------------------------------------------- C131 CHROMA_SITE

def _site_settings(fmt, up='HOLD', **kw):
    return settings(chroma_format=fmt, chroma_upsample=up, **kw)


def test_r251_c131_chroma_siting():
    """Chroma subsampling and siting: the siting law of each format on a
    one-pixel red column, the linear decoder's monotone chroma, the flat
    frame's legal round trip, XFB against Y422 (RGB averaged before the
    matrix versus chroma after), band invariance, the GPU twin for all
    six formats held and linear, the resident records (alone, before the
    palette, after the copy filter), the refusal by name, the presets."""
    st0 = RenderSettings()
    check('chroma_site_on and xfb_on are False on default settings',
          not SIGT.chroma_site_on(st0) and not SIGT.xfb_on(st0))
    check('the five file formats switch chroma_site_on (block A) and only '
          'XFB_422 switches xfb_on (block B, after the copy filter)',
          all(SIGT.chroma_site_on(_site_settings(f))
              and not SIGT.xfb_on(_site_settings(f))
              for f in SIGT.SITES if f != 'XFB_422')
          and SIGT.xfb_on(_site_settings('XFB_422'))
          and not SIGT.chroma_site_on(_site_settings('XFB_422')))

    # --- the siting law: a 1-px red column at x = 5 over grey, 48 x 64
    h, w = 48, 64
    grey = SIG.U8[128]
    col = np.full((h, w, 3), grey, np.float32)
    col[:, 5] = (SIG.U8[230], SIG.U8[26], SIG.U8[26])
    _y_in, _cb_in, cr_in = _ycc(col)
    cr_red, cr_grey = int(cr_in[0, 5]), int(cr_in[0, 0])

    def cr_site(fmt, up='HOLD'):
        """The stored chroma the decoder was handed: the site arrays."""
        return SIGT.chroma_upsample(
            *SIGT.chroma_sites(SIG.to_u8(col), fmt), fmt, h, w, up == 'LINEAR')[1]

    cr = cr_site('Y411')
    want = (cr_red + 3 * cr_grey + 2) >> 2
    check('Y411 HOLD on a 1-px red column at x = 5: the decoded Cr is '
          'constant over x in 4..7 at (Cr_red + 3 Cr_grey + 2) >> 2 and '
          'grey outside (colour edges in four-pixel steps)',
          bool((cr[:, 4:8] == want).all()) and bool((cr[:, :4] == cr_grey).all())
          and bool((cr[:, 8:] == cr_grey).all()), f'{cr[0, 3:9].tolist()} want {want}')
    out = SIGT.chroma_site(col, _site_settings('Y411'))
    y_out, _cb, cr_out = _ycc(out)
    check('and in the PICTURE: the output Cr is raised over x in 4..7 and '
          'nowhere else, while the luma column stays one pixel wide (the '
          'output luma within 1 level of the input everywhere)',
          bool((cr_out[:, 4:8] > cr_grey + 4).all())
          and bool((np.abs(cr_out[:, :4] - cr_grey) <= 1).all())
          and bool((np.abs(cr_out[:, 8:] - cr_grey) <= 1).all())
          and int(np.abs(y_out - _y_in).max()) <= 1,
          f'Cr row {cr_out[0, 2:10].tolist()}; luma max '
          f'{int(np.abs(y_out - _y_in).max())}')
    cr = cr_site('Y422')
    check('Y422 HOLD: the colour covers x = 4..5 at (Cr_grey + Cr_red + 1) '
          '>> 1 (one sample per two, sited on the even one)',
          bool((cr[:, 4:6] == ((cr_red + cr_grey + 1) >> 1)).all())
          and bool((cr[:, :4] == cr_grey).all())
          and bool((cr[:, 6:] == cr_grey).all()), str(cr[0, 3:8].tolist()))
    dot = np.full((h, w, 3), grey, np.float32)
    dot[7, 5] = (SIG.U8[230], SIG.U8[26], SIG.U8[26])
    cr = SIGT.chroma_upsample(*SIGT.chroma_sites(SIG.to_u8(dot), 'Y420_MPEG1'),
                              'Y420_MPEG1', h, w, False)[1]
    cell = cr != cr_grey
    check('Y420_MPEG1 HOLD on one red PIXEL at (5, 7): the colour covers '
          'exactly the 2x2 cell x 4..5, y 6..7 at the quad average',
          bool(cell[6:8, 4:6].all()) and int(cell.sum()) == 4
          and int(cr[6, 4]) == ((cr_red + 3 * cr_grey + 2) >> 2),
          f'{int(cell.sum())} px, value {int(cr[6, 4])}')
    cb_d, cr_d = SIGT.chroma_upsample(
        *SIGT.chroma_sites(SIG.to_u8(dot), 'Y420_DVPAL'), 'Y420_DVPAL', h, w,
        False)
    check('Y420_DVPAL: Cb is sampled on the even line of the pair and Cr on '
          'the odd one (the red pixel on row 7 moves Cr over rows 6..7 and '
          'leaves Cb alone)',
          bool((cr_d[6:8, 4:6] != cr_grey).all())
          and bool((cb_d == cb_d[0, 0]).all()),
          f'Cr {cr_d[6:8, 4:6].tolist()} Cb moved {int((cb_d != cb_d[0, 0]).sum())}')
    for fmt in ('Y422', 'Y411', 'Y420_MPEG1', 'Y420_MPEG2', 'XFB_422'):
        row = cr_site(fmt, 'LINEAR')[0].astype(np.int64)
        peak = int(np.argmax(row))
        left = row[:peak + 1]
        right = row[peak:]
        check(f'{fmt} LINEAR: the chroma along x is monotone away from the '
              'site (non-decreasing up to the peak, non-increasing after)',
              bool((np.diff(left) >= 0).all()) and bool((np.diff(right) <= 0).all())
              and int(row.max()) > cr_grey, str(row[2:10].tolist()))

    # --- a flat frame is bitwise its legal round trip, every format
    flat = np.full((10, 14, 3), 0.0, np.float32)
    flat[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[40])
    y, cb, cr = _ycc(flat)
    trip = SIG.U8[np.stack(SIG.yuv601_decode(y, cb, cr), -1)]
    ok = True
    for fmt in SIGT.SITES:
        for up in ('HOLD', 'LINEAR'):
            ok &= bool(np.array_equal(
                SIGT.chroma_site(flat, _site_settings(fmt, up)), trip))
    check('a flat frame is bitwise its legal BT.601 round trip for every '
          'format, held and linear (odd sizes: the edge clamps)', ok)

    # --- XFB against Y422
    # (the spec's pair (1,0,0), (0,0,1) gives the SAME sites on both roads --
    # the matrix is linear, 165 / 175 either way; what separates the two
    # is the rounding: a floor average in RGB before the matrix against a
    # half-up average of the chroma after it. Measured on a random frame:
    # 564 of 2048 Cb sites differ, never by more than one level)
    pair = np.zeros((2, 2, 3), np.float32)
    pair[:, 0] = (1.0, 0.0, 0.0)
    pair[:, 1] = (0.0, 0.0, 1.0)
    m = np.array([127, 0, 127])
    cb_x = int(np.clip(SIG.cb601(*m), 16, 240))
    cbs, _crs = SIGT.chroma_sites(SIG.to_u8(pair), 'XFB_422')
    rnd = np.random.default_rng(0).random((64, 64, 3)).astype(np.float32)
    xb, xr = SIGT.chroma_sites(SIG.to_u8(rnd), 'XFB_422')
    yb, yr = SIGT.chroma_sites(SIG.to_u8(rnd), 'Y422')
    ndiff = int((xb != yb).sum() + (xr != yr).sum())
    dmax = int(max(np.abs(xb - yb).max(), np.abs(xr - yr).max()))
    check('XFB_422 averages the pair in RGB FIRST (floor: (1,0,0), (0,0,1) '
          '-> 127, 0, 127), then runs the matrix; Y422 averages the chroma '
          'after it: on a random frame the two site planes differ, and only '
          "by the rounding (at most one level)",
          int(cbs[0, 0]) == cb_x and ndiff > 0 and dmax == 1
          and not np.array_equal(
              SIGT.chroma_site(rnd, _site_settings('XFB_422')),
              SIGT.chroma_site(rnd, _site_settings('Y422'))),
          f'XFB Cb {int(cbs[0, 0])} want {cb_x}; {ndiff} sites differ, max {dmax}')

    # --- band invariance
    rgb, _st = _demo_frame()
    ok = True
    for fmt in ('Y422', 'Y411', 'XFB_422'):
        for up in ('HOLD', 'LINEAR'):
            st = _site_settings(fmt, up)
            whole = SIGT.chroma_site(rgb, st)
            cut = 31
            both = np.concatenate([SIGT.chroma_site(rgb[:cut], st),
                                   SIGT.chroma_site(rgb[cut:], st)], 0)
            ok &= bool(np.array_equal(whole, both))
    check('band invariance: Y422 / Y411 / XFB processed as two row bands '
          '(cut at row 31) equal the whole frame bitwise', ok)
    ok = True
    for fmt in ('Y420_MPEG1', 'Y420_MPEG2', 'Y420_DVPAL'):
        st = _site_settings(fmt, 'HOLD')
        whole = SIGT.chroma_site(rgb, st)
        cut = 30
        both = np.concatenate([SIGT.chroma_site(rgb[:cut], st),
                               SIGT.chroma_site(rgb[cut:], st)], 0)
        ok &= bool(np.array_equal(whole, both))
    check('and the three 4:2:0 formats likewise when the cut is on an even '
          'row (HOLD: a block never straddles the cut)', ok)

    # --- the GPU twin: six formats x HOLD / LINEAR, the demo frame and an
    #     odd-sized random frame (every edge clamp)
    rng = np.random.default_rng(131)
    odd = rng.random((11, 13, 3)).astype(np.float32)
    for fmt in SIGT.SITES:
        for up in ('HOLD', 'LINEAR'):
            st = _site_settings(fmt, up)
            worst, moved, alpha, err = 0.0, 0.0, True, None
            for img in (rgb, odd):
                hh, ww = img.shape[:2]
                got, err = run_stage('CHROMA_SITE', hh, ww, {
                    'resolution': (float(ww), float(hh)),
                    'fmt': SIGT.SITE_FMT[fmt],
                    'interp': 1 if up == 'LINEAR' else 0},
                    {'source': _rgba(img), 'u8lut': SIG.u8_image()})
                if got is None:
                    worst = float('inf')
                    break
                ref = SIGT.chroma_site(img, st)
                worst = max(worst, _dmax(got[..., :3], ref))
                moved = max(moved, float(np.abs(ref - img).max()))
                alpha &= bool((got[..., 3] == 1.0).all())
            check(f'CHROMA_SITE stage, {fmt} {up}: bitwise '
                  'signal_tape.chroma_site on the demo frame and on an odd '
                  '13 x 11 frame (d == 0.0), alpha through, not vacuous',
                  worst == 0.0 and moved > 0.0 and alpha,
                  str(err) if err else f'max {worst}; moved {moved:.4f}')

    # --- the resident road
    out_g, out_c, rec, live = gpu_road(_site_settings(
        'Y420_MPEG1', 'LINEAR', dither='NONE'))
    d = _dmax(out_g, out_c)
    check('CHROMA_SITE on the resident frame (MPEG-1 linear, no dither): the '
          'GPU chain equals the CPU chain bitwise, drawn between the DISPLAY '
          "and the QUANT, 0 readbacks, the render's frame inherited (no "
          'upload), nothing live',
          _rec_ok(rec, live, d, dict(
              resident=True, stages=['DISPLAY', 'CHROMA_SITE', 'QUANT'],
              readbacks=0, uploads=0)),
          f'max {d}; {rec}; {len(live)} live')
    out_g, out_c, rec, live = gpu_road(_site_settings(
        'Y420_MPEG1', 'LINEAR', dither='NONE', color_depth='8'))
    d = _dmax(out_g, out_c)
    stg = rec.get('stages') or []
    check("with colour depth '8' as well (the CD-ROM road): bitwise, and the "
          "chroma stage draws BEFORE the display's own quantiser (a file is "
          'decoded before the display quantises it)',
          d == 0.0 and live == [] and 'CHROMA_SITE' in stg
          and stg.index('CHROMA_SITE') == stg.index('DISPLAY') + 1,
          f'max {d}; {rec}; {len(live)} live')
    out_g, out_c, rec, live = gpu_road(_site_settings(
        'XFB_422', dither='NONE', copy_filter='DEFLICKER'))
    d = _dmax(out_g, out_c)
    stg = rec.get('stages') or []
    check('the order law on the GPU: XFB_422 with the copy filter records '
          'COPY_FILTER BEFORE CHROMA_SITE (filter the RGB rows, then store '
          '4:2:2), bitwise, 0 readbacks',
          _rec_ok(rec, live, d, dict(stages_has=['COPY_FILTER', 'CHROMA_SITE'],
                                     readbacks=0))
          and stg.index('COPY_FILTER') < stg.index('CHROMA_SITE'),
          f'max {d}; {rec}; {len(live)} live')
    st = _site_settings('XFB_422', dither='NONE', copy_filter='DEFLICKER')
    sc = demo_scene(st, with_texture=False)
    out = PO.process(R.render(sc, st), st, **post_kw(sc, st))[..., :3]
    fwd = SIGT.chroma_site(SIG.copy_filter(rgb, st), st)
    rev = SIG.copy_filter(SIGT.chroma_site(rgb, st), st)
    check('and on the CPU: post.process with both on equals '
          'chroma_site(copy_filter(frame)) on the quantised frame, not the '
          'reverse',
          bool(np.array_equal(out, fwd)) and not np.array_equal(fwd, rev),
          f'max {_dmax(out, fwd)}')

    # --- the refusal by name
    with fakedevice.installed():
        check('chroma_site_refusal is None when the stage may draw (Y411, '
              'and XFB_422)',
              CT.chroma_site_refusal(_site_settings(
                  'Y411', render_device='GPU')) is None
              and CT.chroma_site_refusal(_site_settings(
                  'XFB_422', render_device='GPU')) is None)
    check('chroma_site_refusal names the CPU device on the CPU device',
          'CPU' in str(CT.chroma_site_refusal(_site_settings('Y411'))))
    got, text = _probe_false(_site_settings('Y411'),
                             lambda fr: chain.chroma_site(fr, _site_settings(
                                 'Y411', render_device='GPU')))
    check("a device whose probe fails makes the chain print 'CHROMA_SITE on "
          "the CPU: no driver' and return None",
          got is None and 'CHROMA_SITE on the CPU: no driver' in text,
          text.strip())

    # --- the presets
    check('the digital-video presets site chroma the way their formats did: '
          'TOASTER and SGI_BROADCAST Y422 (D1), CD_ROM_FMV Y420_MPEG1, '
          'GAMECUBE XFB_422',
          PRESETS['TOASTER']['settings'].get('chroma_format') == 'Y422'
          and PRESETS['SGI_BROADCAST']['settings'].get('chroma_format') == 'Y422'
          and PRESETS['CD_ROM_FMV']['settings'].get('chroma_format') == 'Y420_MPEG1'
          and PRESETS['GAMECUBE']['settings'].get('chroma_format') == 'XFB_422')


# ------------------------------------------- C129 CABLE_CHROMA / CABLE_RF

def _yiq(rgb):
    v8 = SIG.to_u8(rgb)
    return SIG.yiq_int(v8[..., 0], v8[..., 1], v8[..., 2])


def _yuv(rgb):
    v8 = SIG.to_u8(rgb)
    return SIG.pal_yuv(v8[..., 0], v8[..., 1], v8[..., 2])


def _peak_grad(plane):
    return int(np.abs(np.diff(plane.astype(np.int64), axis=1)).max())


def test_r251_c129_signal_path():
    """The cable: S-Video keeps luma whole and band-limits the encoded
    chroma (Q harder than I; U and V alike for PAL), the both-switches
    notice, the RF laws (the beat in proportion to chroma and moving with
    the frame, deterministic snow, the ghost to the right), the GPU twins
    bitwise, RF over the composite stage within two levels, the loop-bound
    refusals by name, the presets."""
    st0 = RenderSettings()
    check("the default cable is 'RGB': cable_chroma_on and cable_rf_on are "
          'False', not SIGT.cable_chroma_on(st0) and not SIGT.cable_rf_on(st0))

    # --- the S-Video law: a red bar over grey
    h, w = 24, 96
    bar = np.full((h, w, 3), SIG.U8[128], np.float32)
    bar[:, 40:60] = (SIG.U8[230], SIG.U8[26], SIG.U8[26])
    y0, i0, q0 = _yiq(bar)
    out = SIGT.svideo(bar, settings(signal='SVIDEO'))
    y1, i1, q1 = _yiq(out)
    ri = _peak_grad(i1) / _peak_grad(i0)
    rq = _peak_grad(q1) / _peak_grad(q0)
    check('SVIDEO on a red bar over grey: luma whole (the output Y within 2 '
          'levels of the input everywhere: the 8.8 constants round on encode '
          'and decode), the steepest I and Q edges both fall, and Q (0.5 MHz) '
          'falls by a larger ratio than I (1.3 MHz)',
          int(np.abs(y1 - y0).max()) <= 2 and ri < 1.0 and rq < ri,
          f'Y max {int(np.abs(y1 - y0).max())}; I {ri:.3f}; Q {rq:.3f}')
    check('and the chroma spreads beyond the bar while no colour is made: '
          'the pixel just outside the bar carries I, and the row sum of I is '
          'preserved within the rounding (a low-pass, taps summing to 65536)',
          int(i1[0, 39]) > 0 and int(i0[0, 39]) == 0
          and abs(int(i1[0].sum()) - int(i0[0].sum())) <= w,
          f'I[39] {int(i1[0, 39])}; sums {int(i1[0].sum())} vs {int(i0[0].sum())}')
    yu0, u0, v0 = _yuv(bar)
    outp = SIGT.svideo(bar, settings(signal='SVIDEO_PAL'))
    yu1, u1, v1 = _yuv(outp)
    ru = _peak_grad(u1) / _peak_grad(u0)
    rv = _peak_grad(v1) / _peak_grad(v0)
    Pp = SIGT.cable_params(settings(signal='SVIDEO_PAL'), w)
    check('SVIDEO_PAL: luma within 2 levels; U and V go through the SAME '
          'taps (1.3 MHz and 1.3 MHz) and their steepest edges fall by the '
          'same ratio within 10% (measured 0.533 and 0.568: the integer '
          'planes round separately)',
          int(np.abs(yu1 - yu0).max()) <= 2 and ru < 1.0 and rv < 1.0
          and bool(np.array_equal(Pp['taps1'], Pp['taps2']))
          and abs(ru - rv) <= 0.10 * max(ru, rv),
          f'Y max {int(np.abs(yu1 - yu0).max())}; U {ru:.3f}; V {rv:.3f}')
    flat = np.zeros((6, 40, 3), np.float32)
    flat[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[40])
    v8 = SIG.to_u8(flat)
    a = (v8[..., 0], v8[..., 1], v8[..., 2])
    trip_n = SIG.U8[np.stack(SIG.yiq_back(*SIG.yiq_int(*a)), -1)]
    trip_p = SIG.U8[np.stack(SIG.pal_back(*SIG.pal_yuv(*a)), -1)]
    check('a flat frame comes out flat and bitwise its integer round trip, '
          'both cables (the FIR sums to 65536)',
          bool(np.array_equal(SIGT.svideo(flat, settings(signal='SVIDEO')),
                              trip_n))
          and bool(np.array_equal(
              SIGT.svideo(flat, settings(signal='SVIDEO_PAL')), trip_p)))
    got, text = _capture(lambda: SIGT.cable_chroma_on(
        settings(signal='SVIDEO', composite=True)))
    check("S-Video with Composite Video on: the stage still runs and the "
          "console says the cable is 'one or the other'",
          got is True and 'one or the other' in text, text.strip())
    got, text = _capture(lambda: SIGT.cable_chroma_on(
        settings(signal='SVIDEO', composite=False)))
    check('and with Composite Video off it prints nothing',
          got is True and text.strip() == '', text.strip())

    # --- the RF laws
    def rf(img, frame=0, seed=0, **kw):
        base = dict(signal='RF', rf_beat=0.0, rf_snow=0.0, rf_ghost=0.0)
        base.update(kw)
        return SIGT.rf_modulate(img, settings(**base), frame, seed)

    grey = np.full((h, w, 3), SIG.U8[128], np.float32)
    check('RF on a flat grey with the beat at 1: bitwise the low-pass-only '
          'output (the herringbone is proportional to chroma, and grey has '
          'none)', bool(np.array_equal(rf(grey, rf_beat=1.0), rf(grey))))
    # a red that stays inside the gamut under the beat (a fully saturated
    # one clips in the decode and the re-measured luma mean rises by 3)
    red = np.zeros((h, w, 3), np.float32)
    red[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[90])
    yb = _yiq(rf(red, rf_beat=1.0))[0]
    yf = _yiq(rf(red))[0]
    P = SIGT.rf_params(settings(signal='RF', rf_beat=1.0), w, 0, 0)
    check('RF on a flat red with the beat at 1: the luma varies along the '
          'line (the 920 kHz herringbone; a_ph = round(65536 * 0.92 * 52.66 '
          '/ 96) = 33073, about two pixels per cycle) and its mean stays '
          'within 1.5 levels of the low-passed luma (a mean-zero beat, '
          'floored)',
          P['a_ph'] == 33073 and int(yb[0].max() - yb[0].min()) >= 4
          and abs(float(yb.mean()) - float(yf.mean())) <= 1.5,
          f"a_ph {P['a_ph']}; span {int(yb[0].max() - yb[0].min())}; "
          f'mean {float(yb.mean()):.2f} vs {float(yf.mean()):.2f}')
    check('the beat moves with the frame (0.65 cycle per frame): frames 0 '
          'and 1 differ',
          not np.array_equal(rf(red, 0, rf_beat=1.0), rf(red, 1, rf_beat=1.0)))
    rgb, _st = _demo_frame()
    a = rf(rgb, 7, 3, rf_snow=0.2)
    check('the snow is a hash: the same (frame, seed) is bitwise the same '
          'picture; frame + 1 differs; seed + 1 differs; and it moves the '
          'picture',
          bool(np.array_equal(a, rf(rgb, 7, 3, rf_snow=0.2)))
          and not np.array_equal(a, rf(rgb, 8, 3, rf_snow=0.2))
          and not np.array_equal(a, rf(rgb, 7, 4, rf_snow=0.2))
          and not np.array_equal(a, rf(rgb, 7, 3)))
    one = np.zeros((h, w, 3), np.float32)
    x0 = 20
    one[:, x0] = 1.0
    stg = settings(signal='RF', rf_ghost=0.5, rf_ghost_delay=3.0,
                   rf_bandwidth=4.2)
    dpx = SIGT.rf_params(stg, w, 0, 0)['ghost_d']
    yg = _yiq(SIGT.rf_modulate(one, stg))[0][0]
    check('the ghost: one bright column at x0 on black with rf_ghost 0.5 puts '
          'a second bump ghost_d px to the RIGHT at >= 0.4 of the first, and '
          'nothing to the left',
          dpx == int(round(3.0 * w / 52.66)) and dpx >= 3
          and int(yg[x0 + dpx]) >= 0.4 * int(yg[x0])
          and int(yg[x0 - dpx]) == 0,
          f'd {dpx}; Y[x0] {int(yg[x0])}; Y[x0+d] {int(yg[x0 + dpx])}')

    # --- the GPU twins
    src = _rgba(rgb)
    for item in ('SVIDEO', 'SVIDEO_PAL'):
        st = settings(signal=item)
        P = SIGT.cable_params(st, W)
        got, err = run_stage('CABLE_CHROMA', H, W, {
            'resolution': (float(W), float(H)), 'space': P['space'],
            'r1': P['r1'], 'r2': P['r2']},
            {'source': src, 'u8lut': SIG.u8_image(),
             'taps': SIGT.taps_image(P['taps1'], P['taps2'])})
        ref = SIGT.svideo(rgb, st)
        d = _dmax(None if got is None else got[..., :3], ref)
        check(f'CABLE_CHROMA stage, {item}: bitwise signal_tape.svideo on the '
              'demo frame (d == 0.0), alpha through, not vacuous',
              d == 0.0 and float(np.abs(ref - rgb).max()) > 0.0
              and bool((got[..., 3] == 1.0).all()),
              str(err) if got is None else f'max {d}')
    st = settings(signal='RF', rf_beat=0.5, rf_snow=0.05, rf_ghost=0.3)
    P = SIGT.rf_params(st, W, 7, 3)
    u = {k: int(P[k]) for k in SIGT.RF_INTS}
    u['resolution'] = (float(W), float(H))
    got, err = run_stage('CABLE_RF', H, W, u, {
        'source': src, 'u8lut': SIG.u8_image(), 'sin16': SIG.sin16_image(),
        'taps': SIGT.taps_image(P['taps'])})
    ref = SIGT.rf_modulate(rgb, st, 7, 3)
    d = _dmax(None if got is None else got[..., :3], ref)
    check('CABLE_RF stage (beat 0.5, snow 0.05, ghost 0.3, frame 7, seed 3): '
          'bitwise signal_tape.rf_modulate (d == 0.0), alpha through, not '
          'vacuous',
          d == 0.0 and float(np.abs(ref - rgb).max()) > 0.0
          and bool((got[..., 3] == 1.0).all()),
          str(err) if got is None else f'max {d}')
    out_g, out_c, rec, live = gpu_road(settings(dither='NONE', signal='SVIDEO'))
    d = _dmax(out_g, out_c)
    check('S-Video on the resident frame (no dither): the GPU chain equals '
          'the CPU chain bitwise, CABLE_CHROMA drawn, 0 readbacks, nothing '
          'live',
          _rec_ok(rec, live, d, dict(stages_has=['CABLE_CHROMA'], readbacks=0)),
          f'max {d}; {rec}; {len(live)} live')
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', signal='RF', rf_beat=0.5, rf_snow=0.05, rf_ghost=0.3))
    d = _dmax(out_g, out_c)
    check('RF on the resident frame, Composite Video off: bitwise, CABLE_RF '
          'drawn, 0 readbacks, nothing live',
          _rec_ok(rec, live, d, dict(stages_has=['CABLE_RF'], readbacks=0)),
          f'max {d}; {rec}; {len(live)} live')
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', signal='RF', rf_beat=0.5, rf_snow=0.05, rf_ghost=0.3,
        composite=True))
    d = _dmax(out_g, out_c)
    stg_names = rec.get('stages') or []
    moved_px = int((np.abs(out_g - out_c).max(-1) > 0.0).sum())         if out_g.shape == out_c.shape else -1
    check('RF over the composite stage "runs and names its stages": NTSC then '
          'CABLE_RF drawn, nothing live, and the picture within 4/255 of the '
          'CPU chain at under 5% of the pixels (the NTSC stage is CLOSE 0.001 '
          'in the simulator; to_u8 flips a level on a pixel near a '
          'half-level, and the integer YIQ decode carries one flipped level '
          'of I and Q into blue at 1.1 + 1.7: measured 4 levels at 141 of '
          '6912 pixels -- the bitwise claim is the composite-off case)',
          'NTSC' in stg_names and 'CABLE_RF' in stg_names
          and stg_names.index('NTSC') < stg_names.index('CABLE_RF')
          and live == [] and d <= 4.0 / 255.0 + 1e-6
          and 0 <= moved_px <= 0.05 * out_g.shape[0] * out_g.shape[1],
          f'max {d} ({d * 255.0:.2f} levels) at {moved_px} px; {rec}; '
          f'{len(live)} live')

    # --- the refusals by name
    with fakedevice.installed():
        sv = settings(signal='SVIDEO', render_device='GPU')
        check('cable_chroma_refusal is None at 96 px, with and without a '
              "width, and names the 'loop bound' at 16384 px (the Q low-pass "
              'is wider than the 96-tap shader loop)',
              CT.cable_chroma_refusal(sv) is None
              and CT.cable_chroma_refusal(sv, w=96) is None
              and 'loop bound' in str(CT.cable_chroma_refusal(sv, w=16384)),
              str(CT.cable_chroma_refusal(sv, w=16384)))
        rfs = settings(signal='RF', rf_bandwidth=2.0, render_device='GPU')
        check("cable_rf_refusal likewise: None at 96 px, 'loop bound' at "
              '16384 px with the bandwidth at 2.0 MHz',
              CT.cable_rf_refusal(rfs) is None
              and 'loop bound' in str(CT.cable_rf_refusal(rfs, w=16384)),
              str(CT.cable_rf_refusal(rfs, w=16384)))
        wide = np.zeros((2, 2048, 3), np.float32)
        wide[:, ::7] = (0.9, 0.2, 0.1)
        sv._frame_gpu_shaded = True
        fr = chain.Frame(rgb=wide)
        got, text = _capture(lambda: chain.cable_chroma(fr, sv))
        fr.release()
        check('at 2048 px the S-Video chain refuses BY NAME (the Q radius is '
              "117 px): it prints 'CABLE_CHROMA on the CPU: ... loop bound' "
              'and returns None, and the CPU stage runs that frame',
              got is None and 'CABLE_CHROMA on the CPU' in text
              and 'loop bound' in text
              and SIGT.cable_params(sv, 2048)['r2'] == 117
              and SIGT.svideo(wide, sv).shape == wide.shape, text.strip())
    for name, call, stx in (
            ('CABLE_CHROMA', chain.cable_chroma, settings(signal='SVIDEO')),
            ('CABLE_RF', chain.cable_rf, settings(signal='RF'))):
        stx.render_device = 'GPU'
        got, text = _probe_false(stx, lambda fr, c=call, s=stx: c(fr, s))
        check(f"a device whose probe fails makes the chain print '{name} on "
              "the CPU: no driver' and return None",
              got is None and f'{name} on the CPU: no driver' in text,
              text.strip())

    # --- the presets
    p = PRESETS['SVIDEO']['settings']
    check('the SVIDEO preset is the cable it names: signal SVIDEO, Composite '
          'Video off, no composite dials left',
          p.get('signal') == 'SVIDEO' and p.get('composite') is False
          and not any(k in p for k in ('composite_bleed', 'composite_ringing',
                                       'composite_dot_crawl')))
    p = PRESETS.get('RF_MODULATOR')
    check('the RF_MODULATOR preset exists on the Video & Broadcast shelf: '
          'composite through the modulator (bandwidth 3.0 MHz, beat 0.4, snow '
          '0.03, ghost 0.15 at 1.5 us)',
          p is not None and p['category'] == 'BROADCAST'
          and len(p.get('note', '')) >= 40
          and p['settings'].get('signal') == 'RF'
          and p['settings'].get('composite') is True
          and p['settings'].get('rf_bandwidth') == 3.0
          and p['settings'].get('rf_beat') == 0.4
          and p['settings'].get('rf_snow') == 0.03
          and p['settings'].get('rf_ghost') == 0.15
          and p['settings'].get('rf_ghost_delay') == 1.5)


# ------------------------------------------------------ C130 PAL_DECODE

def _pal_trip(rgb):
    """The integer PAL YUV round trip of a frame (the NONE decoder)."""
    v8 = SIG.to_u8(rgb)
    a = (v8[..., 0], v8[..., 1], v8[..., 2])
    return SIG.U8[np.stack(SIG.pal_back(*SIG.pal_yuv(*a)), -1)]


def test_r251_c130_pal_decoder():
    """The PAL receiver: the offset-free matrix (a grey stays grey), the
    delay line's two-line chroma in the same field, the simple decoder's
    Hanover bars (mirror images about the true hue), the eight-field
    crawl, the GPU twin bitwise for every mode, the refusal by name, the
    PAL_TV preset."""
    st0 = RenderSettings()
    check('pal_decode_on is False on default settings',
          not SIGT.pal_decode_on(st0))
    check('the decoder OR the crawl switches the stage on',
          SIGT.pal_decode_on(settings(pal_decoder='DELAY_LINE'))
          and SIGT.pal_decode_on(settings(pal_decoder='SIMPLE'))
          and SIGT.pal_decode_on(settings(pal_crawl=0.5))
          and not SIGT.pal_decode_on(settings(pal_phase_error=20.0)))

    # --- the matrix law
    v = np.arange(256, dtype=np.int32)
    Y, U, V = SIG.pal_yuv(v, v, v)
    back = SIG.pal_back(Y, U, V)
    check('the matrix law: every grey 0..255 encodes to (v, 0, 0) on the '
          'offset-free luma and decodes to itself (with the +16 of the 601 '
          'luma a grey would decode tinted)',
          bool((Y == v).all()) and bool((U == 0).all()) and bool((V == 0).all())
          and all(bool((c == v).all()) for c in back))

    # --- the delay-line law: 1-px rows of red / blue at one luma
    h, w = 24, 32
    ca = (SIG.U8[200], SIG.U8[80], SIG.U8[80])
    cb = (SIG.U8[70], SIG.U8[125], SIG.U8[200])
    rows = np.zeros((h, w, 3), np.float32)
    rows[0::2] = ca
    rows[1::2] = cb
    y_in, u_in, v_in = _yuv(rows)
    check('the stripe frame alternates two colours of ONE integer luma (116) '
          'and different chroma',
          int(y_in.min()) == int(y_in.max()) == 116
          and int(u_in[0, 0]) != int(u_in[1, 0]))
    out = SIGT.pal_decode(rows, settings(pal_decoder='DELAY_LINE'), 0)
    y1, u1, v1 = _yuv(out)
    body_u, body_v = u1[:h - 1], v1[:h - 1]
    check('DELAY_LINE on a progressive frame (the line above is y + 1): rows '
          '0..h-2 all carry the same chroma within 1 level (each is the '
          'floor-average of a red and a blue line: vertical colour resolution '
          'halved), the luma within 2 levels of the input, and the top row '
          '(no line above) keeps its own colour',
          int(body_u.max() - body_u.min()) <= 1
          and int(body_v.max() - body_v.min()) <= 1
          and int(np.abs(y1 - y_in).max()) <= 2
          and bool(np.array_equal(out[h - 1], _pal_trip(rows)[h - 1])),
          f'U {int(body_u.min())}..{int(body_u.max())}; V '
          f'{int(body_v.min())}..{int(body_v.max())}; Y max '
          f'{int(np.abs(y1 - y_in).max())}')
    st_i = settings(pal_decoder='DELAY_LINE', interlace='BLEND')
    out_i = SIGT.pal_decode(rows, st_i, 0)
    check('with the frame fielded (interlace on: the same-field neighbour is '
          'y + 2) the stripes SURVIVE: every row is bitwise the plain round '
          "trip, because the line two above carries the row's own colour",
          SIGT.pal_params(st_i, w, 0)['s_rows'] == 2
          and SIGT.pal_params(settings(pal_decoder='DELAY_LINE'), w, 0)['s_rows'] == 1
          and bool(np.array_equal(out_i, _pal_trip(rows)))
          and not np.array_equal(out_i[0], out_i[1]))
    flat = np.zeros((7, 9, 3), np.float32)
    flat[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[40])
    check('a flat frame is bitwise its integer round trip under both decoders '
          '(SIMPLE at 0 degrees is the NONE decoder: c16 = 65536, s16 = 0)',
          bool(np.array_equal(
              SIGT.pal_decode(flat, settings(pal_decoder='DELAY_LINE'), 0),
              _pal_trip(flat)))
          and bool(np.array_equal(
              SIGT.pal_decode(flat, settings(pal_decoder='SIMPLE'), 0),
              _pal_trip(flat)))
          and bool(np.array_equal(
              SIGT.pal_decode(rows, settings(pal_decoder='SIMPLE'), 0),
              _pal_trip(rows))))

    # --- the Hanover law
    red = np.zeros((8, 12, 3), np.float32)
    red[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[90])
    _y, ur, vr = _yuv(red)
    u_true, v_true = int(ur[0, 0]), int(vr[0, 0])
    P = SIGT.pal_params(settings(pal_decoder='SIMPLE', pal_phase_error=20.0),
                        12, 0)

    def rot(sgn):
        return ((P['c16'] * u_true - sgn * P['s16'] * v_true + 32768) >> 16,
                (sgn * P['s16'] * u_true + P['c16'] * v_true + 32768) >> 16)
    ue, ve = rot(+1)
    uo, vo = rot(-1)
    cross = (ue + uo) * v_true - (ve + vo) * u_true
    dot = (ue - uo) * u_true + (ve - vo) * v_true
    mag = float(np.hypot(u_true, v_true))
    check('the Hanover law, SIMPLE at 20 degrees on a flat red: the even-row '
          'and odd-row chroma are mirror images about the true vector (their '
          'sum parallel to it and their difference perpendicular, within 2 '
          'levels) and they DIFFER (+phi on even lines, -phi on odd)',
          (ue, ve) != (uo, vo) and abs(cross) <= 2 * 2 * mag
          and abs(dot) <= 2 * mag,
          f'true ({u_true}, {v_true}); even ({ue}, {ve}); odd ({uo}, {vo}); '
          f'cross {cross}; dot {dot}')
    out = SIGT.pal_decode(red, settings(pal_decoder='SIMPLE',
                                        pal_phase_error=20.0), 0)
    want_e = SIG.U8[np.stack(SIG.pal_back(_y[0, 0], ue, ve), -1)]
    want_o = SIG.U8[np.stack(SIG.pal_back(_y[0, 0], uo, vo), -1)]
    check('and the PICTURE shows them as bars: every even row is the +phi '
          'colour, every odd row the -phi colour, alternating line by line '
          '(two-line-tall pairs on an interlaced tube)',
          bool((out[0::2] == want_e).all()) and bool((out[1::2] == want_o).all())
          and not np.array_equal(want_e, want_o))
    check('a delay-line set reads no phase error: DELAY_LINE with the phase '
          'error dial at 20 is bitwise DELAY_LINE at 0 (the dial is read '
          'under the Simple decoder only)',
          bool(np.array_equal(
              SIGT.pal_decode(rows, settings(pal_decoder='DELAY_LINE',
                                             pal_phase_error=20.0), 0),
              SIGT.pal_decode(rows, settings(pal_decoder='DELAY_LINE'), 0))))

    # --- the crawl law
    grey = np.full((8, 40, 3), SIG.U8[128], np.float32)
    check('the crawl on a grey frame changes nothing (bitwise: it is '
          'proportional to chroma amplitude)',
          bool(np.array_equal(SIGT.pal_decode(grey, settings(pal_crawl=1.0), 3),
                              grey)))
    sat = np.zeros((8, 40, 3), np.float32)
    sat[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[90])
    f1 = SIGT.pal_decode(sat, settings(pal_crawl=1.0), 1)
    f2 = SIGT.pal_decode(sat, settings(pal_crawl=1.0), 2)
    f9 = SIGT.pal_decode(sat, settings(pal_crawl=1.0), 9)
    yc = _yuv(f1)[0]
    check('the crawl on a flat colour at 1.0: the luma varies along the line '
          'and from line to line (3/4 cycle per line), frames 1 and 2 differ, '
          'and frames 1 and 9 are bitwise equal (the eight-field sequence: '
          '57344 * 8 = 7 * 65536)',
          int(yc[0].max() - yc[0].min()) >= 2
          and not np.array_equal(f1[0], f1[1])
          and not np.array_equal(f1, f2) and bool(np.array_equal(f1, f9)),
          f'span {int(yc[0].max() - yc[0].min())}')

    # --- the GPU twin
    rgb, _st = _demo_frame()
    src = _rgba(rgb)
    cases = (
        ('NONE + crawl 1.0', dict(pal_crawl=1.0)),
        ('DELAY_LINE + crawl 0.5', dict(pal_decoder='DELAY_LINE',
                                        pal_crawl=0.5)),
        ('DELAY_LINE, fielded (s_rows 2)', dict(pal_decoder='DELAY_LINE',
                                                interlace='BLEND')),
        ('SIMPLE at 20 degrees', dict(pal_decoder='SIMPLE',
                                      pal_phase_error=20.0)),
        ('SIMPLE at 45 degrees + crawl 2.0', dict(pal_decoder='SIMPLE',
                                                  pal_phase_error=45.0,
                                                  pal_crawl=2.0)),
    )
    for label, kw in cases:
        st = settings(**kw)
        P = SIGT.pal_params(st, W, 7)
        u = {k: int(P[k]) for k in SIGT.PAL_INTS}
        u['resolution'] = (float(W), float(H))
        got, err = run_stage('PAL_DECODE', H, W, u, {
            'source': src, 'u8lut': SIG.u8_image(),
            'sin16': SIG.sin16_image()})
        ref = SIGT.pal_decode(rgb, st, 7)
        d = _dmax(None if got is None else got[..., :3], ref)
        check(f'PAL_DECODE stage, {label}: bitwise signal_tape.pal_decode on '
              'the demo frame at frame 7 (d == 0.0), alpha through, not '
              'vacuous',
              d == 0.0 and float(np.abs(ref - _pal_trip(rgb)).max()) > 0.0
              and bool((got[..., 3] == 1.0).all()),
              str(err) if got is None else f'max {d}')
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', pal_decoder='DELAY_LINE', pal_crawl=0.5))
    d = _dmax(out_g, out_c)
    check('the PAL receiver on the resident frame (no dither): the GPU chain '
          'equals the CPU chain bitwise, PAL_DECODE drawn, 0 readbacks, '
          'nothing live',
          _rec_ok(rec, live, d, dict(stages_has=['PAL_DECODE'], readbacks=0)),
          f'max {d}; {rec}; {len(live)} live')

    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', composite=True, composite_bleed=0.7,
        composite_ringing=0.3, pal_decoder='DELAY_LINE', pal_crawl=0.4))
    d = _dmax(out_g, out_c)
    stg = rec.get('stages') or []
    moved_px = int((np.abs(out_g - out_c).max(-1) > 0.0).sum())         if out_g.shape == out_c.shape else -1
    check("the PAL_TV chain (the composite cable, then the receiver) 'runs "
          "and names its stages': NTSC then PAL_DECODE, nothing live, within "
          '4/255 of the CPU chain at under 5% of the pixels (the NTSC stage '
          'is CLOSE 0.001; the receiver re-quantises it to bytes and the '
          'integer decode carries a flipped level of U into blue at 2.03: '
          'measured 3 levels at 100 of 6912 pixels -- the bitwise claim is '
          'the composite-off case above)',
          'NTSC' in stg and 'PAL_DECODE' in stg
          and stg.index('NTSC') < stg.index('PAL_DECODE') and live == []
          and d <= 4.0 / 255.0 + 1e-6
          and 0 <= moved_px <= 0.05 * out_g.shape[0] * out_g.shape[1],
          f'max {d} ({d * 255.0:.2f} levels) at {moved_px} px; {rec}')

    # --- the refusal by name
    with fakedevice.installed():
        check('pal_decode_refusal is None when the stage may draw',
              CT.pal_decode_refusal(settings(pal_decoder='SIMPLE',
                                             render_device='GPU')) is None)
    check('pal_decode_refusal names the CPU device on the CPU device',
          'CPU' in str(CT.pal_decode_refusal(settings(pal_decoder='SIMPLE'))))
    stx = settings(pal_decoder='SIMPLE', render_device='GPU')
    got, text = _probe_false(stx, lambda fr: chain.pal_decode(fr, stx))
    check("a device whose probe fails makes the chain print 'PAL_DECODE on "
          "the CPU: no driver' and return None",
          got is None and 'PAL_DECODE on the CPU: no driver' in text,
          text.strip())

    # --- the preset
    p = PRESETS['PAL_TV']
    check('the PAL_TV preset decodes like a PAL-D set: the delay line, the '
          'PAL crawl at 0.4 and the NTSC-shaped dot crawl explicitly 0 (one '
          'crawl, the PAL one); its note says so',
          p['settings'].get('pal_decoder') == 'DELAY_LINE'
          and p['settings'].get('pal_crawl') == 0.4
          and p['settings'].get('composite_dot_crawl') == 0.0
          and 'delay-line' in p['note'] and 'eight-field' in p['note'])


# ------------------------------------------------------------ C128 TAPE

def _tape_st(**kw):
    """A quiet deck: no noise, no head switch, no dropouts unless asked."""
    base = dict(tape='VHS', tape_noise=0.0, tape_head_switch=False,
                tape_dropouts=0.0)
    base.update(kw)
    return settings(**base)


def _centroid(weight):
    """The weight-averaged column of a (h, w) non-negative plane."""
    wsum = weight.sum(0).astype(np.float64)
    return float((wsum * np.arange(weight.shape[1])).sum() / max(wsum.sum(), 1e-9))


def test_r251_c128_tape():
    """The tape path: the bandwidth law per generation and per format, the
    honest identity at 96 px, the Y/C delay to the right (and none on
    Betacam), determinism, the head-switch band, dropouts, the legal
    range, the signed hash, the two-draw GPU twin bitwise, the loop-bound
    refusal by name, the presets."""
    st0 = RenderSettings()
    check('tape_on is False on default settings and every table format '
          'switches it on (nine formats beside Off)',
          not SIGT.tape_on(st0) and len(SIGT.TAPE_TABLE) == 9
          and all(SIGT.tape_on(settings(tape=k)) for k in SIGT.TAPE_TABLE))

    # --- the bandwidth law, on a 320-wide ACHROMATIC bar (R = G = B
    #     everywhere: cb601 and cr601 of a grey are exactly 128, so the
    #     chroma planes are constant and only the luma FIR speaks)
    v = np.arange(256, dtype=np.int32)
    check('cb601 and cr601 of every grey 0..255 are exactly 128 (an '
          'achromatic frame has constant chroma planes: no FIR or delay can '
          'move them)',
          bool((SIG.cb601(v, v, v) == 128).all())
          and bool((SIG.cr601(v, v, v) == 128).all()))
    h, w = 72, 320
    bar = np.full((h, w, 3), SIG.U8[64], np.float32)
    bar[:, 150:170] = SIG.U8[200]

    def peak(st):
        return _peak_grad(_ycc(SIGT.tape_path(bar, st, 0, 0))[0])
    g1, g2, g4 = (peak(_tape_st(tape_generations=n)) for n in (1, 2, 4))
    gs = peak(_tape_st(tape='SVHS'))
    g0 = _peak_grad(_ycc(bar)[0])
    check('the bandwidth law at 320 px (VHS, sigma 0.5 / 0.71 / 1.0 px): the '
          'steepest luma edge falls STRICTLY with generations 1, 2, 4 (every '
          'dub re-applies the low-pass), and S-VHS at one generation (400 '
          'TVL, sigma 0.3 px: this edge survives whole) keeps a steeper edge '
          'than VHS (240 TVL)',
          g0 >= gs > g1 > g2 > g4 > 0, f'{g0} >= {gs} > {g1} > {g2} > {g4}')
    P1 = SIGT.tape_params(_tape_st(), h, w, 0, 0)
    P4 = SIGT.tape_params(_tape_st(tape_generations=4), h, w, 0, 0)
    check('every tap set sums to 65536 exactly (mean-preserving), the radius '
          'is ceil(3 sigma), and N generations is one Gaussian of sigma * '
          'sqrt(N) (N = 4 doubles the radius)',
          int(P1['taps_y'].sum()) == 65536 and int(P1['taps_c'].sum()) == 65536
          and int(P4['taps_y'].sum()) == 65536
          and P1['r_y'] == 2 and P4['r_y'] == 3 and P1['r_c'] == 9
          and P4['r_c'] == 18,
          f"r_y {P1['r_y']}/{P4['r_y']} r_c {P1['r_c']}/{P4['r_c']}")
    bar96 = np.zeros((24, 96, 3), np.float32)     # black, a white bar
    bar96[:, 40:60] = 1.0
    a1 = SIGT.tape_path(bar96, _tape_st(tape_generations=1), 0, 0)
    a2 = SIGT.tape_path(bar96, _tape_st(tape_generations=2), 0, 0)
    a4 = SIGT.tape_path(bar96, _tape_st(tape_generations=4), 0, 0)
    t1 = SIGT.tape_params(_tape_st(tape_generations=1), 24, 96, 0, 0)['taps_y']
    check('the honest law at 96 px, on the ACHROMATIC bar: VHS at one and '
          'two generations is bitwise the same picture (sigma 0.15 / 0.21 px: '
          'the luma FIR is the identity at that size, taps (0, 65536, 0); a '
          'COLOURED bar would differ through the chroma FIR and the delay), '
          'while four generations (taps (251, 65034, 251)) differs',
          bool(np.array_equal(a1, a2)) and not np.array_equal(a1, a4)
          and t1.tolist() == [0, 65536, 0],
          f'taps {t1.tolist()}')
    flat = np.zeros((9, 40, 3), np.float32)
    flat[...] = (SIG.U8[200], SIG.U8[90], SIG.U8[40])
    y, cb, cr = _ycc(flat)
    trip = SIG.U8[np.stack(SIG.yuv601_decode(y, cb, cr), -1)]
    check('a flat frame comes out flat and bitwise its BT.601 round trip, at '
          'one generation and at eight (the FIRs sum to 65536)',
          bool(np.array_equal(SIGT.tape_path(flat, _tape_st(), 0, 0), trip))
          and bool(np.array_equal(
              SIGT.tape_path(flat, _tape_st(tape_generations=8), 0, 0), trip)))

    # --- the Y/C delay law: a red bar on grey at 96 px
    rbar = np.full((24, 96, 3), SIG.U8[128], np.float32)
    rbar[:, 40:52] = (SIG.U8[230], SIG.U8[26], SIG.U8[26])
    yg, _cbg, crg = _ycc(rbar)
    y_grey, cr_grey = int(yg[0, 0]), int(crg[0, 0])

    def offset(fmt):
        st = _tape_st(tape=fmt)
        yo, _cbo, cro = SIGT._tape_fir(SIG.to_u8(rbar),
                                       SIGT.tape_params(st, 24, 96, 0, 0))
        return (_centroid(np.abs(cro - cr_grey)) - _centroid(np.abs(yo - y_grey)),
                SIGT.tape_params(st, 24, 96, 0, 0)['delay'])
    off_v, dly_v = offset('VHS')
    off_b, dly_b = offset('BETACAM')
    check('the Y/C delay law: with VHS the chroma lands to the RIGHT of the '
          'luma by the delay (0.6 us = 1 px at 96) within 1 px; with Betacam '
          '(component, no colour-under) the two centroids agree within 0.5 px',
          dly_v == 1 and dly_b == 0 and abs(off_v - dly_v) <= 1.0
          and off_v > 0.5 and abs(off_b) <= 0.5,
          f'VHS {off_v:+.3f} px (delay {dly_v}); Betacam {off_b:+.3f} px')
    check('and the delay compounds per dub: three generations of VHS at 640 '
          'px delay the chroma by round(3 * 0.6 * 640 / 52.66) = 22 px',
          SIGT.tape_params(_tape_st(tape_generations=3), 480, 640, 0, 0)[
              'delay'] == 22)

    # --- determinism, the head-switch band, dropouts, the ranges
    rgb, _st = _demo_frame()
    noisy = dict(tape='VHS', tape_noise=0.1, tape_head_switch=False)
    a = SIGT.tape_path(rgb, settings(**noisy), 7, 3)
    check('the noise is a hash: the same (frame, seed) is bitwise the same '
          'picture; frame + 1 differs; seed + 1 differs',
          bool(np.array_equal(a, SIGT.tape_path(rgb, settings(**noisy), 7, 3)))
          and not np.array_equal(a, SIGT.tape_path(rgb, settings(**noisy), 8, 3))
          and not np.array_equal(a, SIGT.tape_path(rgb, settings(**noisy), 7, 4)))
    on = SIGT.tape_path(rgb, settings(tape='VHS', tape_noise=0.1), 7, 3)
    hr = SIGT.tape_params(settings(tape='VHS'), H, W, 7, 3)['head_rows']
    check('the head-switch band: with the head switch on, the rows above the '
          'band (72 rows: round(7 * 72 / 480) = 1 row at the BOTTOM, row 0) '
          'are bitwise the head-switch-off picture and the band row differs '
          '(four times the luma noise, torn sideways)',
          hr == 1 and bool(np.array_equal(on[hr:], a[hr:]))
          and not np.array_equal(on[:hr], a[:hr])
          and SIGT.tape_params(settings(tape='VHS'), 480, 640, 7, 3)[
              'head_rows'] == 7)
    dr = SIG.to_u8(SIGT.tape_path(rgb, _tape_st(tape_dropouts=40.0), 7, 3))
    white = (dr == 255).all(-1)
    best = 0
    for row in white:
        run = 0
        for px in row:
            run = run + 1 if px else 0
            best = max(best, run)
    none = SIG.to_u8(SIGT.tape_path(rgb, _tape_st(), 7, 3))
    check('dropouts: 40 per frame at 72 rows leaves at least one row with a '
          'run of >= 8 consecutive white pixels (legal white 235 / 128 / 128 '
          'decodes to 255), and none without the dial',
          best >= 8 and SIG.yuv601_decode(235, 128, 128)[0] == 255
          and not bool((none == 255).all(-1).any()), f'longest run {best}')
    Pn = SIGT.tape_params(settings(tape='VHS', tape_noise=0.5,
                                   tape_generations=8, tape_dropouts=10.0),
                          H, W, 7, 3)
    Yn, Cbn, Crn = SIGT._tape_ycc(*SIGT._tape_fir(SIG.to_u8(rgb), Pn), Pn, H, W)
    check('the range law: with the noise at 0.5 and eight generations every '
          'luma stays in 16..235 and every chroma in 16..240 (a deck\'s FM '
          'demodulator delivers nothing below black or above white)',
          int(Yn.min()) >= 16 and int(Yn.max()) <= 235
          and int(Cbn.min()) >= 16 and int(Cbn.max()) <= 240
          and int(Crn.min()) >= 16 and int(Crn.max()) <= 240,
          f'Y {int(Yn.min())}..{int(Yn.max())} Cb {int(Cbn.min())}..'
          f'{int(Cbn.max())} Cr {int(Crn.min())}..{int(Crn.max())}')
    xx, yy = np.meshgrid(np.arange(32, dtype=np.uint32),
                         np.arange(32, dtype=np.uint32))
    hs = SIG.hash_s16(xx, yy, 12345)
    check('the signed-hash law: hash_s16 is int32 and carries values below 0 '
          '(a uint32 field minus 32768 would wrap)',
          hs.dtype == np.int32 and int(hs.min()) < 0 and int(hs.max()) > 0)

    # --- the GPU twin: TAPE_FIR, TAPE_OUT, and the two together
    twin_cases = (
        ('the demo frame, VHS x3, noise 0.1, dropouts 20', rgb,
         dict(tape='VHS', tape_generations=3, tape_noise=0.1,
              tape_dropouts=20.0)),
        ('a 320 x 72 random frame, U-matic x2 (the tear moves: head_j != 0)',
         np.random.default_rng(128).random((72, 320, 3)).astype(np.float32),
         dict(tape='UMATIC', tape_generations=2, tape_noise=0.05,
              tape_dropouts=30.0)),
        ('the demo frame, Betacam SP (no delay)', rgb,
         dict(tape='BETACAM_SP', tape_noise=0.02)),
    )
    for label, img, kw in twin_cases:
        hh, ww = img.shape[:2]
        st = settings(**kw)
        P = SIGT.tape_params(st, hh, ww, 7, 3)
        res = (float(ww), float(hh))
        fir, err = run_stage('TAPE_FIR', hh, ww, {
            'resolution': res, 'r_y': int(P['r_y']), 'r_c': int(P['r_c']),
            'delay': int(P['delay'])},
            {'source': _rgba(img),
             'taps': SIGT.taps_image(P['taps_y'], P['taps_c'])})
        Yf, Cbf, Crf = SIGT._tape_fir(SIG.to_u8(img), P)
        planes = np.stack([Yf, Cbf, Crf], -1).astype(np.float32)
        d1 = _dmax(None if fir is None else fir[..., :3], planes)
        u = {k: int(P[k]) for k in SIGT.TAPE_OUT_INTS}
        u['resolution'] = res
        out, err2 = run_stage('TAPE_OUT', hh, ww, u, {
            'source': _rgba(planes), 'u8lut': SIG.u8_image()})
        ref = SIG.U8[np.stack(SIGT._tape_out(Yf, Cbf, Crf, P, hh, ww, 7), -1)]
        d2 = _dmax(None if out is None else out[..., :3], ref)
        whole = SIGT.tape_path(img, st, 7, 3)
        check(f'TAPE_FIR then TAPE_OUT, {label}: the FIR draw is bitwise '
              '_tape_fir\'s three integer planes (as floats) and the OUT '
              'draw over them is bitwise _tape_out through the byte table '
              '(both d == 0.0), which IS tape_path; alpha through',
              d1 == 0.0 and d2 == 0.0 and bool(np.array_equal(ref, whole))
              and bool((fir[..., 3] == 1.0).all())
              and bool((out[..., 3] == 1.0).all())
              and float(np.abs(whole - img).max()) > 0.0,
              str(err or err2) if (fir is None or out is None)
              else f"FIR {d1}; OUT {d2}; head_j {P['head_j']}")
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', tape='VHS', tape_generations=3, tape_dropouts=20.0))
    d = _dmax(out_g, out_c)
    check('the tape on the resident frame (VHS x3, dropouts 20, no dither, '
          "depth 24): the GPU chain equals the CPU chain bitwise, recorded as "
          "TAPE after the QUANT, 0 readbacks, the render's frame inherited, "
          'nothing live (the FIR target returned to the pool)',
          _rec_ok(rec, live, d, dict(
              resident=True, stages=['DISPLAY', 'QUANT', 'TAPE'],
              readbacks=0, uploads=0)),
          f'max {d}; {rec}; {len(live)} live')
    out_g, out_c, rec, live = gpu_road(settings(
        dither='NONE', tape='VHS', tape_generations=2, signal='SVIDEO',
        pal_decoder='DELAY_LINE', pal_crawl=0.4))
    d = _dmax(out_g, out_c)
    check('the whole chain in the hardware\'s order -- the tape, the S-Video '
          'cable, the PAL receiver -- stays resident and bitwise: TAPE, '
          'CABLE_CHROMA, PAL_DECODE in that order, 0 readbacks',
          _rec_ok(rec, live, d, dict(
              resident=True,
              stages=['DISPLAY', 'QUANT', 'TAPE', 'CABLE_CHROMA',
                      'PAL_DECODE'], readbacks=0, uploads=0)),
          f'max {d}; {rec}; {len(live)} live')

    # --- the refusal by name
    with fakedevice.installed():
        deep = settings(tape='VHS', tape_generations=8, render_device='GPU')
        check("tape_refusal names the 'loop bound' for VHS at eight "
              'generations and 1280 px (chroma radius 102 px > 96), and is '
              'None at 96 px and with no width on the 96-wide settings',
              'loop bound' in str(CT.tape_refusal(deep, w=1280))
              and CT.tape_refusal(deep, w=96) is None
              and CT.tape_refusal(deep) is None,
              str(CT.tape_refusal(deep, w=1280)))
        wide = np.random.default_rng(3).random((4, 1280, 3)).astype(np.float32)
        deep._frame_gpu_shaded = True
        fr = chain.Frame(rgb=wide)
        got, text = _capture(lambda: chain.tape(fr, deep, frame_no=7, seed=3))
        fr.release()
        check("and the chain says so: 'TAPE on the CPU: the tape's low-pass "
              "radius 102 px exceeds the shader loop bound', returning None "
              '(the CPU stage runs that frame)',
              got is None and 'TAPE on the CPU' in text and '102 px' in text
              and 'loop bound' in text, text.strip())
    check('tape_refusal names the CPU device on the CPU device',
          'CPU' in str(CT.tape_refusal(settings(tape='VHS'))))
    stx = settings(tape='VHS', render_device='GPU')
    got, text = _probe_false(stx, lambda fr: chain.tape(fr, stx))
    check("a device whose probe fails makes the chain print 'TAPE on the "
          "CPU: no driver' and return None",
          got is None and 'TAPE on the CPU: no driver' in text, text.strip())
    st = settings(tape='VHS', tape_generations=3)
    sc = demo_scene(st, with_texture=False)
    img = R.render(sc, st)
    st_c = st.copy()
    ref = PO.process(img, st_c, **post_kw(sc, st_c))
    st.render_device = 'GPU'
    from ..gpu import device as DEV
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

    # --- the presets
    p = PRESETS['VHS']['settings']
    q = PRESETS['CEL_VHS_80S']['settings']
    check('the VHS presets run the tape path under a normal composite cable: '
          'VHS and CEL_VHS_80S record on VHS at one generation, and their '
          'composite bleed drops from 2.0 to 0.5 (the tape carries the chroma '
          'smear; the cable keeps its own)',
          p.get('tape') == 'VHS' and p.get('tape_generations') == 1
          and p.get('tape_noise') == 0.03 and p.get('composite_bleed') == 0.5
          and q.get('tape') == 'VHS' and q.get('tape_generations') == 1
          and q.get('composite_bleed') == 0.5)
    for key in ('VHS', 'CEL_VHS_80S'):
        stp = RenderSettings()
        apply_preset(stp, key)
        check(f'the {key} preset draws on the GPU at its own width (the '
              'low-pass radius is inside the 96-tap loop)',
              CT._loop_bound_refusal('tape', max(
                  SIGT.tape_params(stp, int(stp.resolution_y),
                                   int(stp.resolution_x), 0, 0)[k]
                  for k in ('r_y', 'r_c'))) is None)


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
    print(f'\n{len(FAILS)} failure(s)' if FAILS
          else '\nall post-tape checks passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
