"""R251: the post-signal pack's chain functions (the GPU road of
core/signal_era's stages), bound onto gpu/chain's namespace by the
integrator's bottom-of-file import so `post.process` finds them by name.

ONE signature for every stage: `def <name>(frame, st, frame_no=0, seed=0,
**_k)` returning the Frame (the draw recorded) or None (the CPU one
runs, the reason printed once through `chain._warn`), and
`<name>_refusal(st, w=None, h=None) -> str | None` (a size it needs
defaults to `st.resolution_x/y`). `chain` is imported LAZILY inside each
function: the integrator binds these functions onto the chain namespace
at the end of gpu/chain.py, and the multi-draw orchestrator must test
`NAME in _chain.ENABLED` -- the chain MODULE's global, which the self
test rebinds for its orchestrator measurements -- never a snapshot.

Single-draw stages go through `chain.try_stage` (recorded under their
own NAME in `frame.stages`); the four-draw VOODOO1 line copies
`chain.ntsc`'s shape (own targets, the draws appended to
`frame.pending`, the read targets to `frame.retired`, recorded as
VIDEO_FILTER). CRTC_BLEND in PREVIOUS_FRAME mode reads its output back
by name after the draw (the CRTC keeps it for the next frame's RC2).
"""

import numpy as np

from . import device
from ..core import signal_era as SIG
from ..core import wear as WEAR


def _u8lut():
    return device.upload_cached(('sig_u8',), SIG.u8_image)


def _generic_refusal(st):
    """The device / gate refusals every stage shares, or None."""
    from . import chain as _chain
    ok, why = _chain.available(st)
    if not ok:
        return str(why)
    return None


# ------------------------------------------------------------- C005 VI

_VI_GAMMA_MODES = {'NONE': 0, 'GAMMA': 1, 'GAMMA_DITHER': 2, 'DITHER_ONLY': 3}


def vi_filter_refusal(st, w=None, h=None):
    """None when the VI stage may draw (the depth no-op is
    signal_era.vi_filter_why, printed by the CPU predicate, both roads
    inert), else the generic device reason."""
    why = SIG.vi_filter_why(st)
    if why:
        return why
    return _generic_refusal(st)


def vi_filter(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SIG.vi_filter_on(st):
        return None
    from ..core.post import DEPTH_BITS
    bits = DEPTH_BITS[str(st.color_depth)]
    mode = _VI_GAMMA_MODES.get(str(st.vi_gamma), 0)
    try:
        u8 = _u8lut()
        if mode == 1:
            glut = device.upload_cached(('sig_vi_gamma',), SIG.vi_gamma_image)
        elif mode == 2:
            glut = device.upload_cached(('sig_vi_gdither',),
                                        SIG.vi_gdither_image)
        else:
            glut = u8
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'VI on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return _chain.try_stage('VI', frame, st, {
        'resolution': (float(w), float(h)),
        'levels': tuple(float((1 << b) - 1) for b in bits),
        'shift_r': int(8 - bits[0]), 'shift_g': int(8 - bits[1]),
        'shift_b': int(8 - bits[2]),
        'dedither': 1 if bool(st.vi_dither_filter) else 0,
        'gamma_mode': int(mode),
        'key': int(WEAR._sheet_key(int(frame_no), int(seed), 41))},
        extra_binds={'u8lut': u8, 'glut': glut})


# ----------------------------------------------------- C019 COPY_FILTER

def copy_filter_refusal(st, w=None, h=None):
    if not SIG.copy_filter_on(st):
        return None
    return _generic_refusal(st)


def copy_filter(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SIG.copy_filter_on(st):
        return None
    U, M, L = SIG.COPY_TAPS[str(st.copy_filter)]
    try:
        u8 = _u8lut()
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'COPY_FILTER on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return _chain.try_stage('COPY_FILTER', frame, st, {
        'resolution': (float(w), float(h)),
        'tap_u': int(U), 'tap_m': int(M), 'tap_l': int(L)},
        extra_binds={'u8lut': u8})


# ------------------------------------------------------ C033 CRTC_BLEND

def crtc_blend_refusal(st, w=None, h=None):
    if not SIG.crtc_blend_on(st):
        return None
    return _generic_refusal(st)


def crtc_state():
    """The CRTC's kept output (the module state in core/signal_era), for
    the console and the tests: (frame, key) or None."""
    s = SIG.CRTC_STATE
    if not s:
        return None
    return (s.get('frame'), s.get('key'))


def crtc_blend(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SIG.crtc_blend_on(st):
        return None
    h, w = frame.height, frame.width
    A = min(max(int(st.crtc_alpha), 0), 255)
    prev8, why = SIG.crtc_rc2(st, frame_no, h, w)
    if why:
        SIG._once(why)
    bg8 = SIG.crtc_bg8(st, 1, 1)[0, 0]
    previous = str(st.crtc_blend) == 'PREVIOUS_FRAME' and why is None
    try:
        u8 = _u8lut()
        if previous:
            img = np.concatenate([SIG.U8[np.asarray(prev8, np.int32)],
                                  np.ones((h, w, 1), np.float32)], 2)
            prev_tex = device.upload(img)
        else:
            prev_tex = u8
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'CRTC_BLEND on the CPU: {exc}')
        return None
    got = _chain.try_stage('CRTC_BLEND', frame, st, {
        'resolution': (float(w), float(h)),
        'bg': tuple(float(SIG.U8[int(v)]) for v in bg8),
        'alpha': int(A), 'mode': 0 if previous else 1,
        'inv255': float(np.float32(1.0 / 255.0))},
        extra_binds={'u8lut': u8, 'prev': prev_tex})
    if got is None:
        return None
    if str(st.crtc_blend) == 'PREVIOUS_FRAME':
        rgb = frame.down('CRTC_BLEND',
                         "the CRTC output is kept for the next frame's RC2")
        SIG.crtc_store(st, frame_no, SIG.to_u8(rgb))
    return frame


# ---------------------------------------------------- C069 VIDEO_FILTER

def video_filter_refusal(st, w=None, h=None):
    why = SIG.video_filter_why(st)
    if why:
        return why
    if not SIG.video_filter_on(st):
        return None
    return _generic_refusal(st)


def video_filter(frame, st, frame_no=0, seed=0, **_k):
    """VOODOO1: four VOODOO_LINE draws (left, left, left, right) in
    chain.ntsc's shape, recorded as VIDEO_FILTER; VOODOO2: one
    VOODOO_LOOK draw through try_stage."""
    from . import chain as _chain
    if not SIG.video_filter_on(st):
        return None
    cap = min(max(int(st.video_filter_threshold), 0), 255)
    h, w = frame.height, frame.width
    try:
        u8 = _u8lut()
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'VIDEO_FILTER on the CPU: {exc}')
        return None
    if str(st.video_filter) == 'VOODOO2':
        return _chain.try_stage('VOODOO_LOOK', frame, st, {
            'resolution': (float(w), float(h)),
            'cap': int(cap), 'cap32': int(min(cap, 32)),
            'inv5': float(np.float32(0.2))},
            extra_binds={'u8lut': u8})
    if 'VOODOO_LINE' not in _chain.ENABLED:
        _chain._warn('VIDEO_FILTER on the CPU: VOODOO_LINE is not enabled')
        return None
    ok, why = _chain.available(st)
    if not ok:
        _chain._warn(f'VIDEO_FILTER on the CPU: {why}')
        return None
    line_sh, err = device.compile_stage('VOODOO_LINE',
                                        _chain.STAGES['VOODOO_LINE'])
    if line_sh is None:
        _chain._warn(f'VIDEO_FILTER on the CPU: {err}')
        return None
    try:
        src_tex = frame.source()
        src_target = frame.target
        frame.target = None            # the passes must not free it
        a = device.Target(w, h)
        b = device.Target(w, h)
        targets = [a, b]
        cur_tex = src_tex
        draws = []
        for step, direction in enumerate((-1, -1, -1, 1)):
            tgt = targets[step % 2]
            draws.append((line_sh,
                          {'resolution': (float(w), float(h)),
                           'direction': int(direction), 'cap': int(cap),
                           'expand': 1 if step == 0 else 0},
                          {'source': cur_tex, 'u8lut': u8}, tgt, 'NONE',
                          False, None))
            cur_tex = device.target_texture(tgt)
        out_t = targets[3 % 2]
        frame.pending.extend(draws)
        frame.retired.extend([targets[0]]
                             + ([src_target] if src_target is not None
                                else []))
        frame.target = out_t
        frame._source_tex = None
        frame.stages.append('VIDEO_FILTER')
        return frame
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'VIDEO_FILTER fell back to the CPU: '
                     f'{type(exc).__name__}: {exc}')
        return None
