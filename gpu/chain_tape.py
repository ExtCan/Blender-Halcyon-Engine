"""R251: the post-signal pack's tape / cable / receiver / chroma chain
functions (SIG-2: the GPU road of core/signal_tape's stages), bound onto
gpu/chain's namespace by the integrator's bottom-of-file import so
`post.process` finds them by name.

The pack's ONE signature: `def <name>(frame, st, frame_no=0, seed=0,
**_k)` returning the Frame (the draw recorded) or None (the CPU one runs,
the reason printed once through `chain._warn` as `<STAGE> on the CPU:
{why}` -- never a silent None), and `<name>_refusal(st, w=None, h=None)
-> str | None` (a size it needs defaults to `st.resolution_x/y`).
`chain` is imported LAZILY inside each function (see gpu/chain_signal).

Single-draw stages go through `chain.try_stage`; the tape's two draws
(TAPE_FIR into a private target of integer planes, TAPE_OUT from it)
copy `chain.ntsc`'s shape and are recorded as TAPE.
"""

from . import device
from ..core import signal_era as SIG
from ..core import signal_tape as SIGT


def _u8lut():
    return device.upload_cached(('sig_u8',), SIG.u8_image)


def _sin16():
    return device.upload_cached(('sig_sin16',), SIG.sin16_image)


def _generic_refusal(st):
    """The device / gate refusals every stage shares, or None."""
    from . import chain as _chain
    ok, why = _chain.available(st)
    if not ok:
        return str(why)
    return None


def _res(st, w, h):
    return (int(st.resolution_x) if w is None else int(w),
            int(st.resolution_y) if h is None else int(h))


# ----------------------------------------------------- C131 CHROMA_SITE

def chroma_site_refusal(st, w=None, h=None):
    if not (SIGT.chroma_site_on(st) or SIGT.xfb_on(st)):
        return None
    return _generic_refusal(st)


def chroma_site(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not (SIGT.chroma_site_on(st) or SIGT.xfb_on(st)):
        return None
    try:
        u8 = _u8lut()
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'CHROMA_SITE on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return _chain.try_stage('CHROMA_SITE', frame, st, {
        'resolution': (float(w), float(h)),
        'fmt': int(SIGT.SITE_FMT[str(st.chroma_format)]),
        'interp': 1 if str(st.chroma_upsample) == 'LINEAR' else 0},
        extra_binds={'u8lut': u8})


# ------------------------------------------- C129 CABLE_CHROMA / CABLE_RF

def _loop_bound_refusal(what, radius):
    if radius > SIGT.FIR_BOUND:
        return (f"{what} radius {radius} px exceeds the shader loop bound at "
                'this resolution (the CPU runs it)')
    return None


def cable_chroma_refusal(st, w=None, h=None):
    if str(getattr(st, 'signal', 'RGB')) not in SIGT.CABLE_TABLE:
        return None
    w, h = _res(st, w, h)
    P = SIGT.cable_params(st, w)
    why = _loop_bound_refusal("the cable's chroma", max(P['r1'], P['r2']))
    return why or _generic_refusal(st)


def cable_chroma(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SIGT.cable_chroma_on(st):
        return None
    h, w = frame.height, frame.width
    P = SIGT.cable_params(st, w)
    why = _loop_bound_refusal("the cable's chroma", max(P['r1'], P['r2']))
    if why:
        _chain._warn(f'CABLE_CHROMA on the CPU: {why}')
        return None
    try:
        u8 = _u8lut()
        taps = device.upload_cached(
            ('sig_cable_taps', str(st.signal), int(w)),
            lambda: SIGT.taps_image(P['taps1'], P['taps2']))
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'CABLE_CHROMA on the CPU: {exc}')
        return None
    return _chain.try_stage('CABLE_CHROMA', frame, st, {
        'resolution': (float(w), float(h)),
        'space': int(P['space']), 'r1': int(P['r1']), 'r2': int(P['r2'])},
        extra_binds={'taps': taps, 'u8lut': u8})


def cable_rf_refusal(st, w=None, h=None):
    if not SIGT.cable_rf_on(st):
        return None
    w, h = _res(st, w, h)
    P = SIGT.rf_params(st, w, 0, 0)
    why = _loop_bound_refusal("the RF modulator's luma", P['r_fir'])
    return why or _generic_refusal(st)


def cable_rf(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SIGT.cable_rf_on(st):
        return None
    h, w = frame.height, frame.width
    P = SIGT.rf_params(st, w, int(frame_no), int(seed))
    why = _loop_bound_refusal("the RF modulator's luma", P['r_fir'])
    if why:
        _chain._warn(f'CABLE_RF on the CPU: {why}')
        return None
    try:
        u8 = _u8lut()
        sin16 = _sin16()
        taps = device.upload_cached(
            ('sig_rf_taps', float(st.rf_bandwidth), int(w)),
            lambda: SIGT.taps_image(P['taps']))
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'CABLE_RF on the CPU: {exc}')
        return None
    uniforms = {k: int(P[k]) for k in SIGT.RF_INTS}
    uniforms['resolution'] = (float(w), float(h))
    return _chain.try_stage('CABLE_RF', frame, st, uniforms,
                            extra_binds={'taps': taps, 'sin16': sin16,
                                         'u8lut': u8})


# ------------------------------------------------------ C130 PAL_DECODE

def pal_decode_refusal(st, w=None, h=None):
    if not SIGT.pal_decode_on(st):
        return None
    return _generic_refusal(st)


def pal_decode(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SIGT.pal_decode_on(st):
        return None
    h, w = frame.height, frame.width
    P = SIGT.pal_params(st, w, int(frame_no))
    try:
        u8 = _u8lut()
        sin16 = _sin16()
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'PAL_DECODE on the CPU: {exc}')
        return None
    uniforms = {k: int(P[k]) for k in SIGT.PAL_INTS}
    uniforms['resolution'] = (float(w), float(h))
    return _chain.try_stage('PAL_DECODE', frame, st, uniforms,
                            extra_binds={'sin16': sin16, 'u8lut': u8})


# ------------------------------------------------------------ C128 TAPE

def tape_refusal(st, w=None, h=None):
    if not SIGT.tape_on(st):
        return None
    w, h = _res(st, w, h)
    P = SIGT.tape_params(st, h, w, 0, 0)
    why = _loop_bound_refusal("the tape's low-pass", max(P['r_y'], P['r_c']))
    return why or _generic_refusal(st)


def tape_taps_key(st, w):
    return ('sig_tape_taps', str(st.tape),
            min(max(int(st.tape_generations), 1), 8), int(w))


def tape(frame, st, frame_no=0, seed=0, **_k):
    """TAPE_FIR (the three filtered integer planes, a private target) then
    TAPE_OUT (tear, noise, dropouts, decode) in chain.ntsc's shape,
    recorded as TAPE."""
    from . import chain as _chain
    if not SIGT.tape_on(st):
        return None
    h, w = frame.height, frame.width
    P = SIGT.tape_params(st, h, w, int(frame_no), int(seed))
    why = _loop_bound_refusal("the tape's low-pass", max(P['r_y'], P['r_c']))
    if why:
        _chain._warn(f'TAPE on the CPU: {why}')
        return None
    if 'TAPE_FIR' not in _chain.ENABLED or 'TAPE_OUT' not in _chain.ENABLED:
        _chain._warn('TAPE on the CPU: TAPE_FIR / TAPE_OUT are not enabled')
        return None
    ok, why = _chain.available(st)
    if not ok:
        _chain._warn(f'TAPE on the CPU: {why}')
        return None
    fir_sh, err = device.compile_stage('TAPE_FIR', _chain.STAGES['TAPE_FIR'])
    out_sh, err2 = device.compile_stage('TAPE_OUT', _chain.STAGES['TAPE_OUT'])
    if fir_sh is None or out_sh is None:
        _chain._warn(f'TAPE on the CPU: {err or err2}')
        return None
    try:
        u8 = _u8lut()
        taps = device.upload_cached(
            tape_taps_key(st, w),
            lambda: SIGT.taps_image(P['taps_y'], P['taps_c']))
        src_tex = frame.source()
        src_target = frame.target
        frame.target = None            # the FIR must not free it
        fir_t = device.Target(w, h)
        out_t = device.Target(w, h)
        res = (float(w), float(h))
        out_u = {k: int(P[k]) for k in SIGT.TAPE_OUT_INTS}
        out_u['resolution'] = res
        draws = [
            (fir_sh, {'resolution': res, 'r_y': int(P['r_y']),
                      'r_c': int(P['r_c']), 'delay': int(P['delay'])},
             {'source': src_tex, 'taps': taps}, fir_t, 'NONE', False, None),
            (out_sh, out_u,
             {'source': device.target_texture(fir_t), 'u8lut': u8}, out_t,
             'NONE', False, None),
        ]
        frame.pending.extend(draws)
        frame.retired.extend([fir_t] + ([src_target] if src_target is not None
                                        else []))
        frame.target = out_t
        frame._source_tex = None
        frame.stages.append('TAPE')
        return frame
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'TAPE fell back to the CPU: {type(exc).__name__}: {exc}')
        return None
