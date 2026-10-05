"""R251: the post-signal pack's codec and optical-printer chain functions
(the GPU road of core/signal_codec's stages), bound onto gpu/chain's
namespace by the integrator's bottom-of-file import so `post.process`
finds them by name: `matte_glow`, `mpeg1`, `smacker`.

ONE signature for every stage: `def <name>(frame, st, frame_no=0, seed=0,
**_k)` returning the Frame (the draws recorded) or None (the CPU one
runs, the reason printed once through `chain._warn`), and
`<name>_refusal(st, w=None, h=None) -> str | None` (a size it needs
defaults to `st.resolution_x/y`). `chain` is imported LAZILY inside each
function: the integrator binds these functions onto the chain namespace
at the end of gpu/chain.py, and the orchestrators must test
`NAME in _chain.ENABLED` -- the chain MODULE's global, which the self
test rebinds for its orchestrator measurements -- never a snapshot.

    matte_glow  `2 * passes + 1` draws in chain.ntsc's shape (one
                scratch target the horizontal blurs share, one target
                per pass, the sum), recorded MATTE_GLOW; the gel plane
                arrives as the keyword `gel` (post.process's `_gpu`
                closure forwards keywords)
    mpeg1       five draws: four at the PADDED size (its own target
                pair, so every intermediate the CPU's padded arrays
                hold exists on the GPU too) and the decode at the
                frame's, recorded MPEG1
    smacker     one draw through try_stage; the codec palette is fitted
                on the CPU from ONE named readback when the lock does
                not hold it yet
"""

import numpy as np

from . import device
from ..core import palette as PA
from ..core import signal_codec as SC
from ..core import signal_era as SIG


def _generic_refusal(st):
    """The device / gate refusals every stage shares, or None."""
    from . import chain as _chain
    ok, why = _chain.available(st)
    if not ok:
        return str(why)
    return None


def _compile_all(names, label):
    """Compile every component stage first: (shaders, None) or
    (None, why). A stage missing from the chain's ENABLED refuses by
    name."""
    from . import chain as _chain
    out = []
    for nm in names:
        if nm not in _chain.ENABLED:
            return None, f'{nm} is not enabled'
        sh, err = device.compile_stage(nm, _chain.STAGES[nm])
        if sh is None:
            return None, str(err)
        out.append(sh)
    return out, None


# ------------------------------------------------- C134: the matte glow

def matte_glow_refusal(st, w=None, h=None, gel=None, need_gel=False):
    """None when the matte may draw; else the reason. `h` defaults to
    the settings' own height: the diffusion radii are in pixels at 1080
    lines, and the blur stage's literal loop covers 160 taps a side."""
    if not SC.matte_glow_on(st):
        return None
    h = int(st.resolution_y) if h is None else int(h)
    P = SC.matte_params(st, h)
    for k, R in enumerate(P.radii, 1):
        if R > SC.MATTE_LOOP:
            return (f"the matte's diffusion radius {R} px (pass {k}) exceeds "
                    'the shader loop bound at this resolution')
    if need_gel and gel is None:
        return 'no gel plane reached the chain (the stage is a no-op)'
    return _generic_refusal(st)


def matte_glow(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SC.matte_glow_on(st):
        return None
    gel = _k.get('gel')
    h, w = frame.height, frame.width
    why = matte_glow_refusal(st, w, h, gel=gel, need_gel=True)
    if why is None and gel is not None \
            and tuple(np.asarray(gel).shape) != (h, w, 3):
        why = 'the gel plane does not match the frame (the stage is a no-op)'
    if why:
        _chain._warn(f'MATTE_GLOW on the CPU: {why}')
        return None
    shaders, err = _compile_all(('MATTE_BLUR', 'MATTE_ADD'), 'MATTE_GLOW')
    if shaders is None:
        _chain._warn(f'MATTE_GLOW on the CPU: {err}')
        return None
    blur_sh, add_sh = shaders
    P = SC.matte_params(st, h)
    n = len(P.taps)
    try:
        gel = np.ascontiguousarray(np.asarray(gel, np.float32))
        gel_tex = device.upload(np.concatenate(
            [gel, np.ones((h, w, 1), np.float32)], 2))
        radius = float(st.matte_glow_radius)
        tap_tex = [device.upload_cached(
            ('sig_matte_taps', radius, k + 1, int(h)),
            lambda k=k: SC.matte_taps_image(P.taps[k])) for k in range(n)]
        src_tex = frame.source()
        src_target = frame.target
        frame.target = None            # the draws must not free it
        res = (float(w), float(h))
        scratch = device.Target(w, h)
        blurs = [device.Target(w, h) for _ in range(n)]
        draws = []
        for k in range(n):
            R = int(P.radii[k])
            draws.append((blur_sh, {'resolution': res, 'radius': R,
                                    'dir_x': 1, 'dir_y': 0},
                          {'source': gel_tex, 'taps': tap_tex[k]}, scratch,
                          'NONE', False, None))
            draws.append((blur_sh, {'resolution': res, 'radius': R,
                                    'dir_x': 0, 'dir_y': 1},
                          {'source': device.target_texture(scratch),
                           'taps': tap_tex[k]}, blurs[k],
                          'NONE', False, None))
        out_t = device.Target(w, h)
        binds = {'source': src_tex, 'gel': gel_tex}
        uniforms = {'resolution': res, 'passes': int(n), 'e': float(P.e)}
        for i in range(SC.MATTE_MAX_PASSES):
            binds[f'b{i + 1}'] = device.target_texture(blurs[i]) \
                if i < n else gel_tex            # unused: `passes` gates them
            uniforms[f'w{i + 1}'] = float(P.w[i]) if i < n else 0.0
        draws.append((add_sh, uniforms, binds, out_t, 'NONE', False, None))
        frame.pending.extend(draws)
        frame.retired.extend([scratch] + blurs
                             + ([src_target] if src_target is not None
                                else []))
        frame.target = out_t
        frame._source_tex = None
        frame.stages.append('MATTE_GLOW')
        return frame
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'MATTE_GLOW fell back to the CPU: '
                     f'{type(exc).__name__}: {exc}')
        return None


# ------------------------------------------------------- C132: MPEG-1

MPEG_STAGES = ('MPEG_ENC', 'MPEG_DCT_ROW', 'MPEG_DCT_COL_Q', 'MPEG_IDCT_ROW',
               'MPEG_IDCT_COL_OUT')


def mpeg1_refusal(st, w=None, h=None):
    if not SC.mpeg1_on(st):
        return None
    return _generic_refusal(st)


def mpeg1(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    if not SC.mpeg1_on(st):
        return None
    why = mpeg1_refusal(st)
    if why:
        _chain._warn(f'MPEG1 on the CPU: {why}')
        return None
    shaders, err = _compile_all(MPEG_STAGES, 'MPEG1')
    if shaders is None:
        _chain._warn(f'MPEG1 on the CPU: {err}')
        return None
    enc_sh, row_sh, colq_sh, irow_sh, out_sh = shaders
    h, w = frame.height, frame.width
    P = SC.mpeg1_params(st, int(frame_no), h, w)
    try:
        u8 = device.upload_cached(('sig_u8',), SIG.u8_image)
        dct = device.upload_cached(('sig_mpeg_dct',), SC.mpeg1_dct_image)
        qtab = device.upload_cached(('sig_mpeg_q', int(P.qs)),
                                    lambda: SC.mpeg1_q_image(P.qs))
        src_tex = frame.source()
        src_target = frame.target
        frame.target = None            # the passes must not free it
        pad = (float(P.Wp), float(P.Hp))
        a = device.Target(P.Wp, P.Hp)
        b = device.Target(P.Wp, P.Hp)
        out_t = device.Target(w, h)
        ta = device.target_texture(a)
        tb = device.target_texture(b)
        draws = [
            (enc_sh, {'resolution': pad, 'src_size': (float(w), float(h))},
             {'source': src_tex}, a, 'NONE', False, None),
            (row_sh, {'resolution': pad}, {'source': ta, 'dct': dct}, b,
             'NONE', False, None),
            (colq_sh, {'resolution': pad, 'qs': int(P.qs)},
             {'source': tb, 'dct': dct, 'qtab': qtab}, a, 'NONE', False, None),
            (irow_sh, {'resolution': pad}, {'source': ta, 'dct': dct}, b,
             'NONE', False, None),
            (out_sh, {'resolution': (float(w), float(h))},
             {'source': tb, 'dct': dct, 'u8lut': u8}, out_t,
             'NONE', False, None)]
        frame.pending.extend(draws)
        frame.retired.extend([a, b] + ([src_target] if src_target is not None
                                       else []))
        frame.target = out_t
        frame._source_tex = None
        frame.stages.append('MPEG1')
        return frame
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'MPEG1 fell back to the CPU: '
                     f'{type(exc).__name__}: {exc}')
        return None


# ------------------------------------------------------ C133: Smacker

def smacker_refusal(st, w=None, h=None):
    if not SC.smacker_on(st):
        return None
    return _generic_refusal(st)


def smacker_uniforms(st, w, h):
    t_fill, t_full = SC.smacker_thresholds(float(st.smacker_quality))
    return {'resolution': (float(w), float(h)), 't_fill': int(t_fill),
            't_full': int(t_full),
            'inv255': float(np.float32(1.0 / 255.0))}


def smacker(frame, st, frame_no=0, seed=0, **_k):
    from . import chain as _chain
    from .chain_palette import pal_image
    if not SC.smacker_on(st):
        return None
    why = smacker_refusal(st)
    if why:
        _chain._warn(f'SMACKER on the CPU: {why}')
        return None
    if 'SMACKER' not in _chain.ENABLED:
        _chain._warn('SMACKER on the CPU: SMACKER is not enabled')
        return None
    tables = SC.smacker_palette_cached(st, int(seed))
    if tables is None:
        # the palette needs pixels: one readback BY NAME, then the stage
        # draws from the re-uploaded frame
        rgb = frame.down(
            'SMACKER', 'the codec palette is fitted on the CPU (once per lock)'
            if getattr(st, 'palette_lock', True) else
            'the codec palette is refitted on the CPU from every frame '
            '(Lock Palette is off)')
        tables = SC.smacker_palette(st, rgb, int(seed))
    pal, _lut = tables
    key_b = pal.tobytes()
    try:
        icm_tex = device.upload_cached(
            ('icm', key_b),
            lambda: PA.icm_index_image(PA.get_inverse_colormap(pal)))
        pal_tex = device.upload_cached(('pal', key_b), lambda: pal_image(pal))
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'SMACKER on the CPU: {exc}')
        return None
    h, w = frame.height, frame.width
    return _chain.try_stage('SMACKER', frame, st, smacker_uniforms(st, w, h),
                            extra_binds={'pal': pal_tex, 'icm': icm_tex})
