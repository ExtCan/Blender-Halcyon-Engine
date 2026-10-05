"""R251 C001: the N64 VI coverage filter's GPU road -- two resident draws
over the frame (the AA pass, then the divot), no readback, the NTSC
multi-pass shape without its ping-pong targets. Bound onto gpu/chain's
namespace by the integrator's bottom-of-file import so `post.process`
finds `n64vi` by name.

`chain` is imported LAZILY inside each function (this module is imported
by gpu/chain.py itself). `n64vi_refusal(st)` names the reason the CPU
runs instead ('the N64 VI filter is off', 'the coverage plane is absent
(1 sample per pixel needed)') or None; the device / gate refusals come
from `chain.available`.
"""

import numpy as np

from . import device
from ..core import n64vi as N64VI


def n64vi_refusal(st, w=None, h=None):
    """None when the two VI stages may draw, else the printed reason."""
    if not bool(getattr(st, 'n64_coverage_aa', False)):
        return 'the N64 VI filter is off'
    if getattr(st, '_n64_cvg', None) is None:
        return 'the coverage plane is absent (1 sample per pixel needed)'
    from . import chain as _chain
    ok, why = _chain.available(st)
    if not ok:
        return str(why)
    return None


def _cvg_texture(cvg3):
    """The coverage plane as a cached texture (keyed on its content)."""
    cvg3 = np.ascontiguousarray(np.asarray(cvg3, np.int32))
    key = ('n64cvg', cvg3.shape, hash(cvg3.tobytes()))
    return device.upload_cached(key, lambda: N64VI.cvg_image(cvg3))


def n64vi(frame, st, frame_no=0, seed=0, **_k):
    """The VI's coverage blend and divot on the resident frame: the Frame
    with two draws recorded, or None (the CPU one runs, the reason
    printed once)."""
    from . import chain as _chain
    why = n64vi_refusal(st)
    if why is not None:
        _chain._warn(f'N64VI on the CPU: {why}')
        return None
    cvg3 = np.asarray(st._n64_cvg)
    h, w = frame.height, frame.width
    if tuple(cvg3.shape[:2]) != (h, w):
        _chain._warn("N64VI on the CPU: the coverage plane is not the "
                     "frame's size")
        return None
    try:
        tex = _cvg_texture(cvg3)
    except Exception as exc:                                    # noqa: BLE001
        _chain._warn(f'N64VI on the CPU: {exc}')
        return None
    uni = {'resolution': (float(w), float(h))}
    divot = bool(getattr(st, 'n64_divot', True))
    # BOTH stages compile before EITHER draws: a draw is recorded on the
    # frame in place, so a divot that refused after the AA pass drew would
    # hand the CPU fallback (the whole `n64_vi`) a frame already blended
    # -- the AA applied twice. Refusing here leaves the frame untouched.
    names = ('N64VI_AA', 'N64VI_DIVOT') if divot else ('N64VI_AA',)
    for name in names:
        if name not in _chain.ENABLED:
            return None
        shader, err = device.compile_stage(name, _chain.STAGES[name])
        if shader is None:
            _chain._warn(f'N64VI on the CPU: {name}: {err}')
            return None
    out = _chain.try_stage('N64VI_AA', frame, st, uni, extra_binds={'cvg': tex})
    if out is None:
        return None
    if divot:
        out2 = _chain.try_stage('N64VI_DIVOT', out, st, uni,
                                extra_binds={'cvg': tex})
        if out2 is None:
            # the divot's draw itself failed AFTER the AA pass was
            # recorded: finish THIS stage on the CPU over the AA pass's
            # output (read back by name), never the whole filter again
            rgb = out.down('N64VI_DIVOT', 'the divot draw failed after '
                                          'the coverage blend')
            c8 = N64VI.vi_divot(N64VI.to8(rgb), cvg3)
            out.rgb = (c8.astype(np.float32)
                       / np.float32(255.0)).astype(np.float32)
            return out
        out = out2
    return out
