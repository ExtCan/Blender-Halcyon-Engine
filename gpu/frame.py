"""The frame kept on the GPU between stages (R250, 1.89.0).

Until 1.88.0 every GPU stage began by uploading the frame and ended by
reading it back: the deferred shading read its composite back, the ink
uploaded that picture again, the post chain uploaded it once more per
stage. At the field's 2x supersample each crossing moved 59 MB of RGBA32F.

`Resident` is the handle for a frame that is still on the GPU: the
deferred shading's own target, kept alive instead of returned to the
pool, then the ink's output target, then the resolve's. Its LAW: a
resident target equals the CPU `img` byte for byte. Any CPU stage that
edits `img` releases it -- `edited(st, gbuf, stage)` -- and the next GPU
stage uploads as it did in 1.88.0. Nothing is lost but the residency,
and `LAST['left_gpu']` names the stage that took the frame back up.

Lifetime: `render()` owns the handle (it releases a stale one on entry,
and in a `finally` at its end unless the caller set `st._keep_gpu_frame`,
in which case the caller -- the engine, the viewport, the self test --
releases it in its own `finally` after `post.process`). A dropped handle
buries its target (device.Target.__del__), never destroys it. Release is
idempotent and nulls the target, so a stale reader sees `None` and
uploads with a printed reason instead of dereferencing a freed offscreen.

The resolve (the supersample filter at output size) is the one draw
that lives here: `resolve()` reads the internal-size resident, draws the
`RESOLVE` stage into an output-size target, reads that back once, and
hands the output target over as the new resident.
"""

import numpy as np

#: where the last frame's residency went, for the field test and the
#: console: 'kept' (the stages the frame stayed resident through),
#: 'left_gpu' (the CPU stage that released it, or ''), 'resolve' (the
#: RESOLVE pass's engagement and ms)
LAST = {}


class Resident:
    """A frame on the GPU: an un-freed device.Target and its size."""

    __slots__ = ('target', 'width', 'height', 'stage')

    def __init__(self, target, width, height, stage):
        self.target = target
        self.width = int(width)
        self.height = int(height)
        self.stage = str(stage)

    @property
    def live(self):
        t = self.target
        return t is not None and getattr(t, 'offscreen', None) is not None

    def texture(self):
        """The colour texture handle for a later draw's sampler, or None
        once released."""
        if not self.live:
            return None
        from . import device
        return device.target_texture(self.target)

    def release(self):
        """Return the target to the pool. Idempotent."""
        t = self.target
        self.target = None
        if t is not None:
            try:
                t.free()
            except Exception:                                   # noqa: BLE001
                pass

    def __del__(self):
        # the target's own __del__ buries the offscreen if this handle
        # was forgotten; nothing to do here beyond dropping the reference
        self.target = None


def current(st):
    """The frame's resident handle on the settings, live, or None."""
    r = getattr(st, '_gpu_frame', None)
    if r is None:
        return None
    if not r.live:
        try:
            st._gpu_frame = None
        except Exception:                                       # noqa: BLE001
            pass
        return None
    return r


def install(st, resident, stage):
    """Make `resident` the frame's handle (releasing any previous one)."""
    prev = getattr(st, '_gpu_frame', None)
    if prev is not None and prev is not resident:
        prev.release()
    try:
        st._gpu_frame = resident
    except Exception:                                           # noqa: BLE001
        resident.release()
        return None
    LAST.setdefault('kept', [])
    if stage not in LAST['kept']:
        LAST['kept'].append(str(stage))
    return resident


def edited(st, stage, why=''):
    """A CPU stage changed `img`: the resident frame no longer equals it.

    A pure no-op when nothing is resident (the CPU device never reaches
    the device module through here)."""
    r = getattr(st, '_gpu_frame', None)
    if r is None:
        return
    r.release()
    try:
        st._gpu_frame = None
    except Exception:                                           # noqa: BLE001
        pass
    if not LAST.get('left_gpu'):
        LAST['left_gpu'] = f'{stage}' + (f' ({why})' if why else '')


def release(st):
    """Release whatever is resident (the caller's `finally`)."""
    r = getattr(st, '_gpu_frame', None)
    if r is not None:
        r.release()
        try:
            st._gpu_frame = None
        except Exception:                                       # noqa: BLE001
            pass


def begin_frame(st):
    """Reset the per-frame record; release a handle a previous frame left
    on a reused settings object (motion blur re-renders on the same
    object)."""
    release(st)
    LAST.clear()


# ---------------------------------------------------------------- resolve


def resolve_kernel(ss, st):
    """render._resolve's own normalised float32 kernel, (ss, ss)."""
    from ..core.render import FILTERS
    fn = FILTERS.get(st.aa_filter, FILTERS['BOX'])
    off = (np.arange(ss, dtype=np.float32) + 0.5) / ss - 0.5
    wx = fn(off * 2.0 * st.aa_filter_width)
    wy = wx
    kern = np.outer(wy, wx).astype(np.float32)
    s = kern.sum()
    kern = kern / (s if abs(s) > 1e-8 else 1.0)
    return np.asarray(kern, np.float32)


def resolve(st, W, H, ss):
    """The RESOLVE stage over the resident frame: (image (H, W, 4), None)
    with the output target installed as the new resident, or (None, why)
    with the resident untouched (the CPU `_resolve` then runs on `img`).

    The one door the resolve knocks on; it is never reached without a
    live resident, so a driverless run never gets here."""
    import time as _time
    from . import device
    from .stages import (INTERFACE, STAGES, resolve_flags, resolve_source,
                         resolve_spec, resolve_variant_name)
    r = current(st)
    if r is None:
        return None, 'no frame is resident'
    if int(ss) < 2 or int(ss) > 8:
        return None, f'a supersample factor of {ss} has no GPU resolve'
    rw, rh = W * ss, H * ss
    if (r.width, r.height) != (rw, rh):
        return None, (f'the resident frame is {r.width}x{r.height}, not the '
                      f'{rw}x{rh} the resolve expects')
    t0 = _time.perf_counter()
    # R251 (RAST-B): C094's sample clamp and C122's gamma-2 blend ride
    # dynamic variants of the pinned stage (stages.resolve_source);
    # both off is the 1.89.0 road exactly
    clamp, gamma = resolve_flags(st)
    vname = resolve_variant_name(clamp, gamma)
    if clamp or gamma:
        shader, err = device.compile_dynamic(
            vname, resolve_source(clamp, gamma), resolve_spec(clamp, gamma))
    else:
        shader, err = device.compile_stage('RESOLVE', STAGES['RESOLVE'])
    if shader is None:
        return None, f'the driver rejected the resolve pass: {err}'
    kern = resolve_kernel(ss, st)
    ktex_img = np.zeros((ss, ss, 4), np.float32)
    ktex_img[..., 0] = kern
    out_t = None
    try:
        ktex = device.upload(ktex_img)
        src = r.texture()
        if src is None:
            return None, 'the resident frame was released before the resolve'
        out_t = device.Target(W, H)
        uniforms = {'resolution': (float(W), float(H)), 'ss': int(ss)}
        samplers = {'source': src, 'kernel': ktex}
        if clamp:
            uniforms['clamp_samples'] = 1.0
        if gamma:
            from ..core import raster as _raster
            uniforms['gamma_blend'] = 1.0
            # the CPU's own tables, uploaded once per session
            samplers['gtab'] = device.upload_cached(('gamma2tab',),
                                                    _raster.gamma2_texture)
        got = device.draw_many(
            [(shader, uniforms, samplers, out_t, 'NONE', False, None)],
            read=out_t)
    except Exception as exc:                                    # noqa: BLE001
        if out_t is not None:
            out_t.free()
        return None, f'the resolve pass failed: {type(exc).__name__}: {exc}'
    # the internal-size frame is done with; the output target is the
    # frame now (its readback IS the frame the caller returns)
    r.release()
    res = Resident(out_t, W, H, 'resolve')
    try:
        st._gpu_frame = res
    except Exception:                                           # noqa: BLE001
        res.release()
    kept = LAST.setdefault('kept', [])
    if 'resolve' not in kept:
        kept.append('resolve')
    LAST['resolve'] = {'ss': int(ss), 'ms': (_time.perf_counter() - t0) * 1000.0,
                       'variant': vname}
    _ = INTERFACE
    return np.asarray(got, np.float32), None
