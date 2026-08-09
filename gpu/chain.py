"""Running the validated post stages on the GPU, with a CPU fallback per stage.

Only stages listed in `stages.ENABLED` are attempted, and that list is set by
measured agreement with the CPU path rather than by intent. If a stage is not
enabled, or the GPU refuses it, the CPU function runs instead and the reason is
printed once.
"""

from . import device
from .stages import ENABLED, MASK_KINDS, STAGES

_WARNED = set()


def _warn(msg):
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f'[Halcyon GPU] {msg}')


def available(settings):
    # defense in depth: the caller's gate checks the device too, but a
    # second door that cannot open on the CPU device costs nothing
    if str(getattr(settings, 'render_device', 'CPU')).upper() != 'GPU':
        return False, 'the render device is CPU'
    if not getattr(settings, 'gpu_post', False):
        return False, 'GPU post processing is off'
    ok, why = device.probe()
    return ok, why


def try_stage(name, image, settings, uniforms):
    """Run one stage on the GPU. Returns None to mean "use the CPU one"."""
    if name not in ENABLED:
        return None
    ok, why = available(settings)
    if not ok:
        _warn(f'{name} on the CPU: {why}')
        return None
    shader, err = device.compile_stage(name, STAGES[name])
    if shader is None:
        _warn(f'{name} on the CPU: {err}')
        return None
    try:
        h, w = image.shape[:2]
        tex = device.upload(image)
        target = device.Target(w, h)
        try:
            out = device.draw_fullscreen(shader, uniforms, {'source': tex},
                                         target)
        finally:
            target.free()
        return out[:, :, :3]
    except Exception as exc:                                    # noqa: BLE001
        _warn(f'{name} fell back to the CPU: {type(exc).__name__}: {exc}')
        return None


def display(image, st):
    return try_stage('DISPLAY', image, st, {
        'exposure': float(st.exposure), 'brightness': float(st.brightness),
        'contrast': float(st.contrast), 'saturation': float(st.saturation),
        'gamma': max(float(st.gamma), 1e-3)})


def lens(image, st):
    if abs(float(st.lens_distortion)) < 1e-5 and \
            abs(float(st.chromatic_aberration)) < 1e-5:
        return None
    h, w = image.shape[:2]
    return try_stage('LENS', image, st, {
        'distortion': float(st.lens_distortion),
        'aberration': float(st.chromatic_aberration),
        'edges': 1.0 if st.lens_vignette_edges else 0.0,
        'resolution': (float(w), float(h))})


def ntsc(image, st, frame=0):
    """Composite chroma bleed as the CPU does it: three blur draws, one mix.

    The CPU's triple box re-pads the frame edge before every pass, so the
    only way to match it exactly is to *run* three passes -- an intermediate
    YIQ target ping-pongs through NTSC_BLUR and the final draw reassembles.
    Dot crawl is frame-dependent and stays on the CPU; a frame using it falls
    back whole, because half a composite artefact is worse than either half.
    """
    if not getattr(st, 'composite', False):
        return None
    if float(getattr(st, 'composite_dot_crawl', 0.0)) > 0.0:
        return None
    if 'NTSC' not in ENABLED:
        return None
    ok, why = available(st)
    if not ok:
        _warn(f'NTSC on the CPU: {why}')
        return None
    blur_sh, err = device.compile_stage('NTSC_BLUR', STAGES['NTSC_BLUR'])
    final_sh, err2 = device.compile_stage('NTSC', STAGES['NTSC'])
    if blur_sh is None or final_sh is None:
        _warn(f'NTSC on the CPU: {err or err2}')
        return None
    try:
        h, w = image.shape[:2]
        bleed = float(st.composite_bleed)
        ri = max(int(round(w / 320.0 * 6.0 * bleed)), 1)
        rq = max(int(round(w / 320.0 * 12.0 * bleed)), 1)
        if max(ri, rq) > 96:
            _warn('NTSC on the CPU: the chroma radius exceeds the shader '
                  'loop bound at this resolution')
            return None
        src_tex = device.upload(image)
        current = src_tex
        for step in range(3):
            target = device.Target(w, h)
            try:
                out = device.draw_fullscreen(
                    blur_sh,
                    {'ri': float(ri), 'rq': float(rq), 'ry': 2.0,
                     'to_yiq': 1.0 if step == 0 else 0.0,
                     'resolution': (float(w), float(h))},
                    {'source': current}, target)
            finally:
                target.free()
            current = device.upload(out)
        target = device.Target(w, h)
        try:
            out = device.draw_fullscreen(
                final_sh,
                {'ringing': float(st.composite_ringing)},
                {'source': src_tex, 'blurred': current}, target)
        finally:
            target.free()
        return out[:, :, :3]
    except Exception as exc:                                    # noqa: BLE001
        _warn(f'NTSC fell back to the CPU: {type(exc).__name__}: {exc}')
        return None


def crt(image, st):
    if not st.crt:
        return None
    if st.crt_curvature > 0.0 or st.crt_bloom > 0.0:
        return None            # those stages are not ported; keep it consistent
    h, w = image.shape[:2]
    return try_stage('CRT', image, st, {
        'scanlines': float(st.crt_scanlines),
        'mask_strength': float(st.crt_mask_strength),
        'mask_kind': int(MASK_KINDS.get(st.crt_mask, 0)),
        'vignette': float(st.crt_vignette),
        'resolution': (float(w), float(h))})
