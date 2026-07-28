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
