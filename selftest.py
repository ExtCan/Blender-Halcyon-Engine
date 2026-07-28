"""A self-test that runs on the user's hardware and prints one pasteable report.

I cannot reach your machine. What I can do is send a probe: this runs the checks
I would run myself if I could, and prints them in a form that can be pasted
straight back.

It covers the three things this add-on has had to ship unmeasured:

  * whether the GPU shaders actually compile and run on a real driver, and
    whether their output matches the CPU path they replace
  * how the CPU scales with threads, and whether the worker pool helps
  * where a frame's time goes, per stage

Everything degrades: no GPU means the GPU section reports why and the rest still
runs.
"""

import platform
import sys
import time

import bpy
import numpy as np
from bpy.props import BoolProperty
from bpy.types import Operator

LINE = '-' * 66


def _p(out, text=''):
    out.append(text)
    print(text)


def environment(out):
    _p(out, LINE)
    _p(out, 'HALCYON SELF TEST')
    _p(out, LINE)
    from .version import version_string
    _p(out, f'  addon          : {version_string()}')
    _p(out, f'  blender        : {".".join(str(v) for v in bpy.app.version)}')
    _p(out, f'  platform       : {platform.system()} {platform.release()} '
            f'({platform.machine()})')
    _p(out, f'  python         : {sys.version.split()[0]}')
    _p(out, f'  numpy          : {np.__version__}')
    try:
        import os
        cpus = os.cpu_count()
    except Exception:                                           # noqa: BLE001
        cpus = '?'
    _p(out, f'  logical cores  : {cpus}')
    try:
        import gpu
        _p(out, f'  gpu backend    : {gpu.platform.backend_type_get()}')
        _p(out, f'  gpu vendor     : {gpu.platform.vendor_get()}')
        _p(out, f'  gpu renderer   : {gpu.platform.renderer_get()}')
        _p(out, f'  gl version     : {gpu.platform.version_get()}')
    except Exception as exc:                                    # noqa: BLE001
        _p(out, f'  gpu            : unavailable ({exc})')


def gpu_stages(out):
    """Compile and run every GLSL stage, and compare against the CPU path."""
    from .core import dither as DI
    from .core import post as PO
    from .core.settings import RenderSettings
    from .gpu import device
    from .gpu.stages import MASK_KINDS, STAGES, VALIDATION

    _p(out)
    _p(out, LINE)
    _p(out, 'GPU STAGES  (compiled and run on your driver)')
    _p(out, LINE)
    ok, why = device.probe()
    if not ok:
        _p(out, f'  skipped: {why}')
        return
    _p(out, f'  device: {why}')

    h, w = 64, 96
    rng = np.random.default_rng(0)
    img = rng.random((h, w, 3)).astype(np.float32)
    rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=2)

    st = RenderSettings()
    st.exposure, st.gamma, st.contrast = 1.3, 2.2, 0.15
    st.saturation, st.brightness = 1.25, 0.05
    st.crt, st.crt_scanlines, st.crt_mask = True, 0.4, 'APERTURE'
    st.crt_mask_strength, st.crt_vignette = 0.35, 0.5
    st.crt_curvature = st.crt_bloom = 0.0
    st.lens_distortion, st.chromatic_aberration = 0.25, 3.0
    st.lens_vignette_edges = False
    st.composite, st.composite_bleed = True, 1.0
    st.composite_ringing, st.composite_dot_crawl = 0.5, 0.0

    cases = {
        'DISPLAY': (dict(exposure=1.3, brightness=0.05, contrast=0.15,
                         saturation=1.25, gamma=2.2),
                    lambda: PO.display_transform(img.copy(), st)),
        'CRT': (dict(scanlines=0.4, mask_strength=0.35,
                     mask_kind=MASK_KINDS['APERTURE'], vignette=0.5,
                     resolution=(float(w), float(h))),
                lambda: PO.crt(img.copy(), st)),
        'DITHER': (dict(levels=(32.0, 64.0, 32.0), strength=1.0,
                        matrix_size=4.0, resolution=(float(w), float(h))),
                   lambda: DI.ordered_bits(img.copy(), (5, 6, 5), 'BAYER4', 1.0)),
        'LENS': (dict(distortion=0.25, aberration=3.0, edges=0.0,
                      resolution=(float(w), float(h))),
                 lambda: PO.lens_distortion(img.copy(), st)),
        'NTSC': (dict(bleed=1.0, ringing=0.5,
                      resolution=(float(w), float(h))),
                 lambda: PO.composite_ntsc(img.copy(), st)),
    }

    _p(out, f'  {"stage":9s} {"compile":9s} {"run":7s} {"max diff":>9s} '
            f'{"mean":>9s}  claimed')
    for name, src in STAGES.items():
        claimed = VALIDATION.get(name, ('?', None))[0]
        shader, err = device.compile_stage(name, src)
        if shader is None:
            _p(out, f'  {name:9s} FAILED')
            _p(out, f'      driver said: {err}')
            continue
        uniforms, reference = cases.get(name, ({}, None))
        try:
            tex = device.upload(rgba)
            target = device.Target(w, h)
            try:
                got = device.draw_fullscreen(shader, uniforms,
                                             {'source': tex}, target)[:, :, :3]
            finally:
                target.free()
        except Exception as exc:                                # noqa: BLE001
            _p(out, f'  {name:9s} ok        FAILED')
            _p(out, f'      {type(exc).__name__}: {exc}')
            continue
        if reference is None:
            _p(out, f'  {name:9s} ok        ok      '
                    f'{"":>9s} {"":>9s}  {claimed} (no CPU reference)')
            continue
        want = reference()
        mx = float(np.abs(got - want).max())
        mn = float(np.abs(got - want).mean())
        _p(out, f'  {name:9s} ok        ok      {mx:9.5f} {mn:9.5f}  {claimed}')


def cpu_scaling(out, heavy=False):
    """Thread scaling and the worker pool, on a fixed scene."""
    from .core import parallel, render as core_render
    from .core.settings import RenderSettings
    from .tests.scenebuild import demo_scene

    _p(out)
    _p(out, LINE)
    _p(out, 'CPU SCALING  (same scene, varying thread count)')
    _p(out, LINE)
    # 320x240 is too small to show threading on a fast machine: the whole
    # frame is a few milliseconds and the pool costs more than it saves
    res = (960, 720) if heavy else (640, 480)
    counts = [1, 2, 4, 8, 16, 32]
    try:
        import os
        cpus = os.cpu_count() or 1
    except Exception:                                           # noqa: BLE001
        cpus = 1
    counts = [c for c in counts if c <= max(cpus, 1) * 2]

    base = None
    _p(out, f'  {res[0]}x{res[1]}, supersampling off')
    _p(out, '  (a small frame cannot show thread scaling -- the pool costs more '
            'than it saves)')
    for n in counts:
        st = RenderSettings()
        st.resolution_x, st.resolution_y = res
        st.aa_samples = 1
        st.threads = n
        sc = demo_scene(st)
        core_render.render(sc, st)                       # warm caches
        dt = min((lambda t0: (core_render.render(sc, st),
                              time.perf_counter() - t0)[1])(time.perf_counter())
                 for _ in range(3))
        base = base or dt
        _p(out, f'    {n:3d} threads  {dt:7.3f}s   {base / dt:5.2f}x')

    _p(out)
    _p(out, '  the same frame, threads off vs auto:')
    for label, th in (('one thread', 1), ('auto', 0)):
        st = RenderSettings()
        st.resolution_x, st.resolution_y = res
        st.aa_samples = 4
        st.threads = th
        sc = demo_scene(st)
        core_render.render(sc, st)
        dt = min((lambda t0: (core_render.render(sc, st),
                              time.perf_counter() - t0)[1])(time.perf_counter())
                 for _ in range(3))
        _p(out, f'    {label:12s} {dt:7.3f}s')

    _p(out)
    _p(out, '  worker processes:')
    try:
        _p(out, f'    interpreter : {parallel.find_interpreter()}')
        par, pkg = parallel.package_location()
        _p(out, f'    imports     : {pkg}  from  {par}')
    except Exception as exc:                                    # noqa: BLE001
        _p(out, f'    (bootstrap could not be reported: {exc})')
    st = RenderSettings()
    st.resolution_x, st.resolution_y = res
    st.aa_samples = 1
    sc = demo_scene(st)
    t0 = time.perf_counter()
    core_render.render(sc, st)
    single = time.perf_counter() - t0
    img, why = parallel.render_parallel(sc, st, max(cpus, 2), scene_key='selftest')
    if img is None:
        _p(out, f'    declined: {why}')
    else:
        t0 = time.perf_counter()
        parallel.render_parallel(sc, st, max(cpus, 2), scene_key='selftest')
        pooled = time.perf_counter() - t0
        _p(out, f'    in process  {single:7.3f}s')
        _p(out, f'    pooled      {pooled:7.3f}s   {single / max(pooled, 1e-6):5.2f}x')
    parallel.shutdown()


def frame_breakdown(out):
    from .core import render as core_render, stats as ST
    from .core.settings import RenderSettings
    from .tests.scenebuild import demo_scene

    _p(out)
    _p(out, LINE)
    _p(out, 'FRAME BREAKDOWN  (demo scene, 480x360)')
    _p(out, LINE)
    ST.reset()
    st = RenderSettings()
    st.resolution_x, st.resolution_y = 480, 360
    st.aa_samples = 4
    t0 = time.perf_counter()
    core_render.render(demo_scene(st), st)
    ST.report(total=time.perf_counter() - t0, printer=lambda s: _p(out, s))


class HALCYON_OT_selftest(Operator):
    """Measure this machine and print a report that can be pasted back"""

    bl_idname = 'halcyon.selftest'
    bl_label = "Run Self Test"
    bl_options = {'REGISTER'}

    include_scaling: BoolProperty(
        name="Thread Scaling", default=True,
        description="Render the same scene at several thread counts. Adds "
                    "roughly a minute")
    heavy: BoolProperty(
        name="Larger Frames", default=False,
        description="Use 480x360 instead of 320x240 for the scaling test")

    def execute(self, context):
        out = []
        sections = [('environment', lambda: environment(out)),
                    ('gpu stages', lambda: gpu_stages(out)),
                    ('frame breakdown', lambda: frame_breakdown(out))]
        if self.include_scaling:
            sections.append(('cpu scaling', lambda: cpu_scaling(out, self.heavy)))
        failed = 0
        for name, fn in sections:
            try:
                fn()
            except Exception as exc:                            # noqa: BLE001
                import traceback
                traceback.print_exc()
                failed += 1
                _p(out)
                _p(out, f'  [{name} failed: {type(exc).__name__}: {exc}]')
                _p(out, '  (the remaining sections still ran)')
        _p(out, LINE)
        _p(out, 'end of report -- copy everything from HALCYON SELF TEST down')
        _p(out, LINE)
        try:
            context.window_manager.clipboard = '\n'.join(out)
            msg = "Report printed to the console and copied to the clipboard"
            if failed:
                msg += f" ({failed} section(s) failed)"
            self.report({'WARNING' if failed else 'INFO'}, msg)
        except Exception:                                       # noqa: BLE001
            self.report({'INFO'}, "Report printed to the system console")
        return {'FINISHED'}


CLASSES = (HALCYON_OT_selftest,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
