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
    }

    _p(out, f'  {"stage":9s} {"compile":9s} {"run":7s} {"max diff":>9s} '
            f'{"mean":>9s}  claimed')
    for name, src in STAGES.items():
        if name in ('NTSC', 'NTSC_BLUR'):
            continue                    # multi-pass: measured below as one
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

    # NTSC runs as three blur draws and a combine -- the CPU re-pads the
    # frame edge before each of its three box passes, and matching that
    # exactly takes three real passes. Measured through the orchestrator,
    # with the ENABLED gate lifted for the measurement only.
    try:
        from .gpu import chain, stages as _stages
        claimed = VALIDATION.get('NTSC', ('?', None))[0]
        saved = _stages.ENABLED
        _stages.ENABLED = tuple(set(saved) | {'NTSC', 'NTSC_BLUR'})
        chain.ENABLED = _stages.ENABLED
        # the orchestrator checks the user's own switches before it will
        # draw; this is a measurement, so both are on for its duration.
        # Round one of this section reported SKIPPED for gpu_post, and the
        # 1.25.76 device-switch fix (the chain now refuses the CPU device
        # too -- correctly) SKIPPED it a second time for render_device.
        # Every gate the render path honours, a measurement must satisfy.
        st.gpu_post = True
        _saved_dev = st.render_device
        st.render_device = 'GPU'
        try:
            got = chain.ntsc(img, st)
        finally:
            _stages.ENABLED = saved
            chain.ENABLED = saved
            st.gpu_post = False
            st.render_device = _saved_dev
        if got is None:
            _p(out, f'  {"NTSC":9s} ok        SKIPPED (see console)')
        else:
            want = PO.composite_ntsc(img.copy(), st)
            mx = float(np.abs(got - want).max())
            mn = float(np.abs(got - want).mean())
            _p(out, f'  {"NTSC":9s} ok        ok      {mx:9.5f} {mn:9.5f}  '
                    f'{claimed} (3 blur draws + combine)')
    except Exception as exc:                                    # noqa: BLE001
        _p(out, f'  {"NTSC":9s} FAILED    {type(exc).__name__}: {exc}')


def _bisect_deferred(out, passes):
    """Compile the frame shader's layers separately, smallest first.

    A driver's 'Shader Compile Error' names nothing. Building the same source
    up in four steps -- models, dispatch, G-buffer reader, the full pass --
    turns one opaque failure into a chunk with a name, in the report itself.
    """
    from .gpu import device
    from .gpu import gbuffer as GB
    from .gpu import glsl_shading as GS

    trivial = '\nin vec2 vUV;\nout vec4 Color;\nvoid main() {\n' \
              '    HalcyonSurface s;\n    s.diffuse = vec3(vUV, 0.0);\n' \
              '    Color = vec4(s.diffuse * hal_diffuse_lambert(0.5), 1.0);\n}\n'
    dispatch = '\nin vec2 vUV;\nout vec4 Color;\nvoid main() {\n' \
               '    HalcyonSurface s;\n    s.diffuse = vec3(vUV, 0.0);\n' \
               '    s.specular = vec3(1.0);\n    s.glossiness = 25.0;\n' \
               '    s.roughness = 0.3; s.soften = 0.0; s.ior = 1.45;\n' \
               '    s.tangent = vec3(1.0, 0.0, 0.0);\n' \
               '    s.bitangent = vec3(0.0, 1.0, 0.0);\n' \
               '    vec4 d = hal_evaluate(3, s, vec3(0.0, 0.0, 1.0),\n' \
               '        normalize(vec3(0.3, 0.2, 0.9)),\n' \
               '        normalize(vec3(0.1, 0.1, 0.99)));\n' \
               '    Color = vec4(s.diffuse * d.x + d.yzw, 1.0);\n}\n'
    reader = '\nin vec2 vUV;\nout vec4 Color;\nvoid main() {\n' \
             '    HalcyonFragment f = hal_read_gbuffer(vUV);\n' \
             '    vec4 td = hal_tri_data(max(f.tri, 0.0));\n' \
             '    Color = vec4(f.P + f.N + vec3(td.x, f.uv), \n' \
             '                 f.covered ? 1.0 : 0.0);\n}\n'
    gspec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris'],
             'floats': ['hal_attr_side', 'hal_slot_count', 'hal_tri_side'],
             'vec3': ['hal_eye']}
    probes = [
        ('reflectance models', GS.GLSL + trivial, {'samplers': []}),
        ('model dispatch', GS.GLSL + GS.DISPATCH + dispatch, {'samplers': []}),
        ('G-buffer reader', GB.GLSL + reader, gspec),
        ('the full material pass', passes[0][2] if passes else '',
         {'samplers': gspec['samplers']
          + list(passes[0][3].get('samplers', ())) if passes
          else gspec['samplers'],
          'floats': gspec['floats'], 'vec3': ['hal_eye']}),
    ]
    for label, src, spec in probes:
        if not src:
            continue
        shader, err = device.compile_dynamic(
            'HAL_PROBE_' + label.replace(' ', '_').replace('-', '_'), src,
            spec)
        if shader is None:
            _p(out, f'    {label:24s} REFUSED: {err}')
            return                    # first failure localises it; stop there
        _p(out, f'    {label:24s} compiles')
    _p(out, '    every probe compiles alone -- the failure needs the '
            'console log')


def gpu_deferred(out):
    """Shade real frames on the driver and compare against the CPU frames.

    Two measurements. A small frame proves agreement -- shadow maps included
    now, sun and cube and spot -- and a working-size frame answers the
    question the port exists for: what does the stage that eats 72% of a CPU
    frame cost once it runs on the GPU. Everything here is verified through
    Halcyon's own compiler before it ever reaches a driver; the driver's
    numbers are the part this machine cannot produce.
    """
    from .core import raster
    from .core import render as R
    from .core.scene import Light
    from .core.settings import RenderSettings
    from .gpu import device, shade as GSH
    from .tests.scenebuild import (add_coded_floor, add_normal_mapped_ball,
                                   demo_scene)

    _p(out)
    _p(out, LINE)
    _p(out, 'GPU DEFERRED SHADING  (a real frame, shaded by your driver)')
    _p(out, LINE)
    ok, why = device.probe()
    if not ok:
        _p(out, f'  skipped: {why}')
        return

    w, h = 160, 120
    st = RenderSettings()
    st.resolution_x, st.resolution_y = w, h
    st.aa_samples = 1
    # shadows ON: the maps travel to the GPU now. The demo scene's own SUN
    # and POINT keep their maps (ortho and six-face cube), and a SPOT joins
    # for the perspective map.
    sc = demo_scene(st, with_texture=False)
    # the ball wears a tangent-space normal map (the chain runs on the GPU,
    # Strength 0.8, Bump Strength 0.65) and the floor's tiles come from a
    # coded shader node -- its GLSL inlined natively, mangled, sockets
    # baked -- so this small frame proves both on your driver, not just in
    # the headless compiler
    add_normal_mapped_ball(sc)
    add_coded_floor(sc)
    # the box reflects the blend sky: the environment-reflection term's
    # sphere-map arithmetic proves itself on the driver too
    sc.materials[2].reflect_level = 0.3
    sc.lights.append(
        Light(type='SPOT', name='Rim', position=(-4.0, 2.5, 4.0),
              direction=(0.55, -0.35, -0.75), color=(1.0, 0.3, 0.8),
              energy=400.0, shadow='MAP', shadow_bias=0.02,
              spot_size=0.9, spot_blend=0.3))

    import time as _time
    t0 = _time.perf_counter()
    cpu_img = R.render(sc, st)
    t_cpu = _time.perf_counter() - t0

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)

    passes, why, atlases = GSH.plan_frame(job, g)
    if passes is None:
        _p(out, f'  frame did not qualify: {why}')
        return
    _p(out, f'  {len(passes)} material pass(es): '
            + ', '.join(p[1] for p in passes))

    t0 = _time.perf_counter()
    got, info = GSH.shade_frame(job, g)
    t_cold = _time.perf_counter() - t0
    if got is None:
        _p(out, f'  FAILED: {info}')
        _p(out, '  narrowing down which part the driver refused:')
        _bisect_deferred(out, passes)
        _p(out, '  (if a probe failed, the console holds the shader info '
                'log -- copy those lines too)')
        return
    # the second frame is the one an animation renders: the shader compiles
    # are cached per scene, so what remains is pack, upload, draw, read back
    t0 = _time.perf_counter()
    got2, _info2 = GSH.shade_frame(job, g)
    t_warm = _time.perf_counter() - t0
    if got2 is not None:
        got = got2
    cov = g.tri >= 0
    diff = np.abs(got[cov] - cpu_img[cov][:, :3])
    _p(out, f'  covered pixels : {int(cov.sum())}')
    _p(out, f'  max difference : {float(diff.max()):.6f}')
    _p(out, f'  mean difference: {float(diff.mean()):.7f}')
    _p(out, f'  CPU frame      : {t_cpu * 1000.0:7.1f} ms (full render)')
    _p(out, f'  GPU cold frame : {t_cold * 1000.0:7.1f} ms '
            f'(compiles {len(passes)} shaders, then shades)')
    _p(out, f'  GPU warm frame : {t_warm * 1000.0:7.1f} ms '
            f'(pack + upload + {len(passes)} passes + read back)')
    tm = dict(GSH.LAST_TIMINGS)
    if tm:
        _p(out, f'    of which     : {tm.get("plan_ms", 0.0):5.1f} ms plan, '
                f'{tm.get("pack_upload_ms", 0.0):5.1f} ms pack+upload, '
                f'{tm.get("draw_read_ms", 0.0):5.1f} ms draw+read, '
                f'{tm.get("composite_ms", 0.0):5.1f} ms composite')
    _p(out, '  the warm number is what an animation pays per frame; the '
            'cold one is paid')
    _p(out, '  once per scene. Shadow maps ride along: this frame has sun, '
            'cube and spot')
    _p(out, '  shadows in it, the ball is red marble (integer-hash pattern, '
            'generated')
    _p(out, '  coordinates) under a tangent-space normal map, the floor '
            'tiles are a coded')
    _p(out, '  shader node running its own GLSL natively, and the box '
            'reflects the sky')

    # ---- the number the whole port exists for: the 72% at working size
    w2, h2 = 480, 360
    st2 = RenderSettings()
    st2.resolution_x, st2.resolution_y = w2, h2
    st2.aa_samples = 1
    sc2 = demo_scene(st2, with_texture=False)
    sc2.lights.append(
        Light(type='SPOT', name='Rim', position=(-4.0, 2.5, 4.0),
              direction=(0.55, -0.35, -0.75), color=(1.0, 0.3, 0.8),
              energy=400.0, shadow='MAP', shadow_bias=0.02,
              spot_size=0.9, spot_blend=0.3))
    t0 = _time.perf_counter()
    cpu2 = R.render(sc2, st2)
    t_cpu2 = _time.perf_counter() - t0
    view2, _p2, vp2, eye2 = R.camera_matrices(sc2.camera, w2, h2)
    g2 = raster.GBuffer(w2, h2)
    raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w2, h2, gbuf=g2)
    job2 = R.ShadeJob(sc2, st2, {}, None, view2, eye2, w2, h2)
    got2, info2 = GSH.shade_frame(job2, g2)          # cold at this size
    t0 = _time.perf_counter()
    got2, info2 = GSH.shade_frame(job2, g2)
    t_warm2 = _time.perf_counter() - t0
    _p(out)
    _p(out, f'  at {w2}x{h2}, shadows on (where the CPU spends 72% of its '
            f'frame shading):')
    if got2 is None:
        _p(out, f'    FAILED: {info2}')
    else:
        cov2 = g2.tri >= 0
        d2 = np.abs(got2[cov2] - cpu2[cov2][:, :3])
        _p(out, f'    agreement      : max {float(d2.max()):.6f}  '
                f'mean {float(d2.mean()):.7f}')
        _p(out, f'    CPU full frame : {t_cpu2 * 1000.0:7.1f} ms')
        _p(out, f'    GPU warm shade : {t_warm2 * 1000.0:7.1f} ms '
                f'(replaces the shade stage)')
        tm2 = dict(GSH.LAST_TIMINGS)
        if tm2:
            _p(out, f'    of which       : '
                    f'{tm2.get("plan_ms", 0.0):5.1f} ms plan, '
                    f'{tm2.get("pack_upload_ms", 0.0):5.1f} ms pack+upload, '
                    f'{tm2.get("draw_read_ms", 0.0):5.1f} ms draw+read, '
                    f'{tm2.get("composite_ms", 0.0):5.1f} ms composite')


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


def viewport_device(out):
    """The viewport preview through the device switch, on this driver.

    Since 1.25.83 the CPU/GPU switch governs the interactive preview too:
    the viewport worker borrows the F12 marshal for its driver bursts. The
    live viewport renders on a worker thread and crosses each burst to the
    main thread; here the same code runs synchronously (this operator IS
    the main thread, which owns the context and cannot pump its own
    timers), so this section proves the FRAMES -- draft and refine, CPU
    against driver, through the viewport's own worker path including its
    post chain -- and the crossings themselves are the same marshal every
    F12 since 1.25.53 stands on.
    """
    import time as _time

    from .core.settings import RenderSettings
    from .gpu import chain as _CH
    from .gpu import craster as _CRA
    from .gpu import device
    from .gpu import shade as _GSH
    from .preview import DRAFT_FACTOR, Viewport, shape_settings
    from .tests.scenebuild import demo_scene

    _p(out)
    _p(out, LINE)
    _p(out, 'VIEWPORT PREVIEW  (the device switch reaches the viewport now)')
    _p(out, LINE)
    ok, why = device.probe()
    if not ok:
        _p(out, f'  skipped: {why}')
        return

    w, h = 480, 360

    def _frame(dev, draft):
        st = RenderSettings()
        st.render_device = dev
        st.preview_scale = 1
        st.aa_samples = 1
        sc = demo_scene(st, with_texture=True)
        shape_settings(st, w, h)
        vp = Viewport()
        vp.set_scene(sc, st)
        vp.abort = False          # kick() clears this; we render synchronously
        key = vp._key(sc.camera, w, h)
        t0 = _time.perf_counter()
        vp._render(None, sc, st, sc.camera, w, h, key, vp.version, draft)
        dt = _time.perf_counter() - t0
        if vp.frame is None:
            raise RuntimeError(f'the {dev} viewport frame never parked')
        return vp, vp.frame.astype(np.float64), dt

    hit = {'R': False, 'S': False, 'P': False}
    saved = (_CRA.raster_into_gbuffer, _GSH.shade_frame,
             _GSH.shade_fragments_frame, _CH.try_stage)

    def _w(fn, flag):
        def inner(*a, **k):
            got = fn(*a, **k)
            first = got[0] if isinstance(got, tuple) else got
            if first is not None and first is not False:
                hit[flag] = True
            return got
        return inner

    _, cpu_refine, t_cpu_refine = _frame('CPU', draft=False)
    _, _cpu_d, t_cpu_draft = _frame('CPU', draft=True)
    (_CRA.raster_into_gbuffer, _GSH.shade_frame,
     _GSH.shade_fragments_frame, _CH.try_stage) = (
        _w(saved[0], 'R'), _w(saved[1], 'S'), _w(saved[2], 'S'),
        _w(saved[3], 'P'))
    try:
        _frame('GPU', draft=False)              # cold: plans + compiles
        vp_g, gpu_refine, t_gpu_refine = _frame('GPU', draft=False)
        _, _gpu_d, t_gpu_draft = _frame('GPU', draft=True)
    finally:
        (_CRA.raster_into_gbuffer, _GSH.shade_frame,
         _GSH.shade_fragments_frame, _CH.try_stage) = saved

    eng = ''.join(c if hit[c] else '-' for c in 'RSP')
    if cpu_refine.shape != gpu_refine.shape:
        _p(out, f'  FAILED: refine shapes differ '
                f'({cpu_refine.shape} vs {gpu_refine.shape})')
        return
    a = np.abs(cpu_refine - gpu_refine)
    fl = int((a.reshape(a.shape[0], a.shape[1], -1).max(axis=2)
              > 1e-2).sum())
    dw, dh = max(w // DRAFT_FACTOR, 4), max(h // DRAFT_FACTOR, 4)
    _p(out, f'  refine {w}x{h}, draft {dw}x{dh} '
            f'(preview scale 1, stages {eng})')
    _p(out, f'    max difference : {a.max():.6f}')
    _p(out, f'    px off by >0.01: {fl} of {w * h}')
    _p(out, f'    CPU refine     : {t_cpu_refine * 1000.0:7.1f} ms   '
            f'draft {t_cpu_draft * 1000.0:6.1f} ms')
    _p(out, f'    GPU refine     : {t_gpu_refine * 1000.0:7.1f} ms   '
            f'draft {t_gpu_draft * 1000.0:6.1f} ms  (warm, through the '
            f'worker path)')
    engaged = getattr(vp_g, 'last_engaged', None)
    if engaged == 'GPU' and fl == 0:
        _p(out, '    the switch reaches the viewport: drafts and refines '
                'shade on your driver, and')
        _p(out, '    a busy driver (an F12 mid-flight) falls back to the '
                'CPU frame with the reason printed')
    else:
        _p(out, f'    engagement letter reads {engaged!r} with {fl} px '
                f'off -- paste this section')


def feature_matrix(out):
    """Every feature, CPU device vs GPU device, on the REAL driver.

    Same rows the headless suite proves bit-exact without a driver. Here
    each row renders on both devices; the GPU column either reproduces
    the CPU picture within the deferred bar (the driver ran it) or routes
    to the CPU by name and matches exactly (the switch working as
    designed). Post-engaged rows compare against the stage table's own
    CLOSE claims (dither's ordered patterns differ up to ~0.03 by
    design). Wrong pixels are the only failure.
    """
    import time as _time

    from .core import post as _post
    from .core import render as R
    from .gpu import chain as _CH
    from .gpu import craster as _CRA
    from .gpu import shade as _GSH
    from .tests.featurematrix import ROWS, build

    _p(out)
    _p(out, LINE)
    _p(out, 'FEATURE x DEVICE MATRIX  (every feature, CPU device vs GPU '
            'device, 96x72)')
    _p(out, LINE)
    _p(out, '  stages column: R = compute raster engaged, S = deferred/'
            'layer shading engaged,')
    _p(out, '  P = post stage(s) engaged; "-" = that stage ran on the '
            'CPU (reasons print above)')
    _p(out)

    def _run(sc, st):
        img = R.render(sc, st)
        return _post.process(img, st, frame=1, seed=st.seed,
                             target_size=(st.resolution_x,
                                          st.resolution_y),
                             allow_resize=False,
                             depth=getattr(sc, 'last_depth', None),
                             shaft_sources=getattr(sc, 'last_shafts',
                                                   None))

    t_all = _time.perf_counter()
    n_gpu = n_routed = n_fail = 0
    saved = (_CRA.raster_into_gbuffer, _GSH.shade_frame,
             _GSH.shade_fragments_frame, _CH.try_stage)
    for key, _o, _s in ROWS:
        try:
            scC, stC = build(key)
            cpu = _run(scC, stC)
            scG, stG = build(key)
            stG.render_device = 'GPU'
            hit = {'R': False, 'S': False, 'P': False}

            def _w(fn, flag):
                def inner(*a, **k):
                    got = fn(*a, **k)
                    first = got[0] if isinstance(got, tuple) else got
                    # identity tests only: the raster returns a BOOL ok,
                    # the shaders an image-or-None, a stage array-or-None
                    if first is not None and first is not False:
                        hit[flag] = True
                    return got
                return inner

            _CRA.raster_into_gbuffer = _w(saved[0], 'R')
            _GSH.shade_frame = _w(saved[1], 'S')
            _GSH.shade_fragments_frame = _w(saved[2], 'S')
            _CH.try_stage = _w(saved[3], 'P')
            try:
                gpu = _run(scG, stG)
            finally:
                (_CRA.raster_into_gbuffer, _GSH.shade_frame,
                 _GSH.shade_fragments_frame, _CH.try_stage) = saved
            if cpu.shape != gpu.shape:
                verdict, d, fl = 'FAIL (shape)', -1.0, -1
            else:
                d = float(np.abs(np.asarray(cpu, np.float64)
                                 - gpu).max())
                a = np.abs(np.asarray(cpu, np.float64) - gpu)
                fl = int((a.reshape(a.shape[0], a.shape[1], -1)
                          .max(axis=2) > 1e-2).sum())
                bar = 0.05 if hit['P'] else 6e-3
                verdict = 'ok' if d <= bar else f'FAIL (bar {bar:g})'
            eng = ''.join(c if hit[c[0]] else '-' for c in 'RSP')
            full = hit['R'] and hit['S']
            if verdict == 'ok' and full:
                n_gpu += 1
            elif verdict == 'ok':
                n_routed += 1
            else:
                n_fail += 1
            _p(out, f'  {key:<34.34s} {d:9.6f}  {fl:6d}   {eng}'
                    + ('' if verdict == 'ok' else f'   {verdict}'))
        except Exception as _exc:                               # noqa: BLE001
            n_fail += 1
            _p(out, f'  {key:<34.34s} CRASHED: {type(_exc).__name__}: '
                    f'{_exc}')
    (_CRA.raster_into_gbuffer, _GSH.shade_frame,
     _GSH.shade_fragments_frame, _CH.try_stage) = saved
    _p(out)
    _p(out, f'  {len(ROWS)} rows in {_time.perf_counter() - t_all:.1f}s: '
            f'{n_gpu} raster+shade on the driver and matched, {n_routed} '
            f'partially routed to the CPU by name and matched, '
            f'{n_fail} FAILED')
    if n_fail == 0:
        _p(out, '  every feature works with the GPU device: the driver '
                'reproduces it, or the switch routes it honestly')
    else:
        _p(out, '  a FAILED row is a wrong picture -- paste this table')


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


def gpu_compute(out):
    """The compute rasteriser, run on the driver and diffed against fill().

    The round-19 probe proved the API; this section now runs the actual
    kernel -- the CPU rasteriser's exact rules, one thread per pixel over
    CPU-binned tiles -- on the demo scene, and counts DIFFERING PIXELS
    against the CPU G-buffer. The claim is exactness, so the number that
    matters is an integer, not a tolerance. Nothing here changes how
    anything renders yet: the raster still runs on the CPU until this
    section has proven the dispatch on real hardware.
    """
    _p(out)
    _p(out, LINE)
    _p(out, 'GPU COMPUTE RASTERISER  (the last stage, diffed on your driver)')
    _p(out, LINE)
    from .core import raster as CR
    from .core import render as R
    from .core import stats as ST
    from .core.settings import RenderSettings
    from .gpu import craster as CRA, device
    from .tests.scenebuild import demo_scene

    ok, why = device.probe()
    if not ok:
        _p(out, f'  skipped: {why}')
        return
    import time as _time
    w, h = 320, 240
    st = RenderSettings()
    st.resolution_x, st.resolution_y = w, h
    st.aa_samples = 1
    sc = demo_scene(st, with_texture=False)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)

    t0 = _time.perf_counter()
    g = CR.GBuffer(w, h)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    t_cpu = _time.perf_counter() - t0

    t0 = _time.perf_counter()
    outi, err = CRA.raster_on_device(sc.mesh, vp, w, h)
    t_gpu_cold = _time.perf_counter() - t0
    if outi is None:
        _p(out, f'  FAILED: {err}')
        return
    t0 = _time.perf_counter()
    outi, err = CRA.raster_on_device(sc.mesh, vp, w, h)
    t_gpu = _time.perf_counter() - t0

    import numpy as np
    ids = outi['ids']
    aux = outi['aux']
    tri = np.round(ids[:, :, 3]).astype(np.int32)
    diff_tri = int((tri != g.tri).sum())
    cov = g.tri >= 0
    ib = np.stack([g.bary[:, :, 0], g.bary[:, :, 1],
                   1.0 - g.bary[:, :, 0] - g.bary[:, :, 1]], -1)
    d_b = float(np.abs(ids[:, :, :3][cov] - ib[cov]).max()) if cov.any() \
        else 0.0
    d_z = float(np.abs(aux[:, :, 0][cov] - g.zndc[cov]).max())
    d_f = int((aux[:, :, 1][cov] > 0.5).astype(bool).sum()
              - g.front[cov].sum()) if cov.any() else 0
    _p(out, f'  frame           : {w}x{h}, {sc.mesh.tris.shape[0]} '
            'triangles')
    _p(out, f'  DIFFERING PIXELS: {diff_tri} of {w * h} (triangle ids)')
    _p(out, f'  bary max diff   : {d_b:.9f}')
    _p(out, f'  zndc max diff   : {d_z:.9f}')
    _p(out, f'  front mismatch  : {d_f}')
    _p(out, f'  CPU rasterise   : {t_cpu * 1000.0:7.1f} ms')
    _p(out, f'  compute cold    : {t_gpu_cold * 1000.0:7.1f} ms '
            '(compile + pack + dispatch + read)')
    _p(out, f'  compute warm    : {t_gpu * 1000.0:7.1f} ms '
            '(pack + dispatch + read back)')
    tm = outi.get('timings') or {}
    if tm:
        _p(out, f'    of which      : {tm.get("clip_ms", 0.0):5.1f} ms '
                f'clip+project, {tm.get("pack_ms", 0.0):5.1f} ms pack+bin, '
                f'{tm.get("upload_ms", 0.0):5.1f} ms upload, '
                f'{tm.get("dispatch_read_ms", 0.0):5.1f} ms dispatch+read')
    if diff_tri == 0:
        _p(out, '  the kernel IS fill(): every pixel picked the same '
                'triangle your CPU picked')
    else:
        _p(out, '  nonzero differing pixels: paste this back and the next '
                'round hunts them')

    # ---- the decision number: the same race at working size
    w2, h2 = 480, 360
    st2 = RenderSettings()
    st2.resolution_x, st2.resolution_y = w2, h2
    st2.aa_samples = 1
    sc2 = demo_scene(st2, with_texture=False)
    _v2, _p2b, vp2, _e2 = R.camera_matrices(sc2.camera, w2, h2)
    t0 = _time.perf_counter()
    g2 = CR.GBuffer(w2, h2)
    CR.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w2, h2, gbuf=g2)
    t_cpu2 = _time.perf_counter() - t0
    out2, err2 = CRA.raster_on_device(sc2.mesh, vp2, w2, h2)
    t0 = _time.perf_counter()
    out2, err2 = CRA.raster_on_device(sc2.mesh, vp2, w2, h2)
    t_gpu2 = _time.perf_counter() - t0
    _p(out)
    if out2 is None:
        _p(out, f'  at {w2}x{h2}: FAILED: {err2}')
        return
    tri2 = np.round(out2['ids'][:, :, 3]).astype(np.int32)
    d2 = int((tri2 != g2.tri).sum())
    _p(out, f'  at {w2}x{h2}     : {d2} differing px, CPU '
            f'{t_cpu2 * 1000.0:.1f} ms vs compute {t_gpu2 * 1000.0:.1f} ms '
            'warm')
    tm2 = out2.get('timings') or {}
    if tm2:
        _p(out, f'    of which      : {tm2.get("clip_ms", 0.0):5.1f} ms '
                f'clip+project, {tm2.get("pack_ms", 0.0):5.1f} ms pack+bin, '
                f'{tm2.get("upload_ms", 0.0):5.1f} ms upload, '
                f'{tm2.get("dispatch_read_ms", 0.0):5.1f} ms dispatch+read')

    # ---- both stages together: raster AND shade on the GPU, whole frame
    st3 = RenderSettings()
    st3.resolution_x, st3.resolution_y = w2, h2
    st3.aa_samples = 1
    t0 = _time.perf_counter()
    cpu_full = R.render(demo_scene(st3, with_texture=False), st3)
    t_cf = _time.perf_counter() - t0
    st4 = RenderSettings()
    st4.resolution_x, st4.resolution_y = w2, h2
    st4.aa_samples = 1
    st4.render_device = 'GPU'
    st4.gpu_shading = True
    st4.gpu_raster = True
    sc4 = demo_scene(st4, with_texture=False)
    R.render(sc4, st4)                       # cold: compiles both stages
    t0 = _time.perf_counter()
    gpu_full = R.render(sc4, st4)
    t_gf = _time.perf_counter() - t0
    dful = float(np.abs(gpu_full - cpu_full).max())
    _p(out)
    _p(out, f'  FULL FRAME, raster + shade on GPU at {w2}x{h2}:')
    _p(out, f'    max difference : {dful:.6f}')
    _p(out, f'    CPU everything : {t_cf * 1000.0:7.1f} ms')
    _p(out, f'    GPU both stages: {t_gf * 1000.0:7.1f} ms (warm, whole '
            'render() call)')

    # ---- the ray-tracing arc's first hardware number: any-hit traversal
    _p(out)
    _p(out, '  BVH OCCLUSION KERNEL (the ray-tracing arc begins):')
    from .core.bvh import BVH
    from .gpu import rtrace as RTR
    sc5 = demo_scene(st3, with_texture=False)
    bvh = BVH(sc5.mesh.verts, sc5.mesh.tris)
    view5, _p5, vp5, eye5 = R.camera_matrices(sc5.camera, w2, h2)
    g5 = CR.GBuffer(w2, h2)
    CR.rasterize(sc5.mesh.verts, sc5.mesh.tris, vp5, w2, h2, gbuf=g5)
    job5 = R.ShadeJob(sc5, st3, {}, None, view5, eye5, w2, h2)
    py5, px5 = np.nonzero(g5.tri >= 0)
    nr = min(py5.size, 40000)
    rng5 = np.random.default_rng(9)
    pick5 = rng5.choice(py5.size, nr, replace=False)
    ctx5 = job5.context(g5.tri[py5, px5][pick5], g5.bary[py5, px5][pick5],
                        px5[pick5], py5[pick5], np.ones(nr, bool),
                        None, 0, True)
    lp5 = np.asarray(sc5.lights[1].position, np.float32)
    delta5 = lp5[None, :] - ctx5.P
    dist5 = np.linalg.norm(delta5, axis=1)
    ldir5 = (delta5 / dist5[:, None]).astype(np.float32)
    sorg5 = (ctx5.P + ldir5 * 1e-3).astype(np.float32)
    stmax5 = (dist5 - 2e-3).astype(np.float32)
    t0 = _time.perf_counter()
    want5 = bvh.occluded(sorg5, ldir5, stmax5)
    t_cpu5 = _time.perf_counter() - t0
    got5, err5 = RTR.occluded_on_device(bvh, sorg5, ldir5, stmax5)
    if got5 is None:
        _p(out, f'    FAILED: {err5}')
        return
    t0 = _time.perf_counter()
    got5, err5 = RTR.occluded_on_device(bvh, sorg5, ldir5, stmax5)
    t_gpu5 = _time.perf_counter() - t0
    mism5 = int((got5 != want5).sum())
    _p(out, f'    {nr} shadow rays at the point light, '
            f'{sc5.mesh.tris.shape[0]} triangles')
    _p(out, f'    MISMATCHED RAYS: {mism5} of {nr}  '
            f'(CPU shadowed {int(want5.sum())})')
    _p(out, f'    CPU occluded   : {t_cpu5 * 1000.0:7.1f} ms')
    _p(out, f'    compute warm   : {t_gpu5 * 1000.0:7.1f} ms '
            '(pack + upload + dispatch + read)')
    if mism5 == 0:
        _p(out, '    the traversal IS bvh.occluded(): the ray-shadowed '
                'frame below stands on it')

    # ---- the integration: Shadow Method RAY through the whole pipeline,
    # rasterised AND shaded on the driver, against the CPU's own frame
    _p(out)
    _p(out, '  RAY-SHADOWED DEFERRED FRAME (Shadow Method RAY, whole '
            'render):')
    st6 = RenderSettings()
    st6.resolution_x, st6.resolution_y = w2, h2
    st6.aa_samples = 1
    st6.shadows = True
    st6.shadow_default = 'RAY'
    t0 = _time.perf_counter()
    cpu6 = R.render(demo_scene(st6, with_texture=False), st6)
    t_c6 = _time.perf_counter() - t0
    st7 = RenderSettings()
    st7.resolution_x, st7.resolution_y = w2, h2
    st7.aa_samples = 1
    st7.shadows = True
    st7.shadow_default = 'RAY'
    st7.render_device = 'GPU'
    st7.gpu_shading = True
    st7.gpu_raster = True
    sc7 = demo_scene(st7, with_texture=False)
    R.render(sc7, st7)             # cold: compiles the traversal in-frame
    t0 = _time.perf_counter()
    gpu6 = R.render(sc7, st7)
    t_g6 = _time.perf_counter() - t0
    d6 = float(np.abs(gpu6 - cpu6).max())
    flip6 = int((np.abs(gpu6 - cpu6).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {d6:.6f}')
    _p(out, f'    px off by >0.01: {flip6} of {w2 * h2} '
            '(a flipped shadow-edge ray would land here)')
    _p(out, f'    CPU everything : {t_c6 * 1000.0:7.1f} ms '
            '(build BVH + trace + shade)')
    _p(out, f'    GPU both stages: {t_g6 * 1000.0:7.1f} ms (warm, whole '
            'render() call)')
    if d6 < 6e-3:
        _p(out, '    hard ray-traced shadows shade on your driver')

    # ---- SOFT shadows + ambient occlusion: the deterministic-sampling
    # arc, on the driver. Every jittered ray is a pure function of
    # (pixel, sample, stream, seed) through the pattern hash and the
    # shared unit-circle table, so both devices draw the SAME rays --
    # the refusals this replaced were 'a random stream the CPU batch
    # order owns' and 'ambient occlusion is not ported'
    _p(out)
    _p(out, '  SOFT SHADOWS + AMBIENT OCCLUSION (deterministic sampling, '
            'whole render):')
    st9 = RenderSettings()
    st9.resolution_x, st9.resolution_y = w2, h2
    st9.aa_samples = 1
    st9.shadows = True
    st9.shadow_default = 'RAY'
    st9.shadow_samples = 4
    st9.ambient_occlusion = True
    st9.ao_samples = 4
    st9.ao_distance = 2.0
    st9.ao_intensity = 1.0
    sc9 = demo_scene(st9, with_texture=False)
    sc9.lights[1].radius = 0.8
    sc9.materials[2].reflect_level = 0.5
    st9.raytrace = True
    st9.ray_depth = 1
    t0 = _time.perf_counter()
    cpu9 = R.render(sc9, st9)
    t_c9 = _time.perf_counter() - t0
    st9g = RenderSettings()
    for f9 in ('resolution_x', 'resolution_y', 'aa_samples', 'shadows',
               'shadow_default', 'shadow_samples', 'ambient_occlusion',
               'ao_samples', 'ao_distance', 'ao_intensity', 'raytrace',
               'ray_depth'):
        setattr(st9g, f9, getattr(st9, f9))
    st9g.render_device = 'GPU'
    st9g.gpu_shading = True
    st9g.gpu_raster = True
    sc9g = demo_scene(st9g, with_texture=False)
    sc9g.lights[1].radius = 0.8
    sc9g.materials[2].reflect_level = 0.5
    R.render(sc9g, st9g)                     # cold
    t0 = _time.perf_counter()
    gpu9 = R.render(sc9g, st9g)
    t_g9 = _time.perf_counter() - t0
    d9 = float(np.abs(gpu9 - cpu9).max())
    flip9 = int((np.abs(gpu9 - cpu9).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {d9:.6f}')
    _p(out, f'    px off by >0.01: {flip9} of {w2 * h2} (a flipped '
            'jittered ray would land here)')
    _p(out, f'    CPU everything : {t_c9 * 1000.0:7.1f} ms')
    _p(out, f'    GPU everything : {t_g9 * 1000.0:7.1f} ms (warm, whole '
            'render() call)')
    if d9 < 6e-3:
        _p(out, '    soft penumbras and occluded creases, hash-jittered '
                'identically on both devices -- two more refusals gone')

    # ---- the reflections arc's kernel: closest-hit, on the driver
    _p(out)
    _p(out, '  CLOSEST-HIT KERNEL (reflections are next):')
    from .core import mathx as _M
    py7, px7 = np.nonzero(g5.tri >= 0)
    nr7 = min(py7.size, 40000)
    rng7 = np.random.default_rng(21)
    pick7 = rng7.choice(py7.size, nr7, replace=False)
    ctx7 = job5.context(g5.tri[py7, px7][pick7], g5.bary[py7, px7][pick7],
                        px7[pick7], py7[pick7], np.ones(nr7, bool),
                        None, 0, True)
    N7 = _M.normalize(ctx7.N)
    V7 = -_M.normalize(ctx7.I)
    R7 = _M.reflect(-V7, N7).astype(np.float32)
    org7 = (ctx7.P + N7 * 1e-3).astype(np.float32)
    tmax7 = np.full(nr7, 1e30, np.float32)
    t0 = _time.perf_counter()
    wid7, wt7, wu7, wv7 = bvh.intersect(org7, R7, tmax7)
    t_cpu7 = _time.perf_counter() - t0
    got7, err7 = RTR.intersect_on_device(bvh, org7, R7, tmax7)
    if got7 is None:
        _p(out, f'    FAILED: {err7}')
        return
    t0 = _time.perf_counter()
    got7, err7 = RTR.intersect_on_device(bvh, org7, R7, tmax7)
    t_gpu7 = _time.perf_counter() - t0
    gid7, gt7, gu7, gv7 = got7
    mism7 = int((gid7 != wid7).sum())
    both7 = (wid7 >= 0) & (gid7 == wid7)
    dt7 = float(np.abs(gt7[both7] - wt7[both7]).max()) if both7.any() else 0.0
    duv7 = float(max(np.abs(gu7[both7] - wu7[both7]).max(),
                     np.abs(gv7[both7] - wv7[both7]).max())) \
        if both7.any() else 0.0
    _p(out, f'    {nr7} reflection rays off real surfaces, '
            f'{sc5.mesh.tris.shape[0]} triangles')
    _p(out, f'    MISMATCHED HITS: {mism7} of {nr7}  '
            f'(CPU hit {int((wid7 >= 0).sum())}, ties decided by the '
            'baked visit order)')
    _p(out, f'    t max diff     : {dt7:.9f}   bary max diff: {duv7:.9f}')
    _p(out, f'    CPU intersect  : {t_cpu7 * 1000.0:7.1f} ms')
    _p(out, f'    compute warm   : {t_gpu7 * 1000.0:7.1f} ms '
            '(pack + upload + dispatch + read)')
    if mism7 == 0:
        _p(out, '    the traversal IS bvh.intersect(): the reflected '
                'frame below stands on it')

    # ---- the arc's integration: one traced bounce, whole pipeline
    _p(out)
    _p(out, '  RAY-REFLECTED DEFERRED FRAME (ray tracing ON, one bounce, '
            'whole render):')
    st8 = RenderSettings()
    st8.resolution_x, st8.resolution_y = w2, h2
    st8.aa_samples = 1
    st8.shadows = True
    st8.raytrace = True
    st8.ray_depth = 1
    sc8 = demo_scene(st8, with_texture=False)
    sc8.materials[2].reflect_level = 0.5      # the Box becomes a mirror
    t0 = _time.perf_counter()
    cpu8 = R.render(sc8, st8)
    t_c8 = _time.perf_counter() - t0
    st9 = RenderSettings()
    st9.resolution_x, st9.resolution_y = w2, h2
    st9.aa_samples = 1
    st9.shadows = True
    st9.raytrace = True
    st9.ray_depth = 1
    st9.render_device = 'GPU'
    st9.gpu_shading = True
    st9.gpu_raster = True
    sc9 = demo_scene(st9, with_texture=False)
    sc9.materials[2].reflect_level = 0.5
    R.render(sc9, st9)          # cold: compiles primary + secondary passes
    t0 = _time.perf_counter()
    gpu8 = R.render(sc9, st9)
    t_g8 = _time.perf_counter() - t0
    d8 = float(np.abs(gpu8 - cpu8).max())
    bad8 = np.abs(gpu8 - cpu8).max(axis=2) > 1e-2
    flip8 = int(bad8.sum())
    _p(out, f'    max difference : {d8:.6f}')
    _p(out, f'    px off by >0.01: {flip8} of {w2 * h2}')
    _p(out, f'    CPU everything : {t_c8 * 1000.0:7.1f} ms '
            '(build BVH + trace + recursive shade)')
    _p(out, f'    GPU everything : {t_g8 * 1000.0:7.1f} ms (warm, whole '
            'render() call)')
    from .gpu import shade as _GSH
    tm8 = dict(_GSH.LAST_TIMINGS)
    if tm8.get('reflect_ms'):
        _p(out, f'    of which trace + secondary passes: '
                f'{tm8["reflect_ms"]:.1f} ms')
    if d8 < 6e-3:
        _p(out, '    one traced bounce shades on your driver -- the '
                'refusal this arc existed to lift is lifting')

    # ---- refraction joins, in the FIELD's own shape: the Ball becomes
    # normal-mapped WAVY glass (a master graph with a Normal chain -- the
    # rays bend through _ray_context's CPU-exact evaluation) and the Box
    # stays a mirror, both sweeps walking one tree in one frame
    _p(out)
    _p(out, '  RAY-REFRACTED DEFERRED FRAME (NOISE-into-BUMP wavy glass '
            '+ mirror -- the exact Water anatomy, whole render):')
    from .tests.scenebuild import ImageBuffer as _IB
    from .tests.scenebuild import add_normal_mapped_ball as _wavy

    def _glass(sc):
        # the field's EXACT Water anatomy: Blender NOISE through a BUMP
        # node into the master Normal socket. The noise is the sin-fract
        # family the GLSL emitter refuses by name -- so its height image
        # evaluates on the CPU, float64 sin and all, and the GPU takes
        # its neighbour differences from that image by texelFetch
        _wavy(sc)                             # full master socket list
        gr = sc.materials[1].graph

        def _sk(nm, t, d, l=None):
            return {'name': nm, 'type': t, 'default': d, 'link': l}

        gr['nodes']['noise'] = {
            'id': 'noise', 'bl_idname': 'ShaderNodeTexNoise',
            'props': {},
            'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                       _sk('Scale', 'VALUE', 6.0),
                       _sk('Detail', 'VALUE', 2.0),
                       _sk('Roughness', 'VALUE', 0.5),
                       _sk('Distortion', 'VALUE', 0.0)],
            'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                        {'name': 'Color', 'type': 'RGBA'}]}
        gr['nodes']['bump'] = {
            'id': 'bump', 'bl_idname': 'ShaderNodeBump',
            'props': {'invert': False},
            'inputs': [_sk('Strength', 'VALUE', 0.8),
                       _sk('Distance', 'VALUE', 0.6),
                       _sk('Height', 'VALUE', 0.5, ['noise', 0]),
                       _sk('Normal', 'VECTOR', [0, 0, 0])],
            'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
        for sock in gr['nodes']['hal']['inputs']:
            if sock['name'] == 'Normal':
                sock['link'] = ['bump', 0]
            if sock['name'] == 'Diffuse Color':
                sock['link'] = None           # flat tint: the lerp needs it
                sock['default'] = [0.2, 0.5, 0.8, 1.0]
            if sock['name'] == 'Opacity':
                sock['default'] = 0.4
            if sock['name'] == 'IOR':
                sock['default'] = 1.33
            if sock['name'] == 'Reflection':
                sock['default'] = 0.3         # water reflects the sky too
        sc.materials[2].reflect_level = 0.5   # the Box stays a mirror
        # and the sky is the field's own: BANDS -- quantised gradient,
        # now expressed in the env term the reflections sample
        sc.world.mode = 'BANDS'
        sc.world.band_count = 6
        sc.world.band_softness = 0.15

    stA = RenderSettings()
    stA.resolution_x, stA.resolution_y = w2, h2
    stA.aa_samples = 1
    stA.shadows = True
    stA.raytrace = True
    stA.ray_depth = 1
    stA.transparency = 'NONE'
    scA = demo_scene(stA, with_texture=False)
    _glass(scA)
    t0 = _time.perf_counter()
    cpuA = R.render(scA, stA)
    t_cA = _time.perf_counter() - t0
    stB = RenderSettings()
    stB.resolution_x, stB.resolution_y = w2, h2
    stB.aa_samples = 1
    stB.shadows = True
    stB.raytrace = True
    stB.ray_depth = 1
    stB.transparency = 'NONE'
    stB.render_device = 'GPU'
    stB.gpu_shading = True
    stB.gpu_raster = True
    scB = demo_scene(stB, with_texture=False)
    _glass(scB)
    R.render(scB, stB)                        # cold
    ST.reset()
    ST.enable(True)
    t0 = _time.perf_counter()
    gpuA = R.render(scB, stB)
    t_gA = _time.perf_counter() - t0
    ST.enable(False)
    dA = float(np.abs(gpuA - cpuA).max())
    flipA = int((np.abs(gpuA - cpuA).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {dA:.6f}')
    _p(out, f'    px off by >0.01: {flipA} of {w2 * h2}')
    _p(out, f'    CPU everything : {t_cA * 1000.0:7.1f} ms')
    _p(out, f'    GPU everything : {t_gA * 1000.0:7.1f} ms (warm, whole '
            'render() call)')
    tmA = dict(_GSH.LAST_TIMINGS)
    _p(out, '    shade split    : '
            f'plan {tmA.get("plan_ms", 0.0):.1f}, '
            f'pack+upload {tmA.get("pack_upload_ms", 0.0):.1f}, '
            f'draw+read {tmA.get("draw_read_ms", 0.0):.1f}, '
            f'sweeps {tmA.get("reflect_ms", 0.0):.1f} '
            f'(of which CPU ray build '
            f'{tmA.get("ray_build_ms", 0.0):.1f}), '
            f'other {tmA.get("composite_ms", 0.0):.1f} ms')
    _p(out, '    the warm frame, stage by stage (where every ms went):')
    ST.report(total=t_gA, printer=lambda s: _p(out, '    ' + s))
    # the frame above ran through render(): when the driver REJECTS
    # shade_frame, render() shades on the CPU and the match still reads
    # 0.000012 -- CPU against CPU. The reason goes to the Blender
    # console, invisible here. So ask shade_frame directly and put the
    # answer IN the report: if both 'shade (GPU)' and 'shade' appear in
    # the stage table above, THIS line is why
    try:
        viewB, _projB, vpB, eyeB = R.camera_matrices(scB.camera, w2, h2)
        gB = CR.GBuffer(w2, h2)
        CR.rasterize(scB.mesh.verts, scB.mesh.tris, vpB, w2, h2, gbuf=gB)
        texB = R.prepare_textures(scB, stB)
        bvhB = R._cached_bvh(scB, scB.mesh)
        jobB = R.ShadeJob(scB, stB, texB, bvhB, viewB, eyeB, w2, h2)
        gotB, whyB = _GSH.shade_frame(jobB, gB)
        if gotB is not None:
            _p(out, '    shade_frame directly: OK (the driver shaded '
                    'this frame)')
        else:
            _p(out, f'    shade_frame directly: FELL BACK: {whyB}')
    except Exception as _exc:                                    # noqa: BLE001
        _p(out, f'    shade_frame directly: probe itself failed: {_exc}')
    if dA < 6e-3:
        _p(out, '    NOISE-driven glass refracts through its bent rays -- '
                'the sin-fract height evaluates on the CPU into the '
                'pre-pass, exactly -- and the mirror reflects. The Water '
                'anatomy works on your driver')
        # ---- ray depth 2: the recursion tree, on the driver. Two
        # mirrors and branching glass make mirror-in-mirror; the second
        # bounce composites backward with the HIT material's constants
        _p(out)
        _p(out, '  RAY DEPTH 2 (the recursion tree, whole render):')
        stD2 = RenderSettings()
        stD2.resolution_x, stD2.resolution_y = w2, h2
        stD2.aa_samples = 1
        stD2.shadows = True
        stD2.raytrace = True
        stD2.ray_depth = 2
        stD2.transparency = 'NONE'
        scD2 = demo_scene(stD2, with_texture=False)
        scD2.materials[2].reflect_level = 0.6
        scD2.materials[0].reflect_level = 0.4
        scD2.materials[1].opacity = 0.5
        scD2.materials[1].ior = 1.2
        t0 = _time.perf_counter()
        cpuD2 = R.render(scD2, stD2)
        t_cD2 = _time.perf_counter() - t0
        stD2g = RenderSettings()
        for fD in ('resolution_x', 'resolution_y', 'aa_samples', 'shadows',
                   'raytrace', 'ray_depth', 'transparency'):
            setattr(stD2g, fD, getattr(stD2, fD))
        stD2g.render_device = 'GPU'
        stD2g.gpu_shading = True
        stD2g.gpu_raster = True
        scD2g = demo_scene(stD2g, with_texture=False)
        scD2g.materials[2].reflect_level = 0.6
        scD2g.materials[0].reflect_level = 0.4
        scD2g.materials[1].opacity = 0.5
        scD2g.materials[1].ior = 1.2
        R.render(scD2g, stD2g)               # cold
        t0 = _time.perf_counter()
        gpuD2 = R.render(scD2g, stD2g)
        t_gD2 = _time.perf_counter() - t0
        dD2 = float(np.abs(gpuD2 - cpuD2).max())
        flipD2 = int((np.abs(gpuD2 - cpuD2).max(axis=2) > 1e-2).sum())
        _p(out, f'    max difference : {dD2:.6f}')
        _p(out, f'    px off by >0.01: {flipD2} of {w2 * h2}')
        _p(out, f'    CPU everything : {t_cD2 * 1000.0:7.1f} ms')
        _p(out, f'    GPU everything : {t_gD2 * 1000.0:7.1f} ms (warm, '
                'whole render() call)')
        if dD2 < 6e-3:
            _p(out, '    mirror-in-mirror on your driver -- the ray arc '
                    'is COMPLETE: shadows hard and soft, occlusion, '
                    'reflections, refraction, any depth')
        # ---- every world reflects: the env term for rich skies is
        # evaluated by the renderer itself along the reflected rays and
        # composited after readback -- exact for ANY world, the Bryce
        # sky lab included. Six refusals fell to one mechanism.
        _p(out)
        _p(out, '  RICH WORLDS IN REFLECTIONS (the Bryce sky lab in a '
                'mirror, whole render):')
        from .core.scene import World as _World
        stW = RenderSettings()
        stW.resolution_x, stW.resolution_y = w2, h2
        stW.aa_samples = 1
        stW.shadows = True
        stW.raytrace = True
        stW.ray_depth = 1
        scW = demo_scene(stW, with_texture=False)
        scW.materials[2].reflect_level = 0.6
        scW.materials[0].reflect_level = 0.3
        scW.world = _World()
        scW.world.mode = 'BRYCE'
        t0 = _time.perf_counter()
        cpuW = R.render(scW, stW)
        t_cW = _time.perf_counter() - t0
        stWg = RenderSettings()
        for fW in ('resolution_x', 'resolution_y', 'aa_samples', 'shadows',
                   'raytrace', 'ray_depth'):
            setattr(stWg, fW, getattr(stW, fW))
        stWg.render_device = 'GPU'
        stWg.gpu_shading = True
        stWg.gpu_raster = True
        scWg = demo_scene(stWg, with_texture=False)
        scWg.materials[2].reflect_level = 0.6
        scWg.materials[0].reflect_level = 0.3
        scWg.world = _World()
        scWg.world.mode = 'BRYCE'
        R.render(scWg, stWg)                 # cold
        t0 = _time.perf_counter()
        gpuW = R.render(scWg, stWg)
        t_gW = _time.perf_counter() - t0
        dW = float(np.abs(gpuW - cpuW).max())
        flipW = int((np.abs(gpuW - cpuW).max(axis=2) > 1e-2).sum())
        _p(out, f'    max difference : {dW:.6f}')
        _p(out, f'    px off by >0.01: {flipW} of {w2 * h2}')
        _p(out, f'    CPU everything : {t_cW * 1000.0:7.1f} ms')
        _p(out, f'    GPU everything : {t_gW * 1000.0:7.1f} ms (warm, '
                'whole render() call)')
        if dW < 6e-3:
            _p(out, '    the sky lab reflects exactly -- STARFIELD, '
                    'BRYCE, PHYSICAL, HDRI, world graphs and the ground '
                    'plane all travel now, evaluated by the renderer '
                    'itself along the reflected rays')
    elif flipA:
        # this frame's GLSL (the bump main pass, the secondary passes of
        # a two-sweep frame) had its FIRST real driver run when the
        # glued-declaration fix landed -- a handful of off pixels now has
        # several possible mechanisms. Classify the bad pixels, then
        # re-render with ONE suspect disabled at a time: whichever
        # variant zeroes the count names the mechanism for next round
        _refracted_diagnosis(out, scB, stB, cpuA, gpuA, w2, h2)
    elif flip8:
        # localize the disagreement so the next round aims, not guesses:
        # rebuild the frame's exact reflection rays, trace them on BOTH
        # devices, and say whose pixels the bad ones are
        _p(out, '    DIAGNOSIS (the numbers the next round needs):')
        try:
            g8 = CR.GBuffer(w2, h2)
            CR.rasterize(sc8.mesh.verts, sc8.mesh.tris, vp5, w2, h2,
                         gbuf=g8)
            job8 = R.ShadeJob(sc8, st8, {}, bvh, view5, eye5, w2, h2)
            _GSH._PLAN_CACHE.clear()
            p8p, why8p, a8p = _GSH.plan_frame(job8, g8)
            rp8 = (a8p or {}).get('__reflect') if p8p is not None else None
            if rp8 is None:
                _p(out, f'      (could not re-plan: {why8p})')
            else:
                py8, px8, org8, dir8 = _GSH._reflection_rays(job8, g8, rp8)
                wid8, wt8, wu8, wv8 = bvh.intersect(
                    org8, dir8, np.full(py8.size, 1e30, np.float32))
                dev8, errd8 = RTR.intersect_frame(bvh, org8, dir8)
                mirror8 = np.zeros(bad8.shape, bool)
                mirror8[py8, px8] = True
                on8 = int((bad8 & mirror8).sum())
                off8 = int((bad8 & ~mirror8).sum())
                _p(out, f'      bad px on the mirror: {on8}   '
                        f'elsewhere: {off8}')
                if dev8 is None:
                    _p(out, f'      frame-ray device trace FAILED: {errd8}')
                else:
                    gid8 = dev8[0]
                    tmis8 = int((gid8 != wid8).sum())
                    _p(out, f'      frame-ray trace agreement: {tmis8} of '
                            f'{py8.size} hit ids differ (CPU vs device, '
                            'the exact rays)')
                    if tmis8:
                        fl = np.zeros(bad8.shape, bool)
                        sel = gid8 != wid8
                        fl[py8[sel], px8[sel]] = True
                        _p(out, f'      bad px that are ALSO trace-'
                                f'flipped: {int((bad8 & fl).sum())}')
                        # the same rays through FRESH textures (the
                        # kernel path): separates texture state from
                        # arithmetic
                        onv8, erron8 = RTR.intersect_on_device(
                            bvh, org8, dir8,
                            np.full(py8.size, 1e30, np.float32))
                        if onv8 is not None:
                            _p(out, '      same rays, fresh-texture '
                                    'kernel path: '
                                    f'{int((onv8[0] != wid8).sum())} '
                                    'differ')
                        # determinism: the cached path against itself
                        dev8b, _e2 = RTR.intersect_frame(bvh, org8, dir8)
                        if dev8b is not None:
                            _p(out, '      cached path vs itself: '
                                    f'{int((dev8b[0] != gid8).sum())} '
                                    'differ (nonzero = nondeterministic)')
                        # what the flips look like: near-t neighbours
                        # (arithmetic) or wild values (misread data)
                        idxs = np.nonzero(sel)[0][:3]
                        for k in idxs:
                            _p(out, f'      flip: cpu id {int(wid8[k])} '
                                    f't {float(wt8[k]):.6g}  ->  dev id '
                                    f'{int(gid8[k])} t '
                                    f'{float(dev8[1][k]):.6g}')
                    else:
                        missed8 = wid8 < 0
                        mm8 = np.zeros(bad8.shape, bool)
                        mm8[py8[missed8], px8[missed8]] = True
                        _p(out, '      the trace agrees ray for ray -- '
                                'of the bad mirror px, '
                                f'{int((bad8 & mm8).sum())} are ray '
                                'MISSES (world_color composite) and '
                                f'{int((bad8 & mirror8 & ~mm8).sum())} '
                                'are hits (secondary-pass shading)')
        except Exception as _exc:                               # noqa: BLE001
            _p(out, f'      diagnosis itself failed: {_exc}')

    # ---- GOURAUD / FLAT RATES: 'the scene shading rate is VERTEX' was
    # the refusal every console-preset frame hit -- most period presets
    # select Gouraud on purpose. The CPU lights the corners over white
    # (its own code: shadows, env, the model formula), the driver
    # interpolates and multiplies by the per-pixel albedo -- MODULATE.
    _p(out)
    _p(out, '  GOURAUD / FLAT SHADING RATES (textured demo, whole '
            'render):')
    for rateG, lblG in (('VERTEX', 'Gouraud'), ('FACE', 'flat  ')):
        stGc = RenderSettings()
        stGc.resolution_x, stGc.resolution_y = w2, h2
        stGc.aa_samples = 1
        stGc.shadows = True
        stGc.shading_rate = rateG
        scGc = demo_scene(stGc, with_texture=True)
        t0 = _time.perf_counter()
        cpuG = R.render(scGc, stGc)
        t_cG = _time.perf_counter() - t0
        stGg = RenderSettings()
        stGg.resolution_x, stGg.resolution_y = w2, h2
        stGg.aa_samples = 1
        stGg.shadows = True
        stGg.shading_rate = rateG
        stGg.render_device = 'GPU'
        stGg.gpu_shading = True
        stGg.gpu_raster = True
        scGg = demo_scene(stGg, with_texture=True)
        R.render(scGg, stGg)                 # cold: compiles the passes
        t0 = _time.perf_counter()
        gpuG = R.render(scGg, stGg)
        t_gG = _time.perf_counter() - t0
        dG = float(np.abs(gpuG - cpuG).max())
        flG = int((np.abs(gpuG - cpuG).max(axis=2) > 1e-2).sum())
        _p(out, f'    {lblG} : max {dG:.6f}, px off by >0.01: {flG} of '
                f'{w2 * h2}; CPU {t_cG * 1000.0:7.1f} ms, GPU '
                f'{t_gG * 1000.0:7.1f} ms (warm, whole render)')
    _p(out, '    banded vertex light over a sharp per-pixel texture, on '
            'your driver -- the refusal every Gouraud preset hit is '
            'lifted')

    # ---- PROJECTED LIGHT TEXTURES: sixth-generation projective
    # texturing. A spot throws its image through the cone (the slide
    # projector), a sun tiles its image across the world (the cloud
    # shadow); the GLSL loop samples with the CPU's own bilinear texel
    # arithmetic, and the headless mirror already matches at 0.000000 --
    # this block is the DRIVER's number for the same frame.
    _p(out)
    _p(out, '  PROJECTED LIGHT TEXTURES (spot gobo + tiled sun cookie, '
            'whole render):')

    def _cookie_scene(stk):
        sck = demo_scene(stk, with_texture=False)
        from .core.scene import Light as _L
        ckp = np.zeros((8, 8, 4), np.float32)
        ckp[:, :, 3] = 1.0
        ckp[::2, ::2, :3] = 1.0
        ckp[1::2, 1::2, :3] = 1.0
        ckp[:, :, 1] *= 0.3
        ckp[0, :, 2] = 1.0
        spotk = _L(type='SPOT', name='proj', position=(0.0, -4.0, 6.0),
                   direction=(0.0, 0.45, -0.9), color=(1.0, 1.0, 1.0),
                   energy=800.0, spot_size=1.0, spot_blend=0.2,
                   shadow='NONE', decay='INVERSE_SQUARE')
        spotk.cookie = ckp
        sunk = _L(type='SUN', name='clouds', direction=(-0.5, 0.4, -0.75),
                  color=(1.0, 0.97, 0.9), energy=3.0, shadow='NONE')
        sunk.cookie = ckp
        sunk.cookie_scale = 3.0
        sck.lights = [spotk, sunk]
        return sck

    stKc = RenderSettings()
    stKc.resolution_x, stKc.resolution_y = w2, h2
    stKc.aa_samples = 1
    stKc.shadows = False
    t0 = _time.perf_counter()
    cpuK = R.render(_cookie_scene(stKc), stKc)
    t_cK = _time.perf_counter() - t0
    stKg = RenderSettings()
    stKg.resolution_x, stKg.resolution_y = w2, h2
    stKg.aa_samples = 1
    stKg.shadows = False
    stKg.render_device = 'GPU'
    stKg.gpu_shading = True
    stKg.gpu_raster = True
    scKg = _cookie_scene(stKg)
    R.render(scKg, stKg)                     # cold
    t0 = _time.perf_counter()
    gpuK = R.render(scKg, stKg)
    t_gK = _time.perf_counter() - t0
    dK = float(np.abs(gpuK - cpuK).max())
    flK = int((np.abs(gpuK - cpuK).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {dK:.6f}')
    _p(out, f'    px off by >0.01: {flK} of {w2 * h2}')
    _p(out, f'    CPU everything : {t_cK * 1000.0:7.1f} ms')
    _p(out, f'    GPU everything : {t_gK * 1000.0:7.1f} ms (warm, whole '
            'render() call)')
    if dK < 6e-3:
        _p(out, '    the projector and the cloud shadow light on your '
                'driver -- set an image on any Spot or Sun lamp')

    # ---- TRANSPARENT LAYERS: the A-buffer's fragments shade per depth
    # layer through the same deferred machinery -- the stage the summary
    # line named on the field frame ('transparency shading 25.7s' of
    # 33.7). Two glass materials at once, both faces of each, the env
    # term on a layer, the real alpha chain, the additive merge.
    _p(out)
    _p(out, '  TRANSPARENT LAYERS (two-material glass, whole render):')
    stT = RenderSettings()
    stT.resolution_x, stT.resolution_y = w2, h2
    stT.aa_samples = 1
    stT.shadows = True
    scT = demo_scene(stT, with_texture=False)
    scT.materials[1].opacity = 0.5
    scT.materials[1].reflect_level = 0.25
    scT.materials[2].opacity = 0.6
    t0 = _time.perf_counter()
    cpuT = R.render(scT, stT)
    t_cT = _time.perf_counter() - t0
    stTg = RenderSettings()
    stTg.resolution_x, stTg.resolution_y = w2, h2
    stTg.aa_samples = 1
    stTg.shadows = True
    stTg.render_device = 'GPU'
    stTg.gpu_shading = True
    stTg.gpu_raster = True
    stTg.layer_gpu_min_frac = 0.0        # every layer on the driver:
    #                                      this section PROVES the GPU
    #                                      machinery, so nothing routes
    #                                      away from it
    scTg = demo_scene(stTg, with_texture=False)
    scTg.materials[1].opacity = 0.5
    scTg.materials[1].reflect_level = 0.25
    scTg.materials[2].opacity = 0.6
    R.render(scTg, stTg)                     # cold: compiles the layers
    t0 = _time.perf_counter()
    gpuT = R.render(scTg, stTg)
    t_gT = _time.perf_counter() - t0
    dT = float(np.abs(gpuT - cpuT).max())
    flipT = int((np.abs(gpuT - cpuT).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {dT:.6f}')
    _p(out, f'    px off by >0.01: {flipT} of {w2 * h2}')
    _p(out, f'    CPU everything : {t_cT * 1000.0:7.1f} ms')
    _p(out, f'    GPU everything : {t_gT * 1000.0:7.1f} ms (warm, whole '
            'render() call -- the opaque frame AND every glass layer)')
    if dT < 6e-3:
        _p(out, '    the glass shades on your driver, layer by layer -- '
                'the stage the 33.7s field frame spent 25.7s on')
    else:
        # corner the disagreement at the FRAGMENT, not the pixel: the
        # exact CPU call the compositor makes vs the driver's layer
        # draws, on identical sorted ranks. The histograms name the
        # mechanism class -- alpha-only flips point at the alpha/blend
        # path, one-rank clustering at the merge, one-material
        # clustering at its pass, backface clustering at facing, and a
        # nonzero driver-vs-itself count at nondeterminism.
        try:
            viewT, _pT, vpT, eyeT = R.camera_matrices(scT.camera, w2, h2)
            opqT, trT = R._split_by_alpha(scT, scT.mesh, stT)
            gT = CR.GBuffer(w2, h2)
            CR.rasterize(scT.mesh.verts, scT.mesh.tris, vpT, w2, h2,
                         subset=opqT, gbuf=gT,
                         depth_bits=stT.depth_precision)
            fT = CR.FragmentList()
            CR.rasterize(scT.mesh.verts, scT.mesh.tris, vpT, w2, h2,
                         cull='NONE', subset=trT, gbuf=gT, frags=fT,
                         depth_write=False,
                         depth_bits=stT.depth_precision)
            jobT = R.ShadeJob(scT, stT, {}, None, viewT, eyeT, w2, h2)
            pxT, pyT, triT, depT, barT, froT = fT.finish()
            kT = depT <= CR.abuf_depth_limit(gT.depth[pyT, pxT])
            pxT, pyT, triT, depT, barT, froT = (a[kT] for a in
                                                (pxT, pyT, triT, depT,
                                                 barT, froT))
            centT = scT.mesh.verts[scT.mesh.tris].mean(axis=1)
            vzT = np.abs((centT - jobT.eye[None, :])
                         @ jobT.view[:3, :3].T)[:, 2]
            keyT = vzT[triT].astype(np.float32)
            pixT = pyT.astype(np.int64) * gT.width + pxT
            oT = np.lexsort((keyT, pixT))
            pixT, pxT, pyT, triT, barT, froT = (a[oT] for a in
                                                (pixT, pxT, pyT, triT,
                                                 barT, froT))
            grT = np.zeros(pixT.size, np.int64)
            ngT = np.nonzero(pixT[1:] != pixT[:-1])[0] + 1
            grT[ngT] = ngT
            np.maximum.accumulate(grT, out=grT)
            rkT = np.arange(pixT.size, dtype=np.int64) - grT
            cpuC = R._shade_chunked(jobT, triT, barT, pxT, pyT, froT,
                                    None, stT)
            _GSH._PLAN_CACHE.clear()
            drvC, whyC = _GSH.shade_fragments_frame(jobT, gT, triT,
                                                    barT, pxT, pyT, rkT)
            _p(out, '    DIAGNOSIS (fragment-level, the numbers the '
                    'next round needs):')
            if drvC is None:
                _p(out, f'      the driver re-shade refused: {whyC}')
            else:
                dC = np.abs(drvC - cpuC).max(axis=1)
                badC = dC > 1e-2
                _p(out, f'      fragment flips >0.01: '
                        f'{int(badC.sum())} of {triT.size}   '
                        f'max {float(dC.max()):.6f}')
                if badC.any():
                    daT = np.abs(drvC[:, 3] - cpuC[:, 3]) > 1e-2
                    drT = np.abs(drvC[:, :3]
                                 - cpuC[:, :3]).max(axis=1) > 1e-2
                    _p(out, f'      channel structure: rgb-only '
                            f'{int((badC & drT & ~daT).sum())}, '
                            f'alpha-only '
                            f'{int((badC & daT & ~drT).sum())}, both '
                            f'{int((badC & daT & drT).sum())}')
                    mTd = scT.mesh.mat_index[triT] \
                        if scT.mesh.mat_index is not None else \
                        np.zeros(triT.size, np.int32)
                    _p(out, '      by rank: ' + str(
                        {int(r): int((badC & (rkT == r)).sum())
                         for r in np.unique(rkT[badC])}))
                    _p(out, '      by material: ' + str(
                        {int(m): int((badC & (mTd == m)).sum())
                         for m in np.unique(mTd[badC])}))
                    _p(out, f'      by facing: front '
                            f'{int((badC & froT).sum())}, back '
                            f'{int((badC & ~froT).sum())}')
                    drv2, _w2d = _GSH.shade_fragments_frame(
                        jobT, gT, triT, barT, pxT, pyT, rkT)
                    if drv2 is not None:
                        ndT = int((np.abs(drv2 - drvC).max(axis=1)
                                   > 1e-4).sum())
                        _p(out, f'      driver vs itself: {ndT} '
                                'fragments differ (nonzero = '
                                'nondeterministic)')
                    for kf in np.argsort(dC)[::-1][:3]:
                        kf = int(kf)
                        _p(out, f'      flip: rank {int(rkT[kf])} mat '
                                f'{int(mTd[kf])} front '
                                f'{bool(froT[kf])}  cpu '
                                f'{np.round(cpuC[kf], 4).tolist()}  drv '
                                f'{np.round(drvC[kf], 4).tolist()}')
        except Exception as _excT:                              # noqa: BLE001
            _p(out, f'      (fragment diagnosis failed: {_excT})')

    # ---- RAY-TRACED GLASS LAYERS: the field's last named refusal
    # ('transparent layers under ray tracing recurse on the CPU',
    # 25.1s of the 33.3s frame). Each layer's fragments are now a
    # PRIMARY surface for the same sweeps the opaque frame runs:
    # refraction through the glass, reflections among the mirrors, the
    # recursion tree, the alpha chain riding untouched.
    _p(out)
    _p(out, '  RAY-TRACED GLASS LAYERS (glass + mirrors, ray depth 1, '
            'whole render):')
    stR = RenderSettings()
    stR.resolution_x, stR.resolution_y = w2, h2
    stR.aa_samples = 1
    stR.shadows = True
    stR.raytrace = True
    stR.ray_depth = 1
    scR = demo_scene(stR, with_texture=False)
    scR.materials[1].opacity = 0.5
    scR.materials[0].reflect_level = 0.2
    scR.materials[2].reflect_level = 0.5
    t0 = _time.perf_counter()
    cpuR = R.render(scR, stR)
    t_cR = _time.perf_counter() - t0
    stRg = RenderSettings()
    stRg.resolution_x, stRg.resolution_y = w2, h2
    stRg.aa_samples = 1
    stRg.shadows = True
    stRg.raytrace = True
    stRg.ray_depth = 1
    stRg.render_device = 'GPU'
    stRg.gpu_shading = True
    stRg.gpu_raster = True
    stRg.layer_gpu_min_frac = 0.0        # prove the machinery, not the
    #                                      routing
    scRg = demo_scene(stRg, with_texture=False)
    scRg.materials[1].opacity = 0.5
    scRg.materials[0].reflect_level = 0.2
    scRg.materials[2].reflect_level = 0.5
    R.render(scRg, stRg)                     # cold
    t0 = _time.perf_counter()
    gpuR = R.render(scRg, stRg)
    t_gR = _time.perf_counter() - t0
    dR = float(np.abs(gpuR - cpuR).max())
    flipR = int((np.abs(gpuR - cpuR).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {dR:.6f}')
    _p(out, f'    px off by >0.01: {flipR} of {w2 * h2}')
    _p(out, f'    CPU everything : {t_cR * 1000.0:7.1f} ms')
    _p(out, f'    GPU everything : {t_gR * 1000.0:7.1f} ms (warm, whole '
            'render() call -- opaque frame, glass layers AND their rays)')
    if dR < 6e-3:
        _p(out, '    the glass recurses on your driver -- the refusal '
                'the 25-second field frames named is lifted')
    else:
        _p(out, '    (the fragment diagnosis in the section above '
                'applies here too: re-run with ray tracing off to '
                'separate the layer draw from the sweeps)')

    # ---- BUMPY GLASS LAYERS: the field's last named material shape
    # ("'Material.008': a Bump pre-pass on a transparent layer") -- the
    # Water anatomy AS glass. Height pre-passes draw per rank over each
    # layer's own ids (sin-fract Noise chains CPU-evaluated over the
    # rank's virtual surface, exactly), and the CPU side now shades
    # rank by rank with per-rank gradient fields -- the port surfaced
    # and fixed a chunk-dependence in the OLD mixed-batch shading.
    _p(out)
    _p(out, '  BUMPY GLASS LAYERS (NOISE-into-BUMP glass + mirror, ray '
            'on, whole render):')
    from .tests.scenebuild import add_normal_mapped_ball as _add_nm

    def _skB(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def _bumpy_glass(sc):
        _add_nm(sc)
        gB = sc.materials[1].graph
        gB['nodes']['hnoise'] = {
            'id': 'hnoise', 'bl_idname': 'ShaderNodeTexNoise',
            'props': {'noise_dimensions': '3D'},
            'inputs': [_skB('Vector', 'VECTOR', [0, 0, 0]),
                       _skB('Scale', 'VALUE', 10.0),
                       _skB('Detail', 'VALUE', 2.0),
                       _skB('Roughness', 'VALUE', 0.5),
                       _skB('Distortion', 'VALUE', 0.0)],
            'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                        {'name': 'Color', 'type': 'RGBA'}]}
        gB['nodes']['bump'] = {
            'id': 'bump', 'bl_idname': 'ShaderNodeBump',
            'props': {'invert': False},
            'inputs': [_skB('Strength', 'VALUE', 0.8),
                       _skB('Distance', 'VALUE', 0.6),
                       _skB('Height', 'VALUE', 0.5, ['hnoise', 0]),
                       _skB('Normal', 'VECTOR', [0, 0, 0])],
            'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
        for s in gB['nodes']['hal']['inputs']:
            if s['name'] == 'Normal':
                s['link'] = ['bump', 0]
            if s['name'] == 'Diffuse Color':
                s['link'] = None
                s['default'] = [0.3, 0.6, 0.9, 1.0]
            if s['name'] == 'Opacity':
                s['default'] = 0.5
            if s['name'] == 'IOR':
                s['default'] = 1.33
        sc.materials[1].has_alpha = True
        sc.materials[2].reflect_level = 0.5

    stB2 = RenderSettings()
    stB2.resolution_x, stB2.resolution_y = w2, h2
    stB2.aa_samples = 1
    stB2.shadows = True
    stB2.raytrace = True
    stB2.ray_depth = 1
    scB2 = demo_scene(stB2, with_texture=False)
    _bumpy_glass(scB2)
    t0 = _time.perf_counter()
    cpuB2 = R.render(scB2, stB2)
    t_cB2 = _time.perf_counter() - t0
    stB2g = RenderSettings()
    stB2g.resolution_x, stB2g.resolution_y = w2, h2
    stB2g.aa_samples = 1
    stB2g.shadows = True
    stB2g.raytrace = True
    stB2g.ray_depth = 1
    stB2g.render_device = 'GPU'
    stB2g.gpu_shading = True
    stB2g.gpu_raster = True
    stB2g.layer_gpu_min_frac = 0.0       # prove the machinery, not the
    #                                      routing
    scB2g = demo_scene(stB2g, with_texture=False)
    _bumpy_glass(scB2g)
    R.render(scB2g, stB2g)                   # cold
    t0 = _time.perf_counter()
    gpuB2 = R.render(scB2g, stB2g)
    t_gB2 = _time.perf_counter() - t0
    dB2 = float(np.abs(gpuB2 - cpuB2).max())
    flipB2 = int((np.abs(gpuB2 - cpuB2).max(axis=2) > 1e-2).sum())
    _p(out, f'    max difference : {dB2:.6f}')
    _p(out, f'    px off by >0.01: {flipB2} of {w2 * h2}')
    _p(out, f'    CPU everything : {t_cB2 * 1000.0:7.1f} ms')
    _p(out, f'    GPU everything : {t_gB2 * 1000.0:7.1f} ms (warm, '
            'whole render() call)')
    if dB2 < 6e-3:
        _p(out, '    wavy glass shades, bends and recurses on your '
                'driver -- the LAST named refusal of the field frames '
                'is lifted')

    # ---- SCISSORED LAYERS: the per-layer passes and readbacks are
    # clipped to each layer's own bounding box (1.25.61) -- pure
    # transport, so the SAME frame with the scissor off must match BIT
    # FOR BIT. This is the A/B that stands the newer region-read driver
    # path next to the proven full-frame texture read, on your driver.
    _p(out)
    _p(out, '  SCISSORED LAYERS (same bumpy-glass frame, scissor off '
            'vs on):')
    _sc_saved = getattr(_GSH, 'REGION_DRAWS', True)
    try:
        _GSH.REGION_DRAWS = False
        t0 = _time.perf_counter()
        gpuB2f = R.render(scB2g, stB2g)
        t_fB2 = _time.perf_counter() - t0
    finally:
        _GSH.REGION_DRAWS = _sc_saved
    dSc = float(np.abs(gpuB2 - gpuB2f).max())
    _p(out, f'    scissored vs full-frame: max {dSc:.6f} '
            f'({"BIT-IDENTICAL -- regions are pure transport" if dSc == 0.0 else "NOT identical -- the region path disagrees; turn the Debug toggle off and paste this"})')
    _p(out, f'    full-frame render      : {t_fB2 * 1000.0:7.1f} ms '
            f'(scissored {t_gB2 * 1000.0:.1f} ms)')

    # ---- HYBRID LAYER ROUTING: the driver pays full-frame fixed costs
    # per depth layer while the CPU pays per fragment, so sparse deep
    # layers route to the proven per-rank CPU path (whole layers only,
    # never split). The bumpy frame alone cannot split -- one glass
    # ball is two layers of EXACTLY equal size (front and back faces,
    # 8542 each on the first field run) and no threshold fits between
    # equals -- so this frame makes the MIRROR see-through too: two
    # transparent materials with different footprints give layers of
    # different sizes, and a threshold between the two middle DISTINCT
    # sizes forces a real mix, proven against both pure runs.
    _p(out)
    _p(out, '  HYBRID LAYER ROUTING (bumpy glass + glass mirror, '
            'forced GPU/CPU mix):')

    def _hybrid_scene(sth):
        sch = demo_scene(sth, with_texture=False)
        _bumpy_glass(sch)
        sch.materials[2].opacity = 0.6      # the mirror is glass now
        return sch

    stHc = RenderSettings()
    stHc.resolution_x, stHc.resolution_y = w2, h2
    stHc.aa_samples = 1
    stHc.shadows = True
    stHc.raytrace = True
    stHc.ray_depth = 1
    t0 = _time.perf_counter()
    cpuH = R.render(_hybrid_scene(stHc), stHc)
    t_cH = _time.perf_counter() - t0

    def _hybrid_gpu_settings(frac):
        sth = RenderSettings()
        sth.resolution_x, sth.resolution_y = w2, h2
        sth.aa_samples = 1
        sth.shadows = True
        sth.raytrace = True
        sth.ray_depth = 1
        sth.render_device = 'GPU'
        sth.gpu_shading = True
        sth.gpu_raster = True
        sth.layer_gpu_min_frac = frac
        return sth

    stH0 = _hybrid_gpu_settings(0.0)
    t0 = _time.perf_counter()
    gpuH = R.render(_hybrid_scene(stH0), stH0)
    t_gH = _time.perf_counter() - t0
    countsH = list((getattr(R, 'LAST_ROUTING', {}) or {})
                   .get('counts', []))
    valsH = sorted({int(c) for c in countsH if c > 0}, reverse=True)
    if len(valsH) < 2:
        _p(out, '    (frame cannot split -- every layer is the same '
                f'size: {countsH})')
    else:
        # a threshold BETWEEN the two largest DISTINCT layer sizes:
        # the biggest layer(s) stay on the driver, the rest route
        thrH = (valsH[0] + valsH[1]) / 2.0
        stH = _hybrid_gpu_settings(thrH / float(w2 * h2))
        t0 = _time.perf_counter()
        hybB = R.render(_hybrid_scene(stH), stH)
        t_hB = _time.perf_counter() - t0
        rtH = dict(getattr(R, 'LAST_ROUTING', {}) or {})
        _p(out, f'    routed         : GPU {rtH.get("gpu_ranks", 0)} '
                f'layers ({rtH.get("gpu_frags", 0)} frags), CPU '
                f'{rtH.get("cpu_ranks", 0)} layers '
                f'({rtH.get("cpu_frags", 0)} frags), threshold '
                f'{rtH.get("thresh", 0)} of per-layer {countsH}')
        if not rtH.get('gpu_ranks') or not rtH.get('cpu_ranks') \
                or 'refused' in rtH:
            _p(out, '    NOT A MIX -- the routing test is vacuous on '
                    'this frame'
                    + (f" (driver refused: {rtH['refused']})"
                       if 'refused' in rtH else
                       '; the numbers above name why'))
        dGC = float(np.abs(gpuH - cpuH).max())
        flGC = int((np.abs(gpuH - cpuH).max(axis=2) > 1e-2).sum())
        _p(out, f'    all-GPU vs all-CPU: max {dGC:.6f}, px off by '
                f'>0.01: {flGC} of {w2 * h2}')
        dHg = float(np.abs(hybB - gpuH).max())
        dHc = float(np.abs(hybB - cpuH).max())
        flHg = int((np.abs(hybB - gpuH).max(axis=2) > 1e-2).sum())
        flHc = int((np.abs(hybB - cpuH).max(axis=2) > 1e-2).sum())
        _p(out, f'    vs all-GPU     : max {dHg:.6f}, px off by >0.01: '
                f'{flHg} of {w2 * h2}')
        _p(out, f'    vs all-CPU     : max {dHc:.6f}, px off by >0.01: '
                f'{flHc} of {w2 * h2}')
        _p(out, f'    hybrid render  : {t_hB * 1000.0:7.1f} ms (warm '
                f'GPU {t_gH * 1000.0:.1f}, CPU {t_cH * 1000.0:.1f})')
        ltH = dict(getattr(_GSH, 'LAST_LAYER_TIMINGS', {}) or {})
        if ltH:
            sumH = (ltH.get('plan_ms', 0.0) + ltH.get('compile_ms', 0.0)
                    + ltH.get('upload_ms', 0.0) + ltH.get('draw_ms', 0.0)
                    + ltH.get('read_ms', 0.0) + ltH.get('sweep_ms', 0.0))
            _p(out, f'    split accounts : buckets {sumH:.1f} ms + other '
                    f"{ltH.get('other_ms', 0.0):.1f} ms = total "
                    f"{ltH.get('total_ms', 0.0):.1f} ms "
                    f"({'sound' if sumH <= ltH.get('total_ms', 0.0) + 1.0 else 'OVERLAPPING -- a bucket is double-counted'})")
        if dHg < 6e-3 and dHc < 6e-3 and rtH.get('gpu_ranks') \
                and rtH.get('cpu_ranks') and 'refused' not in rtH:
            _p(out, '    the mixed frame matches both pure paths: '
                    'routing is invisible in the picture')
        if flGC > 0 or dGC >= 6e-3:
            # the driver and the CPU disagree on THIS scene -- the one
            # with a material that is reflective AND see-through as
            # layers -- while the headless mirror of the same passes
            # matches the CPU to 1e-5. That acquits the maths and
            # indicts something only the driver runs. Corner it per
            # FRAGMENT, on the driver, with the histograms that name
            # the mechanism class: one-material clustering points at
            # its pass, one-rank at the merge, alpha-only at the blend,
            # backface clustering at facing, and a nonzero
            # driver-vs-itself count at nondeterminism.
            _p(out, '    DIAGNOSIS (fragment-level, on the driver -- '
                    'the headless mirror matches the CPU, so these '
                    'numbers are the driver\'s own):')
            try:
                stD = stHc
                scD = _hybrid_scene(stD)
                R._build_shadows(scD, stD, scD.mesh)
                viewD, _pD, vpD, eyeD = R.camera_matrices(scD.camera,
                                                          w2, h2)
                opqD, trD = R._split_by_alpha(scD, scD.mesh, stD)
                gD = CR.GBuffer(w2, h2)
                CR.rasterize(scD.mesh.verts, scD.mesh.tris, vpD, w2, h2,
                             subset=opqD, gbuf=gD,
                             depth_bits=stD.depth_precision)
                fD = CR.FragmentList()
                CR.rasterize(scD.mesh.verts, scD.mesh.tris, vpD, w2, h2,
                             cull='NONE', subset=trD, gbuf=gD, frags=fD,
                             depth_write=False,
                             depth_bits=stD.depth_precision)
                texD = R.prepare_textures(scD, stD)
                jobD = R.ShadeJob(scD, stD, texD,
                                  R._cached_bvh(scD, scD.mesh), viewD,
                                  eyeD, w2, h2)
                pxD, pyD, triD, depD, barD, froD = fD.finish()
                kD = depD <= CR.abuf_depth_limit(gD.depth[pyD, pxD])
                pxD, pyD, triD, depD, barD, froD = (
                    a[kD] for a in (pxD, pyD, triD, depD, barD, froD))
                centD = scD.mesh.verts[scD.mesh.tris].mean(axis=1)
                vzD = np.abs((centD - jobD.eye[None, :])
                             @ jobD.view[:3, :3].T)[:, 2]
                keyD = vzD[triD].astype(np.float32)
                pixD = pyD.astype(np.int64) * gD.width + pxD
                oD = np.lexsort((keyD, pixD))
                pixD, pxD, pyD, triD, barD, froD = (
                    a[oD] for a in (pixD, pxD, pyD, triD, barD, froD))
                grD = np.zeros(pixD.size, np.int64)
                ngD = np.nonzero(pixD[1:] != pixD[:-1])[0] + 1
                grD[ngD] = ngD
                np.maximum.accumulate(grD, out=grD)
                rkD = np.arange(pixD.size, dtype=np.int64) - grD
                cpuD = R._shade_fragments_cpu(jobD, triD, barD, pxD,
                                              pyD, froD, rkD, stD)
                _GSH._PLAN_CACHE.clear()
                drvD, whyD = _GSH.shade_fragments_frame(
                    jobD, gD, triD, barD, pxD, pyD, rkD)
                if drvD is None:
                    _p(out, f'      the driver re-shade refused: {whyD}')
                else:
                    dD = np.abs(drvD - cpuD).max(axis=1)
                    badD = dD > 1e-2
                    _p(out, f'      fragment flips >0.01: '
                            f'{int(badD.sum())} of {triD.size}   '
                            f'max {float(dD.max()):.6f}')
                    if badD.any():
                        miD = scD.mesh.mat_index[triD] \
                            if scD.mesh.mat_index is not None else \
                            np.zeros(triD.size, np.int32)
                        daD = np.abs(drvD[:, 3] - cpuD[:, 3]) > 1e-2
                        drD = np.abs(drvD[:, :3]
                                     - cpuD[:, :3]).max(axis=1) > 1e-2
                        _p(out, f'      channels: rgb-only '
                                f'{int((badD & drD & ~daD).sum())}, '
                                f'alpha-only '
                                f'{int((badD & daD & ~drD).sum())}, '
                                f'both {int((badD & daD & drD).sum())}')
                        _p(out, '      by material: ' + str(
                            {int(m): int((badD & (miD == m)).sum())
                             for m in np.unique(miD[badD])})
                            + '  (1=bumpy glass, 2=glass mirror)')
                        _p(out, '      by rank: ' + str(
                            {int(r): int((badD & (rkD == r)).sum())
                             for r in np.unique(rkD[badD])}))
                        _p(out, f'      by facing: front '
                                f'{int((badD & froD).sum())}, back '
                                f'{int((badD & ~froD).sum())}')
                        drv2D, _w2D = _GSH.shade_fragments_frame(
                            jobD, gD, triD, barD, pxD, pyD, rkD)
                        if drv2D is not None:
                            ndD = int((np.abs(drv2D - drvD).max(axis=1)
                                       > 1e-4).sum())
                            _p(out, f'      driver vs itself: {ndD} '
                                    'fragments differ (nonzero = '
                                    'nondeterministic)')
                        for kf in np.argsort(dD)[::-1][:3]:
                            kf = int(kf)
                            _p(out, f'      flip: rank {int(rkD[kf])} '
                                    f'mat {int(miD[kf])} front '
                                    f'{bool(froD[kf])}  cpu '
                                    f'{np.round(cpuD[kf], 4).tolist()}'
                                    f'  drv '
                                    f'{np.round(drvD[kf], 4).tolist()}')
                        # ---- suspect A/B: re-shade the SAME fragments
                        # with one suspect swapped out at a time. The
                        # variant that ZEROES the count names the
                        # mechanism; a variant that leaves it untouched
                        # is exonerated with the same number.
                        _p(out, '      SUSPECT A/B (the variant that '
                                'zeroes the count names the mechanism):')

                        def _flipsD(col, ref):
                            return int((np.abs(col - ref).max(axis=1)
                                        > 1e-2).sum())

                        # the front-end intersector under the driver's
                        # own draws: the layer sweeps trace through the
                        # compute closest-hit kernel, the CPU reference
                        # through bvh.intersect -- grazing reflection
                        # rays can tie between the two. Slow (the
                        # front-end interprets every ray) but decisive.
                        from .gpu import rtrace as _RTd
                        _saved_if = _RTd.intersect_frame
                        try:
                            _RTd.intersect_frame = \
                                lambda bvh, org, dirs, tmax=1e30: (
                                    _RTd.simulate_intersect(
                                        bvh, org, dirs, tmax), None)
                            _GSH._PLAN_CACHE.clear()
                            colB, _wB = _GSH.shade_fragments_frame(
                                jobD, gD, triD, barD, pxD, pyD, rkD)
                        finally:
                            _RTd.intersect_frame = _saved_if
                        _p(out, '        front-end intersector, driver '
                                'draws: '
                                + (f'{_flipsD(colB, cpuD)} flips'
                                   if colB is not None
                                   else f'refused: {_wB}'))
                        # the scissor off: regions proved bit-identical
                        # on the bumpy frame, but THIS frame is the one
                        # that disagrees -- prove it here too
                        _saved_rg = getattr(_GSH, 'REGION_DRAWS', True)
                        try:
                            _GSH.REGION_DRAWS = False
                            _GSH._PLAN_CACHE.clear()
                            colC, _wC = _GSH.shade_fragments_frame(
                                jobD, gD, triD, barD, pxD, pyD, rkD)
                        finally:
                            _GSH.REGION_DRAWS = _saved_rg
                        _p(out, '        scissor off                  '
                                '  : '
                                + (f'{_flipsD(colC, cpuD)} flips'
                                   if colC is not None
                                   else f'refused: {_wC}'))
                        # the mirror stops reflecting, against its OWN
                        # CPU reference: zero here means every flip
                        # lives in the reflection term
                        _mat2D = scD.materials[2]
                        _saved_rf = _mat2D.reflect_level
                        try:
                            _mat2D.reflect_level = 0.0
                            cpuD2 = R._shade_fragments_cpu(
                                jobD, triD, barD, pxD, pyD, froD, rkD,
                                stD)
                            _GSH._PLAN_CACHE.clear()
                            colE, _wE = _GSH.shade_fragments_frame(
                                jobD, gD, triD, barD, pxD, pyD, rkD)
                        finally:
                            _mat2D.reflect_level = _saved_rf
                        _p(out, '        mirror reflect 0 (own ref)   '
                                '  : '
                                + (f'{_flipsD(colE, cpuD2)} flips'
                                   if colE is not None
                                   else f'refused: {_wE}'))
                        # ---- the intersector CROSS-CHECK: kernel-code
                        # edits (precise, texelFetch) left the flips
                        # bit-identical, so the divergence is not in
                        # the kernel's arithmetic -- it is in what the
                        # kernel is FED or how its answer travels.
                        # Capture the sweeps' OWN rays, then ask all
                        # three intersectors the same question:
                        #   cached  = intersect_frame (the render path:
                        #             cached tree uploads)
                        #   fresh   = intersect_on_device (same kernel,
                        #             tree packed+uploaded fresh)
                        #   frontend= simulate_intersect (no driver)
                        # cached!=fresh with fresh==frontend convicts
                        # the CACHED UPLOAD; cached==fresh!=frontend is
                        # a real kernel divergence; all equal means the
                        # intersector was never the mechanism and the
                        # earlier zero was a side effect -- each verdict
                        # is one line.
                        try:
                            recR = []
                            _orig_if = _RTd.intersect_frame

                            def _rec_if(bvh, org, dirs, tmax=1e30):
                                recR.append((org.copy(), dirs.copy()))
                                return _orig_if(bvh, org, dirs, tmax)

                            _RTd.intersect_frame = _rec_if
                            try:
                                _GSH._PLAN_CACHE.clear()
                                _GSH.shade_fragments_frame(
                                    jobD, gD, triD, barD, pxD, pyD, rkD)
                            finally:
                                _RTd.intersect_frame = _orig_if
                            n_rays = 0
                            cf = cfe = ffe = 0
                            tie_a = tie_b = tie_c = 0
                            bad_ex = None
                            bad_kf = None
                            for orgR, dirR in recR:
                                aR, _wa = _orig_if(jobD.bvh, orgR, dirR)
                                tie_a += int(getattr(
                                    _RTd, 'LAST_TIE_ROUTED', 0))
                                bR, _wb = _RTd.intersect_on_device(
                                    jobD.bvh, orgR, dirR, 1e30)
                                tie_b += int(getattr(
                                    _RTd, 'LAST_TIE_ROUTED', 0))
                                cR = _RTd.simulate_intersect(
                                    jobD.bvh, orgR, dirR, 1e30)
                                tie_c += int(getattr(
                                    _RTd, 'LAST_TIE_ROUTED', 0))
                                if aR is None or bR is None:
                                    _p(out, '      cross-check refused: '
                                            f'{_wa or _wb}')
                                    break
                                n_rays += orgR.shape[0]
                                m_cf = aR[0] != bR[0]
                                m_kf = aR[0] != cR[0]
                                cf += int(m_cf.sum())
                                cfe += int(m_kf.sum())
                                ffe += int((bR[0] != cR[0]).sum())
                                if bad_ex is None and m_cf.any():
                                    k2 = int(np.nonzero(m_cf)[0][0])
                                    bad_ex = (
                                        orgR[k2].tolist(),
                                        dirR[k2].tolist(),
                                        int(aR[0][k2]), int(bR[0][k2]),
                                        int(cR[0][k2]))
                                if bad_kf is None and m_kf.any():
                                    k3 = int(np.nonzero(m_kf)[0][0])
                                    bad_kf = (
                                        orgR[k3].tolist(),
                                        dirR[k3].tolist(),
                                        int(aR[0][k3]),
                                        float(aR[1][k3]),
                                        int(cR[0][k3]),
                                        float(cR[1][k3]))
                            _p(out, '      intersector cross-check on '
                                    f'the sweeps\' own rays ({n_rays} '
                                    'rays):')
                            _p(out, f'        cached vs fresh   : {cf} '
                                    'mismatched hit ids')
                            _p(out, '        cached vs frontend: '
                                    f'{cfe} mismatched hit ids')
                            _p(out, '        fresh  vs frontend: '
                                    f'{ffe} mismatched hit ids')
                            _p(out, '        noise-window ties re-'
                                    'resolved on the CPU: cached '
                                    f'{tie_a}, fresh {tie_b}, frontend '
                                    f'{tie_c} (nonzero NAMES coincident '
                                    'contact geometry -- here the box '
                                    'bottom resting on the floor; all '
                                    'three intersectors take the '
                                    "reference's own winner for "
                                    'exactly those rays)')
                            if bad_ex is not None:
                                _p(out, '        first cached/fresh '
                                        'split: org '
                                        f'{np.round(bad_ex[0], 4).tolist()}'
                                        f' dir '
                                        f'{np.round(bad_ex[1], 4).tolist()}'
                                        f' -> cached {bad_ex[2]}, fresh '
                                        f'{bad_ex[3]}, frontend '
                                        f'{bad_ex[4]}')
                            if bad_kf is not None:
                                _p(out, '        first kernel/frontend '
                                        'split: org '
                                        f'{np.round(bad_kf[0], 4).tolist()}'
                                        f' dir '
                                        f'{np.round(bad_kf[1], 4).tolist()}'
                                        f' -> kernel id {bad_kf[2]} t '
                                        f'{bad_kf[3]:.7f}, frontend id '
                                        f'{bad_kf[4]} t {bad_kf[5]:.7f}')
                        except Exception as _excX:              # noqa: BLE001
                            _p(out, f'      (cross-check failed: {_excX})')
                    else:
                        _p(out, '      the per-fragment shading MATCHES '
                                '-- the whole-frame difference is in '
                                'the opaque base or the composite, not '
                                'the layers')
            except Exception as _excH:                          # noqa: BLE001
                _p(out, f'      (diagnosis failed: {_excH})')


def _refracted_diagnosis(out, scB, stB, cpuA, gpuA, w2, h2):
    """Corner the refracted frame's disagreement on the driver itself.

    Headless analysis already ruled the BANDS sky OUT for this scene (a
    band flip here moves a pixel ~0.01, the report said 0.19), which
    leaves suspects that only the driver can separate: the shadow taps
    inside secondary (hit) shading, the bump bend's first hardware run,
    and the refraction sweep itself. Two instruments: classify WHERE the
    bad pixels live (surface, ray membership, what their rays hit), then
    re-render the same frame with one suspect disabled at a time --
    whichever variant zeroes the count names the mechanism.
    """
    import numpy as np

    from .core import raster as CR
    from .core import render as R
    from .core.settings import RenderSettings
    from .gpu import shade as _GSH

    _p(out, '    DIAGNOSIS (the numbers the next round needs):')
    try:
        badA = np.abs(gpuA - cpuA).max(axis=2) > 1e-2
        viewD, _pd, vpD, eyeD = R.camera_matrices(scB.camera, w2, h2)
        gD = CR.GBuffer(w2, h2)
        CR.rasterize(scB.mesh.verts, scB.mesh.tris, vpD, w2, h2, gbuf=gD)
        texD = R.prepare_textures(scB, stB)
        bvhD = R._cached_bvh(scB, scB.mesh)
        jobD = R.ShadeJob(scB, stB, texD, bvhD, viewD, eyeD, w2, h2)
        miD = np.where(gD.tri >= 0,
                       scB.mesh.mat_index[np.clip(gD.tri, 0, None)], -1)
        names = {0: 'floor', 1: 'glass', 2: 'mirror', -1: 'miss (world)'}

        def nm(m):
            return names.get(int(m), f'mat{int(m)}')

        hist = {nm(m): int((badA & (miD == m)).sum())
                for m in np.unique(miD[badA])}
        _p(out, f'      bad px by surface: {hist}')

        # ray membership, and what the bad pixels' own rays hit -- the
        # rays and the trace are CPU-side and proven, so this is exact
        pD, whyD, aD = _GSH.plan_frame(jobD, gD)
        rpD = (aD or {}).get('__reflect') if pD is not None else None
        if rpD is None:
            _p(out, f'      (could not re-plan to classify rays: {whyD})')
        else:
            for label, mk in (('reflection', _GSH._reflection_rays),
                              ('refraction', _GSH._refraction_rays)):
                ry, rx, ro, rd = mk(jobD, gD, rpD)
                if ry.size == 0:
                    continue
                hidD, _tD, _uD, _vD = bvhD.intersect(
                    ro, rd, np.full(ry.size, 1e30, np.float32))
                on = badA[ry, rx]
                hm = np.where(hidD >= 0,
                              scB.mesh.mat_index[np.clip(hidD, 0, None)],
                              -1)
                hh = {nm(m): int((on & (hm == m)).sum())
                      for m in np.unique(hm[on])} if on.any() else {}
                _p(out, f'      bad px that are {label} px: '
                        f'{int(on.sum())} of {ry.size}   their hits: {hh}')

        # the worst three, in numbers: channel structure tells shadow
        # flips (big, all channels, light-coloured) from tint or band
        # errors (small, colour-shaped)
        flat = np.abs(gpuA - cpuA).max(axis=2)
        for k in np.argsort(flat.ravel())[::-1][:3]:
            yy, xx = divmod(int(k), w2)
            _p(out, f'      worst: ({yy},{xx}) on {nm(miD[yy, xx])}  '
                    f'cpu {np.round(cpuA[yy, xx][:3], 4)}  '
                    f'gpu {np.round(gpuA[yy, xx][:3], 4)}')

        # ---- the A/B quartet: one suspect off per re-render. Counts
        # only; the variant that reads 0 is the conviction
        stC = RenderSettings()
        for f in ('resolution_x', 'resolution_y', 'aa_samples', 'shadows',
                  'raytrace', 'ray_depth', 'transparency'):
            setattr(stC, f, getattr(stB, f))

        bump_node = scB.materials[1].graph['nodes'].get('bump')
        bump_sock = None
        if bump_node is not None:
            bump_sock = next((s for s in bump_node['inputs']
                              if s['name'] == 'Strength'), None)

        def _ab(tag, mutate, restore):
            try:
                mutate()
                _GSH._PLAN_CACHE.clear()
                ref = R.render(scB, stC)
                got = R.render(scB, stB)
                n = int((np.abs(got - ref).max(axis=2) > 1e-2).sum())
                mx = float(np.abs(got - ref).max())
                _p(out, f'      {tag:<26s}: {n} px off  (max {mx:.6f})')
            except Exception as exc:                            # noqa: BLE001
                _p(out, f'      {tag:<26s}: variant failed: {exc}')
            finally:
                restore()
                _GSH._PLAN_CACHE.clear()

        def _set_shadows(v):
            stB.shadows = v
            stC.shadows = v

        def _set_strength(v):
            scB.world.strength = v

        def _set_bump(v):
            if bump_sock is not None:
                bump_sock['default'] = v

        def _set_refract(v):
            stB.ray_refraction = v
            stC.ray_refraction = v

        old_strength = float(getattr(scB.world, 'strength', 1.0))
        old_bump = None if bump_sock is None else bump_sock['default']
        _ab('shadows OFF', lambda: _set_shadows(False),
            lambda: _set_shadows(True))
        _ab('sky strength 0 (no env)', lambda: _set_strength(0.0),
            lambda: _set_strength(old_strength))
        if bump_sock is not None:
            _ab('bump strength 0 (flat)', lambda: _set_bump(0.0),
                lambda: _set_bump(old_bump))
        _ab('ray refraction OFF', lambda: _set_refract(False),
            lambda: _set_refract(True))
    except Exception as _exc:                                   # noqa: BLE001
        _p(out, f'      diagnosis itself failed: {_exc}')


def _gpu_compute_probe_retired(out):
    """The round-19 API probe, kept for reference; superseded above."""
    _p(out)
    _p(out, LINE)
    _p(out, 'GPU COMPUTE CAPABILITY  (for the compute rasteriser study)')
    _p(out, LINE)
    from .gpu import device
    ok, why = device.probe()
    if not ok:
        _p(out, f'  skipped: {why}')
        return
    import gpu as _gpu

    # ---- what the module surface offers
    types_attrs = sorted(a for a in dir(_gpu.types)
                         if any(k in a.lower() for k in
                                ('storage', 'buf', 'compute')))
    top_attrs = sorted(a for a in dir(_gpu)
                       if 'compute' in a.lower())
    _p(out, f'  gpu module      : compute-ish toplevel {top_attrs or "none"}')
    _p(out, f'  gpu.types       : {types_attrs or "none relevant"}')
    try:
        info = _gpu.types.GPUShaderCreateInfo()
        info_attrs = sorted(a for a in dir(info)
                            if any(k in a.lower() for k in
                                   ('compute', 'local', 'storage', 'image',
                                    'ssbo')))
        _p(out, f'  CreateInfo      : {info_attrs or "none relevant"}')
    except Exception as exc:                                    # noqa: BLE001
        _p(out, f'  CreateInfo      : failed to instantiate: {exc}')
        return

    # ---- the most likely API shape, attempted step by step
    steps = []

    def step(name, fn):
        try:
            r = fn()
            steps.append((name, 'ok', r))
            return r
        except Exception as exc:                                # noqa: BLE001
            steps.append((name, 'FAIL', f'{type(exc).__name__}: {exc}'))
            return None

    src = '''
void main()
{
    ivec2 xy = ivec2(gl_GlobalInvocationID.xy);
    imageStore(hal_out, xy, vec4(float(xy.x), float(xy.y), 7.0, 1.0));
}
'''
    built = {}

    def make_info():
        ci = _gpu.types.GPUShaderCreateInfo()
        ci.local_group_size(8, 8)
        ci.image(0, 'RGBA32F', 'FLOAT_2D', 'hal_out',
                 qualifiers={'WRITE'})
        ci.compute_source(src)
        built['info'] = ci
        return 'local_group_size(8,8) + image(WRITE) + compute_source'
    step('CreateInfo compute setup', make_info)
    if built.get('info') is not None:
        def compile_it():
            built['shader'] = _gpu.shader.create_from_info(built['info'])
            return 'compiled'
        step('compile compute shader', compile_it)
    if built.get('shader') is not None:
        def dispatch_it():
            tex = _gpu.types.GPUTexture((16, 16), format='RGBA32F')
            sh = built['shader']
            sh.bind()
            sh.image('hal_out', tex)
            _gpu.compute.dispatch(sh, 2, 2, 1)
            buf = tex.read()
            import numpy as np
            arr = np.array(buf.to_list() if hasattr(buf, 'to_list')
                           else buf, dtype=np.float32).reshape(16, 16, 4)
            probe = arr[3, 5]
            good = abs(probe[0] - 5.0) < 0.5 and abs(probe[1] - 3.0) < 0.5 \
                and abs(probe[2] - 7.0) < 0.5
            return (f'dispatched and read back; pixel(5,3) = '
                    f'{probe[:3].tolist()} -> '
                    f'{"CORRECT" if good else "unexpected"}')
        step('dispatch + readback', dispatch_it)

    for name, status, detail in steps:
        _p(out, f'  {name:26s}: {status}  {detail}')
    if any(s[1] == 'FAIL' for s in steps):
        _p(out, '  a FAIL above is still an answer: the error text names '
                'the real API,')
        _p(out, '  and the next build reads it')
    else:
        _p(out, '  everything a compute rasteriser needs is present; the '
                'port can start')


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
                    ('gpu deferred shading', lambda: gpu_deferred(out)),
                    ('gpu compute capability', lambda: gpu_compute(out)),
                    ('viewport', lambda: viewport_device(out)),
                    ('feature matrix', lambda: feature_matrix(out)),
                    ('frame breakdown', lambda: frame_breakdown(out))]
        if self.include_scaling:
            sections.append(('cpu scaling', lambda: cpu_scaling(out, self.heavy)))
        failed = 0
        # the measurements own the driver for the duration: a live rendered
        # viewport keeps redrawing while this operator runs, and its GPU
        # frames would interleave with the timings -- it renders on the
        # CPU (and says so) until the report is done
        from .gpu.marshal import PIPELINE
        PIPELINE.acquire()
        try:
            for name, fn in sections:
                try:
                    fn()
                except Exception as exc:                        # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    failed += 1
                    _p(out)
                    _p(out, f'  [{name} failed: {type(exc).__name__}: {exc}]')
                    _p(out, '  (the remaining sections still ran)')
        finally:
            PIPELINE.release()
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
