"""The RenderEngine: final renders, tile reporting and the viewport preview."""

import time

import bpy
import numpy as np

from . import export
from .core import post
from .core import render as core_render
from .core import stats as ST


class HalcyonRenderEngine(bpy.types.RenderEngine):
    bl_idname = 'HALCYON_RENDER'
    bl_label = "Halcyon"
    bl_use_preview = True
    bl_use_shading_nodes_custom = False
    bl_use_eevee_viewport = False
    bl_use_custom_freestyle = False
    bl_use_alembic_procedural = False
    # True so Blender binds a GPU context around render() -- the final F12
    # render runs on a render thread where gpu.* otherwise has no context
    # at all ("No active GPU context found") and every stage fell back to
    # the CPU. The self-test never saw it: operators run on the main
    # thread, which has one. This is the difference between the GPU port
    # working in a report and working on F12.
    #
    # …and holding that context froze the interface for the whole frame:
    # Blender cannot draw a window while the render thread owns the GPU,
    # so a 33-second frame read "Not Responding" for 33 seconds -- the
    # field named it twice. The default is now the viewport's own
    # architecture applied to F12: the render thread runs with NO context
    # and every GPU burst (compile, upload, draw, read back) is
    # marshalled to the main thread, which always has one, through
    # gpu/marshal.py. The interface breathes between bursts. The Debug
    # panel's "Hold GPU Context" restores the old behaviour (it can
    # start bursts a tick sooner), background renders keep it
    # automatically (no interface to freeze, and -b may give the main
    # loop no timers), and every marshalling failure falls back to the
    # CPU frame with the reason printed -- never a broken picture.
    bl_use_gpu_context = False

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except TypeError:
            super().__init__()
        self._scene = None
        self._scene_key = None
        self._draw_data = None
        self._last_hash = None
        self._vp = None

    def __del__(self):
        vp = getattr(self, '_vp', None)
        if vp is not None:
            vp.abort = True

    # ------------------------------------------------------------ final render
    def render(self, depsgraph):
        bscene = depsgraph.scene
        tw, th = self._target_size(bscene)
        preview = bool(getattr(self, 'is_preview', False))
        settings = _settings_from_scene(bscene, tw, th, preview)
        warnings = []

        # GPU context stewardship. `granted` is what Blender already did
        # for THIS render (it read the class attribute when the job
        # started); the class attribute is then re-synced from the
        # setting so the NEXT render follows it. Background renders
        # always hold: there is no interface to keep alive, and a
        # windowless main loop may never run a timer.
        background = bool(getattr(bpy.app, 'background', False))
        granted = bool(type(self).bl_use_gpu_context)
        hold = bool(getattr(settings, 'gpu_hold_context', False)) or \
            background
        type(self).bl_use_gpu_context = hold
        from .gpu import marshal as _marshal
        marshalled = (not granted
                      and str(getattr(settings, 'render_device',
                                      'CPU')).upper() == 'GPU'
                      and not background)
        if marshalled:
            _marshal.enable()
        try:
            self._render_body(depsgraph, bscene, tw, th, preview,
                              settings, warnings)
        finally:
            if marshalled:
                _marshal.disable()

    def _render_body(self, depsgraph, bscene, tw, th, preview, settings,
                     warnings):

        from .version import version_string
        print(f"[Halcyon] {version_string()} rendering "
              f"{tw}x{th}  pass={settings.debug_pass}"
              f"  wire={settings.wire_mode}")
        self.update_stats("Halcyon", "Exporting scene")
        ST.reset()
        ST.enable(bool(settings.show_stats))
        _apply_debug_prefs()
        export_started = time.time()
        try:
            with ST.track('export scene (Blender side)'):
                scene = export.export_scene(depsgraph, settings, warnings)
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            if not preview:
                self.report({'ERROR'}, f"Halcyon export failed: {exc}")
            return

        # the device plan needs the exported scene, so it runs after the export
        # rather than before it, where `scene` did not yet exist -- the
        # UnboundLocalError went straight into a bare except and the plan has
        # never once run
        try:
            from .gpu import capability as _cap
            dev, _stages, notes = _cap.plan(scene, settings)
            if str(settings.render_device).upper() == 'GPU':
                print(f"[Halcyon] device: {dev}")
                # a stage toggled off under the GPU device is worth one
                # plain line each -- the 10-second frame that looked broken
                # was exactly this, silently
                for name, label in (('gpu_raster', 'GPU Rasteriser'),
                                    ('gpu_shading', 'GPU Shading'),
                                    ('gpu_post', 'GPU Post Processing')):
                    if not getattr(settings, name, True):
                        print(f"[Halcyon]   {label} is OFF (Debug panel) -- "
                              "this stage runs on the CPU")
                for note in notes:
                    print(f"[Halcyon]   {note}")
        except Exception:                                       # noqa: BLE001
            pass

        def on_progress(frac, msg):
            if self.test_break():
                raise _Cancelled()
            self.update_progress(float(frac))
            self.update_stats("Halcyon", msg)

        image = None
        gpu_whole_frame = str(getattr(settings, 'render_device',
                                      'CPU')).upper() == 'GPU' and \
            bool(getattr(settings, 'gpu_shading', False))
        if settings.use_processes and not preview and \
                core_render.wanted_passes(settings):
            print("[Halcyon] extra render passes are on, so this frame renders "
                  "in-process: the worker pool sends back pixels, not buffers")
        elif settings.use_processes and not preview and gpu_whole_frame:
            # the pool splits the frame into bands, and a band shades on
            # the CPU -- the deferred pass is whole-frame by design. With
            # the GPU device on, pooling would silently throw the driver
            # away: the field's 33-second render was exactly this, with
            # Task Manager honestly reading 0% GPU. The faster machine
            # wins; the pool stays available by turning GPU Shading off.
            print("[Halcyon] worker pool skipped: GPU Shading renders "
                  "whole frames in-process (a pooled band would shade on "
                  "the CPU). Turn GPU Shading off to use the pool instead")
        elif settings.use_processes and not preview:
            from .core import parallel as _par
            n = int(settings.process_count) or _resolve_cpus()
            key = (id(depsgraph), bscene.frame_current, tw, th)
            try:
                with ST.track('worker pool'):
                    image, why = _par.render_parallel(scene, settings, n,
                                                      scene_key=key,
                                                      progress=on_progress)
            except _Cancelled:
                return
            except Exception as exc:                            # noqa: BLE001
                image, why = None, f'{type(exc).__name__}: {exc}'
            if image is None and why:
                print(f"[Halcyon] process pool unavailable ({why}); "
                      f"rendering in this process")

        try:
            if image is None:
                image = core_render.render(scene, settings, progress=on_progress)
        except _Cancelled:
            return
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            if not preview:
                self.report({'ERROR'}, f"Halcyon render failed: {exc}")
            return

        self.update_stats("Halcyon", "Post processing")
        try:
            with ST.track('post processing'):
                final = post.process(image, settings, frame=scene.frame,
                                     seed=settings.seed, target_size=(tw, th),
                                     allow_resize=False,
                                     depth=getattr(scene, 'last_depth', None),
                                     shaft_sources=getattr(scene, 'last_shafts',
                                                           None))
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            if not preview:
                self.report({'WARNING'}, f"Post chain failed: {exc}")
            final = post.fit_to(np.clip(image, 0.0, 1.0), (tw, th))

        with ST.track('deliver to Blender'):
            self._deliver(final, bscene,
                          getattr(scene, 'last_passes', None))

        # reporting during a preview render pushes UI work onto the preview
        # thread for a thumbnail nobody is reading
        if not preview:
            for w in set(warnings) | set(getattr(scene, 'unsupported', ()) or ()):
                self.report({'WARNING'}, str(w))
        elapsed = time.time() - export_started
        if settings.show_stats:
            ST.report(total=elapsed)
        # the one-line answer to "where did this frame go", printed for
        # EVERY render: the 33-second mystery frame had its breakdown
        # collected all along, gated behind a panel nobody had ticked
        tops = ST.top(3)
        if tops:
            line = ', '.join(f'{n} {t:.1f}s' for n, t in tops)
            tail = '' if settings.show_stats else \
                "  (the Debug panel's Timing Breakdown toggle prints " \
                'the full table)'
            print(f"[Halcyon] {elapsed:.1f}s -- top stages: {line}{tail}")
        self.update_stats("Halcyon", f"Done in {elapsed:.1f}s")
        self.update_progress(1.0)

    # ------------------------------------------------------------------ passes
    def update_render_passes(self, scene=None, renderlayer=None):
        """Tell Blender which extra passes this render will contain.

        Without this the render result holds one pass, Combined, and everything
        else the compositor offers reads as black. Halcyon was force-enabling
        Blender's own Passes panel while delivering exactly one pass, which is
        the worst of both: the controls were there and none of them did
        anything.
        """
        self.register_pass(scene, renderlayer, "Combined", 4, "RGBA", 'COLOR')
        st = _settings_from_scene(scene, 1, 1) if scene is not None else None
        if st is None:
            return
        for name, chans, chan_id, kind in PASS_SPEC:
            if name in core_render.wanted_passes(st):
                self.register_pass(scene, renderlayer, name, chans, chan_id, kind)

    def _deliver_passes(self, result, buffers, w, h):
        """Write the extra passes, each fitted to Blender's own buffer."""
        try:
            passes = result.layers[0].passes
        except (IndexError, AttributeError):
            return
        for name, buf in buffers.items():
            try:
                target = passes[name]
            except (KeyError, TypeError):
                continue          # not enabled on this view layer
            arr = np.asarray(buf, np.float32)
            if arr.shape[0] != h or arr.shape[1] != w:
                arr = post.fit_to(arr, (w, h))
            chans = int(getattr(target, 'channels', arr.shape[2]) or arr.shape[2])
            if arr.shape[2] < chans:
                arr = np.concatenate(
                    [arr, np.zeros(arr.shape[:2] + (chans - arr.shape[2],),
                                   np.float32)], axis=2)
            elif arr.shape[2] > chans:
                arr = arr[:, :, :chans]
            flat = np.ascontiguousarray(np.nan_to_num(
                arr, nan=0.0, posinf=1e10, neginf=0.0).reshape(-1))
            try:
                target.rect.foreach_set(flat)
            except Exception:                                   # noqa: BLE001
                try:
                    target.rect = flat.reshape(-1, chans).tolist()
                except Exception:                               # noqa: BLE001
                    pass

    def _target_size(self, bscene):
        """The exact buffer size Blender has allocated for this render.

        `size_x`/`size_y` are what the engine was actually asked for, and are the
        only correct answer during a preview render, where the scene's own
        resolution has nothing to do with the thumbnail being drawn.
        """
        w = int(getattr(self, 'size_x', 0) or 0)
        h = int(getattr(self, 'size_y', 0) or 0)
        if w > 0 and h > 0:
            return w, h
        r = bscene.render
        pct = max(r.resolution_percentage, 1) / 100.0
        return max(int(r.resolution_x * pct), 1), max(int(r.resolution_y * pct), 1)

    def _deliver(self, final, bscene, extra=None):
        """Hand the finished image back through the render result.

        The buffer size is dictated by Blender, never by the image: writing more
        floats than `rect` holds overruns a C buffer and takes the process down
        with it. Post can legitimately resize (Pixel Scale, pixel aspect), so the
        result is fitted to the allocated size before a single value is written.
        """
        w, h = self._target_size(bscene)
        final = np.asarray(final, np.float32)
        if final.ndim != 3 or final.shape[2] not in (3, 4):
            return
        if final.shape[2] == 3:
            final = np.concatenate(
                [final, np.ones(final.shape[:2] + (1,), np.float32)], axis=2)
        if final.shape[0] != h or final.shape[1] != w:
            final = post.fit_to(final, (w, h))
        final = np.nan_to_num(final, nan=0.0, posinf=1.0, neginf=0.0)

        result = self.begin_result(0, 0, w, h)
        try:
            if extra:
                self._deliver_passes(result, extra, w, h)
            layer = result.layers[0].passes["Combined"]
            # Both buffers are bottom-row-first: the rasteriser maps NDC y = -1
            # to row 0, and Blender's rect expects the bottom row first.
            flat = np.ascontiguousarray(final.reshape(-1, 4))
            expected = w * h * 4
            if flat.size != expected:                    # belt and braces
                flat = np.resize(flat, expected)
            try:
                layer.rect.foreach_set(flat)
            except Exception:                                   # noqa: BLE001
                layer.rect = flat.reshape(-1, 4).tolist()
        finally:
            self.end_result(result)

    # --------------------------------------------------------------- viewport
    #
    # The shape Blender's own engines use: view_update syncs data (main
    # thread, bpy access is legal), view_draw only DRAWS. The old path
    # rendered synchronously inside the draw callback -- the whole UI froze
    # for every frame, seconds at a time on a real scene, and any failure
    # left the viewport permanently blank. Rendering now happens on a
    # background thread against the bpy-free exported scene; every draw
    # blits the newest finished frame and kicks a fresh render only when
    # the camera, the region or the scene actually moved on.

    def view_update(self, context, depsgraph):
        vp = self._viewport()
        try:
            region = getattr(context, 'region', None)
            w = int(getattr(region, 'width', 0) or 0) or 640
            h = int(getattr(region, 'height', 0) or 0) or 480
            settings = _viewport_settings(depsgraph.scene, w, h)
            scene = export.export_scene(depsgraph, settings)
            vp.set_scene(scene, settings)
        except Exception:                                       # noqa: BLE001
            import traceback
            vp.complain('the viewport export failed',
                        traceback.format_exc())
        self.tag_redraw()

    def view_draw(self, context, depsgraph):
        import gpu

        vp = self._viewport()
        region = context.region
        w, h = max(int(region.width), 4), max(int(region.height), 4)

        if vp.scene is None:
            # Blender promises a view_update before the first view_draw, but
            # a failed export leaves nothing to render -- say so on screen
            # rather than showing an eternal void
            vp.draw_placeholder(depsgraph)
            return

        vp.want(_view_camera(context), w, h)
        vp.kick(self)

        if vp.frame is None:
            vp.draw_placeholder(depsgraph)
            return
        try:
            tex = vp.texture(gpu)
            self.bind_display_space_shader(depsgraph.scene)
            _draw_texture(tex, w, h)
            self.unbind_display_space_shader()
        except Exception:                                       # noqa: BLE001
            import traceback
            vp.complain('drawing the viewport frame failed',
                        traceback.format_exc())

    def _viewport(self):
        if getattr(self, '_vp', None) is None:
            from .preview import Viewport
            self._vp = Viewport()
        return self._vp


class _Cancelled(Exception):
    pass


def _apply_debug_prefs():
    """Push the developer preferences into the bpy-free core."""
    from .core import nodeeval
    try:
        p = bpy.context.preferences.addons[__package__].preferences
        nodeeval.STRICT = bool(p.debug_mode and p.strict_nodes)
    except Exception:                                           # noqa: BLE001
        nodeeval.STRICT = False


def _resolve_cpus():
    import os
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


SCALE_FACTOR = {'NONE': 1, '2X': 2, '3X': 3, '4X': 4}


def _settings_from_scene(bscene, target_w, target_h, preview=False):
    """Scene settings mapped onto the buffer Blender actually asked for.

    Pixel Scale is interpreted as an *internal* downscale: the output stays the
    size Blender wants, and the engine renders at 1/N of it and blows the result
    back up with nearest-neighbour. That is what makes the setting useful (a
    genuine chunky-pixel render filling a normal output) and it keeps the
    delivered buffer exactly the size Blender allocated.
    """
    hs = getattr(bscene, 'halcyon', None)
    if hs is None:
        from .core.settings import RenderSettings
        st = RenderSettings()
    else:
        st = hs.to_settings()

    # Blender's own Film > Transparent is where people look for this, so it
    # wins when it is on; the Halcyon toggle can also enable it independently.
    if getattr(bscene.render, 'film_transparent', False):
        st.film_transparent = True
    n = SCALE_FACTOR.get(str(st.output_scale), 1)
    st.resolution_x = max(int(target_w) // n, 1)
    st.resolution_y = max(int(target_h) // n, 1)
    # Blender applies its own pixel aspect at display time; applying it here too
    # would double the stretch and change the buffer size
    st.pixel_aspect_x = st.pixel_aspect_y = 1.0
    # Halcyon's own thread count wins; 0 means "ask Blender", and Blender's AUTO
    # means "ask the machine"
    if int(st.threads) <= 0 and bscene.render.threads_mode != 'AUTO':
        st.threads = int(bscene.render.threads)

    if preview:
        # Thumbnails are tiny, generated in bulk, and drawn on a background
        # thread. Anything that costs seconds here is felt as a hang.
        st.aa_mode = 'NONE'
        st.aa_samples = 1
        st.shadows = False
        st.raytrace = False
        st.ambient_occlusion = False
        st.glow = st.star_filter = st.lens_flare = False
        st.jpeg_artifacts = False
        st.crt = False
        st.composite = False
        st.output_scale = 'NONE'
        st.resolution_x = min(max(int(target_w), 1), 256)
        st.resolution_y = min(max(int(target_h), 1), 256)
    return st


def _viewport_settings(bscene, w, h):
    """Render settings for an interactive preview of the region.

    The look stays the F12 look -- dither, CRT, the whole post chain are the
    point of this engine -- but everything about WHERE the work runs is
    pinned to the one shape that is safe from a viewport worker thread: the
    CPU. The GPU stages need a GPU context, worker threads have none, and
    the old path running them inside the draw callback is the prime suspect
    for the viewport rendering nothing at all under Vulkan.
    """
    settings = _settings_from_scene(bscene, max(w, 4), max(h, 4))
    scale = max(int(settings.preview_scale), 1)
    settings.resolution_x = max(w // scale, 4)
    settings.resolution_y = max(h // scale, 4)
    settings.aa_mode = 'NONE'
    settings.aa_samples = 1
    settings.output_scale = 'NONE'
    settings.pixel_aspect_x = settings.pixel_aspect_y = 1.0
    settings.render_device = 'CPU'
    settings.gpu_shading = settings.gpu_post = False
    settings.gpu_raster = False
    settings.use_processes = False
    settings.show_stats = False
    settings.progressive = False
    return settings


def _view_camera(context):
    """The viewport's own camera as a Halcyon Camera, or None."""
    rv3d = context.region_data
    if rv3d is None:
        return None
    from .core.scene import Camera
    view_matrix = np.asarray(rv3d.view_matrix, np.float32)
    persp = bool(getattr(rv3d, 'is_perspective', True))
    space = getattr(context, 'space_data', None)
    return Camera(
        matrix_world=np.linalg.inv(view_matrix).astype(np.float32),
        projection=np.asarray(rv3d.window_matrix, np.float32),
        type='PERSP' if persp else 'ORTHO',
        clip_start=float(getattr(space, 'clip_start', 0.1) or 0.1),
        clip_end=float(getattr(space, 'clip_end', 1000.0) or 1000.0))


def _draw_texture(tex, w, h):
    """Blit a texture over the region.

    The template idiom, with two amendments: TRI_STRIP instead of the
    deprecated TRI_FAN `draw_texture_2d` still carries (the fan leaves in
    Blender 6.0, and its warning was flooding consoles once per redraw),
    and premultiplied blending so a Film > Transparent frame composites
    over the viewport instead of overwriting it with black.
    """
    import gpu
    from gpu_extras.batch import batch_for_shader
    shader = gpu.shader.from_builtin('IMAGE')
    batch = batch_for_shader(
        shader, 'TRI_STRIP',
        {'pos': ((0, 0), (w, 0), (0, h), (w, h)),
         'texCoord': ((0, 0), (1, 0), (0, 1), (1, 1))})
    gpu.state.blend_set('ALPHA_PREMULT')
    try:
        shader.uniform_sampler('image', tex)
        batch.draw(shader)
    finally:
        gpu.state.blend_set('NONE')


#: (name, channels, channel ids, type) exactly as Blender names them, so a
#: Halcyon Z pass drops into a comp built for Cycles without rewiring
PASS_SPEC = (
    ("Depth", 1, "Z", 'VALUE'),
    ("Normal", 3, "XYZ", 'VECTOR'),
    ("Position", 3, "XYZ", 'VECTOR'),
    ("UV", 3, "UVA", 'VECTOR'),
    ("IndexOB", 1, "X", 'VALUE'),
    ("IndexMA", 1, "X", 'VALUE'),
)


# ---------------------------------------------------------------- UI plumbing

# Blender marks its own engine-agnostic property panels by listing
# 'BLENDER_RENDER' in COMPAT_ENGINES. Discovering them beats maintaining a list
# by hand: the hand-written one was missing the material slot list, the UV map
# list, colour attributes, vertex groups and colour management, and it would
# have gone on rotting with every Blender release.
GENERIC_MARKER = 'BLENDER_RENDER'

# Panels for features this engine genuinely does not implement. Showing a
# control that silently does nothing is worse than not showing it.
EXCLUDED_PANELS = {
    'RENDER_PT_freestyle', 'RENDER_PT_freestyle_line_style',
    'VIEWLAYER_PT_freestyle', 'VIEWLAYER_PT_freestyle_lineset',
    'VIEWLAYER_PT_freestyle_style_modules',
    'VIEWLAYER_PT_freestyle_lineset_collection',
    'MATERIAL_PT_freestyle_line', 'RENDER_PT_gpencil',
    'RENDER_PT_simplify_greasepencil',
}

# Panels Blender does not mark as generic but that are safe and needed here.
FORCED_PANELS = {
    'MATERIAL_PT_context_material', 'MATERIAL_PT_viewport',
    'DATA_PT_context_mesh', 'DATA_PT_uv_texture', 'DATA_PT_vertex_colors',
    'DATA_PT_mesh_attributes', 'DATA_PT_vertex_groups', 'DATA_PT_shape_keys',
    'DATA_PT_normals', 'DATA_PT_customdata', 'DATA_PT_face_maps',
    'DATA_PT_context_light', 'DATA_PT_light', 'DATA_PT_EEVEE_light',
    'DATA_PT_spot', 'DATA_PT_area',
    'WORLD_PT_context_world', 'WORLD_PT_viewport_display',
    'RENDER_PT_color_management', 'RENDER_PT_color_management_curves',
}
# VIEWLAYER_PT_layer_passes is deliberately *not* forced: it lists Mist,
# Vector, Denoising Data and the light-component passes, none of which this
# engine produces. Halcyon has its own Passes panel offering the six it can
# actually fill, which is the same rule EXCLUDED_PANELS exists for.

_patched = []


def _panel_classes():
    seen = set()
    stack = list(bpy.types.Panel.__subclasses__())
    while stack:
        cls = stack.pop()
        name = getattr(cls, '__name__', '')
        if name in seen:
            continue
        seen.add(name)
        stack.extend(cls.__subclasses__())
        yield name, cls


def enable_compatible_panels():
    """Let Halcyon use every stock property panel that is engine-agnostic."""
    global _patched
    _patched = []
    engine = HalcyonRenderEngine.bl_idname
    for name, cls in _panel_classes():
        if name in EXCLUDED_PANELS:
            continue
        compat = getattr(cls, 'COMPAT_ENGINES', None)
        if compat is None:
            continue
        if GENERIC_MARKER not in compat and name not in FORCED_PANELS:
            continue
        if getattr(cls, 'bl_space_type', '') != 'PROPERTIES':
            continue
        try:
            if engine not in compat:
                compat.add(engine)
                _patched.append(cls)
        except Exception:                                       # noqa: BLE001
            pass
    return _patched


def disable_compatible_panels():
    engine = HalcyonRenderEngine.bl_idname
    for cls in _patched:
        try:
            cls.COMPAT_ENGINES.discard(engine)
        except Exception:                                       # noqa: BLE001
            pass
    _patched.clear()


def register():
    bpy.utils.register_class(HalcyonRenderEngine)
    enable_compatible_panels()


def unregister():
    try:
        from .core import parallel as _par
        _par.shutdown()
    except Exception:                                           # noqa: BLE001
        pass
    disable_compatible_panels()
    try:
        bpy.utils.unregister_class(HalcyonRenderEngine)
    except Exception:                                           # noqa: BLE001
        pass
