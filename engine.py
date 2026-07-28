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

    def __del__(self):
        pass

    # ------------------------------------------------------------ final render
    def render(self, depsgraph):
        bscene = depsgraph.scene
        tw, th = self._target_size(bscene)
        preview = bool(getattr(self, 'is_preview', False))
        settings = _settings_from_scene(bscene, tw, th, preview)
        warnings = []

        self.update_stats("Halcyon", "Exporting scene")
        ST.reset()
        ST.enable(bool(settings.show_stats))
        _apply_debug_prefs()
        try:
            from .gpu import capability as _cap
            dev, _stages, notes = _cap.plan(scene, settings)
            if str(settings.render_device).upper() == 'GPU':
                print(f"[Halcyon] device: {dev}")
                for note in notes:
                    print(f"[Halcyon]   {note}")
        except Exception:                                       # noqa: BLE001
            pass
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

        def on_progress(frac, msg):
            if self.test_break():
                raise _Cancelled()
            self.update_progress(float(frac))
            self.update_stats("Halcyon", msg)

        image = None
        if settings.use_processes and not preview:
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
            self._deliver(final, bscene)

        # reporting during a preview render pushes UI work onto the preview
        # thread for a thumbnail nobody is reading
        if not preview:
            for w in set(warnings) | set(getattr(scene, 'unsupported', ()) or ()):
                self.report({'WARNING'}, str(w))
        elapsed = time.time() - export_started
        if settings.show_stats:
            ST.report(total=elapsed)
        self.update_stats("Halcyon", f"Done in {elapsed:.1f}s")
        self.update_progress(1.0)

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

    def _deliver(self, final, bscene):
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
    def view_update(self, context, depsgraph):
        # the scene only needs re-exporting when something actually changed;
        # orbiting the view fires view_draw, not view_update
        self._scene = None
        self._scene_key = None
        self.tag_redraw()

    def view_draw(self, context, depsgraph):
        import gpu
        from gpu_extras.presets import draw_texture_2d

        region = context.region
        w, h = region.width, region.height
        settings = _settings_from_scene(depsgraph.scene, max(w, 4), max(h, 4))
        scale = max(int(settings.preview_scale), 1)
        settings.resolution_x = max(w // scale, 4)
        settings.resolution_y = max(h // scale, 4)
        settings.aa_mode = 'NONE'
        settings.aa_samples = 1
        settings.output_scale = 'NONE'
        settings.pixel_aspect_x = settings.pixel_aspect_y = 1.0
        settings.raytrace = False
        settings.ambient_occlusion = False

        try:
            # Re-exporting the whole scene on every redraw makes orbiting cost
            # a full mesh conversion per frame. The export is cached and only
            # rebuilt when view_update says something changed; the camera is
            # cheap and is re-applied every draw.
            scene = self._scene
            if scene is None:
                scene = export.export_scene(depsgraph, settings)
                self._scene = scene
            scene.settings = settings
            _apply_view_camera(scene, context, settings)
            img = core_render.render(scene, settings)
            img = post.process(img, settings,
                               target_size=(settings.resolution_x,
                                            settings.resolution_y))
        except Exception:                                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            return

        ih, iw = img.shape[:2]
        if img.shape[2] == 3:
            img = np.concatenate([img, np.ones((ih, iw, 1), np.float32)], 2)
        flat = np.ascontiguousarray(
            np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0),
            dtype=np.float32).ravel()
        if flat.size != ih * iw * 4:            # never hand gpu a short buffer
            return
        buf = gpu.types.Buffer('FLOAT', ih * iw * 4, flat.tolist())
        tex = gpu.types.GPUTexture((iw, ih), format='RGBA16F', data=buf)
        self.bind_display_space_shader(depsgraph.scene)
        draw_texture_2d(tex, (0, 0), w, h)
        self.unbind_display_space_shader()


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


def _apply_view_camera(scene, context, settings):
    """Point the render at the viewport's own camera."""
    rv3d = context.region_data
    if rv3d is None:
        return
    from .core.scene import Camera
    view_matrix = np.asarray(rv3d.view_matrix, np.float32)
    scene.camera = Camera(matrix_world=np.linalg.inv(view_matrix).astype(np.float32),
                          projection=np.asarray(rv3d.window_matrix, np.float32),
                          clip_start=context.space_data.clip_start,
                          clip_end=context.space_data.clip_end)


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
    'VIEWLAYER_PT_layer_passes',
}

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
