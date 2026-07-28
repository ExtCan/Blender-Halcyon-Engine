"""Panels. Grouped the way the settings actually relate, not alphabetically."""

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator, Panel

from .core.settings import RESOLUTION_PRESETS
from .presets.library import PRESETS, apply_preset

ENGINE = 'HALCYON_RENDER'

# error-diffusion kernels, the ones the wavefront path accelerates
_DIFFUSION_KERNELS = {'FLOYD', 'JJN', 'STUCKI', 'ATKINSON', 'BURKES', 'SIERRA',
                      'SIERRA_LITE'}


class HalcyonPanel:
    COMPAT_ENGINES = {ENGINE}
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE


# ------------------------------------------------------------------ operators


class HALCYON_OT_apply_preset(Operator):
    bl_idname = 'halcyon.apply_preset'
    bl_label = "Apply Preset"
    bl_description = "Load this renderer's settings over the current ones"
    bl_options = {'REGISTER', 'UNDO'}

    preset: StringProperty()

    reset: BoolProperty(
        name="Reset First", default=True,
        description="Return every setting to its default before applying, so "
                    "nothing carries over from the previous preset")

    def execute(self, context):
        import dataclasses

        from .core.settings import RenderSettings
        from .presets.library import PRESERVED

        key = self.preset or context.scene.halcyon.preset
        entry = PRESETS.get(key)
        if not entry:
            self.report({'ERROR'}, f"Unknown preset '{key}'")
            return {'CANCELLED'}
        hs = context.scene.halcyon

        if self.reset:
            # property_unset restores each property's registered default, which
            # is generated from the dataclass -- so this and the bpy-free
            # reset_settings() cannot disagree.
            for f in dataclasses.fields(RenderSettings):
                if f.name in PRESERVED or not hasattr(hs, f.name):
                    continue
                try:
                    hs.property_unset(f.name)
                except Exception:                               # noqa: BLE001
                    pass

        for name, value in entry['settings'].items():
            if not hasattr(hs, name):
                continue
            try:
                setattr(hs, name, value)
            except (TypeError, ValueError):
                pass
        r = context.scene.render
        if self.reset and 'resolution_x' not in entry['settings']:
            r.pixel_aspect_x = r.pixel_aspect_y = 1.0
        if 'resolution_x' in entry['settings']:
            r.resolution_x = entry['settings']['resolution_x']
            r.resolution_y = entry['settings']['resolution_y']
            r.resolution_percentage = 100
        if 'pixel_aspect_x' in entry['settings']:
            r.pixel_aspect_x = entry['settings']['pixel_aspect_x']
            r.pixel_aspect_y = entry['settings']['pixel_aspect_y']
        if entry['settings'].get('film_transparent') is not None:
            r.film_transparent = bool(entry['settings']['film_transparent'])
        self.report({'INFO'}, f"Applied {entry['label']}"
                              + (" (settings reset first)" if self.reset else ""))
        return {'FINISHED'}


class HALCYON_OT_set_resolution(Operator):
    bl_idname = 'halcyon.set_resolution'
    bl_label = "Set Resolution"
    bl_description = "Set the output resolution and pixel aspect to a period format"
    bl_options = {'REGISTER', 'UNDO'}

    key: EnumProperty(items=lambda self, ctx: [
        (k, k.replace('_', ' ').title(), f"{v[0]}x{v[1]}")
        for k, v in RESOLUTION_PRESETS.items()])

    def execute(self, context):
        entry = RESOLUTION_PRESETS.get(self.key)
        if not entry:
            return {'CANCELLED'}
        r = context.scene.render
        r.resolution_x = int(entry[0])
        r.resolution_y = int(entry[1])
        r.resolution_percentage = 100
        if len(entry) >= 4:
            r.pixel_aspect_x = float(entry[2])
            r.pixel_aspect_y = float(entry[3])
        return {'FINISHED'}


# -------------------------------------------------------------------- panels


FREE_DISCLAIMER = (
    "This Addon is and always will be free. If you paid for this, you were "
    "scammed. Please demand your money back and report the seller"
)


def draw_disclaimer(layout, width=58, boxed=True):
    """The anti-resale notice, wrapped to whatever panel it is drawn in."""
    target = layout.box() if boxed else layout
    col = target.column(align=True)
    col.scale_y = 0.85
    lines = _wrap(FREE_DISCLAIMER, width)
    for i, line in enumerate(lines):
        col.label(text=line, icon='ERROR' if i == 0 else 'BLANK1')
    return target


class HalcyonPreferences(bpy.types.AddonPreferences):
    """Shown in Preferences > Add-ons, which is where someone who installed a
    resold copy will actually look."""

    bl_idname = __package__

    debug_mode: BoolProperty(
        name="Developer Options", default=False,
        description="Show the diagnostic tools: render passes, the per-frame "
                    "timing breakdown, the scene dump and the experimental "
                    "worker pool. Off by default so they stay out of the way "
                    "of normal work")
    strict_nodes: BoolProperty(
        name="Strict Node Evaluation", default=False,
        description="Raise when a node fails instead of falling back to passing "
                    "its input through. The fallback still produces "
                    "plausible-looking output, which can hide a broken node for "
                    "a long time")

    def draw(self, context):
        layout = self.layout
        box = layout.box()
        box.alert = True
        col = box.column(align=True)
        col.scale_y = 0.9
        for i, line in enumerate(_wrap(FREE_DISCLAIMER, 72)):
            col.label(text=line, icon='ERROR' if i == 0 else 'BLANK1')

        col = layout.column(align=True)
        col.label(text="Halcyon is licensed GPL-3.0-or-later, the same as "
                       "Blender itself.", icon='FILE_TEXT')
        col.label(text="You are free to use, modify and share it. Nobody is "
                       "entitled to charge you for it.", icon='BLANK1')
        layout.separator()
        col = layout.column(align=True)
        col.prop(self, 'debug_mode')
        sub = col.column(align=True)
        sub.active = self.debug_mode
        sub.prop(self, 'strict_nodes')
        if self.debug_mode:
            note = layout.row()
            note.active = False
            note.label(text="A Debug panel is now shown in Render Properties",
                       icon='INFO')
        layout.separator()
        row = layout.row()
        row.active = False
        row.label(text="Halcyon Render Engine "
                       + '.'.join(str(v) for v in _version()), icon='INFO')


def _version():
    from .version import version
    return version()


class HALCYON_OT_fix_view_transform(Operator):
    bl_idname = 'halcyon.fix_view_transform'
    bl_label = "Set View Transform to Standard"
    bl_description = ("Halcyon hands Blender pixels that are already "
                      "display-referred; a second view transform double-applies")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        context.scene.view_settings.view_transform = 'Standard'
        self.report({'INFO'}, "View transform set to Standard")
        return {'FINISHED'}


class HALCYON_PT_presets(HalcyonPanel, Panel):
    bl_label = "Halcyon Presets"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        hs = context.scene.halcyon
        col = layout.column(align=True)
        col.prop(hs, 'preset', text="")
        op = col.operator('halcyon.apply_preset', icon='IMPORT')
        op.preset = hs.preset
        op.reset = True
        row = col.row(align=True)
        op = row.operator('halcyon.apply_preset', text="Add On Top",
                          icon='PLUS')
        op.preset = hs.preset
        op.reset = False
        op = row.operator('halcyon.apply_preset', text="Reset All",
                          icon='LOOP_BACK')
        op.preset = 'DEFAULT'
        op.reset = True
        entry = PRESETS.get(hs.preset)
        if entry:
            box = layout.box()
            box.scale_y = 0.8
            for line in _wrap(entry['note'], 46):
                box.label(text=line)
        layout.separator()
        layout.menu('HALCYON_MT_resolutions', icon='OUTPUT')
        layout.separator()
        draw_disclaimer(layout)


class HALCYON_MT_resolutions(bpy.types.Menu):
    bl_idname = 'HALCYON_MT_resolutions'
    bl_label = "Period Resolutions"

    def draw(self, context):
        for k, v in RESOLUTION_PRESETS.items():
            self.layout.operator('halcyon.set_resolution',
                                 text=f"{k.replace('_', ' ').title()}  "
                                      f"({v[0]}x{v[1]})").key = k


class HALCYON_PT_sampling(HalcyonPanel, Panel):
    bl_label = "Sampling"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'aa_mode')
        sub = col.column()
        sub.active = hs.aa_mode != 'NONE'
        sub.prop(hs, 'aa_samples')
        sub.prop(hs, 'aa_filter')
        sub.prop(hs, 'aa_filter_width')
        col.separator()
        col.prop(hs, 'seed')


class HALCYON_PT_geometry(HalcyonPanel, Panel):
    bl_label = "Geometry"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'backface_cull')
        col.prop(hs, 'two_sided_lighting')
        col.separator()
        col.prop(hs, 'vertex_snap')
        sub = col.column()
        sub.active = hs.vertex_snap
        sub.prop(hs, 'vertex_snap_grid')
        col.separator()
        col.prop(hs, 'depth_sort')
        sub = col.column()
        sub.active = hs.depth_sort == 'PAINTERS'
        sub.prop(hs, 'painters_key')
        col.prop(hs, 'depth_precision')


class HALCYON_PT_shading(HalcyonPanel, Panel):
    bl_label = "Shading"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'shading_rate')
        col.prop(hs, 'default_model')
        col.prop(hs, 'force_model')
        col.prop(hs, 'normal_source')
        col.prop(hs, 'displacement_scale')
        col.separator()
        col.prop(hs, 'specular_in_gamma')
        col.prop(hs, 'clamp_specular')
        col.prop(hs, 'light_clamp')


class HALCYON_PT_lighting(HalcyonPanel, Panel):
    bl_label = "Lighting"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'global_ambient')
        col.prop(hs, 'global_ambient_level')
        col.separator()
        col.prop(hs, 'max_lights')
        sub = col.column()
        sub.active = hs.max_lights > 0
        sub.prop(hs, 'light_limit_mode')
        col.prop(hs, 'light_falloff_default')


class HALCYON_PT_shadows(HalcyonPanel, Panel):
    bl_label = "Shadows"
    bl_parent_id = 'HALCYON_PT_lighting'

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'shadows', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.shadows
        col.prop(hs, 'shadow_default')
        col.prop(hs, 'shadow_map_size')
        col.prop(hs, 'shadow_bias')
        col.prop(hs, 'shadow_softness')
        col.prop(hs, 'shadow_samples')


class HALCYON_PT_ao(HalcyonPanel, Panel):
    bl_label = "Ambient Occlusion"
    bl_parent_id = 'HALCYON_PT_lighting'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'ambient_occlusion', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.ambient_occlusion
        col.label(text="Not period correct -- off by default", icon='INFO')
        col.prop(hs, 'ao_distance')
        col.prop(hs, 'ao_samples')
        col.prop(hs, 'ao_intensity')


class HALCYON_PT_raytrace(HalcyonPanel, Panel):
    bl_label = "Ray Tracing"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'raytrace', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.raytrace
        col.prop(hs, 'ray_depth')
        col.prop(hs, 'ray_reflection')
        col.prop(hs, 'ray_refraction')
        col.prop(hs, 'ray_shadows')
        col.prop(hs, 'ray_bias')
        col.separator()
        col.prop(hs, 'env_reflection')


class HALCYON_PT_textures(HalcyonPanel, Panel):
    bl_label = "Textures"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'tex_filter')
        col.prop(hs, 'tex_mipmap')
        sub = col.column()
        sub.active = hs.tex_mipmap
        sub.prop(hs, 'tex_mip_bias')
        col.separator()
        col.prop(hs, 'tex_perspective')
        col.separator()
        col.prop(hs, 'tex_max_size')
        col.prop(hs, 'tex_quantize')
        col.prop(hs, 'tex_wrap_default')


class HALCYON_PT_transparency(HalcyonPanel, Panel):
    bl_label = "Transparency"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'transparency')
        if hs.transparency == 'STIPPLE':
            col.prop(hs, 'stipple_pattern')
        col.prop(hs, 'max_transparent_layers')
        col.prop(hs, 'alpha_bits')
        col.prop(hs, 'alpha_threshold')


class HALCYON_PT_fog(HalcyonPanel, Panel):
    bl_label = "Fog / Depth Cue"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'fog', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.fog
        col.prop(hs, 'fog_mode')
        col.prop(hs, 'fog_color')
        if hs.fog_mode in ('LINEAR', 'TABLE16'):
            col.prop(hs, 'fog_start')
            col.prop(hs, 'fog_end')
        else:
            col.prop(hs, 'fog_density')
        col.prop(hs, 'fog_vertex')


class HALCYON_PT_effects(HalcyonPanel, Panel):
    bl_label = "Optical Effects"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column(heading="Glow")
        col.prop(hs, 'glow', text="Enable")
        sub = col.column()
        sub.active = hs.glow
        sub.prop(hs, 'glow_threshold')
        sub.prop(hs, 'glow_radius')
        sub.prop(hs, 'glow_intensity')
        sub.prop(hs, 'glow_quality')
        col = layout.column(heading="Star Filter")
        col.prop(hs, 'star_filter', text="Enable")
        sub = col.column()
        sub.active = hs.star_filter
        sub.prop(hs, 'star_points')
        sub.prop(hs, 'star_length')
        sub.prop(hs, 'star_rotation')
        sub.prop(hs, 'star_intensity')
        col = layout.column(heading="Lens")
        col.prop(hs, 'lens_distortion')
        col.prop(hs, 'chromatic_aberration')
        col.prop(hs, 'lens_vignette_edges')
        col = layout.column(heading="Light Shafts")
        sub = col.column()
        sub.prop(hs, 'shaft_threshold')
        sub.prop(hs, 'shaft_length')
        sub.prop(hs, 'shaft_decay')
        sub.prop(hs, 'shaft_samples')
        row = col.row()
        row.active = False
        row.label(text="Set Volumetric on a light to cast them", icon='LIGHT')
        col = layout.column(heading="Depth of Field")
        col.prop(hs, 'dof', text="Enable")
        sub = col.column()
        sub.active = hs.dof
        sub.prop(hs, 'dof_focus')
        sub.prop(hs, 'dof_amount')
        sub.prop(hs, 'dof_layers')
        sub.prop(hs, 'dof_max_radius')
        col = layout.column(heading="Lens Flare")
        col.prop(hs, 'lens_flare', text="Enable")
        sub = col.column()
        sub.active = hs.lens_flare
        sub.prop(hs, 'flare_intensity')
        sub.prop(hs, 'flare_ghosts')
        sub.prop(hs, 'flare_streak')


class HALCYON_PT_colour(HalcyonPanel, Panel):
    bl_label = "Colour Depth"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'color_depth')
        indexed = hs.color_depth in ('8', '4', 'HAM8', 'HAM6')
        sub = col.column()
        sub.active = indexed
        sub.prop(hs, 'palette_mode')
        s2 = sub.column()
        s2.active = hs.palette_mode == 'ADAPTIVE'
        s2.prop(hs, 'palette_size')
        s2.prop(hs, 'palette_method')
        s2.prop(hs, 'palette_lock')
        if hs.palette_lock:
            s2.operator('halcyon.clear_palette_cache', icon='FILE_REFRESH')
        col.separator()
        col.prop(hs, 'dither')
        s3 = col.column()
        s3.active = hs.dither != 'NONE'
        s3.prop(hs, 'dither_strength')
        s3.prop(hs, 'dither_serpentine')
        if hs.dither_serpentine and hs.dither in _DIFFUSION_KERNELS:
            note = s3.row()
            note.active = False
            note.label(text="Off is ~2x faster (diagonal processing)",
                       icon='SORTTIME')


class HALCYON_PT_display(HalcyonPanel, Panel):
    bl_label = "Display"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        view = getattr(context.scene, 'view_settings', None)
        if view is not None and getattr(view, 'view_transform', 'Standard') \
                not in ('Standard', 'Raw'):
            box = layout.box()
            box.alert = True
            box.label(text=f"Blender's view transform is "
                           f"{view.view_transform}", icon='ERROR')
            col = box.column(align=True)
            col.scale_y = 0.8
            for line in _wrap("Halcyon already outputs display-referred pixels, "
                              "so a second transform on top will wash them out. "
                              "Set it to Standard.", 44):
                col.label(text=line)
            box.operator('halcyon.fix_view_transform', icon='FILE_REFRESH')
            layout.separator()
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'color_management')
        col.prop(hs, 'exposure')
        col.prop(hs, 'gamma')
        col.prop(hs, 'brightness')
        col.prop(hs, 'contrast')
        col.prop(hs, 'saturation')
        col.separator()
        col.prop(hs, 'output_scale')
        col.prop(hs, 'pixel_grid')
        col.separator()
        col.prop(hs, 'film_transparent')


class HALCYON_PT_crt(HalcyonPanel, Panel):
    bl_label = "CRT Simulation"
    bl_parent_id = 'HALCYON_PT_display'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'crt', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.crt
        col.prop(hs, 'crt_scanlines')
        col.prop(hs, 'crt_mask')
        sub = col.column()
        sub.active = hs.crt_mask != 'NONE'
        sub.prop(hs, 'crt_mask_strength')
        col.prop(hs, 'crt_bloom')
        col.prop(hs, 'crt_curvature')
        col.prop(hs, 'crt_vignette')


class HALCYON_PT_composite(HalcyonPanel, Panel):
    bl_label = "Composite Video"
    bl_parent_id = 'HALCYON_PT_display'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'composite', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.composite
        col.prop(hs, 'composite_bleed')
        col.prop(hs, 'composite_ringing')
        col.prop(hs, 'composite_dot_crawl')
        col.separator()
        col.prop(hs, 'interlace')


class HALCYON_PT_jpeg(HalcyonPanel, Panel):
    bl_label = "JPEG Artefacts"
    bl_parent_id = 'HALCYON_PT_display'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'jpeg_artifacts', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.jpeg_artifacts
        col.prop(hs, 'jpeg_quality')
        col.prop(hs, 'jpeg_passes')
        col.prop(hs, 'block_size')


class HALCYON_PT_performance(HalcyonPanel, Panel):
    bl_label = "Performance"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'cache_shadows')
        col.prop(hs, 'fast_background')
        row = col.row()
        row.active = False
        row.label(text=f"{_cpu_count()} logical cores detected", icon='INFO')
        col.separator()
        col.prop(hs, 'preview_scale')



# ------------------------------------------------------------ data panels


class HALCYON_PT_world_ground(HalcyonPanel, Panel):
    bl_label = "Infinite Ground"
    bl_parent_id = 'HALCYON_PT_world'

    @classmethod
    def poll(cls, context):
        # shown whatever the sky mode is: the ground works under a node tree
        # as readily as under a gradient, and hiding it made it unfindable
        return context.engine == ENGINE and context.world is not None

    def draw_header(self, context):
        self.layout.prop(context.world.halcyon, 'ground_plane', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        col.active = hs.ground_plane
        col.prop(hs, 'ground_mode')
        col.prop(hs, 'ground_height')
        col.prop(hs, 'ground_color')
        if hs.ground_mode in ('CHECKER', 'NOISE'):
            col.prop(hs, 'ground_color2')
        if hs.ground_mode != 'SOLID':
            col.prop(hs, 'ground_scale')
        if hs.ground_mode == 'OCEAN':
            col.prop(hs, 'ocean_choppiness')
            col.prop(hs, 'ocean_speed')
        col.separator()
        col.prop(hs, 'ground_fade')
        if not hs.ground_plane:
            note = layout.column(align=True)
            note.active = False
            note.scale_y = 0.8
            for line in _wrap("This is a world property, not an object -- there "
                              "is nothing to add to the scene. Tick the box in "
                              "this panel's header.", 46):
                note.label(text=line)


class _BrycePanel(HalcyonPanel):
    """Sub-panels that only make sense for the Bryce atmosphere."""

    bl_parent_id = 'HALCYON_PT_world'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.engine == ENGINE and context.world is not None
                and context.world.halcyon.mode == 'BRYCE')


class HALCYON_PT_world_sun(_BrycePanel, Panel):
    bl_label = "Sun & Corona"
    bl_options = set()

    @classmethod
    def poll(cls, context):
        return (context.engine == ENGINE and context.world is not None
                and context.world.halcyon.mode in ('BRYCE', 'PHYSICAL'))

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        if hs.mode == 'BRYCE':
            col.prop(hs, 'celestial')
        col.prop(hs, 'sun_elevation')
        col.prop(hs, 'sun_rotation')
        col.prop(hs, 'sun_intensity')
        if hs.mode == 'BRYCE' and hs.celestial == 'MOON':
            col.prop(hs, 'moon_color')
            col.prop(hs, 'moon_size')
            col.prop(hs, 'moon_phase')
            col.prop(hs, 'moon_earthshine')
        else:
            col.prop(hs, 'sun_color')
            if hs.mode == 'BRYCE':
                col.prop(hs, 'sun_glow')
                col.prop(hs, 'sun_corona')
            col.separator()
            col.prop(hs, 'sun_disc')
            sub = col.column()
            sub.active = hs.sun_disc
            sub.prop(hs, 'sun_size')


class HALCYON_PT_world_atmosphere(_BrycePanel, Panel):
    bl_label = "Atmosphere"
    bl_options = set()

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column(heading="Haze")
        col.prop(hs, 'haze_density', text="Amount")
        sub = col.column()
        sub.active = hs.haze_density > 0.0
        sub.prop(hs, 'haze_color')
        sub.prop(hs, 'haze_height')
        sub.prop(hs, 'haze_sun_tint')
        sub.prop(hs, 'haze_blend_sky')
        col = layout.column(heading="Atmosphere")
        col.prop(hs, 'atmosphere_density', text="Density")
        sub = col.column()
        sub.active = hs.atmosphere_density > 0.0
        sub.prop(hs, 'atmosphere_color')
        sub.prop(hs, 'atmosphere_falloff')
        col = layout.column(heading="Ground Fog")
        col.prop(hs, 'fog_density', text="Amount")
        sub = col.column()
        sub.active = hs.fog_density > 0.0
        sub.prop(hs, 'fog_color')
        sub.prop(hs, 'fog_height')


class HALCYON_PT_world_cumulus(_BrycePanel, Panel):
    bl_label = "Cumulus"

    def draw_header(self, context):
        self.layout.prop(context.world.halcyon, 'clouds', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        col.active = hs.clouds
        col.prop(hs, 'cloud_cover')
        col.prop(hs, 'cloud_density')
        col.prop(hs, 'cloud_scale')
        col.prop(hs, 'cloud_height')
        col.prop(hs, 'cloud_thickness')
        col.prop(hs, 'cloud_softness')
        col.prop(hs, 'cloud_detail')
        col.prop(hs, 'cloud_seed')
        col.separator()
        col.prop(hs, 'cloud_wind')
        col.prop(hs, 'cloud_wind_angle')
        col.separator()
        col.prop(hs, 'cloud_color')
        col.prop(hs, 'cloud_shadow')
        col.prop(hs, 'cloud_rim')
        col.prop(hs, 'cloud_ambience')
        col.prop(hs, 'cloud_shadows')


class HALCYON_PT_world_stratus(_BrycePanel, Panel):
    bl_label = "Stratus"

    def draw_header(self, context):
        self.layout.prop(context.world.halcyon, 'stratus', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        col.active = hs.stratus
        col.prop(hs, 'stratus_amount')
        col.prop(hs, 'stratus_density')
        col.prop(hs, 'stratus_scale')
        col.prop(hs, 'stratus_altitude')
        col.prop(hs, 'stratus_squash')
        col.prop(hs, 'stratus_sharpness')
        col.prop(hs, 'stratus_detail')
        col.separator()
        col.prop(hs, 'stratus_color')


class HALCYON_PT_world_effects(_BrycePanel, Panel):
    bl_label = "Rainbow & Stars"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column(heading="Rainbow")
        col.prop(hs, 'rainbow', text="Enable")
        sub = col.column()
        sub.active = hs.rainbow
        sub.prop(hs, 'rainbow_intensity')
        sub.prop(hs, 'rainbow_radius')
        sub.prop(hs, 'rainbow_width')
        sub.prop(hs, 'rainbow_secondary')
        col = layout.column(heading="Stars")
        col.prop(hs, 'stars', text="Enable")
        sub = col.column()
        sub.active = hs.stars
        sub.prop(hs, 'star_density')
        sub.prop(hs, 'star_brightness')


def prefs(context=None):
    """Add-on preferences, or None if they cannot be reached."""
    import bpy as _bpy
    ctx = context or getattr(_bpy, 'context', None)
    try:
        return ctx.preferences.addons[__package__].preferences
    except Exception:                                           # noqa: BLE001
        return None


def debug_enabled(context=None):
    p = prefs(context)
    return bool(getattr(p, 'debug_mode', False))


def material_state(mat):
    """(model, converted) for a material, without evaluating anything."""
    if mat is None:
        return '-', False
    hs = getattr(mat, 'halcyon', None)
    if hs is not None and hs.use_override:
        return hs.model, True
    if mat.use_nodes and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.bl_idname == 'HALCYON_ShaderNode':
                return node.model, True
        for node in mat.node_tree.nodes:
            if node.bl_idname == 'HALCYON_CodeNode':
                return 'CODED', True
    return 'auto', False


class HALCYON_UL_materials(bpy.types.UIList):
    """Material slots with the Halcyon model each one resolves to."""

    def draw_item(self, context, layout, data, item, icon, active_data,
                  active_propname, index):
        mat = item.material
        if self.layout_type in {'DEFAULT', 'COMPACT'}:
            row = layout.row(align=True)
            if mat is None:
                row.label(text="(empty slot)", icon='MATERIAL')
                return
            row.prop(mat, 'name', text="", emboss=False,
                     icon_value=layout.icon(mat))
            model, converted = material_state(mat)
            sub = row.row()
            sub.alignment = 'RIGHT'
            sub.active = converted
            sub.label(text=model.replace('_', ' ').title() if converted
                      else "not converted")
            row.label(text="", icon='CHECKMARK' if converted else 'DOT')
        else:
            layout.label(text="", icon_value=icon)


class HALCYON_OT_clear_palette_cache(Operator):
    bl_idname = 'halcyon.clear_palette_cache'
    bl_label = "Rebuild Palette"
    bl_description = ("Discard the locked palette so the next render builds a "
                      "fresh one from the current scene")

    def execute(self, context):
        from .core.palette import clear_caches
        clear_caches()
        self.report({'INFO'}, "Palette cache cleared")
        return {'FINISHED'}


class HALCYON_OT_diagnostics(Operator):
    bl_idname = 'halcyon.diagnostics'
    bl_label = "Print Halcyon Diagnostics"
    bl_description = ("Dump what the engine actually exports -- node trees, "
                      "materials, world and settings -- to the system console")

    def execute(self, context):
        import pprint
        from . import export as _export
        from .engine import _settings_from_scene
        scene = context.scene
        depsgraph = context.evaluated_depsgraph_get()
        st = _settings_from_scene(scene, scene.render.resolution_x,
                                  scene.render.resolution_y)
        warnings = []
        try:
            exported = _export.export_scene(depsgraph, st, warnings)
        except Exception as exc:                                # noqa: BLE001
            self.report({'ERROR'}, f"export failed: {exc}")
            return {'CANCELLED'}
        print("=" * 70)
        print("HALCYON DIAGNOSTICS")
        print("=" * 70)
        mesh = exported.mesh
        print(f"triangles      : {0 if mesh is None or mesh.tris is None else len(mesh.tris)}")
        print(f"objects        : {len(exported.objects)}")
        print(f"lights         : {[(l.type, round(l.energy, 2)) for l in exported.lights]}")
        print(f"images         : {list(getattr(exported, 'images', {}))}")
        print(f"world mode     : {exported.world.mode}  graph={'yes' if exported.world.graph else 'no'}")
        print(f"debug pass     : {st.debug_pass}")
        print(f"resolution     : {st.resolution_x}x{st.resolution_y} "
              f"scale={st.output_scale} aa={st.aa_mode}/{st.aa_samples}")
        for m in exported.materials:
            print(f"-- material {m.name!r} model={m.model} override={m.use_override} "
                  f"alpha={getattr(m, 'has_alpha', False)}")
            if m.graph:
                for nid, nd in m.graph['nodes'].items():
                    links = [(i['name'], i['link']) for i in nd['inputs'] if i['link']]
                    print(f"     {nd['bl_idname']:34s} props={nd['props']} links={links}")
                print(f"     output node = {m.graph['output']}")
        if exported.world.graph:
            print("-- world graph")
            for nid, nd in exported.world.graph['nodes'].items():
                links = [(i['name'], i['link']) for i in nd['inputs'] if i['link']]
                print(f"     {nd['bl_idname']:34s} props={nd['props']} links={links}")
            print(f"     output node = {exported.world.graph['output']}")
        if warnings:
            print("warnings:", warnings)
        print("=" * 70)
        self.report({'INFO'}, "Diagnostics written to the system console")
        return {'FINISHED'}


class HALCYON_PT_debug(HalcyonPanel, Panel):
    bl_label = "Debug"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and debug_enabled(context)

    def draw(self, context):
        layout = self.layout
        hs = context.scene.halcyon
        p = prefs(context)

        col = layout.column()
        col.use_property_split = True
        col.prop(hs, 'debug_pass')
        col.prop(hs, 'show_stats')
        if hs.show_stats:
            note = col.row()
            note.active = False
            note.label(text="Breakdown prints to the system console",
                       icon='CONSOLE')
        layout.operator('halcyon.diagnostics', icon='CONSOLE')

        box = layout.box()
        box.label(text="Measure this machine", icon='SYSTEM')
        col = box.column(align=True)
        col.operator('halcyon.selftest', icon='PLAY')
        note = box.column(align=True)
        note.active = False
        note.scale_y = 0.8
        for line in _wrap("Compiles and runs the GPU shaders on your driver, "
                          "times thread scaling and the worker pool, and prints "
                          "a report to the console and clipboard.", 46):
            note.label(text=line)

        box = layout.box()
        box.label(text="Experimental", icon='ERROR')
        col = box.column()
        col.use_property_split = True
        col.prop(hs, 'render_device')
        if str(hs.render_device).upper() == 'GPU':
            from .gpu import capability as _cap
            from .gpu import device as _dev
            sub = box.column(align=True)
            sub.scale_y = 0.85
            sub.label(text=_dev.describe(), icon='INFO')
            sub.separator()
            for feat, support, why in _cap.summary():
                row = sub.row()
                if support == _cap.BOTH:
                    row.label(text=feat.replace('_', ' ').title(),
                              icon='CHECKMARK')
                elif support == _cap.NEVER:
                    row.active = False
                    row.label(text=feat.replace('_', ' ').title()
                              + " — CPU only, always", icon='X')
                else:
                    row.active = False
                    row.label(text=feat.replace('_', ' ').title()
                              + " — CPU for now", icon='TIME')
            note = box.row()
            note.active = False
            note.label(text="Unsupported work falls back automatically",
                       icon='INFO')
        col.separator()
        col.prop(hs, 'use_processes')
        sub = col.column()
        sub.active = hs.use_processes
        sub.prop(hs, 'process_count')
        note = box.row()
        note.active = False
        note.label(text="Speedup unverified; falls back if workers fail")

        if p is not None and p.strict_nodes:
            row = layout.row()
            row.active = False
            row.label(text="Strict node evaluation is on", icon='CHECKMARK')


class HALCYON_PT_material(HalcyonPanel, Panel):
    bl_label = "Halcyon Material"
    bl_context = "material"

    @classmethod
    def poll(cls, context):
        # An object with no material at all still needs this panel -- it is
        # where the slot list and the New button live. Requiring a material to
        # already exist made it impossible to create the first one from here.
        if context.engine != ENGINE:
            return False
        ob = context.object
        return (context.material is not None
                or (ob is not None and hasattr(ob, 'material_slots')))

    def draw(self, context):
        layout = self.layout
        ob = context.object
        mat = context.material

        if ob is not None and hasattr(ob, 'material_slots'):
            slots = ob.material_slots
            # always drawn, whatever the slot count: this is the only way to add
            # a material from this panel
            row = layout.row()
            row.template_list('HALCYON_UL_materials', '', ob, 'material_slots',
                              ob, 'active_material_index',
                              rows=min(max(len(slots), 2), 8))
            col = row.column(align=True)
            col.operator('object.material_slot_add', icon='ADD', text="")
            col.operator('object.material_slot_remove', icon='REMOVE', text="")
            if len(slots) > 1:
                col.separator()
                col.operator('object.material_slot_select',
                             icon='RESTRICT_SELECT_OFF', text="")

            slot = slots[ob.active_material_index] if len(slots) else None
            if slot is not None:
                layout.template_ID(slot, 'material', new='material.new')
            else:
                layout.operator('object.material_slot_add',
                                text="New Material Slot", icon='ADD')

            if len(slots) > 1:
                n_conv = sum(1 for sl in slots if material_state(sl.material)[1])
                sub = layout.row()
                sub.active = False
                sub.label(text=f"{n_conv} of {len(slots)} slots converted")
            layout.separator()

        if mat is None:
            info = layout.column(align=True)
            info.active = False
            info.label(text="No material on this slot yet", icon='INFO')
            info.label(text="Use New above, then convert it")
            return
        hs = mat.halcyon

        has_master = bool(mat.use_nodes and mat.node_tree and any(
            n.bl_idname == 'HALCYON_ShaderNode' for n in mat.node_tree.nodes))
        box = layout.box()
        row = box.row()
        row.label(text="Halcyon Shader" if has_master else "Convert Material",
                  icon='CHECKMARK' if has_master else 'NODE_MATERIAL')
        col = box.column(align=True)
        op = col.operator('halcyon.convert_materials', text="Convert This Material",
                          icon='MATERIAL')
        op.scope = 'ACTIVE'
        op.force = has_master
        op = col.operator('halcyon.convert_materials',
                          text="Convert Selected Objects", icon='RESTRICT_SELECT_OFF')
        op.scope = 'SELECTED'
        op = col.operator('halcyon.convert_materials', text="Convert Whole Scene",
                          icon='SCENE_DATA')
        op.scope = 'SCENE'
        box.label(text="Textures are relinked, not discarded", icon='INFO')

        box = layout.box()
        box.label(text="Start from a template", icon='PRESET')
        box.operator_menu_enum('halcyon.material_template', 'template',
                               text="Material Templates", icon='MATERIAL')

        layout.separator()
        layout.prop(hs, 'use_override')
        if not hs.use_override:
            layout.label(text="Using this material's node tree", icon='NODETREE')
        layout.use_property_split = True
        col = layout.column()
        col.active = hs.use_override
        col.prop(hs, 'model')
        col.separator()
        col.prop(hs, 'diffuse')
        col.prop(hs, 'diffuse_level')
        col.prop(hs, 'specular')
        col.prop(hs, 'specular_level')
        col.prop(hs, 'glossiness')
        col.prop(hs, 'soften')
        col.separator()
        col.prop(hs, 'roughness')
        col.prop(hs, 'metallic')
        col.prop(hs, 'anisotropy')
        col.prop(hs, 'aniso_rotation')
        col.separator()
        col.prop(hs, 'ambient_level')
        col.prop(hs, 'emission')
        col.prop(hs, 'emission_level')
        col.prop(hs, 'opacity')
        col.prop(hs, 'ior')
        col.prop(hs, 'reflect_level')
        col.separator()
        row = col.row(align=True)
        row.prop(hs, 'two_sided')
        row.prop(hs, 'shadeless')
        row = col.row(align=True)
        row.prop(hs, 'cast_shadow')
        row.prop(hs, 'receive_shadow')
        col.prop(hs, 'wire')
        if hs.wire:
            col.prop(hs, 'wire_size')


class HALCYON_PT_light(HalcyonPanel, Panel):
    bl_label = "Halcyon Light"
    bl_context = "data"

    @classmethod
    def poll(cls, context):
        return (context.engine == ENGINE and context.light is not None)

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        light = context.light
        hs = light.halcyon
        col = layout.column()
        col.prop(light, 'color')
        col.prop(light, 'energy')
        col.separator()
        col.prop(hs, 'decay')
        if hs.decay == 'CUSTOM':
            col.prop(hs, 'decay_start')
            col.prop(hs, 'decay_end')
        if light.type == 'SPOT':
            col.prop(hs, 'hotspot')
        col.separator()
        col.prop(hs, 'shadow')
        sub = col.column()
        sub.active = hs.shadow != 'NONE'
        sub.prop(hs, 'shadow_map_size')
        sub.prop(hs, 'shadow_bias')
        sub.prop(hs, 'shadow_softness')
        sub.prop(hs, 'shadow_samples')
        sub.prop(hs, 'shadow_density')
        sub.prop(hs, 'shadow_color')
        col.separator()
        row = col.row(align=True)
        row.prop(hs, 'diffuse_only')
        row.prop(hs, 'specular_only')
        col.prop(hs, 'negative')
        col.prop(hs, 'ambient_only')
        col.prop(hs, 'volumetric')
        col.separator()
        col.prop(hs, 'exclude_collection')
        sub = col.column()
        sub.active = hs.exclude_collection is not None
        sub.prop(hs, 'exclude_mode')


class HALCYON_PT_world(HalcyonPanel, Panel):
    bl_label = "Halcyon World"
    bl_context = "world"

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and context.world is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        col.prop(hs, 'mode')
        if hs.mode != 'NODES':
            col.prop(hs, 'strength')
            col.prop(hs, 'rotation')
        col.separator()

        m = hs.mode
        if m == 'SOLID':
            col.prop(hs, 'color')
        elif m in ('GRADIENT', 'BRYCE'):
            col.prop(hs, 'horizon')
            if m == 'BRYCE':
                col.prop(hs, 'use_sky_mid')
                sub = col.column()
                sub.active = hs.use_sky_mid
                sub.prop(hs, 'sky_mid')
                sub.prop(hs, 'sky_mid_height')
            col.prop(hs, 'zenith')
            col.prop(hs, 'gradient_falloff')
            if m == 'GRADIENT':
                col.prop(hs, 'blend_mode')
                col.prop(hs, 'horizon_height')
        elif m == 'HDRI':
            col.template_ID(hs, 'env_image', open='image.open')
            col.prop(hs, 'env_mapping')
            col.prop(hs, 'env_filter')
            col.prop(hs, 'env_tint')
        elif m == 'PHYSICAL':
            col.prop(hs, 'turbidity')
            col.prop(hs, 'ground_albedo')

        if m in ('GRADIENT', 'BRYCE', 'PHYSICAL'):
            col.separator()
            col.prop(hs, 'show_ground')
            sub = col.column()
            sub.active = hs.show_ground
            sub.prop(hs, 'ground_color')

        col.separator()
        col.label(text="Ambient")
        col.prop(hs, 'ambient')
        col.prop(hs, 'ambient_level')


class HALCYON_PT_world_clouds(HalcyonPanel, Panel):
    bl_label = "Clouds"
    bl_parent_id = 'HALCYON_PT_world'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return (context.engine == ENGINE and context.world is not None
                and context.world.halcyon.mode == 'BRYCE')

    def draw_header(self, context):
        self.layout.prop(context.world.halcyon, 'clouds', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        col.active = hs.clouds
        col.prop(hs, 'cloud_cover')
        col.prop(hs, 'cloud_density')
        col.prop(hs, 'cloud_scale')
        col.prop(hs, 'cloud_height')
        col.prop(hs, 'cloud_detail')
        col.prop(hs, 'cloud_softness')
        col.prop(hs, 'cloud_seed')
        col.separator()
        col.prop(hs, 'cloud_color')
        col.prop(hs, 'cloud_shadow')


def _cpu_count():
    import os
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _wrap(text, width):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines


CLASSES = (
    HALCYON_OT_apply_preset, HALCYON_OT_set_resolution, HALCYON_MT_resolutions,
    HALCYON_OT_fix_view_transform, HALCYON_UL_materials,
    HalcyonPreferences,
    HALCYON_PT_presets, HALCYON_PT_sampling, HALCYON_PT_geometry,
    HALCYON_PT_shading, HALCYON_PT_lighting, HALCYON_PT_shadows, HALCYON_PT_ao,
    HALCYON_PT_raytrace, HALCYON_PT_textures, HALCYON_PT_transparency,
    HALCYON_PT_fog, HALCYON_PT_effects, HALCYON_PT_colour, HALCYON_PT_display,
    HALCYON_PT_crt, HALCYON_PT_composite, HALCYON_PT_jpeg,
    HALCYON_PT_performance, HALCYON_PT_debug,
    HALCYON_OT_clear_palette_cache,
    HALCYON_OT_diagnostics, HALCYON_PT_material,
    HALCYON_PT_light, HALCYON_PT_world, HALCYON_PT_world_sun,
    HALCYON_PT_world_atmosphere, HALCYON_PT_world_cumulus,
    HALCYON_PT_world_stratus, HALCYON_PT_world_effects,
    HALCYON_PT_world_ground,
)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
