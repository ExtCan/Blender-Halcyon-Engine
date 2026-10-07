"""Panels. Grouped the way the settings actually relate, not alphabetically."""

import bpy
from bpy.props import BoolProperty, EnumProperty, StringProperty
from bpy.types import Operator, Panel

from .core.settings import (RESOLUTION_GROUPS, RESOLUTION_PRESETS,
                            resolution_label)
from .presets.library import DEVICE_KEYS, PRESETS, apply_preset

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


def _poke_updated(context, *ids):
    """Tell Blender that `ids` changed, so the rendered view redraws NOW.

    A preset writes dozens of properties through plain setattr, and none of
    those writes tags the datablock for the depsgraph -- so a Bryce sky
    preset applied over a rendered viewport showed the OLD sky until the
    user nudged any slider (whose update callback finally did the tagging).
    Applying a preset must poke the same machinery a slider pokes.
    """
    for _id in ids:
        if _id is None:
            continue
        try:
            _id.update_tag()
        except Exception:                                       # noqa: BLE001
            pass
    try:
        scr = getattr(context, 'screen', None)
        for area in (scr.areas if scr else ()):
            area.tag_redraw()
    except Exception:                                           # noqa: BLE001
        pass


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
            if name in DEVICE_KEYS:
                continue                # a look never chooses the device
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
        _poke_updated(context, context.scene)
        self.report({'INFO'}, f"Applied {entry['label']}"
                              + (" (settings reset first)" if self.reset else ""))
        return {'FINISHED'}


class HALCYON_OT_set_resolution(Operator):
    bl_idname = 'halcyon.set_resolution'
    bl_label = "Set Resolution"
    bl_description = "Set the output resolution and pixel aspect to a period format"
    bl_options = {'REGISTER', 'UNDO'}

    key: EnumProperty(items=lambda self, ctx: [
        (k, resolution_label(k), f"{v[0]}x{v[1]}")
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
    bl_label = "Disable Blender's Color Management (Raw)"
    bl_description = ("Halcyon's output is display-referred and this panel "
                      "is its whole grading chain. ANY Blender view "
                      "transform regrades it a second time -- even "
                      "'Standard', which encodes for scene-linear data "
                      "and washes the render grey. Raw shows the "
                      "engine's pixels untouched")
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .engine import _pin_display
        _pin_display(context.scene)
        self.report({'INFO'}, "Blender color management disabled (Raw)")
        return {'FINISHED'}


class HALCYON_PT_presets(HalcyonPanel, Panel):
    bl_label = "Halcyon Presets"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        hs = context.scene.halcyon
        # the device switch, first thing in render properties: choosing GPU
        # turns on every hardware-proven stage (raster, shading, post), and
        # anything a frame uses that the GPU does not reproduce falls back
        # per stage with the reason on the console
        row = layout.row(align=True)
        row.scale_y = 1.3
        row.prop(hs, 'render_device', expand=True)
        layout.separator()
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
        draw_disclaimer(layout)


def _resolution_group_menu(index, label, keys):
    """One submenu per category -- Blender menus take no arguments, so each
    category gets its own generated class."""

    def draw(self, context):
        for k in keys:
            v = RESOLUTION_PRESETS[k]
            aspect = "" if v[2] == v[3] else "  wide px" if v[2] > v[3] else "  tall px"
            self.layout.operator(
                'halcyon.set_resolution',
                text=f"{resolution_label(k)}  ({v[0]}x{v[1]}){aspect}").key = k

    return type(f'HALCYON_MT_res_group_{index}', (bpy.types.Menu,), {
        'bl_idname': f'HALCYON_MT_res_group_{index}',
        'bl_label': label,
        'draw': draw,
    })


RESOLUTION_MENUS = tuple(_resolution_group_menu(i, label, keys)
                         for i, (label, keys) in enumerate(RESOLUTION_GROUPS))


class HALCYON_OT_halo_ramp(Operator):
    """R198: give a halo material its colour ramp widget.

    A ColorRamp cannot live on a PropertyGroup, so the widget rides a
    dedicated, UNLINKED ColorRamp node stashed in the material's node
    tree -- the serializer only walks from the output, so shading
    never sees it, and the exporter samples it into a small LUT."""

    bl_idname = 'halcyon.halo_ramp'
    bl_label = "Add Halo Colour Ramp"
    bl_description = ("Add a colour ramp to this halo material -- the "
                      "gradient then follows the ramp instead of the "
                      "two colours")
    bl_options = {'REGISTER', 'UNDO'}

    remove: BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return context.material is not None

    def execute(self, context):
        mat = context.material
        if not mat.use_nodes:
            mat.use_nodes = True
        tree = mat.node_tree
        node = tree.nodes.get('__halo_ramp')
        if self.remove:
            if node is not None:
                tree.nodes.remove(node)
            return {'FINISHED'}
        if node is None:
            node = tree.nodes.new('ShaderNodeValToRGB')
            node.name = '__halo_ramp'
            node.label = "Halo Ramp"
            node.location = (-600, -600)
        return {'FINISHED'}


class HALCYON_MT_resolutions(bpy.types.Menu):
    bl_idname = 'HALCYON_MT_resolutions'
    bl_label = "Resolution Presets"

    def draw(self, context):
        for menu in RESOLUTION_MENUS:
            self.layout.menu(menu.bl_idname)


class HALCYON_PT_output(HalcyonPanel, Panel):
    """R194: the resolution presets live with Blender's own Format
    fields now -- the Output tab is where everyone looks for them."""

    bl_label = "Halcyon Output"
    bl_context = "output"

    def draw(self, context):
        layout = self.layout
        r = context.scene.render
        layout.menu('HALCYON_MT_resolutions', icon='OUTPUT')
        row = layout.row()
        row.active = False
        row.label(text=f"Current: {r.resolution_x} x {r.resolution_y}"
                       + ("" if r.pixel_aspect_x == r.pixel_aspect_y
                          else "  (shaped pixels)"))
        cam = context.scene.camera
        if cam is not None and getattr(cam.data, 'type', '') == 'PANO':
            note = layout.column()
            note.active = False
            note.label(text="Panoramic camera: pair with a Panoramas "
                            "& 360 preset", icon='INFO')


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
        if hs.aa_mode in ('EDGE', 'ADAPTIVE'):
            # both flag edge pixels off the depth buffer; neither runs
            # the downfilter (Edge tents in place, Adaptive averages
            # its own samples), so the filter rows would be dead knobs
            sub.prop(hs, 'aa_edge_threshold')
        else:
            sub.prop(hs, 'aa_filter')
            sub.prop(hs, 'aa_filter_width')
        sub = col.column()                              # R251 C127
        sub.active = hs.aa_mode in ('NONE', 'SUPERSAMPLE')
        sub.prop(hs, 'aa_sample_pattern')
        col.separator()                                 # R251 C001
        col.prop(hs, 'n64_coverage_aa')
        sub = col.column()
        sub.active = hs.n64_coverage_aa
        sub.prop(hs, 'n64_divot')
        # R251 (RAST-B): the two resolve-side period dials -- the
        # LightWave clamp works at one sample too, the Blender 2.41
        # gamma blend only with samples to blend
        col.prop(hs, 'aa_clamp_samples')
        sub = col.column()
        sub.active = hs.aa_mode not in ('NONE', 'EDGE', 'ADAPTIVE') and hs.aa_samples > 1
        sub.prop(hs, 'aa_gamma_blend')
        col.separator()
        col.prop(hs, 'stereo_mode')
        sub = col.column()
        sub.active = hs.stereo_mode != 'NONE'
        sub.prop(hs, 'stereo_eye_distance')
        sub.prop(hs, 'stereo_convergence')
        sub.prop(hs, 'stereo_parallax_layers')
        sub2 = sub.column()
        sub2.active = hs.stereo_parallax_layers
        sub2.prop(hs, 'stereo_parallax_max')
        col.separator()
        col.prop(hs, 'camera_yshear')
        col.prop(hs, 'pano_parts')
        col.separator()
        col.prop(hs, 'motion_blur')
        sub = col.column()
        sub.active = hs.motion_blur
        sub.prop(hs, 'motion_shutter')
        sub.prop(hs, 'motion_steps')
        sub.prop(hs, 'motion_blur_mode')
        if hs.motion_blur_mode == 'MAX_SLICES':
            sub.prop(hs, 'motion_samples')
            sub.prop(hs, 'motion_dither')
            sub2 = sub.column()
            sub2.active = hs.motion_dither > 0.0
            sub2.prop(hs, 'motion_dither_tile')
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
        col.prop(hs, 'vertex_quantize')                # R251 C012
        sub = col.column()
        sub.active = hs.vertex_quantize != 'NONE'
        sub.prop(hs, 'vertex_units')
        col.prop(hs, 'pixel_center')                   # R251 C084
        col.separator()
        col.prop(hs, 'depth_sort')
        sub = col.column()
        sub.active = hs.depth_sort == 'PAINTERS'
        sub.prop(hs, 'painters_key')
        if hs.painters_key == 'ORDERING_TABLE':        # R251 C004
            sub.prop(hs, 'ot_length')
            sub.prop(hs, 'ot_far')
        sub = col.column()                              # R251 C007/C026/C075
        sub.active = hs.depth_sort != 'PAINTERS'
        sub.prop(hs, 'depth_encoding')
        sub = col.column()
        sub.active = hs.depth_sort == 'PAINTERS' or \
            hs.depth_encoding in ('LINEAR', 'W_FIXED')
        sub.prop(hs, 'depth_precision')
        col.prop(hs, 'near_clip_mode')                 # R251 C027


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
        col.prop(hs, 'material_override')
        if str(getattr(hs, 'material_override', 'NONE')) != 'NONE':
            col.prop(hs, 'override_color')
        col.prop(hs, 'normal_source')
        # R251 C119 (MAT-B): the REYES grid is a per-pixel rate's dial;
        # drawn greyed at VERTEX / FACE, never hidden
        sub = col.column()
        sub.active = hs.shading_rate == 'PIXEL'
        sub.prop(hs, 'shading_rate_area')
        col.prop(hs, 'displacement_scale')
        col.separator()
        col.prop(hs, 'specular_in_gamma')
        col.prop(hs, 'specular_viewer')
        col.prop(hs, 'clamp_specular')
        col.prop(hs, 'light_clamp')
        col.prop(hs, 'crand_per_frame')


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
        col.prop(hs, 'sss', text="Subsurface Scattering")
        col.separator()
        col.prop(hs, 'max_lights')
        sub = col.column()
        sub.active = hs.max_lights > 0
        sub.prop(hs, 'light_limit_mode')
        col.prop(hs, 'light_falloff_default')
        # the repair for lamps brought in via plain File > Append:
        # Blender's own append converts 2.79 energies to its modern
        # units (every sun arrives at 1.0 W). The watch stamps the
        # classic file's OWN values back on automatically the moment
        # they arrive; the button does the same by hand for scenes
        # appended before the watch existed (or with it off)
        col.separator()
        col.prop(hs, 'auto_fix_appended_lamps',
                 text="Auto-Fix Appended Lamps")
        col.operator('halcyon.fix_appended_lamps', icon='LIGHT')


class HALCYON_PT_spot_cones(HalcyonPanel, Panel):
    bl_label = "Spot Cones"
    bl_parent_id = "HALCYON_PT_lighting"
    bl_context = "render"
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'spot_cones', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.spot_cones
        col.prop(hs, 'spot_cone_density')
        col.prop(hs, 'spot_cone_samples')
        col.prop(hs, 'spot_cone_falloff')
        col.prop(hs, 'spot_cone_reach')
        note = col.column(align=True)
        note.active = False
        note.scale_y = 0.8
        note.label(text="Each spot light's own Volumetric value", icon='INFO')
        note.label(text="decides whether it has a beam, and how strong.")
        # R222: the real volumes -- always live; a scene without volume
        # containers pays nothing, so there is no master switch to forget
        layout.separator()
        vcol = layout.column()
        vcol.prop(context.scene.halcyon, 'volume_steps')
        vcol.prop(context.scene.halcyon, 'volume_shadows')
        vnote = vcol.column(align=True)
        vnote.active = False
        vnote.scale_y = 0.8
        vnote.label(text="A mesh whose material output links a Volume",
                    icon='INFO')
        vnote.label(text="chain marches as a container inside its bound.")


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
        if hs.shadow_default == 'PLANAR':
            col.prop(hs, 'planar_plane_z')          # R251 C052
        col.prop(hs, 'shadow_map_size')
        col.prop(hs, 'shadow_map_depth')            # R251 C117
        col.prop(hs, 'shadow_bias')
        col.prop(hs, 'shadow_softness')
        col.prop(hs, 'shadow_samples')
        col.prop(hs, 'modvol_scale')                # R251 C020
        col.operator('halcyon.adopt_shadow_settings',
                     text="All Lights Use These Settings",
                     icon='FILE_REFRESH').scope = 'SCENE'


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


class HALCYON_PT_radiosity(HalcyonPanel, Panel):
    bl_label = "Radiosity"
    bl_parent_id = 'HALCYON_PT_lighting'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'radiosity', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.radiosity
        col.label(text="One bounce of colour bleed, the 1996 way",
                  icon='INFO')
        col.prop(hs, 'radiosity_samples')
        col.prop(hs, 'radiosity_distance')
        col.prop(hs, 'radiosity_intensity')
        col.prop(hs, 'radiosity_spacing')
        if hs.radiosity and hs.ambient_occlusion:
            col.label(text="Supersedes Ambient Occlusion while on",
                      icon='INFO')


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
        sub = col.column()
        sub.active = hs.raytrace and hs.ray_reflection
        sub.prop(hs, 'reflection_blur')
        sub2 = sub.column()
        sub2.active = hs.reflection_blur > 0.0
        sub2.prop(hs, 'reflection_blur_samples')
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
        # R251 texture pack: the rows in the machine's order; greying
        # follows sample_opts (core/texture.py) so a greyed dial is inert
        sub = col.column()
        sub.active = hs.tex_filter in {'BILINEAR', 'TRILINEAR'}
        sub.prop(hs, 'tex_frac_bits')                                # C080
        col.prop(hs, 'tex_mipmap')
        sub_bias = col.column()
        sub_bias.active = (hs.tex_mipmap and hs.tex_lod_source != 'GS_Q') \
            or hs.tex_filter == 'SUMMED_AREA'                        # C088 (live without mips), C022 (inert under GS_Q)
        sub_bias.prop(hs, 'tex_mip_bias')
        # R251 TEX-2: the level roads, sample_opts rules (1) and (2)
        sub = col.column()
        sub.active = hs.tex_mipmap and hs.tex_filter in {'NEAREST', 'BILINEAR', 'N64_3POINT', 'TRILINEAR'}
        sub.prop(hs, 'tex_mip_select')                               # C072
        sub.prop(hs, 'tex_lod_source')                               # C022 / C079
        sub2 = sub.column()
        sub2.active = hs.tex_lod_source == 'GS_Q'
        sub2.prop(hs, 'tex_lod_k')
        sub2.prop(hs, 'tex_lod_l')
        sub.prop(hs, 'tex_lod_sharpen')                              # C008
        col.separator()
        col.prop(hs, 'tex_perspective')
        col.separator()
        col.prop(hs, 'tex_max_size')
        col.prop(hs, 'tex_quantize')
        sub_fmt = col.column()
        sub_fmt.active = hs.tex_tmem_format == 'OFF'
        sub_fmt.prop(hs, 'tex_format')                               # C074 (greyed under TMEM, A2.1)
        col.prop(hs, 'tex_tmem_format')                              # C013
        col.prop(hs, 'tex_compress')                                 # C024
        col.prop(hs, 'tex_wrap_default')
        col.prop(hs, 'tex_clamp_mode')                               # C077
        col.prop(hs, 'tex_colorkey')                                 # C083
        sub = col.column()
        sub.active = hs.tex_colorkey
        sub.prop(hs, 'tex_colorkey_range')


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
        if hs.transparency in ('SORTED', 'ABUFFER'):
            # R251: the blend unit and the composite order (DS)
            col.prop(hs, 'blend_equation')
            col.prop(hs, 'translucent_order')
            col.prop(hs, 'translucent_depth_write')
        col.prop(hs, 'max_transparent_layers')
        col.prop(hs, 'alpha_bits')
        col.prop(hs, 'alpha_threshold')
        # R251: the framebuffer format is read in EVERY mode (the frame
        # itself is what it truncates); RGBA6 always dithers, so its
        # knob is dead there and is not drawn
        col.prop(hs, 'framebuffer')
        if hs.framebuffer != 'NONE' and not hs.framebuffer.startswith('GC'):
            row = col.row(align=True)
            row.prop(hs, 'fb_dither')
            if hs.framebuffer.startswith('VOODOO'):
                row.prop(hs, 'fb_dither_subtract')


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
        col.prop(hs, 'fog_depth')
        col.prop(hs, 'fog_color')
        col.prop(hs, 'fog_color_source')      # R251 F008 (LIGHT-A2)
        _ranged = hs.fog_mode in ('LINEAR', 'TABLE16', 'GTE_1Z')
        if _ranged:
            col.prop(hs, 'fog_start')
            col.prop(hs, 'fog_end')
        elif hs.fog_mode == 'GROUND':
            # R251 F009: POV's fog_type 2 reads Distance and the layer
            col.prop(hs, 'fog_density')
            col.prop(hs, 'fog_ground_offset')
            col.prop(hs, 'fog_ground_alt')
        else:
            col.prop(hs, 'fog_density')
        # R251: the hardware tables (F002 / F003 / F022); inert under
        # Ground Fog (a ray integral has no depth curve to fill them)
        sub = col.column()
        sub.active = hs.fog_mode != 'GROUND'
        sub.prop(hs, 'fog_table')
        if hs.fog_table == 'PVR128' and not _ranged:
            col.prop(hs, 'fog_end', text="Fog End (density register)")
        if hs.fog_table == 'DS32' and not _ranged:
            col.prop(hs, 'fog_start', text="Fog Start (table entry 0)")
            col.prop(hs, 'fog_end', text="Fog End (last table entry)")
        sub = col.column()
        sub.active = hs.fog_table == 'VOODOO64'
        sub.prop(hs, 'fog_dither')
        col.prop(hs, 'fog_vertex')
        col.prop(hs, 'fog_face')      # R251 F005 (LIGHT-A2)
        col.prop(hs, 'fog_range_adjust')    # R251 F004 (LIGHT-A2)
        col.prop(hs, 'fog_bands')
        # R251 F015: the Model 3 spotlight's share of the fog colour
        col.prop(hs, 'fog_spot')
        # R251 F006: Model 3's fogAmbient and System 22's second cz bank
        col.prop(hs, 'fog_ambient')
        sub = col.column()
        sub.active = hs.fog and hs.fog_mode in ('LINEAR', 'TABLE16', 'GTE_1Z')
        sub.prop(hs, 'fog_bank1_start')
        sub.prop(hs, 'fog_bank1_end')
        # R251 F010: POV's turbulent fog
        col.prop(hs, 'fog_turbulence')
        sub = col.column()
        sub.active = hs.fog_turbulence > 0.0
        sub.prop(hs, 'fog_turb_depth')
        col.separator()
        col.prop(hs, 'fog_height')
        sub = col.column()
        sub.active = hs.fog_height and hs.fog_mode != 'GROUND'
        sub.prop(hs, 'fog_height_top')
        sub.prop(hs, 'fog_height_falloff')


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
        # R251 C134: the optical printer's backlit mattes
        col = layout.column(heading="Matte Glow")
        col.prop(hs, 'matte_glow', text="Enable")
        mg = col.column()
        mg.active = hs.matte_glow
        mg.prop(hs, 'matte_glow_radius')
        mg.prop(hs, 'matte_glow_passes')
        mg.prop(hs, 'matte_glow_exposure')
        row = col.row()
        row.active = False
        row.label(text="Set a Glow Gel on a material to expose it",
                  icon='MATERIAL')
        col = layout.column(heading="Depth of Field")
        col.prop(hs, 'dof', text="Enable")
        sub = col.column()
        sub.active = hs.dof
        sub.prop(hs, 'dof_focus')
        sub.prop(hs, 'dof_method')
        if hs.dof_method == 'LENS_ACCUMULATE':
            sub.prop(hs, 'dof_lens_pattern')
            sub.prop(hs, 'dof_lens_samples')
            row = sub.row()
            row.active = False
            row.label(text="f-number: the camera's Depth of Field > F-Stop")
        else:
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
        # R251 (C050 / C060): the era roads that OWN the colour stage --
        # while one is on, Colour Depth and the palette controls idle
        col.prop(hs, 'attribute_cells')
        col.prop(hs, 'scanline_palette')
        _era_free = hs.attribute_cells == 'NONE' and \
            hs.scanline_palette == 'NONE'
        cd = col.row()
        cd.active = _era_free
        cd.prop(hs, 'color_depth')
        indexed = hs.color_depth in ('8', '4', 'HAM8', 'HAM6')
        sub = col.column()
        # R202: a non-adaptive palette (Custom included) forces itself
        # at ANY depth, so the controls only dim while they truly idle
        sub.active = (indexed or hs.palette_mode != 'ADAPTIVE') and _era_free
        sub.prop(hs, 'palette_mode')
        if hs.palette_mode == 'CUSTOM':
            sub.template_ID(hs, 'palette_image', open='image.open')
            note = sub.row()
            note.active = False
            if getattr(hs, 'palette_image', None) is not None:
                iw, ih = tuple(hs.palette_image.size)[:2]
                note.label(text=f"{iw}x{ih} image -- its colours become "
                                "the whole frame's palette",
                           icon='COLOR')
            else:
                note.label(text="Pick an image (Image editor > Image > "
                                "Make Palette Table builds one)",
                           icon='INFO')
        s2a = sub.column()
        # the size doubles as the Custom image's colour cap
        s2a.active = hs.palette_mode in ('ADAPTIVE', 'CUSTOM')
        s2a.prop(hs, 'palette_size')
        s2 = sub.column()
        s2.active = hs.palette_mode in ('ADAPTIVE', 'EHB')
        s2.prop(hs, 'palette_method')
        s2.prop(hs, 'palette_lock')
        if hs.palette_lock:
            s2.operator('halcyon.clear_palette_cache', icon='FILE_REFRESH')
        # R251 (C061): the register / DAC lattice; applies to fixed modes
        # too, so it draws with the palette controls, lock or not
        sub.prop(hs, 'palette_bits')
        col.separator()
        col.prop(hs, 'dither')
        s3 = col.column()
        # R251 (C050 / C060): the cell and scanline fits take an ordered
        # dither only; diffusion and noise are inert there
        _era_inert = (not _era_free) and hs.dither not in (
            'NONE', 'BAYER2', 'BAYER4', 'BAYER8', 'BAYER16', 'HALFTONE',
            'COLUMNS')
        s3.active = hs.dither != 'NONE' and not _era_inert and \
            (not _era_free or hs.color_depth not in ('CRY16', 'YJK'))
        if _era_inert:
            note = col.row()
            note.active = False
            note.label(text="Attribute cells and scanline palettes take "
                            "an ordered dither only", icon='INFO')
        s3.prop(hs, 'dither_strength')
        s3.prop(hs, 'dither_serpentine')
        if hs.dither_serpentine and hs.dither in _DIFFUSION_KERNELS:
            note = s3.row()
            note.active = False
            note.label(text="Off is ~2x faster (diagonal processing)",
                       icon='SORTTIME')
        # R251 post-signal: the machine's scan-out, on the frame it wrote
        col.separator()
        vi = col.column()
        vi.active = hs.color_depth in ('15', '16')
        vi.prop(hs, 'vi_dither_filter')
        vi.prop(hs, 'vi_gamma')
        col.prop(hs, 'copy_filter')
        col.prop(hs, 'crtc_blend')
        sub2 = col.column()
        sub2.active = hs.crtc_blend != 'NONE'
        sub2.prop(hs, 'crtc_alpha')
        sub2.prop(hs, 'crtc_bg_color')
        col.prop(hs, 'video_filter')
        sub3 = col.column()
        sub3.active = hs.video_filter != 'NONE' and hs.color_depth == '16'
        sub3.prop(hs, 'video_filter_threshold')
        if hs.color_depth in ('CRY16', 'YJK'):
            # R251 (C011 / C059): the integer encodes have no dither seat
            note = col.row()
            note.active = False
            note.label(text="This encode takes no dither", icon='INFO')


class HALCYON_PT_display(HalcyonPanel, Panel):
    bl_label = "Display"
    bl_context = "render"

    def draw(self, context):
        layout = self.layout
        view = getattr(context.scene, 'view_settings', None)
        if view is not None and getattr(view, 'view_transform', 'Raw') \
                != 'Raw':
            box = layout.box()
            box.alert = True
            box.label(text=f"Blender's view transform is "
                           f"{view.view_transform}", icon='ERROR')
            col = box.column(align=True)
            col.scale_y = 0.8
            for line in _wrap("Blender's color management regrades "
                              "Halcyon's display-referred output -- even "
                              "Standard washes it grey. This panel is the "
                              "engine's own grading; Blender's should be "
                              "Raw (the engine pins it on the next "
                              "update).", 44):
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
        # R251 (C092 / C093): the 3D Studio / Max video-out pair
        col.separator()
        col.prop(hs, 'super_black')
        sb = col.column()
        sb.active = hs.super_black
        sb.prop(hs, 'super_black_threshold')
        col.separator()
        col.prop(hs, 'video_color_check')
        vc = col.column()
        vc.active = hs.video_color_check != 'NONE'
        vc.prop(hs, 'video_system')
        vc.prop(hs, 'video_ire_limit')
        col.separator()
        col.prop(hs, 'output_scale')
        col.prop(hs, 'pixel_grid')
        col.separator()
        col.prop(hs, 'film_transparent')
        col.separator()
        col.prop(hs, 'watermark')
        if hs.watermark:
            note = col.column(align=True)
            note.active = False
            note.scale_y = 0.8
            note.label(text="%F frame  %R resolution  %V version",
                       icon='INFO')
            note.label(text="%D date  %T time  %S render time  "
                            "%B Blender")
            note.label(text="Ink: &%r &%g &%b &%y &%c &%m  "
                            "(&%w white again)")


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
        # R251 SIG-2: the cable is one or the other -- the composite rows
        # grey under the S-Video items
        col.active = hs.composite and hs.signal not in ('SVIDEO', 'SVIDEO_PAL')
        col.prop(hs, 'composite_bleed')
        col.prop(hs, 'composite_ringing')
        dc = col.column()
        dc.active = hs.pal_crawl <= 0.0      # one crawl: the PAL one wins
        dc.prop(hs, 'composite_dot_crawl')
        col.separator()
        col.prop(hs, 'interlace')
        # R251 SIG-2: the cable, the tape, the PAL receiver and the digital
        # formats' chroma -- a fresh column, never gated by Composite Video
        c2 = layout.column()
        c2.separator()
        c2.prop(hs, 'signal')
        rf = c2.column()
        rf.active = hs.signal == 'RF'
        rf.prop(hs, 'rf_bandwidth')
        rf.prop(hs, 'rf_beat')
        rf.prop(hs, 'rf_snow')
        rf.prop(hs, 'rf_ghost')
        rf.prop(hs, 'rf_ghost_delay')
        c2.separator()
        c2.prop(hs, 'tape')
        tp = c2.column()
        tp.active = hs.tape != 'NONE'
        tp.prop(hs, 'tape_generations')
        tp.prop(hs, 'tape_noise')
        tp.prop(hs, 'tape_head_switch')
        tp.prop(hs, 'tape_dropouts')
        c2.separator()
        c2.prop(hs, 'pal_decoder')
        pl = c2.column()
        pl.active = hs.pal_decoder == 'SIMPLE'
        pl.prop(hs, 'pal_phase_error')
        c2.prop(hs, 'pal_crawl')
        c2.separator()
        c2.prop(hs, 'chroma_format')
        cu = c2.column()
        cu.active = hs.chroma_format != 'NONE'
        cu.prop(hs, 'chroma_upsample')


class HALCYON_PT_film(HalcyonPanel, Panel):
    """R230: the era looks -- the cel photographed and printed."""
    bl_label = "Cel Film"
    bl_parent_id = 'HALCYON_PT_display'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        # R236: the colour process -- records, dyes, key, registration
        box = layout.box()
        box.label(text="Colour Process", icon='SEQ_CHROMA_SCOPE')
        bc = box.column()
        bc.prop(hs, 'film_process')
        sub = bc.column()
        sub.active = hs.film_process != 'NONE'
        sub.prop(hs, 'film_process_amount')
        sub.prop(hs, 'film_exposure')
        sub.prop(hs, 'film_gamma')
        sub.prop(hs, 'film_density')
        sub.prop(hs, 'film_filters')
        sub.prop(hs, 'film_dye_purity')
        sub.prop(hs, 'film_key')
        row = sub.row(align=True)
        row.prop(hs, 'film_halation')
        row.prop(hs, 'film_halation_radius')
        sub.prop(hs, 'film_register')
        col = layout.column()
        col.prop(hs, 'film_grade')
        sub = col.column()
        sub.active = hs.film_grade != 'NONE'
        sub.prop(hs, 'film_grade_amount')
        col.separator()
        col.prop(hs, 'film_softness')
        col.prop(hs, 'film_weave')
        col.prop(hs, 'film_flicker')
        col.separator()
        col.prop(hs, 'film_hold')
        col.separator()
        # R237: the print's wear -- grain, dust, hairs, scratches, cues
        box = layout.box()
        box.label(text="Print Wear", icon='MOD_NOISE')
        bc = box.column()
        bc.prop(hs, 'film_grain')
        sub = bc.column()
        sub.active = hs.film_grain > 0.0
        sub.prop(hs, 'film_grain_size')
        sub.prop(hs, 'film_grain_clump')
        sub.prop(hs, 'film_grain_chroma')
        bc.separator()
        bc.prop(hs, 'film_dust')
        sub = bc.column()
        sub.active = hs.film_dust > 0.0
        sub.prop(hs, 'film_dust_size')
        sub.prop(hs, 'film_dust_negative')
        sub.prop(hs, 'film_dust_cel')
        bc.separator()
        bc.prop(hs, 'film_hairs')
        sub = bc.column()
        sub.active = hs.film_hairs > 0.0
        row = sub.row(align=True)
        row.prop(hs, 'film_hair_length')
        row.prop(hs, 'film_hair_width')
        sub.prop(hs, 'film_hair_hold')
        bc.separator()
        bc.prop(hs, 'film_scratches')
        sub = bc.column()
        sub.active = hs.film_scratches > 0.0
        sub.prop(hs, 'film_scratch_side')
        row = sub.row(align=True)
        row.prop(hs, 'film_scratch_width')
        row.prop(hs, 'film_scratch_hold')
        bc.separator()
        bc.prop(hs, 'film_reel')
        box = layout.box()
        box.label(text="Paint", icon='COLOR')
        bc = box.column()
        bc.prop(hs, 'film_misregister')
        bc.prop(hs, 'film_bleed')
        box = layout.box()
        box.label(text="Print", icon='TEXTURE')
        bc = box.column()
        bc.prop(hs, 'film_halftone')
        sub = bc.column()
        sub.active = hs.film_halftone > 0.0
        sub.prop(hs, 'film_halftone_pitch')
        note = layout.column(align=True)
        note.active = False
        note.scale_y = 0.8
        for line in _wrap("Every stage is a pure function of the frame "
                          "number and the seed: the same frame renders "
                          "the same bits on either device. Shoot On 2 or "
                          "3 skips the held frames' renders entirely -- "
                          "render sequences from their first frame.", 46):
            note.label(text=line)


class HALCYON_PT_backgrounds(HalcyonPanel, Panel):
    """R233: the painted background road and the Fleischer setback."""
    bl_label = "Painted Backgrounds"
    bl_parent_id = 'HALCYON_PT_display'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'bg_paint')
        sub = col.column()
        sub.active = hs.bg_paint > 0.0
        sub.prop(hs, 'bg_stroke_size')
        sub.prop(hs, 'bg_stroke_length')
        sub.prop(hs, 'bg_direction')
        row = sub.row(align=True)
        row.prop(hs, 'bg_angle')
        row.prop(hs, 'bg_spread', text="Spread")
        sub.prop(hs, 'bg_bristles')
        sub.prop(hs, 'bg_variation')
        sub.prop(hs, 'bg_smooth')
        sub.prop(hs, 'bg_paper')
        box = layout.box()
        box.label(text="Setback", icon='VIEW_CAMERA')
        bc = box.column()
        bc.prop(hs, 'setback')
        sub = bc.column()
        sub.active = hs.setback > 0.0
        sub.prop(hs, 'setback_start')
        sub.prop(hs, 'setback_range')
        sub.prop(hs, 'setback_sky')
        note = layout.column(align=True)
        note.active = False
        note.scale_y = 0.8
        for line in _wrap("Materials whose Paint Mode is Background take "
                          "the road: strokes fixed on the surface, drawn "
                          "far to near at a screen-constant size, no ink "
                          "unless the material says Always. Cel materials "
                          "stay sharp over the setback.", 46):
            note.label(text=line)


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
        col.active = hs.jpeg_artifacts and not hs.mpeg1   # one codec at a time
        col.prop(hs, 'jpeg_quality')
        col.prop(hs, 'jpeg_passes')
        col.prop(hs, 'block_size')
        # R251 SIG-3: the era's other codecs, each under its own switch
        cd = layout.column(heading="MPEG-1")
        cd.prop(hs, 'mpeg1', text="Enable")
        mp = cd.column()
        mp.active = hs.mpeg1
        mp.prop(hs, 'mpeg1_qscale')
        mp.prop(hs, 'mpeg1_gop')
        cd = layout.column(heading="Smacker")
        cd.prop(hs, 'smacker', text="Enable")
        sm = cd.column()
        sm.active = hs.smacker
        sm.prop(hs, 'smacker_quality')


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
        col.prop(hs, 'orbit_scale')



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
        if hs.ground_mode not in ('MATERIAL', 'MODE7'):
            col.prop(hs, 'ground_color')
        # R203 field find: every two-colour mode owns the second
        # colour, and every mode's dials draw -- Tiles and Lava were
        # UNEDITABLE because this panel never showed their controls
        if hs.ground_mode in ('CHECKER', 'NOISE', 'TILES', 'DESERT',
                              'SNOW', 'LAVA'):
            col.prop(hs, 'ground_color2')
        if hs.ground_mode not in ('SOLID', 'MODE7'):
            col.prop(hs, 'ground_scale')
        if hs.ground_mode == 'TILES':
            col.prop(hs, 'ground_grout')
            col.prop(hs, 'ground_grout_glow')
            col.prop(hs, 'ground_tile_shade')
            note = col.row()
            note.active = False
            note.label(text="Thin grout + glow above 1 = the neon "
                            "grid floor", icon='INFO')
        elif hs.ground_mode == 'LAVA':
            # R204: embers earned their own colour
            col.prop(hs, 'ground_color3', text="Ember Colour")
            col.prop(hs, 'ground_crack_width')
            col.prop(hs, 'ground_glow')
            col.prop(hs, 'ground_pulse')
        elif hs.ground_mode == 'SNOW':
            # R204: the glints earned a colour and a strength
            col.prop(hs, 'ground_color3', text="Glint Colour")
            col.prop(hs, 'ground_sparkle')
        elif hs.ground_mode == 'DESERT':
            col.prop(hs, 'ground_ridge')
        elif hs.ground_mode == 'MATERIAL':
            col.template_ID(hs, 'ground_material')
            note = col.row()
            note.active = False
            note.label(text="The material's node graph paints the "
                            "plane to the horizon", icon='INFO')
        elif hs.ground_mode == 'OCEAN':
            col.prop(hs, 'ocean_choppiness')
            col.prop(hs, 'ocean_speed')
        elif hs.ground_mode == 'MODE7':
            # R251 C048: the map, its texel size and the M7SEL over-map
            # rule; the floor is unlit and unfaded, so those dials stay off
            col.template_ID(hs, 'ground_image', open='image.open')
            col.prop(hs, 'mode7_texel_size')
            col.prop(hs, 'mode7_over')
        col.separator()
        # R204 field find: the floors ignored every lamp in the scene.
        # This dial hands them to the lighting; OCEAN keeps its own
        # sun-and-sky model and doesn't need it
        if hs.ground_mode not in ('OCEAN', 'MODE7'):
            col.prop(hs, 'ground_lighting')
        if hs.ground_mode != 'MODE7':
            col.prop(hs, 'ground_fade')
        if not hs.ground_plane:
            note = layout.column(align=True)
            note.active = False
            note.scale_y = 0.8
            for line in _wrap("This is a world property, not an object -- there "
                              "is nothing to add to the scene. Tick the box in "
                              "this panel's header.", 46):
                note.label(text=line)

        if hs.ground_plane and hs.ground_mode == 'OCEAN':
            layout.separator()
            # Bryce kept its waters in the Materials Library, not the Sky
            # Lab, so this is its own library and picking a sky never
            # touches it
            lib = layout.box()
            lib.label(text="Water Presets", icon='MOD_FLUIDSIM')
            lib.prop(hs, 'water_preset', text="")
            lib.operator('halcyon.water_preset', text="Apply Preset",
                         icon='CHECKMARK')
            row = lib.row(align=True)
            op = row.operator('halcyon.water_save', text="Save As...",
                              icon='FILE_TICK')
            op.to_library = False
            op = row.operator('halcyon.water_save', text="Add to Library",
                              icon='ADD')
            op.to_library = True
            lib.operator('halcyon.water_load', text="Import Preset...",
                         icon='IMPORT')

            box = layout.column()
            box.label(text="Water", icon='MOD_OCEAN')
            box.prop(hs, 'ocean_deep')
            box.prop(hs, 'ocean_shallow')
            box.prop(hs, 'ocean_transparency')
            box.separator()
            box.prop(hs, 'ocean_choppiness', text="Wave Height")
            box.prop(hs, 'ocean_wave_scale')
            box.prop(hs, 'ocean_detail')
            box.prop(hs, 'ocean_sparkle')
            box.prop(hs, 'ocean_horizon_smooth')
            box.prop(hs, 'ocean_wind_angle')
            box.prop(hs, 'ocean_spread')
            box.prop(hs, 'ocean_speed')
            box.separator()
            box.prop(hs, 'ocean_glitter')
            sub = box.column()
            sub.active = hs.ocean_glitter > 0.0
            sub.prop(hs, 'ocean_glitter_size')
            box.separator()
            box.prop(hs, 'ocean_foam')
            sub = box.column()
            sub.active = hs.ocean_foam > 0.0
            sub.prop(hs, 'ocean_foam_color')


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
            col.prop(hs, 'moon_softness')
        else:
            col.prop(hs, 'sun_color')
            if hs.mode == 'BRYCE':
                col.prop(hs, 'sun_glow')
                col.prop(hs, 'sun_glow_color')
                col.prop(hs, 'sun_corona')
        if hs.mode == 'BRYCE':
            col.separator()
            col.prop(hs, 'shadow_color')
            col.prop(hs, 'shadow_intensity')
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
        col.prop(hs, 'haze_base_height')
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
        sub.prop(hs, 'fog_base_height')
        sub.prop(hs, 'fog_sun_tint')
        sub.prop(hs, 'fog_blend_sky')
        col.separator()
        col.prop(hs, 'color_perspective')
        col.prop(hs, 'volumetric_world')


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
        col.prop(hs, 'cloud_frequency')
        col.prop(hs, 'cloud_amplitude')
        col.prop(hs, 'cloud_turbulence')
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
        col.separator()
        col.prop(hs, 'spherical_clouds')
        col.prop(hs, 'link_clouds_to_view')
        col.prop(hs, 'fixed_cloud_plane')


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
        col.prop(hs, 'stratus_frequency')
        col.prop(hs, 'stratus_amplitude')
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
        sub.prop(hs, 'star_size')
        sub.prop(hs, 'old_stars')
        col.separator()
        col.prop(hs, 'comets')
        sub = col.column()
        sub.active = hs.comets > 0.0
        sub.prop(hs, 'comet_count')
        sub.prop(hs, 'comet_speed')
        sub.prop(hs, 'comet_length')
        sub.prop(hs, 'comet_width')
        sub.prop(hs, 'comet_tail_sun')
        sub.prop(hs, 'comet_color')


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
    if _uses_nodes(mat) and mat.node_tree:
        for node in mat.node_tree.nodes:
            if node.bl_idname == 'HALCYON_ShaderNode':
                return node.model, True
            if node.bl_idname == 'HALCYON_ConsoleShaderNode':
                # R252: the Console Emulation Shader names its machine
                from .core.console import label_of
                try:
                    props = {k: getattr(node, k) for k in
                             ('console', 'gc_type', 'm2_type', 'm3_type',
                              'ds_type', 'ps1_type', 'ps2_type', 'psp_type',
                              'sat_type', 'n64_type', 's22_type', 'dc_type',
                              'pc_type', 'pcx_base', 'md_type', 'sfx_type',
                              'jag_type', 'tdo_type', 'rw_type')
                             if hasattr(node, k)}
                    return label_of(props), True
                except Exception:                               # noqa: BLE001
                    return 'Console', True
            if node.bl_idname == 'HALCYON_BIMaterialNode':
                return ('CONSTANT' if node.shadeless else
                        f'BI {node.diff_shader}/{node.spec_shader}'), True
            if node.bl_idname in ('HALCYON_MaxStandardNode',
                                  'HALCYON_MaxRaytraceNode'):
                return f'3ds Max {node.shader_type}', True
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


class HALCYON_OT_adopt_shadow_settings(Operator):
    """Point lights back at the render settings' shadow quality.

    A light whose own Map Size or Bias is set overrides the render
    sliders -- and every light saved before 1.30.1 carries the old
    per-light defaults (512 / 0.02) explicitly, which is why raising the
    render setting's Shadow Map Size changed nothing in older scenes.
    This clears the per-light values to 0 = inherit.
    """

    bl_idname = 'halcyon.adopt_shadow_settings'
    bl_label = "Use Render Shadow Settings"
    bl_options = {'REGISTER', 'UNDO'}

    scope: EnumProperty(name="Scope", items=(
        ('ACTIVE', "Active Light", ""),
        ('SCENE', "Every Light in the Scene", "")), default='SCENE')

    def execute(self, context):
        if self.scope == 'ACTIVE':
            lights = [context.light] if getattr(context, 'light', None) \
                else []
        else:
            lights = [ob.data for ob in context.scene.objects
                      if getattr(ob, 'type', '') == 'LIGHT'
                      and getattr(ob, 'data', None) is not None]
        n = 0
        for light in lights:
            hs = getattr(light, 'halcyon', None)
            if hs is None:
                continue
            if hs.shadow_map_size != 0 or hs.shadow_bias != 0.0:
                hs.shadow_map_size = 0
                hs.shadow_bias = 0.0
                n += 1
        self.report({'INFO'},
                    f"{n} light(s) now inherit the render shadow settings"
                    if n else "Every light already inherits them")
        return {'FINISHED'}


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
        from .core.render import material_model, material_wire_size
        print("=" * 70)
        print("HALCYON DIAGNOSTICS")
        print("=" * 70)
        print(f"version        : {'.'.join(str(v) for v in _version())}  "
              f"({__package__})")
        mesh = exported.mesh
        print(f"triangles      : {0 if mesh is None or mesh.tris is None else len(mesh.tris)}")
        print(f"objects        : {len(exported.objects)}")
        print(f"lights         : {[(l.type, round(l.energy, 2)) for l in exported.lights]}")
        print(f"images         : {list(getattr(exported, 'images', {}))}")
        print(f"world mode     : {exported.world.mode}  graph={'yes' if exported.world.graph else 'no'}")
        print(f"render pass    : {st.debug_pass}"
              + ("" if st.debug_pass == 'BEAUTY'
                 else "   (the beauty image is replaced by this)"))
        wire = [(i, m.name) for i, m in enumerate(exported.materials)
                if material_model(m, st) == 'WIREFRAME']
        print(f"wire materials : {wire if wire else 'none resolve to WIREFRAME'}"
              f"   mode={st.wire_mode} angle={st.wire_angle}")
        for i, name in wire:
            print(f"                 [{i}] {name!r} wire size "
                  f"{material_wire_size(exported.materials[i])}")
        tri_count = 0 if mesh is None or mesh.tris is None else len(mesh.tris)
        px = max(st.resolution_x * st.resolution_y, 1)
        if wire and tri_count:
            print(f"                 ~{px * 0.5 / tri_count:.1f} pixels per "
                  f"triangle -- under about 5, All Edges fills solid")
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
        col.prop(hs, 'viewport_gpu')
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
        # the device switch itself lives at the top of render properties
        # now; this box keeps the per-stage toggles and the capability table
        if str(hs.render_device).upper() == 'GPU':
            from .gpu import capability as _cap
            from .gpu import device as _dev
            gcol = box.column()
            gcol.use_property_split = True
            gcol.prop(hs, 'gpu_shading')
            gcol.prop(hs, 'gpu_raster')
            gcol.prop(hs, 'gpu_hold_context')
            gcol.prop(hs, 'gpu_scissor')
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

            slot = slots[ob.active_material_index] if len(slots) else None
            if slot is not None:
                layout.template_ID(slot, 'material', new='halcyon.material_new')
            else:
                layout.operator('object.material_slot_add',
                                text="New Material Slot", icon='ADD')

            # the edit-mode trio: put THIS material on the selected faces,
            # or grab every face already wearing it
            if slot is not None and getattr(ob, 'mode', '') == 'EDIT':
                sub = layout.row(align=True)
                sub.operator('object.material_slot_assign', text="Assign")
                sub.operator('object.material_slot_select', text="Select")
                sub.operator('object.material_slot_deselect', text="Deselect")

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

        has_master = bool(_uses_nodes(mat) and mat.node_tree and any(
            n.bl_idname in ('HALCYON_ShaderNode', 'HALCYON_BIMaterialNode',
                            'HALCYON_MaxStandardNode', 'HALCYON_MaxRaytraceNode',
                            'HALCYON_ConsoleShaderNode')
            for n in mat.node_tree.nodes))
        box = layout.box()
        row = box.row()
        row.label(text="Halcyon Shader" if has_master else "Convert Material",
                  icon='CHECKMARK' if has_master else 'NODE_MATERIAL')
        hal = context.scene.halcyon
        det = box.column(align=True)
        det.prop(hal, 'convert_detection', text="Detection")
        if hal.convert_detection == 'SET':
            det.prop(hal, 'convert_model', text="")
        _model = 'AUTO' if hal.convert_detection == 'AUTO' \
            else hal.convert_model
        col = box.column(align=True)
        op = col.operator('halcyon.convert_materials', text="Convert This Material",
                          icon='MATERIAL')
        op.scope = 'ACTIVE'
        op.force = has_master
        op.model = _model
        op = col.operator('halcyon.convert_materials',
                          text="Convert Selected Objects", icon='RESTRICT_SELECT_OFF')
        op.scope = 'SELECTED'
        op.model = _model
        op = col.operator('halcyon.convert_materials', text="Convert Whole Scene",
                          icon='SCENE_DATA')
        op.scope = 'SCENE'
        op.model = _model
        box.label(text="Textures are relinked, not discarded", icon='INFO')

        col = box.column(align=True)
        col.label(text="To Blender Internal:", icon='SHADING_SOLID')
        op = col.operator('halcyon.convert_to_bi', text="This Material",
                          icon='MATERIAL')
        op.scope = 'ACTIVE'
        op = col.operator('halcyon.convert_to_bi', text="Selected Objects",
                          icon='RESTRICT_SELECT_OFF')
        op.scope = 'SELECTED'
        op = col.operator('halcyon.convert_to_bi', text="Whole Scene",
                          icon='SCENE_DATA')
        op.scope = 'SCENE'

        # R202: the template shelf moved to the Shader Editor's Add
        # menu (Add > Pre-Made > Bryce / Halcyon) -- a note points the
        # way for anyone who reaches for it here
        note = layout.row()
        note.active = False
        note.label(text="Templates: Shader Editor > Add > Pre-Made",
                   icon='PRESET')

        box = layout.box()
        box.label(text="Bake", icon='RENDER_STILL')
        col = box.column(align=True)
        op = col.operator('halcyon.bake_lightmap', text="Bake Lightmap",
                          icon='LIGHT_SUN')
        op.mode = 'COMBINED'
        op = col.operator('halcyon.bake_lightmap',
                          text="Bake Ambient Occlusion", icon='SHADING_RENDERED')
        op.mode = 'AO'
        box.label(text="Into the active UV layout, as a new image",
                  icon='INFO')

        layout.separator()
        # R208: hair geometry -- the strand convention switch
        layout.prop(hs, 'strand')
        row = layout.row(align=True)
        row.prop(hs, 'alpha_mode', text="Alpha")
        if hs.alpha_mode in ('CLIP', 'CLIP_BLEND'):
            row.prop(hs, 'alpha_clip', text="")
        # R251: the per-material blend equation (PS1 glow beside glass)
        row = layout.row(align=True)
        row.prop(hs, 'blend_mode', text="Blend")
        if hs.blend_mode == 'THIN_WALL':
            # R251 C095: Max's Thickness Offset rides the Blend row
            row.prop(hs, 'thin_wall_offset', text="")
        if hs.blend_mode == 'IMAGINE_FOG':
            # R251 C101: Imagine's Fog Length rides the Blend row
            row.prop(hs, 'fog_length', text="")
        # R251 C126: Blender 2.4x's Zoffs / ZInvert
        row = layout.row(align=True)
        row.prop(hs, 'z_offset')
        row.prop(hs, 'z_invert')
        # R219: shadow flags apply to EVERY material, node-shaded or
        # overridden -- they sat in the override column below, greyed
        # out for the node materials that needed them most (fur shells)
        row = layout.row(align=True)
        row.prop(hs, 'cast_shadow')
        row.prop(hs, 'receive_shadow')
        # R251 C134: the gel of the Tron printer's matte -- a flag for
        # every material, node-shaded or overridden
        layout.prop(hs, 'glow_gel')
        # R251 C020/C036: the material as an authored shadow volume
        layout.prop(hs, 'volume_role')
        row = layout.row(align=True)
        row.prop(hs, 'polygon_id')
        if hs.volume_role == 'DS_SHADOW':
            row.prop(hs, 'shadow_alpha')
        # R220: this material's say over the cartoon outline pass
        # R233: the cel or the painting
        row = layout.row(align=True)
        row.prop(hs, 'paint_mode', text="Paint Mode")
        row = layout.row(align=True)
        row.prop(hs, 'ink_mode', text="Ink")
        if hs.ink_mode != 'OFF':
            row = layout.row(align=True)
            row.prop(hs, 'ink_use_color', text="")
            sub = row.row(align=True)
            sub.active = hs.ink_use_color
            sub.prop(hs, 'ink_color', text="")
            row.prop(hs, 'ink_width')
            # R239: the Guilty Gear vertex-colour line control
            row = layout.row(align=True)
            row.prop(hs, 'ink_vc', text="Line Control")
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
        col.prop(hs, 'wire')
        if hs.wire or hs.model == 'WIREFRAME':
            col.prop(hs, 'wire_size')

        # ---- Halo: BI's other material type, never gated by Override
        layout.separator()
        hcol = layout.column()
        hcol.prop(hs, 'halo')
        if hs.halo:
            sub = hcol.column()
            sub.prop(hs, 'halo_shape')
            if hs.halo_shape == 'IMAGE':
                sub.template_ID(hs, 'halo_image', open='image.open')
            row = sub.row(align=True)
            row.prop(hs, 'halo_aspect')
            row.prop(hs, 'halo_rotation')
            sub.prop(hs, 'halo_color')
            sub.prop(hs, 'halo_gradient')
            if hs.halo_gradient:
                row = sub.row(align=True)
                row.prop(hs, 'halo_gradient_type', text="")
                row.prop(hs, 'halo_gradient_noise')
                if hs.halo_gradient_noise > 0.0:
                    sub.prop(hs, 'halo_grad_noise_speed')
                ramp_node = None
                if mat is not None and mat.use_nodes and mat.node_tree:
                    ramp_node = mat.node_tree.nodes.get('__halo_ramp')
                if ramp_node is not None:
                    sub.template_color_ramp(ramp_node, 'color_ramp',
                                            expand=True)
                    op = sub.operator('halcyon.halo_ramp',
                                      text="Remove Ramp (use the two "
                                           "colours)", icon='X')
                    op.remove = True
                else:
                    sub.prop(hs, 'halo_color2')
                    op = sub.operator('halcyon.halo_ramp',
                                      text="Use a Colour Ramp instead",
                                      icon='COLOR')
                    op.remove = False
            row = sub.row(align=True)
            row.prop(hs, 'halo_noise')
            row.prop(hs, 'halo_noise_scale')
            if hs.halo_noise > 0.0:
                sub.prop(hs, 'halo_noise_speed')
            row = sub.row(align=True)
            row.prop(hs, 'halo_bolts')
            row.prop(hs, 'halo_bolt_width')
            if hs.halo_bolts:
                row = sub.row(align=True)
                row.prop(hs, 'halo_bolt_color', text="")
                row.prop(hs, 'halo_bolt_speed')
            row = sub.row(align=True)
            row.prop(hs, 'halo_rays')
            row.prop(hs, 'halo_ray_sharp')
            if hs.halo_rays:
                sub.prop(hs, 'halo_ray_color', text="Ray Colour")
            sub.prop(hs, 'halo_alpha')
            sub.prop(hs, 'halo_size')
            sub.prop(hs, 'halo_hardness')
            sub.prop(hs, 'halo_add')
            sub.prop(hs, 'halo_seed')
            row = sub.row(align=True)
            row.prop(hs, 'halo_rand_hue', text="Hue")
            row.prop(hs, 'halo_rand_sat', text="Sat")
            row.prop(hs, 'halo_rand_val', text="Val")
            row = sub.row(align=True)
            row.prop(hs, 'halo_hue_shift', text="Hue Shift")
            row.prop(hs, 'halo_sat_shift', text="Sat")
            row.prop(hs, 'halo_val_shift', text="Val")
            row = sub.row(align=True)
            row.prop(hs, 'halo_pulse')
            row.prop(hs, 'halo_flicker')
            if hs.halo_pulse > 0.0 or hs.halo_flicker > 0.0:
                row = sub.row(align=True)
                psub = row.row()
                psub.active = hs.halo_pulse > 0.0
                psub.prop(hs, 'halo_pulse_speed')
                fsub = row.row()
                fsub.active = hs.halo_flicker > 0.0
                fsub.prop(hs, 'halo_flicker_speed')
            row = sub.row(align=True)
            row.prop(hs, 'halo_spin')
            row.prop(hs, 'halo_anim_speed')
            row = sub.row(align=True)
            row.prop(hs, 'halo_rings')
            rsub = row.row()
            rsub.active = hs.halo_rings
            rsub.prop(hs, 'halo_ring_count', text="")
            if hs.halo_rings:
                sub.prop(hs, 'halo_ring_color')
                sub.prop(hs, 'halo_ring_width')
                sub.prop(hs, 'halo_rings_even')
            row = sub.row(align=True)
            row.prop(hs, 'halo_lines')
            lsub = row.row()
            lsub.active = hs.halo_lines
            lsub.prop(hs, 'halo_line_count', text="")
            if hs.halo_lines:
                sub.prop(hs, 'halo_line_color')
                sub.prop(hs, 'halo_line_width')
            row = sub.row(align=True)
            row.prop(hs, 'halo_star')
            ssub = row.row()
            ssub.active = hs.halo_star
            ssub.prop(hs, 'halo_star_tips', text="")
            row = sub.row(align=True)
            row.prop(hs, 'halo_xalpha')
            row.prop(hs, 'halo_soft')
            row = sub.row(align=True)
            row.prop(hs, 'halo_shaded')
            row.prop(hs, 'halo_puno')
            note = sub.row()
            note.active = False
            note.label(text="The mesh's vertices render as glows; "
                            "its faces do not draw", icon='INFO')


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
        if light.type == 'AREA':
            # no decay menu for an area lamp: 2.79's lamp_get_visibility
            # skipped its falloff switch for LA_AREA, and so does the
            # engine -- distance speaks only through the form factor.
            # The BI pair instead: Distance (la->dist, the dist^2/A
            # normalisation) and Gamma (la->k, its pow shaping)
            col.prop(hs, 'decay_end', text="Distance")
            col.prop(hs, 'area_gamma')
        else:
            col.prop(hs, 'decay')
            if hs.decay == 'CUSTOM':
                col.prop(hs, 'decay_start')
                col.prop(hs, 'decay_end')
            # R251 F013: the POV fade distance / GX ref_dist is Falloff
            # End; GL's kl/kq are the Lin/Quad sliders (BI_SLIDERS reads
            # them too and drew them nowhere)
            if hs.decay in ('POV_FADE_LINEAR', 'POV_FADE_SQUARE') or \
                    hs.decay.startswith('GX_'):
                col.prop(hs, 'decay_end')
            if hs.decay in ('GL_3TERM', 'BI_SLIDERS'):
                col.prop(hs, 'decay_ld1')
                col.prop(hs, 'decay_ld2')
            if hs.decay.startswith('GX_'):
                col.prop(hs, 'gx_ref_brite')
        if light.type == 'SPOT':
            col.prop(hs, 'hotspot')
            # R251 F012: the cone law; the exponent only where a law
            # reads it
            col.prop(hs, 'spot_law')
            if hs.spot_law in ('GL11', 'POV'):
                col.prop(hs, 'spot_exponent')
            # R251 F015: the Model 3 screen spotlight
            col.prop(hs, 'screen_spot')
        col.separator()
        col.prop(hs, 'shadow')
        sub = col.column()
        sub.active = hs.shadow != 'NONE'
        sub.prop(hs, 'shadow_map_size')
        sub.prop(hs, 'shadow_bias')
        sub.prop(hs, 'shadow_map_depth')            # R251 C117
        if hs.shadow_map_size > 0 or hs.shadow_bias > 0.0:
            note = sub.row()
            note.active = False
            note.label(text="Overrides the render setting; 0 = inherit",
                       icon='INFO')
            sub.operator('halcyon.adopt_shadow_settings',
                         text="Use Render Shadow Settings",
                         icon='FILE_REFRESH').scope = 'ACTIVE'
        sub.prop(hs, 'shadow_softness')
        sub.prop(hs, 'shadow_samples')
        sub.prop(hs, 'shadow_density')
        sub.prop(hs, 'shadow_color')
        # R219: every lamp kind projects its cookie -- Spot through the
        # cone, Sun tiled, Point wrapped around, Area off its face
        col.separator()
        col.template_ID(hs, 'cookie', open='image.open')
        sub = col.column()
        sub.active = hs.cookie is not None
        sub.prop(hs, 'cookie_strength')
        if light.type == 'SUN':
            sub.prop(hs, 'cookie_scale')
        sub.prop(hs, 'cookie_extend')
        sub.prop(hs, 'cookie_filter')
        if hs.cookie is not None:
            note = sub.row()
            note.active = False
            note.label(text="The lamp's Angle/Radius softens the "
                            "projection", icon='INFO')
        col.separator()
        row = col.row(align=True)
        row.prop(hs, 'diffuse_only')
        row.prop(hs, 'specular_only')
        col.prop(hs, 'negative')
        col.prop(hs, 'only_shadow')
        col.prop(hs, 'ambient_only')
        if light.type == 'SUN':
            col.prop(hs, 'hemi')
        col.prop(hs, 'volumetric')
        if light.type in ('SPOT', 'POINT', 'AREA'):
            # the in-air beam kinds; a SUN's volumetric drives the
            # screen-space shafts instead (no apex to march from)
            sub = col.column()
            sub.active = hs.volumetric > 0.0
            sub.prop(hs, 'volumetric_occlusion')
        col.separator()
        col.prop(hs, 'flare')
        sub = col.column()
        sub.active = hs.flare > 0.0
        sub.prop(hs, 'flare_scale')
        row = sub.row(align=True)
        row.prop(hs, 'flare_streaks')
        row.prop(hs, 'flare_rings')
        row.prop(hs, 'flare_ghosts')
        if hs.flare > 0.0:
            note = sub.column()
            note.active = False
            note.label(text="Draws where THIS LAMP is visible in "
                            "frame, and fades", icon='INFO')
            note.label(text="as it hides. A Sun flares at its spot in "
                            "the sky.", icon='BLANK1')
            # R195: the live answer -- is the flare's anchor in the
            # render camera's frame RIGHT NOW? The field set the dial
            # and saw nothing; this line says why before a render does
            status = None
            try:
                from bpy_extras import object_utils as _ou
                from mathutils import Vector as _V
                cam = context.scene.camera
                ob = context.object
                if cam is not None and ob is not None:
                    if light.type == 'SUN':
                        d = (ob.matrix_world.to_3x3()
                             @ _V((0.0, 0.0, -1.0))).normalized()
                        pt = cam.matrix_world.translation - d * 1000.0
                    else:
                        pt = ob.matrix_world.translation
                    co = _ou.world_to_camera_view(context.scene, cam, pt)
                    if co.z <= 0.0:
                        status = ("Anchor is BEHIND the render camera "
                                  "-- no flare", 'ERROR')
                    elif -0.2 <= co.x <= 1.2 and -0.2 <= co.y <= 1.2:
                        status = ("Anchor is in the render camera's "
                                  "frame", 'CHECKMARK')
                    else:
                        status = ("Anchor is OUTSIDE the render "
                                  "camera's frame -- no flare", 'ERROR')
            except Exception:                                   # noqa: BLE001
                status = None
            if status:
                srow = sub.row()
                srow.label(text=status[0], icon=status[1])
                if light.type == 'SUN' and status[1] == 'ERROR':
                    hint = sub.row()
                    hint.active = False
                    hint.label(text="Tilt the sun so it shines TOWARD "
                                    "the camera's view", icon='BLANK1')
        if light.type in ('SPOT', 'SUN'):
            col.separator()
            col.prop(hs, 'caustics')
            sub = col.column()
            sub.active = hs.caustics > 0.0
            sub.prop(hs, 'caustics_scale')
            sub.prop(hs, 'caustics_speed')
        col.separator()
        col.prop(hs, 'exclude_collection')
        sub = col.column()
        sub.active = hs.exclude_collection is not None
        sub.prop(hs, 'exclude_mode')


class HALCYON_PT_wireframe(HalcyonPanel, Panel):
    """The wireframe controls, which had no panel at all until now."""

    bl_label = "Wireframe"
    bl_context = "render"
    bl_parent_id = 'HALCYON_PT_shading'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.prop(hs, 'wire_mode')
        sub = col.column()
        sub.active = hs.wire_mode in ('CREASE', 'BEAM')
        sub.prop(hs, 'wire_angle')
        # R251 (RAST-B): C063 Elite's dot distance, C055 the vector beam
        sub = col.column()
        sub.active = hs.wire_mode == 'ELITE'
        sub.prop(hs, 'wire_dot_distance')
        sub = col.column()
        sub.active = hs.wire_mode == 'BEAM'
        sub.prop(hs, 'beam_machine')
        sub.prop(hs, 'beam_sigma')
        col.separator()
        col.prop(hs, 'render_wire')
        sub = col.column()
        sub.active = hs.render_wire
        sub.prop(hs, 'wire_width')
        sub.prop(hs, 'wire_color')
        note = layout.column(align=True)
        note.active = False
        note.scale_y = 0.8
        for line in _wrap("All Edges draws every triangle edge, which is what "
                          "these renderers did -- and once triangles are a "
                          "couple of pixels across, every pixel is within a "
                          "pixel of an edge and the surface fills in solid. "
                          "That is arithmetic, not a setting. Creases & "
                          "Silhouette is the way out; each material's own Wire "
                          "Size sets the width.", 46):
            note.label(text=line)


class HALCYON_PT_outline(HalcyonPanel, Panel):
    bl_label = "Cartoon Outlines"
    bl_context = "render"
    bl_parent_id = 'HALCYON_PT_shading'
    bl_options = {'DEFAULT_CLOSED'}

    def draw_header(self, context):
        self.layout.prop(context.scene.halcyon, 'outline', text="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column()
        col.active = hs.outline
        col.prop(hs, 'outline_color')
        col.prop(hs, 'outline_width')
        col.prop(hs, 'outline_opacity')
        col.separator()
        col.prop(hs, 'outline_objects')
        col.prop(hs, 'outline_materials')
        col.prop(hs, 'outline_depth')
        sub = col.column()
        sub.active = hs.outline and hs.outline_depth
        sub.prop(hs, 'outline_depth_threshold')
        col.prop(hs, 'outline_normals')
        sub = col.column()
        sub.active = hs.outline and hs.outline_normals
        sub.prop(hs, 'outline_normal_angle')
        # R220: the artist's own marked edges as interior ink
        col.prop(hs, 'outline_marked')
        # R234: the inker's line -- three more sources
        row = col.row(align=True)
        row.prop(hs, 'outline_form')
        sub = row.row(align=True)
        sub.active = hs.outline and hs.outline_form
        sub.prop(hs, 'outline_form_threshold', text="")
        row = col.row(align=True)
        row.prop(hs, 'outline_shadow')
        sub = row.row(align=True)
        sub.active = hs.outline and (hs.outline_shadow or hs.ink_isophote > 0)
        sub.prop(hs, 'outline_shadow_level', text="")
        row = col.row(align=True)
        row.prop(hs, 'outline_tone')
        sub = row.row(align=True)
        sub.active = hs.outline and hs.outline_tone
        sub.prop(hs, 'outline_tone_threshold', text="")
        col.prop(hs, 'outline_over_sky')
        # R227: the ink style pack -- the box every dial of the
        # 40s-brush / 80s-trace contrast lives in
        box = layout.box()
        box.active = hs.outline
        box.label(text="Line Style", icon='RNDCURVE')
        bc = box.column()
        bc.prop(hs, 'ink_style')
        if hs.ink_style == 'PENCIL':
            row = bc.row(align=True)
            row.prop(hs, 'ink_pencil_strokes')
            row.prop(hs, 'ink_pencil_spread')
        bc.prop(hs, 'ink_reference_height')
        bc.prop(hs, 'ink_taper')
        bc.prop(hs, 'ink_interior_scale')
        bc.prop(hs, 'ink_shadow_side')
        row = bc.row(align=True)
        row.prop(hs, 'ink_weight_noise')
        row.prop(hs, 'ink_weight_scale')
        row = bc.row(align=True)
        row.prop(hs, 'ink_boil')
        row.prop(hs, 'ink_boil_fps')
        row.prop(hs, 'ink_boil_scale')
        bc.prop(hs, 'ink_grain')
        # R231: the drawn line -- ends, roughness, drift, gaps, texture
        row = bc.row(align=True)
        row.prop(hs, 'ink_end_taper')
        row.prop(hs, 'ink_end_length')
        row = bc.row(align=True)
        row.prop(hs, 'ink_roughness')
        row.prop(hs, 'ink_roughness_scale')
        row = bc.row(align=True)
        row.prop(hs, 'ink_drift')
        row.prop(hs, 'ink_gaps')
        row = bc.row(align=True)
        row.prop(hs, 'ink_texture', text="")
        if hs.ink_texture != 'SOLID':
            row.prop(hs, 'ink_texture_amount')
        # R234: the inker's line -- the light weight, the stroke road,
        # the anchor
        row = bc.row(align=True)
        row.prop(hs, 'ink_isophote')
        row.prop(hs, 'ink_isophote_range')
        row = bc.row(align=True)
        row.prop(hs, 'ink_smooth')
        row.prop(hs, 'ink_pressure')
        row.prop(hs, 'ink_overshoot')
        bc.prop(hs, 'ink_anchor')
        bc.prop(hs, 'ink_color_mode')
        if hs.ink_color_mode == 'FILL':
            bc.prop(hs, 'ink_fill_darken')
        elif hs.ink_color_mode == 'GRADIENT':
            bc.prop(hs, 'ink_color2')
            bc.prop(hs, 'ink_gradient')
        note = layout.column(align=True)
        note.active = False
        note.scale_y = 0.8
        for line in _wrap("Ink is drawn at the internal resolution: with "
                          "Supersample on, the line anti-aliases on the "
                          "way down. Width is internal pixels -- raise it "
                          "under heavy AA, or set True To Height. Any "
                          "Line Style dial off its default draws from a "
                          "distance field: anti-aliased, per-pixel "
                          "width; the defaults are the plain mask line.",
                          46):
            note.label(text=line)


# ------------------------------------------------------------- sky presets


def _sky_library_dir():
    from .properties import sky_library_dir
    return sky_library_dir()


def _user_skies():
    from .properties import user_skies
    return user_skies()


def sky_preset_items(self=None, context=None):
    from .properties import sky_preset_items as items
    return items(self, context)


class HALCYON_OT_sky_preset(Operator):
    """Load the selected sky onto this world"""

    bl_idname = 'halcyon.sky_preset'
    bl_label = "Apply Preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: StringProperty(
        name="Sky", default='',
        description="Leave empty to use whatever the World panel has selected")
    reset: BoolProperty(
        name="Reset First", default=True,
        description="Return every sky setting to its default before applying, "
                    "so nothing carries over from the last one")

    @classmethod
    def poll(cls, context):
        return context.world is not None

    def execute(self, context):
        from .presets import skies as SK
        hs = context.world.halcyon
        key = self.preset or getattr(hs, 'sky_preset', '') or ''
        if not key:
            self.report({'WARNING'}, "No sky selected")
            return {'CANCELLED'}
        if key.startswith('FILE:'):
            try:
                with open(key[5:], 'r', encoding='utf-8') as fh:
                    ok, msg = SK.loads(hs, fh.read(), reset=self.reset)
            except OSError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
        else:
            ok, msg = SK.apply_sky(hs, key, reset=self.reset)
        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        _poke_updated(context, context.world, context.scene)
        self.report({'INFO'}, f"Sky: {msg}")
        return {'FINISHED'}


class HALCYON_OT_sky_save(Operator):
    """Save this world's sky to a .halsky file"""

    bl_idname = 'halcyon.sky_save'
    bl_label = "Save Sky"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.halsky", options={'HIDDEN'})
    to_library: BoolProperty(
        name="Add to Library", default=False,
        description="Save into Halcyon's own folder so it appears in the sky "
                    "list next to the built-in ones")
    sky_name: StringProperty(name="Name", default="My Sky")

    @classmethod
    def poll(cls, context):
        return context.world is not None

    def invoke(self, context, event):
        import os
        if self.to_library:
            return self.execute(context)
        if not self.filepath:
            self.filepath = (self.sky_name or 'sky') + '.halsky'
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import os
        from .presets import skies as SK
        text = SK.dumps(context.world.halcyon, self.sky_name)
        path = self.filepath
        if self.to_library:
            folder = _sky_library_dir()
            if not folder:
                self.report({'ERROR'}, "Cannot reach the preset folder")
                return {'CANCELLED'}
            safe = ''.join(c for c in (self.sky_name or 'Sky')
                           if c.isalnum() or c in ' _-').strip() or 'Sky'
            path = os.path.join(folder, safe + '.halsky')
        if not path:
            self.report({'ERROR'}, "No file path")
            return {'CANCELLED'}
        if not path.lower().endswith('.halsky'):
            path += '.halsky'
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text)
        except OSError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Saved {os.path.basename(path)}")
        return {'FINISHED'}


class HALCYON_OT_sky_load(Operator):
    """Add a .halsky file to the preset list, without changing this sky"""

    bl_idname = 'halcyon.sky_load'
    bl_label = "Import Preset"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.halsky", options={'HIDDEN'})
    select: BoolProperty(
        name="Select After Importing", default=True,
        description="Point the preset list at what was just imported. It is "
                    "still not applied until Apply Preset is pressed")

    @classmethod
    def poll(cls, context):
        return context.world is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json
        import os
        import shutil

        from .presets import skies as SK
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No file")
            return {'CANCELLED'}
        # read it before copying it: a file that is not a sky should be
        # refused at the door rather than landing in the library and failing
        # every time somebody picks it
        try:
            with open(self.filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self.report({'ERROR'}, f"Cannot read it: {exc}")
            return {'CANCELLED'}
        if not isinstance(data, dict) or data.get('format') != SK.FORMAT:
            self.report({'ERROR'}, "That is not a Halcyon sky file")
            return {'CANCELLED'}

        folder = _sky_library_dir()
        if not folder:
            self.report({'ERROR'}, "Cannot reach the preset folder")
            return {'CANCELLED'}
        name = os.path.basename(self.filepath)
        dest = os.path.join(folder, name)
        stem, ext = os.path.splitext(name)
        n = 2
        while os.path.exists(dest) and not os.path.samefile(
                self.filepath, dest):
            dest = os.path.join(folder, f'{stem} {n}{ext}')
            n += 1
        try:
            if os.path.abspath(dest) != os.path.abspath(self.filepath):
                shutil.copyfile(self.filepath, dest)
        except OSError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if self.select:
            try:
                context.world.halcyon.sky_preset = 'FILE:' + dest
            except TypeError:
                pass          # the enum has not rebuilt yet; harmless
        self.report({'INFO'},
                    f"Added {os.path.splitext(os.path.basename(dest))[0]} "
                    f"to the preset list")
        return {'FINISHED'}


# ----------------------------------------------------------- water presets
#
# The same three operators as the sky library, against the other half of the
# World. They are deliberately not shared code: the file extensions, the
# folders and the field sets differ, and a generic version of this would be
# longer than both of them.


def _water_library_dir():
    from .properties import water_library_dir
    return water_library_dir()


def _user_waters():
    from .properties import user_waters
    return user_waters()


def water_preset_items(self=None, context=None):
    from .properties import water_preset_items as items
    return items(self, context)


class HALCYON_OT_water_preset(Operator):
    """Load the selected water onto this world's infinite plane"""

    bl_idname = 'halcyon.water_preset'
    bl_label = "Apply Preset"
    bl_options = {'REGISTER', 'UNDO'}

    preset: StringProperty(
        name="Water", default='',
        description="Leave empty to use whatever the World panel has selected")
    reset: BoolProperty(
        name="Reset First", default=True,
        description="Return every water setting to its default before "
                    "applying, so nothing carries over from the last one")

    @classmethod
    def poll(cls, context):
        return context.world is not None

    def execute(self, context):
        from .presets import waters as WA
        hs = context.world.halcyon
        key = self.preset or getattr(hs, 'water_preset', '') or ''
        if not key:
            self.report({'WARNING'}, "No water selected")
            return {'CANCELLED'}
        if key.startswith('FILE:'):
            try:
                with open(key[5:], 'r', encoding='utf-8') as fh:
                    ok, msg = WA.loads(hs, fh.read(), reset=self.reset)
            except OSError as exc:
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
        else:
            ok, msg = WA.apply_water(hs, key, reset=self.reset)
        if not ok:
            self.report({'ERROR'}, msg)
            return {'CANCELLED'}
        _poke_updated(context, context.world, context.scene)
        self.report({'INFO'}, f"Water: {msg}")
        return {'FINISHED'}


class HALCYON_OT_water_save(Operator):
    """Save this world's water to a .halwater file"""

    bl_idname = 'halcyon.water_save'
    bl_label = "Save Water"

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.halwater", options={'HIDDEN'})
    to_library: BoolProperty(
        name="Add to Library", default=False,
        description="Save into Halcyon's own folder so it appears in the "
                    "water list next to the built-in ones")
    water_name: StringProperty(name="Name", default="My Water")

    @classmethod
    def poll(cls, context):
        return context.world is not None

    def invoke(self, context, event):
        if self.to_library:
            return self.execute(context)
        if not self.filepath:
            self.filepath = (self.water_name or 'water') + '.halwater'
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import os

        from .presets import waters as WA
        text = WA.dumps(context.world.halcyon, self.water_name)
        path = self.filepath
        if self.to_library:
            folder = _water_library_dir()
            if not folder:
                self.report({'ERROR'}, "Cannot reach the preset folder")
                return {'CANCELLED'}
            safe = ''.join(c for c in (self.water_name or 'Water')
                           if c.isalnum() or c in ' _-').strip() or 'Water'
            path = os.path.join(folder, safe + '.halwater')
        if not path:
            self.report({'ERROR'}, "No file path")
            return {'CANCELLED'}
        if not path.lower().endswith('.halwater'):
            path += '.halwater'
        try:
            with open(path, 'w', encoding='utf-8') as fh:
                fh.write(text)
        except OSError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.report({'INFO'}, f"Saved {os.path.basename(path)}")
        return {'FINISHED'}


class HALCYON_OT_water_load(Operator):
    """Add a .halwater file to the preset list, without changing this water"""

    bl_idname = 'halcyon.water_load'
    bl_label = "Import Preset"
    bl_options = {'REGISTER'}

    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default="*.halwater", options={'HIDDEN'})
    select: BoolProperty(
        name="Select After Importing", default=True,
        description="Point the preset list at what was just imported. It is "
                    "still not applied until Apply Preset is pressed")

    @classmethod
    def poll(cls, context):
        return context.world is not None

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        import json
        import os
        import shutil

        from .presets import waters as WA
        if not self.filepath or not os.path.isfile(self.filepath):
            self.report({'ERROR'}, "No file")
            return {'CANCELLED'}
        # checked before it is copied, so a wrong pick leaves no trace
        try:
            with open(self.filepath, 'r', encoding='utf-8') as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            self.report({'ERROR'}, f"Cannot read it: {exc}")
            return {'CANCELLED'}
        if not isinstance(data, dict) or data.get('format') != WA.FORMAT:
            self.report({'ERROR'}, "That is not a Halcyon water file")
            return {'CANCELLED'}

        folder = _water_library_dir()
        if not folder:
            self.report({'ERROR'}, "Cannot reach the preset folder")
            return {'CANCELLED'}
        name = os.path.basename(self.filepath)
        dest = os.path.join(folder, name)
        stem, ext = os.path.splitext(name)
        n = 2
        while os.path.exists(dest) and not os.path.samefile(
                self.filepath, dest):
            dest = os.path.join(folder, f'{stem} {n}{ext}')
            n += 1
        try:
            if os.path.abspath(dest) != os.path.abspath(self.filepath):
                shutil.copyfile(self.filepath, dest)
        except OSError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        if self.select:
            try:
                context.world.halcyon.water_preset = 'FILE:' + dest
            except TypeError:
                pass          # the enum has not rebuilt yet; harmless
        self.report({'INFO'},
                    f"Added {os.path.splitext(os.path.basename(dest))[0]} "
                    f"to the preset list")
        return {'FINISHED'}


class HALCYON_PT_passes(HalcyonPanel, Panel):
    """The extra buffers written alongside the image, for the compositor."""

    bl_label = "Halcyon Passes"
    bl_context = "view_layer"

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.scene.halcyon
        col = layout.column(heading="Data")
        col.prop(hs, 'pass_depth')
        col.prop(hs, 'pass_normal')
        col.prop(hs, 'pass_position')
        col.prop(hs, 'pass_uv')
        col.separator()
        col.prop(hs, 'pass_object_index')
        col.prop(hs, 'pass_material_index')
        col.prop(hs, 'pass_mist')
        # R253: the frame passes and the BI light split
        col = layout.column(heading="Frame")
        col.prop(hs, 'pass_environment')
        col.prop(hs, 'pass_beauty')
        col = layout.column(heading="Light")
        col.prop(hs, 'pass_diffuse')
        col.prop(hs, 'pass_specular')
        col.prop(hs, 'pass_ambient')
        col.prop(hs, 'pass_emission')
        col.prop(hs, 'pass_shadow')
        col.prop(hs, 'pass_ao')
        col.prop(hs, 'pass_color')
        col.separator()
        col.prop(hs, 'pass_lights')
        note = layout.column(align=True)
        note.active = False
        note.scale_y = 0.8
        for line in _wrap("Data passes (Depth to Mist) skip the display chain, "
                          "the palette and the dither; Env and Beauty are "
                          "linear colour. The Light passes shade the frame on "
                          "the CPU by name and sum to the Beauty (Diffuse + "
                          "Specular + Ambient + Emission). Blender's own "
                          "Passes panel stays hidden because Vector and "
                          "Denoising Data are not produced.", 46):
            note.label(text=line)
        if hs.use_processes:
            box = layout.box()
            box.scale_y = 0.8
            box.label(text="Worker Processes send back pixels, not buffers,",
                      icon='INFO')
            box.label(text="so a frame with passes on renders in-process.")


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

        # the library is Bryce's, and only means anything under Bryce's sky
        if hs.mode == 'BRYCE':
            box = layout.box()
            box.label(text="Sky Presets", icon='WORLD')
            # the browsable gallery: every built-in sky ships a rendered
            # thumbnail (drawn by the engine's own sky module), so the
            # picker is a wall of pictures rather than 300 names
            try:
                box.template_icon_view(hs, 'sky_preset', show_labels=True,
                                       scale=5.0)
            except Exception:                                   # noqa: BLE001
                pass
            box.prop(hs, 'sky_preset', text="")
            box.operator('halcyon.sky_preset', text="Apply Preset",
                         icon='CHECKMARK')
            row = box.row(align=True)
            op = row.operator('halcyon.sky_save', text="Save As...",
                              icon='FILE_TICK')
            op.to_library = False
            op = row.operator('halcyon.sky_save', text="Add to Library",
                              icon='ADD')
            op.to_library = True
            box.operator('halcyon.sky_load', text="Import Preset...",
                         icon='IMPORT')

        col = layout.column()
        col.prop(hs, 'mode')
        if hs.mode != 'NODES':
            col.prop(hs, 'strength')
            col.prop(hs, 'rotation')
        col.separator()

        m = hs.mode
        if m == 'SOLID':
            col.prop(hs, 'color')
        elif m == 'PAINTED':
            # R233: the painted backdrop
            col.prop(hs, 'paint_look')
            col.prop(hs, 'paint_angle')
            col.prop(hs, 'paint_seed')
            col.separator()
            col.prop(hs, 'horizon')
            col.prop(hs, 'zenith')
            col.prop(hs, 'gradient_falloff')
            col.prop(hs, 'blend_mode')
            col.prop(hs, 'horizon_height')
            box = layout.box()
            box.label(text="Brush", icon='COLOR')
            box.prop(hs, 'paint_streaks')
            box.prop(hs, 'paint_streak_scale')
            box.prop(hs, 'paint_streak_angle')
            row = box.row(align=True)
            row.prop(hs, 'paint_dabs')
            row.prop(hs, 'paint_dab_scale', text="Scale")
            box = layout.box()
            box.label(text="Clouds", icon='MOD_FLUIDSIM')
            box.prop(hs, 'paint_clouds')
            box.prop(hs, 'paint_cloud_scale')
            box.prop(hs, 'paint_cloud_softness')
            box.prop(hs, 'paint_cloud_height')
            box.prop(hs, 'paint_cloud_color')
            box.prop(hs, 'paint_cloud_shadow')
            box = layout.box()
            box.label(text="Board", icon='FILE_TEXT')
            box.prop(hs, 'paint_paper')
            box.prop(hs, 'paint_paper_scale')
            box.prop(hs, 'paint_wash')
        elif m in ('GRADIENT', 'BANDS', 'BRYCE'):
            if m == 'BRYCE':
                col.prop(hs, 'sky_mode')
                if hs.sky_mode == 'SOFT':
                    note = col.row()
                    note.active = False
                    note.label(text="Horizon comes from the Sun Glow Colour",
                               icon='INFO')
            col.prop(hs, 'horizon')
            if m == 'BRYCE':
                col.prop(hs, 'use_sky_mid')
                sub = col.column()
                sub.active = hs.use_sky_mid
                sub.prop(hs, 'sky_mid')
                sub.prop(hs, 'sky_mid_height')
            col.prop(hs, 'zenith')
            col.prop(hs, 'gradient_falloff')
            if m in ('GRADIENT', 'BANDS'):
                col.prop(hs, 'blend_mode')
                col.prop(hs, 'horizon_height')
            if m == 'BANDS':
                col.separator()
                col.prop(hs, 'band_count')
                col.prop(hs, 'band_softness')
        elif m == 'STARFIELD':
            col.prop(hs, 'color', text="Space")
            col.separator()
            col.prop(hs, 'star_density')
            col.prop(hs, 'star_brightness')
            col.prop(hs, 'star_size')
            col.prop(hs, 'star_twinkle')
            col.prop(hs, 'old_stars')
            col.separator()
            col.prop(hs, 'nebula')
            sub = col.column()
            sub.active = hs.nebula > 0.0
            sub.prop(hs, 'nebula_color')
            sub.prop(hs, 'nebula_scale')
            sub.prop(hs, 'nebula_detail')
        elif m == 'HDRI':
            col.template_ID(hs, 'env_image', open='image.open')
            col.prop(hs, 'env_mapping')
            col.prop(hs, 'env_filter')
            col.prop(hs, 'env_tint')
        elif m == 'PHYSICAL':
            col.prop(hs, 'turbidity')
            col.prop(hs, 'ground_albedo')
        elif m == 'CYLINDER':
            # R251 C056: Doom's cylinder sky rides the world's image slot
            col.template_ID(hs, 'env_image', open='image.open')
            col.prop(hs, 'sky_cylinder_repeats')
            col.prop(hs, 'sky_cylinder_mid')
        elif m == 'LW_GRADIENT':
            # R251 C100: LightWave's Backdrop panel, top to bottom
            col.prop(hs, 'lw_zenith')
            col.prop(hs, 'lw_sky')
            col.prop(hs, 'lw_sky_squeeze')
            col.separator()
            col.prop(hs, 'lw_ground')
            col.prop(hs, 'lw_nadir')
            col.prop(hs, 'lw_ground_squeeze')

        if m in ('GRADIENT', 'BANDS', 'BRYCE', 'PHYSICAL', 'PAINTED',
                 'CYLINDER'):
            col.separator()
            col.prop(hs, 'show_ground')
            sub = col.column()
            sub.active = hs.show_ground
            sub.prop(hs, 'ground_color')

        col.separator()
        col.label(text="Ambient")
        col.prop(hs, 'ambient')
        col.prop(hs, 'ambient_level')
        col.separator()
        col.label(text="Exposure (Blender Internal)")
        row = col.row(align=True)
        row.prop(hs, 'exposure')
        row.prop(hs, 'exposure_range', text="Range")
        # R251 C038: the DS rear-plane depth bitmap (a Z pass the
        # z-buffer honours, tiled 1:1 with the DS's offset)
        col.separator()
        col.label(text="Backdrop Depth (Nintendo DS)")
        col.template_ID(hs, 'backdrop_depth_image', open='image.open')
        sub = col.column()
        sub.active = hs.backdrop_depth_image is not None
        row = sub.row(align=True)
        row.prop(hs, 'backdrop_offset_x')
        row.prop(hs, 'backdrop_offset_y')


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


class HALCYON_PT_world_weather(HalcyonPanel, Panel):
    """R200: the Weather overlay -- rain, snow, embers, ash falling in
    front of the picture, under any sky mode (it weathers the sky, it
    never replaces it)."""
    bl_label = "Weather"
    bl_parent_id = 'HALCYON_PT_world'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.engine == ENGINE and context.world is not None

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        hs = context.world.halcyon
        col = layout.column()
        col.prop(hs, 'weather')
        if str(hs.weather) == 'NONE':
            return
        col.prop(hs, 'weather_color')
        col.prop(hs, 'weather_opacity')
        col.separator()
        col.prop(hs, 'weather_density')
        col.prop(hs, 'weather_size')
        col.prop(hs, 'weather_layers')
        col.separator()
        col.prop(hs, 'weather_speed')
        col.prop(hs, 'weather_angle')
        col.prop(hs, 'weather_drift')
        if str(hs.weather) == 'RAIN':
            col.prop(hs, 'weather_streak')
        row = col.row(align=True)
        row.prop(hs, 'weather_glow')
        row.prop(hs, 'weather_flicker')
        col.prop(hs, 'weather_seed')
        if str(hs.weather) == 'RAIN':
            note = col.row()
            note.active = False
            note.label(text="Acid rain: pick a green colour",
                       icon='INFO')
        elif str(hs.weather) == 'EMBERS':
            note = col.row()
            note.active = False
            note.label(text="Set Angle to 180° so embers rise",
                       icon='INFO')


def _uses_nodes(mat):
    """Material.use_nodes without the 6.0 deprecation warning per redraw.

    The attribute is going away because it will always be true; until it
    does, honour it where it exists -- silently, because UI draw code runs
    every redraw and the warning flooded the console once per panel.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            return bool(getattr(mat, 'use_nodes', True))
    except Exception:                                           # noqa: BLE001
        return True


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
    HALCYON_OT_apply_preset, HALCYON_OT_set_resolution,
    HALCYON_OT_halo_ramp,
    HALCYON_OT_adopt_shadow_settings,
) + RESOLUTION_MENUS + (
    HALCYON_MT_resolutions, HALCYON_PT_output,
    HALCYON_OT_fix_view_transform,
    HALCYON_OT_sky_preset, HALCYON_OT_sky_save, HALCYON_OT_sky_load,
    HALCYON_OT_water_preset, HALCYON_OT_water_save, HALCYON_OT_water_load,
    HALCYON_UL_materials,
    HalcyonPreferences,
    HALCYON_PT_presets, HALCYON_PT_sampling, HALCYON_PT_geometry,
    HALCYON_PT_shading, HALCYON_PT_lighting, HALCYON_PT_spot_cones,
    HALCYON_PT_shadows, HALCYON_PT_ao, HALCYON_PT_radiosity,
    HALCYON_PT_raytrace, HALCYON_PT_textures, HALCYON_PT_transparency,
    HALCYON_PT_fog, HALCYON_PT_effects, HALCYON_PT_colour, HALCYON_PT_display,
    HALCYON_PT_crt, HALCYON_PT_composite, HALCYON_PT_film,
    HALCYON_PT_backgrounds, HALCYON_PT_jpeg,
    HALCYON_PT_performance, HALCYON_PT_debug,
    HALCYON_OT_clear_palette_cache,
    HALCYON_OT_diagnostics, HALCYON_PT_material,
    HALCYON_PT_light, HALCYON_PT_wireframe, HALCYON_PT_outline,
    HALCYON_PT_passes,
    HALCYON_PT_world, HALCYON_PT_world_sun,
    HALCYON_PT_world_atmosphere, HALCYON_PT_world_cumulus,
    HALCYON_PT_world_stratus, HALCYON_PT_world_effects,
    HALCYON_PT_world_ground, HALCYON_PT_world_weather,
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
