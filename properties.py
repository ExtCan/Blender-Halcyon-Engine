"""The Blender-side settings, generated from core.settings.RenderSettings.

Generating rather than hand-writing guarantees the UI and the renderer can never
drift apart: every dataclass field becomes exactly one property, and
`to_settings()` copies them straight back.
"""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import PropertyGroup

from .core.settings import RESOLUTION_PRESETS, RenderSettings
from .core.shading import MODEL_ITEMS
from .presets.library import preset_items


def _items(*pairs):
    return [(a, b, c) for a, b, c in pairs]


AA_MODE = _items(
    ('NONE', "None", "One sample per pixel -- hard, aliased edges"),
    ('SUPERSAMPLE', "Supersample", "Render larger and filter down"),
    ('EDGE', "Edge Only", "Extra samples only where geometry IDs differ"),
    ('ACCUMULATE', "Accumulation Buffer", "Jittered passes averaged together"),
)
AA_FILTER = _items(
    ('BOX', "Box", "Flat average -- what most 1990s renderers used"),
    ('TRIANGLE', "Triangle", "Linear falloff"),
    ('GAUSS', "Gaussian", "Soft"),
    ('CATROM', "Catmull-Rom", "Sharpening"),
    ('MITCHELL', "Mitchell-Netravali", "Balanced"),
)
SUBPIXEL = _items(
    ('FLOAT', "Floating Point", "Modern sub-pixel accuracy"),
    ('FIXED_4', "Fixed 1/4 pixel", "Quarter-pixel snapping"),
    ('FIXED_1', "Fixed 1 pixel", "Whole-pixel vertices"),
    ('INTEGER', "Integer", "PlayStation-style integer raster coordinates"),
)
DEPTH_SORT = _items(
    ('ZBUFFER', "Z-Buffer", "Per-pixel depth test. Always correct"),
    ('PAINTERS', "Painter's Algorithm",
     "Compare whole polygons instead of fragments, as hardware without a depth "
     "buffer had to. Interpenetrating surfaces meet along a polygon edge rather "
     "than their true intersection, and a large polygon can be wrongly occluded "
     "by a small nearer one"),
)
PAINTERS_KEY = _items(
    ('CENTROID', "Centroid", "Sort on the middle of each polygon. The usual choice"),
    ('NEAREST', "Nearest Vertex",
     "Sort on the closest corner. Fewer errors on surfaces that face the "
     "camera, more on long polygons running away from it"),
    ('FARTHEST', "Farthest Vertex", "Sort on the furthest corner"),
)
SHADING_RATE = _items(
    ('PIXEL', "Per Pixel (Phong)", "Shade every fragment"),
    ('VERTEX', "Per Vertex (Gouraud)", "Shade at vertices and interpolate colour"),
    ('FACE', "Per Face (Flat)", "One colour per polygon"),
)
FALLOFF = _items(
    ('NONE', "None", "No attenuation"),
    ('INVERSE', "Inverse", "1/d"),
    ('INVERSE_SQUARE', "Inverse Square", "1/d^2, physically correct"),
    ('CUSTOM', "Custom Range", "Linear ramp between start and end"),
)
SHADOW_MODE = _items(
    ('NONE', "None", "No shadows"),
    ('MAP', "Shadow Maps", "Depth maps rendered from each light"),
    ('RAY', "Ray Traced", "Hard or area-sampled shadow rays"),
    ('PER_LIGHT', "Per Light", "Use each light's own setting"),
)
TEX_FILTER = _items(
    ('NEAREST', "Nearest", "Point sampling -- chunky texels"),
    ('BILINEAR', "Bilinear", "Smooth"),
    ('TRILINEAR', "Trilinear", "Bilinear plus mip blending"),
    ('N64_3POINT', "3-Point (N64)", "The Reality Coprocessor's triangular filter"),
)
WRAP = _items(('REPEAT', "Repeat", ""), ('EXTEND', "Extend", ""),
              ('CLIP', "Clip", ""), ('MIRROR', "Mirror", ""))
TRANSPARENCY = _items(
    ('NONE', "Opaque", "Ignore alpha"),
    ('STIPPLE', "Screen Door", "Dithered stipple, as hardware without blending did"),
    ('SORTED', "Sorted Blend", "Depth-sorted alpha blending"),
    ('ABUFFER', "A-Buffer", "Per-pixel fragment lists, correct through any depth"),
)
FOG_MODE = _items(
    ('LINEAR', "Linear", ""), ('EXP', "Exponential", ""),
    ('EXP2', "Exponential Squared", ""),
    ('TABLE16', "16-Step Table", "Banded fixed-function fog table"),
)
COLOR_DEPTH = _items(
    ('32', "32-bit (8:8:8:8)", "True colour with alpha"),
    ('24', "24-bit (8:8:8)", "True colour"),
    ('16', "16-bit (5:6:5)", "High colour"),
    ('15', "15-bit (5:5:5)", "High colour, PlayStation and Voodoo"),
    ('12', "12-bit (4:4:4)", "Amiga AGA register depth"),
    ('8', "8-bit indexed", "256 colours from a palette"),
    ('4', "4-bit indexed", "16 colours"),
    ('1', "1-bit", "Monochrome"),
    ('HAM8', "Amiga HAM8", "Hold-and-modify, 6 bits per channel"),
    ('HAM6', "Amiga HAM6", "Hold-and-modify, 4 bits per channel"),
)
PALETTE_MODE = _items(
    ('ADAPTIVE', "Adaptive", "Built from this image's own colours"),
    ('VGA256', "VGA Default 256", "The IBM VGA BIOS palette"),
    ('MAC256', "Macintosh System", "The System 7 256-colour table"),
    ('WEB216', "Web Safe 216", "The browser-safe cube"),
    ('FIXED_666', "6:6:6 Cube", "216-entry RGB cube"),
    ('WIN20', "Windows 20", "The reserved system colours"),
    ('EGA16', "EGA 16", "IBM Enhanced Graphics Adapter"),
    ('CGA4', "CGA 4", "Cyan/magenta/white"),
    ('GRAY', "Greyscale", ""),
    ('CUSTOM', "Custom", "From a palette image"),
)
PALETTE_METHOD = _items(
    ('MEDIAN_CUT', "Median Cut", "Heckbert's algorithm, the period standard"),
    ('OCTREE', "Octree", "Gervautz-Purgathofer"),
    ('POPULARITY', "Popularity", "Most frequent colours"),
    ('KMEANS', "K-Means", "Slowest, best quality"),
)
DITHER = _items(
    ('NONE', "None", ""),
    ('BAYER2', "Bayer 2x2", ""), ('BAYER4', "Bayer 4x4", ""),
    ('BAYER8', "Bayer 8x8", ""), ('BAYER16', "Bayer 16x16", ""),
    ('HALFTONE', "Halftone", "Clustered dot"),
    ('FLOYD', "Floyd-Steinberg", ""), ('JJN', "Jarvis-Judice-Ninke", ""),
    ('STUCKI', "Stucki", ""), ('ATKINSON', "Atkinson", "The Macintosh kernel"),
    ('BURKES', "Burkes", ""), ('SIERRA', "Sierra", ""),
    ('SIERRA_LITE', "Sierra Lite", ""), ('NOISE', "Blue Noise", ""),
)
COLOR_MGMT = _items(
    ('NONE', "None (Period Correct)", "No transform -- 1990s renderers had none"),
    ('SRGB', "sRGB", "Modern display transform"),
    ('FILMIC', "Filmic", "Highlight rolloff"),
    ('REINHARD', "Reinhard", "Simple tone map"),
)
CRT_MASK = _items(
    ('NONE', "None", ""),
    ('APERTURE', "Aperture Grille", "Trinitron vertical stripes"),
    ('SLOT', "Slot Mask", "Staggered slots"),
    ('SHADOW', "Shadow Mask", "Triad dots"),
)
INTERLACE = _items(
    ('NONE', "None", ""),
    ('FIELDS', "Fields", "Alternate scan lines per frame"),
    ('BLEND', "Blended", "Darken alternate lines"),
)
OUTPUT_SCALE = _items(('NONE', "1x", ""), ('2X', "2x", ""), ('3X', "3x", ""),
                      ('4X', "4x", ""))
DEBUG_PASS = _items(
    ('BEAUTY', "Beauty", ""), ('DEPTH', "Depth", ""), ('NORMAL', "Normal", ""),
    ('UV', "UV", ""), ('MATID', "Material ID", ""), ('OVERDRAW', "Overdraw", ""),
    ('WIREFRAME', "Wireframe", ""),
)
BUCKET = _items(('HILBERT', "Hilbert", ""), ('SPIRAL', "Spiral", ""),
                ('TOP', "Top to Bottom", ""), ('RANDOM', "Random", ""))
LIGHT_LIMIT = _items(
    ('BRIGHTEST', "Brightest", "Keep the strongest lights"),
    ('NEAREST', "Nearest", "Keep the closest lights"),
    ('FIRST', "First", "Keep them in scene order"),
)
NORMAL_SOURCE = _items(
    ('AUTO', "Auto", "Use the mesh's own smooth/flat flags"),
    ('SMOOTH', "Force Smooth", ""), ('FACE', "Force Faceted", ""),
)
FORCE_MODEL = [('NONE', "Don't Override", "Let each material choose")] + \
    [(a, b, c) for a, b, c in MODEL_ITEMS]

RENDER_DEVICE = _items(
    ('CPU', "CPU", "Everything on the CPU. Every feature works"),
    ('GPU', "GPU", "Move the proven post stages to the GPU. Features that "
                   "cannot run there fall back to the CPU automatically"),
)

ENUMS = {
    'render_device': RENDER_DEVICE,
    'aa_mode': AA_MODE, 'aa_filter': AA_FILTER, 'subpixel_precision': SUBPIXEL,
    'depth_sort': DEPTH_SORT, 'painters_key': PAINTERS_KEY, 'shading_rate': SHADING_RATE,
    'default_model': [(a, b, c) for a, b, c in MODEL_ITEMS],
    'force_model': FORCE_MODEL, 'normal_source': NORMAL_SOURCE,
    'light_falloff_default': FALLOFF, 'shadow_default': SHADOW_MODE,
    'light_limit_mode': LIGHT_LIMIT, 'tex_filter': TEX_FILTER,
    'tex_wrap_default': WRAP, 'transparency': TRANSPARENCY,
    'stipple_pattern': DITHER, 'fog_mode': FOG_MODE, 'glow_quality':
    _items(('BOX', "Box", ""), ('GAUSS', "Gaussian", "")),
    'color_depth': COLOR_DEPTH, 'palette_mode': PALETTE_MODE,
    'palette_method': PALETTE_METHOD, 'dither': DITHER,
    'color_management': COLOR_MGMT, 'crt_mask': CRT_MASK,
    'interlace': INTERLACE, 'output_scale': OUTPUT_SCALE,
    'bucket_order': BUCKET, 'debug_pass': DEBUG_PASS,
    'res_preset': [('CUSTOM', "Custom", "")] +
    [(k, k.replace('_', ' ').title(), f"{v[0]}x{v[1]}")
     for k, v in RESOLUTION_PRESETS.items()],
}

# field -> (min, max, soft_min, soft_max, step/precision hints)
RANGES = {
    'aa_samples': (1, 64), 'aa_filter_width': (0.1, 4.0),
    'aa_edge_threshold': (0.0, 1.0), 'vertex_snap_grid': (0.05, 16.0),
    'depth_precision': (4, 32), 'polygon_offset': (-10.0, 10.0),
    'light_clamp': (0.0, 1000.0), 'ao_distance': (0.001, 1000.0),
    'ao_samples': (1, 256), 'ao_intensity': (0.0, 1.0),
    'global_ambient_level': (0.0, 10.0), 'max_lights': (0, 64),
    'process_count': (0, 64), 'shadow_map_size': (32, 4096), 'shadow_bias': (0.0, 10.0),
    'shadow_softness': (0.0, 32.0), 'shadow_samples': (1, 64),
    'ray_depth': (0, 16), 'ray_bias': (0.0, 1.0),
    'reflection_blur_samples': (1, 64), 'tex_mip_bias': (-4.0, 4.0),
    'tex_aniso': (1, 16), 'tex_max_size': (0, 4096), 'tex_quantize': (0, 256),
    'tex_affine_subdiv': (0, 64), 'alpha_bits': (1, 8),
    'alpha_threshold': (0.0, 1.0), 'fog_start': (0.0, 100000.0),
    'fog_end': (0.0, 100000.0), 'fog_density': (0.0, 10.0),
    'lens_distortion': (-1.0, 1.0), 'chromatic_aberration': (0.0, 8.0),
    'shaft_threshold': (0.0, 4.0), 'shaft_length': (0.0, 1.0),
    'shaft_decay': (0.5, 1.0), 'shaft_samples': (2, 128),
    'dof_focus': (0.01, 10000.0), 'dof_amount': (0.0, 8.0),
    'dof_layers': (2, 16), 'dof_max_radius': (1.0, 128.0),
    'displacement_scale': (0.0, 8.0),
    'glow_threshold': (0.0, 4.0), 'glow_radius': (1.0, 128.0),
    'glow_intensity': (0.0, 4.0), 'star_points': (2, 12),
    'star_length': (1.0, 256.0), 'star_rotation': (0.0, 6.2832),
    'star_intensity': (0.0, 4.0), 'flare_intensity': (0.0, 4.0),
    'flare_ghosts': (1, 12), 'flare_streak': (0.0, 2.0),
    'palette_size': (2, 256), 'dither_strength': (0.0, 2.0),
    'exposure': (0.0, 64.0), 'gamma': (0.1, 5.0), 'contrast': (-1.0, 4.0),
    'saturation': (0.0, 4.0), 'brightness': (-1.0, 1.0),
    'crt_scanlines': (0.0, 1.0), 'crt_mask_strength': (0.0, 1.0),
    'crt_bloom': (0.0, 2.0), 'crt_curvature': (0.0, 2.0),
    'crt_vignette': (0.0, 2.0), 'composite_bleed': (0.0, 2.0),
    'composite_ringing': (0.0, 2.0), 'composite_dot_crawl': (0.0, 2.0),
    'jpeg_quality': (1, 100), 'jpeg_passes': (1, 8), 'block_size': (4, 16),
    'tile_size': (16, 1024), 'threads': (0, 64), 'preview_scale': (1, 16),
    'max_texture_memory': (0, 4096), 'seed': (0, 2 ** 30),
    'wire_width': (0.1, 8.0), 'resolution_x': (1, 16384),
    'resolution_y': (1, 16384), 'pixel_aspect_x': (0.01, 100.0),
    'pixel_aspect_y': (0.01, 100.0), 'jitter_seed': (0, 2 ** 30),
    'clip_near_epsilon': (1e-6, 1.0), 'decay_start': (0.0, 10000.0),
}

LABELS = {
    'aa_mode': "Anti-Aliasing", 'aa_samples': "Samples",
    'aa_filter': "Filter", 'aa_filter_width': "Filter Width",
    'vertex_snap': "Vertex Snapping",
    'vertex_snap_grid': "Snap Grid (px)",
    'depth_precision': "Z-Buffer Bits", 'depth_sort': "Depth Method",
    'shading_rate': "Shading Rate", 'default_model': "Default Shader",
    'force_model': "Override Shader",
    'specular_in_gamma': "Specular in Gamma Space",
    'max_lights': "Light Limit", 'shadow_default': "Shadow Method",
    'tex_filter': "Texture Filter",
    'tex_perspective': "Perspective Correction",
    'tex_max_size': "Texture Size Limit", 'tex_quantize': "Texture Colours",
    'threads': "Threads", 'film_transparent': "Transparent Film",
    'cache_shadows': "Cache Shadow Maps", 'show_stats': "Timing Breakdown",
    'fast_background': "Fast Background",
    'use_processes': "Use Worker Processes", 'process_count': "Processes",
    'gpu_post': "GPU Post Processing", 'render_device': "Device",
    'palette_lock': "Lock Palette", 'displacement_scale': "Displacement",
    'color_depth': "Colour Depth", 'palette_mode': "Palette",
    'palette_method': "Quantiser", 'dither': "Dither",
    'color_management': "View Transform", 'crt': "CRT Simulation",
    'composite': "Composite Video", 'jpeg_artifacts': "JPEG Artefacts",
    'output_scale': "Pixel Scale", 'debug_pass': "Render Pass",
}

DESCRIPTIONS = {
    'threads': "How many threads share the shading work. Measured on a 20-core "
               "machine this is neutral at best and about 3% slower at worst, "
               "because NumPy releases the interpreter lock only for large "
               "array operations and the node evaluator is dominated by Python "
               "dispatch between small ones. Defaults to 1 for that reason. "
               "Worker Processes is the route that actually parallelises",
    'render_device': "Where the frame is computed. GPU currently moves only the "
                     "post stages that have been measured against the CPU path "
                     "on real hardware; everything else stays on the CPU and "
                     "the panel says which features and why",
    'gpu_post': "Run the parallel post stages on the GPU through Blender's own "
                "gpu module, the layer EEVEE is built on. Experimental: the "
                "shaders were written on a machine with no GPU and have never "
                "been executed by a driver. Falls back to the CPU stage by "
                "stage, and says why on the console",
    'lens_distortion': "Barrel distortion below zero, pincushion above. What a "
                       "cheap lens does to straight lines",
    'chromatic_aberration': "Splits the colour channels radially, because a "
                            "real lens does not focus red and blue in the same "
                            "place. The clearest tell that an image went "
                            "through glass",
    'dof': "Defocus by splitting the frame into depth slabs and blurring each. "
           "What compositors of the era did, at a handful of blurs rather than "
           "hundreds of samples",
    'dof_focus': "Distance from the camera that stays sharp",
    'dof_amount': "How quickly things go soft either side of the focus distance",
    'dof_layers': "Number of depth slabs. More is smoother and slower",
    'shaft_threshold': "How bright a pixel must be to throw a shaft",
    'shaft_length': "How far the streaks reach toward the light",
    'displacement_scale': "Strength of the bump derived from a material's "
                          "Displacement output. Geometry is not tessellated -- "
                          "the height becomes a normal perturbation, which is "
                          "what scanline renderers of the era did",
    'painters_key': "Which point on a polygon decides its sort order. Changing "
                    "it moves where Painter's algorithm goes wrong",
    'max_transparent_layers': "How many overlapping transparent surfaces a "
                              "pixel may accumulate. Beyond a handful the "
                              "furthest ones contribute almost nothing but cost "
                              "as much as the first. 0 means no limit",
    'fast_background': "Evaluate the world once per output pixel instead of "
                       "once per supersample. A sky is smooth almost "
                       "everywhere, and at 4x this is sixteen times less work "
                       "for the background. Turn it off if a sharp sun disc or "
                       "a detailed HDRI shows aliasing",
    'show_stats': "Print a per-stage timing breakdown to the system console "
                  "after each frame, so it is clear where the time actually "
                  "went rather than where it is assumed to go",
    'cache_shadows': "Reuse shadow maps while the lights and geometry hold "
                     "still. Each map is a full rasterisation pass and a point "
                     "light needs six, so this is most of the saving on an "
                     "animation with static lighting",
    'use_processes': "Split each frame across separate Python processes. "
                     "Threads only run in parallel where NumPy releases the "
                     "interpreter lock; processes have no shared lock at all. "
                     "Falls back to rendering in Blender if workers cannot "
                     "start, and the reason is printed to the console",
    'process_count': "How many worker processes to start. 0 uses one per core",
    'dither_serpentine': "Alternate the scan direction each row, which hides the "
                         "directional worming error diffusion can produce. "
                         "Turning it off is about twice as fast, because the "
                         "diffusion can then be processed a diagonal at a time "
                         "instead of a pixel at a time",
    'palette_lock': "Build the adaptive palette once and reuse it. Stops the "
                    "colours crawling between frames of an animation, and skips "
                    "rebuilding the palette on every frame",
    'film_transparent': "Render the background with zero alpha so it can be "
                        "composited. Off means an opaque background. Blender's "
                        "own Film > Transparent also switches this on",
    'threads': "Worker threads used for shading. 0 follows Blender's own "
               "Performance setting, which normally means one per core. "
               "Shading is split into independent chunks, so this scales well",
    'preview_scale': "Viewport preview is rendered at 1/N resolution and scaled "
                     "up. Raise it for a faster, chunkier preview",
    'output_scale': "Render at 1/N of the output resolution and scale back up "
                    "with nearest-neighbour. The output stays the size you set, "
                    "and the render costs N squared times less",
    'vertex_snap': "Round transformed vertices to a pixel grid, as the "
                   "PlayStation's integer GTE did",
    'tex_perspective': "Turn off for the affine texture warp of hardware with "
                       "no perspective divide per texel",
    'specular_in_gamma': "Compute highlights in display space. 1990s renderers "
                         "had no linear workflow, and the blown-out speculars "
                         "are a large part of the look",
    'color_management': "Leave at None for period-correct output. Blender's own "
                        "view transform is bypassed by this engine",
    'depth_sort': "Painter's algorithm reproduces the sorting errors of "
                  "hardware without a depth buffer",
    'max_lights': "Hardware light limit. Lights beyond this are dropped, as on "
                  "fixed-function pipelines. Zero means unlimited",
}

COLOR_FIELDS = {'global_ambient', 'fog_color', 'wire_color'}


def _build():
    """Turn the dataclass into a dict of bpy properties."""
    import dataclasses
    props = {}
    for f in dataclasses.fields(RenderSettings):
        name = f.name
        default = f.default
        label = LABELS.get(name, name.replace('_', ' ').title())
        desc = DESCRIPTIONS.get(name, '')
        if name in ENUMS:
            props[name] = EnumProperty(name=label, description=desc,
                                       items=ENUMS[name],
                                       default=default if any(
                                           i[0] == default for i in ENUMS[name])
                                       else ENUMS[name][0][0])
        elif isinstance(default, bool):
            props[name] = BoolProperty(name=label, description=desc,
                                       default=default)
        elif isinstance(default, int):
            lo, hi = RANGES.get(name, (0, 2 ** 31 - 1))
            props[name] = IntProperty(name=label, description=desc,
                                      default=default, min=int(lo), max=int(hi),
                                      soft_min=int(lo), soft_max=int(hi))
        elif isinstance(default, float):
            lo, hi = RANGES.get(name, (-1e6, 1e6))
            props[name] = FloatProperty(name=label, description=desc,
                                        default=default, min=lo, max=hi,
                                        soft_min=lo, soft_max=hi)
        elif isinstance(default, tuple) or name in COLOR_FIELDS:
            props[name] = FloatVectorProperty(
                name=label, description=desc, subtype='COLOR', size=3,
                default=tuple(default), min=0.0, max=1.0)
        else:
            props[name] = StringProperty(name=label, description=desc,
                                         default=str(default or ''))
    return props


class HalcyonSettings(PropertyGroup):
    """All Halcyon render settings, mirroring core.settings.RenderSettings."""

    __annotations__ = _build()

    preset: EnumProperty(
        name="Preset",
        description="Load the settings of a specific 1990s renderer or machine",
        items=lambda self, ctx: preset_items(),
    )
    ui_tab: EnumProperty(
        name="Tab", items=_items(
            ('SAMPLING', "Sampling", ""), ('SHADING', "Shading", ""),
            ('OUTPUT', "Output", ""), ('DISPLAY', "Display", "")),
        default='SAMPLING')

    def to_settings(self):
        """Copy into a plain RenderSettings for the bpy-free renderer."""
        import dataclasses
        st = RenderSettings()
        for f in dataclasses.fields(RenderSettings):
            if not hasattr(self, f.name):
                continue
            v = getattr(self, f.name)
            if isinstance(f.default, tuple):
                v = tuple(v)
            setattr(st, f.name, v)
        return st


class HalcyonMaterialSettings(PropertyGroup):
    """Per-material overrides, for materials that don't use a node tree."""

    use_override: BoolProperty(
        name="Halcyon Shader", default=False,
        description="Shade this material with a fixed model instead of its node tree")
    model: EnumProperty(name="Model", items=[(a, b, c) for a, b, c in MODEL_ITEMS],
                        default='PHONG')
    diffuse: FloatVectorProperty(name="Diffuse", subtype='COLOR', size=3,
                                 default=(0.8, 0.8, 0.8), min=0.0, max=1.0)
    diffuse_level: FloatProperty(name="Diffuse Level", default=1.0, min=0.0, max=2.0)
    specular: FloatVectorProperty(name="Specular", subtype='COLOR', size=3,
                                  default=(1.0, 1.0, 1.0), min=0.0, max=1.0)
    specular_level: FloatProperty(name="Specular Level", default=0.5, min=0.0, max=4.0)
    glossiness: FloatProperty(name="Glossiness", default=25.0, min=0.5, max=8192.0)
    ambient_level: FloatProperty(name="Ambient", default=1.0, min=0.0, max=4.0)
    emission: FloatVectorProperty(name="Self-Illumination", subtype='COLOR', size=3,
                                  default=(0.0, 0.0, 0.0), min=0.0, max=1.0)
    emission_level: FloatProperty(name="Self-Illum Level", default=0.0, min=0.0,
                                  max=64.0)
    opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0)
    ior: FloatProperty(name="IOR", default=1.45, min=1.0, max=4.0)
    roughness: FloatProperty(name="Roughness", default=0.3, min=0.0, max=1.0)
    anisotropy: FloatProperty(name="Anisotropy", default=0.0, min=-1.0, max=1.0)
    aniso_rotation: FloatProperty(name="Aniso Rotation", default=0.0, min=0.0,
                                  max=6.2832)
    metallic: FloatProperty(name="Metalness", default=0.0, min=0.0, max=1.0)
    reflect_level: FloatProperty(name="Reflection", default=0.0, min=0.0, max=1.0)
    soften: FloatProperty(name="Soften", default=0.0, min=0.0, max=1.0,
                          description="3D Studio's specular softening at grazing angles")
    two_sided: BoolProperty(name="Two Sided", default=True)
    shadeless: BoolProperty(name="Shadeless", default=False)
    cast_shadow: BoolProperty(name="Cast Shadows", default=True)
    receive_shadow: BoolProperty(name="Receive Shadows", default=True)
    wire: BoolProperty(name="Wireframe", default=False)
    wire_size: FloatProperty(name="Wire Size", default=1.0, min=0.1, max=16.0)


class HalcyonLightSettings(PropertyGroup):
    decay: EnumProperty(
        name="Falloff",
        items=[('DEFAULT', "Scene Default",
                "Use the falloff set in Render Properties > Lighting")] + FALLOFF,
        default='DEFAULT')
    decay_start: FloatProperty(name="Falloff Start", default=0.0, min=0.0)
    decay_end: FloatProperty(name="Falloff End", default=25.0, min=0.0)
    shadow: EnumProperty(name="Shadows", items=_items(
        ('NONE', "None", ""), ('MAP', "Shadow Map", ""),
        ('RAY', "Ray Traced", "")), default='MAP')
    shadow_map_size: IntProperty(name="Map Size", default=512, min=32, max=4096)
    shadow_bias: FloatProperty(name="Bias", default=0.02, min=0.0, max=10.0)
    shadow_softness: FloatProperty(name="Softness", default=1.0, min=0.0, max=32.0)
    shadow_samples: IntProperty(name="Samples", default=4, min=1, max=64)
    shadow_density: FloatProperty(name="Density", default=1.0, min=0.0, max=1.0)
    shadow_color: FloatVectorProperty(name="Shadow Colour", subtype='COLOR', size=3,
                                      default=(0.0, 0.0, 0.0), min=0.0, max=1.0)
    negative: BoolProperty(name="Negative", default=False,
                           description="Subtract light instead of adding it, as "
                                       "3D Studio and LightWave allowed")
    diffuse_only: BoolProperty(name="Diffuse Only", default=False)
    specular_only: BoolProperty(name="Specular Only", default=False)
    ambient_only: BoolProperty(name="Ambient Only", default=False)
    hotspot: FloatProperty(name="Hotspot", default=0.0, min=0.0, max=3.1416,
                           description="Inner cone angle, the 3D Studio "
                                       "hotspot/falloff pair")
    volumetric: FloatProperty(
        name="Volumetric", default=0.0, min=0.0, max=4.0,
        description="Scatters light along the view ray toward this lamp, "
                    "giving the shafts and haze a bright source throws through "
                    "an atmosphere")
    exclude_collection: PointerProperty(
        name="Light Linking", type=bpy.types.Collection,
        description="A collection this lamp treats specially. Every 1990s "
                    "package let you say which objects a light touched, and it "
                    "is still the quickest way to control a render")
    exclude_mode: EnumProperty(name="Linking", default='EXCLUDE', items=_items(
        ('EXCLUDE', "Exclude", "The collection is not lit by this lamp"),
        ('ONLY', "Only", "Nothing but the collection is lit by this lamp")))


def _col(name, default, desc=''):
    return FloatVectorProperty(name=name, description=desc, subtype='COLOR',
                               size=3, default=default, min=0.0, max=1.0)


class HalcyonWorldSettings(PropertyGroup):
    mode: EnumProperty(name="Sky", default='NODES', items=_items(
        ('NODES', "Use Node Tree", "Evaluate the world's own shader nodes"),
        ('SOLID', "Solid Colour", "A single flat background colour"),
        ('GRADIENT', "Gradient", "Horizon to zenith blend, with an optional ground"),
        ('BRYCE', "Bryce Atmosphere",
         "Layered sky: gradient, sun glow, haze band and a fractal cloud deck"),
        ('PHYSICAL', "Physical Sky", "Preetham analytic daylight"),
        ('HDRI', "Image / HDRI", "Wrap an image around the scene")))
    strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=64.0)
    rotation: FloatProperty(name="Rotation", default=0.0, min=-6.2832, max=6.2832,
                            subtype='ANGLE',
                            description="Spin the sky around the vertical axis")
    ambient: _col("Ambient", (0.0, 0.0, 0.0))
    ambient_level: FloatProperty(name="Ambient Level", default=1.0, min=0.0, max=8.0)

    color: _col("Colour", (0.05, 0.05, 0.06))
    horizon: _col("Horizon", (0.55, 0.65, 0.80))
    zenith: _col("Zenith", (0.10, 0.25, 0.65))
    ground_color: _col("Ground", (0.18, 0.15, 0.12))
    show_ground: BoolProperty(name="Ground Plane", default=False,
                              description="Colour everything below the horizon")
    horizon_height: FloatProperty(name="Horizon Height", default=0.0,
                                  min=-1.0, max=1.0)
    gradient_falloff: FloatProperty(name="Falloff", default=1.0, min=0.01, max=8.0,
                                    description="Higher values keep the horizon "
                                                "colour further up the sky")
    blend_mode: EnumProperty(name="Blend", default='LINEAR', items=_items(
        ('LINEAR', "Linear", ""), ('SMOOTH', "Smooth", ""),
        ('SHARP', "Sharp", ""), ('EASE', "Ease", "")))

    sun_elevation: FloatProperty(name="Sun Altitude", default=0.35,
                                 min=-1.5708, max=1.5708, subtype='ANGLE')
    sun_rotation: FloatProperty(name="Sun Azimuth", default=0.6,
                                min=-6.2832, max=6.2832, subtype='ANGLE')
    sun_color: _col("Sun Colour", (1.0, 0.94, 0.82))
    sun_size: FloatProperty(name="Sun Size", default=0.03, min=0.0, max=1.5,
                            subtype='ANGLE')
    sun_intensity: FloatProperty(name="Sun Intensity", default=1.0, min=0.0, max=64.0)
    sun_glow: FloatProperty(name="Sun Glow", default=0.35, min=0.0, max=1.0,
                            description="Width of the halo around the sun")
    sun_disc: BoolProperty(name="Sun Disc", default=True)

    celestial: EnumProperty(name="Body", default='SUN', items=_items(
        ('SUN', "Sun", "A bright disc with a corona"),
        ('MOON', "Moon", "A disc with a terminator, so it shows a phase")))
    moon_phase: FloatProperty(
        name="Phase", default=0.25, min=0.0, max=1.0,
        description="0 and 1 are new, 0.5 is full. The terminator sweeps "
                    "across the disc between them")
    moon_color: _col("Moon Colour", (0.86, 0.88, 0.95))
    moon_size: FloatProperty(name="Moon Size", default=0.045, min=0.0, max=1.0,
                             subtype='ANGLE')
    moon_earthshine: FloatProperty(
        name="Earthshine", default=0.06, min=0.0, max=1.0,
        description="Faint light on the unlit part of the disc, reflected back "
                    "off the planet")
    sky_mid: _col("Mid Sky", (0.35, 0.50, 0.78))
    sky_mid_height: FloatProperty(name="Mid Height", default=0.35, min=0.01,
                                  max=0.99)
    use_sky_mid: BoolProperty(
        name="Three-Stop Gradient", default=True,
        description="A third colour between horizon and zenith, as Bryce's "
                    "dome gradient allowed")
    atmosphere_density: FloatProperty(
        name="Atmosphere", default=0.0, min=0.0, max=4.0,
        description="Exponential depth haze over the whole dome, on top of the "
                    "horizon band")
    atmosphere_falloff: FloatProperty(name="Atmosphere Falloff", default=1.0,
                                      min=0.01, max=8.0)
    atmosphere_color: _col("Atmosphere Colour", (0.70, 0.78, 0.90))
    haze_blend_sky: FloatProperty(
        name="Blend With Sky", default=0.5, min=0.0, max=1.0,
        description="How much the haze takes the sky's own colour instead of "
                    "its swatch")
    cloud_wind: FloatProperty(
        name="Wind Speed", default=0.0, min=0.0, max=64.0,
        description="Drifts both cloud decks across the sky over time. Stratus "
                    "moves slower, as height dictates")
    cloud_wind_angle: FloatProperty(name="Wind Direction", default=0.0,
                                    min=-6.2832, max=6.2832, subtype='ANGLE')
    cloud_ambience: FloatProperty(
        name="Ambience", default=0.35, min=0.0, max=1.0,
        description="How much sky light fills the shadowed side of a cloud")
    cloud_shadows: FloatProperty(
        name="Cloud Shadows", default=0.0, min=0.0, max=1.0,
        description="Casts the cumulus deck onto the infinite ground below it, "
                    "sampled from the same noise so a shadow always lands "
                    "under a cloud")
    sun_corona: FloatProperty(name="Corona", default=1.0, min=0.0, max=4.0,
                              description="Strength of the wide outer halo")

    haze_color: _col("Haze Colour", (0.82, 0.86, 0.92))
    haze_density: FloatProperty(name="Haze", default=0.45, min=0.0, max=1.0,
                                description="Atmospheric perspective. Thickens "
                                            "toward the horizon")
    haze_height: FloatProperty(name="Haze Height", default=0.22, min=0.01, max=2.0)
    haze_sun_tint: FloatProperty(name="Sun Tint", default=0.5, min=0.0, max=1.0,
                                 description="How much the haze takes the sun's "
                                             "colour when looking toward it")
    fog_color: _col("Fog Colour", (0.90, 0.90, 0.88))
    fog_density: FloatProperty(name="Fog", default=0.0, min=0.0, max=1.0,
                               description="Ground-hugging fog, separate from "
                                           "haze as it was in Bryce")
    fog_height: FloatProperty(name="Fog Height", default=0.05, min=0.005, max=1.0)

    clouds: BoolProperty(name="Cumulus", default=True)
    cloud_color: _col("Cloud Colour", (1.0, 1.0, 1.0))
    cloud_shadow: _col("Cloud Base", (0.42, 0.45, 0.55))
    cloud_cover: FloatProperty(name="Cover", default=0.5, min=0.0, max=1.0)
    cloud_density: FloatProperty(name="Opacity", default=0.95, min=0.0, max=1.0)
    cloud_height: FloatProperty(name="Altitude", default=1.0, min=0.05, max=20.0)
    cloud_scale: FloatProperty(name="Frequency", default=1.4, min=0.01, max=64.0,
                               description="Bryce's Cloud Frequency: larger "
                                           "values make bigger, fewer clouds")
    cloud_detail: IntProperty(name="Detail", default=5, min=1, max=10)
    cloud_softness: FloatProperty(name="Fuzziness", default=1.0, min=0.01, max=8.0)
    cloud_thickness: FloatProperty(name="Thickness", default=0.35, min=0.0, max=2.0,
                                   description="Depth of the deck, which drives "
                                               "the self-shadowing on its base")
    cloud_rim: FloatProperty(name="Sun Rim", default=0.4, min=0.0, max=4.0,
                             description="Bright edge where the sun catches the "
                                         "sunward side of a cloud")
    cloud_seed: IntProperty(name="Seed", default=0, min=0, max=9999)

    stratus: BoolProperty(name="Stratus", default=False)
    stratus_color: _col("Stratus Colour", (0.95, 0.95, 0.98))
    stratus_amount: FloatProperty(name="Cover", default=0.45, min=0.0, max=1.0)
    stratus_density: FloatProperty(name="Opacity", default=0.6, min=0.0, max=1.0)
    stratus_altitude: FloatProperty(name="Altitude", default=3.0, min=0.1, max=40.0)
    stratus_scale: FloatProperty(name="Frequency", default=3.0, min=0.01, max=64.0)
    stratus_detail: IntProperty(name="Detail", default=4, min=1, max=10)
    stratus_sharpness: FloatProperty(name="Fuzziness", default=1.4, min=0.01, max=8.0)
    stratus_squash: FloatProperty(name="Streak", default=1.0, min=0.05, max=8.0,
                                  description="Stretches the layer into wind-blown "
                                              "streaks")

    rainbow: BoolProperty(name="Rainbow", default=False)
    rainbow_intensity: FloatProperty(name="Intensity", default=0.35, min=0.0, max=4.0)
    rainbow_radius: FloatProperty(name="Radius", default=42.0, min=5.0, max=90.0,
                                  description="Degrees from the antisolar point. "
                                              "42 is where a real bow sits")
    rainbow_width: FloatProperty(name="Width", default=3.0, min=0.2, max=20.0)
    rainbow_secondary: FloatProperty(name="Secondary Bow", default=0.5,
                                     min=0.0, max=2.0)
    stars: BoolProperty(name="Stars", default=False)
    star_density: FloatProperty(name="Density", default=0.5, min=0.0, max=1.0)
    star_brightness: FloatProperty(name="Brightness", default=0.8, min=0.0, max=4.0)

    turbidity: FloatProperty(name="Turbidity", default=2.5, min=1.0, max=10.0,
                             description="Atmospheric haziness. 2 is a clear "
                                         "day, 6 is city smog")
    ground_albedo: FloatProperty(name="Ground Albedo", default=0.3, min=0.0, max=1.0)

    ground_plane: BoolProperty(
        name="Infinite Ground", default=False,
        description="An endless plane, intersected analytically rather than "
                    "built from geometry. POV-Ray and Bryce both offered one, "
                    "and it costs nothing per frame however far it reaches")
    ground_mode: EnumProperty(name="Surface", default='SOLID', items=_items(
        ('SOLID', "Solid", "One flat colour"),
        ('CHECKER', "Checker", "The infinite chequerboard of a thousand ray "
                               "tracing demos"),
        ('NOISE', "Fractal", "Two colours mixed by fractal noise, for terrain "
                             "seen from height"),
        ('OCEAN', "Ocean", "Animated waves reflecting the sky, with a Fresnel "
                           "term so it mirrors at glancing angles")))
    ground_height: FloatProperty(name="Height", default=0.0, min=-1e4, max=1e4)
    ground_scale: FloatProperty(name="Scale", default=2.0, min=0.001, max=1e4)
    ground_color2: _col("Second Colour", (0.55, 0.52, 0.48))
    ground_fade: FloatProperty(
        name="Distance Fade", default=60.0, min=0.0, max=1e5,
        description="How far the plane reaches before it has faded into the "
                    "horizon. Without this it reads as a flat sheet rather "
                    "than as ground going away")
    ocean_choppiness: FloatProperty(name="Choppiness", default=0.35, min=0.0,
                                    max=4.0)
    ocean_speed: FloatProperty(name="Wave Speed", default=1.0, min=0.0, max=16.0)
    env_image: PointerProperty(name="Image", type=bpy.types.Image)
    env_mapping: EnumProperty(name="Projection", items=_items(
        ('EQUIRECT', "Equirectangular", "Latitude/longitude panorama"),
        ('MIRRORBALL', "Mirror Ball", "The sphere map of the era"),
        ('SCREEN', "Screen", "")), default='EQUIRECT')
    env_filter: EnumProperty(name="Filter", items=_items(
        ('BILINEAR', "Bilinear", ""), ('NEAREST', "Nearest", "")),
        default='BILINEAR')
    env_tint: _col("Tint", (1.0, 1.0, 1.0))
    sky_blend: BoolProperty(name="Sky Gradient", default=False,
                            options={'HIDDEN'})


CLASSES = (HalcyonSettings, HalcyonMaterialSettings, HalcyonLightSettings,
           HalcyonWorldSettings)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.halcyon = bpy.props.PointerProperty(type=HalcyonSettings)
    bpy.types.Material.halcyon = bpy.props.PointerProperty(
        type=HalcyonMaterialSettings)
    bpy.types.Light.halcyon = bpy.props.PointerProperty(type=HalcyonLightSettings)
    bpy.types.World.halcyon = bpy.props.PointerProperty(type=HalcyonWorldSettings)


def unregister():
    for attr, owner in (('halcyon', bpy.types.World), ('halcyon', bpy.types.Light),
                        ('halcyon', bpy.types.Material), ('halcyon', bpy.types.Scene)):
        if hasattr(owner, attr):
            try:
                delattr(owner, attr)
            except Exception:                                   # noqa: BLE001
                pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
