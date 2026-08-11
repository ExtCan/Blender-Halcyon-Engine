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

from .core.settings import (RESOLUTION_GROUPS, RenderSettings,
                            resolution_description, resolution_label)
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
    ('INTEGER', "Integer",
     "Integer raster coordinates -- vertices land on the whole-pixel grid, "
     "the PS1 wobble at its coarsest. Snapping rounds rather than "
     "truncates, so this shares Fixed 1's grid"),
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
WIRE_MODE = _items(
    ('ALL', "All Edges", "Every triangle edge. What a wireframe renderer drew "
                         "-- and on a dense mesh at a period resolution every "
                         "pixel is within a pixel of an edge, so the surface "
                         "fills in solid"),
    ('CREASE', "Creases & Silhouette",
     "Only the outline and the edges where the surface turns by more than the "
     "angle below. Stays a wireframe however many triangles are behind it"),
)
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
    'debug_pass': DEBUG_PASS,
    'wire_mode': WIRE_MODE,
    # grouped: an item with an empty identifier is a category separator
    'res_preset': [('CUSTOM', "Custom", "")] + [
        item
        for label, keys in RESOLUTION_GROUPS
        for item in [('', label, '')] + [
            (k, resolution_label(k), resolution_description(k)) for k in keys]
    ],
}

# field -> (min, max, soft_min, soft_max, step/precision hints)
RANGES = {
    'outline_width': (1, 8),
    'outline_opacity': (0.0, 1.0),
    'outline_depth_threshold': (0.0005, 1.0),
    'outline_normal_angle': (5.0, 179.0),
    'aa_samples': (1, 64), 'aa_filter_width': (0.1, 4.0),
    'aa_edge_threshold': (0.0, 100.0), 'vertex_snap_grid': (0.05, 16.0),
    'depth_precision': (4, 32), 
    'light_clamp': (0.0, 1000.0), 'ao_distance': (0.001, 1000.0),
    'ao_samples': (1, 256), 'ao_intensity': (0.0, 1.0),
    'radiosity_samples': (1, 64), 'radiosity_distance': (0.01, 1000.0),
    'radiosity_spacing': (1, 8),
    'radiosity_intensity': (0.0, 4.0), 'reflection_blur': (0.0, 45.0),
    'global_ambient_level': (0.0, 10.0), 'max_lights': (0, 64),
    'process_count': (0, 64), 'shadow_map_size': (32, 4096), 'shadow_bias': (0.0, 10.0),
    'shadow_softness': (0.0, 32.0), 'shadow_samples': (1, 64),
    'ray_depth': (0, 16), 'ray_bias': (0.0, 1.0),
    'reflection_blur_samples': (1, 64), 'tex_mip_bias': (-4.0, 4.0),
    'tex_aniso': (1, 16), 'tex_max_size': (0, 4096), 'tex_quantize': (0, 256),
    'tex_affine_subdiv': (0, 64), 'alpha_bits': (1, 8),
    'alpha_threshold': (0.0, 1.0), 'fog_start': (0.0, 100000.0),
    'fog_end': (0.0, 100000.0), 'fog_density': (0.0, 10.0),
    'fog_height_top': (-100000.0, 100000.0),
    'fog_height_falloff': (0.0, 100.0),
    'motion_shutter': (0.0, 4.0), 'motion_steps': (2, 64),
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
    'threads': (0, 64), 'preview_scale': (1, 16),
    'seed': (0, 2 ** 30),
    'wire_width': (0.1, 8.0), 'wire_angle': (0.0, 180.0), 'resolution_x': (1, 16384),
    'resolution_y': (1, 16384), 'pixel_aspect_x': (0.01, 100.0),
    'pixel_aspect_y': (0.01, 100.0), 
    'clip_near_epsilon': (1e-6, 1.0), 'decay_start': (0.0, 10000.0),
}

LABELS = {
    'aa_mode': "Anti-Aliasing", 'aa_samples': "Samples",
    'aa_filter': "Filter", 'aa_filter_width': "Filter Width",
    'aa_edge_threshold': "Edge Depth Threshold",
    'motion_blur': "Motion Blur", 'motion_shutter': "Shutter (frames)",
    'motion_steps': "Blur Steps",
    'fog_height': "Height Fog", 'fog_height_top': "Fog Top",
    'fog_height_falloff': "Height Falloff",
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
    'gpu_shading': "GPU Shading",
    'gpu_raster': "GPU Rasteriser",
    'gpu_hold_context': "Hold GPU Context (freezes UI)",
    'gpu_scissor': "Scissor Layer Passes",
    'viewport_gpu': "Viewport GPU",
    'radiosity': "Radiosity",
    'radiosity_samples': "Gather Samples",
    'radiosity_distance': "Gather Distance",
    'radiosity_intensity': "Bleed Intensity",
    'radiosity_spacing': "Gather Spacing",
    'reflection_blur': "Reflection Blur",
    'reflection_blur_samples': "Blur Samples",
    'watermark': "Burn-In Text",
    'palette_lock': "Lock Palette", 'displacement_scale': "Displacement",
    'color_depth': "Colour Depth", 'palette_mode': "Palette",
    'palette_method': "Quantiser", 'dither': "Dither",
    'color_management': "View Transform", 'crt': "CRT Simulation",
    'composite': "Composite Video", 'jpeg_artifacts': "JPEG Artefacts",
    'output_scale': "Pixel Scale", 'debug_pass': "Render Pass",
    'render_wire': "Wireframe Overlay", 'wire_width': "Wire Width",
    'outline': "Cartoon Outlines", 'outline_color': "Ink Colour",
    'outline_width': "Ink Width", 'outline_opacity': "Ink Opacity",
    'outline_objects': "Object Edges",
    'outline_materials': "Material Edges",
    'outline_depth': "Depth Breaks",
    'outline_depth_threshold': "Depth Threshold",
    'outline_normals': "Creases",
    'outline_normal_angle': "Crease Angle",
    'outline_over_sky': "Ink Silhouettes",
    'wire_color': "Wire Colour", 'wire_mode': "Edges",
    'wire_angle': "Crease Angle",
    'pass_depth': "Depth", 'pass_normal': "Normal",
    'pass_position': "Position", 'pass_uv': "UV",
    'pass_object_index': "Object Index", 'pass_material_index': "Material Index",
}

DESCRIPTIONS = {
    # ---------------------------------------------------------- output
    'res_preset': "Jump the output straight to a period format -- VGA, "
                  "SVGA, broadcast -- with the pixel aspect that format "
                  "actually used, instead of dialling four numbers",
    'resolution_x': "Width of the rendered frame in pixels. Period software "
                    "lived at 320-800; the engine goes as high as you like",
    'resolution_y': "Height of the rendered frame in pixels",
    'pixel_aspect_x': "Horizontal pixel stretch. Broadcast and mode-13h "
                      "formats used non-square pixels; 1.0 is square",
    'pixel_aspect_y': "Vertical pixel stretch. Set with X to reproduce a "
                      "format's true pixel shape; 1.0 is square",
    # ---------------------------------------------------- anti-aliasing
    'aa_mode': "How edges are smoothed. None is the raw hard-edged raster, "
               "Supersample renders larger and filters down (the era's "
               "quality switch), Edge Only spends samples where geometry "
               "ids change, Accumulation averages jittered whole frames",
    'aa_samples': "Samples per pixel for the chosen mode. Supersample "
                  "rounds it to a square (24 renders at 5x5); more is "
                  "smoother and proportionally slower",
    'aa_filter': "The downfilter shape used to combine samples. Box is the "
                 "period default; Tent and Gauss trade a little sharpness "
                 "for calmer edges",
    'aa_filter_width': "Radius of the downfilter in output pixels. Wider "
                       "is softer; 1.0 keeps each pixel to its own samples",
    'motion_blur': "Blur moving objects across the shutter interval by "
                   "averaging time-offset frames, as the era's renderers "
                   "faked it -- expect the cost of Steps extra renders",
    'motion_shutter': "How long the virtual shutter stays open, in frames. "
                      "0.5 is a 180-degree film shutter; longer smears more",
    'motion_steps': "Time samples across the shutter. Few steps show the "
                    "classic stepped ghosting; many approach smooth blur "
                    "at a render each",
    # ------------------------------------------------------- rasteriser
    'backface_cull': "Skip triangles facing away from the camera at the "
                     "raster, as period hardware did. Closed meshes look "
                     "identical and draw faster; open shells lose their "
                     "insides",
    'two_sided_lighting': "Light both faces of every surface: a normal "
                          "facing away from the light flips before shading. "
                          "How the era rendered single-sided geometry "
                          "without black backs",
    'subpixel_precision': "How many fractional bits vertex positions keep "
                          "at the raster. Full precision is modern and "
                          "steady; fewer bits snap vertices to a coarser "
                          "grid and edges wobble as things move -- the "
                          "PS1-era jitter, by its real mechanism",
    'vertex_snap_grid': "Snap screen-space vertices to this fraction of a "
                        "pixel before rasterising. 0 is off; larger steps "
                        "give the polygon shimmer of fixed-point hardware",
    'depth_precision': "Bits of depth buffer. 32 is exact for any scene; "
                       "24 is the period standard; 16 brings the z-fighting "
                       "and poke-through of budget hardware, faithfully",
    'clip_near_epsilon': "Safety margin for the near-plane clip, in NDC "
                         "units. Raise it only if geometry grazing the "
                         "camera shows cracks; lowering it buys nothing",
    # ---------------------------------------------------------- shading
    'default_model': "The reflectance model a material gets when its node "
                     "tree does not choose one -- the engine-wide house "
                     "style. Each entry's tooltip describes its era",
    'force_model': "Override EVERY material's model with one choice for "
                   "this render -- the whole scene as wireframe, flat, or "
                   "chalk in one switch. A diagnostic and a look in itself",
    'shading_rate': "Where lighting is computed: every pixel (Phong-era "
                    "per-pixel), once per vertex and interpolated "
                    "(Gouraud, the hardware look), or once per face "
                    "(flat). The single biggest period-look switch",
    'normal_source': "Which normals shading uses: the mesh's smoothed "
                     "normals, or faceted face normals everywhere -- the "
                     "un-smoothed look of early scanline output",
    'clamp_specular': "Cap each highlight's brightness at this value "
                      "before it saturates the frame. 0 is uncapped; the "
                      "era clamped to keep 8-bit channels from blowing out",
    'light_clamp': "Cap the summed light on any surface point. Keeps "
                   "stacked lights inside the period's dynamic range "
                   "instead of washing to white; 0 is uncapped",
    'ambient_occlusion': "Darken creases and contact areas with hemisphere "
                         "rays -- not period-authentic, but period-adjacent "
                         "grime that reads well on chunky geometry",
    'ao_distance': "How far an occlusion ray looks for nearby geometry. "
                   "Short keeps the effect to contact shadows; long "
                   "darkens whole rooms",
    'ao_samples': "Occlusion rays per pixel. More is smoother and slower; "
                  "8 reads clean at period resolutions",
    'ao_intensity': "Strength of the occlusion darkening. 1 is full "
                    "effect; fractions fade it toward none",
    'global_ambient': "Colour of the light that reaches everything from "
                      "nowhere -- the flat ambient term every 1990s "
                      "renderer had. Tints every unlit area",
    'global_ambient_level': "Multiplier on the global ambient colour. The "
                            "fastest single knob for overall scene "
                            "brightness outside the lights themselves",
    'light_falloff_default': "How light dims with distance when a light "
                             "does not choose: physically correct inverse "
                             "square, the gentler inverse, the era's "
                             "none-at-all, or a custom start/end ramp",
    'light_limit_mode': "Which lights survive when the scene has more than "
                        "the Light Limit: the brightest, the nearest, or "
                        "simply the first -- emulating fixed-function "
                        "hardware's light slots",
    # ---------------------------------------------------------- shadows
    'shadows': "Master switch for all shadowing. Off is the flat, "
               "floating look of the earliest real-time output",
    'shadow_default': "How shadows are computed when a light does not "
                      "choose: depth maps rendered from each light (soft, "
                      "fast, the period standard), traced rays (hard and "
                      "exact), or per-light choice",
    'shadow_map_size': "Resolution of each light's depth map. Small maps "
                       "give blocky period shadows; large maps sharpen "
                       "contact edges. Lights can override individually",
    'shadow_bias': "World-space offset that keeps a surface from shadowing "
                   "itself. Too low shows acne stippling; too high "
                   "detaches shadows from their objects (Peter Panning)",
    'shadow_softness': "Blur radius of mapped shadows, in shadow-map "
                       "texels. 0 is hard-edged; more taps a wider "
                       "neighbourhood for softer penumbras",
    'shadow_samples': "Taps per pixel for soft mapped shadows, or rays per "
                      "pixel for soft traced shadows from area-sized "
                      "lights. More is smoother and slower",
    # ------------------------------------------------------ ray tracing
    'raytrace': "Master switch for traced reflections and refractions -- "
                "the checkbox that separated the raytracers from the "
                "scanliners. Materials still choose their own amounts",
    'ray_depth': "How many times a ray may bounce between mirrors or "
                 "through glass before giving up and taking the sky. "
                 "Two facing mirrors show this number directly",
    'ray_reflection': "Allow traced reflections for materials that ask "
                      "for them. Off, reflective surfaces fall back to "
                      "the environment term",
    'ray_refraction': "Allow traced refraction through transparent "
                      "materials with an IOR. Off, glass becomes simple "
                      "alpha transparency",
    'ray_bias': "How far a spawned ray steps off its surface before "
                "testing the world, in world units. Too low and surfaces "
                "shadow and reflect themselves as speckle; too high and "
                "contact detail goes missing",
    'env_reflection': "Let materials reflect the world background where "
                      "rays are off or exhausted -- the era's spherical "
                      "environment map trick",
    # --------------------------------------------------------- textures
    'tex_filter': "How textures are sampled: Nearest is the chunky texel "
                  "look, Bilinear smooths, Trilinear adds mip blending, "
                  "EWA is the quality path. The single most visible "
                  "texture-era switch",
    'tex_mipmap': "Build and use mip pyramids so distant textures calm "
                  "down instead of sparkling. Off reproduces the shimmer "
                  "of software that never mipped",
    'tex_mip_bias': "Shift mip selection sharper (negative) or blurrier "
                    "(positive), in levels. Period hardware often biased "
                    "sharp and lived with the noise",
    'tex_aniso': "Maximum anisotropy for EWA filtering. Higher keeps "
                 "floors readable at grazing angles; 1 is isotropic",
    'tex_max_size': "Downsample any texture larger than this before use, "
                    "as period VRAM budgets forced. 0 keeps full size",
    'tex_quantize': "Reduce every texture to this many colours before "
                    "sampling -- palettised texture memory, with its "
                    "banding, exactly as shipped games had it",
    'tex_affine_subdiv': "For affine (PS1-style) texture mapping: cut each "
                         "triangle into this many pieces so the warp stays "
                         "bounded. 1 is the full classic wobble",
    'tex_wrap_default': "What textures do past their edges when the node "
                        "does not say: repeat, mirror, clamp, or clip to "
                        "nothing",
    # ----------------------------------------------------- transparency
    'transparency': "How see-through surfaces composite: sorted alpha "
                    "blending, the A-buffer's exact per-pixel lists, or "
                    "Screen Door's dither-pattern holes (no sorting, no "
                    "blending, pure period)",
    'stipple_pattern': "The hole pattern Screen Door transparency punches, "
                       "per opacity level. Each entry is a real pattern "
                       "family from shipped hardware",
    'alpha_bits': "Bits of opacity resolution for the alpha channel. "
                  "Fewer bits step smooth fades into visible bands",
    'alpha_threshold': "Below this opacity a Screen Door pixel is simply "
                       "skipped -- the cutoff that kept near-invisible "
                       "surfaces from costing fill",
    # --------------------------------------------------------------- fog
    'fog': "Master switch for distance fog, the era's draw-distance "
           "disguise and mood in one",
    'fog_mode': "The fog curve: Linear between Start and End, or the two "
                "exponential falls hardware offered. Linear is the most "
                "controllable; Exp2 is the deepest soup",
    'fog_color': "The colour distant geometry fades toward. Matching the "
                 "sky hides the far clip; contrasting it makes fog a look",
    'fog_start': "Distance where linear fog begins, in world units. "
                 "Nearer than this is fully clear",
    'fog_end': "Distance where linear fog saturates; beyond this "
               "everything is fog colour",
    'fog_density': "Steepness of the exponential fog modes. Small numbers "
                   "haze the horizon; large ones swallow the middle ground",
    'fog_vertex': "Compute fog per vertex and interpolate, as fixed-"
                  "function pipelines did -- visible banding on long "
                  "triangles, which is the point",
    'fog_height': "Limit fog to below a world height, thinning with "
                  "altitude -- valley mist instead of uniform soup",
    'fog_height_top': "World height where height fog has fully thinned "
                      "to nothing",
    'fog_height_falloff': "How sharply height fog thins between its base "
                          "and top. Higher hugs the ground",
    # -------------------------------------------------------------- glow
    'glow': "Bloom around bright areas -- the soft television halo every "
            "period FMV had. Threshold picks what counts as bright",
    'glow_threshold': "Brightness above which a pixel feeds the glow. "
                      "Lower pulls midtones into the halo; higher keeps "
                      "glow to highlights",
    'glow_radius': "Size of the halo in pixels at output resolution",
    'glow_intensity': "Strength of the glow added back over the frame",
    'glow_quality': "Blur passes for the halo. More is smoother and "
                    "slower; low counts show the period's boxy bloom",
    'star_filter': "Cross-screen star spikes on bright points, the "
                   "camera-filter look pasted over renders of the era",
    'star_points': "How many spikes each star throws",
    'star_length': "Length of the spikes in pixels",
    'star_rotation': "Rotation of the whole star pattern, in degrees",
    'star_intensity': "Brightness of the star spikes",
    'lens_flare': "Draw a lens flare from the brightest light in frame -- "
                  "ghosts, streak and all. The single most period effect "
                  "there is",
    'flare_intensity': "Overall strength of the flare elements",
    'flare_ghosts': "How many aperture ghosts march across the frame "
                    "opposite the light",
    'flare_streak': "Strength of the horizontal anamorphic streak",
    # ------------------------------------------------------------ colour
    'color_depth': "Bits per pixel of the delivered frame. 24 is "
                   "truecolor; 16 and below quantise with the exact "
                   "banding of the era's framebuffers; HAM modes "
                   "reproduce the Amiga's tricks precisely",
    'palette_mode': "For palettised depths: use a fixed period palette, "
                    "or fit one to this frame the way the era's "
                    "converters did",
    'palette_size': "Number of palette entries when fitting: 256 is VGA, "
                    "16 is EGA territory",
    'palette_method': "How the fitted palette is chosen -- median cut is "
                      "the classic, octree the smoother",
    'dither': "The error-spreading pattern that sells low colour depths: "
              "ordered Bayer matrices in period sizes, or Floyd-"
              "Steinberg diffusion",
    'dither_strength': "How much of the quantisation error the dither "
                       "spreads. 1 is full correction; less shows more "
                       "banding on purpose",
    'exposure': "Linear brightness multiplier applied before the display "
                "curve. 1 is neutral",
    'gamma': "The display curve's exponent. 1 is neutral; below brightens "
             "midtones, above deepens them. Period CRTs lived near 2.2",
    'contrast': "S-curve strength around middle grey. 0 is neutral",
    'saturation': "Colour intensity. 1 is neutral, 0 is greyscale, above "
                  "1 pushes toward the era's oversaturated promos",
    'brightness': "Flat offset added to the frame after exposure. 0 is "
                  "neutral",
    'input_gamma_naive': "Treat texture pixels as already-linear the way "
                         "naive period software did, instead of "
                         "converting from sRGB. Changes every texture's "
                         "midtones; authentic, not correct",
    # ---------------------------------------------------------- CRT & TV
    'crt': "Draw the frame as a CRT would show it: scanlines, phosphor "
           "mask, bloom and curvature, each with its own control",
    'crt_scanlines': "Darkened lines between the picture's rows. The "
                     "strength of the CRT's most recognisable artefact",
    'crt_mask': "The phosphor arrangement simulated: aperture grille, "
                "slot mask or shadow mask",
    'crt_mask_strength': "How visibly the phosphor mask dims the picture",
    'crt_bloom': "How much bright areas bleed into neighbouring "
                 "phosphors",
    'crt_curvature': "Bulge of the simulated glass. 0 is flat",
    'crt_vignette': "Darkening toward the tube's corners",
    'composite': "Push the frame through a simulated composite video "
                 "cable: colour bleed, ringing and dot crawl, the way "
                 "most people actually saw this era",
    'composite_bleed': "How far chroma smears horizontally -- the "
                       "rainbow fringing on sharp colour edges",
    'composite_ringing': "Ghost echoes after hard luma edges, from the "
                         "cable's bandwidth limit",
    'composite_dot_crawl': "The crawling checkerboard on coloured edges "
                           "where chroma and luma interfere",
    'interlace': "Render alternating fields with a one-frame comb offset "
                 "on motion -- broadcast video's signature",
    'jpeg_artifacts': "Recompress the frame as a period JPEG, 8x8 blocks "
                      "and all -- the look of every mid-90s CD-ROM still",
    'jpeg_quality': "The simulated JPEG quality. Lower is blockier",
    'jpeg_passes': "Recompression generations. Each pass compounds the "
                   "damage, as re-saved files did",
    'block_size': "Pixelate the frame into blocks of this size after "
                  "rendering. 1 is off; larger is chunkier",
    'pixel_grid': "Darken the seams between output pixels, as LCD "
                  "previews and some grabbers showed them",
    # ------------------------------------------------------------- misc
    'progressive': "Deliver the frame in coarse-to-fine passes while it "
                   "renders instead of row by row -- the era's preview "
                   "refinement, and a quicker first look now",
    'seed': "Random seed for every stochastic effect: soft shadows, AO, "
            "dither jitter. Same seed, same picture, every time",
    'render_wire': "Draw the mesh's edges over the finished shading -- "
                   "the hidden-line overlay of period modellers, at "
                   "render quality",
    'wire_mode': "Which edges the overlay inks: every triangle edge, "
                 "only marked/creased ones, or silhouette and folds",
    'wire_angle': "For angle-based wire modes: the crease angle in "
                  "degrees above which an edge is inked",
    'wire_color': "Colour of the inked wire edges",
    'wire_width': "Width of the inked edges, in output pixels",
    'outline': "Ink cartoon outlines over the frame, drawn from the "
               "renderer's own buffers -- silhouettes, material borders, "
               "depth breaks and creases, each its own toggle below. At "
               "high anti-aliasing the line smooths beautifully",
    'outline_color': "Colour of the ink. Black for classic cel work; try "
                     "a dark tone of the scene's palette for softer looks",
    'outline_width': "Thickness of the ink in INTERNAL pixels -- under "
                     "Supersample the delivered line is this divided by "
                     "the sample grid, so raise it for heavy AA",
    'outline_opacity': "How solidly the ink covers what is under it. 1 is "
                       "full ink; fractions tint instead of cover",
    'outline_objects': "Ink where one object ends and another begins -- "
                       "the silhouette lines",
    'outline_materials': "Ink where the material changes on a surface, "
                         "even inside one object",
    'outline_depth': "Ink where depth jumps -- edges a silhouette test "
                     "misses when an object overlaps itself",
    'outline_depth_threshold': "How large a depth jump counts, as a "
                               "fraction of the nearer distance. Smaller "
                               "inks more overlaps; larger keeps ink to "
                               "big steps",
    'outline_normals': "Ink creases: edges where the surface turns harder "
                       "than the angle below -- the cube's edges, a "
                       "cylinder's rim",
    'outline_normal_angle': "The crease angle in degrees. Smaller inks "
                            "gentler turns; 90 keeps ink to hard corners",
    'outline_over_sky': "Let the ink land on the background at object "
                        "silhouettes, not only on other surfaces",
    'pass_depth': "Also deliver a Depth pass: each pixel's distance from "
                  "the camera, for compositing",
    'pass_normal': "Also deliver a Normal pass: the shading normal per "
                   "pixel, for relighting tricks",
    'pass_position': "Also deliver a world Position pass per pixel",
    'pass_uv': "Also deliver the UV coordinates per pixel",
    'pass_object_index': "Also deliver each pixel's object index, for "
                         "per-object masks downstream",
    'pass_material_index': "Also deliver each pixel's material index, "
                           "for per-material masks downstream",
    'debug_pass': "Replace the delivered image with one internal buffer "
                  "-- depth, normals, UVs, overdraw, wireframe -- to see "
                  "what the renderer sees",
    # ------------------------------------------------- layered rendering
    'layer_gpu_min_frac': "Depth layers whose pixel share is below this "
                          "fraction shade on the CPU rather than paying a "
                          "GPU pass's fixed costs. 0 sends everything to "
                          "the GPU, 1 keeps every layer on the CPU",
    # ------------------------------------------------------ lens effects
    'lens_vignette_edges': "Darkening toward the frame's corners from the "
                           "simulated lens itself, independent of the CRT "
                           "vignette",
    'shaft_decay': "How quickly volumetric light shafts fade along their "
                   "length. Higher dies faster",
    'shaft_samples': "March steps per pixel for light shafts. More is "
                     "smoother and slower",
    'dof_max_radius': "Cap on the depth-of-field blur circle, in pixels. "
                      "Keeps extreme defocus affordable",
    'radiosity': "One bounce of gathered colour bleed -- the era's "
                 "Radiosity checkbox. Rays that see sky return the ambient "
                 "colour; rays that land on a surface return its flat "
                 "diffuse. Supersedes plain Ambient Occlusion while on",
    'radiosity_samples': "Hemisphere rays gathered per pixel. 8 is the "
                         "period look; more is smoother and slower",
    'radiosity_distance': "How far a gather ray reaches before it counts "
                          "as seeing sky",
    'radiosity_intensity': "Strength of the colour a surface lends its "
                           "neighbours",
    'radiosity_spacing': "Gather every Nth pixel and blend between the "
                         "points -- the interpolated mode the era "
                         "actually shipped. 1 gathers every pixel; 2 "
                         "casts a quarter of the rays, 4 a sixteenth",
    'reflection_blur': "Cone angle in degrees for blurry (glossy) ray "
                       "reflections. 0 keeps mirrors sharp. Blurry frames "
                       "shade on the CPU -- the deferred pass traces one "
                       "ray per fragment",
    'reflection_blur_samples': "Jittered rays averaged per reflective "
                               "fragment when Reflection Blur is above 0",
    'watermark': "Burn a line of text into the corner of the final frame, "
                 "the VTR way. Tokens: %F frame number, %R resolution, "
                 "%V engine version, %D date, %T time of day, %S render "
                 "time, %B Blender version. Ink colours: &%r red, "
                 "&%g green, &%b blue, &%y yellow, &%c cyan, &%m magenta, "
                 "&%w white again -- each colours everything after it. "
                 "Empty means no burn-in",
    'viewport_gpu': "Let the viewport preview use the GPU device (when the "
                    "top switch is GPU). Turn OFF to force every viewport "
                    "frame onto the CPU while F12 keeps the driver -- the "
                    "bisect switch for viewport-only driver problems: if a "
                    "glitch follows this toggle, it lives in the viewport's "
                    "GPU path; if it stays, it never did",

    'spot_cones': "Draw the visible beam of every spot light whose Volumetric "
                  "value is above zero. The view ray is intersected with the "
                  "cone and a few samples are summed along whatever falls "
                  "inside it -- which is what LightWave and 3D Studio did, "
                  "rather than integrating a volume",
    'spot_cone_samples': "Samples along each view ray. Low counts band, and "
                         "the banding is the period artefact rather than a "
                         "defect -- 8 to 16 is where those renderers sat",
    'spot_cone_density': "Overall strength of every cone. Each light's own "
                         "Volumetric value scales it further",
    'spot_cone_falloff': "How fast scattering fades with distance from the "
                         "lamp. 2 is inverse-square; lower carries further",
    'spot_cone_reach': "How far a beam is drawn when nothing stops it",
    'threads': "How many threads share the shading work. Measured on a 20-core "
               "machine this is neutral at best and about 3% slower at worst, "
               "because NumPy releases the interpreter lock only for large "
               "array operations and the node evaluator is dominated by Python "
               "dispatch between small ones. Defaults to 1 for that reason. "
               "Worker Processes is the route that actually parallelises",
    'render_device': "Where the frame is computed. GPU rasterises the frame, "
                     "shades it and runs the post stages on your driver -- "
                     "each measured against the CPU frame on real hardware, "
                     "and each falling back per frame with the reason on the "
                     "console when a scene uses something still CPU-only "
                     "(ray tracing, for now). Flipping to GPU turns the "
                     "proven stages on; the Debug panel can switch them off "
                     "individually",
    'gpu_post': "Run the parallel post stages on the GPU through Blender's own "
                "gpu module, the layer EEVEE is built on. Measured against "
                "the CPU stages on real hardware. Falls back to the CPU stage by "
                "stage, and says why on the console",
    'gpu_hold_context': "Give the render thread the GPU context for the "
                        "whole frame, as before 1.25.53. Bursts start "
                        "instantly but Blender's interface cannot redraw "
                        "until the frame ends (Not Responding on long "
                        "renders). Off, the interface stays live and each "
                        "GPU burst runs on the main thread instead. Takes "
                        "effect from the next render",
    'gpu_scissor': "Limit each transparent depth layer's GPU passes and "
                   "readbacks to the layer's own bounding box. The same "
                   "pixels shade either way -- the self test proves the "
                   "two paths identical on your driver -- but a sparse "
                   "layer stops paying for the whole frame. Turn off only "
                   "if a driver disagrees with scissored reads; the "
                   "picture is then the proven full-frame path",
    'gpu_shading': "Shade the frame on the GPU: the G-buffer is shaded in "
                   "one full-screen pass per material -- shadow maps, image "
                   "textures, converted master-shader materials, normal-map "
                   "chains, coded shader nodes, the period pattern "
                   "textures, matcap, backface and environment reflections "
                   "all included, with sun, point, spot and area lights. "
                   "Measured against the CPU frame at 0.00002 max "
                   "difference on real hardware. Frames using what the GLSL "
                   "does not reproduce yet -- ray-traced shadows, fog, ray "
                   "tracing, the N64 texture filter, the Bump node -- "
                   "shade on the CPU with the reason on the console. "
                   "Shader compiles, plans and unchanged uploads are all "
                   "cached across frames",
    'gpu_raster': "Rasterise the frame on the GPU too: a compute-shader "
                  "port of the CPU rasteriser's own fill rules, one thread "
                  "per pixel, measured at ZERO differing pixels against the "
                  "CPU on real hardware and several times faster at "
                  "working sizes. Frames it cannot reproduce yet -- "
                  "Painter's depth sort, affine texture mode, overdraw "
                  "debugging, banded worker renders -- rasterise on the "
                  "CPU with the reason on the console",
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
    'aa_edge_threshold': "Depth step, in scene units, that counts as a crease "
                         "worth smoothing. Edge AA always softens polygon "
                         "silhouettes; this adds interior depth breaks. 0 "
                         "smooths silhouettes only",
    'ray_shadows': "Master switch for ray-traced shadows. Off, lights set to "
                   "trace (and lights with no shadow map to fall back on) "
                   "cast no shadow at all",
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
            extra = {}
            if name == 'render_device':
                # the top-of-panel switch MEANS it: flipping to GPU turns
                # the proven stages on, so a scene saved back when they
                # defaulted off cannot silently render a 10-second CPU
                # frame under a switch that says GPU. Flipping to CPU
                # leaves them alone; the Debug toggles still opt out
                def _device_flip(self, _context):
                    if str(self.render_device).upper() == 'GPU':
                        self.gpu_raster = True
                        self.gpu_shading = True
                        self.gpu_post = True
                extra['update'] = _device_flip
            props[name] = EnumProperty(name=label, description=desc,
                                       items=ENUMS[name],
                                       default=default if any(
                                           i[0] == default for i in ENUMS[name])
                                       else ENUMS[name][0][0], **extra)
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


# The preset list is import-time static, so the enum is too. It used to be
# a dynamic callback, and its first entry is a category HEADER ('', ...):
# a dynamic enum's unset value is index 0, which resolved to the header's
# empty identifier and made Blender log "current value '0' matches no enum"
# on EVERY redraw of the presets panel -- the field's console flood. A
# static list with an explicit default names a real entry from the start.
# (Kept at module level: Blender requires static enum item strings to
# outlive the property.)
_PRESET_ITEMS = preset_items()


class HalcyonSettings(PropertyGroup):
    """All Halcyon render settings, mirroring core.settings.RenderSettings."""

    __annotations__ = _build()

    preset: EnumProperty(
        name="Preset",
        description="Load the settings of a specific 1990s renderer or machine",
        items=_PRESET_ITEMS,
        default='DEFAULT',
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
    shadow_map_size: IntProperty(
        name="Map Size", default=0, min=0, max=4096,
        description="Shadow map resolution for this light. 0 uses the "
                    "render setting's Shadow Map Size")
    shadow_bias: FloatProperty(
        name="Bias", default=0.0, min=0.0, max=10.0,
        description="Depth offset that stops a surface shadowing itself. "
                    "0 uses the render setting's Shadow Bias")
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
    cookie: PointerProperty(
        name="Projected Texture", type=bpy.types.Image,
        description="An image this lamp projects -- the gobo/cookie of the "
                    "sixth-generation consoles. A Spot throws it through its "
                    "cone like a slide projector (Splinter Cell's window "
                    "patterns); a Sun tiles it across the world as a cloud "
                    "shadow. Point and Area lamps ignore it")
    cookie_strength: FloatProperty(
        name="Projection Strength", default=1.0, min=0.0, max=1.0,
        description="Blend between plain light (0) and the fully projected "
                    "image (1)")
    cookie_scale: FloatProperty(
        name="Projection Scale", default=10.0, min=0.01,
        description="Sun only: world size of one tile of the projected "
                    "image, in scene units")


def _col(name, default, desc=''):
    return FloatVectorProperty(name=name, description=desc, subtype='COLOR',
                               size=3, default=default, min=0.0, max=1.0)


# --------------------------------------------------------------- sky library
#
# The bpy-touching half of the sky preset system. `presets/skies.py` stays
# bpy-free because the renderer reads it; finding the folder Blender keeps
# user presets in does not belong there.


def sky_library_dir():
    """Where saved skies live, alongside Blender's own presets.

    Never fatal: a build that cannot hand out a scripts folder should cost the
    saved-sky list, not the whole World panel.
    """
    import os
    try:
        path = bpy.utils.user_resource('SCRIPTS', path="presets/halcyon_skies",
                                       create=True)
    except Exception:                                           # noqa: BLE001
        return None
    return path if path and os.path.isdir(path) else None


def user_skies():
    """(path, label) for every sky saved into the library, sorted."""
    import os
    out = []
    folder = sky_library_dir()
    if not folder:
        return out
    try:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith('.halsky'):
                out.append((os.path.join(folder, name),
                            os.path.splitext(name)[0]))
    except OSError:
        pass
    return out


#: Blender does not keep a reference to the strings an items callback returns,
#: and a garbage-collected enum item is a corrupted menu. The list is held here
#: for as long as the property might read it.
_SKY_ITEMS = []


_SKY_PREVIEWS = [None]


def sky_previews():
    """The thumbnail collection, loaded lazily from presets/thumbs/.

    Every built-in sky ships a rendered thumbnail (the engine's own sky
    module drew them). Never fatal: without bpy.utils.previews the enum
    simply has no pictures.
    """
    if _SKY_PREVIEWS[0] is not None:
        return _SKY_PREVIEWS[0]
    import os
    try:
        import bpy.utils.previews
        pcoll = bpy.utils.previews.new()
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'halcyon', 'presets', 'thumbs')
        if not os.path.isdir(base):
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'presets', 'thumbs')
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.endswith('.png'):
                    key = name[:-4]
                    pcoll.load(key, os.path.join(base, name), 'IMAGE')
        _SKY_PREVIEWS[0] = pcoll
    except Exception:                                           # noqa: BLE001
        _SKY_PREVIEWS[0] = {}
    return _SKY_PREVIEWS[0]


def sky_preset_items(self=None, context=None):
    from .presets.skies import sky_items
    pcoll = sky_previews()
    items = []
    for i, (k, label, note) in enumerate(sky_items()):
        icon = 0
        try:
            if pcoll and k in pcoll:
                icon = pcoll[k].icon_id
        except Exception:                                       # noqa: BLE001
            icon = 0
        items.append((k, label, note, icon, i) if icon else (k, label, note))
    saved = user_skies()
    if saved:
        items.append(('', "Saved Skies", ''))
        for path, label in saved:
            items.append(('FILE:' + path, label, "A sky you saved"))
    # a mixed 3/5-tuple list confuses Blender: normalise to 5-tuples
    norm = []
    for j, it in enumerate(items):
        if len(it) == 3:
            norm.append((it[0], it[1], it[2], 0, len(sky_items()) + j))
        else:
            norm.append(it)
    _SKY_ITEMS[:] = norm
    return _SKY_ITEMS


def water_library_dir():
    """Where saved waters live. Never fatal, for the same reason as skies."""
    import os
    try:
        path = bpy.utils.user_resource('SCRIPTS', path="presets/halcyon_waters",
                                       create=True)
    except Exception:                                           # noqa: BLE001
        return None
    return path if path and os.path.isdir(path) else None


def user_waters():
    """(path, label) for every water saved into the library, sorted."""
    import os
    out = []
    folder = water_library_dir()
    if not folder:
        return out
    try:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith('.halwater'):
                out.append((os.path.join(folder, name),
                            os.path.splitext(name)[0]))
    except OSError:
        pass
    return out


_WATER_ITEMS = []


def water_preset_items(self=None, context=None):
    from .presets.waters import water_items
    items = [(k, label, note) for k, label, note in water_items()]
    saved = user_waters()
    if saved:
        items.append(('', "Saved Waters", ''))
        for path, label in saved:
            items.append(('FILE:' + path, label, "A water you saved"))
    _WATER_ITEMS[:] = items
    return _WATER_ITEMS


class HalcyonWorldSettings(PropertyGroup):
    sky_preset: EnumProperty(
        name="Sky Preset", items=sky_preset_items,
        description="Pick a sky, then press Apply Preset. Choosing one here "
                    "does not change anything on its own")
    water_preset: EnumProperty(
        name="Water Preset", items=water_preset_items,
        description="Pick a water, then press Apply Preset. Choosing one here "
                    "does not change anything on its own. Waters and skies "
                    "are separate libraries and neither overwrites the other")

    mode: EnumProperty(name="Sky", default='NODES', items=_items(
        ('NODES', "Use Node Tree", "Evaluate the world's own shader nodes"),
        ('SOLID', "Solid Colour", "A single flat background colour"),
        ('GRADIENT', "Gradient", "Horizon to zenith blend, with an optional ground"),
        ('BANDS', "Banded Gradient",
         "The same blend cut into flat steps, the way a 256-colour palette "
         "could only ever render one"),
        ('STARFIELD', "Starfield",
         "Space: a flat backdrop, stars all the way round, optional nebula"),
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

    # ------------------------------------------------------ Bryce's Sky Lab
    sky_mode: EnumProperty(name="Sky Mode", default='CUSTOM', items=_items(
        ('SOFT', "Soft Sky",
         "Bryce's default: the horizon is derived from the sun's glow colour "
         "and only the dome is set by hand, which is why every Bryce sky "
         "warmed toward the sun without anybody choosing to"),
        ('CUSTOM', "Custom Sky", "All three gradient stops set directly")))
    sun_glow_color: _col("Sun Glow Colour", (1.0, 0.86, 0.62),
                         "The corona's own colour, which Bryce kept separate "
                         "from the sun's light colour")
    shadow_color: _col("Shadow Colour", (0.30, 0.34, 0.45),
                       "What the shaded side of a cloud is tinted toward")
    shadow_intensity: FloatProperty(name="Shadow Intensity", default=1.0,
                                    min=0.0, max=1.0)
    fog_base_height: FloatProperty(
        name="Fog Base Height", default=0.0, min=-1.0, max=1.0,
        description="Where the fog bank starts. Below this it is solid, above "
                    "it falls away over the fog height")
    haze_base_height: FloatProperty(
        name="Haze Base Height", default=0.0, min=-1.0, max=1.0,
        description="Where the haze starts. The Sky Lab has this and the Sky "
                    "& Fog palette does not")
    fog_blend_sky: FloatProperty(name="Fog Blend With Sky", default=0.0,
                                 min=0.0, max=1.0)
    fog_sun_tint: FloatProperty(name="Fog Blend With Sun", default=0.0,
                                min=0.0, max=1.0)
    color_perspective: FloatProperty(
        name="Colour Perspective", default=0.0, min=0.0, max=4.0,
        description="How fast distance takes the haze colour. Bryce applied "
                    "this to everything in the scene, not only to the sky")
    volumetric_world: FloatProperty(
        name="Volumetric World", default=0.0, min=0.0, max=4.0,
        description="Shafts of sunlight through the atmosphere, which in Bryce "
                    "was the render-time setting nobody left on by accident")
    cloud_frequency: FloatProperty(
        name="Frequency", default=1.0, min=0.05, max=16.0,
        description="How tight the cloud pattern is. Bryce's own control name")
    cloud_amplitude: FloatProperty(
        name="Amplitude", default=1.0, min=0.0, max=4.0,
        description="How far the pattern swings either side of the cover "
                    "threshold -- which separates billows without changing how "
                    "much sky is covered")
    cloud_turbulence: FloatProperty(name="Turbulence", default=1.0, min=0.05,
                                    max=2.0)
    spherical_clouds: BoolProperty(
        name="Spherical Clouds", default=True,
        description="On, clouds stay puffy out to the horizon; off, they "
                    "stretch into streaks, which is what the switch did")
    link_clouds_to_view: BoolProperty(
        name="Link Clouds to View", default=True,
        description="Keep the cloud pattern fixed relative to the camera, so "
                    "moving does not change which clouds are where. Turn it "
                    "off for real parallax -- but note that Cloud Height is a "
                    "dome parameter rather than a distance, so a low deck "
                    "slides a long way for a small move")
    fixed_cloud_plane: BoolProperty(name="Fixed Cloud Plane", default=True)
    stratus_frequency: FloatProperty(name="Stratus Frequency", default=1.0,
                                     min=0.05, max=16.0)
    stratus_amplitude: FloatProperty(name="Stratus Amplitude", default=1.0,
                                     min=0.0, max=4.0)
    moon_softness: FloatProperty(
        name="Softness", default=0.05, min=0.0, max=1.0,
        description="How hard the terminator is across the moon's disc")
    comets: FloatProperty(
        name="Comet Intensity", default=0.0, min=0.0, max=4.0,
        description="Bryce put comets in the Celestial tab, and they are the "
                    "reason its night skies were never just a starfield")
    comet_count: IntProperty(name="Comets", default=3, min=1, max=32)
    comet_speed: FloatProperty(
        name="Comet Speed", default=0.05, min=0.0, max=4.0,
        description="How fast each comet runs around its own great circle, in "
                    "radians of sky per second. At zero they stand still, as "
                    "they used to. They all start where a still frame puts "
                    "them, so turning this up never empties the first frame")
    comet_length: FloatProperty(
        name="Tail Length", default=0.10, min=0.005, max=1.5,
        description="How far the tail reaches behind the head, as a fraction "
                    "of the sky. Each comet varies either side of it")
    comet_width: FloatProperty(
        name="Tail Width", default=0.006, min=0.0005, max=0.2,
        description="How wide the tail is at the head. It flares out from "
                    "there and dims as it goes, which is the shape a comet's "
                    "dust tail actually has")
    comet_tail_sun: FloatProperty(
        name="Tail Direction", default=0.6, min=0.0, max=1.0,
        description="0 trails the comet's own path, the way a dust tail does; "
                    "1 points straight away from the sun, the way an ion tail "
                    "does. Real comets show both at once, so the truth is in "
                    "between")
    comet_color: _col("Comet Colour", (1.0, 0.96, 0.88))

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
    star_size: FloatProperty(
        name="Star Size", default=0.35, min=0.01, max=1.0,
        description="Diameter of a star within its cell. Small values give "
                    "single-pixel points, which is what these looked like")
    star_twinkle: FloatProperty(name="Twinkle", default=0.0, min=0.0, max=1.0,
                                description="Animated flicker, per star")
    nebula: FloatProperty(
        name="Nebula", default=0.0, min=0.0, max=4.0,
        description="Turbulent cloud behind the stars. Zero leaves plain space")
    nebula_color: _col("Nebula Colour", (0.35, 0.15, 0.55))
    nebula_scale: FloatProperty(name="Nebula Scale", default=2.0, min=0.05,
                                max=32.0)
    nebula_detail: IntProperty(name="Nebula Detail", default=5, min=1, max=10)

    band_count: IntProperty(
        name="Bands", default=8, min=1, max=64,
        description="How many flat steps the gradient is cut into. A 256-colour "
                    "machine could spare about this many for the sky")
    band_softness: FloatProperty(
        name="Softness", default=0.0, min=0.0, max=1.0,
        description="Rounds the step edges. 0 is the hard band the hardware "
                    "actually gave you")

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
        ('GRID', "Neon Grid", "Glowing gridlines to the horizon -- the "
                              "synthwave floor. The second colour is the "
                              "line glow"),
        ('TILES', "Tiles", "Square tiles with grout and per-tile shading. "
                           "The second colour is the grout"),
        ('DESERT', "Dunes", "Wind-ribbed sand ridges warped by noise, the "
                            "second colour on the crests"),
        ('SNOW', "Snowfield", "A bright field with blue-shadowed hollows "
                              "and sparse sun glints"),
        ('LAVA', "Lava", "Dark crust over glowing cracks; the second "
                         "colour is the heat, pulsing slowly"),
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
    ocean_wind_angle: FloatProperty(name="Wind Direction", default=0.6,
                                    min=-6.2832, max=6.2832, subtype='ANGLE',
                                    description="Waves run mostly with the wind")
    ocean_spread: FloatProperty(
        name="Spread", default=0.6, min=0.0, max=1.0,
        description="How far the shorter waves fan off the wind. 0 is a "
                    "regular swell, 1 is confused chop")
    ocean_wave_scale: FloatProperty(
        name="Wave Size", default=1.0, min=0.02, max=500.0, soft_min=0.05,
        soft_max=40.0, subtype='DISTANCE', unit='LENGTH',
        description="The length of the longest wave train, crest to crest. "
                    "The shorter trains are fractions of it, so this sets the "
                    "size of the whole sea at once. It no longer multiplies "
                    "the ground Scale, which is the chequerboard's and has "
                    "nothing to do with water")
    ocean_detail: IntProperty(name="Wave Detail", default=5, min=1, max=10,
                              description="How many wave trains are summed")
    ocean_sparkle: FloatProperty(
        name="Horizon Shimmer", default=1.0, min=0.0, max=1.0,
        description="Where a pixel covers many waves, take the sample from a "
                    "random point inside it rather than its centre. Sampling "
                    "the centre makes the wave trains beat against the pixel "
                    "grid and fills the distance with moire fringes; this "
                    "turns the same detail into the fine shimmer a Bryce "
                    "ocean has. Turn it down to see the fringes")
    ocean_horizon_smooth: FloatProperty(
        name="Horizon Smoothing", default=0.0, min=0.0, max=1.0,
        description="Fades out waves too small for a pixel to draw cleanly. "
                    "Bryce did none of this -- its water kept every wave to "
                    "the horizon and compressed them into a band of shimmer, "
                    "which is what its pictures look like. Turn this up for "
                    "smoother distant water, at the cost of it going to glass")
    ocean_deep: _col("Deep Colour", (0.03, 0.09, 0.13))
    ocean_shallow: _col("Shallow Colour", (0.06, 0.22, 0.26))
    ocean_glitter: FloatProperty(
        name="Sun Glitter", default=1.0, min=0.0, max=16.0,
        description="The sun's reflection smeared down the wave slopes. Waves "
                    "too small to draw widen the path rather than vanishing, "
                    "which is what makes it spread toward the horizon")
    ocean_glitter_size: FloatProperty(name="Glitter Width", default=0.45,
                                      min=0.01, max=4.0)
    ocean_foam: FloatProperty(
        name="Foam", default=0.0, min=0.0, max=1.0,
        description="Off by default: Bryce had no foam control, and adding one "
                    "unasked would be inventing a feature it did not have")
    ocean_foam_color: _col("Foam Colour", (0.92, 0.95, 0.96))
    ocean_transparency: FloatProperty(name="Transparency", default=0.25,
                                      min=0.0, max=1.0)

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


#: tooltips for group properties declared inline above -- patched into the
#: deferred property definitions below, so every control in every panel
#: carries a real explanation without repeating boilerplate at each site
GROUP_DOCS = {
    'HalcyonSettings': {
        'ui_tab': "Which page of Halcyon's settings this panel shows. "
                  "Purely a UI switch; it changes nothing in the render",
    },
    'HalcyonMaterialSettings': {
        'model': "The reflectance model this override shades with -- each "
                 "entry's tooltip names its era and character",
        'diffuse': "The surface's base colour under white light",
        'diffuse_level': "How strongly the diffuse term contributes. 0 "
                         "kills the base colour entirely",
        'specular': "Colour of the highlight. Plastics keep it white; "
                    "period metals tinted it to fake conductor response",
        'specular_level': "Brightness of the highlight. 0 is fully matte",
        'glossiness': "Tightness of the highlight: low is a broad sheen, "
                      "high is a small hard sparkle. The classic Phong "
                      "exponent, under its 3D Studio name",
        'ambient_level': "How much of the scene's ambient light this "
                         "surface accepts. The era's per-material shadow "
                         "filler",
        'emission': "Light the surface gives off by itself, unaffected by "
                    "any lamp -- the Self-Illumination of the period",
        'emission_level': "Multiplier on the emission colour",
        'opacity': "How solid the surface is. Below 1 it composites "
                   "through the transparency mode the render settings "
                   "chose",
        'ior': "Index of refraction for traced glass. 1.0 bends nothing; "
               "1.45 is glass; 1.33 water",
        'roughness': "Micro-surface roughness for the models that read it "
                     "(Oren-Nayar, Cook-Torrance, Minnaert)",
        'anisotropy': "Stretches the highlight along one direction, for "
                      "brushed metal. 0 is round",
        'aniso_rotation': "Rotates the stretched highlight's direction",
        'metallic': "Blends the surface toward conductor behaviour: "
                    "diffuse falls away and reflections take the base "
                    "colour",
        'reflect_level': "How much traced mirror reflection the surface "
                         "adds. 0 is none; it costs rays above that",
        'two_sided': "Shade back faces as if they were front faces, so "
                     "open geometry has no black inside",
        'shadeless': "Skip lighting entirely and show the diffuse colour "
                     "flat -- the CONSTANT model by another switch",
        'cast_shadow': "Whether this surface blocks light on its way to "
                       "other surfaces",
        'receive_shadow': "Whether other objects' shadows darken this "
                          "surface",
        'wire': "Draw this material's triangle edges over its shading, "
                "per-material rather than scene-wide",
        'wire_size': "Width of this material's inked edges, in rendered "
                     "pixels",
    },
    'HalcyonLightSettings': {
        'decay': "How this light dims with distance: physically correct "
                 "inverse square, the gentler inverse, the era's none, or "
                 "a custom start/end ramp",
        'decay_start': "Distance where the custom ramp starts dimming. "
                       "Closer than this is full brightness",
        'decay_end': "Distance where the custom ramp reaches zero",
        'shadow': "This light's own shadow method: depth map, traced "
                  "rays, or none -- overriding the scene default when "
                  "the scene is set to Per Light",
        'shadow_softness': "Blur of this light's mapped shadow, in "
                           "shadow-map texels. 0 is hard",
        'shadow_samples': "Taps or rays per pixel for this light's soft "
                          "shadows",
        'shadow_density': "How dark this light's shadows get. 1 is full "
                          "occlusion; less lets colour bleed through, as "
                          "the era's fake fill lights did",
        'shadow_color': "Colour inside this light's shadows instead of "
                        "black -- the tinted-shadow trick of period art",
        'diffuse_only': "This light affects only the diffuse term, "
                        "leaving highlights untouched",
        'specular_only': "This light affects only highlights, adding "
                         "sparkle without lifting the surface",
        'ambient_only': "This light adds flat ambient everywhere instead "
                        "of directional light",
        'exclude_mode': "Whether the collection below is kept OUT of this "
                        "light, or is the only thing kept IN it",
    },
    'HalcyonWorldSettings': {
        'mode': "What the sky IS: the material node tree, a flat colour, "
                "a gradient, banded steps, a starfield, the Bryce sky "
                "lab, a physical atmosphere, or an HDRI image",
        'strength': "Multiplier on everything the sky contributes -- "
                    "background, ambient and reflections together",
        'ambient': "Colour of the light the world adds from every "
                   "direction, independent of any lamp",
        'ambient_level': "Multiplier on the world's ambient colour",
        'color': "The flat background colour in Solid mode",
        'horizon': "Sky colour at the horizon in Gradient and Bands "
                   "modes",
        'zenith': "Sky colour straight up in Gradient and Bands modes",
        'ground_color': "Colour below the horizon when Show Ground is on",
        'horizon_height': "Vertical position of the horizon line, as the "
                          "ray direction's Z. 0 is level with the camera",
        'blend_mode': "The shape of the horizon-to-zenith blend: linear, "
                      "smooth, sharp or eased",
        'sky_mode': "Bryce's Sky Mode: Soft Sky derives the dome from the "
                    "sun's own colour; Custom Sky exposes the three "
                    "colour stops directly",
        'shadow_intensity': "Bryce's shadow-strength slider: how dark the "
                            "sky-lab lighting draws its shadows",
        'fog_blend_sky': "How much the fog band takes its colour from the "
                         "sky rather than its own",
        'fog_sun_tint': "How much the fog band warms toward the sun's "
                        "colour near the sun",
        'cloud_turbulence': "How hard the cloud noise is folded. Bryce's "
                            "third cloud control: higher is stormier",
        'fixed_cloud_plane': "Anchor the cloud pattern to the world "
                             "rather than the view, so orbiting the "
                             "camera does not slide the deck",
        'stratus_frequency': "Tightness of the stratus layer's pattern",
        'stratus_amplitude': "Contrast swing of the stratus layer about "
                             "its cover threshold",
        'comet_count': "How many comets streak the celestial sphere",
        'comet_color': "Colour of the comet heads and tails",
        'sun_elevation': "Height of the sun above the horizon, in "
                         "radians. Near 0 is sunset; 1.57 is overhead",
        'sun_rotation': "Compass direction of the sun, in radians",
        'sun_color': "Colour of the sun disc and the light it throws "
                     "into the sky model",
        'sun_size': "Angular size of the visible sun disc",
        'sun_intensity': "Brightness of the sun's contribution to the "
                         "dome",
        'sun_disc': "Whether the sun itself is drawn, or only its light",
        'celestial': "Master switch for the celestial layer: moon, "
                     "stars, comets",
        'moon_color': "Colour of the moon disc drawn on the night dome",
        'moon_size': "Angular size of the moon disc",
        'sky_mid': "Custom Sky's middle colour stop, between horizon "
                   "and zenith",
        'sky_mid_height': "Where the middle stop sits between horizon "
                          "(0) and zenith (1)",
        'atmosphere_falloff': "How quickly the sky colour transitions "
                              "happen with altitude. Higher hugs the "
                              "horizon",
        'atmosphere_color': "Overall atmospheric tint layered over the "
                            "dome",
        'cloud_wind_angle': "Compass direction the cloud decks drift, "
                            "in radians",
        'haze_color': "Colour of the horizon haze band",
        'haze_height': "Vertical thickness of the haze band",
        'fog_color': "Colour of the Bryce fog band at the horizon",
        'fog_height': "Vertical thickness of the Bryce fog band",
        'clouds': "Master switch for the cumulus cloud deck",
        'cloud_color': "Colour of the cumulus deck's sunlit faces",
        'cloud_shadow': "How darkly the deck shades its own undersides",
        'cloud_cover': "How much of the sky the cumulus deck covers. "
                       "Bryce's cover slider",
        'cloud_density': "How solid each cloud reads against the sky "
                         "behind it",
        'cloud_height': "Altitude of the cumulus deck on the dome. "
                        "Lower decks race past; higher ones sit still",
        'cloud_detail': "Noise octaves in the cloud pattern. More is "
                        "crinklier and slower",
        'cloud_softness': "Softness of the cloud edges against the sky",
        'cloud_seed': "Random seed for the cloud pattern. Change it for "
                      "a different sky with the same settings",
        'stratus': "Master switch for the high thin stratus layer",
        'stratus_color': "Colour of the stratus wisps",
        'stratus_amount': "Coverage of the stratus layer",
        'stratus_density': "How solid the stratus wisps read",
        'stratus_altitude': "Altitude of the stratus layer on the dome",
        'stratus_scale': "Size of the stratus features",
        'stratus_detail': "Noise octaves in the stratus pattern",
        'stratus_sharpness': "Edge hardness of the stratus wisps",
        'rainbow': "Draw a rainbow opposite the sun, as Bryce could",
        'rainbow_intensity': "Brightness of the rainbow arc",
        'rainbow_width': "Angular width of the rainbow band",
        'rainbow_secondary': "Strength of the fainter, colour-reversed "
                             "outer bow",
        'stars': "Master switch for the star layer of night skies",
        'star_density': "How many stars fill the sphere",
        'star_brightness': "Brightness of the star points",
        'nebula_color': "Colour of the faint nebula wash behind the "
                        "stars",
        'nebula_scale': "Size of the nebula's billows",
        'nebula_detail': "Noise octaves in the nebula wash",
        'ground_albedo': "How much light the physical atmosphere's "
                         "ground bounces back into the sky",
        'ground_mode': "What the infinite ground plane is made of -- "
                       "each entry is its own material with its own "
                       "controls",
        'ground_height': "World height of the infinite ground plane",
        'ground_scale': "Feature size of the ground material's pattern",
        'ground_color2': "The ground material's secondary colour, where "
                         "its pattern uses one",
        'ocean_deep': "Water colour looking into deep water",
        'ocean_shallow': "Water colour near the surface and crests",
        'ocean_glitter_size': "Size of the sun-glitter sparkles on the "
                              "water",
        'ocean_foam_color': "Colour of the foam along wave crests",
        'ocean_transparency': "How much the water lets the sky's "
                              "reflection give way to its own colour",
        'ocean_choppiness': "How steep and broken the waves are",
        'ocean_speed': "How fast the waves animate over frames",
        'env_image': "The image used as the world in HDRI mode",
        'env_mapping': "How the image wraps the sphere: equirectangular, "
                       "mirror ball, or screen-locked",
        'env_filter': "Filtering used when sampling the environment "
                      "image",
        'env_tint': "Colour multiplied over the environment image",
        'sky_blend': "Blend the lower sky toward the horizon colour for "
                     "a softer meeting with the ground",
    },
}


def _apply_group_docs():
    """Patch GROUP_DOCS into the deferred property definitions.

    Works on both the real bpy (property functions return a deferred with
    `.keywords`) and the test fake (`_Prop.kw`), and only fills holes --
    an inline description written at the definition site always wins.
    """
    for cls in (HalcyonSettings, HalcyonMaterialSettings,
                HalcyonLightSettings, HalcyonWorldSettings):
        docs = GROUP_DOCS.get(cls.__name__) or {}
        for pname, desc in docs.items():
            ann = cls.__annotations__.get(pname)
            if ann is None:
                continue
            for attr in ('keywords', 'kw'):
                kw = getattr(ann, attr, None)
                if isinstance(kw, dict) and not kw.get('description'):
                    kw['description'] = desc


_apply_group_docs()


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
