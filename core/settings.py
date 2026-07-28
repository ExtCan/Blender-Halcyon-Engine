"""RenderSettings -- the complete knob set, as a plain dataclass.

The Blender PropertyGroup in halcyon/properties.py mirrors this field-for-field
and copies across at export time. Keeping it here means the renderer, the tests
and the preset files all speak the same language.
"""

from dataclasses import dataclass, fields
from typing import Tuple


@dataclass
class RenderSettings:
    # ---------------------------------------------------------------- output
    max_transparent_layers: int = 16
    fast_background: bool = True
    cache_shadows: bool = True
    gpu_post: bool = False
    use_processes: bool = False
    process_count: int = 0
    displacement_scale: float = 1.0
    lens_distortion: float = 0.0
    chromatic_aberration: float = 0.0
    lens_vignette_edges: bool = True
    shaft_threshold: float = 0.85
    shaft_length: float = 0.6
    shaft_decay: float = 0.92
    shaft_samples: int = 24
    dof: bool = False
    dof_focus: float = 8.0
    dof_amount: float = 1.0
    dof_layers: int = 5
    dof_max_radius: float = 24.0
    palette_lock: bool = True
    film_transparent: bool = False
    res_preset: str = 'CUSTOM'
    resolution_x: int = 320
    resolution_y: int = 240
    pixel_aspect_x: float = 1.0
    pixel_aspect_y: float = 1.0
    # ------------------------------------------------------------- sampling
    aa_mode: str = 'SUPERSAMPLE'      # NONE | SUPERSAMPLE | EDGE | ACCUMULATE
    aa_samples: int = 1               # supersample factor (1..8)
    aa_filter: str = 'BOX'            # BOX | TRIANGLE | GAUSS | CATROM | MITCHELL
    aa_filter_width: float = 1.0
    aa_edge_threshold: float = 0.1
    jitter: bool = False
    jitter_seed: int = 0
    # ------------------------------------------------------------- geometry
    backface_cull: bool = False
    two_sided_lighting: bool = True
    subpixel_precision: str = 'FLOAT'  # FLOAT | FIXED_4 | FIXED_1 | INTEGER
    vertex_snap: bool = False          # PlayStation-style vertex jitter
    vertex_snap_grid: float = 1.0      # in pixels of the *output* image
    depth_precision: int = 24          # z-buffer bits (8..32); low = fighting
    depth_sort: str = 'ZBUFFER'
    painters_key: str = 'CENTROID'        # ZBUFFER | PAINTERS | ZBUFFER_NOWRITE
    polygon_offset: float = 0.0
    clip_near_epsilon: float = 1e-4
    # ------------------------------------------------------------- shading
    default_model: str = 'PHONG'
    force_model: str = 'NONE'          # override every material's model
    shading_rate: str = 'PIXEL'        # PIXEL | VERTEX (global Gouraud) | FACE
    normal_source: str = 'AUTO'        # AUTO | SPLIT | FACE | VERTEX
    specular_in_gamma: bool = True     # 90s renderers lit in display space
    clamp_specular: bool = True
    light_clamp: float = 0.0           # 0 = off
    ambient_occlusion: bool = False
    ao_distance: float = 1.0
    ao_samples: int = 8
    ao_intensity: float = 1.0
    # ------------------------------------------------------------- lighting
    global_ambient: Tuple[float, float, float] = (0.05, 0.05, 0.06)
    global_ambient_level: float = 1.0
    light_falloff_default: str = 'INVERSE_SQUARE'
    max_lights: int = 8                # hardware-style light limit; 0 = no limit
    light_limit_mode: str = 'BRIGHTEST'  # BRIGHTEST | NEAREST | FIRST
    shadows: bool = True
    shadow_default: str = 'MAP'
    shadow_map_size: int = 512
    shadow_bias: float = 0.02
    shadow_softness: float = 1.0
    shadow_samples: int = 4
    # ------------------------------------------------------------ raytracing
    raytrace: bool = False
    ray_depth: int = 2
    ray_reflection: bool = True
    ray_refraction: bool = True
    ray_shadows: bool = True
    ray_bias: float = 1e-3
    reflection_blur_samples: int = 1
    env_reflection: bool = True        # sphere-map reflections when no rays
    # ------------------------------------------------------------- textures
    tex_filter: str = 'NEAREST'        # NEAREST | BILINEAR | TRILINEAR | N64_3POINT
    tex_mipmap: bool = False
    tex_mip_bias: float = 0.0
    tex_aniso: int = 1
    tex_max_size: int = 0              # 0 = unlimited; else clamp to N (power of 2)
    tex_quantize: int = 0              # 0 = off; else colours per texture
    tex_perspective: bool = True       # False = affine mapping (PS1 warp)
    tex_affine_subdiv: int = 0         # affine correction subdivision, 0 = none
    tex_wrap_default: str = 'REPEAT'
    # ---------------------------------------------------------- transparency
    transparency: str = 'SORTED'       # NONE | STIPPLE | SORTED | ABUFFER
    stipple_pattern: str = 'BAYER4'
    alpha_bits: int = 8                # 1 = binary stencil alpha
    alpha_threshold: float = 0.5
    # --------------------------------------------------------------- depth cue
    fog: bool = False
    fog_mode: str = 'LINEAR'           # LINEAR | EXP | EXP2 | TABLE16
    fog_color: Tuple[float, float, float] = (0.5, 0.55, 0.65)
    fog_start: float = 5.0
    fog_end: float = 40.0
    fog_density: float = 0.05
    fog_vertex: bool = False           # per-vertex fog (Voodoo/PS1 style)
    # ------------------------------------------------------------- post: glow
    glow: bool = False
    glow_threshold: float = 0.85
    glow_radius: float = 12.0
    glow_intensity: float = 0.6
    glow_quality: str = 'GAUSS'        # GAUSS | BOX | KAWASE
    star_filter: bool = False
    star_points: int = 4
    star_length: float = 30.0
    star_rotation: float = 0.0
    star_intensity: float = 0.5
    lens_flare: bool = False
    flare_intensity: float = 0.5
    flare_ghosts: int = 5
    flare_streak: float = 0.4
    # ------------------------------------------------------- post: colour depth
    color_depth: str = '24'            # 32 | 24 | 16 | 15 | 12 | 8 | 4 | 1 | HAM8 | HAM6
    palette_mode: str = 'ADAPTIVE'     # ADAPTIVE | FIXED_666 | WEB216 | VGA256 | \
                                       # MAC256 | WIN20 | EGA16 | CGA4 | GRAY | CUSTOM
    palette_size: int = 256
    palette_method: str = 'MEDIAN_CUT'  # MEDIAN_CUT | OCTREE | POPULARITY | KMEANS
    palette_dither_first: bool = True
    dither: str = 'NONE'               # NONE | BAYER2 | BAYER4 | BAYER8 | FLOYD | \
                                       # JJN | STUCKI | ATKINSON | BURKES | SIERRA | \
                                       # SIERRA_LITE | NOISE | HALFTONE
    dither_strength: float = 1.0
    dither_serpentine: bool = True
    # ---------------------------------------------------- post: display / CRT
    exposure: float = 1.0
    gamma: float = 1.0
    contrast: float = 0.0
    saturation: float = 1.0
    brightness: float = 0.0
    color_management: str = 'NONE'     # NONE (naive 90s) | SRGB | FILMIC_OFF
    input_gamma_naive: bool = True     # skip sRGB->linear on textures
    crt: bool = False
    crt_scanlines: float = 0.0
    crt_mask: str = 'NONE'             # NONE | APERTURE | SLOT | SHADOW
    crt_mask_strength: float = 0.4
    crt_bloom: float = 0.0
    crt_curvature: float = 0.0
    crt_vignette: float = 0.0
    composite: bool = False            # NTSC composite artefacting
    composite_bleed: float = 0.5
    composite_ringing: float = 0.3
    composite_dot_crawl: float = 0.0
    interlace: str = 'NONE'            # NONE | ODD | EVEN | FIELD_RENDER
    # ------------------------------------------------------ post: compression
    jpeg_artifacts: bool = False
    jpeg_quality: int = 60
    jpeg_passes: int = 1
    block_size: int = 8
    # ------------------------------------------------------------ post: scale
    output_scale: str = 'NONE'         # NONE | NEAREST_2X | NEAREST_3X | NEAREST_4X
    pixel_grid: bool = False
    # ------------------------------------------------------------ performance
    tile_size: int = 128
    threads: int = 0                   # 0 = auto
    bucket_order: str = 'HILBERT'      # HILBERT | TOP | CENTER | RANDOM
    preview_scale: int = 4
    max_texture_memory: int = 0
    progressive: bool = True
    # ------------------------------------------------------------------ misc
    seed: int = 0
    render_wire: bool = False
    wire_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wire_width: float = 1.0
    show_stats: bool = False
    watermark: str = ''
    debug_pass: str = 'BEAUTY'         # BEAUTY | DEPTH | NORMAL | UV | MATID | \
                                       # DIFFUSE | SPECULAR | AMBIENT | SHADOW | \
                                       # OVERDRAW | WIREFRAME

    # ------------------------------------------------------------------ utils
    def copy(self):
        return RenderSettings(**{f.name: getattr(self, f.name) for f in fields(self)})

    def apply(self, d):
        """Apply a preset dict; unknown keys are ignored (forward compatible)."""
        known = {f.name for f in fields(self)}
        for k, v in d.items():
            if k in known:
                setattr(self, k, v)
        return self

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


RESOLUTION_PRESETS = {
    # label: (x, y, aspect_x, aspect_y)
    'CGA':          (320, 200, 1.0, 1.2),
    'VGA_13H':      (320, 200, 1.0, 1.2),
    'QVGA':         (320, 240, 1.0, 1.0),
    'VGA':          (640, 480, 1.0, 1.0),
    'MAC_CLASSIC':  (512, 342, 1.0, 1.0),
    'MAC_13':       (640, 480, 1.0, 1.0),
    'SVGA':         (800, 600, 1.0, 1.0),
    'XGA':          (1024, 768, 1.0, 1.0),
    'AMIGA_PAL':    (320, 256, 1.0, 1.0),
    'AMIGA_HIRES':  (640, 512, 1.0, 1.0),
    'NTSC_D1':      (720, 486, 10.0, 11.0),
    'PAL_D1':       (720, 576, 59.0, 54.0),
    'NTSC_TOASTER': (752, 480, 10.0, 11.0),
    'PSX':          (320, 240, 1.0, 1.0),
    'PSX_HI':       (512, 240, 1.0, 2.0),
    'N64':          (320, 240, 1.0, 1.0),
    'QUAKE':        (320, 200, 1.0, 1.2),
}
