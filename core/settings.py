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
    spot_cones: bool = False
    spot_cone_density: float = 1.0
    spot_cone_samples: int = 12
    spot_cone_falloff: float = 2.0
    spot_cone_reach: float = 64.0
    render_device: str = 'CPU'
    # all three act only when render_device is GPU: the top-of-panel switch
    # is the choice, and these default on so choosing GPU means the GPU.
    # Each stays a Debug-panel toggle for opting back out per stage
    gpu_post: bool = True
    gpu_shading: bool = True
    gpu_raster: bool = True
    # hold the GPU context on the render thread for the whole frame (the
    # pre-1.25.53 behaviour: fastest possible bursts, but Blender's
    # interface cannot draw until the frame ends). Off, the interface
    # stays live and GPU bursts are marshalled to the main thread
    gpu_hold_context: bool = False
    # A-buffer rank routing: a depth layer whose fragment count falls
    # below this fraction of the frame's pixels shades on the proven
    # per-rank CPU path even when GPU shading is on. The GPU pays
    # full-frame FIXED costs per layer (a draw and a readback-sync
    # cover every pixel whether three fragments live there or a
    # million); the CPU pays per FRAGMENT. Routing is by WHOLE layer,
    # never splitting one, so each path receives complete ranks and
    # the per-rank fields both paths build are exactly the ones the
    # pure runs build. 0.02 is set FROM the first field routing line
    # (1.25.60, 16 layers, 9.4M fragments): ~0.4s of fixed driver cost
    # per layer against ~3.7us per fragment on 20 CPU cores puts the
    # break-even near 3% of the frame -- 2% keeps a safety margin, so
    # a routed layer is still a near-certain win. 0.0 keeps every
    # layer on the GPU. Internal for now -- the printed routing line
    # is the dial's evidence before it earns a panel row.
    layer_gpu_min_frac: float = 0.02
    # scissor the per-layer GPU passes (and their readbacks) to each
    # depth layer's own bounding box. Pure transport: the same pixels
    # shade either way, and the self test proves the two paths
    # bit-identical on the driver. The toggle exists because scissored
    # reads are a newer driver path than full-frame texture reads --
    # if a driver ever disagrees, turn it off and the picture is the
    # proven full-frame one
    gpu_scissor: bool = True
    # the viewport's own device gate, for BISECTING field problems: OFF
    # forces every viewport frame onto the CPU while F12 keeps the switch
    viewport_gpu: bool = True
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
    # accumulation motion blur: the frame renders motion_steps times across
    # the shutter (re-exported at each subframe) and averages -- the
    # accumulation-buffer trails of the era, paid for honestly at N frames
    motion_blur: bool = False
    motion_shutter: float = 0.5        # shutter open time, in frames
    motion_steps: int = 5
    # ------------------------------------------------------------- geometry
    backface_cull: bool = False
    two_sided_lighting: bool = True
    subpixel_precision: str = 'FLOAT'  # FLOAT | FIXED_4 | FIXED_1 | INTEGER
    vertex_snap: bool = False          # PlayStation-style vertex jitter
    vertex_snap_grid: float = 1.0      # in pixels of the *output* image
    depth_precision: int = 24          # z-buffer bits (8..32); low = fighting
    depth_sort: str = 'ZBUFFER'        # ZBUFFER | PAINTERS
    painters_key: str = 'CENTROID'     # CENTROID | NEAREST | FARTHEST
    # the camera raster's near-plane epsilon. 1e-5 IS the value
    # the rasterisers have always run; the setting used to claim
    # 1e-4 and drive nothing (found by the settings audit)
    clip_near_epsilon: float = 1e-5
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
    # one-bounce gathered ambient: the era's "Radiosity" checkbox
    # (LightWave 5.6, MAX, POV-Ray 3). Replaces the flat ambient term with
    # a hemisphere gather: rays that see sky return the ambient colour,
    # rays that hit a surface return that surface's flat diffuse -- colour
    # bleed. Supersedes plain AO while on (the gather IS occlusion-aware).
    radiosity: bool = False
    radiosity_samples: int = 8
    radiosity_distance: float = 3.0
    radiosity_intensity: float = 1.0
    #: the era's INTERPOLATED radiosity (LightWave's shipping mode): gather
    #: on a sparse pixel grid and blend between the points. 1 gathers every
    #: pixel (the 1.25.95 behaviour, exact); 2 casts a quarter of the rays,
    #: 4 a sixteenth. Softly blurred bleed -- which is what the period's
    #: radiosity looked like -- and the same picture on either device.
    radiosity_spacing: int = 2
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
    #: blurry (glossy) reflections: cone half-angle in DEGREES. 0 keeps
    #: mirror reflections; above 0 each reflective fragment averages
    #: `reflection_blur_samples` jittered rays -- LightWave's Reflection
    #: Blurring, MAX's raytrace blur. The samples slider shipped in
    #: 1.25.4x and was read by NOTHING until this pair was completed.
    reflection_blur: float = 0.0
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
    # height fog: the fog thins with world height above fog_height_top --
    # the layered ground mist the sixth-generation consoles drew
    fog_height: bool = False
    fog_height_top: float = 2.0        # world Z where the fog starts thinning
    fog_height_falloff: float = 0.5    # how fast it thins per unit above
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
    threads: int = 1                   # 0 = auto
    preview_scale: int = 4
    progressive: bool = True
    # ------------------------------------------------------------------ misc
    seed: int = 0
    render_wire: bool = False
    # ALL draws every triangle edge, which is what the model always did and
    # what fills a dense mesh solid; CREASE draws only silhouettes and edges
    # where the surface actually turns, which stays a wireframe however many
    # triangles are behind it
    wire_mode: str = 'ALL'            # ALL | CREASE
    wire_angle: float = 25.0
    wire_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wire_width: float = 1.0
    # ------------------------------------------------- cartoon outlines
    # ink drawn from the G-buffer's own boundaries -- object ids, material
    # ids, depth breaks, normal creases -- at the internal resolution, so
    # supersampling anti-aliases the line on the way down
    outline: bool = False
    outline_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    outline_width: int = 1
    outline_opacity: float = 1.0
    outline_objects: bool = True
    outline_materials: bool = False
    outline_depth: bool = True
    outline_depth_threshold: float = 0.02
    outline_normals: bool = True
    outline_normal_angle: float = 60.0
    outline_over_sky: bool = True
    show_stats: bool = False
    watermark: str = ''
    # extra outputs written alongside the beauty image, for the compositor
    pass_depth: bool = False
    pass_normal: bool = False
    pass_position: bool = False
    pass_uv: bool = False
    pass_object_index: bool = False
    pass_material_index: bool = False
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
    # key: (x, y, aspect_x, aspect_y)
    #
    # Pixel aspects follow the format's own standard where one exists (D1 and
    # DV are 10:11 / 59:54 by SMPTE, the CIF family is 12:11 by H.261, SVCD
    # 15:11 / 59:36 by its spec); where no standard names a number, the aspect
    # is whatever fills a 4:3 tube with the mode's pixel grid, which is what
    # the hardware actually did. Every key that ever shipped stays: keys are
    # enum identifiers and operator arguments saved inside .blend files.

    # --- televisions and broadcast
    'NTSC_SQ':      (640, 480, 1.0, 1.0),
    'PAL_SQ':       (768, 576, 1.0, 1.0),
    'NTSC_D1':      (720, 486, 10.0, 11.0),
    'PAL_D1':       (720, 576, 59.0, 54.0),
    'NTSC_D1_WIDE': (720, 486, 40.0, 33.0),
    'PAL_D1_WIDE':  (720, 576, 118.0, 81.0),
    'NTSC_TOASTER': (752, 480, 10.0, 11.0),
    'HDTV_720':     (1280, 720, 1.0, 1.0),
    'HDTV_1080':    (1920, 1080, 1.0, 1.0),

    # --- computer monitors
    'HERCULES':     (720, 348, 1.0, 1.55),
    'EGA':          (640, 350, 1.0, 1.37),
    'QVGA':         (320, 240, 1.0, 1.0),
    'VGA':          (640, 480, 1.0, 1.0),
    'SVGA':         (800, 600, 1.0, 1.0),
    'XGA':          (1024, 768, 1.0, 1.0),
    'SXGA':         (1280, 1024, 1.0, 1.0),
    'UXGA':         (1600, 1200, 1.0, 1.0),
    'SUN_WS':       (1152, 900, 1.0, 1.0),
    'NEXT_MEGAPIXEL': (1120, 832, 1.0, 1.0),
    'MAC_13':       (640, 480, 1.0, 1.0),
    'MAC_16':       (832, 624, 1.0, 1.0),
    'MAC_PORTRAIT': (640, 870, 1.0, 1.0),
    'MAC_TWO_PAGE': (1152, 870, 1.0, 1.0),

    # --- home computers
    'CGA':          (320, 200, 1.0, 1.2),
    'VGA_13H':      (320, 200, 1.0, 1.2),
    'QUAKE':        (320, 200, 1.0, 1.2),
    'MAC_CLASSIC':  (512, 342, 1.0, 1.0),
    'ATARI_ST':     (320, 200, 1.0, 1.2),
    'AMIGA_NTSC':   (320, 200, 1.0, 1.2),
    'AMIGA_PAL':    (320, 256, 1.0, 1.0),
    'AMIGA_HIRES':  (640, 512, 1.0, 1.0),

    # --- game consoles
    'SNES':         (256, 224, 8.0, 7.0),
    'GENESIS':      (320, 224, 32.0, 35.0),
    'SATURN':       (352, 240, 10.0, 11.0),
    'PSX':          (320, 240, 1.0, 1.0),
    'PSX_HI':       (512, 240, 1.0, 2.0),
    'N64':          (320, 240, 1.0, 1.0),
    'N64_HI':       (640, 480, 1.0, 1.0),
    'DREAMCAST':    (640, 480, 1.0, 1.0),
    'GAMECUBE':     (640, 480, 1.0, 1.0),
    'PS2':          (640, 448, 14.0, 15.0),
    'XBOX':         (640, 480, 1.0, 1.0),

    # --- video formats
    'QCIF':         (176, 144, 12.0, 11.0),
    'CIF':          (352, 288, 12.0, 11.0),
    'VCD_NTSC':     (352, 240, 10.0, 11.0),
    'VCD_PAL':      (352, 288, 12.0, 11.0),
    'SVCD_NTSC':    (480, 480, 15.0, 11.0),
    'SVCD_PAL':     (480, 576, 59.0, 36.0),
    'DV_NTSC':      (720, 480, 10.0, 11.0),
    'DV_PAL':       (720, 576, 59.0, 54.0),
    'QUICKTIME_160': (160, 120, 1.0, 1.0),

    # --- pictures and textures
    'QUICKTAKE':    (640, 480, 1.0, 1.0),
    'DC120':        (1280, 960, 1.0, 1.0),
    'PHOTOCD_BASE': (768, 512, 1.0, 1.0),
    'PHOTOCD_4BASE': (1536, 1024, 1.0, 1.0),
    'PHOTOCD_16BASE': (3072, 2048, 1.0, 1.0),
    'TEXTURE_128':  (128, 128, 1.0, 1.0),
    'TEXTURE_256':  (256, 256, 1.0, 1.0),
    'TEXTURE_512':  (512, 512, 1.0, 1.0),
}

#: the categories the UI shows, in order. Every RESOLUTION_PRESETS key appears
#: in exactly one group -- the test suite holds the two tables to each other.
RESOLUTION_GROUPS = (
    ("Televisions", ('NTSC_SQ', 'PAL_SQ', 'NTSC_D1', 'PAL_D1',
                     'NTSC_D1_WIDE', 'PAL_D1_WIDE', 'NTSC_TOASTER',
                     'HDTV_720', 'HDTV_1080')),
    ("Computer Monitors", ('HERCULES', 'EGA', 'QVGA', 'VGA', 'SVGA', 'XGA',
                           'SXGA', 'UXGA', 'SUN_WS', 'NEXT_MEGAPIXEL',
                           'MAC_13', 'MAC_16', 'MAC_PORTRAIT',
                           'MAC_TWO_PAGE')),
    ("Home Computers", ('CGA', 'VGA_13H', 'QUAKE', 'MAC_CLASSIC', 'ATARI_ST',
                        'AMIGA_NTSC', 'AMIGA_PAL', 'AMIGA_HIRES')),
    ("Game Consoles", ('SNES', 'GENESIS', 'SATURN', 'PSX', 'PSX_HI', 'N64',
                       'N64_HI', 'DREAMCAST', 'GAMECUBE', 'PS2', 'XBOX')),
    ("Video Formats", ('QCIF', 'CIF', 'VCD_NTSC', 'VCD_PAL', 'SVCD_NTSC',
                       'SVCD_PAL', 'DV_NTSC', 'DV_PAL', 'QUICKTIME_160')),
    ("Pictures & Textures", ('QUICKTAKE', 'DC120', 'PHOTOCD_BASE',
                             'PHOTOCD_4BASE', 'PHOTOCD_16BASE', 'TEXTURE_128',
                             'TEXTURE_256', 'TEXTURE_512')),
)

#: display names where Title Case of the key would be wrong or unhelpful
RESOLUTION_LABELS = {
    'NTSC_SQ': "NTSC Square Pixel", 'PAL_SQ': "PAL Square Pixel",
    'NTSC_D1': "NTSC D1", 'PAL_D1': "PAL D1",
    'NTSC_D1_WIDE': "NTSC D1 Widescreen", 'PAL_D1_WIDE': "PAL D1 Widescreen",
    'NTSC_TOASTER': "Video Toaster NTSC",
    'HDTV_720': "HDTV 720p", 'HDTV_1080': "HDTV 1080",
    'HERCULES': "Hercules Mono", 'EGA': "EGA", 'QVGA': "QVGA / VGA Mode X",
    'VGA': "VGA", 'SVGA': "Super VGA", 'XGA': "XGA", 'SXGA': "SXGA",
    'UXGA': "UXGA", 'SUN_WS': "Sun Workstation",
    'NEXT_MEGAPIXEL': "NeXT MegaPixel",
    'MAC_13': "Mac 13\" RGB", 'MAC_16': "Mac 16\"",
    'MAC_PORTRAIT': "Mac Portrait", 'MAC_TWO_PAGE': "Mac Two-Page",
    'CGA': "CGA", 'VGA_13H': "VGA Mode 13h", 'QUAKE': "Quake / DOS Games",
    'MAC_CLASSIC': "Mac Classic", 'ATARI_ST': "Atari ST Low",
    'AMIGA_NTSC': "Amiga NTSC Lores", 'AMIGA_PAL': "Amiga PAL Lores",
    'AMIGA_HIRES': "Amiga PAL Hires",
    'SNES': "Super NES", 'GENESIS': "Genesis / Mega Drive",
    'SATURN': "Saturn", 'PSX': "PlayStation", 'PSX_HI': "PlayStation Hi-Res",
    'N64': "Nintendo 64", 'N64_HI': "Nintendo 64 Hi-Res",
    'DREAMCAST': "Dreamcast", 'GAMECUBE': "GameCube",
    'PS2': "PlayStation 2", 'XBOX': "Xbox",
    'QCIF': "QCIF Videophone", 'CIF': "CIF Videoconference",
    'VCD_NTSC': "Video CD NTSC", 'VCD_PAL': "Video CD PAL",
    'SVCD_NTSC': "Super Video CD NTSC", 'SVCD_PAL': "Super Video CD PAL",
    'DV_NTSC': "DV NTSC", 'DV_PAL': "DV PAL",
    'QUICKTIME_160': "QuickTime Web Movie",
    'QUICKTAKE': "Apple QuickTake", 'DC120': "Kodak DC120 Megapixel",
    'PHOTOCD_BASE': "Photo CD Base", 'PHOTOCD_4BASE': "Photo CD 4Base",
    'PHOTOCD_16BASE': "Photo CD 16Base",
    'TEXTURE_128': "Game Texture 128", 'TEXTURE_256': "Game Texture 256",
    'TEXTURE_512': "Game Texture 512",
}


def resolution_label(key):
    return RESOLUTION_LABELS.get(key, key.replace('_', ' ').title())


def resolution_description(key):
    x, y, ax, ay = RESOLUTION_PRESETS[key]
    if ax == ay:
        return f"{x}x{y}"
    return f"{x}x{y}, {ax:g}:{ay:g} pixels"
