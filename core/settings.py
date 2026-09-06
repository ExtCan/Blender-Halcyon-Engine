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
    # R222: real marched volume containers (a material whose output
    # links a Volume chain). Steps per ray; low counts band, and the
    # banding is the era's own slicing artefact
    volume_steps: int = 48
    volume_shadows: bool = True
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
    aa_mode: str = 'SUPERSAMPLE'      # NONE | SUPERSAMPLE | EDGE | ADAPTIVE | ACCUMULATE
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
    # ------------------------------------------------------------- stereo
    # parallel cameras with an off-axis (asymmetric-frustum) shift, so
    # the convergence plane sits at zero parallax with no vertical error
    stereo_mode: str = 'NONE'          # NONE | ANAGLYPH | SBS | CROSS
    stereo_eye_distance: float = 0.065
    stereo_convergence: float = 8.0
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
    sss: bool = True                   # BI's R_SSS master switch
    # workflow, not shading: the append watch (classic lamps fixed the
    # moment File > Append brings them in). Read by append_watch.py.
    auto_fix_appended_lamps: bool = True
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
    # R204: the hard alpha test now defaults OFF. At its old default of
    # 0.5 the cutoff applied in EVERY transparency mode, so a glass at
    # 0.4 opacity simply vanished from Sorted and A-Buffer frames --
    # blended transparency should blend. Raise it for classic cut-out
    # alpha (foliage cards, chain-link fences).
    alpha_threshold: float = 0.0
    # --------------------------------------------------------------- depth cue
    fog: bool = False
    fog_mode: str = 'LINEAR'           # LINEAR | EXP | EXP2 | TABLE16
    fog_color: Tuple[float, float, float] = (0.5, 0.55, 0.65)
    fog_start: float = 5.0
    fog_end: float = 40.0
    fog_density: float = 0.05
    fog_vertex: bool = False           # per-vertex fog (Voodoo/PS1 style)
    # R223: banded depth fog -- transmittance quantized to cel steps
    # (0 = off; 2+ = the anime painted-planes distance look)
    fog_bands: int = 0
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
    #: R202: CUSTOM palette mode's colours, extracted from the picked
    #: palette image at export (tuple of (r,g,b) floats, luma-sorted,
    #: deterministic). Empty = no image picked; CUSTOM then behaves as
    #: it always did (adaptive), so nothing existing changes.
    palette_colors: tuple = ()
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
    orbit_scale: int = 0               # motion drafts: 0 = auto budget,
    #                                    N = fixed 1/N of the region
    progressive: bool = True
    material_override: str = 'NONE'    # NONE | CLAY: every material a
    #                                    plain matte for test renders
    override_color: tuple = (0.7, 0.7, 0.7)
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
    # R220: draw the artist's own marked edges (Freestyle marks, sharp,
    # creases) as interior ink, hidden-line-removed against the z-buffer
    outline_marked: bool = False
    # R227: the ink style pack -- every default is the mask road's own
    # look, so an untouched scene never enters the distance-field road
    ink_style: str = 'CLEAN'          # CLEAN | BRUSH | PENCIL
    ink_reference_height: int = 0     # 0 = widths in internal pixels
    ink_taper: float = 0.0            # thick near, thin far
    ink_interior_scale: float = 1.0   # creases / material breaks / marks
    ink_weight_noise: float = 0.0     # the hand's pressure along the line
    ink_weight_scale: float = 24.0    # its screen scale, pixels
    ink_shadow_side: float = 0.0      # thicker on the key lamp's shadow side
    ink_boil: float = 0.0             # displacement amplitude, pixels
    ink_boil_fps: int = 12            # the boil clock (12 = on twos at 24)
    ink_boil_scale: float = 18.0      # displacement noise scale, pixels
    ink_pencil_strokes: int = 3
    ink_pencil_spread: float = 1.5
    ink_grain: float = 0.0
    ink_color_mode: str = 'FIXED'     # FIXED | FILL | GRADIENT
    ink_fill_darken: float = 0.45
    ink_color2: Tuple[float, float, float] = (0.35, 0.1, 0.45)
    ink_gradient: str = 'DEPTH'       # DEPTH | VERTICAL | LIGHT
    # R231: the drawn line -- stroke ends, roughness, drift, gaps, texture
    ink_end_taper: float = 0.0        # the line thins toward stroke ends
    ink_end_length: float = 12.0      # over this many pixels
    ink_roughness: float = 0.0        # edge irregularity, in patches
    ink_roughness_scale: float = 6.0  # its coarsest scale, pixels
    ink_drift: float = 0.0            # the hand drifts off the contour, px
    ink_gaps: float = 0.0             # dry-brush skips
    ink_texture: str = 'SOLID'        # SOLID | STREAKS | CHARCOAL
    ink_texture_amount: float = 0.6   # how much of the texture
    # R234: the inker's line (core/lines.py) -- three more line sources
    # and the Shadow Level they and the isophote weight share
    outline_form: bool = False        # the valleys of the facing
    outline_form_threshold: float = 0.3
    outline_shadow: bool = False      # the terminator, inked
    outline_shadow_level: float = 0.1  # n.l where the light stops
    outline_tone: bool = False        # flow-guided DoG of the shaded frame
    outline_tone_threshold: float = 0.15
    # ... the isophote weight, the stroke road, the surface anchor
    ink_isophote: float = 0.0         # Goodwin's thick-and-thin by the light
    ink_isophote_range: float = 24.0  # the walk that earns the full width, px
    ink_smooth: float = 0.0           # chain smoothing, pixels (sigma)
    ink_pressure: float = 0.0         # heavier through curves, a pause at corners
    ink_overshoot: float = 0.0        # lines run past ends and junctions, px
    ink_anchor: str = 'SCREEN'        # SCREEN | SURFACE
    # R230: the era looks -- the cel photographed and printed (core/film.py)
    film_grade: str = 'NONE'          # NONE | TECHNICOLOR | EASTMAN_70S | TV_80S | VHS | MONO
    film_grade_amount: float = 1.0    # how much of the stock's response
    film_softness: float = 0.0        # optical softness, sigma in pixels
    film_weave: float = 0.0           # gate weave amplitude, pixels
    film_dust: float = 0.0            # dust and hairs on the print, density
    film_grain: float = 0.0           # emulsion grain
    film_flicker: float = 0.0         # projector flicker
    film_hold: int = 1                # shoot on ones (1), twos (2), threes (3)
    film_halftone: float = 0.0        # the print's dot screen, amount
    film_halftone_pitch: float = 6.0  # dot cell, pixels
    film_misregister: float = 0.0     # the cel laid off its pegs, pixels
    film_bleed: float = 0.0           # paint soaking under the line, pixels
    # R236: the colour PROCESS (core/film.colour_process) -- the records,
    # the curve to dye, the dyes' impurities, the key, the registration
    film_process: str = 'NONE'        # NONE | THREE_STRIP | TWO_COLOUR
    film_process_amount: float = 1.0
    film_exposure: float = 0.0        # the printer light, stops: down is denser
    film_gamma: float = 1.3           # the print's straight-line slope
    film_density: float = 2.4         # the dyes' maximum density
    film_filters: float = 0.3         # the taking filters' sharpness
    film_dye_purity: float = 0.5      # 0 = the documented impurities, 1 = ideal dyes
    film_key: float = 0.0             # the silver key image under the dyes
    film_halation: float = 0.0        # light scattered in the negative's base
    film_halation_radius: float = 12.0  # its reach, pixels
    film_register: float = 0.0        # dye-transfer registration error, pixels
    # R237: the print's wear (sizes at 1080 lines, scaled by the frame's height)
    film_grain_size: float = 1.0      # the grain's width, pixels
    film_grain_clump: float = 0.0     # the share of grain in clumps three grains wide
    film_grain_chroma: float = 0.5    # 0 = one sheet for every record, 1 = each its own
    film_dust_size: float = 1.5       # the specks' mean radius, pixels
    film_dust_negative: float = 0.25  # the share of dust that sat on the negative (clear specks)
    film_dust_cel: float = 0.0        # the share that sat on the cel (held with the cel)
    film_hairs: float = 0.0           # hairs in the projector gate, about this many at a time
    film_hair_length: float = 90.0    # pixels
    film_hair_width: float = 1.5      # pixels
    film_hair_hold: int = 12          # frames a hair stays, about
    film_scratches: float = 0.0       # scratches running, about this many at a time
    film_scratch_width: float = 1.2   # pixels
    film_scratch_hold: int = 48       # frames a scratch runs, about
    film_scratch_side: str = 'EMULSION'  # EMULSION | BASE
    film_reel: float = 0.0            # reel length in minutes; cue marks at its end (0 = none)
    # R233: the painted background road (core/gouache.py) -- Meier strokes
    # on materials in Background paint mode, and the Fleischer setback
    bg_paint: float = 0.0             # stroke coverage; 0 = off
    bg_stroke_size: float = 22.0      # stroke width, output pixels
    bg_stroke_length: float = 3.0     # length over width
    bg_direction: str = 'GRADIENT'    # GRADIENT | NORMAL | ANGLE
    bg_angle: float = 20.0            # degrees, for ANGLE and the fallbacks
    bg_spread: float = 0.5            # per-stroke turn, 0..1 of 45 degrees
    bg_bristles: float = 0.6          # bristle streaks across each stroke
    bg_variation: float = 0.25        # per-stroke lightening/darkening
    bg_smooth: float = 2.0            # the abstracted base under the strokes, sigma px
    bg_paper: float = 0.0             # the board's tooth over the painting
    setback: float = 0.0              # the lens on the miniature: max sigma, px
    setback_start: float = 10.0       # eye distance where the softness begins
    setback_range: float = 20.0       # distance over which it reaches Setback
    setback_sky: bool = True          # the sky sits at the far end
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
    'PC98':         (640, 400, 1.0, 1.0),
    'WXGA':         (1280, 800, 1.0, 1.0),
    'SXGA_PLUS':    (1400, 1050, 1.0, 1.0),
    'WSXGA_PLUS':   (1680, 1050, 1.0, 1.0),
    'WUXGA':        (1920, 1200, 1.0, 1.0),
    'QXGA':         (2048, 1536, 1.0, 1.0),
    # R241: the panels between and past those
    'XGA_PLUS':     (1152, 864, 1.0, 1.0),
    'HD_1366':      (1366, 768, 1.0, 1.0),
    'HD_900':       (1600, 900, 1.0, 1.0),
    'WQXGA':        (2560, 1600, 1.0, 1.0),
    'QSXGA':        (2560, 2048, 1.0, 1.0),
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
    'AMIGA_LACE':   (640, 400, 1.0, 1.0),
    'ZX_SPECTRUM':  (256, 192, 1.0, 1.0),
    'C64':          (320, 200, 1.0, 1.2),
    'MSX':          (256, 192, 1.0, 1.0),
    'APPLE_II':     (280, 192, 1.0, 1.0),
    'ATARI_8BIT':   (320, 192, 1.0, 1.0),
    'BBC_MICRO':    (640, 256, 1.0, 2.0),
    # R241: more of the 8- and 16-bit rooms (aspects by the same rule:
    # whatever fills the machine's own 4:3 tube -- or its own square
    # monitor where it shipped with one)
    'CPC':          (320, 200, 1.0, 1.2),
    'VIC20':        (176, 184, 46.0, 33.0),
    'ATARI_ST_MED': (640, 200, 5.0, 12.0),
    'ATARI_ST_HI':  (640, 400, 1.0, 1.0),
    'X68000':       (512, 512, 4.0, 3.0),
    'PC88':         (640, 200, 5.0, 12.0),

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
    'NES':          (256, 240, 8.0, 7.0),
    'GAMEBOY':      (160, 144, 1.0, 1.0),
    'GBA':          (240, 160, 1.0, 1.0),
    'NDS':          (256, 192, 1.0, 1.0),
    'PSP':          (480, 272, 1.0, 1.0),
    'NEOGEO':       (320, 224, 32.0, 35.0),
    'CPS2':         (384, 224, 8.0, 7.0),
    'VIRTUAL_BOY':  (384, 224, 1.0, 1.0),
    # R241: the consoles and handhelds between the ones above
    'TG16':         (256, 224, 8.0, 7.0),
    'CD_3DO':       (320, 240, 1.0, 1.0),
    'GAMEGEAR':     (160, 144, 6.0, 5.0),
    'LYNX':         (160, 102, 1.0, 1.0),
    'WONDERSWAN':   (224, 144, 1.0, 1.0),

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
    'DVD_NTSC':     (720, 480, 10.0, 11.0),
    'DVD_PAL':      (720, 576, 59.0, 54.0),
    'HDV_1080':     (1440, 1080, 4.0, 3.0),
    'DVCPRO_HD':    (1280, 1080, 3.0, 2.0),

    # --- pictures and textures
    'QUICKTAKE':    (640, 480, 1.0, 1.0),
    'DC120':        (1280, 960, 1.0, 1.0),
    'PHOTOCD_BASE': (768, 512, 1.0, 1.0),
    'PHOTOCD_4BASE': (1536, 1024, 1.0, 1.0),
    'PHOTOCD_16BASE': (3072, 2048, 1.0, 1.0),
    'TEXTURE_128':  (128, 128, 1.0, 1.0),
    'TEXTURE_256':  (256, 256, 1.0, 1.0),
    'TEXTURE_512':  (512, 512, 1.0, 1.0),
    'TEXTURE_1024': (1024, 1024, 1.0, 1.0),
    'TEXTURE_2048': (2048, 2048, 1.0, 1.0),
    'TEXTURE_4096': (4096, 4096, 1.0, 1.0),
    'MAVICA':       (640, 480, 1.0, 1.0),

    # --- panoramas and 360 (R194): pair the wide ones with the
    # Panoramic camera; the 2:1 sizes are the standard environment-
    # texture footprints
    'QTVR_CLASSIC': (2496, 768, 1.0, 1.0),
    'PANO_2K':      (2048, 512, 1.0, 1.0),
    'PANO_4K':      (4096, 1024, 1.0, 1.0),
    'PANO_8K':      (8192, 2048, 1.0, 1.0),
    'ENV_1K':       (1024, 512, 1.0, 1.0),
    'ENV_2K':       (2048, 1024, 1.0, 1.0),
    'ENV_4K':       (4096, 2048, 1.0, 1.0),
    'ENV_8K':       (8192, 4096, 1.0, 1.0),

    # --- film scans (R241): the scanner's frames, for the cel and
    # film rounds -- full aperture is the whole camera gate, Academy
    # the sound-era 1.375 inside it, the anamorphic scope frame is
    # squeezed 2:1 in the pixel (unsqueeze happens in the aspect),
    # Super 16 the 1.66 single-perf gate
    'FILM_FULL_2K':    (2048, 1556, 1.0, 1.0),
    'FILM_FULL_4K':    (4096, 3112, 1.0, 1.0),
    'FILM_ACADEMY_2K': (1828, 1332, 1.0, 1.0),
    'FILM_SCOPE_2K':   (1828, 1556, 2.0, 1.0),
    'FILM_SUPER16_2K': (2048, 1234, 1.0, 1.0),

    # --- modern and general (R194)
    'QHD_1440':     (2560, 1440, 1.0, 1.0),
    'UHD_4K':       (3840, 2160, 1.0, 1.0),
    'CINEMA_FLAT':  (1998, 1080, 1.0, 1.0),
    'CINEMA_SCOPE': (2048, 858, 1.0, 1.0),
    'SQUARE_1K':    (1024, 1024, 1.0, 1.0),
    'SOCIAL_SQUARE': (1080, 1080, 1.0, 1.0),
    'SOCIAL_STORY': (1080, 1920, 1.0, 1.0),
    'A4_150':       (1754, 1240, 1.0, 1.0),
    'A4_300':       (3508, 2480, 1.0, 1.0),
    'DCI_2K':       (2048, 1080, 1.0, 1.0),
    'DCI_4K':       (4096, 2160, 1.0, 1.0),
    'UHD_5K':       (5120, 2880, 1.0, 1.0),
    'UHD_8K':       (7680, 4320, 1.0, 1.0),
    'FHD_ULTRAWIDE': (2560, 1080, 1.0, 1.0),
    'QHD_ULTRAWIDE': (3440, 1440, 1.0, 1.0),
    'VERTICAL_FHD': (1080, 1920, 1.0, 1.0),
    'VERTICAL_4K':  (2160, 3840, 1.0, 1.0),
    'SQUARE_2K':    (2048, 2048, 1.0, 1.0),
    'A3_300':       (4961, 3508, 1.0, 1.0),
    'US_LETTER_300': (3300, 2550, 1.0, 1.0),
    # R241
    'SOCIAL_PORTRAIT': (1080, 1350, 1.0, 1.0),
    'DQHD':         (5120, 1440, 1.0, 1.0),
    'UWQHD_5K2K':   (5120, 2160, 1.0, 1.0),
}

#: the categories the UI shows, in order. Every RESOLUTION_PRESETS key appears
#: in exactly one group -- the test suite holds the two tables to each other.
RESOLUTION_GROUPS = (
    ("Televisions", ('NTSC_SQ', 'PAL_SQ', 'NTSC_D1', 'PAL_D1',
                     'NTSC_D1_WIDE', 'PAL_D1_WIDE', 'NTSC_TOASTER',
                     'HDTV_720', 'HDTV_1080')),
    ("Computer Monitors", ('HERCULES', 'EGA', 'QVGA', 'VGA', 'SVGA', 'XGA',
                           'XGA_PLUS', 'HD_1366', 'HD_900',
                           'SXGA', 'SXGA_PLUS', 'UXGA', 'WXGA',
                           'WSXGA_PLUS', 'WUXGA', 'WQXGA', 'QXGA',
                           'QSXGA', 'SUN_WS',
                           'NEXT_MEGAPIXEL', 'PC98',
                           'MAC_13', 'MAC_16', 'MAC_PORTRAIT',
                           'MAC_TWO_PAGE')),
    ("Home Computers", ('CGA', 'VGA_13H', 'QUAKE', 'MAC_CLASSIC', 'ATARI_ST',
                        'AMIGA_NTSC', 'AMIGA_PAL', 'AMIGA_HIRES',
                        'AMIGA_LACE', 'ZX_SPECTRUM', 'C64', 'MSX',
                        'APPLE_II', 'ATARI_8BIT', 'BBC_MICRO',
                        'CPC', 'VIC20', 'ATARI_ST_MED', 'ATARI_ST_HI',
                        'X68000', 'PC88')),
    ("Game Consoles", ('NES', 'SNES', 'GENESIS', 'NEOGEO', 'CPS2',
                       'TG16', 'SATURN', 'PSX', 'PSX_HI', 'N64', 'N64_HI',
                       'CD_3DO', 'DREAMCAST', 'GAMECUBE', 'PS2', 'XBOX',
                       'GAMEBOY', 'GBA', 'NDS', 'PSP', 'VIRTUAL_BOY',
                       'GAMEGEAR', 'LYNX', 'WONDERSWAN')),
    ("Video Formats", ('QCIF', 'CIF', 'VCD_NTSC', 'VCD_PAL', 'SVCD_NTSC',
                       'SVCD_PAL', 'DV_NTSC', 'DV_PAL', 'DVD_NTSC',
                       'DVD_PAL', 'HDV_1080', 'DVCPRO_HD',
                       'QUICKTIME_160')),
    ("Pictures & Textures", ('QUICKTAKE', 'MAVICA', 'DC120',
                             'PHOTOCD_BASE', 'PHOTOCD_4BASE',
                             'PHOTOCD_16BASE', 'TEXTURE_128',
                             'TEXTURE_256', 'TEXTURE_512',
                             'TEXTURE_1024', 'TEXTURE_2048',
                             'TEXTURE_4096')),
    ("Panoramas & 360", ('QTVR_CLASSIC', 'PANO_2K', 'PANO_4K', 'PANO_8K',
                         'ENV_1K', 'ENV_2K', 'ENV_4K', 'ENV_8K')),
    ("Film Scans", ('FILM_FULL_2K', 'FILM_FULL_4K', 'FILM_ACADEMY_2K',
                    'FILM_SCOPE_2K', 'FILM_SUPER16_2K')),
    ("Modern & General", ('QHD_1440', 'UHD_4K', 'UHD_5K', 'UHD_8K',
                          'DCI_2K', 'DCI_4K', 'CINEMA_FLAT',
                          'CINEMA_SCOPE', 'FHD_ULTRAWIDE',
                          'QHD_ULTRAWIDE', 'VERTICAL_FHD', 'VERTICAL_4K',
                          'SQUARE_1K', 'SQUARE_2K', 'SOCIAL_SQUARE',
                          'SOCIAL_STORY', 'SOCIAL_PORTRAIT',
                          'DQHD', 'UWQHD_5K2K',
                          'A4_150', 'A4_300', 'A3_300',
                          'US_LETTER_300')),
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
    'QTVR_CLASSIC': "QTVR Cylinder Classic",
    'PANO_2K': "Panorama 2K (4:1)", 'PANO_4K': "Panorama 4K (4:1)",
    'PANO_8K': "Panorama 8K (4:1)",
    'ENV_1K': "Environment 1K (2:1)", 'ENV_2K': "Environment 2K (2:1)",
    'ENV_4K': "Environment 4K (2:1)", 'ENV_8K': "Environment 8K (2:1)",
    'QHD_1440': "QHD 1440p", 'UHD_4K': "4K UHD",
    'CINEMA_FLAT': "Cinema Flat 1.85:1", 'CINEMA_SCOPE': "Cinema Scope 2.39:1",
    'SQUARE_1K': "Square 1024", 'SOCIAL_SQUARE': "Social Square 1080",
    'SOCIAL_STORY': "Social Story 9:16",
    'A4_150': "A4 Print 150dpi", 'A4_300': "A4 Print 300dpi",
    'NES': "NES / Famicom", 'GAMEBOY': "Game Boy",
    'GBA': "Game Boy Advance", 'NDS': "Nintendo DS",
    'PSP': "PlayStation Portable", 'NEOGEO': "Neo Geo",
    'CPS2': "Capcom CPS-2 Arcade", 'VIRTUAL_BOY': "Virtual Boy",
    'PC98': "NEC PC-9801", 'WXGA': "WXGA", 'SXGA_PLUS': "SXGA+",
    'WSXGA_PLUS': "WSXGA+", 'WUXGA': "WUXGA", 'QXGA': "QXGA",
    'AMIGA_LACE': "Amiga NTSC Hires Lace",
    'ZX_SPECTRUM': "ZX Spectrum", 'C64': "Commodore 64",
    'MSX': "MSX", 'APPLE_II': "Apple II Hi-Res",
    'ATARI_8BIT': "Atari 8-bit", 'BBC_MICRO': "BBC Micro Mode 0",
    'DVD_NTSC': "DVD NTSC", 'DVD_PAL': "DVD PAL",
    'HDV_1080': "HDV 1080 (anamorphic)",
    'DVCPRO_HD': "DVCPRO HD 1080 (anamorphic)",
    'TEXTURE_1024': "Game Texture 1K", 'TEXTURE_2048': "Game Texture 2K",
    'TEXTURE_4096': "Game Texture 4K", 'MAVICA': "Sony Mavica FD",
    'DCI_2K': "DCI 2K Cinema", 'DCI_4K': "DCI 4K Cinema",
    'UHD_5K': "5K UHD+", 'UHD_8K': "8K UHD",
    'FHD_ULTRAWIDE': "Ultrawide FHD 21:9",
    'QHD_ULTRAWIDE': "Ultrawide QHD 21:9",
    'VERTICAL_FHD': "Vertical FHD 9:16", 'VERTICAL_4K': "Vertical 4K 9:16",
    'SQUARE_2K': "Square 2048",
    'A3_300': "A3 Print 300dpi", 'US_LETTER_300': "US Letter 300dpi",
    # R241
    'XGA_PLUS': "XGA+", 'HD_1366': "HD Laptop 1366",
    'HD_900': "HD+ 1600x900", 'WQXGA': "WQXGA", 'QSXGA': "QSXGA",
    'CPC': "Amstrad CPC", 'VIC20': "VIC-20",
    'ATARI_ST_MED': "Atari ST Medium", 'ATARI_ST_HI': "Atari ST High Mono",
    'X68000': "Sharp X68000", 'PC88': "NEC PC-88",
    'TG16': "TurboGrafx-16 / PC Engine", 'CD_3DO': "3DO",
    'GAMEGEAR': "Game Gear", 'LYNX': "Atari Lynx",
    'WONDERSWAN': "WonderSwan",
    'FILM_FULL_2K': "35mm Full Aperture 2K",
    'FILM_FULL_4K': "35mm Full Aperture 4K",
    'FILM_ACADEMY_2K': "35mm Academy 2K",
    'FILM_SCOPE_2K': "CinemaScope 2K (Anamorphic)",
    'FILM_SUPER16_2K': "Super 16 2K",
    'SOCIAL_PORTRAIT': "Social Portrait 4:5",
    'DQHD': "Super Ultrawide DQHD", 'UWQHD_5K2K': "Ultrawide 5K2K",
}


def resolution_label(key):
    return RESOLUTION_LABELS.get(key, key.replace('_', ' ').title())


def resolution_description(key):
    x, y, ax, ay = RESOLUTION_PRESETS[key]
    if ax == ay:
        return f"{x}x{y}"
    return f"{x}x{y}, {ax:g}:{ay:g} pixels"
