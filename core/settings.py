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
    dof_method: str = 'POST'         # POST | LENS_ACCUMULATE (R251 C098: REYES / accumulation-buffer lens passes, one full render per point)
    dof_lens_pattern: str = 'HALTON_DISC'   # HALTON_DISC | MAX_SPIRAL
    dof_lens_samples: int = 23       # one full render per point
    palette_lock: bool = True
    film_transparent: bool = False
    res_preset: str = 'CUSTOM'
    resolution_x: int = 320
    resolution_y: int = 240
    pixel_aspect_x: float = 1.0
    pixel_aspect_y: float = 1.0
    # R253: the render region (Blender's Ctrl+B border) -- fractions of
    # the frame, bottom-left origin, exactly scene.render.border_min_x..
    # max_y (row 0 is the bottom here too). The engine derives them from
    # scene.render at F12 (properties._DERIVED_FIELDS: Blender's own
    # Output > Format panel is their UI) and the viewport worker sets
    # them from the drawn rect. Off / the full frame is the identity:
    # render.region_pixels() returns None and every old scene renders
    # through exactly the pre-R253 code paths
    use_border: bool = False
    border_min_x: float = 0.0
    border_min_y: float = 0.0
    border_max_x: float = 1.0
    border_max_y: float = 1.0
    # R253: viewport-only -- in camera view the rendered viewport shades
    # only the camera frame's pixels (engine._view_rect reads it; F12
    # never does)
    viewport_camera_frame: bool = False
    # ------------------------------------------------------------- sampling
    aa_mode: str = 'SUPERSAMPLE'      # NONE | SUPERSAMPLE | EDGE | ADAPTIVE | ACCUMULATE
    aa_samples: int = 1               # supersample factor (1..8)
    aa_filter: str = 'BOX'            # BOX | TRIANGLE | GAUSS | CATROM | MITCHELL
    aa_filter_width: float = 1.0
    aa_sample_pattern: str = 'GRID'   # R251 C127: GRID | JITTER -- REYES per-subpixel jitter from the integer hash
    aa_clamp_samples: bool = False     # R251 C094: LightWave Limit Dynamic Range -- min(sample, 1) before the AA filter
    aa_gamma_blend: bool = False       # R251 C122: Blender 2.41's gamma-2 OSA sample blend (400-entry tables)
    aa_edge_threshold: float = 0.1
    n64_coverage_aa: bool = False      # R251 C001: RDP 3-bit coverage + VI scan-out blend (needs aa_samples 1)
    n64_divot: bool = True             # R251 C001: the VI divot median after the blend
    # accumulation motion blur: the frame renders motion_steps times across
    # the shutter (re-exported at each subframe) and averages -- the
    # accumulation-buffer trails of the era, paid for honestly at N frames
    motion_blur: bool = False
    motion_shutter: float = 0.5        # shutter open time, in frames
    motion_steps: int = 5
    motion_blur_mode: str = 'MEAN'    # MEAN | MAX_SLICES | LW_FIELD (R251 C090)
    motion_samples: int = 10          # Max's Samples (MAX_SLICES): slices each pixel averages
    motion_dither: float = 0.0        # Max 3+ multi-pass Dither Strength
    motion_dither_tile: int = 32      # Max 3+ multi-pass Tile Size (px)
    # ------------------------------------------------------------- stereo
    # parallel cameras with an off-axis (asymmetric-frustum) shift, so
    # the convergence plane sits at zero parallax with no vertical error
    stereo_mode: str = 'NONE'          # NONE | ANAGLYPH | SBS | CROSS
    stereo_eye_distance: float = 0.065
    stereo_convergence: float = 8.0
    stereo_parallax_layers: bool = False   # C016 Virtual Boy: one render, whole-pixel shift per object
    stereo_parallax_max: int = 16          # C016 the largest whole-pixel parallax; the sky takes it
    # ------------------------------------------------------------- camera (R251)
    camera_yshear: bool = False       # C058 Heretic/Build Y-shear: level view + projection slide by f*tan(pitch)
    pano_parts: int = 1               # C125 Blender 2.4 Pano + Xparts: N planar strips, butted; 1 = off
    # ------------------------------------------------------------- geometry
    backface_cull: bool = False
    two_sided_lighting: bool = True
    subpixel_precision: str = 'FLOAT'  # FLOAT | FIXED_4 | FIXED_1 | INTEGER
    pixel_center: str = 'HALF'        # R251 C084: HALF | INTEGER_D3D -- Direct3D 3-9 sampled at the integer corner
    vertex_snap: bool = False          # PlayStation-style vertex jitter
    vertex_snap_grid: float = 1.0      # in pixels of the *output* image
    vertex_quantize: str = 'NONE'     # R251 C012: NONE | PS1 | N64 -- integer vertex formats on a world lattice
    vertex_units: float = 64.0        # R251 C012: lattice units per world unit
    depth_precision: int = 24          # z-buffer bits (8..32); low = fighting
    depth_encoding: str = 'LINEAR'    # R251 C007/C026/C075: LINEAR | N64_FLOAT18 | GC_14E2 | GC_13E3 | GC_12E4 | W_FIXED | VOODOO_W16
    depth_sort: str = 'ZBUFFER'        # ZBUFFER | PAINTERS
    painters_key: str = 'CENTROID'     # CENTROID | NEAREST | FARTHEST | ORDERING_TABLE (R251 C004)
    ot_length: int = 4096            # R251 C004: PS1 ordering-table entries (painters_key ORDERING_TABLE)
    ot_far: float = 40.0              # R251 C004: scene distance of the last bucket
    # the camera raster's near-plane epsilon. 1e-5 IS the value
    # the rasterisers have always run; the setting used to claim
    # 1e-4 and drive nothing (found by the settings audit)
    clip_near_epsilon: float = 1e-5
    near_clip_mode: str = 'CLIP'      # R251 C027: CLIP | REJECT -- whole-triangle rejection at near/far/guard 6.4 (PS2 VU1, PS1)
    # ------------------------------------------------------------- shading
    default_model: str = 'PHONG'
    force_model: str = 'NONE'          # override every material's model
    shading_rate: str = 'PIXEL'        # PIXEL | VERTEX (global Gouraud) | FACE
    normal_source: str = 'AUTO'        # AUTO | SPLIT | FACE | VERTEX
    shading_rate_area: float = 0.0   # R251: REYES ShadingRate as an AREA in pixels; 0 = off (per-pixel), PRMan's 1.0 = one shade per pixel-area micropolygon
    specular_in_gamma: bool = True     # 90s renderers lit in display space
    # PIXEL | AXIS -- the true eye vector, or one camera axis for the
    # frame (GL 1.1 infinite viewer, Sega Model, DS) (R251)
    specular_viewer: str = 'PIXEL'
    clamp_specular: bool = True
    light_clamp: float = 0.0           # 0 = off
    # fold the frame number into crand's hash: POV's flicker on purpose (R251)
    crand_per_frame: bool = False
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
    shadow_map_depth: str = 'CLASSIC'    # R251 C117: CLASSIC | MIDPOINT -- Woo's halfway map (Maya Use Mid Dist, Blender Classic-Halfway)
    shadow_softness: float = 1.0
    shadow_samples: int = 4
    planar_plane_z: float = 0.0     # R251 C052: the receiver plane Z of PLANAR shadows (Blinn / Model 1)
    modvol_scale: int = 128     # R251 C020: the Dreamcast FPU_SHAD_SCALE register for modifier volumes (colour * scale / 256)
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
    # R251 texture pack (TEX-1, wave 1)
    tex_format: str = 'NONE'          # R251: once-per-upload texel format (Glide/D3D/PCX/Model 2/NCC/DS); NONE = the image's own precision
    tex_tmem_format: str = 'OFF'      # R251: N64 4 KB TMEM budget by format (size, palette, depth from one byte rule); OFF = none; owns the texel format
    tex_compress: str = 'NONE'        # R251: baked block compression (DXT1 / Xbox NV2A decode / GameCube CMPR / Dreamcast VQ); NONE = uncompressed
    tex_frac_bits: str = 'FLOAT'      # R251: bilinear weight precision (FLOAT | BITS_4 Voodoo1/Verite | BITS_8 Voodoo2)
    # R251 texture pack (wave 2, TEX-2): plumbed through sample_opts on both roads, inert until pass 2
    tex_clamp_mode: str = 'EDGE'      # R251: how Extend nodes clamp (EDGE = CLAMP_TO_EDGE; GL_CLAMP = OpenGL 1.1 border, the half-texel seam)
    tex_colorkey: bool = False        # R251: Glide/D3D chroma key tested AFTER filtering; alpha cut-outs stored as opaque black at prep
    tex_colorkey_range: int = 0       # R251: Voodoo2 chromaRange in 8-bit levels per channel (0 = Voodoo1/D3D exact match)
    tex_mip_select: str = 'FILTER'    # R251: how a mip level is chosen (FILTER = as today | BLEND | NEAREST_LEVEL GL/D3D | DITHER_VOODOO)
    tex_lod_source: str = 'DERIVATIVE'  # R251: where the mip LOD comes from (DERIVATIVE = screen derivatives | GS_Q = PS2 log2(1/|Q|) << L + K | TRIANGLE = one level per polygon)
    tex_lod_k: float = 0.0            # R251: GS TEX1 K, signed 7.4 fixed in levels (quantised to 1/16)
    tex_lod_l: int = 0                # R251: GS TEX1 L shift (0..3)
    tex_lod_sharpen: bool = False     # R251: N64 G_TD_SHARPEN: extrapolate level 0 away from level 1 under magnification, 9-bit clamp
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
    blend_equation: str = 'ALPHA'      # R251: the blend unit's per-layer formula -- ALPHA | PS1_AVG | PS1_ADD | PS1_SUB | PS1_QUARTER | SATURN_HALF | SATURN_SHADOW | SATURN_HALF_LUM | THREEDO_SUB | THREEDO_XOR | SNES_ADD | SNES_SUB | SNES_ADD_HALF | SNES_SUB_HALF | GBA | DS | FUZZ | THIN_WALL | IMAGINE_FOG (integer, no alpha weighting except GBA/DS)
    translucent_order: str = 'DEPTH'   # R251: DEPTH | Y_SORT (DS auto-sort: bottom row, top row, submission) | SUBMISSION
    translucent_depth_write: bool = False  # R251: DS POLYGON_ATTR bit 11 -- composited fragments occlude later ones
    framebuffer: str = 'NONE'          # R251: NONE | PS2_CT16 | GC_RGBA6 | VOODOO_565_4X4 | VOODOO_565_2X2 -- the buffer every write truncates into, read back per blend
    fb_dither: bool = True             # R251: the buffer's write dither (GS DTHE, Voodoo grDitherMode); GC always dithers
    fb_dither_subtract: bool = True    # R251: Voodoo fbzMode bit 19, the dithered read-back
    # --------------------------------------------------------------- depth cue
    fog: bool = False
    fog_mode: str = 'LINEAR'           # LINEAR | EXP | EXP2 | TABLE16 | GTE_1Z (R251)
    # R251: the accelerator's own fog table, filled by the fog_mode curve
    fog_table: str = 'NONE'            # NONE | VOODOO64 | PVR128 | DS32
    # R251: W = eye depth; Z = Direct3D's post-projection z in 0..1
    fog_depth: str = 'W'               # W | Z
    fog_color: Tuple[float, float, float] = (0.5, 0.55, 0.65)
    fog_start: float = 5.0
    fog_end: float = 40.0
    fog_density: float = 0.05
    fog_vertex: bool = False           # per-vertex fog (Voodoo/PS1 style)
    # R251: Voodoo2 fogMode bit 6, the 4x4 matrix added to the blend fraction
    fog_dither: bool = False
    # R251 F005 (LIGHT-A2): one fog value per polygon from its mean vertex
    # depth (Namco System 21)
    fog_face: bool = False
    # R251 F004: GX_InitFogAdjTable -- the planar fog depth scaled by the
    # pixel column's secant (GameCube)
    fog_range_adjust: bool = False
    # R251 F006: Model 3 fogAmbient (scales the fog colour), and System
    # 22's cz bank 1 -- a second Start/End pair, inert while end <= start
    fog_ambient: float = 1.0
    fog_bank1_start: float = 0.0
    fog_bank1_end: float = 0.0
    # R251 F008: FIXED | BACKDROP -- LightWave's Use Backdrop Color
    fog_color_source: str = 'FIXED'
    # R251 F009: POV-Ray fog_type 2 (ground fog: density 1/(1+Y^2) above
    # the offset, Y in units of the altitude, the atan integral)
    fog_ground_offset: float = 0.0
    fog_ground_alt: float = 1.0
    # R251 F010: POV-Ray fog turbulence <t> as one scalar (0 = off) and
    # turb_depth (the share of the fog distance the noise may remove)
    fog_turbulence: float = 0.0
    fog_turb_depth: float = 0.5
    # R223: banded depth fog -- transmittance quantized to cel steps
    # (0 = off; 2+ = the anime painted-planes distance look)
    fog_bands: int = 0
    # Model 3 spotlight fog attenuation: the lobe's share added to the
    # fog colour (R251)
    fog_spot: float = 0.0
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
    palette_bits: str = 'NONE'         # NONE | BITS_1 | BITS_2 | CPC_27 | BITS_3 | \
                                       # BITS_4 | BITS_5 | BITS_6 -- R251: the palette
                                       # RAM / DAC precision every register is snapped
                                       # to before the per-pixel search (C061)
    attribute_cells: str = 'NONE'      # NONE | ZX_SPECTRUM | MSX1 | C64_HIRES | C64_MULTI
                                       # -- R251: colour per character cell from the
                                       # machine's fixed set, least squared error per
                                       # cell (C050); owns the colour stage
    scanline_palette: str = 'NONE'     # NONE | SPECTRUM_512 | DYNAMIC_HIRES | SHAM --
                                       # R251: the registers rewritten every scanline, a
                                       # median cut per line (C060); CPU only, refused
                                       # by name on the GPU road
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
    super_black: bool = False          # R251: 3D Studio / Max Super Black -- covered
                                       # pixels floored at the threshold for luminance
                                       # keying (C092)
    super_black_threshold: int = 15    # 0..255, Max's default
    video_color_check: str = 'NONE'    # NONE | FLAG_BLACK | SCALE_LUMA | SCALE_SAT --
                                       # R251: 3D Studio / Max Video Color Check on the
                                       # composite envelope (C093)
    video_system: str = 'NTSC'         # NTSC | PAL
    video_ire_limit: str = 'IRE_120'   # IRE_120 | IRE_110
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
    # ------------------------------------------------ post: signal (R251)
    # the machine's scan-out stages, between the framebuffer and the glass
    # (core/signal_era.py; every default leaves every pixel untouched)
    vi_dither_filter: bool = False   # R251: N64 VI DITHER_FILTER_ENABLE
    vi_gamma: str = 'NONE'              # NONE | GAMMA | GAMMA_DITHER | DITHER_ONLY (N64 VI GAMMA_ENABLE bits)
    copy_filter: str = 'NONE'        # NONE | DEFLICKER | DEFLICKER_AA (GameCube/Wii EFB copy, R251)
    crtc_blend: str = 'NONE'        # NONE | PREVIOUS_FRAME | BG_COLOR (PS2 GS PMODE, R251)
    crtc_alpha: int = 128           # PMODE ALP
    crtc_bg_color: tuple = (0.0, 0.0, 0.0)   # GS BGCOLOR
    video_filter: str = 'NONE'      # NONE | VOODOO1 | VOODOO2 (3dfx scan-out '22-bit' filter, R251)
    video_filter_threshold: int = 64   # SST_VIDEO_FILTER_THRESHOLD
    # R251 SIG-2: the digital formats' chroma, the tape, the cable, the
    # receiver (core/signal_tape.py; every default leaves every pixel untouched)
    chroma_format: str = 'NONE'     # NONE | Y422 | Y411 | Y420_MPEG1 | Y420_MPEG2 | Y420_DVPAL | XFB_422 (R251)
    chroma_upsample: str = 'HOLD'   # HOLD | LINEAR: how the decoder rebuilt chroma between the sites
    signal: str = 'RGB'              # RGB | SVIDEO | SVIDEO_PAL | RF (the cable, R251)
    rf_bandwidth: float = 3.2       # MHz: the RF modulator's luma low-pass
    rf_beat: float = 0.0            # the 920 kHz sound-chroma herringbone, in proportion to chroma
    rf_snow: float = 0.0            # thermal noise as a fraction of white, hashed per pixel and frame
    rf_ghost: float = 0.0           # the multipath ghost's strength
    rf_ghost_delay: float = 1.0     # us along the 52.66 us NTSC active line
    pal_decoder: str = 'NONE'       # NONE | DELAY_LINE | SIMPLE (PAL receiver, R251)
    pal_phase_error: float = 0.0    # degrees, SIMPLE only: Hanover bars
    pal_crawl: float = 0.0          # the 4.43 MHz subcarrier left in luma, eight-field sequence
    tape: str = 'NONE'              # NONE | VHS | SVHS | BETAMAX | UMATIC | VIDEO8 | HI8 | BETACAM | BETACAM_SP | TYPE_C (R251)
    tape_generations: int = 1       # dubs deep: the low-pass at sigma * sqrt(N), the noise * sqrt(N)
    tape_noise: float = 0.02        # FM demodulation noise on luma as a fraction of white (chroma at half)
    tape_head_switch: bool = True   # the head-switch tear: the bottom 7 lines (at 480) torn sideways, x4 noise
    tape_dropouts: float = 0.0      # expected dropouts per frame: white dashes 8..48 px at hashed rows
    # R251 SIG-3: the codecs the display showed and the optical printer
    # (core/signal_codec.py; every default leaves every pixel untouched)
    mpeg1: bool = False              # MPEG-1 intra recode (Video CD, R251 C132)
    mpeg1_qscale: int = 8            # quantizer_scale for I pictures, 1..31
    mpeg1_gop: int = 15              # frames per group of pictures; 0 = every frame an I picture
    smacker: bool = False            # Smacker 4x4 block recode over the frame palette (R251 C133)
    smacker_quality: float = 0.5     # the encoder's budget, 0..1
    matte_glow: bool = False         # the Tron printer's backlit Kodalith mattes (R251 C134)
    matte_glow_radius: float = 8.0   # the first diffusion pass's sigma, px at 1080 lines
    matte_glow_passes: int = 3       # diffusion passes at radius x1, x2, x4... (1..5)
    matte_glow_exposure: float = 1.0   # what one crisp pass adds; the passes add 1/2, 1/4...
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
    wire_mode: str = 'ALL'            # ALL | CREASE | ELITE | BEAM
    wire_angle: float = 25.0
    wire_color: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wire_width: float = 1.0
    wire_dot_distance: float = 0.0     # R251 C063: Elite rule -- the object collapses to a dot past this distance (0 = never)
    beam_machine: str = 'AVG'          # R251 C055: DVG | AVG | STARWARS -- the vector generator's intensity, colour and dwell
    beam_sigma: float = 0.7            # R251 C055: phosphor spot sigma in pixels at 480 lines
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
    # R251 (MAT-A C049): the Super FX screen heights (SCMR), the SNES's 8:7
    'SUPERFX_192':  (256, 192, 8.0, 7.0),
    'SUPERFX_160':  (256, 160, 8.0, 7.0),
    'SUPERFX_128':  (256, 128, 8.0, 7.0),
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
    ("Game Consoles", ('NES', 'SNES', 'SUPERFX_192', 'SUPERFX_160',
                       'SUPERFX_128', 'GENESIS', 'NEOGEO', 'CPS2',
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
    'SUPERFX_192': "Super FX 256x192 (Star Fox)",
    'SUPERFX_160': "Super FX 256x160", 'SUPERFX_128': "Super FX 256x128",
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
