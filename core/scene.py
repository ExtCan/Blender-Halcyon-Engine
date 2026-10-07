"""Renderer-side scene description.

Plain dataclasses + numpy arrays only. The Blender exporter (halcyon/export.py)
fills these in; the renderer never touches bpy. Everything is in world space
unless a field says otherwise.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ImageBuffer:
    """A decoded image: float32 (H,W,4), origin at bottom-left (Blender order)."""
    name: str = ""
    pixels: Optional[np.ndarray] = None
    colorspace: str = 'sRGB'          # 'sRGB' | 'Non-Color'
    mips: Optional[List[np.ndarray]] = None

    @property
    def width(self):
        return 0 if self.pixels is None else self.pixels.shape[1]

    @property
    def height(self):
        return 0 if self.pixels is None else self.pixels.shape[0]


@dataclass
class Material:
    """One material slot. `graph` is the serialised node tree (see nodeeval)."""
    name: str = "Material"
    index: int = 0
    graph: Optional[Dict[str, Any]] = None
    # Fallback surface used when there is no node graph at all.
    model: str = 'PHONG'
    use_override: bool = False
    diffuse: tuple = (0.8, 0.8, 0.8)
    diffuse_level: float = 1.0
    specular: tuple = (1.0, 1.0, 1.0)
    specular_level: float = 0.5
    glossiness: float = 25.0
    ambient_level: float = 1.0
    emission: tuple = (0.0, 0.0, 0.0)
    emission_level: float = 0.0
    opacity: float = 1.0
    ior: float = 1.45
    roughness: float = 0.3
    anisotropy: float = 0.0
    aniso_rotation: float = 0.0
    metallic: float = 0.0
    reflect_level: float = 0.0
    reflect_map: Optional[ImageBuffer] = None
    two_sided: bool = True
    shadeless: bool = False
    receive_shadow: bool = True
    cast_shadow: bool = True
    # R251 C134: the gel this material's Kodalith matte is backlit
    # through in the Tron printer stage (black = no matte)
    glow_gel: tuple = (0.0, 0.0, 0.0)
    # R251 C020/C036: this material's triangles are an AUTHORED VOLUME
    # (never drawn as geometry): a Dreamcast modifier volume (stencil
    # parity x FPU_SHAD_SCALE, INCLUDE or EXCLUDE) or a Nintendo DS
    # shadow polygon (depth-fail mask, then a 5-bit blend of `diffuse`
    # at `shadow_alpha`, skipping pixels of its own `polygon_id`)
    volume_role: str = 'NONE'   # NONE | DC_INCLUDE | DC_EXCLUDE | DS_SHADOW
    polygon_id: int = 0         # the DS attribute-buffer polygon ID, 0..63
    shadow_alpha: int = 16      # the DS shadow polygon's 5-bit alpha, 1..30
    # R208: this material dresses HAIR geometry (ribbon strands or fur
    # shells): the colour layer carries strand data (r intercept,
    # g random, b length, a thickness) and Hair Info reads it. The
    # graph root carries the same flag for both shading devices.
    strand: bool = False
    # R211: how the material's alpha reaches the frame. BLEND is the
    # layer road (A-buffer fragments, sorted and composited -- glass).
    # CLIP is punch-through, the era's cut-out alpha: the alpha chain is
    # tested against `alpha_clip` and the surface is either fully there
    # or fully absent, so its fragments live in the depth-buffered pass
    # and shade ONCE -- no layers, no sorting, no per-layer passes. The
    # shading law forces alpha to exactly 0 or 1 under CLIP on every
    # road (camera, layers, rays), whichever road drew it.
    alpha_mode: str = 'BLEND'
    alpha_clip: float = 0.5
    blend_mode: str = 'INHERIT'        # R251: INHERIT | the blend_equation items | ENV_HOLE
    z_offset: float = 0.0              # R251: Blender 2.4x Zoffs, see-through fragments sort/test this much nearer
    z_invert: bool = False             # R251: Blender 2.4x ZInvert, the material's fragments sort far-first
    thin_wall_offset: float = 0.5      # R251: Max Thin Wall Refraction 'Thickness Offset', read under blend_mode THIN_WALL
    fog_length: float = 1.0            # R251: Imagine 'Fog Length', read under blend_mode IMAGINE_FOG
    # R220: per-material ink. The cartoon outline pass was one global
    # render setting; these let a material opt out of it, force it on
    # (even with the global switch off), and carry its own ink colour
    # and width. Defaults are exact inheritance -- bitwise the global
    # behaviour.
    ink_mode: str = 'INHERIT'         # INHERIT | ON | OFF
    ink_use_color: bool = False       # False = the global Ink Colour
    ink_color: tuple = (0.0, 0.0, 0.0)
    ink_width: int = 0                # 0 = the global Ink Width
    # R239: the Guilty Gear line control -- the mesh's vertex colours
    # steer this material's line (ASW's own convention: ALPHA times the
    # width, 0.5 as set / 0 erased / 1 doubled; BLUE holds interior and
    # marked lines back until the surface nears its silhouette)
    ink_vc: str = 'OFF'               # OFF | ARCSYS
    # R233: the cel or the painting. CEL is everything that shipped
    # before; BACKGROUND takes the painted background road (brush strokes
    # over its lit colour, the ink suppressed unless ink_mode says ON)
    # and the setback's softness
    paint_mode: str = 'CEL'           # CEL | BACKGROUND
    wire: bool = False
    wire_size: float = 1.0
    face_texture: bool = False
    # Halo material (R191): BI's MA_TYPE_HALO. None = ordinary surface;
    # a dict turns the mesh's VERTICES into depth-tested billboard
    # glows (shadeHaloFloat, transcribed). Keys, 2.79 names in parens:
    # size (hasize), hardness (har), add (add, 0..1), seed (seed1),
    # alpha (alpha), color (r,g,b), rings/lines/star_points (ringc/
    # linec/starc; 0 = off), line_color (spec rgb), ring_color (mir
    # rgb), xalpha, soft, shaded, puno (the mode bits).
    halo: Optional[Dict[str, Any]] = None
    # Cached compiled coded-shader programs keyed by node name.
    programs: Dict[str, Any] = field(default_factory=dict)
    # Filled by the renderer: node names that could not be evaluated.
    unsupported: List[str] = field(default_factory=list)


@dataclass
class MeshData:
    """Triangulated geometry, world space, one flat soup per scene."""
    verts: np.ndarray = None          # (V,3) float32
    normals: np.ndarray = None        # (V,3) float32 -- per corner (split)
    uvs: np.ndarray = None            # (V,2) float32
    uvs2: Optional[np.ndarray] = None  # (V,2) secondary UV
    uv_names: Optional[list] = None    # layer names, [active, secondary]
    colors: np.ndarray = None         # (V,4) float32
    # R246: the colour layer's NAME (the one `colors` carries), so a
    # Color Attribute node naming it resolves on both devices
    color_name: Optional[str] = None
    tris: np.ndarray = None           # (T,3) int32 indices into verts
    mat_index: np.ndarray = None      # (T,) int32 -> index into Scene.materials
    obj_index: np.ndarray = None      # (T,) int32 -> index into Scene.objects
    face_normals: np.ndarray = None   # (T,3) float32
    smooth: np.ndarray = None         # (T,) bool
    # R220: the artist's marked edges (Sharp / crease / Freestyle mark)
    # for the ink pass, folded per triangle: uint8 (T,), bit k set = the
    # edge opposite corner k is marked. None = no marks anywhere.
    ink_tri_mask: Optional[np.ndarray] = None


@dataclass
class ObjectInfo:
    name: str = ""
    location: tuple = (0.0, 0.0, 0.0)
    matrix_world: Any = None
    color: tuple = (1.0, 1.0, 1.0, 1.0)
    index: int = 0
    random: float = 0.0
    visible_camera: bool = True
    visible_shadow: bool = True
    #: the classic Auto Smooth angle threshold (Object.smoothresh) the
    #: MA_RAYBIAS terminator fix reads; 0 = none
    smoothresh: float = 0.0
    cast_shadow: bool = True
    receive_shadow: bool = True
    holdout: bool = False
    # R223: a fluid-sim smoke domain's voxels, when this object carries
    # one and the cache is readable: {'res': (rx, ry, rz), 'density':
    # flat float32 (Mantaflow layout, x fastest), 'flame': same or
    # None}. The volume marcher multiplies its density field by it.
    smoke_grid: Optional[Dict[str, Any]] = None


@dataclass
class Light:
    type: str = 'POINT'               # POINT | SUN | SPOT | AREA | AMBIENT
    name: str = "Light"
    position: tuple = (0.0, 0.0, 0.0)
    direction: tuple = (0.0, 0.0, -1.0)
    color: tuple = (1.0, 1.0, 1.0)
    energy: float = 1000.0
    radius: float = 0.0
    # Spot
    spot_size: float = 1.2            # full cone angle, radians
    spot_blend: float = 0.15
    hotspot: float = 0.0              # derived falloff/hotspot pair (radians)
    # R251 F012: the cone law -- BLENDER | GL11 | POV | GX_FLAT | GX_COS |
    # GX_COS2 | GX_SHARP | GX_RING1 | GX_RING2 -- and GL's
    # GL_SPOT_EXPONENT / POV's tightness (0 = flat, both defaults)
    spot_law: str = 'BLENDER'
    spot_exponent: float = 0.0
    # R251 F015: Sega Model 3's viewport spotlight -- an ellipse pinned
    # to the screen at the lamp's projected position; no cone, no shadow
    screen_spot: bool = False
    # Area
    area_size: tuple = (1.0, 1.0)
    area_shape: str = 'SQUARE'
    area_x: tuple = (1.0, 0.0, 0.0)
    area_y: tuple = (0.0, 1.0, 0.0)
    # BI's area lamp Gamma (la->k): shapes pow(stokes*areasize, k)
    area_gamma: float = 1.0
    # 90s-style decay
    decay: str = 'DEFAULT'     # NONE | INVERSE | INVERSE_SQUARE | CUSTOM
    decay_start: float = 0.0
    decay_end: float = 25.0
    decay_ld1: float = 0.0     # BI Lin/Quad sliders (att1/att2)
    decay_ld2: float = 0.0
    bi_sphere: bool = False    # BI's Sphere clamp at decay_end
    # R251 F013: GX_InitLightDistAttn's ref_brite -- the fraction of
    # the energy left at decay_end under the three GX_* decay laws
    gx_ref_brite: float = 0.5
    # Shadowing
    shadow: str = 'MAP'               # NONE | MAP | RAY | PLANAR
    # 0 = inherit the render setting. The old defaults (512 / 0.02) sat in
    # front of the global sliders and made them unreachable: `light.x or
    # settings.x` never fell through (found by the settings audit)
    shadow_map_size: int = 0
    shadow_bias: float = 0.0
    shadow_map_depth: str = 'INHERIT'   # INHERIT | CLASSIC | MIDPOINT (R251 C117)
    shadow_softness: float = 1.0      # shadow-map blur radius in texels
    shadow_samples: int = 4
    shadow_color: tuple = (0.0, 0.0, 0.0)
    shadow_density: float = 1.0
    # Period features
    negative: bool = False
    # R251 F014: Blender Internal's LA_ONLYSHADOW -- the lamp lights
    # nothing and subtracts its plain diffuse where its shadow falls
    only_shadow: bool = False
    diffuse_only: bool = False
    specular_only: bool = False
    affect_diffuse: bool = True
    affect_specular: bool = True
    volumetric: float = 0.0
    # Beam Occlusion: march samples the lamp cannot reach scatter
    # nothing, so the visible beam STOPS at a mesh in its way. Costs a
    # shadow ray per march sample, so it is a per-lamp choice
    volumetric_occlusion: bool = False
    # Lens flare: the Video Post / LightWave anatomy, anchored on THIS
    # lamp's screen position and faded by its visibility -- unlike the
    # image-space post flare, which chases any bright pixel
    flare: float = 0.0                # element intensity; 0 = no flare
    flare_scale: float = 1.0
    flare_streaks: int = 6            # star spokes (0 = none)
    flare_rings: int = 1              # chromatic halo rings (0 = none)
    flare_ghosts: int = 6             # aperture ghosts along the axis
    # Caustics: the animated pool-light web, baked per frame into a
    # procedural cookie -- SPOT projects it through the cone, SUN tiles
    # it across the world (seamlessly; the pattern is periodic)
    caustics: float = 0.0
    caustics_scale: float = 4.0       # SPOT: cells across the cone;
    #                                   SUN: world units per tile
    caustics_speed: float = 1.0
    exclude_objects: tuple = ()
    exclude_mode: str = 'EXCLUDE'
    ambient_only: bool = False
    # Projected texture (gobo / cookie): sixth-generation projective
    # texturing -- a SPOT projects its image through the cone like a slide
    # projector, a SUN tiles it across the world perpendicular to its rays
    # (cloud shadows), a POINT wraps it around itself like a pierced
    # lantern (lat-long), an AREA carries it on its face like a printed
    # gel (R219: every lamp kind projects). `cookie` is an (H,W,3|4)
    # float32 array (or anything with a .pixels attribute holding one).
    cookie: Any = None
    cookie_strength: float = 1.0
    cookie_scale: float = 10.0        # SUN only: world size of one tile
    # R219: how the image continues past its edges. AUTO is the era
    # default each projection always had -- SUN/HEMI tile (REPEAT),
    # SPOT and AREA clamp (EXTEND), POINT wraps around (REPEAT).
    # CLIP is the projector's gate: outside the slide, no light.
    cookie_extend: str = 'AUTO'       # AUTO | REPEAT | EXTEND | CLIP
    # R219: the lookup filter, same texel arithmetic on both devices.
    # The projection's focus blur needs no field of its own: it reads
    # the lamp's existing `radius` (SUN/HEMI: the Angle slider in
    # radians; others: Radius in scene units). 0 = razor sharp.
    cookie_filter: str = 'BILINEAR'   # BILINEAR | CLOSEST | CUBIC
    # the light's own X/Y axes, for oriented projection. Export fills them
    # from the object matrix; hand-built scenes may leave them None and get
    # a stable basis derived from the direction.
    frame_x: Any = None
    frame_y: Any = None
    # Runtime
    shadow_map: Any = None


@dataclass
class Camera:
    matrix_world: Any = None          # (4,4)
    projection: Any = None            # (4,4), from calc_matrix_camera
    type: str = 'PERSP'
    lens: float = 50.0
    sensor: float = 36.0
    clip_start: float = 0.1
    clip_end: float = 1000.0
    ortho_scale: float = 6.0
    shift_x: float = 0.0
    shift_y: float = 0.0
    dof: bool = False
    focus_distance: float = 5.0
    fstop: float = 2.8


@dataclass
class World:
    # NODES|SOLID|GRADIENT|BANDS|STARFIELD|BRYCE|PHYSICAL|HDRI
    mode: str = 'NODES'
    strength: float = 1.0
    rotation: float = 0.0
    color: tuple = (0.05, 0.05, 0.06)
    ambient: tuple = (0.0, 0.0, 0.0)
    ambient_level: float = 1.0
    #: BI's Exposure panel (wrld_exposure_correct): 0 exposure at
    #: range 1 is the identity. linfac/logfac derive verbatim.
    exposure: float = 0.0
    exposure_range: float = 1.0
    graph: Optional[Dict[str, Any]] = None
    env_image: Optional[ImageBuffer] = None
    env_mapping: str = 'EQUIRECT'     # EQUIRECT | MIRRORBALL | SCREEN
    horizon: tuple = (0.55, 0.65, 0.80)
    zenith: tuple = (0.10, 0.25, 0.65)
    ground_color: tuple = (0.18, 0.15, 0.12)
    show_ground: bool = False
    horizon_height: float = 0.0
    gradient_falloff: float = 1.0
    blend_mode: str = 'LINEAR'        # LINEAR|SMOOTH|SHARP|EASE
    # R251 C038: the DS rear-plane depth bitmap (eye-space distance, row 0 = bottom)
    backdrop_depth: Optional[Any] = None
    backdrop_offset: tuple = (0, 0)
    # ------------------------------------------------ R251 sky-camera
    # C056 Cylinder Sky (Doom): the image rides env_image; Doom's numbers
    sky_cylinder_repeats: int = 4      # Doom: 1024 columns per turn over a 256-wide texture
    sky_cylinder_mid: float = 0.78125  # Doom: texture row 100 of 128 on the centre line
    # LightWave Backdrop (R251): four colours, hard horizon, whole-number squeezes
    # (C100; the LW 5-7 manual's example values as 8-bit/255, float32 literals)
    lw_zenith: tuple = (0.0, 0.156862745, 0.31372549)
    lw_sky: tuple = (0.470588235, 0.705882353, 0.941176471)
    lw_ground: tuple = (0.196078431, 0.156862745, 0.117647059)
    lw_nadir: tuple = (0.392156863, 0.31372549, 0.235294118)
    lw_sky_squeeze: int = 2
    lw_ground_squeeze: int = 2
    # ------------------------------------------------ R253 cube map
    # The 1990s skybox (CUBEMAP): six square faces from one packed cross /
    # strip image (which rides env_image, as CYLINDER's does) or from six
    # image slots named the OpenGL or the Quake 2 / Half-Life way. Every
    # default is neutral: no mode reads these unless mode == 'CUBEMAP'.
    cube_source: str = 'SINGLE'          # SINGLE | SIX
    cube_layout: str = 'AUTO'            # AUTO | HCROSS | VCROSS | HSTRIP | VSTRIP
    cube_convention: str = 'OPENGL'      # OPENGL | QUAKE2 (sky.CUBE_CONVENTIONS)
    cube_filter: str = 'NEAREST'         # NEAREST | BILINEAR, never across a seam
    cube_face_rot: tuple = (0, 0, 0, 0, 0, 0)       # quarter turns per slot, 0..3
    cube_face_flip: tuple = (False, False, False, False, False, False)
    cube_image_px: Optional[ImageBuffer] = None     # the six slots, positional:
    cube_image_nx: Optional[ImageBuffer] = None     # OpenGL +X -X +Y -Y +Z -Z or
    cube_image_py: Optional[ImageBuffer] = None     # Quake rt lf up dn bk ft
    cube_image_ny: Optional[ImageBuffer] = None
    cube_image_pz: Optional[ImageBuffer] = None
    cube_image_nz: Optional[ImageBuffer] = None
    # ------------------------------------------------ Bryce's Sky Lab
    # Bryce's Sky & Fog palette offered a Sky Mode: Soft Sky drove the dome
    # from the sun's own colour, Custom Sky exposed the three stops directly.
    sky_mode: str = 'CUSTOM'            # SOFT | CUSTOM
    sun_glow_color: tuple = (1.0, 0.86, 0.62)
    shadow_color: tuple = (0.30, 0.34, 0.45)
    shadow_intensity: float = 1.0
    # the Atmosphere tab keeps a base height for each of fog and haze, and
    # blends each toward the sun's colour and the sky's own
    fog_base_height: float = 0.0
    haze_base_height: float = 0.0
    fog_blend_sky: float = 0.0
    fog_sun_tint: float = 0.0
    color_perspective: float = 0.0
    volumetric_world: float = 0.0
    # the Cloud Cover tab's own names: frequency and amplitude, not "scale"
    cloud_frequency: float = 1.0
    cloud_amplitude: float = 1.0
    cloud_turbulence: float = 1.0
    spherical_clouds: bool = True
    # On by default, which is both what the control is for and what Halcyon
    # did before it existed: the cloud pattern does not react to the camera at
    # all. Off gives real parallax against cloud_height, which is a dome
    # parameter rather than a distance -- so a low deck slides a long way for
    # a small move, exactly as a low deck would.
    link_clouds_to_view: bool = True
    fixed_cloud_plane: bool = True
    stratus_frequency: float = 1.0
    stratus_amplitude: float = 1.0
    # the Celestial tab
    moon_softness: float = 0.05
    comets: float = 0.0
    comet_count: int = 3
    comet_speed: float = 0.05
    comet_length: float = 0.10
    comet_width: float = 0.006
    comet_tail_sun: float = 0.6
    comet_color: tuple = (1.0, 0.96, 0.88)
    # sun, shared by BRYCE and PHYSICAL
    sun_elevation: float = 0.35
    sun_rotation: float = 0.6
    sun_color: tuple = (1.0, 0.94, 0.82)
    sun_size: float = 0.03
    sun_intensity: float = 1.0
    sun_glow: float = 0.35
    sun_disc: bool = True
    # Bryce haze and clouds
    sun_corona: float = 1.0
    # Bryce's Sun & Moon: one body, swapped
    celestial: str = 'SUN'              # SUN|MOON
    moon_phase: float = 0.25            # 0 new, 0.5 full, 1 new again
    moon_color: tuple = (0.86, 0.88, 0.95)
    moon_size: float = 0.045
    moon_earthshine: float = 0.06
    # a third gradient stop, as Bryce's dome editor had
    sky_mid: tuple = (0.35, 0.50, 0.78)
    sky_mid_height: float = 0.35
    use_sky_mid: bool = True
    # atmosphere proper, rather than a single haze band
    atmosphere_density: float = 0.0
    atmosphere_falloff: float = 1.0
    atmosphere_color: tuple = (0.70, 0.78, 0.90)
    haze_blend_sky: float = 0.5
    # clouds drift, and shade what is under them
    cloud_wind: float = 0.0
    cloud_wind_angle: float = 0.0
    cloud_ambience: float = 0.35
    cloud_shadows: float = 0.0
    # Bryce keeps haze (distance/altitude) and fog (ground-hugging) separate
    haze_color: tuple = (0.82, 0.86, 0.92)
    haze_density: float = 0.45
    haze_height: float = 0.22
    haze_sun_tint: float = 0.5
    fog_color: tuple = (0.90, 0.90, 0.88)
    fog_density: float = 0.0
    fog_height: float = 0.05
    # cumulus deck
    clouds: bool = True
    cloud_color: tuple = (1.0, 1.0, 1.0)
    cloud_shadow: tuple = (0.42, 0.45, 0.55)
    cloud_cover: float = 0.5
    cloud_density: float = 0.95
    cloud_height: float = 1.0
    cloud_scale: float = 1.4
    cloud_detail: int = 5
    cloud_softness: float = 1.0
    cloud_thickness: float = 0.35
    cloud_rim: float = 0.4
    cloud_seed: int = 0
    # stratus deck
    stratus: bool = False
    stratus_color: tuple = (0.95, 0.95, 0.98)
    stratus_amount: float = 0.45
    stratus_density: float = 0.6
    stratus_altitude: float = 3.0
    stratus_scale: float = 3.0
    stratus_detail: int = 4
    stratus_sharpness: float = 1.4
    stratus_squash: float = 1.0
    # the Bryce extras
    rainbow: bool = False
    rainbow_intensity: float = 0.35
    rainbow_radius: float = 42.0
    rainbow_width: float = 3.0
    rainbow_secondary: float = 0.5
    stars: bool = False
    star_density: float = 0.5
    star_brightness: float = 0.8
    # starfield mode: stars all the way round, with no dome under them
    star_size: float = 0.35
    star_twinkle: float = 0.0
    # the pre-1.38 star drawing: cell-sized squares whose pixel size
    # followed the render resolution. Off = fixed-angular-size discs
    old_stars: bool = False
    nebula: float = 0.0
    nebula_color: tuple = (0.35, 0.15, 0.55)
    nebula_scale: float = 2.0
    nebula_detail: int = 5
    # banded gradient
    band_count: int = 8
    band_softness: float = 0.0
    # ------------------------------------------- R233: the painted sky
    # A background painting on a flat panel in front of the camera, the
    # way the animation stand's background sat behind the cel: the
    # gradient's colours brushed in gouache (streaks and, optionally,
    # dabs), painted clouds lit on top and shadowed beneath, the board's
    # tooth and a watercolour granulation. Everything reads the panel's
    # tangent coordinates, so a pan crosses the painting and a tilt
    # climbs it; directions behind the panel fade to the plain gradient
    # (the back of the stage). Evaluated per direction on the CPU on
    # both device roads, like every rich sky -- exact by construction.
    paint_look: str = 'CUSTOM'          # CUSTOM | GOUACHE_DAY | WATERCOLOUR_DUSK | FLEISCHER_NIGHT | STORYBOARD
    paint_angle: float = 90.0           # the panel's direction, degrees from +X (+Y default)
    paint_seed: int = 0
    paint_streaks: float = 0.35         # the brush's streaks on the gradient
    paint_streak_scale: float = 12.0    # streaks across 45 degrees of view
    paint_streak_angle: float = 0.0     # degrees off horizontal
    paint_dabs: float = 0.0             # impasto dabs over the streaks (costly)
    paint_dab_scale: float = 10.0
    paint_clouds: float = 0.35          # cloud coverage
    paint_cloud_scale: float = 2.2
    paint_cloud_softness: float = 0.3   # 0 dry-brush, 1 airbrushed
    paint_cloud_color: tuple = (0.97, 0.96, 0.93)
    paint_cloud_shadow: tuple = (0.58, 0.60, 0.70)
    paint_cloud_height: float = 0.05    # where the deck sits above the horizon
    paint_paper: float = 0.3            # the board's tooth
    paint_paper_scale: float = 8.0
    paint_wash: float = 0.25            # watercolour granulation (pigment density)
    # physical
    turbidity: float = 2.5
    ground_albedo: float = 0.3
    # HDRI
    # an infinite ground plane, intersected analytically in the background
    ground_plane: bool = False
    # SOLID|CHECKER|NOISE|TILES|DESERT|SNOW|LAVA|OCEAN|MATERIAL
    # ('GRID' from old exports still renders, as thin-glow TILES)
    ground_mode: str = 'SOLID'
    ground_height: float = 0.0
    ground_scale: float = 2.0
    ground_color2: tuple = (0.55, 0.52, 0.48)
    ground_fade: float = 60.0
    # R203: the tile dials the field could not reach, the lava rebuild
    # dials, and the material-ground graph (serialized at export from
    # the picked material; programs ride beside it)
    ground_grout: float = 0.04
    ground_tile_shade: float = 0.25
    ground_grout_glow: float = 1.0
    ground_crack_width: float = 0.35
    ground_glow: float = 1.0
    ground_pulse: float = 0.15
    # R204: the colours the field could see but not touch, and the
    # lighting dial that lets the plane answer the scene's lamps.
    # ground_color3 is the third tone (snow glints, lava embers);
    # ground_sparkle scales the snow glitter; ground_ridge is the
    # desert crest strength that used to be baked at 0.6; and
    # ground_lighting blends flat-emissive (0, the old pixels
    # bitwise) toward fully lit by sun, lamps and cast shadows (1).
    ground_color3: tuple = (1.0, 1.0, 1.0)
    ground_sparkle: float = 1.0
    ground_ridge: float = 0.6
    ground_lighting: float = 1.0
    ground_graph: Optional[Dict[str, Any]] = None
    ground_programs: Optional[Dict[str, Any]] = None
    # R251 C048 Mode 7 (SNES / GBA) floor: the ground group
    ground_image: Optional[ImageBuffer] = None   # R251 Mode 7 map (an image datablock: never in a preset)
    mode7_texel_size: float = 1.0
    mode7_over: str = 'WRAP'                # WRAP | TRANSPARENT | TILE0 (M7SEL)
    ocean_choppiness: float = 0.35
    ocean_speed: float = 1.0
    # the water, the other half of a Bryce picture
    ocean_wind_angle: float = 0.6
    ocean_spread: float = 0.6
    ocean_wave_scale: float = 1.0
    ocean_detail: int = 5
    ocean_sparkle: float = 1.0
    ocean_horizon_smooth: float = 0.0
    ocean_deep: tuple = (0.03, 0.09, 0.13)
    ocean_shallow: tuple = (0.06, 0.22, 0.26)
    ocean_glitter: float = 1.0
    ocean_glitter_size: float = 0.45
    ocean_foam: float = 0.0
    ocean_foam_color: tuple = (0.92, 0.95, 0.96)
    ocean_transparency: float = 0.25
    env_filter: str = 'BILINEAR'
    env_tint: tuple = (1.0, 1.0, 1.0)
    sky_blend: bool = False
    mist: bool = False
    mist_start: float = 5.0
    mist_depth: float = 25.0
    mist_color: tuple = (0.5, 0.55, 0.6)
    mist_falloff: str = 'LINEAR'      # LINEAR | QUADRATIC | INVERSE_QUADRATIC
    mist_intensity: float = 1.0
    # ------------------------------------------------ R200: Weather
    # a deterministic screen-space particle overlay IN FRONT of the
    # picture -- rain, snow, embers, ash -- composited over geometry
    # and sky alike, never replacing either. Angle 0 falls straight
    # down; pi rises straight up; between is the diagonal. All pure
    # functions of (seed, layer, particle, time): identical across
    # runs, devices, refine passes and supersample factors
    weather: str = 'NONE'             # NONE|RAIN|SNOW|EMBERS|ASH
    weather_density: float = 1.0
    weather_size: float = 1.0
    weather_speed: float = 1.0
    weather_angle: float = 0.0
    weather_drift: float = 0.2
    weather_color: tuple = (0.85, 0.90, 1.0)
    weather_opacity: float = 0.8
    weather_layers: int = 3
    weather_streak: float = 1.0
    weather_glow: float = 0.0
    weather_flicker: float = 0.0
    weather_seed: int = 0


@dataclass
class Scene:
    mesh: MeshData = None
    materials: List[Material] = field(default_factory=list)
    objects: List[ObjectInfo] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)
    camera: Camera = None
    world: World = field(default_factory=World)
    settings: Any = None              # RenderSettings (core.settings)
    frame: int = 1
    fps: float = 24.0
    time: float = 0.0
    unit_scale: float = 1.0
    #: R191 halo points: list of groups, each {'mat': material index,
    #: 'pos': (n,3) float32 world positions, 'seeds': (n,) ints
    #: (ma->seed1 + running vertex index, exactly make_render_halos),
    #: 'sizes': (n,) float32 (None = the material's size), 'normals':
    #: (n,3) world vertex normals (only when the material's puno asks
    #: for facing-scaled sizes)}. Built by the exporter from every
    #: mesh/particle/point-cloud object wearing a halo material.
    halos: Optional[List[Dict[str, Any]]] = None
    #: R251 C020/C036: the authored shadow volumes the exporter split off
    #: the surface mesh (one per (object, volume-role material) pair, in
    #: object order then material order): {'name', 'verts' (V,3) float32,
    #: 'tris' (T,3) int32, 'role' DC_INCLUDE|DC_EXCLUDE|DS_SHADOW,
    #: 'polygon_id', 'alpha', 'color'}. Never rasterised, never a caster,
    #: never inked, never in the BVH -- core/shadowmask.py reads them.
    shadow_volumes: list = field(default_factory=list)

    def tri_count(self):
        return 0 if self.mesh is None or self.mesh.tris is None else len(self.mesh.tris)


def clip_socket(mat):
    """How a CLIP material's alpha chain can be evaluated on its own.

    The punch-through road (R211) resolves visibility BEFORE shading:
    each clip fragment evaluates only the material's alpha and the
    survivors join the depth-buffered pass. That is only honest when
    the alpha genuinely IS one evaluable chain. Returns
    ('node', surface_node, socket_name) for a graph whose surface node
    exposes a plain Opacity/Alpha input, ('const', None, opacity) for a
    graph-less material, or (None, None, reason) naming exactly why the
    material's alpha cannot be lifted out -- those keep the blend road,
    where the shading law still forces CLIP alpha to hard 0/1.
    """
    g = getattr(mat, 'graph', None)
    if not g:
        return 'const', None, float(getattr(mat, 'opacity', 1.0))
    nodes = g.get('nodes', {}) or {}
    out = nodes.get(g.get('output'))
    if out is None:
        return 'const', None, float(getattr(mat, 'opacity', 1.0))
    link = None
    for s in out.get('inputs', ()):
        if s.get('name') == 'Surface':
            link = s.get('link')
            break
    if not link:
        return 'const', None, float(getattr(mat, 'opacity', 1.0))
    nd = nodes.get(link[0])
    if nd is None:
        return None, None, 'the surface link points at no node'
    bid = nd.get('bl_idname', '')
    sock = {'HALCYON_ShaderNode': 'Opacity',
            'ShaderNodeBsdfPrincipled': 'Alpha'}.get(bid)
    if sock is None:
        return None, None, (f'its surface is {bid or "unknown"}, whose '
                            'alpha is not one plain socket')
    if bid == 'HALCYON_ShaderNode':
        for s in nd.get('inputs', ()):
            if s.get('name') != 'Edge Opacity':
                continue
            d = s.get('default')
            if s.get('link') or (d is not None
                                 and abs(float(d) - 1.0) > 1e-4):
                return None, None, ('Edge Opacity shapes its '
                                    'silhouette after the alpha socket')
    return 'node', nd, sock


#: Math operations whose output is exactly 0.0 or 1.0 by construction
_BINARY_MATH = frozenset({'GREATER_THAN', 'LESS_THAN', 'COMPARE'})


def alpha_chain_is_binary(mat, nd, sockname):
    """True when the alpha chain PROVABLY yields only 0 or 1.

    The proof is structural and conservative: the node feeding the
    surface's alpha socket is a Math comparison (Greater Than, Less
    Than, Compare), whose output is exactly 0.0 or 1.0 whatever its
    inputs do. Every shell-fur material ever built by the Fur Shells
    operator ends in exactly that node.
    """
    g = getattr(mat, 'graph', None)
    if not g or nd is None:
        return False
    for sk in nd.get('inputs', ()):
        if sk.get('name') != sockname:
            continue
        link = sk.get('link')
        if not link:
            return False
        src = (g.get('nodes', {}) or {}).get(link[0])
        if src is None or src.get('bl_idname') != 'ShaderNodeMath':
            return False
        op = str((src.get('props') or {}).get('operation', 'ADD'))
        # the Math node's clamp flag cannot change a 0/1 output
        return op in _BINARY_MATH
    return False


def material_see_through(mat):
    """R251 (transparency pack): the ONE see-through predicate of the
    transparent split (`_split_by_alpha`) and the GPU layer loop --
    the reason string a material rasterises into the A-buffer for, or
    None when it stays in the opaque pass. Order: an Env hole is opaque
    for the split (the sky is drawn through it, C126); a constant
    opacity below 1; the exporter's alpha evidence; a per-material
    Blend Mode that is neither Inherit nor Alpha (a PS1 additive glow at
    opacity 1.0 must reach the transparent pass -- the mode is NOT
    alpha evidence, `_alpha_reason` never reads it); the PS2 two-pass
    Clip+Blend (C031). Both roads call this, so the layer plan and the
    split can never disagree on which material holds fragments."""
    mode = str(getattr(mat, 'blend_mode', 'INHERIT') or 'INHERIT')
    if mode == 'ENV_HOLE':
        return None
    opacity = float(getattr(mat, 'opacity', 1.0))
    if opacity < 0.999:
        return f'Opacity {opacity:.3f}'
    if getattr(mat, 'has_alpha', False):
        return str(getattr(mat, 'alpha_why', None)
                   or 'flagged see-through on export')
    if mode not in ('INHERIT', 'ALPHA'):
        return f'Blend Mode {mode}'
    if str(getattr(mat, 'alpha_mode', 'BLEND')) == 'CLIP_BLEND':
        return 'Alpha Mode Clip+Blend (PS2 two-pass)'
    return None


def material_is_absent(mat):
    """R247: a CLIP material whose alpha is a CONSTANT below its
    threshold has no fragment anywhere -- it draws no surface in ANY
    transparency mode, the way a volume container draws none.

    The punch-through law says a CLIP material is fully there or fully
    absent, the same on every road; with Transparency None or Screen
    Door the clip stage does not run and such a material rendered
    SOLID -- a coincident outline shell (the Sparking! ZERO import's
    MI_LNE000, the game's inverted hull left flat on the body) then
    z-fought the body into black speckles under those two modes and
    was clean under the other two. A constant alpha needs no stage to
    resolve: the surface is not there. Structural and conservative:
    the surface node's plain alpha socket, unlinked, at a default
    below the threshold; or a graph-less material's own opacity.
    """
    if str(getattr(mat, 'alpha_mode', 'BLEND')) != 'CLIP':
        return False
    kind, nd, extra = clip_socket(mat)
    thr = max(float(getattr(mat, 'alpha_clip', 0.5)), 1e-6)
    if kind == 'const':
        return float(extra) < thr
    if kind != 'node' or nd is None:
        return False
    for sk in nd.get('inputs', ()):
        if sk.get('name') != extra:
            continue
        if sk.get('link'):
            return False
        d = sk.get('default')
        try:
            return d is not None and float(d) < thr
        except (TypeError, ValueError):
            return False
    return False


def clip_road(mat):
    """The one predicate for the punch-through road (R213).

    Returns (threshold, kind, node, sock_or_const, note) when the
    material's alpha resolves in the z-pass, else (None, None, None,
    None, why). Two ways on:

    - Alpha Mode CLIP, with a liftable chain (R211) -- the user asked.
    - Alpha Mode BLEND whose chain is PROVABLY binary (R213: the field
      frame spent 77 of 81 seconds blend-compositing a Greater Than
      that only ever says 0 or 1). For binary alpha the blend road and
      the z-buffer produce the identical picture -- except the blend
      road also sorts, caps layers, and refuses the GPU -- so the
      engine takes the fast road automatically. Old scenes' fur
      materials, built before Alpha Mode existed, are exactly this.
    """
    kind, nd, extra = clip_socket(mat)
    mode = str(getattr(mat, 'alpha_mode', 'BLEND'))
    if mode in ('CLIP', 'CLIP_BLEND'):
        # (R251 C031: Clip+Blend's opaque half rides the same road;
        # the note string stays 'Alpha Mode Clip')
        if kind is None:
            return None, None, None, None, str(extra)
        return (float(getattr(mat, 'alpha_clip', 0.5)), kind, nd, extra,
                'Alpha Mode Clip')
    if kind == 'node' and alpha_chain_is_binary(mat, nd, str(extra)):
        # any threshold strictly inside (0, 1) tests a 0/1 chain
        # identically; 0.5 is the canonical one
        return 0.5, kind, nd, extra, 'binary alpha, detected'
    return None, None, None, None, 'Alpha Mode Blend'
