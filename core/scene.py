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
    wire: bool = False
    wire_size: float = 1.0
    face_texture: bool = False
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
    colors: np.ndarray = None         # (V,4) float32
    tris: np.ndarray = None           # (T,3) int32 indices into verts
    mat_index: np.ndarray = None      # (T,) int32 -> index into Scene.materials
    obj_index: np.ndarray = None      # (T,) int32 -> index into Scene.objects
    face_normals: np.ndarray = None   # (T,3) float32
    smooth: np.ndarray = None         # (T,) bool


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
    cast_shadow: bool = True
    receive_shadow: bool = True
    holdout: bool = False


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
    # Area
    area_size: tuple = (1.0, 1.0)
    area_shape: str = 'SQUARE'
    area_x: tuple = (1.0, 0.0, 0.0)
    area_y: tuple = (0.0, 1.0, 0.0)
    # 90s-style decay
    decay: str = 'DEFAULT'     # NONE | INVERSE | INVERSE_SQUARE | CUSTOM
    decay_start: float = 0.0
    decay_end: float = 25.0
    # Shadowing
    shadow: str = 'MAP'               # NONE | MAP | RAY
    shadow_map_size: int = 512
    shadow_bias: float = 0.02
    shadow_softness: float = 1.0      # shadow-map blur radius in texels
    shadow_samples: int = 4
    shadow_color: tuple = (0.0, 0.0, 0.0)
    shadow_density: float = 1.0
    # Period features
    negative: bool = False
    diffuse_only: bool = False
    specular_only: bool = False
    affect_diffuse: bool = True
    affect_specular: bool = True
    volumetric: float = 0.0
    exclude_objects: tuple = ()
    exclude_mode: str = 'EXCLUDE'
    ambient_only: bool = False
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
    nebula: float = 0.0
    nebula_color: tuple = (0.35, 0.15, 0.55)
    nebula_scale: float = 2.0
    nebula_detail: int = 5
    # banded gradient
    band_count: int = 8
    band_softness: float = 0.0
    # physical
    turbidity: float = 2.5
    ground_albedo: float = 0.3
    # HDRI
    # an infinite ground plane, intersected analytically in the background
    ground_plane: bool = False
    ground_mode: str = 'SOLID'          # SOLID|CHECKER|NOISE|OCEAN
    ground_height: float = 0.0
    ground_scale: float = 2.0
    ground_color2: tuple = (0.55, 0.52, 0.48)
    ground_fade: float = 60.0
    ocean_choppiness: float = 0.35
    ocean_speed: float = 1.0
    env_filter: str = 'BILINEAR'
    env_tint: tuple = (1.0, 1.0, 1.0)
    sky_blend: bool = False
    mist: bool = False
    mist_start: float = 5.0
    mist_depth: float = 25.0
    mist_color: tuple = (0.5, 0.55, 0.6)
    mist_falloff: str = 'LINEAR'      # LINEAR | QUADRATIC | INVERSE_QUADRATIC
    mist_intensity: float = 1.0


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

    def tri_count(self):
        return 0 if self.mesh is None or self.mesh.tris is None else len(self.mesh.tris)
