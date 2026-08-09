"""Node classes for Halcyon's procedural textures.

Generated from one spec table so a socket cannot drift away from the evaluator
that reads it — the same reason the render settings are generated from their
dataclass.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Node

from .shader_nodes import HalcyonNodeBase

AXIS = (('X', "X", ""), ('Y', "Y", ""), ('Z', "Z", ""))

F = 'NodeSocketFloat'
C = 'NodeSocketColor'
V = 'NodeSocketVector'

GREY_A = (0.85, 0.85, 0.85, 1.0)
GREY_B = (0.18, 0.18, 0.18, 1.0)

# name, label, icon, description, sockets, enum/int props, outputs
SPECS = [
    ('Marble', "Marble", 'TEXTURE',
     "Sine banding displaced by turbulence, as POV-Ray and 3D Studio made it",
     [(V, 'Vector', None), (F, 'Scale', 4.0), (F, 'Turbulence', 1.0),
      (F, 'Veins', 1.0), (F, 'Sharpness', 1.0),
      (C, 'Color 1', (0.92, 0.90, 0.86, 1.0)),
      (C, 'Color 2', (0.14, 0.13, 0.16, 1.0))],
     {'octaves': ('int', 5, 1, 10, "Octaves"),
      'axis': ('enum', 'X', AXIS, "Vein Axis")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Wood', "Wood", 'TEXTURE',
     "Concentric growth rings around an axis, warped by turbulence",
     [(V, 'Vector', None), (F, 'Scale', 2.0), (F, 'Rings', 8.0),
      (F, 'Turbulence', 0.35), (F, 'Grain', 0.4),
      (C, 'Color 1', (0.42, 0.26, 0.12, 1.0)),
      (C, 'Color 2', (0.68, 0.47, 0.24, 1.0))],
     {'octaves': ('int', 4, 1, 10, "Octaves"),
      'axis': ('enum', 'Z', AXIS, "Trunk Axis")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Granite', "Granite", 'TEXTURE',
     "Stacked high-frequency noise stretched into speckled stone",
     [(V, 'Vector', None), (F, 'Scale', 12.0), (F, 'Contrast', 1.6),
      (F, 'Speckle', 0.35), (C, 'Color 1', GREY_A), (C, 'Color 2', GREY_B)],
     {'octaves': ('int', 6, 1, 10, "Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Dents', "Dents", 'MOD_DISPLACE',
     "Sparse pitted dents, the 3D Studio material of the same name",
     [(V, 'Vector', None), (F, 'Scale', 6.0), (F, 'Size', 1.0), (F, 'Depth', 1.0),
      (C, 'Color 1', GREY_A), (C, 'Color 2', GREY_B)],
     {'octaves': ('int', 3, 1, 8, "Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Crackle', "Crackle", 'MOD_EXPLODE',
     "The boundary network between cells: crazed glaze, dried mud, veins",
     [(V, 'Vector', None), (F, 'Scale', 6.0), (F, 'Randomness', 1.0),
      (F, 'Width', 0.06), (F, 'Smooth', 0.02),
      (C, 'Color 1', (0.05, 0.05, 0.05, 1.0)), (C, 'Color 2', GREY_A)],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('Plasma', "Plasma", 'COLORSET_10_VEC',
     "Interfering sine fields with optional palette cycling. The demoscene one",
     [(V, 'Vector', None), (F, 'Scale', 3.0), (F, 'Complexity', 3.0),
      (F, 'Speed', 1.0), (C, 'Color 1', (0.0, 0.0, 0.6, 1.0)),
      (C, 'Color 2', (1.0, 0.3, 0.0, 1.0))],
     {'animate': ('bool', True, "Animate"),
      'cycle_palette': ('bool', True, "Cycle Palette")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Ripples', "Ripples", 'MOD_WAVE',
     "Concentric waves from several sources, interfering",
     [(V, 'Vector', None), (F, 'Scale', 2.0), (F, 'Frequency', 8.0),
      (F, 'Decay', 0.6), (F, 'Speed', 1.0), (C, 'Color 1', GREY_B),
      (C, 'Color 2', GREY_A)],
     {'sources': ('int', 3, 1, 12, "Sources"),
      'seed': ('int', 0, 0, 9999, "Seed"),
      'animate': ('bool', True, "Animate")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Starfield', "Starfield", 'OUTLINER_OB_LIGHT',
     "Randomly placed stars with size and twinkle, for 1990s space scenes",
     [(V, 'Vector', None), (F, 'Scale', 40.0), (F, 'Density', 0.5),
      (F, 'Size', 0.35), (F, 'Twinkle', 0.0),
      (C, 'Sky Color', (0.0, 0.0, 0.02, 1.0)),
      (C, 'Star Color', (1.0, 1.0, 0.95, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('Weave', "Weave", 'MOD_CLOTH',
     "Over-under fabric weave with separate warp and weft threads",
     [(V, 'Vector', None), (F, 'Scale', 8.0), (F, 'Thickness', 0.35),
      (F, 'Gap', 0.08), (F, 'Distortion', 0.0),
      (C, 'Warp Color', (0.65, 0.18, 0.18, 1.0)),
      (C, 'Weft Color', (0.20, 0.22, 0.45, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac'), (F, 'Thread')]),

    ('Scratches', "Scratches", 'MOD_NOISE',
     "Fine anisotropic scuffs, for brushed and worn metal",
     [(V, 'Vector', None), (F, 'Scale', 3.0), (F, 'Width', 0.02),
      (F, 'Length', 1.0), (F, 'Anisotropy', 1.0),
      (C, 'Color 1', (0.05, 0.05, 0.05, 1.0)), (C, 'Color 2', (1.0, 1.0, 1.0, 1.0))],
     {'count': ('int', 6, 1, 64, "Count"), 'seed': ('int', 0, 0, 9999, "Seed")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Tiles', "Tiles", 'MESH_GRID',
     "Rectangular tiles with grout, row offset, bevel shading and per-tile variation",
     [(V, 'Vector', None), (F, 'Scale', 1.0), (F, 'Rows', 4.0),
      (F, 'Columns', 4.0), (F, 'Grout', 0.06), (F, 'Offset', 0.0),
      (F, 'Bevel', 0.15), (F, 'Variation', 0.25),
      (C, 'Tile Color', (0.75, 0.72, 0.66, 1.0)),
      (C, 'Grout Color', (0.30, 0.29, 0.27, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac'), (F, 'Tile ID')]),

    ('MatcapUV', "Matcap Coordinates", 'MATSPHERE',
     "Sphere-map coordinates from the view-space normal. Feed the Vector into "
     "an Image Texture of a lit sphere, then into the shader's Matcap input",
     [(F, 'Scale', 1.0)], {},
     [(V, 'Vector'), (F, 'Facing')]),

    ('Spiral', "Spiral", 'FORCE_VORTEX',
     "Archimedean spiral banding around an axis",
     [(V, 'Vector', None), (F, 'Scale', 2.0), (F, 'Turns', 4.0),
      (F, 'Sharpness', 1.0), (F, 'Twist', 0.0),
      (C, 'Color 1', GREY_B), (C, 'Color 2', GREY_A)],
     {'axis': ('enum', 'Z', AXIS, "Axis")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Noise', "Fractal Noise", 'MOD_OCEAN',
     "The raw integer-hash fractal field, three profiles. Reach for this "
     "where Blender's own Noise texture would go: that one's sin-fract hash "
     "cannot travel to the GPU, this one travels exactly",
     [(V, 'Vector', None), (F, 'Scale', 4.0), (F, 'Lacunarity', 2.0),
      (F, 'Gain', 0.5), (C, 'Color 1', GREY_B), (C, 'Color 2', GREY_A)],
     {'kind': ('enum', 'SMOOTH', (('SMOOTH', "Smooth", "Plain fBm"),
                                  ('TURBULENT', "Turbulent",
                                   "Folded noise; the cusps are the point"),
                                  ('RIDGED', "Ridged",
                                   "Folded and squared -- Musgrave's ridge "
                                   "profile, for mountains and veins")),
               "Profile"),
      'octaves': ('int', 5, 1, 10, "Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Cells', "Cells (Worley)", 'LIGHTPROBE_VOLUME',
     "Worley's cellular texture (SIGGRAPH 1996): distances to scattered "
     "feature points, or the cells themselves as flat shades",
     [(V, 'Vector', None), (F, 'Scale', 4.0), (F, 'Randomness', 1.0),
      (C, 'Color 1', (0.05, 0.05, 0.05, 1.0)), (C, 'Color 2', GREY_A)],
     {'feature': ('enum', 'F1', (('F1', "F1 (Nearest)",
                                  "Distance to the nearest point"),
                                 ('F2', "F2 (Second)",
                                  "Distance to the second point"),
                                 ('BORDER', "F2 - F1",
                                  "Ridges along the cell borders"),
                                 ('CELL', "Cell Shade",
                                  "Each cell a flat hashed shade -- the "
                                  "stained-glass look")),
                  "Feature")},
     [(C, 'Color'), (F, 'Fac'), (F, 'Cell ID')]),

    ('Static', "TV Static", 'IMAGE_ALPHA',
     "Per-cell white noise reseeded every frame -- an untuned television. "
     "It lives on the SURFACE (scale sets the set's pixel size), so it "
     "sits on an in-scene screen the way it should",
     [(V, 'Vector', None), (F, 'Scale', 64.0),
      (C, 'Color 1', (0.02, 0.02, 0.02, 1.0)),
      (C, 'Color 2', (0.9, 0.9, 0.9, 1.0))],
     {'animate': ('bool', True, "Animate")},
     [(C, 'Color'), (F, 'Fac')]),

    # ------------------------------------------- the POV-Ray pattern family

    ('Bozo', "Bozo", 'TEXTURE',
     "POV-Ray's bozo: plain noise under a colour map, optionally displaced by "
     "turbulence. Half the materials of the era started here",
     [(V, 'Vector', None), (F, 'Scale', 4.0), (F, 'Turbulence', 0.0),
      (F, 'Lacunarity', 2.0),
      (C, 'Color 1', (0.20, 0.35, 0.70, 1.0)),
      (C, 'Color 2', (1.0, 1.0, 1.0, 1.0))],
     {'octaves': ('int', 4, 1, 10, "Turbulence Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Agate', "Agate", 'TEXTURE',
     "POV-Ray's agate: a band thrown about by a large turbulence and raised to "
     "0.77, which is what makes it read as layered stone",
     [(V, 'Vector', None), (F, 'Scale', 2.0), (F, 'Turbulence', 1.0),
      (F, 'Bands', 1.1), (F, 'Sharpness', 0.77),
      (C, 'Color 1', (0.62, 0.40, 0.26, 1.0)),
      (C, 'Color 2', (0.96, 0.92, 0.84, 1.0))],
     {'octaves': ('int', 6, 1, 10, "Octaves"),
      'axis': ('enum', 'Z', AXIS, "Band Axis")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Leopard', "Leopard", 'TEXTURE',
     "POV-Ray's leopard: three sines summed and squared, leaving a rounded "
     "spot per cell. Every animal-print material was this",
     [(V, 'Vector', None), (F, 'Scale', 3.0), (F, 'Spot', 1.0),
      (C, 'Color 1', (0.86, 0.68, 0.30, 1.0)),
      (C, 'Color 2', (0.14, 0.10, 0.06, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('Onion', "Onion", 'MESH_UVSPHERE',
     "POV-Ray's onion: concentric spherical shells around the origin",
     [(V, 'Vector', None), (F, 'Scale', 3.0), (F, 'Thickness', 1.0),
      (F, 'Sharpness', 1.0),
      (C, 'Color 1', (0.90, 0.86, 0.70, 1.0)),
      (C, 'Color 2', (0.55, 0.42, 0.22, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('Bumps', "Bumps", 'MOD_SMOOTH',
     "POV-Ray's bumps: smooth noise read as a height field. Feed Fac to a Bump "
     "node, or to the shader's Displacement",
     [(V, 'Vector', None), (F, 'Scale', 3.0), (F, 'Roundness', 1.0),
      (F, 'Lacunarity', 2.0), (F, 'Gain', 0.5),
      (C, 'Color 1', GREY_B), (C, 'Color 2', GREY_A)],
     {'octaves': ('int', 1, 1, 8, "Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Wrinkles', "Wrinkles", 'MOD_CLOTH',
     "POV-Ray's wrinkles: folded noise summed at halving amplitude, creasing "
     "wherever an octave crosses zero. Crumpled paper and foil",
     [(V, 'Vector', None), (F, 'Scale', 4.0), (F, 'Lacunarity', 2.0),
      (F, 'Crease', 1.0), (C, 'Color 1', GREY_B), (C, 'Color 2', GREY_A)],
     {'octaves': ('int', 8, 1, 12, "Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Brick', "Brick", 'MOD_BUILD',
     "Running-bond brickwork with mortar courses, bevelled edges and a "
     "per-brick id so no two bricks need be the same colour",
     [(V, 'Vector', None), (F, 'Scale', 1.0), (F, 'Width', 0.25),
      (F, 'Height', 0.125), (F, 'Mortar', 0.05), (F, 'Offset', 0.5),
      (F, 'Bevel', 0.08), (F, 'Variation', 0.25),
      (C, 'Brick Color', (0.55, 0.24, 0.17, 1.0)),
      (C, 'Mortar Color', (0.78, 0.76, 0.72, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac'), (F, 'Brick ID')]),
]


def _make(name, label, icon, desc, sockets, props, outputs):
    ann = {}
    for key, spec in props.items():
        if spec[0] == 'int':
            _k, default, lo, hi, plabel = spec
            ann[key] = IntProperty(name=plabel, default=default, min=lo, max=hi)
        elif spec[0] == 'bool':
            _k, default, plabel = spec
            ann[key] = BoolProperty(name=plabel, default=default)
        else:
            _k, default, items, plabel = spec
            ann[key] = EnumProperty(name=plabel, items=list(items), default=default)

    def init(self, context):
        for kind, sock_name, default in sockets:
            sock = self.inputs.new(kind, sock_name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
        for kind, out_name in outputs:
            self.outputs.new(kind, out_name)

    def draw_buttons(self, context, layout):
        for key in props:
            layout.prop(self, key, text="")

    cls = type(f'HALCYON_{name}Node', (Node, HalcyonNodeBase), {
        '__doc__': desc,
        'bl_idname': f'HALCYON_{name}Node',
        'bl_label': label,
        'bl_icon': icon,
        'bl_width_default': 160,
        '__annotations__': ann,
        'init': init,
        'draw_buttons': draw_buttons,
    })
    return cls


NODES = tuple(_make(*spec) for spec in SPECS)

# what the exporter must copy across for each of them
NODE_PROPS = {f'HALCYON_{spec[0]}Node': tuple(spec[5].keys()) for spec in SPECS}


class NODE_MT_halcyon_textures(bpy.types.Menu):
    bl_idname = 'NODE_MT_halcyon_textures'
    bl_label = "Halcyon Textures"

    def draw(self, context):
        layout = self.layout
        for cls in NODES:
            op = layout.operator('node.add_node', text=cls.bl_label,
                                 icon=getattr(cls, 'bl_icon', 'NONE'))
            op.type = cls.bl_idname
            op.use_transform = True


def register():
    for cls in NODES:
        bpy.utils.register_class(cls)
    bpy.utils.register_class(NODE_MT_halcyon_textures)


def unregister():
    try:
        bpy.utils.unregister_class(NODE_MT_halcyon_textures)
    except Exception:                                           # noqa: BLE001
        pass
    for cls in reversed(NODES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass
