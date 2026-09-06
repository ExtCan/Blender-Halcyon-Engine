"""Node classes for Halcyon's procedural textures.

Generated from one spec table so a socket cannot drift away from the evaluator
that reads it — the same reason the render settings are generated from their
dataclass.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty, StringProperty
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
     "Sphere-map coordinates from the view-space normal or reflection. "
     "Feed the Vector into an Image Texture of a lit sphere, then into "
     "the shader's Matcap input -- or into a Gradient with Centered on, "
     "for a view-locked sphere gradient",
     [(F, 'Scale', 1.0), (V, 'Offset', None)],
     {'centered': ('bool', False,
                   "Output around the origin (-0.5..0.5) instead of image "
                   "space (0..1), so a Spherical gradient fed this vector "
                   "lands centred instead of cornered"),
      'source': ('enum', 'NORMAL',
                 (('NORMAL', "Normal",
                   "Project the view-space NORMAL -- the classic matcap, "
                   "BI's texco Nor"),
                  ('REFLECTION', "Reflection",
                   "Project the view-space REFLECTION vector -- the "
                   "env-map chrome trick, BI's texco Refl")),
                 "Source")},
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
     "cannot travel to the GPU, this one travels exactly. Dimensions picks "
     "the lattice: 1D along X, 2D flat, 3D solid, 4D with W as a free axis "
     "-- animate W for evolution, or drive it in a circle for a seamless "
     "loop",
     [(V, 'Vector', None), (F, 'Scale', 4.0), (F, 'W', 0.0),
      (F, 'Lacunarity', 2.0),
      (F, 'Gain', 0.5), (C, 'Color 1', GREY_B), (C, 'Color 2', GREY_A)],
     {'kind': ('enum', 'SMOOTH', (('SMOOTH', "Smooth", "Plain fBm"),
                                  ('TURBULENT', "Turbulent",
                                   "Folded noise; the cusps are the point"),
                                  ('RIDGED', "Ridged",
                                   "Folded and squared -- Musgrave's ridge "
                                   "profile, for mountains and veins")),
               "Profile"),
      'dims': ('enum', '3D', (('1D', "1D", "Along the vector's X only"),
                              ('2D', "2D", "The vector's X and Y"),
                              ('3D', "3D", "The full vector -- the classic"),
                              ('4D', "4D", "The vector plus the W socket; "
                               "drive W over time for evolving noise")),
               "Dimensions"),
      'octaves': ('int', 5, 1, 10, "Octaves")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Caustics', "Caustics", 'OUTLINER_OB_LIGHTPROBE',
     "The pool-light web: bright Voronoi cell edges, each cell's point "
     "slowly orbiting so the web writhes. Every 1990s pool floor, ocean "
     "shallows and water dungeon was this pattern; the matching lamp "
     "setting projects the same web as a light cookie",
     [(V, 'Vector', None), (F, 'Scale', 6.0), (F, 'Speed', 1.0),
      (C, 'Color 1', (0.02, 0.10, 0.16, 1.0)),
      (C, 'Color 2', (0.75, 0.95, 1.0, 1.0))],
     {'animate': ('bool', True, "Animate")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Water', "Water Noise", 'MOD_FLUIDSIM',
     "Layered animated water: 1 to 12 drifting noise layers, folded toward "
     "crests by Choppiness. Loop closes the animation exactly over Loop "
     "Frames by moving time onto a circle through the 4D lattice -- a "
     "looping deck breathes and shimmers in place rather than drifting, "
     "which is the honest trade for a perfect cycle",
     [(V, 'Vector', None), (F, 'Scale', 6.0), (F, 'Speed', 1.0),
      (F, 'Choppiness', 0.5),
      (C, 'Color 1', (0.04, 0.10, 0.18, 1.0)),
      (C, 'Color 2', (0.55, 0.75, 0.85, 1.0))],
     {'layers': ('int', 5, 1, 12, "Layers"),
      'animate': ('bool', True, "Animate"),
      'loop': ('bool', False, "Loop"),
      'loop_frames': ('int', 48, 2, 1000, "Loop Frames"),
      'fps': ('int', 24, 1, 240, "Frame Rate")},
     [(C, 'Color'), (F, 'Fac')]),

    ('Gradient', "Gradient (Shaped)", 'COLORSET_03_VEC',
     "A gradient with a chosen SHAPE -- linear, reflected, spherical, "
     "square, diamond, conical, spiral -- plus centre, rotation, repeat "
     "and easing. Feed it the Matcap Coordinates node (Centered on) for a "
     "view-locked sphere gradient, or a ramp for full colour control",
     [(V, 'Vector', None), (F, 'Scale', 1.0), (V, 'Center', None),
      (F, 'Rotation', 0.0),
      (C, 'Color 1', (0.0, 0.0, 0.0, 1.0)),
      (C, 'Color 2', (1.0, 1.0, 1.0, 1.0))],
     {'shape': ('enum', 'LINEAR', (
         ('LINEAR', "Linear", "Along X through the centre"),
         ('REFLECTED', "Reflected", "Peaks at the centre line, falls both "
          "ways"),
         ('SPHERICAL', "Spherical", "Falls with distance from the centre"),
         ('QUADRATIC', "Quadratic Sphere", "Spherical squared -- softer "
          "shoulder"),
         ('SQUARE', "Square", "Falls with chessboard distance -- square "
          "rings"),
         ('DIAMOND', "Diamond", "Falls with taxicab distance -- diamond "
          "rings"),
         ('CONICAL', "Conical", "Sweeps the angle around the centre"),
         ('SPIRAL', "Spiral", "The conical sweep advanced by distance")),
         "Shape"),
      'repeat': ('enum', 'NONE', (
          ('NONE', "Clip", "Hold 0 and 1 outside the ramp"),
          ('REPEAT', "Repeat", "Wrap the ramp endlessly"),
          ('PINGPONG', "Ping-Pong", "Wrap back and forth, seamlessly")),
          "Repeat"),
      'easing': ('enum', 'NONE', (
          ('NONE', "Linear", "The raw ramp"),
          ('SMOOTH', "Smooth", "Ease both ends (smoothstep)"),
          ('SHARP', "Sharp", "Square the ramp -- fast start, slow end")),
          "Easing")},
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

    ('FurTufts', "Fur Tufts", 'STRANDS',
     "The shell-fur texture: round strand cross-sections, one per cell, "
     "whose Height peaks at the tuft's centre and falls to zero at its "
     "rim. Test Height against Hair Info's Intercept on a shell stack "
     "and each layer keeps a smaller disc, so tufts taper into strand "
     "tips instead of square columns. Sample it in UV space so the "
     "shells line up",
     [(V, 'Vector', None), (F, 'Scale', 48.0), (F, 'Coverage', 1.0),
      (F, 'Taper', 1.0), (F, 'Variation', 0.35),
      (C, 'Color 1', (0.10, 0.06, 0.04, 1.0)),
      (C, 'Color 2', (0.55, 0.38, 0.22, 1.0))],
     {},
     [(C, 'Color'), (F, 'Height'), (F, 'Random')]),

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
     "Leopard rosettes: dark rings broken into arcs on jittered cells, a "
     "warmer patch inside each, the odd solid spot between. Spot sizes the "
     "rosettes, Break widens the ring gaps; Fac is the dark-ring mask",
     [(V, 'Vector', None), (F, 'Scale', 3.0), (F, 'Spot', 1.0),
      (F, 'Jitter', 0.85), (F, 'Break', 0.55),
      (C, 'Color 1', (0.82, 0.63, 0.37, 1.0)),
      (C, 'Color 2', (0.11, 0.07, 0.04, 1.0)),
      (C, 'Interior', (0.58, 0.38, 0.16, 1.0))],
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

# ------------------------------------------------------- the 2D media (R232)
#
# The hand's marks as textures AND converters: every one but Paint Strokes
# and Paper takes a per-pixel TONE (lightness, 1 = bare paper) -- a Shader
# to RGB luminance, a Facing, a dot with a light direction -- and lays the
# medium down to that tone, so a lit 3D surface reads as a hatched, stippled,
# scribbled, charcoaled or washed drawing. Unlinked, the tone is a dial and
# the node is a plain texture. Space puts the marks on the surface or on the
# camera's own paper; Boil redraws them every N frames.

PAPER = (0.93, 0.91, 0.86, 1.0)
INK = (0.08, 0.07, 0.09, 1.0)
GRAPHITE = (0.18, 0.17, 0.19, 1.0)

MEDIA_PROPS = {
    'space': ('enum', 'VECTOR', (
        ('VECTOR', "Vector",
         "The Vector input (generated coordinates when unlinked): the "
         "marks sit on the surface and move with it"),
        ('SCREEN', "Screen",
         "The camera's own paper: the marks sit on the frame, Scale "
         "cells across its height, the way a drawing does -- the object "
         "moves under them (the 'shower door'; Boil hides it)"),
        ('VIEW', "View",
         "The camera's sphere: the marks sit on the world's directions "
         "as seen from the eye, Scale cells across a hemisphere, so a "
         "pan or a tilt leaves them fixed on the scene and only a dolly "
         "swims them -- the Obra Dinn answer to the shower door")),
        "Space"),
    'seed': ('int', 0, 0, 9999, "Seed"),
    'boil': ('int', 0, 0, 24, "Boil Frames"),
}

TONE = (F, 'Tone', 0.5)
#: R235: Winkenbach's indication -- 1 draws the medium everywhere, 0
#: leaves the paper bare; link a mask (a vertex colour, a texture, an
#: inverted Facing) and the drawing is detailed only where it says
INDICATION = (F, 'Indication', 1.0)
DIRECTION_PROP = ('enum', 'ANGLE', (
    ('ANGLE', "Angle",
     "Every stroke at the Angle: the flat, mechanical hatch"),
    ('FORM', "Form",
     "Strokes wrap the form: along the silhouette-parallel tangent (the "
     "normal crossed with the view) plus the Angle -- the isophote "
     "direction of the classic pen drawing. Twelve coherent lane fields "
     "at 15-degree steps, cross-faded, so a turning direction never "
     "breaks a lane"),
    ('SLOPE', "Slope",
     "Strokes run down the form: along the tangent that turns away from "
     "the eye fastest (the view on the surface) plus the Angle -- radial "
     "about a sphere's centre. The same twelve cross-faded fields")),
    "Direction")
PLACEMENT_PROP = ('enum', 'SIZE', (
    ('SIZE', "Size & Count",
     "Dots on a jittered lattice, present by the tone and growing a "
     "little as it darkens; Fine adds a second lattice in the darks"),
    ('COUNT', "Count",
     "Secord's rule: dots of ONE size whose number carries the tone -- "
     "three nested lattices (1, 4 and 16 dots a cell) in a fixed order, "
     "coarse first, so a darkening tone adds dots without moving one and "
     "the crowd stays evenly spread at every density")),
    "Placement")

SPECS += [
    ('Hatching', "Hatching", 'ALIGN_JUSTIFY',
     "Pen cross-hatching laid to a tone: layers of strokes fill in as the "
     "tone darkens (the tonal art map), each stroke wobbling, varying in "
     "pressure, drawn in lifted segments and breaking dry. Link a Shader "
     "to RGB luminance to Tone and a lit surface hatches itself; Screen "
     "space puts the strokes on the paper; Direction Form or Slope wraps "
     "the strokes round the form; Indication keeps them to where a mask "
     "says. Fac is the ink",
     [(V, 'Vector', None), (F, 'Scale', 12.0), TONE, (F, 'Angle', 45.0),
      (F, 'Width', 0.35), (F, 'Length', 6.0), (F, 'Wobble', 0.5),
      (F, 'Breaks', 0.2), INDICATION, (C, 'Color 1', PAPER),
      (C, 'Color 2', INK)],
     dict({'layers': ('int', 2, 1, 4, "Layers"), 'direction': DIRECTION_PROP},
          **MEDIA_PROPS),
     [(C, 'Color'), (F, 'Fac')]),

    ('Scribble', "Pencil Scribble", 'RNDCURVE',
     "Pencil scribble shading to a tone: strokes bent by a curling warp "
     "(past Curl 1 they fold and loop), fanned in shallow layers, the "
     "graphite catching the paper's tooth so light pressure speckles and "
     "heavy pressure fills; Blend is the tortillon smearing it into the "
     "valleys. Fac is the graphite",
     [(V, 'Vector', None), (F, 'Scale', 10.0), TONE, (F, 'Angle', 30.0),
      (F, 'Width', 0.3), (F, 'Curl', 0.5), (F, 'Pressure', 0.7),
      (F, 'Grain', 0.6), (F, 'Blend', 0.0), INDICATION,
      (C, 'Color 1', PAPER), (C, 'Color 2', GRAPHITE)],
     dict({'layers': ('int', 2, 1, 4, "Layers"), 'direction': DIRECTION_PROP},
          **MEDIA_PROPS),
     [(C, 'Color'), (F, 'Fac')]),

    ('Stipple', "Stipple", 'SNAP_GRID',
     "Pen stippling to a tone: dots on a jittered lattice whose DENSITY "
     "carries the tone, growing a little as it darkens; Fine adds a "
     "second, finer lattice in the darks so a full black can close. "
     "Placement Count is Secord's stippling: one dot size, the number "
     "carries the tone, a nested crowd. Fac is the ink",
     [(V, 'Vector', None), (F, 'Scale', 30.0), TONE, (F, 'Size', 0.6),
      (F, 'Jitter', 0.8), (F, 'Fine', 1.0), INDICATION,
      (C, 'Color 1', PAPER), (C, 'Color 2', INK)],
     dict({'placement': PLACEMENT_PROP}, **MEDIA_PROPS),
     [(C, 'Color'), (F, 'Fac')]),

    ('Charcoal', "Charcoal", 'MOD_NOISE',
     "Charcoal to a tone: the stick deposits on the paper's tooth, peaks "
     "first, streaked along Angle, smudged by a slow thumb, its own "
     "pressure banding across the stroke -- a light tone speckles, a dark "
     "one fills; Blend is the stump flattening the tooth. Fac is the "
     "charcoal",
     [(V, 'Vector', None), (F, 'Scale', 6.0), TONE, (F, 'Angle', 20.0),
      (F, 'Grain', 0.6), (F, 'Streak', 0.5), (F, 'Smudge', 0.4),
      (F, 'Blend', 0.0), INDICATION,
      (C, 'Color 1', PAPER), (C, 'Color 2', (0.12, 0.11, 0.11, 1.0))],
     dict(MEDIA_PROPS), [(C, 'Color'), (F, 'Fac')]),

    ('PaintStrokes', "Paint Strokes", 'COLOR',
     "Brush dabs, later ones over earlier: rounded dabs Length by Width "
     "cells, turned off Angle by up to Spread, bristle-streaked, drying "
     "toward their ends, each lightened or darkened by Variation. Color "
     "2 is the paint (link a lit colour and the strokes carry it), Color 1 "
     "the canvas between dabs; Fac is the paint value, Stroke ID each "
     "dab's own random",
     [(V, 'Vector', None), (F, 'Scale', 8.0), (F, 'Angle', 25.0),
      (F, 'Length', 2.6), (F, 'Width', 0.85), (F, 'Spread', 0.5),
      (F, 'Bristles', 0.6), (F, 'Variation', 0.3),
      (C, 'Color 1', (0.90, 0.87, 0.80, 1.0)),
      (C, 'Color 2', (0.55, 0.45, 0.30, 1.0))],
     dict(MEDIA_PROPS), [(C, 'Color'), (F, 'Fac'), (F, 'Stroke ID')]),

    ('Wash', "Ink Wash", 'MOD_FLUIDSIM',
     "An ink or watercolour wash to a tone: Levels flat washes, each "
     "adding a share of the dark where its threshold passes, pigment "
     "pooling inside every wash's edge, settling into the paper's "
     "granulation, the edges wandering with Bleed. Fac is the pigment",
     [(V, 'Vector', None), (F, 'Scale', 4.0), TONE, (F, 'Pooling', 0.6),
      (F, 'Granulation', 0.4), (F, 'Bleed', 0.4), INDICATION,
      (C, 'Color 1', PAPER), (C, 'Color 2', (0.10, 0.15, 0.30, 1.0))],
     dict({'levels': ('int', 3, 1, 6, "Levels")}, **MEDIA_PROPS),
     [(C, 'Color'), (F, 'Fac')]),

    ('Paper', "Paper", 'FILE_TEXT',
     "Paper: fine Tooth, long Fibres, slow Mottle, as a height in 0..1 "
     "(Fac) and a colour between the two. Multiply it under a drawing, or "
     "feed Fac to a Bump node for the sheet's own relief",
     [(V, 'Vector', None), (F, 'Scale', 6.0), (F, 'Tooth', 0.6),
      (F, 'Fibres', 0.3), (F, 'Mottle', 0.4),
      (C, 'Color 1', (0.86, 0.83, 0.76, 1.0)),
      (C, 'Color 2', (0.97, 0.96, 0.92, 1.0))],
     {'space': MEDIA_PROPS['space'], 'seed': MEDIA_PROPS['seed']},
     [(C, 'Color'), (F, 'Fac')]),
]


#: R241: every generated property carries a real tooltip. The names
#: mean the same thing on every node that uses them, so one table
#: serves the whole shelf; a node wanting different words can override
#: by (bl_idname, prop) below.
_PROP_TIPS = {
    'octaves': "Fractal detail: how many noise layers stack, each "
               "half the size of the last. More octaves add fine "
               "grain and cost a little more",
    'seed': "Picks a different random variation of the same pattern. "
            "Change it until the marks fall where you like; the same "
            "seed always redraws the same marks",
    'space': "Where the marks live: on the surface (they stick to the "
             "object as it moves), on the camera's paper (the "
             "media-on-glass look), or in view space (marks hold "
             "still while the camera turns)",
    'boil': "Redraws the marks every this-many frames -- the hand-"
            "drawn boil of animation shot on twos or threes. 0 keeps "
            "one drawing for the whole shot",
    'animate': "Drives the pattern with the scene clock so it moves "
               "on its own as the frames advance; off freezes it at "
               "its Phase or Offset value",
    'axis': "Which object axis the pattern runs along -- a marble's "
            "veins, a wood's grain, a spiral's spine",
    'layers': "How many tone layers the marks build up through: each "
              "darker tone adds another pass of marks over the last, "
              "the way a hand fills a drawing in",
    'direction': "How the strokes run: at the Stroke Angle, wrapped "
                 "along the form's silhouette, or down the form's "
                 "slope -- the hatcher's three habits",
    'cycle_palette': "Runs the plasma through its palette over time "
                     "(the demoscene colour-cycle trick) instead of "
                     "holding one mapping still",
    'sources': "How many ripple centres spread rings across the "
               "surface -- one calm drop or a whole rain",
    'count': "How many scratches are drawn across the pattern's "
             "space; each keeps its own place by the Seed",
    'centered': "Centres the matcap coordinates on the object rather "
                "than the screen, so the capture sticks to the form "
                "instead of sliding with the framing",
    'source': "Which normal feeds the matcap lookup: the shading "
              "normal (bumps included) or the smooth geometric one",
    'kind': "The noise's character: smooth value noise, billowed "
            "ridges, or sharp ridged creases",
    'dims': "Whether the noise varies in 2D (flat, stable under "
            "motion along the third axis) or full 3D",
    'loop': "Makes the water's motion repeat seamlessly every Loop "
            "Frames, for cycles and game textures",
    'loop_frames': "The length of the seamless water cycle, in "
                   "frames, when Loop is on",
    'fps': "The clock the water's motion counts in, frames per "
           "second -- match the scene's rate for on-twos work",
    'shape': "The gradient's geometry: linear, reflected, spherical "
             "falloff, square or diamond rings, a conical sweep or a "
             "spiral of it",
    'repeat': "What happens past the ramp's end: hold the last "
              "colour, wrap around, or bounce back and forth "
              "seamlessly",
    'easing': "The ramp's pace: linear, eased at both ends "
              "(smoothstep), or squared for a fast start and slow "
              "finish",
    'feature': "What the cell field measures: the distance to the "
               "nearest point, the second nearest, their difference "
               "(cell walls), or the cell's own flat colour",
    'placement': "How the stipple fills tone: one dot size whose "
                 "spacing carries the grey, or counted dots that only "
                 "ever ADD as the tone darkens",
    'levels': "How many flat pools the wash quantizes into before "
              "its edges pool and darken -- fewer reads bolder",
}

#: thin item docs the specs left empty, by identifier
_ITEM_TIPS = {
    'X': "Run the pattern along the object's X axis",
    'Y': "Run the pattern along the object's Y axis",
    'Z': "Run the pattern along the object's Z axis",
    'SMOOTH': "Smooth value noise -- the plain rolling field",
    'NONE': "No shaping: the raw linear ramp as measured",
}


def _make(name, label, icon, desc, sockets, props, outputs, shows=None):
    """`shows` (R243): {socket or prop name: (controller prop, {values})}
    -- the socket or property is shown only while the controller enum
    holds one of the values (Max's Color Correction shows Standard OR
    Advanced lightness controls, the rewire menus only under Custom).
    A linked socket always stays visible."""
    shows = dict(shows or {})
    controllers = {rule[0] for rule in shows.values()}

    def _apply_shows(self):
        for sock in self.inputs:
            rule = shows.get(sock.name)
            if rule is None:
                continue
            ctl, values = rule
            want = str(getattr(self, ctl, '')) in values
            try:
                sock.hide = bool(not want and not sock.is_linked)
            except (AttributeError, TypeError):
                pass

    def _on_controller(self, context):
        _apply_shows(self)

    ann = {}
    for key, spec in props.items():
        tip = _PROP_TIPS.get(key, '')
        if spec[0] == 'int':
            _k, default, lo, hi, plabel = spec
            ann[key] = IntProperty(name=plabel, default=default, min=lo,
                                   max=hi, description=tip)
        elif spec[0] == 'bool':
            _k, default, plabel = spec
            ann[key] = BoolProperty(name=plabel, default=default,
                                    description=tip)
        elif spec[0] == 'string':
            # R243: a named layer (Max's Vertex Color map's Channel Name)
            _k, default, plabel = spec
            ann[key] = StringProperty(name=plabel, default=default,
                                      description=tip)
        else:
            _k, default, items, plabel = spec[:4]
            # R241: back-fill any item doc the spec left thin
            items = [(i[0], i[1],
                      (i[2] if len(i) > 2 and len(str(i[2])) >= 20
                       else _ITEM_TIPS.get(i[0], str(i[2]) if len(i) > 2
                                           else '')))
                     for i in items]
            kw = {}
            if key in controllers:
                kw['update'] = _on_controller
            ann[key] = EnumProperty(name=plabel, items=list(items),
                                    default=default, description=tip, **kw)

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
        if shows:
            _apply_shows(self)

    def draw_buttons(self, context, layout):
        # R242: only enums draw bare (their value IS the label); a
        # checkbox or a count without its name is unreadable
        for key, spec in props.items():
            rule = shows.get(key)
            if rule is not None and str(getattr(self, rule[0], '')) not in rule[1]:
                continue
            if spec[0] in ('int', 'bool', 'string'):
                layout.prop(self, key)
            elif len(spec) > 4 and spec[4]:
                # R243: an enum whose value alone would not say which
                # control it is (a rewired channel, a layer's blend)
                layout.prop(self, key)
            else:
                layout.prop(self, key, text="")

    def ensure_sockets(self):
        """R235: a node saved before a socket existed grows it at load
        (the master's own rule -- topology changes only at file load),
        in the spec's order, at the spec's default."""
        names = [s[1] for s in sockets]
        for kind, sock_name, default in sockets:
            if self.inputs.get(sock_name) is None:
                sock = self.inputs.new(kind, sock_name)
                if default is not None:
                    try:
                        sock.default_value = default
                    except (TypeError, ValueError):
                        pass
        for want, sock_name in enumerate(names):
            have = [i for i, sk in enumerate(self.inputs)
                    if sk.name == sock_name]
            if have and have[0] != want and want < len(self.inputs):
                try:
                    self.inputs.move(have[0], want)
                except Exception:                               # noqa: BLE001
                    pass
        if shows:
            _apply_shows(self)

    cls = type(f'HALCYON_{name}Node', (Node, HalcyonNodeBase), {
        '__doc__': desc,
        'bl_idname': f'HALCYON_{name}Node',
        'bl_label': label,
        'bl_icon': icon,
        'bl_width_default': 160,
        '__annotations__': ann,
        'init': init,
        'draw_buttons': draw_buttons,
        'ensure_sockets': ensure_sockets,
    })
    return cls


NODES = tuple(_make(*spec) for spec in SPECS)

# what the exporter must copy across for each of them
NODE_PROPS = {f'HALCYON_{spec[0]}Node': tuple(spec[5].keys()) for spec in SPECS}


# R242: the Textures menu is categorised. Every pattern node belongs to
# exactly one category below (a census test holds the union to NODES,
# less the Matcap Coordinates node that now lives in the Vector family
# and the BI Texture that moved to the Blender Internal family).
_BY_NAME = {cls.bl_idname: cls for cls in NODES}


def _cat(*names):
    return tuple(_BY_NAME[f'HALCYON_{n}Node'] for n in names)


TEXTURE_FAMILIES = (
    ("Noise & Fractal", 'MOD_NOISE',
     _cat('Noise', 'Bozo', 'Plasma', 'Water', 'Static', 'Wrinkles',
          'Bumps', 'Dents', 'Crackle')),
    ("Stone, Wood & Organic", 'TEXTURE',
     _cat('Marble', 'Agate', 'Granite', 'Wood', 'Onion', 'Leopard',
          'Cells', 'Spiral')),
    ("Tiles & Fabric", 'MESH_GRID',
     _cat('Brick', 'Tiles', 'Weave', 'Scratches')),
    ("Sky, Water & Effects", 'WORLD',
     _cat('Starfield', 'Caustics', 'Ripples', 'FurTufts', 'Gradient')),
    ("2D Media", 'FILE_TEXT',
     _cat('Hatching', 'Scribble', 'Stipple', 'Charcoal', 'PaintStrokes',
          'Wash', 'Paper')),
)

#: pattern nodes that draw in another family's submenu instead
MOVED_OUT = ('HALCYON_MatcapUVNode',)


def _texture_menu(title, members):
    def draw(self, context):
        layout = self.layout
        for cls in members:
            op = layout.operator('node.add_node', text=cls.bl_label,
                                 icon=getattr(cls, 'bl_icon', 'NONE'))
            op.type = cls.bl_idname
            op.use_transform = True
    ident = 'NODE_MT_halcyon_tex_' + ''.join(
        ch for ch in title.lower().replace(' ', '_') if ch.isalnum() or ch == '_')
    return type(ident, (bpy.types.Menu,), {
        'bl_idname': ident, 'bl_label': title, 'draw': draw})


TEXTURE_SUBMENUS = tuple(_texture_menu(t, m) for t, _i, m in TEXTURE_FAMILIES)


class NODE_MT_halcyon_textures(bpy.types.Menu):
    bl_idname = 'NODE_MT_halcyon_textures'
    bl_label = "Halcyon Textures"

    def draw(self, context):
        layout = self.layout
        for (title, icon, _m), sub in zip(TEXTURE_FAMILIES, TEXTURE_SUBMENUS):
            layout.menu(sub.bl_idname, icon=icon)


def register():
    for cls in NODES:
        bpy.utils.register_class(cls)
    from . import bitex_node
    bitex_node.register()
    for sub in TEXTURE_SUBMENUS:
        bpy.utils.register_class(sub)
    bpy.utils.register_class(NODE_MT_halcyon_textures)


def unregister():
    for cls in (NODE_MT_halcyon_textures,) + tuple(TEXTURE_SUBMENUS):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass
    from . import bitex_node
    bitex_node.unregister()
    for cls in reversed(NODES):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass
