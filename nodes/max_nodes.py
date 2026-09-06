"""3ds Max's materials, maps, utilities and coordinates as Halcyon nodes
(R242, the Max study; R243, its second reading) -- the 3DS Max submenu
of the Halcyon Add menu.

Every node here takes the CONTROLS the 2012 reference documents for its
Max original, under Max's names, at Max's defaults. The maps ride
core/maxmaps.py (CPU) and gpu/procedural.py's MAX_GLSL (GPU), the
materials compose closures the way Max's compound materials composed
sub-materials, the utilities are Max's map-tree helpers, and the
coordinates node is the Coordinates rollout every 2D map carried.
Generated through pattern_nodes._make, so a socket cannot drift from the
evaluator that reads it.
"""

import bpy
from bpy.props import BoolProperty, EnumProperty, IntProperty
from bpy.types import Node

from .pattern_nodes import _make

F = 'NodeSocketFloat'
C = 'NodeSocketColor'
V = 'NodeSocketVector'
S = 'NodeSocketShader'

WHITE = (1.0, 1.0, 1.0, 1.0)
BLACK = (0.0, 0.0, 0.0, 1.0)

NOISE_KIND = (
    ('REGULAR', "Regular", "Plain noise: one octave, the smooth rolling "
     "field -- Max's default type"),
    ('FRACTAL', "Fractal", "Octaves of the noise stacked at halving size, "
     "Levels deep -- clouds, dust, the mottle on anything"),
    ('TURBULENCE', "Turbulence", "Fractal built from the noise's absolute "
     "value, so every octave folds into cusps -- marble veins, fire, the "
     "billow in smoke"),
)

CELL_KIND = (
    ('CIRCULAR', "Circular", "Round cells: the field is the distance to "
     "the nearest point, so every cell is a bubble"),
    ('CHIPS', "Chips", "Square-ish cells: the same points measured "
     "corner-wise, so the cells are plates and flakes -- mosaic, scales, "
     "cracked paint"),
)

TILE_PATTERN = (
    ('STACK', "Stack Bond", "Every course aligned: tiles in a plain grid "
     "-- bathroom walls, floor tiles"),
    ('RUNNING', "Running Bond", "Each course offset by half a tile -- the "
     "common brick wall"),
    ('ENGLISH', "English Bond", "Courses of stretchers alternate with "
     "courses of headers (half-width bricks)"),
    ('FLEMISH', "Flemish Bond", "Stretcher and header alternate along "
     "every course, offset course to course"),
)

FALLOFF_TYPE = (
    ('PERP_PARALLEL', "Perpendicular / Parallel", "Front where the "
     "surface faces the direction, Side where it runs parallel -- the "
     "silhouette mask, Max's default"),
    ('TOWARDS_AWAY', "Towards / Away", "Front where the surface faces "
     "the direction, Side where it faces away -- a hemisphere split, no "
     "silhouette band"),
    ('FRESNEL', "Fresnel", "The dielectric reflectance at the IOR (Max's "
     "full Fresnel equation, not an approximation): Side grows at grazing "
     "angles the way glass and water go mirror-like"),
    ('SHADOW_LIGHT', "Shadow / Light", "How much lamp light lands on the "
     "surface: Front in the dark, Side in full light -- Lambert against "
     "the lamp list, the Light Meter's road, so it shades on the CPU"),
    ('DISTANCE', "Distance Blend", "Side nearer than Near Distance, Front "
     "past Far Distance, blended between (Max's order) -- the depth-fade "
     "mask; Extrapolate lets the blend run past both distances"),
)

GRADIENT_TYPE = (
    ('LINEAR', "Linear", "The ramp runs along V, bottom to top -- Max's "
     "default"),
    ('RADIAL', "Radial", "The ramp runs outward from the tile's centre"),
    ('BOX', "Box", "The ramp runs outward in a square from the centre"),
    ('DIAGONAL', "Diagonal", "The ramp runs along the tile's diagonal"),
    ('FOUR_CORNER', "Four Corner", "Max's asymmetric corner gradient, "
     "v squared over u"),
    ('PONG', "Pong", "The ramp bounces: u over v, folded back past 1"),
    ('SPIRAL', "Spiral", "The ramp follows the angle about the origin"),
    ('SWEEP', "Sweep", "The ramp sweeps once around the tile's centre"),
    ('TARTAN', "Tartan", "The ramp is the nearer of the two axes' "
     "distances from the centre, inverted -- a plaid when tiled"),
    ('NORMAL', "Normal", "The ramp reads the view angle: 0 facing the "
     "camera, 1 at the silhouette"),
    ('MAPPED', "Mapped", "The ramp reads the Mapped input -- link any "
     "map's Fac to recolour it through the ramp"),
)

FALLOFF_DIR = (
    ('VIEW', "Viewing Direction", "Measured against the line to the "
     "camera -- rims, silhouettes, view-dependent fades"),
    ('WORLD_X', "World X", "Measured against the world's X axis"),
    ('WORLD_Y', "World Y", "Measured against the world's Y axis"),
    ('WORLD_Z', "World Z", "Measured against the world's up: Front on top "
     "faces, Side on the underside"),
)

COORD_SOURCE = (
    ('MAP_CHANNEL', "Explicit Map Channel", "The mesh's own UV layer -- "
     "Max's default, the unwrapped coordinates"),
    ('OBJECT_XYZ', "Planar from Object XYZ", "The object's normalised "
     "bounds (Generated): the map sticks to the object as it moves"),
    ('WORLD_XYZ', "Planar from World XYZ", "World position: the map stays "
     "put in the world while the object moves through it"),
    ('SCREEN', "Screen", "The frame itself, 0..1 across the picture -- "
     "backdrops and screen-locked overlays"),
)

COMPOSITE_MODE = (
    ('ADD', "Additive", "The layer's colour is added over what is beneath, "
     "scaled by its Amount -- Max's A button"),
    ('MIX', "Mix", "The layer replaces what is beneath by its Amount, as a "
     "Blend with no mask -- Max's M button. (Max's Subtractive has no "
     "lobe form and is not built)"),
)

ALPHA_FROM = (
    ('MAP1', "Map 1", "The alpha comes from the first colour's alpha"),
    ('MAP2', "Map 2", "The alpha comes from the second colour's alpha"),
    ('MULTIPLY', "Multiply", "The alphas multiply, like the colours"),
)

# R243: the remaining maps' menus
GRADIENT_SHAPE = (
    ('LINEAR', "Linear", "The ramp runs along V: Color 3 at the bottom, "
     "Color 1 at the top -- Max's default"),
    ('RADIAL', "Radial", "The ramp runs outward from the tile's centre: "
     "Color 3 at the centre, Color 1 at the corners' reach"),
)

CC_CHANNELS = (
    ('NORMAL', "Normal", "Each channel reads itself: red from red, green "
     "from green, blue from blue, alpha from alpha"),
    ('MONO', "Monochrome", "Red, green and blue all read the colour's "
     "mean intensity -- a greyscale, Max's Monochrome"),
    ('INVERT', "Invert", "Red, green and blue read their inverses (1 "
     "minus the channel); the alpha stays -- a negative"),
    ('CUSTOM', "Custom", "Each output channel reads the source picked "
     "below: any channel, its inverse, the mono intensity, one or zero"),
)

REWIRE_ITEMS = (
    ('RED', "Red", "Reads the source's red channel"),
    ('GREEN', "Green", "Reads the source's green channel"),
    ('BLUE', "Blue", "Reads the source's blue channel"),
    ('ALPHA', "Alpha", "Reads the source's alpha channel"),
    ('RED_INV', "Red Inverted", "Reads 1 minus the source's red"),
    ('GREEN_INV', "Green Inverted", "Reads 1 minus the source's green"),
    ('BLUE_INV', "Blue Inverted", "Reads 1 minus the source's blue"),
    ('ALPHA_INV', "Alpha Inverted", "Reads 1 minus the source's alpha"),
    ('MONO', "Monochrome", "Reads the mean of red, green and blue"),
    ('ONE', "One", "Reads a constant 1 -- full"),
    ('ZERO', "Zero", "Reads a constant 0 -- empty"),
)

LIGHTNESS_MODE = (
    ('STANDARD', "Standard", "Brightness and Contrast: the colour "
     "stretched about mid-grey by the contrast and shifted by the "
     "brightness, both percentages -- Max's Standard"),
    ('ADVANCED', "Advanced", "Gain, Gamma, Pivot and Lift: the colour "
     "times Gain percent, raised to 1 / Gamma about the Pivot, plus "
     "Lift -- Max's Advanced, applied to RGB together"),
)

BLEND_MODE = (
    ('NORMAL', "Normal", "The layer covers what is beneath by its alpha"),
    ('AVERAGE', "Average", "The mean of the layer and what is beneath"),
    ('ADDITION', "Addition", "The layer added to what is beneath, "
     "unclamped -- it can run past white (Linear Dodge is the clamped "
     "form)"),
    ('SUBTRACT', "Subtract", "The layer taken from what is beneath, "
     "unclamped (Linear Burn is the clamped form)"),
    ('DARKEN', "Darken", "The darker of the two, channel by channel"),
    ('MULTIPLY', "Multiply", "The two multiplied -- shadow, dirt, stain"),
    ('COLOR_BURN', "Color Burn", "What is beneath darkened by the "
     "layer's darkness: 1 minus (1 minus base) over the layer"),
    ('LINEAR_BURN', "Linear Burn", "The sum minus one, held at black"),
    ('LIGHTEN', "Lighten", "The lighter of the two, channel by channel"),
    ('SCREEN', "Screen", "Both projected together: one minus the product "
     "of their inverses -- glow, light leaks"),
    ('COLOR_DODGE', "Color Dodge", "What is beneath brightened by the "
     "layer's brightness: the base over (1 minus the layer)"),
    ('LINEAR_DODGE', "Linear Dodge", "The sum, held at white"),
    ('SPOTLIGHT', "Spotlight", "Twice the product, held at white -- "
     "Max's Spotlight"),
    ('SPOTLIGHT_BLEND', "Spotlight Blend", "The product plus the base, "
     "held at white -- Max's Spotlight Blend"),
    ('OVERLAY', "Overlay", "Multiply where the base is dark, Screen "
     "where it is light -- contrast that keeps the base's tone"),
    ('SOFT_LIGHT', "Soft Light", "A gentle dodge or burn by the layer, "
     "about its mid-grey"),
    ('HARD_LIGHT', "Hard Light", "Multiply where the layer is dark, "
     "Screen where it is light -- Overlay with the roles swapped"),
    ('PIN_LIGHT', "Pin Light", "The layer replaces the base only where "
     "it is the more extreme -- lighter above mid-grey, darker below"),
    ('HARD_MIX', "Hard Mix", "Every channel snaps to black or white by "
     "whether the sum passes one -- posterised"),
    ('DIFFERENCE', "Difference", "The absolute difference -- black "
     "where they match, inverting where they oppose"),
    ('EXCLUSION', "Exclusion", "Difference with less contrast: the sum "
     "minus twice the product"),
    ('HUE', "Hue", "The layer's hue on the base's saturation and "
     "lightness"),
    ('SATURATION', "Saturation", "The layer's saturation on the base's "
     "hue and lightness"),
    ('COLOR', "Color", "The layer's hue and saturation on the base's "
     "lightness -- a tint that keeps the shading"),
    ('VALUE', "Value", "The layer's lightness on the base's hue and "
     "saturation"),
)

VCOL_CHANNEL = (
    ('VERTEX_COLOR', "Vertex Color", "The painted colour itself -- Max's "
     "channel 0, the vertex colour"),
    ('VERTEX_ALPHA', "Vertex Alpha", "The colour layer's alpha as a grey "
     "-- Max's channel -2, the vertex alpha"),
)

SUB_CHANNEL = (
    ('ALL', "All", "The whole colour"),
    ('RED', "Red", "The red channel alone, as a grey"),
    ('GREEN', "Green", "The green channel alone, as a grey"),
    ('BLUE', "Blue", "The blue channel alone, as a grey"),
)

XYZ_SOURCE = (
    ('OBJECT_XYZ', "Object XYZ", "The object's own local coordinates: "
     "the map sticks to the object as it moves -- Max's default"),
    ('WORLD_XYZ', "World XYZ", "World position: the map stays put in "
     "the world while the object moves through it"),
    ('MAP_CHANNEL', "Explicit Map Channel", "The mesh's UV layer as "
     "X and Y (W is 0) -- Max's explicit map channel"),
    ('VERTEX_COLOR', "Vertex Color Channel", "The painted colour's red, "
     "green and blue as X, Y, Z -- Max's vertex colour channel"),
    ('GENERATED', "Generated (Halcyon)", "Halcyon's normalised bounds, "
     "0..1 across the object -- what every 3D map here reads with its "
     "Vector unlinked; not a Max source, kept so this rollout can add "
     "its Offset, Tiling and Angle to that default look"),
)

_MAX_PROP_TIPS = {
    'kind': "The noise type: Regular (one octave), Fractal (Levels of "
            "octaves), or Turbulence (the folded, cusped fractal)",
    'chips': "Round cells (Circular) or corner-measured plates (Chips)",
    'fractal': "Stacks Iterations of the cell field at halving size, so "
               "cells carry smaller cells -- rock, sponge, cracked mud",
    'iterations': "How many octaves the fractal stacks; each costs one "
                  "more pass of the pattern",
    'pattern': "The bond: how the courses of tiles align with each other",
    'sets': "How many wave sets ripple across the field, each from its "
            "own centre at its own wavelength -- more sets, richer "
            "interference",
    'dist3d': "Places the wave centres on a sphere about the origin (3D) "
              "rather than a flat circle (2D)",
    'falloff_type': "What the mask measures: facing versus the "
                    "direction, a hemisphere split, the Fresnel curve, "
                    "or distance from the camera",
    'direction': "The direction the facing types measure against",
    'source': "Which coordinates the node starts from before its "
              "offset, tiling, mirror and angle are applied",
    'mirror_u': "Every second repeat flips in U, so tiles meet their "
                "reflections seamlessly (Max's Mirror)",
    'mirror_v': "Every second repeat flips in V, so tiles meet their "
                "reflections seamlessly (Max's Mirror)",
    'use_curve': "Passes the mask through the mixing curve: below Lower "
                 "it is all Material 1, above Upper all Material 2, "
                 "smoothed between -- Max's Use Curve",
    'mode1': "How layer 1 lands on the base: added by its Amount, or "
             "mixed in by it",
    'mode2': "How layer 2 lands on the result so far: added or mixed",
    'mode3': "How layer 3 lands on the result so far: added or mixed",
    'mode4': "How layer 4 lands on the result so far: added or mixed",
    'invert': "Inverts the colour (1 minus each channel) -- Max Output's "
              "Invert",
    'clamp': "Holds every channel inside 0..1 after Level and Offset -- "
             "Max Output's Clamp",
    'alpha_from': "Which input's alpha the result carries: the first "
                  "map's, the second's, or their product",
    'invert_mask': "Uses 1 minus the mask, so the map shows where the "
                   "mask is dark",
    'blend': "Blends the deepest water colour into the first land colour "
             "at the shoreline instead of cutting between them -- Max's "
             "Blend Water/Land",
    'extrapolate': "Lets Distance Blend run past Near and Far instead of "
                   "holding its ends, so the colours extrapolate beyond "
                   "Front and Side -- Max's Extrapolate",
    'gradient_type': "The shape the ramp follows across the tile: Max's "
                     "eleven gradient types",
    'alpha_from_rgb': "Replaces the alpha with the colour's mean "
                      "intensity -- Max Output's Alpha From RGB Intensity",
    # R243: the remaining maps
    'shape': "Linear runs the ramp along V, bottom to top; Radial runs it "
             "outward from the tile's centre",
    'channels': "Max's Channels preset: Normal, Monochrome, Invert, or "
                "Custom with a source picked per output channel below",
    'rewire_r': "Custom channels: what the red output reads",
    'rewire_g': "Custom channels: what the green output reads",
    'rewire_b': "Custom channels: what the blue output reads",
    'rewire_a': "Custom channels: what the alpha output reads",
    'lightness': "Which lightness controls apply: Standard's Brightness "
                 "and Contrast, or Advanced's Gain, Gamma, Pivot and Lift",
    'blend2': "How Layer 2 lands on Layer 1: one of Max's twenty-five "
              "blend modes",
    'blend3': "How Layer 3 lands on the result of the layers beneath it",
    'blend4': "How Layer 4 lands on the result of the layers beneath it",
    'blend5': "How Layer 5 lands on the result of the layers beneath it",
    'channel': "Max's Map Channel: the vertex colour (0) or the vertex "
               "alpha (-2). Max's vertex illumination has no Blender "
               "counterpart",
    'sub_channel': "Max's Sub Channel: the whole colour, or one channel "
                   "as a grey",
    'layer_name': "Max's Channel Name: the colour attribute to read; "
                  "empty reads the active layer (the only one the GPU "
                  "road carries -- a named layer shades on the CPU)",
    'xyz_source': "Which 3D coordinates the rollout starts from before "
                  "its Offset, Angle and Tiling",
}

#: Max's Size controls come across at Max's own defaults divided by 100:
#: Max sizes its maps in scene units on a hundred-unit object, Halcyon's
#: generated coordinates span one unit, so the default look is the same
SIZE_TIP = ("Max's Size: the map's cell size in the coordinate's units "
            "(coordinate / Size). Max's default assumes a 100-unit object; "
            "Halcyon's generated coordinates span one unit, so the default "
            "here is Max's divided by 100 -- the same look on a unit box")

# name, label, icon, description, sockets, props, outputs
TEXTURE_SPECS = [
    ('MaxNoise', "Noise (Max)", 'MOD_NOISE',
     "3ds Max's Noise map on Max's own algorithm: the 4D lattice noise "
     "(Phase is its fourth axis) as Regular, Fractal or Turbulence, "
     "Levels deep (a fraction of a level counts), cut between the Low "
     "and High thresholds. Size is Max's Size",
     [(V, 'Vector', None), (F, 'Size', 0.25), (F, 'Levels', 3.0),
      (F, 'Low', 0.0), (F, 'High', 1.0), (F, 'Phase', 0.0),
      (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {'kind': ('enum', 'REGULAR', NOISE_KIND, "Noise Type")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxCellular', "Cellular (Max)", 'MESH_GRID',
     "3ds Max's Cellular map on Max's own algorithm: points scattered "
     "by the Poisson counts of every lattice cell, Circular reading the "
     "nearest point's squared distance over Spread, Chips the gap "
     "between the two nearest; the field cut by Low / Mid / High into "
     "the Cell Color and the two Division Colors, Variation scaling "
     "each cell's colour by its own random; Fractal stacks Iterations "
     "of it at the lacunarity 2 minus Roughness. Mosaic, scales, "
     "sponge, rock. Max's costliest map, here as there",
     [(V, 'Vector', None), (F, 'Size', 0.05), (F, 'Spread', 0.5),
      (F, 'Roughness', 0.0), (F, 'Variation', 0.0), (F, 'Low', 0.0),
      (F, 'Mid', 0.5), (F, 'High', 1.0), (C, 'Cell Color', WHITE),
      (C, 'Division Color 1', (0.5, 0.5, 0.5, 1.0)),
      (C, 'Division Color 2', BLACK)],
     {'chips': ('enum', 'CIRCULAR', CELL_KIND, "Cell Characteristics"),
      'fractal': ('bool', False, "Fractal"),
      'iterations': ('int', 3, 1, 8, "Iterations")},
     [(C, 'Color'), (F, 'Fac'), (F, 'Cell ID')]),

    ('MaxSmoke', "Smoke (Max)", 'MOD_FLUIDSIM',
     "3ds Max's Smoke map on Max's own algorithm: the folded noise "
     "summed over Iterations at doubling size, each octave drifting "
     "with its own velocity as Phase advances (the drift growing 2.4x "
     "per octave), raised to Exponent (higher hollows the wisps out). "
     "Animate Phase for smoke that moves",
     [(V, 'Vector', None), (F, 'Size', 0.4), (F, 'Phase', 0.0),
      (F, 'Exponent', 1.5), (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {'iterations': ('int', 5, 1, 20, "Iterations")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxSpeckle', "Speckle (Max)", 'MOD_DISPLACE',
     "3ds Max's Speckle map on Max's own algorithm: six octaves of the "
     "noise summed and capped, so Color 1 shows only in the specks "
     "where the sum falls short -- granite chips, dust, the flecks in "
     "a paint. Size sets their scale",
     [(V, 'Vector', None), (F, 'Size', 0.6), (C, 'Color 1', BLACK),
      (C, 'Color 2', WHITE)],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('MaxSplat', "Splat (Max)", 'MOD_EXPLODE',
     "3ds Max's Splat map on Max's own algorithm: one minus the product "
     "over Iterations of the noise stepped about Threshold, so the "
     "drops shrink octave by octave. Spatter, camouflage, the Jackson "
     "Pollock floor",
     [(V, 'Vector', None), (F, 'Size', 0.4), (F, 'Threshold', 0.2),
      (C, 'Color 1', WHITE), (C, 'Color 2', (0.1, 0.1, 0.3, 1.0))],
     {'iterations': ('int', 4, 1, 5, "Iterations")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxStucco', "Stucco (Max)", 'MOD_SMOOTH',
     "3ds Max's Stucco map on Max's own algorithm: the noise past "
     "Threshold over Thickness, shaped by Max's knee curve -- meant for "
     "the bump input, where it is plaster's relief; as a colour it is "
     "two-tone mottle",
     [(V, 'Vector', None), (F, 'Size', 0.2), (F, 'Thickness', 0.15),
      (F, 'Threshold', 0.57), (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('MaxMarble', "Marble (Max)", 'MOD_OCEAN',
     "3ds Max's Marble map on Max's own algorithm: veins on a "
     "seventeen-band cycle along the object's X, the bands displaced "
     "by noise, a bright band inside each vein and a dark heart, Vein "
     "Width the cycle's rate, Size the noise's. Color 1 is the vein, "
     "Color 2 the stone",
     [(V, 'Vector', None), (F, 'Size', 0.7), (F, 'Vein Width', 0.025),
      (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('MaxPerlinMarble', "Perlin Marble (Max)", 'MOD_OCEAN',
     "3ds Max's Perlin Marble map on Max's own algorithm: Levels of "
     "turbulence drive a sine along X, and the value runs Perlin's "
     "thirteen-knot colour spline between Color 1 and Color 2 and their "
     "Saturation-scaled tones (Max's Saturation darkens below 100). "
     "The classic marble, Ken Perlin's own",
     [(V, 'Vector', None), (F, 'Size', 0.5), (F, 'Levels', 8.0),
      (C, 'Color 1', (0.94, 0.94, 0.94, 1.0)), (F, 'Saturation 1', 85.0),
      (C, 'Color 2', (0.18, 0.10, 0.06, 1.0)), (F, 'Saturation 2', 70.0)],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('MaxWood', "Wood (Max)", 'MOD_SCREW',
     "3ds Max's Wood map on Max's own algorithm: rings about the "
     "object's X axis, every coordinate jittered by Radial Noise, the "
     "ring radius jittered by Axial Noise along the grain, Grain "
     "Thickness the ring spacing (Max's Size). Color 1 the early wood, "
     "Color 2 the late",
     [(V, 'Vector', None), (F, 'Grain Thickness', 0.07),
      (F, 'Radial Noise', 1.0), (F, 'Axial Noise', 1.0),
      (C, 'Color 1', (0.66, 0.46, 0.27, 1.0)),
      (C, 'Color 2', (0.32, 0.20, 0.10, 1.0))],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('MaxDent', "Dent (Max)", 'MOD_DISPLACE',
     "3ds Max's Dent map on Max's own algorithm: the folded value noise "
     "summed over Iterations, cubed, times Strength -- pits and dings "
     "for a bump, mottle for a colour",
     [(V, 'Vector', None), (F, 'Size', 2.0), (F, 'Strength', 20.0),
      (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {'iterations': ('int', 2, 1, 10, "Iterations")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxPlanet', "Planet (Max)", 'WORLD',
     "3ds Max's Planet map on Max's own algorithm: the value noise plus "
     "a fifth of it at the Island Factor's frequency, three water "
     "colours by depth below Ocean % and five land colours by height "
     "above it, Blend Water/Land softening the shoreline. The 1990s "
     "globe, the alien world in one node",
     [(V, 'Vector', None), (F, 'Continent Size', 0.4),
      (F, 'Island Factor', 0.5), (F, 'Ocean %', 60.0),
      (C, 'Water 1', (0.12, 0.30, 0.55, 1.0)),
      (C, 'Water 2', (0.06, 0.18, 0.42, 1.0)),
      (C, 'Water 3', (0.02, 0.08, 0.25, 1.0)),
      (C, 'Land 1', (0.75, 0.70, 0.45, 1.0)),
      (C, 'Land 2', (0.30, 0.50, 0.20, 1.0)),
      (C, 'Land 3', (0.20, 0.38, 0.14, 1.0)),
      (C, 'Land 4', (0.45, 0.40, 0.32, 1.0)),
      (C, 'Land 5', (0.95, 0.95, 0.97, 1.0))],
     {'blend': ('bool', False, "Blend Water/Land")},
     [(C, 'Color'), (F, 'Elevation')]),

    ('MaxWaves', "Waves (Max)", 'MOD_WAVE',
     "3ds Max's Waves map on Max's own algorithm: Num Wave Sets of "
     "concentric ripples from centres the C runtime's rand() places at "
     "Wave Radius from the Random Seed (on a sphere for 3D, a circle "
     "for 2D), wavelengths between Wave Len Min and Max, summed at "
     "Amplitude, Phase rolling them. Water, and the interference "
     "patterns of the era. Distances are Max's divided by 100",
     [(V, 'Vector', None), (F, 'Wave Radius', 10.0),
      (F, 'Wave Len Min', 0.5), (F, 'Wave Len Max', 0.5),
      (F, 'Amplitude', 1.0), (F, 'Phase', 0.0),
      (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {'sets': ('int', 3, 1, 50, "Num Wave Sets"),
      'dist3d': ('bool', True, "3D Distribution"),
      'seed': ('int', 30159, 0, 65535, "Random Seed")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxSwirl', "Swirl (Max)", 'FORCE_VORTEX',
     "3ds Max's Swirl map on Max's own algorithm: the UV point spun "
     "about Center X / Y by Twist turns per squared radius, then "
     "Constant Detail octaves of the noise at gain Color Contrast; the "
     "Swirl Color over the Base Color by Swirl Amount times Swirl "
     "Intensity times that sum, unclamped, so the arms overshoot into "
     "pure colour as Max's do",
     [(V, 'Vector', None), (F, 'Tiling', 1.0), (F, 'Center X', -0.5),
      (F, 'Center Y', -0.5), (F, 'Twist', 1.0), (F, 'Swirl Intensity', 2.0),
      (F, 'Swirl Amount', 1.0), (F, 'Constant Detail', 4.0),
      (F, 'Color Contrast', 0.4),
      (C, 'Base Color', (0.2, 0.2, 0.6, 1.0)),
      (C, 'Swirl Color', (0.9, 0.9, 0.95, 1.0))],
     {'seed': ('int', 0, 0, 9999, "Random Seed")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxChecker', "Checker (Max)", 'MESH_GRID',
     "3ds Max's Checker map on Max's own algorithm: Color 1 where "
     "exactly one of u and v is past its half, Color 2 elsewhere (two "
     "checks per tile, as Max), Soften blurring the edges by Max's "
     "integrated square wave. Tiling repeats it",
     [(V, 'Vector', None), (F, 'Tiling', 1.0), (F, 'Soften', 0.0),
      (C, 'Color 1', BLACK), (C, 'Color 2', WHITE)],
     {}, [(C, 'Color'), (F, 'Fac')]),

    ('MaxTiles', "Tiles (Max)", 'MESH_GRID',
     "3ds Max's Tiles map on Max's own mortar geometry: a bond pattern "
     "(Stack, Running, English, Flemish) with Horizontal / Vertical "
     "Count, the grout as a percentage Gap, every second course "
     "shifted by Line Shift plus a Random Shift, a percentage of Holes, "
     "Fade Variance and Color Variance scaling each tile by its own "
     "random. Tile ID drives per-tile variation downstream",
     [(V, 'Vector', None), (F, 'Tiling', 1.0), (F, 'Horizontal Count', 4.0),
      (F, 'Vertical Count', 4.0), (F, 'Horizontal Gap', 0.5),
      (F, 'Vertical Gap', 0.5), (F, 'Line Shift', 0.5),
      (F, 'Random Shift', 0.0), (F, 'Holes', 0.0),
      (F, 'Fade Variance', 0.05), (F, 'Color Variance', 0.0),
      (C, 'Tile Color', (0.65, 0.35, 0.28, 1.0)),
      (C, 'Grout Color', (0.75, 0.73, 0.68, 1.0))],
     {'pattern': ('enum', 'RUNNING', TILE_PATTERN, "Preset Type"),
      'seed': ('int', 33862, 0, 65535, "Random Seed")},
     [(C, 'Color'), (F, 'Fac'), (F, 'Tile ID')]),

    ('MaxGradientRamp', "Gradient Ramp (Max)", 'COLOR',
     "3ds Max's Gradient Ramp map on Max's own gradient shapes: Linear, "
     "Radial, Box, Diagonal, Four Corner, Pong, Spiral, Sweep, Tartan, "
     "Normal (the view angle) or Mapped (a linked map's intensity), the "
     "coordinate running a three-flag ramp -- Color 1 at 0, Color 2 at "
     "Color 2 Position, Color 3 at 1. (Max's editor takes any number "
     "of flags; three are built)",
     [(V, 'Vector', None), (F, 'Tiling', 1.0), (F, 'Color 2 Position', 0.5),
      (F, 'Mapped', 0.5), (C, 'Color 1', BLACK),
      (C, 'Color 2', (0.5, 0.5, 0.5, 1.0)), (C, 'Color 3', WHITE)],
     {'gradient_type': ('enum', 'LINEAR', GRADIENT_TYPE, "Gradient Type")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxGradient', "Gradient (Max)", 'COLOR',
     "3ds Max's Gradient map on Max's own algorithm: three colours -- "
     "Color 3 at the bottom, Color 2 at Color 2 Position, Color 1 at the "
     "top -- Linear along V or Radial from the tile's centre; Noise "
     "Amount adds Max's noise at Noise Size (Regular, Fractal or "
     "Turbulence, Noise Levels deep, a fraction of a level counting, "
     "Noise Phase its third axis), cut between the Threshold Low and "
     "High with Smooth rounding the cut's corners. Tiling repeats it",
     [(V, 'Vector', None), (F, 'Tiling', 1.0), (C, 'Color 1', BLACK),
      (C, 'Color 2', (0.5, 0.5, 0.5, 1.0)), (C, 'Color 3', WHITE),
      (F, 'Color 2 Position', 0.5), (F, 'Noise Amount', 0.0),
      (F, 'Noise Size', 1.0), (F, 'Noise Phase', 0.0), (F, 'Noise Levels', 4.0),
      (F, 'Threshold Low', 0.0), (F, 'Threshold High', 1.0),
      (F, 'Threshold Smooth', 0.0)],
     {'shape': ('enum', 'LINEAR', GRADIENT_SHAPE, "Gradient Type"),
      'kind': ('enum', 'REGULAR', NOISE_KIND, "Noise Type")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxFalloff', "Falloff (Max)", 'MATSPHERE',
     "3ds Max's Falloff map exactly: Front where the surface faces the "
     "direction, Side where it does not -- Perpendicular / Parallel "
     "(the silhouette), Towards / Away (the hemisphere), Fresnel (the "
     "full dielectric equation at the IOR), Shadow / Light (how much "
     "lamp light lands here, the Light Meter's road -- shades on the "
     "CPU), or Distance Blend (Side nearer than Near Distance, Front "
     "past Far, blended between; Extrapolate lets it run past both). "
     "Rim masks, gradients that follow the view. Distances are Max's "
     "divided by 100",
     [(F, 'Near Distance', 0.0), (F, 'Far Distance', 1.0),
      (F, 'IOR', 1.6), (C, 'Front', BLACK), (C, 'Side', WHITE)],
     {'falloff_type': ('enum', 'PERP_PARALLEL', FALLOFF_TYPE, "Falloff Type"),
      'direction': ('enum', 'VIEW', FALLOFF_DIR, "Falloff Direction"),
      'extrapolate': ('bool', False, "Extrapolate")},
     [(C, 'Color'), (F, 'Fac')]),
]

UTILITY_SPECS = [
    ('MaxMix', "Mix (Max)", 'COLOR',
     "3ds Max's Mix map: Color 1 toward Color 2 by the Mix Amount (link "
     "a map to it); Use Curve passes the amount through the mixing "
     "curve's Lower / Upper transition zone -- a soft threshold on any "
     "mask",
     [(C, 'Color 1', BLACK), (C, 'Color 2', WHITE), (F, 'Mix Amount', 0.5),
      (F, 'Upper', 0.7), (F, 'Lower', 0.3)],
     {'use_curve': ('bool', False, "Use Curve")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxRGBTint', "RGB Tint (Max)", 'COLOR',
     "3ds Max's RGB Tint map: each channel of the input picks up its own "
     "tint colour -- red through R Tint, green through G Tint, blue "
     "through B Tint -- summed. Recolour a greyscale, swap channels, "
     "cross-process",
     [(C, 'Color', (0.8, 0.8, 0.8, 1.0)), (C, 'R Tint', (1.0, 0.0, 0.0, 1.0)),
      (C, 'G Tint', (0.0, 1.0, 0.0, 1.0)), (C, 'B Tint', (0.0, 0.0, 1.0, 1.0))],
     {}, [(C, 'Color')]),

    ('MaxOutput', "Output (Max)", 'OUTPUT',
     "3ds Max's Output map, in Max's order: RGB Level scales the colour, "
     "RGB Offset shifts it (by the alpha, as Max), Output Amount scales "
     "colour and alpha, Invert flips the colour, Clamp holds it in "
     "0..1, Alpha From RGB rewrites the alpha from the intensity -- the "
     "levels stage every Max map carried on its Output rollout",
     [(C, 'Color', (0.8, 0.8, 0.8, 1.0)), (F, 'RGB Level', 1.0),
      (F, 'RGB Offset', 0.0), (F, 'Output Amount', 1.0)],
     {'invert': ('bool', False, "Invert"),
      'clamp': ('bool', False, "Clamp"),
      'alpha_from_rgb': ('bool', False, "Alpha From RGB")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxMask', "Mask (Max)", 'IMAGE_ALPHA',
     "3ds Max's Mask map: the Map shows where the Mask is bright and "
     "fades to black where it is dark, its alpha scaled the same way; "
     "Invert Mask swaps that. The Alpha output is the mask itself, for "
     "opacity chains",
     [(C, 'Map', (0.8, 0.8, 0.8, 1.0)), (F, 'Mask', 1.0)],
     {'invert_mask': ('bool', False, "Invert Mask")},
     [(C, 'Color'), (F, 'Alpha')]),

    ('MaxRGBMultiply', "RGB Multiply (Max)", 'COLOR',
     "3ds Max's RGB Multiply map: the two colours multiplied channel by "
     "channel, the alpha taken from Map 1, Map 2 or their product -- "
     "the dirt-over-diffuse composite of the era",
     [(C, 'Color 1', (0.8, 0.8, 0.8, 1.0)), (C, 'Color 2', WHITE)],
     {'alpha_from': ('enum', 'MULTIPLY', ALPHA_FROM, "Alpha From")},
     [(C, 'Color'), (F, 'Alpha')]),

    ('MaxCompositeMap', "Composite Map (Max)", 'NODE_COMPOSITING',
     "3ds Max's Composite MAP on Max's own algorithm (the Composite "
     "material is under Materials): Layer 1 at the bottom and up to four "
     "layers over it in order, each with its Opacity percentage, a Mask "
     "(link any map's Fac) and one of Max's twenty-five blend modes, "
     "landed by Max's 'over' -- the blend where both are present, the "
     "layer alone where only it is, over their union alpha; a result "
     "short of full alpha is premultiplied, as Max's. A layer whose "
     "Color is unlinked is an empty slot and is skipped, as Max skips "
     "it (Layer 1 alone reads its swatch), and so is a layer at "
     "Opacity 0",
     [(C, 'Layer 1', (0.8, 0.8, 0.8, 1.0)), (F, 'Opacity 1', 100.0),
      (F, 'Mask 1', 1.0),
      (C, 'Layer 2', WHITE), (F, 'Opacity 2', 100.0), (F, 'Mask 2', 1.0),
      (C, 'Layer 3', WHITE), (F, 'Opacity 3', 100.0), (F, 'Mask 3', 1.0),
      (C, 'Layer 4', WHITE), (F, 'Opacity 4', 100.0), (F, 'Mask 4', 1.0),
      (C, 'Layer 5', WHITE), (F, 'Opacity 5', 100.0), (F, 'Mask 5', 1.0)],
     {'blend2': ('enum', 'NORMAL', BLEND_MODE, "Layer 2", True),
      'blend3': ('enum', 'NORMAL', BLEND_MODE, "Layer 3", True),
      'blend4': ('enum', 'NORMAL', BLEND_MODE, "Layer 4", True),
      'blend5': ('enum', 'NORMAL', BLEND_MODE, "Layer 5", True)},
     [(C, 'Color'), (F, 'Alpha')]),

    ('MaxColorCorrection', "Color Correction (Max)", 'COLOR',
     "3ds Max's Color Correction map on Max's own algorithm, in Max's "
     "order: the Channels rewired (Normal, Monochrome, Invert, or Custom "
     "with a source per channel), then in Max's HSL the Hue Shift in "
     "degrees, Saturation added as a percentage and the hue pulled "
     "toward the Hue Tint's by Strength, then the Lightness -- "
     "Standard's Brightness and Contrast about mid-grey, or Advanced's "
     "Gain (percent), Gamma about the Pivot, and Lift. (Max's separate "
     "per-channel Advanced controls are not built: Advanced applies to "
     "R, G and B together)",
     [(C, 'Color', (0.8, 0.8, 0.8, 1.0)), (F, 'Hue Shift', 0.0),
      (F, 'Saturation', 0.0), (C, 'Hue Tint', WHITE), (F, 'Strength', 0.0),
      (F, 'Brightness', 0.0), (F, 'Contrast', 0.0), (F, 'Gain', 100.0),
      (F, 'Gamma', 1.0), (F, 'Pivot', 1.0), (F, 'Lift', 0.0)],
     {'channels': ('enum', 'NORMAL', CC_CHANNELS, "Channels"),
      'rewire_r': ('enum', 'RED', REWIRE_ITEMS, "Red", True),
      'rewire_g': ('enum', 'GREEN', REWIRE_ITEMS, "Green", True),
      'rewire_b': ('enum', 'BLUE', REWIRE_ITEMS, "Blue", True),
      'rewire_a': ('enum', 'ALPHA', REWIRE_ITEMS, "Alpha", True),
      'lightness': ('enum', 'STANDARD', LIGHTNESS_MODE, "Lightness")},
     [(C, 'Color'), (F, 'Fac')]),

    ('MaxVertexColor', "Vertex Color (Max)", 'GROUP_VCOL',
     "3ds Max's Vertex Color map: the mesh's painted colour -- the "
     "active colour attribute, or the layer named in Channel Name -- as "
     "Max's Vertex Color channel or its Vertex Alpha, the Sub Channel "
     "taking All of it or one of Red, Green, Blue as a grey. Fac is the "
     "mean intensity. (Max's Vertex Illumination channel has no Blender "
     "counterpart and is not built)",
     [],
     {'channel': ('enum', 'VERTEX_COLOR', VCOL_CHANNEL, "Map Channel"),
      'sub_channel': ('enum', 'ALL', SUB_CHANNEL, "Sub Channel"),
      'layer_name': ('string', '', "Channel Name")},
     [(C, 'Color'), (F, 'Fac')]),
]

VECTOR_SPECS = [
    ('MaxCoords', "Coordinates (Max)", 'MOD_UVPROJECT',
     "3ds Max's Coordinates rollout: the source (map channel, object "
     "XYZ, world XYZ or screen), then Offset, Tiling, Mirror and the "
     "W Angle, in Max's order -- feed the Vector into any texture. "
     "Tiling below 1 with Mirror off holds the edge, as Tile-off did",
     [(V, 'Vector', None), (F, 'Offset U', 0.0), (F, 'Offset V', 0.0),
      (F, 'Tiling U', 1.0), (F, 'Tiling V', 1.0), (F, 'Angle W', 0.0)],
     {'source': ('enum', 'MAP_CHANNEL', COORD_SOURCE, "Source"),
      'mirror_u': ('bool', False, "Mirror U"),
      'mirror_v': ('bool', False, "Mirror V")},
     [(V, 'Vector')]),

    ('MaxXYZCoords', "XYZ Coordinates (Max)", 'ORIENTATION_GLOBAL',
     "3ds Max's 3D map Coordinates rollout on Max's own transform: the "
     "source (Object XYZ, World XYZ, the explicit map channel, the "
     "vertex colour channel, or Halcyon's generated bounds -- what the "
     "3D maps read unlinked) moved by Offset, turned by the Angles in "
     "degrees (X, then Y, then Z) and scaled by Tiling, the scale "
     "leaving the turned offset alone as Max's matrix does. Feed the "
     "Vector into any 3D map. Object XYZ is the object's own frame, on "
     "both devices",
     [(V, 'Vector', None), (F, 'Offset X', 0.0), (F, 'Offset Y', 0.0),
      (F, 'Offset Z', 0.0), (F, 'Tiling X', 1.0), (F, 'Tiling Y', 1.0),
      (F, 'Tiling Z', 1.0), (F, 'Angle X', 0.0), (F, 'Angle Y', 0.0),
      (F, 'Angle Z', 0.0)],
     {'xyz_source': ('enum', 'OBJECT_XYZ', XYZ_SOURCE, "Source")},
     [(V, 'Vector')]),
]

MATERIAL_SPECS = [
    ('MaxBlend', "Blend (Max)", 'NODE_MATERIAL',
     "3ds Max's Blend material: Material 1 and Material 2 mixed by the "
     "Mix Amount, or by a Mask map linked to it; Use Curve makes the "
     "mask's Lower / Upper zone a soft threshold. Dirt over paint, rust "
     "through chrome, snow on a roof",
     [(S, 'Material 1', None), (S, 'Material 2', None), (F, 'Mix Amount', 0.5),
      (F, 'Upper', 0.75), (F, 'Lower', 0.25)],
     {'use_curve': ('bool', False, "Use Curve")},
     [(S, 'Surface')]),

    ('MaxDoubleSided', "Double Sided (Max)", 'NODE_MATERIAL',
     "3ds Max's Double Sided material: the Facing material on faces "
     "toward the camera, the Back material on faces away; Translucency "
     "lets each show through the other -- leaves, cloth, a lampshade "
     "with its lining",
     [(S, 'Facing', None), (S, 'Back', None), (F, 'Translucency', 0.0)],
     {}, [(S, 'Surface')]),

    ('MaxTopBottom', "Top/Bottom (Max)", 'NODE_MATERIAL',
     "3ds Max's Top/Bottom material: the Top material on faces whose "
     "normal points up (world Z), the Bottom material on those pointing "
     "down, Position sliding the split along the normal and Blend "
     "softening it -- snow on top, moss beneath, dust that settles",
     [(S, 'Top', None), (S, 'Bottom', None), (F, 'Blend', 0.0),
      (F, 'Position', 0.5)],
     {}, [(S, 'Surface')]),

    ('MaxShellac', "Shellac (Max)", 'NODE_MATERIAL',
     "3ds Max's Shellac material: the Shellac material's colour ADDED "
     "over the Base by Shellac Color Blend (1 = 100 percent, above "
     "overloads) -- a clear coat's sheen, a glaze, a highlight layer",
     [(S, 'Base', None), (S, 'Shellac', None), (F, 'Color Blend', 1.0)],
     {}, [(S, 'Surface')]),

    ('MaxComposite', "Composite (Max)", 'NODE_MATERIAL',
     "3ds Max's Composite material: the Base with up to four materials "
     "layered over it in order, each Additive (added by its Amount, 0..2) "
     "or Mix (blended in by its Amount, 0..1) -- decals, grime, stickers "
     "and wear built up layer by layer",
     [(S, 'Base', None), (S, 'Material 1', None), (F, 'Amount 1', 1.0),
      (S, 'Material 2', None), (F, 'Amount 2', 1.0),
      (S, 'Material 3', None), (F, 'Amount 3', 1.0),
      (S, 'Material 4', None), (F, 'Amount 4', 1.0)],
     {'mode1': ('enum', 'MIX', COMPOSITE_MODE, "Layer 1"),
      'mode2': ('enum', 'MIX', COMPOSITE_MODE, "Layer 2"),
      'mode3': ('enum', 'MIX', COMPOSITE_MODE, "Layer 3"),
      'mode4': ('enum', 'MIX', COMPOSITE_MODE, "Layer 4")},
     [(S, 'Surface')]),
]


# ========================================== R243: the Max material nodes
# Max's OWN material panels as nodes -- the third and fourth master
# shaders after the Halcyon Shader and the BI Material node. Every
# socket shows Max's label and unit (percentages 0..100, Orientation in
# degrees) over a master-compatible IDENTIFIER where the unit matches,
# or a 'Max ...' identifier where the evaluator must convert it -- so
# the bake/frame machinery treats the node exactly like the master (the
# BI node's idiom), and a percent chain can never reach the GPU raw.

MAX_SHADER_TYPE = (
    ('ANISOTROPIC', "Anisotropic", "Max's Anisotropic shader: a Gaussian "
     "highlight stretched by Anisotropy along a frame turned by "
     "Orientation -- brushed metal, hair, satin"),
    ('BLINN', "Blinn", "Max's Blinn: the half-vector highlight raised to "
     "4 x 2^(10 g) -- the Standard material's default shader"),
    ('METAL', "Metal", "Max's Metal: Cook-Torrance with Beckmann roughness "
     "and a Fresnel from the diffuse colour, the highlight in the "
     "metal's own colour; Specular Level dims the diffuse"),
    ('MULTI_LAYER', "Multi-Layer", "Max's Multi-Layer: Oren-Nayar diffuse "
     "under two anisotropic highlights, the second showing through what "
     "the first leaves -- lacquer over metal, wet paint"),
    ('OREN_NAYAR_BLINN', "Oren-Nayar-Blinn", "Max's matte shader: the "
     "rough Oren-Nayar diffuse (Roughness, Diffuse Level) under the "
     "Blinn highlight -- fabric, clay, terra cotta"),
    ('PHONG', "Phong", "Max's Phong: the reflected-ray highlight raised to "
     "2^(10 g) -- plastic, the classic look"),
    ('STRAUSS', "Strauss", "Max's Strauss: Glossiness and Metalness alone "
     "shape the surface, its highlight colour sliding from the light's "
     "to the diffuse by Metalness"),
    ('TRANSLUCENT', "Translucent Shader", "Max's Translucent: Blinn under "
     "Lambert, with the Translucent Color lit from either side of the "
     "surface -- leaves, lampshades, frosted glass"),
)

#: shader type -> the master's model identifier
MAX_MODEL_FOR = {
    'ANISOTROPIC': 'MAX_ANISOTROPIC', 'BLINN': 'MAX_BLINN',
    'METAL': 'MAX_METAL', 'MULTI_LAYER': 'MAX_MULTI_LAYER',
    'OREN_NAYAR_BLINN': 'MAX_OREN_NAYAR_BLINN', 'PHONG': 'MAX_PHONG',
    'STRAUSS': 'MAX_STRAUSS', 'TRANSLUCENT': 'MAX_TRANSLUCENT',
}

MAX_FALLOFF_DIR = (
    ('IN', "In", "Opacity falls off toward the inside: the surface grows "
     "transparent where it faces the camera, stays opaque at the edges "
     "-- Max's In"),
    ('OUT', "Out", "Opacity falls off toward the outside: the surface "
     "grows transparent at the edges, stays opaque where it faces the "
     "camera -- Max's Out, the soap-bubble look"),
)

#: (socket kind, Max's label, identifier, default). Identifiers named
#: after the master carry the master's unit; 'Max ...' identifiers are
#: Max's own (percent, degrees) and the evaluator converts them.
MAX_STANDARD_SOCKETS = (
    ('NodeSocketColor', 'Ambient', 'Max Ambient', (0.588, 0.588, 0.588, 1.0)),
    ('NodeSocketColor', 'Diffuse', 'Diffuse Color', (0.588, 0.588, 0.588, 1.0)),
    ('NodeSocketColor', 'Specular', 'Specular Color', (0.9, 0.9, 0.9, 1.0)),
    ('NodeSocketFloat', 'Self-Illumination', 'Max Self-Illum', 0.0),
    ('NodeSocketColor', 'Self-Illum Color', 'Max Self-Illum Color', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketFloat', 'Opacity', 'Max Opacity', 100.0),
    ('NodeSocketFloat', 'Specular Level', 'Max Specular Level', 0.0),
    ('NodeSocketFloat', 'Glossiness', 'Glossiness', 10.0),
    ('NodeSocketFloat', 'Soften', 'Soften', 0.1),
    ('NodeSocketFloat', 'Diffuse Level', 'Max Diffuse Level', 100.0),
    ('NodeSocketFloat', 'Roughness', 'Max Roughness', 0.0),
    ('NodeSocketFloat', 'Anisotropy', 'Max Anisotropy', 50.0),
    ('NodeSocketFloat', 'Orientation', 'Max Orientation', 0.0),
    ('NodeSocketColor', 'Specular Color 2', 'Specular Color 2', (0.9, 0.9, 0.9, 1.0)),
    ('NodeSocketFloat', 'Specular Level 2', 'Max Specular Level 2', 0.0),
    ('NodeSocketFloat', 'Glossiness 2', 'Glossiness 2', 25.0),
    ('NodeSocketFloat', 'Anisotropy 2', 'Max Anisotropy 2', 0.0),
    ('NodeSocketFloat', 'Orientation 2', 'Max Orientation 2', 0.0),
    ('NodeSocketFloat', 'Metalness', 'Max Metalness', 0.0),
    ('NodeSocketColor', 'Translucent Color', 'Translucent Color', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketFloat', 'Index of Refraction', 'IOR', 1.5),
    ('NodeSocketFloat', 'Falloff Amount', 'Max Falloff Amount', 0.0),
    ('NodeSocketFloat', 'Reflection', 'Max Reflection', 0.0),
    ('NodeSocketColor', 'Reflection Color', 'Reflection Color', (1.0, 1.0, 1.0, 1.0)),
    ('NodeSocketFloat', 'Bump', 'Bump Height', 0.5),
    ('NodeSocketVector', 'Normal', 'Normal', None),
)

MAX_STANDARD_DOCS = {
    'Ambient': "Max's Ambient colour: how the surface takes the scene's "
               "ambient light. Locked to Diffuse by default (Ambient = "
               "Diffuse); Halcyon carries its intensity as the ambient level",
    'Diffuse': "Max's Diffuse colour, the base colour of the surface; link "
               "a map here for Max's Diffuse Color map slot (Diffuse Map "
               "Amount blends it over the swatch)",
    'Specular': "Max's Specular colour, the highlight's tint (Metal and "
                "Strauss colour their highlight from the diffuse instead)",
    'Self-Illumination': "Max's Self-Illumination as a percentage: the "
                         "surface glows with its own diffuse colour by this "
                         "much, and its lit shading dims by the same share, "
                         "exactly Max's lerp toward the glow",
    'Self-Illum Color': "Max's Self-Illumination Color, used when the Color "
                        "checkbox is on: the glow is this colour, added, "
                        "with no dimming of the shading",
    'Opacity': "Max's Opacity percentage: 100 is solid, 0 invisible. Below "
               "100 the material takes the transparent road",
    'Specular Level': "Max's Specular Level percentage: the highlight's "
                      "strength, 0 (Max's default) to 999",
    'Glossiness': "Max's Glossiness percentage: the highlight's sharpness, "
                  "the exponent 2^(10 g) (Phong) or 4 x 2^(10 g) (Blinn)",
    'Soften': "Max's Soften: below this N.L the highlight's cosine folds by "
              "r (2 - r), so a highlight at a grazing light stays soft",
    'Diffuse Level': "Max's Diffuse Level percentage (Anisotropic, Multi-"
                     "Layer, Oren-Nayar-Blinn, Translucent): scales the "
                     "diffuse shading without changing its colour",
    'Roughness': "Max's Diffuse Roughness percentage (Multi-Layer, Oren-"
                 "Nayar-Blinn): the Oren-Nayar roughness, 0 Lambert to 100 "
                 "chalk",
    'Anisotropy': "Max's Anisotropy percentage: 0 a round highlight, 100 a "
                  "streak",
    'Orientation': "Max's Orientation in degrees: turns the highlight's "
                   "stretch about the normal",
    'Specular Color 2': "Max's second Specular Layer colour (Multi-Layer)",
    'Specular Level 2': "Max's second layer's Level percentage (Multi-Layer): "
                        "0 turns the layer off, Max's default",
    'Glossiness 2': "Max's second layer's Glossiness percentage (Multi-Layer)",
    'Anisotropy 2': "Max's second layer's Anisotropy percentage (Multi-Layer)",
    'Orientation 2': "Max's second layer's Orientation in degrees (Multi-Layer)",
    'Metalness': "Max's Metalness percentage (Strauss): 0 a dielectric, 100 "
                 "a metal whose highlight takes the diffuse colour",
    'Translucent Color': "Max's Translucent Color (Translucent Shader): the "
                         "colour light scatters through as, from either side "
                         "of the surface. Black is opaque",
    'Index of Refraction': "Max's Index of Refraction: bends the rays through "
                           "a transparent surface (1.0 none, 1.5 glass, 2.4 "
                           "diamond)",
    'Falloff Amount': "Max's Opacity Falloff Amount percentage: how much the "
                      "opacity falls off In or Out by the view angle (0 none)",
    'Reflection': "Max's Reflection map slot as its Amount percentage: 0 "
                  "none, 100 a mirror -- Halcyon traces the reflection "
                  "rather than reading a map",
    'Reflection Color': "Tint of the traced reflection (Max's reflection map "
                        "colour would carry this)",
    'Bump': "Max's Bump map slot: link a height map here; Bump Amount "
            "(Max's spinner, 30 by default) sets its strength",
    'Normal': "A replacement shading normal (a Normal Map node's output)",
}

#: sockets every shader type shows
MAX_STANDARD_BASE = {'Ambient', 'Diffuse', 'Specular', 'Self-Illumination',
                     'Self-Illum Color', 'Opacity', 'Specular Level',
                     'Glossiness', 'Soften', 'Index of Refraction',
                     'Falloff Amount', 'Reflection', 'Reflection Color',
                     'Bump', 'Normal'}
#: extra sockets per shader type (Max's shader-specific rollout)
MAX_STANDARD_EXTRA = {
    'ANISOTROPIC': {'Diffuse Level', 'Anisotropy', 'Orientation'},
    'BLINN': set(),
    'METAL': set(),
    'MULTI_LAYER': {'Diffuse Level', 'Roughness', 'Anisotropy', 'Orientation',
                    'Specular Color 2', 'Specular Level 2', 'Glossiness 2',
                    'Anisotropy 2', 'Orientation 2'},
    'OREN_NAYAR_BLINN': {'Diffuse Level', 'Roughness'},
    'PHONG': set(),
    'STRAUSS': {'Metalness'},
    'TRANSLUCENT': {'Diffuse Level', 'Translucent Color'},
}
#: sockets a shader type does NOT show even though the base has them
MAX_STANDARD_HIDE = {
    'METAL': {'Specular', 'Soften'},
    'STRAUSS': {'Specular', 'Specular Level', 'Soften'},
    'TRANSLUCENT': {'Soften'},
}

MAX_RAYTRACE_SHADING = tuple(
    it for it in MAX_SHADER_TYPE
    if it[0] in ('PHONG', 'BLINN', 'METAL', 'OREN_NAYAR_BLINN', 'ANISOTROPIC'))

MAX_RAYTRACE_SOCKETS = (
    ('NodeSocketColor', 'Ambient', 'Max Ambient', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketColor', 'Diffuse', 'Diffuse Color', (0.588, 0.588, 0.588, 1.0)),
    ('NodeSocketColor', 'Reflect', 'Max Reflect Color', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketColor', 'Luminosity', 'Max Luminosity', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketColor', 'Transparency', 'Max Transparency', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketFloat', 'Index of Refr', 'IOR', 1.55),
    ('NodeSocketColor', 'Specular Color', 'Specular Color', (1.0, 1.0, 1.0, 1.0)),
    ('NodeSocketFloat', 'Specular Level', 'Max Specular Level', 50.0),
    ('NodeSocketFloat', 'Glossiness', 'Glossiness', 40.0),
    ('NodeSocketFloat', 'Soften', 'Soften', 0.1),
    ('NodeSocketFloat', 'Diffuse Level', 'Max Diffuse Level', 100.0),
    ('NodeSocketFloat', 'Roughness', 'Max Roughness', 0.0),
    ('NodeSocketFloat', 'Anisotropy', 'Max Anisotropy', 50.0),
    ('NodeSocketFloat', 'Orientation', 'Max Orientation', 0.0),
    ('NodeSocketFloat', 'Bump', 'Bump Height', 0.5),
    ('NodeSocketVector', 'Normal', 'Normal', None),
)

MAX_RAYTRACE_DOCS = {
    'Ambient': "Max Raytrace's Ambient colour; Halcyon carries its intensity "
               "as the ambient level. Black (Max's default) means no ambient",
    'Diffuse': "Max Raytrace's Diffuse colour, the base colour",
    'Reflect': "Max Raytrace's Reflect colour: how much of the traced "
               "reflection the surface shows, per channel -- black none, "
               "white a mirror. Its intensity is the reflection amount, its "
               "hue the reflection's tint",
    'Luminosity': "Max Raytrace's Luminosity: the surface's own glow, added "
                  "without any lamp",
    'Transparency': "Max Raytrace's Transparency colour: black solid, white "
                    "fully see-through; its intensity sets the opacity",
    'Index of Refr': "Max Raytrace's Index of Refraction for the refracted "
                     "rays behind a transparent surface",
    'Specular Color': "Max Raytrace's Specular Color, the highlight's tint "
                      "(white by default)",
    'Specular Level': "Max Raytrace's Specular Level percentage: the "
                      "highlight's strength, 50 by default",
    'Glossiness': "Max Raytrace's Glossiness percentage: the highlight's "
                  "sharpness",
    'Soften': "Max's Soften: folds the highlight below this N.L",
    'Diffuse Level': "Max's Diffuse Level percentage (Anisotropic, Oren-"
                     "Nayar-Blinn)",
    'Roughness': "Max's Diffuse Roughness percentage (Oren-Nayar-Blinn)",
    'Anisotropy': "Max's Anisotropy percentage (Anisotropic)",
    'Orientation': "Max's Orientation in degrees (Anisotropic)",
    'Bump': "Max's Bump map slot: link a height map; Bump Amount sets its "
            "strength",
    'Normal': "A replacement shading normal (a Normal Map node's output)",
}

MAX_RAYTRACE_BASE = {'Ambient', 'Diffuse', 'Reflect', 'Luminosity',
                     'Transparency', 'Index of Refr', 'Specular Color',
                     'Specular Level', 'Glossiness', 'Soften', 'Bump', 'Normal'}
MAX_RAYTRACE_EXTRA = {
    'ANISOTROPIC': {'Diffuse Level', 'Anisotropy', 'Orientation'},
    'OREN_NAYAR_BLINN': {'Diffuse Level', 'Roughness'},
    'PHONG': set(), 'BLINN': set(), 'METAL': set(),
}
MAX_RAYTRACE_HIDE = {'METAL': {'Specular Color', 'Soften'}}


class _MaxMaterialBase:
    """Shared machinery of the two Max material nodes: sockets by
    (kind, label, identifier, default), documented, grown at load,
    hidden per shader type -- the BI Material node's idiom."""

    SOCKETS = ()
    DOCS = {}
    BASE = set()
    EXTRA = {}
    HIDE = {}

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname in ('ShaderNodeTree',)

    def _make_sockets(self):
        for kind, name, ident, default in self.SOCKETS:
            try:
                sock = self.inputs.new(kind, name, identifier=ident)
            except TypeError:
                sock = self.inputs.new(kind, ident)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
            doc = self.DOCS.get(name)
            if doc:
                try:
                    sock.description = doc
                except (AttributeError, TypeError):
                    pass

    def ensure_sockets(self):
        """Create any socket this saved instance predates (file load)."""
        have = {s.name for s in self.inputs}
        have |= {getattr(s, 'identifier', s.name) for s in self.inputs}
        for kind, name, ident, default in self.SOCKETS:
            if name in have or ident in have:
                continue
            try:
                sock = self.inputs.new(kind, name, identifier=ident)
            except TypeError:
                try:
                    sock = self.inputs.new(kind, ident)
                except Exception:                               # noqa: BLE001
                    continue
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass

    def refresh_sockets(self):
        kind = str(self.shader_type)
        keep = (set(self.BASE) | set(self.EXTRA.get(kind, ()))) \
            - set(self.HIDE.get(kind, ()))
        for sock in self.inputs:
            sock.hide = bool(sock.name not in keep and not sock.is_linked)


class HALCYON_MaxStandardNode(Node, _MaxMaterialBase):
    """3ds Max's Standard material as one node, 1:1 with its panels.

    Shader Basic Parameters (Wire, Faceted, the shader type), the
    chosen shader's own rollout (Ambient / Diffuse / Specular with the
    locks, Self-Illumination as a percentage or a colour, Opacity,
    Specular Level, Glossiness, Soften, and each shader's extras:
    Diffuse Level, Roughness, Anisotropy, Orientation, the second
    layer, Metalness, Translucent Color) and the Extended Parameters
    Halcyon can honour (Opacity Falloff In/Out and Amount, Index of
    Refraction) plus the Maps that have a road here (Diffuse with its
    Amount, Bump with its Amount, Reflection as an amount). Every
    control is Max's, in Max's unit; the node shades through the
    "(3ds Max)" models on both devices.
    """

    bl_idname = 'HALCYON_MaxStandardNode'
    bl_label = "Standard (3ds Max)"
    bl_icon = 'MATERIAL'
    bl_width_default = 220

    SOCKETS = MAX_STANDARD_SOCKETS
    DOCS = MAX_STANDARD_DOCS
    BASE = MAX_STANDARD_BASE
    EXTRA = MAX_STANDARD_EXTRA
    HIDE = MAX_STANDARD_HIDE

    def _update(self, context):
        self.refresh_sockets()

    shader_type: EnumProperty(
        name="Shader", items=MAX_SHADER_TYPE, default='BLINN', update=_update,
        description="Max's Shader Basic Parameters menu: which of the "
                    "eight Standard shaders the material uses -- each "
                    "shows its own rollout of controls below")
    wire: BoolProperty(
        name="Wire", default=False,
        description="Max's Wire: render the mesh as its edges instead of "
                    "its faces (Halcyon's wireframe road, shadeless)")
    faceted: BoolProperty(
        name="Faceted", default=False,
        description="Max's Faceted: every face shades flat by its own "
                    "normal, whatever the shader -- the un-smoothed look")
    ad_lock: BoolProperty(
        name="Ambient = Diffuse", default=True,
        description="Max's Ambient/Diffuse lock (on by default): the "
                    "Ambient colour follows the Diffuse colour")
    self_illum_color: BoolProperty(
        name="Self-Illum Color", default=False,
        description="Max's Self-Illumination Color checkbox: on, the glow "
                    "is the Self-Illum Color, added; off, the percentage "
                    "glows the diffuse colour and dims the shading")
    falloff: EnumProperty(
        name="Falloff", items=MAX_FALLOFF_DIR, default='IN',
        description="Max's Opacity Falloff direction: In fades the "
                    "facing surface, Out fades the edges")
    diffuse_map_amount: IntProperty(
        name="Diffuse Map Amount", default=100, min=0, max=100,
        description="Max's Diffuse Color map Amount: how much a linked "
                    "map replaces the Diffuse swatch, 100 entirely")
    bump_amount: IntProperty(
        name="Bump Amount", default=30, min=-999, max=999,
        description="Max's Bump map Amount: the strength of a linked "
                    "Bump height map (Max's default 30 is Halcyon's "
                    "strength 1)")

    def init(self, context):
        self._make_sockets()
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Shades with the chosen 3ds Max "
            "Standard shader on both devices")
        self.refresh_sockets()

    def draw_buttons(self, context, layout):
        layout.prop(self, 'shader_type', text="")
        row = layout.row(align=True)
        row.prop(self, 'wire')
        row.prop(self, 'faceted')
        layout.prop(self, 'ad_lock')
        layout.prop(self, 'self_illum_color')
        layout.prop(self, 'falloff')
        layout.prop(self, 'diffuse_map_amount')
        layout.prop(self, 'bump_amount')


class HALCYON_MaxRaytraceNode(Node, _MaxMaterialBase):
    """3ds Max's Raytrace material as one node: Shading (Phong, Blinn,
    Metal, Oren-Nayar-Blinn, Anisotropic), Ambient, Diffuse, Reflect
    (a colour whose intensity is the traced reflection's amount and
    whose hue its tint), Luminosity (the material's own glow),
    Transparency (a colour whose intensity is the see-through amount),
    Index of Refraction, the Specular Highlight (colour, level,
    glossiness, soften) and the Bump slot. Max's Extended Parameters
    (Extra Lighting, Translucency, Fluorescence, Color Density, Fog,
    reflection attenuation) are not built.
    """

    bl_idname = 'HALCYON_MaxRaytraceNode'
    bl_label = "Raytrace (3ds Max)"
    bl_icon = 'MATERIAL'
    bl_width_default = 220

    SOCKETS = MAX_RAYTRACE_SOCKETS
    DOCS = MAX_RAYTRACE_DOCS
    BASE = MAX_RAYTRACE_BASE
    EXTRA = MAX_RAYTRACE_EXTRA
    HIDE = MAX_RAYTRACE_HIDE

    def _update(self, context):
        self.refresh_sockets()

    shader_type: EnumProperty(
        name="Shading", items=MAX_RAYTRACE_SHADING, default='PHONG',
        update=_update,
        description="Max Raytrace's Shading menu: Phong (its default), "
                    "Blinn, Metal, Oren-Nayar-Blinn or Anisotropic")
    faceted: BoolProperty(
        name="Faceted", default=False,
        description="Max's Faceted: every face shades flat by its own "
                    "normal, whatever the shader")
    bump_amount: IntProperty(
        name="Bump Amount", default=30, min=-999, max=999,
        description="Max's Bump map Amount: the strength of a linked "
                    "Bump height map (Max's default 30 is Halcyon's "
                    "strength 1)")

    def init(self, context):
        self._make_sockets()
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Shades with the chosen shader, "
            "traced reflections and refraction on both devices")
        self.refresh_sockets()

    def draw_buttons(self, context, layout):
        layout.prop(self, 'shader_type', text="")
        layout.prop(self, 'faceted')
        layout.prop(self, 'bump_amount')


MATERIAL_NODE_CLASSES = (HALCYON_MaxStandardNode, HALCYON_MaxRaytraceNode)
#: their props, for the exporter
MATERIAL_NODE_PROPS = {
    'HALCYON_MaxStandardNode': ('shader_type', 'wire', 'faceted', 'ad_lock',
                                'self_illum_color', 'falloff',
                                'diffuse_map_amount', 'bump_amount'),
    'HALCYON_MaxRaytraceNode': ('shader_type', 'faceted', 'bump_amount'),
}

SPECS = TEXTURE_SPECS + UTILITY_SPECS + VECTOR_SPECS + MATERIAL_SPECS

#: R243: which sockets and menus a node shows under which menu value --
#: Max's rollouts show only the active lightness group and the rewire
#: menus only under Custom (pattern_nodes._make's `shows`)
SHOW_RULES = {
    'MaxColorCorrection': {
        'rewire_r': ('channels', {'CUSTOM'}), 'rewire_g': ('channels', {'CUSTOM'}),
        'rewire_b': ('channels', {'CUSTOM'}), 'rewire_a': ('channels', {'CUSTOM'}),
        'Brightness': ('lightness', {'STANDARD'}), 'Contrast': ('lightness', {'STANDARD'}),
        'Gain': ('lightness', {'ADVANCED'}), 'Gamma': ('lightness', {'ADVANCED'}),
        'Pivot': ('lightness', {'ADVANCED'}), 'Lift': ('lightness', {'ADVANCED'}),
    },
    'MaxFalloff': {
        'IOR': ('falloff_type', {'FRESNEL'}),
        'Near Distance': ('falloff_type', {'DISTANCE'}),
        'Far Distance': ('falloff_type', {'DISTANCE'}),
        'extrapolate': ('falloff_type', {'DISTANCE'}),
        'direction': ('falloff_type', {'PERP_PARALLEL', 'TOWARDS_AWAY'}),
    },
    'MaxGradientRamp': {'Mapped': ('gradient_type', {'MAPPED'})},
}


def _build(specs):
    out = []
    for spec in specs:
        name, label, icon, desc, sockets, props, outputs = spec
        # the Max nodes' own prop words, over the pattern shelf's table
        from . import pattern_nodes as _PN
        saved = {}
        for key in props:
            if key in _MAX_PROP_TIPS:
                saved[key] = _PN._PROP_TIPS.get(key)
                _PN._PROP_TIPS[key] = _MAX_PROP_TIPS[key]
        try:
            cls = _make(name, label, icon, desc, sockets, props, outputs,
                        shows=SHOW_RULES.get(name))
        finally:
            for key, old in saved.items():
                if old is None:
                    _PN._PROP_TIPS.pop(key, None)
                else:
                    _PN._PROP_TIPS[key] = old
        out.append(cls)
    return tuple(out)


TEXTURE_NODES = _build(TEXTURE_SPECS)
UTILITY_NODES = _build(UTILITY_SPECS)
VECTOR_NODES = _build(VECTOR_SPECS)
MATERIAL_NODES = _build(MATERIAL_SPECS)
NODES = TEXTURE_NODES + UTILITY_NODES + VECTOR_NODES + MATERIAL_NODES

#: what the exporter must copy across for each of them
NODE_PROPS = {f'HALCYON_{spec[0]}Node': tuple(spec[5].keys()) for spec in SPECS}
NODE_PROPS.update(MATERIAL_NODE_PROPS)

#: the 3DS Max submenu's own families: (title, icon, members)
MAX_FAMILIES = (
    ("Materials", 'NODE_MATERIAL', MATERIAL_NODE_CLASSES + MATERIAL_NODES),
    ("Textures", 'TEXTURE', TEXTURE_NODES),
    ("Utilities", 'TOOL_SETTINGS', UTILITY_NODES),
    ("Vectors", 'ORIENTATION_NORMAL', VECTOR_NODES),
)


def _family_menu(title, members):
    def draw(self, context):
        layout = self.layout
        for cls in members:
            op = layout.operator('node.add_node', text=cls.bl_label,
                                 icon=getattr(cls, 'bl_icon', 'NONE'))
            op.type = cls.bl_idname
            op.use_transform = True
    ident = 'NODE_MT_halcyon_max_' + title.lower()
    return type(ident, (bpy.types.Menu,), {
        'bl_idname': ident, 'bl_label': title, 'draw': draw})


MAX_SUBMENUS = tuple(_family_menu(t, m) for t, _i, m in MAX_FAMILIES)


class NODE_MT_halcyon_max(bpy.types.Menu):
    bl_idname = 'NODE_MT_halcyon_max'
    bl_label = "3DS Max"

    def draw(self, context):
        layout = self.layout
        for (title, icon, _m), sub in zip(MAX_FAMILIES, MAX_SUBMENUS):
            layout.menu(sub.bl_idname, icon=icon)


def register():
    for cls in MATERIAL_NODE_CLASSES + NODES:
        bpy.utils.register_class(cls)
    for sub in MAX_SUBMENUS:
        bpy.utils.register_class(sub)
    bpy.utils.register_class(NODE_MT_halcyon_max)


def unregister():
    for cls in (NODE_MT_halcyon_max,) + tuple(MAX_SUBMENUS) \
            + tuple(reversed(NODES)) + tuple(reversed(MATERIAL_NODE_CLASSES)):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass
