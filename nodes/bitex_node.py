"""The Blender Internal Texture node.

One node that IS the classic texture engine: Clouds, Wood, Marble,
Magic, Blend, Stucci, Noise, Musgrave, Voronoi and Distorted Noise,
with the original noise bases, the original parameters, the original
colorband -- evaluated by core/bitex.py on the CPU and its generated
GLSL twin on the GPU. The legacy importer builds these automatically
from 2.79-and-earlier files so old procedural materials render as
themselves; it is also a perfectly good hand tool for anyone chasing
the exact look 1990s Blender made.

The node owns the classic mapping arithmetic too: texture space is
-1..1 (Classic Space converts modern 0..1 coordinates), and Offset
and Size apply exactly as an old material's Map Input panel applied
them -- size * (co + offset), offset first.
"""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, FloatVectorProperty, IntProperty)
from bpy.types import Node, PropertyGroup

from .shader_nodes import HalcyonNodeBase

TYPE_ITEMS = [
    ('CLOUDS', "Clouds", "Fractal turbulence -- the default texture of "
     "1990s Blender, on every landscape and every skin"),
    ('WOOD', "Wood", "Bands or rings, straight or thrown by noise, with "
     "the sine, saw or triangle profile of the original"),
    ('MARBLE', "Marble", "Sine bands displaced by turbulence; Soft, Sharp "
     "and Sharper are the original square-root ladder"),
    ('MAGIC', "Magic", "The trigonometric cascade -- psychedelic bands "
     "that only ever came out of this exact formula"),
    ('BLEND', "Blend", "The classic gradient family: linear, quadratic, "
     "eased, diagonal, spherical, halo and radial"),
    ('STUCCI', "Stucci", "Noise offset through itself -- plaster walls "
     "and orange-peel surfaces, usually driving bump"),
    ('NOISE', "Noise", "True per-pixel static. Blender Internal rolled "
     "new dice every render; Halcyon seeds them per frame so renders "
     "stay reproducible"),
    ('MUSGRAVE', "Musgrave", "The five Musgrave fractals with H, "
     "lacunarity, octaves, offset and gain -- terrain out of the box"),
    ('VORONOI', "Voronoi", "Worley cells with the four feature weights, "
     "seven distance metrics and the three cell-colour modes"),
    ('DISTNOISE', "Distorted Noise", "One noise basis pushed through "
     "another -- the distortion amount sets how far"),
]

BASIS_ITEMS = [
    ('BLENDER_ORIGINAL', "Blender Original", "The in-house noise Blender "
     "shipped with from the beginning -- the default basis of almost "
     "every old file"),
    ('ORIGINAL_PERLIN', "Original Perlin", "Perlin's 1985 noise, "
     "Blender's exact orgPerlinNoise -- the +10000 domain shift, "
     "hashvectf gradients and the 1.5 scale"),
    ('IMPROVED_PERLIN', "Improved Perlin", "Perlin's 2002 noise over "
     "Blender's own hash table"),
    ('VORONOI_F1', "Voronoi F1", "Distance to the nearest feature point"),
    ('VORONOI_F2', "Voronoi F2", "Distance to the second feature point"),
    ('VORONOI_F3', "Voronoi F3", "Distance to the third feature point"),
    ('VORONOI_F4', "Voronoi F4", "Distance to the fourth feature point"),
    ('VORONOI_F2F1', "Voronoi F2-F1", "The crevice between the two "
     "nearest points"),
    ('VORONOI_CRACKLE', "Voronoi Crackle", "F2-F1 pushed to a crack "
     "network"),
    ('CELL_NOISE', "Cell Noise", "One flat random value per lattice "
     "cell"),
]

#: enum identifier -> the Tex.noisebasis code the engine evaluates
BASIS_CODES = {'BLENDER_ORIGINAL': 0, 'ORIGINAL_PERLIN': 1,
               'IMPROVED_PERLIN': 2, 'VORONOI_F1': 3, 'VORONOI_F2': 4,
               'VORONOI_F3': 5, 'VORONOI_F4': 6, 'VORONOI_F2F1': 7,
               'VORONOI_CRACKLE': 8, 'CELL_NOISE': 14}

WAVE_ITEMS = [('SIN', "Sine", "The smooth original"),
              ('SAW', "Saw", "A hard ramp that snaps back"),
              ('TRI', "Triangle", "Up and down at the same slope")]

CB_IPO_ITEMS = [
    ('LINEAR', "Linear", "Straight blends between stops"),
    ('EASE', "Ease", "Smoothstep blends between stops"),
    ('B_SPLINE', "B-Spline", "A smooth curve that approaches stops "
     "without touching them, exactly as the old ramp drew it"),
    ('CARDINAL', "Cardinal", "A smooth curve through every stop"),
    ('CONSTANT', "Constant", "Hold each stop until the next"),
]

CB_IPO_CODES = {'LINEAR': 0, 'EASE': 1, 'B_SPLINE': 2, 'CARDINAL': 3,
                'CONSTANT': 4}


class HalcyonBIStop(PropertyGroup):
    position: FloatProperty(
        name="Position", default=0.0, min=0.0, max=1.0,
        description="Where along the band this stop sits")
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        description="The stop's colour and alpha")


class HALCYON_BITextureNode(Node, HalcyonNodeBase):
    """A Blender Internal procedural texture, evaluated by the original
    algorithms. Imported legacy materials arrive as these; the settings
    are the classic texture panel's, name for name
    """

    bl_idname = 'HALCYON_BITextureNode'
    bl_label = "BI Texture"
    bl_icon = 'TEXTURE'
    bl_width_default = 190

    tex_type: EnumProperty(name="Type", items=TYPE_ITEMS, default='CLOUDS',
                           description="Which classic texture this is")
    noise_basis: EnumProperty(
        name="Basis", items=BASIS_ITEMS, default='BLENDER_ORIGINAL',
        description="The noise the texture is built from -- the old "
                    "Noise Basis selector")
    noise_basis2: EnumProperty(
        name="Distortion Basis", items=BASIS_ITEMS,
        default='BLENDER_ORIGINAL',
        description="For Distorted Noise: the basis doing the "
                    "distorting (the first basis is the one being "
                    "distorted)")
    wave: EnumProperty(
        name="Wave", items=WAVE_ITEMS, default='SIN',
        description="Wood and Marble band profile -- Blender Internal's "
                    "noisebasis2")
    hard_noise: BoolProperty(
        name="Hard", default=False,
        description="Fold the noise about its middle (the old "
                    "Soft/Hard toggle). Hard noise creases; soft "
                    "noise billows")
    noise_size: FloatProperty(
        name="Noise Size", default=0.25, min=0.0001, max=10.0,
        description="Feature size, exactly as the old panel measured "
                    "it: bigger numbers give bigger blobs")
    noise_depth: IntProperty(
        name="Depth", default=2, min=0, max=30,
        description="Octaves of turbulence layered into the texture")
    turbulence: FloatProperty(
        name="Turbulence", default=5.0, min=0.0, max=200.0,
        description="How hard the noise throws the pattern around, in "
                    "the original units (Marble and Wood expect ~5, "
                    "Stucci divides by 200 just as it always did)")

    # --------------------------------------------------- per-type settings
    clouds_color: BoolProperty(
        name="Color Clouds", default=False,
        description="Three decorrelated turbulence channels instead of "
                    "one grey -- the old Clouds 'Color' stype")
    wood_type: EnumProperty(
        name="Pattern", default='BAND', items=[
            ('BAND', "Bands", "Straight grain"),
            ('RING', "Rings", "Growth rings around the origin"),
            ('BANDNOISE', "Band Noise", "Straight grain thrown by noise"),
            ('RINGNOISE', "Ring Noise", "Rings thrown by noise")],
        description="Wood's four classic patterns")
    marble_type: EnumProperty(
        name="Sharpness", default='SOFT', items=[
            ('SOFT', "Soft", "The plain wave"),
            ('SHARP', "Sharp", "Square root of the wave: tighter veins"),
            ('SHARPER', "Sharper", "Fourth root: knife-edge veins")],
        description="Marble's vein sharpness ladder")
    blend_type: EnumProperty(
        name="Progression", default='LIN', items=[
            ('LIN', "Linear", "A straight ramp along X"),
            ('QUAD', "Quadratic", "The ramp squared"),
            ('EASE', "Ease", "The ramp smoothstepped"),
            ('DIAG', "Diagonal", "A ramp along X and Y together"),
            ('SPHERE', "Spherical", "Falls with distance from the "
             "centre"),
            ('HALO', "Quadratic Sphere", "Spherical squared -- the halo "
             "look"),
            ('RAD', "Radial", "Sweeps the angle around the centre")],
        description="The gradient's progression")
    blend_flip: BoolProperty(
        name="Flip XY", default=False,
        description="Swap the gradient's axes, as the old Flip XY "
                    "button did")
    stucci_type: EnumProperty(
        name="Profile", default='PLASTIC', items=[
            ('PLASTIC', "Plastic", "The plain offset noise"),
            ('WALLIN', "Wall In", "Dents pressed into the surface"),
            ('WALLOUT', "Wall Out", "Bumps raised out of the surface")],
        description="Stucci's three wall profiles")
    musgrave_type: EnumProperty(
        name="Fractal", default='FBM', items=[
            ('MFRACTAL', "Multifractal", "Octaves multiplied together"),
            ('RIDGEDMF', "Ridged Multifractal", "Folded ridges -- "
             "mountains and lava crusts"),
            ('HYBRIDMF', "Hybrid Multifractal", "Smooth valleys under "
             "ridged peaks"),
            ('FBM', "fBm", "Plain fractional Brownian motion"),
            ('HTERRAIN', "Hetero Terrain", "Flat lowlands, rough "
             "highlands")],
        description="Which of Musgrave's five fractals to run")
    mg_h: FloatProperty(
        name="Dimension (H)", default=1.0, min=0.0001, max=2.0,
        description="The fractal increment: higher is smoother")
    mg_lacunarity: FloatProperty(
        name="Lacunarity", default=2.0, min=0.0, max=6.0,
        description="Frequency step between octaves")
    mg_octaves: FloatProperty(
        name="Octaves", default=2.0, min=0.0, max=8.0,
        description="How many octaves, fractional part blending the "
                    "last one in")
    mg_offset: FloatProperty(
        name="Offset", default=1.0, min=0.0, max=6.0,
        description="The sea-level term of the terrain fractals")
    mg_gain: FloatProperty(
        name="Gain", default=1.0, min=0.0, max=6.0,
        description="How strongly ridges feed the next octave (ridged "
                    "and hybrid only)")
    ns_outscale: FloatProperty(
        name="Intensity", default=1.0, min=0.0, max=10.0,
        description="Output scale -- the old iScale slider")
    vn_w1: FloatProperty(
        name="W1", default=1.0, min=-2.0, max=2.0,
        description="Weight of the nearest feature distance")
    vn_w2: FloatProperty(
        name="W2", default=0.0, min=-2.0, max=2.0,
        description="Weight of the second distance")
    vn_w3: FloatProperty(
        name="W3", default=0.0, min=-2.0, max=2.0,
        description="Weight of the third distance")
    vn_w4: FloatProperty(
        name="W4", default=0.0, min=-2.0, max=2.0,
        description="Weight of the fourth distance")
    vn_mexp: FloatProperty(
        name="Exponent", default=2.5, min=0.01, max=10.0,
        description="Minkowski exponent, used by that metric only")
    vn_distm: EnumProperty(
        name="Distance", default='DISTANCE', items=[
            ('DISTANCE', "Actual Distance", "Plain euclidean distance"),
            ('DISTANCE_SQUARED', "Distance Squared", "Euclidean squared "
             "-- darker wells, cheaper metric"),
            ('MANHATTAN', "Manhattan", "Grid-walk distance: square "
             "cells"),
            ('CHEBYCHEV', "Chebychev", "Chessboard distance"),
            ('MINKOVSKY_HALF', "Minkowski 1/2", "Spiky stars"),
            ('MINKOVSKY_FOUR', "Minkowski 4", "Rounded squares"),
            ('MINKOVSKY', "Minkowski", "The general metric, shaped by "
             "the exponent")],
        description="How distance is measured between feature points")
    vn_coltype: EnumProperty(
        name="Coloring", default='INTENSITY', items=[
            ('INTENSITY', "Intensity", "Grey distances only"),
            ('POSITION', "Position", "Cells coloured by their feature "
             "point"),
            ('POSITION_OUTLINE', "Position and Outline", "Cell colours "
             "darkened at the borders"),
            ('POSITION_OUTLINE_INTENSITY', "Position, Outline and "
             "Intensity", "Borders scaled by the distance value too")],
        description="Voronoi's four colour modes")
    dist_amount: FloatProperty(
        name="Distortion", default=1.0, min=0.0, max=10.0,
        description="How far the distortion basis pushes the source "
                    "noise")

    # ------------------------------------------------ output conditioning
    bright: FloatProperty(
        name="Brightness", default=1.0, min=0.0, max=2.0,
        description="The classic post slider: 1 leaves the texture "
                    "alone")
    contrast: FloatProperty(
        name="Contrast", default=1.0, min=0.0, max=5.0,
        description="Contrast about the midpoint, applied after the "
                    "texture and colorband exactly as BI applied it")
    saturation: FloatProperty(
        name="Saturation", default=1.0, min=0.0, max=2.0,
        description="Colour saturation of RGB results")
    rgb_factors: FloatVectorProperty(
        name="RGB Factors", size=3, default=(1.0, 1.0, 1.0), min=0.0,
        max=2.0, description="Per-channel multipliers on colour "
                             "results (the old R, G, B sliders)")
    use_clamp: BoolProperty(
        name="Clamp", default=True,
        description="Clamp the result to 0..1 (BI's default; old files "
                    "that unchecked it import with it off)")

    # -------------------------------------------------- classic mapping
    classic_space: BoolProperty(
        name="Classic Space", default=True,
        description="Treat incoming 0..1 coordinates as the classic "
                    "-1..1 texture space (what every 2.79 material "
                    "expected). Turn off to feed modern coordinates "
                    "unchanged")
    tex_offset: FloatVectorProperty(
        name="Offset", size=3, default=(0.0, 0.0, 0.0), min=-10.0,
        max=10.0, description="The Map Input offset, added BEFORE size "
                              "exactly as the old pipeline did")
    tex_size: FloatVectorProperty(
        name="Size", size=3, default=(1.0, 1.0, 1.0), min=-100.0,
        max=100.0, description="The Map Input size, multiplied after "
                               "the offset")

    # ---------------------------------------------------------- colorband
    use_colorband: BoolProperty(
        name="Colorband", default=False,
        description="Run the intensity through the texture's own "
                    "colour band, exactly as the old Colors panel did")
    coba_ipotype: EnumProperty(
        name="Interpolation", items=CB_IPO_ITEMS, default='LINEAR',
        description="How colours blend between stops")
    stops: CollectionProperty(type=HalcyonBIStop)

    def init(self, context):
        self.inputs.new('NodeSocketVector', 'Vector')
        self.outputs.new('NodeSocketColor', 'Color')
        self.outputs.new('NodeSocketFloat', 'Fac')
        self.outputs.new('NodeSocketFloat', 'Alpha')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'tex_type', text="")
        t = self.tex_type
        if t in ('CLOUDS', 'WOOD', 'MARBLE', 'STUCCI', 'MUSGRAVE',
                 'DISTNOISE'):
            layout.prop(self, 'noise_basis', text="")
        if t == 'DISTNOISE':
            layout.prop(self, 'noise_basis2', text="")
        if t in ('CLOUDS', 'WOOD', 'MARBLE', 'STUCCI'):
            layout.prop(self, 'hard_noise')
        if t == 'CLOUDS':
            layout.prop(self, 'clouds_color')
            layout.prop(self, 'noise_depth')
        elif t == 'WOOD':
            layout.prop(self, 'wood_type', text="")
            layout.prop(self, 'wave', text="")
        elif t == 'MARBLE':
            layout.prop(self, 'marble_type', text="")
            layout.prop(self, 'wave', text="")
            layout.prop(self, 'noise_depth')
        elif t == 'MAGIC':
            layout.prop(self, 'noise_depth')
        elif t == 'BLEND':
            layout.prop(self, 'blend_type', text="")
            layout.prop(self, 'blend_flip')
        elif t == 'STUCCI':
            layout.prop(self, 'stucci_type', text="")
        elif t == 'NOISE':
            layout.prop(self, 'noise_depth')
        elif t == 'MUSGRAVE':
            layout.prop(self, 'musgrave_type', text="")
            col = layout.column(align=True)
            col.prop(self, 'mg_h')
            col.prop(self, 'mg_lacunarity')
            col.prop(self, 'mg_octaves')
            if self.musgrave_type in ('RIDGEDMF', 'HYBRIDMF', 'HTERRAIN'):
                col.prop(self, 'mg_offset')
            if self.musgrave_type in ('RIDGEDMF', 'HYBRIDMF'):
                col.prop(self, 'mg_gain')
            col.prop(self, 'ns_outscale')
        elif t == 'VORONOI':
            layout.prop(self, 'vn_distm', text="")
            if self.vn_distm == 'MINKOVSKY':
                layout.prop(self, 'vn_mexp')
            layout.prop(self, 'vn_coltype', text="")
            row = layout.row(align=True)
            row.prop(self, 'vn_w1', text="W1")
            row.prop(self, 'vn_w2', text="W2")
            row = layout.row(align=True)
            row.prop(self, 'vn_w3', text="W3")
            row.prop(self, 'vn_w4', text="W4")
            layout.prop(self, 'ns_outscale')
        if t in ('CLOUDS', 'WOOD', 'MARBLE', 'STUCCI', 'MUSGRAVE',
                 'VORONOI', 'DISTNOISE'):
            layout.prop(self, 'noise_size')
        if t in ('WOOD', 'MARBLE'):
            layout.prop(self, 'turbulence')
        if t == 'STUCCI':
            layout.prop(self, 'turbulence')
        if t == 'MAGIC':
            layout.prop(self, 'turbulence')
        if t == 'DISTNOISE':
            layout.prop(self, 'dist_amount')
        if self.use_colorband:
            layout.prop(self, 'coba_ipotype', text="")
            layout.label(text=f"{len(self.stops)} colorband stops "
                              "(imported)")
        layout.prop(self, 'use_colorband')

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        layout.separator()
        col = layout.column(align=True)
        col.prop(self, 'bright')
        col.prop(self, 'contrast')
        col.prop(self, 'saturation')
        col.prop(self, 'rgb_factors')
        layout.prop(self, 'use_clamp')
        layout.separator()
        layout.prop(self, 'classic_space')
        col = layout.column(align=True)
        col.prop(self, 'tex_offset')
        col.prop(self, 'tex_size')


#: everything the exporter must carry across (stops handled separately)
BI_NODE_PROPS = (
    'tex_type', 'noise_basis', 'noise_basis2', 'wave', 'hard_noise',
    'noise_size', 'noise_depth', 'turbulence', 'clouds_color',
    'wood_type', 'marble_type', 'blend_type', 'blend_flip',
    'stucci_type', 'musgrave_type', 'mg_h', 'mg_lacunarity',
    'mg_octaves', 'mg_offset', 'mg_gain', 'ns_outscale', 'vn_w1',
    'vn_w2', 'vn_w3', 'vn_w4', 'vn_mexp', 'vn_distm', 'vn_coltype',
    'dist_amount', 'bright', 'contrast', 'saturation', 'rgb_factors',
    'use_clamp', 'classic_space', 'tex_offset', 'tex_size',
    'use_colorband', 'coba_ipotype',
)

CLASSES = (HalcyonBIStop, HALCYON_BITextureNode)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
