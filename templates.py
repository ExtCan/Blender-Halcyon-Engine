"""Ready-made materials, built from the Halcyon shader and its own textures.

Each entry is a recipe rather than a saved blend file: the node tree is
constructed on demand, so a template always matches the current version of the
nodes instead of rotting into a set of sockets that no longer exist.

They are also the quickest way to see what the master shader's less obvious
inputs actually do — Fresnel on the chrome, the sun rim on the wax, edge opacity
on the glass, the sheen sockets on the silk.

Every template carries a category: SIMPLE recipes set master-shader sockets
only; ADVANCED ones wire the engine's own procedural textures in, sometimes
through a Bump node into the normal (the proven Water anatomy). The menu
draws the two groups under their own headings.

Texture entries come in two shapes: the original 4-tuple
(idname, {node props}, {input sockets}, target) linking the node's first
output straight to a master socket, and the dict form
{'node', 'props', 'inputs', 'output', 'target', 'bump'} which can pick a
named output and, when 'bump' is set, route it through a ShaderNodeBump of
that strength into the target (for Normal).
"""

import bpy
from bpy.props import EnumProperty
from bpy.types import Menu, Operator

# (model, {socket: value}, [(texture node, {props}, {socket: value}, target)])
TEMPLATES = {
    # ------------------------------------------------------------- simple
    'CHROME': {
        'label': "Chrome",
        'category': 'SIMPLE',
        'note': "Metal shader, no diffuse, hard Fresnel. Reflection needs ray "
                "tracing on; without it the environment colour stands in",
        'model': 'METAL',
        'inputs': {'Diffuse Color': (0.55, 0.57, 0.60, 1.0),
                   'Diffuse Level': 0.04, 'Specular Level': 1.0,
                   'Specular Color': (0.92, 0.96, 1.0, 1.0),
                   'Glossiness': 900.0, 'Metalness': 1.0, 'Soften': 0.05,
                   'Reflection': 0.95,
                   'Reflection Color': (0.96, 0.98, 1.0, 1.0),
                   'Fresnel': 0.9, 'Fresnel Power': 5.0,
                   'Fresnel Color': (0.85, 0.92, 1.0, 1.0)},
    },
    'GOLD': {
        'label': "Gold",
        'category': 'SIMPLE',
        'note': "The highlight takes the base colour rather than the light's, "
                "which is what keeps gold gold in its own reflection",
        'model': 'METAL',
        'inputs': {'Diffuse Color': (1.0, 0.72, 0.29, 1.0),
                   'Diffuse Level': 0.12,
                   'Specular Color': (1.0, 0.86, 0.55, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 340.0,
                   'Metalness': 1.0, 'Soften': 0.04,
                   'Reflection': 0.72,
                   'Reflection Color': (1.0, 0.80, 0.45, 1.0),
                   'Fresnel': 0.5, 'Fresnel Power': 4.0,
                   'Fresnel Color': (1.0, 0.9, 0.6, 1.0),
                   'Sheen': 0.15, 'Sheen Color': (1.0, 0.85, 0.55, 1.0),
                   'Sheen Roughness': 0.5},
    },
    'GLASS': {
        'label': "Glass",
        'category': 'SIMPLE',
        'note': "Thin in the middle and thick at the silhouette, which is how "
                "glass reads without refracting anything",
        'model': 'BLINN',
        'inputs': {'Diffuse Color': (0.88, 0.93, 0.95, 1.0),
                   'Diffuse Level': 0.06, 'Specular Level': 1.0,
                   'Glossiness': 650.0, 'Opacity': 0.10,
                   'Edge Opacity': 0.95, 'IOR': 1.52,
                   'Refraction Amount': 1.0,
                   'Fresnel': 1.1, 'Fresnel Power': 4.0,
                   'Reflection': 0.5,
                   'Reflection Color': (0.95, 0.98, 1.0, 1.0)},
    },
    'PLASTIC': {
        'label': "Shiny Plastic",
        'category': 'SIMPLE',
        'note': "The 1990s default: a tight white highlight over flat colour",
        'model': 'PHONG',
        'inputs': {'Diffuse Color': (0.75, 0.15, 0.12, 1.0),
                   'Specular Level': 0.9, 'Glossiness': 90.0,
                   'Soften': 0.12, 'Fresnel': 0.35, 'Fresnel Power': 3.5,
                   'Reflection': 0.08},
    },
    'RUBBER': {
        'label': "Rubber",
        'category': 'SIMPLE',
        'note': "Oren-Nayar keeps the edges bright, which is what stops matte "
                "surfaces looking like flat paint",
        'model': 'OREN_NAYAR',
        'inputs': {'Diffuse Color': (0.08, 0.08, 0.09, 1.0),
                   'Roughness': 0.9, 'Specular Level': 0.10,
                   'Glossiness': 10.0, 'Sheen': 0.30,
                   'Sheen Color': (0.55, 0.55, 0.58, 1.0),
                   'Sheen Roughness': 0.85},
        'textures': [{'node': 'HALCYON_DentsNode', 'props': {},
                      'inputs': {'Scale': 22.0},
                      'output': 'Fac', 'target': 'Normal',
                      'bump': 0.08}],
    },
    'TOON': {
        'label': "Cel Shaded",
        'category': 'SIMPLE',
        'note': "Banded diffuse with a hard highlight and a rim to lift it off "
                "the background",
        'model': 'TOON',
        'inputs': {'Diffuse Color': (0.30, 0.55, 0.85, 1.0),
                   'Toon Size': 0.55, 'Toon Smooth': 0.02,
                   'Specular Level': 0.6, 'Rim Amount': 0.8,
                   'Rim Light': (1.0, 1.0, 0.9, 1.0), 'Rim Power': 4.0},
    },
    'VELVET': {
        'label': "Velvet",
        'category': 'SIMPLE',
        'note': "Minnaert darkens the middle and lifts the edges, which is the "
                "whole look of pile fabric",
        'model': 'MINNAERT',
        'inputs': {'Diffuse Color': (0.42, 0.04, 0.14, 1.0),
                   'Roughness': 0.85, 'Specular Level': 0.10,
                   'Glossiness': 14.0,
                   'Sheen': 0.85, 'Sheen Color': (0.95, 0.55, 0.65, 1.0),
                   'Sheen Roughness': 0.65,
                   'Rim Amount': 0.45, 'Rim Power': 2.6,
                   'Rim Light': (0.95, 0.55, 0.65, 1.0)},
    },
    'HOLOGRAM': {
        'label': "Hologram",
        'category': 'SIMPLE',
        'note': "Transparent in the middle, bright at the silhouette, scanned "
                "with an ordered dither",
        'model': 'CONSTANT',
        'inputs': {'Diffuse Color': (0.2, 0.9, 0.8, 1.0), 'Opacity': 0.15,
                   'Edge Opacity': 0.95, 'Fresnel': 2.0, 'Fresnel Power': 2.0,
                   'Self-Illumination': (0.1, 0.5, 0.45, 1.0)},
    },
    'WIREFRAME': {
        'label': "Wireframe",
        'category': 'SIMPLE',
        'note': "Edges only, the rest see-through",
        'model': 'WIREFRAME',
        'inputs': {'Diffuse Color': (0.2, 1.0, 0.4, 1.0)},
    },
    'PORCELAIN': {
        'label': "Porcelain",
        'category': 'SIMPLE',
        'note': "Near-white with a tight highlight and a Fresnel rim: the "
                "glazed-china read every teapot demo was after",
        'model': 'BLINN',
        'inputs': {'Diffuse Color': (0.94, 0.93, 0.90, 1.0),
                   'Specular Level': 0.95, 'Glossiness': 420.0,
                   'Soften': 0.06, 'Reflection': 0.12,
                   'Fresnel': 0.7, 'Fresnel Power': 3.5,
                   'Fresnel Color': (0.85, 0.90, 1.0, 1.0)},
    },
    'CANDY': {
        'label': "Candy Apple",
        'category': 'SIMPLE',
        'note': "Saturated colour under a wet coat: high gloss plus a little "
                "reflection. The show-car paint of every 90s logo spin",
        'model': 'PHONG',
        'inputs': {'Diffuse Color': (0.72, 0.03, 0.04, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 480.0,
                   'Soften': 0.03, 'Reflection': 0.38,
                   'Fresnel': 0.8, 'Fresnel Power': 4.5,
                   'Fresnel Color': (1.0, 0.85, 0.8, 1.0)},
    },
    'CLAY': {
        'label': "Terracotta Clay",
        'category': 'SIMPLE',
        'note': "Fired earth: rough diffuse, almost no highlight. What matte "
                "test renders looked like before everything went shiny",
        'model': 'OREN_NAYAR',
        'inputs': {'Diffuse Color': (0.62, 0.35, 0.24, 1.0),
                   'Roughness': 0.72, 'Specular Level': 0.05,
                   'Glossiness': 7.0},
        'textures': [{'node': 'HALCYON_DentsNode', 'props': {},
                      'inputs': {'Scale': 9.0},
                      'output': 'Fac', 'target': 'Normal',
                      'bump': 0.12}],
    },
    'SILK': {
        'label': "Silk",
        'category': 'SIMPLE',
        'note': "Ward's anisotropic sheen stretched along the threads -- the "
                "sheen sockets doing the work they were added for",
        'model': 'WARD',
        'inputs': {'Diffuse Color': (0.50, 0.32, 0.52, 1.0),
                   'Specular Level': 0.5, 'Glossiness': 40.0,
                   'Anisotropy': 0.6, 'Sheen': 0.6,
                   'Sheen Color': (0.92, 0.82, 0.95, 1.0),
                   'Sheen Roughness': 0.4},
    },
    'GHOST': {
        'label': "Ghost",
        'category': 'SIMPLE',
        'note': "Barely there in the middle, a pale glow at the edge. Edge "
                "opacity and a soft self-illumination do all of it",
        'model': 'CONSTANT',
        'inputs': {'Diffuse Color': (0.55, 0.75, 0.90, 1.0),
                   'Opacity': 0.08, 'Edge Opacity': 0.6,
                   'Fresnel': 1.5, 'Fresnel Power': 2.0,
                   'Self-Illumination': (0.12, 0.20, 0.30, 1.0)},
    },
    'CAR_PAINT': {
        'label': "Car Paint",
        'category': 'SIMPLE',
        'note': "A deep base under a clear coat: multi-layer highlight, "
                "reflection, and a Fresnel colour shift toward the horizon",
        'model': 'MULTI_LAYER',
        'inputs': {'Diffuse Color': (0.42, 0.02, 0.05, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 460.0,
                   'Metalness': 0.35, 'Soften': 0.04,
                   'Reflection': 0.45, 'Fresnel': 0.9,
                   'Fresnel Power': 4.5,
                   'Fresnel Color': (0.9, 0.55, 0.35, 1.0)},
    },
    'NEON': {
        'label': "Neon Sign",
        'category': 'SIMPLE',
        'note': "Pure self-illumination, no shading at all. Turn the Glow "
                "post effect on and it blooms like the real tube",
        'model': 'CONSTANT',
        'inputs': {'Diffuse Color': (0.0, 0.0, 0.0, 1.0),
                   'Diffuse Level': 0.0,
                   'Self-Illumination': (1.0, 0.2, 0.8, 1.0)},
    },
    # ----------------------------------------------------------- advanced
    'BRUSHED': {
        'label': "Brushed Metal",
        'category': 'ADVANCED',
        'note': "Anisotropic highlight stretched by a Scratches texture "
                "driving the rotation",
        'model': 'ANISOTROPIC',
        'inputs': {'Diffuse Color': (0.42, 0.44, 0.47, 1.0),
                   'Specular Level': 0.95, 'Glossiness': 150.0,
                   'Anisotropy': 0.85, 'Metalness': 0.9,
                   'Reflection': 0.28,
                   'Reflection Color': (0.9, 0.92, 0.95, 1.0),
                   'Fresnel': 0.45, 'Fresnel Power': 4.0},
        'textures': [('HALCYON_ScratchesNode', {'count': 24},
                      {'Scale': 6.0, 'Width': 0.01}, 'Anisotropic Rotation')],
    },
    'MARBLE': {
        'label': "Polished Marble",
        'category': 'ADVANCED',
        'note': "Solid marble with veins running through the object, not "
                "wrapped around it",
        'model': 'BLINN',
        'inputs': {'Specular Level': 0.85, 'Glossiness': 320.0,
                   'Soften': 0.05, 'Reflection': 0.18,
                   'Fresnel': 0.45, 'Fresnel Power': 3.5},
        'textures': [('HALCYON_MarbleNode', {'octaves': 6},
                      {'Scale': 2.4, 'Turbulence': 1.5}, 'Diffuse Color')],
    },
    'WOOD': {
        'label': "Varnished Wood",
        'category': 'ADVANCED',
        'note': "Growth rings turned about the object's own axis",
        'model': 'BLINN_PHONG',
        'inputs': {'Specular Level': 0.6, 'Glossiness': 130.0,
                   'Soften': 0.05, 'Reflection': 0.10,
                   'Fresnel': 0.5, 'Fresnel Power': 3.5},
        'textures': [('HALCYON_WoodNode', {'octaves': 5},
                      {'Scale': 2.0, 'Rings': 11.0, 'Turbulence': 0.55},
                      'Diffuse Color')],
    },
    'TERRAIN': {
        'label': "Terrain",
        'category': 'ADVANCED',
        'note': "Granite mixed into the base colour, with displacement driving "
                "the bump",
        'model': 'LAMBERT',
        'inputs': {'Specular Level': 0.05},
        'textures': [('HALCYON_GraniteNode', {'octaves': 7},
                      {'Scale': 5.0, 'Contrast': 1.4}, 'Diffuse Color')],
    },
    'WATER': {
        'label': "Water",
        'category': 'ADVANCED',
        'note': "The proven Water anatomy: interfering ripples through a Bump "
                "into the normal over a glassy blend. With ray tracing on it "
                "truly refracts (IOR 1.33); without it the sorted blend and "
                "the environment stand in",
        'model': 'BLINN',
        'inputs': {'Diffuse Color': (0.10, 0.22, 0.28, 1.0),
                   'Diffuse Level': 0.3, 'Specular Level': 1.0,
                   'Glossiness': 320.0, 'Opacity': 0.35, 'Edge Opacity': 0.85,
                   'IOR': 1.33, 'Reflection': 0.5, 'Fresnel': 1.2,
                   'Fresnel Power': 3.0, 'Refraction Amount': 1.0},
        'textures': [{'node': 'HALCYON_RipplesNode',
                      'props': {'sources': 4, 'seed': 7},
                      'inputs': {'Scale': 1.5, 'Frequency': 9.0,
                                 'Decay': 0.55},
                      'output': 'Fac', 'target': 'Normal', 'bump': 0.35}],
    },
    'LAVA': {
        'label': "Lava",
        'category': 'ADVANCED',
        'note': "Molten cracks glowing between dark plates: the crackle "
                "boundary network feeds self-illumination, ridged noise "
                "roughens the crust. Made for the Glow post effect",
        'model': 'LAMBERT',
        'inputs': {'Diffuse Color': (0.10, 0.03, 0.02, 1.0),
                   'Diffuse Level': 0.9, 'Specular Level': 0.05,
                   'Glossiness': 10.0},
        'textures': [{'node': 'HALCYON_CrackleNode',
                      'inputs': {'Scale': 3.0, 'Width': 0.14, 'Smooth': 0.05,
                                 'Color 1': (1.0, 0.32, 0.03, 1.0),
                                 'Color 2': (0.05, 0.01, 0.005, 1.0)},
                      'output': 'Color', 'target': 'Self-Illumination'},
                     {'node': 'HALCYON_NoiseNode',
                      'props': {'kind': 'RIDGED', 'octaves': 5},
                      'inputs': {'Scale': 5.0},
                      'output': 'Fac', 'target': 'Normal', 'bump': 0.45}],
    },
    'TILE_FLOOR': {
        'label': "Tiled Floor",
        'category': 'ADVANCED',
        'note': "Bevelled tiles with grout and per-tile variation under a "
                "glossy coat -- the kitchen floor of every raytracer demo",
        'model': 'BLINN',
        'inputs': {'Specular Level': 0.8, 'Glossiness': 120.0,
                   'Fresnel': 0.5, 'Reflection': 0.15},
        'textures': [{'node': 'HALCYON_TilesNode',
                      'inputs': {'Scale': 2.0, 'Grout': 0.05, 'Bevel': 0.3,
                                 'Variation': 0.35},
                      'output': 'Color', 'target': 'Diffuse Color'}],
    },
    'BRICK_WALL': {
        'label': "Brick Wall",
        'category': 'ADVANCED',
        'note': "Running-bond brickwork with mortar courses and per-brick "
                "variation, matte as fired clay",
        'model': 'LAMBERT',
        'inputs': {'Specular Level': 0.04},
        'textures': [{'node': 'HALCYON_BrickNode',
                      'inputs': {'Scale': 2.0, 'Variation': 0.35},
                      'output': 'Color', 'target': 'Diffuse Color'}],
    },
    'HAMMERED': {
        'label': "Hammered Copper",
        'category': 'ADVANCED',
        'note': "Metal dented by the Dents texture through a Bump -- each pit "
                "catches the highlight on its own",
        'model': 'METAL',
        'inputs': {'Diffuse Color': (0.72, 0.45, 0.20, 1.0),
                   'Diffuse Level': 0.2, 'Specular Level': 1.0,
                   'Glossiness': 90.0, 'Metalness': 1.0, 'Reflection': 0.45,
                   'Fresnel': 0.8},
        'textures': [{'node': 'HALCYON_DentsNode',
                      'props': {'octaves': 3},
                      'inputs': {'Scale': 5.0, 'Depth': 1.0},
                      'output': 'Fac', 'target': 'Normal', 'bump': 0.5}],
    },
    'LEOPARD': {
        'label': "Leopard Print",
        'category': 'ADVANCED',
        'note': "POV-Ray's leopard spots straight into the base colour, with "
                "a soft fabric highlight",
        'model': 'BLINN_PHONG',
        'inputs': {'Specular Level': 0.15, 'Glossiness': 20.0},
        'textures': [{'node': 'HALCYON_LeopardNode',
                      'inputs': {'Scale': 4.0},
                      'output': 'Color', 'target': 'Diffuse Color'}],
    },
    'CLOTH': {
        'label': "Woven Cloth",
        'category': 'ADVANCED',
        'note': "Warp and weft threads over-under, rough as fabric should be",
        'model': 'OREN_NAYAR',
        'inputs': {'Roughness': 0.8, 'Specular Level': 0.08,
                   'Glossiness': 8.0},
        'textures': [{'node': 'HALCYON_WeaveNode',
                      'inputs': {'Scale': 10.0},
                      'output': 'Color', 'target': 'Diffuse Color'}],
    },
    'DEAD_CHANNEL': {
        'label': "Dead Channel",
        'category': 'ADVANCED',
        'note': "An untuned television: per-cell static reseeded every frame, "
                "self-lit so it reads in a dark room. Scale sets the set's "
                "pixel size on the surface",
        'model': 'CONSTANT',
        'inputs': {'Diffuse Color': (0.02, 0.02, 0.02, 1.0),
                   'Diffuse Level': 0.1},
        'textures': [{'node': 'HALCYON_StaticNode',
                      'inputs': {'Scale': 48.0},
                      'output': 'Color', 'target': 'Self-Illumination'}],
    },

    # ------------------------------------------- R204: Bryce 1995
    # The DECODED library. R203 recreated these from preview
    # pixels; the field called that 'nothing like they do in
    # Bryce'. Every channel below now comes straight out of the
    # preset's own C3dBaseMaterial record (core/brycemat.py
    # decode_material: diffuse/ambient/specular colours, the
    # per-channel specular coefficients, transparency,
    # reflection, bump, metallic flag, refractive index -- the
    # glass ladder reads 1.12/1.52/1.68, Diamond 2.55, exactly
    # as MetaTools saved them), and every texture's palette,
    # frequency and octave count out of its 3dTxt_TxtData
    # record, with the 1995 preview arbitrating which component
    # paints. The notes quote the original library text.
    'B95_POLISHED_GOLD': {
        'label': 'Polished Gold',
        'note': 'From the Bryce 1995 library',
        'model': 'METAL',
        'category': 'SIMPLE',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.804, 0.678, 0.141, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 138.0,
            'Reflection': 0.72,
            'Reflection Color': (0.804, 0.678, 0.141, 1.0),
            'Metalness': 1.0,
        },
    },
    'B95_BRUSHED_GOLD': {
        'label': 'Brushed Gold',
        'note': 'From the Bryce 1995 library',
        'model': 'METAL',
        'category': 'ADVANCED',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.804, 0.678, 0.141, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 10.0,
            'Reflection': 0.72,
            'Reflection Color': (0.804, 0.678, 0.141, 1.0),
            'Metalness': 1.0,
        },
    },
    'B95_POLISHED_SILVER': {
        'label': 'Polished Silver',
        'note': 'From the Bryce 1995 library',
        'model': 'METAL',
        'category': 'SIMPLE',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.78, 0.78, 0.78, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 138.0,
            'Reflection': 0.72,
            'Reflection Color': (0.78, 0.78, 0.78, 1.0),
            'Metalness': 1.0,
        },
    },
    'B95_BRUSHED_SILVER': {
        'label': 'Brushed Silver',
        'note': 'From the Bryce 1995 library',
        'model': 'METAL',
        'category': 'ADVANCED',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.78, 0.78, 0.78, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 10.0,
            'Reflection': 0.72,
            'Reflection Color': (0.78, 0.78, 0.78, 1.0),
            'Metalness': 1.0,
        },
    },
    'B95_BUMPY_GOLD': {
        'label': 'Bumpy Gold',
        'note': 'Bryce 1995: A nice gold brick texture...',
        'model': 'METAL',
        'category': 'ADVANCED',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.718, 0.635, 0.0, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 0.392,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 41.0,
            'Reflection': 0.72,
            'Reflection Color': (0.718, 0.635, 0.0, 1.0),
            'Metalness': 1.0,
        },
        'textures': [
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (0.518, 0.588, 0.694, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.232,
            },
        ],
    },
    'B95_PITTED_GOLD': {
        'label': 'Pitted Gold',
        'note': 'Bryce 1995: A gold with many imperfections...',
        'model': 'METAL',
        'category': 'ADVANCED',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.718, 0.635, 0.0, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 0.392,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 41.0,
            'Reflection': 0.72,
            'Reflection Color': (0.718, 0.635, 0.0, 1.0),
            'Metalness': 1.0,
        },
        'textures': [
            {
                'node': 'HALCYON_DentsNode',
                'props': {},
                'inputs': {'Scale': 4.0, 'Color 1': (0.345, 0.275, 0.02, 1.0), 'Color 2': (0.09, 0.0, 0.204, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.226,
            },
        ],
    },
    'B95_MIRROR': {
        'label': 'Mirror',
        'note': 'Bryce 1995: Perfectly reflective mirror surface.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.0, 0.129, 0.808, 1.0),
            'Diffuse Level': 0.45,
            'Specular Level': 1.0,
            'Specular Color': (0.918, 0.936, 1.0, 1.0),
            'Glossiness': 106.0,
            'Reflection': 1.0,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
        },
    },
    'B95_METALLIC_CHROME': {
        'label': 'Metallic Chrome',
        'note': 'Bryce 1995: Similar to the Mirror, but with a deep blue diffuse element as well as less reflectivity.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.0, 0.129, 0.808, 1.0),
            'Diffuse Level': 0.235,
            'Specular Level': 1.0,
            'Specular Color': (0.918, 0.936, 1.0, 1.0),
            'Glossiness': 106.0,
            'Reflection': 0.498,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
        },
    },
    'B95_NICE_COPPER': {
        'label': 'Nice Copper',
        'note': 'From the Bryce 1995 library',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'METAL',
        'inputs': {
            'Diffuse Color': (0.773, 0.475, 0.153, 1.0),
            'Diffuse Level': 0.278,
            'Ambient': 0.136,
            'Specular Level': 0.392,
            'Specular Color': (1.0, 0.991, 0.515, 1.0),
            'Glossiness': 53.0,
            'Reflection': 0.271,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 3},
                'inputs': {'Scale': 3.51, 'Contrast': 1.2, 'Color 1': (1.0, 1.0, 0.341, 1.0), 'Color 2': (0.58, 0.525, 0.525, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.063,
            },
        ],
    },
    'B95_RUSTIC_IRON': {
        'label': 'Rustic Iron',
        'note': 'From the Bryce 1995 library',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'METAL',
        'inputs': {
            'Diffuse Level': 0.882,
            'Ambient': 0.249,
            'Specular Level': 0.471,
            'Specular Color': (1.0, 0.864, 0.657, 1.0),
            'Glossiness': 18.0,
            'Reflection': 0.271,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
        },
        'textures': [
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 1.13, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (0.64, 0.62, 0.6, 1.0), 'Color 2': (0.4, 0.4, 0.4, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 1.13, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (0.64, 0.62, 0.6, 1.0), 'Color 2': (0.4, 0.4, 0.4, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.069,
            },
        ],
    },
    'B95_LIGHT_WOOD': {
        'label': 'Light Wood',
        'note': 'Bryce 1995: Good wood for smaller objects... Has a very high frequency "grain" to it.',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'WOOD',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.489,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 0.5, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (1.0, 0.776, 0.498, 1.0), 'Color 2': (0.596, 0.322, 0.208, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_BLEACHED_WOOD': {
        'label': 'Bleached Wood',
        'note': 'From the Bryce 1995 library',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'WOOD',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.306,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 0.5, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (1.0, 0.796, 0.576, 1.0), 'Color 2': (0.35, 0.279, 0.202, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_WALNUT_WOOD': {
        'label': 'Walnut Wood',
        'note': 'Bryce 1995: Nice dark wood...',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'WOOD',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.489,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 0.87, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (0.6, 0.32, 0.21, 1.0), 'Color 2': (0.39, 0.0, 0.0, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_POLISHED_WALNUT': {
        'label': 'Polished Walnut',
        'note': 'Bryce 1995: Specularity added for shine...',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'WOOD',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.489,
            'Specular Level': 0.604,
            'Specular Color': (1.0, 0.894, 0.749, 1.0),
            'Glossiness': 115.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 0.5, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (0.6, 0.32, 0.21, 1.0), 'Color 2': (0.39, 0.0, 0.0, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_WARPED_WOOD': {
        'label': 'Warped Wood',
        'note': 'Bryce 1995: First of several wood textures; color pattern has the appearance of a bump map.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'WOOD',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.489,
            'Specular Level': 0.345,
            'Specular Color': (1.0, 0.894, 0.749, 1.0),
            'Glossiness': 115.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WoodNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 0.87, 'Rings': 6.0, 'Turbulence': 0.35, 'Color 1': (1.0, 0.776, 0.498, 1.0), 'Color 2': (0.596, 0.322, 0.208, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_LIGHT_GLASS': {
        'label': 'Light Glass',
        'note': 'Bryce 1995: This glass has a low refraction level and increasing reflection at low angles.  Again, no color...',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.992, 0.992, 0.992, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Reflection': 0.102,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
            'Opacity': 0.553,
            'IOR': 1.12,
            'Refraction Amount': 1.0,
        },
    },
    'B95_STANDARD_GLASS': {
        'label': 'Standard Glass',
        'note': 'Bryce 1995: This glass has the refraction of typical glass.  There is a moderate amount of automatic reflectivity.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.992, 0.992, 0.992, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.52,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_HEAVY_GLASS': {
        'label': 'Heavy Glass',
        'note': 'Bryce 1995: This glass has a fairly high amount of refraction.  Again, the amount of reflectivity is a function of the amount of refraction.  Notice how reflectio',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.992, 0.992, 0.992, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.68,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_CRYSTAL': {
        'label': 'Crystal',
        'note': 'Bryce 1995: This glass has a high amount of refraction!  Crystal is a very hard substance.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.992, 0.992, 0.992, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.88,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_DIAMOND': {
        'label': 'Diamond',
        'note': 'Bryce 1995: This glass is the hardest and has the most refraction.  Diamonds are, in fact, the hardest substances on earth!',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.992, 0.992, 0.992, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 2.55,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_SMOKED_GLASS': {
        'label': 'Smoked Glass',
        'note': 'Bryce 1995: Here is a standard glass with a moderate-grey diffuse color.  This gives a smoked effect without affecting the shadow.  For transparent materials, the',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.478, 0.478, 0.478, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.52,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_ROSE_GLASS': {
        'label': 'Rose Glass',
        'note': 'Bryce 1995: Another diffuse-colored standard glass.  Again, the shadow color is unaffected.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (1.0, 0.722, 0.969, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.52,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_GREEN_GLASS': {
        'label': 'Green Glass',
        'note': 'Bryce 1995: Another diffuse-colored glass, this time green.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.655, 1.0, 0.647, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.52,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_AQUA_GLASS': {
        'label': 'Aqua Glass',
        'note': 'Bryce 1995: Another combo colored glass with a nice marine feel.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'GLASS',
        'inputs': {
            'Diffuse Color': (0.127, 0.69, 0.676, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
            'Opacity': 0.5,
            'IOR': 1.52,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
    },
    'B95_SANDSTONE': {
        'label': 'Sandstone',
        'note': 'Bryce 1995: Multi-channelled procedural texture... Very realistic, layered rock.',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.235,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 9},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (0.71, 0.62, 0.51, 1.0), 'Color 2': (0.29, 0.18, 0.15, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 9},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (0.71, 0.62, 0.51, 1.0), 'Color 2': (0.29, 0.18, 0.15, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.264,
            },
        ],
    },
    'B95_PITTED_GRANITE': {
        'label': 'Pitted Granite',
        'note': 'Bryce 1995: Multi-channelled procedural texture with elaborate bump mapping... Very realistic!',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.188,
            'Specular Level': 0.455,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 221.0,
        },
        'textures': [
            {
                'node': 'HALCYON_MarbleNode',
                'props': {'octaves': 3},
                'inputs': {'Scale': 3.5, 'Turbulence': 1.6, 'Veins': 3.0, 'Color 1': (0.533, 0.314, 0.231, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_MarbleNode',
                'props': {'octaves': 3},
                'inputs': {'Scale': 3.5, 'Turbulence': 1.6, 'Veins': 3.0, 'Color 1': (0.533, 0.314, 0.231, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.251,
            },
        ],
    },
    'B95_WET_ROCK': {
        'label': 'Wet Rock',
        'note': 'Bryce 1995: Multi-channel, altitude sensitive rock.',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.282,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (0.808, 0.733, 0.565, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_AgateNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 2.32, 'Turbulence': 0.5, 'Bands': 6.0, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.22, 0.18, 0.18, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.464,
            },
        ],
    },
    'B95_MARBLE_RED_AND_WHITE': {
        'label': 'Marble, Red and White',
        'note': 'Bryce 1995: Very realistic!  Recommended for artificial objects.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.151,
            'Specular Level': 0.576,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
        },
        'textures': [
            {
                'node': 'HALCYON_MarbleNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 3.5, 'Turbulence': 1.6, 'Veins': 3.0, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.651, 0.357, 0.341, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_MARBLE_BLACK_ON_WHITE': {
        'label': 'Marble, Black on White',
        'note': 'Bryce 1995: More marble, though much more stark than the others...',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 0.784,
            'Ambient': 0.358,
            'Specular Level': 0.22,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
        },
        'textures': [
            {
                'node': 'HALCYON_MarbleNode',
                'props': {'octaves': 3},
                'inputs': {'Scale': 3.5, 'Turbulence': 1.6, 'Veins': 3.0, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_MARBLE_BLUE_AND_WHITE': {
        'label': 'Marble, Blue and White',
        'note': 'Bryce 1995: Looks sorta like a satellite photo of the earth!',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 0.969,
            'Ambient': 0.254,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_MarbleNode',
                'props': {'octaves': 5},
                'inputs': {'Scale': 3.5, 'Turbulence': 1.6, 'Veins': 3.0, 'Color 1': (0.93, 0.93, 1.0, 1.0), 'Color 2': (0.0, 0.02, 0.51, 1.0)},
                'target': 'Diffuse Color',
            },
        ],
    },
    'B95_RIVERBED': {
        'label': 'Riverbed',
        'note': 'Bryce 1995: Detailed bump works well for stoney floors.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.132,
            'Specular Level': 0.22,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (0.898, 0.851, 0.725, 1.0), 'Color 2': (0.314, 0.298, 0.254, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_AgateNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.72, 'Turbulence': 0.5, 'Bands': 6.0, 'Color 1': (0.467, 0.482, 0.467, 1.0), 'Color 2': (0.163, 0.169, 0.163, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.251,
            },
        ],
    },
    'B95_KRYPTONITE': {
        'label': 'Kryptonite',
        'note': 'Bryce 1995: Lois Lane is having this cut for her wedding ring.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'MINERAL',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.32,
            'Specular Level': 1.0,
            'Specular Color': (0.854, 1.0, 0.854, 1.0),
            'Glossiness': 24.0,
        },
        'textures': [
            {
                'node': 'HALCYON_CrackleNode',
                'props': {},
                'inputs': {'Scale': 7.66, 'Width': 0.08, 'Color 1': (0.157, 0.424, 0.369, 1.0), 'Color 2': (0.522, 0.82, 0.671, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_CrackleNode',
                'props': {},
                'inputs': {'Scale': 7.66, 'Width': 0.08, 'Color 1': (0.157, 0.424, 0.369, 1.0), 'Color 2': (0.522, 0.82, 0.671, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.502,
            },
        ],
    },
    'B95_CARIBBEAN_RESORT': {
        'label': 'Caribbean Resort',
        'note': 'Bryce 1995: Randomized waves with strong transmission and high refraction index.  Reflectivity gets higher at low angles.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'WATER',
        'inputs': {
            'Diffuse Color': (0.027, 1.0, 0.922, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 0.635,
            'Specular Color': (0.359, 0.875, 1.0, 1.0),
            'Glossiness': 63.0,
            'Opacity': 0.502,
            'IOR': 1.31,
            'Refraction Amount': 1.0,
            'Reflection': 0.18,
            'Fresnel': 0.6,
            'Fresnel Power': 4.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WaterNode',
                'props': {},
                'inputs': {'Scale': 6.0, 'Color 1': (0.086, 0.722, 0.729, 1.0), 'Color 2': (0.0, 0.271, 0.463, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.113,
            },
        ],
    },
    'B95_DEEP_BLUE': {
        'label': 'Deep Blue',
        'note': 'Bryce 1995: Water highly transparent with "altitude sensitive" colors; blues get stronger as light penetrates deeper.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'WATER',
        'inputs': {
            'Diffuse Color': (0.115, 0.779, 0.824, 1.0),
            'Diffuse Level': 0.3,
            'Specular Level': 0.51,
            'Specular Color': (0.371, 0.714, 1.0, 1.0),
            'Glossiness': 84.0,
            'Reflection': 0.157,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
            'Opacity': 0.51,
            'IOR': 1.16,
            'Refraction Amount': 1.0,
        },
        'textures': [
            {
                'node': 'HALCYON_WaterNode',
                'props': {},
                'inputs': {'Scale': 6.0, 'Color 1': (0.086, 0.722, 0.729, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.113,
            },
        ],
    },
    'B95_MERCURY_SURFACE': {
        'label': 'Mercury Surface',
        'note': 'Bryce 1995: Little transparency, but heavy reflection.',
        'model': 'PHONG',
        'category': 'SIMPLE',
        'family': 'LIQUID',
        'inputs': {
            'Diffuse Color': (0.988, 0.988, 0.988, 1.0),
            'Diffuse Level': 0.953,
            'Ambient': 0.254,
            'Specular Level': 1.0,
            'Specular Color': (0.591, 0.7, 0.988, 1.0),
            'Glossiness': 68.0,
            'Reflection': 0.773,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
        },
        'textures': [
            {
                'node': 'HALCYON_WaterNode',
                'props': {},
                'inputs': {'Scale': 1.5, 'Color 1': (0.275, 0.627, 0.682, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.288,
            },
        ],
    },
    'B95_BRYCE_COLA': {
        'label': 'Bryce Cola',
        'note': 'From the Bryce 1995 library',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'LIQUID',
        'inputs': {
            'Diffuse Color': (0.11, 0.055, 0.0, 1.0),
            'Diffuse Level': 0.339,
            'Ambient': 0.203,
            'Specular Level': 0.51,
            'Specular Color': (1.0, 0.334, 0.0, 1.0),
            'Glossiness': 810.0,
            'Reflection': 0.068,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
            'Opacity': 0.712,
        },
        'textures': [
            {
                'node': 'HALCYON_WaterNode',
                'props': {},
                'inputs': {'Scale': 6.0, 'Color 1': (0.0, 0.271, 0.463, 1.0), 'Color 2': (0.0, 0.0, 0.0, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.113,
            },
        ],
    },
    'B95_GLOWING_WATER': {
        'label': 'Glowing Water',
        'note': 'Bryce 1995: From the labs at Area 51.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'LIQUID',
        'inputs': {
            'Diffuse Color': (1.0, 0.996, 0.996, 1.0),
            'Diffuse Level': 0.678,
            'Ambient': 0.075,
            'Specular Level': 0.604,
            'Specular Color': (1.0, 0.996, 0.996, 1.0),
            'Glossiness': 35.0,
            'Reflection': 0.847,
            'Reflection Color': (1.0, 1.0, 1.0, 1.0),
            'Opacity': 0.455,
        },
        'textures': [
            {
                'node': 'HALCYON_WaterNode',
                'props': {},
                'inputs': {'Scale': 6.0, 'Color 1': (0.855, 0.745, 0.678, 1.0), 'Color 2': (1.0, 1.0, 0.0, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.383,
            },
        ],
    },
    'B95_SUMMER_CLOUDS': {
        'label': 'Summer Clouds',
        'note': 'Bryce 1995: Reflective quality makes sky brighter...',
        'model': 'TRANSLUCENT',
        'category': 'ADVANCED',
        'family': 'CLOUD',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 1.2,
            'Specular Level': 0.0,
            'Translucency': 0.75,
            'Edge Opacity': 0.15,
        },
        'textures': [
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.76, 'Turbulence': 0.6, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.102, 0.216, 0.894, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.76, 'Turbulence': 0.6, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.102, 0.216, 0.894, 1.0)},
                'output': 'Fac',
                'target': 'Opacity',
            },
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.76, 'Turbulence': 0.6, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.102, 0.216, 0.894, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.056,
            },
        ],
    },
    'B95_WISPY_AFTERNOON': {
        'label': 'Wispy Afternoon',
        'note': 'Bryce 1995: Good layer for simple summer images.',
        'model': 'TRANSLUCENT',
        'category': 'ADVANCED',
        'family': 'CLOUD',
        'inputs': {
            'Diffuse Color': (1.0, 1.0, 1.0, 1.0),
            'Diffuse Level': 0.478,
            'Ambient': 1.2,
            'Specular Level': 0.0,
            'Translucency': 0.75,
            'Edge Opacity': 0.15,
        },
        'textures': [
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (0.612, 0.812, 0.855, 1.0), 'Color 2': (0.333, 0.533, 0.749, 1.0)},
                'output': 'Fac',
                'target': 'Opacity',
            },
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (0.612, 0.812, 0.855, 1.0), 'Color 2': (0.333, 0.533, 0.749, 1.0)},
                'target': 'Self-Illumination',
            },
        ],
    },
    'B95_SMOKE_STACK': {
        'label': 'Smoke Stack',
        'note': 'Bryce 1995: Volumetric texture... Wrap around sphere rather than assigning to a plane for 3D cloud "objects".',
        'model': 'TRANSLUCENT',
        'category': 'ADVANCED',
        'family': 'CLOUD',
        'inputs': {
            'Diffuse Level': 0.588,
            'Ambient': 0.508,
            'Specular Level': 1.0,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 10.0,
            'Translucency': 0.75,
            'Edge Opacity': 0.15,
        },
        'textures': [
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.698, 0.667, 0.525, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.698, 0.667, 0.525, 1.0)},
                'output': 'Fac',
                'target': 'Opacity',
            },
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.698, 0.667, 0.525, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.056,
            },
        ],
    },
    'B95_NIGHT_CLOUDS': {
        'label': 'Night Clouds',
        'note': 'Bryce 1995: Low specular coefficients and darker colors let these work well in night scenes.',
        'model': 'TRANSLUCENT',
        'category': 'ADVANCED',
        'family': 'CLOUD',
        'inputs': {
            'Diffuse Level': 0.3,
            'Ambient': 0.659,
            'Specular Level': 0.29,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 172.0,
            'Translucency': 0.75,
            'Edge Opacity': 0.15,
        },
        'textures': [
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 3},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (0.502, 1.0, 1.0, 1.0), 'Color 2': (0.0, 0.141, 0.145, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_BozoNode',
                'props': {'octaves': 3},
                'inputs': {'Scale': 1.2, 'Turbulence': 0.6, 'Color 1': (0.502, 1.0, 1.0, 1.0), 'Color 2': (0.0, 0.141, 0.145, 1.0)},
                'output': 'Fac',
                'target': 'Opacity',
            },
        ],
    },
    'B95_ARIZONA': {
        'label': 'Arizona',
        'note': 'Bryce 1995: Single component procedural texture with slope sensitivity... Good desert material.',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'TERRAIN',
        'inputs': {
            'Diffuse Level': 0.953,
            'Ambient': 0.089,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.627, 0.518, 0.337, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (1.0, 1.0, 1.0, 1.0), 'Color 2': (0.627, 0.518, 0.337, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.584,
            },
        ],
    },
    'B95_GRASSY_PEAKS': {
        'label': 'Grassy Peaks',
        'note': 'Bryce 1995: Single channel texture, altitude dependent.',
        'model': 'LAMBERT',
        'category': 'ADVANCED',
        'family': 'TERRAIN',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.235,
            'Specular Level': 0.0,
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (0.827, 0.667, 0.537, 1.0), 'Color 2': (0.251, 0.447, 0.165, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 3.0, 'Contrast': 1.2, 'Color 1': (0.827, 0.667, 0.537, 1.0), 'Color 2': (0.251, 0.447, 0.165, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.584,
            },
        ],
    },
    'B95_FIRST_SNOW': {
        'label': 'First Snow',
        'note': 'Bryce 1995: Single channel procedural texture... Very slope and altitude sensitive, occurence of snow increases as slope decreases.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'TERRAIN',
        'inputs': {
            'Diffuse Level': 0.922,
            'Ambient': 0.075,
            'Specular Level': 0.271,
            'Specular Color': (0.686, 0.686, 0.641, 1.0),
            'Glossiness': 38.0,
        },
        'textures': [
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 8.0, 'Contrast': 1.2, 'Color 1': (0.129, 0.055, 0.102, 1.0), 'Color 2': (0.045, 0.019, 0.036, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_GraniteNode',
                'props': {'octaves': 6},
                'inputs': {'Scale': 8.0, 'Contrast': 1.2, 'Color 1': (0.129, 0.055, 0.102, 1.0), 'Color 2': (0.045, 0.019, 0.036, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.822,
            },
        ],
    },
    'B95_GRAND_CANYON': {
        'label': 'Grand Canyon',
        'note': 'Bryce 1995: Brown and orange layered rocky surface.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'TERRAIN',
        'inputs': {
            'Diffuse Level': 1.0,
            'Ambient': 0.141,
            'Specular Level': 0.106,
            'Specular Color': (1.0, 1.0, 1.0, 1.0),
            'Glossiness': 35.0,
        },
        'textures': [
            {
                'node': 'HALCYON_AgateNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 1.5, 'Turbulence': 0.5, 'Bands': 6.0, 'Color 1': (0.784, 0.639, 0.427, 1.0), 'Color 2': (0.765, 0.463, 0.365, 1.0)},
                'target': 'Diffuse Color',
            },
            {
                'node': 'HALCYON_AgateNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 1.5, 'Turbulence': 0.5, 'Bands': 6.0, 'Color 1': (0.784, 0.639, 0.427, 1.0), 'Color 2': (0.765, 0.463, 0.365, 1.0)},
                'output': 'Fac',
                'target': 'Normal',
                'bump': 0.759,
            },
        ],
    },
    'B95_VOLCANO_HEART': {
        'label': 'Volcano Heart',
        'note': 'Bryce 1995: Lava with dark rock increasing with slope.',
        'model': 'PHONG',
        'category': 'ADVANCED',
        'family': 'TERRAIN',
        'inputs': {
            'Diffuse Color': (1.0, 0.839, 0.694, 1.0),
            'Diffuse Level': 0.0,
            'Ambient': 1.162,
            'Specular Level': 0.831,
            'Specular Color': (0.38, 0.278, 0.169, 1.0),
            'Glossiness': 21.0,
        },
        'textures': [
            {
                'node': 'HALCYON_MarbleNode',
                'props': {'octaves': 4},
                'inputs': {'Scale': 3.5, 'Turbulence': 1.6, 'Veins': 3.0, 'Color 1': (0.0, 0.0, 0.0, 1.0), 'Color 2': (1.0, 0.0, 0.0, 1.0)},
                'target': 'Self-Illumination',
            },
        ],
    },

}


def _shader_input(shader, key):
    """A shader input by IDENTIFIER first, then display name.

    The BI Material node shows BI's labels (Hardness, Refr, Alpha)
    over master-compatible identifiers (Glossiness, IOR, Opacity);
    specs and texture targets speak identifiers, so look there first.
    """
    for s in shader.inputs:
        if getattr(s, 'identifier', None) == key:
            return s
    return shader.inputs.get(key)


def _set_socket(sock, value):
    if sock is None or not hasattr(sock, 'default_value'):
        return
    try:
        if hasattr(sock.default_value, '__len__') and hasattr(value,
                                                              '__len__'):
            n = min(len(sock.default_value), len(value))
            for i in range(n):
                sock.default_value[i] = value[i]
        elif hasattr(sock.default_value, '__len__'):
            for i in range(len(sock.default_value)):
                sock.default_value[i] = float(value)
        else:
            sock.default_value = value if not isinstance(value,
                                                         (int, float)) \
                else float(value)
    except (TypeError, ValueError):
        pass


def build_spec(mat, spec, clear=True, offset=(0.0, 0.0), select=False):
    """Build a spec-described node cluster into a material's tree.

    `clear=True` is the original behaviour (the legacy importer's):
    the tree is REPLACED. R203: `clear=False` is the Add-menu road --
    the cluster is INSERTED beside whatever is already there, nothing
    existing is touched or re-linked, the new nodes arrive selected
    (old selection dropped) and shifted by `offset`, and the cluster
    reaches the material output ONLY if the active output's Surface
    was sitting unlinked. Clicking a Pre-Made must never eat the
    shader someone was working on.

    The spec is the TEMPLATES shape: {'model', 'inputs', 'textures'}.
    Texture entries take the documented keys plus the extended set the
    legacy importer emits: 'coords' (UV/Generated/Object via a Texture
    Coordinate node), 'mapping' ({'location','scale'} through a Mapping
    node -- offset and size the way Blender Internal applied them),
    'mix' ({'fac','blend','base'}: a MixRGB against the base colour, the
    slot's own influence factor as the mix factor), 'scale_fac' (a Math
    multiply for value channels under full influence), and 'invert'
    (the old negative-influence toggle).
    """
    from . import compat as _compat
    if not _compat.uses_nodes(mat):
        # the WRITE is not deprecated (and is required in 5.x to turn
        # a flat legacy material into a node one); only the read warns
        mat.use_nodes = True
    tree = mat.node_tree
    pre = set()
    existing_out = None
    if clear:
        tree.nodes.clear()
    else:
        pre = {n.name for n in tree.nodes}
        for n in tree.nodes:
            if n.bl_idname == 'ShaderNodeOutputMaterial' and (
                    existing_out is None
                    or getattr(n, 'is_active_output', False)):
                existing_out = n

    if existing_out is not None:
        out = existing_out
    else:
        out = tree.nodes.new('ShaderNodeOutputMaterial')
        out.location = (320, 0)
    bi = spec.get('bi')
    if bi:
        # the BI Material node: the whole 2.79 panel, 1:1. Props first
        # (stops into their collections), THEN refresh_sockets so the
        # panel toggles decide visibility, THEN the socket values.
        shader = tree.nodes.new('HALCYON_BIMaterialNode')
        shader.location = (0, 0)
        for k, v in bi.items():
            if k in ('ramp_dif_stops', 'ramp_spec_stops'):
                which = 'dif' if k == 'ramp_dif_stops' else 'spec'
                ipo = bi.get(f'ramp_{which}_ipo', 0)
                if hasattr(shader, 'set_ramp_stops'):
                    # populates the gradient widget AND the fallback
                    # rows in one go
                    shader.set_ramp_stops(which, v, ipo)
                    continue
                coll = shader.dif_stops if k == 'ramp_dif_stops' \
                    else shader.spec_stops
                try:
                    coll.clear()
                    for stop in v:
                        s = coll.add()
                        s.position = float(stop[0])
                        s.color = (float(stop[1]), float(stop[2]),
                                   float(stop[3]), float(stop[4]))
                except Exception:                               # noqa: BLE001
                    pass
                continue
            if k == 'ramp_dif_ipo' or k == 'ramp_spec_ipo':
                # the spec carries do_colorband's integer; the node
                # prop is the enum
                ipo_names = ('LINEAR', 'EASE', 'B_SPLINE', 'CARDINAL',
                             'CONSTANT')
                try:
                    setattr(shader, k, ipo_names[int(v)]
                            if 0 <= int(v) < 5 else 'LINEAR')
                except (TypeError, ValueError):
                    pass
                continue
            if k == 'light_group_lights':
                continue        # export re-resolves from the collection
            try:
                setattr(shader, k, v)
            except (TypeError, ValueError):
                pass
        try:
            shader.refresh_sockets()
        except Exception:                                       # noqa: BLE001
            pass
    else:
        shader = tree.nodes.new('HALCYON_ShaderNode')
        shader.location = (0, 0)
        shader.model = spec['model']

    for name, value in spec.get('inputs', {}).items():
        _set_socket(_shader_input(shader, name), value)

    y = 200
    coord_node = None
    matcap_nodes = {}       # MATCAP_NORMAL / MATCAP_REFLECT -> node
    uvmap_nodes = {}        # named UV layer -> its UV Map node
    last_link = {}          # target socket name -> last source socket
    for entry in spec.get('textures', []):
        if not isinstance(entry, dict):
            idname, props, inputs, target = entry
            entry = {'node': idname, 'props': props, 'inputs': inputs,
                     'target': target}
        try:
            node = tree.nodes.new(entry['node'])
        except Exception:                                       # noqa: BLE001
            continue
        node.location = (-320, y)
        for k, v in entry.get('props', {}).items():
            if k == 'stops' and hasattr(node, 'stops'):
                # a colorband: (pos, r, g, b, a) tuples into the node's
                # own stop collection
                try:
                    node.stops.clear()
                    for stop in v:
                        s = node.stops.add()
                        s.position = float(stop[0])
                        s.color = (float(stop[1]), float(stop[2]),
                                   float(stop[3]), float(stop[4]))
                except (TypeError, ValueError, AttributeError):
                    pass
                continue
            if hasattr(node, k):
                try:
                    setattr(node, k, v)
                except (TypeError, ValueError):
                    pass
        for k, v in entry.get('inputs', {}).items():
            _set_socket(node.inputs.get(k), v)

        # ---- coordinate plumbing: TexCoord (shared) -> Mapping -> Vector
        vec_dst = node.inputs.get('Vector')
        coords = entry.get('coords')
        uv_layer = entry.get('uv_layer')
        if vec_dst is not None and coords == 'UV' and uv_layer:
            # the slot names its UV layer: a UV Map node (shared per
            # name) instead of the TexCoord's active-layer output --
            # through the Mapping fold when the slot carries one
            un = uvmap_nodes.get(uv_layer)
            if un is None:
                try:
                    un = tree.nodes.new('ShaderNodeUVMap')
                    un.location = (-880, -500 - 120 * len(uvmap_nodes))
                    un.uv_map = uv_layer
                    uvmap_nodes[uv_layer] = un
                except Exception:                               # noqa: BLE001
                    un = None
            if un is not None:
                try:
                    mp = entry.get('mapping')
                    if mp:
                        mnode = tree.nodes.new('ShaderNodeMapping')
                        mnode.location = (-640, y)
                        _set_socket(mnode.inputs.get('Location'),
                                    mp.get('location', (0, 0, 0)))
                        _set_socket(mnode.inputs.get('Scale'),
                                    mp.get('scale', (1, 1, 1)))
                        tree.links.new(un.outputs['UV'],
                                       mnode.inputs['Vector'])
                        tree.links.new(mnode.outputs['Vector'], vec_dst)
                    else:
                        tree.links.new(un.outputs['UV'], vec_dst)
                except Exception:                               # noqa: BLE001
                    pass
        elif vec_dst is not None and isinstance(coords, str) and \
                coords.startswith('MATCAP'):
            # BI's Nor and Refl coords: the Matcap Coordinates node,
            # one per source kind, shared across slots like TexCoord
            key = coords
            mnode = matcap_nodes.get(key)
            if mnode is None:
                try:
                    mnode = tree.nodes.new('HALCYON_MatcapUVNode')
                    mnode.location = (-880, -260 if key.endswith(
                        'REFLECT') else -380)
                    mnode.source = 'REFLECTION' \
                        if key == 'MATCAP_REFLECT' else 'NORMAL'
                    matcap_nodes[key] = mnode
                except Exception:                               # noqa: BLE001
                    mnode = None
            if mnode is not None:
                try:
                    mp = entry.get('mapping')
                    if mp:
                        # the slot's Offset/Size window applies to REFL
                        # coords too -- texco_mapping runs the same
                        # size*(uv-0.5)+ofs+0.5 fold whatever the
                        # coordinate source (the Mask's env map lived
                        # in a 0.7x0.8 window at +1.15 the import
                        # never applied)
                        mpn = tree.nodes.new('ShaderNodeMapping')
                        mpn.location = (-640, y)
                        _set_socket(mpn.inputs.get('Location'),
                                    mp.get('location', (0, 0, 0)))
                        _set_socket(mpn.inputs.get('Scale'),
                                    mp.get('scale', (1, 1, 1)))
                        tree.links.new(mnode.outputs['Vector'],
                                       mpn.inputs['Vector'])
                        tree.links.new(mpn.outputs['Vector'], vec_dst)
                    else:
                        tree.links.new(mnode.outputs['Vector'], vec_dst)
                except Exception:                               # noqa: BLE001
                    pass
        elif vec_dst is not None and (coords or entry.get('mapping')):
            if coord_node is None:
                coord_node = tree.nodes.new('ShaderNodeTexCoord')
                coord_node.location = (-880, 0)
            src_out = coord_node.outputs.get(coords or 'Generated')
            if src_out is not None:
                try:
                    mp = entry.get('mapping')
                    if mp:
                        mnode = tree.nodes.new('ShaderNodeMapping')
                        mnode.location = (-640, y)
                        _set_socket(mnode.inputs.get('Location'),
                                    mp.get('location', (0, 0, 0)))
                        _set_socket(mnode.inputs.get('Scale'),
                                    mp.get('scale', (1, 1, 1)))
                        tree.links.new(src_out, mnode.inputs['Vector'])
                        tree.links.new(mnode.outputs['Vector'], vec_dst)
                    else:
                        tree.links.new(src_out, vec_dst)
                except Exception:                               # noqa: BLE001
                    pass

        src = None
        want = entry.get('output')
        if want is not None:
            src = node.outputs.get(want)
        if src is None and node.outputs:
            src = node.outputs[0]
        dst = _shader_input(shader, entry['target'])
        if dst is None or src is None:
            y -= 260
            continue
        try:
            if entry.get('invert'):
                inv = tree.nodes.new('ShaderNodeInvert')
                inv.location = (-200, y - 60)
                tree.links.new(src, inv.inputs['Color'])
                src = inv.outputs['Color']
            mix = entry.get('mix')
            vb = entry.get('vblend')
            rgbb = entry.get('rgbblend')
            scale_fac = entry.get('scale_fac')
            strength = entry.get('bump')

            def _wire_slot_flags(dst_node, spec_d):
                """The MTex texflag props + the Color/Alpha feeds an
                RGB-yielding texture needs (band clouds, images). The
                Alpha wire follows BI's alpha law: only a texture that
                actually EXPOSES alpha (a colorband, or an image with
                Use Alpha) feeds it; otherwise the input stays at its
                default 1.0, exactly imagewrap's ta = 1."""
                for k in ('tex_rgb', 'rgbtoint', 'negative', 'alphamix',
                          'map_alpha', 'calc_alpha', 'neg_alpha'):
                    if k in spec_d and hasattr(dst_node, k):
                        try:
                            setattr(dst_node, k, bool(spec_d[k]))
                        except (TypeError, ValueError):
                            pass
                if spec_d.get('tex_rgb'):
                    csrc = node.outputs.get('Color')
                    cdst = dst_node.inputs.get('Color')
                    if csrc is not None and cdst is not None:
                        tree.links.new(csrc, cdst)
                    if spec_d.get('img_alpha', True):
                        asrc = node.outputs.get('Alpha')
                        adst = dst_node.inputs.get('Alpha')
                        if asrc is not None and adst is not None:
                            tree.links.new(asrc, adst)

            if rgbb is not None:
                # a BI colour channel: texture_rgb_blend as the BI
                # Color Influence node -- the texture supplies tcol
                # and a per-pixel factor when it yields RGB, only the
                # factor when it does not (the SLOT colour is tcol
                # then), and Base chains slot to slot
                cn = tree.nodes.new('HALCYON_BIRGBBlendNode')
                cn.location = (-140, y)
                try:
                    cn.blend = rgbb.get('blend', 'MIX')
                except (TypeError, ValueError):
                    pass
                _wire_slot_flags(cn, rgbb)
                prev = last_link.get(entry['target'])
                if prev is not None:
                    tree.links.new(prev, cn.inputs['Base'])
                else:
                    _set_socket(cn.inputs.get('Base'),
                                rgbb.get('base', (0.8, 0.8, 0.8, 1.0)))
                fsrc = node.outputs.get('Fac')
                if fsrc is not None:
                    tree.links.new(fsrc, cn.inputs['Intensity'])
                elif not rgbb.get('tex_rgb'):
                    tree.links.new(src, cn.inputs['Intensity'])
                _set_socket(cn.inputs.get('Factor'),
                            float(rgbb.get('factor', 1.0)))
                _set_socket(cn.inputs.get('Slot Color'),
                            tuple(rgbb.get('slot_color', (1, 0, 1)))
                            + (1.0,))
                tree.links.new(cn.outputs['Color'], dst)
                last_link[entry['target']] = cn.outputs['Color']
            elif vb is not None:
                # a BI value channel: texture_value_blend as the BI
                # Influence node -- Base chains slot to slot in the
                # channel's own units; hardness scales /128 in, x128
                # out with BI's 1..511 clamp
                infl = tree.nodes.new('HALCYON_BIInfluenceNode')
                infl.location = (-140, y)
                try:
                    infl.blend = vb.get('blend', 'MIX')
                except (TypeError, ValueError):
                    pass
                _wire_slot_flags(infl, vb)
                prev = last_link.get(entry['target'])
                if prev is not None:
                    tree.links.new(prev, infl.inputs['Base'])
                else:
                    _set_socket(infl.inputs.get('Base'),
                                float(vb.get('base', 0.0)))
                tree.links.new(src, infl.inputs['Intensity'])
                _set_socket(infl.inputs.get('Factor'),
                            float(vb.get('factor', 1.0)))
                _set_socket(infl.inputs.get('DVar'),
                            float(vb.get('dvar', 1.0)))
                out_sock = infl.outputs['Value']
                last_link[entry['target']] = out_sock
                scale = float(vb.get('scale', 1.0))
                if abs(scale - 1.0) > 1e-9:
                    mul = tree.nodes.new('ShaderNodeMath')
                    mul.location = (-70, y)
                    try:
                        mul.operation = 'MULTIPLY'
                    except (TypeError, ValueError):
                        pass
                    _set_socket(mul.inputs[1], scale)
                    tree.links.new(out_sock, mul.inputs[0])
                    out_sock = mul.outputs[0]
                cl = vb.get('clamp')
                if cl:
                    for op, val in (('MAXIMUM', cl[0]), ('MINIMUM',
                                                         cl[1])):
                        mnode2 = tree.nodes.new('ShaderNodeMath')
                        mnode2.location = (-30, y)
                        try:
                            mnode2.operation = op
                        except (TypeError, ValueError):
                            pass
                        _set_socket(mnode2.inputs[1], float(val))
                        tree.links.new(out_sock, mnode2.inputs[0])
                        out_sock = mnode2.outputs[0]
                tree.links.new(out_sock, dst)
            elif mix is not None:
                mixn = tree.nodes.new('ShaderNodeMixRGB')
                mixn.location = (-140, y)
                if hasattr(mixn, 'blend_type'):
                    try:
                        mixn.blend_type = mix.get('blend', 'MIX')
                    except (TypeError, ValueError):
                        pass
                _set_socket(mixn.inputs.get('Fac'), mix.get('fac', 1.0))
                prev = last_link.get(entry['target'])
                if prev is not None:
                    # BI stacks texture slots: each blends ONTO the
                    # previous slot's result. Setting Color1 to the
                    # static base here instead used to RE-link the
                    # shader input, and Blender keeps only the last
                    # link -- every earlier slot's chain went dark
                    # (the field's 'mixed textures aren't plugged in')
                    tree.links.new(prev, mixn.inputs['Color1'])
                else:
                    _set_socket(mixn.inputs.get('Color1'),
                                mix.get('base', (0.8, 0.8, 0.8, 1.0)))
                tree.links.new(src, mixn.inputs['Color2'])
                tree.links.new(mixn.outputs['Color'], dst)
                last_link[entry['target']] = mixn.outputs['Color']
            elif scale_fac is not None:
                mul = tree.nodes.new('ShaderNodeMath')
                mul.location = (-140, y)
                try:
                    mul.operation = 'MULTIPLY'
                except (TypeError, ValueError):
                    pass
                _set_socket(mul.inputs[1], float(scale_fac))
                tree.links.new(src, mul.inputs[0])
                src2 = mul.outputs[0]
                prev = last_link.get(entry['target'])
                if prev is not None:
                    # a second contribution to the same scalar input
                    # (several bump slots): sum them, as BI's influence
                    # stack summed its normal perturbations
                    add = tree.nodes.new('ShaderNodeMath')
                    add.location = (-70, y)
                    try:
                        add.operation = 'ADD'
                    except (TypeError, ValueError):
                        pass
                    tree.links.new(prev, add.inputs[0])
                    tree.links.new(src2, add.inputs[1])
                    src2 = add.outputs[0]
                tree.links.new(src2, dst)
                last_link[entry['target']] = src2
            elif strength is not None:
                # the proven anatomy: height -> Bump -> normal. The engine
                # renders the height chain to a pre-pass and differences it
                # the CPU's own way, so this travels to the GPU exactly
                bump = tree.nodes.new('ShaderNodeBump')
                bump.location = (-140, node.location[1])
                s = bump.inputs.get('Strength')
                if s is not None:
                    s.default_value = float(strength)
                tree.links.new(src, bump.inputs['Height'])
                tree.links.new(bump.outputs['Normal'], dst)
            else:
                prev = last_link.get(entry['target'])
                if prev is not None and entry.get('style') != 'color':
                    add = tree.nodes.new('ShaderNodeMath')
                    add.location = (-70, y)
                    try:
                        add.operation = 'ADD'
                    except (TypeError, ValueError):
                        pass
                    tree.links.new(prev, add.inputs[0])
                    tree.links.new(src, add.inputs[1])
                    src = add.outputs[0]
                # a colour slot lands here only at full influence with
                # MIX blending, where BI's result IS the texture -- it
                # replaces the running chain rather than adding to it
                tree.links.new(src, dst)
                last_link[entry['target']] = src
        except Exception:                                       # noqa: BLE001
            pass
        y -= 260

    surf_in = out.inputs.get('Surface') if out is not None else None
    if clear or (surf_in is not None and not surf_in.is_linked):
        # replacing, or dropping into an empty stage: take the output.
        # An occupied Surface is someone's work -- leave it alone
        try:
            tree.links.new(shader.outputs['Surface'], surf_in)
        except Exception:                                       # noqa: BLE001
            pass
    shader.refresh_sockets()
    mat.halcyon.use_override = False
    if not clear:
        dx, dy = float(offset[0]), float(offset[1])
        for n in tree.nodes:
            fresh = n.name not in pre
            if fresh and (dx or dy):
                try:
                    n.location = (n.location[0] + dx, n.location[1] + dy)
                except Exception:                               # noqa: BLE001
                    pass
            if select:
                try:
                    n.select = fresh
                except Exception:                               # noqa: BLE001
                    pass
    return shader


def build(mat, key):
    """Replace a material's tree with the named template."""
    spec = TEMPLATES.get(key)
    if spec is None:
        return False, f'unknown template {key!r}'
    build_spec(mat, spec)
    return True, f'{mat.name}: {spec["label"]}'


def category_keys(category):
    """Template keys in one category, sorted by label."""
    return sorted((k for k, v in TEMPLATES.items()
                   if v.get('category') == category),
                  key=lambda k: TEMPLATES[k]['label'])


#: R203: the Pre-Made families, one submenu each -- the material-kind
#: taxonomy the field asked for (Water, Metal, Mineral, Liquid, ...).
#: Order is display order.
FAMILIES = (
    ('METAL', "Metal"),
    ('MINERAL', "Mineral"),
    ('GLASS', "Glass"),
    ('WATER', "Water"),
    ('LIQUID', "Liquid"),
    ('WOOD', "Wood"),
    ('CLOUD', "Cloud & Fog"),
    ('TERRAIN', "Terrain"),
    ('SURFACES', "Surfaces"),
    ('EFFECTS', "Effects"),
)

#: family per template key, for the entries that predate the 'family'
#: field; new entries carry their own.
_FAMILY_OVERRIDES = {
    'CHROME': 'METAL', 'GOLD': 'METAL', 'BRUSHED': 'METAL',
    'HAMMERED': 'METAL',
    'MARBLE': 'MINERAL', 'CLAY': 'MINERAL', 'PORCELAIN': 'MINERAL',
    'BRICK_WALL': 'MINERAL', 'TILE_FLOOR': 'MINERAL',
    'GLASS': 'GLASS',
    'WATER': 'WATER',
    'LAVA': 'LIQUID',
    'TERRAIN': 'TERRAIN',
    'PLASTIC': 'SURFACES', 'RUBBER': 'SURFACES', 'VELVET': 'SURFACES',
    'SILK': 'SURFACES', 'CANDY': 'SURFACES', 'CAR_PAINT': 'SURFACES',
    'WOOD': 'WOOD', 'CLOTH': 'SURFACES', 'LEOPARD': 'SURFACES',
    'TOON': 'EFFECTS', 'GHOST': 'EFFECTS', 'HOLOGRAM': 'EFFECTS',
    'NEON': 'EFFECTS', 'WIREFRAME': 'EFFECTS',
    'DEAD_CHANNEL': 'EFFECTS',
}


def family_of(key):
    fam = TEMPLATES.get(key, {}).get('family')
    if fam:
        return fam
    fam = _FAMILY_OVERRIDES.get(key)
    if fam:
        return fam
    return 'SURFACES'


def family_keys(family):
    """Template keys in one Pre-Made family, sorted by label."""
    return sorted((k for k in TEMPLATES if family_of(k) == family),
                  key=lambda k: TEMPLATES[k]['label'])


def template_items(self=None, context=None):
    return [(k, v['label'], v['note']) for k, v in sorted(
        TEMPLATES.items(), key=lambda kv: kv[1]['label'])]


class HALCYON_OT_material_template(Operator):
    """R203: drop a ready-made node cluster into the material being
    edited -- added BESIDE what is there, never replacing it, and it
    only takes the output when the output was sitting empty"""

    bl_idname = 'halcyon.material_template'
    bl_label = "Add Pre-Made Material"
    bl_options = {'REGISTER', 'UNDO'}

    template: EnumProperty(name="Template", items=template_items)

    @classmethod
    def poll(cls, context):
        return getattr(context, 'material', None) is not None

    def execute(self, context):
        # no dialog, no rebuild: the Add menu should behave like the
        # Add menu -- click, and the nodes are simply THERE
        mat = context.material
        spec = TEMPLATES.get(self.template)
        if spec is None:
            self.report({'ERROR'}, f'unknown template {self.template!r}')
            return {'CANCELLED'}
        # place the cluster clear of the existing nodes: just right
        # of the rightmost one (fresh empty trees keep the origin)
        dx = dy = 0.0
        tree = getattr(mat, 'node_tree', None)
        if tree is not None and len(getattr(tree, 'nodes', ())):
            try:
                dx = max(n.location[0] for n in tree.nodes) + 640.0
                dy = 0.0
            except Exception:                                   # noqa: BLE001
                dx = 640.0
        try:
            build_spec(mat, spec, clear=False, offset=(dx, dy),
                       select=True)
        except Exception as exc:                                # noqa: BLE001
            self.report({'ERROR'}, f'{spec["label"]}: {exc}')
            return {'CANCELLED'}
        self.report({'INFO'},
                    f"Added '{spec['label']}' beside the existing "
                    "nodes; it takes the output only if the output "
                    "was empty")
        return {'FINISHED'}


class HALCYON_MT_material_templates(Menu):
    """The template shelf, grouped: plain surface recipes first, then the
    ones that wire the engine's own textures in."""

    bl_idname = 'HALCYON_MT_material_templates'
    bl_label = "Material Templates"

    def draw(self, context):
        layout = self.layout
        for cat, heading in (('SIMPLE', "Simple"), ('ADVANCED', "Advanced")):
            layout.label(text=heading)
            for key in category_keys(cat):
                op = layout.operator('halcyon.material_template',
                                     text=TEMPLATES[key]['label'])
                op.template = key
            if cat == 'SIMPLE':
                layout.separator()


# ---------------------------------------------- R203: the Pre-Made menu
#
# Home: INSIDE the Shader Editor's Halcyon menu (Add > Halcyon >
# Pre-Made), one submenu per material family. Clicking an entry drops
# the node cluster in beside the existing nodes -- no dialog, nothing
# replaced.


def _family_menu(fam, label):
    def draw(self, context):
        for key in family_keys(fam):
            op = self.layout.operator('halcyon.material_template',
                                      text=TEMPLATES[key]['label'])
            op.template = key
    return type(f'NODE_MT_halcyon_premade_{fam.lower()}', (Menu,), {
        'bl_idname': f'NODE_MT_halcyon_premade_{fam.lower()}',
        'bl_label': label,
        'draw': draw,
    })


_FAMILY_MENUS = tuple(_family_menu(f, lbl) for f, lbl in FAMILIES)


class NODE_MT_halcyon_premade(Menu):
    """Ready-made materials by family -- the parsed Bryce 1995
    library and the engine's own recipes, dropped in beside whatever
    the tree already holds."""

    bl_idname = 'NODE_MT_halcyon_premade'
    bl_label = "Pre-Made"

    def draw(self, context):
        for cls in _FAMILY_MENUS:
            fam = cls.bl_idname.rsplit('_', 1)[-1].upper()
            if family_keys(fam):
                self.layout.menu(cls.bl_idname)


CLASSES = (HALCYON_OT_material_template, HALCYON_MT_material_templates,
           ) + _FAMILY_MENUS + (NODE_MT_halcyon_premade,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
