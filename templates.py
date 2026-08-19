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
                   'Diffuse Level': 0.05, 'Specular Level': 1.0,
                   'Glossiness': 400.0, 'Metalness': 1.0,
                   'Reflection': 0.85, 'Fresnel': 1.2, 'Fresnel Power': 4.0},
    },
    'GOLD': {
        'label': "Gold",
        'category': 'SIMPLE',
        'note': "The highlight takes the base colour rather than the light's, "
                "which is what keeps gold gold in its own reflection",
        'model': 'METAL',
        'inputs': {'Diffuse Color': (1.0, 0.77, 0.34, 1.0),
                   'Diffuse Level': 0.15, 'Specular Color': (1.0, 0.85, 0.5, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 220.0,
                   'Metalness': 1.0, 'Reflection': 0.6, 'Fresnel': 0.6},
    },
    'GLASS': {
        'label': "Glass",
        'category': 'SIMPLE',
        'note': "Thin in the middle and thick at the silhouette, which is how "
                "glass reads without refracting anything",
        'model': 'BLINN',
        'inputs': {'Diffuse Color': (0.85, 0.90, 0.92, 1.0),
                   'Diffuse Level': 0.1, 'Specular Level': 1.0,
                   'Glossiness': 300.0, 'Opacity': 0.12, 'Edge Opacity': 0.9,
                   'IOR': 1.52, 'Fresnel': 1.5, 'Fresnel Power': 3.0,
                   'Reflection': 0.4},
    },
    'PLASTIC': {
        'label': "Shiny Plastic",
        'category': 'SIMPLE',
        'note': "The 1990s default: a tight white highlight over flat colour",
        'model': 'PHONG',
        'inputs': {'Diffuse Color': (0.75, 0.15, 0.12, 1.0),
                   'Specular Level': 0.85, 'Glossiness': 60.0,
                   'Fresnel': 0.4},
    },
    'RUBBER': {
        'label': "Rubber",
        'category': 'SIMPLE',
        'note': "Oren-Nayar keeps the edges bright, which is what stops matte "
                "surfaces looking like flat paint",
        'model': 'OREN_NAYAR',
        'inputs': {'Diffuse Color': (0.08, 0.08, 0.09, 1.0),
                   'Roughness': 0.9, 'Specular Level': 0.08,
                   'Glossiness': 8.0},
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
        'inputs': {'Diffuse Color': (0.45, 0.05, 0.15, 1.0),
                   'Roughness': 0.8, 'Specular Level': 0.15,
                   'Rim Amount': 0.5, 'Rim Light': (0.9, 0.5, 0.6, 1.0)},
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
        'inputs': {'Diffuse Color': (0.93, 0.93, 0.90, 1.0),
                   'Specular Level': 0.9, 'Glossiness': 260.0,
                   'Fresnel': 0.8, 'Fresnel Power': 3.0},
    },
    'CANDY': {
        'label': "Candy Apple",
        'category': 'SIMPLE',
        'note': "Saturated colour under a wet coat: high gloss plus a little "
                "reflection. The show-car paint of every 90s logo spin",
        'model': 'PHONG',
        'inputs': {'Diffuse Color': (0.80, 0.05, 0.05, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 140.0,
                   'Reflection': 0.25, 'Fresnel': 0.6},
    },
    'CLAY': {
        'label': "Terracotta Clay",
        'category': 'SIMPLE',
        'note': "Fired earth: rough diffuse, almost no highlight. What matte "
                "test renders looked like before everything went shiny",
        'model': 'OREN_NAYAR',
        'inputs': {'Diffuse Color': (0.62, 0.35, 0.24, 1.0),
                   'Roughness': 0.7, 'Specular Level': 0.04,
                   'Glossiness': 6.0},
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
        'inputs': {'Diffuse Color': (0.45, 0.02, 0.05, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 180.0,
                   'Reflection': 0.35, 'Fresnel': 1.0, 'Fresnel Power': 4.0,
                   'Fresnel Color': (0.9, 0.5, 0.3, 1.0)},
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
                   'Specular Level': 0.9, 'Glossiness': 90.0,
                   'Anisotropy': 0.8, 'Metalness': 0.9, 'Fresnel': 0.5},
        'textures': [('HALCYON_ScratchesNode', {'count': 24},
                      {'Scale': 6.0, 'Width': 0.01}, 'Anisotropic Rotation')],
    },
    'MARBLE': {
        'label': "Polished Marble",
        'category': 'ADVANCED',
        'note': "Solid marble with veins running through the object, not "
                "wrapped around it",
        'model': 'BLINN',
        'inputs': {'Specular Level': 0.7, 'Glossiness': 180.0, 'Fresnel': 0.5},
        'textures': [('HALCYON_MarbleNode', {'octaves': 6},
                      {'Scale': 3.0, 'Turbulence': 1.2}, 'Diffuse Color')],
    },
    'WOOD': {
        'label': "Varnished Wood",
        'category': 'ADVANCED',
        'note': "Growth rings turned about the object's own axis",
        'model': 'BLINN_PHONG',
        'inputs': {'Specular Level': 0.5, 'Glossiness': 70.0, 'Fresnel': 0.6},
        'textures': [('HALCYON_WoodNode', {'octaves': 4},
                      {'Scale': 2.0, 'Rings': 9.0, 'Turbulence': 0.4},
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


def build_spec(mat, spec):
    """Replace a material's tree with a spec-described one.

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
    tree.nodes.clear()

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

    tree.links.new(shader.outputs['Surface'], out.inputs['Surface'])
    shader.refresh_sockets()
    mat.halcyon.use_override = False
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


def template_items(self=None, context=None):
    return [(k, v['label'], v['note']) for k, v in sorted(
        TEMPLATES.items(), key=lambda kv: kv[1]['label'])]


class HALCYON_OT_material_template(Operator):
    """Rebuild this material from a ready-made Halcyon setup"""

    bl_idname = 'halcyon.material_template'
    bl_label = "Apply Material Template"
    bl_options = {'REGISTER', 'UNDO'}

    template: EnumProperty(name="Template", items=template_items)

    @classmethod
    def poll(cls, context):
        return getattr(context, 'material', None) is not None

    def execute(self, context):
        ok, msg = build(context.material, self.template)
        self.report({'INFO'} if ok else {'ERROR'}, msg)
        return {'FINISHED'} if ok else {'CANCELLED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


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


CLASSES = (HALCYON_OT_material_template, HALCYON_MT_material_templates)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
