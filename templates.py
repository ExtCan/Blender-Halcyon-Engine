"""Ready-made materials, built from the Halcyon shader and its own textures.

Each entry is a recipe rather than a saved blend file: the node tree is
constructed on demand, so a template always matches the current version of the
nodes instead of rotting into a set of sockets that no longer exist.

They are also the quickest way to see what the master shader's less obvious
inputs actually do — Fresnel on the chrome, the sun rim on the wax, edge opacity
on the glass.
"""

import bpy
from bpy.props import EnumProperty
from bpy.types import Operator

# (model, {socket: value}, [(texture node, {props}, {socket: value}, target)])
TEMPLATES = {
    'CHROME': {
        'label': "Chrome",
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
        'note': "The highlight takes the base colour rather than the light's, "
                "which is what keeps gold gold in its own reflection",
        'model': 'METAL',
        'inputs': {'Diffuse Color': (1.0, 0.77, 0.34, 1.0),
                   'Diffuse Level': 0.15, 'Specular Color': (1.0, 0.85, 0.5, 1.0),
                   'Specular Level': 1.0, 'Glossiness': 220.0,
                   'Metalness': 1.0, 'Reflection': 0.6, 'Fresnel': 0.6},
    },
    'BRUSHED': {
        'label': "Brushed Metal",
        'note': "Anisotropic highlight stretched by a Scratches texture "
                "driving the rotation",
        'model': 'ANISOTROPIC',
        'inputs': {'Diffuse Color': (0.42, 0.44, 0.47, 1.0),
                   'Specular Level': 0.9, 'Glossiness': 90.0,
                   'Anisotropy': 0.8, 'Metalness': 0.9, 'Fresnel': 0.5},
        'textures': [('HALCYON_ScratchesNode', {'count': 24},
                      {'Scale': 6.0, 'Width': 0.01}, 'Anisotropic Rotation')],
    },
    'GLASS': {
        'label': "Glass",
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
        'note': "The 1990s default: a tight white highlight over flat colour",
        'model': 'PHONG',
        'inputs': {'Diffuse Color': (0.75, 0.15, 0.12, 1.0),
                   'Specular Level': 0.85, 'Glossiness': 60.0,
                   'Fresnel': 0.4},
    },
    'RUBBER': {
        'label': "Rubber",
        'note': "Oren-Nayar keeps the edges bright, which is what stops matte "
                "surfaces looking like flat paint",
        'model': 'OREN_NAYAR',
        'inputs': {'Diffuse Color': (0.08, 0.08, 0.09, 1.0),
                   'Roughness': 0.9, 'Specular Level': 0.08,
                   'Glossiness': 8.0},
    },
    'MARBLE': {
        'label': "Polished Marble",
        'note': "Solid marble with veins running through the object, not "
                "wrapped around it",
        'model': 'BLINN',
        'inputs': {'Specular Level': 0.7, 'Glossiness': 180.0, 'Fresnel': 0.5},
        'textures': [('HALCYON_MarbleNode', {'octaves': 6},
                      {'Scale': 3.0, 'Turbulence': 1.2}, 'Diffuse Color')],
    },
    'WOOD': {
        'label': "Varnished Wood",
        'note': "Growth rings turned about the object's own axis",
        'model': 'BLINN_PHONG',
        'inputs': {'Specular Level': 0.5, 'Glossiness': 70.0, 'Fresnel': 0.6},
        'textures': [('HALCYON_WoodNode', {'octaves': 4},
                      {'Scale': 2.0, 'Rings': 9.0, 'Turbulence': 0.4},
                      'Diffuse Color')],
    },
    'TERRAIN': {
        'label': "Terrain",
        'note': "Granite mixed into the base colour, with displacement driving "
                "the bump",
        'model': 'LAMBERT',
        'inputs': {'Specular Level': 0.05},
        'textures': [('HALCYON_GraniteNode', {'octaves': 7},
                      {'Scale': 5.0, 'Contrast': 1.4}, 'Diffuse Color')],
    },
    'TOON': {
        'label': "Cel Shaded",
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
        'note': "Minnaert darkens the middle and lifts the edges, which is the "
                "whole look of pile fabric",
        'model': 'MINNAERT',
        'inputs': {'Diffuse Color': (0.45, 0.05, 0.15, 1.0),
                   'Roughness': 0.8, 'Specular Level': 0.15,
                   'Rim Amount': 0.5, 'Rim Light': (0.9, 0.5, 0.6, 1.0)},
    },
    'HOLOGRAM': {
        'label': "Hologram",
        'note': "Transparent in the middle, bright at the silhouette, scanned "
                "with an ordered dither",
        'model': 'CONSTANT',
        'inputs': {'Diffuse Color': (0.2, 0.9, 0.8, 1.0), 'Opacity': 0.15,
                   'Edge Opacity': 0.95, 'Fresnel': 2.0, 'Fresnel Power': 2.0,
                   'Self-Illumination': (0.1, 0.5, 0.45, 1.0)},
    },
    'WIREFRAME': {
        'label': "Wireframe",
        'note': "Edges only, the rest see-through",
        'model': 'WIREFRAME',
        'inputs': {'Diffuse Color': (0.2, 1.0, 0.4, 1.0)},
    },
}


def build(mat, key):
    """Replace a material's tree with the named template."""
    spec = TEMPLATES.get(key)
    if spec is None:
        return False, f'unknown template {key!r}'
    if not mat.use_nodes:
        mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    out = tree.nodes.new('ShaderNodeOutputMaterial')
    out.location = (320, 0)
    shader = tree.nodes.new('HALCYON_ShaderNode')
    shader.location = (0, 0)
    shader.model = spec['model']

    for name, value in spec.get('inputs', {}).items():
        sock = shader.inputs.get(name)
        if sock is None or not hasattr(sock, 'default_value'):
            continue
        try:
            if hasattr(sock.default_value, '__len__'):
                n = min(len(sock.default_value), len(value))
                for i in range(n):
                    sock.default_value[i] = value[i]
            else:
                sock.default_value = float(value)
        except (TypeError, ValueError):
            pass

    y = 200
    for idname, props, inputs, target in spec.get('textures', []):
        try:
            node = tree.nodes.new(idname)
        except Exception:                                       # noqa: BLE001
            continue
        node.location = (-320, y)
        y -= 260
        for k, v in props.items():
            if hasattr(node, k):
                try:
                    setattr(node, k, v)
                except (TypeError, ValueError):
                    pass
        for k, v in inputs.items():
            sock = node.inputs.get(k)
            if sock is not None and hasattr(sock, 'default_value'):
                try:
                    sock.default_value = v
                except (TypeError, ValueError):
                    pass
        dst = shader.inputs.get(target)
        if dst is not None and node.outputs:
            try:
                tree.links.new(node.outputs[0], dst)
            except Exception:                                   # noqa: BLE001
                pass

    tree.links.new(shader.outputs['Surface'], out.inputs['Surface'])
    shader.refresh_sockets()
    mat.halcyon.use_override = False
    return True, f'{mat.name}: {spec["label"]}'


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


CLASSES = (HALCYON_OT_material_template,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)


def unregister():
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
