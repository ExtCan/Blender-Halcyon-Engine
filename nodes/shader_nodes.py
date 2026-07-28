"""Halcyon's own shader nodes.

The coded-shader node is the interesting one: it compiles real GLSL or HLSL and
*reads the uniform declarations back out* to build its input sockets. Declare
`uniform float rimPower = 2.5;` and a Rim Power socket appears, defaulted to
2.5. Declare `out vec4 Color;` and an output socket appears. The node is driven
by the shader, not by a fixed set of slots the user has to map onto.
"""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Node, NodeSocket

from ..core.shading import MODEL_ITEMS
from ..shaders.compiler import DEFAULT_GLSL, DEFAULT_HLSL, try_compile

ENGINE = 'HALCYON_RENDER'


class HalcyonNodeBase:
    """Shared behaviour: only show up in shader trees."""

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname in ('ShaderNodeTree',)


# =========================================================== classic shader


# What each input does. The list of models that use it is derived from RELEVANT
# below rather than written out again, so the two cannot disagree.
SOCKET_DOCS = {
    'Diffuse Color': "Base colour of the surface under direct light. Plug a "
                     "texture in here for anything patterned",
    'Diffuse Level': "How much of the diffuse term reaches the image. 0 leaves "
                     "only the highlight, which is how chrome is made",
    'Specular Color': "Colour of the highlight. White for plastic and painted "
                      "surfaces; tint it toward the base colour for metal",
    'Specular Level': "Strength of the highlight. 0 gives a completely matte "
                      "surface whatever the model",
    'Glossiness': "Tightness of the highlight, as a cosine exponent. Low values "
                  "give a broad sheen, high values a small hard glint. This is "
                  "the period control -- the microfacet models use Roughness "
                  "instead",
    'Roughness': "Surface microstructure, 0 polished to 1 completely rough. "
                 "Drives the microfacet and rough-diffuse models; the "
                 "cosine-lobe models use Glossiness",
    'Metalness': "Blends the highlight toward the diffuse colour and suppresses "
                 "the diffuse term, so the surface reads as metal rather than "
                 "as a painted object",
    'Anisotropy': "Stretches the highlight along the surface. 0 is round; "
                  "higher values give the streak of brushed metal, hair or "
                  "satin. Negative stretches the other way",
    'Anisotropic Rotation': "Turns the direction the highlight stretches in, in "
                            "radians around the surface normal",
    'Soften': "3D Studio's Soften: rolls the highlight off at grazing angles so "
              "it does not terminate in a hard edge at the silhouette",
    'Ambient': "How strongly the surface picks up the scene's ambient light. "
               "1990s renderers had no bounce light, so this is what keeps "
               "shadowed areas from going black",
    'Self-Illumination': "Colour the surface emits on its own, added after "
                         "lighting. It does not light anything else -- this "
                         "engine has no bounce",
    'Opacity': "1 is solid, 0 is invisible. Anything below 1 sends the surface "
               "through the transparency mode set in Render Properties",
    'IOR': "Index of refraction. Bends rays passing through a transparent "
           "surface, and feeds the Fresnel term of the microfacet models. "
           "Glass is about 1.5, water 1.33",
    'Reflection': "How much of the ray-traced reflection is mixed in. Needs Ray "
                  "Tracing enabled in Render Properties; without it this falls "
                  "back to the environment colour",
    'Translucency': "How much light passes through from behind. Paper, leaves "
                    "and lampshades",
    'Toon Size': "Where the light-to-dark step falls, as a fraction of the "
                 "diffuse range",
    'Toon Smooth': "How soft that step is. 0 is a hard cel edge",
    'Normal': "Replaces the shading normal, for normal and bump mapping",
    'Fresnel': "Brightens the highlight toward the silhouette, the way a real "
               "surface reflects more at grazing angles. The cheapest way to "
               "stop a plastic surface looking flat",
    'Fresnel Power': "How tightly the Fresnel effect hugs the silhouette. "
                     "Higher values confine it to a thinner edge",
    'Fresnel Color': "Tint of the Fresnel boost. White for a clear coat, or "
                     "tint it for anodised metal and soap-film effects",
    'Rim Light': "Colour of the rim term added at the silhouette",
    'Rim Amount': "Strength of an additive rim light. Unlike Fresnel this does "
                  "not need a light source -- it is the backlight cheat every "
                  "1990s demo used to lift a subject off its background",
    'Rim Power': "How tight the rim band is. Higher confines it to the edge",
    'Matcap': "A sphere-mapped image sampled by the view-space normal, giving "
              "a whole material's worth of lighting from one picture. Feed it "
              "an Image Texture through a Matcap Coordinates node",
    'Matcap Blend': "How much the Matcap replaces the lit result. 1 is pure "
                    "matcap and ignores the scene lights entirely",
    'Reflection Color': "Colour the Reflection amount is multiplied by. Plug an "
                        "environment image in here for a reflection map that "
                        "costs nothing, with no ray tracing needed",
    'Edge Opacity': "Opacity at the silhouette, blended toward by the Fresnel "
                    "curve. Below 1 it thins the edges for holograms; above the "
                    "centre opacity it thickens them, which is how glass reads",
    'Backface Color': "Colour used where a surface faces away from the camera",
    'Vertex Color': "Colour carried on the mesh's own vertices. Leave it "
                    "unlinked and set Vertex Color Mix above zero to use the "
                    "active colour attribute directly -- which is how the "
                    "packages that had this worked, since a vertex colour was "
                    "a property of the model rather than something you routed",
    'Sheen': "A velvet lobe added on top of whichever model is chosen: light "
             "scattered back toward the viewer at grazing angles, which is "
             "what makes velvet, suede and dusty cloth bright at their edges "
             "and dark face-on",
    'Sheen Color': "Colour of the sheen lobe. Real velvet's sheen is close to "
                   "white however deeply the pile is dyed",
    'Sheen Roughness': "Width of the sheen band. 0 confines it to the "
                       "silhouette; 1 spreads it across the whole surface",
    'Bump Strength': "Scales how far the Normal input is allowed to bend the "
                     "shading normal away from the surface. 0 ignores the bump "
                     "entirely, 1 uses it as given, above 1 exaggerates it",
    'Refraction Amount': "How much of the ray traced *through* a transparent "
                         "surface is used. 1 is glass; lower values keep what "
                         "is behind the surface where it is, which is how a "
                         "scanline renderer's alpha blend looked",
    'Vertex Color Mix': "How much the vertex colour replaces Diffuse Color. At "
                        "1 the mesh's colours are the surface colour outright, "
                        "which is what the flat-shaded era used them for",
    'Backface Mix': "How strongly Backface Color replaces the normal shading on "
                    "back faces. Useful on single-sided leaves, cloth and cards",
}


# Which models each input genuinely affects. ALL means the parameter is applied
# outside the reflectance function -- in the lighting loop -- so it works the
# same whichever model is chosen. The rest were measured by perturbing each
# parameter and seeing which models changed their output, and a test re-runs
# that measurement against this table so it cannot rot.
ALL = '*'

SOCKET_MODELS = {
    'Diffuse Color': ALL,
    'Diffuse Level': ALL,
    # Strauss derives its highlight colour from metalness and the base colour
    # rather than reading this socket; Toon does read it.
    'Specular Color': ('GOURAUD', 'FLAT', 'PHONG', 'BLINN_PHONG', 'BLINN',
                       'COOK_TORRANCE', 'WARD', 'ANISOTROPIC', 'MULTI_LAYER',
                       'TOON'),
    'Specular Level': ALL,
    'Glossiness': ('GOURAUD', 'FLAT', 'PHONG', 'BLINN_PHONG', 'BLINN',
                   'ANISOTROPIC', 'METAL', 'STRAUSS', 'MULTI_LAYER'),
    'Roughness': ('COOK_TORRANCE', 'OREN_NAYAR', 'MINNAERT', 'WARD'),
    'Metalness': ALL,
    'Anisotropy': ('WARD', 'ANISOTROPIC'),
    'Anisotropic Rotation': ('WARD', 'ANISOTROPIC'),
    'Soften': ALL,
    'Ambient': ALL,
    'Self-Illumination': ALL,
    'Opacity': ALL,
    'IOR': ('BLINN', 'COOK_TORRANCE'),
    'Reflection': ALL,
    'Translucency': ('TRANSLUCENT',),
    'Toon Size': ('TOON',),
    'Toon Smooth': ('TOON',),
    'Normal': ALL,
    'Fresnel': ALL, 'Fresnel Power': ALL, 'Fresnel Color': ALL,
    'Rim Light': ALL, 'Rim Amount': ALL, 'Rim Power': ALL,
    'Matcap': ALL, 'Matcap Blend': ALL, 'Reflection Color': ALL,
    'Edge Opacity': ALL, 'Backface Color': ALL, 'Backface Mix': ALL,
    'Vertex Color': ALL, 'Vertex Color Mix': ALL,
    'Sheen': ALL, 'Sheen Color': ALL, 'Sheen Roughness': ALL,
    'Bump Strength': ALL, 'Refraction Amount': ALL,
}

# measured parameters -- the ones a test verifies against the shading code
MEASURED = {'Specular Color', 'Glossiness', 'Roughness', 'Anisotropy',
            'Anisotropic Rotation', 'IOR', 'Toon Size', 'Toon Smooth'}


def _models_using(socket_name, relevant=None, all_models=None):
    """Human-readable note on which models an input affects."""
    who = SOCKET_MODELS.get(socket_name)
    if who is None:
        return ""
    if who == ALL:
        return "affects every model"
    pretty = ', '.join(m.replace('_', ' ').title() for m in who)
    if len(who) == 1:
        return f"only affects {pretty}"
    return f"affects {pretty}"


class HALCYON_ShaderNode(Node, HalcyonNodeBase):
    """A 1990s reflectance model with its own controls"""

    bl_idname = 'HALCYON_ShaderNode'
    bl_label = "Halcyon Shader"
    bl_icon = 'SHADING_RENDERED'
    bl_width_default = 200

    def _update(self, context):
        self.refresh_sockets()

    model: EnumProperty(name="Model", items=[(a, b, c) for a, b, c in MODEL_ITEMS],
                        default='PHONG', update=_update)
    toon_steps: IntProperty(name="Toon Steps", default=2, min=1, max=16)

    # which sockets each model actually uses -- the rest are hidden, not removed,
    # so switching models never loses a connection
    MODEL_ORDER = tuple(m[0] for m in MODEL_ITEMS)

    RELEVANT = {
        'LAMBERT': {'Diffuse Color', 'Diffuse Level', 'Ambient', 'Opacity',
                    'Self-Illumination', 'Normal', 'Bump Strength'},
        'GOURAUD': None, 'FLAT': None, 'PHONG': None, 'BLINN_PHONG': None,
        'BLINN': None, 'COOK_TORRANCE': None,
        'OREN_NAYAR': {'Diffuse Color', 'Diffuse Level', 'Roughness', 'Ambient',
                       'Opacity', 'Self-Illumination', 'Normal',
                       'Bump Strength'},
        'MINNAERT': {'Diffuse Color', 'Diffuse Level', 'Roughness', 'Ambient',
                     'Opacity', 'Self-Illumination', 'Normal',
                     'Bump Strength'},
        'WARD': None, 'ANISOTROPIC': None, 'METAL': None, 'STRAUSS': None,
        'MULTI_LAYER': None,
        'TOON': {'Diffuse Color', 'Diffuse Level', 'Specular Color',
                 'Specular Level', 'Toon Size', 'Toon Smooth', 'Ambient',
                 'Opacity', 'Self-Illumination', 'Normal', 'Bump Strength'},
        'TRANSLUCENT': {'Diffuse Color', 'Diffuse Level', 'Translucency',
                        'Ambient', 'Opacity', 'Self-Illumination', 'Normal',
                        'Bump Strength'},
        'CONSTANT': {'Diffuse Color', 'Opacity', 'Self-Illumination'},
        'WIREFRAME': {'Diffuse Color', 'Opacity'},
    }

    SOCKETS = (
        ('NodeSocketColor', 'Diffuse Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketFloat', 'Diffuse Level', 1.0),
        ('NodeSocketColor', 'Specular Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Specular Level', 0.5),
        ('NodeSocketFloat', 'Glossiness', 25.0),
        ('NodeSocketFloat', 'Roughness', 0.3),
        ('NodeSocketFloat', 'Metalness', 0.0),
        ('NodeSocketFloat', 'Anisotropy', 0.0),
        ('NodeSocketFloat', 'Anisotropic Rotation', 0.0),
        ('NodeSocketFloat', 'Soften', 0.0),
        ('NodeSocketFloat', 'Ambient', 1.0),
        ('NodeSocketColor', 'Self-Illumination', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Opacity', 1.0),
        ('NodeSocketFloat', 'IOR', 1.45),
        ('NodeSocketFloat', 'Reflection', 0.0),
        ('NodeSocketFloat', 'Translucency', 0.0),
        ('NodeSocketFloat', 'Toon Size', 0.5),
        ('NodeSocketFloat', 'Toon Smooth', 0.05),
        ('NodeSocketVector', 'Normal', None),
        ('NodeSocketFloat', 'Fresnel', 0.0),
        ('NodeSocketFloat', 'Fresnel Power', 3.0),
        ('NodeSocketColor', 'Fresnel Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketColor', 'Rim Light', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Rim Amount', 0.0),
        ('NodeSocketFloat', 'Rim Power', 3.0),
        ('NodeSocketColor', 'Matcap', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Matcap Blend', 0.0),
        ('NodeSocketColor', 'Reflection Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Edge Opacity', 1.0),
        ('NodeSocketColor', 'Backface Color', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Backface Mix', 0.0),
        ('NodeSocketColor', 'Vertex Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Vertex Color Mix', 0.0),
        ('NodeSocketFloat', 'Sheen', 0.0),
        ('NodeSocketColor', 'Sheen Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Sheen Roughness', 0.3),
        ('NodeSocketFloat', 'Bump Strength', 1.0),
        ('NodeSocketFloat', 'Refraction Amount', 1.0),
    )

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            sock = self.inputs.new(kind, name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
            if name in ('Glossiness',):
                sock.default_value = 25.0
            self._document(sock, name)
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Collapses to the chosen reflectance "
            "model when Halcyon renders it")
        self.refresh_sockets()

    def _document(self, sock, name):
        """Attach the tooltip. Older builds have no settable socket
        description, so this must never be fatal."""
        doc = SOCKET_DOCS.get(name)
        if not doc:
            return
        models = _models_using(name)
        try:
            sock.description = f"{doc}. ({models})" if models else doc
        except (AttributeError, TypeError):
            pass

    #: applied after the reflectance model, so never hidden
    ALWAYS = ('Fresnel', 'Fresnel Power', 'Fresnel Color', 'Rim Light',
              'Rim Amount', 'Rim Power', 'Matcap', 'Matcap Blend',
              'Reflection Color', 'Edge Opacity', 'Backface Color',
              'Backface Mix', 'Sheen', 'Sheen Color', 'Sheen Roughness',
              'Refraction Amount')

    def refresh_sockets(self):
        keep = self.RELEVANT.get(self.model)
        if keep is not None:
            keep = set(keep) | set(self.ALWAYS)
        for sock in self.inputs:
            sock.hide = bool(keep is not None and sock.name not in keep
                             and not sock.is_linked)
            self._document(sock, sock.name)

    def model_description(self):
        for ident, _label, desc in MODEL_ITEMS:
            if ident == self.model:
                return desc
        return ""

    def draw_buttons(self, context, layout):
        layout.prop(self, 'model', text="")
        if self.model == 'TOON':
            layout.prop(self, 'toon_steps')
        used = sum(1 for _k, n, _d in self.SOCKETS
                   if SOCKET_MODELS.get(n) == ALL
                   or (SOCKET_MODELS.get(n) and self.model in SOCKET_MODELS[n]))
        row = layout.row()
        row.active = False
        row.label(text=f"{used} of {len(self.SOCKETS)} inputs used",
                  icon='HIDE_OFF')

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, 'model', text="")
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8
        for line in _wrap_text(self.model_description(), 40):
            col.label(text=line)
        layout.separator()
        layout.label(text="Inputs this model uses:", icon='HIDE_OFF')
        col = layout.column(align=True)
        col.scale_y = 0.8
        shown = 0
        for _kind, name, _default in self.SOCKETS:
            who = SOCKET_MODELS.get(name)
            if who == ALL or (who and self.model in who):
                col.label(text="  " + name)
                shown += 1
        if shown < len(self.SOCKETS):
            layout.label(text=f"{len(self.SOCKETS) - shown} inputs are ignored "
                              f"by this model", icon='INFO')

    def draw_label(self):
        for ident, label, _ in MODEL_ITEMS:
            if ident == self.model:
                return label
        return self.bl_label


# ======================================================== coded shader node

LANGUAGES = (
    ('GLSL', "GLSL", "OpenGL Shading Language"),
    ('HLSL', "HLSL", "High Level Shading Language (Direct3D)"),
)

KIND_TO_SOCKET = {
    'VALUE': 'NodeSocketFloat',
    'INT': 'NodeSocketInt',
    'BOOL': 'NodeSocketBool',
    'VECTOR': 'NodeSocketVector',
    'VECTOR2': 'NodeSocketVector',
    'RGBA': 'NodeSocketColor',
    'MATRIX': 'NodeSocketFloat',
    'IMAGE': 'NodeSocketColor',
}


def _wrap_text(text, width):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _pretty(name):
    """rimPower -> Rim Power, base_color -> Base Color."""
    out = []
    prev_lower = False
    for ch in name.replace('_', ' '):
        if ch.isupper() and prev_lower:
            out.append(' ')
        out.append(ch)
        prev_lower = ch.islower() or ch.isdigit()
    return ''.join(out).strip().title()


_REBUILD_QUEUED = [False]


def _iter_trees():
    for coll in (getattr(bpy.data, 'materials', ()),
                 getattr(bpy.data, 'worlds', ()),
                 getattr(bpy.data, 'node_groups', ())):
        for block in coll:
            tree = getattr(block, 'node_tree', None) or (
                block if hasattr(block, 'nodes') else None)
            if tree is not None:
                yield tree


def _run_rebuilds():
    """Recompile every node that asked for it, outside any update callback.

    Nothing is carried across the timer except a flag stored on the nodes
    themselves -- holding a node pointer across a callback is its own way to
    crash, because the tree may have been rebuilt in between.
    """
    _REBUILD_QUEUED[0] = False
    for tree in _iter_trees():
        for node in list(getattr(tree, 'nodes', ())):
            if node.bl_idname == 'HALCYON_CodeNode' and \
                    getattr(node, 'needs_rebuild', False):
                node.needs_rebuild = False
                try:
                    node.compile_source()
                except Exception as exc:                        # noqa: BLE001
                    node.error = f'{type(exc).__name__}: {exc}'
    return None                       # one-shot


def _schedule_rebuild():
    if _REBUILD_QUEUED[0]:
        return
    _REBUILD_QUEUED[0] = True
    try:
        bpy.app.timers.register(_run_rebuilds, first_interval=0.0)
    except Exception:                                           # noqa: BLE001
        # no timer available (headless, or during registration): do it now,
        # which is safe precisely because we are not inside a callback
        _run_rebuilds()


class HALCYON_CodeNode(Node, HalcyonNodeBase):
    """Write a shader in GLSL or HLSL; its uniforms become input sockets"""

    bl_idname = 'HALCYON_CodeNode'
    bl_label = "Coded Shader"
    bl_icon = 'TEXT'
    bl_width_default = 260

    def _lang_changed(self, context):
        # Sockets must not be added or removed from inside an RNA update
        # callback -- Blender is mid-update and it segfaults. Worse, setting
        # source_text here fires _source_changed as well, so the rebuild would
        # happen inside a *nested* callback. Both are deferred instead.
        if self._busy:
            return
        self._busy = True
        try:
            if not self.source and not self.source_text.strip():
                self.source_text = DEFAULT_GLSL if self.language == 'GLSL' \
                    else DEFAULT_HLSL
        finally:
            self._busy = False
        self.needs_rebuild = True
        _schedule_rebuild()

    def _source_changed(self, context):
        if self._busy:
            return
        self.needs_rebuild = True
        _schedule_rebuild()

    language: EnumProperty(name="Language", items=LANGUAGES, default='GLSL',
                           update=_lang_changed)
    needs_rebuild: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    _busy = False
    source: PointerProperty(name="Text", type=bpy.types.Text,
                            description="A text datablock holding the shader",
                            update=_source_changed)
    source_text: StringProperty(name="Inline Source", default='',
                                options={'HIDDEN'})
    as_surface: BoolProperty(
        name="Output as Surface", default=False,
        description="Emit the first output as a shader closure instead of colour")
    error: StringProperty(name="Error", default='', options={'HIDDEN'})
    warn: StringProperty(name="Warning", default='', options={'HIDDEN'})
    auto_compile: BoolProperty(name="Auto Compile", default=True)

    def init(self, context):
        self.source_text = DEFAULT_GLSL
        self.outputs.new('NodeSocketColor', 'Color')
        self.compile_source()

    # ....................................................... compilation
    def get_source(self):
        if self.source is not None:
            try:
                return self.source.as_string()
            except Exception:                                   # noqa: BLE001
                return ''
        return self.source_text or ''

    def compile_source(self):
        src = self.get_source()
        if not src.strip():
            self.error = ''
            return None
        prog, err = try_compile(src, self.language)
        if prog is None:
            self.error = err or 'compile failed'
            return None
        self.error = ''
        self.warn = '; '.join(prog.warnings[:3]) if prog.warnings else ''
        self.rebuild_sockets(prog)
        return prog

    def rebuild_sockets(self, prog):
        """Sockets follow the shader's declarations, keeping existing links."""
        wanted_in = prog.uniform_schema()
        wanted_out = prog.output_schema()

        keep = {}
        for sock in self.inputs:
            keep[sock.name] = (sock.default_value if hasattr(sock, 'default_value')
                               else None,
                               [(l.from_node.name, l.from_socket.name)
                                for l in sock.links])
        links_out = []
        for sock in self.outputs:
            for l in sock.links:
                links_out.append((sock.name, l.to_node.name, l.to_socket.name))

        tree = self.id_data
        self.inputs.clear()
        for u in wanted_in:
            kind = u.get('kind', 'VALUE')
            stype = KIND_TO_SOCKET.get(kind, 'NodeSocketFloat')
            label = _pretty(u['name'])
            sock = self.inputs.new(stype, label)
            sock.halcyon_uniform = u['name']
            sock.halcyon_is_image = (kind == 'IMAGE')
            dv = u.get('default')
            if dv is not None and hasattr(sock, 'default_value'):
                try:
                    if kind == 'RGBA':
                        v = list(dv) + [1.0] * (4 - len(dv)) if hasattr(dv, '__len__') \
                            else [float(dv)] * 3 + [1.0]
                        sock.default_value = v[:4]
                    elif kind in ('VECTOR', 'VECTOR2'):
                        v = list(dv) if hasattr(dv, '__len__') else [float(dv)] * 3
                        sock.default_value = (v + [0.0, 0.0, 0.0])[:3]
                    else:
                        sock.default_value = float(dv if not hasattr(dv, '__len__')
                                                   else dv[0])
                except (TypeError, ValueError):
                    pass
            prev = keep.get(label)
            if prev and prev[0] is not None and hasattr(sock, 'default_value'):
                try:
                    sock.default_value = prev[0]
                except (TypeError, ValueError):
                    pass

        self.outputs.clear()
        if self.as_surface:
            self.outputs.new('NodeSocketShader', 'Surface')
        for o in wanted_out:
            stype = KIND_TO_SOCKET.get(o.get('kind', 'RGBA'), 'NodeSocketColor')
            sock = self.outputs.new(stype, _pretty(o['name']))
            sock.halcyon_key = o['name']

        # restore links that still have somewhere to go
        for name, (dv, srcs) in keep.items():
            sock = self.inputs.get(name)
            if sock is None:
                continue
            for from_name, from_sock in srcs:
                node = tree.nodes.get(from_name)
                if node and node.outputs.get(from_sock):
                    try:
                        tree.links.new(node.outputs[from_sock], sock)
                    except Exception:                           # noqa: BLE001
                        pass
        for out_name, to_name, to_sock in links_out:
            sock = self.outputs.get(out_name)
            node = tree.nodes.get(to_name)
            if sock is not None and node is not None and node.inputs.get(to_sock):
                try:
                    tree.links.new(sock, node.inputs[to_sock])
                except Exception:                               # noqa: BLE001
                    pass

    # ............................................................. drawing
    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, 'language', expand=True)
        layout.prop(self, 'source', text="")
        row = layout.row(align=True)
        row.operator('halcyon.compile_shader', icon='FILE_REFRESH').node = self.name
        row.operator('halcyon.new_shader_text', icon='ADD', text="")
        layout.prop(self, 'as_surface')
        if self.error:
            box = layout.box()
            box.alert = True
            for line in self.error.split('\n')[:4]:
                box.label(text=line, icon='ERROR')
        elif self.warn:
            box = layout.box()
            for line in self.warn.split(';')[:3]:
                box.label(text=line.strip(), icon='INFO')

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, 'auto_compile')
        layout.separator()
        layout.label(text="Available inputs:")
        col = layout.column(align=True)
        for name in ('vPosition', 'vNormal', 'vUV', 'vColor', 'vView',
                     'gl_FragCoord', 'iTime', 'iResolution'):
            col.label(text="  " + name)

    def draw_label(self):
        return f"{self.language} Shader"


class HALCYON_OT_compile_shader(bpy.types.Operator):
    bl_idname = 'halcyon.compile_shader'
    bl_label = "Compile"
    bl_description = "Compile the shader and rebuild this node's sockets"
    bl_options = {'REGISTER', 'UNDO'}

    node: StringProperty()

    def execute(self, context):
        space = context.space_data
        tree = getattr(space, 'edit_tree', None) or getattr(space, 'node_tree', None)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node) or context.active_node
        if node is None or node.bl_idname != 'HALCYON_CodeNode':
            return {'CANCELLED'}
        prog = node.compile_source()
        if prog is None:
            self.report({'ERROR'}, node.error or "Compile failed")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Compiled: {len(node.inputs)} uniforms, "
                              f"{len(node.outputs)} outputs")
        return {'FINISHED'}


class HALCYON_OT_new_shader_text(bpy.types.Operator):
    bl_idname = 'halcyon.new_shader_text'
    bl_label = "New Shader Text"
    bl_description = "Create a text datablock with a starter shader"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        node = context.active_node
        lang = getattr(node, 'language', 'GLSL')
        text = bpy.data.texts.new(f"halcyon_{lang.lower()}")
        text.write(DEFAULT_GLSL if lang == 'GLSL' else DEFAULT_HLSL)
        if node is not None and node.bl_idname == 'HALCYON_CodeNode':
            node.source = text
        return {'FINISHED'}


# ========================================================== retro utilities


class HALCYON_PosterizeNode(Node, HalcyonNodeBase):
    """Quantise a colour to a fixed number of levels per channel"""

    bl_idname = 'HALCYON_PosterizeNode'
    bl_label = "Posterize"
    bl_icon = 'IMAGE_ZDEPTH'

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Color').default_value = (.8, .8, .8, 1)
        self.inputs.new('NodeSocketFloat', 'Levels').default_value = 8.0
        self.outputs.new('NodeSocketColor', 'Color')


class HALCYON_DitherNode(Node, HalcyonNodeBase):
    """Ordered dither against the screen position, at shading time"""

    bl_idname = 'HALCYON_DitherNode'
    bl_label = "Ordered Dither"
    bl_icon = 'TEXTURE'

    pattern: EnumProperty(name="Pattern", items=(
        ('BAYER2', "Bayer 2x2", ""), ('BAYER4', "Bayer 4x4", ""),
        ('BAYER8', "Bayer 8x8", ""), ('HALFTONE', "Halftone", "")),
        default='BAYER4')

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Color').default_value = (.8, .8, .8, 1)
        self.inputs.new('NodeSocketFloat', 'Levels').default_value = 4.0
        self.inputs.new('NodeSocketFloat', 'Strength').default_value = 1.0
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'pattern', text="")


class HALCYON_DepthCueNode(Node, HalcyonNodeBase):
    """Blend toward a fog colour by distance, the way depth cueing worked"""

    bl_idname = 'HALCYON_DepthCueNode'
    bl_label = "Depth Cue"
    bl_icon = 'MOD_FLUIDSIM'

    mode: EnumProperty(name="Falloff", items=(
        ('LINEAR', "Linear", ""), ('EXP', "Exponential", ""),
        ('EXP2', "Exponential Squared", ""),
        ('TABLE16', "16-Step Table", "")), default='LINEAR')

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Color').default_value = (.8, .8, .8, 1)
        self.inputs.new('NodeSocketColor', 'Fog Color').default_value = (.5, .55, .65, 1)
        self.inputs.new('NodeSocketFloat', 'Start').default_value = 5.0
        self.inputs.new('NodeSocketFloat', 'End').default_value = 40.0
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode', text="")


class HALCYON_ScreenInfoNode(Node, HalcyonNodeBase):
    """Screen-space information: pixel coordinates, depth, facing"""

    bl_idname = 'HALCYON_ScreenInfoNode'
    bl_label = "Screen Info"
    bl_icon = 'VIEW_CAMERA'

    def init(self, context):
        self.outputs.new('NodeSocketVector', 'Screen UV')
        self.outputs.new('NodeSocketVector', 'Pixel')
        self.outputs.new('NodeSocketFloat', 'Depth')
        self.outputs.new('NodeSocketFloat', 'Facing')
        self.outputs.new('NodeSocketFloat', 'Frame')
        self.outputs.new('NodeSocketFloat', 'Time')


NODES = (HALCYON_ShaderNode, HALCYON_CodeNode, HALCYON_PosterizeNode,
         HALCYON_DitherNode, HALCYON_DepthCueNode, HALCYON_ScreenInfoNode)
OPERATORS = (HALCYON_OT_compile_shader, HALCYON_OT_new_shader_text)


# ------------------------------------------------------------- the Add menu

def draw_add_menu(self, context):
    """4.0 removed nodeitems_utils, so append to the Add menu directly."""
    tree = getattr(context.space_data, 'edit_tree', None)
    if tree is None or tree.bl_idname != 'ShaderNodeTree':
        return
    if context.engine != ENGINE:
        return
    layout = self.layout
    layout.separator()
    layout.menu('NODE_MT_halcyon_add', icon='SHADING_RENDERED')


class NODE_MT_halcyon_add(bpy.types.Menu):
    bl_idname = 'NODE_MT_halcyon_add'
    bl_label = "Halcyon"

    def draw(self, context):
        layout = self.layout
        for cls in NODES:
            op = layout.operator('node.add_node', text=cls.bl_label,
                                 icon=getattr(cls, 'bl_icon', 'NONE'))
            op.type = cls.bl_idname
            op.use_transform = True
        layout.separator()
        layout.menu('NODE_MT_halcyon_textures', icon='TEXTURE')


_menu_owner = None


def register():
    global _menu_owner
    from . import pattern_nodes
    pattern_nodes.register()
    for sock_cls in (NodeSocket,):
        if not hasattr(sock_cls, 'halcyon_uniform'):
            sock_cls.halcyon_uniform = StringProperty(default='')
            sock_cls.halcyon_key = StringProperty(default='')
            sock_cls.halcyon_image_key = StringProperty(default='')
            sock_cls.halcyon_is_image = BoolProperty(default=False)
    for cls in OPERATORS + NODES:
        bpy.utils.register_class(cls)
    bpy.utils.register_class(NODE_MT_halcyon_add)
    from .. import compat
    _menu_owner = compat.register_node_menu(draw_add_menu)


def unregister():
    from . import pattern_nodes
    pattern_nodes.unregister()
    from .. import compat
    compat.unregister_node_menu(draw_add_menu, _menu_owner)
    for cls in (NODE_MT_halcyon_add,) + tuple(reversed(NODES + OPERATORS)):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass
