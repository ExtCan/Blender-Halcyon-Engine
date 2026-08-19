"""Halcyon's own shader nodes.

The coded-shader node is the interesting one: it compiles real GLSL or HLSL and
*reads the uniform declarations back out* to build its input sockets. Declare
`uniform float rimPower = 2.5;` and a Rim Power socket appears, defaulted to
2.5. Declare `out vec4 Color;` and an output socket appears. The node is driven
by the shader, not by a fixed set of slots the user has to map onto.
"""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Node, NodeSocket, PropertyGroup

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
    'Bump Height': "A greyscale height field bumped straight into the shading "
                   "normal -- plug any texture here and its bright parts rise. "
                   "Behind the scenes this is exactly a Bump node between the "
                   "texture and Normal, scaled by Bump Strength, so it renders "
                   "identically on both devices. Unlinked, it does nothing",
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
                       'TOON', 'BI_COOKTORR', 'BI_PHONG', 'BI_BLINN'),
    'Specular Level': ALL,
    'Glossiness': ('GOURAUD', 'FLAT', 'PHONG', 'BLINN_PHONG', 'BLINN',
                   'ANISOTROPIC', 'METAL', 'STRAUSS', 'MULTI_LAYER',
                   'BI_COOKTORR', 'BI_PHONG', 'BI_BLINN'),
    'Roughness': ('COOK_TORRANCE', 'OREN_NAYAR', 'MINNAERT', 'WARD'),
    'Metalness': ALL,
    'Anisotropy': ('WARD', 'ANISOTROPIC'),
    'Anisotropic Rotation': ('WARD', 'ANISOTROPIC'),
    'Soften': ALL,
    'Ambient': ALL,
    'Self-Illumination': ALL,
    'Opacity': ALL,
    'IOR': ('BLINN', 'COOK_TORRANCE', 'BI_BLINN'),
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
    'Bump Strength': ALL, 'Bump Height': ALL, 'Refraction Amount': ALL,
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
    wire_size: FloatProperty(
        name="Wire Size", default=1.0, min=0.05, max=16.0,
        description="Width of the drawn edge, in rendered pixels. A material "
                    "shaded as Wireframe had no reachable width at all before "
                    "-- on a dense mesh at a period resolution every pixel is "
                    "within a pixel of an edge, and the surface fills in")

    # which sockets each model actually uses -- the rest are hidden, not removed,
    # so switching models never loses a connection
    MODEL_ORDER = tuple(m[0] for m in MODEL_ITEMS)

    RELEVANT = {
        'LAMBERT': {'Diffuse Color', 'Diffuse Level', 'Ambient', 'Opacity',
                    'Self-Illumination', 'Normal', 'Bump Strength', 'Bump Height'},
        'GOURAUD': None, 'FLAT': None, 'PHONG': None, 'BLINN_PHONG': None,
        'BLINN': None, 'COOK_TORRANCE': None,
        'BI_COOKTORR': None, 'BI_PHONG': None,
        'BI_BLINN': None,
        'OREN_NAYAR': {'Diffuse Color', 'Diffuse Level', 'Roughness', 'Ambient',
                       'Opacity', 'Self-Illumination', 'Normal',
                       'Bump Strength', 'Bump Height'},
        'MINNAERT': {'Diffuse Color', 'Diffuse Level', 'Roughness', 'Ambient',
                     'Opacity', 'Self-Illumination', 'Normal',
                     'Bump Strength', 'Bump Height'},
        'WARD': None, 'ANISOTROPIC': None, 'METAL': None, 'STRAUSS': None,
        'MULTI_LAYER': None,
        'TOON': {'Diffuse Color', 'Diffuse Level', 'Specular Color',
                 'Specular Level', 'Toon Size', 'Toon Smooth', 'Ambient',
                 'Opacity', 'Self-Illumination', 'Normal', 'Bump Strength', 'Bump Height'},
        'TRANSLUCENT': {'Diffuse Color', 'Diffuse Level', 'Translucency',
                        'Ambient', 'Opacity', 'Self-Illumination', 'Normal',
                        'Bump Strength', 'Bump Height'},
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
        ('NodeSocketFloat', 'Bump Height', 0.5),
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

    def ensure_sockets(self):
        """Create any spec socket this saved instance predates.

        NOT called from update callbacks (socket topology changes there
        are forbidden by the guard test, for good reason): the load-post
        migration below runs it at file load, where mutation is safe --
        so an old file gains Bump Height the moment it opens.
        """
        have = {s.name for s in self.inputs}
        for kind, name, default in self.SOCKETS:
            if name in have:
                continue
            try:
                sock = self.inputs.new(kind, name)
                if default is not None:
                    sock.default_value = default
            except Exception:                                   # noqa: BLE001
                pass

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
        if self.model == 'WIREFRAME':
            layout.prop(self, 'wire_size')
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


#: nodes currently inside their own update callback, by pointer
_UPDATING = set()


def _node_key(node):
    try:
        return node.as_pointer()
    except Exception:                                           # noqa: BLE001
        return id(node)


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
        # callback -- Blender is mid-update and it segfaults -- so the rebuild
        # is only ever flagged here and done from a timer.
        #
        # The re-entrancy guard used to be `self._busy`, an ordinary Python
        # attribute. Blender hands out a fresh wrapper object on every access
        # to a node, so that attribute was written to a temporary and read back
        # as the class default: the guard was never once closed. It lives in a
        # module-level set now, keyed by the node's own pointer, which is the
        # only identity that survives the wrapper being rebuilt.
        key = _node_key(self)
        if key in _UPDATING:
            return
        _UPDATING.add(key)
        try:
            if not self.source and not self.source_text.strip():
                self.source_text = DEFAULT_GLSL if self.language == 'GLSL' \
                    else DEFAULT_HLSL
        finally:
            _UPDATING.discard(key)
        self.needs_rebuild = True
        _schedule_rebuild()

    def _source_changed(self, context):
        if _node_key(self) in _UPDATING:
            return
        self.needs_rebuild = True
        _schedule_rebuild()

    language: EnumProperty(name="Language", items=LANGUAGES, default='GLSL',
                           update=_lang_changed)
    needs_rebuild: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
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

        # `sock.default_value` on a colour or vector socket is a live view into
        # the socket's own memory, not a copy. Holding one across
        # `inputs.clear()` leaves a pointer into freed memory, and reading it
        # back afterwards to restore the value is a use-after-free -- which
        # takes Blender down rather than raising. Every value is materialised
        # here, before anything is cleared.
        keep = {}
        for sock in self.inputs:
            value = None
            if hasattr(sock, 'default_value'):
                dv = sock.default_value
                value = tuple(dv) if hasattr(dv, '__len__') else float(dv)
            keep[sock.name] = (value,
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


class HALCYON_BIInfluenceNode(Node, HalcyonNodeBase):
    """Blender Internal's value-channel influence, as one node.

    2.79's texture_value_blend(): the texture intensity does not become
    the value -- it blends the Base toward DVar by Intensity x Factor,
    in the slot's blend mode. Every imported hardness/emit/alpha slot
    rides one of these, and it is available for hand wiring too.
    """

    bl_idname = 'HALCYON_BIInfluenceNode'
    bl_label = "BI Influence"
    bl_icon = 'MOD_HUE_SATURATION'

    blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f'texture_value_blend {m}') for m in (
                'MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                'LIGHT', 'SCREEN', 'OVERLAY', 'SOFT', 'LINEAR')])
    # ---- do_material_tex's slot flags, set by the importer from the
    # texture kind and the MTex texflag. With tex_rgb on, the value
    # channel takes the COLOUR's Rec.709 luminance (or its alpha under
    # AlphaMix) as the intensity, exactly as an RGB texture on a value
    # channel did in BI; Intensity is then only the no-colour fallback.
    tex_rgb: BoolProperty(
        name="Texture Yields RGB", default=False,
        description="The linked texture produces colour (a colorband, "
                    "an image, Magic): the intensity is taken from the "
                    "Color input per BI's rules")
    rgbtoint: BoolProperty(
        name="RGB to Intensity", default=False,
        description="MTex RGBToIntensity: collapse the colour to its "
                    "luminance before anything else")
    negative: BoolProperty(
        name="Negative", default=False,
        description="MTex Negative: invert the texture output")
    alphamix: BoolProperty(
        name="Alpha Mix", default=False,
        description="MTex Calculate Alpha mix: the texture's alpha is "
                    "the intensity")
    calc_alpha: BoolProperty(
        name="Calculate Alpha", default=False,
        description="TEX_CALCALPHA: the alpha is max(r,g,b), exactly "
                    "imagewrap")
    neg_alpha: BoolProperty(
        name="Negate Alpha", default=False,
        description="TEX_NEGALPHA: the alpha inverts after it is "
                    "decided")

    def init(self, context):
        self.inputs.new('NodeSocketFloat', 'Base').default_value = 0.5
        self.inputs.new('NodeSocketFloat', 'Intensity').default_value = 0.0
        self.inputs.new('NodeSocketFloat', 'Factor').default_value = 1.0
        self.inputs.new('NodeSocketFloat', 'DVar').default_value = 1.0
        self.inputs.new('NodeSocketColor', 'Color').default_value = \
            (1.0, 1.0, 1.0, 1.0)
        self.inputs.new('NodeSocketFloat', 'Alpha').default_value = 1.0
        self.outputs.new('NodeSocketFloat', 'Value')
        docs = {
            'Base': "The channel's own value before the texture",
            'Intensity': "The texture intensity (a Fac output)",
            'Factor': "The influence slider; negative flips the blend, "
                      "exactly BI's slider",
            'DVar': "The blend TARGET -- BI's DVar slider, not the "
                    "texture value",
            'Color': "The texture's colour output, read when the "
                     "texture yields RGB",
            'Alpha': "The texture's alpha, read under Alpha Mix",
        }
        for s in self.inputs:
            d = docs.get(s.name)
            if d:
                try:
                    s.description = d
                except (AttributeError, TypeError):
                    pass

    def draw_buttons(self, context, layout):
        layout.prop(self, 'blend', text="")
        if self.tex_rgb or self.rgbtoint or self.negative or self.alphamix:
            row = layout.row(align=True)
            row.label(text=('RGB ' if self.tex_rgb else '')
                      + ('toInt ' if self.rgbtoint else '')
                      + ('Neg ' if self.negative else '')
                      + ('AlphaMix' if self.alphamix else ''))


class HALCYON_BIRGBBlendNode(Node, HalcyonNodeBase):
    """Blender Internal's colour-channel influence, as one node.

    2.79's texture_rgb_blend(): the base colour blends toward `tcol` by
    the texture's per-pixel factor times the influence slider. `tcol`
    is the texture's own colour when it yields one -- its ALPHA is then
    the factor -- and the SLOT's colour swatch when it does not, with
    the intensity as the factor. Every imported Color/Specular
    Color/Mirror Color slot rides one of these.
    """

    bl_idname = 'HALCYON_BIRGBBlendNode'
    bl_label = "BI Color Influence"
    bl_icon = 'MOD_HUE_SATURATION'

    blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f'texture_rgb_blend {m}') for m in (
                'MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                'LIGHT', 'SCREEN', 'OVERLAY', 'HUE', 'SAT', 'VAL',
                'COLOR', 'SOFT', 'LINEAR')])
    tex_rgb: BoolProperty(
        name="Texture Yields RGB", default=False,
        description="The linked texture produces colour: it supplies "
                    "tcol, and its alpha is the per-pixel factor")
    rgbtoint: BoolProperty(
        name="RGB to Intensity", default=False,
        description="MTex RGBToIntensity: collapse the colour to its "
                    "luminance first; the slot colour becomes tcol")
    negative: BoolProperty(
        name="Negative", default=False,
        description="MTex Negative: invert the texture output")
    alphamix: BoolProperty(
        name="Alpha Mix", default=False,
        description="MTex Calculate Alpha mix")
    calc_alpha: BoolProperty(
        name="Calculate Alpha", default=False,
        description="TEX_CALCALPHA: the alpha is max(r,g,b), exactly "
                    "imagewrap")
    neg_alpha: BoolProperty(
        name="Negate Alpha", default=False,
        description="TEX_NEGALPHA: the alpha inverts after it is "
                    "decided")
    map_alpha: BoolProperty(
        name="Slot Also Maps Alpha", default=False,
        description="The same slot drives Alpha: BI then keeps the "
                    "intensity as the factor unless Alpha Mix is on")

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Base').default_value = \
            (0.8, 0.8, 0.8, 1.0)
        self.inputs.new('NodeSocketColor', 'Color').default_value = \
            (1.0, 1.0, 1.0, 1.0)
        self.inputs.new('NodeSocketFloat', 'Intensity').default_value = 0.0
        self.inputs.new('NodeSocketFloat', 'Alpha').default_value = 1.0
        self.inputs.new('NodeSocketFloat', 'Factor').default_value = 1.0
        self.inputs.new('NodeSocketColor', 'Slot Color').default_value = \
            (1.0, 0.0, 1.0, 1.0)
        self.outputs.new('NodeSocketColor', 'Color')
        docs = {
            'Base': "The channel's own colour before this slot",
            'Color': "The texture's colour output, used as tcol when "
                     "the texture yields RGB",
            'Intensity': "The texture intensity (a Fac output): the "
                         "per-pixel factor for intensity textures",
            'Alpha': "The texture's alpha: the per-pixel factor for "
                     "RGB textures",
            'Factor': "The influence slider (0..1 in BI for colours)",
            'Slot Color': "The MTex colour swatch: tcol whenever the "
                          "texture yields no RGB (BI's pink default)",
        }
        for s in self.inputs:
            d = docs.get(s.name)
            if d:
                try:
                    s.description = d
                except (AttributeError, TypeError):
                    pass

    def draw_buttons(self, context, layout):
        layout.prop(self, 'blend', text="")


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


class HALCYON_PixelateNode(Node, HalcyonNodeBase):
    """Snap a coordinate to a coarse texel grid, for chunky low-res texturing"""

    bl_idname = 'HALCYON_PixelateNode'
    bl_label = "Pixelate"
    bl_icon = 'MOD_REMESH'

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = ("The coordinate to snap. Unlinked means the UV map, "
                         "since fat texels live in texture space")
        self.inputs.new('NodeSocketFloat', 'Pixels X').default_value = 64.0
        self.inputs.new('NodeSocketFloat', 'Pixels Y').default_value = 64.0
        z = self.inputs.new('NodeSocketFloat', 'Pixels Z')
        z.default_value = 0.0
        z.description = "0 leaves the third axis untouched (2D pixelation)"
        self.outputs.new('NodeSocketVector', 'Vector')
        self.outputs[0].description = (
            "Each axis snapped to the centre of its cell. Feed any texture's "
            "Vector input for instant fat texels")


class HALCYON_ScrollNode(Node, HalcyonNodeBase):
    """Animated UV transform: the scrolling water, lava and conveyor trick"""

    bl_idname = 'HALCYON_ScrollNode'
    bl_label = "UV Scroll"
    bl_icon = 'ANIM'

    animate: BoolProperty(name="Animate", default=True)
    fps: IntProperty(
        name="Steps Per Second", default=0, min=0, max=60,
        description="0 scrolls smoothly. Above 0 the clock advances in "
                    "steps, the way texture animation looked at the era's "
                    "frame rates -- 15 is the classic choppy water")

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "The coordinate to move. Unlinked means the UV map"
        self.inputs.new('NodeSocketFloat', 'Scroll X').default_value = 0.1
        self.inputs.new('NodeSocketFloat', 'Scroll Y').default_value = 0.0
        s = self.inputs.new('NodeSocketFloat', 'Spin')
        s.default_value = 0.0
        s.description = ("Rotation about the (0.5, 0.5) UV centre, in turns "
                         "per second")
        self.outputs.new('NodeSocketVector', 'Vector')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')
        layout.prop(self, 'fps')


class HALCYON_ScanlinesNode(Node, HalcyonNodeBase):
    """Darken alternate lines across a surface -- an in-scene CRT screen"""

    bl_idname = 'HALCYON_ScanlinesNode'
    bl_label = "Scanlines"
    bl_icon = 'ALIGN_JUSTIFY'

    animate: BoolProperty(
        name="Roll", default=False,
        description="Drift the lines upward over time, the way a set rolls "
                    "when the vertical hold is off")

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.8, 0.8, 0.8, 1.0)
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = ("Where the lines live. Unlinked means the UV map -- "
                         "a television's scanlines belong to ITS screen, "
                         "not the camera's")
        self.inputs.new('NodeSocketFloat', 'Lines').default_value = 240.0
        self.inputs.new('NodeSocketFloat', 'Darkness').default_value = 0.4
        self.inputs.new('NodeSocketFloat', 'Thickness').default_value = 0.5
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_PaletteNode(Node, HalcyonNodeBase):
    """Snap a colour to the nearest entry of a period hardware palette"""

    bl_idname = 'HALCYON_PaletteNode'
    bl_label = "Hardware Palette"
    bl_icon = 'COLOR'

    palette: EnumProperty(name="Palette", items=(
        ('EGA', "EGA (16)", "The 16 colours of the EGA default palette"),
        ('C64', "C64 (16)", "The Commodore 64's 16 colours"),
        ('CGA', "CGA (4)", "CGA palette 1, high intensity: black, cyan, "
                           "magenta, white"),
        ('GAMEBOY', "Game Boy (4)", "The DMG's four shades of green"),
        ('GRAY4', "Grayscale (4)", "Four grey levels"),
        ('GRAY16', "Grayscale (16)", "Sixteen grey levels"),
        ('RGB332', "RGB 3-3-2 (256)", "Each channel crushed to 3-3-2 bits "
                                      "-- the byte-per-pixel truecolour "
                                      "compromise"),
    ), default='EGA')

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.8, 0.8, 0.8, 1.0)
        self.inputs.new('NodeSocketFloat', 'Mix').default_value = 1.0
        self.outputs.new('NodeSocketColor', 'Color')
        idx = self.outputs.new('NodeSocketFloat', 'Index')
        idx.description = ("The chosen entry's position, 0 to 1 -- drive a "
                           "Color Ramp with it for palette remapping")

    def draw_buttons(self, context, layout):
        layout.prop(self, 'palette', text="")


class HALCYON_ColorCycleNode(Node, HalcyonNodeBase):
    """Rotate a ramp phase over time -- Mark Ferrari's colour cycling"""

    bl_idname = 'HALCYON_ColorCycleNode'
    bl_label = "Color Cycle"
    bl_icon = 'FILE_REFRESH'

    animate: BoolProperty(name="Animate", default=True)

    def init(self, context):
        f = self.inputs.new('NodeSocketFloat', 'Fac')
        f.default_value = 0.0
        f.description = ("The phase to rotate -- typically a texture's Fac, "
                         "with a Color Ramp after this node")
        self.inputs.new('NodeSocketFloat', 'Speed').default_value = 0.5
        s = self.inputs.new('NodeSocketFloat', 'Steps')
        s.default_value = 0.0
        s.description = ("0 cycles smoothly. Above 0 the phase advances in "
                         "that many discrete steps per revolution -- the "
                         "palette-register waterfall")
        self.outputs.new('NodeSocketFloat', 'Fac')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_FlipbookNode(Node, HalcyonNodeBase):
    """Play an N-by-M sprite sheet as an animated texture"""

    bl_idname = 'HALCYON_FlipbookNode'
    bl_label = "Flipbook"
    bl_icon = 'RENDER_ANIMATION'

    animate: BoolProperty(name="Animate", default=True)

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "The coordinate to map into one cell. Unlinked " \
                        "means the UV map"
        self.inputs.new('NodeSocketFloat', 'Columns').default_value = 4.0
        self.inputs.new('NodeSocketFloat', 'Rows').default_value = 4.0
        r = self.inputs.new('NodeSocketFloat', 'Rate')
        r.default_value = 8.0
        r.description = "Cells per second. Fire and explosion sheets of " \
                        "the era ran 8 to 15"
        o = self.inputs.new('NodeSocketFloat', 'Cell Offset')
        o.default_value = 0.0
        o.description = "Which cell to start from -- or, with Animate " \
                        "off, which cell to hold"
        out = self.outputs.new('NodeSocketVector', 'Vector')
        out.description = ("Feed an Image Texture's Vector. Cells read "
                           "left to right, top row first, wrapping at "
                           "the end")

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_UVWaveNode(Node, HalcyonNodeBase):
    """Sine-warp a coordinate -- the underwater and heat-haze wobble"""

    bl_idname = 'HALCYON_UVWaveNode'
    bl_label = "UV Wave"
    bl_icon = 'MOD_WAVE'

    animate: BoolProperty(name="Animate", default=True)

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "The coordinate to wobble. Unlinked means the UV map"
        self.inputs.new('NodeSocketFloat', 'Amplitude X').default_value = 0.02
        self.inputs.new('NodeSocketFloat', 'Amplitude Y').default_value = 0.02
        self.inputs.new('NodeSocketFloat', 'Frequency').default_value = 8.0
        self.inputs.new('NodeSocketFloat', 'Speed').default_value = 1.0
        self.outputs.new('NodeSocketVector', 'Vector')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_HalftoneNode(Node, HalcyonNodeBase):
    """A rotated dot screen whose dots grow where the input darkens"""

    bl_idname = 'HALCYON_HalftoneNode'
    bl_label = "Halftone"
    bl_icon = 'LIGHTPROBE_SPHERE'

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.5, 0.5, 0.5, 1.0)
        c.description = "The shade the dots reproduce (Rec.601 luma -- " \
                        "the NTSC weights)"
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "Where the screen lives. Unlinked means the UV map"
        self.inputs.new('NodeSocketFloat', 'Dots').default_value = 24.0
        a = self.inputs.new('NodeSocketFloat', 'Angle')
        a.default_value = 45.0
        a.description = "Screen angle in degrees. Newsprint runs its " \
                        "black plate at 45"
        self.inputs.new('NodeSocketColor', 'Ink Color').default_value = \
            (0.05, 0.05, 0.05, 1.0)
        self.inputs.new('NodeSocketColor', 'Paper Color').default_value = \
            (0.95, 0.93, 0.88, 1.0)
        self.outputs.new('NodeSocketColor', 'Color')
        self.outputs.new('NodeSocketFloat', 'Fac')


class HALCYON_ThresholdNode(Node, HalcyonNodeBase):
    """Cut a value into 0 or 1 at a level, with an optional soft edge"""

    bl_idname = 'HALCYON_ThresholdNode'
    bl_label = "Threshold"
    bl_icon = 'IPO_CONSTANT'

    def init(self, context):
        self.inputs.new('NodeSocketFloat', 'Fac').default_value = 0.5
        self.inputs.new('NodeSocketFloat', 'Level').default_value = 0.5
        s = self.inputs.new('NodeSocketFloat', 'Smooth')
        s.default_value = 0.0
        s.description = "Width of the soft edge around the level. " \
                        "0 is a hard cut"
        self.outputs.new('NodeSocketFloat', 'Fac')


class HALCYON_QuantizeNode(Node, HalcyonNodeBase):
    """Posterize for a single value: snap a Fac to discrete steps"""

    bl_idname = 'HALCYON_QuantizeNode'
    bl_label = "Quantize"
    bl_icon = 'SEQ_HISTOGRAM'

    def init(self, context):
        self.inputs.new('NodeSocketFloat', 'Fac').default_value = 0.5
        self.inputs.new('NodeSocketFloat', 'Steps').default_value = 4.0
        out = self.outputs.new('NodeSocketFloat', 'Fac')
        out.description = ("The cel-band helper: quantize a lighting or "
                           "texture Fac before it drives a Color Ramp")


RAMP_SPACES = (
    ('RGB', "RGB", "Straight-line blend in linear RGB -- the classic, "
     "with its muddy middles between saturated complements"),
    ('OKLAB', "OKLab", "Blend in the OKLab perceptual space: even "
     "lightness, no muddy middles, hues that pass where you expect"),
    ('OKLCH', "OKLCh", "OKLab in polar form: lightness and chroma blend "
     "straight while HUE rotates the short way round -- rainbow ramps "
     "without grey valleys"),
    ('HSV', "HSV", "Blend hue, saturation and value separately -- the "
     "paint-program ramp, vivid and slightly lawless"),
)


class HALCYON_RampNode(Node, HalcyonNodeBase):
    """A multi-stop colour ramp that blends in a chosen colour SPACE.

    Blender's own Color Ramp mixes in RGB and nothing else; this one adds
    OKLab, OKLCh and HSV, with up to six stops whose positions live on the
    node and whose colours are sockets -- so a stop's colour can be driven
    by another node.
    """

    bl_idname = 'HALCYON_RampNode'
    bl_label = "Color Ramp (Spaces)"
    bl_icon = 'COLOR'
    bl_width_default = 200

    def _update(self, context):
        self.refresh_stops()

    space: EnumProperty(
        name="Space", items=RAMP_SPACES, default='OKLAB', update=_update,
        description="The colour space the blend walks through. The stops "
                    "themselves are always plain colours; only the path "
                    "between them changes")
    stops: IntProperty(
        name="Stops", default=2, min=2, max=6, update=_update,
        description="How many colour stops the ramp uses. Each stop is a "
                    "socket below, with its position alongside")
    positions: FloatVectorProperty(
        name="Positions", size=6, min=0.0, max=1.0,
        default=(0.0, 1.0, 0.5, 0.5, 0.5, 0.5),
        description="Where each stop sits along the ramp, 0 to 1. Stops "
                    "are blended in position order")
    easing: EnumProperty(
        name="Easing", default='LINEAR', update=_update,
        items=(('LINEAR', "Linear", "Even blend between stops"),
               ('SMOOTH', "Smooth", "Ease in and out of every stop"),
               ('CONSTANT', "Constant", "Hold each stop until the next -- "
                "hard bands, the palette look")),
        description="The blend profile between neighbouring stops")

    _STOP_DEFAULTS = ((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0),
                      (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0),
                      (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0))

    def init(self, context):
        f = self.inputs.new('NodeSocketFloat', 'Fac')
        f.default_value = 0.5
        f.description = "The position sampled along the ramp"
        for i in range(6):
            s = self.inputs.new('NodeSocketColor', f'Color {i + 1}')
            s.default_value = self._STOP_DEFAULTS[i]
            s.description = (f"Stop {i + 1}'s colour. Its position is on "
                             "the node body")
        out = self.outputs.new('NodeSocketColor', 'Color')
        out.description = "The blended colour at Fac, in the chosen space"
        self.outputs.new('NodeSocketFloat', 'Alpha')
        self.refresh_stops()

    def refresh_stops(self):
        for i in range(6):
            name = f'Color {i + 1}'
            for s in self.inputs:
                if s.name == name:
                    s.hide = bool(i >= self.stops and not s.is_linked)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'space', text="")
        layout.prop(self, 'easing', text="")
        layout.prop(self, 'stops')
        col = layout.column(align=True)
        for i in range(int(self.stops)):
            col.prop(self, 'positions', index=i, text=f"Pos {i + 1}")


class HALCYON_BlurNode(Node, HalcyonNodeBase):
    """Blur whatever is plugged in, by re-sampling it at shifted points.

    The input chain is evaluated several times at offsets in the surface
    plane and averaged -- true blur of any texture, procedural or image.
    That re-run is CPU work: a material using Blur shades on the CPU and
    says so in the console.
    """

    bl_idname = 'HALCYON_BlurNode'
    bl_label = "Blur"
    bl_icon = 'PROP_CON'

    taps: EnumProperty(
        name="Quality", default='MEDIUM',
        items=(('FAST', "Fast (5)", "Five taps -- soft, slightly boxy"),
               ('MEDIUM', "Medium (9)", "Nine taps -- clean for most "
                "sizes"),
               ('FINE', "Fine (17)", "Seventeen taps -- smooth at large "
                "sizes, at proportional cost")),
        description="How many shifted evaluations the blur averages. More "
                    "taps stay smooth at larger sizes and cost more")

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.5, 0.5, 0.5, 1.0)
        c.description = "The chain to blur -- any texture or pattern"
        s = self.inputs.new('NodeSocketFloat', 'Size')
        s.default_value = 0.05
        s.description = ("Blur radius, in the texture's own coordinate "
                         "units (a fraction of the 0-1 span)")
        out = self.outputs.new('NodeSocketColor', 'Color')
        out.description = "The average of the shifted evaluations"


class HalcyonBIRampStop(PropertyGroup):
    """One colorband stop of a BI material ramp."""
    position: FloatProperty(
        name="Position", default=0.0, min=0.0, max=1.0,
        description="Where along the band this stop sits")
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        description="The stop's colour; alpha is the blend factor at "
                    "this stop")


class HALCYON_OT_bi_ramp_gradient(bpy.types.Operator):
    """Create the gradient widget for a BI material ramp, seeded from
    the fallback stop rows."""
    bl_idname = 'halcyon.bi_ramp_gradient'
    bl_label = "BI Ramp Gradient"
    bl_options = {'INTERNAL', 'UNDO'}

    node_name: StringProperty()
    ramp: EnumProperty(items=[('DIF', "Diffuse", ""),
                              ('SPEC', "Specular", "")], default='DIF')

    def execute(self, context):
        tree = getattr(context.space_data, 'edit_tree', None)
        node = tree.nodes.get(self.node_name) if tree else None
        if node is None or node.bl_idname != 'HALCYON_BIMaterialNode':
            return {'CANCELLED'}
        which = 'dif' if self.ramp == 'DIF' else 'spec'
        stops = node.ramp_stops(which)
        if not stops:
            stops = [(0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0)]
        node.set_ramp_stops(which, stops, node.ramp_ipo(which))
        return {'FINISHED'}


class HALCYON_OT_bi_ramp_stop(bpy.types.Operator):
    """Add or remove a stop on a BI material ramp."""
    bl_idname = 'halcyon.bi_ramp_stop'
    bl_label = "BI Ramp Stop"
    bl_options = {'INTERNAL', 'UNDO'}

    node_name: StringProperty()
    ramp: EnumProperty(items=[('DIF', "Diffuse", ""),
                              ('SPEC', "Specular", "")], default='DIF')
    action: EnumProperty(items=[('ADD', "Add", ""),
                                ('REMOVE', "Remove", "")], default='ADD')
    index: IntProperty(default=-1)

    def execute(self, context):
        tree = getattr(context.space_data, 'edit_tree', None)
        node = tree.nodes.get(self.node_name) if tree else None
        if node is None or node.bl_idname != 'HALCYON_BIMaterialNode':
            return {'CANCELLED'}
        stops = node.dif_stops if self.ramp == 'DIF' else node.spec_stops
        if self.action == 'ADD':
            s = stops.add()
            s.position = 1.0 if len(stops) > 1 else 0.0
            s.color = (1.0, 1.0, 1.0, 1.0) if len(stops) > 1 \
                else (0.0, 0.0, 0.0, 1.0)
        elif 0 <= self.index < len(stops):
            stops.remove(self.index)
        return {'FINISHED'}


class HALCYON_BIMaterialNode(Node, HalcyonNodeBase):
    """Blender Internal's material panel as one node.

    The second master shader: where the Halcyon Shader packages one
    reflectance model, this keeps Blender Internal's DIFFUSE and
    SPECULAR shader menus independent -- the full 5x5 matrix a single
    model enum can never carry (Oren-Nayar diffuse under a CookTorr
    highlight, Fresnel diffuse under WardIso, all of it). Every branch
    is the transcribed 2.79 formula; the sockets show BI's own labels
    (Hardness, Refr, Slope, Alpha) while carrying master-shader
    identifiers underneath, so the whole bake/frame machinery treats
    it exactly like the master.
    """

    bl_idname = 'HALCYON_BIMaterialNode'
    bl_label = "BI Material"
    bl_icon = 'MATERIAL'
    bl_width_default = 200

    def _update(self, context):
        self.refresh_sockets()

    diff_shader: EnumProperty(
        name="Diffuse", update=_update, default='LAMBERT',
        items=[
            ('LAMBERT', "Lambert", "Plain N.L -- BI's default diffuse"),
            ('OREN_NAYAR', "Oren-Nayar",
             "Rough diffuse, driven by Roughness"),
            ('TOON', "Toon",
             "A hard angular band with its own Size and Smooth"),
            ('MINNAERT', "Minnaert",
             "Rim darkening below Darkness 1, rim brightening above"),
            ('FRESNEL', "Fresnel",
             "Replaces the cosine with BI's fresnel_fac of the light "
             "angle -- grazing lights glow"),
        ])
    spec_shader: EnumProperty(
        name="Specular", update=_update, default='COOKTORR',
        items=[
            ('COOKTORR', "CookTorr",
             "pow(N.H, Hardness) / (0.1 + N.V) -- BI's default, with "
             "its 11x grazing brightening"),
            ('PHONG', "Phong",
             "pow(N.H, Hardness) -- BI's Phong was the half-vector "
             "lobe"),
            ('BLINN', "Blinn",
             "Torrance-Sparrow with BI's refraction-index Fresnel; "
             "Refr drives it"),
            ('TOON', "Toon",
             "A hard angular highlight band with its own Size/Smooth"),
            ('WARDISO', "WardIso",
             "Isotropic Gaussian on the microfacet slope; Slope (rms) "
             "drives the width"),
        ])
    shadeless: BoolProperty(
        name="Shadeless", default=False, update=_update,
        description="Emit the diffuse colour flat, ignoring every "
                    "light -- BI's Shadeless toggle")

    # ---- Shading panel extras
    use_cubic: BoolProperty(
        name="Cubic Interpolation", default=False,
        description="Smoothstep the diffuse term, exactly BI's Cubic "
                    "Interpolation -- softer terminators")
    use_tangent_v: BoolProperty(
        name="Tangent Shading", default=False,
        description="Shade with a per-light fake normal built from the "
                    "surface tangent -- BI's anisotropic strand trick")

    # ---- Transparency panel
    use_transparency: BoolProperty(
        name="Transparency", default=False, update=_update,
        description="Enable the transparency panel; off, the material "
                    "is opaque and Alpha is inert, exactly the greyed "
                    "2.79 panel")
    transp_mode: EnumProperty(
        name="Type", default='Z_TRANSPARENCY', update=_update,
        items=[
            ('Z_TRANSPARENCY', "Z Transparency",
             "Plain alpha blending in depth order"),
            ('RAYTRACE', "Raytrace",
             "Refract what lies behind through Ray IOR, tinted by "
             "Filter (needs Raytracing on in the render settings)"),
        ],
        description="How transparency composites (BI's Mask mode is "
                    "not carried: it masked the sky in a scanline "
                    "world this engine does not reproduce)")

    # ---- Mirror panel
    use_mirror: BoolProperty(
        name="Mirror", default=False, update=_update,
        description="Enable ray-mirror reflection; off, Mirror "
                    "sliders are inert, exactly the greyed 2.79 panel")

    # ---- ramps
    use_ramp_dif: BoolProperty(
        name="Diffuse Ramp", default=False,
        description="Recolour the diffuse by a colorband, exactly "
                    "BI's diffuse ramp")
    ramp_dif_input: EnumProperty(
        name="Input", default='SHADER', items=[
            ('SHADER', "Shader", "The diffuse shader's own value"),
            ('ENERGY', "Energy",
             "The lit energy: shader times lamp times shadow"),
            ('NORMAL', "Normal", "The view angle against the normal"),
            ('RESULT', "Result",
             "The final accumulated diffuse, ramped once after all "
             "lamps"),
        ])
    ramp_dif_blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f"ramp_blend {m}") for m in (
                'MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV', 'DIFF',
                'DARK', 'LIGHT', 'OVERLAY', 'DODGE', 'BURN', 'HUE',
                'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR')])
    ramp_dif_factor: FloatProperty(
        name="Factor", default=1.0, min=0.0, max=1.0,
        description="How strongly the ramp recolours")
    ramp_dif_ipo: EnumProperty(
        name="Interpolation", default='LINEAR', items=[
            ('LINEAR', "Linear", "Straight blends between stops"),
            ('EASE', "Ease", "Smoothstep blends between stops"),
            ('B_SPLINE', "B-Spline", "Smooth curve approaching stops"),
            ('CARDINAL', "Cardinal", "Smooth curve through stops"),
            ('CONSTANT', "Constant", "Hard steps at each stop"),
        ])
    dif_stops: CollectionProperty(type=HalcyonBIRampStop)
    dif_ramp_tex: StringProperty(default='')
    use_ramp_spec: BoolProperty(
        name="Specular Ramp", default=False,
        description="Recolour the specular by a colorband, exactly "
                    "BI's specular ramp")
    ramp_spec_input: EnumProperty(
        name="Input", default='SHADER', items=[
            ('SHADER', "Shader", "The specular shader's own value"),
            ('ENERGY', "Energy",
             "The lit energy: shader times lamp times shadow"),
            ('NORMAL', "Normal", "The view angle against the normal"),
            ('RESULT', "Result",
             "The final accumulated specular, ramped once after all "
             "lamps"),
        ])
    ramp_spec_blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f"ramp_blend {m}") for m in (
                'MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV', 'DIFF',
                'DARK', 'LIGHT', 'OVERLAY', 'DODGE', 'BURN', 'HUE',
                'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR')])
    ramp_spec_factor: FloatProperty(
        name="Factor", default=1.0, min=0.0, max=1.0,
        description="How strongly the ramp recolours")
    ramp_spec_ipo: EnumProperty(
        name="Interpolation", default='LINEAR', items=[
            ('LINEAR', "Linear", "Straight blends between stops"),
            ('EASE', "Ease", "Smoothstep blends between stops"),
            ('B_SPLINE', "B-Spline", "Smooth curve approaching stops"),
            ('CARDINAL', "Cardinal", "Smooth curve through stops"),
            ('CONSTANT', "Constant", "Hard steps at each stop"),
        ])
    spec_stops: CollectionProperty(type=HalcyonBIRampStop)
    spec_ramp_tex: StringProperty(default='')

    # ---- Options panel
    use_mist: BoolProperty(
        name="Use Mist", default=True,
        description="Off, the material ignores the scene fog entirely "
                    "-- BI's Use Mist")
    vcol_paint: BoolProperty(
        name="Vertex Color Paint", default=False,
        description="Vertex colours replace the base colour (a linked "
                    "Color chain wins)")
    vcol_light: BoolProperty(
        name="Vertex Color Light", default=False,
        description="Vertex colours (times their alpha) add to the "
                    "emit term -- BI's extra lighting")
    light_group: StringProperty(
        name="Light Group", default='',
        description="Name of a collection: only its lamps light this "
                    "material")
    light_group_exclusive: BoolProperty(
        name="Exclusive", default=False,
        description="Lamps of this group light ONLY materials naming "
                    "the group")

    # ---- Shadow panel
    shadow_receive: BoolProperty(
        name="Receive", default=True,
        description="Off, shadows never darken this material")
    shadow_cast: BoolProperty(
        name="Cast", default=True,
        description="Off, this material's faces are pulled out of "
                    "every shadow map")
    shadow_cast_only: BoolProperty(
        name="Cast Only", default=False,
        description="Invisible to the camera while still casting "
                    "shadows (and appearing in reflections)")
    shadow_only: BoolProperty(
        name="Shadows Only", default=False,
        description="The shadow catcher: renders black at the mean of "
                    "its lamps' shadow, transparent elsewhere")
    sbias: FloatProperty(
        name="Shadow Bias", default=0.0, min=0.0, max=0.25,
        description="2.79's terminator fix: diffuse from shadowed "
                    "lamps fades out below this N.L threshold "
                    "(phongcorr), hiding the jagged shadow terminator")
    raybias: BoolProperty(
        name="Ray Bias", default=False,
        description="Use the object's Auto Smooth angle as the "
                    "terminator threshold for ray-shadowed lamps on "
                    "smooth faces (MA_RAYBIAS)")
    use_obcolor: BoolProperty(
        name="Object Color", default=False,
        description="Multiply the final shaded colour by each "
                    "object's own Color (and opacity by its alpha "
                    "when transparency is on) -- per-object tinting "
                    "with one shared material")

    # ---- Subsurface Scattering panel (2.79's point-cloud dipole,
    # transcribed from sss.c; CPU only -- the GPU plan refuses by name)
    sss_enable: BoolProperty(
        name="Subsurface Scattering", default=False,
        description="BI's two-pass SSS: a pre-pass renders this "
                    "material's lit surface into a point cloud, and "
                    "the dipole gather replaces the diffuse term. "
                    "Runs on both devices -- the GPU walks the same "
                    "octree from a data texture")
    sss_scale: FloatProperty(
        name="Scale", default=0.1, min=0.0001, max=1000.0,
        description="Object scale in Blender units per 1 real-world "
                    "unit of the radius")
    sss_radius: FloatVectorProperty(
        name="Radius", size=3, default=(1.0, 1.0, 1.0), min=0.0001,
        description="Mean free path per channel: how far red, green "
                    "and blue light travel below the surface")
    sss_color: FloatVectorProperty(
        name="Scattering Color", subtype='COLOR', size=3,
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        description="The reflectance the dipole solves for, per "
                    "channel")
    sss_ior: FloatProperty(
        name="IOR", default=1.3, min=0.1, max=2.0,
        description="Index of refraction (skin is about 1.3-1.4)")
    sss_error: FloatProperty(
        name="Error", default=0.05, min=0.0001, max=10.0,
        description="Hierarchy acceptance threshold: lower is more "
                    "exact and slower, exactly 2.79's Error slider")
    sss_colfac: FloatProperty(
        name="Color Factor", default=1.0, min=0.0, max=1.0,
        description="Blend between the scattering colour and white "
                    "in the dipole's reflectance")
    sss_texfac: FloatProperty(
        name="Texture Factor", default=0.0, min=0.0, max=1.0,
        description="How much surface texture survives the scatter: "
                    "0 keeps the full texture, 1 dissolves it")
    sss_front: FloatProperty(
        name="Front", default=1.0, min=0.0, max=2.0,
        description="Front-scattering weight")
    sss_back: FloatProperty(
        name="Back", default=1.0, min=0.0, max=10.0,
        description="Back-scattering weight (light through thin "
                    "parts)")

    #: (socket kind, BI display label, master-compatible identifier,
    #: default). The identifier is what the evaluator, the bake probe
    #: and the frame assembler match on.
    BI_SOCKETS = (
        ('NodeSocketColor', 'Color', 'Diffuse Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketFloat', 'Intensity', 'Diffuse Level', 0.8),
        ('NodeSocketFloat', 'Roughness', 'Roughness', 0.5),
        ('NodeSocketFloat', 'Darkness', 'Darkness', 1.0),
        ('NodeSocketFloat', 'Toon Size', 'Toon Size', 0.5),
        ('NodeSocketFloat', 'Toon Smooth', 'Toon Smooth', 0.1),
        ('NodeSocketFloat', 'Fresnel', 'BI Fresnel', 0.1),
        ('NodeSocketFloat', 'Fresnel Factor', 'BI Fresnel Factor', 0.5),
        ('NodeSocketColor', 'Specular Color', 'Specular Color',
         (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Specular Intensity', 'Specular Level', 0.5),
        ('NodeSocketFloat', 'Hardness', 'Glossiness', 50.0),
        ('NodeSocketFloat', 'Refr', 'IOR', 4.0),
        ('NodeSocketFloat', 'Slope', 'Slope', 0.1),
        ('NodeSocketFloat', 'Spec Toon Size', 'Spec Toon Size', 0.5),
        ('NodeSocketFloat', 'Spec Toon Smooth', 'Spec Toon Smooth', 0.1),
        ('NodeSocketFloat', 'Emit', 'Emit', 0.0),
        ('NodeSocketFloat', 'Ambient', 'Ambient', 1.0),
        ('NodeSocketFloat', 'Translucency', 'Translucency', 0.0),
        ('NodeSocketFloat', 'Alpha', 'Opacity', 1.0),
        ('NodeSocketFloat', 'Transp Fresnel', 'Transp Fresnel', 0.0),
        ('NodeSocketFloat', 'Transp Blend', 'Transp Blend', 1.25),
        ('NodeSocketFloat', 'Transp Specular', 'Transp Specular', 1.0),
        ('NodeSocketFloat', 'Ray IOR', 'Ray IOR', 1.3),
        ('NodeSocketFloat', 'Filter', 'Filter', 0.0),
        ('NodeSocketFloat', 'Mirror', 'Reflection', 0.0),
        ('NodeSocketColor', 'Mirror Color', 'Reflection Color',
         (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Mirror Fresnel', 'Mirror Fresnel', 0.0),
        ('NodeSocketFloat', 'Mirror Blend', 'Mirror Blend', 1.25),
        ('NodeSocketVector', 'Normal', 'Normal', None),
        ('NodeSocketFloat', 'Bump Strength', 'Bump Strength', 1.0),
        ('NodeSocketFloat', 'Bump Height', 'Bump Height', 0.5),
    )

    BI_DOCS = {
        'Color': "Diffuse colour, BI's base colour",
        'Intensity': "Diffuse intensity (Ref): how much of the colour "
                     "reflects",
        'Roughness': "Oren-Nayar surface roughness",
        'Darkness': "Minnaert darkness: below 1 darkens the rim, above "
                    "1 brightens it",
        'Toon Size': "Angular size of the lit toon band (radians)",
        'Toon Smooth': "Softness of the toon band's edge",
        'Fresnel': "Fresnel diffuse: the gradient term of BI's "
                   "fresnel_fac",
        'Fresnel Factor': "Fresnel diffuse: the power term; 0 disables "
                          "and shades flat",
        'Specular Color': "Highlight colour",
        'Specular Intensity': "Specular intensity (Spec): highlight "
                              "strength",
        'Hardness': "Specular hardness 1-511, exactly BI's slider: the "
                    "cosine exponent of the highlight",
        'Refr': "Blinn's refraction index, driving its Fresnel",
        'Slope': "WardIso's standard deviation of the microfacet "
                 "slope (rms)",
        'Spec Toon Size': "Angular size of the toon highlight",
        'Spec Toon Smooth': "Softness of the toon highlight's edge",
        'Emit': "Emit: the diffuse colour glows by this amount, "
                "textures included, exactly as BI multiplied it",
        'Ambient': "How much of the world's ambient colour this "
                   "material receives (Amb)",
        'Translucency': "Light from behind shows through by this much",
        'Alpha': "Opacity, BI's Alpha slider (Transparency Fresnel "
                 "above 0 REPLACES it, as BI did)",
        'Transp Fresnel': "View-angle transparency: 0 keeps Alpha; "
                          "higher fades the facing surface out",
        'Transp Blend': "Blending of the transparency Fresnel; above "
                        "1 turns the gradient the classic way",
        'Transp Specular': "How much highlights stay opaque on "
                           "transparent areas (BI's spectra)",
        'Ray IOR': "Refraction index for Raytrace transparency (the "
                   "Blinn Refr slider is spectral, not refractive)",
        'Filter': "0 refracts untinted; 1 tints the refraction by the "
                  "base colour",
        'Mirror': "Ray-mirror reflectivity",
        'Mirror Color': "Tint of the mirrored reflection",
        'Mirror Fresnel': "View-angle mirror: 0 reflects flat; higher "
                          "keeps reflection at grazing angles only",
        'Mirror Blend': "Blending of the mirror Fresnel",
        'Normal': "Replacement shading normal",
        'Bump Strength': "How far Bump Height may bend the normal",
        'Bump Height': "Height field to bump the surface with",
    }

    #: sockets every shader pair shows (the Transparency and Mirror
    #: panels add theirs behind their toggles)
    BI_BASE = {'Color', 'Intensity', 'Specular Color',
               'Specular Intensity', 'Emit', 'Ambient', 'Translucency',
               'Normal', 'Bump Strength', 'Bump Height'}
    #: extra sockets per DIFFUSE choice (display names)
    BI_DIFF_EXTRA = {'OREN_NAYAR': {'Roughness'},
                     'MINNAERT': {'Darkness'},
                     'TOON': {'Toon Size', 'Toon Smooth'},
                     'FRESNEL': {'Fresnel', 'Fresnel Factor'}}
    #: extra sockets per SPECULAR choice
    BI_SPEC_EXTRA = {'COOKTORR': {'Hardness'}, 'PHONG': {'Hardness'},
                     'BLINN': {'Hardness', 'Refr'},
                     'TOON': {'Spec Toon Size', 'Spec Toon Smooth'},
                     'WARDISO': {'Slope'}}

    def init(self, context):
        for kind, name, ident, default in self.BI_SOCKETS:
            try:
                sock = self.inputs.new(kind, name, identifier=ident)
            except TypeError:
                # an API without the identifier kwarg: fall back to the
                # identifier AS the name, keeping the machinery correct
                # at the cost of the BI label
                sock = self.inputs.new(kind, ident)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
            doc = self.BI_DOCS.get(name)
            if doc:
                try:
                    sock.description = doc
                except (AttributeError, TypeError):
                    pass
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Shades with the chosen "
            "Blender Internal diffuse and specular pair")
        self.refresh_sockets()

    def ensure_sockets(self):
        """Create any socket this saved instance predates (file load)."""
        have = {s.name for s in self.inputs}
        have |= {getattr(s, 'identifier', s.name) for s in self.inputs}
        for kind, name, ident, default in self.BI_SOCKETS:
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
        if self.shadeless:
            keep = {'Color'}
            if self.use_transparency:
                keep.add('Alpha')
        else:
            keep = set(self.BI_BASE)
            keep |= self.BI_DIFF_EXTRA.get(self.diff_shader, set())
            keep |= self.BI_SPEC_EXTRA.get(self.spec_shader, set())
            if self.use_transparency:
                keep |= {'Alpha', 'Transp Fresnel', 'Transp Blend',
                         'Transp Specular'}
                if self.transp_mode == 'RAYTRACE':
                    keep |= {'Ray IOR', 'Filter'}
            if self.use_mirror:
                keep |= {'Mirror', 'Mirror Color', 'Mirror Fresnel',
                         'Mirror Blend'}
        for sock in self.inputs:
            sock.hide = bool(sock.name not in keep
                             and not sock.is_linked)

    # ---- the gradient widget: a hidden Blend texture datablock owns
    # a real ColorRamp per ramp, so the band edits as a band (the 2.79
    # panel's own control) instead of position/colour rows. The stops
    # collection stays the engine's fallback: export prefers the ramp.
    def ramp_texture(self, which, create=False):
        """The hidden texture whose color_ramp IS this ramp's band."""
        attr = f'{which}_ramp_tex'
        name = getattr(self, attr, '')
        try:
            texes = bpy.data.textures
        except AttributeError:
            return None
        tex = texes.get(name) if name else None
        if tex is None and create:
            import uuid
            name = f'.hal_biramp_{uuid.uuid4().hex[:12]}'
            try:
                tex = texes.new(name, type='BLEND')
                tex.use_color_ramp = True
                tex.use_fake_user = True
                setattr(self, attr, name)
            except Exception:                                   # noqa: BLE001
                return None
        return tex

    def ramp_stops(self, which):
        """[(pos, r, g, b, a), ...] -- the GRADIENT when it exists,
        else the fallback collection. This is what export reads."""
        tex = self.ramp_texture(which)
        ramp = getattr(tex, 'color_ramp', None) if tex is not None \
            else None
        if ramp is not None and len(ramp.elements):
            return sorted(
                (float(e.position), float(e.color[0]), float(e.color[1]),
                 float(e.color[2]), float(e.color[3]))
                for e in ramp.elements)
        stops = getattr(self, 'dif_stops' if which == 'dif'
                        else 'spec_stops')
        return sorted((float(s.position), float(s.color[0]),
                       float(s.color[1]), float(s.color[2]),
                       float(s.color[3])) for s in stops)

    def ramp_ipo(self, which):
        """do_colorband's integer, from the gradient's own dropdown
        when it exists, else the enum prop."""
        ipo_map = {'LINEAR': 0, 'EASE': 1, 'B_SPLINE': 2,
                   'CARDINAL': 3, 'CONSTANT': 4}
        tex = self.ramp_texture(which)
        ramp = getattr(tex, 'color_ramp', None) if tex is not None \
            else None
        if ramp is not None:
            return ipo_map.get(str(getattr(ramp, 'interpolation',
                                           'LINEAR')), 0)
        return ipo_map.get(str(getattr(self, f'ramp_{which}_ipo',
                                       'LINEAR')), 0)

    def set_ramp_stops(self, which, stops, ipotype=0):
        """Write stops into BOTH the gradient and the fallback
        collection -- the importer's entry point."""
        coll = getattr(self, 'dif_stops' if which == 'dif'
                       else 'spec_stops')
        try:
            coll.clear()
            for stop in stops:
                s = coll.add()
                s.position = float(stop[0])
                s.color = (float(stop[1]), float(stop[2]),
                           float(stop[3]), float(stop[4]))
        except Exception:                                       # noqa: BLE001
            pass
        ipo_names = ('LINEAR', 'EASE', 'B_SPLINE', 'CARDINAL',
                     'CONSTANT')
        try:
            setattr(self, f'ramp_{which}_ipo',
                    ipo_names[int(ipotype)]
                    if 0 <= int(ipotype) < 5 else 'LINEAR')
        except (TypeError, ValueError):
            pass
        tex = self.ramp_texture(which, create=True)
        ramp = getattr(tex, 'color_ramp', None) if tex is not None \
            else None
        if ramp is None or not stops:
            return
        try:
            ramp.interpolation = ipo_names[int(ipotype)] \
                if 0 <= int(ipotype) < 5 else 'LINEAR'
        except (TypeError, ValueError):
            pass
        try:
            # a ColorRamp always keeps at least one element: position
            # the first, add the rest, then trim extras
            while len(ramp.elements) > 1:
                ramp.elements.remove(ramp.elements[-1])
            ramp.elements[0].position = float(stops[0][0])
            ramp.elements[0].color = tuple(float(x) for x in stops[0][1:5])
            for stop in stops[1:]:
                e = ramp.elements.new(float(stop[0]))
                e.color = tuple(float(x) for x in stop[1:5])
        except Exception:                                       # noqa: BLE001
            pass

    def _draw_ramp(self, layout, prefix, stops_attr):
        box = layout.box()
        row = box.row(align=True)
        row.prop(self, f'ramp_{prefix}_input', text="")
        row.prop(self, f'ramp_{prefix}_blend', text="")
        box.prop(self, f'ramp_{prefix}_factor')
        tex = self.ramp_texture(prefix)
        if tex is None:
            # first touch: offer the gradient; rows stay as fallback
            op = box.operator('halcyon.bi_ramp_gradient',
                              text="Show Gradient", icon='COLOR')
            op.node_name, op.ramp = self.name, \
                'DIF' if prefix == 'dif' else 'SPEC'
        if tex is not None and hasattr(tex, 'color_ramp'):
            box.template_color_ramp(tex, 'color_ramp', expand=True)
            return
        row = box.row(align=True)
        row.prop(self, f'ramp_{prefix}_ipo', text="")
        stops = getattr(self, stops_attr)
        which = 'DIF' if prefix == 'dif' else 'SPEC'
        for i, s in enumerate(stops):
            row = box.row(align=True)
            row.prop(s, 'position', text="Pos")
            row.prop(s, 'color', text="")
            op = row.operator('halcyon.bi_ramp_stop', text="",
                              icon='REMOVE')
            op.node_name, op.ramp, op.action, op.index = \
                self.name, which, 'REMOVE', i
        op = box.operator('halcyon.bi_ramp_stop', text="Add Stop",
                          icon='ADD')
        op.node_name, op.ramp, op.action = self.name, which, 'ADD'
        if len(stops) == 0:
            box.label(text="No stops: the ramp is inert", icon='INFO')

    def draw_buttons(self, context, layout):
        col = layout.column()
        col.prop(self, 'diff_shader', text="Diffuse")
        col.prop(self, 'use_ramp_dif', toggle=False)
        if self.use_ramp_dif:
            self._draw_ramp(col, 'dif', 'dif_stops')
        col.prop(self, 'spec_shader', text="Specular")
        col.prop(self, 'use_ramp_spec', toggle=False)
        if self.use_ramp_spec:
            self._draw_ramp(col, 'spec', 'spec_stops')
        col.separator()
        row = col.row(align=True)
        row.prop(self, 'shadeless', toggle=True)
        row = col.row(align=True)
        row.prop(self, 'use_cubic', text="Cubic", toggle=True)
        row.prop(self, 'use_tangent_v', text="Tangent", toggle=True)
        col.separator()
        col.prop(self, 'use_transparency')
        if self.use_transparency:
            col.row().prop(self, 'transp_mode', expand=True)
        col.prop(self, 'use_mirror')
        col.separator()
        col.label(text="Options:")
        row = col.row(align=True)
        row.prop(self, 'use_mist', text="Mist", toggle=True)
        row = col.row(align=True)
        row.prop(self, 'vcol_paint', text="VCol Paint", toggle=True)
        row.prop(self, 'vcol_light', text="VCol Light", toggle=True)
        row = col.row(align=True)
        row.prop_search(self, 'light_group', bpy.data, 'collections',
                        text="Light Group")
        if self.light_group:
            col.prop(self, 'light_group_exclusive')
        col.separator()
        col.label(text="Shadow:")
        row = col.row(align=True)
        row.prop(self, 'shadow_receive', text="Receive", toggle=True)
        row.prop(self, 'shadow_cast', text="Cast", toggle=True)
        row = col.row(align=True)
        row.prop(self, 'shadow_cast_only', text="Cast Only", toggle=True)
        row.prop(self, 'shadow_only', text="Shadows Only", toggle=True)
        row = col.row(align=True)
        row.prop(self, 'sbias', text="Bias")
        row.prop(self, 'raybias', text="Ray Bias", toggle=True)
        col.prop(self, 'use_obcolor')
        col.separator()
        col.prop(self, 'sss_enable')
        if self.sss_enable:
            box = col.box()
            box.prop(self, 'sss_ior')
            box.prop(self, 'sss_scale')
            box.prop(self, 'sss_color')
            box.prop(self, 'sss_radius')
            row = box.row(align=True)
            row.prop(self, 'sss_colfac', text="Color")
            row.prop(self, 'sss_texfac', text="Texture")
            row = box.row(align=True)
            row.prop(self, 'sss_front', text="Front")
            row.prop(self, 'sss_back', text="Back")
            box.prop(self, 'sss_error')


NODES = (HALCYON_RampNode, HALCYON_BlurNode,
         HALCYON_ShaderNode, HALCYON_BIMaterialNode,
         HALCYON_BIInfluenceNode, HALCYON_BIRGBBlendNode,
         HALCYON_CodeNode, HALCYON_PosterizeNode,
         HALCYON_DitherNode, HALCYON_DepthCueNode, HALCYON_ScreenInfoNode,
         HALCYON_PixelateNode, HALCYON_ScrollNode, HALCYON_ScanlinesNode,
         HALCYON_PaletteNode, HALCYON_ColorCycleNode, HALCYON_FlipbookNode,
         HALCYON_UVWaveNode, HALCYON_HalftoneNode, HALCYON_ThresholdNode,
         HALCYON_QuantizeNode)
#: HalcyonBIRampStop is a PropertyGroup, not a node: it registers here,
#: BEFORE the node whose CollectionProperty points at it, and stays out
#: of the Add menu (which lists NODES only)
OPERATORS = (HalcyonBIRampStop, HALCYON_OT_bi_ramp_stop,
             HALCYON_OT_bi_ramp_gradient,
             HALCYON_OT_compile_shader, HALCYON_OT_new_shader_text)


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

    #: nodes that OPEN a group get a separator drawn above them
    GROUP_STARTS = ('HALCYON_PosterizeNode', 'HALCYON_PixelateNode')

    def draw(self, context):
        layout = self.layout
        for cls in NODES:
            if cls.bl_idname in self.GROUP_STARTS:
                layout.separator()
            op = layout.operator('node.add_node', text=cls.bl_label,
                                 icon=getattr(cls, 'bl_icon', 'NONE'))
            op.type = cls.bl_idname
            op.use_transform = True
        layout.separator()
        layout.menu('NODE_MT_halcyon_textures', icon='TEXTURE')


_menu_owner = None


def _migrate_master_sockets(_arg=None):
    """load_post: saved master nodes gain any socket their file predates.

    Socket creation is forbidden inside update callbacks (the guard test
    holds refresh_sockets to toggles only); file load is the one place
    topology change is safe, so it happens here -- Bump Height appears on
    old files the moment they open.
    """
    try:
        trees = []
        for mat in bpy.data.materials:
            if getattr(mat, 'use_nodes', False) and mat.node_tree:
                trees.append(mat.node_tree)
        for grp in getattr(bpy.data, 'node_groups', []):
            trees.append(grp)
        for tree in trees:
            for node in getattr(tree, 'nodes', []):
                if getattr(node, 'bl_idname', '') in (
                        'HALCYON_ShaderNode', 'HALCYON_BIMaterialNode'):
                    try:
                        node.ensure_sockets()
                    except Exception:                           # noqa: BLE001
                        pass
    except Exception:                                           # noqa: BLE001
        pass


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
    try:
        hs = bpy.app.handlers.load_post
        if _migrate_master_sockets not in hs:
            hs.append(_migrate_master_sockets)
        _migrate_master_sockets()          # the already-open file too
    except Exception:                                           # noqa: BLE001
        pass


def unregister():
    try:
        hs = bpy.app.handlers.load_post
        while _migrate_master_sockets in hs:
            hs.remove(_migrate_master_sockets)
    except Exception:                                           # noqa: BLE001
        pass
    from . import pattern_nodes
    pattern_nodes.unregister()
    from .. import compat
    compat.unregister_node_menu(draw_add_menu, _menu_owner)
    for cls in (NODE_MT_halcyon_add,) + tuple(reversed(NODES + OPERATORS)):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass
