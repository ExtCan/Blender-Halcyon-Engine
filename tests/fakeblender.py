"""A Blender stand-in complete enough to *run* the add-on, not just import it.

`fakebpy` proved the modules import and register. That caught typos and bad
enum defaults, and it caught nothing at all about whether the engine actually
works, because the engine was never executed: six bugs shipped through it in a
row, every one of them a control wired up at one end and not the other.

This goes one level further. It builds a scene -- an object with a real mesh,
a material with a node tree, a light and a camera -- hands it to
`HalcyonRenderEngine.render()` through a fake depsgraph, and captures what
comes back out of `begin_result`/`end_result`. That is the whole path: the
property group, `to_settings`, the exporter, the renderer, the post chain, the
delivery, and the render passes.

It is not Blender and never will be. It cannot catch a segfault, a driver
quirk or an RNA lifetime bug. What it does catch is the thing that kept
getting through: a setting that never arrives, a pass that is never written, a
material whose model is lost on the way.
"""

import types

import numpy as np

from . import fakebpy


# --------------------------------------------------------------- properties


def live(cls, **overrides):
    """An instance of a registered PropertyGroup with its defaults filled in.

    Blender materialises a property's default the moment the group exists.
    The stub leaves the declaration objects in place, so nothing that reads a
    setting could ever have been tested -- which is exactly the layer the bugs
    were in.
    """
    obj = types.SimpleNamespace()
    for name, prop in getattr(cls, '__annotations__', {}).items():
        kw = getattr(prop, 'kw', {})
        default = kw.get('default')
        items = kw.get('items')
        if default is None and items is not None and not callable(items):
            default = items[0][0]
        if default is None and getattr(prop, 'kind', '') == 'PointerProperty':
            default = None
        setattr(obj, name, default)
    for k, v in overrides.items():
        setattr(obj, k, v)
    # the methods the add-on calls on the group, bound to this stand-in
    for name in ('to_settings',):
        fn = getattr(cls, name, None)
        if fn is not None:
            setattr(obj, name, fn.__get__(obj, type(obj)))
    return obj


# ------------------------------------------------------------------- meshes


class _Collection(list):
    """A bpy_prop_collection stand-in: a list that can foreach_get."""

    def __init__(self, items, fields):
        super().__init__(items)
        self._fields = fields

    def foreach_get(self, attr, out):
        vals = []
        for item in self:
            v = getattr(item, attr)
            if hasattr(v, '__len__'):
                vals.extend(v)
            else:
                vals.append(v)
        arr = np.asarray(vals, dtype=out.dtype)
        out[:arr.size] = arr


class _Data(list):
    def foreach_get(self, attr, out):
        vals = []
        for item in self:
            v = getattr(item, attr)
            vals.extend(v) if hasattr(v, '__len__') else vals.append(v)
        arr = np.asarray(vals, dtype=out.dtype)
        out[:arr.size] = arr


class FakeMesh:
    """A triangulated mesh in the shape `export._mesh_arrays` expects."""

    def __init__(self, verts, tris, materials=(), uvs=None, normals=None):
        verts = np.asarray(verts, np.float32)
        tris = np.asarray(tris, np.int32)
        self.vertices = _Collection(
            [types.SimpleNamespace(co=tuple(v), normal=(0.0, 0.0, 1.0))
             for v in verts], ('co', 'normal'))
        # one loop per triangle corner, so loops and corners line up
        loops, loop_tris = [], []
        for t in tris:
            base = len(loops)
            for vi in t:
                loops.append(types.SimpleNamespace(vertex_index=int(vi),
                                                   normal=(0.0, 0.0, 1.0)))
            loop_tris.append((base, base + 1, base + 2))
        self.loops = _Collection(loops, ('vertex_index', 'normal'))
        self.loop_triangles = _Collection(
            [types.SimpleNamespace(loops=lt, material_index=0)
             for lt in loop_tris], ('loops', 'material_index'))
        if normals is None:
            e1 = verts[tris[:, 1]] - verts[tris[:, 0]]
            e2 = verts[tris[:, 2]] - verts[tris[:, 0]]
            fn = np.cross(e1, e2)
            ln = np.linalg.norm(fn, axis=1, keepdims=True)
            fn = fn / np.where(ln < 1e-12, 1.0, ln)
            normals = np.repeat(fn, 3, axis=0)
        self.corner_normals = _Data(
            [types.SimpleNamespace(vector=tuple(n)) for n in normals])
        if uvs is None:
            uvs = np.zeros((len(loops), 2), np.float32)
        self.uv_layers = [types.SimpleNamespace(
            data=_Data([types.SimpleNamespace(uv=tuple(u)) for u in uvs]))]
        self.color_attributes = []
        self.materials = list(materials)
        self.polygons = self.loop_triangles

    def calc_loop_triangles(self):
        return self.loop_triangles


def cube_mesh(size=2.0, materials=()):
    h = size * 0.5
    v = np.array([[-h, -h, -h], [h, -h, -h], [h, h, -h], [-h, h, -h],
                  [-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]], np.float32)
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    tris = []
    for a, b, c, d in quads:
        tris += [(a, b, c), (a, c, d)]
    return FakeMesh(v, np.asarray(tris, np.int32), materials)


def grid_mesh(size=4.0, divisions=12, materials=()):
    """Dense enough that a wireframe has to cope with small triangles."""
    n = divisions + 1
    xs = np.linspace(-size * 0.5, size * 0.5, n)
    v = np.array([[x, y, 0.0] for y in xs for x in xs], np.float32)
    tris = []
    for r in range(divisions):
        for c in range(divisions):
            a = r * n + c
            tris += [(a, a + 1, a + n + 1), (a, a + n + 1, a + n)]
    return FakeMesh(v, np.asarray(tris, np.int32), materials)


def glyph_mesh(rings=20, materials=(), faces=True):
    """A mesh shaped the way a text object's conversion makes one.

    Flat, coplanar, a hole through the middle, slivers and a couple of
    exactly-degenerate triangles where the outline doubles back -- and none
    of the layers a mesh object would carry: no UV map, no colour attribute,
    no material slot. `faces=False` is a text object with its fill turned
    off, which converts to outlines and no surface at all.
    """
    verts, tris = [], []
    for r in (0.5, 0.32):
        for i in range(rings):
            a = i / rings * 2.0 * np.pi
            verts.append((np.cos(a) * r, np.sin(a) * r, 0.0))
    if faces:
        for i in range(rings):
            j = (i + 1) % rings
            tris.append((i, j, rings + i))
            tris.append((j, rings + j, rings + i))
        base = len(verts)
        verts += [(0.6, 0.0, 0.0), (0.6001, 0.0, 0.0), (0.6, 0.4, 0.0)]
        tris += [(base, base + 1, base + 2), (base, base, base + 1),
                 (base + 2, base + 2, base + 2)]
    me = FakeMesh(np.asarray(verts, np.float32),
                  np.asarray(tris, np.int32).reshape(-1, 3), materials)
    me.uv_layers = []
    me.color_attributes = []
    me.materials = list(materials)
    return me


class _Freed(Exception):
    """Reading a mesh Blender has already freed -- a segfault, in real life."""


class _Guarded:
    """A mesh that answers normally until its owner frees it, then raises.

    Blender's `to_mesh()` hands back a mesh owned by the *object*: the next
    `to_mesh()` on that object destroys it, and so does `to_mesh_clear()`.
    Nothing in Python enforces that, so a use-after-free shows up as a crash
    in Blender and as nothing at all in a test. This makes it show up.
    """

    def __init__(self, inner):
        object.__setattr__(self, '_inner', inner)
        object.__setattr__(self, '_alive', True)

    def __getattr__(self, name):
        if not object.__getattribute__(self, '_alive'):
            raise _Freed(f"read .{name} of a mesh that was already freed")
        return getattr(object.__getattribute__(self, '_inner'), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, '_inner'), name, value)

    def _kill(self):
        object.__setattr__(self, '_alive', False)


def geometry_object(name, mesh_factory, matrix=None, kind='FONT'):
    """An object that converts to a mesh under Blender's real lifetime rules.

    `mesh_factory` is called afresh for every conversion, exactly as Blender
    builds a new temporary each time, and the previous result is invalidated.
    """
    state = {'live': None}

    def to_mesh():
        if state['live'] is not None:
            state['live']._kill()
        g = _Guarded(mesh_factory())
        state['live'] = g
        return g

    def to_mesh_clear():
        if state['live'] is not None:
            state['live']._kill()
            state['live'] = None

    m = np.eye(4, dtype=np.float32) if matrix is None else \
        np.asarray(matrix, np.float32)
    return types.SimpleNamespace(
        name=name, type=kind, color=(1, 1, 1, 1), pass_index=0,
        visible_camera=True, visible_shadow=True, is_holdout=False,
        to_mesh=to_mesh, to_mesh_clear=to_mesh_clear,
        matrix_world=m, data=None, halcyon=None)


def depsgraph_of(props_mod, objects, **settings):
    """A depsgraph carrying exactly the objects handed in."""
    scene = types.SimpleNamespace(
        halcyon=live(props_mod.HalcyonSettings, **settings), frame_current=1,
        world=None, camera=None,
        unit_settings=types.SimpleNamespace(scale_length=1.0),
        view_settings=types.SimpleNamespace(view_transform='Standard'),
        render=types.SimpleNamespace(
            film_transparent=False, threads_mode='AUTO', threads=1,
            resolution_x=120, resolution_y=90, resolution_percentage=100,
            use_freestyle=False, fps=24, pixel_aspect_x=1.0,
            pixel_aspect_y=1.0, engine='HALCYON_RENDER'),
        objects=list(objects))

    def instances():
        for o in objects:
            yield types.SimpleNamespace(object=o, matrix_world=_Mat(o.matrix_world),
                                        show_self=True, is_instance=False)

    return types.SimpleNamespace(scene=scene, scene_eval=scene,
                                 object_instances=instances(),
                                 view_layer=types.SimpleNamespace(
                                     name='ViewLayer'))


# ------------------------------------------------------------- node trees


class FakeSocket:
    def __init__(self, name, kind='VALUE', default=0.0):
        self.name = name
        self.identifier = name
        self.type = kind
        self.default_value = default
        self.is_linked = False
        self.links = []
        self.halcyon_uniform = None
        self.halcyon_is_image = False
        self.halcyon_image_key = None
        self.halcyon_key = None


class FakeNode:
    def __init__(self, name, bl_idname, **props):
        self.name = name
        self.bl_idname = bl_idname
        self.mute = False
        self.inputs = []
        self.outputs = []
        self.image = None
        for k, v in props.items():
            setattr(self, k, v)

    def input(self, name):
        for s in self.inputs:
            if s.name == name:
                return s
        return None


class FakeTree:
    def __init__(self, nodes):
        self.nodes = list(nodes)

    def link(self, from_node, out_name, to_node, in_name):
        src = next(s for s in from_node.outputs if s.name == out_name)
        dst = to_node.input(in_name)
        dst.is_linked = True
        dst.links = [types.SimpleNamespace(from_node=from_node, from_socket=src)]


def halcyon_material(name, model='PHONG', toon_steps=2,
                     diffuse=(0.8, 0.75, 0.35, 1.0)):
    """A material whose node tree is a Halcyon Shader into a Material Output."""
    shader = FakeNode('Halcyon Shader', 'HALCYON_ShaderNode',
                      model=model, toon_steps=toon_steps)
    shader.inputs = [FakeSocket('Diffuse Color', 'RGBA', diffuse),
                     FakeSocket('Diffuse Level', 'VALUE', 1.0),
                     FakeSocket('Specular Level', 'VALUE', 0.4),
                     FakeSocket('Glossiness', 'VALUE', 30.0),
                     FakeSocket('Opacity', 'VALUE', 1.0)]
    shader.outputs = [FakeSocket('Surface', 'SHADER', None)]
    out = FakeNode('Material Output', 'ShaderNodeOutputMaterial', target='ALL')
    out.inputs = [FakeSocket('Surface', 'SHADER', None),
                  FakeSocket('Displacement', 'VECTOR', (0.0, 0.0, 0.0))]
    tree = FakeTree([shader, out])
    tree.link(shader, 'Surface', out, 'Surface')

    mat = types.SimpleNamespace(
        name=name, name_full=name, use_nodes=True, node_tree=tree,
        diffuse_color=(diffuse[0], diffuse[1], diffuse[2], 1.0),
        metallic=0.0, roughness=0.4, specular_intensity=0.5,
        blend_method='OPAQUE', halcyon=None)
    return mat


# ------------------------------------------------------------ render result


class FakePass:
    def __init__(self, name, channels):
        self.name = name
        self.channels = channels
        self.data = None

        class _Rect:
            def __init__(self, owner):
                self._owner = owner

            def foreach_set(self, flat):
                self._owner.data = np.asarray(flat, np.float32).copy()

        self.rect = _Rect(self)


class FakeLayer:
    def __init__(self):
        self.passes = {}


class FakeResult:
    def __init__(self):
        self.layers = [FakeLayer()]


# ------------------------------------------------------------------ scene


def build_scene(props_mod, material=None, mesh=None, **settings):
    """A scene the add-on can be pointed at, with live Halcyon settings."""
    hs = live(props_mod.HalcyonSettings, **settings)
    mat = material if material is not None else halcyon_material('Mat')
    if getattr(mat, 'halcyon', None) is None:
        mat.halcyon = live(props_mod.HalcyonMaterialSettings)
    me = mesh if mesh is not None else cube_mesh(materials=[mat])
    me.materials = [mat]

    matrix = np.eye(4, dtype=np.float32)
    ob = types.SimpleNamespace(
        name='Object', type='MESH', color=(1, 1, 1, 1), pass_index=0,
        visible_camera=True, visible_shadow=True, is_holdout=False,
        to_mesh=lambda: me, to_mesh_clear=lambda: None,
        matrix_world=matrix, data=me, halcyon=None)

    light_data = types.SimpleNamespace(
        type='SUN', color=(1.0, 0.96, 0.9), energy=4.0, angle=0.01,
        shadow_soft_size=0.1, spot_size=0.7, spot_blend=0.15,
        size=1.0, size_y=1.0, shape='SQUARE', use_shadow=True,
        cutoff_distance=0.0, use_custom_distance=False, halcyon=None)
    lm = np.eye(4, dtype=np.float32)
    lm[:3, 3] = (3.0, -4.0, 5.0)
    light = types.SimpleNamespace(name='Sun', type='LIGHT', data=light_data,
                                  matrix_world=lm, halcyon=None)

    cam_data = types.SimpleNamespace(
        lens=42.0, sensor_width=36.0, sensor_height=24.0, sensor_fit='AUTO',
        type='PERSP', clip_start=0.1, clip_end=200.0, ortho_scale=6.0,
        shift_x=0.0, shift_y=0.0, dof=types.SimpleNamespace(
            use_dof=False, focus_distance=8.0, focus_object=None,
            aperture_fstop=2.8))
    cm = np.eye(4, dtype=np.float32)
    # looking down -Z after a rotation that puts the cube in frame
    eye = np.array([4.5, -5.5, 3.2], np.float32)
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(fwd, np.array([0, 0, 1.0], np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    cm[:3, 0] = right
    cm[:3, 1] = up
    cm[:3, 2] = -fwd
    cm[:3, 3] = eye
    cam = types.SimpleNamespace(name='Camera', type='CAMERA', data=cam_data,
                                matrix_world=cm, halcyon=None)

    render = types.SimpleNamespace(
        film_transparent=False, threads_mode='AUTO', threads=1,
        resolution_x=120, resolution_y=90, resolution_percentage=100,
        use_freestyle=False, fps=24, pixel_aspect_x=1.0, pixel_aspect_y=1.0,
        engine='HALCYON_RENDER')

    scene = types.SimpleNamespace(
        halcyon=hs, render=render, frame_current=1, world=None, camera=cam,
        unit_settings=types.SimpleNamespace(scale_length=1.0),
        view_settings=types.SimpleNamespace(view_transform='Standard'),
        objects=[ob, light, cam])

    def instances():
        for o in (ob, light, cam):
            yield types.SimpleNamespace(object=o,
                                        matrix_world=_Mat(o.matrix_world),
                                        show_self=True, is_instance=False)

    depsgraph = types.SimpleNamespace(scene=scene, scene_eval=scene,
                                      object_instances=instances(),
                                      view_layer=types.SimpleNamespace(
                                          name='ViewLayer'))
    return depsgraph, scene, hs, mat


class _Mat(np.ndarray):
    """A matrix that answers .copy() the way Blender's does."""

    def __new__(cls, arr):
        return np.asarray(arr, np.float32).view(cls)


# ------------------------------------------------------------------- runner


def run_render(props_mod, engine_mod, **kw):
    """Render one frame through the real engine and capture what it delivers.

    Returns (image, passes, scene) -- the RGBA buffer written into Combined,
    a dict of the extra pass buffers, and the exported Halcyon scene.

    `rig`, if given, is called as rig(eng, depsgraph, bscene) after the
    engine stub is dressed and before render runs -- the hook a test uses
    to bolt on pieces bpy normally provides (frame_set for motion blur,
    a moving light, a ticking clock).
    """
    material = kw.pop('material', None)
    mesh = kw.pop('mesh', None)
    rig = kw.pop('rig', None)
    depsgraph, bscene, hs, _mat = build_scene(props_mod, material, mesh, **kw)

    eng = engine_mod.HalcyonRenderEngine.__new__(engine_mod.HalcyonRenderEngine)
    eng._scene = None
    eng._scene_key = None
    eng._draw_data = None
    eng._last_hash = None
    eng.is_preview = False
    eng.size_x = bscene.render.resolution_x
    eng.size_y = bscene.render.resolution_y

    captured = {'result': None, 'registered': []}

    def begin_result(x, y, w, h):
        res = FakeResult()
        res.layers[0].passes['Combined'] = FakePass('Combined', 4)
        for name, chans in captured['registered']:
            if name != 'Combined':
                res.layers[0].passes[name] = FakePass(name, chans)
        captured['result'] = res
        captured['size'] = (w, h)
        return res

    eng.begin_result = begin_result
    eng.end_result = lambda result: None
    eng.update_stats = lambda *a: None
    eng.update_progress = lambda *a: None
    eng.test_break = lambda: False
    eng.report = lambda kind, msg: captured.setdefault('reports', []).append(msg)
    eng.register_pass = lambda scene, vl, name, chans, ids, kind: \
        captured['registered'].append((name, chans))

    # Blender asks which passes exist before it builds the result
    eng.update_render_passes(bscene, bscene.view_settings)

    if rig is not None:
        rig(eng, depsgraph, bscene)
    eng.render(depsgraph)

    res = captured['result']
    w, h = captured.get('size', (0, 0))
    image = None
    passes = {}
    if res is not None:
        cp = res.layers[0].passes.get('Combined')
        if cp is not None and cp.data is not None:
            image = cp.data.reshape(h, w, 4)
        for name, p in res.layers[0].passes.items():
            if name != 'Combined' and p.data is not None:
                passes[name] = p.data.reshape(h, w, p.channels)
    return image, passes, captured


def install():
    """Install the stub and hand back the modules under test."""
    fakebpy.install()
    import importlib
    bpy = fakebpy.bpy
    if not hasattr(bpy.types, 'UIList'):
        bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    if not hasattr(bpy.types, 'AddonPreferences'):
        bpy.types.AddonPreferences = type('AddonPreferences',
                                          (bpy.types.Panel,), {})
    props_mod = importlib.import_module('halcyon.properties')
    engine_mod = importlib.import_module('halcyon.engine')
    return props_mod, engine_mod
