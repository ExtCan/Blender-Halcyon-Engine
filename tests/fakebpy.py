"""A minimal bpy stand-in.

Not a Blender emulator -- just enough surface to import the add-on's modules and
catch typos, bad enum defaults and registration-time mistakes without needing
Blender installed. Anything it can't check is called out in the README.
"""

import sys
import types


class _Prop:
    def __init__(self, **kw):
        self.kw = kw
        self.default = kw.get('default')
        self.items = kw.get('items')

    def __call__(self, *a, **k):
        return self


def _mkprop(name):
    def factory(**kw):
        p = _Prop(**kw)
        p.kind = name
        if name == 'EnumProperty':
            items = kw.get('items')
            if callable(items):
                pass
            elif items is not None:
                idents = [i[0] for i in items]
                d = kw.get('default')
                if d is not None and d not in idents:
                    raise ValueError(
                        f"EnumProperty default {d!r} not in items {idents[:6]}...")
        return p
    return factory


props = types.ModuleType('bpy.props')
for _n in ('BoolProperty', 'IntProperty', 'FloatProperty', 'StringProperty',
           'EnumProperty', 'FloatVectorProperty', 'IntVectorProperty',
           'BoolVectorProperty', 'PointerProperty', 'CollectionProperty'):
    setattr(props, _n, _mkprop(_n))


class _Base:
    bl_idname = ''
    bl_label = ''
    COMPAT_ENGINES = set()

    def __init__(self, *a, **k):
        pass

    @classmethod
    def poll(cls, ctx):
        return True


class PropertyGroup(_Base):
    pass


class Panel(_Base):
    pass


class Operator(_Base):
    pass


class Menu(_Base):
    pass


class Node(_Base):
    pass


class NodeSocket(_Base):
    pass


class RenderEngine(_Base):
    pass


class Text(_Base):
    pass


class Scene(_Base):
    pass


class Material(_Base):
    pass


class Light(_Base):
    pass


class World(_Base):
    pass


class _Types(types.ModuleType):
    def __init__(self):
        super().__init__('bpy.types')
        self.PropertyGroup = PropertyGroup
        self.Panel = Panel
        self.Operator = Operator
        self.Menu = Menu
        self.Node = Node
        self.NodeSocket = NodeSocket
        self.RenderEngine = RenderEngine
        self.Text = Text
        self.Scene = Scene
        self.Material = Material
        self.Light = Light
        self.World = World
        self.NODE_MT_add = type('NODE_MT_add', (Menu,), {
            'append': classmethod(lambda cls, fn: None),
            'remove': classmethod(lambda cls, fn: None)})

    def __getattr__(self, name):
        # menus are appended to by name, so every stubbed class needs the
        # append/prepend/remove trio -- VIEW3D_MT_add is reached this way
        cls = type(name, (_Base,), {
            'COMPAT_ENGINES': set(),
            'append': classmethod(lambda cls, fn: None),
            'prepend': classmethod(lambda cls, fn: None),
            'remove': classmethod(lambda cls, fn: None)})
        setattr(self, name, cls)
        return cls


utils = types.ModuleType('bpy.utils')
_registered = []


def register_class(cls):
    _registered.append(cls)
    ann = getattr(cls, '__annotations__', {})
    for k, v in ann.items():
        if not isinstance(v, _Prop):
            raise TypeError(f"{cls.__name__}.{k} is not a bpy property: {v!r}")
    return cls


def unregister_class(cls):
    if cls in _registered:
        _registered.remove(cls)


utils.register_class = register_class
utils.unregister_class = unregister_class

app = types.SimpleNamespace(version=(5, 2, 0), background=True)
class _Collection:
    def __init__(self, factory=None):
        self._items = []
        self._factory = factory or (lambda *a, **k: types.SimpleNamespace())

    def new(self, *a, **k):
        item = self._factory(*a, **k)
        self._items.append(item)
        return item

    def remove(self, item):
        if item in self._items:
            self._items.remove(item)


data = types.SimpleNamespace(texts=types.SimpleNamespace(new=lambda n: Text()),
                             materials=_Collection(),
                             meshes=_Collection(),
                             objects=_Collection(),
                             lights=_Collection())
context = types.SimpleNamespace(engine='HALCYON_RENDER')

bpy = types.ModuleType('bpy')
bpy.props = props
bpy.types = _Types()
bpy.utils = utils
bpy.app = app
bpy.data = data
bpy.context = context

gpu = types.ModuleType('gpu')
gpu.types = types.SimpleNamespace(Buffer=object, GPUTexture=object)
gpu_extras = types.ModuleType('gpu_extras')
gpu_presets = types.ModuleType('gpu_extras.presets')
gpu_presets.draw_texture_2d = lambda *a, **k: None
gpu_extras.presets = gpu_presets


def install():
    sys.modules['bpy'] = bpy
    sys.modules['bpy.props'] = props
    sys.modules['bpy.types'] = bpy.types
    sys.modules['bpy.utils'] = utils
    sys.modules['gpu'] = gpu
    sys.modules['gpu_extras'] = gpu_extras
    sys.modules['gpu_extras.presets'] = gpu_presets
    return bpy
