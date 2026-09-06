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
                # Blender's own rule, verbatim: a callback for items
                # may NOT be combined with a default -- registration
                # dies with "'default' cannot be set when 'items' is a
                # function", swallowed into "EnumProperty could not
                # register (see previous error)". The 1.62.0 enable
                # failure, now caught at test time.
                if kw.get('default') is not None:
                    raise ValueError(
                        "EnumProperty(...): 'default' cannot be set "
                        "when 'items' is a function (Blender refuses "
                        "this at registration)")
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
            'prepend': classmethod(lambda cls, fn: None),
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


#: Blender's RNA validation counts EVERY named parameter of a callback --
#: defaulted extras included. `init(self, context, _ins=ins)` is legal
#: Python and registers fine in a naive stub, then Blender refuses the
#: class: 'expected Node, X class "init" function to have 2 args, found
#: 4' (the R216 field paste). The stub now runs the same check, per base
#: type, so the suite catches the disease on ANY class, forever.
_CALLBACK_ARGS = (
    ('Node', {'init': 2, 'copy': 2, 'free': 1, 'update': 1,
              'draw_label': 1, 'draw_buttons': 3, 'draw_buttons_ext': 3,
              'poll': 2}),
    ('NodeSocket', {'draw': 5, 'draw_color': 3}),
    ('Operator', {'execute': 2, 'invoke': 3, 'modal': 3, 'draw': 2,
                  'check': 2, 'cancel': 2, 'poll': 2}),
    ('Menu', {'draw': 2, 'poll': 2}),
    ('Panel', {'draw': 2, 'draw_header': 2, 'poll': 2}),
    ('RenderEngine', {'render': 2, 'update': 3, 'view_update': 3,
                      'view_draw': 3}),
)


def _validate_callback_args(cls):
    import types as _t
    base_map = {'Node': Node, 'NodeSocket': NodeSocket,
                'Operator': Operator, 'Menu': Menu, 'Panel': Panel,
                'RenderEngine': RenderEngine}
    table = None
    base_name = None
    for name, tab in _CALLBACK_ARGS:
        if issubclass(cls, base_map[name]):
            table = tab
            base_name = name
            break
    if table is None:
        return
    for fn_name, want in table.items():
        raw = None
        for c in cls.__mro__:
            if c in base_map.values() or c is object:
                break                       # the stub bases define none
            if fn_name in c.__dict__:
                raw = c.__dict__[fn_name]
                break
        if raw is None:
            continue
        if isinstance(raw, (classmethod, staticmethod)):
            raw = raw.__func__
        if not isinstance(raw, _t.FunctionType):
            continue
        found = raw.__code__.co_argcount
        if found != want:
            raise TypeError(
                f'validating class: expected {base_name}, {cls.__name__} '
                f'class "{fn_name}" function to have {want} args, '
                f'found {found}')


def _validate_icon(cls):
    """R244: Blender validates a class's bl_icon against its icon enum
    at register_class ('validating class: enum "MOD_WOOD" not found in
    (...)' was the field's enable failure). The stub reads the same enum
    (blender_icons.ICONS, the field's own paste) and refuses the same way."""
    icon = getattr(cls, 'bl_icon', None)
    if icon is None:
        return
    from .blender_icons import ICONS
    if icon not in ICONS:
        raise ValueError(f'validating class: enum "{icon}" not found in the '
                         f'Blender 5.2 icon enum ({cls.__name__}.bl_icon)')


def register_class(cls):
    _validate_callback_args(cls)
    _validate_icon(cls)
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


def user_resource(kind, path='', create=False):
    # Blender's own scripts folder. The stub keeps it out of the way, in temp.
    import os
    import tempfile
    base = os.path.join(tempfile.gettempdir(), 'halcyon-fake-scripts', path)
    if create:
        os.makedirs(base, exist_ok=True)
    return base


utils.user_resource = user_resource

class _Timers:
    """bpy.app.timers, pumpable by hand: the test IS the main loop."""

    def __init__(self):
        self.fns = []

    def register(self, fn, first_interval=0.0, persistent=False):
        if fn not in self.fns:
            self.fns.append(fn)

    def unregister(self, fn):
        if fn in self.fns:
            self.fns.remove(fn)

    def is_registered(self, fn):
        return fn in self.fns

    def pump(self):
        """Run each registered timer once; a None return unregisters it,
        exactly as Blender's main loop would."""
        for fn in list(self.fns):
            if fn() is None:
                self.unregister(fn)


app = types.SimpleNamespace(version=(5, 2, 0), background=True,
                            timers=_Timers())
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
