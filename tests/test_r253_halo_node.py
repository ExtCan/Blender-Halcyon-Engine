"""R253 (1.92.0): the Halo node.

The Material tab's halo kit (R191/R194/R198/R200) becomes
nodes/shader_nodes.HALCYON_HaloNode (Shading family): the exporter
reads the node first and the panel only as the deprecated fallback,
a pre-1.92 file grows the node at load (migrate_halo_material), and
the splat (core/render._draw_halos) gains a Falloff menu, a Blend
menu, Over Everything depth, a camera-distance fade, an anamorphic
Stretch, an outer Glow layer, a per-frame seed and per-halo sizes
from an Attribute / Color Attribute / Particle Info link. Every new
key is proven neutral at its default on the old road (array_equal),
and the node is refused BY NAME on both devices (it never rasterises:
halos are a CPU splat over the readback on either road).

    python -m halcyon.tests.test_r253_halo_node
"""
import os
import re
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from .blender_icons import ICONS as VERIFIED_ICONS
from ..core import render as R

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name
          + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------ fake trees

class _Sock:
    def __init__(self, name, value=None, kind='VALUE'):
        self.name = name
        self.identifier = name
        self.default_value = value
        self.type = kind
        self.is_linked = False
        self.links = []
        self.hide = False
        self.description = ''


class _Link:
    def __init__(self, a, b, an, bn):
        self.from_socket, self.to_socket = a, b
        self.from_node, self.to_node = an, bn


class _Inputs(list):
    def get(self, name):
        return next((s for s in self if s.name == name), None)

    def new(self, kind, name):
        k = {'NodeSocketColor': 'RGBA', 'NodeSocketFloat': 'VALUE',
             'NodeSocketShader': 'SHADER'}.get(kind, 'VALUE')
        s = _Sock(name, kind=k)
        self.append(s)
        return s


class _Node:
    def __init__(self, idn, name=None):
        self.bl_idname = idn
        self.name = name or idn
        self.inputs = _Inputs()
        self.outputs = _Inputs()
        self.location = (0, 0)
        self.label = ''
        self.hide = False
        self.mute = False


class _Links(list):
    def new(self, a, b):
        an = getattr(a, '_owner', None)
        bn = getattr(b, '_owner', None)
        ln = _Link(a, b, an, bn)
        self.append(ln)
        b.is_linked = True
        b.links = [ln]
        a.links = getattr(a, 'links', []) + [ln]
        return ln


class _Nodes(list):
    def __init__(self):
        super().__init__()
        self._n = 0

    def get(self, name):
        return next((n for n in self if n.name == name), None)

    def new(self, idn):
        self._n += 1
        n = _Node(idn, f'{idn}.{self._n:03d}')
        if idn == 'HALCYON_HaloNode':
            _fill_halo_node(n)
        elif idn == 'ShaderNodeValToRGB':
            n.outputs.new('NodeSocketColor', 'Color')
            n.outputs.new('NodeSocketFloat', 'Alpha')
            n.color_ramp = _Ramp()
        elif idn == 'ShaderNodeOutputMaterial':
            n.inputs.new('NodeSocketShader', 'Surface')
            n.inputs.new('NodeSocketShader', 'Volume')
            n.is_active_output = True
        elif idn == 'ShaderNodeAttribute':
            n.attribute_name = ''
            n.outputs.new('NodeSocketColor', 'Color')
            n.outputs.new('NodeSocketFloat', 'Fac')
        elif idn == 'ShaderNodeRGB':
            n.outputs.new('NodeSocketColor', 'Color')
        elif idn == 'ShaderNodeValue':
            n.outputs.new('NodeSocketFloat', 'Value')
        elif idn == 'ShaderNodeParticleInfo':
            n.outputs.new('NodeSocketFloat', 'Index')
            n.outputs.new('NodeSocketFloat', 'Size')
        else:
            n.outputs.new('NodeSocketFloat', 'Value')
        for s in list(n.inputs) + list(n.outputs):
            s._owner = n
        self.append(n)
        return n

    def remove(self, n):
        if n in self:
            super().remove(n)


class _Tree:
    def __init__(self):
        self.nodes = _Nodes()
        self.links = _Links()


class _Ramp:
    def evaluate(self, t):
        return (float(t), 1.0 - float(t), 0.5, 1.0)


def _fill_halo_node(n):
    import importlib
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    N = SN.HALCYON_HaloNode
    for kind, name, d in N.SOCKETS:
        n.inputs.new(kind, name).default_value = d
    n.outputs.new('NodeSocketShader', 'Halo')
    for pname, pdef in N.__annotations__.items():
        setattr(n, pname, pdef.kw.get('default'))
    n.image = None


def _halo_tree(**over):
    """A tree with one Halo node (props/sockets overridden by name) and
    a Material Output; returns (tree, node, out)."""
    tree = _Tree()
    node = tree.nodes.new('HALCYON_HaloNode')
    out = tree.nodes.new('ShaderNodeOutputMaterial')
    tree.links.new(node.outputs[0], out.inputs.get('Surface'))
    for k, v in over.items():
        setattr(node, k, v)
    return tree, node, out


def _hs(**kw):
    base = dict(use_override=False, wire_size=1.0, halo=False,
                strand=False, cast_shadow=True, receive_shadow=True,
                alpha_mode='BLEND', alpha_clip=0.5)
    base.update(kw)
    hs = types.SimpleNamespace(**base)
    hs.is_property_set = lambda k, _s=set(kw.pop('_set', ())): k in _s
    return hs


def _mat(name, tree=None, hs=None, use_nodes=True):
    return types.SimpleNamespace(
        name=name, name_full=name, halcyon=hs or _hs(), node_tree=tree,
        use_nodes=use_nodes, diffuse_color=(0.5, 0.5, 0.5, 1.0),
        metallic=0.0, roughness=0.5)


#: the non-default numbers every road is fed, by NODE name
_NODE_VALUES = dict(
    hardness=20, seed=7, rings=True, ring_count=3, rings_even=True,
    ring_width=2.0, lines=True, line_count=5, line_width=1.5, star=True,
    star_tips=6, shape='HEX', noise=0.3, noise_scale=6.0, bolts=2,
    bolt_width=1.5, rays=4, ray_sharp=12.0, gradient=True,
    gradient_type='ANGULAR', gradient_noise=0.2, aspect=2.0,
    rotation=0.3, rand_hue=0.1, rand_sat=0.2, rand_val=0.3,
    hue_shift=0.25, sat_shift=0.8, val_shift=1.2, pulse=0.2,
    flicker=0.1, spin=1.0, anim_speed=2.0, pulse_speed=3.0,
    flicker_speed=0.5, noise_speed=1.5, bolt_speed=2.5,
    grad_noise_speed=0.75, xalpha=True, soft=True, shaded=True,
    puno=True)
_SOCK_VALUES = {'Color': (0.1, 0.2, 0.3, 1.0), 'Size': 0.8,
                'Alpha': 0.7, 'Add': 0.4,
                'Edge Color': (0.3, 0.1, 0.9, 1.0),
                'Ring Color': (0.9, 0.8, 0.1, 1.0),
                'Line Color': (0.2, 0.9, 0.3, 1.0),
                'Ray Color': (0.5, 0.5, 0.1, 1.0),
                'Bolt Color': (0.1, 0.5, 0.5, 1.0)}


def _panel_hs(**extra):
    """The same numbers on the deprecated panel props."""
    import importlib
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    kw = {'halo': True}
    for pname, kind, target in SN.HALO_PANEL_MAP:
        if kind == 'prop':
            kw[pname] = _NODE_VALUES.get(
                target, SN.HALCYON_HaloNode.__annotations__[target]
                .kw.get('default'))
        else:
            v = _SOCK_VALUES[target]
            kw[pname] = tuple(v[:3]) if isinstance(v, tuple) else v
    kw['halo_image'] = None
    kw.update(extra)
    return _hs(**kw)


def _node_mat(name='HaloA', **extra):
    tree, node, out = _halo_tree()
    for k, v in _NODE_VALUES.items():
        setattr(node, k, v)
    for k, v in _SOCK_VALUES.items():
        node.inputs.get(k).default_value = v
    for k, v in extra.items():
        setattr(node, k, v)
    return _mat(name, tree), tree, node


# ------------------------------------------------------------------ tests

def test_halo_node_registers():
    """The node sits in NODES and in exactly one family with a vetted
    icon; every socket and property explains itself; every shading
    property reaches NODE_PROPS; the panel map names real panel props
    and real node targets, and a migrated material is value-identical
    because the defaults agree."""
    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    import importlib
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    PR = importlib.import_module('halcyon.properties')
    from ..export import NODE_PROPS
    N = SN.HALCYON_HaloNode
    check('the Halo node registers in NODES',
          N in SN.NODES and N.bl_idname == 'HALCYON_HaloNode')
    fams = [t for t, _i, m in SN.MENU_FAMILIES if N in m]
    check('and in the Shading family only', fams == ['Shading'], str(fams))
    check('its icon is vetted', N.bl_icon in VERIFIED_ICONS, N.bl_icon)
    bpy.utils.register_class(N)
    check('fakebpy registers the class (callback arities, icon, props)',
          N in fakebpy._registered)
    names = [s[1] for s in N.SOCKETS]
    miss = [n for n in names if len(SN.HALO_SOCKET_DOCS.get(n, '')) < 40]
    check(f'every one of the {len(names)} sockets is documented',
          not miss, ', '.join(miss))
    ann = N.__annotations__
    thin = [k for k, v in ann.items()
            if len(str(v.kw.get('description') or '')) < 40]
    check(f'every one of the {len(ann)} properties carries a real tooltip',
          not thin, ', '.join(thin))
    thin_i = []
    for k, v in ann.items():
        for it in (v.kw.get('items') or ()):
            if len(it[2]) < 12:
                thin_i.append(f'{k}:{it[0]}')
    check('every enum item carries a real line', not thin_i,
          ', '.join(thin_i))
    listed = set(NODE_PROPS['HALCYON_HaloNode'])
    dead = [k for k, v in ann.items()
            if v.kind in ('EnumProperty', 'IntProperty', 'FloatProperty',
                          'BoolProperty', 'StringProperty')
            and k not in listed and k != 'ui_page']
    check('every shading property reaches export.NODE_PROPS', not dead,
          ', '.join(dead))
    ghost = [p for p in listed if p not in ann]
    check('NODE_PROPS names no property the node lacks', not ghost,
          ', '.join(ghost))
    pann = PR.HalcyonMaterialSettings.__annotations__
    bad = []
    for pname, kind, target in SN.HALO_PANEL_MAP:
        if pname not in pann:
            bad.append(f'panel:{pname}')
        if kind == 'sock' and target not in names:
            bad.append(f'socket:{target}')
        if kind == 'prop' and target not in ann:
            bad.append(f'prop:{target}')
    check(f'the {len(SN.HALO_PANEL_MAP)}-row panel map names real props '
          'and targets', not bad, ', '.join(bad))
    unequal = []
    for pname, kind, target in SN.HALO_PANEL_MAP:
        if kind != 'prop' or pname not in pann or target not in ann:
            continue
        a = pann[pname].kw.get('default')
        b = ann[target].kw.get('default')
        if a != b:
            unequal.append(f'{pname}={a!r} vs {target}={b!r}')
    check('every migrated property default equals the panel default',
          not unequal, ', '.join(unequal))
    sock_d = {s[1]: s[2] for s in N.SOCKETS}
    for pname, sname in (('halo_color', 'Color'), ('halo_size', 'Size'),
                         ('halo_alpha', 'Alpha'), ('halo_add', 'Add'),
                         ('halo_color2', 'Edge Color')):
        a = pann[pname].kw.get('default')
        b = sock_d[sname]
        ok = (tuple(a) == tuple(b[:3])) if isinstance(b, tuple) else a == b
        check(f'{sname} socket default matches the panel', ok, f'{a} {b}')
    check('ui_page is the one UI-only property, paged in draw_buttons',
          'ui_page' in ann and N.draw_buttons.__code__.co_argcount == 3)


def test_halo_node_exports_the_spec():
    """A node material exports the panel road's exact dict (shared
    keys equal) plus the node-only keys; Ray/Bolt Colour only once
    their toggle is on; a linked Color Ramp becomes the 32-entry LUT;
    an Attribute on Size becomes size_source; a Math node on Size
    warns by name and leaves it None."""
    from . import fakebpy
    fakebpy.install()
    from .. import export as EX
    import importlib
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    N = SN.HALCYON_HaloNode

    matA, treeA, nodeA = _node_mat()
    wA = []
    mA = EX.export_material(matA, {}, wA)
    check('a node material exports a halo spec', mA.halo is not None)
    matB = _mat('HaloB', None, _panel_hs())
    mB = EX.export_material(matB, {}, [])
    check('the panel road still exports (the deprecated fallback)',
          mB.halo is not None)
    diff = [k for k in mB.halo if mA.halo.get(k) != mB.halo[k]]
    check(f'the node and panel roads agree on all {len(mB.halo)} shared '
          'keys', mA.halo is not None and mB.halo is not None and not diff,
          ', '.join(f'{k}: {mA.halo.get(k)!r} vs {mB.halo[k]!r}'
                    for k in diff[:4]))
    new = ('falloff', 'ring_inner', 'blend', 'depth_mode', 'fade_near',
           'fade_far', 'stretch', 'stretch_angle', 'glow_size',
           'glow_strength', 'animate_seed', 'size_source')
    missing = [k for k in new if k not in mA.halo]
    check('the node-only keys are present', not missing, ', '.join(missing))
    check('and at their neutral defaults',
          mA.halo['falloff'] == 'BI' and mA.halo['ring_inner'] == 0.6
          and mA.halo['blend'] == 'ADD_SLIDER'
          and mA.halo['depth_mode'] == 'ZBUFFER'
          and mA.halo['fade_near'] == 0.0 and mA.halo['fade_far'] == 0.0
          and mA.halo['stretch'] == 1.0 and mA.halo['stretch_angle'] == 0.0
          and mA.halo['glow_size'] == 0.0
          and mA.halo['glow_strength'] == 1.0
          and mA.halo['animate_seed'] is False
          and mA.halo['size_source'] is None
          and 'glow_color' not in mA.halo)
    check("'ray_color' / 'bolt_color' are absent until their Own Colour "
          "toggles", 'ray_color' not in mA.halo and 'bolt_color' not in mA.halo)
    matC, _t, nodeC = _node_mat('HaloC', ray_own_color=True,
                                glow_own_color=True)
    nodeC.inputs.get('Glow Color').default_value = (0.2, 0.4, 0.6, 1.0)
    mC = EX.export_material(matC, {}, [])
    check('Ray Own Colour ships the Ray Color socket',
          mC.halo.get('ray_color') == (0.5, 0.5, 0.1)
          and 'bolt_color' not in mC.halo)
    check('Glow Own Colour ships the Glow Color socket',
          mC.halo.get('glow_color') == (0.2, 0.4, 0.6))
    check('the counts gate on their toggles (3 rings, 5 lines, 6 tips)',
          mA.halo['rings'] == 3 and mA.halo['lines'] == 5
          and mA.halo['star_points'] == 6)
    matD, _t, nodeD = _node_mat('HaloD', rings=False, lines=False,
                                star=False)
    mD = EX.export_material(matD, {}, [])
    check('toggles off zero the counts', mD.halo['rings'] == 0
          and mD.halo['lines'] == 0 and mD.halo['star_points'] == 0)

    # the ramp
    matE, treeE, nodeE = _node_mat('HaloE')
    ramp = treeE.nodes.new('ShaderNodeValToRGB')
    treeE.links.new(ramp.outputs[0], nodeE.inputs.get('Ramp'))
    mE = EX.export_material(matE, {}, [])
    lut = mE.halo.get('ramp')
    check('a Color Ramp linked into Ramp exports the 32x4 LUT',
          lut is not None and len(lut) == 32 and len(lut[0]) == 4
          and lut[0][0] == 0.0 and lut[31][0] == 1.0)
    matF, treeF, nodeF = _node_mat('HaloF', gradient=False)
    treeF.links.new(treeF.nodes.new('ShaderNodeValToRGB').outputs[0],
                    nodeF.inputs.get('Ramp'))
    mF = EX.export_material(matF, {}, [])
    check('the ramp ships only with Gradient on', 'ramp' not in mF.halo)
    check('the node is in the tree serialisation too (the evaluator '
          'names it)', mA.graph is not None
          and any(n.get('bl_idname') == 'HALCYON_HaloNode'
                  for n in mA.graph['nodes'].values()))

    # constants from RGB / Value nodes, strangers warn by name
    matG, treeG, nodeG = _node_mat('HaloG')
    rgb = treeG.nodes.new('ShaderNodeRGB')
    rgb.outputs[0].default_value = (0.9, 0.1, 0.2, 1.0)
    treeG.links.new(rgb.outputs[0], nodeG.inputs.get('Color'))
    val = treeG.nodes.new('ShaderNodeValue')
    val.outputs[0].default_value = 1.7
    treeG.links.new(val.outputs[0], nodeG.inputs.get('Size'))
    wG = []
    mG = EX.export_material(matG, {}, wG)
    check('an RGB node drives Color and a Value node drives Size',
          mG.halo['color'] == (0.9, 0.1, 0.2) and mG.halo['size'] == 1.7
          and not [w for w in wG if 'Halo' in w], str(wG))
    matH, treeH, nodeH = _node_mat('HaloH')
    treeH.links.new(treeH.nodes.new('ShaderNodeMath').outputs[0],
                    nodeH.inputs.get('Alpha'))
    wH = []
    mH = EX.export_material(matH, {}, wH)
    check('a Math node on Alpha warns by name and the socket value is '
          'used', mH.halo['alpha'] == 0.7
          and any('Halo Alpha' in w and 'ShaderNodeMath' in w
                  and 'HaloH' in w for w in wH), str(wH))

    # per-halo sizes
    matI, treeI, nodeI = _node_mat('HaloI')
    attr = treeI.nodes.new('ShaderNodeAttribute')
    attr.attribute_name = 'radius'
    treeI.links.new(attr.outputs[1], nodeI.inputs.get('Size'))
    wI = []
    mI = EX.export_material(matI, {}, wI)
    check("an Attribute on Size exports size_source ('ATTR', 'radius') "
          'and keeps the socket size',
          mI.halo['size_source'] == ('ATTR', 'radius')
          and mI.halo['size'] == 0.8
          and not [w for w in wI if 'Halo' in w], str(wI))
    matJ, treeJ, nodeJ = _node_mat('HaloJ')
    treeJ.links.new(treeJ.nodes.new('ShaderNodeMath').outputs[0],
                    nodeJ.inputs.get('Size'))
    wJ = []
    mJ = EX.export_material(matJ, {}, wJ)
    check('a Math node on Size warns by name and leaves size_source None',
          mJ.halo['size_source'] is None and mJ.halo['size'] == 0.8
          and any('Halo Size' in w and 'ShaderNodeMath' in w for w in wJ),
          str(wJ))
    matK, treeK, nodeK = _node_mat('HaloK')
    pi = treeK.nodes.new('ShaderNodeParticleInfo')
    treeK.links.new(pi.outputs[1], nodeK.inputs.get('Size'))
    mK = EX.export_material(matK, {}, [])
    check("Particle Info > Size exports ('PARTICLE',)",
          mK.halo['size_source'] == ('PARTICLE',))
    check('a material without a Halo node exports no halo spec',
          EX.export_material(_mat('Plain', _Tree()), {}, []).halo is None)


def test_node_wins_and_panel_is_fallback():
    """hs.halo on with different values AND a node: the node's values
    export; hs.halo on and no node: the panel values, the pre-R253
    dict bit for bit; neither: no spec."""
    from . import fakebpy
    fakebpy.install()
    from .. import export as EX
    mat, tree, node = _node_mat('Both')
    mat.halcyon = _panel_hs(halo_hardness=99, halo_seed=200,
                            halo_size=5.0, halo_color=(0.0, 0.0, 1.0))
    m = EX.export_material(mat, {}, [])
    check('with both, the node wins', m.halo['hardness'] == 20
          and m.halo['seed'] == 7 and m.halo['size'] == 0.8
          and m.halo['color'] == (0.1, 0.2, 0.3))
    hs = _panel_hs(halo_hardness=99, halo_seed=200)
    mat2 = _mat('PanelOnly', _Tree(), hs)
    m2 = EX.export_material(mat2, {}, [])
    PRE_R253_KEYS = {
        'size', 'hardness', 'add', 'alpha', 'color', 'seed', 'rings',
        'lines', 'star_points', 'ring_color', 'line_color', 'xalpha',
        'soft', 'shaded', 'puno', 'shape', 'line_width', 'ring_width',
        'gradient', 'color2', 'rand_hue', 'rand_sat', 'rand_val',
        'pulse', 'flicker', 'spin', 'anim_speed', 'aspect', 'rotation',
        'noise', 'noise_scale', 'bolts', 'bolt_width', 'rays',
        'ray_sharp', 'rings_even', 'gradient_type', 'gradient_noise',
        'hue_shift', 'sat_shift', 'val_shift', 'pulse_speed',
        'flicker_speed', 'noise_speed', 'bolt_speed', 'grad_noise_speed'}
    check('without a node, the panel values export',
          m2.halo is not None and m2.halo['hardness'] == 99
          and m2.halo['seed'] == 200)
    check('and the panel dict is the pre-R253 dict, key for key',
          m2.halo is not None and set(m2.halo) == PRE_R253_KEYS,
          str(set(m2.halo or {}) ^ PRE_R253_KEYS))
    check('the factored panel function is what the fallback returns',
          m2.halo == EX._halo_spec_from_panel(hs, mat2, {}))
    mat3 = _mat('Neither', _Tree())
    check('neither: m.halo is None',
          EX.export_material(mat3, {}, []).halo is None)
    check('material_is_halo: node / panel / neither',
          EX.material_is_halo(mat) and EX.material_is_halo(mat2)
          and not EX.material_is_halo(mat3)
          and not EX.material_is_halo(None))
    slot = types.SimpleNamespace(halcyon=types.SimpleNamespace(halo=True))
    check('material_is_halo accepts a slot stub carrying only .halcyon',
          EX.material_is_halo(slot))
    stub = types.SimpleNamespace(name='Stub', halcyon=_hs(),
                                 node_tree=None, diffuse_color=(1, 1, 1, 1),
                                 metallic=0.0, roughness=0.5)
    check('a node_tree=None stub material exports with no halo',
          EX.export_material(stub, {}, []).halo is None
          and EX.halo_node_of(stub) is None)


def test_halo_migration_on_a_fake_tree():
    """A material with the panel kit set and no node grows one Halo
    node with every socket value and property copied, the Own Colour
    flags from is_property_set, the hidden ramp relinked, the output
    on the unlinked Surface, and hs.halo cleared; idempotent; a
    non-halo material is untouched; a linked Surface is kept."""
    from . import fakebpy
    fakebpy.install()
    import importlib
    SN = importlib.import_module('halcyon.nodes.shader_nodes')

    tree = _Tree()
    out = tree.nodes.new('ShaderNodeOutputMaterial')
    ramp = tree.nodes.new('ShaderNodeValToRGB')
    ramp.name = '__halo_ramp'
    hs = _panel_hs(_set=('halo_ray_color',))
    mat = types.SimpleNamespace(name='Legacy', halcyon=hs, node_tree=tree,
                                use_nodes=False)
    node = SN.migrate_halo_material(mat)
    check('migrate_halo_material creates one Halo node and turns nodes on',
          node is not None and node.bl_idname == 'HALCYON_HaloNode'
          and mat.use_nodes is True
          and sum(n.bl_idname == 'HALCYON_HaloNode' for n in tree.nodes) == 1)
    bad = []
    for pname, kind, target in SN.HALO_PANEL_MAP:
        want = getattr(hs, pname)
        if kind == 'sock':
            got = node.inputs.get(target).default_value
            if isinstance(want, tuple):
                want = tuple(want) + (1.0,)
            if got != want:
                bad.append(f'{target}: {got!r} != {want!r}')
        else:
            got = getattr(node, target)
            if got != want:
                bad.append(f'{target}: {got!r} != {want!r}')
    check(f'every one of the {len(SN.HALO_PANEL_MAP)} panel values is '
          'copied', not bad, '; '.join(bad[:4]))
    check('ray_own_color follows is_property_set (True), bolt_own_color '
          'stays False', node.ray_own_color is True
          and node.bolt_own_color is False)
    rs = node.inputs.get('Ramp')
    check('the hidden __halo_ramp widget is linked into Ramp',
          rs.is_linked and rs.links[0].from_socket is ramp.outputs[0])
    surf = out.inputs.get('Surface')
    check('the Halo output lands on the unlinked Surface',
          surf.is_linked and surf.links[0].from_socket is node.outputs[0])
    check('hs.halo clears: the node is the one source of truth',
          hs.halo is False)
    n_before = len(tree.nodes)
    check('a second call is a no-op',
          SN.migrate_halo_material(mat) is None and len(tree.nodes) == n_before)
    hs.halo = True
    check('even with hs.halo forced back on, an existing node blocks a '
          'second node', SN.migrate_halo_material(mat) is None
          and len(tree.nodes) == n_before)
    tree2 = _Tree()
    mat2 = types.SimpleNamespace(name='Plain', halcyon=_hs(),
                                 node_tree=tree2, use_nodes=False)
    check('a non-halo material is untouched',
          SN.migrate_halo_material(mat2) is None and not tree2.nodes
          and mat2.use_nodes is False)
    tree3 = _Tree()
    out3 = tree3.nodes.new('ShaderNodeOutputMaterial')
    other = tree3.nodes.new('ShaderNodeBsdfDiffuse')
    tree3.links.new(other.outputs[0], out3.inputs.get('Surface'))
    mat3 = types.SimpleNamespace(name='Linked', halcyon=_panel_hs(),
                                 node_tree=tree3, use_nodes=True)
    n3 = SN.migrate_halo_material(mat3)
    s3 = out3.inputs.get('Surface')
    check('a Surface already linked keeps its link (the node is still '
          'made and found)', n3 is not None
          and s3.links[0].from_socket is other.outputs[0]
          and SN.halo_node_of(tree3) is n3)
    # halo_node_of: muted nodes are skipped, the Surface-linked one wins
    tree4 = _Tree()
    a = tree4.nodes.new('HALCYON_HaloNode')
    b = tree4.nodes.new('HALCYON_HaloNode')
    out4 = tree4.nodes.new('ShaderNodeOutputMaterial')
    tree4.links.new(b.outputs[0], out4.inputs.get('Surface'))
    check('with two nodes, the one on the Surface wins',
          SN.halo_node_of(tree4) is b)
    b.mute = True
    check('a muted node is skipped', SN.halo_node_of(tree4) is a)
    a.mute = True
    check('all muted: no halo node', SN.halo_node_of(tree4) is None)
    # ensure_sockets grows a missing socket by name on a stub
    stub = types.SimpleNamespace(inputs=_Inputs(),
                                 SOCKETS=SN.HALCYON_HaloNode.SOCKETS)
    for kind, name, d in SN.HALCYON_HaloNode.SOCKETS[:3]:
        stub.inputs.new(kind, name).default_value = d
    SN.HALCYON_HaloNode.ensure_sockets(stub)
    check('ensure_sockets adds the sockets a saved node lacks, with their '
          'tips', [s.name for s in stub.inputs]
          == [s[1] for s in SN.HALCYON_HaloNode.SOCKETS]
          and stub.inputs.get('Glow Size').default_value == 0.0
          and stub.inputs.get('Size').description
          == SN.HALO_SOCKET_DOCS['Size'])


def test_new_options_splat_headless():
    """The old road with every new key at its default is bitwise the
    pre-R253 picture; every new option changes it; OVER draws what the
    z-buffer hides; SCREEN never exceeds white where ADDITIVE does;
    Stretch widens along its angle; per-halo sizes are read; two
    renders of the full new kit are identical."""
    from ..core.scene import Material

    st = base_settings(128, 96)
    st.transparency = 'NONE'
    PT = np.array([[0.0, 0.6, 1.9], [-0.4, 1.0, 1.4]], np.float32)

    def frame(pts=None, sizes=None, frame_no=None, **hk):
        sc = demo_scene(st, with_texture=False)
        spec = {'size': 0.6, 'hardness': 50, 'add': 0.5, 'alpha': 1.0,
                'color': (1.0, 0.7, 0.3), 'seed': 3, 'rings': 0,
                'lines': 0, 'star_points': 0, 'xalpha': False,
                'soft': False, 'shaded': False}
        spec.update(hk)
        m = Material(name='G', index=len(sc.materials), halo=spec)
        sc.materials.append(m)
        grp = {'mat': m.index, 'pos': PT if pts is None else pts}
        if sizes is not None:
            grp['sizes'] = np.asarray(sizes, np.float32)
        sc.halos = [grp]
        if frame_no is not None:
            sc.frame = frame_no
        return sc, R.render(sc, st)

    NEUTRAL = dict(falloff='BI', ring_inner=0.6, blend='ADD_SLIDER',
                   depth_mode='ZBUFFER', fade_near=0.0, fade_far=0.0,
                   stretch=1.0, stretch_angle=0.0, glow_size=0.0,
                   glow_strength=1.0, glow_color=None, animate_seed=False,
                   size_source=None)
    sc0, base = frame()
    plain = R.render(demo_scene(st, with_texture=False), st)
    _s, neutral = frame(**NEUTRAL)
    check('(a) the new keys at their defaults are bitwise-neutral',
          bool(np.array_equal(neutral, base)))
    _s, rich = frame(rings=3, lines=8, rays=6, bolts=2, star_points=5,
                     shape='HEX', gradient=True, color2=(0.1, 0.3, 1.0),
                     soft=True, xalpha=True, aspect=2.0, rotation=0.4)
    _s, rich_n = frame(rings=3, lines=8, rays=6, bolts=2, star_points=5,
                       shape='HEX', gradient=True, color2=(0.1, 0.3, 1.0),
                       soft=True, xalpha=True, aspect=2.0, rotation=0.4,
                       **NEUTRAL)
    check('(a) and around the live kit (soft, aspect, rotation, trim)',
          bool(np.array_equal(rich_n, rich)))

    def differs(a, b):
        return float(np.abs(a - b)[:, :, :3].max()) > 1e-4

    # (b) every option speaks
    for fo in ('GAUSSIAN', 'LINEAR', 'QUADRATIC', 'HARD', 'RING_ONLY'):
        _s, v = frame(falloff=fo)
        check(f'(b) falloff {fo} changes the halo', differs(v, base))
    _s, ring_a = frame(falloff='RING_ONLY', ring_inner=0.3)
    _s, ring_b = frame(falloff='RING_ONLY', ring_inner=0.8)
    check('(b) Ring Inner moves the ring', differs(ring_a, ring_b))
    for bm in ('ALPHA', 'ADDITIVE', 'SCREEN'):
        _s, v = frame(blend=bm)
        check(f'(b) blend {bm} changes the halo (Add 0.5 base)',
              differs(v, base))
    _s, add0 = frame(add=0.0)
    _s, alpha_m = frame(add=0.0, blend='ALPHA')
    _s, alpha_m2 = frame(add=0.9, blend='ALPHA')
    check('(b) ADD_SLIDER at Add 0 is ALPHA bit for bit (whatever Add '
          'says under ALPHA)', bool(np.array_equal(add0, alpha_m))
          and bool(np.array_equal(add0, alpha_m2)))
    _s, add1 = frame(add=1.0)
    _s, addi = frame(add=0.0, blend='ADDITIVE')
    check('(b) ADD_SLIDER at Add 1 is ADDITIVE bit for bit',
          bool(np.array_equal(add1, addi)))
    _s, stre = frame(stretch=3.0)
    check('(b) Stretch 3 changes the halo', differs(stre, base))
    _s, stre_a = frame(stretch=3.0, stretch_angle=1.2)
    check('(b) Stretch Angle turns the streak', differs(stre_a, stre))
    _s, glow = frame(glow_size=2.5)
    check('(b) Glow Size 2.5 adds the outer layer', differs(glow, base))
    _s, glow_w = frame(glow_size=2.5, glow_strength=0.3)
    check('(b) Glow Strength scales it', differs(glow_w, glow))
    _s, glow_c = frame(glow_size=2.5, glow_color=(0.1, 0.2, 1.0))
    check('(b) Glow Color tints it', differs(glow_c, glow))
    cam = sc0.camera.matrix_world
    eye = np.asarray(cam, np.float32)[:3, 3]
    d0 = float(np.linalg.norm(PT[0] - eye))
    _s, fade = frame(fade_near=d0 - 1.0, fade_far=d0 + 1.0)
    check('(b) a fade bracketing the halo dims it', differs(fade, base)
          and differs(fade, plain))
    _s, gone = frame(fade_near=0.0, fade_far=0.5)
    check('(b) Fade Far below the halo distance blanks it to the plain '
          'frame', bool(np.array_equal(gone, plain)))
    _s, s1 = frame(rings=4, animate_seed=True, frame_no=1)
    _s, s2 = frame(rings=4, animate_seed=True, frame_no=2)
    _s, s1n = frame(rings=4, frame_no=1)
    _s, s2n = frame(rings=4, frame_no=2)
    check('(b) Animate Seed re-rolls the rings per frame (and only then)',
          differs(s1, s2) and bool(np.array_equal(s1n, s2n)))
    _s, over = frame(depth_mode='OVER')
    check('(b) Over Everything changes the picture', differs(over, base))

    # (c) the buried point
    buried_pt = np.array([[1.4, -0.4, 0.9]], np.float32)
    _s, hidden = frame(pts=buried_pt)
    _s, shown = frame(pts=buried_pt, depth_mode='OVER')
    check('(c) ZBUFFER hides the point inside the cube',
          float(np.abs(hidden - plain)[:, :, :3].max()) < 1e-6)
    check('(c) OVER draws it', differs(shown, plain))
    _s, shown_soft = frame(pts=buried_pt, depth_mode='OVER', soft=True)
    check('(c) OVER with Soft still softens by what it crosses (differs '
          'from the hard overlay)', differs(shown_soft, shown))

    # (d) screen vs additive
    twin = np.array([[0.0, 0.6, 1.9], [0.02, 0.6, 1.9]], np.float32)
    _s, scr = frame(pts=twin, blend='SCREEN', color=(1.0, 1.0, 1.0))
    _s, addv = frame(pts=twin, blend='ADDITIVE', color=(1.0, 1.0, 1.0))
    # the frame under the glows is itself above 1.0 in places (a lit
    # cube top), so the bar is white OR the frame, whichever is higher
    lit_d = np.abs(addv - plain)[:, :, :3].max(axis=2) > 1e-4
    ceil_d = np.maximum(plain[lit_d][:, :3], 1.0)
    check('(d) two overlapping ADDITIVE glows push past white',
          bool(np.any(addv[lit_d][:, :3] > ceil_d + 0.05)),
          f'{addv[lit_d][:, :3].max()} over {plain[lit_d][:, :3].max()}')
    lit_s = np.abs(scr - plain)[:, :, :3].max(axis=2) > 1e-4
    ceil_s = np.maximum(plain[lit_s][:, :3], 1.0)
    check('(d) two overlapping SCREEN glows never do',
          bool(np.all(scr[lit_s][:, :3] <= ceil_s + 1e-6)),
          str((scr[lit_s][:, :3] - ceil_s).max()))

    # (e) the streak's direction
    def bbox(img):
        lit = np.abs(img - plain)[:, :, :3].max(axis=2) > 1e-4
        ys, xs = np.nonzero(lit)
        return (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1)
    one = np.array([[0.0, 0.6, 1.9]], np.float32)
    _s, b1 = frame(pts=one)
    _s, b3 = frame(pts=one, stretch=3.0, stretch_angle=0.0)
    w1, h1 = bbox(b1)
    w3, h3 = bbox(b3)
    check('(e) Stretch along angle 0 widens the lit box horizontally more '
          'than vertically', w3 / w1 > 1.5 and w3 / w1 > h3 / h1 * 1.3,
          f'{w1}x{h1} -> {w3}x{h3}')
    _s, b3v = frame(pts=one, stretch=3.0, stretch_angle=np.pi / 2)
    w3v, h3v = bbox(b3v)
    check('(e) and along pi/2 vertically', h3v / h1 > 1.5
          and h3v / h1 > w3v / w1 * 1.3, f'{w3v}x{h3v}')

    # (f) per-halo sizes
    _s, sz = frame(sizes=[0.2, 1.2])
    _s, sz_sw = frame(sizes=[1.2, 0.2])
    check("(f) a group's 'sizes' array sizes each halo",
          differs(sz, base) and differs(sz, sz_sw))

    # (g) determinism of the whole kit
    kit = dict(falloff='GAUSSIAN', blend='SCREEN', stretch=2.0,
               stretch_angle=0.5, glow_size=2.0, glow_strength=0.7,
               glow_color=(0.2, 0.5, 1.0), fade_near=1.0,
               fade_far=d0 + 2.0, animate_seed=True, rings=3, lines=6,
               rays=4, bolts=2, shaded=True, soft=True)
    _s, k1 = frame(frame_no=3, **kit)
    _s, k2 = frame(frame_no=3, **kit)
    check('(g) the full new kit is deterministic', bool(np.array_equal(k1, k2))
          and differs(k1, base))
    # the helpers at the function level
    d = np.linspace(0.0, 1.0, 11, dtype=np.float32)
    for fo in ('GAUSSIAN', 'LINEAR', 'QUADRATIC', 'HARD', 'RING_ONLY'):
        f = R._halo_falloff(fo, d, 0.6)
        check(f'_halo_falloff {fo}: 1 at the centre (or inside the ring '
              f'rule), 0 at the rim, within [0,1]',
              f[-1] == 0.0 and f.max() <= 1.0 and f.min() >= 0.0
              and (f[0] == 1.0 if fo != 'RING_ONLY' else f[0] == 0.0),
              str(f.tolist()))
    fx, fy = R._halo_stretch(np.float32(3.0), np.float32(0.0), 3.0, 0.0)
    check('_halo_stretch shrinks the frame along the angle (3 -> 1)',
          abs(float(fx) - 1.0) < 1e-6 and abs(float(fy)) < 1e-6)


def test_sizes_from_attributes():
    """_collect_halo_points reads a POINT float attribute named by the
    node's Attribute link into entry['sizes']; a missing attribute
    leaves it absent and warns; a CORNER layer is refused by name;
    _halo_groups_from forwards 'sizes'."""
    from . import fakebpy
    fakebpy.install()
    from .. import export as EX
    from ..core.scene import Material

    verts = np.array([[0, 0, 0], [1, 0, 0.5], [2, 0, 1.5]], np.float32)
    radii = np.array([0.2, 0.7, 1.1], np.float32)

    class _Verts:
        def __len__(self):
            return len(verts)

        def foreach_get(self, name, buf):
            assert name == 'co'
            buf[:] = verts.ravel()

    class _AttrData:
        def __init__(self, vals, key):
            self.vals, self.key = vals, key

        def __len__(self):
            return len(self.vals)

        def foreach_get(self, name, buf):
            assert name == self.key
            buf[:] = np.asarray(self.vals, np.float32).ravel()

    class _Attr:
        def __init__(self, vals, domain='POINT', data_type='FLOAT',
                     key='value'):
            self.domain, self.data_type = domain, data_type
            self.data = _AttrData(vals, key)

    class _Attrs(dict):
        pass

    me = types.SimpleNamespace(
        name='Cloud', vertices=_Verts(),
        attributes=_Attrs(radius=_Attr(radii),
                          corner=_Attr(radii, domain='CORNER')),
        color_attributes=_Attrs(
            paint=_Attr([[r, 0, 0, 1] for r in radii], key='color',
                        data_type='FLOAT_COLOR'),
            face=_Attr([[r, 0, 0, 1] for r in radii], domain='CORNER',
                       key='color')))

    def slot_for(idn, **props):
        tree, node, out = _halo_tree()
        src = tree.nodes.new(idn)
        for k, v in props.items():
            setattr(src, k, v)
        tree.links.new(src.outputs[1] if idn != 'ShaderNodeVertexColor'
                       else src.outputs[0], node.inputs.get('Size'))
        return types.SimpleNamespace(
            name='HaloMat', halcyon=_hs(), node_tree=tree, use_nodes=True)

    mw = np.eye(4, dtype=np.float32)
    w = []
    pts = EX._collect_halo_points(
        me, mw, [slot_for('ShaderNodeAttribute', attribute_name='radius')],
        w)
    check('a POINT float attribute on the Size link fills entry sizes',
          pts is not None and len(pts) == 1
          and np.array_equal(pts[0].get('sizes'), radii) and not w, str(w))
    w = []
    pts = EX._collect_halo_points(
        me, mw, [slot_for('ShaderNodeAttribute', attribute_name='nope')], w)
    check("a missing attribute leaves 'sizes' absent and warns by name",
          pts is not None and 'sizes' not in pts[0]
          and any("'nope'" in x and 'Halo Size' in x for x in w), str(w))
    w = []
    pts = EX._collect_halo_points(
        me, mw, [slot_for('ShaderNodeAttribute', attribute_name='corner')],
        w)
    check('a CORNER attribute is refused by name',
          'sizes' not in pts[0] and any('CORNER' in x for x in w), str(w))
    tree, node, out = _halo_tree()
    vc = tree.nodes.new('ShaderNodeVertexColor')
    vc.layer_name = 'paint'
    vc.outputs.new('NodeSocketColor', 'Color')
    vc.outputs[0]._owner = vc
    tree.links.new(vc.outputs[0], node.inputs.get('Size'))
    slot_vc = types.SimpleNamespace(name='VC', halcyon=_hs(), node_tree=tree,
                                    use_nodes=True)
    w = []
    pts = EX._collect_halo_points(me, mw, [slot_vc], w)
    check("a POINT colour attribute's red channel sizes the halos",
          pts is not None and np.array_equal(pts[0].get('sizes'), radii)
          and not w, str(w))
    vc.layer_name = 'face'
    w = []
    pts = EX._collect_halo_points(me, mw, [slot_vc], w)
    check('a CORNER colour layer is refused by name',
          'sizes' not in pts[0] and any('CORNER' in x for x in w), str(w))
    # the panel stub of test_legacy still collects (and no sizes)
    legacy = types.SimpleNamespace(halcyon=types.SimpleNamespace(
        halo=True, halo_puno=False))
    pts = EX._collect_halo_points(me, mw, [legacy])
    check('the panel-only slot stub still collects its points',
          pts is not None and len(pts) == 1 and 'sizes' not in pts[0]
          and pts[0]['pos'].shape == (3, 3))
    plain = types.SimpleNamespace(halcyon=types.SimpleNamespace(halo=False))
    check('a surface slot collects nothing',
          EX._collect_halo_points(me, mw, [plain]) is None)
    # the node's puno flag is read for the normals
    tree_p, node_p, _o = _halo_tree(puno=True)
    slot_p = types.SimpleNamespace(name='P', halcyon=_hs(), node_tree=tree_p,
                                   use_nodes=True)

    class _VertsN(_Verts):
        def foreach_get(self, name, buf):
            if name == 'co':
                buf[:] = verts.ravel()
            else:
                buf[:] = np.tile([0.0, 0.0, 1.0], len(verts))
    me_n = types.SimpleNamespace(name='N', vertices=_VertsN())
    pts = EX._collect_halo_points(me_n, mw, [slot_p])
    check("the node's Vertex Normal flag collects the normals",
          pts is not None and pts[0].get('normals') is not None
          and pts[0]['normals'].shape == (3, 3))
    # groups forward the sizes
    mats = [Material(name='G', index=0, halo={'seed': 5})]
    grp = EX._halo_groups_from(
        [{'slot': 0, 'pos': verts, 'sizes': radii}], np.array([0]), mats)
    check("_halo_groups_from forwards 'sizes' and walks the seed",
          len(grp) == 1 and np.array_equal(grp[0]['sizes'], radii)
          and grp[0]['seeds'].tolist() == [5, 6, 7])
    grp2 = EX._halo_groups_from([{'slot': 0, 'pos': verts}], np.array([0]),
                                mats)
    check('and None without them', grp2[0]['sizes'] is None)


def test_halo_node_refuses_by_name_on_both_roads():
    """The GLSL emitter refuses the node by name, the CPU evaluator
    names it and returns an empty surface, and capability documents
    the CPU splat without blocking any plan."""
    from ..gpu import emit as EM
    from ..gpu import capability as CAP
    from ..core.nodeeval import GraphEvaluator, ShadeContext, Closure
    from ..core.settings import RenderSettings
    from .test_render import _sk

    graph = {'output': 'out', 'nodes': {
        'halo': {'id': 'halo', 'bl_idname': 'HALCYON_HaloNode',
                 'props': {'hardness': 50}, 'inputs': [
                     _sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1.0]),
                     _sk('Size', 'VALUE', 0.5)],
                 'outputs': [{'name': 'Halo', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {}, 'inputs': [
                    _sk('Surface', 'SHADER', None, ['halo', 0]),
                    _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    ok, why = EM.can_emit(graph)
    check('the GLSL emitter refuses the Halo node', not ok)
    check('by name, with the CPU-splat reason',
          any(w.startswith('HALCYON_HaloNode (a halo material draws no '
                           'faces') for w in why), str(why))
    check('the REFUSED table names both devices',
          'both' in EM.REFUSED['HALCYON_HaloNode']
          and 'CPU' in EM.REFUSED['HALCYON_HaloNode'])
    n = 8
    ctx = ShadeContext(n)
    ctx.N = np.tile(np.array([[0., 0., 1.]], np.float32), (n, 1))
    ctx.I = np.tile(np.array([[0., 0., -1.]], np.float32), (n, 1))
    ctx.P = np.zeros((n, 3), np.float32)
    ctx.uv = np.zeros((n, 2), np.float32)
    ctx.settings = RenderSettings()
    ev = GraphEvaluator(graph, ctx)
    v = ev.eval_output('halo', 0)
    check('the CPU evaluator names the Halo node',
          any(u.startswith('HALCYON_HaloNode (') and 'draws no faces' in u
              for u in ev.unsupported), str(ev.unsupported))
    check('and returns an empty surface for its Halo output',
          isinstance(v, Closure) and not ev.unresolved, str(ev.unresolved))
    check("capability documents 'halo_splat' as NOT_YET on the GPU",
          CAP.FEATURES.get('halo_splat', (None,))[0] == CAP.NOT_YET
          and 'CPU' in CAP.FEATURES['halo_splat'][1])
    check('without blocking any plan', 'halo_splat' not in CAP.BLOCKING
          and CAP.supports('halo_splat', CAP.CPU)
          and not CAP.supports('halo_splat', CAP.GPU))
    check('summary() lists it', any(r[0] == 'halo_splat'
                                    for r in CAP.summary()))
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    gsrc = ''
    for fn in os.listdir(os.path.join(root, 'gpu')):
        if fn.endswith('.py'):
            gsrc += open(os.path.join(root, 'gpu', fn),
                         encoding='utf-8').read()
    check('no gpu/ module splats halos (the CPU pass is the road on both '
          'devices)', '_draw_halos' not in gsrc and 'def e_halo' not in gsrc)
    rsrc = open(os.path.join(root, 'core', 'render.py'),
                encoding='utf-8').read()
    check("the GPU road still releases the frame at 'halos' before the "
          'splat', "_FR.edited(st, 'halos')" in rsrc)


def test_material_tab_and_operator_sources():
    """ui.py carries the operator and the three panel states (node /
    deprecated kit with the legacy prop lines / button); legacy_import
    converts through migrate_halo_material; the registration list
    holds the operator."""
    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    import importlib
    UI = importlib.import_module('halcyon.ui')
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    usrc = open(os.path.join(root, 'ui.py'), encoding='utf-8').read()
    check('the Material tab reads the node through halo_node_of',
          'halo_node_of' in usrc)
    check('the operator is spelled and registered',
          "'halcyon.halo_node'" in usrc
          and hasattr(UI, 'HALCYON_OT_halo_node')
          and UI.HALCYON_OT_halo_node in UI.CLASSES
          and UI.HALCYON_OT_halo_node.bl_idname == 'halcyon.halo_node')
    check('the legacy kit lines survive in the deprecated branch',
          "'halo_size'" in usrc and "'halo_hardness'" in usrc)
    check('the panel names the three states',
          'Add Halo Node' in usrc and 'deprecated' in usrc
          and 'Halo node:' in usrc)
    check("the panel no longer draws the 'halo' checkbox",
          "hcol.prop(hs, 'halo')" not in usrc)
    for cls in (UI.HALCYON_OT_halo_node,):
        bpy.utils.register_class(cls)
    check('the operator registers under the stub (execute/poll arities)',
          UI.HALCYON_OT_halo_node in fakebpy._registered)
    lsrc = open(os.path.join(root, 'legacy_import.py'),
                encoding='utf-8').read()
    check('legacy_import converts the imported halo material to a node',
          'migrate_halo_material' in lsrc)
    psrc = open(os.path.join(root, 'properties.py'), encoding='utf-8').read()
    check("properties keeps every halo_* prop (the deprecated carrier) "
          "and says so", len(re.findall(r'^    halo_[a-z_0-9]*: ', psrc,
                                        re.M)) >= 52
          and 'DEPRECATED (pre-1.92 panel path)' in psrc)
    nsrc = open(os.path.join(root, 'nodes', 'shader_nodes.py'),
                encoding='utf-8').read()
    check('load_post runs the halo migration before the socket pass',
          nsrc.index('migrate_halo_material(mat)')
          < nsrc.index('for tree in trees:'))
    # the operator on a fake material: converts the legacy kit, else
    # makes a bare node, never a second one
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    tree = _Tree()
    tree.nodes.new('ShaderNodeOutputMaterial')
    mat = types.SimpleNamespace(name='M', halcyon=_panel_hs(), node_tree=tree,
                                use_nodes=False)
    ctx = types.SimpleNamespace(material=mat)
    op = UI.HALCYON_OT_halo_node()
    r = op.execute(ctx)
    check('the operator converts a legacy material', r == {'FINISHED'}
          and SN.halo_node_of(tree) is not None and mat.halcyon.halo is False
          and SN.halo_node_of(tree).hardness == 20)
    n = len(tree.nodes)
    op.execute(ctx)
    check('and never makes a second node', len(tree.nodes) == n)
    tree2 = _Tree()
    out2 = tree2.nodes.new('ShaderNodeOutputMaterial')
    mat2 = types.SimpleNamespace(name='M2', halcyon=_hs(), node_tree=tree2,
                                 use_nodes=True)
    op.execute(types.SimpleNamespace(material=mat2))
    hn = SN.halo_node_of(tree2)
    check('on a plain material it makes a bare node on the Surface',
          hn is not None and out2.inputs.get('Surface').is_linked
          and out2.inputs.get('Surface').links[0].from_socket is hn.outputs[0])
    check('the operator poll wants a material',
          UI.HALCYON_OT_halo_node.poll(types.SimpleNamespace(material=None))
          is False)


def main():
    utf8_console()
    FAILS.clear()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(f'{t.__name__} raised')
    print()
    if FAILS:
        print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS))
        return 1
    print('all R253 halo node checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
