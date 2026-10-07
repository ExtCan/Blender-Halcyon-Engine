"""R253 (1.92.0): the Convert to Anime / Cartoon / Game buttons.

The Material panel's converter gains three more one-click targets --
the Anime Shader, the Cartoon Shader and the Console Emulation Shader
-- with the master conversion's relink-not-reset contract: a texture
in Base Color ends up in Diffuse Color / Paint Color, the normal chain,
the alpha and the emission travel, and the source may be any Blender
shader or one of the engine's own nodes (the master included), carried
by socket name. The decision is bpy-free (core/convert.plan_for) and
proven here table by table; the surgery (convert.convert_material_to,
the one operator) runs on a fake tree built from the real SOCKETS
tuples; the scene settings, the panel and the alpha predicate are
pinned; and a master material renders byte-identically before and
after the round, because a conversion edits a node tree and renders
nothing new.

    python -m halcyon.tests.test_r253_convert_buttons
"""
import os
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import _sk, base_settings
from .scenebuild import demo_scene
from ..core import convert as C
from ..core import console as CON
from ..core import render as R
from ..core import shading as SH
from ..core.scene import Material

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def _root():
    return os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))


def _nodes():
    """The node module under the fake bpy."""
    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    bpy.types.AddonPreferences = type('AddonPreferences',
                                      (bpy.types.Panel,), {})
    import importlib
    return importlib.import_module('halcyon.nodes.shader_nodes')


# ------------------------------------------------------------ the tables


def test_target_tables():
    """Every socket the maps name exists on the node it names; the
    engine-node source tables speak the master's vocabulary; the
    preset-ownership sets mirror the presets; the compat tuple is one
    object shared by the node and the scene setting."""
    SN = _nodes()
    socks = {
        'ANIME': {s[1] for s in SN.HALCYON_AnimeShaderNode.SOCKETS},
        'CARTOON': {s[1] for s in SN.HALCYON_CartoonNode.SOCKETS},
        'CONSOLE': {s[1] for s in SN.HALCYON_ConsoleShaderNode.SOCKETS},
    }
    master = {s[1] for s in SN.HALCYON_ShaderNode.SOCKETS}
    for target, tmap in C.TARGET_MAP.items():
        bad = [v for v in tmap.values() if v not in socks[target]]
        check(f'every TARGET_MAP[{target}] socket exists on the node',
              not bad, ', '.join(bad))
        badk = [k for k in tmap if k not in master]
        check(f'every TARGET_MAP[{target}] key is a master socket',
              not badk, ', '.join(badk))
    check('the console map is the identity over 21 shared names',
          len(C.TARGET_MAP['CONSOLE']) == 21
          and all(k == v for k, v in C.TARGET_MAP['CONSOLE'].items()))
    check('TARGET_NODES names the three nodes',
          C.TARGET_NODES == {'ANIME': 'HALCYON_AnimeShaderNode',
                             'CARTOON': 'HALCYON_CartoonNode',
                             'CONSOLE': 'HALCYON_ConsoleShaderNode'})

    node_socks = {
        C.MASTER_NODE: master,
        C.ANIME_NODE: socks['ANIME'],
        C.CARTOON_NODE: socks['CARTOON'],
        C.CONSOLE_NODE: socks['CONSOLE'],
        C.BI_NODE: {s[1] for s in SN.HALCYON_BIMaterialNode.BI_SOCKETS},
    }
    for idn, table in C.HALCYON_SOURCES.items():
        badt = [t for t, _a in table if t not in master]
        check(f'every HALCYON_SOURCES[{idn}] target is a master socket',
              not badt, ', '.join(badt))
        bada = [a for _t, al in table for a in al if a not in node_socks[idn]]
        check(f'every HALCYON_SOURCES[{idn}] alias exists on that node',
              not bada, ', '.join(bada))
    check('SOURCES (the Blender shaders) is untouched by the fallback',
          C.MASTER_NODE not in C.SOURCES and C.ANIME_NODE not in C.SOURCES)

    check('STYLE_OWNED mirrors ANIME_STYLE_PRESETS, sockets only',
          set(C.STYLE_OWNED) == set(SH.ANIME_STYLE_PRESETS)
          and all('__props' not in v and v for v in C.STYLE_OWNED.values())
          and 'Shadow 1 Softness' in C.STYLE_OWNED['TV_80S']
          and 'Diffuse Color' not in C.STYLE_OWNED['TV_80S'])
    check('ERA_OWNED mirrors CARTOON_ERA_PRESETS, sockets only',
          set(C.ERA_OWNED) == set(SH.CARTOON_ERA_PRESETS)
          and all('shadow_mode' not in v and v for v in C.ERA_OWNED.values())
          and 'Shadow Threshold' in C.ERA_OWNED['UPA_50S']
          and 'Paint Color' not in C.ERA_OWNED['UPA_50S'])
    ann = SN.HALCYON_AnimeShaderNode.__annotations__['compat']
    check('ANIME_COMPAT_ITEMS is the node\'s own compat items object',
          ann.kw['items'] is SH.ANIME_COMPAT_ITEMS
          and [i[0] for i in SH.ANIME_COMPAT_ITEMS]
          == ['GENERIC', 'ARCSYS', 'DBFZ', 'KAKAROT', 'SPARKING',
              'GENSHIN', 'ZZZ']
          and all(len(i[2]) >= 12 for i in SH.ANIME_COMPAT_ITEMS))
    check('anime_specular_size is a stand-in that keeps rougher = wider',
          C.anime_specular_size(8192) < C.anime_specular_size(128)
          < C.anime_specular_size(30) < C.anime_specular_size(2)
          and abs(C.anime_specular_size(30) - 0.9 / 30 ** 0.5) < 1e-9
          and C.anime_specular_size(8192) == 0.02
          and C.anime_specular_size(0.0) == 0.5)
    check('CONSOLE_GLOSS_MAX is 128', C.CONSOLE_GLOSS_MAX == 128.0)
    caught = False
    try:
        C.plan_for('MAX', 'ShaderNodeBsdfPrincipled')
    except ValueError:
        caught = True
    check('an unknown target is refused by name', caught)


# ------------------------------------------------------------- the plans


def test_plan_anime():
    """Principled -> Anime: links by name, alpha into Opacity, the
    emission STRAIGHT (colour and strength both carried, no multiply
    node), the highlight size from the roughness under CUSTOM only."""
    vals = {'Alpha': 0.5, 'Emission Color': [1.0, 1.0, 1.0, 1.0],
            'Emission Strength': 0.0, 'Roughness': 0.3, 'Metallic': 0.0}
    p = C.plan_for('ANIME', 'ShaderNodeBsdfPrincipled', vals,
                   {'Base Color', 'Normal'},
                   {'style': 'CUSTOM', 'compat': 'ARCSYS'})
    pairs = set(p['pairs'])
    check('the texture, the normal chain, the emission colour and its '
          'strength all carry by name',
          {('Diffuse Color', 'Base Color'), ('Normal', 'Normal'),
           ('Self-Illumination', 'Emission Color'),
           ('Emission Strength', 'Emission Strength')} <= pairs,
          str(p['pairs']))
    check('Alpha 0.5 becomes the Opacity pair',
          ('Opacity', 'Alpha') in pairs)
    check('no multiply node for a cel target (the strength socket does it)',
          p['scale_links'] == {} and 'Self-Illumination' not in p['extras'])
    check('the props are the chosen style and compat, compat first',
          list(p['props'].items()) == [('compat', 'ARCSYS'),
                                       ('style', 'CUSTOM')])
    check('the target names the Anime node and the label its style',
          p['target'] == 'HALCYON_AnimeShaderNode'
          and p['label'] == 'Anime Shader (Custom)')
    g = C.glossiness_from_roughness(0.3)
    check('Roughness 0.3 under CUSTOM derives the Specular Size',
          abs(p['extras'].get('Specular Size', -1)
              - C.anime_specular_size(g)) < 1e-9,
          str(p['extras']))
    p2 = C.plan_for('ANIME', 'ShaderNodeBsdfPrincipled',
                    dict(vals, Metallic=1.0), {'Base Color', 'Normal'},
                    {'style': 'TV_80S', 'compat': 'GENERIC'})
    check('under a style the derived highlight keys are the style\'s '
          '(no Specular Size / Level in extras)',
          'Specular Size' not in p2['extras']
          and 'Specular Level' not in p2['extras']
          and p2['props']['style'] == 'TV_80S'
          and p2['label'] == 'Anime Shader (80s TV)', str(p2['extras']))
    check('metal under CUSTOM lifts the Specular Level and says so',
          C.plan_for('ANIME', 'ShaderNodeBsdfPrincipled',
                     dict(vals, Metallic=1.0), set())['extras']
          .get('Specular Level') == 1.0
          and any('matcap' in n for n in p2['notes']))
    p3 = C.plan_for('ANIME', 'ShaderNodeEmission',
                    {'Color': [0.0, 1.0, 0.0, 1.0], 'Strength': 5.0}, set())
    check('the Emission shader\'s Strength alias carries straight',
          ('Emission Strength', 'Strength') in p3['pairs']
          and ('Self-Illumination', 'Color') in p3['pairs']
          and 'Self-Illumination' not in p3['extras'], str(p3))
    p4 = C.plan_for('ANIME', 'ShaderNodeBsdfPrincipled', {},
                    {'Emission Color', 'Emission Strength'})
    check('a LINKED Emission Strength carries as a pair, not a marker',
          ('Emission Strength', 'Emission Strength') in p4['pairs']
          and ('Self-Illumination', 'Emission Color') in p4['pairs']
          and p4['scale_links'] == {})
    p5 = C.plan_for('ANIME', 'ShaderNodeBsdfToon',
                    {'Color': [1, 0, 0, 1], 'Size': 0.7, 'Smooth': 0.1},
                    set(), {'style': 'CUSTOM'})
    check('a Toon BSDF\'s Size / Smooth become the first shadow band',
          ('Shadow 1 Threshold', 'Size') in p5['pairs']
          and ('Shadow 1 Softness', 'Smooth') in p5['pairs'])
    p6 = C.plan_for('ANIME', 'ShaderNodeBsdfToon',
                    {'Color': [1, 0, 0, 1], 'Size': 0.7, 'Smooth': 0.1},
                    {'Size'}, {'style': 'OVA_80S'})
    check('under a style the constant Smooth yields to the preset but the '
          'LINKED Size still carries',
          ('Shadow 1 Threshold', 'Size') in p6['pairs']
          and ('Shadow 1 Softness', 'Smooth') not in p6['pairs']
          and any('preset' in n for n in p6['notes']))
    p7 = C.plan_for('ANIME', 'ShaderNodeBsdfGlass',
                    {'Color': [1, 1, 1, 1], 'IOR': 1.5, 'Roughness': 0.0},
                    set())
    check('what the anime node lacks is dropped and the first drops named',
          any('IOR has no anime equivalent' in n for n in p7['notes'])
          and not any(t == 'IOR' for t, _a in p7['pairs']))


def test_plan_cartoon():
    """Toon -> Cartoon: Size / Smooth become the one shadow band (CUSTOM)
    and yield to an era; an Emission shader becomes flat paint; the
    base colour lands in Paint Color."""
    p = C.plan_for('CARTOON', 'ShaderNodeBsdfToon',
                   {'Size': 0.7, 'Smooth': 0.1, 'Color': [1, 0, 0, 1]},
                   set(), {'era': 'CUSTOM'})
    check('Toon Size 0.7 / Smooth 0.1 become Shadow Threshold / Softness',
          ('Shadow Threshold', 'Size') in p['pairs']
          and ('Shadow Softness', 'Smooth') in p['pairs']
          and ('Paint Color', 'Color') in p['pairs'], str(p['pairs']))
    check('the props are the era', p['props'] == {'era': 'CUSTOM'}
          and p['label'] == 'Cartoon Shader (Custom)')
    p2 = C.plan_for('CARTOON', 'ShaderNodeBsdfToon',
                    {'Size': 0.7, 'Smooth': 0.1, 'Color': [1, 0, 0, 1]},
                    set(), {'era': 'UPA_50S'})
    check('under UPA_50S the era owns the shadow band',
          not any(t in ('Shadow Threshold', 'Shadow Softness')
                  for t, _a in p2['pairs'])
          and ('Paint Color', 'Color') in p2['pairs']
          and p2['label'] == 'Cartoon Shader (UPA Modern (1950s))')
    p3 = C.plan_for('CARTOON', 'ShaderNodeEmission',
                    {'Color': [1, 0.2, 0.2, 1], 'Strength': 1.0}, set())
    check('an Emission shader becomes flat paint: no shadow, no lamp',
          p3['extras'].get('Shadow Amount') == 0.0
          and p3['extras'].get('Lamp Influence') == 0.0
          and any('flat paint' in n for n in p3['notes'])
          and ('Paint Color', 'Color') in p3['pairs']
          and ('Emission Strength', 'Strength') in p3['pairs'])
    p4 = C.plan_for('CARTOON', 'ShaderNodeEmission',
                    {'Color': [1, 0.2, 0.2, 1], 'Strength': 1.0}, set(),
                    {'era': 'GOLDEN_40S'})
    check('under an era the flat-paint extras yield to the era',
          'Shadow Amount' not in p4['extras']
          and 'Lamp Influence' not in p4['extras'])
    p5 = C.plan_for('CARTOON', 'ShaderNodeBsdfPrincipled',
                    {'Alpha': 0.5, 'Emission Color': [1, 1, 1, 1],
                     'Emission Strength': 0.0}, {'Base Color', 'Normal'})
    check('Principled -> Cartoon: Paint Color, Normal, Opacity and the '
          'straight emission',
          {('Paint Color', 'Base Color'), ('Normal', 'Normal'),
           ('Opacity', 'Alpha'), ('Self-Illumination', 'Emission Color'),
           ('Emission Strength', 'Emission Strength')} <= set(p5['pairs'])
          and p5['scale_links'] == {})


def test_plan_console():
    """Principled -> Console: the bi_plan metal move, the exponent
    ceiling, the vertex-colour switch per machine, the master's fold
    and multiply markers, and every machine resolving to a model."""
    p = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled',
                   {'Metallic': 1.0, 'Base Color': [0.9, 0.7, 0.3, 1],
                    'Roughness': 0.05, 'Specular IOR Level': 0.5},
                   set(), {'console': 'PS1'})
    check('metal tints the highlight with the base colour and lifts it',
          p['extras'].get('Specular Color') == (0.9, 0.7, 0.3, 1.0)
          and p['extras'].get('Specular Level', 0) >= 0.8, str(p['extras']))
    check('Roughness 0.05 is clamped to the console ceiling with a note',
          p['extras'].get('Glossiness') == C.CONSOLE_GLOSS_MAX
          and any('clamped' in n for n in p['notes']))
    check('the props carry the machine and the label is the console\'s',
          p['props'] == {'console': 'PS1'}
          and p['label'] == CON.label_of({'console': 'PS1'})
          and p['target'] == 'HALCYON_ConsoleShaderNode')
    for con, prop, val in (('GAMECUBE', 'gc_material_src', 'VTX'),
                           ('PC_FIXED', 'pc_color_vertex', True),
                           ('N64', 'n64_type', 'VERTEX'),
                           ('PS1', None, None)):
        pv = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {},
                        {'Base Color'}, {'console': con},
                        {'Base Color': 'ShaderNodeVertexColor'})
        ok = (('Vertex Color', 'Base Color') in pv['pairs']
              and not any(t == 'Diffuse Color' for t, _a in pv['pairs'])
              and pv['extras'].get('Vertex Color Mix') == 1.0
              and list(pv['props'])[0] == 'console')
        if prop is not None:
            ok = ok and pv['props'].get(prop) == val
        else:
            ok = ok and len(pv['props']) == 1
        check(f'a Vertex Color into Base Color is the {con}\'s own switch',
              ok, str(pv['props']) + ' ' + str(pv['pairs']))
    pt = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled', {},
                    {'Base Color'}, {'console': 'GAMECUBE'},
                    {'Base Color': 'ShaderNodeTexImage'})
    check('a texture into Base Color stays a Diffuse Color chain',
          ('Diffuse Color', 'Base Color') in pt['pairs']
          and 'gc_material_src' not in pt['props'])
    p2 = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled',
                    {'Emission Color': [1, 1, 1, 1], 'Emission Strength': 0.0},
                    set())
    check('white emission at strength 0 is dropped (the master\'s fold)',
          not any(t == 'Self-Illumination' for t, _a in p2['pairs'])
          and p2['extras'].get('Self-Illumination') == (0.0, 0.0, 0.0, 1.0))
    p3 = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled',
                    {'Emission Strength': 3.0}, {'Emission Color'})
    check('strength 3 on a linked colour keeps the VALUE multiply marker',
          p3['scale_links'].get('Self-Illumination') == ('VALUE', 3.0))
    bad = []
    for con, _lab, _d in CON.CONSOLE_ITEMS:
        pc = C.plan_for('CONSOLE', 'ShaderNodeBsdfPrincipled',
                        {'Base Color': [1, 1, 1, 1]}, {'Base Color'},
                        {'console': con}, {'Base Color': 'ShaderNodeVertexColor'})
        res = CON.resolve(pc['props'])
        if res['model'] not in CON.CONSOLE_MODELS:
            bad.append(f'{con}->{res["model"]}')
        if pc['label'] != CON.label_of(pc['props']):
            bad.append(f'{con} label')
    check(f'every one of the {len(CON.CONSOLE_ITEMS)} machines resolves to '
          'a console model and labels as the node would', not bad,
          ', '.join(bad))
    p4 = C.plan_for('CONSOLE', 'ShaderNodeBsdfGlass',
                    {'Color': [1, 1, 1, 1], 'IOR': 1.5, 'Roughness': 0.0},
                    set())
    check('IOR is dropped with a note; Glass\'s Reflection / Opacity extras '
          'carry (the console has both sockets)',
          any('IOR has no console equivalent' in n for n in p4['notes'])
          and p4['extras'].get('Reflection') == 0.6
          and p4['extras'].get('Opacity') == 0.25)


def test_plan_from_halcyon_nodes():
    """The engine's own nodes as sources: the master onward to each
    target, Anime -> Cartoon, BI -> Console, and a cel node back to the
    master through plan()'s fallback (its Emission Strength folds)."""
    SN = _nodes()
    mvals = {s[1]: s[2] for s in SN.HALCYON_ShaderNode.SOCKETS
             if s[2] is not None}
    mvals['Toon Size'] = 0.7
    mvals['Glossiness'] = 100.0
    mlinks = {'Diffuse Color', 'Normal', 'Opacity', 'Self-Illumination'}
    pa = C.plan_for('ANIME', C.MASTER_NODE, mvals, mlinks)
    check('master -> Anime keeps the four links by name and the Toon Size',
          {('Diffuse Color', 'Diffuse Color'), ('Normal', 'Normal'),
           ('Opacity', 'Opacity'), ('Self-Illumination', 'Self-Illumination'),
           ('Shadow 1 Threshold', 'Toon Size')} <= set(pa['pairs'])
          and not any(a not in mvals and a not in mlinks
                      for _t, a in pa['pairs']), str(pa['pairs']))
    pc = C.plan_for('CARTOON', C.MASTER_NODE, mvals, mlinks)
    check('master -> Cartoon puts the Diffuse chain into Paint Color',
          ('Paint Color', 'Diffuse Color') in pc['pairs']
          and ('Shadow Threshold', 'Toon Size') in pc['pairs'])
    pk = C.plan_for('CONSOLE', C.MASTER_NODE, mvals, mlinks)
    got = {t for t, _a in pk['pairs']}
    check('master -> Console is the identity over the 21 shared names',
          got == set(C.TARGET_MAP['CONSOLE'])
          and all(t == a for t, a in pk['pairs'])
          and 'Vertex Color' in got and 'Vertex Color Mix' in got, str(got))
    check('the master\'s explicit Glossiness 100 wins over its Roughness '
          'socket (no re-derived exponent, no clamp)',
          ('Glossiness', 'Glossiness') in pk['pairs']
          and 'Glossiness' not in pk['extras'], str(pk['extras']))
    avals = {s[1]: s[2] for s in SN.HALCYON_AnimeShaderNode.SOCKETS
             if s[2] is not None}
    avals['Emission Strength'] = 2.0
    pac = C.plan_for('CARTOON', C.ANIME_NODE, avals, {'Diffuse Color'})
    check('Anime -> Cartoon carries Diffuse Color -> Paint Color and the '
          'Emission Strength straight',
          ('Paint Color', 'Diffuse Color') in pac['pairs']
          and ('Emission Strength', 'Emission Strength') in pac['pairs']
          and ('Self-Illumination', 'Self-Illumination') in pac['pairs']
          and pac['scale_links'] == {}
          and ('Shadow Threshold', 'Shadow 1 Threshold') in pac['pairs'])
    bvals = {s[1]: s[2] for s in SN.HALCYON_BIMaterialNode.BI_SOCKETS
             if s[2] is not None}
    bvals['Hardness'] = 100.0
    pb = C.plan_for('CONSOLE', C.BI_NODE, bvals, {'Color'})
    check('BI Hardness 100 -> Glossiness 100 on the console, Color -> '
          'Diffuse Color',
          ('Glossiness', 'Hardness') in pb['pairs']
          and ('Diffuse Color', 'Color') in pb['pairs']
          and 'Glossiness' not in pb['extras'])
    bvals['Hardness'] = 400.0
    pb2 = C.plan_for('CONSOLE', C.BI_NODE, bvals, set())
    check('BI Hardness 400 is clamped to the console ceiling',
          pb2['extras'].get('Glossiness') == C.CONSOLE_GLOSS_MAX
          and not any(t == 'Glossiness' for t, _a in pb2['pairs']))
    # the master target, through plan()'s fallback
    pm = C.plan(C.ANIME_NODE, dict(avals, **{
        'Self-Illumination': [1, 1, 1, 1], 'Emission Strength': 0.0}),
        {'Diffuse Color', 'Normal', 'Opacity'}, 'AUTO')
    check('a cel node back to the master carries Diffuse Color / Normal / '
          'Opacity by name and folds a zero Emission Strength',
          {('Diffuse Color', 'Diffuse Color'), ('Normal', 'Normal'),
           ('Opacity', 'Opacity')} <= set(pm['pairs'])
          and not any(t == 'Self-Illumination' for t, _a in pm['pairs'])
          and pm['extras'].get('Self-Illumination') == (0.0, 0.0, 0.0, 1.0)
          and pm['model'] in {m[0] for m in SH.MODEL_ITEMS}, str(pm))
    pm2 = C.plan(C.ANIME_NODE, dict(avals, **{
        'Self-Illumination': [0.5, 0.0, 0.0, 1.0], 'Emission Strength': 2.0}),
        set(), 'AUTO')
    check('...and multiplies a constant one into the colour',
          pm2['extras'].get('Self-Illumination') == (1.0, 0.0, 0.0, 1.0))
    pu = C.plan('ShaderNodeBsdfFromTheFuture', {'Color': [1, 0, 0, 1]})
    check('an unknown source still takes the generic table',
          ('Diffuse Color', 'Color') in pu['pairs'])


# ------------------------------------------------------ the surgery, faked


class _Loc:
    def __init__(self, x=0.0, y=0.0):
        self.x, self.y = float(x), float(y)


class _Sock:
    def __init__(self, name, value=None, node=None, ident=None, stype=None):
        self.name = name
        self.identifier = ident or name
        self.type = stype or ('RGBA' if hasattr(value, '__len__') else
                              'VALUE' if value is not None else 'VECTOR')
        self.default_value = list(value) if hasattr(value, '__len__') \
            else value
        self.is_linked = False
        self.links = []
        self.hide = False
        self.hide_value = False
        self.mute = False
        self.description = ''
        self.node = node


class _Link:
    def __init__(self, a, b):
        self.from_socket, self.to_socket = a, b
        self.from_node, self.to_node = a.node, b.node


class _Inputs(list):
    def __init__(self, node=None):
        super().__init__()
        self.node = node

    def get(self, name):
        return next((s for s in self if s.name == name), None)

    def new(self, kind, name, value=None, ident=None):
        s = _Sock(name, value, self.node, ident,
                  'SHADER' if kind == 'NodeSocketShader' else None)
        self.append(s)
        return s

    def __getitem__(self, key):
        if isinstance(key, str):
            s = self.get(key)
            if s is None:
                raise KeyError(key)
            return s
        return super().__getitem__(key)


class _Node:
    def __init__(self, idn, label=''):
        self.bl_idname = idn
        self.bl_label = label or idn.replace('ShaderNode', '')
        self.inputs = _Inputs(self)
        self.outputs = _Inputs(self)
        self._loc = _Loc()
        self.label = ''
        self.mute = False
        self.hide = False
        self.is_active_output = False

    @property
    def location(self):
        return self._loc

    @location.setter
    def location(self, v):
        self._loc = _Loc(v[0], v[1])


class _Links(list):
    def new(self, a, b):
        ln = _Link(a, b)
        self.append(ln)
        b.is_linked = True
        b.links = [ln]
        a.links = list(getattr(a, 'links', [])) + [ln]
        return ln


class _Nodes(list):
    def __init__(self, tree, SN):
        super().__init__()
        self.tree = tree
        self.SN = SN

    def new(self, idn):
        SN = self.SN
        n = _Node(idn)
        if idn == 'HALCYON_AnimeShaderNode':
            n.bl_label = "Anime Shader"
            for kind, name, d in SN.HALCYON_AnimeShaderNode.SOCKETS:
                n.inputs.new(kind, name, d)
            n.outputs.new('NodeSocketShader', 'Surface')
            for pname, pdef in SN.HALCYON_AnimeShaderNode.__annotations__.items():
                setattr(n, pname, pdef.kw.get('default'))
            n.apply_style = types.MethodType(
                SN.HALCYON_AnimeShaderNode.apply_style, n)
        elif idn == 'HALCYON_CartoonNode':
            n.bl_label = "Cartoon Shader"
            for kind, name, d in SN.HALCYON_CartoonNode.SOCKETS:
                n.inputs.new(kind, name, d)
            n.outputs.new('NodeSocketShader', 'Surface')
            for pname, pdef in SN.HALCYON_CartoonNode.__annotations__.items():
                setattr(n, pname, pdef.kw.get('default'))
            n.apply_era = types.MethodType(SN.HALCYON_CartoonNode.apply_era, n)
        elif idn == 'HALCYON_ConsoleShaderNode':
            n.bl_label = "Console Emulation Shader"
            for kind, name, d in SN.HALCYON_ConsoleShaderNode.SOCKETS:
                n.inputs.new(kind, name, d)
            n.outputs.new('NodeSocketShader', 'Surface')
            for name, _i, d in CON.ENUM_PROPS:
                setattr(n, name, d)
            for name, _k, d in CON.SCALAR_PROPS:
                setattr(n, name, d)
            n.refresh_calls = 0

            def _refresh(node=n):
                node.refresh_calls += 1
            n.refresh_sockets = _refresh
        elif idn == 'ShaderNodeOutputMaterial':
            n.bl_label = "Material Output"
            n.inputs.new('NodeSocketShader', 'Surface')
        elif idn == 'ShaderNodeMix':
            n.inputs.new('NodeSocketFloat', 'Factor', 0.5, 'Factor_Float')
            n.inputs.new('NodeSocketColor', 'A', [0.5, 0.5, 0.5, 1.0], 'A_Color')
            n.inputs.new('NodeSocketColor', 'B', [0.5, 0.5, 0.5, 1.0], 'B_Color')
            n.outputs.new('NodeSocketColor', 'Result', [0, 0, 0, 1], 'Result_Color')
        else:
            raise RuntimeError(f'the fake tree cannot make {idn}')
        self.append(n)
        return n

    def remove(self, n):
        if n in self:
            super().remove(n)


class _Tree:
    def __init__(self, SN):
        self.nodes = _Nodes(self, SN)
        self.links = _Links()


def _principled_tree(SN, feed='ShaderNodeTexImage', emission_link=False):
    """Principled (Base Color <- a texture / vertex colour, Normal <- a
    Normal Map, Alpha 0.5, white emission at strength 0) -> Output."""
    tree = _Tree(SN)
    src = _Node('ShaderNodeBsdfPrincipled', 'Principled BSDF')
    for name, d in (('Base Color', [0.8, 0.8, 0.8, 1.0]), ('Metallic', 0.0),
                    ('Roughness', 0.3), ('IOR', 1.45), ('Alpha', 0.5),
                    ('Specular IOR Level', 0.5),
                    ('Emission Color', [1.0, 1.0, 1.0, 1.0]),
                    ('Emission Strength', 0.0)):
        src.inputs.new('', name, d)
    src.inputs.new('NodeSocketVector', 'Normal', None)
    src.outputs.new('NodeSocketShader', 'BSDF')
    tree.nodes.append(src)
    tex = _Node(feed, 'Image Texture')
    tex.outputs.new('NodeSocketColor', 'Color', [0, 0, 0, 1])
    tree.nodes.append(tex)
    nm = _Node('ShaderNodeNormalMap', 'Normal Map')
    nm.outputs.new('NodeSocketVector', 'Normal', None)
    tree.nodes.append(nm)
    out = tree.nodes.new('ShaderNodeOutputMaterial')
    out.is_active_output = True
    tree.links.new(tex.outputs[0], src.inputs.get('Base Color'))
    tree.links.new(nm.outputs[0], src.inputs.get('Normal'))
    if emission_link:
        em = _Node('ShaderNodeTexImage', 'Emission Map')
        em.outputs.new('NodeSocketColor', 'Color', [0, 0, 0, 1])
        tree.nodes.append(em)
        tree.links.new(em.outputs[0], src.inputs.get('Emission Color'))
        src.inputs.get('Emission Strength').default_value = 3.0
    tree.links.new(src.outputs[0], out.inputs.get('Surface'))
    mat = types.SimpleNamespace(
        name='Mat', name_full='Mat', use_nodes=True, node_tree=tree,
        halcyon=types.SimpleNamespace(use_override=True))
    return mat, src, tex, nm, out


def test_operator_path_headless():
    """convert_material_to on a fake tree built from the real SOCKETS:
    every link, value, prop and preset lands where the plan says; the
    skip rule; the three targets; the multiply road on the console; the
    operator's execute over the active material."""
    SN = _nodes()
    import importlib
    CV = importlib.import_module('halcyon.convert')

    mat, src, tex, nm, out = _principled_tree(SN)
    ok, msg = CV.convert_material_to(mat, 'ANIME',
                                     {'style': 'TV_80S', 'compat': 'ARCSYS'})
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_AnimeShaderNode'), None)
    check('Principled -> Anime converts', ok and node is not None, msg)
    if node is None:
        return
    check('the compat and style landed on the node',
          node.compat == 'ARCSYS' and node.style == 'TV_80S')
    check('the image feeds Diffuse Color, the normal map feeds Normal',
          node.inputs.get('Diffuse Color').is_linked
          and node.inputs.get('Diffuse Color').links[0].from_socket is tex.outputs[0]
          and node.inputs.get('Normal').is_linked
          and node.inputs.get('Normal').links[0].from_socket is nm.outputs[0])
    check('Alpha 0.5 is the node\'s Opacity',
          node.inputs.get('Opacity').default_value == 0.5)
    check('the style preset wrote its softness (and its menus)',
          node.inputs.get('Shadow 1 Softness').default_value
          == SH.ANIME_STYLE_PRESETS['TV_80S']['Shadow 1 Softness']
          and node.tones == SH.ANIME_STYLE_PRESETS['TV_80S']['__props']['tones'])
    check('the white emission rides at strength 0: colour x 0, no multiply',
          list(node.inputs.get('Self-Illumination').default_value)[:3] == [1.0, 1.0, 1.0]
          and node.inputs.get('Emission Strength').default_value == 0.0
          and not any(n.bl_idname == 'ShaderNodeMix' for n in mat.node_tree.nodes))
    check('the output\'s Surface comes from the new node',
          out.inputs.get('Surface').links[0].from_socket is node.outputs[0])
    check('the Principled is muted, relabelled and moved aside',
          src.mute and src.label == 'replaced by Anime Shader'
          and src.location.y == -340.0)
    check('the material override is off and the message names the plan',
          mat.halcyon.use_override is False
          and 'Anime Shader (80s TV)' in msg and 'inputs carried' in msg, msg)
    ok2, msg2 = CV.convert_material_to(mat, 'ANIME', {'style': 'CUSTOM'})
    check('a second pass without force is skipped as "already uses"',
          ok2 is False and 'already uses' in msg2, msg2)
    ok3, _m3 = CV.convert_material_to(mat, 'ANIME', {'style': 'CUSTOM'},
                                      force=True)
    check('force reconverts', ok3)

    mat, src, tex, nm, out = _principled_tree(SN)
    ok, msg = CV.convert_material_to(mat, 'CARTOON', {'era': 'GOLDEN_40S'})
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_CartoonNode'), None)
    check('Principled -> Cartoon converts', ok and node is not None, msg)
    if node is not None:
        check('Paint Color takes the image, the era set the shadow mode',
              node.inputs.get('Paint Color').links[0].from_socket is tex.outputs[0]
              and node.era == 'GOLDEN_40S' and node.shadow_mode == 'TRANSPARENT'
              and node.inputs.get('Shadow Amount').default_value
              == SH.CARTOON_ERA_PRESETS['GOLDEN_40S']['Shadow Amount']
              and node.inputs.get('Opacity').default_value == 0.5
              and src.label == 'replaced by Cartoon Shader')

    mat, src, tex, nm, out = _principled_tree(SN, feed='ShaderNodeVertexColor')
    ok, msg = CV.convert_material_to(mat, 'CONSOLE', {'console': 'GAMECUBE'})
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_ConsoleShaderNode'), None)
    check('Principled -> Console converts', ok and node is not None, msg)
    if node is not None:
        check('a Vertex Color into Base Color becomes the GameCube\'s own '
              'material source',
              node.inputs.get('Vertex Color').is_linked
              and node.inputs.get('Vertex Color').links[0].from_socket is tex.outputs[0]
              and not node.inputs.get('Diffuse Color').is_linked
              and node.inputs.get('Vertex Color Mix').default_value == 1.0
              and node.console == 'GAMECUBE' and node.gc_material_src == 'VTX'
              and node.refresh_calls >= 1
              and 'GameCube' in msg, msg)
    ok4, msg4 = CV.convert_material_to(mat, 'CONSOLE', {'console': 'PS1'})
    check('the console skip rule names the node',
          ok4 is False and 'already uses the Console Emulation Shader' in msg4)

    mat, src, tex, nm, out = _principled_tree(SN, emission_link=True)
    ok, msg = CV.convert_material_to(mat, 'CONSOLE', {'console': 'PS1'})
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_ConsoleShaderNode'), None)
    mul = next((n for n in mat.node_tree.nodes
                if n.bl_idname == 'ShaderNodeMix'), None)
    check('a linked emission at strength 3 goes through a multiply node on '
          'the console (no strength socket there)',
          ok and node is not None and mul is not None
          and node.inputs.get('Self-Illumination').links[0].from_socket
          is mul.outputs[0]
          and list(mul.inputs.get('B').default_value)[:3] == [3.0, 3.0, 3.0],
          msg)
    mat, src, tex, nm, out = _principled_tree(SN, emission_link=True)
    ok, msg = CV.convert_material_to(mat, 'ANIME', {})
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_AnimeShaderNode'), None)
    check('the same tree to Anime wires the map straight and sets the '
          'strength socket to 3',
          ok and node is not None
          and not any(n.bl_idname == 'ShaderNodeMix' for n in mat.node_tree.nodes)
          and node.inputs.get('Self-Illumination').links[0].from_node.bl_label
          == 'Emission Map'
          and node.inputs.get('Emission Strength').default_value == 3.0, msg)

    # the engine's own node as a source: master -> Anime by name
    mat, src, tex, nm, out = _principled_tree(SN)
    master = _Node('HALCYON_ShaderNode', 'Halcyon Shader')
    for kind, name, d in SN.HALCYON_ShaderNode.SOCKETS:
        master.inputs.new(kind, name, d)
    master.outputs.new('NodeSocketShader', 'Surface')
    master.model = 'PHONG'
    mat.node_tree.nodes.append(master)
    mat.node_tree.links.new(tex.outputs[0], master.inputs.get('Diffuse Color'))
    master.inputs.get('Toon Size').default_value = 0.7
    mat.node_tree.links.new(master.outputs[0], out.inputs.get('Surface'))
    ok, msg = CV.convert_material_to(mat, 'ANIME', {'style': 'CUSTOM'})
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_AnimeShaderNode'), None)
    check('a Halcyon Shader material converts to Anime by socket name',
          ok and node is not None
          and node.inputs.get('Diffuse Color').links[0].from_socket is tex.outputs[0]
          and node.inputs.get('Shadow 1 Threshold').default_value == 0.7
          and master.mute, msg)

    # the operator
    mat, src, tex, nm, out = _principled_tree(SN)
    op = CV.HALCYON_OT_convert_to_node()
    op.target, op.scope = 'CARTOON', 'ACTIVE'
    op.style, op.compat, op.era, op.console = 'CUSTOM', 'GENERIC', 'TV_90S', 'PS1'
    op.keep_original, op.force = True, False
    reports = []
    op.report = lambda kind, text: reports.append((kind, text))
    ctx = types.SimpleNamespace(
        active_object=types.SimpleNamespace(active_material=mat),
        scene=types.SimpleNamespace(objects=[]), selected_objects=[])
    rc = op.execute(ctx)
    node = next((n for n in mat.node_tree.nodes
                 if n.bl_idname == 'HALCYON_CartoonNode'), None)
    check('the operator converts the active material with the scene\'s era',
          rc == {'FINISHED'} and node is not None and node.era == 'TV_90S'
          and reports and '1 converted' in reports[-1][1], str(reports))
    rc2 = op.execute(ctx)
    check('a second run reports it already converted',
          rc2 == {'FINISHED'} and '1 already converted' in reports[-1][1],
          str(reports[-1]))
    check('the operator is registered and documented',
          'halcyon.convert_to_node' in [c.bl_idname for c in CV.CLASSES]
          and len(CV.HALCYON_OT_convert_to_node.bl_description) >= 40
          and CV.HALCYON_OT_convert_to_node.bl_options == {'REGISTER', 'UNDO'})
    tann = CV.HALCYON_OT_convert_to_node.__annotations__
    check('the target and scope enums are static with real lines',
          [i[0] for i in tann['target'].kw['items']] == ['ANIME', 'CARTOON', 'CONSOLE']
          and all(len(i[2]) >= 12 for i in tann['target'].kw['items'])
          and all(len(i[2]) >= 12 for i in tann['scope'].kw['items'])
          and tann['target'].kw['default'] == 'ANIME'
          and all(tann[k].kind == 'StringProperty'
                  for k in ('style', 'compat', 'era', 'console')))


# --------------------------------------------------- settings and the panel


def test_panel_and_scene_props():
    """The four scene choices are static enums with defaults and real
    tooltips, never render settings; the panel draws the nine buttons
    through one operator without disturbing the two count-pins; the
    slot list labels a cel material as converted."""
    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    bpy.types.AddonPreferences = type('AddonPreferences',
                                      (bpy.types.Panel,), {})
    import importlib
    P = importlib.import_module('halcyon.properties')
    ann = P.HalcyonSettings.__annotations__
    want = {'convert_anime_style': ('CUSTOM', SH.ANIME_STYLE_ITEMS),
            'convert_anime_compat': ('GENERIC', SH.ANIME_COMPAT_ITEMS),
            'convert_cartoon_era': ('CUSTOM', SH.CARTOON_ERA_ITEMS),
            'convert_console': ('PS1', CON.CONSOLE_ITEMS)}
    bad = []
    for k, (d, items) in want.items():
        a = ann.get(k)
        if a is None:
            bad.append(f'{k} missing')
            continue
        if callable(a.kw.get('items')) or a.kw.get('items') is not items:
            bad.append(f'{k} items')
        if a.kw.get('default') != d:
            bad.append(f'{k} default {a.kw.get("default")}')
        if len(str(a.kw.get('description') or '')) < 40:
            bad.append(f'{k} tooltip')
    check('the four scene choices are static, defaulted, documented and '
          'read the nodes\' own tuples', not bad, ', '.join(bad))
    import dataclasses
    from ..core.settings import RenderSettings
    fields = {f.name for f in dataclasses.fields(RenderSettings)}
    check('none of them is a render setting',
          not (set(want) & fields))
    cm = ann['convert_model']
    check('convert_model stays the master menu',
          [i[0] for i in cm.kw['items']] == list(SH.MASTER_MODELS))

    usrc = open(os.path.join(_root(), 'ui.py'), encoding='utf-8').read()
    check('the panel draws the three targets through one operator',
          "operator('halcyon.convert_to_node'" in usrc
          and 'To Anime Shader:' in usrc and 'To Cartoon Shader:' in usrc
          and 'To Game (Console):' in usrc
          and all(f"'{k}'" in usrc for k in want))
    check('the two count-pins hold: three master buttons, three BI buttons',
          usrc.count('op.model = _model') == 3
          and usrc.count("operator('halcyon.convert_to_bi'") == 3)
    i = usrc.index('has_master = bool(')
    j = usrc.index('for n in mat.node_tree.nodes))', i)
    check('the header counts a cel material as converted',
          'HALCYON_AnimeShaderNode' in usrc[i:j]
          and 'HALCYON_CartoonNode' in usrc[i:j])
    check('the choice rows draw the scene props',
          all(f"prop(hal, '{k}'" in usrc or f"row.prop(hal, prop" in usrc
              for k in want))
    from .blender_icons import ICONS
    check('the icons are real', all(i in ICONS for i in
                                    ('IPO_CONSTANT', 'SYSTEM', 'MATERIAL',
                                     'RESTRICT_SELECT_OFF', 'SCENE_DATA')))
    ui = importlib.import_module('halcyon.ui')

    def fake(idn, **props):
        node = types.SimpleNamespace(bl_idname=idn, **props)
        return types.SimpleNamespace(
            use_nodes=True, node_tree=types.SimpleNamespace(nodes=[node]),
            halcyon=types.SimpleNamespace(use_override=False, model='PHONG'))
    check('material_state names a converted Anime / Cartoon material',
          ui.material_state(fake('HALCYON_AnimeShaderNode', style='TV_80S'))
          == ('Anime TV_80S', True)
          and ui.material_state(fake('HALCYON_AnimeShaderNode', style='CUSTOM'))
          == ('Anime', True)
          and ui.material_state(fake('HALCYON_CartoonNode', era='UPA_50S'))
          == ('Cartoon UPA_50S', True)
          and ui.material_state(fake('ShaderNodeBsdfPrincipled'))
          == ('auto', False))

    # the drawing itself, on a recording layout
    class _Op(types.SimpleNamespace):
        pass

    class _Lay:
        def __init__(self, log):
            self.log = log

        def row(self, **k):
            return self

        def column(self, **k):
            return self

        def box(self):
            return self

        def label(self, **k):
            self.log.append(('label', k.get('text'), k.get('icon')))

        def prop(self, data, name, **k):
            self.log.append(('prop', name))

        def operator(self, idn, **k):
            op = _Op()
            self.log.append(('op', idn, k.get('text'), k.get('icon'), op))
            return op

    log = []
    hal = types.SimpleNamespace(convert_anime_style='OVA_80S',
                                convert_anime_compat='GENSHIN',
                                convert_cartoon_era='XEROX_60S',
                                convert_console='N64')
    ui._draw_convert_target(_Lay(log), hal, 'CONSOLE', "To Game (Console):",
                            'SYSTEM', {'HALCYON_ConsoleShaderNode'})
    ops = [e for e in log if e[0] == 'op']
    check('one block draws its label, its choice and three scope buttons '
          'carrying the scene choices',
          log[0] == ('label', "To Game (Console):", 'SYSTEM')
          and ('prop', 'convert_console') in log and len(ops) == 3
          and [o[4].scope for o in ops] == ['ACTIVE', 'SELECTED', 'SCENE']
          and all(o[1] == 'halcyon.convert_to_node' and o[4].target == 'CONSOLE'
                  and o[4].console == 'N64' and o[4].style == 'OVA_80S'
                  and o[4].compat == 'GENSHIN' and o[4].era == 'XEROX_60S'
                  for o in ops)
          and [o[4].force for o in ops] == [True, False, False], str(log))
    log.clear()
    ui._draw_convert_target(_Lay(log), hal, 'ANIME', "To Anime Shader:",
                            'IPO_CONSTANT', set())
    check('the anime block draws both its menus and no force on a fresh tree',
          ('prop', 'convert_anime_style') in log
          and ('prop', 'convert_anime_compat') in log
          and all(e[4].force is False for e in log if e[0] == 'op'))


# ----------------------------------------------------- the alpha predicate


def test_alpha_predicate_sees_the_cel_nodes():
    """_alpha_reason reads Opacity on the Anime / Cartoon / Console nodes
    (Edge Opacity on the console) exactly as it reads the master's --
    and answers None at the nodes' default 1.0 unlinked, so no saved
    cel material changes its layer plan."""
    from . import fakebpy
    fakebpy.install()
    import importlib
    EX = importlib.import_module('halcyon.export')

    class FakeMat:
        blend_method = 'OPAQUE'

    def graph(idn, opacity=1.0, link=False, edge=None):
        ins = [_sk('Diffuse Color' if idn != 'HALCYON_CartoonNode'
                   else 'Paint Color', 'RGBA', [0.5, 0.5, 0.5, 1.0]),
               _sk('Opacity', 'VALUE', opacity, ['tex', 1] if link else None)]
        if edge is not None:
            ins.append(_sk('Edge Opacity', 'VALUE', edge))
        return {'output': 'out', 'nodes': {
            'n': {'id': 'n', 'bl_idname': idn, 'props': {}, 'inputs': ins,
                  'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                    'props': {'image': 'img'},
                    'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Alpha', 'type': 'VALUE'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [_sk('Surface', 'SHADER', None, ['n', 0])],
                    'outputs': []}}}

    def why(g):
        return EX._alpha_reason(FakeMat(), Material(name='t', graph=g))

    for idn in ('HALCYON_AnimeShaderNode', 'HALCYON_CartoonNode',
                'HALCYON_ConsoleShaderNode'):
        check(f'{idn}: Opacity 1.0 unlinked is opaque (unchanged)',
              why(graph(idn)) is None)
        check(f'{idn}: Opacity 0.5 is see-through by name',
              why(graph(idn, 0.5)) == 'Opacity 0.500',
              str(why(graph(idn, 0.5))))
        check(f'{idn}: a linked Opacity is see-through by name',
              why(graph(idn, link=True)) == 'its Opacity socket is linked')
    check('the console\'s Edge Opacity counts; the cel nodes have none',
          why(graph('HALCYON_ConsoleShaderNode', edge=0.5)) == 'Edge Opacity 0.500'
          and why(graph('HALCYON_AnimeShaderNode', edge=0.5)) is None)
    check('a master graph at 1.0 is still opaque, at 0.12 still see-through',
          why(graph('HALCYON_ShaderNode')) is None
          and why(graph('HALCYON_ShaderNode', 0.12)) == 'Opacity 0.120')
    pr = graph('ShaderNodeBsdfPrincipled')
    pr['nodes']['n']['inputs'][1]['name'] = 'Alpha'
    check('a Principled graph answers as before',
          why(pr) is None)
    pr['nodes']['n']['inputs'][1]['default'] = 0.3
    check('...and its Alpha 0.3 is still named',
          why(pr) == 'Principled Alpha 0.300')


# ------------------------------------------------------- default neutrality


def test_old_scenes_render_identically():
    """Nothing in the renderer moved: a master material through the real
    engine stub renders byte-identically before and after the new code
    is imported and exercised; the scene's new settings never reach
    RenderSettings; and plan()'s pinned fold cases hold through the
    fallback-bearing lookup."""
    from . import fakeblender as FB
    props, engine = FB.install()
    a = FB.run_render(props, engine)[0]
    check('a frame renders', a is not None)
    if a is None:
        return
    a = np.array(a, copy=True)
    # the new code, imported and exercised between the two renders
    import importlib
    CV = importlib.import_module('halcyon.convert')
    SN = _nodes()
    mat, _s, _t, _n, _o = _principled_tree(SN)
    CV.convert_material_to(mat, 'ANIME', {'style': 'MODERN_20S'})
    for target in ('ANIME', 'CARTOON', 'CONSOLE'):
        C.plan_for(target, 'ShaderNodeBsdfPrincipled', {'Roughness': 0.2},
                   {'Base Color'})
    props, engine = FB.install()
    b = FB.run_render(props, engine)[0]
    check('the frame is byte-identical after the new code ran',
          b is not None and a.shape == b.shape and np.array_equal(a, b),
          '' if b is None else f'{float(np.abs(a - b).max()):.2e}')
    hs = FB.live(props.HalcyonSettings)
    from ..core.settings import RenderSettings
    check('the scene settings at their defaults still make the default '
          'RenderSettings (the four choices are UI state)',
          hs.to_settings() == RenderSettings()
          and hs.convert_anime_style == 'CUSTOM'
          and hs.convert_console == 'PS1')

    # the master converter's pinned fold cases, through the new lookup
    p = C.plan('ShaderNodeBsdfPrincipled',
               values={'Base Color': [0.8, 0.2, 0.2, 1.0],
                       'Emission Color': [1.0, 1.0, 1.0, 1.0],
                       'Emission Strength': 0.0})
    check('plan(): white at strength 0 is still dropped',
          not any(t == 'Self-Illumination' for t, _a in p['pairs'])
          and p['extras'].get('Self-Illumination') == (0.0, 0.0, 0.0, 1.0))
    p2 = C.plan('ShaderNodeBsdfPrincipled',
                values={'Emission Color': [1.0, 0.5, 0.0, 1.0],
                        'Emission Strength': 2.0})
    check('plan(): constants still multiply',
          p2['extras'].get('Self-Illumination') == (2.0, 1.0, 0.0, 1.0))
    p3 = C.plan('ShaderNodeBsdfPrincipled', values={'Emission Strength': 3.0},
                links={'Emission Color'})
    p4 = C.plan('ShaderNodeBsdfPrincipled', values={},
                links={'Emission Color', 'Emission Strength'})
    check('plan(): the VALUE and LINK markers still fire',
          p3['scale_links'].get('Self-Illumination') == ('VALUE', 3.0)
          and p4['scale_links'].get('Self-Illumination')
          == ('LINK', 'Emission Strength'))
    p6 = C.plan('ShaderNodeEmission', values={'Color': [0.0, 1.0, 0.0, 1.0],
                                              'Strength': 5.0})
    p7 = C.plan('ShaderNodeBsdfPrincipled', values={'Emission': [0, 0, 0, 1]})
    check('plan(): the Emission node and the 3.x socket are untouched',
          p6['extras'].get('Self-Illumination') == (0.0, 5.0, 0.0, 1.0)
          and [a for (t, a) in p7['pairs'] if t == 'Self-Illumination']
          == ['Emission'] and not p7['scale_links'])
    check('plan(): choose_model and the Blender-shader tables are as before',
          C.choose_model('ShaderNodeBsdfToon', {}, set()) == 'TOON'
          and len(C.SOURCES) == 14
          and all(t[0] in {s[1] for s in SN.HALCYON_ShaderNode.SOCKETS}
                  for tb in C.SOURCES.values() for t in tb))
    src = open(os.path.join(_root(), 'convert.py'), encoding='utf-8').read()
    check('convert_material keeps its pinned surgery strings',
          "p.get('scale_links')" in src and '_new_multiply' in src)
    sc = demo_scene(base_settings(32, 24))
    check('the demo scene still builds (nothing in scenebuild moved)',
          sc is not None and len(sc.materials) > 0)


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
    print('all R253 convert-buttons checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
