"""Legacy importer tests: the classic .blend parser, the BI -> Halcyon
material mapping, and the node-tree construction, all without Blender.

Run with:  python3 -m halcyon.tests.test_legacy

The fixtures are real classic-encoded .blend files written by
legacy_fixture.py -- genuine block headers and a genuine DNA1 catalogue --
including a deliberate old-pointer collision between two DATA blocks in
different ID spans, because real files reuse heap addresses and a reader
with one flat pointer map decodes the wrong struct without ever noticing.
"""

import gzip
import types

import numpy as np

from ..core import blend279 as B
from ..core import blend279_map as M
from . import legacy_fixture as FX

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name +
          (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


def _close(a, b, tol=1e-5):
    return abs(float(a) - float(b)) <= tol


# ------------------------------------------------------------------ parser


def test_parse_279():
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data)
    check('2.79 file parses without warnings', not sc['warnings'],
          str(sc['warnings']))
    check('version and pointer size read from the header',
          sc['version'] == '279' and sc['pointer_size'] == 4)

    obs = {o['name']: o for o in sc['objects']}
    check('all six objects arrive',
          set(obs) == {'Cube', 'Plane', 'Spot', 'Camera', 'Rig', 'Path'})
    check('kinds follow Object.type',
          obs['Cube']['kind'] == 'MESH' and obs['Spot']['kind'] == 'LAMP'
          and obs['Camera']['kind'] == 'CAMERA')
    check('selection comes from the scene base list, flag & 1',
          obs['Cube']['selected'] and obs['Spot']['selected']
          and not obs['Plane']['selected'])
    check('obmat translation row survives',
          np.allclose(obs['Cube']['matrix'][3][:3], (1.0, 2.0, 3.0)))

    g = obs['Cube']['data']
    check('cube geometry: 8 verts, 6 quads, 24 loops',
          g['verts'].shape == (8, 3) and len(g['polys']) == 6
          and all(len(p) == 4 for p in g['polys']))
    check('vertex coordinates exact',
          np.allclose(sorted(map(tuple, g['verts'])),
                      sorted(map(tuple, FX.CUBE_VERTS))))
    check('short normals decode to unit-ish diagonals',
          _close(abs(g['normals'][0][0]), 18918 / 32767, 1e-4))
    check('per-face material indices survive',
          g['mat_ids'] == FX.CUBE_MAT_NR)
    check('loop UVs land per corner',
          g['uv_loops'].shape == (24, 2)
          and np.allclose(g['uv_loops'][:4], FX.QUAD_UV))
    check('loop colours decode as bytes / 255',
          g['col_loops'].shape == (24, 4)
          and _close(g['col_loops'][1][2], 200 / 255, 1e-3))

    check('matbits decide which slots the object overrides',
          obs['Cube']['mat_ptrs'] == truth['cube_slots'])
    check('every object type is listed (empty, curve included)',
          obs['Rig']['kind'] == 'EMPTY' and obs['Path']['kind'] == 'CURVE'
          and obs['Rig']['selected'] and not obs['Path']['selected'])
    check('curve material slots resolve like mesh ones',
          obs['Path']['mat_ptrs'] == truth['curve_slots'])

    skin = sc['materials'][truth['materials']['skin']]
    check('material fields: colour, hardness, shader pair',
          _close(skin['r'], 0.8) and skin['har'] == 50
          and skin['diff_shader'] == 1 and skin['spec_shader'] == 2)
    check('the sparse mtex[18] array yields exactly its two slots',
          len(skin['slots']) == 2)
    s0, s5 = skin['slots']
    check('image slot: UV coords, colour channel, factor, uv name',
          s0['texco'] == 16 and s0['mapto'] == 1
          and _close(s0['colfac'], 0.85) and s0['uvname'] == 'UVMap')
    check('image path travels through Tex -> Image',
          s0['tex']['kind'] == 'IMAGE'
          and s0['tex']['image_path'] == '//textures/tex.png')
    check('packed image bytes come out of the file itself',
          s0['tex'].get('packed') == FX.PACKED_PAYLOAD)
    check('procedural slot: clouds with its noise fields',
          s5['tex']['kind'] == 'CLOUDS'
          and _close(s5['tex']['noisesize'], 0.35)
          and s5['tex']['noisedepth'] == 3)
    cb = s5['tex'].get('colorband')
    check('the colorband arrives: stops, positions, ipotype',
          cb is not None and len(cb['stops']) == 3
          and cb['ipotype'] == 1
          and _close(cb['stops'][1][0], 0.5)
          and _close(cb['stops'][1][1], 0.9))
    check('full Tex fields read (basis, flag, bright/contrast)',
          s5['tex']['noisebasis'] == 0 and s5['tex']['flag'] == 1
          and _close(s5['tex']['bright'], 1.1)
          and _close(s5['tex']['contrast'], 0.9))
    check('slot mapping offset and size read as vectors',
          np.allclose(s5['ofs'], (0.0, 0.25, 0.0))
          and np.allclose(s5['size'], (2.0, 2.0, 2.0)))

    chrome = sc['materials'][truth['materials']['chrome']]
    check('mode bits and mirror fields',
          bool(chrome['mode'] & B.MA_RAYMIRROR)
          and _close(chrome['ray_mirror'], 0.75))
    check('toon parameters arrive in param[4]',
          _close(chrome['param'][0], 0.6) and _close(chrome['param'][1],
                                                     0.05))

    la = obs['Spot']['data']
    check('lamp fields: spot in radians, energy, colour',
          la['kind'] == 'SPOT' and _close(la['spotsize'], 0.7853982)
          and _close(la['energy'], 1.5) and _close(la['g'], 0.9))
    ca = obs['Camera']['data']
    check("camera 'type' reads at the file's declared width (char)",
          ca['type'] == 0 and _close(ca['lens'], 35.0))
    check('world gradient and mist fields',
          _close(sc['world']['zenb'], 0.8) and sc['world']['skytype'] == 3
          and _close(sc['world']['misi'], 0.2))


def test_geometry_flag():
    """geometry=False skips vertex extraction but keeps slots/materials --
    the appender path's cheap parse."""
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data, geometry=False)
    cube = [o for o in sc['objects'] if o['name'] == 'Cube'][0]
    check('geometry=False: mesh payload carries no vertices',
          'verts' not in (cube['data'] or {}))
    check('geometry=False: slots still matbits-resolved',
          cube['mat_ptrs'] == truth['cube_slots'])
    check('geometry=False: materials still complete',
          len(sc['materials'][truth['materials']['skin']]['slots']) == 2)


def test_pointer_collision_scoped():
    """The cube's MVert DATA and Skin's first MTex DATA share ONE old
    pointer. Both must decode correctly -- that is the span map working."""
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data)
    g = [o for o in sc['objects'] if o['name'] == 'Cube'][0]['data']
    skin = sc['materials'][truth['materials']['skin']]
    check('collided vertex block still decodes as vertices',
          np.allclose(sorted(map(tuple, g['verts'])),
                      sorted(map(tuple, FX.CUBE_VERTS))))
    check('collided mtex block still decodes as a texture slot',
          skin['slots'][0]['texco'] == 16
          and _close(skin['slots'][0]['colfac'], 0.85))
    check('a DATA block shadowing a Tex pointer cannot steal the hop',
          skin['slots'][0]['tex']['kind'] == 'IMAGE'
          and skin['slots'][0]['tex'].get('image_path')
          == '//textures/tex.png')


def test_parse_249_era():
    data, truth = FX.build_249()
    sc = B.read_legacy_scene(data)
    check('2.49 file parses', sc['version'] == '249', str(sc['warnings']))
    obs = {o['name']: o for o in sc['objects']}
    check('name[24] IDs decode (prefix stripped)',
          'Old' in obs and 'Lamp' in obs)
    g = obs['Old']['data']
    check('MFace era: a quad and a tri, v4 == 0 marks the tri',
          g['polys'] == [(0, 1, 2, 3), (3, 2, 4)])
    check('TFace UVs unpack per corner (4 then 3)',
          g['uv_loops'].shape == (7, 2)
          and np.allclose(g['uv_loops'][4:7],
                          [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5)]))
    ma = sc['materials'][truth['material']]
    check('2.4x material: ztransp alpha, hardness, phong',
          _close(ma['alpha'], 0.65) and ma['har'] == 96
          and ma['spec_shader'] == 1)
    check('fields the era lacks come back as None, not garbage',
          ma['slots'][0]['difffac'] is None)
    check('param[4] falls back to the classic defaults when absent',
          ma['param'] == (0.5, 0.1, 0.5, 0.1))
    check('marble slot with 2.4x factor set',
          ma['slots'][0]['tex']['kind'] == 'MARBLE'
          and _close(ma['slots'][0]['colfac'], 1.0))
    check('degree-era spot size arrives raw (conversion is mapping-side)',
          _close(obs['Lamp']['data']['spotsize'], 45.0))


def test_parse_variants():
    data, truth = FX.build_279()
    zc = B.read_legacy_scene(gzip.compress(data))
    check('gzip-compressed files decompress and parse',
          len(zc['objects']) == 6)

    d8, t8 = FX.build_279(psize=8)
    s8 = B.read_legacy_scene(d8)
    cube8 = [o for o in s8['objects'] if o['name'] == 'Cube'][0]
    check('8-byte pointer files parse identically',
          s8['pointer_size'] == 8
          and cube8['mat_ptrs'] == t8['cube_slots']
          and len(s8['materials'][t8['materials']['skin']]['slots']) == 2)

    dbe, tbe = FX.build_279(end='>')
    sbe = B.read_legacy_scene(dbe)
    check('big-endian files parse identically',
          sbe['endian'] == '>'
          and sbe['materials'][tbe['materials']['skin']]['har'] == 50)

    try:
        B.read_legacy_scene(FX.build_modern_stub())
        check('a 2.80+ file is refused with a helpful error', False)
    except B.BlendError as e:
        check('a 2.80+ file is refused with a helpful error',
              'legacy importer reads 2.79' in str(e))
    try:
        B.read_legacy_scene(b'GIF89a definitely not a blend file......')
        check('junk input is refused', False)
    except B.BlendError:
        check('junk input is refused', True)


# ----------------------------------------------------------------- mapping


def test_material_mapping_279():
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data)
    skin = M.material_spec(sc['materials'][truth['materials']['skin']], 279)
    check('the BI node keeps BOTH menus: Oren-Nayar diffuse AND Blinn '
          'spec, no trade at all',
          skin['bi'] is not None
          and skin['bi']['diff_shader'] == 'OREN_NAYAR'
          and skin['bi']['spec_shader'] == 'BLINN',
          str(skin.get('bi')))
    check('and the old highlight-trade warning is GONE',
          not any('highlight' in w for w in skin['warnings']),
          str(skin['warnings']))
    ins = skin['inputs']
    check('Oren roughness travels RAW (no clamp to the master scale)',
          'Roughness' in ins)
    check("Blinn carries the file's own Refr slider",
          _close(ins['IOR'], 4.0) or ins['IOR'] >= 1.0)
    check('colour, level, gloss map across',
          ins['Diffuse Color'][:3] == tuple(
              np.float32((0.8, 0.55, 0.45)).tolist())
          and _close(ins['Diffuse Level'], 0.8)
          and _close(ins['Glossiness'], 50.0))
    check('emit 0 leaves Self-Illumination unset',
          'Self-Illumination' not in ins)
    check('no transparency bit -> Opacity untouched', 'Opacity' not in ins)
    check('translucency 0.1 sets the socket without changing model',
          _close(ins['Translucency'], 0.1))
    tex = skin['textures']
    check('two slots -> two graph entries', len(tex) == 2)
    img = [t for t in tex if t['node'] == 'ShaderNodeTexImage'][0]
    check('image entry: UV coords, texture_rgb_blend against the base '
          'colour (an image yields RGB, so its alpha is the per-pixel '
          'factor)',
          img['coords'] == 'UV' and img['target'] == 'Diffuse Color'
          and _close(img['rgbblend']['factor'], 0.85)
          and img['rgbblend']['blend'] == 'MIX'
          and img['rgbblend']['tex_rgb'] is True
          and _close(img['rgbblend']['base'][0], 0.8))
    check('image path and uv name travel with the entry',
          img['image']['path'] == '//textures/tex.png'
          and img.get('uv_layer') == 'UVMap')
    nor = [t for t in tex if t['node'] == 'HALCYON_BITextureNode'][0]
    check('normal slot -> BI Texture into Bump Height, Fac output',
          nor['target'] == 'Bump Height' and nor['output'] == 'Fac'
          and nor['coords'] == 'Generated')
    check('norfac travels as the slot strength (master strength 1)',
          _close(ins['Bump Strength'], 1.0)
          and _close(nor.get('scale_fac', 1.0), 0.6))
    p = nor['props']
    check('the Tex block travels verbatim onto the node',
          p['tex_type'] == 'CLOUDS' and _close(p['noise_size'], 0.35)
          and p['noise_depth'] == 3 and p['hard_noise'] is False
          and p['noise_basis'] == 'BLENDER_ORIGINAL')
    check('mtex offset and size live ON the node (classic arithmetic)',
          p['tex_size'] == (2.0, 2.0, 2.0)
          and _close(p['tex_offset'][1], 0.25)
          and ('mapping' not in nor or nor['mapping'] is None))
    check('the colorband rides along: stops + ease interpolation',
          p.get('use_colorband') and len(p['stops']) == 3
          and p['coba_ipotype'] == 'EASE'
          and _close(p['stops'][1][0], 0.5))
    check('bright/contrast arrive on the node',
          _close(p['bright'], 1.1) and _close(p['contrast'], 0.9))

    chrome = M.material_spec(
        sc['materials'][truth['materials']['chrome']], 279)
    check('toon diffuse arrives as the Toon MENU, spec pair intact',
          chrome['bi'] is not None
          and chrome['bi']['diff_shader'] == 'TOON',
          str(chrome.get('bi')))
    check('toon size and smooth from the diffuse param pair',
          _close(chrome['inputs']['Toon Size'], 0.6)
          and _close(chrome['inputs']['Toon Smooth'], 0.05))
    check('ray mirror -> the Mirror PANEL with its colour',
          chrome['bi'].get('use_mirror') is True
          and _close(chrome['inputs']['Reflection'], 0.75)
          and _close(chrome['inputs']['Reflection Color'][2], 1.0))


def test_material_mapping_249():
    data, truth = FX.build_249()
    sc = B.read_legacy_scene(data)
    old = M.material_spec(sc['materials'][truth['material']], 249)
    check('phong spec pair -> the PHONG menu on the BI node',
          old['bi'] is not None and old['bi']['spec_shader'] == 'PHONG',
          str(old.get('bi')))
    check('a 2.4x ztransp (no checkbox bit yet) still switches the '
          'panel on, alpha -> Opacity',
          old['bi'].get('use_transparency') is True
          and old['bi'].get('transp_mode') == 'Z_TRANSPARENCY'
          and _close(old['inputs']['Opacity'], 0.65))
    check('no raytransp -> no Ray IOR forced',
          'Ray IOR' not in old['inputs'])
    t = old['textures'][0]
    check('marble slot: multiply blend rides texture_rgb_blend, and an '
          'intensity texture carries the SLOT colour as tcol',
          t['node'] == 'HALCYON_BITextureNode'
          and t['rgbblend']['blend'] == 'MUL'
          and _close(t['rgbblend']['factor'], 1.0)
          and t['rgbblend']['tex_rgb'] is False
          and 'slot_color' in t['rgbblend'])
    check('era Tex fields default cleanly on the node',
          t['props']['tex_type'] == 'MARBLE'
          and t['props']['marble_type'] == 'SOFT'
          and _close(t['props']['noise_size'], 0.6)
          and _close(t['props']['turbulence'], 5.0)
          and t['props']['use_clamp'] is True)


def test_mapping_units():
    """Channel routing rules on hand-built slots -- the corners the
    fixtures don't reach."""
    def slot(**kw):
        base = {'texco': 32, 'mapto': 1, 'maptoneg': 0, 'blendtype': 0,
                'ofs': (0, 0, 0), 'size': (1, 1, 1), 'uvname': '',
                'colfac': 1.0, 'norfac': 0.5, 'varfac': 1.0,
                'tex': {'kind': 'CLOUDS', 'noisesize': 0.25,
                        'noisedepth': 2, 'noisetype': 0, 'turbul': 5.0}}
        base.update(kw)
        return base

    def mat(**kw):
        m = {'name': 'T', 'r': 0.5, 'g': 0.5, 'b': 0.5, 'specr': 1.0,
             'specg': 1.0, 'specb': 1.0, 'har': 60, 'spec': 0.5,
             'ref': 0.8, 'alpha': 1.0, 'emit': 0.0, 'amb': 1.0,
             'translucency': 0.0, 'ray_mirror': 0.0, 'mode': 0,
             'diff_shader': 0, 'spec_shader': 1, 'param': (0.5, 0.1,
                                                           0.5, 0.1),
             'slots': []}
        m.update(kw)
        return m

    # value channels ride texture_value_blend; 2.4x falls back to varfac
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_SPEC,
                                         specfac=None, varfac=0.4)]))
    t = sp['textures'][0]
    check('value channel: per-channel factor missing -> varfac drives '
          'the blend factor',
          t['target'] == 'Specular Level'
          and _close(t['vblend']['factor'], 0.4))
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_HAR, hardfac=1.0)]))
    check('a hardness channel carries the /128 units and clamp',
          sp['textures'][0]['vblend'].get('scale') == 128.0)

    # negative influence -> invert
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_ALPHA,
                                         maptoneg=B.MAP_ALPHA,
                                         alphafac=1.0)]))
    check('maptoneg bit -> invert entry',
          sp['textures'][0].get('invert') is True)

    # several influence bits on one slot -> several entries
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_COL | B.MAP_NORM,
                                         colfac=0.5)]))
    check('one slot, two influences -> two entries',
          len(sp['textures']) == 2
          and {t['target'] for t in sp['textures']}
          == {'Diffuse Color', 'Bump Height'})

    # several normal slots ALL arrive, each with its own strength
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_NORM, norfac=0.5),
                                    slot(mapto=B.MAP_NORM, norfac=0.3)]))
    bumps = [t for t in sp['textures'] if t['target'] == 'Bump Height']
    check('every bump slot arrives, strengths as multiplies',
          len(bumps) == 2
          and _close(bumps[0].get('scale_fac', 1.0), 0.5)
          and _close(bumps[1]['scale_fac'], 0.3)
          and _close(sp['inputs']['Bump Strength'], 1.0))
    # the channels 1.34.2 skipped
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_COLMIR,
                                         mirrfac=1.0)]))
    check('mirror colour maps to Reflection Color',
          sp['textures'][0]['target'] == 'Reflection Color')
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_AMB, ambfac=0.5)]))
    check('ambient influence maps to Ambient through the value blend',
          sp['textures'][0]['target'] == 'Ambient'
          and _close(sp['textures'][0]['vblend']['factor'], 0.5))
    sp = M.material_spec(mat(slots=[slot(mapto=B.MAP_DISPLACE,
                                         dispfac=0.7)]))
    check('displacement rides the bump input',
          sp['textures'][0]['target'] == 'Bump Height'
          and _close(sp['textures'][0].get('scale_fac', 1.0), 0.7))

    # procedural translations
    def tex_of(kind, **kw):
        t = {'kind': kind, 'noisesize': 0.25, 'noisedepth': 2,
             'noisetype': 0, 'turbul': 5.0}
        t.update(kw)
        return t
    sp = M.material_spec(mat(slots=[slot(tex=tex_of('BLEND', stype=1,
                                                     flag=2))]))
    check('blend: stype and Flip XY arrive verbatim',
          sp['textures'][0]['node'] == 'HALCYON_BITextureNode'
          and sp['textures'][0]['props']['blend_type'] == 'QUAD'
          and sp['textures'][0]['props']['blend_flip'] is True)
    sp = M.material_spec(mat(slots=[slot(tex=tex_of(
        'MUSGRAVE', stype=1, mg_H=0.7, mg_lacunarity=2.3, mg_octaves=5.0,
        mg_offset=0.9, mg_gain=2.0, ns_outscale=1.5))]))
    p = sp['textures'][0]['props']
    check('musgrave: all five parameters travel',
          p['musgrave_type'] == 'RIDGEDMF' and _close(p['mg_h'], 0.7)
          and _close(p['mg_lacunarity'], 2.3)
          and _close(p['mg_octaves'], 5.0) and _close(p['mg_gain'], 2.0)
          and _close(p['ns_outscale'], 1.5))
    sp = M.material_spec(mat(slots=[slot(tex=tex_of('NOISE'))]))
    check('BI Noise -> the BI Texture node, deterministic static',
          sp['textures'][0]['props']['tex_type'] == 'NOISE')
    sp = M.material_spec(mat(slots=[slot(tex=tex_of('CLOUDS',
                                                    noisetype=1))]))
    check('hard noise flag travels',
          sp['textures'][0]['props']['hard_noise'] is True)
    sp = M.material_spec(mat(slots=[slot(tex=tex_of(
        'VORONOI', vn_w1=0.5, vn_w2=-0.5, vn_distm=3, vn_coltype=2,
        vn_mexp=3.0, ns_outscale=0.8))]))
    p = sp['textures'][0]['props']
    check('voronoi: weights, metric and colour mode travel',
          _close(p['vn_w1'], 0.5) and _close(p['vn_w2'], -0.5)
          and p['vn_distm'] == 'CHEBYCHEV'
          and p['vn_coltype'] == 'POSITION_OUTLINE')
    sp = M.material_spec(mat(slots=[slot(tex=tex_of('MAGIC', turbul=7.0,
                                                    noisedepth=4))]))
    check('magic: depth and turbulence travel',
          sp['textures'][0]['props']['tex_type'] == 'MAGIC'
          and sp['textures'][0]['props']['noise_depth'] == 4
          and _close(sp['textures'][0]['props']['turbulence'], 7.0))
    sp = M.material_spec(mat(slots=[slot(tex=tex_of(
        'DISTNOISE', noisebasis=14, noisebasis2=2, dist_amount=3.5))]))
    p = sp['textures'][0]['props']
    check('distorted noise: both bases and the amount travel',
          p['noise_basis'] == 'CELL_NOISE'
          and p['noise_basis2'] == 'IMPROVED_PERLIN'
          and _close(p['dist_amount'], 3.5))
    sp = M.material_spec(mat(slots=[slot(tex=tex_of('CLOUDS',
                                                    noisebasis=1))]))
    check('Original Perlin imports SILENTLY now -- it is the real '
          'orgPerlinNoise, no fallback to warn about',
          not any('Original Perlin' in w for w in sp['warnings'])
          and sp['textures'][0]['props']['noise_basis']
          == 'ORIGINAL_PERLIN')
    sp = M.material_spec(mat(slots=[slot(tex=tex_of('ENVMAP'))]))
    check('envmap slots are skipped with a note',
          not sp['textures'] and any('EnvMap' in w for w in sp['warnings']))
    # an image slot with ofs/size folds the classic pipeline into Mapping.
    # texco_mapping (verbatim, R158 fetch) runs size*(uv-0.5)+ofs+0.5, and
    # Mapping computes scale*uv+location, so Location = ofs + (1-size)/2.
    # The old pin (size*ofs-size+1)/2 coincides at ofs=0 or size=2 -- the
    # values below distinguish the two laws on BOTH axes.
    sp = M.material_spec(mat(slots=[slot(
        tex={'kind': 'IMAGE', 'image_path': '//t.png', 'image_name': 't'},
        ofs=(0.25, -0.3, 0.0), size=(3.0, 0.5, 1.0))]))
    mp2 = sp['textures'][0]['mapping']
    check('image ofs/size fold to Location=ofs+(1-size)/2, Scale=size '
          '(verbatim texco_mapping)',
          _close(mp2['scale'][0], 3.0) and _close(mp2['scale'][1], 0.5)
          and _close(mp2['location'][0], 0.25 + 0.5 * (1.0 - 3.0))
          and _close(mp2['location'][1], -0.3 + 0.5 * (1.0 - 0.5))
          and _close(mp2['location'][2], 0.0), str(mp2))

    # shadeless / wire / shaders -- the BI node keeps every pair
    check('shadeless -> the Shadeless toggle on the BI node',
          M.material_spec(mat(mode=B.MA_SHLESS))['bi']['shadeless']
          is True)
    check('wire -> WIREFRAME on the master (the one look the BI node '
          'does not carry)',
          M.material_spec(mat(mode=B.MA_WIRE))['model'] == 'WIREFRAME'
          and M.material_spec(mat(mode=B.MA_WIRE))['bi'] is None)
    check('oren-nayar keeps its menu AND its raw roughness',
          M.material_spec(mat(diff_shader=1, spec=0.05,
                              roughness=0.85))['bi']['diff_shader']
          == 'OREN_NAYAR'
          and _close(M.material_spec(
              mat(diff_shader=1, spec=0.05,
                  roughness=0.85))['inputs']['Roughness'], 0.85))
    check('minnaert darkness travels RAW (BI units, no /2 squeeze)',
          _close(M.material_spec(mat(diff_shader=3, spec=0.05,
                                     darkness=1.5))['inputs']['Darkness'],
                 1.5))
    check('fresnel diffuse -> the FRESNEL menu with its own pair',
          M.material_spec(mat(diff_shader=4,
                              spec=0.0))['bi']['diff_shader']
          == 'FRESNEL'
          and 'BI Fresnel' in M.material_spec(mat(diff_shader=4,
                                                  spec=0.0))['inputs'])
    check('cook-torrance pair keeps the CookTorr menu',
          M.material_spec(mat(spec_shader=0))['bi']['spec_shader']
          == 'COOKTORR')
    check('wardiso keeps its menu with the rms Slope RAW',
          M.material_spec(mat(spec_shader=4,
                              rms=0.2))['bi']['spec_shader'] == 'WARDISO'
          and _close(M.material_spec(mat(spec_shader=4,
                                         rms=0.2))['inputs']['Slope'],
                     0.2))
    # raytransp switches the panel to RAYTRACE with its own Ray IOR
    sp = M.material_spec(mat(mode=B.MA_RAYTRANSP, alpha=0.3, ang=1.45,
                             filter=0.2))
    check('raytransp -> RAYTRACE mode, opacity, Ray IOR and Filter',
          sp['bi']['use_transparency'] is True
          and sp['bi']['transp_mode'] == 'RAYTRACE'
          and _close(sp['inputs']['Opacity'], 0.3)
          and _close(sp['inputs']['Ray IOR'], 1.45)
          and _close(sp['inputs']['Filter'], 0.2))
    # emit is BI's own float slider on the node
    sp = M.material_spec(mat(emit=2.0))
    check("emit rides the node's Emit slider, not a colour synth",
          _close(sp['inputs']['Emit'], 2.0)
          and 'Self-Illumination' not in sp['inputs'])
    # the panel flags decode from the mode bits
    sp = M.material_spec(mat(mode=B.MA_SHADOW | B.MA_TANGENT_V
                             | B.MA_VERTEXCOL | B.MA_NOMIST
                             | B.MA_ONLYCAST,
                             shade_flag=B.MA_CUBIC, mode2=0))
    check('mode bits -> the panel flags, 1:1',
          sp['bi']['shadow_receive'] is True
          and sp['bi']['use_tangent_v'] is True
          and sp['bi']['vcol_light'] is True
          and sp['bi']['use_mist'] is False
          and sp['bi']['shadow_cast_only'] is True
          and sp['bi']['use_cubic'] is True
          and sp['bi']['shadow_cast'] is False,
          str(sp['bi']))
    # ramps arrive with stops, enums and factor -- only when switched on
    band = {'stops': [(0.0, 0.0, 0.0, 0.0, 1.0),
                      (1.0, 1.0, 0.2, 0.1, 0.8)], 'ipotype': 1}
    sp = M.material_spec(mat(mode=B.MA_RAMP_COL, ramp_col_band=band,
                             rampin_col=1, rampblend_col=4,
                             rampfac_col=0.7))
    check('a diffuse ramp maps: input ENERGY, blend SCREEN, factor, '
          'ease stops',
          sp['bi'].get('use_ramp_dif') is True
          and sp['bi']['ramp_dif_input'] == 'ENERGY'
          and sp['bi']['ramp_dif_blend'] == 'SCREEN'
          and _close(sp['bi']['ramp_dif_factor'], 0.7)
          and sp['bi']['ramp_dif_ipo'] == 1
          and len(sp['bi']['ramp_dif_stops']) == 2
          and _close(sp['bi']['ramp_dif_stops'][1][4], 0.8),
          str(sp['bi'].get('ramp_dif_input')))
    sp = M.material_spec(mat(ramp_col_band=band))
    check('a colorband WITHOUT its mode bit stays off (bands persist '
          'once created; phantom.blend proves it)',
          not sp['bi'].get('use_ramp_dif'))
    # the light group resolves to lamp names at parse time
    sp = M.material_spec(mat(group_name='Rim', mode=B.MA_GROUP_NOLAY,
                             group_lights=['Rim.L', 'Rim.R']))
    check('a light group arrives as name + lamp list + Exclusive',
          sp['bi']['light_group'] == 'Rim'
          and sp['bi']['light_group_exclusive'] is True
          and sp['bi']['light_group_lights'] == ['Rim.L', 'Rim.R'])


def test_field_import_fidelity():
    """The five-blend field round: coordinates, value blends, Original
    Perlin, the vcol gate -- each defect's fix, held forever."""
    import numpy as np
    from ..core import bitex as BX
    from ..core import blend279_map as M2

    # ---- 1) the TEXCO table matches 2.79 DNA (the scrambled first cut
    # sent REFL env maps to Object space: the field's grey matcaps)
    check('TEXCO constants are the DNA values',
          (B.TEXCO_ORCO, B.TEXCO_REFL, B.TEXCO_NORM, B.TEXCO_GLOB,
           B.TEXCO_UV, B.TEXCO_OBJECT) == (1, 2, 4, 8, 16, 32))
    w = []
    check('REFL routes to the reflection matcap, NOR to the normal one',
          M2._texcoord(2, w, 'm') == 'MATCAP_REFLECT'
          and M2._texcoord(4, w, 'm') == 'MATCAP_NORMAL'
          and M2._texcoord(32, w, 'm') == 'Object'
          and M2._texcoord(1, w, 'm') == 'Generated'
          and M2._texcoord(16, w, 'm') == 'UV')

    # ---- 2) texture_value_blend: the C's own shape
    tin = np.array([0.0, 0.25, 0.5, 1.0], np.float32)
    base = np.full(4, 0.4, np.float32)
    got = BX.texture_value_blend(1.0, base, tin, 1.0, 'MIX')
    check('MIX blends the BASE toward DVar by intensity (never "tex '
          'becomes the value")',
          np.allclose(got, tin * 1.0 + (1 - tin) * 0.4, atol=1e-6),
          str(got))
    got = BX.texture_value_blend(1.0, base, tin, 1.0, 'MUL')
    check('MUL is (1-fac+Tin*fac*DVar)*base',
          np.allclose(got, (0.0 + tin) * 0.4, atol=1e-6))
    got = BX.texture_value_blend(1.0, base, tin, -1.0, 'MIX')
    check('a NEGATIVE factor flips the blend weights, as the C does',
          np.allclose(got, (1 - tin) * 1.0 + tin * 0.4, atol=1e-6))
    check('HUE on a value channel is ZERO, exactly the untouched C '
          'default',
          np.allclose(BX.texture_value_blend(1.0, base, tin, 1.0,
                                             'HUE'), 0.0))

    # ---- 3) the spec carries vblend for BI value channels
    def mkslot(**kw):
        s = {'texco': 16, 'mapto': B.MAP_HAR, 'maptoneg': 0,
             'blendtype': 1, 'ofs': (0, 0, 0), 'size': (1, 1, 1),
             'uvname': '', 'hardfac': 0.8, 'def_var': 1.0,
             'tex': {'kind': 'CLOUDS', 'noisesize': 0.25,
                     'noisedepth': 2, 'noisetype': 0}}
        s.update(kw)
        return s
    sp = M2.material_spec({'name': 'M', 'har': 100, 'spec': 0.5,
                           'mode': 0, 'slots': [mkslot()]}, 279)
    t = sp['textures'][0]
    vb = t.get('vblend')
    check('a hardness slot rides texture_value_blend: MUL, the SIGNED '
          'factor, DVar, /128 units and the 1..511 clamp',
          vb is not None and vb['blend'] == 'MUL'
          and abs(vb['factor'] - 0.8) < 1e-6
          and abs(vb['dvar'] - 1.0) < 1e-6
          and abs(vb['base'] - 100.0 / 128.0) < 1e-6
          and vb.get('scale') == 128.0 and vb.get('clamp') == (1.0, 511.0),
          str(vb))
    sp = M2.material_spec({'name': 'M', 'har': 50, 'spec': 0.5,
                           'mode': 0, 'emit': 0.0,
                           'slots': [mkslot(mapto=B.MAP_EMIT,
                                            emitfac=1.0,
                                            blendtype=0)]}, 279)
    vb = sp['textures'][0].get('vblend')
    check('an EMIT slot blends the Emit float the same way (the white '
          'self-illumination bug: garbage coords lit it, the blend '
          'shapes it)',
          vb is not None and vb['blend'] == 'MIX'
          and abs(vb['base'] - 0.0) < 1e-6
          and sp['textures'][0]['target'] == 'Emit', str(vb))

    # ---- 4) Original Perlin: real, table-driven, and NOT the improved
    # fallback any more
    xs = np.linspace(-3.0, 4.0, 257, dtype=np.float32)
    ys = xs * 0.7 + 0.3
    zs = xs * -0.4 + 1.1
    o = BX.org_perlin_noise(xs, ys, zs)
    n = BX.new_perlin(xs, ys, zs)
    check('org_perlin_noise is alive, bounded and NOT improved perlin',
          bool(np.isfinite(o).all()) and float(np.abs(o).max()) <= 1.5001
          and float(o.std()) > 0.05
          and float(np.abs(o - n).max()) > 0.05,
          f'std {o.std():.3f} diff {np.abs(o - n).max():.3f}')
    check('basis 1 dispatches to it (unsigned wrapper)',
          np.allclose(BX.basis_u(1)(xs, ys, zs),
                      0.5 + 0.5 * o, atol=1e-6))

    # ---- 5) the GPU twins agree: influence node and operlin
    from ..core.nodeeval import GraphEvaluator
    from ..gpu import procedural as PR
    from ..shaders.compiler import try_compile
    ok_all = True
    worst_m = ''
    rng = np.random.default_rng(11)
    nq = 128
    base_a = rng.uniform(0, 1, nq).astype(np.float32)
    tin_a = rng.uniform(0, 1, nq).astype(np.float32)
    for mode in ('MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                 'LIGHT', 'SCREEN', 'OVERLAY', 'SOFT', 'LINEAR'):
        for facg in (1.0, 0.6, -0.8):
            ref = BX.texture_value_blend(0.7, base_a, tin_a, facg, mode)
            # the node, via a tiny graph
            g = {'output': 'out', 'nodes': {
                'i': {'id': 'i', 'bl_idname': 'HALCYON_BIInfluenceNode',
                      'props': {'blend': mode},
                      'inputs': [
                          {'name': 'Base', 'type': 'VALUE',
                           'default': None, 'link': None},
                          {'name': 'Intensity', 'type': 'VALUE',
                           'default': None, 'link': None},
                          {'name': 'Factor', 'type': 'VALUE',
                           'default': facg, 'link': None},
                          {'name': 'DVar', 'type': 'VALUE',
                           'default': 0.7, 'link': None}],
                      'outputs': [{'name': 'Value', 'type': 'VALUE'}]}}}
            import types as _t
            ctx = _t.SimpleNamespace(n=nq)
            ev = GraphEvaluator(g, ctx, {}, None)
            base_in = base_a
            tin_in = tin_a
            # drive the two inputs directly through the cache
            ev.cache[('i', 'Base')] = base_in
            ev.cache[('i', 'Intensity')] = tin_in
            from ..core.nodeeval import n_bi_influence
            got = n_bi_influence(
                _t.SimpleNamespace(
                    input=lambda node, name, kind: {
                        'Base': base_in, 'Intensity': tin_in,
                        'Factor': np.full(nq, facg, np.float32),
                        'DVar': np.full(nq, 0.7, np.float32)}[name],
                    ctx=ctx, n=nq),
                g['nodes']['i'])['Value']
            e = float(np.abs(got - ref).max())
            if e > 1e-6:
                ok_all = False
                worst_m = f'{mode}/{facg}: {e:.2e}'
    check('the Influence node matches the C transcription across all '
          '12 modes and the sign flip', ok_all, worst_m)

    # ---- 6) the GPU orgPerlin twin, through the bitex table texture
    from ..core.bitex_tables import table_pixels
    from ..core.texture import Texture
    from ..gpu.procedural import OKRAMP_GLSL, PATTERN_GLSL
    src = OKRAMP_GLSL + PATTERN_GLSL['bitex'] + """
uniform vec3 pin;
out vec4 Color;
void main() { Color = vec4(bi_operlin(pin), 0.0, 0.0, 1.0); }
"""
    prog, err = try_compile(src, 'GLSL')
    check('bi_operlin compiles into the BI GLSL library',
          prog is not None, str(err))
    if prog is not None:
        tab = Texture(table_pixels(), colorspace='Non-Color',
                      filt='NEAREST', wrap='EXTEND')
        P = rng.uniform(-3, 3, (256, 3)).astype(np.float32)
        cpu = BX.org_perlin_noise(P[:, 0], P[:, 1], P[:, 2])
        out, _d = prog.run({'pin': P, 'hal_bitex_tab': tab}, {}, 256)
        e = float(np.abs(out['Color'][:, 0] - cpu).max())
        check('Original Perlin matches CPU vs GPU exactly', e < 1e-5,
              f'max {e:.2e}')

    # ---- 7) scene layers: hidden-layer objects stay OUT (the field's
    # five hidden lamps and the floating hand)
    from .. import legacy_import as LI
    sdata = {'scene_lay': 0x2, 'objects': [
        {'name': 'lit', 'kind': 'LAMP', 'layers': 0x2, 'selected': False},
        {'name': 'hid', 'kind': 'LAMP', 'layers': 0x1, 'selected': False},
        {'name': 'hand', 'kind': 'MESH', 'layers': 0x1,
         'selected': False},
        {'name': 'body', 'kind': 'MESH', 'layers': 0x8002,
         'selected': False}]}
    names, _per = LI.plan_objects(sdata, False, True, True)
    check('hidden-layer lamps and meshes are skipped, visible kept',
          names == ['lit', 'body'], str(names))
    names2, _p2 = LI.plan_objects(sdata, False, True, True,
                                  include_hidden=True)
    check('...unless hidden objects are asked for',
          len(names2) == 4)

    # ---- 8) the RENDER-ACTIVE UV layer is the primary at export (the
    # field's head sampled the edit-active layer: white face)
    import types as _t2
    from .. import export as EX

    class _UVData:
        def __init__(self, val):
            self._v = val

        def foreach_get(self, key, buf):
            buf[:] = self._v
    lay_a = _t2.SimpleNamespace(name='UVMap', active_render=False,
                                data=_t2.SimpleNamespace(
                                    foreach_get=lambda b: b.__setitem__(
                                        slice(None), 0.25)))
    lay_b = _t2.SimpleNamespace(name='UV_SINGLE', active_render=True,
                                data=_t2.SimpleNamespace(
                                    foreach_get=lambda b: b.__setitem__(
                                        slice(None), 0.75)))
    picked = next((i for i, l in enumerate([lay_a, lay_b])
                   if getattr(l, 'active_render', False)), 0)
    check('the export primary-layer pick lands on active_render',
          picked == 1)

    # ---- 9) a named-layer slot wires a UV Map node in build_spec
    socks2, specs2 = _shader_socket_table()
    from .. import templates as TPL
    m3 = _fake_material(socks2, specs2)
    spec3 = {'model': None,
             'bi': {'diff_shader': 'LAMBERT', 'spec_shader': 'COOKTORR'},
             'inputs': {},
             'textures': [{'node': 'ShaderNodeTexImage', 'props': {},
                           'inputs': {}, 'output': 'Color',
                           'target': 'Diffuse Color', 'coords': 'UV',
                           'uv_layer': 'Blood'}]}
    TPL.build_spec(m3, spec3)
    uvn = [n for n in m3.node_tree.nodes
           if n.bl_idname == 'ShaderNodeUVMap']
    check("a slot naming its UV layer gets a UV Map node ('Blood'), "
          'not the active stand-in',
          len(uvn) == 1 and getattr(uvn[0], 'uv_map', '') == 'Blood'
          and any(a.node is uvn[0] and b.name == 'Vector'
                  for a, b in m3.node_tree.links), str(len(uvn)))


def test_bi_colour_channels():
    """The mask-material round: colour channels are texture_rgb_blend, and
    a slot's texflag decides what the channel actually receives.

    The field's puppet pinned all of it: 'Black' (CLOUDS + colorband +
    RGBToIntensity, slot colour black, COLSPEC|HAR at MUL) rendered a
    waxy constant sheen because the colour channel rode a MixRGB at a
    CONSTANT factor and the value channel took the raw pre-band noise;
    BI's own answers are specular colour ZERO (tcol is the slot colour,
    black) and hardness driven by the BAND's luminance."""
    import types as _t
    import numpy as np
    from ..core import bitex as BX
    from ..core import blend279_map as M2
    from ..core.nodeeval import n_bi_influence, n_bi_rgb_blend

    # ---- 1) the scalar reference: texture_rgb_blend against a direct
    # per-channel transcription of the VERBATIM C (R155 fetch). The
    # value twin is NOT the reference any more: the C's rgb fn uses
    # facm = 1-fact for MUL/SCREEN/OVERLAY where the value fn rederives
    # 1-facg, and DARK is a mix-toward-min in both.
    def _ref_rgb(tex, out, fact, facg, mode):
        f = fact * facg
        fm = 1.0 - f
        if mode == 'MIX':
            return f * tex + fm * out
        if mode == 'MUL':
            return (fm + f * tex) * out
        if mode == 'SCREEN':
            return 1.0 - (fm + f * (1.0 - tex)) * (1.0 - out)
        if mode == 'OVERLAY':
            return (out * (fm + 2.0 * f * tex) if out < 0.5 else
                    1.0 - (fm + 2.0 * f * (1.0 - tex)) * (1.0 - out))
        if mode == 'SUB':
            return -f * tex + out
        if mode == 'ADD':
            return f * tex + out
        if mode == 'DIV':
            return fm * out + f * out / tex if tex != 0.0 else out
        if mode == 'DIFF':
            return fm * out + f * abs(tex - out)
        if mode == 'DARK':
            return min(out, tex) * f + out * fm
        if mode == 'LIGHT':
            return max(f * tex, out)
        return None
    rng = np.random.default_rng(7)
    nq = 96
    base3 = rng.uniform(0, 1, (nq, 3)).astype(np.float32)
    tex3 = rng.uniform(0, 1, (nq, 3)).astype(np.float32)
    tin = rng.uniform(0, 1, nq).astype(np.float32)
    ok = True
    worst = ''
    for mode in ('MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                 'LIGHT', 'SCREEN', 'OVERLAY'):
        got = BX.texture_rgb_blend(tex3, base3, tin, 0.85, mode)
        ref = np.asarray([
            _ref_rgb(float(tex3[i, c]), float(base3[i, c]),
                     float(tin[i]), 0.85, mode)
            for i in range(nq) for c in range(3)],
            np.float32).reshape(nq, 3)
        e = float(np.abs(got - ref).max())
        if e > 1e-5:
            ok = False
            worst = f'{mode}: {e:.2e}'
    check('texture_rgb_blend matches the verbatim C per channel across '
          'the ten direct shapes', ok, worst)
    # and the value twin's own verbatim quirks hold: DARK mixes toward
    # the min, SOFT leaves its last term unscaled
    vd = BX.texture_value_blend(0.3, np.float32(0.8), np.float32(0.5),
                                1.0, 'DARK')
    check('value DARK is min-mix (verbatim), not min(fact*tex,out)',
          abs(float(vd) - (min(0.8, 0.3) * 0.5 + 0.8 * 0.5)) < 1e-6,
          str(vd))
    vs_ = BX.texture_value_blend(0.4, np.float32(0.6), np.float32(0.5),
                                 1.0, 'SOFT')
    scf = 1.0 - (1.0 - 0.4) * (1.0 - 0.6)
    ref_soft = 0.5 * 0.6 + 0.5 * ((1.0 - 0.6) * 0.4 * 0.6) + 0.6 * scf
    check('value SOFT leaves the (out*scf) term unscaled (verbatim)',
          abs(float(vs_) - ref_soft) < 1e-6, f'{float(vs_)} vs {ref_soft}')
    hue = BX.texture_rgb_blend(tex3, base3, tin, 1.0, 'HUE')
    check('HUE delegates to ramp_blend (real for colours, where the '
          'value fn returned 0)',
          hue.shape == (nq, 3) and float(np.abs(hue).max()) > 0.0
          and not np.allclose(hue, 0.0))

    # ---- 2) the node plumbing: intensity texture -> slot colour tcol
    def _ev(vals):
        return _t.SimpleNamespace(
            input=lambda node, name, kind: vals[name],
            ctx=_t.SimpleNamespace(n=nq), n=nq)
    ones4 = np.ones((nq, 4), np.float32)
    slotc = np.zeros((nq, 4), np.float32)
    slotc[:, 0] = 0.2
    slotc[:, 2] = 0.9
    node = {'bl_idname': 'HALCYON_BIRGBBlendNode',
            'props': {'blend': 'MUL', 'tex_rgb': False},
            'inputs': [], 'outputs': [{'name': 'Color'}]}
    got = n_bi_rgb_blend(_ev({'Base': np.concatenate(
        [base3, np.ones((nq, 1), np.float32)], 1),
        'Intensity': tin, 'Alpha': np.ones(nq, np.float32),
        'Color': ones4, 'Slot Color': slotc,
        'Factor': np.ones(nq, np.float32)}), node)['Color'][:, :3]
    ref = BX.texture_rgb_blend(slotc[:, :3], base3, tin, 1.0, 'MUL')
    check('an intensity texture blends the SLOT colour by per-pixel '
          'intensity x slider',
          float(np.abs(got - ref).max()) < 1e-6)

    # black slot colour at MUL/full influence: the verbatim C keeps
    # (1-Tin)*base -- facm is 1-fact, so the channel DIMS with the
    # texture rather than zeroing (the puppet body's specular truth,
    # corrected by the R155 source fetch)
    black = np.zeros((nq, 4), np.float32)
    gotb = n_bi_rgb_blend(_ev({'Base': np.concatenate(
        [base3, np.ones((nq, 1), np.float32)], 1),
        'Intensity': tin, 'Alpha': np.ones(nq, np.float32),
        'Color': ones4, 'Slot Color': black,
        'Factor': np.ones(nq, np.float32)}), node)['Color'][:, :3]
    refb = (1.0 - tin)[:, None] * base3
    check("MUL against a BLACK slot colour keeps (1-Tin)*base -- the "
          "verbatim C's facm=1-fact (the puppet body's textured dim)",
          float(np.abs(gotb - refb).max()) < 1e-6)

    # ---- 3) an RGB texture: tcol is the texture, the factor its ALPHA
    node_rgb = {'bl_idname': 'HALCYON_BIRGBBlendNode',
                'props': {'blend': 'MIX', 'tex_rgb': True},
                'inputs': [], 'outputs': [{'name': 'Color'}]}
    alpha = rng.uniform(0, 1, nq).astype(np.float32)
    col4 = np.concatenate([tex3, alpha[:, None]], 1)
    got = n_bi_rgb_blend(_ev({'Base': np.concatenate(
        [base3, np.ones((nq, 1), np.float32)], 1),
        'Intensity': np.full(nq, 0.123, np.float32), 'Alpha': alpha,
        'Color': col4, 'Slot Color': slotc,
        'Factor': np.full(nq, 0.85, np.float32)}), node_rgb)['Color'][:, :3]
    ref = BX.texture_rgb_blend(tex3, base3, alpha, 0.85, 'MIX')
    check('an RGB texture supplies tcol and its ALPHA as the factor '
          '(the intensity input is ignored)',
          float(np.abs(got - ref).max()) < 1e-6)

    # ---- 4) RGBToIntensity collapses to Rec.709 luminance FIRST, and
    # the slot colour returns as tcol -- Black, exactly
    node_r2i = {'bl_idname': 'HALCYON_BIRGBBlendNode',
                'props': {'blend': 'MUL', 'tex_rgb': True,
                          'rgbtoint': True},
                'inputs': [], 'outputs': [{'name': 'Color'}]}
    got = n_bi_rgb_blend(_ev({'Base': np.concatenate(
        [base3, np.ones((nq, 1), np.float32)], 1),
        'Intensity': np.zeros(nq, np.float32), 'Alpha': alpha,
        'Color': col4, 'Slot Color': slotc,
        'Factor': np.ones(nq, np.float32)}), node_r2i)['Color'][:, :3]
    ref = BX.texture_rgb_blend(slotc[:, :3], base3,
                               BX.rec709_lum(tex3), 1.0, 'MUL')
    check('RGBToIntensity: luminance becomes the factor and the slot '
          'colour becomes tcol',
          float(np.abs(got - ref).max()) < 1e-6)

    # ---- 5) the value channel under the same flags: hardness from the
    # BAND's luminance, not the raw noise (the porcelain marbling)
    node_v = {'bl_idname': 'HALCYON_BIInfluenceNode',
              'props': {'blend': 'MUL', 'tex_rgb': True,
                        'rgbtoint': True},
              'inputs': [], 'outputs': [{'name': 'Value'}]}
    raw = rng.uniform(0, 1, nq).astype(np.float32)
    got = n_bi_influence(_ev({'Base': np.full(nq, 85.0 / 128.0,
                                              np.float32),
                              'Intensity': raw, 'Alpha': alpha,
                              'Color': col4,
                              'Factor': np.full(nq, 0.95, np.float32),
                              'DVar': np.ones(nq, np.float32)}),
                         node_v)['Value']
    ref = BX.texture_value_blend(1.0, np.full(nq, 85.0 / 128.0,
                                              np.float32),
                                 BX.rec709_lum(tex3), 0.95, 'MUL')
    check('a value channel under RGBToIntensity blends by the COLOUR '
          "chain's luminance (the mask's banded hardness)",
          float(np.abs(got - ref).max()) < 1e-6)
    # and negative inverts it
    node_vn = {'bl_idname': 'HALCYON_BIInfluenceNode',
               'props': {'blend': 'MUL', 'tex_rgb': True,
                         'rgbtoint': True, 'negative': True},
               'inputs': [], 'outputs': [{'name': 'Value'}]}
    gotn = n_bi_influence(_ev({'Base': np.full(nq, 85.0 / 128.0,
                                               np.float32),
                               'Intensity': raw, 'Alpha': alpha,
                               'Color': col4,
                               'Factor': np.full(nq, 0.95, np.float32),
                               'DVar': np.ones(nq, np.float32)}),
                          node_vn)['Value']
    refn = BX.texture_value_blend(1.0, np.full(nq, 85.0 / 128.0,
                                               np.float32),
                                  1.0 - BX.rec709_lum(tex3), 0.95, 'MUL')
    check('Negative inverts after the collapse, exactly '
          'do_material_tex order',
          float(np.abs(gotn - refn).max()) < 1e-6)

    # ---- 6) the converter emits the flags from the real DNA: a
    # Black-shaped material through material_spec
    mat = {'name': 'Black', 'mode': 0, 'diff_shader': 0, 'spec_shader': 2,
           'r': 0.013, 'g': 0.013, 'b': 0.013, 'ref': 1.0, 'spec': 1.0,
           'specr': 0.0116, 'specg': 0.0116, 'specb': 0.0116, 'har': 85,
           'amb': 1.0, 'emit': 0.0, 'alpha': 1.0,
           'slots': [{'texco': 16, 'mapto': 260, 'maptoneg': 0,
                      'blendtype': 1, 'texflag': 17665,
                      'r': 0.0, 'g': 0.0, 'b': 0.0,
                      'colspecfac': 1.0, 'hardfac': 0.95, 'def_var': 1.0,
                      'uvname': '',
                      'tex': {'kind': 'CLOUDS', 'name': 'T', 'flag': 9,
                              'stype': 0, 'noisebasis': 0,
                              'noisetype': 1, 'noisesize': 0.12,
                              'noisedepth': 2, 'bright': 1.0,
                              'contrast': 1.1,
                              'colorband': {'stops': [
                                  (0.0, 0, 0, 0, 0),
                                  (1.0, 1, 1, 1, 1)], 'ipotype': 0}}}]}
    sp = M2.material_spec(mat, 279)
    ent = {e['target']: e for e in sp['textures']}
    rb = ent['Specular Color'].get('rgbblend')
    vb = ent['Glossiness'].get('vblend')
    check('a COLSPEC slot emits rgbblend with the DNA flags and the '
          'slot colour',
          rb is not None and rb['blend'] == 'MUL'
          and rb['tex_rgb'] is True and rb['rgbtoint'] is True
          and rb['slot_color'] == (0.0, 0.0, 0.0)
          and _close(rb['factor'], 1.0), str(rb))
    check('the HAR slot carries the same flags on its vblend',
          vb is not None and vb['tex_rgb'] is True
          and vb['rgbtoint'] is True and vb['blend'] == 'MUL', str(vb))

    # a stencil slot warns by name until the chaining imports
    mat2 = dict(mat, name='St')
    mat2['slots'] = [dict(mat['slots'][0], texflag=17665 | 2)]
    sp2 = M2.material_spec(mat2, 279)
    check('a Stencil slot warns that the gating is not imported',
          any('Stencil' in w for w in sp2['warnings']),
          str(sp2['warnings']))

    # ---- 7) build_spec wires the colour influence completely
    socks, specs = _shader_socket_table()
    from .. import templates as TPL
    fm = _fake_material(socks, specs)
    TPL.build_spec(fm, sp)
    tree = fm.node_tree
    nodes = {}
    for n in tree.nodes:
        nodes.setdefault(n.bl_idname, []).append(n)
    check('the graph carries one colour influence and one value '
          'influence',
          len(nodes.get('HALCYON_BIRGBBlendNode', [])) == 1
          and len(nodes.get('HALCYON_BIInfluenceNode', [])) == 1,
          str(sorted(nodes)))
    cn = nodes['HALCYON_BIRGBBlendNode'][0]
    check('the flags landed on the node',
          cn.tex_rgb is True and cn.rgbtoint is True
          and cn.blend == 'MUL')

    def linked2(src_id, out_name, in_name, dst_id):
        return any(getattr(a.node, 'bl_idname', '') == src_id
                   and a.name == out_name and b.name == in_name
                   and getattr(b.node, 'bl_idname', '') == dst_id
                   for a, b in tree.links)
    check('texture Color, Fac AND Alpha all reach the colour influence',
          linked2('HALCYON_BITextureNode', 'Color', 'Color',
                  'HALCYON_BIRGBBlendNode')
          and linked2('HALCYON_BITextureNode', 'Fac', 'Intensity',
                      'HALCYON_BIRGBBlendNode')
          and linked2('HALCYON_BITextureNode', 'Alpha', 'Alpha',
                      'HALCYON_BIRGBBlendNode'))
    check('texture Color and Alpha also feed the VALUE influence for '
          'its luminance collapse',
          linked2('HALCYON_BITextureNode', 'Color', 'Color',
                  'HALCYON_BIInfluenceNode')
          and linked2('HALCYON_BITextureNode', 'Alpha', 'Alpha',
                      'HALCYON_BIInfluenceNode'))

    # ---- 8) the per-pixel grant matches IDENTIFIERS: the BI node's
    # 'Hardness' display name over the 'Glossiness' identifier put five
    # sun-lit minutes on the field's CPU
    from ..gpu.material import per_pixel_fields
    g = {'output': 'out', 'nodes': {
        'bi': {'id': 'bi', 'bl_idname': 'HALCYON_BIMaterialNode',
               'props': {},
               'inputs': [
                   {'name': 'Hardness', 'identifier': 'Glossiness',
                    'type': 'VALUE', 'default': 50.0, 'link': ['x', 0]},
                   {'name': 'Spec', 'identifier': 'Specular Level',
                    'type': 'VALUE', 'default': 0.5, 'link': ['x', 0]},
                   {'name': 'Ref', 'identifier': 'Diffuse Level',
                    'type': 'VALUE', 'default': 0.8, 'link': None}],
               'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [{'name': 'Surface', 'type': 'SHADER',
                            'default': None, 'link': ['bi', 0]}],
                'outputs': []}}}
    pf = per_pixel_fields(g)
    check("a linked 'Hardness' socket grants per-pixel glossiness "
          'through its IDENTIFIER (the five-minute-frame bug)',
          'glossiness' in pf and pf['glossiness'][0] == 'Hardness'
          and 'specular_level' in pf and 'diffuse_level' not in pf,
          str(pf))

    # ---- 9) the image ALPHA LAW (verbatim imagewrap, R158): talpha
    # only under Use Alpha and not Calculate; Calculate -> max(r,g,b);
    # NEITHER -> the file's alpha channel is IGNORED (ta = 1.0);
    # Tex.flag&4 inverts. Wiring the file alpha unconditionally made
    # the Mask's streaks-on-transparency env map contribute nothing.
    def imgslot(imaflag=0, texflag=0, **kw):
        s = {'texco': 16, 'mapto': B.MAP_COL, 'maptoneg': 0,
             'blendtype': 0, 'colfac': 1.0, 'def_var': 1.0, 'uvname': '',
             'r': 1.0, 'g': 0.0, 'b': 1.0,
             'ofs': (0.0, 0.0, 0.0), 'size': (1.0, 1.0, 1.0),
             'tex': {'kind': 'IMAGE', 'name': 'T', 'flag': texflag,
                     'imaflag': imaflag,
                     'image_path': '//t.png', 'image_name': 't'}}
        s.update(kw)
        return s

    def spec_of(sl):
        return M2.material_spec({'name': 'A', 'mode': 0, 'r': 0.5,
                                 'g': 0.5, 'b': 0.5, 'ref': 0.8,
                                 'spec': 0.2, 'har': 50, 'alpha': 1.0,
                                 'slots': [sl]}, 279)

    def rb_of(sl):
        return spec_of(sl)['textures'][0]['rgbblend']

    check('Use Alpha (imaflag&2) grants the alpha feed',
          rb_of(imgslot(imaflag=7))['img_alpha'] is True)
    check("without Use Alpha the file's alpha channel is IGNORED "
          '(imagewrap ta = 1.0)',
          rb_of(imgslot(imaflag=5))['img_alpha'] is False)
    rbc = rb_of(imgslot(imaflag=2 | 32))
    check('Calculate wins over Use Alpha: ta = max(r,g,b), never the '
          'file channel',
          rbc['img_alpha'] is False and rbc['calc_alpha'] is True)
    check('Tex.flag&4 inverts the alpha (neg_alpha)',
          rb_of(imgslot(imaflag=2, texflag=4))['neg_alpha'] is True
          and rb_of(imgslot(imaflag=2))['neg_alpha'] is False)
    check('procedural colour always carries its band alpha (the law '
          'gates IMAGES only)',
          rb['img_alpha'] is True and rb['calc_alpha'] is False)
    fm3 = _fake_material(socks, specs)
    TPL.build_spec(fm3, spec_of(imgslot(imaflag=0)))
    check('an image WITHOUT Use Alpha never wires its Alpha output '
          'into the blend',
          not any(b.name == 'Alpha'
                  and getattr(b.node, 'bl_idname', '')
                  == 'HALCYON_BIRGBBlendNode'
                  for a, b in fm3.node_tree.links))

    # ---- 10) the matcap WINDOW: a REFL slot's ofs/size fold rides
    # MatcapUV -> Mapping -> image Vector (the Mask's env map lived in
    # a 0.7x0.8 window at +1.15 the import never applied), and the
    # fake pattern node carries the spec defaults so headless
    # serialization matches production (Scale=0 flattened every
    # headless matcap to one dead texel)
    msl = imgslot(imaflag=2, texco=2, ofs=(0.15, 1.15, 0.0),
                  size=(0.7, 0.8, 1.0))
    spm = spec_of(msl)
    em = spm['textures'][0]
    check('a REFL slot keeps its coords AND its mapping window '
          '(verbatim fold: location = ofs + (1-size)/2)',
          em.get('coords') == 'MATCAP_REFLECT'
          and em.get('mapping') is not None
          and _close(em['mapping']['scale'][0], 0.7)
          and _close(em['mapping']['location'][0],
                     0.15 + 0.5 * (1.0 - 0.7))
          and _close(em['mapping']['location'][1],
                     1.15 + 0.5 * (1.0 - 0.8)), str(em.get('mapping')))
    fm4 = _fake_material(socks, specs)
    TPL.build_spec(fm4, spm)
    t4 = fm4.node_tree
    mc = [n for n in t4.nodes if n.bl_idname == 'HALCYON_MatcapUVNode']
    mpn = [n for n in t4.nodes if n.bl_idname == 'ShaderNodeMapping']
    check('build_spec routes MatcapUV -> Mapping -> image Vector',
          len(mc) == 1 and len(mpn) == 1
          and any(a.node is mc[0] and b.node is mpn[0]
                  and b.name == 'Vector' for a, b in t4.links)
          and any(a.node is mpn[0]
                  and getattr(b.node, 'bl_idname', '')
                  == 'ShaderNodeTexImage' and b.name == 'Vector'
                  for a, b in t4.links),
          f'matcap {len(mc)} mapping {len(mpn)}')
    loc_in = mpn[0].inputs.get('Location') if mpn else None
    check('the Mapping window carries the fold values on its sockets',
          loc_in is not None
          and _close(loc_in.default_value[0], 0.15 + 0.5 * (1.0 - 0.7))
          and _close(loc_in.default_value[1], 1.15 + 0.5 * (1.0 - 0.8)))
    check('the fake pattern node carries the SPEC defaults: MatcapUV '
          'Scale is 1.0, not the flattening 0 (the R158 headless '
          'matcap bug class)',
          bool(mc) and _close(mc[0].inputs.get('Scale').default_value,
                              1.0))
    check('...and with Use Alpha set, the matcap image DOES feed its '
          'alpha through',
          any(b.name == 'Alpha'
              and getattr(b.node, 'bl_idname', '')
              == 'HALCYON_BIRGBBlendNode' for a, b in t4.links))


def test_lamp_world_mapping():
    d9, _ = FX.build_249()
    s9 = B.read_legacy_scene(d9)
    la9 = [o for o in s9['objects'] if o['kind'] == 'LAMP'][0]['data']
    lm = M.lamp_map(la9, 249)
    check('pre-2.70 spot size converts degrees -> radians',
          _close(lm['spot_size'], 0.7853982, 1e-4))
    d7, _ = FX.build_279()
    s7 = B.read_legacy_scene(d7)
    la7 = [o for o in s7['objects'] if o['kind'] == 'LAMP'][0]['data']
    check('2.70+ spot size passes through untouched',
          _close(M.lamp_map(la7, 279)['spot_size'], 0.7853982, 1e-4))
    check('hemi imports as ITSELF now -- the dome light is real',
          M.lamp_map({'kind': 'HEMI', 'name': 'H'}, 279)['type'] == 'HEMI'
          and not M.lamp_map({'kind': 'HEMI', 'name': 'H'},
                             279)['warnings'])
    wm = M.world_map(s7['world'])
    check('world -> gradient with mist',
          wm['blend'] and _close(wm['zenith'][2], 0.8)
          and _close(wm['mist']['start'], 5.0)
          and _close(wm['mist']['depth'], 40.0))

    # ---- the 2.79 per-lamp shadow rule, verbatim shade_one_light +
    # convertblender precedence: a buffer exists only for a SPOT with
    # the buffer bit and WINS there; otherwise the Ray bit traces;
    # otherwise the lamp casts NO shadow at all (the field's lamps
    # with neither bit were importing as MAP-shadowed -- darker than
    # 2.79 ever drew them)
    def sh(kind, mode):
        return M.lamp_map({'kind': kind, 'name': 'L', 'mode': mode},
                          279)['shadow']
    check('SUN with both bits: the buffer is inert off-spot, Ray wins',
          sh('SUN', M.LA_SHAD_BUF | M.LA_SHAD_RAY) == 'RAY')
    check('POINT with only the buffer bit casts NO shadow (2.79 built '
          'buffers for spots alone)',
          sh('POINT', M.LA_SHAD_BUF) == 'NONE')
    check('SPOT with both bits: the buffer wins (lar->shb is checked '
          'first)', sh('SPOT', M.LA_SHAD_BUF | M.LA_SHAD_RAY) == 'MAP')
    check('SPOT with only the Ray bit traces',
          sh('SPOT', M.LA_SHAD_RAY) == 'RAY')
    check('no shadow bits -> no shadow at all',
          sh('POINT', 0) == 'NONE' and sh('SUN', 0) == 'NONE')

    # ---- the two import routes produce the SAME lamp: _build_light
    # (parser) and _enrich_lamp (appender) both ride _apply_lamp_bi.
    # The appender used to keep Blender's own watt conversion and set
    # an unbounded INVERSE falloff -- several times darker than BI at
    # working distances ('the lights are too dark' whenever a scene
    # arrived through the appender)
    import types as _t

    from .. import legacy_import as LI2
    parsed = {'kind': 'POINT', 'name': 'P', 'r': 1.0, 'g': 0.5,
              'b': 0.25, 'energy': 2.0, 'dist': 25.0,
              'falloff_type': 2, 'att1': 0.0, 'att2': 1.0,
              'mode': M.LA_SHAD_RAY | M.LA_SPHERE,
              'spotsize': 0.785, 'spotblend': 0.15}

    def fake_light():
        return _t.SimpleNamespace(
            color=None, energy=0.0, spot_size=0.0, spot_blend=0.0,
            halcyon=_t.SimpleNamespace(
                hemi=False, shadow='MAP', decay='DEFAULT',
                decay_start=0.0, decay_end=25.0, decay_ld1=0.0,
                decay_ld2=0.0, bi_sphere=False, negative=False,
                specular_only=False, diffuse_only=False))
    a = fake_light()
    LI2._apply_lamp_bi(a, M.lamp_map(parsed, 279))
    b = fake_light()
    b.energy = 123.0            # Blender's own append conversion...
    b.color = (9, 9, 9)         # ...must be OVERRIDDEN by the file's
    warn2 = []
    LI2._enrich_lamp(b, parsed, 279, warn2)
    same = (abs(a.energy - b.energy) < 1e-6
            and tuple(a.color) == tuple(b.color)
            and a.halcyon.decay == b.halcyon.decay == 'BI_SQUARE'
            and a.halcyon.shadow == b.halcyon.shadow == 'RAY'
            and a.halcyon.bi_sphere and b.halcyon.bi_sphere
            and abs(a.energy - 2.0 * 4 * 3.14159265 ** 2) < 1e-3)
    check('the appender route now produces EXACTLY the parser route\'s '
          'lamp: BI energy, BI falloff, BI shadow rule, sphere bit',
          same, f'{a.energy} vs {b.energy}, {a.halcyon.decay} vs '
                f'{b.halcyon.decay}, {a.halcyon.shadow} vs '
                f'{b.halcyon.shadow}')


# ---------------------------------------------------- node-tree construction


class _FSock:
    def __init__(self, node, name, kind):
        self.node, self.name = node, name
        if kind == 'RGBA':
            self.default_value = [0.0, 0.0, 0.0, 1.0]
        elif kind == 'VECTOR':
            self.default_value = [0.0, 0.0, 0.0]
        else:
            self.default_value = 0.0


class _FSocks(list):
    def get(self, name):
        return next((s for s in self if s.name == name), None)

    def __getitem__(self, key):                 # bpy allows name indexing
        if isinstance(key, str):
            s = self.get(key)
            if s is None:
                raise KeyError(key)
            return s
        return list.__getitem__(self, key)

    def new(self, kind, name):
        s = _FSock(None, name, {'NodeSocketColor': 'RGBA',
                                'NodeSocketVector': 'VECTOR'}.get(kind,
                                                                  'VALUE'))
        self.append(s)
        return s


class _FStops(list):
    class _Stop:
        def __init__(self):
            self.position = 0.0
            self.color = (0.0, 0.0, 0.0, 1.0)

    def add(self):
        s = self._Stop()
        self.append(s)
        return s

    def clear(self):
        del self[:]


_NODE_TABLE = {
    'ShaderNodeOutputMaterial': ([('Surface', 'SHADER')], []),
    'HALCYON_BITextureNode': ([('Vector', 'VECTOR')],
                              [('Color', 'RGBA'), ('Fac', 'VALUE'),
                               ('Alpha', 'VALUE')]),
    'HALCYON_BIInfluenceNode': ([('Base', 'VALUE'),
                                 ('Intensity', 'VALUE'),
                                 ('Factor', 'VALUE'), ('DVar', 'VALUE'),
                                 ('Color', 'RGBA'), ('Alpha', 'VALUE')],
                                [('Value', 'VALUE')]),
    'HALCYON_BIRGBBlendNode': ([('Base', 'RGBA'), ('Color', 'RGBA'),
                                ('Intensity', 'VALUE'),
                                ('Alpha', 'VALUE'), ('Factor', 'VALUE'),
                                ('Slot Color', 'RGBA')],
                               [('Color', 'RGBA')]),
    'ShaderNodeUVMap': ([], [('UV', 'VECTOR')]),
}

#: default prop values real nodes carry (see _FNode)
_NODE_ATTRS = {
    'ShaderNodeMixRGB': (('blend_type', 'MIX'), ('use_clamp', False)),
    'ShaderNodeMath': (('operation', 'ADD'), ('use_clamp', False)),
    'ShaderNodeUVMap': (('uv_map', ''),),
    'HALCYON_BIInfluenceNode': (('blend', 'MIX'), ('tex_rgb', False),
                                ('rgbtoint', False), ('negative', False),
                                ('alphamix', False), ('calc_alpha', False),
                                ('neg_alpha', False)),
    'HALCYON_BIRGBBlendNode': (('blend', 'MIX'), ('tex_rgb', False),
                               ('rgbtoint', False), ('negative', False),
                               ('alphamix', False), ('map_alpha', False),
                               ('calc_alpha', False),
                               ('neg_alpha', False)),
}

_NODE_TABLE_TAIL = {
    'ShaderNodeTexImage': ([('Vector', 'VECTOR')],
                           [('Color', 'RGBA'), ('Alpha', 'VALUE')]),
    # REAL Blender socket order -- Normal sits between Generated and UV.
    # A fake that omits it desynchronizes name-vs-index dispatch between
    # the devices, which is exactly the divergence the emitters guard
    # against (found sampling an image at hal_N.xy in a headless frame).
    'ShaderNodeTexCoord': ([], [('Generated', 'VECTOR'),
                                ('Normal', 'VECTOR'), ('UV', 'VECTOR'),
                                ('Object', 'VECTOR'), ('Camera', 'VECTOR'),
                                ('Window', 'VECTOR'),
                                ('Reflection', 'VECTOR')]),
    'ShaderNodeMapping': ([('Vector', 'VECTOR'), ('Location', 'VECTOR'),
                           ('Rotation', 'VECTOR'), ('Scale', 'VECTOR')],
                          [('Vector', 'VECTOR')]),
    'ShaderNodeMixRGB': ([('Fac', 'VALUE'), ('Color1', 'RGBA'),
                          ('Color2', 'RGBA')], [('Color', 'RGBA')]),
    'ShaderNodeMath': ([('Value', 'VALUE'), ('Value', 'VALUE')],
                       [('Value', 'VALUE')]),
    'ShaderNodeBump': ([('Strength', 'VALUE'), ('Distance', 'VALUE'),
                        ('Height', 'VALUE'), ('Normal', 'VECTOR')],
                       [('Normal', 'VECTOR')]),
    'ShaderNodeInvert': ([('Fac', 'VALUE'), ('Color', 'RGBA')],
                         [('Color', 'RGBA')]),
    'ShaderNodeTexMagic': ([('Vector', 'VECTOR'), ('Scale', 'VALUE'),
                            ('Distortion', 'VALUE')],
                           [('Color', 'RGBA'), ('Fac', 'VALUE')]),
}
_NODE_TABLE.update(_NODE_TABLE_TAIL)


class _FNode:
    def __init__(self, idname, shader_sockets=None, pattern_specs=None):
        self.bl_idname = idname
        self.location = (0, 0)
        self.inputs, self.outputs = _FSocks(), _FSocks()
        if idname == 'HALCYON_ShaderNode':
            for kind, name, default in shader_sockets:
                s = _FSock(self, name, {'NodeSocketColor': 'RGBA',
                                        'NodeSocketVector':
                                        'VECTOR'}.get(kind, 'VALUE'))
                if default is not None:
                    if isinstance(s.default_value, list) and \
                            hasattr(default, '__len__'):
                        s.default_value = list(default)
                    elif not isinstance(s.default_value, list):
                        s.default_value = default
                self.inputs.append(s)
            self.outputs.append(_FSock(self, 'Surface', 'SHADER'))
            self.model = 'PHONG'
            self.refresh_sockets = lambda: None
        elif idname == 'HALCYON_BIMaterialNode':
            # the BI node: BI display names over master identifiers,
            # from the real class's own socket table
            from ..nodes.shader_nodes import HALCYON_BIMaterialNode
            for kind, name, ident, default in \
                    HALCYON_BIMaterialNode.BI_SOCKETS:
                s = _FSock(self, name, {'NodeSocketColor': 'RGBA',
                                        'NodeSocketVector':
                                        'VECTOR'}.get(kind, 'VALUE'))
                s.identifier = ident
                if default is not None:
                    if isinstance(s.default_value, list) and \
                            hasattr(default, '__len__'):
                        s.default_value = list(default)
                    elif not isinstance(s.default_value, list):
                        s.default_value = default
                self.inputs.append(s)
            self.outputs.append(_FSock(self, 'Surface', 'SHADER'))
            self.dif_stops = _FStops()
            self.spec_stops = _FStops()
            self.refresh_sockets = lambda: None
        elif idname in _NODE_TABLE:
            ins, outs = _NODE_TABLE[idname]
            for name, kind in ins:
                self.inputs.append(_FSock(self, name, kind))
            for name, kind in outs:
                self.outputs.append(_FSock(self, name, kind))
            # REAL nodes carry these props; a fake without them makes
            # `hasattr` guards skip the assignment and every MULTIPLY
            # mix exports as default MIX (the probe's white face)
            for k, v in _NODE_ATTRS.get(idname, ()):
                setattr(self, k, v)
            if idname == 'HALCYON_BITextureNode':
                self.stops = _FStops()
                # build_spec only sets attributes the node has, so the
                # fake needs the real property names present
                from ..nodes.bitex_node import BI_NODE_PROPS
                for k in BI_NODE_PROPS:
                    setattr(self, k, None)
        elif idname.startswith('HALCYON_') and pattern_specs is not None:
            base = idname[len('HALCYON_'):-len('Node')]
            spec = next(s for s in pattern_specs if s[0] == base)
            for kind, name, default in spec[4]:
                s = _FSock(self, name,
                           {'NodeSocketColor': 'RGBA',
                            'NodeSocketVector': 'VECTOR'}
                           .get(kind, 'VALUE'))
                # REAL nodes carry the spec's default; a fake without
                # it serialized Scale=0 on the Matcap node and flattened
                # every headless matcap to one dead texel (the same
                # class of bug as the MixRGB blend_type fakes of R151)
                if default is not None:
                    if isinstance(s.default_value, list) and \
                            hasattr(default, '__len__'):
                        s.default_value = list(default)
                    elif not isinstance(s.default_value, list):
                        s.default_value = default
                self.inputs.append(s)
            for kind, name in spec[6]:
                self.outputs.append(_FSock(
                    self, name, {'NodeSocketColor': 'RGBA',
                                 'NodeSocketVector': 'VECTOR'}
                    .get(kind, 'VALUE')))
        else:
            ins, outs = _NODE_TABLE.get(idname, ([], []))
            for name, kind in ins:
                self.inputs.append(_FSock(self, name, kind))
            for name, kind in outs:
                self.outputs.append(_FSock(self, name, kind))
            if idname == 'HALCYON_BITextureNode':
                self.stops = _FStops()


def _fake_material(shader_sockets, pattern_specs):
    links = []
    nodes = []

    class _Nodes(list):
        def new(self, idname):
            n = _FNode(idname, shader_sockets, pattern_specs)
            self.append(n)
            return n

        def clear(self):
            del self[:]

    class _Links(list):
        def new(self, a, b):
            # real Blender keeps ONE link per input socket: a second
            # link REPLACES the first. The fake list used to keep
            # both, which let a re-linking bug pass the wiring tests
            # while the field showed dangling texture nodes
            self[:] = [(x, y) for x, y in self if y is not b]
            self.append((a, b))

    tree = types.SimpleNamespace(nodes=_Nodes(nodes), links=_Links(links))
    return types.SimpleNamespace(
        name='Fake', use_nodes=True, node_tree=tree,
        halcyon=types.SimpleNamespace(use_override=True))


def _shader_socket_table():
    """The master shader's socket list, via the fake bpy."""
    from . import fakebpy
    bpy = fakebpy.install()
    if not hasattr(bpy.types, 'UIList'):
        bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    from ..nodes.pattern_nodes import SPECS
    from ..nodes.shader_nodes import HALCYON_ShaderNode
    return HALCYON_ShaderNode.SOCKETS, SPECS


def test_build_spec_wiring():
    socks, specs = _shader_socket_table()
    from .. import templates

    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data)
    spec = M.material_spec(sc['materials'][truth['materials']['skin']], 279)
    mat = _fake_material(socks, specs)
    shader = templates.build_spec(mat, spec)
    tree = mat.node_tree
    made = {}
    for n in tree.nodes:
        made.setdefault(n.bl_idname, []).append(n)

    check('build_spec makes the BI MATERIAL node with both menus set',
          shader.bl_idname == 'HALCYON_BIMaterialNode'
          and shader.diff_shader == 'OREN_NAYAR'
          and shader.spec_shader == 'BLINN')
    check('an image node, a BI texture, a colour influence and a coord '
          'node appear',
          all(k in made for k in
              ('ShaderNodeTexImage', 'HALCYON_BITextureNode',
               'HALCYON_BIRGBBlendNode', 'ShaderNodeTexCoord')),
          str(sorted(made)))
    check('no Mapping node: the BI node owns the classic arithmetic',
          'ShaderNodeMapping' not in made)
    check('exactly one shared texture-coordinate node',
          len(made.get('ShaderNodeTexCoord', [])) == 1)

    links = [(a.node if hasattr(a, 'node') else None, a.name, b.name,
              b.node if hasattr(b, 'node') else None)
             for a, b in tree.links]

    def linked(src_id, out_name, in_name, dst_id):
        return any(a is not None and a.bl_idname == src_id
                   and an == out_name and bn == in_name
                   and d is not None and d.bl_idname == dst_id
                   for a, an, bn, d in links)

    check('image colour AND alpha feed the colour influence (the alpha '
          'is the per-pixel factor for an RGB texture)',
          linked('ShaderNodeTexImage', 'Color', 'Color',
                 'HALCYON_BIRGBBlendNode')
          and linked('ShaderNodeTexImage', 'Alpha', 'Alpha',
                     'HALCYON_BIRGBBlendNode'))
    check('colour influence -> the Color socket (Diffuse Color '
          'identifier) on the BI node',
          linked('HALCYON_BIRGBBlendNode', 'Color', 'Color',
                 'HALCYON_BIMaterialNode'))
    check('BI texture Fac reaches Bump Height through its strength '
          'multiply',
          linked('HALCYON_BITextureNode', 'Fac', 'Value',
                 'ShaderNodeMath')
          and linked('ShaderNodeMath', 'Value', 'Bump Height',
                     'HALCYON_BIMaterialNode'))
    check('coords feed the BI texture directly (offset/size on-node)',
          linked('ShaderNodeTexCoord', 'Generated', 'Vector',
                 'HALCYON_BITextureNode'))
    check('the image slot names its UV layer and rides a UV Map node',
          linked('ShaderNodeUVMap', 'UV', 'Vector',
                 'ShaderNodeTexImage'))
    check('shader surface -> material output',
          linked('HALCYON_BIMaterialNode', 'Surface', 'Surface',
                 'ShaderNodeOutputMaterial'))

    cinf = made['HALCYON_BIRGBBlendNode'][0]
    check('the colour influence carries the slider and the base colour, '
          'and knows its texture yields RGB',
          _close(cinf.inputs.get('Factor').default_value, 0.85)
          and _close(cinf.inputs.get('Base').default_value[0], 0.8)
          and cinf.tex_rgb is True)
    bi = made['HALCYON_BITextureNode'][0]
    check('the BI node carries type, size, depth, offset and band',
          bi.tex_type == 'CLOUDS' and _close(bi.noise_size, 0.35)
          and bi.noise_depth == 3 and _close(bi.tex_size[0], 2.0)
          and _close(bi.tex_offset[1], 0.25))
    check('the colorband stops land in the node collection',
          len(bi.stops) == 3 and _close(bi.stops[1].position, 0.5)
          and _close(bi.stops[1].color[0], 0.9))
    check('bump strength socket rides at 1 (slot factor is a multiply)',
          _close(shader.inputs.get('Bump Strength').default_value, 1.0))
    check("the Hardness socket (Glossiness identifier) carries the "
          "file's hardness",
          _close(shader.inputs.get('Hardness').default_value, 50.0))
    check('the material is marked as node-driven (no override)',
          mat.halcyon.use_override is False)

    # a value-scaled channel builds a Math multiply
    spec2 = {'model': 'PHONG', 'inputs': {},
             'textures': [{'node': 'HALCYON_NoiseNode', 'props': {},
                           'inputs': {}, 'output': 'Fac',
                           'target': 'Specular Level', 'scale_fac': 0.4}]}
    mat2 = _fake_material(socks, specs)
    templates.build_spec(mat2, spec2)
    mades2 = [n for n in mat2.node_tree.nodes
              if n.bl_idname == 'ShaderNodeMath']
    check('scale_fac routes through a Math multiply',
          len(mades2) == 1
          and _close(mades2[0].inputs[1].default_value, 0.4))

    # an inverted channel builds an Invert
    spec3 = {'model': 'PHONG', 'inputs': {},
             'textures': [{'node': 'HALCYON_NoiseNode', 'props': {},
                           'inputs': {}, 'output': 'Fac',
                           'target': 'Opacity', 'invert': True}]}
    mat3 = _fake_material(socks, specs)
    templates.build_spec(mat3, spec3)
    check('invert routes through an Invert node',
          any(n.bl_idname == 'ShaderNodeInvert'
              for n in mat3.node_tree.nodes))


def test_multi_slot_color_chain():
    """BI stacks texture slots: each colour slot blends ONTO the result
    of the one before it. The first build wired every MixRGB's Color1 to
    the STATIC base colour instead, so with two colour slots the second
    mix's link to Diffuse Color REPLACED the first slot's (Blender keeps
    one link per input) and every earlier slot's chain dangled unlinked
    -- the field's 'mixed textures aren't plugged in', on the real
    file's suit: the base skin at full MIX, then the phantom grunge layer
    MULTIPLYed over it, which rendered as base-colour murk with no suit
    texture at all."""
    socks, specs = _shader_socket_table()
    from .. import templates

    def _img(**extra):
        e = {'node': 'ShaderNodeTexImage', 'props': {}, 'inputs': {},
             'output': 'Color', 'target': 'Diffuse Color',
             'style': 'color'}
        e.update(extra)
        return e

    base = (0.12, 0.12, 0.04, 1.0)
    spec = {'model': 'LAMBERT', 'inputs': {'Diffuse Color': base},
            'textures': [
                _img(),                                   # full-MIX slot
                _img(mix={'fac': 1.0, 'blend': 'MULTIPLY',
                          'base': base}),                 # grunge multiply
                _img(mix={'fac': 0.5, 'blend': 'MIX',
                          'base': base}),                 # decal at half
            ]}
    mat = _fake_material(socks, specs)
    shader = templates.build_spec(mat, spec)
    tree = mat.node_tree
    images = [n for n in tree.nodes
              if n.bl_idname == 'ShaderNodeTexImage']
    mixes = [n for n in tree.nodes if n.bl_idname == 'ShaderNodeMixRGB']
    check('three image nodes and two mixes appear',
          len(images) == 3 and len(mixes) == 2)

    # real semantics live in the fake now: one link per input socket
    def src_of(node, input_name):
        for a, b in tree.links:
            if b.node is node and b.name == input_name:
                return a
        return None

    m1, m2 = mixes
    c1 = src_of(m1, 'Color1')
    check("the second slot's mix blends ONTO the first slot's texture",
          c1 is not None and c1.node is images[0],
          'Color1 came from ' + (c1.node.bl_idname if c1 else 'nothing'))
    c2 = src_of(m2, 'Color1')
    check("the third slot's mix blends onto the second mix's result",
          c2 is not None and c2.node is m1)
    dsrc = src_of(shader, 'Diffuse Color')
    check('the LAST mix is what reaches the shader',
          dsrc is not None and dsrc.node is m2)

    # every texture node must reach the shader (the field symptom was
    # exactly this walk failing)
    incoming = {}
    for a, b in tree.links:
        incoming.setdefault(b.node, []).append(a.node)
    reached, stack = set(), [shader]
    while stack:
        n = stack.pop()
        if id(n) in reached:
            continue
        reached.add(id(n))
        stack.extend(incoming.get(n, []))
    check('every image node is plugged in (reaches the shader)',
          all(id(n) in reached for n in images),
          f'{sum(id(n) not in reached for n in images)} dangling')

    # a full-influence MIX colour slot AFTER another slot REPLACES the
    # running chain (BI: result = mix(result, tex, 1.0) = tex) -- it must
    # not detour through the value channels' ADD
    spec_r = {'model': 'LAMBERT', 'inputs': {'Diffuse Color': base},
              'textures': [
                  _img(mix={'fac': 0.5, 'blend': 'MIX', 'base': base}),
                  _img(),
              ]}
    mat_r = _fake_material(socks, specs)
    shader_r = templates.build_spec(mat_r, spec_r)
    tree_r = mat_r.node_tree
    imgs_r = [n for n in tree_r.nodes
              if n.bl_idname == 'ShaderNodeTexImage']
    d_r = None
    for a, b in tree_r.links:
        if b.node is shader_r and b.name == 'Diffuse Color':
            d_r = a
    check('a full-MIX colour slot replaces the chain, no ADD detour',
          d_r is not None and d_r.node is imgs_r[1]
          and not any(n.bl_idname == 'ShaderNodeMath'
                      for n in tree_r.nodes))

    # value channels keep their summing behaviour (several bump slots)
    spec_v = {'model': 'PHONG', 'inputs': {},
              'textures': [
                  {'node': 'HALCYON_NoiseNode', 'props': {}, 'inputs': {},
                   'output': 'Fac', 'target': 'Specular Level',
                   'style': 'value'},
                  {'node': 'HALCYON_NoiseNode', 'props': {}, 'inputs': {},
                   'output': 'Fac', 'target': 'Specular Level',
                   'style': 'value'},
              ]}
    mat_v = _fake_material(socks, specs)
    templates.build_spec(mat_v, spec_v)
    check('two full-strength value slots still sum through an ADD',
          any(n.bl_idname == 'ShaderNodeMath'
              for n in mat_v.node_tree.nodes))


def test_operator_registration():
    from . import fakebpy
    bpy = fakebpy.install()
    if not hasattr(bpy.types, 'UIList'):
        bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    from .. import legacy_import
    legacy_import.register()
    try:
        names = [c.bl_idname for c in legacy_import.CLASSES]
        check('the operator registers under fakebpy',
              'halcyon.append_legacy' in names)
        op = legacy_import.HALCYON_OT_append_legacy
        check('file browser filter targets .blend files',
              op.__annotations__['filter_glob'].kw.get('default')
              == '*.blend')
        for prop in ('only_selected', 'import_lights', 'import_cameras',
                     'import_world', 'scale'):
            check(f'option {prop} has a real tooltip',
                  len(op.__annotations__[prop].kw.get('description', ''))
                  >= 20)
    finally:
        legacy_import.unregister()



# --------------------------------------------- the BI texture engine itself


def test_bitex_engine():
    """core/bitex.py structural truths: the transcribed algorithms hold
    the properties the originals hold."""
    from ..core import bitex as BX
    rng = np.random.default_rng(7)
    P = rng.uniform(-4, 4, (3000, 3)).astype(np.float32)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    n0 = BX.org_blender_noise(x, y, z)
    check('org blender noise lives in 0..1 around 0.5',
          float(n0.min()) >= 0.0 and float(n0.max()) <= 1.0
          and 0.45 < float(n0.mean()) < 0.55)
    check('cell noise is constant within a cell',
          float(BX.cellnoise_u(np.array([3.2]), np.array([5.7]),
                               np.array([-2.1]))[0])
          == float(BX.cellnoise_u(np.array([3.9]), np.array([5.1]),
                                  np.array([-2.9]))[0]))
    da, pa = BX.voronoi(x, y, z)
    check('voronoi distances ascend and match their points',
          bool((np.diff(da, axis=1) >= -1e-6).all())
          and np.allclose(np.linalg.norm(P - pa[:, :3], axis=1), da[:, 0],
                          atol=1e-4))
    check('turbulence at zero octaves equals one noise',
          np.allclose(BX.gturbulence(0.25, x, y, z, 0, 0, 0),
                      BX.gnoise(0.25, x, y, z, 0, 0), atol=1e-6))
    for tt in (1, 2, 3, 4, 5, 6, 7, 11, 12, 13):
        tin, rgb = BX.evaluate({'type': tt}, P)
        check(f'texture type {tt} evaluates finite in range',
              bool(np.isfinite(tin).all())
              and (rgb is None or bool(np.isfinite(rgb).all())))
    stops = [(0.0, 1.0, 0.0, 0.0, 1.0), (1.0, 0.0, 0.0, 1.0, 1.0)]
    cb = BX.colorband_eval(stops, np.array([0.0, 0.5, 1.0], np.float32))
    check('colorband endpoints and midpoint exact',
          np.allclose(cb[:, 0], [1.0, 0.5, 0.0])
          and np.allclose(cb[:, 2], [0.0, 0.5, 1.0]))
    tin, rgba = BX.evaluate(
        {'type': 1, 'flag': 1, 'coba': stops, 'coba_ipotype': 0}, P)
    check('the colorband turns an intensity texture into colour',
          rgba is not None and rgba.shape[1] == 4)
    t1, _ = BX.evaluate({'type': 1}, P)
    t2, _ = BX.evaluate({'type': 1, 'bright': 1.5}, P)
    check('brightness lifts the intensity',
          float((t2 - t1).mean()) > 0.2)
    # chunked evaluation: identical numbers, bounded memory
    big = rng.uniform(-4, 4, (BX._CHUNK + 4111, 3)).astype(np.float32)
    whole, _ = BX.evaluate({'type': 12}, big)
    head, _ = BX._evaluate_chunk({'type': 12}, big[:2000])
    check('chunked evaluate matches the unchunked numbers',
          np.allclose(whole[:2000], head, atol=1e-6)
          and whole.shape[0] == big.shape[0])
    _tin, rgba = BX.evaluate({'type': 4}, big)     # magic: rgb across chunks
    check('chunked colour textures reassemble their rgba',
          rgba is not None and rgba.shape == (big.shape[0], 4))


def test_bitex_gpu_parity():
    """The generated GLSL twin against the CPU, through the engine's own
    shader compiler -- the same tables, the same numbers."""
    from ..core import bitex as bx
    from ..gpu.procedural import OKRAMP_GLSL, PATTERN_GLSL
    from ..shaders.compiler import try_compile

    from ..core.bitex_tables import table_pixels
    from ..core.texture import Texture

    src = OKRAMP_GLSL + PATTERN_GLSL['bitex'] + """
uniform vec3 P;
uniform float mode;
out vec4 Color;
void main() {
    int m = int(mode);
    float f = 0.0;
    if (m == 0) { f = bi_onoise(P); }
    else if (m == 1) { f = bi_nperlin(P); }
    else if (m == 2) { f = bi_cell_u(P); }
    else if (m == 3) { f = bi_gturb(0.25, P, 2, 0, 0); }
    else if (m == 4) { f = bi_gturb(0.31, P, 4, 1, 2); }
    else if (m == 5) { f = bi_tex_wood(P, 3, 1, 0.25, 8.0, 1, 0); }
    else if (m == 6) { f = bi_tex_marble(P, 2, 2, 0.4, 6.0, 3, 0, 0); }
    else if (m == 7) { f = bi_tex_blend(P, 5, 0); }
    else if (m == 8) { f = bi_tex_stucci(P, 2, 0.25, 40.0, 0, 0); }
    else if (m == 9) { f = bi_tex_musgrave(P / 0.25, 1, 0.8, 2.2, 5.5, 0.9, 2.0, 1.1, 0); }
    else if (m == 10) { vec4 r = bi_tex_voronoi(P / 0.25, 0.7, -0.3, 0.2, 0.1, 3.5, 6, 1.2, 0); f = r.a; }
    else if (m == 11) { f = bi_tex_distnoise(P / 0.25, 2.3, 0, 14); }
    else if (m == 12) { f = bi_gnoise(0.2, P, 0, 8); }
    Color = vec4(f, 0.0, 0.0, 1.0);
}
"""
    prog, err = try_compile(src, 'GLSL')
    check('the BI GLSL library compiles (tables and all)',
          prog is not None, str(err))
    if prog is None:
        return
    tab = Texture(table_pixels(), colorspace='Non-Color',
                  filt='NEAREST', wrap='EXTEND')
    rng = np.random.default_rng(11)
    P = rng.uniform(-3, 3, (256, 3)).astype(np.float32)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    cases = {
        0: ('org blender', bx.org_blender_noise(x, y, z)),
        1: ('improved perlin', bx.new_perlin(x, y, z)),
        2: ('cell noise', bx.cellnoise_u(x, y, z)),
        3: ('turbulence soft', bx.gturbulence(0.25, x, y, z, 2, 0, 0)),
        4: ('turbulence hard', bx.gturbulence(0.31, x, y, z, 4, 1, 2)),
        5: ('wood ringnoise saw', bx._wood(
            {'stype': 3, 'noisebasis2': 1, 'noisesize': 0.25,
             'turbul': 8.0, 'noisetype': 1, 'noisebasis': 0}, x, y, z)[0]),
        6: ('marble sharper tri', bx._marble(
            {'stype': 2, 'noisebasis2': 2, 'noisesize': 0.4, 'turbul': 6.0,
             'noisedepth': 3, 'noisetype': 0, 'noisebasis': 0},
            x, y, z)[0]),
        7: ('blend halo', bx._blend({'stype': 5, 'flag': 0}, x, y, z)[0]),
        8: ('stucci wallout', bx._stucci(
            {'stype': 2, 'noisesize': 0.25, 'turbul': 40.0, 'noisetype': 0,
             'noisebasis': 0}, x, y, z)[0]),
        9: ('musgrave ridged', bx._musgrave(
            {'stype': 1, 'mg_H': 0.8, 'mg_lacunarity': 2.2,
             'mg_octaves': 5.5, 'mg_offset': 0.9, 'mg_gain': 2.0,
             'ns_outscale': 1.1, 'noisebasis': 0},
            x / 0.25, y / 0.25, z / 0.25)[0]),
        10: ('voronoi minkowski', bx._voronoi_tex(
            {'vn_w1': 0.7, 'vn_w2': -0.3, 'vn_w3': 0.2, 'vn_w4': 0.1,
             'vn_mexp': 3.5, 'vn_distm': 6, 'ns_outscale': 1.2,
             'vn_coltype': 0}, x / 0.25, y / 0.25, z / 0.25)[0]),
        11: ('distorted noise', bx._distnoise(
            {'dist_amount': 2.3, 'noisebasis': 0, 'noisebasis2': 14},
            x / 0.25, y / 0.25, z / 0.25)[0]),
        12: ('crackle', bx.gnoise(0.2, x, y, z, 0, 8)),
    }
    for m, (label, cpu) in cases.items():
        out, _d = prog.run({'P': P,
                            'mode': np.full(len(P), m, np.float32),
                            'hal_bitex_tab': tab}, {}, len(P))
        gpu = np.asarray(out['Color'])[:, 0]
        d = float(np.abs(gpu - np.asarray(cpu)).max())
        # 1e-4: float32 cancellation in marble's huge wave argument and
        # voronoi near-ties; everything else sits at ~1e-6
        check(f'GPU twin: {label}', d < 1e-4, f'max diff {d:.2e}')


def _bi_ctx(n=48):
    """A ShadeContext with a varied generated field, for emit parity."""
    from ..core.nodeeval import ShadeContext
    from ..core.settings import RenderSettings
    c = ShadeContext(n)
    c.settings = RenderSettings()
    g = np.stack([np.linspace(-1.4, 2.1, n),
                  np.linspace(1.8, -0.6, n) ** 1,
                  np.linspace(-0.9, 1.3, n)], 1).astype(np.float32)
    c.generated = g
    c.uv = np.abs(g[:, :2]) * 0.4
    c.P = g * 2.0
    c.N = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    c.I = np.tile(np.array([[0.1, 0.2, -0.97]], np.float32), (n, 1))
    return c


def test_frame_emit_semantics():
    """Three emit-level rules a real 2.79 file crashed on.

    1. An IMAGELESS Image Texture node emits the CPU's own constants
       (opaque black) and registers NO sampler -- it used to register a
       None-keyed sampler, and ONE such node refused the whole GPU frame
       plan, dropping every material to a full-resolution CPU fallback
       (the field's rendered-view crash).
    2. Texture Coordinate outputs dispatch by NAME, the CPU evaluator's
       own rule -- positional dispatch shaded a different picture the
       moment names and positions disagreed (an image sampled at the
       shading normal).
    3. The colorband unroll declares arrays as `vec4 name[N]` -- the
       `vec4[N] name` form is real GLSL but the form shader front-ends
       (Halcyon's own included) are least likely to parse, and an
       uncompilable pass is a whole frame lost.
    """
    from ..core.nodeeval import GraphEvaluator
    from ..gpu.emit import Emitter
    from ..gpu.procedural import OKRAMP_GLSL, PATTERN_GLSL
    from ..shaders.compiler import try_compile
    from ..core.bitex_tables import table_pixels
    from ..core.texture import Texture

    n = 48

    # ---- 1: the imageless image node
    igraph = {'output': None, 'nodes': {'n': {
        'id': 'n', 'bl_idname': 'ShaderNodeTexImage',
        'props': {'image': None, 'interpolation': 'Closest',
                  'extension': 'REPEAT'},
        'inputs': [{'name': 'Vector', 'type': 'VECTOR',
                    'default': [0, 0, 0], 'link': None}],
        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                    {'name': 'Alpha', 'type': 'VALUE'}]}}}
    em = Emitter(igraph)
    var, vt = em.output('n', 0)
    check('imageless image node registers no sampler',
          not em.samplers)
    check('imageless image node emits opaque black',
          'vec4(0.0, 0.0, 0.0, 1.0)' in em.body())
    em2 = Emitter(igraph)
    avar, at = em2.output('n', 1)
    check('imageless image node emits alpha one',
          '= 1.0' in em2.body())
    ctx = _bi_ctx(n)
    want = GraphEvaluator(igraph, ctx).eval_output('n', 0)
    check('imageless constants are the CPU constants',
          bool((np.asarray(want)[:, :3] == 0.0).all())
          and bool((np.asarray(want)[:, 3] == 1.0).all()))

    # ---- 2: Texture Coordinate dispatches by output name
    tgraph = {'output': None, 'nodes': {'t': {
        'id': 't', 'bl_idname': 'ShaderNodeTexCoord', 'props': {},
        'inputs': [],
        # deliberately MISSING Normal, so name and position disagree
        'outputs': [{'name': 'Generated', 'type': 'VECTOR'},
                    {'name': 'UV', 'type': 'VECTOR'},
                    {'name': 'Object', 'type': 'VECTOR'}]}}}
    em3 = Emitter(tgraph)
    em3.output('t', 1)
    check('tex-coord output 1 named UV emits the uv, not the normal',
          'vec3(hal_uv, 0.0)' in em3.body()
          and 'hal_N' not in em3.body())
    em4 = Emitter(tgraph)
    em4.output('t', 2)
    check('tex-coord output 2 named Object emits world P',
          'hal_P' in em4.body())

    # ---- 3: the colorband unroll, form and numbers
    coba = [(0.0, 0.05, 0.02, 0.4, 0.0), (0.32, 0.9, 0.25, 0.1, 0.5),
            (0.61, 0.2, 0.8, 0.55, 0.8), (1.0, 1.0, 1.0, 1.0, 1.0)]
    bigraph = {'output': None, 'nodes': {'n': {
        'id': 'n', 'bl_idname': 'HALCYON_BITextureNode',
        'props': {'tex_type': 'CLOUDS', 'noise_basis': 'BLENDER_ORIGINAL',
                  'noise_size': 0.37, 'noise_depth': 2,
                  'use_colorband': True, 'coba_ipotype': 'EASE',
                  'coba': coba},
        'inputs': [{'name': 'Vector', 'type': 'VECTOR',
                    'default': [0, 0, 0], 'link': None}],
        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                    {'name': 'Fac', 'type': 'VALUE'},
                    {'name': 'Alpha', 'type': 'VALUE'}]}}}
    em5 = Emitter(bigraph)
    cvar, cvt = em5.output('n', 0)
    body = em5.body()
    import re as _re
    check('colorband arrays declare as `vec4 name[N]`',
          _re.search(r'vec4 _v\d+\[4\] = vec4\[4\]\(', body) is not None
          and _re.search(r'vec4\[4\] _v', body) is None)
    src = OKRAMP_GLSL + PATTERN_GLSL['bitex'] + """
uniform vec3 hal_generated;
out vec4 Color;
void main() {
%s
    Color = %s;
}
""" % (body, cvar)
    prog, err = try_compile(src, 'GLSL')
    check('the colorband pass compiles in the engine front-end',
          prog is not None, str(err))
    if prog is not None:
        tab = Texture(table_pixels(), colorspace='Non-Color',
                      filt='NEAREST', wrap='EXTEND')
        out, _d = prog.run({'hal_generated': ctx.generated,
                            'hal_bitex_tab': tab}, {}, n)
        got = np.asarray(out['Color'])[:, :3]
        want = np.asarray(GraphEvaluator(bigraph, _bi_ctx(n))
                          .eval_output('n', 0), np.float32)[:, :3]
        d = float(np.abs(got - want).max())
        check('colorband GLSL matches the CPU colorband', d < 2e-3,
              f'max diff {d:.2e}')


def test_empty_image_slot_mapping():
    """An image slot with no path and no packed data maps to NOTHING --
    Blender Internal adds nothing for an imageless texture. Emitting a
    black-sampling node instead darkened the colour chain and refused
    the GPU plan (the real file holds nine such slots)."""
    from ..core import blend279_map as BM
    import copy
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data, geometry=False)
    skin = next(m for m in sc['materials'].values() if m['name'] == 'Skin')
    doctored = copy.deepcopy(skin)
    tx = doctored['slots'][0]['tex']
    check('the fixture slot under test is the packed image',
          tx.get('kind') == 'IMAGE' and tx.get('packed'))
    tx['packed'] = None
    tx['image_path'] = ''
    warnings = []
    spec = BM.material_spec(doctored, 279)
    warnings = spec['warnings']
    kinds = [e.get('node') for e in spec['textures']]
    check('the imageless slot maps to no texture entry',
          'ShaderNodeTexImage' not in kinds)
    check('the other slots still arrive',
          'HALCYON_BITextureNode' in kinds)
    check('the skip is named in the warnings',
          any('imageless' in w for w in warnings))
    # the intact material keeps its image entry
    spec2 = BM.material_spec(skin, 279)
    check('a packed image slot still maps',
          any(e.get('image') for e in spec2['textures']))


# --------------------------------------------------- the appender's helpers


def test_appender_planning():
    """plan_objects and match_slot_materials: the pure logic between the
    parser and bpy.data.libraries.load."""
    from . import fakebpy
    bpy = fakebpy.install()
    if not hasattr(bpy.types, 'UIList'):
        bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    from .. import legacy_import as LI
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data, geometry=False)

    names, per = LI.plan_objects(sc, False, True, True)
    check('the plan requests every object type',
          set(names) == {'Cube', 'Plane', 'Spot', 'Camera', 'Rig', 'Path'})
    names, per = LI.plan_objects(sc, True, True, True)
    check('selected-only follows the saved selection',
          set(names) == {'Cube', 'Spot', 'Camera', 'Rig'})
    names, per = LI.plan_objects(sc, False, False, False)
    check('light and camera toggles prune the plan',
          'Spot' not in names and 'Camera' not in names
          and 'Cube' in names)

    names, per = LI.plan_objects(sc, False, True, True)
    cube = per['Cube']
    fake = types.SimpleNamespace(material_slots=[
        types.SimpleNamespace(material=types.SimpleNamespace(
            name='Skin', name_full='Skin')),
        types.SimpleNamespace(material=types.SimpleNamespace(
            name='Chrome.001', name_full='Chrome.001'))])
    pairs = LI.match_slot_materials(fake, cube, sc['materials'])
    check('both slots pair with their BI data despite a .001 rename',
          len(pairs) == 2
          and pairs[0][2]['name'] == 'Skin'
          and pairs[1][2]['name'] == 'Chrome')
    check('slot indices travel with the pairing',
          pairs[0][0] == 0 and pairs[1][0] == 1)
    empty = types.SimpleNamespace(material_slots=[])
    check('objects without slots pair to nothing',
          LI.match_slot_materials(empty, per['Rig'],
                                  sc['materials']) == [])

    # the name path: what conversion actually rides on
    check('append rename suffixes strip back to the classic name',
          LI.base_name('Chrome.001') == 'Chrome'
          and LI.base_name('Skin') == 'Skin'
          and LI.base_name('Mat.0012') == 'Mat'
          and LI.base_name('v1.2') == 'v1.2')
    by_name = LI.parsed_materials_by_name(sc['materials'])
    check('parsed materials index by their unique names',
          set(by_name) == {'Skin', 'Chrome'}
          and by_name['Skin']['har'] == 50)
    fake2 = types.SimpleNamespace(material_slots=[
        types.SimpleNamespace(material=types.SimpleNamespace(
            name='Skin.001', name_full='Skin.001')),
        types.SimpleNamespace(material=types.SimpleNamespace(
            name='Skin.001', name_full='Skin.001')),
        types.SimpleNamespace(material=types.SimpleNamespace(
            name='NotInFile', name_full='NotInFile'))])
    seen = LI.collect_slot_materials([fake2])
    check('slot materials collect once per datablock',
          set(seen) == {'Skin.001', 'NotInFile'})
    check('renamed appended material still finds its BI data by name',
          LI.base_name(seen['Skin.001'].name) in by_name)


def test_conversion_planning():
    """The exact-first name ladder and the pointer-led conversion plan.

    Field-driven: one real file skipped twelve materials because they
    were LITERALLY named 'Material.001'..'Material.012' in the classic
    file and the old code stripped the suffix before looking them up;
    another really held both 'RedWire' and 'RedWire.001' as different
    materials, so the strip silently converted the wrong one."""
    from . import fakebpy
    bpy = fakebpy.install()
    if not hasattr(bpy.types, 'UIList'):
        bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    from .. import legacy_import as LI
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data, geometry=False)
    by_name = LI.parsed_materials_by_name(sc['materials'])

    check('name ladder tries the exact name before stripping',
          list(LI.material_name_candidates('Material.001'))
          == ['Material.001', 'Material'])
    check('double suffixes walk back one at a time',
          list(LI.material_name_candidates('A.001.002'))
          == ['A.001.002', 'A.001', 'A'])
    check('short numeric tails are not append suffixes',
          list(LI.material_name_candidates('v1.2')) == ['v1.2'])

    wire = {'name': 'RedWire'}
    wire1 = {'name': 'RedWire.001'}
    bn = {'RedWire': wire, 'RedWire.001': wire1}
    check('exact-first keeps look-alike classic names apart',
          LI.find_parsed_material('RedWire.001', bn)[0] is wire1
          and LI.find_parsed_material('RedWire', bn)[0] is wire)
    check('a collision rename still strips back to its classic name',
          LI.find_parsed_material('RedWire.002', bn) == (wire, 'RedWire'))
    check('unknown names miss cleanly',
          LI.find_parsed_material('Nope.001', bn) == (None, None))

    names, per = LI.plan_objects(sc, False, True, True)

    def fake_ob(mat_names):
        return types.SimpleNamespace(material_slots=[
            types.SimpleNamespace(material=types.SimpleNamespace(
                name=nm, name_full=nm)) for nm in mat_names])

    cube = fake_ob(['Skin.007', 'Chrome.007'])      # renamed on append
    plan, stats = LI.plan_conversions(
        [('Cube', cube)], per, sc['materials'], by_name)
    check('slot pointers convert index-precisely despite renames',
          len(plan) == 2
          and all(r == 'pointer' for _k, _b, _m, r, _n in plan)
          and plan[0][2]['name'] == 'Skin'
          and plan[1][2]['name'] == 'Chrome'
          and stats['ptr_slots'] == 2 and stats['unmatched'] == [])
    plan2, _s2 = LI.plan_conversions(
        [('Rig', fake_ob(['Skin.001']))], per, sc['materials'], by_name)
    check('the name route takes over when pointers are absent',
          len(plan2) == 1 and plan2[0][3] == 'name'
          and plan2[0][2]['name'] == 'Skin'
          and plan2[0][4] == 'Skin')        # noted as a stripped match
    plan3, _s3 = LI.plan_conversions(
        [('Rig', fake_ob(['Skin']))], per, sc['materials'], by_name)
    check('an exact name match carries no strip note',
          len(plan3) == 1 and plan3[0][3] == 'name'
          and plan3[0][4] is None)
    plan4, s4 = LI.plan_conversions(
        [('Rig', fake_ob(['NotInFile']))], per, sc['materials'], by_name)
    check('unmatched names are reported, not guessed',
          plan4 == [] and s4['unmatched'] == ['NotInFile'])
    many = [('Cube', cube), ('Plane', fake_ob(['Skin', 'Skin']))]
    plan5, _s5 = LI.plan_conversions(many, per, sc['materials'], by_name)
    keys = [k for k, *_ in plan5]
    check('shared materials plan exactly one conversion each',
          len(keys) == len(set(keys)))


# ---------------------------------------------------------------------- run


def test_scene_pipeline_mapping():
    """R159: the file's OWN pipeline, read from the DNA.

    Every field .blend says display_device sRGB + view 'Default' --
    2.79 rendered BI scene-linear and sRGB-ENCODED the frame on
    display. Halcyon showed the same arithmetic RAW, which is the
    field's 'lighting is dark / I think the Gamma might be different'.
    The parse, the mapping, and the two pipeline ends are pinned here.
    """
    import numpy as np
    from ..core import blend279_map as M2
    from ..core import mathx as MX

    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data)
    cm = sc.get('scene_cm')
    rd = sc.get('scene_render')
    check("the 279 scene's view transform and display device parse",
          cm is not None and cm['view_transform'] == 'Default'
          and cm['display_device'] == 'sRGB'
          and abs(cm['exposure'] - 0.25) < 1e-6
          and abs(cm['gamma'] - 1.0) < 1e-6, str(cm))
    check("the 279 RenderData parses: frame size, %, transparent sky",
          rd is not None and rd['xsch'] == 1920 and rd['ysch'] == 1920
          and rd['size'] == 50 and rd['alphamode'] == 1
          and rd['osa'] == 16, str(rd))
    sm, w = M2.scene_settings_map(sc)
    check("'Default' on sRGB maps to Halcyon's OWN pipeline: SRGB "
          'encode out, linearized textures in',
          sm['color_management'] == 'SRGB'
          and sm['input_gamma_naive'] is False and not w, str(sm))
    check('OCIO exposure is STOPS: 0.25 -> x2^0.25',
          abs(sm['exposure'] - 2.0 ** 0.25) < 1e-9)
    check('the frame settings travel: 1920x1920 @50%, transparent film '
          "(the field's 960x960 white-background F12 was a transparent "
          'PNG on a white viewer)',
          sm['res_x'] == 1920 and sm['res_pct'] == 50
          and sm['film_transparent'] is True)

    # a pre-OCIO file has no pipeline block: period-correct NONE
    d9, _t9 = FX.build_249()
    s9 = B.read_legacy_scene(d9)
    sm9, _w9 = M2.scene_settings_map(s9)
    check('a 2.49 file (no OCIO in the DNA) maps to NONE + naive '
          'inputs -- bytes in, bytes out, exactly that era',
          s9.get('scene_cm') is None
          and sm9['color_management'] == 'NONE'
          and sm9['input_gamma_naive'] is True)

    # the mapping's edges
    smd, _ = M2.scene_settings_map(
        {'scene_cm': {'display_device': 'None',
                      'view_transform': 'Default'}})
    check("display device 'None' (the author turned CM off) maps to "
          'NONE + naive', smd['color_management'] == 'NONE'
          and smd['input_gamma_naive'] is True)
    smr, _ = M2.scene_settings_map(
        {'scene_cm': {'display_device': 'sRGB',
                      'view_transform': 'Raw'}})
    check("view 'Raw' keeps linear inputs but drops the encode",
          smr['color_management'] == 'NONE'
          and smr['input_gamma_naive'] is False)
    smf, wf = M2.scene_settings_map(
        {'scene_cm': {'display_device': 'sRGB',
                      'view_transform': 'Filmic'}})
    check("an unimplemented view ('Filmic') warns BY NAME and falls "
          'back to the sRGB encode',
          smf['color_management'] == 'SRGB'
          and any('Filmic' in x for x in wf), str(wf))

    # the curve itself: the exact piecewise sRGB OETF, both directions
    xs = np.array([0.0, 0.0031308, 0.01, 0.05, 0.2, 0.5, 1.0],
                  np.float32)
    enc = MX.linear_to_srgb(xs)
    check('linear_to_srgb is the piecewise OETF: 12.92 slope below '
          'the knee, 1.055x^(1/2.4)-0.055 above',
          abs(float(enc[1]) - 0.0031308 * 12.92) < 1e-6
          and abs(float(enc[4]) - (1.055 * 0.2 ** (1 / 2.4) - 0.055))
          < 1e-6
          and abs(float(enc[6]) - 1.0) < 1e-6, str(enc))
    check('the encode LIFTS midtones the way the field saw 2.79 do: '
          'linear 0.2 displays near 0.48',
          0.45 < float(enc[4]) < 0.51)
    back = MX.srgb_to_linear(enc)
    check('srgb_to_linear inverts it to 1e-6',
          float(np.abs(back - xs).max()) < 1e-6)

    # the import applies it to the scene (both routes share this call)
    import types as _t
    from .. import legacy_import as LI
    applied = {}

    class _HS:
        def __setattr__(self, k, v):
            applied[k] = v

    scene = _t.SimpleNamespace(
        halcyon=_HS(),
        render=_t.SimpleNamespace(resolution_x=0, resolution_y=0,
                                  resolution_percentage=100))
    warns = []
    LI._apply_scene_pipeline(scene, sc, warns)
    check('_apply_scene_pipeline lands every field on the scene: the '
          'encode, the input end, exposure, film and frame size',
          applied.get('color_management') == 'SRGB'
          and applied.get('input_gamma_naive') is False
          and abs(applied.get('exposure', 0) - 2.0 ** 0.25) < 1e-9
          and applied.get('film_transparent') is True
          and scene.render.resolution_x == 1920
          and scene.render.resolution_percentage == 50, str(applied))


def test_appended_route_end_to_end():
    """R160: the operator's APPENDED branch, driven headlessly.

    The field reported every imported sun at exactly 1.0 -- Blender's
    own append default -- across several updates: the one code path no
    harness had ever executed was the appended branch of execute()
    itself (fakebpy has no bpy.data.libraries, so every prior test
    silently exercised the fallback). This drives it with a stub
    library whose lamps arrive at Blender's default 1.0, and asserts
    the enrichment overrides them with the FILE's energies, leaves a
    receipt in the log, and survives a material-stage crash."""
    import types as _t
    from . import fakebpy
    bpy = fakebpy.install()
    from .. import legacy_import as LI

    path = '/tmp/halcyon_fix279_e2e.blend'
    data, truth = FX.build_279()
    with open(path, 'wb') as fh:
        fh.write(data)
    parsed = B.read_legacy_scene(path, geometry=False)
    file_names = [o['name'] for o in parsed['objects']]

    class _AnyNS:
        """Accepts any attribute, records them (bpy prop groups)."""

        def __init__(self):
            object.__setattr__(self, 'seen', {})

        def __setattr__(self, k, v):
            self.seen[k] = v

        def __getattr__(self, k):
            try:
                return object.__getattribute__(self, 'seen')[k]
            except KeyError:
                return None

    made_lights = {}

    class _Ob:
        """Hashable (identity) like a real bpy object; namespaces are
        not, and execute() builds a set of them."""

        def __init__(self, **kw):
            self.__dict__.update(kw)

        def select_set(self, *_a):
            pass

    def _stub_object(name):
        kind = next((o['kind'] for o in parsed['objects']
                     if o['name'] == name), 'MESH')
        data_blk = None
        if kind == 'LAMP':
            # exactly what Blender's own append hands over: the sun/
            # spot at the DEFAULT strength, the file's energy dropped
            data_blk = _t.SimpleNamespace(
                name=name, energy=1.0, color=(1.0, 1.0, 1.0),
                spot_size=0.5, spot_blend=0.1, halcyon=_AnyNS())
            made_lights[name] = data_blk
        elif kind == 'MESH':
            data_blk = _t.SimpleNamespace(name=name, materials=[])
        return _Ob(name=name, data=data_blk, material_slots=[],
                   parent=None, constraints=[], modifiers=[],
                   hide_render=False, hide_viewport=False,
                   users_collection=[1])

    class _Load:
        def __init__(self, _fp):
            self.dfrom = _t.SimpleNamespace(objects=list(file_names))
            self.dto = _t.SimpleNamespace(objects=[])

        def __enter__(self):
            return self.dfrom, self.dto

        def __exit__(self, *_exc):
            self.dto.objects = [_stub_object(n)
                                for n in self.dto.objects]
            return False

    class _LinkList(list):
        def link(self, ob):
            self.append(ob)

    def _collection_new(name):
        return _t.SimpleNamespace(name=name, objects=_LinkList(),
                                  children=_LinkList())

    bpy.data.libraries = _t.SimpleNamespace(load=_Load)
    bpy.data.collections = _t.SimpleNamespace(new=_collection_new)
    if not hasattr(bpy.data, 'texts'):
        bpy.data.texts = _t.SimpleNamespace(
            new=lambda n: _t.SimpleNamespace(write=lambda s: None))

    scene = _t.SimpleNamespace(
        collection=_t.SimpleNamespace(children=_LinkList()),
        halcyon=_AnyNS(), world=None,
        render=_t.SimpleNamespace(resolution_x=0, resolution_y=0,
                                  resolution_percentage=100),
        display_settings=_t.SimpleNamespace(display_device=''),
        view_settings=_t.SimpleNamespace(view_transform='', look='',
                                         exposure=0.0, gamma=1.0))
    ctx = _t.SimpleNamespace(
        scene=scene,
        view_layer=_t.SimpleNamespace(
            objects=_t.SimpleNamespace(active=None)))

    reports = []
    op = LI.HALCYON_OT_append_legacy.__new__(LI.HALCYON_OT_append_legacy)
    op.filepath = path
    op.only_selected = False
    op.import_lights = True
    op.import_cameras = True
    op.import_world = False
    op.include_hidden = False
    op.scale = 1.0
    op.report = lambda kind, msg: reports.append((tuple(kind), msg))

    warnings_log = []
    _orig_texts_new = bpy.data.texts.new

    class _Log:
        lines = []

        def write(self, s):
            warnings_log.append(s)
    bpy.data.texts.new = lambda n: _Log()
    try:
        result = op.execute(ctx)
    finally:
        bpy.data.texts.new = _orig_texts_new
    check('the appended route completes', result == {'FINISHED'},
          str(result))
    la = made_lights.get('Spot')
    want = 1.5 * 4.0 * 3.14159265 ** 2      # the FILE's energy, BI units
    check("the appended lamp's DEFAULT 1.0 is overridden with the "
          "file's own energy (the every-sun-reads-1.0 field report)",
          la is not None and abs(la.energy - want) < 1e-3,
          f'energy {getattr(la, "energy", None)}')
    check('the spot keeps its buffer-shadow rule through the stub too',
          la is not None and la.halcyon.seen.get('shadow') == 'MAP')
    joined = '\n'.join(warnings_log)
    check('the import log carries the lamp receipt',
          'lamps enriched' in joined and 'file energy 1.5' in joined,
          joined[-300:])
    check('...and the UNCONDITIONAL lamp-stage count line (the '
          'the truncated field log had neither branch line, an absence the two '
          'branches cannot produce -- this line cannot be skipped)',
          'lamp stage reached: 1 lamp object(s) among' in joined)
    check('...and the hardened writer\'s terminator',
          'log complete (' in joined)
    from ..version import version_string as _vstr
    check('the import log names the running version (stale installs '
          'become visible)', f'Halcyon {_vstr()}' in joined)
    check('the scene pipeline landed on the appended route too',
          scene.halcyon.seen.get('color_management') == 'SRGB'
          and scene.render.resolution_x == 1920)

    # a material-stage crash must NOT starve the lamps (the fence)
    made_lights.clear()
    import types as _t2
    orig_plan = LI.plan_conversions
    LI.plan_conversions = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError('boom'))
    try:
        op2 = LI.HALCYON_OT_append_legacy.__new__(
            LI.HALCYON_OT_append_legacy)
        for k, v in (('filepath', path), ('only_selected', False),
                     ('import_lights', True), ('import_cameras', True),
                     ('import_world', False), ('include_hidden', False),
                     ('scale', 1.0)):
            setattr(op2, k, v)
        op2.report = lambda *a, **kw: None
        r2 = op2.execute(ctx)
    finally:
        LI.plan_conversions = orig_plan
    la2 = made_lights.get('Spot')
    check('a material-planning crash is fenced: import completes and '
          'the lamps STILL get their energies',
          r2 == {'FINISHED'} and la2 is not None
          and abs(la2.energy - want) < 1e-3,
          f'r={r2} energy={getattr(la2, "energy", None)}')

    # ---- hidden-layer objects: the pure set, and the flags on the
    # fallback builder
    check('hidden_object_names: layer outside the mask -> hidden',
          LI.hidden_object_names(
              {'A': {'layers': 0x2}, 'B': {'layers': 0x1}}, 0x1)
          == {'A'})
    check('...and an absent mask hides nothing (pre-layer eras)',
          LI.hidden_object_names({'A': {'layers': 0x2}}, 0) == set())
    lamp_parsed = next(o for o in parsed['objects']
                       if o['name'] == 'Spot')
    coll2 = _t2.SimpleNamespace(objects=_LinkList())
    warns2 = []
    imported2, _nm, _ni = LI._fallback_import(
        ctx, path, {'Spot': lamp_parsed}, {}, 279, '/tmp', coll2,
        warns2, hidden={'Spot'})
    check('a hidden-layer object imports HIDDEN from viewport and '
          'render, as 2.79 kept it',
          len(imported2) == 1 and imported2[0].hide_render is True
          and imported2[0].hide_viewport is True)
    check('the fallback route leaves the same lamp receipt',
          any('file energy 1.5' in w for w in warns2), str(warns2))

    # a light whose energy setter clamps (driver quirk, future API
    # change) is CALLED OUT by the readback, never silent
    class _Clamped:
        color = (1, 1, 1)

        @property
        def energy(self):
            return 1.0

        @energy.setter
        def energy(self, v):
            pass                              # refuses the write
    warns3 = []
    LI._apply_lamp_bi(_Clamped(), {'type': 'SUN', 'name': 'Odd',
                                   'energy': 3.0, 'color': (1, 1, 1)},
                      warns3)
    check('an energy write that does not stick is a NAMED warning '
          '(the every-sun-reads-1.0 mystery can never be silent again)',
          any('did not stick' in w and 'Odd' in w for w in warns3),
          str(warns3))


def test_fix_appended_lamps():
    """R165: the repair operator for lamps brought in via PLAIN File >
    Append -- the route on which no Halcyon code ever runs.

    The field's every-lamp-reads-1.0 reports persisted across rounds
    of importer fixes; the standing hypothesis is that the scene never
    went through File > Import > Legacy .blend at all, but through
    Blender's own append, which converts every 2.79 lamp to its modern
    default. The fixer reads the classic file after the fact and
    stamps its OWN values onto the scene's lights by name -- exact
    first, then the .001 rename suffixes walked off, object names
    before lamp-data names."""
    import types as _t
    from . import fakebpy
    bpy = fakebpy.install()
    from .. import legacy_import as LI

    # ---- the pure matching ladder
    by_obj = {'Spot': {'id': 1}, 'Sun.001': {'id': 2}}
    by_data = {'Lamp': {'id': 3}}
    check('exact object name matches first',
          LI._match_parsed_lamp('Spot', 'X', by_obj, by_data)
          == ({'id': 1}, 'Spot', 'object'))
    check('a file object legitimately NAMED with a suffix matches '
          'exactly, no strip',
          LI._match_parsed_lamp('Sun.001', '', by_obj, by_data)
          == ({'id': 2}, 'Sun.001', 'object'))
    check('an append rename walks its suffixes off one at a time',
          LI._match_parsed_lamp('Spot.001.002', '', by_obj, by_data)
          == ({'id': 1}, 'Spot', 'object'))
    check('lamp-data names catch a hand-renamed object',
          LI._match_parsed_lamp('KeyLight', 'Lamp.003', by_obj, by_data)
          == ({'id': 3}, 'Lamp', 'lamp data'))
    check('no match -> all None',
          LI._match_parsed_lamp('Ghost', 'Ghost', by_obj, by_data)
          == (None, None, None))

    # ---- the operator, headless, against the real fixture file
    path = '/tmp/halcyon_fix_lamps.blend'
    data, _truth = FX.build_279()
    with open(path, 'wb') as fh:
        fh.write(data)

    class _AnyNS:
        def __init__(self):
            object.__setattr__(self, 'seen', {})

        def __setattr__(self, k, v):
            self.seen[k] = v

        def __getattr__(self, k):
            try:
                return object.__getattribute__(self, 'seen')[k]
            except KeyError:
                return None

    class _Ob:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def _light(ob_name, data_name):
        # exactly what plain File > Append leaves behind: Blender's
        # own converted default strength, the file's energy gone
        blk = _t.SimpleNamespace(name=data_name, energy=1.0,
                                 color=(1.0, 1.0, 1.0), spot_size=0.1,
                                 spot_blend=0.0, halcyon=_AnyNS())
        return _Ob(name=ob_name, type='LIGHT', data=blk)

    spot = _light('Spot', 'Spot')             # exact-name match
    spot001 = _light('Spot.001', 'Spot.001')  # append-rename match
    ghost = _light('Ghost', 'Ghost')          # nothing in the file
    cube = _Ob(name='Cube', type='MESH',
               data=_t.SimpleNamespace(name='Cube', energy=1.0))

    scene = _t.SimpleNamespace(
        objects=[spot, spot001, ghost, cube],
        halcyon=_AnyNS(), world=None,
        render=_t.SimpleNamespace(resolution_x=0, resolution_y=0,
                                  resolution_percentage=100),
        display_settings=_t.SimpleNamespace(display_device=''),
        view_settings=_t.SimpleNamespace(view_transform='', look='',
                                         exposure=0.0, gamma=1.0))
    ctx = _t.SimpleNamespace(scene=scene, selected_objects=[])
    if not hasattr(bpy.data, 'texts'):
        bpy.data.texts = _t.SimpleNamespace(
            new=lambda n: _t.SimpleNamespace(write=lambda s: None))
    log_lines = []
    _orig_texts_new = bpy.data.texts.new

    class _Log:
        def write(self, s):
            log_lines.append(s)
    bpy.data.texts.new = lambda n: _Log()

    def _op(**kw):
        op = LI.HALCYON_OT_fix_appended_lamps.__new__(
            LI.HALCYON_OT_fix_appended_lamps)
        op.filepath = path
        op.only_selected = False
        op.apply_pipeline = False
        for k, v in kw.items():
            setattr(op, k, v)
        op.reports = []
        op.report = lambda kind, msg, _r=op: _r.reports.append(
            (tuple(kind), msg))
        return op

    try:
        op = _op()
        result = op.execute(ctx)
        want = 1.5 * 4.0 * 3.14159265 ** 2    # the FILE's spot, BI units
        check('the fixer completes', result == {'FINISHED'}, str(result))
        check("an exact-name light gets the file's own energy",
              abs(spot.data.energy - want) < 1e-3,
              f'energy {spot.data.energy}')
        check('...and the full BI conversion rides along: shadow rule, '
              'colour, spot cone, decay distance',
              spot.data.halcyon.seen.get('shadow') == 'MAP'
              and abs(spot.data.spot_size - 0.7853982) < 1e-6
              and abs(spot.data.color[1] - 0.9) < 1e-6
              and _close(spot.data.halcyon.seen.get('decay_end'), 25.0))
        check('a .001 append rename matches the same file lamp',
              abs(spot001.data.energy - want) < 1e-3,
              f'energy {spot001.data.energy}')
        check('a light the file does not know keeps its values',
              ghost.data.energy == 1.0)
        check('a mesh named like nothing in the file is not a target '
              '(no crash, untouched)', cube.data.energy == 1.0)
        joined = ''.join(log_lines)
        from ..version import version_string as _vstr2
        check('the log opens with the running version',
              f'Halcyon {_vstr2()}' in joined, joined[:120])
        check('the log carries a receipt per fixed light',
              'Spot: file energy 1.5 ->' in joined
              and 'Spot.001: file energy 1.5 ->' in joined, joined)
        check("...naming the file lamp a rename matched (it wasn't "
              'the exact name)', "(matched file object 'Spot')" in joined)
        check('unmatched lights are reported by name, with what the '
              'file offers', 'Ghost' in joined
              and 'the file offers these lamp names' in joined)
        check('the fixer does NOT touch the scene pipeline by default',
              'color_management' not in scene.halcyon.seen
              and scene.render.resolution_x == 0)

        # ---- Selected Lights Only
        spot.data.energy = 1.0
        spot001.data.energy = 1.0
        ctx.selected_objects = [spot001, cube]
        r2 = _op(only_selected=True).execute(ctx)
        check('Selected Lights Only fixes just the selection',
              r2 == {'FINISHED'} and abs(spot001.data.energy - want) < 1e-3
              and spot.data.energy == 1.0)
        ctx.selected_objects = []
        r3 = _op(only_selected=True).execute(ctx)
        check('...and an empty selection cancels with a hint',
              r3 == {'CANCELLED'})

        # ---- the optional scene pipeline, exactly the importer's
        r4 = _op(apply_pipeline=True).execute(ctx)
        check('Scene Pipeline Too lands the sRGB view and frame size',
              r4 == {'FINISHED'}
              and scene.halcyon.seen.get('color_management') == 'SRGB'
              and scene.render.resolution_x == 1920)

        # ---- a scene with no lights at all
        ctx2 = _t.SimpleNamespace(
            scene=_t.SimpleNamespace(objects=[cube], halcyon=_AnyNS()),
            selected_objects=[])
        r5 = _op().execute(ctx2)
        check('a lightless scene cancels with a warning',
              r5 == {'CANCELLED'})
    finally:
        bpy.data.texts.new = _orig_texts_new


def test_append_watch():
    """R166: the lamp fix happens AT APPEND TIME, hands-free.

    The user's verdict on the manual fixer: 'It works, BUT it needs
    to happen when appending... convenience/ease of use is key.'
    blend_import_post (Blender 4.1+, contract confirmed against the
    current rna_blendfile_import.cc: one BlendImportContext argument
    carrying import_items with the actual `id` pointers and per-item
    source filepaths) triggers the SAME fixer core automatically --
    gated on the Halcyon engine, the scene toggle, append-not-link,
    and a provably-classic source file."""
    import gzip as _gz
    import types as _t
    from . import fakebpy
    bpy = fakebpy.install()
    from .. import append_watch as AW
    from .. import legacy_import as LI

    path = '/tmp/halcyon_watch_279.blend'
    data, _truth = FX.build_279()
    with open(path, 'wb') as fh:
        fh.write(data)

    # ---- the header sniff: classic in, everything else out
    check('a classic header reads its version',
          AW.blend_header_version(path) == 279)
    gzpath = '/tmp/halcyon_watch_279.blend.gz.blend'
    with open(gzpath, 'wb') as fh:
        fh.write(_gz.compress(data))
    check('...through gzip, as 2.4x-era saves are wrapped',
          AW.blend_header_version(gzpath) == 279)
    modpath = '/tmp/halcyon_watch_modern.blend'
    with open(modpath, 'wb') as fh:
        fh.write(b'BLENDER-v502' + b'\x00' * 64)
    check('a modern header is NOT classic',
          AW.blend_header_version(modpath) == 502)
    zstpath = '/tmp/halcyon_watch_zstd.blend'
    with open(zstpath, 'wb') as fh:
        fh.write(b'\x28\xb5\x2f\xfd' + b'\x00' * 64)
    check('Zstandard files are modern by definition -> None',
          AW.blend_header_version(zstpath) is None)
    check('garbage and missing files are None, never a raise',
          AW.blend_header_version('/tmp/halcyon_watch_missing.blend')
          is None)

    # ---- collect_light_jobs: exactly the appended lights, by file
    class _Ob:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    def _light(ob_name, data_name):
        blk = _t.SimpleNamespace(name=data_name, energy=1.0,
                                 color=(1.0, 1.0, 1.0), spot_size=0.1,
                                 spot_blend=0.0, library=None,
                                 halcyon=_t.SimpleNamespace())
        return _Ob(name=ob_name, type='LIGHT', data=blk, library=None)

    def _item(idb, id_type='OBJECT', action='MAKE_LOCAL', fp=path):
        return _t.SimpleNamespace(
            id=idb, id_type=id_type, append_action=action,
            source_libraries=[_t.SimpleNamespace(filepath=fp)],
            source_library=None)

    spot = _light('Spot', 'Spot')
    mesh = _Ob(name='Cube', type='MESH',
               data=_t.SimpleNamespace(name='Cube'), library=None)
    linked = _light('Linked', 'Linked')
    linked.library = object()
    bare = _t.SimpleNamespace(name='Spot', energy=1.0,
                              color=(1.0, 1.0, 1.0), spot_size=0.1,
                              spot_blend=0.0, library=None,
                              halcyon=_t.SimpleNamespace())
    jobs = AW.collect_light_jobs([
        _item(spot),
        _item(mesh),                              # not a light
        _item(linked),                            # library-owned
        _item(_light('Kept', 'Kept'), action='KEEP_LINKED'),
        _item(_light('Reused', 'Reused'), action='REUSE_LOCAL'),
        _item(None),                              # id missing
        _item(bare, id_type='LIGHT'),             # a data-section append
        _t.SimpleNamespace(id=_light('NoPath', 'NoPath'),
                           id_type='OBJECT', append_action='MAKE_LOCAL',
                           source_libraries=[], source_library=None),
    ])
    check('only the appended light object and the bare Light survive '
          'the item filter',
          list(jobs) == [path]
          and [o.name for o in jobs[path]['objects']] == ['Spot']
          and [l.name for l in jobs[path]['lights']] == ['Spot'])

    # ---- the handler end-to-end against the real fixture
    scene = _t.SimpleNamespace(
        render=_t.SimpleNamespace(engine='HALCYON_RENDER'),
        halcyon=_t.SimpleNamespace(auto_fix_appended_lamps=True))
    bpy.context = _t.SimpleNamespace(scene=scene)
    if not hasattr(bpy.data, 'texts'):
        bpy.data.texts = _t.SimpleNamespace(
            new=lambda n: _t.SimpleNamespace(write=lambda s: None))
    log_lines = []
    _orig_texts_new = bpy.data.texts.new

    class _Log:
        def write(self, s):
            log_lines.append(s)
    bpy.data.texts.new = lambda n: _Log()

    def _ctx(items, **kw):
        base = dict(process_stage='DONE', options=set(),
                    import_items=items)
        base.update(kw)
        return _t.SimpleNamespace(**base)

    want = 1.5 * 4.0 * 3.14159265 ** 2
    try:
        # a real append emits BOTH an Object item and a Light item for
        # one lamp -- and item.id of the Light item IS ob.data, the
        # same datablock, which is what the identity dedup rides on
        AW.on_blend_import_post(_ctx([_item(spot),
                                      _item(spot.data,
                                            id_type='LIGHT')]))
        check("the watch stamps the file's energy the moment the "
              'append lands', abs(spot.data.energy - want) < 1e-3,
              f'energy {spot.data.energy}')
        check('...with the shadow rule riding along',
              getattr(spot.data.halcyon, 'shadow', None) == 'MAP')
        check('a lamp arriving as Object item + Light item is ONE '
              'lamp: fixed once, one receipt',
              ''.join(log_lines).count('file energy 1.5') == 1,
              ''.join(log_lines))
        joined = ''.join(log_lines)
        from ..version import version_string as _vstr3
        check('the watch log opens with the version and says what it '
              'did', f'Halcyon {_vstr3()}' in joined
              and 'append watch fixed 1 of 1' in joined, joined[:200])
        check('the hardened writer terminates the log',
              'log complete (' in joined)

        # a TRULY bare light (no object item) matches by data name
        log_lines.clear()
        bare2 = _t.SimpleNamespace(name='Spot.001', energy=1.0,
                                   color=(1.0, 1.0, 1.0), spot_size=0.1,
                                   spot_blend=0.0, library=None,
                                   halcyon=_t.SimpleNamespace())
        AW.on_blend_import_post(_ctx([_item(bare2, id_type='LIGHT')]))
        check('a data-section append (no object) is matched through '
              'the lamp-data pool, rename suffix walked off',
              abs(bare2.energy - want) < 1e-3, f'energy {bare2.energy}')

        # ---- the gates, each proven to hold the watch back
        def _fresh():
            return _light('Spot', 'Spot')

        g1 = _fresh()
        scene.render.engine = 'CYCLES'
        AW.on_blend_import_post(_ctx([_item(g1)]))
        check("another engine's append is never touched",
              g1.data.energy == 1.0)
        scene.render.engine = 'HALCYON_RENDER'

        g2 = _fresh()
        scene.halcyon.auto_fix_appended_lamps = False
        AW.on_blend_import_post(_ctx([_item(g2)]))
        check('the scene toggle turns the watch off',
              g2.data.energy == 1.0)
        scene.halcyon.auto_fix_appended_lamps = True

        g3 = _fresh()
        AW.on_blend_import_post(_ctx([_item(g3)], options={'LINK'}))
        check('a LINK import passes through untouched',
              g3.data.energy == 1.0)

        g4 = _fresh()
        AW.on_blend_import_post(_ctx([_item(g4)],
                                     process_stage='INIT'))
        check('the pre stage of the contract is not acted on',
              g4.data.energy == 1.0)

        g5 = _fresh()
        AW.on_blend_import_post(_ctx([_item(g5, fp=modpath)]))
        check('a modern source file is left to Blender entirely',
              g5.data.energy == 1.0)

        # ---- robustness: the handler must never raise
        AW.on_blend_import_post()
        AW.on_blend_import_post(None)
        AW.on_blend_import_post(_t.SimpleNamespace())
        AW.on_blend_import_post(_ctx([_t.SimpleNamespace()]))
        bpy.context = None
        AW.on_blend_import_post(_ctx([_item(_fresh())]))
        bpy.context = _t.SimpleNamespace(scene=scene)
        check('no context shape aborts the append', True)

        # ---- idempotence: the watch re-applied is the same light
        again = _fresh()
        AW.on_blend_import_post(_ctx([_item(again)]))
        first = again.data.energy
        AW.on_blend_import_post(_ctx([_item(again)]))
        check('running the watch twice is the same as once',
              abs(again.data.energy - first) < 1e-12
              and abs(first - want) < 1e-3)
    finally:
        bpy.data.texts.new = _orig_texts_new

    # ---- registration against a stub handler list
    handlers = _t.SimpleNamespace(blend_import_post=[])
    bpy.app = getattr(bpy, 'app', _t.SimpleNamespace())
    bpy.app.handlers = handlers
    AW.register()
    AW.register()
    check('register adds the handler exactly once',
          handlers.blend_import_post == [AW.on_blend_import_post])
    check('the handler is marked persistent (survives file loads)',
          AW.on_blend_import_post.__dict__.get('_bpy_persistent',
                                               'unmarked')
          != 'unmarked' or AW.persistent.__name__ == 'persistent')
    AW.unregister()
    check('unregister removes it', handlers.blend_import_post == [])

    # ---- the hardened log writer, in isolation
    wrote = []

    class _GoodText:
        def write(self, s):
            wrote.append(s)
    bpy.data.texts.new = lambda n: _GoodText()
    try:
        LI.write_log_text('t', ['a', 'b'])
        check('whole-payload write carries every line and the '
              'terminator', wrote == ['a\nb\nlog complete (2 lines)\n'])

        class _PoisonText:
            def write(self, s):
                if 'poison' in s:
                    raise UnicodeEncodeError('utf-8', 'x', 0, 1, 'bad')
                wrote.append(s)
        wrote.clear()
        bpy.data.texts.new = lambda n: _PoisonText()
        LI.write_log_text('t', ['a', 'poison line', 'b'])
        joined = ''.join(wrote)
        check('a poison line falls back to per-line writes: the OTHER '
              'lines and the terminator survive, the bad line is a '
              'marker NAMING the exception (the field-log truncation '
              'can never be silent again)',
              'a\n' in joined and 'b\n' in joined
              and 'log complete (3 lines)' in joined
              and 'UnicodeEncodeError' in joined, repr(joined))

        def _no_text(_n):
            raise RuntimeError('no texts here')
        bpy.data.texts.new = _no_text
        check('no text datablock available -> None, never a raise',
              LI.write_log_text('t', ['a']) is None)
    finally:
        bpy.data.texts.new = _orig_texts_new


def test_sss_import():
    """R162: the Subsurface Scattering panel travels DNA -> spec ->
    node props -> the engine graph, and the flag gates it exactly."""
    from ..core import blend279_map as M2

    base = {'name': 'Skin', 'mode': 0, 'diff_shader': 0,
            'spec_shader': 0, 'r': 0.8, 'g': 0.6, 'b': 0.5,
            'ref': 0.8, 'spec': 0.2, 'har': 50, 'alpha': 1.0,
            'sss_flag': 1, 'sss_scale': 0.05,
            'sss_radius': (3.67, 1.37, 0.68),
            'sss_col': (0.9, 0.9, 0.9), 'sss_ior': 1.44,
            'sss_error': 0.02, 'sss_colfac': 0.5, 'sss_texfac': 0.25,
            'sss_front': 1.2, 'sss_back': 4.0, 'slots': []}
    sp = M2.material_spec(base, 279)
    bi = sp['bi']
    check('sss_flag & MA_DIFF_SSS maps EVERY panel field onto the BI '
          'node props',
          bi.get('sss_enable') is True
          and abs(bi['sss_scale'] - 0.05) < 1e-6
          and abs(bi['sss_radius'][0] - 3.67) < 1e-5
          and abs(bi['sss_color'][2] - 0.9) < 1e-6
          and abs(bi['sss_ior'] - 1.44) < 1e-6
          and abs(bi['sss_error'] - 0.02) < 1e-6
          and abs(bi['sss_colfac'] - 0.5) < 1e-6
          and abs(bi['sss_texfac'] - 0.25) < 1e-6
          and abs(bi['sss_front'] - 1.2) < 1e-6
          and abs(bi['sss_back'] - 4.0) < 1e-6, str(bi.get('sss_scale')))
    off = M2.material_spec(dict(base, sss_flag=0), 279)
    check('the panel OFF leaves the node clean (no sss props at all)',
          'sss_enable' not in off['bi'])

    # through build_spec onto the (fake) node, then engine-side pickup
    socks, specs = _shader_socket_table()
    from .. import templates as TPL
    from ..core.render import bi_sss_params
    fm = _fake_material(socks, specs)
    shader = TPL.build_spec(fm, sp)
    check('build_spec lands the props on the BI node',
          getattr(shader, 'sss_enable', None) is True
          and abs(float(getattr(shader, 'sss_ior', 0)) - 1.44) < 1e-6)
    graph = {'nodes': {'bi': {'bl_idname': 'HALCYON_BIMaterialNode',
                              'props': dict(bi)}}}
    import types as _t
    params = bi_sss_params(_t.SimpleNamespace(graph=graph,
                                              shadeless=False))
    check('bi_sss_params reads the serialized graph back into engine '
          'parameters', params is not None
          and abs(params['back'] - 4.0) < 1e-6
          and abs(params['radius'][1] - 1.37) < 1e-5)
    from ..export import NODE_PROPS
    check('the exporter serializes every SSS prop',
          all(k in NODE_PROPS['HALCYON_BIMaterialNode']
              for k in ('sss_enable', 'sss_scale', 'sss_radius',
                        'sss_color', 'sss_ior', 'sss_error',
                        'sss_colfac', 'sss_texfac', 'sss_front',
                        'sss_back')))


def test_lamp_loop_tail_import():
    """R164: the tail features travel from the DNA."""
    from ..core import blend279_map as M2

    base = {'name': 'T', 'mode': 0, 'diff_shader': 0, 'spec_shader': 0,
            'r': 0.8, 'g': 0.6, 'b': 0.5, 'ref': 0.8, 'spec': 0.2,
            'har': 50, 'alpha': 1.0, 'slots': []}
    sp = M2.material_spec(dict(base, sbias=0.12,
                               mode=0x400000, shade_flag=3), 279)
    bi = sp['bi']
    check('sbias, MA_RAYBIAS (0x400000) and MA_OBCOLOR (shade_flag&2) '
          'land on the BI node',
          abs(bi.get('sbias', 0) - 0.12) < 1e-6
          and bi.get('raybias') is True
          and bi.get('use_obcolor') is True)
    off = M2.material_spec(base, 279)['bi']
    check('all three stay OFF the node when the DNA says off',
          'sbias' not in off and 'raybias' not in off
          and 'use_obcolor' not in off)

    lm = M2.lamp_map({'name': 'L', 'kind': 'SPOT', 'type': 2,
                      'energy': 1.0, 'r': 1, 'g': 1, 'b': 1,
                      'dist': 25.0, 'mode': 1, 'falloff_type': 2,
                      'shdwr': 0.3, 'shdwg': 0.1, 'shdwb': 0.05,
                      'spotsize': 0.8, 'spotblend': 0.15}, 279)
    check("the lamp's shadow colour travels",
          abs(lm['shadow_color'][0] - 0.3) < 1e-6
          and abs(lm['shadow_color'][2] - 0.05) < 1e-6)

    warns = []
    wm = M2.world_map({'horr': 0.1, 'horg': 0.1, 'horb': 0.1,
                       'zenr': 0, 'zeng': 0, 'zenb': 0,
                       'ambr': 0, 'ambg': 0, 'ambb': 0, 'skytype': 1,
                       'exp': 0.35, 'range': 2.5, 'misi': 0.8,
                       'miststa': 4.0, 'mistdist': 30.0, 'mistype': 0,
                       'misthi': 5.0}, warns)
    check('world exposure and range travel',
          abs(wm['exposure'] - 0.35) < 1e-6
          and abs(wm['exposure_range'] - 2.5) < 1e-6)
    check('mist carries its CURVE (mistype 0 = quadratic) and the '
          'Height limit warns BY NAME',
          wm['mist']['falloff'] == 'QUADRATIC'
          and any('Height' in w for w in warns), str(warns))

    # object colour + smoothresh parse from the fixture file
    data, truth = FX.build_279()
    sc = B.read_legacy_scene(data, geometry=False)
    ob = next(o for o in sc['objects'] if o['kind'] == 'MESH')
    check('objects carry col and smoothresh through the parse '
          '(defaults when the fixture never set them)',
          'col' in ob and 'smoothresh' in ob)


def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith('test_') and callable(v)]
    for name, fn in tests:
        print(name)
        fn()
    print()
    if FAILS:
        print(f'{len(FAILS)} FAILED: {FAILS}')
        return 1
    print('all legacy-import checks passed')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
