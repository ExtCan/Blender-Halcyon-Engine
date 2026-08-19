"""Blender Internal -> Halcyon: the material conversion itself.

This module is bpy-free: it takes the plain dicts blend279.py extracts and
returns a template SPEC -- the same shape templates.py builds node trees
from -- so every decision here is testable without Blender.

The two systems rhyme because they grew from the same decade. BI's hardness
IS a cosine exponent, which is exactly what the master shader's Glossiness
is; BI's diffuse/specular shader pair collapses onto one Halcyon model by
letting the more distinctive half win (Toon and Minnaert say more about a
material than Phong does). Influence factors travel as real graph structure:
a colour influence under 1.0 becomes a MixRGB with the base colour, in the
slot's own blend mode; a value influence becomes a Math multiply. Nothing
is silently rounded to "close enough" without a note in `warnings`.
"""

from . import blend279 as B

#: BI diffuse shader codes
DIFF_LAMBERT, DIFF_ORENNAYAR, DIFF_TOON, DIFF_MINNAERT, DIFF_FRESNEL = \
    0, 1, 2, 3, 4
#: BI specular shader codes
SPEC_COOKTORR, SPEC_PHONG, SPEC_BLINN, SPEC_TOON, SPEC_WARDISO = 0, 1, 2, 3, 4

#: MixRGB blend_type for MTex.blendtype
BLEND_TYPES = {0: 'MIX', 1: 'MULTIPLY', 2: 'ADD', 3: 'SUBTRACT',
               4: 'DIVIDE', 5: 'DARKEN', 6: 'DIFFERENCE', 7: 'LIGHTEN',
               8: 'SCREEN', 9: 'OVERLAY', 10: 'HUE', 11: 'SATURATION',
               12: 'VALUE', 13: 'COLOR', 14: 'SOFT_LIGHT',
               15: 'LINEAR_LIGHT'}

#: MTex.blendtype -> the BI Influence node's value-blend mode. HUE
#: through COLOR are absent on purpose: texture_value_blend left its
#: result at 0.0 for them (scalars have no hue)
MTEX_VALUE_BLENDS = {0: 'MIX', 1: 'MUL', 2: 'ADD', 3: 'SUB', 4: 'DIV',
                     5: 'DARK', 6: 'DIFF', 7: 'LIGHT', 8: 'SCREEN',
                     9: 'OVERLAY', 14: 'SOFT', 15: 'LINEAR'}

#: MTex.blendtype -> the BI Color Influence node's rgb-blend mode:
#: texture_rgb_blend's own switch, every case present (HUE..COLOR are
#: real for colours; SOFT/LINEAR delegate to ramp_blend as the C does)
MTEX_RGB_BLENDS = {0: 'MIX', 1: 'MUL', 2: 'ADD', 3: 'SUB', 4: 'DIV',
                   5: 'DARK', 6: 'DIFF', 7: 'LIGHT', 8: 'SCREEN',
                   9: 'OVERLAY', 10: 'HUE', 11: 'SAT', 12: 'VAL',
                   13: 'COLOR', 14: 'SOFT', 15: 'LINEAR'}

#: MTex.texflag bits (DNA_texture_types.h)
MTEX_RGBTOINT = 1
MTEX_STENCIL = 2
MTEX_NEGATIVE = 4
MTEX_ALPHAMIX = 8

#: Blend-texture stype -> Gradient (Shaped) settings. QUAD is the ramp
#: squared and EASE is a smoothstep, which are exactly the node's SHARP
#: and SMOOTH easings; DIAG is the linear ramp turned 45 degrees.
BLEND_STYPES = {0: ('LINEAR', 'NONE', 0.0),        # LIN
                1: ('LINEAR', 'SHARP', 0.0),       # QUAD
                2: ('LINEAR', 'SMOOTH', 0.0),      # EASE
                3: ('LINEAR', 'NONE', 0.7853982),  # DIAG
                4: ('SPHERICAL', 'NONE', 0.0),     # SPHERE
                5: ('QUADRATIC', 'NONE', 0.0),     # HALO
                6: ('CONICAL', 'NONE', 0.0)}       # RADIAL


#: BI integer codes -> node enum identifiers
BASIS_IDENTS = {0: 'BLENDER_ORIGINAL', 1: 'ORIGINAL_PERLIN',
                2: 'IMPROVED_PERLIN', 3: 'VORONOI_F1', 4: 'VORONOI_F2',
                5: 'VORONOI_F3', 6: 'VORONOI_F4', 7: 'VORONOI_F2F1',
                8: 'VORONOI_CRACKLE', 14: 'CELL_NOISE'}
WAVE_IDENTS = {0: 'SIN', 1: 'SAW', 2: 'TRI'}
WOOD_IDENTS = {0: 'BAND', 1: 'RING', 2: 'BANDNOISE', 3: 'RINGNOISE'}
MARBLE_IDENTS = {0: 'SOFT', 1: 'SHARP', 2: 'SHARPER'}
BLEND_IDENTS = {0: 'LIN', 1: 'QUAD', 2: 'EASE', 3: 'DIAG', 4: 'SPHERE',
                5: 'HALO', 6: 'RAD'}
STUCCI_IDENTS = {0: 'PLASTIC', 1: 'WALLIN', 2: 'WALLOUT'}
MUSGRAVE_IDENTS = {0: 'MFRACTAL', 1: 'RIDGEDMF', 2: 'HYBRIDMF', 3: 'FBM',
                   4: 'HTERRAIN'}
DISTM_IDENTS = {0: 'DISTANCE', 1: 'DISTANCE_SQUARED', 2: 'MANHATTAN',
                3: 'CHEBYCHEV', 4: 'MINKOVSKY_HALF', 5: 'MINKOVSKY_FOUR',
                6: 'MINKOVSKY'}
COLTYPE_IDENTS = {0: 'INTENSITY', 1: 'POSITION', 2: 'POSITION_OUTLINE',
                  3: 'POSITION_OUTLINE_INTENSITY'}
CB_IPO_IDENTS = {0: 'LINEAR', 1: 'EASE', 2: 'B_SPLINE', 3: 'CARDINAL',
                 4: 'CONSTANT'}

#: Tex.flag bits the node carries across
_TEX_FLIPBLEND, _TEX_COLORBAND, _TEX_NO_CLAMP = 2, 1, 1024


def _f(v, default=0.0):
    return default if v is None else float(v)


def _i(v, default=0):
    return default if v is None else int(v)


def pick_model(mat, warnings):
    """One Halcyon model from BI's diffuse/specular shader pair."""
    mode = _i(mat.get('mode'))
    if mode & B.MA_WIRE:
        return 'WIREFRAME'
    if mode & B.MA_SHLESS:
        return 'CONSTANT'
    diff = _i(mat.get('diff_shader'))
    spec = _i(mat.get('spec_shader'))
    spec_level = _f(mat.get('spec'), 0.5)
    if diff == DIFF_TOON or spec == SPEC_TOON:
        return 'TOON'
    if diff in (DIFF_MINNAERT, DIFF_ORENNAYAR):
        # BI shades diffuse and specular as independent terms; a Halcyon
        # model is a package. The matte models carry no highlight, so a
        # material that leans on its highlight keeps it and gives up the
        # subtler edge effect -- the visible term wins.
        matte = 'MINNAERT' if diff == DIFF_MINNAERT else 'OREN_NAYAR'
        if spec_level <= 0.15:
            return matte
        warnings.append(
            f"{mat.get('name', '?')}: {matte.replace('_', '-').title()} "
            f'diffuse with a strong highlight; the specular model keeps '
            'the highlight')
        return {SPEC_COOKTORR: 'BI_COOKTORR', SPEC_PHONG: 'BI_PHONG',
                SPEC_BLINN: 'BI_BLINN',
                SPEC_WARDISO: 'WARD'}.get(spec, 'BI_COOKTORR')
    if diff == DIFF_FRESNEL:
        # no Fresnel diffuse model; Lambert plus the Fresnel rim sockets
        # is the same visual idea (bright where glancing)
        warnings.append(f"{mat.get('name', '?')}: Fresnel diffuse "
                        'approximated as Lambert + Fresnel rim')
        return 'LAMBERT'
    # the BI-exact transcriptions: pow(N.H, hardness) family with
    # BI's own grazing and Fresnel terms, so Hardness finally
    # drives the highlight the way the original file was authored
    return {SPEC_COOKTORR: 'BI_COOKTORR', SPEC_PHONG: 'BI_PHONG',
            SPEC_BLINN: 'BI_BLINN',
            SPEC_WARDISO: 'WARD'}.get(spec, 'BI_COOKTORR')


def _texcoord(texco, warnings, name):
    """MTex.texco -> a coordinate route for the node build.

    'UV' | 'Generated' | 'Object' | 'Camera' route through a Texture
    Coordinate node; 'MATCAP_NORMAL' and 'MATCAP_REFLECT' route through
    the Matcap UV node -- BI's texco NOR (the classic matcap) and REFL
    (the env-map chrome trick), both a view-frame sphere projection.
    None means the node's own default (object space) already matches.
    """
    if texco is None:
        return None
    if texco & B.TEXCO_UV:
        return 'UV'
    if texco & B.TEXCO_REFL:
        return 'MATCAP_REFLECT'
    if texco & B.TEXCO_NORM:
        return 'MATCAP_NORMAL'
    if texco & B.TEXCO_ORCO:
        return 'Generated'
    if texco & B.TEXCO_OBJECT:
        return 'Object'
    if texco & B.TEXCO_GLOB:
        # world position; the Texture Coordinate node's Camera output is
        # eye-relative, Object is per-object -- neither is world. The
        # engine's Generated is the object's own box; Object space is
        # the nearest stable stand-in and matches for unmoved objects.
        warnings.append(f'{name}: Global texture coordinates approximated '
                        'with Object coordinates')
        return 'Object'
    if texco & B.TEXCO_WINDOW:
        return 'Window'
    warnings.append(f'{name}: texture coordinate mode 0x{texco:x} has no '
                    'counterpart; the slot uses object space')
    return None


def _vfac(slot, key):
    """A value-channel influence factor: the 2.5x per-channel field when
    the file has it, else the 2.4x catch-all varfac."""
    v = slot.get(key)
    if v is None:
        v = slot.get('varfac')
    return _f(v, 1.0)


def _tex_yields_rgb(slot, idname, props):
    """Whether this slot's texture returns COLOUR -- multitex's TEX_RGB
    bit, decided statically the way bitex.evaluate decides it: images
    and Magic always do, a colorband makes any procedural do, colour
    Clouds and colour Voronoi do."""
    if idname == 'ShaderNodeTexImage':
        return True
    if props.get('use_colorband'):
        return True
    tt = props.get('tex_type')
    if tt == 'MAGIC':
        return True
    if tt == 'CLOUDS' and props.get('clouds_color'):
        return True
    if tt == 'VORONOI' and props.get('vn_coltype', 'INTENSITY') \
            != 'INTENSITY':
        return True
    return False


#: Tex.imaflag / Tex.flag bits (DNA_texture_types.h)
TEX_USEALPHA = 2
TEX_CALCALPHA = 32
TEX_NEGALPHA = 4          # Tex.flag


def _slot_flags(slot, tex_rgb):
    """The MTex texflag bits the influence nodes carry, plus the image
    ALPHA LAW -- verbatim imagewrap (R158 fetch):

        talpha  only when TEX_USEALPHA (and not CALCALPHA)
        CALCALPHA: ta = max(r,g,b)
        neither:   ta = 1.0  (the file's alpha channel is IGNORED)
        TEX_NEGALPHA inverts ta

    Wiring the image's real alpha unconditionally made a
    streaks-on-transparency env map contribute NOTHING: its alpha is
    ~0 almost everywhere, and BI never read it (the Mask's invisible
    matcap)."""
    tf = _i(slot.get('texflag'))
    tex = slot.get('tex') or {}
    imaflag = _i(tex.get('imaflag'))
    texfl = _i(tex.get('flag'))
    is_img = tex.get('kind') == 'IMAGE'
    return {'tex_rgb': bool(tex_rgb),
            'rgbtoint': bool(tf & MTEX_RGBTOINT and tex_rgb),
            'negative': bool(tf & MTEX_NEGATIVE),
            'alphamix': bool(tf & MTEX_ALPHAMIX),
            # procedural colour (a colorband) always carries its band
            # alpha; images only under Use Alpha
            'img_alpha': (not is_img) or bool(
                (imaflag & TEX_USEALPHA)
                and not (imaflag & TEX_CALCALPHA)),
            'calc_alpha': bool(is_img and (imaflag & TEX_CALCALPHA)),
            'neg_alpha': bool(is_img and (texfl & TEX_NEGALPHA))}


def _tex_node(slot, warnings, mat_name):
    """One MTex slot's texture -> (node idname, props, inputs, output) or
    None when there is no counterpart.

    Every procedural becomes a HALCYON_BITextureNode carrying the Tex
    block's fields verbatim -- the original algorithms evaluate them, so
    a 1996 Clouds is THE 1996 Clouds. Images stay ShaderNodeTexImage.
    """
    tex = slot.get('tex') or {}
    kind = tex.get('kind', 'NONE')
    label = f"{mat_name}/{tex.get('name') or kind}"

    if kind == 'IMAGE':
        return ('ShaderNodeTexImage', {}, {}, 'Color')
    if kind == 'ENVMAP':
        warnings.append(f'{label}: EnvMap slots do not import; the '
                        "master shader's Reflection does that job now")
        return None
    if kind in ('PLUGIN', 'POINTDENSITY', 'VOXELDATA', 'OCEAN', 'NONE'):
        warnings.append(f'{label}: {kind} textures have no counterpart; '
                        'slot skipped')
        return None

    type_idents = {'CLOUDS': 'CLOUDS', 'WOOD': 'WOOD', 'MARBLE': 'MARBLE',
                   'MAGIC': 'MAGIC', 'BLEND': 'BLEND', 'STUCCI': 'STUCCI',
                   'NOISE': 'NOISE', 'MUSGRAVE': 'MUSGRAVE',
                   'VORONOI': 'VORONOI', 'DISTNOISE': 'DISTNOISE'}
    ident = type_idents.get(kind)
    if ident is None:
        warnings.append(f'{label}: {kind} textures have no counterpart; '
                        'slot skipped')
        return None

    nbas = _i(tex.get('noisebasis'))
    stype = _i(tex.get('stype'))
    flag = _i(tex.get('flag'))
    props = {
        'tex_type': ident,
        'noise_basis': BASIS_IDENTS.get(nbas, 'BLENDER_ORIGINAL'),
        'hard_noise': bool(_i(tex.get('noisetype'))),
        'noise_size': max(_f(tex.get('noisesize'), 0.25), 1e-4),
        'noise_depth': max(_i(tex.get('noisedepth'), 2), 0),
        'turbulence': _f(tex.get('turbul'), 5.0),
        'bright': _f(tex.get('bright'), 1.0),
        'contrast': _f(tex.get('contrast'), 1.0),
        'saturation': _f(tex.get('saturation'), 1.0),
        'rgb_factors': (_f(tex.get('rfac'), 1.0), _f(tex.get('gfac'), 1.0),
                        _f(tex.get('bfac'), 1.0)),
        'use_clamp': not (flag & _TEX_NO_CLAMP),
    }
    if ident == 'CLOUDS':
        props['clouds_color'] = stype == 1
    elif ident == 'WOOD':
        props['wood_type'] = WOOD_IDENTS.get(stype, 'BAND')
        props['wave'] = WAVE_IDENTS.get(_i(tex.get('noisebasis2')), 'SIN')
    elif ident == 'MARBLE':
        props['marble_type'] = MARBLE_IDENTS.get(stype, 'SOFT')
        props['wave'] = WAVE_IDENTS.get(_i(tex.get('noisebasis2')), 'SIN')
    elif ident == 'BLEND':
        props['blend_type'] = BLEND_IDENTS.get(stype, 'LIN')
        props['blend_flip'] = bool(flag & _TEX_FLIPBLEND)
    elif ident == 'STUCCI':
        props['stucci_type'] = STUCCI_IDENTS.get(stype, 'PLASTIC')
    elif ident == 'MUSGRAVE':
        props['musgrave_type'] = MUSGRAVE_IDENTS.get(stype, 'FBM')
        props['mg_h'] = _f(tex.get('mg_H'), 1.0)
        props['mg_lacunarity'] = _f(tex.get('mg_lacunarity'), 2.0)
        props['mg_octaves'] = _f(tex.get('mg_octaves'), 2.0)
        props['mg_offset'] = _f(tex.get('mg_offset'), 1.0)
        props['mg_gain'] = _f(tex.get('mg_gain'), 1.0)
        props['ns_outscale'] = _f(tex.get('ns_outscale'), 1.0)
    elif ident == 'VORONOI':
        props['vn_w1'] = _f(tex.get('vn_w1'), 1.0)
        props['vn_w2'] = _f(tex.get('vn_w2'), 0.0)
        props['vn_w3'] = _f(tex.get('vn_w3'), 0.0)
        props['vn_w4'] = _f(tex.get('vn_w4'), 0.0)
        props['vn_mexp'] = _f(tex.get('vn_mexp'), 2.5)
        props['vn_distm'] = DISTM_IDENTS.get(_i(tex.get('vn_distm')),
                                             'DISTANCE')
        props['vn_coltype'] = COLTYPE_IDENTS.get(_i(tex.get('vn_coltype')),
                                                 'INTENSITY')
        props['ns_outscale'] = _f(tex.get('ns_outscale'), 1.0)
    elif ident == 'DISTNOISE':
        props['noise_basis2'] = BASIS_IDENTS.get(
            _i(tex.get('noisebasis2')), 'BLENDER_ORIGINAL')
        props['dist_amount'] = _f(tex.get('dist_amount'), 1.0)

    cb = tex.get('colorband')
    if (flag & _TEX_COLORBAND) and cb and cb.get('stops'):
        props['use_colorband'] = True
        props['stops'] = list(cb['stops'])
        props['coba_ipotype'] = CB_IPO_IDENTS.get(cb.get('ipotype', 0),
                                                  'LINEAR')
        if cb.get('color_mode'):
            warnings.append(f'{label}: HSV/HSL colorband blending '
                            'approximated in RGB')

    # colour texture types drive colour channels from Color; intensity
    # ones expose Fac (evaluate() greys Color anyway)
    return ('HALCYON_BITextureNode', props, {}, 'Color')


#: (mapto bit, master socket, value-or-color, factor field)
_CHANNELS = (
    (B.MAP_COL, 'Diffuse Color', 'color', 'colfac'),
    (B.MAP_COLSPEC, 'Specular Color', 'color', 'colspecfac'),
    (B.MAP_COLMIR, 'Reflection Color', 'color', 'mirrfac'),
    (B.MAP_NORM, 'Bump Height', 'bump', 'norfac'),
    (B.MAP_REF, 'Diffuse Level', 'value', 'difffac'),
    (B.MAP_SPEC, 'Specular Level', 'value', 'specfac'),
    (B.MAP_EMIT, 'Self-Illumination', 'color', 'emitfac'),
    (B.MAP_ALPHA, 'Opacity', 'value', 'alphafac'),
    (B.MAP_HAR, 'Glossiness', 'value', 'hardfac'),
    (B.MAP_RAYMIRR, 'Reflection', 'value', 'raymirrfac'),
    (B.MAP_TRANSLU, 'Translucency', 'value', 'translfac'),
    (B.MAP_AMB, 'Ambient', 'value', 'ambfac'),
    # BI displacement has no Halcyon counterpart; its height feeds the
    # bump input instead, which keeps the detail visible
    (B.MAP_DISPLACE, 'Bump Height', 'bump', 'dispfac'),
)


#: BI shader codes -> the BI node's enum identifiers, DNA order
DIFF_NAMES = ('LAMBERT', 'OREN_NAYAR', 'TOON', 'MINNAERT', 'FRESNEL')
SPEC_NAMES = ('COOKTORR', 'PHONG', 'BLINN', 'TOON', 'WARDISO')
#: ramp enums, DNA order (material.c MA_RAMP_IN_* / MA_RAMP_*)
RAMP_INPUTS = ('SHADER', 'ENERGY', 'NORMAL', 'RESULT')
RAMP_BLENDS = ('MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV', 'DIFF',
               'DARK', 'LIGHT', 'OVERLAY', 'DODGE', 'BURN', 'HUE',
               'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR')


def _bi_ramp_props(mat, mode, props, warnings, name):
    """The two material ramps -> BI node props (only when BI showed
    them: the colorband can exist while the checkbox is off)."""
    for bit, prefix, band_key, in_key, bl_key, fac_key in (
            (B.MA_RAMP_COL, 'ramp_dif', 'ramp_col_band',
             'rampin_col', 'rampblend_col', 'rampfac_col'),
            (B.MA_RAMP_SPEC, 'ramp_spec', 'ramp_spec_band',
             'rampin_spec', 'rampblend_spec', 'rampfac_spec')):
        if not (mode & bit):
            continue
        band = mat.get(band_key)
        if not band or not band.get('stops'):
            warnings.append(f'{name}: a ramp is switched on but its '
                            'colorband is empty; ramp skipped')
            continue
        ri = _i(mat.get(in_key))
        rb = _i(mat.get(bl_key))
        props[f'use_{prefix}'] = True
        props[f'{prefix}_input'] = RAMP_INPUTS[ri] \
            if 0 <= ri < len(RAMP_INPUTS) else 'SHADER'
        props[f'{prefix}_blend'] = RAMP_BLENDS[rb] \
            if 0 <= rb < len(RAMP_BLENDS) else 'MIX'
        props[f'{prefix}_factor'] = max(0.0, min(
            _f(mat.get(fac_key), 1.0), 1.0))
        props[f'{prefix}_ipo'] = int(band.get('ipotype') or 0)
        props[f'{prefix}_stops'] = [tuple(s) for s in band['stops']]


def material_spec(mat, version=279):
    """A BI material dict -> a template spec plus its warnings.

    The spec is templates.build_spec() food: {'inputs', 'textures'} and
    -- since the BI node carries the whole 2.79 panel -- 'bi', the
    node's props. The one material the master shader still takes is
    Wire (the BI node has no wireframe). Inputs are keyed by the BI
    node's socket IDENTIFIERS, which are the master's names, so the
    texture machinery serves both.
    """
    warnings = []
    name = mat.get('name') or 'Material'
    mode = _i(mat.get('mode'))
    diff = _i(mat.get('diff_shader'))
    spec = _i(mat.get('spec_shader'))
    model = None
    bi = None

    r, g, b = _f(mat.get('r'), 0.8), _f(mat.get('g'), 0.8), \
        _f(mat.get('b'), 0.8)
    emit = _f(mat.get('emit'))
    param = mat.get('param') or (0.5, 0.1, 0.5, 0.1)

    inputs = {
        'Diffuse Color': (r, g, b, 1.0),
        'Diffuse Level': _f(mat.get('ref'), 0.8),
        'Specular Color': (_f(mat.get('specr'), 1.0),
                           _f(mat.get('specg'), 1.0),
                           _f(mat.get('specb'), 1.0), 1.0),
        'Specular Level': _f(mat.get('spec'), 0.5),
        'Glossiness': float(max(_i(mat.get('har'), 50), 1)),
        'Ambient': _f(mat.get('amb'), 1.0),
    }

    if mode & B.MA_WIRE:
        # the one look the BI node does not carry; the master's
        # WIREFRAME model is the same idea
        model = 'WIREFRAME'
        if emit > 0.0:
            inputs['Self-Illumination'] = (emit * r, emit * g,
                                           emit * b, 1.0)
    else:
        # ---- the BI node: every panel mapped 1:1, no approximations
        props = {
            'diff_shader': DIFF_NAMES[diff]
            if 0 <= diff < len(DIFF_NAMES) else 'LAMBERT',
            'spec_shader': SPEC_NAMES[spec]
            if 0 <= spec < len(SPEC_NAMES) else 'COOKTORR',
            'shadeless': bool(mode & B.MA_SHLESS),
            'use_cubic': bool(_i(mat.get('shade_flag')) & B.MA_CUBIC),
            'use_tangent_v': bool(mode & B.MA_TANGENT_V),
            'use_mist': not (mode & B.MA_NOMIST),
            'vcol_light': bool(mode & B.MA_VERTEXCOL),
            'vcol_paint': bool(mode & B.MA_VERTEXCOLP),
            'shadow_receive': bool(mode & B.MA_SHADOW),
            'shadow_cast': bool(_i(mat.get('mode2'), 1)
                                & B.MA_CASTSHADOW),
            'shadow_cast_only': bool(mode & B.MA_ONLYCAST),
            'shadow_only': bool(mode & B.MA_ONLYSHADOW),
        }
        # R164: the shadow-bias terminator fix (shade_one_light's
        # phongcorr), Ray Bias, and the object-colour modulation
        sbias = _f(mat.get('sbias'))
        if sbias != 0.0:
            props['sbias'] = sbias
        if mode & 0x400000:                     # MA_RAYBIAS
            props['raybias'] = True
        if _i(mat.get('shade_flag')) & 2:       # MA_OBCOLOR
            props['use_obcolor'] = True
        # ---- the Subsurface Scattering panel (MA_DIFF_SSS = 1 in
        # sss_flag): every field sss_create_tree_mat reads, verbatim
        if _i(mat.get('sss_flag')) & 1:
            rad3 = mat.get('sss_radius') or (1.0, 1.0, 1.0)
            col3 = mat.get('sss_col') or (1.0, 1.0, 1.0)
            props.update({
                'sss_enable': True,
                'sss_scale': _f(mat.get('sss_scale'), 0.1),
                'sss_radius': tuple(_f(v, 1.0) for v in rad3),
                'sss_color': tuple(_f(v, 1.0) for v in col3),
                'sss_ior': _f(mat.get('sss_ior'), 1.3),
                'sss_error': _f(mat.get('sss_error'), 0.05),
                'sss_colfac': _f(mat.get('sss_colfac'), 1.0),
                'sss_texfac': _f(mat.get('sss_texfac'), 0.0),
                'sss_front': _f(mat.get('sss_front'), 1.0),
                'sss_back': _f(mat.get('sss_back'), 1.0),
            })
        inputs['Emit'] = emit
        inputs['Translucency'] = _f(mat.get('translucency'))

        # the shader-specific sliders, RAW -- the node speaks BI units
        if diff == DIFF_ORENNAYAR:
            inputs['Roughness'] = _f(mat.get('roughness'), 0.5)
        elif diff == DIFF_MINNAERT:
            inputs['Darkness'] = _f(mat.get('darkness'), 1.0)
        elif diff == DIFF_TOON:
            inputs['Toon Size'] = _f(param[0], 0.5)
            inputs['Toon Smooth'] = _f(param[1], 0.1)
        elif diff == DIFF_FRESNEL:
            # fresnel_fac(lv, vn, param[0], param[1]): grad then power
            inputs['BI Fresnel'] = _f(param[0], 0.1)
            inputs['BI Fresnel Factor'] = _f(param[1], 0.5)
        if spec == SPEC_BLINN:
            inputs['IOR'] = max(_f(mat.get('refrac'), 4.0), 1.0)
        elif spec == SPEC_TOON:
            inputs['Spec Toon Size'] = _f(param[2], 0.5)
            inputs['Spec Toon Smooth'] = _f(param[3], 0.1)
        elif spec == SPEC_WARDISO:
            inputs['Slope'] = max(_f(mat.get('rms'), 0.1), 0.001)

        # ---- the Transparency panel: the CHECKBOX gates (2.5+ files;
        # phantom-class 2.79 saves carry MA_ZTRANSP even on opaque
        # defaults, so the method bit alone must NOT count), the method
        # bits pick the mode, sliders travel verbatim. Pre-2.5 files
        # had no checkbox: the method bit WAS the switch.
        try:
            vnum = int(version or 279)
        except (TypeError, ValueError):
            vnum = 279
        transp_on = bool(mode & (B.MA_TRANSP | B.MA_RAYTRANSP)) \
            if vnum >= 250 else \
            bool(mode & (B.MA_ZTRANSP | B.MA_RAYTRANSP))
        if transp_on:
            props['use_transparency'] = True
            props['transp_mode'] = 'RAYTRACE' \
                if mode & B.MA_RAYTRANSP else 'Z_TRANSPARENCY'
            inputs['Opacity'] = _f(mat.get('alpha'), 1.0)
            inputs['Transp Fresnel'] = _f(mat.get('fresnel_tra'))
            inputs['Transp Blend'] = _f(mat.get('fresnel_tra_i'), 1.25)
            inputs['Transp Specular'] = _f(mat.get('spectra'), 1.0)
            if mode & B.MA_RAYTRANSP:
                inputs['Ray IOR'] = max(_f(mat.get('ang'), 1.0), 1.0)
                inputs['Filter'] = max(0.0, min(
                    _f(mat.get('filter')), 1.0))

        # ---- the Mirror panel
        if mode & B.MA_RAYMIRROR:
            props['use_mirror'] = True
            inputs['Reflection'] = _f(mat.get('ray_mirror'))
            inputs['Reflection Color'] = (_f(mat.get('mirr'), 1.0),
                                          _f(mat.get('mirg'), 1.0),
                                          _f(mat.get('mirb'), 1.0), 1.0)
            inputs['Mirror Fresnel'] = _f(mat.get('fresnel_mir'))
            inputs['Mirror Blend'] = _f(mat.get('fresnel_mir_i'), 1.25)

        # ---- ramps and the light group
        _bi_ramp_props(mat, mode, props, warnings, name)
        group_lights = mat.get('group_lights')
        if mat.get('group_name'):
            props['light_group'] = mat['group_name']
            props['light_group_exclusive'] = bool(mode
                                                  & B.MA_GROUP_NOLAY)
            if group_lights:
                props['light_group_lights'] = list(group_lights)
            else:
                warnings.append(
                    f"{name}: light group '{mat['group_name']}' has no "
                    'lamps in the file; the material will receive NO '
                    'direct light')
        bi = props

    # ------------------------------------------------------- texture slots
    textures = []
    bump_used = False
    for slot in mat.get('slots', []):
        made = _tex_node(slot, warnings, name)
        if made is None:
            continue
        idname, props, tinputs, output = made
        mapto = _i(slot.get('mapto'))
        mapneg = _i(slot.get('maptoneg'))
        coords = _texcoord(slot.get('texco'), warnings, name)
        ofs = slot.get('ofs') or (0.0, 0.0, 0.0)
        size = slot.get('size') or (1.0, 1.0, 1.0)
        mapping = None
        if idname == 'HALCYON_BITextureNode':
            # the node applies size*(co+ofs) itself, in classic space --
            # the exact Map Input arithmetic, no Mapping node needed
            props['tex_offset'] = tuple(float(o) for o in ofs)
            props['tex_size'] = tuple(float(s) for s in size)
        elif any(abs(o) > 1e-6 for o in ofs) or \
                any(abs(s - 1.0) > 1e-6 for s in size):
            # verbatim texco_mapping's image branch (R158 fetch):
            #   texvec = size*(uv - 0.5) + ofs + 0.5
            # -- the offset is added UNSCALED in the half-centred 0..1
            # domain, so the Mapping fold is Scale = size and
            # Location = ofs + 0.5*(1 - size). The old fold (derived
            # from a recalled -1..1 chain) scaled the offset by
            # 0.5*size: every offset image slot sampled the wrong
            # window, the Mask's env-map matcap worst of all.
            mapping = {'location': tuple(
                float(ofs[i]) + 0.5 * (1.0 - float(size[i]))
                for i in range(3)),
                'scale': tuple(float(s) for s in size)}
        tex = slot.get('tex') or {}
        pxa = (slot.get('projx'), slot.get('projy'), slot.get('projz'))
        if pxa not in ((None, None, None), (1, 2, 3)):
            warnings.append(
                f'{name}: a slot remaps its projection axes to '
                f'{pxa}; that swizzle is not imported and the slot '
                'samples straight X/Y/Z')
        image = None
        if tex.get('kind') == 'IMAGE':
            image = {'path': tex.get('image_path') or '',
                     'name': tex.get('image_name') or 'image',
                     'packed': tex.get('packed')}
            if not image['path'] and not image['packed']:
                # BI adds NOTHING for an imageless texture slot -- the
                # texture returns zero intensity and the channel stays
                # untouched. Emitting a black-sampling node instead
                # darkened the colour chain AND refused the GPU frame
                # plan (one refused material sends the WHOLE frame to
                # the CPU) -- both halves of a real field crash.
                warnings.append(f'{name}: image slot '
                                f'{tex.get("name") or "?"} has no file '
                                'path and no packed data; skipped, as '
                                'Blender Internal adds nothing for an '
                                'imageless texture')
                continue

        routed = False
        tex_rgb = _tex_yields_rgb(slot, idname, props)
        flags = _slot_flags(slot, tex_rgb)
        if _i(slot.get('texflag')) & MTEX_STENCIL:
            warnings.append(
                f'{name}: a slot sets Stencil, which gates every LATER '
                'slot through this one; that chaining is not imported '
                'yet, so the later slots apply at full strength')
        for bit, target, style, fkey in _CHANNELS:
            if not (mapto & bit):
                continue
            if bi is not None and target == 'Self-Illumination':
                # the BI node's Emit is BI's own float slider; emit
                # textures scale it as a VALUE, not a colour synth
                target, style = 'Emit', 'value'
            routed = True
            entry = {'node': idname, 'props': dict(props),
                     'inputs': dict(tinputs), 'output': output,
                     'target': target, 'style': style}
            if coords:
                entry['coords'] = coords
            if mapping:
                entry['mapping'] = dict(mapping)
            if image:
                entry['image'] = dict(image)
            if slot.get('uvname'):
                entry['uv_layer'] = slot['uvname']
            if mapneg & bit:
                entry['invert'] = True
            if style == 'color' and bi is not None:
                # a BI colour channel is texture_rgb_blend: the base
                # blends toward tcol by the PER-PIXEL factor (intensity
                # or texture alpha) times the slider. tcol is the
                # texture's colour only when it yields one; an
                # intensity texture contributes the SLOT colour -- the
                # Influence panel's swatch, pink by default -- which no
                # import carried before
                entry['rgbblend'] = dict(
                    flags,
                    blend=MTEX_RGB_BLENDS.get(
                        _i(slot.get('blendtype')), 'MIX'),
                    factor=max(0.0, min(_f(slot.get(fkey), 1.0), 1.0)),
                    slot_color=(_f(slot.get('r'), 1.0),
                                _f(slot.get('g'), 0.0),
                                _f(slot.get('b'), 1.0)),
                    map_alpha=bool(mapto & B.MAP_ALPHA),
                    base=tuple(inputs.get(target))
                    if inputs.get(target) is not None
                    else (0.8, 0.8, 0.8, 1.0))
            elif style == 'color':
                fac = max(0.0, min(_vfac(slot, fkey), 1.0))
                base = inputs.get(target)
                if fac < 1.0 - 1e-6 or \
                        _i(slot.get('blendtype')) != 0:
                    entry['mix'] = {
                        'fac': fac,
                        'blend': BLEND_TYPES.get(
                            _i(slot.get('blendtype')), 'MIX'),
                        'base': tuple(base) if base is not None
                        else (0.8, 0.8, 0.8, 1.0)}
            elif style == 'value':
                entry['output'] = 'Fac' if idname.startswith('HALCYON_') \
                    else output
                if bi is not None:
                    # BI's value channel is texture_value_blend: the
                    # base value blends TOWARD DVar by intensity x the
                    # SIGNED factor, in the slot's blend mode -- not
                    # "texture becomes the value". Hardness runs it in
                    # /128 units, exactly do_material_tex
                    bt = _i(slot.get('blendtype'))
                    vb_mode = MTEX_VALUE_BLENDS.get(bt)
                    if vb_mode is None:
                        warnings.append(
                            f'{name}: blend type {bt} on a value '
                            'channel produced ZERO in BI; imported as '
                            'Mix instead')
                        vb_mode = 'MIX'
                    base_v = inputs.get(target)
                    if base_v is None:
                        base_v = {'Emit': 0.0, 'Opacity': 1.0,
                                  'Ambient': 1.0, 'Reflection': 0.0,
                                  'Translucency': 0.0,
                                  'Diffuse Level': 0.8,
                                  'Specular Level': 0.5,
                                  'Glossiness': 50.0}.get(target, 0.0)
                    base_v = float(base_v)
                    vb = dict(flags, blend=vb_mode,
                              factor=_f(slot.get(fkey),
                                        _f(slot.get('varfac'), 1.0)),
                              dvar=_f(slot.get('def_var'), 1.0),
                              base=base_v)
                    if target == 'Glossiness':
                        vb['base'] = base_v / 128.0
                        vb['scale'] = 128.0
                        vb['clamp'] = (1.0, 511.0)
                    entry['vblend'] = vb
                else:
                    fac = max(0.0, min(_vfac(slot, fkey), 1.0))
                    if fac < 1.0 - 1e-6:
                        entry['scale_fac'] = fac
            elif style == 'bump':
                entry['output'] = 'Fac' if idname.startswith('HALCYON_') \
                    else output
                fac = abs(_vfac(slot, fkey)) if fkey != 'norfac' \
                    else abs(_f(slot.get('norfac'),
                               slot.get('varfac') or 0.5))
                if bump_used:
                    # several height sources: each carries its own
                    # strength and build_spec sums them into the one
                    # Bump Height input
                    entry['scale_fac'] = fac
                else:
                    bump_used = True
                    if abs(fac - 1.0) > 1e-6:
                        entry['scale_fac'] = fac
                    inputs['Bump Strength'] = 1.0
            textures.append(entry)
        if not routed:
            warnings.append(f'{name}: a texture slot maps to no channel '
                            'Halcyon shades with; slot skipped')

    return {'name': name, 'model': model, 'bi': bi, 'inputs': inputs,
            'textures': textures, 'warnings': warnings}


#: Lamp.mode bits (DNA_lamp_types.h)
LA_SHAD_BUF = 0x1
LA_LAYER = 0x4
LA_NEG = 0x10
LA_ONLYSHADOW = 0x20
LA_SPHERE = 0x40
LA_SQUARE = 0x80
LA_NO_DIFF = 0x800
LA_SHAD_RAY = 0x2000
LA_NO_SPEC = 0x4000


def lamp_map(la, version=279):
    """BI lamp dict -> plain settings for the operator to apply."""
    kind = la.get('kind', 'POINT')
    mode = _i(la.get('mode'))
    out = {'name': la.get('name') or 'Lamp',
           'type': {'POINT': 'POINT', 'SUN': 'SUN', 'SPOT': 'SPOT',
                    'HEMI': 'HEMI', 'AREA': 'AREA'}.get(kind, 'POINT'),
           'color': (_f(la.get('r'), 1.0), _f(la.get('g'), 1.0),
                     _f(la.get('b'), 1.0)),
           'energy': _f(la.get('energy'), 1.0),
           'distance': _f(la.get('dist'), 25.0),
           # BI falloff, verbatim lamp_get_visibility (R155):
           # 0 constant, 1 inverse linear D/(D+d), 2 inverse square
           # D/(D+d*d), 4 the Lin/Quad sliders. CURVE (3) and inverse
           # coefficients (5) approximate as inverse linear, loudly
           'falloff': {0: 'CONSTANT', 1: 'INVERSE_LINEAR',
                       2: 'INVERSE_SQUARE', 4: 'SLIDERS'}.get(
                           _i(la.get('falloff_type'), 1),
                           'INVERSE_LINEAR'),
           'ld1': _f(la.get('att1'), 0.0),
           'ld2': _f(la.get('att2'), 0.0),
           # the mode bits BI's light loop honours per lamp
           'sphere': bool(mode & LA_SPHERE),
           'negative': bool(mode & LA_NEG),
           'no_diffuse': bool(mode & LA_NO_DIFF),
           'no_specular': bool(mode & LA_NO_SPEC),
           # 2.79's shadow rule, verbatim from shade_one_light +
           # convertblender: a buffer (shb) exists ONLY for a SPOT
           # with the buffer bit, and it wins; otherwise the Ray bit
           # traces; otherwise the lamp casts NO shadow at all. Both
           # bits set at once is common in 2.4x-era files -- the spot
           # keeps its buffer, everything else rides the ray bit.
           'shadow': ('MAP' if kind == 'SPOT' and (mode & LA_SHAD_BUF)
                      else 'RAY' if mode & LA_SHAD_RAY else 'NONE'),
           # the lamp's SHADOW COLOUR: lashdw tints the shadowed part
           # of the diffuse (shade_one_light adds
           # lashdw*(i_noshad - i)*lacol back). Black = classic.
           'shadow_color': (_f(la.get('shdwr')), _f(la.get('shdwg')),
                            _f(la.get('shdwb'))),
           'warnings': []}
    if _i(la.get('falloff_type'), 1) not in (0, 1, 2, 4) and \
            kind not in ('SUN', 'HEMI'):
        out['warnings'].append(
            f"{out['name']}: falloff type {la.get('falloff_type')} "
            'approximated as Inverse Linear')
    if mode & LA_ONLYSHADOW:
        out['warnings'].append(
            f"{out['name']}: Only Shadow lamps (shadow subtraction "
            'without light) are not imported yet; the lamp lights '
            'normally')
    if mode & LA_LAYER:
        out['warnings'].append(
            f"{out['name']}: the This Layer Only restriction is not "
            'imported; the lamp lights every layer')
    if kind == 'SPOT' and (mode & LA_SQUARE):
        out['warnings'].append(
            f"{out['name']}: square spot cones are not imported; the "
            'cone imports round')
    if kind == 'SPOT':
        size = _f(la.get('spotsize'), 45.0)
        if version < 270:
            size = size * 3.14159265 / 180.0   # degrees before 2.70
        out['spot_size'] = size
        out['spot_blend'] = _f(la.get('spotblend'), 0.15)
    if kind == 'AREA':
        out['area_size'] = _f(la.get('area_size'), 1.0)
        out['area_size_y'] = _f(la.get('area_sizey'),
                                _f(la.get('area_size'), 1.0))
    return out


def world_map(wo, warnings=None):
    """BI world dict -> Halcyon world settings (a horizon/zenith
    gradient, ambient colour, exposure, and mist when it was on)."""
    if not wo:
        return None
    out = {'horizon': (_f(wo.get('horr'), 0.05), _f(wo.get('horg'), 0.05),
                       _f(wo.get('horb'), 0.05)),
           'zenith': (_f(wo.get('zenr'), 0.0), _f(wo.get('zeng'), 0.0),
                      _f(wo.get('zenb'), 0.0)),
           'ambient': (_f(wo.get('ambr'), 0.0), _f(wo.get('ambg'), 0.0),
                       _f(wo.get('ambb'), 0.0)),
           'blend': bool(_i(wo.get('skytype')) & 1),   # WO_SKYBLEND
           # R164: the Exposure panel (wrld_exposure_correct's inputs;
           # exp=0, range=1 is the identity and the default)
           'exposure': _f(wo.get('exp'), 0.0),
           'exposure_range': _f(wo.get('range'), 1.0)}
    if _f(wo.get('misi')) > 0.0:
        # mistfactor's own curves: 0 quadratic (fac*fac), 1 linear,
        # 2 inverse quadratic (sqrt)
        out['mist'] = {'intensity': _f(wo.get('misi')),
                       'start': _f(wo.get('miststa'), 5.0),
                       'depth': _f(wo.get('mistdist'), 25.0),
                       'falloff': {0: 'QUADRATIC', 1: 'LINEAR',
                                   2: 'INVERSE_QUADRATIC'}.get(
                                       _i(wo.get('mistype')), 'LINEAR')}
        if warnings is not None and _f(wo.get('misthi')) != 0.0:
            warnings.append(
                'World mist Height falloff is not imported; mist '
                'covers all heights')
    return out


def scene_settings_map(sdata):
    """The file's OWN pipeline -> Halcyon render settings.

    2.79 (any 2.5+) renders Blender Internal scene-linear and shows it
    through the scene's OCIO view; 'Default' on an sRGB display is the
    piecewise sRGB encode. A Halcyon frame carries the same arithmetic,
    so matching the user's F12 means running the same two ends of the
    pipeline: linearize sRGB-tagged textures on the way IN
    (input_gamma_naive False) and apply the sRGB curve on the way OUT
    (color_management SRGB). Material, lamp and slot colours are
    ALREADY linear in the DNA -- they pass through untouched, which is
    why only the two ends move.

    Pre-2.5 files (scene_cm None) had no OCIO: bytes in, bytes out --
    Halcyon's period-correct NONE + naive-input mode IS that pipeline.

    Returns (settings_dict, warnings). settings_dict keys:
    color_management, input_gamma_naive, exposure, gamma, and (when the
    file carries them) res_x, res_y, res_pct, film_transparent.
    """
    warnings = []
    cm = (sdata or {}).get('scene_cm')
    out = {}
    if cm is None:
        out['color_management'] = 'NONE'
        out['input_gamma_naive'] = True
    else:
        device = str(cm.get('display_device') or 'sRGB')
        view = str(cm.get('view_transform') or 'Default')
        if device == 'None':
            # the author turned color management off: bytes straight
            # through, both ends
            out['color_management'] = 'NONE'
            out['input_gamma_naive'] = True
        elif view in ('Default', 'Standard', 'sRGB'):
            out['color_management'] = 'SRGB'
            out['input_gamma_naive'] = False
        elif view in ('Raw', 'None'):
            # linear render shown raw -- textures still linearized
            out['color_management'] = 'NONE'
            out['input_gamma_naive'] = False
        else:
            warnings.append(
                f"Scene view transform '{view}' is not implemented; "
                "rendering with the sRGB display encode instead")
            out['color_management'] = 'SRGB'
            out['input_gamma_naive'] = False
        # OCIO exposure is in STOPS (x2^e, before the curve); Halcyon's
        # exposure multiplies before the curve too
        e = _f(cm.get('exposure'), 0.0)
        out['exposure'] = float(2.0 ** e) if abs(e) > 1e-6 else 1.0
        g = _f(cm.get('gamma'), 1.0)
        out['gamma'] = g if g > 1e-3 else 1.0
        look = str(cm.get('look') or 'None')
        if look not in ('None', ''):
            warnings.append(f"Scene look '{look}' is not implemented "
                            "and was dropped")
    rd = (sdata or {}).get('scene_render')
    if rd:
        if rd.get('xsch') and rd.get('ysch'):
            out['res_x'] = int(rd['xsch'])
            out['res_y'] = int(rd['ysch'])
            out['res_pct'] = int(rd.get('size') or 100)
        # R_ALPHAPREMUL: the F12 sky is transparent
        if rd.get('alphamode') is not None:
            out['film_transparent'] = (int(rd['alphamode']) == 1)
    return out, warnings
