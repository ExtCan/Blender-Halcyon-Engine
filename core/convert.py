"""Deciding how to convert a Blender material to the Halcyon Shader.

The *decision* lives here and imports nothing from bpy, so the mapping tables
and the model choice can be tested directly. The node surgery that acts on the
decision is in halcyon/convert.py.

Socket names are given as alias lists because Blender renamed several Principled
inputs in 4.0 -- 'Specular' became 'Specular IOR Level', 'Emission' became
'Emission Color', 'Transmission' became 'Transmission Weight'. Accepting both
means one code path covers 3.x through 5.x.
"""

MASTER_NODE = 'HALCYON_ShaderNode'

# (halcyon socket, [source socket aliases])
PRINCIPLED = [
    ('Diffuse Color', ['Base Color']),
    ('Metalness', ['Metallic']),
    ('Roughness', ['Roughness']),
    ('Specular Level', ['Specular IOR Level', 'Specular']),
    ('Self-Illumination', ['Emission Color', 'Emission']),
    ('Opacity', ['Alpha']),
    ('IOR', ['IOR']),
    ('Anisotropy', ['Anisotropic']),
    ('Anisotropic Rotation', ['Anisotropic Rotation']),
    ('Normal', ['Normal']),
]

DIFFUSE = [
    ('Diffuse Color', ['Color']),
    ('Roughness', ['Roughness']),
    ('Normal', ['Normal']),
]

GLOSSY = [
    ('Specular Color', ['Color']),
    ('Roughness', ['Roughness']),
    ('Anisotropy', ['Anisotropy']),
    ('Anisotropic Rotation', ['Rotation']),
    ('Normal', ['Normal']),
]

EMISSION = [
    ('Self-Illumination', ['Color']),
    ('Diffuse Color', ['Color']),
]

GLASS = [
    ('Specular Color', ['Color']),
    ('Roughness', ['Roughness']),
    ('IOR', ['IOR']),
    ('Normal', ['Normal']),
]

TOON = [
    ('Diffuse Color', ['Color']),
    ('Toon Size', ['Size']),
    ('Toon Smooth', ['Smooth']),
    ('Normal', ['Normal']),
]

TRANSLUCENT = [
    ('Diffuse Color', ['Color']),
    ('Normal', ['Normal']),
]

SHEEN = [
    ('Diffuse Color', ['Color']),
    ('Roughness', ['Roughness']),
    ('Normal', ['Normal']),
]

SUBSURFACE = [
    ('Diffuse Color', ['Color']),
    ('Translucency', ['Scale']),
    ('Normal', ['Normal']),
]

# EEVEE's Specular BSDF is the spec/gloss workflow -- the DirectX-era
# model this engine wears natively, so the mapping is nearly literal
SPECULAR = [
    ('Diffuse Color', ['Base Color']),
    ('Specular Color', ['Specular']),
    ('Roughness', ['Roughness']),
    ('Self-Illumination', ['Emissive Color']),
    ('Normal', ['Normal']),
]

SOURCES = {
    'ShaderNodeBsdfPrincipled': PRINCIPLED,
    'ShaderNodeBsdfDiffuse': DIFFUSE,
    'ShaderNodeBsdfGlossy': GLOSSY,
    'ShaderNodeBsdfAnisotropic': GLOSSY,
    'ShaderNodeBsdfMetallic': GLOSSY,
    'ShaderNodeEmission': EMISSION,
    'ShaderNodeBsdfGlass': GLASS,
    'ShaderNodeBsdfRefraction': GLASS,
    'ShaderNodeBsdfToon': TOON,
    'ShaderNodeBsdfTranslucent': TRANSLUCENT,
    'ShaderNodeBsdfSheen': SHEEN,
    'ShaderNodeBsdfVelvet': SHEEN,
    'ShaderNodeSubsurfaceScattering': SUBSURFACE,
    'ShaderNodeEeveeSpecular': SPECULAR,
}

# constants applied after the socket mapping, for sources whose character is not
# carried by any single socket
EXTRAS = {
    'ShaderNodeEmission': {'Diffuse Level': 0.0, 'Specular Level': 0.0},
    'ShaderNodeBsdfGlass': {'Reflection': 0.6, 'Opacity': 0.25},
    'ShaderNodeBsdfRefraction': {'Reflection': 0.3, 'Opacity': 0.2},
    'ShaderNodeBsdfTranslucent': {'Translucency': 1.0, 'Specular Level': 0.0},
    'ShaderNodeBsdfDiffuse': {'Specular Level': 0.0},
    'ShaderNodeBsdfGlossy': {'Diffuse Level': 0.1, 'Specular Level': 1.0},
    'ShaderNodeBsdfMetallic': {'Diffuse Level': 0.0, 'Specular Level': 1.0,
                               'Metalness': 1.0},
    'ShaderNodeSubsurfaceScattering': {'Specular Level': 0.1},
    'ShaderNodeEeveeSpecular': {'Specular Level': 1.0},
}


def glossiness_from_roughness(r):
    """The classic roughness-to-exponent mapping, matching the renderer's own."""
    r = max(min(float(r), 1.0), 0.0)
    r4 = max(r * r * r * r, 1e-5)
    return max(min(2.0 / r4 - 2.0, 8192.0), 0.5)


def choose_model(idname, values=None, links=None):
    """Pick the reflectance model that best carries the source shader over.

    Values are whatever constants the source had; links is the set of socket
    names that were driven by other nodes, because a linked Metallic or
    Anisotropic means the parameter matters even when its constant reads zero.
    """
    values = values or {}
    links = set(links or ())

    def val(*names, default=0.0):
        for n in names:
            if n in values:
                try:
                    v = values[n]
                    return float(v[0]) if hasattr(v, '__len__') else float(v)
                except (TypeError, ValueError):
                    return default
        return default

    if idname == 'ShaderNodeEmission':
        return 'CONSTANT'
    if idname == 'ShaderNodeBsdfToon':
        return 'TOON'
    if idname == 'ShaderNodeBsdfTranslucent':
        return 'TRANSLUCENT'
    if idname in ('ShaderNodeBsdfGlass', 'ShaderNodeBsdfRefraction'):
        return 'BLINN'
    if idname in ('ShaderNodeBsdfSheen', 'ShaderNodeBsdfVelvet'):
        return 'MINNAERT'
    if idname == 'ShaderNodeSubsurfaceScattering':
        return 'TRANSLUCENT'
    if idname == 'ShaderNodeBsdfMetallic':
        return 'METAL'
    if idname == 'ShaderNodeEeveeSpecular':
        return 'BLINN_PHONG'

    aniso = abs(val('Anisotropic', 'Anisotropy'))
    if aniso > 0.01 or 'Anisotropic' in links or 'Anisotropy' in links:
        return 'ANISOTROPIC'

    metal = val('Metallic')
    if metal > 0.5 or 'Metallic' in links:
        return 'METAL'

    if idname == 'ShaderNodeBsdfDiffuse':
        return 'OREN_NAYAR' if val('Roughness') > 0.3 else 'LAMBERT'
    if idname in ('ShaderNodeBsdfGlossy', 'ShaderNodeBsdfAnisotropic'):
        return 'COOK_TORRANCE' if val('Roughness') > 0.25 else 'PHONG'

    if idname == 'ShaderNodeBsdfPrincipled':
        rough = val('Roughness', default=0.5)
        if rough > 0.6:
            return 'OREN_NAYAR'
        return 'BLINN_PHONG'
    return 'PHONG'


def plan(idname, values=None, links=None, model='AUTO'):
    """Work out the conversion without touching any Blender data.

    Returns a dict with the chosen model, the socket pairs to relink or copy,
    the constants to apply afterwards, and any notes worth reporting.
    """
    values = values or {}
    links = set(links or ())
    table = SOURCES.get(idname)
    notes = []
    if table is None:
        notes.append(f"{idname} has no direct equivalent; "
                     "colour and normal carried over where present")
        table = [('Diffuse Color', ['Color', 'Base Color']),
                 ('Normal', ['Normal'])]

    pairs = []
    for target, aliases in table:
        for alias in aliases:
            if alias in links or alias in values:
                pairs.append((target, alias))
                break

    extras = dict(EXTRAS.get(idname, {}))
    # a roughness constant also sets the specular exponent, which is the
    # parameter the period models actually shade with
    for _t, alias in pairs:
        if alias == 'Roughness' and 'Roughness' not in links:
            try:
                r = values.get('Roughness', 0.5)
                r = float(r[0]) if hasattr(r, '__len__') else float(r)
                extras['Glossiness'] = glossiness_from_roughness(r)
            except (TypeError, ValueError):
                pass
            break

    scale_links = {}
    if idname == 'ShaderNodeBsdfPrincipled':
        try:
            t = values.get('Transmission Weight', values.get('Transmission', 0.0))
            t = float(t[0]) if hasattr(t, '__len__') else float(t)
            if t > 0.01:
                extras['Opacity'] = max(0.0, 1.0 - t)
                extras['Reflection'] = min(1.0, t * 0.6)
                notes.append("transmission mapped to opacity and reflection")
        except (TypeError, ValueError):
            pass

    if idname in ('ShaderNodeBsdfPrincipled', 'ShaderNodeEmission'):
        # Emission Strength actually FOLDS now. The old code noted "strength
        # folded into the colour" and then copied Emission Color alone --
        # and since 4.0 Principled defaults to a WHITE emission colour at
        # strength ZERO, every default material converted to a self-lit
        # white one. This was the bright-white converter bug.
        pairs, extras2, more, sl = _fold_emission(idname, values, links, pairs)
        extras.update(extras2)
        notes.extend(more)
        scale_links.update(sl)

    if idname == 'ShaderNodeEeveeSpecular':
        try:
            t = values.get('Transparency', 0.0)
            t = float(t[0]) if hasattr(t, '__len__') else float(t)
            if t > 0.01:
                extras['Opacity'] = max(0.0, 1.0 - t)
                notes.append("transparency mapped to opacity")
        except (TypeError, ValueError):
            pass

    chosen = choose_model(idname, values, links) if model == 'AUTO' else model
    return {'model': chosen, 'pairs': pairs, 'extras': extras, 'notes': notes,
            'scale_links': scale_links, 'source': idname}


#: which socket carries the emission colour / strength, per source node
_EMIT_COLOR = {'ShaderNodeBsdfPrincipled': ('Emission Color', 'Emission'),
               'ShaderNodeEmission': ('Color',)}
_EMIT_STRENGTH = {'ShaderNodeBsdfPrincipled': ('Emission Strength',),
                  'ShaderNodeEmission': ('Strength',)}


def _fold_emission(idname, values, links, pairs):
    """Fold Emission Strength into Self-Illumination, for real.

    Four cases, each exact:
    - strength constant 0: emission is OFF. The colour -- whatever it says,
      and since Blender 4.0 it says white by default -- must not be carried.
    - strength constant, colour constant: the product is the emission.
    - strength constant (not 1), colour LINKED: the link is kept and marked
      for a multiply node so the strength still applies.
    - strength LINKED: the link pair is kept and marked for a multiply node
      fed by the strength link itself.
    Returns (pairs, extras, notes, scale_links).
    """
    extras, notes, scale_links = {}, [], {}
    col_aliases = _EMIT_COLOR.get(idname, ())
    str_aliases = _EMIT_STRENGTH.get(idname, ())
    emit_pairs = [(t, a) for (t, a) in pairs
                  if t == 'Self-Illumination' and a in col_aliases]
    if not emit_pairs:
        return pairs, extras, notes, scale_links

    s_alias = next((a for a in str_aliases if a in links), None)
    if s_alias is not None:
        # dynamic strength: keep the colour link and multiply by the
        # strength link at the material side
        scale_links['Self-Illumination'] = ('LINK', s_alias)
        notes.append("emission strength is node-driven; a multiply node "
                     "carries it")
        return pairs, extras, notes, scale_links

    s = 1.0
    for a in str_aliases:
        if a in values:
            try:
                v = values[a]
                s = float(v[0]) if hasattr(v, '__len__') else float(v)
            except (TypeError, ValueError):
                s = 1.0
            break

    col_alias = next((a for (t, a) in emit_pairs), None)
    col_linked = col_alias in links if col_alias is not None else False

    if s <= 0.0:
        # emission off: drop the colour entirely, however white it is
        pairs = [(t, a) for (t, a) in pairs
                 if not (t == 'Self-Illumination' and a in col_aliases)]
        extras['Self-Illumination'] = (0.0, 0.0, 0.0, 1.0)
        if col_linked or _nonblack(values.get(col_alias)):
            notes.append("emission strength is 0; the emission colour "
                         "was dropped")
        return pairs, extras, notes, scale_links

    if col_linked:
        if abs(s - 1.0) > 1e-6:
            scale_links['Self-Illumination'] = ('VALUE', s)
            notes.append(f"emission strength {s:g} carried by a multiply "
                         "node")
        return pairs, extras, notes, scale_links

    c = values.get(col_alias)
    if c is not None and abs(s - 1.0) > 1e-6:
        try:
            cc = [float(x) for x in (c if hasattr(c, '__len__') else (c,) * 3)]
            while len(cc) < 3:
                cc.append(cc[-1])
            pairs = [(t, a) for (t, a) in pairs
                     if not (t == 'Self-Illumination' and a in col_aliases)]
            extras['Self-Illumination'] = (cc[0] * s, cc[1] * s, cc[2] * s,
                                           1.0)
            notes.append(f"emission strength {s:g} folded into the colour")
        except (TypeError, ValueError):
            pass
    return pairs, extras, notes, scale_links


def _nonblack(c):
    try:
        if c is None:
            return False
        if hasattr(c, '__len__'):
            return any(float(x) > 1e-6 for x in list(c)[:3])
        return float(c) > 1e-6
    except (TypeError, ValueError):
        return False


# ------------------------------------------------- conversion TO BI material

BI_NODE = 'HALCYON_BIMaterialNode'


def _f1(values, *names, default=0.0):
    for n in names:
        if n in values:
            try:
                v = values[n]
                return float(v[0]) if hasattr(v, '__len__') else float(v)
            except (TypeError, ValueError):
                return default
    return default


def _c3(values, *names, default=None):
    for n in names:
        if n in values:
            v = values[n]
            try:
                if hasattr(v, '__len__'):
                    cc = [float(x) for x in list(v)[:3]]
                    while len(cc) < 3:
                        cc.append(cc[-1])
                    return tuple(cc)
                return (float(v),) * 3
            except (TypeError, ValueError):
                return default
    return default


def bi_plan(idname, values=None, links=None):
    """How a Blender shader maps onto the BI material node. bpy-free.

    Returns {'props': node properties, 'sockets': {BI socket name: value},
    'links': [(BI socket name, source socket alias)], 'notes': [...]}.
    The mapping follows what 2.79's own importer would have done in
    reverse: hardness from roughness through the renderer's own curve
    (clamped to BI's 1..511), emission through the Emit float against
    the diffuse chain, alpha through the transparency panel, and an
    Emission shader becoming a Shadeless material -- which is exactly
    what Shadeless was for.
    """
    values = values or {}
    links = set(links or ())
    props = {'diff_shader': 'LAMBERT', 'spec_shader': 'COOKTORR'}
    sockets = {}
    out_links = []
    notes = []

    def link_or_value(bi_name, aliases, value=None):
        for a in aliases:
            if a in links:
                out_links.append((bi_name, a))
                return True
        if value is not None:
            sockets[bi_name] = value
        return False

    if idname == 'ShaderNodeEmission':
        props['shadeless'] = True
        c = _c3(values, 'Color', default=(0.8, 0.8, 0.8))
        s = _f1(values, 'Strength', default=1.0)
        linked = link_or_value('Color', ['Color'],
                               tuple(min(x * s, 1.0) for x in c) + (1.0,))
        if linked and abs(s - 1.0) > 1e-6:
            notes.append(f"emission strength {s:g} cannot scale a linked "
                         "colour on a shadeless material")
        elif not linked and s > 1.0:
            notes.append("shadeless clamps at 1; emission brighter than "
                         "that is a light, not a material")
        notes.append("Emission became Shadeless -- BI's flat, unlit colour")
        return {'props': props, 'sockets': sockets, 'links': out_links,
                'notes': notes}

    base_aliases = ['Base Color', 'Color']
    base = _c3(values, *base_aliases, default=(0.8, 0.8, 0.8))
    link_or_value('Color', base_aliases, tuple(base) + (1.0,))
    sockets['Intensity'] = 1.0   # Principled's diffuse is Color x 1

    if idname in ('ShaderNodeBsdfGlossy', 'ShaderNodeBsdfAnisotropic',
                  'ShaderNodeBsdfMetallic'):
        # a pure mirror shader: all highlight, no diffuse
        sockets['Intensity'] = 0.0
        sockets['Specular Color'] = tuple(base) + (1.0,)
        sockets['Specular Intensity'] = 1.0
        notes.append("glossy source: diffuse off, highlight carries the "
                     "colour")
    if idname == 'ShaderNodeBsdfTranslucent':
        sockets['Translucency'] = 1.0
        sockets['Specular Intensity'] = 0.0

    rough = _f1(values, 'Roughness', default=0.5)
    if 'Roughness' in links:
        notes.append("a node-driven Roughness cannot become a Hardness "
                     "exponent; the constant default was used")
    hard = glossiness_from_roughness(rough)
    sockets['Hardness'] = max(1.0, min(hard, 511.0))
    if hard > 511.0:
        notes.append("hardness clamped to BI's 511 ceiling")
    if rough > 0.6 and idname in ('ShaderNodeBsdfPrincipled',
                                  'ShaderNodeBsdfDiffuse'):
        props['diff_shader'] = 'OREN_NAYAR'
        sockets['Roughness'] = rough
        notes.append("rough surface: Oren-Nayar diffuse")

    spec = _f1(values, 'Specular IOR Level', 'Specular', default=0.5)
    if 'Specular Intensity' not in sockets:
        link_or_value('Specular Intensity',
                      ['Specular IOR Level', 'Specular'], spec)
    else:
        spec = sockets['Specular Intensity']

    metal = _f1(values, 'Metallic', default=0.0)
    if metal > 0.5 or 'Metallic' in links:
        # BI has no metalness; the era's move was tinting the highlight
        # with the base colour and dropping the diffuse
        sockets['Specular Color'] = tuple(base) + (1.0,)
        sockets['Intensity'] = max(0.0, 1.0 - metal) \
            if 'Metallic' not in links else 0.0
        sockets['Specular Intensity'] = max(spec, 0.8)
        notes.append("BI has no metalness; the highlight is tinted with "
                     "the base colour instead")

    # emission -> Emit, BI's float against the diffuse chain
    e_col = _c3(values, 'Emission Color', 'Emission', default=(0.0, 0.0, 0.0))
    e_str = _f1(values, 'Emission Strength', default=1.0)
    if 'Emission Strength' in links or \
            any(a in links for a in ('Emission Color', 'Emission')):
        notes.append("node-driven emission has no BI equivalent; Emit "
                     "stayed 0")
    elif e_col is not None and e_str > 0.0 and _nonblack(e_col):
        base_m = max(sum(base) / 3.0, 1e-3)
        emit_m = sum(e_col) / 3.0
        sockets['Emit'] = min(e_str * emit_m / base_m, 20.0)
        if any(abs(e_col[i] - base[i]) > 0.05 for i in range(3)):
            notes.append("BI's Emit glows in the diffuse colour; a "
                         "different emission tint cannot be carried")

    alpha = _f1(values, 'Alpha', default=1.0)
    if 'Alpha' in links:
        props['use_transparency'] = True
        out_links.append(('Alpha', 'Alpha'))
    elif alpha < 1.0 - 1e-6:
        props['use_transparency'] = True
        sockets['Alpha'] = alpha

    ior = _f1(values, 'IOR', default=0.0)
    if props.get('use_transparency') and ior > 1.0 + 1e-6:
        props['transp_mode'] = 'RAYTRACE'
        sockets['Ray IOR'] = min(max(ior, 1.0), 5.0)
        notes.append("transparency with an IOR: BI Raytrace mode")

    if 'Normal' in links:
        out_links.append(('Normal', 'Normal'))
    return {'props': props, 'sockets': sockets, 'links': out_links,
            'notes': notes}
