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
        try:
            e = values.get('Emission Strength', 0.0)
            e = float(e[0]) if hasattr(e, '__len__') else float(e)
            if e > 0.0 and e != 1.0:
                notes.append(f"emission strength {e:g} folded into the colour")
        except (TypeError, ValueError):
            pass

    chosen = choose_model(idname, values, links) if model == 'AUTO' else model
    return {'model': chosen, 'pairs': pairs, 'extras': extras, 'notes': notes,
            'source': idname}
