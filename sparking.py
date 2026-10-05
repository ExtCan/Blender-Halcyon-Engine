"""Sparking! ZERO material import (R246, the reading corrected R247).

The game's character materials are Unreal material instances: a flat
colour, a greyscale line-art sheet, a 16x256 tone strip and a parameter
set per part. FModel exports each instance as a .json (the `Textures`
map and the `Parameters` block: Colors, Scalars, Switches, Properties)
with the referenced textures as PNGs in the same content tree. This
module reads that export and builds the Anime Shader graph that
reproduces the material under the SPARKING compatibility mode, read
against the field's reconstruction of the master material (an Unreal
remake of M_ChrToon, its expression graph walked node by node):

    Color1 x Color2           -> Diffuse Color, AS EXPORTED. The export's
                                 R,G,B are Unreal's linear colour and the
                                 Hex beside them is that value's sRGB
                                 encoding -- the swatch the artist chose
                                 (the SS4 fur #752A31, the vest #656465).
                                 The remake raises Color1 and
                                 GradientAdjust1 to 2.2 before use;
                                 applied to values that are already
                                 linear that put the fur at 2% and the
                                 vest at 1% -- the field's "PURE BLACK
                                 FUR". Not applied.
    Mask1 (linear)            -> Game Texture (greyscale detail, a multiply)
    GradientTexture (sRGB)    -> Shadow Ramp, read down its height by the
                                 half-Lambert cosine, the tone lifted from
                                 GradientAdjust1 (the shadow floor, as
                                 exported) to white by the strip's value:
                                 tone = floor + (1 - floor) x strip
    SpecularColor / M_Gloss / SpecularShininess / SpecularSmooth
                              -> carried onto the highlight's colour, size
                                 and sharpness with Specular Level 0: the
                                 reconstruction has no highlight term and
                                 the gate these drive is not established
                                 (read as pow(N.H, shininess) over the
                                 alpha, M_Gloss x the HDR peak as the
                                 level, it covered the SS4 shoulder pads
                                 in white)
    RimLightColor, RimlightSize
                              -> the rim's colour and power, carried with
                                 Rim Amount 0 (the remake gates a white
                                 rim on the lit side's edge and blends it
                                 as soft light; raise Rim Amount for it)
    LineColor                 -> Line Color (own colour)
    ColorTexture1             -> the base where the instance's Color1 is
                                 BLACK: the mouth instances hand their
                                 colour to the palette in that slot (the
                                 common Mouth_00, or the character's own
                                 *_MTH_00). Every other instance carries
                                 Mouth_00 there as the master's default
                                 and a flat Color1, and the palette is
                                 not read (its UVs would land on it)
    EyeTexture                -> the eye (EyeColor on the sheet's red, the
                                 highlight on its blue, the sclera --
                                 EyeColor2 -- outside the disc its alpha
                                 marks; opaque)
    DynamicLightAmbient       -> Ambient (the character lights itself)
    MI_LNE000                 -> the game's outline shell: an inverted
                                 hull (the body's triangles again, wound
                                 the other way) the game's vertex shader
                                 pushes out along the normal by the COL0
                                 width. The export carries it flat
                                 against the body, where a coarse hull
                                 pokes through at every crease -- the
                                 field's "white lines" over the abs.
                                 Halcyon inks the outline from the
                                 G-buffer, so the shell becomes a punch-
                                 through at Opacity 0 in its own line
                                 colour: nothing of it draws or shadows

What the reconstruction leaves out is left out here and reported by
name, never guessed: ReflectionTexture (its matcap), the Toplight, the
FaceLighting masks, ShadowStep, the damage layers, the sweat drops'
trigger, the vertex colours (COL0 is the outline width: 1 on the body,
0 inside the mouth, the hull's own push). The remake's soft light over
the whole base (black where the rim is off, which squares the colour)
is not applied.

Nothing from the game ships: the decode reads the user's own export at
import time. Pure Python up to the Blender operator at the bottom.
"""
import json
import os

import numpy as np

import bpy
from bpy.props import BoolProperty, CollectionProperty, StringProperty

try:                                        # fakebpy has no bpy_extras
    from bpy_extras.io_utils import ImportHelper
except ImportError:                         # pragma: no cover
    class ImportHelper:
        pass


#: texture basenames the game uses as "nothing plugged in" -- Mouth_00
#: is the ColorTexture1 slot's default on every part but the mouth
#: itself, which is told apart by its black Color1 (read as a texture on
#: a body part it painted the lips grey: the field's "the mouth lip areas
#: are incorrectly colored")
NEUTRAL_TEXTURES = frozenset((
    'T_MaskDefault', 'T_MaskDefault1', 'T_AlphaDefault', 'T_ColorDefault',
    'T_DecalDefault', 'Mouth_00'))

#: image extensions FModel writes, in the order they are tried
IMAGE_EXTS = ('.png', '.tga', '.jpg', '.jpeg', '.dds')

#: the tone strip's colour space: a colour texture (the remake samples
#: it with the default sampler), so 134 grey reads 0.235 of the light;
#: the decode below linearises the strip the same way when it reads the
#: bands into the sliders, so the two roads agree
TONE_STRIP_COLORSPACE = 'sRGB'

#: the game's shared instances -- every character's sections wear these
#: names (the outline shell, the mouth, the sweat), and their .json lives
#: under Characters/Common/Materials, not beside the character's own; the
#: operator finds them for the slots the UEFormat importer leaves, and
#: REBUILDS them on every import: they are the game's, never the user's,
#: and a build left by an earlier reading (1.87.0 gave the shell an Anime
#: Shader in a grey) is exactly what a re-import must replace
SHARED_PREFIXES = ('MI_LNE', 'MI_MTH', 'MI_SWT')

#: the outline shell's material name (the game's own), for an export
#: that lacks its .json
OUTLINE_PREFIX = 'MI_LNE'

#: the ID property stamped on every material this importer builds, with
#: the version that built it -- a re-import rebuilds a slot an older
#: reading built, and leaves a tree the user made alone
BUILT_TAG = 'halcyon_sparking'


# ------------------------------------------------------------------ reading


def read_mi(path):
    """The FModel material-instance export as a dict, validated to the
    shape this module reads ({'Textures': {...}, 'Parameters': {...})."""
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or 'Parameters' not in data:
        raise ValueError(f'{os.path.basename(path)}: not an FModel material '
                         'instance export (no Parameters block)')
    data.setdefault('Textures', {})
    p = data['Parameters']
    for key in ('Colors', 'Scalars', 'Switches', 'Properties'):
        if not isinstance(p.get(key), dict):
            p[key] = {}
    return data


def texture_name(ref):
    """'/Game/SS/Characters/Common/Textures/T_ToneSKN04.T_ToneSKN04' ->
    'T_ToneSKN04'. A bare name comes back unchanged."""
    if not ref:
        return ''
    base = str(ref).rsplit('/', 1)[-1]
    return base.split('.')[0]


def is_neutral(ref):
    return texture_name(ref) in NEUTRAL_TEXTURES


def resolve_texture(ref, json_path, exists=os.path.exists):
    """The exported image for a '/Game/...' reference, found beside the
    JSON: FModel writes '/Game/SS/X/T.T' as '<root>/SS/X/T.png' (or
    under a 'Content' folder), so every ancestor of the JSON's folder
    is tried as the root, with and without 'Content'. None when no
    file is there -- the caller reports it by name."""
    if not ref:
        return None
    rel = str(ref).split('.')[0]
    if rel.startswith('/Game/'):
        rel = rel[len('/Game/'):]
    elif rel.startswith('/Engine/'):
        return None
    rel = rel.lstrip('/')
    folder = os.path.dirname(os.path.abspath(json_path))
    seen = set()
    while folder and folder not in seen:
        seen.add(folder)
        for root in (folder, os.path.join(folder, 'Content')):
            for ext in IMAGE_EXTS:
                cand = os.path.join(root, *rel.split('/')) + ext
                if exists(cand):
                    return cand
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    return None


# ----------------------------------------------------------------- decoding


def _rgb(colors, name, default=(1.0, 1.0, 1.0)):
    """A colour parameter as exported -- Unreal's linear value, which is
    what the sockets take (the Hex the export writes beside it is that
    value's sRGB encoding: the swatch the artist chose). Negatives
    clamp; values above 1 keep their overflow (the glow accessories,
    the HDR specular)."""
    c = colors.get(name)
    if not isinstance(c, dict):
        return tuple(max(float(v), 0.0) for v in default)
    return tuple(max(float(c.get(k, d)), 0.0)
                 for k, d in zip('RGB', default))


def srgb_hex(rgb):
    """The export's Hex for a linear colour -- the swatch as the artist
    saw it; the import log names colours this way."""
    from .core.mathx import linear_to_srgb
    v = linear_to_srgb(np.clip(np.asarray(rgb, np.float32)[:3], 0.0, 1.0))
    return '#' + ''.join(f'{int(round(float(x) * 255.0)):02X}' for x in v)


def _alpha(colors, name, default=1.0):
    c = colors.get(name)
    if not isinstance(c, dict):
        return float(default)
    return float(c.get('A', default))


def _scalar(scalars, name, default=0.0):
    v = scalars.get(name)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def specular_size(alpha, shininess):
    """The game gates its highlight where pow(N.H, shininess) exceeds
    the specular colour's alpha; the Anime Shader gates where the
    WRAPPED half-vector exceeds 1 - Specular Size. Same edge:
    size = (1 - alpha ** (1 / shininess)) / 2."""
    a = min(max(float(alpha), 0.0), 1.0)
    s = max(float(shininess), 1e-4)
    thr = a ** (1.0 / s) if a > 0.0 else 0.0
    return max(0.0, min(0.5, (1.0 - thr) * 0.5))


def is_outline(mi):
    """The game's outline shell instance (MI_LNE000): the unlit, masked
    line material -- ChrToonlineColor2 and no Color1 of its own."""
    C = mi.get('Parameters', {}).get('Colors', {})
    return 'ChrToonlineColor2' in C and 'Color1' not in C


def decode_outline(mi, name='MI_LNE000'):
    """The outline shell as Halcyon's material: its line colour on a
    Halcyon Shader at Opacity 0, punch-through, casting nothing. The
    hull the export carries is the body's own triangles wound the other
    way and left where the game's vertex shader would push them out;
    flat against the body it pokes through at every crease (the field's
    "white lines" over the abs). Halcyon inks the outline from the
    G-buffer, so the shell draws nothing."""
    C = mi.get('Parameters', {}).get('Colors', {}) if mi else {}
    colour = tuple(min(a * b, 1.0) for a, b in zip(
        _rgb(C, 'ChrToonlineColor2', (0.0, 0.0, 0.0)),
        _rgb(C, 'ChrColorMult', (1.0, 1.0, 1.0))))
    notes = [f"the game's outline shell: an inverted hull its vertex "
             "shader pushes out along the normal (the export carries it "
             "flat against the body, where it pokes through at the "
             f"creases); its line colour {srgb_hex(colour)} on a Halcyon "
             "Shader at Opacity 0, Alpha Mode Clip, casting no shadow -- "
             "nothing of it draws; Halcyon inks the outline from the "
             "G-buffer (Line Color on each part)"]
    inputs = {'Diffuse Color': colour + (1.0,), 'Diffuse Level': 1.0,
              'Specular Level': 0.0, 'Ambient': 0.0, 'Opacity': 0.0}
    return {'name': name, 'kind': 'outline', 'props': {'model': 'LAMBERT'},
            'inputs': inputs, 'textures': {}, 'notes': notes,
            'flags': {'alpha_mode': 'CLIP', 'alpha_clip': 0.5,
                      'cast_shadow': False}}


def decode(mi, name='Material'):
    """The FModel instance as Halcyon's own parameter set: node props,
    socket values, texture roles (name -> the game's reference), and
    the notes naming what was read as what and what was left alone.
    The outline shell's instance decodes to its own kind (see
    decode_outline); everything else is an Anime Shader."""
    if is_outline(mi):
        return decode_outline(mi, name)
    P = mi.get('Parameters', {})
    C = P.get('Colors', {})
    S = P.get('Scalars', {})
    T = mi.get('Textures', {})
    notes = []

    color1 = _rgb(C, 'Color1', (0.8, 0.8, 0.8))
    color2 = _rgb(C, 'Color2', (1.0, 1.0, 1.0))
    # the shadow floor: the strip lifts from it to white, so a floor at
    # or above white (the glow accessories carry 2.0) is no shading at
    # all -- held at 1
    floor = tuple(min(v, 1.0) for v in _rgb(C, 'GradientAdjust1', (0.0, 0.0, 0.0)))
    step = _scalar(S, 'ShadowStep', 0.0)
    spec_rgb = _rgb(C, 'SpecularColor', (0.0, 0.0, 0.0))
    spec_a = _alpha(C, 'SpecularColor', 1.0)
    gloss = _scalar(S, 'M_Gloss', 1.0)
    shin = _scalar(S, 'SpecularShininess', 1.0)
    smooth = _scalar(S, 'SpecularSmooth', 0.0)
    rim_int = _scalar(S, 'Rimlight_Intensity', 0.0)
    rim_size = _scalar(S, 'RimlightSize', 0.65)
    rim_rgb = tuple(min(a * b, 1.0) for a, b in zip(
        _rgb(C, 'RimLightColor', (1.0, 1.0, 1.0)),
        _rgb(C, 'RimlightColorMult_1', (1.0, 1.0, 1.0))))
    line = tuple(min(v, 1.0) for v in _rgb(C, 'LineColor', (0.0, 0.0, 0.0)))
    refl = _scalar(S, 'UseReflection', 0.0)
    top = _scalar(S, 'ChrToplight_Intensity', 0.0)
    ambient = _rgb(C, 'DynamicLightAmbient', (0.0, 0.0, 0.0))
    # the highlight: the reconstruction carries no specular term, and
    # the gate SpecularColor's alpha, M_Gloss, SpecularShininess and
    # SpecularSmooth drive is not established -- read as pow(N.H,
    # shininess) over the alpha with M_Gloss x the HDR peak as the
    # level, it covered the SS4 shoulder pads in white. The colour
    # (its HDR peak normalised), size and sharpness go on the node
    # with Specular Level 0, and the note says so by the numbers
    spec_peak = max(spec_rgb)
    spec_raw = spec_rgb
    if spec_peak > 1.0:
        spec_rgb = tuple(v / spec_peak for v in spec_rgb)
    if spec_peak > 1e-6 and gloss > 1e-6:
        notes.append(f'SpecularColor ({spec_raw[0]:.2f}, {spec_raw[1]:.2f}, '
                     f'{spec_raw[2]:.2f}) alpha {spec_a:.2f}, M_Gloss '
                     f'{gloss:g}, SpecularShininess {shin:g}, SpecularSmooth '
                     f'{smooth:g}: the reconstruction has no highlight term '
                     'and the gate these drive is not established -- the '
                     'colour, size and sharpness are on the node with '
                     'Specular Level 0; raise it for a highlight')
    # an HDR Color1 (the glowing accessories carry 2.0 with an
    # Emissive2_Mult) is read as a glow: the paint normalised, the
    # overflow driving Self-Illumination -- stated in the notes
    glow = max(color1)
    emission = None
    if glow > 1.0:
        emission = (tuple(v / glow for v in color1), glow - 1.0)
        color1 = tuple(v / glow for v in color1)
        notes.append(f'Color1 peaks at {glow:.2f}: read as a glow -- the '
                     'paint normalised, the overflow into Self-Illumination')
    # Color2 multiplies the whole base in the remake; a flat colour
    # folds it in here, a textured base multiplies it in the graph
    color2_on = any(abs(v - 1.0) > 1e-6 for v in color2)
    base = tuple(min(a * b, 1.0) for a, b in zip(color1, color2)) \
        if color2_on else color1
    inputs = {
        'Diffuse Color': base + (1.0,),
        'Shadow Bias': 0.0,
        'Shadow 1 Softness': 0.0,
        'Shadow 2 Softness': 0.0,
        'Specular Color': spec_rgb + (1.0,),
        'Specular Level': 0.0,
        'Specular Size': specular_size(spec_a, shin),
        'Specular Sharpness': max(smooth * 0.5, 0.0),
        'Rim Color': rim_rgb + (1.0,),
        # the remake gates a white rim on the lit side's edge (Fresnel
        # over a threshold that rises into the shadow) and soft-lights
        # it in; the colour and size are carried with the amount at 0
        'Rim Amount': 0.0,
        'Rim Power': max(0.1, rim_size),
        'Line Color': line + (1.0,),
        # the character lights itself: the strip carries the shadow's
        # colour, and the game's own ambient term is DynamicLightAmbient
        # (0 on nearly every part) -- never the scene's default lift
        'Ambient': float(sum(ambient) / 3.0),
    }
    if emission is not None:
        inputs['Self-Illumination'] = emission[0] + (1.0,)
        inputs['Emission Strength'] = emission[1]
    props = {'compat': 'SPARKING', 'tones': 'TWO', 'line_source': 'CUSTOM'}

    textures = {}
    mask = T.get('Mask1')
    if mask and not is_neutral(mask):
        textures['game'] = mask
    ramp = T.get('GradientTexture')
    if ramp:
        textures['ramp'] = ramp
    ctex = T.get('ColorTexture1')
    black1 = max(color1) <= 1e-6
    if ctex and (black1 or not is_neutral(ctex)):
        # a black Color1 hands the base to the palette in the slot (the
        # mouth: the closed line dark brown, the teeth, the tongue); on
        # every other part the slot's Mouth_00 is the master's default
        # under a flat colour and is not read
        textures['color'] = ctex
        if black1:
            notes.append(f'Color1 is black: ColorTexture1 {texture_name(ctex)} '
                         'is the base (the mouth palette)')
    eye = T.get('EyeTexture')
    if eye and not is_neutral(eye):
        textures['eye'] = eye
        inputs['Diffuse Color'] = (1.0, 1.0, 1.0, 1.0)
    if refl > 1e-6:
        notes.append(f'UseReflection {refl:g} with {texture_name(T.get("ReflectionTexture", ""))}: '
                     'the reflection sphere map is not decoded (its channel '
                     'layout is not established) -- left off')
    if top > 1e-6:
        notes.append(f'ChrToplight_Intensity {top:g}: the top light is not '
                     'decoded -- left off')
    for k in ('FaceLightMask1', 'FaceLightMask2', 'FaceLightMask3',
              'BodyLightMask1', 'BodyLightMask2'):
        ref = T.get(k)
        if ref and not is_neutral(ref):
            notes.append(f'{k} {texture_name(ref)}: face/body light masks '
                         'are not decoded -- left off')
    if rim_int > 1e-6:
        notes.append(f'Rimlight_Intensity {rim_int:g}, RimlightSize {rim_size:g}: '
                     'the rim is carried with Rim Amount 0 (the remake soft-'
                     'lights a white Fresnel edge on the lit side) -- raise '
                     'Rim Amount to see it')
    if step > 1e-6:
        notes.append(f'ShadowStep {step:g}: not read -- the remake indexes '
                     'the strip by the half-Lambert cosine alone')
    if 'ramp' not in textures:
        notes.append('no GradientTexture: the tone sliders carry the '
                     'defaults -- set them from the character\'s strip')
    return {'name': name, 'kind': 'surface', 'props': props,
            'inputs': inputs, 'textures': textures, 'color2': color2,
            'color2_on': color2_on, 'floor': floor,
            'eye_color': tuple(min(v, 1.0) for v in
                               _rgb(C, 'EyeColor', (1.0, 1.0, 1.0))),
            'sclera': tuple(min(v, 1.0) for v in
                            _rgb(C, 'EyeColor2', (1.0, 1.0, 1.0))),
            'notes': notes, 'flags': {}}


def decode_tone_strip(pixels, floor=(0.0, 0.0, 0.0), min_rows=4,
                      colorspace=None):
    """The tone strip's bands as the Anime Shader's slider values, so
    the material still reads right with the ramp unlinked. `pixels` is
    the strip as loaded (H,W,C float, bottom-left origin: the top row
    is the lit end), decoded from `colorspace` (TONE_STRIP_COLORSPACE
    unless said otherwise) the way the ramp road decodes the texture;
    each band's value lifts the `floor` (GradientAdjust1, linear) toward
    white -- the remake's lerp. Runs shorter than `min_rows` (the strips'
    3-row compression transitions) begin the band below. Returns the
    dict of socket values plus the 'tones' prop, or None when the strip
    has a single band."""
    px = np.asarray(pixels, np.float32)
    if px.ndim != 3 or px.shape[0] < 2:
        return None
    col = px[::-1, 0, :3]                      # top row first
    if (colorspace or TONE_STRIP_COLORSPACE) == 'sRGB':
        from .core.mathx import srgb_to_linear
        col = srgb_to_linear(np.ascontiguousarray(col))
    h = col.shape[0]
    q = np.round(col * 32.0) / 32.0
    runs = []                                  # [start, end, rgb]
    start = 0
    for y in range(1, h + 1):
        if y == h or np.any(q[y] != q[start]):
            runs.append([start, y, col[start:y].mean(0)])
            start = y
    merged = []
    carry = None
    for i, r in enumerate(runs):
        if (r[1] - r[0]) < min_rows and i + 1 < len(runs):
            # a compression transition: the band below starts here
            carry = r[0] if carry is None else carry
            continue
        if carry is not None:
            r = [carry, r[1], r[2]]
            carry = None
        if (r[1] - r[0]) < min_rows and merged:
            merged[-1][1] = r[1]
            continue
        merged.append(r)
    if len(merged) < 2:
        return None
    bands = merged[:3]
    a = np.asarray(floor, np.float32)

    def tone(rgb):
        # floor + (1 - floor) * strip, per channel
        return tuple(float(v) for v in
                     np.clip(a + (1.0 - a) * rgb, 0.0, 1.0)) + (1.0,)

    out = {'Shadow 1 Threshold': float(1.0 - bands[0][1] / h),
           'Shadow 1 Color': tone(bands[1][2]),
           'Shadow 1 Softness': 0.0, 'Shadow 2 Softness': 0.0,
           'tones': 'TWO'}
    if len(bands) >= 3:
        out['tones'] = 'THREE'
        out['Shadow 2 Threshold'] = float(1.0 - bands[1][1] / h)
        out['Shadow 2 Color'] = tone(bands[2][2])
    return out


# ------------------------------------------------------- the graph itself


def _sock(name, kind, default, link=None):
    return {'name': name, 'type': kind, 'default': default, 'link': link}


def _node(nid, idname, props, inputs, outputs, location=(0, 0)):
    return {'id': nid, 'bl_idname': idname, 'props': dict(props),
            'inputs': inputs,
            'outputs': [{'name': n, 'type': t} for n, t in outputs],
            'location': location}


def _image_node(nid, key, colorspace, y):
    return _node(nid, 'ShaderNodeTexImage',
                 {'image': key, 'interpolation': 'Linear',
                  'extension': 'REPEAT', 'projection': 'FLAT',
                  'colorspace': colorspace},
                 [_sock('Vector', 'VECTOR', [0.0, 0.0, 0.0])],
                 [('Color', 'RGBA'), ('Alpha', 'VALUE')], (-820, y))


def _mix_node(nid, blend, fac, c1, c2, y, fac_link=None, c1_link=None,
              c2_link=None, x=-420):
    return _node(nid, 'ShaderNodeMixRGB', {'blend_type': blend},
                 [_sock('Fac', 'VALUE', fac, fac_link),
                  _sock('Color1', 'RGBA', list(c1), c1_link),
                  _sock('Color2', 'RGBA', list(c2), c2_link)],
                 [('Color', 'RGBA')], (x, y))


def build_graph(dec, images):
    """The Anime Shader graph for one decoded material. `images` maps
    texture roles ('game', 'ramp', 'color', 'eye') to image keys (the
    names the scene's image table -- or bpy.data.images -- carries);
    a role without an image falls back to the flat parameters. The
    dict is nodeeval's own serialized shape, so it renders headless as
    it is and `graph_to_tree` rebuilds it in Blender node for node."""
    from .nodes.shader_nodes import HALCYON_AnimeShaderNode as AN
    kinds = {'NodeSocketColor': 'RGBA', 'NodeSocketFloat': 'VALUE',
             'NodeSocketVector': 'VECTOR'}
    if dec.get('kind') == 'outline':
        return _outline_graph(dec)
    nodes = {}
    links = {}
    y = 420
    game = images.get('game') if 'game' in dec['textures'] else None
    ramp = images.get('ramp') if 'ramp' in dec['textures'] else None
    ctex = images.get('color') if 'color' in dec['textures'] else None
    eye = images.get('eye') if 'eye' in dec['textures'] else None

    if eye:
        # the eye sheet is channel data, not a picture: red and green
        # everywhere but the pupil ring, blue only at the highlight,
        # and the ALPHA is the iris's disc. EyeColor where the red is
        # (the iris), black where it is not (the pupil), white where
        # the blue is (the highlight) -- and outside the disc the
        # sclera, EyeColor2 (white in every shipped instance). The eye
        # is opaque: the alpha is a mask, never a cut-out (a cut-out
        # showed the socket through the whites)
        nodes['eye'] = _image_node('eye', eye, 'sRGB', y)
        nodes['eyech'] = _node('eyech', 'ShaderNodeSeparateColor',
                               {'mode': 'RGB'},
                               [_sock('Color', 'RGBA', [0, 0, 0, 1],
                                      ['eye', 0])],
                               [('Red', 'VALUE'), ('Green', 'VALUE'),
                                ('Blue', 'VALUE')], (-620, y))
        ec = list(dec['eye_color']) + [1.0]
        nodes['iris'] = _mix_node('iris', 'MIX', 1.0, [0, 0, 0, 1], ec, y,
                                  fac_link=['eyech', 0])
        nodes['eyehl'] = _mix_node('eyehl', 'MIX', 1.0, [0, 0, 0, 1],
                                   [1, 1, 1, 1], y - 40,
                                   fac_link=['eyech', 2],
                                   c1_link=['iris', 0], x=-320)
        sclera = list(dec.get('sclera', (1.0, 1.0, 1.0))) + [1.0]
        nodes['eyemix'] = _mix_node('eyemix', 'MIX', 1.0, sclera,
                                    [0, 0, 0, 1], y - 80,
                                    fac_link=['eye', 1],
                                    c2_link=['eyehl', 0], x=-220)
        links['Diffuse Color'] = ['eyemix', 0]
        y -= 260
    elif ctex:
        # the instance's own colour texture (the mouth's palette) is
        # the base; Color2 multiplies it below
        nodes['ctex'] = _image_node('ctex', ctex, 'sRGB', y)
        links['Diffuse Color'] = ['ctex', 0]
        y -= 260
    if game:
        nodes['mask'] = _image_node('mask', game, 'Non-Color', y)
        links['Game Texture'] = ['mask', 0]
        y -= 260
    if ramp:
        # tone = floor + (1 - floor) * strip, per channel: a multiply
        # by (1 - floor) then an add of the floor -- the remake's lerp
        # from GradientAdjust1 to white by the strip
        nodes['tone'] = _image_node('tone', ramp, TONE_STRIP_COLORSPACE, y)
        nodes['tone']['props']['extension'] = 'EXTEND'
        fl = [float(v) for v in dec['floor']]
        if max(fl) > 1e-6:
            span = [max(1.0 - v, 0.0) for v in fl] + [1.0]
            nodes['span'] = _mix_node('span', 'MULTIPLY', 1.0, [0, 0, 0, 1],
                                      span, y, c1_link=['tone', 0], x=-620)
            nodes['floor'] = _mix_node('floor', 'ADD', 1.0, [0, 0, 0, 1],
                                       fl + [1.0], y, c1_link=['span', 0])
            links['Shadow Ramp'] = ['floor', 0]
        else:
            links['Shadow Ramp'] = ['tone', 0]
        y -= 260
    if dec.get('color2_on') and 'Diffuse Color' in links:
        # a textured base still owes the Color2 multiply
        nodes['c2'] = _mix_node('c2', 'MULTIPLY', 1.0, [0, 0, 0, 1],
                                list(dec['color2']) + [1.0], y,
                                c1_link=links['Diffuse Color'], x=-120)
        links['Diffuse Color'] = ['c2', 0]

    ins = []
    for kind, sname, dflt in AN.SOCKETS:
        t = kinds.get(kind, 'VALUE')
        v = dec['inputs'].get(sname, dflt)
        if v is None:
            v = {'VALUE': 0.0, 'RGBA': [0.0, 0.0, 0.0, 1.0],
                 'VECTOR': [0.0, 0.0, 0.0]}[t]
        v = list(v) if hasattr(v, '__len__') else float(v)
        ins.append(_sock(sname, t, v, links.get(sname)))
    nodes['anime'] = _node('anime', 'HALCYON_AnimeShaderNode', dec['props'],
                           ins, [('Surface', 'SHADER')], (0, 0))
    nodes['out'] = _node('out', 'ShaderNodeOutputMaterial', {},
                         [_sock('Surface', 'SHADER', None, ['anime', 0]),
                          _sock('Displacement', 'VECTOR', [0.0, 0.0, 0.0])],
                         [], (320, 0))
    return {'output': 'out', 'nodes': nodes}


def _outline_graph(dec):
    """The outline shell's tree: a Halcyon Shader (Lambert) in the line
    colour at Opacity 0 -- with the material's Alpha Mode Clip (the
    flags beside the decode) the punch-through road drops every
    fragment before shading, on both devices."""
    from .nodes.shader_nodes import HALCYON_ShaderNode as SN
    kinds = {'NodeSocketColor': 'RGBA', 'NodeSocketFloat': 'VALUE',
             'NodeSocketVector': 'VECTOR', 'HALCYON_BlendValueSocket': 'VALUE'}
    ins = []
    for kind, sname, dflt in SN.SOCKETS:
        t = kinds.get(kind, 'VALUE')
        v = dec['inputs'].get(sname, dflt)
        if v is None:
            ins.append(_sock(sname, t, None))
            continue
        v = list(v) if hasattr(v, '__len__') else float(v)
        ins.append(_sock(sname, t, v))
    nodes = {'shader': _node('shader', 'HALCYON_ShaderNode', dec['props'],
                             ins, [('Surface', 'SHADER')], (0, 0)),
             'out': _node('out', 'ShaderNodeOutputMaterial', {},
                          [_sock('Surface', 'SHADER', None, ['shader', 0]),
                           _sock('Displacement', 'VECTOR', [0.0, 0.0, 0.0])],
                          [], (320, 0))}
    return {'output': 'out', 'nodes': nodes}


def apply_flags(target, dec):
    """The decode's material flags (Alpha Mode, clip threshold, shadow
    casting) onto `target` -- a material's `halcyon` settings in
    Blender, the Material dataclass headless. Only the outline shell
    carries any."""
    for k, v in dec.get('flags', {}).items():
        try:
            setattr(target, k, v)
        except Exception:                                       # noqa: BLE001
            pass


def find_instance_json(name, near_json):
    """`<name>.json` anywhere under the export's Characters (or SS, or
    Content) tree, searched upward from a picked JSON -- the shared
    instances (MI_LNE000, MI_MTH000, MI_SWT000) live under
    Characters/Common/Materials, never beside the character's own.
    None when the export lacks it."""
    if not near_json:
        return None
    target = name + '.json'
    start = os.path.dirname(os.path.abspath(near_json))
    beside = os.path.join(start, target)
    if os.path.exists(beside):
        return beside
    # the export tree's root: the nearest ancestor named Characters (or
    # SS, or Content); an export laid out some other way is searched
    # from the picked folder's parent, never from the drive's root
    folder, root = start, None
    seen = set()
    while folder not in seen:
        seen.add(folder)
        if os.path.basename(folder) in ('Characters', 'SS', 'Content'):
            root = folder
            break
        parent = os.path.dirname(folder)
        if parent == folder:
            break
        folder = parent
    if root is None:
        root = os.path.dirname(start)
    for walk_root, dirs, files in os.walk(root):
        # textures are the bulk of an export and never hold a .json
        dirs[:] = [d for d in dirs if d != 'Textures']
        if target in files and 'Materials' in walk_root.split(os.sep):
            return os.path.join(walk_root, target)
    return None


def is_outline_name(name):
    return str(name).startswith(OUTLINE_PREFIX)


def graph_to_tree(mat, graph, images):
    """Rebuild a serialized graph as the material's node tree, node for
    node. `images` maps image keys to bpy images. Props the node lacks
    are skipped; the image key 'colorspace' prop sets the image's own
    colour space (the tone strip is data)."""
    from . import compat as _compat
    if not _compat.uses_nodes(mat):
        mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    made = {}
    for nid, nd in graph['nodes'].items():
        node = tree.nodes.new(nd['bl_idname'])
        try:
            node.name = nid
        except Exception:                                       # noqa: BLE001
            pass
        try:
            node.location = tuple(nd.get('location', (0, 0)))
        except Exception:                                       # noqa: BLE001
            pass
        for k, v in nd.get('props', {}).items():
            if k == 'image':
                img = images.get(v)
                if img is not None:
                    node.image = img
                continue
            if k == 'colorspace':
                img = images.get(nd['props'].get('image'))
                try:
                    img.colorspace_settings.name = v
                except Exception:                               # noqa: BLE001
                    pass
                continue
            if hasattr(node, k):
                try:
                    setattr(node, k, v)
                except (TypeError, ValueError):
                    pass
        for s in nd.get('inputs', ()):
            sock = node.inputs.get(s['name']) if hasattr(node.inputs, 'get') \
                else None
            if sock is None or s.get('default') is None:
                continue
            try:
                sock.default_value = s['default']
            except (TypeError, ValueError, AttributeError):
                pass
        made[nid] = node
    for nid, nd in graph['nodes'].items():
        for s in nd.get('inputs', ()):
            lk = s.get('link')
            if not lk or lk[0] not in made:
                continue
            src = made[lk[0]]
            try:
                tree.links.new(src.outputs[int(lk[1])],
                               made[nid].inputs[s['name']])
            except Exception:                                   # noqa: BLE001
                pass
    surface = made.get('anime') or made.get('shader')
    try:
        surface.refresh_sockets()
    except Exception:                                           # noqa: BLE001
        pass
    return surface


# ------------------------------------------------------------ the operator


def _load_image(path, colorspace):
    img = bpy.data.images.load(path, check_existing=True)
    try:
        img.colorspace_settings.name = colorspace
    except Exception:                                           # noqa: BLE001
        pass
    return img


ROLE_COLORSPACE = {'game': 'Non-Color', 'ramp': TONE_STRIP_COLORSPACE,
                   'color': 'sRGB', 'eye': 'sRGB'}


def import_material(json_path, fill_existing=True, load_image=_load_image,
                    materials=None, name=None):
    """One FModel .json into a Blender material named after it (an
    existing material of that name -- the empty slot the UEFormat
    importer left -- is filled in place when `fill_existing`). A
    `json_path` of None with an outline `name` (an export without
    MI_LNE000.json) builds the shell's material from the name alone.
    Returns (material, decoded, notes)."""
    materials = bpy.data.materials if materials is None else materials
    if json_path is None:
        if not is_outline_name(name):
            raise ValueError(f'{name}: no .json to read')
        dec = decode_outline({}, name)
        dec['notes'].append(f'{name}.json is not in the export: built from '
                            "the section's name (the game's own)")
    else:
        name = name or os.path.splitext(os.path.basename(json_path))[0]
        mi = read_mi(json_path)
        dec = decode(mi, name)
    notes = list(dec['notes'])
    images = {}
    for role, ref in dec['textures'].items():
        path = resolve_texture(ref, json_path)
        if path is None:
            notes.append(f'{role}: {texture_name(ref)} not found beside the '
                         'JSON -- the flat value stands in')
            continue
        try:
            images[role] = load_image(path, ROLE_COLORSPACE[role])
        except Exception as exc:                                # noqa: BLE001
            notes.append(f'{role}: {os.path.basename(path)} failed to load '
                         f'({exc})')
    keys = {role: getattr(img, 'name', role) for role, img in images.items()}
    if 'ramp' in images:
        try:
            px = _image_pixels(images['ramp'])
            bands = decode_tone_strip(px, dec['floor']) if px is not None \
                else None
        except Exception:                                       # noqa: BLE001
            bands = None
        if bands:
            dec['props']['tones'] = bands.pop('tones')
            dec['inputs'].update(bands)
    graph = build_graph(dec, keys)
    mat = materials.get(name) if fill_existing else None
    if mat is None:
        mat = materials.new(name)
    graph_to_tree(mat, graph, {v: images[r] for r, v in keys.items()})
    try:
        mat.halcyon.use_override = False
        apply_flags(mat.halcyon, dec)
    except Exception:                                           # noqa: BLE001
        pass
    from .version import VERSION as _V
    stamp_built(mat, '.'.join(str(v) for v in _V))
    return mat, dec, notes


def has_halcyon_tree(mat):
    """True when the material already carries a Halcyon surface node --
    the slots the UEFormat importer creates have no node tree at all."""
    try:
        if not mat.use_nodes or mat.node_tree is None:
            return False
        return any(str(getattr(n, 'bl_idname', '')).startswith('HALCYON_')
                   for n in mat.node_tree.nodes)
    except Exception:                                           # noqa: BLE001
        return False


def is_shared_name(name):
    return str(name).startswith(SHARED_PREFIXES)


def built_by(mat):
    """The importer version stamped on the material, '' when none (a
    tree the user built, or a slot still bare)."""
    try:
        return str(mat.get(BUILT_TAG, '') or '')
    except Exception:                                           # noqa: BLE001
        try:
            return str(getattr(mat, BUILT_TAG, '') or '')
        except Exception:                                       # noqa: BLE001
            return ''


def stamp_built(mat, version):
    try:
        mat[BUILT_TAG] = str(version)
    except Exception:                                           # noqa: BLE001
        try:
            setattr(mat, BUILT_TAG, str(version))
        except Exception:                                       # noqa: BLE001
            pass


def slot_is_owed(mat, version):
    """Whether a not-picked 'MI_*' slot is rebuilt on this import: a
    bare slot (no Halcyon tree); a shared instance, always (the game's,
    never the user's -- and the field's shell kept an older reading's
    grey Anime Shader through a re-import because it 'had a tree');
    a slot this importer built under another version (stale). A tree
    the user built, or one this version built, is left alone."""
    n = str(getattr(mat, 'name', ''))
    if not has_halcyon_tree(mat):
        return True
    if is_shared_name(n):
        return True
    tag = built_by(mat)
    return bool(tag) and tag != str(version)


def shared_slots(materials, picked, near_json, version=''):
    """The material slots a character's import still owes: every
    'MI_*' material that was not picked and is owed (slot_is_owed),
    paired with its .json found under the export (None for an outline
    shell the export has no .json for -- built from the name). The
    shared instances live under Characters/Common/Materials; a partial
    pick of the character's own folder is covered the same way."""
    out = []
    for mat in materials:
        n = str(getattr(mat, 'name', ''))
        if n in picked or not n.startswith('MI_') or not slot_is_owed(mat, version):
            continue
        path = find_instance_json(n, near_json)
        if path is None and not is_outline_name(n):
            continue
        out.append((n, path))
    return out


def _image_pixels(img):
    """A bpy image's pixels as (H,W,4) float32, bottom-left origin."""
    try:
        from . import compat as _compat
        return _compat.image_pixels(img)
    except Exception:                                           # noqa: BLE001
        return None


class HALCYON_OT_import_sparking(bpy.types.Operator, ImportHelper):
    """Build Halcyon materials from Sparking! ZERO material instances
    exported by FModel (.json), with their textures found beside the
    JSON. Import the character with the UEFormat add-on first: its
    empty material slots carry the instance names, this fills them in
    place, and the slots the pick did not cover -- the game's shared
    outline shell, mouth and sweat -- are found under the export's
    Common materials
    """

    bl_idname = 'halcyon.import_sparking'
    bl_label = "Import Sparking! ZERO Materials"
    bl_options = {'REGISTER', 'UNDO'}

    filename_ext = '.json'
    filepath: StringProperty(subtype='FILE_PATH')
    filter_glob: StringProperty(default='*.json', options={'HIDDEN'})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={'HIDDEN', 'SKIP_SAVE'})
    directory: StringProperty(subtype='DIR_PATH',
                              options={'HIDDEN', 'SKIP_SAVE'})
    fill_existing: BoolProperty(
        name="Fill Existing Materials", default=True,
        description="A material already named after the JSON (the empty "
                    "slot the UEFormat importer created for it) is rebuilt "
                    "in place, so the character's parts pick the result up "
                    "without reassigning; off makes a fresh material")
    def execute(self, context):
        paths = []
        if self.files:
            for f in self.files:
                paths.append(os.path.join(self.directory, f.name))
        elif self.filepath:
            paths.append(self.filepath)
        jobs = [(None, p) for p in paths]
        if self.fill_existing and paths:
            # the slots the UEFormat importer left that the pick did not
            # cover -- the game's shared instances above all (the outline
            # shell, the mouth, the sweat), whose .json lives under
            # Characters/Common/Materials
            picked = {os.path.splitext(os.path.basename(p))[0] for p in paths}
            from .version import VERSION as _V
            jobs.extend(shared_slots(bpy.data.materials, picked, paths[0],
                                     '.'.join(str(v) for v in _V)))
        done = 0
        lines = []
        for name, path in jobs:
            try:
                mat, dec, notes = import_material(path, self.fill_existing,
                                                  name=name)
            except Exception as exc:                            # noqa: BLE001
                lines.append(f'{name or os.path.basename(path)}: FAILED {exc}')
                continue
            done += 1
            roles = ', '.join(f'{r} {texture_name(v)}'
                              for r, v in dec['textures'].items()) or 'no textures'
            kind = 'outline shell' if dec.get('kind') == 'outline' else 'SPARKING'
            where = '' if path is None or name is None else \
                f' (not picked: found under {os.path.basename(os.path.dirname(path))}' \
                + (', rebuilt -- the game\'s shared instance' if is_shared_name(name) else '') + ')'
            lines.append(f'{mat.name}: {kind}, {roles}{where}')
            lines.extend(f'    {n}' for n in notes)
        try:
            from .legacy_import import safe_print, write_log_text
            for line in lines:
                safe_print(f'[halcyon sparking] {line}')
            write_log_text('Sparking ZERO import log', lines)
        except Exception:                                       # noqa: BLE001
            pass
        if done:
            self.report({'INFO'}, f'{done} Sparking! ZERO material(s) built '
                                  "-- receipts in text editor > 'Sparking "
                                  "ZERO import log'")
            return {'FINISHED'}
        self.report({'WARNING'}, 'no material built: '
                    + ('; '.join(lines)[:200] if lines else 'no files'))
        return {'CANCELLED'}


def menu_func(self, context):
    self.layout.operator('halcyon.import_sparking',
                         text="Sparking! ZERO Material (FModel .json) — Halcyon")


CLASSES = (HALCYON_OT_import_sparking,)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func)


def unregister():
    try:
        bpy.types.TOPBAR_MT_file_import.remove(menu_func)
    except Exception:                                           # noqa: BLE001
        pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
