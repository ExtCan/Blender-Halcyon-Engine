"""R251 (1.90.0) material pack, MAT-A: the period COMBINERS on the CPU road.

Every fixed-function machine of 1988-2005 split a lit pixel into two
halves -- the LIGHT half, evaluated once per vertex or per polygon, and
the TEXEL half, sampled per pixel -- and recombined them with its own
integer rule.  Halcyon already has the split (render._shade_interpolated:
the LIGHT pass over a white surface at the vertex / face rate, the ALBEDO
pass per pixel, `light * albedo` at one line); this module owns that one
line for the period models and the small pieces of machinery the items
need around it:

- `rate_for_model`: a period combine is a shading RATE plus a rule
  (RATE_FIXED / COMBINE_MODELS; the three combines keyed on the lighting
  pack's names through LIGHTING_COMBINE take lighting's own rate rows).
- `face_bary`: the provoking-vertex flat shading (C043) -- the FACE road
  samples at one corner instead of the centroid.
- the specular carry (`light_extras`, `light_alpha`): the LIGHT pass's
  free alpha channel carries the specular sum (its luminance, or its RED
  for the DS, or the Mega Drive's intensity class) so the per-pixel rule
  can add the highlight AFTER the texel, as the D3D / GS / DS did.
- `corner_rgb`: the lit corner saturated and quantised to the machine's
  depth BEFORE interpolation (the lit-colour half of raster.md's C040).
- the combines themselves (`cb_*`), float32 / exact-integer numpy, one
  op per statement where a rounding follows, ties named per function;
  gpu/combine.py holds the GLSL twins, bitwise in the simulator.
- `recombine`: the dispatch at the hook, `None` unless a period model is
  present in the chunk (the plain road is then byte-identical: the
  caller's own `light * albedo` runs).
- the Super FX plot (C049): a CPU-chosen pair of palette entries per
  face, packed into the corner texture for the GPU, selected per pixel by
  output-pixel parity on both roads.

Refusals print once per render through `refuse` (`[Halcyon] <model>:
shading as <fallback> -- <why>`) and the GPU probe raises `Refusal`
with the same text (gpu/shade._probe_material's gate).
"""
import numpy as np

from . import mathx as MX
from . import shading as SH

f32 = np.float32

#: the float32 reciprocals, computed ONCE at import (review item 7): the
#: GLSL literal is each value's %.9g -- 0.00392156886, 0.0322580636,
#: 0.0158730168 -- which round-trips the float32 exactly, so both roads
#: multiply by the same bits (`k * R255` differs from `k / 255` for 126 of
#: the 256 k; a driver's `/` is 2.5 ULP anyway)
R255 = f32(1.0) / f32(255.0)
R31 = f32(1.0) / f32(31.0)
R63 = f32(1.0) / f32(63.0)
R255_LIT = '0.00392156886'
R31_LIT = '0.0322580636'
R63_LIT = '0.0158730168'

# ------------------------------------------------------------------ the rate

#: the items whose rate IS the machine's (a flat-shading claim, a per-face
#: class, a per-face plot pair; PCX is VERTEX because at FACE the three
#: corners are equal and the item collapses to the plain product)
RATE_FIXED = {'FLAT_GL_LAST': 'FACE', 'FLAT_D3D_FIRST': 'FACE',
              'SUPERFX_PLOT': 'FACE', 'PCX_INTENSITY': 'VERTEX',
              'MEGA_DRIVE_SH': 'FACE'}
#: the combines: the scene's VERTEX or FACE rate, VERTEX when the scene
#: shades per pixel (a combiner needs the light / albedo split to exist).
#: DS_FIXED is the LIGHTING pack's item (F018) and stays PIXEL-capable
#: (its GLSL branch, hal_dstab and LIGHT-B2's pins): the DS modulate keyed
#: on it (LIGHTING_COMBINE) applies at the VERTEX / FACE scene rates -- the
#: NDS preset's -- and a per-pixel scene shades it per pixel without the
#: blend (material.md coordination item (4), the alternative; CHANGELOG)
COMBINE_MODELS = frozenset({'PS1_MODULATE', 'PS2_HIGHLIGHT', 'SATURN_ADD',
                            'N64_COMBINE', 'S22_MODULATE',
                            'D3D_SEPARATE_SPEC', 'DS_TOON', 'DS_HIGHLIGHT'})
#: the lighting pack's names this pack's combines key on (review item 50):
#: the Sega boards take their rate from lighting's RATE_FOR_MODEL rows
LIGHTING_COMBINE = {'SEGA_MODEL2': 'luma64', 'SEGA_MODEL3': 'luma64',
                    'DS_FIXED': 'ds'}
#: this pack's 13 MODEL_ITEMS (core/shading.py appends them after the
#: lighting pack's four)
PERIOD_MODELS = ('FLAT_GL_LAST', 'FLAT_D3D_FIRST', 'PS1_MODULATE',
                 'PS2_HIGHLIGHT', 'SATURN_ADD', 'N64_COMBINE', 'S22_MODULATE',
                 'D3D_SEPARATE_SPEC', 'PCX_INTENSITY', 'DS_TOON',
                 'DS_HIGHLIGHT', 'MEGA_DRIVE_SH', 'SUPERFX_PLOT')
#: the models `recombine` dispatches (a period combine, or a lighting name
#: with a combine keyed on it)
RECOMBINE_MODELS = frozenset({'PS1_MODULATE', 'PS2_HIGHLIGHT', 'SATURN_ADD',
                              'N64_COMBINE', 'S22_MODULATE',
                              'D3D_SEPARATE_SPEC', 'PCX_INTENSITY',
                              'DS_TOON', 'DS_HIGHLIGHT', 'MEGA_DRIVE_SH',
                              'SUPERFX_PLOT'}) | frozenset(LIGHTING_COMBINE)


def rate_for_model(model, st):
    """The period rate, or None when the existing rows decide."""
    r = RATE_FIXED.get(model)
    if r is not None:
        return r
    if model in COMBINE_MODELS:
        return st.shading_rate if st.shading_rate in ('VERTEX', 'FACE') \
            else 'VERTEX'
    return None


# -------------------------------------------------------------- refusals

class Refusal(Exception):
    """A period item that cannot run as asked: the message is the printed
    reason on the CPU road and the probe's `why` on the GPU road."""


def refuse(st, key, why, fallback='FLAT'):
    """Print `[Halcyon] <key>: shading as <fallback> -- <why>` once per
    render (keyed on `st._period_refusals`, reset by render())."""
    seen = getattr(st, '_period_refusals', None)
    if seen is None:
        seen = set()
        try:
            st._period_refusals = seen
        except Exception:                                       # noqa: BLE001
            pass
    text = f'[Halcyon] {key}: shading as {fallback} -- {why}'
    if text not in seen:
        seen.add(text)
        print(text)
    return text


# ------------------------------------------------------ C043: provoking vertex

_FACE_BARY = {'FLAT_GL_LAST': (0.0, 0.0, 1.0), 'FLAT_D3D_FIRST': (1.0, 0.0, 0.0)}


def _models_of(job, st):
    """One resolved model name per material (render.material_model, the
    call _shade_all already makes)."""
    from .render import material_model
    mats = getattr(job.scene, 'materials', None) or []
    if not mats:
        return [material_model(None, st)]
    return [material_model(m, st) for m in mats]


def _mat_of(job, tri_subset):
    """The material index of every triangle in `tri_subset`."""
    mesh = job.scene.mesh
    if getattr(mesh, 'mat_index', None) is None:
        return np.zeros(np.asarray(tri_subset).shape[0], np.int64)
    return np.asarray(mesh.mat_index)[np.asarray(tri_subset)]


def face_bary(job, tri_subset, st):
    """C043: the FACE road's sample point per triangle -- the centroid
    (float32 1/3 x3, the old constant bit for bit) for every model except
    FLAT_GL_LAST (the LAST corner: OpenGL 1.x GL_FLAT, the PSP) and
    FLAT_D3D_FIRST (the FIRST corner: Direct3D 3-9 D3DSHADE_FLAT). A
    one-hot barycentric makes the context interpolate position AND normal
    at that corner, so the face takes the corner's own Gouraud lighting."""
    tri_subset = np.asarray(tri_subset)
    T = int(tri_subset.shape[0])
    bary = np.full((T, 3), 1.0 / 3.0, np.float32)
    models = _models_of(job, st)
    if not any(m in _FACE_BARY for m in models):
        return bary
    mat_idx = _mat_of(job, tri_subset)
    for mi, model in enumerate(models):
        one = _FACE_BARY.get(model)
        if one is None:
            continue
        sel = mat_idx == mi
        if np.any(sel):
            bary[sel] = np.asarray(one, np.float32)
    return bary


# ------------------------------------------------------ the specular carry

_SPEC_LUM = frozenset({'PS2_HIGHLIGHT', 'D3D_SEPARATE_SPEC'})
_SPEC_RED = frozenset({'DS_TOON', 'DS_HIGHLIGHT'})


def light_extras(model):
    """What the LIGHT pass must keep apart for `model`: the specular sum
    for the four carriers (PS2 / D3D carry its luminance, the DS pair its
    RED), plus the key lamp's lit fraction for the Mega Drive class."""
    if model in _SPEC_LUM or model in _SPEC_RED:
        return {'want_spec': True}
    if model == 'MEGA_DRIVE_SH':
        return {'want_spec': True, 'want_key_lit': True}
    return None


def wants_spec(model):
    return light_extras(model) is not None


def fogs_carry(model):
    """The GS fogged AFTER the texture function, highlight included: the
    PS2 carry is multiplied by the corner's fog transmittance."""
    return model == 'PS2_HIGHLIGHT'


def lobe_model(model):
    """The lobe a model lights with (the DS pair takes DS_FIXED's)."""
    return SH.LOBE_ALIAS.get(model, model)


def _expand5(v5):
    """GBATEK's 5 -> 6 bit expansion, x ? 2x + 1 : 0."""
    return np.where(v5 > 0, f32(2.0) * v5 + f32(1.0), f32(0.0)).astype(f32)


def light_alpha(model, rgb, extras, fog=None):
    """The LIGHT pass's alpha for a carrying model (float32 (n,)), or None.

    PS2_HIGHLIGHT / D3D_SEPARATE_SPEC: the specular sum's Rec.601 luminance
    on the machine's grid (0x80 = 1.0 / 8-bit saturated); PS2's is first
    multiplied by the fog transmittance `fog` when given. DS_TOON /
    DS_HIGHLIGHT: the DS's own saturated lit RED, `clip(rgb.r +
    spec_acc.r, 0, 1)` on the 5-bit grid expanded to 6 (the index the
    rasteriser reads). MEGA_DRIVE_SH: the intensity class 0 / 1 / 2
    (NORMAL / SHADOW / HIGHLIGHT); a missing `key_lit` means the key lamp
    never lit anything -- every face SHADOW (review item 54)."""
    if extras is None:
        return None
    n = int(np.asarray(rgb).shape[0])
    spec = extras.get('spec_acc')
    if spec is None:
        spec = np.zeros((n, 3), f32)
    spec = np.asarray(spec, f32)
    if model in _SPEC_LUM:
        S = np.asarray(MX.luminance_601(spec), f32)
        if model == 'PS2_HIGHLIGHT':
            if fog is not None:
                S = (S * np.asarray(fog, f32).reshape(-1)).astype(f32)
            q = np.clip(np.rint(S * f32(128.0)), 0.0, 255.0).astype(f32)
            return (q / f32(128.0)).astype(f32)
        q = np.rint(np.clip(S, 0.0, 1.0) * f32(255.0)).astype(f32)
        return (q * R255).astype(f32)
    if model in _SPEC_RED:
        red = np.clip(np.asarray(rgb, f32)[:, 0] + spec[:, 0], 0.0, 1.0)
        m = red.astype(f32) * f32(31.0)
        v5 = np.floor(m + f32(0.5)).astype(f32)
        return (_expand5(v5) * R63).astype(f32)
    if model == 'MEGA_DRIVE_SH':
        key_lit = extras.get('key_lit')
        key_lit = np.zeros(n, f32) if key_lit is None \
            else np.asarray(key_lit, f32).reshape(-1)
        slum = np.asarray(MX.luminance_601(spec), f32)
        return np.where(slum >= f32(0.5), f32(2.0),
                        np.where(key_lit < f32(0.5), f32(1.0),
                                 f32(0.0))).astype(f32)
    return None


# ------------------------------------------------------ the corner quantiser

def corner_rgb(model, rgb):
    """The lit corner saturated and quantised to the machine's depth
    BEFORE interpolation (per channel, float32), or None (untouched)."""
    rgb = np.asarray(rgb, f32)
    if model in ('PS1_MODULATE', 'PS2_HIGHLIGHT'):
        c8 = np.clip(np.rint(rgb * f32(128.0)), 0.0, 255.0).astype(f32)
        return (c8 / f32(128.0)).astype(f32)
    if model in ('N64_COMBINE', 'D3D_SEPARATE_SPEC'):
        c8 = np.rint(np.clip(rgb, 0.0, 1.0) * f32(255.0)).astype(f32)
        return (c8 * R255).astype(f32)
    if model == 'SATURN_ADD':
        g5 = np.clip(np.rint(rgb * f32(16.0)), 0.0, 31.0).astype(f32)
        return (g5 / f32(16.0)).astype(f32)
    if model == 'DS_FIXED':
        m = np.clip(rgb, 0.0, 1.0).astype(f32) * f32(31.0)
        v5 = np.floor(m + f32(0.5)).astype(f32)
        return (_expand5(v5) * R63).astype(f32)
    if model == 'S22_MODULATE':
        c8 = np.clip(np.rint(rgb * f32(64.0)), 0.0, 255.0).astype(f32)
        return (c8 / f32(64.0)).astype(f32)
    return None


# ------------------------------------------------------------ the combines

def _t8(A):
    return np.rint(np.clip(np.asarray(A, f32), 0.0, 1.0) * f32(255.0)).astype(f32)


def cb_ps1(L, A):
    """PlayStation GPU texture blend: texel * vertex colour / 128, 0x80 =
    1.0, saturating at 0xFF (ties half to even, the GLSL roundEven)."""
    t8 = _t8(A)
    s8 = np.clip(np.rint(np.asarray(L, f32) * f32(128.0)), 0.0, 255.0).astype(f32)
    v = t8 * s8
    o = np.floor(v / f32(128.0))
    o = np.minimum(o, f32(255.0)).astype(f32)
    return (o * R255).astype(f32)


def cb_saturn(L, A):
    """VDP1 Gouraud add: 5-bit texel + (5-bit table value - 16), clamped
    0..31."""
    t5 = np.rint(np.clip(np.asarray(A, f32), 0.0, 1.0) * f32(31.0)).astype(f32)
    g5 = np.clip(np.rint(np.asarray(L, f32) * f32(16.0)), 0.0, 31.0).astype(f32)
    o = t5 + g5
    o = o - f32(16.0)
    o = np.clip(o, 0.0, 31.0).astype(f32)
    return (o * R31).astype(f32)


def cb_n64(L, A):
    """RDP colour combiner, first cycle (TEXEL0 - 0) * SHADE + 0:
    ((t * s) + 0x80) >> 8 on 8-bit values."""
    t8 = _t8(A)
    s8 = np.clip(np.rint(np.asarray(L, f32) * f32(255.0)), 0.0, 255.0).astype(f32)
    v = t8 * s8
    v = v + f32(128.0)
    o = np.floor(v / f32(256.0)).astype(f32)
    return (o * R255).astype(f32)


def cb_ds(L, A):
    """DS texture blend on 6-bit channels: ((t + 1)(v + 1) - 1) >> 6
    (GBATEK; half-up rounding, the DS's integer arithmetic)."""
    m = np.clip(np.asarray(A, f32), 0.0, 1.0).astype(f32) * f32(63.0)
    m = m + f32(0.5)
    t6 = np.floor(m).astype(f32)
    m = np.clip(np.asarray(L, f32), 0.0, 1.0).astype(f32) * f32(63.0)
    m = m + f32(0.5)
    v6 = np.floor(m).astype(f32)
    v = (t6 + f32(1.0)) * (v6 + f32(1.0))
    v = v - f32(1.0)
    o = np.floor(v / f32(64.0)).astype(f32)
    return (o * R63).astype(f32)


def cb_s22(L, A):
    """Namco System 22: texel * 8-bit shade >> 6, unity at 0x40,
    saturating (MAME's scale_imm_and_clamp(shade << 2))."""
    t8 = _t8(A)
    s8 = np.clip(np.rint(np.asarray(L, f32) * f32(64.0)), 0.0, 255.0).astype(f32)
    v = t8 * s8
    o = np.floor(v / f32(64.0))
    o = np.minimum(o, f32(255.0)).astype(f32)
    return (o * R255).astype(f32)


def cb_ps2hl(Lrgb, S, A):
    """GS HIGHLIGHT: ((t * s) >> 7) + Af, truncated at 0xFF; `S` is the
    carried specular luminance (0x80 = 1.0)."""
    t8 = _t8(A)
    s8 = np.clip(np.rint(np.asarray(Lrgb, f32) * f32(128.0)), 0.0, 255.0).astype(f32)
    v = t8 * s8
    m8 = np.floor(v / f32(128.0)).astype(f32)
    a8 = np.clip(np.rint(np.asarray(S, f32) * f32(128.0)), 0.0, 255.0).astype(f32)
    o = m8 + a8[:, None]
    o = np.minimum(o, f32(255.0)).astype(f32)
    return (o * R255).astype(f32)


def cb_d3dspec(Lrgb, S, A):
    """Direct3D 5-7 / GL 1.2 separate specular / PVR offset colour: the
    texel modulates the diffuse only, the specular is added after and
    clamped at 1 -- three statements, this order."""
    p = np.asarray(A, f32) * np.asarray(Lrgb, f32)
    o = p + np.asarray(S, f32)[:, None]
    return np.minimum(o, f32(1.0)).astype(f32)


def _lum(c):
    c = np.asarray(c, f32)
    l = f32(0.299) * c[:, 0]
    l = l + f32(0.587) * c[:, 1]
    l = l + f32(0.114) * c[:, 2]
    return l.astype(f32)


def cb_pcx(c0, c1, c2, bary, A):
    """PowerVR PCX: one base colour per polygon (the mean of the three lit
    corners) times ONE interpolated scalar intensity (corner luminance
    over the base's luminance)."""
    c0 = np.asarray(c0, f32)[:, :3]
    c1 = np.asarray(c1, f32)[:, :3]
    c2 = np.asarray(c2, f32)[:, :3]
    s = c0 + c1
    s = s + c2
    base = s * f32(0.333333343)
    base = np.clip(base, 0.0, 1.0).astype(f32)
    lb = np.maximum(_lum(base), f32(1e-4)).astype(f32)
    i0 = np.clip(_lum(c0) / lb, 0.0, 1.0).astype(f32)
    i1 = np.clip(_lum(c1) / lb, 0.0, 1.0).astype(f32)
    i2 = np.clip(_lum(c2) / lb, 0.0, 1.0).astype(f32)
    b = np.asarray(bary, f32)
    I = i0 * b[:, 0]
    I = I + i1 * b[:, 1]
    I = I + i2 * b[:, 2]
    p = np.asarray(A, f32)[:, :3] * base
    return (p * I[:, None]).astype(f32)


def cb_luma64(L, A):
    """Sega Model 1/2/3 colour translate: the lit term collapses to one
    luminance quantised to 64 steps BEFORE it meets colour; each 5-bit
    channel through a linear 64-entry ramp (the margin rule on the one
    non-power-of-two divide: floor((num + 0.5) / 1953) == num // 1953)."""
    lum = np.asarray(MX.luminance_601(np.asarray(L, f32)[:, :3]), f32)
    l8 = np.floor(np.clip(lum, 0.0, 1.0) * f32(255.0)).astype(f32)
    lum6 = np.minimum(np.floor(l8 / f32(4.0)), f32(63.0)).astype(f32)
    ca = np.clip(np.asarray(A, f32)[:, :3], 0.0, 1.0).astype(f32) * f32(31.0)
    c5 = np.floor(ca + f32(0.5)).astype(f32)
    num = c5 * lum6[:, None]
    num = num * f32(255.0)
    q = num + f32(0.5)
    o8 = np.floor(q / f32(1953.0)).astype(f32)
    return (o8 * R255).astype(f32)


# --------------------------------------------------- C034: the DS toon table

def _master_node(mat):
    graph = getattr(mat, 'graph', None) if mat is not None else None
    if not graph:
        return None
    for node in graph.get('nodes', {}).values():
        if node.get('bl_idname') == 'HALCYON_ShaderNode':
            return node
    return None


def ds_toon_table(mat):
    """(32,) float32 of 6-bit values: the material's Toon Size and Toon
    Steps sampled at 32 stops with HARD edges (smooth 0.0: the DS had no
    smooth band), each 5-bit stop expanded x ? 2x + 1 : 0. A LINKED Toon
    Size socket raises Refusal (the table is one per material)."""
    size, steps = 0.5, 2.0
    node = _master_node(mat)
    if node is not None:
        for sock in node.get('inputs', ()):
            if sock.get('name') == 'Toon Size':
                if sock.get('link'):
                    raise Refusal('Toon Size linked: the DS table is one '
                                  'per material')
                try:
                    size = float(sock.get('default', 0.5))
                except (TypeError, ValueError):
                    size = 0.5
        try:
            steps = float(node.get('props', {}).get('toon_steps', 2) or 2)
        except (TypeError, ValueError):
            steps = 2.0
    else:
        size = float(getattr(mat, 'toon_size', 0.5) or 0.5) \
            if mat is not None else 0.5
        steps = float(getattr(mat, 'toon_steps', 2.0) or 2.0) \
            if mat is not None else 2.0
    x = np.arange(32, dtype=f32) / f32(31.0)
    t = np.asarray(SH.diffuse_toon(x, f32(size), f32(0.0), f32(steps)), f32)
    g5 = np.rint(np.clip(t, 0.0, 1.0) * f32(31.0)).astype(f32)
    return _expand5(g5)


def cb_ds_toon(red, A, tab, highlight):
    """The DS mode-2 polygon: the carried lit red (6-bit grid) indexes the
    32-entry table (half up: red6 >> 1); TOON modulates the entry with
    the texel, HIGHLIGHT adds it once more, truncated at 63."""
    red = np.asarray(red, f32)
    m = red * f32(31.0)
    idx = np.clip(np.floor(m + f32(0.5)), 0.0, 31.0).astype(np.int32)
    cs = np.asarray(tab, f32)[idx]
    ta = np.clip(np.asarray(A, f32)[:, :3], 0.0, 1.0).astype(f32) * f32(63.0)
    ta = ta + f32(0.5)
    t6 = np.floor(ta).astype(f32)
    v = (t6 + f32(1.0)) * (cs + f32(1.0))[:, None]
    v = v - f32(1.0)
    c6 = np.floor(v / f32(64.0)).astype(f32)
    if highlight:
        c6 = np.minimum(c6 + cs[:, None], f32(63.0)).astype(f32)
    return (c6 * R63).astype(f32)


# --------------------------------------------- C057: the Mega Drive S/H DAC

RAMP_NORMAL = (0, 52, 87, 116, 144, 172, 206, 255)
RAMP_SHADOW = (0, 29, 52, 70, 87, 101, 116, 130)
RAMP_HIGHLIGHT = (130, 144, 158, 172, 187, 206, 228, 255)
RAMPS = np.array([RAMP_NORMAL, RAMP_SHADOW, RAMP_HIGHLIGHT], np.float32)


def cb_mdsh(cls, A):
    """The VDP's shadow / highlight operator: the albedo crushed to 3 bits
    per channel (half to even) and read through the class's measured DAC
    ramp -- a table, never a multiply."""
    c = np.floor(np.asarray(cls, f32) + f32(0.5))
    c = np.clip(c, 0.0, 2.0).astype(np.int32)
    c3 = np.rint(np.clip(np.asarray(A, f32)[:, :3], 0.0, 1.0) * f32(7.0))
    c3 = np.clip(c3, 0.0, 7.0).astype(np.int32)
    o8 = RAMPS[c[:, None], c3].astype(f32)
    return (o8 * R255).astype(f32)


# ------------------------------------------------- C049: the Super FX plot

def plot_ss(st):
    """The supersample factor per axis (render()'s own expression)."""
    if str(getattr(st, 'aa_mode', 'NONE')) == 'SUPERSAMPLE':
        return max(int(np.round(np.sqrt(max(int(st.aa_samples), 1)))), 1)
    return 1


def plot_palette(st):
    """(K, 3) float32, K <= 16: the frame's FIXED palette, or Refusal."""
    from . import palette as PA
    mode = str(getattr(st, 'palette_mode', 'ADAPTIVE'))
    if mode == 'CUSTOM':
        cols = getattr(st, 'palette_colors', None)
        if not cols:
            raise Refusal('the plot pairs entries of a FIXED palette; '
                          'palette_mode CUSTOM with no palette image is '
                          'built from the finished frame')
        P = np.asarray(cols, f32).reshape(-1, 3)
    elif mode == 'ADAPTIVE' or mode not in PA.FIXED_PALETTES:
        raise Refusal('the plot pairs entries of a FIXED palette; '
                      f'palette_mode {mode} is built from the finished frame')
    else:
        P = np.asarray(PA.FIXED_PALETTES[mode](int(st.palette_size)), f32)
        P = P.reshape(-1, 3)
    if P.shape[0] > 16:
        raise Refusal('the Super FX dither flag is ignored above 16 colours '
                      f'(4 bpp): palette_mode {mode} has {P.shape[0]} entries')
    return P


def plot_gate(st):
    """The three Super FX gates in one place (the probe's twin)."""
    P = plot_palette(st)
    g = float(getattr(st, 'gamma', 1.0))
    cm = str(getattr(st, 'color_management', 'NONE'))
    if cm != 'NONE' or g != 1.0:
        raise Refusal(f'the plot writes palette entries, and gamma {g:g} / '
                      f'color_management {cm} would move them off the '
                      'palette')
    return P


def _plot_pairs(colors, P):
    """Per colour (F, 3) the (i, j) pair, i <= j, whose DOUBLED mean is
    nearest in 8-bit RGB (exact integers); ties to the smallest i, then j."""
    P = np.asarray(P, f32).reshape(-1, 3)
    K = int(P.shape[0])
    c8 = np.rint(np.clip(np.asarray(colors, f32), 0.0, 1.0) * f32(255.0))
    pal8 = np.rint(P * f32(255.0))
    ii, jj = np.triu_indices(K)
    s = (pal8[ii] + pal8[jj]).astype(np.int64)                # (M, 3)
    d = (2 * c8.astype(np.int64))[:, None, :] - s[None, :, :]
    dist = (d * d).sum(axis=2)                                # (F, M)
    m = np.argmin(dist, axis=1)
    return ii[m].astype(np.int64), jj[m].astype(np.int64)


def _plot_face_albedo(job, faces, st):
    """The ALBEDO pass at the centroid of each face, cached on the job for
    the frame (created lazily; dropped with the job)."""
    from .render import _shade_chunked
    cache = getattr(job, '_plot_albedo', None)
    if cache is None:
        cache = job._plot_albedo = {}
    faces = np.asarray(faces, np.int64)
    need = np.array([f for f in faces if int(f) not in cache], np.int64)
    if need.size:
        saved = getattr(job, 'rate_mode', None)
        try:
            job.rate_mode = 'ALBEDO'
            bary = np.full((need.size, 3), 1.0 / 3.0, np.float32)
            alb = _shade_chunked(job, need.astype(np.int32), bary, None,
                                 None, None, None, st)[:, :3]
        finally:
            job.rate_mode = saved
        for k, f in enumerate(need):
            cache[int(f)] = np.asarray(alb[k], f32)
    return np.stack([cache[int(f)] for f in faces]).astype(f32) \
        if faces.size else np.zeros((0, 3), f32)


def plot_pairs_for_faces(job, faces, L_faces, st, P):
    """(Pi, Pj) (F, 3) float32 palette values for `faces` lit by `L_faces`
    (the FACE light over white, rgb)."""
    A = _plot_face_albedo(job, faces, st)
    c = np.clip(np.asarray(L_faces, f32)[:, :3] * A, 0.0, 1.0)
    i, j = _plot_pairs(c, P)
    return P[i], P[j]


def plot_parity(px, py, ss):
    """The output-pixel parity of internal pixels (row 0 = bottom on both
    roads): floor((p + 0.5) / ss) per axis, then (xo + yo) mod 2."""
    xo = np.floor((np.asarray(px, f32) + f32(0.5)) / f32(ss))
    yo = np.floor((np.asarray(py, f32) + f32(0.5)) / f32(ss))
    ps = xo + yo
    return (ps - f32(2.0) * np.floor(ps / f32(2.0))).astype(f32)


def pack_face_corners(out, sel, col, job, st, mi):
    """vertex_light_corners' FACE packing: `col` to all three corners for
    every model; for a SUPERFX_PLOT material that passes the gates, Pi to
    corner 0, Pj to corner 1 and `col` to corner 2 (alpha untouched)."""
    for c in range(3):
        out[sel * 3 + c] = col
    models = _models_of(job, st)
    model = models[mi] if mi < len(models) else None
    if model != 'SUPERFX_PLOT':
        return
    try:
        P = plot_gate(st)
    except Refusal as r:
        refuse(st, 'SUPERFX_PLOT', str(r))
        return
    Pi, Pj = plot_pairs_for_faces(job, sel, col[:, :3], st, P)
    out[sel * 3 + 0, :3] = Pi
    out[sel * 3 + 1, :3] = Pj


def effective_model(model, mat, st):
    """The model a material actually shades with after its refusals: a
    DS_TOON / DS_HIGHLIGHT with a linked Toon Size shades as DS_FIXED, a
    SUPERFX_PLOT that fails its gates as FLAT."""
    if model in _SPEC_RED:
        try:
            ds_toon_table(mat)
        except Refusal as r:
            refuse(st, model, str(r), fallback='DS_FIXED')
            return 'DS_FIXED'
    elif model == 'SUPERFX_PLOT':
        try:
            plot_gate(st)
        except Refusal as r:
            refuse(st, model, str(r))
            return 'FLAT'
    return model


def probe_gate(model, mat, st):
    """The GPU probe's twin of `effective_model`: raises Refusal with the
    CPU's text, so the pass refuses by name and the CPU draws the material
    on its fallback road."""
    if model in _SPEC_RED:
        return ds_toon_table(mat)
    if model == 'SUPERFX_PLOT':
        plot_gate(st)
    return None


# ------------------------------------------------------------- the hook

def recombine(job, tri_idx, light, c0, c1, c2, bary, alb, px, py, st):
    """The R:5554 hook: `None` unless a period model is present among the
    chunk's materials; else the per-pixel colour (n, 3) float32 with the
    plain product `light * albedo` on every non-period fragment."""
    scene = job.scene
    mats = getattr(scene, 'materials', None) or []
    models = _models_of(job, st)
    eff = [effective_model(m, mats[i] if i < len(mats) else None, st)
           for i, m in enumerate(models)]
    if not any(m in RECOMBINE_MODELS for m in eff):
        return None
    tri_idx = np.asarray(tri_idx)
    mat_idx = _mat_of(job, tri_idx)
    light = np.asarray(light, f32)
    alb = np.asarray(alb, f32)
    out = (light[:, :3] * alb[:, :3]).astype(f32)
    L = light[:, :3]
    A = alb[:, :3]
    for mi, model in enumerate(eff):
        if model not in RECOMBINE_MODELS:
            continue
        sel = np.nonzero(mat_idx == mi)[0]
        if sel.size == 0:
            continue
        kind = LIGHTING_COMBINE.get(model, model)
        if kind == 'luma64':
            out[sel] = cb_luma64(L[sel], A[sel])
        elif kind == 'ds':
            out[sel] = cb_ds(L[sel], A[sel])
        elif model == 'PS1_MODULATE':
            out[sel] = cb_ps1(L[sel], A[sel])
        elif model == 'SATURN_ADD':
            out[sel] = cb_saturn(L[sel], A[sel])
        elif model == 'N64_COMBINE':
            out[sel] = cb_n64(L[sel], A[sel])
        elif model == 'S22_MODULATE':
            out[sel] = cb_s22(L[sel], A[sel])
        elif model == 'PS2_HIGHLIGHT':
            out[sel] = cb_ps2hl(L[sel], light[sel, 3], A[sel])
        elif model == 'D3D_SEPARATE_SPEC':
            out[sel] = cb_d3dspec(L[sel], light[sel, 3], A[sel])
        elif model == 'PCX_INTENSITY':
            if c0 is None:
                # FACE rate: the three corners are equal, the item is the
                # plain product (RATE_FIXED keeps PCX at VERTEX)
                continue
            out[sel] = cb_pcx(c0[sel], c1[sel], c2[sel], bary[sel], A[sel])
        elif model in _SPEC_RED:
            tab = ds_toon_table(mats[mi] if mi < len(mats) else None)
            out[sel] = cb_ds_toon(light[sel, 3], A[sel], tab,
                                  model == 'DS_HIGHLIGHT')
        elif model == 'MEGA_DRIVE_SH':
            out[sel] = cb_mdsh(light[sel, 3], A[sel])
        elif model == 'SUPERFX_PLOT':
            P = plot_gate(st)
            faces, first, inv = np.unique(tri_idx[sel], return_index=True,
                                          return_inverse=True)
            Pi, Pj = plot_pairs_for_faces(job, faces, L[sel][first], st, P)
            if px is None or py is None:
                out[sel] = Pi[inv]
            else:
                par = plot_parity(np.asarray(px)[sel], np.asarray(py)[sel],
                                  plot_ss(st))
                out[sel] = np.where((par > 0.5)[:, None], Pj[inv], Pi[inv])
    return out
