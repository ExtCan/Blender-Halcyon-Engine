"""Fog: the depth-cue curves as pure float32 functions (R251, 1.90.0).

`render.apply_fog` is the caller; this module holds the arithmetic so the
tests can call every curve on a depth ramp and the GLSL twin
(`gpu/material.FOG_GLSL`, `hal_fog`) can mirror it statement for
statement. Three roads live here:

* `legacy(...)` -- the 1.89.0 `apply_fog` body, moved verbatim (the same
  float32 order: NumPy 2's weak scalars keep every `python_float op
  float32_array` at float32). Every 1.89.0 frame is bitwise by
  construction because this is the same code.
* `factor(...)` / `apply(...)` -- the extended road, taken only when
  `extended(st)` is true (a non-default new dial): the four legacy curves
  re-implemented in the same float32 order, plus the period mechanisms --
  the PlayStation GTE's hyperbolic 1/z cue (F001), 3dfx Glide's 64-entry
  w-table with the Voodoo2 fog dither (F002), the PowerVR2's 128-entry
  log table (F003), the Nintendo DS's 32-entry density table (F022) and
  Direct3D's z-buffer pixel fog (F007).
* `pack_fog_texels(job)` -- the VALUES the GPU pass reads through the
  `hal_fogtab` data texture (texelFetch only), one (1, 512, 4) float32
  row, packed with `np.float32` of the SAME Python expressions the CPU
  evaluates. The texel map is lighting.md section 0's; every slot below
  cites it.

Wave 2 (LIGHT-A2) adds, on the same extended road: the GameCube's
horizontal fog range adjust (F004, `gc_adjust_knots` / `range_adjust`),
Namco System 21's one-value-per-polygon fog (F005, `face_depth`), Model 3
/ System 22's per-material burn-through, bias and second bank (F006,
`burn_bias_bank`, `bank_consts`), LightWave's backdrop fog target (F008,
`fog_target`, `backdrop_of`), POV-Ray's ground fog integral and its sky
closed form (F009, `ground_fog_integral`, `ground_fog_sky`) and POV-Ray's
turbulent fog (F010, `turbulence`). Every one of them is mirrored line
for line by `hal_fog` (gpu/material.FOG_GLSL) and, for the sky, by
`hal_ground_fog_sky` (gpu/sky.py).

Tie rules, named once: `np.round` (half-to-even, GLSL roundEven) for the
per-vertex 1/8 quantisation and the bank index; `floor` everywhere else;
integer shifts for the hardware tables; the GTE's `>> 12` is a floor;
the ground fog's five-way select in its stated order.
"""

import math

import numpy as np

from . import dither as DI
from . import shading as SH

F32 = np.float32
LEGACY_MODES = ('LINEAR', 'EXP', 'EXP2', 'TABLE16')
#: the hardware tables the accelerators indexed instead of a curve
TABLES = ('VOODOO64', 'PVR128', 'DS32')
#: the hal_fogtab row length (texels); the map is lighting.md section 0
TEXELS = 512
#: section 0's map, texel 227 = (vs.x, vs.y, vs.z, 0) the axis viewer (F011);
#: gpu/material.AXIS_VIEWER_TEXEL names the same slot for the texelFetch
AXIS_VIEWER_TEXEL = 227
#: MAME's dither_matrix_4x4 for the Voodoo2 fog dither, as integers 0..15
#: (DI.BAYER4 holds k/16 in float32, so rint(k/16 * 16) is exact)
M4 = np.rint(DI.BAYER4 * 16.0).astype(np.int64)

_TABLE_CACHE = {}


# ------------------------------------------------------------------ readers

def mode_of(st):
    return str(getattr(st, 'fog_mode', 'LINEAR') or 'LINEAR')


def table_of(st):
    """The hardware table, or NONE. Inert (read as NONE) under GROUND
    fog (F009, wave 2): a ray integral has no depth curve to fill it."""
    t = str(getattr(st, 'fog_table', 'NONE') or 'NONE')
    if mode_of(st) == 'GROUND':
        return 'NONE'
    return t


def depth_of(st):
    """W (eye depth) or Z (Direct3D's post-projection z, F007). Z applies
    to the four legacy curves only; the console modes and the hardware
    tables keep eye depth (their machines' own domain)."""
    z = str(getattr(st, 'fog_depth', 'W') or 'W')
    if z == 'Z' and mode_of(st) in LEGACY_MODES and table_of(st) == 'NONE':
        return 'Z'
    return 'W'


def vertex_quantised(st):
    """Rule (vi): Per-Vertex Fog's 1/8 rounding is SKIPPED by every mode
    or table that carries its own quantisation (GTE_1Z's 12-bit floor,
    the three tables, per-polygon fog), so the myth never stacks on a
    mechanism."""
    return (bool(getattr(st, 'fog_vertex', False))
            and mode_of(st) in LEGACY_MODES
            and table_of(st) == 'NONE'
            and not bool(getattr(st, 'fog_face', False)))


def extended(st, surf=None):
    """True when any NEW dial is off its default, i.e. the frame needs
    the extended road. False for every 1.89.0 field set, so those frames
    take `legacy` and stay bitwise. `fog_dither` alone is not a dial:
    only the 64-entry table reads it."""
    if mode_of(st) not in LEGACY_MODES:
        return True
    if table_of(st) != 'NONE':
        return True
    if depth_of(st) != 'W':
        return True
    # wave 2 (LIGHT-A2): F004 / F005 / F008 / F006 / F010
    if bool(getattr(st, 'fog_range_adjust', False)):
        return True
    if bool(getattr(st, 'fog_face', False)):
        return True
    if source_of(st) == 'BACKDROP':
        return True
    if float(getattr(st, 'fog_ambient', 1.0)) != 1.0:
        return True
    if bank1_valid(st):
        return True
    if turb_on(st):
        return True
    if spot_on(st):
        return True
    if surf is not None and material_dials(surf):
        return True
    return False


# ------------------------------------------------------------- the constants

def span32(st):
    return F32(max(float(st.fog_end) - float(st.fog_start), 1e-5))


def gte_consts(st):
    """(A, B, A32, B32): the GTE's affine-in-1/z factor, 0 at Fog Start
    and 1 at Fog End -- float64 once, rounded once (F001)."""
    s = max(float(st.fog_start), 1e-5)
    e = max(float(st.fog_end), s + 1e-5)
    A = 1.0 / (1.0 / e - 1.0 / s)               # negative
    B = -A / s
    return A, B, F32(A), F32(B)


def fill_curve(st):
    """c(w): the fog_mode curve in float64, filling a hardware table at
    each entry's own depth (section 0's fill-curve rule; TABLE16 is read
    as LINEAR; GROUND has no curve and fills as LINEAR -- inert anyway)."""
    mode = mode_of(st)
    start = float(st.fog_start)
    end = float(st.fog_end)
    span = max(end - start, 1e-5)
    dens = float(st.fog_density)
    if mode == 'EXP':
        return lambda w: 1.0 - math.exp(-dens * w)
    if mode == 'EXP2':
        return lambda w: 1.0 - math.exp(-((dens * w) ** 2))
    if mode == 'GTE_1Z':
        A, B, _a, _b = gte_consts(st)
        return lambda w: min(max(A / w + B, 0.0), 1.0)
    return lambda w: min(max((w - start) / span, 0.0), 1.0)


def _table_key(st, which):
    return (which, mode_of(st), float(st.fog_start), float(st.fog_end),
            float(st.fog_density))


def voodoo64_table(st):
    """Glide's 64-entry table: (T[64], delta[64]) as int64.

    Entry i sits at w_i = 2^(3 + i/4) / (8 - i%4) (guFogTableIndexToW);
    T[i] = round(255 * c(w_i)); delta[i] is Glide's 6.2 register field --
    the entry difference in bits 7:2, i.e. FOUR times the difference with
    the documented ceiling of 63 per interval (a steeper fill saturates
    exactly as the hardware's did; disclosed)."""
    key = _table_key(st, 'VOODOO64')
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    c = fill_curve(st)
    T = np.zeros(64, np.int64)
    for i in range(64):
        w = 2.0 ** (3 + (i >> 2)) / (8 - (i & 3))
        T[i] = int(round(255.0 * c(w)))
    delta = np.zeros(64, np.int64)
    for i in range(64):
        diff = int(T[min(i + 1, 63)] - T[i])
        delta[i] = min(max(diff, 0), 63) << 2
    _TABLE_CACHE[key] = (T, delta)
    return T, delta


def pvr_density(st):
    """The CLX2's fog density register: a 1.7 mantissa with an 8-bit
    power-of-two exponent -- Fog End quantised to its top 16 bits."""
    den32 = F32(max(float(st.fog_end), 1e-3))
    bits = den32.view(np.uint32) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def pvr128_table(st):
    """The PowerVR2's 129 entries (T[129] int64) and the density register
    den_q: entry i sits at KallistiOS's InverseW_Depth(i) =
    den_q / (2^(i>>4) * ((i&15)+16)/16); entry 0 is the farthest."""
    key = _table_key(st, 'PVR128')
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    den_q = pvr_density(st)
    c = fill_curve(st)
    T = np.zeros(129, np.int64)
    for i in range(129):
        w = float(den_q) / (2.0 ** (i >> 4) * ((i & 15) + 16) / 16.0)
        T[i] = int(round(255.0 * c(w)))
    _TABLE_CACHE[key] = (T, den_q)
    return T, den_q


def ds32_table(st):
    """The DS's FOG_TABLE: 32 densities of 7 bits at equal steps from Fog
    Start (FOG_OFFSET) to Fog End (the last register entry), replicated
    to melonDS's 34-entry form T[32] = T[33] = T[31]. Returns
    (T[34] int64, start32, step32) with step32 = span / 31."""
    key = _table_key(st, 'DS32')
    hit = _TABLE_CACHE.get(key)
    if hit is not None:
        return hit
    start = float(st.fog_start)
    end = float(st.fog_end)
    c = fill_curve(st)
    T = np.zeros(34, np.int64)
    for i in range(32):
        w = start + i * (end - start) / 31.0
        T[i] = int(round(127.0 * c(w)))
    T[32] = T[33] = T[31]
    start32 = F32(start)
    step32 = F32(max(end - start, 1e-5) / 31.0)
    _TABLE_CACHE[key] = (T, start32, step32)
    return T, start32, step32


def projection_consts(scene, width, height):
    """F007: (n32, k32, ortho) from the projection THE RASTER USED --
    perspective k32 = far/(far-near) so zd = k32*(1 - n32/d) is D3D's own
    z' = f/(f-n)*(1 - n/z); orthographic k32 = 1/(far-near) so
    zd = (d - n32)*k32, linear as an ortho z-buffer is."""
    from . import render as _R
    cam = getattr(scene, 'camera', None)
    proj = _R.camera_matrices(cam, int(width), int(height))[1]
    p = np.asarray(proj, np.float64)
    a, b = float(p[2, 2]), float(p[2, 3])
    ortho = abs(a - 1.0) < 1e-12 or abs(a + 1.0) < 1e-12 \
        or str(getattr(cam, 'type', 'PERSP')) == 'ORTHO'
    if ortho:
        if abs(a) < 1e-30:
            return F32(0.0), F32(0.0), True
        n = (b + 1.0) / a
        f = (b - 1.0) / a
        return F32(n), F32(1.0 / max(f - n, 1e-30)), True
    n = b / (a - 1.0)
    f = b / (a + 1.0)
    return F32(n), F32(f / max(f - n, 1e-30)), False


# ------------------------------------------------------------- the per-point

def _c(a):
    return np.ascontiguousarray(np.asarray(a, np.float32))


def z_from_w(d, n32, k32, ortho):
    """F007: eye depth -> the z-buffer's depth, one op per statement."""
    if ortho:
        t = d - n32
        return t * k32
    dd = np.maximum(d, F32(1e-6))
    q = n32 / dd
    t = F32(1.0) - q
    return k32 * t


def gte_1z(d, A32, B32):
    """F001: the GTE's depth cue. f = 1 - floor(clamp(A/z + B) * 4096)/4096."""
    dd = np.maximum(d, F32(1e-6))
    q = A32 / dd
    t = q + B32
    t = np.clip(t, F32(0.0), F32(1.0))
    ir = np.floor(t * F32(4096.0))
    t = ir * F32(0.000244140625)
    return F32(1.0) - t


def _pow2_ladder(x, top):
    """e = the number of thresholds 2^k (k = 1..top) x reaches; p = 2^e."""
    e = np.zeros(x.shape, np.int64)
    for k in range(1, top + 1):
        e += (x >= F32(2.0 ** k)).astype(np.int64)
    p = np.ones(x.shape, np.float32)
    for k in range(1, top + 1):
        p = np.where(e >= k, p * F32(2.0), p).astype(np.float32)
    return e, p


def voodoo64_lookup(d, T, delta, px=None, py=None, dither=False):
    """F002: Glide's table on 1/w with MAME's apply_fogging shift chain.

    w = clamp(d, 1, 65535); the exponent and top two mantissa bits index
    the 64 entries, the next 8 mantissa bits interpolate through the 6.2
    delta; Voodoo2's fog dither adds the 4x4 matrix at the pixel before
    the final shift. Returns f = 1 - fogblend/256."""
    w = np.maximum(d, F32(1.0))
    w = np.minimum(w, F32(65535.0))
    e, p = _pow2_ladder(w, 15)
    r = p / w
    m = F32(1.0) - r
    m = m * F32(8.0)
    fi = np.floor(m)
    i = 4 * e + fi.astype(np.int64)
    frac = m - fi
    frac8 = np.floor(frac * F32(256.0)).astype(np.int64)
    dv = (delta[i] * frac8) >> 6
    if dither and px is not None and py is not None:
        dv = dv + M4[np.asarray(py, np.int64) & 3, np.asarray(px, np.int64) & 3]
    dv = dv >> 4
    fb = T[i] + dv + 1
    f_op = fb.astype(np.float32) * F32(0.00390625)
    return F32(1.0) - f_op


def pvr128_lookup(d, T, den_q):
    """F003: the CLX2's log table. x = clamp(den_q / d, 1, 255.999985);
    the exponent (8 octaves) and top 4 mantissa bits index, the next 8
    mantissa bits blend two 8-bit densities; f = (255 - a) / 255."""
    dd = np.maximum(d, F32(1e-6))
    invw = F32(1.0) / dd
    x = den_q * invw
    x = np.clip(x, F32(1.0), F32(255.999985))
    e, p = _pow2_ladder(x, 7)
    m = x / p
    mm = m - F32(1.0)
    m4 = np.floor(mm * F32(16.0)).astype(np.int64)
    idx = 16 * e + m4
    f8 = np.floor(mm * F32(4096.0)).astype(np.int64) - m4 * 256
    hi = T[idx]
    lo = T[idx + 1]
    a = (lo * f8 + hi * (255 - f8)) >> 8
    return (255 - a).astype(np.float32) / F32(255.0)


def ds32_lookup(d, T, start32, step32):
    """F022: melonDS's CalculateFogDensity on eye depth. Entry = floor of
    (d - start)/step, the 17-bit fraction interpolates two 7-bit
    densities, 127 reads as 128; f = 1 - D/128."""
    t = d - start32
    t = t / step32
    t = np.clip(t, F32(0.0), F32(32.0))
    fi = np.floor(t)
    fr = t - fi
    f17 = np.floor(fr * F32(131072.0)).astype(np.int64)
    i = fi.astype(np.int64)
    dsum = T[i] * (131072 - f17) + T[i + 1] * f17
    D = dsum >> 17
    D = np.where(D >= 127, 128, D)
    f_op = D.astype(np.float32) * F32(0.0078125)
    return F32(1.0) - f_op


def legacy_curve(d, st, bank=None):
    """The four 1.89.0 curves on a clamped depth, in apply_fog's float32
    order, before the shared clamp (the extended road's copy). `bank`
    (F006, `bank_consts`) hands the per-point Start / End / span when
    some points read System 22's bank 1; EXP / EXP2 read density only."""
    mode = mode_of(st)
    if bank is not None and mode in ('LINEAR', 'TABLE16'):
        if mode == 'LINEAR':
            return (bank['end'] - d) / bank['span']
        t = np.clip((d - bank['start']) / bank['span'], F32(0.0), F32(1.0))
        return F32(1.0) - np.floor(t * F32(16.0)) / F32(16.0)
    if mode == 'LINEAR':
        return (F32(float(st.fog_end)) - d) / span32(st)
    if mode == 'EXP':
        return np.exp(F32(-float(st.fog_density)) * d)
    if mode == 'EXP2':
        t = F32(float(st.fog_density)) * d
        return np.exp(-(t * t))
    if mode == 'TABLE16':
        t = np.clip((d - F32(float(st.fog_start))) / span32(st), F32(0.0),
                    F32(1.0))
        return F32(1.0) - np.floor(t * F32(16.0)) / F32(16.0)
    # any other mode (GROUND before wave 2's integral) reads as LINEAR,
    # exactly the GLSL's else branch
    return (F32(float(st.fog_end)) - d) / span32(st)


def _bands(f, st):
    fb = int(getattr(st, 'fog_bands', 0) or 0)
    if fb >= 2:
        f = np.floor(f * fb + 0.5) / fb
    return f


def _height(f, st, P):
    if getattr(st, 'fog_height', False) and P is not None \
            and mode_of(st) != 'GROUND':
        above = np.maximum(P[:, 2] - float(st.fog_height_top), 0.0)
        h = np.exp(-above * max(float(st.fog_height_falloff), 0.0))
        f = np.where(h >= 1.0, f, 1.0 - (1.0 - f) * h)
    return f


def _size_of(st, ctx):
    w = getattr(ctx, 'width', None) if ctx is not None else None
    h = getattr(ctx, 'height', None) if ctx is not None else None
    if not w or not h:
        w = int(getattr(st, 'resolution_x', 96) or 96)
        h = int(getattr(st, 'resolution_y', 72) or 72)
    return int(w), int(h)


def factor(depth, st, scene, P=None, ctx=None, surf=None):
    """The transmittance f per point on the extended road: the shared
    depth, the selected curve or table, the shared clamp, bands and
    height layer -- `hal_fog` mirrors this statement for statement.

    Wave 2's slots, in the order both roads run them: the polygon's mean
    corner depth replaces the pixel's (F005), the GameCube column secant
    scales it (F004), the per-vertex rounding, the shared max, the
    z-buffer conversion (F007), the turbulence on the mode's distance
    (F010), then the curve -- the ground integral (F009) or a table or a
    curve on the point's bank (F006) -- the clamp, the material's burn /
    bias (F006), bands, height."""
    d = _c(depth)
    px = getattr(ctx, 'px', None) if ctx is not None else None
    py = getattr(ctx, 'py', None) if ctx is not None else None
    mode = mode_of(st)
    eye = _eye_of(ctx, scene)
    if face_on(st) and ctx is not None \
            and getattr(ctx, 'tri', None) is not None \
            and getattr(ctx, 'scene', None) is not None:
        # [fog_face, F005]: one depth per polygon, the mean of its three
        # corners' planar depths (the pixel's own depth is discarded)
        d = face_depth(ctx)
    if range_adjust_on(st, scene) and px is not None:
        # [range_adjust, F004]: the column's secant scales the depth
        w, h = _size_of(st, ctx)
        d = range_adjust(d, px, st, scene, w, h)
    if vertex_quantised(st):
        d = np.round(d * 8.0) / 8.0
    d = np.maximum(d, 0.0).astype(np.float32)
    if depth_of(st) == 'Z':
        w, h = _size_of(st, ctx)
        n32, k32, ortho = projection_consts(scene, w, h)
        d = z_from_w(d, n32, k32, ortho)
    turb = turb_on(st)
    if turb and mode != 'GROUND' and P is not None and eye is not None:
        # [turbulence, F010]: POV's one read at the fogged segment's
        # middle scales the distance the curve reads
        d = turbulence(d, P, eye, st)
    table = table_of(st)
    bank = bank_consts(st, surf)
    if mode == 'GROUND' and P is not None and eye is not None:
        # [GROUND, F009]: the atan integral along the eye ray
        f = ground_fog_integral(P, eye, st, turb=turb)
    elif table == 'VOODOO64':
        T, delta = voodoo64_table(st)
        px = getattr(ctx, 'px', None) if ctx is not None else None
        py = getattr(ctx, 'py', None) if ctx is not None else None
        f = voodoo64_lookup(d, T, delta, px, py,
                            bool(getattr(st, 'fog_dither', False)))
    elif table == 'PVR128':
        T, den_q = pvr128_table(st)
        f = pvr128_lookup(d, T, den_q)
    elif table == 'DS32':
        T, start32, step32 = ds32_table(st)
        f = ds32_lookup(d, T, start32, step32)
    elif mode == 'GTE_1Z':
        _a, _b, A32, B32 = gte_consts(st)
        if bank is not None:
            f = gte_1z(d, bank['A'], bank['B'])
        else:
            f = gte_1z(d, A32, B32)
    else:
        f = legacy_curve(d, st, bank)
    f = np.clip(f, 0.0, 1.0).astype(np.float32)
    # [material fog, F006]: the material's bias and burn-through
    f = burn_bias_bank(f, surf)
    f = _bands(f, st)
    f = _height(f, st, P)
    return np.asarray(f, np.float32)


def fog_factor(depth, st, scene=None, P=None, ctx=None, surf=None):
    """The design's name for `factor` (tests and later packs call it)."""
    return factor(depth, st, scene, P, ctx, surf)


def fog_dither_voodoo2(px, py):
    """The Voodoo2 fog dither term at a pixel: MAME's 4x4 matrix entry."""
    return M4[np.asarray(py, np.int64) & 3, np.asarray(px, np.int64) & 3]


def table_lookup(st, w, px=None, py=None):
    """f at depth w through the hardware table `st.fog_table` names
    (VOODOO64 / PVR128 / DS32), the tests' one door to the three lookups."""
    d = _c(w)
    table = table_of(st)
    if table == 'VOODOO64':
        T, delta = voodoo64_table(st)
        return voodoo64_lookup(d, T, delta, px, py,
                               bool(getattr(st, 'fog_dither', False)))
    if table == 'PVR128':
        T, den_q = pvr128_table(st)
        return pvr128_lookup(d, T, den_q)
    if table == 'DS32':
        T, start32, step32 = ds32_table(st)
        return ds32_lookup(d, T, start32, step32)
    raise ValueError(f'no hardware fog table selected ({table})')


def blend(rgb, f, st, ctx=None):
    """The shared blend toward the fog target: the Fog Colour x fog
    ambient (F006), or the backdrop / world colour per point (F008,
    `fog_target`); `rgb * f + col * (1 - f)` in the 1.89.0 order."""
    f = np.asarray(f, np.float32)[:, None]
    col = fog_target(st, ctx, f.shape[0])
    sf = getattr(ctx, 'spot_fog', None) if ctx is not None else None
    if sf is not None and spot_on(st):
        # [spot fog, F015]: Supermodel's spotFogColor x fogAttenuation --
        # the Model 3 screen spotlights' lobe (light_surface's ctx.spot_fog,
        # colE * (en * el) summed over the lamps) times Spotlight Fog,
        # added to the target; one multiply, one add, on both roads
        sp = F32(float(getattr(st, 'fog_spot', 0.0) or 0.0)) * \
            np.asarray(sf, np.float32)
        col = (col + sp).astype(np.float32)
    return (rgb * f + col * (1.0 - f)).astype(np.float32)


def apply(rgb, depth, st, scene, P=None, ctx=None, surf=None):
    """The extended road's apply_fog: factor, then the shared blend."""
    rgb = _c(rgb)
    _cpu_notes(st, scene)
    f = factor(depth, st, scene, P, ctx, surf)
    return blend(rgb, f, st, ctx)


def legacy(rgb, depth, settings, scene, P=None):
    """The 1.89.0 apply_fog body, verbatim (the `vertex_rate` flag was
    dead -- called nowhere -- and is gone). Every frame with no new dial
    runs THIS, so it is bitwise the previous release by construction."""
    if settings.fog_vertex:
        # quantise the depth so the blend steps rather than sweeps, which is
        # what interpolating a per-vertex factor looks like on coarse geometry
        depth = np.round(depth * 8.0) / 8.0
    mode = settings.fog_mode
    d = np.maximum(depth, 0.0)
    if mode == 'LINEAR':
        f = (settings.fog_end - d) / max(settings.fog_end - settings.fog_start, 1e-5)
    elif mode == 'EXP':
        f = np.exp(-settings.fog_density * d)
    elif mode == 'EXP2':
        f = np.exp(-((settings.fog_density * d) ** 2))
    else:                                   # TABLE16 -- the fixed-function LUT
        t = np.clip((d - settings.fog_start) /
                    max(settings.fog_end - settings.fog_start, 1e-5), 0.0, 1.0)
        f = 1.0 - np.floor(t * 16.0) / 16.0
    f = np.clip(f, 0.0, 1.0)
    fb = int(getattr(settings, 'fog_bands', 0) or 0)
    if fb >= 2:
        # R223: banded depth fog -- the transmittance quantized to cel
        # steps, so distance reads as flat painted planes (the anime
        # background trick, and the PS1's own coarse fog tables).
        # Round-to-band, so near surfaces stay clear and the far end
        # saturates instead of everything sliding one band down
        f = np.floor(f * fb + 0.5) / fb
    if getattr(settings, 'fog_height', False) and P is not None:
        # h = 1 below the top (full fog), exp falloff above; the fog AMOUNT
        # (1 - f) scales by h, so high surfaces come out of the mist. Where
        # h is exactly 1 the transmittance passes through UNTOUCHED --
        # 1-(1-f) re-rounds f by an ulp, and an inert control must be inert
        above = np.maximum(P[:, 2] - float(settings.fog_height_top), 0.0)
        h = np.exp(-above * max(float(settings.fog_height_falloff), 0.0))
        f = np.where(h >= 1.0, f, 1.0 - (1.0 - f) * h)
    f = f[:, None]
    col = np.asarray(settings.fog_color, np.float32)[None, :]
    return (rgb * f + col * (1.0 - f)).astype(np.float32)


# ------------------------------------------------- wave 2 (LIGHT-A2) readers

_NOTED = set()


def _note_once(msg):
    """Print a named fallback once per process (the CPU road has no plan
    to hang a once-per-plan note on)."""
    if msg not in _NOTED:
        _NOTED.add(msg)
        print(msg)


def plan_notes(st, ortho):
    """The fog notes a plan prints once (gpu/shade._fog_structure): the
    GameCube range adjust under an orthographic camera (F004, A20) and
    a hardware table under Ground Fog (section 0's fill-curve rule)."""
    out = []
    if bool(getattr(st, 'fog', False)) and \
            bool(getattr(st, 'fog_range_adjust', False)) and ortho:
        out.append('[Halcyon] fog range adjust is inert under an orthographic '
                   'camera (GX builds its table from P00)')
    raw = str(getattr(st, 'fog_table', 'NONE') or 'NONE')
    if bool(getattr(st, 'fog', False)) and mode_of(st) == 'GROUND' \
            and raw != 'NONE':
        out.append(f'[Halcyon] fog table {raw} is inert under Ground Fog (it '
                   'integrates along the ray)')
    return out


def _cpu_notes(st, scene):
    cam = getattr(scene, 'camera', None) if scene is not None else None
    ortho = str(getattr(cam, 'type', 'PERSP')) == 'ORTHO'
    for msg in plan_notes(st, ortho):
        _note_once(msg)


def face_on(st):
    """F005: one fog value per polygon (Namco System 21)."""
    return bool(getattr(st, 'fog_face', False))


def source_of(st):
    """F008: FIXED (the Fog Colour) or BACKDROP (LightWave's Use Backdrop
    Color)."""
    s = str(getattr(st, 'fog_color_source', 'FIXED') or 'FIXED')
    return 'BACKDROP' if s == 'BACKDROP' else 'FIXED'


def turb_on(st):
    """F010: POV's fog turbulence, 0 = off."""
    return float(getattr(st, 'fog_turbulence', 0.0) or 0.0) > 0.0


def spot_on(st):
    """F015 (the fog half): Spotlight Fog > 0."""
    return float(getattr(st, 'fog_spot', 0.0) or 0.0) > 0.0


def spot_fog_of(job, px, py, depth):
    """A52: the screen spotlights' fog lobe colE * (en * el), summed over
    the frame's screen-spot lamps in select_lights' order, for points
    that ARE pixels -- what light_surface accumulates in ctx.spot_fog;
    `fog_for_points` calls it for the refusal road (a fog_cpu material
    under a Model 3 headlight). None when no lamp is a screen spot or
    the points have no pixel. Light linking masks are not applied here
    (a linked screen spot on a traced material: disclosed)."""
    from . import lights as LI
    from .render import camera_matrices
    st = job.settings
    scene = job.scene
    if px is None or py is None or depth is None:
        return None
    lights = [l for l in LI.select_lights(scene.lights, st)
              if getattr(l, 'screen_spot', False)
              and getattr(l, 'type', '') == 'SPOT']
    if not lights:
        return None
    vp = camera_matrices(scene.camera, int(job.width), int(job.height))[2]
    acc = None
    for light in lights:
        pr = LI.screen_spot_params(light, vp, int(job.width), int(job.height),
                                   scene.camera)
        en, el, _lobe = LI.screen_spot_lobe(px, py, depth, *pr)
        sign = -1.0 if getattr(light, 'negative', False) else 1.0
        col = np.asarray(light.color, np.float32) * np.float32(
            sign * float(light.energy) / (4.0 * np.pi))
        sf = col[None, :] * (en * el)[:, None]
        acc = sf.astype(np.float32) if acc is None \
            else (acc + sf).astype(np.float32)
    return acc


def bank1_valid(st):
    """F006: System 22's bank 1 is live while its End is past its Start."""
    return float(getattr(st, 'fog_bank1_end', 0.0) or 0.0) > \
        float(getattr(st, 'fog_bank1_start', 0.0) or 0.0)


def bank1_consts(st):
    """(start1_32, end1_32, span1_32) of bank 1 (texel 5, F006)."""
    s1 = float(getattr(st, 'fog_bank1_start', 0.0) or 0.0)
    e1 = float(getattr(st, 'fog_bank1_end', 0.0) or 0.0)
    return F32(s1), F32(e1), F32(max(e1 - s1, 1e-5))


class _Bank:
    """A Start/End pair wearing RenderSettings' names, for gte_consts."""

    def __init__(self, start, end):
        self.fog_start = float(start)
        self.fog_end = float(end)


def material_dials(surf):
    """F006: True when some point of `surf` carries a non-default fog
    dial (burn-through, bias, or bank 1). The GPU decides the same per
    material from its bake (`gpu/material.fog_material_dials`)."""
    if surf is None:
        return False
    burn = getattr(surf, 'fog_burn', None)
    bias = getattr(surf, 'fog_bias', None)
    bank = getattr(surf, 'fog_bank', None)
    if burn is not None and bool(np.any(np.asarray(burn) != 0.0)):
        return True
    if bias is not None and bool(np.any(np.asarray(bias) != 0.0)):
        return True
    if bank is not None and bool(np.any(np.rint(np.asarray(bank)) >= 1.0)):
        return True
    return False


def bank_of(surf):
    """The bank index per point: min(max(round(bank), 0), 1), half-to-
    even (GLSL roundEven), as an int array."""
    bank = np.asarray(getattr(surf, 'fog_bank', 0.0), np.float32)
    return np.clip(np.rint(bank), 0.0, 1.0).astype(np.int64)


def bank_consts(st, surf):
    """F006: the per-point curve constants when some point reads bank 1
    -- {'start', 'end', 'span', 'A', 'B'} float32 arrays -- or None
    (every point on bank 0, bank 1 inert, or a density-only curve). The
    GLSL selects texel 5 / texel 6 by the same rule
    (`(bk == 1) ? fp5 : fp0`, then `fpb.w > 0.5`)."""
    if surf is None or not bank1_valid(st):
        return None
    if mode_of(st) not in ('LINEAR', 'TABLE16', 'GTE_1Z'):
        return None
    if getattr(surf, 'fog_bank', None) is None:
        return None
    use = bank_of(surf) == 1
    if not bool(use.any()):
        return None
    s1, e1, sp1 = bank1_consts(st)
    s0 = F32(float(st.fog_start))
    e0 = F32(float(st.fog_end))
    sp0 = span32(st)
    out = {'start': np.where(use, s1, s0).astype(np.float32),
           'end': np.where(use, e1, e0).astype(np.float32),
           'span': np.where(use, sp1, sp0).astype(np.float32)}
    if mode_of(st) == 'GTE_1Z':
        _a, _b, A0, B0 = gte_consts(st)
        _a1, _b1, A1, B1 = gte_consts(_Bank(s1, e1))
        out['A'] = np.where(use, A1, A0).astype(np.float32)
        out['B'] = np.where(use, B1, B0).astype(np.float32)
    return out


def burn_bias_bank(f, surf):
    """F006's `[material fog]` lines after the curve's clamp: the opacity
    op = 1 - f, plus System 22's cz delta (Fog Bias), clamped, times
    (1 - Model 3's light modifier Fog Burn-Through); f = 1 - op. Only
    when the material carries a dial (an inert control must be inert:
    1 - (1 - f) re-rounds f by an ulp)."""
    if not material_dials(surf):
        return f
    n = f.shape[0]
    burn = np.broadcast_to(np.asarray(getattr(surf, 'fog_burn', 0.0),
                                      np.float32), (n,))
    bias = np.broadcast_to(np.asarray(getattr(surf, 'fog_bias', 0.0),
                                      np.float32), (n,))
    bank = np.broadcast_to(np.asarray(getattr(surf, 'fog_bank', 0.0),
                                      np.float32), (n,))
    # per POINT: exactly the GPU's per-material rule (a bake is constant
    # per material), so a dial-less material sharing a batch (the
    # refusal road's readback_surf) keeps its f untouched
    sel = (burn != 0.0) | (bias != 0.0) | (np.rint(bank) >= 1.0)
    op = F32(1.0) - f
    op = op + bias
    op = np.clip(op, F32(0.0), F32(1.0))
    bt = F32(1.0) - burn
    op = op * bt
    f2 = F32(1.0) - op
    return np.where(sel, f2, f).astype(np.float32)


def readback_surf(passes, mi):
    """The refusal road's `surf` (F006): per pixel the fog dials of the
    `fog_cpu` material it belongs to (`binds['fog_mat']`), or None when
    no such material carries a dial."""
    from types import SimpleNamespace
    fm = {int(m): tuple(b.get('fog_mat')) for m, _n, _s, b in passes
          if (b or {}).get('fog_mat')}
    if not fm or all(v == (0.0, 0.0, 0.0) for v in fm.values()):
        return None
    mi = np.asarray(mi, np.int64)
    burn = np.zeros(mi.shape, np.float32)
    bias = np.zeros(mi.shape, np.float32)
    bank = np.zeros(mi.shape, np.float32)
    for m, (b0, b1, b2) in fm.items():
        sel = mi == m
        burn[sel] = F32(b0)
        bias[sel] = F32(b1)
        bank[sel] = F32(b2)
    return SimpleNamespace(fog_burn=burn, fog_bias=bias, fog_bank=bank)


def _eye_of(ctx, scene):
    eye = getattr(ctx, 'camera_pos', None) if ctx is not None else None
    if eye is None and scene is not None:
        cam = getattr(scene, 'camera', None)
        mw = getattr(cam, 'matrix_world', None)
        if mw is not None:
            eye = np.asarray(mw, np.float32)[:3, 3]
    return None if eye is None else np.asarray(eye, np.float32)


# ---- F004: the GameCube's horizontal fog range adjust

def range_adjust_on(st, scene):
    """True under a perspective camera (GX builds the table from P00;
    an orthographic frame takes k = 1 with a note, A20)."""
    if not bool(getattr(st, 'fog_range_adjust', False)):
        return False
    cam = getattr(scene, 'camera', None) if scene is not None else None
    return str(getattr(cam, 'type', 'PERSP')) != 'ORTHO'


def gc_adjust_consts(st, width):
    """(cx32, k32): the INTERNAL frame centre and the 32-OUTPUT-pixel step
    per internal pixel, 1 / (32 * ss) with ss the supersample factor
    (one multiply on both roads; a division by ss = 3 is not exact)."""
    w_out = max(int(getattr(st, 'resolution_x', width) or width), 1)
    ss = max(1, int(round(float(width) / float(w_out))))
    return F32(float(width) * 0.5), F32(1.0 / (32.0 * ss))


def gc_adjust_knots(scene, st, width, height):
    """GX_InitFogAdjTable (libogc gx.c): ten 4.8 fixed-point secants at
    32-output-pixel offsets from the viewport centre, tan = (i+1)*32 *
    (2/W_out) / P00, r = floor(sqrt(1 + tan^2) * 256) & 0x0fff; K[i] =
    r / 256 as float32. P00 is the frame's own projection."""
    from . import render as _R
    cam = getattr(scene, 'camera', None)
    proj = _R.camera_matrices(cam, int(width), int(height))[1]
    p00 = float(np.asarray(proj, np.float64)[0, 0])
    if abs(p00) < 1e-30:
        p00 = 1e-30
    w_out = max(int(getattr(st, 'resolution_x', width) or width), 1)
    K = np.zeros(10, np.float32)
    for i in range(10):
        tanv = (i + 1) * 32 * (2.0 / w_out) / p00
        r = int(np.floor(np.sqrt(1.0 + tanv * tanv) * 256.0)) & 0x0fff
        K[i] = F32(r / 256.0)
    return K


def range_adjust(d, px, st, scene, width, height):
    """F004 per point: the column's offset from the centre in 32-output-
    pixel steps, the table interpolated (K[-1] = 1 at the centre), the
    planar depth scaled; order-free per pixel."""
    K = gc_adjust_knots(scene, st, width, height)
    cx32, k32 = gc_adjust_consts(st, width)
    dx = np.asarray(px, np.float32) + F32(0.5)
    dx = dx - cx32
    dx = np.abs(dx)
    t = dx * k32
    j = np.minimum(np.floor(t), F32(9.0))
    ji = j.astype(np.int64)
    ka = np.where(ji == 0, F32(1.0), K[np.maximum(ji - 1, 0)])
    kb = K[ji]
    fr = t - j
    kd = kb - ka
    kd = kd * fr
    k = ka + kd
    return (d * k).astype(np.float32)


# ---- F005: Namco System 21's one fog value per polygon

def face_depth(ctx):
    """The polygon's fog depth: the mean of its three corners' planar
    depths (the F000 chain per corner, the fixed corner order 0, 1, 2,
    the mean as ONE multiply by float32(1/3))."""
    mesh = ctx.scene.mesh
    tri = np.asarray(ctx.tri, np.int64)
    V = np.asarray(mesh.verts, np.float32)[np.asarray(mesh.tris)[tri]]
    eye = np.asarray(ctx.camera_pos, np.float32)
    r = np.asarray(ctx.view_matrix, np.float32)[2, :3]
    zs = []
    for c in range(3):
        e = V[:, c] - eye[None, :]
        z = e[:, 0] * r[0]
        z = z + e[:, 1] * r[1]
        z = z + e[:, 2] * r[2]
        zs.append(np.abs(z))
    ds = zs[0] + zs[1]
    ds = ds + zs[2]
    return (ds * F32(1.0 / 3.0)).astype(np.float32)


# ---- F008: LightWave's backdrop fog target

def backdrop_of(job):
    """The frame's backdrop at every pixel, (h, w, 3) float32 -- built
    once by render() (A30) or, for a caller that has none (the self
    test's own jobs), here, and kept on the job."""
    bd = getattr(job, 'backdrop', None)
    if bd is None:
        from . import sky as _SKY
        bd = _SKY.backdrop_rgb(job.scene, job.settings, int(job.width),
                               int(job.height), getattr(job, 'vp', None),
                               job.eye, getattr(job, 'textures', None),
                               ss=int(getattr(job, 'ss', 1) or 1))
        job.backdrop = bd
    return bd


def backdrop_atlas(job, announce=False):
    """The `hal_backdrop` atlas entry: (key, builder) with the key a crc
    of the bytes (not the 14.7 MB at 720p) so `upload_cached` compares
    cheaply; RGBA float32 at the internal size."""
    import zlib
    bd = np.ascontiguousarray(backdrop_of(job), np.float32)
    h, w = bd.shape[:2]
    arr = np.ones((h, w, 4), np.float32)
    arr[..., :3] = bd
    key = ('backdrop', zlib.crc32(arr.tobytes()), (h, w))
    if announce:
        mb = arr.nbytes / (1024.0 * 1024.0)
        print(f"[Halcyon GPU] backdrop fog: the CPU's backdrop ({w}x{h}, "
              f"{mb:.1f} MB) is uploaded as the fog target")
    return key, (lambda a=arr: a)


def _backdrop_of_ctx(ctx, st):
    bd = getattr(ctx, 'backdrop', None)
    if bd is not None:
        return bd
    job = getattr(ctx, 'job', None)
    if job is not None:
        bd = backdrop_of(job)
    else:
        from . import sky as _SKY
        w, h = _size_of(st, ctx)
        bd = _SKY.backdrop_rgb(ctx.scene, st, w, h, None,
                               np.asarray(ctx.camera_pos, np.float32),
                               getattr(ctx, 'textures', None) or {})
    try:
        ctx.backdrop = bd
    except Exception:                                           # noqa: BLE001
        pass
    return bd


def fog_target(st, ctx, n):
    """The colour the blend fades toward, (n, 3) or (1, 3) float32:
    FIXED -- the Fog Colour times Fog Ambient (F006); BACKDROP (F008) --
    the backdrop at the point's own pixel, or, for a point with no pixel
    (a reflection hit, a transparent layer, a vertex-lit corner), the
    world colour along its own direction (LightWave's fog had no other
    answer for a reflected ray, A10)."""
    if source_of(st) == 'BACKDROP' and ctx is not None:
        px = getattr(ctx, 'px', None)
        py = getattr(ctx, 'py', None)
        if px is not None and py is not None:
            bd = _backdrop_of_ctx(ctx, st)
            if bd is not None:
                return np.ascontiguousarray(
                    bd[np.asarray(py, np.int64), np.asarray(px, np.int64)],
                    np.float32)
        I = getattr(ctx, 'I', None)
        sc = getattr(ctx, 'scene', None)
        if I is not None and sc is not None:
            from .render import world_color
            eye = _eye_of(ctx, sc)
            col = world_color(sc, st, np.asarray(I, np.float32),
                              getattr(ctx, 'textures', None) or {}, n,
                              eye=eye)
            return np.ascontiguousarray(col, np.float32)
    col = np.asarray(st.fog_color, np.float32)[None, :]
    amb = float(getattr(st, 'fog_ambient', 1.0))
    if amb != 1.0:
        col = (col * F32(amb)).astype(np.float32)
    return col


# ---- F009: POV-Ray's ground fog (fog_type 2), the atan integral

def ground_consts(st, eye):
    """(o32, ia32, y1_32, alt32): the offset, 1/altitude, the eye's own
    Y and the altitude, float32 once (texel 225)."""
    o32 = F32(float(getattr(st, 'fog_ground_offset', 0.0) or 0.0))
    alt = max(float(getattr(st, 'fog_ground_alt', 1.0) or 1.0), 1e-3)
    ia32 = F32(1.0 / alt)
    ez = F32(np.asarray(eye, np.float32)[2])
    y1 = (ez - o32) * ia32
    return o32, ia32, F32(y1), F32(alt)


def ground_fog_integral(P, eye, st, turb=False):
    """POV's ComputeGroundFogDepth on the eye -> P segment: density
    1/(1+Y^2) above the offset (Y in altitudes), the five cases of the
    integral's mean density m, transmittance exp(-length * m * density).
    Every candidate is computed and the select picks (A32: the divides
    by zero of a discarded lane never reach f). A hit integrates the
    eye -> hit segment (A34, stated)."""
    P = np.asarray(P, np.float32)
    eye = np.asarray(eye, np.float32)
    o32, ia32, y1, _alt = ground_consts(st, eye)
    den32 = F32(float(st.fog_density))
    y2 = P[:, 2] - o32
    y2 = y2 * ia32
    dP = P - eye[None, :]
    sx = dP[:, 0] * dP[:, 0]
    sx = sx + dP[:, 1] * dP[:, 1]
    sx = sx + dP[:, 2] * dP[:, 2]
    dd = np.sqrt(sx).astype(np.float32)
    if turb:
        dd = turbulence(dd, P, eye, st)
    with np.errstate(divide='ignore', invalid='ignore'):
        a1 = F32(np.arctan(y1))
        a2 = np.arctan(y2)
        mB = (a2 - y1) / (y2 - y1)
        mC = (a1 - y2) / (y1 - y2)
        mD = (a1 - a2) / (y1 - y2)
        q = y1 * y1
        q = q + F32(1.0)
        mE = F32(1.0) / q
        both = (y1 <= F32(0.0)) & (y2 <= F32(0.0))
        m = np.where(both, F32(1.0),
                     np.where(y1 <= F32(0.0), mB,
                              np.where(y2 <= F32(0.0), mC,
                                       np.where(np.abs(y1 - y2) > F32(1e-6),
                                                mD, mE)))).astype(np.float32)
    e = dd * m
    e = e * den32
    e = -e
    return np.exp(e).astype(np.float32)


def ground_fog_sky_consts(st, eye):
    """(num32, den32, col32): alt * (pi/2 - atan(y1)) as one float32
    product, the density, the fog colour x ambient -- the miss ray's
    closed form needs nothing per pixel but its elevation."""
    o32, ia32, y1, alt32 = ground_consts(st, eye)
    a1 = F32(np.arctan(y1))
    hp = F32(math.pi / 2.0)
    t = hp - a1
    num = alt32 * t
    col = np.asarray(st.fog_color, np.float32)
    amb = F32(float(getattr(st, 'fog_ambient', 1.0)))
    if float(amb) != 1.0:
        col = (col * amb).astype(np.float32)
    return F32(num), F32(float(st.fog_density)), col


def ground_fog_sky(cols, dirs, st, eye):
    """POV fogs a ray that hits nothing: T = exp(-(alt * (pi/2 -
    atan(y1)) / D.z) * density) for a rising ray, 0 (the fog colour)
    for a level or falling one; the sky is fogged by elevation
    (`_background_image`'s tail; gpu/sky.hal_ground_fog_sky is the
    twin on the same ray)."""
    cols = np.asarray(cols, np.float32)
    dirs = np.asarray(dirs, np.float32)
    num32, den32, col32 = ground_fog_sky_consts(st, eye)
    dz = np.ascontiguousarray(dirs[:, 2])
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        q = num32 / dz
        e = q * den32
        e = -e
        T = np.exp(e)
    T = np.where(dz > F32(0.0), T, F32(0.0)).astype(np.float32)[:, None]
    out = cols * T
    g = F32(1.0) - T
    out = out + col32[None, :] * g
    return out.astype(np.float32)


# ---- F010: POV-Ray's turbulent fog

def turbulence(dd, P, eye, st):
    """POV's ComputeConstantFogDepth: one turbulence read at the middle
    of the fogged segment scales its length, faded with distance by
    k = exp(-length * density): dd *= 1 - k * min(1, turb * depth).
    Halcyon's integer-hash value turbulence at POV's octaves 6 / lambda
    2 / omega 0.5, one scalar (POV's is a Perlin vector; disclosed);
    no frame term, so a frame renders the same bits every time."""
    from . import patterns as PT
    turb32 = F32(float(getattr(st, 'fog_turbulence', 0.0) or 0.0))
    q32 = F32(float(getattr(st, 'fog_turb_depth', 0.5)))
    L32 = F32(float(st.fog_density))
    P = np.asarray(P, np.float32)
    eye = np.asarray(eye, np.float32)
    pm = eye[None, :] + P
    pm = pm * F32(0.5)
    pm = pm * turb32
    u = np.asarray(PT.turbulence(np.ascontiguousarray(pm), octaves=6,
                                 lacunarity=2.0, gain=0.5), np.float32)
    dd = np.asarray(dd, np.float32)
    e = -dd
    e = e * L32
    k = np.exp(e)
    uq = u * q32
    uq = np.minimum(uq, F32(1.0))
    kk = k * uq
    sc = F32(1.0) - kk
    return (dd * sc).astype(np.float32)


# --------------------------------------------------------------- the GPU side

def structure(st):
    """`consts['fog']`: WHICH branches `hal_fog` emits (the plan
    signature carries every field read here). None when fog is off."""
    if not bool(getattr(st, 'fog', False)):
        return None
    return {
        'mode': mode_of(st),
        'vertex': vertex_quantised(st),
        'bands': int(getattr(st, 'fog_bands', 0) or 0) >= 2,
        'height': bool(getattr(st, 'fog_height', False))
        and mode_of(st) != 'GROUND',
        'table': table_of(st),
        'dither': table_of(st) == 'VOODOO64'
        and bool(getattr(st, 'fog_dither', False)),
        'depth': depth_of(st),
        # wave 2 (LIGHT-A2); `range_adjust` is emitted only under a
        # perspective camera (gpu/shade._fog_structure adds 'ortho')
        'range_adjust': bool(getattr(st, 'fog_range_adjust', False)),
        'face': face_on(st),
        'source': source_of(st),
        'bank1': bank1_valid(st)
        and mode_of(st) in ('LINEAR', 'TABLE16', 'GTE_1Z'),
        'turb': turb_on(st),
    }


def pack_fog_texels(job):
    """The hal_fogtab row: (1, 512, 4) float32, lighting.md section 0's
    map. Everything not enabled packs zeros. Slots this wave fills:
    0 (start, end, span, density), 1 (col * ambient, bands),
    2 (top, falloff, n32, k32), 3 (view[2, :3]), 4.x (ambient),
    32..95 VOODOO64 (T, delta), 96..223 PVR128 (T[i], T[i+1]),
    224 (A32, B32, den_q, step32), 227 (the axis viewer, F011),
    240..255 the 4x4 dither, 256..289 DS32.

    The two CAMERA rows are per-frame data, not fog data: texel 3 (the
    CPU's own view row -- F007's z-fog depth and F015's screen-spot
    depth) and texel 227 (SH.axis_viewer(job.view), the fixed viewer of
    F011 / the console light units GX_LIGHT and DS_FIXED) pack whenever
    the texture packs at all -- section 0's one rule registers it under
    AXIS, the AXIS_MODELS and a screen-spot lamp with fog OFF, and the
    GLSL reads those texels for the frame's constant vectors, so they
    sit ABOVE the fog gate. Bitwise the CPU's bytes (a texelFetch)."""
    st = job.settings
    tab = np.zeros((1, TEXELS, 4), np.float32)
    view = np.asarray(job.view, np.float32)
    tab[0, 3] = (view[2, 0], view[2, 1], view[2, 2], F32(0.0))
    tab[0, AXIS_VIEWER_TEXEL, :3] = SH.axis_viewer(view)
    if not bool(getattr(st, 'fog', False)):
        return tab
    start = float(st.fog_start)
    end = float(st.fog_end)
    tab[0, 0] = (F32(start), F32(end), span32(st), F32(float(st.fog_density)))
    amb = F32(float(getattr(st, 'fog_ambient', 1.0)))
    col = np.asarray(st.fog_color, np.float32)
    if float(amb) != 1.0:
        col = (col * amb).astype(np.float32)
    tab[0, 1] = (col[0], col[1], col[2],
                 F32(float(int(getattr(st, 'fog_bands', 0) or 0))))
    n32, k32 = F32(0.0), F32(0.0)
    if depth_of(st) == 'Z':
        n32, k32, _o = projection_consts(job.scene, job.width, job.height)
    tab[0, 2] = (F32(float(getattr(st, 'fog_height_top', 0.0))),
                 F32(max(float(getattr(st, 'fog_height_falloff', 0.0)),
                         0.0)),
                 n32, k32)
    tab[0, 4, 0] = amb
    table = table_of(st)
    mode = mode_of(st)
    if table == 'VOODOO64':
        T, delta = voodoo64_table(st)
        tab[0, 32:96, 0] = T.astype(np.float32)
        tab[0, 32:96, 1] = delta.astype(np.float32)
        tab[0, 240:256, 0] = M4.ravel().astype(np.float32)
    elif table == 'PVR128':
        T, den_q = pvr128_table(st)
        tab[0, 96:224, 0] = T[:128].astype(np.float32)
        tab[0, 96:224, 1] = T[1:129].astype(np.float32)
        tab[0, 224, 2] = den_q
    elif table == 'DS32':
        T, start32, step32 = ds32_table(st)
        tab[0, 256:290, 0] = T.astype(np.float32)
        tab[0, 224, 3] = step32
    if mode == 'GTE_1Z' and table == 'NONE':
        _a, _b, A32, B32 = gte_consts(st)
        tab[0, 224, 0] = A32
        tab[0, 224, 1] = B32
    # ---- wave 2 (LIGHT-A2)
    # texel 4 = (ambient, cx32, k32, fog_spot32): F004's internal centre
    # and the 32-OUTPUT-pixel step per internal pixel (one multiply on
    # both roads, A19 / A80); F015's fog lobe gain
    cx32, k32 = gc_adjust_consts(st, job.width)
    tab[0, 4, 1] = cx32
    tab[0, 4, 2] = k32
    tab[0, 4, 3] = F32(float(getattr(st, 'fog_spot', 0.0) or 0.0))
    if range_adjust_on(st, job.scene):
        # texels 8..17: the ten 4.8 secants of GX_InitFogAdjTable
        K = gc_adjust_knots(job.scene, st, job.width, job.height)
        tab[0, 8:18, 0] = K
    if bank1_valid(st):
        # texel 5 = bank 1 (start, end, span, 1.0 = valid); texel 6 = its
        # GTE constants (A1, B1) -- F006
        s1, e1, sp1 = bank1_consts(st)
        tab[0, 5] = (s1, e1, sp1, F32(1.0))
        if mode == 'GTE_1Z':
            _a1, _b1, A1, B1 = gte_consts(_Bank(s1, e1))
            tab[0, 6, 0] = A1
            tab[0, 6, 1] = B1
    if mode == 'GROUND':
        # texel 225 = (o32, ia32, y1_32, alt32) -- F009
        o32, ia32, y1_32, alt32 = ground_consts(st, job.eye)
        tab[0, 225] = (o32, ia32, y1_32, alt32)
    if turb_on(st):
        # texel 226 = (turb32, q32, 0, 0) -- F010
        tab[0, 226, 0] = F32(float(getattr(st, 'fog_turbulence', 0.0)))
        tab[0, 226, 1] = F32(float(getattr(st, 'fog_turb_depth', 0.5)))
    return tab
