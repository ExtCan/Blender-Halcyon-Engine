"""R251: the era colour roads of the framebuffer stage (post-palette pack).

CPU functions, pure `(rgb, st, ...)` -> float32 (H, W, 3), order-free per
pixel, every one the reference its GPU twin (gpu/stages_palette.py) is
held bitwise against in the simulator:

- `snap_registers` (C061): the palette RAM / DAC precision every register
  is snapped to before the per-pixel search (an Atari ST's 16 from 512,
  the Amiga's from 4096, the VGA DAC's 6 bits).
- `ehb_reduce` (C054): Extra Half-Brite -- 32 fitted 12-bit registers and
  their component-wise halves (the sixth bitplane), 64 on screen.
- `cry16` (C011): the Atari Jaguar's CRY pixel -- 8-bit intensity times a
  256-cell chroma table on the RGB cube's upper surface.
- `yjk` (C059): the MSX2+ V9958's YJK pixel -- 5-bit luma per pixel,
  chroma shared by every aligned group of four.
- `reduce_depth_era`: the dispatcher `core/post.reduce_depth` asks first;
  None means the 1.89.0 road, untouched.

`cached_peek(key)` is the one shared per-process peek into the adaptive
palette cache (the post-signal pack imports it from here).

Imports `numpy`, `palette` and `dither` only -- never `post`, which
imports this module at its head.
"""

import functools

import numpy as np

from . import dither as DI
from . import palette as PA

# ------------------------------------------------------------------ shared

#: the CPU's own k / 255 quotients in float32 -- pinned equal to
#: gpu/chain.quant_lut((8, 8, 8)) for every channel by the pack's tests
LUT255 = (np.arange(256, dtype=np.float32) / np.float32(255)).astype(np.float32)

#: notes printed once per (road, dither): a dither that has no seat in an
#: integer encode is inert, and the console says so
_NOTED = set()


def cached_peek(key):
    """The adaptive palette cached under `key` (the P:316 lock key
    `('ADAPTIVE', size, method, seed)`), or None when it has not been built
    -- so a chain function can ask "is the locked palette built?" without
    pixels and without spending a readback. The cache holds the builder's
    RAW result (palette.cached_adaptive); the register snap (C061) lives
    in the caller."""
    return PA._PALETTE_CACHE.get(key)


def note_ignored(road, kind):
    kind = str(kind)
    if kind == 'NONE':
        return
    msg = (f'colour depth {road}: the encode has no dither seat; '
           f'Dither {kind} ignored')
    if msg not in _NOTED:
        _NOTED.add(msg)
        print(f'[Halcyon] {msg}')


# -------------------------------------------------- C061: register depth

#: palette_bits item -> levels per channel (0 = the fitted 8-bit values)
REGISTER_LEVELS = {'NONE': 0, 'BITS_1': 2, 'BITS_2': 4, 'CPC_27': 3,
                   'BITS_3': 8, 'BITS_4': 16, 'BITS_5': 32, 'BITS_6': 64}


def snap_registers(pal, st):
    """The palette snapped to the machine's register / DAC lattice
    (`st.palette_bits`), or `pal` ITSELF (the same object) when the depth is
    NONE or unknown -- the 1.89.0 road is bitwise untouched. Duplicates
    after the snap are kept: the inverse colormap's argmin picks the lowest
    duplicate, the named tie rule."""
    levels = REGISTER_LEVELS.get(str(getattr(st, 'palette_bits', 'NONE')), 0)
    if levels < 2:
        return pal
    return PA.snap_levels(pal, levels)


# ------------------------------------------------ C054: Extra Half-Brite

_EHB_LUTS = {}
_EHB_ORDER = []
_EHB_LIMIT = 8


def ehb_registers(P32):
    """(q32, q64, P64): the 4:4:4 registers of a fitted 32 (half to even),
    the 64 with the sixth bitplane's halves appended, and the CPU's own
    k / 15 quotients as float32."""
    q32 = np.round(np.clip(np.asarray(P32, np.float32), 0.0, 1.0)
                   * np.float32(15)).astype(np.int32)
    q64 = np.concatenate([q32, q32 >> 1], 0)
    P64 = (q64.astype(np.float32) / np.float32(15)).astype(np.float32)
    return q32, q64, P64


def ehb_lut(q64):
    """int32[4096]: for every 4:4:4 code (r << 8) | (g << 4) | b the argmin
    over the 64 registers of the integer squared distance in 0..15 units,
    np.argmin = lowest index on ties. Cached by the registers' bytes, capped
    at 8 (the _ICM_LIMIT rule): with the lock off every frame fits a new 32."""
    q = np.ascontiguousarray(np.asarray(q64, np.int32))
    key = q.tobytes()
    hit = _EHB_LUTS.get(key)
    if hit is not None:
        return hit
    codes = np.arange(4096, dtype=np.int32)
    cell = np.stack([codes >> 8, (codes >> 4) & 15, codes & 15], 1)   # (4096, 3)
    d = cell[:, None, :] - q[None, :, :]                                # (4096, 64, 3)
    dist = (d * d).sum(axis=2)
    lut = np.argmin(dist, axis=1).astype(np.int32)
    _EHB_LUTS[key] = lut
    _EHB_ORDER.append(key)
    while len(_EHB_ORDER) > _EHB_LIMIT:
        _EHB_LUTS.pop(_EHB_ORDER.pop(0), None)
    return lut


def ehb_lut_image(q64):
    """The LUT as a (64, 64, 4) float32 index texture: .r[y, x] =
    lut[y * 64 + x] (code >> 6, code & 63)."""
    lut = ehb_lut(q64)
    img = np.zeros((64, 64, 4), np.float32)
    img[:, :, 0] = lut.reshape(64, 64).astype(np.float32)
    img[:, :, 3] = 1.0
    return img


def ehb_key(st, seed):
    return ('ADAPTIVE', 32, str(st.palette_method), int(seed))


def ehb_fit(rgb, st, seed):
    """The 32 registers fitted from this frame (cached under the P:316 lock
    key at size 32 when Lock Palette is on): the RAW median cut."""
    key = ehb_key(st, seed)
    flat = np.asarray(rgb, np.float32).reshape(-1, 3)
    if getattr(st, 'palette_lock', True):
        return PA.cached_adaptive(
            key, lambda: PA.get_palette('ADAPTIVE', 32, flat,
                                        st.palette_method, seed))
    return PA.get_palette('ADAPTIVE', 32, flat, st.palette_method, seed)


def ehb_reduce(rgb, st, seed=0):
    """Extra Half-Brite: 32 fitted 12-bit registers plus their exact halves.
    Ties: half to even on the lattice round, lowest index in the LUT."""
    rgb = np.asarray(rgb, np.float32)
    h, w = rgb.shape[:2]
    P32 = ehb_fit(rgb, st, seed)
    _q32, q64, P64 = ehb_registers(P32)
    lut = ehb_lut(q64)
    kind = str(st.dither)
    strength = float(st.dither_strength)
    if kind in DI.KERNELS or kind == 'NOISE':
        return np.asarray(DI.apply_dither(rgb, P64, kind, strength,
                                          st.dither_serpentine, seed=seed),
                          np.float32)
    pert = rgb
    if kind in DI.ORDERED:
        tm = DI.threshold_map(kind, h, w)
        spacing = DI._palette_spacing(P64)
        pert = rgb + ((tm - 0.5) * strength * spacing)[..., None]
    q = np.round(np.clip(pert, 0.0, 1.0) * np.float32(15)).astype(np.int32)
    code = (q[..., 0] << 8) | (q[..., 1] << 4) | q[..., 2]
    return P64[lut[code]]


# ---------------------------------------------------------- C011: CRY16


@functools.lru_cache(maxsize=None)
def _cry_tables():
    """(CRY_TABLE uint8[256, 3], CRY_DIV uint8[256, 256], CRY_MUL uint8[256,
    256], CRY_ARGMIN uint8[3, 256, 256]) -- built once per process, all
    integer.

    CRY_TABLE is HALCYON'S OWN DERIVATION of the 16 x 16 chroma square (the
    manual's modifier tables were not transcribed this round; the
    disagreement is unmeasured -- `cry_table_diff` is the hook): cell
    cr = y * 16 + x; grid coordinate u = (x - 8) / 8 for x <= 8 else
    (x - 8) / 7, v likewise from y; seven anchors W (0,0) white, B (-1,-1)
    blue, M (0,-1) magenta, R (1,-1) red, Y (1,1) yellow, G (0,1) green,
    C (-1,1) cyan; six triangles (W,B,M) (W,M,R) (W,R,Y) (W,Y,G) (W,G,C)
    (W,C,B) tile the square, each in one cube face, so every cell's max is
    exactly 255 (a true upper-surface point), white is 0x88, the primaries
    and secondaries sit on the corners and edge midpoints, greys are exact.
    """
    anchors = {
        'W': ((0.0, 0.0), (255, 255, 255)),
        'B': ((-1.0, -1.0), (0, 0, 255)),
        'M': ((0.0, -1.0), (255, 0, 255)),
        'R': ((1.0, -1.0), (255, 0, 0)),
        'Y': ((1.0, 1.0), (255, 255, 0)),
        'G': ((0.0, 1.0), (0, 255, 0)),
        'C': ((-1.0, 1.0), (0, 255, 255)),
    }
    tris = [('W', 'B', 'M'), ('W', 'M', 'R'), ('W', 'R', 'Y'),
            ('W', 'Y', 'G'), ('W', 'G', 'C'), ('W', 'C', 'B')]

    def coord(i):
        return (i - 8) / 8.0 if i <= 8 else (i - 8) / 7.0

    table = np.zeros((256, 3), np.uint8)
    for cr in range(256):
        x, y = cr & 15, cr >> 4
        u, v = coord(x), coord(y)
        rgb = None
        for a0, a1, a2 in tris:
            (u0, v0), c0 = anchors[a0]
            (u1, v1), c1 = anchors[a1]
            (u2, v2), c2 = anchors[a2]
            det = (u1 - u0) * (v2 - v0) - (u2 - u0) * (v1 - v0)
            w1 = ((u - u0) * (v2 - v0) - (u2 - u0) * (v - v0)) / det
            w2 = ((u1 - u0) * (v - v0) - (u - u0) * (v1 - v0)) / det
            w0 = 1.0 - w1 - w2
            if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9:
                rgb = [w0 * c0[k] + w1 * c1[k] + w2 * c2[k] for k in range(3)]
                break
        assert rgb is not None, cr
        table[cr] = np.clip(np.round(np.asarray(rgb, np.float64)), 0, 255
                            ).astype(np.uint8)

    I = np.arange(256, dtype=np.int64)[:, None]
    V = np.arange(256, dtype=np.int64)[None, :]
    Is = np.maximum(I, 1)
    div = np.minimum((V * 255) // Is, 255).astype(np.uint8)     # row 0 = row 1
    mul = ((V * I) // 255).astype(np.uint8)

    argmin = np.zeros((3, 256, 256), np.uint8)
    t = table.astype(np.int32)                                   # (256, 3)
    a_ = np.arange(256, dtype=np.int32)
    for f in range(3):
        others = [k for k in range(3) if k != f]
        for b in range(256):
            chroma = np.zeros((256, 3), np.int32)                # one row per a
            chroma[:, f] = 255
            chroma[:, others[0]] = a_
            chroma[:, others[1]] = b
            dist = np.abs(chroma[:, None, :] - t[None, :, :]).sum(axis=2)
            argmin[f, :, b] = np.argmin(dist, axis=1).astype(np.uint8)
    return table, div, mul, argmin


def cry_tables():
    return _cry_tables()


def cry_table_diff(manual):
    """The measuring hook for the round that transcribes the manual's
    modifier tables: the count of the 256 cells off by more than one
    8-bit step in any channel against `manual` ((256, 3) uint8)."""
    table = _cry_tables()[0].astype(np.int32)
    m = np.asarray(manual).astype(np.int32).reshape(256, 3)
    return int((np.abs(table - m) > 1).any(axis=1).sum())


def cry16(rgb):
    """The Jaguar's CRY pixel: 8-bit intensity = max(r, g, b), the chroma
    cell from the 16 x 16 table, decoded back. Ties: np.round half to even
    on the 8-bit value; the FIRST maximal channel picks the face; lowest
    index in the argmin table."""
    table, div, mul, argmin = _cry_tables()
    rgb = np.asarray(rgb, np.float32)
    R8 = np.round(np.clip(rgb, 0.0, 1.0) * np.float32(255)).astype(np.int32)
    I = R8.max(axis=2)
    Is = np.maximum(I, 1)
    face = np.argmax(R8 == I[..., None], axis=2)
    r, g, b = R8[..., 0], R8[..., 1], R8[..., 2]
    a = np.where(face == 0, g, r)
    bb = np.where(face == 2, g, b)
    ca = div[Is, a].astype(np.int32)
    cb = div[Is, bb].astype(np.int32)
    cr = argmin[face, ca, cb].astype(np.int32)
    t = table[cr].astype(np.int32)                               # (H, W, 3)
    out8 = mul[I[..., None], t]
    return LUT255[out8]


def cry_images():
    """The stage's constant textures: `cry_tabs` (256, 512, 4) with
    .r[I, v] = CRY_DIV[I, v] and .r[I, 256 + v] = CRY_MUL[I, v];
    `cry_argmin` (256, 768, 4) with .r[b, f * 256 + a] = CRY_ARGMIN[f, a, b];
    `cry_table` (1, 256, 4) with rgb = CRY_TABLE as 0..255 floats."""
    table, div, mul, argmin = _cry_tables()
    tabs = np.zeros((256, 512, 4), np.float32)
    tabs[:, :256, 0] = div.astype(np.float32)
    tabs[:, 256:, 0] = mul.astype(np.float32)
    tabs[:, :, 3] = 1.0
    am = np.zeros((256, 768, 4), np.float32)
    for f in range(3):
        am[:, f * 256:(f + 1) * 256, 0] = argmin[f].T.astype(np.float32)
    am[:, :, 3] = 1.0
    tab = np.zeros((1, 256, 4), np.float32)
    tab[0, :, :3] = table.astype(np.float32)
    tab[0, :, 3] = 1.0
    return {'cry_tabs': tabs, 'cry_argmin': am, 'cry_table': tab}


# ------------------------------------------------------------- C059: YJK


def _yjk_encode5(rgb):
    """(y, j, k) int32 per pixel from the 5-bit channels (Grauw's encode)."""
    R8 = np.round(np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
                  * np.float32(255)).astype(np.int32)
    r, g, b = R8[..., 0] >> 3, R8[..., 1] >> 3, R8[..., 2] >> 3
    n = -2 * r - g + 2
    fl = n // 4
    u = b - fl
    y = (u + 1) // 2
    return y, r - y, g - y


def _yjk_group_sums(v):
    """Sum of `v` (H, W) over each aligned group of four output columns; the
    right edge repeats its last pixel so every group sums exactly four."""
    h, w = v.shape
    ng = (w + 3) // 4
    cols = np.minimum(4 * np.arange(ng)[:, None] + np.arange(4)[None, :], w - 1)
    return v[:, cols].sum(axis=2)                                  # (H, ng)


def _yjk_mean(s):
    return np.clip(np.round(s.astype(np.float32) * np.float32(0.25)),
                   -32, 31).astype(np.int32)


def yjk_groups(rgb):
    """The (H, ceil(W / 4), 2) int32 array of (J, K) per aligned group."""
    _y, j, k = _yjk_encode5(rgb)
    J = _yjk_mean(_yjk_group_sums(j))
    K = _yjk_mean(_yjk_group_sums(k))
    return np.stack([J, K], axis=2)


def yjk(rgb):
    """The V9958's YJK pixel: 5-bit luma per pixel, J and K averaged over
    each aligned group of four (half to even) and clamped to six signed
    bits, decoded with Grauw's rounding. Ties: np.round half to even on the
    8-bit value and on the group mean; floor division everywhere else."""
    rgb = np.asarray(rgb, np.float32)
    h, w = rgb.shape[:2]
    y, j, k = _yjk_encode5(rgb)
    J = _yjk_mean(_yjk_group_sums(j))
    K = _yjk_mean(_yjk_group_sums(k))
    gx = np.arange(w) >> 2
    Jp = J[:, gx]
    Kp = K[:, gx]
    R = np.clip(y + Jp, 0, 31)
    G = np.clip(y + Kp, 0, 31)
    B = np.clip((5 * y - 2 * Jp - Kp + 2) // 4, 0, 31)
    v5 = np.stack([R, G, B], axis=2)
    v8 = (v5 << 3) | (v5 >> 2)
    return LUT255[v8]


# ------------------------------------------------------- the dispatcher


def era_mode(st):
    """Which era road owns the framebuffer stage under `st`, or None: the
    first match owns it -- cells > scanline > CRY16 > YJK > EHB -- and a
    second active era mode is printed once as ignored."""
    depth = str(st.color_depth)
    cells = str(getattr(st, 'attribute_cells', 'NONE'))
    scan = str(getattr(st, 'scanline_palette', 'NONE'))
    active = []
    if cells in CELL_SIZE:
        active.append('CELLS')
    if scan in SCANLINE_LEVELS:
        active.append('SCANLINE')
    if depth == 'CRY16':
        active.append('CRY16')
    elif depth == 'YJK':
        active.append('YJK')
    if str(st.palette_mode) == 'EHB':
        active.append('EHB')
    if not active:
        return None
    owner = active[0]
    for other in active[1:]:
        msg = f'colour depth: {owner} owns the stage; {other} ignored'
        if msg not in _NOTED:
            _NOTED.add(msg)
            print(f'[Halcyon] {msg}')
    return owner


def reduce_depth_era(rgb, st, seed=0):
    """The era colour roads' half of core/post.reduce_depth: the picture
    when an era mode is selected, None otherwise (the 1.89.0 road)."""
    owner = era_mode(st)
    if owner is None:
        return None
    if owner == 'CELLS':
        return attribute_cells(rgb, st, seed)
    if owner == 'SCANLINE':
        return scanline_palette(rgb, st, seed)
    if owner == 'CRY16':
        note_ignored('CRY16', st.dither)
        return cry16(rgb)
    if owner == 'YJK':
        note_ignored('YJK', st.dither)
        return yjk(rgb)
    return ehb_reduce(rgb, st, seed)


# ======================================================================
# ---- wave 2 (PAL-2): C092 Super Black, C093 Video Color Check, C050
# attribute cells, C060 per-scanline palettes
# ======================================================================

# ---------------------------------------------------- C092: Super Black


def pack_coverage(cov):
    """The (H, W) boolean coverage plane packed for the SUPERBLACK stage:
    64 pixels per texel, 16 bits per channel -- texel x = px.x >> 6,
    channel (px.x >> 4) & 3, bit px.x & 15 -- as exact integer-valued
    float32 (< 2^16). Shape (H, ceil(W / 64), 4)."""
    cov = np.asarray(cov, bool)
    h, w = cov.shape
    nt = -(-w // 64)
    pad = np.zeros((h, nt * 64), np.int64)
    pad[:, :w] = cov
    weights = (1 << np.arange(16, dtype=np.int64))
    return (pad.reshape(h, nt, 4, 16) * weights).sum(axis=3).astype(np.float32)


def unpack_coverage(packed, w):
    """pack_coverage's inverse: the (H, w) boolean plane."""
    p = np.asarray(packed, np.float32).astype(np.int64)            # (H, nt, 4)
    bits = (p[..., None] >> np.arange(16, dtype=np.int64)) & 1     # (H, nt, 4, 16)
    return bits.reshape(p.shape[0], -1)[:, :int(w)].astype(bool)


def super_black_threshold(st):
    """The floor as the CPU's own float32 quotient T / 255 (the GPU's
    uniform is this very value)."""
    t = int(np.clip(int(getattr(st, 'super_black_threshold', 15)), 0, 255))
    return np.float32(t) / np.float32(255)


def super_black(rgb, cov, st):
    """3D Studio's / 3ds Max's Super Black: every geometry-covered pixel is
    floored at the threshold (per channel, an IEEE max), uncovered pixels
    are untouched -- so a luminance keyer can tell the background's 0
    from the darkest rendered surface. No tie rule (a max)."""
    rgb = np.asarray(rgb, np.float32)
    thr = super_black_threshold(st)
    out = rgb.copy()
    np.maximum(rgb, thr, out=out, where=np.asarray(cov, bool)[..., None])
    return out


# --------------------------------------------- C093: Video Color Check

#: video_color_check item -> the LEGALISE stage's int `mode`
LEGAL_MODES = {'FLAG_BLACK': 1, 'SCALE_LUMA': 2, 'SCALE_SAT': 3}


def legal_limits(st):
    """(hi, lo) float32: the composite envelope's peak and trough in
    display units -- (limit - setup) / gain with NTSC-M's 7.5 IRE setup and
    92.5 of picture, or PAL's 0 and 100; the peak 120 or 110 IRE, the
    trough -20 IRE either way."""
    pal = str(getattr(st, 'video_system', 'NTSC')) == 'PAL'
    setup, gain = (0.0, 100.0) if pal else (7.5, 92.5)
    vmax = 110.0 if str(getattr(st, 'video_ire_limit', 'IRE_120')) == 'IRE_110' \
        else 120.0
    return (np.float32((vmax - setup) / gain),
            np.float32((-20.0 - setup) / gain))


def legal_terms(rgb):
    """(y, c2): the catalogue's 3-decimal YIQ luma and the squared chroma
    amplitude, float32, one op per statement (the GLSL mirrors it)."""
    f = np.float32
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = r * f(0.299)
    t = g * f(0.587)
    y = y + t
    t = b * f(0.114)
    y = y + t
    i = r * f(0.596)
    t = g * f(0.274)
    i = i - t
    t = b * f(0.322)
    i = i - t
    q = r * f(0.211)
    t = g * f(0.523)
    q = q - t
    t = b * f(0.312)
    q = q + t
    ii = i * i
    qq = q * q
    c2 = ii + qq
    return y, c2


def video_color_check(rgb, st):
    """3D Studio's / 3ds Max's Video Color Check on the composite envelope
    (luma plus the chroma subcarrier's amplitude, in IRE): illegal pixels
    are flagged black, scaled in luminance, or scaled in saturation toward
    their own luma. Compares strict as written; float32 throughout, every
    line one NumPy op in the GLSL's order. Legal pixels are the identity
    (the final clip is the display range, where they already sit)."""
    rgb = np.asarray(rgb, np.float32)
    mode = LEGAL_MODES.get(str(getattr(st, 'video_color_check', 'NONE')))
    if mode is None:
        return rgb
    f = np.float32
    hi, lo = legal_limits(st)
    y, c2 = legal_terms(rgb)
    a = hi - y
    b_ = y - lo
    aa = a * a
    bb = b_ * b_
    peak = (a < f(0.0)) | (c2 > aa)
    trough = c2 > bb
    eps = f(1e-6)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        if mode == 1:
            out = np.where((peak | trough)[..., None], f(0.0), rgb)
        elif mode == 2:
            cc = np.sqrt(c2)
            d = y + cc
            s = hi / d
            out = np.where(peak[..., None], rgb * s[..., None], rgb)
            yy = np.where(peak, y * s, y)
            cc = np.where(peak, cc * s, cc)
            b2 = yy - lo
            m = (cc > b2) & (cc > eps)
            k = b2 / cc
            k = np.minimum(k, f(1.0))
            k = np.maximum(k, f(0.0))
            dlt = out - yy[..., None]
            e = dlt * k[..., None]
            out = np.where(m[..., None], yy[..., None] + e, out)
        else:
            cc = np.sqrt(c2)
            m = (peak | trough) & (cc > eps)
            k1 = a / cc
            k2 = b_ / cc
            k = np.minimum(k1, k2)
            k = np.minimum(k, f(1.0))
            k = np.maximum(k, f(0.0))
            dlt = rgb - y[..., None]
            e = dlt * k[..., None]
            out = np.where(m[..., None], y[..., None] + e, rgb)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


#: the wave entry's name for the same function
legalise = video_color_check


# ------------------------------------------------ C050: attribute cells

#: the ZX Spectrum's attribute colours 0..7 = black, blue, red, magenta,
#: green, cyan, yellow, white at level 215 (0xD7) normal / 255 bright per
#: set channel (Sinclair BASIC manual ch. 16); black is (0, 0, 0) in both
ZX_LEVELS = np.array(
    [[[(215 if c & 2 else 0), (215 if c & 4 else 0), (215 if c & 1 else 0)]
      for c in range(8)],
     [[(255 if c & 2 else 0), (255 if c & 4 else 0), (255 if c & 1 else 0)]
      for c in range(8)]], dtype=np.uint8)

#: the TMS9918A's fifteen colours in VDP order 1..15 (Sean Young's
#: TMS9918A documentation, http://bifi.msxnet.org/msxnet/tech/tms9918a.txt:
#: the datasheet's Y / R-Y / B-Y converted -- the table openMSX ships)
TMS9918_15 = np.array([
    (0, 0, 0), (33, 200, 66), (94, 220, 120), (84, 85, 237), (125, 118, 252),
    (180, 85, 33), (32, 206, 227), (255, 85, 85), (255, 121, 120),
    (204, 194, 63), (222, 208, 135), (33, 176, 59), (201, 91, 201),
    (204, 204, 204), (255, 255, 255)], dtype=np.uint8)

#: attribute_cells item -> (cell width, cell height) in OUTPUT pixels; row
#: 0 is the picture's bottom row and cells are anchored bottom-left (a
#: partial cell at the top or right edge is the pixels it has)
CELL_SIZE = {'ZX_SPECTRUM': (8, 8), 'MSX1': (8, 1), 'C64_HIRES': (8, 8),
             'C64_MULTI': (4, 8)}


def _c64_8():
    return np.round(PA.C64_16 * np.float32(255)).astype(np.int32)


@functools.lru_cache(maxsize=None)
def cells_sets(mode, bg=0):
    """(MACH int32 (M, 3), SET_IDX int32 (K, S)): the machine's base
    colours and, per candidate set k, the S base-colour indices a cell may
    hold. ZX: k = bright * 64 + ink * 8 + paper (K 128, S 2; the base
    array is the 16 rows bright * 8 + c). MSX1: k = (a - 1) * 15 + (b - 1)
    (K 225). C64_HIRES: k = a * 16 + b (K 256). C64_MULTI: the 560 triples
    a < b < c in lexicographic order, each with the screen-wide `bg`
    first (S 4)."""
    mode = str(mode)
    if mode == 'ZX_SPECTRUM':
        mach = ZX_LEVELS.reshape(16, 3).astype(np.int32)
        idx = np.array([(br * 8 + ink, br * 8 + paper) for br in range(2)
                        for ink in range(8) for paper in range(8)], np.int32)
    elif mode == 'MSX1':
        mach = TMS9918_15.astype(np.int32)
        idx = np.array([(a, b) for a in range(15) for b in range(15)],
                       np.int32)
    elif mode == 'C64_HIRES':
        mach = _c64_8()
        idx = np.array([(a, b) for a in range(16) for b in range(16)],
                       np.int32)
    elif mode == 'C64_MULTI':
        mach = _c64_8()
        bg = int(bg)
        idx = np.array([(bg, a, b, c) for a in range(16)
                        for b in range(a + 1, 16)
                        for c in range(b + 1, 16)], np.int32)
    else:
        raise ValueError(f'attribute_cells {mode!r}')
    mach.setflags(write=False)
    idx.setflags(write=False)
    return mach, idx


@functools.lru_cache(maxsize=None)
def cells_spacing(mode):
    """The ordered dither's step for a machine: DI._palette_spacing over
    its DISTINCT colours (the Spectrum's fifteen: its two blacks are one
    colour), a Python float as DI.ordered_palette's is."""
    mach, _idx = cells_sets(str(mode), 0)
    uniq = np.unique(mach, axis=0)
    return DI._palette_spacing((uniq.astype(np.float32) / np.float32(255)
                                ).astype(np.float32))


def cells_sets_image(mode, bg=0):
    """The CELLS stages' `sets` texture: an (S, K, 4) float32 image with
    .rgb[s, k] the set colour as 0..255 floats (texel x = k, y = s)."""
    mach, idx = cells_sets(str(mode), int(bg))
    k, s = idx.shape
    img = np.zeros((s, k, 4), np.float32)
    img[:, :, :3] = mach[idx].transpose(1, 0, 2).astype(np.float32)
    img[:, :, 3] = 1.0
    return img


def cells_dither(st):
    """The ordered kind the cell fit perturbs by, or None: only DI.ORDERED's
    square power-of-two matrices have a seat (error diffusion, NOISE and
    Blue Noise are inert -- the era's converters were ordered)."""
    kind = str(getattr(st, 'dither', 'NONE'))
    m = DI.ORDERED.get(kind)
    if m is None or m.shape[0] != m.shape[1] or \
            (m.shape[0] & (m.shape[0] - 1)) != 0:
        return None
    return kind


def cells_bg_key(seed=0):
    return ('C64_MULTI_BG', int(seed))


def cells_bg_count(rgb):
    """The C64 multicolour screen background: the lowest index maximising
    the count of pixels whose nearest of the sixteen (integer squared
    distance on the UNPERTURBED frame's 8-bit values, np.argmin = lowest
    index on ties) it is."""
    r8 = np.round(np.clip(np.asarray(rgb, np.float32), 0.0, 1.0)
                  * np.float32(255)).astype(np.int32).reshape(-1, 3)
    c64 = _c64_8()
    counts = np.zeros(16, np.int64)
    for s in range(0, r8.shape[0], 65536):
        d = ((r8[s:s + 65536, None, :] - c64[None, :, :]) ** 2).sum(-1)
        counts += np.bincount(np.argmin(d, axis=1), minlength=16)
    return int(np.argmax(counts))


def cells_bg(rgb, st, seed=0):
    """The background index for this frame: counted once per lock (cached
    like the adaptive palette) or from every frame when the lock is off."""
    if getattr(st, 'palette_lock', True):
        return int(PA.cached_adaptive(cells_bg_key(seed),
                                      lambda: cells_bg_count(rgb)))
    return cells_bg_count(rgb)


def attribute_cells(rgb, st, seed=0):
    """Colour per character cell: every cell holds only its machine's two
    colours (three plus the screen background in C64 multicolour), the set
    chosen by least squared error over the cell's pixels on the 8-bit
    values. Integers end to end after the 8-bit round. Ties: np.round half
    to even on the 8-bit value; np.argmin = the lowest set k, then the
    lowest colour s. Order-free: a cell's fit is a sum over its pixels."""
    rgb = np.asarray(rgb, np.float32)
    mode = str(st.attribute_cells)
    cw, ch = CELL_SIZE[mode]
    h, w = rgb.shape[:2]
    bg = cells_bg(rgb, st, seed) if mode == 'C64_MULTI' else 0
    mach, set_idx = cells_sets(mode, bg)
    n_sets = set_idx.shape[0]
    kind = cells_dither(st)
    if kind is None:
        note_ignored('attribute cells', getattr(st, 'dither', 'NONE'))
        pert = rgb
    else:
        tm = DI.threshold_map(kind, h, w)
        strength = float(st.dither_strength)
        spacing = cells_spacing(mode)
        pert = rgb + ((tm - 0.5) * strength * spacing)[..., None]
    p8 = np.round(np.clip(pert, 0.0, 1.0) * np.float32(255)).astype(np.int32)
    cells_x = -(-w // cw)
    wp = cells_x * cw
    out8 = np.empty((h, w, 3), np.int32)
    xcell = np.arange(w) // cw
    for y0 in range(0, h, ch):
        blk = p8[y0:y0 + ch]                                   # (rows, w, 3)
        rows = blk.shape[0]
        dm = ((blk[:, :, None, :] - mach[None, None, :, :]) ** 2).sum(-1)
        dset = dm[:, :, set_idx].min(axis=3)                   # (rows, w, K)
        if wp != w:                                            # a partial cell
            pad = np.zeros((rows, wp, n_sets), dset.dtype)     # is the pixels
            pad[:, :w] = dset                                  # it has
            dset = pad
        cost = dset.reshape(rows, cells_x, cw, n_sets).sum(axis=(0, 2))
        kstar = np.argmin(cost, axis=1)                        # (cells_x,)
        choice = set_idx[kstar][xcell]                         # (w, S)
        dsel = np.take_along_axis(dm, np.broadcast_to(
            choice[None], (rows,) + choice.shape), axis=2)     # (rows, w, S)
        sstar = np.argmin(dsel, axis=2)                        # (rows, w)
        pick = np.take_along_axis(np.broadcast_to(
            choice[None], (rows,) + choice.shape), sstar[..., None],
            axis=2)[..., 0]
        out8[y0:y0 + ch] = mach[pick]
    return LUT255[out8]


def cells_fit(rgb, st, seed=0):
    """The (cells_y, cells_x) int32 array of the chosen set index per cell
    (what the CELLS_FIT pass draws) -- the tests' and the self test's view
    of the first pass."""
    rgb = np.asarray(rgb, np.float32)
    mode = str(st.attribute_cells)
    cw, ch = CELL_SIZE[mode]
    h, w = rgb.shape[:2]
    bg = cells_bg(rgb, st, seed) if mode == 'C64_MULTI' else 0
    mach, set_idx = cells_sets(mode, bg)
    kind = cells_dither(st)
    pert = rgb
    if kind is not None:
        tm = DI.threshold_map(kind, h, w)
        pert = rgb + ((tm - 0.5) * float(st.dither_strength)
                      * cells_spacing(mode))[..., None]
    p8 = np.round(np.clip(pert, 0.0, 1.0) * np.float32(255)).astype(np.int32)
    cells_x, cells_y = -(-w // cw), -(-h // ch)
    out = np.zeros((cells_y, cells_x), np.int32)
    for cy in range(cells_y):
        blk = p8[cy * ch:(cy + 1) * ch]
        dm = ((blk[:, :, None, :] - mach[None, None, :, :]) ** 2).sum(-1)
        dset = dm[:, :, set_idx].min(axis=3)
        for cx in range(cells_x):
            out[cy, cx] = int(np.argmin(
                dset[:, cx * cw:(cx + 1) * cw].sum(axis=(0, 1))))
    return out


# --------------------------------------- C060: per-scanline palettes

#: scanline_palette item -> levels per channel of the machine's registers
SCANLINE_LEVELS = {'SPECTRUM_512': 8, 'DYNAMIC_HIRES': 16, 'SHAM': 16}


def scanline_zones(mode, w):
    """The register-reload zones of one line: three thirds for Spectrum
    512 (the last takes the remainder), the whole line otherwise. Three
    fixed thirds stand in for Spectrum 512's staggered reloads: the
    counts (48 a line, 16 at any x) are the machine's, the timing is not."""
    w = int(w)
    if str(mode) == 'SPECTRUM_512' and w >= 3:
        return [(0, w // 3), (w // 3, 2 * w // 3), (2 * w // 3, w)]
    return [(0, w)]


def _scanline_pert(rgb, st, levels):
    """(base, pert): the clipped UNPERTURBED pixels the fit reads and the
    pixels the search sees -- an ordered dither perturbs by its threshold
    in lattice steps (DI.ordered_bits's own unit, over levels - 1) AFTER
    the fit; diffusion, NOISE and Blue Noise are inert."""
    h, w = rgb.shape[:2]
    base = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    kind = cells_dither(st)
    if kind is not None:
        tm = DI.threshold_map(kind, h, w)
        strength = float(st.dither_strength)
        pert = rgb + ((tm - 0.5) * strength
                      / np.float32(levels - 1))[..., None]
    else:
        note_ignored('scanline palette', getattr(st, 'dither', 'NONE'))
        pert = rgb
    return base, np.clip(pert, 0.0, 1.0).astype(np.float32)


def scanline_palettes(rgb, st, seed=0):
    """The (H, zones, 16, 3) float32 register tables: per line and zone,
    Halcyon's median cut over the UNPERTURBED pixels of that zone, snapped
    to the machine's lattice (3:3:3 for the ST, 4:4:4 for the Amiga)."""
    rgb = np.asarray(rgb, np.float32)
    mode = str(st.scanline_palette)
    levels = SCANLINE_LEVELS[mode]
    h, w = rgb.shape[:2]
    zones = scanline_zones(mode, w)
    base = np.clip(rgb, 0.0, 1.0).astype(np.float32)
    out = np.zeros((h, len(zones), 16, 3), np.float32)
    for y in range(h):
        for z, (x0, x1) in enumerate(zones):
            out[y, z] = PA.snap_levels(PA.median_cut(base[y, x0:x1], 16),
                                       levels)
    return out


def scanline_palette(rgb, st, seed=0):
    """The palette registers rewritten every scanline: each line (each
    third of a line for Spectrum 512) is fitted with a 16-colour median
    cut snapped to the machine's registers and searched by float32
    squared distance (np.argmin = lowest index on ties); Sliced HAM
    restarts the HAM6 encode on every line with that line's sixteen (the
    modifies win on a strict <). Rows are independent; the fit is a
    reduction of the zone; SHAM's encode is sequential along the row --
    CPU only, refused by name on the GPU road."""
    rgb = np.asarray(rgb, np.float32)
    mode = str(st.scanline_palette)
    levels = SCANLINE_LEVELS[mode]
    h, w = rgb.shape[:2]
    zones = scanline_zones(mode, w)
    base, pert = _scanline_pert(rgb, st, levels)
    out = np.empty((h, w, 3), np.float32)
    for y in range(h):
        for x0, x1 in zones:
            pal = PA.snap_levels(PA.median_cut(base[y, x0:x1], 16), levels)
            px = pert[y, x0:x1]
            if mode != 'SHAM':
                out[y, x0:x1] = pal[PA.nearest_brute(px, pal)]
            else:
                out[y] = PA.sham_row(
                    (np.round(px * np.float32(15)) / np.float32(15)
                     ).astype(np.float32), pal)
    return out
