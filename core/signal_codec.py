"""R251: the post-signal pack's codec and optical-printer stages (SIG-3).

Three stages, bpy-free and device-free, `(rgb, st, ...)` in, float32
`(H, W, 3)` out, row 0 the bottom of the picture (post.process):

    C134  matte_glow   Tron's backlit Kodalith mattes: every material with
                       a Glow Gel is a clear-or-opaque matte exposed in its
                       gel's colour, once crisp and once per diffusion pass
                       at doubling radius and halving exposure, summed on
                       the LINEAR frame before the film stages
    C132  mpeg1_intra  Video CD: 8x8 DCT blocks in 4:2:0 through MPEG-1's
                       default intra matrix times a linear quantiser scale,
                       the 8-bit DC in steps of 8, the mismatch-control
                       oddification, the GOP pumping the scale per frame
    C133  smacker      RAD's Smacker: a 256-colour frame palette and 4x4
                       blocks coded Fill (one colour), Mono (two entries
                       and a mask) or Full (sixteen)

Every stage function OPENS with `if not <name>_on(st): return rgb`, so
it is the identity on any settings that do not switch it on (the
activity test lives in the wiring AND here).

The arithmetic both roads share, by construction:
- the matte: float32 taps computed once in float64 and rounded once
  (data both roads read), summed in tap order -R..R with one multiply
  and one add per tap, horizontal then vertical, every pass from the
  GEL (never cascaded); the exposures added crisp first, then passes
  1..N. No upper clamp: the display stage's clip is the film's shoulder.
- MPEG-1: the pack's own float32 DCT table (float64 once, rounded
  once); per plane row DCT, column DCT, quantise, row IDCT, column
  IDCT, each an 8-term sum in k order with one multiply and one add per
  term; the quantiser's reciprocal `1 / (2 qs W)` rounded ONCE to
  float32 and fetched by both roads. Ties: DC half to even, AC
  truncation after +-0.5, oddification toward zero, the IDCT's output
  half to even, the chroma quad `+2 >> 2`.
- Smacker: integers end to end on the frame's bytes; the palette's
  float32 colours are fetched, never recomputed; ties by (luma, index)
  in the rank, `<=` toward the dark entry in a mono block, the inverse
  colormap's own tie baked into its table.

The GPU twins are gpu/stages_codec.py (MATTE_BLUR, MATTE_ADD, the five
MPEG_* passes, SMACKER), orchestrated by gpu/chain_codec.py.
"""

import dataclasses
import math

import numpy as np

from . import palette as PA
from .signal_era import (U8, _once, cached_peek, cb601, cr601, to_u8, y601,
                         yuv601_decode)

# ====================================================================
# C134: the backlit matte glow (Tron, 1982)
# ====================================================================

#: the GPU blur's literal loop bound: taps -160..160
MATTE_LOOP = 160
#: the most diffusion passes one matte is exposed through (the MATTE_ADD
#: stage binds b1..b5)
MATTE_MAX_PASSES = 5


def matte_glow_on(st):
    return bool(getattr(st, 'matte_glow', False))


def gel_plane(gbuf, mesh, materials, ss):
    """The gel colour of the material under every OUTPUT pixel, (H, W, 3)
    float32: black where the pixel is empty or its material has no gel.

    The matte is "this material has a gel", one bit, in the gel's
    colour. Read off the G-buffer's triangle ids at the internal
    resolution and taken at the top-left sample of each output pixel
    (`[::ss, ::ss]`, as scene.last_depth is: an id is data, never
    averaged). A mesh with no material indices gives a black plane, so
    the stage adds nothing and prints nothing -- the plane exists."""
    cov = np.asarray(gbuf.mask(), bool)
    mats = list(materials or ())
    n = len(mats)
    table = np.zeros((n + 1, 3), np.float32)
    for i, m in enumerate(mats):
        g = getattr(m, 'glow_gel', None)
        if g is None:
            continue
        g = np.asarray(tuple(g)[:3], np.float32)
        if g.shape == (3,):
            table[i] = np.maximum(g, np.float32(0.0))
    mi = getattr(mesh, 'mat_index', None) if mesh is not None else None
    mid = np.full(cov.shape, n, np.int32)
    if mi is not None and n > 0:
        tri = np.asarray(gbuf.tri)
        ids = np.asarray(mi, np.int32)[tri[cov]]
        ids = np.where((ids >= 0) & (ids < n), ids, n).astype(np.int32)
        mid[cov] = ids
    ss = max(int(ss), 1)
    if ss > 1:
        mid = mid[::ss, ::ss]
    return np.ascontiguousarray(table[mid], np.float32)


def matte_taps(sigma):
    """The Gaussian of one diffusion pass as float32 taps -R..R (R =
    max(ceil(3 sigma), 1)): float64 once, normalised to sum 1, rounded
    ONCE to float32 -- data both roads read."""
    sigma = max(float(sigma), 1e-6)
    R = max(int(math.ceil(3.0 * sigma)), 1)
    k = np.arange(-R, R + 1, dtype=np.float64)
    w = np.exp(-(k * k) / (2.0 * sigma * sigma))
    w = w / w.sum()
    return w.astype(np.float32)


@dataclasses.dataclass
class MatteParams:
    e: np.float32          # the crisp exposure
    taps: list             # per pass: float32 taps -R..R
    w: list                # per pass: np.float32(exposure / 2**k)
    radii: list            # per pass: R


def matte_params(st, h):
    """The printer's exposures for a frame `h` rows tall: the first
    diffusion pass's sigma is Matte Radius pixels at 1080 lines; each
    further pass doubles it and halves its exposure."""
    radius = float(st.matte_glow_radius)
    passes = min(max(int(st.matte_glow_passes), 1), MATTE_MAX_PASSES)
    exposure = float(st.matte_glow_exposure)
    taps, ws, radii = [], [], []
    for k in range(1, passes + 1):
        sigma = radius * float(2 ** (k - 1)) * float(h) / 1080.0
        t = matte_taps(sigma)
        taps.append(t)
        radii.append((t.size - 1) // 2)
        ws.append(np.float32(exposure / float(2 ** k)))
    return MatteParams(np.float32(exposure), taps, ws, radii)


def _blur_axis(img, taps, axis):
    img = np.asarray(img, np.float32)
    R = (int(taps.size) - 1) // 2
    n = img.shape[axis]
    pos = np.arange(n)
    acc = np.zeros(img.shape, np.float32)
    for i in range(-R, R + 1):
        idx = np.clip(pos + i, 0, n - 1)
        t = np.take(img, idx, axis=axis) * np.float32(taps[i + R])
        acc = acc + t
    return acc


def blur_h(img, taps):
    """One tap set along the rows: i in -R..R in order, one multiply and
    one add per tap in float32, the index clamped to the frame."""
    return _blur_axis(img, taps, 1)


def blur_v(img, taps):
    """The same tap set along the columns."""
    return _blur_axis(img, taps, 0)


def matte_glow_why(rgb, gel, st=None):
    """Why the stage is a no-op on this frame, or None when the gel
    plane is there and matches the frame."""
    if st is not None and getattr(st, '_viewport', False):
        return ('matte glow: the optical printer is an F12 stage; the '
                'viewport draws the frame without the mattes\' exposures')
    if gel is None:
        return ('matte glow: no gel plane reached post (render through '
                'the engine, or pass gel=); the stage is a no-op')
    g = np.asarray(gel)
    shape = tuple(np.asarray(rgb).shape[:2]) if not isinstance(rgb, tuple) \
        else tuple(rgb[:2])
    if g.ndim != 3 or g.shape[2] != 3 or tuple(g.shape[:2]) != shape:
        return (f'matte glow: the gel plane {tuple(g.shape)} does not match '
                f'the frame {shape} (a stitched, stereo or resized frame '
                'has no single material plane); the stage is a no-op')
    return None


def matte_glow_ready(st, gel, shape):
    """The wiring's gate: True when the stage has its gel plane; else
    the named reason is printed once and the frame is left where it is
    (no readback is spent on a no-op)."""
    why = matte_glow_why(tuple(shape), gel, st)
    if why:
        _once(why)
        return False
    return True


def matte_glow(rgb, st, gel=None):
    """The optical printer: `rgb` (linear, H x W x 3) plus every matte's
    exposures in its gel's colour."""
    if not matte_glow_on(st):
        return rgb
    why = matte_glow_why(rgb, gel, st)
    if why:
        _once(why)
        return rgb
    rgb = np.asarray(rgb, np.float32)
    gel = np.asarray(gel, np.float32)
    P = matte_params(st, rgb.shape[0])
    t = gel * P.e
    out = rgb + t                                   # the crisp exposure
    for k in range(len(P.taps)):
        b = blur_v(blur_h(gel, P.taps[k]), P.taps[k])
        t = b * P.w[k]
        out = out + t
    return np.maximum(out, np.float32(0.0))


def matte_taps_image(taps):
    """One pass's taps as the (1, 321, 4) float32 image the MATTE_BLUR
    stage fetches: r = the tap centred at texel 160, zero beyond R."""
    img = np.zeros((1, 2 * MATTE_LOOP + 1, 4), np.float32)
    R = (int(taps.size) - 1) // 2
    img[0, MATTE_LOOP - R:MATTE_LOOP + R + 1, 0] = taps
    img[0, :, 3] = 1.0
    return img


# ====================================================================
# C132: MPEG-1 intra blocks (Video CD, 1993)
# ====================================================================

#: ISO/IEC 11172-2's default intra quantiser matrix, W[v][u]
MPEG1_INTRA = np.array(
    [[8, 16, 19, 22, 26, 27, 29, 34], [16, 16, 22, 24, 27, 29, 34, 37],
     [19, 22, 26, 27, 29, 34, 34, 38], [22, 22, 26, 27, 29, 34, 37, 40],
     [22, 26, 27, 29, 32, 35, 40, 48], [26, 27, 29, 32, 35, 40, 48, 58],
     [26, 27, 29, 34, 38, 46, 56, 69], [27, 29, 35, 38, 46, 56, 69, 83]],
    np.int32)


def _dct8():
    k = np.arange(8, dtype=np.float64)
    D = np.zeros((8, 8), np.float64)
    for u in range(8):
        c = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
        D[u] = c * np.cos((2.0 * k + 1.0) * u * math.pi / 16.0)
    return D.astype(np.float32)


#: the pack's own orthonormal DCT-II table D[u][k], float64 once,
#: rounded once to float32 (not post._dct_matrix; not an encoder's
#: integer IDCT -- disclosed)
DCT8 = _dct8()


def mpeg1_on(st):
    return bool(getattr(st, 'mpeg1', False))


def mpeg1_qs(st, frame):
    """This frame's quantizer_scale: the I scale on I and P pictures
    (Test Model 5's K_p = 1.0; every third picture of the group is a P),
    1.4 times it on B pictures (K_b = 1.4, half up), capped at 31.
    A group of 0 codes every frame as an I picture."""
    qs_i = min(max(int(st.mpeg1_qscale), 1), 31)
    gop = max(int(st.mpeg1_gop), 0)
    if gop == 0:
        return qs_i
    k = int(frame) % gop
    if k == 0 or k % 3 == 0:
        return qs_i
    return min(31, (14 * qs_i + 5) // 10)


@dataclasses.dataclass
class Mpeg1Params:
    qs: int
    Wp: int
    Hp: int
    inv: np.ndarray        # float32 (8, 8): 1 / (2 qs W[v][u]), rounded once


def mpeg1_inv(qs):
    return (np.float32(1.0) / (np.float32(2 * int(qs))
                               * MPEG1_INTRA.astype(np.float32))
            ).astype(np.float32)


def mpeg1_params(st, frame, h, w):
    qs = mpeg1_qs(st, frame)
    return Mpeg1Params(int(qs), ((int(w) + 15) // 16) * 16,
                       ((int(h) + 15) // 16) * 16, mpeg1_inv(qs))


def _mpeg1_quant(X, qs, inv):
    """Quantise and reconstruct a plane of coefficients (float32, the
    block position of a sample is (y & 7, x & 7) = (v, u)): the DC in
    steps of 8, half to even; every AC level truncated after +-0.5,
    rebuilt as `(|QF| qs W) >> 3` and made odd toward zero (mismatch
    control), clipped to -2048..2047. float32 out."""
    X = np.asarray(X, np.float32)
    h, w = X.shape
    v = (np.arange(h) & 7)[:, None]
    u = (np.arange(w) & 7)[None, :]
    Wm = MPEG1_INTRA[v, u]
    iv = np.asarray(inv, np.float32)[v, u]
    dc = np.round(X * np.float32(0.125)) * np.float32(8.0)
    t = X * np.float32(16.0)
    t = t * iv
    s = np.where(X > 0, np.float32(0.5),
                 np.where(X < 0, np.float32(-0.5), np.float32(0.0))
                 ).astype(np.float32)
    t = t + s
    qf = np.trunc(t).astype(np.int32)
    m = np.abs(qf) * np.int32(qs) * Wm
    m = m >> 3
    m = np.where((m != 0) & ((m & 1) == 0), m - 1, m)
    r = np.where(qf < 0, -m, m)
    r = np.clip(r, -2048, 2047).astype(np.float32)
    return np.where((v == 0) & (u == 0), dc, r).astype(np.float32)


def _dct_rows(p, table_of):
    """sum over k of table_of(u, k) * p[:, col0 + k], k in order."""
    h, w = p.shape
    x = np.arange(w)
    u = x & 7
    col0 = x & ~7
    acc = np.zeros((h, w), np.float32)
    for k in range(8):
        t = p[:, col0 + k] * table_of(u, k)[None, :]
        acc = acc + t
    return acc


def _dct_cols(p, table_of):
    h, w = p.shape
    y = np.arange(h)
    v = y & 7
    row0 = y & ~7
    acc = np.zeros((h, w), np.float32)
    for k in range(8):
        t = p[row0 + k, :] * table_of(v, k)[:, None]
        acc = acc + t
    return acc


def _fwd(u, k):
    return DCT8[u, k]


def _inv_t(u, k):
    return DCT8[k, u]


def _mpeg1_plane_stages(p, qs, inv):
    """(row DCT, quantised coefficients, row IDCT, decoded bytes) of one
    plane (float32, sides multiples of 8)."""
    p = np.asarray(p, np.float32)
    r1 = _dct_rows(p, _fwd)
    X = _dct_cols(r1, _fwd)
    Rq = _mpeg1_quant(X, qs, inv)
    i1 = _dct_rows(Rq, _inv_t)
    O = _dct_cols(i1, _inv_t)
    out = np.clip(np.round(O) + np.float32(128.0), 0.0, 255.0).astype(np.int32)
    return r1, Rq, i1, out


def _mpeg1_plane(p, qs, inv):
    return _mpeg1_plane_stages(p, qs, inv)[3]


def _mpeg1_encode(v8):
    """The padded bytes -> (Y - 128, Cb - 128 per 2x2 cell, Cr - 128 per
    cell) as float32 planes: BT.601 integers, the chroma quad averaged
    `(a + b + c + d + 2) >> 2` (4:2:0, centred)."""
    R, G, B = v8[..., 0], v8[..., 1], v8[..., 2]
    Y = y601(R, G, B)
    Cb = cb601(R, G, B)
    Cr = cr601(R, G, B)

    def quad(c):
        return (c[0::2, 0::2] + c[0::2, 1::2] + c[1::2, 0::2]
                + c[1::2, 1::2] + 2) >> 2
    return ((Y - 128).astype(np.float32), (quad(Cb) - 128).astype(np.float32),
            (quad(Cr) - 128).astype(np.float32))


def _mpeg1_stages(v8, P):
    """The CPU's intermediates in the GPU passes' own layout (float32
    (Hp, Wp, 3): r = luma, g / b = the chroma of the pixel's 2x2 cell),
    for the twin tests: ENC, DCT_ROW, DCT_COL_Q, IDCT_ROW, and the
    decoded bytes (Hp, Wp, 3) int32."""
    y, cb, cr = _mpeg1_encode(v8)

    def up(c):
        return np.repeat(np.repeat(c, 2, axis=0), 2, axis=1)
    planes = [_mpeg1_plane_stages(p, P.qs, P.inv) for p in (y, cb, cr)]
    enc = np.stack([y, up(cb), up(cr)], 2).astype(np.float32)
    outs = [enc]
    for i in range(3):
        outs.append(np.stack([planes[0][i], up(planes[1][i]),
                              up(planes[2][i])], 2).astype(np.float32))
    R, G, B = yuv601_decode(planes[0][3], up(planes[1][3]), up(planes[2][3]))
    outs.append(np.stack([R, G, B], 2).astype(np.int32))
    return outs


def mpeg1_intra(rgb, st, frame=0):
    """The frame recoded as MPEG-1 intra blocks (float32 on the byte
    lattice)."""
    if not mpeg1_on(st):
        return rgb
    rgb = np.asarray(rgb, np.float32)
    h, w = rgb.shape[:2]
    P = mpeg1_params(st, frame, h, w)
    v8 = np.pad(to_u8(rgb), ((0, P.Hp - h), (0, P.Wp - w), (0, 0)), 'edge')
    y, cb, cr = _mpeg1_encode(v8)
    y8 = _mpeg1_plane(y, P.qs, P.inv)
    cb8 = _mpeg1_plane(cb, P.qs, P.inv)
    cr8 = _mpeg1_plane(cr, P.qs, P.inv)
    yy = np.arange(P.Hp)[:, None] >> 1
    xx = np.arange(P.Wp)[None, :] >> 1
    # chroma held per cell: the VCD decoder's cheap upsample (named)
    R, G, B = yuv601_decode(y8, cb8[yy, xx], cr8[yy, xx])
    out = np.stack([R, G, B], 2)[:h, :w]
    return np.ascontiguousarray(U8[out], np.float32)


def mpeg1_dct_image():
    """DCT8 as the (8, 8, 4) float32 image the MPEG passes fetch:
    r = D[u][k] at texel (k, u)."""
    img = np.zeros((8, 8, 4), np.float32)
    img[:, :, 0] = DCT8
    img[:, :, 3] = 1.0
    return img


def mpeg1_q_image(qs):
    """The quantiser table of one scale as an (8, 8, 4) float32 image:
    r = W[v][u] (as float), g = 1 / (2 qs W[v][u]) at texel (u, v)."""
    img = np.zeros((8, 8, 4), np.float32)
    img[:, :, 0] = MPEG1_INTRA.astype(np.float32)
    img[:, :, 1] = mpeg1_inv(qs)
    img[:, :, 3] = 1.0
    return img


# ====================================================================
# C133: Smacker 4x4 blocks (RAD Game Tools, 1994)
# ====================================================================

def smacker_on(st):
    return bool(getattr(st, 'smacker', False))


def _smacker_key(st, seed):
    return ('ADAPTIVE', 256, str(st.palette_method), int(seed))


def smacker_tables(pal):
    """(palette float32 (N, 3), the cached 6-bit inverse colormap's
    int32 table) for a palette."""
    pal = np.ascontiguousarray(np.asarray(pal, np.float32).reshape(-1, 3))
    icm = PA.get_inverse_colormap(pal, 6)
    return pal, np.asarray(icm.lut, np.int32)


def smacker_palette(st, rgb, seed):
    """The codec's 256-colour frame palette, under the SAME lock key the
    palette quantiser uses (post._palette_for), so a following 8-bit
    quant maps onto these colours; and its cached inverse colormap."""
    rgb = np.asarray(rgb, np.float32)

    def build():
        return PA.get_palette('ADAPTIVE', 256, rgb.reshape(-1, 3),
                              st.palette_method, seed)
    if getattr(st, 'palette_lock', True):
        pal = PA.cached_adaptive(_smacker_key(st, seed), build)
    else:
        pal = build()
    return smacker_tables(pal)


def smacker_palette_cached(st, seed):
    """The locked palette's tables when the lock already holds them
    (no pixels needed), else None."""
    if not getattr(st, 'palette_lock', True):
        return None
    pal = cached_peek(_smacker_key(st, seed))
    if pal is None:
        return None
    return smacker_tables(pal)


def smacker_thresholds(q):
    """(t_fill, t_full) of the encoder's budget: a block whose luma
    range is under t_fill is one colour; a block whose two-colour error
    exceeds t_full keeps all sixteen. Halcyon's own thresholds (RAD's
    encoder decisions are unpublished), the codec's block grammar."""
    q = min(max(float(q), 0.0), 1.0)
    t_fill = int(round(255.0 * (0.02 + 0.2 * (1.0 - q))))
    tq = int(round(64.0 * (1.0 - q)))
    return t_fill, 48 * tq * tq


def nearest8(lut, pal8, c8):
    """The palette index of 8-bit colours through the inverse colormap
    (the cell by integer floor `(c * 64) // 255`, capped at 63) and the
    entry's own 8-bit colour."""
    c8 = np.asarray(c8, np.int32)
    q = np.clip((c8 * 64) // 255, 0, 63)
    idx = lut[(q[..., 0] << 12) | (q[..., 1] << 6) | q[..., 2]]
    return idx, pal8[idx]


def _d2(a, b):
    d = a - b
    return d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2]


def smacker_with(rgb, pal, lut, quality, detail=False):
    """The block coder over a given palette and cube (what both roads
    run once the palette exists). `detail=True` also returns the block
    kinds (0 fill, 1 mono, 2 full) as (Hb, Wb) int32."""
    rgb = np.asarray(rgb, np.float32)
    h, w = rgb.shape[:2]
    pal = np.asarray(pal, np.float32)
    pal8 = to_u8(pal)
    t_fill, t_full = smacker_thresholds(quality)
    v8 = np.pad(to_u8(rgb), ((0, (-h) % 4), (0, (-w) % 4), (0, 0)), 'edge')
    Hp, Wp = v8.shape[:2]
    hb, wb = Hp // 4, Wp // 4
    nb = hb * wb
    # pixel i = dy * 4 + dx, dy from the block's lowest row (row 0 = bottom)
    B = v8.reshape(hb, 4, wb, 4, 3).transpose(0, 2, 1, 3, 4).reshape(nb, 16, 3)
    lum = (77 * B[..., 0] + 150 * B[..., 1] + 29 * B[..., 2]) >> 8
    lmin = lum.min(1)
    lmax = lum.max(1)
    mean = (B.sum(1) + 8) >> 4
    i_mean, _p = nearest8(lut, pal8, mean)
    # rank by (luma, index): exactly eight dark pixels per block
    lt = lum[:, None, :] < lum[:, :, None]
    eq = lum[:, None, :] == lum[:, :, None]
    jlt = np.arange(16)[None, :] < np.arange(16)[:, None]
    rank = (lt | (eq & jlt[None])).sum(2)
    dark = rank < 8
    s0 = (B * dark[..., None]).sum(1)
    s1 = (B * (~dark)[..., None]).sum(1)
    m0 = (s0 + 4) >> 3
    m1 = (s1 + 4) >> 3
    i0, p0 = nearest8(lut, pal8, m0)
    i1, p1 = nearest8(lut, pal8, m1)
    d0 = _d2(B, p0[:, None, :])
    d1 = _d2(B, p1[:, None, :])
    E = np.minimum(d0, d1).sum(1)
    i_full, _p = nearest8(lut, pal8, B)
    fill = (lmax - lmin) < t_fill
    full = ~fill & (E > t_full)
    idx = np.where(fill[:, None], i_mean[:, None],
                   np.where(full[:, None], i_full,
                            np.where(d0 <= d1, i0[:, None], i1[:, None])))
    out = pal[idx].reshape(hb, wb, 4, 4, 3).transpose(0, 2, 1, 3, 4)
    out = np.ascontiguousarray(out.reshape(Hp, Wp, 3)[:h, :w], np.float32)
    if detail:
        kind = np.where(fill, 0, np.where(full, 2, 1)).astype(np.int32)
        return out, kind.reshape(hb, wb)
    return out


def smacker(rgb, st, seed=0):
    """The frame recoded as Smacker blocks over its 256-colour palette
    (every output pixel is a palette entry's own float32 colour)."""
    if not smacker_on(st):
        return rgb
    rgb = np.asarray(rgb, np.float32)
    pal, lut = smacker_palette(st, rgb, seed)
    return smacker_with(rgb, pal, lut, float(st.smacker_quality))
