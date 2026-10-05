"""Dithering.

Ordered (Bayer) dithering is fully vectorised. Error diffusion is inherently
sequential, so it runs a per-pixel loop against an inverse colormap -- which is
precisely how the period software did it, and fast enough because the lookup is
O(1).
"""

import numpy as np


def bayer(n):
    """Recursive Bayer threshold matrix of size n x n (n = 2,4,8,16)."""
    m = np.array([[0, 2], [3, 1]], dtype=np.float32)
    size = 2
    while size < n:
        m = np.block([[4 * m, 4 * m + 2], [4 * m + 3, 4 * m + 1]]).astype(np.float32)
        size *= 2
    return m / (size * size)


BAYER2 = bayer(2)
BAYER4 = bayer(4)
BAYER8 = bayer(8)

HALFTONE8 = np.array([
    [24, 10, 12, 26, 35, 47, 49, 37],
    [8, 0, 2, 14, 45, 59, 61, 51],
    [22, 6, 4, 16, 43, 57, 63, 53],
    [30, 20, 18, 28, 33, 41, 55, 39],
    [34, 46, 48, 36, 25, 11, 13, 27],
    [44, 58, 60, 50, 9, 1, 3, 15],
    [42, 56, 62, 52, 23, 7, 5, 17],
    [32, 40, 54, 38, 31, 21, 19, 29],
], dtype=np.float32) / 64.0

# (dx, dy, weight) with the divisor folded in
KERNELS = {
    'FLOYD': ([(1, 0, 7), (-1, 1, 3), (0, 1, 5), (1, 1, 1)], 16.0),
    'JJN': ([(1, 0, 7), (2, 0, 5),
             (-2, 1, 3), (-1, 1, 5), (0, 1, 7), (1, 1, 5), (2, 1, 3),
             (-2, 2, 1), (-1, 2, 3), (0, 2, 5), (1, 2, 3), (2, 2, 1)], 48.0),
    'STUCKI': ([(1, 0, 8), (2, 0, 4),
                (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2),
                (-2, 2, 1), (-1, 2, 2), (0, 2, 4), (1, 2, 2), (2, 2, 1)], 42.0),
    'ATKINSON': ([(1, 0, 1), (2, 0, 1),
                  (-1, 1, 1), (0, 1, 1), (1, 1, 1), (0, 2, 1)], 8.0),
    'BURKES': ([(1, 0, 8), (2, 0, 4),
                (-2, 1, 2), (-1, 1, 4), (0, 1, 8), (1, 1, 4), (2, 1, 2)], 32.0),
    'SIERRA': ([(1, 0, 5), (2, 0, 3),
                (-2, 1, 2), (-1, 1, 4), (0, 1, 5), (1, 1, 4), (2, 1, 2),
                (-1, 2, 2), (0, 2, 3), (1, 2, 2)], 32.0),
    'SIERRA_LITE': ([(1, 0, 2), (-1, 1, 1), (0, 1, 1)], 4.0),
}

ORDERED = {'BAYER2': BAYER2, 'BAYER4': BAYER4, 'BAYER8': BAYER8,
           'BAYER16': bayer(16), 'HALFTONE': HALFTONE8}


def threshold_map(kind, h, w, frame=1, seed=0):
    if kind == 'N64_NOISE':
        # R251 C015: the RDP's random alpha compare -- a FULL-FRAME
        # 8-bit hash per (pixel, frame, seed), never a tiled map
        return noise_threshold_map(h, w, frame, seed)
    m = ORDERED.get(kind)
    if m is None:
        return None
    ty = int(np.ceil(h / m.shape[0]))
    tx = int(np.ceil(w / m.shape[1]))
    return np.tile(m, (ty, tx))[:h, :w]


def ordered_bits(img, bits_rgb, kind='BAYER4', strength=1.0):
    """Ordered-dither an image to a fixed per-channel bit depth."""
    h, w = img.shape[:2]
    tm = threshold_map(kind, h, w)
    out = np.empty_like(img)
    for ch in range(3):
        levels = float((1 << bits_rgb[ch]) - 1)
        v = np.clip(img[..., ch], 0.0, 1.0) * levels
        if tm is not None:
            v = v + (tm - 0.5) * strength
        out[..., ch] = np.clip(np.round(v), 0, levels) / levels
    return out


def diffusion_bits(img, bits_rgb, kind='FLOYD', strength=1.0,
                   serpentine=True, seed=0):
    """Error diffusion straight onto per-channel level lattices.

    A High Color lattice (5-6-5 is 65,536 entries) is far too large to
    treat as a palette, but it is separable: the nearest lattice colour is
    the nearest level per channel, so each channel diffuses independently
    against its own grey ramp. This is exactly what a 90s converter did
    when it wrote Floyd-Steinberg RGB565.
    """
    out = np.empty_like(img)
    for ch in range(3):
        ramp = np.linspace(0.0, 1.0, 1 << bits_rgb[ch], dtype=np.float32)
        pal = np.stack([ramp, ramp, ramp], axis=1)
        mono = np.repeat(img[:, :, ch:ch + 1], 3, axis=2)
        out[:, :, ch] = apply_dither(mono, pal, kind, strength,
                                     serpentine, seed=seed)[:, :, 0]
    return out


def ordered_palette(img, palette, kind='BAYER4', strength=1.0, icm=None):
    """Ordered dithering against an arbitrary palette.

    Perturbs each pixel by the threshold matrix scaled by the palette's mean
    nearest-neighbour spacing, then snaps -- the standard approach for
    non-uniform palettes.
    """
    from .palette import InverseColormap
    h, w = img.shape[:2]
    tm = threshold_map(kind, h, w)
    if icm is None:
        icm = InverseColormap(palette)
    spacing = _palette_spacing(palette)
    if tm is not None:
        pert = img + ((tm - 0.5) * strength * spacing)[..., None]
    else:
        pert = img
    idx = icm.lookup(np.clip(pert, 0.0, 1.0))
    return palette[idx], idx


def _palette_spacing(palette):
    p = np.asarray(palette, np.float32)
    if len(p) < 2:
        return 0.5
    n = min(len(p), 256)
    q = p[:n]
    d = ((q[:, None, :] - q[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(d, 1e9)
    return float(np.sqrt(d.min(axis=1)).mean())


_WAVE_CACHE = {}


def _wave_skew(kern):
    """Smallest b making t = x + b*y a valid schedule for this kernel.

    Every kernel offset must land strictly later in t, i.e. dx + b*dy > 0. The
    binding constraints are the offsets that push error left and down, so
    b > max(-dx/dy) over those.
    """
    b = 1
    for dx, dy, _w in kern:
        if dy > 0:
            need = int(np.floor(-dx / dy)) + 1
            b = max(b, need)
        elif dy == 0 and dx <= 0:
            return None                      # cannot be scheduled
    for dx, dy, _w in kern:
        if dx + b * dy <= 0:
            return None
    return b


def _wavefronts(w, h, b):
    """Pixel indices grouped by t = x + b*y, built once per image size."""
    key = (w, h, b)
    hit = _WAVE_CACHE.get(key)
    if hit is not None:
        return hit
    ys = np.arange(h, dtype=np.int64)
    groups = []
    for t in range(0, (w - 1) + b * (h - 1) + 1):
        y0 = max(0, -(-(t - w + 1) // b))
        y1 = min(h - 1, t // b)
        if y1 < y0:
            continue
        y = ys[y0:y1 + 1]
        x = t - b * y
        groups.append((x.astype(np.int64), y.astype(np.int64)))
    if len(_WAVE_CACHE) > 4:
        _WAVE_CACHE.clear()
    _WAVE_CACHE[key] = groups
    return groups


def error_diffusion_wavefront(img, palette, kind='FLOYD', strength=1.0, icm=None):
    """Error diffusion, one anti-diagonal at a time.

    Error diffusion looks strictly serial, but it is not: a pixel only depends
    on neighbours up and to the left, so every pixel on the skewed diagonal
    x + b*y = t is independent of the others on it. Processing a whole diagonal
    at once turns a 307,200-step Python loop at 640x480 into about 1,600 NumPy
    operations, and the result is identical because the dependency order is
    still respected -- ordering among mutually independent pixels cannot matter.

    Within one diagonal and one kernel offset the targets are provably distinct
    -- two different source pixels cannot map to the same target under a fixed
    translation -- so plain fancy indexing accumulates correctly and the far
    slower unbuffered np.add.at is unnecessary. Offsets are applied as separate
    statements, so contributions from different offsets still add up.

    Not valid for serpentine traversal: alternating the scan direction breaks
    the schedule, and the caller falls back to the sequential path.
    """
    from .palette import get_inverse_colormap
    if icm is None:
        icm = get_inverse_colormap(palette)
    kern, div = KERNELS.get(kind, KERNELS['FLOYD'])
    kern = [(dx, dy, wgt / div * strength) for dx, dy, wgt in kern]
    b = _wave_skew(kern)
    if b is None:
        return None
    h, w = img.shape[:2]
    pal = np.asarray(palette, np.float32)
    # float64 to match the sequential path, which accumulates in Python floats.
    # Mixing precisions makes the two paths disagree on borderline pixels.
    buf = np.array(img[:, :, :3], dtype=np.float64)
    pal64 = pal.astype(np.float64)
    idx_map = np.zeros((h, w), np.int32)
    lut = icm.lut
    bits = icm.bits
    n = icm.n
    nmax = n - 1

    for x, y in _wavefronts(w, h, b):
        # R202: the working value is CLAMPED before quantising and
        # before the error -- diffusion against a palette that cannot
        # reach a hue (a sunset table under a blue sky) used to let
        # the un-representable error accumulate along the scan into
        # smears that erased the picture. Bounded error per pixel is
        # what the era's converters did, and it caps the artefact at
        # ordinary dither texture. Kept bit-identical to the
        # sequential road below, which clamps the same way.
        c = np.clip(buf[y, x], 0.0, 1.0)
        q = np.clip(c * n, 0, nmax).astype(np.int32)
        k = lut[(q[:, 0] << (2 * bits)) | (q[:, 1] << bits) | q[:, 2]]
        idx_map[y, x] = k
        chosen = pal64[k]
        err = c - chosen
        buf[y, x] = chosen
        for dx, dy, wgt in kern:
            nx = x + dx
            ny = y + dy
            ok = (nx >= 0) & (nx < w) & (ny < h)
            if not ok.all():
                if not ok.any():
                    continue
                buf[ny[ok], nx[ok]] += err[ok] * wgt
            else:
                buf[ny, nx] += err * wgt
    return buf.astype(np.float32), idx_map


def error_diffusion(img, palette, kind='FLOYD', strength=1.0, serpentine=True,
                    icm=None):
    """Sequential error diffusion. Returns (rgb_out, index_map).

    The dependency on the pixel to the right makes this genuinely serial -- it
    cannot be vectorised the way the rest of the engine is. What it can avoid is
    NumPy's scalar overhead: indexing an ndarray element by element costs far
    more than indexing a Python list, and at three lookups plus several writes
    per pixel that dominates. The buffer is therefore unpacked into flat Python
    floats for the duration of the loop and packed back at the end.
    """
    from .palette import get_inverse_colormap
    if icm is None:
        icm = get_inverse_colormap(palette)
    if not serpentine:
        fast = error_diffusion_wavefront(img, palette, kind, strength, icm)
        if fast is not None:
            return fast
    kern, div = KERNELS.get(kind, KERNELS['FLOYD'])
    kern = [(dx, dy, wgt / div * strength) for dx, dy, wgt in kern]
    h, w = img.shape[:2]
    pal = np.asarray(palette, np.float32)

    buf = np.array(img[:, :, :3], dtype=np.float32, copy=True)
    flat = buf.reshape(-1).tolist()                 # h*w*3 plain floats
    lut = icm.lut.tolist()                          # plain ints
    pal_list = [(float(c[0]), float(c[1]), float(c[2])) for c in pal]
    idx_flat = [0] * (h * w)

    bits = icm.bits
    n = icm.n
    shift_r = 2 * bits
    stride = w * 3
    nmax = n - 1

    for y in range(h):
        base = y * stride
        if serpentine and (y & 1):
            xs = range(w - 1, -1, -1)
            flip = -1
        else:
            xs = range(w)
            flip = 1
        for x in xs:
            o = base + x * 3
            # R202: clamp before quantise AND before the error (the
            # wavefront twin does the same) -- see the note there
            r = flat[o]
            g = flat[o + 1]
            b = flat[o + 2]
            if r < 0.0:
                r = 0.0
            elif r > 1.0:
                r = 1.0
            if g < 0.0:
                g = 0.0
            elif g > 1.0:
                g = 1.0
            if b < 0.0:
                b = 0.0
            elif b > 1.0:
                b = 1.0

            ri = int(r * n)
            gi = int(g * n)
            bi = int(b * n)
            if ri < 0:
                ri = 0
            elif ri > nmax:
                ri = nmax
            if gi < 0:
                gi = 0
            elif gi > nmax:
                gi = nmax
            if bi < 0:
                bi = 0
            elif bi > nmax:
                bi = nmax

            k = lut[(ri << shift_r) | (gi << bits) | bi]
            idx_flat[y * w + x] = k
            pr, pg, pb = pal_list[k]
            er = r - pr
            eg = g - pg
            eb = b - pb
            flat[o] = pr
            flat[o + 1] = pg
            flat[o + 2] = pb

            for dx, dy, wgt in kern:
                nx = x + dx * flip
                if nx < 0 or nx >= w:
                    continue
                ny = y + dy
                if ny >= h:
                    continue
                t = (base + dy * stride) + nx * 3
                flat[t] += er * wgt
                flat[t + 1] += eg * wgt
                flat[t + 2] += eb * wgt

    out = np.asarray(flat, np.float32).reshape(h, w, 3)
    idx_map = np.asarray(idx_flat, np.int32).reshape(h, w)
    return out, idx_map


def noise_dither(img, palette, strength=1.0, seed=0, icm=None):
    from .palette import get_inverse_colormap
    if icm is None:
        icm = get_inverse_colormap(palette)
    rng = np.random.default_rng(seed)
    spacing = _palette_spacing(palette)
    pert = img + (rng.random(img.shape[:2], dtype=np.float32) - 0.5)[..., None] * spacing * strength
    idx = icm.lookup(np.clip(pert, 0, 1))
    return palette[idx], idx


def apply_dither(img, palette, kind='NONE', strength=1.0, serpentine=True,
                 seed=0, icm=None, return_index=False):
    """Dispatch to the right dither and map to `palette`."""
    from .palette import get_inverse_colormap
    if icm is None:
        icm = get_inverse_colormap(palette)
    if kind in KERNELS:
        out, idx = error_diffusion(img, palette, kind, strength, serpentine, icm)
    elif kind == 'NOISE':
        out, idx = noise_dither(img, palette, strength, seed, icm)
    elif kind in ORDERED:
        out, idx = ordered_palette(img, palette, kind, strength, icm)
    else:
        idx = icm.lookup(np.clip(img, 0, 1))
        out = palette[idx]
    return (out, idx) if return_index else out


# ---------------------------------------------------------------- R251
# The transparency pack (1.90.0): Screen Door's column mesh, the numbered
# stipple patterns, and the framebuffer formats' pack / read-back integer
# arithmetic (PS2 PSMCT16, GameCube RGBA6, 3dfx Voodoo RGB565). Every
# integer step is np.int32; `>>` on a negative int32 is NumPy's arithmetic
# (floor) shift, which is what every one of these chips did.

#: the Mega Drive / SNES pseudo-hires 1x2 column mesh: threshold 0.25 at an
#: even column, 0.75 at an odd one -- under the keep rule `alpha > thr` an
#: opacity in (0.25, 0.75] draws every other pixel column, above 0.75 all,
#: below 0.25 none. Columns are the frame's own pixel columns.
COLUMNS = np.array([[0.25, 0.75]], np.float32)
ORDERED['COLUMNS'] = COLUMNS

#: Screen Door's numbered patterns (properties.STIPPLE_PATTERN carries the
#: same numbers, explicitly, so a saved .blend keeps its pattern): the five
#: ordered kinds keep the positions the DITHER list gave them, COLUMNS and
#: N64_NOISE sit past DITHER's fourteen. N64_NOISE (15) is the RDP's random
#: alpha compare, a per-frame hash and not a threshold map (C015).
STIPPLE_NUMBERS = {'BAYER2': 1, 'BAYER4': 2, 'BAYER8': 3, 'BAYER16': 4,
                   'HALFTONE': 5, 'COLUMNS': 14, 'N64_NOISE': 15}


def noise_threshold_at(x, y, frame, seed):
    """R251 C015 (N64 RDP `dither_alpha_en`): the fresh 8-bit random the
    alpha compare tests against, at pixel (x, y) of frame `frame` under
    `seed` -- `film._hash_u32_raw`'s low byte (the hash the GRAIN stage
    already twins), keyed by ONE integer `frame * 0x10001 + seed` so a
    frame step and a seed step never collide within 65536 frames.
    Values 0..255 as float32 (exactly representable); the RDP's LFSR is
    replaced by Halcyon's own hash, disclosed."""
    from .film import _hash_u32_raw
    key = (int(frame) * 0x10001 + int(seed)) & 0x7fffffff
    r8 = _hash_u32_raw(np.asarray(x), np.asarray(y), key) & np.uint32(0xff)
    return r8.astype(np.float32)


def noise_threshold_map(h, w, frame, seed):
    """The full-frame (h, w) float32 map of `noise_threshold_at` -- the
    CPU indexes it `tm[py, px]` (row 0 = bottom) and the GPU road uploads
    THIS array (four columns per RGBA32F texel), so both devices compare
    the same 8-bit random at every pixel."""
    ys, xs = np.mgrid[0:int(h), 0:int(w)]
    return noise_threshold_at(xs, ys, frame, seed)


def pattern(kind, x, y, frame=0, seed=0):
    """The Screen Door threshold at pixel (x, y) for a numbered or named
    stipple pattern: `kind` is a STIPPLE_NUMBERS name or its number. An
    ordered kind indexes its threshold map (tiled from row 0, exactly as
    `threshold_map` tiles it); N64_NOISE returns the RDP's per-(pixel,
    frame, seed) 8-bit random (0..255, compared against round(a*255));
    a kind with no map returns None so the caller falls back by name
    (Bayer 4x4, printed)."""
    if not isinstance(kind, str):
        names = {v: k for k, v in STIPPLE_NUMBERS.items()}
        kind = names.get(int(kind), '')
    if str(kind) == 'N64_NOISE':
        return noise_threshold_at(x, y, frame, seed)
    m = ORDERED.get(str(kind))
    if m is None:
        return None
    xx = np.asarray(x, np.int64) % m.shape[1]
    yy = np.asarray(y, np.int64) % m.shape[0]
    return m[yy, xx]


# The framebuffer formats' matrices, as the hardware (or its verified
# emulator) holds them:
#: the Voodoo's 4x4 (MAME voodoo_render.h s_dither_matrix_4x4)
FB_M4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9],
                  [15, 7, 13, 5]], np.int32)
#: the Voodoo's '2x2' (MAME s_dither_matrix_2x2 -- the real {8,10;11,9})
FB_M2 = np.array([[8, 10], [11, 9]], np.int32)
#: the GS's DIMX as gsKit's default {4,2,5,3, 0,6,1,7, 5,3,4,2, 1,7,0,6}
#: read as the register's signed 3-bit two's-complement fields
FB_DIMX = np.array([[-4, 2, -3, 3], [0, -2, 1, -1], [-3, 3, -4, 2],
                    [1, -1, 0, -2]], np.int32)
#: Flipper's 2x2 (Dolphin's hardware test, PixelShaderGen.cpp)
FB_D2GC = np.array([[0, 2], [3, 1]], np.int32)

FB_FORMATS = ('NONE', 'PS2_CT16', 'GC_RGBA6', 'VOODOO_565_4X4',
              'VOODOO_565_2X2')


def fb_dither_value(fmt, xx, yy, dither):
    """The matrix entry each pixel adds at a write, int32, same shape as
    `xx`/`yy` (frame coordinates: `yy` is the frame row, so a banded
    worker indexes exactly as the whole frame does)."""
    xx = np.asarray(xx, np.int32)
    yy = np.asarray(yy, np.int32)
    fmt = str(fmt)
    if fmt == 'GC_RGBA6':
        return FB_D2GC[yy & 1, xx & 1]            # RGBA6 always dithers
    if not dither:
        return np.zeros(np.broadcast(xx, yy).shape, np.int32)
    if fmt == 'PS2_CT16':
        return FB_DIMX[yy & 3, xx & 3]
    if fmt == 'VOODOO_565_4X4':
        return FB_M4[yy & 3, xx & 3]
    if fmt == 'VOODOO_565_2X2':
        return FB_M2[yy & 1, xx & 1]
    return np.zeros(np.broadcast(xx, yy).shape, np.int32)


def fb_pack8(c8, xx, yy, fmt, dither):
    """One write into the buffer: 8-bit int32 rgb (..., 3) -> the STORED
    value as its 8-bit read-back expansion (..., 3) int32. The expansion
    is injective, so the 5/6-bit integer the buffer holds is recovered
    exactly by the inverse shift. PS2: (c8 + DIMX) >> 3, zero-fill
    expand (the GS reads its 5 bits back as-is); GC: Dolphin's prescale
    `c8 - (c8 >> 6)` plus the 2x2, >> 2, bit-replicated; Voodoo: MAME's
    hardware-verified 565 pack, bit-replicated."""
    c8 = np.asarray(c8, np.int32)
    fmt = str(fmt)
    d = fb_dither_value(fmt, xx, yy, dither)[..., None]
    if fmt == 'PS2_CT16':
        c5 = np.clip((c8 + d) >> 3, 0, 31)
        return c5 << 3
    if fmt == 'GC_RGBA6':
        t = c8 - (c8 >> 6) + d
        c6 = np.clip(t >> 2, 0, 63)
        return (c6 << 2) | (c6 >> 4)
    if fmt in ('VOODOO_565_4X4', 'VOODOO_565_2X2'):
        r8 = c8[..., 0]
        g8 = c8[..., 1]
        b8 = c8[..., 2]
        dd = d[..., 0]
        r5 = np.clip(((r8 << 1) - (r8 >> 4) + (r8 >> 7) + dd) >> 4, 0, 31)
        g6 = np.clip(((g8 << 2) - (g8 >> 4) + (g8 >> 6) + dd) >> 4, 0, 63)
        b5 = np.clip(((b8 << 1) - (b8 >> 4) + (b8 >> 7) + dd) >> 4, 0, 31)
        out = np.empty_like(c8)
        out[..., 0] = (r5 << 3) | (r5 >> 2)
        out[..., 1] = (g6 << 2) | (g6 >> 4)
        out[..., 2] = (b5 << 3) | (b5 >> 2)
        return out
    return c8


def fb_read8(s8, xx, yy, fmt, dither, subtract):
    """The destination a blend SEES, from the stored expansion (..., 3)
    int32. The GS and the EFB read back exactly what they hold; the Voodoo
    with fbzMode bit 19 ('alpha dither subtraction') adds the matrix value
    back before blending -- and adds nothing when nothing was added at the
    write (Buffer Dither off)."""
    s8 = np.asarray(s8, np.int32)
    fmt = str(fmt)
    if fmt in ('VOODOO_565_4X4', 'VOODOO_565_2X2') and subtract and dither:
        d = fb_dither_value(fmt, xx, yy, dither)
        out = np.empty_like(s8)
        out[..., 0] = np.minimum(255, s8[..., 0] + ((15 - d) >> 1))
        out[..., 1] = np.minimum(255, s8[..., 1] + ((15 - d) >> 2))
        out[..., 2] = np.minimum(255, s8[..., 2] + ((15 - d) >> 1))
        return out
    return s8


def fb_over8(f8, a8, b8):
    """The 8-bit integer over every 16-bit-era blend unit ran (C070's
    rule; the PS2 and the GC blend in 8 bits too): (F*a + B*(255-a) + 128)
    >> 8, int32 per channel. `a8` is (...,) and broadcasts over rgb."""
    f8 = np.asarray(f8, np.int32)
    b8 = np.asarray(b8, np.int32)
    a8 = np.asarray(a8, np.int32)[..., None]
    return (f8 * a8 + b8 * (255 - a8) + 128) >> 8
