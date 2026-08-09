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


def threshold_map(kind, h, w):
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
        c = buf[y, x]
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
            r = flat[o]
            g = flat[o + 1]
            b = flat[o + 2]

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
