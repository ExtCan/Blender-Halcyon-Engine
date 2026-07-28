"""Colour quantisation: adaptive palettes, period-accurate fixed palettes, and
an inverse colormap for fast nearest-colour lookup (exactly the trick the 90s
used, and the reason we can afford per-pixel error diffusion at all).

All palettes are float32 (N,3) in 0..1.
"""

import numpy as np

# --------------------------------------------------------------- fixed sets

EGA16 = np.array([
    (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
    (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
    (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
    (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
], dtype=np.float32) / 255.0

CGA4 = np.array([
    (0, 0, 0), (85, 255, 255), (255, 85, 255), (255, 255, 255),
], dtype=np.float32) / 255.0

CGA4_RED = np.array([
    (0, 0, 0), (85, 255, 85), (255, 85, 85), (255, 255, 85),
], dtype=np.float32) / 255.0

WIN20 = np.array([
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0),
    (0, 0, 128), (128, 0, 128), (0, 128, 128), (192, 192, 192),
    (192, 220, 192), (166, 202, 240), (255, 251, 240), (160, 160, 164),
    (128, 128, 128), (255, 0, 0), (0, 255, 0), (255, 255, 0),
    (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255),
], dtype=np.float32) / 255.0


def web216():
    lv = np.array([0, 51, 102, 153, 204, 255], np.float32) / 255.0
    r, g, b = np.meshgrid(lv, lv, lv, indexing='ij')
    return np.stack([r.ravel(), g.ravel(), b.ravel()], axis=1).astype(np.float32)


def cube666():
    return web216()


def cube332():
    r = np.linspace(0, 1, 8, dtype=np.float32)
    g = np.linspace(0, 1, 8, dtype=np.float32)
    b = np.linspace(0, 1, 4, dtype=np.float32)
    R, G, B = np.meshgrid(r, g, b, indexing='ij')
    return np.stack([R.ravel(), G.ravel(), B.ravel()], axis=1).astype(np.float32)


def grayscale(n=256):
    v = np.linspace(0.0, 1.0, n, dtype=np.float32)
    return np.stack([v, v, v], axis=1)


def vga256():
    """The IBM VGA BIOS default 256-colour palette (6-bit DAC).

    0-15   EGA colours
    16-31  16-step grey ramp
    32-247 9 blocks of 24 hues (3 value levels x 3 saturation levels)
    248-255 black
    """
    from .mathx import hsv_to_rgb
    pal = np.zeros((256, 3), np.float32)
    pal[0:16] = EGA16
    grey = np.array([0, 5, 8, 11, 14, 17, 20, 24, 28, 32, 36, 40, 45, 50, 56, 63],
                    np.float32) / 63.0
    pal[16:32] = grey[:, None].repeat(3, axis=1)
    vals = (1.0, 0.45, 0.26)
    sats = (1.0, 0.56, 0.25)
    i = 32
    for v in vals:
        for s in sats:
            for h in range(24):
                r, g, b = hsv_to_rgb(np.float32(h / 24.0), np.float32(s), np.float32(v))
                pal[i] = (float(r), float(g), float(b))
                i += 1
    pal[248:256] = 0.0
    # snap to the 6-bit DAC the hardware actually had
    return (np.round(pal * 63.0) / 63.0).astype(np.float32)


def mac256():
    """Apple's classic 8-bit system palette: 6x6x6 cube + 4 ten-step ramps."""
    lv = np.linspace(1.0, 0.0, 6, dtype=np.float32)
    cols = []
    for r in lv:
        for g in lv:
            for b in lv:
                cols.append((r, g, b))
    cols = [c for c in cols if not (c[0] == c[1] == c[2])]      # 210
    ramp = np.array([238, 221, 187, 170, 136, 119, 85, 68, 34, 17], np.float32) / 255.0
    for v in ramp:
        cols.append((v, 0.0, 0.0))
    for v in ramp:
        cols.append((0.0, v, 0.0))
    for v in ramp:
        cols.append((0.0, 0.0, v))
    for v in ramp:
        cols.append((v, v, v))
    cols.append((1.0, 1.0, 1.0))
    cols.append((0.0, 0.0, 0.0))
    pal = np.array(cols[:256], np.float32)
    if len(pal) < 256:
        pal = np.vstack([pal, np.zeros((256 - len(pal), 3), np.float32)])
    return pal


def amiga_ocs(n=32):
    """Amiga OCS/ECS 12-bit colour register set (adaptive, but 4 bits/channel)."""
    return None  # generated adaptively then snapped; see snap_bits


FIXED_PALETTES = {
    'EGA16': lambda size: EGA16,
    'CGA4': lambda size: CGA4,
    'WIN20': lambda size: WIN20,
    'WEB216': lambda size: web216(),
    'FIXED_666': lambda size: cube666(),
    'FIXED_332': lambda size: cube332(),
    'VGA256': lambda size: vga256(),
    'MAC256': lambda size: mac256(),
    'GRAY': lambda size: grayscale(max(2, int(size))),
}


# ---------------------------------------------------------------- adaptive


def median_cut(pixels, n_colors, bits=5):
    """Heckbert's median cut. pixels: (N,3) float 0..1."""
    n_colors = max(2, int(n_colors))
    q = np.clip((pixels * ((1 << bits) - 1)).astype(np.int32), 0, (1 << bits) - 1)
    key = (q[:, 0] << (2 * bits)) | (q[:, 1] << bits) | q[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    cols = np.stack([(uniq >> (2 * bits)) & ((1 << bits) - 1),
                     (uniq >> bits) & ((1 << bits) - 1),
                     uniq & ((1 << bits) - 1)], axis=1).astype(np.float32)
    cols = cols / float((1 << bits) - 1)
    if len(cols) <= n_colors:
        pal = np.zeros((n_colors, 3), np.float32)
        pal[:len(cols)] = cols
        return pal

    boxes = [(np.arange(len(cols)), counts.astype(np.int64))]
    while len(boxes) < n_colors:
        best, best_i = -1.0, -1
        for i, (idx, cnt) in enumerate(boxes):
            if len(idx) < 2:
                continue
            c = cols[idx]
            ext = c.max(axis=0) - c.min(axis=0)
            vol = float(ext.max()) * float(np.log1p(cnt.sum()))
            if vol > best:
                best, best_i = vol, i
        if best_i < 0:
            break
        idx, cnt = boxes.pop(best_i)
        c = cols[idx]
        ext = c.max(axis=0) - c.min(axis=0)
        axis = int(np.argmax(ext))
        order = np.argsort(c[:, axis], kind='stable')
        idx, cnt = idx[order], cnt[order]
        cum = np.cumsum(cnt)
        half = cum[-1] * 0.5
        split = int(np.searchsorted(cum, half)) + 1
        split = max(1, min(len(idx) - 1, split))
        boxes.append((idx[:split], cnt[:split]))
        boxes.append((idx[split:], cnt[split:]))

    pal = np.zeros((n_colors, 3), np.float32)
    for i, (idx, cnt) in enumerate(boxes[:n_colors]):
        w = cnt.astype(np.float32)
        pal[i] = (cols[idx] * w[:, None]).sum(axis=0) / max(w.sum(), 1.0)
    return pal


def popularity(pixels, n_colors, bits=5):
    q = np.clip((pixels * ((1 << bits) - 1)).astype(np.int32), 0, (1 << bits) - 1)
    key = (q[:, 0] << (2 * bits)) | (q[:, 1] << bits) | q[:, 2]
    uniq, counts = np.unique(key, return_counts=True)
    order = np.argsort(-counts)[:n_colors]
    u = uniq[order]
    cols = np.stack([(u >> (2 * bits)) & ((1 << bits) - 1),
                     (u >> bits) & ((1 << bits) - 1),
                     u & ((1 << bits) - 1)], axis=1).astype(np.float32)
    pal = cols / float((1 << bits) - 1)
    if len(pal) < n_colors:
        pal = np.vstack([pal, np.zeros((n_colors - len(pal), 3), np.float32)])
    return pal


def octree(pixels, n_colors, max_level=6):
    """Gervautz-Purgathofer octree quantisation."""
    n_colors = max(2, int(n_colors))
    q = np.clip((pixels * 255.0).astype(np.int32), 0, 255)
    nodes = {}
    order = []

    def path(r, g, b):
        p = []
        for lvl in range(max_level):
            sh = 7 - lvl
            p.append((((r >> sh) & 1) << 2) | (((g >> sh) & 1) << 1) | ((b >> sh) & 1))
        return tuple(p)

    uq, cnt = np.unique(q[:, 0].astype(np.int64) * 65536 +
                        q[:, 1].astype(np.int64) * 256 + q[:, 2], return_counts=True)
    rs = (uq >> 16).astype(np.int32)
    gs = ((uq >> 8) & 255).astype(np.int32)
    bs = (uq & 255).astype(np.int32)
    for i in range(len(uq)):
        key = path(int(rs[i]), int(gs[i]), int(bs[i]))
        for lvl in range(1, max_level + 1):
            k = key[:lvl]
            e = nodes.get(k)
            if e is None:
                e = [0.0, 0.0, 0.0, 0, lvl]
                nodes[k] = e
                order.append(k)
            e[0] += float(rs[i]) * cnt[i]
            e[1] += float(gs[i]) * cnt[i]
            e[2] += float(bs[i]) * cnt[i]
            e[3] += int(cnt[i])

    leaves = [k for k in nodes if len(k) == max_level]
    while len(leaves) > n_colors:
        deepest = max(len(k) for k in leaves)
        parents = sorted({k[:-1] for k in leaves if len(k) == deepest},
                         key=lambda p: nodes[p][3])
        for p in parents:
            kids = [k for k in leaves if len(k) == deepest and k[:-1] == p]
            if not kids:
                continue
            leaves = [k for k in leaves if k not in kids]
            leaves.append(p)
            if len(leaves) <= n_colors:
                break
    pal = np.zeros((n_colors, 3), np.float32)
    for i, k in enumerate(leaves[:n_colors]):
        e = nodes[k]
        c = max(e[3], 1)
        pal[i] = (e[0] / c / 255.0, e[1] / c / 255.0, e[2] / c / 255.0)
    return pal


def kmeans(pixels, n_colors, iters=12, seed=0):
    rng = np.random.default_rng(seed)
    n_colors = max(2, int(n_colors))
    sub = pixels
    if len(sub) > 60000:
        sub = sub[rng.choice(len(sub), 60000, replace=False)]
    pal = median_cut(sub, n_colors)
    for _ in range(iters):
        idx = nearest_brute(sub, pal)
        for k in range(n_colors):
            m = idx == k
            if m.any():
                pal[k] = sub[m].mean(axis=0)
    return pal


def build_palette(pixels, n_colors, method='MEDIAN_CUT', seed=0):
    flat = pixels.reshape(-1, 3).astype(np.float32)
    if len(flat) > 400000:
        step = len(flat) // 400000 + 1
        flat = flat[::step]
    if method == 'OCTREE':
        return octree(flat, n_colors)
    if method == 'POPULARITY':
        return popularity(flat, n_colors)
    if method == 'KMEANS':
        return kmeans(flat, n_colors, seed=seed)
    return median_cut(flat, n_colors)


def get_palette(mode, size=256, pixels=None, method='MEDIAN_CUT', seed=0):
    if mode in FIXED_PALETTES:
        return FIXED_PALETTES[mode](size)
    if pixels is None:
        return grayscale(size)
    return build_palette(pixels, size, method, seed)


# ------------------------------------------------------------ nearest lookup


def nearest_brute(colors, palette, chunk=200000):
    """Exact nearest palette index. colors (N,3) -> (N,) int32."""
    out = np.empty(len(colors), np.int32)
    p = palette.astype(np.float32)
    for s in range(0, len(colors), chunk):
        c = colors[s:s + chunk]
        d = ((c[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        out[s:s + chunk] = np.argmin(d, axis=1)
    return out


_ICM_CACHE = {}
_ICM_ORDER = []
_ICM_LIMIT = 8


def get_inverse_colormap(palette, bits=6):
    """Cached nearest-colour cube.

    Building one costs a good fraction of a second, and it depends only on the
    palette -- so an animation that keeps its palette builds it once instead of
    once per frame.
    """
    pal = np.ascontiguousarray(np.asarray(palette, np.float32))
    key = (pal.tobytes(), int(bits))
    hit = _ICM_CACHE.get(key)
    if hit is not None:
        return hit
    icm = InverseColormap(pal, bits)
    _ICM_CACHE[key] = icm
    _ICM_ORDER.append(key)
    while len(_ICM_ORDER) > _ICM_LIMIT:
        _ICM_CACHE.pop(_ICM_ORDER.pop(0), None)
    return icm


_PALETTE_CACHE = {}


def cached_adaptive(key, builder):
    """Reuse a previously built adaptive palette.

    Recomputing it per frame is not only slow, it makes the colours crawl
    between frames of an animation because median cut lands somewhere slightly
    different each time.
    """
    hit = _PALETTE_CACHE.get(key)
    if hit is None:
        hit = builder()
        if len(_PALETTE_CACHE) > 16:
            _PALETTE_CACHE.clear()
        _PALETTE_CACHE[key] = hit
    return hit


def clear_caches():
    _ICM_CACHE.clear()
    _ICM_ORDER.clear()
    _PALETTE_CACHE.clear()


class InverseColormap:
    """Pre-computed nearest-colour cube -- O(1) lookups for error diffusion."""

    # a 6-bit cube against a 256-entry palette is 67 million distances; done in
    # one array that is an 800 MB temporary, and the machine spends its time in
    # the memory system rather than the ALU. Chunked to stay in cache.
    CHUNK = 32768

    def __init__(self, palette, bits=6):
        self.palette = np.asarray(palette, np.float32)
        self.bits = int(bits)
        n = 1 << self.bits
        self.n = n
        pal = self.palette
        axis = (np.arange(n, dtype=np.float32) + 0.5) / n
        total = n * n * n
        lut = np.empty(total, np.int32)
        # |c - p|^2 = |c|^2 - 2 c.p + |p|^2, and |c|^2 is constant across the
        # palette for a given cell, so it cannot change which entry is nearest.
        # Dropping it leaves a single matrix product, which BLAS does far
        # faster than a broadcast subtract over a 67-million-element temporary.
        pal_sq = (pal * pal).sum(axis=1)
        idx = np.arange(self.CHUNK, dtype=np.int64)
        for start in range(0, total, self.CHUNK):
            count = min(self.CHUNK, total - start)
            i = idx[:count] + start
            c = np.empty((count, 3), np.float32)
            c[:, 0] = axis[(i >> (2 * self.bits)) & (n - 1)]
            c[:, 1] = axis[(i >> self.bits) & (n - 1)]
            c[:, 2] = axis[i & (n - 1)]
            d = pal_sq[None, :] - 2.0 * (c @ pal.T)
            lut[start:start + count] = np.argmin(d, axis=1)
        self.lut = lut

    def lookup(self, colors):
        b = self.bits
        n = self.n
        q = np.clip((np.asarray(colors, np.float32) * n).astype(np.int32), 0, n - 1)
        return self.lut[(q[..., 0] << (2 * b)) | (q[..., 1] << b) | q[..., 2]]

    def lookup1(self, r, g, b_):
        n = self.n
        bb = self.bits
        ri = min(n - 1, max(0, int(r * n)))
        gi = min(n - 1, max(0, int(g * n)))
        bi = min(n - 1, max(0, int(b_ * n)))
        return int(self.lut[(ri << (2 * bb)) | (gi << bb) | bi])


def snap_bits(img, rbits, gbits, bbits):
    """Truncate to a hardware colour depth (e.g. 5-6-5, 5-5-5, 4-4-4)."""
    out = np.empty_like(img)
    for ch, bits in enumerate((rbits, gbits, bbits)):
        levels = float((1 << bits) - 1)
        out[..., ch] = np.round(np.clip(img[..., ch], 0.0, 1.0) * levels) / levels
    return out


def quantize_image(rgb, n_colors, method='MEDIAN_CUT', dither='NONE',
                   strength=1.0, palette=None, serpentine=True, seed=0):
    """Map an image to an N-colour palette. Returns (rgb_out, palette)."""
    from .dither import apply_dither
    if palette is None:
        palette = build_palette(rgb, n_colors, method, seed)
    out = apply_dither(rgb, palette, kind=dither, strength=strength,
                       serpentine=serpentine, seed=seed)
    return out, palette


# ------------------------------------------------------------------- Amiga


def ham_encode(rgb, bits=8):
    """Amiga HAM6 / HAM8 encoding, done honestly: scanline-sequential, one
    channel modifiable per pixel, plus a small base palette."""
    h, w = rgb.shape[:2]
    depth = 4 if bits == 6 else 6
    levels = float((1 << depth) - 1)
    base_n = 16 if bits == 6 else 64
    base = build_palette(rgb, base_n, 'MEDIAN_CUT')
    base = np.round(base * levels) / levels
    q = np.round(np.clip(rgb, 0, 1) * levels) / levels
    out = np.empty_like(q)
    base_scaled = base
    for y in range(h):
        cur = np.array([0.0, 0.0, 0.0], np.float32)
        row = q[y]
        for x in range(w):
            tgt = row[x]
            d_base = ((base_scaled - tgt) ** 2).sum(axis=1)
            i_base = int(np.argmin(d_base))
            best = float(d_base[i_base])
            cand = base_scaled[i_base]
            for ch in range(3):
                c = cur.copy()
                c[ch] = tgt[ch]
                e = float(((c - tgt) ** 2).sum())
                if e < best:
                    best = e
                    cand = c
            cur = np.asarray(cand, np.float32).copy()
            out[y, x] = cur
    return out, base
