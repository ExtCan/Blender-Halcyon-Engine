"""Post chain: everything between the shaded framebuffer and the file on disk.

Order matters, and this is the order the hardware imposed:

    optical (glow / star / flare, in linear light)
      -> display transform (exposure, gamma, contrast, saturation)
      -> colour depth + dither + palette      [the framebuffer]
      -> composite NTSC encode/decode         [the cable]
      -> interlace                            [the signal]
      -> CRT mask, scanlines, curvature       [the glass]
      -> JPEG artefacts                       [the file]
      -> nearest-neighbour upscale / pixel aspect

Doing glow before quantisation and scanlines after it is not a stylistic
choice: a 1996 machine glowed in the framebuffer and scanned on the tube.
"""

import numpy as np

from . import dither as DI
from . import mathx as M
from . import palette as PA

DEPTH_BITS = {
    '32': (8, 8, 8), '24': (8, 8, 8), '16': (5, 6, 5), '15': (5, 5, 5),
    '12': (4, 4, 4), '9': (3, 3, 3), '8': (3, 3, 2),
}


# ------------------------------------------------------------------ helpers


def _blur_h(img, radius):
    """Separable box blur, repeated 3x -- a fast, accurate Gaussian."""
    r = int(max(radius, 0))
    if r <= 0:
        return img
    out = img
    k = 2 * r + 1
    for _ in range(3):
        pad = np.pad(out, ((0, 0), (r, r), (0, 0)), mode='edge')
        cs = np.cumsum(pad, axis=1, dtype=np.float32)
        cs = np.concatenate([np.zeros((cs.shape[0], 1, cs.shape[2]), np.float32), cs], 1)
        out = (cs[:, k:] - cs[:, :-k]) / k
    return out


def blur(img, radius):
    if radius <= 0:
        return img
    out = _blur_h(img, radius)
    out = np.transpose(_blur_h(np.transpose(out, (1, 0, 2)), radius), (1, 0, 2))
    return out.astype(np.float32)


def _resample(img, xs, ys):
    """Bilinear sample of an (H,W,C) image at float coords."""
    h, w = img.shape[:2]
    x = np.clip(xs, 0, w - 1)
    y = np.clip(ys, 0, h - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    a = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    b = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


# ------------------------------------------------------------------ optical


def glow(rgb, st):
    """Framebuffer bloom: threshold, blur, add back."""
    if not st.glow or st.glow_intensity <= 0:
        return rgb
    lum = M.luminance(rgb)
    over = np.maximum(lum - st.glow_threshold, 0.0)[:, :, None]
    bright = rgb * (over / np.maximum(lum[:, :, None], 1e-5))
    r = max(int(st.glow_radius), 1)
    if st.glow_quality == 'BOX':
        b = _blur_h(np.transpose(_blur_h(np.transpose(bright, (1, 0, 2)), r),
                                 (1, 0, 2)), r)
    else:
        b = blur(bright, r)
    return (rgb + b * st.glow_intensity).astype(np.float32)


def star_filter(rgb, st):
    """Cross-screen filter: streaks radiating from every highlight."""
    if not st.star_filter or st.star_intensity <= 0:
        return rgb
    lum = M.luminance(rgb)
    mask = (lum > st.glow_threshold)[:, :, None]
    bright = rgb * mask
    if not mask.any():
        return rgb
    h, w = rgb.shape[:2]
    acc = np.zeros_like(rgb)
    n = max(int(st.star_points), 2)
    length = max(int(st.star_length), 1)
    yy, xx = np.mgrid[0:h, 0:w]
    for i in range(n):
        ang = st.star_rotation + i * (np.pi * 2.0 / n)
        dx, dy = np.cos(ang), np.sin(ang)
        streak = np.zeros_like(rgb)
        for s in range(1, length):
            fall = (1.0 - s / length) ** 2
            sx = np.clip(xx - dx * s, 0, w - 1).astype(np.int32)
            sy = np.clip(yy - dy * s, 0, h - 1).astype(np.int32)
            streak += bright[sy, sx] * fall
        acc += streak / length
    return (rgb + acc * st.star_intensity).astype(np.float32)


def lens_flare(rgb, st):
    """Ghost images of the bright spots mirrored through the frame centre."""
    if not st.lens_flare or st.flare_intensity <= 0:
        return rgb
    h, w = rgb.shape[:2]
    lum = M.luminance(rgb)
    bright = rgb * (lum > max(st.glow_threshold, 0.9))[:, :, None]
    if not np.any(bright):
        return rgb
    small = blur(bright, max(w // 64, 2))
    acc = np.zeros_like(rgb)
    tints = np.array([[1.0, 0.7, 0.4], [0.4, 0.8, 1.0], [0.8, 1.0, 0.6],
                      [1.0, 0.5, 0.9], [0.6, 0.6, 1.0]], np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * 0.5
    for g in range(max(int(st.flare_ghosts), 1)):
        scale = -(g + 1) * 0.45 + 0.2
        sx = cx + (xx - cx) / max(abs(scale), 0.05) * np.sign(scale)
        sy = cy + (yy - cy) / max(abs(scale), 0.05) * np.sign(scale)
        acc += _resample(small, sx, sy) * tints[g % len(tints)][None, None, :]
    if st.flare_streak > 0:
        wide = blur(bright, max(w // 8, 4))
        acc += wide * st.flare_streak
    return (rgb + acc * st.flare_intensity).astype(np.float32)


# -------------------------------------------------------- display transform


def display_transform(rgb, st):
    out = rgb * float(st.exposure)
    if st.color_management == 'FILMIC':
        out = out / (out + 0.6)
    elif st.color_management == 'REINHARD':
        out = out / (1.0 + out)
    elif st.color_management == 'SRGB':
        out = M.linear_to_srgb(np.clip(out, 0.0, 1.0))
    out = out + float(st.brightness)
    if abs(st.contrast) > 1e-6:
        c = float(st.contrast)
        out = np.clip((out - 0.5) * (1.0 + c) + 0.5, 0.0, None)
    if abs(st.saturation - 1.0) > 1e-6:
        lum = M.luminance(out)[:, :, None]
        out = lum + (out - lum) * float(st.saturation)
    g = float(st.gamma)
    if abs(g - 1.0) > 1e-6:
        out = np.power(np.maximum(out, 0.0), 1.0 / max(g, 1e-3))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------ colour depth


def _palette_for(st, size, rgb, seed, mode=None):
    """The palette for this frame, reused across frames when locked.

    An adaptive palette recomputed every frame does not merely cost time: median
    cut lands somewhere slightly different on each frame, so the colours crawl.
    Locking it is both faster and steadier, which is why it defaults to on.
    """
    mode = mode or st.palette_mode
    if mode != 'ADAPTIVE' or not getattr(st, 'palette_lock', True):
        return PA.get_palette(mode, size, rgb.reshape(-1, 3),
                              st.palette_method, seed)
    key = ('ADAPTIVE', int(size), str(st.palette_method), int(seed))
    return PA.cached_adaptive(
        key, lambda: PA.get_palette(mode, size, rgb.reshape(-1, 3),
                                    st.palette_method, seed))


def reduce_depth(rgb, st, seed=0):
    """Framebuffer quantisation: bit depth, palette and dither together."""
    depth = str(st.color_depth)
    kind = st.dither
    strength = float(st.dither_strength)

    if depth in ('HAM6', 'HAM8'):
        bits = 8 if depth == 'HAM8' else 6
        base = _palette_for(st, 16 if bits == 6 else 64, rgb, seed,
                            mode=('ADAPTIVE' if st.palette_mode == 'ADAPTIVE'
                                  else st.palette_mode))
        if kind != 'NONE':
            rgb = DI.apply_dither(rgb, base, kind, strength, st.dither_serpentine,
                                  seed=seed)
        img, _base = PA.ham_encode(rgb, bits)
        return img.astype(np.float32)

    if depth == '1':
        lum = M.luminance(rgb)[:, :, None]
        pal = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], np.float32)
        img = np.repeat(lum, 3, axis=2)
        if kind == 'NONE':
            return (img > 0.5).astype(np.float32)
        return DI.apply_dither(img, pal, kind, strength, st.dither_serpentine,
                               seed=seed)

    if depth in ('8', '4') or st.palette_mode != 'ADAPTIVE' or \
            st.color_depth in ('8', '4'):
        if depth in DEPTH_BITS and st.palette_mode == 'ADAPTIVE' and depth != '8':
            pass
        else:
            size = int(st.palette_size)
            if depth == '4':
                size = min(size, 16)
            elif depth == '8':
                size = min(size, 256)
            pal = _palette_for(st, size, rgb, seed)
            if kind == 'NONE':
                icm = PA.InverseColormap(pal)
                idx = icm.lookup(rgb.reshape(-1, 3))
                return pal[idx].reshape(rgb.shape).astype(np.float32)
            return DI.apply_dither(rgb, pal, kind, strength, st.dither_serpentine,
                                   seed=seed)

    bits = DEPTH_BITS.get(depth, (8, 8, 8))
    if kind == 'NONE':
        return PA.snap_bits(rgb, *bits)
    if kind.startswith('BAYER') or kind in ('HALFTONE', 'NOISE'):
        return DI.ordered_bits(rgb, bits, kind, strength)
    pal = PA.cube666() if bits == (8, 8, 8) else None
    levels = [np.linspace(0, 1, 1 << b, dtype=np.float32) for b in bits]
    grid = np.stack(np.meshgrid(*levels, indexing='ij'), -1).reshape(-1, 3)
    if grid.shape[0] > 4096:
        return PA.snap_bits(rgb, *bits)
    return DI.apply_dither(rgb, grid.astype(np.float32), kind, strength,
                           st.dither_serpentine, seed=seed)


# --------------------------------------------------------- composite / CRT

RGB2YIQ = np.array([[0.299, 0.587, 0.114],
                    [0.5959, -0.2746, -0.3213],
                    [0.2115, -0.5227, 0.3112]], np.float32)
YIQ2RGB = np.linalg.inv(RGB2YIQ).astype(np.float32)


def composite_ntsc(rgb, st, frame=0):
    """Bandwidth-limit the chroma the way a composite cable did."""
    if not st.composite:
        return rgb
    h, w = rgb.shape[:2]
    yiq = rgb @ RGB2YIQ.T
    bleed = float(st.composite_bleed)
    ri = max(int(round(w / 320.0 * 6.0 * bleed)), 1)
    rq = max(int(round(w / 320.0 * 12.0 * bleed)), 1)
    yiq[:, :, 1] = _blur_h(yiq[:, :, 1:2], ri)[:, :, 0]
    yiq[:, :, 2] = _blur_h(yiq[:, :, 2:3], rq)[:, :, 0]
    if st.composite_ringing > 0:
        y = yiq[:, :, 0]
        soft = _blur_h(y[:, :, None], 2)[:, :, 0]
        yiq[:, :, 0] = y + (y - soft) * float(st.composite_ringing) * 2.0
    if st.composite_dot_crawl > 0:
        xs = np.arange(w, dtype=np.float32)[None, :]
        ys = np.arange(h, dtype=np.float32)[:, None]
        phase = (xs * 0.5 + ys * 0.5 + frame * 0.5) * np.pi
        crawl = np.sin(phase) * float(st.composite_dot_crawl) * 0.06
        chroma = np.sqrt(yiq[:, :, 1] ** 2 + yiq[:, :, 2] ** 2)
        yiq[:, :, 0] += crawl * chroma
    return np.clip(yiq @ YIQ2RGB.T, 0.0, 1.0).astype(np.float32)


def interlace(rgb, st, frame=0):
    if st.interlace == 'NONE':
        return rgb
    out = rgb.copy()
    field = frame % 2 if st.interlace == 'FIELDS' else 0
    rows = np.arange(rgb.shape[0])
    off = (rows % 2) != field
    if st.interlace == 'FIELDS':
        out[off] = rgb[np.clip(rows[off] - 1, 0, rgb.shape[0] - 1)]
    else:
        out[off] *= 0.55
    return out


def crt(rgb, st):
    if not st.crt:
        return rgb
    h, w = rgb.shape[:2]
    out = rgb
    if st.crt_bloom > 0:
        out = np.clip(out + blur(out, max(int(w / 160), 1)) * st.crt_bloom, 0, 2)
    if st.crt_mask != 'NONE' and st.crt_mask_strength > 0:
        out = out * _mask(h, w, st.crt_mask, st.crt_mask_strength)
    if st.crt_scanlines > 0:
        rows = np.arange(h, dtype=np.float32)[:, None, None]
        s = 1.0 - st.crt_scanlines * (0.5 + 0.5 * np.cos(rows * np.pi))
        out = out * np.clip(s, 0.0, 1.0)
    if st.crt_curvature > 0:
        out = _barrel(out, st.crt_curvature)
    if st.crt_vignette > 0:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        nx = (xx / max(w - 1, 1)) * 2 - 1
        ny = (yy / max(h - 1, 1)) * 2 - 1
        r2 = nx * nx + ny * ny
        out = out * np.clip(1.0 - r2 * st.crt_vignette * 0.5, 0.0, 1.0)[:, :, None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _mask(h, w, kind, strength):
    m = np.ones((h, w, 3), np.float32)
    cols = np.arange(w)
    if kind == 'APERTURE':
        for c in range(3):
            m[:, :, c] = np.where(cols % 3 == c, 1.0, 1.0 - strength)[None, :]
    elif kind == 'SLOT':
        rows = np.arange(h)[:, None]
        stagger = (rows // 2 % 2) * 1
        idx = (cols[None, :] + stagger) % 3
        for c in range(3):
            m[:, :, c] = np.where(idx == c, 1.0, 1.0 - strength)
        m *= np.where(rows % 4 == 3, 1.0 - strength * 0.5, 1.0)[:, :, None]
    elif kind == 'SHADOW':
        rows = np.arange(h)[:, None]
        idx = (cols[None, :] + (rows % 2) * 2) % 3
        for c in range(3):
            m[:, :, c] = np.where(idx == c, 1.0, 1.0 - strength)
    return m


def _barrel(img, amount):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2 - 1
    ny = (yy / max(h - 1, 1)) * 2 - 1
    r2 = nx * nx + ny * ny
    k = amount * 0.25
    sx = nx * (1.0 + k * r2)
    sy = ny * (1.0 + k * r2)
    inside = (np.abs(sx) <= 1.0) & (np.abs(sy) <= 1.0)
    px = (sx * 0.5 + 0.5) * (w - 1)
    py = (sy * 0.5 + 0.5) * (h - 1)
    out = _resample(img, px, py)
    return out * inside[:, :, None]


# ------------------------------------------------------------ lens and depth


def lens_distortion(rgb, st):
    """Barrel or pincushion distortion with per-channel dispersion.

    Splitting the three channels by slightly different amounts is chromatic
    aberration: a real lens does not focus red and blue to the same place, and
    the fringing that produces is one of the clearest tells of an image that
    went through glass rather than straight to a framebuffer.
    """
    k = float(st.lens_distortion)
    ca = float(st.chromatic_aberration)
    if abs(k) < 1e-5 and abs(ca) < 1e-5:
        return rgb
    h, w = rgb.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx / max(w - 1, 1)) * 2.0 - 1.0
    ny = (yy / max(h - 1, 1)) * 2.0 - 1.0
    r2 = nx * nx + ny * ny
    out = np.empty_like(rgb)
    # red pushed out, blue pulled in, green left alone
    for ch, spread in ((0, 1.0), (1, 0.0), (2, -1.0)):
        amount = k + ca * spread * 0.02
        sx = nx * (1.0 + amount * r2)
        sy = ny * (1.0 + amount * r2)
        px = (sx * 0.5 + 0.5) * (w - 1)
        py = (sy * 0.5 + 0.5) * (h - 1)
        out[:, :, ch] = _resample(rgb[:, :, ch:ch + 1], px, py)[:, :, 0]
    if st.lens_vignette_edges:
        inside = (np.abs(nx * (1.0 + k * r2)) <= 1.0) & \
                 (np.abs(ny * (1.0 + k * r2)) <= 1.0)
        out *= inside[:, :, None]
    return out.astype(np.float32)


def light_shafts(rgb, st, sources):
    """Radial scattering from a bright source -- the god rays of the era.

    Done exactly as it was then: take what is bright, smear it outward from the
    light's position on screen in a few halving steps, and add it back. No
    volume is integrated; the illusion is entirely in the streaks.
    """
    if not sources:
        return rgb
    h, w = rgb.shape[:2]
    lum = M.luminance(rgb)
    bright = rgb * (lum > float(st.shaft_threshold))[:, :, None]
    if not np.any(bright):
        return rgb
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    total = np.zeros_like(rgb)
    for (sx, sy), strength in sources:
        if strength <= 0.0:
            continue
        cx = (sx * 0.5 + 0.5) * (w - 1)
        cy = (sy * 0.5 + 0.5) * (h - 1)
        acc = np.zeros_like(rgb)
        weight = 0.0
        decay = float(st.shaft_decay)
        amp = 1.0
        steps = max(int(st.shaft_samples), 1)
        for i in range(steps):
            t = (i + 1) / steps * float(st.shaft_length)
            px = xx + (cx - xx) * t
            py = yy + (cy - yy) * t
            acc += _resample(bright, px, py) * amp
            weight += amp
            amp *= decay
        total += acc / max(weight, 1e-6) * strength
    return (rgb + total).astype(np.float32)


def depth_of_field(rgb, depth, st):
    """Layered defocus: split by depth, blur each slab, composite back to front.

    Sampling a lens properly needs many rays per pixel. Splitting the frame into
    a few depth slabs and blurring each by its circle of confusion is what
    compositors of the era did, and it costs a handful of blurs instead.
    """
    if not st.dof or depth is None:
        return rgb
    finite = np.isfinite(depth)
    if not finite.any():
        return rgb
    focus = float(st.dof_focus)
    scale = float(st.dof_amount)
    layers = max(int(st.dof_layers), 2)
    d = np.where(finite, depth, focus)
    coc = np.clip(np.abs(d - focus) / max(focus, 1e-3) * scale, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, layers + 1)
    out = rgb.copy()
    for i in range(layers):
        lo, hi = edges[i], edges[i + 1]
        m = (coc >= lo) & (coc < hi if i < layers - 1 else coc <= hi)
        if not m.any():
            continue
        radius = int(round(hi * float(st.dof_max_radius)))
        if radius < 1:
            continue
        layer = rgb * m[:, :, None]
        cover = m.astype(np.float32)[:, :, None]
        bl = blur(layer, radius)
        bc = blur(cover, radius)
        a = np.clip(bc, 0.0, 1.0)
        out = out * (1.0 - a) + np.where(bc > 1e-5, bl / np.maximum(bc, 1e-5),
                                         0.0) * a
    return out.astype(np.float32)


# -------------------------------------------------------------- JPEG blocks

_Q_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99],
], np.float32)
_Q_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
] + [[99] * 8] * 4, np.float32)


def _dct_matrix(n=8):
    k = np.arange(n, dtype=np.float32)
    m = np.cos((2 * k[None, :] + 1) * k[:, None] * np.pi / (2 * n))
    m[0] *= np.sqrt(1.0 / n)
    m[1:] *= np.sqrt(2.0 / n)
    return m.astype(np.float32)


def jpeg_artifacts(rgb, st):
    """A real 8x8 DCT round-trip -- the blocking and ringing are the genuine
    article, not a blur pretending to be compression."""
    if not st.jpeg_artifacts:
        return rgb
    q = np.clip(int(st.jpeg_quality), 1, 100)
    scale = (5000.0 / q) if q < 50 else (200.0 - 2.0 * q)
    ql = np.clip(np.floor((_Q_LUMA * scale + 50) / 100), 1, 255)
    qc = np.clip(np.floor((_Q_CHROMA * scale + 50) / 100), 1, 255)
    out = rgb
    bs = int(st.block_size) if st.block_size in (4, 8, 16) else 8
    D = _dct_matrix(bs)
    ql = _fit_q(ql, bs)
    qc = _fit_q(qc, bs)
    for _ in range(max(int(st.jpeg_passes), 1)):
        yiq = out @ RGB2YIQ.T
        h, w = yiq.shape[:2]
        ph = (-h) % bs
        pw = (-w) % bs
        pad = np.pad(yiq, ((0, ph), (0, pw), (0, 0)), mode='edge')
        H, W = pad.shape[:2]
        blocks = pad.reshape(H // bs, bs, W // bs, bs, 3).transpose(0, 2, 4, 1, 3)
        blocks = blocks * 255.0 - 128.0
        coef = np.einsum('ij,abcjk,lk->abcil', D, blocks, D)
        qtab = np.stack([ql, qc, qc])[None, None, :, :, :]
        coef = np.round(coef / qtab) * qtab
        rec = np.einsum('ji,abcjk,kl->abcil', D, coef, D)
        rec = (rec + 128.0) / 255.0
        pad = rec.transpose(0, 3, 1, 4, 2).reshape(H, W, 3)
        out = np.clip(pad[:h, :w] @ YIQ2RGB.T, 0.0, 1.0).astype(np.float32)
    return out


def _fit_q(q, bs):
    if bs == 8:
        return q
    idx = np.linspace(0, 7, bs).astype(np.int32)
    return q[np.ix_(idx, idx)]


# ------------------------------------------------------------ output scaling


def apply_pixel_aspect(rgb, st):
    ax, ay = float(st.pixel_aspect_x), float(st.pixel_aspect_y)
    if abs(ax - ay) < 1e-6:
        return rgb
    h, w = rgb.shape[:2]
    if ax > ay:
        nw = int(round(w * ax / ay))
        nh = h
    else:
        nw = w
        nh = int(round(h * ay / ax))
    yy, xx = np.mgrid[0:nh, 0:nw].astype(np.float32)
    return _resample(rgb, xx * (w / nw), yy * (h / nh))


def upscale(rgb, st):
    n = {'NONE': 1, '2X': 2, '3X': 3, '4X': 4}.get(str(st.output_scale), 1)
    if n <= 1:
        return rgb
    out = np.repeat(np.repeat(rgb, n, axis=0), n, axis=1)
    if st.pixel_grid and n >= 2:
        out[::n, :, :] *= 0.82
        out[:, ::n, :] *= 0.82
    return out


# ------------------------------------------------------------------ the chain


def _gpu_stage(name, rgb, st):
    """Ask the GPU chain for a stage. None means the CPU one runs."""
    if not getattr(st, 'gpu_post', False):
        return None
    try:
        from ..gpu import chain
    except Exception:                                           # noqa: BLE001
        return None
    fn = getattr(chain, name, None)
    return fn(rgb, st) if fn is not None else None


def fit_to(rgb, size):
    """Nearest-resample to exactly (w, h). Nearest keeps the pixels crisp."""
    w, h = int(size[0]), int(size[1])
    if rgb.shape[1] == w and rgb.shape[0] == h:
        return rgb
    ys = np.minimum((np.arange(h) * (rgb.shape[0] / h)).astype(np.int64),
                    rgb.shape[0] - 1)
    xs = np.minimum((np.arange(w) * (rgb.shape[1] / w)).astype(np.int64),
                    rgb.shape[1] - 1)
    return np.ascontiguousarray(rgb[ys][:, xs])


def process(image, st, frame=0, seed=0, target_size=None, allow_resize=True,
            depth=None, shaft_sources=None):
    """Linear RGBA framebuffer -> final display-referred RGBA.

    Row order is preserved; row 0 stays the bottom of the picture.

    `target_size` is a hard guarantee on the returned dimensions. The chain can
    legitimately change the image size -- pixel aspect stretches it, Pixel Scale
    multiplies it -- and a caller writing into a fixed-size buffer must not be
    handed something bigger. Blender's render result is exactly such a buffer,
    and overrunning it corrupts the heap rather than raising.
    """
    rgb = np.asarray(image, np.float32)
    alpha = rgb[:, :, 3:4].copy() if rgb.shape[2] == 4 else None
    rgb = rgb[:, :, :3]

    # A render pass other than Beauty is *data*, not a picture. Palettes,
    # dither, CRT masks and JPEG blocks destroy it -- a depth ramp quantised to
    # 16 colours reads as black, and an overdraw heat map snaps to one blue.
    if str(getattr(st, 'debug_pass', 'BEAUTY')) != 'BEAUTY':
        rgb = np.clip(rgb, 0.0, 1.0)
        if abs(float(st.gamma) - 1.0) > 1e-6:
            rgb = np.power(np.maximum(rgb, 0.0), 1.0 / max(float(st.gamma), 1e-3))
        rgb = upscale(rgb, st)
        if target_size is not None:
            rgb = fit_to(rgb, target_size)
        if alpha is not None:
            if alpha.shape[:2] != rgb.shape[:2]:
                alpha = fit_to(alpha, (rgb.shape[1], rgb.shape[0]))
            return np.concatenate([rgb, np.clip(alpha, 0, 1)], 2).astype(np.float32)
        return rgb.astype(np.float32)

    rgb = depth_of_field(rgb, depth, st)
    rgb = light_shafts(rgb, st, shaft_sources)
    rgb = glow(rgb, st)
    rgb = star_filter(rgb, st)
    rgb = lens_flare(rgb, st)
    gpu_out = _gpu_stage('display', rgb, st)
    rgb = gpu_out if gpu_out is not None else display_transform(rgb, st)
    rgb = reduce_depth(rgb, st, seed)
    rgb = composite_ntsc(rgb, st, frame)
    rgb = interlace(rgb, st, frame)
    gpu_out = _gpu_stage('crt', rgb, st)
    rgb = gpu_out if gpu_out is not None else crt(rgb, st)
    rgb = lens_distortion(rgb, st)
    rgb = jpeg_artifacts(rgb, st)
    if allow_resize:
        rgb = apply_pixel_aspect(rgb, st)
    rgb = upscale(rgb, st)
    if target_size is not None:
        rgb = fit_to(rgb, target_size)

    if alpha is not None:
        if alpha.shape[:2] != rgb.shape[:2]:
            h, w = rgb.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            alpha = _resample(alpha, xx * (alpha.shape[1] / w),
                              yy * (alpha.shape[0] / h))
        return np.concatenate([rgb, np.clip(alpha, 0, 1)], axis=2).astype(np.float32)
    return rgb.astype(np.float32)
