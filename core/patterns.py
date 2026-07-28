"""Solid procedural patterns, of the kind 1990s renderers shipped.

These are the textures that came in the box with POV-Ray, 3D Studio, Bryce's
Deep Texture Editor and Infini-D — marble veined with turbulence, wood turned
from concentric rings, granite built by stacking noise octaves. They are solid
textures: evaluated in 3D, so a shape carved out of marble has veins that run
through it rather than a picture wrapped round it.

bpy-free. The noise primitives live here and are shared with core/sky.py.
"""

import numpy as np

TWO_PI = np.float32(2.0 * np.pi)


# ---------------------------------------------------------------- primitives


def hash3(ix, iy, iz):
    h = (ix * 374761393 + iy * 668265263 + iz * 1274126177) & 0x7fffffff
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xffff).astype(np.float32) / 65535.0


def hash3f(p, salt=0.0):
    c = np.floor(p)
    return hash3(c[:, 0].astype(np.int64), c[:, 1].astype(np.int64),
                 (c[:, 2] + salt).astype(np.int64))


def value_noise(p):
    """Trilinearly interpolated value noise in 0..1."""
    i = np.floor(p).astype(np.int64)
    f = p - i
    f = f * f * (3.0 - 2.0 * f)
    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    c = {}
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                c[(dx, dy, dz)] = hash3(ix + dx, iy + dy, iz + dz)
    x00 = c[(0, 0, 0)] + (c[(1, 0, 0)] - c[(0, 0, 0)]) * f[:, 0]
    x10 = c[(0, 1, 0)] + (c[(1, 1, 0)] - c[(0, 1, 0)]) * f[:, 0]
    x01 = c[(0, 0, 1)] + (c[(1, 0, 1)] - c[(0, 0, 1)]) * f[:, 0]
    x11 = c[(0, 1, 1)] + (c[(1, 1, 1)] - c[(0, 1, 1)]) * f[:, 0]
    y0 = x00 + (x10 - x00) * f[:, 1]
    y1 = x01 + (x11 - x01) * f[:, 1]
    return (y0 + (y1 - y0) * f[:, 2]).astype(np.float32)


def signed_noise(p):
    return value_noise(p) * 2.0 - 1.0


def fbm(p, octaves=5, lacunarity=2.0, gain=0.5):
    total = np.zeros(p.shape[0], np.float32)
    amp, norm, freq = 1.0, 0.0, 1.0
    for _ in range(int(max(octaves, 1))):
        total += value_noise(p * freq) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-6)


def turbulence(p, octaves=5, lacunarity=2.0, gain=0.5):
    """Sum of |signed noise|. The cusps are the point."""
    total = np.zeros(p.shape[0], np.float32)
    amp, norm, freq = 1.0, 0.0, 1.0
    for _ in range(int(max(octaves, 1))):
        total += np.abs(signed_noise(p * freq)) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / max(norm, 1e-6)


def worley(p, jitter=1.0, n_closest=2):
    """Distances to the nearest feature points. Returns (F1, F2, cell id)."""
    cell = np.floor(p)
    best = np.full((p.shape[0], max(n_closest, 1)), 1e9, np.float32)
    best_id = np.zeros(p.shape[0], np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = cell + np.array([dx, dy, dz], np.float32)
                jx = hash3f(c, 0.0)
                jy = hash3f(c, 31.0)
                jz = hash3f(c, 71.0)
                pt = c + np.stack([jx, jy, jz], 1) * jitter + (0.5 * (1.0 - jitter))
                d = np.linalg.norm(p - pt, axis=1)
                closer = d < best[:, 0]
                if best.shape[1] > 1:
                    best[:, 1] = np.where(closer, best[:, 0],
                                          np.minimum(best[:, 1], d))
                best_id = np.where(closer, hash3f(c, 137.0), best_id)
                best[:, 0] = np.minimum(best[:, 0], d)
    return best[:, 0], (best[:, 1] if best.shape[1] > 1 else best[:, 0]), best_id


# ------------------------------------------------------------------ patterns


def marble(p, turb=1.0, octaves=5, veins=1.0, sharpness=1.0, axis=0):
    """Sine banding displaced by turbulence -- the POV-Ray marble."""
    t = turbulence(p, octaves=octaves) * turb
    coord = p[:, int(axis) % 3]
    v = np.sin((coord + t * 4.0) * np.pi * max(veins, 1e-3))
    v = v * 0.5 + 0.5
    return np.power(np.clip(v, 0.0, 1.0), max(sharpness, 0.01)).astype(np.float32)


def wood(p, rings=8.0, turb=0.35, octaves=4, grain=0.4, axis=2):
    """Concentric rings around an axis, warped by turbulence, plus fine grain."""
    ax = int(axis) % 3
    a, b = [i for i in range(3) if i != ax]
    r = np.sqrt(p[:, a] ** 2 + p[:, b] ** 2)
    r = r + turbulence(p, octaves=octaves) * turb
    v = np.sin(r * max(rings, 1e-3) * TWO_PI) * 0.5 + 0.5
    if grain > 0.0:
        fine = value_noise(np.stack([p[:, a] * 48.0, p[:, b] * 48.0,
                                     p[:, ax] * 3.0], 1))
        v = np.clip(v + (fine - 0.5) * grain, 0.0, 1.0)
    return v.astype(np.float32)


def granite(p, octaves=6, contrast=1.6, speckle=0.35):
    """Stacked high-frequency noise, then contrast-stretched into speckles."""
    v = fbm(p, octaves=octaves, lacunarity=2.4, gain=0.62)
    if speckle > 0.0:
        v = v + (value_noise(p * 9.0) - 0.5) * speckle
    v = np.clip((v - 0.5) * max(contrast, 0.01) + 0.5, 0.0, 1.0)
    return v.astype(np.float32)


def dents(p, size=1.0, octaves=3, depth=1.0):
    """3D Studio's Dent: sparse worley pits rather than smooth noise."""
    f1, _f2, _i = worley(p / max(size, 1e-3), jitter=1.0)
    v = np.clip(1.0 - f1, 0.0, 1.0)
    v = np.power(v, max(1.0 / max(depth, 0.01), 0.01))
    if octaves > 1:
        v = v * 0.7 + turbulence(p, octaves=int(octaves)) * 0.3
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def crackle(p, jitter=1.0, width=0.06, smooth=0.02):
    """The boundary network between Worley cells -- crazed glaze, dried mud."""
    f1, f2, _i = worley(p, jitter=jitter, n_closest=2)
    edge = f2 - f1
    w = max(width, 1e-4)
    v = 1.0 - np.clip((edge - w) / max(smooth, 1e-4), 0.0, 1.0)
    return np.clip(v, 0.0, 1.0).astype(np.float32)


def plasma(p, scale=1.0, time=0.0, complexity=3.0):
    """Interfering sine fields -- the demoscene plasma, and it belongs here."""
    x, y = p[:, 0] * scale, p[:, 1] * scale
    c = max(complexity, 0.1)
    v = np.sin(x * c + time)
    v = v + np.sin((y * c + time) * 0.7)
    v = v + np.sin((x + y) * c * 0.5 + time * 1.3)
    v = v + np.sin(np.sqrt(x * x + y * y) * c * 1.1 + time * 0.8)
    return ((v * 0.25) * 0.5 + 0.5).astype(np.float32)


def ripples(p, sources=3, frequency=8.0, time=0.0, decay=0.6, seed=0):
    """Concentric waves from several point sources, interfering."""
    total = np.zeros(p.shape[0], np.float32)
    rng = np.random.default_rng(int(seed) + 1234)
    n = int(max(sources, 1))
    for i in range(n):
        c = (rng.random(3).astype(np.float32) - 0.5) * 2.0
        d = np.linalg.norm(p - c[None, :], axis=1)
        total += np.sin(d * max(frequency, 1e-3) - time * 3.0) * \
            np.exp(-d * max(decay, 0.0))
    return (total / n * 0.5 + 0.5).astype(np.float32)


def starfield(p, density=0.5, size=0.35, twinkle=0.0, time=0.0):
    """Points on a grid with random brightness. For 1990s space scenes."""
    cell = np.floor(p)
    h = hash3f(p, 0.0)
    thresh = 1.0 - np.clip(density, 0.0, 1.0) * 0.25
    frac = p - cell
    centre = np.stack([hash3f(p, 11.0), hash3f(p, 23.0), hash3f(p, 47.0)], 1)
    d = np.linalg.norm(frac - centre, axis=1)
    radius = max(size, 1e-3) * 0.5
    disc = np.clip(1.0 - d / radius, 0.0, 1.0)
    mag = np.where(h > thresh, (h - thresh) / max(1.0 - thresh, 1e-4), 0.0)
    if twinkle > 0.0:
        phase = hash3f(p, 91.0) * TWO_PI
        mag = mag * (1.0 - twinkle * 0.5 * (1.0 + np.sin(time * 3.0 + phase)) * 0.5)
    return np.clip(disc * mag, 0.0, 1.0).astype(np.float32)


def weave(p, thickness=0.35, gap=0.08, warp=0.0):
    """Over-under fabric. Returns (value, is_warp) so the two threads can differ."""
    x, y = p[:, 0], p[:, 1]
    if warp > 0.0:
        x = x + signed_noise(p * 3.0) * warp
        y = y + signed_noise(p * 3.0 + 17.0) * warp
    fx, fy = x - np.floor(x), y - np.floor(y)
    over = ((np.floor(x) + np.floor(y)) % 2) == 0
    t = np.clip(thickness, 0.01, 0.99)
    band_x = np.abs(fx - 0.5) < t * 0.5
    band_y = np.abs(fy - 0.5) < t * 0.5
    shade_x = np.cos((fx - 0.5) / max(t, 1e-3) * np.pi) * 0.5 + 0.5
    shade_y = np.cos((fy - 0.5) / max(t, 1e-3) * np.pi) * 0.5 + 0.5
    on_warp = np.where(over, band_x, band_y)
    val = np.where(on_warp, shade_x, shade_y)
    empty = ~(band_x | band_y)
    val = np.where(empty, 0.0, val)
    if gap > 0.0:
        edge = np.minimum(np.abs(fx - 0.5), np.abs(fy - 0.5))
        val = val * np.clip(edge / max(gap, 1e-4), 0.0, 1.0)
    return np.clip(val, 0.0, 1.0).astype(np.float32), on_warp


def scratches(p, count=6, width=0.02, length=1.0, seed=0, anisotropy=1.0):
    """Fine anisotropic scuffs, for brushed metal."""
    total = np.zeros(p.shape[0], np.float32)
    rng = np.random.default_rng(int(seed) + 77)
    n = int(max(count, 1))
    for _ in range(n):
        ang = rng.random() * np.pi
        ang = ang * (1.0 - anisotropy)
        dx, dy = np.cos(ang), np.sin(ang)
        offset = (rng.random() - 0.5) * 4.0
        proj = p[:, 0] * (-dy) + p[:, 1] * dx + offset
        along = p[:, 0] * dx + p[:, 1] * dy
        band = np.exp(-(proj * proj) / max(width * width, 1e-8))
        mask = np.clip(1.0 - np.abs(along) / max(length, 1e-3), 0.0, 1.0)
        jag = 0.6 + 0.4 * value_noise(np.stack([along * 24.0, np.full_like(along, 0.0),
                                                np.full_like(along, 0.0)], 1))
        total = np.maximum(total, band * mask * jag)
    return np.clip(total, 0.0, 1.0).astype(np.float32)


def tiles(p, rows=4.0, columns=4.0, grout=0.06, offset=0.0, bevel=0.15):
    """Rectangular tiles with grout lines and a bevelled edge shade."""
    y = p[:, 1] * max(rows, 1e-3)
    row = np.floor(y)
    x = p[:, 0] * max(columns, 1e-3) + row * offset
    fx, fy = x - np.floor(x), y - row
    g = max(grout, 0.0) * 0.5
    inside = (fx > g) & (fx < 1.0 - g) & (fy > g) & (fy < 1.0 - g)
    edge = np.minimum(np.minimum(fx - g, 1.0 - g - fx),
                      np.minimum(fy - g, 1.0 - g - fy))
    shade = np.clip(edge / max(bevel, 1e-4), 0.0, 1.0)
    val = np.where(inside, 0.35 + 0.65 * shade, 0.0)
    tid = hash3(np.floor(x).astype(np.int64), row.astype(np.int64),
                np.zeros_like(row, np.int64))
    return val.astype(np.float32), tid.astype(np.float32), inside


def spiral(p, turns=4.0, sharpness=1.0, axis=2, twist=0.0):
    """Archimedean spiral banding around an axis."""
    ax = int(axis) % 3
    a, b = [i for i in range(3) if i != ax]
    ang = np.arctan2(p[:, b], p[:, a])
    r = np.sqrt(p[:, a] ** 2 + p[:, b] ** 2)
    v = np.sin(ang * max(turns, 1e-3) + r * TWO_PI + p[:, ax] * twist)
    v = v * 0.5 + 0.5
    return np.power(np.clip(v, 0.0, 1.0), max(sharpness, 0.01)).astype(np.float32)
