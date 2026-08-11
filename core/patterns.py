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


# ------------------------------------------------- deterministic sampling
#
# Soft ray shadows and ambient occlusion average jittered rays. Drawing the
# jitter from a sequential random stream made the PICTURE depend on batch
# order and thread count -- the same disease the bump chunk seam had -- and
# made the sampling impossible to reproduce on a GPU. These primitives make
# every sample a pure function of (pixel, sample index, stream, seed),
# riding the SAME integer hash the pattern textures proved bit-exact on
# real drivers. The angle comes from a 256-entry unit-circle table rather
# than sin/cos of the hash, because a driver's transcendentals round
# differently and an occlusion ray is a cliff: the table is float32 DATA,
# identical on every device that reads it.

#: cos/sin pairs for 256 evenly spaced angles, computed in float64 once and
#: frozen to float32 -- the GPU reads these exact values as a texture
CIRCLE256 = np.stack([
    np.cos(2.0 * np.pi * np.arange(256) / 256.0),
    np.sin(2.0 * np.pi * np.arange(256) / 256.0),
], axis=1).astype(np.float32)


def sample_u(spx, spy, z):
    """One uniform in [0, 1] per pixel: hash3 of the pixel and a salt.

    `spx`/`spy` are integer pixel coordinates (any int dtype), `z` the
    stream salt combining sample index, stream id and seed. 16-bit
    stratification -- plenty for jitter, and exactly reproducible."""
    return hash3(np.asarray(spx, np.int64), np.asarray(spy, np.int64),
                 np.int64(z))


def sample_circle(u):
    """(cos, sin) for a hash draw `u`, from the shared table.

    The index recovers the hash's own 16-bit integer (u * 65535 is exact
    in float32) and keeps its low 8 bits -- the same arithmetic the GLSL
    side runs, cliff-free by construction."""
    ai = (np.asarray(u * np.float32(65535.0) + np.float32(0.5),
                     np.float32)).astype(np.int64) & 255
    return CIRCLE256[ai, 0], CIRCLE256[ai, 1]


#: hash3's three lattice constants, shared by the corner-reuse fast path
_HX = 374761393
_HY = 668265263
_HZ = 1274126177


def value_noise(p):
    """Trilinearly interpolated value noise in 0..1.

    The eight corner hashes share their first line: hash3's pre-mix value
    is LINEAR in (ix, iy, iz), so corner (dx, dy, dz) is just the cell's
    base sum plus a constant offset. One multiply-sum replaces eight
    (int64 addition wraps associatively, so `(base + off) & mask` is
    bit-identical to hashing the shifted coordinates), and only the
    xorshift mix runs per corner. Same 16-bit values, ~30% of the cloud
    layer's cost gone.
    """
    fl = np.floor(p)
    # fract and the fade curve in the INPUT's own float32: subtracting the
    # int64 cell index silently promoted the whole lerp chain to float64 --
    # double the memory traffic for low bits the GLSL twin (fract in f32)
    # never had. The corner VALUES are exact 16-bit hashes either way.
    f = p - fl
    f = f * f * (3.0 - 2.0 * f)
    i = fl.astype(np.int64)
    base = i[:, 0] * _HX + i[:, 1] * _HY + i[:, 2] * _HZ

    def corner(off):
        h = (base + off) & 0x7fffffff
        h = (h ^ (h >> 13)) * 1274126177
        return ((h ^ (h >> 16)) & 0xffff).astype(np.float32) / 65535.0

    c000 = corner(0)
    c100 = corner(_HX)
    c010 = corner(_HY)
    c110 = corner(_HX + _HY)
    c001 = corner(_HZ)
    c101 = corner(_HX + _HZ)
    c011 = corner(_HY + _HZ)
    c111 = corner(_HX + _HY + _HZ)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    x00 = c000 + (c100 - c000) * fx
    x10 = c010 + (c110 - c010) * fx
    x01 = c001 + (c101 - c001) * fx
    x11 = c011 + (c111 - c011) * fx
    y0 = x00 + (x10 - x00) * fy
    y1 = x01 + (x11 - x01) * fy
    return (y0 + (y1 - y0) * fz).astype(np.float32)


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


def ridged(p, octaves=5, lacunarity=2.0, gain=0.5):
    """Ridged fractal: each octave folds signed noise and squares the fold.

    Musgrave's ridged multifractal profile (Texturing & Modeling, in
    period): n = 1 - |signed noise|, squared so the ridge line sharpens,
    summed at decaying amplitude. The squaring is the character -- without
    it this is just inverted turbulence.
    """
    total = np.zeros(p.shape[0], np.float32)
    amp, norm, freq = 1.0, 0.0, 1.0
    for _ in range(int(max(octaves, 1))):
        s = 1.0 - np.abs(signed_noise(p * freq))
        total += (s * s) * amp
        norm += amp
        amp *= gain
        freq *= lacunarity
    return (total / max(norm, 1e-6)).astype(np.float32)


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


def noise_fractal(p, kind=0, octaves=5, lacunarity=2.0, gain=0.5):
    """The raw fractal noise field, three profiles.

    kind 0 = smooth fBm, 1 = turbulence (folded, cusped), 2 = ridged
    (folded and squared). This is the node-facing face of the integer-hash
    noise the whole pattern library rides -- the one to reach for where
    Blender's own Noise texture would be used, because that one's
    sin-fract hash cannot travel to a driver and this one travels exactly.
    """
    k = int(kind)
    if k == 1:
        return turbulence(p, octaves=octaves, lacunarity=lacunarity,
                          gain=gain)
    if k == 2:
        return ridged(p, octaves=octaves, lacunarity=lacunarity, gain=gain)
    return fbm(p, octaves=octaves, lacunarity=lacunarity, gain=gain)


def cells(p, jitter=1.0, feature=0):
    """Worley's cellular texture (SIGGRAPH 1996 -- in period), by feature.

    feature 0 = F1 (distance to the nearest point), 1 = F2, 2 = F2-F1
    (ridges on the cell borders), 3 = the cell's own hashed id as a flat
    shade -- the stained-glass look. Returns (fac, cell id) so the id can
    drive per-cell variation whatever the feature.
    """
    f1, f2, cid = worley(p, jitter=jitter, n_closest=2)
    k = int(feature)
    if k == 1:
        v = f2
    elif k == 2:
        v = f2 - f1
    elif k == 3:
        v = cid
    else:
        v = f1
    return np.clip(v, 0.0, 1.0).astype(np.float32), cid.astype(np.float32)


def tv_static(p, frame=0):
    """Per-cell white noise, reseeded every frame -- an untuned television.

    Solid like every other pattern: the cells live in the SURFACE's own
    space (scale the vector for the set's pixel size), so the static sits
    on an in-scene screen the way it should, not on the camera. The frame
    number salts the hash, which is what makes it crawl.
    """
    c = np.floor(p)
    return hash3(c[:, 0].astype(np.int64), c[:, 1].astype(np.int64),
                 c[:, 2].astype(np.int64) + int(frame) * 7919)


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


# ------------------------------------------------- the POV-Ray pattern family
#
# These seven are the ones that came in POV-Ray's own pattern list and were
# copied through 3D Studio's material editor and Bryce's Deep Texture Editor
# after it. Each is written from the pattern's published definition rather than
# approximated by something that looks similar, for the same reason the
# reflectance models are.


def bozo(p, turb=0.0, octaves=4, lacunarity=2.0):
    """POV-Ray's `bozo`: plain value noise, optionally turbulence-displaced.

    The simplest pattern in the box and the one most 1990s materials were built
    on -- a colour map over noise. With turbulence above zero the point is
    displaced before it is sampled, which is what turned bozo into clouds.
    """
    q = p
    if turb > 1e-6:
        d = np.stack([signed_noise(p + 11.3), signed_noise(p + 47.1),
                      signed_noise(p + 83.7)], axis=1)
        for i in range(1, int(max(octaves, 1))):
            f = lacunarity ** i
            d = d + np.stack([signed_noise((p + 11.3) * f),
                              signed_noise((p + 47.1) * f),
                              signed_noise((p + 83.7) * f)], axis=1) / f
        q = p + d * turb
    return np.clip(value_noise(q), 0.0, 1.0).astype(np.float32)


def agate(p, turb=1.0, octaves=6, bands=1.1, sharpness=0.77, axis=2):
    """POV-Ray's `agate`: a sine band along one axis, thrown about by a large
    turbulence and then raised to 0.77.

    POV computes `pow(0.5 * (sin(1.3 * turb + 1.1 * z) + 1), 0.77)`, and that
    0.77 is the whole character of the pattern -- it pushes the midtones up so
    the bands read as layered stone rather than as a sine wave.
    """
    ax = int(axis) % 3
    t = turbulence(p, octaves=octaves) * 2.0 - 1.0
    v = 0.5 * (np.sin(1.3 * t * max(turb, 0.0) +
                      max(bands, 1e-3) * p[:, ax]) + 1.0)
    return np.power(np.clip(v, 0.0, 1.0),
                    max(sharpness, 0.01)).astype(np.float32)


def leopard(p, spot=1.0):
    """POV-Ray's `leopard`: ((sin x + sin y + sin z) / 3) squared.

    Three interfering sines squared, which lands a rounded spot in the middle
    of every unit cell. It is the pattern every 1990s "animal print" material
    was actually made of.
    """
    s = (np.sin(p[:, 0]) + np.sin(p[:, 1]) + np.sin(p[:, 2])) / 3.0
    v = s * s
    return np.power(np.clip(v, 0.0, 1.0), max(spot, 0.01)).astype(np.float32)


def onion(p, thickness=1.0, sharpness=1.0):
    """POV-Ray's `onion`: concentric spherical shells around the origin.

    The value ramps 0 to 1 across each shell, so a colour map over it gives the
    layers. Thickness scales the shell spacing; sharpness bends the ramp.
    """
    r = np.linalg.norm(p, axis=1) / max(thickness, 1e-4)
    v = r - np.floor(r)
    return np.power(np.clip(v, 0.0, 1.0),
                    max(sharpness, 0.01)).astype(np.float32)


def bumps(p, roundness=1.0, octaves=1, lacunarity=2.0, gain=0.5):
    """POV-Ray's `bumps`: smooth noise, read as a height field.

    A single octave by default, because that is what makes a bump rather than a
    crumple -- add octaves and it becomes terrain. Roundness above 1 flattens
    the troughs and leaves the peaks proud, which is the difference between a
    bumped surface and a noisy one.
    """
    v = fbm(p, octaves=octaves, lacunarity=lacunarity, gain=gain)
    v = np.clip(v, 0.0, 1.0)
    v = v * v * (3.0 - 2.0 * v)                 # smoothstep: rounds the tops
    return np.power(v, max(roundness, 0.01)).astype(np.float32)


def wrinkles(p, octaves=8, lacunarity=2.0, crease=1.0):
    """POV-Ray's `wrinkles`: folded noise summed at halving amplitude.

    Every octave is |signed noise|, so each one creases where it crosses zero.
    Ten octaves is what POV used; the creases from the coarse ones are the
    folds and the fine ones are the paper's grain.
    """
    total = np.zeros(p.shape[0], np.float32)
    amp, norm, freq = 1.0, 0.0, 1.0
    for _ in range(int(max(octaves, 1))):
        total += np.abs(signed_noise(p * freq)) * amp
        norm += amp
        amp *= 0.5
        freq *= lacunarity
    # folded noise averages about a quarter of its range, so the sum is
    # lifted to reach the top of a colour map rather than sitting in the
    # bottom third of it. A fixed gain, not a per-batch normalisation: the
    # renderer shades in chunks, and anything normalised across a chunk would
    # tile differently in every one of them
    v = np.clip(1.4 * total / max(norm, 1e-6), 0.0, 1.0)
    return np.power(v, max(crease, 0.01)).astype(np.float32)


def brick(p, width=0.25, height=0.125, mortar=0.05, offset=0.5, bevel=0.12):
    """Running-bond brickwork with mortar courses.

    POV's `brick` returns 0 in the mortar and 1 in the brick and nothing in
    between. This keeps that -- `inside` is the hard answer -- but also returns
    a bevel ramp for shading the edges and a per-brick id, because a wall where
    every brick is exactly the same colour is the one thing that never looked
    right.
    """
    w = max(float(width), 1e-3)
    h = max(float(height), 1e-3)
    row = np.floor(p[:, 2] / h)
    shift = row * float(offset) * w
    u = (p[:, 0] + shift) / w
    col = np.floor(u)
    fu = u - col
    fv = p[:, 2] / h - row
    m = np.clip(float(mortar), 0.0, 0.49)
    inside = ((fu > m) & (fu < 1.0 - m) & (fv > m) & (fv < 1.0 - m))
    b = max(float(bevel), 1e-4)
    du = np.minimum(fu - m, (1.0 - m) - fu) / b
    dv = np.minimum(fv - m, (1.0 - m) - fv) / b
    ramp = np.clip(np.minimum(du, dv), 0.0, 1.0)
    fac = np.where(inside, ramp, 0.0).astype(np.float32)
    bid = hash3(col.astype(np.int64), row.astype(np.int64),
                np.zeros(p.shape[0], np.int64))
    return fac, bid.astype(np.float32), inside
