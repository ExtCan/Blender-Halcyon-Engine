"""Blender Internal texture evaluation, ported 1:1.

This is the texture engine that shipped inside Blender from the 1990s
until 2.79, re-implemented over NumPy so a legacy file's Clouds, Wood,
Marble, Magic, Blend, Stucci, Musgrave, Voronoi and Distorted Noise
render as THEMSELVES rather than as lookalikes. The algorithms are
transcriptions of the published 2.79b source (GPL-2.0-or-later):

* noise bases, turbulence, voronoi and musgrave: blenlib/intern/noise.c
* the texture-type formulas and BRICONT(RGB): render_texture.c and
  render/intern/include/texture.h, functions transcribed verbatim
* the dispatcher (including the 1/noisesize pre-scale that ONLY
  musgrave/voronoi/distnoise receive) : multitex()
* colorbands: blenkernel/intern/texture.c do_colorband()

The three lookup tables live in bitex_tables.py with their own
provenance notes. Everything here is bpy-free and vectorised over
(N,) coordinate arrays; the GPU twin lives in gpu/procedural.py and
must match this module to float precision -- the same twin discipline
as every other Halcyon pattern.

Two knowing departures, both recorded in the changelog:

* TEX_NOISE (the static texture) used a per-thread RNG in BI -- a
  different picture every render by design. Halcyon is deterministic,
  so it hashes pixel and frame instead: same look, same file, same
  frame -> same pixels.
* The 'Original Perlin' basis (STDPERLIN) is Perlin's 1985 noise over
  the same hash/gradient tables (org_perlin_noise below) -- it shared
  Blender's tables all along, so the old Improved-Perlin fallback was
  never necessary. The field's clouds finally asked for the real one.
"""

import numpy as np

from .bitex_tables import HASH, HASHPNT3, HASHVEC3

# ---------------------------------------------------------------- constants

#: Tex->type
TEX_CLOUDS, TEX_WOOD, TEX_MARBLE, TEX_MAGIC, TEX_BLEND = 1, 2, 3, 4, 5
TEX_STUCCI, TEX_NOISE, TEX_IMAGE = 6, 7, 8
TEX_MUSGRAVE, TEX_VORONOI, TEX_DISTNOISE = 11, 12, 13

#: noise bases (Tex->noisebasis / noisebasis2)
B_BLENDER, B_STDPERLIN, B_NEWPERLIN = 0, 1, 2
B_VORONOI_F1, B_VORONOI_F2, B_VORONOI_F3, B_VORONOI_F4 = 3, 4, 5, 6
B_VORONOI_F2F1, B_VORONOI_CRACKLE, B_CELLNOISE = 7, 8, 14

#: stypes, per family (DNA_texture_types.h)
STY_CLOUDS_COLOR = 1
STY_WOOD_BAND, STY_WOOD_RING, STY_WOOD_BANDNOISE, STY_WOOD_RINGNOISE = \
    0, 1, 2, 3
STY_MARBLE_SOFT, STY_MARBLE_SHARP, STY_MARBLE_SHARPER = 0, 1, 2
WAVE_SIN, WAVE_SAW, WAVE_TRI = 0, 1, 2
STY_BLEND_LIN, STY_BLEND_QUAD, STY_BLEND_EASE, STY_BLEND_DIAG = 0, 1, 2, 3
STY_BLEND_SPHERE, STY_BLEND_HALO, STY_BLEND_RAD = 4, 5, 6
STY_STUCCI_PLASTIC, STY_STUCCI_WALLIN, STY_STUCCI_WALLOUT = 0, 1, 2
STY_MG_MFRACTAL, STY_MG_RIDGED, STY_MG_HYBRID = 0, 1, 2
STY_MG_FBM, STY_MG_HTERRAIN = 3, 4

TEX_COLORBAND, TEX_FLIPBLEND, TEX_NO_CLAMP = 1, 2, 1024


def _f32(a):
    return np.asarray(a, np.float32)


# ------------------------------------------------------------- noise bases
#
# Every basis has an unsigned (0..1) and a signed (-1..1) form, exactly
# as noise.c pairs them.


def org_blender_noise(x, y, z):
    """orgBlenderNoise(): the default basis since the beginning."""
    x, y, z = _f32(x), _f32(y), _f32(z)
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    ox, oy, oz = x - fx, y - fy, z - fz
    ix = fx.astype(np.int64)
    iy = fy.astype(np.int64)
    iz = fz.astype(np.int64)
    jx, jy, jz = ox - 1.0, oy - 1.0, oz - 1.0

    cn1, cn2, cn3 = ox * ox, oy * oy, oz * oz
    cn4, cn5, cn6 = jx * jx, jy * jy, jz * jz
    cn1 = 1.0 - 3.0 * cn1 + 2.0 * cn1 * ox
    cn2 = 1.0 - 3.0 * cn2 + 2.0 * cn2 * oy
    cn3 = 1.0 - 3.0 * cn3 + 2.0 * cn3 * oz
    cn4 = 1.0 - 3.0 * cn4 - 2.0 * cn4 * jx
    cn5 = 1.0 - 3.0 * cn5 - 2.0 * cn5 * jy
    cn6 = 1.0 - 3.0 * cn6 - 2.0 * cn6 * jz

    b00 = HASH[HASH[ix & 255] + (iy & 255)]
    b10 = HASH[HASH[(ix + 1) & 255] + (iy & 255)]
    b01 = HASH[HASH[ix & 255] + ((iy + 1) & 255)]
    b11 = HASH[HASH[(ix + 1) & 255] + ((iy + 1) & 255)]
    b20, b21 = iz & 255, (iz + 1) & 255

    n = np.full(x.shape, 0.5, np.float32)
    for i_fac, bz, bxy, px, py, pz in (
            (cn1 * cn2 * cn3, b20, b00, ox, oy, oz),
            (cn1 * cn2 * cn6, b21, b00, ox, oy, jz),
            (cn1 * cn5 * cn3, b20, b01, ox, jy, oz),
            (cn1 * cn5 * cn6, b21, b01, ox, jy, jz),
            (cn4 * cn2 * cn3, b20, b10, jx, oy, oz),
            (cn4 * cn2 * cn6, b21, b10, jx, oy, jz),
            (cn4 * cn5 * cn3, b20, b11, jx, jy, oz),
            (cn4 * cn5 * cn6, b21, b11, jx, jy, jz)):
        h = HASHVEC3[HASH[bz + bxy]]
        n = n + i_fac * (h[:, 0] * px + h[:, 1] * py + h[:, 2] * pz)
    return np.clip(n, 0.0, 1.0).astype(np.float32)


def org_blender_noise_s(x, y, z):
    return 2.0 * org_blender_noise(x, y, z) - 1.0


def _npfade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _grad(h, x, y, z):
    h = h & 15
    u = np.where(h < 8, x, y)
    v = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
    return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)


def new_perlin(x, y, z):
    """newPerlin(): Perlin's improved noise over Blender's hash table."""
    x, y, z = _f32(x), _f32(y), _f32(z)
    u, v, w = np.floor(x), np.floor(y), np.floor(z)
    X = u.astype(np.int64) & 255
    Y = v.astype(np.int64) & 255
    Z = w.astype(np.int64) & 255
    x, y, z = x - u, y - v, z - w
    u, v, w = _npfade(x), _npfade(y), _npfade(z)
    A = HASH[X] + Y
    AA, AB = HASH[A] + Z, HASH[A + 1] + Z
    B = HASH[X + 1] + Y
    BA, BB = HASH[B] + Z, HASH[B + 1] + Z

    def lerp(t, a, b):
        return a + t * (b - a)

    return lerp(w, lerp(v, lerp(u, _grad(HASH[AA], x, y, z),
                                _grad(HASH[BA], x - 1, y, z)),
                        lerp(u, _grad(HASH[AB], x, y - 1, z),
                             _grad(HASH[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, _grad(HASH[AA + 1], x, y, z - 1),
                             _grad(HASH[BA + 1], x - 1, y, z - 1)),
                     lerp(u, _grad(HASH[AB + 1], x, y - 1, z - 1),
                          _grad(HASH[BB + 1], x - 1, y - 1,
                                z - 1)))).astype(np.float32)


def new_perlin_u(x, y, z):
    return 0.5 + 0.5 * new_perlin(x, y, z)


def org_perlin_noise(x, y, z):
    """orgPerlinNoise(): Perlin's ORIGINAL 1985 noise, Blender's exact
    form -- the +10000 domain shift, the 256-fold lattice, gradients
    from hashvectf, s-curve fades and the final 1.5 scale. This is the
    'Original Perlin' basis (STDPERLIN), which fell back to Improved
    Perlin until the field's clouds asked for the real thing."""
    def setup(v):
        t = _f32(v) + np.float32(10000.0)
        b0 = t.astype(np.int64) & 255
        b1 = (b0 + 1) & 255
        r0 = t - np.floor(t)
        r1 = r0 - 1.0
        return b0, b1, r0.astype(np.float32), r1.astype(np.float32)

    def surve(t):
        return t * t * (3.0 - 2.0 * t)

    bx0, bx1, rx0, rx1 = setup(x)
    by0, by1, ry0, ry1 = setup(y)
    bz0, bz1, rz0, rz1 = setup(z)

    i = HASH[bx0]
    j = HASH[bx1]
    b00 = HASH[i + by0]
    b10 = HASH[j + by0]
    b01 = HASH[i + by1]
    b11 = HASH[j + by1]

    sx, sy, sz = surve(rx0), surve(ry0), surve(rz0)

    def at(bxy, bz, rx, ry, rz):
        h = HASHVEC3[HASH[bxy + bz]]
        return rx * h[:, 0] + ry * h[:, 1] + rz * h[:, 2]

    def lerp(t, a, b):
        return a + t * (b - a)

    a = lerp(sx, at(b00, bz0, rx0, ry0, rz0), at(b10, bz0, rx1, ry0, rz0))
    b = lerp(sx, at(b01, bz0, rx0, ry1, rz0), at(b11, bz0, rx1, ry1, rz0))
    c = lerp(sy, a, b)
    a = lerp(sx, at(b00, bz1, rx0, ry0, rz1), at(b10, bz1, rx1, ry0, rz1))
    b = lerp(sx, at(b01, bz1, rx0, ry1, rz1), at(b11, bz1, rx1, ry1, rz1))
    d = lerp(sy, a, b)
    return (1.5 * lerp(sz, c, d)).astype(np.float32)


def org_perlin_noise_u(x, y, z):
    return 0.5 + 0.5 * org_perlin_noise(x, y, z)


def cellnoise_u(x, y, z):
    """cellNoiseU(): the integer-hash cell noise, with the precision
    nudge BI applies to dodge unit-coordinate ties."""
    x = (_f32(x) + 0.000001) * 1.00001
    y = (_f32(y) + 0.000001) * 1.00001
    z = (_f32(z) + 0.000001) * 1.00001
    xi = np.floor(x).astype(np.int64)
    yi = np.floor(y).astype(np.int64)
    zi = np.floor(z).astype(np.int64)
    n = (xi + yi * 1301 + zi * 314159).astype(np.uint32)
    n = n ^ (n << np.uint32(13))
    val = (n * (n * n * np.uint32(15731) + np.uint32(789221))
           + np.uint32(1376312589))
    return (val.astype(np.float64) / 4294967296.0).astype(np.float32)


def cellnoise_s(x, y, z):
    return 2.0 * cellnoise_u(x, y, z) - 1.0


def cellnoise_v3(x, y, z):
    """cellNoiseV(): three decorrelated cell noises, as BI colours
    voronoi cells."""
    r = cellnoise_u(x, y, z)
    g = cellnoise_u(y, x, z)
    b = cellnoise_u(y, z, x)
    return np.stack([r, g, b], axis=-1)


_VOFF = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1)
                  for k in (-1, 0, 1)], np.int64)


def voronoi(x, y, z, me=2.5, dtype=0):
    """voronoi(): distances to (and positions of) the four nearest
    feature points, with BI's seven distance metrics. Ties resolve to
    the earlier neighbour cell exactly as the sequential insertion in
    noise.c resolves them."""
    x, y, z = _f32(x), _f32(y), _f32(z)
    n = x.shape[0]
    p = np.stack([x, y, z], axis=1)
    base = np.floor(p).astype(np.int64)                     # (N,3)
    cells = base[:, None, :] + _VOFF[None, :, :]            # (N,27,3)
    cx, cy, cz = cells[..., 0], cells[..., 1], cells[..., 2]
    # HASHPNT(x,y,z): hashpntf + 3*hash[(hash[(hash[z&255]+y)&255]+x)&255]
    hidx = HASH[(HASH[(HASH[cz & 255] + cy) & 255] + cx) & 255]
    pnt = HASHPNT3[hidx] + cells.astype(np.float32)         # (N,27,3)
    d = p[:, None, :] - pnt                                 # (N,27,3)
    ax, ay, az = np.abs(d[..., 0]), np.abs(d[..., 1]), np.abs(d[..., 2])
    if dtype == 1:                                          # squared
        dist = (d * d).sum(axis=2)
    elif dtype == 2:                                        # manhattan
        dist = ax + ay + az
    elif dtype == 3:                                        # chebychev
        dist = np.maximum(np.maximum(ax, ay), az)
    elif dtype == 4:                                        # minkovsky 1/2
        s = np.sqrt(ax) + np.sqrt(ay) + np.sqrt(az)
        dist = s * s
    elif dtype == 5:                                        # minkovsky 4
        q = d * d
        dist = np.sqrt(np.sqrt((q * q).sum(axis=2)))
    elif dtype == 6:                                        # minkovsky e
        me = max(float(me), 1e-6)
        dist = (ax ** me + ay ** me + az ** me) ** (1.0 / me)
    else:                                                   # real distance
        dist = np.sqrt((d * d).sum(axis=2))
    # stable ascending sort == sequential strict-< insertion
    order = np.argsort(dist.astype(np.float32), axis=1, kind='stable')[:, :4]
    rows = np.arange(n)[:, None]
    da = dist[rows, order].astype(np.float32)               # (N,4)
    pa = pnt[rows, order].reshape(n, 12).astype(np.float32)  # (N,12)
    return da, pa


def _voronoi_base(f):
    def u(x, y, z):
        da, _pa = voronoi(x, y, z)
        return f(da)
    return u


_VOR_U = {
    B_VORONOI_F1: _voronoi_base(lambda da: da[:, 0]),
    B_VORONOI_F2: _voronoi_base(lambda da: da[:, 1]),
    B_VORONOI_F3: _voronoi_base(lambda da: da[:, 2]),
    B_VORONOI_F4: _voronoi_base(lambda da: da[:, 3]),
    B_VORONOI_F2F1: _voronoi_base(lambda da: da[:, 1] - da[:, 0]),
}


def voronoi_crackle_u(x, y, z):
    da, _pa = voronoi(x, y, z)
    return np.minimum(10.0 * (da[:, 1] - da[:, 0]), 1.0).astype(np.float32)


def basis_u(nbas):
    """The unsigned (0..1) noise for a basis code."""
    if nbas == B_BLENDER:
        return org_blender_noise
    if nbas == B_STDPERLIN:
        return org_perlin_noise_u
    if nbas == B_NEWPERLIN:
        return new_perlin_u
    if nbas in _VOR_U:
        return _VOR_U[nbas]
    if nbas == B_VORONOI_CRACKLE:
        return voronoi_crackle_u
    if nbas == B_CELLNOISE:
        return cellnoise_u
    return org_blender_noise


def basis_s(nbas):
    """The signed (-1..1) noise for a basis code."""
    if nbas == B_BLENDER:
        return org_blender_noise_s
    if nbas == B_STDPERLIN:
        return org_perlin_noise
    if nbas == B_NEWPERLIN:
        return new_perlin
    if nbas == B_CELLNOISE:
        return cellnoise_s
    u = basis_u(nbas)

    def s(x, y, z):
        return 2.0 * u(x, y, z) - 1.0
    return s


def gnoise(noisesize, x, y, z, hard, nbas):
    """BLI_gNoise(): one octave, hard-folded when asked."""
    x, y, z = _f32(x), _f32(y), _f32(z)
    if noisesize != 0.0:
        inv = np.float32(1.0 / noisesize)
        x, y, z = x * inv, y * inv, z * inv
    t = basis_u(nbas)(x, y, z)
    if hard:
        return np.abs(2.0 * t - 1.0).astype(np.float32)
    return t.astype(np.float32)


def gturbulence(noisesize, x, y, z, oct_, hard, nbas):
    """BLI_gTurbulence(): oct+1 octaves summed at halving amplitude,
    normalised by (1<<oct)/((1<<(oct+1))-1)."""
    x, y, z = _f32(x), _f32(y), _f32(z)
    if noisesize != 0.0:
        inv = np.float32(1.0 / noisesize)
        x, y, z = x * inv, y * inv, z * inv
    fn = basis_u(nbas)
    oct_ = max(int(oct_), 0)
    total = np.zeros(x.shape, np.float32)
    amp = np.float32(1.0)
    for _ in range(oct_ + 1):
        t = fn(x, y, z)
        if hard:
            t = np.abs(2.0 * t - 1.0)
        total = total + t * amp
        amp = amp * np.float32(0.5)
        x, y, z = x * 2.0, y * 2.0, z * 2.0
    total *= np.float32((1 << oct_) / float((1 << (oct_ + 1)) - 1))
    return total


# ---------------------------------------------------------------- musgrave
#
# F. Kenton Musgrave's fractal family exactly as noise.c carries it:
# integer octaves iterate, the fractional remainder blends the last one.


def mg_fbm(x, y, z, H, lacunarity, octaves, nbas):
    fn = basis_s(nbas)
    x, y, z = _f32(x), _f32(y), _f32(z)
    pwHL = np.float32(lacunarity ** -H)
    pwr = np.float32(1.0)
    value = np.zeros(x.shape, np.float32)
    for _ in range(int(octaves)):
        value = value + fn(x, y, z) * pwr
        pwr *= pwHL
        x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
    rmd = octaves - np.floor(octaves)
    if rmd != 0.0:
        value = value + np.float32(rmd) * fn(x, y, z) * pwr
    return value


def mg_multifractal(x, y, z, H, lacunarity, octaves, nbas):
    fn = basis_s(nbas)
    x, y, z = _f32(x), _f32(y), _f32(z)
    pwHL = np.float32(lacunarity ** -H)
    pwr = np.float32(1.0)
    value = np.ones(x.shape, np.float32)
    for _ in range(int(octaves)):
        value = value * (fn(x, y, z) * pwr + 1.0)
        pwr *= pwHL
        x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
    rmd = octaves - np.floor(octaves)
    if rmd != 0.0:
        value = value * (np.float32(rmd) * fn(x, y, z) * pwr + 1.0)
    return value


def mg_hetero_terrain(x, y, z, H, lacunarity, octaves, offset, nbas):
    fn = basis_s(nbas)
    x, y, z = _f32(x), _f32(y), _f32(z)
    pwHL = np.float32(lacunarity ** -H)
    pwr = pwHL
    value = np.float32(offset) + fn(x, y, z)
    x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
    for _ in range(1, int(octaves)):
        increment = (fn(x, y, z) + offset) * pwr * value
        value = value + increment
        pwr *= pwHL
        x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
    rmd = octaves - np.floor(octaves)
    if rmd != 0.0:
        increment = (fn(x, y, z) + offset) * pwr * value
        value = value + np.float32(rmd) * increment
    return value


def mg_hybrid_multifractal(x, y, z, H, lacunarity, octaves, offset, gain,
                           nbas):
    fn = basis_s(nbas)
    x, y, z = _f32(x), _f32(y), _f32(z)
    pwHL = np.float32(lacunarity ** -H)
    pwr = pwHL
    result = fn(x, y, z) + offset
    weight = np.float32(gain) * result
    x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
    for _ in range(1, int(octaves)):
        weight = np.minimum(weight, 1.0)
        signal = (fn(x, y, z) + offset) * pwr
        pwr *= pwHL
        result = result + weight * signal
        weight = weight * np.float32(gain) * signal
        x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
    rmd = octaves - np.floor(octaves)
    if rmd != 0.0:
        result = result + np.float32(rmd) * (fn(x, y, z) + offset) * pwr
    return result


def mg_ridged_multifractal(x, y, z, H, lacunarity, octaves, offset, gain,
                           nbas):
    fn = basis_s(nbas)
    x, y, z = _f32(x), _f32(y), _f32(z)
    pwHL = np.float32(lacunarity ** -H)
    pwr = pwHL
    signal = (np.float32(offset) - np.abs(fn(x, y, z))) ** 2
    result = signal.copy()
    weight = np.ones(x.shape, np.float32)
    for _ in range(1, int(octaves)):
        x, y, z = x * lacunarity, y * lacunarity, z * lacunarity
        weight = np.clip(signal * gain, 0.0, 1.0)
        signal = (np.float32(offset) - np.abs(fn(x, y, z))) ** 2 * weight
        result = result + signal * pwr
        pwr *= pwHL
    return result


def mg_vlnoise(x, y, z, distortion, nbas1, nbas2):
    """mg_VLNoise(): domain-distorted noise. nbas1 distorts, nbas2 is
    the source, offset by the +13.5 shifts noise.c uses to decorrelate
    the three distortion channels."""
    x, y, z = _f32(x), _f32(y), _f32(z)
    d = basis_s(nbas1)
    rx = d(x + 13.5, y + 13.5, z + 13.5) * distortion
    ry = d(x, y, z) * distortion
    rz = d(x - 13.5, y - 13.5, z - 13.5) * distortion
    return basis_s(nbas2)(x + rx, y + ry, z + rz)


# ------------------------------------------------------------ wave shapes


def _tex_sin(a):
    return (0.5 + 0.5 * np.sin(a)).astype(np.float32)


def _tex_saw(a):
    b = 2 * np.pi
    a = np.mod(a, b)
    a = np.where(a < 0, a + b, a)
    return (a / b).astype(np.float32)


def _tex_tri(a):
    b = 2 * np.pi
    rmax = 1.0
    return (rmax - 2.0 * np.abs(np.floor(a * (1.0 / b) + 0.5)
                                - a * (1.0 / b))).astype(np.float32)


_WAVE = {WAVE_SIN: _tex_sin, WAVE_SAW: _tex_saw, WAVE_TRI: _tex_tri}


# --------------------------------------------------------------- colorband


def _rgb_to_hsv_np(rgb):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, 1e-20), 0.0)
    dz = np.maximum(d, 1e-20)
    h = np.where(mx == r, (g - b) / dz % 6.0,
                 np.where(mx == g, (b - r) / dz + 2.0, (r - g) / dz + 4.0))
    h = np.where(d == 0, 0.0, h / 6.0)
    return np.stack([h, s, mx], axis=-1)


def _hsv_to_rgb_np(hsv):
    h, s, v = hsv[..., 0] * 6.0, hsv[..., 1], hsv[..., 2]
    i = np.floor(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    i = i.astype(np.int64) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)


#: MTex.blendtype order (DNA): the value-blend switch below follows it
MTEX_BLEND_ORDER = ('MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                    'LIGHT', 'SCREEN', 'OVERLAY', 'HUE', 'SAT', 'VAL',
                    'COLOR', 'SOFT', 'LINEAR')


def texture_value_blend(tex, out, fact, facg, blendtype):
    """2.79's texture_value_blend(), transcribed and vectorized.

    THE detail the first conversion missed: a BI value channel does not
    push the texture into the slider -- it blends the slider toward
    `tex` = the slot's DVar, by `fact` = the texture intensity times
    `facg` = the influence factor. A negative factor swaps the blend
    weights (the C's flip). HUE..COLOR are meaningless for scalars and
    return 0, exactly as the C's untouched `in = 0.0` does.
    """
    tex = np.float32(tex)
    out = _f32(out)
    fact = _f32(fact)
    facg = np.float32(facg)
    flip = bool(facg < 0.0)
    facg = np.float32(abs(float(facg)))
    fact = fact * facg
    facm = 1.0 - fact
    if flip:
        fact, facm = facm, fact
    if blendtype == 'MIX':
        return (fact * tex + facm * out).astype(np.float32)
    if blendtype == 'MUL':
        facm = 1.0 - facg
        return ((facm + fact * tex) * out).astype(np.float32)
    if blendtype == 'SCREEN':
        facm = 1.0 - facg
        return (1.0 - (facm + fact * (1.0 - tex))
                * (1.0 - out)).astype(np.float32)
    if blendtype == 'SUB':
        return (-fact * tex + out).astype(np.float32)
    if blendtype == 'ADD':
        return (fact * tex + out).astype(np.float32)
    if blendtype == 'DIV':
        if float(tex) == 0.0:
            return np.zeros_like(out, np.float32)
        return (facm * out + fact * out / tex).astype(np.float32)
    if blendtype == 'DIFF':
        return (facm * out + fact * np.abs(tex - out)).astype(np.float32)
    if blendtype == 'DARK':
        # the C mixes TOWARD the per-channel min -- verified verbatim
        # (R155): in = min_ff(out, tex)*fact + out*facm. The first
        # transcription had min(fact*tex, out), which is the LIGHT
        # case's shape mirrored, not what ships in 2.79.
        return (np.minimum(out, tex) * fact + out * facm).astype(
            np.float32)
    if blendtype == 'LIGHT':
        col = fact * tex
        return np.where(col > out, col, out).astype(np.float32)
    if blendtype == 'OVERLAY':
        facm = 1.0 - facg
        low = out * (facm + 2.0 * fact * tex)
        high = 1.0 - (facm + 2.0 * fact * (1.0 - tex)) * (1.0 - out)
        return np.where(out < 0.5, low, high).astype(np.float32)
    if blendtype == 'SOFT':
        # verbatim: in = facm*out + fact*((1-out)*tex*out) + (out*scf)
        # -- the LAST term is NOT scaled by fact (2.79 ships it so;
        # ramp_blend's SOFT differs, and the two are not the same fn)
        scf = 1.0 - (1.0 - tex) * (1.0 - out)
        return (facm * out + fact * ((1.0 - out) * tex * out)
                + (out * scf)).astype(np.float32)
    if blendtype == 'LINEAR':
        return np.where(tex > 0.5,
                        out + fact * (2.0 * (tex - 0.5)),
                        out + fact * (2.0 * tex - 1.0)).astype(np.float32)
    # HUE / SAT / VAL / COLOR: the C leaves `in` at 0.0
    return np.zeros_like(out, np.float32)


#: IMB_colormanagement_get_luminance with the default OCIO config:
#: Rec.709 weights. This is the collapse RGBToIntensity and every
#: rgb-texture-on-a-value-channel runs through in do_material_tex.
LUM_R, LUM_G, LUM_B = 0.2126, 0.7152, 0.0722


def rec709_lum(rgb):
    rgb = np.asarray(rgb, np.float32)
    return (LUM_R * rgb[..., 0] + LUM_G * rgb[..., 1]
            + LUM_B * rgb[..., 2]).astype(np.float32)


def texture_rgb_blend(tex, out, fact, facg, blendtype):
    """2.79's texture_rgb_blend(), transcribed per channel.

    `texture_value_blend`'s colour sibling, from the same switch in
    render_texture.c: `tex` is tcol -- the texture's own RGB when it
    yields one, the SLOT colour when it does not -- `out` the channel
    being driven, `fact` the per-pixel intensity (texres.tin: band or
    image alpha, or the intensity itself), `facg` the influence slider
    (never negative here: the colour sliders are 0..1, so the value
    twin's flip does not exist in the C). Differences from the value
    twin, kept deliberately: DIV on a zero channel KEEPS the base
    (`in` aliases `out` at every call site, so an unwritten channel
    is the old value -- the value fn returns its local 0.0 instead),
    and HUE/SAT/VAL/COLOR plus SOFT/LINEAR delegate to ramp_blend on
    a copy of the base rather than being zeroed.
    `tex` (N,3)/(3,), `out` (N,3), `fact` (N,) or scalar; returns (N,3).
    """
    from . import shading as _SH
    out = np.asarray(out, np.float32)
    if out.ndim == 1:
        out = out[None, :]
    n = out.shape[0]
    tex = np.asarray(tex, np.float32)
    if tex.ndim == 1:
        tex = np.broadcast_to(tex[None, :], (n, 3))
    fact = _f32(fact)
    if fact.shape == ():
        fact = np.full(n, float(fact), np.float32)
    facg = np.float32(facg)
    fact = (fact * facg).astype(np.float32)[:, None]
    facm = 1.0 - fact
    if blendtype == 'MIX':
        return (fact * tex + facm * out).astype(np.float32)
    if blendtype == 'MUL':
        # verbatim (R155): the rgb fn's MUL/SCREEN/OVERLAY use
        # facm = 1 - fact (the MULTIPLIED factor) -- unlike the value
        # twin, whose same cases rederive facm = 1 - facg. The 1.35.30
        # first cut assumed the twins matched; the fetched source says
        # otherwise, so a black-slot-colour MUL keeps (1-Tin)*base
        # instead of zeroing.
        return ((facm + fact * tex) * out).astype(np.float32)
    if blendtype == 'SCREEN':
        return (1.0 - (facm + fact * (1.0 - tex))
                * (1.0 - out)).astype(np.float32)
    if blendtype == 'SUB':
        return (-fact * tex + out).astype(np.float32)
    if blendtype == 'ADD':
        return (fact * tex + out).astype(np.float32)
    if blendtype == 'DIV':
        safe = np.where(tex != 0.0, tex, 1.0)
        blended = facm * out + fact * out / safe
        return np.where(tex != 0.0, blended, out).astype(np.float32)
    if blendtype == 'DIFF':
        return (facm * out + fact * np.abs(tex - out)).astype(np.float32)
    if blendtype == 'DARK':
        # verbatim: min_ff(out,tex)*fact + out*facm -- mix toward min
        return (np.minimum(out, tex) * fact + out * facm).astype(
            np.float32)
    if blendtype == 'LIGHT':
        col = fact * tex
        return np.where(col > out, col, out).astype(np.float32)
    if blendtype == 'OVERLAY':
        low = out * (facm + 2.0 * fact * tex)
        high = 1.0 - (facm + 2.0 * fact * (1.0 - tex)) * (1.0 - out)
        return np.where(out < 0.5, low, high).astype(np.float32)
    if blendtype in ('HUE', 'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR'):
        # the C copies `out` into `in` and calls ramp_blend with the
        # multiplied factor -- the same ramp_blend the material ramps
        # use, already transcribed as shading.bi_ramp_blend
        return _SH.bi_ramp_blend(blendtype, out.copy(),
                                 fact[:, 0], tex).astype(np.float32)
    return out.astype(np.float32)


def colorband_eval(stops, tin, ipotype=0):
    """do_colorband(): stops [(pos, r, g, b, a), ...] position-sorted,
    ipotype 0 linear / 1 ease / 2 b-spline / 3 cardinal / 4 constant.

    Follows the C exactly: the scan finds the first stop with pos > in
    (cbd1); the one before it is cbd2; running off either end clones
    the edge stop as a virtual one at pos 0 or 1. fac runs from cbd1
    toward cbd2, and the spline modes weight cbd3..cbd0 with the key
    basis functions.
    """
    tin = _f32(tin)
    n = len(stops)
    out = np.zeros(tin.shape + (4,), np.float32)
    if n == 0:
        return out
    arr = np.asarray([[s[0], s[1], s[2], s[3], s[4]] for s in stops],
                     np.float32)
    if n == 1:
        out[:] = arr[0, 1:5]
        return out
    pos = arr[:, 0]
    a = np.searchsorted(pos, tin, side='right')   # first pos > in
    below = a == 0
    above = a == n
    i_right = np.clip(a, 0, n - 1)                # cbd1 (virtual at end)
    i_left = np.clip(a - 1, 0, n - 1)             # cbd2 (virtual at start)
    rpos = np.where(above, 1.0, pos[i_right])
    lpos = np.where(below, 0.0, pos[i_left])
    rcol = arr[i_right][:, 1:5]
    lcol = arr[i_left][:, 1:5]
    span = lpos - rpos                            # BI measures from cbd1
    fac = np.where(np.abs(span) > 1e-9,
                   (tin - rpos) / np.where(np.abs(span) > 1e-9, span, 1.0),
                   np.where(above, 1.0, 0.0))
    if ipotype == 4:                  # constant: hold cbd2 (the left)
        out[:] = lcol
        return out
    if ipotype in (2, 3):             # b-spline / cardinal
        i0 = np.clip(np.where(a >= n - 1, i_right, i_right + 1), 0, n - 1)
        i3 = np.clip(np.where(a < 2, i_left, i_left - 1), 0, n - 1)
        c0, c1, c2, c3 = arr[i0][:, 1:5], rcol, lcol, arr[i3][:, 1:5]
        t = np.clip(fac, 0.0, 1.0)[:, None]
        t2, t3 = t * t, t * t * t
        if ipotype == 3:              # KEY_CARDINAL basis
            w0 = -0.5 * t3 + t2 - 0.5 * t
            w1 = 1.5 * t3 - 2.5 * t2 + 1.0
            w2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            w3 = 0.5 * t3 - 0.5 * t2
        else:                         # KEY_BSPLINE basis
            w0 = -0.16666666 * t3 + 0.5 * t2 - 0.5 * t + 0.16666666
            w1 = 0.5 * t3 - t2 + 0.66666666
            w2 = -0.5 * t3 + 0.5 * t2 + 0.5 * t + 0.16666666
            w3 = 0.16666666 * t3
        res = w3 * c3 + w2 * c2 + w1 * c1 + w0 * c0
        return np.clip(res, 0.0, 1.0).astype(np.float32)
    if ipotype == 1:                  # ease
        # linear/ease hold the edge stop outside the range
        fac = np.clip(fac, 0.0, 1.0)
        fac = 3.0 * fac * fac - 2.0 * fac * fac * fac
    else:
        fac = np.clip(fac, 0.0, 1.0)
    out = (1.0 - fac[:, None]) * rcol + fac[:, None] * lcol
    return out.astype(np.float32)


# ---------------------------------------------------------- texture types
#
# Each returns (tin, rgb-or-None); the caller applies colorband and
# BRICONT(RGB), exactly as multitex() sequences it.


def _clouds(p, x, y, z):
    hard = p['noisetype'] != 0
    tin = gturbulence(p['noisesize'], x, y, z, p['noisedepth'], hard,
                      p['noisebasis'])
    if p['stype'] == STY_CLOUDS_COLOR:
        tg = gturbulence(p['noisesize'], y, x, z, p['noisedepth'], hard,
                         p['noisebasis'])
        tb = gturbulence(p['noisesize'], y, z, x, p['noisedepth'], hard,
                         p['noisebasis'])
        return tin, np.stack([tin, tg, tb], axis=-1)
    return tin, None


def _wood(p, x, y, z):
    wf = _WAVE.get(p['noisebasis2'], _tex_sin)
    hard = p['noisetype'] != 0
    st = p['stype']
    if st == STY_WOOD_BAND:
        tin = wf((x + y + z) * 10.0)
    elif st == STY_WOOD_RING:
        tin = wf(np.sqrt(x * x + y * y + z * z) * 20.0)
    elif st == STY_WOOD_BANDNOISE:
        wi = p['turbul'] * gnoise(p['noisesize'], x, y, z, hard,
                                  p['noisebasis'])
        tin = wf((x + y + z) * 10.0 + wi)
    else:
        wi = p['turbul'] * gnoise(p['noisesize'], x, y, z, hard,
                                  p['noisebasis'])
        tin = wf(np.sqrt(x * x + y * y + z * z) * 20.0 + wi)
    return tin, None


def _marble(p, x, y, z):
    wf = _WAVE.get(p['noisebasis2'], _tex_sin)
    hard = p['noisetype'] != 0
    n = 5.0 * (x + y + z)
    mi = n + p['turbul'] * gturbulence(p['noisesize'], x, y, z,
                                       p['noisedepth'], hard,
                                       p['noisebasis'])
    mi = wf(mi)
    if p['stype'] == STY_MARBLE_SHARP:
        mi = np.sqrt(mi)
    elif p['stype'] == STY_MARBLE_SHARPER:
        mi = np.sqrt(np.sqrt(mi))
    return mi.astype(np.float32), None


def _magic(p, x, y, z):
    n = int(p['noisedepth'])
    turb = np.float32(p['turbul'] / 5.0)
    mx = np.sin((x + y + z) * 5.0).astype(np.float32)
    my = np.cos((-x + y - z) * 5.0).astype(np.float32)
    mz = -np.cos((-x - y + z) * 5.0).astype(np.float32)
    if n > 0:
        mx, my, mz = mx * turb, my * turb, mz * turb
        my = -np.cos(mx - my + mz) * turb
        if n > 1:
            mx = np.cos(mx - my - mz) * turb
            if n > 2:
                mz = np.sin(-mx - my - mz) * turb
                if n > 3:
                    mx = -np.cos(-mx + my - mz) * turb
                    if n > 4:
                        my = -np.sin(-mx + my + mz) * turb
                        if n > 5:
                            my = -np.cos(-mx + my + mz) * turb
                            if n > 6:
                                mx = np.cos(mx + my + mz) * turb
                                if n > 7:
                                    mz = np.sin(mx + my - mz) * turb
                                    if n > 8:
                                        mx = -np.cos(-mx - my + mz) * turb
                                        if n > 9:
                                            my = -np.sin(mx - my + mz) \
                                                * turb
    if float(turb) != 0.0:
        t2 = turb * 2.0
        mx, my, mz = mx / t2, my / t2, mz / t2
    rgb = np.stack([0.5 - mx, 0.5 - my, 0.5 - mz], axis=-1)
    tin = rgb.mean(axis=-1).astype(np.float32)
    return tin, rgb.astype(np.float32)


def _blend(p, x, y, z):
    if p.get('flag', 0) & TEX_FLIPBLEND:
        x, y = y, x
    st = p['stype']
    if st == STY_BLEND_LIN:
        tin = (1.0 + x) / 2.0
    elif st == STY_BLEND_QUAD:
        tin = (1.0 + x) / 2.0
        tin = np.where(tin < 0, 0.0, tin * tin)
    elif st == STY_BLEND_EASE:
        tin = np.clip((1.0 + x) / 2.0, 0.0, 1.0)
        t2 = tin * tin
        tin = 3.0 * t2 - 2.0 * t2 * tin
    elif st == STY_BLEND_DIAG:
        tin = (2.0 + x + y) / 4.0
    elif st == STY_BLEND_RAD:
        tin = np.arctan2(y, x) / (2 * np.pi) + 0.5
    else:
        tin = np.maximum(1.0 - np.sqrt(x * x + y * y + z * z), 0.0)
        if st == STY_BLEND_HALO:
            tin = tin * tin
    return tin.astype(np.float32), None


def _stucci(p, x, y, z):
    hard = p['noisetype'] != 0
    b2 = gnoise(p['noisesize'], x, y, z, hard, p['noisebasis'])
    ofs = p['turbul'] / 200.0
    if p['stype'] != STY_STUCCI_PLASTIC:
        ofs = ofs * b2 * b2                      # wall in / wall out
    tin = gnoise(p['noisesize'], x, y, z + ofs, hard, p['noisebasis'])
    if p['stype'] == STY_STUCCI_WALLOUT:
        tin = 1.0 - tin
    return np.maximum(tin, 0.0).astype(np.float32), None


def _texnoise(p, x, y, z, frame=0):
    """BI's static reseeded per eval from a thread RNG; Halcyon hashes
    the coordinates and frame instead, so renders stay reproducible."""
    xi = np.floor(x * 1e4).astype(np.int64)
    yi = np.floor(y * 1e4).astype(np.int64)
    n = (xi + yi * 1301 + (np.int64(frame) + 7) * 314159).astype(np.uint32)
    n = n ^ (n << np.uint32(13))
    ran = (n * (n * n * np.uint32(15731) + np.uint32(789221))
           + np.uint32(1376312589))
    div = np.float64(3.0)
    shift = np.uint32(29)
    val = ((ran >> shift) & np.uint32(3)).astype(np.float64)
    loop = int(p['noisedepth'])
    for _ in range(loop):
        shift = shift - np.uint32(2)
        val = val * ((ran >> shift) & np.uint32(3)).astype(np.float64)
        div *= 3.0
    return (val / div).astype(np.float32), None


def _musgrave(p, x, y, z):
    st = p['stype']
    H, lac, octs = p['mg_H'], p['mg_lacunarity'], p['mg_octaves']
    nbas = p['noisebasis']
    if st in (STY_MG_MFRACTAL, STY_MG_FBM):
        fn = mg_multifractal if st == STY_MG_MFRACTAL else mg_fbm
        tin = p['ns_outscale'] * fn(x, y, z, H, lac, octs, nbas)
    elif st in (STY_MG_RIDGED, STY_MG_HYBRID):
        fn = mg_ridged_multifractal if st == STY_MG_RIDGED \
            else mg_hybrid_multifractal
        tin = p['ns_outscale'] * fn(x, y, z, H, lac, octs,
                                    p['mg_offset'], p['mg_gain'], nbas)
    else:
        tin = p['ns_outscale'] * mg_hetero_terrain(x, y, z, H, lac, octs,
                                                   p['mg_offset'], nbas)
    return tin.astype(np.float32), None


def _voronoi_tex(p, x, y, z):
    w1, w2 = p['vn_w1'], p['vn_w2']
    w3, w4 = p['vn_w3'], p['vn_w4']
    aw1, aw2, aw3, aw4 = abs(w1), abs(w2), abs(w3), abs(w4)
    sc = aw1 + aw2 + aw3 + aw4
    if sc != 0.0:
        sc = p['ns_outscale'] / sc
    da, pa = voronoi(x, y, z, p['vn_mexp'], p['vn_distm'])
    tin = sc * np.abs(w1 * da[:, 0] + w2 * da[:, 1] + w3 * da[:, 2]
                      + w4 * da[:, 3])
    rgb = None
    if p['vn_coltype']:
        rgb = aw1 * cellnoise_v3(pa[:, 0], pa[:, 1], pa[:, 2])
        rgb = rgb + aw2 * cellnoise_v3(pa[:, 3], pa[:, 4], pa[:, 5])
        rgb = rgb + aw3 * cellnoise_v3(pa[:, 6], pa[:, 7], pa[:, 8])
        rgb = rgb + aw4 * cellnoise_v3(pa[:, 9], pa[:, 10], pa[:, 11])
        if p['vn_coltype'] >= 2:
            t1 = np.minimum((da[:, 1] - da[:, 0]) * 10.0, 1.0)
            if p['vn_coltype'] == 3:
                t1 = t1 * tin
            else:
                t1 = t1 * sc
            rgb = rgb * t1[:, None]
        else:
            rgb = rgb * sc
    return tin.astype(np.float32), \
        None if rgb is None else rgb.astype(np.float32)


def _distnoise(p, x, y, z):
    tin = mg_vlnoise(x, y, z, p['dist_amount'], p['noisebasis'],
                     p['noisebasis2'])
    return tin.astype(np.float32), None


_TYPES = {TEX_CLOUDS: _clouds, TEX_WOOD: _wood, TEX_MARBLE: _marble,
          TEX_MAGIC: _magic, TEX_BLEND: _blend, TEX_STUCCI: _stucci,
          TEX_NOISE: _texnoise, TEX_MUSGRAVE: _musgrave,
          TEX_VORONOI: _voronoi_tex, TEX_DISTNOISE: _distnoise}

#: everything a Tex carries that evaluate() reads, with BI's defaults
DEFAULTS = {
    'type': TEX_CLOUDS, 'stype': 0, 'noisebasis': 0, 'noisebasis2': 0,
    'noisetype': 0, 'noisesize': 0.25, 'noisedepth': 2, 'turbul': 5.0,
    'bright': 1.0, 'contrast': 1.0, 'saturation': 1.0,
    'rfac': 1.0, 'gfac': 1.0, 'bfac': 1.0, 'flag': 0,
    'mg_H': 1.0, 'mg_lacunarity': 2.0, 'mg_octaves': 2.0,
    'mg_offset': 1.0, 'mg_gain': 1.0, 'ns_outscale': 1.0,
    'vn_w1': 1.0, 'vn_w2': 0.0, 'vn_w3': 0.0, 'vn_w4': 0.0,
    'vn_mexp': 2.5, 'vn_distm': 0, 'vn_coltype': 0,
    'dist_amount': 1.0,
    'coba': None, 'coba_ipotype': 0,
}


def _evaluate_chunk(params, texvec, frame=0):
    """multitex(): evaluate one BI texture at (N,3) texture-space
    coordinates (the classic -1..1 space, already offset and sized).

    Returns (tin (N,), rgba (N,4) or None). The colorband and
    brightness/contrast/saturation pipeline runs exactly in BI's
    order: type formula -> colorband -> BRICONT(RGB).
    """
    p = dict(DEFAULTS)
    p.update({k: v for k, v in params.items() if v is not None})
    texvec = np.asarray(texvec, np.float32)
    x, y, z = texvec[:, 0], texvec[:, 1], texvec[:, 2]

    tt = int(p['type'])
    fn = _TYPES.get(tt)
    if fn is None:
        return np.zeros(x.shape, np.float32), None
    if tt in (TEX_MUSGRAVE, TEX_VORONOI, TEX_DISTNOISE):
        # multitex() pre-scales ONLY these three by 1/noisesize
        ns = p['noisesize'] if p['noisesize'] != 0 else 1e-6
        inv = np.float32(1.0 / ns)
        x, y, z = x * inv, y * inv, z * inv
    if tt == TEX_NOISE:
        tin, rgb = fn(p, x, y, z, frame)
    else:
        tin, rgb = fn(p, x, y, z)

    if (p['flag'] & TEX_COLORBAND) and p['coba']:
        rgba = colorband_eval(p['coba'], tin, int(p['coba_ipotype']))
        rgb = rgba[:, :3]
        alpha = rgba[:, 3]
    else:
        alpha = None

    no_clamp = bool(p['flag'] & TEX_NO_CLAMP)
    if rgb is not None:
        # BRICONTRGB
        con, bri = np.float32(p['contrast']), np.float32(p['bright'])
        out = np.empty(rgb.shape, np.float32)
        for i, fac in enumerate((p['rfac'], p['gfac'], p['bfac'])):
            out[:, i] = fac * ((rgb[:, i] - 0.5) * con + bri - 0.5)
        if not no_clamp:
            out = np.maximum(out, 0.0)
        if p['saturation'] != 1.0:
            hsv = _rgb_to_hsv_np(out)
            hsv[..., 1] *= np.float32(p['saturation'])
            out = _hsv_to_rgb_np(hsv).astype(np.float32)
            if p['saturation'] > 1.0 and not no_clamp:
                out = np.maximum(out, 0.0)
        a = alpha if alpha is not None else np.ones(tin.shape, np.float32)
        rgba = np.concatenate([out, a[:, None]], axis=1)
        # tin follows the colorband/rgb result the way BI reports it
        tin_out = np.clip(tin, 0.0, 1.0) if not no_clamp \
            else tin.astype(np.float32)
        return tin_out, rgba.astype(np.float32)

    # BRICONT (intensity only)
    tin = (tin - 0.5) * np.float32(p['contrast']) \
        + np.float32(p['bright']) - 0.5
    if not no_clamp:
        tin = np.clip(tin, 0.0, 1.0)
    return tin.astype(np.float32), None


#: lanes per evaluation slice: bounds voronoi's (N,27,3) intermediates to
#: ~20 MB instead of letting a 2-million-pixel pre-pass allocate gigabytes
_CHUNK = 1 << 16


def evaluate(params, texvec, frame=0):
    """multitex() over any number of points, sliced to bound memory.

    The voronoi neighbourhood alone is a 27x blow-up of the input; at
    full render resolution that is gigabytes if taken in one bite, and a
    machine that starts swapping mid-render looks exactly like a crash.
    Same numbers, bounded peak.
    """
    texvec = np.asarray(texvec, np.float32)
    n = texvec.shape[0]
    if n <= _CHUNK:
        return _evaluate_chunk(params, texvec, frame)
    tins, rgbas = [], []
    for s in range(0, n, _CHUNK):
        t, r = _evaluate_chunk(params, texvec[s:s + _CHUNK], frame)
        tins.append(t)
        rgbas.append(r)
    tin = np.concatenate(tins)
    if rgbas[0] is None:
        return tin, None
    return tin, np.concatenate(rgbas)


def classic_texvec(vec, ofs, size, classic_space=True):
    """The mtex coordinate pipeline: modern 0..1 coordinates into the
    classic -1..1 texture space, then texvec = size*(co + ofs) -- the
    exact order do_material_tex applies (offset BEFORE size)."""
    v = np.asarray(vec, np.float32)
    if classic_space:
        v = v * 2.0 - 1.0
    return (np.asarray(size, np.float32)[None, :]
            * (v + np.asarray(ofs, np.float32)[None, :])).astype(np.float32)


# ------------------------------------------------- node enum translations
#
# The node stores UI enum identifiers; the engine evaluates BI codes.
# Kept here (bpy-free) so nodeeval and the GPU emitter share one map.

ENUM_TYPE = {'CLOUDS': TEX_CLOUDS, 'WOOD': TEX_WOOD, 'MARBLE': TEX_MARBLE,
             'MAGIC': TEX_MAGIC, 'BLEND': TEX_BLEND, 'STUCCI': TEX_STUCCI,
             'NOISE': TEX_NOISE, 'MUSGRAVE': TEX_MUSGRAVE,
             'VORONOI': TEX_VORONOI, 'DISTNOISE': TEX_DISTNOISE}
ENUM_BASIS = {'BLENDER_ORIGINAL': 0, 'ORIGINAL_PERLIN': 1,
              'IMPROVED_PERLIN': 2, 'VORONOI_F1': 3, 'VORONOI_F2': 4,
              'VORONOI_F3': 5, 'VORONOI_F4': 6, 'VORONOI_F2F1': 7,
              'VORONOI_CRACKLE': 8, 'CELL_NOISE': 14}
ENUM_WAVE = {'SIN': 0, 'SAW': 1, 'TRI': 2}
ENUM_WOOD = {'BAND': 0, 'RING': 1, 'BANDNOISE': 2, 'RINGNOISE': 3}
ENUM_MARBLE = {'SOFT': 0, 'SHARP': 1, 'SHARPER': 2}
ENUM_BLEND = {'LIN': 0, 'QUAD': 1, 'EASE': 2, 'DIAG': 3, 'SPHERE': 4,
              'HALO': 5, 'RAD': 6}
ENUM_STUCCI = {'PLASTIC': 0, 'WALLIN': 1, 'WALLOUT': 2}
ENUM_MUSGRAVE = {'MFRACTAL': 0, 'RIDGEDMF': 1, 'HYBRIDMF': 2, 'FBM': 3,
                 'HTERRAIN': 4}
ENUM_DISTM = {'DISTANCE': 0, 'DISTANCE_SQUARED': 1, 'MANHATTAN': 2,
              'CHEBYCHEV': 3, 'MINKOVSKY_HALF': 4, 'MINKOVSKY_FOUR': 5,
              'MINKOVSKY': 6}
ENUM_COLTYPE = {'INTENSITY': 0, 'POSITION': 1, 'POSITION_OUTLINE': 2,
                'POSITION_OUTLINE_INTENSITY': 3}
ENUM_CB_IPO = {'LINEAR': 0, 'EASE': 1, 'B_SPLINE': 2, 'CARDINAL': 3,
               'CONSTANT': 4}


def node_params(props):
    """A serialized HALCYON_BITextureNode props dict -> evaluate() params.

    Accepts enum identifier strings or raw integer codes, so both the
    live node and the exported form feed the same evaluator.
    """
    def code(table, key, default=0):
        v = props.get(key)
        if isinstance(v, str):
            return table.get(v, default)
        return int(v) if v is not None else default

    def g(key, default):
        # a prop can arrive as None (a graph serialized before the prop
        # existed, or a fake node in the suite) -- None means "use the
        # node's default", exactly what a real bpy node would report
        v = props.get(key)
        return default if v is None else v

    tt = code(ENUM_TYPE, 'tex_type', TEX_CLOUDS)
    stype = 0
    if tt == TEX_CLOUDS:
        stype = 1 if props.get('clouds_color') else 0
    elif tt == TEX_WOOD:
        stype = code(ENUM_WOOD, 'wood_type')
    elif tt == TEX_MARBLE:
        stype = code(ENUM_MARBLE, 'marble_type')
    elif tt == TEX_BLEND:
        stype = code(ENUM_BLEND, 'blend_type')
    elif tt == TEX_STUCCI:
        stype = code(ENUM_STUCCI, 'stucci_type')
    elif tt == TEX_MUSGRAVE:
        stype = code(ENUM_MUSGRAVE, 'musgrave_type', 3)
    rgbf = props.get('rgb_factors') or (1.0, 1.0, 1.0)
    flag = 0
    if tt == TEX_BLEND and props.get('blend_flip'):
        flag |= TEX_FLIPBLEND
    if not props.get('use_clamp', True):
        flag |= TEX_NO_CLAMP
    if props.get('use_colorband') and props.get('coba'):
        flag |= TEX_COLORBAND
    return {
        'type': tt, 'stype': stype,
        'noisebasis': code(ENUM_BASIS, 'noise_basis'),
        'noisebasis2': (code(ENUM_BASIS, 'noise_basis2')
                        if tt == TEX_DISTNOISE
                        else code(ENUM_WAVE, 'wave')),
        'noisetype': 1 if props.get('hard_noise') else 0,
        'noisesize': float(g('noise_size', 0.25)),
        'noisedepth': int(g('noise_depth', 2)),
        'turbul': float(g('turbulence', 5.0)),
        'bright': float(g('bright', 1.0)),
        'contrast': float(g('contrast', 1.0)),
        'saturation': float(g('saturation', 1.0)),
        'rfac': float(rgbf[0]), 'gfac': float(rgbf[1]),
        'bfac': float(rgbf[2]), 'flag': flag,
        'mg_H': float(g('mg_h', 1.0)),
        'mg_lacunarity': float(g('mg_lacunarity', 2.0)),
        'mg_octaves': float(g('mg_octaves', 2.0)),
        'mg_offset': float(g('mg_offset', 1.0)),
        'mg_gain': float(g('mg_gain', 1.0)),
        'ns_outscale': float(g('ns_outscale', 1.0)),
        'vn_w1': float(g('vn_w1', 1.0)),
        'vn_w2': float(g('vn_w2', 0.0)),
        'vn_w3': float(g('vn_w3', 0.0)),
        'vn_w4': float(g('vn_w4', 0.0)),
        'vn_mexp': float(g('vn_mexp', 2.5)),
        'vn_distm': code(ENUM_DISTM, 'vn_distm'),
        'vn_coltype': code(ENUM_COLTYPE, 'vn_coltype'),
        'dist_amount': float(g('dist_amount', 1.0)),
        'coba': props.get('coba'),
        'coba_ipotype': code(ENUM_CB_IPO, 'coba_ipotype'),
    }
