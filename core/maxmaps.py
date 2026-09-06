"""3ds Max's map library, on Max's own algorithms (R243, the Max study's
second reading).

R242 built the shelf from the 2012 reference's CONTROLS; the field then
supplied the mental images MetaSL/HLSL ports of the maps that ship with
Max, and this module now follows those algorithms step for step: the
512-lattice gradient noise every 3D map rides (Perlin's 1989 noise as
max_texutil.cpp carries it, in 3D and in 4D with Phase as time), the
Worley variant behind Cellular (Poisson point counts per cell, SQUARED
distances, the fractal sum by lacunarity), the 20-periodic linear value
noise behind Dent, Planet and Wood, Smoke's velocity-shifted turbulence,
Speckle's six-octave sum, Splat's product of smoothsteps, Stucco's knee
curve, Swirl's twist-then-fBm, Waves' rand()-seeded wave sets, Checker's
integrated soften, Marble's seventeen-band veins, Perlin Marble's
thirteen-knot spline and the Gradient Ramp's shapes.

What is NOT Autodesk's: the random tables. Max reads its permutation,
gradient and random tables from data files; Halcyon derives every table
value from its own integer hash (patterns.hash3), so the maps have Max's
character and controls, on Halcyon's own lattice. No Autodesk data
ships here. The CPU reads the tables materialised once at import (a
gather is cheaper than a hash per lattice corner); the GPU twins in
gpu/procedural.py compute the same hash inline -- the same value
either way, which is what the twin tests pin.

Arithmetic discipline: every expression is written in the order and
the float32 the GLSL twin evaluates it in (a + t (b - a), never mix();
x - y floor(x / y) for mod), so the two sides agree to the bit where
the maths allows and to a few ulps where sin, sqrt and pow enter.

Conventions shared with patterns.py: `p` is (n, 3) float32 in the
node's coordinate space (already divided by the map's Size); scalar
controls are batch constants; every map returns a float32 field the
node ramps between its colours (the colour-rule maps return colours).
"""

import numpy as np

from . import patterns as PT

f32 = np.float32
i64 = np.int64
TWO_PI = f32(6.2831853)
PI = f32(3.14159265)
B = 512                      # the lattice, exactly Max's
RN = f32(10000.0)            # Max's coordinate offset before the lattice


def _mod(x, y):
    """GLSL mod(): x - y floor(x / y), in float32 -- the twin's formula,
    not numpy's exact fmod, so the two sides split a band identically."""
    x = np.asarray(x, f32)
    y = f32(y)
    return (x - y * np.floor(x / y)).astype(f32)


CHUNK = 16384


def _chunked(fn):
    """Run a map over its points 16k at a time: the octave loops keep
    their temporaries in cache, which halves the cost of every noise
    map at frame-sized batches (measured, not assumed). Results are
    concatenated, so the picture is the same to the bit."""
    def run(p, *args, **kw):
        p = np.asarray(p, f32)
        n = p.shape[0]
        if n <= CHUNK:
            return fn(p, *args, **kw)
        parts = [fn(p[i:i + CHUNK], *args, **kw) for i in range(0, n, CHUNK)]
        if isinstance(parts[0], tuple):
            return tuple(np.concatenate([q[j] for q in parts])
                         for j in range(len(parts[0])))
        return np.concatenate(parts)
    run.__name__ = fn.__name__
    run.__doc__ = fn.__doc__
    run.inner = fn
    return run


# ---------------------------------------------------------------- tables
# Halcyon's stand-ins for Max's data files: every entry a pure function
# of its index through the pattern library's integer hash.

def perm(i):
    """P[i]: Max's permutation table -- ours, hash-derived, 0..511."""
    i = np.asarray(i, i64)
    return (PT.hash3(i, i64(7), i64(13)) * f32(511.99)).astype(i64)


def _grad(i, k, salt):
    return (PT.hash3(np.asarray(i, i64), i64(31 + k), i64(salt)) * f32(2.0)
            - f32(1.0)).astype(f32)


def grad3(i):
    """G3[i]: a unit 3-vector per lattice index (Max's g3 table)."""
    g = np.stack([_grad(i, 0, 71), _grad(i, 1, 71), _grad(i, 2, 71)], -1)
    d = (g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2]).astype(f32)
    return (g / np.sqrt(d)[:, None]).astype(f32)


def grad4(i):
    """G4[i]: a unit 4-vector per lattice index (Max's g4 table)."""
    g = np.stack([_grad(i, 0, 73), _grad(i, 1, 73), _grad(i, 2, 73),
                  _grad(i, 3, 73)], -1)
    d = (g[:, 0] * g[:, 0] + g[:, 1] * g[:, 1] + g[:, 2] * g[:, 2]
         + g[:, 3] * g[:, 3]).astype(f32)
    return (g / np.sqrt(d)[:, None]).astype(f32)


def rand01(i):
    """Max's rand01 table: a uniform per integer index."""
    return PT.hash3(np.asarray(i, i64), i64(3), i64(5))


def rand02(i):
    """Max's rand02 table: a second uniform per integer index."""
    return PT.hash3(np.asarray(i, i64), i64(11), i64(17))


def noise1(i):
    """Max's 1D random table (Bricks' per-brick randoms): a uniform per
    integer index."""
    return PT.hash3(np.asarray(i, i64), i64(19), i64(23))


def dent_table(ix, iy, iz):
    """Max's 21x21x21 'dent' noise table (Dent, Planet, Wood): a uniform
    per lattice corner, periodic every 20 -- entry 20 IS entry 0."""
    ix = np.asarray(ix, i64) % 20
    iy = np.asarray(iy, i64) % 20
    iz = np.asarray(iz, i64) % 20
    return PT.hash3(ix + i64(101), iy + i64(203), iz + i64(307))


_IDX = np.arange(1024, dtype=i64)
#: P[0..1023]: noise indexes reach 511 + 511 (Max wraps its 512 table
#: with a second copy; ours is the hash, so the index runs on)
PERM_TABLE = perm(_IDX)
_G3 = grad3(_IDX)
G3X, G3Y, G3Z = (np.ascontiguousarray(_G3[:, k]) for k in range(3))
_G4 = grad4(_IDX)
G4X, G4Y, G4Z, G4W = (np.ascontiguousarray(_G4[:, k]) for k in range(4))
_Y = np.arange(10000, dtype=i64)
_K = np.arange(9, dtype=i64)

#: max_texutil's point-count probabilities (the Poisson CDF, mean 3)
_POINT_PROBS = (0.049787, 0.199148, 0.423190, 0.647232, 0.815263,
                0.916082, 0.966492, 0.988096, 0.996197)


def _point_count(u):
    ct = np.full(np.shape(u), 8, i64)
    for i in range(8, -1, -1):
        ct = np.where(u < f32(_POINT_PROBS[i]), i, ct)
    return ct


#: CellFunction's per-cell data, keyed by cell id mod 10000 exactly as
#: Max keys its 10000-entry random tables: the point count and up to
#: nine points (x from rand02(y + 3 + 3k), y from y + 2 + 3k, z from
#: y + 1 + 3k -- the port's cntr walk, z first). Points past the cell's
#: count sit at 1e16, so their squared distance (1e32) can never win:
#: the CPU walks all nine without a mask, the GPU breaks at the count,
#: and both land on the same nearest points.
CELL_COUNT = _point_count(rand01(_Y)).astype(i64)
_FAR = f32(1e16)
_ACTIVE = _K[:, None] < CELL_COUNT[None, :]                          # (9, 10000)
CELL_X = np.where(_ACTIVE, rand02(_Y[None, :] + 3 + 3 * _K[:, None]), _FAR).astype(f32)
CELL_Y = np.where(_ACTIVE, rand02(_Y[None, :] + 2 + 3 * _K[:, None]), _FAR).astype(f32)
CELL_Z = np.where(_ACTIVE, rand02(_Y[None, :] + 1 + 3 * _K[:, None]), _FAR).astype(f32)
#: [iz, iy, ix], 22 entries an axis: 20 and 21 repeat 0 and 1, so a
#: mod() that lands exactly on 20 still reads inside the table
_D = np.arange(22, dtype=i64)
DENT_TABLE = dent_table(_D[None, None, :], _D[None, :, None],
                        _D[:, None, None]).astype(f32)


# ------------------------------------------------------------- the noise

def _setup(v):
    """max_texutil's setup(): the lattice cell pair and the fractions."""
    t = (np.asarray(v, f32) + RN).astype(f32)
    it = t.astype(i64)
    b0 = it & (B - 1)
    b1 = (b0 + 1) & (B - 1)
    r0 = (t - it.astype(f32)).astype(f32)
    r1 = (r0 - f32(1.0)).astype(f32)
    return b0, b1, r0, r1


def _s_curve(t):
    return (t * t * (f32(3.0) - f32(2.0) * t)).astype(f32)


def _lerp(t, a, b):
    return (a + t * (b - a)).astype(f32)


def noise3(p):
    """Max's noise3: Perlin's gradient noise on the 512 lattice, in
    -1..1 (nominally). p is (n, 3)."""
    bx0, bx1, rx0, rx1 = _setup(p[:, 0])
    by0, by1, ry0, ry1 = _setup(p[:, 1])
    bz0, bz1, rz0, rz1 = _setup(p[:, 2])
    i = PERM_TABLE[bx0]
    j = PERM_TABLE[bx1]
    b00 = PERM_TABLE[i + by0]
    b10 = PERM_TABLE[j + by0]
    b01 = PERM_TABLE[i + by1]
    b11 = PERM_TABLE[j + by1]
    sx, sy, sz = _s_curve(rx0), _s_curve(ry0), _s_curve(rz0)

    def dot(idx, rx, ry, rz):
        return (rx * G3X[idx] + ry * G3Y[idx] + rz * G3Z[idx]).astype(f32)

    u = dot(b00 + bz0, rx0, ry0, rz0)
    v = dot(b10 + bz0, rx1, ry0, rz0)
    a = _lerp(sx, u, v)
    u = dot(b01 + bz0, rx0, ry1, rz0)
    v = dot(b11 + bz0, rx1, ry1, rz0)
    b = _lerp(sx, u, v)
    c = _lerp(sy, a, b)
    u = dot(b00 + bz1, rx0, ry0, rz1)
    v = dot(b10 + bz1, rx1, ry0, rz1)
    a = _lerp(sx, u, v)
    u = dot(b01 + bz1, rx0, ry1, rz1)
    v = dot(b11 + bz1, rx1, ry1, rz1)
    b = _lerp(sx, u, v)
    d = _lerp(sy, a, b)
    return _lerp(sz, c, d)


def noise4(p, w):
    """Max's noise4: the 4D lattice noise, w the time (Noise's Phase)."""
    bx0, bx1, rx0, rx1 = _setup(p[:, 0])
    by0, by1, ry0, ry1 = _setup(p[:, 1])
    bz0, bz1, rz0, rz1 = _setup(p[:, 2])
    bw0, bw1, rw0, rw1 = _setup(np.full(p.shape[0], f32(w), f32))
    i = PERM_TABLE[bx0]
    j = PERM_TABLE[bx1]
    b00 = PERM_TABLE[i + by0]
    b10 = PERM_TABLE[j + by0]
    b01 = PERM_TABLE[i + by1]
    b11 = PERM_TABLE[j + by1]
    c00 = PERM_TABLE[b00 + bz0]
    c10 = PERM_TABLE[b10 + bz0]
    c01 = PERM_TABLE[b01 + bz0]
    c11 = PERM_TABLE[b11 + bz0]
    d00 = PERM_TABLE[b00 + bz1]
    d10 = PERM_TABLE[b10 + bz1]
    d01 = PERM_TABLE[b01 + bz1]
    d11 = PERM_TABLE[b11 + bz1]
    sx, sy, sz, sw = _s_curve(rx0), _s_curve(ry0), _s_curve(rz0), _s_curve(rw0)

    def dot(idx, rx, ry, rz, rw):
        return (rx * G4X[idx] + ry * G4Y[idx] + rz * G4Z[idx]
                + rw * G4W[idx]).astype(f32)

    def octant(bw, rw):
        u = dot(c00 + bw, rx0, ry0, rz0, rw)
        v = dot(c10 + bw, rx1, ry0, rz0, rw)
        a = _lerp(sx, u, v)
        u = dot(c01 + bw, rx0, ry1, rz0, rw)
        v = dot(c11 + bw, rx1, ry1, rz0, rw)
        b = _lerp(sx, u, v)
        c = _lerp(sy, a, b)
        u = dot(d00 + bw, rx0, ry0, rz1, rw)
        v = dot(d10 + bw, rx1, ry0, rz1, rw)
        a = _lerp(sx, u, v)
        u = dot(d01 + bw, rx0, ry1, rz1, rw)
        v = dot(d11 + bw, rx1, ry1, rz1, rw)
        b = _lerp(sx, u, v)
        d = _lerp(sy, a, b)
        return _lerp(sz, c, d)

    e = octant(bw0, rw0)
    f = octant(bw1, rw1)
    return _lerp(sw, e, f)


def noise3ds(p):
    """max_texutil's noise3DS: the noise boosted by 1.65 and clamped."""
    return np.clip(f32(1.65) * noise3(p), -1.0, 1.0).astype(f32)


def NOISE(p):
    """max_texutil's NOISE(): noise3DS remapped to 0..1."""
    return ((f32(1.0) + noise3ds(p)) * f32(0.5)).astype(f32)


def dent_noise(p):
    """Max's dent noise (tex_dent / tex_planet / tex_wood): the 21-cube
    table read trilinearly with LINEAR weights, periodic every 20
    units, in 0..1."""
    p = np.asarray(p, f32)
    m = _mod(p, 20.0)
    i = m.astype(i64)
    fx, fy, fz = _mod(m[:, 0], 1.0), _mod(m[:, 1], 1.0), _mod(m[:, 2], 1.0)
    ix, iy, iz = i[:, 0], i[:, 1], i[:, 2]
    ix1, iy1, iz1 = ix + 1, iy + 1, iz + 1
    t = DENT_TABLE
    n = t[iz, iy, ix]
    n00 = (n + fx * (t[iz, iy, ix1] - n)).astype(f32)
    n = t[iz1, iy, ix]
    n01 = (n + fx * (t[iz1, iy, ix1] - n)).astype(f32)
    n = t[iz, iy1, ix]
    n10 = (n + fx * (t[iz, iy1, ix1] - n)).astype(f32)
    n = t[iz1, iy1, ix]
    n11 = (n + fx * (t[iz1, iy1, ix1] - n)).astype(f32)
    n0 = (n00 + fy * (n10 - n00)).astype(f32)
    n1 = (n01 + fy * (n11 - n01)).astype(f32)
    return (n0 + fz * (n1 - n0)).astype(f32)


def wood_noise(x):
    """tex_wood's one-dimensional noise: the dent table's first row,
    linear between entries, periodic every 20, in 0..1."""
    m = _mod(np.asarray(x, f32), 20.0)
    ix = m.astype(i64)
    fx = _mod(m, 1.0)
    n0 = DENT_TABLE[0, 0, ix]
    n1 = DENT_TABLE[0, 0, ix + 1]
    return (n0 + fx * (n1 - n0)).astype(f32)


def threshold(x, a, b):
    """max_texutil's threshold: 0 below a, 1 above b, linear between."""
    a, b = f32(a), f32(b)
    if a == b:
        return np.clip(x, 0.0, 1.0).astype(f32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(f32)


def smoothstep(a, b, x):
    """max_texutil's smoothstep (the HLSL one: a < b)."""
    a, b = f32(a), f32(max(float(b), float(a) + 1e-6))
    t = np.clip((np.asarray(x, f32) - a) / (b - a), 0.0, 1.0).astype(f32)
    return (t * t * (f32(3.0) - f32(2.0) * t)).astype(f32)


# ---------------------------------------------------------------- Noise

@_chunked
def mx_noise(p, kind=0, levels=3.0, low=0.0, high=1.0, phase=0.0):
    """tex_noise.cpp: Regular = the 4D noise remapped; Fractal = the
    sum noise(p 2^i)/2^i over Levels (a fractional level enters at its
    fraction), remapped; Turbulence = the same sum of |noise|, NOT
    remapped; then the Low/High threshold."""
    k = int(kind)
    lev = float(np.clip(levels, 1.0, 10.0))
    n = p.shape[0]
    if k == 0:
        res = ((f32(1.0) + noise4(p, phase)) * f32(0.5)).astype(f32)
    else:
        count = int(np.ceil(lev))
        rest = lev - np.floor(lev)
        total = np.zeros(n, f32)
        f = f32(1.0)
        for i in range(count):
            factor = f32(rest) if (i == count - 1 and rest > 0.0) else f32(1.0)
            nz = noise4(p * f, phase)
            if k == 2:
                nz = np.abs(nz)
            total = (total + factor * nz / f).astype(f32)
            f = f32(f * f32(2.0))
        res = (f32(0.5) * (total + f32(1.0))).astype(f32) if k == 1 else total
    if low < high:
        res = threshold(res, low, high)
    return np.clip(res, 0.0, 1.0).astype(f32)


# ------------------------------------------------------------- Cellular

def cell_function(p, want2=True):
    """max_texutil's CellFunction: the smallest (and, for Chips, the
    two smallest) SQUARED distances to the Poisson-scattered points of
    the 27 cells about p, and the id of the nearest point. The cell id
    is P[x] + P[y] 512 + P[z] 262144 of the cell's lattice coordinates;
    its points come from the 10000-entry table by id mod 10000."""
    n = p.shape[0]
    ip = np.floor(p).astype(i64)
    fx0 = (ip[:, 0].astype(f32) - p[:, 0]).astype(f32)
    fy0 = (ip[:, 1].astype(f32) - p[:, 1]).astype(f32)
    fz0 = (ip[:, 2].astype(f32) - p[:, 2]).astype(f32)
    f1 = np.full(n, 1e30, f32)
    f2 = np.full(n, 1e30, f32)
    # the winner as a small float code (cell 0..26) * 9 + point 0..8,
    # exact in float32 and cheaper to carry than the 27-bit id itself
    best = np.zeros(n, f32)
    px = [PERM_TABLE[(ip[:, 0] + d) & (B - 1)] for d in (-1, 0, 1)]
    py = [PERM_TABLE[(ip[:, 1] + d) & (B - 1)] * 512 for d in (-1, 0, 1)]
    pz = [PERM_TABLE[(ip[:, 2] + d) & (B - 1)] * 262144 for d in (-1, 0, 1)]
    cids = []
    for dx in (-1, 0, 1):
        fipx = (fx0 + f32(dx)).astype(f32)
        for dy in (-1, 0, 1):
            fipy = (fy0 + f32(dy)).astype(f32)
            for dz in (-1, 0, 1):
                fipz = (fz0 + f32(dz)).astype(f32)
                cid = px[dx + 1] + py[dy + 1] + pz[dz + 1]
                y = cid - (cid // 10000) * 10000
                c = len(cids)
                cids.append(cid)
                for k in range(9):
                    qx = CELL_X[k][y] + fipx
                    qy = CELL_Y[k][y] + fipy
                    qz = CELL_Z[k][y] + fipz
                    d = (qx * qx + qy * qy + qz * qz).astype(f32)
                    if want2:
                        # the second-nearest, without a branch: when d
                        # beats f1 the old f1 becomes f2, else f2 keeps
                        # the smaller of itself and d
                        f2 = np.minimum(f2, np.maximum(f1, d))
                    closer = d < f1
                    best = best + closer * (f32(c * 9 + k) - best)
                    f1 = np.minimum(f1, d)
    code = best.astype(i64)
    fid = np.stack(cids, 0)[code // 9, np.arange(n)] + code % 9
    return f1, f2, fid


def fractal_cell_function(p, iterations, lacunarity, want2=True):
    """max_texutil's FractalCellFunction: the cell distances summed
    over iterations at rising frequency, each divided by its
    frequency; a fractional iteration enters at its fraction. The cell
    id is the LAST pass's, as Max's is."""
    it = int(min(float(iterations), 25.0))
    rem = float(iterations) - it
    f1, f2, fid = cell_function(p, want2)
    u = f32(lacunarity)
    for _i in range(1, it):
        d1, d2, fid = cell_function((p * u).astype(f32), want2)
        f1 = (f1 + d1 / u).astype(f32)
        f2 = (f2 + d2 / u).astype(f32)
        u = f32(u * f32(lacunarity))
    if rem > 0.0:
        d1, d2, fid = cell_function((p * u).astype(f32), want2)
        f1 = (f1 + f32(rem) * d1 / u).astype(f32)
        f2 = (f2 + f32(rem) * d2 / u).astype(f32)
    return f1, f2, fid


@_chunked
def mx_cellular(p, chips=False, spread=0.5, fractal=False, iterations=3.0,
                roughness=0.0):
    """tex_cellular.cpp: the point offset by 1000, then Circular reads
    the nearest point's squared distance over Spread; Chips reads one
    minus the gap between the two nearest over half the Spread.
    Returns (u, cell id)."""
    q = (p + f32(1000.0)).astype(f32)
    lac = 2.0 - float(roughness)
    if fractal:
        f1, f2, fid = fractal_cell_function(q, iterations, lac, bool(chips))
    else:
        f1, f2, fid = cell_function(q, bool(chips))
    if chips:
        u = f32(1.0) - (f2 - f1) / f32(max(float(spread) * 0.5, 1e-6))
    else:
        u = f1 / f32(max(float(spread), 1e-6))
    return u.astype(f32), fid


def cellular_colors(u, fid, cell, div1, div2, low=0.0, mid=0.5, high=1.0,
                    variation=0.0):
    """The Cellular map's colour rule exactly: Variation scales the
    cell colour by the cell's own random; below Low the cell colour;
    Low..Mid the cell colour to Division 1; Mid..High Division 1 to
    Division 2; above High Division 2."""
    cell = np.asarray(cell, f32).copy()
    var = float(variation) / 50.0
    if var > 0.0:
        vr = (rand01(np.asarray(fid, i64) % 10000) * f32(var)
              + f32(1.0 - var * 0.5)).astype(f32)
        cell[:, :3] = np.clip(cell[:, :3] * vr[:, None], 0.0, 1.0)
    low, mid, high = f32(low), f32(mid), f32(high)
    ml = f32(max(float(mid - low), 1e-6))
    hm = f32(max(float(high - mid), 1e-6))
    t1 = ((u - low) / ml).astype(f32)[:, None]
    t2 = ((u - mid) / hm).astype(f32)[:, None]
    out = np.where(u[:, None] < low, cell,
                   np.where(u[:, None] > high, div2,
                            np.where(u[:, None] < mid,
                                     div1 * t1 + (f32(1.0) - t1) * cell,
                                     div2 * t2 + (f32(1.0) - t2) * div1)))
    out = out.astype(f32)
    out[:, 3] = 1.0
    return out


# ------------------------------------------------- Smoke, Speckle, Splat

def smoke_velocities(count=20):
    """Max's smoke velocity table (its own random directions); ours
    from the hash, one vector in -1..1 per iteration."""
    i = np.arange(count, dtype=i64)
    return np.stack([PT.hash3(i, i64(41), i64(43)) * f32(2.0) - f32(1.0),
                     PT.hash3(i, i64(47), i64(53)) * f32(2.0) - f32(1.0),
                     PT.hash3(i, i64(59), i64(61)) * f32(2.0) - f32(1.0)],
                    1).astype(f32)


SMOKE_VEL = smoke_velocities()


@_chunked
def mx_smoke(p, iterations=5, phase=0.0, exponent=1.5):
    """tex_smoke.cpp: |noise3| summed over Iterations at doubling
    frequency, each octave drifted by its own velocity times Phase
    (the drift growing 2.4x per octave), clamped, raised to Exponent."""
    x = np.asarray(p, f32).copy()
    mag = np.zeros(p.shape[0], f32)
    s, ft, hfb = f32(1.0), f32(1.0), f32(2.4)
    for i in range(int(max(1, min(iterations, 20)))):
        k = f32(ft * f32(phase))
        r = (x + SMOKE_VEL[i][None, :] * k).astype(f32)
        mag = (mag + np.abs(noise3(r)) / s).astype(f32)
        x = (x * f32(2.0)).astype(f32)
        s = f32(s * f32(2.0))
        ft = f32(ft * hfb)
    d = np.minimum(mag, f32(1.0))
    return np.power(d, f32(max(float(exponent), 1e-4))).astype(f32)


@_chunked
def mx_speckle(p):
    """tex_speckle.cpp: the point times 10, six octaves of NOISE,
    halving, summed; capped."""
    q = (np.asarray(p, f32) * f32(10.0)).astype(f32)
    total = np.zeros(p.shape[0], f32)
    s = f32(1.0)
    for _i in range(6):
        total = (total + NOISE(q) / s).astype(f32)
        s = f32(s * f32(2.0))
        q = (q * f32(2.0)).astype(f32)
    return np.minimum(total, f32(1.0)).astype(f32)


@_chunked
def mx_splat(p, iterations=4, threshold=0.25):
    """tex_splat.cpp: one minus the product over Iterations of the
    smoothstep of NOISE about Threshold (a 0.02 band)."""
    q = np.asarray(p, f32).copy()
    fact = np.ones(p.shape[0], f32)
    for _i in range(int(max(1, min(iterations, 5)))):
        t = np.minimum(NOISE(q), f32(1.0))
        fact = (fact * smoothstep(f32(threshold) - f32(0.02),
                                  f32(threshold) + f32(0.02), t)).astype(f32)
        q = (q * f32(2.0)).astype(f32)
    return (f32(1.0) - fact).astype(f32)


@_chunked
def mx_stucco(p, thickness=0.22, threshold=0.39):
    """tex_stucco.cpp: the noise past Threshold over Thickness, shaped
    by a knee curve (CRV 0.2)."""
    CRV, K, K1 = f32(0.2), f32(0.625), f32(3.125)
    f = (f32(0.5) * (noise3(p) + f32(1.0))).astype(f32)
    f = ((f - f32(threshold)) / f32(max(float(thickness), 1e-6))).astype(f32)
    out = np.where(f < CRV, K1 * f * f,
                   np.where(f < f32(1.0) - CRV, K * (f32(2.0) * f - CRV),
                            f32(1.0) - K1 * (f32(1.0) - f) * (f32(1.0) - f)))
    out = np.where(f >= 1.0, f32(1.0), out)
    out = np.where(f <= 0.0, f32(0.0), out)
    return out.astype(f32)


# ----------------------------------------------------------------- Swirl

@_chunked
def mx_swirl(p, cx=-0.5, cy=-0.5, twist=0.75, intensity=5.0, amount=2.0,
             detail=4, contrast=0.2, seed=0.0):
    """tex_swirl.cpp: the UV point spun about its centre by Twist turns
    per squared radius, then Constant Detail octaves of noise3 at gain
    Color Contrast; value = Amount * Intensity * the sum (unclamped:
    the colours extrapolate and the node clamps, as Max's does)."""
    u = (p[:, 0] + f32(cx)).astype(f32)
    v = (p[:, 1] + f32(cy)).astype(f32)
    rsq = (u * u + v * v).astype(f32)
    ang = (f32(twist) * TWO_PI * rsq).astype(f32)
    sn, cs = np.sin(ang).astype(f32), np.cos(ang).astype(f32)
    ppx = (v * cs - u * sn).astype(f32)
    ppy = (v * sn + u * cs).astype(f32)
    ppz = np.full(p.shape[0], f32(seed), f32)
    a = np.zeros(p.shape[0], f32)
    l, o = f32(1.0), f32(1.0)
    for _i in range(int(max(1, min(detail, 7)))):
        a = (a + o * noise3(np.stack([ppx * l, ppy * l, ppz * l], 1))).astype(f32)
        l = f32(l * f32(2.0))
        o = f32(o * f32(contrast))
    return (f32(amount) * f32(intensity) * a).astype(f32)


# ---------------------------------------------------------------- Planet

@_chunked
def mx_planet(p, island=0.25):
    """tex_planet.cpp: the dent noise plus a fifth of it at the Island
    Factor's frequency."""
    p = np.asarray(p, f32)
    return (dent_noise(p) + dent_noise((p * f32(island)).astype(f32))
            / f32(5.0)).astype(f32)


def planet_colors(d, cols, ocean_pct=50.0, blend=True):
    """The Planet map's eight-colour rule exactly: below sea level the
    depth runs the first three colours (and into the fourth when Blend
    Water/Land is on); above it the height runs colours four to eight."""
    n = d.shape[0]
    land = float(np.clip(ocean_pct, 0.0, 100.0)) / 100.0
    land = f32(min(max(land, 1e-4), 1.0 - 1e-4))
    c = [np.asarray(x, f32) for x in cols]
    water = d < land
    dw = (d / land * f32(3.0)).astype(f32)
    iw = np.minimum(dw.astype(i64), 2)
    fw = (dw - iw.astype(f32)).astype(f32)[:, None]
    omf = (f32(1.0) - fw).astype(f32)
    w0 = omf * c[0] + fw * c[1]
    w1 = omf * c[1] + fw * c[2]
    w2 = (omf * c[2] + fw * c[3]) if blend else np.broadcast_to(c[2], (n, 4))
    wcol = np.where(iw[:, None] == 0, w0, np.where(iw[:, None] == 1, w1, w2))
    dl = ((d - land) / (f32(1.0) - land) * f32(5.0)).astype(f32)
    il = np.minimum(np.maximum(dl, f32(0.0)).astype(i64), 5)
    fl = (dl - il.astype(f32)).astype(f32)[:, None]
    omf = (f32(1.0) - fl).astype(f32)
    l0 = omf * c[3] + fl * c[4]
    l1 = omf * c[4] + fl * c[5]
    l2 = omf * c[5] + fl * c[6]
    l3 = omf * c[6] + fl * c[7]
    lcol = np.where(il[:, None] == 0, l0,
                    np.where(il[:, None] == 1, l1,
                             np.where(il[:, None] == 2, l2,
                                      np.where(il[:, None] == 3, l3,
                                               np.broadcast_to(c[7], (n, 4))))))
    return np.where(water[:, None], wcol, lcol).astype(f32)


# ----------------------------------------------------------------- Waves

def msvc_rand(h):
    """The C runtime's rand() Max seeds its wave sets with: the LCG
    step in 32-bit wrap, the value bits 16..30. Returns (state, value)."""
    h = (int(h) * 214013 + 2531011) & 0xffffffff
    return h, (h >> 16) & 0x7fff


def wave_sets(count, radius, len_min, len_max, dist3d, seed):
    """tex_water's init: the centres, periods and rates of the wave
    sets from the seed, exactly Max's rand() walk (x, y for 3D, z, then
    the period)."""
    h = int(seed) & 0xffffffff
    sets = []
    for _i in range(count):
        h, r = msvc_rand(h)
        cx = f32(r) / f32(16384.0) - f32(1.0)
        if dist3d:
            h, r = msvc_rand(h)
            cy = f32(r) / f32(16384.0) - f32(1.0)
        else:
            cy = f32(0.0)
        h, r = msvc_rand(h)
        cz = f32(r) / f32(16384.0) - f32(1.0)
        ln = f32(np.sqrt(f32(cx * cx + cy * cy + cz * cz)))
        d = f32(radius) / f32(max(float(ln), 1e-6))
        h, r = msvc_rand(h)
        period = f32(r) / f32(32768.0) * (f32(len_max) - f32(len_min)) + f32(len_min)
        period = f32(max(float(period), 1e-6))
        rate = f32(np.sqrt(f32(max(float(len_max), 1e-6)) / period))
        sets.append((f32(cx * d), f32(cy * d), f32(cz * d), period, rate))
    return sets


def mx_waves(p, sets=3, radius=20.0, len_min=1.0, len_max=1.0,
             amplitude=1.0, phase=0.0, dist3d=True, seed=30159):
    """tex_water.cpp: for each wave set, v = (p - centre) / period, the
    wave 0.5 (1 + sin((|v| - Phase * rate) 2pi)) weighted by
    period / Wave Len Max; the sum times Amplitude over the count."""
    count = int(max(1, min(sets, 50)))
    ws = wave_sets(count, radius, len_min, len_max, bool(dist3d), seed)
    n = np.zeros(p.shape[0], f32)
    lmax = f32(max(float(len_max), 1e-6))
    for cx, cy, cz, period, rate in ws:
        vx = ((p[:, 0] - cx) / period).astype(f32)
        vy = ((p[:, 1] - cy) / period).astype(f32)
        vz = ((p[:, 2] - cz) / period).astype(f32)
        d = np.sqrt((vx * vx + vy * vy + vz * vz).astype(f32)).astype(f32)
        t = (f32(0.5) * (f32(1.0) + np.sin(((d - f32(phase) * rate) * TWO_PI)
                                             .astype(f32)).astype(f32))).astype(f32)
        n = (n + t * period / lmax).astype(f32)
    v = (n * f32(amplitude) / f32(count)).astype(f32)
    return np.minimum(v, f32(1.0)).astype(f32)


# --------------------------------------------------------------- Checker

def _sintegral(x):
    fl = np.floor(x).astype(f32)
    return (fl * f32(0.5) + np.maximum(f32(0.0), (x - fl) - f32(0.5))).astype(f32)


def mx_checker(p, soften=0.0):
    """tex_checker.cpp: with Soften, the square wave's integral over a
    Soften-wide window in u and in v, combined as s t + (1 - s)(1 - t);
    without it the plain parity test. 0 = Color 1, 1 = Color 2."""
    u, v = p[:, 0].astype(f32), p[:, 1].astype(f32)
    if float(soften) <= 0.0:
        fu = (u - np.floor(u)).astype(f32)
        fv = (v - np.floor(v)).astype(f32)
        one = np.logical_xor(fu > 0.5, fv > 0.5)
        return np.where(one, f32(0.0), f32(1.0)).astype(f32)
    du = f32(soften)
    hdu = f32(du * f32(0.5))
    s = ((_sintegral(u + hdu) - _sintegral(u - hdu)) / du).astype(f32)
    t = ((_sintegral(v + hdu) - _sintegral(v - hdu)) / du).astype(f32)
    return (s * t + (f32(1.0) - s) * (f32(1.0) - t)).astype(f32)


# ----------------------------------------------------------------- Tiles

TILE_PATTERNS = ('STACK', 'RUNNING', 'ENGLISH', 'FLEMISH')


def _boxstep(a, b, x):
    a = np.asarray(a, f32)
    b = np.asarray(b, f32)
    return np.clip((x - a) / (b - a), 0.0, 1.0).astype(f32)


def mx_tiles(p, pattern=1, hcount=4.0, vcount=4.0, hgap=0.5, vgap=0.5,
             line_shift=0.5, random_shift=0.0, holes=0.0, fade=0.05,
             color_var=0.0, seed=33862):
    """tex_bricks.cpp on Max's own mortar geometry: mortar = gap/100,
    brick width 1/(count + mortar), the box-stepped mortar bands in
    both axes, every second course shifted by Line Shift (plus Random
    Shift per course), English and Flemish courses by their header
    widths, Holes removing a percentage of bricks by their random, Fade
    and Color Variance scaling each brick by its random. Returns
    (mixer, brick random, fade factor)."""
    mtx = f32(f32(hgap) * f32(0.01))
    mty = f32(f32(vgap) * f32(0.01))
    bw = f32(f32(1.0) / (f32(hcount) + mtx))
    bh = f32(f32(1.0) / (f32(vcount) + mty))
    mwf = f32(mtx / bw)
    mhf = f32(mty / bh)
    ss = (p[:, 0] / bw).astype(f32)
    tt = (p[:, 1] / bh).astype(f32)
    tbrick = np.floor(tt).astype(f32)
    odd = _mod(tbrick, 2.0) > 0.5
    ss = (ss + f32(random_shift)
          * (PT.hash3(tbrick.astype(i64), i64(seed), i64(29)) - f32(0.5))).astype(f32)
    k = int(pattern)
    w = np.ones(p.shape[0], f32)
    if k == 1:                                     # running bond
        ss = np.where(odd, ss + f32(line_shift), ss).astype(f32)
    elif k == 2:                                   # English: header courses
        w = np.where(odd, f32(0.5), f32(1.0)).astype(f32)
        ss = np.where(odd, ss + f32(line_shift) * f32(0.5), ss).astype(f32)
    elif k == 3:                                   # Flemish
        ss = np.where(odd, ss + f32(line_shift) * f32(1.5), ss).astype(f32)
    if k == 3:
        t = _mod(ss, 1.5)
        head = t >= 1.0
        sbrick = (np.floor(ss / f32(1.5)) * f32(2.0)
                  + np.where(head, f32(1.0), f32(0.0))).astype(f32)
        fs = np.where(head, (t - f32(1.0)) / f32(0.5), t).astype(f32)
        w = np.where(head, f32(0.5), f32(1.0)).astype(f32)
    else:
        sbrick = np.floor(ss / w).astype(f32)
        fs = ((ss - sbrick * w) / w).astype(f32)
    ft = (tt - tbrick).astype(f32)
    mw = (mwf / w).astype(f32)
    wx = _boxstep(mw, mw + f32(1e-4), fs) - _boxstep(f32(1.0) - mw, f32(1.0) - mw + f32(1e-4), fs)
    wy = _boxstep(mhf, mhf + f32(1e-4), ft) - _boxstep(f32(1.0) - mhf, f32(1.0) - mhf + f32(1e-4), ft)
    mixer = (wx * wy).astype(f32)
    sb, tb = sbrick.astype(i64), tbrick.astype(i64)
    rnd = PT.hash3(sb, tb, i64(seed))
    hole = PT.hash3(sb, tb, i64(seed) + i64(97)) < f32(holes) / f32(100.0)
    mixer = np.where(hole, f32(0.0), mixer).astype(f32)
    n1 = (rnd - f32(0.5)).astype(f32)
    fade_v = ((f32(1.0) + f32(color_var) * n1) * (f32(1.0) + f32(fade) * n1)).astype(f32)
    return mixer, rnd.astype(f32), np.maximum(fade_v, f32(0.0)).astype(f32)


# ---------------------------------------------------------------- Marble

@_chunked
def mx_marble(p, width=0.025):
    """tex_marble.cpp: the point times 500 (FACT), veins on a
    seventeen-band cycle along x, the band index displaced by noise; a
    bright band inside the veins, a dark one at their heart."""
    q = (np.asarray(p, f32) * f32(500.0)).astype(f32)
    x, y, z = q[:, 0], q[:, 1], q[:, 2]
    r = np.stack([x / f32(100.0), y / f32(200.0), z / f32(200.0)], 1).astype(f32)
    d = ((x + f32(10000.0)) * f32(width) + f32(7.0) * NOISE(r)).astype(f32)
    idx = _mod(d, 17.0)
    r2 = np.stack([x / f32(70.0), y / f32(50.0), z / f32(50.0)], 1).astype(f32)
    vein = (f32(0.7) + f32(0.2) * NOISE(r2)).astype(f32)
    r3 = np.stack([x / f32(100.0), y / f32(100.0), x / f32(100.0)], 1).astype(f32)
    n3 = NOISE(r3)
    dd = (np.abs(d - np.floor(d / f32(17.0)) * f32(17.0) - f32(10.5)) * f32(0.1538462)).astype(f32)
    body = (f32(0.4) + f32(0.3) * dd + f32(0.2) * n3).astype(f32)
    heart = (f32(0.2) * (f32(1.0) + n3)).astype(f32)
    out = np.where(idx < 4.0, vein,
                   np.where((idx < 9.0) | (idx >= 12.0), body, heart))
    return np.clip(out, 0.0, 1.0).astype(f32)


# --------------------------------------------------------- Perlin Marble

_CR = ((-0.5, 1.5, -1.5, 0.5), (1.0, -2.5, 2.0, -0.5),
       (-0.5, 0.0, 0.5, 0.0), (0.0, 1.0, 0.0, 0.0))


def color_spline(x, knots):
    """max_texutil's color_spline: the Catmull-Rom over the knots."""
    nk = len(knots)
    nspans = nk - 3
    x = (np.clip(np.asarray(x, f32), 0.0, 1.0) * f32(nspans)).astype(f32)
    span = np.minimum(x.astype(i64), nspans - 1)
    x = (x - span.astype(f32)).astype(f32)
    K = np.stack(knots, 0).astype(f32)               # (nk, n, 3)
    idx = np.arange(x.shape[0])
    k0, k1, k2, k3 = (K[span + j, idx] for j in range(4))

    def row(j):
        c = [f32(v) for v in _CR[j]]
        return (c[0] * k0 + c[1] * k1 + c[2] * k2 + c[3] * k3).astype(f32)

    c3, c2, c1, c0 = row(0), row(1), row(2), row(3)
    xx = x[:, None]
    return (((c3 * xx + c2) * xx + c1) * xx + c0).astype(f32)


def _value_scale(rgb, sat):
    """Perlin Marble's Saturation: the colour's HSV value scaled by
    2 sat - 1 (sat in 0..1) -- which at the defaults DARKENS."""
    k = f32(2.0 * float(sat) - 1.0)
    return np.clip(rgb * k, 0.0, 1.0).astype(f32)


@_chunked
def mx_perlin_marble(p, levels=8):
    """tex_perlin.cpp: turbulence over Levels of NOISE, the vein
    coordinate sin(x + 4 turb - 3) clamped."""
    p = np.asarray(p, f32)
    x = p[:, 0]
    turb = np.zeros(p.shape[0], f32)
    freq = f32(1.0)
    for _lv in range(int(max(1, min(levels, 8)))):
        turb = (turb + np.abs(NOISE((p * freq).astype(f32))) / freq).astype(f32)
        freq = f32(freq * f32(2.0))
    return np.clip(np.sin((x + (f32(4.0) * turb - f32(3.0))).astype(f32)),
                   0.0, 1.0).astype(f32)


def perlin_marble_colors(csp, c0, c1, sat1=0.85, sat2=0.70):
    """The thirteen knots of Perlin's marble spline, Max's order."""
    c0 = np.asarray(c0, f32)[:, :3]
    c1 = np.asarray(c1, f32)[:, :3]
    lc0 = _value_scale(c0, sat1)
    dc1 = _value_scale(c1, sat2)
    knots = [lc0, lc0, c0, c0, c0, lc0, lc0, c1, c1, dc1, dc1, lc0, dc1]
    rgb = color_spline(csp, knots)
    return np.concatenate([rgb, np.ones((rgb.shape[0], 1), f32)], 1).astype(f32)


# ------------------------------------------------------------------ Wood

@_chunked
def mx_wood(p, radial=0.5, axial=0.5):
    """tex_wood.cpp: each axis jittered by Radial Noise, the ring
    radius about x jittered again and by Axial Noise along x, the
    rings as a smoothstep band per unit radius."""
    p = np.asarray(p, f32)
    px = (p[:, 0] + wood_noise(p[:, 0]) * f32(radial)).astype(f32)
    py = (p[:, 1] + wood_noise(p[:, 1]) * f32(radial)).astype(f32)
    pz = (p[:, 2] + wood_noise(p[:, 2]) * f32(radial)).astype(f32)
    r = np.sqrt((py * py + pz * pz).astype(f32)).astype(f32)
    px = (px / f32(4.0)).astype(f32)
    r = (r + (wood_noise(r) + f32(axial) * wood_noise(px))).astype(f32)
    r = _mod(r, 1.0)
    return (smoothstep(0.0, 0.8, r) - smoothstep(0.83, 1.0, r)).astype(f32)


# ------------------------------------------------------------------ Dent

@_chunked
def mx_dent(p, strength=20.0, iterations=2):
    """tex_dent.cpp: the point times 50, |0.5 - dent noise| summed over
    Iterations at doubling frequency, cubed, times Strength."""
    q = (np.asarray(p, f32) * f32(50.0)).astype(f32)
    s = f32(1.0)
    mag = np.zeros(p.shape[0], f32)
    for _i in range(int(max(1, min(iterations, 10)))):
        mag = (mag + np.abs(f32(0.5) - dent_noise(q)) / s).astype(f32)
        s = f32(s * f32(2.0))
        q = (q * f32(2.0)).astype(f32)
    return np.clip(mag * mag * mag * f32(strength), 0.0, 1.0).astype(f32)


# --------------------------------------------------------- Gradient Ramp

GRAD_TYPES = ('FOUR_CORNER', 'BOX', 'DIAGONAL', 'LINEAR', 'NORMAL', 'PONG',
              'RADIAL', 'SPIRAL', 'SWEEP', 'TARTAN', 'MAPPED')


def mx_gradient_ramp(u, v, kind=3, ndv=None, mapped=None):
    """tex_gradramp.cpp's gradient shapes, u v in 0..1 -> the ramp
    coordinate in 0..1 (Normal reads the view angle, Mapped the linked
    map's intensity)."""
    u = np.asarray(u, f32)
    v = np.asarray(v, f32)
    k = int(kind)
    if k == 0:                                     # four corner
        a = np.where(u > 0.0, (v * v) / np.maximum(u, f32(1e-6)) * u, v)
    elif k == 1:                                   # box
        lu = np.abs(u - f32(0.5)) * f32(2.0)
        lv = np.abs(v - f32(0.5)) * f32(2.0)
        a = np.minimum(np.maximum(lu, lv), f32(1.0))
    elif k == 2:                                   # diagonal
        a = np.sqrt(f32(2.0) * (v - u) * (v - u))
    elif k == 4:                                   # normal (view angle)
        a = f32(1.0) - np.clip(np.abs(ndv), 0.0, 1.0) if ndv is not None else v
    elif k == 5:                                   # pong
        a = np.where(v > 0.0, u / np.maximum(v, f32(1e-6)), v)
        a = np.where(a > 1.0, np.where(u > 0.0, v / np.maximum(u, f32(1e-6)), f32(0.0)), a)
        a = np.minimum(a, f32(1.0))
    elif k == 6:                                   # radial
        a = np.minimum(np.sqrt((u - f32(0.5)) ** 2 + (v - f32(0.5)) ** 2) * f32(2.0), f32(1.0))
    elif k == 7:                                   # spiral
        ln = np.sqrt(u * u + v * v)
        c = np.clip(np.where(ln > 0, u / np.maximum(ln, f32(1e-6)), f32(1.0)), -1.0, 1.0)
        a = np.minimum(np.arccos(c).astype(f32) * f32(180.0 / np.pi) / f32(90.0), f32(1.0))
    elif k == 8:                                   # sweep
        lu, lv = u - f32(0.5), v - f32(0.5)
        ln = np.sqrt(lu * lu + lv * lv)
        c = np.clip(np.where(ln > 0, lu / np.maximum(ln, f32(1e-6)), f32(1.0)), -1.0, 1.0)
        x = np.arccos(c).astype(f32) * f32(180.0 / np.pi)
        x = np.where(lv > 0.0, f32(360.0) - x, x)
        a = np.minimum(x / f32(360.0), f32(1.0))
    elif k == 9:                                   # tartan
        lu = np.abs(u - f32(0.5)) * f32(2.0)
        lv = np.abs(v - f32(0.5)) * f32(2.0)
        a = f32(1.0) - np.minimum(np.minimum(lu, lv), f32(1.0))
    elif k == 10:                                  # mapped
        a = np.asarray(mapped, f32) if mapped is not None else v
    else:                                          # linear
        a = v
    return np.clip(a, 0.0, 1.0).astype(f32)


# --------------------------------------------------------------- Falloff

def max_fresnel(ndv, ior):
    """Falloff's Fresnel exactly: the unpolarised dielectric equation
    on the cosine between the view and the half-vector of view and
    reflection -- which is |N . V|."""
    c = np.clip(np.abs(np.asarray(ndv, f32)), 0.0, 1.0).astype(f32)
    ior = np.asarray(ior, f32)
    g2 = (ior * ior + c * c - f32(1.0)).astype(f32)
    g = np.sqrt(np.maximum(g2, f32(0.0))).astype(f32)
    gc = np.maximum(g + c, f32(1e-6)).astype(f32)
    t = ((c * gc - f32(1.0)) / (c * gc + f32(1.0))).astype(f32)
    f = (((g - c) * (g - c)) / (f32(2.0) * gc * gc) * (f32(1.0) + t * t)).astype(f32)
    return np.where(g2 < 0.0, f32(1.0), np.clip(f, 0.0, 1.0)).astype(f32)


def mix_curve(lower, upper, x):
    """Mix / Blend / Falloff's mixing curve: a smoothstep between
    Lower and Upper (max_Mix's maxMixCurve)."""
    lo, hi = float(lower), float(upper)
    if hi <= lo:
        return np.where(np.asarray(x, f32) >= f32(hi), f32(1.0), f32(0.0)).astype(f32)
    t = np.clip((np.asarray(x, f32) - f32(lo)) / f32(hi - lo), 0.0, 1.0).astype(f32)
    return (t * t * (f32(3.0) - f32(2.0) * t)).astype(f32)


# ================================================ R243: the remaining maps
# Gradient, Color Correction and the Composite map -- Max's own
# algorithms again (tex_gradient's noise and sramp, ColorCorrection's
# HSL and its two lightness roads, Composite's twenty-five blend modes
# and its "over" compositing), on the tables above.

# --------------------------------------------------------------- Gradient

def sramp(x, a, b, d):
    """max_texutil's sramp: x held between a and b with the two corners
    rounded over +-d -- the Gradient map's noise threshold."""
    x = np.asarray(x, f32)
    a, b, d = f32(a), f32(b), f32(d)
    if d <= 0.0:
        return np.clip(x, a, b).astype(f32)
    p0, p1, p2, p3 = a - d, a + d, b - d, b + d
    q1 = ((x - p0) / (f32(2.0) * d)).astype(f32)
    q2 = ((p3 - x) / (f32(2.0) * d)).astype(f32)
    low = (a + q1 * q1 * d).astype(f32)
    high = (b - q2 * q2 * d).astype(f32)
    out = np.where(x <= p0, a,
                   np.where(x >= p3, b,
                            np.where((x >= p1) & (x <= p2), x,
                                     np.where((x > p0) & (x < p1), low, high))))
    return out.astype(f32)


def mx_gradient_noise(p, kind, levels, low, high, smooth):
    """tex_gradient's NoiseFunc: Max's noise3 plain (Regular), or Levels
    of it summed at doubling frequency and halving weight (Fractal;
    Turbulence folds each octave), a fraction of a level counting by
    its fraction; then, when Low < High, the Low..High threshold
    through sramp with corners of half the span times Smooth."""
    p = np.asarray(p, f32)
    kind = int(kind)
    levels = float(levels)
    if kind == 0 or (kind == 1 and levels == 1.0):
        res = noise3(p)
    else:
        res = np.zeros(p.shape[0], f32)
        f = f32(1.0)
        lv = f32(levels)
        while lv >= 1.0:
            n = noise3((p * f).astype(f32))
            if kind == 2:
                n = np.abs(n)
            res = (res + n / f).astype(f32)
            f = f32(f * f32(2.0))
            lv = f32(lv - f32(1.0))
        if lv > 0.0:
            n = noise3((p * f).astype(f32))
            if kind == 2:
                n = np.abs(n)
            res = (res + lv * n / f).astype(f32)
    if float(low) < float(high):
        sd = f32((float(high) - float(low)) * 0.5 * float(smooth))
        res = (f32(2.0) * sramp((res + f32(1.0)) / f32(2.0), low, high, sd)
               - f32(1.0)).astype(f32)
    return res.astype(f32)


def mx_gradient(u, v, kind, amount, noise):
    """tex_gradient's GradFunc: Linear reads v, Radial the distance from
    the tile's centre doubled and capped; Noise Amount times the noise
    is added and the result clamped."""
    u = np.asarray(u, f32)
    v = np.asarray(v, f32)
    if int(kind) == 1:
        lu = (u - f32(0.5)).astype(f32)
        lv = (v - f32(0.5)).astype(f32)
        a = np.minimum(np.sqrt(lu * lu + lv * lv).astype(f32) * f32(2.0), f32(1.0))
    else:
        a = v.copy()
    if float(amount) != 0.0:
        a = np.clip(a + f32(amount) * np.asarray(noise, f32), 0.0, 1.0)
    return a.astype(f32)


def gradient_colors(a, pos, c1, c2, c3):
    """Max's three-colour Gradient: Color 3 at 0 up to Color 2 at Color 2
    Position, then on to Color 1 at 1 (Max's Color #1 sits at the top
    of the ramp). Exactly at the position the colour is Color 2."""
    a = np.asarray(a, f32)[:, None]
    pos = f32(pos)
    lo = np.where(pos > 0.0, pos, f32(1.0))
    hi = np.where(pos < 1.0, f32(1.0) - pos, f32(1.0))
    t1 = (a / lo).astype(f32)
    t2 = ((a - pos) / hi).astype(f32)
    below = c3 * (f32(1.0) - t1) + c2 * t1
    above = c2 * (f32(1.0) - t2) + c1 * t2
    out = np.where(a < pos, below, np.where(a > pos, above, c2))
    return out.astype(f32)


# ------------------------------------------------ Color Correction's HSL

MIN_COLOR_DELTA = f32(0.00001)


def hsl_rotate(a, b, c):
    """max_ColorCorrection's rotate: a brought into b..c by whole spans,
    the span count truncated as the reference truncates it."""
    a = np.asarray(a, f32)
    b, c = f32(b), f32(c)
    delta = f32(c - b)
    f1 = np.trunc((c - a) / delta).astype(f32)
    a = np.where(a < b, a + delta * f1, a).astype(f32)
    f2 = np.trunc((a - b) / delta).astype(f32)
    a = np.where(a > c, a - delta * f2, a).astype(f32)
    return a


def calc_hue(c):
    """CalcHue: the hue in degrees, 0 for a grey (delta under 1e-5)."""
    c = np.asarray(c, f32)
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mn = np.minimum(r, np.minimum(g, b))
    mx = np.maximum(r, np.maximum(g, b))
    delta = (mx - mn).astype(f32)
    grey = delta < MIN_COLOR_DELTA
    safe = np.where(grey, f32(1.0), delta).astype(f32)
    h = np.where(r == mx, (g - b) / safe,
                 np.where(g == mx, f32(2.0) + (b - r) / safe,
                          f32(4.0) + (r - g) / safe)).astype(f32)
    h = hsl_rotate(h * f32(60.0), 0.0, 360.0)
    return np.where(grey, f32(0.0), h).astype(f32)


def calc_lum(c):
    c = np.asarray(c, f32)
    mn = np.minimum(c[:, 0], np.minimum(c[:, 1], c[:, 2]))
    mx = np.maximum(c[:, 0], np.maximum(c[:, 1], c[:, 2]))
    return np.clip((mx + mn) / f32(2.0), 0.0, 1.0).astype(f32)


def calc_sat(c):
    c = np.asarray(c, f32)
    mn = np.minimum(c[:, 0], np.minimum(c[:, 1], c[:, 2]))
    mx = np.maximum(c[:, 0], np.maximum(c[:, 1], c[:, 2]))
    lum = calc_lum(c)
    zero = (mn == mx) | (lum == 0.0)
    d1 = np.where(zero, f32(1.0), f32(2.0) * lum).astype(f32)
    d2 = np.where(zero, f32(1.0), f32(2.0) - f32(2.0) * lum).astype(f32)
    d2 = np.where(d2 == 0.0, f32(1.0), d2).astype(f32)
    s = np.where(lum <= 0.5, np.clip((mx - mn) / d1, 0.0, 1.0),
                 np.clip((mx - mn) / d2, 0.0, 1.0)).astype(f32)
    return np.where(zero, f32(0.0), s).astype(f32)


def hue_to_rgb(v1, v2, h):
    h = hsl_rotate(h, 0.0, 1.0)
    return np.where(f32(6.0) * h < 1.0, v1 + (v2 - v1) * f32(6.0) * h,
                    np.where(f32(2.0) * h < 1.0, v2,
                             np.where(f32(3.0) * h < 2.0,
                                      v1 + (v2 - v1) * (f32(2.0 / 3.0) - h) * f32(6.0),
                                      v1))).astype(f32)


def hsl_to_rgb(h, s, lum):
    """HSLtoRGB: the reference's own conversion, the result saturated."""
    h = np.asarray(h, f32)
    s = np.asarray(s, f32)
    lum = np.asarray(lum, f32)
    q = np.where(lum < 0.5, lum * (f32(1.0) + s), lum + s - lum * s).astype(f32)
    p = (f32(2.0) * lum - q).astype(f32)
    hk = (h / f32(360.0)).astype(f32)
    r = hue_to_rgb(p, q, hk + f32(1.0 / 3.0))
    g = hue_to_rgb(p, q, hk)
    b = hue_to_rgb(p, q, hk - f32(1.0 / 3.0))
    return np.clip(np.stack([r, g, b], 1), 0.0, 1.0).astype(f32)


REWIRE = ('RED', 'GREEN', 'BLUE', 'ALPHA', 'RED_INV', 'GREEN_INV',
          'BLUE_INV', 'ALPHA_INV', 'MONO', 'ONE', 'ZERO')


def rewire_value(index, c):
    """ColorCorrection's rewire_value: one output channel from the
    source colour by index (R G B A, their inverses, mono, one, zero)."""
    c = np.asarray(c, f32)
    k = int(index)
    if k < 4:
        return c[:, k].astype(f32)
    if k < 8:
        return (f32(1.0) - c[:, k - 4]).astype(f32)
    if k == 8:
        return ((c[:, 0] + c[:, 1] + c[:, 2]) / f32(3.0)).astype(f32)
    if k == 9:
        return np.ones(c.shape[0], f32)
    return np.zeros(c.shape[0], f32)


def color_correction(c, rewire, hue_shift, saturation, tint, strength,
                     advanced, brightness, contrast, gain, gamma, pivot, lift):
    """max_ColorCorrection in Max's order: the channels rewired, the
    hue shifted and the saturation moved in HSL with the hue pulled
    toward the tint's by Strength, then the lightness -- Standard's
    brightness / contrast about mid-grey, or Advanced's gain, gamma
    about the pivot, and lift. The per-pixel controls are arrays."""
    c = np.asarray(c, f32)
    out = np.stack([rewire_value(rewire[k], c) for k in range(4)], 1).astype(f32)
    rgb = out[:, :3]
    h = calc_hue(rgb)
    s = calc_sat(rgb)
    lum = calc_lum(rgb)
    h = hsl_rotate(h + np.asarray(hue_shift, f32), 0.0, 360.0)
    s = np.clip(s + np.asarray(saturation, f32) / f32(100.0), 0.0, 1.0).astype(f32)
    ht = calc_hue(np.asarray(tint, f32))
    h = (h + (ht - h) * (np.asarray(strength, f32) / f32(100.0))).astype(f32)
    rgb = hsl_to_rgb(h, s, lum)
    if not advanced:
        b = (np.asarray(brightness, f32) / f32(100.0)).astype(f32)[:, None]
        k = (np.asarray(contrast, f32) / f32(100.0)).astype(f32)[:, None]
        rgb = ((rgb - f32(0.5)) * (f32(1.0) + k) + f32(0.5) + b).astype(f32)
    else:
        gn = np.asarray(gain, f32)[:, None]
        gm = np.maximum(np.asarray(gamma, f32), f32(1e-6))[:, None]
        pv = np.asarray(pivot, f32)
        pv = np.where(pv == 0.0, f32(1e-6), pv).astype(f32)[:, None]
        lf = np.asarray(lift, f32)[:, None]
        base = np.maximum(((rgb * gn) / f32(100.0)) / pv, f32(0.0)).astype(f32)
        rgb = (pv * np.power(base, f32(1.0) / gm) + lf).astype(f32)
    out[:, :3] = rgb
    return out.astype(f32)


# ----------------------------------------------------- the Composite map

BLEND_MODES = ('NORMAL', 'AVERAGE', 'ADDITION', 'SUBTRACT', 'DARKEN',
               'MULTIPLY', 'COLOR_BURN', 'LINEAR_BURN', 'LIGHTEN', 'SCREEN',
               'COLOR_DODGE', 'LINEAR_DODGE', 'SPOTLIGHT', 'SPOTLIGHT_BLEND',
               'OVERLAY', 'SOFT_LIGHT', 'HARD_LIGHT', 'PIN_LIGHT', 'HARD_MIX',
               'DIFFERENCE', 'EXCLUSION', 'HUE', 'SATURATION', 'COLOR', 'VALUE')


def _hsl_blend(hue_src, sat_src, lum_src, fg):
    rgb = hsl_to_rgb(calc_hue(hue_src), calc_sat(sat_src), calc_lum(lum_src))
    return np.concatenate([rgb, fg[:, 3:4]], 1).astype(f32)


def mx_blend(mode, fg, bg):
    """max_Composite's blend(): the layer (fg) against what is beneath
    (bg), all four channels, Max's twenty-five modes in Max's order.
    Addition and Subtract run unclamped as Max's do (Linear Dodge and
    Linear Burn are their clamped forms)."""
    fg = np.asarray(fg, f32)
    bg = np.asarray(bg, f32)
    m = int(mode)
    one, zero, half = f32(1.0), f32(0.0), f32(0.5)
    if m == 0:
        return fg.copy()
    if m == 1:
        return ((fg + bg) / f32(2.0)).astype(f32)
    if m == 2:
        return (fg + bg).astype(f32)
    if m == 3:
        return (bg - fg).astype(f32)
    if m == 4:
        return np.minimum(fg, bg).astype(f32)
    if m == 5:
        return (fg * bg).astype(f32)
    if m == 6:
        safe = np.where(fg != 0.0, fg, one)
        return np.where(fg != 0.0, np.maximum(one - (one - bg) / safe, zero), zero).astype(f32)
    if m == 7:
        return np.maximum(fg + bg - one, zero).astype(f32)
    if m == 8:
        return np.maximum(fg, bg).astype(f32)
    if m == 9:
        return (fg + bg - fg * bg).astype(f32)
    if m == 10:
        safe = np.where(fg != 1.0, one - fg, one)
        return np.where(fg != 1.0, np.minimum(bg / safe, one), one).astype(f32)
    if m == 11:
        return np.minimum(fg + bg, one).astype(f32)
    if m == 12:
        return np.minimum(f32(2.0) * fg * bg, one).astype(f32)
    if m == 13:
        return np.minimum(fg * bg + bg, one).astype(f32)
    if m == 14:
        return np.clip(np.where(bg > 0.5, one - f32(2.0) * (one - fg) * (one - bg),
                                f32(2.0) * fg * bg), 0.0, 1.0).astype(f32)
    if m == 15:
        return np.clip(np.where(fg > 0.5,
                                bg + (f32(2.0) * fg - one) * np.sqrt(np.maximum(bg, zero)) - bg,
                                bg * (bg + f32(2.0) * fg * (one - bg))), 0.0, 1.0).astype(f32)
    if m == 16:
        return np.clip(np.where(fg > 0.5, one - f32(2.0) * (one - fg) * (one - bg),
                                f32(2.0) * fg * bg), 0.0, 1.0).astype(f32)
    if m == 17:
        pick = ((fg > 0.5) & (fg > bg)) | ((fg < 0.5) & (fg < bg))
        return np.where(pick, fg, bg).astype(f32)
    if m == 18:
        return np.where(fg + bg <= 1.0, zero, one).astype(f32)
    if m == 19:
        return np.abs(fg - bg).astype(f32)
    if m == 20:
        return (fg + bg - f32(2.0) * fg * bg).astype(f32)
    if m == 21:
        return _hsl_blend(fg, bg, bg, fg)
    if m == 22:
        return _hsl_blend(bg, fg, bg, fg)
    if m == 23:
        return _hsl_blend(fg, fg, bg, fg)
    if m == 24:
        return _hsl_blend(bg, bg, fg, fg)
    return fg.copy()


def composite_layer(res, fg, opacity_pct, mask, mode):
    """One layer of max_Composite over the result so far: the layer's
    colour un-premultiplied, its alpha scaled by the mask then by
    Opacity; the first visible layer simply becomes the result, every
    later one lands by Max's "over" -- the blend where both are
    present, the layer where only it is, the result where only it is,
    over the union alpha."""
    res = np.asarray(res, f32)
    fg = np.asarray(fg, f32).copy()
    a = fg[:, 3].copy()
    unp = (a != 1.0) & (a != 0.0)
    fg[:, :3] = np.where(unp[:, None], fg[:, :3] / np.where(unp, a, f32(1.0))[:, None],
                         fg[:, :3])
    fa = (a * np.asarray(mask, f32)).astype(f32)
    fa = (fa * (f32(opacity_pct) / f32(100.0))).astype(f32)
    fg[:, 3] = fa
    ra = res[:, 3]
    bl = mx_blend(mode, fg, res)
    alpha = (fa + (f32(1.0) - fa) * ra).astype(f32)
    safe = np.where(alpha != 0.0, alpha, f32(1.0)).astype(f32)
    rgb = ((bl[:, :3] * (fa * ra)[:, None]
            + fg[:, :3] * (fa * (f32(1.0) - ra))[:, None]
            + res[:, :3] * ((f32(1.0) - fa) * ra)[:, None]) / safe[:, None]).astype(f32)
    over = np.concatenate([rgb, alpha[:, None]], 1).astype(f32)
    return np.where((ra == 0.0)[:, None], fg, over).astype(f32)


def composite_finish(res):
    """max_Composite's last step: a result short of full alpha is
    premultiplied by it."""
    res = np.asarray(res, f32).copy()
    a = res[:, 3]
    res[:, :3] = np.where((a != 1.0)[:, None], res[:, :3] * a[:, None], res[:, :3])
    return res.astype(f32)
