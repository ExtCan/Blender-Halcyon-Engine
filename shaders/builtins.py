"""Runtime library for compiled shaders.

Values are numpy arrays with a leading lane axis:
    scalar   (N,)  or (1,) for uniforms
    vecN     (N,K) or (1,K)
    matCxR   (N,C,R) column-major, so m[c] is a column -- matching GLSL
Every function here is written to broadcast a lane count of 1 against N.
"""

import numpy as np

from .gtypes import (BOOL, FLOAT, INT, VEC2, VEC3, VEC4, GType)

F32 = np.float32


# ------------------------------------------------------------------ context

class Ctx:
    """Per-invocation state that some builtins need (derivatives, samplers)."""

    def __init__(self, n=1, px=None, py=None, width=0, height=0, tri=None,
                 samplers=None, filt='NEAREST', wrap='REPEAT'):
        self.n = n
        self.px = px
        self.py = py
        self.width = width
        self.height = height
        self.tri = tri
        self.samplers = samplers or {}
        self.filt = filt
        self.wrap = wrap


_CTX = Ctx()


def set_ctx(ctx):
    global _CTX
    _CTX = ctx
    return ctx


def ctx():
    return _CTX


# ---------------------------------------------------------------- utilities


def lanes(*vals):
    n = 1
    for v in vals:
        if isinstance(v, np.ndarray) and v.ndim >= 1:
            n = max(n, v.shape[0])
    return n


def bc(x, n):
    """Broadcast a value's lane axis to n. Structs and arrays recurse."""
    if x is None:
        return None
    if isinstance(x, dict):
        return {k: bc(v, n) for k, v in x.items()}
    if isinstance(x, list):
        return [bc(v, n) for v in x]
    if not isinstance(x, np.ndarray):
        return np.full(n, x, F32)
    if x.ndim == 0:
        return np.full(n, x, x.dtype)
    if x.shape[0] == n:
        return x
    if x.shape[0] == 1:
        return np.broadcast_to(x, (n,) + x.shape[1:])
    return x


def splat(x, k):
    """Scalar (N,) -> (N,k)."""
    if isinstance(x, np.ndarray) and x.ndim == 2:
        if x.shape[1] == k:
            return x
        if x.shape[1] == 1:
            return np.repeat(x, k, axis=1)
    a = np.asarray(x)
    if a.ndim == 0:
        return np.full((1, k), a, F32)
    return np.repeat(a[:, None], k, axis=1)


def vec(k, *parts):
    """GLSL vector constructor: flattens components, splats a lone scalar."""
    cols = []
    n = 1
    for p in parts:
        a = np.asarray(p)
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a[:, None]
        n = max(n, a.shape[0])
        cols.append(a)
    if len(cols) == 1 and cols[0].shape[1] == 1:
        return np.repeat(bc(cols[0], n), k, axis=1).astype(cols[0].dtype)
    if len(cols) == 1 and cols[0].shape[1] >= k:
        return cols[0][:, :k]
    cols = [bc(c, n) for c in cols]
    out = np.concatenate(cols, axis=1)
    if out.shape[1] < k:                       # pad like GLSL never does, but be kind
        out = np.concatenate([out, np.zeros((out.shape[0], k - out.shape[1]),
                                            out.dtype)], axis=1)
    return out[:, :k]


def mat(c, r, *parts):
    """Matrix constructor: mat3(1.0) -> identity*1, mat3(v1,v2,v3) -> columns."""
    if len(parts) == 1:
        a = np.asarray(parts[0])
        if a.ndim == 3:
            n = a.shape[0]
            out = np.zeros((n, c, r), F32)
            k = min(c, a.shape[1])
            j = min(r, a.shape[2])
            out[:, :k, :j] = a[:, :k, :j]
            for i in range(min(c, r)):
                if i >= k or i >= j:
                    out[:, i, i] = 1.0
            return out
        s = a if a.ndim else a.reshape(1)
        n = s.shape[0] if s.ndim else 1
        out = np.zeros((n, c, r), F32)
        for i in range(min(c, r)):
            out[:, i, i] = s
        return out
    if len(parts) == c and np.asarray(parts[0]).ndim == 2:
        n = lanes(*[np.asarray(p) for p in parts])
        return np.stack([bc(np.asarray(p, F32), n) for p in parts], axis=1)
    flat = []
    for p in parts:
        a = np.asarray(p, F32)
        if a.ndim == 0:
            flat.append(a.reshape(1, 1))
        elif a.ndim == 1:
            flat.append(a[:, None])
        else:
            for j in range(a.shape[1]):
                flat.append(a[:, j:j + 1])
    n = lanes(*flat)
    m = np.concatenate([bc(f, n) for f in flat], axis=1)
    return m.reshape(n, c, r)


def sel(mask, a, b):
    """Masked assignment: where(mask, a, b) with lane broadcasting."""
    if mask is True:
        return a
    if mask is False:
        return b
    if a is None or b is None:
        return a if a is not None else b
    if isinstance(a, dict):
        return {k: sel(mask, v, b.get(k, v)) for k, v in a.items()}
    if isinstance(a, list):
        return [sel(mask, v, b[i] if i < len(b) else v) for i, v in enumerate(a)]
    a = np.asarray(a)
    b = np.asarray(b)
    n = max(lanes(a, b), mask.shape[0] if isinstance(mask, np.ndarray) else 1)
    # A 1-D array of 2..4 elements is ambiguous: it could be that many lanes of
    # a scalar, or one uniform vector. The other operand settles it -- this is
    # what a vec3 read out of a `uniform vec3 x[N]` array looks like.
    if a.ndim == 1 and b.ndim == 2 and a.shape[0] == b.shape[1] and n != a.shape[0]:
        a = np.broadcast_to(a[None, :], (n, a.shape[0]))
    if b.ndim == 1 and a.ndim == 2 and b.shape[0] == a.shape[1] and n != b.shape[0]:
        b = np.broadcast_to(b[None, :], (n, b.shape[0]))
    a = bc(a, n)
    b = bc(b, n)
    m = bc(np.asarray(mask), n)
    if a.ndim == 2:
        m = m[:, None] if m.ndim == 1 else m
    elif a.ndim == 3:
        m = m[:, None, None] if m.ndim == 1 else m
    out = np.where(m, a, b)
    return out


def any_(m):
    if m is True:
        return True
    if m is False:
        return False
    return bool(np.any(m))


def to_bool(x):
    a = np.asarray(x)
    if a.dtype == bool:
        return a
    return a != 0


def to_float(x):
    a = np.asarray(x)
    return a.astype(F32) if a.dtype != F32 else a


def to_int(x):
    a = np.asarray(x)
    return a.astype(np.int32)


# ---------------------------------------------------------------- swizzling


def sw(v, idx):
    a = np.asarray(v)
    if a.ndim == 1:
        a = a[:, None]
    if len(idx) == 1:
        return a[:, idx[0]]
    return a[:, idx]


def sw_set(v, idx, val):
    a = np.array(v, copy=True)
    if a.ndim == 1:
        a = a[:, None]
    n = max(a.shape[0], lanes(np.asarray(val)))
    a = np.array(bc(a, n), copy=True)
    val = np.asarray(val)
    if val.ndim == 1 and len(idx) > 1:
        val = splat(val, len(idx))
    val = bc(val, n)
    if len(idx) == 1:
        a[:, idx[0]] = val if val.ndim == 1 else val[:, 0]
    else:
        for k, c in enumerate(idx):
            a[:, c] = val[:, k] if val.ndim == 2 else val
    return a


# ------------------------------------------------------------- matrix maths


def mat_mul_vec(m, v):
    n = max(m.shape[0], v.shape[0])
    m = bc(m, n)
    v = bc(v, n)
    return np.einsum('ncr,nc->nr', m, v)


def vec_mul_mat(v, m):
    n = max(m.shape[0], v.shape[0])
    m = bc(m, n)
    v = bc(v, n)
    return np.einsum('nr,ncr->nc', v, m)


def mat_mul_mat(a, b):
    n = max(a.shape[0], b.shape[0])
    a = bc(a, n)
    b = bc(b, n)
    return np.einsum('nkr,nck->ncr', a, b)


def transpose(m):
    return np.transpose(m, (0, 2, 1))


def determinant(m):
    return np.linalg.det(np.transpose(m, (0, 2, 1))).astype(F32)


def inverse(m):
    t = np.transpose(m, (0, 2, 1))
    inv = np.linalg.inv(t)
    return np.transpose(inv, (0, 2, 1)).astype(F32)


def matrixCompMult(a, b):
    return a * b


def outerProduct(a, b):
    return np.einsum('nr,nc->ncr', a, b)


# ------------------------------------------------------------ scalar / genType


def _uf(f):
    def g(x):
        return f(np.asarray(x, F32))
    return g


radians = _uf(np.radians)
degrees = _uf(np.degrees)
sin = _uf(np.sin)
cos = _uf(np.cos)
tan = _uf(np.tan)
asin = lambda x: np.arcsin(np.clip(np.asarray(x, F32), -1.0, 1.0))
acos = lambda x: np.arccos(np.clip(np.asarray(x, F32), -1.0, 1.0))
sinh = _uf(np.sinh)
cosh = _uf(np.cosh)
tanh = _uf(np.tanh)
asinh = _uf(np.arcsinh)
acosh = _uf(np.arccosh)
atanh = _uf(np.arctanh)
exp = _uf(np.exp)
log = lambda x: np.log(np.maximum(np.asarray(x, F32), 1e-30))
exp2 = _uf(np.exp2)
log2 = lambda x: np.log2(np.maximum(np.asarray(x, F32), 1e-30))
sqrt = lambda x: np.sqrt(np.maximum(np.asarray(x, F32), 0.0))
inversesqrt = lambda x: 1.0 / np.sqrt(np.maximum(np.asarray(x, F32), 1e-30))
abs_ = lambda x: np.abs(x)
sign = lambda x: np.sign(np.asarray(x, F32))
floor = _uf(np.floor)
ceil = _uf(np.ceil)
trunc = _uf(np.trunc)
round_ = _uf(np.round)
roundEven = _uf(np.round)
isnan = lambda x: np.isnan(np.asarray(x, F32))
isinf = lambda x: np.isinf(np.asarray(x, F32))
log10 = lambda x: np.log10(np.maximum(np.asarray(x, F32), 1e-30))


def atan(y, x=None):
    if x is None:
        return np.arctan(np.asarray(y, F32))
    return np.arctan2(np.asarray(y, F32), np.asarray(x, F32))


def atan2(y, x):
    return np.arctan2(np.asarray(y, F32), np.asarray(x, F32))


def pow_(x, y):
    x = np.asarray(x, F32)
    y = np.asarray(y, F32)
    if x.ndim == 2 and y.ndim == 1:
        y = splat(y, x.shape[1])
    if y.ndim == 2 and x.ndim == 1:
        x = splat(x, y.shape[1])
    return np.power(np.maximum(x, 0.0), y)


def fract(x):
    x = np.asarray(x, F32)
    return x - np.floor(x)


def mod(x, y):
    x = np.asarray(x, F32)
    y = np.asarray(y, F32)
    if x.ndim == 2 and y.ndim == 1:
        y = splat(y, x.shape[1])
    return x - y * np.floor(x / np.where(np.abs(y) < 1e-30, 1e-30, y))


def fmod(x, y):
    return np.fmod(np.asarray(x, F32), np.asarray(y, F32))


def min_(a, b):
    a, b = _align(a, b)
    return np.minimum(a, b)


def max_(a, b):
    a, b = _align(a, b)
    return np.maximum(a, b)


def _align(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 2 and b.ndim == 1:
        b = splat(b, a.shape[1])
    elif b.ndim == 2 and a.ndim == 1:
        a = splat(a, b.shape[1])
    return a, b


def clamp(x, lo, hi):
    x, lo = _align(x, lo)
    x, hi = _align(x, hi)
    return np.minimum(np.maximum(x, lo), hi)


def saturate(x):
    return np.clip(np.asarray(x, F32), 0.0, 1.0)


def mix(a, b, t):
    a, b = _align(a, b)
    if isinstance(t, np.ndarray) and t.dtype == bool:
        return sel(t, b, a)
    a2, t2 = _align(a, t)
    return a2 + (_align(b, a2)[0] - a2) * t2


def lerp(a, b, t):
    return mix(a, b, t)


def step(edge, x):
    edge, x = _align(edge, x)
    return (x >= edge).astype(F32)


def smoothstep(e0, e1, x):
    e0, x = _align(e0, x)
    e1, x = _align(e1, x)
    t = np.clip((x - e0) / np.where(np.abs(e1 - e0) < 1e-30, 1e-30, e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def dot_(a, b):
    a = np.asarray(a, F32)
    b = np.asarray(b, F32)
    if a.ndim == 1 and b.ndim == 1:
        return a * b
    a, b = _align(a, b)
    return np.sum(a * b, axis=1)


def length(v):
    v = np.asarray(v, F32)
    if v.ndim == 1:
        return np.abs(v)
    return np.sqrt(np.maximum(np.sum(v * v, axis=1), 0.0))


def distance(a, b):
    return length(np.asarray(a, F32) - np.asarray(b, F32))


def normalize(v):
    v = np.asarray(v, F32)
    if v.ndim == 1:
        return np.sign(v)
    l = np.sqrt(np.maximum(np.sum(v * v, axis=1), 1e-30))
    return v / l[:, None]


def cross(a, b):
    n = max(lanes(np.asarray(a)), lanes(np.asarray(b)))
    return np.cross(bc(np.asarray(a, F32), n), bc(np.asarray(b, F32), n))


def faceforward(n, i, nref):
    d = dot_(nref, i)
    return sel(d < 0.0, n, -np.asarray(n))


def reflect(i, n):
    i = np.asarray(i, F32)
    n = np.asarray(n, F32)
    return i - 2.0 * dot_(n, i)[:, None] * n


def refract(i, n, eta):
    i = np.asarray(i, F32)
    n = np.asarray(n, F32)
    eta = np.asarray(eta, F32)
    if eta.ndim == 1:
        eta = eta[:, None]
    d = dot_(n, i)[:, None]
    k = 1.0 - eta * eta * (1.0 - d * d)
    r = eta * i - (eta * d + np.sqrt(np.maximum(k, 0.0))) * n
    return np.where(k < 0.0, 0.0, r)


def lessThan(a, b):
    a, b = _align(a, b)
    return a < b


def lessThanEqual(a, b):
    a, b = _align(a, b)
    return a <= b


def greaterThan(a, b):
    a, b = _align(a, b)
    return a > b


def greaterThanEqual(a, b):
    a, b = _align(a, b)
    return a >= b


def equal(a, b):
    a, b = _align(a, b)
    return a == b


def notEqual(a, b):
    a, b = _align(a, b)
    return a != b


def any_comp(v):
    v = np.asarray(v)
    return np.any(v, axis=1) if v.ndim == 2 else to_bool(v)


def all_comp(v):
    v = np.asarray(v)
    return np.all(v, axis=1) if v.ndim == 2 else to_bool(v)


def not_(v):
    return ~to_bool(v)


# ------------------------------------------------------------------ texture


def texture(samp, uv, bias=None):
    """sampler2D lookup -> vec4."""
    if samp is None:
        n = lanes(np.asarray(uv))
        return np.zeros((n, 4), F32)
    c = ctx()
    u = np.asarray(uv, F32)
    if u.ndim == 1:
        u = u[:, None]
    return samp.sample(u[:, 0], u[:, 1], filt=c.filt, wrap=c.wrap).astype(F32)


def texture2D(samp, uv, bias=None):
    return texture(samp, uv, bias)


def textureLod(samp, uv, lod):
    if samp is None:
        return np.zeros((lanes(np.asarray(uv)), 4), F32)
    c = ctx()
    u = np.asarray(uv, F32)
    l = np.asarray(lod, F32)
    n = max(u.shape[0], l.shape[0] if l.ndim else 1)
    return samp.sample(bc(u, n)[:, 0], bc(u, n)[:, 1], filt='TRILINEAR',
                       wrap=c.wrap, lod=bc(np.atleast_1d(l), n)).astype(F32)


def tex2D(samp, uv):
    return texture(samp, uv)


def tex2Dlod(samp, uvlod):
    u = np.asarray(uvlod, F32)
    return textureLod(samp, u[:, :2], u[:, 3] if u.shape[1] > 3 else u[:, 2])


def texelFetch(samp, xy, lod=0):
    if samp is None:
        return np.zeros((lanes(np.asarray(xy)), 4), F32)
    p = np.asarray(xy)
    x = np.clip(p[:, 0].astype(np.int64), 0, samp.width - 1)
    y = np.clip(p[:, 1].astype(np.int64), 0, samp.height - 1)
    return samp.pixels[y, x].astype(F32)


def textureSize(samp, lod=0):
    if samp is None:
        return np.array([[1, 1]], np.int32)
    return np.array([[samp.width, samp.height]], np.int32)


def textureGrad(samp, uv, dx, dy):
    from ..core.texture import compute_lod
    if samp is None:
        return np.zeros((lanes(np.asarray(uv)), 4), F32)
    lod = compute_lod(np.asarray(dx, F32), np.asarray(dy, F32), samp.width, samp.height)
    return textureLod(samp, uv, lod)


# -------------------------------------------------------------- derivatives


def _scatter_diff(v, axis):
    """True screen-space derivative via the pixel coordinates in the context."""
    c = ctx()
    a = np.asarray(v, F32)
    if c.px is None or c.width <= 0:
        return np.zeros_like(a)
    if a.ndim == 1:
        a2 = a[:, None]
    else:
        a2 = a
    n = a2.shape[0]
    if n != len(c.px):
        return np.zeros_like(a)
    comps = a2.shape[1]
    img = np.zeros((c.height, c.width, comps), F32)
    valid = np.zeros((c.height, c.width), bool)
    img[c.py, c.px] = a2
    valid[c.py, c.px] = True
    if axis == 0:
        d = np.zeros_like(img)
        d[:, :-1] = img[:, 1:] - img[:, :-1]
        ok = np.zeros_like(valid)
        ok[:, :-1] = valid[:, 1:] & valid[:, :-1]
    else:
        d = np.zeros_like(img)
        d[:-1, :] = img[1:, :] - img[:-1, :]
        ok = np.zeros_like(valid)
        ok[:-1, :] = valid[1:, :] & valid[:-1, :]
    d[~ok] = 0.0
    out = d[c.py, c.px]
    return out[:, 0] if a.ndim == 1 else out


def dFdx(v):
    return _scatter_diff(v, 0)


def dFdy(v):
    return _scatter_diff(v, 1)


def fwidth(v):
    return np.abs(dFdx(v)) + np.abs(dFdy(v))


# ------------------------------------------------------ noise / era helpers


def _hash3(p):
    x = np.sin(p[:, 0] * 127.1 + p[:, 1] * 311.7 + p[:, 2] * 74.7) * 43758.5453
    return x - np.floor(x)


def _grad(ix, iy, iz, fx, fy, fz):
    h = np.sin(ix * 127.1 + iy * 311.7 + iz * 74.7) * 43758.5453
    h = h - np.floor(h)
    u = h * 6.2831853
    h2 = np.sin(ix * 269.5 + iy * 183.3 + iz * 246.1) * 43758.5453
    h2 = h2 - np.floor(h2)
    gz = h2 * 2.0 - 1.0
    r = np.sqrt(np.maximum(1.0 - gz * gz, 0.0))
    return np.cos(u) * r * fx + np.sin(u) * r * fy + gz * fz


def noise3(p):
    """Gradient (Perlin-style) noise in -1..1."""
    p = np.asarray(p, F32)
    if p.ndim == 1:
        p = np.stack([p, np.zeros_like(p), np.zeros_like(p)], axis=1)
    if p.shape[1] == 2:
        p = np.concatenate([p, np.zeros((p.shape[0], 1), F32)], axis=1)
    i = np.floor(p)
    f = p - i
    w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    acc = np.zeros(p.shape[0], F32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                gx = i[:, 0] + dx
                gy = i[:, 1] + dy
                gz = i[:, 2] + dz
                g = _grad(gx, gy, gz, f[:, 0] - dx, f[:, 1] - dy, f[:, 2] - dz)
                wx = w[:, 0] if dx else 1.0 - w[:, 0]
                wy = w[:, 1] if dy else 1.0 - w[:, 1]
                wz = w[:, 2] if dz else 1.0 - w[:, 2]
                acc += g * wx * wy * wz
    return np.clip(acc * 1.5, -1.0, 1.0)


def noise(p):
    return noise3(p) * 0.5 + 0.5


def fbm(p, octaves=4.0, lacunarity=2.0, gain=0.5):
    p = np.asarray(p, F32)
    oc = int(np.clip(np.max(np.atleast_1d(octaves)), 1, 8))
    lac = float(np.max(np.atleast_1d(lacunarity)))
    g = float(np.max(np.atleast_1d(gain)))
    total = np.zeros(p.shape[0], F32)
    amp = 1.0
    freq = 1.0
    norm = 0.0
    for _ in range(oc):
        total += noise3(p * freq) * amp
        norm += amp
        amp *= g
        freq *= lac
    return total / max(norm, 1e-6)


def quantize(v, steps):
    s = np.maximum(np.asarray(steps, F32), 1.0)
    v = np.asarray(v, F32)
    if v.ndim == 2 and np.ndim(s) == 1:
        s = s[:, None]
    return np.floor(v * s) / s


def posterize(v, steps):
    return quantize(v, steps)


BAYER4 = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
                   F32) + 0.5) / 16.0


def dither4x4(v, coord=None):
    """Ordered-dither a 0..1 value using the fragment's screen position."""
    c = ctx()
    v = np.asarray(v, F32)
    if coord is not None:
        p = np.asarray(coord, F32)
        xi = p[:, 0].astype(np.int64) & 3
        yi = p[:, 1].astype(np.int64) & 3
    elif c.px is not None:
        xi = np.asarray(c.px) & 3
        yi = np.asarray(c.py) & 3
    else:
        return v
    t = BAYER4[yi, xi]
    if v.ndim == 2:
        t = t[:, None]
    return (v > t).astype(F32)


def hsv2rgb(v):
    from ..core.mathx import hsv_to_rgb
    v = np.asarray(v, F32)
    r, g, b = hsv_to_rgb(v[:, 0], v[:, 1], v[:, 2])
    return np.stack([r, g, b], axis=1).astype(F32)


def rgb2hsv(v):
    from ..core.mathx import rgb_to_hsv
    v = np.asarray(v, F32)
    h, s, x = rgb_to_hsv(v[:, 0], v[:, 1], v[:, 2])
    return np.stack([h, s, x], axis=1).astype(F32)


def luminance(v):
    v = np.asarray(v, F32)
    return v[:, 0] * 0.299 + v[:, 1] * 0.587 + v[:, 2] * 0.114


def mul(a, b):
    """HLSL mul(): dispatches on the operand shapes."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 3 and b.ndim == 3:
        return mat_mul_mat(a, b)
    if a.ndim == 3 and b.ndim == 2:
        return mat_mul_vec(a, b)
    if a.ndim == 2 and b.ndim == 3:
        return vec_mul_mat(a, b)
    return a * b


def clip_(x):
    return x


def rsqrt(x):
    return inversesqrt(x)


def frac(x):
    return fract(x)


def ddx(x):
    return dFdx(x)


def ddy(x):
    return dFdy(x)


def mad(a, b, c):
    return a * b + c


def asfloat(x):
    return np.asarray(x, F32)


def asint(x):
    return np.asarray(x).astype(np.int32)


# --------------------------------------------------- signature / type table
#
# rule: callable(argtypes) -> GType

def _same(a):
    return a[0]


def _scalar_of(a):
    return a[0].scalar().with_base('float') if a else FLOAT


def _float(a):
    return FLOAT


def _bool_vec(a):
    t = a[0]
    return GType('bool', t.n, 0)


def _bool(a):
    return BOOL


def _vec3(a):
    return VEC3


def _vec4(a):
    return VEC4


def _mat_from(a):
    return a[0]


def _promote2(a):
    from .gtypes import promote
    return promote(a[0], a[1]) if len(a) > 1 else a[0]


def _mul_rule(a):
    x, y = a[0], a[1]
    if x.is_matrix and y.is_matrix:
        return x
    if x.is_matrix:
        return GType('float', x.rows)
    if y.is_matrix:
        return GType('float', y.n)
    return _promote2(a)


BUILTINS = {
    # name: (runtime, rule, min_args, max_args)
    'radians': ('radians', _same, 1, 1), 'degrees': ('degrees', _same, 1, 1),
    'sin': ('sin', _same, 1, 1), 'cos': ('cos', _same, 1, 1),
    'tan': ('tan', _same, 1, 1), 'asin': ('asin', _same, 1, 1),
    'acos': ('acos', _same, 1, 1), 'atan': ('atan', _same, 1, 2),
    'atan2': ('atan2', _same, 2, 2),
    'sinh': ('sinh', _same, 1, 1), 'cosh': ('cosh', _same, 1, 1),
    'tanh': ('tanh', _same, 1, 1), 'asinh': ('asinh', _same, 1, 1),
    'acosh': ('acosh', _same, 1, 1), 'atanh': ('atanh', _same, 1, 1),
    'pow': ('pow_', _same, 2, 2), 'exp': ('exp', _same, 1, 1),
    'log': ('log', _same, 1, 1), 'log10': ('log10', _same, 1, 1),
    'exp2': ('exp2', _same, 1, 1), 'log2': ('log2', _same, 1, 1),
    'sqrt': ('sqrt', _same, 1, 1), 'inversesqrt': ('inversesqrt', _same, 1, 1),
    'rsqrt': ('rsqrt', _same, 1, 1),
    'abs': ('abs_', _same, 1, 1), 'sign': ('sign', _same, 1, 1),
    'floor': ('floor', _same, 1, 1), 'ceil': ('ceil', _same, 1, 1),
    'trunc': ('trunc', _same, 1, 1), 'round': ('round_', _same, 1, 1),
    'roundEven': ('roundEven', _same, 1, 1),
    'fract': ('fract', _same, 1, 1), 'frac': ('frac', _same, 1, 1),
    'mod': ('mod', _same, 2, 2), 'fmod': ('fmod', _same, 2, 2),
    'min': ('min_', _promote2, 2, 2), 'max': ('max_', _promote2, 2, 2),
    'clamp': ('clamp', _same, 3, 3), 'saturate': ('saturate', _same, 1, 1),
    'mix': ('mix', _promote2, 3, 3), 'lerp': ('lerp', _promote2, 3, 3),
    'step': ('step', lambda a: a[1], 2, 2),
    'smoothstep': ('smoothstep', lambda a: a[2], 3, 3),
    'isnan': ('isnan', _bool_vec, 1, 1), 'isinf': ('isinf', _bool_vec, 1, 1),
    'length': ('length', _float, 1, 1), 'distance': ('distance', _float, 2, 2),
    'dot': ('dot_', _float, 2, 2), 'cross': ('cross', _vec3, 2, 2),
    'normalize': ('normalize', _same, 1, 1),
    'faceforward': ('faceforward', _same, 3, 3),
    'reflect': ('reflect', _same, 2, 2), 'refract': ('refract', _same, 3, 3),
    'matrixCompMult': ('matrixCompMult', _same, 2, 2),
    'outerProduct': ('outerProduct', lambda a: GType('float', a[1].n, a[0].n), 2, 2),
    'transpose': ('transpose', lambda a: GType('float', a[0].rows, a[0].n), 1, 1),
    'determinant': ('determinant', _float, 1, 1),
    'inverse': ('inverse', _same, 1, 1),
    'mul': ('mul', _mul_rule, 2, 2), 'mad': ('mad', _same, 3, 3),
    'lessThan': ('lessThan', _bool_vec, 2, 2),
    'lessThanEqual': ('lessThanEqual', _bool_vec, 2, 2),
    'greaterThan': ('greaterThan', _bool_vec, 2, 2),
    'greaterThanEqual': ('greaterThanEqual', _bool_vec, 2, 2),
    'equal': ('equal', _bool_vec, 2, 2), 'notEqual': ('notEqual', _bool_vec, 2, 2),
    'any': ('any_comp', _bool, 1, 1), 'all': ('all_comp', _bool, 1, 1),
    'not': ('not_', _bool_vec, 1, 1),
    'texture': ('texture', _vec4, 2, 3), 'texture2D': ('texture2D', _vec4, 2, 3),
    'textureLod': ('textureLod', _vec4, 3, 3),
    'texture2DLod': ('textureLod', _vec4, 3, 3),
    'textureGrad': ('textureGrad', _vec4, 4, 4),
    'texelFetch': ('texelFetch', _vec4, 2, 3),
    'textureSize': ('textureSize', lambda a: GType('int', 2), 1, 2),
    'tex2D': ('tex2D', _vec4, 2, 2), 'tex2Dlod': ('tex2Dlod', _vec4, 2, 2),
    'dFdx': ('dFdx', _same, 1, 1), 'dFdy': ('dFdy', _same, 1, 1),
    'ddx': ('ddx', _same, 1, 1), 'ddy': ('ddy', _same, 1, 1),
    'fwidth': ('fwidth', _same, 1, 1),
    'clip': ('clip_', _same, 1, 1),
    'asfloat': ('asfloat', lambda a: a[0].with_base('float'), 1, 1),
    'asint': ('asint', lambda a: a[0].with_base('int'), 1, 1),
    # Halcyon extensions
    'noise': ('noise', _float, 1, 1), 'noise3': ('noise3', _float, 1, 1),
    'fbm': ('fbm', _float, 1, 4),
    'quantize': ('quantize', _same, 2, 2), 'posterize': ('posterize', _same, 2, 2),
    'dither4x4': ('dither4x4', _same, 1, 2),
    'hsv2rgb': ('hsv2rgb', _vec3, 1, 1), 'rgb2hsv': ('rgb2hsv', _vec3, 1, 1),
    'luminance': ('luminance', _float, 1, 1),
}


# ------------------------------------------------- operators used by codegen

I32 = np.int32


def _op_align(a, b):
    """Align a binary operand pair: vec op scalar splats the scalar."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim == 3 or b.ndim == 3:
        if a.ndim == 3 and b.ndim == 3:
            return a, b
        if a.ndim == 3:
            return a, (b[:, None, None] if b.ndim == 1 else b)
        return (a[:, None, None] if a.ndim == 1 else a), b
    if a.ndim == 2 and b.ndim == 2:
        if a.shape[1] != b.shape[1]:
            if a.shape[1] == 1:
                a = np.repeat(a, b.shape[1], axis=1)
            elif b.shape[1] == 1:
                b = np.repeat(b, a.shape[1], axis=1)
        return a, b
    if a.ndim == 2 and b.ndim <= 1:
        return a, splat(b, a.shape[1])
    if b.ndim == 2 and a.ndim <= 1:
        return splat(a, b.shape[1]), b
    return a, b


def add_c(a, b):
    a, b = _op_align(a, b)
    return a + b


def sub_c(a, b):
    a, b = _op_align(a, b)
    return a - b


def mul_c(a, b):
    a, b = _op_align(a, b)
    return a * b


def div_c(a, b):
    a, b = _op_align(a, b)
    b = np.where(np.abs(b) < 1e-30, np.sign(b) * 1e-30 + 1e-30, b)
    return a / b


def imod(a, b):
    a, b = _op_align(a, b)
    bi = np.where(b == 0, 1, b)
    return np.mod(a, bi)


def idx_axis(v, i, axis=1):
    """Dynamic component / column index."""
    a = np.asarray(v)
    idx = np.asarray(i)
    if idx.ndim == 0:
        return np.take(a, int(idx), axis=axis)
    n = max(a.shape[0], idx.shape[0])
    a = bc(a, n)
    idx = np.clip(bc(idx, n).astype(np.int64), 0, a.shape[axis] - 1)
    if a.ndim == 2:
        return a[np.arange(n), idx]
    return a[np.arange(n), idx, :]


def idx_list(lst, i):
    idx = np.asarray(i)
    if idx.ndim == 0:
        return lst[int(idx) % len(lst)]
    out = lst[0]
    for k in range(1, len(lst)):
        out = sel(idx == k, lst[k], out)
    return out


def struct_set(d, field, value):
    out = dict(d)
    out[field] = value
    return out


def col_set(m, c, value):
    a = np.array(np.asarray(m), copy=True)
    n = max(a.shape[0], lanes(np.asarray(value)))
    a = np.array(bc(a, n), copy=True)
    a[:, c, :] = bc(np.asarray(value, F32), n)
    return a
