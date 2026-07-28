"""Vector math helpers operating on flat (N,3) / (N,) numpy arrays.

Pure numpy. No bpy imports anywhere under halcyon.core -- this keeps the
renderer testable outside Blender and lets the whole pipeline be exercised
by the test harness.
"""

import numpy as np

EPS = 1e-8


def v3(x, y, z, n=None):
    """Build an (N,3) array from scalars or arrays."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    z = np.asarray(z, dtype=np.float32)
    if n is None:
        n = max(x.size, y.size, z.size)
    out = np.empty((n, 3), dtype=np.float32)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = z
    return out


def broadcast3(a, n):
    """Coerce a scalar / (3,) / (N,3) into (N,3)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 0:
        return np.repeat(a.reshape(1, 1), 3, axis=1).repeat(n, axis=0)
    if a.ndim == 1:
        if a.shape[0] == 3:
            return np.broadcast_to(a.reshape(1, 3), (n, 3)).copy()
        return np.repeat(a.reshape(-1, 1), 3, axis=1)
    return a


def broadcast1(a, n):
    """Coerce a scalar / (N,) / (N,1) / (N,3) into (N,)."""
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 0:
        return np.full(n, float(a), dtype=np.float32)
    if a.ndim == 2:
        if a.shape[1] == 1:
            return a[:, 0]
        return a.mean(axis=1)
    if a.shape[0] == 1:
        return np.full(n, float(a[0]), dtype=np.float32)
    return a


def dot(a, b, keepdims=False):
    return np.sum(a * b, axis=-1, keepdims=keepdims)


def length(a, keepdims=False):
    return np.sqrt(np.maximum(np.sum(a * a, axis=-1, keepdims=keepdims), 0.0))


def normalize(a):
    ln = length(a, keepdims=True)
    return a / np.maximum(ln, EPS)


def cross(a, b):
    return np.cross(a, b)


def reflect(i, n):
    """GLSL reflect(): i - 2*dot(n,i)*n  (i points *toward* the surface)."""
    return i - 2.0 * dot(i, n, keepdims=True) * n


def refract(i, n, eta):
    """GLSL refract(). eta may be scalar or (N,1)."""
    eta = np.asarray(eta, dtype=np.float32)
    if eta.ndim == 1:
        eta = eta[:, None]
    ndi = dot(n, i, keepdims=True)
    k = 1.0 - eta * eta * (1.0 - ndi * ndi)
    out = eta * i - (eta * ndi + np.sqrt(np.maximum(k, 0.0))) * n
    return np.where(k < 0.0, 0.0, out)


def faceforward(n, i, nref=None):
    if nref is None:
        nref = n
    return np.where(dot(nref, i, keepdims=True) < 0.0, n, -n)


def saturate(x):
    return np.clip(x, 0.0, 1.0)


def mix(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = saturate((x - e0) / np.maximum(e1 - e0, EPS))
    return t * t * (3.0 - 2.0 * t)


def luminance(rgb, coeffs=(0.2126, 0.7152, 0.0722)):
    c = np.asarray(coeffs, dtype=np.float32)
    return rgb[..., 0] * c[0] + rgb[..., 1] * c[1] + rgb[..., 2] * c[2]


def luminance_601(rgb):
    """NTSC / Rec.601 luma -- what 90s software actually used."""
    return luminance(rgb, (0.299, 0.587, 0.114))


def transform_points(m, p):
    """m: (4,4) row-major numpy. p: (N,3). Returns (N,3)."""
    m = np.asarray(m, dtype=np.float32)
    return p @ m[:3, :3].T + m[:3, 3]


def transform_dirs(m, v):
    m = np.asarray(m, dtype=np.float32)
    return v @ m[:3, :3].T


def transform_normals(m_inv, n):
    """Normal matrix = transpose(inverse(M)); pass the inverse in."""
    m_inv = np.asarray(m_inv, dtype=np.float32)
    return n @ m_inv[:3, :3]


def transform_h(m, p):
    """Homogeneous transform. p: (N,3) -> (N,4)."""
    m = np.asarray(m, dtype=np.float32)
    n = p.shape[0]
    ph = np.empty((n, 4), dtype=np.float32)
    ph[:, :3] = p
    ph[:, 3] = 1.0
    return ph @ m.T


def orthonormal_basis(n):
    """Build a tangent frame around unit normals n: (N,3) -> (t, b)."""
    up = np.where(np.abs(n[:, 2:3]) < 0.999, np.array([[0.0, 0.0, 1.0]], np.float32),
                  np.array([[1.0, 0.0, 0.0]], np.float32))
    t = normalize(cross(up, n))
    b = cross(n, t)
    return t, b


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float32)
    return np.where(c <= 0.04045, c / 12.92, np.power(np.maximum(c + 0.055, 0.0) / 1.055, 2.4))


def linear_to_srgb(c):
    c = np.asarray(c, dtype=np.float32)
    c = np.maximum(c, 0.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1.0 / 2.4) - 0.055)


def gamma_encode(c, g):
    if abs(g - 1.0) < 1e-6:
        return c
    return np.power(np.maximum(c, 0.0), 1.0 / g)


def gamma_decode(c, g):
    if abs(g - 1.0) < 1e-6:
        return c
    return np.power(np.maximum(c, 0.0), g)


def hsv_to_rgb(h, s, v):
    h = np.asarray(h, np.float32) % 1.0
    s = np.clip(np.asarray(s, np.float32), 0.0, 1.0)
    v = np.asarray(v, np.float32)
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = (i.astype(np.int32) % 6)
    r = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [v, q, p, p, t, v])
    g = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [t, v, v, q, p, p])
    b = np.select([i == 0, i == 1, i == 2, i == 3, i == 4, i == 5], [p, p, t, v, v, q])
    return r, g, b


def rgb_to_hsv(r, g, b):
    r = np.asarray(r, np.float32)
    g = np.asarray(g, np.float32)
    b = np.asarray(b, np.float32)
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    s = np.where(mx > 0, d / np.maximum(mx, EPS), 0.0)
    h = np.zeros_like(mx)
    dd = np.maximum(d, EPS)
    h = np.where(mx == r, ((g - b) / dd) % 6.0, h)
    h = np.where(mx == g, ((b - r) / dd) + 2.0, h)
    h = np.where(mx == b, ((r - g) / dd) + 4.0, h)
    h = np.where(d <= 0, 0.0, h / 6.0) % 1.0
    return h, s, mx


def safe_pow(x, e):
    return np.power(np.maximum(x, 0.0), e)


def look_at(eye, target, up=(0, 0, 1)):
    """Build a Blender-style camera matrix_world (camera looks down -Z)."""
    eye = np.asarray(eye, np.float32)
    target = np.asarray(target, np.float32)
    up = np.asarray(up, np.float32)
    fwd = target - eye
    fwd = fwd / max(float(np.linalg.norm(fwd)), EPS)
    right = np.cross(fwd, up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-5:
        right = np.array([1.0, 0.0, 0.0], np.float32)
    else:
        right = right / rn
    trueup = np.cross(right, fwd)
    m = np.eye(4, dtype=np.float32)
    m[:3, 0] = right
    m[:3, 1] = trueup
    m[:3, 2] = -fwd
    m[:3, 3] = eye
    return m


def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / np.tan(fovy * 0.5)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = (2.0 * zfar * znear) / (znear - zfar)
    m[3, 2] = -1.0
    return m


def orthographic(scale, aspect, znear, zfar):
    m = np.zeros((4, 4), dtype=np.float32)
    r = scale * 0.5 * aspect
    t = scale * 0.5
    m[0, 0] = 1.0 / r
    m[1, 1] = 1.0 / t
    m[2, 2] = -2.0 / (zfar - znear)
    m[2, 3] = -(zfar + znear) / (zfar - znear)
    m[3, 3] = 1.0
    return m
