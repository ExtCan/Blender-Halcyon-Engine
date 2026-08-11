"""Evaluate a serialised Blender shader node tree, per fragment, on flat arrays.

The exporter turns a `bpy` node tree into plain dicts (see halcyon/export.py);
this module walks that graph pull-style with memoisation. Colour ramps and curve
widgets arrive pre-sampled as 256-entry LUTs, which is exact and avoids
reimplementing Blender's curve interpolation.

Shader (closure) sockets carry a `Closure`: a weighted list of components. The
renderer collapses that into one of its period reflectance models -- so a tree
built for Cycles still renders, translated rather than ignored.
"""

import os

import numpy as np

# Raise on a node failure instead of falling back to pass-through. The fallback
# still produces plausible-looking output, which is how a crash inside the Noise
# texture stayed invisible through three rounds of testing. Set from the add-on
# preferences, or with HALCYON_DEBUG=1 when running headless.
STRICT = False

from . import patterns as PT

from . import mathx as M
from .texture import Texture, env_equirect_uv, env_sphere_uv

VALUE, RGBA, VECTOR, SHADER = 'VALUE', 'RGBA', 'VECTOR', 'SHADER'


# --------------------------------------------------------------- closures

class Closure:
    """A weighted sum of surface components."""

    __slots__ = ('items',)

    def __init__(self, items=None):
        self.items = items or []

    def add(self, kind, weight, **params):
        self.items.append((kind, weight, params))
        return self

    def scaled(self, w):
        out = Closure()
        for kind, wt, p in self.items:
            out.items.append((kind, wt * w if np.ndim(wt) == np.ndim(w) else wt * w, p))
        return out

    def __add__(self, other):
        return Closure(self.items + other.items)

    def __len__(self):
        return len(self.items)


class ShadeContext:
    """Everything a node can ask about the fragment being shaded."""

    def __init__(self, n):
        self.n = n
        z3 = np.zeros((n, 3), np.float32)
        self.P = z3.copy()              # world position
        self.N = z3.copy()              # shading normal
        self.Ng = z3.copy()             # geometric normal
        self.T = None                   # tangent
        self.I = z3.copy()              # incident (camera -> surface)
        self.uv = np.zeros((n, 2), np.float32)
        self.uv2 = np.zeros((n, 2), np.float32)
        self.vcol = np.ones((n, 4), np.float32)
        self.object_loc = z3.copy()
        self.object_color = np.ones((n, 4), np.float32)
        self.object_index = np.zeros(n, np.float32)
        self.object_random = np.zeros(n, np.float32)
        self.object_matrix_inv = None   # (n,4,4) or None
        self.generated = z3.copy()
        self.backfacing = np.zeros(n, np.float32)
        self.px = None
        self.py = None
        self.width = 0
        self.height = 0
        self.tri = None
        self.depth = np.zeros(n, np.float32)
        self.camera_pos = np.zeros(3, np.float32)
        self.view_matrix = None
        self.time = 0.0
        self.frame = 1
        self.random = np.zeros(n, np.float32)
        self.is_camera_ray = True
        self.ray_depth = 0
        self.settings = None
        self.images = {}
        self.attributes = {}
        self.duv = None
        self.dvv = None
        self._object_matrix_inv = None
        self._obj_mats = None
        self._obj_idx = None

    @property
    def object_matrix_inv(self):
        """Built on first use: gathering a 4x4 per fragment costs real memory
        and almost no graph asks for object space."""
        if self._object_matrix_inv is None and self._obj_mats is not None:
            self._object_matrix_inv = self._obj_mats[self._obj_idx]
        return self._object_matrix_inv

    @object_matrix_inv.setter
    def object_matrix_inv(self, v):
        self._object_matrix_inv = v


# ------------------------------------------------------------- conversions


def _is_const(v):
    """Socket defaults arrive as Python scalars/lists; per-point data is always
    an ndarray. That distinction is what tells a constant RGB triple apart from
    three shading points."""
    return not isinstance(v, np.ndarray)


def to_value(v, n):
    if v is None or isinstance(v, Closure):
        return np.zeros(n, np.float32)
    a = np.asarray(v, np.float32)
    if a.ndim == 0:
        return np.full(n, float(a), np.float32)
    if _is_const(v) and a.ndim == 1:
        if a.shape[0] >= 4:
            return np.full(n, float(M.luminance(a[None, :3])[0]), np.float32)
        if a.shape[0] == 3:
            return np.full(n, float(a.mean()), np.float32)
        return np.full(n, float(a[0]) if a.size else 0.0, np.float32)
    if a.ndim == 1:
        return _fit(a, n)
    if a.shape[1] >= 3:
        if a.shape[1] == 4:
            return _fit(M.luminance(a[:, :3]), n)
        return _fit(a.mean(axis=1), n)
    return _fit(a[:, 0], n)


def to_color(v, n):
    if v is None or isinstance(v, Closure):
        return np.zeros((n, 4), np.float32)
    a = np.asarray(v, np.float32)
    if a.ndim == 0:
        out = np.empty((1, 4), np.float32)
        out[0, :3] = float(a)
        out[0, 3] = 1.0
        return np.broadcast_to(out, (n, 4)).copy()
    if _is_const(v) and a.ndim == 1:
        out = np.ones((1, 4), np.float32)
        if a.shape[0] >= 4:
            out[0] = a[:4]
        elif a.shape[0] == 3:
            out[0, :3] = a
        else:
            out[0, :3] = a[0] if a.size else 0.0
        return np.broadcast_to(out, (n, 4)).copy()
    if a.ndim == 1:
        out = np.ones((a.shape[0], 4), np.float32)
        out[:, 0] = a
        out[:, 1] = a
        out[:, 2] = a
        return _fit(out, n)
    if a.shape[1] == 4:
        return _fit(a, n)
    if a.shape[1] == 3:
        return _fit(np.concatenate([a, np.ones((a.shape[0], 1), np.float32)], 1), n)
    return _fit(np.repeat(a[:, :1], 4, axis=1), n)


def to_vector(v, n):
    if v is None or isinstance(v, Closure):
        return np.zeros((n, 3), np.float32)
    a = np.asarray(v, np.float32)
    if a.ndim == 0:
        return np.full((n, 3), float(a), np.float32)
    if _is_const(v) and a.ndim == 1:
        out = np.zeros((1, 3), np.float32)
        if a.shape[0] >= 3:
            out[0] = a[:3]
        else:
            out[0, :a.shape[0]] = a
        return np.broadcast_to(out, (n, 3)).copy()
    if a.ndim == 1:
        return _fit(np.repeat(a[:, None], 3, axis=1), n)
    if a.shape[1] >= 3:
        return _fit(a[:, :3], n)
    return _fit(np.concatenate([a, np.zeros((a.shape[0], 3 - a.shape[1]),
                                            np.float32)], 1), n)


def _fit(a, n):
    if a.shape[0] == n:
        return a
    if a.shape[0] == 1:
        return np.broadcast_to(a, (n,) + a.shape[1:]).copy()
    if a.shape[0] > n:
        return a[:n]
    pad = np.repeat(a[-1:], n - a.shape[0], axis=0)
    return np.concatenate([a, pad], axis=0)


def coerce(v, kind, n):
    if kind == VALUE:
        return to_value(v, n)
    if kind == VECTOR:
        return to_vector(v, n)
    if kind == SHADER:
        return v if isinstance(v, Closure) else Closure()
    return to_color(v, n)


# ------------------------------------------------------------------ helpers


def lut_eval(lut, t):
    """Sample a pre-baked LUT (K,C) at t in 0..1."""
    lut = np.asarray(lut, np.float32)
    k = lut.shape[0]
    x = np.clip(np.asarray(t, np.float32), 0.0, 1.0) * (k - 1)
    i0 = np.floor(x).astype(np.int32)
    i1 = np.minimum(i0 + 1, k - 1)
    f = (x - i0)
    a = lut[i0]
    b = lut[i1]
    if lut.ndim == 1:
        return a + (b - a) * f
    return a + (b - a) * f[:, None]


def hash3(x, y, z):
    h = np.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453123
    return h - np.floor(h)


def _perlin_grad(ix, iy, iz, fx, fy, fz):
    h = hash3(ix, iy, iz)
    ang = h * 6.283185307
    h2 = hash3(ix + 17.13, iy - 9.7, iz + 3.31)
    gz = h2 * 2.0 - 1.0
    r = np.sqrt(np.maximum(1.0 - gz * gz, 0.0))
    return np.cos(ang) * r * fx + np.sin(ang) * r * fy + gz * fz


def perlin(p):
    p = np.asarray(p, np.float32)
    i = np.floor(p)
    f = p - i
    w = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    acc = np.zeros(p.shape[0], np.float32)
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                g = _perlin_grad(i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz,
                                 f[:, 0] - dx, f[:, 1] - dy, f[:, 2] - dz)
                wx = w[:, 0] if dx else 1.0 - w[:, 0]
                wy = w[:, 1] if dy else 1.0 - w[:, 1]
                wz = w[:, 2] if dz else 1.0 - w[:, 2]
                acc += g * wx * wy * wz
    return np.clip(acc * 1.4, -1.0, 1.0)


def fractal_noise(p, detail=2.0, roughness=0.5, lacunarity=2.0, distortion=0.0):
    p = np.asarray(p, np.float32)
    if distortion is not None and np.any(np.asarray(distortion) != 0.0):
        d = np.asarray(distortion, np.float32)
        if d.ndim == 0:
            d = np.full(p.shape[0], float(d), np.float32)
        p = p + np.stack([perlin(p + 13.5), perlin(p + 43.5), perlin(p + 71.5)],
                         axis=1) * d[:, None]
    det = float(np.max(np.atleast_1d(detail)))
    octaves = int(np.clip(np.floor(det), 0, 15))
    total = np.zeros(p.shape[0], np.float32)
    amp = 1.0
    norm = 0.0
    freq = 1.0
    rough = float(np.mean(np.atleast_1d(roughness)))
    lac = float(np.mean(np.atleast_1d(lacunarity)))
    for _ in range(octaves + 1):
        total += perlin(p * freq) * amp
        norm += amp
        amp *= rough
        freq *= lac
    return (total / max(norm, 1e-6)) * 0.5 + 0.5


def voronoi(p, scale=1.0, randomness=1.0, metric='EUCLIDEAN', feature='F1',
            exponent=0.5):
    p = np.asarray(p, np.float32) * scale
    i = np.floor(p)
    f = p - i
    best = np.full(p.shape[0], 1e9, np.float32)
    second = np.full(p.shape[0], 1e9, np.float32)
    best_col = np.zeros((p.shape[0], 3), np.float32)
    best_pos = np.zeros((p.shape[0], 3), np.float32)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                cx = i[:, 0] + dx
                cy = i[:, 1] + dy
                cz = i[:, 2] + dz
                jx = hash3(cx, cy, cz)
                jy = hash3(cx + 3.7, cy + 1.3, cz + 9.1)
                jz = hash3(cx - 5.2, cy + 7.7, cz - 2.4)
                off = np.stack([dx + (jx - 0.5) * randomness + 0.5,
                                dy + (jy - 0.5) * randomness + 0.5,
                                dz + (jz - 0.5) * randomness + 0.5], axis=1) - f
                if metric == 'MANHATTAN':
                    d = np.abs(off).sum(axis=1)
                elif metric == 'CHEBYCHEV':
                    d = np.abs(off).max(axis=1)
                elif metric == 'MINKOWSKI':
                    e = max(float(np.mean(np.atleast_1d(exponent))), 0.1)
                    d = np.power(np.power(np.abs(off), e).sum(axis=1), 1.0 / e)
                else:
                    d = np.sqrt((off * off).sum(axis=1))
                closer = d < best
                second = np.where(closer, best, np.minimum(second, d))
                best = np.where(closer, d, best)
                col = np.stack([jx, jy, jz], axis=1)
                best_col = np.where(closer[:, None], col, best_col)
                best_pos = np.where(closer[:, None], off + f + i, best_pos)
    dist = second - best if feature == 'SMOOTH_F1' else (second if feature == 'F2' else best)
    return dist, best_col, best_pos / max(scale, 1e-6)


def blackbody(kelvin):
    """Planckian locus -> linear RGB (Neil Bartlett's approximation)."""
    t = np.clip(np.asarray(kelvin, np.float32), 800.0, 12000.0) / 100.0
    r = np.where(t <= 66, 255.0,
                 329.698727446 * np.power(np.maximum(t - 60, 1e-3), -0.1332047592))
    g = np.where(t <= 66,
                 99.4708025861 * np.log(np.maximum(t, 1e-3)) - 161.1195681661,
                 288.1221695283 * np.power(np.maximum(t - 60, 1e-3), -0.0755148492))
    b = np.where(t >= 66, 255.0,
                 np.where(t <= 19, 0.0,
                          138.5177312231 * np.log(np.maximum(t - 10, 1e-3)) - 305.0447927307))
    rgb = np.stack([r, g, b], axis=1) / 255.0
    return np.clip(rgb, 0.0, 1.0) ** 2.2


def wavelength_to_rgb(nm):
    nm = np.clip(np.asarray(nm, np.float32), 380.0, 780.0)
    r = np.zeros_like(nm)
    g = np.zeros_like(nm)
    b = np.zeros_like(nm)
    m = (nm >= 380) & (nm < 440)
    r = np.where(m, -(nm - 440) / 60.0, r)
    b = np.where(m, 1.0, b)
    m = (nm >= 440) & (nm < 490)
    g = np.where(m, (nm - 440) / 50.0, g)
    b = np.where(m, 1.0, b)
    m = (nm >= 490) & (nm < 510)
    g = np.where(m, 1.0, g)
    b = np.where(m, -(nm - 510) / 20.0, b)
    m = (nm >= 510) & (nm < 580)
    r = np.where(m, (nm - 510) / 70.0, r)
    g = np.where(m, 1.0, g)
    m = (nm >= 580) & (nm < 645)
    r = np.where(m, 1.0, r)
    g = np.where(m, -(nm - 645) / 65.0, g)
    m = nm >= 645
    r = np.where(m, 1.0, r)
    return np.clip(np.stack([r, g, b], axis=1), 0.0, 1.0)


# ------------------------------------------------------------- the evaluator


class GraphEvaluator:
    def __init__(self, graph, ctx, images=None, programs=None):
        self.graph = graph or {}
        self.nodes = self.graph.get('nodes', {})
        self.ctx = ctx
        self.n = ctx.n
        self.cache = {}
        self.images = images if images is not None else {}
        self.programs = programs if programs is not None else {}
        self.unsupported = set()
        self.errors = []
        self.depth = 0

    # ---------------------------------------------------------- socket fetch
    def input(self, node, name_or_index, kind=None):
        ins = node.get('inputs', [])
        sock = None
        if isinstance(name_or_index, int):
            if name_or_index < len(ins):
                sock = ins[name_or_index]
        else:
            for s in ins:
                if s.get('name') == name_or_index or \
                        s.get('identifier') == name_or_index:
                    sock = s
                    break
        if sock is None:
            return coerce(None, kind or VALUE, self.n)
        k = kind or sock.get('type', VALUE)
        link = sock.get('link')
        if link:
            v = self.eval_output(link[0], link[1])
            return coerce(v, k, self.n)
        return coerce(sock.get('default'), k, self.n)

    def has_link(self, node, name):
        for s in node.get('inputs', []):
            if s.get('name') == name or s.get('identifier') == name:
                return bool(s.get('link'))
        return False

    def eval_output(self, node_id, out_index):
        key = (node_id, out_index)
        if key in self.cache:
            return self.cache[key]
        node = self.nodes.get(node_id)
        if node is None:
            return None
        self.depth += 1
        if self.depth > 200:
            self.depth -= 1
            return None
        try:
            vals = self.eval_node(node)
        except Exception as exc:                             # noqa: BLE001
            if STRICT or os.environ.get('HALCYON_DEBUG'):
                raise
            # record *why*, not just that it happened: a node that raises falls
            # back to pass-through, which still produces plausible-looking
            # variation and can hide a real bug for a long time
            self.unsupported.add(
                f"{node.get('bl_idname', '?')}: {type(exc).__name__}: {exc}"[:160])
            self.errors.append((node.get('id'), node.get('bl_idname'), repr(exc)))
            vals = None
        finally:
            self.depth -= 1
        if vals is None:
            vals = {}
        outs = node.get('outputs', [])
        for i, o in enumerate(outs):
            v = vals.get(o.get('name')) if isinstance(vals, dict) else None
            if v is None and isinstance(vals, dict):
                v = vals.get(i)
            self.cache[(node_id, i)] = v
        return self.cache.get(key)

    def eval_node(self, node):
        fn = DISPATCH.get(node.get('bl_idname'))
        if fn is None:
            self.unsupported.add(node.get('bl_idname', '?'))
            return self.fallback(node)
        return fn(self, node)

    def fallback(self, node):
        """Unknown node: pass the first matching input through, else zeros."""
        out = {}
        ins = node.get('inputs', [])
        for o in node.get('outputs', []):
            k = o.get('type', VALUE)
            src = None
            for s in ins:
                if s.get('type') == k:
                    src = s
                    break
            if src is not None:
                link = src.get('link')
                v = self.eval_output(link[0], link[1]) if link else src.get('default')
                out[o.get('name')] = coerce(v, k, self.n)
            else:
                out[o.get('name')] = coerce(None, k, self.n)
        return out

    # -------------------------------------------------------------- surface
    def evaluate_surface(self):
        out_id = self.graph.get('output')
        if not out_id or out_id not in self.nodes:
            return None, None
        node = self.nodes[out_id]
        surf = self.input(node, 'Surface', SHADER)
        disp = None
        if self.has_link(node, 'Displacement'):
            disp = self.input(node, 'Displacement', VECTOR)
        return surf, disp


def _prop(node, name, default=None):
    return node.get('props', {}).get(name, default)


# ============================================================== input nodes


def n_tex_coord(ev, node):
    c = ev.ctx
    gen = c.generated
    obj = c.P
    if c.object_matrix_inv is not None:
        obj = np.einsum('nij,nj->ni', c.object_matrix_inv[:, :3, :3], c.P) + \
            c.object_matrix_inv[:, :3, 3]
    cam = c.P - c.camera_pos[None, :]
    win = np.zeros((c.n, 3), np.float32)
    if c.px is not None and c.width:
        win[:, 0] = (c.px + 0.5) / c.width
        win[:, 1] = (c.py + 0.5) / c.height
    refl = M.reflect(M.normalize(c.I), M.normalize(c.N))
    return {'Generated': gen, 'Normal': c.N, 'UV': np.concatenate(
        [c.uv, np.zeros((c.n, 1), np.float32)], 1), 'Object': obj,
        'Camera': cam, 'Window': win, 'Reflection': refl}


def n_uvmap(ev, node):
    c = ev.ctx
    name = _prop(node, 'uv_map', '')
    uv = c.attributes.get('uv:' + name, c.uv) if name else c.uv
    return {'UV': np.concatenate([uv, np.zeros((c.n, 1), np.float32)], 1)}


def n_geometry(ev, node):
    c = ev.ctx
    inc = M.normalize(c.I)
    return {'Position': c.P, 'Normal': c.N, 'Tangent': c.T if c.T is not None
            else M.orthonormal_basis(c.N)[0], 'True Normal': c.Ng,
            'Incoming': -inc, 'Parametric': np.concatenate(
                [c.uv, np.zeros((c.n, 1), np.float32)], 1),
            'Backfacing': c.backfacing, 'Pointiness': np.zeros(c.n, np.float32),
            'Random Per Island': c.random}


def n_object_info(ev, node):
    c = ev.ctx
    return {'Location': c.object_loc, 'Color': c.object_color,
            'Alpha': c.object_color[:, 3], 'Object Index': c.object_index,
            'Material Index': np.zeros(c.n, np.float32), 'Random': c.object_random}


def n_camera_data(ev, node):
    c = ev.ctx
    d = np.sqrt(np.maximum(((c.P - c.camera_pos[None, :]) ** 2).sum(1), 0.0))
    return {'View Vector': M.normalize(c.I), 'View Z Depth': c.depth, 'View Distance': d}


def n_wireframe(ev, node):
    """Distance-to-edge as a factor: 1 inside the wire width, 0 outside.

    Exact geometry on the fragment's own triangle, in world units or --
    with Pixel Size on -- in output pixels via the same perspective
    factors the mip footprint rides. The cel-ink look this feeds is
    exactly what the engine's own Wireframe Overlay draws post-hoc; the
    NODE puts it in the material where a graph can colour with it.
    """
    c = ev.ctx
    size = ev.input(node, 'Size', VALUE)
    supplier = getattr(c, 'wire_fields', None)
    if supplier is None:
        ev.unsupported.add('ShaderNodeWireframe (this shading point has no '
                           'triangle identity -- corner lighting reads 0)')
        return {'Fac': np.zeros(ev.n, np.float32)}
    cached = getattr(c, '_wire_cache', None)
    if cached is None:
        cached = supplier()
        c._wire_cache = cached
    dist, wpp = cached
    half = np.maximum(size, 0.0) * 0.5
    if _prop(node, 'use_pixel_size', False):
        half = half * wpp
    return {'Fac': (dist <= half).astype(np.float32)}


def n_vector_transform(ev, node):
    """World / camera / object conversions for points, vectors and normals.

    Camera space is the job's own view matrix; object space is the
    per-fragment inverse object matrix the exporter carries. Normals ride
    the inverse transpose and come back unit length, exactly the linear
    algebra everything else in the renderer uses.
    """
    c = ev.ctx
    v = ev.input(node, 'Vector', VECTOR)
    vt = str(_prop(node, 'vector_type', 'VECTOR'))
    src = str(_prop(node, 'convert_from', 'WORLD'))
    dst = str(_prop(node, 'convert_to', 'OBJECT'))
    if src == dst:
        return {'Vector': v}

    def to_world(space):
        if space == 'CAMERA':
            if c.view_matrix is None:
                return None
            return np.linalg.inv(np.asarray(c.view_matrix,
                                            np.float32))[None, :, :]
        if space == 'OBJECT':
            inv = c.object_matrix_inv
            if inv is None:
                ev.unsupported.add(
                    'ShaderNodeVectorTransform (no object matrices at this '
                    'shading point; object space reads as world)')
                return None
            return np.linalg.inv(inv)
        return None                                            # WORLD

    def from_world(space):
        if space == 'CAMERA':
            if c.view_matrix is None:
                return None
            return np.asarray(c.view_matrix, np.float32)[None, :, :]
        if space == 'OBJECT':
            inv = c.object_matrix_inv
            if inv is None:
                ev.unsupported.add(
                    'ShaderNodeVectorTransform (no object matrices at this '
                    'shading point; object space reads as world)')
                return None
            return inv
        return None                                            # WORLD

    A = to_world(src)
    B = from_world(dst)
    if A is None and B is None:
        return {'Vector': v}
    m = B if A is None else (A if B is None else B @ A)        # (k,4,4)
    rot = m[:, :3, :3]
    if vt == 'POINT':
        out = np.einsum('nij,nj->ni', np.broadcast_to(
            rot, (ev.n, 3, 3)), v) + np.broadcast_to(m[:, :3, 3], (ev.n, 3))
    elif vt == 'NORMAL':
        it = np.linalg.inv(rot).transpose(0, 2, 1)
        out = M.normalize(np.einsum('nij,nj->ni', np.broadcast_to(
            it, (ev.n, 3, 3)), v))
    else:
        out = np.einsum('nij,nj->ni', np.broadcast_to(
            rot, (ev.n, 3, 3)), v)
    return {'Vector': out.astype(np.float32)}


def n_ambient_occlusion(ev, node):
    """Per-material AO with REAL rays: the engine's own deterministic
    cosine-hemisphere sampler (a distinct hash salt, so a material's AO
    never correlates with the lighting pass's), against the same BVH the
    ray features use -- built through the content cache when no ray
    feature has built one yet. Sample count is the node's own; Distance
    is per fragment. `inside` flips the hemisphere; `only_local` cannot
    be honoured (the export merges objects into one mesh) and says so.
    """
    c = ev.ctx
    col = ev.input(node, 'Color', RGBA)
    dist = np.maximum(ev.input(node, 'Distance', VALUE), np.float32(1e-6))
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') \
        else c.N
    samples = max(int(_prop(node, 'samples', 16) or 16), 1)
    bvh = getattr(c, 'bvh', None)
    scene = getattr(c, 'scene', None)
    if bvh is None and scene is not None and \
            getattr(scene, 'mesh', None) is not None:
        from . import render as _render
        try:
            bvh = _render._cached_bvh(scene, scene.mesh)
        except Exception:                                       # noqa: BLE001
            bvh = None
    if bvh is None:
        ev.unsupported.add('ShaderNodeAmbientOcclusion (no geometry to '
                           'query at this shading point -- AO reads open)')
        ao = np.ones(ev.n, np.float32)
    else:
        from . import patterns as PT
        N = M.normalize(nrm)
        if _prop(node, 'inside', False):
            N = -N
        t, b = M.orthonormal_basis(N)
        st = c.settings
        bias = max(float(getattr(st, 'ray_bias', 1e-3) or 1e-3), 1e-4)
        origin = c.P + N * bias
        seed = int(getattr(st, 'seed', 0) or 0)
        if c.spx is not None:
            spx, spy = c.spx, c.spy
        elif c.tri is not None:
            spx = np.asarray(c.tri, np.int64)
            spy = np.zeros(ev.n, np.int64)
        else:
            spx = spy = np.zeros(ev.n, np.int64)
        occ = np.zeros(ev.n, np.float32)
        for k in range(samples):
            # salt 6151: prime, distinct from the light loop (131) and the
            # AO PASS (8389) -- a material's dirt mask must not move when
            # the lighting's own occlusion is toggled
            z = 2 * k + 6151 + 7919 * seed
            u1 = PT.sample_u(spx, spy, z)
            ca, sa = PT.sample_circle(PT.sample_u(spx, spy, z + 1))
            r = np.sqrt(u1)
            d = M.normalize(t * (r * ca)[:, None] + b * (r * sa)[:, None] +
                            N * np.sqrt(np.maximum(np.float32(1.0) - u1,
                                                   np.float32(0.0)))[:, None])
            occ += bvh.occluded(origin, d,
                                dist.astype(np.float32)).astype(np.float32)
        ao = np.clip(1.0 - occ / samples, 0.0, 1.0)
        if _prop(node, 'only_local', False):
            ev.unsupported.add('ShaderNodeAmbientOcclusion (Only Local: the '
                               'export merges objects, so the whole scene '
                               'occludes)')
    out = col.copy()
    out[:, :3] = out[:, :3] * ao[:, None]
    return {'Color': out, 'AO': ao}


def n_attribute(ev, node):
    c = ev.ctx
    name = _prop(node, 'attribute_name', '')
    v = c.attributes.get(name)
    if v is None:
        col = c.vcol
    else:
        col = to_color(v, c.n)
    return {'Color': col, 'Vector': col[:, :3], 'Fac': M.luminance(col[:, :3]),
            'Alpha': col[:, 3]}


def n_vertex_color(ev, node):
    c = ev.ctx
    name = _prop(node, 'layer_name', '')
    col = c.attributes.get('col:' + name, c.vcol) if name else c.vcol
    col = to_color(col, c.n)
    return {'Color': col, 'Alpha': col[:, 3]}


def n_fresnel(ev, node):
    c = ev.ctx
    ior = ev.input(node, 'IOR', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else c.N
    from .shading import fresnel_dielectric
    cosi = np.abs(M.dot(M.normalize(nrm), -M.normalize(c.I)))
    return {'Fac': fresnel_dielectric(cosi, np.maximum(ior, 1.0001))}


def n_layer_weight(ev, node):
    c = ev.ctx
    blend = np.clip(ev.input(node, 'Blend', VALUE), 0.0, 0.99999)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else c.N
    cosi = np.abs(M.dot(M.normalize(nrm), -M.normalize(c.I)))
    from .shading import fresnel_dielectric
    eta = np.where(blend < 0.5, 1.0 / np.maximum(1.0 - blend * 2.0, 1e-5),
                   1.0 + (blend - 0.5) * 2.0)
    fac = fresnel_dielectric(cosi, np.maximum(eta, 1.0001))
    b = np.where(blend < 0.5, blend * 2.0, 0.5 / np.maximum(1.0 - blend, 1e-5) * 0.5)
    facing = np.power(np.maximum(1.0 - cosi, 0.0),
                      np.where(blend < 0.5, 0.5 / np.maximum(blend, 1e-5),
                               2.0 * (1.0 - blend)))
    return {'Fresnel': fac, 'Facing': np.clip(facing, 0.0, 1.0)}


def n_light_path(ev, node):
    c = ev.ctx
    one = np.ones(c.n, np.float32)
    zero = np.zeros(c.n, np.float32)
    cam = one if c.is_camera_ray else zero
    return {'Is Camera Ray': cam, 'Is Shadow Ray': zero,
            'Is Diffuse Ray': one - cam, 'Is Glossy Ray': one - cam,
            'Is Singular Ray': zero, 'Is Reflection Ray': one - cam,
            'Is Transmission Ray': zero, 'Ray Length': c.depth,
            'Ray Depth': np.full(c.n, float(c.ray_depth), np.float32),
            'Diffuse Depth': zero, 'Glossy Depth': zero, 'Transparent Depth': zero,
            'Transmission Depth': zero}


def n_rgb(ev, node):
    return {'Color': coerce(_prop(node, 'value', [0.5, 0.5, 0.5, 1.0]), RGBA, ev.n)}


def n_value(ev, node):
    return {'Value': coerce(_prop(node, 'value', 0.5), VALUE, ev.n)}


def n_tangent(ev, node):
    c = ev.ctx
    t = c.T if c.T is not None else M.orthonormal_basis(c.N)[0]
    return {'Tangent': t}


# ============================================================ texture nodes


def _tex_vector(ev, node, default='generated'):
    c = ev.ctx
    if ev.has_link(node, 'Vector'):
        return ev.input(node, 'Vector', VECTOR)
    if default == 'uv':
        return np.concatenate([c.uv, np.zeros((c.n, 1), np.float32)], 1)
    return c.generated


def _get_image(ev, node):
    key = _prop(node, 'image')
    if key is None:
        return None
    return ev.images.get(key)


def n_tex_image(ev, node):
    c = ev.ctx
    tex = _get_image(ev, node)
    vec = _tex_vector(ev, node, 'uv')
    interp = _prop(node, 'interpolation', 'Linear')
    ext = _prop(node, 'extension', 'REPEAT')
    proj = _prop(node, 'projection', 'FLAT')
    st = c.settings
    filt = {'Closest': 'NEAREST', 'Linear': 'BILINEAR', 'Cubic': 'BILINEAR',
            'Smart': 'BILINEAR'}.get(interp, 'BILINEAR')
    if st is not None and getattr(st, 'tex_filter', None):
        filt = st.tex_filter if st.tex_filter != 'TRILINEAR' or tex is None else 'TRILINEAR'
    wrap = {'REPEAT': 'REPEAT', 'EXTEND': 'EXTEND', 'CLIP': 'CLIP',
            'MIRROR': 'MIRROR'}.get(ext, 'REPEAT')
    if tex is None:
        col = np.zeros((c.n, 4), np.float32)
        col[:, 3] = 1.0
        return {'Color': col, 'Alpha': np.ones(c.n, np.float32)}
    u, v = vec[:, 0], vec[:, 1]
    if proj == 'SPHERE':
        d = M.normalize(vec)
        u, v = env_equirect_uv(d)
    elif proj == 'TUBE':
        u = np.arctan2(vec[:, 1], vec[:, 0]) / (2 * np.pi) + 0.5
        v = vec[:, 2] * 0.5 + 0.5
    elif proj == 'BOX':
        u, v = _box_project(vec, c.N)
    lod = None
    if filt == 'TRILINEAR' and c.duv is not None \
            and not ev.has_link(node, 'Vector') and proj == 'FLAT':
        # the derivatives describe the RAW UV attribute: a linked Vector
        # chain or a non-flat projection resamples through a transform
        # the chain rule was never applied to, so those keep the top
        # level rather than filtering with the wrong footprint
        from .texture import compute_lod
        an = int(getattr(st, 'tex_aniso', 1) or 1) if st is not None else 1
        bias = float(getattr(st, 'tex_mip_bias', 0.0) or 0.0) \
            if st is not None else 0.0
        if an > 1:
            col = tex.sample(u, v, filt=filt, wrap=wrap, aniso=an,
                             duv=c.duv, dvv=c.dvv, bias=bias)
            return {'Color': col, 'Alpha': col[:, 3]}
        lod = compute_lod(c.duv, c.dvv, tex.width, tex.height, bias)
    col = tex.sample(u, v, filt=filt, wrap=wrap, lod=lod)
    return {'Color': col, 'Alpha': col[:, 3]}


def _box_project(vec, nrm):
    ax = np.abs(nrm)
    dom = np.argmax(ax, axis=1)
    u = np.where(dom == 0, vec[:, 1], np.where(dom == 1, vec[:, 0], vec[:, 0]))
    v = np.where(dom == 0, vec[:, 2], np.where(dom == 1, vec[:, 2], vec[:, 1]))
    return u, v


def n_tex_environment(ev, node):
    c = ev.ctx
    tex = _get_image(ev, node)
    vec = _tex_vector(ev, node)
    if tex is None:
        col = np.zeros((c.n, 4), np.float32)
        col[:, 3] = 1.0
        return {'Color': col}
    d = M.normalize(vec)
    if _prop(node, 'projection', 'EQUIRECTANGULAR') == 'MIRROR_BALL':
        u, v = env_sphere_uv(d)
    else:
        u, v = env_equirect_uv(d)
    interp = _prop(node, 'interpolation', 'Linear')
    filt = 'NEAREST' if interp == 'Closest' else 'BILINEAR'
    return {'Color': tex.sample(u, v, filt=filt, wrap='EXTEND')}


def n_tex_checker(ev, node):
    vec = _tex_vector(ev, node)
    c1 = ev.input(node, 'Color1', RGBA)
    c2 = ev.input(node, 'Color2', RGBA)
    scale = ev.input(node, 'Scale', VALUE)
    p = vec * scale[:, None]
    s = (np.floor(p[:, 0]).astype(np.int64) + np.floor(p[:, 1]).astype(np.int64) +
         np.floor(p[:, 2]).astype(np.int64)) % 2
    fac = (s == 0).astype(np.float32)
    return {'Color': np.where(fac[:, None] > 0.5, c1, c2), 'Fac': fac}


def n_tex_gradient(ev, node):
    vec = _tex_vector(ev, node)
    t = _prop(node, 'gradient_type', 'LINEAR')
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    if t == 'QUADRATIC':
        f = np.maximum(x, 0.0) ** 2
    elif t == 'EASING':
        r = np.clip(x, 0.0, 1.0)
        f = r * r * (3.0 - 2.0 * r)
    elif t == 'DIAGONAL':
        f = (x + y) * 0.5
    elif t == 'RADIAL':
        f = np.arctan2(y, x) / (2 * np.pi) + 0.5
    elif t in ('QUADRATIC_SPHERE', 'SPHERICAL'):
        r = np.sqrt(np.maximum((vec * vec).sum(1), 0.0))
        f = np.maximum(1.0 - r, 0.0)
        if t == 'QUADRATIC_SPHERE':
            f = f * f
    else:
        f = x
    fac = np.clip(f, 0.0, 1.0).astype(np.float32)
    col = np.stack([fac, fac, fac, np.ones_like(fac)], axis=1)
    return {'Fac': fac, 'Color': col}


def n_tex_noise(ev, node):
    """Noise Texture. `noise_type` selects the fractal, as Blender 4.1+ folded
    the old Musgrave node into this one; ignoring it left every type rendering
    as plain fBm."""
    vec = _tex_vector(ev, node)
    scale = ev.input(node, 'Scale', VALUE)
    detail = ev.input(node, 'Detail', VALUE)
    rough = ev.input(node, 'Roughness', VALUE)
    dist = ev.input(node, 'Distortion', VALUE)
    lac = ev.input(node, 'Lacunarity', VALUE) if any(
        s.get('name') == 'Lacunarity' for s in node.get('inputs', [])) else np.full(
        ev.n, 2.0, np.float32)
    p = vec * scale[:, None]
    f = fractal_noise(p, detail, rough, lac, dist)
    r = fractal_noise(p + 13.7, detail, rough, lac, dist)
    g = fractal_noise(p + 41.3, detail, rough, lac, dist)
    col = np.stack([f, r, g, np.ones_like(f)], axis=1)
    ntype = _prop(node, 'noise_type', 'FBM')
    if ntype != 'FBM':
        off = ev.input(node, 'Offset', VALUE) if any(
            s.get('name') == 'Offset' for s in node.get('inputs', [])) else \
            np.zeros(ev.n, np.float32)
        gain = ev.input(node, 'Gain', VALUE) if any(
            s.get('name') == 'Gain' for s in node.get('inputs', [])) else \
            np.ones(ev.n, np.float32)
        f = _fractal_variant(ntype, p, detail, rough, lac, off, gain)
        if _prop(node, 'normalize', True):
            f = np.clip(f * 0.5 + 0.5, 0.0, 1.0)
        col = np.stack([f, f, f, np.ones_like(f)], axis=1).astype(np.float32)
    return {'Fac': f.astype(np.float32), 'Color': col}


def n_tex_white_noise(ev, node):
    vec = _tex_vector(ev, node)
    v = hash3(vec[:, 0] * 97.3, vec[:, 1] * 57.7, vec[:, 2] * 13.9)
    col = np.stack([v, hash3(v * 3.1, v * 7.7, 1.3), hash3(v * 11.3, 2.7, v * 5.1),
                    np.ones_like(v)], axis=1)
    return {'Value': v, 'Color': col}


def n_tex_wave(ev, node):
    vec = _tex_vector(ev, node)
    scale = ev.input(node, 'Scale', VALUE)
    distortion = ev.input(node, 'Distortion', VALUE)
    detail = ev.input(node, 'Detail', VALUE)
    dscale = ev.input(node, 'Detail Scale', VALUE)
    wt = _prop(node, 'wave_type', 'BANDS')
    prof = _prop(node, 'wave_profile', 'SIN')
    direction = _prop(node, 'bands_direction', 'X')
    p = vec * scale[:, None]
    if wt == 'RINGS':
        n = np.sqrt(np.maximum((p * p).sum(1), 0.0))
    else:
        n = {'X': p[:, 0], 'Y': p[:, 1], 'Z': p[:, 2]}.get(
            direction, p[:, 0] + p[:, 1] + p[:, 2])
        if direction == 'DIAGONAL':
            n = p[:, 0] + p[:, 1] + p[:, 2]
    n = n * 20.0
    n = n + distortion * fractal_noise(p * dscale[:, None], detail, 0.5, 2.0, 0.0) * 20.0
    if prof == 'SAW':
        f = (n / (2 * np.pi)) % 1.0
    elif prof == 'TRI':
        f = np.abs(((n / np.pi) % 2.0) - 1.0)
    else:
        f = 0.5 + 0.5 * np.sin(n - np.pi * 0.5)
    f = np.clip(f, 0.0, 1.0).astype(np.float32)
    return {'Fac': f, 'Color': np.stack([f, f, f, np.ones_like(f)], axis=1)}


def n_tex_voronoi(ev, node):
    vec = _tex_vector(ev, node)
    scale = ev.input(node, 'Scale', VALUE)
    rand = ev.input(node, 'Randomness', VALUE)
    exp = ev.input(node, 'Exponent', VALUE)
    metric = _prop(node, 'distance', 'EUCLIDEAN')
    feature = _prop(node, 'feature', 'F1')
    d, col, pos = voronoi(vec, float(np.mean(scale)) if np.ndim(scale) else scale,
                          float(np.mean(rand)), metric, feature, float(np.mean(exp)))
    d = d.astype(np.float32)
    return {'Distance': d, 'Color': np.concatenate(
        [col, np.ones((col.shape[0], 1), np.float32)], 1), 'Position': pos,
        'Fac': d}


def n_tex_magic(ev, node):
    vec = _tex_vector(ev, node)
    scale = ev.input(node, 'Scale', VALUE)
    distort = ev.input(node, 'Distortion', VALUE)
    depth = int(_prop(node, 'turbulence_depth', 2))
    p = vec * scale[:, None]
    x = np.sin((p[:, 0] + p[:, 1] + p[:, 2]) * 5.0)
    y = np.cos((-p[:, 0] + p[:, 1] - p[:, 2]) * 5.0)
    z = -np.cos((-p[:, 0] - p[:, 1] + p[:, 2]) * 5.0)
    d = distort
    for i in range(max(depth, 1)):
        x2 = np.sin(y * d) * 0.5 + 0.5
        y2 = np.cos(z * d) * 0.5 + 0.5
        z2 = -np.cos(x * d) * 0.5 + 0.5
        x, y, z = x2 * 2 - 1, y2 * 2 - 1, z2 * 2 - 1
        d = d * 0.85 + 0.15
    col = np.stack([0.5 - x * 0.5, 0.5 - y * 0.5, 0.5 - z * 0.5,
                    np.ones_like(x)], axis=1).astype(np.float32)
    return {'Color': col, 'Fac': M.luminance(col[:, :3])}


def n_tex_brick(ev, node):
    vec = _tex_vector(ev, node)
    c1 = ev.input(node, 'Color1', RGBA)
    c2 = ev.input(node, 'Color2', RGBA)
    mortar = ev.input(node, 'Mortar', RGBA)
    scale = ev.input(node, 'Scale', VALUE)
    msize = ev.input(node, 'Mortar Size', VALUE)
    bias = ev.input(node, 'Bias', VALUE)
    bw = ev.input(node, 'Brick Width', VALUE)
    rh = ev.input(node, 'Row Height', VALUE)
    offset = float(_prop(node, 'offset', 0.5))
    freq = int(_prop(node, 'offset_frequency', 2))
    sq_freq = int(_prop(node, 'squash_frequency', 2))
    squash = float(_prop(node, 'squash', 1.0))
    p = vec * scale[:, None]
    row = np.floor(p[:, 1] / np.maximum(rh, 1e-5))
    off = np.where((row % freq) == 0, offset, 0.0)
    sq = np.where((row % sq_freq) == 0, squash, 1.0)
    x = p[:, 0] / np.maximum(bw * sq, 1e-5) + off
    bx = np.floor(x)
    fx = x - bx
    fy = p[:, 1] / np.maximum(rh, 1e-5) - row
    tint = hash3(bx, row, 0.0) + bias
    mm = np.maximum(msize, 0.0)
    in_mortar = (fx < mm) | (fx > 1.0 - mm) | (fy < mm) | (fy > 1.0 - mm)
    base = c1 + (c2 - c1) * np.clip(tint, 0.0, 1.0)[:, None]
    col = np.where(in_mortar[:, None], mortar, base)
    return {'Color': col.astype(np.float32),
            'Fac': in_mortar.astype(np.float32)}


def n_tex_musgrave(ev, node):
    vec = _tex_vector(ev, node)
    scale = ev.input(node, 'Scale', VALUE)
    detail = ev.input(node, 'Detail', VALUE)
    dim = ev.input(node, 'Dimension', VALUE)
    lac = ev.input(node, 'Lacunarity', VALUE)
    p = vec * scale[:, None]
    h = float(np.mean(dim)) if np.ndim(dim) else 1.0
    f = fractal_noise(p, detail, 1.0 / max(h, 0.1), float(np.mean(lac)) or 2.0, 0.0)
    return {'Fac': f, 'Height': f}


def n_tex_ies(ev, node):
    return {'Fac': np.ones(ev.n, np.float32)}


# ============================================================== colour nodes


def _mix_blend(mode, a, b, t):
    """Blender's MixRGB blend modes on (N,4) colours (alpha kept from a)."""
    ac = a[:, :3]
    bc = b[:, :3]
    t3 = t[:, None]
    if mode == 'MIX':
        r = ac + (bc - ac) * t3
    elif mode == 'ADD':
        r = ac + bc * t3
    elif mode == 'MULTIPLY':
        r = ac * (1.0 - t3 + t3 * bc)
    elif mode == 'SUBTRACT':
        r = ac - bc * t3
    elif mode == 'SCREEN':
        r = 1.0 - (1.0 - ac) * (1.0 - t3 * bc)
    elif mode == 'DIVIDE':
        r = ac * (1.0 - t3) + t3 * ac / np.maximum(bc, 1e-6)
    elif mode == 'DIFFERENCE':
        r = ac + (np.abs(ac - bc) - ac) * t3
    elif mode == 'DARKEN':
        r = ac + (np.minimum(ac, bc) - ac) * t3
    elif mode == 'LIGHTEN':
        r = ac + (np.maximum(ac, bc) - ac) * t3
    elif mode == 'OVERLAY':
        lo = 2.0 * ac * bc
        hi = 1.0 - 2.0 * (1.0 - ac) * (1.0 - bc)
        r = ac + (np.where(ac < 0.5, lo, hi) - ac) * t3
    elif mode == 'SOFT_LIGHT':
        scr = 1.0 - (1.0 - ac) * (1.0 - bc)
        r = ac + t3 * ((1.0 - ac) * bc * ac + ac * scr - ac)
    elif mode == 'LINEAR_LIGHT':
        r = ac + t3 * (2.0 * bc - 1.0)
    elif mode == 'DODGE':
        r = ac + (ac / np.maximum(1.0 - bc, 1e-6) - ac) * t3
    elif mode == 'BURN':
        r = ac + ((1.0 - (1.0 - ac) / np.maximum(bc, 1e-6)) - ac) * t3
    elif mode in ('HUE', 'SATURATION', 'COLOR', 'VALUE'):
        ha, sa, va = M.rgb_to_hsv(ac[:, 0], ac[:, 1], ac[:, 2])
        hb, sb, vb = M.rgb_to_hsv(bc[:, 0], bc[:, 1], bc[:, 2])
        if mode == 'HUE':
            h, s, v = hb, sa, va
        elif mode == 'SATURATION':
            h, s, v = ha, sb, va
        elif mode == 'COLOR':
            h, s, v = hb, sb, va
        else:
            h, s, v = ha, sa, vb
        rr, gg, bb = M.hsv_to_rgb(h, s, v)
        mixed = np.stack([rr, gg, bb], axis=1)
        r = ac + (mixed - ac) * t3
    else:
        r = ac + (bc - ac) * t3
    return np.concatenate([r.astype(np.float32), a[:, 3:]], axis=1)


def n_mix_rgb(ev, node):
    fac = np.clip(ev.input(node, 'Fac', VALUE), 0.0, 1.0)
    a = ev.input(node, 1, RGBA)
    b = ev.input(node, 2, RGBA)
    mode = _prop(node, 'blend_type', 'MIX')
    out = _mix_blend(mode, a, b, fac)
    if _prop(node, 'use_clamp', False):
        out = np.clip(out, 0.0, 1.0)
    return {'Color': out}


MIX_IDENT_SUFFIX = {'FLOAT': '_Float', 'VECTOR': '_Vector',
                    'RGBA': '_Color'}


def mix_socket(node, base, dtype):
    """The Mix node's socket NAME to ask for, disambiguated.

    Blender's Mix node carries one A/B/Factor PER data type, all with
    the same display name -- only the identifier ('A_Color') tells them
    apart, and asking for plain 'A' under RGBA silently reads the FLOAT
    socket's default instead of the user's linked colour. Real exports
    carry identifiers; hand-built test graphs may use plain names, so
    those still resolve.
    """
    ident = base + MIX_IDENT_SUFFIX.get(str(dtype), '_Color')
    for s in node.get('inputs', []):
        if s.get('identifier') == ident:
            return ident
    return base


def n_mix(ev, node):
    dtype = _prop(node, 'data_type', 'RGBA')
    mode = _prop(node, 'blend_type', 'MIX')
    clamp_f = _prop(node, 'clamp_factor', True)
    fac = ev.input(node, mix_socket(node, 'Factor', 'FLOAT'), VALUE)
    if clamp_f:
        fac = np.clip(fac, 0.0, 1.0)
    if dtype == 'FLOAT':
        a = ev.input(node, mix_socket(node, 'A', dtype), VALUE)
        b = ev.input(node, mix_socket(node, 'B', dtype), VALUE)
        return {'Result': a + (b - a) * fac}
    if dtype == 'VECTOR':
        a = ev.input(node, mix_socket(node, 'A', dtype), VECTOR)
        b = ev.input(node, mix_socket(node, 'B', dtype), VECTOR)
        return {'Result': a + (b - a) * fac[:, None]}
    a = ev.input(node, mix_socket(node, 'A', dtype), RGBA)
    b = ev.input(node, mix_socket(node, 'B', dtype), RGBA)
    out = _mix_blend(mode, a, b, fac)
    if _prop(node, 'clamp_result', False):
        out = np.clip(out, 0.0, 1.0)
    return {'Result': out}


def n_invert(ev, node):
    fac = ev.input(node, 'Fac', VALUE)[:, None]
    col = ev.input(node, 'Color', RGBA)
    inv = np.concatenate([1.0 - col[:, :3], col[:, 3:]], 1)
    return {'Color': col + (inv - col) * fac}


def n_hue_sat(ev, node):
    h = ev.input(node, 'Hue', VALUE)
    s = ev.input(node, 'Saturation', VALUE)
    v = ev.input(node, 'Value', VALUE)
    fac = ev.input(node, 'Fac', VALUE)[:, None]
    col = ev.input(node, 'Color', RGBA)
    hh, ss, vv = M.rgb_to_hsv(col[:, 0], col[:, 1], col[:, 2])
    hh = (hh + h - 0.5) % 1.0
    ss = np.clip(ss * s, 0.0, 1.0)
    vv = vv * v
    r, g, b = M.hsv_to_rgb(hh, ss, vv)
    out = np.stack([r, g, b], axis=1)
    out = np.concatenate([out, col[:, 3:]], 1).astype(np.float32)
    return {'Color': col + (out - col) * fac}


def n_bright_contrast(ev, node):
    col = ev.input(node, 'Color', RGBA)
    bright = ev.input(node, 'Bright', VALUE)[:, None]
    contrast = ev.input(node, 'Contrast', VALUE)[:, None]
    a = 1.0 + contrast
    b = bright - contrast * 0.5
    rgb = np.maximum(a * col[:, :3] + b, 0.0)
    return {'Color': np.concatenate([rgb, col[:, 3:]], 1).astype(np.float32)}


def n_gamma(ev, node):
    col = ev.input(node, 'Color', RGBA)
    g = ev.input(node, 'Gamma', VALUE)[:, None]
    rgb = np.power(np.maximum(col[:, :3], 0.0), np.maximum(g, 1e-4))
    return {'Color': np.concatenate([rgb, col[:, 3:]], 1).astype(np.float32)}


def n_rgb_curve(ev, node):
    col = ev.input(node, 'Color', RGBA)
    fac = ev.input(node, 'Fac', VALUE)[:, None]
    lut = _prop(node, 'lut')
    if lut is None:
        return {'Color': col}
    lut = np.asarray(lut, np.float32)          # (K,4): combined,R,G,B
    out = np.empty_like(col)
    for ch in range(3):
        v = lut_eval(lut[:, 0], col[:, ch])
        out[:, ch] = lut_eval(lut[:, ch + 1], v)
    out[:, 3] = col[:, 3]
    return {'Color': col + (out - col) * fac}


def n_float_curve(ev, node):
    val = ev.input(node, 'Value', VALUE)
    fac = ev.input(node, 'Factor', VALUE)
    lut = _prop(node, 'lut')
    if lut is None:
        return {'Value': val}
    out = lut_eval(np.asarray(lut, np.float32), val)
    return {'Value': val + (out - val) * fac}


def n_vector_curve(ev, node):
    vec = ev.input(node, 'Vector', VECTOR)
    fac = ev.input(node, 'Fac', VALUE)[:, None]
    lut = _prop(node, 'lut')
    if lut is None:
        return {'Vector': vec}
    lut = np.asarray(lut, np.float32)
    lo = _prop(node, 'range_min', [-1.0, -1.0, -1.0])
    hi = _prop(node, 'range_max', [1.0, 1.0, 1.0])
    out = np.empty_like(vec)
    for ch in range(3):
        t = (vec[:, ch] - lo[ch]) / max(hi[ch] - lo[ch], 1e-6)
        out[:, ch] = lut_eval(lut[:, ch], np.clip(t, 0, 1))
    return {'Vector': vec + (out - vec) * fac}


def n_val_to_rgb(ev, node):
    fac = ev.input(node, 'Fac', VALUE)
    lut = _prop(node, 'lut')
    if lut is None:
        c = np.stack([fac, fac, fac, np.ones_like(fac)], axis=1)
        return {'Color': c, 'Alpha': np.ones_like(fac)}
    col = lut_eval(np.asarray(lut, np.float32), fac)
    return {'Color': col.astype(np.float32), 'Alpha': col[:, 3].astype(np.float32)}


def n_rgb_to_bw(ev, node):
    col = ev.input(node, 'Color', RGBA)
    return {'Val': M.luminance(col[:, :3])}


def n_blackbody(ev, node):
    t = ev.input(node, 'Temperature', VALUE)
    rgb = blackbody(t)
    return {'Color': np.concatenate([rgb, np.ones((ev.n, 1), np.float32)], 1)}


def n_wavelength(ev, node):
    w = ev.input(node, 'Wavelength', VALUE)
    rgb = wavelength_to_rgb(w)
    return {'Color': np.concatenate([rgb, np.ones((ev.n, 1), np.float32)], 1)}


# ============================================================== vector nodes


def n_mapping(ev, node):
    vec = ev.input(node, 'Vector', VECTOR)
    loc = ev.input(node, 'Location', VECTOR)
    rot = ev.input(node, 'Rotation', VECTOR)
    scl = ev.input(node, 'Scale', VECTOR)
    mode = _prop(node, 'vector_type', 'POINT')
    if mode == 'TEXTURE':
        v = vec - loc
        v = _rotate_xyz(v, -rot)
        v = v / np.where(np.abs(scl) < 1e-8, 1e-8, scl)
        return {'Vector': v.astype(np.float32)}
    v = vec * scl
    v = _rotate_xyz(v, rot)
    if mode == 'POINT':
        v = v + loc
    if mode == 'NORMAL':
        v = M.normalize(v)
    return {'Vector': v.astype(np.float32)}


def _rotate_xyz(v, rot):
    x, y, z = rot[:, 0], rot[:, 1], rot[:, 2]
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    px, py, pz = v[:, 0], v[:, 1], v[:, 2]
    y1 = py * cx - pz * sx
    z1 = py * sx + pz * cx
    x2 = px * cy + z1 * sy
    z2 = -px * sy + z1 * cy
    x3 = x2 * cz - y1 * sz
    y3 = x2 * sz + y1 * cz
    return np.stack([x3, y3, z2], axis=1)


def n_vector_math(ev, node):
    op = _prop(node, 'operation', 'ADD')
    a = ev.input(node, 0, VECTOR)
    b = ev.input(node, 1, VECTOR)
    c = ev.input(node, 2, VECTOR)
    s = ev.input(node, 'Scale', VALUE) if any(
        x.get('name') == 'Scale' for x in node.get('inputs', [])) else None
    out_v = None
    out_f = None
    if op == 'ADD':
        out_v = a + b
    elif op == 'SUBTRACT':
        out_v = a - b
    elif op == 'MULTIPLY':
        out_v = a * b
    elif op == 'DIVIDE':
        out_v = a / np.where(np.abs(b) < 1e-8, 1e-8, b)
    elif op == 'CROSS_PRODUCT':
        out_v = np.cross(a, b)
    elif op == 'PROJECT':
        d = M.dot(b, b)
        out_v = b * (M.dot(a, b) / np.maximum(d, 1e-8))[:, None]
    elif op == 'REFLECT':
        out_v = M.reflect(a, M.normalize(b))
    elif op == 'REFRACT':
        out_v = M.refract(M.normalize(a), M.normalize(b), s if s is not None else 1.0)
    elif op == 'FACEFORWARD':
        out_v = np.where(M.dot(c, b, keepdims=True) < 0, a, -a)
    elif op == 'DOT_PRODUCT':
        out_f = M.dot(a, b)
    elif op == 'DISTANCE':
        out_f = M.length(a - b)
    elif op == 'LENGTH':
        out_f = M.length(a)
    elif op == 'SCALE':
        out_v = a * (s if s is not None else np.ones(ev.n, np.float32))[:, None]
    elif op == 'NORMALIZE':
        out_v = M.normalize(a)
    elif op == 'ABSOLUTE':
        out_v = np.abs(a)
    elif op == 'MINIMUM':
        out_v = np.minimum(a, b)
    elif op == 'MAXIMUM':
        out_v = np.maximum(a, b)
    elif op == 'FLOOR':
        out_v = np.floor(a)
    elif op == 'CEIL':
        out_v = np.ceil(a)
    elif op == 'FRACTION':
        out_v = a - np.floor(a)
    elif op == 'MODULO':
        out_v = np.mod(a, np.where(np.abs(b) < 1e-8, 1e-8, b))
    elif op == 'WRAP':
        rng = b - c
        out_v = np.where(np.abs(rng) < 1e-8, c, c + np.mod(a - c, rng))
    elif op == 'SNAP':
        out_v = np.floor(a / np.where(np.abs(b) < 1e-8, 1e-8, b)) * b
    elif op == 'SINE':
        out_v = np.sin(a)
    elif op == 'COSINE':
        out_v = np.cos(a)
    elif op == 'TANGENT':
        out_v = np.tan(a)
    else:
        out_v = a
    res = {}
    if out_v is not None:
        res['Vector'] = out_v.astype(np.float32)
        res['Value'] = M.length(out_v).astype(np.float32)
    if out_f is not None:
        res['Value'] = out_f.astype(np.float32)
        res['Vector'] = np.zeros((ev.n, 3), np.float32)
    return res


def n_vector_rotate(ev, node):
    vec = ev.input(node, 'Vector', VECTOR)
    center = ev.input(node, 'Center', VECTOR)
    axis = ev.input(node, 'Axis', VECTOR)
    angle = ev.input(node, 'Angle', VALUE)
    rot = ev.input(node, 'Rotation', VECTOR)
    mode = _prop(node, 'rotation_type', 'AXIS_ANGLE')
    inv = _prop(node, 'invert', False)
    p = vec - center
    if mode == 'EULER_XYZ':
        out = _rotate_xyz(p, -rot if inv else rot)
    else:
        if mode == 'X_AXIS':
            ax = np.tile(np.array([[1.0, 0, 0]], np.float32), (ev.n, 1))
        elif mode == 'Y_AXIS':
            ax = np.tile(np.array([[0, 1.0, 0]], np.float32), (ev.n, 1))
        elif mode == 'Z_AXIS':
            ax = np.tile(np.array([[0, 0, 1.0]], np.float32), (ev.n, 1))
        else:
            ax = M.normalize(axis)
        a = -angle if inv else angle
        ca = np.cos(a)[:, None]
        sa = np.sin(a)[:, None]
        out = p * ca + np.cross(ax, p) * sa + ax * M.dot(ax, p)[:, None] * (1.0 - ca)
    return {'Vector': (out + center).astype(np.float32)}


def n_normal(ev, node):
    d = coerce(_prop(node, 'direction', [0.0, 0.0, 1.0]), VECTOR, ev.n)
    v = ev.input(node, 'Normal', VECTOR)
    return {'Normal': d, 'Dot': M.dot(M.normalize(v), M.normalize(d))}


# ------------------------------------------------ colour-space ramp helpers

def _rgb_to_oklab(rgb):
    """Linear sRGB -> OKLab (Ottosson 2020), vectorised."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b
    l = np.cbrt(np.maximum(l, 0.0))
    m = np.cbrt(np.maximum(m, 0.0))
    s = np.cbrt(np.maximum(s, 0.0))
    return np.stack([
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s], axis=1)


def _oklab_to_rgb(lab):
    L, A, B = lab[:, 0], lab[:, 1], lab[:, 2]
    l = L + 0.3963377774 * A + 0.2158037573 * B
    m = L - 0.1055613458 * A - 0.0638541728 * B
    s = L - 0.0894841775 * A - 1.2914855480 * B
    l, m, s = l * l * l, m * m * m, s * s * s
    return np.stack([
        +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s], axis=1)


def _rgb_to_hsv_np(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    d = mx - mn
    h = np.zeros_like(mx)
    nz = d > 1e-12
    rc = np.where(nz, (mx - r) / np.where(nz, d, 1.0), 0.0)
    gc = np.where(nz, (mx - g) / np.where(nz, d, 1.0), 0.0)
    bc = np.where(nz, (mx - b) / np.where(nz, d, 1.0), 0.0)
    h = np.where(mx == r, bc - gc, h)
    h = np.where(mx == g, 2.0 + rc - bc, h)
    h = np.where((mx == b) & (mx != r) & (mx != g), 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-12, d / np.where(mx > 1e-12, mx, 1.0), 0.0)
    return np.stack([h, s, mx], axis=1)


def _hsv_to_rgb_np(hsv):
    h, s, v = hsv[:, 0] % 1.0, hsv[:, 1], hsv[:, 2]
    i = np.floor(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i.astype(np.int64) % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=1)


def _ramp_blend(a_rgb, b_rgb, t, space):
    """Blend two (N,3) colours by (N,) t in the named space."""
    t3 = t[:, None]
    if space == 'OKLAB':
        return _oklab_to_rgb(_rgb_to_oklab(a_rgb) +
                             (_rgb_to_oklab(b_rgb) -
                              _rgb_to_oklab(a_rgb)) * t3)
    if space == 'OKLCH':
        la = _rgb_to_oklab(a_rgb)
        lb = _rgb_to_oklab(b_rgb)
        ca = np.sqrt(la[:, 1] ** 2 + la[:, 2] ** 2)
        cb = np.sqrt(lb[:, 1] ** 2 + lb[:, 2] ** 2)
        ha = np.arctan2(la[:, 2], la[:, 1])
        hb = np.arctan2(lb[:, 2], lb[:, 1])
        dh = hb - ha
        dh = dh - np.round(dh / (2 * np.pi)) * 2 * np.pi   # short way round
        L = la[:, 0] + (lb[:, 0] - la[:, 0]) * t
        C = ca + (cb - ca) * t
        H = ha + dh * t
        return _oklab_to_rgb(np.stack([L, C * np.cos(H), C * np.sin(H)],
                                      axis=1))
    if space == 'HSV':
        ha = _rgb_to_hsv_np(a_rgb)
        hb = _rgb_to_hsv_np(b_rgb)
        dh = hb[:, 0] - ha[:, 0]
        dh = dh - np.round(dh)                             # short way round
        out = ha + (hb - ha) * t3
        out[:, 0] = (ha[:, 0] + dh * t) % 1.0
        return _hsv_to_rgb_np(out)
    return a_rgb + (b_rgb - a_rgb) * t3


def n_halcyon_ramp(ev, node):
    fac = np.clip(ev.input(node, 'Fac', VALUE), 0.0, 1.0)
    n_stops = int(np.clip(_prop(node, 'stops', 2), 2, 6))
    pos_raw = _prop(node, 'positions',
                    (0.0, 1.0, 0.5, 0.5, 0.5, 0.5))
    space = str(_prop(node, 'space', 'OKLAB'))
    ease = str(_prop(node, 'easing', 'LINEAR'))
    stops = []
    for i in range(n_stops):
        col = ev.input(node, f'Color {i + 1}', RGBA)
        stops.append((float(pos_raw[i]), col))
    stops.sort(key=lambda s: s[0])
    out = stops[0][1][:, :3].copy()
    alpha = stops[0][1][:, 3].copy()
    for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
        span = max(p1 - p0, 1e-6)
        t = np.clip((fac - p0) / span, 0.0, 1.0)
        if ease == 'SMOOTH':
            t = t * t * (3.0 - 2.0 * t)
        elif ease == 'CONSTANT':
            t = np.where(t >= 1.0, 1.0, 0.0)
        seg = fac >= p0
        blend = _ramp_blend(c0[:, :3], c1[:, :3], t, space)
        out = np.where(seg[:, None], blend, out)
        alpha = np.where(seg, c0[:, 3] + (c1[:, 3] - c0[:, 3]) * t, alpha)
    col = np.concatenate([out, alpha[:, None]], axis=1).astype(np.float32)
    return {'Color': col, 'Alpha': alpha.astype(np.float32)}


#: blur tap tables: (dx, dy, weight) rings, deterministic and shared with
#: any future GPU twin. Weights normalised at build
def _blur_taps(kind):
    if kind == 'FAST':
        pts = [(0, 0, 2.0), (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0),
               (0, -1, 1.0)]
    elif kind == 'FINE':
        pts = [(0, 0, 2.0)]
        for k in range(8):
            a = 2.0 * np.pi * k / 8.0
            pts.append((np.cos(a), np.sin(a), 1.0))
            pts.append((0.55 * np.cos(a + 0.3927),
                        0.55 * np.sin(a + 0.3927), 1.2))
    else:
        pts = [(0, 0, 2.0)]
        for k in range(8):
            a = 2.0 * np.pi * k / 8.0
            pts.append((np.cos(a), np.sin(a), 1.0))
    arr = np.asarray(pts, np.float64)
    arr[:, 2] /= arr[:, 2].sum()
    return arr.astype(np.float32)


def n_halcyon_blur(ev, node):
    """Average the linked chain over shifted evaluation points.

    Each tap re-evaluates the upstream subtree with the context's texture
    spaces (generated, uv, object, world position) shifted in the surface
    plane -- true blur of whatever is connected. CPU-only by design; the
    GPU plan refuses it by name and the material shades here.
    """
    if not ev.has_link(node, 'Color'):
        return {'Color': ev.input(node, 'Color', RGBA)}
    size = ev.input(node, 'Size', VALUE)
    taps = _blur_taps(str(_prop(node, 'taps', 'MEDIUM')))
    link = None
    for s in node.get('inputs', ()):
        if s.get('name') == 'Color':
            link = s.get('link')
    c = ev.ctx
    from ..core import mathx as M
    t_axis, b_axis = M.orthonormal_basis(M.normalize(c.N))
    acc = np.zeros((ev.n, 4), np.float32)
    for dx, dy, wgt in taps:
        if dx == 0.0 and dy == 0.0:
            sub_ctx = c
        else:
            sub_ctx = _shifted_ctx(c, t_axis, b_axis,
                                   size * np.float32(dx),
                                   size * np.float32(dy))
        sub = GraphEvaluator({'output': None, 'nodes': ev.nodes}, sub_ctx,
                             ev.images, ev.programs)
        v = sub.eval_output(link[0], link[1])
        acc += coerce(v, RGBA, ev.n) * np.float32(wgt)
    return {'Color': acc.astype(np.float32)}


def _shifted_ctx(c, t_axis, b_axis, du, dv):
    """A shallow context copy with every texture space shifted in-plane."""
    import copy
    sub = copy.copy(c)
    d3 = t_axis * du[:, None] + b_axis * dv[:, None]
    sub.P = c.P + d3
    sub.generated = c.generated + d3
    sub.uv = c.uv + np.stack([du, dv], axis=1)
    sub.uv2 = c.uv2 + np.stack([du, dv], axis=1)
    return sub


def desugar_master_bump(graph):
    """Bump Height on the master shader becomes a REAL Bump node, in place.

    The socket is sugar: a linked greyscale height cannot honestly grow a
    second bump implementation on each device, so it is rewritten -- once,
    idempotently -- into the ShaderNodeBump both backends already prove:
    Height takes the link, Strength takes the master's Bump Strength,
    the master's own Normal chain (if any) threads through underneath,
    and the master's Normal input then points at the synthesized node.
    Called at the top of every render, so the CPU evaluator, the GPU
    plan and the emitter all see the SAME desugared tree.
    """
    if not graph:
        return graph
    nodes = graph.get('nodes')
    if not nodes:
        return graph
    for nid in list(nodes.keys()):
        nd = nodes[nid]
        if nd.get('bl_idname') != 'HALCYON_ShaderNode':
            continue
        ins = nd.get('inputs') or []
        bh = next((s for s in ins if s.get('name') == 'Bump Height'), None)
        if bh is None or not bh.get('link'):
            continue
        synth = f'__bump_{nid}'
        if synth in nodes:                        # already desugared
            continue
        bs = next((s for s in ins if s.get('name') == 'Bump Strength'), None)
        nrm = next((s for s in ins if s.get('name') == 'Normal'), None)
        strength = {'name': 'Strength', 'type': 'VALUE',
                    'default': (bs or {}).get('default', 1.0),
                    'link': (bs or {}).get('link')}
        nodes[synth] = {
            'id': synth, 'bl_idname': 'ShaderNodeBump', 'props': {},
            'inputs': [
                strength,
                {'name': 'Distance', 'type': 'VALUE', 'default': 1.0,
                 'link': None},
                {'name': 'Height', 'type': 'VALUE', 'default': 0.5,
                 'link': list(bh['link'])},
                {'name': 'Normal', 'type': 'VECTOR', 'default': [0, 0, 0],
                 'link': list(nrm['link']) if nrm and nrm.get('link')
                 else None},
            ],
            'outputs': [{'name': 'Normal', 'type': 'VECTOR'}],
        }
        if nrm is None:
            nrm = {'name': 'Normal', 'type': 'VECTOR', 'default': [0, 0, 0],
                   'link': None}
            ins.append(nrm)
        nrm['link'] = [synth, 0]
        bh['link'] = None            # consumed; nothing reads it downstream
    return graph


def n_bump(ev, node):
    c = ev.ctx
    strength = ev.input(node, 'Strength', VALUE)
    dist = ev.input(node, 'Distance', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else c.N
    if c.px is None or c.width <= 0:
        return {'Normal': nrm}
    field = _bump_field_for(c, node)
    if field is not None:
        # a whole-material gradient pre-pass exists (built when this
        # material is too covered for one shading batch): gather from it,
        # so a chunk boundary can never cut the waves. The height chain
        # is not re-evaluated here -- the field already ran it.
        gx, gy = field
        dhdx, dhdy = gx[c.py, c.px], gy[c.py, c.px]
    else:
        height = ev.input(node, 'Height', VALUE)
        dhdx, dhdy = _screen_grad(c, height)
    t, b = M.orthonormal_basis(M.normalize(nrm))
    scale = (strength * dist)[:, None]
    invert = -1.0 if _prop(node, 'invert', False) else 1.0
    out = M.normalize(M.normalize(nrm) - invert * scale *
                      (t * dhdx[:, None] + b * dhdy[:, None]) * 20.0)
    return {'Normal': out.astype(np.float32)}


def _bump_field_for(c, node):
    """The whole-material gradient grids for this node, if pre-passed.

    Fields are keyed by (material index, node id); the batch is
    single-material wherever `c.px` is real, so the batch's first
    triangle names the material.
    """
    fields = getattr(c, 'bump_fields', None)
    tri = getattr(c, 'tri', None)
    if not fields or tri is None or np.size(tri) == 0:
        return None
    mesh = getattr(getattr(c, 'scene', None), 'mesh', None)
    mat_index = getattr(mesh, 'mat_index', None) if mesh is not None \
        else None
    mi = int(mat_index[tri[0]]) if mat_index is not None else 0
    return fields.get((mi, node.get('id')))


def _screen_grad(c, values):
    img = np.zeros((c.height, c.width), np.float32)
    valid = np.zeros((c.height, c.width), bool)
    img[c.py, c.px] = values
    valid[c.py, c.px] = True
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, :-1] = np.where(valid[:, 1:] & valid[:, :-1], img[:, 1:] - img[:, :-1], 0.0)
    gy[:-1, :] = np.where(valid[1:, :] & valid[:-1, :], img[1:, :] - img[:-1, :], 0.0)
    return gx[c.py, c.px], gy[c.py, c.px]


def n_normal_map(ev, node):
    c = ev.ctx
    col = ev.input(node, 'Color', RGBA)
    strength = ev.input(node, 'Strength', VALUE)[:, None]
    n = M.normalize(c.N)
    t = c.T if c.T is not None else M.orthonormal_basis(n)[0]
    t = M.normalize(t)
    b = np.cross(n, t)
    tn = col[:, :3] * 2.0 - 1.0
    space = _prop(node, 'space', 'TANGENT')
    if space == 'OBJECT' or space == 'WORLD':
        out = M.normalize(tn)
    else:
        out = M.normalize(t * tn[:, 0:1] + b * tn[:, 1:2] + n * tn[:, 2:3])
    out = M.normalize(n + (out - n) * strength)
    return {'Normal': out.astype(np.float32)}


def n_displacement(ev, node):
    h = ev.input(node, 'Height', VALUE)
    scale = ev.input(node, 'Scale', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else ev.ctx.N
    return {'Displacement': (nrm * (h * scale)[:, None]).astype(np.float32)}


# =========================================================== converter nodes


MATH_OPS = {
    'ADD': lambda a, b, c: a + b,
    'SUBTRACT': lambda a, b, c: a - b,
    'MULTIPLY': lambda a, b, c: a * b,
    'DIVIDE': lambda a, b, c: a / np.where(np.abs(b) < 1e-9, 1e-9, b),
    'MULTIPLY_ADD': lambda a, b, c: a * b + c,
    'POWER': lambda a, b, c: np.power(np.maximum(a, 0.0), b),
    'LOGARITHM': lambda a, b, c: np.log(np.maximum(a, 1e-9)) /
    np.log(np.maximum(b, 1e-9)),
    'SQRT': lambda a, b, c: np.sqrt(np.maximum(a, 0.0)),
    'INVERSE_SQRT': lambda a, b, c: 1.0 / np.sqrt(np.maximum(a, 1e-9)),
    'ABSOLUTE': lambda a, b, c: np.abs(a),
    'EXPONENT': lambda a, b, c: np.exp(a),
    'MINIMUM': lambda a, b, c: np.minimum(a, b),
    'MAXIMUM': lambda a, b, c: np.maximum(a, b),
    'LESS_THAN': lambda a, b, c: (a < b).astype(np.float32),
    'GREATER_THAN': lambda a, b, c: (a > b).astype(np.float32),
    'SIGN': lambda a, b, c: np.sign(a),
    'COMPARE': lambda a, b, c: (np.abs(a - b) <= c).astype(np.float32),
    'SMOOTH_MIN': lambda a, b, c: _smoothmin(a, b, c),
    'SMOOTH_MAX': lambda a, b, c: -_smoothmin(-a, -b, c),
    'ROUND': lambda a, b, c: np.round(a),
    'FLOOR': lambda a, b, c: np.floor(a),
    'CEIL': lambda a, b, c: np.ceil(a),
    'TRUNC': lambda a, b, c: np.trunc(a),
    'FRACT': lambda a, b, c: a - np.floor(a),
    'MODULO': lambda a, b, c: np.mod(a, np.where(np.abs(b) < 1e-9, 1e-9, b)),
    'FLOORED_MODULO': lambda a, b, c: np.mod(a, np.where(np.abs(b) < 1e-9, 1e-9, b)),
    'WRAP': lambda a, b, c: np.where(np.abs(b - c) < 1e-9, c,
                                     c + np.mod(a - c, b - c)),
    'SNAP': lambda a, b, c: np.floor(a / np.where(np.abs(b) < 1e-9, 1e-9, b)) * b,
    'PINGPONG': lambda a, b, c: _pingpong(a, b),
    'SINE': lambda a, b, c: np.sin(a),
    'COSINE': lambda a, b, c: np.cos(a),
    'TANGENT': lambda a, b, c: np.tan(a),
    'ARCSINE': lambda a, b, c: np.arcsin(np.clip(a, -1, 1)),
    'ARCCOSINE': lambda a, b, c: np.arccos(np.clip(a, -1, 1)),
    'ARCTANGENT': lambda a, b, c: np.arctan(a),
    'ARCTAN2': lambda a, b, c: np.arctan2(a, b),
    'SINH': lambda a, b, c: np.sinh(a),
    'COSH': lambda a, b, c: np.cosh(a),
    'TANH': lambda a, b, c: np.tanh(a),
    'RADIANS': lambda a, b, c: np.radians(a),
    'DEGREES': lambda a, b, c: np.degrees(a),
}


def _smoothmin(a, b, k):
    k = np.maximum(k, 1e-6)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1 - h) + a * h - k * h * (1.0 - h)


def _pingpong(a, b):
    b = np.where(np.abs(b) < 1e-9, 1e-9, b)
    return np.abs(np.mod(a - b, 2 * b) - b)


def n_math(ev, node):
    op = _prop(node, 'operation', 'ADD')
    a = ev.input(node, 0, VALUE)
    b = ev.input(node, 1, VALUE)
    c = ev.input(node, 2, VALUE)
    f = MATH_OPS.get(op, MATH_OPS['ADD'])
    v = f(a, b, c).astype(np.float32)
    if _prop(node, 'use_clamp', False):
        v = np.clip(v, 0.0, 1.0)
    return {'Value': v}


def n_clamp(ev, node):
    v = ev.input(node, 'Value', VALUE)
    lo = ev.input(node, 'Min', VALUE)
    hi = ev.input(node, 'Max', VALUE)
    if _prop(node, 'clamp_type', 'MINMAX') == 'RANGE':
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        lo, hi = lo2, hi2
    return {'Result': np.clip(v, lo, hi)}


def n_map_range(ev, node):
    v = ev.input(node, 'Value', VALUE)
    fmin = ev.input(node, 'From Min', VALUE)
    fmax = ev.input(node, 'From Max', VALUE)
    tmin = ev.input(node, 'To Min', VALUE)
    tmax = ev.input(node, 'To Max', VALUE)
    t = (v - fmin) / np.where(np.abs(fmax - fmin) < 1e-9, 1e-9, fmax - fmin)
    itype = _prop(node, 'interpolation_type', 'LINEAR')
    if itype == 'SMOOTHSTEP':
        t = np.clip(t, 0, 1)
        t = t * t * (3 - 2 * t)
    elif itype == 'SMOOTHERSTEP':
        t = np.clip(t, 0, 1)
        t = t * t * t * (t * (t * 6 - 15) + 10)
    elif itype == 'STEPPED':
        steps = ev.input(node, 'Steps', VALUE)
        t = np.floor(np.clip(t, 0, 1) * (steps + 1)) / np.maximum(steps, 1e-6)
    out = tmin + t * (tmax - tmin)
    if _prop(node, 'clamp', True) and itype != 'LINEAR' or _prop(node, 'clamp', True):
        out = np.clip(out, np.minimum(tmin, tmax), np.maximum(tmin, tmax))
    return {'Result': out.astype(np.float32)}


def n_separate_xyz(ev, node):
    v = ev.input(node, 'Vector', VECTOR)
    return {'X': v[:, 0], 'Y': v[:, 1], 'Z': v[:, 2]}


def n_combine_xyz(ev, node):
    return {'Vector': np.stack([ev.input(node, 'X', VALUE),
                                ev.input(node, 'Y', VALUE),
                                ev.input(node, 'Z', VALUE)], axis=1).astype(np.float32)}


def n_separate_color(ev, node):
    col = ev.input(node, 'Color', RGBA)
    mode = _prop(node, 'mode', 'RGB')
    if mode == 'HSV' or mode == 'HSL':
        h, s, v = M.rgb_to_hsv(col[:, 0], col[:, 1], col[:, 2])
        return {'Red': h, 'Green': s, 'Blue': v, 'Alpha': col[:, 3]}
    return {'Red': col[:, 0], 'Green': col[:, 1], 'Blue': col[:, 2],
            'Alpha': col[:, 3]}


def n_combine_color(ev, node):
    r = ev.input(node, 'Red', VALUE)
    g = ev.input(node, 'Green', VALUE)
    b = ev.input(node, 'Blue', VALUE)
    mode = _prop(node, 'mode', 'RGB')
    if mode in ('HSV', 'HSL'):
        r2, g2, b2 = M.hsv_to_rgb(r, g, b)
        r, g, b = r2, g2, b2
    return {'Color': np.stack([r, g, b, np.ones_like(r)], axis=1).astype(np.float32)}


def n_separate_rgb(ev, node):
    col = ev.input(node, 'Image', RGBA)
    return {'R': col[:, 0], 'G': col[:, 1], 'B': col[:, 2]}


def n_combine_rgb(ev, node):
    return {'Image': np.stack([ev.input(node, 'R', VALUE),
                               ev.input(node, 'G', VALUE),
                               ev.input(node, 'B', VALUE),
                               np.ones(ev.n, np.float32)], axis=1)}


def n_separate_hsv(ev, node):
    col = ev.input(node, 'Color', RGBA)
    h, s, v = M.rgb_to_hsv(col[:, 0], col[:, 1], col[:, 2])
    return {'H': h, 'S': s, 'V': v}


def n_combine_hsv(ev, node):
    r, g, b = M.hsv_to_rgb(ev.input(node, 'H', VALUE), ev.input(node, 'S', VALUE),
                           ev.input(node, 'V', VALUE))
    return {'Color': np.stack([r, g, b, np.ones(ev.n, np.float32)], axis=1)}


def n_shader_to_rgb(ev, node):
    """Works properly here: we are a rasteriser, so the shader is already
    evaluated in screen space."""
    cl = ev.input(node, 'Shader', SHADER)
    col = ev.ctx.settings and None
    rgb = resolve_preview_color(cl, ev.ctx)
    return {'Color': rgb, 'Alpha': rgb[:, 3]}


def resolve_preview_color(cl, ctx):
    n = ctx.n
    out = np.zeros((n, 4), np.float32)
    out[:, 3] = 1.0
    if not isinstance(cl, Closure):
        return out
    from .render import shade_closure_flat
    return shade_closure_flat(cl, ctx)


# ============================================================== shader nodes


def _w(ev, v=None):
    return np.ones(ev.n, np.float32) if v is None else v


def n_bsdf_diffuse(ev, node):
    col = ev.input(node, 'Color', RGBA)
    rough = ev.input(node, 'Roughness', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None
    cl = Closure()
    cl.add('DIFFUSE', _w(ev), color=col, roughness=rough, normal=nrm,
           model='OREN_NAYAR' if np.any(rough > 0.01) else 'LAMBERT')
    return {'BSDF': cl}


def n_bsdf_glossy(ev, node):
    col = ev.input(node, 'Color', RGBA)
    rough = ev.input(node, 'Roughness', VALUE)
    aniso = ev.input(node, 'Anisotropy', VALUE)
    rot = ev.input(node, 'Rotation', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None
    dist = _prop(node, 'distribution', 'GGX')
    model = 'WARD' if np.any(np.abs(aniso) > 1e-4) else (
        'BLINN' if dist in ('BECKMANN', 'MULTI_GGX') else 'COOK_TORRANCE')
    cl = Closure()
    cl.add('GLOSSY', _w(ev), color=col, roughness=rough, normal=nrm,
           anisotropy=aniso, rotation=rot, model=model)
    return {'BSDF': cl}


def n_bsdf_glass(ev, node):
    col = ev.input(node, 'Color', RGBA)
    rough = ev.input(node, 'Roughness', VALUE)
    ior = ev.input(node, 'IOR', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None
    cl = Closure()
    cl.add('GLASS', _w(ev), color=col, roughness=rough, ior=ior, normal=nrm)
    return {'BSDF': cl}


def n_bsdf_metallic(ev, node):
    """Blender 4.x's Metallic BSDF: a pure conductor. The era name for that
    is the METAL model -- tinted specular, no diffuse term."""
    col = ev.input(node, 'Base Color', RGBA)
    rough = ev.input(node, 'Roughness', VALUE)
    has = {s.get('name') for s in node.get('inputs', [])}
    aniso = ev.input(node, 'Anisotropy', VALUE) if 'Anisotropy' in has \
        else np.zeros(ev.n, np.float32)
    rot = ev.input(node, 'Rotation', VALUE) if 'Rotation' in has \
        else np.zeros(ev.n, np.float32)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None
    cl = Closure()
    cl.add('GLOSSY', _w(ev), color=col, roughness=rough, normal=nrm,
           anisotropy=aniso, rotation=rot,
           model='WARD' if np.any(np.abs(aniso) > 1e-4) else 'METAL',
           metallic=np.ones(ev.n, np.float32))
    return {'BSDF': cl}


def n_bsdf_specular(ev, node):
    """EEVEE's Specular BSDF: the spec/gloss workflow -- base colour plus a
    SPECULAR COLOUR and a roughness. That is the DirectX-era material
    model this whole engine wears, so the translation is nearly literal:
    a Lambert diffuse plus a Blinn-Phong specular tinted by the socket."""
    base = ev.input(node, 'Base Color', RGBA)
    spec = ev.input(node, 'Specular', RGBA)
    rough = ev.input(node, 'Roughness', VALUE)
    has = {s.get('name') for s in node.get('inputs', [])}
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None
    cl = Closure()
    cl.add('DIFFUSE', _w(ev), color=base,
           roughness=np.zeros(ev.n, np.float32), normal=nrm, model='LAMBERT')
    cl.add('GLOSSY', _w(ev), color=spec, roughness=rough, normal=nrm,
           model='BLINN_PHONG')
    if 'Emissive Color' in has:
        emis = ev.input(node, 'Emissive Color', RGBA)
        if np.any(emis[:, :3] > 1e-6):
            cl.add('EMISSION', _w(ev), color=emis,
                   strength=np.ones(ev.n, np.float32))
    if 'Transparency' in has:
        transp = np.clip(ev.input(node, 'Transparency', VALUE), 0.0, 1.0)
        if np.any(transp > 1e-4):
            cl.add('TRANSPARENT', transp,
                   color=np.ones((ev.n, 4), np.float32))
    return {'BSDF': cl}


def n_bsdf_refraction(ev, node):
    col = ev.input(node, 'Color', RGBA)
    rough = ev.input(node, 'Roughness', VALUE)
    ior = ev.input(node, 'IOR', VALUE)
    cl = Closure()
    cl.add('REFRACTION', _w(ev), color=col, roughness=rough, ior=ior)
    return {'BSDF': cl}


def n_bsdf_transparent(ev, node):
    col = ev.input(node, 'Color', RGBA)
    cl = Closure()
    cl.add('TRANSPARENT', _w(ev), color=col)
    return {'BSDF': cl}


def n_bsdf_translucent(ev, node):
    col = ev.input(node, 'Color', RGBA)
    cl = Closure()
    cl.add('TRANSLUCENT', _w(ev), color=col)
    return {'BSDF': cl}


def n_bsdf_toon(ev, node):
    col = ev.input(node, 'Color', RGBA)
    size = ev.input(node, 'Size', VALUE)
    smooth = ev.input(node, 'Smooth', VALUE)
    cl = Closure()
    cl.add('DIFFUSE', _w(ev), color=col, model='TOON', toon_size=size,
           toon_smooth=smooth)
    return {'BSDF': cl}


def n_bsdf_velvet(ev, node):
    col = ev.input(node, 'Color', RGBA)
    sigma = ev.input(node, 'Sigma', VALUE)
    cl = Closure()
    cl.add('DIFFUSE', _w(ev), color=col, model='MINNAERT', roughness=sigma)
    return {'BSDF': cl}


def n_bsdf_anisotropic(ev, node):
    return n_bsdf_glossy(ev, node)


def n_subsurface(ev, node):
    col = ev.input(node, 'Color', RGBA)
    cl = Closure()
    cl.add('DIFFUSE', _w(ev), color=col, model='OREN_NAYAR',
           roughness=np.full(ev.n, 0.5, np.float32))
    cl.add('TRANSLUCENT', np.full(ev.n, 0.35, np.float32), color=col)
    return {'BSSRDF': cl, 'BSDF': cl}


def n_emission(ev, node):
    col = ev.input(node, 'Color', RGBA)
    strength = ev.input(node, 'Strength', VALUE)
    cl = Closure()
    cl.add('EMISSION', _w(ev), color=col, strength=strength)
    return {'Emission': cl}


def n_background(ev, node):
    col = ev.input(node, 'Color', RGBA)
    strength = ev.input(node, 'Strength', VALUE)
    cl = Closure()
    cl.add('BACKGROUND', _w(ev), color=col, strength=strength)
    return {'Background': cl}


def n_holdout(ev, node):
    cl = Closure()
    cl.add('HOLDOUT', _w(ev))
    return {'Holdout': cl}


def n_principled(ev, node):
    base = ev.input(node, 'Base Color', RGBA)
    metallic = ev.input(node, 'Metallic', VALUE)
    rough = ev.input(node, 'Roughness', VALUE)
    ior = ev.input(node, 'IOR', VALUE)
    alpha = ev.input(node, 'Alpha', VALUE)
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None
    spec = ev.input(node, 'Specular IOR Level', VALUE) if any(
        s.get('name') == 'Specular IOR Level' for s in node.get('inputs', [])) \
        else ev.input(node, 'Specular', VALUE)
    trans = ev.input(node, 'Transmission Weight', VALUE) if any(
        s.get('name') == 'Transmission Weight' for s in node.get('inputs', [])) \
        else ev.input(node, 'Transmission', VALUE)
    emis = ev.input(node, 'Emission Color', VALUE * 0 + RGBA) if any(
        s.get('name') == 'Emission Color' for s in node.get('inputs', [])) else None
    estr = ev.input(node, 'Emission Strength', VALUE) if any(
        s.get('name') == 'Emission Strength' for s in node.get('inputs', [])) else None
    aniso = ev.input(node, 'Anisotropic', VALUE) if any(
        s.get('name') == 'Anisotropic' for s in node.get('inputs', [])) else None
    cl = Closure()
    diff_w = (1.0 - metallic) * (1.0 - np.clip(trans, 0, 1))
    cl.add('DIFFUSE', diff_w, color=base, roughness=rough, normal=nrm,
           model='OREN_NAYAR')
    gloss_col = base.copy()
    gloss_col[:, :3] = base[:, :3] * metallic[:, None] + \
        (1.0 - metallic[:, None]) * 1.0
    cl.add('GLOSSY', np.maximum(spec, metallic), color=gloss_col, roughness=rough,
           normal=nrm, model='COOK_TORRANCE',
           anisotropy=aniso if aniso is not None else np.zeros(ev.n, np.float32),
           metallic=metallic)
    if np.any(trans > 1e-4):
        cl.add('GLASS', np.clip(trans, 0, 1) * (1.0 - metallic), color=base,
               roughness=rough, ior=ior, normal=nrm)
    if emis is not None and estr is not None and np.any(estr > 1e-6):
        cl.add('EMISSION', _w(ev), color=emis, strength=estr)
    if np.any(alpha < 1.0):
        cl.add('TRANSPARENT', 1.0 - np.clip(alpha, 0, 1),
               color=np.ones((ev.n, 4), np.float32))
    return {'BSDF': cl}


def n_mix_shader(ev, node):
    fac = np.clip(ev.input(node, 'Fac', VALUE), 0.0, 1.0)
    a = ev.input(node, 1, SHADER)
    b = ev.input(node, 2, SHADER)
    out = Closure()
    for kind, w, p in a.items:
        out.items.append((kind, w * (1.0 - fac), p))
    for kind, w, p in b.items:
        out.items.append((kind, w * fac, p))
    return {'Shader': out}


def n_add_shader(ev, node):
    a = ev.input(node, 0, SHADER)
    b = ev.input(node, 1, SHADER)
    return {'Shader': a + b}


def n_output_material(ev, node):
    return {}


# ------------------------------------------- named honesty for what is NOT

def _n_named(reason, outputs=None):
    """A node this renderer will not pretend to run: the evaluation names
    itself and WHY (surfacing in the render warnings), then produces the
    least-wrong neutral output instead of the generic passthrough."""
    def n(ev, node):
        ev.unsupported.add(f"{node.get('bl_idname', '?')} ({reason})")
        return outputs(ev, node) if outputs else ev.fallback(node)
    return n


_NO_VOLUME = ('volumetrics are not in this renderer; the era faked them -- '
              'Height Fog and a spot light\'s Volumetric cone are the tools')


def _n_empty_volume(ev, node):
    return {'Volume': Closure()}


def _n_bevel_out(ev, node):
    nrm = ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') \
        else ev.ctx.N
    return {'Normal': nrm}


def _n_falloff_out(ev, node):
    s = ev.input(node, 'Strength', VALUE)
    return {'Quadratic': s, 'Linear': s, 'Constant': s}


# ------------------------------------------------------------- Halcyon nodes


def _opt(ev, node, name, kind, default):
    """Read a socket, or fall back if the node predates it.

    A material saved before an input existed has no such socket, and reading a
    missing one yields zero -- which for Edge Opacity meant every older material
    silently turned its silhouette invisible. Newer inputs must default to
    doing nothing.
    """
    for sock in node.get('inputs', ()):
        if sock.get('name') == name:
            return ev.input(node, name, kind)
    n = ev.n
    if kind == RGBA:
        out = np.ones((n, 4), np.float32)
        out[:, :3] = np.asarray(default, np.float32)
        return out
    return np.full(n, float(default), np.float32)


def n_halcyon_shader(ev, node):
    p = node.get('props', {})
    model = p.get('model', 'PHONG')
    cl = Closure()

    # Vertex colour blends over the diffuse input rather than sitting beside
    # it. An unlinked socket reads the mesh's own colour attribute, because in
    # the packages that had this a vertex colour was a property of the model,
    # not something you routed through a graph.
    base = ev.input(node, 'Diffuse Color', RGBA)
    vmix = np.clip(_opt(ev, node, 'Vertex Color Mix', VALUE, 0.0), 0.0, 1.0)
    if np.any(vmix > 0.0):
        if ev.has_link(node, 'Vertex Color'):
            vcol = ev.input(node, 'Vertex Color', RGBA)
        else:
            mesh_col = getattr(ev.ctx, 'vcol', None)
            vcol = (coerce(mesh_col, RGBA, ev.n) if mesh_col is not None
                    else coerce((1.0, 1.0, 1.0, 1.0), RGBA, ev.n))
        base = base + (vcol - base) * vmix[:, None]

    kw = dict(
        color=base,
        diffuse_level=ev.input(node, 'Diffuse Level', VALUE),
        spec_color=ev.input(node, 'Specular Color', RGBA),
        spec_level=ev.input(node, 'Specular Level', VALUE),
        glossiness=ev.input(node, 'Glossiness', VALUE),
        roughness=ev.input(node, 'Roughness', VALUE),
        ambient=ev.input(node, 'Ambient', VALUE),
        emission=ev.input(node, 'Self-Illumination', RGBA),
        opacity=ev.input(node, 'Opacity', VALUE),
        ior=ev.input(node, 'IOR', VALUE),
        anisotropy=ev.input(node, 'Anisotropy', VALUE),
        rotation=ev.input(node, 'Anisotropic Rotation', VALUE),
        metallic=ev.input(node, 'Metalness', VALUE),
        soften=ev.input(node, 'Soften', VALUE),
        reflect=ev.input(node, 'Reflection', VALUE),
        translucency=ev.input(node, 'Translucency', VALUE),
        fresnel=_opt(ev, node, 'Fresnel', VALUE, 0.0),
        fresnel_power=_opt(ev, node, 'Fresnel Power', VALUE, 3.0),
        fresnel_color=_opt(ev, node, 'Fresnel Color', RGBA, (1, 1, 1)),
        rim_color=_opt(ev, node, 'Rim Light', RGBA, (1, 1, 1)),
        rim=_opt(ev, node, 'Rim Amount', VALUE, 0.0),
        rim_power=_opt(ev, node, 'Rim Power', VALUE, 3.0),
        matcap=_opt(ev, node, 'Matcap', RGBA, (0, 0, 0)),
        matcap_blend=_opt(ev, node, 'Matcap Blend', VALUE, 0.0),
        reflect_color=_opt(ev, node, 'Reflection Color', RGBA, (1, 1, 1)),
        edge_opacity=_opt(ev, node, 'Edge Opacity', VALUE, 1.0),
        backface_color=_opt(ev, node, 'Backface Color', RGBA, (0, 0, 0)),
        backface_mix=_opt(ev, node, 'Backface Mix', VALUE, 0.0),
        sheen=_opt(ev, node, 'Sheen', VALUE, 0.0),
        sheen_color=_opt(ev, node, 'Sheen Color', RGBA, (1, 1, 1)),
        sheen_roughness=_opt(ev, node, 'Sheen Roughness', VALUE, 0.3),
        bump_strength=_opt(ev, node, 'Bump Strength', VALUE, 1.0),
        refraction=_opt(ev, node, 'Refraction Amount', VALUE, 1.0),
        toon_size=ev.input(node, 'Toon Size', VALUE),
        toon_smooth=ev.input(node, 'Toon Smooth', VALUE),
        # Toon Steps is a node property rather than a socket -- it is a count,
        # and a count has no business being a float someone can plug a texture
        # into. It still has to reach the surface, which for four releases it
        # did not: the shading code read the default 2 whatever the node said.
        toon_steps=float(p.get('toon_steps', 2) or 2),
        normal=ev.input(node, 'Normal', VECTOR) if ev.has_link(node, 'Normal') else None,
        model=model,
    )
    cl.add('HALCYON', _w(ev), **kw)
    return {'Surface': cl, 'BSDF': cl}


def n_halcyon_code(ev, node):
    """Coded-shader node: run the compiled GLSL/HLSL program."""
    c = ev.ctx
    key = node.get('id')
    prog = ev.programs.get(key)
    outs = {}
    if prog is None:
        for o in node.get('outputs', []):
            outs[o.get('name')] = coerce(None, o.get('type', RGBA), ev.n)
        return outs
    uniforms = {}
    for s in node.get('inputs', []):
        nm = s.get('identifier') or s.get('name')
        gl = s.get('uniform') or nm
        if s.get('is_image'):
            uniforms[gl] = ev.images.get(s.get('image'))
            continue
        k = s.get('type', VALUE)
        val = ev.input(node, s.get('name'), k)
        if k == VALUE:
            uniforms[gl] = np.asarray(val, np.float32)
        elif k == RGBA:
            uniforms[gl] = np.asarray(val, np.float32)
        else:
            uniforms[gl] = np.asarray(val, np.float32)
    inputs = _shader_inputs(c)
    from ..shaders.builtins import Ctx as SCtx
    sctx = SCtx(n=c.n, px=c.px, py=c.py, width=c.width, height=c.height,
                tri=c.tri, filt=getattr(c.settings, 'tex_filter', 'NEAREST'),
                wrap='REPEAT')
    res, discard = prog.run(uniforms, inputs, c.n, ctx=sctx)
    for o in node.get('outputs', []):
        key_ = o.get('key') or o.get('name')
        v = res.get(key_)
        if v is None and o.get('name') == 'Result':
            v = res.get('__return')
        outs[o.get('name')] = coerce(v, o.get('type', RGBA), ev.n)
    if _prop(node, 'as_surface', False):
        cl = Closure()
        col = coerce(res.get(list(res.keys())[0]) if res else None, RGBA, ev.n)
        cl.add('EMISSION', _w(ev), color=col,
               strength=np.ones(ev.n, np.float32))
        outs['Surface'] = cl
    outs['__discard'] = discard
    return outs


def _shader_inputs(c):
    n = c.n
    fc = np.zeros((n, 4), np.float32)
    if c.px is not None:
        fc[:, 0] = c.px + 0.5
        fc[:, 1] = c.py + 0.5
    fc[:, 2] = c.depth
    fc[:, 3] = 1.0
    view = -M.normalize(c.I)
    return {
        'fragcoord': fc, 'fragcoord2': fc[:, :2],
        'position': c.P, 'normal': c.N, 'geonormal': c.Ng,
        'tangent': c.T if c.T is not None else M.orthonormal_basis(c.N)[0],
        'uv': c.uv, 'uv2': c.uv2, 'color': c.vcol,
        'view': view, 'incident': M.normalize(c.I),
        'object': c.generated, 'screenuv': fc[:, :2] / max(c.width, 1),
        'depth': c.depth, 'backfacing': c.backfacing,
        'camera': np.broadcast_to(c.camera_pos[None, :], (n, 3)).copy(),
        'time': np.float32(c.time), 'frame': np.int32(c.frame),
        'resolution': np.array([[c.width, c.height, 1.0]], np.float32),
        'random': c.random,
    }


def n_reroute(ev, node):
    ins = node.get('inputs', [])
    if not ins:
        return {}
    s = ins[0]
    link = s.get('link')
    v = ev.eval_output(link[0], link[1]) if link else s.get('default')
    outs = node.get('outputs', [])
    return {outs[0].get('name') if outs else 'Output': v}


def n_group(ev, node):
    """Node groups: evaluate the inner tree with the outer inputs bound."""
    inner = node.get('group')
    if not inner:
        return ev.fallback(node)
    sub = GraphEvaluator({'nodes': inner.get('nodes', {}),
                          'output': inner.get('output')}, ev.ctx,
                         ev.images, ev.programs)
    sub.cache = {}
    bound = {}
    for i, s in enumerate(node.get('inputs', [])):
        k = s.get('type', VALUE)
        bound[i] = ev.input(node, i, k)
        bound[s.get('name')] = bound[i]
    sub.group_inputs = bound
    out_id = inner.get('group_output')
    res = {}
    if out_id and out_id in sub.nodes:
        onode = sub.nodes[out_id]
        for i, s in enumerate(onode.get('inputs', [])):
            nm = s.get('name')
            if nm in (None, ''):
                continue
            res[nm] = sub.input(onode, i, s.get('type', VALUE))
    ev.unsupported |= sub.unsupported
    return res


def n_group_input(ev, node):
    bound = getattr(ev, 'group_inputs', {}) or {}
    out = {}
    for i, o in enumerate(node.get('outputs', [])):
        nm = o.get('name')
        v = bound.get(nm, bound.get(i))
        out[nm] = coerce(v, o.get('type', VALUE), ev.n)
    return out


def n_group_output(ev, node):
    return {}



def n_halcyon_posterize(ev, node):
    col = ev.input(node, 'Color', RGBA)
    lv = np.maximum(ev.input(node, 'Levels', VALUE), 1.0)[:, None]
    rgb = np.floor(np.clip(col[:, :3], 0.0, 1.0) * lv) / np.maximum(lv - 1.0, 1.0)
    return {'Color': np.concatenate([np.clip(rgb, 0, 1), col[:, 3:]], 1).astype(np.float32)}


def n_halcyon_dither(ev, node):
    from .dither import threshold_map
    c = ev.ctx
    col = ev.input(node, 'Color', RGBA)
    lv = np.maximum(ev.input(node, 'Levels', VALUE), 2.0)
    strength = ev.input(node, 'Strength', VALUE)
    kind = _prop(node, 'pattern', 'BAYER4')
    if c.px is None:
        t = np.zeros(c.n, np.float32)
    else:
        tm = threshold_map(kind, 64, 64)
        t = tm[np.asarray(c.py) % tm.shape[0], np.asarray(c.px) % tm.shape[1]]
        t = np.asarray(t, np.float32) - 0.5
    step = 1.0 / np.maximum(lv - 1.0, 1.0)
    biased = col[:, :3] + (t * strength * step)[:, None]
    q = np.round(np.clip(biased, 0.0, 1.0) * (lv - 1.0)[:, None]) / \
        np.maximum(lv - 1.0, 1.0)[:, None]
    return {'Color': np.concatenate([np.clip(q, 0, 1), col[:, 3:]], 1).astype(np.float32)}


def n_halcyon_depth_cue(ev, node):
    c = ev.ctx
    col = ev.input(node, 'Color', RGBA)
    fog = ev.input(node, 'Fog Color', RGBA)
    start = ev.input(node, 'Start', VALUE)
    end = ev.input(node, 'End', VALUE)
    mode = _prop(node, 'mode', 'LINEAR')
    d = c.depth
    if mode == 'EXP':
        f = np.exp(-np.maximum(d - start, 0.0) / np.maximum(end - start, 1e-5) * 3.0)
    elif mode == 'EXP2':
        t = np.maximum(d - start, 0.0) / np.maximum(end - start, 1e-5)
        f = np.exp(-(t * t) * 3.0)
    elif mode == 'TABLE16':
        t = np.clip((d - start) / np.maximum(end - start, 1e-5), 0.0, 1.0)
        f = 1.0 - np.floor(t * 16.0) / 16.0
    else:
        f = (end - d) / np.maximum(end - start, 1e-5)
    f = np.clip(f, 0.0, 1.0)[:, None]
    rgb = col[:, :3] * f + fog[:, :3] * (1.0 - f)
    return {'Color': np.concatenate([rgb, col[:, 3:]], 1).astype(np.float32)}


def n_halcyon_screen_info(ev, node):
    c = ev.ctx
    n = c.n
    suv = np.zeros((n, 3), np.float32)
    pix = np.zeros((n, 3), np.float32)
    if c.px is not None and c.width:
        suv[:, 0] = (np.asarray(c.px) + 0.5) / c.width
        suv[:, 1] = (np.asarray(c.py) + 0.5) / c.height
        pix[:, 0] = np.asarray(c.px, np.float32)
        pix[:, 1] = np.asarray(c.py, np.float32)
    facing = np.abs(M.dot(M.normalize(c.N), -M.normalize(c.I)))
    return {'Screen UV': suv, 'Pixel': pix, 'Depth': c.depth,
            'Facing': facing.astype(np.float32),
            'Frame': np.full(n, float(c.frame), np.float32),
            'Time': np.full(n, float(c.time), np.float32)}


def n_halcyon_pixelate(ev, node):
    """Snap each axis to the centre of a coarse cell. Counts below 1 leave
    the axis untouched, which is what makes the Z default of 0 mean 2D.
    The cell index clamps to count-1, standard texel addressing -- a
    coordinate of exactly 1.0 belongs to the LAST texel, not one past."""
    v = _tex_vector(ev, node, 'uv')
    out = np.array(v, np.float32, copy=True)
    for axis, name in enumerate(('Pixels X', 'Pixels Y', 'Pixels Z')):
        cnt = ev.input(node, name, VALUE)
        active = cnt >= 1.0
        safe = np.maximum(cnt, 1.0)
        snapped = (np.minimum(np.floor(v[:, axis] * safe), safe - 1.0)
                   + 0.5) / safe
        out[:, axis] = np.where(active, snapped, v[:, axis])
    return {'Vector': out.astype(np.float32)}


def _stepped_time(ctx, node):
    """The scroll clock: ctx.time, optionally quantised to whole steps."""
    t = float(ctx.time) if _prop(node, 'animate', True) else 0.0
    fps = int(_prop(node, 'fps', 0) or 0)
    if fps > 0:
        t = np.floor(np.float32(t) * fps) / np.float32(fps)
    return np.float32(t)


def n_halcyon_scroll(ev, node):
    """The era's texture animation: offset and spin the UV over time."""
    v = _tex_vector(ev, node, 'uv')
    t = _stepped_time(ev.ctx, node)
    sx = ev.input(node, 'Scroll X', VALUE)
    sy = ev.input(node, 'Scroll Y', VALUE)
    spin = ev.input(node, 'Spin', VALUE)
    ang = spin * (t * np.float32(2.0 * np.pi))
    ca, sa = np.cos(ang), np.sin(ang)
    dx = v[:, 0] - 0.5
    dy = v[:, 1] - 0.5
    out = np.array(v, np.float32, copy=True)
    out[:, 0] = (dx * ca - dy * sa) + 0.5 + sx * t
    out[:, 1] = (dx * sa + dy * ca) + 0.5 + sy * t
    return {'Vector': out.astype(np.float32)}


def n_halcyon_scanlines(ev, node):
    """Alternate-line darkening in the SURFACE's own space -- an in-scene
    CRT. The camera-space version of this look is the CRT post stage; this
    node is for the television standing in the shot."""
    c = ev.ctx
    col = ev.input(node, 'Color', RGBA)
    v = _tex_vector(ev, node, 'uv')
    lines = np.maximum(ev.input(node, 'Lines', VALUE), 1.0)
    dark = np.clip(ev.input(node, 'Darkness', VALUE), 0.0, 1.0)
    thick = np.clip(ev.input(node, 'Thickness', VALUE), 0.0, 1.0)
    t = float(c.time) if _prop(node, 'animate', False) else 0.0
    # six lines a second when rolling -- a set with its vertical hold off
    y = v[:, 1] * lines - np.float32(t) * 6.0
    on = (y - np.floor(y)) < thick
    mul = 1.0 - dark * on.astype(np.float32)
    rgb = col[:, :3] * mul[:, None]
    return {'Color': np.concatenate([rgb, col[:, 3:]], 1).astype(np.float32)}


def n_halcyon_palette(ev, node):
    """Nearest entry of a period hardware palette, or a 3-3-2 bit crush."""
    from .palette import NODE_PALETTES
    col = ev.input(node, 'Color', RGBA)
    mix = np.clip(ev.input(node, 'Mix', VALUE), 0.0, 1.0)
    mode = _prop(node, 'palette', 'EGA')
    src = np.clip(col[:, :3].astype(np.float32), 0.0, 1.0)
    if mode == 'RGB332':
        r = np.round(src[:, 0] * 7.0) / np.float32(7.0)
        g = np.round(src[:, 1] * 7.0) / np.float32(7.0)
        b = np.round(src[:, 2] * 3.0) / np.float32(3.0)
        snapped = np.stack([r, g, b], 1)
        idx = (np.round(src[:, 0] * 7.0) * 32.0 + np.round(src[:, 1] * 7.0)
               * 4.0 + np.round(src[:, 2] * 3.0)) / np.float32(255.0)
    else:
        pal = NODE_PALETTES.get(mode, NODE_PALETTES['EGA'])
        d = src[:, None, :] - pal[None, :, :]
        d = (d * d).sum(axis=2)
        pick = np.argmin(d, axis=1)
        snapped = pal[pick]
        idx = pick.astype(np.float32) / np.float32(max(len(pal) - 1, 1))
    rgb = src + (snapped - src) * mix[:, None]
    return {'Color': np.concatenate([rgb, col[:, 3:]], 1).astype(np.float32),
            'Index': idx.astype(np.float32)}


def n_halcyon_flipbook(ev, node):
    """An N-by-M sprite sheet as an animated texture -- the era's fire,
    explosions and waterfalls. Cells read left to right, TOP row first
    (sheets are authored top-down; UVs run bottom-up), wrapping at the
    end."""
    c = ev.ctx
    v = _tex_vector(ev, node, 'uv')
    cols = np.maximum(np.floor(ev.input(node, 'Columns', VALUE)), 1.0)
    rows = np.maximum(np.floor(ev.input(node, 'Rows', VALUE)), 1.0)
    rate = ev.input(node, 'Rate', VALUE)
    offset = ev.input(node, 'Cell Offset', VALUE)
    t = np.float32(float(c.time) if _prop(node, 'animate', True) else 0.0)
    cell0 = np.floor(offset + t * rate)
    total = cols * rows
    cell = cell0 - total * np.floor(cell0 / total)
    cy = np.floor(cell / cols)
    cx = cell - cols * cy
    out = np.array(v, np.float32, copy=True)
    out[:, 0] = (v[:, 0] + cx) / cols
    out[:, 1] = (v[:, 1] + (rows - 1.0 - cy)) / rows
    return {'Vector': out.astype(np.float32)}


def n_halcyon_uv_wave(ev, node):
    """Sine-warp -- the underwater wobble, heat haze, Mode-7 waves."""
    c = ev.ctx
    v = _tex_vector(ev, node, 'uv')
    ax = ev.input(node, 'Amplitude X', VALUE)
    ay = ev.input(node, 'Amplitude Y', VALUE)
    freq = ev.input(node, 'Frequency', VALUE)
    speed = ev.input(node, 'Speed', VALUE)
    t = np.float32(float(c.time) if _prop(node, 'animate', True) else 0.0)
    ph = t * speed
    tp = np.float32(2.0 * np.pi)
    out = np.array(v, np.float32, copy=True)
    out[:, 0] = v[:, 0] + np.sin((v[:, 1] * freq + ph) * tp) * ax
    out[:, 1] = v[:, 1] + np.sin((v[:, 0] * freq + ph + 0.25) * tp) * ay
    return {'Vector': out.astype(np.float32)}


def n_halcyon_halftone(ev, node):
    """A rotated dot screen: dots grow where the input darkens. The dot
    radius is 0.7071*sqrt(1-luma), so full black covers the whole cell
    (the corner is 0.7071 away) and white prints nothing. Luma is
    Rec.601 -- the NTSC weights, the period answer."""
    col = ev.input(node, 'Color', RGBA)
    v = _tex_vector(ev, node, 'uv')
    dots = np.maximum(ev.input(node, 'Dots', VALUE), 1e-3)
    ang = ev.input(node, 'Angle', VALUE) * np.float32(np.pi / 180.0)
    ca, sa = np.cos(ang), np.sin(ang)
    src = np.clip(col[:, :3], 0.0, 1.0)
    luma = src[:, 0] * np.float32(0.299) + src[:, 1] * np.float32(0.587) \
        + src[:, 2] * np.float32(0.114)
    gx = (v[:, 0] * ca - v[:, 1] * sa) * dots
    gy = (v[:, 0] * sa + v[:, 1] * ca) * dots
    fx = gx - np.floor(gx) - 0.5
    fy = gy - np.floor(gy) - 0.5
    d = np.sqrt(fx * fx + fy * fy)
    r = np.float32(0.70710678) * np.sqrt(np.maximum(1.0 - luma, 0.0))
    ink = (d < r).astype(np.float32)
    paper = ev.input(node, 'Paper Color', RGBA)
    inkc = ev.input(node, 'Ink Color', RGBA)
    rgb = paper[:, :3] + (inkc[:, :3] - paper[:, :3]) * ink[:, None]
    return {'Color': np.concatenate([rgb, col[:, 3:]], 1).astype(np.float32),
            'Fac': ink}


def n_halcyon_threshold(ev, node):
    """0 or 1 at a level, with an optional smoothstep edge."""
    fac = ev.input(node, 'Fac', VALUE)
    level = ev.input(node, 'Level', VALUE)
    smooth = ev.input(node, 'Smooth', VALUE)
    hard = (fac >= level).astype(np.float32)
    t = np.clip((fac - (level - smooth * 0.5)) / np.maximum(smooth, 1e-6),
                0.0, 1.0)
    soft = t * t * (3.0 - 2.0 * t)
    return {'Fac': np.where(smooth > 1e-6, soft,
                            hard).astype(np.float32)}


def n_halcyon_quantize(ev, node):
    """Posterize for one value, with Posterize's own arithmetic."""
    fac = ev.input(node, 'Fac', VALUE)
    s = np.maximum(np.floor(ev.input(node, 'Steps', VALUE)), 1.0)
    q = np.floor(np.clip(fac, 0.0, 1.0) * s) / np.maximum(s - 1.0, 1.0)
    return {'Fac': np.clip(q, 0.0, 1.0).astype(np.float32)}


def n_halcyon_color_cycle(ev, node):
    """Rotate a ramp phase over time -- palette-register colour cycling.

    Put it between a texture's Fac and a Color Ramp: the ramp's colours
    march through the pattern, which is how the era animated waterfalls
    without moving a single vertex."""
    c = ev.ctx
    fac = ev.input(node, 'Fac', VALUE)
    speed = ev.input(node, 'Speed', VALUE)
    steps = ev.input(node, 'Steps', VALUE)
    t = np.float32(float(c.time) if _prop(node, 'animate', True) else 0.0)
    phase = speed * t
    q = np.maximum(np.floor(steps), 1.0)
    stepped = np.floor(phase * q) / q
    phase = np.where(steps >= 1.0, stepped, phase)
    out = fac + phase
    return {'Fac': (out - np.floor(out)).astype(np.float32)}



# Preetham et al., "A Practical Analytic Model for Daylight" (SIGGRAPH 1999) --
# in period, and the model Blender's own Preetham sky option implements.
_PREETHAM = {
    'Y':  ((0.1787, -1.4630), (-0.3554, 0.4275), (-0.0227, 5.3251),
           (0.1206, -2.5771), (-0.0670, 0.3703)),
    'x':  ((-0.0193, -0.2592), (-0.0665, 0.0008), (-0.0004, 0.2125),
           (-0.0641, -0.8989), (-0.0033, 0.0452)),
    'y':  ((-0.0167, -0.2608), (-0.0950, 0.0092), (-0.0079, 0.2102),
           (-0.0441, -1.6537), (-0.0109, 0.0529)),
}


def _preetham_coeffs(channel, T):
    return [a * T + b for a, b in _PREETHAM[channel]]


def _perez(cos_theta, gamma, cos_gamma, c):
    A, B, C, D, E = c
    cos_theta = np.maximum(cos_theta, 0.01)
    return ((1.0 + A * np.exp(B / cos_theta)) *
            (1.0 + C * np.exp(D * gamma) + E * cos_gamma * cos_gamma))


def _xyY_to_rgb(x, y, Y):
    y = np.where(np.abs(y) < 1e-5, 1e-5, y)
    X = x * Y / y
    Z = (1.0 - x - y) * Y / y
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    b = 0.0557 * X - 0.2040 * Y + 1.0570 * Z
    return np.clip(np.stack([r, g, b], axis=1), 0.0, None).astype(np.float32)


def n_tex_sky(ev, node):
    """Sky Texture. Analytic daylight, driven by sun elevation and turbidity.

    Blender's Nishita option is a full atmospheric scattering simulation; this
    is the Preetham analytic model, which is what the period would have used and
    is a close enough match in shape and colour for a renderer that is about to
    quantise the result to 256 colours anyway. Sun elevation, rotation, the sun
    disc and turbidity all behave as expected.
    """
    c = ev.ctx
    n = ev.n
    d = M.normalize(_tex_vector(ev, node, 'generated')) if ev.has_link(node, 'Vector') \
        else M.normalize(c.I) * -1.0 if c.I is not None else np.tile(
            np.array([[0, 0, 1.0]], np.float32), (n, 1))
    # the background pass points I along the view ray, so the sky direction is
    # the ray direction itself
    if not ev.has_link(node, 'Vector') and c.I is not None:
        d = M.normalize(c.I)

    elev = float(_prop(node, 'sun_elevation', 0.26))
    rot = float(_prop(node, 'sun_rotation', 0.0))
    T = float(np.clip(_prop(node, 'turbidity', 2.2), 1.0, 10.0))
    strength = float(_prop(node, 'sun_intensity', 1.0))
    sun_disc = bool(_prop(node, 'sun_disc', True))
    sun_size = float(_prop(node, 'sun_size', 0.009512))
    ground = float(_prop(node, 'ground_albedo', 0.3))

    sun = np.array([np.cos(elev) * np.cos(rot), np.cos(elev) * np.sin(rot),
                    np.sin(elev)], np.float32)

    up = np.clip(d[:, 2], -1.0, 1.0)
    cos_theta = np.maximum(up, 0.0)
    cos_gamma = np.clip(d @ sun, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    theta_s = max(np.pi * 0.5 - elev, 0.0)
    chi = (4.0 / 9.0 - T / 120.0) * (np.pi - 2.0 * theta_s)
    zenith_Y = (4.0453 * T - 4.9710) * np.tan(chi) - 0.2155 * T + 2.4192
    zenith_Y = max(zenith_Y, 0.0) * 0.06

    t2, t3 = theta_s ** 2, theta_s ** 3
    T2 = T * T
    zx = ((0.00166 * t3 - 0.00375 * t2 + 0.00209 * theta_s) * T2 +
          (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * theta_s + 0.00394) * T +
          (0.11693 * t3 - 0.21196 * t2 + 0.06052 * theta_s + 0.25886))
    zy = ((0.00275 * t3 - 0.00610 * t2 + 0.00317 * theta_s) * T2 +
          (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * theta_s + 0.00516) * T +
          (0.15346 * t3 - 0.26756 * t2 + 0.06670 * theta_s + 0.26688))

    cs = np.cos(theta_s)
    denom = [_perez(np.array([max(cs, 0.01)], np.float32),
                    np.array([theta_s], np.float32),
                    np.array([np.cos(theta_s)], np.float32),
                    _preetham_coeffs(ch, T))[0] for ch in ('Y', 'x', 'y')]
    num_Y = _perez(cos_theta, gamma, cos_gamma, _preetham_coeffs('Y', T))
    num_x = _perez(cos_theta, gamma, cos_gamma, _preetham_coeffs('x', T))
    num_y = _perez(cos_theta, gamma, cos_gamma, _preetham_coeffs('y', T))

    Y = zenith_Y * num_Y / max(denom[0], 1e-4)
    x = zx * num_x / max(denom[1], 1e-4)
    y = zy * num_y / max(denom[2], 1e-4)
    rgb = _xyY_to_rgb(x, y, Y) * strength

    if sun_disc:
        disc = gamma < max(sun_size, 1e-4)
        if disc.any():
            rgb[disc] += np.array([1.0, 0.95, 0.85], np.float32) * 8.0 * strength

    # below the horizon the model is undefined; fade to a ground colour
    below = up < 0.0
    if below.any():
        t = np.clip(-up[below] * 4.0, 0.0, 1.0)[:, None]
        gcol = np.array([ground, ground * 0.95, ground * 0.85], np.float32)
        horizon = rgb[below]
        rgb[below] = horizon * (1.0 - t) + gcol[None, :] * t

    out = np.ones((n, 4), np.float32)
    out[:, :3] = np.nan_to_num(rgb, nan=0.0, posinf=1.0, neginf=0.0)
    return {'Color': out}



def n_tex_gabor(ev, node):
    """Gabor noise (Blender 4.3+). Sum of randomly-oriented Gabor kernels.

    Lagae et al., "Procedural Noise using Sparse Gabor Convolution" -- a windowed
    cosine per impulse, which is what gives it the directional, banded character
    a plain value noise cannot produce.
    """
    vec = _tex_vector(ev, node, 'generated')
    scale = ev.input(node, 'Scale', VALUE)
    freq = np.maximum(ev.input(node, 'Frequency', VALUE), 1e-4)
    aniso = np.clip(ev.input(node, 'Anisotropy', VALUE), 0.0, 1.0)
    orient = ev.input(node, 'Orientation', VALUE)
    gtype = _prop(node, 'gabor_type', '2D')

    p = vec * scale[:, None]
    if gtype == '2D':
        p = p.copy()
        p[:, 2] = 0.0

    cell = np.floor(p)
    acc = np.zeros(ev.n, np.float32)
    weight = 0.0
    zr = (0,) if gtype == '2D' else (-1, 0, 1)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in zr:
                off = np.array([dx, dy, dz], np.float32)
                c = cell + off
                h1 = _cell_hash(c, 0.0)
                h2 = _cell_hash(c, 17.0)
                h3 = _cell_hash(c, 41.0)
                centre = c + np.stack([h1, h2, h3 if gtype != '2D'
                                       else np.zeros_like(h3)], axis=1)
                d = p - centre
                r2 = (d * d).sum(axis=1)
                # Gaussian window
                g = np.exp(-np.pi * r2 * 2.0)
                # kernel direction: blend a random angle toward the user's
                ang_rand = h3 * 2.0 * np.pi
                ang = ang_rand * (1.0 - aniso) + orient * aniso
                kx, ky = np.cos(ang), np.sin(ang)
                phase = (d[:, 0] * kx + d[:, 1] * ky) * freq * 2.0 * np.pi
                acc += g * np.cos(phase)
                weight += 1.0
    val = acc / max(np.sqrt(weight), 1e-6)
    fac = np.clip(val * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
    col = np.stack([fac, fac, fac, np.ones_like(fac)], axis=1)
    return {'Value': fac, 'Fac': fac, 'Color': col, 'Phase': fac,
            'Intensity': np.abs(val).astype(np.float32)}


def _cell_hash(c, salt):
    x = c[:, 0] * 127.1 + c[:, 1] * 311.7 + c[:, 2] * 74.7 + salt
    h = np.sin(x) * 43758.5453
    return (h - np.floor(h)).astype(np.float32)



def _fractal_variant(kind, p, detail, rough, lac, offset, gain):
    """Multifractal family, from Musgrave's chapter in Texturing & Modeling.

    Blender merged these into the Noise node's `noise_type`; each one builds its
    octaves differently, so they cannot share fBm's accumulation.
    """
    octaves = int(np.clip(np.mean(detail), 1, 12))
    H = np.clip(np.mean(rough), 0.0, 1.0)
    L = float(np.clip(np.mean(lac), 1.001, 8.0))
    off = np.mean(offset)
    gn = np.mean(gain)
    freq = 1.0
    if kind == 'MULTIFRACTAL':
        val = np.ones(p.shape[0], np.float32)
        for i in range(octaves):
            val *= (perlin(p * freq) * (L ** (-H * i)) + 1.0)
            freq *= L
        return val - 1.0
    if kind == 'HETERO_TERRAIN':
        val = perlin(p) + off
        freq = L
        for i in range(1, octaves):
            incr = (perlin(p * freq) + off) * (L ** (-H * i))
            val += incr * val
            freq *= L
        return val - 1.0
    if kind in ('HYBRID_MULTIFRACTAL', 'RIDGED_MULTIFRACTAL'):
        ridged = kind == 'RIDGED_MULTIFRACTAL'
        sig = perlin(p)
        if ridged:
            sig = off - np.abs(sig)
            sig = sig * sig
        else:
            sig = sig + off
        val = sig.copy()
        w = sig.copy()
        freq = L
        for i in range(1, octaves):
            w = np.clip(w * gn, 0.0, 1.0)
            s2 = perlin(p * freq)
            if ridged:
                s2 = off - np.abs(s2)
                s2 = s2 * s2
            else:
                s2 = s2 + off
            s2 = s2 * (L ** (-H * i))
            val += w * s2
            w = w * s2
            freq *= L
        return val - 1.0
    return fractal_noise(p, detail, rough, lac)



# --------------------------------------------------- Halcyon pattern textures

def _pat_vec(ev, node):
    return _tex_vector(ev, node, 'generated') * ev.input(node, 'Scale', VALUE)[:, None]


def _pat_out(ev, node, fac, col1=None, col2=None):
    """Standard Colour/Fac pair, ramping between the node's two colours."""
    f = np.clip(np.asarray(fac, np.float32), 0.0, 1.0)
    if col1 is None:
        col1 = ev.input(node, 'Color 1', RGBA)
    if col2 is None:
        col2 = ev.input(node, 'Color 2', RGBA)
    col = col1 + (col2 - col1) * f[:, None]
    return {'Color': col.astype(np.float32), 'Fac': f}


def n_pat_marble(ev, node):
    p = _pat_vec(ev, node)
    f = PT.marble(p, ev.input(node, 'Turbulence', VALUE).mean(),
                  int(_prop(node, 'octaves', 5)),
                  ev.input(node, 'Veins', VALUE).mean(),
                  ev.input(node, 'Sharpness', VALUE).mean(),
                  {'X': 0, 'Y': 1, 'Z': 2}.get(_prop(node, 'axis', 'X'), 0))
    return _pat_out(ev, node, f)


def n_pat_wood(ev, node):
    p = _pat_vec(ev, node)
    f = PT.wood(p, ev.input(node, 'Rings', VALUE).mean(),
                ev.input(node, 'Turbulence', VALUE).mean(),
                int(_prop(node, 'octaves', 4)),
                ev.input(node, 'Grain', VALUE).mean(),
                {'X': 0, 'Y': 1, 'Z': 2}.get(_prop(node, 'axis', 'Z'), 2))
    return _pat_out(ev, node, f)


def n_pat_granite(ev, node):
    p = _pat_vec(ev, node)
    f = PT.granite(p, int(_prop(node, 'octaves', 6)),
                   ev.input(node, 'Contrast', VALUE).mean(),
                   ev.input(node, 'Speckle', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_dents(ev, node):
    p = _pat_vec(ev, node)
    f = PT.dents(p, ev.input(node, 'Size', VALUE).mean(),
                 int(_prop(node, 'octaves', 3)),
                 ev.input(node, 'Depth', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_crackle(ev, node):
    p = _pat_vec(ev, node)
    f = PT.crackle(p, ev.input(node, 'Randomness', VALUE).mean(),
                   ev.input(node, 'Width', VALUE).mean(),
                   ev.input(node, 'Smooth', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_plasma(ev, node):
    p = _pat_vec(ev, node)
    t = ev.ctx.time if _prop(node, 'animate', True) else 0.0
    f = PT.plasma(p, 1.0, float(t) * ev.input(node, 'Speed', VALUE).mean(),
                  ev.input(node, 'Complexity', VALUE).mean())
    if _prop(node, 'cycle_palette', True):
        # the palette-cycled rainbow that made plasma demos move
        ph = f * 6.2832 + float(t) * 0.6
        col = np.stack([np.sin(ph) * .5 + .5, np.sin(ph + 2.094) * .5 + .5,
                        np.sin(ph + 4.188) * .5 + .5,
                        np.ones_like(f)], 1).astype(np.float32)
        return {'Color': col, 'Fac': f}
    return _pat_out(ev, node, f)


def n_pat_ripples(ev, node):
    p = _pat_vec(ev, node)
    t = ev.ctx.time if _prop(node, 'animate', True) else 0.0
    f = PT.ripples(p, int(_prop(node, 'sources', 3)),
                   ev.input(node, 'Frequency', VALUE).mean(),
                   float(t) * ev.input(node, 'Speed', VALUE).mean(),
                   ev.input(node, 'Decay', VALUE).mean(),
                   int(_prop(node, 'seed', 0)))
    return _pat_out(ev, node, f)


def n_pat_starfield(ev, node):
    p = _pat_vec(ev, node)
    t = ev.ctx.time
    f = PT.starfield(p, ev.input(node, 'Density', VALUE).mean(),
                     ev.input(node, 'Size', VALUE).mean(),
                     ev.input(node, 'Twinkle', VALUE).mean(), float(t))
    sky = ev.input(node, 'Sky Color', RGBA)
    star = ev.input(node, 'Star Color', RGBA)
    col = sky + (star - sky) * f[:, None]
    return {'Color': col.astype(np.float32), 'Fac': f}


def n_pat_weave(ev, node):
    p = _pat_vec(ev, node)
    f, on_warp = PT.weave(p, ev.input(node, 'Thickness', VALUE).mean(),
                          ev.input(node, 'Gap', VALUE).mean(),
                          ev.input(node, 'Distortion', VALUE).mean())
    warp_c = ev.input(node, 'Warp Color', RGBA)
    weft_c = ev.input(node, 'Weft Color', RGBA)
    base = np.where(on_warp[:, None], warp_c, weft_c)
    col = base * f[:, None]
    col[:, 3] = 1.0
    return {'Color': col.astype(np.float32), 'Fac': f,
            'Thread': on_warp.astype(np.float32)}


def n_pat_scratches(ev, node):
    p = _pat_vec(ev, node)
    f = PT.scratches(p, int(_prop(node, 'count', 6)),
                     ev.input(node, 'Width', VALUE).mean(),
                     ev.input(node, 'Length', VALUE).mean(),
                     int(_prop(node, 'seed', 0)),
                     ev.input(node, 'Anisotropy', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_tiles(ev, node):
    p = _pat_vec(ev, node)
    f, tid, inside = PT.tiles(p, ev.input(node, 'Rows', VALUE).mean(),
                              ev.input(node, 'Columns', VALUE).mean(),
                              ev.input(node, 'Grout', VALUE).mean(),
                              ev.input(node, 'Offset', VALUE).mean(),
                              ev.input(node, 'Bevel', VALUE).mean())
    tile_c = ev.input(node, 'Tile Color', RGBA)
    grout_c = ev.input(node, 'Grout Color', RGBA)
    vary = ev.input(node, 'Variation', VALUE)
    shade = 1.0 + (tid - 0.5) * vary
    col = np.where(inside[:, None], tile_c * (f * shade)[:, None], grout_c)
    col[:, 3] = 1.0
    return {'Color': col.astype(np.float32), 'Fac': f, 'Tile ID': tid}


def n_pat_spiral(ev, node):
    p = _pat_vec(ev, node)
    f = PT.spiral(p, ev.input(node, 'Turns', VALUE).mean(),
                  ev.input(node, 'Sharpness', VALUE).mean(),
                  {'X': 0, 'Y': 1, 'Z': 2}.get(_prop(node, 'axis', 'Z'), 2),
                  ev.input(node, 'Twist', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_bozo(ev, node):
    p = _pat_vec(ev, node)
    f = PT.bozo(p, ev.input(node, 'Turbulence', VALUE).mean(),
                int(_prop(node, 'octaves', 4)),
                ev.input(node, 'Lacunarity', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_agate(ev, node):
    p = _pat_vec(ev, node)
    f = PT.agate(p, ev.input(node, 'Turbulence', VALUE).mean(),
                 int(_prop(node, 'octaves', 6)),
                 ev.input(node, 'Bands', VALUE).mean(),
                 ev.input(node, 'Sharpness', VALUE).mean(),
                 {'X': 0, 'Y': 1, 'Z': 2}.get(_prop(node, 'axis', 'Z'), 2))
    return _pat_out(ev, node, f)


def n_pat_leopard(ev, node):
    p = _pat_vec(ev, node)
    f = PT.leopard(p, ev.input(node, 'Spot', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_onion(ev, node):
    p = _pat_vec(ev, node)
    f = PT.onion(p, ev.input(node, 'Thickness', VALUE).mean(),
                 ev.input(node, 'Sharpness', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_bumps(ev, node):
    p = _pat_vec(ev, node)
    f = PT.bumps(p, ev.input(node, 'Roundness', VALUE).mean(),
                 int(_prop(node, 'octaves', 1)),
                 ev.input(node, 'Lacunarity', VALUE).mean(),
                 ev.input(node, 'Gain', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_wrinkles(ev, node):
    p = _pat_vec(ev, node)
    f = PT.wrinkles(p, int(_prop(node, 'octaves', 8)),
                    ev.input(node, 'Lacunarity', VALUE).mean(),
                    ev.input(node, 'Crease', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_brick(ev, node):
    p = _pat_vec(ev, node)
    f, bid, inside = PT.brick(p, ev.input(node, 'Width', VALUE).mean(),
                              ev.input(node, 'Height', VALUE).mean(),
                              ev.input(node, 'Mortar', VALUE).mean(),
                              ev.input(node, 'Offset', VALUE).mean(),
                              ev.input(node, 'Bevel', VALUE).mean())
    brick_c = ev.input(node, 'Brick Color', RGBA)
    mortar_c = ev.input(node, 'Mortar Color', RGBA)
    vary = ev.input(node, 'Variation', VALUE)
    shade = 1.0 + (bid - 0.5) * vary
    col = np.where(inside[:, None], brick_c * (f * shade)[:, None], mortar_c)
    col[:, 3] = 1.0
    return {'Color': col.astype(np.float32), 'Fac': f, 'Brick ID': bid}


_NOISE_KIND = {'SMOOTH': 0, 'TURBULENT': 1, 'RIDGED': 2}
_CELL_FEATURE = {'F1': 0, 'F2': 1, 'BORDER': 2, 'CELL': 3}


def n_pat_noise(ev, node):
    p = _pat_vec(ev, node)
    dims = str(_prop(node, 'dims', '3D'))
    if dims == '1D':
        p = p[:, :1]
    elif dims == '2D':
        p = p[:, :2]
    elif dims == '4D':
        w = ev.input(node, 'W', VALUE) * ev.input(node, 'Scale', VALUE)
        p = np.concatenate([p, w[:, None].astype(np.float32)], axis=1)
    f = PT.noise_fractal(p, _NOISE_KIND.get(_prop(node, 'kind', 'SMOOTH'), 0),
                         int(_prop(node, 'octaves', 5)),
                         ev.input(node, 'Lacunarity', VALUE).mean(),
                         ev.input(node, 'Gain', VALUE).mean())
    return _pat_out(ev, node, f)


def n_pat_water(ev, node):
    p = _pat_vec(ev, node)
    t = ev.ctx.time if _prop(node, 'animate', True) else 0.0
    loop = int(_prop(node, 'loop_frames', 48)) if _prop(node, 'loop', False) \
        else 0
    f = PT.water(p[:, :2], float(t),
                 layers=int(_prop(node, 'layers', 5)),
                 speed=float(ev.input(node, 'Speed', VALUE).mean()),
                 choppiness=float(
                     ev.input(node, 'Choppiness', VALUE).mean()),
                 loop_frames=loop, fps=float(_prop(node, 'fps', 24)))
    return _pat_out(ev, node, f)


def n_pat_gradient_shaped(ev, node):
    c = ev.ctx
    vec = ev.input(node, 'Vector', VECTOR) if ev.has_link(node, 'Vector') \
        else c.generated
    centre = ev.input(node, 'Center', VECTOR)
    scale = ev.input(node, 'Scale', VALUE)
    rot = ev.input(node, 'Rotation', VALUE)
    q = (np.asarray(vec, np.float32) - np.asarray(centre, np.float32)) * \
        scale[:, None]
    ca, sa = np.cos(-rot), np.sin(-rot)
    x = q[:, 0] * ca - q[:, 1] * sa
    y = q[:, 0] * sa + q[:, 1] * ca
    z = q[:, 2]
    shape = str(_prop(node, 'shape', 'LINEAR'))
    if shape == 'REFLECTED':
        f = 1.0 - np.abs(x)
    elif shape == 'SPHERICAL':
        f = 1.0 - np.sqrt(x * x + y * y + z * z)
    elif shape == 'QUADRATIC':
        r = np.maximum(1.0 - np.sqrt(x * x + y * y + z * z), 0.0)
        f = r * r
    elif shape == 'SQUARE':
        f = 1.0 - np.maximum(np.abs(x), np.abs(y))
    elif shape == 'DIAMOND':
        f = 1.0 - (np.abs(x) + np.abs(y))
    elif shape == 'CONICAL':
        f = np.arctan2(y, x) / (2.0 * np.pi) + 0.5
    elif shape == 'SPIRAL':
        f = np.arctan2(y, x) / (2.0 * np.pi) + 0.5 + \
            np.sqrt(x * x + y * y)
        f = f - np.floor(f)
    else:
        f = x + 0.5
    rep = str(_prop(node, 'repeat', 'NONE'))
    if rep == 'REPEAT':
        f = f - np.floor(f)
    elif rep == 'PINGPONG':
        h = (f * 0.5 - np.floor(f * 0.5)) * 2.0    # sawtooth 0..2
        f = 1.0 - np.abs(h - 1.0)                  # triangle 0..1..0
    f = np.clip(f, 0.0, 1.0)
    ease = str(_prop(node, 'easing', 'NONE'))
    if ease == 'SMOOTH':
        f = f * f * (3.0 - 2.0 * f)
    elif ease == 'SHARP':
        f = f * f
    return _pat_out(ev, node, f.astype(np.float32))


def n_pat_cells_tex(ev, node):
    p = _pat_vec(ev, node)
    f, cid = PT.cells(p, ev.input(node, 'Randomness', VALUE).mean(),
                      _CELL_FEATURE.get(_prop(node, 'feature', 'F1'), 0))
    out = _pat_out(ev, node, f)
    out['Cell ID'] = cid
    return out


def n_pat_static(ev, node):
    p = _pat_vec(ev, node)
    frame = int(ev.ctx.frame) if _prop(node, 'animate', True) else 0
    f = PT.tv_static(p, frame)
    return _pat_out(ev, node, f)



def n_halcyon_matcap_uv(ev, node):
    """Sphere-map coordinates from the view-space normal.

    The trick 1990s renderers used for environment reflections, and the same one
    a modern matcap uses: index a picture of a lit sphere by where the normal
    points relative to the camera. One image carries an entire material.
    """
    c = ev.ctx
    N = M.normalize(c.N)
    V = -M.normalize(c.I)
    up = np.array([0.0, 0.0, 1.0], np.float32)
    right = M.normalize(np.cross(np.broadcast_to(up[None, :], V.shape), V))
    degenerate = (right * right).sum(1) < 1e-8
    if degenerate.any():
        right[degenerate] = np.array([1.0, 0.0, 0.0], np.float32)
    upv = np.cross(V, right)
    x = M.dot(N, right)
    y = M.dot(N, upv)
    scale = ev.input(node, 'Scale', VALUE)
    # Centered maps the sphere about the origin (-0.5..0.5) instead of
    # image space (0..1): |vector| then measures distance from the sphere
    # CENTRE, which is what a Spherical gradient wants. Offset shifts the
    # result either way -- the mapping controls a matcap-driven gradient
    # needs to land where the artist points it.
    centre = np.float32(0.0 if _prop(node, 'centered', False) else 0.5)
    off = ev.input(node, 'Offset', VECTOR)   # missing socket coerces to zero
    uv = np.stack([x * 0.5 * scale + centre, y * 0.5 * scale + centre,
                   np.zeros(ev.n, np.float32)], axis=1) + \
        np.asarray(off, np.float32)
    return {'Vector': uv.astype(np.float32), 'Facing':
            np.clip(M.dot(N, V), 0.0, 1.0).astype(np.float32)}


DISPATCH = {
    # input
    'ShaderNodeTexCoord': n_tex_coord,
    'ShaderNodeUVMap': n_uvmap,
    'ShaderNodeWireframe': n_wireframe,
    'ShaderNodeVectorTransform': n_vector_transform,
    'ShaderNodeAmbientOcclusion': n_ambient_occlusion,
    'ShaderNodeBevel': _n_named(
        'needs closest-geometry queries the BVH does not answer yet; '
        'the normal passes through unbent', _n_bevel_out),
    'ShaderNodeLightFalloff': _n_named(
        'falloff lives on the LAMPS in this renderer -- each light\'s own '
        'Falloff setting, where the era put it; Strength passes through',
        _n_falloff_out),
    'ShaderNodeVolumeAbsorption': _n_named(_NO_VOLUME, _n_empty_volume),
    'ShaderNodeVolumeScatter': _n_named(_NO_VOLUME, _n_empty_volume),
    'ShaderNodeVolumePrincipled': _n_named(_NO_VOLUME, _n_empty_volume),
    'ShaderNodeVolumeCoefficients': _n_named(_NO_VOLUME, _n_empty_volume),
    'ShaderNodeVolumeInfo': _n_named(_NO_VOLUME),
    'ShaderNodeTexPointDensity': _n_named(_NO_VOLUME),
    'ShaderNodeParticleInfo': _n_named(
        'particles are not exported; instance them as real geometry'),
    'ShaderNodePointInfo': _n_named(
        'point clouds are not exported; instance them as real geometry'),
    'ShaderNodeHairInfo': _n_named(
        'curves are not exported; convert them to a mesh'),
    'ShaderNodeBsdfRayPortal': _n_named(
        'a Cycles portal, with nothing here to portal to',
        lambda ev, node: {'BSDF': Closure()}),
    'ShaderNodeScript': _n_named(
        'OSL is not in this renderer -- the Coded Shader node (GLSL/HLSL) '
        'is the native equivalent'),
    'ShaderNodeOutputAOV': _n_named(
        'AOV outputs are not collected; the Debug panel\'s Render Pass '
        'menu is the equivalent', lambda ev, node: {}),
    'ShaderNodeNewGeometry': n_geometry,
    'ShaderNodeObjectInfo': n_object_info,
    'ShaderNodeCameraData': n_camera_data,
    'ShaderNodeAttribute': n_attribute,
    'ShaderNodeVertexColor': n_vertex_color,
    'ShaderNodeFresnel': n_fresnel,
    'ShaderNodeLayerWeight': n_layer_weight,
    'ShaderNodeLightPath': n_light_path,
    'ShaderNodeRGB': n_rgb,
    'ShaderNodeValue': n_value,
    'ShaderNodeTangent': n_tangent,
    # texture
    'ShaderNodeTexImage': n_tex_image,
    'ShaderNodeTexEnvironment': n_tex_environment,
    'ShaderNodeTexChecker': n_tex_checker,
    'ShaderNodeTexGradient': n_tex_gradient,
    'ShaderNodeTexNoise': n_tex_noise,
    'ShaderNodeTexWhiteNoise': n_tex_white_noise,
    'ShaderNodeTexWave': n_tex_wave,
    'ShaderNodeTexVoronoi': n_tex_voronoi,
    'ShaderNodeTexMagic': n_tex_magic,
    'ShaderNodeTexBrick': n_tex_brick,
    'ShaderNodeTexMusgrave': n_tex_musgrave,
    'ShaderNodeTexIES': n_tex_ies,
    # colour
    'ShaderNodeMixRGB': n_mix_rgb,
    'ShaderNodeMix': n_mix,
    'ShaderNodeInvert': n_invert,
    'ShaderNodeHueSaturation': n_hue_sat,
    'ShaderNodeBrightContrast': n_bright_contrast,
    'ShaderNodeGamma': n_gamma,
    'ShaderNodeRGBCurve': n_rgb_curve,
    'ShaderNodeFloatCurve': n_float_curve,
    'ShaderNodeVectorCurve': n_vector_curve,
    'ShaderNodeValToRGB': n_val_to_rgb,
    'ShaderNodeRGBToBW': n_rgb_to_bw,
    'ShaderNodeBlackbody': n_blackbody,
    'ShaderNodeWavelength': n_wavelength,
    # vector
    'ShaderNodeMapping': n_mapping,
    'ShaderNodeVectorMath': n_vector_math,
    'ShaderNodeVectorRotate': n_vector_rotate,
    'ShaderNodeNormal': n_normal,
    'ShaderNodeBump': n_bump,
    'ShaderNodeNormalMap': n_normal_map,
    'ShaderNodeDisplacement': n_displacement,
    'ShaderNodeVectorDisplacement': n_displacement,
    # converter
    'ShaderNodeMath': n_math,
    'ShaderNodeClamp': n_clamp,
    'ShaderNodeMapRange': n_map_range,
    'ShaderNodeSeparateXYZ': n_separate_xyz,
    'ShaderNodeCombineXYZ': n_combine_xyz,
    'ShaderNodeSeparateColor': n_separate_color,
    'ShaderNodeCombineColor': n_combine_color,
    'ShaderNodeSeparateRGB': n_separate_rgb,
    'ShaderNodeCombineRGB': n_combine_rgb,
    'ShaderNodeSeparateHSV': n_separate_hsv,
    'ShaderNodeCombineHSV': n_combine_hsv,
    'ShaderNodeShaderToRGB': n_shader_to_rgb,
    # shaders
    'ShaderNodeBsdfDiffuse': n_bsdf_diffuse,
    'ShaderNodeBsdfGlossy': n_bsdf_glossy,
    'ShaderNodeBsdfMetallic': n_bsdf_metallic,
    'ShaderNodeEeveeSpecular': n_bsdf_specular,
    'ShaderNodeBsdfAnisotropic': n_bsdf_anisotropic,
    'ShaderNodeBsdfGlass': n_bsdf_glass,
    'ShaderNodeBsdfRefraction': n_bsdf_refraction,
    'ShaderNodeBsdfTransparent': n_bsdf_transparent,
    'ShaderNodeBsdfTranslucent': n_bsdf_translucent,
    'ShaderNodeBsdfToon': n_bsdf_toon,
    'ShaderNodeBsdfVelvet': n_bsdf_velvet,
    'ShaderNodeBsdfSheen': n_bsdf_velvet,
    'ShaderNodeBsdfHair': n_bsdf_diffuse,
    'ShaderNodeBsdfHairPrincipled': n_bsdf_diffuse,
    'ShaderNodeSubsurfaceScattering': n_subsurface,
    'ShaderNodeEmission': n_emission,
    'ShaderNodeBackground': n_background,
    'ShaderNodeHoldout': n_holdout,
    'ShaderNodeBsdfPrincipled': n_principled,
    'ShaderNodeMixShader': n_mix_shader,
    'ShaderNodeAddShader': n_add_shader,
    'ShaderNodeOutputMaterial': n_output_material,
    'ShaderNodeOutputWorld': n_output_material,
    'ShaderNodeOutputLight': n_output_material,
    # structure
    'NodeReroute': n_reroute,
    'NodeMuted': lambda ev, node: ev.fallback(node),
    'ShaderNodeGroup': n_group,
    'NodeGroupInput': n_group_input,
    'NodeGroupOutput': n_group_output,
    # Halcyon
    'HALCYON_ShaderNode': n_halcyon_shader,
    'HALCYON_CodeNode': n_halcyon_code,
    'HALCYON_PosterizeNode': n_halcyon_posterize,
    'HALCYON_DitherNode': n_halcyon_dither,
    'HALCYON_DepthCueNode': n_halcyon_depth_cue,
    'HALCYON_ScreenInfoNode': n_halcyon_screen_info,
    'HALCYON_PixelateNode': n_halcyon_pixelate,
    'HALCYON_ScrollNode': n_halcyon_scroll,
    'HALCYON_ScanlinesNode': n_halcyon_scanlines,
    'HALCYON_PaletteNode': n_halcyon_palette,
    'HALCYON_ColorCycleNode': n_halcyon_color_cycle,
    'HALCYON_FlipbookNode': n_halcyon_flipbook,
    'HALCYON_UVWaveNode': n_halcyon_uv_wave,
    'HALCYON_HalftoneNode': n_halcyon_halftone,
    'HALCYON_ThresholdNode': n_halcyon_threshold,
    'HALCYON_QuantizeNode': n_halcyon_quantize,
    'ShaderNodeTexSky': n_tex_sky,
    'ShaderNodeTexGabor': n_tex_gabor,
    'HALCYON_MarbleNode': n_pat_marble,
    'HALCYON_WoodNode': n_pat_wood,
    'HALCYON_GraniteNode': n_pat_granite,
    'HALCYON_DentsNode': n_pat_dents,
    'HALCYON_CrackleNode': n_pat_crackle,
    'HALCYON_PlasmaNode': n_pat_plasma,
    'HALCYON_RipplesNode': n_pat_ripples,
    'HALCYON_StarfieldNode': n_pat_starfield,
    'HALCYON_WeaveNode': n_pat_weave,
    'HALCYON_ScratchesNode': n_pat_scratches,
    'HALCYON_TilesNode': n_pat_tiles,
    'HALCYON_SpiralNode': n_pat_spiral,
    'HALCYON_BozoNode': n_pat_bozo,
    'HALCYON_AgateNode': n_pat_agate,
    'HALCYON_LeopardNode': n_pat_leopard,
    'HALCYON_OnionNode': n_pat_onion,
    'HALCYON_BumpsNode': n_pat_bumps,
    'HALCYON_WrinklesNode': n_pat_wrinkles,
    'HALCYON_BrickNode': n_pat_brick,
    'HALCYON_NoiseNode': n_pat_noise,
    'HALCYON_WaterNode': n_pat_water,
    'HALCYON_GradientNode': n_pat_gradient_shaped,
    'HALCYON_CellsNode': n_pat_cells_tex,
    'HALCYON_StaticNode': n_pat_static,
    'HALCYON_MatcapUVNode': n_halcyon_matcap_uv,
    'HALCYON_RampNode': n_halcyon_ramp,
    'HALCYON_BlurNode': n_halcyon_blur,
}
