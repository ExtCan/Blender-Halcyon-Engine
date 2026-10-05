"""R251 C119: REYES micropolygon shading rate, constant interpolation
(Pixar REYES 1987; PRMan 3.x-11 RiShadingRate / RiShadingInterpolation
"constant"; BMRT).

REYES dices each primitive into a grid of micropolygons in the surface's
own (u, v), sized so each covers about ShadingRate pixels of screen
AREA, shades the grid once per cell and -- under "constant"
interpolation -- gives every pixel inside a cell one shaded value:
facets that follow the surface's parametrisation, not the screen, and
hold their size as the object approaches. Halcyon dices each TRIANGLE's
barycentric plane into n x n cells (a reconstruction on triangles, not
on the primitive's own parametrisation -- disclosed in the CHANGELOG)
and snaps every fragment's barycentrics to its cell's (min u, min v)
corner, the REYES grid vertex, BEFORE the shade. `shading_rate_area`
is an area in OUTPUT pixels (RISpec 3.2 s4.2.7), so a supersampled
frame scales the area by ss * ss and shades once per output-pixel-area
cell at any AA.

Both roads snap with the same arithmetic: the CPU here (float32, one op
per statement), the GPU in `snap_glsl` (multiply, floor, integer clamp,
multiply by the CPU's OWN float32 reciprocal fetched from the `hal_reyes`
data texture -- never a divide on the GPU). The snap is therefore
bitwise; the shaded frame sits on the deferred bar as every PIXEL-rate
frame does. Rate 0 (the default) builds no grid and calls no snap on
either road: identity by construction.
"""
import math

import numpy as np


class Grid:
    """The per-triangle (n, inv_n) pair: `n` int32 cells per barycentric
    axis, `inv_n` = float32(1) / float32(n) computed ONCE on the CPU."""

    __slots__ = ('n', 'inv_n', 'rate')

    def __init__(self, n, inv_n, rate):
        self.n = n
        self.inv_n = inv_n
        self.rate = float(rate)

    @property
    def side(self):
        """The padded-square side of the `hal_reyes` data texture."""
        return int(math.ceil(math.sqrt(float(max(int(self.n.shape[0]), 1)))))


def micro_grid(mesh, vp, rw, rh, rate):
    """(T,) cells per barycentric axis and their float32 reciprocals.

    The three vertices are projected with the frame's OWN `vp` (the
    matrix the raster used, `h = vp @ (x, y, z, 1)`, `w = h[3]`) into
    the internal frame's pixels `sx = (h0/w*0.5 + 0.5)*rw`, `sy =
    (h1/w*0.5 + 0.5)*rh` in float64; the screen area `A = 0.5 * |(x1 -
    x0)(y2 - y0) - (x2 - x0)(y1 - y0)|` gives `n = max(1, min(4096,
    ceil(sqrt(A / rate))))`. A triangle with any corner at `w <= 0`
    (behind the eye) takes n = 1. `rate` is ALREADY scaled by ss * ss
    by the caller (an area in output pixels; the internal frame is
    `rw = W * ss`).
    """
    rate = float(rate)
    tris = np.asarray(mesh.tris, np.int64)
    T = int(tris.shape[0]) if tris.ndim == 2 else 0
    n = np.ones(max(T, 0), np.int32)
    if T > 0 and rate > 0.0:
        v = np.asarray(mesh.verts, np.float64)
        m = np.asarray(vp, np.float64)
        h = v @ m[:, :3].T + m[:, 3][None, :]           # (V, 4) = vp @ (x, y, z, 1)
        w = h[:, 3]
        ok = w > 0.0
        ws = np.where(ok, w, 1.0)
        sx = (h[:, 0] / ws * 0.5 + 0.5) * float(rw)
        sy = (h[:, 1] / ws * 0.5 + 0.5) * float(rh)
        x0, y0 = sx[tris[:, 0]], sy[tris[:, 0]]
        x1, y1 = sx[tris[:, 1]], sy[tris[:, 1]]
        x2, y2 = sx[tris[:, 2]], sy[tris[:, 2]]
        area = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
        cells = np.ceil(np.sqrt(area / rate))
        cells = np.where(np.isfinite(cells), cells, 1.0)
        n = np.clip(cells, 1.0, 4096.0).astype(np.int32)
        behind = ~(ok[tris[:, 0]] & ok[tris[:, 1]] & ok[tris[:, 2]])
        n = np.where(behind, np.int32(1), n).astype(np.int32)
    inv_n = (np.float32(1.0) / n.astype(np.float32)).astype(np.float32)
    return Grid(n, inv_n, rate)


def snap(bary, tri, grid):
    """Snap (F, 3) float32 barycentrics to their cell's min corner.

    Six statements, float32, in this order on BOTH roads:
        m1 = b1 * n; k1 = floor(m1); m2 = b2 * n; k2 = floor(m2)
        k2 = (k1 + k2 > n - 1) ? (n - 1) - k1 : k2     (the diagonal clamp)
        g1 = k1 * inv; g2 = k2 * inv; g0 = 1 - g1; g0 = g0 - g2
    A barycentric a hair below zero (a shared edge's rounding) would
    floor to k = -1: k1 and k2 are clamped at 0 first, both roads.
    """
    b = np.asarray(bary, np.float32)
    t = np.asarray(tri, np.int64)
    n = grid.n[t].astype(np.float32)
    inv = grid.inv_n[t].astype(np.float32)
    m1 = b[:, 1] * n
    k1 = np.floor(m1)
    m2 = b[:, 2] * n
    k2 = np.floor(m2)
    k1 = np.maximum(k1, np.float32(0.0))
    k2 = np.maximum(k2, np.float32(0.0))
    top = n - np.float32(1.0)
    ks = k1 + k2
    k2 = np.where(ks > top, top - k1, k2).astype(np.float32)
    g1 = (k1 * inv).astype(np.float32)
    g2 = (k2 * inv).astype(np.float32)
    g0 = (np.float32(1.0) - g1).astype(np.float32)
    g0 = (g0 - g2).astype(np.float32)
    out = np.empty_like(b)
    out[:, 0] = g0
    out[:, 1] = g1
    out[:, 2] = g2
    return out


def rate_of(st):
    return float(getattr(st, 'shading_rate_area', 0.0) or 0.0)


def grid_for(job):
    """The frame's grid, built ONCE per job (None at rate 0).

    `render()` stamps `job.vp` and `job.ss` (R250); a job built by a
    test rig without them takes the camera's own matrices at the job's
    size and ss 1 -- the same numbers `render()` would use without AA
    jitter.
    """
    rate = rate_of(getattr(job, 'settings', None))
    if rate <= 0.0:
        return None
    g = getattr(job, 'reyes_grid', None)
    if g is not None:
        return g
    mesh = getattr(job.scene, 'mesh', None)
    if mesh is None or getattr(mesh, 'tris', None) is None \
            or np.asarray(mesh.tris).size == 0:
        return None
    vp = getattr(job, 'vp', None)
    if vp is None:
        from .render import camera_matrices
        vp = camera_matrices(job.scene.camera, job.width, job.height)[2]
    ss = int(getattr(job, 'ss', 1) or 1)
    g = micro_grid(mesh, vp, job.width, job.height, rate * ss * ss)
    try:
        job.reyes_grid = g
    except AttributeError:
        pass
    return g


def consts_for(st, scene):
    """`consts['reyes']`: `{'side': side}` when the rate is on (the snap
    code is emitted only then -- a gate AND a bake, both in st_sig
    through `shading_rate_area`), else None."""
    if rate_of(st) <= 0.0:
        return None
    mesh = getattr(scene, 'mesh', None)
    tris = getattr(mesh, 'tris', None) if mesh is not None else None
    T = int(np.asarray(tris).shape[0]) if tris is not None and np.asarray(tris).ndim == 2 else 0
    if T <= 0:
        return None
    return {'side': int(math.ceil(math.sqrt(float(max(T, 1)))))}


def image(job):
    """The `hal_reyes` data texture: (side, side, 4) float32, R = n,
    G = inv_n per triangle, padded square, `texelFetch` only."""
    g = grid_for(job)
    side = g.side
    img = np.zeros((side * side, 4), np.float32)
    T = int(g.n.shape[0])
    img[:T, 0] = g.n.astype(np.float32)
    img[:T, 1] = g.inv_n
    return img.reshape(side, side, 4)


def snap_glsl(side):
    """The GLSL twin of `snap`, reading (n, inv_n) from `hal_reyes`.

    `row = floor((f.tri + 0.5) / side)`: `k*side / side` may land at
    `k - 1 ULP` under GLSL's 2.5-ULP division bound and floor to the
    previous row; the +0.5 margin keeps every quotient at least 0.5/side
    from an integer. The column is the subtraction form, never mod().
    One op per statement where a floor follows, so FMA contraction
    cannot cross a cell boundary.
    """
    s = f'{float(side):.9g}'
    if '.' not in s and 'e' not in s:
        s += '.0'
    return f"""
// R251 C119 (MAT-B): the REYES micropolygon snap -- (n, inv_n) per
// triangle from hal_reyes, the CPU's own float32 numbers
uniform sampler2D hal_reyes;
HalcyonFragment hal_reyes_snap(HalcyonFragment f)
{{
    if (!f.covered) return f;
    float row = floor((f.tri + 0.5) / {s});
    float colx = f.tri - {s} * row;
    vec2 g = texelFetch(hal_reyes, ivec2(int(colx), int(row)), 0).rg;
    float n = g.x;
    float inv = g.y;
    float m1 = f.bary.y * n;
    float k1 = floor(m1);
    float m2 = f.bary.z * n;
    float k2 = floor(m2);
    k1 = max(k1, 0.0);
    k2 = max(k2, 0.0);
    float top = n - 1.0;
    float ks = k1 + k2;
    k2 = (ks > top) ? (top - k1) : k2;
    float g1 = k1 * inv;
    float g2 = k2 * inv;
    float g0 = 1.0 - g1;
    g0 = g0 - g2;
    f.bary = vec3(g0, g1, g2);
    f.P = hal_interp(f.tri, f.bary, 0);
    f.N = normalize(hal_interp(f.tri, f.bary, 1));
    f.uv = hal_interp(f.tri, f.bary, 2).xy;
    f.uv2 = hal_interp4(f.tri, f.bary, 2).zw;
    return f;
}}
"""


#: the printed refusals (the GPU names them; the CPU road runs)
REFUSE_AFFINE = ('REYES shading rate under affine texture mapping: the uv '
                 'rides the screen-linear barycentrics (hal_gb_idslin), a '
                 'second grid the snap does not cover; shades on the CPU')
REFUSE_PASS = ("REYES shading rate snaps the camera G-buffer's "
               'barycentrics; a hit/layer pass has none -- shades on the CPU')
REFUSE_BUMP = ('REYES shading rate with a Bump height pre-pass: the GPU '
               'pre-pass samples the height at the pixel, the CPU at the '
               'cell corner; shades on the CPU')


def shading_rate_snap(bary, tri, grid):
    """The public name of the snap (the build marker's `def`): `snap`."""
    return snap(bary, tri, grid)
