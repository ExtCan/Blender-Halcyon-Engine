"""Software scanline / z-buffer rasteriser.

This is a real rasteriser, not a wrapper around anything: clip-space transform,
near-plane Sutherland-Hodgman clipping, integer-snappable screen coordinates,
top-left fill rule, perspective-correct barycentrics, and an A-buffer fragment
capture path for sorted transparency (Carpenter 1984 -- period correct).

The G-buffer stores, per pixel, the *original* triangle id plus barycentric
weights over that triangle's three corners. Every vertex attribute (normal, uv,
colour, tangent...) is reconstructed later from those weights, so the rasteriser
never has to know what attributes exist.
"""

import numpy as np

from . import mathx as M

EMPTY = -1


class GBuffer:
    __slots__ = ('width', 'height', 'tri', 'bary', 'bary_lin', 'depth', 'zndc',
                 'overdraw', 'front')

    def __init__(self, width, height):
        self.width = int(width)
        self.height = int(height)
        self.tri = np.full((height, width), EMPTY, dtype=np.int32)
        self.bary = np.zeros((height, width, 3), dtype=np.float32)
        self.depth = np.full((height, width), np.inf, dtype=np.float32)
        self.zndc = np.full((height, width), 1.0, dtype=np.float32)
        self.front = np.ones((height, width), dtype=bool)
        self.overdraw = np.zeros((height, width), dtype=np.int32)
        self.bary_lin = None

    def alloc_linear(self):
        """Screen-linear barycentrics -- the affine texture warp of the era."""
        if self.bary_lin is None:
            self.bary_lin = np.zeros((self.height, self.width, 3), dtype=np.float32)
        return self.bary_lin

    def mask(self):
        return self.tri >= EMPTY + 1


class FragmentList:
    """Unbounded A-buffer: one entry per transparent fragment."""

    def __init__(self):
        self.px = []
        self.py = []
        self.tri = []
        self.depth = []
        self.bary = []
        self.front = []

    def add(self, px, py, tri, depth, bary, front, bary_lin=None):
        if px.size == 0:
            return
        self.px.append(px.astype(np.int32))
        self.py.append(py.astype(np.int32))
        self.tri.append(np.full(px.size, tri, dtype=np.int32) if np.isscalar(tri) else tri.astype(np.int32))
        self.depth.append(depth.astype(np.float32))
        self.bary.append(bary.astype(np.float32))
        self.front.append(front.astype(bool) if not np.isscalar(front)
                          else np.full(px.size, bool(front)))

    def finish(self):
        if not self.px:
            z = np.zeros(0, np.int32)
            return (z, z, z, np.zeros(0, np.float32), np.zeros((0, 3), np.float32),
                    np.zeros(0, bool))
        return (np.concatenate(self.px), np.concatenate(self.py),
                np.concatenate(self.tri), np.concatenate(self.depth),
                np.concatenate(self.bary), np.concatenate(self.front))

    def __len__(self):
        return int(sum(a.size for a in self.px))


# ---------------------------------------------------------------- transform


def project(verts, mvp, width, height, snap=0.0, near_eps=1e-5):
    """World verts -> clip space, plus screen coords for the non-clipped case.

    Returns (clip (V,4), screen (V,2), invw (V,), zndc (V,)).
    Vertices behind the eye keep valid clip coords; the clipper deals with them.
    """
    n = verts.shape[0]
    ph = np.empty((n, 4), dtype=np.float32)
    ph[:, :3] = verts
    ph[:, 3] = 1.0
    clip = ph @ np.asarray(mvp, dtype=np.float32).T
    w = clip[:, 3]
    safe_w = np.where(np.abs(w) < near_eps, near_eps, w)
    invw = (1.0 / safe_w).astype(np.float32)
    ndc = clip[:, :3] * invw[:, None]
    sx = (ndc[:, 0] * 0.5 + 0.5) * width
    sy = (ndc[:, 1] * 0.5 + 0.5) * height
    screen = np.stack([sx, sy], axis=1).astype(np.float32)
    if snap > 0.0:
        screen = np.round(screen / snap) * snap
    return clip, screen, invw, ndc[:, 2].astype(np.float32)


def _clip_near(cp, cb, near_eps):
    """Sutherland-Hodgman against z >= -w for one polygon.

    cp: (K,4) clip positions, cb: (K,3) barycentric-over-original weights.
    """
    out_p = []
    out_b = []
    k = len(cp)
    for i in range(k):
        a_p, a_b = cp[i], cb[i]
        b_p, b_b = cp[(i + 1) % k], cb[(i + 1) % k]
        fa = a_p[2] + a_p[3]
        fb = b_p[2] + b_p[3]
        ina = fa >= near_eps
        inb = fb >= near_eps
        if ina:
            out_p.append(a_p)
            out_b.append(a_b)
        if ina != inb:
            denom = fa - fb
            if abs(denom) < 1e-12:
                continue
            t = fa / denom
            out_p.append(a_p + (b_p - a_p) * t)
            out_b.append(a_b + (b_b - a_b) * t)
    return out_p, out_b


IDENT_BARY = np.eye(3, dtype=np.float32)


def build_screen_tris(clip, tris, width, height, snap=0.0, near_eps=1e-5,
                      depth_bits=24, clip_far=False):
    """Clip + project every triangle. Returns flat arrays ready for filling.

    Output arrays are per *emitted* triangle (clipping can create more than one):
      sx,sy  (E,3) screen coords
      iw     (E,3) 1/w
      z      (E,3) ndc z
      bw     (E,3,3) barycentric weight of each emitted corner over the original
      src    (E,)  index into `tris`
    """
    tris = np.asarray(tris, dtype=np.int32)
    cp = clip[tris]                                    # (T,3,4)
    f = cp[:, :, 2] + cp[:, :, 3]                      # near plane function
    inside = f >= near_eps
    n_in = inside.sum(axis=1)
    all_in = n_in == 3
    straddle = (n_in > 0) & ~all_in

    idx_in = np.nonzero(all_in)[0]
    parts_p = [cp[idx_in]]
    parts_b = [np.broadcast_to(IDENT_BARY, (idx_in.size, 3, 3)).copy()]
    parts_src = [idx_in.astype(np.int32)]

    for t in np.nonzero(straddle)[0]:
        poly_p, poly_b = _clip_near(list(cp[t]), list(IDENT_BARY), near_eps)
        if len(poly_p) < 3:
            continue
        for j in range(1, len(poly_p) - 1):
            parts_p.append(np.stack([poly_p[0], poly_p[j], poly_p[j + 1]])[None])
            parts_b.append(np.stack([poly_b[0], poly_b[j], poly_b[j + 1]])[None])
            parts_src.append(np.array([t], dtype=np.int32))

    if len(parts_p) == 1 and idx_in.size == 0:
        e = np.zeros((0, 3), np.float32)
        return (e, e.copy(), e.copy(), e.copy(),
                np.zeros((0, 3, 3), np.float32), np.zeros(0, np.int32))

    P = np.concatenate(parts_p, axis=0).astype(np.float32)   # (E,3,4)
    B = np.concatenate(parts_b, axis=0).astype(np.float32)   # (E,3,3)
    S = np.concatenate(parts_src, axis=0)

    w = P[:, :, 3]
    w = np.where(np.abs(w) < near_eps, near_eps, w)
    iw = (1.0 / w).astype(np.float32)
    ndc = P[:, :, :3] * iw[:, :, None]
    sx = ((ndc[:, :, 0] * 0.5 + 0.5) * width).astype(np.float32)
    sy = ((ndc[:, :, 1] * 0.5 + 0.5) * height).astype(np.float32)
    z = ndc[:, :, 2].astype(np.float32)
    if snap > 0.0:
        sx = np.round(sx / snap) * snap
        sy = np.round(sy / snap) * snap
    if depth_bits < 32:
        steps = float((1 << int(max(2, depth_bits))) - 1)
        z = np.round((z * 0.5 + 0.5) * steps) / steps * 2.0 - 1.0
    return sx, sy, iw, z, B, S


# ------------------------------------------------------------------- filling


def fill(gbuf, sx, sy, iw, z, bw, src, cull='NONE', frags=None, flat_depth=None,
         depth_write=True, depth_test=True, count_overdraw=False,
         tri_offset=0, z_offset=0.0, tri_map=None):
    """Fill emitted triangles into a GBuffer (and/or a FragmentList).

    cull: 'NONE' | 'BACK' | 'FRONT'
    frags: FragmentList to append to instead of / as well as writing the gbuf.
    """
    W, H = gbuf.width, gbuf.height
    n = sx.shape[0]
    if n == 0:
        return 0
    written = 0

    x0, x1, x2 = sx[:, 0], sx[:, 1], sx[:, 2]
    y0, y1, y2 = sy[:, 0], sy[:, 1], sy[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    live = np.abs(area) > 1e-9
    if cull == 'BACK':
        live &= area > 0.0          # CCW front faces, y-up screen. See below.
    elif cull == 'FRONT':
        live &= area < 0.0

    bxmin = np.maximum(np.floor(np.minimum(np.minimum(x0, x1), x2)).astype(np.int32), 0)
    bxmax = np.minimum(np.ceil(np.maximum(np.maximum(x0, x1), x2)).astype(np.int32), W - 1)
    bymin = np.maximum(np.floor(np.minimum(np.minimum(y0, y1), y2)).astype(np.int32), 0)
    bymax = np.minimum(np.ceil(np.maximum(np.maximum(y0, y1), y2)).astype(np.int32), H - 1)
    live &= (bxmax >= bxmin) & (bymax >= bymin)

    order = np.nonzero(live)[0]
    depth = gbuf.depth
    tri_b = gbuf.tri
    bary_b = gbuf.bary
    zndc_b = gbuf.zndc
    front_b = gbuf.front

    for t in order:
        xa, xb, xc = x0[t], x1[t], x2[t]
        ya, yb, yc = y0[t], y1[t], y2[t]
        ar = area[t]
        inv_area = 1.0 / ar
        xs = np.arange(bxmin[t], bxmax[t] + 1, dtype=np.float32) + 0.5
        ys = np.arange(bymin[t], bymax[t] + 1, dtype=np.float32) + 0.5
        X = xs[None, :]
        Y = ys[:, None]

        e0 = (xc - xb) * (Y - yb) - (yc - yb) * (X - xb)
        e1 = (xa - xc) * (Y - yc) - (ya - yc) * (X - xc)
        e2 = (xb - xa) * (Y - ya) - (yb - ya) * (X - xa)
        if ar > 0:
            inside = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
        else:
            inside = (e0 <= 0) & (e1 <= 0) & (e2 <= 0)
        if not inside.any():
            continue

        yy, xx = np.nonzero(inside)
        l0 = e0[yy, xx] * inv_area
        l1 = e1[yy, xx] * inv_area
        l2 = e2[yy, xx] * inv_area

        src_tri = int(src[t]) + tri_offset
        if tri_map is not None:
            src_tri = int(tri_map[int(src[t])])
        if flat_depth is not None:
            # Painter's algorithm: the whole polygon carries one depth, so the
            # depth test decides between polygons rather than between fragments
            zz = np.full(l0.shape, float(flat_depth[src_tri]) + z_offset,
                         np.float32)
        else:
            zz = (l0 * z[t, 0] + l1 * z[t, 1] + l2 * z[t, 2]) + z_offset
        px = xx + bxmin[t]
        py = yy + bymin[t]

        if depth_test:
            keep = zz < depth[py, px]
            if not keep.any():
                continue
            if not keep.all():
                px, py, l0, l1, l2, zz = px[keep], py[keep], l0[keep], l1[keep], l2[keep], zz[keep]

        iw0, iw1, iw2 = iw[t, 0], iw[t, 1], iw[t, 2]
        invw = l0 * iw0 + l1 * iw1 + l2 * iw2
        invw = np.where(np.abs(invw) < 1e-20, 1e-20, invw)
        p0 = l0 * iw0 / invw
        p1 = l1 * iw1 / invw
        p2 = l2 * iw2 / invw
        b = (p0[:, None] * bw[t, 0][None, :] +
             p1[:, None] * bw[t, 1][None, :] +
             p2[:, None] * bw[t, 2][None, :])

        src_tri = int(src[t]) + tri_offset
        if tri_map is not None:
            src_tri = int(tri_map[int(src[t])])
        is_front = ar < 0.0

        if count_overdraw:
            np.add.at(gbuf.overdraw, (py, px), 1)

        if frags is not None:
            frags.add(px, py, src_tri, zz, b, is_front)
        if depth_write:
            depth[py, px] = zz
            tri_b[py, px] = src_tri
            bary_b[py, px] = b
            if gbuf.bary_lin is not None:
                lb = (l0[:, None] * bw[t, 0][None, :] +
                      l1[:, None] * bw[t, 1][None, :] +
                      l2[:, None] * bw[t, 2][None, :])
                gbuf.bary_lin[py, px] = lb
            zndc_b[py, px] = zz
            front_b[py, px] = is_front
        written += px.size
    return written


BATCH_MIN_TRIS = 24          # below this the loop wins; setup dominates


def rasterize(verts, tris, mvp, width, height, cull='NONE', snap=0.0,
              depth_bits=24, subset=None, gbuf=None, frags=None,
              depth_write=True, depth_test=True, count_overdraw=False,
              z_offset=0.0, near_eps=1e-5, batched=None, flat_depth=None,
              scissor=None):
    """Convenience: project + clip + fill in one call.

    `batched` selects the loop-free rasteriser; None picks automatically. The
    reference per-triangle path is kept because it is the simpler code and the
    batched one is validated against it in the test suite.

    `scissor` is a (y0, y1) row range. Triangles that fall entirely outside it
    are dropped before filling. That is what makes splitting a frame across
    processes worth doing: without it every worker rasterises the whole mesh
    for its own slice, and sixty slices means sixty rasterisations.
    """
    if gbuf is None:
        gbuf = GBuffer(width, height)
    tri_map = None
    if subset is not None:
        tri_map = np.asarray(subset, dtype=np.int32)
        tris = tris[tri_map]
    clip, _, _, _ = project(verts, mvp, width, height, snap=0.0, near_eps=near_eps)

    if scissor is not None and tris.shape[0]:
        # Drop triangles outside the band *before* clipping, not after. Clipping
        # every triangle in every band is the cost that made splitting a frame
        # across processes lose to not splitting it. Only triangles wholly in
        # front of the near plane can be judged this cheaply; any that straddle
        # it are kept and sorted out by the clipper as usual.
        y0s, y1s = scissor
        w = clip[:, 3]
        tw = w[tris]
        infront = (tw > near_eps).all(axis=1)
        if infront.any():
            ndc_y = clip[:, 1] / np.where(np.abs(w) < near_eps, near_eps, w)
            sy_all = (ndc_y * 0.5 + 0.5) * height
            ty = sy_all[tris]
            lo = ty.min(axis=1)
            hi = ty.max(axis=1)
            outside = infront & ((hi < y0s - 1.0) | (lo > y1s + 1.0))
            if outside.any():
                keep_tris = ~outside
                tris = tris[keep_tris]
                if tri_map is not None:
                    tri_map = tri_map[keep_tris]
                elif subset is None:
                    tri_map = np.nonzero(keep_tris)[0].astype(np.int32)

    sx, sy, iw, z, bw, src = build_screen_tris(clip, tris, width, height, snap=snap,
                                               near_eps=near_eps, depth_bits=depth_bits)
    if scissor is not None and sx.shape[0]:
        y0, y1 = scissor
        lo = sy.min(axis=1)
        hi = sy.max(axis=1)
        keep = (hi >= y0) & (lo < y1)
        if not keep.all():
            sx, sy, iw, z, bw, src = (a[keep] for a in (sx, sy, iw, z, bw, src))
    if batched is None:
        # overdraw counting needs the sequential semantics to stay exact
        batched = (sx.shape[0] >= BATCH_MIN_TRIS) and not count_overdraw
    if batched:
        fill_batched(gbuf, sx, sy, iw, z, bw, src, cull=cull, frags=frags,
                     flat_depth=flat_depth, depth_write=depth_write,
                     depth_test=depth_test, z_offset=z_offset, tri_map=tri_map)
    else:
        fill(gbuf, sx, sy, iw, z, bw, src, cull=cull, frags=frags,
             flat_depth=flat_depth, depth_write=depth_write,
             depth_test=depth_test, count_overdraw=count_overdraw,
             z_offset=z_offset, tri_map=tri_map)
    return gbuf


# ------------------------------------------------------- attribute fetching


def fetch(attr, tris, tri_idx, bary):
    """Interpolate a per-vertex attribute at shaded fragments.

    attr: (V,C) or (V,)   tris: (T,3)   tri_idx: (N,)   bary: (N,3)
    """
    idx = tris[tri_idx]                       # (N,3)
    a = attr[idx]                             # (N,3[,C])
    if a.ndim == 2:
        return (a * bary).sum(axis=1)
    return (a * bary[:, :, None]).sum(axis=1)


def fetch_face(attr, tri_idx):
    return attr[tri_idx]


def screen_derivatives(image, valid, tri_id):
    """Finite-difference derivatives that respect triangle boundaries.

    image: (H,W,C) attribute laid out in screen space.
    Returns (ddx, ddy) with the same shape.
    """
    ddx = np.zeros_like(image)
    ddy = np.zeros_like(image)
    same_x = np.zeros(tri_id.shape, dtype=bool)
    same_x[:, :-1] = (tri_id[:, :-1] == tri_id[:, 1:]) & valid[:, :-1] & valid[:, 1:]
    same_y = np.zeros(tri_id.shape, dtype=bool)
    same_y[:-1, :] = (tri_id[:-1, :] == tri_id[1:, :]) & valid[:-1, :] & valid[1:, :]

    fwd_x = np.zeros_like(image)
    fwd_x[:, :-1] = image[:, 1:] - image[:, :-1]
    fwd_y = np.zeros_like(image)
    fwd_y[:-1, :] = image[1:, :] - image[:-1, :]

    ddx[same_x] = fwd_x[same_x]
    ddy[same_y] = fwd_y[same_y]
    # backward difference where the forward neighbour was a different triangle
    bx = ~same_x & valid
    bx[:, 1:] &= (tri_id[:, 1:] == tri_id[:, :-1]) & valid[:, :-1]
    bx[:, 0] = False
    ddx[bx] = fwd_x[np.roll(bx, -1, axis=1)]
    by = ~same_y & valid
    by[1:, :] &= (tri_id[1:, :] == tri_id[:-1, :]) & valid[:-1, :]
    by[0, :] = False
    ddy[by] = fwd_y[np.roll(by, -1, axis=0)]
    return ddx, ddy


# ------------------------------------------------------- batched rasteriser

def _size_classes(span):
    """Round each triangle's bounding box up to a power of two."""
    span = np.maximum(span, 1)
    return (1 << np.ceil(np.log2(span.astype(np.float64))).astype(np.int64))


_GRID_CACHE = {}


def _grid(h, w):
    key = (h, w)
    g = _GRID_CACHE.get(key)
    if g is None:
        oy, ox = np.mgrid[0:h, 0:w]
        g = (oy[None, :, :].astype(np.int64), ox[None, :, :].astype(np.int64))
        if len(_GRID_CACHE) < 128:
            _GRID_CACHE[key] = g
    return g


LARGE_TRI_PX = 16384         # a 128x128 box; above this the loop amortises fine


def fill_batched(gbuf, sx, sy, iw, z, bw, src, cull='NONE', frags=None,
                 flat_depth=None, depth_write=True, depth_test=True, tri_offset=0,
                 z_offset=0.0, tri_map=None, max_batch_px=4_000_000):
    """Same result as fill(), without the per-triangle Python loop.

    Small triangles are bucketed by bounding-box size class -- separately in
    width and height, so a long thin triangle is not padded out to a square --
    and every candidate pixel in a bucket is tested in one vectorised sweep.
    Triangles with large boxes are handed to the sequential path instead, where
    the per-triangle overhead is amortised by the pixel count and the batched
    path's padding would be pure waste.

    The z-resolve sorts fragments per pixel and takes the nearest, which picks
    the same winner a sequential depth test would: a fragment only survives if
    it beats both the existing buffer and every other fragment on its pixel.

    Verified bit-identical to fill() by tests/test_render.py.
    """
    W, H = gbuf.width, gbuf.height
    n = sx.shape[0]
    if n == 0:
        return 0

    x0, x1, x2 = sx[:, 0], sx[:, 1], sx[:, 2]
    y0, y1, y2 = sy[:, 0], sy[:, 1], sy[:, 2]
    area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

    live = np.abs(area) > 1e-9
    if cull == 'BACK':
        live &= area > 0.0
    elif cull == 'FRONT':
        live &= area < 0.0

    bxmin = np.maximum(np.floor(np.minimum(np.minimum(x0, x1), x2)), 0).astype(np.int64)
    bxmax = np.minimum(np.ceil(np.maximum(np.maximum(x0, x1), x2)), W - 1).astype(np.int64)
    bymin = np.maximum(np.floor(np.minimum(np.minimum(y0, y1), y2)), 0).astype(np.int64)
    bymax = np.minimum(np.ceil(np.maximum(np.maximum(y0, y1), y2)), H - 1).astype(np.int64)
    live &= (bxmax >= bxmin) & (bymax >= bymin)

    bw_px = bxmax - bxmin + 1
    bh_px = bymax - bymin + 1

    big = live & ((bw_px * bh_px) > LARGE_TRI_PX)
    written = 0
    if big.any():
        sel = np.nonzero(big)[0]
        written += fill(gbuf, sx[sel], sy[sel], iw[sel], z[sel], bw[sel], src[sel],
                        cull=cull, frags=frags, flat_depth=flat_depth,
                        depth_write=depth_write,
                        depth_test=depth_test, tri_offset=tri_offset,
                        z_offset=z_offset, tri_map=tri_map)

    idx_all = np.nonzero(live & ~big)[0]
    if idx_all.size == 0:
        return written
    if idx_all.size < BATCH_MIN_TRIS:
        # too few small triangles to pay for bucketing -- this is the common
        # case at high supersampling, where everything is large in pixels
        return written + fill(
            gbuf, sx[idx_all], sy[idx_all], iw[idx_all], z[idx_all],
            bw[idx_all], src[idx_all], cull=cull, frags=frags,
            depth_write=depth_write, depth_test=depth_test,
            tri_offset=tri_offset, z_offset=z_offset, tri_map=tri_map)

    wc = _size_classes(bw_px[idx_all])
    hc = _size_classes(bh_px[idx_all])
    key = wc * 65536 + hc

    px_all, py_all, zz_all, b_all, blin_all = [], [], [], [], []
    tri_all, front_all = [], []

    for k in np.unique(key):
        members = idx_all[key == k]
        SW = int(k // 65536)
        SH = int(k % 65536)
        per = max(int(max_batch_px // max(SW * SH, 1)), 1)
        for start in range(0, members.size, per):
            t = members[start:start + per]
            oy, ox = _grid(SH, SW)
            bx = bxmin[t][:, None, None]
            by = bymin[t][:, None, None]
            X = (bx + ox).astype(np.float32) + 0.5
            Y = (by + oy).astype(np.float32) + 0.5

            xa, xb, xc = x0[t][:, None, None], x1[t][:, None, None], x2[t][:, None, None]
            ya, yb, yc = y0[t][:, None, None], y1[t][:, None, None], y2[t][:, None, None]
            e0 = (xc - xb) * (Y - yb) - (yc - yb) * (X - xb)
            e1 = (xa - xc) * (Y - yc) - (ya - yc) * (X - xc)
            e2 = (xb - xa) * (Y - ya) - (yb - ya) * (X - xa)
            pos = (area[t] > 0)[:, None, None]
            inside = np.where(pos, (e0 >= 0) & (e1 >= 0) & (e2 >= 0),
                              (e0 <= 0) & (e1 <= 0) & (e2 <= 0))
            inside &= (ox < bw_px[t][:, None, None]) & (oy < bh_px[t][:, None, None])
            if not inside.any():
                continue

            ti, yy, xx = np.nonzero(inside)
            inv_area = (1.0 / area[t])[ti]
            l0 = e0[ti, yy, xx] * inv_area
            l1 = e1[ti, yy, xx] * inv_area
            l2 = e2[ti, yy, xx] * inv_area
            px = (bxmin[t][ti] + xx).astype(np.int32)
            py = (bymin[t][ti] + yy).astype(np.int32)
            src_tri = src[t][ti].astype(np.int32) + tri_offset
            if tri_map is not None:
                src_tri = tri_map[src[t][ti]].astype(np.int32)
            if flat_depth is not None:
                zz = flat_depth[src_tri].astype(np.float32) + z_offset
            else:
                zz = (l0 * z[t, 0][ti] + l1 * z[t, 1][ti] +
                      l2 * z[t, 2][ti]) + z_offset

            if depth_test:
                keep = zz < gbuf.depth[py, px]
                if not keep.any():
                    continue
                ti, l0, l1, l2, px, py, zz, src_tri = (
                    a[keep] for a in (ti, l0, l1, l2, px, py, zz, src_tri))

            iw0, iw1, iw2 = iw[t, 0][ti], iw[t, 1][ti], iw[t, 2][ti]
            invw = l0 * iw0 + l1 * iw1 + l2 * iw2
            invw = np.where(np.abs(invw) < 1e-20, 1e-20, invw)
            P = np.stack([l0 * iw0 / invw, l1 * iw1 / invw, l2 * iw2 / invw], axis=1)
            bwt = bw[t][ti]
            b = np.einsum('nk,nkc->nc', P, bwt)

            px_all.append(px)
            py_all.append(py)
            zz_all.append(zz.astype(np.float32))
            b_all.append(b.astype(np.float32))
            tri_all.append(src_tri)
            front_all.append((area[t] < 0.0)[ti])
            if gbuf.bary_lin is not None:
                L = np.stack([l0, l1, l2], axis=1)
                blin_all.append(np.einsum('nk,nkc->nc', L, bwt).astype(np.float32))

    if not px_all:
        return written
    px = np.concatenate(px_all)
    py = np.concatenate(py_all)
    zz = np.concatenate(zz_all)
    b = np.concatenate(b_all)
    tri = np.concatenate(tri_all)
    front = np.concatenate(front_all)
    blin = np.concatenate(blin_all) if blin_all else None

    if frags is not None:
        frags.add(px, py, tri, zz, b, front)

    if depth_write:
        pix = py.astype(np.int64) * W + px
        order = np.lexsort((zz, pix))
        pix_s = pix[order]
        first = np.empty(pix_s.size, bool)
        first[0] = True
        np.not_equal(pix_s[1:], pix_s[:-1], out=first[1:])
        win = order[first]
        wx, wy = px[win], py[win]
        better = zz[win] < gbuf.depth[wy, wx]
        win = win[better]
        wx, wy = wx[better], wy[better]
        gbuf.depth[wy, wx] = zz[win]
        gbuf.zndc[wy, wx] = zz[win]
        gbuf.tri[wy, wx] = tri[win]
        gbuf.bary[wy, wx] = b[win]
        gbuf.front[wy, wx] = front[win]
        if blin is not None:
            gbuf.bary_lin[wy, wx] = blin[win]
    return written + int(px.size)
