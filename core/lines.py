"""Line arc 2 -- R234 (1.77.0): the inker's line.

The 1.70 style pack made the line a distance field, 1.74 gave it a
brush's body: ends, roughness, drift, gaps, textures. What it still
lacked was the thing the field's eye named first -- "the outlines look
artificial. If you look at old cartoons, the outlines are dynamic" --
and the Imitation Study named the three roads to it. This module is
those three:

1. THE ISOPHOTE WEIGHT (Goodwin, Vollick & Hertzmann 2007). An inker's
   line is heavy where the form turns away from the light and light
   where it faces it, heavier on big smooth forms than on thin
   features, heavier near than far -- and one measurement gives all of
   that: the image-space distance from a silhouette point, walked
   inward along the surface's own normal, to the isophote where the
   light stops (n.l = the Shadow Level). Thick where that walk is
   long (a broad shadowed flank), thin where it is short (a lit edge,
   a thin feature the walk leaves at once), and perspective for free
   because the walk is measured in pixels. `isophote_distance` is that
   walk, from the silhouette seeds, on the G-buffer: it stops at the
   isophote, at the object's own edge, or at a depth step.

2. THREE MORE LINE SOURCES, all interior classes (the thin-inner line):
   FORM LINES -- the valleys of n.v (DeCarlo's image-space suggestive
   contours: the fold of a cheek, the crease of a concave form, found
   where the facing dips, by Steger's sub-pixel valley detector so
   the line is one pixel wide and never fires on the silhouette's own
   fall-off); SHADOW LINES -- the terminator itself, the isophote at
   the Shadow Level, inked the way a cel inker traced the painter's
   shadow boundary; TONE LINES -- flow-guided difference-of-Gaussians
   on the shaded frame (Kang 2007 / Winnemoller 2012's XDoG): a line
   wherever the tone steps, along the local edge flow so it holds
   together, thinned to its darkest pixel.

3. THE STROKE ROAD. The G-buffer's contour is a pixel staircase; a pen
   draws a curve. The thinned contour is read as a GRAPH -- every
   contour pixel and the runs of its eight neighbours; two runs is a
   path pixel, one an end, three a junction -- and then everything is
   a LOCAL operation on that graph, so a pooled band computes what the
   whole frame computes: Taubin (shrink-free) smoothing of the pixel
   positions; the turning angle of the smoothed chain as PRESSURE (the
   pen bears down through a curve and pauses at a corner: a blob);
   OVERSHOOT -- every true end and every chain arriving at a junction
   runs past it along its tangent, tapering, for a hashed fraction of
   the dial (the crossed corners of a 40s drawing); the chains are
   drawn back as seeds carrying their ORIGIN pixel, so every width,
   colour and depth lookup downstream still reads a surface pixel.
   Nothing here needs the previous frame: the per-end amounts hash the
   end's world position (Kalnins' coherence without state), and the
   SURFACE anchor in ink.py samples the line's noises at the surface
   point under the line, so the hand's pressure rides the object, not
   the screen.

Every stage is bounded in rows -- the march by its range, the graph
operations by their iteration count, the filters by their kernels --
and render._ink_reach_rows keeps that many context rows, so bands draw
bitwise what the whole frame draws. All on the CPU from the shared
G-buffer on either device road, the outline doctrine.
"""

import numpy as np

from . import film as FILM
from . import ink as INK
from . import raster

INK_ANCHOR_ITEMS = (
    ('SCREEN', "Screen",
     "The line's noises (weight, roughness, drift, gaps, streaks) are "
     "laid out on the screen: still under a still camera, and they "
     "slide over a moving object -- the traced-cel convention"),
    ('SURFACE', "Surface",
     "The noises are sampled at the surface point under the line, so "
     "the hand's pressure travels with the object and the camera can "
     "move without a shower-door: coherent without any previous frame"),
)

# (dy, dx) around the eight-ring, clockwise from east
OFFS8 = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


# ------------------------------------------------------- surface fields


def surface_attrs(mesh, gbuf, idx, want_p=True, want_n=True):
    """World positions and unit normals at the flat pixel indices `idx`
    (covered pixels): the vertex attributes gathered per corner and
    summed by the barycentrics -- four times faster than the generic
    fetch at a frame's size, and a subset costs only its own size. The
    normal is the smooth one on smooth faces, the face normal on flat
    ones, exactly what the shading read."""
    idx = np.asarray(idx, np.int64)
    t = gbuf.tri.reshape(-1)[idx]
    t = np.where(t >= 0, t, 0)
    tv = mesh.tris[t]
    b = gbuf.bary.reshape(-1, 3)[idx]
    P = None
    N = None
    if want_p:
        V = mesh.verts
        P = (V[tv[:, 0]] * b[:, 0:1] + V[tv[:, 1]] * b[:, 1:2]
             + V[tv[:, 2]] * b[:, 2:3]).astype(np.float32)
    if want_n:
        fn = getattr(mesh, 'face_normals', None)
        vn = getattr(mesh, 'normals', None)
        if vn is not None:
            N = (vn[tv[:, 0]] * b[:, 0:1] + vn[tv[:, 1]] * b[:, 1:2]
                 + vn[tv[:, 2]] * b[:, 2:3]).astype(np.float32)
            sm = getattr(mesh, 'smooth', None)
            if sm is not None and fn is not None:
                N = np.where(np.asarray(sm)[t][:, None], N, fn[t])
        elif fn is not None:
            N = np.asarray(fn[t], np.float32)
        else:
            N = np.zeros((idx.size, 3), np.float32)
        ln = np.sqrt((N * N).sum(1))
        N = (N / np.maximum(ln, np.float32(1e-12))[:, None]).astype(np.float32)
    return P, N


def surface_normals(mesh, gbuf, cov):
    """Per-pixel unit normals, flat (H*W, 3) float32; zero off-surface."""
    flat = cov.reshape(-1)
    idx = np.nonzero(flat)[0]
    out = np.zeros((flat.size, 3), np.float32)
    if idx.size:
        out[idx] = surface_attrs(mesh, gbuf, idx, want_p=False)[1]
    return out


def surface_points(mesh, gbuf, cov):
    """Per-pixel world positions, flat (H*W, 3) float32; zero off-surface."""
    flat = cov.reshape(-1)
    idx = np.nonzero(flat)[0]
    out = np.zeros((flat.size, 3), np.float32)
    if idx.size:
        out[idx] = surface_attrs(mesh, gbuf, idx, want_n=False)[0]
    return out


def key_light(scene):
    """The key lamp of the line: the first non-ambient light in scene
    order (the convention the shadow-side weight set in 1.70)."""
    for l in (getattr(scene, 'lights', None) or ()):
        if str(getattr(l, 'type', '')).upper() != 'AMBIENT':
            return l
    return None


def light_vectors(light, P):
    """Unit vectors toward the key lamp at points P (n, 3): a Sun or
    Hemi lamp's one direction, a point / spot / area lamp's true
    direction from every point (a point lamp's rotation means nothing
    and is not read)."""
    n = P.shape[0]
    kind = str(getattr(light, 'type', 'POINT')).upper()
    if kind in ('SUN', 'HEMI'):
        d = np.asarray(getattr(light, 'direction', (0.0, 0.0, -1.0)),
                       np.float32)
        ln = float(np.linalg.norm(d))
        L = (-d / ln) if ln > 1e-9 else np.array([0.0, 0.0, 1.0], np.float32)
        return np.broadcast_to(L.astype(np.float32), (n, 3)).copy()
    pos = np.asarray(getattr(light, 'position', (0.0, 0.0, 0.0)), np.float32)
    L = pos[None, :] - np.asarray(P, np.float32)
    ln = np.sqrt((L * L).sum(1))
    return (L / np.maximum(ln, np.float32(1e-9))[:, None]).astype(np.float32)


def ndl_field(scene, mesh, gbuf, cov, N=None, P=None):
    """n.l of the key lamp per pixel, UNclipped (negative in shadow),
    flat float32; 1 off-surface and when there is no lamp."""
    light = key_light(scene)
    flat_cov = cov.reshape(-1)
    if light is None:
        return np.ones(flat_cov.shape, np.float32)
    if N is None:
        N = surface_normals(mesh, gbuf, cov)
    kind = str(getattr(light, 'type', 'POINT')).upper()
    if kind not in ('SUN', 'HEMI') and P is None:
        P = surface_points(mesh, gbuf, cov)
    L = light_vectors(light, P if P is not None else N)
    ndl = (N * L).sum(1)
    return np.where(flat_cov, ndl, 1.0).astype(np.float32)


def ndv_field(mesh, gbuf, cov, eye, N=None, P=None):
    """|n.v| per pixel (the facing), flat float32; 0 off-surface."""
    if N is None:
        N = surface_normals(mesh, gbuf, cov)
    if P is None:
        P = surface_points(mesh, gbuf, cov)
    V = np.asarray(eye, np.float32)[None, :] - P
    ln = np.sqrt((V * V).sum(1))
    V = V / np.maximum(ln, np.float32(1e-9))[:, None]
    ndv = np.abs((N * V).sum(1))
    return np.where(cov.reshape(-1), ndv, 0.0).astype(np.float32)


def pixel_world_size(proj, depth, H):
    """The world-space size of one internal pixel at view `depth` (a
    flat float32 array): depth * 2 / (proj[1,1] * H) under a perspective
    projection, the constant 2 / (proj[1,1] * H) under an orthographic
    one -- the scale that lets a screen dial mean the same thing on the
    surface."""
    p = np.asarray(proj, np.float64)
    f = float(p[1, 1]) if abs(float(p[1, 1])) > 1e-12 else 1.0
    base = 2.0 / (f * float(H))
    if abs(float(p[3, 2])) > 1e-9:
        return (np.asarray(depth, np.float32) * np.float32(base)).astype(
            np.float32)
    return np.full(np.asarray(depth).shape, np.float32(base), np.float32)


# --------------------------------------------------- the isophote weight


def screen_normal_dirs(vp, P, N, W, H):
    """The image-space direction of the surface normal at points P (unit
    2-vectors in pixels; zero where the normal projects to a point):
    the projection of P + eps*N minus the projection of P."""
    m = np.asarray(vp, np.float64)
    P = np.asarray(P, np.float64)
    N = np.asarray(N, np.float64)
    aw = P @ m[3, :3] + m[3, 3]
    eps = 1e-3 * np.maximum(np.abs(aw), 1e-3)
    Q = P + N * eps[:, None]
    a = P @ m[:2, :3].T + m[:2, 3]
    b = Q @ m[:2, :3].T + m[:2, 3]
    bw = Q @ m[3, :3] + m[3, 3]
    aw = np.where(np.abs(aw) < 1e-9, 1e-9, aw)
    bw = np.where(np.abs(bw) < 1e-9, 1e-9, bw)
    dx = (b[:, 0] / bw - a[:, 0] / aw) * (0.5 * W)
    dy = (b[:, 1] / bw - a[:, 1] / aw) * (0.5 * H)
    ln = np.sqrt(dx * dx + dy * dy)
    ok = ln > 1e-12
    dx = np.where(ok, dx / np.where(ok, ln, 1.0), 0.0)
    dy = np.where(ok, dy / np.where(ok, ln, 1.0), 0.0)
    return dx.astype(np.float32), dy.astype(np.float32)


def isophote_distance(sil, scene, mesh, gbuf, omap, dmap, cov, vp, eye,
                      level, rng_px, depth_thr=0.02):
    """Goodwin's isophote distance at every silhouette seed pixel: the
    walk from the seed inward along the surface normal, one pixel a
    step, until the light comes back (n.l >= level), the object ends
    (another object, the sky) or the depth steps (a fold of the same
    object hiding its far side) -- capped at `rng_px`. Returns the
    distance in pixels, flat float32 (0 off the seeds and at lit
    seeds). Where the normal faces the camera (a face-on step edge)
    the inward direction is the boundary's own: away from the
    neighbours the seed differs from. Everything is gathered at the
    seeds and the pixels the walk visits -- never a frame-wide field."""
    H, W = sil.shape
    out = np.zeros(H * W, np.float32)
    sy, sx = np.nonzero(sil)
    n = sy.size
    light = key_light(scene)
    if n == 0 or rng_px <= 0.0 or light is None:
        return out
    sflat = sy * W + sx
    Ps, Ns = surface_attrs(mesh, gbuf, sflat)
    nx, ny = screen_normal_dirs(vp, Ps, Ns, W, H)
    # how much of the normal lies in the image plane: all of it at a
    # true silhouette, none of it on a face-on step edge
    Vv = np.asarray(eye, np.float32)[None, :] - Ps
    Vv = Vv / np.maximum(np.sqrt((Vv * Vv).sum(1)), np.float32(1e-9))[:, None]
    cosv = np.clip(np.abs((Ns * Vv).sum(1)), 0.0, 1.0)
    strength = np.sqrt(np.maximum(1.0 - cosv * cosv, 0.0))
    # inward = minus the outward projected normal
    dx = -nx
    dy = -ny
    # the boundary's inward direction: away from the across-neighbours
    bx = np.zeros(n, np.float32)
    by = np.zeros(n, np.float32)
    o0 = omap[sy, sx]
    d0 = dmap[sy, sx]
    for ddy, ddx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        qy = np.clip(sy + ddy, 0, H - 1)
        qx = np.clip(sx + ddx, 0, W - 1)
        across = (omap[qy, qx] != o0) | \
            (dmap[qy, qx] > d0 * np.float32(1.0 + depth_thr))
        bx -= np.float32(ddx) * across
        by -= np.float32(ddy) * across
    bl = np.sqrt(bx * bx + by * by)
    use_n = strength > 0.25
    bok = bl > 0.0
    dirx = np.where(use_n, dx, np.where(bok, bx / np.where(bok, bl, 1.0), 0.0))
    diry = np.where(use_n, dy, np.where(bok, by / np.where(bok, bl, 1.0), 0.0))

    def ndl_at(flat_idx):
        Pq, Nq = surface_attrs(mesh, gbuf, flat_idx)
        L = light_vectors(light, Pq)
        return (Nq * L).sum(1)

    alive = (ndl_at(sflat) < level) & ((dirx != 0.0) | (diry != 0.0))
    dist = np.zeros(n, np.float32)
    px = sx.astype(np.float32) + np.float32(0.5)
    py = sy.astype(np.float32) + np.float32(0.5)
    prev = d0.astype(np.float32)
    steps = int(np.ceil(rng_px))
    for k in range(1, steps + 1):
        if not alive.any():
            break
        qx = np.clip(np.floor(px + dirx * np.float32(k)).astype(np.int64),
                     0, W - 1)
        qy = np.clip(np.floor(py + diry * np.float32(k)).astype(np.int64),
                     0, H - 1)
        same = cov[qy, qx] & (omap[qy, qx] == o0)
        dq = dmap[qy, qx]
        near = np.minimum(dq, prev)
        jump = np.abs(dq - prev) > np.float32(depth_thr) * np.maximum(
            near, np.float32(1e-4))
        # n.l only where the walk is still alive and on the surface
        lit = np.zeros(n, bool)
        ask = alive & same
        if ask.any():
            ai = np.nonzero(ask)[0]
            lit[ai] = ndl_at(qy[ai] * W + qx[ai]) >= level
        stop = (~same) | jump | lit
        go = alive & ~stop
        dist = np.where(go, np.float32(k), dist)
        alive = go
        prev = np.where(go, dq, prev)
    out[sflat] = np.minimum(dist, np.float32(rng_px))
    return out


def isophote_factor(dist_flat, rng_px, amount):
    """The width factor from the isophote distance: the full width at
    the range, a quarter of it at zero (the lit side keeps a line),
    eased, and blended by `amount` toward the plain width."""
    t = np.clip(dist_flat / np.float32(max(rng_px, 1e-6)), 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    f = np.float32(0.25) + np.float32(0.75) * t
    return (np.float32(1.0 - amount) + np.float32(amount) * f).astype(
        np.float32)


# ------------------------------------------------------ the line sources


def dilate8(mask, r):
    """Grow a boolean mask by r pixels in the eight-neighbourhood."""
    m = np.asarray(mask, bool)
    for _ in range(int(r)):
        g = m.copy()
        g[:, 1:] |= m[:, :-1]
        g[:, :-1] |= m[:, 1:]
        g[1:, :] |= m[:-1, :]
        g[:-1, :] |= m[1:, :]
        g[1:, 1:] |= m[:-1, :-1]
        g[1:, :-1] |= m[:-1, 1:]
        g[:-1, 1:] |= m[1:, :-1]
        g[:-1, :-1] |= m[1:, 1:]
        m = g
    return m


def _gauss2(f, sigma):
    return FILM.gaussian(np.asarray(f, np.float32)[:, :, None], sigma)[:, :, 0]


def _masked_gauss(f, mask, sigma):
    """A Gaussian that never reads outside `mask`: the normalised
    convolution of (f * m, m), the ratio where the mask has weight."""
    m = np.asarray(mask, np.float32)
    pair = np.stack([np.asarray(f, np.float32) * m, m], -1)
    b = FILM.gaussian(pair, sigma)
    w = b[:, :, 1]
    return np.where(w > 1e-6, b[:, :, 0] / np.maximum(w, 1e-6), 0.0).astype(
        np.float32)


def _d1(f):
    gx = np.zeros_like(f)
    gy = np.zeros_like(f)
    gx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) * np.float32(0.5)
    gy[1:-1, :] = (f[2:, :] - f[:-2, :]) * np.float32(0.5)
    return gx, gy


def form_seeds(ndv, cov, near_sil, sigma, thr, omap=None):
    """FORM LINES: the valleys of the facing |n.v| -- image-space
    suggestive contours. The field is smoothed by `sigma` (edge-padded,
    zero off-surface), its Hessian's larger eigenvalue is the valley's
    depth across it, and Steger's test keeps only the pixel that holds
    the valley's floor (the first derivative along the eigenvector
    vanishes within half a pixel): a one-pixel line that never fires on
    the silhouette's monotonic fall-off. A valley has two walls: with
    `omap` both walls (sigma + 1 pixels out along the eigenvector) must
    be the seed's own object, so a dip the smoothing borrowed from a
    neighbouring object is not a fold. `thr` is the eigenvalue floor in
    facing per pixel squared."""
    F = _gauss2(np.where(cov, ndv, 0.0).astype(np.float32), sigma)
    gx, gy = _d1(F)
    Fxx = np.zeros_like(F)
    Fyy = np.zeros_like(F)
    Fxy = np.zeros_like(F)
    Fxx[:, 1:-1] = F[:, 2:] - np.float32(2.0) * F[:, 1:-1] + F[:, :-2]
    Fyy[1:-1, :] = F[2:, :] - np.float32(2.0) * F[1:-1, :] + F[:-2, :]
    Fxy[1:-1, 1:-1] = (F[2:, 2:] - F[2:, :-2] - F[:-2, 2:] + F[:-2, :-2]) \
        * np.float32(0.25)
    tr = Fxx + Fyy
    det = Fxx * Fyy - Fxy * Fxy
    disc = np.sqrt(np.maximum(tr * tr * np.float32(0.25) - det, 0.0))
    lam = tr * np.float32(0.5) + disc
    off = np.abs(Fxy) > 1e-12
    ex = np.where(off, Fxy, lam - Fyy)
    ey = np.where(off, lam - Fxx, Fxy)
    el = np.sqrt(ex * ex + ey * ey)
    ok = el > 1e-12
    ex = np.where(ok, ex / np.where(ok, el, 1.0), 1.0)
    ey = np.where(ok, ey / np.where(ok, el, 1.0), 0.0)
    second = Fxx * ex * ex + np.float32(2.0) * Fxy * ex * ey + Fyy * ey * ey
    first = gx * ex + gy * ey
    s_ok = np.abs(second) > 1e-12
    tpos = np.where(s_ok, -first / np.where(s_ok, second, 1.0), 9.0)
    out = (lam > np.float32(thr)) & (np.abs(tpos) <= 0.5) & cov \
        & ~near_sil & (ndv > np.float32(0.1))
    if omap is not None and out.any():
        H, W = cov.shape
        oy, ox = np.nonzero(out)
        r = float(sigma) + 1.0
        same = np.ones(oy.size, bool)
        for sgn in (1.0, -1.0):
            qx = np.clip(np.floor(ox + 0.5 + sgn * r * ex[oy, ox]).astype(
                np.int64), 0, W - 1)
            qy = np.clip(np.floor(oy + 0.5 + sgn * r * ey[oy, ox]).astype(
                np.int64), 0, H - 1)
            same &= cov[qy, qx] & (omap[qy, qx] == omap[oy, ox])
        out[oy[~same], ox[~same]] = False
    return out


def shadow_seeds(ndl, cov, omap, level, near_sil):
    """SHADOW LINES: the terminator -- every pixel on the dark side of
    the Shadow Level with a lit four-neighbour of the same object. One
    pixel wide on the shadow's side, the way an inker's shadow line sat
    inside the painter's shadow."""
    H, W = cov.shape
    dark = cov & (ndl < np.float32(level))
    lit = cov & (ndl >= np.float32(level))
    out = np.zeros((H, W), bool)
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        a = (slice(max(dy, 0), H + min(dy, 0)), slice(max(dx, 0), W + min(dx, 0)))
        b = (slice(max(-dy, 0), H + min(-dy, 0)),
             slice(max(-dx, 0), W + min(-dx, 0)))
        out[a] |= dark[a] & lit[b] & (omap[a] == omap[b])
    return out & ~near_sil


def tone_seeds(img, cov, near_sil, sigma_e, thr, omap=None):
    """TONE LINES: flow-guided difference-of-Gaussians on the shaded
    frame's luminance (display gamma, so a shadow's step counts like a
    highlight's). The edge flow is the minor eigenvector of the
    smoothed structure tensor; the DoG (sigma_e against 1.6 sigma_e)
    is integrated along that flow for coherence and thinned across it
    to the darkest pixel. A line where the response is below -thr: the
    dark side of every tone step -- the terminator, a cast shadow's
    edge, a painted highlight's rim, a texture's edge. A tone line is
    a step WITHIN a surface: with `omap` the brighter side of the step
    (two pixels out along the gradient) must be the seed's own object,
    so the dark lobe a silhouette casts into a dark object against a
    bright background is not a line. The sky is zero luminance, so a
    surface's negative lobe against it falls off the surface and is
    never a seed. The flow run and the thinning are computed only
    where the DoG itself is already dark-side (a quarter of the
    threshold), a per-pixel gate that keeps the run cheap."""
    H, W = cov.shape
    lum = (np.asarray(img[:, :, :3], np.float32) @ np.array(
        [0.2126, 0.7152, 0.0722], np.float32)).astype(np.float32)
    Lp = np.where(cov, np.clip(lum, 0.0, 1.0), 0.0).astype(np.float32) \
        ** np.float32(1.0 / 2.2)
    # the edge flow from the structure tensor of the surface-only field
    gx, gy = _d1(Lp)
    inner = cov.copy()
    inner[:, 1:] &= cov[:, :-1]
    inner[:, :-1] &= cov[:, 1:]
    inner[1:, :] &= cov[:-1, :]
    inner[:-1, :] &= cov[1:, :]
    gx = np.where(inner, gx, 0.0).astype(np.float32)
    gy = np.where(inner, gy, 0.0).astype(np.float32)
    T = FILM.gaussian(np.stack([gx * gx, gx * gy, gy * gy], -1),
                      1.5 * sigma_e)
    E, Fx, G = T[:, :, 0], T[:, :, 1], T[:, :, 2]
    lam1 = np.float32(0.5) * (E + G + np.sqrt((E - G) ** 2 + np.float32(4.0)
                                              * Fx * Fx))
    vx = Fx
    vy = lam1 - E
    vl = np.sqrt(vx * vx + vy * vy)
    ok = vl > 1e-9
    vx = np.where(ok, vx / np.where(ok, vl, 1.0), 1.0).astype(np.float32)
    vy = np.where(ok, vy / np.where(ok, vl, 1.0), 0.0).astype(np.float32)
    # the DoG
    D = _gauss2(Lp, sigma_e) - np.float32(0.98) * _gauss2(Lp, 1.6 * sigma_e)
    D = np.where(cov, D, 0.0).astype(np.float32)
    cand = (D < -np.float32(0.25 * thr)) & cov & ~near_sil
    if not cand.any():
        return np.zeros((H, W), bool)
    run = dilate8(cand, 1)
    ry, rx = np.nonzero(run)
    tx = -vy[ry, rx]                           # along the edge
    ty = vx[ry, rx]
    sm = 2.5 * sigma_e
    M = int(np.ceil(2.0 * sm))
    ms = np.arange(-M, M + 1, dtype=np.float32)
    gm = np.exp(-0.5 * (ms / np.float32(sm)) ** 2).astype(np.float32)
    gm /= gm.sum()
    fx = rx.astype(np.float32)
    fy = ry.astype(np.float32)
    acc = np.zeros(ry.size, np.float32)
    for i, mm in enumerate(ms):
        acc += gm[i] * INK.bilinear_at(D, fx + tx * mm, fy + ty * mm)
    Ff = np.zeros((H, W), np.float32)
    Ff[ry, rx] = acc
    # thin across the flow: keep the darkest pixel of the response
    cy, cx = np.nonzero(cand)
    ux = vx[cy, cx]
    uy = vy[cy, cx]
    here = Ff[cy, cx]
    left = INK.bilinear_at(Ff, cx + ux, cy + uy)
    right = INK.bilinear_at(Ff, cx - ux, cy - uy)
    keep = (here < -np.float32(thr)) & (here <= left) & (here <= right)
    if omap is not None:
        # the brighter side, two pixels out along the gradient
        fxc = cx.astype(np.float32) + np.float32(0.5)
        fyc = cy.astype(np.float32) + np.float32(0.5)
        ax = np.clip(np.floor(fxc + 2.0 * ux).astype(np.int64), 0, W - 1)
        ay = np.clip(np.floor(fyc + 2.0 * uy).astype(np.int64), 0, H - 1)
        bx = np.clip(np.floor(fxc - 2.0 * ux).astype(np.int64), 0, W - 1)
        by = np.clip(np.floor(fyc - 2.0 * uy).astype(np.int64), 0, H - 1)
        a_br = Lp[ay, ax] >= Lp[by, bx]
        qx = np.where(a_br, ax, bx)
        qy = np.where(a_br, ay, by)
        keep &= cov[qy, qx] & (omap[qy, qx] == omap[cy, cx])
    out = np.zeros((H, W), bool)
    out[cy[keep], cx[keep]] = True
    return out


# ------------------------------------------------------- the stroke road


def contour_graph(seed, prune=2):
    """The thinned contour as a graph. Returns a dict: `py`, `px` (the
    contour pixels), `nb` (n, 8) neighbour indices around the ring (-1
    where none), `runs` (n,) the number of runs of set neighbours
    around the ring, `rid` (n, 8) the run each neighbour belongs to (1
    or 2 for path pixels; 0 where none), `wt` (n, 8) the averaging
    weights over the first two runs, `rep1`, `rep2` (n,) one member of
    each run (-1 where none). Path pixels have two runs, ends one,
    junctions three or more. `prune` passes strip one-pixel spurs off
    junctions first."""
    thin = INK._thin_4to8(seed)
    H, W = thin.shape

    def build(th):
        py, px = np.nonzero(th)
        n = py.size
        idx = np.full((H, W), -1, np.int64)
        idx[py, px] = np.arange(n)
        nb = np.full((n, 8), -1, np.int64)
        for k, (dy, dx) in enumerate(OFFS8):
            qy = py + dy
            qx = px + dx
            ok = (qy >= 0) & (qy < H) & (qx >= 0) & (qx < W)
            nb[:, k] = np.where(ok, idx[np.clip(qy, 0, H - 1),
                                        np.clip(qx, 0, W - 1)], -1)
        v = nb >= 0
        prev = np.roll(v, 1, axis=1)
        starts = v & ~prev
        n_runs = starts.sum(1)
        rid = np.cumsum(starts, axis=1)
        # a run wrapping past the ring's end is the run that started
        # last: its members before the first start read 0
        wrap = (rid == 0) & v
        rid = np.where(wrap, n_runs[:, None], rid)
        rid = np.where(v, rid, 0)
        return py, px, nb, v, n_runs, rid

    py, px, nb, v, n_runs, rid = build(thin)
    for _ in range(int(prune)):
        if py.size == 0:
            break
        # a spur: one run, and that run holds a junction pixel
        junc = n_runs >= 3
        jn = np.where(v, junc[np.maximum(nb, 0)], False)
        spur = (n_runs == 1) & jn.any(1)
        if not spur.any():
            break
        thin[py[spur], px[spur]] = False
        py, px, nb, v, n_runs, rid = build(thin)
    n = py.size
    size = np.zeros((n, 3), np.int64)
    for r in (1, 2):
        size[:, r] = (rid == r).sum(1)
    inv = np.where(size > 0, 1.0 / np.maximum(size, 1), 0.0)
    wt = np.zeros((n, 8), np.float32)
    for r in (1, 2):
        wt += np.where(rid == r, inv[:, r][:, None], 0.0)
    path = n_runs == 2
    wt = np.where(path[:, None], wt * np.float32(0.5), 0.0).astype(np.float32)
    ar = np.arange(8)[None, :]
    first1 = np.where((rid == 1), ar, 99).min(1)
    first2 = np.where((rid == 2), ar, 99).min(1)
    rep1 = np.where(first1 < 8, nb[np.arange(n), np.minimum(first1, 7)], -1)
    rep2 = np.where(first2 < 8, nb[np.arange(n), np.minimum(first2, 7)], -1)
    return {'thin': thin, 'py': py, 'px': px, 'nb': nb, 'runs': n_runs,
            'rid': rid, 'wt': wt, 'rep1': rep1, 'rep2': rep2, 'H': H, 'W': W}


def _run_mean(g, field, r):
    """Mean of a per-pixel field over each pixel's run r (n,) or (n,c)."""
    nb = g['nb']
    sel = (g['rid'] == r)
    cnt = np.maximum(sel.sum(1), 1)
    f = np.asarray(field)
    if f.ndim == 1:
        acc = np.where(sel, f[np.maximum(nb, 0)], 0.0).sum(1)
        return (acc / cnt).astype(np.float32)
    acc = np.where(sel[:, :, None], f[np.maximum(nb, 0)], 0.0).sum(1)
    return (acc / cnt[:, None]).astype(np.float32)


def smooth_positions(g, iters, lam=0.5, mu=-0.53, pinned=None):
    """Taubin smoothing of the contour pixels' positions along the graph:
    a shrink step and an inflate step per iteration, path pixels only
    (ends, junctions and `pinned` corners hold their place). Bounded:
    after k iterations a position depends on pixels within k steps
    along the chain."""
    n = g['py'].size
    pos = np.stack([g['px'] + 0.5, g['py'] + 0.5], 1).astype(np.float32)
    if n == 0 or iters <= 0:
        return pos
    nb = np.maximum(g['nb'], 0)
    wt = g['wt']
    path = g['runs'] == 2
    if pinned is not None:
        path = path & ~pinned
    sm = pos.copy()
    for _ in range(int(iters)):
        for w in (lam, mu):
            m = (wt[:, :, None] * sm[nb]).sum(1)
            upd = sm + np.float32(w) * (m - sm)
            sm = np.where(path[:, None], upd, sm).astype(np.float32)
    return sm


def curvature(g, sm, passes=6):
    """The NET turning of the smoothed chain at every path pixel, in
    radians per pixel: the signed turning angle between the runs on
    either side, averaged `passes` times along the chain with its sign
    kept -- so a pixel staircase's jog (a turn and the turn back)
    cancels to nothing, a corner keeps its whole angle and a curve its
    steady rate. Returned unsigned."""
    n = g['py'].size
    if n == 0:
        return np.zeros(0, np.float32)
    path = g['runs'] == 2
    m1 = _run_mean(g, sm, 1)
    m2 = _run_mean(g, sm, 2)
    t1 = sm - m1
    t2 = m2 - sm
    l1 = np.sqrt((t1 * t1).sum(1))
    l2 = np.sqrt((t2 * t2).sum(1))
    cross = t1[:, 0] * t2[:, 1] - t1[:, 1] * t2[:, 0]
    dot = (t1 * t2).sum(1)
    ang = np.arctan2(cross, dot)
    kap = np.where(path, ang / np.maximum((l1 + l2) * np.float32(0.5),
                                          np.float32(1e-6)), 0.0)
    kap = kap.astype(np.float32)
    for _ in range(int(passes)):
        a = _run_mean(g, kap, 1)
        b = _run_mean(g, kap, 2)
        kap = np.where(path, (kap + a + b) / np.float32(3.0), kap).astype(
            np.float32)
    return np.abs(kap).astype(np.float32)


def _walk_back(g, start, prev0, steps):
    """From the pixels `start` (each an end, or a chain pixel next to a
    junction with `prev0` naming that junction), walk `steps` pixels
    back along the chain, never turning into the run the previous
    pixel came from and never into a junction. Returns the pixel
    reached and how many steps were taken."""
    nb = g['nb']
    rid = g['rid']
    rep1 = g['rep1']
    rep2 = g['rep2']
    runs = g['runs']
    cur = start.copy()
    prev = prev0.copy()
    taken = np.zeros(start.shape, np.int64)
    for _ in range(int(steps)):
        match = (nb[cur] == prev[:, None]) & (prev[:, None] >= 0)
        prev_run = np.where(match, rid[cur], 0).max(1)
        r1 = rep1[cur]
        r2 = rep2[cur]
        nxt = np.where(prev_run == 1, r2, r1)
        ok = (nxt >= 0) & (runs[cur] <= 2) & (runs[cur] >= 1)
        ok &= runs[np.maximum(nxt, 0)] <= 2
        prev = np.where(ok, cur, prev)
        cur = np.where(ok, nxt, cur)
        taken = taken + ok.astype(np.int64)
    return cur, taken


def stroke_road(seed, P_at, rs, smooth_px, pressure, overshoot_px,
                salt, frame_edge=True):
    """The stroke road on one seed class. Returns (mask, origin, weight):
    the re-drawn seed mask, the flat origin pixel per seed pixel (-1
    where none) and a per-seed-pixel width factor (pressure, the
    overshoot's taper). `P_at(flat)` gives world positions, which hash
    the per-end overshoot; `rs` is the resolution scale; dials in
    internal pixels."""
    H, W = seed.shape
    g = contour_graph(seed)
    py, px = g['py'], g['px']
    n = py.size
    origin = np.full(H * W, -1, np.int64)
    weight = np.ones(H * W, np.float32)
    if n == 0:
        return np.zeros((H, W), bool), origin, weight
    src = (py * W + px).astype(np.int64)
    iters = int(min(64, np.ceil(2.0 * smooth_px * smooth_px))) \
        if smooth_px > 0.0 else 0
    runs = g['runs']
    path = runs == 2
    # the chain's turning, read off a copy smoothed two pixels wide (the
    # pixel staircase turns 45 degrees at every step): corners pin the
    # smoothing (a box keeps its corners) and drive the pressure
    kap = None
    if iters > 0 or pressure > 0.0:
        kap = curvature(g, smooth_positions(g, 8))
    pinned = None
    if kap is not None and iters > 0:
        pinned = kap > np.float32(0.15 / max(rs, 1e-6))
    sm = smooth_positions(g, iters, pinned=pinned)
    # pressure: the pen bears down through a curve, pauses at a corner
    wpix = np.ones(n, np.float32)
    if pressure > 0.0:
        kref = np.float32(0.25 / max(rs, 1e-6))
        t = np.clip(kap / kref, 0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        wpix = (1.0 + np.float32(1.5 * pressure) * t).astype(np.float32)
    # the points to draw: every pixel, the midpoints toward its runs
    pts = [sm]
    srcs = [src]
    wts = [wpix]
    prio = [np.zeros(n, np.int64)]
    for r in (1, 2):
        has = (g['rid'] == r).any(1)
        m = _run_mean(g, sm, r)
        mid = np.where(has[:, None], (sm + m) * np.float32(0.5), sm)
        pts.append(mid)
        srcs.append(src)
        wts.append(wpix)
        prio.append(np.ones(n, np.int64))
    if overshoot_px > 0.0:
        # chain ends: true ends off the frame border, and path pixels
        # whose run holds a junction (the chain arrives there)
        border = (py == 0) | (py == H - 1) | (px == 0) | (px == W - 1)
        is_end = (runs == 1) & ~(border if frame_edge else False)
        junc = runs >= 3
        v = g['nb'] >= 0
        jn = np.where(v, junc[np.maximum(g['nb'], 0)], False)
        at_junc = path & jn.any(1)
        cand = np.nonzero(is_end | at_junc)[0]
        if cand.size:
            # the junction neighbour of an arrival: the first junction
            # around its ring; the walk starts away from it
            jn_c = jn[cand]
            jk = np.where(jn_c, np.arange(8)[None, :], 99).min(1)
            jpix = np.where(jk < 8, g['nb'][cand, np.minimum(jk, 7)], -1)
            prev0 = np.where(at_junc[cand], jpix, -1)
            back, taken = _walk_back(g, cand, prev0, 4)
            ok = taken >= 2
            e = cand[ok]
            back = back[ok]
            jsel = at_junc[e] & (jpix[ok] >= 0)
            start = np.where(jsel[:, None], sm[np.maximum(jpix[ok], 0)],
                             sm[e])
            t = start - sm[back]
            tl = np.sqrt((t * t).sum(1))
            t = t / np.maximum(tl, np.float32(1e-6))[:, None]
            # the amount hashes the end's world position: coherent from
            # frame to frame without a previous frame
            Pe = P_at(src[e])
            q = np.floor(Pe * np.float32(64.0)).astype(np.int64)
            h1 = INK._hash_u32(q[:, 0] * 3 + q[:, 2], q[:, 1] * 5 - q[:, 2],
                               salt + 91)
            h2 = INK._hash_u32(q[:, 1] * 7 + q[:, 0], q[:, 2] * 3 + q[:, 1],
                               salt + 92)
            go = h2 < 0.6
            length = overshoot_px * (0.5 + 0.5 * h1)
            ls = np.where(go, length, 0.0).astype(np.float32)
            _stubs(pts, srcs, wts, prio, start, t, ls, src[e], wpix[e],
                   overshoot_px)
        # crossed corners: at a corner of the chain (a local peak of the
        # turning), each arriving edge runs on past it along its own
        # tangent -- the 40s model sheet's crossed box corner
        if kap is not None:
            a = _run_mean(g, kap, 1)
            b = _run_mean(g, kap, 2)
            pk = path & (kap > np.float32(0.15 / max(rs, 1e-6))) \
                & (kap >= a) & (kap >= b)
            corners = np.nonzero(pk)[0]
            if corners.size:
                for r_from, r_other in ((1, 2), (2, 1)):
                    other = g['rep2'] if r_other == 2 else g['rep1']
                    back, taken = _walk_back(g, corners, other[corners], 4)
                    ok = taken >= 2
                    c = corners[ok]
                    if not c.size:
                        continue
                    t = sm[c] - sm[back[ok]]
                    tl = np.sqrt((t * t).sum(1))
                    t = t / np.maximum(tl, np.float32(1e-6))[:, None]
                    Pc = P_at(src[c])
                    q = np.floor(Pc * np.float32(64.0)).astype(np.int64)
                    h1 = INK._hash_u32(q[:, 0] * 3 + q[:, 2] + r_from,
                                       q[:, 1] * 5 - q[:, 2], salt + 93)
                    h2 = INK._hash_u32(q[:, 1] * 7 + q[:, 0],
                                       q[:, 2] * 3 + q[:, 1] + r_from,
                                       salt + 94)
                    ls = np.where(h2 < 0.6, overshoot_px * (0.5 + 0.5 * h1),
                                  0.0).astype(np.float32)
                    _stubs(pts, srcs, wts, prio, sm[c], t, ls, src[c],
                           wpix[c], overshoot_px)
    Pp = np.concatenate(pts, 0)
    S = np.concatenate(srcs, 0)
    Wt = np.concatenate(wts, 0)
    Pr = np.concatenate(prio, 0)
    ix = np.floor(Pp[:, 0]).astype(np.int64)
    iy = np.floor(Pp[:, 1]).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    key = (iy * W + ix)[ok]
    S = S[ok]
    Wt = Wt[ok]
    Pr = Pr[ok]
    # one deterministic winner per pixel: the lowest priority, then the
    # lowest origin index
    order = np.lexsort((S, Pr, key))
    key = key[order]
    S = S[order]
    Wt = Wt[order]
    first = np.ones(key.size, bool)
    first[1:] = key[1:] != key[:-1]
    key = key[first]
    origin[key] = S[first]
    weight[key] = Wt[first]
    mask = np.zeros(H * W, bool)
    mask[key] = True
    return mask.reshape(H, W), origin, weight


def _stubs(pts, srcs, wts, prio, start, t, ls, src, w0, overshoot_px):
    """Append the overshoot stubs: from `start` along the unit tangents
    `t` for `ls` pixels each, the width tapering to nothing at the tip."""
    n_s = int(np.ceil(overshoot_px))
    for s in range(1, n_s + 1):
        on = ls >= s
        if not on.any():
            break
        fs = np.float32(s)
        frac = fs / np.maximum(ls, np.float32(1e-6))
        taper = np.sqrt(np.clip(1.0 - frac, 0.0, 1.0))
        pts.append((start + t * fs)[on])
        srcs.append(src[on])
        wts.append((w0 * taper)[on].astype(np.float32))
        prio.append(np.full(int(on.sum()), 2 + s, np.int64))


def road_on(st):
    """True when any stroke-road dial is off its default."""
    return (float(getattr(st, 'ink_smooth', 0.0)) > 0.0
            or float(getattr(st, 'ink_pressure', 0.0)) > 0.0
            or float(getattr(st, 'ink_overshoot', 0.0)) > 0.0)


def road_rows(st, rs):
    """How many internal rows the stroke road reads past a band: the
    smoothing's iterations (one chain step each), the curvature's
    passes and run means, the overshoot's walk and stub."""
    rows = 0.0
    sm = float(getattr(st, 'ink_smooth', 0.0)) * rs
    if sm > 0.0:
        rows += min(64, np.ceil(2.0 * sm * sm)) + 2
    if sm > 0.0 or float(getattr(st, 'ink_pressure', 0.0)) > 0.0:
        # the turning: an 8-iteration smoothing, two run means, six
        # averaging passes, and the corner peak test's run means
        rows += 8 + 2 + 6 + 2
    ov = float(getattr(st, 'ink_overshoot', 0.0)) * rs
    if ov > 0.0:
        rows += ov + 8
    return rows


def source_rows(st, rs):
    """Rows the new line sources read past a band (both roads): the
    form filter's kernel, the shadow line's neighbour, the tone line's
    tensor blur, DoG, flow run and thinning."""
    rows = 0.0
    if getattr(st, 'outline_form', False):
        rows = max(rows, 3.0 * 2.0 * rs + 2.0 * 2.0 * rs + 4)
    if getattr(st, 'outline_shadow', False):
        rows = max(rows, 2.0)
    if getattr(st, 'outline_tone', False):
        se = 1.2 * rs
        rows = max(rows, 3.0 * 1.5 * se + 3.0 * 1.6 * se + 2.0 * 2.5 * se + 4)
    return rows
