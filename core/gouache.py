"""The painted background road and the setback (R233).

A 1940s background was a painting: gouache or oil on board, brushed, with
no ink line, photographed under the cels. And where the Fleischers wanted
depth they built the set as a miniature behind the cel and photographed
it through a lens -- the stereoptical setback -- so the background had
real shadows and a photographic softness while the cels stayed crisp.

This module gives a material a PAINT MODE. Cel (the default, everything
1.71 shipped) is the cel: flat paint, the ink line. Background is the
painting: the surface's lit colour is laid down as brush strokes, Meier's
way (SIGGRAPH 1996) -- particles fixed on the surface, each drawn as a
stroke in screen space at a screen-constant size, far to near, with its
colour, direction and size from the frame itself -- over an abstracted
base, with the ink suppressed. Because the particles live on the mesh,
the painting is coherent under camera motion by construction; because a
particle's place in its triangle's sequence never changes, a camera that
comes closer adds strokes at the END of every sequence, fading them in,
and never moves a stroke that is already there.

The setback is the lens: a softness by depth on the Background materials
(and the sky), the Cel materials composited sharp over it.

Both run once on the CPU from the shared G-buffer, in the render() tail
before the ink, on either device road -- parity by construction -- and
every field they read is a pure function of the frame (a stroke is a hash
of its triangle and its index), so a pooled band that keeps the reach
rows draws the frame's strokes bitwise.
"""

import numpy as np

from . import film as FILM
from . import media as MD
from . import patterns as PT

BG_DIRECTION_ITEMS = (
    ('GRADIENT', "Follow the Colour",
     "Strokes run along the base colour's contours (across its gradient), "
     "the way a painter follows the form -- Hertzmann's rule"),
    ('NORMAL', "Follow the Surface",
     "Strokes run across the surface normal's screen direction, the way "
     "Meier oriented them: along the silhouette, round the form"),
    ('ANGLE', "Fixed Angle",
     "Strokes at Stroke Angle, spread by Spread -- a flat, decorative "
     "brush"),
)

#: the particle budget of a frame: past it every triangle's sequence is
#: cut proportionally, so a giant ground plane cannot ask for a million
MAX_STROKES = 120000


def paint_lut(scene):
    """[material index] -> True for a Background material (one extra
    False for the sky slot)."""
    mats = getattr(scene, 'materials', None) or []
    return np.array([str(getattr(m, 'paint_mode', 'CEL')).upper()
                     == 'BACKGROUND' for m in mats] + [False], bool)


def any_background(scene):
    return bool(paint_lut(scene)[:-1].any())


def strokes_on(scene, st):
    return float(getattr(st, 'bg_paint', 0.0)) > 0.0 and any_background(scene)


def setback_on(scene, st):
    if float(getattr(st, 'setback', 0.0)) <= 0.0:
        return False
    return any_background(scene) or bool(getattr(st, 'setback_sky', True))


def on(scene, st):
    return strokes_on(scene, st) or setback_on(scene, st)


def stroke_px(st, ss):
    """The stroke size in INTERNAL pixels: an output-pixel dial times the
    supersample factor, so the dial means what it says on the frame."""
    return float(max(getattr(st, 'bg_stroke_size', 22.0), 1.0)) * float(ss)


def reach_rows(scene, st, ss):
    """Rows past a band the road reads or writes: the longest stroke's
    reach (a stroke seeded in the context rows paints into the band),
    the abstraction blur's taps, the setback's taps."""
    rows = 0.0
    if strokes_on(scene, st):
        size = stroke_px(st, ss)
        length = size * float(max(getattr(st, 'bg_stroke_length', 3.0), 1.0))
        rows += length * 0.5 * 1.3 + 3.0
        rows += 3.0 * float(max(getattr(st, 'bg_smooth', 2.0), 0.0)) * ss
    if setback_on(scene, st):
        rows += 3.0 * float(max(getattr(st, 'setback', 0.0), 0.0)) * ss + 1.0
    return int(np.ceil(rows))


# ---------------------------------------------------------------- helpers


def _masked_blur(rgb, mask, sigma):
    """A normalised-convolution Gaussian: only `mask` pixels contribute
    and only they are read back, so a sharp cel never bleeds into the
    painting beside it (nor the painting into the cel)."""
    if sigma <= 1e-3:
        return rgb
    m = mask.astype(np.float32)[:, :, None]
    num = FILM.gaussian(rgb * m, sigma)
    den = FILM.gaussian(m, sigma)
    out = num / np.maximum(den, 1e-6)
    return np.where(mask[:, :, None], out, rgb).astype(np.float32)


def _luminance(rgb):
    return (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152
            + rgb[..., 2] * 0.0722).astype(np.float32)


def _h(a, b, k):
    """A per-(triangle, particle, channel) hash in [0, 1]."""
    return PT.hash3(np.asarray(a, np.int64), np.asarray(b, np.int64),
                    np.int64(k))


# ---------------------------------------------------------------- strokes


def seed_particles(scene, gbuf, bgm, vp, eye, st, ss, seed=0):
    """Meier's particles for this frame: one sequence per visible
    Background triangle, cut to the screen density the stroke size asks.

    For triangle t the k-th particle sits at barycentrics hashed from
    (t, k) -- fixed on the surface for all time -- and the frame keeps
    particles 0..n_t where n_t is the triangle's projected area over the
    stroke spacing squared. A camera that comes closer raises n_t and
    appends; the last particle fades in over its fractional count, so
    nothing pops. Returns the visible particles' screen positions, world
    positions, triangle, index, and fade -- unsorted.
    """
    mesh = scene.mesh
    seen = np.unique(gbuf.tri[bgm])
    if seen.size == 0:
        return None
    # EVERY Background triangle of the mesh takes part in the budget, not
    # only the ones this G-buffer shows: a pooled band sees fewer
    # triangles than the frame, and a budget cut that read the band's
    # own total would cut its sequences differently from the frame's
    lut = paint_lut(scene)
    if mesh.mat_index is not None:
        all_bg = np.nonzero(lut[np.clip(mesh.mat_index, 0, lut.size - 1)])[0]
    else:
        all_bg = np.arange(mesh.tris.shape[0]) if lut[0] else seen
    tris = all_bg.astype(np.int64)
    verts = mesh.verts
    tv = mesh.tris[tris]                                  # (T, 3)
    P0, P1, P2 = verts[tv[:, 0]], verts[tv[:, 1]], verts[tv[:, 2]]
    # the projected area, vertices clamped in front of the eye
    m = np.asarray(vp, np.float32)
    H, W = gbuf.tri.shape

    def proj(P):
        ph = np.concatenate([P, np.ones((P.shape[0], 1), np.float32)], 1)
        c = ph @ m.T
        w = np.maximum(c[:, 3], 1e-4)
        sx = (c[:, 0] / w * 0.5 + 0.5) * W
        sy = (c[:, 1] / w * 0.5 + 0.5) * H
        return sx, sy

    x0, y0 = proj(P0)
    x1, y1 = proj(P1)
    x2, y2 = proj(P2)
    area_px = 0.5 * np.abs((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0))
    area_px = np.minimum(area_px, 4.0 * float(W * H))
    size = stroke_px(st, ss)
    spacing = max(size * 0.85, 1.0)
    need = area_px / (spacing * spacing)
    total = float(need.sum())
    if total > MAX_STROKES:
        need = need * (MAX_STROKES / total)
    n_t = np.ceil(need).astype(np.int64)
    n_t = np.minimum(n_t, 65535)
    # the sequences are the frame's; only the triangles this G-buffer
    # shows can have visible particles, so only they are expanded
    keep = (n_t > 0) & np.isin(tris, seen)
    if not keep.any():
        return None
    tris, need, n_t = tris[keep], need[keep], n_t[keep]
    P0, P1, P2 = P0[keep], P1[keep], P2[keep]
    # expand: particle k of triangle t
    rep = np.repeat(np.arange(tris.size), n_t)
    k = np.arange(rep.size) - np.repeat(np.cumsum(n_t) - n_t, n_t)
    t_id = tris[rep]
    r1 = _h(t_id, k, 1 + seed * 7)
    r2 = _h(t_id, k, 2 + seed * 7)
    s = np.sqrt(r1)
    b0 = 1.0 - s
    b1 = s * (1.0 - r2)
    b2 = s * r2
    P = (P0[rep] * b0[:, None] + P1[rep] * b1[:, None] + P2[rep] * b2[:, None])
    fade = np.clip(need[rep] - k, 0.0, 1.0).astype(np.float32)
    ph = np.concatenate([P, np.ones((P.shape[0], 1), np.float32)], 1)
    c = ph @ m.T
    w = c[:, 3]
    front = w > 1e-4
    ws = np.where(front, w, 1.0)
    sx = (c[:, 0] / ws * 0.5 + 0.5) * W
    sy = (c[:, 1] / ws * 0.5 + 0.5) * H
    inside = front & (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
    if not inside.any():
        return None
    px = np.clip(np.floor(sx), 0, W - 1).astype(np.int64)
    py = np.clip(np.floor(sy), 0, H - 1).astype(np.int64)
    # visibility: the pixel is a Background pixel, and the surface it
    # shows is no nearer than the particle (its own triangle, or a
    # same-object neighbour within one percent of the eye distance)
    vis = inside & bgm[py, px]
    if not vis.any():
        return None
    tp = np.where(vis, gbuf.tri[py, px], 0)
    same_tri = tp == t_id
    if mesh.obj_index is not None:
        same_obj = mesh.obj_index[tp] == mesh.obj_index[t_id]
    else:
        same_obj = np.ones(tp.shape, bool)
    tvp = mesh.tris[tp]
    bp = gbuf.bary[py, px]
    Ppix = (verts[tvp[:, 0]] * bp[:, 0:1] + verts[tvp[:, 1]] * bp[:, 1:2]
            + verts[tvp[:, 2]] * bp[:, 2:3])
    e = np.asarray(eye, np.float32)[None, :]
    dpix = np.linalg.norm(Ppix - e, axis=1)
    dpar = np.linalg.norm(P - e, axis=1)
    near_enough = dpar <= dpix * 1.01 + 1e-4
    vis &= same_tri | (same_obj & near_enough)
    if not vis.any():
        return None
    return {'sx': sx[vis].astype(np.float32), 'sy': sy[vis].astype(np.float32),
            'px': px[vis], 'py': py[vis], 'P': P[vis], 'tri': t_id[vis],
            'k': k[vis], 'fade': fade[vis], 'dist': dpar[vis].astype(np.float32),
            'tris': tris, 'n_t': n_t}


def _stroke_dirs(part, base, gbuf, scene, view, st, seed):
    """Per-particle stroke direction (cos, sin) in screen space."""
    mode = str(getattr(st, 'bg_direction', 'GRADIENT')).upper()
    spread = float(np.clip(getattr(st, 'bg_spread', 0.5), 0.0, 1.0))
    jitter = (_h(part['tri'], part['k'], 5 + seed * 7) - 0.5) * spread * np.pi * 0.5
    if mode == 'ANGLE':
        ang = np.radians(float(getattr(st, 'bg_angle', 20.0))) + jitter
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    if mode == 'NORMAL' and scene.mesh.face_normals is not None:
        N = scene.mesh.face_normals[part['tri']]
        vn = N @ np.asarray(view, np.float32)[:3, :3].T
        # along the silhouette: perpendicular to the normal's screen
        # direction; a face-on normal has none, so fall back to the angle
        gx, gy = vn[:, 0], vn[:, 1]
        base_ang = np.arctan2(gx, -gy)
        flat = np.hypot(gx, gy) < 1e-3
        base_ang = np.where(flat, np.radians(float(getattr(st, 'bg_angle', 20.0))),
                            base_ang)
        ang = base_ang + jitter
        return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    # GRADIENT: across the base colour's gradient (along its contours)
    lum = _luminance(base)
    gx = np.zeros_like(lum)
    gy = np.zeros_like(lum)
    gx[:, 1:-1] = lum[:, 2:] - lum[:, :-2]
    gy[1:-1, :] = lum[2:, :] - lum[:-2, :]
    sgx = FILM.gaussian(gx[:, :, None], 1.5)[:, :, 0]
    sgy = FILM.gaussian(gy[:, :, None], 1.5)[:, :, 0]
    px, py = part['px'], part['py']
    ggx = sgx[py, px]
    ggy = sgy[py, px]
    mag = np.hypot(ggx, ggy)
    base_ang = np.arctan2(ggx, -ggy)          # perpendicular to the gradient
    # a flat colour has no contour: the brush sweeps at Stroke Angle
    # there (the painter's default), turned only by Spread
    weak = mag < 1e-4
    base_ang = np.where(weak, np.radians(float(getattr(st, 'bg_angle', 20.0))),
                        base_ang)
    ang = base_ang + jitter
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def paint_strokes(img, scene, gbuf, bgm, base, part, st, ss, view, seed=0):
    """Draw the particles as strokes, far to near, over `img` in place.

    Each stroke is a rounded rectangle Length by Width in screen pixels
    (its own hash varying both), turned to its direction, bristle-
    streaked across on value noise, drying toward its end, its colour the
    abstracted base at its own pixel lightened or darkened by Variation.
    It paints only Background and sky pixels -- the cel in front is never
    touched -- and composites over with the road's coverage as its alpha.
    """
    H, W = bgm.shape
    cov = gbuf.tri >= 0
    allowed = bgm | ~cov
    amount = float(np.clip(getattr(st, 'bg_paint', 0.0), 0.0, 1.0))
    size = stroke_px(st, ss)
    ratio = float(max(getattr(st, 'bg_stroke_length', 3.0), 1.0))
    bristles = float(np.clip(getattr(st, 'bg_bristles', 0.6), 0.0, 1.0))
    variation = float(np.clip(getattr(st, 'bg_variation', 0.25), 0.0, 1.0))
    tri, k = part['tri'], part['k']
    h_len = _h(tri, k, 3 + seed * 7)
    h_wid = _h(tri, k, 4 + seed * 7)
    h_val = _h(tri, k, 7 + seed * 7)
    h_bri = _h(tri, k, 8 + seed * 7)
    hl = size * ratio * 0.5 * (0.7 + 0.6 * h_len)
    hw = size * 0.5 * (0.75 + 0.5 * h_wid)
    ca, sa = _stroke_dirs(part, base, gbuf, scene, view, st, seed)
    col = base[part['py'], part['px']] * (1.0 + variation * (h_val - 0.5))[:, None]
    alpha_p = amount * part['fade']
    # far to near, ties by triangle then index: an order-free rule
    order = np.lexsort((k, tri, -part['dist']))
    rgb = img[:, :, :3]
    for i in order:
        L = hl[i]
        Wd = hw[i]
        r = int(np.ceil(np.hypot(L, Wd))) + 1
        cx, cy = float(part['sx'][i]), float(part['sy'][i])
        x0 = max(int(np.floor(cx)) - r, 0)
        x1 = min(int(np.floor(cx)) + r + 1, W)
        y0 = max(int(np.floor(cy)) - r, 0)
        y1 = min(int(np.floor(cy)) + r + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5 - cx
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5 - cy
        ex = xs[None, :]
        ey = ys[:, None]
        al = ex * ca[i] + ey * sa[i]
        ac = -ex * sa[i] + ey * ca[i]
        wdry = Wd * (1.0 - 0.3 * np.clip(al / L, 0.0, 1.0))
        qa = al / L
        qc = ac / wdry
        q = (qa * qa) * (qa * qa) + (qc * qc) * (qc * qc)
        a = np.clip((1.0 - q) / 0.15, 0.0, 1.0)
        if bristles > 0.0:
            b = PT.value_noise2(np.stack([(ac / Wd * 4.5 + h_bri[i] * 64.0).ravel(),
                                          (al / L * 1.5 + h_val[i] * 64.0).ravel()],
                                         1).astype(np.float32)).reshape(a.shape)
            shade = 1.0 + bristles * 0.5 * (b - 0.5)
        else:
            shade = 1.0
        a = a * alpha_p[i] * allowed[y0:y1, x0:x1]
        if not a.any():
            continue
        c = col[i][None, None, :] * (shade[:, :, None] if bristles > 0.0 else 1.0)
        block = rgb[y0:y1, x0:x1]
        block += (c - block) * a[:, :, None]
    return img


# ---------------------------------------------------------------- setback


def setback(img, gbuf, scene, bgm, eye, st, ss):
    """The lens on the miniature: a softness by eye distance on the
    Background materials (and the sky), the cels untouched.

    Sigma grows from 0 at Setback Start to Setback over Setback Range,
    in output pixels; the sky sits at the far end. Three masked
    Gaussians (0, half, full) mixed per pixel by its own sigma -- a mip
    blend, band-invariant with the full kernel's reach.
    """
    amount = float(max(getattr(st, 'setback', 0.0), 0.0)) * float(ss)
    if amount <= 1e-3:
        return img
    cov = gbuf.tri >= 0
    region = bgm.copy()
    if bool(getattr(st, 'setback_sky', True)):
        region |= ~cov
    if not region.any():
        return img
    start = float(max(getattr(st, 'setback_start', 10.0), 0.0))
    rng = float(max(getattr(st, 'setback_range', 20.0), 1e-3))
    # eye distance per pixel from the G-buffer's own surface points
    mesh = scene.mesh
    safe = np.where(cov, gbuf.tri, 0)
    tv = mesh.tris[safe]
    b = gbuf.bary
    P = (mesh.verts[tv[..., 0]] * b[..., 0:1] + mesh.verts[tv[..., 1]] * b[..., 1:2]
         + mesh.verts[tv[..., 2]] * b[..., 2:3])
    dist = np.linalg.norm(P - np.asarray(eye, np.float32)[None, None, :], axis=-1)
    t = np.clip((dist - start) / rng, 0.0, 1.0)
    t = np.where(cov, t, 1.0).astype(np.float32)
    rgb = img[:, :, :3]
    half = _masked_blur(rgb, region, amount * 0.5)
    full = _masked_blur(rgb, region, amount)
    # mix: sigma < half -> between sharp and half; above -> half to full
    s = t * amount
    w1 = np.clip(s / (amount * 0.5), 0.0, 1.0)[:, :, None]
    w2 = np.clip((s - amount * 0.5) / (amount * 0.5), 0.0, 1.0)[:, :, None]
    mixed = rgb + (half - rgb) * w1
    mixed = mixed + (full - mixed) * w2
    out = img.copy()
    out[:, :, :3] = np.where(region[:, :, None], mixed, rgb)
    return out


# ------------------------------------------------------------------ apply


def apply(scene, gbuf, img, st, vp, view, eye, frame=0, seed=0, ss=1.0):
    """The painted background road on the frame, in place: the abstracted
    base, the strokes, the board's tooth, then the setback."""
    cov = gbuf.tri >= 0
    safe = np.where(cov, gbuf.tri, 0)
    lut = paint_lut(scene)
    mesh = scene.mesh
    if mesh.mat_index is not None:
        mi = np.where(cov, mesh.mat_index[safe], lut.size - 1)
    else:
        mi = np.where(cov, 0, lut.size - 1)
    bgm = lut[np.clip(mi, 0, lut.size - 1)] & cov
    out = np.asarray(img, np.float32).copy()
    if strokes_on(scene, st) and bgm.any():
        smooth = float(max(getattr(st, 'bg_smooth', 2.0), 0.0)) * float(ss)
        base = _masked_blur(out[:, :, :3], bgm, smooth)
        amount = float(np.clip(getattr(st, 'bg_paint', 0.0), 0.0, 1.0))
        # the abstracted base shows between strokes, by the road's amount
        out[:, :, :3] = np.where(bgm[:, :, None],
                                 out[:, :, :3] + (base - out[:, :, :3]) * amount,
                                 out[:, :, :3])
        part = seed_particles(scene, gbuf, bgm, vp, eye, st, ss, seed)
        if part is not None:
            out = paint_strokes(out, scene, gbuf, bgm, base, part, st, ss,
                                view, seed)
        paper = float(np.clip(getattr(st, 'bg_paper', 0.0), 0.0, 1.0))
        if paper > 0.0:
            H, W = bgm.shape
            yy, xx = np.nonzero(bgm)
            x = (xx.astype(np.float32) + 0.5) / np.float32(H) * np.float32(8.0)
            y = (yy.astype(np.float32) + 0.5) / np.float32(H) * np.float32(8.0)
            tooth = MD.paper(x, y, 0.7, 0.3, 0.4, MD.salt_for(seed, 0, 0))
            out[yy, xx, :3] *= (1.0 + paper * 0.8 * (tooth - 0.5))[:, None]
    if setback_on(scene, st):
        out = setback(out, gbuf, scene, bgm, eye, st, ss)
    return out
