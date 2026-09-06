"""The ink style pack -- R227 (1.70.0).

The cartoon outline pass (render.apply_outline) draws its line from a
boolean edge mask: the G-buffer's boundaries, dilated to an integer
width, painted one colour. That is a trace machine: every line the same
weight, crisp, mechanical -- the 1980s cel look, and exactly what an
80s anime wants. It is not what a 1940s inker's brush did, and the
field said so: "the ink is still a little limited in style options...
displacements (optionally animated), a pencily look option, gradients,
better scaling" -- and then "a 40s Looney Tunes cartoon is vastly
different to an 80s anime."

This module is the second ink road, opt-in by any style dial, and it
works from DISTANCE FIELDS instead of masks:

- `chamfer(seed)` is a (5,7,11) chamfer distance transform -- two
  row-vectorised passes, 0.07 s at 1080p, within 0.4 px of Euclidean
  inside 20 px -- with an optional nearest-seed FEATURE map, so a
  pixel on the sky side of a silhouette knows which surface pixel
  owns its line (that pixel's material width, colour, depth).
- The line's half-width is a PLANE, not a number: the material's
  width, scaled true to a reference frame height, tapered with
  distance (thick near, thin far -- the brush convention), thicker on
  the shadow side of the key lamp, and varied along the line by a
  low-frequency noise (the hand's pressure). Interior lines (creases,
  material breaks, marked edges) take their own scale -- the classic
  thick-outer / thin-inner drawing.
- Coverage is anti-aliased from the distance: a CLEAN line has a
  one-pixel transition (the xerox / trace-machine edge), a BRUSH line
  a soft bleeding edge, a PENCIL line is several offset strokes with
  per-pixel grain.
- BOIL displaces the finished line by a noise field that changes on
  a clock -- 12 steps a second is "on twos" at 24 fps, 8 is "on
  threes" -- the hand-traced wobble of cel animation, deterministic
  per (pixel, phase).
- Colour: the material's colour, the FILL colour darkened (the xerox
  and iro-trace self-coloured line), or a GRADIENT by depth, height
  or light.

Every noise here is an integer hash of (pixel, phase, seed): a pure
function of its inputs, so bands, workers, refine passes and both
devices draw the same line. The pass stays on the CPU on both device
roads (the outline doctrine), so no GLSL twin is needed.
"""

import numpy as np

INF = np.int32(1 << 28)

INK_STYLE_ITEMS = (
    ('CLEAN', "Clean",
     "A crisp, even line with a one-pixel anti-aliased edge -- the trace "
     "machine and the xerox: the 80s cel look"),
    ('BRUSH', "Brush",
     "A soft, bleeding edge, as ink from a brush on cel. Pair it with "
     "Taper and Weight Noise for the 1940s inker's thick-and-thin"),
    ('PENCIL', "Pencil",
     "Several offset strokes with per-pixel grain -- a sketched, "
     "construction-line look (Pencil Strokes / Spread / Grain shape it)"),
)

INK_COLOR_ITEMS = (
    ('FIXED', "Ink Colour",
     "The material's ink colour (or the global Ink Colour)"),
    ('FILL', "From Fill",
     "The line takes the colour of the surface it outlines, darkened by "
     "Fill Darken -- the self-coloured line of Disney's hand-inked "
     "features and the iro-trace of 80s anime"),
    ('GRADIENT', "Gradient",
     "Blends from the ink colour to Ink Colour 2 by the Gradient axis"),
)

INK_TEXTURE_ITEMS = (
    ('SOLID', "Solid", "Even ink through the whole line"),
    ('STREAKS', "Brush Streaks",
     "The brush's hairs: lighter streaks running ALONG the stroke, "
     "denser where the line thins -- ink on cel"),
    ('CHARCOAL', "Charcoal",
     "A soft, grainy line: the edge blooms wide and the body breaks "
     "into tooth, as a stick dragged over paper"),
)

INK_GRADIENT_ITEMS = (
    ('DEPTH', "Depth", "Ink colour nearest the camera, Ink Colour 2 farthest"),
    ('VERTICAL', "Vertical", "Ink colour at the bottom of the frame, Ink "
                             "Colour 2 at the top"),
    ('LIGHT', "Light", "Ink colour on the lit side of the key lamp, Ink "
                       "Colour 2 on the shadow side"),
)


# ------------------------------------------------------------ distance


def chamfer(seed, feature=False):
    """(5,7,11) chamfer distance transform of a boolean seed mask.

    Returns the distance in pixels (float32, H x W; 0 on seeds), and
    with `feature=True` also the flat index of the nearest seed pixel
    per pixel (int32; -1 where the mask is empty). Two passes, each
    row vectorised: the in-row propagation d[i] = min_j (d[j] + 5(i-j))
    is a prefix minimum of (d[j] - 5j), and its argmin is the last
    position where that prefix minimum was set -- so the feature rides
    the same accumulate. Exactly deterministic (integer arithmetic).
    """
    seed = np.asarray(seed, bool)
    H, W = seed.shape
    d = np.where(seed, np.int32(0), INF).astype(np.int32)
    if not seed.any():
        if feature:
            return (np.full((H, W), np.float32(INF / 5.0), np.float32),
                    np.full((H, W), -1, np.int32))
        return np.full((H, W), np.float32(INF / 5.0), np.float32)
    f = None
    if feature:
        flat = np.arange(H * W, dtype=np.int32).reshape(H, W)
        f = np.where(seed, flat, np.int32(-1)).astype(np.int32)
    idx = np.arange(W, dtype=np.int32) * 5
    ar = np.arange(W, dtype=np.int32)

    def shifted(r, off, add, fill):
        out = np.full(W, fill, np.int32)
        if off > 0:
            out[:-off] = r[off:] + add
        elif off < 0:
            out[-off:] = r[:off] + add
        else:
            out[:] = r + add
        return out

    def do_pass(order, sgn):
        for y in order:
            row = d[y]
            cands = [row]
            fcands = [f[y]] if feature else None
            y1 = y - sgn
            if 0 <= y1 < H:
                r1 = d[y1]
                for off, add in ((0, 5), (-1, 7), (1, 7), (-2, 11), (2, 11)):
                    cands.append(shifted(r1, off, add, INF))
                    if feature:
                        fcands.append(shifted(f[y1], off, 0, -1))
            y2 = y - 2 * sgn
            if 0 <= y2 < H:
                r2 = d[y2]
                for off, add in ((-1, 11), (1, 11)):
                    cands.append(shifted(r2, off, add, INF))
                    if feature:
                        fcands.append(shifted(f[y2], off, 0, -1))
            if len(cands) > 1:
                C = np.stack(cands)
                if feature:
                    k = np.argmin(C, axis=0)
                    row = C[k, ar]
                    frow = np.stack(fcands)[k, ar]
                else:
                    row = C.min(axis=0)
            elif feature:
                frow = fcands[0]
            if sgn > 0:
                v = row - idx
                acc = np.minimum.accumulate(v)
                row = acc + idx
                if feature:
                    src = np.maximum.accumulate(np.where(v == acc, ar, -1))
                    frow = frow[src]
            else:
                v = row[::-1] - idx
                acc = np.minimum.accumulate(v)
                row = (acc + idx)[::-1]
                if feature:
                    src = np.maximum.accumulate(np.where(v == acc, ar, -1))
                    frow = frow[::-1][src][::-1]
            d[y] = row
            if feature:
                f[y] = frow

    do_pass(range(H), 1)
    do_pass(range(H - 1, -1, -1), -1)
    dist = d.astype(np.float32) * np.float32(0.2)
    if feature:
        return dist, f
    return dist


# --------------------------------------------------------------- strokes


def _thin_4to8(seed):
    """Collapse a 4-connected contour (the boundary staircase every
    neighbour test produces) to an 8-connected one: a pixel with a seed
    to one side AND a seed below is redundant -- those two are diagonal
    neighbours of each other -- and drops. Two passes, one per
    staircase orientation (left-and-down, then right-and-down on what
    remains), so a 2x2 block thins to an L instead of a hole."""
    s = np.asarray(seed, bool).copy()
    for side in ('left', 'right'):
        horiz = np.zeros_like(s)
        if side == 'left':
            horiz[:, 1:] = s[:, :-1]
        else:
            horiz[:, :-1] = s[:, 1:]
        down = np.zeros_like(s)
        down[:-1, :] = s[1:, :]
        s = s & ~(horiz & down)
    return s


def _neighbours8(mask):
    """How many of each pixel's eight neighbours are set (int32)."""
    m = np.asarray(mask, bool).astype(np.int32)
    n = np.zeros_like(m)
    n[:, 1:] += m[:, :-1]
    n[:, :-1] += m[:, 1:]
    n[1:, :] += m[:-1, :]
    n[:-1, :] += m[1:, :]
    n[1:, 1:] += m[:-1, :-1]
    n[1:, :-1] += m[:-1, 1:]
    n[:-1, 1:] += m[1:, :-1]
    n[:-1, :-1] += m[1:, 1:]
    return n


def stroke_ends(seed):
    """Where strokes begin, end and meet: the endpoints (one neighbour)
    and junctions (three or more) of the thinned contour. A closed
    loop has none; a line that vanishes behind another object ends
    where it meets that object's contour -- exactly where an inker's
    brush lifted or landed."""
    thin = _thin_4to8(seed)
    n8 = _neighbours8(thin)
    return thin & ((n8 <= 1) | (n8 >= 3))


def tangent_at(dist, by, bx):
    """The unit tangent of the nearest line at pixels (by, bx), from
    the distance field's gradient (perpendicular to the line): central
    differences on the whole-frame field, gathered at the band."""
    H, W = dist.shape
    xl = np.clip(bx - 1, 0, W - 1)
    xr = np.clip(bx + 1, 0, W - 1)
    yd = np.clip(by - 1, 0, H - 1)
    yu = np.clip(by + 1, 0, H - 1)
    gx = dist[by, xr] - dist[by, xl]
    gy = dist[yu, bx] - dist[yd, bx]
    ln = np.sqrt(gx * gx + gy * gy)
    ok = ln > 1e-6
    tx = np.where(ok, -gy / np.where(ok, ln, 1.0), 1.0)
    ty = np.where(ok, gx / np.where(ok, ln, 1.0), 0.0)
    return tx.astype(np.float32), ty.astype(np.float32)


def fbm(x, y, k, octaves=3):
    """Three octaves of value noise, amplitudes halving, in [-1, 1]
    (normalised): the multi-scale irregularity of a real edge."""
    total = np.zeros_like(np.asarray(x, np.float32))
    amp = 1.0
    norm = 0.0
    for o in range(int(octaves)):
        total += amp * (2.0 * value_noise(x * (2.0 ** o) + 13.1 * o,
                                          y * (2.0 ** o) + 7.7 * o,
                                          k + 17 * o) - 1.0)
        norm += amp
        amp *= 0.5
    return (total / norm).astype(np.float32)


# ---------------------------------------------------------------- noise


def _hash_u32(ix, iy, k):
    """A 32-bit integer hash of (ix, iy, k) -> float32 in [0, 1)."""
    x = (np.atleast_1d(np.asarray(ix, np.int64)) & 0xffffffff).astype(np.uint32)
    y = (np.atleast_1d(np.asarray(iy, np.int64)) & 0xffffffff).astype(np.uint32)
    h = x * np.uint32(0x9E3779B1) ^ (y + np.uint32(0x85EBCA77)) \
        ^ np.uint32((int(k) * 0xC2B2AE3D) & 0xffffffff)
    h ^= h >> np.uint32(15)
    h *= np.uint32(0x2C1B3C6D)
    h ^= h >> np.uint32(12)
    h *= np.uint32(0x297A2D39)
    h ^= h >> np.uint32(15)
    return (h & np.uint32(0xffffff)).astype(np.float32) * np.float32(
        1.0 / 16777216.0)


def white_noise(h, w, k):
    """Per-pixel hash noise in [0, 1) for a frame, phase k."""
    yy, xx = np.mgrid[0:h, 0:w]
    return _hash_u32(xx, yy, k)


def value_noise(x, y, k):
    """Smooth lattice noise in [0, 1) at float coordinates (x, y), phase k
    -- bilinear over hashed lattice corners, smoothstep-eased."""
    x = np.asarray(x, np.float32)
    y = np.asarray(y, np.float32)
    x0 = np.floor(x)
    y0 = np.floor(y)
    tx = x - x0
    ty = y - y0
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    ix = x0.astype(np.int64)
    iy = y0.astype(np.int64)
    a = _hash_u32(ix, iy, k)
    b = _hash_u32(ix + 1, iy, k)
    c = _hash_u32(ix, iy + 1, k)
    d = _hash_u32(ix + 1, iy + 1, k)
    return ((a + (b - a) * tx) + ((c + (d - c) * tx) - (a + (b - a) * tx))
            * ty).astype(np.float32)


def bilinear(field, x, y):
    """Sample an (H, W[, C]) float field at float pixel coordinates,
    clamped to the frame."""
    H, W = field.shape[:2]
    x = np.clip(np.asarray(x, np.float32), 0.0, W - 1.0)
    y = np.clip(np.asarray(y, np.float32), 0.0, H - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    tx = (x - x0).astype(np.float32)
    ty = (y - y0).astype(np.float32)
    if field.ndim == 3:
        tx = tx[..., None]
        ty = ty[..., None]
    a = field[y0, x0]
    b = field[y0, x1]
    c = field[y1, x0]
    d = field[y1, x1]
    top = a + (b - a) * tx
    bot = c + (d - c) * tx
    return (top + (bot - top) * ty).astype(np.float32)


def boil_phase(time, fps):
    """The boil clock: which displacement field is showing at `time`
    (seconds). fps 12 changes the field twelve times a second -- on
    twos at 24 fps; 8 is on threes; 0 holds one field (static wobble)."""
    fps = int(fps or 0)
    if fps <= 0:
        return 0
    return int(np.floor(float(time) * fps + 1e-6))


def material_iro(graph):
    """R229: the Anime Shader's Line Colour menu, read from a serialized
    graph: (mode, colour, darken) -- mode 'INK' (the ink settings decide,
    the default and the answer for every other master), 'IRO' (the
    surface's own shaded colour darkened by `darken`, per pixel, on the
    distance-field road) or 'CUSTOM' (`colour`, the Line Color socket's
    own value -- a linked chain is not read: lines are inked per
    material, and the tooltip says so)."""
    if not graph:
        return ('INK', None, 0.0)
    nodes = graph.get('nodes') or {}
    out_id = graph.get('output')
    out = nodes.get(out_id) if out_id is not None else None
    node = None
    if out is not None:
        for sock in out.get('inputs', ()):
            if sock.get('name') == 'Surface' and sock.get('link'):
                node = nodes.get(sock['link'][0])
                break
    if node is None:
        # no output link recorded: the lone anime node, if there is one
        for nd in nodes.values():
            if nd.get('bl_idname') == 'HALCYON_AnimeShaderNode':
                node = nd
                break
    if node is None or node.get('bl_idname') != 'HALCYON_AnimeShaderNode':
        return ('INK', None, 0.0)
    mode = str((node.get('props') or {}).get('line_source', 'INK')).upper()
    if mode not in ('IRO', 'CUSTOM'):
        return ('INK', None, 0.0)
    col = (0.0, 0.0, 0.0)
    darken = 0.55
    for sock in node.get('inputs', ()):
        nm = sock.get('name')
        if nm == 'Line Color' and sock.get('default') is not None:
            d = sock['default']
            try:
                col = (float(d[0]), float(d[1]), float(d[2]))
            except (TypeError, IndexError, ValueError):
                pass
        elif nm == 'Line Darken' and sock.get('default') is not None:
            try:
                darken = float(sock['default'])
            except (TypeError, ValueError):
                pass
    return (mode, col, float(np.clip(darken, 0.0, 1.0)))


def scene_iro(scene):
    """R229: per-material (mode, colour, darken) for every material of a
    scene, from their graphs."""
    return [material_iro(getattr(m, 'graph', None))
            for m in (getattr(scene, 'materials', None) or [])]


def styled_on(st, scene=None):
    """True when any style dial leaves the mask roads -- the trigger of
    the distance-field road. Every default is the CLEAN, mask-identical
    setting, so an untouched scene never enters here. R229: an
    Iro-Trace material (its line takes the surface's own colour per
    pixel) needs the feature-tracking road too."""
    if scene is not None and any(m[0] == 'IRO' for m in scene_iro(scene)):
        return True
    # R239: the Guilty Gear line control reads per-pixel widths off the
    # vertex colours -- only the distance-field road can honour a
    # per-vertex width, so any controlled material routes here
    if scene is not None and any(
            str(getattr(m, 'ink_vc', 'OFF') or 'OFF').upper() != 'OFF'
            for m in (getattr(scene, 'materials', None) or ())):
        return True
    return (str(getattr(st, 'ink_style', 'CLEAN')).upper() != 'CLEAN'
            or float(getattr(st, 'ink_taper', 0.0)) != 0.0
            or float(getattr(st, 'ink_interior_scale', 1.0)) != 1.0
            or float(getattr(st, 'ink_weight_noise', 0.0)) != 0.0
            or float(getattr(st, 'ink_shadow_side', 0.0)) != 0.0
            or float(getattr(st, 'ink_boil', 0.0)) != 0.0
            or float(getattr(st, 'ink_grain', 0.0)) != 0.0
            or str(getattr(st, 'ink_color_mode', 'FIXED')).upper() != 'FIXED'
            or int(getattr(st, 'ink_reference_height', 0) or 0) > 0
            # R231: the drawn line's dials
            or float(getattr(st, 'ink_end_taper', 0.0)) != 0.0
            or float(getattr(st, 'ink_roughness', 0.0)) != 0.0
            or float(getattr(st, 'ink_drift', 0.0)) != 0.0
            or float(getattr(st, 'ink_gaps', 0.0)) != 0.0
            or str(getattr(st, 'ink_texture', 'SOLID')).upper() != 'SOLID'
            # R234: the inker's line -- the isophote weight, the stroke
            # road, the surface anchor
            or float(getattr(st, 'ink_isophote', 0.0)) != 0.0
            or float(getattr(st, 'ink_smooth', 0.0)) != 0.0
            or float(getattr(st, 'ink_pressure', 0.0)) != 0.0
            or float(getattr(st, 'ink_overshoot', 0.0)) != 0.0
            or str(getattr(st, 'ink_anchor', 'SCREEN')).upper() != 'SCREEN')


# --------------------------------------------------------------- the road


def _linear_depth(zndc, proj):
    """NDC z -> view distance along the axis, from the projection."""
    if proj is None:
        return np.asarray(zndc, np.float32)
    p = np.asarray(proj, np.float64)
    m22, m23, m32 = float(p[2, 2]), float(p[2, 3]), float(p[3, 2])
    z = np.asarray(zndc, np.float64)
    if abs(m32) > 1e-9:
        # perspective: z_ndc = (m22 z + m23) / (m32 z) -> z = m23 / (m32 z_ndc - m22)
        den = m32 * z - m22
        den = np.where(np.abs(den) < 1e-12, 1e-12, den)
        out = np.abs(m23 / den)
    else:
        # orthographic: z_ndc = m22 z + m23
        out = np.abs((z - m23) / (m22 if abs(m22) > 1e-12 else 1e-12))
    return out.astype(np.float32)


def apply(scene, gbuf, img, st, seeds, plane, vp=None, proj=None, eye=None):
    """The distance-field ink road.

    `seeds` = (sil, interior, md): the one-sided silhouette seed (the
    NEARER pixel of every object/depth/sky boundary whose material inks),
    the two-sided interior seed (material breaks, creases) and the
    marked-edge distance field (or None). `plane` = dict with `cov`,
    `pmat` (per-pixel material index, n_m for the sky), `on`, `off`,
    `width` (per-material width LUT in internal pixels, sky entry = the
    global), `color` (LUT), `g_color`, `opacity`, `over_sky`.

    Two distance transforms run whole-frame; everything after them runs
    on the BAND -- the pixels within the widest possible line plus the
    boil and pencil reach -- so the per-pixel work is a fraction of the
    frame (peak-performance rule: the style road costs a chamfer, not a
    second render).

    R234 (core/lines.py): the ISOPHOTE weight scales every silhouette
    seed's width by its walk to the light; the STROKE ROAD redraws the
    seeds as smoothed chains with pressure and overshoot, each seed
    pixel carrying its ORIGIN (the surface pixel the lookups read);
    the SURFACE anchor samples the hand's noises at the surface point
    under the line.
    """
    from . import lines as LN
    sil, inter, md = seeds
    cov = plane['cov']
    H, W = cov.shape
    pmat = plane['pmat']
    opacity = float(plane['opacity'])
    over_sky = bool(plane['over_sky'])
    if opacity <= 0.0:
        return img
    style = str(getattr(st, 'ink_style', 'CLEAN')).upper()
    rs = 1.0
    ref = int(getattr(st, 'ink_reference_height', 0) or 0)
    if ref > 0:
        rs = float(H) / float(ref)
    mesh = scene.mesh
    pmat_flat = pmat.reshape(-1)
    # R234: the inker's dials
    isophote = float(np.clip(getattr(st, 'ink_isophote', 0.0), 0.0, 1.0))
    iso_range = float(max(getattr(st, 'ink_isophote_range', 24.0), 1.0)) * rs
    shadow_level = float(np.clip(getattr(st, 'outline_shadow_level', 0.1),
                                 -1.0, 1.0))
    pressure = float(np.clip(getattr(st, 'ink_pressure', 0.0), 0.0, 1.0))
    smooth_px = float(max(getattr(st, 'ink_smooth', 0.0), 0.0)) * rs
    overshoot = float(max(getattr(st, 'ink_overshoot', 0.0), 0.0)) * rs
    anchor = str(getattr(st, 'ink_anchor', 'SCREEN')).upper()
    road = LN.road_on(st)

    def P_at(idx):
        """World positions at flat pixel indices (a subset gather)."""
        return LN.surface_attrs(mesh, gbuf, idx, want_n=False)[0]

    taper = float(np.clip(getattr(st, 'ink_taper', 0.0), 0.0, 1.0))
    mode = str(getattr(st, 'ink_color_mode', 'FIXED')).upper()
    axis = str(getattr(st, 'ink_gradient', 'DEPTH')).upper()
    shadow_side = float(np.clip(getattr(st, 'ink_shadow_side', 0.0), 0.0, 1.0))
    wnoise = float(np.clip(getattr(st, 'ink_weight_noise', 0.0), 0.0, 1.0))
    inner = float(max(getattr(st, 'ink_interior_scale', 1.0), 0.0))
    time = float(getattr(scene, 'time', 0.0) or 0.0)
    boil = float(max(getattr(st, 'ink_boil', 0.0), 0.0))
    phase = boil_phase(time, getattr(st, 'ink_boil_fps', 12)) if boil > 0.0 \
        else 0
    seed_salt = int(getattr(st, 'seed', 0) or 0) * 7919
    spread = float(max(getattr(st, 'ink_pencil_spread', 1.5), 0.0)) \
        if style == 'PENCIL' else 0.0
    # R231: the drawn line -- stroke ends, roughness, drift, gaps, texture
    end_taper = float(np.clip(getattr(st, 'ink_end_taper', 0.0), 0.0, 1.0))
    end_len = float(max(getattr(st, 'ink_end_length', 12.0), 1.0)) * rs
    rough = float(np.clip(getattr(st, 'ink_roughness', 0.0), 0.0, 1.0))
    rough_scale = max(float(getattr(st, 'ink_roughness_scale', 6.0)), 1.0) * rs
    drift = float(max(getattr(st, 'ink_drift', 0.0), 0.0)) * rs
    gaps = float(np.clip(getattr(st, 'ink_gaps', 0.0), 0.0, 1.0))
    texture = str(getattr(st, 'ink_texture', 'SOLID')).upper()
    tex_amt = float(np.clip(getattr(st, 'ink_texture_amount', 0.6), 0.0, 1.0))

    # the mask road draws width w as 2w-1 pixels (a seed dilated w-1
    # each way); the same w here is the half-width w-0.5, so a style
    # switch never thins a line -- and width 1 becomes a true one-pixel
    # line instead of the two-sided seed
    w_lut = np.maximum(np.asarray(plane['width'], np.float32)
                       - np.float32(0.5), np.float32(0.5)) * np.float32(rs)
    # R239: the Guilty Gear line control's width plane (vertex ALPHA
    # times two: 0.5 paints the width as set, 1 doubles it)
    vc_a = plane.get('vc_a')
    h_max = float(w_lut.max()) * (1.0 + taper) * (1.0 + shadow_side) \
        * (1.0 + wnoise) * (1.0 + rough) * (1.0 + 1.5 * pressure) + drift
    if vc_a is not None:
        h_max = h_max * 2.0
    if texture == 'CHARCOAL':
        h_max = h_max * 1.1 + 1.0

    # ---- per-source quantities: depth (taper, gradient) and N.L
    need_depth = taper > 0.0 or (mode == 'GRADIENT' and axis == 'DEPTH') \
        or anchor == 'SURFACE'
    tdep_flat = None
    depth_flat = None
    if need_depth:
        zz = np.where(cov, gbuf.depth, 0.0).astype(np.float32)
        depth = _linear_depth(zz, proj)
        depth_flat = depth.reshape(-1)
        dmin, dmax = _scene_depth_range(mesh, vp, proj, depth, cov)
        tdep = np.clip((depth - dmin) / max(dmax - dmin, 1e-6), 0.0, 1.0)
        tdep_flat = tdep.astype(np.float32).reshape(-1)
    need_ndl = shadow_side > 0.0 or (mode == 'GRADIENT' and axis == 'LIGHT')
    ndl_flat = _key_ndl(scene, mesh, gbuf, cov) if need_ndl else None

    def half_at(src):
        """The half-width owned by the source pixels `src` (flat)."""
        h = w_lut[pmat_flat[src]]
        if vc_a is not None:
            # R239: the painted width -- the ASW convention's alpha,
            # doubled so 0.5 is the width as set
            h = h * vc_a[src]
        if taper > 0.0:
            h = h * (1.0 + taper * (1.0 - 2.0 * tdep_flat[src]))
        if shadow_side > 0.0:
            h = h * (1.0 + shadow_side * (1.0 - ndl_flat[src]))
        return h.astype(np.float32)

    # ---- R234: the isophote weight, measured on the surface seeds
    iso_fac = None
    if isophote > 0.0 and sil.any():
        obj_i = mesh.obj_index
        omap_i = np.where(cov, obj_i[np.where(cov, gbuf.tri, 0)], -1) \
            if obj_i is not None else np.where(cov, 0, -1)
        dmap_i = np.where(cov, gbuf.depth, 1e12).astype(np.float32)
        eye_i = np.asarray(eye, np.float32) if eye is not None else \
            _eye_from_vp(vp)
        d_iso = LN.isophote_distance(
            sil, scene, mesh, gbuf, omap_i, dmap_i, cov, vp, eye_i,
            shadow_level, iso_range,
            depth_thr=max(float(getattr(st, 'outline_depth_threshold',
                                        0.02)), 1e-5))
        iso_fac = LN.isophote_factor(d_iso, iso_range, isophote)

    # ---- R234: the stroke road -- the seeds redrawn as smoothed chains
    org_sil = None
    w_sil = None
    w_int = None
    if road:
        sil, org_sil, w_sil = LN.stroke_road(
            sil, P_at, rs, smooth_px, pressure, overshoot,
            seed_salt + 7)
        if inter.any():
            inter, _org_int, w_int = LN.stroke_road(
                inter, P_at, rs, smooth_px, pressure, overshoot,
                seed_salt + 13)

    # ---- distances (whole frame), then the band
    dsil, feat = chamfer(sil, feature=True)
    has_sil = feat >= 0
    dint = None
    fint = None
    if inter.any():
        if road:
            dint, fint = chamfer(inter, feature=True)
        else:
            dint = chamfer(inter)
    dend = None
    if end_taper > 0.0:
        # R231: where the strokes begin, end and meet -- the brush
        # lifts and lands there, so the line thins toward those points
        ends = stroke_ends(sil)
        dend = chamfer(ends) if ends.any() else None
    margin = boil + spread + 1.5
    band = has_sil & (dsil < h_max + margin)
    if dint is not None:
        band |= dint < h_max * inner + margin
    if md is not None:
        band |= md < h_max * inner + margin
    band &= ~plane['off']
    if not over_sky:
        band &= cov
    if not band.any():
        return img
    by, bx = np.nonzero(band)
    n = by.size
    bflat = (by * W + bx).astype(np.int64)
    feat_b = feat[by, bx].astype(np.int64)
    has_b = feat_b >= 0
    feat_b = np.where(has_b, feat_b, bflat)
    # the ORIGIN: the surface pixel every lookup reads -- the seed
    # itself off the stroke road, the pixel the redrawn seed came from
    # on it
    if org_sil is not None:
        orig_b = org_sil[feat_b]
        orig_b = np.where(orig_b >= 0, orig_b, feat_b)
    else:
        orig_b = feat_b

    # inside: the pixel belongs to the owner's object (the line's inner
    # half); outside: the far side -- sky or the object behind
    obj = mesh.obj_index
    if obj is not None:
        omap = np.where(cov, obj[np.where(cov, gbuf.tri, 0)], -1).reshape(-1)
        inside_b = cov[by, bx] & (omap[bflat] == omap[orig_b])
    else:
        inside_b = cov[by, bx]
    half_sil = half_at(orig_b)
    if iso_fac is not None:
        half_sil = (half_sil * iso_fac[orig_b]).astype(np.float32)
    if w_sil is not None:
        half_sil = (half_sil * w_sil[feat_b]).astype(np.float32)
    d_b = dsil[by, bx]

    # R234: the SURFACE anchor -- the hand's noises read the surface
    # point under the line (the nearest seed's origin; the pixel's own
    # surface point off any line), scaled so the screen dial keeps
    # its meaning at that depth; three slices of the lattice noise
    # make a 3-D field of it
    if anchor == 'SURFACE' and proj is not None:
        a_src = orig_b.copy()
        if dint is not None and fint is not None:
            fi = fint[by, bx].astype(np.int64)
            nearer = (fi >= 0) & (dint[by, bx] < d_b)
            a_src = np.where(nearer, fi, a_src)
        elif dint is not None:
            a_src = np.where(cov[by, bx] & (dint[by, bx] < d_b), bflat, a_src)
        Pa = P_at(a_src)
        pxw = LN.pixel_world_size(proj, depth_flat[a_src], H)

        def hand_noise(scale_px, ox, oy, k):
            s = np.maximum(pxw * np.float32(scale_px), np.float32(1e-9))
            u = Pa[:, 0] / s
            v = Pa[:, 1] / s
            w = Pa[:, 2] / s
            nz = (value_noise(u + ox, v + oy, k)
                  + value_noise(v + ox * 0.7, w + oy * 1.3, k + 3)
                  + value_noise(w + ox * 1.1, u + oy * 0.6, k + 5)) \
                * np.float32(1.0 / 3.0)
            return np.clip((nz - 0.5) * np.float32(1.7) + 0.5, 0.0, 1.0) \
                .astype(np.float32)

        def hand_fbm(scale_px, k):
            s = np.maximum(pxw * np.float32(scale_px), np.float32(1e-9))
            u = Pa[:, 0] / s
            v = Pa[:, 1] / s
            w = Pa[:, 2] / s
            f = (fbm(u, v, k) + fbm(v, w, k + 3) + fbm(w, u, k + 5)) \
                * np.float32(1.0 / 3.0)
            return np.clip(f * np.float32(1.7), -1.0, 1.0).astype(np.float32)
    else:
        def hand_noise(scale_px, ox, oy, k):
            return value_noise(bx / scale_px + ox, by / scale_px + oy, k)

        def hand_fbm(scale_px, k):
            return fbm(bx / scale_px, by / scale_px, k)

    if drift > 0.0:
        # R231: the hand drifts -- the line's centre wanders in and out
        # of the true contour by a slow noise (the weight scale), the
        # same field on both sides so the line stays one stroke
        ws_d = max(float(getattr(st, 'ink_weight_scale', 24.0)), 1.0)
        dn = drift * (2.0 * hand_noise(ws_d, 5.3, 9.1,
                                       phase * 3 + 23 + seed_salt) - 1.0)
        d_b = np.where(inside_b, d_b + dn, d_b - dn)
    db_sil = np.where(inside_b, d_b + 0.5, d_b - 0.5)
    cov_sil = np.where(has_b, half_sil + 0.5 - db_sil, -1.0)
    half_self = half_at(bflat)
    if w_int is not None and fint is not None:
        fi = fint[by, bx].astype(np.int64)
        half_int = (half_self * np.where(fi >= 0, w_int[np.maximum(fi, 0)],
                                         1.0)).astype(np.float32)
    else:
        half_int = half_self
    wf = None
    if wnoise > 0.0:
        ws = max(float(getattr(st, 'ink_weight_scale', 24.0)), 1.0)
        nz = hand_noise(ws, 0.0, 0.0, phase * 3 + 11 + seed_salt)
        wf = (1.0 + wnoise * (2.0 * nz - 1.0)).astype(np.float32)
    if dend is not None:
        # R231: stroke ends -- the half-width falls toward every end
        # and junction over End Length, never below 15 percent
        te = np.clip(dend[by, bx] / np.float32(end_len), 0.0, 1.0)
        te = te * te * (3.0 - 2.0 * te)
        ef = (1.0 - end_taper * 0.85 * (1.0 - te)).astype(np.float32)
        wf = ef if wf is None else wf * ef
    if rough > 0.0:
        # R231: roughness -- three octaves of edge irregularity, in
        # PATCHES (a slow noise gates where the line gets rough), so the
        # line is occasionally rough rather than uniformly hairy
        patch = hand_noise(rough_scale * 7.0, 3.7, 1.9,
                           phase * 3 + 31 + seed_salt)
        patch = np.clip((patch - 0.25) / 0.5, 0.0, 1.0)
        rf = hand_fbm(rough_scale, phase * 3 + 41 + seed_salt)
        rr = (1.0 + rough * patch * rf).astype(np.float32)
        wf = rr if wf is None else wf * rr
    if wf is not None:
        half_sil = half_sil * wf
        half_self = half_self * wf
        half_int = half_int * wf
        cov_sil = np.where(has_b, half_sil + 0.5 - db_sil, -1.0)
    cov_int = None
    if dint is not None:
        if road:
            # the redrawn interior chain is one-sided like every other
            # chain seed: the seed pixel is the line's centre
            cov_int = half_int * inner + 0.5 - dint[by, bx]
        else:
            cov_int = half_int * inner - dint[by, bx]     # two-sided seed
    cov_md = None
    if md is not None:
        cov_md = half_self * inner + 0.5 - md[by, bx]

    def profile(c, h):
        if c is None:
            return None
        if texture == 'CHARCOAL':
            # R231: the stick's bloom -- a soft edge wider than the
            # brush's around a dark core, then the tooth below breaks
            # the body
            tw = np.maximum(np.float32(1.5), h * np.float32(1.1))
            t = np.clip((c + tw * 0.5) / tw, 0.0, 1.0)
            return (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        if style == 'BRUSH':
            # a soft edge: the transition widens with the line
            tw = np.maximum(np.float32(1.0), h * np.float32(0.8))
            t = np.clip((c + tw * 0.5) / tw, 0.0, 1.0)
            return (t * t * (3.0 - 2.0 * t)).astype(np.float32)
        return np.clip(c, 0.0, 1.0).astype(np.float32)

    C = profile(cov_sil, half_sil)
    for extra in (profile(cov_int, half_int * inner),
                  profile(cov_md, half_self * inner)):
        if extra is not None:
            C = np.maximum(C, extra)
    if gaps > 0.0:
        # R231: dry-brush skips -- short breaks (a few pixels) where a
        # fine noise runs high, more readily where the stroke is thin;
        # the skip is never a missing edge, only a lifted hair of it
        gs = 4.0 * rs
        gn = hand_noise(gs, 11.3, 4.7, phase * 3 + 53 + seed_salt)
        thin = 1.0 - np.clip(half_sil / np.maximum(w_lut.max(), 1e-6),
                             0.0, 1.0)
        g0 = 1.0 - gaps * (0.45 + 0.25 * thin)
        cut = np.clip((gn - g0) / 0.06, 0.0, 1.0)
        C = C * (1.0 - cut * cut * (3.0 - 2.0 * cut))
    if texture == 'STREAKS' and tex_amt > 0.0:
        # R231: the brush's hairs -- streaks along the stroke: noise
        # stretched along the local tangent (20 px) and pinched across
        # it (1.3 px), lightening the body of the line
        tx, ty = tangent_at(dsil, by, bx)
        # a canonical tangent (the gradient flips sign across the seed,
        # the tangent must not), the ALONG coordinate from the source
        # pixel (every pixel of one cross-section shares it) and the
        # ACROSS coordinate from the signed distance: the streaks run
        # parallel to the line at fixed offsets, unbroken at its centre
        flip = (tx < 0.0) | ((tx == 0.0) & (ty < 0.0))
        tx = np.where(flip, -tx, tx)
        ty = np.where(flip, -ty, ty)
        fx = (feat_b % W).astype(np.float32)
        fy = (feat_b // W).astype(np.float32)
        across = np.maximum(np.float32(1.0), half_sil * np.float32(0.4))
        u = (fx * tx + fy * ty) / (26.0 * rs)
        v = np.where(inside_b, d_b, -d_b) / across
        sn = value_noise(u + 2.1, v + 6.3, phase * 3 + 61 + seed_salt)
        streak = np.clip((sn - 0.55) / 0.3, 0.0, 1.0)
        streak = streak * streak * (3.0 - 2.0 * streak)
        C = C * (1.0 - tex_amt * 0.55 * streak)
    elif texture == 'CHARCOAL' and tex_amt > 0.0:
        # R231: the paper's tooth under the stick -- a fine grain and a
        # coarser blotch, both by the pixel's own hash
        tooth = _hash_u32(bx, by, phase * 5 + 71 + seed_salt)
        blotch = value_noise(bx / (2.5 * rs), by / (2.5 * rs),
                             phase * 3 + 73 + seed_salt)
        C = C * (1.0 - tex_amt * (0.25 * tooth + 0.35 * blotch))

    # ---- the colour at the line's source pixel
    src = np.where(has_b & ~inside_b, orig_b, bflat)
    col_lut = np.asarray(plane['color'], np.float32)
    K = col_lut[pmat_flat[src]]
    if mode == 'FILL':
        darken = float(np.clip(getattr(st, 'ink_fill_darken', 0.45), 0.0, 1.0))
        fill = img[:, :, :3].reshape(-1, 3)[src]
        K = (fill * np.float32(1.0 - darken)).astype(np.float32)
    elif mode == 'GRADIENT':
        c2 = np.asarray(getattr(st, 'ink_color2', (0.35, 0.1, 0.45)),
                        np.float32)
        if axis == 'VERTICAL':
            t = by.astype(np.float32) / max(H - 1, 1)
        elif axis == 'LIGHT':
            t = 1.0 - ndl_flat[src]
        else:
            t = tdep_flat[src]
        K = (K + (c2[None, :] - K) * t[:, None]).astype(np.float32)
    iro = plane.get('iro')
    if iro is not None:
        # R229: an Iro-Trace material's line takes the surface's own
        # shaded colour at the source pixel, darkened by its own
        # amount -- per material, over whatever the global mode chose
        iro_lut = np.asarray(iro, np.float32)
        dk = iro_lut[pmat_flat[src]]
        on = dk >= 0.0
        if np.any(on):
            fill = img[:, :, :3].reshape(-1, 3)[src]
            K = np.where(on[:, None],
                         fill * np.maximum(1.0 - dk, 0.0)[:, None], K) \
                .astype(np.float32)

    # ---- the styles that move the finished line: boil, pencil, grain
    need_warp = boil > 0.0 or style == 'PENCIL'
    if need_warp:
        C_full = np.zeros((H, W), np.float32)
        C_full[by, bx] = C
        K_full = np.zeros((H, W, 3), np.float32)
        K_full[by, bx] = K
        if boil > 0.0:
            bs = max(float(getattr(st, 'ink_boil_scale', 18.0)), 1.0)
            ux = boil * (2.0 * value_noise(bx / bs, by / bs,
                                           phase * 2 + 101 + seed_salt) - 1.0)
            uy = boil * (2.0 * value_noise(bx / bs + 31.7, by / bs + 17.3,
                                           phase * 2 + 102 + seed_salt) - 1.0)
        else:
            ux = np.zeros(n, np.float32)
            uy = np.zeros(n, np.float32)
        if style == 'PENCIL':
            strokes = int(np.clip(getattr(st, 'ink_pencil_strokes', 3), 1, 6))
            acc = np.zeros(n, np.float32)
            Ksum = np.zeros((n, 3), np.float32)
            wsum = np.zeros(n, np.float32)
            for k in range(strokes):
                ox = spread * (2.0 * float(_hash_u32(k, phase,
                                                     7 + seed_salt)[0]) - 1.0)
                oy = spread * (2.0 * float(_hash_u32(k + 17, phase,
                                                     9 + seed_salt)[0]) - 1.0)
                sx = bx + ox + ux
                sy = by + oy + uy
                ck = bilinear_at(C_full, sx, sy) * np.float32(0.72)
                acc = 1.0 - (1.0 - acc) * (1.0 - ck)
                Ksum += bilinear_at(K_full, sx, sy) * ck[:, None]
                wsum += ck
            C = acc.astype(np.float32)
            K = np.where(wsum[:, None] > 1e-6,
                         Ksum / np.maximum(wsum, 1e-6)[:, None], K)
        else:
            C = bilinear_at(C_full, bx + ux, by + uy)
            K = bilinear_at(K_full, bx + ux, by + uy)
    grain = float(np.clip(getattr(st, 'ink_grain', 0.0), 0.0, 1.0))
    if grain > 0.0:
        C = C * (1.0 - grain * _hash_u32(bx, by, phase * 5 + 3 + seed_salt))
    C = np.clip(C, 0.0, 1.0).astype(np.float32)

    a = (C * np.float32(opacity))[:, None]
    img[by, bx, :3] = img[by, bx, :3] * (1.0 - a) + K * a
    if over_sky:
        img[by, bx, 3] = np.maximum(img[by, bx, 3], a[:, 0])
    return img


def bilinear_at(field, x, y):
    """Sample an (H, W[, C]) field at float coordinates given as 1-D
    arrays, clamped to the frame."""
    H, W = field.shape[:2]
    x = np.clip(np.asarray(x, np.float32), 0.0, W - 1.0)
    y = np.clip(np.asarray(y, np.float32), 0.0, H - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, W - 1)
    y1 = np.minimum(y0 + 1, H - 1)
    tx = (x - x0).astype(np.float32)
    ty = (y - y0).astype(np.float32)
    if field.ndim == 3:
        tx = tx[:, None]
        ty = ty[:, None]
        a = field[y0, x0]
        b = field[y0, x1]
        c = field[y1, x0]
        d = field[y1, x1]
    else:
        # flat gathers: the same values as field[y, x], without the
        # 2-D fancy-index bookkeeping (R234: the tone lines' flow run
        # made this the hot loop; a channelled field gathers faster
        # the 2-D way)
        flat = field.reshape(-1)
        r0 = y0 * W
        r1 = y1 * W
        a = flat[r0 + x0]
        b = flat[r0 + x1]
        c = flat[r1 + x0]
        d = flat[r1 + x1]
    top = a + (b - a) * tx
    bot = c + (d - c) * tx
    return (top + (bot - top) * ty).astype(np.float32)


def _scene_depth_range(mesh, vp, proj, depth, cov):
    """The near and far of the taper: the MESH's own depth extent under
    this camera (every vertex in front of the near plane), never the
    frame's covered pixels -- a pooled band sees only its rows, and a
    range read off the pixels would taper each band differently."""
    try:
        verts = np.asarray(mesh.verts, np.float32)
        m = np.asarray(vp, np.float32)
        clip = verts @ m[:3, :3].T + m[:3, 3][None, :]
        w = verts @ m[3, :3] + m[3, 3]
        ok = w > 1e-5
        if ok.any() and vp is not None:
            zn = clip[ok, 2] / w[ok]
            d = _linear_depth(zn.astype(np.float32), proj)
            return float(d.min()), float(d.max())
    except Exception:                                           # noqa: BLE001
        pass
    if cov.any():
        return float(depth[cov].min()), float(depth[cov].max())
    return 0.0, 1.0


def _key_ndl(scene, mesh, gbuf, cov):
    """N.L of the key lamp per pixel (flat, clipped to [0, 1]), 1
    off-surface. R234: the smooth normal the shading read and the
    lamp's true direction per pixel (a point lamp's rotation used to be
    read as its direction, which means nothing) -- core/lines.py."""
    from . import lines as LN
    ndl = LN.ndl_field(scene, mesh, gbuf, cov)
    return np.where(cov.reshape(-1), np.clip(ndl, 0.0, 1.0), 1.0).astype(
        np.float32)


def _eye_from_vp(vp):
    """The camera position from a view-projection matrix: the point
    that maps to clip w = 0 and (x, y) = 0 -- the centre of projection.
    Solves the affine part; falls back to the origin."""
    try:
        m = np.asarray(vp, np.float64)
        # the eye satisfies m[3] . (e, 1) = 0 and m[0..1] . (e, 1) = 0
        A = np.stack([m[0, :3], m[1, :3], m[3, :3]])
        b = -np.array([m[0, 3], m[1, 3], m[3, 3]])
        return np.linalg.solve(A, b).astype(np.float32)
    except Exception:                                           # noqa: BLE001
        return np.zeros(3, np.float32)
