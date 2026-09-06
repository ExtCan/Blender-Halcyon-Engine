"""The 2D media: the hand's marks as textures (R232).

Hatching, pencil scribble, stipple, charcoal, brush strokes, wash and
paper -- the marks a 1940s ink-and-paint department, a storyboard artist
or a background painter made, as fields a material can read. Each is a
pure function of a 2D coordinate, a TONE (lightness, 1 = bare paper,
0 = solid dark -- what a Shader to RGB luminance hands over directly),
a few dials and an integer SALT, and each has a GLSL twin in
gpu/procedural.py written line for line in the same operation order, so
both devices draw the same marks.

The salt is where the boil lives: a drawing that is redrawn every N frames
reseeds through it, and a per-node seed keeps two materials' hatching from
being the same hatching. Every random here is the integer hash the rest of
the pattern library rides -- no sin-fract, no generator state -- which is
what lets the marks travel to a driver exactly.

Conventions, shared with patterns.py: float32 in, float32 out, and every
scalar derived from a dial is derived in float32 in the SAME operation
order as the GLSL (a dial that reaches a `floor` -- the charcoal's streak
stretch, the lattice scales -- cannot afford a double-rounded Python
float). Angles that reach a `floor` (the hatching and scribble lanes, the
charcoal's streak axis, the brush direction) are baked to float32 cos/sin
by the CALLER on both devices, so no driver's cos decides which lane a
pixel is in.
"""

import numpy as np

from . import patterns as PT

f32 = np.float32

#: hash channels per salt: the salt occupies the lattice's z axis in
#: blocks of 64, so a pattern may use channels 0..63 freely
SALT_STRIDE = 64

#: the scribble's layer fan, degrees: the shallow crossing of a pencil
#: going back and forth (the hatching fans over 180)
SCRIBBLE_FAN = 70.0


def salt_for(seed, frame, boil):
    """The 16-bit salt of a node: its seed folded with the boil key.

    `boil` frames per redraw (0 = a still drawing): key = frame // boil.
    Integer arithmetic only, so the GLSL twin (`int` division, mask)
    reaches the same number on every driver.
    """
    b = int(boil)
    key = (max(int(frame), 0) // b) if b > 0 else 0
    return (int(seed) * 7919 + key * 104729) & 0xffff


def _n2(x, y, salt, k):
    """Bilinear value noise on the (salt, channel) plane."""
    return PT.value_noise2s(np.stack([x, y], 1).astype(np.float32),
                            int(salt) * SALT_STRIDE + int(k))


def _h(ix, iy, salt, k):
    """A per-cell hash on the (salt, channel) plane."""
    return PT.hash3(ix, iy, np.int64(int(salt) * SALT_STRIDE + int(k)))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def clamp01(v):
    return f32(min(max(float(v), 0.0), 1.0))


def rotation(angle_deg, k=0, layers=1, fan=180.0):
    """(cos, sin) as float32 for layer k of `layers` fanned over `fan` deg.

    Baked by the caller and emitted as literals by the GPU twin.
    """
    th = np.radians(float(angle_deg)
                    + float(k) * float(fan) / max(int(layers), 1))
    return np.float32(np.cos(th)), np.float32(np.sin(th))


def darkness(tone):
    return np.clip(1.0 - np.asarray(tone, np.float32), 0.0, 1.0) \
        .astype(np.float32)


def view_coords(dx, dy, dz):
    """The View space of the media (R233): the direction from the eye to
    the point, octahedrally unfolded onto the unit square.

    Lucas Pope's answer to the shower door in Return of the Obra Dinn:
    map the pattern onto a sphere around the camera, oriented to the
    WORLD, so turning the camera leaves the marks fixed on the scene and
    only a dolly swims them. The direction is normalised by its L1 length
    (abs, add, divide -- no sqrt, so both devices reach the same bits
    from the same P and eye), then the lower hemisphere folds into the
    square's corners: continuous everywhere but the nadir, and periodic
    across the square's edges by the fold. Returns (u, v) in [-1, 1].
    """
    ax = np.abs(dx)
    ay = np.abs(dy)
    az = np.abs(dz)
    s = np.maximum(ax + ay + az, 1e-12)
    u = dx / s
    v = dy / s
    au = np.abs(u)
    av = np.abs(v)
    fu = (1.0 - av) * np.sign(u)
    fv = (1.0 - au) * np.sign(v)
    lower = dz < 0.0
    return (np.where(lower, fu, u).astype(np.float32),
            np.where(lower, fv, v).astype(np.float32))


# ------------------------------------------------------- the direction field

#: the direction field's bins: twelve over 180 degrees. A hatching per bin
#: is one coherent lane field (R235); a pixel between two bins cross-fades
#: them, so a direction that turns across a form never breaks a lane
DIRECTION_BINS = 12

DIRECTION_ITEMS = (
    ('ANGLE', "Angle", "Every stroke at the Angle: the flat, mechanical hatch"),
    ('FORM', "Form",
     "Strokes wrap the form: along the silhouette-parallel tangent (the "
     "normal crossed with the view), the isophote direction of the "
     "classic pen drawing, plus the Angle"),
    ('SLOPE', "Slope",
     "Strokes run down the form: along the tangent that turns away from "
     "the eye fastest (the view projected onto the surface), plus the "
     "Angle -- radial about a sphere's centre"),
)


def tangent_2d(N, V, mode, screen, up_frame=True):
    """The per-pixel stroke direction of a FORM or SLOPE hatch, as a 2-D
    vector in the medium's domain. FORM = N x V (the tangent parallel to
    the silhouette), SLOPE = V projected onto the tangent plane; unit
    length, zero where degenerate (a face-on point's slope). On the
    camera's paper (`screen`) the tangent is expressed in the matcap
    frame -- right = up x V, up' = V x right, the frame the Matcap
    Coordinates node builds -- else its world x and y are read straight
    off (the generated coordinates' own axes). Same float32 ops as the
    GLSL twin."""
    N = np.asarray(N, np.float32)
    V = np.asarray(V, np.float32)
    if mode == 'FORM':
        t3 = np.cross(N, V).astype(np.float32)
    else:
        ndv = (N * V).sum(1, keepdims=True).astype(np.float32)
        t3 = (V - N * ndv).astype(np.float32)
    tl = np.sqrt((t3 * t3).sum(1)).astype(np.float32)
    ok = tl > np.float32(1e-6)
    t3 = np.where(ok[:, None], t3 / np.where(ok, tl, 1.0)[:, None], 0.0) \
        .astype(np.float32)
    if screen:
        up = np.array([0.0, 0.0, 1.0], np.float32)
        r0 = np.cross(np.broadcast_to(up[None, :], V.shape), V).astype(
            np.float32)
        rl = (r0 * r0).sum(1)
        deg = rl < np.float32(1e-8)
        right = np.where(deg[:, None], np.array([1.0, 0.0, 0.0], np.float32),
                         r0 / np.sqrt(np.where(deg, 1.0, rl))[:, None]) \
            .astype(np.float32)
        upv = np.cross(V, right).astype(np.float32)
        tx = (t3 * right).sum(1).astype(np.float32)
        ty = (t3 * upv).sum(1).astype(np.float32)
        return tx, ty
    return (np.ascontiguousarray(t3[:, 0], np.float32),
            np.ascontiguousarray(t3[:, 1], np.float32))


def direction_bins(tx, ty):
    """The bin coordinate b in [0, 12) of a 2-D direction: its angle folded
    onto 180 degrees (a hatch has no arrow), in units of 15 degrees.
    Zero-length directions read bin 0."""
    tx = np.asarray(tx, np.float32)
    ty = np.asarray(ty, np.float32)
    th = np.arctan2(ty, tx).astype(np.float32)
    b = th * f32(DIRECTION_BINS / np.pi)
    b = b - np.floor(b / f32(DIRECTION_BINS)) * f32(DIRECTION_BINS)
    b = np.minimum(np.maximum(b, f32(0.0)), f32(11.999))
    zero = (tx * tx + ty * ty) < np.float32(1e-8)
    return np.where(zero, f32(0.0), b).astype(np.float32)


def bin_rotations(angle_deg, layers, fan=180.0):
    """The baked (cos, sin) table: DIRECTION_BINS rows of `layers` pairs,
    row i at Angle + 15 i degrees. Literals on the GPU twin."""
    L = max(int(layers), 1)
    return [[rotation(float(angle_deg) + 15.0 * i, k, L, fan)
             for k in range(L)] for i in range(DIRECTION_BINS)]


def blend_bins(fn, x, y, tone, b, table, *args, **kw):
    """Evaluate a directed medium: `fn(x, y, tone, rots, *args, **kw)` on
    the two bins each pixel sits between, cross-faded by a smoothstep of
    its fraction (only the last third of a bin blends), summed in the
    order f0 * (1 - w) + f1 * w exactly as the twin writes it."""
    bf = np.floor(b)
    i0 = np.minimum(bf.astype(np.int64), DIRECTION_BINS - 1)
    i1 = np.where(i0 == DIRECTION_BINS - 1, 0, i0 + 1)
    w = smoothstep(f32(0.35), f32(0.65), (b - bf).astype(np.float32))
    out = np.zeros(x.shape[0], np.float32)
    tone = np.asarray(tone, np.float32)
    for i in range(DIRECTION_BINS):
        s0 = i0 == i
        s1 = i1 == i
        sel = s0 | s1
        if not sel.any():
            continue
        f = fn(x[sel], y[sel], tone[sel], table[i], *args, **kw)
        full = np.zeros(x.shape[0], np.float32)
        full[sel] = f
        out = np.where(s0, out + full * (f32(1.0) - w), out)
        out = np.where(s1, out + full * w, out)
    return out.astype(np.float32)


# ------------------------------------------------------------------ hatching


def hatching(x, y, tone, rots, width=0.35, length=6.0, wobble=0.5,
             breaks=0.2, salt=0, layers=None):
    """Cross-hatching whose layers fill in as the tone darkens.

    `rots` is the list of baked (cos, sin) pairs, one per layer: layer k
    draws from darkness k/L upward, its lines thickening over the next
    1/L of the range before the next layer starts -- the tonal art map,
    continuous in tone -- and over the last fifth of the range every
    layer swells to fill its lane, so a full dark closes to solid ink.
    Each lane (unit spacing across the layer's direction) carries one
    stroke: a per-lane hash offsets its wobble, pressure and break noise
    along the stroke and jitters its pen width; strokes are drawn in
    segments of `length` with lifted ends, and `breaks` opens dry gaps.
    Layers combine as ink over ink. `layers` (default: the number of
    rotations) is the layer count the tone is divided by; passing fewer
    rotations than layers draws only those layers.
    """
    d = darkness(tone)
    L = max(int(layers) if layers else len(rots), 1)
    remain = np.ones(x.shape[0], np.float32)
    wid = clamp01(width)
    ln = f32(max(float(length), 1e-3))
    wob_amt = f32(wobble) * f32(0.6)
    brk = f32(breaks) * f32(0.6)
    fill = np.clip((d - 0.8) / 0.2, 0.0, 1.0)
    for k, (c, s) in enumerate(rots):
        u = x * c + y * s
        v = -x * s + y * c
        lane = np.floor(v)
        li = lane.astype(np.int64)
        lk = np.zeros_like(li)
        h0 = _h(li, lk, salt, k * 4 + 0)
        h1 = _h(li, lk, salt, k * 4 + 1)
        h2 = _h(li, lk, salt, k * 4 + 2)
        off = h0 * 512.0
        wob = (_n2(u * 0.35 + off, np.full_like(u, 0.5), salt, k) - 0.5) \
            * wob_amt
        a = (v - lane) - 0.5 + wob
        wk = np.clip((d - f32(k) / f32(L)) * f32(L), 0.0, 1.0)
        hw = 0.5 * (wid * (0.35 + 0.65 * wk) * (0.8 + 0.4 * h1)
                    + (f32(1.0) - wid) * fill)
        t = u / ln + h2
        tl = t - np.floor(t)
        ends = np.clip(np.minimum(tl, 1.0 - tl) / 0.12, 0.0, 1.0)
        ends = ends + (1.0 - ends) * fill
        press = 0.75 + 0.25 * _n2(u * 1.7 + off, np.full_like(u, 2.5),
                                  salt, k)
        gap = smoothstep(brk - f32(0.06), brk + f32(0.06),
                         _n2(u * 1.1 + off, np.full_like(u, 4.5), salt, k))
        gap = gap + (1.0 - gap) * fill
        line = np.clip((hw * ends - np.abs(a)) / 0.08 + 0.5, 0.0, 1.0) \
            * press * gap * smoothstep(0.0, 0.15, wk)
        remain = remain * (1.0 - line)
    return (1.0 - remain).astype(np.float32)


# ------------------------------------------------------------------ scribble


def scribble(x, y, tone, rots, width=0.3, curl=0.5, pressure=0.7,
             grain=0.6, salt=0, blend=0.0):
    """Pencil scribble: hatch lanes bent by a slow domain warp.

    The coordinate is displaced by `curl` * 2.5 cells of two-octave
    low-frequency noise before the lanes are taken, so the strokes bend,
    bunch and -- past curl 1, where the warp folds -- loop back on
    themselves the way a scribbling pencil does. Layers fan over
    SCRIBBLE_FAN degrees. Graphite deposits on the paper's tooth first:
    light pressure marks only the peaks, heavy pressure fills the
    valleys; pressure grows with the darkness, and over the last fifth
    of the range the lanes fill to solid graphite. `blend` (R235) is
    the tortillon of Sousa and Buchanan's pencil model: it takes
    graphite off the peaks and pushes it into the valleys -- the stroke
    loses half its contrast and a soft halo two and a half lanes wide
    appears around it, at the pressure's own tone.
    """
    d = darkness(tone)
    L = max(len(rots), 1)
    fill = np.clip((d - 0.8) / 0.2, 0.0, 1.0)
    bl = clamp01(blend)
    amp = f32(curl) * f32(2.5)
    xw = x + ((_n2(x * 0.4, y * 0.4, salt, 40) - 0.5)
              + (_n2(x * 0.9, y * 0.9, salt, 43) - 0.5) * 0.4) * amp
    yw = y + ((_n2(x * 0.4, y * 0.4, salt, 41) - 0.5)
              + (_n2(x * 0.9, y * 0.9, salt, 47) - 0.5) * 0.4) * amp
    tooth = _n2(x * 18.0, y * 18.0, salt, 42)
    g = f32(0.12) + f32(0.5) * (f32(1.0) - clamp01(grain))
    pd_base = f32(pressure)
    wid = f32(width)
    remain = np.ones(x.shape[0], np.float32)
    for k, (c, s) in enumerate(rots):
        u = xw * c + yw * s
        v = -xw * s + yw * c
        wob = (_n2(u * 0.6, v * 0.6 + f32(7.3) * f32(k), salt, 44 + k)
               - 0.5) * 0.5
        vv = v + wob
        a = (vv - np.floor(vv)) - 0.5
        wk = np.clip((d - f32(k) / f32(L)) * f32(L), 0.0, 1.0)
        hw = 0.5 * (wid * (0.4 + 0.6 * wk) + (f32(1.0) - wid) * fill)
        press = (0.75 + 0.25 * _n2(u * 4.0, v * 0.9, salt, 48 + k)) * pd_base
        line = np.clip((hw - np.abs(a)) / 0.1 + 0.5, 0.0, 1.0)
        pd = press * (0.6 + 0.6 * d)
        dep = line * np.clip((tooth - (1.0 - pd)) / g + 0.5 + fill, 0.0, 1.0)
        if bl > 0.0:
            # the blender: the deposit without the tooth at half strength,
            # and a soft halo (a triangle 2.5 half-widths wide) of the
            # graphite pushed out of the stroke
            hw2 = np.maximum(hw * f32(2.5), f32(1e-4))
            halo = np.clip((hw2 - np.abs(a)) / hw2, 0.0, 1.0) * pd * f32(0.5)
            smear = np.maximum(line * pd * f32(0.5), halo)
            dep = dep * (f32(1.0) - bl) + smear * bl
        dep = dep * smoothstep(0.0, 0.15, wk)
        remain = remain * (1.0 - dep)
    return (1.0 - remain).astype(np.float32)


# ------------------------------------------------------------------- stipple


STIPPLE_PLACEMENT_ITEMS = (
    ('SIZE', "Size & Count",
     "Dots on a jittered lattice, present by the tone and growing a "
     "little as it darkens; Fine adds a second lattice in the darks"),
    ('COUNT', "Count",
     "Secord's rule: dots of ONE size whose number carries the tone -- "
     "three nested lattices (1, 4 and 16 dots a cell) in a fixed order, "
     "coarse first, so a darkening tone adds dots without moving one, "
     "and the crowd stays evenly spread at every density"),
)

#: the COUNT placement's lattices: (scale, rank low, rank high) -- three
#: nested lattices at 4, 2 and 1 cells' spacing holding 1, 4 and 16 dots
#: per 4x4 block, 21 in all, each level's ranks in its own range so a
#: dot's rank is uniform over the 21 and the count is proportional to
#: the darkness; the finest level is one dot per cell, the SIZE
#: placement's own full density
_COUNT_LEVELS = ((0.25, 0.0, 1.0 / 21.0),
                 (0.5, 1.0 / 21.0, 5.0 / 21.0),
                 (1.0, 5.0 / 21.0, 1.0))


def stipple(x, y, tone, size=0.6, jitter=0.8, fine=1.0, salt=0,
            placement='SIZE'):
    """Pen stipple: dots on a jittered lattice, present by darkness.

    A cell's dot exists where its hash falls below the darkness, so dot
    DENSITY carries the tone the way a stippled drawing does; dots also
    grow with darkness. A second lattice at twice the frequency fills the
    darks (`fine`), and over the last fifth of the range the dots swell
    into one another, so a full black closes to solid ink -- the
    stippler's own way of reaching black.

    `placement` COUNT (R235) is Secord's stippling instead: dots of one
    size (Size, in cells, as the SIZE placement's), their NUMBER the
    tone. Three nested lattices (spacing 4, 2 and 1 cells -- 1, 4 and
    16 dots per four-cell block) each hand their dots a rank in their
    own range of [0, 1), coarse first; a dot shows where its rank is
    below the darkness. So the count is proportional to the darkness, a
    darkening tone only ever adds dots, the full crowd is one dot per
    cell, and the crowd is a jittered hierarchy -- evenly spread at
    every density, no clumps of a white-noise scatter. The last fifth
    of the range still swells the dots to a solid black.
    """
    d = darkness(tone)
    fill = np.clip((d - 0.8) / 0.2, 0.0, 1.0)
    best = np.zeros(x.shape[0], np.float32)
    jit = f32(jitter) * f32(0.9)
    fn = clamp01(fine)
    size32 = f32(size)
    if str(placement).upper() == 'COUNT':
        for level, (sc, lo, hi) in enumerate(_COUNT_LEVELS):
            sc = f32(sc)
            px = x * sc
            py = y * sc
            cx0 = np.floor(px).astype(np.int64)
            cy0 = np.floor(py).astype(np.int64)
            # one size: Size cells of the medium, whatever the level's
            # own spacing (the dot is drawn in that level's units)
            rad = size32 * f32(0.5) * (f32(1.0) + f32(1.5) * fill) * sc
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cx = cx0 + dx
                    cy = cy0 + dy
                    jx = _h(cx, cy, salt, 30 + level * 4)
                    jy = _h(cx, cy, salt, 31 + level * 4)
                    pres = _h(cx, cy, salt, 32 + level * 4)
                    rs = _h(cx, cy, salt, 33 + level * 4)
                    rank = f32(lo) + pres * f32(hi - lo)
                    ddx = px - (cx.astype(np.float32) + 0.5 + (jx - 0.5) * jit)
                    ddy = py - (cy.astype(np.float32) + 0.5 + (jy - 0.5) * jit)
                    dist = np.sqrt(ddx * ddx + ddy * ddy)
                    r = rad * (0.85 + 0.3 * rs)
                    dot = np.clip((r - dist) / 0.06 + 0.5, 0.0, 1.0)
                    dot = np.where(rank < d, dot, 0.0).astype(np.float32)
                    best = np.maximum(best, dot)
        return best.astype(np.float32)
    for level in (0, 1):
        if level == 1 and fn <= 0.0:
            break
        sc = f32(1.0) if level == 0 else f32(2.0)
        lv = f32(1.0) if level == 0 else f32(0.7)
        px = x * sc
        py = y * sc
        cx0 = np.floor(px).astype(np.int64)
        cy0 = np.floor(py).astype(np.int64)
        dens = d if level == 0 else np.clip((d - 0.5) * 2.0, 0.0, 1.0) * fn
        rad = size32 * 0.5 * (0.6 + 0.7 * d + 1.5 * fill) * lv
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cx = cx0 + dx
                cy = cy0 + dy
                jx = _h(cx, cy, salt, 20 + level * 4)
                jy = _h(cx, cy, salt, 21 + level * 4)
                pres = _h(cx, cy, salt, 22 + level * 4)
                rs = _h(cx, cy, salt, 23 + level * 4)
                ddx = px - (cx.astype(np.float32) + 0.5 + (jx - 0.5) * jit)
                ddy = py - (cy.astype(np.float32) + 0.5 + (jy - 0.5) * jit)
                dist = np.sqrt(ddx * ddx + ddy * ddy)
                r = rad * (0.75 + 0.5 * rs)
                dot = np.clip((r - dist) / 0.06 + 0.5, 0.0, 1.0)
                dot = np.where(pres < dens, dot, 0.0).astype(np.float32)
                best = np.maximum(best, dot)
    return best.astype(np.float32)


# ------------------------------------------------------------------ charcoal


def charcoal(x, y, tone, rot, grain=0.6, streak=0.5, smudge=0.4, salt=0,
             blend=0.0):
    """Charcoal: the stick deposits on the paper's tooth, peaks first.

    `rot` is the baked (cos, sin) of the stroke direction; `streak`
    stretches the tooth along it, the way a dragged stick streaks. The
    tooth is three octaves of noise plus the paper's own cell grain,
    stretched to span the whole tone range. Pressure is the darkness
    plus a slow smudge field (a thumb blending the tone) and the stick's
    own pressure in bands across the stroke; coverage is a threshold of
    the tooth against that pressure, crisp at high `grain`, so a light
    tone speckles and a dark one fills. `blend` (R235) is the stump:
    it flattens the tooth toward its mean and widens the threshold, so
    the charcoal that sat on the peaks spreads into the valleys as an
    even tone.
    """
    d = darkness(tone)
    bl = clamp01(blend)
    c, s = rot
    u = x * c + y * s
    v = -x * s + y * c
    ex = f32(1.0) / (f32(1.0) + f32(streak) * f32(4.0))
    t1 = _n2(u * 6.0 * ex, v * 6.0, salt, 60)
    t2 = _n2(u * 14.0 * ex, v * 14.0, salt, 61)
    t3 = _n2(u * 30.0 * ex, v * 30.0, salt, 63)
    t4 = _h(np.floor(u * 30.0 * ex).astype(np.int64),
            np.floor(v * 30.0).astype(np.int64), salt, 65)
    tooth = np.clip((t1 * 0.3 + t2 * 0.3 + t3 * 0.2 + t4 * 0.2 - 0.5) * 2.5
                    + 0.5, 0.0, 1.0)
    # the thumb and the bands move charcoal that is there: nothing on
    # bare paper, nothing left to move at solid black
    sm = ((_n2(x * 0.6, y * 0.6, salt, 62) - 0.5) * f32(smudge)
          + (_n2(u * 0.25, v * 1.6, salt, 64) - 0.5) * 0.45) \
        * np.clip(4.0 * d * (1.0 - d), 0.0, 1.0)
    g = f32(0.04) + f32(0.25) * (f32(1.0) - clamp01(grain))
    if bl > 0.0:
        tooth = tooth + (f32(0.5) - tooth) * (bl * f32(0.6))
        g = g + bl * f32(0.4)
    P = d * (1.0 + 2.0 * g) - g + sm
    cov = smoothstep(1.0 - P - g, 1.0 - P + g, tooth)
    return np.clip(cov, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------- paint strokes


def paint_slope(spread):
    """The per-dab turn as a SLOPE: tan(45 deg * spread), baked float32."""
    return f32(np.tan(np.radians(45.0 * float(clamp01(spread)))))


def paint_strokes(x, y, rot, length=2.6, width=0.85, slope=None,
                  bristles=0.6, variation=0.3, salt=0):
    """Brush dabs on a jittered lattice, later dabs over earlier ones.

    One dab per cell, a rounded rectangle about `length` by `width`
    (cells; each dab's own size varies by its hash) turned off the baked
    (cos, sin) direction by a per-dab amount up to the baked `slope`
    (paint_slope: tan of 45 deg * spread) -- a hashed SLOPE, not a hashed
    angle, so no driver's sin decides a dab's edge. Bristle streaks run
    along each dab on two octaves of value noise across it; the dab thins
    toward its end as the brush dries; `variation` lightens or darkens
    whole dabs. The two highest-ORDER dabs covering a pixel composite
    over the canvas (a two-deep painter's algorithm), so a dab's soft
    edge shows the dab beneath rather than a canvas halo. Returns
    (value, alpha, id): the composite paint value (1 = the paint colour
    as given), its coverage over the canvas, and the winning dab's hash
    (0 on bare canvas).
    """
    c0, s0 = rot
    hl0 = f32(max(float(length), 1e-3)) * f32(0.5)
    hw0 = f32(max(float(width), 1e-3)) * f32(0.5)
    slope = f32(paint_slope(0.5) if slope is None else slope)
    cx0 = np.floor(x).astype(np.int64)
    cy0 = np.floor(y).astype(np.int64)
    n = x.shape[0]
    o1 = np.full(n, -1.0, np.float32)     # best order
    v1 = np.zeros(n, np.float32)
    a1 = np.zeros(n, np.float32)
    i1 = np.zeros(n, np.float32)
    o2 = np.full(n, -1.0, np.float32)     # runner-up
    v2 = np.zeros(n, np.float32)
    a2 = np.zeros(n, np.float32)
    br = f32(bristles)
    var = f32(variation)
    for dx in (-2, -1, 0, 1, 2):
        for dy in (-2, -1, 0, 1, 2):
            cx = cx0 + dx
            cy = cy0 + dy
            jx = _h(cx, cy, salt, 0)
            jy = _h(cx, cy, salt, 1)
            hj = _h(cx, cy, salt, 2)
            order = _h(cx, cy, salt, 3)
            val = _h(cx, cy, salt, 4)
            bid = _h(cx, cy, salt, 5)
            ex = x - (cx.astype(np.float32) + 0.5 + (jx - 0.5) * 0.9)
            ey = y - (cy.astype(np.float32) + 0.5 + (jy - 0.5) * 0.9)
            # the dab's own direction: the base direction turned by a
            # hashed slope, normalised with one sqrt
            t = (hj - 0.5) * (2.0 * slope)
            inv = 1.0 / np.sqrt(1.0 + t * t)
            dxr = (c0 - s0 * t) * inv
            dyr = (s0 + c0 * t) * inv
            al = ex * dxr + ey * dyr
            ac = -ex * dyr + ey * dxr
            hl = hl0 * (0.7 + 0.6 * bid)
            hw = hw0 * (0.8 + 0.4 * hj) * (1.0 - 0.3 * np.clip(al / hl,
                                                                0.0, 1.0))
            qa = al / hl
            qc = ac / hw
            q = (qa * qa) * (qa * qa) + (qc * qc) * (qc * qc)
            alpha = np.clip((1.0 - q) / 0.12, 0.0, 1.0)
            cover = alpha > 0.0
            b1 = _n2(ac * 9.0 + bid * 64.0, al * 0.5 + val * 64.0, salt, 6)
            b2 = _n2(ac * 21.0 + val * 64.0, al * 1.1 + bid * 64.0, salt, 7)
            bri = b1 * 0.6 + b2 * 0.4
            value = (1.0 + var * (val - 0.5)) * (1.0 + br * (bri - 0.5))
            value = value.astype(np.float32)
            win = cover & (order > o1)
            second = cover & ~win & (order > o2)
            # a new winner demotes the old one to runner-up
            o2 = np.where(win, o1, np.where(second, order, o2))
            v2 = np.where(win, v1, np.where(second, value, v2))
            a2 = np.where(win, a1, np.where(second, alpha, a2))
            o1 = np.where(win, order, o1)
            v1 = np.where(win, value, v1)
            a1 = np.where(win, alpha, a1)
            i1 = np.where(win, bid, i1)
    # composite: runner-up over the canvas, winner over that
    under_v = v2 * a2
    under_a = a2
    value = under_v + (v1 - under_v) * a1
    alpha = under_a + (1.0 - under_a) * a1
    return (value.astype(np.float32), alpha.astype(np.float32),
            i1.astype(np.float32))


# ---------------------------------------------------------------------- wash


def wash(x, y, tone, levels=3, pooling=0.6, granulation=0.4, bleed=0.4,
         salt=0):
    """Ink or watercolour wash: the tone in flat washes with pooled edges.

    `levels` washes each add 1/levels of darkness where the (bleed-
    wandered) darkness passes their threshold; pigment pools inside each
    wash's edge (`pooling`) and settles into the paper's tooth
    (`granulation`, two octaves).
    """
    d = darkness(tone)
    n = max(min(int(levels), 6), 1)
    dn = d + (_n2(x * 0.9, y * 0.9, salt, 70) - 0.5) * (f32(0.5) * f32(bleed))
    cov = np.zeros(x.shape[0], np.float32)
    pool = f32(pooling)
    share = f32(1.0) / f32(n)
    for k in range(1, n + 1):
        e = dn - f32(k) / f32(n + 1)
        ak = smoothstep(0.0, 0.03, e)
        r = np.clip(1.0 - e / 0.06, 0.0, 1.0)
        rim = pool * (r * r) * ak
        cov = cov + share * ak * (1.0 + rim)
    g1 = _n2(x * 9.0, y * 9.0, salt, 71)
    g2 = _n2(x * 21.0, y * 21.0, salt, 72)
    gr = 1.0 + f32(granulation) * ((g1 * 0.6 + g2 * 0.4) - 0.5)
    return np.clip(cov * gr, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------- paper


def paper(x, y, tooth=0.6, fibres=0.3, mottle=0.4, salt=0):
    """Paper: fine tooth, long fibres, slow mottle, as a height in 0..1."""
    t1 = _n2(x * 9.0, y * 9.0, salt, 80)
    t2 = _n2(x * 23.0, y * 23.0, salt, 81)
    tv = t1 * 0.6 + t2 * 0.4
    f1 = _n2(x * 1.3, y * 16.0, salt, 82)
    f2 = _n2(x * 16.0, y * 1.3, salt, 83)
    fb = (f1 + f2) * 0.5
    m1 = _n2(x * 0.8, y * 0.8, salt, 84)
    m2 = _n2(x * 1.6, y * 1.6, salt, 85)
    mv = m1 * 0.65 + m2 * 0.35
    h = 0.5 + f32(tooth) * (tv - 0.5) + f32(fibres) * (fb - 0.5) \
        + f32(mottle) * (mv - 0.5)
    return np.clip(h, 0.0, 1.0).astype(np.float32)
