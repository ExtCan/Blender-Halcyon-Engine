"""Soft shadows from a shadow map whose width follows the LAMP'S SIZE.

A lamp with a size (POINT / SPOT Radius, SUN Angle, AREA Size and Size Y)
casts a penumbra that is sharp where the caster touches the receiver and
widens with the caster-to-receiver distance and with the lamp. A depth
map holds enough to reproduce that: per receiver, find the casters the
lamp's footprint can see (a blocker search), take their mean depth, size
the penumbra by similar triangles and filter the map over exactly that
footprint. This module is that lookup for the CPU road; gpu/softmap.py
emits the same statements as GLSL, tap for tap.

No lamp comes here at Samples 1, and a lamp of size 0 comes only when
the user's Softness is above one texel (its footprint is then the
Softness alone, `zero_shape`): `lights.visibility` keeps the constant
PCF lookup for the rest (`ShadowMap.lookup`), untouched (`lookup_shape`).

THE LOOKUP (one lamp, one receiver point P' already stepped off its
surface by the caller -- `offset_scale` says how far --, shading normal N,
ndl = clamp(N.L, 0, 1), slope bias sl, S = the user Softness in texels,
F filter taps, B search taps)

  1. project P' exactly as ShadowMap.lookup does: texel coordinates
     (u, v), the receiver depth d_r, `inside`; a cube picks its face by
     the major axis of P' - O.
  2. the receiver plane (through P', normal N) as a depth function of a
     texel offset (du, dv). k = size / (2 extent) texels per tangent unit
     (perspective) or per world unit (ortho); nd = max(ndl, 0.1):
        perspective: c = -|P' - O| nd;  gu = (N.s)/(c k);  gv = (N.u)/(c k)
                     depth = 1 / max(1/d_r + gu du + gv dv, 1/d_r)
        ortho      : gu = (N.s)/(nd k); gv = (N.u)/(nd k)
                     depth = min(d_r + gu du + gv dv, d_r)
     (s, u: the map's right and up world axes; 1/depth is linear on a
     plane under a perspective map.) The depth is never taken deeper than
     the receiver itself: nothing beyond d_r stands between it and the
     lamp, and a plane continued past a concave corner would otherwise
     run behind the neighbouring surface and read it as a caster.
     EVERY compare below is made against this plane depth at the
     tap's own texel centre, so a wide kernel on a tilted receiver needs
     no grown bias and no grown normal offset -- the two things that used
     to erase the shadow as the lamp grew.
  3. the lamp as two texel vectors per unit of t, (A, Bv): a unit-disc tap
     (a, b) lands at t (a A + b Bv), with t(d) = 1/d - 1/d_r (perspective)
     or d_r - d (ortho) for a caster at depth d.
        POINT / SPOT, radius R (a disc facing the receiver), with
            x = ndc_x extent, y = ndc_y extent, s = sqrt(1 + x^2 + y^2):
            A  = k R (1 + x^2/(1+s), x y/(1+s))
            Bv = k R (x y/(1+s), 1 + y^2/(1+s))
        AREA, half-axes E1, E2 (world), e_i = (E_i.s, E_i.u, E_i.f):
            A  = k (e1.x - x e1.z, e1.y - y e1.z),  Bv likewise with e2
            (SQUARE / RECTANGLE scale the half-axes by 2/sqrt(pi): the
            disc pattern stands for the rectangle at equal area)
        SUN, angle: A = (k tan(angle/2), 0),  Bv = (0, k tan(angle/2))
     amax = max(|A|, |Bv|).
  4. BLOCKER SEARCH at t_s = min(max(t(d_near), 0), cap / amax). d_near
     is the smallest depth the map holds WITHIN THE SEARCH'S OWN REACH
     of this receiver (no caster the search could find is nearer); cap
     bounds the penumbra RADIUS: a tangent of 1/8 at the lamp under a
     perspective map (size/16 texels of a cube face, the same angle for
     a Spot's cone of any width, a quarter of the map at most), size/16
     under an ortho map -- a fraction of the map, never a texel count,
     so the same lamp is equally soft on a 512 and a 2048 map
     (`cap_texels`).
     d_near comes from the map's REACH PYRAMID (`map_pyramid`): per
     level a grid of tiles 4, 8, 16 .. texels wide, each holding the
     smallest depth over the 3 x 3 tiles around it, so ONE texel answers
     "the nearest depth within r texels of (u, v)" for any r up to the
     tile's width. Starting from the cap (the widest reach there is),
     SEARCH_STEPS times: r = t_s (1 + 4/F) amax + S + 3, d_near =
     max(d_near, the pyramid at r), t_s again. Every step is
     conservative -- no caster inside the current reach is nearer than
     its minimum, so none of them matters beyond the next, smaller
     reach -- and it ends at the reach of the casters that stand near
     this receiver. The nearest depth of the whole map used to bound
     the search for every receiver: under a tilted Sun that is the near
     corner of the ground, tens of units nearer than any caster, the
     search taps stood units apart and stepped over the casters, so the
     same lamp and objects cast a different shadow on a larger ground.
     A receiver the pyramid shows NO depth nearer than d_r - sl within
     reach is lit, exactly (no tap could be a blocker: the plane's
     depth is never deeper than d_r), and skips the search.
     Taps: the centre, the convex hull of the filter pattern and interior
     taps of it. Tap (a, b) reads the texel at
        floor((u, v) + t_s (1 + 4/F) (a A + b Bv) + (S + 2) (a, b)).
     No blocker: lit. Every tap blocked: umbra.
  5. t_p = clamp(t(mean blocker depth), 0, t_s), then a PROBE with the
     same taps at t_p: none blocked -> lit, all blocked -> umbra. A wide
     search averages casters the lamp's footprint at this receiver never
     reaches (the top of a wall for a point at its foot), so the estimate
     is taken once more from the casters the probe found:
     t_f = clamp(t(their mean depth), 0, t_p). It costs no fetch and
     tightens the shadow where the caster meets the receiver.
  6. FILTER: the lamp's penumbra (radius t_f amax texels) and the user's
     Softness S are two blurs of one edge, so the radii add in
     quadrature: r = sqrt((t_f amax)^2 + S^2), fq = r / (t_f amax + S).
     q = max(6 r / F, 1), st0 = the largest of 1, 2, 4 .. 64 not above q,
     st1 = 2 st0, wb = clamp((q/st0 - sqrt 2) / (2 - sqrt 2), 0, 1). The
     F taps sit at
        p = (u, v) + fq (t_f (a A + b Bv) + S (a, b))
     and each is a BILINEAR compare on the lattice of stride st: texels
     (x0 + {0,1}) st + floor(st/2) with x0 = floor(p.x / st - 0.5) and
     the fractions as weights. lit = mean(st0) (1 - wb) + mean(st1) wb.
     The lattice ramp is one and a half to three tap spacings wide, so
     the sum is a continuous ramp at any tap count: no ghost copies, no
     bands, on a curved receiver either.

A cube map's kernel crosses face seams: a tap that leaves its face reads
the neighbouring face along the same direction from the lamp, with the
stored depth re-expressed along this face's axis (`_CubeFetch`). Nothing
is added to the baked map for it.
"""

import numpy as np

EPS = 1e-6

#: the penumbra / search radius cap. Under a perspective map it is an
#: ANGLE at the lamp (a tangent, so a Point lamp's cube and a Spot's cone
#: of any width give one lamp the same reach: an eighth is a sixteenth
#: of a cube face), never more than a quarter of the map; under an ortho
#: map a sixteenth of the map. Always a fraction of the map, not a texel
#: count, so the same lamp is equally soft on a 512 and a 2048 map.
#: Wider caps were measured and dropped: the far penumbra of a very
#: large lamp gets closer to the traced one, but one mean caster depth
#: over so wide a search lets light into the umbra and the twelve
#: search taps step over casters (a Sun of 0.6 rad: steps of 0.3)
TAN_CAP = 1.0 / 8.0
MAP_CAP = 1.0 / 4.0
ORTHO_CAP = 1.0 / 16.0
#: the receiver-plane slope clamp (N.L below this is treated as this)
NDL_MIN = 0.1
#: the search disc is (1 + SEARCH_GROW / F) times the widest filter disc
#: plus SEARCH_PAD texels (the bilinear footprint)
SEARCH_GROW = 4.0
SEARCH_PAD = 2.0
#: THE REACH PYRAMID. Its finest tile is 2 ** PYR_BASE texels wide: the
#: smallest reach ever asked is the search pad plus one texel (3 at
#: Softness 0), so finer levels would never be read, and starting at 4
#: keeps the whole pyramid at a twelfth of the map's texels
PYR_BASE = 2
#: how many times, at most, the search's reach is tightened against the
#: pyramid. One query costs one texel, and a query that does not move
#: the bound ends them (the next would ask the same cell). Each step
#: shrinks the reach by the factor the receiver's own slope leaves: on
#: open ground a Sun of 0.2 rad 40 degrees off the vertical comes from
#: the cap's 164 texels (a 2048 map) to 53, 13, 6 and its fixed point
#: of 5; near a caster it stops at that caster's reach in two or three
SEARCH_STEPS = 6
#: a receiver with nothing nearer than itself within reach skips the
#: search (CPU road only; the tests turn it off to prove it moves no bit)
SEARCH_SKIP = True
#: the lattice strides above 1 (a comparison ladder on both roads)
STRIDES = (2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
#: the lattice stride follows LATTICE * r / F: three times the mean
#: spacing of the F taps across the kernel. A tap's bilinear ramp is then
#: wider than the gaps the Vogel disc leaves when it is projected on an
#: edge's normal, so the taps do not cross a caster's edge one flat step
#: at a time. At one spacing a ball in a box's penumbra showed the F + 1
#: levels as stripes (a quarter of the penumbra flat at 16 taps); at two
#: 4 % was still flat; at three none is, and the widths are the traced
#: ones (the ramp is a small blur beside the kernel itself). The price is
#: a coarser reading of a caster's outline under a wide lamp: the edge of
#: a sliding caster's penumbra wanders by up to 3 texels where it
#: wandered by 1.5 (radius 0.5 at 7 units, a 512 map)
LATTICE = 6.0
#: the lattice's stride ACROSS a footprint's short side: from a ratio of
#: ANISO_FROM between the kernel's blur radii along the map's two axes
#: it is divided by the largest of ANISO_STEPS (and 1) not above the
#: ratio, cross-faded to the next (`_filter`)
ANISO_FROM = 1.5
ANISO_STEPS = (2.0, 4.0, 8.0, 16.0)
#: (CPU road only; the tests turn it off to prove a round footprint
#: keeps its one stride bit for bit)
ANISO = True
SQRT2 = 1.41421356
SQRT2_SPAN = 0.58578644

_GOLD = np.pi * (3.0 - np.sqrt(5.0))
#: the caller's normal offset is this fraction of the classic one where
#: the receiver plane serves the compare exactly, and grows to the whole
#: of it between N.L = GRAZE_HI and NDL_MIN, where the slope clamp bites
OFFSET_BASE = 0.25
GRAZE_HI = 0.3
_TABLES = {}
f32 = np.float32


def lamp_shape(light):
    """The lamp's size as the lookup reads it, or None for a lamp of size
    0 (which keeps the constant PCF lookup). SUN carries an ANGLE in
    `radius`, AREA its rectangle, POINT / SPOT a world radius."""
    kind = str(getattr(light, 'type', 'POINT')).upper()
    if kind == 'SUN':
        ang = float(getattr(light, 'radius', 0.0) or 0.0)
        if ang <= 0.0:
            return None
        return {'kind': 'SUN', 'tan': float(np.tan(ang * 0.5))}
    if kind == 'AREA':
        asz = getattr(light, 'area_size', (0.0, 0.0)) or (0.0, 0.0)
        hx, hy = float(asz[0]) * 0.5, float(asz[1]) * 0.5
        if hx <= 0.0 and hy <= 0.0:
            return None
        ax = np.asarray(getattr(light, 'area_x', (1, 0, 0)), np.float64)
        ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
        ay = np.asarray(getattr(light, 'area_y', (0, 1, 0)), np.float64)
        ay = ay / max(float(np.linalg.norm(ay)), 1e-9)
        disc = str(getattr(light, 'area_shape', 'SQUARE')) \
            in ('DISK', 'ELLIPSE')
        # the tap pattern is a unit disc: an ellipse uses the half-axes
        # as they are, a rectangle the disc of equal area
        g = 1.0 if disc else 2.0 / np.sqrt(np.pi)
        return {'kind': 'RECT', 'E1': ax * (max(hx, 0.0) * g),
                'E2': ay * (max(hy, 0.0) * g)}
    r = float(getattr(light, 'radius', 0.0) or 0.0)
    if r <= 0.0:
        return None
    return {'kind': 'SPHERE', 'R': r}


def offset_scale(ndl):
    """How much of the classic normal offset a sized lamp's lookup point
    takes, per point. The plane-biased compare needs none of it on a flat
    receiver, and a lifted point reads its caster's shadow displaced (it
    can even land inside a thin caster lying close to the floor), so the
    offset shrinks to OFFSET_BASE of itself -- a margin for shading
    normals that are not the facet's and for a Bias of 0. Towards grazing
    light the plane's slope is clamped (NDL_MIN) and no longer exact:
    there the whole classic offset returns."""
    w = np.clip((f32(GRAZE_HI) - ndl) / f32(GRAZE_HI - NDL_MIN),
                f32(0.0), f32(1.0))
    return (f32(OFFSET_BASE) + f32(1.0 - OFFSET_BASE) * w).astype(np.float32)


def zero_shape(light):
    """The shape of a lamp of size 0 in the sized lookup's own terms: no
    footprint at all (A = Bv = 0), so the only blur is the user's
    Softness. The kind follows the map the lamp bakes -- an ortho map
    for a Sun, a rectangle of two null half-axes for an Area (the text
    an Area's GLSL carries at any size), a null radius otherwise."""
    kind = str(getattr(light, 'type', 'POINT')).upper()
    if kind == 'SUN':
        return {'kind': 'SUN', 'tan': 0.0, 'zero': True}
    if kind == 'AREA':
        return {'kind': 'RECT', 'E1': np.zeros(3, np.float64),
                'E2': np.zeros(3, np.float64), 'zero': True}
    return {'kind': 'SPHERE', 'R': 0.0, 'zero': True}


def lookup_shape(light, samples, soft=0.0):
    """The lamp's shape when its mapped shadow takes the per-receiver
    lookup of this module, else None (the constant PCF lookup).

    Samples 1 is None for every lamp: one sample is the hard single-tap
    lookup whatever the lamp's size, the rule the traced road has always
    had (one sample, one hard ray), and what the 'hard shadow maps'
    presets ask for with Samples 1 on lamps that carry Blender's default
    sizes.

    A lamp WITH a size comes here at any Softness. A lamp of size 0
    comes here when the user's Softness (`soft`, the lamp's dial times
    the render's, in texels) is above 1: the blur is then wider than a
    texel, and the constant lookup could only afford it by lifting the
    receiver off its surface by the blur's radius -- which shrank the
    shadow and, at 16 texels, erased it. Here the same blur is F
    plane-biased bilinear compares (`zero_shape`: the filter of a sized
    lamp with no lamp term). At Softness 1 and below a lamp of size 0
    keeps the constant lookup, bit for bit."""
    if int(samples) <= 1:
        return None
    shape = lamp_shape(light)
    if shape is None and float(soft) > 1.0:
        return zero_shape(light)
    return shape


def tap_counts(samples):
    """(F, B) from the Samples dial (2 and up; 1 never comes here): F =
    4 filter taps per sample, 16 to 64; B = F/2 search taps, 12 to 32
    (the centre tap comes on top)."""
    F = int(min(max(4 * int(samples), 16), 64))
    return F, int(min(max(F // 2, 12), 32))


def cap_texels(size, extent, persp):
    """The penumbra / search radius cap of one map face, in its texels
    (float32, the constant both roads read)."""
    if persp:
        cap = min(MAP_CAP * float(size),
                  TAN_CAP * float(size) / (2.0 * float(extent)))
    else:
        cap = ORTHO_CAP * float(size)
    return f32(cap)


def _vogel(n):
    i = np.arange(n, dtype=np.float64)
    r = np.sqrt((i + 0.5) / n)
    th = i * _GOLD
    return np.stack([r * np.cos(th), r * np.sin(th)], 1).astype(np.float32)


def _hull(pts):
    """Indices of the convex-hull vertices of 2D points (monotone chain)."""
    order = sorted(range(len(pts)),
                   key=lambda i: (float(pts[i][0]), float(pts[i][1])))

    def cross(o, a, b):
        return (float(pts[a][0]) - float(pts[o][0])) * \
               (float(pts[b][1]) - float(pts[o][1])) - \
               (float(pts[a][1]) - float(pts[o][1])) * \
               (float(pts[b][0]) - float(pts[o][0]))
    lower, upper = [], []
    for i in order:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], i) <= 0:
            lower.pop()
        lower.append(i)
    for i in reversed(order):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], i) <= 0:
            upper.pop()
        upper.append(i)
    return sorted(set(lower[:-1] + upper[:-1]))


def filter_taps(F):
    """The F filter taps: a Vogel disc, (F, 2) float32."""
    key = ('f', int(F))
    if key not in _TABLES:
        _TABLES[key] = _vogel(int(F))
    return _TABLES[key]


def search_taps(F, B):
    """The blocker-search pattern, (B + 1, 2) float32: the centre, the
    CONVEX HULL of the filter pattern (the search disc is the filter disc
    scaled up, so a caster edge any filter tap can reach is reached by a
    search tap first), then interior taps of the same pattern spread
    evenly by index, B in all (never fewer than the hull)."""
    key = ('s', int(F), int(B))
    if key not in _TABLES:
        f = filter_taps(F)
        hull = _hull(f)
        rest = [i for i in range(int(F)) if i not in hull]
        want = max(int(B) - len(hull), 0)
        step = len(rest) / max(want, 1)
        inner = [rest[int(j * step)] for j in range(min(want, len(rest)))]
        idx = sorted(set(hull) | set(inner))
        _TABLES[key] = np.concatenate(
            [np.zeros((1, 2), np.float32), f[idx]], 0)
    return _TABLES[key]


def search_table(F, B, soft):
    """Per search tap (a, b, pad_a, pad_b) in float32 -- the constants
    BOTH roads read (the GLSL bakes them as literals), so a tap lands on
    the same texel on either device."""
    t = search_taps(F, B)
    pad = f32(float(soft) + SEARCH_PAD)
    return [(f32(a), f32(b), f32(pad * f32(a)), f32(pad * f32(b)))
            for a, b in t]


def filter_table(F, soft):
    """Per filter tap (a, b, soft_a, soft_b) in float32, shared with the
    GLSL emitter like `search_table`."""
    t = filter_taps(F)
    s = f32(float(soft))
    return [(f32(a), f32(b), f32(s * f32(a)), f32(s * f32(b)))
            for a, b in t]


def search_grow(F):
    return f32(1.0 + SEARCH_GROW / float(F))


def map_axes(vp, persp):
    """(s, u, f): the map's right, up and depth world axes, unit float32
    rows read off its view-projection matrix."""
    m = np.asarray(vp, np.float64)
    s = m[0, :3] / max(float(np.linalg.norm(m[0, :3])), 1e-30)
    u = m[1, :3] / max(float(np.linalg.norm(m[1, :3])), 1e-30)
    f = m[3, :3] if persp else m[2, :3]
    f = f / max(float(np.linalg.norm(f)), 1e-30)
    return (s.astype(np.float32), u.astype(np.float32),
            f.astype(np.float32))


def face_axes(face):
    ax = getattr(face, 'axes', None)
    return ax if ax is not None else map_axes(face.vp, face.persp)


def rect_axes(shape, axes):
    """An AREA lamp's two half-axes in one map face's frame:
    e_i = (E_i.s, E_i.u, E_i.f), float32."""
    s, u, f = (np.asarray(a, np.float64) for a in axes)
    out = []
    for e in (shape['E1'], shape['E2']):
        out.append(np.array([e @ s, e @ u, e @ f], np.float32))
    return out


def map_d_min(sm):
    """The smallest depth the lamp's map (every face of a cube) holds."""
    d = getattr(sm, 'd_min', None)
    if d is not None:
        return float(d)
    faces = getattr(sm, 'faces', None)
    if faces is not None:
        return min(float(f.depth.min()) for f in faces)
    return float(sm.depth.min())


def stamp_scene(scene):
    """Park the scene's radius -- `lights.scene_bounds`, what every map
    of this bake was framed by -- on each lamp's map (`scene_radius`).
    core/render calls it right after the bake; a map baked any other
    way carries none and its Bias is used as it is."""
    mesh = getattr(scene, 'mesh', None)
    verts = getattr(mesh, 'verts', None)
    if verts is None or not getattr(verts, 'size', 0):
        return
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    radius = float(np.linalg.norm(hi - lo)) * 0.5 + 1e-4
    for light in (getattr(scene, 'lights', None) or ()):
        sm = getattr(light, 'shadow_map', None)
        if sm is not None:
            try:
                sm.scene_radius = radius
            except AttributeError:
                pass


def scene_bias(sm, bias):
    """The Bias a mapped lookup runs with. It is a WORLD length (0.02 by
    default), made for scenes a unit or more across: in a scene modelled
    at centimetre size it is larger than the casters stand off their
    receivers and every mapped shadow is gone. Under a scene radius of
    one unit the Bias scales with the radius -- the default is then 2 %
    of it, where a shadow stays whole -- and at one unit or more it is
    the dial's own value, bit for bit."""
    r = getattr(sm, 'scene_radius', None)
    if r is None or not (float(r) < 1.0):
        return bias
    return bias * float(r)


def pyr_layout(size):
    """Where one face's reach pyramid keeps its levels, for a map face
    of `size` texels: {'levels': [(T, n, ox, oy)], 'w', 'h'} -- level j
    is an n x n grid of tiles T = 4 * 2 ** j texels wide whose first
    cell sits at (ox, oy) of a w x h block. The finest level fills the
    left of the block and the coarser ones stack in a column beside it
    (a mip chain's layout); the last level is one tile. Pure structure:
    both roads and the GPU pack read it.

    Tile x of a level covers the texels x T - 1 .. (x + 1) T - 2: the
    grid stands ONE TEXEL off the map's own, so the middle of a map and
    its halves and quarters -- where a lamp aimed at a centred scene
    puts whole lines of receivers -- are not tile borders, and the last
    bit of such a receiver's coordinate cannot choose its tile."""
    key = ('p', int(size))
    if key not in _TABLES:
        n = -(-(int(size) + 1) // (1 << PYR_BASE))
        t = 1 << PYR_BASE
        levels = [(t, n, 0, 0)]
        n0, y = n, 0
        while n > 1:
            n = -(-n // 2)
            t *= 2
            levels.append((t, n, n0, y))
            y += n
        _TABLES[key] = {
            'levels': levels,
            'w': n0 + (levels[1][1] if len(levels) > 1 else 0),
            'h': max(n0, y),
            # the per-level columns the CPU gathers through
            'n1': np.array([lv[1] - 1 for lv in levels], np.float32),
            'ox': np.array([lv[2] for lv in levels], np.int32),
            'oy': np.array([lv[3] for lv in levels], np.int32),
        }
    return _TABLES[key]


def _pool_first(d):
    """The minima of the finest tiles of one face, (n, n): tile x covers
    the texels 4x - 1 .. 4x + 2 (`pyr_layout`), so each of the four
    texel columns of a tile is one strided slice of the map."""
    f = 1 << PYR_BASE
    a = np.asarray(d, np.float32)
    # whole rows first (contiguous), then the columns of what is left
    rows = np.full((-(-(a.shape[0] + 1) // f), a.shape[1]), np.inf,
                   np.float32)
    for k in range(f):
        # texel f x - 1 + k of tile x
        src = a[f - 1::f] if k == 0 else a[k - 1::f]
        dst = rows[1:1 + src.shape[0]] if k == 0 else rows[:src.shape[0]]
        np.minimum(dst, src, out=dst)
    out = np.full((rows.shape[0], -(-(a.shape[1] + 1) // f)), np.inf,
                  np.float32)
    for k in range(f):
        src = rows[:, f - 1::f] if k == 0 else rows[:, k - 1::f]
        dst = out[:, 1:1 + src.shape[1]] if k == 0 \
            else out[:, :src.shape[1]]
        np.minimum(dst, src, out=dst)
    return out


def _pool_next(a):
    """2 x 2 minimum pooling of a stack of grids (.., n, n); an odd edge
    pools what tiles it has."""
    n = a.shape[-1]
    h = -(-n // 2)
    out = np.full(a.shape[:-2] + (h, h), np.inf, a.dtype)
    for dy in range(2):
        for dx in range(2):
            src = a[..., dy::2, dx::2]
            dst = out[..., :src.shape[-2], :src.shape[-1]]
            np.minimum(dst, src, out=dst)
    return out


def _dilate3(e):
    """The minimum over each 3 x 3 neighbourhood of a stack of
    (n + 2, n + 2) grids whose rim is the tiles beyond the face:
    (.., n, n)."""
    n = e.shape[-1] - 2
    out = e[..., 1:n + 1, 1:n + 1].copy()
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                np.minimum(out, e[..., dy:dy + n, dx:dx + n], out=out)
    return out


#: a depth read ACROSS a cube seam is a float32 quotient of products
#: (`_CubeFetch`): the bound stored for it sits this far below the exact
#: value, so no rounding of the lookup's own arithmetic undercuts it
_SEAM_SLACK = 1.0 - 4e-6
#: the softest user Softness (texels) whose reach a cube's seam ring
#: still covers; past it a reach that crosses a seam is bounded by the
#: nearest depth of the whole cube alone (`seam_ring`)
_SEAM_SOFT = 8.0


def seam_ring(size):
    """How many of the finest tiles a cube face's pyramid is continued
    past each edge: the first power of two of tiles that covers the
    widest reach a search asks (the cap's, at four taps per sample, with
    a Softness up to _SEAM_SOFT)."""
    reach = float(cap_texels(size, 1.0, True)) * (1.0 + SEARCH_GROW / 16.0) \
        + _SEAM_SOFT + SEARCH_PAD + 1.0
    w = 1
    while w * (1 << PYR_BASE) < reach:
        w *= 2
    return w


def _seam_grids(axes, plain, size, ring):
    """The finest level of a cube's six faces, each continued past its
    edges -- `ring` tiles before the first, 2 `ring` after the last (a
    coarser level's last tile can start a whole tile past the edge):
    (6, n + 3 ring, n + 3 ring), every tile beyond a face holding a
    lower bound of each depth a tap landing in it can read.

    A tap past the edge of a cube face reads the neighbouring face along
    the same direction from the lamp and divides the stored depth by
    that direction's component along the neighbour's axis
    (`_CubeFetch`). For a tile past the edge towards a neighbour G, the
    part beyond the edge is a rectangle of directions; its image on G
    is a convex quad, so the tiles of G under the image's bounding box
    hold every texel a tap can land on, and the largest divisor over
    the rectangle is at one of its corners. An edge tile the grid's own
    offset (or a ragged size) leaves partly outside the face takes the
    same bound for that part. All six faces at once: `axes` is
    (6, 3, 3), the faces' right / up / depth rows, `plain` (6, n, n)."""
    t = float(1 << PYR_BASE)
    n = plain.shape[-1]
    e = np.full((6, n + 3 * ring, n + 3 * ring), np.inf, np.float32)
    e[:, ring:ring + n, ring:ring + n] = plain
    s, u, f = axes[:, 0], axes[:, 1], axes[:, 2]
    idx = np.arange(-ring, n + 2 * ring, dtype=np.float64)  # extended tiles
    lo = idx * t - 1.0                                  # texel spans
    hi = lo + t
    fsel = np.arange(6)[:, None, None]
    eps = 1e-4
    to_ndc = 2.0 / size
    # the other way: every extended tile's borders, J + 1 of them
    pb = np.append(lo, hi[-1]) * to_ndc - 1.0
    # each face's own tiles, every one with its eight neighbours folded
    # in: one cell of it answers a bounding box up to three tiles wide
    rim = np.full((6, n + 2, n + 2), np.inf, np.float32)
    rim[:, 1:n + 1, 1:n + 1] = plain
    near3 = _dilate3(rim)

    def per_face(v):
        return v[:, None, None]

    def corners(a, fn):
        # over the four corners of each tile: (6, K + 1, J + 1) -> (6, K, J)
        return fn(fn(a[:, :-1, :-1], a[:, 1:, :-1]),
                  fn(a[:, :-1, 1:], a[:, 1:, 1:]))

    for along in (0, 1):
        for sign in (1.0, -1.0):
            # the tiles that reach past this edge (a run of neighbours),
            # with the span along the edge's normal clipped to the part
            # beyond it: K + 1 borders
            if sign > 0:
                sel = np.nonzero(hi > size)[0]
                pa = np.append(max(lo[sel[0]], float(size)), hi[sel])
            else:
                sel = np.nonzero(lo < 0.0)[0]
                pa = np.append(lo[sel], min(hi[sel[-1]], 0.0))
            pa = pa * to_ndc - 1.0
            g = np.argmax((axes[:, along] * sign) @ f.T, axis=1)
            sg, ug, fg = s[g], u[g], f[g]
            # the grid of the tiles' corners as directions d = xr s +
            # yr u + f per face: (K + 1, J + 1)
            xr, yr = (pa[:, None], pb[None, :]) if along == 0 \
                else (pb[None, :], pa[:, None])

            def along_axis(ax):
                return xr[None] * per_face((s * ax).sum(1)) \
                    + yr[None] * per_face((u * ax).sum(1)) \
                    + per_face((f * ax).sum(1))
            md = along_axis(fg)                         # (6, K + 1, J + 1)
            gx = ((along_axis(sg) / md * 0.5 + 0.5) * size + 1.0) / t
            gy = ((along_axis(ug) / md * 0.5 + 0.5) * size + 1.0) / t
            x0 = np.clip(np.floor(corners(gx, np.minimum) - eps), 0, n - 1) \
                .astype(np.int64)
            x1 = np.clip(np.floor(corners(gx, np.maximum) + eps), 0, n - 1) \
                .astype(np.int64)
            y0 = np.clip(np.floor(corners(gy, np.minimum) - eps), 0, n - 1) \
                .astype(np.int64)
            y1 = np.clip(np.floor(corners(gy, np.maximum) + eps), 0, n - 1) \
                .astype(np.int64)
            if int((y1 - y0).max()) <= 2 and int((x1 - x0).max()) <= 2:
                # the image of a tile this fine is under three tiles
                # wide: the folded cell at the box's middle covers it
                low = near3[g[:, None, None], (y0 + y1 + 1) // 2,
                            (x0 + x1 + 1) // 2].astype(np.float64)
            else:
                pg = plain[g]
                low = np.full(x0.shape, np.inf)
                for dy in range(int((y1 - y0).max()) + 1):
                    for dx in range(int((x1 - x0).max()) + 1):
                        np.minimum(low, pg[fsel, np.minimum(y0 + dy, y1),
                                           np.minimum(x0 + dx, x1)],
                                   out=low)
            low = low / corners(md, np.maximum) * _SEAM_SLACK   # (6, K, J)
            # down to float32 without rounding a bound up
            l32 = low.astype(np.float32)
            l32 = np.where(l32 > low, np.nextafter(l32, f32(-np.inf)), l32)
            run = slice(int(sel[0]), int(sel[-1]) + 1)
            if along == 0:
                np.minimum(e[:, :, run], l32.transpose(0, 2, 1),
                           out=e[:, :, run])
            else:
                np.minimum(e[:, run, :], l32, out=e[:, run, :])
    return e


def _build_pyramid(sm):
    """The reach pyramid of a map: one float32 block per face, laid out
    by `pyr_layout`. A cell of level j holds the smallest depth a tap
    can read within the 3 x 3 tiles around its own -- across the seams
    of a cube too -- so the cell under (u, v) at the level whose tile is
    at least r wide bounds every depth within r texels of (u, v). The
    one-tile top level of a flat map holds the nearest depth of the
    whole map, which is what a reach wider than the map gets (a flat
    map's taps never leave it: `_MapFetch` reads the edge texel).

    A cube's faces are continued past their edges ONCE, at the finest
    level, by `seam_ring` tiles read off the neighbouring faces
    (`_seam_grids`); every coarser level pools that. A level whose tile
    is wider than the ring knows nothing beyond the face: its cells
    along an edge hold 0, which bounds nothing, and the lookup's own
    floor (the nearest depth of the whole cube) serves there."""
    faces = getattr(sm, 'faces', None)
    cube = faces is not None
    if not cube:
        faces = [sm]
    size = int(faces[0].size)
    lay = pyr_layout(size)
    plain = np.stack([_pool_first(f.depth) for f in faces])
    ring = 0
    if cube:
        axes = np.array([[np.asarray(a, np.float64) for a in face_axes(f)]
                         for f in faces])
        ring = seam_ring(size)
        wide = _seam_grids(axes, plain, size, ring)
    blocks = np.zeros((len(faces), lay['h'], lay['w']), np.float32)
    for li, (t, n, ox, oy) in enumerate(lay['levels']):
        if li:
            plain = _pool_next(plain)
            if ring > 1:
                wide = _pool_next(wide)
            ring //= 2
        if ring:
            e = wide[:, ring - 1:ring + n + 1, ring - 1:ring + n + 1]
        else:
            e = np.full((len(faces), n + 2, n + 2),
                        0.0 if cube else np.inf, np.float32)
            e[:, 1:n + 1, 1:n + 1] = plain
            if cube:
                # past the ring: a tile that is not wholly inside the
                # face (the first always, by the grid's offset; the last
                # when it overhangs) knows nothing of its outer part
                last = n if n * t - 1 > size else n + 1
                e[:, 1, :] = e[:, :, 1] = 0.0
                e[:, last:, :] = 0.0
                e[:, :, last:] = 0.0
        blocks[:, oy:oy + n, ox:ox + n] = _dilate3(e)
    return list(blocks)


def map_pyramid(sm):
    """The map's reach pyramid, built once from its depth and parked on
    it (`ShadowMap.pyr` / `CubeShadow.pyr`; a bake that rewrites `depth`
    in place must reset the slot to None, as it must `digest`)."""
    p = getattr(sm, 'pyr', None)
    if p is None:
        p = _build_pyramid(sm)
        try:
            sm.pyr = p
        except AttributeError:
            pass
    return p


def _reach_min(block, lay, u, v, r):
    """The smallest depth within `r` texels of (u, v), read from one
    cell of the reach pyramid: the level whose tile is the first at
    least r wide (the top level for any wider reach)."""
    m, e = np.frexp(r)
    # r = m 2^e with m in [0.5, 1): the smallest power of two not below r
    lv = e - (m == f32(0.5))
    li = np.clip(lv - PYR_BASE, 0, len(lay['levels']) - 1)
    sh = -(li + PYR_BASE)
    n1 = lay['n1'][li]
    x = np.clip(np.floor(np.ldexp(u + f32(1.0), sh)), f32(0.0), n1)
    y = np.clip(np.floor(np.ldexp(v + f32(1.0), sh)), f32(0.0), n1)
    return block[lay['oy'][li] + y.astype(np.int32),
                 lay['ox'][li] + x.astype(np.int32)]


class _Pts:
    """The per-point state of one face's lookup; `take` narrows it to the
    points still undecided."""

    __slots__ = ('u', 'v', 'dr', 'gu', 'gv', 'sl', 'ax', 'ay', 'bx', 'by',
                 'ts', 'amax')

    def take(self, idx):
        o = _Pts()
        for k in self.__slots__:
            setattr(o, k, getattr(self, k)[idx])
        return o


class _MapFetch:
    """Depth texels of one flat map: a tap past the edge reads the edge
    texel (and is compared against the plane at THAT texel)."""

    def __init__(self, face):
        self.depth = face.depth
        self.hi = f32(face.size - 1)

    def __call__(self, xf, yf):
        xf = np.clip(xf, f32(0.0), self.hi)
        yf = np.clip(yf, f32(0.0), self.hi)
        return self.depth[yf.astype(np.int32), xf.astype(np.int32)], xf, yf


class _CubeFetch:
    """Depth texels of one cube face, continued past its edge: a texel
    outside the face is the direction x s + y u + f from the lamp, read
    from the face that direction belongs to and divided by its depth
    along that face's axis -- the depth along THIS face's axis, which is
    what the receiver plane is measured in."""

    def __init__(self, faces, fi):
        self.faces = faces
        self.fi = fi
        self.size = int(faces[fi].size)
        self.axes = [face_axes(f) for f in faces]
        self.inv = f32(2.0 / self.size)

    def __call__(self, xf, yf):
        size = self.size
        sz1 = size - 1
        xi = xf.astype(np.int32)
        yi = yf.astype(np.int32)
        out = (xi | yi | (sz1 - xi) | (sz1 - yi)) < 0
        depth = self.faces[self.fi].depth
        if not out.any():
            return depth[yi, xi], xf, yf
        occ = np.empty(xf.shape, np.float32)
        inn = ~out
        occ[inn] = depth[yi[inn], xi[inn]]
        s, u, f = self.axes[self.fi]
        xr = (xf[out] + f32(0.5)) * self.inv - f32(1.0)
        yr = (yf[out] + f32(0.5)) * self.inv - f32(1.0)
        d = xr[:, None] * s[None, :] + yr[:, None] * u[None, :] + f[None, :]
        a = np.abs(d)
        major = np.argmax(a, axis=1)
        comp = d[np.arange(d.shape[0]), major]
        g = major * 2 + (comp < 0)
        val = np.empty(d.shape[0], np.float32)
        hi = f32(sz1)
        for gi in np.unique(g):
            m = g == gi
            sg, ug, fg = self.axes[int(gi)]
            dm = d[m]
            md = dm[:, 0] * fg[0] + dm[:, 1] * fg[1] + dm[:, 2] * fg[2]
            ds = dm[:, 0] * sg[0] + dm[:, 1] * sg[1] + dm[:, 2] * sg[2]
            du = dm[:, 0] * ug[0] + dm[:, 1] * ug[1] + dm[:, 2] * ug[2]
            xg = np.clip(np.floor((ds / md * f32(0.5) + f32(0.5))
                                  * f32(size)), f32(0.0), hi)
            yg = np.clip(np.floor((du / md * f32(0.5) + f32(0.5))
                                  * f32(size)), f32(0.0), hi)
            val[m] = self.faces[int(gi)].depth[
                yg.astype(np.int32), xg.astype(np.int32)] / md
        occ[out] = val
        return occ, xf, yf


def _face(face, fetch, rect, P, N, ndl, slope, soft, shape, dmin,
          tab_s, tab_f, stats=None, pyr=None):
    """The lookup of M points against ONE map face. Returns lit (M,).
    `pyr` is the face's reach pyramid as (block, layout, is a cube's)."""
    n = P.shape[0]
    size = face.size
    persp = bool(face.persp)
    # -- 1. the projection, statement for statement ShadowMap.lookup's
    ph = np.empty((n, 4), np.float32)
    ph[:, :3] = P
    ph[:, 3] = 1.0
    clip = ph @ face.vp.T
    w = clip[:, 3]
    safe = np.where(np.abs(w) < EPS, EPS, w)
    ndc = clip[:, :3] / safe[:, None]
    inside = (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0) & \
             (ndc[:, 2] <= 1.0) & (w > 0.0 if persp else True)
    lit = np.ones(n, np.float32)
    act = np.nonzero(inside)[0]
    if act.size == 0:
        return lit
    if act.size < n:
        ndc, P, N, ndl, slope = ndc[act], P[act], N[act], ndl[act], slope[act]
    d_r = face._linearise(ndc[:, 2])
    pt = _Pts()
    pt.u = ((ndc[:, 0] * f32(0.5) + f32(0.5)) * f32(size)).astype(np.float32)
    pt.v = ((ndc[:, 1] * f32(0.5) + f32(0.5)) * f32(size)).astype(np.float32)
    pt.sl = slope.astype(np.float32)
    # -- 2. the receiver plane
    s_ax, u_ax, _f_ax = face_axes(face)
    ext = f32(face.extent)
    k = f32(size / (2.0 * face.extent))
    ns = N[:, 0] * s_ax[0] + N[:, 1] * s_ax[1] + N[:, 2] * s_ax[2]
    nu = N[:, 0] * u_ax[0] + N[:, 1] * u_ax[1] + N[:, 2] * u_ax[2]
    nd = np.maximum(ndl, f32(NDL_MIN))
    m = d_r.shape[0]
    if persp:
        rel = P - face.origin[None, :]
        deu = np.sqrt(rel[:, 0] * rel[:, 0] + rel[:, 1] * rel[:, 1]
                      + rel[:, 2] * rel[:, 2])
        c = -(deu * nd)
        pt.gu = (ns / (c * k)).astype(np.float32)
        pt.gv = (nu / (c * k)).astype(np.float32)
        x = ndc[:, 0] * ext
        y = ndc[:, 1] * ext
        pt.dr = (f32(1.0) / np.maximum(d_r, f32(1e-9))).astype(np.float32)
        # -- 3. the lamp's footprint per unit of t
        if shape['kind'] == 'SPHERE':
            kr = f32(float(k) * float(shape['R']))
            s1 = f32(1.0) + np.sqrt(f32(1.0) + x * x + y * y)
            pt.ax = kr * (f32(1.0) + x * x / s1)
            pt.ay = kr * (x * y / s1)
            pt.bx = pt.ay
            pt.by = kr * (f32(1.0) + y * y / s1)
        else:
            e1, e2 = rect
            pt.ax = k * (e1[0] - x * e1[2])
            pt.ay = k * (e1[1] - y * e1[2])
            pt.bx = k * (e2[0] - x * e2[2])
            pt.by = k * (e2[1] - y * e2[2])
        t_min = np.maximum(
            f32(1.0) / np.maximum(f32(dmin), f32(1e-9)) - pt.dr, f32(0.0))
    else:
        pt.gu = (ns / (nd * k)).astype(np.float32)
        pt.gv = (nu / (nd * k)).astype(np.float32)
        pt.dr = d_r.astype(np.float32)
        kt = f32(float(k) * float(shape['tan']))
        pt.ax = np.full(m, kt, np.float32)
        pt.ay = np.zeros(m, np.float32)
        pt.bx = np.zeros(m, np.float32)
        pt.by = np.full(m, kt, np.float32)
        t_min = np.maximum(pt.dr - f32(dmin), f32(0.0))
    pt.amax = np.maximum(np.sqrt(pt.ax * pt.ax + pt.ay * pt.ay),
                         np.sqrt(pt.bx * pt.bx + pt.by * pt.by))
    cap = cap_texels(size, face.extent, persp)
    cap_t = (cap / np.maximum(pt.amax, f32(1e-12))).astype(np.float32)
    grow = search_grow(len(tab_f))
    res = np.ones(m, np.float32)
    sel = None
    if pyr is None:
        # no pyramid (a direct call with a bare face): the search is
        # bounded by the nearest depth of the whole map
        pt.ts = np.minimum(t_min, cap_t).astype(np.float32)
    else:
        # -- 4a. the search's reach, tightened against the pyramid. It
        # starts at the cap -- the widest any receiver can ask -- and
        # each step reads the nearest depth within the current reach
        # and takes the reach that depth leaves
        blk, lay, cube = pyr
        padp1 = f32(f32(float(soft) + SEARCH_PAD) + f32(1.0))
        half = f32(size * 0.5)
        dlow = f32(f32(dmin) * f32(_SEAM_SLACK))

        def nearest(idx, r):
            uu = pt.u if idx is None else pt.u[idx]
            vv = pt.v if idx is None else pt.v[idx]
            q = _reach_min(blk, lay, uu, vv, r)
            if cube:
                # the nearest depth of the whole cube bounds a tap too:
                # stored along its own face's axis, it is read across a
                # seam divided by no more than the tap's own distance
                # from the face's centre, in half-faces. It serves where
                # the pyramid knows nothing past an edge (a reach wider
                # than the seam ring)
                far_out = np.maximum(np.abs(uu - half) + r,
                                     np.abs(vv - half) + r) / half
                q = np.maximum(q, dlow / np.maximum(far_out, f32(1.0)))
            return q

        def reach(nr, dr, capt):
            if persp:
                t = f32(1.0) / np.maximum(nr, f32(1e-9)) - dr
            else:
                t = dr - nr
            return np.minimum(np.maximum(t, f32(0.0)), capt)

        # the cap's own reach first: one number for the whole lamp (a
        # footprint with no lamp term asks the pad's alone)
        lamp_on = pt.amax > f32(0.0)
        near = nearest(None, np.where(lamp_on, f32(cap * grow), f32(0.0))
                       + padp1)
        ts = reach(near, pt.dr, cap_t).astype(np.float32)
        # nothing within reach is nearer than the receiver less its
        # bias: no tap can be a blocker (the plane's depth is never
        # deeper than the receiver's own, the value `plane` floors at),
        # so the point is lit and skips the search
        lim = ((f32(1.0) / pt.dr) if persp else pt.dr) - pt.sl
        asked = m
        # the later steps, each only where the last one moved the bound
        # and the point is not lit yet: an unmoved bound asks the same
        # cell again
        idx = np.nonzero(lamp_on & (near < lim))[0]
        for _step in range(1, SEARCH_STEPS):
            if idx.size == 0:
                break
            asked += int(idx.size)
            q = nearest(idx, ts[idx] * grow * pt.amax[idx] + padp1)
            moved = q > near[idx]
            idx = idx[moved]
            near[idx] = q[moved]
            ts[idx] = reach(near[idx], pt.dr[idx], cap_t[idx])
            idx = idx[near[idx] < lim[idx]]
        pt.ts = ts
        lit_now = near >= lim
        if not SEARCH_SKIP:
            lit_now = np.zeros(m, bool)
        if stats is not None:
            stats['bound'] = stats.get('bound', 0) + int(m)
            stats['asked'] = stats.get('asked', 0) + int(asked)
            stats['skip'] = stats.get('skip', 0) + int(lit_now.sum())
        sel = np.nonzero(~lit_now)[0]
        if sel.size == 0:
            if act.size < n:
                lit[act] = res
                return lit
            return res
        if sel.size < m:
            pt = pt.take(sel)
        else:
            sel = None

    def plane(p, du, dv):
        lin = p.dr + p.gu * du + p.gv * dv
        if persp:
            # 1/depth, floored at the receiver's own: no deeper than d_r
            return f32(1.0) / np.maximum(lin, p.dr)
        return np.minimum(lin, p.dr)

    def blockers(p, t, want_depth):
        cnt = np.zeros(p.u.shape[0], np.float32)
        dsum = np.zeros(p.u.shape[0], np.float32) if want_depth else None
        tg = t * grow
        for a, b, pa, pb in tab_s:
            xf = np.floor(p.u + (tg * (a * p.ax + b * p.bx) + pa))
            yf = np.floor(p.v + (tg * (a * p.ay + b * p.by) + pb))
            occ, xf, yf = fetch(xf, yf)
            pd = plane(p, xf + f32(0.5) - p.u, yf + f32(0.5) - p.v)
            blk = occ < pd - p.sl
            cnt += blk
            if want_depth:
                dsum += np.where(blk, occ, f32(0.0))
        return cnt, dsum

    # -- 4. the blocker search
    nt = f32(len(tab_s))
    nb, dsum = blockers(pt, pt.ts, True)
    if stats is not None:
        stats['search'] = stats.get('search', 0) + int(nb.shape[0])
    rs = np.ones(nb.shape[0], np.float32)
    rs[nb >= nt] = 0.0
    idx = np.nonzero((nb > 0.0) & (nb < nt))[0]
    if idx.size:
        # -- 5. the penumbra estimate, then the probe at its own radius
        p2 = pt.take(idx)
        davg = dsum[idx] / np.maximum(nb[idx], f32(1.0))
        if persp:
            tp = f32(1.0) / np.maximum(davg, f32(1e-9)) - p2.dr
        else:
            tp = p2.dr - davg
        tp = np.minimum(np.maximum(tp, f32(0.0)), p2.ts).astype(np.float32)
        # a footprint with no lamp term (a lamp of size 0 under the
        # user's Softness) puts the probe's taps exactly on the search's:
        # the probe would return what the search just did, so it is
        # skipped there (the GLSL branches on the same amax > 0)
        live = np.nonzero(p2.amax > f32(0.0))[0]
        if live.size == idx.size:
            nb2, dsum2 = blockers(p2, tp, True)
        else:
            nb2, dsum2 = nb[idx].copy(), dsum[idx].copy()
            if live.size:
                nb2[live], dsum2[live] = blockers(p2.take(live), tp[live],
                                                  True)
        if stats is not None:
            stats['probe'] = stats.get('probe', 0) + int(live.size)
        rs[idx[nb2 >= nt]] = 0.0
        keep = np.nonzero((nb2 > 0.0) & (nb2 < nt))[0]
        if keep.size:
            # -- 6. the estimate again from the casters the probe itself
            # found (never wider than the probe), then the filter --
            # penumbra points only
            p3 = p2.take(keep)
            davg2 = dsum2[keep] / np.maximum(nb2[keep], f32(1.0))
            if persp:
                tf = f32(1.0) / np.maximum(davg2, f32(1e-9)) - p3.dr
            else:
                tf = p3.dr - davg2
            tf = np.minimum(np.maximum(tf, f32(0.0)), tp[keep]) \
                .astype(np.float32)
            rs[idx[keep]] = _filter(p3, tf, soft, tab_f, fetch, plane, stats)
    if sel is None:
        res = rs
    else:
        res[sel] = rs
    if act.size < n:
        lit[act] = res
        return lit
    return res


def _filter(p, tf, soft, tab_f, fetch, plane, stats=None):
    fn = f32(len(tab_f))
    # the lamp's penumbra and the user's Softness are two blurs of one
    # edge: their radii add in quadrature, and the taps (laid out for the
    # plain sum) are drawn in by the ratio
    rl = tf * p.amax
    s2 = f32(soft) * f32(soft)
    r = np.sqrt(rl * rl + s2)
    fq = (r / np.maximum(rl + f32(soft), f32(1e-12))).astype(np.float32)
    q = np.maximum(f32(LATTICE) * r / fn, f32(1.0)).astype(np.float32)
    st0 = np.ones(q.shape[0], np.float32)
    for lv in STRIDES:
        st0 = np.where(q >= f32(lv), f32(lv), st0)
    wb = np.clip((q / st0 - f32(SQRT2)) / f32(SQRT2_SPAN),
                 f32(0.0), f32(1.0)).astype(np.float32)
    # A STRIP's footprint (an Area lamp far longer than wide) is as long
    # along one axis of the map as the stride above allows for, and a
    # fraction of that along the other: there the taps stand that much
    # closer, and a lattice as coarse as the long side needs blurred
    # the lamp's sharp side to two or three times its width. The
    # kernel's blur radius along each map axis (rx, ry), their ratio kq:
    # the stride ACROSS the short one is divided by the largest of 1, 2,
    # 4 .. 16 not above kq, cross-faded to the next like the stride
    # itself. Under a ratio of ANISO_FROM -- every round lamp: a disc's
    # footprint reaches 1.41 at a cube face's edge -- nothing changes
    ex = tf * np.sqrt(p.ax * p.ax + p.bx * p.bx)
    ey = tf * np.sqrt(p.ay * p.ay + p.by * p.by)
    rx = np.sqrt(ex * ex + s2)
    ry = np.sqrt(ey * ey + s2)
    kq = (np.maximum(rx, ry)
          / np.maximum(np.minimum(rx, ry), f32(1e-12))).astype(np.float32)
    k0 = np.ones(q.shape[0], np.float32)
    for lv in ANISO_STEPS:
        k0 = np.where(kq >= f32(lv), f32(lv), k0)
    wa = np.clip((kq / k0 - f32(ANISO_FROM)) / f32(2.0 - ANISO_FROM),
                 f32(0.0), f32(1.0)).astype(np.float32)
    if not ANISO:
        k0 = np.ones(q.shape[0], np.float32)
        wa = np.zeros(q.shape[0], np.float32)
    thin_x = rx <= ry

    def strides(st, k, idx=None):
        thin = np.maximum(st / k, f32(1.0))
        tx = thin_x if idx is None else thin_x[idx]
        return np.where(tx, thin, st), np.where(tx, st, thin)

    blend = np.nonzero(wb > 0.0)[0]
    fine = np.nonzero(wa > 0.0)[0]
    both = np.nonzero((wb > 0.0) & (wa > 0.0))[0]
    pb = p.take(blend) if blend.size else None
    pf = p.take(fine) if fine.size else None
    pbf = p.take(both) if both.size else None
    st1 = (st0 * f32(2.0)).astype(np.float32)
    sx0, sy0 = strides(st0, k0)
    if blend.size:
        sx1, sy1 = strides(st1[blend], k0[blend], blend)
    if fine.size:
        sx0f, sy0f = strides(st0[fine], k0[fine] * f32(2.0), fine)
    if both.size:
        sx1f, sy1f = strides(st1[both], k0[both] * f32(2.0), both)
    if stats is not None:
        stats['filter'] = stats.get('filter', 0) + int(q.shape[0])
        stats['filter2'] = stats.get('filter2', 0) + int(blend.size) \
            + int(fine.size) + int(both.size)
        stats['strip'] = stats.get('strip', 0) \
            + int(((k0 > f32(1.0)) | (wa > f32(0.0))).sum())

    def tap(pp, px, py, sx, sy):
        gx = px / sx - f32(0.5)
        gy = py / sy - f32(0.5)
        x0 = np.floor(gx)
        y0 = np.floor(gy)
        fx = gx - x0
        fy = gy - y0
        hx = np.floor(sx * f32(0.5))
        hy = np.floor(sy * f32(0.5))
        acc = np.zeros(px.shape[0], np.float32)
        for dy, wy in ((0.0, f32(1.0) - fy), (1.0, fy)):
            for dx, wx in ((0.0, f32(1.0) - fx), (1.0, fx)):
                occ, xf, yf = fetch((x0 + f32(dx)) * sx + hx,
                                    (y0 + f32(dy)) * sy + hy)
                pd = plane(pp, xf + f32(0.5) - pp.u, yf + f32(0.5) - pp.v)
                acc += (wx * wy) * (pd - pp.sl <= occ)
        return acc

    acc0 = np.zeros(q.shape[0], np.float32)
    acc1 = np.zeros(blend.size, np.float32)
    acc0f = np.zeros(fine.size, np.float32)
    acc1f = np.zeros(both.size, np.float32)
    for a, b, sa, sb in tab_f:
        px = p.u + fq * (tf * (a * p.ax + b * p.bx) + sa)
        py = p.v + fq * (tf * (a * p.ay + b * p.by) + sb)
        acc0 += tap(p, px, py, sx0, sy0)
        if blend.size:
            # the second lattice only where the cross-fade reaches it
            acc1 += tap(pb, px[blend], py[blend], sx1, sy1)
        if fine.size:
            # and the finer one across a strip's short side
            acc0f += tap(pf, px[fine], py[fine], sx0f, sy0f)
        if both.size:
            acc1f += tap(pbf, px[both], py[both], sx1f, sy1f)
    if fine.size:
        w2 = wa[fine]
        acc0[fine] = acc0[fine] * (f32(1.0) - w2) + acc0f * w2
    out = acc0 / fn
    if blend.size:
        if both.size:
            at = np.searchsorted(blend, both)
            w2 = wa[both]
            acc1[at] = acc1[at] * (f32(1.0) - w2) + acc1f * w2
        w1 = wb[blend]
        out[blend] = out[blend] * (f32(1.0) - w1) + (acc1 / fn) * w1
    return out


def lookup(sm, shape, P, N, ndl, slope, soft, samples, mask=None,
           stats=None):
    """Lit fraction in 0..1 per point for a SIZED lamp's shadow map.

    P is the lookup point (already stepped off the surface), N the
    shading normal, ndl = clamp(N.L, 0, 1), slope the per-point depth
    bias, soft the user Softness in texels, samples the lamp's resolved
    Samples. `mask` marks the points whose term is read; the rest report
    lit (the traced road's contract)."""
    n = P.shape[0]
    lit = np.ones(n, np.float32)
    P = np.asarray(P, np.float32)
    N = np.asarray(N, np.float32)
    ndl = np.asarray(ndl, np.float32)
    slope = np.broadcast_to(np.asarray(slope, np.float32), (n,))
    if mask is not None:
        sel = np.nonzero(mask)[0]
        if sel.size == 0:
            return lit
        if sel.size < n:
            lit[sel] = lookup(sm, shape, P[sel], N[sel], ndl[sel],
                              slope[sel], soft, samples, None, stats)
            return lit
    soft = max(float(soft), 0.0)
    F, B = tap_counts(samples)
    tab_s = search_table(F, B, soft)
    tab_f = filter_table(F, soft)
    dmin = map_d_min(sm)
    faces = getattr(sm, 'faces', None)
    rect = shape['kind'] == 'RECT'
    blocks = map_pyramid(sm)
    lay = pyr_layout((sm if faces is None else faces[0]).size)
    if faces is None:
        return _face(sm, _MapFetch(sm),
                     rect_axes(shape, face_axes(sm)) if rect else None,
                     P, N, ndl, slope, soft, shape, dmin, tab_s, tab_f,
                     stats, (blocks[0], lay, False))
    d = P - sm.origin[None, :]
    major = np.argmax(np.abs(d), axis=1)
    sign = d[np.arange(n), major] >= 0
    face = major * 2 + np.where(sign, 0, 1)
    for fi in range(6):
        m = np.nonzero(face == fi)[0]
        if m.size == 0:
            continue
        lit[m] = _face(faces[fi], _CubeFetch(faces, fi),
                       rect_axes(shape, face_axes(faces[fi])) if rect
                       else None,
                       P[m], N[m], ndl[m], slope[m], soft, shape, dmin,
                       tab_s, tab_f, stats, (blocks[fi], lay, True))
    return lit
