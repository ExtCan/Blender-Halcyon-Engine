"""The objects the era tested itself with, built rather than shipped.

Every renderer of the 1990s had the same three things on its demo reel: a
teapot, a Cornell box and a checkerboard. They are here as generators rather
than as mesh files, for the same reason the material templates are recipes
rather than saved node trees -- a generator has a resolution knob, weighs
nothing in the zip, and cannot drift out of step with the code that reads it.

bpy-free. Each function returns `(verts, faces)`: a list of (x, y, z) tuples
and a list of index tuples, in Blender's Z-up convention, ready for
`from_pydata`. Faces are quads wherever the surface is a quad, because a
period renderer would have been handed quads and because the seam pattern of a
quad mesh is part of how these objects read.

Nothing here imports the renderer either, so the shapes can be tested on their
own: closed, wound consistently, and the right size.
"""

import math

# ---------------------------------------------------------------- primitives


def _bezier3(p0, p1, p2, p3, t):
    """A cubic Bezier at t, on tuples of any dimension."""
    u = 1.0 - t
    a, b = u * u * u, 3.0 * u * u * t
    c, d = 3.0 * u * t * t, t * t * t
    return tuple(a * p0[i] + b * p1[i] + c * p2[i] + d * p3[i]
                 for i in range(len(p0)))


def _bezier3_tangent(p0, p1, p2, p3, t):
    u = 1.0 - t
    a, b, c = 3.0 * u * u, 6.0 * u * t, 3.0 * t * t
    return tuple(a * (p1[i] - p0[i]) + b * (p2[i] - p1[i]) + c * (p3[i] - p2[i])
                 for i in range(len(p0)))


def _grid_faces(rows, cols, base=0, wrap_cols=False, flip=False):
    """Quads over a (rows x cols) lattice of vertices.

    `flip` reverses the winding. Everything swept here walks *down* the profile
    while going anticlockwise around the axis, and those two together wind the
    quad clockwise as seen from outside -- which points every normal into the
    solid. So the sweeps ask for the flip, and a test asserts the result
    encloses a positive volume.
    """
    faces = []
    for r in range(rows - 1):
        last = cols if wrap_cols else cols - 1
        for c in range(last):
            c1 = (c + 1) % cols
            a = base + r * cols + c
            b = base + r * cols + c1
            d = base + (r + 1) * cols + c1
            e = base + (r + 1) * cols + c
            faces.append((a, e, d, b) if flip else (a, b, d, e))
    return faces


def _dedup_face(f):
    """Drop repeats at a pole, so a degenerate quad becomes a triangle."""
    out = []
    for i in f:
        if i not in out:
            out.append(i)
    return tuple(out) if len(out) >= 3 else None


# ------------------------------------------------------------ the teapot
#
# Newell modelled the teapot in 1975 as bicubic Bezier patches, and the
# rotational parts of it -- rim, body, lid, base -- are one profile swept
# around the vertical axis. That sweep is reproduced here from the profile,
# using the constant Newell used for a quarter circle: 0.56, not the 0.5523
# that actually approximates one. The teapot is very slightly not round, and
# has been in every render of it since; rounding that off would be tidying up
# somebody else's decision.
#
# The handle and the spout are not surfaces of revolution and are not the
# original control points either -- they are swept here to the published
# silhouette. Said plainly because it matters: this is the Utah teapot's shape
# and proportions, not a copy of the 1975 control-point file.

NEWELL_QUARTER = 0.56

#: (radius, height) control rows of the profile, one tuple per Bezier row.
#: Consecutive bands share their end row, which is what keeps the silhouette
#: continuous across a patch boundary.
TEAPOT_PROFILE = (
    ((1.4, 2.4), (1.3375, 2.53125), (1.4375, 2.53125), (1.5, 2.4)),      # rim
    ((1.5, 2.4), (1.75, 1.875), (2.0, 1.35), (2.0, 0.9)),                # body
    ((2.0, 0.9), (2.0, 0.45), (1.5, 0.225), (1.5, 0.15)),                # body
    ((1.5, 0.15), (1.5, 0.075), (1.425, 0.0), (0.0, 0.0)),               # base
    ((0.0, 3.15), (0.8, 3.15), (0.0, 2.85), (0.2, 2.7)),                 # knob
    # The lid is taken out to 1.4 to meet the rim's inner edge rather than
    # stopping at the original 1.3. That 0.1 was an annular slot straight into
    # the shell -- the original teapot is famously not watertight, and had no
    # bottom either until somebody added one -- and through it the culled and
    # unculled renders legitimately disagree. Closing it costs a tenth of a
    # unit of lid radius and buys a solid whose culled render is provably the
    # same picture.
    ((0.2, 2.7), (0.4, 2.55), (1.3, 2.55), (1.4, 2.4)),                  # lid
)

#: centre-line control points (x, z) and the half-widths across and through
#: the strap at each of them. The handle is a flattened strap and the spout a
#: tapering round tube, which is what tells them apart at a glance.
TEAPOT_HANDLE = (((-1.5, 2.10), (-3.45, 2.35), (-3.45, 0.95), (-1.55, 0.72)),
                 ((0.30, 0.16), (0.30, 0.20), (0.30, 0.20), (0.28, 0.13)))
TEAPOT_SPOUT = (((1.55, 0.95), (2.35, 1.05), (2.55, 2.05), (3.25, 2.35)),
                ((0.62, 0.62), (0.45, 0.45), (0.30, 0.30), (0.20, 0.14)))


def _revolve_band(rows, segments, steps):
    """Sweep one profile band around Z as a Bezier surface of revolution."""
    verts, ring_start = [], []
    for i in range(steps + 1):
        t = i / steps
        r, z = _bezier3(rows[0], rows[1], rows[2], rows[3], t)
        ring_start.append((r, z))
    for r, z in ring_start:
        for s in range(segments):
            a = 2.0 * math.pi * s / segments
            verts.append((r * math.cos(a), r * math.sin(a), z))
    return verts, ring_start


def _sweep_tube(centre, widths, segments, steps, flat_axis=True):
    """Sweep an ellipse along a cubic centre-line lying in the XZ plane.

    The cross-section stays upright rather than twisting with the curve: a
    strap moulded from a sheet does not roll over as it bends, and neither did
    the ones these packages modelled.
    """
    p0, p1, p2, p3 = centre
    w0, w1, w2, w3 = widths
    verts = []
    for i in range(steps + 1):
        t = i / steps
        cx, cz = _bezier3(p0, p1, p2, p3, t)
        wy, wr = _bezier3(w0, w1, w2, w3, t)
        tx, tz = _bezier3_tangent(p0, p1, p2, p3, t)
        ln = math.hypot(tx, tz) or 1.0
        # normal to the centre-line, in the XZ plane
        nx, nz = -tz / ln, tx / ln
        for s in range(segments):
            a = 2.0 * math.pi * s / segments
            ca, sa = math.cos(a), math.sin(a)
            verts.append((cx + nx * wr * ca, wy * sa, cz + nz * wr * ca))
    return verts


def _cap_ring(verts, faces, first, segments, inner_first):
    """Close one end of a swept tube with a fan to its own centre.

    A tube left open is not a hole you can see -- both of the teapot's end
    inside the body -- right up until backface culling is switched on, when the
    far interior is culled too and the pot is suddenly see-through. Capping
    makes the object closed, and a closed object is one whose culled render is
    provably identical to its unculled one.

    The winding is decided by *measuring* rather than by reasoning about which
    way the ring turns: the fan is built, its normal compared against the
    direction of the next ring in, and reversed if it points the wrong way. The
    sign argument that would replace this is exactly the kind that was wrong
    once already in this file.
    """
    ring = [first + i for i in range(segments)]
    def centroid(idx):
        return tuple(sum(verts[i][k] for i in idx) / len(idx) for k in range(3))

    cx, cy, cz = centroid(ring)
    inward = centroid([inner_first + i for i in range(segments)])
    centre = len(verts)
    verts.append((cx, cy, cz))

    a, b = verts[ring[0]], verts[ring[1]]
    u = (a[0] - cx, a[1] - cy, a[2] - cz)
    v = (b[0] - cx, b[1] - cy, b[2] - cz)
    nx = u[1] * v[2] - u[2] * v[1]
    ny = u[2] * v[0] - u[0] * v[2]
    nz = u[0] * v[1] - u[1] * v[0]
    to_inside = (inward[0] - cx, inward[1] - cy, inward[2] - cz)
    flip = (nx * to_inside[0] + ny * to_inside[1] + nz * to_inside[2]) > 0.0

    for i in range(segments):
        p, q = ring[i], ring[(i + 1) % segments]
        faces.append((centre, q, p) if flip else (centre, p, q))


def utah_teapot(steps=8, segments=16):
    """The Utah teapot, swept from its profile at the requested resolution.

    `steps` subdivides each Bezier band along the profile and `segments` goes
    around, so the default is a little over 2,000 quads -- about what a 1996
    machine would have been given.
    """
    steps = max(int(steps), 2)
    segments = max(int(segments), 4)
    verts, faces = [], []
    for rows in TEAPOT_PROFILE:
        base = len(verts)
        band, _ = _revolve_band(rows, segments, steps)
        verts.extend(band)
        for f in _grid_faces(steps + 1, segments, base, wrap_cols=True,
                             flip=True):
            g = _dedup_face(f)
            if g:
                faces.append(g)
    for centre, widths in (TEAPOT_HANDLE, TEAPOT_SPOUT):
        base = len(verts)
        rings = steps * 2 + 1
        verts.extend(_sweep_tube(centre, widths, segments, steps * 2))
        faces.extend(_grid_faces(rings, segments, base, wrap_cols=True,
                                 flip=True))
        _cap_ring(verts, faces, base, segments, base + segments)
        _cap_ring(verts, faces, base + (rings - 1) * segments, segments,
                  base + (rings - 2) * segments)
    return verts, faces


# --------------------------------------------------------- the Cornell box
#
# The Cornell box is a *measurement*, not a model: the original scene was
# built, photographed and reported in millimetres so that a renderer's output
# could be compared against a photograph of the real thing. Those millimetres
# are below, converted to metres and to Z-up, and nothing about them is
# stylistic -- the walls really are not square to each other, and the boxes
# really are rotated by those odd angles.

#: (x, y, z) in the original right-handed Y-up millimetre frame
_CORNELL_MM = {
    'floor': ((552.8, 0.0, 0.0), (0.0, 0.0, 0.0),
              (0.0, 0.0, 559.2), (549.6, 0.0, 559.2)),
    'ceiling': ((556.0, 548.8, 0.0), (556.0, 548.8, 559.2),
                (0.0, 548.8, 559.2), (0.0, 548.8, 0.0)),
    'back': ((549.6, 0.0, 559.2), (0.0, 0.0, 559.2),
             (0.0, 548.8, 559.2), (556.0, 548.8, 559.2)),
    'right': ((0.0, 0.0, 559.2), (0.0, 0.0, 0.0),
              (0.0, 548.8, 0.0), (0.0, 548.8, 559.2)),
    'left': ((552.8, 0.0, 0.0), (549.6, 0.0, 559.2),
             (556.0, 548.8, 559.2), (556.0, 548.8, 0.0)),
    'light': ((343.0, 548.7, 227.0), (343.0, 548.7, 332.0),
              (213.0, 548.7, 332.0), (213.0, 548.7, 227.0)),
}

_CORNELL_SHORT = (
    ((130.0, 165.0, 65.0), (82.0, 165.0, 225.0),
     (240.0, 165.0, 272.0), (290.0, 165.0, 114.0)),      # top
    ((290.0, 0.0, 114.0), (290.0, 165.0, 114.0),
     (240.0, 165.0, 272.0), (240.0, 0.0, 272.0)),
    ((130.0, 0.0, 65.0), (130.0, 165.0, 65.0),
     (290.0, 165.0, 114.0), (290.0, 0.0, 114.0)),
    ((82.0, 0.0, 225.0), (82.0, 165.0, 225.0),
     (130.0, 165.0, 65.0), (130.0, 0.0, 65.0)),
    ((240.0, 0.0, 272.0), (240.0, 165.0, 272.0),
     (82.0, 165.0, 225.0), (82.0, 0.0, 225.0)),
)

_CORNELL_TALL = (
    ((423.0, 330.0, 247.0), (265.0, 330.0, 296.0),
     (314.0, 330.0, 456.0), (472.0, 330.0, 406.0)),      # top
    ((423.0, 0.0, 247.0), (423.0, 330.0, 247.0),
     (472.0, 330.0, 406.0), (472.0, 0.0, 406.0)),
    ((472.0, 0.0, 406.0), (472.0, 330.0, 406.0),
     (314.0, 330.0, 456.0), (314.0, 0.0, 456.0)),
    ((314.0, 0.0, 456.0), (314.0, 330.0, 456.0),
     (265.0, 330.0, 296.0), (265.0, 0.0, 296.0)),
    ((265.0, 0.0, 296.0), (265.0, 330.0, 296.0),
     (423.0, 330.0, 247.0), (423.0, 0.0, 247.0)),
)

#: which group each quad belongs to, in the order cornell_box() emits them
CORNELL_GROUPS = ('white', 'white', 'white', 'green', 'red', 'light',
                  'short', 'tall')


def cornell_box(scale=1.0 / 552.8, centred=True):
    """The Cornell box at its published dimensions.

    Returns `(verts, faces, groups)`, where `groups` names the material each
    face wants: white, red, green, light, short, tall. The default scale puts
    the box a shade over one Blender unit across, and `centred` moves the
    origin to the middle of the floor so it drops onto the world origin the
    way an object added from a menu should.
    """
    ox = 552.8 * 0.5 if centred else 0.0
    oz = 559.2 * 0.5 if centred else 0.0

    def conv(p):
        # The original is Y-up right-handed. Sending it to Z-up as (x, z, y)
        # is the obvious move and it is wrong: that map has determinant -1, so
        # the box arrives mirrored and the red and green walls swap sides.
        # Negating the new Y keeps the handedness, and leaves the open face
        # toward +Y, where a camera naturally sits.
        return ((p[0] - ox) * scale, -(p[2] - oz) * scale, p[1] * scale)

    verts, faces, groups = [], [], []

    def quad(pts, group):
        base = len(verts)
        verts.extend(conv(p) for p in pts)
        # That map has determinant +1, so the winding survives it untouched and
        # every normal goes on pointing into the box the way the published
        # data has them. A test asserts exactly that, because it is the kind of
        # thing that is invisible until a one-sided material is put on a wall.
        faces.append((base, base + 1, base + 2, base + 3))
        groups.append(group)

    for key, group in (('floor', 'white'), ('ceiling', 'white'),
                       ('back', 'white'), ('right', 'green'),
                       ('left', 'red'), ('light', 'light')):
        quad(_CORNELL_MM[key], group)
    for pts in _CORNELL_SHORT:
        quad(pts, 'short')
    for pts in _CORNELL_TALL:
        quad(pts, 'tall')
    return verts, faces, groups


# ------------------------------------------------------------- the teaset
#
# Newell modelled a whole tea service, and the cup, saucer and spoon are the
# rest of it. They are lathed here from their profiles, which is how they were
# made: a surface of revolution is one curve and a sweep, and every modeller of
# the period had that tool before it had anything else.
#
# One rule governs every profile in this file, and getting it wrong is silent:
# **a profile must be walked with the solid on its right**, which for the
# teapot means down the outside. Walk it the other way and the sweep is wound
# inside out -- invisible with the z-buffer, invisible in a shaded render, and
# then the whole object disappears the moment backface culling is switched on
# and you see its far interior instead. These three read more naturally written
# bottom-up, so they are written that way and reversed before sweeping.

CUP_PROFILE = (
    ((0.0, 0.0), (0.35, 0.0), (0.62, 0.02), (0.66, 0.10)),      # base
    ((0.66, 0.10), (0.70, 0.30), (0.86, 0.55), (0.92, 0.78)),   # outer wall
    ((0.92, 0.78), (0.90, 0.80), (0.88, 0.80), (0.86, 0.78)),   # lip
    ((0.86, 0.78), (0.80, 0.55), (0.64, 0.30), (0.60, 0.12)),   # inner wall
    ((0.60, 0.12), (0.40, 0.10), (0.20, 0.09), (0.0, 0.09)),    # inside floor
)

SAUCER_PROFILE = (
    ((0.0, 0.0), (0.6, 0.0), (1.2, 0.0), (1.55, 0.02)),         # underside
    ((1.55, 0.02), (1.66, 0.06), (1.70, 0.13), (1.68, 0.17)),   # rim
    ((1.68, 0.17), (1.30, 0.13), (0.80, 0.08), (0.0, 0.07)),    # well
)

#: the spoon: a swept handle and a lathed bowl squashed along Y
SPOON_HANDLE = (((0.0, 0.06), (0.55, 0.10), (1.15, 0.16), (1.55, 0.30)),
                ((0.055, 0.030), (0.050, 0.026), (0.055, 0.030),
                 (0.075, 0.022)))
SPOON_BOWL = (((0.0, 0.0), (0.14, 0.0), (0.26, 0.03), (0.30, 0.09)),
              ((0.30, 0.09), (0.26, 0.10), (0.16, 0.075), (0.0, 0.065)))


def _reversed_profile(profile):
    """The same curve walked the other way: bands back to front, rows too."""
    return tuple(tuple(reversed(rows)) for rows in reversed(profile))


def _lathe(profile, steps, segments):
    verts, faces = [], []
    for rows in profile:
        base = len(verts)
        band, _ = _revolve_band(rows, segments, steps)
        verts.extend(band)
        for f in _grid_faces(steps + 1, segments, base, wrap_cols=True,
                             flip=True):
            g = _dedup_face(f)
            if g:
                faces.append(g)
    return verts, faces


def teacup(steps=6, segments=16):
    return _lathe(_reversed_profile(CUP_PROFILE), max(int(steps), 2),
                  max(int(segments), 4))


def saucer(steps=5, segments=16):
    return _lathe(_reversed_profile(SAUCER_PROFILE), max(int(steps), 2),
                  max(int(segments), 4))


def teaspoon(steps=6, segments=12, squash=0.45):
    """A lathed bowl flattened across Y, with a swept handle out of its side."""
    steps, segments = max(int(steps), 2), max(int(segments), 4)
    verts, faces = _lathe(_reversed_profile(SPOON_BOWL), steps, segments)
    verts = [(x, y * squash, z * 0.75) for x, y, z in verts]
    base = len(verts)
    rings = steps * 2 + 1
    verts.extend(_sweep_tube(SPOON_HANDLE[0], SPOON_HANDLE[1], segments,
                             steps * 2))
    faces.extend(_grid_faces(rings, segments, base, wrap_cols=True, flip=True))
    _cap_ring(verts, faces, base, segments, base + segments)
    _cap_ring(verts, faces, base + (rings - 1) * segments, segments,
              base + (rings - 2) * segments)
    return verts, faces


#: what the teaset lays out, and where. (name, builder, translation, scale)
TEASET_LAYOUT = (
    ('Teapot', 'teapot', (0.0, 0.0, 0.0), 1.0),
    ('Saucer', 'saucer', (4.2, -1.4, 0.0), 1.0),
    ('Teacup', 'teacup', (4.2, -1.4, 0.12), 1.0),
    ('Teaspoon', 'teaspoon', (4.0, 0.9, 0.08), 1.0),
)


# ------------------------------------------------------ the checkerboard


def checker_plane(size=20.0, divisions=8):
    """A checkerboard as geometry: alternating faces, not a texture.

    Every ray tracer's first picture stood its spheres on one of these, and it
    was a real chequered surface rather than an image -- which is why the
    squares stayed square right out to the horizon instead of swimming. The
    face groups alternate so two materials can be assigned, and that works
    under flat shading, in a wireframe and on a machine with four colours.
    """
    divisions = max(int(divisions), 1)
    step = float(size) / divisions
    half = float(size) * 0.5
    verts = [(c * step - half, r * step - half, 0.0)
             for r in range(divisions + 1) for c in range(divisions + 1)]
    faces, groups = [], []
    for r in range(divisions):
        for c in range(divisions):
            a = r * (divisions + 1) + c
            faces.append((a, a + 1, a + divisions + 2, a + divisions + 1))
            groups.append((r + c) % 2)
    return verts, faces, groups


# ---------------------------------------------------------------- helpers


def bounds(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))


def signed_volume(verts, faces):
    """Six times the volume the faces enclose, signed by their winding.

    Positive means the normals point outward. It is exact for these shapes even
    though the band seams are topologically open, because the two rings at a
    seam are at identical positions -- the divergence theorem does not care
    that they are different vertices.
    """
    V = verts
    total = 0.0
    for f in faces:
        for k in range(1, len(f) - 1):
            a, b, c = V[f[0]], V[f[k]], V[f[k + 1]]
            total += (a[0] * (b[1] * c[2] - b[2] * c[1])
                      - a[1] * (b[0] * c[2] - b[2] * c[0])
                      + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0
    return total


def is_manifoldish(verts, faces):
    """Every edge used once or twice, and no face indexing off the end.

    Not a full manifold test -- a swept tube is open at its ends by design --
    but it catches the mistakes a generator actually makes: an off-by-one in a
    lattice, a wrap that does not, a degenerate quad left in.
    """
    n = len(verts)
    edges = {}
    for f in faces:
        if len(f) < 3 or len(set(f)) != len(f):
            return False, f'degenerate face {f}'
        for i, a in enumerate(f):
            b = f[(i + 1) % len(f)]
            if a >= n or b >= n or a < 0 or b < 0:
                return False, f'index out of range in {f}'
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1
    bad = [e for e, c in edges.items() if c > 2]
    if bad:
        return False, f'{len(bad)} edge(s) shared by more than two faces'
    return True, f'{len(verts)} verts, {len(faces)} faces'


# ------------------------------------------------------------ the terrain


def _terrain_grid(size, divisions):
    """R253: the (divisions+1)^2 lattice -- the R201 statements, lifted
    out of `terrain()` unchanged so the picture, erosion and export
    roads share one grid with the fractal road."""
    import numpy as np

    divisions = max(int(divisions), 2)
    n1 = divisions + 1
    size = float(size)
    step = size / divisions
    half = size * 0.5

    cc = np.arange(n1, dtype=np.float32) * step - half
    gx, gy = np.meshgrid(cc, cc, indexing='xy')
    xs = gx.ravel()
    ys = gy.ravel()
    return xs, ys, n1, half, step


def _terrain_domain(xs, ys, feature_scale, seed):
    """R253: the seeded noise domain (R201 statements, verbatim)."""
    import numpy as np

    fs = max(float(feature_scale), 1e-3)
    # the seed walks the noise DOMAIN: two Wang-hashed offsets plus a
    # third-axis shelf, so seeds are uncorrelated, not slid copies
    from .patterns import _wang01p
    s = np.uint32(int(seed) & 0xFFFFFFFF)
    ox = float(_wang01p(np.array([s ^ np.uint32(0x2545F491)],
                                 np.uint32))[0]) * 512.0
    oy = float(_wang01p(np.array([s ^ np.uint32(0x9E3779B9)],
                                 np.uint32))[0]) * 512.0
    oz = float(int(seed) % 97) * 7.13
    p = np.stack([xs / fs + ox, ys / fs + oy,
                  np.full_like(xs, oz)], axis=1).astype(np.float32)
    return p, fs


def _smooth01(t):
    """The smoothstep the landforms shape with (clipped, float-preserving)."""
    import numpy as np
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _terrain_base(kind, p, xs, ys, half, fs, octs, gain, rg, direction,
                  heightmap=None, heightmap_scale=1.0,
                  heightmap_offset=0.0, heightmap_channel='LUMA',
                  heightmap_sampling='BILINEAR', heightmap_tile=False):
    """R253: the landform field BEFORE the shaping stages.

    The seven R201/R202 branches are the original statements in the
    original order (the pinned hashes prove it); the four R253 kinds
    are appended after them and never touch the chain above. Returns
    the flat (n1*n1,) float32 field, already clipped to >= 0.
    """
    import numpy as np

    from . import patterns as PT

    _smooth = _smooth01
    n = xs.shape[0]

    if kind == 'HILLS':
        # R202: pure rolling fBm, softened once more -- pastureland
        h = _smooth(PT.fbm(p, octaves=octs, gain=gain))
    elif kind == 'CANYON':
        # a high mesa with winding cuts carved OUT of it: the ridged
        # fractal's crest lines widened into channels, strata texture
        # over the tops and sharper walls down the cuts
        mesa = (0.55 + 0.45 * _smooth(
            PT.fbm(p * 0.5 + 3.7, octaves=max(octs - 2, 1),
                   gain=gain))) \
            * (0.92 + 0.16 * (PT.fbm(p * 3.1 + 17.9, octaves=octs,
                                     gain=gain) - 0.5))
        r = PT.ridged(p, octaves=octs, gain=gain)
        cut = _smooth((r - 0.58) / 0.16)
        wash = (PT.fbm(p * 2.2 + 71.3, octaves=octs, gain=gain)
                - 0.5) * 0.08
        h = np.clip(mesa * (1.0 - 0.9 * cut) + wash * cut, 0.0, None)
    elif kind == 'DUNES':
        # directional sand waves: a triangle wave across the wind
        # axis, its crests wobbled and its amplitude breathing with
        # low-frequency noise. `direction` turns the wind
        ca, sa = float(np.cos(direction)), float(np.sin(direction))
        along = (xs * ca + ys * sa) / fs
        wob = (PT.fbm(p * 0.7 + 11.3, octaves=max(octs - 2, 1),
                      gain=gain) - 0.5) * 2.2
        tphase = np.mod(along * 1.5 + wob, 1.0)
        tri = 1.0 - np.abs(2.0 * tphase - 1.0)
        env = 0.35 + 0.65 * PT.fbm(p * 0.35 + 27.1,
                                   octaves=max(octs - 3, 1), gain=gain)
        ripple = (PT.fbm(p * 4.0 + 43.9, octaves=max(octs - 3, 1),
                         gain=gain) - 0.5) * 0.06
        h = np.power(tri, 1.5) * env * 0.75 + ripple
    elif kind == 'CRATER':
        # one great ring -- but a REAL one: the rim radius wanders
        # with angle noise (no impact ever cut a compass circle), its
        # height varies along the ring, ejecta roughens the apron and
        # rubble breaks the floor
        rr = np.sqrt(xs * xs + ys * ys) / max(half, 1e-6)
        rwob = (PT.fbm(p * 1.5 + 53.7, octaves=max(octs - 2, 1),
                       gain=gain) - 0.5)
        rw = rr * (1.0 + rwob * 0.22)
        rim_h = 0.55 + 0.45 * PT.fbm(p * 2.0 + 7.9,
                                     octaves=max(octs - 2, 1),
                                     gain=gain)
        rim = np.exp(-np.square((rw - 0.55) / 0.075)) * rim_h
        floor_m = _smooth((0.48 - rw) / 0.18)
        base = _smooth((rw - 0.5) / 0.5) * 0.20
        ejecta = (PT.fbm(p * 1.2 + 17.3, octaves=octs, gain=gain)
                  - 0.5) * 0.22 * _smooth((rw - 0.45) / 0.3)
        rubble = (PT.fbm(p * 2.3 + 91.1, octaves=octs, gain=gain)
                  - 0.5) * 0.14 * floor_m
        h = np.clip(rim + base + ejecta + rubble
                    - floor_m * 0.34 + 0.20, 0.0, None)
    elif kind == 'VOLCANO':
        # a stratovolcano: steep upper cone, deep bitten caldera,
        # ridged lava flanks and radial gullying worn down the sides
        rr = np.sqrt(xs * xs + ys * ys) / max(half, 1e-6)
        cone = np.power(np.clip(1.0 - rr * 1.05, 0.0, None), 1.4)
        caldera = np.exp(-np.square(rr / 0.13)) * 0.8
        th = np.arctan2(ys, xs)
        gully = np.abs(np.sin(th * 7.0
                              + (PT.fbm(p * 0.9 + 31.7,
                                        octaves=max(octs - 3, 1),
                                        gain=gain) - 0.5) * 6.0))
        flank = (PT.ridged(p * 1.3, octaves=octs, gain=gain)
                 - 0.5) * 0.28
        h = np.clip(cone * (1.0 - 0.16 * gully) - caldera, 0.0, None) \
            * 1.05 + flank * np.clip(cone, 0.0, 1.0)
        h = np.clip(h, 0.0, None)
    elif kind == 'PLATEAU':
        # table-land: fBm pushed through a hard shoulder, flat on top
        f = PT.fbm(p, octaves=octs, gain=gain)
        top = _smooth((f - 0.45) / 0.12)
        h = top * 0.72 + f * 0.18
    elif kind == 'PICTURE':
        # R253: the Bryce 'Picture' terrain -- the heights ARE an
        # image. A (n1,n1) field is taken as it is; raw (H,W[,4])
        # pixels are sampled onto the grid first (bottom row first =
        # the grid's -Y row 0, so the picture lies on the ground as
        # seen from above). No image at all is a flat zero plane --
        # the operator reports the missing picture by name, the
        # geometry never raises
        n1 = int(round(n ** 0.5))
        if heightmap is None:
            h = np.zeros(n, np.float32)
        else:
            hm = np.asarray(heightmap, np.float32)
            if hm.shape != (n1, n1) and hm.size != n:
                hm = sample_heightmap(hm, n1 - 1, heightmap_channel,
                                      heightmap_sampling, heightmap_tile)
            h = hm.reshape(-1).astype(np.float32)
            h = np.float32(heightmap_scale) * h \
                + np.float32(heightmap_offset)
    elif kind == 'ARCHIPELAGO':
        # R253: many islands from a flat sea -- fBm pushed through a
        # soft threshold at 0.55 (the sea is exactly 0.0 where the
        # noise sits below it), each island crowned by a secondary
        # ridged crest that only grows where land stands
        f = PT.fbm(p * 0.8 + 5.3, octaves=octs, gain=gain)
        land = _smooth((f - 0.55) / 0.12)
        crest = PT.ridged(p * 1.7 + 23.1, octaves=octs, gain=gain)
        h = land * (0.45 + 0.55 * crest * land)
    elif kind == 'BADLANDS':
        # R253: eroded strata -- a ridged fractal under a low-frequency
        # fBm envelope, pre-terraced into nine shelves with a smoothed
        # riser, then three talus passes so the risers crumble the way
        # the Bryce 'Erode' button crumbled a terraced mesa
        r = PT.ridged(p * 1.1 + 41.7, octaves=octs, gain=gain)
        env = 0.4 + 0.6 * _smooth(PT.fbm(p * 0.4 + 13.1,
                                         octaves=max(octs - 2, 1),
                                         gain=gain))
        h = np.clip(r * env * 1.6, 0.0, 1.0)
        tq = h * 9.0
        fl = np.floor(tq)
        fr = tq - fl
        h = (fl + fr * fr * fr * (fr * (fr * 6.0 - 15.0) + 10.0)) / 9.0
        n1 = int(round(n ** 0.5))
        h = thermal_erosion(h.reshape(n1, n1), 3, 0.02, 0.5).reshape(-1)
    elif kind == 'VALLEY':
        # R253: a river valley -- rolling fBm with ONE channel carved
        # along `direction`: a smoothstep trough whose centre line
        # wanders with a low-frequency noise of the along-valley
        # coordinate only, its floor flattened to a constant shelf
        ca, sa = float(np.cos(direction)), float(np.sin(direction))
        along = (xs * ca + ys * sa) / fs
        across = (ys * ca - xs * sa) / max(half, 1e-6)
        q = np.stack([along * 0.45 + 61.3, np.full_like(along, 7.7),
                      p[:, 2]], axis=1).astype(np.float32)
        centre = (PT.fbm(q, octaves=max(octs - 3, 1), gain=gain)
                  - 0.5) * 0.9
        dist = np.abs(across - centre)
        trough = _smooth((dist - 0.10) / 0.22)
        base = 0.25 + 0.75 * _smooth(PT.fbm(p * 0.9 + 3.3, octaves=octs,
                                            gain=gain))
        h = 0.08 + np.clip(base - 0.08, 0.0, None) * trough
    else:                                       # MOUNTAIN, the R201 road
        h = PT.fbm(p, octaves=octs, gain=gain)
        if rg > 0.0:
            h = h * (1.0 - rg) \
                + PT.ridged(p, octaves=octs, gain=gain) * rg
    h = np.clip(h, 0.0, None)
    return h


# ---------------------------------------------- R253: the terrain stages
#
# Every helper below is a pure NumPy function on a (n1, n1) float32
# field, bpy-free, and each is guarded in `terrain_heights` by its own
# default so a 1.91 terrain executes exactly the 1.91 statements. The
# dial names follow the Bryce Terrain Editor (Picture, Erode, Smoothing,
# Raise / Zero Edges, Spikes, Symmetric); the erosion models follow
# Musgrave, Kolb & Mace, 'The Synthesis and Rendering of Eroded Fractal
# Terrains' (SIGGRAPH 1989), on a 4-neighbour grid.


def _as2d(h):
    """(n1*n1,) -> (n1, n1) view; a reshape, so bitwise."""
    n1 = int(round(h.shape[0] ** 0.5))
    return h.reshape(n1, n1)


def _luma(px):
    """Rec.709 luma of (..., 3+) float32 pixels, float32 throughout."""
    import numpy as np
    return (px[..., 0] * np.float32(0.2126)
            + px[..., 1] * np.float32(0.7152)) \
        + px[..., 2] * np.float32(0.0722)


def sample_heightmap(pixels, divisions, channel='LUMA',
                     sampling='BILINEAR', tile=False):
    """R253: an (H,W,4) or (H,W) image -> the (n1,n1) float32 height
    field the grid wants.

    `channel` LUMA takes the Rec.709 weights (0.2126 / 0.7152 /
    0.0722); 'R' / 'G' / 'B' / 'A' take one plane (an image without an
    alpha plane reads A as 1). BILINEAR samples at u = i / divisions
    over [0, W-1] x [0, H-1] with edge clamp -- so a (n1,n1) picture at
    `divisions` = n1-1 lands on the grid bitwise; NEAREST rounds to the
    pixel. `tile` samples over [0, W) x [0, H) with modulo instead, so
    the grid's last row and column wrap to the first (a seamless
    picture makes a seamless terrain). Rows follow Blender's pixel
    buffer: row 0 is the BOTTOM of the picture and the grid's -Y edge.
    """
    import numpy as np

    px = np.asarray(pixels, np.float32)
    if px.ndim == 2:
        field = px
    else:
        ch = str(channel or 'LUMA').upper()
        if ch == 'LUMA':
            field = _luma(px)
        else:
            idx = {'R': 0, 'G': 1, 'B': 2, 'A': 3}.get(ch, 0)
            if idx < px.shape[-1]:
                field = px[..., idx]
            else:
                field = np.ones(px.shape[:2], np.float32)
    field = np.ascontiguousarray(field, np.float32)
    H, W = field.shape
    divisions = max(int(divisions), 2)
    n1 = divisions + 1
    i = np.arange(n1, dtype=np.float64)
    nearest = str(sampling or 'BILINEAR').upper() == 'NEAREST'
    if tile:
        u = i / divisions * W
        v = i / divisions * H
        if nearest:
            ix = np.rint(u).astype(np.int64) % W
            iy = np.rint(v).astype(np.int64) % H
            return field[iy[:, None], ix[None, :]]
        x0 = np.floor(u)
        y0 = np.floor(v)
        fx = (u - x0).astype(np.float32)
        fy = (v - y0).astype(np.float32)
        x0 = x0.astype(np.int64) % W
        y0 = y0.astype(np.int64) % H
        x1 = (x0 + 1) % W
        y1 = (y0 + 1) % H
    else:
        u = i / divisions * (W - 1)
        v = i / divisions * (H - 1)
        if nearest:
            ix = np.clip(np.rint(u), 0, W - 1).astype(np.int64)
            iy = np.clip(np.rint(v), 0, H - 1).astype(np.int64)
            return field[iy[:, None], ix[None, :]]
        x0 = np.floor(u)
        y0 = np.floor(v)
        fx = (u - x0).astype(np.float32)
        fy = (v - y0).astype(np.float32)
        x0 = np.clip(x0, 0, W - 1).astype(np.int64)
        y0 = np.clip(y0, 0, H - 1).astype(np.int64)
        x1 = np.minimum(x0 + 1, W - 1)
        y1 = np.minimum(y0 + 1, H - 1)
    # lerp as a + (b - a) * f: at f == 0 the result IS a, bitwise, so
    # a picture the size of the grid comes through untouched
    c00 = field[y0[:, None], x0[None, :]]
    c10 = field[y0[:, None], x1[None, :]]
    c01 = field[y1[:, None], x0[None, :]]
    c11 = field[y1[:, None], x1[None, :]]
    fxr = fx[None, :]
    fyr = fy[:, None]
    top = c00 + (c10 - c00) * fxr
    bot = c01 + (c11 - c01) * fxr
    return (top + (bot - top) * fyr).astype(np.float32)


def heights_to_pixels(h01):
    """R253: the export road -- a (n1,n1) field as (n1,n1,4) float32
    pixels, bottom row first, RGB = height, A = 1. The inverse of
    `sample_heightmap(..., 'R', 'NEAREST')` at the grid's own size,
    bitwise, so Export Heightmap -> Picture round-trips."""
    import numpy as np
    h = np.asarray(h01, np.float32)
    out = np.empty(h.shape + (4,), np.float32)
    out[..., 0] = h
    out[..., 1] = h
    out[..., 2] = h
    out[..., 3] = 1.0
    return out


def fractal_detail(h, p, octaves, amount, gain=0.5):
    """R253: fine fBm octaves on ANY base -- a picture gets procedural
    detail, a landform gets grit. Adds amount * (fbm(8p) - 0.5), then
    keeps the field at or above zero."""
    import numpy as np
    from . import patterns as PT
    octs = max(int(octaves), 1)
    d = (PT.fbm(p * 8.0, octaves=octs, gain=gain) - 0.5) * np.float32(amount)
    return np.clip(h + d, 0.0, None).astype(np.float32)


def symmetrize_heights(h, mode):
    """R253: Bryce's 'Symmetric' brush as a whole-field stage.

    'X' mirrors the -X half onto the +X half (column i onto n1-1-i),
    'Y' rows, 'XY' both, 'RADIAL' averages the four 90-degree
    rotations -- summed as (h + r180) + (r90 + r270), which is its OWN
    90-degree rotation bitwise because IEEE addition commutes."""
    import numpy as np
    mode = str(mode or 'NONE').upper()
    if mode == 'NONE':
        return h
    h = np.array(h, np.float32, copy=True)
    n1 = h.shape[1]
    if mode in ('X', 'XY'):
        half = n1 // 2
        h[:, n1 - half:] = h[:, :half][:, ::-1]
    if mode in ('Y', 'XY'):
        half = h.shape[0] // 2
        h[h.shape[0] - half:, :] = h[:half, :][::-1, :]
    if mode == 'RADIAL':
        if h.shape[0] != h.shape[1]:
            return h
        r90 = np.rot90(h, 1)
        r180 = np.rot90(h, 2)
        r270 = np.rot90(h, 3)
        h = ((h + r180) + (r90 + r270)) * np.float32(0.25)
    return h.astype(np.float32)


def _shift_into(dst, src, drow, dcol):
    """Add `src` into `dst` shifted by (drow, dcol); what would leave
    the grid stays put (the caller never budgets it)."""
    R, C = dst.shape
    rs = slice(max(drow, 0), R + min(drow, 0))
    rd = slice(max(-drow, 0), R + min(-drow, 0))
    cs = slice(max(dcol, 0), C + min(dcol, 0))
    cd = slice(max(-dcol, 0), C + min(-dcol, 0))
    dst[rs, cs] += src[rd, cd]


#: the four grid neighbours as (row shift, column shift)
_NB4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def thermal_erosion(h, iterations, talus=0.02, rate=0.5):
    """R253: talus slide (Musgrave 1989's thermal weathering).

    Per iteration, every cell looks at its four neighbours; where it
    stands more than `talus` above one, that excess is a drop. Each
    drop moves rate/4 of itself to that neighbour. The quarter is the
    bound at which a cell fed by all four neighbours can never overshoot
    them and a cell feeding all four can never undershoot them, so for
    any rate in [0, 1]: mass is conserved (what leaves a cell lands in
    a neighbour -- edges never spill), the field's maximum never rises
    and its minimum never falls. Material that would leave the grid is
    simply not budgeted. float32 in and out, float64 inside; no
    reductions over the field, so the same seed slides the same way on
    every machine."""
    import numpy as np
    iters = int(iterations)
    if iters <= 0:
        return h
    hh = np.asarray(h, np.float64).copy()
    c = min(max(float(rate), 0.0), 1.0) * 0.25
    t = float(talus)
    for _ in range(iters):
        moved = np.zeros_like(hh)
        gain = np.zeros_like(hh)
        for drow, dcol in _NB4:
            nb = np.full_like(hh, np.inf)
            # the neighbour at (row+drow, col+dcol), seen from each cell
            R, C = hh.shape
            rs = slice(max(drow, 0), R + min(drow, 0))
            rd = slice(max(-drow, 0), R + min(-drow, 0))
            cs = slice(max(dcol, 0), C + min(dcol, 0))
            cd = slice(max(-dcol, 0), C + min(-dcol, 0))
            nb[rd, cd] = hh[rs, cs]
            drop = hh - nb - t
            drop = np.where(np.isfinite(drop) & (drop > 0.0), drop, 0.0)
            flow = drop * c
            moved += flow
            _shift_into(gain, flow, drow, dcol)
        hh = hh - moved + gain
    return hh.astype(np.float32)


def hydraulic_erosion(h, iterations, rain=0.01, solubility=0.1,
                      evaporation=0.5, capacity=0.05):
    """R253: the water-table approximation of Musgrave 1989's hydraulic
    erosion. Each pass: rain falls on every cell; the water dissolves
    `solubility` of itself from the rock (capped at the rock there)
    into sediment; water and its sediment flow to lower neighbours by
    the drop in water-table height (rock + water), the quarter rule
    of the thermal pass again; `evaporation` of the water leaves, and
    sediment above `capacity` of the remaining water is deposited back
    as rock. When the passes end the sediment still in suspension
    settles, so rock + sediment is conserved throughout (to rounding)
    and the heights come back as float32 with no water on them."""
    import numpy as np
    iters = int(iterations)
    if iters <= 0:
        return h
    rock = np.asarray(h, np.float64).copy()
    water = np.zeros_like(rock)
    sed = np.zeros_like(rock)
    rain = max(float(rain), 0.0)
    sol = min(max(float(solubility), 0.0), 1.0)
    evap = min(max(float(evaporation), 0.0), 1.0)
    cap = max(float(capacity), 0.0)
    R, C = rock.shape
    for _ in range(iters):
        water = water + rain
        dissolve = np.minimum(sol * water, rock)
        rock = rock - dissolve
        sed = sed + dissolve
        table = rock + water
        w_out = np.zeros_like(rock)
        s_out = np.zeros_like(rock)
        w_in = np.zeros_like(rock)
        s_in = np.zeros_like(rock)
        frac = np.where(water > 0.0, sed / np.where(water > 0.0, water, 1.0),
                        0.0)
        for drow, dcol in _NB4:
            nb = np.full_like(table, np.inf)
            rs = slice(max(drow, 0), R + min(drow, 0))
            rd = slice(max(-drow, 0), R + min(-drow, 0))
            cs = slice(max(dcol, 0), C + min(dcol, 0))
            cd = slice(max(-dcol, 0), C + min(-dcol, 0))
            nb[rd, cd] = table[rs, cs]
            drop = table - nb
            drop = np.where(np.isfinite(drop) & (drop > 0.0), drop, 0.0)
            wf = np.minimum(drop * 0.25, water * 0.25)
            sf = wf * frac
            w_out += wf
            s_out += sf
            _shift_into(w_in, wf, drow, dcol)
            _shift_into(s_in, sf, drow, dcol)
        water = water - w_out + w_in
        sed = sed - s_out + s_in
        water = water * (1.0 - evap)
        excess = np.maximum(sed - cap * water, 0.0)
        rock = rock + excess
        sed = sed - excess
    rock = rock + sed
    return rock.astype(np.float32)


def smooth_heights(h, passes):
    """R253: Bryce's Smoothing -- the 3x3 cross kernel (4,1,1,1,1)/8
    with edge padding, `passes` times. A convex blend, so the result is
    clipped into the input's own [min, max] (rounding can never push a
    shelf above the peak it averages)."""
    import numpy as np
    n = int(passes)
    if n <= 0:
        return h
    hh = np.asarray(h, np.float64)
    lo, hi = float(hh.min()), float(hh.max())
    for _ in range(n):
        pad = np.pad(hh, 1, mode='edge')
        hh = (pad[1:-1, 1:-1] * 4.0 + pad[:-2, 1:-1] + pad[2:, 1:-1]
              + pad[1:-1, :-2] + pad[1:-1, 2:]) * 0.125
    return np.clip(hh, lo, hi).astype(np.float32)


def spike_heights(h, p, amount, scale=4.0, slope_only=False):
    """R253: Bryce's Spikes / 'slope noise' -- a thresholded value
    noise (only its top 40 percent) adds needles of `amount`; with
    `slope_only` the needles grow on the steep faces (the local
    gradient magnitude, clipped to 1) and leave flats alone."""
    import numpy as np
    from . import patterns as PT
    a = float(amount)
    if a == 0.0:
        return h
    n1 = h.shape[0]
    vn = PT.value_noise((p * np.float32(scale)).astype(np.float32))
    spk = np.clip(vn - np.float32(0.6), 0.0, None) * np.float32(2.5)
    spk = spk.reshape(n1, n1)
    if slope_only:
        gy, gx = np.gradient(np.asarray(h, np.float32))
        slope = np.clip(np.sqrt(gx * gx + gy * gy) * n1, 0.0, 1.0)
        spk = spk * slope
    return (h + spk * np.float32(a)).astype(np.float32)


def island_mask(xs, ys, half, falloff, shape='RADIAL'):
    """R253: the island ring as a field. RADIAL is the R201 expression
    exactly (the terrain multiplies by it the way it always did, so
    the default is bitwise); SQUARE uses the Chebyshev distance, so
    all four edges reach zero at the same falloff."""
    import numpy as np
    isl = max(float(falloff), 1e-6)
    if str(shape or 'RADIAL').upper() == 'SQUARE':
        r = np.maximum(np.abs(xs), np.abs(ys)) / half
    else:
        r = np.sqrt(xs * xs + ys * ys) / half
    t = np.clip((1.0 - r) / isl, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def edge_shape(h, mode, edge_height=0.0):
    """R253: Bryce's Zero Edges / Raise Edges. 'FALLOFF' leaves the
    field to the island ring; 'ZERO' forces the outermost ring to
    exactly 0.0; 'RAISE' lifts the outer eighth of the grid toward
    `edge_height` with a smoothstep -- the bowl, the crater wall, the
    valley that ends in a ridge."""
    import numpy as np
    mode = str(mode or 'FALLOFF').upper()
    if mode == 'ZERO':
        h = np.array(h, np.float32, copy=True)
        h[0, :] = 0.0
        h[-1, :] = 0.0
        h[:, 0] = 0.0
        h[:, -1] = 0.0
        return h
    if mode == 'RAISE':
        n1 = h.shape[0]
        band = max(n1 // 8, 1)
        i = np.arange(n1)
        d = np.minimum(i, n1 - 1 - i)
        d2 = np.minimum(d[:, None], d[None, :]).astype(np.float32)
        t = _smooth01(1.0 - d2 / np.float32(band))
        eh = np.float32(edge_height)
        return (h + (eh - h) * t).astype(np.float32)
    return h


def seamless_heights(h):
    """R253: a tileable field. The core (divisions x divisions) part is
    cross-faded with its own half-period roll along each axis -- the
    weight is 1 at the wrap and 0 at the centre, so the wrap sees ONE
    continuous surface -- and the grid's last row / column are then
    copies of the first, bitwise, so tiles meet exactly."""
    import numpy as np
    h = np.array(h, np.float32, copy=True)
    n1 = h.shape[0]
    n = n1 - 1
    if n < 2:
        return h
    core = h[:n, :n]
    i = np.arange(n, dtype=np.float32)
    c = n * 0.5
    w = _smooth01(np.abs(i - c) / c)
    rolled = np.roll(core, n // 2, axis=1)
    core = core + (rolled - core) * w[None, :]
    rolled = np.roll(core, n // 2, axis=0)
    core = core + (rolled - core) * w[:, None]
    h[:n, :n] = core
    h[n, :n] = core[0, :]
    h[:n, n] = core[:, 0]
    h[n, n] = core[0, 0]
    return h


def water_quad(size, z):
    """R253: the water plane -- one CCW quad of `size` at height `z`,
    in the terrain's own frame (centred, Z up)."""
    half = float(size) * 0.5
    z = float(z)
    verts = [(-half, -half, z), (half, -half, z),
             (half, half, z), (-half, half, z)]
    return verts, [(0, 1, 2, 3)]


def terrain_heights(size=24.0, divisions=96, height=6.0, feature_scale=10.0,
                    octaves=6, roughness=0.5, ridge=0.5, island=0.4,
                    terraces=0, sea_level=0.0, seed=0, kind='MOUNTAIN',
                    direction=0.0,
                    heightmap=None, heightmap_scale=1.0,
                    heightmap_offset=0.0, heightmap_channel='LUMA',
                    heightmap_sampling='BILINEAR', heightmap_tile=False,
                    detail_octaves=0, detail_amount=0.0, symmetry='NONE',
                    thermal_iterations=0, thermal_talus=0.02,
                    thermal_rate=0.5, hydraulic_iterations=0,
                    hydraulic_rain=0.01, hydraulic_solubility=0.1,
                    hydraulic_evaporation=0.5, hydraulic_capacity=0.05,
                    smooth_passes=0, spikes=0.0, spike_scale=4.0,
                    island_shape='RADIAL', edge_mode='FALLOFF',
                    edge_height=0.0, seamless=False):
    """R253: the terrain's NORMALISED heights -- (xs, ys, h01) with h01
    a (n1, n1) float32 field before the height multiply.

    The stage pipeline, in order: base landform -> clip >= 0 ->
    fractal detail -> symmetry -> thermal erosion -> hydraulic erosion
    -> smoothing -> spikes -> island ring / edge shape -> terraces ->
    sea floor -> seamless. Erosion runs BEFORE the island ring so the
    ring's law (the rim is zero) holds after erosion. Every stage is
    skipped at its default, so the default pipeline is the 1.91
    `terrain()` statement for statement. `height` is accepted (the
    operator hands one dict to both this and `heights_to_mesh`) and
    not used here.
    """
    import numpy as np

    xs, ys, n1, half, _step = _terrain_grid(size, divisions)
    p, fs = _terrain_domain(xs, ys, feature_scale, seed)

    octs = max(int(octaves), 1)
    gain = min(max(float(roughness), 0.05), 0.95)
    rg = min(max(float(ridge), 0.0), 1.0)
    kind = str(kind or 'MOUNTAIN').upper()

    h = _terrain_base(kind, p, xs, ys, half, fs, octs, gain, rg, direction,
                      heightmap=heightmap, heightmap_scale=heightmap_scale,
                      heightmap_offset=heightmap_offset,
                      heightmap_channel=heightmap_channel,
                      heightmap_sampling=heightmap_sampling,
                      heightmap_tile=heightmap_tile)

    if int(detail_octaves) > 0 and float(detail_amount) != 0.0:
        h = fractal_detail(h, p, detail_octaves, detail_amount, gain)
    sym = str(symmetry or 'NONE').upper()
    if sym != 'NONE':
        h = symmetrize_heights(_as2d(h), sym).reshape(-1)
    if int(thermal_iterations) > 0:
        h = thermal_erosion(_as2d(h), thermal_iterations, thermal_talus,
                            thermal_rate).reshape(-1)
    if int(hydraulic_iterations) > 0:
        h = hydraulic_erosion(_as2d(h), hydraulic_iterations,
                              hydraulic_rain, hydraulic_solubility,
                              hydraulic_evaporation,
                              hydraulic_capacity).reshape(-1)
    if int(smooth_passes) > 0:
        h = smooth_heights(_as2d(h), smooth_passes).reshape(-1)
    if float(spikes) != 0.0:
        h = spike_heights(_as2d(h), p, spikes, spike_scale).reshape(-1)

    isl = min(max(float(island), 0.0), 1.0)
    if isl > 0.0:
        # radial smoothstep ring: full height inside (1-island) of the
        # half-size, zero at the corner-safe edge radius
        h = h * island_mask(xs, ys, half, isl, island_shape)
    em = str(edge_mode or 'FALLOFF').upper()
    if em != 'FALLOFF':
        h = edge_shape(_as2d(h), em, edge_height).reshape(-1)

    steps = int(terraces)
    if steps > 0:
        # Bryce terracing: quantise with a smoothed riser so shelves
        # read as strata, not stair-step aliasing
        tq = h * steps
        fl = np.floor(tq)
        fr = tq - fl
        h = (fl + fr * fr * (3.0 - 2.0 * fr)) / steps

    sea = min(max(float(sea_level), 0.0), 0.95)
    if sea > 0.0:
        h = np.maximum(h, sea)

    if seamless:
        h = seamless_heights(_as2d(h)).reshape(-1)

    return xs, ys, _as2d(np.asarray(h, np.float32))


def heights_to_mesh(xs, ys, h01, height, divisions=None):
    """R253: the R201 mesh emit -- (verts, faces) from a height field,
    the vertex tuple and quad loops unchanged."""
    import numpy as np

    h = np.asarray(h01).reshape(-1)
    n1 = int(round(h.shape[0] ** 0.5))
    divisions = n1 - 1 if divisions is None else int(divisions)
    height = float(height)
    zs = (h * height).astype(np.float32)
    verts = [(float(xs[i]), float(ys[i]), float(zs[i]))
             for i in range(n1 * n1)]
    faces = []
    for r in range(divisions):
        row = r * n1
        for c in range(divisions):
            a = row + c
            faces.append((a, a + 1, a + n1 + 1, a + n1))
    return verts, faces


def terrain(size=24.0, divisions=96, height=6.0, feature_scale=10.0,
            octaves=6, roughness=0.5, ridge=0.5, island=0.4,
            terraces=0, sea_level=0.0, seed=0, kind='MOUNTAIN',
            direction=0.0, **stages):
    """R201: the Bryce mountain -- a fractal heightfield as a grid mesh.

    The Altitude & Slope node shipped with nothing to stand on: this is
    the terrain that node was built to paint. A (divisions+1)^2 vertex
    grid, displaced by a blend of smooth fBm (rolling hills) and RIDGED
    fractal (the knife-edged crests Bryce terrains are remembered for),
    with the shaping dials the Terrain Editor had:

    - `feature_scale`: metres per large noise feature -- bigger numbers
      mean broader mountains.
    - `octaves` / `roughness`: detail levels and how much each level
      contributes (fBm gain).
    - `ridge`: 0 is all rolling fBm, 1 is all ridged crest.
    - `island`: fades the height to zero toward the grid's edge with a
      smoothstep ring -- a mountain that ENDS inside the grid instead
      of slicing off at the boundary. 0 disables.
    - `terraces`: N > 0 quantises the height into N shelves with a
      smoothed riser between them -- Bryce's terracing.
    - `sea_level`: fraction of the height range flattened into a floor
      -- the water table the ocean plane sits on.
    - `seed`: re-deals the mountain; the same seed is the same mountain
      on every machine (hash-lattice noise, no RNG state anywhere).
    - `kind` (R202): the terrain TYPE. MOUNTAIN is the R201 fractal
      exactly; HILLS is soft pasture; CANYON a cut mesa; DUNES
      directional sand waves (turned by `direction`, radians); CRATER
      one great ringed impact; VOLCANO a cone with its caldera bitten
      out; PLATEAU hard-shouldered table-land. Island, terracing and
      sea level compose with every kind.
    - R253: four more kinds -- PICTURE (the heights are an image, see
      `heightmap`), ARCHIPELAGO, BADLANDS, VALLEY -- and the stage
      dials of `terrain_heights` (heightmap_*, detail_*, symmetry,
      thermal_*, hydraulic_*, smooth_passes, spikes, spike_scale,
      island_shape, edge_mode, edge_height, seamless), every one
      defaulted to 'off' so a 1.91 call builds the 1.91 mesh bitwise.

    Returns (verts, faces) as plain lists: quads, counter-clockwise
    seen from above, ready for `validate` and `from_pydata`.
    """
    xs, ys, h01 = terrain_heights(
        size, divisions, height, feature_scale, octaves, roughness,
        ridge, island, terraces, sea_level, seed, kind, direction,
        **stages)
    return heights_to_mesh(xs, ys, h01, height)


# ------------------------------------------------------------ hair (R208)
#
# Two generators behind "Hair/Particle compatibility":
#
# `hair_ribbons` turns strand polylines into camera-facing triangle
# ribbons -- the exact primitive Blender Internal rendered strands as: a
# flat band that always shows the camera its face, tapering root to tip.
# The strand's shading data rides the mesh's own colour layer, one
# convention shared by everything hairlike in the engine:
#
#     colour.r = intercept (0 at the root, 1 at the tip)
#     colour.g = per-strand random (deterministic hash of the seed)
#     colour.b = strand length, world units
#     colour.a = local thickness, world units
#
# The Hair Info node reads those channels; the Vertex Color node reads
# them too, which is a feature -- a graph can grade a strand any way it
# likes without a special attribute system. UVs carry the EMITTER'S
# root UV, constant along the strand, so image textures paint hair the
# way 2.79 painted it.
#
# `fur_shells` builds the other period hair: shell texturing -- N
# inflated copies of a mesh, each carrying its height in the SAME
# colour convention (intercept = shell height), so one Hair Info node
# serves strands and shells alike. Alpha-tested tufts against the
# height give the classic Dreamcast/GameCube fur; that material is
# wired by the Add-menu operator, but any graph works.


def _strand_hash(i, seed):
    """Deterministic per-strand random in [0,1) -- no RNG state."""
    import numpy as np
    h = np.sin(np.float64(i) * 127.1 + np.float64(seed) * 311.7) \
        * 43758.5453123
    return np.float32(h - np.floor(h))


def hair_ribbons(strands, eye, root_width=0.02, tip_width=0.004,
                 uv_roots=None, seed=0):
    """Strand polylines -> camera-facing ribbon mesh.

    `strands`: sequence of (K,3) arrays (K >= 2), world space.
    `eye`: (3,) camera position the ribbons face.
    Returns (verts, tris, normals, uvs, colors) as float32/int32 arrays,
    colours in the strand convention above. Degenerate strands (under
    two distinct points) are skipped. Everything is a pure function of
    the inputs -- same strands, same picture.
    """
    import numpy as np
    eye = np.asarray(eye, np.float32)
    all_v = []
    all_t = []
    all_n = []
    all_uv = []
    all_c = []
    base = 0
    for si, pts in enumerate(strands):
        P = np.asarray(pts, np.float32)
        if P.ndim != 2 or P.shape[0] < 2:
            continue
        K = P.shape[0]
        seg = np.diff(P, axis=0)
        seglen = np.sqrt((seg * seg).sum(1))
        total = float(seglen.sum())
        if total <= 1e-9:
            continue
        # arclength intercept, 0 at root, 1 at tip
        t = np.concatenate([[0.0], np.cumsum(seglen)]).astype(np.float32)
        t /= np.float32(total)
        # tangent at each point: central difference, one-sided at ends
        tan = np.empty_like(P)
        tan[0] = seg[0]
        tan[-1] = seg[-1]
        if K > 2:
            tan[1:-1] = P[2:] - P[:-2]
        tl = np.sqrt((tan * tan).sum(1, keepdims=True))
        tan = tan / np.maximum(tl, 1e-12)
        view = P - eye[None, :]
        vl = np.sqrt((view * view).sum(1, keepdims=True))
        view = view / np.maximum(vl, 1e-12)
        side = np.cross(tan, view)
        sl = np.sqrt((side * side).sum(1, keepdims=True))
        # a strand aimed straight at the camera has no cross product;
        # borrow any perpendicular so the ribbon keeps its width
        bad = (sl[:, 0] < 1e-6)
        if bad.any():
            alt = np.cross(tan[bad], np.array([0.0, 0.0, 1.0],
                                              np.float32)[None, :])
            al = np.sqrt((alt * alt).sum(1, keepdims=True))
            flat = (al[:, 0] < 1e-6)
            if flat.any():
                alt[flat] = np.array([1.0, 0.0, 0.0], np.float32)
                al[flat] = 1.0
            side[bad] = alt / np.maximum(al, 1e-12)
            sl[bad] = 1.0
        side = side / np.maximum(sl, 1e-12)
        w = (np.float32(root_width)
             + (np.float32(tip_width) - np.float32(root_width)) * t)
        half = (side * (w * 0.5)[:, None]).astype(np.float32)
        v = np.empty((K * 2, 3), np.float32)
        v[0::2] = P - half
        v[1::2] = P + half
        # the ribbon shows the camera its face
        n = np.repeat(-view, 2, axis=0).astype(np.float32)
        rnd = _strand_hash(si, seed)
        c = np.empty((K * 2, 4), np.float32)
        c[:, 0] = np.repeat(t, 2)
        c[:, 1] = rnd
        c[:, 2] = np.float32(total)
        c[:, 3] = np.repeat(w, 2)
        if uv_roots is not None and si < len(uv_roots):
            uv = np.tile(np.asarray(uv_roots[si], np.float32)[None, :],
                         (K * 2, 1))
        else:
            uv = np.tile(np.array([rnd, 0.0], np.float32)[None, :],
                         (K * 2, 1))
        idx = np.arange(K - 1, dtype=np.int32) * 2 + base
        tri = np.empty(((K - 1) * 2, 3), np.int32)
        tri[0::2, 0] = idx
        tri[0::2, 1] = idx + 1
        tri[0::2, 2] = idx + 2
        tri[1::2, 0] = idx + 1
        tri[1::2, 1] = idx + 3
        tri[1::2, 2] = idx + 2
        all_v.append(v)
        all_t.append(tri)
        all_n.append(n)
        all_uv.append(uv)
        all_c.append(c)
        base += K * 2
    if not all_v:
        z3 = np.zeros((0, 3), np.float32)
        return (z3, np.zeros((0, 3), np.int32), z3.copy(),
                np.zeros((0, 2), np.float32), np.zeros((0, 4), np.float32))
    return (np.concatenate(all_v), np.concatenate(all_t),
            np.concatenate(all_n), np.concatenate(all_uv),
            np.concatenate(all_c))


def fur_shells(verts, normals, tris, count=12, length=0.25, curve=1.0,
               comb=(0.0, 0.0, 0.0), uvs=None):
    """A mesh -> its shell stack, in the strand colour convention.

    `count` shells from the surface (height 0) to `length` along the
    vertex normals; `curve` biases the spacing (above 1 packs shells
    near the root, where fur is densest); `comb` is a world-space
    offset the shells lean into quadratically -- gravity, or a parting.
    Colour.r carries the shell height, so a Hair Info node's Intercept
    grades the stack exactly as it grades a strand. Returns
    (verts, tris, normals, uvs, colors).
    """
    import numpy as np
    V = np.asarray(verts, np.float32)
    N = np.asarray(normals, np.float32)
    T = np.asarray(tris, np.int32)
    comb = np.asarray(comb, np.float32)
    count = max(int(count), 2)
    nv = V.shape[0]
    out_v = []
    out_t = []
    out_n = []
    out_c = []
    out_uv = []
    uv_in = np.asarray(uvs, np.float32) if uvs is not None else None
    for i in range(count):
        h = np.float32((i / (count - 1)) ** max(float(curve), 1e-3))
        out_v.append(V + N * (h * np.float32(length))
                     + comb[None, :] * (h * h))
        out_t.append(T + i * nv)
        out_n.append(N)
        c = np.empty((nv, 4), np.float32)
        c[:, 0] = h
        c[:, 1] = 0.5
        c[:, 2] = np.float32(length)
        c[:, 3] = 1.0
        out_c.append(c)
        out_uv.append(uv_in if uv_in is not None
                      else np.zeros((nv, 2), np.float32))
    return (np.concatenate(out_v).astype(np.float32),
            np.concatenate(out_t).astype(np.int32),
            np.concatenate(out_n).astype(np.float32),
            np.concatenate(out_uv).astype(np.float32),
            np.concatenate(out_c).astype(np.float32))
