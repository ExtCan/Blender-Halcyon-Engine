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


def terrain(size=24.0, divisions=96, height=6.0, feature_scale=10.0,
            octaves=6, roughness=0.5, ridge=0.5, island=0.4,
            terraces=0, sea_level=0.0, seed=0, kind='MOUNTAIN',
            direction=0.0):
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

    Returns (verts, faces) as plain lists: quads, counter-clockwise
    seen from above, ready for `validate` and `from_pydata`.
    """
    import numpy as np

    from . import patterns as PT

    divisions = max(int(divisions), 2)
    n1 = divisions + 1
    size = float(size)
    height = float(height)
    fs = max(float(feature_scale), 1e-3)
    step = size / divisions
    half = size * 0.5

    cc = np.arange(n1, dtype=np.float32) * step - half
    gx, gy = np.meshgrid(cc, cc, indexing='xy')
    xs = gx.ravel()
    ys = gy.ravel()

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

    octs = max(int(octaves), 1)
    gain = min(max(float(roughness), 0.05), 0.95)
    rg = min(max(float(ridge), 0.0), 1.0)
    kind = str(kind or 'MOUNTAIN').upper()

    def _smooth(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

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
    else:                                       # MOUNTAIN, the R201 road
        h = PT.fbm(p, octaves=octs, gain=gain)
        if rg > 0.0:
            h = h * (1.0 - rg) \
                + PT.ridged(p, octaves=octs, gain=gain) * rg
    h = np.clip(h, 0.0, None)

    isl = min(max(float(island), 0.0), 1.0)
    if isl > 0.0:
        # radial smoothstep ring: full height inside (1-island) of the
        # half-size, zero at the corner-safe edge radius
        r = np.sqrt(xs * xs + ys * ys) / half
        t = np.clip((1.0 - r) / max(isl, 1e-6), 0.0, 1.0)
        h = h * (t * t * (3.0 - 2.0 * t))

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
