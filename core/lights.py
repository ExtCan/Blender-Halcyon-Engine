"""Light evaluation: attenuation, spot cones, shadow maps and ray shadows.

Shadow maps are rendered with the same rasteriser as the beauty pass, then
compared in *linear light-space distance* rather than NDC z -- which makes the
bias a world-space number the user can reason about, instead of a magic
depth-buffer epsilon that has to be retuned for every scene scale.
"""

import threading

import numpy as np

from . import mathx as M
from . import raster
from . import softmap as SM

EPS = 1e-6


# --------------------------------------------------------------- projections


def _look_at_lh(eye, target, up):
    f = M.normalize(np.asarray(target, np.float32) - np.asarray(eye, np.float32))
    if abs(float(np.dot(f, up))) > 0.999:
        up = np.array([1.0, 0.0, 0.0], np.float32)
    s = M.normalize(np.cross(f, up))
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -float(np.dot(s, eye))
    m[1, 3] = -float(np.dot(u, eye))
    m[2, 3] = float(np.dot(f, eye))
    return m


def _persp(fov_y, aspect, near, far):
    t = 1.0 / np.tan(fov_y * 0.5)
    m = np.zeros((4, 4), np.float32)
    m[0, 0] = t / aspect
    m[1, 1] = t
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _ortho(half, near, far):
    m = np.eye(4, dtype=np.float32)
    m[0, 0] = 1.0 / half
    m[1, 1] = 1.0 / half
    m[2, 2] = -2.0 / (far - near)
    m[2, 3] = -(far + near) / (far - near)
    return m


CUBE_DIRS = (
    (np.array([1.0, 0, 0], np.float32), np.array([0, 0, 1.0], np.float32)),
    (np.array([-1.0, 0, 0], np.float32), np.array([0, 0, 1.0], np.float32)),
    (np.array([0, 1.0, 0], np.float32), np.array([0, 0, 1.0], np.float32)),
    (np.array([0, -1.0, 0], np.float32), np.array([0, 0, 1.0], np.float32)),
    (np.array([0, 0, 1.0], np.float32), np.array([0, 1.0, 0], np.float32)),
    (np.array([0, 0, -1.0], np.float32), np.array([0, 1.0, 0], np.float32)),
)


class ShadowMap:
    """One depth image rendered from a light."""

    # `digest` is filled lazily by gpu/shade.map_digest (an exact content
    # digest of `depth`, the GPU atlas's identity); a map whose depth is
    # rewritten in place must reset it to None
    __slots__ = ('vp', 'depth', 'near', 'far', 'persp', 'size', 'origin',
                 'extent', 'axes', 'd_min', 'digest')

    def __init__(self, vp, zndc, near, far, persp, size, origin, extent=1.0):
        self.vp = vp
        self.extent = float(extent)
        self.near = float(near)
        self.far = float(far)
        self.persp = bool(persp)
        self.size = int(size)
        self.origin = np.asarray(origin, np.float32)
        if zndc is None or raster.is_blank(zndc):
            # R254: no caster reached this map (a cube face that sees
            # nothing, a face outside a wide spot's cone). Every texel is
            # CLASSIC's empty texel -- `linearise(1.0)`, the value the
            # array path below gives a plane of ones -- and every blank
            # map of one size and value shares ONE read-only plane: 16 MB
            # and a full-plane pass saved per empty 2048 face, the same
            # array at every lookup
            empty = self._linearise(np.ones(1, np.float32))[0]
            self.depth = _blank_plane(self.size, empty)
        elif isinstance(zndc, tuple):
            # R251 C117: the MIDPOINT map (Woo 1992, Maya 'Use Mid Dist',
            # Blender Classic-Halfway). `_render_depth` hands (z1, z2,
            # has2): the nearest and the second-nearest caster per texel
            # and where a second one exists. The texel stores the halfway
            # point of the two LINEAR light-space distances (Blender
            # averaged its integer z; the compare domain here is linear,
            # LI:1-7), so a receiver compares against a value inside the
            # caster's thickness -- no bias, no acne, no Peter-Panning.
            # A texel that sees a lone surface stores CLASSIC's own empty
            # texel (`linearise(1.0)`, the far plane: mental ray's rule)
            # -- computed through the array path, never `float32(far)`,
            # which differs from it by two roundings on an ortho map and
            # by 2nf/((f+n)-(f-n)) != f on every perspective map
            z1, z2, has2 = zndc
            d1 = self._linearise(z1)
            d2 = self._linearise(z2)
            h1 = d1 * np.float32(0.5)
            h2 = d2 * np.float32(0.5)
            mid = h1 + h2
            empty = self._linearise(np.ones(1, np.float32))[0]
            self.depth = np.where(has2, mid, empty).astype(np.float32)
        else:
            self.depth = self._linearise(zndc)
        # what a SIZED lamp's lookup (core/softmap.py) reads beside the
        # depth: the map's right / up / depth world axes and the nearest
        # depth it holds (no caster stands nearer, which bounds the
        # blocker search). Neither depends on the lamp's size -- nothing
        # on a map does, so a size edit never re-bakes one
        self.axes = SM.map_axes(vp, self.persp)
        self.d_min = float(self.depth.min()) if self.depth.size else \
            float(self.far)

    def _linearise(self, z):
        z = np.clip(np.asarray(z, np.float32), -1.0, 1.0)
        n, f = self.near, self.far
        if self.persp:
            den = (f + n) - z * (f - n)
            den = np.where(np.abs(den) < EPS, EPS, den)
            return (2.0 * n * f / den).astype(np.float32)
        return ((z * 0.5 + 0.5) * (f - n) + n).astype(np.float32)

    def texel_size(self, dist):
        """World-space width of one shadow texel at `dist` from the light."""
        if self.persp:
            return 2.0 * self.extent * np.maximum(dist, self.near) / self.size
        return np.full_like(np.asarray(dist, np.float32),
                            2.0 * self.extent / self.size)

    def lookup(self, P, bias, softness, samples):
        """Fraction of `samples` taps that are lit, per point. P is (N,3)."""
        n = P.shape[0]
        ph = np.empty((n, 4), np.float32)
        ph[:, :3] = P
        ph[:, 3] = 1.0
        clip = ph @ self.vp.T
        w = clip[:, 3]
        safe = np.where(np.abs(w) < EPS, EPS, w)
        ndc = clip[:, :3] / safe[:, None]
        inside = (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0) & \
                 (ndc[:, 2] <= 1.0) & (w > 0.0 if self.persp else True)
        dist = self._linearise(ndc[:, 2])
        u = (ndc[:, 0] * 0.5 + 0.5) * self.size
        v = (ndc[:, 1] * 0.5 + 0.5) * self.size
        taps = max(int(samples), 1)
        radius = max(float(softness), 0.0)
        lit = np.zeros(n, np.float32)
        offs = _pcf_offsets(taps) * radius
        for ox, oy in offs:
            xi = np.clip(np.floor(u + ox).astype(np.int32), 0, self.size - 1)
            yi = np.clip(np.floor(v + oy).astype(np.int32), 0, self.size - 1)
            occ = self.depth[yi, xi]
            lit += (dist - bias <= occ).astype(np.float32)
        lit /= len(offs)
        return np.where(inside, lit, 1.0).astype(np.float32)


_PCF_CACHE = {}

#: the planes blank maps share, by (size, empty value); see ShadowMap
_BLANK_PLANES = {}


def _blank_plane(size, value):
    key = (int(size), float(value))
    plane = _BLANK_PLANES.get(key)
    if plane is None:
        plane = np.full((int(size), int(size)), value, np.float32)
        plane.flags.writeable = False
        if len(_BLANK_PLANES) >= 6:
            _BLANK_PLANES.pop(next(iter(_BLANK_PLANES)))
        _BLANK_PLANES[key] = plane
    return plane


def _pcf_offsets(n):
    if n in _PCF_CACHE:
        return _PCF_CACHE[n]
    if n <= 1:
        pts = np.zeros((1, 2), np.float32)
    else:
        # Vogel disc -- even coverage without the axis artefacts of a grid
        gr = np.pi * (3.0 - np.sqrt(5.0))
        i = np.arange(n, dtype=np.float32)
        r = np.sqrt((i + 0.5) / n)
        th = i * gr
        pts = np.stack([r * np.cos(th), r * np.sin(th)], axis=1).astype(np.float32)
    _PCF_CACHE[n] = pts
    return pts


class CubeShadow:
    """Six ShadowMaps for an omnidirectional light."""

    __slots__ = ('faces', 'origin', 'd_min')

    def __init__(self, faces, origin):
        self.faces = faces
        self.origin = np.asarray(origin, np.float32)
        # the nearest depth any of the six faces holds (see ShadowMap)
        self.d_min = min(float(f.d_min) for f in faces)

    def texel_size(self, dist):
        return self.faces[0].texel_size(dist)

    def lookup(self, P, bias, softness, samples):
        d = P - self.origin[None, :]
        ax = np.abs(d)
        major = np.argmax(ax, axis=1)
        sign = d[np.arange(d.shape[0]), major] >= 0
        face = major * 2 + np.where(sign, 0, 1)
        out = np.ones(P.shape[0], np.float32)
        bias_arr = np.ndim(bias) > 0
        for fi in range(6):
            m = face == fi
            if not m.any():
                continue
            out[m] = self.faces[fi].lookup(P[m], bias[m] if bias_arr else bias,
                                           softness, samples)
        return out


# --------------------------------------------------------------- map baking


def scene_bounds(verts):
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    centre = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - lo)) * 0.5 + 1e-4
    return centre.astype(np.float32), radius


class _DepthFrags(raster.FragmentList):
    """R251 C117: a fragment collector that keeps only (px, py, depth).

    The G-buffer road's reading of the midpoint map's second surface
    (tests/test_r254_bake_speed.py holds the depth-only bake against it);
    the full FragmentList would hold bary/front/tri for ~1.4 GB per cube
    face on a 4096-texel map at depth complexity 3."""

    def add(self, px, py, tri, depth, bary, front, bary_lin=None):
        if px.size == 0:
            return
        self.px.append(px.astype(np.int32))
        self.py.append(py.astype(np.int32))
        self.depth.append(depth.astype(np.float32))

    def finish(self):
        if not self.px:
            z = np.zeros(0, np.int32)
            return z, z, np.zeros(0, np.float32)
        return (np.concatenate(self.px), np.concatenate(self.py),
                np.concatenate(self.depth))


#: triangles per run of the G-buffer road's second raster (see above)
MIDPOINT_RUN = 20000


def _box_misses(bounds, vp, size):
    """True when an axis-aligned box of casters cannot put a fragment on
    a size x size map: all eight corners behind the near plane, or all
    eight in front of it and outside the same side of the map by four
    texels. Both are linear half-spaces of clip space, so what holds for
    the corners holds for every triangle inside the box; a box that
    straddles the near plane is never judged (the clipper makes new
    corners there). Float64 on eight points with a margin a thousand
    float32 roundings wide: the per-triangle reject and the fills decide
    everything this cannot."""
    lo, hi = bounds
    c = np.array([[x, y, z, 1.0] for x in (float(lo[0]), float(hi[0]))
                  for y in (float(lo[1]), float(hi[1]))
                  for z in (float(lo[2]), float(hi[2]))], np.float64)
    clip = c @ np.asarray(vp, np.float64).T
    x, y, w = clip[:, 0], clip[:, 1], clip[:, 3]
    f = clip[:, 2] + w
    tol = 1e-4 * max(float(np.abs(clip).max()), 1e-6)
    if bool((f < -tol).all()):
        return True
    if not bool((f > tol).all()):
        return False
    g = (1.0 + 8.0 / float(size)) * w
    return bool((x > g + tol).all() or (x < -g - tol).all()
                or (y > g + tol).all() or (y < -g - tol).all())


#: R254: the frame's depth baker when the caller names none: None bakes
#: every map with the NumPy fill below. core/render hands the GPU
#: device's baker to `build_shadow_maps` per frame (gpu/craster.
#: DepthBaker) and never sets this; it is the hook for a tool or a test
#: that wants every bake of the process on a baker of its own
DEPTH_BAKER = None


def _render_depth(verts, tris, vp, size, cull='NONE', midpoint=False,
                  bounds=None, baker=None):
    """THE BAKE: vertex positions, caster triangles, a view-projection
    and a size in; the map's NDC depth plane out (1.0 where nothing
    landed), or the MIDPOINT triple (z1, z2, has2).

    Every map this engine builds comes through this one function -- a
    whole map, a cached static layer, the layer of the objects that
    moved, a map baked on its first lookup, the maps a worker pool is
    handed -- and always through the module's name for it, so a device
    bake can stand here: whatever returns these texels for these
    arguments is a bake. `bounds` (lo, hi) is an axis-aligned box around
    the triangles' vertices, when the caller has one: a map the box
    cannot reach returns blank without reading a vertex.

    `baker`, when the frame has one (the GPU device's: gpu/craster.
    DepthBaker), is asked first for a CLASSIC plane -- baker(verts, tris,
    vp, size, cull) -- and answers it or None; None is a refusal the
    baker names itself, and the fill below bakes the face. A MIDPOINT
    map's second surface is the fill's alone.

    R254: depth only (core/raster.rasterize_depth: the same clipper,
    coverage and z as the beauty rasteriser, bitwise its zndc, none of
    its G-buffer), in front of it a per-map reject of the triangles that
    cannot own a fragment. The R87 note that stood here -- a frustum
    cull 'kept ~100% of triangles' and cost 15% -- was measured on maps
    that enclose the whole scene (a sun, a spot); a cube face keeps a
    quarter to a half, and the reject is one byte per vertex instead of
    the gather that cull paid.

    R251 C117: the MIDPOINT map's second surface is, per texel, the
    nearest fragment BEYOND the A-buffer tolerance of z1 -- THE NAMED
    TIE RULE: a fragment within `abuf_depth_limit` of the nearest is the
    SAME surface (the two triangles at a shared caster edge both include
    the boundary texel by the wobble window and would otherwise register
    the front surface twice: midpoint = front, self-shadow speckle along
    every caster's wireframe). Corollary: a closed shell thinner than
    the tolerance is one surface and casts nothing. Order-free by
    construction: z1 is a min; z2 is the min over a SET whose membership
    depends only on z1; a min is associative, commutative and exact in
    float32."""
    if bounds is not None and _box_misses(bounds, vp, size):
        return raster.rasterize_depth(verts, tris[:0], vp, size, size,
                                      cull=cull, second=midpoint)
    if baker is not None and not midpoint and tris.shape[0]:
        plane = baker(verts, tris, vp, size, cull)
        if plane is not None:
            return plane
    return raster.rasterize_depth(verts, tris, vp, size, size, cull=cull,
                                  second=midpoint)


def midpoint_map(light, settings):
    """R251 C117: does this lamp's map store Woo's midpoint? The lamp's
    own Map Depth (INHERIT / CLASSIC / MIDPOINT) overrides the render's
    (CLASSIC / MIDPOINT) -- Maya's Use Mid Dist and Blender's buffer
    type were per lamp. One resolve for the bake, the CPU lookup and
    the GPU's `_shadow_meta`."""
    own = str(getattr(light, 'shadow_map_depth', 'INHERIT') or 'INHERIT')
    mode = own if own != 'INHERIT' else \
        str(getattr(settings, 'shadow_map_depth', 'CLASSIC') or 'CLASSIC')
    return mode == 'MIDPOINT'


def shadow_clip_planes(light, near_auto, far_auto):
    """R253 (F): Blender Internal's la->clipsta / la->clipend as the
    shadow buffer's near and far planes. 0 keeps the automatic plane
    (the 1.89 rule, bitwise: the same python expressions the caller
    evaluated); a Start above 0 replaces the near plane, an End past
    the near plane replaces the far one. Both roads read the planes
    off the built map (the GPU shadow meta copies near/far/vp), so
    the clipped map is one texture on both devices."""
    cs = float(getattr(light, 'shadow_clip_start', 0.0) or 0.0)
    ce = float(getattr(light, 'shadow_clip_end', 0.0) or 0.0)
    near = cs if cs > 0.0 else near_auto
    far = ce if ce > near else far_auto
    return near, far


#: R254: a Spot bakes ONE perspective map while the field of view its
#: cone needs stays at or under this (126 degrees); a wider one bakes the
#: Point lamp's cube. A single map stretched towards 172 degrees spent 18
#: texels per unit tangent where a 60 degree cone has 372, and the
#: texel-scaled normal offset lifted the receiver out of its own shadow
#: (150 degrees: 62% of the shadow lost)
SPOT_CUBE_FOV = 2.2

#: the half angle of the cone that holds a whole cube face: acos(1/sqrt 3)
_CUBE_FACE_REACH = float(np.arccos(1.0 / np.sqrt(3.0)))


def spot_map_fov(light):
    """The field of view a Spot's shadow map must span, in radians and
    uncapped: its cone with the 15% margin every Spot map always had --
    the diagonal of a Square cone, and under GX Ring 2 the angle at
    which the ring's light ends (libogc's RING2 is centred ON the cutoff
    and reaches zero at cos = c - (1 - c) / sqrt 2, 9.5 degrees past a
    60 degree cone; the map used to stop at the cutoff and the shadow
    was cut off inside the lit ring). The one place the bake, the cache
    key and the choice between one map and a cube read it."""
    size = float(light.spot_size)
    square = bool(getattr(light, 'spot_square', False))
    if str(getattr(light, 'spot_law', 'BLENDER') or 'BLENDER') != 'GX_RING2':
        if square:
            # R253 (A): a Square cone lights a square patch whose corners
            # sit sqrt(2) further out than the round cone's edge -- widen
            # the buffer to the diagonal so the corners keep their shadows
            half_t = float(np.tan(size * 0.5)) * np.sqrt(2.0)
            return max(2.0 * float(np.arctan(half_t)) * 1.15, 0.05)
        return max(size * 1.15, 0.05)
    c = float(np.cos(size * 0.5))
    edge = float(np.arccos(max(c - (1.0 - c) / np.sqrt(2.0), -1.0)))
    if square:
        edge = float(np.arctan(np.tan(min(edge, 1.5)) * np.sqrt(2.0)))
    return max(2.0 * edge * 1.15, 0.05)


def _map_size(light, settings):
    """The side of a lamp's map: its own Map Size or the render's,
    32..4096, divided for a viewport draft (preview.py sets
    `_shadow_map_div` on a draft's settings and divides the texel-unit
    Softness with it; a refine and an F12 never carry it)."""
    size = int(light.shadow_map_size or settings.shadow_map_size)
    size = max(32, min(4096, size))
    div = int(getattr(settings, '_shadow_map_div', 1) or 1)
    if div > 1:
        size = min(size, max(64, size // div))
    return size


class _MapPlan:
    """One lamp's map before it is baked: the frames of its faces and
    everything the built map records. `skip` marks cube faces that are
    never baked (outside a wide Spot's cone: no lit point looks there)."""

    __slots__ = ('faces', 'skip', 'near', 'far', 'persp', 'size',
                 'origin', 'extent', 'mid', 'cube')

    def blob(self):
        """Every number a bake of this plan reads, as bytes: two plans
        with one blob rasterise identical texels from identical
        casters."""
        head = np.array([self.near, self.far, self.extent, self.size,
                         self.persp, self.mid, self.cube], np.float64)
        return head.tobytes() + bytes(bytearray(self.skip)) + b''.join(
            np.asarray(vp, np.float32).tobytes() for vp in self.faces)

    def face(self, fi, zndc):
        """The ShadowMap of face `fi` from its depth plane."""
        return ShadowMap(self.faces[fi], zndc, self.near, self.far,
                         self.persp, self.size, self.origin, self.extent)

    def build(self, depths):
        """The ShadowMap / CubeShadow of one depth plane per face."""
        return self.join([self.face(fi, z) for fi, z in enumerate(depths)])

    def join(self, maps):
        if self.cube:
            return CubeShadow(list(maps), self.origin)
        return maps[0]


def _plan_map(light, settings, centre, radius):
    """The plan of `light`'s map in a scene of these bounds. The
    expressions are the ones the bake always evaluated, in their order:
    a plan is bitwise the frame the old code built."""
    plan = _MapPlan()
    plan.size = _map_size(light, settings)
    # R251 C117: CLASSIC (nearest + bias) or Woo's MIDPOINT map
    plan.mid = midpoint_map(light, settings)
    plan.cube = False
    pos = np.asarray(light.position, np.float32)
    if light.type == 'SUN':
        d = M.normalize(np.asarray(light.direction, np.float32))
        eye = centre - d * (radius * 2.5)
        view = _look_at_lh(eye, centre, np.array([0, 0, 1.0], np.float32))
        near = 1e-3
        far = radius * 5.0
        half = radius * 1.05
        plan.faces = [_ortho(half, near, far) @ view]
        plan.skip = [False]
        plan.near, plan.far, plan.persp = near, far, False
        plan.origin, plan.extent = eye, half
        return plan
    near = max(radius * 0.005, 1e-3)
    far = max(float(np.linalg.norm(centre - pos)) + radius * 1.5, near * 10)
    wide = None
    if light.type == 'SPOT':
        wide = spot_map_fov(light)
        if wide <= SPOT_CUBE_FOV:
            d = M.normalize(np.asarray(light.direction, np.float32))
            view = _look_at_lh(pos, pos + d,
                               np.array([0, 0, 1.0], np.float32))
            # R253 (F): BI's ClipSta / ClipEnd replace the automatic
            # planes (0 = automatic, the expressions above untouched)
            near, far = shadow_clip_planes(light, near, far)
            fov = min(wide, 3.0)
            plan.faces = [_persp(fov, 1.0, near, far) @ view]
            plan.skip = [False]
            plan.near, plan.far, plan.persp = near, far, True
            plan.origin, plan.extent = pos, float(np.tan(fov * 0.5))
            return plan
    # a Point lamp's cube -- also an Area lamp's, and a Spot's whose
    # cone is too wide for one perspective map
    if light.type != 'AREA':
        # R253 (F): the cube's six faces share BI's clip pair. R254: an
        # Area lamp keeps the automatic planes -- its panel never drew
        # the pair and the tooltips say Spot and Point only, yet a value
        # left on a lamp that was once a Spot clipped its map unseen
        near, far = shadow_clip_planes(light, near, far)
    proj = _persp(np.pi * 0.5, 1.0, near, far)
    plan.faces, plan.skip = [], []
    axis = None
    if wide is not None:
        axis = M.normalize(np.asarray(light.direction, np.float32))
    for fwd, up in CUBE_DIRS:
        plan.faces.append(proj @ _look_at_lh(pos, pos + fwd, up))
        skip = False
        if axis is not None:
            # a face wholly outside the cone the map must span: nothing
            # the Spot lights looks through it, so it stays blank -- as
            # everything outside the single map always read lit
            ang = float(np.arccos(np.clip(float(np.dot(axis, fwd)),
                                          -1.0, 1.0)))
            skip = ang - _CUBE_FACE_REACH > min(wide * 0.5, np.pi)
        plan.skip.append(bool(skip))
    plan.near, plan.far, plan.persp = near, far, True
    plan.origin, plan.extent, plan.cube = pos, 1.0, True
    return plan


def _light_shadow_signature(light, settings=None):
    """What ONE lamp's bake reads of the lamp, resolved the way the bake
    resolves it -- the lamp's part of its cache key (R254).

    A SUN's frame is built from the scene's bounds and its DIRECTION: its
    position is not here, so a Sun dragged across the viewport re-bakes
    nothing. A Point or Area lamp's cube is built from its POSITION: its
    rotation is not here, so aiming an Area lamp re-bakes nothing. Only
    a Spot reads both, with its cone (Spot Size, Square, and whether the
    law is GX Ring 2, which widens the map). ClipSta / ClipEnd count for
    the lamps that use them (Spot and Point). The Map Size and the Map
    Depth are the RESOLVED ones (the lamp's own or the render's, the
    viewport draft's divisor applied), so a render setting a lamp
    overrides never misses that lamp, and Shadow Method is no part of
    any key: a lamp that wants a map wants the same map under Shadow
    Maps and under Per Light.

    Colour and energy are absent -- they tint the LIGHT, never the depth
    -- and so is the lamp's SIZE (Radius, a Sun's Angle, an Area's
    Size): the penumbra is measured per receiver at lookup
    (core/softmap.py) and the built map carries nothing derived from it.
    Without `settings` the lamp's own values stand unresolved (0 = the
    render's size, INHERIT)."""
    if settings is not None:
        size = _map_size(light, settings)
        mid = midpoint_map(light, settings)
    else:
        size = int(getattr(light, 'shadow_map_size', 0) or 0)
        mid = str(getattr(light, 'shadow_map_depth', 'INHERIT')) == 'MIDPOINT'
    clip = (round(float(getattr(light, 'shadow_clip_start', 0.0) or 0.0), 6),
            round(float(getattr(light, 'shadow_clip_end', 0.0) or 0.0), 6))
    if light.type == 'SUN':
        return ('SUN', tuple(np.round(light.direction, 5)), size, mid)
    pos = tuple(np.round(light.position, 5))
    if light.type == 'SPOT':
        return ('SPOT', pos, tuple(np.round(light.direction, 5)),
                round(float(light.spot_size), 5),
                bool(getattr(light, 'spot_square', False)),
                str(getattr(light, 'spot_law', 'BLENDER') or 'BLENDER')
                == 'GX_RING2', clip, size, mid)
    if light.type == 'AREA':
        clip = (0.0, 0.0)
    return ('CUBE', pos, clip, size, mid)


def _shadow_base_signature(scene, settings, caster_tris):
    """What EVERY shadow map of a frame depends on beside its own lamp:
    the geometry and the caster set, each read in full.

    mathx.content_digest of every vertex position (a strided sum served
    the old maps for a move between the stride's samples, and for one
    whose components cancel), of every triangle index (the same
    positions re-topologised are other casters) and of the caster list
    (it was a count and an index sum: two sets of equal count and equal
    sum collided). R254: the shadow settings left this key -- each lamp
    carries what its own bake resolves from them
    (`_light_shadow_signature`), so switching Shadow Maps to Per Light,
    or moving the render's Map Size under lamps that set their own,
    misses nothing."""
    mesh = scene.mesh
    v, t = mesh.verts, mesh.tris
    return ((int(v.shape[0]), M.content_digest(v),
             int(t.shape[0]), M.content_digest(t)),
            None if caster_tris is None else
            (int(caster_tris.size), M.content_digest(caster_tris)))


# ------------------------------------------------- who looks a map up
#
# R254: a map is baked for a lamp when something will look it up. The
# bake used to walk every lamp whose Shadows menu resolved to a map --
# lamps the Light Limit had dropped, a Hemi (which casts no shadow), a
# Screen Spotlight (which casts none either), lamps at Power 0 -- and
# each paid one to six rasterisations for texels nothing read.
#
# The readers of `light.shadow_map` are `visibility` (the CPU lookup)
# and the GPU plan's `_shadow_meta` / signatures, which only ever ask
# for the lamps of `frame_lights`. So the rule has three answers:
#
#   None     the lamp has no shadow term at all: the render's Shadows
#            off, an AMBIENT lamp, Shadows None, a method other than a
#            map -- and a HEMI, for which `visibility` answers ones
#            before it looks at a map and the plan emits no function.
#   'bake'   a lamp of `frame_lights` whose shadow a surface may read:
#            baked now (in threads), on both roads.
#   'defer'  a lamp that still WOULD get a map but that no surface is
#            known to read: one the Light Limit dropped, and -- on a
#            frame without a BVH -- a Screen Spotlight, an ambient lamp,
#            a lamp that adds no light while nothing reads a shadow
#            apart from the light. It is baked on its first lookup
#            (`visibility` -> `_deferred_map`), so a reader this rule
#            does not know -- today the volume march, which lights a
#            container from every lamp, and the ground plane under a
#            Screen Spotlight -- gets exactly the map it always got, and
#            a lamp nobody looks up is never baked.
#
# The deferral is what keeps the rule right while the readers change:
# nothing here assumes who reads, only who certainly does.


def lamp_is_dark(light):
    """Power exactly 0 or a black colour: the lamp's radiance is zero at
    every point. The RAW Power, never the light style's gain -- a
    strobing lamp keeps its map (the per-light cache makes it pay once)
    and its plan."""
    return float(light.energy) == 0.0 \
        or not any(float(c) != 0.0 for c in light.color)


def lamp_has_no_lobe(light):
    """Neither a diffuse nor a specular lobe is left by the lamp's flags
    (Diffuse / Specular off, or Diffuse Only with Specular Only)."""
    dif = bool(getattr(light, 'affect_diffuse', True)) \
        and not getattr(light, 'specular_only', False)
    spc = bool(getattr(light, 'affect_specular', True)) \
        and not getattr(light, 'diffuse_only', False)
    return not dif and not spc


def lamp_adds_nothing(light):
    """The lamp's own contribution to a lit surface is exactly zero: no
    radiance, or no lobe to carry it. A Hemi is never judged here (its
    ground colour is a second radiance, and it has no shadow to save)."""
    if str(getattr(light, 'type', '')).upper() == 'HEMI':
        return False
    return lamp_is_dark(light) or lamp_has_no_lobe(light)


def wants_shadow_map(light, settings):
    """The bake's oldest gate: the render's Shadows on, a lamp that is
    not AMBIENT, Shadows not None and a method that resolves to a
    map."""
    if not settings.shadows or light.type == 'AMBIENT':
        return False
    mode = light.shadow if settings.shadow_default == 'PER_LIGHT' else \
        settings.shadow_default
    return mode == 'MAP' and light.shadow != 'NONE'


def map_roles(scene, settings):
    """None / 'bake' / 'defer' for each lamp of the scene, in order (the
    section note above is the rule)."""
    lights = scene.lights or []
    frame = {id(l) for l in frame_lights(scene, settings)}
    # a reader that takes a shadow apart from its lamp's light (the
    # Shadow pass, a Shadows Only material, a paint or cel key) makes
    # every lamp's shadow count: render._build_shadows says whether one
    # exists; unsaid, one is assumed
    quiet = bool(getattr(settings, '_shadow_quiet', False))
    # a lamp OF THE FRAME goes without a map only on a frame with no
    # BVH. With one, the GPU plan reads a missing map as a request for
    # traced shadows (`_shadow_meta`: the ray fallback, and its refusal
    # under a caster mask), so such a frame keeps every map it had
    alone = bool(getattr(settings, '_shadow_untraced', False))
    # the ambient lamps (Ambient Only): where the module has the one
    # predicate every reader skips them by, they shade nothing directly
    # and their map is only promised. Without it an Ambient Only lamp
    # still lights and shadows as a lamp, and is baked as one
    ambient = globals().get('is_ambient_lamp')
    roles = []
    for light in lights:
        if not wants_shadow_map(light, settings) or light.type == 'HEMI':
            roles.append(None)
        elif id(light) not in frame:
            roles.append('defer')
        elif not alone:
            roles.append('bake')
        elif light.type == 'SPOT' and getattr(light, 'screen_spot', False):
            roles.append('defer')
        elif ambient is not None and ambient(light):
            roles.append('defer')
        elif quiet and lamp_adds_nothing(light):
            roles.append('defer')
        else:
            roles.append('bake')
    return roles


def planned_map_bytes(scene, settings):
    """Bytes of depth the maps `build_shadow_maps` would bake now for
    this scene hold at most (every face counted whole)."""
    total = 0
    for light, role in zip(scene.lights or [], map_roles(scene, settings)):
        if role != 'bake':
            continue
        size = _map_size(light, settings)
        cube = light.type not in ('SUN', 'SPOT') or (
            light.type == 'SPOT' and spot_map_fov(light) > SPOT_CUBE_FOV)
        total += (6 if cube else 1) * size * size * 4
    return total


def map_deferred(light):
    """True while `light` holds a deferred map this frame: nothing is
    known to read its shadow. The GPU plan may answer 'no shadow term'
    for such a lamp (it asks only for `frame_lights`, and a frame lamp is
    deferred only when no surface reads its shadow on either road)."""
    return light.__dict__.get('_shadow_defer') is not None


def _no_promise():
    return None


class _Deferred:
    """A lamp's map, promised: baked through the cache the first time
    `visibility` looks it up. It crosses a pickle as nothing (a worker
    process plans its own frame) and a deep copy as itself."""

    __slots__ = ('scene', 'settings', 'casters', 'index')

    def __init__(self, scene, settings, casters, index):
        self.scene = scene
        self.settings = settings
        self.casters = casters
        self.index = index

    def __reduce__(self):
        return (_no_promise, ())

    def __deepcopy__(self, memo):
        return self


def _deferred_map(light):
    """Keep a deferred lamp's promise (see `map_roles`): bake its map
    now, once, whichever thread asks first."""
    rec = light.__dict__.get('_shadow_defer')
    if rec is None:
        return None
    with _BAKE_LOCK:
        if light.__dict__.get('_shadow_defer') is rec:
            lights = rec.scene.lights or []
            src = lights[rec.index] if rec.index < len(lights) else None
            if src is not None:
                _bake_lamps(_Frame(rec.scene, rec.settings, rec.casters),
                            [rec.index],
                            cache=getattr(rec.settings, 'cache_shadows',
                                          True))
                if src is not light:
                    light.shadow_map = src.shadow_map
                    src.__dict__.pop('_shadow_defer', None)
            light.__dict__.pop('_shadow_defer', None)
    return light.shadow_map


def drop_shadow_maps(scene):
    """Forget every lamp's map and promise. A frame that builds none --
    Ray Traced, Shadows off -- must not leave an earlier frame's map on
    a Scene that is rendered again: `visibility` falls through to it
    once the traced branch is off."""
    for light in (getattr(scene, 'lights', None) or ()):
        light.shadow_map = None
        light.__dict__.pop('_shadow_defer', None)
        light.__dict__.pop('_shadow_key', None)


# ------------------------------------------------------------ the cache

#: bytes of depth the cache may hold beyond what the frame in hand uses
SHADOW_CACHE_BYTES = 512 * 1024 * 1024

#: a static layer is kept while the objects that moved own at most this
#: share of the casters (past it a full bake costs about the same)
LAYER_MAX_MOVED = 1.0 / 3.0

#: the most objects a mesh may hold for its maps to be layered
LAYER_MAX_OBJECTS = 20000


def _plane_bytes(a):
    """Bytes a depth plane really holds (a blank plane holds none)."""
    if a is None or raster.is_blank(a) \
            or any(a is b for b in _BLANK_PLANES.values()):
        return 0
    return int(a.nbytes)


def _map_bytes(sm):
    faces = getattr(sm, 'faces', None) or [sm]
    return sum(_plane_bytes(f.depth) for f in faces)


class _Layer:
    """What a lamp's map keeps so that a moved object re-rasterises only
    itself: the plan it was baked for, the per-object digests at its
    last bake, and -- once something has moved -- the depth planes of
    everything that did NOT (`static`), the objects left out of them
    (`excl`) and the digests those planes were baked from."""

    __slots__ = ('blob', 'last', 'static', 'excl', 'static_od', 'maps')

    def nbytes(self):
        return sum(_plane_bytes(z) for z in (self.static or ()))

    def reset(self, excl=None, static_od=None):
        self.static = None
        self.excl, self.static_od = excl, static_od
        # the face maps of the static planes alone: a face the moved
        # objects do not reach IS its static map, built once
        self.maps = {}


class _ShadowCache(dict):
    """Baked maps by key, least recently used first, bounded by BYTES.

    The bound was a count: past 64 entries everything was dropped. A
    dragged lamp parks one map per pose, an animated mesh one per lamp
    per frame, so two 2048 point lamps held 6 GB after 32 frames -- and
    the clear then threw away the maps of the lamps that had not moved.
    Now the oldest entries leave while the depth held exceeds
    SHADOW_CACHE_BYTES plus what the frame in hand uses, and an entry
    that frame uses never leaves (a frame with more lamps than the
    budget would otherwise bake them in a circle)."""

    def __init__(self):
        super().__init__()
        self.layers = {}                # (base, lamp signature) -> _Layer
        self.frame = set()              # keys the frame in hand uses

    def clear(self):
        super().clear()
        self.layers.clear()
        self.frame.clear()

    def begin_frame(self):
        self.frame = set()

    def touch(self, key):
        """`key`'s map, marked as used by the frame in hand."""
        sm = self.get(key)
        if sm is not None:
            self.frame.add(key)
            # dicts keep insertion order: re-inserting moves to the end
            super().__delitem__(key)
            super().__setitem__(key, sm)
        return sm

    def store(self, key, sm):
        if key in self:
            super().__delitem__(key)
        super().__setitem__(key, sm)
        self.frame.add(key)
        self.trim()

    def held_bytes(self):
        seen, total = set(), 0
        for sm in self.values():
            if id(sm) not in seen:
                seen.add(id(sm))
                total += _map_bytes(sm)
        return total + sum(lay.nbytes() for lay in self.layers.values())

    def trim(self):
        used = sum(_map_bytes(self[k]) for k in self.frame if k in self)
        budget = SHADOW_CACHE_BYTES + used
        total = self.held_bytes()
        if total <= budget:
            return
        live = {(k[1], k[2]) for k in self.frame}
        for lk in list(self.layers):
            if total <= budget:
                return
            if (lk[1], lk[2]) in live:
                continue
            total -= self.layers.pop(lk).nbytes()
        for key in list(self.keys()):
            if total <= budget:
                return
            if key in self.frame:
                continue
            total -= _map_bytes(self[key])
            super().__delitem__(key)


_SHADOW_CACHE = _ShadowCache()

# one bake at a time touches the cache: a frame's own build, and the
# deferred maps its shading threads may ask for
_BAKE_LOCK = threading.RLock()

#: per-object vertex ranges by topology (see `_Frame.objects`)
_RANGE_CACHE = {}


def clear_shadow_cache():
    _SHADOW_CACHE.clear()
    _RANGE_CACHE.clear()


class _Frame:
    """What one frame's bakes share: the casters, the scene's bounds and
    the geometry's part of every cache key, read once."""

    def __init__(self, scene, settings, caster_tris, baker=None):
        self.scene = scene
        self.settings = settings
        self.casters = caster_tris
        # who bakes this frame's CLASSIC planes beside the NumPy fill
        # (None: nobody); see `_render_depth`
        self.baker = DEPTH_BAKER if baker is None else baker
        mesh = scene.mesh
        self.verts = mesh.verts
        self.tris = mesh.tris if caster_tris is None else \
            mesh.tris[caster_tris]
        self.centre, self.radius = scene_bounds(mesh.verts)
        self._base = None
        self._objects = False
        self._plans = {}
        self._splits = {}

    def base(self):
        if self._base is None:
            self._base = _shadow_base_signature(self.scene, self.settings,
                                                self.casters)
        return self._base

    def plan(self, li):
        p = self._plans.get(li)
        if p is None:
            p = self._plans[li] = _plan_map(self.scene.lights[li],
                                            self.settings, self.centre,
                                            self.radius)
        return p

    def key(self, li):
        base = self.base()
        sig = _light_shadow_signature(self.scene.lights[li], self.settings)
        if self.baker is not None and not self.plan(li).mid:
            # a baker's maps are kept apart from the fill's: a device
            # reproduces the fill's texels to the last ulp of z, not bit
            # for bit, and a CPU-device frame must never be served one
            # (a MIDPOINT map is the fill's whoever bakes the frame)
            sig = sig + (('baker', str(getattr(self.baker, 'identity',
                                               'baker'))),)
        return (base[0], base[1], sig)

    def split(self, excl):
        """(casters of every other object, casters of the objects in
        `excl`, the box around those objects' vertices)."""
        hit = self._splits.get(excl)
        if hit is None:
            obj_c, _od, rng, _okey = self.objects()
            sel = np.isin(obj_c, np.fromiter(excl, np.int64, len(excl)))
            hit = self._splits[excl] = (
                self.tris[~sel], self.tris[sel],
                _tri_bounds(self.verts, rng, excl))
        return hit

    def objects(self):
        """(object of each caster triangle, per-object digests, per-
        object vertex ranges, the topology's key), or None when the mesh
        cannot be layered: no object index, or objects whose vertices
        are not each one block of the vertex array (then one object's
        digest would not cover exactly its own casters)."""
        if self._objects is not False:
            return self._objects
        self._objects = None
        mesh = self.scene.mesh
        obj = getattr(mesh, 'obj_index', None)
        tris = mesh.tris
        if obj is None or tris is None or len(obj) != tris.shape[0] \
                or tris.shape[0] == 0:
            return None
        tkey = (self.base()[0][2], self.base()[0][3], int(len(obj)),
                M.content_digest(obj))
        rng = _RANGE_CACHE.get(tkey)
        if rng is None:
            obj = np.asarray(obj, np.int64)
            n_obj = int(obj.max()) + 1
            rng = ()
            if 0 <= int(obj.min()) and n_obj <= LAYER_MAX_OBJECTS:
                big = np.iinfo(np.int64).max
                lo = np.full(n_obj, big, np.int64)
                hi = np.full(n_obj, -1, np.int64)
                np.minimum.at(lo, obj, tris.min(axis=1).astype(np.int64))
                np.maximum.at(hi, obj, tris.max(axis=1).astype(np.int64))
                order = np.argsort(lo, kind='stable')
                order = order[hi[order] >= 0]
                if order.size and bool(
                        (hi[order][:-1] < lo[order][1:]).all()):
                    rng = (lo, hi)
            if len(_RANGE_CACHE) >= 8:
                _RANGE_CACHE.pop(next(iter(_RANGE_CACHE)))
            _RANGE_CACHE[tkey] = rng
        if not rng:
            return None
        lo, hi = rng
        v = self.verts
        if not v.flags.c_contiguous:
            return None
        digests = tuple(
            M.content_digest(v[int(a):int(b) + 1]) if b >= 0 else 0
            for a, b in zip(lo, hi))
        obj_c = np.asarray(mesh.obj_index)
        if self.casters is not None:
            obj_c = obj_c[self.casters]
        self._objects = (obj_c, digests, rng,
                         (self.base()[0][0],) + tkey)
        return self._objects


def _run_depth_jobs(frame, jobs):
    """Rasterise `jobs` -- (triangles, vp, size, midpoint, bounds) each
    -- and return their depth planes in order.

    The renders are independent: each map is its own buffer, its own
    depth min-compares, deterministic in isolation, so thread scheduling
    cannot move a bit. The rasteriser is big-array NumPy that releases
    the interpreter lock, which is the one shape of work threads
    genuinely scale on this renderer (the shading loop is not -- see the
    Threads tooltip). R254: the gate counts TEXELS as well as triangles
    -- the cost follows both, and a 29,000-triangle scene with one 2048
    cube used to bake its six faces one after another."""
    if not jobs:
        return []
    verts = frame.verts
    baker = frame.baker

    def one(j):
        return _render_depth(verts, j[0], j[1], j[2], midpoint=j[3],
                             bounds=j[4], baker=baker)

    if baker is not None:
        # a device is ONE queue: its dispatches cross to the main thread
        # one at a time (gpu/device._main), so the faces go in line
        return [one(j) for j in jobs]
    work = sum(int(j[2]) * int(j[2]) for j in jobs)
    tris = max(int(j[0].shape[0]) for j in jobs)
    if len(jobs) > 1 and (tris >= 50000 or work >= 2_000_000):
        import concurrent.futures as _fut
        import os as _os
        workers = min(len(jobs), max((_os.cpu_count() or 2) - 1, 2))
        with _fut.ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, jobs))
    return [one(j) for j in jobs]


def _tri_bounds(verts, rng, objects):
    """The box around the vertex blocks of `objects`."""
    lo, hi = rng
    los, his = [], []
    for k in objects:
        a, b = int(lo[k]), int(hi[k])
        if b < a:
            continue
        blk = verts[a:b + 1]
        los.append(blk.min(axis=0))
        his.append(blk.max(axis=0))
    if not los:
        return None
    return np.min(los, axis=0), np.max(his, axis=0)


def _bake_lamps(frame, indices, cache=True):
    """Bake the maps of the lamps at `indices` and hand each lamp its
    map; with `cache`, through the per-light store.

    A map whose lamp, casters and topology are the ones of an earlier
    bake but whose VERTICES moved is not rasterised whole again
    (R254). A CLASSIC texel is the minimum z over the fragments that
    land on it and a fragment's z depends on its own triangle and the
    map's frame only, so

        depth(all casters) = linearise(min(zndc(static), zndc(moved)))

    bit for bit (`ShadowMap._linearise` clips z to [-1, 1], so an empty
    texel's 1.0 and a fragment beyond the far plane land on one float).
    The first frame something moves bakes the layer of everything that
    did not -- about the cost of the full bake it replaces -- and every
    later frame of the same objects moving re-rasterises those objects
    alone and takes one minimum per face. It holds while the plan is
    bitwise the layer's (the scene's bounds move a Sun's frame and every
    far plane) and the casters, the topology and the object index are;
    a MIDPOINT map (its second surface is not a plain minimum), a mesh
    without an object index, or more than a third of the casters moving
    take the full bake."""
    lights = frame.scene.lights
    store = _SHADOW_CACHE if cache else None
    jobs = []
    full = []                 # (li, plan, key, first job, layer key, od)
    layered = []              # (li, plan, key, layer, static job, moved job, od)

    def face_jobs(plan, tris, mid, box):
        j0 = len(jobs)
        for vp, skip in zip(plan.faces, plan.skip):
            jobs.append((tris[:0] if skip else tris, vp, plan.size, mid,
                         box))
        return j0

    for li in indices:
        light = lights[li]
        plan = frame.plan(li)
        key = frame.key(li) if cache else None
        if store is not None:
            hit = store.touch(key)
            if hit is not None:
                light.shadow_map = hit
                light._shadow_key = key
                continue
        objs = frame.objects() if (store is not None and not plan.mid) \
            else None
        lkey = od = lay = None
        if objs is not None:
            od = objs[1]
            lkey = (objs[3], key[1], key[2])
            lay = store.layers.get(lkey)
            if lay is not None and (lay.blob != plan.blob()
                                    or len(lay.last) != len(od)):
                lay = None
        if lay is not None:
            rest = [k for k in range(len(od)) if k not in (lay.excl or ())]
            fresh = lay.static is None or any(
                od[k] != lay.static_od[k] for k in rest)
            s0 = None
            if fresh:
                # the layer of what did not move since this lamp's last
                # bake (a full bake's cost, once per gesture)
                excl = frozenset(k for k in range(len(od))
                                 if od[k] != lay.last[k])
                t_static, t_moved, _box = frame.split(excl)
                if t_moved.shape[0] <= LAYER_MAX_MOVED * frame.tris.shape[0]:
                    lay.reset(excl, od)
                    s0 = face_jobs(plan, t_static, False, None)
                    fresh = False
            if not fresh:
                _t_static, t_moved, box = frame.split(lay.excl)
                m0 = face_jobs(plan, t_moved, False, box)
                layered.append((li, plan, key, lay, s0, m0, od))
                continue
        j0 = face_jobs(plan, frame.tris, plan.mid, None)
        full.append((li, plan, key, j0, lkey, od))
    depths = _run_depth_jobs(frame, jobs)
    for li, plan, key, j0, lkey, od in full:
        sm = plan.build(depths[j0:j0 + len(plan.faces)])
        lights[li].shadow_map = sm
        if store is not None:
            lights[li]._shadow_key = key
            if lkey is not None:
                lay = _Layer()
                lay.blob, lay.last = plan.blob(), od
                lay.reset()
                store.layers[lkey] = lay
            store.store(key, sm)
    for li, plan, key, lay, s0, m0, od in layered:
        nf = len(plan.faces)
        if s0 is not None:
            lay.static = depths[s0:s0 + nf]
        maps = []
        for fi, (zs, zm) in enumerate(zip(lay.static,
                                          depths[m0:m0 + nf])):
            if raster.is_blank(zm):
                face = lay.maps.get(fi)
                if face is None:
                    face = lay.maps[fi] = plan.face(fi, zs)
            elif raster.is_blank(zs):
                face = plan.face(fi, zm)
            else:
                face = plan.face(fi, np.minimum(zs, zm))
            maps.append(face)
        lay.last = od
        sm = plan.join(maps)
        lights[li].shadow_map = sm
        lights[li]._shadow_key = key
        store.store(key, sm)


def _plan_frame(scene, settings, caster_tris, only=None, baker=None):
    """Reset the lamps (all, or the indices in `only`), decide each
    one's role and leave the deferred ones their promise. Returns
    (frame, indices to bake now), or (None, []) when nothing can be
    baked."""
    lights = scene.lights or []
    for li, light in enumerate(lights):
        if only is not None and li not in only:
            continue
        light.shadow_map = None
        light.__dict__.pop('_shadow_defer', None)
        light.__dict__.pop('_shadow_key', None)
    mesh = scene.mesh
    if mesh is None or mesh.verts is None or mesh.tris is None \
            or not mesh.verts.size:
        return None, []
    if (mesh.tris if caster_tris is None else caster_tris).shape[0] == 0:
        return None, []
    roles = map_roles(scene, settings)
    now = []
    for li, role in enumerate(roles):
        if role is None or (only is not None and li not in only):
            continue
        if role == 'bake':
            now.append(li)
        else:
            lights[li]._shadow_defer = _Deferred(scene, settings,
                                                 caster_tris, li)
    if not now:
        return None, []
    return _Frame(scene, settings, caster_tris, baker=baker), now


def build_shadow_maps(scene, settings, caster_tris=None, baker=None):
    """Bake a shadow map (or cube) for every light a reader will look up
    (`map_roles`).

    Each map is a rasterisation pass, and a point light needs six. None
    of it changes while the lights and geometry hold still, which for
    most of an animation is all of it. The cache is PER LIGHT: the old
    whole-list fingerprint meant nudging ONE lamp of fifteen
    re-rasterised every map in the scene ('if I wanted to move
    lights... it will lag like crazy'); now only the moved lamp's own
    map rebuilds, a colour, energy or size edit rebuilds nothing at
    all, and a moved OBJECT re-rasterises itself (`_bake_lamps`).

    `baker` is the frame's depth baker (see `_render_depth`): the GPU
    device's on a GPU-device frame, None otherwise. A map promised to a
    later lookup (`map_roles`: 'defer') is baked by the fill when it is
    asked for -- that lookup comes from a shading thread.
    """
    with _BAKE_LOCK:
        cache = bool(getattr(settings, 'cache_shadows', True))
        if cache:
            _SHADOW_CACHE.begin_frame()
        frame, now = _plan_frame(scene, settings, caster_tris, baker=baker)
        if frame is not None:
            _bake_lamps(frame, now, cache=cache)


def _build_shadow_maps(scene, settings, caster_tris=None, only=None):
    """Build maps past the cache for every light that is read, or --
    with `only`, a set of light indices -- just those, leaving the rest
    untouched."""
    with _BAKE_LOCK:
        frame, now = _plan_frame(scene, settings, caster_tris, only=only)
        if frame is not None:
            _bake_lamps(frame, now, cache=False)


def adopt_shadow_maps(scene):
    """Park the maps a scene's lamps already carry under the keys they
    were baked for. A worker process calls it on the scene it is handed:
    the parent baked each map once, and the worker's own build then
    finds every one of them instead of rasterising the set again for its
    band (four workers used to do four bakes on the same cores)."""
    with _BAKE_LOCK:
        for light in (getattr(scene, 'lights', None) or ()):
            key = light.__dict__.get('_shadow_key')
            if key is not None and light.shadow_map is not None:
                _SHADOW_CACHE.store(key, light.shadow_map)
        _SHADOW_CACHE.begin_frame()



# ------------------------------------------------------------- light sampling


def resolved_decay(light, settings=None):
    """The falloff law a lamp actually runs: its own Falloff, or the
    render's Light Falloff while the lamp is left on Scene Default.

    R254: the readers of the law's NAME -- the Blender Internal 0.001
    snap in sample() and the volumetric beam's reach here, the GPU
    packer and emitter and the panel's slider rows by the same two
    lines against their own settings -- used to test the lamp's own
    unresolved 'DEFAULT', and so treated a Scene Default lamp
    differently from the same law chosen on the lamp."""
    mode = str(getattr(light, 'decay', 'DEFAULT') or 'DEFAULT')
    if mode == 'DEFAULT' and settings is not None:
        mode = str(getattr(settings, 'light_falloff_default', 'DEFAULT'))
    return mode


def attenuate(light, dist, settings=None):
    mode = light.decay
    if mode == 'DEFAULT' and settings is not None:
        mode = settings.light_falloff_default
    start = float(light.decay_start)
    # R254: Falloff Start is a Custom Range value and nothing else. The
    # two inverse laws used to divide by (dist - start): a Start left
    # over from a Custom Range lamp (the panel draws the slider for
    # that law only) put a pole at d = Start and 1/EPS inside it -- the
    # whole disc under the lamp at the ceiling. Start 0, the default,
    # is the same statement it always was
    d = np.maximum(dist, EPS)
    if mode == 'NONE':
        # R253: no early return any more -- the Sphere wrap below is
        # the SAME statements this branch used to run (LA_SPHERE
        # applies OUTSIDE the falloff switch in the C, so even a
        # Constant-falloff lamp clips at its Distance), and the Near
        # Attenuation tail must see every curve
        att = np.ones_like(dist)
    elif mode == 'INVERSE':
        att = 1.0 / d
    elif mode == 'CUSTOM':
        end = max(float(light.decay_end), start + EPS)
        att = np.clip(1.0 - (dist - start) / (end - start), 0.0, 1.0)
    elif mode == 'BI_LINEAR':
        # Blender Internal's Inverse Linear: D / (D + d), where D is
        # the lamp's Distance (decay_end). Bounded at 1, halves at D --
        # the curve every classic file was lit against, and nothing
        # like an unbounded 1/d
        D = max(float(light.decay_end), EPS)
        att = D / (D + np.maximum(dist, 0.0))
    elif mode == 'BI_SQUARE':
        # Blender Internal's Inverse Square, verbatim from 2.79's
        # lamp_get_visibility (R155): D / (D + d*d). The C annotates it
        # as the r12045 'hack' itself -- a true inverse square would be
        # D^2/(D^2+d^2), which is what this shipped as until the source
        # was letter-checked: 3-4x too bright over the field's typical
        # D=25 rigs.
        D = max(float(light.decay_end), EPS)
        att = D / (D + np.maximum(dist, 0.0) ** 2)
    elif mode == 'BI_SLIDERS':
        # verbatim: D/(D+ld1*d) * D^2/(D^2+ld2*d^2), each factor only
        # when its slider is positive -- the 2.4x Quad lamp's att1/att2
        D = max(float(light.decay_end), EPS)
        ld1 = float(getattr(light, 'decay_ld1', 0.0) or 0.0)
        ld2 = float(getattr(light, 'decay_ld2', 0.0) or 0.0)
        att = np.ones_like(dist)
        if ld1 > 0.0:
            att = att * (D / (D + ld1 * np.maximum(dist, 0.0)))
        if ld2 > 0.0:
            att = att * ((D * D) / (D * D + ld2
                                    * np.maximum(dist, 0.0) ** 2))
    elif mode in DECAY_LAWS_R251 or mode in DECAY_LAWS_R253:
        # R251 F013: OpenGL's three-term attenuation, POV-Ray's fade
        # laws and the GameCube's GX distance tables -- float32, one op
        # per statement, exactly the GLSL `decay_law_glsl` emits.
        # R253 (D): the two nominal-distance laws ride the same road
        att = decay_law(light, mode, dist)
    else:
        att = 1.0 / (d * d)
    if getattr(light, 'bi_sphere', False):
        # LA_SPHERE, verbatim: *= (D - d)/D, hard zero past the lamp's
        # Distance
        D = max(float(light.decay_end), EPS)
        t = D - np.maximum(dist, 0.0)
        att = np.where(t <= 0.0, 0.0, att * t / D)
    ns = float(getattr(light, 'decay_near_start', 0.0) or 0.0)
    ne = float(getattr(light, 'decay_near_end', 0.0) or 0.0)
    if ne > ns:
        # R253 (E): 3ds Max's Near Attenuation -- the light fades IN
        # from nothing at Start to full at End (the Attenuation
        # Parameters rollout; the manual draws the ramp and names no
        # curve, so it is linear here). Float32, one op per statement,
        # exactly the GLSL lines _one_light_source emits after `att`
        # is final; inert (no statement at all) while End <= Start
        att = near_attenuation(att, dist, ns, ne)
    return att.astype(np.float32)


def near_attenuation(att, dist, ns, ne):
    """R253 (E): att times the linear fade-in over [ns, ne] of max(dist,
    0) -- the CPU half of the GLSL `at_nt` statements."""
    dm = np.maximum(np.asarray(dist, np.float32), np.float32(0.0))
    span = np.float32(ne - ns)
    t = dm - np.float32(ns)
    t = t / span
    t = np.clip(t, np.float32(0.0), np.float32(1.0))
    return (np.asarray(att, np.float32) * t).astype(np.float32)


def spot_falloff(light, L):
    """L points surface -> light, so the cone test uses -L.

    Blender Internal's spot, verbatim from lamp_get_visibility (R155):
    spotsi = cos(spot_size/2) is the hard cutoff; the blend band is
    spotbl = (1-spotsi)*spot_blend wide IN COSINE UNITS with a
    smoothstep across it; and the whole cone then MULTIPLIES by the
    raw cosine, so a spot dims toward its own edge even outside the
    blend band. The old shape (a squared smoothstep between two
    invented cosines) was both too dark in the band and missing the
    cosine roll-off -- the field's spot rigs read wrong both ways."""
    d = M.normalize(np.asarray(light.direction, np.float32))
    inpr = -M.dot(L, np.broadcast_to(d[None, :], L.shape))
    if spot_scale_on(light):
        # R254: a Spot scaled unevenly across its beam -- Blender's
        # elliptical cone. The cosine every law reads is taken after
        # the narrower axis is squeezed (the Square cone's max rides
        # the same squeezed pair: a rectangular pyramid)
        inpr = scaled_cos(light, L, inpr)
    elif getattr(light, 'spot_square', False):
        # R253 (A): BI's LA_SQUARE -- the cosine the cone law reads is
        # measured along the wider of the lamp's two sides, so every
        # law (BI blend, GL11, POV, GX) gets a square version
        inpr = square_cos(light, L, inpr)
    return spot_law_factor(light, inpr)


def spot_scale_on(light):
    """R254: True when the lamp carries an elliptical cone -- its
    spot_scale pair is not (1, 1). A lamp without the field (a
    hand-built Light of an older shape) is round."""
    sc = getattr(light, 'spot_scale', None)
    if sc is None:
        return False
    try:
        return (float(sc[0]), float(sc[1])) != (1.0, 1.0)
    except (TypeError, ValueError, IndexError):
        return False


def scaled_cos(light, L, z):
    """R254: the cosine to the axis of an ELLIPTICAL cone, Blender's
    light_spot_attenuation (eevee_light_lib): the direction in lamp
    space with its side / up components multiplied by spot_scale, then
    1/sqrt(1 + (x/z)^2 + (y/z)^2); zero behind the lamp. With Square
    the wider of the two squeezed sides is taken instead of their sum
    (square_cos's max). `z` is the plain cosine (-L . direction), the
    frame is cookie_frame's. Float32, one op per statement -- the CPU
    half of the GLSL `sc_*` lines."""
    s, u, _f = cookie_frame(light)
    sc = getattr(light, 'spot_scale', (1.0, 1.0))
    x = -(L @ s.astype(np.float32))
    y = -(L @ u.astype(np.float32))
    x = x * np.float32(sc[0])
    y = y * np.float32(sc[1])
    zz = np.maximum(np.asarray(z, np.float32), np.float32(1e-20))
    with np.errstate(over='ignore', invalid='ignore'):
        # behind the lamp x/zz overflows; the select below discards
        # that lane, exactly the GLSL ternary
        ax = x / zz
        ay = y / zz
        if getattr(light, 'spot_square', False):
            ax = np.abs(ax)
            ay = np.abs(ay)
            m = np.maximum(ax, ay)
            q = m * m
        else:
            q = ax * ax
            t = ay * ay
            q = q + t
        q = np.float32(1.0) + q
        c = np.float32(1.0) / np.sqrt(q)
    return np.where(z > 0.0, c, np.float32(0.0)).astype(np.float32)


def square_cos(light, L, z):
    """R253 (A): Blender Internal's Square spot, verbatim from 2.79
    shadeoutput.c lamp_get_visibility's LA_SQUARE block: the view
    vector in lamp space, x = max(|x/z|, |y/z|), inpr = 1/sqrt(1 + x^2)
    (cos(atan(x))); zero behind the lamp. `z` is the plain cosine to
    the axis (-L . direction), the frame is cookie_frame's (side, up),
    so a projected slide fills the square corner to corner. Float32,
    one op per statement -- the CPU half of the GLSL `sq_*` lines."""
    s, u, _f = cookie_frame(light)
    x = -(L @ s.astype(np.float32))
    y = -(L @ u.astype(np.float32))
    zz = np.maximum(np.asarray(z, np.float32), np.float32(1e-20))
    with np.errstate(over='ignore', invalid='ignore'):
        # behind the lamp x/zz overflows to inf (and 1/sqrt(inf) to 0);
        # the select below discards that lane, exactly the GLSL ternary
        ax = np.abs(x / zz)
        ay = np.abs(y / zz)
        m = np.maximum(ax, ay)
        q = m * m
        q = np.float32(1.0) + q
        c = np.float32(1.0) / np.sqrt(q)
    return np.where(z > 0.0, c, np.float32(0.0)).astype(np.float32)


def spot_blend_band(light, spotsi):
    """R254: the width of the Blender Internal cone's smoothstep band, in
    cosine units, measured inward from the cutoff cosine `spotsi`.

    Hotspot 0 (the default): Blender's Blend, (1 - spotsi) * spot_blend
    -- the caller's own `spotsi` (the CPU's float32, the packer's
    double) in the expression each site always evaluated, so such a
    lamp moves no bit. Hotspot above 0: the band runs from the cone
    edge to the HOTSPOT's cosine, cos(min(hotspot, spot_size)/2) -
    cos(spot_size/2) -- full strength inside the inner angle, the
    smoothstep down to the Spot Size, the 3D Studio hotspot / falloff
    pair. Computed in float64 and rounded once by the caller, so the
    CPU's band and the GPU's texel are the same float32. A hotspot at
    or past the cone gives 0: the hard edge.

    The ONE place the band is computed: _spot_blender, the GPU packer
    (hal_lights texel 2.w), the baked literal, the plan signature's
    "a blend exists" bit and the visible beam's edge all call it.
    """
    hot = float(getattr(light, 'hotspot', 0.0) or 0.0)
    if hot > 0.0:
        size = float(light.spot_size)
        return float(np.cos(min(hot, size) * 0.5)) \
            - float(np.cos(size * 0.5))
    return (1.0 - spotsi) * float(light.spot_blend)


def _spot_blender(light, inpr):
    """Blender Internal's cone, verbatim (the 1.89.0 spot_falloff body);
    R254: the band is spot_blend_band's -- Blend, or the Hotspot's."""
    spotsi = np.float32(np.cos(float(light.spot_size) * 0.5))
    spotbl = np.float32(spot_blend_band(light, spotsi))
    t = inpr - spotsi
    if spotbl != 0.0:
        i = np.clip(t / spotbl, 0.0, 1.0)
        soft = np.where(t < spotbl, 3.0 * i * i - 2.0 * i * i * i, 1.0)
    else:
        soft = np.ones_like(inpr)
    fac = np.where(inpr <= spotsi, 0.0, soft * inpr)
    return fac.astype(np.float32)


def cookie_frame(light):
    """(side, up, forward) unit axes for a light's projected texture.

    Export fills frame_x/frame_y from the object matrix so the image turns
    with the lamp, exactly as a slide in a projector would. A hand-built
    light without them gets a stable derived basis.
    """
    f = M.normalize(np.asarray(light.direction, np.float32))
    fx = getattr(light, 'frame_x', None)
    fy = getattr(light, 'frame_y', None)
    if fx is not None and fy is not None:
        return (M.normalize(np.asarray(fx, np.float32)),
                M.normalize(np.asarray(fy, np.float32)), f)
    s, u = M.orthonormal_basis(f[None, :])
    return s[0], u[0], f


def cookie_spot_axes(light, s, u):
    """R254: a Spot's slide axes with the elliptical cone's squeeze
    folded in: (side * sx, up * sy) as float32. ONE place, read by the
    CPU lookup and by the GPU's baked literals, so both project the
    same slide."""
    sc = getattr(light, 'spot_scale', (1.0, 1.0))
    return (np.asarray(s, np.float32) * np.float32(sc[0]),
            np.asarray(u, np.float32) * np.float32(sc[1]))


#: R219: the extension each projection always had, before it was a
#: choice -- SUN/HEMI tile their cloud shadow, a SPOT's cone edge lands
#: on the image edge and clamps, a POINT's lat-long wraps around its own
#: seam, an AREA's face clamps at its border.
_COOKIE_AUTO_EXTEND = {'SUN': 'REPEAT', 'HEMI': 'REPEAT',
                       'SPOT': 'EXTEND', 'POINT': 'REPEAT',
                       'AREA': 'EXTEND'}


def cookie_extend(light):
    """The light's resolved cookie extension: REPEAT, EXTEND or CLIP."""
    mode = str(getattr(light, 'cookie_extend', 'AUTO') or 'AUTO').upper()
    if mode == 'AUTO':
        return _COOKIE_AUTO_EXTEND.get(getattr(light, 'type', 'POINT'),
                                       'EXTEND')
    return mode


def cookie_filter(light):
    """The light's cookie filter, in the Texture sampler's vocabulary."""
    f = str(getattr(light, 'cookie_filter', 'BILINEAR')
            or 'BILINEAR').upper()
    return 'NEAREST' if f == 'CLOSEST' else f


def cookie_blur_uv(light):
    """Per-axis focus-blur radius in UV units, from the lamp's SOURCE
    size -- the slider the lamp already has (R219 field: "The Angle
    slider should effect how blurry it is").

    A big light source smears anything it projects; the mapping per
    projection, with `radius` the exported Angle (SUN/HEMI, radians,
    full diameter) or Radius (others, scene units):

    - SUN/HEMI: the slide floats one tile above the world, so an
      angular radius a spreads each point by tan(a) of a tile:
      tan(radius/2).
    - SPOT: the gate sits one unit down the throw and spans
      2*tan(spot_size/2); a source disc of radius r seen across it:
      r / (2*tan(spot_size/2)).
    - POINT: the source's angular radius from one unit away, over the
      lat-long span: atan(r)/2pi across, atan(r)/pi down.
    - AREA: the printed gel is the size of the face; the source radius
      over the face: r/size_x, r/size_y.

    0 (the dataclass default) means razor sharp -- and bitwise exactly
    the pre-R219 image, because a sub-texel sigma collapses the box
    widths to 1 and the preparation returns the source array untouched.
    """
    r = float(getattr(light, 'radius', 0.0) or 0.0)
    if r <= 0.0:
        return (0.0, 0.0)
    kind = getattr(light, 'type', 'POINT')
    if kind in ('SUN', 'HEMI'):
        b = float(np.tan(min(r, 3.0) * 0.5))
        return (b, b)
    if kind == 'SPOT':
        tanh = max(np.tan(float(getattr(light, 'spot_size', 1.2)) * 0.5),
                   1e-6)
        b = r / (2.0 * float(tanh))
        if spot_scale_on(light):
            # R254: the elliptical cone's gate is narrower across the
            # squeezed axis, so the same source disc covers more of it
            sc = light.spot_scale
            return (b * float(sc[0]), b * float(sc[1]))
        return (b, b)
    if kind == 'POINT':
        a = float(np.arctan(r))
        return (a / (2.0 * np.pi), a / np.pi)
    if kind == 'AREA':
        try:
            sx, sy = (float(v) for v in
                      getattr(light, 'area_size', (1.0, 1.0))[:2])
        except Exception:                                       # noqa: BLE001
            sx = sy = 1.0
        return (r / max(sx, 1e-6), r / max(sy, 1e-6))
    return (0.0, 0.0)


def _box_passes(sigma):
    """Three box half-widths that together approximate a gaussian of
    `sigma` texels -- the classic triple-box, integer widths, O(n) at
    ANY radius (the R167 rule: a huge Angle costs the same as a small
    one). Returns a list of ODD box widths; all 1 = identity."""
    if sigma < 0.58:
        return [1, 1, 1]
    n = 3.0
    wi = np.sqrt(12.0 * sigma * sigma / n + 1.0)
    wl = int(np.floor(wi))
    if wl % 2 == 0:
        wl -= 1
    wl = max(wl, 1)
    wu = wl + 2
    mi = (12.0 * sigma * sigma - n * wl * wl - 4.0 * n * wl - 3.0 * n) \
        / (-4.0 * wl - 4.0)
    m = int(round(mi))
    m = min(max(m, 0), 3)
    return [wl] * m + [wu] * (3 - m)


def _box1d(a, width, axis, pad_mode):
    """One box pass along `axis` via a float64 running sum -- exact,
    deterministic, radius-independent cost."""
    if width <= 1:
        return a
    n = a.shape[axis]
    half = min((width - 1) // 2, n)
    w = 2 * half + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (half, half)
    kw = {'mode': pad_mode}
    if pad_mode == 'constant':
        kw['constant_values'] = 0.0
    padded = np.pad(a, pad, **kw)
    c = np.cumsum(padded, axis=axis, dtype=np.float64)
    zshape = list(c.shape)
    zshape[axis] = 1
    c = np.concatenate([np.zeros(zshape, np.float64), c], axis=axis)
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(w, w + n)
    lo[axis] = slice(0, n)
    return ((c[tuple(hi)] - c[tuple(lo)]) / float(w)).astype(np.float32)


_COOKIE_PAD = {'REPEAT': 'wrap', 'EXTEND': 'edge', 'CLIP': 'constant'}


def cookie_fingerprint(light, fresh=False):
    """R254: a fingerprint of EVERY texel of the lamp's cookie -- the
    crc32 of its float32 bytes -- or None without one.

    The cookie's caches (the blurred preparation, the CPU sampler's
    texture, the GPU plan and the device upload) were keyed on the sum
    of every 7th texel: 48 of 49 texels were never read, and a mirror
    or half-turn of an image whose width is 1 mod 7 (8, 64, 512, 4096
    px) maps that lattice onto itself, so a flipped slide or a small
    painted detail kept serving the old image.

    The crc costs about 0.3 ms per megabyte, and the shading loop asks
    many times a frame, so it is remembered on the lamp against the
    pixel ARRAY it was taken from (a held reference: an address cannot
    be reused under it). A new array is fingerprinted on sight;
    `fresh=True` recomputes -- render() does so once per frame and the
    GPU plan signature on every plan, which is what notices an array
    edited in place.
    """
    ck = getattr(light, 'cookie', None)
    if ck is None:
        return None
    src = getattr(ck, 'pixels', ck)
    memo = getattr(light, '_cookie_fp', None)
    if not fresh and isinstance(memo, tuple) and memo[0] is src:
        return memo[1]
    try:
        import zlib
        px = np.ascontiguousarray(np.asarray(src, np.float32))
        fp = int(zlib.crc32(px.tobytes() if px.size < 4096 else px))
    except Exception:                                           # noqa: BLE001
        return None
    try:
        light._cookie_fp = (src, fp)
    except Exception:                                           # noqa: BLE001
        pass
    return fp


def refresh_cookie_fingerprints(lights):
    """R254: fingerprint every cookie's texels for this frame (see
    cookie_fingerprint) -- one crc per cookie lamp, before any reader."""
    for light in lights or ():
        if getattr(light, 'cookie', None) is not None:
            cookie_fingerprint(light, fresh=True)


def cookie_pixels(light):
    """The image a lamp actually projects -- prepared ONCE and shared
    verbatim by the CPU sampler and the GPU upload, so both devices read
    identical texels (the determinism doctrine applied to gobos).

    Preparation is the focus blur: the lamp's source size (Angle /
    Radius) smeared into the image itself by the triple-box gaussian,
    padded in the light's own extension mode so a tiling cloud shadow
    blurs across its seam and a projector gate darkens toward its edge.
    Cached on the light, keyed on the source pixels and every dial that
    shapes the result; a caustic lamp's synthesized web is exempt (it
    re-bakes every frame and stays crisp, exactly as before R219).
    Returns an (H,W,C) float32 array, or None without a cookie.
    """
    ck = getattr(light, 'cookie', None)
    if ck is None:
        return None
    src = getattr(ck, 'pixels', ck)
    px = np.asarray(src, np.float32)
    if px.ndim == 2:
        px = px[:, :, None]
    if getattr(light, '_caustic_cookie', False):
        bu = bv = 0.0
    else:
        bu, bv = cookie_blur_uv(light)
    ext = cookie_extend(light)
    # R254: keyed on every texel (cookie_fingerprint), not on the sum
    # of every 7th
    key = (px.shape, cookie_fingerprint(light),
           round(float(bu), 9), round(float(bv), 9), ext)
    prep = getattr(light, '_cookie_prep', None)
    if isinstance(prep, tuple) and prep[0] == key:
        return prep[1]
    h, w = px.shape[:2]
    su = float(bu) * w * 0.5
    sv = float(bv) * h * 0.5
    if su < 0.29 and sv < 0.29:
        out = px                       # identity: the pre-R219 bytes
    else:
        pad = _COOKIE_PAD.get(ext, 'edge')
        out = px
        for bw in _box_passes(su):
            out = _box1d(out, bw, 1, pad)
        for bw in _box_passes(sv):
            out = _box1d(out, bw, 0, pad)
    try:
        light._cookie_prep = (key, out)
    except Exception:                                           # noqa: BLE001
        pass
    return out


def _cookie_texture(light):
    px = cookie_pixels(light)
    key = getattr(light, '_cookie_prep', (None, None))[0]
    ct = getattr(light, '_cookie_tex', None)
    if isinstance(ct, tuple) and ct[0] == key:
        return ct[1]
    from .texture import Texture
    tex = Texture(np.asarray(px, np.float32), name='cookie',
                  colorspace='Non-Color')
    try:
        light._cookie_tex = (key, tex)
    except Exception:                                           # noqa: BLE001
        pass
    return tex


def cookie_xform_on(light):
    """R253 (G): whether the cookie carries a non-identity UV transform
    (an offset, a tiling other than 1 or a rotation) -- structure on
    the GPU road (the lookup function opens with the transform or not),
    and the gate on the CPU's own statements."""
    try:
        ox, oy = (float(v) for v in getattr(light, 'cookie_offset',
                                            (0.0, 0.0))[:2])
        tx, ty = (float(v) for v in getattr(light, 'cookie_tile',
                                            (1.0, 1.0))[:2])
    except Exception:                                           # noqa: BLE001
        return False
    rot = float(getattr(light, 'cookie_rotation', 0.0) or 0.0)
    return ox != 0.0 or oy != 0.0 or tx != 1.0 or ty != 1.0 or rot != 0.0


def cookie_xform_consts(light):
    """R253 (G): (ox, oy, tx, ty, c, s) float32 -- the one place the
    transform's numbers are computed; the GPU bakes exactly these as
    literals (the cookie road's convention: side/up/strength are
    literals too)."""
    ox, oy = (float(v) for v in getattr(light, 'cookie_offset',
                                        (0.0, 0.0))[:2])
    tx, ty = (float(v) for v in getattr(light, 'cookie_tile',
                                        (1.0, 1.0))[:2])
    rot = float(getattr(light, 'cookie_rotation', 0.0) or 0.0)
    return (np.float32(ox), np.float32(oy), np.float32(tx), np.float32(ty),
            np.float32(np.cos(rot)), np.float32(np.sin(rot)))


def cookie_xform(light, cu, cv):
    """R253 (G): the projector map's Coordinates rollout (3ds Max) and
    the 2.4x lamp texture's ofs/size on the cookie's uv -- a rotation
    about the slide's centre, then the tiling, then the offset, float32
    and one op per statement (the GLSL `ck_*` lines are these
    statements). The CLIP gate and the focus blur stay in texture space
    after it, untouched."""
    ox, oy, tx, ty, c, s = cookie_xform_consts(light)
    u = np.asarray(cu, np.float32) - np.float32(0.5)
    v = np.asarray(cv, np.float32) - np.float32(0.5)
    u2 = u * c
    t = v * s
    u2 = u2 - t
    v2 = u * s
    t2 = v * c
    v2 = v2 + t2
    u2 = u2 * tx
    v2 = v2 * ty
    cu = u2 + np.float32(0.5)
    cu = cu + ox
    cv = v2 + np.float32(0.5)
    cv = cv + oy
    return cu.astype(np.float32), cv.astype(np.float32)


def cookie_factor(light, P, L):
    """Per-point rgb multiplier from a light's projected texture.

    The sixth-generation consoles' projective texturing, one mapping per
    lamp kind (R219: ALL of them project):

    - SPOT maps its full cone onto the image -- the cone edge lands on
      the image edge, a slide projector (Splinter Cell's windows).
    - SUN/HEMI project along their rays and tile every `cookie_scale`
      world units -- the scrolling cloud shadow of the era.
    - POINT wraps the image around itself lat-long, seam at the back:
      a pierced lantern, a disco ball.
    - AREA carries the image on its face like a printed gel and throws
      it straight forward, one face-width wide.

    Filter and extension are the light's cookie_filter/cookie_extend
    (CLIP reads zero outside the slide: the projector's gate, so at
    strength 1 there is no light past the image). The GLSL mirror
    writes out the same texel arithmetic. Returns (N,3), all ones where
    the projection is undefined (behind a spot or an area's face).
    """
    n = P.shape[0]
    kind = getattr(light, 'type', 'POINT')
    if getattr(light, 'cookie', None) is None:
        return np.ones((n, 3), np.float32)
    s, u, f = cookie_frame(light)
    tex = _cookie_texture(light)
    wrap = cookie_extend(light)
    filt = cookie_filter(light)
    strength = float(np.clip(getattr(light, 'cookie_strength', 1.0), 0.0, 1.0))
    # R253 (G): the uv transform, applied right before every lookup and
    # ONLY when it is not the identity (no statement at the default)
    xf = cookie_xform_on(light)
    if kind in ('SUN', 'HEMI'):
        scale = max(float(getattr(light, 'cookie_scale', 10.0)), 1e-6)
        cu = (P @ s.astype(np.float32)) / scale
        cv = (P @ u.astype(np.float32)) / scale
        if xf:
            cu, cv = cookie_xform(light, cu, cv)
        rgb = tex.sample(cu.astype(np.float32), cv.astype(np.float32),
                         filt=filt, wrap=wrap)[:, :3]
        return (1.0 + (rgb - 1.0) * strength).astype(np.float32)
    if kind == 'POINT':
        # light -> surface direction in the lamp's own frame, unrolled
        # to lat-long: u around the forward axis (atan2, seam at the
        # back), v from the equator (asin) -- both land inside [0,1]
        d = -L
        dx = d @ s.astype(np.float32)
        dy = d @ u.astype(np.float32)
        dz = d @ f.astype(np.float32)
        cu = np.arctan2(dx, dz) * np.float32(1.0 / (2.0 * np.pi)) + 0.5
        cv = np.arcsin(np.clip(dy, -1.0, 1.0)) * np.float32(1.0 / np.pi) \
            + 0.5
        if xf:
            cu, cv = cookie_xform(light, cu, cv)
        rgb = tex.sample(cu.astype(np.float32), cv.astype(np.float32),
                         filt=filt, wrap=wrap)[:, :3]
        return (1.0 + (rgb - 1.0) * strength).astype(np.float32)
    if kind == 'AREA':
        # parallel throw off the face: offset in the lamp plane over the
        # face size; behind the face the projection is undefined
        try:
            sx, sy = (float(v) for v in
                      getattr(light, 'area_size', (1.0, 1.0))[:2])
        except Exception:                                       # noqa: BLE001
            sx = sy = 1.0
        rel = P - np.asarray(light.position, np.float32)[None, :]
        dz = rel @ f.astype(np.float32)
        cu = (rel @ s.astype(np.float32)) / max(sx, 1e-6) + 0.5
        cv = (rel @ u.astype(np.float32)) / max(sy, 1e-6) + 0.5
        if xf:
            cu, cv = cookie_xform(light, cu, cv)
        rgb = tex.sample(cu.astype(np.float32), cv.astype(np.float32),
                         filt=filt, wrap=wrap)[:, :3]
        out = 1.0 + (rgb - 1.0) * strength
        out[dz <= 1e-6] = 1.0
        return out.astype(np.float32)
    # SPOT: direction light -> surface, expressed in the light's own frame;
    # the full cone spans the image, so uv = d_side / (d_fwd * 2 tan(half))
    d = -L
    dz = d @ f.astype(np.float32)
    tanh = max(np.tan(float(light.spot_size) * 0.5), 1e-6)
    safe = np.maximum(dz, 1e-6) * (2.0 * tanh)
    if spot_scale_on(light):
        # R254: the slide is squeezed with the elliptical cone -- its
        # edges stay on the cone's edges (the frame axes carry the
        # scale, as the GPU bakes them)
        s, u = cookie_spot_axes(light, s, u)
    cu = (d @ s.astype(np.float32)) / safe + 0.5
    cv = (d @ u.astype(np.float32)) / safe + 0.5
    if xf:
        cu, cv = cookie_xform(light, cu, cv)
    rgb = tex.sample(cu.astype(np.float32), cv.astype(np.float32),
                     filt=filt, wrap=wrap)[:, :3]
    out = 1.0 + (rgb - 1.0) * strength
    out[dz <= 1e-6] = 1.0
    return out.astype(np.float32)


def sample(light, P, settings, area_sample=None):
    """Direction to the light, incoming radiance and distance.

    Returns (L (N,3) unit, radiance (N,3), dist (N,)).
    """
    n = P.shape[0]
    col = np.asarray(light.color, np.float32)[None, :]
    # R253 (H): the light style's gain for this frame (1.0 exactly for
    # NORMAL, so x * 1.0 is the old double bitwise); the same python
    # double the GPU packer scales by
    energy = float(light.energy) * float(getattr(light, 'style_gain', 1.0))
    if light.type in ('SUN', 'HEMI'):
        # HEMI is directional exactly like SUN -- what changes is the
        # SHADING (the 0.5+0.5*N.L wrap, in light_surface), not the
        # geometry. BI hemis had no distance and no falloff.
        d = M.normalize(np.asarray(light.direction, np.float32))
        L = np.broadcast_to(-d[None, :], (n, 3)).copy()
        dist = np.full(n, 1e9, np.float32)
        rad = np.broadcast_to(col * energy, (n, 3)).copy()
        if getattr(light, 'cookie', None) is not None:
            rad = rad * cookie_factor(light, P, L)
        return L, rad.astype(np.float32), dist
    pos = np.asarray(light.position, np.float32)
    if light.type == 'AREA' and area_sample is not None:
        pos = area_sample
        delta = pos[None, :] - P if pos.ndim == 1 else pos - P
    else:
        delta = pos[None, :] - P
    dist = np.sqrt(np.maximum((delta * delta).sum(axis=1), EPS)).astype(np.float32)
    L = delta / dist[:, None]
    if light.type == 'AREA':
        # lamp_get_visibility SKIPS its falloff switch for LA_AREA:
        # visifac stays 1.0, and every bit of an area lamp's distance
        # behaviour lives in the form factor (area_inp) instead --
        # dist^2/A times the Stokes contour. Decay dials, Sphere and
        # the 0.001 snap never applied to area lamps in 2.79
        att = np.ones_like(dist)
    else:
        att = attenuate(light, dist, settings)
    if light.type == 'SPOT':
        att = att * spot_falloff(light, L)
    if resolved_decay(light, settings).startswith('BI_') or \
            getattr(light, 'bi_sphere', False):
        # lamp_get_visibility's tail, verbatim (R155): a combined
        # visifac at or below 0.001 is snapped to zero. R254: on the
        # RESOLVED law -- a Scene Default lamp under a Blender Internal
        # render default is the same lamp as one with the law chosen
        att = np.where(att <= 0.001, 0.0, att)
    scale = energy / (4.0 * np.pi)
    rad = col * (scale * att)[:, None]
    if getattr(light, 'cookie', None) is not None:
        # R219: every lamp kind projects -- SPOT through its cone,
        # POINT wrapped around itself, AREA off its face
        rad = rad * cookie_factor(light, P, L)
    return L.astype(np.float32), rad.astype(np.float32), dist


#: R225: a DISK/ELLIPSE area lamp's contour is a regular polygon with
#: this many corners, scaled to the ellipse's exact AREA (see
#: area_corners). 16 keeps the polygon within 1% of the disc's radius
#: everywhere (max radial deviation 1 - cos(pi/16) = 1.9% before the
#: equal-area scale, +-1% after) at four times the rectangle's cost;
#: the GPU twin bakes the same corners as literals.
AREA_DISC_SIDES = 16

#: the shortest side an area lamp LIGHTS with (world units). The Stokes
#: contour sums the arc angles between unit vectors to the lamp's
#: corners and multiplies by dist^2 / A; on a face much smaller than
#: this the arcs are rounding noise and A is nearly zero -- a Size 0
#: lamp (Blender lets it be typed) lit the scene with noise seventy-six
#: times too bright, and the float32 GLSL twin broke fifty times sooner
#: (3% off the CPU at 0.005, 84% at 0.001). The far-field energy does
#: not depend on the size at all (BI's area lamp is total-power-like),
#: so a face clamped here IS the small lamp, drawn cleanly: at 0.02 the
#: GPU twin is within 0.3% of the CPU. One constant for the three
#: places that build the face: area_emit_area, area_corners and the
#: GLSL rectangle (gpu/material._area_function)
AREA_MIN_SIDE = 0.02


def area_is_disc(light):
    """True for the round area shapes (Blender's DISK and ELLIPSE)."""
    return str(getattr(light, 'area_shape', 'SQUARE')).upper() \
        in ('DISK', 'ELLIPSE')


def area_emit_area(light):
    """The lamp's EMITTING area: sx*sy for the rectangles, pi/4*sx*sy
    for the ellipses. The form factor's dist^2/A normalisation divides
    by this, so a disc and a square of the same size throw the same
    far-field energy (BI's area lamp is total-power-like: its energy
    does not grow with its size) and differ only in SHAPE."""
    sx, sy = area_lit_sides(light)
    if area_is_disc(light):
        return sx * sy * (np.pi * 0.25)
    return sx * sy


def area_lit_sides(light):
    """(sx, sy): the two sides the lamp's LIGHTING is computed with --
    its Size, never under AREA_MIN_SIDE. The traced shadow still reads
    the true Size (a Size 0 lamp casts a hard shadow)."""
    asz = getattr(light, 'area_size', (1.0, 1.0))
    sx = max(float(asz[0]), AREA_MIN_SIDE)
    sy = max(float(asz[1]) if len(asz) > 1 else float(asz[0]),
             AREA_MIN_SIDE)
    return sx, sy


def area_corners(light, dtype=np.float64):
    """The lamp face's corners (K, 3) in BI's contour winding.

    A SQUARE/RECTANGLE lamp is area_lamp_vectors' four corners in its
    order (-x-y, -x+y, +x+y, +x-y) -- the exact arithmetic the
    rectangle road always used. R225: a DISK/ELLIPSE lamp is a regular
    AREA_DISC_SIDES-gon in the same winding (angles falling from 225
    degrees), stretched by the two half-sizes and scaled by
    sqrt(2 pi / (K sin(2 pi / K))) so the polygon's area equals the
    ellipse's -- the Stokes contour of that polygon IS the disc's
    energy to within the polygon's radial error, and the far field
    matches the true disc exactly. "The Area light doesn't actually
    adapt to the shape of the light" -- now the illumination does,
    the way the soft shadows and beam cones already did.
    """
    pos = np.asarray(light.position, dtype)
    ax = np.asarray(getattr(light, 'area_x', None)
                    if getattr(light, 'area_x', None) is not None
                    else (1.0, 0.0, 0.0), dtype)
    ay = np.asarray(getattr(light, 'area_y', None)
                    if getattr(light, 'area_y', None) is not None
                    else (0.0, 1.0, 0.0), dtype)
    sx, sy = area_lit_sides(light)
    hx, hy = ax * dtype(sx * 0.5), ay * dtype(sy * 0.5)
    if not area_is_disc(light):
        return np.stack((pos - hx - hy, pos - hx + hy,
                         pos + hx + hy, pos + hx - hy))
    k = AREA_DISC_SIDES
    scale = np.sqrt(2.0 * np.pi / (k * np.sin(2.0 * np.pi / k)))
    th = 1.25 * np.pi - np.arange(k, dtype=np.float64) * (2.0 * np.pi / k)
    cs = (np.cos(th) * scale).astype(dtype)
    sn = (np.sin(th) * scale).astype(dtype)
    return pos[None, :] + hx[None, :] * cs[:, None] + hy[None, :] * sn[:, None]


def area_inp(light, P, N, want_back=False):
    """BI's area lamp energy -- shadeoutput.c area_lamp_energy, verbatim.

    The Stokes contour integral over the lamp face's corners, in
    DOUBLE precision exactly as the C (the acos of near-parallel unit
    vectors is why BI used doubles here): for each edge, the arc angle
    times the normal's projection on the edge plane's normal. The sum
    replaces N.L as the diffuse shader input `inp`, and the specular
    term is multiplied by it -- shade_one_light's LA_AREA branches.

    Scaled and shaped as area_lamp_energy_multisample at one sample
    (the 2.79 default, and the exact integral -- the jittered tiles
    only Monte-Carlo the same quantity):

        inp = pow(stokes * areasize, k),  areasize = dist^2 / A

    with `dist` the lamp's Distance (decay_end), A the face's emitting
    area (area_emit_area) and `k` the 2.79 area lamp's Gamma. A point
    behind the lamp's plane gets 0 (single sided). The rectangle
    shapes run BI's four corners bitwise; R225: DISK/ELLIPSE run the
    same contour over the equal-area polygon of area_corners, so a
    round lamp lights like a round lamp (the era formula only knew
    corners, so it gets more of them). lamp_get_visibility gives an
    area lamp visifac 1.0 (its falloff switch is skipped): the ONLY
    distance behaviour an area lamp has is this form factor's.

    Returns (n,) float32; with `want_back`, (front, back) where back
    is the same energy through the flipped normal -- BI's translucency
    reruns the whole lamp loop with vn negated, and the Stokes sum is
    linear in vn so both sides fall out of one contour.
    """
    n = P.shape[0]
    pos = np.asarray(light.position, np.float64)
    d = np.asarray(light.direction, np.float64)
    d = d / max(float(np.linalg.norm(d)), 1e-12)          # lar->vec
    corners = area_corners(light)
    P64 = P.astype(np.float64)
    vec = []
    for c in corners:
        v = P64 - c[None, :]                              # VECSUB(v, co, area)
        v /= np.maximum(np.sqrt((v * v).sum(axis=1)), 1e-300)[:, None]
        vec.append(v)
    # BI shades in a flipped frame: shade_input_set_normals negates
    # normals to lie ALONG the view ray (dot(facenor, view) >= 0 with
    # view camera->point), so shade_one_light's vn points AWAY from
    # the viewer and lv runs lamp->surface. Symmetric dots cancel the
    # flips, but the Stokes contour sees vn bare: entering with
    # Halcyon's outward N and BI's corner winding lands exactly one
    # sign off. vn = -N is the C's own frame (proven by compiling
    # area_lamp_energy verbatim: vn toward the lamp gives 0, away
    # gives the positive contour).
    vn = -N.astype(np.float64)
    fac = np.zeros(n, np.float64)
    nc = len(vec)
    for i in range(nc):
        a, b = vec[i], vec[(i + 1) % nc]
        cr = np.cross(a, b)
        cr /= np.maximum(np.sqrt((cr * cr).sum(axis=1)), 1e-300)[:, None]
        # saacos_d: acos clamped into its domain
        ang = np.arccos(np.clip((a * b).sum(axis=1), -1.0, 1.0))
        fac += ang * (vn * cr).sum(axis=1)
    dist = max(float(getattr(light, 'decay_end', 25.0)), 1e-6)
    areasize = dist * dist / area_emit_area(light)
    k = float(getattr(light, 'area_gamma', 1.0) or 1.0)
    # single sided: behind the lamp's plane nothing arrives
    front = ((P64 - pos[None, :]) * d[None, :]).sum(axis=1) >= 0.0
    inp = np.power(np.maximum(fac, 0.0) * areasize, k)
    inp = np.where(front, inp, 0.0).astype(np.float32)
    if not want_back:
        return inp
    binp = np.power(np.maximum(-fac, 0.0) * areasize, k)
    binp = np.where(front, binp, 0.0).astype(np.float32)
    return inp, binp


def area_samples(light, count, rng):
    """Points on an area light's surface, for soft ray-traced shadows."""
    pos = np.asarray(light.position, np.float32)
    if count <= 1:
        return [pos]
    ax = np.asarray(light.area_x, np.float32) * (light.area_size[0] * 0.5)
    ay = np.asarray(light.area_y, np.float32) * (light.area_size[1] * 0.5)
    out = []
    for i in range(count):
        u = rng.random() * 2.0 - 1.0
        v = rng.random() * 2.0 - 1.0
        if light.area_shape in ('DISK', 'ELLIPSE'):
            r = np.sqrt(rng.random())
            th = rng.random() * 2 * np.pi
            u, v = r * np.cos(th), r * np.sin(th)
        out.append(pos + ax * u + ay * v)
    return out


# ---------------------------------------------- soft traced shadows
#
# The N rays one pixel fires at a sized lamp are ONE stratified pattern.
# They used to be N independent hash draws -- white noise, whose error
# is sqrt(p(1 - p) / N): at the default 4 rays a penumbra showed five
# grey levels scattered with an RMS of 0.20. Now a pixel draws TWO
# hashes, a radial shift h1 and a turn h2, and ray k of N takes
#
#     u1 = (k + h1) / N               N equal-area strata, one shared shift
#     u2 = frac(turn_k + h2)          the pattern's own turn, rotated by h2
#
# with turn_k the base-2 radical inverse of k when N is a power of two
# (the N points are then a (0,2)-net: every dyadic box of area 1/N holds
# exactly one) and k times the golden ratio's fraction otherwise (evenly
# spread whatever N is). Measured on half-plane occluders over every
# orientation, the RMS falls to 0.53 / 0.37 / 0.38 of the old law at
# 4 / 8 / 16 rays on a disc and 0.61 / 0.50 / 0.40 on a rectangle, for
# two hashes a pixel instead of two a ray. The disc shapes read the turn
# as an index into the shared 256-entry circle table (an integer add,
# identical on every device); the rectangle reads it as its second
# coordinate.

#: the fraction of the golden ratio: its multiples spread evenly over
#: [0, 1) for any count
SOFT_PHI = 0.61803398875
#: the salt of an averaging road's pass index (the 10000th prime): an
#: accumulation pass, a lens pass or a motion-blur slice traces its own
#: rays instead of the first pass's again
SOFT_PASS_SALT = 104729


def soft_ray_salt(light_index, seed, sample_pass=0):
    """The hash salt of one lamp's soft-ray pattern, wrapped to the 31
    bits both roads hash (a large Seed must stay a legal GLSL int)."""
    from . import patterns as PT
    return PT.wrap_salt(131 * int(light_index) + 7919 * int(seed)
                        + SOFT_PASS_SALT * int(sample_pass))


def soft_ray_turns(samples):
    """The N rays' turns, as fractions of a full turn in [0, 1): the
    base-2 radical inverse of k for a power-of-two count, k golden
    fractions otherwise. Both roads bake THESE numbers."""
    n = max(int(samples), 1)
    if n & (n - 1) == 0:
        out = []
        for k in range(n):
            r, f, q = 0.0, 0.5, k
            while q:
                r += f * (q & 1)
                q >>= 1
                f *= 0.5
            out.append(r)
        return out
    return [(k * SOFT_PHI) % 1.0 for k in range(n)]


def soft_ray_turn_index(turn):
    """A turn as a step through the 256-entry circle table."""
    return int(round(256.0 * float(turn))) & 255


#: a sun's angular diameter never reaches the full half-space: EEVEE's
#: own ceiling (179.9 degrees). At pi the tangent of the half-angle is
#: 1.6e16 and every ray left perpendicular to the sun
SUN_MAX_ANGLE = float(np.radians(179.9))
#: above this angular diameter (radians) a sun's rays are drawn uniform
#: in the SOLID ANGLE of its cap. Below it the classic law stands -- a
#: uniform point in the tangent-plane disc of radius tan(angle / 2) --
#: which is the same thing to within 1% up to here and crowds the rim
#: beyond (at 2 rad only 12% of the rays fell inside half the radius,
#: where a quarter belong)
SUN_CAP_ANGLE = 0.5


def sun_soft_law(angle):
    """('DISC', tan(a / 2)) or ('CAP', 1 - cos(a / 2)) for a sun of
    angular diameter `angle`, the constant as the float32 both roads
    bake."""
    a = min(max(float(angle), 0.0), SUN_MAX_ANGLE)
    if a > SUN_CAP_ANGLE:
        return 'CAP', float(np.float32(1.0 - np.cos(a * 0.5)))
    return 'DISC', float(np.float32(np.tan(a * 0.5)))


def casts_shadow(light, settings):
    """Whether `visibility` could return anything but ones for this
    lamp: its early-outs, as a predicate (R238: the fixed key folds
    the casters' visibility in and skips the rest)."""
    if str(getattr(light, 'type', '')).upper() == 'HEMI':
        return False
    if not settings.shadows or getattr(light, 'shadow', 'NONE') == 'NONE':
        return False
    mode = light.shadow if settings.shadow_default == 'PER_LIGHT' else \
        settings.shadow_default
    # R251 C052: a PLANAR lamp's lighting term is untouched by its own
    # polygon (no self-shadow, the era's rule): it is no caster here
    return mode not in ('NONE', 'PLANAR')


def shadow_sample_count(light, settings):
    """Taps (mapped) or rays (traced) for ONE light: its own Samples
    when set, else the render's. 0 on the light means inherit -- the
    same rule its Map Size and Bias overrides follow -- so a lamp left
    alone reads exactly the render setting."""
    own = int(getattr(light, 'shadow_samples', 0) or 0)
    return own if own > 0 else max(1, int(getattr(settings, 'shadow_samples', 1)))


def visibility(light, P, N, L, dist, settings, bvh=None, rng=None,
               sample_xy=None, mask=None):
    """Shadow term in 0..1 (1 = fully lit).

    `sample_xy` is (spx, spy, light_index): the integer pixel identity and
    light stream for DETERMINISTIC soft-shadow sampling. With it, every
    jittered ray is a pure function of (pixel, sample, light, seed) --
    the same picture whatever the batch order, thread count or device.
    Without it (contexts that have no pixel identity), the legacy
    sequential stream still runs.

    `mask`, when given, marks the samples whose shadow term is actually
    READ by the caller (a sample whose diffuse and specular are both
    zero multiplies vis into nothing); traced rays are skipped outside
    it and those samples report fully lit. The pictures are identical
    -- the mask must only ever exclude samples whose contribution is
    zero -- and the caller owns that proof.
    """
    n = P.shape[0]
    if light.type == 'HEMI':
        # BI never shadowed hemi lamps ("hemi doesn't support shadows")
        return np.ones(n, np.float32)
    if not settings.shadows or light.shadow == 'NONE':
        return np.ones(n, np.float32)
    mode = light.shadow if settings.shadow_default == 'PER_LIGHT' else \
        settings.shadow_default
    # R251 C052: PLANAR returns BEFORE the ray fallback below -- a map-less
    # lamp with a BVH present would otherwise TRACE on top of its polygon
    # (the polygon is drawn by core/shadowmask.py over the finished frame)
    if mode in ('NONE', 'PLANAR'):
        return np.ones(n, np.float32)

    # R254: a map nobody was known to read was not baked with the
    # frame's (`map_roles`); the first lookup bakes it, here
    if light.shadow_map is None and mode == 'MAP' \
            and light.__dict__.get('_shadow_defer') is not None:
        _deferred_map(light)

    # ray_shadows is the master switch for TRACED shadows: RAY-mode lights
    # obey it too. It used to gate only the no-map fallback -- a state no
    # ordinary scene can reach, since every MAP-mode light gets a map built
    # -- which left the toggle effectively unreachable from the UI (found
    # by the settings audit)
    if settings.ray_shadows and (
            mode == 'RAY' or (light.shadow_map is None and bvh is not None)):
        if bvh is None:
            return np.ones(n, np.float32)
        bias = max(settings.ray_bias, 1e-4)
        origin = P + N * bias + L * bias
        maxt = np.where(dist > 1e8, 1e9, dist * (1.0 - 1e-3))
        # R208: the per-triangle caster mask (object Visibility >
        # Shadow, BI Shadow > Cast, Material.cast_shadow) -- the same
        # mask the shadow-map bake honours, as a BVH cast filter
        cast = getattr(settings, '_shadow_cast_tri', None)
        kind = str(getattr(light, 'type', 'POINT')).upper()
        # what "size" means per lamp: SUN's is an ANGLE, AREA's is its
        # rectangle, POINT/SPOT a world radius. The old code read
        # light.radius for all of them -- for a SUN that jittered a
        # 1e9-unit-long direction by a few hundredths of a unit (an
        # angle of ~3e-11: nothing), and for an AREA it read Blender's
        # shadow_soft_size, which area lamps do not carry (0: nothing).
        # This is why "changing light sizes doesn't actually work".
        area_hx = area_hy = 0.0
        if kind == 'AREA':
            asz = getattr(light, 'area_size', (0.0, 0.0))
            area_hx, area_hy = float(asz[0]) * 0.5, float(asz[1]) * 0.5
            soft_on = area_hx > 0.0 or area_hy > 0.0
        else:
            soft_on = float(light.radius) > 0.0
        samples = shadow_sample_count(light, settings) if soft_on else 1
        # Density lightens a traced shadow exactly as it lightens a
        # mapped one (the map tail's own expression). At the default
        # 1.0 the visibility is returned untouched: 1 - (1 - v) is not
        # bitwise v in float32, and a lamp left alone must not move
        dens = float(getattr(light, 'shadow_density', 1.0))

        def _dense(v):
            if dens == 1.0:
                return v
            return (1.0 - (1.0 - v) * dens).astype(np.float32)

        if samples == 1:
            hit = bvh.occluded(origin, L, maxt, mask=mask, cast=cast)
            return _dense((~hit).astype(np.float32))
        acc = np.zeros(n, np.float32)
        t, b = M.orthonormal_basis(L)
        disk = True
        if kind == 'AREA':
            ax = np.asarray(light.area_x, np.float32)
            ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
            ay = np.asarray(light.area_y, np.float32)
            ay = ay / max(float(np.linalg.norm(ay)), 1e-9)
            lpos = np.asarray(light.position, np.float32)
            disk = str(getattr(light, 'area_shape', 'SQUARE')) \
                in ('DISK', 'ELLIPSE')
        elif kind == 'SUN':
            sun_law, sun_k = sun_soft_law(light.radius)
            sun_k = np.float32(sun_k)

        def _soft_ray(u1, ca_, sa_, u2):
            """(Lj, mt): one soft-shadow ray per pixel. `u1` is the
            area coordinate, (ca_, sa_) the unit-circle point of the
            turn and `u2` the turn itself as a fraction (the rectangle's
            second coordinate; the round shapes never read it).

            AREA aims at a real point on the rectangle (or ellipse) --
            penumbras follow the lamp's true shape and size; SUN tilts
            the direction inside its angular disc; POINT/SPOT jitter the
            target inside a world-space sphere of the radius, exactly
            the classic behaviour.
            """
            if kind == 'AREA':
                if disk:
                    rr = np.sqrt(u1)
                    uu, vv = rr * ca_, rr * sa_
                else:
                    uu, vv = u1 * 2.0 - 1.0, u2 * 2.0 - 1.0
                ps = lpos[None, :] + ax[None, :] * (area_hx * uu)[:, None] \
                    + ay[None, :] * (area_hy * vv)[:, None]
                delta = ps - P
                ds = np.sqrt(np.maximum((delta * delta).sum(axis=1), EPS))
                return delta / ds[:, None], ds * (1.0 - 1e-3)
            if kind == 'SUN':
                if sun_law == 'CAP':
                    # a wide sun: uniform in the solid angle of its cap
                    # (cos of the tilt is uniform between 1 and cos of
                    # the half-angle). L, t and b are orthonormal, so
                    # the sum is the unit ray itself
                    ct = np.float32(1.0) - u1 * sun_k
                    sn = np.sqrt(np.maximum(np.float32(1.0) - ct * ct,
                                            np.float32(0.0)))
                    return (L * ct[:, None] + t * (sn * ca_)[:, None]
                            + b * (sn * sa_)[:, None]), maxt
                # the size is Blender's light.angle: tilt the ray inside
                # that cone, which is what a sun's size IS
                r = np.sqrt(u1) * sun_k
                jitter = t * (r * ca_)[:, None] + b * (r * sa_)[:, None]
                return M.normalize(L + jitter), maxt
            r = np.sqrt(u1) * np.float32(light.radius)
            jitter = t * (r * ca_)[:, None] + b * (r * sa_)[:, None]
            return M.normalize(L * dist[:, None] + jitter), maxt

        if sample_xy is not None:
            # the stratified pattern (see soft_ray_turns): two hashes
            # for the pixel, then every ray is arithmetic on them
            from . import patterns as PT
            spx, spy, li = sample_xy
            z0 = soft_ray_salt(li, getattr(settings, 'seed', 0) or 0,
                               getattr(settings, '_sample_pass', 0) or 0)
            h1 = PT.sample_u(spx, spy, z0)
            h2 = PT.sample_u(spx, spy, z0 + 1)
            # the hash's own 16-bit integer (h2 * 65535 is exact in
            # float32), as sample_circle recovers it
            h2i = np.asarray(h2 * np.float32(65535.0) + np.float32(0.5),
                             np.float32).astype(np.int64)
            invn = np.float32(1.0 / samples)
            turns = soft_ray_turns(samples)
            for k in range(samples):
                u1 = (np.float32(k) + h1) * invn
                ai = (h2i + soft_ray_turn_index(turns[k])) & 255
                u2 = None
                if not disk:
                    u2 = np.float32(turns[k]) + h2
                    u2 = u2 - np.floor(u2)
                Lj, mt = _soft_ray(u1, PT.CIRCLE256[ai, 0],
                                   PT.CIRCLE256[ai, 1], u2)
                acc += (~bvh.occluded(origin, Lj, mt, mask=mask,
                                      cast=cast)).astype(np.float32)
            return _dense(acc / samples)
        rng = rng or np.random.default_rng(settings.seed)
        for _ in range(samples):
            u1 = np.full(n, rng.random(), np.float32)
            u2 = np.full(n, rng.random(), np.float32)
            Lj, mt = _soft_ray(u1, np.cos(u2 * 2.0 * np.pi),
                               np.sin(u2 * 2.0 * np.pi), u2)
            acc += (~bvh.occluded(origin, Lj, mt, mask=mask,
                                  cast=cast)).astype(np.float32)
        return _dense(acc / samples)

    sm = light.shadow_map
    if sm is None:
        return np.ones(n, np.float32)
    # R251 C117: under a MIDPOINT map the compare lands inside the
    # caster's thickness, so the bias and the normal offset are ZERO --
    # `P + N * 0.0 == P` bitwise and slope 0.0: the same lookup, two
    # zeroed inputs (Woo's rule, Maya's Use Mid Dist)
    mid = midpoint_map(light, settings)
    bias = 0.0 if mid else float(light.shadow_bias or settings.shadow_bias)
    ndl = np.clip(M.dot(N, L), 0.0, 1.0)
    slope = bias * (1.0 + 2.0 * (1.0 - ndl))
    # the user's Softness: a constant blur in texels (the lamp's own
    # dial multiplies the render's)
    soft = max(float(light.shadow_softness) * settings.shadow_softness, 0.0)
    # normal offset: step off the surface by a texel or so, scaled by how
    # obliquely the light hits. Removes acne without the detached shadows a
    # large constant depth bias produces.
    texel = sm.texel_size(np.linalg.norm(P - sm.origin[None, :], axis=1))
    shape = SM.lookup_shape(light, shadow_sample_count(light, settings),
                            soft)
    if shape is not None:
        # a SIZED lamp (Radius, a Sun's Angle, an Area's Size) at more
        # than one Sample -- one Sample is the hard single-tap lookup
        # below for every lamp, as one traced ray is hard. The penumbra
        # is measured per receiver: a blocker search over the lamp's
        # footprint, a width by similar triangles, a filter that wide
        # (core/softmap.py). Every compare there is made against the
        # receiver PLANE's depth at the tap, so the offset carries no
        # blur factor and is scaled DOWN where that plane is exact
        # (`offset_scale`): nothing lifts the receiver out from under
        # its caster as the lamp or the Softness grows. A lamp of size
        # 0 whose Softness is wider than one texel takes the same road
        # with no lamp term (`SM.zero_shape`): the dial blurs the edge
        # and no longer shrinks the shadow
        if mid:
            offset = np.zeros(n, np.float32)
        else:
            offset = texel * (1.5 + 2.5 * np.sqrt(
                np.maximum(1.0 - ndl * ndl, 0.0))) * SM.offset_scale(ndl)
        lit = SM.lookup(sm, shape, P + N * offset[:, None], N, ndl, slope,
                        soft, shadow_sample_count(light, settings),
                        mask=mask)
        dens = float(light.shadow_density)
        return (1.0 - (1.0 - lit) * dens).astype(np.float32)
    if mid:
        offset = np.zeros(n, np.float32)
    else:
        offset = texel * (1.5 + 2.5 * np.sqrt(np.maximum(1.0 - ndl * ndl, 0.0))) * \
            max(1.0, soft)
    lit = sm.lookup(P + N * offset[:, None], slope, soft,
                    shadow_sample_count(light, settings))
    dens = float(light.shadow_density)
    return (1.0 - (1.0 - lit) * dens).astype(np.float32)


def limit_reference(scene):
    """What the Light Limit measures against (R254): (pts, guard, eye).

    `pts` is the 3 x 3 x 3 grid on the bounds of the scene's geometry
    (27 points, float32; None without vertices) and `guard` 5% of the
    bounds' diagonal -- Brightest averages what a lamp delivers over
    them, so it does not depend on the camera and an orbit never changes
    the set. `eye` is the camera's position, which Nearest measures from
    (the middle of the bounds for a scene without a camera; None with
    neither)."""
    pts, guard, eye = None, 0.0, None
    verts = getattr(getattr(scene, 'mesh', None), 'verts', None)
    if verts is not None and len(verts):
        v = np.asarray(verts, np.float32)
        lo = v.min(axis=0).astype(np.float64)
        hi = v.max(axis=0).astype(np.float64)
        t = np.array([0.0, 0.5, 1.0])
        g = np.stack(np.meshgrid(t, t, t, indexing='ij'), -1).reshape(-1, 3)
        pts = (lo[None, :] + g * (hi - lo)[None, :]).astype(np.float32)
        guard = max(0.05 * float(np.linalg.norm(hi - lo)), 1e-3)
        eye = (lo + hi) * 0.5
    mw = getattr(getattr(scene, 'camera', None), 'matrix_world', None)
    if mw is not None:
        eye = np.asarray(mw, np.float64)[:3, 3]
    return pts, guard, eye


def limit_strength(light, pts, guard, settings):
    """What a lamp delivers to the scene, in the light loop's own units:
    the number the Light Limit's Brightest ranks on (R254).

    A Sun's (or hemi's) strength IS the radiance every surface receives.
    A point or spot lamp's Power is divided by 4 pi and by its falloff
    in sample(), so it is ranked on exactly that: |Power| / (4 pi) times
    the lamp's own law, averaged over the grid on the scene's bounds
    (limit_reference). No distance is taken below `guard`, so a lamp
    that sits on a grid point is not ranked at the pole of its own
    curve. An area lamp reads no falloff: its strength is its form
    factor (area_inp, Distance and Gamma included) on a surface facing
    it, averaged the same way -- zero behind its face. A spot ranks as
    the point lamp it is inside its cone: the cone is not read (weighted
    at 27 points, a narrow key spot would fall between them).

    The RAW Power, never the style gain: a strobing lamp must not change
    the selection (and with it the GPU plan) from frame to frame."""
    k = abs(float(light.energy)) * float(np.mean(light.color))
    if light.type in ('SUN', 'HEMI'):
        return k
    delta = np.asarray(light.position, np.float64)[None, :] \
        - np.asarray(pts, np.float64)
    d = np.sqrt((delta * delta).sum(axis=1))
    if light.type == 'AREA':
        n = (delta / np.maximum(d, 1e-9)[:, None]).astype(np.float32)
        w = float(np.mean(area_inp(light, np.asarray(pts, np.float32), n)))
    else:
        w = float(np.mean(attenuate(
            light, np.maximum(d, guard).astype(np.float32), settings)))
    return k * w / (4.0 * np.pi)


def is_ambient_lamp(light):
    """True for a lamp that is ONLY ambient: the AMBIENT type, and any
    lamp with Ambient Only ticked (R254).

    Such a lamp feeds the ambient pool (ambient_light_split) and
    nothing else: it has no direction to shade from, so it is no
    member of the frame's lamp list -- no diffuse or specular lobe, no
    shadow, no beam, shaft or flare, no slot under the Light Limit.
    Ambient Only used to be read by the pool alone: the lamp kept
    lighting and shadowing as before with the ambient added on top.
    This is the one predicate every reader of the lamp list asks."""
    return str(getattr(light, 'type', '')).upper() == 'AMBIENT' or \
        bool(getattr(light, 'ambient_only', False))


def lights_volume(light):
    """The sign a lamp's scatter takes in a medium (a volume container,
    its own visible beam): +1.0, -1.0 for a Negative lamp, or 0.0 for a
    lamp that lights no medium at all -- an ambient lamp (no position)
    and an Only Shadow lamp (it adds no light anywhere, it only takes
    it away where its shadow falls on a surface) (R254).

    The media used to read none of the contribution flags: a Negative
    lamp brightened the fog it should darken, an Only Shadow lamp lit
    it. +1.0 for every lamp without those flags, so a scene that does
    not use them marches exactly as before."""
    if is_ambient_lamp(light) or getattr(light, 'only_shadow', False):
        return 0.0
    return -1.0 if getattr(light, 'negative', False) else 1.0


#: R254: how the Light Limit chooses. `reference` is limit_reference's
#: (pts, guard, eye), or a callable returning it that is read only when
#: the limit actually drops a lamp (frame_lights hands in the scene's):
#: Brightest ranks on what each lamp delivers over the scene
#: (limit_strength), Nearest on its distance from the camera with a Sun or
#: hemi at distance 0 (it reaches everything), ties in scene order, and
#: the survivors come back in SCENE order -- the order the per-material
#: light limit and the GPU plan's selection signature both assume.
#: Without a reference (a bare lamp list, a scene with nothing to measure
#: against) the ranking is the raw one it always was. A list at or under
#: the limit comes back untouched.
def select_lights(lights, settings, reference=None):
    """Emulate a hardware light limit by keeping only the strongest N."""
    active = [l for l in lights if not is_ambient_lamp(l)]
    limit = int(settings.max_lights)
    if limit <= 0 or len(active) <= limit:
        return active
    # R254: the world's own light (core/worldlight.py) is the sky's sun --
    # the limit ranks the LAMPS for the slots left beside it and never
    # drops it. No scene without World Light carries the mark.
    kept = [l for l in active if getattr(l, 'world_light', False)]
    if kept:
        active = [l for l in active if not getattr(l, 'world_light', False)]
        limit = max(limit - len(kept), 0)
        if limit == 0:
            return kept
    if settings.light_limit_mode == 'FIRST':
        return active[:limit] + kept
    if callable(reference):
        reference = reference()
    pts, guard, eye = reference if reference is not None \
        else (None, 0.0, None)
    nearest = settings.light_limit_mode == 'NEAREST'
    if (eye is None) if nearest else (pts is None):
        key = [float(l.energy) * float(np.mean(l.color)) for l in active]
        order = np.argsort(key)[::-1]
        return [active[i] for i in order[:limit]] + kept
    if nearest:
        e = np.asarray(eye, np.float64)
        key = [0.0 if l.type in ('SUN', 'HEMI') else
               float(np.linalg.norm(np.asarray(l.position, np.float64) - e))
               for l in active]
        order = np.argsort(np.asarray(key, np.float64), kind='stable')
    else:
        key = [limit_strength(l, pts, guard, settings) for l in active]
        order = np.argsort(-np.asarray(key, np.float64), kind='stable')
    # the world's light is the last lamp of the scene (core/worldlight
    # appends it), so the survivors stay in scene order
    return [active[i] for i in sorted(int(i) for i in order[:limit])] + kept


def frame_lights(scene, settings):
    """The lamps this frame shades under the Light Limit: the ONE
    selection the CPU light loop, the GPU plan and its texel repack,
    the screen-spot fog, the ground plane, the light meter node and the
    planar shadows all read (R254).

    They used to call select_lights with the bare lamp list, so no
    caller could say where the scene or the camera was: Nearest never
    measured a distance and Brightest compared a Sun's strength with a
    point lamp's watts. Nothing is parked on the scene or the settings:
    the reference is measured inside the call, and only when the limit
    actually drops a lamp (a scene at or under it takes the old path
    and never looks at a vertex)."""
    return select_lights(getattr(scene, 'lights', None) or [], settings,
                         reference=lambda: limit_reference(scene))


class _NoLimit:
    """Settings that keep every lamp: frame_lights under it is the list
    the limit chooses among (limit_report's denominator)."""
    max_lights = 0
    light_limit_mode = 'FIRST'


#: R254: the last F12's limit report, for the Lighting panel to draw:
#: {'unlit': [names], 'total': n, 'max_lights': n, 'mode': str}, empty
#: while every lamp was lit. A report only -- no render reads it.
LAST_LIMIT_REPORT = {}


def limit_report(scene, settings, remember=False):
    """One line naming the lamps the Light Limit leaves unlit this frame,
    or None when every lamp is lit. The limit used to drop lamps in
    silence -- at the factory 8 a ninth lamp simply did nothing.
    `remember` keeps the answer in LAST_LIMIT_REPORT (the F12 does)."""
    lights = frame_lights(scene, _NoLimit)
    kept = {id(l) for l in frame_lights(scene, settings)}
    out = [str(getattr(l, 'name', '') or l.type) for l in lights
           if id(l) not in kept]
    if remember:
        LAST_LIMIT_REPORT.clear()
        if out:
            LAST_LIMIT_REPORT.update(
                unlit=list(out), total=len(lights),
                max_lights=int(settings.max_lights),
                mode=str(settings.light_limit_mode))
    if not out:
        return None
    names = ', '.join(out[:6])
    if len(out) > 6:
        names += f' and {len(out) - 6} more'
    mode = str(settings.light_limit_mode).capitalize()
    return (f'Light Limit {int(settings.max_lights)} ({mode}): '
            f'{len(out)} of {len(lights)} lamps are not lit ({names}). '
            'Raise Light Limit in Render Properties > Lighting, or set it '
            'to 0 for no limit')


#: R254: the flat ambient level above which an ambient lamp is called
#: out (the panel's red row, the render's report): past 2.0 every
#: surface is over twice its own colour
AMBIENT_BRIGHT = 2.0


def ambient_level(light):
    """The flat level an ambient lamp adds to the pool, as one number:
    its Energy times its strongest colour channel (R254)."""
    try:
        col = max(float(c) for c in getattr(light, 'color', (1.0, 1.0, 1.0)))
    except (TypeError, ValueError):
        col = 1.0
    return abs(float(getattr(light, 'energy', 0.0))) * col


def ambient_report(scene):
    """One line naming the ambient lamps (is_ambient_lamp) whose level
    whites the frame out, or None (R254).

    The pool reads an ambient lamp's Energy as a FLAT level -- colour x
    Energy, 1.0 = white ambient -- whatever the lamp's kind, where every
    other use of a Point lamp's Energy divides its watts by 4 pi d^2.
    Ambient Only ticked on a lamp at a working wattage therefore gives
    hundreds of times white, and used to say nothing."""
    out = [(str(getattr(l, 'name', '') or l.type), ambient_level(l))
           for l in (getattr(scene, 'lights', None) or ())
           if is_ambient_lamp(l) and ambient_level(l) > AMBIENT_BRIGHT]
    if not out:
        return None
    names = ', '.join(f"'{n}' at level {v:.4g}" for n, v in out[:4])
    if len(out) > 4:
        names += f' and {len(out) - 4} more'
    return (f'Ambient Only: {names}. An ambient lamp\'s Energy is a flat '
            'level (1.0 = white ambient), not watts, so the frame whites '
            'out. Lower it in the Light panel, under Ambient Only')


def ambient_light_split(scene, settings):
    """(engine_part, world_part) of the ambient pool.

    The split matters to the BI material node: Blender Internal's
    ambient rule -- a FLAT add, untinted by the diffuse colour -- is
    about the WORLD's ambient (and ambient-type lights, which model
    the same thing). The engine's own Global Ambient setting has no
    BI counterpart: flat-adding it lifted every imported black by the
    default (0.05, 0.05, 0.06) -- the field's 'blueish tint to the
    dark parts' -- so it keeps the classic diffuse-tinted behaviour
    on every model."""
    eng = np.asarray(settings.global_ambient, np.float32) \
        * settings.global_ambient_level
    wrld = np.zeros(3, np.float32)
    if scene.world is not None:
        wrld = wrld + np.asarray(scene.world.ambient, np.float32) * \
            scene.world.ambient_level
    for l in scene.lights:
        if l.type == 'AMBIENT' or l.ambient_only:
            # R253 (H): an ambient-only lamp flickers with its style too
            wrld = wrld + np.asarray(l.color, np.float32) * (
                l.energy * float(getattr(l, 'style_gain', 1.0)))
    return eng.astype(np.float32), wrld.astype(np.float32)


def ambient_light(scene, settings):
    eng, wrld = ambient_light_split(scene, settings)
    return (eng + wrld).astype(np.float32)


# ------------------------------------------------- R251 LIGHT-B1: lamp laws
#
# F012 spot cone laws, F013 decay laws, F015 the Model 3 screen spotlight.
# Every function here is float32, one op per statement, and is the CPU
# half of a GLSL twin (gpu/material.py spot_law_glsl / decay_law_glsl /
# the screen-spot block) that runs the same statements in the same
# order, so the simulator reproduces it bit for bit. The coefficient
# helpers are the ONE place the numbers are computed: the GPU packer
# (pack_light_texels) uploads exactly what these return.

#: R251 F012: the cone laws a spot lamp may take (Light.spot_law).
#: BLENDER is the 1.89.0 behaviour (Blender Internal's smoothstep band
#: times the raw cosine); GL11 is glLight's cutoff + GL_SPOT_EXPONENT;
#: POV is lightsource.cpp's flat hotspot with a cubic_spline edge and
#: tightness; the GX_* six are libogc GX_InitLightSpot's angular
#: functions (FLAT, COS, COS2, SHARP, RING1, RING2).
SPOT_LAWS = ('BLENDER', 'GL11', 'POV', 'GX_FLAT', 'GX_COS', 'GX_COS2',
             'GX_SHARP', 'GX_RING1', 'GX_RING2')

#: R251 F013: the decay laws added to the shared FALLOFF enum.
DECAY_LAWS_R251 = ('GL_3TERM', 'POV_FADE_LINEAR', 'POV_FADE_SQUARE',
                   'GX_GENTLE', 'GX_MEDIUM', 'GX_STEEP')

#: R253 (D): the nominal-distance laws appended to FALLOFF -- (D/d)^n
#: with D = Falloff End: 3ds Max's Decay Inverse / Inverse Square with
#: Start R0 (Lum = (R0/R)^n, the 3ds Max 3 User Reference's Attenuation
#: Parameters), LightWave 6/7's Intensity Falloff Inverse Distance /
#: Inverse Distance^2 with its Range (nominal distance). Exactly full
#: strength at D, brighter inside, the plain law beyond.
DECAY_LAWS_R253 = ('NOMINAL_INVERSE', 'NOMINAL_INV_SQUARE')

_GX_DEGENERATE_SAID = set()


def spot_law_coeffs(light):
    """(cr, a0, a1, a2), four float32, for the lamp's cone law.

    `cr` is the cosine the law compares against: POV-Ray's radius,
    cos(min(hotspot, spot_size)/2) (the clamp keeps cr >= the cone's
    cutoff cosine on both roads, so a hotspot dragged past the cone
    gives the hard edge at the cone); for the GX functions the cutoff
    cosine itself. (a0, a1, a2) are libogc GX_InitLightSpot's
    coefficients for the GX laws, computed in float64 and rounded to
    float32 ONCE -- SHARP's a0 divided by (1 - cr)^2 as its other two
    terms are (the libogc transcription slip, disclosed in the
    CHANGELOG). Zeros for the other laws. The GPU packer uploads these
    very values (hal_lights texels 4.yzw and 5.x).
    """
    law = str(getattr(light, 'spot_law', 'BLENDER') or 'BLENDER')
    size = float(light.spot_size)
    cf = float(np.cos(size * 0.5))
    if law == 'POV':
        hot = float(getattr(light, 'hotspot', 0.0) or 0.0)
        cr = float(np.cos(min(hot, size) * 0.5))
    else:
        cr = cf
    a0 = a1 = a2 = 0.0
    if law.startswith('GX_'):
        c = cr
        if 1.0 - c < 1e-6:
            # a cone too narrow for the rational functions (cr -> 1
            # divides by zero): the law degenerates to FLAT, said once
            key = (getattr(light, 'name', ''), law)
            if key not in _GX_DEGENERATE_SAID:
                _GX_DEGENERATE_SAID.add(key)
                print(f"[Halcyon] spot '{key[0]}': Spot Size below 0.003 "
                      f"rad degenerates the {law} cone law to GX_FLAT")
            law = 'GX_FLAT'
            c = min(c, 1.0 - 1e-6)
        D = (1.0 - c) * (1.0 - c)
        if law == 'GX_FLAT':
            a0, a1, a2 = -1000.0 * c, 1000.0, 0.0
        elif law == 'GX_COS':
            a0, a1, a2 = -c / (1.0 - c), 1.0 / (1.0 - c), 0.0
        elif law == 'GX_COS2':
            a0, a1, a2 = 0.0, -c / (1.0 - c), 1.0 / (1.0 - c)
        elif law == 'GX_SHARP':
            a0, a1, a2 = c * (c - 2.0) / D, 2.0 / D, -1.0 / D
        elif law == 'GX_RING1':
            a0, a1, a2 = -4.0 * c / D, 4.0 * (1.0 + c) / D, -4.0 / D
        elif law == 'GX_RING2':
            a0, a1, a2 = 1.0 - 2.0 * c * c / D, 4.0 * c / D, -2.0 / D
    return (np.float32(cr), np.float32(a0), np.float32(a1), np.float32(a2))


def spot_law_factor(light, inpr):
    """The cone factor for cos(angle to the axis) `inpr`, per `spot_law`.

    BLENDER: the 1.89.0 body untouched. GL11: 0 outside the cutoff, the
    cosine to Cone Exponent inside (exponent 0: exactly 1). POV: flat 1
    inside the hotspot, the Hermite (3 - 2t) t^2 down to the cone edge,
    times cos^exponent; the Hermite rides the VALUES road (both forms
    computed, selected by `pd < 1e-6` -- dragging the hotspot onto the
    cone edge changes a texel, never the plan). GX_*: libogc's
    quadratic in the cosine, saturated to [0, 1] (GX's lighting terms
    saturate before the colour multiply; FLAT's 1000x slope IS a
    saturating step). Float32, one op per statement; the GLSL twin is
    gpu/material.py spot_law_glsl.
    """
    law = str(getattr(light, 'spot_law', 'BLENDER') or 'BLENDER')
    inpr = np.asarray(inpr, np.float32)
    if law == 'BLENDER' or law not in SPOT_LAWS:
        return _spot_blender(light, inpr)
    cf = np.float32(np.cos(float(light.spot_size) * 0.5))
    exp32 = np.float32(getattr(light, 'spot_exponent', 0.0) or 0.0)
    if law == 'GL11':
        if exp32 > 0.0:
            p = M.safe_pow(np.maximum(inpr, np.float32(0.0)), exp32)
            fac = np.where(inpr < cf, np.float32(0.0), p)
        else:
            fac = np.where(inpr < cf, np.float32(0.0), np.float32(1.0))
        return fac.astype(np.float32)
    cr, a0, a1, a2 = spot_law_coeffs(light)
    if law == 'POV':
        if exp32 > 0.0:
            att = M.safe_pow(np.maximum(inpr, np.float32(0.0)), exp32)
        else:
            att = np.ones_like(inpr)
        pd = np.float32(cr - cf)
        with np.errstate(divide='ignore', invalid='ignore'):
            t0 = inpr - cf
            t0 = t0 / pd
        t0 = np.clip(t0, np.float32(0.0), np.float32(1.0))
        t1 = np.where(inpr < cr, np.float32(0.0), np.float32(1.0))
        t = np.where(pd < 1e-6, t1, t0).astype(np.float32)
        t2 = t * t
        s = np.float32(3.0) - np.float32(2.0) * t
        s = s * t2
        att = np.where(inpr < cr, att * s, att)
        fac = np.where(inpr <= 0.0, np.float32(0.0), att)
        return fac.astype(np.float32)
    # GX_*
    c = inpr
    ang = a1 * c
    ang = a0 + ang
    cc = c * c
    cc = a2 * cc
    ang = ang + cc
    fac = np.clip(ang, np.float32(0.0), np.float32(1.0))
    return fac.astype(np.float32)


def gx_dist_coeffs(light, mode):
    """(k1, k2) float32 of libogc GX_InitLightDistAttn for a GX decay
    mode: ref_dist = Falloff End, ref_brite = the lamp's gx_ref_brite,
    computed in float64 and rounded once. The lamp is exactly ref_brite
    bright at ref_dist. Zeros for any other mode."""
    rb = float(getattr(light, 'gx_ref_brite', 0.5) or 0.5)
    rb = min(max(rb, 1e-3), 1.0 - 1e-3)
    rd = max(float(light.decay_end), EPS)
    k1 = k2 = 0.0
    if mode == 'GX_GENTLE':
        k1 = (1.0 - rb) / (rb * rd)
    elif mode == 'GX_MEDIUM':
        k1 = 0.5 * (1.0 - rb) / (rb * rd)
        k2 = 0.5 * (1.0 - rb) / (rb * rd * rd)
    elif mode == 'GX_STEEP':
        k2 = (1.0 - rb) / (rb * rd * rd)
    return np.float32(k1), np.float32(k2)


def decay_law(light, mode, dist):
    """R251 F013: the six decay laws on `dm = max(dist, 0)`, float32, one
    op per statement -- the CPU half of gpu/material.py decay_law_glsl.

    GL_3TERM: 1/(1 + ld1 d + ld2 d^2) (glLight's kc, kl, kq with kc = 1;
    both sliders 0 = GL's default, no falloff). POV_FADE_LINEAR /
    POV_FADE_SQUARE: 2/(1 + (d/D)^P), P = 1, 2 -- bounded at 2, exactly
    1 at D = Falloff End (POV's fade_distance-0 form and other powers
    are not modelled). GX_*: 1/(1 + k1 d + k2 d^2) with (k1, k2) from
    gx_dist_coeffs.
    """
    dm = np.maximum(np.asarray(dist, np.float32), np.float32(0.0))
    one = np.float32(1.0)
    if mode == 'GL_3TERM':
        ld1 = np.float32(getattr(light, 'decay_ld1', 0.0) or 0.0)
        ld2 = np.float32(getattr(light, 'decay_ld2', 0.0) or 0.0)
        t = ld1 * dm
        u = dm * dm
        u = ld2 * u
        s = one + t
        s = s + u
        att = one / s
    elif mode in ('POV_FADE_LINEAR', 'POV_FADE_SQUARE'):
        D = np.float32(max(float(light.decay_end), EPS))
        q = dm / D
        if mode == 'POV_FADE_SQUARE':
            q = q * q
        s = one + q
        att = np.float32(2.0) / s
    elif mode in DECAY_LAWS_R253:
        # R253 (D): (D/d)^n, D = Falloff End (the nominal distance),
        # d floored at 1e-6 so the lamp's own position is finite --
        # exactly 1.0 at d = D (float32 D/D), 0.5 / 0.25 at 2D
        D = np.float32(max(float(light.decay_end), EPS))
        dq = np.maximum(dm, np.float32(1e-6))
        att = D / dq
        if mode == 'NOMINAL_INV_SQUARE':
            att = att * att
    else:
        k1, k2 = gx_dist_coeffs(light, mode)
        t = k1 * dm
        u = dm * dm
        u = k2 * u
        s = one + t
        s = s + u
        att = one / s
    return att.astype(np.float32)


# ------------------------------------------------- R253 (H): light styles
#
# Quake (id Software, 1996), QuakeC world.qc: the twelve worldspawn
# lightstyle strings, 'a' = 0, 'm' = normal, 'z' = double, advanced ten
# steps per second by the client (cl_main.c CL_ParseLightstyle +
# r_light.c R_AnimateLight: `i = (int)(cl.time*10)`, `k = ...[i %
# length] - 'a'; k = k*22` -- Quake's own unit is 22/256 per letter,
# so its 'm' is 264/256 = 1.03; Halcyon normalises 'm' to EXACTLY 1.0
# so NORMAL is bitwise inert). The gain is a pure function of
# (style, rate, scene.time): deterministic per frame on both devices.

#: the twelve world.qc strings, in Quake's own order (0 = normal)
LIGHT_STYLES = {
    'NORMAL': 'm',
    'FLICKER_A': 'mmnmmommommnonmmonqnmmo',
    'SLOW_STRONG_PULSE': 'abcdefghijklmnopqrstuvwxyzyxwvutsrqponmlkjihgfedcba',
    'CANDLE_A': 'mmmmmaaaaammmmmaaaaaabcdefgabcdefg',
    'FAST_STROBE': 'mamamamamama',
    'GENTLE_PULSE': 'jklmnopqrstuvwxyzyxwvutsrqponmlkj',
    'FLICKER_B': 'nmonqnmomnmomomno',
    'CANDLE_B': 'mmmaaaabcdefgmmmmaaaammmaamm',
    'CANDLE_C': 'mmmaaammmaaammmabcdefaaaammmmabcdefmmmaaaa',
    'SLOW_STROBE': 'aaaaaaaazzzzzzzz',
    'FLUORESCENT_FLICKER': 'mmamammmmammamamaaamammma',
    'SLOW_PULSE_NO_BLACK': 'abcdefghijklmnopqrrqponmlkjihgfedcba',
}


def style_char_gain(ch):
    """One lightstyle letter's gain: (ord - 'a') / 12, so 'a' = 0, 'm' =
    exactly 1.0 and 'z' = 25/12 (Quake's own step is 22/256 per letter;
    the normalisation is Halcyon's, so the NORMAL style is inert)."""
    return (ord(ch) - 97) / 12.0


def style_gain(light, time):
    """R253 (H): the lamp's gain at `time` seconds -- the character of
    its style string at floor(time * rate), wrapping at the string's
    length (R_AnimateLight's `i % length`). NORMAL (and an unknown
    style) is 1.0 exactly."""
    key = str(getattr(light, 'light_style', 'NORMAL') or 'NORMAL')
    sstr = LIGHT_STYLES.get(key, 'm')
    if sstr == 'm':
        return 1.0
    rate = float(getattr(light, 'light_style_rate', 10.0) or 10.0)
    i = int(np.floor(float(time) * rate)) % len(sstr)
    return style_char_gain(sstr[i])


def style_step(light, time):
    """(step, steps) of the lamp's style string at `time` seconds: the
    0-based character style_gain reads there and the string's length --
    (0, 1) for NORMAL. For the panel's status line (R254); the same
    index expression as style_gain."""
    key = str(getattr(light, 'light_style', 'NORMAL') or 'NORMAL')
    sstr = LIGHT_STYLES.get(key, 'm')
    if sstr == 'm':
        return 0, 1
    rate = float(getattr(light, 'light_style_rate', 10.0) or 10.0)
    return int(np.floor(float(time) * rate)) % len(sstr), len(sstr)


def apply_light_styles(scene):
    """R253 (H): write every lamp's `style_gain` for THIS frame from
    scene.time -- called before the shadow maps build (render) and by
    ShadeJob, so the CPU loop, the beams, the fog lobe, the ambient
    pool and the GPU packer all read one number. Idempotent."""
    t = float(getattr(scene, 'time', 0.0) or 0.0)
    for l in (getattr(scene, 'lights', None) or ()):
        try:
            l.style_gain = float(style_gain(l, t))
        except Exception:                                       # noqa: BLE001
            pass


def screen_spot_params(light, vp, w, h, camera=None):
    """R251 F015: the Model 3 viewport spotlight's ellipse for one frame,
    six float32 -- (cx, cy, w, h, start, aext) in INTERNAL pixels.

    (cx, cy) is the lamp position projected by `vp` exactly as the
    rasteriser projects a vertex (`clip = (pos, 1) @ vp.T`, then
    `(ndc*0.5 + 0.5) * size`; row 0 = the bottom of the frame, pixel
    centres sit at +0.5); the half axes are half the Spot Size as a
    fraction of the frame HEIGHT (Halcyon's own mapping of the board's
    pixel size, stated in the tooltip); `start` is the camera's near
    clip (the board's 1/float depth start) and `aext` = |Falloff End|
    (the extent). The GPU packer uploads these very values (hal_lights
    texels 6 and 7); the CPU loop reads them per batch.
    """
    pos = np.asarray(light.position, np.float32)
    clip = np.append(pos, 1.0).astype(np.float32) @ \
        np.asarray(vp, np.float32).T
    cw = float(clip[3])
    if abs(cw) < 1e-9:
        cw = 1e-9
    ndc_x = float(clip[0]) / cw
    ndc_y = float(clip[1]) / cw
    cx = np.float32((ndc_x * 0.5 + 0.5) * float(w))
    cy = np.float32((ndc_y * 0.5 + 0.5) * float(h))
    half = np.float32(0.5 * float(light.spot_size) * float(h))
    start = np.float32(float(getattr(camera, 'clip_start', 0.1) or 0.1)
                       if camera is not None else 0.1)
    aext = np.float32(abs(float(light.decay_end)))
    return cx, cy, half, half, start, aext


def screen_spot_lobe(px, py, depth, cx, cy, w, h, start, aext):
    """R251 F015: Supermodel's ComputeSpotlight, float32, one op per
    statement -- (en, el, lobe), three (N,) arrays.

    el = max(0, 1 - ((x - cx)/w)^2 - ((y - cy)/h)^2) is the screen
    ellipse at the pixel centre; en gates on depth >= start; the depth
    window is 1/((d/(1 + |extent|) - 1)^2) with d = min(start + aext -
    depth, 0) (full from `start` to start + aext, then fading; q >= 1
    always, so the division never sees zero); lobe = window * ellipse.
    The GLSL twin (gpu/material.py screen_spot_glsl) runs the same
    statements in the same order.
    """
    px = np.asarray(px, np.float32)
    py = np.asarray(py, np.float32)
    depth = np.asarray(depth, np.float32)
    half = np.float32(0.5)
    ex = px + half
    ex = ex - cx
    ex = ex / w
    ey = py + half
    ey = ey - cy
    ey = ey / h
    ex = ex * ex
    ey = ey * ey
    el = np.float32(1.0) - ex
    el = el - ey
    el = np.maximum(el, np.float32(0.0))
    en = np.where(depth >= start, np.float32(1.0),
                  np.float32(0.0)).astype(np.float32)
    z = -depth
    dd = np.float32(start + aext)
    dd = dd + z
    dd = np.minimum(dd, np.float32(0.0))
    qa = np.float32(np.float32(1.0) + aext)
    q = dd / qa
    q = q - np.float32(1.0)
    q = q * q
    rng = en / q
    lobe = rng * el
    return (en.astype(np.float32), el.astype(np.float32),
            lobe.astype(np.float32))
