"""Visible spotlight cones -- the beam you can see in the air.

Every package of the era had these and none of them integrated a volume
properly: LightWave called them volumetric lights, 3D Studio called them Volume
Lights, and both were marching a handful of samples down the view ray and adding
up whatever fell inside the cone. That is what this does, because that is what
it looked like.

The cone is intersected analytically rather than tessellated into geometry. A
ray against an infinite double cone is a quadratic, and the two roots bracket
the segment of the view ray that is inside the beam. Clip that segment by the
scene depth so the beam stops at whatever it hits, march a few samples along
what is left, and sum.

Sample count is exposed because low counts band, and the banding is part of the
look -- it is the same slicing artefact those renderers had, and hiding it with
a big default would be the wrong kind of accurate.
"""

import numpy as np

from . import mathx as M

EPS = 1e-6


def _cone_segment(origin, rays, apex, axis, cos_half):
    """Where a view ray enters and leaves an infinite cone.

    Returns (t0, t1, hit). Both roots of the quadratic, ordered, with anything
    behind the apex or behind the camera discarded. `hit` is False where the
    ray misses the cone entirely.
    """
    co = origin - apex[None, :]
    rd = np.einsum('ij,j->i', rays, axis)
    cd = float(np.dot(co[0], axis)) if co.shape[0] == 1 else np.einsum(
        'ij,j->i', co, axis)
    rc = np.einsum('ij,ij->i', rays, co)
    cc = np.einsum('ij,ij->i', co, co)

    k = cos_half * cos_half
    a = rd * rd - k
    b = 2.0 * (rd * cd - rc * k)
    c = cd * cd - cc * k

    disc = b * b - 4.0 * a * c
    hit = disc >= 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))

    # a -> 0 means the ray runs parallel to the cone's surface: one root only
    near_par = np.abs(a) < EPS
    denom = np.where(near_par, 1.0, 2.0 * a)
    t_a = (-b - sq) / denom
    t_b = (-b + sq) / denom
    lin = np.where(np.abs(b) > EPS, -c / np.where(np.abs(b) > EPS, b, 1.0), 0.0)
    t_a = np.where(near_par, lin, t_a)
    t_b = np.where(near_par, np.inf, t_b)

    t0 = np.minimum(t_a, t_b)
    t1 = np.maximum(t_a, t_b)

    # the quadratic describes a double cone; keep only the half the light faces
    def forward(t):
        p = origin + rays * t[:, None]
        return np.einsum('ij,j->i', p - apex[None, :], axis) > 0.0

    f0 = forward(np.where(np.isfinite(t0), t0, 0.0))
    f1 = forward(np.where(np.isfinite(t1), t1, 0.0))

    # if the near root is on the mirrored half, the segment starts at the far one
    t0 = np.where(f0, t0, np.where(f1, t1, np.inf))
    t1 = np.where(f0 & f1, t1, np.where(f0 | f1, np.inf, -np.inf))
    hit = hit & (f0 | f1)
    return t0, t1, hit


#: how far a beam is drawn when nothing stops it. A cone is infinite and its
#: contribution converges under inverse-square falloff, but the integration
#: needs a finite bound, and a visible beam that never ends looks wrong anyway.
DEFAULT_REACH = 64.0


def spot_cone(origin, rays, depth, light, samples=12, density=1.0,
              falloff=2.0, edge=None, max_distance=0.0, occlude=None):
    """Scattered light along each view ray for one spot light.

    `origin` is (1,3) or (N,3), `rays` are unit view directions (N,3), `depth`
    is the distance to the nearest surface per ray (inf where nothing was hit).
    Returns (N,) scattering amounts, before the light's colour is applied.
    """
    n = rays.shape[0]
    if str(getattr(light, 'type', '')).upper() != 'SPOT':
        return np.zeros(n, np.float32)

    apex = np.asarray(light.position, np.float32)
    axis = M.normalize(np.asarray(light.direction, np.float32)[None, :])[0]
    half = max(float(getattr(light, 'spot_size', 1.2)) * 0.5, 1e-3)
    cos_half = float(np.cos(min(half, np.pi * 0.5 - 1e-3)))
    if edge is None:
        edge = float(getattr(light, 'spot_blend', 0.15))

    if origin.ndim == 1:
        origin = origin[None, :]
    if origin.shape[0] == 1:
        origin = np.repeat(origin, n, axis=0)

    t0, t1, hit = _cone_segment(origin, rays, apex, axis, cos_half)

    # the beam cannot start behind the camera, and it stops at the first surface
    t0 = np.maximum(t0, 0.0)
    reach = max_distance if max_distance > 0.0 else DEFAULT_REACH
    limit = np.where(np.isfinite(depth), depth, reach)
    limit = np.minimum(limit, t0 + reach)
    t1 = np.minimum(np.where(np.isfinite(t1), t1, limit), limit)

    span = np.where(np.isfinite(t0) & np.isfinite(t1), t1 - t0, 0.0)
    span = np.nan_to_num(span, nan=0.0, posinf=0.0, neginf=0.0)
    t0 = np.nan_to_num(t0, nan=0.0, posinf=0.0, neginf=0.0)
    live = hit & (span > EPS)
    if not live.any():
        return np.zeros(n, np.float32)

    steps = max(int(samples), 1)
    # sample at segment midpoints: with few steps this is visibly better than
    # sampling the ends, and few steps is the whole point
    offsets = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    total = np.zeros(n, np.float32)

    for off in offsets:
        t = t0 + span * off
        p = origin + rays * t[:, None]
        d = p - apex[None, :]
        dist = np.sqrt(np.maximum(np.einsum('ij,ij->i', d, d), EPS))
        cosang = np.einsum('ij,j->i', d, axis) / dist

        # angular falloff -- soft toward the rim, as the spot's own blend does
        inner = cos_half + (1.0 - cos_half) * np.clip(edge, 0.0, 0.999)
        ang = np.clip((cosang - cos_half) / np.maximum(inner - cos_half, EPS),
                      0.0, 1.0)
        ang = ang * ang * (3.0 - 2.0 * ang)

        atten = 1.0 / np.maximum(np.power(dist, falloff), EPS)
        contrib = np.where(live, ang * atten, 0.0)
        if occlude is not None:
            # beam shadowing: a sample the lamp cannot reach scatters
            # nothing -- the option that makes the beam STOP at a mesh
            contrib = np.where(occlude(p, live & (contrib > 0.0)),
                               0.0, contrib)
        total += contrib.astype(np.float32)

    return (total * (span / steps) * density * np.where(live, 1.0, 0.0)
            ).astype(np.float32)


def point_glow(origin, rays, depth, light, samples=12, density=1.0,
               falloff=2.0, max_distance=0.0, occlude=None):
    """Scattered light around a POINT lamp: the omni volume glow.

    The beam region is a sphere: `max_distance` bounds it (a light's
    Custom Range end narrows it further at the caller), the view ray's
    segment inside it is found analytically, and the same midpoint march
    the spot cone uses integrates 1/d^falloff with a smooth fade to the
    sphere's edge so the boundary never draws itself.
    """
    n = rays.shape[0]
    centre = np.asarray(light.position, np.float32)
    R = max_distance if max_distance > 0.0 else DEFAULT_REACH
    if origin.ndim == 1:
        origin = origin[None, :]
    if origin.shape[0] == 1:
        origin = np.repeat(origin, n, axis=0)

    oc = origin - centre[None, :]
    b = np.einsum('ij,ij->i', rays, oc)
    c0 = np.einsum('ij,ij->i', oc, oc) - R * R
    disc = b * b - c0
    hit = disc > 0.0
    sq = np.sqrt(np.maximum(disc, 0.0))
    t0 = np.maximum(-b - sq, 0.0)
    t1 = -b + sq
    limit = np.where(np.isfinite(depth), depth, t1)
    t1 = np.minimum(t1, limit)
    span = np.where(hit, t1 - t0, 0.0)
    span = np.nan_to_num(span, nan=0.0, posinf=0.0, neginf=0.0)
    live = hit & (span > EPS)
    if not live.any():
        return np.zeros(n, np.float32)

    steps = max(int(samples), 1)
    offsets = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    total = np.zeros(n, np.float32)
    for off in offsets:
        t = t0 + span * off
        p = origin + rays * t[:, None]
        d = p - centre[None, :]
        dist = np.sqrt(np.maximum(np.einsum('ij,ij->i', d, d), EPS))
        fade = np.clip(1.0 - (dist / R) * (dist / R), 0.0, 1.0)
        atten = fade / np.maximum(np.power(dist, falloff), EPS)
        contrib = np.where(live, atten, 0.0)
        if occlude is not None:
            contrib = np.where(occlude(p, live & (contrib > 0.0)),
                               0.0, contrib)
        total += contrib.astype(np.float32)
    return (total * (span / steps) * density * np.where(live, 1.0, 0.0)
            ).astype(np.float32)


def area_beam(origin, rays, depth, light, samples=12, density=1.0,
              falloff=2.0, max_distance=0.0, occlude=None):
    """Scattered light in front of an AREA lamp: a soft-edged slab beam.

    The beam is the lamp's rectangle (or ellipse) swept along its
    facing direction -- an open box, intersected as three slab pairs in
    the lamp's own frame, marched like the cone. The cross-section
    fades over a margin proportional to the lamp's size, so the beam
    reads as light, not as geometry; falloff runs on the distance
    travelled from the panel.
    """
    n = rays.shape[0]
    pos = np.asarray(light.position, np.float32)
    f = M.normalize(np.asarray(light.direction, np.float32)[None, :])[0]
    ax = np.asarray(getattr(light, 'area_x', (1, 0, 0)), np.float32)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
    ay = np.asarray(getattr(light, 'area_y', (0, 1, 0)), np.float32)
    ay = ay / max(float(np.linalg.norm(ay)), 1e-9)
    asz = getattr(light, 'area_size', (1.0, 1.0))
    hx = max(float(asz[0]) * 0.5, 1e-4)
    hy = max(float(asz[1]) * 0.5, 1e-4)
    margin = 0.35 * max(hx, hy)
    reach = max_distance if max_distance > 0.0 else DEFAULT_REACH
    disk = str(getattr(light, 'area_shape', 'SQUARE')) in ('DISK', 'ELLIPSE')

    if origin.ndim == 1:
        origin = origin[None, :]
    if origin.shape[0] == 1:
        origin = np.repeat(origin, n, axis=0)

    # ray in the lamp frame
    q = origin - pos[None, :]
    qu = q @ ax
    qv = q @ ay
    qz = q @ f
    du = rays @ ax
    dv = rays @ ay
    dz = rays @ f

    def _slab(o, d, lo, hi):
        safe = np.where(np.abs(d) < 1e-9, 1e-9, d)
        ta = (lo - o) / safe
        tb = (hi - o) / safe
        t_in = np.minimum(ta, tb)
        t_out = np.maximum(ta, tb)
        flat = np.abs(d) < 1e-9
        inside = (o >= lo) & (o <= hi)
        t_in = np.where(flat, np.where(inside, -np.inf, np.inf), t_in)
        t_out = np.where(flat, np.where(inside, np.inf, -np.inf), t_out)
        return t_in, t_out

    ui, uo = _slab(qu, du, -(hx + margin), hx + margin)
    vi, vo = _slab(qv, dv, -(hy + margin), hy + margin)
    zi, zo = _slab(qz, dz, 0.0, reach)
    t0 = np.maximum(np.maximum(ui, vi), np.maximum(zi, 0.0))
    t1 = np.minimum(np.minimum(uo, vo), zo)
    limit = np.where(np.isfinite(depth), depth, reach * 4.0)
    t1 = np.minimum(t1, limit)
    span = t1 - t0
    span = np.nan_to_num(span, nan=0.0, posinf=0.0, neginf=0.0)
    live = span > EPS
    if not live.any():
        return np.zeros(n, np.float32)

    steps = max(int(samples), 1)
    offsets = (np.arange(steps, dtype=np.float32) + 0.5) / steps
    total = np.zeros(n, np.float32)
    for off in offsets:
        t = t0 + span * off
        u = qu + du * t
        v = qv + dv * t
        z = qz + dz * t
        if disk:
            e = np.sqrt(np.maximum((u / hx) ** 2 + (v / hy) ** 2, EPS))
            lat = np.clip((1.0 + margin / max(hx, hy) - e)
                          / max(margin / max(hx, hy), 1e-6), 0.0, 1.0)
        else:
            eu = np.clip(1.0 - (np.abs(u) - hx) / margin, 0.0, 1.0)
            ev = np.clip(1.0 - (np.abs(v) - hy) / margin, 0.0, 1.0)
            lat = np.minimum(eu, ev)
        lat = lat * lat * (3.0 - 2.0 * lat)
        fwd = np.clip(z / max(reach * 0.05, 1e-6), 0.0, 1.0)   # soft start
        atten = 1.0 / np.maximum(np.power(np.maximum(z, EPS), falloff), EPS)
        end_fade = np.clip(1.0 - (z / reach) * (z / reach), 0.0, 1.0)
        contrib = np.where(live & (z > 0.0),
                           lat * fwd * end_fade * atten, 0.0)
        if occlude is not None:
            p = origin + rays * t[:, None]
            contrib = np.where(occlude(p, live & (contrib > 0.0)),
                               0.0, contrib)
        total += contrib.astype(np.float32)
    return (total * (span / steps) * density * np.where(live, 1.0, 0.0)
            ).astype(np.float32)


def add_cones(rgb, origin, rays, depth, lights, st):
    """Add every spot light's visible cone into a rendered frame."""
    if not getattr(st, 'spot_cones', False):
        return rgb
    strength = float(getattr(st, 'spot_cone_density', 1.0))
    if strength <= 0.0:
        return rgb
    samples = int(getattr(st, 'spot_cone_samples', 12))
    falloff = float(getattr(st, 'spot_cone_falloff', 2.0))
    out = rgb
    for light in lights or ():
        amount = float(getattr(light, 'volumetric', 0.0))
        if amount <= 0.0 or str(getattr(light, 'type', '')).upper() != 'SPOT':
            continue
        scatter = spot_cone(origin, rays, depth, light, samples=samples,
                            density=strength * amount, falloff=falloff)
        if not scatter.any():
            continue
        col = np.asarray(light.color, np.float32)[None, :]
        energy = float(getattr(light, 'energy', 1000.0))
        out = out + scatter[:, None] * col * energy * (1.0 / np.pi)
    return out


def reference(origin, rays, depth, light, samples=4096, density=1.0,
              falloff=2.0, max_distance=DEFAULT_REACH):
    """A brute-force march used only to check the analytic version.

    Walks the whole ray in fixed steps and tests containment explicitly rather
    than solving for the entry and exit points. Far too slow to render with,
    which is the point: it shares no code with the fast path.
    """
    n = rays.shape[0]
    apex = np.asarray(light.position, np.float32)
    axis = M.normalize(np.asarray(light.direction, np.float32)[None, :])[0]
    half = max(float(light.spot_size) * 0.5, 1e-3)
    cos_half = float(np.cos(min(half, np.pi * 0.5 - 1e-3)))
    edge = float(light.spot_blend)
    if origin.ndim == 1:
        origin = origin[None, :]
    if origin.shape[0] == 1:
        origin = np.repeat(origin, n, axis=0)

    limit = np.minimum(np.where(np.isfinite(depth), depth, max_distance),
                       max_distance)
    step = limit / samples
    total = np.zeros(n, np.float32)
    for i in range(samples):
        t = step * (i + 0.5)
        p = origin + rays * t[:, None]
        d = p - apex[None, :]
        dist = np.sqrt(np.maximum(np.einsum('ij,ij->i', d, d), EPS))
        cosang = np.einsum('ij,j->i', d, axis) / dist
        inside = cosang > cos_half
        inner = cos_half + (1.0 - cos_half) * np.clip(edge, 0.0, 0.999)
        ang = np.clip((cosang - cos_half) / np.maximum(inner - cos_half, EPS),
                      0.0, 1.0)
        ang = ang * ang * (3.0 - 2.0 * ang)
        atten = 1.0 / np.maximum(np.power(dist, falloff), EPS)
        total += np.where(inside, ang * atten, 0.0).astype(np.float32) * step
    return (total * density).astype(np.float32)
