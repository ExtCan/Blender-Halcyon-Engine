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
              falloff=2.0, edge=None, max_distance=0.0):
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
        total += np.where(live, ang * atten, 0.0).astype(np.float32)

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
