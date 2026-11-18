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


def beam_gain(light):
    """R254: the radiance one unit of marched scatter is worth for `light`.

    The marchers return density * integral(shape / d^falloff) along the
    view ray, with density = the render's Density x the lamp's
    Volumetric. Single scattering in a uniform medium adds

        sigma_s * phase * I / d^2   per unit of path,

    I being the intensity the lamp's SURFACES are lit with and phase the
    isotropic 1 / (4 pi). Density x Volumetric counts sigma_s in units
    of 1 / (4 pi) per world unit (1.0 is a mean free path of 4 pi =
    12.6 units: a thick stage haze; the slider's 4.0 is smoke), so the
    gain is I / (16 pi^2):

    - SPOT, POINT: I = energy / (4 pi), what lights.sample() lights a
      surface with -- energy / (64 pi^3).
    - AREA: the form factor's far field is 2 D^2 times a point lamp's
      (lights.area_inp: stokes * D^2 / A, D the lamp's Distance), so
      I = energy * 2 D^2 / (4 pi) -- energy * D^2 / (32 pi^3). The beam
      follows Distance exactly as the lamp's surfaces do, whatever
      scale the lamp's Power is expressed on. Gamma shapes the surface
      field only.

    The old composite multiplied by energy / pi: 64 pi^2 = 632 times
    this for a spot or point lamp, so any beam at the slider's unit
    value was a solid white wedge. The light style's gain rides the
    energy, as on every other road.
    """
    energy = float(getattr(light, 'energy', 0.0)) \
        * float(getattr(light, 'style_gain', 1.0))
    if str(getattr(light, 'type', '')).upper() == 'AREA':
        d = max(float(getattr(light, 'decay_end', 25.0)), 1e-6)
        return energy * d * d / (32.0 * np.pi ** 3)
    return energy / (64.0 * np.pi ** 3)


def beam_law(light):
    """R254: the cone law the visible beam follows -- the lamp's own."""
    from . import lights as LI
    law = str(getattr(light, 'spot_law', 'BLENDER') or 'BLENDER')
    return law if law in LI.SPOT_LAWS else 'BLENDER'


def beam_edge(light):
    """R254: the Blender Internal beam's soft edge as a fraction of
    (1 - cos(half)): Blend, or the Hotspot's band (lights.spot_blend_band
    over the cone's depth in cosine units). Hotspot 0 returns Blend
    itself, the float the beam always read."""
    if float(getattr(light, 'hotspot', 0.0) or 0.0) <= 0.0:
        return float(getattr(light, 'spot_blend', 0.15))
    from . import lights as LI
    c = float(np.cos(float(getattr(light, 'spot_size', 1.2)) * 0.5))
    return float(LI.spot_blend_band(light, c)) / max(1.0 - c, EPS)


def beam_cos_out(light, cos_half):
    """R254: the cosine the beam's cone is CUT at. Every law is dark at
    the Spot Size except GX Ring 2, which is at full strength there and
    fades 1 - 2 (c - cr)^2 / (1 - cr)^2 to zero at cr - (1 - cr) /
    sqrt(2): its beam is marched out to that zero, or it would end in a
    bright cut rim and lose the outer half of its ring. Clamped below a
    right angle, as cos_half is."""
    if beam_law(light) != 'GX_RING2':
        return cos_half
    c = float(np.cos(float(getattr(light, 'spot_size', 1.2)) * 0.5))
    out = c - (1.0 - c) / float(np.sqrt(2.0))
    return float(max(out, np.cos(np.pi * 0.5 - 1e-3)))


def beam_angular(light, law, cosang, cos_half, edge):
    """R254: the beam's brightness across the cone for cos(angle to the
    axis). Blender Internal: the smoothstep over the blend band the beam
    always drew (no cosine roll-off: the air is not a surface). Every
    other law: lights.spot_law_factor itself, so POV's flat hotspot and
    Hermite edge, both cosine exponents, GL's hard rim and the GX rings
    shape the beam as they shape the pool."""
    if law == 'BLENDER':
        inner = cos_half + (1.0 - cos_half) * np.clip(edge, 0.0, 0.999)
        ang = np.clip((cosang - cos_half) / np.maximum(inner - cos_half, EPS),
                      0.0, 1.0)
        return ang * ang * (3.0 - 2.0 * ang)
    from . import lights as LI
    return LI.spot_law_factor(light, np.asarray(cosang, np.float32))


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
        edge = beam_edge(light)
    law = beam_law(light)

    if origin.ndim == 1:
        origin = origin[None, :]
    if origin.shape[0] == 1:
        origin = np.repeat(origin, n, axis=0)

    # R254: a Spot scaled unevenly across its beam lights an ELLIPSE,
    # and its shaft is the same elliptical cone: in the lamp's own
    # frame with the side / up components multiplied by spot_scale the
    # cone is round again (half-angle unchanged), a line stays a line,
    # and the view ray's parameter only stretches by the squeezed
    # direction's length. A round lamp builds none of it
    squeeze = None
    sc = getattr(light, 'spot_scale', None)
    if sc is not None and (float(sc[0]), float(sc[1])) != (1.0, 1.0):
        from . import lights as _LI
        s_ax, u_ax, f_ax = _LI.cookie_frame(light)
        squeeze = np.stack([np.asarray(s_ax, np.float32) * np.float32(sc[0]),
                            np.asarray(u_ax, np.float32) * np.float32(sc[1]),
                            np.asarray(f_ax, np.float32)], axis=0)
        o_q = (origin - apex[None, :]) @ squeeze.T
        r_q = rays @ squeeze.T
        r_len = np.sqrt(np.maximum(np.einsum('ij,ij->i', r_q, r_q), EPS))
        t0, t1, hit = _cone_segment(
            o_q, r_q / r_len[:, None], np.zeros(3, np.float32),
            np.array([0.0, 0.0, 1.0], np.float32),
            beam_cos_out(light, cos_half))
        t0 = t0 / r_len
        t1 = t1 / r_len
    else:
        t0, t1, hit = _cone_segment(origin, rays, apex, axis,
                                    beam_cos_out(light, cos_half))

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
        if squeeze is None:
            cosang = np.einsum('ij,j->i', d, axis) / dist
        else:
            # the angle to the axis in the squeezed frame; the distance
            # the falloff reads stays the real one
            d_q = d @ squeeze.T
            cosang = d_q[:, 2] / np.sqrt(np.maximum(
                np.einsum('ij,ij->i', d_q, d_q), EPS))

        # angular falloff -- soft toward the rim, as the spot's own blend
        # does (R254: or as the lamp's cone law shapes it)
        ang = beam_angular(light, law, cosang, cos_half, edge)

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
    # R254: the samples are spaced by ANGLE about the lamp, not by
    # distance along the ray. The segment through a 64-unit sphere is
    # tens of units long and 1 / d^2 peaks within a unit or two of the
    # ray's closest approach, so twelve evenly spaced midpoints mostly
    # stepped over the peak: the glow came out as rings, twice the true
    # integral at 12 samples and a third of it at 2. With t = tc + D
    # tan(theta) (tc the closest approach, D its distance from the
    # lamp) dt = d^2 / D dtheta, which cancels the inverse square: the
    # sum is smooth and within a percent at any count
    tc = -b
    perp = np.sqrt(np.maximum(np.einsum('ij,ij->i', oc, oc) - b * b, 1e-6))
    th0 = np.arctan((t0 - tc) / perp)
    th1 = np.arctan((t1 - tc) / perp)
    dth = np.where(live, th1 - th0, 0.0)
    for off in offsets:
        t = tc + perp * np.tan(th0 + dth * off)
        p = origin + rays * t[:, None]
        d = p - centre[None, :]
        dist = np.sqrt(np.maximum(np.einsum('ij,ij->i', d, d), EPS))
        fade = np.clip(1.0 - (dist / R) * (dist / R), 0.0, 1.0)
        atten = fade / np.maximum(np.power(dist, falloff), EPS)
        contrib = np.where(live, atten * (dist * dist / perp), 0.0)
        if occlude is not None:
            contrib = np.where(occlude(p, live & (contrib > 0.0)),
                               0.0, contrib)
        total += contrib.astype(np.float32)
    return (total * (dth / steps) * density * np.where(live, 1.0, 0.0)
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
        out = out + scatter[:, None] * col * beam_gain(light)
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
    edge = beam_edge(light)
    law = beam_law(light)
    cos_out = beam_cos_out(light, cos_half)
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
        inside = cosang > cos_out
        # R254: the same law statement as the fast path (the law is the
        # lamp's, not the march's: what this twin checks is the segment)
        ang = beam_angular(light, law, cosang, cos_half, edge)
        atten = 1.0 / np.maximum(np.power(dist, falloff), EPS)
        total += np.where(inside, ang * atten, 0.0).astype(np.float32) * step
    return (total * density).astype(np.float32)
