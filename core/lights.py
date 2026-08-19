"""Light evaluation: attenuation, spot cones, shadow maps and ray shadows.

Shadow maps are rendered with the same rasteriser as the beauty pass, then
compared in *linear light-space distance* rather than NDC z -- which makes the
bias a world-space number the user can reason about, instead of a magic
depth-buffer epsilon that has to be retuned for every scene scale.
"""

import numpy as np

from . import mathx as M
from . import raster

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

    __slots__ = ('vp', 'depth', 'near', 'far', 'persp', 'size', 'origin',
                 'extent')

    def __init__(self, vp, zndc, near, far, persp, size, origin, extent=1.0):
        self.vp = vp
        self.extent = float(extent)
        self.near = float(near)
        self.far = float(far)
        self.persp = bool(persp)
        self.size = int(size)
        self.origin = np.asarray(origin, np.float32)
        self.depth = self._linearise(zndc)

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

    __slots__ = ('faces', 'origin')

    def __init__(self, faces, origin):
        self.faces = faces
        self.origin = np.asarray(origin, np.float32)

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


def _render_depth(verts, tris, vp, size, cull='NONE'):
    # A per-map frustum cull was tried here (R87) and REVERTED on the
    # measurement: a concentrated high-poly object sits inside most map
    # frustums, so the cull kept ~100% of triangles and its own gather
    # cost made the stage 15% SLOWER. The rasteriser's early discard
    # already handles the faces that see nothing. The honest lever for
    # high-poly shadow cost is the rasteriser's per-triangle speed.
    gb = raster.GBuffer(size, size)
    raster.rasterize(verts, tris, vp, size, size, cull=cull, gbuf=gb,
                     depth_bits=32)
    return gb.zndc


_SHADOW_CACHE = {}


def clear_shadow_cache():
    _SHADOW_CACHE.clear()


def _shadow_base_signature(scene, settings, caster_tris):
    """Fingerprint of what EVERY shadow map depends on: the geometry,
    the shadow settings and the caster set -- everything except the
    light itself."""
    mesh = scene.mesh
    v = mesh.verts
    step = max(1, v.shape[0] // 512)
    geo = (v.shape[0], float(v[::step].sum()), float(v[::step, 0].dot(
        np.arange(v[::step].shape[0], dtype=np.float32))))
    return (geo, settings.shadows, settings.shadow_default,
            int(settings.shadow_map_size),
            # size AND sum: two caster sets of equal count (say, two
            # same-sized materials toggling Shadow > Cast) must not
            # collide in the cache
            None if caster_tris is None else
            (int(caster_tris.size), int(caster_tris.sum())))


def _light_shadow_signature(light):
    """The one light's own contribution to its map: pose and lens.

    Colour and energy are deliberately absent -- they tint the LIGHT,
    never the depth -- so palette edits ride the cache untouched."""
    return (light.type, tuple(np.round(light.position, 5)),
            tuple(np.round(light.direction, 5)),
            round(float(light.spot_size), 5), light.shadow,
            int(light.shadow_map_size))


def build_shadow_maps(scene, settings, caster_tris=None):
    """Bake a shadow map (or cube) for every light that wants one.

    Each map is a full rasterisation pass, and a point light needs six.
    None of it changes while the lights and geometry hold still, which
    for most of an animation is all of it. The cache is PER LIGHT:
    the old whole-list fingerprint meant nudging ONE lamp of fifteen
    re-rasterised every map in the scene ('if I wanted to move
    lights... it will lag like crazy'); now only the moved lamp's own
    map rebuilds, and a colour or energy edit rebuilds nothing at all.
    """
    if getattr(settings, 'cache_shadows', True) and scene.mesh is not None \
            and scene.mesh.verts is not None and scene.mesh.verts.size:
        try:
            base = _shadow_base_signature(scene, settings, caster_tris)
        except Exception:                                       # noqa: BLE001
            base = None
        if base is not None:
            misses = []
            for i, light in enumerate(scene.lights):
                try:
                    key = (base, _light_shadow_signature(light))
                except Exception:                               # noqa: BLE001
                    misses.append((i, None))
                    continue
                hit = _SHADOW_CACHE.get(key)
                if hit is not None:
                    light.shadow_map = hit
                else:
                    misses.append((i, key))
            if misses:
                _build_shadow_maps(scene, settings, caster_tris,
                                   only={i for i, _k in misses})
                for i, key in misses:
                    sm = scene.lights[i].shadow_map
                    if key is not None and sm is not None:
                        if len(_SHADOW_CACHE) > 64:
                            _SHADOW_CACHE.clear()
                        _SHADOW_CACHE[key] = sm
            return
    _build_shadow_maps(scene, settings, caster_tris)


def _build_shadow_maps(scene, settings, caster_tris=None, only=None):
    """Build maps for every map-needing light, or -- with `only`, a set
    of light indices -- just those, leaving the rest untouched (the
    per-light cache hands the untouched ones their parked maps)."""
    mesh = scene.mesh
    if mesh is None or mesh.verts is None or mesh.tris is None:
        return
    tris = mesh.tris
    if caster_tris is not None:
        tris = tris[caster_tris]
    if tris.shape[0] == 0:
        return
    centre, radius = scene_bounds(mesh.verts)

    # PLAN first (cheap geometry per map), RENDER second -- because the
    # renders are independent: each map is its own buffer, its own depth
    # min-compares, deterministic in isolation, so thread scheduling
    # cannot move a bit. On a high-poly scene a point light is six full
    # rasterisations and they used to run one after another; the
    # rasteriser is big-array NumPy that releases the interpreter lock,
    # which is the one shape of work threads genuinely scale on this
    # renderer (the shading loop is not -- see the Threads tooltip).
    jobs = []                      # (assign, vp, size) -- assign(depth)
    for li, light in enumerate(scene.lights):
        if only is not None and li not in only:
            continue
        light.shadow_map = None
        if not settings.shadows or light.type == 'AMBIENT':
            continue
        mode = light.shadow if settings.shadow_default == 'PER_LIGHT' else \
            settings.shadow_default
        if mode != 'MAP' or light.shadow == 'NONE':
            continue
        size = int(light.shadow_map_size or settings.shadow_map_size)
        size = max(32, min(4096, size))
        pos = np.asarray(light.position, np.float32)
        if light.type == 'SUN':
            d = M.normalize(np.asarray(light.direction, np.float32))
            eye = centre - d * (radius * 2.5)
            view = _look_at_lh(eye, centre, np.array([0, 0, 1.0], np.float32))
            near = 1e-3
            far = radius * 5.0
            half = radius * 1.05
            vp = _ortho(half, near, far) @ view

            def assign(depth, light=light, vp=vp, near=near, far=far,
                       size=size, eye=eye, half=half):
                light.shadow_map = ShadowMap(vp, depth, near, far, False,
                                             size, eye, half)
            jobs.append((assign, vp, size))
        elif light.type == 'SPOT':
            d = M.normalize(np.asarray(light.direction, np.float32))
            view = _look_at_lh(pos, pos + d, np.array([0, 0, 1.0], np.float32))
            near = max(radius * 0.005, 1e-3)
            far = max(float(np.linalg.norm(centre - pos)) + radius * 1.5, near * 10)
            fov = min(max(float(light.spot_size) * 1.15, 0.05), 3.0)
            vp = _persp(fov, 1.0, near, far) @ view

            def assign(depth, light=light, vp=vp, near=near, far=far,
                       size=size, pos=pos, fov=fov):
                light.shadow_map = ShadowMap(vp, depth, near, far, True,
                                             size, pos,
                                             float(np.tan(fov * 0.5)))
            jobs.append((assign, vp, size))
        else:
            near = max(radius * 0.005, 1e-3)
            far = max(float(np.linalg.norm(centre - pos)) + radius * 1.5, near * 10)
            proj = _persp(np.pi * 0.5, 1.0, near, far)
            slots = [None] * len(CUBE_DIRS)
            pending = {'n': len(CUBE_DIRS)}
            for fi, (fwd, up) in enumerate(CUBE_DIRS):
                view = _look_at_lh(pos, pos + fwd, up)
                vp = proj @ view

                def assign(depth, light=light, vp=vp, near=near, far=far,
                           size=size, pos=pos, fi=fi, slots=slots,
                           pending=pending):
                    slots[fi] = ShadowMap(vp, depth, near, far, True,
                                          size, pos, 1.0)
                    pending['n'] -= 1
                    if pending['n'] == 0:
                        light.shadow_map = CubeShadow(list(slots), pos)
                jobs.append((assign, vp, size))

    if not jobs:
        return
    parallel = len(jobs) > 1 and tris.shape[0] >= 50000
    if parallel:
        import concurrent.futures as _fut
        import os as _os
        workers = min(len(jobs), max((_os.cpu_count() or 2) - 1, 2))
        with _fut.ThreadPoolExecutor(max_workers=workers) as pool:
            depths = list(pool.map(
                lambda j: _render_depth(mesh.verts, tris, j[1], j[2]), jobs))
        for (assign, _vp, _s), depth in zip(jobs, depths):
            assign(depth)
    else:
        for assign, vp, size in jobs:
            assign(_render_depth(mesh.verts, tris, vp, size))


# ------------------------------------------------------------- light sampling


def attenuate(light, dist, settings=None):
    mode = light.decay
    if mode == 'DEFAULT' and settings is not None:
        mode = settings.light_falloff_default
    start = float(light.decay_start)
    if mode == 'NONE':
        att = np.ones_like(dist)
        if getattr(light, 'bi_sphere', False):
            # LA_SPHERE applies OUTSIDE the falloff switch in the C --
            # even a Constant-falloff lamp clips at its Distance
            D = max(float(light.decay_end), EPS)
            t = D - np.maximum(dist, 0.0)
            att = np.where(t <= 0.0, 0.0, att * t / D)
        return att.astype(np.float32)
    d = np.maximum(dist - start, EPS) if start > 0 else np.maximum(dist, EPS)
    if mode == 'INVERSE':
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
    else:
        att = 1.0 / (d * d)
    if getattr(light, 'bi_sphere', False):
        # LA_SPHERE, verbatim: *= (D - d)/D, hard zero past the lamp's
        # Distance
        D = max(float(light.decay_end), EPS)
        t = D - np.maximum(dist, 0.0)
        att = np.where(t <= 0.0, 0.0, att * t / D)
    return att.astype(np.float32)


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
    spotsi = np.float32(np.cos(float(light.spot_size) * 0.5))
    spotbl = np.float32((1.0 - spotsi) * float(light.spot_blend))
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


def _cookie_texture(light):
    tex = getattr(light, '_cookie_tex', None)
    if tex is None:
        from .texture import Texture
        px = getattr(light, 'cookie', None)
        px = getattr(px, 'pixels', px)
        tex = Texture(np.asarray(px, np.float32), name='cookie',
                      colorspace='Non-Color')
        try:
            light._cookie_tex = tex
        except Exception:                                       # noqa: BLE001
            pass
    return tex


def cookie_factor(light, P, L):
    """Per-point rgb multiplier from a light's projected texture.

    The sixth-generation consoles' projective texturing: a SPOT maps its
    full cone onto the image (the cone edge lands on the image edge, the
    lookup clamps outside it), a SUN projects the image along its rays and
    REPEATS it every `cookie_scale` world units -- the scrolling cloud
    shadow of the era. Bilinear both here and in the GLSL mirror, with the
    same texel arithmetic. Returns (N,3), all ones where the projection is
    undefined (behind a spot). POINT and AREA lights return ones -- a
    single 2D image has no defined mapping around a point.
    """
    n = P.shape[0]
    kind = getattr(light, 'type', 'POINT')
    if getattr(light, 'cookie', None) is None or \
            kind not in ('SPOT', 'SUN'):
        return np.ones((n, 3), np.float32)
    s, u, f = cookie_frame(light)
    tex = _cookie_texture(light)
    strength = float(np.clip(getattr(light, 'cookie_strength', 1.0), 0.0, 1.0))
    if kind == 'SUN':
        scale = max(float(getattr(light, 'cookie_scale', 10.0)), 1e-6)
        cu = (P @ s.astype(np.float32)) / scale
        cv = (P @ u.astype(np.float32)) / scale
        rgb = tex.sample(cu.astype(np.float32), cv.astype(np.float32),
                         filt='BILINEAR', wrap='REPEAT')[:, :3]
        return (1.0 + (rgb - 1.0) * strength).astype(np.float32)
    # SPOT: direction light -> surface, expressed in the light's own frame;
    # the full cone spans the image, so uv = d_side / (d_fwd * 2 tan(half))
    d = -L
    dz = d @ f.astype(np.float32)
    tanh = max(np.tan(float(light.spot_size) * 0.5), 1e-6)
    safe = np.maximum(dz, 1e-6) * (2.0 * tanh)
    cu = (d @ s.astype(np.float32)) / safe + 0.5
    cv = (d @ u.astype(np.float32)) / safe + 0.5
    rgb = tex.sample(cu.astype(np.float32), cv.astype(np.float32),
                     filt='BILINEAR', wrap='EXTEND')[:, :3]
    out = 1.0 + (rgb - 1.0) * strength
    out[dz <= 1e-6] = 1.0
    return out.astype(np.float32)


def sample(light, P, settings, area_sample=None):
    """Direction to the light, incoming radiance and distance.

    Returns (L (N,3) unit, radiance (N,3), dist (N,)).
    """
    n = P.shape[0]
    col = np.asarray(light.color, np.float32)[None, :]
    energy = float(light.energy)
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
    att = attenuate(light, dist, settings)
    if light.type == 'SPOT':
        att = att * spot_falloff(light, L)
    if str(getattr(light, 'decay', '')).startswith('BI_') or \
            getattr(light, 'bi_sphere', False):
        # lamp_get_visibility's tail, verbatim (R155): a combined
        # visifac at or below 0.001 is snapped to zero
        att = np.where(att <= 0.001, 0.0, att)
    scale = energy / (4.0 * np.pi)
    rad = col * (scale * att)[:, None]
    if light.type == 'SPOT' and getattr(light, 'cookie', None) is not None:
        rad = rad * cookie_factor(light, P, L)
    return L.astype(np.float32), rad.astype(np.float32), dist


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
    if mode == 'NONE':
        return np.ones(n, np.float32)

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
        samples = max(1, int(settings.shadow_samples)) if light.radius > 0 else 1
        if samples == 1:
            hit = bvh.occluded(origin, L, maxt, mask=mask)
            return (~hit).astype(np.float32)
        acc = np.zeros(n, np.float32)
        t, b = M.orthonormal_basis(L)
        if sample_xy is not None:
            from . import patterns as PT
            spx, spy, li = sample_xy
            seed = int(getattr(settings, 'seed', 0) or 0)
            radius = np.float32(light.radius)
            for k in range(samples):
                z = 2 * k + 131 * int(li) + 7919 * seed
                u1 = PT.sample_u(spx, spy, z)
                ca, sa = PT.sample_circle(PT.sample_u(spx, spy, z + 1))
                r = np.sqrt(u1) * radius
                jitter = t * (r * ca)[:, None] + b * (r * sa)[:, None]
                Lj = M.normalize(L * dist[:, None] + jitter)
                acc += (~bvh.occluded(origin, Lj, maxt,
                                      mask=mask)).astype(np.float32)
            return acc / samples
        rng = rng or np.random.default_rng(settings.seed)
        for _ in range(samples):
            r = np.sqrt(rng.random()) * light.radius
            th = rng.random() * 2 * np.pi
            jitter = t * (r * np.cos(th)) + b * (r * np.sin(th))
            Lj = M.normalize(L * dist[:, None] + jitter)
            acc += (~bvh.occluded(origin, Lj, maxt,
                                  mask=mask)).astype(np.float32)
        return acc / samples

    sm = light.shadow_map
    if sm is None:
        return np.ones(n, np.float32)
    bias = float(light.shadow_bias or settings.shadow_bias)
    ndl = np.clip(M.dot(N, L), 0.0, 1.0)
    slope = bias * (1.0 + 2.0 * (1.0 - ndl))
    soft = max(float(light.shadow_softness) * settings.shadow_softness, 0.0)
    # normal offset: step off the surface by a texel or so, scaled by how
    # obliquely the light hits. Removes acne without the detached shadows a
    # large constant depth bias produces.
    texel = sm.texel_size(np.linalg.norm(P - sm.origin[None, :], axis=1))
    offset = texel * (1.5 + 2.5 * np.sqrt(np.maximum(1.0 - ndl * ndl, 0.0))) * \
        max(1.0, soft)
    lit = sm.lookup(P + N * offset[:, None], slope, soft, settings.shadow_samples)
    dens = float(light.shadow_density)
    return (1.0 - (1.0 - lit) * dens).astype(np.float32)


def select_lights(lights, settings, centre=None):
    """Emulate a hardware light limit by keeping only the strongest N."""
    active = [l for l in lights if l.type != 'AMBIENT']
    limit = int(settings.max_lights)
    if limit <= 0 or len(active) <= limit:
        return active
    if settings.light_limit_mode == 'FIRST':
        return active[:limit]
    if settings.light_limit_mode == 'NEAREST' and centre is not None:
        key = [float(np.linalg.norm(np.asarray(l.position, np.float32) - centre))
               for l in active]
        order = np.argsort(key)
    else:
        key = [float(l.energy) * float(np.mean(l.color)) for l in active]
        order = np.argsort(key)[::-1]
    return [active[i] for i in order[:limit]]


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
            wrld = wrld + np.asarray(l.color, np.float32) * l.energy
    return eng.astype(np.float32), wrld.astype(np.float32)


def ambient_light(scene, settings):
    eng, wrld = ambient_light_split(scene, settings)
    return (eng + wrld).astype(np.float32)
