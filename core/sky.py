"""Sky and background models.

Six modes, because "the sky" meant very different things to the packages this
engine is imitating. Bryce built one out of a gradient, a haze layer and fractal
clouds; Infini-D gave you two colours and a blend; LightWave users wrapped a
photograph around the scene. All of them are here, and none of them is the
Blender node graph, which remains available as its own mode.

bpy-free, like everything else under core/.
"""

import numpy as np

from . import mathx as M
from .patterns import fbm, hash3 as _hash3, turbulence, value_noise as _value_noise

MODES = ('NODES', 'SOLID', 'GRADIENT', 'BRYCE', 'PHYSICAL', 'HDRI')


def _rotate_z(d, angle):
    if abs(angle) < 1e-6:
        return d
    c, s = np.cos(angle), np.sin(angle)
    out = np.empty_like(d)
    out[:, 0] = d[:, 0] * c - d[:, 1] * s
    out[:, 1] = d[:, 0] * s + d[:, 1] * c
    out[:, 2] = d[:, 2]
    return out


def _sun_vector(elevation, rotation):
    return np.array([np.cos(elevation) * np.cos(rotation),
                     np.cos(elevation) * np.sin(rotation),
                     np.sin(elevation)], np.float32)


def _blend(t, mode):
    t = np.clip(t, 0.0, 1.0)
    if mode == 'SMOOTH':
        return t * t * (3.0 - 2.0 * t)
    if mode == 'SHARP':
        return t * t
    if mode == 'EASE':
        return np.sqrt(t)
    return t


# ------------------------------------------------------------------- noise





def solid(world, dirs):
    n = dirs.shape[0]
    return np.broadcast_to(np.asarray(world.color, np.float32)[None, :],
                           (n, 3)).copy()


def gradient(world, dirs):
    """Horizon-to-zenith blend with an optional separate ground colour."""
    up = np.clip(dirs[:, 2], -1.0, 1.0)
    hor = np.asarray(world.horizon, np.float32)[None, :]
    zen = np.asarray(world.zenith, np.float32)[None, :]
    gnd = np.asarray(world.ground_color, np.float32)[None, :]
    height = float(world.horizon_height)
    falloff = max(float(world.gradient_falloff), 0.01)

    above = np.clip((up - height) / max(1.0 - height, 1e-3), 0.0, 1.0)
    t = _blend(np.power(above, falloff), world.blend_mode)[:, None]
    sky = hor + (zen - hor) * t

    if world.show_ground:
        below = np.clip((height - up) / max(1.0 + height, 1e-3), 0.0, 1.0)
        b = _blend(np.power(below, falloff), world.blend_mode)[:, None]
        sky = np.where(up[:, None] < height, hor + (gnd - hor) * b, sky)
    return sky.astype(np.float32)



def _dome_project(dirs, altitude, scale, squash=1.0):
    """Project view rays onto a horizontal plane at `altitude`.

    Rays near the horizon hit the plane a very long way off, which is what
    compresses the cloud deck toward the horizon exactly as it does in life.
    """
    up = np.maximum(dirs[:, 2], 1e-3)
    d = altitude / up
    d = np.minimum(d, altitude * 60.0)          # stop the horizon going singular
    x = dirs[:, 0] * d / max(scale, 1e-3)
    y = dirs[:, 1] * d / max(scale, 1e-3) * squash
    return x.astype(np.float32), y.astype(np.float32)


def _cloud_layer(dirs, altitude, scale, coverage, sharpness, octaves, seed,
                 kind='CUMULUS', thickness=0.3, squash=1.0, drift=(0.0, 0.0)):
    """Coverage mask for one cloud deck. Returns (alpha, bulk).

    `bulk` is a second sample taken further along the ray, used to shade the
    underside so the deck reads as having depth rather than being a decal.
    """
    x, y = _dome_project(dirs, altitude, scale, squash)
    # wind moves the deck across the sky over time
    x = x + np.float32(drift[0] / max(scale, 1e-3))
    y = y + np.float32(drift[1] / max(scale, 1e-3))
    z = np.full(x.shape[0], seed * 5.31, np.float32)
    p = np.stack([x, y, z], axis=1)
    if kind == 'STRATUS':
        # stretched, wispy, lower contrast
        p2 = np.stack([x * 0.35, y * 2.4, z], axis=1)
        f = fbm(p2, octaves=octaves, lacunarity=2.3, gain=0.55)
    else:
        f = turbulence(p, octaves=octaves, lacunarity=2.0, gain=0.5)
        f = 1.0 - f                              # cusps become the bright tops
    cov = 1.0 - float(np.clip(coverage, 0.0, 1.0))
    a = np.clip((f - cov) / max(1.0 - cov, 1e-3), 0.0, 1.0)
    a = np.power(a, max(float(sharpness), 0.01))

    off = max(float(thickness), 0.0) * 0.6
    if off > 1e-4:
        p_lo = p + np.array([off, off * 0.5, 0.0], np.float32)
        f_lo = (1.0 - turbulence(p_lo, octaves=max(octaves - 1, 1))
                if kind != 'STRATUS' else fbm(p_lo, octaves=max(octaves - 1, 1)))
        bulk = np.clip((f_lo - cov) / max(1.0 - cov, 1e-3), 0.0, 1.0)
    else:
        bulk = a
    # fade the deck out at the horizon, where the projection stops meaning much
    a = a * np.clip(dirs[:, 2] * 12.0, 0.0, 1.0)
    return a.astype(np.float32), bulk.astype(np.float32)


def cloud_cover_at(world, xy, time=0.0):
    """Cumulus coverage directly above a point on the ground.

    Bryce could cast its cloud deck onto the terrain below it. Sampling the same
    noise the deck is drawn from, at the point the shadow ray passes through,
    keeps the two in step -- a shadow always lands under a cloud rather than
    near one.
    """
    if not world.clouds:
        return np.zeros(xy.shape[0], np.float32)
    scale = max(float(world.cloud_scale), 1e-3)
    drift = float(world.cloud_wind) * float(time)
    ang = float(world.cloud_wind_angle)
    ox, oy = np.cos(ang) * drift, np.sin(ang) * drift
    p = np.stack([(xy[:, 0] + ox) / scale, (xy[:, 1] + oy) / scale,
                  np.full(xy.shape[0], float(world.cloud_seed) * 5.31,
                          np.float32)], axis=1)
    f = 1.0 - turbulence(p, octaves=int(world.cloud_detail))
    cov = 1.0 - float(np.clip(world.cloud_cover, 0.0, 1.0))
    a = np.clip((f - cov) / max(1.0 - cov, 1e-3), 0.0, 1.0)
    return np.power(a, max(float(world.cloud_softness), 0.01)).astype(np.float32)


def _moon(dirs, world, sun_dir):
    """A disc with a terminator, so it shows a phase."""
    ang = np.arccos(np.clip(dirs @ sun_dir, -1.0, 1.0))
    size = max(float(world.moon_size), 1e-4)
    disc = ang < size
    out = np.zeros((dirs.shape[0], 3), np.float32)
    if not disc.any():
        return out
    col = np.asarray(world.moon_color, np.float32)
    # position across the disc, along the axis the terminator sweeps
    up = np.array([0.0, 0.0, 1.0], np.float32)
    right = M.normalize(np.cross(up, sun_dir))
    if not np.isfinite(right).all() or float(np.dot(right, right)) < 1e-8:
        right = np.array([1.0, 0.0, 0.0], np.float32)
    across = (dirs[disc] @ right) / size
    phase = float(np.clip(world.moon_phase, 0.0, 1.0))
    # terminator position: -1 at new, +1 at the other new, 0 at full
    edge = np.cos(phase * 2.0 * np.pi)
    lit = np.clip((across - edge) * 6.0, 0.0, 1.0) if phase <= 0.5 else \
        np.clip((edge - across) * 6.0, 0.0, 1.0)
    shine = float(world.moon_earthshine)
    out[disc] = col[None, :] * (lit * (1.0 - shine) + shine)[:, None] * \
        float(world.sun_intensity)
    return out


def _rainbow(dirs, sun, intensity, radius, width, secondary):
    """A bow at `radius` from the antisolar point, red outermost.

    Bryce had a rainbow toggle and it is one of the most recognisable things it
    could put in a sky, so it is here with the real geometry: primary bow with
    red outside, dimmer secondary with the order reversed.
    """
    if intensity <= 0.0:
        return 0.0
    anti = -sun
    ang = np.degrees(np.arccos(np.clip(dirs @ anti, -1.0, 1.0)))
    out = np.zeros((dirs.shape[0], 3), np.float32)
    # wavelength ramp across the band, violet inside -> red outside
    spectrum = np.array([[0.55, 0.0, 0.75], [0.0, 0.3, 0.95], [0.0, 0.85, 0.4],
                         [0.95, 0.95, 0.0], [1.0, 0.55, 0.0], [1.0, 0.1, 0.1]],
                        np.float32)

    def band(centre, w, flip, gain):
        t = (ang - (centre - w * 0.5)) / max(w, 1e-3)
        inside = (t >= 0.0) & (t <= 1.0)
        if not inside.any():
            return
        tt = np.clip(t[inside], 0.0, 1.0)
        if flip:
            tt = 1.0 - tt
        idx = tt * (len(spectrum) - 1)
        i0 = np.floor(idx).astype(np.int32)
        i1 = np.minimum(i0 + 1, len(spectrum) - 1)
        fr = (idx - i0)[:, None]
        col = spectrum[i0] * (1.0 - fr) + spectrum[i1] * fr
        falloff = np.sin(np.clip(t[inside], 0.0, 1.0) * np.pi)[:, None]
        out[inside] += col * falloff * gain

    band(radius, width, False, intensity)
    if secondary > 0.0:
        band(radius * 1.22, width * 1.8, True, intensity * secondary * 0.45)
    return out


def _stars(dirs, density, brightness, seed):
    if brightness <= 0.0:
        return 0.0
    p = dirs * (140.0 + density * 260.0)
    cell = np.floor(p)
    h = _hash3(cell[:, 0].astype(np.int64), cell[:, 1].astype(np.int64),
               cell[:, 2].astype(np.int64) + int(seed))
    thresh = 1.0 - np.clip(density, 0.0, 1.0) * 0.06
    hit = h > thresh
    mag = np.zeros(dirs.shape[0], np.float32)
    if hit.any():
        frac = (h[hit] - thresh) / max(1.0 - thresh, 1e-4)
        mag[hit] = frac * brightness
    mag = mag * np.clip(dirs[:, 2] * 4.0, 0.0, 1.0)
    tint = np.stack([mag, mag * 0.97, mag * 0.9], axis=1)
    return tint.astype(np.float32)


def bryce(world, dirs):
    """Bryce's Sky & Fog, layer for layer.

    Bryce did not model the atmosphere -- it stacked artistic layers in the Sky
    Lab, and that stack is what makes a Bryce sky recognisable at a glance:

        sky dome gradient
          + sun corona (a wide glow, not a physical scattering term)
          + haze thickening toward the horizon and taking the sun's colour
          + stratus deck (high, wispy)
          + cumulus deck (low, billowy, lit from the sun side)
          + ground-hugging fog
          + optional rainbow at the antisolar point
          + optional stars

    Each layer is independently controllable for the same reason it was there.
    """
    n = dirs.shape[0]
    up = np.clip(dirs[:, 2], -1.0, 1.0)
    sun = _sun_vector(float(world.sun_elevation), float(world.sun_rotation))
    cos_sun = np.clip(dirs @ sun, -1.0, 1.0)
    sun_c = np.asarray(world.sun_color, np.float32)

    # ---- sky dome gradient
    hor = np.asarray(world.horizon, np.float32)[None, :]
    zen = np.asarray(world.zenith, np.float32)[None, :]
    t = np.power(np.clip(up, 0.0, 1.0), max(float(world.gradient_falloff), 0.01))
    if world.use_sky_mid:
        # three stops rather than two, as Bryce's dome gradient allowed
        mid = np.asarray(world.sky_mid, np.float32)[None, :]
        m = float(np.clip(world.sky_mid_height, 0.01, 0.99))
        lower = np.clip(t / m, 0.0, 1.0)[:, None]
        upper = np.clip((t - m) / max(1.0 - m, 1e-3), 0.0, 1.0)[:, None]
        col = np.where(t[:, None] < m, hor + (mid - hor) * lower,
                       mid + (zen - mid) * upper)
    else:
        col = hor + (zen - hor) * t[:, None]

    # ---- sun corona: a broad glow plus a tight core, as Bryce's sun did
    glow = float(world.sun_glow)
    inten = float(world.sun_intensity)
    if glow > 0.0 and inten > 0.0:
        tight = np.power(np.clip(cos_sun, 0.0, 1.0),
                         max(4.0, 400.0 * (1.0 - glow) + 4.0))
        broad = np.power(np.clip(cos_sun, 0.0, 1.0),
                         max(1.5, 24.0 * (1.0 - glow) + 1.5))
        corona = tight * 0.75 + broad * 0.35 * float(world.sun_corona)
        col = col + sun_c[None, :] * (corona * glow * inten)[:, None]

    # ---- haze: thickens toward the horizon, and warms toward the sun
    hz = float(world.haze_density)
    if hz > 0.0:
        haze_c = np.asarray(world.haze_color, np.float32)[None, :]
        band = np.exp(-np.maximum(up, 0.0) / max(float(world.haze_height), 1e-3))
        warm = np.power(np.clip(cos_sun, 0.0, 1.0), 3.0) * float(world.haze_sun_tint)
        hc = haze_c + (sun_c[None, :] - haze_c) * warm[:, None]
        # Bryce let haze take the sky's own colour rather than its swatch
        blend = float(np.clip(world.haze_blend_sky, 0.0, 1.0))
        hc = hc * (1.0 - blend) + col * blend
        col = col + (hc - col) * np.clip(band * hz, 0.0, 1.0)[:, None]

    # a proper exponential atmosphere on top of the haze band
    ad = float(world.atmosphere_density)
    if ad > 0.0:
        ac = np.asarray(world.atmosphere_color, np.float32)[None, :]
        depth = 1.0 / np.maximum(np.abs(up) + 0.05, 0.05)
        k = 1.0 - np.exp(-depth * ad * max(float(world.atmosphere_falloff), 0.01))
        col = col + (ac - col) * np.clip(k, 0.0, 1.0)[:, None]

    # ---- stratus deck (high and thin, drawn first so cumulus sit in front)
    if world.stratus:
        a, _bulk = _cloud_layer(
            dirs, float(world.stratus_altitude), float(world.stratus_scale),
            float(world.stratus_amount), float(world.stratus_sharpness),
            int(world.stratus_detail), int(world.cloud_seed) + 7,
            kind='STRATUS', thickness=0.0, squash=float(world.stratus_squash),
            drift=(np.cos(float(world.cloud_wind_angle)) * float(world.cloud_wind)
                   * float(getattr(world, '_time', 0.0)) * 0.6,
                   np.sin(float(world.cloud_wind_angle)) * float(world.cloud_wind)
                   * float(getattr(world, '_time', 0.0)) * 0.6))
        sc = np.asarray(world.stratus_color, np.float32)[None, :]
        lit = np.clip(cos_sun * 0.5 + 0.5, 0.0, 1.0)[:, None]
        col = col + (sc * (0.6 + 0.4 * lit) - col) * \
            (a * float(world.stratus_density))[:, None]

    # ---- cumulus deck
    if world.clouds:
        d = float(world.cloud_wind) * float(getattr(world, '_time', 0.0))
        ang = float(world.cloud_wind_angle)
        drift = (np.cos(ang) * d, np.sin(ang) * d)
        a, bulk = _cloud_layer(
            dirs, float(world.cloud_height), float(world.cloud_scale),
            float(world.cloud_cover), float(world.cloud_softness),
            int(world.cloud_detail), int(world.cloud_seed),
            kind='CUMULUS', thickness=float(world.cloud_thickness), drift=drift)
        top = np.asarray(world.cloud_color, np.float32)[None, :]
        base = np.asarray(world.cloud_shadow, np.float32)[None, :]
        # self-shadowing: where the second sample is thicker, the underside is
        # in shadow. This is what stops the deck looking like flat cut-outs.
        shade = np.clip(bulk - a * 0.5, 0.0, 1.0)[:, None]
        lit = np.clip(cos_sun * 0.5 + 0.5, 0.0, 1.0)[:, None]
        amb = float(np.clip(world.cloud_ambience, 0.0, 1.0))
        body = base + (top - base) * np.clip(
            lit * (1.0 - shade * 0.9) * (1.0 - amb) + amb, 0.0, 1.0)
        # a rim of sun colour on the sunward edge
        rim = np.power(np.clip(cos_sun, 0.0, 1.0), 8.0)[:, None] * \
            float(world.cloud_rim)
        body = body + sun_c[None, :] * rim * a[:, None]
        col = col + (body - col) * (a * float(world.cloud_density))[:, None]

    # ---- ground-hugging fog, which in Bryce is separate from haze
    fg = float(world.fog_density)
    if fg > 0.0:
        fog_c = np.asarray(world.fog_color, np.float32)[None, :]
        band = np.exp(-np.maximum(up, 0.0) / max(float(world.fog_height), 1e-3))
        col = col + (fog_c - col) * np.clip(band * fg, 0.0, 1.0)[:, None]

    # ---- rainbow and stars
    if world.rainbow:
        col = col + _rainbow(dirs, sun, float(world.rainbow_intensity),
                             float(world.rainbow_radius),
                             float(world.rainbow_width),
                             float(world.rainbow_secondary))
    if world.stars:
        col = col + _stars(dirs, float(world.star_density),
                           float(world.star_brightness), int(world.cloud_seed))

    # ---- the sun or the moon, drawn last so nothing dims it
    if world.celestial == 'MOON':
        col = col + _moon(dirs, world, sun)
    elif world.sun_disc:
        disc = np.arccos(cos_sun) < max(float(world.sun_size), 1e-4)
        if disc.any():
            col[disc] = sun_c * inten * 6.0

    if world.show_ground:
        gnd = np.asarray(world.ground_color, np.float32)[None, :]
        below = np.clip(-up * 8.0, 0.0, 1.0)[:, None]
        col = col * (1.0 - below) + gnd * below
    return np.clip(np.nan_to_num(col, nan=0.0, posinf=1.0, neginf=0.0),
                   0.0, None).astype(np.float32)


def physical(world, dirs):
    """Preetham analytic daylight, driven by the same sun controls."""
    from .nodeeval import _preetham_coeffs, _perez, _xyY_to_rgb
    elev = float(world.sun_elevation)
    T = float(np.clip(world.turbidity, 1.0, 10.0))
    sun = _sun_vector(elev, float(world.sun_rotation))
    up = np.clip(dirs[:, 2], -1.0, 1.0)
    cos_theta = np.maximum(up, 0.0)
    cos_gamma = np.clip(dirs @ sun, -1.0, 1.0)
    gamma = np.arccos(cos_gamma)

    theta_s = max(np.pi * 0.5 - elev, 0.0)
    chi = (4.0 / 9.0 - T / 120.0) * (np.pi - 2.0 * theta_s)
    zenith_Y = max((4.0453 * T - 4.9710) * np.tan(chi) - 0.2155 * T + 2.4192,
                   0.0) * 0.06
    t2, t3 = theta_s ** 2, theta_s ** 3
    T2 = T * T
    zx = ((0.00166 * t3 - 0.00375 * t2 + 0.00209 * theta_s) * T2 +
          (-0.02903 * t3 + 0.06377 * t2 - 0.03202 * theta_s + 0.00394) * T +
          (0.11693 * t3 - 0.21196 * t2 + 0.06052 * theta_s + 0.25886))
    zy = ((0.00275 * t3 - 0.00610 * t2 + 0.00317 * theta_s) * T2 +
          (-0.04214 * t3 + 0.08970 * t2 - 0.04153 * theta_s + 0.00516) * T +
          (0.15346 * t3 - 0.26756 * t2 + 0.06670 * theta_s + 0.26688))
    cs = np.array([max(np.cos(theta_s), 0.01)], np.float32)
    ts = np.array([theta_s], np.float32)
    den = [_perez(cs, ts, np.cos(ts), _preetham_coeffs(ch, T))[0]
           for ch in ('Y', 'x', 'y')]
    Y = zenith_Y * _perez(cos_theta, gamma, cos_gamma,
                          _preetham_coeffs('Y', T)) / max(den[0], 1e-4)
    x = zx * _perez(cos_theta, gamma, cos_gamma,
                    _preetham_coeffs('x', T)) / max(den[1], 1e-4)
    y = zy * _perez(cos_theta, gamma, cos_gamma,
                    _preetham_coeffs('y', T)) / max(den[2], 1e-4)
    col = _xyY_to_rgb(x, y, Y)
    size = max(float(world.sun_size), 1e-4)
    if world.sun_disc:
        disc = gamma < size
        if disc.any():
            col[disc] += np.asarray(world.sun_color, np.float32) * \
                float(world.sun_intensity) * 8.0
    if world.show_ground:
        gnd = np.asarray(world.ground_color, np.float32)[None, :]
        below = np.clip(-up * 8.0, 0.0, 1.0)[:, None]
        col = col * (1.0 - below) + gnd * below
    return np.nan_to_num(col, nan=0.0, posinf=1.0, neginf=0.0)


def hdri(world, dirs, textures):
    """An image wrapped around the scene, equirectangular or mirror ball."""
    from .texture import env_equirect_uv, env_sphere_uv
    n = dirs.shape[0]
    tex = None
    key = getattr(world.env_image, 'name', None) if world.env_image else None
    if key:
        tex = textures.get(key)
    if tex is None:
        tex = textures.get('world_env')
    if tex is None:
        return solid(world, dirs)
    if world.env_mapping == 'MIRRORBALL':
        u, v = env_sphere_uv(dirs)
    else:
        u, v = env_equirect_uv(dirs)
    filt = 'NEAREST' if world.env_filter == 'NEAREST' else 'BILINEAR'
    col = tex.sample(u, v, filt=filt, wrap='EXTEND')[:, :3]
    tint = np.asarray(world.env_tint, np.float32)[None, :]
    return (col * tint).astype(np.float32)


# ------------------------------------------------------------ infinite ground


def _wave_normal(p, world, time):
    """A few crossed sine trains. Enough for water at a distance, which is all
    an infinite plane is ever seen at."""
    amp = float(world.ocean_choppiness)
    scale = max(float(world.ground_scale), 1e-3)
    t = float(time) * float(world.ocean_speed)
    x, y = p[:, 0] / scale, p[:, 1] / scale
    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    freq, weight = 1.0, 1.0
    for i in range(4):
        ang = 0.7 + i * 1.9
        kx, ky = np.cos(ang), np.sin(ang)
        phase = (x * kx + y * ky) * freq + t * (1.0 + i * 0.3)
        c = np.cos(phase) * weight
        dx += c * kx * freq
        dy += c * ky * freq
        freq *= 1.9
        weight *= 0.55
    n = np.stack([-dx * amp, -dy * amp, np.ones_like(dx)], axis=1)
    return M.normalize(n)


def ground_plane(world, dirs, sky_col, eye, time=0.0, textures=None):
    """Shade an infinite plane where the view ray dips below it.

    An infinite floor cannot be geometry, so it is intersected analytically in
    the background pass -- which is exactly how POV-Ray and Bryce provided one.
    Distance haze does the rest: the plane fades into the horizon colour, and
    without that it reads as a flat sheet rather than as ground going away.
    """
    dz = dirs[:, 2]
    below = dz < -1e-5
    if not below.any():
        return sky_col
    eye = np.asarray(eye, np.float32)
    t = np.full(dirs.shape[0], -1.0, np.float32)
    t[below] = (float(world.ground_height) - eye[2]) / dz[below]
    hit = below & (t > 0.0)
    if not hit.any():
        return sky_col

    p = eye[None, :] + dirs[hit] * t[hit, None]
    dist = t[hit]
    col = np.broadcast_to(np.asarray(world.ground_color, np.float32)[None, :],
                          (p.shape[0], 3)).copy()
    scale = max(float(world.ground_scale), 1e-3)

    if world.ground_mode == 'CHECKER':
        cell = np.floor(p[:, 0] / scale) + np.floor(p[:, 1] / scale)
        alt = np.asarray(world.ground_color2, np.float32)[None, :]
        col = np.where((cell % 2 == 0)[:, None], col, alt)
    elif world.ground_mode == 'NOISE':
        from .patterns import fbm
        f = fbm(np.stack([p[:, 0] / scale, p[:, 1] / scale,
                          np.zeros(p.shape[0], np.float32)], 1), octaves=5)
        alt = np.asarray(world.ground_color2, np.float32)[None, :]
        col = col + (alt - col) * f[:, None]

    if world.ground_mode == 'OCEAN':
        n = _wave_normal(p, world, time)
        v = -dirs[hit]
        r = M.reflect(-v, n)
        r[:, 2] = np.abs(r[:, 2])                 # never reflect the sea floor
        refl = evaluate(world, r, None, strength=False)
        if refl is None:
            refl = np.broadcast_to(np.asarray(world.horizon, np.float32)[None, :],
                                   (p.shape[0], 3)).copy()
        # Fresnel: water is a mirror at glancing angles and glass overhead
        facing = np.clip(M.dot(n, v), 0.0, 1.0)
        f = 0.02 + 0.98 * np.power(1.0 - facing, 5.0)
        deep = np.asarray(world.ground_color, np.float32)[None, :]
        col = deep * (1.0 - f)[:, None] + refl * f[:, None]

    if float(world.cloud_shadows) > 0.0 and world.clouds:
        # trace up to the cloud deck along the light direction and see what is
        # in the way, which is how the shadow lands under the cloud
        sun = _sun_vector(float(world.sun_elevation), float(world.sun_rotation))
        if sun[2] > 0.05:
            up_t = (float(world.cloud_height) - float(world.ground_height)) / sun[2]
            hit_xy = p[:, :2] + sun[None, :2] * up_t
            cover = cloud_cover_at(world, hit_xy, time)
            k = np.clip(cover * float(world.cloud_shadows), 0.0, 1.0)[:, None]
            shade_col = np.asarray(world.cloud_shadow, np.float32)[None, :]
            col = col * (1.0 - k) + col * shade_col * k

    # haze with distance, which is what makes it read as receding ground
    fade = float(world.ground_fade)
    if fade > 0.0:
        k = np.clip(dist / max(fade, 1e-3), 0.0, 1.0)[:, None]
        col = col * (1.0 - k) + sky_col[hit] * k

    out = sky_col.copy()
    out[hit] = col
    return out


# ---------------------------------------------------------------- dispatch


def evaluate(world, dirs, textures=None, strength=True, eye=None,
             time=0.0):
    try:
        world._time = time
    except Exception:                                           # noqa: BLE001
        pass
    """Background radiance along `dirs` for the world's chosen mode."""
    dirs = M.normalize(np.asarray(dirs, np.float32))
    rot = float(getattr(world, 'rotation', 0.0))
    if abs(rot) > 1e-6:
        dirs = _rotate_z(dirs, -rot)
    mode = getattr(world, 'mode', 'NODES')
    if mode == 'SOLID':
        col = solid(world, dirs)
    elif mode == 'GRADIENT':
        col = gradient(world, dirs)
    elif mode == 'BRYCE':
        col = bryce(world, dirs)
    elif mode == 'PHYSICAL':
        col = physical(world, dirs)
    elif mode == 'HDRI':
        col = hdri(world, dirs, textures or {})
    else:
        return None                      # caller falls back to the node graph
    if strength:
        col = col * float(getattr(world, 'strength', 1.0))
    if getattr(world, 'ground_plane', False) and eye is not None:
        col = ground_plane(world, dirs, col.astype(np.float32), eye, time,
                           textures)
    return col.astype(np.float32)
