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

MODES = ('NODES', 'SOLID', 'GRADIENT', 'BANDS', 'STARFIELD', 'BRYCE',
         'PHYSICAL', 'HDRI')


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



def bands(world, dirs):
    """The gradient, quantised into a fixed number of flat steps.

    This is not a stylised gradient -- it is what a gradient *was* on a machine
    with 256 colours and most of them already spent on the scene. The sky got a
    handful of entries, so it arrived as visible bands, and the bands moved
    when the camera did. Reproducing that here rather than leaving it to the
    palette stage matters, because the palette stage is quantising the whole
    frame at once: a sky that was already stepped keeps its steps whatever the
    rest of the image spends its colours on.

    Steps are cut in the blend parameter rather than in the output colour, so
    the band edges land at the same heights whichever two colours are set.
    """
    up = np.clip(dirs[:, 2], -1.0, 1.0)
    hor = np.asarray(world.horizon, np.float32)[None, :]
    zen = np.asarray(world.zenith, np.float32)[None, :]
    gnd = np.asarray(world.ground_color, np.float32)[None, :]
    height = float(world.horizon_height)
    falloff = max(float(world.gradient_falloff), 0.01)
    steps = max(int(getattr(world, 'band_count', 8)), 1)
    soft = float(np.clip(getattr(world, 'band_softness', 0.0), 0.0, 1.0))

    def quantise(t):
        # `steps` bands means `steps` colours *including both ends*, so the
        # divisor is one less than the count. Dividing by the count instead
        # leaves a band at the zenith that is infinitesimally thin -- it only
        # ever gets hit exactly at t = 1 -- and every palette then carries one
        # entry it never spends.
        if steps == 1:
            return np.zeros_like(t)
        s = np.minimum(np.floor(t * steps), steps - 1)
        if soft > 1e-4:
            frac = t * steps - np.floor(t * steps)
            e = np.clip((frac - (1.0 - soft)) / max(soft, 1e-4), 0.0, 1.0)
            s = np.minimum(s + e * e * (3.0 - 2.0 * e), steps - 1)
        return np.clip(s / (steps - 1), 0.0, 1.0)

    above = np.clip((up - height) / max(1.0 - height, 1e-3), 0.0, 1.0)
    t = quantise(_blend(np.power(above, falloff), world.blend_mode))[:, None]
    sky = hor + (zen - hor) * t

    if world.show_ground:
        below = np.clip((height - up) / max(1.0 + height, 1e-3), 0.0, 1.0)
        b = quantise(_blend(np.power(below, falloff), world.blend_mode))[:, None]
        sky = np.where(up[:, None] < height, hor + (gnd - hor) * b, sky)
    return sky.astype(np.float32)


def starfield(world, dirs):
    """Nothing but space: a flat backdrop, stars, and optional nebula.

    Every package shipped a space scene and none of them lit one with a sky
    model. The background was a colour, the stars were points scattered on it,
    and if there was a nebula it was noise through a colour map. There is no
    horizon here at all -- stars go all the way round, which the Bryce star
    layer deliberately does not do because it sits under a sky dome.
    """
    n = dirs.shape[0]
    col = np.broadcast_to(np.asarray(world.color, np.float32)[None, :],
                          (n, 3)).copy()

    amount = float(getattr(world, 'nebula', 0.0))
    if amount > 1e-4:
        scale = max(float(getattr(world, 'nebula_scale', 2.0)), 1e-3)
        p = dirs * scale
        v = turbulence(p, octaves=int(getattr(world, 'nebula_detail', 5)))
        v = np.clip((v - 0.35) * 2.2, 0.0, 1.0) ** 1.6
        neb = np.asarray(getattr(world, 'nebula_color', (0.35, 0.15, 0.55)),
                         np.float32)[None, :]
        col = col + neb * (v * amount)[:, None]

    bright = float(getattr(world, 'star_brightness', 0.8))
    if bright > 1e-4:
        from .patterns import starfield as _star_pattern
        size = float(getattr(world, 'star_size', 0.35))
        scale = 60.0 + float(getattr(world, 'star_density', 0.5)) * 340.0
        mag = _star_pattern(dirs * scale,
                            float(getattr(world, 'star_density', 0.5)),
                            size, float(getattr(world, 'star_twinkle', 0.0)),
                            float(getattr(world, '_time', 0.0)))
        # stars are not all white: hot ones read blue, cool ones amber, and a
        # single hash per cell is enough to say which
        tint = _hash3f_dirs(dirs * scale)
        warm = np.array([1.0, 0.86, 0.70], np.float32)
        cool = np.array([0.74, 0.84, 1.0], np.float32)
        star_col = cool[None, :] + (warm[None, :] - cool[None, :]) * tint[:, None]
        col = col + star_col * (mag * bright)[:, None]
    return col.astype(np.float32)


def _hash3f_dirs(p):
    c = np.floor(p)
    return _hash3(c[:, 0].astype(np.int64), c[:, 1].astype(np.int64),
                  c[:, 2].astype(np.int64) + 613)


def _dome_project(dirs, altitude, scale, squash=1.0, spherical=True):
    """Project view rays onto a horizontal plane at `altitude`.

    Rays near the horizon hit the plane a very long way off, which is what
    compresses the cloud deck toward the horizon exactly as it does in life.

    `spherical` is Bryce's Spherical Clouds switch. With it off you get the
    plain plane projection above, and clouds smear into streaks at the horizon.
    With it on the distance is rolled off toward a dome instead, so a cloud a
    long way off stays the shape it started as -- which is the whole reason the
    switch existed.
    """
    up = np.maximum(dirs[:, 2], 1e-3)
    d = altitude / up
    cap = altitude * 60.0
    if spherical:
        # a smooth approach to the cap rather than a hard clamp: the deck keeps
        # shrinking toward the horizon but never runs away to infinity
        d = cap * d / (cap + d)
    else:
        d = np.minimum(d, cap)      # stop the horizon going singular
    x = dirs[:, 0] * d / max(scale, 1e-3)
    y = dirs[:, 1] * d / max(scale, 1e-3) * squash
    return x.astype(np.float32), y.astype(np.float32)


def _cloud_layer(dirs, altitude, scale, coverage, sharpness, octaves, seed,
                 kind='CUMULUS', thickness=0.3, squash=1.0, drift=(0.0, 0.0),
                 frequency=1.0, amplitude=1.0, turb=1.0, spherical=True,
                 parallax=(0.0, 0.0)):
    """Coverage mask for one cloud deck. Returns (alpha, bulk).

    `bulk` is a second sample taken further along the ray, used to shade the
    underside so the deck reads as having depth rather than being a decal.

    `frequency`, `amplitude` and `turb` are Bryce's own three cloud controls,
    under Bryce's own names. Frequency is how tight the pattern is, amplitude
    is how far it swings either side of the cover threshold -- which is what
    turns a soft overcast into separated billows without changing how much sky
    is covered -- and turbulence is how hard the noise is folded.

    `spherical` is Bryce's Spherical Clouds switch. Off, the deck is projected
    onto a flat plane and the clouds stretch toward the horizon; on, the
    projection is pulled back toward the dome so they stay puffy out to the
    edge, which is what the option was for.
    """
    x, y = _dome_project(dirs, altitude, scale, squash,
                         spherical=bool(spherical))
    # Parallax from the camera's own position, weighted by how steeply the ray
    # looks. A cloud overhead is at the deck's height and swings past you as
    # you move; a cloud on the horizon is effectively at infinity and does not
    # move at all. Adding the offset flat -- which is what the first cut did --
    # slides the whole sky including the horizon, and that reads as the clouds
    # racing whenever the camera so much as orbits.
    if parallax[0] or parallax[1]:
        w = np.clip(dirs[:, 2], 0.0, 1.0)
        x = x + np.float32(parallax[0] / max(scale, 1e-3)) * w
        y = y + np.float32(parallax[1] / max(scale, 1e-3)) * w
    x = x * max(float(frequency), 1e-3)
    y = y * max(float(frequency), 1e-3)
    # wind moves the deck across the sky over time
    x = x + np.float32(drift[0] / max(scale, 1e-3))
    y = y + np.float32(drift[1] / max(scale, 1e-3))
    z = np.full(x.shape[0], seed * 5.31, np.float32)
    p = np.stack([x, y, z], axis=1)
    gain = 0.5 * max(float(turb), 0.05)
    if kind == 'STRATUS':
        # stretched, wispy, lower contrast
        p2 = np.stack([x * 0.35, y * 2.4, z], axis=1)
        f = fbm(p2, octaves=octaves, lacunarity=2.3, gain=min(gain + 0.05, 0.95))
    else:
        f = turbulence(p, octaves=octaves, lacunarity=2.0,
                       gain=min(gain, 0.95))
        f = 1.0 - f                              # cusps become the bright tops
    cov = 1.0 - float(np.clip(coverage, 0.0, 1.0))
    # amplitude swings the field about the threshold rather than about zero, so
    # raising it separates the billows without changing how much sky is covered
    amp = max(float(amplitude), 0.0)
    if abs(amp - 1.0) > 1e-4:
        f = np.clip(cov + (f - cov) * amp, 0.0, 1.0)
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
    # Softness is how hard the terminator lands. Bryce put it next to the
    # phase, because a hard one reads as a cut-out and a soft one as a sphere.
    soft = float(np.clip(getattr(world, 'moon_softness', 0.05), 0.0, 1.0))
    k = 1.0 / max(soft, 1e-3) * 0.3
    lit = np.clip((across - edge) * k, 0.0, 1.0) if phase <= 0.5 else \
        np.clip((edge - across) * k, 0.0, 1.0)
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


def _comets(dirs, sun, world, intensity, count, seed=0, time=0.0):
    """Streaks across the night sky. Bryce put them in the Celestial tab.

    Each comet is a bright head with a tail falling away behind it -- which is
    what Bryce drew, and it is the reason its night skies never looked like a
    plain starfield. Here the head also *moves*: it runs around a great circle
    of its own, at Comet Speed, so a frame range shows it crossing the sky.
    Time is the scene's, so it is the same comet in the same place on every
    machine that renders that frame.

    Which way the tail points is not arbitrary either. A comet's ion tail is
    blown directly away from the sun and its dust tail trails its own path, so
    the two disagree and the truth is somewhere between; Tail Direction is
    that mix. The sun vector had been an argument of this function since it
    was written and was never once read -- the tails pointed wherever the
    random number generator sent them.
    """
    if intensity <= 0.0 or count <= 0:
        return 0.0
    rng = np.random.default_rng(int(seed) + 9871)
    speed = float(getattr(world, 'comet_speed', 0.0))
    length0 = max(float(getattr(world, 'comet_length', 0.10)), 1e-3)
    width0 = max(float(getattr(world, 'comet_width', 0.006)), 1e-4)
    anti = float(np.clip(getattr(world, 'comet_tail_sun', 0.6), 0.0, 1.0))
    col_c = np.asarray(getattr(world, 'comet_color', (1.0, 0.96, 0.88)),
                       np.float32)
    phase = float(time) * speed
    out = np.zeros((dirs.shape[0], 3), np.float32)
    for i in range(int(count)):
        # the comet's path is a great circle: two orthonormal vectors span it
        # and the head runs round from one toward the other
        u = rng.normal(size=3).astype(np.float32)
        u /= max(float(np.linalg.norm(u)), 1e-6)
        if u[2] < 0.15:                      # start it above the horizon
            u[2] = abs(u[2]) + 0.2
            u /= max(float(np.linalg.norm(u)), 1e-6)
        w = rng.normal(size=3).astype(np.float32)
        w -= u * float(np.dot(w, u))
        w /= max(float(np.linalg.norm(w)), 1e-6)

        # each one at its own pace, so they do not travel as a flock. They all
        # start where they were drawn standing still, which is above the
        # horizon, so turning the speed up never empties the first frame
        ph = phase * (0.6 + 1.4 * float(rng.random()))
        v = np.cos(ph) * u + np.sin(ph) * w              # the head, now
        v /= max(float(np.linalg.norm(v)), 1e-6)
        motion = -np.sin(ph) * u + np.cos(ph) * w        # where it is going

        # tail: behind its own motion, blended toward straight away from the
        # sun, then flattened back onto the sphere at the head
        t = -motion * (1.0 - anti) - np.asarray(sun, np.float32) * anti
        t -= v * float(np.dot(t, v))
        nt = float(np.linalg.norm(t))
        if nt < 1e-5:                        # tail exactly along the view axis
            t = -motion
            t -= v * float(np.dot(t, v))
            nt = max(float(np.linalg.norm(t)), 1e-6)
        t /= nt

        along = dirs @ t                                 # +ve down the tail
        across = dirs @ np.cross(v, t).astype(np.float32)
        head = dirs @ v                                  # 1 at the head

        # A comet is a compact head with a tail behind it, and the tail is
        # bounded at both ends. The old profile bounded it at the far end
        # only, so the streak ran on *in front of* the head as far as a cos^8
        # falloff allowed -- around forty degrees. That is why they drew as
        # long straight lines rather than as comets.
        length = length0 * (0.55 + 0.9 * float(rng.random()))
        tt = along / length
        # the coma: a small round glow at the head. 2(1-cos) is the chord
        # squared, which is the angle squared for anything this small
        coma_r = width0 * 2.5
        coma = np.exp(-2.0 * np.maximum(1.0 - head, 0.0) /
                      max(coma_r * coma_r, 1e-9))
        # the tail: behind the head only, flaring and fading as it goes
        span = np.clip(tt, 0.0, 1.0)
        width = width0 * (1.0 + 2.5 * span)
        band = np.exp(-(across / width) ** 2)
        live = (tt > 0.0) & (tt < 1.0) & (head > 0.0)
        mag = coma + band * (1.0 - span) ** 2 * live
        out += col_c[None, :] * (mag * intensity)[:, None]
    return out.astype(np.float32)


def _atmos_band(up, density, height, base):
    """Bryce's fog/haze profile: a band that starts at a base height.

    Both fog and haze in the Sky Lab have a Base Height as well as a height,
    and the base is what lets a fog bank sit *above* the camera or start part
    way up a cliff instead of always hugging zero.
    """
    h = max(float(height), 1e-3)
    b = float(base)
    band = np.exp(-np.maximum(up - b, 0.0) / h)
    band = np.where(up < b, 1.0, band)
    return np.clip(band * float(density), 0.0, 1.0)


def bryce(world, dirs, eye=None):
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
    # Link Clouds to View keeps the pattern still as the camera moves, which is
    # what the switch was for -- with it off the deck is nailed to the world
    # and slides past you. Fixed Cloud Plane measures the deck's height from
    # the camera instead of from the ground, so climbing never puts you inside
    # it, which is the other half of the same problem.
    eye = np.zeros(3, np.float32) if eye is None else \
        np.asarray(eye, np.float32).reshape(3)
    cloud_off = (0.0, 0.0) if bool(getattr(world, 'link_clouds_to_view', True)) \
        else (float(eye[0]), float(eye[1]))
    cloud_lift = 0.0 if bool(getattr(world, 'fixed_cloud_plane', True)) \
        else float(eye[2])
    cos_sun = np.clip(dirs @ sun, -1.0, 1.0)
    sun_c = np.asarray(world.sun_color, np.float32)

    # ---- sky dome gradient
    #
    # Bryce's Sky & Fog palette had a Sky Mode. Custom Sky is the three stops
    # as set; Soft Sky derived the horizon from the sun's own colour and left
    # only the dome to the user, which is why every default Bryce sky warmed
    # toward the sun without anybody choosing to make it.
    soft = str(getattr(world, 'sky_mode', 'CUSTOM')) == 'SOFT'
    glow_c = np.asarray(getattr(world, 'sun_glow_color',
                                world.sun_color), np.float32)
    hor = np.asarray(world.horizon, np.float32)[None, :]
    zen = np.asarray(world.zenith, np.float32)[None, :]
    if soft:
        # the horizon takes the glow colour, dimmed, and the mid stop sits
        # halfway between it and the dome
        hor = (hor * 0.35 + glow_c[None, :] * 0.65)
    t = np.power(np.clip(up, 0.0, 1.0), max(float(world.gradient_falloff), 0.01))
    if world.use_sky_mid:
        # three stops rather than two, as Bryce's dome gradient allowed
        mid = np.asarray(world.sky_mid, np.float32)[None, :]
        if soft:
            mid = (hor + zen) * 0.5
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
        # Bryce's Sun Glow Colour is its own swatch, separate from the light's
        col = col + glow_c[None, :] * (corona * glow * inten)[:, None]

    # Bryce's Sky Lab stacks its layers in a fixed order, and the order is
    # half of why a Bryce sky reads as one. Sky dome first, then whatever is
    # *beyond* the atmosphere -- stars, comets, the sun or moon -- then the
    # cloud decks in front of them, and only then the atmosphere itself,
    # because haze and fog sit between the viewer and all of it. Getting this
    # wrong is visible: stars used to shine through the clouds, and clouds at
    # the horizon used to stay crisp while the sky behind them hazed over.

    # ---- beyond the atmosphere
    amount = float(getattr(world, 'nebula', 0.0))
    if amount > 1e-4:
        # the starfield mode's nebula wash, now under the Bryce dome too:
        # a night sky with nebula settings used to silently ignore them
        # (STARFIELD had the term, BRYCE never grew it). Same formula,
        # same colour map, sitting behind the stars.
        scale = max(float(getattr(world, 'nebula_scale', 2.0)), 1e-3)
        v = turbulence(dirs * scale,
                       octaves=int(getattr(world, 'nebula_detail', 5)))
        v = np.clip((v - 0.35) * 2.2, 0.0, 1.0) ** 1.6
        neb = np.asarray(getattr(world, 'nebula_color', (0.35, 0.15, 0.55)),
                         np.float32)[None, :]
        col = col + neb * (v * amount)[:, None]
    if world.stars:
        col = col + _stars(dirs, float(world.star_density),
                           float(world.star_brightness), int(world.cloud_seed))
    if float(getattr(world, 'comets', 0.0)) > 0.0:
        col = col + _comets(dirs, sun, world, float(world.comets),
                            int(getattr(world, 'comet_count', 3)),
                            int(world.cloud_seed),
                            float(getattr(world, '_time', 0.0)))

    # ---- the sun or the moon, in the dome rather than in front of it
    if world.celestial == 'MOON':
        col = col + _moon(dirs, world, sun)
    elif world.sun_disc:
        disc = np.arccos(cos_sun) < max(float(world.sun_size), 1e-4)
        if disc.any():
            col[disc] = sun_c * inten * 6.0


    # ---- the cloud decks, in front of all of that
    cloud_alpha = np.zeros(n, np.float32)
    # stratus first, so cumulus sit in front of it
    if world.stratus:
        a, _bulk = _cloud_layer(
            dirs, max(float(world.stratus_altitude) - cloud_lift, 0.05),
            float(world.stratus_scale),
            float(world.stratus_amount), float(world.stratus_sharpness),
            int(world.stratus_detail), int(world.cloud_seed) + 7,
            kind='STRATUS', thickness=0.0, squash=float(world.stratus_squash),
            frequency=float(getattr(world, 'stratus_frequency', 1.0)),
            amplitude=float(getattr(world, 'stratus_amplitude', 1.0)),
            turb=float(getattr(world, 'cloud_turbulence', 1.0)),
            spherical=bool(getattr(world, 'spherical_clouds', True)),
            drift=(np.cos(float(world.cloud_wind_angle)) * float(world.cloud_wind)
                   * float(getattr(world, '_time', 0.0)) * 0.6,
                   np.sin(float(world.cloud_wind_angle)) * float(world.cloud_wind)
                   * float(getattr(world, '_time', 0.0)) * 0.6),
            parallax=cloud_off)
        sc = np.asarray(world.stratus_color, np.float32)[None, :]
        lit = np.clip(cos_sun * 0.5 + 0.5, 0.0, 1.0)[:, None]
        cloud_alpha = np.maximum(cloud_alpha, a * float(world.stratus_density))
        col = col + (sc * (0.6 + 0.4 * lit) - col) * \
            (a * float(world.stratus_density))[:, None]

    # ---- cumulus deck
    if world.clouds:
        d = float(world.cloud_wind) * float(getattr(world, '_time', 0.0))
        ang = float(world.cloud_wind_angle)
        drift = (np.cos(ang) * d, np.sin(ang) * d)
        a, bulk = _cloud_layer(
            dirs, max(float(world.cloud_height) - cloud_lift, 0.05),
            float(world.cloud_scale),
            float(world.cloud_cover), float(world.cloud_softness),
            int(world.cloud_detail), int(world.cloud_seed),
            kind='CUMULUS', thickness=float(world.cloud_thickness), drift=drift,
            parallax=cloud_off,
            frequency=float(getattr(world, 'cloud_frequency', 1.0)),
            amplitude=float(getattr(world, 'cloud_amplitude', 1.0)),
            turb=float(getattr(world, 'cloud_turbulence', 1.0)),
            spherical=bool(getattr(world, 'spherical_clouds', True)))
        top = np.asarray(world.cloud_color, np.float32)[None, :]
        base = np.asarray(world.cloud_shadow, np.float32)[None, :]
        # self-shadowing: where the second sample is thicker, the underside is
        # in shadow. This is what stops the deck looking like flat cut-outs.
        shade = np.clip(bulk - a * 0.5, 0.0, 1.0)[:, None]
        lit = np.clip(cos_sun * 0.5 + 0.5, 0.0, 1.0)[:, None]
        amb = float(np.clip(world.cloud_ambience, 0.0, 1.0))
        # the Sky & Fog palette's Shadow Colour, at its Shadow Intensity, is
        # what the shaded side of a Bryce cloud is tinted with
        sh_c = np.asarray(getattr(world, 'shadow_color', (0, 0, 0)),
                          np.float32)[None, :]
        sh_i = float(np.clip(getattr(world, 'shadow_intensity', 1.0), 0.0, 1.0))
        base = base + (sh_c - base) * sh_i * 0.5
        body = base + (top - base) * np.clip(
            lit * (1.0 - shade * 0.9) * (1.0 - amb) + amb, 0.0, 1.0)
        # a rim of sun colour on the sunward edge
        rim = np.power(np.clip(cos_sun, 0.0, 1.0), 8.0)[:, None] * \
            float(world.cloud_rim)
        body = body + sun_c[None, :] * rim * a[:, None]
        cloud_alpha = np.maximum(cloud_alpha, a * float(world.cloud_density))
        col = col + (body - col) * (a * float(world.cloud_density))[:, None]


    # ---- Volumetric World: the haze lights up along the rays that reach the
    # sun through a gap in the deck, which is what Bryce's setting bought and
    # what it charged so much render time for. Here it costs the cloud alpha
    # that was computed anyway.
    vol = float(getattr(world, 'volumetric_world', 0.0))
    if vol > 0.0 and inten > 0.0:
        clear = 1.0 - np.clip(cloud_alpha, 0.0, 1.0)
        # a shaft needs something to scatter off, so it is proportional to the
        # haze that is there. Without that the control just floods the frame,
        # which is what a broad lobe and a large gain did on the first attempt.
        scatter = float(world.haze_density) * 0.6 + \
            float(world.atmosphere_density) * 0.2
        shaft = np.power(np.clip(cos_sun, 0.0, 1.0), 24.0) * clear
        col = col + sun_c[None, :] * \
            (shaft * vol * 0.12 * inten * scatter)[:, None]

    # ---- and the atmosphere last, because it is between you and everything
    # ---- haze: thickens toward the horizon, and warms toward the sun
    hz = float(world.haze_density)
    if hz > 0.0:
        haze_c = np.asarray(world.haze_color, np.float32)[None, :]
        band = _atmos_band(up, 1.0, float(world.haze_height),
                           float(getattr(world, 'haze_base_height', 0.0)))
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

    # ---- ground-hugging fog, which in Bryce is separate from haze
    fg = float(world.fog_density)
    if fg > 0.0:
        fog_c = np.asarray(world.fog_color, np.float32)[None, :]
        band = _atmos_band(up, 1.0, float(world.fog_height),
                           float(getattr(world, 'fog_base_height', 0.0)))
        # the Atmosphere tab gives fog the same two blends it gives haze
        warm = np.power(np.clip(cos_sun, 0.0, 1.0), 3.0) * \
            float(getattr(world, 'fog_sun_tint', 0.0))
        fc = fog_c + (sun_c[None, :] - fog_c) * warm[:, None]
        fb = float(np.clip(getattr(world, 'fog_blend_sky', 0.0), 0.0, 1.0))
        fc = fc * (1.0 - fb) + col * fb
        col = col + (fc - col) * np.clip(band * fg, 0.0, 1.0)[:, None]

    # the rainbow lives in the atmosphere, so it is drawn with it
    if world.rainbow:
        col = col + _rainbow(dirs, sun, float(world.rainbow_intensity),
                             float(world.rainbow_radius),
                             float(world.rainbow_width),
                             float(world.rainbow_secondary))
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


def _hash01(ix, iy, salt):
    """A repeatable [0,1) per integer cell. Integer mixing, not sin-fract:
    the sample positions out here run to hundreds of metres and a sine hash
    loses its low bits long before that."""
    h = (ix * np.int64(73856093)) ^ (iy * np.int64(19349663)) ^ np.int64(salt)
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return ((h & np.int64(0xFFFFFF)).astype(np.float32) / float(0x1000000))


def _wave_normal(p, world, time, lod=None):
    """A directional wave spectrum, the way an ocean actually behaves.

    The old version crossed four sine trains at fixed angles, which reads as
    corrugation rather than as water: real waves run mostly *with* the wind,
    with the shorter ones fanned out either side of it. `ocean_spread` is that
    fan -- zero gives a perfectly regular swell, one gives confused chop.

    `lod` is the pixel footprint of each sample, and `ocean_horizon_smooth`
    decides what is done about it. Waves smaller than the pixel they land in
    cannot be drawn cleanly, only aliased -- so a modern renderer fades them
    out, and the water goes smooth with distance.

    Bryce did not do that. Its ocean was a procedural water material on an
    infinite plane, evaluated per pixel with nothing filtering it, so the
    waves kept going all the way to the horizon and compressed into a band of
    fine shimmer rather than flattening into glass. That shimmer is not an
    artefact of the reproduction; it is what the pictures look like. So the
    fade is off by default and lives behind a control, because turning water
    to a mirror at the far end is the less accurate of the two.
    """
    amp = float(world.ocean_choppiness)
    # the length of the longest wave train, in world units, and nothing else.
    # It used to be multiplied by `ground_scale` -- the size of the chequer
    # squares -- so the waves changed size when you resized a pattern that is
    # not even drawn under water, and Wave Scale meant something different in
    # every scene.
    scale = max(float(getattr(world, 'ocean_wave_scale', 1.0)), 1e-3)
    t = float(time) * float(world.ocean_speed)
    wind = float(getattr(world, 'ocean_wind_angle', 0.6))
    spread = float(np.clip(getattr(world, 'ocean_spread', 0.6), 0.0, 1.0))
    octaves = int(max(getattr(world, 'ocean_detail', 5), 1))

    # 0 keeps every train at every distance, which is Bryce; 1 fades the ones
    # a pixel cannot resolve, which is smooth and modern and not what Bryce
    # looked like
    smooth = float(np.clip(getattr(world, 'ocean_horizon_smooth', 0.0),
                           0.0, 1.0))

    x, y = p[:, 0] / scale, p[:, 1] / scale

    # Where a pixel covers many wavelengths, sampling the middle of it makes
    # the trains beat against the pixel grid and the far water fills with
    # moire fringes -- regular, diagonal, and unmistakably a rendering
    # artefact. Bryce's water shimmered instead, because a noise field
    # undersampled gives speckle, not fringes. Taking the sample from a fixed
    # random point inside the pixel rather than its centre gives the same
    # thing: the contrast is untouched, it is only decorrelated between
    # neighbours. Deterministic in world space, so still water stays still.
    spark = float(np.clip(getattr(world, 'ocean_sparkle', 1.0), 0.0, 1.0))
    if lod is not None and spark > 0.0:
        cell = np.maximum(np.asarray(lod, np.float32), 1e-6)
        ix = np.floor(p[:, 0] / cell).astype(np.int64)
        iy = np.floor(p[:, 1] / cell).astype(np.int64)
        off = (cell / scale) * spark
        x = x + (_hash01(ix, iy, 0x9E37) - 0.5) * off
        y = y + (_hash01(ix, iy, 0x85EB) - 0.5) * off

    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    # variance of the slope that could not be drawn, which is not the same as
    # slope that is not there: a wave smaller than a pixel still tilts the
    # water inside it, and the way that shows up is a *wider* glitter rather
    # than a flat mirror. Dropping it outright is what turns distant water to
    # glass, and it is the single thing that stops an ocean reading as one.
    lost = np.zeros_like(x)
    freq, weight = 1.0, 1.0
    rng = np.random.default_rng(int(world.cloud_seed) + 4242)
    for i in range(octaves):
        # each train fans further off the wind as it gets shorter, which is
        # what makes a swell read as a swell and chop read as chop
        off = (rng.random() - 0.5) * 2.0 * spread * (0.35 + 0.65 * i / octaves)
        ang = wind + off * 1.4
        kx, ky = np.cos(ang), np.sin(ang)
        # every train starts wherever it likes. Without this they all peak
        # together at the origin and the sea reads as corrugated iron, which
        # is exactly what showed up once the waves were small enough to see
        start = rng.random() * 6.28318
        phase = (x * kx + y * ky) * freq + t * (1.0 + i * 0.3) * np.sqrt(freq) \
            + start
        w = weight
        if lod is not None and smooth > 0.0:
            # fade a train out as its wavelength approaches the pixel
            # footprint, in the proportion asked for and no more
            wavelength = scale / max(freq, 1e-6)
            keep = np.clip(wavelength / np.maximum(lod * 2.0, 1e-6) - 1.0,
                           0.0, 1.0)
            w = w * (1.0 - smooth + smooth * keep)
        c = np.cos(phase) * w
        dx = dx + c * kx * freq
        dy = dy + c * ky * freq
        # whatever this octave lost to the pixel footprint, in slope terms
        faded = (weight - w) * freq
        lost = lost + 0.5 * (faded * amp) ** 2
        freq *= 1.9
        weight *= 0.55
    n = np.stack([-dx * amp, -dy * amp, np.ones_like(dx)], axis=1)
    return M.normalize(n), lost.astype(np.float32)


def _ocean(world, dirs, hit_dirs, p, dist, sky_col_hit, time, lod):
    """The infinite water plane, put together the way a Bryce picture is.

    Bryce's water was a plane with a water material on it, and what made it
    read as water rather than as a mirror was the stack, not any one part:
    Fresnel so it is glass overhead and a mirror at the horizon, a deep colour
    under it that the shallow tint climbs toward as the angle steepens, and the
    sun's own reflection smeared down the wave slopes into a glitter path.

    That glitter path is the piece that was missing. It is not a highlight on a
    flat plane -- it is the sun found in the *distribution* of wave normals, so
    it widens with the chop and narrows as the water calms, and it lands where
    the sun actually is rather than where a specular term would put it.
    """
    n, sub = _wave_normal(p, world, time, lod=lod)
    v = -hit_dirs
    r = M.reflect(-v, n)
    r[:, 2] = np.abs(r[:, 2])                 # never reflect the sea floor
    refl = evaluate(world, r, None, strength=False,
                    eye=getattr(world, '_eye', None),
                    time=float(getattr(world, '_time', 0.0)))
    if refl is None:
        refl = np.broadcast_to(np.asarray(world.horizon, np.float32)[None, :],
                               (p.shape[0], 3)).copy()

    # Slope too small to draw is still slope. It scatters what the water
    # mirrors instead of reflecting it cleanly, so the reflection loses its
    # edges and settles toward the colour of the sky around the horizon.
    # Without this, water past the point where the waves stop being drawable
    # turns to a sheet of glass -- which is what "the waves fade away" looks
    # like, and it happened over most of the picture.
    if sub is not None:
        blur = np.clip(sub * 5.0, 0.0, 0.85).astype(np.float32)
        wide = np.asarray(getattr(world, 'horizon', (0.6, 0.7, 0.8)),
                          np.float32)[None, :]
        refl = refl + (wide - refl) * blur[:, None]

    facing = np.clip(M.dot(n, v), 0.0, 1.0)
    fres = 0.02 + 0.98 * np.power(1.0 - facing, 5.0)

    deep = np.asarray(getattr(world, 'ocean_deep',
                              world.ground_color), np.float32)[None, :]
    shallow = np.asarray(getattr(world, 'ocean_shallow',
                                 world.ground_color2), np.float32)[None, :]
    # looking straight down you see into the water; at a glancing angle the
    # path through it is longer and it goes to the deep colour
    body = deep + (shallow - deep) * np.power(facing, 0.6)[:, None]
    trans = float(np.clip(getattr(world, 'ocean_transparency', 0.25), 0.0, 1.0))
    body = body * (1.0 - trans * 0.5 * facing[:, None])
    # Water is lit from the whole sky, not only from what it mirrors --
    # without this the troughs go to near black and Bryce's water never did.
    # It *multiplies* the body rather than adding to it, because what comes
    # back up is skylight the water scattered, and water that scatters nothing
    # returns nothing. Added flat, it was a floor: the darkest waters could
    # not be dark, and a black lagoon under a blue sky came out the same mid
    # blue as everything else.
    zen = np.asarray(world.zenith, np.float32)[None, :]
    body = body * (1.0 + zen * 1.6 * float(np.clip(world.strength, 0.0, 4.0)))

    col = body * (1.0 - fres)[:, None] + refl * fres[:, None]

    # ---- the sun's glitter path
    glit = float(getattr(world, 'ocean_glitter', 1.0))
    if glit > 0.0 and float(world.sun_intensity) > 0.0:
        sun = _sun_vector(float(world.sun_elevation), float(world.sun_rotation))
        h = M.normalize(v + sun[None, :])
        ndh = np.clip(M.dot(n, h), 0.0, 1.0)
        # the width of the path is the width of the slope distribution, so
        # calmer water gives a tighter, brighter streak
        rough = max(float(world.ocean_choppiness), 1e-3) * \
            max(float(getattr(world, 'ocean_glitter_size', 0.45)), 1e-3)
        # the waves too small to draw widen the path instead of disappearing,
        # which is why a real glitter path spreads out toward the horizon
        rough_eff = np.sqrt(rough * rough + sub)
        power = np.clip(2.0 / (rough_eff * rough_eff) - 2.0, 2.0, 20000.0)
        spec = np.power(ndh, power)
        # a broader lobe is a dimmer one, or the horizon turns into a wall
        spec = spec * np.clip(power / np.maximum(2.0 / (rough * rough), 1e-6),
                              0.0, 1.0) ** 0.25
        # nothing glitters where the sun is not up, and nothing glitters
        # through the back of a wave
        up_mask = np.clip(sun[2] * 6.0, 0.0, 1.0)
        spec = spec * np.clip(M.dot(n, sun[None, :]), 0.0, 1.0) * up_mask
        sun_c = np.asarray(world.sun_color, np.float32)[None, :]
        col = col + sun_c * (spec * glit * float(world.sun_intensity))[:, None]

    # ---- foam on the crests, off by default: Bryce had no foam control and
    # putting one in unasked would be inventing a feature it did not have
    foam = float(getattr(world, 'ocean_foam', 0.0))
    if foam > 0.0:
        from .patterns import turbulence as _turb
        crest = np.clip(1.0 - facing * 0.0 + (1.0 - n[:, 2]) * 6.0 - 1.0,
                        0.0, 1.0)
        speck = _turb(p * (2.0 / max(float(getattr(world, 'ocean_wave_scale',
                                                   1.0)), 1e-3)),
                      octaves=4)
        mask = np.clip(crest * (0.4 + 0.6 * speck) * foam, 0.0, 1.0)
        fc = np.asarray(getattr(world, 'ocean_foam_color',
                                (0.92, 0.95, 0.96)), np.float32)[None, :]
        col = col + (fc - col) * mask[:, None]
    return col


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
    elif world.ground_mode == 'GRID':
        # the neon wireframe floor of every synthwave sleeve: thin bright
        # lines on the base colour, glowing wider with distance so the
        # grid survives minification instead of aliasing away
        gx = np.abs((p[:, 0] / scale) - np.round(p[:, 0] / scale))
        gy = np.abs((p[:, 1] / scale) - np.round(p[:, 1] / scale))
        px_w = np.clip(dist * np.float32(
            getattr(world, '_pixel_angle', 0.001) or 0.001) / scale,
            0.008, 0.25)
        line = np.maximum(1.0 - gx / px_w, 0.0) + \
            np.maximum(1.0 - gy / px_w, 0.0)
        line = np.clip(line, 0.0, 1.0)
        alt = np.asarray(world.ground_color2, np.float32)[None, :]
        col = col + (alt * 1.6 - col) * line[:, None]
    elif world.ground_mode == 'TILES':
        # bathhouse tiles: square cells, grout lines, a hashed per-tile
        # shade so the floor is not one flat repeat
        from .patterns import hash3
        cx = np.floor(p[:, 0] / scale)
        cy = np.floor(p[:, 1] / scale)
        fx = p[:, 0] / scale - cx
        fy = p[:, 1] / scale - cy
        grout = (np.minimum(np.minimum(fx, 1.0 - fx),
                            np.minimum(fy, 1.0 - fy)) < 0.04)
        shade = hash3(cx.astype(np.int64), cy.astype(np.int64),
                      np.int64(7)) * 0.25 + 0.75
        alt = np.asarray(world.ground_color2, np.float32)[None, :]
        col = col * shade[:, None]
        col = np.where(grout[:, None], alt, col)
    elif world.ground_mode == 'DESERT':
        # wind-ribbed dunes: long sine ridges displaced by low noise,
        # shaded by their own slope against the sun direction
        from .patterns import fbm
        u = p[:, 0] / scale
        v = p[:, 1] / scale
        warp = fbm(np.stack([u * 0.35, v * 0.35,
                             np.zeros(p.shape[0], np.float32)], 1),
                   octaves=3)
        ridge = np.sin((v + warp * 2.5) * np.float32(np.pi) * 2.0)
        rib = np.abs(ridge) ** 0.7
        alt = np.asarray(world.ground_color2, np.float32)[None, :]
        col = col + (alt - col) * (rib * 0.6 + warp * 0.25)[:, None]
    elif world.ground_mode == 'SNOW':
        # a bright field with sparse sun glints and faint blue shadowing
        # in the hollows
        from .patterns import fbm, hash3
        f = fbm(np.stack([p[:, 0] / scale, p[:, 1] / scale,
                          np.zeros(p.shape[0], np.float32)], 1), octaves=4)
        base = np.asarray(world.ground_color, np.float32)[None, :]
        hollow = np.asarray(world.ground_color2, np.float32)[None, :]
        col = base + (hollow - base) * (f * 0.5)[:, None]
        g = hash3((p[:, 0] * 37.0).astype(np.int64),
                  (p[:, 1] * 37.0).astype(np.int64), np.int64(3))
        glint = (g > 0.995).astype(np.float32) * \
            np.clip(2.0 - dist * 0.02, 0.0, 1.0)
        col = col + glint[:, None] * 0.8
    elif world.ground_mode == 'LAVA':
        # crusted rock over glowing cracks: inverted-crackle veins carry
        # the second colour as EMISSIVE heat, pulsing faintly over time
        from .patterns import turbulence
        u = np.stack([p[:, 0] / scale, p[:, 1] / scale,
                      np.zeros(p.shape[0], np.float32)], 1)
        v = turbulence(u, octaves=5)
        crack = np.clip((v - 0.62) * 6.0, 0.0, 1.0)
        pulse = 0.85 + 0.15 * np.float32(np.sin(float(time) * 1.7))
        glow = np.asarray(world.ground_color2, np.float32)[None, :]
        col = col * (1.0 - crack[:, None]) + \
            glow * (crack * 2.2 * pulse)[:, None]

    if world.ground_mode == 'OCEAN':
        # How much water one pixel covers. The footprint is not square: a ray
        # that grazes the plane is stretched a long way *along* the view, but
        # stays narrow *across* it, and a wave train running across the view
        # is still perfectly resolvable at that distance. Taking the long axis
        # -- which is what this did -- over-blurred by 1/sqrt(grazing), a
        # factor of ten near the horizon, and deleted almost every wave in the
        # picture. The area-equivalent square is the honest number.
        grazing = np.maximum(-dirs[hit][:, 2], 1e-4)
        angle = float(getattr(world, '_pixel_angle', 0.0))
        width = float(getattr(world, '_pixel_width', 0.0))
        if angle > 0.0:
            across = dist * angle
        elif width > 0.0:
            across = np.full_like(dist, width)     # orthographic: no falloff
        else:
            across = dist * 0.001
        lod = across / np.sqrt(grazing)            # sqrt(across * along)
        col = _ocean(world, dirs, dirs[hit], p, dist, sky_col[hit], time, lod)

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

    # haze with distance, which is what makes it read as receding ground.
    # Bryce called the rate Colour Perspective and applied it to the whole
    # scene; above zero it takes over from the plain linear fade and rolls off
    # exponentially, which is what stops a distant plane having a visible edge
    # where the fade runs out.
    fade = float(world.ground_fade)
    cp = float(getattr(world, 'color_perspective', 0.0))
    if cp > 0.0 and fade > 0.0:
        k = (1.0 - np.exp(-dist / max(fade, 1e-3) * cp))[:, None]
        col = col * (1.0 - k) + sky_col[hit] * k
    elif fade > 0.0:
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
        if eye is not None:
            # stashed so the water's reflection can evaluate the *same* sky
            # rather than one rendered from the origin
            world._eye = eye
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
    elif mode == 'BANDS':
        col = bands(world, dirs)
    elif mode == 'STARFIELD':
        col = starfield(world, dirs)
    elif mode == 'BRYCE':
        col = bryce(world, dirs, eye=eye)
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
