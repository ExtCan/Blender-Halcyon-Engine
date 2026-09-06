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
         'PHYSICAL', 'HDRI', 'PAINTED')


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


# ------------------------------------------------------- R233: painted sky

#: the painted sky's era looks: World field values the Look menu writes
#: (CUSTOM writes nothing). Named after the background departments they
#: imitate, not after any studio's own colour keys.
PAINTED_LOOKS = {
    'GOUACHE_DAY': {
        'horizon': (0.60, 0.74, 0.80), 'zenith': (0.20, 0.40, 0.74),
        'gradient_falloff': 0.55, 'blend_mode': 'SMOOTH',
        'paint_streaks': 0.35, 'paint_streak_scale': 12.0,
        'paint_streak_angle': 0.0, 'paint_dabs': 0.0,
        'paint_clouds': 0.4, 'paint_cloud_scale': 2.2,
        'paint_cloud_softness': 0.25,
        'paint_cloud_color': (0.99, 0.98, 0.95),
        'paint_cloud_shadow': (0.60, 0.63, 0.75), 'paint_cloud_height': 0.05,
        'paint_paper': 0.3, 'paint_paper_scale': 8.0, 'paint_wash': 0.2,
    },
    'WATERCOLOUR_DUSK': {
        'horizon': (0.93, 0.72, 0.52), 'zenith': (0.30, 0.34, 0.56),
        'gradient_falloff': 0.5, 'blend_mode': 'SMOOTH',
        'paint_streaks': 0.2, 'paint_streak_scale': 9.0,
        'paint_streak_angle': 0.0, 'paint_dabs': 0.0,
        'paint_clouds': 0.35, 'paint_cloud_scale': 1.6,
        'paint_cloud_softness': 0.6,
        'paint_cloud_color': (0.98, 0.84, 0.72),
        'paint_cloud_shadow': (0.52, 0.42, 0.56), 'paint_cloud_height': 0.0,
        'paint_paper': 0.45, 'paint_paper_scale': 7.0, 'paint_wash': 0.6,
    },
    'FLEISCHER_NIGHT': {
        'horizon': (0.075, 0.065, 0.095), 'zenith': (0.010, 0.012, 0.030),
        'gradient_falloff': 0.8, 'blend_mode': 'SMOOTH',
        'paint_streaks': 0.35, 'paint_streak_scale': 10.0,
        'paint_streak_angle': -8.0, 'paint_dabs': 0.0,
        'paint_clouds': 0.5, 'paint_cloud_scale': 1.8,
        'paint_cloud_softness': 0.1,
        'paint_cloud_color': (0.115, 0.105, 0.140),
        'paint_cloud_shadow': (0.018, 0.018, 0.032), 'paint_cloud_height': 0.05,
        'paint_paper': 0.2, 'paint_paper_scale': 8.0, 'paint_wash': 0.15,
    },
    'STORYBOARD': {
        'horizon': (0.90, 0.89, 0.86), 'zenith': (0.72, 0.72, 0.70),
        'gradient_falloff': 1.0, 'blend_mode': 'LINEAR',
        'paint_streaks': 0.25, 'paint_streak_scale': 14.0,
        'paint_streak_angle': 4.0, 'paint_dabs': 0.0,
        'paint_clouds': 0.3, 'paint_cloud_scale': 2.0,
        'paint_cloud_softness': 0.05,
        'paint_cloud_color': (0.96, 0.96, 0.94),
        'paint_cloud_shadow': (0.62, 0.62, 0.60), 'paint_cloud_height': 0.05,
        'paint_paper': 0.4, 'paint_paper_scale': 9.0, 'paint_wash': 0.1,
    },
}

PAINTED_LOOK_ITEMS = (
    ('CUSTOM', "Custom", "Your own dials"),
    ('GOUACHE_DAY', "Gouache Day",
     "A feature background department's daytime sky: opaque gouache, "
     "brushed flat, cumulus with a shadowed underside"),
    ('WATERCOLOUR_DUSK', "Watercolour Dusk",
     "A 1930s short's sky: a wet wash from warm horizon to cool zenith, "
     "soft clouds, the paper's granulation showing"),
    ('FLEISCHER_NIGHT', "Fleischer Night",
     "The Superman sky: near-black deco gradient, heavy low cloud, the "
     "brush dragged diagonally, a dark palette"),
    ('STORYBOARD', "Storyboard",
     "Grey wash on toothy board, dry-brushed clouds -- the sky a story "
     "sketch has"),
)


def apply_painted_look(world, key):
    """Write a look's dials onto the world (CUSTOM writes nothing)."""
    look = PAINTED_LOOKS.get(str(key))
    if not look:
        return False
    for k, v in look.items():
        setattr(world, k, v)
    return True


def painted(world, dirs):
    """A background painting on a flat panel in front of the camera.

    The panel faces the camera `paint_angle` degrees round from +X, one
    unit away, so a direction's panel point is its tangent coordinate
    (d.r / d.f, d.z / d.f) -- exact arithmetic, no seam in front, the
    picture crossed by a pan and climbed by a tilt the way the animation
    stand's background was. Directions behind the panel fade to the plain
    gradient over the last 10 degrees of grazing, the back of the stage.

    On the panel, in the order the painter worked: the gradient in flat
    colour, the clouds (a four-octave field thresholded by coverage,
    biased above the horizon, its edge dry-brushed or airbrushed by
    Softness, lit on top and shadowed beneath by the field's own vertical
    difference), the brush's streaks (value noise stretched along the
    stroke direction, two octaves), optional impasto dabs (the 1.75 Paint
    Strokes field, the costly part), the board's tooth, and a watercolour
    granulation as pigment density in Bousseau's law
    C' = C - (C - C^2)(d - 1).
    """
    from . import media as MD
    base = gradient(world, dirs)
    ang = np.radians(float(getattr(world, 'paint_angle', 90.0)))
    fx, fy = np.float32(np.cos(ang)), np.float32(np.sin(ang))
    d = dirs.astype(np.float32)
    df = d[:, 0] * fx + d[:, 1] * fy
    dfc = np.maximum(df, np.float32(0.08))
    x = (d[:, 0] * fy - d[:, 1] * fx) / dfc
    y = d[:, 2] / dfc
    salt = MD.salt_for(int(getattr(world, 'paint_seed', 0)), 0, 0)
    col = base.copy()

    # ---- clouds
    cover = float(np.clip(getattr(world, 'paint_clouds', 0.35), 0.0, 1.0))
    if cover > 0.0:
        cs = np.float32(max(float(getattr(world, 'paint_cloud_scale', 2.2)),
                            0.05))
        soft = float(np.clip(getattr(world, 'paint_cloud_softness', 0.3),
                             0.0, 1.0))
        height = float(getattr(world, 'paint_cloud_height', 0.2))

        def field(px, py, k):
            # three octaves, stretched: the summed noises crowd the middle
            # and a cloud is a SHAPE, not a haze
            f = (MD._n2(px * cs, py * cs, salt, k) * 0.55
                 + MD._n2(px * cs * 2.0, py * cs * 2.0, salt, k + 1) * 0.3
                 + MD._n2(px * cs * 4.0, py * cs * 4.0, salt, k + 2) * 0.15)
            return (f - 0.5) * 2.4 + 0.5
        c = field(x, y, 10)
        # the edge: a dry brush leaves it ragged, an airbrush smooth
        c = c + (1.0 - soft) * 0.16 * (MD._n2(x * cs * 9.0, y * cs * 9.0,
                                              salt, 14) - 0.5)
        hb = MD.smoothstep(height - 0.2, height + 0.1, y)
        thr = 1.0 - 0.9 * cover * hb
        e = 0.01 + soft * 0.10
        cov = MD.smoothstep(thr - e, thr + e, c)
        # lit on top, shadowed beneath: the coarsest octave's vertical
        # difference says which side of the blob this is
        c1 = MD._n2(x * cs, y * cs, salt, 10)
        cu1 = MD._n2(x * cs, (y + np.float32(0.3) / cs) * cs, salt, 10)
        bottom = MD.smoothstep(-0.03, 0.06, cu1 - c1)
        lit = np.asarray(getattr(world, 'paint_cloud_color',
                                 (0.97, 0.96, 0.93)), np.float32)[None, :]
        shd = np.asarray(getattr(world, 'paint_cloud_shadow',
                                 (0.58, 0.60, 0.70)), np.float32)[None, :]
        ccol = lit + (shd - lit) * (bottom * 0.85)[:, None]
        col = col + (ccol - col) * (cov * 0.95)[:, None]

    # ---- the brush: streaks along the stroke direction
    streaks = float(np.clip(getattr(world, 'paint_streaks', 0.5), 0.0, 1.0))
    sc = np.float32(max(float(getattr(world, 'paint_streak_scale', 12.0)),
                        0.05))
    c0, s0 = MD.rotation(float(getattr(world, 'paint_streak_angle', 0.0)))
    u = x * c0 + y * s0
    v = -x * s0 + y * c0
    if streaks > 0.0:
        st1 = MD._n2(u * sc * 0.3, v * sc * 2.5, salt, 20)
        st2 = MD._n2(u * sc * 0.9, v * sc * 7.0, salt, 21)
        streak = st1 * 0.65 + st2 * 0.35 - 0.5
        col = col * (1.0 + streaks * 0.5 * streak)[:, None]
    dabs = float(np.clip(getattr(world, 'paint_dabs', 0.0), 0.0, 1.0))
    if dabs > 0.0:
        ds = np.float32(max(float(getattr(world, 'paint_dab_scale', 10.0)),
                            0.05))
        val, alpha, _sid = MD.paint_strokes(x * ds, y * ds, (c0, s0), 2.6,
                                            0.85, MD.paint_slope(0.9), 0.6,
                                            0.3, salt + 1)
        col = col * (1.0 + dabs * 0.5 * (val - 1.0) * alpha)[:, None]

    # ---- the board and the water
    paper = float(np.clip(getattr(world, 'paint_paper', 0.3), 0.0, 1.0))
    ps = np.float32(max(float(getattr(world, 'paint_paper_scale', 8.0)), 0.05))
    if paper > 0.0:
        # the board: two tooth octaves and one run of fibres
        t1 = MD._n2(x * ps, y * ps, salt, 40)
        t2 = MD._n2(x * ps * 2.6, y * ps * 2.6, salt, 41)
        fb = MD._n2(x * ps * 0.15, y * ps * 1.8, salt, 42)
        tooth = t1 * 0.55 + t2 * 0.3 + fb * 0.15
        col = col * (1.0 + paper * 0.8 * (tooth - 0.5))[:, None]
    wash = float(np.clip(getattr(world, 'paint_wash', 0.25), 0.0, 1.0))
    if wash > 0.0:
        g = MD._n2(x * ps * 1.4, y * ps * 1.4, salt, 30)
        dens = (1.0 + wash * 1.6 * (g - 0.5))[:, None]
        cc = np.clip(col, 0.0, 1.0)
        col = cc - (cc - cc * cc) * (dens - 1.0)

    # ---- the back of the stage
    w = MD.smoothstep(0.0, 0.18, df)[:, None]
    out = base + (col - base) * w
    return np.clip(out, 0.0, None).astype(np.float32)


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
        size = float(getattr(world, 'star_size', 0.35))
        density = float(getattr(world, 'star_density', 0.5))
        scale = 60.0 + density * 340.0
        if getattr(world, 'old_stars', False):
            # the pre-1.38 stars, kept verbatim behind the switch: the
            # disc was measured in 3D grid space, where the sky sphere
            # cuts each cell at a different depth, so star sizes came
            # out essentially random and grew with render resolution
            from .patterns import starfield as _star_pattern
            mag = _star_pattern(dirs * scale, density,
                                size,
                                float(getattr(world, 'star_twinkle', 0.0)),
                                float(getattr(world, '_time', 0.0)))
        else:
            mag = _star_discs(dirs, scale, density, size,
                              float(getattr(world, 'star_twinkle', 0.0)),
                              float(getattr(world, '_time', 0.0)),
                              seed=0, thresh_k=0.25,
                              min_ang=_star_floor(world))
        # stars are not all white: hot ones read blue, cool ones amber, and a
        # single hash per cell is enough to say which
        tint = _hash3f_dirs(dirs * scale)
        warm = np.array([1.0, 0.86, 0.70], np.float32)
        cool = np.array([0.74, 0.84, 1.0], np.float32)
        star_col = cool[None, :] + (warm[None, :] - cool[None, :]) * tint[:, None]
        col = col + star_col * (mag * bright)[:, None]
    return col.astype(np.float32)


def _star_floor(world):
    """The smallest angular radius worth drawing: just over half a pixel.

    Without a floor, a correctly sized star at a preview resolution can
    fall entirely between the sample points and the sky goes empty; the
    era's stars were never smaller than the pixel that carried them.
    A renderer stamps the real pixel angle onto the world before
    evaluating; a caller that has none (preset thumbnails, probes,
    reflection rays before the stamp) gets the pixel of the era's
    640x480 at a standard lens, so stars exist at SOME size everywhere
    rather than being points of measure zero.
    """
    ang = float(getattr(world, '_pixel_angle', 0.0) or 0.0)
    if ang <= 0.0:
        ang = 0.003                     # ~one 350-line pixel at a 60-deg fov
    return ang * 0.6


def _star_discs(dirs, scale, density, size, twinkle, time, seed=0,
                thresh_k=0.25, min_ang=0.0):
    """Round stars of a FIXED angular size, whatever the resolution.

    The old drawing measured stars in grid units. For the Bryce layer
    that meant the whole grid cell lit up -- a star was a square whose
    pixel size followed the render resolution, so a sky tuned at 320x240
    turned into a field of blocks at 1920. For the starfield mode the
    disc was measured in 3D cell space, where the sky sphere cuts every
    cell at a different depth, so sizes came out arbitrary.

    Here a star is a point ON the sky -- the cell's hashed centre,
    projected to the unit sphere -- and its disc is measured as an angle
    between directions. Star Size 0.35 means 0.35 of a grid cell's
    angular width, at every resolution, in every direction.
    """
    p = dirs * scale
    cell = np.floor(p)
    ix = cell[:, 0].astype(np.int64)
    iy = cell[:, 1].astype(np.int64)
    iz = cell[:, 2].astype(np.int64) + int(seed)
    h = _hash3(ix, iy, iz)
    thresh = 1.0 - np.clip(density, 0.0, 1.0) * float(thresh_k)
    mag = np.where(h > thresh, (h - thresh) / max(1.0 - thresh, 1e-4),
                   np.float32(0.0)).astype(np.float32)
    live = mag > 0.0
    if not live.any():
        return np.zeros(dirs.shape[0], np.float32)
    # the star's own direction: its cell's hashed interior point, pushed
    # out to the unit sky
    centre = cell[live] + np.stack([_hash3(ix[live], iy[live], iz[live] + 11),
                                    _hash3(ix[live], iy[live], iz[live] + 23),
                                    _hash3(ix[live], iy[live], iz[live] + 47)],
                                   axis=1)
    s = M.normalize(centre.astype(np.float32))
    d_ang = np.linalg.norm(dirs[live] - s, axis=1)   # chord ~ angle out here
    rad = max(float(size), 1e-3) * 0.5 / max(float(scale), 1e-3)
    rad = max(rad, float(min_ang))
    # a solid core with a half-radius fade, not a soft cone: the era's
    # stars were hard points of light, and a linear falloff across the
    # whole disc reads as a blob
    disc = np.clip((1.0 - d_ang / rad) * 2.0, 0.0, 1.0)
    m = mag[live] * disc
    if twinkle > 0.0:
        phase = _hash3(ix[live], iy[live], iz[live] + 91) * (2.0 * np.pi)
        m = m * (1.0 - twinkle * 0.5 *
                 (1.0 + np.sin(time * 3.0 + phase)) * 0.5)
    out = np.zeros(dirs.shape[0], np.float32)
    out[live] = np.clip(m, 0.0, 1.0)
    return out


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


def _stars(dirs, density, brightness, seed, world=None):
    if brightness <= 0.0:
        return 0.0
    scale = 140.0 + density * 260.0
    if world is not None and getattr(world, 'old_stars', False):
        # the pre-1.38 drawing, verbatim: every pixel in a starred cell
        # lit up, so a star was a SQUARE the size of the cell -- one or
        # two pixels at 320x240, a nine-pixel block at 1920. Kept as Old
        # Stars for scenes tuned to it.
        p = dirs * scale
        cell = np.floor(p)
        h = _hash3(cell[:, 0].astype(np.int64), cell[:, 1].astype(np.int64),
                   cell[:, 2].astype(np.int64) + int(seed))
        thresh = 1.0 - np.clip(density, 0.0, 1.0) * 0.06
        hit = h > thresh
        mag = np.zeros(dirs.shape[0], np.float32)
        if hit.any():
            frac = (h[hit] - thresh) / max(1.0 - thresh, 1e-4)
            mag[hit] = frac * brightness
    else:
        size = float(getattr(world, 'star_size', 0.35)) \
            if world is not None else 0.35
        mag = _star_discs(dirs, scale, density, size,
                          twinkle=0.0, time=0.0, seed=int(seed),
                          thresh_k=0.06,
                          min_ang=_star_floor(world) if world is not None
                          else 0.0) * brightness
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
                           float(world.star_brightness), int(world.cloud_seed),
                           world=world)
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
    # The phase runs in float64 and is wrapped BEFORE the cosine. Sample
    # positions near the horizon run to hundreds of thousands of units, and
    # float32 keeps about seven digits: by a few tens of thousands the phase
    # had no fractional part left at all, so the far water was cos() of
    # rounding noise -- a band of garbage that moved with the camera. Double
    # precision carries the position out to the cap, and the wrap hands the
    # cosine a small, exact angle.
    x64 = x.astype(np.float64)
    y64 = y.astype(np.float64)
    two_pi = 2.0 * np.pi
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
        phase = np.mod((x64 * kx + y64 * ky) * freq
                       + t * (1.0 + i * 0.3) * np.sqrt(freq) + start,
                       two_pi).astype(np.float32)
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
    # every ray that dips below the plane hits it. The old -1e-5 threshold
    # left a sliver of rays -- a band a few pixels tall at the horizon --
    # that pointed down but were declared "level", so the sky showed
    # through in a bright line between the water and the horizon. The
    # distance is capped instead: past the cap the haze owns the pixel
    # anyway, and the cap is approached smoothly so no seam is drawn.
    below = dz < 0.0
    if not below.any():
        return sky_col
    eye = np.asarray(eye, np.float32)
    t = np.full(dirs.shape[0], -1.0, np.float32)
    t_cap = np.float32(5.0e5)
    with np.errstate(divide='ignore', over='ignore'):
        raw = (float(world.ground_height) - eye[2]) / \
            np.minimum(dz[below], np.float32(-1e-12)).astype(np.float64)
    raw = np.where(raw > 0.0, raw, -1.0)
    # smooth approach to the cap, the _dome_project treatment: the plane
    # keeps receding but never runs off to infinity, so float32 never
    # sees a coordinate it cannot hold
    t[below] = np.where(raw > 0.0,
                        (t_cap * raw / (t_cap + raw)), -1.0).astype(np.float32)
    hit = below & (t > 0.0)
    if not hit.any():
        return sky_col

    p = eye[None, :] + dirs[hit] * t[hit, None]
    dist = t[hit]
    col = np.broadcast_to(np.asarray(world.ground_color, np.float32)[None, :],
                          (p.shape[0], 3)).copy()
    scale = max(float(world.ground_scale), 1e-3)
    # R204: modes with a self-luminous part split it out so the scene
    # lighting stage can shade the surface without dimming the glow.
    # None means "col is all there is" (diffuse) and costs nothing.
    g_diff = None
    g_emit = None

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
    elif world.ground_mode in ('TILES', 'GRID'):
        # bathhouse tiles: square cells, grout lines, a hashed per-tile
        # shade. R203: the grout WIDTH and the per-tile shade variance
        # are real dials now (the field could not edit either), the
        # grout can GLOW past 1 (which is the synthwave floor -- the
        # separate Neon Grid mode retired into exactly this: thin
        # bright grout), and old exports still saying GRID render as
        # that thin-glow tiling rather than falling to a flat sheet
        from .patterns import hash3
        legacy_grid = world.ground_mode == 'GRID'
        gw = 0.012 if legacy_grid else float(
            getattr(world, 'ground_grout', 0.04) or 0.0)
        gw = min(max(gw, 0.0), 0.45)
        vari = 0.0 if legacy_grid else min(max(float(
            getattr(world, 'ground_tile_shade', 0.25) or 0.0), 0.0), 1.0)
        gglow = 1.6 if legacy_grid else max(float(
            getattr(world, 'ground_grout_glow', 1.0) or 0.0), 0.0)
        cx = np.floor(p[:, 0] / scale)
        cy = np.floor(p[:, 1] / scale)
        fx = p[:, 0] / scale - cx
        fy = p[:, 1] / scale - cy
        edge = np.minimum(np.minimum(fx, 1.0 - fx),
                          np.minimum(fy, 1.0 - fy))
        grout = edge < gw
        shade = hash3(cx.astype(np.int64), cy.astype(np.int64),
                      np.int64(7)) * vari + (1.0 - vari)
        alt = np.asarray(world.ground_color2, np.float32)[None, :] \
            * np.float32(gglow)
        col = col * shade[:, None]
        col = np.where(grout[:, None], alt, col)
        if gglow > 1.0:
            # the grout's glow past 1 is SELF-lit: under scene lighting
            # the neon floor keeps burning inside cast shadows, exactly
            # as a neon floor should. The first 1.0 of it stays diffuse
            # (a lit tile floor with bright paint), the excess rides as
            # emission outside the lighting blend.
            base2 = np.asarray(world.ground_color2, np.float32)[None, :]
            tiles_lit = np.broadcast_to(
                np.asarray(world.ground_color, np.float32)[None, :],
                col.shape) * shade[:, None]
            g_diff = np.where(grout[:, None],
                              np.broadcast_to(base2, col.shape), tiles_lit)
            g_emit = np.where(grout[:, None],
                              base2 * np.float32(gglow - 1.0),
                              np.float32(0.0))
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
        # R204: the crest strength is a dial (0.6 was baked in)
        rs_ = min(max(float(getattr(world, 'ground_ridge', 0.6)
                            or 0.0), 0.0), 1.0)
        col = col + (alt - col) * (rib * rs_ + warp * 0.25)[:, None]
    elif world.ground_mode == 'SNOW':
        # a bright field with sparse sun glints and faint blue shadowing
        # in the hollows. R204 field find: the glints were a hardcoded
        # white -- a colour you could SEE but not change. They wear
        # ground_color3 now, with a Sparkle amount dial; the defaults
        # (white, 1.0) are the old field bit for bit
        from .patterns import fbm, hash3
        f = fbm(np.stack([p[:, 0] / scale, p[:, 1] / scale,
                          np.zeros(p.shape[0], np.float32)], 1), octaves=4)
        base = np.asarray(world.ground_color, np.float32)[None, :]
        hollow = np.asarray(world.ground_color2, np.float32)[None, :]
        col = base + (hollow - base) * (f * 0.5)[:, None]
        g = hash3((p[:, 0] * 37.0).astype(np.int64),
                  (p[:, 1] * 37.0).astype(np.int64), np.int64(3))
        spark = max(float(getattr(world, 'ground_sparkle', 1.0)
                          or 0.0), 0.0)
        glint = (g > 0.995).astype(np.float32) * \
            np.clip(2.0 - dist * 0.02, 0.0, 1.0)
        gcol = np.asarray(tuple(getattr(world, 'ground_color3',
                                        (1.0, 1.0, 1.0)))[:3],
                          np.float32)[None, :]
        col = col + glint[:, None] * (0.8 * spark) * gcol
    elif world.ground_mode == 'LAVA':
        # R203 rebuild (field: "just looks bad and can't be edited").
        # A real lava field now: plates of darkened crust separated by
        # ridged fissures, heat BLEEDING outward from every crack (the
        # old hard threshold drew hairlines on a flat sheet), embers
        # freckling the hot zones, and three dials -- Crack Width,
        # Glow, Pulse -- where there were none.
        from .patterns import fbm, ridged, turbulence
        cw = min(max(float(getattr(world, 'ground_crack_width', 0.35)
                           or 0.35), 0.02), 2.0)
        gs = max(float(getattr(world, 'ground_glow', 1.0) or 0.0), 0.0)
        pu = min(max(float(getattr(world, 'ground_pulse', 0.15)
                           or 0.0), 0.0), 1.0)
        u = np.stack([p[:, 0] / scale, p[:, 1] / scale,
                      np.zeros(p.shape[0], np.float32)], 1)
        rid = ridged(u * 0.9, octaves=5)
        # fissure field: the ridge crests, widened by the dial; the
        # heat term falls off SMOOTHLY away from each crack
        fis = np.clip((rid - (0.72 - cw * 0.25)) / max(cw * 0.22, 1e-3),
                      0.0, 1.0)
        heat = fis * fis
        bleed = np.clip((rid - (0.72 - cw * 0.6)) / max(cw * 0.6, 1e-3),
                        0.0, 1.0) ** 2 * 0.45
        # crust: the base colour broken into plates, darker between
        crust_v = fbm(u * 1.7 + 31.0, octaves=4)
        crust = col * (0.55 + 0.6 * crust_v)[:, None]
        pulse = 1.0 - pu + pu * np.float32(
            0.5 + 0.5 * np.sin(float(time) * 1.7))
        glow = np.asarray(world.ground_color2, np.float32)[None, :]
        # R204: the embers wear their own colour (ground_color3 as a
        # tint on the glow; white = exactly the old picture)
        etint = np.asarray(tuple(getattr(world, 'ground_color3',
                                         (1.0, 1.0, 1.0)))[:3],
                           np.float32)[None, :]
        emb = fbm(u * 6.3 + 77.7, octaves=3)
        ember = np.clip((emb - 0.78) * 8.0, 0.0, 1.0) * heat
        core_hot = np.clip(heat * 2.4 + bleed, 0.0, 3.2) * gs * pulse
        emb_hot = np.clip(ember * 1.5, 0.0, 3.2) * gs * pulse
        # R204: the molten glow is EMISSION -- scene lighting shades the
        # crust plates but a shadow across lava never dims the heat
        lava_d = crust * (1.0 - np.clip(heat + bleed, 0.0, 1.0))[:, None]
        lava_e1 = glow * core_hot[:, None]
        lava_e2 = glow * etint * emb_hot[:, None]
        col = lava_d + lava_e1 + lava_e2
        g_diff, g_emit = lava_d, lava_e1 + lava_e2
    elif world.ground_mode == 'MATERIAL':
        # R203: the ground wears a MATERIAL -- the picked material's
        # node graph, evaluated at the plane's own points (P on the
        # plane, N straight up, the view ray as incidence, UVs tiled
        # by Scale). Whatever the graph says the surface looks like,
        # the plane looks like, to the horizon.
        g = getattr(world, 'ground_graph', None)
        if g:
            from .nodeeval import (Closure, GraphEvaluator,
                                   ShadeContext, to_color, to_value)
            n_pts = p.shape[0]
            ctx = ShadeContext(n_pts)
            ctx.P = p.astype(np.float32)
            up_n = np.zeros((n_pts, 3), np.float32)
            up_n[:, 2] = 1.0
            ctx.N = up_n
            ctx.Ng = up_n.copy()
            ctx.I = dirs[hit].astype(np.float32)
            ctx.uv = (p[:, :2] / scale).astype(np.float32)
            ctx.generated = (ctx.P / scale).astype(np.float32)
            ctx.time = float(time)
            try:
                ev = GraphEvaluator(
                    g, ctx, textures or {},
                    getattr(world, 'ground_programs', None) or {})
                cl, _op = ev.evaluate_surface()
                if isinstance(cl, Closure) and cl.items:
                    acc = np.zeros((n_pts, 3), np.float32)
                    eacc = None
                    for kind, wgt, pr in cl.items:
                        c3 = to_color(pr.get('color'), n_pts)[:, :3]
                        st = pr.get('strength')
                        s = (to_value(st, n_pts)[:, None]
                             if st is not None else 1.0)
                        wv = np.asarray(wgt, np.float32).reshape(-1)
                        if wv.shape[0] != n_pts:
                            wv = np.broadcast_to(wv, (n_pts,))
                        term = c3 * s * wv[:, None]
                        # R204: EMISSION closures are self-lit -- the
                        # scene lighting stage must not shadow them
                        if str(kind).upper() == 'EMISSION':
                            eacc = term if eacc is None else eacc + term
                        else:
                            acc += term
                    col = acc if eacc is None else acc + eacc
                    g_diff, g_emit = acc, eacc
            except Exception:                                   # noqa: BLE001
                pass

    # ------------------------------------------------- R204: scene lighting
    # "The infinite floors don't react to lighting either." Now they do.
    # The renderer hands the plane a callback (world._ground_light) that
    # returns per-point irradiance: the ambient pool plus every lamp's
    # contribution against the plane's straight-up normal -- Lambert for
    # sun/point/spot, the wrap term for hemis, the Stokes form factor for
    # area lamps -- with cast shadows from scene geometry (ray-traced or
    # shadow-mapped, whichever the lamp uses). Scene Lighting blends from
    # the old self-lit flat look (0, those pixels bit for bit -- this
    # whole stage is skipped) to fully lit (1). Self-luminous parts (lava
    # heat, neon grout past 1, EMISSION closures) ride outside the blend:
    # a shadow across them never dims the glow. OCEAN keeps its own
    # sun-and-sky model and never enters. When the scene has no lamps at
    # all the renderer hands no callback: nothing to react to, and the
    # picture stays exactly what it always was.
    if world.ground_mode != 'OCEAN':
        lit = min(max(float(getattr(world, 'ground_lighting', 1.0)
                            or 0.0), 0.0), 1.0)
        light_fn = getattr(world, '_ground_light', None)
        if lit > 0.0 and light_fn is not None:
            try:
                irr = light_fn(p)
            except Exception:                                   # noqa: BLE001
                irr = None
            if irr is not None:
                d_part = col if g_diff is None else g_diff
                col = d_part * (np.float32(1.0 - lit) +
                                np.float32(lit) *
                                np.asarray(irr, np.float32))
                if g_emit is not None:
                    col = col + g_emit

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
    elif mode == 'PAINTED':
        col = painted(world, dirs)
    else:
        return None                      # caller falls back to the node graph
    if strength:
        col = col * float(getattr(world, 'strength', 1.0))
    if getattr(world, 'ground_plane', False) and eye is not None:
        col = ground_plane(world, dirs, col.astype(np.float32), eye, time,
                           textures)
    return col.astype(np.float32)


# ---------------------------------------------------------------- weather


def _wthr01(idx, salt):
    """Deterministic per-particle random in [0,1): the Wang mix the
    halos and caustics use, so a drop's path is a pure function of
    (seed, layer, particle) -- identical across runs and devices."""
    from .patterns import _wang01p
    u = (np.asarray(idx, np.int64) & 0xFFFFFFFF).astype(np.uint32)
    return _wang01p(u ^ np.uint32(salt))


#: the styles: how each kind splats and composites. RAIN and EMBERS
#: are light (additive, the era's sprite weather); SNOW and ASH are
#: matter (alpha-over). Acid rain is RAIN wearing a green colour.
WEATHER_KINDS = ('NONE', 'RAIN', 'SNOW', 'EMBERS', 'ASH')


def weather_overlay(img, world, w, h, time=0.0, out_wh=None,
                    sel_mask=None):
    """R200: the Weather overlay -- rain, snow, embers, ash.

    A screen-space particle field composited IN FRONT of the finished
    frame (geometry, halos and sky alike -- weather never replaces the
    sky, it falls in front of it). Every particle's position is a pure
    function of (weather_seed, layer, index, time): the same frame
    renders identically across runs, devices, refine passes and
    supersample factors, and scrubbing the timeline is stable because
    nothing integrates -- position is evaluated, not stepped.

    Layers give parallax: layer 0 is nearest (largest, fastest,
    brightest), each deeper layer smaller, slower and dimmer by the
    same factor. Angle 0 falls straight DOWN the screen, pi rises
    (embers), anything between is the diagonal; Drift wobbles each
    particle across its travel line on its own hashed phase. Speed,
    sizes, streak lengths and drift amplitudes are resolution-true
    (scaled by frame height against the 480-line reference), so the
    same scene keeps the same look at any resolution and under any
    supersample.

    `out_wh` is the OUTPUT frame size (before supersampling): the
    particle count is derived from it, never from the internal buffer,
    so a supersampled render draws the same drops in the same places,
    only sharper. `sel_mask` (a refine pass) composites only the
    flagged pixels -- values there are identical to a full pass,
    per-pixel independence as everywhere else in the engine.

    Returns img (modified in place where weather lands). NONE, zero
    density and zero opacity all return the buffer untouched --
    bitwise-neutral, and every existing scene has exactly that.
    """
    kind = str(getattr(world, 'weather', 'NONE') or 'NONE').upper()
    if kind == 'NONE' or kind not in WEATHER_KINDS:
        return img
    dens = float(getattr(world, 'weather_density', 1.0) or 0.0)
    op = float(np.clip(getattr(world, 'weather_opacity', 0.8), 0.0, 1.0))
    if dens <= 0.0 or op <= 0.0:
        return img
    size = max(float(getattr(world, 'weather_size', 1.0) or 1.0), 0.05)
    speed = max(float(getattr(world, 'weather_speed', 1.0) or 0.0), 0.0)
    ang = float(getattr(world, 'weather_angle', 0.0) or 0.0)
    drift = max(float(getattr(world, 'weather_drift', 0.2) or 0.0), 0.0)
    col = np.asarray(tuple(getattr(world, 'weather_color',
                                   (0.85, 0.90, 1.0)))[:3], np.float32)
    layers = int(np.clip(int(getattr(world, 'weather_layers', 3) or 1),
                         1, 4))
    streak = max(float(getattr(world, 'weather_streak', 1.0) or 0.0), 0.0)
    glow = float(np.clip(getattr(world, 'weather_glow', 0.0), 0.0, 1.0))
    flick = float(np.clip(getattr(world, 'weather_flicker', 0.0),
                          0.0, 1.0))
    seed = int(getattr(world, 'weather_seed', 0) or 0)
    t = float(time)
    ow, oh = out_wh if out_wh is not None else (w, h)
    # resolution-true units against the era's 480-line reference, the
    # same convention the halo line widths settled on (R197)
    rs = max(float(h) / 480.0, 0.25)
    # count follows the OUTPUT area, capped for the perf ultimatum --
    # a fuller sky comes from Density, not from megapixels
    n0 = int(round(dens * 150.0 * (float(ow) * float(oh))
                   / (640.0 * 480.0)))
    n0 = int(np.clip(n0, 1, 4000))
    dx_s, dy_s = float(np.sin(ang)), float(np.cos(ang))   # screen dirs
    field = _WthrField(h, w)
    for L in range(layers):
        fk = 1.0 / (1.0 + 0.65 * L)
        n = max(int(round(n0 * (0.7 + 0.3 * fk))), 1)
        idx = np.arange(n, dtype=np.int64) * 4 + L * 1048573 \
            + seed * 8191
        bx = _wthr01(idx, 0x2545F491)          # base x fraction
        by = _wthr01(idx + 1, 0x9E3779B9)      # base y fraction
        ph = _wthr01(idx + 2, 0x85EBCA6B)      # drift phase
        ph2 = _wthr01(idx + 3, 0xA511E9B3)     # flicker phase
        spd_px = speed * 340.0 * rs * fk
        # rows run bottom-first, so "down the screen" is -y in rows
        x = bx * w + dx_s * spd_px * t
        y = by * h - dy_s * spd_px * t
        if drift > 0.0:
            wob = np.sin(2.0 * np.pi * (t * 0.35 + ph)) \
                * drift * 34.0 * rs * fk
            x = x + wob * dy_s
            y = y + wob * dx_s
        x = np.mod(x, float(w))
        y = np.mod(y, float(h))
        bri = np.full(n, op * (0.45 + 0.55 * fk), np.float32)
        if flick > 0.0:
            tw = 0.5 + 0.5 * np.sin(2.0 * np.pi * (t * 3.1 + ph2))
            bri = bri * (1.0 - flick * tw).astype(np.float32)
        r_core = max(size * (2.6 if kind in ('SNOW', 'ASH') else 1.1)
                     * rs * fk, 0.55)
        if kind == 'RAIN' and streak > 0.0:
            # the streak: samples back along the travel line, the
            # tail fading -- the classic sprite rain stroke, drawn
            # thin (a 3x3 window per sample keeps a storm cheap).
            # A bigger drop trails a longer stroke: Size scales the
            # length too (x1.0 at the default -- neutral), which is
            # also what keeps the dial honest when the radius floor
            # engages at preview resolutions
            slen = streak * 15.0 * rs * fk * (0.6 + 0.4 * size)
            ns = max(int(min(6.0 + streak * 6.0, 22.0)), 2)
            jf = np.arange(ns, dtype=np.float32) / float(ns - 1)
            sx = x[:, None] - dx_s * slen * jf[None, :]
            sy = y[:, None] + dy_s * slen * jf[None, :]
            sw = (bri[:, None] * (1.0 - 0.75 * jf[None, :])
                  / (0.45 * ns)).astype(np.float32)
            _wthr_splat(field, sx.ravel(), sy.ravel(), sw.ravel(),
                        max(r_core * 0.35, 0.55))
        else:
            _wthr_splat(field, x, y, bri, r_core)
        if glow > 0.0:
            _wthr_splat(field, x, y, bri * 0.22 * glow, r_core * 3.0)
    ca = np.clip(field.resolve(), 0.0, 1.0)
    if sel_mask is not None:
        ca = ca * np.asarray(sel_mask, np.float32)
    live = ca > 0.0
    if not live.any():
        return img
    # composite only where weather actually landed -- rain covers a
    # few percent of the frame, and dense whole-frame arithmetic here
    # was most of the overlay's cost
    cav = ca[live][:, None]
    rgb = img[:, :, :3]
    a = img[:, :, 3]
    if kind in ('RAIN', 'EMBERS'):
        rgb[live] += cav * col[None, :]
    else:
        rgb[live] = rgb[live] * (1.0 - cav) + cav * col[None, :]
    a[live] += ca[live] * (1.0 - a[live])
    return img


class _WthrField:
    """Deferred splat accumulator: every layer's (index, weight) pairs
    pool up and ONE bincount resolves them -- the scatter is the
    overlay's hot path and bincount amortises beautifully."""

    def __init__(self, h, w):
        self.h, self.w = h, w
        self.idx = []
        self.wgt = []

    def resolve(self):
        if not self.idx:
            return np.zeros((self.h, self.w), np.float32)
        flat = np.concatenate(self.idx)
        wts = np.concatenate(self.wgt)
        return np.bincount(flat, weights=wts,
                           minlength=self.h * self.w) \
            .reshape(self.h, self.w).astype(np.float32)


def _wthr_splat(field, x, y, wgt, radius):
    """Queue quadratic-bump splats, vectorised over (particle, kernel
    offset). Radius is in pixels; the kernel window is capped at 8 px
    (a 17x17 tap ceiling per splat) so a 4K frame full of snow stays
    inside the performance envelope -- the one disclosed bound on the
    resolution-true scaling. No sqrt on the hot path."""
    h, w = field.h, field.w
    r = float(min(max(radius, 0.4), 8.0))
    R = max(int(np.ceil(r)), 1)
    offs = np.arange(-R, R + 1, dtype=np.float32)
    oy, ox = np.meshgrid(offs, offs, indexing='ij')
    ox = ox.ravel()[None, :]
    oy = oy.ravel()[None, :]
    xi = np.floor(x).astype(np.int32)[:, None] + ox.astype(np.int32)
    yi = np.floor(y).astype(np.int32)[:, None] + oy.astype(np.int32)
    fx = (np.asarray(x, np.float32)[:, None] - np.float32(0.5)) - xi
    fy = (np.asarray(y, np.float32)[:, None] - np.float32(0.5)) - yi
    d2 = (fx * fx + fy * fy) * np.float32(1.0 / (r * r))
    kw = np.maximum(np.float32(1.0) - d2, np.float32(0.0))
    kw = kw * kw * np.asarray(wgt, np.float32)[:, None]
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h) & (kw > 0.0)
    if not ok.any():
        return
    field.idx.append(yi[ok].astype(np.int64) * w + xi[ok])
    field.wgt.append(kw[ok].astype(np.float64))
