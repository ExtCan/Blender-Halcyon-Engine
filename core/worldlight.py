"""R254: the world's own luminary as a scene light.

The world used to be a backdrop plus an ambient colour: a Bryce sky could
show a blazing low sun and the scene under it was lit only by whatever
lamps existed. With `World.world_light` on, the luminary the sky DRAWS
becomes one ordinary SUN `Light` in `scene.lights` -- it lights, shadows,
ranks under the light limit, splits into passes and reaches the GPU's
`hal_lights` texels exactly as a lamp does, because past this module it
IS a lamp. Nothing downstream knows the difference except
`lights.select_lights`, which never lets the hardware light limit drop it.

What each world mode contributes (`luminary`):

    BRYCE      the sun, or the moon when `celestial` is MOON -- same
               direction (Bryce's one body, swapped), the moon's colour,
               MOON_LIGHT of the sun's strength scaled by the lit
               fraction of the phase the disc shows (plus earthshine)
    PHYSICAL   the sun (the fields BRYCE shares)
    NODES      the first Sky Texture node's sun, when its Vector input is
               unlinked (a mapped sky has no direction this can know)
    HDRI /     the brightest region of the image, found once per image
    CUBEMAP    (`find_image_sun`) -- or, with `world_light_find` MANUAL,
               the direction and colour typed into the panel
    the rest   no luminary: no light, and the panel says so

The direction is the one the PICTURE shows: sky.evaluate turns view
directions by -rotation before sampling, so the luminary's world-space
direction is its sky-space direction turned by +rotation (NODES does not
read the rotation, and neither does its sun here).

The light is made at render entry (`sync`, called beside
`collect_exclusive_lights`), not at export: the export fast path for lamp
drags rebuilds `scene.lights` from the lamps alone, hand-built scenes
never see the exporter, and the image search wants the PREPARED textures
(the sky the frame will actually draw). `sync` is idempotent -- it drops
the synthetic light it made last time and appends a fresh one -- and with
the toggle off and no synthetic light present it does not touch the scene
at all, which is the neutrality every existing render rests on.
"""
import numpy as np

from . import mathx as M
from . import sky as SKY
from .scene import Light

#: the synthetic light's name: light groups, per-lamp passes and the
#: console all call it this
NAME = 'World Light'

#: full moonlight as a fraction of the same sky's sunlight. The real ratio
#: (about 1 / 400 000) is a black frame; Bryce's night scenes read as a dim
#: cool day, and a quarter is that picture. World Light Strength scales it.
MOON_LIGHT = 0.25

#: the lamp energy of a luminary at intensity 1. The Lambert term divides
#: by pi, so at pi a surface facing the luminary squarely reads exactly
#: its own colour: intensity 1 is 'full daylight', not a dim lamp.
UNIT_ENERGY = float(np.pi)

#: the world modes that have a luminary to offer
MODES_WITH_LUMINARY = ('BRYCE', 'PHYSICAL', 'NODES', 'HDRI', 'CUBEMAP')
IMAGE_MODES = ('HDRI', 'CUBEMAP')

#: the Sky Texture node draws its disc in this colour (nodeeval.n_tex_sky)
NODE_SUN_COLOR = (1.0, 0.95, 0.85)

#: the widest Sun angle the light is given, radians (the default Bryce
#: disc, 2 x 0.03). The shadow softness follows the disc the sky draws up
#: to here and no further: a 'cinema' sun or a low-resolution skybox's
#: blob spans many degrees, and the engine's map shadows blur away to
#: nothing as a Sun lamp's Angle passes about 0.1 -- a sky that showed a
#: sun and cast no shadow would be the dead control this feature exists
#: to remove.
MAX_ANGLE = 0.06

#: MANUAL's shadow softness: the real sun's angular diameter, radians
MANUAL_ANGLE = 0.00918

_PRINTED = set()
_FIND_CACHE = {}
FIND_CACHE_MAX = 4

#: what the last sync decided, in words -- the panel shows it under the
#: toggle so a detected sun can be checked against the picture
LAST_REPORT = {'text': ''}


def _print_once(msg):
    if msg not in _PRINTED:
        _PRINTED.add(msg)
        print(f'[Halcyon] world light: {msg}')


def moon_lit_fraction(phase, earthshine=0.0):
    """How much of the moon's disc is lit at `phase` (0 new, 0.5 full, 1
    new again), with the earthshine floor `sky._moon` gives the dark side:
    the disc's mean brightness, so the light follows the picture."""
    phase = float(np.clip(phase, 0.0, 1.0))
    lit = 0.5 - 0.5 * float(np.cos(phase * 2.0 * np.pi))
    shine = float(np.clip(earthshine, 0.0, 1.0))
    return lit * (1.0 - shine) + shine


def horizon_fade(direction, angular_radius):
    """0 with the disc wholly under the horizon, 1 wholly above, linear in
    the elevation between -- a setting sun dims out instead of popping."""
    elev = float(np.arcsin(np.clip(float(direction[2]), -1.0, 1.0)))
    r = max(float(angular_radius), 1e-4)
    return float(np.clip((elev + r) / (2.0 * r), 0.0, 1.0))


def _to_world(local, world):
    """Sky space -> world space: the +rotation turn that undoes
    sky.evaluate's -rotation on the view direction."""
    d = np.asarray(local, np.float32).reshape(1, 3)
    d = SKY._rotate_z(d, float(getattr(world, 'rotation', 0.0)))
    return M.normalize(d)[0].astype(np.float32)


# --------------------------------------------------------- the image search


def _image_source(world, textures):
    """(sample(dirs) -> rgb, cache key, held pixels, texel width) for an
    image world, or None without a usable image."""
    mode = str(getattr(world, 'mode', ''))
    tint = tuple(round(float(v), 6)
                 for v in getattr(world, 'env_tint', (1.0, 1.0, 1.0)))
    if mode == 'CUBEMAP':
        atlas = SKY.cube_atlas(world, textures)
        if atlas is None:
            return None
        px = atlas.pixels
        key = ('CUBEMAP', id(px), tuple(np.shape(px)),
               str(getattr(world, 'cube_filter', 'NEAREST')), tint)
        return (lambda d: SKY.cubemap(world, d, textures), key, px,
                int(np.shape(px)[1]) * 4)
    tex = SKY.env_texture(world, textures)
    if tex is None:
        return None
    px = tex.pixels
    key = ('HDRI', id(px), tuple(np.shape(px)),
           str(getattr(world, 'env_mapping', 'EQUIRECT')),
           str(getattr(world, 'env_filter', 'BILINEAR')), tint)
    return (lambda d: SKY.hdri(world, d, textures), key, px,
            int(np.shape(px)[1]))


def find_image_sun(world, textures):
    """The brightest region of an HDRI / cube-map world.

    Returns ({'dir': sky-space unit vector, 'color': rgb with its largest
    channel 1, 'angle': angular diameter in radians}, None) or
    (None, why).

    The search reads the SKY, not the file: the sky's own sampler is
    evaluated over a latitude-longitude grid about as fine as the image, so
    every mapping (equirectangular, mirror ball, each cube layout and
    convention, the face turns) is covered by the one rule the frame
    itself draws with. The grid is pooled into cells of about 1.4 degrees
    (a hot texel alone does not win), the brightest cell is the candidate,
    and the answer is the luminance-weighted centre of the bright texels
    around it.

    A wrong sun is worse than none, so two cases are refused by name: a
    sky with no region standing out from its median, and a sky where an
    equally bright region lies somewhere else (an 8-bit image clips its
    sun and its white clouds to the same 1.0). Both leave the Manual
    direction to the user.
    """
    src = _image_source(world, textures)
    if src is None:
        return None, 'the world has no usable image'
    sample, key, held, tex_w = src
    hit = _FIND_CACHE.get(key)
    if hit is not None and hit[0] is held:
        return hit[1]
    W = 512 if tex_w <= 512 else (1024 if tex_w <= 1024 else 2048)
    H = W // 2
    lon = ((np.arange(W, dtype=np.float32) + 0.5) / W * 2.0 - 1.0) * np.pi
    lat = ((np.arange(H, dtype=np.float32) + 0.5) / H - 0.5) * np.pi
    cl, sl = np.cos(lat), np.sin(lat)
    col = np.empty((H, W, 3), np.float32)
    rows = max(1, 65536 // W)
    for j0 in range(0, H, rows):
        j1 = min(j0 + rows, H)
        d = np.empty((j1 - j0, W, 3), np.float32)
        d[:, :, 0] = cl[j0:j1, None] * np.cos(lon)[None, :]
        d[:, :, 1] = cl[j0:j1, None] * np.sin(lon)[None, :]
        d[:, :, 2] = sl[j0:j1, None]
        col[j0:j1] = np.asarray(sample(d.reshape(-1, 3)),
                                np.float32)[:, :3].reshape(j1 - j0, W, 3)
    col = np.nan_to_num(col, nan=0.0, posinf=0.0, neginf=0.0)
    lum = np.maximum(col @ np.array([0.2126, 0.7152, 0.0722], np.float32),
                     0.0)
    p = W // 256
    cells = lum.reshape(H // p, p, W // p, p).mean(axis=(1, 3))
    ch, cw = cells.shape
    # 3x3 box: longitude wraps, latitude clamps
    pad = np.concatenate([cells[:1], cells, cells[-1:]], axis=0)
    pad = np.concatenate([pad[:, -1:], pad, pad[:, :1]], axis=1)
    smooth = sum(pad[a:a + ch, b:b + cw]
                 for a in range(3) for b in range(3)) / 9.0
    peak = float(smooth.max())
    med = float(np.median(cells))

    def done(result):
        while len(_FIND_CACHE) >= FIND_CACHE_MAX:
            _FIND_CACHE.pop(next(iter(_FIND_CACHE)))
        _FIND_CACHE[key] = (held, result)
        return result

    if peak <= 1e-6 or peak < 1.25 * max(med, 1e-6):
        return done((None, 'no region of the image stands out as a sun '
                           f'(brightest {peak:.3f}, median {med:.3f})'))
    cj, ci = np.unravel_index(int(np.argmax(smooth)), smooth.shape)
    clat = (np.arange(ch, dtype=np.float32) + 0.5) / ch * np.pi - np.pi / 2
    clon = ((np.arange(cw, dtype=np.float32) + 0.5) / cw * 2.0 - 1.0) * np.pi
    cdir = np.stack([np.cos(clat)[:, None] * np.cos(clon)[None, :],
                     np.cos(clat)[:, None] * np.sin(clon)[None, :],
                     np.broadcast_to(np.sin(clat)[:, None], (ch, cw))],
                    axis=2)
    rivals = (smooth >= 0.9 * peak) & ((cdir @ cdir[cj, ci])
                                       < np.cos(np.radians(25.0)))
    if rivals.any():
        return done((None, 'the image has more than one equally bright '
                           'region (a clipped 8-bit sky cannot tell its '
                           'sun from its white clouds)'))
    # the bright texels of the 5x5 cells around the candidate
    jj = np.arange(max(cj - 2, 0) * p, min(cj + 3, ch) * p)
    ii = np.arange((ci - 2) * p, (ci + 3) * p) % W
    blk = lum[np.ix_(jj, ii)]
    thr = 0.5 * (float(blk.max()) + med)
    wgt = np.maximum(blk - thr, 0.0) * cl[jj][:, None]
    if float(wgt.sum()) <= 0.0:
        wgt = (blk >= float(blk.max())) * cl[jj][:, None]
    dirs = np.stack([cl[jj][:, None] * np.cos(lon[ii])[None, :],
                     cl[jj][:, None] * np.sin(lon[ii])[None, :],
                     np.broadcast_to(sl[jj][:, None], blk.shape)], axis=2)
    centre = (dirs * wgt[:, :, None]).sum(axis=(0, 1))
    n = float(np.linalg.norm(centre))
    if n <= 1e-9:
        return done((None, 'the bright region has no centre'))
    centre = (centre / n).astype(np.float32)
    rgb = (col[np.ix_(jj, ii)] * wgt[:, :, None]).sum(axis=(0, 1))
    top = float(rgb.max())
    rgb = tuple(float(v) / top for v in rgb) if top > 1e-9 \
        else (1.0, 1.0, 1.0)
    # the region's solid angle as a disc: its angular diameter
    omega = float(((blk > thr) * cl[jj][:, None]).sum()) \
        * (np.pi / H) * (2.0 * np.pi / W)
    angle = float(np.clip(2.0 * np.sqrt(max(omega, 0.0) / np.pi),
                          0.0, 0.35))
    return done(({'dir': centre, 'color': rgb, 'angle': angle}, None))


# ------------------------------------------------------------- the luminary


def _sky_node_sun(world):
    """The first Sky Texture node's sun, or (None, why)."""
    graph = getattr(world, 'graph', None) or {}
    nodes = graph.get('nodes', {}) if isinstance(graph, dict) else {}
    for key in sorted(nodes, key=str):
        node = nodes[key]
        if node.get('bl_idname') != 'ShaderNodeTexSky':
            continue
        if any(s.get('link') for s in (node.get('inputs') or ())
               if s.get('name') == 'Vector'
               or s.get('identifier') == 'Vector'):
            return None, ('the Sky Texture node is mapped through its '
                          'Vector input; its sun has no fixed direction')
        props = node.get('props') or {}
        return {'elevation': float(props.get('sun_elevation', 0.26)),
                'rotation': float(props.get('sun_rotation', 0.0)),
                'intensity': float(props.get('sun_intensity', 1.0)),
                'size': float(props.get('sun_size', 0.009512))}, None
    return None, 'the world node tree has no Sky Texture node'


def luminary(world, textures=None):
    """What the world offers as a light: ({'dir': world-space unit vector
    TOWARD the luminary, 'color', 'energy', 'angle', 'kind'}, None), or
    (None, why) when the world has none to give. `energy` is in intensity
    units, before UNIT_ENERGY and World Light Strength; `textures` may be a callable returning the
    prepared textures (only image worlds call it)."""
    mode = str(getattr(world, 'mode', 'NODES'))
    strength = float(getattr(world, 'strength', 1.0))
    if mode in ('BRYCE', 'PHYSICAL'):
        local = SKY._sun_vector(float(world.sun_elevation),
                                float(world.sun_rotation))
        moon = mode == 'BRYCE' and \
            str(getattr(world, 'celestial', 'SUN')) == 'MOON'
        size = float(world.moon_size if moon else world.sun_size)
        energy = float(world.sun_intensity) * strength
        color = tuple(float(v) for v in
                      (world.moon_color if moon else world.sun_color))[:3]
        if moon:
            energy *= MOON_LIGHT * moon_lit_fraction(
                world.moon_phase, getattr(world, 'moon_earthshine', 0.0))
        d = _to_world(local, world)
        energy *= horizon_fade(d, size)
        if energy <= 0.0:
            return None, ('the moon' if moon else 'the sun') + \
                ' is below the horizon or has no intensity'
        return {'dir': d, 'color': color, 'energy': energy,
                'angle': 2.0 * max(size, 0.0),
                'kind': 'MOON' if moon else 'SUN'}, None
    if mode == 'NODES':
        sun, why = _sky_node_sun(world)
        if sun is None:
            return None, why
        d = SKY._sun_vector(sun['elevation'], sun['rotation'])
        d = M.normalize(d.reshape(1, 3))[0].astype(np.float32)
        energy = sun['intensity'] * horizon_fade(d, sun['size'])
        if energy <= 0.0:
            return None, "the Sky Texture's sun is below the horizon " \
                         'or has no intensity'
        return {'dir': d, 'color': NODE_SUN_COLOR, 'energy': energy,
                'angle': 2.0 * max(sun['size'], 0.0), 'kind': 'NODE'}, None
    if mode in IMAGE_MODES:
        if strength <= 0.0:
            return None, 'the world strength is zero'
        if str(getattr(world, 'world_light_find', 'DETECT')) == 'MANUAL':
            local = SKY._sun_vector(
                float(getattr(world, 'world_light_elevation', 0.7)),
                float(getattr(world, 'world_light_azimuth', 0.6)))
            color = tuple(float(v) for v in getattr(
                world, 'world_light_color', (1.0, 1.0, 1.0)))[:3]
            return {'dir': _to_world(local, world), 'color': color,
                    'energy': strength, 'angle': MANUAL_ANGLE,
                    'kind': 'MANUAL'}, None
        if callable(textures):
            textures = textures()
        found, why = find_image_sun(world, textures or {})
        if found is None:
            return None, why + '; set World Light > Find to Manual and ' \
                               'give the direction by hand'
        return {'dir': _to_world(found['dir'], world),
                'color': found['color'], 'energy': strength,
                'angle': found['angle'], 'kind': 'IMAGE'}, None
    return None, f'a {mode} world has no sun or moon to light with'


def make_light(world, textures=None):
    """The world's luminary as a SUN `Light`, or None (the reason printed
    once). The light is marked `world_light = True`."""
    lum, why = luminary(world, textures)
    if lum is None:
        LAST_REPORT['text'] = f'No world light: {why}'
        _print_once(f'none -- {why}')
        return None
    energy = lum['energy'] * UNIT_ENERGY * \
        float(getattr(world, 'world_light_strength', 1.0))
    if energy <= 0.0:
        LAST_REPORT['text'] = 'No world light: World Light Strength is 0'
        return None
    shadow = str(getattr(world, 'world_light_shadow', 'MAP') or 'MAP')
    if shadow not in ('NONE', 'MAP', 'RAY'):
        shadow = 'MAP'
    toward = np.asarray(lum['dir'], np.float32)
    lt = Light(type='SUN', name=NAME,
               direction=tuple(float(-v) for v in toward),
               color=tuple(float(v) for v in lum['color']),
               energy=float(energy),
               radius=float(min(max(lum['angle'], 0.0), MAX_ANGLE)),
               shadow=shadow,
               volumetric=max(float(getattr(world, 'world_light_shafts',
                                            0.0)), 0.0))
    lt.world_light = True
    elev = float(np.degrees(np.arcsin(np.clip(float(toward[2]), -1, 1))))
    azim = float(np.degrees(np.arctan2(float(toward[1]), float(toward[0]))))
    words = {'SUN': 'the sky\'s sun', 'MOON': 'the sky\'s moon',
             'NODE': 'the Sky Texture\'s sun',
             'IMAGE': 'the sun found in the image',
             'MANUAL': 'the direction set by hand'}[lum['kind']]
    LAST_REPORT['text'] = (f'Lit by {words}: elevation {elev:.1f}, '
                           f'azimuth {azim:.1f} degrees')
    return lt


def is_world_light(light):
    return bool(getattr(light, 'world_light', False))


def sync(scene, textures=None):
    """Make `scene.lights` carry exactly the world's light: the synthetic
    one from the last sync dropped, a fresh one appended when
    `world.world_light` is on and the world has a luminary. Returns the
    light or None. A scene with the toggle off and no synthetic light is
    not touched at all."""
    world = getattr(scene, 'world', None)
    lights = getattr(scene, 'lights', None) or []
    want = world is not None and bool(getattr(world, 'world_light', False))
    had = any(is_world_light(l) for l in lights)
    if not want and not had:
        return None
    lt = make_light(world, textures) if want else None
    scene.lights = [l for l in lights if not is_world_light(l)] + \
        ([lt] if lt is not None else [])
    return lt
