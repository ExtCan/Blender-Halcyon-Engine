"""R254: the mapped soft shadow's blocker search is bounded per receiver.

An independent review of the per-receiver lookup (core/softmap.py, twin
gpu/softmap.py) found that a sized Sun over a large ground lost its
penumbra: the blocker search reached as far as the nearest depth of the
WHOLE map allowed, which under a tilted Sun is the near corner of the
ground, so its 13 taps stood units apart and stepped over the casters.
The same lamp and objects cast a different shadow on a larger ground, and
the Angle no longer softened it.

Each map now carries a REACH PYRAMID (`softmap.map_pyramid`): per level a
grid of tiles, each holding the smallest depth within the 3 x 3 tiles
around it. One texel answers "the nearest depth within r texels of this
receiver", the search's reach is tightened against it three times, and a
receiver with nothing nearer than itself within reach skips the search.
Each test below states one law as a number:

    a  the pyramid is a lower bound of every depth a tap can read, across
       cube seams and on ragged map sizes
    b  a Sun's shadow does not depend on how large the ground is, follows
       the traced one, and widens with the Angle
    c  a caster beside the lamp that shadows nothing in view does not
       move a point lamp's shadows
    d  a receiver with no caster in reach skips the search, and the skip
       changes no value
    e  the GPU twin shades the CPU picture; the pyramid rides the depth
       pack under the lamp's cells, keyed by layout and stamped by content
    f  lamps the lookup does not serve keep their lookup, text and pack
    g  a strip Area lamp's thin side stays sharp (SP-2): the filter's
       lattice takes a finer stride across a footprint's short side, and
       a round footprint keeps its one stride bit for bit
    h  a lamp at one Sample is hard and does not shrink under a Softness
       it cannot draw
    i  a scene smaller than a unit keeps its mapped shadows (SP-3): the
       Bias is bounded by the scene the map was baked for

    python -m halcyon.tests.test_r254_soft_search
"""
import sys
import traceback

import numpy as np

from . import utf8_console
from . import test_r254_map_soft as MS
from .scenebuild import plane, sphere
from ..core import lights as LI
from ..core import softmap as SM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def note(text):
    print(f'  note {text}')


# ------------------------------------------------------------------ helpers

SUN_FROM = (4.0, -3.0, 6.0)


def _character(ground):
    """The review's scene: a figure (a box and a head) and a crate on a
    ground plane of `ground` units."""
    return [plane(z=0.0, size=float(ground), mat=0, obj=0),
            MS._box((-0.3, -0.2, 0.0), (0.3, 0.2, 1.5)),
            sphere(centre=(0.0, 0.0, 1.75), radius=0.25, segs=24, rings=16,
                   mat=1, obj=1),
            MS._box((1.2, 0.6, 0.6), (1.8, 1.2, 1.2))]


def _sun_scene(angle, ground, map_size, mode='MAP', **stkw):
    st = MS._settings(shadow_map_size=int(map_size), **stkw)
    lt = MS._lamp('SUN', angle, mode=mode, pos=SUN_FROM, aim=(0.0, 0.0, 0.0))
    sc = MS._finish(_character(ground), lt, st, eye=(0.5, -7.0, 5.0),
                    at=(0.3, 0.8, 0.3))
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    return sc, st


_FLOOR = None


def _floor():
    """Floor points around the figure, on the side its shadow falls."""
    global _FLOOR
    if _FLOOR is None:
        gx, gy = np.meshgrid(np.linspace(-4.0, 3.0, 141),
                             np.linspace(-1.5, 4.5, 121))
        _FLOOR = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                          1).astype(np.float32)
    return _FLOOR


def _stats(v):
    """(umbra points, penumbra points, mass) of a shadow term."""
    return (int((v < 0.05).sum()), int(((v >= 0.05) & (v <= 0.95)).sum()),
            float((1.0 - v).sum()))


def _brute_min(fetch, u, v, r):
    """The smallest depth any tap within r texels of (u, v) can read,
    through the lookup's own fetcher (a cube's crosses its seams)."""
    xs = np.arange(np.floor(u - r), np.floor(u + r) + 1, dtype=np.float32)
    ys = np.arange(np.floor(v - r), np.floor(v + r) + 1, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    occ, _x, _y = fetch(gx.ravel(), gy.ravel())
    return float(occ.min())


# -------------------------------------------------------------------- tests


def test_a_the_pyramid_is_a_lower_bound():
    """Every cell bounds what a tap can read: 0 queries above the brute
    force minimum, on a Sun's, a Spot's and a Point's map, across the
    cube's seams, on a size that is no multiple of the tile."""
    rng = np.random.default_rng(5)
    for size in (64, 100):
        for kind in ('SUN', 'SPOT', 'POINT'):
            sc, st = MS._prep(kind, 0.3, shadow_map_size=size)
            sm = sc.lights[0].shadow_map
            blocks = SM.map_pyramid(sm)
            lay = SM.pyr_layout(size)
            faces = getattr(sm, 'faces', None)
            bad, n, exact = 0, 0, 0
            for fi in range(len(blocks)):
                fetch = SM._MapFetch(sm) if faces is None \
                    else SM._CubeFetch(faces, fi)
                for _ in range(120):
                    u = f32(rng.uniform(0, size))
                    v = f32(rng.uniform(0, size))
                    r = f32(rng.choice([3.0, 4.0, 4.5, 8.0, 13.0, 31.0,
                                        size * 0.3]))
                    q = float(SM._reach_min(blocks[fi], lay, np.array([u]),
                                            np.array([v]), np.array([r]))[0])
                    # a tap lands on floor(u + offset), |offset| <= r - 1
                    b = _brute_min(fetch, u, v, r - 1.0)
                    n += 1
                    bad += q > b
                    exact += q == b
            check(f'{kind} map {size}: no cell of the reach pyramid is '
                  'above the nearest depth a tap within its reach reads',
                  bad == 0 and exact > 0,
                  f'{bad} of {n} above, {exact} exact; '
                  f'{len(lay["levels"])} levels in a block '
                  f'{lay["w"]} x {lay["h"]}')
    lay = SM.pyr_layout(512)
    check('the pyramid starts at tiles of 4 texels and is a twelfth of '
          'the map',
          lay['levels'][0][0] == 4 and lay['levels'][-1][1] == 1
          and lay['w'] * lay['h'] < 512 * 512 / 9,
          f'{lay["w"]} x {lay["h"]} cells for a 512 face')
    sc, st = MS._prep('POINT', 0.3, shadow_map_size=64)
    sm = sc.lights[0].shadow_map
    first = SM.map_pyramid(sm)
    check('it is built once per map and parked on it (the `pyr` slot)',
          SM.map_pyramid(sm) is first and sm.pyr is first
          and 'pyr' in LI.ShadowMap.__slots__
          and 'pyr' in LI.CubeShadow.__slots__)
    sm.pyr = None
    again = SM.map_pyramid(sm)
    check('a reset slot rebuilds the same pyramid from the depth',
          again is not first
          and all(np.array_equal(a, b) for a, b in zip(again, first)))


def test_b_a_sun_shadow_does_not_depend_on_the_ground():
    """SP-1. The same Sun, the same objects, the same world size of a
    shadow texel (ground 20 on a 512 map, ground 80 on a 2048 map)."""
    P = _floor()
    sc, st = _sun_scene(0.2, 20.0, 512)
    v20 = MS._vis(sc, st, P)
    ref = MS._ref(sc, st, P, nsamp=64)
    sc, st = _sun_scene(0.2, 80.0, 2048)
    v80 = MS._vis(sc, st, P)
    s20, s80, sr = _stats(v20), _stats(v80), _stats(ref)
    moved = np.abs(v80 - v20)
    check('a Sun of 0.2 rad casts the same shadow on a ground of 20 and '
          'of 80 units (it lost 13 % of its mass and half its penumbra)',
          abs(s80[2] - s20[2]) < 0.02 * s20[2]
          and int((moved > 0.25).sum()) == 0
          and abs(s80[1] - s20[1]) < 0.05 * s20[1],
          f'umbra / penumbra / mass {s20[0]} / {s20[1]} / {s20[2]:.1f} '
          f'on 20, {s80[0]} / {s80[1]} / {s80[2]:.1f} on 80; '
          f'{int((moved > 0.25).sum())} points moved by more than 0.25, '
          f'{int((moved > 0.1).sum())} by more than 0.1')
    check('and it is the traced shadow within 5 % of its mass on both',
          abs(s20[2] - sr[2]) < 0.05 * sr[2]
          and abs(s80[2] - sr[2]) < 0.05 * sr[2],
          f'traced {sr[0]} / {sr[1]} / {sr[2]:.1f}')
    pens = []
    for ang in (0.05, 0.1, 0.2):
        sc, st = _sun_scene(ang, 80.0, 2048)
        pens.append(_stats(MS._vis(sc, st, P))[1])
    check('on the large ground a larger Angle gives a wider penumbra (it '
          'stayed at one width)',
          pens[0] < 0.9 * pens[1] and pens[1] < 0.9 * pens[2],
          f'penumbra points at 0.05 / 0.1 / 0.2 rad: {pens}')
    # what the search does on the open ground, far from the figure
    stats = {}
    sm = sc.lights[0].shadow_map
    far = np.array([[2.5, -1.0, 0.0]], np.float32)
    n = np.array([[0.0, 0.0, 1.0]], np.float32)
    seen = []
    orig = SM._reach_min

    def spy(block, lay, u, v, r):
        seen.append(float(np.asarray(r).ravel()[0]))
        return orig(block, lay, u, v, r)
    SM._reach_min = spy
    try:
        SM.lookup(sm, SM.lamp_shape(sc.lights[0]), far, n,
                  np.array([0.7], np.float32), np.array([0.03], np.float32),
                  1.0, 4, stats=stats)
    finally:
        SM._reach_min = orig
    check("away from the figure the search comes from the cap's 164 "
          'texels down to a few, a texel fetched per step',
          2 < len(seen) <= SM.SEARCH_STEPS and seen[0] > 150.0
          and seen[-1] < 8.0 and stats.get('asked') == len(seen),
          f'reach per step, texels: {[round(r, 1) for r in seen]}')


def _fixture_scene(radius, fix, map_size=512):
    """Poles, a rail and a ball under a point lamp; `fix` adds a small
    plate just ABOVE the lamp, shadowing nothing on the floor."""
    lp = (0.0, -2.5, 4.0)
    parts = [plane(z=0.0, size=16.0, mat=0, obj=0)]
    for i, wd in enumerate((0.05, 0.1, 0.2, 0.4)):
        x = -2.4 + 1.5 * i
        parts.append(MS._box((x - wd / 2, -wd / 2, 0.0),
                             (x + wd / 2, wd / 2, 2.0)))
    parts.append(MS._box((-3.0, 1.2, 1.0), (3.0, 1.3, 1.1)))
    parts.append(sphere(centre=(2.8, -0.8, 0.3), radius=0.3, mat=1, obj=1))
    if fix:
        parts.append(MS._box((lp[0] - 0.2, lp[1] - 0.2, lp[2] + 0.30),
                             (lp[0] + 0.2, lp[1] + 0.2, lp[2] + 0.32)))
    st = MS._settings(shadow_map_size=map_size)
    lt = MS._lamp('POINT', radius, pos=lp, aim=(0.0, 0.5, 0.0))
    sc = MS._finish(parts, lt, st, eye=(0.0, -6.5, 3.6), at=(0.0, 0.8, 0.3))
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    return sc, st


def test_c_a_caster_beside_the_lamp_moves_nothing():
    """The same mechanism under a point lamp: a ceiling rose 0.3 above
    the lamp is the nearest depth of the whole cube, and used to widen
    the search of every receiver to the cap."""
    gx, gy = np.meshgrid(np.linspace(-4.0, 4.0, 161),
                         np.linspace(-1.0, 5.0, 121))
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                 1).astype(np.float32)
    for radius in (0.1, 0.25):
        sc, st = _fixture_scene(radius, False)
        a = MS._vis(sc, st, P)
        sc, st = _fixture_scene(radius, True)
        b = MS._vis(sc, st, P)
        d = np.abs(a - b)
        check(f'POINT radius {radius}: a plate above the lamp leaves the '
              "floor's shadows where they were",
              int((d > 0.25).sum()) <= 1 and int((d > 0.05).sum()) <= 40,
              f'{int((d > 0.05).sum())} floor points moved by more than '
              f'0.05, {int((d > 0.25).sum())} by more than 0.25, largest '
              f'{float(d.max()):.3f}')


def test_d_no_caster_in_reach_skips_the_search():
    """A receiver the pyramid shows nothing nearer than itself within
    reach of is lit without one search tap, and exactly so."""
    gx, gy = np.meshgrid(np.linspace(-6.0, 6.0, 121),
                         np.linspace(-6.0, 6.0, 121))
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                 1).astype(np.float32)
    N = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (P.shape[0], 1))

    def run(sc, st, shape=None):
        lt = sc.lights[0]
        L, _rad, dist = LI.sample(lt, P, st)
        ndl = np.clip((N * L).sum(1), 0.0, 1.0).astype(np.float32)
        stats = {}
        out = SM.lookup(lt.shadow_map, shape or SM.lamp_shape(lt), P, N, ndl,
                        0.02 * (1.0 + 2.0 * (1.0 - ndl)), 1.0, 4,
                        stats=stats)
        return out, stats

    for kind, size in (('POINT', 0.25), ('AREA', 1.0), ('SPOT', 0.25),
                       ('SUN', 0.05)):
        st = MS._settings()
        # straight down, so the floor stands at one depth of each map
        # (an Area lamp bakes a cube whatever its aim: a hair off the
        # vertical gives it its two axes)
        lt = MS._lamp(kind, size, pos=(0.0, 0.0, 6.0),
                      aim=(0.0, 1e-3 if kind == 'AREA' else 0.0, 0.0))
        sc = MS._wall(lt, st)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        out, stats = run(sc, st)
        SM.SEARCH_SKIP = False
        try:
            full, stats_full = run(sc, st)
        finally:
            SM.SEARCH_SKIP = True
        share = stats.get('skip', 0) / max(stats.get('bound', 0), 1)
        check(f'{kind} over a floor: most receivers skip the search',
              share > 0.6 and stats.get('search', 0)
              == stats['bound'] - stats['skip'],
              f'{stats.get("skip", 0)} of {stats.get("bound", 0)} points '
              f'({share:.0%}); searched {stats.get("search", 0)} against '
              f'{stats_full.get("search", 0)} without the skip')
        check(f'{kind}: the skip changes no value',
              np.array_equal(out, full))
    # a lamp of size 0 under the user's Softness: the footprint has no
    # lamp term, so the search's taps never depended on its reach. With
    # the skip it is bitwise the lookup without a pyramid at all
    orig = SM._face

    def bare(*a, **k):
        return orig(*a[:13])
    for kind in ('POINT', 'SPOT', 'SUN', 'AREA'):
        st = MS._settings(shadow_softness=4.0)
        lt = MS._lamp(kind, 0.0, pos=(1.0, -3.0, 6.0))
        if kind == 'AREA':
            lt.area_size = (0.0, 0.0)
        sc = MS._wall(lt, st)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        out, stats = run(sc, st, SM.zero_shape(lt))
        SM._face = bare
        try:
            old, _s = run(sc, st, SM.zero_shape(lt))
        finally:
            SM._face = orig
        check(f'{kind} of size 0 at Softness 4: the same values as the '
              'search bounded by the whole map, bit for bit',
              np.array_equal(out, old)
              and (stats.get('skip', 0) > 0 or kind in ('SPOT', 'SUN')),
              f'{stats.get("skip", 0)} of {stats.get("bound", 0)} skipped')


def _pack(atlases):
    key, build, stamp = atlases['hal_shadowpack']
    return key, build(), stamp


def test_e_the_gpu_twin():
    """Two roads, one picture, on both value roads; the pyramid is rows
    of the depth pack."""
    cases = []
    st = MS._settings(shadow_map_size=1024)
    lt = MS._lamp('SUN', 0.2, pos=SUN_FROM, aim=(0.0, 0.0, 0.0))
    cases.append(('SUN 0.2 on a ground of 80 units',
                  MS._finish(_character(80.0), lt, st, eye=(0.5, -7.0, 5.0),
                             at=(0.3, 0.8, 0.3)), st))
    sc, st = _fixture_scene(0.25, True, 256)
    cases.append(('POINT 0.25 with a plate above it (six faces)', sc, st))
    st = MS._settings(shadow_samples=16)
    cases.append(('SPOT 0.3 at Samples 16',
                  MS._wall(MS._lamp('SPOT', 0.3), st), st))
    st = MS._settings(shadow_softness=3.0)
    cases.append(('AREA 1.5 x 0.5 at Softness 3',
                  MS._wall(MS._lamp('AREA', (1.5, 0.5)), st), st))
    st = MS._settings(shadow_softness=4.0)
    cases.append(('POINT of size 0 at Softness 4',
                  MS._wall(MS._lamp('POINT', 0.0), st), st))
    for label, sc, st in cases:
        worst = []
        for texels in (True, False):
            GSH.SHADOW_TEXELS = texels
            try:
                LI.clear_shadow_cache()
                cpu, sim, cov, passes, atlases, why = MS._gpu(sc, st)
            finally:
                GSH.SHADOW_TEXELS = True
            if not check(f'{label}: the deferred pass plans '
                         f'({"texels" if texels else "literals"})',
                         sim is not None, str(why)):
                continue
            d = np.abs(sim[cov] - cpu[cov][:, :3]).max(axis=1)
            worst.append(float(d.max()))
            src = MS._sources(passes)
            if texels:
                kept = (src, atlases)
        if len(worst) == 2:
            check(f'{label}: the simulator shades the CPU picture on both '
                  'value roads', max(worst) < 6e-3,
                  f'max |sim - cpu| {worst[0]:.6f} (texels), '
                  f'{worst[1]:.6f} (literals)')
    # the pack: the lamp's depth cells, then its pyramid's rows
    sc, st = _fixture_scene(0.25, False, 128)
    cpu, sim, cov, passes, atlases, why = MS._gpu(sc, st)
    key, px, stamp = _pack(atlases)
    sm = sc.lights[0].shadow_map
    lay = SM.pyr_layout(128)
    blocks = SM.map_pyramid(sm)
    check("a sized lamp's band of the pack is its six faces and, under "
          'them, the six blocks of its reach pyramid',
          px.ndim == 2 and px.shape == (2 * 128 + lay['h'], 3 * 128)
          and all(np.array_equal(
              px[(fi // 3) * 128:(fi // 3 + 1) * 128,
                 (fi % 3) * 128:(fi % 3 + 1) * 128], f.depth)
              for fi, f in enumerate(sm.faces))
          and all(np.array_equal(
              px[256:256 + lay['h'], fi * lay['w']:(fi + 1) * lay['w']], b)
              for fi, b in enumerate(blocks)), str(px.shape))
    check('that the rows exist is layout (the key says PYR); the stamp is '
          "still the faces' content digests",
          key == ('shadowpack', ('shadow', 128, (3, 2), 'CLS', 'PYR'))
          and stamp == (tuple(GSH.map_digest(f) for f in sm.faces),),
          str(key))
    src = MS._sources(passes)
    check('the text reads the pyramid through one function, at most '
          'SEARCH_STEPS times, and branches on what it finds',
          'float hal_sm_near0(' in src and 'if (near < lim) {' in src
          and src.count('if (near < lim && more > 0.5) {')
          == (SM.SEARCH_STEPS - 1) * src.count('float hal_sm_near0('),
          f"{src.count('hal_sm_near0(')} mentions in "
          f"{src.count('float hal_sm_near0(')} pass texts")
    # a moved caster re-stamps the same key, and the pyramid follows it
    sc2, st2 = _fixture_scene(0.25, True, 128)
    _c, _s, _cov, _p, atlases2, _w = MS._gpu(sc2, st2)
    key2, px2, stamp2 = _pack(atlases2)
    check('a caster added to the scene keeps the key, moves the stamp and '
          'the pyramid rows',
          key2 == key and stamp2 != stamp
          and not np.array_equal(px2[256:], px[256:]))
    # the dial that sends a lamp here is structure of its own
    sc3, st3 = _fixture_scene(0.25, False, 128)
    sc3.lights[0].shadow_samples = 1
    _c, _s, _cov, passes3, atlases3, _w = MS._gpu(sc3, st3)
    key3, px3, _st3 = _pack(atlases3)
    check('the same lamp at Samples 1 (the hard lookup) packs no pyramid',
          key3 == ('shadowpack', ('shadow', 128, (3, 2), 'CLS'))
          and px3.shape == (2 * 128, 3 * 128)
          and 'hal_sm_near' not in MS._sources(passes3), str(key3))


def test_f_other_lamps_keep_their_lookup():
    """Default neutrality: a lamp of size 0 at Softness 1 or less, a lamp
    at Samples 1 and a traced lamp never reach the pyramid."""
    built = []
    orig = SM._build_pyramid

    def counting(sm):
        built.append(sm)
        return orig(sm)
    SM._build_pyramid = counting
    try:
        for label, kind, size, lkw, skw in (
                ('size 0 at Softness 1', 'POINT', 0.0, {}, {}),
                ('size 0 at Softness 0.5', 'SUN', 0.0, {},
                 {'shadow_softness': 0.5}),
                ('a sized lamp at Samples 1', 'SPOT', 0.4, {},
                 {'shadow_samples': 1}),
                ('a traced lamp', 'POINT', 0.3, {'shadow': 'RAY'}, {})):
            st = MS._settings(**skw)
            sc = MS._wall(MS._lamp(kind, size, **lkw), st)
            LI.clear_shadow_cache()
            cpu, sim, cov, passes, atlases, why = MS._gpu(sc, st)
            src = MS._sources(passes) if passes else ''
            meta, _a, _w = GSH._shadow_meta(sc.lights[0], st)
            pack = atlases.get('hal_shadowpack') if atlases else None
            check(f'{label}: no pyramid is built, packed or read',
                  not built and 'hal_sm_near' not in src
                  and (meta is None or 'pyr_rows' not in meta)
                  and (pack is None or 'PYR' not in pack[0][1]),
                  f'{len(built)} built')
    finally:
        SM._build_pyramid = orig


def _ball_penumbra(lamp, st):
    """The shadow term over the lit cap of a ball standing in a box's
    penumbra: (share of its penumbra points whose 3 x 3 neighbourhood
    does not change at all, largest neighbour step)."""
    sc = MS._finish([plane(z=0.0, size=14.0, mat=0, obj=0),
                     MS._box((-0.6, -0.6, 1.0), (0.6, 0.6, 2.2)),
                     sphere(centre=(-2.2, 0.8, 0.6), radius=0.6, mat=1,
                            obj=1)], lamp, st)
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    c = np.array([-2.2, 0.8, 0.6])
    to = np.asarray(lamp.position, np.float64) - c
    to /= np.linalg.norm(to)
    e1 = np.cross(to, [0.0, 0.0, 1.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, to)
    g = np.linspace(-0.75, 0.75, 97)
    a, b = np.meshgrid(g, g)
    n = to[None, None, :] * np.sqrt(np.maximum(
        1.0 - a * a - b * b, 0.0))[..., None] \
        + a[..., None] * e1 + b[..., None] * e2
    pb = (c + 0.6 * n).reshape(-1, 3).astype(np.float32)
    nb = n.reshape(-1, 3).astype(np.float32)
    lb, _rad, db = LI.sample(lamp, pb, st)
    t = LI.visibility(lamp, pb, nb, lb.astype(np.float32), db,
                      st).reshape(a.shape)
    pad = np.pad(t, 1, mode='edge')
    nine = np.stack([pad[i:i + t.shape[0], j:j + t.shape[1]]
                     for i in range(3) for j in range(3)], 0)
    span = nine.max(0) - nine.min(0)
    pen = (t > 0.05) & (t < 0.95)
    step = max(float(np.abs(np.diff(t, axis=0)).max()),
               float(np.abs(np.diff(t, axis=1)).max()))
    return float((span[pen] < 1e-4).mean()), int(pen.sum()), step


def test_g_a_strip_lamp_keeps_its_thin_side_sharp():
    """SP-2. An Area lamp 0.2 x 2.0: the shadow edge ACROSS its thin side
    is as narrow as the traced one, the thin Size moves it, and the
    ball in a box's penumbra carries no stripes."""
    strip = {'area_shape': 'RECTANGLE'}
    sc, st = MS._prep('AREA', (0.2, 2.0), lamp_kw=strip)
    w = MS._widths(sc, st)
    r = MS._widths(sc, st, MS._ref)
    ok = all(abs(a - b) <= 0.35 * b + 0.02 for a, b in zip(w[1:], r[1:])) \
        and w[0] <= r[0] + 0.05
    check('a 0.2 x 2.0 strip: the edge across its thin side is the traced '
          'width within a third (it was 2 to 3 times as wide)', ok,
          'mapped ' + ' '.join(f'{x:.3f}' for x in w) + ', traced '
          + ' '.join(f'{x:.3f}' for x in r))
    mid = []
    for sx in (0.1, 0.8):
        sc, st = MS._prep('AREA', (sx, 2.0), lamp_kw=strip)
        mid.append((MS._widths(sc, st)[2], MS._widths(sc, st, MS._ref)[2]))
    grow_map = mid[1][0] / mid[0][0]
    grow_ray = mid[1][1] / mid[0][1]
    check('the thin Size from 0.1 to 0.8 widens that edge as the traced '
          'road does (it grew by half of that)',
          grow_map > 0.7 * grow_ray,
          f'mapped x{grow_map:.2f} ({mid[0][0]:.3f} -> {mid[1][0]:.3f}), '
          f'traced x{grow_ray:.2f} ({mid[0][1]:.3f} -> {mid[1][1]:.3f})')
    # the long side is still as soft as it was
    sc, st = MS._prep('AREA', (2.0, 0.2), lamp_kw=strip)
    w = MS._widths(sc, st)
    r = MS._widths(sc, st, MS._ref)
    check('turned by a quarter, its long side across the edge: the '
          'traced width within a sixth',
          all(abs(a - b) <= 0.16 * b + 0.02 for a, b in zip(w, r)),
          'mapped ' + ' '.join(f'{x:.3f}' for x in w) + ', traced '
          + ' '.join(f'{x:.3f}' for x in r))
    # a curved receiver: no flat patches (the reason the lattice is as
    # wide as it is; a stride taken from the short radius alone left
    # half of this penumbra flat)
    for label, size in (('0.2 x 2.0', (0.2, 2.0)), ('2.0 x 0.2', (2.0, 0.2))):
        st = MS._settings()
        lamp = MS._lamp('AREA', size, pos=(2.5, -1.5, 5.0),
                        aim=(0.0, 0.0, 0.8), **strip)
        flat, npen, step = _ball_penumbra(lamp, st)
        check(f'strip {label} on a ball in a penumbra: a gradient, no '
              'stripes', flat < 0.01 and step < 0.2 and npen > 1000,
              f'{flat:.2%} of {npen} penumbra points flat, largest '
              f'neighbour step {step:.3f}')
    # round footprints take the one stride they always did
    gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 71),
                         np.linspace(0.2, 6.5, 64))
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                 1).astype(np.float32)
    for kind, size, lkw, same in (('POINT', 0.5, {}, True),
                                  ('SPOT', 0.5, {}, True),
                                  ('SUN', 0.1, {}, True),
                                  ('AREA', 1.0, {}, True),
                                  ('AREA', 1.0, {'area_shape': 'DISK'},
                                   True),
                                  ('AREA', (0.2, 2.0), strip, False)):
        sc, st = MS._prep(kind, size, lamp_kw=lkw)
        a = MS._vis(sc, st, P)
        SM.ANISO = False
        try:
            b = MS._vis(sc, st, P)
        finally:
            SM.ANISO = True
        check(f'{kind} {size} {lkw.get("area_shape", "")}: '
              + ('one stride, bit for bit' if same
                 else 'a finer stride across its short side'),
              np.array_equal(a, b) == same,
              f'{int((a != b).sum())} of {a.size} points differ')
    # both roads
    for size, skw in (((0.2, 2.0), {}), ((2.0, 0.2), {}),
                      ((0.25, 3.0), {'shadow_samples': 16})):
        worst = []
        for texels in (True, False):
            GSH.SHADOW_TEXELS = texels
            try:
                st = MS._settings(**skw)
                sc = MS._wall(MS._lamp('AREA', size, **strip), st)
                LI.clear_shadow_cache()
                cpu, sim, cov, passes, _atl, why = MS._gpu(sc, st)
            finally:
                GSH.SHADOW_TEXELS = True
            if sim is None:
                worst.append(1e9)
                continue
            worst.append(float(np.abs(sim[cov] - cpu[cov][:, :3]).max()))
        check(f'strip {size} {skw or ""}: the simulator shades the CPU '
              'picture on both value roads', max(worst) < 6e-3,
              f'max |sim - cpu| {worst[0]:.6f} (texels), {worst[1]:.6f} '
              '(literals)')


def test_h_one_sample_does_not_shrink_under_softness():
    """One Sample is the hard single-tap lookup for every lamp, and one
    tap cannot blur: the Softness must not move it. The normal offset
    carried max(1, Softness), which lifted the receiver off its surface
    and shrank the hard shadow."""
    P, cell = MS._floor_grid()
    for kind, size in (('POINT', 0.0), ('SPOT', 0.0), ('SUN', 0.0),
                       ('POINT', 0.25)):
        first, same, masses = None, True, []
        for soft in (1.0, 4.0, 8.0, 16.0):
            sc, st = MS._prep(kind, size, shadow_samples=1,
                              shadow_softness=soft)
            v = MS._vis(sc, st, P)
            masses.append(float((1.0 - v).sum() * cell))
            if first is None:
                first = v
            same = same and np.array_equal(v, first)
        check(f'{kind} size {size} at Samples 1: Softness 1 / 4 / 8 / 16 '
              'is one hard shadow, bit for bit', same
              and len(np.unique(first)) == 2,
              'mass ' + ' / '.join(f'{m:.2f}' for m in masses))
    # the lamp's own dials: Samples 1 on the lamp, Softness 8 on the lamp
    sc, st = MS._prep('POINT', 0.0, lamp_kw={'shadow_samples': 1})
    a = MS._vis(sc, st, P)
    sc, st = MS._prep('POINT', 0.0, lamp_kw={'shadow_samples': 1,
                                             'shadow_softness': 8.0})
    b = MS._vis(sc, st, P)
    check("the lamp's own Samples 1 and Softness 8: the same shadow",
          np.array_equal(a, b))
    # the other road: the factor rides its texel (and is a literal on
    # the literal road), so the text does not move with the Softness
    texts = {}
    for texels in (True, False):
        for soft in (1.0, 8.0):
            GSH.SHADOW_TEXELS = texels
            try:
                st = MS._settings(shadow_samples=1, shadow_softness=soft)
                sc = MS._wall(MS._lamp('POINT', 0.0), st)
                LI.clear_shadow_cache()
                cpu, sim, cov, passes, atlases, why = MS._gpu(sc, st)
            finally:
                GSH.SHADOW_TEXELS = True
            d = float(np.abs(sim[cov] - cpu[cov][:, :3]).max()) \
                if sim is not None else 1e9
            texts[(texels, soft)] = MS._sources(passes) if passes else ''
            check(f'Samples 1 at Softness {soft:g} '
                  f'({"texels" if texels else "literals"}): the simulator '
                  'shades the CPU picture', d < 1e-4, f'max {d:.6f}')
            if texels and soft == 8.0:
                tex = atlases['hal_lights'][1]() if callable(
                    atlases['hal_lights'][1]) else atlases['hal_lights'][1]
                tex = np.asarray(tex)
                check("the constant lookup's offset factor is the texel "
                      'h2.y = 1 at any Softness (it was max(1, Softness))',
                      float(tex[0, 16 + 2, 1]) == 1.0
                      and float(tex[0, 16 + 2, 2]) == 8.0,
                      str(tex[0, 16 + 2]))
    check('the text of a hard lamp does not move with the Softness on '
          'either value road',
          texts[(True, 1.0)] == texts[(True, 8.0)]
          and texts[(False, 1.0)] == texts[(False, 8.0)]
          and 'hal_shadow_vis0' in texts[(True, 1.0)])
    # no shipped preset pairs mapped shadows at one Sample with a
    # Softness above 1: none of their pictures moves
    from ..presets import library as PL
    paired = [k for k, v in PL.PRESETS.items()
              if isinstance(v, dict)
              and int(v.get('settings', {}).get('shadow_samples', 4)) == 1
              and float(v.get('settings', {}).get('shadow_softness', 1.0))
              > 1.0
              and v.get('settings', {}).get('shadow_default', 'MAP')
              != 'RAY']
    check('no shipped preset pairs mapped shadows at Samples 1 with a '
          'Softness above 1', not paired and len(PL.PRESETS) > 20,
          f'{len(PL.PRESETS)} presets, paired: {paired}')


def _scaled_wall(scale, kind, size, **stkw):
    """The wall scene with every length times `scale` (a Sun's angle is
    no length), the lamp's energy scaled to keep the picture."""
    st = MS._settings(**stkw)
    k = float(scale)
    lt = MS._lamp(kind, size if kind == 'SUN' else size * k,
                  pos=tuple(c * k for c in MS.LAMP_POS),
                  aim=tuple(c * k for c in MS.LAMP_AIM))
    if kind != 'SUN':
        lt.energy = lt.energy * k * k
    sc = MS._finish([plane(z=0.0, size=14.0 * k, mat=0, obj=0),
                     MS._box((-0.6 * k, -0.15 * k, 0.0),
                             (0.6 * k, 0.15 * k, 3.0 * k))], lt, st,
                    eye=(7.0 * k, -3.5 * k, 7.0 * k),
                    at=(0.3 * k, 1.9 * k, 0.0))
    sc.camera.clip_start = 0.1 * k
    sc.camera.clip_end = 200.0 * k
    return sc, st


def test_i_a_small_scene_keeps_its_mapped_shadows():
    """SP-3. The default Bias is 0.02 WORLD units: in a scene modelled
    at centimetre size it is larger than the casters stand off their
    receivers, and every mapped shadow was gone."""
    import contextlib
    import io
    P1, cell = MS._floor_grid()
    for kind, size in (('POINT', 0.0), ('POINT', 0.25), ('SUN', 0.05),
                       ('SPOT', 0.0)):
        mass = {}
        for scale in (1.0, 0.01):
            sc, st = _scaled_wall(scale, kind, size)
            LI.clear_shadow_cache()
            LI.build_shadow_maps(sc, st)
            if scale == 1.0:
                bare = MS._vis(sc, st, P1)
            SM.stamp_scene(sc)
            v = MS._vis(sc, st, (P1 * f32(scale)).astype(np.float32))
            mass[scale] = float((1.0 - v).sum() * cell)
            if scale == 1.0:
                same = np.array_equal(v, bare)
                radius = float(sc.lights[0].shadow_map.scene_radius)
        check(f'{kind} size {size}: the scene at a hundredth of its size '
              'casts the same shadow (it cast none)',
              abs(mass[0.01] - mass[1.0]) < 0.12 * mass[1.0],
              f'mass {mass[1.0]:.2f} at scale 1, {mass[0.01]:.2f} at 0.01 '
              '(in the scaled floor grid)')
        check(f'{kind} size {size}: at a radius of {radius:.1f} units the '
              "Bias is the dial's own, bit for bit", same and radius > 1.0)
    # the rule itself
    sc, st = _scaled_wall(0.01, 'POINT', 0.0)
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    sm = sc.lights[0].shadow_map
    unstamped = SM.scene_bias(sm, 0.02)
    SM.stamp_scene(sc)
    r = float(sm.scene_radius)
    check('under a scene radius of one unit the Bias scales with the '
          'radius (the default is then 2 % of it); a map nobody stamped '
          'keeps the dial',
          unstamped == 0.02 and 0.05 < r < 0.2
          and SM.scene_bias(sm, 0.02) == 0.02 * r
          and SM.scene_bias(sm, 0.0) == 0.0, f'radius {r:.4f}')
    sm.scene_radius = 1.0
    check('at exactly one unit it is the dial',
          SM.scene_bias(sm, 0.02) == 0.02)
    # the render path stamps the maps it bakes and says so once
    for scale, want in ((0.01, True), (1.0, False)):
        sc, st = _scaled_wall(scale, 'POINT', 0.0)
        LI.clear_shadow_cache()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            img = np.asarray(MS.R.render(sc, st))
        st2 = MS._settings(shadows=False)
        sc2, _st = _scaled_wall(scale, 'POINT', 0.0)
        with contextlib.redirect_stdout(io.StringIO()):
            ref = np.asarray(MS.R.render(sc2, st2))
        lost = float((ref[..., :3] - img[..., :3]).sum())
        said = 'Bias' in out.getvalue() and 'scaled to the scene' \
            in out.getvalue()
        check(f'a frame at scale {scale:g}: the shadow is in the picture'
              + (', and the console says the Bias was scaled' if want
                 else ', and the console says nothing about the Bias'),
              lost > 1.0 and said == want
              and getattr(sc.lights[0].shadow_map, 'scene_radius', None)
              is not None, f'light the shadow removes: {lost:.1f}')
    # both roads read one Bias
    for kind, size in (('POINT', 0.0), ('POINT', 0.25), ('SUN', 0.05)):
        sc, st = _scaled_wall(0.01, kind, size)
        LI.clear_shadow_cache()
        with contextlib.redirect_stdout(io.StringIO()):
            cpu, sim, cov, passes, atlases, why = MS._gpu(sc, st)
        meta, _a, _w = GSH._shadow_meta(sc.lights[0], st)
        d = float(np.abs(sim[cov] - cpu[cov][:, :3]).max()) \
            if sim is not None else 1e9
        check(f'{kind} size {size} at scale 0.01: the simulator shades '
              'the CPU picture with the scaled Bias',
              d < 6e-3 and meta is not None
              and 0.0 < meta['bias'] < 0.005,
              f'max |sim - cpu| {d:.6f}, Bias {meta["bias"]:.5f}'
              if meta else str(why))


def main():
    utf8_console()
    FAILS.clear()
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for t in tests:
        print(t.__name__)
        try:
            t()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(f'{t.__name__} raised')
    print()
    if FAILS:
        print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS))
        return 1
    print('all R254 soft-search checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
