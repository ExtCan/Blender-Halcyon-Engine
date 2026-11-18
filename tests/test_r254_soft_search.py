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
