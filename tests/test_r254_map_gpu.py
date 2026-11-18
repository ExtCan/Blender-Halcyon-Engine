"""R254 map-gpu: what the Softness dial does to a mapped shadow, what the
device keeps resident, and what an edit costs.

The field reports: 'the lights don't get softer / harder with size ... some
of the general options just don't work properly' and 'Shadow map is slower
than ray traced shadows' (a GPU render device, the rendered viewport). Three
things were behind the mapped half of them, and each test states one law as
a number that the tree before this package fails:

    a  Softness above one texel BLURS a size-0 lamp's shadow. It shrank
       it and, at 16 texels, erased it (the lookup point was lifted off
       the surface by the blur's radius); at one texel or less, and at
       Samples 1, the constant lookup is untouched, bit for bit
    b  ... on the GPU road too: the simulator twin shades the CPU picture
    c  the device never keeps an old shadow atlas: a caster nudged
       between the rows the old fingerprint sampled is drawn where it is
    d  a lamp edit is a plan-cache HIT: nudging a mapped lamp, dragging
       its size, Softness, Bias, Density or Shadow Colour, or a traced
       lamp's size compiles nothing and draws the right frame
    e  structure still re-plans, and only structure
    f  the texel road draws the literal road's frame, and the texels are
       the float32 values the CPU computes with
    g  the pack is ONE channel (R32F) on the device, the fake device and
       the simulator
    h  the plan cache keeps a draft's and a refine's plan through a drag
       and holds no map alive
    i  the Samples tooltip says which lookup a Softness selects

    python -m halcyon.tests.test_r254_map_gpu
"""
import copy
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from . import fakedevice
from . import test_r254_map_soft as MS
from .scenebuild import plane
from ..core import lights as LI
from ..core import raster as CRg
from ..core import render as R
from ..core import softmap as SM
from ..core.bvh import BVH
from ..gpu import device as DEV
from ..gpu import material as GM
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


# ---------------------------------------------------------------- helpers


def _job(sc, st, w=96, h=72, bvh=None):
    view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = CRg.GBuffer(w, h)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    return R.ShadeJob(sc, st, {}, bvh, view, eye, w, h), g


def _sim(sc, st, w=96, h=72, ray=False):
    """(cpu frame, simulated frame, covered mask, passes, atlases, why):
    the CPU road's frame and the GLSL of the plan through the simulator."""
    LI.clear_shadow_cache()
    R._GBUF_CACHE.clear()
    cpu = R.render(sc, st)
    bvh = BVH.cached(sc.mesh.verts, sc.mesh.tris) if ray else None
    job, g = _job(sc, st, w, h, bvh)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    if passes is None:
        return cpu, None, g.tri >= 0, None, None, why
    img, _hit = GSH.simulate(job, g, passes, atlases)
    return cpu, img, g.tri >= 0, passes, atlases, None


def _cpu_frame(sc, st):
    """The CPU road's frame of the scene as it stands (its own Scene
    copy, so the GPU road's lamps keep the maps they hold)."""
    st2 = copy.copy(st)
    st2.render_device = 'CPU'
    return R.render(copy.deepcopy(sc), st2)


def _fresh():
    LI.clear_shadow_cache()
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()


def _edge(sc, st, y=2.5):
    """(10-90 % width, 50 % crossing, grey levels, largest step) of the
    shadow edge on the floor line y; the width is None without a full
    transition."""
    v = MS._vis(sc, st, MS._xline(y)).astype(np.float64)
    e = MS._edge(MS.XS, v)
    step = float(np.abs(np.diff(v)).max())
    if e is None:
        return None, None, 0, step
    t = (v - v.min()) / (v.max() - v.min())
    x = MS.XS
    if t[-1] < t[0]:
        t, x = t[::-1], x[::-1]
    i = int(np.nonzero(t >= 0.5)[0][0])
    x50 = float(x[i]) if i == 0 or t[i] == t[i - 1] else float(
        x[i - 1] + (0.5 - t[i - 1]) / (t[i] - t[i - 1]) * (x[i] - x[i - 1]))
    return float(e[0]), x50, int(e[1]), step


# ------------------------------------------- a. Softness blurs, it does not shrink


def test_a_softness_blurs_a_size_zero_lamp():
    """soft-erode-01 / size-map-v01: `offset = texel * (1.5 + 2.5 sin) *
    max(1, Softness)` lifted the lookup point off the floor by the blur's
    radius. POINT, 512 map, Softness 1 / 4 / 8 / 16: mass 7.47 / 6.15 /
    4.73 / 2.67; a Sun at 16 cast 0.82 of 3.16. A size-0 lamp above one
    texel now takes core/softmap's lookup with no lamp term."""
    for kind in ('POINT', 'SPOT', 'SUN'):
        sc, st = MS._prep(kind, 0.0, shadow_softness=1.0)
        m1, a1 = MS._mass(sc, st)
        rows, ok_mass, ok_ramp, widths = [], True, True, []
        for s in (4.0, 8.0, 16.0):
            sc, st = MS._prep(kind, 0.0, shadow_softness=s)
            m, a = MS._mass(sc, st)
            w, x50, lv, step = _edge(sc, st, 1.5)
            rows.append(f'S{s:g}: mass {m:.2f}, width '
                        f'{"none" if w is None else f"{w:.3f}"}, '
                        f'{lv} levels, step {step:.3f}')
            ok_mass = ok_mass and abs(m / m1 - 1.0) < 0.15
            ok_ramp = ok_ramp and w is not None and lv > 12 and step < 0.1
            widths.append(w)
        check(f'{kind} size 0: the shadow keeps its mass from Softness 1 '
              f'to 16 (within 15% of {m1:.2f})', ok_mass, '; '.join(rows))
        check(f'{kind} size 0: the edge is a continuous ramp (more than '
              '12 levels, no step above 0.1) -- it was five ghost copies',
              ok_ramp)
        check(f'{kind} size 0: the Softness still widens the edge, about '
              'in proportion',
              all(w is not None for w in widths)
              and 1.6 < widths[1] / widths[0] < 2.4
              and 1.6 < widths[2] / widths[1] < 2.4,
              ' '.join('none' if w is None else f'{w:.3f}' for w in widths))
        # the 50 % line stays on the hard edge: the shadow is blurred in
        # place (old: it walked 0.04 inwards at Softness 4, 0.07 at 8)
        _w, x0h, _l, _s = _edge(*MS._prep(kind, 0.0, shadow_softness=0.0),
                                y=1.5)
        drift = []
        for s in (4.0, 8.0):
            _w, x50, _l, _s = _edge(*MS._prep(kind, 0.0, shadow_softness=s),
                                    y=1.5)
            drift.append(None if x50 is None or x0h is None
                         else abs(x50 - x0h))
        tex = MS._texel(sc)
        check(f'{kind} size 0: the half-dark line stays on the hard edge '
              'at Softness 4 and 8 (within a texel and a half)',
              all(d is not None and d < 1.5 * tex for d in drift),
              f'{drift} (texel {tex:.3f})')
    # what is NOT touched: Softness at or below 1, and Samples 1
    P, _cell = MS._floor_grid()
    for kind in ('POINT', 'SPOT', 'SUN', 'AREA'):
        same = True
        for kw in ({'shadow_softness': 0.0}, {'shadow_softness': 0.5},
                   {'shadow_softness': 1.0},
                   {'shadow_softness': 1.0, 'shadow_samples': 16},
                   {'shadow_softness': 4.0, 'shadow_samples': 1},
                   {'shadow_softness': 1.0,
                    'shadow_map_depth': 'MIDPOINT'},
                   {'shadow_softness': 2.0,
                    'lamp_kw': {'shadow_softness': 0.5}}):
            sc, st = MS._prep(kind, 0.0, **kw)
            same = same and np.array_equal(MS._vis(sc, st, P),
                                           MS._classic(sc, st, P))
        check(f'{kind} size 0: Softness at or below 1 (lamp x render) and '
              'Samples 1 keep the constant lookup, bitwise', same)
    lt = MS._lamp('POINT', 0.0)
    check('the rule is one function: a size-0 lamp takes the sized lookup '
          'above Softness 1 at Samples above 1, and only then',
          SM.lookup_shape(lt, 4, 1.0) is None
          and SM.lookup_shape(lt, 1, 8.0) is None
          and SM.lookup_shape(lt, 4) is None
          and bool((SM.lookup_shape(lt, 4, 1.001) or {}).get('zero'))
          and SM.lookup_shape(MS._lamp('SUN', 0.0), 4, 2.0)['kind'] == 'SUN'
          and SM.lookup_shape(MS._lamp('AREA', 0.0), 4, 2.0)['kind'] == 'RECT')
    # leaving size 0 under a Softness above 1 is no step any more: the
    # smallest size takes the same lookup with a vanishing lamp term
    for kind, tiny in (('POINT', 1e-4), ('SUN', 1e-5)):
        sc, st = MS._prep(kind, 0.0, shadow_softness=4.0)
        v0 = MS._vis(sc, st, P)
        sc, st = MS._prep(kind, tiny, shadow_softness=4.0)
        v1 = MS._vis(sc, st, P)
        d = float(np.abs(v0 - v1).max())
        check(f'{kind}: at Softness 4 the shadow of size 0 and of size '
              f'{tiny:g} are one picture (it was a step: the classic '
              'lookup against the sized one)', d < 0.03,
              f'max |term| difference {d:.4f}')


# ------------------------------------------------------ b. the GPU twin


def test_b_the_gpu_twin_blurs_it_too():
    """Two roads, one picture: the same statements as GLSL."""
    cases = [('POINT', {}, {'shadow_softness': 4.0}),
             ('SPOT', {}, {'shadow_softness': 2.0}),
             ('SUN', {}, {'shadow_softness': 8.0}),
             ('POINT', {}, {'shadow_softness': 4.0, 'shadow_samples': 16}),
             ('SPOT', {'shadow_softness': 3.0, 'shadow_bias': 0.03,
                       'shadow_density': 0.6}, {}),
             ('POINT', {}, {'shadow_softness': 4.0,
                            'shadow_map_depth': 'MIDPOINT'})]
    for kind, lkw, skw in cases:
        st = MS._settings(**skw)
        sc = MS._wall(MS._lamp(kind, 0.0, **lkw), st)
        cpu, sim, cov, passes, _atl, why = _sim(sc, st)
        label = f'{kind} size 0 {lkw or ""} {skw or ""}'
        if not check(f'{label}: the deferred pass plans', sim is not None,
                     str(why)):
            continue
        d = np.abs(sim[cov] - cpu[cov][:, :3]).max(axis=1)
        check(f'{label}: the simulator shades the CPU picture through '
              'the sized lookup',
              float(d.max()) < 1e-4 and 'hal_sm_blk0' in MS._sources(passes),
              f'max {float(d.max()):.6f} over {int(cov.sum())} px')
    # an AREA lamp of size 0 is degenerate input: its own LIGHTING term
    # differs between the roads with shadows off. Its shadow term is the
    # point lamp's on the CPU, bit for bit (a rectangle of two null
    # half-axes), and the twin is judged on the term alone
    P, _cell = MS._floor_grid()
    sc_a, st_a = MS._prep('AREA', 0.0, shadow_softness=4.0)
    sc_p, st_p = MS._prep('POINT', 0.0, shadow_softness=4.0)
    check('AREA size 0 at Softness 4: the CPU term is the size-0 point '
          "lamp's, bitwise",
          np.array_equal(MS._vis(sc_a, st_a, P), MS._vis(sc_p, st_p, P)))
    st = MS._settings(shadow_softness=4.0)
    lt = MS._lamp('AREA', 0.0, mode='NONE', energy=20.0)
    cpu_n, sim_n, cov, _p, _a, _w = _sim(MS._wall(lt, st), st)
    lt = MS._lamp('AREA', 0.0, energy=20.0)
    cpu, sim, cov, passes, _a, why = _sim(MS._wall(lt, st), st)
    if check('AREA size 0 at Softness 4 plans', sim is not None, str(why)):
        # the shadow TERM of each road: its shadowed frame over its own
        # unshadowed one (no ambient, no specular in this scene)
        cn, sn = cpu_n[..., 0], sim_n[..., 0]
        good = cov & (cn > 1e-3) & (sn > 1e-3) & (cn < 60.0) & (sn < 60.0)
        tc = cpu[..., 0][good] / cn[good]
        ts = sim[..., 0][good] / sn[good]
        d = np.abs(tc - ts)
        part = int(((tc > 0.05) & (tc < 0.95)).sum())
        check("AREA size 0 at Softness 4: the twin's shadow term is the "
              "CPU's (each road's frame over its own unshadowed frame)",
              float(d.max()) < 6e-3 and int(good.sum()) > 2000
              and part > 20 and 'hal_sm_blk0' in MS._sources(passes),
              f'max {float(d.max()):.6f} over {int(good.sum())} px, '
              f'{part} in the penumbra')
    # the constant lookup's text is still emitted for Softness <= 1
    st = MS._settings(shadow_softness=1.0)
    sc = MS._wall(MS._lamp('POINT', 0.0), st)
    cpu, sim, cov, passes, _a, why = _sim(sc, st)
    d = np.abs(sim[cov] - cpu[cov][:, :3]).max(axis=1)
    check('Softness 1 on a size-0 lamp plans the constant lookup and the '
          'twin matches', 'hal_sm_' not in MS._sources(passes)
          and float(d.max()) < 1e-4, f'max {float(d.max()):.6f}')


# ------------------------------------------ c. the device keeps no old atlas


def _plate_scene(st, x, y):
    """A floor and a small thin plate high above it, under a spot that
    looks down: as the lamp sees it the plate is a few map texels."""
    lt = MS._lamp('SPOT', 0.0, pos=(0.0, -0.4, 6.0), aim=(0.0, 0.0, 0.0))
    return MS._finish([plane(z=0.0, size=14.0, mat=0, obj=0),
                       MS._box((x - 0.25, y - 0.04, 1.6),
                               (x + 0.25, y + 0.04, 1.62))], lt, st,
                      eye=(0.0, -5.0, 6.0), at=(0.0, 0.2, 0.0))


def test_c_the_device_never_keeps_an_old_atlas():
    """gpu-atlas-stale-01: the atlas key and the plan signature held
    round(depth[::23].sum(), 3) per face -- every 23rd ROW. A caster that
    slides along the rows, or sits between the sampled rows, kept the key:
    `upload_cached` returned the resident texture and the shadow was drawn
    in the previous frame's place (0.33 over 32 px)."""
    def bake(x, y):
        st = MS._settings(shadow_map_size=512)
        sc = _plate_scene(st, x, y)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        sm = sc.lights[0].shadow_map
        return sm, round(float(sm.depth[::23].sum()), 3)

    # find two plate positions the old fingerprint cannot tell apart
    cands = [(0.0, 0.0)] + [(x, 0.0) for x in (0.3, 0.6, 0.9, 1.2)] \
        + [(0.0, y) for y in (0.3, 0.6, 0.9, 1.2)]
    baked = [(p,) + bake(*p) for p in cands]
    pair = None
    for i in range(len(baked)):
        for j in range(i + 1, len(baked)):
            if baked[i][2] == baked[j][2] and not np.array_equal(
                    baked[i][1].depth, baked[j][1].depth):
                pair = (baked[i], baked[j])
    if not check('two plate positions bake DIFFERENT maps with the same '
                 'every-23rd-row sum (the old key and plan signature)',
                 pair is not None,
                 str([(p, k) for p, _sm, k in baked][:5])):
        return
    (p1, sm1, k1), (p2, sm2, _k2) = pair
    check('the exact digest tells them apart, and a re-bake of the same '
          'map gets the same one',
          GSH.map_digest(sm1) != GSH.map_digest(sm2)
          and GSH.map_digest(bake(*p1)[0]) == GSH.map_digest(sm1),
          f'{p1} / {p2}, row sum {k1}')

    def frames(positions):
        out = []
        for p in positions:
            st = MS._settings(shadow_map_size=512, render_device='GPU')
            sc = _plate_scene(st, *p)
            g = np.asarray(R.render(sc, st))
            eng = bool(R.LAST_GPU_VERDICT.get('engaged'))
            out.append((g, eng, sc, st))
        return out

    _fresh()
    with fakedevice.installed() as dev:
        (g1, e1, _sc1, _st1), (g2, e2, sc2, st2) = frames([p1, p2])
        packs = [k for k in dev.cache if isinstance(k, tuple) and k
                 and k[0] == 'shadowpack']
        replaced = dev.calls['replaced']
    _fresh()
    c2 = _cpu_frame(sc2, st2)
    d = np.abs(g2[..., :3] - c2[..., :3]).max(axis=2)
    check('frame 2 on the SAME device is the CPU frame of the moved '
          'caster (it drew the shadow in the old place)',
          e1 and e2 and float(d.max()) < 6e-3,
          f'max {float(d.max()):.4f}, {int((d > 6e-3).sum())} px, '
          f'gpu engaged {e1 and e2}')
    check('and the frames differ, so the test is not vacuous',
          float(np.abs(g1 - g2).max()) > 0.05)
    check('one resident shadow pack: the changed atlas REPLACED it under '
          'the same key (the stamp rule)', len(packs) == 1 and replaced >= 1,
          f'{len(packs)} pack(s), replaced {replaced}')
    # a moved caster twelve times: still one pack
    _fresh()
    with fakedevice.installed() as dev:
        frames([(0.1 * k, 0.0) for k in range(6)])
        packs = [v for k, v in dev.cache.items() if isinstance(k, tuple)
                 and k and k[0] == 'shadowpack']
    check('six positions of the caster leave ONE pack on the device '
          '(there were seven)', len(packs) == 1, f'{len(packs)}')


# -------------------------------------------- d. an edit is a plan-cache hit


def _tick_pos(sc, st, i):
    l = sc.lights[0]
    l.position = (l.position[0] + 0.15, l.position[1], l.position[2])


def _tick_attr(name, v0, dv, on_settings=False):
    def f(sc, st, i):
        setattr(st if on_settings else sc.lights[0], name, v0 + dv * (i + 1))
    return f


def _tick_area(sc, st, i):
    v = 1.1 + 0.3 * (i + 1)
    sc.lights[0].area_size = (v, v * 0.5)


def _tick_colour(sc, st, i):
    sc.lights[0].shadow_color = (0.1 + 0.2 * (i + 1), 0.0, 0.05)


#: (label, kind, size, method, lamp attributes, settings, one tick)
EDITS = [
    ('a mapped point lamp nudged', 'POINT', 0.0, 'MAP', {}, {}, _tick_pos),
    ('a mapped spot nudged', 'SPOT', 0.3, 'MAP', {}, {}, _tick_pos),
    ('a mapped sun turned', 'SUN', 0.0, 'MAP', {}, {},
     lambda sc, st, i: setattr(sc.lights[0], 'direction',
                               (0.1 * (i + 1), 0.55, -0.8))),
    ('Radius dragged, mapped', 'POINT', 0.3, 'MAP', {}, {},
     _tick_attr('radius', 0.3, 0.2)),
    ('Radius dragged, traced', 'POINT', 0.3, 'RAY', {}, {},
     _tick_attr('radius', 0.3, 0.2)),
    ('Sun Angle dragged, mapped', 'SUN', 0.03, 'MAP', {}, {},
     _tick_attr('radius', 0.03, 0.04)),
    ('Sun Angle dragged, traced', 'SUN', 0.03, 'RAY', {}, {},
     _tick_attr('radius', 0.03, 0.04)),
    ('Area Size dragged, mapped', 'AREA', 1.1, 'MAP', {}, {}, _tick_area),
    ('Area Size dragged, traced', 'AREA', 1.1, 'RAY', {}, {}, _tick_area),
    ("the lamp's Softness dragged", 'POINT', 0.0, 'MAP',
     {'shadow_softness': 2.0}, {}, _tick_attr('shadow_softness', 2.0, 1.5)),
    ("the render's Softness dragged", 'SPOT', 0.2, 'MAP', {},
     {'shadow_softness': 2.0}, _tick_attr('shadow_softness', 2.0, 1.5, True)),
    ("the lamp's Bias dragged", 'POINT', 0.0, 'MAP', {'shadow_bias': 0.02},
     {}, _tick_attr('shadow_bias', 0.02, 0.4)),
    ("the render's Bias dragged", 'SPOT', 0.0, 'MAP', {},
     {'shadow_bias': 0.02}, _tick_attr('shadow_bias', 0.02, 0.4, True)),
    ('Density dragged, mapped', 'POINT', 0.0, 'MAP', {'shadow_density': 0.9},
     {}, _tick_attr('shadow_density', 0.9, -0.2)),
    ('Density dragged, traced', 'POINT', 0.0, 'RAY', {'shadow_density': 0.9},
     {}, _tick_attr('shadow_density', 0.9, -0.2)),
    ('Shadow Colour dragged', 'SPOT', 0.0, 'MAP',
     {'shadow_color': (0.1, 0.0, 0.05)}, {}, _tick_colour),
]


def test_d_a_lamp_edit_is_a_plan_cache_hit():
    """perf-gpu-lamp-move-recompile-04 / vp-size-drag-rebake-recompile /
    size-ray-11: each of these was a plan MISS and one driver compile per
    material pass, every tick (the field's cost of one such storm: 25
    shaders, 33.9 s) -- because the values were printed into the shader.
    They ride hal_lights; the pack rides its content stamp."""
    w, h = 64, 48
    for label, kind, size, mode, lkw, skw, tick in EDITS:
        st = MS._settings(w, h, shadow_map_size=128, render_device='GPU',
                          **skw)
        sc = MS._wall(MS._lamp(kind, size, mode=mode, **lkw), st)
        _fresh()
        with fakedevice.installed() as dev:
            R.render(sc, st)
            eng = bool(R.LAST_GPU_VERDICT.get('engaged'))
            c0 = dev.calls['compile']
            miss0 = GSH.PLAN_STATS['miss']
            hit0 = GSH.PLAN_STATS['hit']
            worst, moved = 0.0, 0.0
            prev = None
            for i in range(2):
                tick(sc, st, i)
                g = np.asarray(R.render(sc, st))
                eng = eng and bool(R.LAST_GPU_VERDICT.get('engaged'))
                c = _cpu_frame(sc, st)
                worst = max(worst, float(np.abs(g[..., :3]
                                                - c[..., :3]).max()))
                if prev is not None:
                    moved = max(moved, float(np.abs(g - prev).max()))
                prev = g
            comp = dev.calls['compile'] - c0
            miss = GSH.PLAN_STATS['miss'] - miss0
            hit = GSH.PLAN_STATS['hit'] - hit0
            packs = sum(1 for k in dev.cache if isinstance(k, tuple) and k
                        and k[0] == 'shadowpack')
        check(f'{label}: two ticks compile nothing and re-plan nothing',
              eng and c0 > 0 and comp == 0 and miss == 0 and hit >= 2,
              f'first frame {c0} compiles, then {comp}; plan miss {miss} '
              f'hit {hit}; gpu engaged {eng}')
        check(f'{label}: and each frame is the CPU frame of the edited '
              'scene', worst < 6e-3 and moved > 1e-3 and packs <= 1,
              f'max {worst:.6f}; the edit moved the picture by {moved:.3f}; '
              f'{packs} pack(s) resident')


# ------------------------------------------------ e. structure still re-plans


def test_e_structure_replans_and_only_structure():
    """A value the plan bakes into shader text must be in the signature;
    a value that rides a texel must not be."""
    def sig(kind, size, mode='MAP', lkw=None, **skw):
        st = MS._settings(**skw)
        sc = MS._wall(MS._lamp(kind, size, mode=mode, **(lkw or {})), st)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        bvh = BVH.cached(sc.mesh.verts, sc.mesh.tris) if mode == 'RAY' \
            else None
        job, _g = _job(sc, st, bvh=bvh)
        return GSH._plan_sig(job, GSH._mesh_key(sc.mesh))

    base = sig('POINT', 0.3)
    values = [
        ('a mapped lamp moved', sig('POINT', 0.3, lkw={
            'position': (0.5, -4.0, 6.0)})),
        ('its Radius', sig('POINT', 0.6)),
        ("the lamp's Softness and Bias", sig('POINT', 0.3, lkw={
            'shadow_softness': 3.0, 'shadow_bias': 0.05})),
        ("the render's Softness and Bias", sig('POINT', 0.3,
                                               shadow_softness=3.0,
                                               shadow_bias=0.05)),
    ]
    for what, s in values:
        check(f'{what} is a value: the plan signature does not move',
              s == base)
    # (whether the Density term is 1.0 stays one bit of the light
    # signature: a traced shadow emits the term only off 1.0)
    check("a mapped lamp's Density is a value off 1.0",
          sig('POINT', 0.3, lkw={'shadow_density': 0.9})
          == sig('POINT', 0.3, lkw={'shadow_density': 0.4}))
    check('the Shadow Colour is a value once it is not black',
          sig('POINT', 0.3, lkw={'shadow_color': (0.2, 0.0, 0.0)})
          == sig('POINT', 0.3, lkw={'shadow_color': (0.0, 0.3, 0.1)}))
    check("a traced lamp's Radius and Density are values",
          sig('POINT', 0.3, 'RAY', lkw={'shadow_density': 0.8})
          == sig('POINT', 0.7, 'RAY', lkw={'shadow_density': 0.4}))
    check("a Spot's cone and aim are values under Shadow Maps",
          sig('SPOT', 0.0, lkw={'spot_size': 1.0})
          == sig('SPOT', 0.0, lkw={'spot_size': 1.3,
                                   'direction': (0.1, 0.6, -0.8)}))
    check("an Area lamp's Size, turn and place are values (mapped and "
          'traced)',
          sig('AREA', (1.0, 0.5)) == sig('AREA', (1.4, 0.3), lkw={
              'position': (0.3, -4.0, 6.0), 'area_x': (0.0, 1.0, 0.0),
              'area_y': (1.0, 0.0, 0.0)})
          and sig('AREA', (1.0, 0.5), 'RAY') == sig('AREA', (1.4, 0.3), 'RAY'))
    structure = [
        ('a size appearing on a size-0 lamp (the lookup changes)',
         sig('POINT', 0.0), sig('POINT', 0.3)),
        ('Softness crossing one texel on a size-0 lamp',
         sig('POINT', 0.0, shadow_softness=1.0),
         sig('POINT', 0.0, shadow_softness=1.5)),
        ('Samples', base, sig('POINT', 0.3, shadow_samples=8)),
        ("the lamp's own Samples", base,
         sig('POINT', 0.3, lkw={'shadow_samples': 8})),
        ('the Map Size', base, sig('POINT', 0.3, shadow_map_size=256)),
        ('the Map Depth rule', base,
         sig('POINT', 0.3, shadow_map_depth='MIDPOINT')),
        ('the shadow method', base, sig('POINT', 0.3, 'RAY')),
        ('a traced Density leaving 1.0', sig('POINT', 0.3, 'RAY'),
         sig('POINT', 0.3, 'RAY', lkw={'shadow_density': 0.5})),
        ('a Shadow Colour leaving black', base,
         sig('POINT', 0.3, lkw={'shadow_color': (0.2, 0.0, 0.0)})),
        ("an Area lamp's shape", sig('AREA', 1.0),
         sig('AREA', 1.0, lkw={'area_shape': 'DISK'})),
        ("a traced Sun's law (disc or cap)", sig('SUN', 0.2, 'RAY'),
         sig('SUN', 2.0, 'RAY')),
    ]
    for what, a, b in structure:
        check(f'{what} is structure: a new plan', a != b)
    # and a structure flip on a kept device draws the right frame
    st = MS._settings(64, 48, shadow_map_size=128, render_device='GPU')
    sc = MS._wall(MS._lamp('POINT', 0.0), st)
    _fresh()
    with fakedevice.installed() as dev:
        R.render(sc, st)
        c0 = dev.calls['compile']
        sc.lights[0].radius = 0.4
        g1 = np.asarray(R.render(sc, st))
        c1 = dev.calls['compile']
        d1 = float(np.abs(g1[..., :3] - _cpu_frame(sc, st)[..., :3]).max())
        sc.lights[0].radius = 0.0
        st.shadow_softness = 4.0
        g2 = np.asarray(R.render(sc, st))
        d2 = float(np.abs(g2[..., :3] - _cpu_frame(sc, st)[..., :3]).max())
    check('a lamp gaining a size on a kept device compiles the sized '
          'lookup and draws the CPU frame; so does Softness 4 at size 0',
          c1 > c0 and d1 < 6e-3 and d2 < 6e-3,
          f'{c0} -> {c1} compiles, max {d1:.6f} / {d2:.6f}')


# ----------------------------------- f. the texel road is the literal road


def _both_roads(sc, st, ray=False):
    """(texel frame, literal frame, texel sources, literal sources)."""
    out = []
    for flag in (True, False):
        old = GSH.SHADOW_TEXELS
        GSH.SHADOW_TEXELS = flag
        try:
            _cpu, sim, _cov, passes, _atl, why = _sim(sc, st, ray=ray)
        finally:
            GSH.SHADOW_TEXELS = old
        out.append((sim, None if passes is None else MS._sources(passes),
                    why))
    return out


def test_f_the_texel_road_is_the_literal_road():
    """The values are the same float32 numbers either way: the frame the
    texels shade is the frame the literals shaded, to the bit."""
    cases = [('POINT', 0.0, 'MAP', {}, {}),
             ('SPOT', 0.0, 'MAP', {'shadow_bias': 0.03,
                                   'shadow_density': 0.6,
                                   'shadow_color': (0.2, 0.05, 0.0)},
              {'shadow_samples': 7}),
             ('SUN', 0.0, 'MAP', {}, {'shadow_softness': 0.0}),
             ('POINT', 0.0, 'MAP', {}, {'shadow_softness': 4.0}),
             ('POINT', 0.3, 'MAP', {}, {}),
             ('SUN', 0.05, 'MAP', {}, {'shadow_softness': 2.0}),
             ('AREA', (1.0, 0.5), 'MAP', {}, {}),
             ('POINT', 0.3, 'MAP', {}, {'shadow_map_depth': 'MIDPOINT'}),
             ('POINT', 0.3, 'RAY', {'shadow_density': 0.7}, {}),
             ('SUN', 0.1, 'RAY', {}, {}),
             ('SUN', 2.0, 'RAY', {}, {}),
             ('AREA', (1.0, 0.5), 'RAY', {}, {}),
             ('AREA', 1.0, 'RAY', {'area_shape': 'DISK',
                                   'area_gamma': 0.8}, {}),
             ('AREA', 1.0, 'NONE', {'area_gamma': 1.3}, {})]
    for kind, size, mode, lkw, skw in cases:
        st = MS._settings(**skw)
        sc = MS._wall(MS._lamp(kind, size, mode=mode, **lkw), st)
        (a, ta, wa), (b, tb, wb) = _both_roads(sc, st, ray=(mode == 'RAY'))
        label = f'{kind} size {size} {mode} {lkw or ""} {skw or ""}'
        if not check(f'{label}: both roads plan',
                     a is not None and b is not None, f'{wa} / {wb}'):
            continue
        check(f'{label}: the texel frame is the literal frame, bitwise',
              np.array_equal(a, b) and ta != tb,
              f'max {float(np.abs(a - b).max()):.3g}')

    # two lamps of one structure and different VALUES: one shader text
    def text(pos, size, lkw, skw, kind='POINT', mode='MAP'):
        st = MS._settings(**skw)
        sc = MS._wall(MS._lamp(kind, size, mode=mode, pos=pos, **lkw), st)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        bvh = BVH.cached(sc.mesh.verts, sc.mesh.tris) if mode == 'RAY' \
            else None
        job, g = _job(sc, st, bvh=bvh)
        GSH._PLAN_CACHE.clear()
        passes, why, _atl = GSH.plan_frame(job, g)
        return MS._sources(passes)
    for kind, mode, s1, s2 in (('POINT', 'MAP', 0.3, 0.7),
                               ('SPOT', 'MAP', 0.0, 0.0),
                               ('POINT', 'RAY', 0.3, 0.7),
                               ('AREA', 'RAY', (1.0, 0.5), (1.6, 0.4))):
        t1 = text(MS.LAMP_POS, s1, {'shadow_bias': 0.02,
                                    'shadow_density': 0.9,
                                    'shadow_color': (0.1, 0.0, 0.0)},
                  {'shadow_softness': 1.0}, kind, mode)
        t2 = text((0.6, -3.5, 5.5), s2, {'shadow_bias': 0.04,
                                         'shadow_density': 0.5,
                                         'shadow_color': (0.0, 0.2, 0.1)},
                  {'shadow_softness': 0.7 if s1 == 0.0 else 2.5}, kind, mode)
        check(f'{kind} {mode}: two lamps that differ in place, size, '
              'Softness, Bias, Density and Shadow Colour emit ONE text',
              t1 == t2 and 'hal_ltex(' in t1)

    # the texels are the CPU's float32 values
    st = MS._settings(shadow_softness=2.5)
    lt = MS._lamp('POINT', 0.3, shadow_bias=0.03, shadow_density=0.7,
                  shadow_color=(0.2, 0.1, 0.0))
    sc = MS._wall(lt, st)
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    meta, atlas, _why = GSH._shadow_meta(lt, st)
    tail = GM.light_tail_layout([lt], [meta])
    tex = GM.pack_light_texels([lt], None, shadows=[meta])
    mb = tail['map'][0]
    sm = lt.shadow_map
    k = f32(sm.faces[0].size / (2.0 * sm.faces[0].extent))
    ok = (tex.shape == (1, tail['width'], 4)
          and mb == GM.LIGHT_TEXEL_STRIDE
          and tail['width'] == mb + GM.MAP_HEAD + 6 * GM.MAP_FACE
          and np.array_equal(tex[0, mb, :3], sm.origin)
          and tex[0, mb, 3] == f32(0.03)
          and tex[0, mb + 1, 0] == f32(sm.faces[0].near)
          and tex[0, mb + 1, 2] == f32(sm.faces[0].far + sm.faces[0].near)
          and tex[0, mb + 2, 2] == f32(2.5) and tex[0, mb + 2, 3] == f32(0.7)
          and tex[0, mb + 3, 0] == f32(2.5 + SM.SEARCH_PAD)
          and tex[0, mb + 3, 1] == k
          and tex[0, mb + 3, 2] == f32(float(k) * 0.3)
          and tex[0, mb + 3, 3] == SM.cap_texels(sm.faces[0].size,
                                                 sm.faces[0].extent, True)
          and tex[0, mb + 4, 0] == f32(sm.d_min))
    for fi, face in enumerate(sm.faces):
        b = mb + GM.MAP_HEAD + fi * GM.MAP_FACE
        ok = ok and np.array_equal(tex[0, b:b + 4],
                                   np.asarray(face.vp, np.float32))
        ok = ok and all(np.array_equal(tex[0, b + 4 + a, :3], face.axes[a])
                        for a in range(3))
    check("a mapped lamp's tail block holds the float32 values the CPU "
          'lookup reads (origin, Bias, planes, Softness, pad, k, k R, the '
          'cap, the nearest depth, every face row and axis)', ok)
    check('texels 12 / 13 hold the Shadow Colour, Density and the size',
          np.array_equal(tex[0, 12, :3], np.asarray((0.2, 0.1, 0.0), f32))
          and tex[0, 12, 3] == f32(0.7) and tex[0, 13, 0] == f32(0.3))
    sun = MS._lamp('SUN', 2.0)
    area = MS._lamp('AREA', (1.4, 0.6))
    t2 = GM.pack_light_texels([sun, area], None, shadows=[None, None])
    ax = np.asarray(area.area_x, f32)
    ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
    check("a Sun's texel 13.x is its traced law's constant and an Area "
          "lamp's 14 / 15 its unit axes and half sizes; its face follows "
          'in the tail',
          t2[0, 13, 0] == f32(LI.sun_soft_law(2.0)[1])
          and np.array_equal(t2[0, 16 + 14, :3], ax)
          and t2[0, 16 + 14, 3] == f32(0.7) and t2[0, 16 + 15, 3] == f32(0.3)
          and t2.shape[1] == 32 + 2
          and np.array_equal(t2[0, 32:34, :3], GM.area_face_texels(area)))
    check('without shadow metas the texture keeps its n x 16 texels '
          '(texels 12-15 are filled from the lamp alone)',
          GM.pack_light_texels([lt]).shape == (1, 16, 4)
          and GM.pack_light_texels([lt])[0, 12, 3] == f32(0.7))


# --------------------------------------------------- g. one channel of depth


class _StubGPU:
    """A stand-in for Blender's gpu module that records what a texture
    upload asks for; `refuse` names a format the build cannot create."""

    def __init__(self, refuse=None):
        self.made = []
        outer = self

        class Buffer:
            def __init__(self, kind, n, data):
                self.kind, self.n = kind, int(n)
                self.data = np.asarray(data, np.float32).copy()

        class GPUTexture:
            def __init__(self, size, format='RGBA8', data=None):  # noqa: A002
                if format == refuse:
                    raise ValueError(f'{format} is not supported')
                outer.made.append((tuple(size), format, data.n,
                                   data.data.copy()))

        self.types = types.SimpleNamespace(Buffer=Buffer,
                                           GPUTexture=GPUTexture)


def test_g_the_pack_is_one_channel():
    """perf-gpu-lamp-move-recompile-04 (b): the pack was RGBA32F with one
    channel used -- 384 MB for one 2048 point lamp, re-uploaded whole on
    every nudge. One float per texel is a quarter of it."""
    st = MS._settings(shadow_map_size=128)
    lt = MS._lamp('POINT', 0.3)
    sc = MS._wall(lt, st)
    cpu, sim, cov, passes, atlases, why = _sim(sc, st)
    key, build, stamp = atlases['hal_shadowpack']
    px = build()
    sm = lt.shadow_map
    # R254 soft-search: a lamp on the per-receiver lookup (this one has a
    # Radius) carries its reach pyramid under its cells -- rows derived
    # from the same depth texels, announced by the key's PYR. The pins
    # were `px.shape == (2 * 128, 3 * 128)` and a key without PYR
    from ..core import softmap as _SMP
    lay = _SMP.pyr_layout(128)
    check('the pack builds as ONE channel of float32: the six faces in '
          "their cells and, under them, the lamp's reach pyramid",
          px.ndim == 2 and px.dtype == np.float32
          and px.shape == (2 * 128 + lay['h'], 3 * 128)
          and all(np.array_equal(
              px[(fi // 3) * 128:(fi // 3 + 1) * 128,
                 (fi % 3) * 128:(fi % 3 + 1) * 128], f.depth)
              for fi, f in enumerate(sm.faces))
          and all(np.array_equal(
              px[256:, fi * lay['w']:(fi + 1) * lay['w']], blk)
              for fi, blk in enumerate(_SMP.map_pyramid(sm)))
          and not px[256:, 6 * lay['w']:].any(), str(px.shape))
    check('its key is the layout and its stamp the content',
          key == ('shadowpack', ('shadow', 128, (3, 2), 'CLS', 'PYR'))
          and stamp == (tuple(GSH.map_digest(f) for f in sm.faces),))
    tex = GSH.sim_texture(px)
    from ..shaders import builtins as BI
    got = BI.texelFetch(tex, np.array([[5, 7], [130, 200]], np.int32))
    check('the simulator samples it as an R32F texture: (r, 0, 0, 1)',
          isinstance(tex, GSH.RedTexture)
          and np.array_equal(got[:, 0], px[[7, 200], [5, 130]])
          and np.array_equal(got[:, 1:], [[0, 0, 1], [0, 0, 1]]))
    d = np.abs(sim[cov] - cpu[cov][:, :3]).max(axis=1)
    check('and the frame it shades is the CPU frame',
          float(d.max()) < 1e-4, f'max {float(d.max()):.6f}')
    # the fake device holds one float per texel
    st = MS._settings(64, 48, shadow_map_size=128, render_device='GPU')
    sc = MS._wall(MS._lamp('POINT', 0.0), st)
    _fresh()
    with fakedevice.installed() as dev:
        R.render(sc, st)
        packs = [v for k, v in dev.cache.items() if isinstance(k, tuple)
                 and k and k[0] == 'shadowpack']
    check('the fake device keeps it as R32F: 4 bytes a texel (it was 16)',
          len(packs) == 1 and packs[0].fmt == 'R32F'
          and packs[0].pixels.nbytes == 2 * 128 * 3 * 128 * 4,
          f'{[(p.fmt, p.pixels.nbytes) for p in packs]}')
    # the real upload path, over a stand-in gpu module
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    saved = sys.modules.get('gpu')
    try:
        stub = _StubGPU()
        sys.modules['gpu'] = stub
        DEV._upload_impl(arr)
        DEV._upload_impl(np.zeros((3, 4, 4), np.float32))
        made = list(stub.made)
        stub = _StubGPU(refuse='R32F')
        sys.modules['gpu'] = stub
        DEV._upload_impl(arr)
        fell = list(stub.made)
    finally:
        if saved is None:
            sys.modules.pop('gpu', None)
        else:
            sys.modules['gpu'] = saved
    check('gpu/device uploads a 2D array as R32F, one float a texel, and '
          'everything else as RGBA32F',
          len(made) == 2 and made[0][:3] == ((4, 3), 'R32F', 12)
          and np.array_equal(made[0][3], arr.ravel())
          and made[1][:3] == ((4, 3), 'RGBA32F', 48),
          str([m[:3] for m in made]))
    check('a build that cannot create R32F gets the same values in the .r '
          'of an RGBA32F texture',
          len(fell) == 1 and fell[0][:3] == ((4, 3), 'RGBA32F', 48)
          and np.array_equal(fell[0][3].reshape(3, 4, 4)[..., 0], arr),
          str([m[:3] for m in fell]))


# ---------------------------------------------------- h. the plan cache


def test_h_the_plan_cache_holds_draft_and_refine():
    """The cache emptied itself past four entries, and every nudge of a
    mapped lamp was a new entry: a drag threw the draft's and the
    refine's plans away every fifth tick."""
    def scene(map_size):
        st = MS._settings(shadow_map_size=map_size)
        return MS._wall(MS._lamp('POINT', 0.3), st), st

    sc_d, st_d = scene(64)            # a draft: a small frame, a small map
    sc_r, st_r = scene(128)           # the refine
    _fresh()
    GSH.PLAN_STATS['hit'] = GSH.PLAN_STATS['miss'] = 0
    stamps = []
    for i in range(6):
        for sc, st, (w, h) in ((sc_d, st_d, (48, 36)), (sc_r, st_r, (96, 72))):
            l = sc.lights[0]
            l.position = (0.1 * i, l.position[1], l.position[2])
            LI.build_shadow_maps(sc, st)
            job, g = _job(sc, st, w, h)
            passes, why, atlases = GSH.plan_frame(job, g)
            stamps.append(atlases['hal_shadowpack'][2])
    check('a six-tick drag alternating a draft and a refine builds two '
          'plans and hits them ten times (it built twelve)',
          GSH.PLAN_STATS['miss'] == 2 and GSH.PLAN_STATS['hit'] == 10
          and len(GSH._PLAN_CACHE) == 2, str(GSH.PLAN_STATS))
    check('every hit carried the CURRENT maps under their own stamp',
          len(set(stamps)) == 12)
    kept = [e[2]['hal_shadowpack'] for e in GSH._PLAN_CACHE.values()]
    check('a parked plan holds no depth map (its pack entry is the key '
          'alone; every hit rebuilds it)',
          all(k[1] is None and k[2] is None for k in kept))
    # the cap evicts the least recently used plan, one at a time
    _fresh()
    sizes = [(40 + 4 * i, 30 + 3 * i) for i in range(GSH.PLAN_CACHE_MAX + 3)]
    sigs = []
    for w, h in sizes:
        job, g = _job(sc_r, st_r, w, h)
        LI.build_shadow_maps(sc_r, st_r)
        GSH.plan_frame(job, g)
        sigs.append(next(reversed(GSH._PLAN_CACHE)))
        # keep the first plan in use, as a viewport keeps its refine
        job0, g0 = _job(sc_r, st_r, *sizes[0])
        GSH.plan_frame(job0, g0)
    check('past the cap the oldest unused plan goes, one at a time, and '
          'a plan in use stays',
          len(GSH._PLAN_CACHE) == GSH.PLAN_CACHE_MAX
          and sigs[0] in GSH._PLAN_CACHE and sigs[-1] in GSH._PLAN_CACHE
          and sigs[1] not in GSH._PLAN_CACHE and GSH.PLAN_CACHE_MAX >= 8,
          f'{len(GSH._PLAN_CACHE)} of {GSH.PLAN_CACHE_MAX}')


def test_i_the_samples_tooltip_names_the_rule():
    """The render's Samples said a lamp of size 0 'takes this many taps
    of the Softness blur': true only up to one texel of Softness now."""
    import importlib

    from . import fakebpy
    fakebpy.install()
    try:
        import halcyon.properties as P
    except ImportError:
        P = importlib.import_module('..properties', __package__)
    importlib.reload(P)
    samp = ' '.join(str(P.HalcyonSettings.__annotations__['shadow_samples']
                        .kw.get('description', '') or '').split())
    check("the render's Samples says any lamp under a Softness above 1 "
          'takes the smooth filter, and a size-0 lamp its own tap count '
          'only at Softness 1 or less',
          len(samp) >= 40 and 'Softness above 1' in samp
          and 'size 0 at Softness 1 or less' in samp
          and '4 taps per sample (16 to 64)' in samp
          and '1 is a hard edge' in samp, samp)


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
    print('all R254 map-gpu checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
