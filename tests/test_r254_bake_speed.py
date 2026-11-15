"""R254: the shadow-map road bakes less, and bakes it faster.

The field report: 'Shadow map is slower than ray traced shadows.' Per pixel
a map lookup was always the cheaper term; the road lost on every frame that
BAKED, and it baked far more than anything read:

    a  the bake is depth only -- no G-buffer -- and bitwise the texels the
       beauty rasteriser left (CLASSIC and MIDPOINT, a lamp inside the
       geometry, grazing lamps, odd sizes, a caster subset, clip planes,
       degenerate and coincident triangles, NaN and infinite depths)
    b  the per-map reject drops only triangles that cannot own a fragment
    c  a map is baked for the lamps something looks up: not for a lamp the
       Light Limit dropped, a Hemi, a Screen Spotlight or a lamp that adds
       no light -- and every reader that DOES look (a volume, the ground
       plane, the Shadow pass, a Shadows Only material, the paint) still
       gets the picture it always got, on both roads
    d  a lamp's cache key holds what its own bake reads: a moved Sun, a
       rotated Point lamp, an aimed Area lamp, a render setting the lamp
       overrides re-rasterise nothing
    e  a viewport draft bakes draft-sized maps, kept apart from the
       refine's
    f  a moved object re-rasterises itself over a cached static layer,
       bitwise the full bake
    g  the cache is bounded by bytes and never drops what the frame uses
    h  the bake's threads follow texels as well as triangles
    i  a worker pool's parent bakes once and the workers adopt the maps
    j  a wide Spot bakes a cube, a GX Ring 2 Spot's map reaches its ring,
       an Area lamp ignores the clip planes its panel never showed
    k  the coarse-map note measures from the camera; a frame that builds
       no map leaves none behind

    python -m halcyon.tests.test_r254_bake_speed
"""
import concurrent.futures as fut
import contextlib
import copy
import io
import pickle
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import _vol_graph, _with_volume, base_settings
from .test_r253_compositing import _bi_graph
from .scenebuild import (_mesh_concat, cube, demo_scene, look_at_matrix,
                         plane, sphere)
from ..core import lights as LI
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..core.settings import RenderSettings
from ..gpu import shade as GSH

FAILS = []
W, H = 96, 72


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


# ------------------------------------------------------------------ helpers


def _settings(size=128, **kw):
    st = base_settings(W, H, shadows=True, transparency='NONE')
    st.shadow_default = 'MAP'
    st.shadow_map_size = size
    st.global_ambient_level = 0.0
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _lamp(kind, pos=(3.0, -3.0, 5.0), aim=(0.0, 0.0, 0.5), **kw):
    d = np.array(aim, np.float64) - np.array(pos, np.float64)
    d = d / np.linalg.norm(d)
    l = Light(type=kind, name=kw.pop('name', 'L'), position=tuple(pos),
              direction=tuple(d), color=(1, 1, 1),
              energy=3.0 if kind in ('SUN', 'HEMI') else 500.0, radius=0.0,
              shadow='MAP')
    if kind == 'SPOT':
        l.spot_size = 1.2
    if kind == 'AREA':
        l.area_size = (0.0, 0.0)
        x = np.cross(d, np.array([0, 0, 1.0]))
        x /= np.linalg.norm(x)
        l.area_x = tuple(x)
        l.area_y = tuple(np.cross(x, d))
    for k, v in kw.items():
        setattr(l, k, v)
    return l


def _scene(lights, st, seg=16, ball=True, mats=None):
    """A floor, four spheres and (object 5) a small ball that moves."""
    parts = [plane(z=0.0, size=14.0, mat=0, obj=0)]
    k = 1
    for x, y in ((-2.0, -1.0), (2.0, -1.0), (-2.0, 1.5), (2.0, 1.5)):
        parts.append(sphere(centre=(x, y, 1.0), radius=0.9, mat=1, obj=k,
                            segs=seg, rings=max(seg // 2, 4)))
        k += 1
    if ball:
        parts.append(sphere(centre=(0.0, -2.0, 0.4), radius=0.4, mat=1,
                            obj=k, segs=12, rings=6))
        k += 1
    mesh = _mesh_concat(parts)
    mesh.smooth = (mesh.mat_index == 1)
    objs = [ObjectInfo(name=f'O{i}', index=i, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=np.float32))
            for i in range(k)]
    cam = Camera(matrix_world=look_at_matrix((0.0, -8.5, 6.0),
                                             (0.0, 0.3, 0.5)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    sc = Scene(mesh=mesh, materials=mats or [
        Material(name='Floor', index=0, model='LAMBERT',
                 diffuse=(0.8, 0.8, 0.8), specular_level=0.0),
        Material(name='Ball', index=1, model='PHONG',
                 diffuse=(0.8, 0.3, 0.2), specular_level=0.5)],
        objects=objs, lights=list(lights), camera=cam,
        world=World(color=(0.02, 0.02, 0.03), ambient=(0.0, 0.0, 0.0)),
        settings=st)
    sc.images = {}
    return sc


def _faces(sm):
    return list(sm.faces) if hasattr(sm, 'faces') else [sm]


def _maps(sc):
    return [None if l.shadow_map is None else
            [f.depth for f in _faces(l.shadow_map)] for l in sc.lights]


def _same_maps(a, b):
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if (x is None) != (y is None):
            return False
        if x is not None and (len(x) != len(y) or not all(
                np.array_equal(p, q) for p, q in zip(x, y))):
            return False
    return True


@contextlib.contextmanager
def _spy():
    """Every bake that runs inside: [(triangles, size, midpoint)]."""
    calls = []
    real = LI._render_depth

    def counted(verts, tris, vp, size, *a, **k):
        calls.append((int(tris.shape[0]), int(size),
                      bool(k.get('midpoint', False))))
        return real(verts, tris, vp, size, *a, **k)
    LI._render_depth = counted
    try:
        yield calls
    finally:
        LI._render_depth = real


@contextlib.contextmanager
def _bake_every_lamp():
    """The old rule: a map for every lamp whose menu resolves to one."""
    real = LI.map_roles

    def roles(scene, settings):
        return ['bake' if LI.wants_shadow_map(l, settings) else None
                for l in (scene.lights or [])]
    LI.map_roles = roles
    try:
        yield
    finally:
        LI.map_roles = real


def _cold(sc, st):
    LI.clear_shadow_cache()
    return np.asarray(R.render(sc, st))


def _gbuffer_depth(verts, tris, vp, size, cull='NONE', midpoint=False):
    """The bake as it was: the beauty rasteriser into a G-buffer, the
    MIDPOINT second surface as a chunked fragment sweep."""
    gb = CRg.GBuffer(size, size)
    CRg.rasterize(verts, tris, vp, size, size, cull=cull, gbuf=gb,
                  depth_bits=32)
    if not midpoint:
        return gb.zndc
    z1 = gb.zndc
    z2 = np.full((size, size), 1.0, np.float32)
    has2 = np.zeros((size, size), bool)
    for start in range(0, int(tris.shape[0]), LI.MIDPOINT_RUN):
        run = tris[start:start + LI.MIDPOINT_RUN]
        gb2 = CRg.GBuffer(size, size)
        fl = LI._DepthFrags()
        CRg.rasterize(verts, run, vp, size, size, cull=cull, gbuf=gb2,
                      frags=fl, depth_write=False, depth_bits=32)
        px, py, zz = fl.finish()
        if px.size == 0:
            continue
        sel = zz > CRg.abuf_depth_limit(z1[py, px])
        if not sel.any():
            continue
        np.minimum.at(z2, (py[sel], px[sel]), zz[sel])
        has2[py[sel], px[sel]] = True
    return z1, z2, has2


def _eq_depth(a, b):
    if isinstance(a, tuple):
        return all(np.array_equal(x, y) for x, y in zip(a, b))
    return np.array_equal(a, b)


def _gpu(sc, st):
    """(cpu frame, simulated frame, covered mask, passes, why)."""
    cpu = np.asarray(R.render(sc, st))
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, W, H)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    if passes is None:
        return cpu, None, g.tri >= 0, None, why
    img, _hit = GSH.simulate(job, g, passes, atlases)
    return cpu, np.asarray(img), g.tri >= 0, passes, None


# -------------------------------------------------------------------- tests


def test_a_the_bake_is_depth_only_and_bitwise():
    """perf-depth-only-bake-05 / perf-bake-01: the bake ran the beauty
    rasteriser into a 29-byte-per-texel G-buffer to keep one depth."""
    built = []
    real = CRg.GBuffer

    class Counted(real):
        def __init__(self, *a, **k):
            built.append(1)
            super().__init__(*a, **k)
    st = _settings(128)
    sc = _scene([_lamp('POINT'), _lamp('SUN', pos=(1, 2, 6), name='S'),
                 _lamp('SPOT', pos=(-3, -2, 4), name='T',
                       shadow_map_depth='MIDPOINT')], st)
    CRg.GBuffer = Counted
    try:
        LI._build_shadow_maps(sc, st)
    finally:
        CRg.GBuffer = real
    check('a bake of a cube, a Sun map and a MIDPOINT Spot map builds no '
          'G-buffer', not built and all(l.shadow_map is not None
                                        for l in sc.lights),
          f'{len(built)} built')

    cases = []

    def add(label, sc, st, cast=None):
        cases.append((label, sc, st, cast))
    st = _settings(128)
    add('POINT inside the sphere cluster',
        _scene([_lamp('POINT', pos=(0.0, 0.2, 1.0))], st), st)
    add('POINT inside one sphere',
        _scene([_lamp('POINT', pos=(2.0, 1.5, 1.0))], st), st)
    add('POINT 2 cm above the floor',
        _scene([_lamp('POINT', pos=(1.0, 0.2, 0.02))], st), st)
    add('SPOT 2 cm above the floor, grazing',
        _scene([_lamp('SPOT', pos=(5.0, 0.0, 0.02), aim=(0.0, 0.0, 0.3),
                      spot_size=1.8)], st), st)
    st = _settings(177)
    add('POINT, odd map size 177', _scene([_lamp('POINT')], st), st)
    st = _settings(48)
    add('SUN + SPOT, map size 48',
        _scene([_lamp('SUN'), _lamp('SPOT', pos=(-3, -2, 4), name='T')],
               st), st)
    st = _settings(128)
    add('SPOT ClipSta 4.0 / ClipEnd 7.5 (cuts the casters)',
        _scene([_lamp('SPOT', shadow_clip_start=4.0, shadow_clip_end=7.5)],
               st), st)
    add('SPOT Square cone', _scene([_lamp('SPOT', spot_square=True)], st),
        st)
    sc = _scene([_lamp('POINT'), _lamp('SUN', pos=(1, 2, 6), name='S')], st)
    add('caster subset (the floor does not cast)', sc, st,
        np.nonzero(sc.mesh.mat_index == 1)[0])
    # a soup: small triangles, zero-area ones, coincident pairs
    rng = np.random.default_rng(5)
    sc = _scene([_lamp('POINT', pos=(0.5, 0.2, 2.0)),
                 _lamp('SPOT', pos=(4, -4, 6), name='T', spot_size=1.3)], st)
    m = sc.mesh
    C = rng.uniform(-4, 4, (6000, 3)).astype(np.float32)
    C[:, 2] = np.abs(C[:, 2]) * 0.6 + 0.05
    Vn = (np.repeat(C, 3, 0)
          + rng.uniform(-0.15, 0.15, (18000, 3))).astype(np.float32)
    Vn[:600] = np.repeat(C[:200], 3, 0)
    newt = np.arange(18000, dtype=np.int32).reshape(-1, 3) + m.verts.shape[0]
    m.verts = np.concatenate([m.verts, Vn])
    m.tris = np.concatenate([m.tris, newt, newt[200:400].copy()])
    m.obj_index = None
    add('a triangle soup: degenerate and coincident triangles', sc, st)
    st2 = base_settings(W, H)
    st2.shadow_map_size = 160
    sc = demo_scene(st2, with_texture=False)
    for l in sc.lights:
        if l.type != 'AMBIENT':
            l.shadow = 'MAP'
    add('the demo scene, its own lamps', sc, st2)
    bad = n = 0
    for label, sc, st, cast in cases:
        for mid in (False, True):
            st.shadow_map_depth = 'MIDPOINT' if mid else 'CLASSIC'
            LI._build_shadow_maps(sc, st, cast)
            tris = sc.mesh.tris if cast is None else sc.mesh.tris[cast]
            ok = True
            for l in sc.lights:
                if l.shadow_map is None:
                    continue
                for f in _faces(l.shadow_map):
                    ref = _gbuffer_depth(sc.mesh.verts, tris, f.vp, f.size,
                                         midpoint=mid)
                    new = LI._render_depth(sc.mesh.verts, tris, f.vp,
                                           f.size, midpoint=mid)
                    old_map = LI.ShadowMap(f.vp, ref, f.near, f.far,
                                           f.persp, f.size, f.origin,
                                           f.extent)
                    ok = ok and _eq_depth(ref, new) \
                        and np.array_equal(old_map.depth, f.depth) \
                        and old_map.d_min == f.d_min
            n += 1
            bad += not ok
            if not ok:
                print(f'       differs: {label} '
                      f'{"MIDPOINT" if mid else "CLASSIC"}')
    check('every map is bitwise the G-buffer road\'s, zndc and built '
          f'ShadowMap, over {n} cases', bad == 0, f'{bad} differ')

    # the fills' own statements: NaN and infinite depths, both windings,
    # big and small boxes, straight against fill / fill_batched
    rng = np.random.default_rng(11)
    S = 96
    for trial in range(4):
        nt = 220
        c = rng.uniform(-10, S + 10, (nt, 1, 2)).astype(np.float32)
        ext = rng.choice([1.5, 4.0, 20.0, 150.0], (nt, 1, 1)).astype(
            np.float32)
        p = (c + rng.uniform(-1, 1, (nt, 3, 2)).astype(np.float32) * ext)
        sx = np.ascontiguousarray(p[:, :, 0])
        sy = np.ascontiguousarray(p[:, :, 1])
        z = rng.uniform(-1.2, 1.4, (nt, 3)).astype(np.float32)
        z[5] = np.nan
        z[6, 1] = np.nan
        z[7] = np.inf
        z[8] = -np.inf
        z[9, 0] = np.inf
        iw = np.ones((nt, 3), np.float32)
        bw = np.broadcast_to(np.eye(3, dtype=np.float32), (nt, 3, 3)).copy()
        src = np.arange(nt, dtype=np.int32)
        for cull in ('NONE', 'BACK', 'FRONT'):
            for fill in (CRg.fill, CRg.fill_batched):
                gb = CRg.GBuffer(S, S)
                with np.errstate(all='ignore'):
                    fill(gb, sx, sy, iw, z, bw, src, cull=cull,
                         depth_bits=32)
                    d = np.full(S * S, np.inf, np.float32)
                    for pix, zz in CRg.depth_fragments(sx, sy, z, S, S,
                                                       cull):
                        np.minimum.at(d, pix, zz)
                mine = np.where(d == np.inf, np.float32(1.0), d).reshape(S, S)
                if not np.array_equal(mine, gb.zndc, equal_nan=True):
                    bad += 1
    check('depth_fragments is the fills\' zndc with NaN, +inf and -inf '
          'depths, under every cull mode', bad == 0, f'{bad} differ')
    blank = LI._render_depth(sc.mesh.verts, sc.mesh.tris[:0],
                             np.eye(4, dtype=np.float32), 64)
    sm = LI.ShadowMap(np.eye(4, dtype=np.float32), blank, 0.1, 10.0, True,
                      64, (0, 0, 0))
    ref = LI.ShadowMap(np.eye(4, dtype=np.float32),
                       np.ones((64, 64), np.float32), 0.1, 10.0, True, 64,
                       (0, 0, 0))
    check('a map nothing landed on is the map of a plane of ones, and '
          'holds no plane of its own',
          CRg.is_blank(blank) and np.array_equal(sm.depth, ref.depth)
          and sm.d_min == ref.d_min and LI._map_bytes(sm) == 0
          and LI._map_bytes(ref) == 64 * 64 * 4)


def test_b_the_reject_only_drops_what_cannot_land():
    """verify-highpoly-triangle-bound-14: every triangle of the scene went
    through the clipper of every map."""
    st = _settings(128)
    sc = _scene([_lamp('POINT', pos=(0.3, -0.5, 2.5))], st, seg=64)
    LI._build_shadow_maps(sc, st)
    V, T = sc.mesh.verts, sc.mesh.tris
    kept = []
    same = True
    for f in sc.lights[0].shadow_map.faces:
        clip = CRg.clip_positions(V, f.vp)
        keep = CRg.frustum_keep(clip, T, f.size, f.size)
        kept.append(float(keep.mean()))
        old = CRg.DEPTH_REJECT_MIN_TRIS
        try:
            CRg.DEPTH_REJECT_MIN_TRIS = 10 ** 9
            ref = CRg.rasterize_depth(V, T, f.vp, f.size, f.size)
            CRg.DEPTH_REJECT_MIN_TRIS = 0
            new = CRg.rasterize_depth(V, T, f.vp, f.size, f.size)
        finally:
            CRg.DEPTH_REJECT_MIN_TRIS = old
        same = same and np.array_equal(ref, new)
        # what the reject drops owns no fragment by itself either
        if (~keep).any():
            alone = CRg.rasterize_depth(V, T[~keep], f.vp, f.size, f.size)
            same = same and CRg.is_blank(alone)
    check('a cube face keeps the triangles of its own frustum (well under '
          'the whole mesh) and every face is bitwise the unrejected bake',
          same and max(kept) < 0.75 and min(kept) < 0.3,
          ' '.join(f'{k:.2f}' for k in kept))
    # triangles hugging the planes: a soup crossing the near plane and the
    # frustum's sides at every scale
    rng = np.random.default_rng(3)
    vp = sc.lights[0].shadow_map.faces[0].vp
    bad = 0
    for scale in (0.02, 0.3, 3.0):
        C = rng.uniform(-3, 3, (9000, 3)).astype(np.float32) \
            + np.array([0.3, -0.5, 2.5], np.float32)
        Vs = (np.repeat(C, 3, 0)
              + rng.uniform(-scale, scale, (27000, 3))).astype(np.float32)
        Ts = np.arange(27000, dtype=np.int32).reshape(-1, 3)
        ref = _gbuffer_depth(Vs, Ts, vp, 96)
        new = LI._render_depth(Vs, Ts, vp, 96)
        bad += not np.array_equal(ref, new)
        for mid in (True,):
            bad += not _eq_depth(_gbuffer_depth(Vs, Ts, vp, 96, midpoint=mid),
                                 LI._render_depth(Vs, Ts, vp, 96,
                                                  midpoint=mid))
    check('a soup straddling the near plane and the sides at three scales '
          'bakes bitwise the G-buffer road', bad == 0, f'{bad} differ')
    # the box test: a caster box the map cannot see returns blank at once
    lo, hi = V.min(axis=0), V.max(axis=0)
    faces = sc.lights[0].shadow_map.faces         # the lamp is above the box
    above = [LI._box_misses((lo, hi), f.vp, f.size) for f in faces]
    st_in = _settings(128)
    sc_in = _scene([_lamp('POINT', pos=(0.3, -0.5, 1.0))], st_in)
    LI._build_shadow_maps(sc_in, st_in)
    inside = [LI._box_misses((lo, hi), f.vp, f.size)
              for f in sc_in.lights[0].shadow_map.faces]
    far_box = (np.array([50.0, 50.0, 50.0]), np.array([51.0, 51.0, 51.0]))
    miss = [LI._box_misses(far_box, f.vp, f.size) for f in faces]
    check('a box the lamp stands inside is never judged; a lamp above the '
          'box skips the face that looks up; a far box misses the three '
          'faces that look away from it',
          not any(inside)
          and above == [False, False, False, False, True, False]
          and miss == [False, True, False, True, False, True],
          f'{inside} {above} {miss}')
    with _spy() as calls:
        z = LI._render_depth(V, T, faces[4].vp, faces[4].size,
                             bounds=(lo, hi))
    check('...and that face comes back blank without a vertex being read',
          CRg.is_blank(z) and np.array_equal(
              np.asarray(z), _gbuffer_depth(V, T, faces[4].vp,
                                            faces[4].size)))


def test_c_maps_are_baked_for_the_lamps_something_reads():
    """dead-lamp-shadow-bake, limit-maps-perf-01, perf-limit-bake-01,
    verify-light-limit-bakes-12, hemi-cube-bake-01,
    screen-spot-map-bake-01."""
    def run(lights, st=None, mats=None, mutate=None, sc_fn=None):
        """(frame, bakes, maps) under the rule and with every map baked."""
        out = []
        for every in (False, True):
            s = copy.copy(st) if st is not None else _settings(64)
            sc = sc_fn(lights(), s) if sc_fn else _scene(lights(), s,
                                                         mats=mats)
            if mutate:
                mutate(sc, s)
            LI.clear_shadow_cache()
            with _spy() as calls:
                if every:
                    with _bake_every_lamp():
                        img = np.asarray(R.render(sc, s))
                else:
                    img = np.asarray(R.render(sc, s))
            out.append((img, len(calls),
                        [l.shadow_map is not None for l in sc.lights], sc, s))
        return out

    # the Light Limit
    def twelve():
        return [_lamp('POINT', pos=(4 * np.cos(a), 4 * np.sin(a), 4.0),
                      name=f'P{i}', energy=200.0 + 30 * i)
                for i, a in enumerate(np.linspace(0, 2 * np.pi, 12,
                                                  endpoint=False))]
    (a, na, ma, _s, _t), (b, nb, _mb, _s2, _t2) = run(twelve)
    check('12 point lamps under Light Limit 8 bake the 8 cubes that are '
          'shaded (48 faces; every lamp: 72) and the frame is the same',
          na == 48 and nb == 72 and sum(ma) == 8 and np.array_equal(a, b),
          f'{na} / {nb} faces')

    # a Hemi and a Screen Spotlight cast no shadow
    def hemi():
        return [_lamp('POINT'), _lamp('HEMI', pos=(0, 0, 6), name='H',
                                      energy=0.5)]
    (a, na, ma, _s, _t), (b, nb, _mb, _s2, _t2) = run(hemi)
    check('a Hemi lamp left on Shadow Map bakes nothing (it baked a six-'
          'face cube nothing read)',
          na == 6 and nb == 12 and ma == [True, False]
          and np.array_equal(a, b), f'{na} / {nb} faces')

    def screen():
        return [_lamp('POINT'), _lamp('SPOT', pos=(-3, -2, 4), name='SS',
                                      screen_spot=True)]
    (a, na, ma, _s, _t), (b, nb, _mb, _s2, _t2) = run(screen)
    check('a Screen Spotlight bakes nothing while only surfaces look',
          na == 6 and nb == 7 and ma == [True, False]
          and np.array_equal(a, b), f'{na} / {nb} faces')

    # lamps that add no light
    def dark():
        return [_lamp('POINT'),
                _lamp('POINT', pos=(-3, -2, 4), name='Off', energy=0.0),
                _lamp('SPOT', pos=(0, 3, 5), name='Black', color=(0, 0, 0)),
                _lamp('POINT', pos=(2, 2, 4), name='NoLobe',
                      diffuse_only=True, specular_only=True)]
    (a, na, ma, sc, s), (b, nb, _mb, _s2, _t2) = run(dark)
    check('Power 0, a black colour, no lobe left: no map, the same frame',
          na == 6 and nb == 19 and ma == [True, False, False, False]
          and np.array_equal(a, b), f'{na} / {nb} faces')
    check('...and none of the three is asked for a shadow by the surface '
          'loop (LI.map_deferred: still only promised)',
          [LI.map_deferred(l) for l in sc.lights]
          == [False, True, True, True])
    # ...unless something reads a shadow apart from the lamp's light
    for label, kw, mats in (
            ('the Shadow pass', {'pass_shadow': True}, None),
            ('a Shadows Only material', {}, [
                Material(name='Catcher', index=0,
                         graph=_bi_graph({'shadow_only': True})),
                Material(name='Ball', index=1, model='PHONG',
                         diffuse=(0.8, 0.3, 0.2))]),
            ('the paint (CARTOON keeps the strongest lamp\'s verdict)', {}, [
                Material(name='Floor', index=0, model='CARTOON',
                         diffuse=(0.8, 0.8, 0.8)),
                Material(name='Ball', index=1, model='PHONG',
                         diffuse=(0.8, 0.3, 0.2))])):
        st = _settings(64, **kw)
        (a, na, ma, _s, _t), (b, nb, _mb, _s2, _t2) = run(
            lambda: dark()[:2], st=st, mats=mats)
        with _bake_every_lamp():
            pass
        st0 = _settings(64, **kw)
        sc0 = _scene([dark()[0]], st0, mats=mats)
        alone = _cold(sc0, st0)
        check(f'{label} reads a Power 0 lamp\'s shadow: its map is baked '
              'and the frame is the every-map frame',
              ma == [True, True] and na == nb == 12
              and np.array_equal(a, b))
        if 'paint' in label or 'Only' in label:
            check('   (and that shadow is in the picture: the frame is not '
                  'the frame without the lamp)',
                  not np.array_equal(a, alone))
    quiet = R.shadow_read_apart_from_light
    st = _settings(64)
    sc = _scene([_lamp('POINT')], st)
    said = [quiet(sc, st)]
    for props in ({'light_source': 'CAMERA'}, {'shadow_only': True},
                  {'model': 'CARTOON'}):
        sc.materials[1] = Material(name='M', index=1, graph={
            'output': 'out', 'nodes': {'n': {
                'id': 'n', 'bl_idname': 'HALCYON_AnimeShaderNode',
                'props': props, 'inputs': [], 'outputs': []}}})
        said.append(quiet(sc, st))
    sc.materials[1] = Material(name='M', index=1, graph={
        'output': 'out', 'nodes': {'n': {
            'id': 'n', 'bl_idname': 'HALCYON_ConsoleShaderNode',
            'props': {}, 'inputs': [], 'outputs': []}}})
    said.append(quiet(sc, st))
    st.default_model = 'CARTOON'
    sc.materials[1] = Material(name='Ball', index=1, model='PHONG')
    said.append(quiet(sc, st))
    check('shadow_read_apart_from_light: plain materials are quiet; a cel '
          'key, Shadows Only, the paint model, a Console node and a paint '
          'default model are not', said == [False, True, True, True, True,
                                            True], str(said))

    # the readers the rule does not name still get their map: on demand
    def vol_scene(lights, st):
        sc = demo_scene(st, with_texture=False)
        sc.world.ambient = (0.0, 0.0, 0.0)
        sc.lights = lights
        return _with_volume(sc, _vol_graph(Density=0.35),
                            lo=(-3.0, -3.0, 0.0), hi=(3.0, 3.0, 3.2))

    def sun_point():
        return [Light(type='SUN', name='Sun', direction=(-0.4, 0.35, -0.85),
                      energy=2.0, shadow='NONE'),
                Light(type='POINT', name='Pt', position=(-1.3, 0.2, 6.0),
                      energy=3000.0, shadow='MAP')]
    st = base_settings(W, H, max_lights=1, light_limit_mode='FIRST')
    st.shadow_map_size = 64
    st.global_ambient = (0.0, 0.0, 0.0)
    (a, na, ma, sc, s), (b, nb, _mb, _s2, _t2) = run(sun_point, st=st,
                                                     sc_fn=vol_scene)
    sc2 = vol_scene([l for l in sun_point()], copy.copy(st))
    sc2.lights[1].shadow = 'NONE'
    unshadowed = _cold(sc2, sc2.settings if False else copy.copy(st))
    check('a lamp the Light Limit dropped still shadows a volume it '
          'lights: the march\'s first lookup bakes its map (the frame is '
          'the every-map frame, and not the unshadowed one)',
          ma == [False, True] and na == nb == 6 and np.array_equal(a, b)
          and not np.array_equal(a, unshadowed))
    LI.clear_shadow_cache()
    sc3 = vol_scene(sun_point(), copy.copy(st))
    s3 = copy.copy(st)
    with _spy() as calls:
        R._build_shadows(sc3, s3, sc3.mesh)
    check('...and not before: the frame\'s own build leaves it a promise',
          not calls and sc3.lights[1].shadow_map is None
          and LI.map_deferred(sc3.lights[1]))

    def ground(sc, st):
        sc.world.ground_plane = True
        sc.world.ground_lighting = 1.0
        sc.world.ground_height = -0.5
    (a, na, ma, _s, _t), (b, nb, _mb, _s2, _t2) = run(screen, mutate=ground)
    check('the ground plane under a Screen Spotlight keeps that lamp\'s '
          'shadow (it lights the ground as a spot): baked on demand',
          ma == [True, True] and np.array_equal(a, b))

    def off():
        return dark()[:2]
    (a, na, ma, _s, _t), (b, nb, _mb, _s2, _t2) = run(off, mutate=ground)
    check('...and the ground does not ask a lamp of no radiance',
          ma == [True, False] and np.array_equal(a, b))

    # a promise is not data: a pickle carries none, a copy shares it
    st = _settings(64, max_lights=1, light_limit_mode='FIRST')
    sc = _scene([_lamp('POINT'), _lamp('POINT', pos=(-3, -2, 4), name='B')],
                st)
    LI.clear_shadow_cache()
    R._build_shadows(sc, st, sc.mesh)
    twin = pickle.loads(pickle.dumps(sc.lights[1]))
    cp = copy.deepcopy(sc.lights[1])
    P = np.array([[0.5, 0.2, 0.0], [2.0, -1.0, 0.0]], np.float32)
    N = np.tile(np.array([[0, 0, 1.0]], np.float32), (2, 1))
    L, _rad, dist = LI.sample(cp, P, st)
    v_copy = LI.visibility(cp, P, N, L.astype(np.float32), dist, st)
    v_orig = LI.visibility(sc.lights[1], P, N, L.astype(np.float32), dist,
                           st)
    check('a deferred lamp pickles without its promise; a deep copy keeps '
          'it and gets the same map on its first lookup',
          not LI.map_deferred(twin) and twin.shadow_map is None
          and cp.shadow_map is not None
          and sc.lights[1].shadow_map is cp.shadow_map
          and np.array_equal(v_copy, v_orig))

    # the same answer on the GPU road
    for label, lights in (('a dark lamp', off), ('a Screen Spotlight',
                                                screen), ('a Hemi', hemi)):
        frames = []
        for every in (False, True):
            st = _settings(64)
            sc = _scene(lights(), st)
            LI.clear_shadow_cache()
            if every:
                with _bake_every_lamp():
                    frames.append(_gpu(sc, st))
            else:
                frames.append(_gpu(sc, st))
        (cpu, sim, cov, _p, why), (_c2, sim2, _cov2, _p2, _w2) = frames
        if not check(f'{label}: the deferred frame plans', sim is not None
                     and sim2 is not None, str(why)):
            continue
        check(f'{label}: the simulated frame is the every-map frame and the '
              'CPU frame within the deferred bar',
              np.array_equal(sim, sim2)
              and float(np.abs(sim[cov] - cpu[cov][:, :3]).max()) < 6e-3,
              f'{float(np.abs(sim[cov] - cpu[cov][:, :3]).max()):.6f}')

    # the traced road: a lamp that adds nothing is not asked either
    asked = []
    real = LI.visibility

    def spy(light, *a, **k):
        asked.append(light.name)
        return real(light, *a, **k)
    st = _settings(64, shadow_default='RAY')
    sc = _scene(off(), st)
    st1 = _settings(64, shadow_default='RAY')
    one = _cold(_scene(off()[:1], st1), st1)
    LI.visibility = spy
    try:
        two = _cold(sc, st)
    finally:
        LI.visibility = real
    check('Ray Traced: a Power 0 lamp traces no shadow ray and the frame '
          'is the frame without it', 'Off' not in asked and 'L' in asked
          and np.array_equal(one, two))


def test_d_a_key_holds_what_its_bake_reads():
    """verify-key-overreach-11: position, direction and cone sat in every
    lamp's key whatever its kind; the raw render settings in every base
    key."""
    def scene3(st):
        return _scene([_lamp('POINT', name='P'),
                       _lamp('SUN', pos=(1, 2, 6), name='S'),
                       _lamp('AREA', pos=(-3, -2, 4), name='A'),
                       _lamp('SPOT', pos=(0, 3, 5), name='T')], st)
    st = _settings(64)
    sc = scene3(st)
    _cold(sc, st)
    P, S, A, T = sc.lights

    def edit(label, fn, want, st_fn=None):
        if st_fn:
            st_fn(st)
        fn()
        before = [l.shadow_map for l in sc.lights]
        with _spy() as calls:
            img = np.asarray(R.render(sc, st))
        kept = [a is b for a, b in zip(before,
                                       [l.shadow_map for l in sc.lights])]
        fresh = _cold(scene_like(), st)
        ok = len(calls) == want and np.array_equal(img, fresh)
        check(f'{label}: {want} faces rasterised and the frame is a cold '
              'frame', ok, f'{len(calls)} faces, kept {kept}')
        np.asarray(R.render(sc, st))          # re-park this scene's maps

    def scene_like():
        s2 = scene3(st)
        for a, b in zip(s2.lights, sc.lights):
            a.__dict__.update({k: v for k, v in b.__dict__.items()
                               if k not in ('shadow_map', '_shadow_key',
                                            '_shadow_defer')})
        return s2

    edit('a Sun translated', lambda: setattr(S, 'position', (9.0, 9.0, 2.0)),
         0)
    edit('a Point lamp rotated',
         lambda: setattr(P, 'direction', (0.0, 0.6, -0.8)), 0)

    def aim():
        A.direction = (0.3, 0.2, -0.93)
        x = np.cross(np.array(A.direction), np.array([0, 0, 1.0]))
        x /= np.linalg.norm(x)
        A.area_x = tuple(x)
        A.area_y = tuple(np.cross(x, np.array(A.direction)))
    edit('an Area lamp aimed', aim, 0)
    edit('a Point lamp\'s Spot Size edited',
         lambda: setattr(P, 'spot_size', 0.4), 0)
    edit('a Spot\'s cone law changed between laws of one reach',
         lambda: setattr(T, 'spot_law', 'GX_COS'), 0)
    edit('a lamp\'s Map Depth set to the render\'s own value',
         lambda: setattr(P, 'shadow_map_depth', 'CLASSIC'), 0)
    edit('Shadow Method Shadow Maps -> Per Light (every lamp is on maps)',
         lambda: None, 0,
         st_fn=lambda s: setattr(s, 'shadow_default', 'PER_LIGHT'))
    for l in sc.lights:
        l.shadow_map_size = 64
    np.asarray(R.render(sc, st))
    edit('the render\'s Map Size doubled under lamps that set their own',
         lambda: None, 0,
         st_fn=lambda s: setattr(s, 'shadow_map_size', 128))
    edit('the lamps recoloured and re-powered', lambda: (
        setattr(P, 'color', (1.0, 0.5, 0.2)), setattr(T, 'energy', 900.0)),
        0)
    # the controls: what a bake reads still re-bakes it, and only it
    edit('a Sun rotated', lambda: setattr(S, 'direction',
                                          (0.2, -0.3, -0.93)), 1)
    edit('a Point lamp moved', lambda: setattr(P, 'position',
                                               (3.2, -3.0, 5.0)), 6)
    edit('an Area lamp moved', lambda: setattr(A, 'position',
                                               (-3.0, -2.2, 4.0)), 6)
    edit('a Spot aimed', lambda: setattr(T, 'direction',
                                         (0.0, -0.5, -0.87)), 1)
    edit('a Spot\'s law set to GX Ring 2 (the map widens)',
         lambda: setattr(T, 'spot_law', 'GX_RING2'), 1)
    edit('a Point lamp\'s ClipEnd set', lambda: setattr(
        P, 'shadow_clip_end', 9.0), 6)
    edit('a lamp\'s own Map Depth set to MIDPOINT', lambda: setattr(
        S, 'shadow_map_depth', 'MIDPOINT'), 1)
    sig = LI._light_shadow_signature
    check('the signature resolves what the bake resolves: a lamp that '
          'inherits follows the render, a lamp that overrides does not',
          sig(_lamp('SUN'), _settings(64)) != sig(_lamp('SUN'),
                                                  _settings(128))
          and sig(_lamp('SUN', shadow_map_size=64), _settings(64))
          == sig(_lamp('SUN', shadow_map_size=64), _settings(128))
          and sig(_lamp('POINT'), _settings(64))
          == sig(_lamp('POINT'), _settings(64, shadow_default='PER_LIGHT')))
    # the caster set and the topology are read whole
    st = _settings(64)
    sc = _scene([_lamp('POINT')], st)
    n = sc.mesh.tris.shape[0]
    a = np.arange(n, dtype=np.int64)
    c1 = np.delete(a, [10, 21])
    c2 = np.delete(a, [11, 20])                 # same count, same index sum
    b1 = LI._shadow_base_signature(sc, st, c1)
    b2 = LI._shadow_base_signature(sc, st, c2)
    t0 = LI._shadow_base_signature(sc, st, None)
    sc.mesh.tris = sc.mesh.tris[::-1].copy()
    t1 = LI._shadow_base_signature(sc, st, None)
    check('two caster sets of equal count and equal index sum have '
          'different keys, and so do two orders of one triangle list',
          b1 != b2 and t0 != t1)


def test_e_a_viewport_draft_bakes_draft_sized_maps():
    """perf-viewport-fullsize-bake-06 / perf-draft-bake-01: a draft shrank
    the picture and baked full-size maps for it."""
    st = _settings(256)
    sc = _scene([_lamp('POINT'), _lamp('SUN', pos=(1, 2, 6), name='S',
                                       shadow_map_size=512),
                 _lamp('SPOT', pos=(-3, -2, 4), name='T',
                       shadow_map_size=64)], st)
    LI.clear_shadow_cache()
    with _spy() as calls:
        np.asarray(R.render(sc, st))
    full = [(f.size, l.shadow_map) for l in sc.lights
            for f in _faces(l.shadow_map)[:1]]
    draft = st.copy()
    draft._shadow_map_div = 4
    draft.shadow_softness = float(st.shadow_softness) / 4
    sc.settings = draft
    with _spy() as dcalls:
        np.asarray(R.render(sc, draft))
    small = [_faces(l.shadow_map)[0].size for l in sc.lights]
    check('a draft at divisor 4 bakes maps a quarter of the side (never '
          'under 64): 256 / 512 / 64 -> 64 / 128 / 64',
          [s for s, _m in full] == [256, 512, 64] and small == [64, 128, 64]
          and sorted({c[1] for c in dcalls}) == [64, 128],
          f'{small}')
    sc.settings = st
    with _spy() as rcalls:
        np.asarray(R.render(sc, st))
    back = [l.shadow_map for l in sc.lights]
    with _spy() as d2calls:
        np.asarray(R.render(sc, draft))
    check('the refine after the draft is served its own full-size maps, '
          'and the next draft its draft maps: nothing is rasterised twice',
          not rcalls and not d2calls
          and all(a is b for a, (_s, b) in zip(back, full)))
    check('a settings copy carries the divisor through the roads inside a '
          'frame, and a settings object that never had one copies none',
          draft.copy()._shadow_map_div == 4
          and '_shadow_map_div' not in st.copy().__dict__)
    # the blur keeps its world width: the Softness is in texels. Measured
    # as the Sun's shadow mass on the floor: with the Softness divided the
    # draft keeps the refine's; undivided, four times the texels of a map
    # a quarter of the side is sixteen times the blur and the shadow of a
    # size-0 lamp erodes away
    gx, gy = np.meshgrid(np.linspace(-5, 5, 120), np.linspace(-4, 4, 100))
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                 1).astype(np.float32)
    N = np.tile(np.array([[0, 0, 1.0]], np.float32), (P.shape[0], 1))

    def mass(s, soft):
        s = copy.copy(s)
        s.shadow_softness = soft
        l = sc.lights[1]
        sc.settings = s
        np.asarray(R.render(sc, s))
        L, _r, dist = LI.sample(l, P, s)
        v = LI.visibility(l, P, N, L.astype(np.float32), dist, s)
        return float((1.0 - v).sum())
    m_full = mass(st, 4.0)
    m_draft = mass(draft, 1.0)
    m_wrong = mass(draft, 4.0)
    check('Softness / 4 on the quarter-size map keeps the refine\'s '
          'shadow (undivided, the blur is sixteen times the refine\'s and '
          'the shadow erodes)',
          m_full > 50 and abs(m_draft - m_full) <= 0.15 * m_full
          and abs(m_wrong - m_full) > 0.3 * m_full,
          f'{m_full:.0f} / {m_draft:.0f} / undivided {m_wrong:.0f}')
    src = open(__file__.replace('tests/test_r254_bake_speed.py',
                                'preview.py').replace(
        'tests\\test_r254_bake_speed.py', 'preview.py'),
        encoding='utf-8').read()
    check('preview.py sets the divisor and divides the Softness on a '
          'draft\'s settings copy only',
          'settings._shadow_map_div = div' in src
          and 'float(settings.shadow_softness) / div' in src
          and src.index('if draft:') < src.index('_shadow_map_div'))


def test_f_a_moved_object_rerasterises_itself():
    """verify-object-move-layers-13: any vertex change missed every lamp
    and every map re-rasterised every caster."""
    def build(st, **kw):
        return _scene([_lamp('POINT', name='P'),
                       _lamp('SUN', pos=(1, 2, 6), name='S'),
                       _lamp('SPOT', pos=(-3, -2, 4), name='T'),
                       _lamp('AREA', pos=(0, 3, 5), name='A')], st, **kw)
    count = {}
    st = _settings(96)
    sc = build(st, seg=24)
    mesh = sc.mesh
    n_all = count['casters'] = int(mesh.tris.shape[0])
    ball = int(mesh.obj_index.max())
    bv = np.unique(mesh.tris[mesh.obj_index == ball])
    n_ball = int((mesh.obj_index == ball).sum())
    n_other = int((mesh.obj_index == 2).sum())
    other = np.unique(mesh.tris[mesh.obj_index == 2])
    LI.clear_shadow_cache()
    np.asarray(R.render(sc, st))

    def step(label, fn, want_max=None, want_full=False, first=False):
        n_all = count['casters']
        fn()
        with _spy() as calls:
            img = np.asarray(R.render(sc, st))
        got = _maps(sc)
        saved = LI._SHADOW_CACHE
        LI._SHADOW_CACHE = LI._ShadowCache()
        try:
            fresh = np.asarray(R.render(sc, st))
            ref = _maps(sc)
        finally:
            LI._SHADOW_CACHE = saved
        np.asarray(R.render(sc, st))
        biggest = max((c[0] for c in calls), default=0)
        ok = _same_maps(got, ref) and np.array_equal(img, fresh)
        if want_full:
            ok = ok and biggest == n_all
        elif first:
            ok = ok and biggest == n_all - want_max
        elif want_max is not None:
            ok = ok and 0 < biggest <= want_max
        check(f'{label}: every map and the frame are a cold bake\'s, '
              'bitwise', ok,
              f'largest raster {biggest} of {n_all} triangles, '
              f'{len(calls)} faces')

    def move(v, dx, axis=0):
        def fn():
            mesh.verts[v, axis] += dx
        return fn
    step('the ball moves: the first step bakes the layer of everything '
         'else', move(bv, 0.2), want_max=n_ball, first=True)
    for k in range(3):
        step(f'the ball moves again ({k + 2}): only its own triangles are '
             'rasterised', move(bv, 0.2), want_max=n_ball)
    step('the ball goes back where it started (a cache hit or its own '
         'triangles)', move(bv, -0.8), want_max=n_ball)
    step('another object starts moving: a new layer, of everything but '
         'it', move(other, 0.3), want_max=n_other, first=True)
    step('...then it and the ball together: a layer of everything but the '
         'two', lambda: (move(other, 0.1)(), move(bv, 0.1, axis=1)()),
         want_max=n_other + n_ball, first=True)
    step('...and the two of them move on: only their own triangles',
         lambda: (move(other, 0.1)(), move(bv, 0.1, axis=1)()),
         want_max=n_other + n_ball)
    step('the ball leaves the scene\'s bounds (every far plane and the '
         'Sun\'s frame move): whole bakes', move(bv, 12.0), want_full=True)
    step('...and while it keeps moving out there it keeps moving the '
         'bounds: whole bakes', move(bv, 0.1), want_full=True)
    step('back inside', move(bv, -12.1), want_full=True)
    big = np.unique(mesh.tris[np.isin(mesh.obj_index, (1, 2, 3))])
    step('three of four spheres move at once (over a third of the '
         'casters): whole bakes', move(big, 0.05), want_full=True)

    # MIDPOINT maps are never layered; nor is a mesh without objects
    st = _settings(96, shadow_map_depth='MIDPOINT')
    sc = build(st)
    mesh = sc.mesh
    count['casters'] = int(mesh.tris.shape[0])
    bv = np.unique(mesh.tris[mesh.obj_index == int(mesh.obj_index.max())])
    LI.clear_shadow_cache()
    np.asarray(R.render(sc, st))
    for k in range(2):
        step(f'MIDPOINT, the ball moves ({k + 1}): whole bakes',
             move(bv, 0.2), want_full=True)
    st = _settings(96)
    sc = build(st)
    mesh = sc.mesh
    bv = np.unique(mesh.tris[mesh.obj_index == int(mesh.obj_index.max())])
    mesh.obj_index = None
    LI.clear_shadow_cache()
    np.asarray(R.render(sc, st))
    for k in range(2):
        step(f'no object index, a block of vertices moves ({k + 1}): whole '
             'bakes', move(bv, 0.2), want_full=True)

    # a caster subset, and a non-caster that moves
    st = _settings(96)
    sc = build(st)
    mesh = sc.mesh
    count['casters'] = int((mesh.obj_index != 1).sum())
    sc.objects[1].cast_shadow = False
    ball = int(mesh.obj_index.max())
    bv = np.unique(mesh.tris[mesh.obj_index == ball])
    ghost = np.unique(mesh.tris[mesh.obj_index == 1])
    LI.clear_shadow_cache()
    np.asarray(R.render(sc, st))
    step('a caster subset, the ball moves', move(bv, 0.2),
         want_max=n_ball, first=True)
    step('a caster subset, the ball moves again', move(bv, 0.2),
         want_max=n_ball)
    step('the object that does not cast moves: a new layer of every '
         'caster', move(ghost, 0.2), want_max=0, first=True)
    with _spy() as calls:
        mesh.verts[ghost, 0] += 0.1
        np.asarray(R.render(sc, st))
    check('...and when it moves again no triangle is rasterised at all',
          max((c[0] for c in calls), default=0) == 0)


def test_g_the_cache_is_bounded_by_bytes():
    """perf-cache-hoard-07 / perf-cache-bytes-01 / size-map-v03: one map
    per pose was kept until 65 entries existed, then everything -- the
    maps of the lamps that had not moved included -- was dropped."""
    st = _settings(32)
    sc = _scene([_lamp('POINT', name='Moves'),
                 _lamp('POINT', pos=(-3, -2, 4), name='Still')], st, seg=8)
    LI.clear_shadow_cache()
    np.asarray(R.render(sc, st))
    still = sc.lights[1].shadow_map
    rebakes = 0
    for k in range(70):
        sc.lights[0].position = (3.0 + 0.01 * (k + 1), -3.0, 5.0)
        LI.build_shadow_maps(sc, st)
        rebakes += sc.lights[1].shadow_map is not still
    check('70 nudges of one lamp never re-bake the lamp that did not move '
          '(the 65th used to)', rebakes == 0, f'{rebakes} re-bakes')
    old = LI.SHADOW_CACHE_BYTES
    try:
        LI.SHADOW_CACHE_BYTES = 3 * 6 * 32 * 32 * 4
        LI.clear_shadow_cache()
        held = []
        for k in range(12):
            sc.lights[0].position = (3.0 + 0.01 * (k + 1), -3.0, 5.0)
            LI.build_shadow_maps(sc, st)
            held.append(LI._SHADOW_CACHE.held_bytes())
        frame = sum(LI._map_bytes(l.shadow_map) for l in sc.lights)
        check('the depth held stays under the budget plus the frame\'s own '
              'maps however long the drag',
              max(held) <= LI.SHADOW_CACHE_BYTES + frame
              and len(LI._SHADOW_CACHE) < 8,
              f'{max(held)} bytes, {len(LI._SHADOW_CACHE)} entries')
        LI.SHADOW_CACHE_BYTES = 0
        LI.clear_shadow_cache()
        with _spy() as calls:
            LI.build_shadow_maps(sc, st)
            first = len(calls)
            LI.build_shadow_maps(sc, st)
        check('a budget of zero still keeps what the frame in hand uses: '
              'the same frame again rasterises nothing',
              first == 12 and len(calls) == 12
              and len(LI._SHADOW_CACHE) == 2)
    finally:
        LI.SHADOW_CACHE_BYTES = old
    LI.clear_shadow_cache()
    check('clear_shadow_cache empties the maps and the layers',
          len(LI._SHADOW_CACHE) == 0 and not LI._SHADOW_CACHE.layers)


def test_h_the_bake_threads_by_texels_too():
    """perf-thread-gate-09: threads only above 50,000 triangles, while the
    cost follows texels as well."""
    used = []
    real = fut.ThreadPoolExecutor

    class Spy(real):
        def __init__(self, *a, **k):
            used.append(k.get('max_workers'))
            super().__init__(*a, **k)

    def bake(size):
        st = _settings(size)
        sc = _scene([_lamp('POINT')], st, seg=8)
        used.clear()
        fut.ThreadPoolExecutor = Spy
        try:
            LI._build_shadow_maps(sc, st)
        finally:
            fut.ThreadPoolExecutor = real
        return bool(used), sc

    small, sc = bake(128)
    big, sc2 = bake(640)
    check('a few hundred triangles: a 128 cube bakes in line, a 640 cube '
          '(2.4 million texels) in threads',
          sc.mesh.tris.shape[0] < 50000 and not small and big)
    st = _settings(640)
    serial = _scene([_lamp('POINT')], st, seg=8)

    class Line:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, it):
            return [fn(x) for x in it]
    fut.ThreadPoolExecutor = Line
    try:
        LI._build_shadow_maps(serial, st)
    finally:
        fut.ThreadPoolExecutor = real
    check('the threaded faces are bitwise the in-line faces',
          _same_maps(_maps(serial), _maps(sc2)))


def test_i_a_worker_adopts_the_parents_maps():
    """perf-worker-pool-rebake-08: every worker process baked the whole set
    of maps for its own band."""
    st = _settings(96)
    st.resolution_x, st.resolution_y = 160, 120
    sc = _scene([_lamp('POINT'), _lamp('SUN', pos=(1, 2, 6), name='S')], st)
    LI.clear_shadow_cache()
    whole = np.asarray(R.render(sc, st))
    LI.clear_shadow_cache()
    with _spy() as calls:
        baked = R.prebake_shadows(sc, st)
    check('the parent bakes the frame\'s maps once (7 faces)',
          baked and len(calls) == 7
          and all(l.shadow_map is not None for l in sc.lights))
    payload = pickle.dumps(('scene', sc, st),
                           protocol=pickle.HIGHEST_PROTOCOL)
    # the worker's side: an empty cache, the unpickled scene, a band
    LI.clear_shadow_cache()
    _k, sc_w, st_w = pickle.loads(payload)
    LI.adopt_shadow_maps(sc_w)
    with _spy() as wcalls:
        band = np.asarray(R.render(sc_w, st_w, band=(40, 80)))
    check('a worker handed that scene rasterises no map for its band, and '
          'the band is the in-process frame\'s rows',
          not wcalls and np.array_equal(band, whole[40:80]),
          f'{len(wcalls)} faces')
    LI.clear_shadow_cache()
    _k, sc_w, st_w = pickle.loads(payload)
    with _spy() as wcalls:
        np.asarray(R.render(sc_w, st_w, band=(40, 80)))
    check('(without the adoption it bakes all 7 again)', len(wcalls) == 7)
    LI.clear_shadow_cache()
    with _spy() as calls:
        skipped = R.prebake_shadows(sc, st, limit=1000)
    check('a set of maps too large to hand over is not baked by the parent '
          'at all', skipped is False and not calls
          and all(l.shadow_map is None for l in sc.lights))
    check('planned_map_bytes counts the faces the build would bake',
          LI.planned_map_bytes(sc, st) == 7 * 96 * 96 * 4)
    # the real pool, when this interpreter can start one
    from ..core import parallel as P
    st = _settings(96)
    st.resolution_x, st.resolution_y = 320, 240
    sc = _scene([_lamp('POINT'), _lamp('SUN', pos=(1, 2, 6), name='S')], st)
    LI.clear_shadow_cache()
    ref = np.asarray(R.render(sc, st))
    LI.clear_shadow_cache()
    img, err = P.render_parallel(sc, st, workers=2, scene_key='bake-speed')
    if img is None:
        check('the worker pool renders (skipped: unavailable here)', True,
              str(err))
    else:
        check('two worker processes handed the parent\'s maps render the '
              'in-process frame, bitwise',
              np.array_equal(np.asarray(img), ref))
    P.shutdown()


def test_j_the_shape_of_a_map():
    """wide-spot-map-01 / v-spot-wide-beam-maps, ring2-map-reach-01,
    clip-area-01."""
    def shadow_px(lamp_kw, mode, scene_kw=None):
        st = _settings(256, shadow_default=mode)
        st.resolution_x, st.resolution_y = 160, 120
        lit = _lamp('SPOT', pos=(0.0, 0.0, 5.0), aim=(0.0, 0.0, 0.0),
                    **lamp_kw)
        sc = _scene([lit], st, **(scene_kw or {}))
        img = _cold(sc, st)
        off = copy.copy(lit)
        off.shadow = 'NONE'
        off.shadow_map = None
        sc2 = _scene([off], st, **(scene_kw or {}))
        base = _cold(sc2, st)
        dark = (base[..., :3].sum(2) - img[..., :3].sum(2)) > 0.02
        return dark, sc

    for deg in (150.0, 170.0):
        m, sc = shadow_px({'spot_size': np.radians(deg)}, 'MAP')
        r, _s = shadow_px({'spot_size': np.radians(deg)}, 'RAY')
        sm = sc.lights[0].shadow_map
        missing = int((r & ~m).sum())
        check(f'a {deg:.0f} degree Spot bakes a cube and keeps its shadow '
              '(one stretched map lost more than half of it)',
              hasattr(sm, 'faces') and missing < 0.12 * int(r.sum())
              and int((m & ~r).sum()) < 0.12 * int(r.sum()),
              f'{int(r.sum())} px traced, {missing} missing, '
              f'{int((m & ~r).sum())} spurious')
    sm = sc.lights[0].shadow_map
    check('...of which the face behind the cone is never baked (blank)',
          LI._map_bytes(sm.faces[4]) == 0 and LI._map_bytes(sm.faces[5]) > 0)
    check('the field of view a Spot\'s map needs: the cone and 15%, the '
          'diagonal of a Square cone, the ring\'s end under GX Ring 2',
          abs(LI.spot_map_fov(_lamp('SPOT', spot_size=1.0)) - 1.15) < 1e-12
          and LI.spot_map_fov(_lamp('SPOT', spot_size=1.0,
                                    spot_square=True)) > 1.5
          and LI.spot_map_fov(_lamp('SPOT', spot_size=1.0,
                                    spot_law='GX_RING2')) > 1.4
          and LI.spot_map_fov(_lamp('SPOT', spot_size=1.0,
                                    spot_law='GX_RING1')) == 1.15)
    # one map up to 2.2 rad, bitwise the map it always was
    for kw in ({'spot_size': 1.0}, {'spot_size': 1.9},
               {'spot_size': 1.2, 'spot_square': True},
               {'spot_size': 1.0, 'spot_law': 'GX_COS'}):
        st = _settings(96)
        l = _lamp('SPOT', **kw)
        sc = _scene([l], st)
        LI._build_shadow_maps(sc, st)
        sm = l.shadow_map
        centre, radius = LI.scene_bounds(sc.mesh.verts)
        pos = np.asarray(l.position, np.float32)
        d = LI.M.normalize(np.asarray(l.direction, np.float32))
        near = max(radius * 0.005, 1e-3)
        far = max(float(np.linalg.norm(centre - pos)) + radius * 1.5,
                  near * 10)
        if kw.get('spot_square'):
            half_t = float(np.tan(float(l.spot_size) * 0.5)) * np.sqrt(2.0)
            fov = min(max(2.0 * float(np.arctan(half_t)) * 1.15, 0.05), 3.0)
        else:
            fov = min(max(float(l.spot_size) * 1.15, 0.05), 3.0)
        vp = LI._persp(fov, 1.0, near, far) @ LI._look_at_lh(
            pos, pos + d, np.array([0, 0, 1.0], np.float32))
        check(f'a Spot of {kw} keeps its one perspective map, frame for '
              'frame', not hasattr(sm, 'faces')
              and np.array_equal(sm.vp, vp) and sm.near == near
              and sm.far == far
              and sm.extent == float(np.tan(fov * 0.5))
              and np.array_equal(
                  sm.depth, LI.ShadowMap(vp, _gbuffer_depth(
                      sc.mesh.verts, sc.mesh.tris, vp, 96), near, far, True,
                      96, pos, float(np.tan(fov * 0.5))).depth))
    # GX Ring 2 lights 9.5 degrees past a 60 degree cone: the map follows
    st = _settings(128)
    for law, want in (('BLENDER', False), ('GX_RING2', True)):
        l = _lamp('SPOT', pos=(0.0, 0.0, 5.0), aim=(0.0, 0.0, 0.0),
                  spot_size=np.radians(60.0), spot_law=law)
        sc = _scene([l], st)
        LI._build_shadow_maps(sc, st)
        P = np.array([[5.0 * np.tan(np.radians(37.0)), 0.0, 0.0]],
                     np.float32)
        clip = np.append(P[0], 1.0) @ np.asarray(l.shadow_map.vp).T
        inside = bool(abs(clip[0] / clip[3]) <= 1.0
                      and abs(clip[1] / clip[3]) <= 1.0)
        check(f'{law}: a floor point 37 degrees off the axis is '
              f'{"inside" if want else "outside"} the map',
              inside == want)
    # an Area lamp never used the clip planes its panel did not draw
    st = _settings(96)
    a0 = _lamp('AREA')
    a1 = _lamp('AREA', shadow_clip_start=5.4, shadow_clip_end=6.0)
    p1 = _lamp('POINT', shadow_clip_start=5.4, shadow_clip_end=6.0)
    for l in (a0, a1, p1):
        LI._build_shadow_maps(_scene([l], st), st)
    check('ClipSta / ClipEnd left on an Area lamp no longer clip its map; '
          'a Point lamp\'s still do',
          _same_maps([[f.depth for f in a0.shadow_map.faces]],
                     [[f.depth for f in a1.shadow_map.faces]])
          and a1.shadow_map.faces[0].near == a0.shadow_map.faces[0].near
          and p1.shadow_map.faces[0].near == 5.4
          and p1.shadow_map.faces[0].far == 6.0)
    # the cube-mapped Spot on the GPU road
    st = _settings(128)
    sc = _scene([_lamp('SPOT', pos=(0.0, 0.0, 5.0), aim=(0.0, 0.0, 0.0),
                       spot_size=np.radians(150.0))], st)
    LI.clear_shadow_cache()
    cpu, sim, cov, _p, why = _gpu(sc, st)
    if check('a 150 degree Spot: the deferred frame plans', sim is not None,
             str(why)):
        check('...and the simulator shades the CPU picture',
              float(np.abs(sim[cov] - cpu[cov][:, :3]).max()) < 6e-3,
              f'{float(np.abs(sim[cov] - cpu[cov][:, :3]).max()):.6f}')


def test_k_the_small_ones():
    """note-coarse-01, stale-map-same-scene-01."""
    # the note measured the camera from the world origin
    st = RenderSettings()
    st.resolution_x, st.resolution_y = 320, 240
    st.aa_samples = 1
    st.shadow_map_size = 512
    sc = _scene([_lamp('SUN')], st)
    sc.mesh.verts[:, 0] += 1.0               # the scene sits near the origin
    cam_far = look_at_matrix((1.0, -9.0, 6.0), (1.0, 0.3, 0.5))
    sc.camera = Camera(matrix_world=cam_far, lens=35.0, sensor=36.0,
                       clip_start=0.1, clip_end=200.0)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _cold(sc, st)
    said_far = 'one shadow texel spans several output pixels' in out.getvalue()
    sc.camera = Camera(matrix_world=look_at_matrix((1.0, -1.2, 0.9),
                                                   (1.0, 0.3, 0.5)),
                       lens=120.0, sensor=36.0, clip_start=0.05,
                       clip_end=200.0)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        _cold(sc, st)
    said_near = 'one shadow texel spans several output pixels' in out.getvalue()
    check('the coarse-map note is silent for a camera 10 units from a '
          'scene a texel of which is under two of its pixels, and speaks '
          'for a long lens a unit away', not said_far and said_near,
          f'far {said_far}, near {said_near}')
    # a frame that builds no map leaves none on the lamps
    st = _settings(64)
    sc = _scene([_lamp('POINT')], st)
    mapped = _cold(sc, st)
    st.shadow_default = 'RAY'
    st.ray_shadows = False
    again = np.asarray(R.render(sc, st))
    st2 = _settings(64, shadow_default='RAY', ray_shadows=False)
    fresh = _cold(_scene([_lamp('POINT')], st2), st2)
    check('Ray Traced with Ray Shadows off on a Scene that was rendered '
          'with maps: no map is left behind and the frame is the fresh '
          'frame (the old shadow stayed)',
          sc.lights[0].shadow_map is None and np.array_equal(again, fresh)
          and not np.array_equal(again, mapped))


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
    print('all R254 bake-speed checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
