"""R254: the traced-shadow road and the shadow controls around it.

    (A) Per Light + a lamp on Ray Traced casts (the BVH is built for it,
        and only when something will trace)
    (B) the lamp's own Samples overrides the render's; 0 inherits
    (C) the N soft-shadow rays of a pixel are ONE stratified pattern
    (D) Shadow Density lightens traced shadows too
    (E) an Area lamp's object scale multiplies its Size
    (F) an Area lamp of Size 0 lights as a clean small lamp
    (G) a wide sun is sampled over its cap; the angle stops at 179.9 degrees
    (H) a large Seed never bakes an out-of-range integer literal
    (I) accumulation, lens and refine passes trace their own rays
    (J) the panels follow the method that will run

The rig is the audit's: a floor, a plate whose straight edge shadows it
and one lamp, seen from straight above, so frame / unshadowed frame IS
the visibility and a penumbra is a profile along x. Every law is proven
on the CPU road and through the GLSL front-end (plan_frame + simulate at
the deferred bar 6e-3; the accumulation road through the fake device,
whose draws run the driver's own uniform plumbing). The first test pins
the identity of everything untouched against the previous tree when its
zip sits beside the package; the in-tree neutrality proofs (a hard lamp
never reaches the sampler, a default Density returns the bare
visibility, an ordinary sun keeps the disc law line for line) run
either way.

    python -m halcyon.tests.test_r254_ray_road
"""
import inspect
import os
import re
import sys
import traceback
import types
import warnings

import numpy as np

from . import utf8_console
from .test_render import _prev_engine, base_settings
from .scenebuild import (_mesh_concat, demo_scene, look_at_matrix, plane)
from . import featurematrix as FM
from ..core import lights as LI
from ..core import patterns as PT
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..core.settings import RenderSettings
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
PREV_ZIP = 'halcyon-1.92.0.zip'

W, H = 160, 64
LPOS = (-2.0, 0.0, 6.0)
PLATE_Z = 2.0
CAM_H = 14.0
LENS, SENSOR = 100.0, 36.0
PX_PER_UNIT = W / (CAM_H * SENSOR / LENS)
#: a point lamp's shadow edge on the floor
EDGE_X = (0.0 - LPOS[0]) * (PLATE_Z / (LPOS[2] - PLATE_Z))
ROWS = slice(H // 2 - 16, H // 2 + 16)
KINDS = ('POINT', 'SPOT', 'SUN', 'AREA')
#: a size per kind that gives a penumbra a few dozen pixels wide
SIZES = {'POINT': 0.5, 'SPOT': 0.5, 'SUN': 0.2, 'AREA': 1.0}


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def note(text):
    print(f'  note {text}')


# --------------------------------------------------------------------- rig


def _kit(pkg=None):
    """The classes the rig is built from: this tree's, or those of the
    previous engine's package `pkg` (so the previous render() shades a
    scene made of its own dataclasses)."""
    if pkg is None:
        return types.SimpleNamespace(
            Camera=Camera, Light=Light, Material=Material,
            ObjectInfo=ObjectInfo, Scene=Scene, World=World, plane=plane,
            mesh_concat=_mesh_concat, look_at=look_at_matrix,
            base_settings=base_settings)
    import importlib
    sc = importlib.import_module(pkg + '.core.scene')
    sb = importlib.import_module(pkg + '.tests.scenebuild')
    tr = importlib.import_module(pkg + '.tests.test_render')
    return types.SimpleNamespace(
        Camera=sc.Camera, Light=sc.Light, Material=sc.Material,
        ObjectInfo=sc.ObjectInfo, Scene=sc.Scene, World=sc.World,
        plane=sb.plane, mesh_concat=sb._mesh_concat,
        look_at=sb.look_at_matrix, base_settings=tr.base_settings)


def _settings(kit=None, **kw):
    st = (kit or _kit()).base_settings(W, H)
    st.aa_mode = 'NONE'
    st.global_ambient = (0.0, 0.0, 0.0)
    for k, v in kw.items():
        if not hasattr(st, k):
            raise AttributeError(k)
        setattr(st, k, v)
    return st


def _quad(x0, x1, y0, y1, z, mat, obj):
    V = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], f32)
    N = np.tile(np.array([[0, 0, 1.0]], f32), (4, 1))
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], f32)
    T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    return (V, N, UV, T, mat, obj)


def _scene(lights, st, kit=None):
    """The edge scene: floor at z = 0, a plate at z = 2 covering x < 0,
    the camera straight above the shadow edge."""
    K = kit or _kit()
    mats = [K.Material(name='Floor', index=0, model='LAMBERT',
                       diffuse=(0.8, 0.8, 0.8), specular_level=0.0),
            K.Material(name='Plate', index=1, model='LAMBERT',
                       diffuse=(0.8, 0.3, 0.2), specular_level=0.0)]
    mesh = K.mesh_concat([K.plane(z=0.0, size=30.0, mat=0, obj=0),
                          _quad(-9.0, 0.0, -9.0, 9.0, PLATE_Z, 1, 1)])
    mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
    objs = [K.ObjectInfo(name=n, index=i, location=(0, 0, 0),
                         matrix_world=np.eye(4, dtype=f32))
            for i, n in enumerate(('Floor', 'Plate'))]
    cam = K.Camera(matrix_world=K.look_at((EDGE_X, 0.0, CAM_H),
                                          (EDGE_X, 0.0, 0.0), up=(0, 1, 0)),
                   lens=LENS, sensor=SENSOR, clip_start=0.1, clip_end=200.0)
    sc = K.Scene(mesh=mesh, materials=mats, objects=objs, lights=list(lights),
                 camera=cam, world=K.World(color=(0.0, 0.0, 0.0),
                                           ambient=(0.0, 0.0, 0.0)),
                 settings=st)
    sc.images = {}
    return sc


def _lamp(kind, size=0.0, shadow='RAY', pos=LPOS, kit=None, **kw):
    """One lamp at `pos` aimed at the shadow edge. `size` is what the
    user's size dial writes: Radius (POINT / SPOT), Angle (SUN, radians)
    or the edge length (AREA)."""
    Light = (kit or _kit()).Light                               # noqa: N806
    d = np.array([EDGE_X, 0.0, 0.0]) - np.array(pos)
    d = d / np.linalg.norm(d)
    base = dict(type=kind, name='L', position=pos, direction=tuple(d),
                color=(1, 1, 1), shadow=shadow)
    if kind == 'SUN':
        lt = Light(energy=3.0, radius=size, **base)
    elif kind == 'AREA':
        lt = Light(energy=1.0, **base)
        lt.area_size = (size, size)
        y = np.array([0.0, 1.0, 0.0])
        x = np.cross(d, y)
        lt.area_x = tuple(x / np.linalg.norm(x))
        lt.area_y = tuple(y)
    else:
        lt = Light(energy=600.0, radius=size, **base)
        if kind == 'SPOT':
            lt.spot_size = 1.2
            lt.spot_blend = 0.15
    for k, v in kw.items():
        setattr(lt, k, v)
    return lt


def _cpu(sc, st):
    """(frame, the BVH render() built or None, how many it built)."""
    LI.clear_shadow_cache()
    built = []
    real = R._cached_bvh

    def spy(scene, mesh):
        b = real(scene, mesh)
        built.append(b)
        return b
    R._cached_bvh = spy
    try:
        img = np.asarray(R.render(sc, st))
    finally:
        R._cached_bvh = real
    return img, (built[-1] if built else None), len(built)


def _job(sc, st, bvh):
    """The deferred frame's job on a fresh scene, after the prologue
    render() runs before it shades."""
    LI.clear_shadow_cache()
    R.collect_exclusive_lights(sc)
    R._build_shadows(sc, st, sc.mesh)
    st._shadow_cast_tri = R._caster_keep_tri(sc, sc.mesh) \
        if st.shadows and st.ray_shadows else None
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)
    b = R._cached_bvh(sc, sc.mesh) if bvh else None
    return R.ShadeJob(sc, st, R.prepare_textures(sc, st), b, view, eye,
                      W, H), g


def _sim(sc, st, bvh):
    """(rgb, covered, why, passes) through plan_frame + simulate."""
    job, g = _job(sc, st, bvh)
    GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, g)
    if passes is None:
        return None, None, str(why), None
    out, hit = GSH.simulate(job, g, passes, atl)
    if out is None:
        return None, None, str(hit), passes
    return np.asarray(out)[..., :3], g.tri >= 0, None, passes


def _pair(make_lights, st_kw):
    """The same frame on the CPU and through the GLSL front-end:
    (cpu frame, max |sim - cpu| over covered pixels or None, why,
    passes, bvh built)."""
    st = _settings(**st_kw)
    img, bvh, _n = _cpu(_scene(make_lights(), st), st)
    st2 = _settings(**st_kw)
    sim, cov, why, passes = _sim(_scene(make_lights(), st2), st2,
                                 bvh is not None)
    if sim is None:
        return img, None, why, passes, bvh
    d = float(np.abs(sim[cov] - img[cov][:, :3]).max())
    return img, d, None, passes, bvh


def _bar(label, d, why, bar=6e-3):
    check(f'{label}: the GLSL frame is the CPU frame within the deferred '
          'bar', d is not None and d < bar,
          str(why) if d is None else f'max {d:.6f}')


def _floor_x():
    """World x of every pixel column's centre on the floor."""
    return EDGE_X + (np.arange(W) + 0.5 - W / 2.0) / PX_PER_UNIT


def _vis(img, base):
    """(x, mean vis per column, vis per row and column) on the open
    floor right of the plate."""
    lum = img[..., :3].mean(axis=2)
    lb = base[..., :3].mean(axis=2)
    x = _floor_x()
    ok = x > 0.0
    v = np.where(lb[ROWS] > 1e-6, lum[ROWS] / np.maximum(lb[ROWS], 1e-6),
                 0.0)
    return x[ok], v[:, ok].mean(axis=0), v[:, ok]


def _unshadowed(kind, size, **lamp_kw):
    st = _settings(shadows=False)
    return _cpu(_scene([_lamp(kind, size, **lamp_kw)], st), st)[0]


def _truth(kind, size, x, n=6000, seed=7, shape='SQUARE'):
    """Monte-Carlo ground truth of the lamp's visible fraction along the
    floor row y = 0, from the lamp's own geometry (a disc of the radius
    facing the point, the sun's angular disc, the rectangle or ellipse)
    against the plate's straight edge."""
    rng = np.random.default_rng(seed)
    lp = np.array(LPOS)
    u1, u2 = rng.random(n), rng.random(n)
    out = np.zeros(x.size)
    for i, xv in enumerate(x):
        P = np.array([xv, 0.0, 0.0])
        if kind == 'SUN':
            d = np.array([EDGE_X, 0.0, 0.0]) - lp
            L = -d / np.linalg.norm(d)
            t = np.cross([0, 1.0, 0], L)
            t /= np.linalg.norm(t)
            b = np.cross(L, t)
            r = np.sqrt(u1) * np.tan(size * 0.5)
            dl = L[None] + t[None] * (r * np.cos(2 * np.pi * u2))[:, None] \
                + b[None] * (r * np.sin(2 * np.pi * u2))[:, None]
        else:
            if kind == 'AREA':
                d = np.array([EDGE_X, 0.0, 0.0]) - lp
                d /= np.linalg.norm(d)
                ay = np.array([0.0, 1.0, 0.0])
                ax = np.cross(ay, d)
                ax /= np.linalg.norm(ax)
                if shape in ('DISK', 'ELLIPSE'):
                    rr = np.sqrt(u1)
                    uu = rr * np.cos(2 * np.pi * u2)
                    vv = rr * np.sin(2 * np.pi * u2)
                else:
                    uu, vv = u1 * 2 - 1, u2 * 2 - 1
                Sp = lp[None] + ax[None] * (uu * size * 0.5)[:, None] \
                    + ay[None] * (vv * size * 0.5)[:, None]
            else:
                L = lp - P
                L /= np.linalg.norm(L)
                t = np.cross([0, 1.0, 0], L)
                t /= np.linalg.norm(t)
                b = np.cross(L, t)
                r = np.sqrt(u1) * size
                Sp = lp[None] + t[None] * (r * np.cos(2 * np.pi * u2))[:, None] \
                    + b[None] * (r * np.sin(2 * np.pi * u2))[:, None]
            dl = Sp - P[None]
        out[i] = float((P[0] + dl[:, 0] * (PLATE_Z / dl[:, 2]) > 0.0).mean())
    return out


def _grain(img, base, tv):
    """(RMS of vis against the truth inside the penumbra, distinct grey
    levels there, share of its pixels more than one level of four rays
    off the truth, worst column-mean bias)."""
    _x, _v, rows = _vis(img, base)
    pen = (tv > 0.02) & (tv < 0.98)
    err = rows[:, pen] - tv[pen][None, :]
    return (float(np.sqrt((err ** 2).mean())),
            len(np.unique(np.round(rows[:, pen], 4))),
            float((np.abs(err) > 0.25).mean()),
            float(np.abs(rows[:, pen].mean(axis=0) - tv[pen]).max()))


class _Rays:
    """A stand-in BVH that occludes nothing and keeps every ray it is
    asked about: the sampler's own directions, without a scene."""

    def __init__(self):
        self.origins, self.dirs, self.maxt = [], [], []

    def occluded(self, origin, dirs, maxt, mask=None, cast=None):
        self.origins.append(np.asarray(origin, np.float64).copy())
        self.dirs.append(np.asarray(dirs, np.float64).copy())
        self.maxt.append(np.asarray(maxt, np.float64).copy())
        return np.zeros(origin.shape[0], bool)


def _rays_of(light, st, n=1024, **st_private):
    """(L, dist, P, the captured stand-in) for `n` floor points whose
    pixel identities tile a 32-wide grid."""
    P = np.zeros((n, 3), f32)
    P[:, 0] = 1.0
    N = np.tile(np.array([[0, 0, 1.0]], f32), (n, 1))
    L, _rad, dist = LI.sample(light, P, st)
    for k, v in st_private.items():
        setattr(st, k, v)
    cap = _Rays()
    spx = np.arange(n, dtype=np.int64) % 32
    spy = np.arange(n, dtype=np.int64) // 32
    vis = LI.visibility(light, P, N, L, dist, st, bvh=cap,
                        sample_xy=(spx, spy, 0))
    return L, dist, P, cap, vis, (spx, spy)


def _sources(passes, atlases=None):
    """Every fragment source a plan would hand the driver."""
    out = []

    def walk(o, depth=0):
        if isinstance(o, str):
            if 'void main' in o:
                out.append(o)
        elif isinstance(o, dict) and depth < 6:
            for v in o.values():
                walk(v, depth + 1)
        elif isinstance(o, (list, tuple)) and depth < 6:
            for v in o:
                walk(v, depth + 1)
    walk(passes)
    walk({k: v for k, v in (atlases or {}).items() if k.startswith('__')})
    return out


def _shadow_fn(src, i=0):
    """The text of hal_shadow_vis{i} inside a pass source."""
    a = src.find(f'float hal_shadow_vis{i}(')
    if a < 0:
        return ''
    b = src.find('\n}\n', a)
    return src[a:b + 3]


def _ui():
    """(ui, properties) under the bpy stub the panel tests draw with."""
    from . import fakebpy
    stub = fakebpy.install()
    stub.types.UIList = type('UIList', (stub.types.Panel,), {})
    stub.types.AddonPreferences = type('AddonPreferences',
                                       (stub.types.Panel,), {})
    from .. import properties as PR
    from .. import ui as UI
    return UI, PR


# ---------------------------------------------- 0. the identity of the rest


def test_0_identity_of_everything_untouched():
    """What this round did not mean to move renders bitwise the previous
    tree, each tree shading scenes built from its own classes: HARD
    traced shadows of every lamp kind, occlusion, radiosity and blurred
    reflections at small and large seeds (their salts are wrapped now --
    the same hash), area lamps of ordinary size, shadows off. Mapped
    shadows are another package's this round and are not pinned here."""
    RP = _prev_engine(PREV_ZIP)
    if RP is None:
        note(f'{PREV_ZIP} is not beside the package -- the bitwise pin '
             'against the previous tree is skipped here (the in-tree '
             'neutrality proofs of each section still run)')
        return
    import importlib
    pkg = RP.__name__.rsplit('.', 2)[0]
    PFM = importlib.import_module(pkg + '.tests.featurematrix')
    PLI = importlib.import_module(pkg + '.core.lights')
    PK = _kit(pkg)

    def same(label, now, was):
        check(f'{label}: bitwise the previous tree',
              now.shape == was.shape and bool(np.array_equal(now, was)),
              '' if now.shape != was.shape else
              f'max {float(np.abs(now - was).max()):.3g}')

    demo_cases = (
        ('the demo scene, hard traced shadows', 'demo',
         dict(shadow_default='RAY')),
        ('the demo scene, shadows off', 'demo', dict(shadows=False)),
        ('traced shadows + occlusion, seed 0', 'demo',
         dict(shadow_default='RAY', ambient_occlusion=True)),
        ('traced shadows + occlusion, seed 600000', 'demo',
         dict(shadow_default='RAY', ambient_occlusion=True, seed=600000)),
        ('radiosity, seed 7', 'demo',
         dict(shadow_default='RAY', radiosity=True, seed=7)),
        ('full-rate radiosity, seed 2**30', 'demo',
         dict(shadow_default='RAY', radiosity=True, radiosity_spacing=1,
              seed=2 ** 30)),
        ('blurred traced reflections, seed 600000', 'mirror',
         dict(shadow_default='RAY', raytrace=True, ray_depth=1,
              reflection_blur=0.2, reflection_blur_samples=4, seed=600000)),
        ('the area-lamp scene, shadows off', 'area', dict(shadows=False)),
        ('the area-lamp scene, hard traced shadows', 'area',
         dict(shadow_default='RAY')))
    for label, key, kw in demo_cases:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')     # the old tree's int overflow
            st = base_settings(64, 48, **kw)
            LI.clear_shadow_cache()
            now = np.asarray(R.render(FM.SCENES[key](st), st))
            st2 = PK.base_settings(64, 48, **kw)
            PLI.clear_shadow_cache()
            was = np.asarray(RP.render(PFM.SCENES[key](st2), st2))
        same(label, now, was)
    for kind in KINDS:
        # a size-0 lamp traces hard; the AREA lamp keeps a real face
        # (0.25: its LIGHTING is pinned) and one ray, so the pattern is
        # not in play on either tree
        size = 0.25 if kind == 'AREA' else 0.0
        for kw in (dict(shadow_default='RAY', shadow_samples=1),
                   dict(shadow_default='PER_LIGHT', raytrace=True,
                        shadow_samples=1)):
            st = _settings(**kw)
            LI.clear_shadow_cache()
            now = np.asarray(R.render(_scene([_lamp(kind, size)], st), st))
            st2 = _settings(kit=PK, **kw)
            PLI.clear_shadow_cache()
            was = np.asarray(RP.render(
                _scene([_lamp(kind, size, kit=PK)], st2, kit=PK), st2))
            same(f"the rig, {kind} lamp, {kw['shadow_default']}", now, was)


# --------------------------------- A. Per Light + a lamp on Ray Traced casts


def test_a_per_light_ray_lamp_casts():
    """Shadow Method = Per Light with the lamp's own menu on Ray Traced:
    the lamp casts (it cast nothing unless Ray Tracing or AO happened to
    be on), the frame IS the render-wide Ray Traced frame, the GLSL
    follows, and no BVH is built when nothing will trace."""
    for kind in KINDS:
        # hard shadows (one ray; the area lamp keeps an ordinary face):
        # this section is about WHETHER the lamp casts
        size = 0.25 if kind == 'AREA' else 0.0
        base = _unshadowed(kind, size)
        st = _settings(shadow_default='PER_LIGHT', shadow_samples=1)
        img, bvh, _n = _cpu(_scene([_lamp(kind, size, 'RAY')], st), st)
        x, v, _r = _vis(img, base)
        dark = 1.0 - float(v[x < 0.6].mean())
        check(f'{kind}: Per Light + lamp Ray Traced casts a shadow with '
              'Ray Tracing and AO off', bvh is not None and dark > 0.95,
              f'umbra darkness {dark:.3f}, BVH built {bvh is not None}')
        st_r = _settings(shadow_default='RAY', shadow_samples=1)
        ref, _b, _n = _cpu(_scene([_lamp(kind, size, 'RAY')], st_r), st_r)
        check(f'{kind}: ...and it is the render-wide Ray Traced frame '
              'bitwise', bool(np.array_equal(img, ref)))
        _i, d, why, _p, _b = _pair(
            lambda: [_lamp(kind, size, 'RAY')],
            dict(shadow_default='PER_LIGHT', shadow_samples=1))
        _bar(f'{kind}: Per Light + lamp Ray Traced', d, why)

    # one mapped and one traced lamp under Per Light: both cast
    def two():
        return [_lamp('SPOT', 0.0, 'MAP'),
                _lamp('SPOT', 0.0, 'RAY', pos=(-2.0, 3.0, 6.0))]
    lit = _settings(shadows=False)
    base = _cpu(_scene(two(), lit), lit)[0]
    st = _settings(shadow_default='PER_LIGHT')
    img, bvh, _n = _cpu(_scene(two(), st), st)
    x, v, _r = _vis(img, base)
    check('a mapped and a traced lamp under Per Light both cast',
          bvh is not None and float(v[x < 0.5].mean()) < 0.05,
          f'umbra vis {float(v[x < 0.5].mean()):.3f}')

    # no BVH where nothing traces, and the frame does not move
    cases = (
        ('render Shadow Maps (the default)', dict(), 'MAP'),
        ('Per Light with only mapped lamps',
         dict(shadow_default='PER_LIGHT'), 'MAP'),
        ('Per Light + lamp Ray Traced, Ray Shadows off',
         dict(shadow_default='PER_LIGHT', ray_shadows=False), 'RAY'),
        ('Per Light + lamp Ray Traced, Shadows off',
         dict(shadow_default='PER_LIGHT', shadows=False), 'RAY'),
        ('Per Light + lamp None', dict(shadow_default='PER_LIGHT'), 'NONE'),
        ('render Ray Traced, Ray Shadows off',
         dict(shadow_default='RAY', ray_shadows=False), 'MAP'))
    for label, kw, lsh in cases:
        st = _settings(**kw)
        _img, _bvh, n = _cpu(_scene([_lamp('POINT', 0.0, lsh)], st), st)
        check(f'{label}: no BVH is built', n == 0, f'{n} built')
    check('a hemi lamp on Ray Traced under Per Light builds no BVH',
          not R._per_light_ray_shadows(
              types.SimpleNamespace(lights=[Light(type='HEMI', shadow='RAY')]),
              _settings(shadow_default='PER_LIGHT')))
    # Ray Shadows off under render Ray Traced: lit, with or without a BVH
    st = _settings(shadow_default='RAY', ray_shadows=False)
    a = _cpu(_scene([_lamp('POINT', 0.0)], st), st)[0]
    st = _settings(shadow_default='RAY', ray_shadows=False, raytrace=True)
    b, bvh, _n = _cpu(_scene([_lamp('POINT', 0.0)], st), st)
    check('render Ray Traced with Ray Shadows off is the same frame with '
          'and without a BVH in reach (the master returns first)',
          bvh is not None and bool(np.array_equal(a, b)))


# ------------------------------------ B. the lamp's own Samples: 0 inherits


def test_b_lamp_samples_override():
    """A lamp's own Samples sizes ITS taps or rays; 0 reads the render's.
    Nothing read it before: the dial moved no pixel on either road."""
    # traced: lamp 16 under render 4 IS render 16, bitwise
    def frame(lamp_n, scene_n, method='RAY', kind='POINT', **lkw):
        s = _settings(shadow_default=method, shadow_samples=scene_n,
                      shadow_softness=4.0)
        return _cpu(_scene([_lamp(kind, SIZES[kind], 'MAP',
                                  shadow_samples=lamp_n, **lkw)], s), s)[0]
    for method in ('RAY', 'MAP'):
        a = frame(0, 4, method)
        b = frame(16, 4, method)
        c = frame(0, 16, method)
        d = frame(4, 4, method)
        tag = 'traced' if method == 'RAY' else 'mapped'
        check(f'{tag}: the lamp\'s Samples moves the frame',
              float(np.abs(a - b).max()) > 0.02,
              f'max {float(np.abs(a - b).max()):.4f}')
        check(f'{tag}: lamp 16 under render 4 is render 16 bitwise',
              bool(np.array_equal(b, c)))
        check(f'{tag}: a lamp at the render\'s own count is the inherited '
              'frame bitwise', bool(np.array_equal(a, d)))
        _i, dd, why, passes, _b = _pair(
            lambda: [_lamp('POINT', 0.5, 'MAP', shadow_samples=16)],
            dict(shadow_default=method, shadow_samples=4,
                 shadow_softness=4.0))
        _bar(f'{tag}: lamp Samples 16 under render 4', dd, why)

    st = RenderSettings()
    st.shadow_samples = 4
    check('shadow_sample_count: 0 inherits, a set value wins, never under 1',
          LI.shadow_sample_count(Light(), st) == 4
          and LI.shadow_sample_count(Light(shadow_samples=16), st) == 16
          and LI.shadow_sample_count(Light(shadow_samples=1), st) == 1
          and LI.shadow_sample_count(
              Light(), types.SimpleNamespace(shadow_samples=0)) == 1)
    check('the Light dataclass default is 0 = inherit',
          Light().shadow_samples == 0)

    # the GPU meta and the light signature carry the resolved count
    stg = _settings(shadow_default='RAY', shadow_samples=4)
    meta, _a, _w = GSH._shadow_meta(_lamp('POINT', 0.5, shadow_samples=16),
                                    stg, bvh=object())
    meta0, _a, _w = GSH._shadow_meta(_lamp('POINT', 0.5), stg, bvh=object())
    check('the ray meta unrolls the lamp\'s own count (16) and the '
          'inherited one (4)', meta['samples'] == 16 and meta0['samples'] == 4)
    check('the lamp\'s Samples is in the light signature (the unroll is '
          'baked)', GSH._light_sig(_lamp('POINT', 0.5, shadow_samples=16))
          != GSH._light_sig(_lamp('POINT', 0.5)))

    # the panel property and the export
    UI, PR = _ui()
    ann = PR.HalcyonLightSettings.__annotations__['shadow_samples']
    check('the panel\'s Samples defaults to 0 with 0 allowed',
          ann.kw.get('default') == 0 and ann.kw.get('min') == 0)
    from .test_r253_light_options import _export_fake
    check('export hands the lamp\'s Samples to the renderer',
          _export_fake('POINT', shadow_samples=12).shadow_samples == 12)

    # the adopt operator clears every override it sits under
    def fake_light(**kw):
        return types.SimpleNamespace(type='LIGHT', data=types.SimpleNamespace(
            halcyon=types.SimpleNamespace(**kw)))
    lights = [fake_light(shadow_map_size=0, shadow_bias=0.0,
                         shadow_samples=32, shadow_map_depth='INHERIT'),
              fake_light(shadow_map_size=0, shadow_bias=0.0,
                         shadow_samples=0, shadow_map_depth='MIDPOINT'),
              fake_light(shadow_map_size=0, shadow_bias=0.0,
                         shadow_samples=0, shadow_map_depth='INHERIT'),
              fake_light(shadow_map_size=512, shadow_bias=0.02,
                         shadow_samples=8, shadow_map_depth='CLASSIC')]
    op = UI.HALCYON_OT_adopt_shadow_settings.__new__(
        UI.HALCYON_OT_adopt_shadow_settings)
    op.scope = 'SCENE'
    said = []
    op.report = lambda kind, msg: said.append(msg)
    res = op.execute(types.SimpleNamespace(
        scene=types.SimpleNamespace(objects=lights), light=None))
    check('All Lights Use These Settings resets Samples to 0 and Map '
          'Depth to Inherit beside Map Size and Bias',
          res == {'FINISHED'} and all(
              l.data.halcyon.shadow_samples == 0
              and l.data.halcyon.shadow_map_depth == 'INHERIT'
              and l.data.halcyon.shadow_map_size == 0
              and l.data.halcyon.shadow_bias == 0.0 for l in lights))
    check('...and counts the three lamps that carried an override',
          bool(said) and said[0].startswith('3 light'), str(said[:1]))


# ----------------------------------------- C. one stratified pattern a pixel


def test_c_stratified_soft_rays():
    """The N rays of a pixel are one stratified pattern on two hashes:
    less grain at the same count, the same picture for the same seed,
    the same rays on both roads -- and a hard lamp never reaches it."""
    # grain against the lamp's own geometry, per kind, at the default 4
    for kind, shape in (('POINT', None), ('SPOT', None), ('SUN', None),
                        ('AREA', 'SQUARE'), ('AREA', 'DISK')):
        size = SIZES[kind]
        lkw = {'area_shape': shape} if shape else {}
        tag = kind + (f' {shape}' if shape else '')
        base = _unshadowed(kind, size, **lkw)
        x = _floor_x()
        x = x[x > 0.0]
        tv = _truth(kind, size, x, shape=shape or 'SQUARE')
        img, d, why, passes, _b = _pair(
            lambda: [_lamp(kind, size, **lkw)],
            dict(shadow_default='RAY', shadow_samples=4))
        rms, levels, far, bias = _grain(img, base, tv)
        check(f'{tag}: 4 rays leave a grain under 0.15 (independent draws '
              'gave 0.20)', rms < 0.15,
              f'RMS {rms:.3f}, {levels} levels, {far * 100:.1f}% of the '
              f'penumbra over one level off, bias {bias:.3f}')
        check(f'{tag}: the penumbra is unbiased', bias < 0.11,
              f'worst column mean {bias:.3f} off the truth')
        _bar(f'{tag}: 4 stratified rays', d, why)
        fn = _shadow_fn(_sources(passes)[0]) if passes else ''
        check(f'{tag}: the GLSL draws two hashes for the pixel, not two '
              'per ray', fn.count('hal_smp_hash3(') == 2,
              f"{fn.count('hal_smp_hash3(')} hash calls")

    # more rays, less grain; a non-power-of-two count travels too
    base = _unshadowed('POINT', 0.5)
    x = _floor_x()
    x = x[x > 0.0]
    tv = _truth('POINT', 0.5, x)
    got = {}
    for n_rays in (2, 4, 5, 8, 16):
        img, d, why, _p, _b = _pair(
            lambda: [_lamp('POINT', 0.5)],
            dict(shadow_default='RAY', shadow_samples=n_rays))
        got[n_rays] = _grain(img, base, tv)[0]
        _bar(f'POINT, {n_rays} rays', d, why)
    check('grain falls with the count, faster than one over the root '
          '(8 rays under 0.08, 16 under 0.055)',
          got[2] > got[4] > got[8] > got[16] and got[8] < 0.08
          and got[16] < 0.055,
          ', '.join(f'{k}: {v:.3f}' for k, v in got.items()))

    # deterministic: same seed, same picture; threads do not matter
    def soft(**kw):
        s = _settings(shadow_default='RAY', **kw)
        return _cpu(_scene([_lamp('POINT', 0.5)], s), s)[0]
    a = soft()
    check('the same seed gives the same picture', bool(np.array_equal(a, soft())))
    check('another seed moves the grain', float(np.abs(a - soft(seed=7)).max()) > 0.05)
    check('the picture does not depend on the thread count',
          bool(np.array_equal(soft(threads=1), soft(threads=3))))

    # the pattern itself
    check('a power-of-two count takes the base-2 radical inverse',
          LI.soft_ray_turns(4) == [0.0, 0.5, 0.25, 0.75]
          and LI.soft_ray_turns(8)[:4] == [0.0, 0.5, 0.25, 0.75]
          and LI.soft_ray_turns(1) == [0.0])
    g5 = LI.soft_ray_turns(5)
    check('any other count takes golden fractions',
          all(abs(g5[k] - (k * 0.61803398875) % 1.0) < 1e-12
              for k in range(5)))
    check('a turn is a step through the 256-entry circle table',
          [LI.soft_ray_turn_index(t) for t in LI.soft_ray_turns(4)]
          == [0, 128, 64, 192] and LI.soft_ray_turn_index(0.9999) == 0)

    # the rays a POINT lamp fires: radius^2 / R^2 is the area coordinate
    for n_rays in (4, 5, 16):
        st = _settings(shadow_default='RAY', shadow_samples=n_rays)
        lt = _lamp('POINT', 0.5)
        L, dist, P, cap, _v, _id = _rays_of(lt, st)
        lp = np.asarray(lt.position, np.float64)
        strata = []
        for d in cap.dirs:
            # where the ray meets the plane through the lamp facing P
            Lf = L.astype(np.float64)
            tt = dist.astype(np.float64) / (d * Lf).sum(axis=1)
            hit = P.astype(np.float64) + d * tt[:, None]
            u1 = ((hit - lp[None]) ** 2).sum(axis=1) / 0.25
            strata.append(np.floor(np.clip(u1, 0, 0.999999) * n_rays))
        strata = np.sort(np.stack(strata, axis=1), axis=1)
        whole = (strata == np.arange(n_rays)[None, :]).all(axis=1)
        check(f'{n_rays} rays: every pixel puts exactly one ray in each of '
              f'its {n_rays} equal-area rings', float(whole.mean()) > 0.995,
              f'{float(whole.mean()) * 100:.1f}% of pixels')

    # a hard lamp never reaches the sampler: no seed, no pass index, and
    # the GLSL is the old function to the character
    for kind in KINDS:
        def hard(**kw):
            priv = kw.pop('_sample_pass', None)
            s = _settings(shadow_default='RAY', **kw)
            if priv is not None:
                s._sample_pass = priv
            return _cpu(_scene([_lamp(kind, 0.0)], s), s)[0]
        h0 = hard()
        check(f'{kind}: a size-0 lamp\'s traced frame ignores the seed and '
              'the pass index (bitwise)',
              bool(np.array_equal(h0, hard(seed=99)))
              and bool(np.array_equal(h0, hard(_sample_pass=3))))
    stg = _settings(shadow_default='RAY')
    meta, _a, _w = GSH._shadow_meta(_lamp('POINT', 0.0), stg, bvh=object())
    text = GM._shadow_function(0, meta, {'resolution': (W, H), 'seed': 5})
    check('the hard traced function is emitted to the character as before',
          text == ('float hal_shadow_vis0(vec3 P, vec3 N, vec3 L, float dist)\n'
                   '{\n'
                   '    vec3 org = P + N * 0.001 + L * 0.001;\n'
                   '    float maxt = (dist > 1e8) ? 1e9 : dist * (1.0 - 1e-3);\n'
                   '    return 1.0 - hal_bvh_occluded(org, L, maxt);\n'
                   '}\n'), repr(text[:200]))
    _i, _d, _w, passes, _b = _pair(lambda: [_lamp('POINT', 0.0)],
                                   dict(shadow_default='RAY'))
    src = _sources(passes)[0]
    check('a hard traced pass declares no pass scalar and no sampling '
          'primitives', 'hal_spass' not in src and 'hal_smp_hash3' not in src)


# ------------------------------------------------ D. Density, traced as well


def test_d_density_on_traced_shadows():
    """Density lightens a traced shadow exactly as it lightens a mapped
    one; at the default the visibility is returned untouched."""
    for kind, size in (('POINT', 0.0), ('POINT', 0.5), ('SPOT', 0.5),
                       ('SUN', 0.0), ('SUN', 0.2), ('AREA', 1.0)):
        base = _unshadowed(kind, size)
        full, d1, why1, _p, _b = _pair(lambda: [_lamp(kind, size)],
                                       dict(shadow_default='RAY'))
        part, d2, why2, _p, _b = _pair(
            lambda: [_lamp(kind, size, shadow_density=0.4)],
            dict(shadow_default='RAY'))
        x, v0, _r = _vis(full, base)
        _x, v1, _r = _vis(part, base)
        u0, u1 = float(v0[x < 0.35].mean()), float(v1[x < 0.35].mean())
        check(f'{kind} size {size}: Density 0.4 lifts the traced umbra '
              'from 0 to 0.6', abs(u0) < 1e-3 and abs(u1 - 0.6) < 2e-3,
              f'{u0:.4f} -> {u1:.4f}')
        _bar(f'{kind} size {size}: Density 0.4', d2, why2)
    # a mapped and a traced lamp agree on what Density means
    st = _settings(shadow_default='RAY')
    r = _cpu(_scene([_lamp('SUN', 0.0, shadow_density=0.25)], st), st)[0]
    st = _settings(shadow_default='MAP', shadow_map_size=2048)
    m = _cpu(_scene([_lamp('SUN', 0.0, shadow_density=0.25)], st), st)[0]
    base = _unshadowed('SUN', 0.0)
    x, vr, _r = _vis(r, base)
    _x, vm, _r = _vis(m, base)
    check('the same Density gives the same umbra traced and mapped',
          abs(float(vr[x < 0.5].mean()) - float(vm[x < 0.5].mean())) < 2e-3,
          f'traced {float(vr[x < 0.5].mean()):.4f} mapped '
          f'{float(vm[x < 0.5].mean()):.4f}')
    st = _settings(shadow_default='RAY')
    z = _cpu(_scene([_lamp('POINT', 0.5, shadow_density=0.0)], st), st)[0]
    check('Density 0 is no shadow at all',
          float(np.abs(z - _unshadowed('POINT', 0.5)).max()) < 1e-6)

    # neutrality: Density 1.0 returns the bare visibility, on both roads
    class Half:
        def occluded(self, origin, dirs, maxt, mask=None, cast=None):
            return (np.arange(origin.shape[0]) % 2).astype(bool)
    n = 64
    P = np.zeros((n, 3), f32)
    N = np.tile(np.array([[0, 0, 1.0]], f32), (n, 1))
    st = _settings(shadow_default='RAY')
    for size in (0.0, 0.5):
        lt = _lamp('POINT', size)
        L, _rad, dist = LI.sample(lt, P, st)
        ids = (np.arange(n, dtype=np.int64), np.zeros(n, np.int64), 0)
        v = LI.visibility(lt, P, N, L, dist, st, bvh=Half(), sample_xy=ids)
        want = (1 - (np.arange(n) % 2)).astype(f32)
        check(f'size {size}: at Density 1.0 the visibility is exactly lit '
              '1.0 / shadowed 0.0 (no 1 - (1 - v) round trip)',
              v.dtype == np.float32 and bool(np.array_equal(v, want)))
        lt.shadow_density = 0.4
        v4 = LI.visibility(lt, P, N, L, dist, st, bvh=Half(), sample_xy=ids)
        check(f'size {size}: at Density 0.4 it is the map tail\'s own '
              'expression', bool(np.array_equal(
                  v4, (1.0 - (1.0 - want) * 0.4).astype(f32))))
    stg = _settings(shadow_default='RAY')
    for size in (0.0, 0.5):
        m1, _a, _w = GSH._shadow_meta(_lamp('POINT', size), stg, bvh=object())
        m4, _a, _w = GSH._shadow_meta(_lamp('POINT', size, shadow_density=0.4),
                                      stg, bvh=object())
        consts = {'resolution': (W, H), 'seed': 0}
        t1 = GM._shadow_function(0, m1, consts)
        t4 = GM._shadow_function(0, m4, consts)
        check(f'size {size}: the GLSL carries the density term only when '
              'it is not 1.0', m4['density'] == 0.4
              and '1.0 - (1.0 - (' not in t1 and '1.0 - (1.0 - (' in t4
              and t1 != t4)
    check('a traced lamp\'s Density is in the light signature',
          GSH._light_sig(_lamp('POINT', 0.5, shadow_density=0.4))
          != GSH._light_sig(_lamp('POINT', 0.5)))


# ------------------------------------- E. an Area lamp's object scale


def _exported_area(size, scale=(1.0, 1.0, 1.0), shape='SQUARE', size_y=None):
    """export_light on a fake Area lamp object whose world matrix carries
    `scale`, standing where the rig's lamp stands."""
    from . import fakebpy
    fakebpy.install()
    from .. import export as EX
    ref = _lamp('AREA', size)
    mw = np.eye(4, dtype=f32)
    mw[:3, 0] = np.asarray(ref.area_x, np.float64) * scale[0]
    mw[:3, 1] = np.asarray(ref.area_y, np.float64) * scale[1]
    mw[:3, 2] = -np.asarray(ref.direction, np.float64) * scale[2]
    mw[:3, 3] = LPOS
    data = types.SimpleNamespace(
        type='AREA', color=(1.0, 1.0, 1.0), energy=1.0, halcyon=None,
        shape=shape, size=size, size_y=size if size_y is None else size_y,
        shadow_soft_size=0.0, use_shadow=True)
    lt = EX.export_light(types.SimpleNamespace(name='L', data=data), mw)
    lt.shadow = 'RAY'
    return lt, mw


def test_e_area_lamp_object_scale():
    """S in the viewport makes an Area lamp bigger, as Blender does it:
    Size 0.25 scaled x4 IS the Size 1.0 lamp. It used to light the floor
    sixteen times brighter and keep the 0.25 lamp's penumbra."""
    lt, mw = _exported_area(1.0)
    check('an unscaled lamp exports its Size and the raw matrix columns '
          '(the old lines, untouched)',
          tuple(lt.area_size) == (1.0, 1.0)
          and bool(np.array_equal(np.asarray(lt.area_x, f32), mw[:3, 0]))
          and bool(np.array_equal(np.asarray(lt.area_y, f32), mw[:3, 1])))
    ls, mws = _exported_area(0.25, (4.0, 4.0, 4.0))
    check('a lamp scaled x4 exports Size x 4 on unit axes',
          np.allclose(ls.area_size, (1.0, 1.0), atol=1e-6)
          and abs(float(np.linalg.norm(ls.area_x)) - 1.0) < 1e-6
          and abs(float(np.linalg.norm(ls.area_y)) - 1.0) < 1e-6)
    lr, _m = _exported_area(0.5, (4.0, 1.0, 1.0))
    check('a non-uniform scale makes a rectangle',
          np.allclose(lr.area_size, (2.0, 0.5), atol=1e-6))
    le, _m = _exported_area(0.5, (2.0, 2.0, 2.0), shape='ELLIPSE', size_y=0.25)
    check('an ellipse scales both of its sizes',
          np.allclose(le.area_size, (1.0, 0.5), atol=1e-6))

    def measure(make):
        lit_st = _settings(shadows=False)
        lit = _cpu(_scene([make()], lit_st), lit_st)[0]
        img, d, why, _p, _b = _pair(
            lambda: [make()], dict(shadow_default='RAY', shadow_samples=16))
        x, v, _r = _vis(img, lit)
        lo, hi = float(v.min()), float(v.max())
        t = (v - lo) / max(hi - lo, 1e-9)
        w = float(x[np.argmax(t >= 0.9)] - x[np.argmax(t >= 0.1)])
        return float(lit[ROWS][:, _floor_x() > 1.8, :3].mean()), w, img, d, why
    one = measure(lambda: _exported_area(1.0)[0])
    scaled = measure(lambda: _exported_area(0.25, (4.0, 4.0, 4.0))[0])
    small = measure(lambda: _exported_area(0.25)[0])
    check('Size 0.25 scaled x4 lights the floor as Size 1.0 does (it was '
          'x16)', abs(scaled[0] / one[0] - 1.0) < 1e-3,
          f'x{scaled[0] / one[0]:.4f}')
    check('...and throws Size 1.0\'s penumbra, four times the 0.25 lamp\'s',
          abs(scaled[1] / one[1] - 1.0) < 0.05 and one[1] / small[1] > 3.0,
          f'scaled {scaled[1]:.3f}, Size 1.0 {one[1]:.3f}, Size 0.25 '
          f'{small[1]:.3f}')
    check('...the two frames are the same frame',
          float(np.abs(scaled[2] - one[2]).max()) < 1e-6,
          f'max {float(np.abs(scaled[2] - one[2]).max()):.2e}')
    _bar('the scaled Area lamp, traced', scaled[3], scaled[4])
    rect = measure(lambda: _exported_area(0.5, (4.0, 1.0, 1.0))[0])
    same = measure(lambda: _exported_area(2.0, shape='RECTANGLE',
                                          size_y=0.5)[0])
    check('a square scaled (4, 1, 1) is the 2.0 x 0.5 rectangle',
          float(np.abs(rect[2] - same[2]).max()) < 1e-6)


# ------------------------------------------ F. an Area lamp of Size 0


def test_f_area_lamp_of_size_zero():
    """A Size 0 (or tiny) Area lamp lights as the clean AREA_MIN_SIDE
    lamp on both roads. It lit the floor with noise, 76 times too
    bright, and the GLSL twin was off by 64."""
    ref_st = _settings(shadows=False)
    ref = _cpu(_scene([_lamp('AREA', 0.25)], ref_st), ref_st)[0]
    floor = _floor_x() > 1.8
    ref_lit = float(ref[ROWS][:, floor, :3].mean())

    def rough(img):
        a = img[ROWS][:, floor, :3].mean(axis=2)
        return float(np.abs(np.diff(a, axis=1)).mean() / max(a.mean(), 1e-9))
    ref_rough = rough(ref)
    cases = (('SQUARE of Size 0', dict(size=0.0)),
             ('SQUARE of Size 1e-5', dict(size=1e-5)),
             ('SQUARE of Size 0.001', dict(size=0.001)),
             ('DISK of Size 0', dict(size=0.0, area_shape='DISK')),
             ('RECTANGLE 1 x 0', dict(size=1.0, sy=0.0)))
    for label, kw in cases:
        def mk():
            lt = _lamp('AREA', kw['size'])
            if 'sy' in kw:
                lt.area_size = (kw['size'], kw['sy'])
                lt.area_shape = 'RECTANGLE'
            if 'area_shape' in kw:
                lt.area_shape = kw['area_shape']
            return lt
        st = _settings(shadows=False)
        img, _b, _n = _cpu(_scene([mk()], st), st)
        lit = float(img[ROWS][:, floor, :3].mean())
        check(f'{label}: the floor is lit like the 0.25 lamp (far-field '
              'energy does not depend on the size)',
              bool(np.isfinite(img).all()) and abs(lit / ref_lit - 1.0) < 0.01,
              f'{lit:.4f} against {ref_lit:.4f}')
        check(f'{label}: no noise (neighbouring pixels as smooth as the '
              '0.25 lamp\'s)', rough(img) < ref_rough * 1.5 + 1e-4,
              f'{rough(img):.5f} against {ref_rough:.5f}')
        st2 = _settings(shadows=False)
        sim, cov, why, _p = _sim(_scene([mk()], st2), st2, False)
        if sim is None:
            check(f'{label}: the GLSL twin plans', False, str(why))
            continue
        rel = np.abs(sim[cov] - img[cov][:, :3]).max(axis=1) \
            / np.maximum(img[cov][:, :3].max(axis=1), 1e-6)
        check(f'{label}: the GLSL twin is within 1% of the CPU everywhere',
              float(rel.max()) < 0.01, f'max relative {float(rel.max()):.4f}')

    check('one shared minimum side', LI.AREA_MIN_SIDE == 0.02)
    # lamps at or above the minimum are untouched: the face is the Size's
    for size in (0.02, 0.25, 3.0):
        lt = _lamp('AREA', size)
        c = LI.area_corners(lt)
        ax, ay = np.asarray(lt.area_x, np.float64), np.asarray(lt.area_y, np.float64)
        pos = np.asarray(lt.position, np.float64)
        hx, hy = ax * np.float64(size * 0.5), ay * np.float64(size * 0.5)
        want = np.stack((pos - hx - hy, pos - hx + hy, pos + hx + hy,
                         pos + hx - hy))
        check(f'Size {size}: the lit face is the Size\'s own rectangle '
              'bitwise', bool(np.array_equal(c, want))
              and LI.area_emit_area(lt) == size * size)
    lt = _lamp('AREA', 0.0)
    txt = GM._area_function(0, lt, {'light_texels': True})
    half = float(np.linalg.norm(np.asarray(lt.area_x, np.float64) * 0.01))
    m = re.search(r'vec3 hax = vec3\(([^)]*)\);', txt)
    got = np.linalg.norm([float(v) for v in m.group(1).split(',')]) if m else -1
    check('the GLSL rectangle is built from the same clamped sides '
          '(half-axis 0.01 for Size 0)', abs(got - half) < 1e-7,
          f'{got:.7f}')
    # the traced shadow still reads the true Size: Size 0 stays hard
    st = _settings(shadow_default='RAY')
    img, _b, _n = _cpu(_scene([_lamp('AREA', 0.0)], st), st)
    x, _v, rows = _vis(img, _unshadowed('AREA', 0.0))
    check('a Size 0 lamp still casts a hard traced shadow (two levels)',
          len(np.unique(np.round(rows, 3))) <= 3,
          f'{len(np.unique(np.round(rows, 3)))} levels')


# --------------------------------------------------- G. wide suns


def test_g_wide_sun():
    """A sun wider than 0.5 rad is sampled uniformly over its cap and
    its angle stops at 179.9 degrees; an ordinary sun keeps the
    tangent-plane disc line for line."""
    st = _settings(shadow_default='RAY', shadow_samples=64)
    for ang in (0.6, 1.0, 2.0, 3.0, float(np.pi)):
        L, _dist, _P, cap, _v, _id = _rays_of(_lamp('SUN', ang), st)
        D = np.concatenate(cap.dirs)
        D /= np.linalg.norm(D, axis=1, keepdims=True)
        tilt = np.arccos(np.clip(D @ np.asarray(L[0], np.float64), -1, 1))
        half = min(ang, float(np.radians(179.9))) * 0.5
        inside = float((tilt < half * 0.5).mean())
        ideal = (1 - np.cos(half * 0.5)) / (1 - np.cos(half))
        check(f'angle {ang:.3f}: the rays are uniform over the cap (the '
              'share inside half the angular radius)',
              abs(inside - ideal) < 0.012 and float(tilt.max()) <= half + 1e-4,
              f'{inside:.3f} against {ideal:.3f}; widest tilt '
              f'{float(tilt.max()):.4f} of {half:.4f}')

    law, k = LI.sun_soft_law(0.2)
    check('up to 0.5 rad: the disc law with tan(angle / 2)',
          law == 'DISC' and k == float(f32(np.tan(0.1)))
          and LI.sun_soft_law(0.5)[0] == 'DISC')
    law, k = LI.sun_soft_law(2.0)
    check('above 0.5 rad: the cap law with 1 - cos(angle / 2)',
          law == 'CAP' and k == float(f32(1.0 - np.cos(1.0))))
    check('the angle is clamped to 179.9 degrees (EEVEE\'s ceiling)',
          LI.sun_soft_law(float(np.pi)) == LI.sun_soft_law(
              float(np.radians(179.9)))
          and LI.sun_soft_law(float(np.pi))[1] < 1.0)

    # an ordinary sun: the disc law on the stratified pattern, bitwise
    st = _settings(shadow_default='RAY', shadow_samples=4)
    lt = _lamp('SUN', 0.2)
    L, _dist, _P, cap, _v, (spx, spy) = _rays_of(lt, st)
    from ..core import mathx as M
    t, b = M.orthonormal_basis(L)
    z0 = LI.soft_ray_salt(0, 0, 0)
    h1 = PT.sample_u(spx, spy, z0)
    h2 = PT.sample_u(spx, spy, z0 + 1)
    h2i = np.asarray(h2 * f32(65535.0) + f32(0.5), f32).astype(np.int64)
    same = True
    for kk in range(4):
        u1 = (f32(kk) + h1) * f32(0.25)
        ai = (h2i + (0, 128, 64, 192)[kk]) & 255
        r = np.sqrt(u1) * f32(np.tan(0.1))
        jit = t * (r * PT.CIRCLE256[ai, 0])[:, None] \
            + b * (r * PT.CIRCLE256[ai, 1])[:, None]
        same = same and bool(np.array_equal(M.normalize(L + jit),
                                            cap.dirs[kk].astype(f32)))
    check('a 0.2 rad sun fires normalize(L + disc jitter) on the '
          'stratified draws, bitwise', same)
    for ang, tok, absent in ((0.2, 'normalize(L + jit)', 'float ct = '),
                             (2.0, 'float ct = 1.0 - u1 * ', 'normalize(L + jit)')):
        meta, _a, _w = GSH._shadow_meta(_lamp('SUN', ang), st, bvh=object())
        txt = GM._shadow_function(0, meta, {'resolution': (W, H), 'seed': 0})
        check(f'angle {ang}: the GLSL emits the same law',
              tok in txt and absent not in txt)
    for ang in (0.2, 2.0, float(np.pi)):
        img, d, why, _p, _b = _pair(lambda: [_lamp('SUN', ang)],
                                    dict(shadow_default='RAY'))
        check(f'angle {ang:.3f}: the frame is finite',
              bool(np.isfinite(img).all()))
        _bar(f'a sun of {ang:.3f} rad, traced', d, why)


# ------------------------------------------ H. seeds and integer literals


def test_h_large_seed_literals():
    """The sampling salts are wrapped to 31 bits: a Seed above 542357
    baked an integer literal no GLSL compiler accepts into the soft-ray,
    occlusion and radiosity functions. The hash reads only those bits,
    so no picture moves."""
    px, py = np.meshgrid(np.arange(48, dtype=np.int64),
                         np.arange(32, dtype=np.int64))
    px, py = px.ravel(), py.ravel()

    def ref_hash(z):
        """hash3 in exact integer arithmetic on the BARE salt."""
        out = np.zeros(px.size, f32)
        for i in range(px.size):
            h = (int(px[i]) * 374761393 + int(py[i]) * 668265263
                 + int(z) * 1274126177) & 0x7fffffff
            h = ((h ^ (h >> 13)) * 1274126177) & 0xffffffffffffffff
            out[i] = f32((h ^ (h >> 16)) & 0xffff) / f32(65535.0)
        return out
    ok = True
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for z in (0, 5, 2 ** 31 - 1, 2 ** 31, 7919 * 600000 + 131,
                  7919 * 2 ** 30 + 8389 + 7):
            try:
                ok = ok and bool(np.array_equal(PT.sample_u(px, py, z),
                                                ref_hash(z)))
            except RuntimeWarning:
                ok = False
    check('sample_u on a wrapped salt is the bare salt\'s hash bitwise, '
          'with no integer overflow', ok)
    check('wrap_salt keeps the low 31 bits',
          PT.wrap_salt(2 ** 31 + 5) == 5 and PT.wrap_salt(7) == 7
          and LI.soft_ray_salt(3, 2 ** 30) < 2 ** 31)

    lit_re = re.compile(r'(?<![\w.])(\d+)(?![\w.])')
    for seed in (5, 271188, 600000, 2 ** 30):
        kw = dict(shadow_default='RAY', seed=seed, ambient_occlusion=True,
                  ao_samples=4)
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            img, d, why, passes, _b = _pair(lambda: [_lamp('POINT', 0.5)], kw)
        srcs = _sources(passes) if passes else []
        big = max([int(m.group(1)) for s in srcs
                   for m in lit_re.finditer(s)] or [0])
        check(f'seed {seed}: soft rays + occlusion plan (both functions in '
              'the pass), and every integer literal fits int32',
              bool(srcs) and any('float hal_ao(' in s for s in srcs)
              and any('hal_shadow_vis0(' in s for s in srcs)
              and big <= 2147483647, f'largest {big}')
        _bar(f'seed {seed}: soft rays + occlusion', d, why)
        check(f'seed {seed}: no arithmetic warning on the CPU road',
              not [w for w in wl if 'overflow' in str(w.message)],
              '; '.join(str(w.message)[:60] for w in wl))
    kw = dict(shadow_default='RAY', seed=2 ** 30, radiosity=True,
              radiosity_samples=4, radiosity_spacing=1)
    _i, d, why, passes, _b = _pair(lambda: [_lamp('POINT', 0.0)], kw)
    srcs = _sources(passes) if passes else []
    big = max([int(m.group(1)) for s in srcs for m in lit_re.finditer(s)]
              or [0])
    check('seed 2**30: the full-rate radiosity gather is in the pass and '
          'its literals fit int32 too',
          bool(srcs) and any('hal_rad_at(' in s for s in srcs)
          and 2 ** 30 <= big <= 2147483647, f'largest {big} {why or ""}')
    _bar('seed 2**30: radiosity gather', d, why)

    glslang = os.environ.get('HALCYON_GLSLANG', '')
    if not (glslang and os.path.exists(glslang)):
        note('HALCYON_GLSLANG is not set -- the strict compile of the '
             'soft-ray passes is skipped here')
        return
    from . import glsl_strict
    bad = []
    n = 0
    for kind, lkw in (('POINT', {}), ('SPOT', {}), ('SUN', {}),
                      ('SUN', {'radius': 2.0}), ('AREA', {}),
                      ('AREA', {'area_shape': 'DISK'}),
                      ('POINT', {'shadow_density': 0.4})):
        for seed, n_rays in ((0, 4), (2 ** 30, 5)):
            _i, _d, _w, passes, _b = _pair(
                lambda: [_lamp(kind, SIZES[kind], **lkw)],
                dict(shadow_default='RAY', seed=seed, shadow_samples=n_rays,
                     ambient_occlusion=True, ao_samples=2))
            for src in _sources(passes) if passes else ['#error no plan']:
                okc, msg = glsl_strict.compile_source(src)
                n += 1
                if not okc:
                    bad.append(f'{kind} {lkw} seed {seed}: {msg[:200]}')
    check('every soft-ray pass compiles under strict GLSL 4.30', not bad,
          f'{n} sources' if not bad else bad[0])


# ---------------------------------- I. the averaging roads' pass index


def test_i_averaging_roads_trace_their_own_rays():
    """Accumulation passes, lens passes and adaptive refines each trace
    their own soft-shadow pattern: sixteen passes left the penumbra as
    grainy as one. The index travels through every settings copy, and
    on the GPU road it rides one scalar so the passes share one plan."""
    st = RenderSettings()
    st._sample_pass = 9
    st._accum_jitter = (0.1, 0.2)
    st._stereo = 'L'
    c = st.copy()
    check('copy() carries the pass index and drops every other private '
          'flag', getattr(c, '_sample_pass', 0) == 9
          and not hasattr(c, '_accum_jitter') and not hasattr(c, '_stereo'))
    check('a plain settings object copies with no index at all',
          not hasattr(RenderSettings().copy(), '_sample_pass'))
    base = _unshadowed('POINT', 0.5)
    x = _floor_x()
    x = x[x > 0.0]
    tv = _truth('POINT', 0.5, x)

    def grain(**kw):
        s = _settings(shadow_default='RAY', **kw)
        return _grain(_cpu(_scene([_lamp('POINT', 0.5)], s), s)[0], base, tv)[0]
    one = grain()
    for label, kw, n in (
            ('the Accumulation Buffer x4',
             dict(aa_mode='ACCUMULATE', aa_samples=4), 4),
            ('Lens Passes x4 (focus on the floor)',
             dict(dof=True, dof_method='LENS_ACCUMULATE', dof_focus=CAM_H,
                  dof_lens_samples=4), 4),
            ('accumulation x2 with Lens Passes x2 inside',
             dict(aa_mode='ACCUMULATE', aa_samples=2, dof=True,
                  dof_method='LENS_ACCUMULATE', dof_focus=CAM_H,
                  dof_lens_samples=2), 4)):
        g = grain(**kw)
        check(f'{label}: {n} averaged frames carry about half the grain of '
              'one (they carried 0.97 of it)', g / one < 0.62,
              f'{g:.3f} against {one:.3f}: {g / one:.2f} (ideal '
              f'{1 / np.sqrt(n):.2f})')

    seen = {0}
    top = RenderSettings()
    for a in range(64):
        s1 = top.copy()
        s1._sample_pass = top.sample_pass_for(a)
        seen.add(s1._sample_pass)
        for b in (0, 1, 63):
            s2 = s1.copy()
            s2._sample_pass = s1.sample_pass_for(b)
            seen.add(s2._sample_pass)
    check('a pass, a pass inside a pass and the frame outside all get '
          'different indices that a float32 carries exactly',
          len(seen) == 1 + 64 + 64 * 3 and max(seen) < 2 ** 24
          and top.sample_pass_for(0) != 0)
    check('the salt takes the index',
          LI.soft_ray_salt(1, 2, 3) == (131 + 2 * 7919 + 3 * 104729)
          and LI.soft_ray_salt(0, 0, 0) == 0)

    # every pass of an accumulation sees its own index, the lens passes
    # inside too; the frame outside sees none
    seen = []
    real = LI.visibility

    def spy(light, P, N, L, dist, settings, *a, **k):
        seen.append(int(getattr(settings, '_sample_pass', 0) or 0))
        return real(light, P, N, L, dist, settings, *a, **k)
    LI.visibility = spy
    try:
        s = _settings(shadow_default='RAY', aa_mode='ACCUMULATE', aa_samples=3,
                      dof=True, dof_method='LENS_ACCUMULATE', dof_focus=CAM_H,
                      dof_lens_samples=2)
        _cpu(_scene([_lamp('POINT', 0.5)], s), s)
        nested = sorted(set(seen))
        del seen[:]
        s = _settings(shadow_default='RAY', aa_mode='ADAPTIVE', aa_samples=4)
        _cpu(_scene([_lamp('POINT', 0.5)], s), s)
        adaptive = sorted(set(seen))
        del seen[:]
        s = _settings(shadow_default='RAY')
        _cpu(_scene([_lamp('POINT', 0.5)], s), s)
        plain = sorted(set(seen))
    finally:
        LI.visibility = real
    check('3 accumulation passes x 2 lens passes shade with six different '
          'indices; a plain frame with index 0',
          len(nested) == 6 and 0 not in nested and plain == [0],
          f'{nested} / {plain}')
    check('the adaptive refines carry indices of their own beside the '
          'base frame\'s 0', adaptive[0] == 0 and len(adaptive) >= 2,
          str(adaptive))
    _ui()                                   # the bpy stub engine.py needs
    from .. import engine as EN
    src = inspect.getsource(EN)
    body = src[src.index('def _render_motion_accumulated'):]
    body = body[:body.index('\n    def ', 10)]
    check('the motion-blur slices set the index and put it back',
          'settings._sample_pass = settings.sample_pass_for(k)' in body
          and body.count('settings._sample_pass = pass0') == 2)

    # the GPU road: the driver's own uniform plumbing through the fake
    # device. One plan serves every pass, and the scalar is load-bearing
    from . import fakedevice
    kw = dict(shadow_default='RAY', aa_mode='ACCUMULATE', aa_samples=4)
    s = _settings(**kw)
    cpu = _cpu(_scene([_lamp('POINT', 0.5)], s), s)[0]

    def on_gpu():
        s2 = _settings(render_device='GPU', **kw)
        sc2 = _scene([_lamp('POINT', 0.5)], s2)
        LI.clear_shadow_cache()
        GSH._PLAN_CACHE.clear()
        with fakedevice.installed():
            img = np.asarray(R.render(sc2, s2))
        return img, bool(getattr(s2, '_frame_gpu_shaded', False)), \
            len(GSH._PLAN_CACHE)
    img, shaded, plans = on_gpu()
    d = float(np.abs(img - cpu).max())
    check('an accumulated frame shades on the GPU device and is the CPU '
          'frame within the deferred bar', shaded and d < 6e-3,
          f'gpu shaded {shaded}, max {d:.6f}')
    check('the four passes share ONE plan (the index is a scalar, not a '
          'baked literal)', plans == 1, f'{plans} plans cached')
    real_sp = GSH._sample_pass
    GSH._sample_pass = lambda job: 0.0
    try:
        img0, _s, _p = on_gpu()
    finally:
        GSH._sample_pass = real_sp
    check('...and with the scalar withheld the GPU frame leaves the CPU\'s '
          '(the plumbing is what makes them agree)',
          float(np.abs(img0 - cpu).max()) > 0.03,
          f'max {float(np.abs(img0 - cpu).max()):.4f}')
    _i, _d, _w, passes, _b = _pair(lambda: [_lamp('POINT', 0.5)],
                                   dict(shadow_default='RAY'))
    binds = [x for p in passes for x in p if isinstance(x, dict)]
    check('a soft traced pass declares the hal_spass scalar among its '
          'frame uniforms', any('hal_spass' in (b.get('frame_uniforms') or ())
                                for b in binds)
          and 'uniform float hal_spass;' in _sources(passes)[0])


# ------------------------------------------------------- J. the panels


class _Lay:
    """A recording layout that knows whether a row is drawn active: the
    `active` flags of the row and every layout above it."""

    def __init__(self, rec, parent=None):
        self.rec, self.parent = rec, parent
        self.active = True
        self.enabled = True
        self.alert = False
        self.use_property_split = False
        self.scale_y = 1.0

    def live(self):
        n, ok = self, True
        while n is not None:
            ok = ok and bool(n.active)
            n = n.parent
        return ok

    def prop(self, data, name, **kw):
        self.rec.append(('prop', name, self.live()))

    def column(self, **kw):
        return _Lay(self.rec, self)
    row = column
    box = column

    def separator(self, **kw):
        pass

    def label(self, **kw):
        self.rec.append(('label', str(kw.get('text', '')), self.live()))

    def template_ID(self, data, name, **kw):
        self.rec.append(('prop', name, self.live()))

    def operator(self, idn, **kw):
        self.rec.append(('op', idn, self.live()))
        return types.SimpleNamespace(scope=None)


def _defaults(cls):
    ns = types.SimpleNamespace()
    for k, a in cls.__annotations__.items():
        kw = getattr(a, 'kw', None) or getattr(a, 'keywords', None) or {}
        setattr(ns, k, kw.get('default'))
    return ns


def _draw(panel, scene_hs, light_hs=None, ltype='POINT', **light_kw):
    rec = []
    light = types.SimpleNamespace(type=ltype, halcyon=light_hs,
                                  color=(1, 1, 1), energy=10.0,
                                  shadow_soft_size=0.25, angle=0.01,
                                  size=0.25, **light_kw)
    ctx = types.SimpleNamespace(
        scene=types.SimpleNamespace(halcyon=scene_hs, camera=None),
        light=light, object=None, engine='HALCYON_RENDER')
    panel.draw(types.SimpleNamespace(layout=_Lay(rec)), ctx)
    return rec


def test_j_panels_follow_the_method_that_runs():
    """The lamp's shadow rows grey by the method that will run, the
    panel says when the render's method overrides the lamp's menu (with
    a one-click way to Per Light), and the traced dials are neither
    greyed by a master they do not depend on nor out of reach."""
    UI, PR = _ui()
    shs = _defaults(PR.HalcyonSettings)
    lhs = _defaults(PR.HalcyonLightSettings)
    MAP_ROWS = ('shadow_map_size', 'shadow_bias', 'shadow_map_depth',
                'shadow_clip_start', 'shadow_clip_end', 'shadow_softness')

    def lamp_rows(sdef, lsh, ltype='POINT', **more):
        shs.shadow_default, lhs.shadow = sdef, lsh
        for k, v in more.items():
            setattr(shs if hasattr(shs, k) and not k.startswith('l_') else lhs,
                    k[2:] if k.startswith('l_') else k, v)
        rec = _draw(UI.HALCYON_PT_light, shs, lhs, ltype)
        for k in more:
            base = _defaults(PR.HalcyonLightSettings if k.startswith('l_')
                             else PR.HalcyonSettings)
            name = k[2:] if k.startswith('l_') else k
            setattr(lhs if k.startswith('l_') else shs, name,
                    getattr(base, name))
        act = {n: a for kind, n, a in rec if kind == 'prop'}
        labels = [n for kind, n, a in rec if kind == 'label']
        ops = [n for kind, n, a in rec if kind == 'op']
        return act, labels, ops

    act, labels, ops = lamp_rows('MAP', 'RAY')
    check('render Shadow Maps + lamp menu Ray Traced: the map rows are '
          'drawn ACTIVE (the lamp IS mapped; Clip was greyed while live)',
          all(act[r] for r in MAP_ROWS))
    check('...a note names the render\'s method and the one-click operator '
          'is offered', any('Shadow Maps' in s for s in labels)
          and any('Per Light' in s for s in labels)
          and 'halcyon.shadows_per_light' in ops, str(labels[:4]))
    act, labels, ops = lamp_rows('RAY', 'MAP')
    check('render Ray Traced + lamp menu Shadow Map: the six map rows are '
          'greyed (they were drawn active and moved nothing), Samples / '
          'Density / Colour stay live',
          not any(act[r] for r in MAP_ROWS) and act['shadow_samples']
          and act['shadow_density'] and act['shadow_color'])
    check('...with the note and the operator',
          any('Ray Traced' in s for s in labels)
          and 'halcyon.shadows_per_light' in ops)
    act, labels, ops = lamp_rows('MAP', 'MAP')
    check('at the factory state (both on Shadow Map) nothing is added',
          'halcyon.shadows_per_light' not in ops
          and not any('Per Light' in s for s in labels)
          and all(act[r] for r in MAP_ROWS))
    act, labels, ops = lamp_rows('PER_LIGHT', 'RAY')
    check('Per Light + lamp Ray Traced: map rows greyed, no override note',
          not any(act[r] for r in MAP_ROWS) and act['shadow_samples']
          and 'halcyon.shadows_per_light' not in ops)
    act, labels, ops = lamp_rows('PER_LIGHT', 'MAP')
    check('Per Light + lamp Shadow Map: everything live',
          all(act[r] for r in MAP_ROWS) and act['shadow_samples'])
    for sdef in ('MAP', 'RAY', 'PER_LIGHT'):
        act, labels, ops = lamp_rows(sdef, 'NONE')
        check(f'render {sdef} + lamp None: every shadow row greyed, the '
              'menu itself live', act['shadow'] and not any(
                  act[r] for r in MAP_ROWS + ('shadow_samples',
                                              'shadow_density',
                                              'shadow_color')))
    act, labels, ops = lamp_rows('RAY', 'MAP', shadows=False)
    check('Shadows off for the render: rows greyed and one row says so',
          not act['shadow_density']
          and any('switched off' in s for s in labels))
    act, labels, ops = lamp_rows('RAY', 'MAP', ray_shadows=False)
    check('a traced lamp with Ray Shadows off: an error row, rows greyed',
          any('Ray Shadows is off' in s for s in labels)
          and not act['shadow_density'])
    act, labels, ops = lamp_rows('RAY', 'MAP', shadow_samples=1)
    check('a sized traced lamp at Samples 1: "Samples is 1" is drawn',
          any('Samples is 1' in s for s in labels), str(labels))
    act, labels, ops = lamp_rows('RAY', 'MAP', shadow_samples=1,
                                 l_shadow_samples=8)
    check('...but not when the lamp\'s own Samples lifts it',
          not any('Samples is 1' in s for s in labels))
    act, labels, ops = lamp_rows('MAP', 'MAP', shadow_samples=1)
    check('...and not on a mapped lamp',
          not any('Samples is 1' in s for s in labels))
    act, labels, ops = lamp_rows('PER_LIGHT', 'PLANAR')
    check('a planar lamp under Per Light shows Floor Plane Z',
          act.get('planar_plane_z') is True)
    act, labels, ops = lamp_rows('MAP', 'MAP', 'SUN', l_hemi=True)
    check('a hemi sun: rows greyed and a row says a hemi casts no shadows',
          not act['shadow_density']
          and any('hemi' in s.lower() for s in labels))
    check('light_shadow_method follows the renderer\'s rule',
          UI.light_shadow_method(types.SimpleNamespace(type='POINT'),
                                 types.SimpleNamespace(shadow='RAY'),
                                 types.SimpleNamespace(shadows=True,
                                                       shadow_default='MAP'))
          == 'MAP'
          and UI.light_shadow_method(
              types.SimpleNamespace(type='POINT'),
              types.SimpleNamespace(shadow='RAY'),
              types.SimpleNamespace(shadows=True,
                                    shadow_default='PER_LIGHT')) == 'RAY'
          and UI.light_shadow_method(
              types.SimpleNamespace(type='POINT'),
              types.SimpleNamespace(shadow='NONE'),
              types.SimpleNamespace(shadows=True,
                                    shadow_default='RAY')) == 'NONE')

    # the operator
    check('the Per Light operator is registered',
          UI.HALCYON_OT_shadows_per_light in UI.CLASSES
          and len(UI.HALCYON_OT_shadows_per_light.bl_description) >= 40)
    op = UI.HALCYON_OT_shadows_per_light.__new__(
        UI.HALCYON_OT_shadows_per_light)
    said = []
    op.report = lambda kind, msg: said.append(msg)
    hs = types.SimpleNamespace(shadow_default='MAP')
    ctx = types.SimpleNamespace(scene=types.SimpleNamespace(halcyon=hs),
                                screen=None)
    check('...and sets the render\'s Shadow Method to Per Light',
          op.execute(ctx) == {'FINISHED'} and hs.shadow_default == 'PER_LIGHT'
          and op.execute(ctx) == {'CANCELLED'})

    # the render's Shadows panel
    def scene_rows(sdef, **more):
        shs.shadow_default = sdef
        for k, v in more.items():
            setattr(shs, k, v)
        rec = _draw(UI.HALCYON_PT_shadows, shs)
        base = _defaults(PR.HalcyonSettings)
        for k in more:
            setattr(shs, k, getattr(base, k))
        return ({n: a for kind, n, a in rec if kind == 'prop'},
                [n for kind, n, a in rec if kind == 'label'])
    act, labels = scene_rows('RAY')
    check('Shadows panel under Ray Traced: the map dials grey, Ray '
          'Shadows and Ray Bias are drawn live',
          not act['shadow_map_size'] and not act['shadow_softness']
          and not act['shadow_bias'] and not act['shadow_map_depth']
          and act['shadow_samples'] and act.get('ray_shadows') is True
          and act.get('ray_bias') is True)
    act, labels = scene_rows('RAY', ray_shadows=False)
    check('...and an error row when Ray Shadows is off',
          any('Ray Shadows is off' in s for s in labels))
    act, labels = scene_rows('PER_LIGHT')
    check('Shadows panel under Per Light: map and traced dials both live, '
          'Floor Plane Z drawn (a planar lamp reads it)',
          act['shadow_map_size'] and act.get('ray_shadows') is True
          and 'planar_plane_z' in act)
    act, labels = scene_rows('MAP')
    check('Shadows panel under Shadow Maps: as it was (no traced rows)',
          act['shadow_map_size'] and 'ray_shadows' not in act
          and 'planar_plane_z' not in act)
    act, labels = scene_rows('PLANAR')
    check('Shadows panel under Planar: Floor Plane Z, map dials greyed',
          'planar_plane_z' in act and not act['shadow_map_size'])

    # the Ray Tracing panel: each of the two rows greys by what reads
    # it, never by the panel's master alone
    def rt_rows(**more):
        for k, v in more.items():
            setattr(shs, k, v)
        rec = _draw(UI.HALCYON_PT_raytrace, shs)
        base = _defaults(PR.HalcyonSettings)
        for k in more:
            setattr(shs, k, getattr(base, k))
        return {n: a for kind, n, a in rec if kind == 'prop'}
    act = rt_rows(shadow_default='RAY')
    check('Ray Tracing panel, master off, Shadow Method Ray Traced: Ray '
          'Shadows and Ray Bias are drawn LIVE (they were greyed while '
          'they decided every shadow), the reflection dials stay greyed',
          act['ray_shadows'] and act['ray_bias'] and not act['ray_depth']
          and not act['ray_reflection'])
    act = rt_rows(shadow_default='PER_LIGHT')
    check('...and under Per Light', act['ray_shadows'] and act['ray_bias'])
    act = rt_rows(shadow_default='MAP')
    check('master off, Shadow Maps, nothing traces: both greyed',
          not act['ray_shadows'] and not act['ray_bias'])
    act = rt_rows(shadow_default='MAP', raytrace=True)
    check('master on, Shadow Maps: Ray Bias live (reflections read it), '
          'Ray Shadows greyed (no light traces)',
          act['ray_bias'] and not act['ray_shadows'] and act['ray_depth'])
    act = rt_rows(shadow_default='MAP', ambient_occlusion=True)
    check('occlusion on: Ray Bias live', act['ray_bias'])
    act = rt_rows(shadow_default='RAY', ray_shadows=False)
    check('Ray Traced with Ray Shadows off: the switch stays live to turn '
          'back on, its bias greys', act['ray_shadows']
          and not act['ray_bias'])

    # the tooltips say the rules
    D = PR.DESCRIPTIONS

    def ltip(name):
        return PR.GROUP_DOCS['HalcyonLightSettings'][name]
    check('Shadow Method tooltip: every light, Per Light lets the menu '
          'decide', len(D['shadow_default']) >= 40
          and 'EVERY light' in D['shadow_default']
          and 'Per Light' in D['shadow_default']
          and 'does not choose' not in D['shadow_default'])
    check('Ray Shadows / Ray Bias tooltips: independent of the Ray '
          'Tracing switch', 'does not depend on the Ray Tracing switch'
          in D['ray_shadows'] and 'whether or not' in D['ray_bias']
          and len(D['ray_bias']) >= 40)
    check('the lamp\'s Shadows tooltip names Per Light, its Samples '
          'tooltip the hard shadow at 1, its Density every method',
          'Per Light' in ltip('shadow') and len(ltip('shadow')) >= 40
          and 'hard' in ltip('shadow_samples')
          and len(ltip('shadow_samples')) >= 40
          and 'traced' in ltip('shadow_density')
          and len(ltip('shadow_density')) >= 40,
          f"{ltip('shadow_samples')[:60]}...")


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
    print('all R254 ray-road checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
