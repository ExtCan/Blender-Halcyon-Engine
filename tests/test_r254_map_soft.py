"""R254: under Shadow Maps a lamp's SIZE shapes its shadow.

The field report: 'changing the size of the lights doesn't actually work,
the lights don't get softer / harder with size'. Under Shadow Maps the size
(POINT / SPOT Radius, SUN Angle, AREA Size and Size Y) was one constant blur
stamped on the map at a representative distance, spread over four hard
copies, capped at 24 texels, with the lookup point lifted off its surface by
the same factor -- so a bigger lamp SHRANK its shadow and a Sun at 0.2 rad
cast none at all.

The lookup is per receiver now (core/softmap.py, twin gpu/softmap.py): a
blocker search over the lamp's footprint, a penumbra by similar triangles, a
filter that wide on a bilinear lattice, every compare against the receiver
plane. Each test below states one law of it as a number, and every one of
them except the first fails on the lookup it replaces:

    a  a lamp of size 0 is the old lookup and the old shader text
    b  the shadow keeps its mass as the lamp grows (no shrinking)
    c  the penumbra widens with caster distance and with size, and is the
       traced road's within the map's texel
    d  the ramp is continuous at any Samples above 1 (no ghost copies, no
       bands); Samples 1 is the hard single-tap lookup for every lamp
    e  the same lamp is equally soft on a 512 and a 2048 map, and under a
       narrow spot cone
    f  unrelated scene bounds do not change the blur
    g  an Area's two sizes and its shape reach the shadow
    h  a sized lamp does not erode under the user Softness
    i  a size edit re-bakes nothing
    j  the GPU twin shades the CPU picture; what rides a texel stays out of
       the shader text
    k  a kernel crosses a cube-face seam
    l  no light under a contact, no false shadow on a lit receiver
    m  every caller keeps working (volumes, the caller's mask, the worker
       pool's pickles, a MIDPOINT map)
    n  the tooltips say what the dials now do

    python -m halcyon.tests.test_r254_map_soft
"""
import os
import pickle
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import (_mesh_concat, cube, demo_scene, look_at_matrix,
                         plane, sphere)
from ..core import lights as LI
from ..core import mathx as M
from ..core import raster as CRg
from ..core import render as R
from ..core import softmap as SM
from ..core.bvh import BVH
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
LAMP_POS = (0.0, -4.0, 6.0)
LAMP_AIM = (0.0, 1.5, 0.0)
#: the floor lines behind the wall: the caster-to-receiver distance grows
YS = (0.5, 1.5, 2.5, 3.5)
XS = np.linspace(0.0, 4.5, 1201)


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


def _settings(w=96, h=72, **kw):
    st = base_settings(w, h, shadows=True, transparency='NONE')
    st.shadow_default = 'PER_LIGHT'
    st.global_ambient_level = 0.0
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _box(lo, hi, mat=1, obj=1):
    V, N, UV, T, _m, _o = cube(centre=(0, 0, 0), size=1.0)
    lo = np.asarray(lo, np.float32)
    hi = np.asarray(hi, np.float32)
    V = (V + 0.5) * (hi - lo)[None, :] + lo[None, :]
    return (V.astype(np.float32), N, UV, T, mat, obj)


def _finish(parts, light, st, eye=(7.0, -3.5, 7.0), at=(0.3, 1.9, 0.0),
            smooth_mat=None):
    mats = [Material(name='Floor', index=0, model='LAMBERT',
                     diffuse=(0.8, 0.8, 0.8), specular_level=0.0),
            Material(name='Box', index=1, model='LAMBERT',
                     diffuse=(0.8, 0.3, 0.2), specular_level=0.0)]
    mesh = _mesh_concat(parts)
    mesh.smooth = np.zeros(mesh.tris.shape[0], bool) if smooth_mat is None \
        else (mesh.mat_index == smooth_mat)
    objs = [ObjectInfo(name=n, index=i, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=np.float32))
            for i, n in enumerate(('Floor', 'Box'))]
    cam = Camera(matrix_world=look_at_matrix(eye, at), lens=35.0,
                 sensor=36.0, clip_start=0.1, clip_end=200.0)
    sc = Scene(mesh=mesh, materials=mats, objects=objs,
               lights=[light] if light is not None else [], camera=cam,
               world=World(color=(0.0, 0.0, 0.0), ambient=(0.0, 0.0, 0.0)),
               settings=st)
    sc.images = {}
    return sc


def _wall(light, st, extra=()):
    """A floor and a wall slab STANDING on it (x -0.6..0.6, 3 tall): the
    shadow of its side edge runs from the contact out to the tip, so one
    scene holds every caster-to-receiver distance."""
    return _finish([plane(z=0.0, size=14.0, mat=0, obj=0),
                    _box((-0.6, -0.15, 0.0), (0.6, 0.15, 3.0))]
                   + list(extra), light, st)


def _lamp(kind, size, mode='MAP', pos=LAMP_POS, aim=LAMP_AIM, **kw):
    """size: POINT / SPOT radius (world), SUN angle (rad), AREA edge."""
    d = np.array(aim, np.float64) - np.array(pos, np.float64)
    d = d / np.linalg.norm(d)
    base = dict(type=kind, name='L', position=tuple(pos),
                direction=tuple(d), color=(1, 1, 1), shadow=mode)
    if kind == 'SUN':
        l = Light(energy=3.0, radius=float(size), **base)
    elif kind == 'AREA':
        l = Light(energy=600.0, **base)
        sz = size if isinstance(size, tuple) else (size, size)
        l.area_size = (float(sz[0]), float(sz[1]))
        x = np.cross(d, np.array([0, 0, 1.0]))
        x /= np.linalg.norm(x)
        l.area_x = tuple(x)
        l.area_y = tuple(np.cross(x, d))
    else:
        l = Light(energy=600.0, radius=float(size), **base)
        if kind == 'SPOT':
            l.spot_size = 1.5
    for k, v in kw.items():
        setattr(l, k, v)
    return l


def _prep(kind, size, scene=_wall, lamp_kw=None, **stkw):
    st = _settings(**stkw)
    sc = scene(_lamp(kind, size, **(lamp_kw or {})), st)
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    return sc, st


def _vis(sc, st, P, normal=(0, 0, 1.0), mask=None):
    """The engine's own shadow term at the points P for the scene's lamp."""
    light = sc.lights[0]
    P = np.asarray(P, np.float32)
    n = P.shape[0]
    N = np.tile(np.asarray(normal, np.float32)[None], (n, 1))
    L, _rad, dist = LI.sample(light, P, st)
    return LI.visibility(light, P, N, L.astype(np.float32), dist, st,
                         sample_xy=(np.arange(n, dtype=np.int64),
                                    np.full(n, 7, np.int64), 0), mask=mask)


def _ref(sc, st, P, normal=(0, 0, 1.0), nsamp=256):
    """The lamp's true shape: `nsamp` stratified BVH rays, the same set at
    every point (no hash noise). POINT / SPOT a disc of the radius facing
    the receiver, SUN a cone of the angle, AREA its rectangle or ellipse
    -- the traced road's own meaning of each size."""
    light = sc.lights[0]
    bvh = BVH.cached(sc.mesh.verts, sc.mesh.tris)
    P = np.asarray(P, np.float32)
    n = P.shape[0]
    N = np.tile(np.asarray(normal, np.float32)[None], (n, 1))
    L, _rad, dist = LI.sample(light, P, st)
    L = L.astype(np.float32)
    org = P + N * 2e-3
    acc = np.zeros(n, np.float64)
    i = np.arange(nsamp)
    r = np.sqrt((i + 0.5) / nsamp)
    th = i * np.pi * (3.0 - np.sqrt(5.0))
    da, db = r * np.cos(th), r * np.sin(th)
    if light.type == 'AREA':
        ax = np.asarray(light.area_x, np.float32)
        ay = np.asarray(light.area_y, np.float32)
        hx, hy = light.area_size[0] * 0.5, light.area_size[1] * 0.5
        lp = np.asarray(light.position, np.float32)
        if str(light.area_shape) not in ('DISK', 'ELLIPSE'):
            m = int(round(np.sqrt(nsamp)))
            g = (np.arange(m) + 0.5) / m * 2.0 - 1.0
            da, db = (v.ravel() for v in np.meshgrid(g, g))
        for a, b in zip(da, db):
            dl = (lp + ax * (hx * a) + ay * (hy * b))[None, :] - P
            ds = np.linalg.norm(dl, axis=1)
            acc += ~bvh.occluded(org, (dl / ds[:, None]).astype(np.float32),
                                 ds * (1 - 1e-3))
        return (acc / len(da)).astype(np.float32)
    t, b = M.orthonormal_basis(L)
    for a, bb in zip(da, db):
        if light.type == 'SUN':
            k = float(np.tan(light.radius * 0.5))
            Lj = M.normalize(L + t * (k * a) + b * (k * bb))
            acc += ~bvh.occluded(org, Lj.astype(np.float32),
                                 np.full(n, 1e9, np.float32))
        else:
            v = L * dist[:, None] + t * (light.radius * a) \
                + b * (light.radius * bb)
            ds = np.linalg.norm(v, axis=1)
            acc += ~bvh.occluded(org, (v / ds[:, None]).astype(np.float32),
                                 ds * (1 - 1e-3))
    return (acc / nsamp).astype(np.float32)


def _xline(y, xs=XS):
    return np.stack([xs, np.full_like(xs, y), np.zeros_like(xs)],
                    1).astype(np.float32)


def _edge(x, v, lo=0.1, hi=0.9):
    """(10%-90% width, distinct levels on the ramp, largest single step)
    of ONE shadow edge sampled along x, or None when the line carries no
    full transition. The OUTERMOST crossings count, so displaced ghost
    copies widen the edge instead of hiding inside it."""
    v = np.asarray(v, np.float64)
    x = np.asarray(x, np.float64)
    vmin, vmax = float(v.min()), float(v.max())
    if vmax - vmin < 0.5:
        return None
    t = (v - vmin) / (vmax - vmin)
    if t[-1] < t[0]:
        t, x = t[::-1], x[::-1]
    above_lo = np.nonzero(t > lo)[0]
    below_hi = np.nonzero(t < hi)[0]
    if above_lo.size == 0 or below_hi.size == 0:
        return None
    i0 = max(int(above_lo[0]) - 1, 0)
    i1 = min(int(below_hi[-1]) + 1, len(t) - 1)

    def cross(i, level):
        j = min(i + 1, len(t) - 1)
        if t[j] == t[i]:
            return x[i]
        return x[i] + (level - t[i]) / (t[j] - t[i]) * (x[j] - x[i])
    width = abs(cross(i1 - 1, hi) - cross(i0, lo))
    seg = t[i0:i1 + 1]
    return (float(width), int(len(np.unique(np.round(seg, 4)))),
            float(np.abs(np.diff(v)).max()))


def _widths(sc, st, fn=_vis, ys=YS):
    out = []
    for y in ys:
        e = _edge(XS, fn(sc, st, _xline(y)))
        out.append(None if e is None else e[0])
    return out


_GRID = None


def _floor_grid():
    global _GRID
    if _GRID is None:
        gx, gy = np.meshgrid(np.linspace(-3.5, 3.5, 141),
                             np.linspace(0.2, 6.5, 127))
        _GRID = (np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                          1).astype(np.float32),
                 (7.0 / 140) * (6.3 / 126))
    return _GRID


def _mass(sc, st):
    """(shadow mass, area darker than half) on the floor behind the wall,
    in world units^2: a shadow that only softens keeps its mass, one that
    shrinks or vanishes loses it."""
    P, cell = _floor_grid()
    v = _vis(sc, st, P)
    return float((1.0 - v).sum() * cell), float((v < 0.5).sum() * cell)


def _texel(sc, at=(0.0, 1.5, 0.0)):
    sm = sc.lights[0].shadow_map
    d = np.linalg.norm(np.asarray(at, np.float32) - sm.origin)
    return float(np.asarray(sm.texel_size(np.asarray([d], np.float32)))[0])


def _gpu(sc, st, w=96, h=72):
    """(cpu frame, simulated frame, covered mask, passes, atlases, why)."""
    cpu = R.render(sc, st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = CRg.GBuffer(w, h)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    if passes is None:
        return cpu, None, g.tri >= 0, None, None, why
    img, _hit = GSH.simulate(job, g, passes, atlases)
    return cpu, img, g.tri >= 0, passes, atlases, None


def _sources(passes):
    return '\n'.join(x for ps in passes for x in ps
                     if isinstance(x, str) and 'void main' in x)


def _classic(sc, st, P, normal=(0, 0, 1.0)):
    """The constant PCF lookup as `visibility` has always made it for a
    mapped lamp: the user's Softness as the tap radius, the normal offset
    grown by max(1, Softness)."""
    light = sc.lights[0]
    sm = light.shadow_map
    P = np.asarray(P, np.float32)
    n = P.shape[0]
    N = np.tile(np.asarray(normal, np.float32)[None], (n, 1))
    L, _rad, _dist = LI.sample(light, P, st)
    L = L.astype(np.float32)
    mid = LI.midpoint_map(light, st)
    bias = 0.0 if mid else float(light.shadow_bias or st.shadow_bias)
    ndl = np.clip(M.dot(N, L), 0.0, 1.0)
    slope = bias * (1.0 + 2.0 * (1.0 - ndl))
    soft = max(float(light.shadow_softness) * st.shadow_softness, 0.0)
    texel = sm.texel_size(np.linalg.norm(P - sm.origin[None, :], axis=1))
    offset = np.zeros(n, np.float32) if mid else \
        texel * (1.5 + 2.5 * np.sqrt(np.maximum(1.0 - ndl * ndl, 0.0))) * \
        max(1.0, soft)
    lit = sm.lookup(P + N * offset[:, None], slope, soft,
                    LI.shadow_sample_count(light, st))
    return (1.0 - (1.0 - lit) * float(light.shadow_density)) \
        .astype(np.float32)


# -------------------------------------------------------------------- tests


def test_a_size_zero_is_the_old_lookup():
    """Default neutrality: a lamp without a size takes the constant PCF
    lookup, statement for statement, and the old shader text."""
    P, _cell = _floor_grid()
    for kind in ('POINT', 'SPOT', 'SUN', 'AREA'):
        for kw in ({}, {'shadow_softness': 4.0, 'shadow_samples': 9},
                   {'shadow_softness': 0.0}, {'shadow_map_depth': 'MIDPOINT'}):
            sc, st = _prep(kind, 0.0, **kw)
            check(f'{kind} size 0 {kw or ""}: the term is the classic PCF '
                  'lookup, bitwise',
                  SM.lamp_shape(sc.lights[0]) is None
                  and np.array_equal(_vis(sc, st, P), _classic(sc, st, P)))
    # the GPU road: no sized key in the meta, the atlas or the text
    for kind in ('POINT', 'SPOT', 'SUN'):
        sc, st = _prep(kind, 0.0)
        meta, atlas, why = GSH._shadow_meta(sc.lights[0], st)
        key, build = atlas
        px = build()
        text = GM._shadow_function(0, dict(meta, voff=0),
                                   {'shadow_samples': 4})
        check(f'{kind} size 0: the shadow meta, atlas and shader text '
              'carry nothing of the sized lookup',
              'soft' not in meta and 'sized' not in key
              and float(np.abs(px[..., 1:]).max()) == 0.0
              and 'hal_sm_' not in text and '.g;' not in text
              and text.count('texelFetch(hal_shadowpack') == 4,
              f'key {key[:4]}')
    # the demo scene (its own SUN + POINT, size 0) plans the classic text
    st = base_settings(96, 72, shadows=True)
    sc = demo_scene(st, with_texture=False)
    _cpu, sim, _cov, passes, atlases, why = _gpu(sc, st)
    check('the demo scene (lamps of size 0) emits no sized lookup',
          passes is not None and 'hal_sm_' not in _sources(passes)
          and all('sized' not in k for k in atlases['hal_shadowpack'][0][1:]),
          str(why))


def test_b_the_shadow_keeps_its_mass():
    """size-map-02: a bigger lamp softens its shadow; it does not shrink
    it and a Sun does not lose it. (Old: POINT 7.47 -> 3.6 at radius 0.5,
    a Sun at 0.2 rad cast nothing.)"""
    for kind, zero, sizes in (('POINT', 0.0, (0.1, 0.5, 1.0, 2.0)),
                              ('SPOT', 0.0, (0.1, 0.5, 1.0)),
                              ('SUN', 0.0, (0.00918, 0.05, 0.2, 0.6)),
                              ('AREA', 0.0, (0.25, 1.0, 3.0))):
        sc, st = _prep(kind, zero)
        m0, a0 = _mass(sc, st)
        rows = []
        ok = m0 > 1.0
        for size in sizes:
            sc, st = _prep(kind, size)
            m, a = _mass(sc, st)
            rows.append(f'{size:g}: {m:.2f}')
            ok = ok and abs(m / m0 - 1.0) < 0.15
        check(f'{kind}: the shadow mass stays within 15% of the size-0 '
              f'mass ({m0:.2f}) at every size', ok, ', '.join(rows))
    # the umbra is still there under a large lamp: the darkest point
    sc, st = _prep('SUN', 0.2)
    v = _vis(sc, st, _xline(0.5, np.linspace(-0.3, 0.3, 61)))
    check('a Sun at 0.2 rad still casts a full umbra behind the wall',
          float(v.max()) < 0.05, f'max term {float(v.max()):.3f}')


def test_c_the_penumbra_follows_distance_and_size():
    """size-map-01 / -04: sharp where the caster touches the receiver,
    wider with the caster-to-receiver distance, wider with the lamp, and
    the traced road's width within the map's own texel."""
    def close(w, r, tex):
        return w is not None and r is not None and \
            abs(w - r) <= 0.25 * r + 2.0 * tex

    for kind, sizes, mapsize in (('SPOT', (0.1, 0.25, 0.5), 2048),
                                 ('POINT', (0.25, 0.5), 512),
                                 ('SUN', (0.05, 0.1), 2048),
                                 ('AREA', (0.5, 1.0), 512)):
        prev = None
        for size in sizes:
            sc, st = _prep(kind, size, shadow_map_size=mapsize)
            w = _widths(sc, st)
            r = _widths(sc, st, _ref)
            tex = _texel(sc)
            got = [x for x in w if x is not None]
            txt = ' '.join('none' if x is None else f'{x:.3f}' for x in w)
            rtx = ' '.join('none' if x is None else f'{x:.3f}' for x in r)
            check(f'{kind} size {size:g}, map {mapsize}: the edge widens '
                  'with the caster distance',
                  len(got) >= 3 and all(b > a * 1.15
                                        for a, b in zip(got, got[1:])),
                  f'map {txt} | traced {rtx}')
            check(f'{kind} size {size:g}, map {mapsize}: each width is the '
                  'traced one within 25% + 2 texels',
                  all(close(a, b, tex) for a, b in zip(w, r)
                      if b is not None),
                  f'texel {tex:.4f}')
            if prev is not None:
                check(f'{kind}: size {size:g} is softer than the size '
                      'before it on every line',
                      all(a > b * 1.2 for a, b in zip(w, prev)
                          if a is not None and b is not None))
            prev = w


def test_d_the_ramp_is_continuous():
    """size-map-05: a smooth penumbra at any Samples above 1 -- four taps
    used to be four displaced hard copies (5 levels, steps of 0.25). One
    Sample stays the hard single-tap lookup for EVERY lamp, as one traced
    ray is hard: the 'hard shadow maps' presets ask for exactly that on
    lamps that carry Blender's default sizes, and it used to shrink."""
    for kind, size in (('POINT', 0.5), ('SUN', 0.1), ('AREA', 1.0)):
        for samples in (2, 4, 16):
            sc, st = _prep(kind, size, shadow_samples=samples)
            e = _edge(XS, _vis(sc, st, _xline(2.5)))
            check(f'{kind} size {size:g} at Samples {samples}: at least 12 '
                  'levels across the penumbra and no step above 0.08',
                  e is not None and e[1] >= 12 and e[2] <= 0.08,
                  'no edge' if e is None else
                  f'{e[1]} levels, largest step {e[2]:.3f}')
    check('Samples maps to 4 filter taps per sample, 16 to 64, and half '
          'as many search taps, 12 to 32',
          [SM.tap_counts(s) for s in (2, 4, 8, 16, 64)]
          == [(16, 12), (16, 12), (32, 16), (64, 32), (64, 32)])
    P, _cell = _floor_grid()
    for kind, size in (('POINT', 0.5), ('SUN', 0.1), ('AREA', 1.0)):
        sc, st = _prep(kind, size, shadow_samples=1)
        hard = _vis(sc, st, P)
        m1 = float((1.0 - hard).sum())
        sc0, st0 = _prep(kind, 0.0, shadow_samples=1)
        m0 = float((1.0 - _vis(sc0, st0, P)).sum())
        check(f'{kind} size {size:g} at Samples 1: the hard single-tap '
              'lookup (two levels, the classic statement bitwise) with the '
              "size-0 lamp's shadow mass -- nothing shrinks",
              SM.lookup_shape(sc.lights[0], 1) is None
              and np.array_equal(hard, _classic(sc, st, P))
              and len(np.unique(hard)) == 2 and abs(m1 / m0 - 1.0) < 1e-6,
              f'{len(np.unique(hard))} levels, mass ratio {m1 / m0:.4f}')
    # the lamp's own dial at 1 makes that lamp hard under a soft render
    sc, st = _prep('POINT', 0.5, shadow_samples=4,
                   lamp_kw={'shadow_samples': 1})
    v = _vis(sc, st, P)
    meta, _atl, _why = GSH._shadow_meta(sc.lights[0], st)
    check("a lamp whose own Samples is 1 is hard under a render at 4, on "
          'both roads (no sized key in its shadow meta)',
          len(np.unique(v)) == 2 and 'soft' not in meta)
    # the lamp's own Samples dial sizes the filter too (0 = the render's)
    sc, st = _prep('POINT', 0.5, shadow_samples=4)
    a = _vis(sc, st, _xline(2.5))
    sc.lights[0].shadow_samples = 16
    b = _vis(sc, st, _xline(2.5))
    st.shadow_samples = 16
    sc.lights[0].shadow_samples = 0
    c = _vis(sc, st, _xline(2.5))
    check("the lamp's own Samples overrides the render's for the sized "
          'lookup (0 inherits)', not np.array_equal(a, b)
          and np.array_equal(b, c))
    # thin casters: poles 2 to 9 texels wide under a lamp of Blender's
    # default size. (The stated limit of the method starts above it: a
    # caster much thinner than the lamp's footprint can fall between the
    # search taps of a WIDE lamp -- radius 0.5 and up here -- and its
    # faint shadow then carries steps; more Samples shrink them.)

    def poles(light, st):
        parts = [plane(z=0.0, size=14.0, mat=0, obj=0)]
        for x, wd in ((-2.0, 0.06), (0.0, 0.12), (2.0, 0.25)):
            parts.append(_box((x - wd / 2, -wd / 2, 0.0),
                              (x + wd / 2, wd / 2, 3.0)))
        return _finish(parts, light, st)
    sc, st = _prep('POINT', 0.25, scene=poles)
    xs = np.linspace(-3.5, 3.5, 1401)
    worst, rms = 0.0, 0.0
    for y in (0.5, 1.5, 3.0):
        v = _vis(sc, st, _xline(y, xs))
        r = _ref(sc, st, _xline(y, xs))
        worst = max(worst, float(np.abs(np.diff(v)).max()))
        rms = max(rms, float(np.sqrt(np.mean((v - r) ** 2))))
    check('thin poles under a radius 0.25 lamp: a continuous shadow that '
          'follows the traced one', worst < 0.12 and rms < 0.04,
          f'largest step {worst:.3f}, rms {rms:.4f}')


def test_e_the_map_size_does_not_change_the_softness():
    """size-map-03: the blur was capped at 24 TEXELS, so the same lamp was
    softer on a 512 map than on a 2048 one and large sizes stopped
    responding. The cap is a fraction of the map now."""
    for kind, size in (('SPOT', 0.5), ('SUN', 0.1)):
        w = {}
        for mapsize in (512, 2048):
            sc, st = _prep(kind, size, shadow_map_size=mapsize)
            w[mapsize] = _widths(sc, st)
            if mapsize == 512:
                tex = _texel(sc)
        pairs = [(a, b) for a, b in zip(w[512], w[2048])
                 if a is not None and b is not None]
        check(f'{kind} size {size:g}: the 512 and the 2048 map give one '
              'penumbra, within two 512 texels',
              len(pairs) >= 3 and all(abs(a - b) <= 2.0 * tex
                                      for a, b in pairs),
              ' '.join(f'{a:.3f}/{b:.3f}' for a, b in pairs)
              + f' texel {tex:.4f}')
    wide = []
    for size in (0.5, 1.0):
        sc, st = _prep('SPOT', size, shadow_map_size=2048)
        wide.append(_widths(sc, st)[1])
    check('on a 2048 map radius 1.0 is still softer than radius 0.5 (the '
          'old cap made them one picture)',
          wide[1] > wide[0] * 1.5, f'{wide[0]:.3f} -> {wide[1]:.3f}')
    check('the penumbra cap is an ANGLE at the lamp under a perspective '
          'map (a sixteenth of a cube face at any map size, a quarter of '
          'a narrow cone\'s map at most) and a sixteenth of an ortho map',
          [float(SM.cap_texels(n, 1.0, True)) for n in (512, 2048)]
          == [32.0, 128.0]
          and float(SM.cap_texels(512, 0.1, True)) == 128.0
          and float(SM.cap_texels(512, 9.0, False)) == 32.0)
    # a NARROW spot: its map covers only its cone, so a cap counted in
    # texels (or as a share of the map) tied the penumbra to the cone
    # angle -- radius 0.5 and 1.0 were one picture under a 0.35 rad cone
    xs = np.linspace(0.0, 2.5, 801)
    got = []
    for size in (0.5, 1.0):
        st = _settings()
        sc = _wall(_lamp('SPOT', size, aim=(0.6, 1.5, 0.0),
                         spot_size=0.35), st)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        P = _xline(2.5, xs)
        e, r = _edge(xs, _vis(sc, st, P)), _edge(xs, _ref(sc, st, P))
        got.append((e, r))
    check('a 0.35 rad spot: radius 1.0 is twice as soft as radius 0.5 and '
          'both are the traced widths within 15%',
          all(e is not None and r is not None
              and abs(e[0] - r[0]) <= 0.15 * r[0] for e, r in got)
          and got[1][0][0] > 1.7 * got[0][0][0],
          ' '.join(f'{e[0]:.3f} (traced {r[0]:.3f})' for e, r in got
                   if e is not None and r is not None))


def test_f_scene_bounds_do_not_change_the_blur():
    """size-map-04: the size was evaluated at ONE representative distance
    (the scene's bounding-box centre), so a far-away object halved a
    Point lamp's blur and a bigger floor tripled a Sun's."""
    far = [_box((30.0, 30.0, 0.0), (31.0, 31.0, 1.0))]
    sc, st = _prep('POINT', 0.25)
    a = _widths(sc, st)[1]
    st2 = _settings()
    sc2 = _wall(_lamp('POINT', 0.25), st2, extra=far)
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc2, st2)
    b = _widths(sc2, st2)[1]
    check('a box 40 units away leaves a Point lamp\'s penumbra alone',
          abs(a - b) <= 0.05 * a, f'{a:.4f} vs {b:.4f}')
    check('no map carries a blur derived from the lamp',
          not hasattr(sc.lights[0].shadow_map, 'soft_extra')
          and not hasattr(LI, '_stamp_soft'))


def test_g_an_area_lamps_sizes_and_shape_reach_the_shadow():
    """size-map-08: the map road read the MEAN of Size and Size Y, in
    every direction; the rectangle's two axes are projected through the
    map face now, and a disc is not a square."""
    w = {}
    for tag, size in (('wide', (0.8, 0.05)), ('tall', (0.05, 0.8))):
        sc, st = _prep('AREA', size)
        w[tag] = (_widths(sc, st)[1], _widths(sc, st, _ref)[1], _texel(sc))
    check('a 0.8 x 0.05 rectangle blurs the edge it lies across, a '
          '0.05 x 0.8 one hardly does',
          w['wide'][0] > 3.0 * w['tall'][0],
          f"{w['wide'][0]:.3f} vs {w['tall'][0]:.3f}")
    check('both follow the traced rectangle',
          all(abs(m - r) <= 0.25 * r + 2.0 * t for m, r, t in w.values()),
          f"traced {w['wide'][1]:.3f} / {w['tall'][1]:.3f}")
    sc, st = _prep('AREA', 1.0)
    sq = _vis(sc, st, _xline(2.5))
    sc.lights[0].area_shape = 'DISK'
    dk = _vis(sc, st, _xline(2.5))
    e_sq, e_dk = _edge(XS, sq), _edge(XS, dk)
    check('a DISK of the same size is a narrower lamp than the SQUARE '
          '(the square stands in as the disc of equal area)',
          e_sq is not None and e_dk is not None and e_dk[0] < e_sq[0],
          f'{e_sq[0]:.3f} vs {e_dk[0]:.3f}')
    a = SM.lamp_shape(_lamp('AREA', (0.0, 0.0)))
    b = SM.lamp_shape(_lamp('AREA', (0.4, 0.0)))
    check('an Area of size 0 x 0 has no size; one live axis is enough',
          a is None and b is not None and b['kind'] == 'RECT')


def test_h_a_sized_lamp_does_not_erode_under_softness():
    """soft-erode-01, the sized half: the normal offset was multiplied by
    the blur radius, so Softness 8 lifted the receiver out from under its
    caster (area 7.4 -> 4.6) and 32 deleted the shadow. A sized lamp
    compares against the receiver plane and carries no blur factor."""
    for kind, size in (('POINT', 0.25), ('SUN', 0.05), ('AREA', 0.5)):
        rows, ok, w = [], True, []
        sc, st = _prep(kind, size, shadow_softness=1.0)
        m1, a1 = _mass(sc, st)
        for s in (1.0, 4.0, 8.0, 16.0):
            sc, st = _prep(kind, size, shadow_softness=s)
            m, a = _mass(sc, st)
            rows.append(f'S{s:g}: {m:.2f} / {a:.2f}')
            # the mass at every Softness; the half-dark area while the
            # blur is still narrower than the shadow itself
            ok = ok and abs(m / m1 - 1.0) < 0.1 and \
                (s > 8.0 or a >= 0.9 * a1)
            w.append(_widths(sc, st)[0])
        check(f'{kind} size {size:g}: the shadow keeps its mass from '
              'Softness 1 to 16 (mass / area darker than half)', ok,
              ', '.join(rows))
        check(f'{kind} size {size:g}: Softness still widens the edge',
              all(x is not None for x in w) and w[3] > w[2] > w[1] > w[0],
              ' '.join('none' if x is None else f'{x:.3f}' for x in w))
    ndl = np.array([1.0, 0.5, 0.3, 0.2, 0.1, 0.0], np.float32)
    sc_ = SM.offset_scale(ndl)
    check('the normal offset is a quarter of the classic one where the '
          'plane is exact and whole towards grazing light',
          np.allclose(sc_, [0.25, 0.25, 0.25, 0.625, 1.0, 1.0], atol=1e-6),
          str(np.round(sc_, 3)))
    # the user Softness of a lamp WITHOUT a size is not touched here
    P, _c = _floor_grid()
    sc, st = _prep('POINT', 0.0, shadow_softness=8.0)
    check('a lamp of size 0 at Softness 8 is still the classic lookup '
          '(its own decision)', np.array_equal(_vis(sc, st, P),
                                               _classic(sc, st, P)))


def test_i_a_size_edit_rebakes_nothing():
    """size-map-07: the size sat in the map cache key because the built
    map carried the blur; every drag step re-rasterised six cube faces
    whose texels had not moved."""
    calls = {'n': 0}
    orig = LI._render_depth

    def counted(*a, **k):
        calls['n'] += 1
        return orig(*a, **k)
    LI._render_depth = counted
    try:
        for kind, sizes in (('POINT', (0.0, 0.1, 0.4)),
                            ('SUN', (0.0, 0.02, 0.1)),
                            ('SPOT', (0.1, 0.5)),
                            ('AREA', (0.2, 1.0))):
            st = _settings()
            lamp = _lamp(kind, sizes[0])
            sc = _wall(lamp, st)
            LI.clear_shadow_cache()
            first = R.render(sc, st)
            baked = calls['n']
            sm = lamp.shadow_map
            frames = [first]
            calls['n'] = 0
            for s in sizes[1:]:
                if kind == 'AREA':
                    lamp.area_size = (s, s)
                else:
                    lamp.radius = s
                frames.append(R.render(sc, st))
            check(f'{kind}: {len(sizes) - 1} size edits rasterise no map '
                  f'(the first frame baked {baked}) and are served the '
                  'same map object',
                  baked >= 1 and calls['n'] == 0 and lamp.shadow_map is sm)
            check(f'{kind}: and every edit still changes the picture',
                  all(float(np.abs(a - b).max()) > 1e-3
                      for a, b in zip(frames, frames[1:])))
    finally:
        LI._render_depth = orig
    a, b = _lamp('POINT', 0.0), _lamp('POINT', 0.7)
    c, d = _lamp('AREA', 0.2), _lamp('AREA', (1.0, 3.0))
    check('the lamp size is no part of the shadow-map signature',
          LI._light_shadow_signature(a) == LI._light_shadow_signature(b)
          and LI._light_shadow_signature(c) == LI._light_shadow_signature(d))
    e = _lamp('POINT', 0.0, pos=(0.0, -4.0, 6.5))
    check('its pose still is',
          LI._light_shadow_signature(a) != LI._light_shadow_signature(e))


def test_j_the_gpu_twin():
    """Two roads, one picture: the GLSL emitted by gpu/softmap.py against
    the CPU through the simulator, and what rides a texel stays out of
    the text."""
    cases = [('POINT', 0.25, {}, {}), ('POINT', 1.0, {}, {}),
             ('SPOT', 0.25, {}, {}), ('SUN', 0.05, {}, {}),
             ('SUN', 0.2, {}, {}), ('AREA', 0.5, {}, {}),
             ('AREA', (2.0, 0.1), {}, {}),
             ('AREA', 1.0, {'area_shape': 'DISK'}, {}),
             ('POINT', 0.25, {}, {'shadow_samples': 16}),
             ('POINT', 0.25, {}, {'shadow_softness': 4.0}),
             ('SUN', 0.05, {}, {'shadow_softness': 0.0}),
             ('POINT', 0.5, {}, {'shadow_map_depth': 'MIDPOINT'}),
             ('SPOT', 0.3, {'shadow_density': 0.6, 'shadow_bias': 0.05,
                            'shadow_softness': 2.0, 'shadow_samples': 8},
              {})]
    for kind, size, lkw, skw in cases:
        st = _settings(**skw)
        sc = _wall(_lamp(kind, size, **lkw), st)
        LI.clear_shadow_cache()
        cpu, sim, cov, passes, _atl, why = _gpu(sc, st)
        label = f'{kind} size {size} {lkw or ""} {skw or ""}'
        if not check(f'{label}: the deferred pass plans', sim is not None,
                     str(why)):
            continue
        d = np.abs(sim[cov] - cpu[cov][:, :3]).max(axis=1)
        # an AREA lamp's own lighting term differs by up to 3e-3 between
        # the roads with shadows off; the shadow term adds nothing to it
        bar = 6e-3 if kind == 'AREA' else 1e-4
        check(f'{label}: the simulator shades the CPU picture',
              float(d.max()) < bar and 'hal_sm_blk0' in _sources(passes),
              f'max {float(d.max()):.6f} over {int(cov.sum())} px')
    # the sized frame is not the unshadowed frame (the twin is not vacuous)
    st = _settings()
    sc = _wall(_lamp('POINT', 0.5), st)
    LI.clear_shadow_cache()
    lit = R.render(sc, st)
    sc.lights[0].shadow = 'NONE'
    check('and the shadow is in the picture',
          float(np.abs(R.render(sc, st) - lit).max()) > 0.05)

    # what rides a texel: the map's nearest depth. A caster moving nearer
    # the lamp (inside the scene's bounds) re-keys the atlas and leaves
    # the shader text alone
    def plan(extra):
        st = _settings()
        sc = _wall(_lamp('POINT', 0.3), st, extra=extra)
        LI.clear_shadow_cache()
        _cpu, sim, _cov, passes, atlases, why = _gpu(sc, st)
        key, build = atlases['hal_shadowpack']
        return _sources(passes), key, build(), sc.lights[0].shadow_map
    t0, k0, px0, sm0 = plan([_box((2.0, 2.0, 0.0), (2.4, 2.4, 0.5))])
    t1, k1, px1, sm1 = plan([_box((2.0, -2.0, 2.0), (2.4, -1.6, 2.9))])
    check('the nearest depth of the map rides the G channel of the '
          "lamp's first atlas texel",
          float(px0[0, 0, 1]) == f32(sm0.d_min)
          and float(px1[0, 0, 1]) == f32(sm1.d_min)
          and sm1.d_min < sm0.d_min
          and float(np.abs(px0[..., 1]).sum()) == float(px0[0, 0, 1]),
          f'{sm0.d_min:.4f} -> {sm1.d_min:.4f}')
    check('a nearer caster re-keys the atlas and leaves the shader text '
          'untouched', k0 != k1 and t0 == t1)
    # what is baked: the size is structure
    st = _settings()
    for a, b, what in ((_lamp('POINT', 0.2), _lamp('POINT', 0.3), 'radius'),
                       (_lamp('SUN', 0.02), _lamp('SUN', 0.05), 'angle'),
                       (_lamp('AREA', (1.0, 0.5)), _lamp('AREA', (1.0, 0.6)),
                        'Size Y'),
                       (_lamp('AREA', 1.0),
                        _lamp('AREA', 1.0, area_shape='DISK'), 'shape')):
        for l in (a, b):
            sc = _wall(l, st)
            LI.clear_shadow_cache()
            LI.build_shadow_maps(sc, st)
        check(f"a mapped lamp's {what} is light-signature structure (it "
              'is baked as a literal)', GSH._light_sig(a) != GSH._light_sig(b))
    # the strict front-end, when it is installed
    try:
        from . import glsl_strict as GS
        have = GS._glslang() is not None
    except Exception:                                           # noqa: BLE001
        have = False
    if not have:
        note('HALCYON_GLSLANG is not set: the strict GLSL compile of the '
             'sized shadow functions did not run here '
             '(tests/glsl_strict.py check_mapped_shadows)')
    else:
        bad = []
        for kind, size in (('POINT', 0.3), ('SPOT', 0.3), ('SUN', 0.05),
                           ('AREA', (0.5, 0.25))):
            st = _settings()
            sc = _wall(_lamp(kind, size), st)
            LI.clear_shadow_cache()
            _c, _s, _m, passes, _a, why = _gpu(sc, st)
            for ps in passes or ():
                for x in ps:
                    if isinstance(x, str) and 'void main' in x:
                        ok, msg = GS.compile_source(x)
                        if not ok:
                            bad.append(f'{kind}: {msg[:200]}')
        check('a strict GLSL 4.30 front-end accepts the sized shadow '
              'function of every lamp kind', not bad, '; '.join(bad[:2]))


def test_k_a_kernel_crosses_a_cube_seam():
    """A shadow edge running ALONG a cube-face seam (the wall's top edge
    lands on the floor line under the -z / +y boundary): a tap that
    leaves its face reads the neighbouring one. Clamped at the face edge
    the line jumps by a full 1.0."""
    def scene(light, st):
        return _finish([plane(z=0.0, size=14.0, mat=0, obj=0),
                        _box((-1.5, -0.15, 0.0), (1.5, 0.15, 1.85))],
                       light, st)
    ys = np.linspace(1.2, 2.8, 641)
    P = np.stack([np.full_like(ys, 0.2), ys, np.zeros_like(ys)],
                 1).astype(np.float32)
    for kind, size in (('POINT', 0.5), ('AREA', 1.0)):
        sc, st = _prep(kind, size, scene=scene)
        v = _vis(sc, st, P)
        r = _ref(sc, st, P)
        step = float(np.abs(np.diff(v)).max())
        rms = float(np.sqrt(np.mean((v - r) ** 2)))
        check(f'{kind} size {size:g}: the penumbra runs through the seam '
              'without a step and follows the traced one',
              step < 0.03 and rms < 0.06 and float(v.min()) < 0.05
              and float(v.max()) > 0.95,
              f'largest step {step:.4f}, rms {rms:.4f}')
    # the neighbour read itself: a texel one past the edge of a face is
    # the neighbouring face's texel along the same direction
    sc, st = _prep('POINT', 0.5, scene=scene)
    cubemap = sc.lights[0].shadow_map
    size = cubemap.faces[0].size
    fetch = SM._CubeFetch(cubemap.faces, 5)
    yy = np.arange(8, size - 8, dtype=np.float32)
    inside, _x, _y = fetch(np.full_like(yy, size - 1.0), yy)
    beyond, _x, _y = fetch(np.full_like(yy, float(size)), yy)
    rel = np.abs(beyond - inside) / np.maximum(inside, 1e-6)
    check('a column one texel past a face edge continues the depth of '
          'the column before it (the median jump is under 2%)',
          float(np.median(rel)) < 0.02, f'median {float(np.median(rel)):.4f}')


def test_l_no_leak_and_no_acne():
    """The receiver-plane compare at work: nothing lit under a contact,
    nothing dark on a lit receiver."""
    # under the contact: the floor just behind the wall's foot
    gx, gy = np.meshgrid(np.linspace(-0.3, 0.3, 13),
                         np.linspace(0.25, 0.6, 8))
    P = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                 1).astype(np.float32)
    for kind, size in (('POINT', 0.25), ('POINT', 0.5), ('SPOT', 0.5),
                       ('SUN', 0.1), ('AREA', 1.0)):
        sc, st = _prep(kind, size)
        v = _vis(sc, st, P)
        check(f'{kind} size {size:g}: no light leaks under the wall\'s '
              'foot', float(v.max()) < 0.02, f'max {float(v.max()):.4f}')

    # a lit ramp at grazing incidence, nothing above it
    def ramp(tilt):
        def scene(light, st):
            V, N, UV, T, _m, _o = plane(z=0.0, size=8.0, mat=0, obj=0)
            c, s = np.cos(tilt), np.sin(tilt)
            Rm = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
            return _finish([((V @ Rm.T).astype(np.float32),
                             (N @ Rm.T).astype(np.float32), UV, T, 0, 0),
                            _box((-6, -6, -3.0), (-5.5, -5.5, -2.5))],
                           light, st)
        return scene
    for kind, size in (('POINT', 0.25), ('POINT', 1.0), ('SUN', 0.2)):
        worst = 0.0
        for deg in (60, 75, 82):
            tilt = np.radians(deg)
            st = _settings()
            l = Light(type=kind, name='L', position=(0.0, 0.0, 9.0),
                      direction=(0.0, 0.0, -1.0), color=(1, 1, 1),
                      energy=900.0, radius=float(size), shadow='MAP')
            sc = ramp(tilt)(l, st)
            LI.clear_shadow_cache()
            LI.build_shadow_maps(sc, st)
            g = np.linspace(-3.0, 3.0, 61)
            a, b = np.meshgrid(g, g)
            c, s = np.cos(tilt), np.sin(tilt)
            Pr = np.stack([a.ravel() * c, b.ravel(), -a.ravel() * s],
                          1).astype(np.float32)
            v = _vis(sc, st, Pr, (float(s), 0.0, float(c)))
            worst = max(worst, float((1.0 - v).mean()))
        check(f'{kind} size {size:g}: a lit ramp tilted 60 to 82 degrees '
              'from the light carries no false shadow',
              worst < 0.005, f'mean (1 - term) at most {worst:.5f}')

    # a concave corner: the wall's lit face just above the floor
    def room(light, st):
        return _finish([plane(z=0.0, size=14.0, mat=0, obj=0),
                        _box((-3.0, 1.0, 0.0), (3.0, 1.3, 3.0))], light, st)
    zs = np.linspace(0.005, 0.6, 120)
    Pw = np.stack([np.full_like(zs, 0.3), np.full_like(zs, 1.0), zs],
                  1).astype(np.float32)
    for kind, size in (('POINT', 0.5), ('SUN', 0.05), ('SUN', 0.2)):
        st = _settings()
        sc = room(_lamp(kind, size, pos=(0.5, -4.0, 6.0),
                        aim=(0.0, 1.0, 0.0)), st)
        LI.clear_shadow_cache()
        LI.build_shadow_maps(sc, st)
        v = _vis(sc, st, Pw, (0.0, -1.0, 0.0))
        check(f'{kind} size {size:g}: the floor does not shadow the wall '
              'face standing on it', float(v.min()) > 0.75
              and float(v.mean()) > 0.99,
              f'min {float(v.min()):.3f} mean {float(v.mean()):.4f}')

    # smooth spheres against the traced road, as frames
    def balls(light, st):
        parts = [plane(z=0.0, size=14.0, mat=0, obj=0)]
        for x, y in ((-1.6, 0.8), (0.0, 1.6), (1.6, 0.8)):
            parts.append(sphere(centre=(x, y, 0.9), radius=0.9, mat=1,
                                obj=1, segs=48, rings=24))
        return _finish(parts, light, st, eye=(0.0, -6.5, 5.0),
                       at=(0.0, 1.0, 0.5), smooth_mat=1)

    def frame(kind, size, mode, **kw):
        st = _settings(160, 120, **kw)
        if mode == 'RAY':
            st.shadow_default = 'RAY'
        sc = balls(_lamp(kind, size, mode), st)
        LI.clear_shadow_cache()
        return R.render(sc, st)
    for kind, size in (('POINT', 0.25), ('SUN', 0.05)):
        none = frame(kind, size, 'NONE')
        ray = frame(kind, size, 'RAY', shadow_samples=32)
        new = frame(kind, size, 'MAP')
        b = none[..., :3].mean(2)
        on_ball = (b > 0.02 * b.max()) & (none[..., 0] > none[..., 2] * 1.5)
        tr = np.clip(ray[..., :3].mean(2) / np.maximum(b, 1e-6), 0, 1)
        tn = np.clip(new[..., :3].mean(2) / np.maximum(b, 1e-6), 0, 1)
        err = float(np.abs(tn - tr)[on_ball].mean())
        dark = int(((tr - tn) > 0.25)[on_ball].sum())
        check(f'{kind} size {size:g}: on smooth spheres the mapped term '
              'is the traced one (no acne, no terminator shift)',
              err < 0.01 and dark < 0.02 * int(on_ball.sum()),
              f'mean |map - traced| {err:.4f}, {dark} of '
              f'{int(on_ball.sum())} px darker by 0.25')


def test_m_every_caller_keeps_working():
    """`visibility` and the maps as their other users hold them."""
    sc, st = _prep('POINT', 0.4)
    light = sc.lights[0]
    P, _c = _floor_grid()
    P = P[::7]
    n = P.shape[0]
    # a volume sample: the caller passes N = L (core/volume.py)
    L, _rad, dist = LI.sample(light, P, st)
    L = L.astype(np.float32)
    Pv = P + np.array([0.0, 0.0, 0.7], np.float32)
    v = LI.visibility(light, Pv, L, L, dist, st)
    check('a volume sample (N = L) gets a finite term in 0..1 with both '
          'lit and shadowed points',
          bool(np.isfinite(v).all()) and float(v.min()) >= 0.0
          and float(v.max()) <= 1.0 and float(v.min()) < 0.2
          and float(v.max()) > 0.8)
    # the caller's mask: only the samples it reads are looked up
    full = _vis(sc, st, P)
    mask = np.arange(n) % 3 == 0
    part = _vis(sc, st, P, mask=mask)
    check('masked-out samples report lit and the rest are the unmasked '
          'term, bitwise',
          np.array_equal(part[mask], full[mask])
          and bool((part[~mask] == 1.0).all()) and float(full.min()) < 0.5)
    stats = {}
    Nn = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    ndl = np.clip(M.dot(Nn, L), 0.0, 1.0)
    SM.lookup(light.shadow_map, SM.lamp_shape(light), P, Nn, ndl, 0.02,
              1.0, 4, mask=mask, stats=stats)
    check('and the search runs on the masked samples only',
          0 < stats.get('search', 0) <= int(mask.sum()),
          f"{stats.get('search', 0)} of {n}")
    # the worker pool pickles the scene's maps to its processes
    for kind, size in (('POINT', 0.4), ('SUN', 0.1)):
        sc, st = _prep(kind, size)
        light = sc.lights[0]
        before = _vis(sc, st, P)
        sm = light.shadow_map
        light.shadow_map = pickle.loads(pickle.dumps(
            sm, protocol=pickle.HIGHEST_PROTOCOL))
        after = _vis(sc, st, P)
        face = sm.faces[0] if hasattr(sm, 'faces') else sm
        check(f'{kind}: a pickled map keeps its axes and nearest depth and '
              'looks up the same term, bitwise',
              np.array_equal(before, after)
              and light.shadow_map.d_min == sm.d_min
              and len(face.axes) == 3)
    # a MIDPOINT map: no bias, no offset, the same lookup
    sc, st = _prep('POINT', 0.4, shadow_map_depth='MIDPOINT')
    v = _vis(sc, st, _xline(2.5))
    e = _edge(XS, v)
    check('a sized lamp on a MIDPOINT map casts a smooth penumbra too',
          e is not None and e[1] >= 12 and e[2] <= 0.08,
          'no edge' if e is None else f'{e[1]} levels, step {e[2]:.3f}')
    # a Bias of 0 (the dial's own minimum) under the plane compare: the
    # open floor on the lamp's side of the wall, which nothing shadows
    gx, gy = np.meshgrid(np.linspace(-3.0, 3.0, 61),
                         np.linspace(-3.0, -0.6, 25))
    front = np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)],
                     1).astype(np.float32)
    for kind, size in (('POINT', 0.25), ('SUN', 0.05), ('SPOT', 0.25)):
        sc, st = _prep(kind, size, shadow_bias=0.0)
        v = _vis(sc, st, front)
        check(f'{kind} size {size:g} at Bias 0: the open floor before the '
              'wall stays fully lit (no acne)',
              float(v.min()) > 0.999, f'min {float(v.min()):.4f}')
    # the buffer clip planes and a Square cone: still a soft edge
    for label, lkw in (('clip start 2', {'shadow_clip_start': 2.0}),
                       ('a Square cone', {'spot_square': True})):
        sc, st = _prep('SPOT', 0.25, lamp_kw=lkw)
        e = _edge(XS, _vis(sc, st, _xline(1.5)))
        check(f'a sized spot with {label} keeps its smooth penumbra',
              e is not None and e[1] >= 12 and 0.1 < e[0] < 0.22,
              'no edge' if e is None else f'width {e[0]:.3f}')
    # the nearest depth is the map's own
    sc, st = _prep('SPOT', 0.4)
    sm = sc.lights[0].shadow_map
    check('a map stores its nearest depth and its axes at bake',
          sm.d_min == float(sm.depth.min())
          and abs(float(np.dot(sm.axes[0], sm.axes[1]))) < 1e-5
          and abs(float(np.linalg.norm(sm.axes[2])) - 1.0) < 1e-5)


def test_n_the_tooltips_say_what_the_dials_do():
    """size-map-05 / -09: the render's Softness and Samples and the
    lamp's own Softness, in words that match the arithmetic."""
    import importlib

    from . import fakebpy
    fakebpy.install()
    try:
        import halcyon.properties as P
    except ImportError:
        P = importlib.import_module('..properties', __package__)
    importlib.reload(P)

    def tip(cls, name):
        return str(getattr(P, cls).__annotations__[name].kw
                   .get('description', '') or '')
    soft, samp = tip('HalcyonSettings', 'shadow_softness'), \
        tip('HalcyonSettings', 'shadow_samples')
    lsoft = tip('HalcyonLightSettings', 'shadow_softness')
    check("the render's Softness is described as a blur ADDED to the "
          "lamp's own penumbra, and says it needs Samples above 1",
          len(soft) >= 40 and 'size' in soft
          and 'Samples above 1' in soft, soft)
    check("the render's Samples names the 4 taps per sample of a sized "
          'lamp and the hard edge at 1', len(samp) >= 40
          and '16 to 64' in samp and 'size 0' in samp
          and '1 is a hard edge' in samp, samp)
    lsamp = tip('HalcyonLightSettings', 'shadow_samples')
    check("the lamp's own Samples says so too", '1 is a' in lsamp, lsamp)
    check("the lamp's Softness says it MULTIPLIES the render's (it never "
          'was texels)', len(lsoft) >= 40 and 'Multiplies' in lsoft
          and 'texels' not in lsoft, lsoft)


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
    print('all R254 map-soft checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
