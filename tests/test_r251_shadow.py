"""R251 shadow pack tests (1.90.0): C117 midpoint shadow map, C052 planar
projected shadows, C020 Dreamcast modifier volumes, C036 DS shadow
polygons. Run with:  python -m halcyon.tests.test_r251_shadow

The first test pins the 1.89.0 zip loudly and the pack's identity at its
defaults (render AND post, bitwise). Every check name reads as the sentence
of what it proves.
"""

import contextlib
import io
import os
import sys

import numpy as np

from ..core import lights as LI
from ..core import post as PO
from ..core import raster as CRg
from ..core import render as R
from ..core import shadowmask as SM
from ..core.scene import Light, Material
from .scenebuild import cube, demo_scene, plane, sphere, _mesh_concat
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name
          + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


W, H = 96, 72
ZIP = 'halcyon-1.89.0.zip'


# ------------------------------------------------------------------ helpers

def _post_kw(sc, st):
    return dict(frame=7, seed=st.seed, target_size=(W, H),
                allow_resize=False,
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None),
                flare_sources=getattr(sc, 'last_flares', None))


def _settings(**kw):
    """The pack's scene settings: 96x72, no AA, shadows on, no transparent
    layer road, no fog -- everything else the caller's."""
    base = dict(shadows=True, transparency='NONE', fog=False)
    base.update(kw)
    return base_settings(W, H, **base)


def _scene(st, **mods):
    sc = demo_scene(st, with_texture=False)
    for k, v in mods.items():
        setattr(sc, k, v)
    return sc


def _prev_pair(build, **kw):
    """(now, prev): this tree's frame and the 1.89.0 zip's, the CURRENT
    settings handed to the old engine (it ignores unknown fields)."""
    RP = _prev_engine(ZIP)
    st = _settings(**kw)
    sc = build(st)
    now = np.asarray(R.render(sc, st))
    st2 = _settings(**kw)
    sc2 = build(st2)
    prev = np.asarray(RP.render(sc2, st2))
    return now, prev


def _same(a, b):
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _maxdiff(a, b):
    return f'max {float(np.abs(a - b).max()):.6g}' if a.shape == b.shape \
        else 'shape'


def _volume(centre, size, role, color=(0.0, 0.0, 0.0), alpha=16,
            polygon_id=1, name='Volume'):
    """An authored volume from scenebuild's cube() 6-tuple."""
    v, _n, _uv, t, _m, _o = cube(centre=centre, size=size)
    return {'name': name, 'verts': v, 'tris': t, 'role': role,
            'polygon_id': int(polygon_id), 'alpha': int(alpha),
            'color': tuple(color)}


def _gbuffer_for(sc, st):
    """The Gouraud rig's raster + job (TR:20084-20091)."""
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, None, view, eye, W, H)
    return g, job, vp, view, eye


def _bake_pack(sc, st, g, vp, view, eye):
    """The mask bake exactly as render() calls it (the rig never sets the
    pack on `st`, review S7): the test calls SM.build itself."""
    snap = R.snap_grid(st)
    near_eps = max(float(getattr(st, 'clip_near_epsilon', 1e-5) or 1e-5),
                   1e-6)
    pl = st.shadows and (st.shadow_default == 'PLANAR' or (
        st.shadow_default == 'PER_LIGHT' and any(
            getattr(l, 'shadow', 'MAP') == 'PLANAR' for l in sc.lights)))
    kt = R._caster_keep_tri(sc, sc.mesh) if pl else None
    st._mask_pack, st._mask_params = SM.build(
        sc, st, g, vp, snap, near_eps,
        None if kt is None else np.nonzero(kt)[0], scissor=None,
        view=view, eye=eye)
    return st._mask_pack


def _capture(fn):
    """Run fn() with stdout captured; returns (result, printed text)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn()
    return out, buf.getvalue()


def _surface_z(sc, g):
    """Every covered pixel's world Z from the G-buffer (float32, the
    bake's own fixed order)."""
    verts = np.asarray(sc.mesh.verts, np.float32)
    ti = np.clip(g.tri, 0, sc.mesh.tris.shape[0] - 1)
    t3 = sc.mesh.tris[ti]
    b = g.bary
    Pz = verts[t3[..., 0], 2] * b[..., 0]
    Pz = Pz + verts[t3[..., 1], 2] * b[..., 1]
    Pz = Pz + verts[t3[..., 2], 2] * b[..., 2]
    return Pz


def _gbuf_of_render(sc, st):
    """The G-buffer render() shaded (its raster, at the frame's own
    parameters), for reading which object owns a pixel."""
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  snap=R.snap_grid(st), depth_bits=st.depth_precision,
                  cull='BACK' if st.backface_cull else 'NONE')
    return g


# =================================================================== FIRST

def test_a_identity_at_defaults():
    """The zip pin, loudly, then the whole pack at its defaults: render AND
    post bitwise the 1.89.0 engine on the demo scene."""
    RP = _prev_engine(ZIP)
    check('the 1.89.0 zip is beside the package', RP is not None)
    if RP is None:
        return
    import importlib
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0]
                                        + '.post')
    for label, kw in (('shadows on', {}),
                      ('shadows off', dict(shadows=False)),
                      ('PER_LIGHT maps', dict(shadow_default='PER_LIGHT')),
                      ('VERTEX rate', dict(shading_rate='VERTEX'))):
        st = _settings(**kw)
        sc = _scene(st)
        img = R.render(sc, st)
        now = PO.process(img, st, **_post_kw(sc, st))
        st2 = _settings(**kw)
        sc2 = _scene(st2)
        pimg = RP.render(sc2, st2)
        prev = prev_post.process(pimg, st2, **_post_kw(sc2, st2))
        check(f'at the defaults ({label}) the render is bitwise 1.89.0',
              _same(np.asarray(img), np.asarray(pimg)),
              _maxdiff(np.asarray(img), np.asarray(pimg)))
        check(f'at the defaults ({label}) render + post is bitwise 1.89.0',
              _same(now, prev), _maxdiff(now, prev))
    st = _settings()
    check('the new settings fields exist at their neutral defaults',
          st.shadow_map_depth == 'CLASSIC' and st.planar_plane_z == 0.0
          and int(st.modvol_scale) == 128)
    check('a demo lamp inherits the map depth and a material is a surface',
          all(getattr(l, 'shadow_map_depth', None) == 'INHERIT'
              for l in _scene(st).lights)
          and Material().volume_role == 'NONE'
          and Material().polygon_id == 0 and Material().shadow_alpha == 16)
    R.render(_scene(st), st)
    check('no pack is baked when no planar lamp and no volume exists',
          getattr(st, '_mask_pack', 'unset') is None
          and getattr(st, '_mask_params', 'unset') is None)


# ==================================================================== C117

def test_c117_midpoint_identity_at_defaults():
    """CLASSIC explicit and by default, every lamp INHERIT, and a lamp set
    to CLASSIC under a CLASSIC scene: all bitwise 1.89.0."""
    RP = _prev_engine(ZIP)
    if RP is None:
        return
    now, prev = _prev_pair(_scene, shadow_map_depth='CLASSIC')
    check('CLASSIC map depth said explicitly is bitwise 1.89.0',
          _same(now, prev), _maxdiff(now, prev))

    def lamp_classic(st):
        sc = _scene(st)
        sc.lights[0].shadow_map_depth = 'CLASSIC'
        return sc
    now, prev = _prev_pair(lamp_classic)
    check("a lamp's explicit CLASSIC under a CLASSIC scene is bitwise 1.89.0",
          _same(now, prev), _maxdiff(now, prev))


def _sun_map(st, sc):
    """Build the maps; return (sun map, point cube)."""
    LI.clear_shadow_cache()
    LI.build_shadow_maps(sc, st)
    return sc.lights[0].shadow_map, sc.lights[1].shadow_map


def test_c117_midpoint_bake_stores_the_halfway_texel():
    """On the SUN map: z1 of the MIDPOINT bake equals CLASSIC's texel;
    where a second surface exists the stored depth is the inclusive
    halfway point of the two linearised depths; lone-surface and empty
    texels store CLASSIC's empty texel; no second surface lies within the
    A-buffer tolerance of the first (the named tie rule)."""
    st_c = _settings(shadow_map_depth='CLASSIC')
    sc_c = _scene(st_c)
    sm_c, _cube_c = _sun_map(st_c, sc_c)
    st_m = _settings(shadow_map_depth='MIDPOINT')
    sc_m = _scene(st_m)
    sm_m, _cube_m = _sun_map(st_m, sc_m)
    check('the SUN map is a ShadowMap on both settings',
          isinstance(sm_c, LI.ShadowMap) and isinstance(sm_m, LI.ShadowMap))
    if not (isinstance(sm_c, LI.ShadowMap) and isinstance(sm_m, LI.ShadowMap)):
        return
    # the raw rasters on the SUN's own matrix
    tris = sc_m.mesh.tris
    verts = sc_m.mesh.verts
    z_cls = LI._render_depth(verts, tris, sm_m.vp, sm_m.size)
    z1, z2, has2 = LI._render_depth(verts, tris, sm_m.vp, sm_m.size,
                                    midpoint=True)
    check('z1 of the MIDPOINT bake is bitwise the CLASSIC texel',
          np.array_equal(z1, z_cls))
    d1 = sm_m._linearise(z1)
    d2 = sm_m._linearise(z2)
    dm = sm_m.depth
    lo = d1[has2] <= dm[has2]
    hi = dm[has2] <= d2[has2]
    check('where a second surface exists the texel is between the two '
          'linearised depths, both bounds inclusive',
          bool(lo.all()) and bool(hi.all()) and int(has2.sum()) > 0,
          f'{int(has2.sum())} texels with a second surface')
    empty = sm_m._linearise(np.ones(1, np.float32))[0]
    lone = (~has2) & (z1 < 1.0)
    check('a lone-surface texel stores the far plane (CLASSIC\'s own empty '
          'texel, mental ray\'s rule)',
          int(lone.sum()) > 0 and bool((dm[lone] == empty).all()),
          f'{int(lone.sum())} lone texels')
    none = z1 >= 1.0
    check('on empty texels the MIDPOINT and CLASSIC maps agree bitwise',
          int(none.sum()) > 0 and np.array_equal(dm[none], sm_c.depth[none]))
    check('the two maps differ somewhere (the bake moves)',
          not np.array_equal(dm, sm_c.depth))
    lim = CRg.abuf_depth_limit(z1)
    check('no second surface lies within the A-buffer tolerance of the '
          'first (a shared-edge double fragment is one surface)',
          bool((z2[has2] > lim[has2]).all()))
    check('the halfway point is the exact float32 sum of two halves',
          np.array_equal(dm[has2], (d1[has2] * np.float32(0.5)
                                    + d2[has2] * np.float32(0.5))))
    # the run size cannot move a texel (order-free by construction)
    old = LI.MIDPOINT_RUN
    try:
        LI.MIDPOINT_RUN = 7
        z1b, z2b, has2b = LI._render_depth(verts, tris, sm_m.vp, sm_m.size,
                                           midpoint=True)
    finally:
        LI.MIDPOINT_RUN = old
    check('the second raster in runs of 7 triangles lands the same texels '
          'as one run (a pure function of the caster set)',
          np.array_equal(z1, z1b) and np.array_equal(z2, z2b)
          and np.array_equal(has2, has2b))
    # the cube shadow builds the same way
    check('a POINT lamp\'s cube faces are midpoint maps too and differ '
          'from the CLASSIC cube',
          isinstance(_cube_m, LI.CubeShadow) and any(
              not np.array_equal(a.depth, b.depth)
              for a, b in zip(_cube_m.faces, _cube_c.faces)))


def test_c117_midpoint_kills_acne_without_bias():
    """The mechanism's law: under MIDPOINT with bias 0, one tap and no
    blur, every lit-hemisphere pixel of the ball looks up as lit with a
    ZERO offset (the midpoint sits inside the ball); the floor under the
    ball still darkens; the picture differs from CLASSIC."""
    def scene(st):
        sc = _scene(st)
        sc.lights[1].shadow = 'NONE'
        return sc
    kw = dict(shadow_bias=0.0, shadow_softness=0.0, shadow_samples=1,
              force_model='LAMBERT')
    st_m = _settings(shadow_map_depth='MIDPOINT', **kw)
    sc_m = scene(st_m)
    mid = R.render(sc_m, st_m)
    st_c = _settings(shadow_map_depth='CLASSIC', shadow_bias=0.02,
                     shadow_softness=0.0, shadow_samples=1,
                     force_model='LAMBERT')
    sc_c = scene(st_c)
    cls = R.render(sc_c, st_c)
    st_n = _settings(shadows=False, **kw)
    sc_n = scene(st_n)
    non = R.render(sc_n, st_n)
    check('MIDPOINT differs from CLASSIC with its default bias (the '
          'picture moves)', not _same(mid, cls))
    sm = sc_m.lights[0].shadow_map
    g = _gbuf_of_render(sc_m, st_m)
    cov = g.mask()
    obj = sc_m.mesh.obj_index[np.clip(g.tri, 0, None)]
    ball = cov & (obj == 1)
    verts = np.asarray(sc_m.mesh.verts, np.float32)
    t3 = sc_m.mesh.tris[np.clip(g.tri, 0, None)]
    b = g.bary
    P = (verts[t3[..., 0]] * b[..., 0:1] + verts[t3[..., 1]] * b[..., 1:2]
         + verts[t3[..., 2]] * b[..., 2:3]).astype(np.float32)
    centre = np.array([-1.3, 0.2, 1.0], np.float32)
    N = P - centre[None, None, :]
    N = N / np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-9)
    L = -np.asarray(sc_m.lights[0].direction, np.float32)
    L = L / np.linalg.norm(L)
    ndl = (N * L[None, None, :]).sum(axis=2)
    lit_hemi = ball & (ndl > 0.3)
    # the pixels whose texel's NEAREST surface is the ball itself (the
    # demo's box stands between the sun and part of the ball: a real
    # cast shadow, measured at 31 px under CLASSIC's default too)
    ph = np.concatenate([P, np.ones(P.shape[:2] + (1,), np.float32)], axis=2)
    clip = ph @ sm.vp.T
    ndc = clip[..., :3] / np.where(np.abs(clip[..., 3:4]) < 1e-9, 1e-9,
                                   clip[..., 3:4])
    dist = sm._linearise(ndc[..., 2])
    z1, _z2, _h2 = LI._render_depth(verts, sc_m.mesh.tris, sm.vp, sm.size,
                                    midpoint=True)
    xi = np.clip(np.floor((ndc[..., 0] * 0.5 + 0.5) * sm.size).astype(int),
                 0, sm.size - 1)
    yi = np.clip(np.floor((ndc[..., 1] * 0.5 + 0.5) * sm.size).astype(int),
                 0, sm.size - 1)
    d1 = sm._linearise(z1)[yi, xi]
    # P IS the front surface along its own texel's ray: the texel's
    # nearest depth sits within the texel-slope error of dist (0.1 at
    # N.L = 0.3 with 0.032 texels); the box stands 1.8 units nearer
    own_tex = (dist - d1) < 0.5
    py, px = np.nonzero(lit_hemi & own_tex)
    vis = sm.lookup(P[py, px], 0.0, 0.0, 1)
    check('under MIDPOINT every lit-hemisphere ball pixel whose texel sees '
          'the ball first looks up as lit with zero bias and zero offset '
          '(the midpoint is inside the ball)',
          py.size > 50 and bool((vis == 1.0).all()),
          f'{int((vis < 1.0).sum())} of {py.size} px in shadow')
    # CLASSIC at bias 0 with no offset: informational only (its normal
    # offset may already hide the acne on the demo ball)
    st_c0 = _settings(shadow_map_depth='CLASSIC', **kw)
    sc_c0 = scene(st_c0)
    R.render(sc_c0, st_c0)
    vis_c = sc_c0.lights[0].shadow_map.lookup(P[py, px], 0.0, 0.0, 1)
    print(f'         (extra) CLASSIC at bias 0, no offset: '
          f'{int((vis_c < 1.0).sum())} of {py.size} lit-hemisphere ball '
          'px self-shadow')
    floor = cov & (obj == 0)
    # the floor region under the ball: floor pixels the SUN's polygon
    # would darken -- read as the pixels where the shadowed frame is
    # darker than the unshadowed one
    dark = floor & (mid[..., :3].sum(axis=2) < non[..., :3].sum(axis=2))
    check('the floor under the ball still darkens under MIDPOINT',
          int(dark.sum()) > 20, f'{int(dark.sum())} darker floor px')
    dark_ball = ball & (mid[..., :3].sum(axis=2)
                        < non[..., :3].sum(axis=2) - 1e-6)
    foreign = (dist - d1) > 1.0
    check('every ball pixel MIDPOINT darkens has a foreign caster at least '
          "one unit nearer the sun (the box's shadow, never acne)",
          int(dark_ball.sum()) > 0 and bool(foreign[dark_ball].all()),
          f'{int(dark_ball.sum())} ball px darker, '
          f'{int((~foreign & dark_ball).sum())} without a foreign caster')


def test_c117_midpoint_per_lamp_override():
    """A CLASSIC scene whose SUN says MIDPOINT: its map equals the
    all-MIDPOINT scene's SUN map, the POINT's cube equals the all-CLASSIC
    cube, the frame differs from both pure scenes, and the lamp signature
    differs between INHERIT and MIDPOINT."""
    st_mix = _settings(shadow_map_depth='CLASSIC')
    sc_mix = _scene(st_mix)
    sc_mix.lights[0].shadow_map_depth = 'MIDPOINT'
    mix = R.render(sc_mix, st_mix)
    st_m = _settings(shadow_map_depth='MIDPOINT')
    sc_m = _scene(st_m)
    allm = R.render(sc_m, st_m)
    st_c = _settings(shadow_map_depth='CLASSIC')
    sc_c = _scene(st_c)
    allc = R.render(sc_c, st_c)
    check("the overridden SUN's map is bitwise the all-MIDPOINT SUN map",
          np.array_equal(sc_mix.lights[0].shadow_map.depth,
                         sc_m.lights[0].shadow_map.depth))
    check("the inheriting POINT's cube is bitwise the all-CLASSIC cube",
          all(np.array_equal(a.depth, b.depth) for a, b in
              zip(sc_mix.lights[1].shadow_map.faces,
                  sc_c.lights[1].shadow_map.faces)))
    check('the mixed frame differs from both pure scenes',
          not _same(mix, allm) and not _same(mix, allc))
    a = Light(type='SUN', name='a')
    b = Light(type='SUN', name='a', shadow_map_depth='MIDPOINT')
    check('the per-lamp cache signature differs between INHERIT and '
          'MIDPOINT (the cache cannot serve across the override)',
          LI._light_shadow_signature(a) != LI._light_shadow_signature(b))
    check('the base signature differs between CLASSIC and MIDPOINT',
          LI._shadow_base_signature(sc_c, st_c, None)
          != LI._shadow_base_signature(sc_m, st_m, None))
    check('midpoint_map resolves INHERIT to the render setting and the '
          'lamp\'s own value over it',
          LI.midpoint_map(a, st_m) and not LI.midpoint_map(a, st_c)
          and LI.midpoint_map(b, st_c)
          and not LI.midpoint_map(Light(shadow_map_depth='CLASSIC'), st_m))


def test_c117_midpoint_gpu_twin():
    """The Gouraud rig at PIXEL rate under MIDPOINT: the plan qualifies,
    the frame matches the CPU at the map's own bar, the emitted SUN
    function carries `float off_amt = 0.0;` under MIDPOINT and not under
    CLASSIC (and under the per-lamp mix only the overridden lamp's does),
    and the gate is in the plan signature by name."""
    import inspect
    from ..gpu import shade as GSH

    def run(depth, lamp0=None):
        st = _settings(shadow_map_depth=depth, shading_rate='PIXEL')
        sc = _scene(st)
        if lamp0 is not None:
            sc.lights[0].shadow_map_depth = lamp0
        cpu = R.render(sc, st)
        g, job, _vp, _v, _e = _gbuffer_for(sc, st)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        return st, sc, cpu, g, job, p, why

    st, sc, cpu, g, job, p, why = run('MIDPOINT')
    check('a MIDPOINT frame qualifies for the deferred pass', p is not None,
          str(why))
    if p is not None:
        out, hit = GSH.simulate(job, g)
        check('the MIDPOINT passes simulate', out is not None, str(hit))
        if out is not None:
            cov = g.tri >= 0
            check('the simulated frame covers exactly the raster\'s pixels',
                  np.array_equal(hit, cov))
            yy, xx = np.nonzero(cov)
            d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
            check('the MIDPOINT frame matches the CPU per pixel at the '
                  "map's own bar (6e-3)",
                  int((d > 1e-2).sum()) == 0 and float(d.max()) < 6e-3,
                  f'max {float(d.max()):.6f}')
        srcs = '\n'.join(s for _mi, _n, s, _b in p)
        check('the emitted SUN function has a zero offset under MIDPOINT',
              'float off_amt = 0.0;' in srcs)
    sig_m = GSH._plan_sig(job, None)
    st_c, sc_c, cpu_c, g_c, job_c, p_c, why_c = run('CLASSIC')
    if p_c is not None:
        srcs_c = '\n'.join(s for _mi, _n, s, _b in p_c)
        check('no zero-offset line under CLASSIC',
              'float off_amt = 0.0;' not in srcs_c)
    sig_c = GSH._plan_sig(job_c, None)
    check('a CLASSIC plan and a MIDPOINT plan have different signatures',
          sig_m != sig_c)
    check("'shadow_map_depth' is in the plan signature by name",
          'shadow_map_depth' in inspect.getsource(GSH._plan_sig))
    check("the per-lamp override is in _light_sig by name",
          'shadow_map_depth' in inspect.getsource(GSH._light_sig))
    # the per-lamp mix: only lamp 0's function carries the zero offset
    st_x, sc_x, cpu_x, g_x, job_x, p_x, why_x = run('CLASSIC', 'MIDPOINT')
    check('the mixed frame qualifies', p_x is not None, str(why_x))
    if p_x is not None:
        src = max((s for _mi, _n, s, _b in p_x), key=len)
        i0 = src.find('float hal_shadow_vis0(')
        i1 = src.find('float hal_shadow_vis1(')
        f0 = src[i0:i1] if 0 <= i0 < i1 else ''
        f1 = src[i1:] if i1 >= 0 else ''
        check('under the per-lamp mix only the overridden lamp\'s function '
              'has the zero offset',
              'float off_amt = 0.0;' in f0 and
              'float off_amt = 0.0;' not in f1.split('void main')[0],
              f'i0={i0} i1={i1}')
        out_x, _h = GSH.simulate(job_x, g_x)
        if out_x is not None:
            yy, xx = np.nonzero(g_x.tri >= 0)
            d = np.abs(out_x[yy, xx] - cpu_x[yy, xx, :3]).max(axis=1)
            check('the mixed frame matches the CPU per pixel at the bar',
                  float(d.max()) < 6e-3, f'max {float(d.max()):.6f}')
    check('a MIDPOINT frame refuses nothing by name (same atlas, same '
          'compare)', p is not None and why is None)


def test_c117_midpoint_device_fallback():
    """Without a driver the GPU device lands on the same CPU code: the
    MIDPOINT frame is bitwise on both devices (the fallback plumbing,
    TR:17670-17676's sense)."""
    st = _settings(shadow_map_depth='MIDPOINT')
    sc = _scene(st)
    cpu = R.render(sc, st)
    st2 = _settings(shadow_map_depth='MIDPOINT')
    st2.render_device = 'GPU'
    sc2 = _scene(st2)
    gpu = R.render(sc2, st2)
    check('the MIDPOINT frame falls back bit-exactly on the GPU device '
          'without a driver', _same(cpu, gpu), _maxdiff(cpu, gpu))


# ==================================================================== C052

def _planar_scene(st, sun_only=True, **lamp):
    """The demo under one PLANAR SUN (the POINT lamp casting none)."""
    sc = _scene(st)
    if sun_only:
        sc.lights[1].shadow = 'NONE'
    for k, v in lamp.items():
        setattr(sc.lights[0], k, v)
    return sc


def _changed(a, b):
    return np.abs(np.asarray(a)[..., :3] - np.asarray(b)[..., :3]).max(axis=2) > 0.0


def test_c052_planar_identity_at_defaults():
    """The field is inert outside PLANAR: `planar_plane_z = 3.0` under MAP
    shadows is bitwise 1.89.0, and so is the pack's neutral default."""
    RP = _prev_engine(ZIP)
    if RP is None:
        return
    now, prev = _prev_pair(_scene, planar_plane_z=3.0, shadow_default='MAP')
    check('planar_plane_z = 3.0 under MAP shadows is bitwise 1.89.0 (the '
          'field is inert outside PLANAR)', _same(now, prev),
          _maxdiff(now, prev))
    now, prev = _prev_pair(_scene, shadow_default='PER_LIGHT',
                           planar_plane_z=-2.0)
    check('planar_plane_z under PER_LIGHT with no PLANAR lamp is bitwise '
          '1.89.0', _same(now, prev), _maxdiff(now, prev))


def test_c052_planar_shadows_land_on_the_plane():
    """The floor receives the SUN's projected polygon: the picture moves,
    every changed pixel is on the plane by the tie rule, the colour and
    density are the polygon's, density is monotonic, a point lamp
    projects a footprint, a lamp below the plane refuses by name, the
    ray tracer never traces the polygon, the plane's own A/B, bands,
    the shadeless floor and the CLIP-decal promotion."""
    for rate in ('PIXEL', 'VERTEX'):
        st = _settings(shadow_default='PLANAR', shading_rate=rate)
        sc = _planar_scene(st)
        on = R.render(sc, st)
        pack = st._mask_pack
        st0 = _settings(shadows=False, shading_rate=rate)
        sc0 = _planar_scene(st0)
        off = R.render(sc0, st0)
        ch = _changed(on, off)
        check(f'at {rate} rate the planar shadow moves the picture',
              int(ch.sum()) > 20, f'{int(ch.sum())} px')
        g = _gbuf_of_render(sc, st)
        obj = sc.mesh.obj_index[np.clip(g.tri, 0, None)]
        _c, radius = LI.scene_bounds(np.asarray(sc.mesh.verts, np.float32))
        eps = np.float32(1e-4) * np.float32(max(1.0, float(radius)))
        Pz = _surface_z(sc, g)
        on_plane = (obj == 0) | (np.abs(Pz - np.float32(0.0)) <= eps)
        check(f'at {rate} rate every changed pixel is on the floor or on '
              'the plane by the tie rule |Pz - z0| <= eps',
              bool(on_plane[ch].all()),
              f'{int((~on_plane & ch).sum())} off-plane px changed')
        check(f'at {rate} rate the pack marks exactly the changed pixels',
              pack is not None and np.array_equal(
                  (pack[0][..., 0] > 0) & (pack[1] >= 0), ch))
    # (c) the polygon's colour is exact under CONSTANT at density 1
    st = _settings(shadow_default='PLANAR', force_model='CONSTANT')
    sc = _planar_scene(st, shadow_color=(1.0, 0.0, 0.0), shadow_density=1.0)
    img = R.render(sc, st)
    m = (st._mask_pack[0][..., 0] > 0) & (st._mask_pack[1] >= 0)
    check('with shadow colour (1, 0, 0) at density 1 every masked pixel is '
          'exactly (1, 0, 0) before post',
          int(m.sum()) > 20 and bool((img[m][:, :3]
                                      == np.array([1, 0, 0], np.float32)).all()))
    # (d) density monotonic; 0.0 bitwise the shadows-off frame
    means = []
    frames = {}
    for dens in (0.0, 0.5, 1.0):
        st = _settings(shadow_default='PLANAR')
        sc = _planar_scene(st, shadow_density=dens)
        frames[dens] = R.render(sc, st)
        m = (st._mask_pack[0][..., 0] > 0) & (st._mask_pack[1] >= 0)
        means.append(float(frames[dens][m][:, :3].mean()))
    st0 = _settings(shadows=False)
    off = R.render(_planar_scene(st0), st0)
    check('density 0.0 renders bitwise the shadows-off frame',
          _same(frames[0.0], off), _maxdiff(frames[0.0], off))
    check('the shadowed region darkens monotonically through density '
          '0.0, 0.5, 1.0', means[0] > means[1] > means[2],
          str([round(v, 4) for v in means]))
    # (e) a POINT lamp above the ball: a footprint around the contact
    st = _settings(shadow_default='PLANAR')
    sc = _planar_scene(st, sun_only=False)
    sc.lights[0].shadow = 'NONE'
    sc.lights[1].position = (-1.3, 0.2, 8.0)
    sc.materials[2].cast_shadow = False       # the ball's disc alone
    R.render(sc, st)
    m = (st._mask_pack[0][..., 0] > 0) & (st._mask_pack[1] >= 0)
    g = _gbuf_of_render(sc, st)
    verts = np.asarray(sc.mesh.verts, np.float32)
    t3 = sc.mesh.tris[np.clip(g.tri, 0, None)]
    b = g.bary
    P = (verts[t3[..., 0]] * b[..., 0:1] + verts[t3[..., 1]] * b[..., 1:2]
         + verts[t3[..., 2]] * b[..., 2:3])
    rxy = np.linalg.norm(P[m][:, :2] - np.array([-1.3, 0.2]), axis=1)
    check('a POINT lamp straight above the ball projects a footprint that '
          'lies within the disc the point projection predicts (r = 8/7 '
          'plus a pixel)',
          int(m.sum()) > 10 and float(rxy.max()) < 8.0 / 7.0 + 0.25,
          f'{int(m.sum())} px, max r {float(rxy.max()) if m.any() else 0:.3f}')
    check("the POINT lamp's bit is lamp index 1 (the original index, not "
          'the selection order)',
          bool((st._mask_pack[0][..., 0][m] == 2.0).all())
          and st._mask_params['planar'][0][0] == 1)
    # (f) a lamp below the plane: named, and bitwise the lamp-off frame
    st = _settings(shadow_default='PLANAR')
    sc = _planar_scene(st, sun_only=False)
    sc.lights[0].shadow = 'NONE'
    sc.lights[1].position = (3.5, -3.0, -1.0)
    below, text = _capture(lambda: R.render(sc, st))
    check('a lamp below the plane prints the below-plane line once',
          text.count('sits below the plane Z=0') == 1, repr(text[-160:]))
    st2 = _settings(shadow_default='PLANAR')
    sc2 = _planar_scene(st2, sun_only=False)
    sc2.lights[0].shadow = 'NONE'
    sc2.lights[1].shadow = 'NONE'
    sc2.lights[1].position = (3.5, -3.0, -1.0)
    check('and renders bitwise the frame with that lamp casting nothing '
          '(a named empty)', _same(below, R.render(sc2, st2)))
    # (g) the ray tracer never traces the map-less PLANAR lamp
    st = _settings(shadow_default='PLANAR', raytrace=True, ray_depth=1)
    sc = _planar_scene(st)
    rt = R.render(sc, st)
    st2 = _settings(shadow_default='PLANAR', raytrace=False)
    sc2 = _planar_scene(st2)
    check('with ray tracing on the PLANAR frame is bitwise the untraced '
          'one (the map-less lamp traces no shadow rays)',
          _same(rt, R.render(sc2, st2)))
    # (h) the plane's own A/B
    st = _settings(shadow_default='PLANAR', planar_plane_z=0.5)
    sc = _planar_scene(st)
    raised = R.render(sc, st)
    check('planar_plane_z = 0.5 differs from 0.0', not _same(raised, frames[1.0]))
    check('and is bitwise the shadows-off frame (no mesh lies on Z = 0.5: '
          'the honest empty)', _same(raised, off), _maxdiff(raised, off))
    # (i) band invariance
    st = _settings(shadow_default='PLANAR')
    whole = np.asarray(R.render(_planar_scene(st), st))
    parts = []
    for bd in ((0, 36), (36, 72)):
        stb = _settings(shadow_default='PLANAR')
        parts.append(np.asarray(R.render(_planar_scene(stb), stb, band=bd)))
    check('the planar shadow is band-invariant, bitwise',
          np.array_equal(np.concatenate(parts, 0), whole))
    # (j) the shadeless floor darkens at PIXEL exactly as at VERTEX rate
    sets = {}
    for rate in ('PIXEL', 'VERTEX'):
        st = _settings(shadow_default='PLANAR', shading_rate=rate)
        sc = _planar_scene(st)
        sc.materials[0].shadeless = True
        on = R.render(sc, st)
        st0 = _settings(shadows=False, shading_rate=rate)
        sc0 = _planar_scene(st0)
        sc0.materials[0].shadeless = True
        sets[rate] = _changed(on, R.render(sc0, st0))
    check('a shadeless floor takes the polygon at PIXEL rate exactly as at '
          'VERTEX rate (the same changed pixels)',
          int(sets['PIXEL'].sum()) > 20
          and np.array_equal(sets['PIXEL'], sets['VERTEX']),
          f"{int(sets['PIXEL'].sum())} vs {int(sets['VERTEX'].sum())} px")
    # (k) the promotion: a CLIP decal on the plane under the polygon
    st = _settings(shadow_default='PLANAR')
    sc = _planar_scene(st)
    R.render(sc, st)
    m = (st._mask_pack[0][..., 0] > 0) & (st._mask_pack[1] >= 0)
    g = _gbuf_of_render(sc, st)
    yy, xx = np.nonzero(m)
    P = (verts[t3[..., 0]] * b[..., 0:1] + verts[t3[..., 1]] * b[..., 1:2]
         + verts[t3[..., 2]] * b[..., 2:3])
    cx, cy = float(np.median(P[yy, xx, 0])), float(np.median(P[yy, xx, 1]))

    def decal_scene(st):
        sc = _planar_scene(st)
        v, n, uv, t, _m, _o = plane(z=5e-4, size=0.8, mat=3, obj=3)
        v = v + np.array([cx, cy, 0.0], np.float32)
        parts = [plane(z=0.0, size=11.0, mat=0, obj=0),
                 sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
                 cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
                 (v, n, uv, t, 3, 3)]
        mesh = _mesh_concat(parts)
        smooth = np.zeros(mesh.tris.shape[0], bool)
        smooth[mesh.mat_index == 1] = True
        mesh.smooth = smooth
        sc.mesh = mesh
        sc.materials.append(Material(name='Decal', index=3, model='LAMBERT',
                                     diffuse=(0.9, 0.9, 0.2),
                                     alpha_mode='CLIP', alpha_clip=0.5,
                                     opacity=0.9))
        from ..core.scene import ObjectInfo
        sc.objects.append(ObjectInfo(name='Decal', index=3,
                                     matrix_world=np.eye(4, dtype=np.float32)))
        return sc
    st = _settings(shadow_default='PLANAR', transparency='SORTED')
    sc = decal_scene(st)
    on = R.render(sc, st)
    ptri = st._mask_pack[1]
    dec = (ptri >= 0) & (sc.mesh.obj_index[np.clip(ptri, 0, None)] == 3)
    st0 = _settings(shadows=False, transparency='SORTED')
    off = R.render(decal_scene(st0), st0)
    ch = _changed(on, off)
    check('a promoted CLIP decal lying on the plane is the opaque winner '
          'the bake saw (its pixels are in the pack)',
          int(dec.sum()) > 4, f'{int(dec.sum())} decal px')
    check('and it is shadowed exactly where the floor under it was (the '
          'receiver test ran on the promoted triangle, after the '
          'punch-through)',
          int((dec & m).sum()) > 4 and np.array_equal(ch[dec], m[dec]),
          f'{int((dec & m).sum())} decal px under the polygon, '
          f'{int((ch != m)[dec].sum())} disagree')


def test_c052_planar_gpu_twin():
    """The Gouraud rig: the test bakes the pack itself; under CONSTANT at
    PIXEL and VERTEX rate (and with the floor shadeless) the readback
    apply lands d == 0.0 on every covered pixel; on the lit demo the
    TR:8157 bar; the readback edit is live; nothing is emitted."""
    from ..gpu import shade as GSH

    def twin(label, rate, shadeless=False, model='CONSTANT', bar=0.0,
             refuses=None):
        kw = dict(shadow_default='PLANAR', shading_rate=rate)
        if model is not None:
            kw['force_model'] = model
        st = _settings(**kw)
        sc = _planar_scene(st, shadow_color=(1.0, 0.0, 0.0),
                           shadow_density=1.0)
        if shadeless:
            sc.materials[0].shadeless = True
        cpu = R.render(sc, st)
        g, job, vp, view, eye = _gbuffer_for(sc, st)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        if refuses is not None:
            # measured: this tree's planner refuses a shadeless material
            # BY NAME ('shadeless materials take the early CPU path'), so
            # the shadeless road is the CPU's on both devices
            check(f'{label}: the plan refuses by name ({refuses})',
                  p is None and refuses in str(why), str(why))
            st2 = _settings(**kw)
            st2.render_device = 'GPU'
            sc2 = _planar_scene(st2, shadow_color=(1.0, 0.0, 0.0),
                                shadow_density=1.0)
            sc2.materials[0].shadeless = True
            gpu = R.render(sc2, st2)
            check(f'{label}: the GPU device lands on the CPU road bitwise',
                  _same(cpu, gpu), _maxdiff(cpu, gpu))
            return None
        check(f'{label}: the frame qualifies for the deferred pass',
              p is not None, str(why))
        if p is None:
            return None
        raw, hit = GSH.simulate(job, g)
        if raw is None:
            check(f'{label}: the passes simulate', False, str(hit))
            return None
        pack = _bake_pack(sc, st, g, vp, view, eye)
        check(f'{label}: the bake finds the planar lamp',
              pack is not None and st._mask_params['planar'])
        out = SM.apply_frame(raw, g, hit, st)
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
        mm = (pack[0][yy, xx, 0] > 0)
        check(f'{label}: on every masked pixel the readback apply is '
              'bitwise the CPU frame (d == 0.0)',
              int(mm.sum()) > 20 and float(d[mm].max()) == 0.0,
              f'{int(mm.sum())} px, max {float(d[mm].max()):.6g}')
        if bar == 0.0:
            check(f'{label}: the readback apply is bitwise the CPU frame '
                  '(d == 0.0 on every covered pixel)', float(d.max()) == 0.0,
                  f'max {float(d.max()):.6g}')
        else:
            check(f'{label}: the frame matches the CPU at the TR:8157 bar',
                  float(d.max()) < bar and np.array_equal(hit, g.tri >= 0),
                  f'max {float(d.max()):.6f}')
        srcs = '\n'.join(s for _mi, _n, s, _b in p)
        check(f'{label}: no pass source carries a mask sampler (nothing is '
              'emitted this round)', 'hal_maskpack' not in srcs
              and 'hal_mp' not in srcs)
        check(f'{label}: the readback edit is live (the masked frame '
              'differs from the raw simulation)',
              not np.array_equal(out, raw))
        return out

    twin('CONSTANT at PIXEL rate', 'PIXEL')
    # measured: at VERTEX rate the pass's corner product and the CPU's
    # `light * alb` differ by one float32 ULP (1.19e-7) on unmasked
    # pixels before the mask -- the existing rig's bar; the polygon's
    # pixels are exact (checked above)
    twin('CONSTANT at VERTEX rate', 'VERTEX', bar=6e-3)
    twin('CONSTANT, shadeless floor', 'PIXEL', shadeless=True,
         refuses='shadeless')
    twin('the lit demo at PIXEL rate', 'PIXEL', model=None, bar=6e-3)


def test_c052_planar_refusal_by_name():
    """25 PLANAR suns with no light limit: the 24-lamp line prints once and
    the two devices agree bitwise (a fallback check, no driver); with the
    Light Limit at one FIRST lamp only lamp 0 casts."""
    def many(st):
        sc = _scene(st)
        sc.lights = [Light(type='SUN', name=f'sun{i}',
                           direction=(-0.62 + 0.01 * i, 0.45, -0.45),
                           energy=0.3, shadow='MAP')
                     for i in range(25)]
        return sc
    st = _settings(shadow_default='PLANAR', max_lights=0)
    sc = many(st)
    cpu, text = _capture(lambda: R.render(sc, st))
    check('25 PLANAR lamps print the 24-lamp line once, naming the lamp',
          text.count('planar shadows: 24 lamps per frame') == 1
          and "lamp 'sun24'" in text, repr(text[-200:]))
    check('the pack carries exactly 24 planar entries',
          len(st._mask_params['planar']) == 24)
    st2 = _settings(shadow_default='PLANAR', max_lights=0)
    st2.render_device = 'GPU'
    sc2 = many(st2)
    gpu, _t = _capture(lambda: R.render(sc2, st2))
    check('both devices render the 25-lamp scene bitwise (the fallback '
          'plumbing, no driver)', _same(cpu, gpu), _maxdiff(cpu, gpu))
    st3 = _settings(shadow_default='PLANAR', max_lights=1,
                    light_limit_mode='FIRST')
    sc3 = many(st3)
    lim = R.render(sc3, st3)
    st4 = _settings(shadow_default='PLANAR', max_lights=1,
                    light_limit_mode='FIRST')
    sc4 = many(st4)
    for l in sc4.lights[1:]:
        l.shadow = 'NONE'
    check('under max_lights = 1 FIRST only lamp 0 casts a polygon (the '
          'Light Limit applies to the bake)',
          _same(lim, R.render(sc4, st4))
          and [i for i, _c, _d in st3._mask_params['planar']] == [0])


def test_c052_planar_device_fallback_and_rows():
    """The three featurematrix rows render, differ from their OFF frames,
    and fall back bitwise on the GPU device; PLANAR lamps are no casters
    for the cel field's key."""
    from .featurematrix import ROWS, build
    keys = [k for k, _o, _s in ROWS if k.startswith('planar shadows')]
    check('the three planar rows are in the feature matrix', len(keys) == 3,
          str(keys))
    for key in keys:
        sc, st = build(key)
        cpu = R.render(sc, st)
        sc2, st2 = build(key)
        st2.render_device = 'GPU'
        gpu = R.render(sc2, st2)
        check(f'row {key!r} falls back bitwise on the GPU device',
              _same(cpu, gpu), _maxdiff(cpu, gpu))
        sc3, st3 = build(key)
        st3.shadow_default = 'NONE'
        check(f'row {key!r} is non-vacuous (differs from no shadows)',
              not _same(cpu, R.render(sc3, st3)))
    st = _settings(shadow_default='PLANAR')
    check('casts_shadow says a PLANAR lamp is no caster (its lighting term '
          'is untouched by its polygon)',
          not LI.casts_shadow(Light(type='SUN'), st)
          and LI.casts_shadow(Light(type='SUN'),
                              _settings(shadow_default='MAP')))


# ==================================================================== C020

_DC_CENTRE, _DC_SIZE = (-1.3, 0.2, 0.5), 1.4


def _dc_scene(st, role='DC_INCLUDE', centre=_DC_CENTRE, size=_DC_SIZE,
              extra=()):
    sc = _scene(st)
    sc.shadow_volumes.append(_volume(centre, size, role, name='Modifier'))
    for e in extra:
        sc.shadow_volumes.append(e)
    return sc


def _dc_law(c, scale):
    """The PowerVR2 rule on a (N, 3) float32 colour: floor(rint(c*255) *
    scale / 256) / 255, every intermediate an exact small integer."""
    c8 = np.clip(np.asarray(c, np.float32), np.float32(0), np.float32(1))
    c8 = c8 * np.float32(255.0)
    c8 = np.rint(c8)
    c8 = c8 * np.float32(scale)
    c8 = c8 * np.float32(0.00390625)
    c8 = np.floor(c8)
    return c8 * np.float32(1.0 / 255.0)


def _mask_pixels(st):
    bits, ptri = st._mask_pack
    return ptri >= 0, bits, ptri


def test_c020_modvol_identity_at_defaults():
    """The dial is inert without a volume: modvol_scale = 200 on the plain
    demo is bitwise 1.89.0, as is a volume-role material on a scene whose
    mesh carries no such triangles."""
    RP = _prev_engine(ZIP)
    if RP is None:
        return
    now, prev = _prev_pair(_scene, modvol_scale=200)
    check('modvol_scale = 200 with no volume in the scene is bitwise 1.89.0',
          _same(now, prev), _maxdiff(now, prev))

    def roles(st):
        sc = _scene(st)
        sc.materials[1].volume_role = 'DS_SHADOW'
        return sc
    now, prev = _prev_pair(roles)
    check('a volume-role material with no split volume is bitwise 1.89.0 '
          '(the exporter, not the renderer, moves triangles)',
          _same(now, prev), _maxdiff(now, prev))
    st = _settings(shadows=False, force_model='CONSTANT')
    sc = _dc_scene(st)
    sc.shadow_volumes.append(_volume(_DS_CENTRE, _DS_SIZE, 'DS_SHADOW'))
    with_vol = R.render(sc, st)
    st0 = _settings(shadows=False, force_model='CONSTANT')
    check('with the render\'s Shadows off no volume is drawn (bitwise the '
          'volume-less frame, no pack baked)',
          _same(with_vol, R.render(_scene(st0), st0))
          and st._mask_pack is None)


def test_c020_modvol_parity_is_the_dreamcast_rule():
    """One INCLUDE volume around the ball's base under CONSTANT: every
    changed pixel obeys the integer law, the changed set is the convex
    cube's front-in-front-and-not-back parity (two culled rasters), the
    EXCLUDE kind changes the complement, the scale is monotonic, two
    overlapping INCLUDE volumes cancel, the coplanar cap does not speckle
    at 24 and 16 bits, a promoted decal inside darkens, the painter's
    sort compares polygons, bands are invariant."""
    off_frames = {}
    for rate in ('PIXEL', 'VERTEX'):
        st = _settings(force_model='CONSTANT', shading_rate=rate)
        sc = _dc_scene(st)
        on = R.render(sc, st)
        st0 = _settings(force_model='CONSTANT', shading_rate=rate)
        off = R.render(_scene(st0), st0)
        off_frames[rate] = off
        ch = _changed(on, off)
        want = _dc_law(off[ch][:, :3], 128)
        check(f'at {rate} rate every changed pixel is floor(rint(c*255)*128'
              '/256)/255 of the unshadowed colour, exactly',
              int(ch.sum()) > 20 and np.array_equal(on[ch][:, :3], want),
              f'{int(ch.sum())} px')
        cov, bits, ptri = _mask_pixels(st)
        check(f'at {rate} rate the changed set is exactly the pack\'s '
              'stencil bit', np.array_equal((bits[..., 1] > 0.5) & cov, ch))
    # (b) the convex cross-check on interior pixels
    st = _settings(force_model='CONSTANT')
    sc = _dc_scene(st)
    on = R.render(sc, st)
    ch = _changed(on, off_frames['PIXEL'])
    g = _gbuf_of_render(sc, st)
    vol = sc.shadow_volumes[0]
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    gf = CRg.GBuffer(W, H)
    CRg.rasterize(vol['verts'], vol['tris'], vp, W, H, cull='BACK', gbuf=gf,
                  depth_bits=st.depth_precision)
    gb = CRg.GBuffer(W, H)
    CRg.rasterize(vol['verts'], vol['tris'], vp, W, H, cull='FRONT', gbuf=gb,
                  depth_bits=st.depth_precision)
    both = gf.mask() & gb.mask()
    interior = both.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            interior &= np.roll(np.roll(both, dy, 0), dx, 1)
    # the named tie rule, written out with the raster's own constants: a
    # face within the A-buffer band of the surface is AT the surface
    tol = np.abs(g.zndc) * CRg.ABUF_DEPTH_TOL_REL + CRg.ABUF_DEPTH_TOL_ABS
    lim = g.zndc - tol
    pred = (gf.zndc < lim) & ~(gb.zndc < lim) & (g.tri >= 0)
    check('on the cube\'s interior pixels the changed set equals '
          '"front face nearer and back face not nearer" from two culled '
          'rasters (the convex cross-check)',
          int(interior.sum()) > 50
          and np.array_equal(ch[interior], pred[interior]),
          f'{int((ch != pred)[interior].sum())} of {int(interior.sum())} '
          'interior px disagree')
    # (c) EXCLUDE changes the complement within the covered pixels
    st_x = _settings(force_model='CONSTANT')
    sc_x = _dc_scene(st_x, role='DC_EXCLUDE')
    ex = R.render(sc_x, st_x)
    chx = _changed(ex, off_frames['PIXEL'])
    covered = g.tri >= 0
    check('DC_EXCLUDE changes exactly the complement of INCLUDE within the '
          'covered pixels (the whole frame outside the volume)',
          np.array_equal(chx, covered & ~ch),
          f'{int((chx != (covered & ~ch)).sum())} px differ')
    # (d) monotonic in the scale; 0 is black inside
    means = []
    for scale in (255, 128, 0):
        st_s = _settings(force_model='CONSTANT', modvol_scale=scale)
        sc_s = _dc_scene(st_s)
        fr = R.render(sc_s, st_s)
        means.append(float(fr[ch][:, :3].mean()))
        if scale == 0:
            check('at scale 0 the inside is black',
                  bool((fr[ch][:, :3] == 0.0).all()))
    check('the inside darkens monotonically through scale 255, 128, 0',
          means[0] > means[1] > means[2], str([round(v, 4) for v in means]))
    # (e) two INCLUDE volumes overlapping: the overlap is unshadowed
    st2 = _settings(force_model='CONSTANT')
    v2 = _volume((-1.3, -0.5, 0.5), 1.4, 'DC_INCLUDE', name='Second')
    sc2 = _dc_scene(st2, extra=(v2,))
    two = R.render(sc2, st2)
    ch2 = _changed(two, off_frames['PIXEL'])
    st3 = _settings(force_model='CONSTANT')
    sc3 = _scene(st3)
    sc3.shadow_volumes.append(v2)
    only2 = _changed(R.render(sc3, st3), off_frames['PIXEL'])
    check('two overlapping INCLUDE volumes cancel in their overlap (parity, '
          'a picture no lamp shadow draws)',
          int((ch & only2).sum()) > 10
          and np.array_equal(ch2, ch ^ only2),
          f'{int((ch & only2).sum())} overlap px')
    # (h) the coplanar cap at 24 and at 16 bits: floor pixels inside the
    # cap's own footprint (the cap raster's coverage) change, no other
    # floor pixel does -- the cap at equal depth is not a crossing
    for bits_ in (24, 16):
        st_c = _settings(force_model='CONSTANT', depth_precision=bits_)
        sc_c = _dc_scene(st_c, centre=(-1.3, 0.2, 0.7), size=1.4)
        on_c = R.render(sc_c, st_c)
        st_c0 = _settings(force_model='CONSTANT', depth_precision=bits_)
        off_c = R.render(_scene(st_c0), st_c0)
        ch_c = _changed(on_c, off_c)
        cov_c, _b, ptri_c = _mask_pixels(st_c)
        floor = cov_c & (sc_c.mesh.obj_index[np.clip(ptri_c, 0, None)] == 0)
        vc = sc_c.shadow_volumes[0]
        cap = CRg.GBuffer(W, H)
        CRg.rasterize(vc['verts'], vc['tris'][2:4], vp, W, H, cull='NONE',
                      gbuf=cap, depth_bits=bits_)
        want = cap.mask() & floor
        check(f'at {bits_}-bit depth a volume standing on the floor darkens '
              'exactly the floor pixels inside its cap (no speckle along '
              'the contact: the coplanar cap is not a crossing)',
              int(want.sum()) > 20
              and np.array_equal(ch_c & floor, want),
              f'{int(((ch_c & floor) != want).sum())} floor px differ')
    # (i) the promotion: a CLIP decal hovering inside a volume over the
    # empty floor darkens on its promoted pixels
    def decal_scene(st, with_vol=True):
        sc = _scene(st)
        v, n, uv, t, _m, _o = plane(z=0.3, size=0.5, mat=3, obj=3)
        v = v + np.array([0.0, -2.5, 0.0], np.float32)
        mesh = _mesh_concat([
            plane(z=0.0, size=11.0, mat=0, obj=0),
            sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
            cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
            (v, n, uv, t, 3, 3)])
        smooth = np.zeros(mesh.tris.shape[0], bool)
        smooth[mesh.mat_index == 1] = True
        mesh.smooth = smooth
        sc.mesh = mesh
        sc.materials.append(Material(name='Decal', index=3, model='LAMBERT',
                                     diffuse=(0.9, 0.9, 0.2),
                                     alpha_mode='CLIP', alpha_clip=0.5,
                                     opacity=0.9))
        from ..core.scene import ObjectInfo
        sc.objects.append(ObjectInfo(name='Decal', index=3,
                                     matrix_world=np.eye(4, dtype=np.float32)))
        if with_vol:
            sc.shadow_volumes.append(_volume((0.0, -2.5, 0.5), 1.4,
                                             'DC_INCLUDE', name='Modifier'))
        return sc
    st_d = _settings(force_model='CONSTANT', transparency='SORTED')
    sc_d = decal_scene(st_d)
    on_d = R.render(sc_d, st_d)
    st_d0 = _settings(force_model='CONSTANT', transparency='SORTED')
    off_d = R.render(decal_scene(st_d0, with_vol=False), st_d0)
    cov_d, _b, ptri_d = _mask_pixels(st_d)
    dec = cov_d & (sc_d.mesh.obj_index[np.clip(ptri_d, 0, None)] == 3)
    ch_d = _changed(on_d, off_d)
    check('a promoted CLIP decal hovering inside a volume is the opaque '
          'winner the bake saw and every one of its pixels darkens',
          int(dec.sum()) > 4 and bool(ch_d[dec].all()),
          f'{int(dec.sum())} decal px, {int((~ch_d & dec).sum())} lit')
    # (j) the painter's sort: parity recomputed from polygon depths
    st_p = _settings(force_model='CONSTANT', depth_sort='PAINTERS',
                     painters_key='CENTROID')
    sc_p = _dc_scene(st_p)
    on_p = R.render(sc_p, st_p)
    st_p0 = _settings(force_model='CONSTANT', depth_sort='PAINTERS',
                      painters_key='CENTROID')
    off_p = R.render(_scene(st_p0), st_p0)
    ch_p = _changed(on_p, off_p)
    cov_p, _b, ptri_p = _mask_pixels(st_p)
    from types import SimpleNamespace
    vp_ = sc_p.shadow_volumes[0]
    fd_vol = R.polygon_depths(SimpleNamespace(verts=vp_['verts'],
                                              tris=vp_['tris']),
                              view, eye, 'CENTROID')
    fd_mesh = R.polygon_depths(sc_p.mesh, view, eye, 'CENTROID')
    gv = CRg.GBuffer(W, H)
    flv = CRg.FragmentList()
    CRg.rasterize(vp_['verts'], vp_['tris'], vp, W, H, cull='NONE', gbuf=gv,
                  frags=flv, depth_write=False,
                  depth_bits=st_p.depth_precision)
    fpx, fpy, ftri, _fz, _fb, _ff = flv.finish()
    cnt = np.zeros((H, W), np.int32)
    zs = fd_mesh[np.clip(ptri_p, 0, None)][fpy, fpx]
    near = fd_vol[ftri] < zs - (np.abs(zs) * CRg.ABUF_DEPTH_TOL_REL
                                + CRg.ABUF_DEPTH_TOL_ABS)
    np.add.at(cnt, (fpy[near], fpx[near]), 1)
    par = ((cnt & 1) == 1) & cov_p
    check('under the painter\'s sort every changed pixel\'s parity, '
          'recomputed from the polygon depths of the floor and of the '
          'volume, is 1 (polygons compare as polygons)',
          int(ch_p.sum()) > 10 and np.array_equal(ch_p, par),
          f'{int((ch_p != par).sum())} px differ')
    # (g) band invariance
    st_w = _settings(force_model='CONSTANT')
    whole = np.asarray(R.render(_dc_scene(st_w), st_w))
    parts = []
    for bd in ((0, 36), (36, 72)):
        stb = _settings(force_model='CONSTANT')
        parts.append(np.asarray(R.render(_dc_scene(stb), stb, band=bd)))
    check('the modifier volume is band-invariant, bitwise',
          np.array_equal(np.concatenate(parts, 0), whole))


def test_c020_modvol_exporter_splits_once():
    """Under the bpy stub a material with a volume role leaves the surface
    mesh at the single split site: the triangles and the corners drop,
    one Scene.shadow_volumes entry carries the object's name, and a
    second export after the role goes back to NONE restores the mesh
    (the role is not frozen in the mesh cache)."""
    import types as _t
    from . import fakeblender as FB
    from ..core.settings import RenderSettings as _RS
    props, _eng = FB.install()
    from .. import export as EX

    def uidobj(name, factory, kind, uid):
        ob = FB.geometry_object(name, factory, kind=kind)
        ob.data = _t.SimpleNamespace(session_uid=uid)
        return ob

    vmat = FB.halcyon_material('VolMat')
    vmat.halcyon = FB.live(props.HalcyonMaterialSettings,
                           volume_role='DC_INCLUDE', polygon_id=7,
                           shadow_alpha=9)
    smat = FB.halcyon_material('SurfMat')
    smat.halcyon = FB.live(props.HalcyonMaterialSettings)
    a = uidobj('Modifier', lambda: FB.cube_mesh(materials=(vmat,)), 'MESH', 31)
    b = uidobj('Surface', lambda: FB.cube_mesh(materials=(smat,)), 'MESH', 32)
    EX._MESH_CACHE.clear()
    EX._DIRTY_DATA.clear()
    was = (EX._CACHE_ARMED, EX._DIRTY_ALL)
    EX._CACHE_ARMED = True
    EX._DIRTY_ALL = False
    try:
        sc1 = EX.export_scene(FB.depsgraph_of(props, [a, b]), _RS(), [])
        n_tris = int(sc1.mesh.tris.shape[0])
        n_verts = int(sc1.mesh.verts.shape[0])
        sc_ref = EX.export_scene(FB.depsgraph_of(props, [b]), _RS(), [])
        check('the volume\'s triangles leave the surface mesh (the merged '
              'mesh is the surface object alone)',
              n_tris == int(sc_ref.mesh.tris.shape[0]),
              f'{n_tris} vs {int(sc_ref.mesh.tris.shape[0])}')
        check('and its corners are compacted away (the bounds never see it)',
              n_verts == int(sc_ref.mesh.verts.shape[0]),
              f'{n_verts} vs {int(sc_ref.mesh.verts.shape[0])}')
        vols = list(sc1.shadow_volumes)
        check('one shadow volume carries the object\'s name, role, id, '
              'alpha and 12 triangles',
              len(vols) == 1 and vols[0]['name'] == 'Modifier'
              and vols[0]['role'] == 'DC_INCLUDE'
              and vols[0]['polygon_id'] == 7 and vols[0]['alpha'] == 9
              and int(vols[0]['tris'].shape[0]) == 12
              and int(vols[0]['verts'].shape[0]) > 0
              and int(vols[0]['tris'].max()) < int(vols[0]['verts'].shape[0]),
              str([(v['name'], v['role'], v['tris'].shape) for v in vols]))
        check('the surface mesh\'s triangle indices stay valid after the '
              'compaction',
              int(sc1.mesh.tris.max()) < n_verts
              and sc1.mesh.mat_index.shape[0] == n_tris
              and sc1.mesh.obj_index.shape[0] == n_tris)
        vmat.halcyon.volume_role = 'NONE'
        sc2 = EX.export_scene(FB.depsgraph_of(props, [a, b]), _RS(), [])
        check('flipping the role back to NONE restores the mesh on the next '
              'export (the role is not frozen in the mesh cache)',
              int(sc2.mesh.tris.shape[0]) == n_tris + 12
              and not sc2.shadow_volumes,
              f'{int(sc2.mesh.tris.shape[0])} tris, '
              f'{len(sc2.shadow_volumes)} volumes')
    finally:
        EX._CACHE_ARMED, EX._DIRTY_ALL = was
        EX._MESH_CACHE.clear()
        EX._DIRTY_DATA.clear()


def test_c020_modvol_gpu_twin():
    """The Gouraud rig (the test bakes the pack): CONSTANT at PIXEL and
    VERTEX rate d == 0.0 on every masked pixel and the frame at the rig's
    bar; the plain lit demo with the volume at the TR:8157 bar; no pass
    emits the block; the plan refuses nothing by name."""
    from ..gpu import shade as GSH

    def twin(label, rate, model='CONSTANT', bar=0.0):
        kw = dict(shading_rate=rate)
        if model is not None:
            kw['force_model'] = model
        st = _settings(**kw)
        sc = _dc_scene(st)
        cpu = R.render(sc, st)
        g, job, vp, view, eye = _gbuffer_for(sc, st)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        check(f'{label}: a volume scene refuses nothing by name (the plan '
              'qualifies)', p is not None and why is None, str(why))
        if p is None:
            return
        raw, hit = GSH.simulate(job, g)
        if raw is None:
            check(f'{label}: the passes simulate', False, str(hit))
            return
        pack = _bake_pack(sc, st, g, vp, view, eye)
        check(f'{label}: the bake finds the volume',
              pack is not None and st._mask_params['modvol'] == 128)
        out = SM.apply_frame(raw, g, hit, st)
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
        mm = pack[0][yy, xx, 1] > 0.5
        check(f'{label}: on every pixel inside the volume the readback '
              'apply is bitwise the CPU frame (d == 0.0)',
              int(mm.sum()) > 20 and float(d[mm].max()) == 0.0,
              f'{int(mm.sum())} px, max {float(d[mm].max()):.6g}')
        if bar == 0.0:
            check(f'{label}: the whole frame is bitwise (d == 0.0)',
                  float(d.max()) == 0.0, f'max {float(d.max()):.6g}')
        else:
            check(f'{label}: the frame matches the CPU at the bar',
                  float(d.max()) < bar, f'max {float(d.max()):.6f}')
        srcs = '\n'.join(s for _mi, _n, s, _b in p)
        check(f'{label}: no pass source carries the mask block',
              'hal_mp' not in srcs and 'hal_maskpack' not in srcs)

    twin('CONSTANT at PIXEL rate', 'PIXEL')
    twin('CONSTANT at VERTEX rate', 'VERTEX', bar=6e-3)
    twin('the lit demo at PIXEL rate', 'PIXEL', model=None, bar=6e-3)
    # the device fallback and the two featurematrix rows
    from .featurematrix import ROWS, build
    keys = [k for k, _o, _s in ROWS if k.startswith('Dreamcast modifier')]
    check('the two modifier-volume rows are in the feature matrix',
          len(keys) == 2, str(keys))
    for key in keys:
        sc, st = build(key)
        cpu = R.render(sc, st)
        sc2, st2 = build(key)
        st2.render_device = 'GPU'
        check(f'row {key!r} falls back bitwise on the GPU device',
              _same(cpu, R.render(sc2, st2)))
        sc3, st3 = build(key)
        sc3.shadow_volumes = []
        check(f'row {key!r} is non-vacuous (the volume moves the picture)',
              not _same(cpu, R.render(sc3, st3)))


# ==================================================================== C036

_DS_CENTRE, _DS_SIZE = (-1.3, 0.2, 0.3), 1.6


def _ds_scene(st, color=(0.0, 0.0, 0.0), alpha=16, polygon_id=1,
              ball_id=1, centre=_DS_CENTRE, size=_DS_SIZE, extra=()):
    sc = _scene(st)
    sc.materials[1].polygon_id = ball_id
    sc.shadow_volumes.append(_volume(centre, size, 'DS_SHADOW', color=color,
                                     alpha=alpha, polygon_id=polygon_id,
                                     name='Shadow'))
    for e in extra:
        sc.shadow_volumes.append(e)
    return sc


def _ds_law(c, cs6, a):
    """The DS blend on a (N, 3) float32 colour: 6-bit channels,
    (Cs*(a+1) + Cd*(31-a)) >> 5, every intermediate an exact integer."""
    c6 = np.clip(np.asarray(c, np.float32), np.float32(0), np.float32(1))
    c6 = c6 * np.float32(63.0)
    c6 = np.rint(c6)
    c6 = c6 * np.float32(31 - a)
    c6 = c6 + np.asarray([np.float32(int(v) * (a + 1)) for v in cs6],
                         np.float32).reshape(1, 3)
    c6 = c6 * np.float32(0.03125)
    c6 = np.floor(c6)
    return c6 * np.float32(1.0 / 63.0)


def test_c036_ds_shadow_identity_at_defaults():
    """polygon_id = 5 and shadow_alpha = 30 on every material with role
    NONE: bitwise 1.89.0 (the fields are inert without a DS volume)."""
    RP = _prev_engine(ZIP)
    if RP is None:
        return

    def fields(st):
        sc = _scene(st)
        for m in sc.materials:
            m.polygon_id = 5
            m.shadow_alpha = 30
        return sc
    now, prev = _prev_pair(fields)
    check('polygon_id and shadow_alpha on surface materials are bitwise '
          '1.89.0 (inert without a DS volume)', _same(now, prev),
          _maxdiff(now, prev))


def test_c036_ds_shadow_polygons_follow_gbatek():
    """One DS volume around the ball, black, alpha 16, ID 1 with the ball
    carrying ID 1 under CONSTANT: the 5-bit blend law at PIXEL and VERTEX
    rate, no ball pixel changes while the floor under it does, ID 2 on
    the ball brings it in, alpha is monotonic, a white polygon
    brightens, two volumes blend in scene order, bands are invariant,
    the coplanar cap does not speckle."""
    offs = {}
    for rate in ('PIXEL', 'VERTEX'):
        st = _settings(force_model='CONSTANT', shading_rate=rate)
        sc = _ds_scene(st)
        on = R.render(sc, st)
        st0 = _settings(force_model='CONSTANT', shading_rate=rate)
        off = R.render(_scene(st0), st0)
        offs[rate] = off
        ch = _changed(on, off)
        want = _ds_law(off[ch][:, :3], (0, 0, 0), 16)
        check(f'at {rate} rate every changed pixel is the DS 5-bit blend of '
              'the unshadowed colour toward black at alpha 16, exactly',
              int(ch.sum()) > 20 and np.array_equal(on[ch][:, :3], want),
              f'{int(ch.sum())} px')
        cov, bits, ptri = _mask_pixels(st)
        obj = sc.mesh.obj_index[np.clip(ptri, 0, None)]
        check(f'at {rate} rate no ball pixel changes (self-exclusion by '
              'polygon ID) while floor pixels under the ball do',
              int((ch & (obj == 1)).sum()) == 0
              and int((ch & (obj == 0)).sum()) > 20,
              f'{int((ch & (obj == 1)).sum())} ball px, '
              f'{int((ch & (obj == 0)).sum())} floor px')
    # (c) the ID rule is live
    st = _settings(force_model='CONSTANT')
    sc = _ds_scene(st, ball_id=2)
    on2 = R.render(sc, st)
    ch2 = _changed(on2, offs['PIXEL'])
    cov, bits, ptri = _mask_pixels(st)
    obj = sc.mesh.obj_index[np.clip(ptri, 0, None)]
    check('with the ball at polygon ID 2 its pixels inside the volume '
          'change too (the ID rule is live)',
          int((ch2 & (obj == 1)).sum()) > 20,
          f'{int((ch2 & (obj == 1)).sum())} ball px')
    # (d) alpha monotonic with black Cs
    st = _settings(force_model='CONSTANT')
    base_on = R.render(_ds_scene(st), st)
    ch = _changed(base_on, offs['PIXEL'])
    means = []
    for a in (1, 16, 30):
        st_a = _settings(force_model='CONSTANT')
        fr = R.render(_ds_scene(st_a, alpha=a), st_a)
        means.append(float(fr[ch][:, :3].mean()))
    check('the shadow darkens monotonically through alpha 1, 16, 30',
          means[0] > means[1] > means[2], str([round(v, 4) for v in means]))
    # (e) a white polygon brightens
    st_w = _settings(force_model='CONSTANT')
    wh = R.render(_ds_scene(st_w, color=(1.0, 1.0, 1.0)), st_w)
    chw = _changed(wh, offs['PIXEL'])
    check('with a white polygon colour the changed pixels are brighter '
          'than the surface (the blend goes toward the material colour)',
          int(chw.sum()) > 20
          and bool((wh[chw][:, :3].sum(axis=1)
                    > offs['PIXEL'][chw][:, :3].sum(axis=1)).all()))
    # (f) two volumes in scene order: overlap = blend(blend(c, v0), v1)
    v0 = _volume(_DS_CENTRE, _DS_SIZE, 'DS_SHADOW', color=(1, 1, 1),
                 alpha=8, polygon_id=1, name='v0')
    v1 = _volume((-1.3, -0.5, 0.3), 1.6, 'DS_SHADOW', color=(0, 0, 0),
                 alpha=20, polygon_id=1, name='v1')
    st_f = _settings(force_model='CONSTANT')
    sc_f = _scene(st_f)
    sc_f.materials[1].polygon_id = 1
    sc_f.shadow_volumes += [v0, v1]
    two = R.render(sc_f, st_f)
    st_r = _settings(force_model='CONSTANT')
    sc_r = _scene(st_r)
    sc_r.materials[1].polygon_id = 1
    sc_r.shadow_volumes += [v1, v0]
    rev = R.render(sc_r, st_r)
    cov, bits, ptri = _mask_pixels(st_f)
    ov = cov & (SM._bit(bits[..., 2], 0) > 0.5) & (SM._bit(bits[..., 2], 1)
                                                   > 0.5)
    c = offs['PIXEL'][ov][:, :3]
    want = _ds_law(_ds_law(c, (63, 63, 63), 8), (0, 0, 0), 20)
    check('where two DS volumes overlap the pixel is blend(blend(c, v0), '
          'v1) in scene order, exactly',
          int(ov.sum()) > 10 and np.array_equal(two[ov][:, :3], want),
          f'{int(ov.sum())} overlap px')
    check('and the reverse order is a different picture there',
          int(ov.sum()) > 10 and not np.array_equal(rev[ov], two[ov]))
    # (g) band invariance
    st_b = _settings(force_model='CONSTANT')
    whole = np.asarray(R.render(_ds_scene(st_b), st_b))
    parts = []
    for bd in ((0, 36), (36, 72)):
        stb = _settings(force_model='CONSTANT')
        parts.append(np.asarray(R.render(_ds_scene(stb), stb, band=bd)))
    check('the DS shadow polygon is band-invariant, bitwise',
          np.array_equal(np.concatenate(parts, 0), whole))
    # (h) the coplanar cap: a volume standing on the floor draws exactly
    # the floor pixels inside its cap (the cap's back face fails at equal
    # depth and sets the mask; the entry face draws)
    for bits_ in (24, 16):
        st_c = _settings(force_model='CONSTANT', depth_precision=bits_)
        sc_c = _ds_scene(st_c, centre=(-1.3, 0.2, 0.8), size=1.6)
        on_c = R.render(sc_c, st_c)
        st_c0 = _settings(force_model='CONSTANT', depth_precision=bits_)
        off_c = R.render(_scene(st_c0), st_c0)
        ch_c = _changed(on_c, off_c)
        cov_c, _b, ptri_c = _mask_pixels(st_c)
        floor = cov_c & (sc_c.mesh.obj_index[np.clip(ptri_c, 0, None)] == 0)
        vc = sc_c.shadow_volumes[0]
        _v, _p, vp, _e = R.camera_matrices(sc_c.camera, W, H)
        cap = CRg.GBuffer(W, H)
        CRg.rasterize(vc['verts'], vc['tris'][2:4], vp, W, H, cull='NONE',
                      gbuf=cap, depth_bits=bits_)
        want = cap.mask() & floor
        check(f'at {bits_}-bit depth a DS volume standing on the floor draws '
              'exactly the floor pixels inside its cap (no speckle along '
              'the contact)',
              int(want.sum()) > 20 and np.array_equal(ch_c & floor, want),
              f'{int(((ch_c & floor) != want).sum())} floor px differ')


def test_c036_ds_shadow_gpu_twin():
    """The Gouraud rig on the CONSTANT scene at PIXEL and VERTEX rate:
    d == 0.0 on every drawn pixel; the lit demo printed as an extra at
    the stated bar (30/32)/63 + 6e-3; no pass emits the block; 25 DS
    volumes print the 24-volume line once and both devices agree."""
    from ..gpu import shade as GSH

    def twin(label, rate, model='CONSTANT', bar=0.0):
        kw = dict(shading_rate=rate)
        if model is not None:
            kw['force_model'] = model
        st = _settings(**kw)
        sc = _ds_scene(st)
        cpu = R.render(sc, st)
        g, job, vp, view, eye = _gbuffer_for(sc, st)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        check(f'{label}: a DS volume scene refuses nothing by name',
              p is not None and why is None, str(why))
        if p is None:
            return
        raw, hit = GSH.simulate(job, g)
        if raw is None:
            check(f'{label}: the passes simulate', False, str(hit))
            return
        pack = _bake_pack(sc, st, g, vp, view, eye)
        check(f'{label}: the bake finds the DS volume',
              pack is not None and len(st._mask_params['ds']) == 1
              and st._mask_params['ds'][0] == (0, (0, 0, 0), 16))
        out = SM.apply_frame(raw, g, hit, st)
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
        mm = pack[0][yy, xx, 2] > 0.5
        if model == 'CONSTANT':
            check(f'{label}: on every drawn pixel the readback apply is '
                  'bitwise the CPU frame (d == 0.0)',
                  int(mm.sum()) > 20 and float(d[mm].max()) == 0.0,
                  f'{int(mm.sum())} px, max {float(d[mm].max()):.6g}')
        if bar == 0.0:
            check(f'{label}: the whole frame is bitwise (d == 0.0)',
                  float(d.max()) == 0.0, f'max {float(d.max()):.6g}')
        elif model == 'CONSTANT':
            check(f'{label}: the frame matches the CPU at the rig\'s bar',
                  float(d.max()) < bar, f'max {float(d.max()):.6f}')
        else:
            stated = (30.0 / 32.0) / 63.0 + 6e-3
            print(f'         (extra) {label}: max {float(d.max()):.6f} '
                  f'against the stated bar {stated:.4f} '
                  f'({"within" if float(d.max()) < stated else "OVER"})')
        srcs = '\n'.join(s for _mi, _n, s, _b in p)
        check(f'{label}: no pass source carries the DS block',
              'hal_db' not in srcs and 'hal_maskpack' not in srcs)

    twin('CONSTANT at PIXEL rate', 'PIXEL')
    twin('CONSTANT at VERTEX rate', 'VERTEX', bar=6e-3)
    twin('the lit demo at PIXEL rate', 'PIXEL', model=None, bar=1.0)
    # the 24-volume limit, both devices
    def many(st):
        sc = _scene(st)
        sc.materials[1].polygon_id = 1
        for i in range(25):
            sc.shadow_volumes.append(_volume((-1.3 + 0.02 * i, 0.2, 0.3),
                                             1.6, 'DS_SHADOW', alpha=2,
                                             polygon_id=1, name=f'vol{i}'))
        return sc
    st = _settings(force_model='CONSTANT')
    sc = many(st)
    cpu, text = _capture(lambda: R.render(sc, st))
    check('25 DS volumes print the 24-volume line once, naming the volume',
          text.count('shadow volumes: 24 per frame') == 1
          and "volume 'vol24'" in text, repr(text[-200:]))
    check('the pack carries exactly 24 DS entries',
          len(st._mask_params['ds']) == 24)
    st2 = _settings(force_model='CONSTANT')
    st2.render_device = 'GPU'
    sc2 = many(st2)
    gpu, _t = _capture(lambda: R.render(sc2, st2))
    check('both devices render the 25-volume scene bitwise (the fallback '
          'plumbing, no driver)', _same(cpu, gpu), _maxdiff(cpu, gpu))
    # the featurematrix row
    from .featurematrix import ROWS, build
    keys = [k for k, _o, _s in ROWS if k.startswith('DS shadow polygon')]
    check('the DS row is in the feature matrix under CONSTANT',
          len(keys) == 1 and dict((k, o) for k, o, _s in ROWS)[keys[0]]
          == {'force_model': 'CONSTANT'}, str(keys))
    if keys:
        sc, st = build(keys[0])
        cpu = R.render(sc, st)
        sc2, st2 = build(keys[0])
        st2.render_device = 'GPU'
        check('the DS row falls back bitwise on the GPU device',
              _same(cpu, R.render(sc2, st2)))
        sc3, st3 = build(keys[0])
        sc3.shadow_volumes = []
        check('the DS row is non-vacuous (the floor under the ball darkens, '
              'the ball does not)', not _same(cpu, R.render(sc3, st3)))


def main():
    from . import utf8_console
    utf8_console()
    order = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in order:
        print(fn.__name__)
        try:
            fn()
        except Exception:                                       # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILS.append(fn.__name__)
    print()
    print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS) if FAILS
          else 'all R251 shadow tests passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
