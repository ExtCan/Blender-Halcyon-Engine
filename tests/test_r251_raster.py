"""R251 raster pack (RAST-A1): C084 integer pixel centres, C027 near-plane
whole-triangle rejection, C004 the PlayStation ordering table, C007 the
N64 18-bit floating z-buffer, C026 the GameCube compressed Z and the
fixed-point W-buffer, C075 the Voodoo 16-bit floating W-buffer.

Run with:  python -m halcyon.tests.test_r251_raster

Every test names the law it proves. The first pins the 1.89.0 zip loudly
and the identity at defaults (render AND post bitwise); each feature then
proves its semantic A/B law on the core function, its GPU twin in the
simulator (d == 0.0 or the stated bar), its refusal by name, and the
door of the device switch where the wave entry asks.
"""
import contextlib
import importlib
import io
import sys
import traceback

import numpy as np

from ..core import post
from ..core import raster
from ..core import render as R
from ..core.scene import Camera, MeshData, ObjectInfo
from ..gpu import craster as CRA
from ..presets.library import PRESETS
from . import featurematrix as FM
from .scenebuild import _mesh_concat, cube, demo_scene, look_at_matrix, \
    plane, sphere
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name
          + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------ shared rigs

W, H = 96, 72


def _wparams(proj, is_ortho=False):
    """The derivation block's own arithmetic (render.py), for rigs that
    call the rasteriser directly."""
    A = float(-proj[2, 2])
    B = float(proj[2, 3])
    near = -B / (1.0 + A) if (1.0 + A) != 0.0 else 0.0
    far = B / (1.0 - A) if (1.0 - A) != 0.0 else 0.0
    return raster.WParams(near=np.float32(near), far=np.float32(far),
                          A=np.float32(A), B=np.float32(B),
                          is_ortho=is_ortho)


def _opts_for(sc, w, h, **kw):
    """(vp, RasterOpts) for a scene, the W parameters derived from its
    projection exactly as render() derives them."""
    _view, proj, vp, _eye = R.camera_matrices(sc.camera, w, h)
    kw.setdefault('wparams', _wparams(proj))
    return vp, raster.RasterOpts(**kw)


def _cpu_gbuf(sc, vp, w, h, opts, depth_bits=24, flat_depth=None,
              cull='NONE'):
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     opts=opts, depth_bits=depth_bits, flat_depth=flat_depth,
                     cull=cull)
    return g


def _twin(sc, vp, w, h, opts, label, depth_bits=24, flat_depth=None,
          refer=False, cull='NONE', zkey_bitwise=False):
    """The TR:16091 bar (zero differing ids, bary < 1e-5, zndc < 1e-6,
    identical front) between `rasterize` and `simulate_raster`; under an
    encoding the key plane and the host-decoded zndc are held BITWISE.
    Returns (gbuf, sim tuple)."""
    g = _cpu_gbuf(sc, vp, w, h, opts, depth_bits=depth_bits,
                  flat_depth=flat_depth, cull=cull)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
        sc.mesh, vp, w, h, depth_bits=depth_bits, opts=opts)
    sim = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w, h, cull=cull,
                              depth_bits=depth_bits, refer=refer, opts=opts,
                              flat_depth=flat_depth)
    tri, bary, zndc, front, _b2, _lin, mark = sim
    d = int((tri != g.tri).sum())
    check(f'{label}: the kernel picks the CPU fill\'s winner at every pixel '
          '(zero differing ids)', d == 0, f'{d} of {w * h} differ')
    cov = g.tri >= 0
    if cov.any() and d == 0:
        db = float(np.abs(bary[cov] - g.bary[cov]).max())
        df = int((front[cov] != g.front[cov]).sum())
        check(f'{label}: barycentrics agree to the ulp (< 1e-5)', db < 1e-5,
              f'{db:.2e}')
        check(f'{label}: front flags identical', df == 0, str(df))
        if zkey_bitwise:
            same_key = bool(np.array_equal(zndc[cov], g.zkey[cov]))
            check(f'{label}: the stored depth CODE is bitwise the CPU\'s '
                  '(aux.x holds the key on both roads)', same_key)
            dec = raster.decode_key(zndc, opts, depth_bits)
            check(f'{label}: the host-decoded zndc is bitwise the CPU\'s '
                  'stored NDC', bool(np.array_equal(dec[cov], g.zndc[cov])))
        else:
            dz = float(np.abs(zndc[cov] - g.zndc[cov]).max())
            check(f'{label}: depth agrees (< 1e-6)', dz < 1e-6, f'{dz:.2e}')
    return g, sim


def _fight_rig(z_a, z_b, w=64, h=48, slant=0.1137):
    """Two full-frame SLANTED quads in clip space (identity projection,
    w = 1): quad A (ids 0, 1) runs from NDC z `z_a - slant` at the bottom
    row to `z_a + slant` at the top, quad B (ids 2, 3) the same around
    `z_b` -- two coplanar-offset floors, the depth-encoding fighting rig:
    B is nearer where z_b < z_a, and the slant walks both across every
    step boundary of the code."""
    # z = zc + slant * (0.37 x + y): planar, varying across BOTH axes with
    # per-pixel steps that are no neat fraction of a 64-unit code (a 0.1
    # slant stepped 204.8 units per column, a lattice that never landed
    # inside the 3.9-unit window), so the frame's pixels spread over the
    # code's steps
    def corners(zc):
        return [[-1, -1, zc - 1.37 * slant], [1, -1, zc - 0.63 * slant],
                [1, 1, zc + 1.37 * slant], [-1, 1, zc + 0.63 * slant]]
    v = np.array(corners(z_a) + corners(z_b), np.float32)
    t = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], np.int32)
    mesh = MeshData(verts=v, tris=t)
    mesh.mat_index = np.zeros(4, np.int32)
    mesh.obj_index = np.array([0, 0, 1, 1], np.int32)
    return mesh


class _Sc:
    """A bare scene holder for the rasteriser rigs."""

    def __init__(self, mesh):
        self.mesh = mesh


def _capture(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*a, **k)
    return out, buf.getvalue()


def _post(img, st):
    return post.process(img, st, frame=1, seed=0,
                        target_size=(st.resolution_x, st.resolution_y),
                        allow_resize=False)


# ================================================================ pins


def test_a_identity_at_defaults():
    """Every new field at its default leaves every pixel bitwise the
    1.89.0 release's: render AND post, the demo scene and the pack's own
    fighting-sheet scene, rendered by this engine and by the whole 1.89.0
    engine imported from the zip beside the package."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the neutrality pin runs)',
          RP is not None)
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0]
                                        + '.post')
    for label, builder in (('the demo scene', lambda st: demo_scene(st)),
                           ('the fighting-sheet scene',
                            FM.SCENES['depth_fight_pair']),
                           ('the near-wall scene', FM.SCENES['near_wall'])):
        st = base_settings(W, H)
        now = np.asarray(R.render(builder(st), st))
        st2 = base_settings(W, H)
        prev = np.asarray(RP.render(builder(st2), st2))
        same = now.shape == prev.shape and bool(np.array_equal(now, prev))
        check(f'{label} at defaults renders bitwise the 1.89.0 engine',
              same, f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
        if same:
            p_now = np.asarray(_post(now, st))
            p_prev = np.asarray(prev_post.process(
                prev, st2, frame=1, seed=0, target_size=(W, H),
                allow_resize=False))
            check(f'{label} at defaults: post.process bitwise the 1.89.0 '
                  'post', p_now.shape == p_prev.shape
                  and bool(np.array_equal(p_now, p_prev)))
    # the five new settings exist at their neutral defaults
    st = base_settings(W, H)
    check('the five new settings default to HALF / CLIP / LINEAR / 4096 / '
          '40.0', (st.pixel_center, st.near_clip_mode, st.depth_encoding,
                   st.ot_length, st.ot_far)
          == ('HALF', 'CLIP', 'LINEAR', 4096, 40.0))
    check('RasterOpts() at defaults keys to the neutral tuple',
          raster.RasterOpts().key() == (0.0, False, None, 'LINEAR', None,
                                        None, False, None))


def test_a_kernel_assembly():
    """The new kernel text survives the compute build: every function
    of the pack sits BEFORE hal_rc_fetch, so _exact_core's two replaced
    spans never swallow it, and the fragment wrapper compiles in the
    simulator with the 15 float push constants (60 of ~128 bytes)."""
    from ..shaders.compiler import try_compile
    names = ('float hal_zkey_n64(', 'float hal_zkey_gc(', 'float hal_zkey_w(',
             'float hal_zkey_voodoo(', 'float hal_zkey(',
             'uniform float hal_flat;', 'uniform float hal_zenc;',
             'if (hal_flat > 0.5) { zz = ca.z; }')
    for src_name, src in (('FRAGMENT_SOURCE', CRA.FRAGMENT_SOURCE),
                          ('COMPUTE_SOURCE', CRA.COMPUTE_SOURCE),
                          ('COMPUTE_SOURCE_LIN', CRA.COMPUTE_SOURCE_LIN)):
        missing = [n for n in names if n not in src]
        check(f'{src_name} carries every new kernel function and uniform',
              not missing, str(missing))
    core = CRA.KERNEL_CORE
    check('every new function sits before hal_rc_fetch (the compute '
          'build\'s replaced span starts there)',
          all(core.index(n) < core.index('vec4 hal_rc_fetch')
              for n in names[:5]))
    check('the compute build swapped the fetchers for texelFetch and kept '
          'the encoding dispatch',
          'texelFetch' in CRA.COMPUTE_SOURCE
          and 'hal_zkey(zz, invw_c, v, stepc)' in CRA.COMPUTE_SOURCE)
    prog, err = try_compile(CRA.FRAGMENT_SOURCE, 'GLSL')
    check('the fragment wrapper compiles in the simulator', prog is not None,
          str(err))
    ku = CRA.kernel_uniforms(None, 24, False)
    check('kernel_uniforms at LINEAR-24: zsteps 2^24-1, every encoding '
          'uniform at rest', ku['hal_zsteps'] == 16777215.0
          and ku['hal_zenc'] == 0.0 and ku['hal_flat'] == 0.0)
    opts = raster.RasterOpts(enc='N64_FLOAT18')
    ku = CRA.kernel_uniforms(opts, 16, False)
    check('under an encoding hal_zsteps is 0.0 (the LINEAR quantise block '
          'never touches a code) and hal_zenc names it',
          ku['hal_zsteps'] == 0.0 and ku['hal_zenc'] == 1.0
          and ku['hal_wob'] == float(np.float32(2.5e-6 * 262143.0)))


# ================================================================ C084


def test_c084_pixel_center():
    """Direct3D 3-9 sampled at the integer corner: the raster's sample
    grid moves half a pixel, the geometry does not."""
    # (b) a quarter-pixel column moves one pixel right: the UV-tile rig
    # (identity vp, 32x32) with one quad spanning x in [10.25, 11.25]
    n = 32
    x0, x1 = 10.25 / n * 2.0 - 1.0, 11.25 / n * 2.0 - 1.0
    v = np.array([[x0, -1, 0], [x1, -1, 0], [x1, 1, 0], [x0, 1, 0]],
                 np.float32)
    t = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    eye4 = np.eye(4, dtype=np.float32)
    cols = {}
    for label, shift in (('HALF', 0.0), ('INTEGER_D3D', 0.5)):
        g = raster.GBuffer(n, n)
        raster.rasterize(v, t, eye4, n, n, gbuf=g,
                         opts=raster.RasterOpts(pixel_shift=shift))
        cols[label] = sorted(set(np.nonzero(g.tri >= 0)[1].tolist()))
    check('HALF: the quarter-pixel column covers column 10 only (its '
          'centre 10.5 is inside [10.25, 11.25])', cols['HALF'] == [10],
          str(cols['HALF']))
    check('INTEGER_D3D: the same quad covers column 11 only (the shifted '
          'span [10.75, 11.75] holds centre 11.5)',
          cols['INTEGER_D3D'] == [11], str(cols['INTEGER_D3D']))
    # the shift is applied after the snap: a snapped vertex at k lands
    # at k + 0.5, exactly on this raster's pixel centre
    clip = np.array([[0.3 / n * 2.0 - 1.0, 0.0, 0.0, 1.0]], np.float32)
    scr, _iw, _z = raster._clip_to_screen(clip, n, n, snap=1.0,
                                          pixel_shift=0.5)
    check('a snapped vertex lands on k + 0.5 (snap first, then the shift)',
          float(scr[0, 0]) == 0.5, str(scr[0]))
    scr2, _iw, _z = raster._clip_to_screen(clip, n, n, snap=1.0)
    check('project() at defaults keeps 1.89.0\'s statements (no shift)',
          float(scr2[0, 0]) == 0.0)
    # the whole frame moves: the demo scene rendered under INTEGER_D3D
    # differs from HALF, and the wire lines move with it
    st = base_settings(W, H, pixel_center='INTEGER_D3D')
    st.shadows = False
    on = np.asarray(R.render(demo_scene(st, with_texture=False), st))
    st0 = base_settings(W, H)
    st0.shadows = False
    off = np.asarray(R.render(demo_scene(st0, with_texture=False), st0))
    moved = int((np.abs(on - off).max(axis=2) > 1e-6).sum())
    check('the demo frame moves under INTEGER_D3D (the sample grid, half a '
          'pixel)', moved > 500, f'{moved} px')
    st = base_settings(W, H, pixel_center='INTEGER_D3D', render_wire=True)
    st.shadows = False
    wire_on = np.asarray(R.render(demo_scene(st, with_texture=False), st))
    st0 = base_settings(W, H, render_wire=True)
    st0.shadows = False
    wire_off = np.asarray(R.render(demo_scene(st0, with_texture=False),
                                   st0))
    check('the wire lines fall on the shifted grid too (edge_distance_exact '
          'takes the shift)', not np.array_equal(wire_on, wire_off))
    # (c) GPU twin at 160x120 on the demo scene, the TR:16091 bar
    st = base_settings(160, 120)
    sc = demo_scene(st, with_texture=False)
    vp, opts = _opts_for(sc, 160, 120, pixel_shift=0.5)
    _twin(sc, vp, 160, 120, opts, 'C084 INTEGER_D3D on the demo scene')
    # (d) the door stays open: a GPU-device render knocks with the shift
    seen = {}
    orig = CRA.raster_into_gbuffer

    def rig(*a, **k):
        seen['opts'] = k.get('opts')
        return orig(*a, **k)
    CRA.raster_into_gbuffer = rig
    try:
        R._GBUF_CACHE.clear()
        st = base_settings(W, H, pixel_center='INTEGER_D3D')
        st.shadows = False
        st.render_device = 'GPU'
        st.gpu_raster = True
        gpu, out = _capture(R.render, demo_scene(st, with_texture=False), st)
    finally:
        CRA.raster_into_gbuffer = orig
    check('raster_into_gbuffer is not refused for the pixel centre: the '
          'door is knocked with pixel_shift 0.5 and no gate names it',
          seen.get('opts') is not None and seen['opts'].pixel_shift == 0.5
          and 'pixel' not in out.lower().split('rasterising on the cpu')[-1]
          .split('\n')[0], out.strip().split('\n')[0][:100])
    check('and, without a driver, the GPU-device frame is bitwise the CPU '
          'frame under INTEGER_D3D', bool(np.array_equal(np.asarray(gpu),
                                                          on)))
    # (e) the FM row and the preset
    check("the FM row 'integer pixel centres (Direct3D 9)' exists on the "
          'demo scene', ('integer pixel centres (Direct3D 9)',
                         {'pixel_center': 'INTEGER_D3D'}, 'demo') in FM.ROWS)
    check('the XBOX preset carries INTEGER_D3D (Direct3D 8 on the NV2A)',
          PRESETS['XBOX']['settings'].get('pixel_center') == 'INTEGER_D3D')
    check('pixel_shift_of reads the setting (0.5 / 0.0)',
          raster.pixel_shift_of(base_settings(W, H, pixel_center='INTEGER_D3D')) == 0.5
          and raster.pixel_shift_of(base_settings(W, H)) == 0.0)


# ================================================================ C027


def test_c027_near_reject():
    """The PS2 VU1 / PS1 rule: a triangle with any vertex past the near
    plane, the far plane or the 6.4-half-screen guard band is dropped
    WHOLE -- coverage only ever shrinks, never a new pixel."""
    st = base_settings(W, H)
    sc = FM.SCENES['near_wall'](st)
    vp, o_clip = _opts_for(sc, W, H)
    _vp, o_rej = _opts_for(sc, W, H, reject=True)
    g_c = _cpu_gbuf(sc, vp, W, H, o_clip)
    g_r = _cpu_gbuf(sc, vp, W, H, o_rej)
    clip_cov = g_c.tri >= 0
    rej_cov = g_r.tri >= 0
    floor = sc.mesh.mat_index == 0
    floor_px_c = int(floor[g_c.tri[clip_cov]].sum())
    floor_px_r = int(floor[g_r.tri[rej_cov]].sum())
    check('the near-wall scene is the case: under CLIP the floor is cut and '
          'drawn', floor_px_c > 200, str(floor_px_c))
    check('under REJECT the straddling floor triangle vanishes: the covered '
          'count strictly drops', int(rej_cov.sum()) < int(clip_cov.sum())
          and floor_px_r < floor_px_c,
          f'{int(clip_cov.sum())} -> {int(rej_cov.sum())} px, floor '
          f'{floor_px_c} -> {floor_px_r}')
    check('REJECT never adds a pixel: its coverage is a subset of CLIP\'s',
          bool((rej_cov & ~clip_cov).sum() == 0))
    # the code's own survivor mask names the pixels the id law holds on
    clip_pts, _s, _i, _z = raster.project(sc.mesh.verts, vp, W, H)
    drop = raster.reject_mask(clip_pts, sc.mesh.tris, near_eps=1e-5)
    surv = ~drop[np.where(clip_cov, g_c.tri, 0)]
    both = rej_cov & clip_cov & surv
    check('every pixel still covered whose CLIP winner survived rejection '
          'keeps the same id', bool(np.all(g_c.tri[both] == g_r.tri[both])),
          f'{int(both.sum())} px compared')
    check('reject_mask drops at least one floor triangle and no triangle '
          'wholly in front of the near plane',
          bool(drop[floor].any())
          and not bool(drop[(clip_pts[sc.mesh.tris][:, :, 2]
                             + clip_pts[sc.mesh.tris][:, :, 3] >= 1e-5)
                            .all(axis=1)
                            & ((np.abs(clip_pts[sc.mesh.tris][:, :, :2])
                                <= 6.4 * clip_pts[sc.mesh.tris][:, :, 3:4])
                               .all(axis=(1, 2)))
                            & (clip_pts[sc.mesh.tris][:, :, 2]
                               <= clip_pts[sc.mesh.tris][:, :, 3])
                            .all(axis=1)].any()))
    # (c) the guard band is the item's fixed 6.4 half-screens
    eye4 = np.eye(4, dtype=np.float32)
    t = np.array([[0, 1, 2]], np.int32)
    for xf, expect_rej in ((7.0, True), (6.0, False)):
        v = np.array([[-0.5, -0.5, 0.0], [xf, -0.5, 0.0], [0.0, 0.5, 0.0]],
                     np.float32)
        g_c = raster.GBuffer(32, 32)
        raster.rasterize(v, t, eye4, 32, 32, gbuf=g_c)
        g_r = raster.GBuffer(32, 32)
        raster.rasterize(v, t, eye4, 32, 32, gbuf=g_r,
                         opts=raster.RasterOpts(reject=True))
        drawn_c = bool((g_c.tri >= 0).any())
        drawn_r = bool((g_r.tri >= 0).any())
        if expect_rej:
            check(f'a triangle with a vertex at NDC x = {xf} (past the 6.4 '
                  'guard) renders under CLIP and is absent under REJECT',
                  drawn_c and not drawn_r)
        else:
            check(f'a triangle with a vertex at NDC x = {xf} (inside the '
                  'guard) survives both', drawn_c and drawn_r
                  and bool(np.array_equal(g_c.tri, g_r.tri)))
    # the far plane is part of the rule
    v = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 1.5], [0.0, 0.5, 0.0]],
                 np.float32)
    g_r = raster.GBuffer(32, 32)
    raster.rasterize(v, t, eye4, 32, 32, gbuf=g_r,
                     opts=raster.RasterOpts(reject=True))
    check('a vertex past the far plane (z > w) drops the triangle whole',
          not bool((g_r.tri >= 0).any()))
    # (d) GPU twin on the near-wall scene with reject=True
    st = base_settings(160, 120)
    sc = FM.SCENES['near_wall'](st)
    vp, opts = _opts_for(sc, 160, 120, reject=True)
    g, _sim = _twin(sc, vp, 160, 120, opts, 'C027 REJECT on the near-wall '
                    'scene')
    check('the twin frame is not vacuous (covered, floor partly gone)',
          bool((g.tri >= 0).any()))
    # the whole frame through render(): REJECT differs from CLIP
    st_c = base_settings(W, H)
    st_c.shadows = False
    st_r = base_settings(W, H, near_clip_mode='REJECT')
    st_r.shadows = False
    a = np.asarray(R.render(FM.SCENES['near_wall'](st_c), st_c))
    b = np.asarray(R.render(FM.SCENES['near_wall'](st_r), st_r))
    check('render(): the near-wall frame under REJECT differs from CLIP '
          '(the vanishing floor)', not np.array_equal(a, b))
    # (e) FM row, presets
    check("the FM row 'whole-triangle near rejection (PS2 VU1)' runs the "
          'near_wall scene', ('whole-triangle near rejection (PS2 VU1)',
                              {'near_clip_mode': 'REJECT'}, 'near_wall')
          in FM.ROWS and 'near_wall' in FM.SCENES)
    check('PS2, PSX and PSX_HIRES carry REJECT',
          all(PRESETS[k]['settings'].get('near_clip_mode') == 'REJECT'
              for k in ('PS2', 'PSX', 'PSX_HIRES')))
    check('build_screen_tris no longer carries the dead clip_far kwarg',
          'clip_far' not in raster.build_screen_tris.__code__.co_varnames)


# ================================================================ C004


def _flat_for(sc, w, h, key, ot=None):
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    return vp, R.polygon_depths(sc.mesh, view, eye, key, ot=ot)


def test_c004_ordering_table():
    """The PlayStation ordering table: whole polygons bucketed by mean
    depth into L integer entries to `ot_far`, far buckets first, the
    first-added on top inside a bucket, polygons past the table or past
    1023x511 px dropped whole -- and Painter's lifted onto the GPU raster
    because the named tie rule made it order-free."""
    st = base_settings(W, H)
    sc = demo_scene(st, with_texture=False)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, W, H)
    # (b) buckets merge as the table shrinks
    counts = []
    for L in (4096, 256, 16):
        fd = R.polygon_depths(sc.mesh, view, eye, 'ORDERING_TABLE',
                              ot=(L, 40.0))
        finite = fd[np.isfinite(fd)]
        n = len(np.unique(finite))
        counts.append(n)
        check(f'ordering table at {L} entries: every finite bucket is an '
              f'integer in 1..{L - 1} and the distinct count ({n}) is <= L',
              n <= L and bool((finite == np.floor(finite)).all())
              and float(finite.min()) >= 1.0 and float(finite.max()) < L)
    check('the distinct bucket count is non-increasing as the table shrinks '
          '(4096 -> 256 -> 16)', counts[0] >= counts[1] >= counts[2],
          str(counts))
    cent = R.polygon_depths(sc.mesh, view, eye, 'CENTROID')
    fd16 = R.polygon_depths(sc.mesh, view, eye, 'ORDERING_TABLE',
                            ot=(16, 40.0))
    check('the bucket is floor(mean depth * L / far) of the CENTROID '
          'reduction (one op per statement, float32)',
          bool(np.array_equal(fd16, np.floor(
              (cent.astype(np.float32) * np.float32(16)) / np.float32(40.0)))))

    def frame(scene_key, **kw):
        st = base_settings(W, H, **kw)
        st.shadows = False
        sc = FM.SCENES[scene_key](st)
        img, out = _capture(R.render, sc, st)
        return np.asarray(img), out

    img_c, _o = frame('demo', depth_sort='PAINTERS')
    img_16, _o = frame('demo', depth_sort='PAINTERS',
                       painters_key='ORDERING_TABLE', ot_length=16)
    check('the picture under 16 buckets differs from CENTROID Painter\'s '
          '(bucket-order swaps)', not np.array_equal(img_c, img_16))
    g_c = _cpu_gbuf(sc, vp, W, H, raster.RasterOpts(), flat_depth=cent)
    g_16 = _cpu_gbuf(sc, vp, W, H, raster.RasterOpts(), flat_depth=fd16)
    check('and the covered MASK is identical when nothing is dropped',
          bool(np.array_equal(g_c.tri >= 0, g_16.tri >= 0)))
    # (c) the far wall
    fd_far = R.polygon_depths(sc.mesh, view, eye, 'ORDERING_TABLE',
                              ot=(4096, 8.0))
    check('ot_far 8.0 on the demo scene puts polygons past the table at '
          '+inf (the floor reaches ~12 units)', bool(np.isinf(fd_far).any())
          and bool(np.isfinite(fd_far).any()))
    g_40 = _cpu_gbuf(sc, vp, W, H, raster.RasterOpts(), flat_depth=R.polygon_depths(
        sc.mesh, view, eye, 'ORDERING_TABLE', ot=(4096, 40.0)))
    g_8 = _cpu_gbuf(sc, vp, W, H, raster.RasterOpts(), flat_depth=fd_far)
    cov40, cov8 = g_40.tri >= 0, g_8.tri >= 0
    check('the far wall strictly reduces the covered count and never adds a '
          'pixel (the PS1 far pop-in)', int(cov8.sum()) < int(cov40.sum())
          and not bool((cov8 & ~cov40).any()),
          f'{int(cov40.sum())} -> {int(cov8.sum())}')
    # (d) the 1023x511 size drop, measured per SOURCE polygon
    Wb, Hb = 2048, 256
    v = np.array([[-1.0, -1.0, 0.0], [2000.0 / Wb * 2.0 - 1.0, -1.0, 0.0],
                  [-1.0, 100.0 / Hb * 2.0 - 1.0, 0.0]], np.float32)
    t = np.array([[0, 1, 2]], np.int32)
    clip = np.concatenate([v, np.ones((3, 1), np.float32)], axis=1)
    sx, *_rest = raster.build_screen_tris(clip, t, Wb, Hb)
    sx2, *_rest = raster.build_screen_tris(clip, t, Wb, Hb,
                                           size_limit=(1023, 511))
    check('a 2000x100 px triangle is emitted without the limit and dropped '
          'under the 1023x511 ordering-table limit',
          sx.shape[0] == 1 and sx2.shape[0] == 0)
    v_ok = v.copy()
    v_ok[1, 0] = 1000.0 / Wb * 2.0 - 1.0
    clip_ok = np.concatenate([v_ok, np.ones((3, 1), np.float32)], axis=1)
    sx3, *_rest = raster.build_screen_tris(clip_ok, t, Wb, Hb,
                                           size_limit=(1023, 511))
    check('a 1000x100 px triangle survives the limit', sx3.shape[0] == 1)
    # (e) the GPU twin: the kernel learns the flat depth
    w2, h2 = 160, 120
    st2 = base_settings(w2, h2)
    sc2 = demo_scene(st2, with_texture=False)
    for key, ot in (('CENTROID', None), ('NEAREST', None),
                    ('ORDERING_TABLE', (4096, 40.0))):
        vp2, fd = _flat_for(sc2, w2, h2, key, ot)
        g, sim = _twin(sc2, vp2, w2, h2, raster.RasterOpts(),
                       f'C004 Painter\'s {key} on the GPU raster',
                       flat_depth=fd)
        cov = g.tri >= 0
        check(f'C004 {key}: the flat depth round-trips bitwise (zndc '
              'identical)', bool(np.array_equal(sim[2][cov], g.zndc[cov])))
    vp2, fd = _flat_for(sc2, w2, h2, 'ORDERING_TABLE', (4096, 40.0))
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc2.mesh, vp2, w2,
                                                         h2, depth_bits=16)
    sim_flat = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w2, h2,
                                   depth_bits=16, refer=True, flat_depth=fd)
    sim_interp = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w2, h2,
                                     depth_bits=16, refer=True)
    n_flat, n_int = int(sim_flat[6].sum()), int(sim_interp[6].sum())
    check('with the referral on at 16 bits the marked count under the flat '
          'depth is <= the same frame\'s marks under interpolated depth (the '
          'z window is off under hal_flat)', n_flat <= n_int,
          f'flat {n_flat} vs interpolated {n_int}')
    # (f) the door is open: Painter's no longer refuses; the two gate
    # refusals left are printed by name
    seen = {}
    orig = CRA.raster_into_gbuffer

    def rig(*a, **k):
        seen['flat'] = k.get('flat_depth')
        seen['n'] = seen.get('n', 0) + 1
        return orig(*a, **k)
    CRA.raster_into_gbuffer = rig
    try:
        R._GBUF_CACHE.clear()
        st = base_settings(W, H, depth_sort='PAINTERS',
                           painters_key='ORDERING_TABLE')
        st.shadows = False
        st.render_device = 'GPU'
        st.gpu_raster = True
        gpu, out = _capture(R.render, demo_scene(st, with_texture=False), st)
        st_o = base_settings(W, H, debug_pass='OVERDRAW')
        st_o.shadows = False
        st_o.render_device = 'GPU'
        st_o.gpu_raster = True
        n_before = seen.get('n', 0)
        _img, out_o = _capture(R.render, demo_scene(st_o, with_texture=False),
                               st_o)
        n_over = seen.get('n', 0) - n_before
        st_b = base_settings(W, H)
        st_b.shadows = False
        st_b.render_device = 'GPU'
        st_b.gpu_raster = True
        n_before = seen.get('n', 0)
        _img, out_b = _capture(R.render, demo_scene(st_b, with_texture=False),
                               st_b, band=(0, 36))
        n_band = seen.get('n', 0) - n_before
    finally:
        CRA.raster_into_gbuffer = orig
    gate_lines = [ln for ln in out.splitlines()
                  if 'rasterising on the CPU' in ln]
    check('a GPU-device Painter\'s frame knocks on the raster door WITH the '
          'flat depth (the Painter\'s gate is gone) and, without a driver, '
          'its refusal names the gpu module, not Painter\'s',
          seen.get('n', 0) >= 1 and seen.get('flat') is not None
          and len(gate_lines) == 1 and 'gpu module' in gate_lines[0]
          and "Painter" not in gate_lines[0],
          (gate_lines or ['(no gate line)'])[0][:90])
    cpu, _o = frame('demo', depth_sort='PAINTERS',
                    painters_key='ORDERING_TABLE')
    check('and the GPU-device frame is bitwise the CPU frame',
          bool(np.array_equal(np.asarray(gpu), cpu)))
    check('the OVERDRAW census refuses BY NAME (printed) and never knocks',
          n_over == 0 and 'the OVERDRAW census is a sequential count' in out_o,
          out_o.strip().split('\n')[0][:90])
    check('a banded frame refuses BY NAME (printed) and never knocks',
          n_band == 0 and 'a banded frame (workers own their rows)' in out_b,
          out_b.strip().split('\n')[0][:90])
    check('LAST_REFERRED carries the replay count (0 without a driver)',
          CRA.LAST_REFERRED.get('count', None) == 0)
    # (g) FM rows and presets
    rows = {r[0]: r for r in FM.ROWS}
    check("the FM rows name the ordering table, ot_far and ot_length",
          rows.get("Painter's ordering table (PS1)") is not None
          and rows.get("Painter's ordering table, far wall, 256 entries",
                       (None, {}))[1].get('ot_far') == 8.0
          and rows.get("Painter's ordering table, far wall, 256 entries",
                       (None, {}))[1].get('ot_length') == 256
          and rows.get("Painter's ordering table, stacked sheets, 256 "
                       "entries", (None, {}, None))[2]
          == 'ordering_table_stack')
    a, _o = frame('ordering_table_stack', depth_sort='PAINTERS')
    b, _o = frame('ordering_table_stack', depth_sort='PAINTERS',
                  painters_key='ORDERING_TABLE', ot_length=256)
    c, _o = frame('ordering_table_stack', depth_sort='PAINTERS',
                  painters_key='ORDERING_TABLE', ot_length=4096)
    check('the stacked sheets mis-sort at 256 entries (one bucket, the '
          'first-added on top) and sort correctly at 4096',
          not np.array_equal(a, b) and bool(np.array_equal(a, c)))
    for k in ('PSX', 'PSX_HIRES'):
        s = PRESETS[k]['settings']
        check(f'{k} carries the ordering table at 4096 entries to 40 units',
              s.get('painters_key') == 'ORDERING_TABLE'
              and s.get('ot_length') == 4096 and s.get('ot_far') == 40.0
              and s.get('depth_sort') == 'PAINTERS')
    check('SATURN, THREEDO and DOOM keep CENTROID (not table-sorted machines)',
          all(PRESETS[k]['settings'].get('painters_key', 'CENTROID')
              == 'CENTROID' for k in ('SATURN', 'THREEDO', 'DOOM')))
    # the Painter's fallback of the encodings
    _img, out = frame('demo', depth_sort='PAINTERS',
                      depth_encoding='N64_FLOAT18')
    img_plain, _o = frame('demo', depth_sort='PAINTERS')
    check('a depth encoding under Painter\'s falls back to LINEAR by name '
          'and renders the plain Painter\'s frame bitwise',
          "is a z-buffer rule; Painter's sort keeps LINEAR this frame" in out
          and bool(np.array_equal(_img, img_plain)))


# ================================================================ C007


def test_c007_n64_float18():
    """The RDP's 14-bit piecewise-floating code of an 18-bit z: 64-unit
    steps over the near half, 1-unit steps at the far end -- coarse
    fighting near, clean far, the opposite of a fixed-point buffer."""
    # (b) the unit law on the code table
    z18 = np.arange(262144, dtype=np.int32)
    code = raster._n64_code(z18)
    check('the 18-bit ramp maps onto exactly 16384 codes (the 14-bit table: '
          '6 x 2048 + 4096)', len(np.unique(code)) == 16384,
          str(len(np.unique(code))))
    near = z18 < 0x20000
    check('near segment: every key is a multiple of 64',
          bool((code[near] % 64 == 0).all()))
    a = z18[near][:-20]
    b = a + 20
    same_block = (a // 64) == (b // 64)
    check('two near values 20 units apart share a key exactly when they '
          'share a 64-block', bool(np.array_equal(
              raster._n64_code(a) == raster._n64_code(b), same_block)))
    far = z18 >= 258048
    check('far segment (z18 >= 0x3F000): every integer is its own key',
          bool(np.array_equal(code[far], z18[far])))
    e = z18 >> 11
    check('the segment steps are 64, 32, 16, 8, 4, 2, 1 by e = z18 >> 11 '
          '(angrylion z_build_com_table)',
          all(int(np.diff(np.unique(code[(e >= lo) & (e < hi)])).min())
              == step
              and int(np.diff(np.unique(code[(e >= lo) & (e < hi)])).max())
              == step
              for lo, hi, step in
              ((0, 0x40, 64), (0x40, 0x60, 32), (0x60, 0x70, 16),
               (0x70, 0x78, 8), (0x78, 0x7c, 4), (0x7c, 0x7e, 2),
               (0x7e, 0x80, 1))))
    opts = raster.RasterOpts(enc='N64_FLOAT18')
    zz = np.linspace(-1.0, 1.0, 5001).astype(np.float32)
    key, dec = raster.encode_depth(zz, np.ones_like(zz), opts, 24)
    zn = np.clip(zz * np.float32(0.5) + np.float32(0.5), 0.0, 1.0)
    z18r = np.floor(zn * np.float32(262143.0)).astype(np.int32)
    check('encode_depth: key = the code of floor(zn * 262143), dec = its '
          'NDC value, both exact float32 integers / one rounding each',
          bool(np.array_equal(key, raster._n64_code(z18r).astype(np.float32)))
          and bool(np.array_equal(dec, (key / np.float32(262143.0))
                                  * np.float32(2.0) - np.float32(1.0)))
          and bool((key == np.floor(key)).all()))
    check('decode_key inverts the same way', bool(np.array_equal(
        raster.decode_key(key, opts, 24), dec)))
    # (c) the picture: the demo scene's depth plane moves, its coverage
    # does not; the fighting-sheet scene's picture differs from LINEAR-24
    # and LINEAR-16 with an identical covered mask
    st = base_settings(W, H)
    sc = demo_scene(st, with_texture=False)
    vp, o_n64 = _opts_for(sc, W, H, enc='N64_FLOAT18')
    _vp, o_lin = _opts_for(sc, W, H)
    g_n = _cpu_gbuf(sc, vp, W, H, o_n64)
    g_l = _cpu_gbuf(sc, vp, W, H, o_lin)
    g_16 = _cpu_gbuf(sc, vp, W, H, o_lin, depth_bits=16)
    check('demo scene: the stored depth under N64_FLOAT18 differs from '
          'LINEAR-24 and from LINEAR-16, the covered mask is identical (an '
          'encoding moves no silhouette)',
          not np.array_equal(g_n.zndc, g_l.zndc)
          and not np.array_equal(g_n.zndc, g_16.zndc)
          and bool(np.array_equal(g_n.tri >= 0, g_l.tri >= 0)))
    check('demo scene: zkey is allocated under the encoding (integer codes, '
          '+inf where empty) and None under LINEAR',
          g_n.zkey is not None and g_l.zkey is None
          and bool(np.isinf(g_n.zkey[g_n.tri < 0]).all())
          and bool((g_n.zkey[g_n.tri >= 0]
                    == np.floor(g_n.zkey[g_n.tri >= 0])).all()))

    def frame(scene_key, **kw):
        st = base_settings(W, H, **kw)
        st.shadows = False
        return np.asarray(R.render(FM.SCENES[scene_key](st), st))
    f_n = frame('depth_fight_pair', depth_encoding='N64_FLOAT18')
    f_l = frame('depth_fight_pair')
    f_16 = frame('depth_fight_pair', depth_precision=16)
    check('the fighting-sheet frame under N64_FLOAT18 differs from LINEAR-24 '
          'and from LINEAR-16', not np.array_equal(f_n, f_l)
          and not np.array_equal(f_n, f_16))
    # (d) the two-plane fighting rig: coarse near, clean far
    w2, h2 = 64, 48
    for seg, z_a, slant, expect_both in (('near', -0.5, 0.1137, True),
                                         ('far', 0.985, 0.008, False)):
        z_b = np.float32(z_a) - np.float32(3e-5)      # B nearer by 3e-5 NDC
        mesh = _fight_rig(float(z_a), float(z_b), w2, h2, slant=slant)
        eye4 = np.eye(4, dtype=np.float32)
        wp = raster.WParams(0.1, 100.0, 0.0, 0.0, False)
        o = raster.RasterOpts(enc='N64_FLOAT18', wparams=wp)
        g = raster.GBuffer(w2, h2)
        raster.rasterize(mesh.verts, mesh.tris, eye4, w2, h2, gbuf=g, opts=o)
        ga = raster.GBuffer(w2, h2)
        raster.rasterize(mesh.verts, mesh.tris[:2], eye4, w2, h2, gbuf=ga,
                         opts=o)
        gb = raster.GBuffer(w2, h2)
        raster.rasterize(mesh.verts, mesh.tris[2:], eye4, w2, h2, gbuf=gb,
                         opts=o, subset=None)
        ids = set(np.unique(g.tri[g.tri >= 0]).tolist())
        if expect_both:
            check(f'{seg} segment (64-unit steps): two planes 3e-5 NDC apart '
                  'FIGHT -- both ids appear', ids & {0, 1} and ids & {2, 3},
                  str(ids))
            tie = ga.zkey == gb.zkey
            check('where the keys tie the named rule picks the lower id '
                  '(plane A)', bool(np.isin(g.tri[tie], [0, 1]).all()),
                  f'{int(tie.sum())} tied px')
            lower = gb.zkey < ga.zkey
            check('where plane B\'s key is lower B wins',
                  bool(np.isin(g.tri[lower], [2, 3]).all()),
                  f'{int(lower.sum())} px')
        else:
            check(f'{seg} segment (1-unit steps): the nearer plane wins '
                  'cleanly -- only its ids appear past the 0x7e boundary',
                  ids <= {2, 3} and bool(ids), str(ids))
    # (e) the GPU twin in the simulator: keys and decoded depths bitwise
    w3, h3 = 160, 120
    st3 = base_settings(w3, h3)
    sc3 = demo_scene(st3, with_texture=False)
    vp3, o3 = _opts_for(sc3, w3, h3, enc='N64_FLOAT18')
    g3, sim3 = _twin(sc3, vp3, w3, h3, o3, 'C007 N64_FLOAT18 on the demo '
                     'scene', zkey_bitwise=True)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc3.mesh, vp3, w3,
                                                         h3, opts=o3)
    simr = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w3, h3,
                               depth_bits=24, refer=True, opts=o3)
    covered = int((g3.tri >= 0).sum())
    n_mark = int(simr[6].sum())
    check('demo scene: the two-candidate referral marks stay below the 10 % '
          'bail (a lone candidate never marks)',
          n_mark <= 0.10 * covered, f'{n_mark} of {covered} covered')
    check('demo scene: the referral run picks the same winners',
          bool(np.array_equal(simr[0], g3.tri)))
    # the rig: marks exist and every marked pixel's replay is the CPU's id
    z_a = -0.5
    z_b = float(np.float32(z_a) - np.float32(3e-5))
    mesh = _fight_rig(z_a, z_b, w2, h2)
    scr = _Sc(mesh)
    eye4 = np.eye(4, dtype=np.float32)
    wp = raster.WParams(0.1, 100.0, 0.0, 0.0, False)
    o = raster.RasterOpts(enc='N64_FLOAT18', wparams=wp)
    g, sim = _twin(scr, eye4, w2, h2, o, 'C007 the fighting rig',
                   zkey_bitwise=True, refer=True)
    mark = sim[6]
    check('the fighting rig marks pixels (two keys within a step, a pre-floor '
          'value near a boundary)', int(mark.sum()) > 0, str(int(mark.sum())))
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(mesh, eye4, w2, h2,
                                                         opts=o)
    corners, c_side, bins, b_side, tiles, tw, th = CRA.pack_raster_inputs(
        sx, sy, iw, z, bw, src, tmap, w2, h2)
    pys, pxs = np.nonzero(mark)
    r_tri, _rb, _rlb, r_z, _rf = CRA.replay_pixels(
        pxs, pys, sx, sy, iw, z, bw, np.asarray(src, np.int64), tiles,
        bins.reshape(-1), tw, 'NONE', 24, opts=o)
    check('every marked pixel\'s replay (encode_depth on the CPU) equals '
          'the CPU fill\'s id and key',
          bool(np.array_equal(r_tri, g.tri[pys, pxs]))
          and bool(np.array_equal(r_z, g.zkey[pys, pxs])))
    # a LONE fragment at a step boundary is never marked
    k = 1000
    z_edge = float(np.float32(64 * k / 262143.0) * np.float32(2.0)
                   - np.float32(1.0))
    lone = _fight_rig(z_edge, 0.9, w2, h2)
    lone.tris = lone.tris[:2]
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(lone, eye4, w2, h2,
                                                         opts=o)
    siml = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w2, h2,
                               depth_bits=24, refer=True, opts=o)
    check('a LONE fragment sitting on a 64-step boundary is NOT marked (the '
          'rule needs two candidates)', int(siml[6].sum()) == 0
          and bool((siml[0] >= 0).any()), str(int(siml[6].sum())))
    # the batched and loop fills agree under the encoding
    g_loop = raster.GBuffer(w3, h3)
    raster.rasterize(sc3.mesh.verts, sc3.mesh.tris, vp3, w3, h3, gbuf=g_loop,
                     opts=o3, batched=False)
    g_bat = raster.GBuffer(w3, h3)
    raster.rasterize(sc3.mesh.verts, sc3.mesh.tris, vp3, w3, h3, gbuf=g_bat,
                     opts=o3, batched=True)
    check('fill and fill_batched agree bitwise under the encoding (tri, zkey, '
          'depth)', bool(np.array_equal(g_loop.tri, g_bat.tri))
          and bool(np.array_equal(g_loop.zkey, g_bat.zkey))
          and bool(np.array_equal(g_loop.depth, g_bat.depth)))
    # the host decode through gbuffer_into
    ids = np.stack([sim3[1][:, :, 0], sim3[1][:, :, 1],
                    1.0 - sim3[1][:, :, 0] - sim3[1][:, :, 1],
                    sim3[0].astype(np.float32)], -1).astype(np.float32)
    aux = np.stack([sim3[2], sim3[3].astype(np.float32), sim3[4],
                    np.zeros_like(sim3[2])], -1).astype(np.float32)
    g_rec = raster.GBuffer(w3, h3)
    CRA.gbuffer_into(g_rec, ids, aux, opts=o3, depth_bits=24)
    check('gbuffer_into decodes the key plane on the host: zkey, depth and '
          'zndc bitwise the CPU fill\'s',
          bool(np.array_equal(g_rec.zkey, g3.zkey))
          and bool(np.array_equal(g_rec.depth, g3.depth))
          and bool(np.array_equal(g_rec.zndc, g3.zndc)))
    # (f) FM rows, the preset, the G-buffer cache carries zkey
    check('the FM rows carry N64_FLOAT18 on the demo and fighting-sheet '
          'scenes', ('N64 18-bit floating z-buffer',
                     {'depth_encoding': 'N64_FLOAT18'}, 'demo') in FM.ROWS
          and ('N64 18-bit floating z-buffer, fighting sheet',
               {'depth_encoding': 'N64_FLOAT18'}, 'depth_fight_pair')
          in FM.ROWS)
    check('the N64 preset carries N64_FLOAT18',
          PRESETS['N64']['settings'].get('depth_encoding') == 'N64_FLOAT18')
    R._GBUF_CACHE.clear()
    st = base_settings(W, H, depth_encoding='N64_FLOAT18')
    st.shadows = False
    sc = demo_scene(st, with_texture=False)
    a = np.asarray(R.render(sc, st))
    b = np.asarray(R.render(sc, st))
    check('a second render hits the G-buffer cache (zkey restored) and is '
          'bitwise the first', R._GBUF_STATS.get('last') == 'HIT'
          and bool(np.array_equal(a, b)), str(R._GBUF_STATS.get('last')))


# ================================================================ C026


def _replay_equals_cpu(sc, vp, w, h, opts, label, depth_bits=24):
    """The referral's contract on a fighting scene: marks exist and every
    marked pixel's CPU replay equals the CPU fill's own winner and key."""
    g, sim = _twin(sc, vp, w, h, opts, label, depth_bits=depth_bits,
                   zkey_bitwise=True, refer=True)
    mark = sim[6]
    n = int(mark.sum())
    covered = int((g.tri >= 0).sum())
    over = n > CRA.REFER_BAIL_FRAC * covered
    check(f'{label}: the two-candidate referral marks pixels (> 0); a '
          'whole-frame coincidence like this sheet '
          + ('EXCEEDS the 10 % budget and falls back whole on the driver by '
             'name (measured, recorded)' if over else
             'stays under the 10 % budget'), n > 0,
          f'{n} of {covered} covered ({100.0 * n / max(covered, 1):.1f} %)')
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
        sc.mesh, vp, w, h, depth_bits=depth_bits, opts=opts)
    src_final = np.asarray(src, np.int64)
    if tmap is not None:
        src_final = np.asarray(tmap, np.int64)[src_final]
    _c, _cs, bins, _bs, tiles, tw, _th = CRA.pack_raster_inputs(
        sx, sy, iw, z, bw, src, tmap, w, h)
    pys, pxs = np.nonzero(mark)
    r_tri, _rb, _rlb, r_z, _rf = CRA.replay_pixels(
        pxs, pys, sx, sy, iw, z, bw, src_final, tiles, bins.reshape(-1),
        tw, 'NONE', depth_bits, opts=opts)
    check(f'{label}: every marked pixel\'s replay equals the CPU fill\'s id '
          'and key', bool(np.array_equal(r_tri, g.tri[pys, pxs]))
          and bool(np.array_equal(r_z, g.zkey[pys, pxs])))
    return g, sim


def test_c026_gc_w_encodings():
    """The GameCube's compressed 16-bit Z (leading-ones octaves of a
    24-bit z: 14e2 / 13e3 / 12e4) and the fixed-point W-buffer (Xbox
    NV2A, D3DZB_USEW): fighting by octave, or uniform in distance."""
    # (b) the GC layouts on the full 24-bit ramp
    z24 = np.arange(1 << 24, dtype=np.int32)
    expect = {'GC_14E2': (512, 128, 65536), 'GC_13E3': (1024, 16, 65536),
              'GC_12E4': (2048, 1, 53248)}
    for enc, (lo_step, hi_step, count) in expect.items():
        Mb, Eb = raster.GC_LAYOUTS[enc]
        code = raster._gc_code(z24, Mb, Eb)
        low = code[z24 < 0x800000]
        top = code[z24 >= (0xFFFFFF - 4 * hi_step)]
        d_low = np.unique(np.diff(np.unique(low)))
        d_top = np.unique(np.diff(np.unique(top)))
        n = len(np.unique(code))
        check(f'{enc}: {lo_step}-unit steps below 0x800000 (e = 0), '
              f'{hi_step}-unit steps at the top, {count} codes in all',
              d_low.tolist() == [lo_step] and d_top.tolist() == [hi_step]
              and n == count, f'low {d_low.tolist()}, top {d_top.tolist()}, '
              f'{n} codes')
        check(f'{enc}: every code is the lower bound of its cell (monotone, '
              'code <= z24)', bool((code <= z24).all())
              and bool((np.diff(code) >= 0).all()))
    check('GC_12E4 spends its exponent bits on range, not count: fewer codes '
          'than 2^16', expect['GC_12E4'][2] < 65536)
    # (c) W_FIXED: uniform in distance
    near, far = np.float32(0.1), np.float32(200.0)
    A = np.float32((far + near) / (near - far)) * np.float32(-1.0)
    B = np.float32((2.0 * far * near) / (near - far))
    wp = raster.WParams(near=near, far=far, A=A, B=B, is_ortho=False)
    o_w = raster.RasterOpts(enc='W_FIXED', wparams=wp)
    steps = np.float32(65535.0)
    rng = far - near
    k = np.arange(0, 16000, dtype=np.float32)
    t_mid = (k * np.float32(4.0) + np.float32(0.5)) / steps   # mid-cell, every 4th step
    w_ramp = (near + t_mid * rng).astype(np.float32)
    invw = (np.float32(1.0) / w_ramp).astype(np.float32)
    key_w, dec_w = raster.encode_depth(np.zeros_like(invw), invw, o_w, 16)
    d = np.diff(key_w)
    check('W_FIXED-16 on a w ramp at every fourth step (mid-cell): consecutive '
          'keys differ by exactly 4 everywhere (np.ptp of the differences is '
          '0) -- uniform in distance', float(np.ptp(d)) == 0.0
          and float(d[0]) == 4.0, f'ptp {float(np.ptp(d))}, d0 {float(d[0])}')
    z_lin = A + B * invw
    key_lin = np.round((z_lin * np.float32(0.5) + np.float32(0.5)) * steps)
    d_lin = np.diff(key_lin)
    check('LINEAR-16 over the same w ramp: its steps shrink with distance '
          '(hundreds of codes per step near the camera, none left at the '
          'far end)', float(d_lin[:100].mean()) > 100.0
          and float(d_lin[-100:].mean()) < 1.0,
          f'{float(d_lin[:100].mean()):.1f} -> {float(d_lin[-100:].mean()):.2f}')
    wq = key_w / steps
    wq = wq * rng
    wq = wq + near
    iwq = np.float32(1.0) / wq
    want = A + B * iwq
    check('W_FIXED decodes the quantised eye depth through the camera\'s '
          'own z = A + B / w, float32 statement by statement (NDC stays NDC '
          'for every consumer, monotone in w)',
          bool(np.array_equal(dec_w, want.astype(np.float32)))
          and bool(np.all(np.diff(dec_w) >= 0.0))
          and bool(np.all(np.abs(wq - w_ramp) <= rng / steps)))
    check('W_FIXED caps its bit count at 24 (the NV2A\'s maximum)',
          raster.encoding_uniforms(o_w, 32)['hal_wsteps'] == float((1 << 24) - 1)
          and raster.encoding_uniforms(o_w, 16)['hal_wsteps'] == 65535.0)
    # (d) the picture: the demo scene's depth plane moves under each item
    # with an identical covered mask; the fighting sheet shows each one
    st = base_settings(W, H)
    sc = demo_scene(st, with_texture=False)
    vp, o_lin = _opts_for(sc, W, H)
    g_l = _cpu_gbuf(sc, vp, W, H, o_lin)
    for enc, bits in (('GC_14E2', 24), ('GC_13E3', 24), ('GC_12E4', 24),
                      ('W_FIXED', 16)):
        _vp, o = _opts_for(sc, W, H, enc=enc)
        g = _cpu_gbuf(sc, vp, W, H, o, depth_bits=bits)
        check(f'demo scene under {enc}: the stored depth differs from '
              'LINEAR-24, the covered mask is identical',
              not np.array_equal(g.zndc, g_l.zndc)
              and bool(np.array_equal(g.tri >= 0, g_l.tri >= 0)))

    def frame(scene_key, **kw):
        st = base_settings(W, H, **kw)
        st.shadows = False
        img, out = _capture(R.render, FM.SCENES[scene_key](st), st)
        return np.asarray(img), out
    f_l, _o = frame('depth_fight_pair')
    for kw in ({'depth_encoding': 'GC_14E2'}, {'depth_encoding': 'GC_12E4'},
               {'depth_encoding': 'W_FIXED', 'depth_precision': 16}):
        f, _o = frame('depth_fight_pair', **kw)
        check(f'the fighting-sheet frame under {kw} differs from LINEAR-24 '
              '(the sheet fights where the step is coarser than its 5e-4 '
              'gap)', not np.array_equal(f, f_l))
    f13, _o = frame('depth_fight_pair', depth_encoding='GC_13E3')
    check('GC_13E3 (20 bits far) resolves the 5e-4 sheet as LINEAR-24 does: '
          'the finest far layout, its picture equal here, its depth plane '
          'not', bool(np.array_equal(f13, f_l)))
    # (e) the orthographic fallback by name
    f_o, out = frame('ortho', depth_encoding='W_FIXED')
    f_o_lin, _o = frame('ortho')
    check('an orthographic camera under W_FIXED renders bitwise the LINEAR '
          'frame and the console names the fallback',
          bool(np.array_equal(f_o, f_o_lin))
          and 'depth encoding W_FIXED needs a perspective camera: LINEAR '
          'depth this frame' in out)
    f_o2, out2 = frame('ortho', depth_encoding='VOODOO_W16')
    check('the same fallback covers VOODOO_W16', bool(np.array_equal(
        f_o2, f_o_lin)) and 'VOODOO_W16 needs a perspective camera' in out2)
    # (f) the GPU twin in the simulator, four items, keys bitwise
    w3, h3 = 160, 120
    st3 = base_settings(w3, h3)
    sc3 = demo_scene(st3, with_texture=False)
    for enc, bits in (('GC_14E2', 24), ('GC_13E3', 24), ('GC_12E4', 24),
                      ('W_FIXED', 16)):
        vp3, o3 = _opts_for(sc3, w3, h3, enc=enc)
        g3, sim3 = _twin(sc3, vp3, w3, h3, o3, f'C026 {enc} on the demo '
                         'scene', depth_bits=bits, zkey_bitwise=True)
        sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
            sc3.mesh, vp3, w3, h3, depth_bits=bits, opts=o3)
        simr = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w3, h3,
                                   depth_bits=bits, refer=True, opts=o3)
        covered = int((g3.tri >= 0).sum())
        n_mark = int(simr[6].sum())
        check(f'C026 {enc}: the referral marks stay under the 10 % bail on '
              'the demo scene', n_mark <= 0.10 * covered,
              f'{n_mark} of {covered}')
    # the fighting sheet under W_FIXED-16: marks and the replay
    sc4 = FM.SCENES['depth_fight_pair'](base_settings(w3, h3))
    vp4, o4 = _opts_for(sc4, w3, h3, enc='W_FIXED')
    _replay_equals_cpu(sc4, vp4, w3, h3, o4, 'C026 W_FIXED-16 on the '
                       'fighting sheet', depth_bits=16)
    sc5 = FM.SCENES['depth_fight_pair'](base_settings(w3, h3))
    vp5, o5 = _opts_for(sc5, w3, h3, enc='GC_14E2')
    _replay_equals_cpu(sc5, vp5, w3, h3, o5, 'C026 GC_14E2 on the '
                       'fighting sheet')
    # (g) FM rows and presets
    check('the FM rows carry GC_14E2 and W_FIXED-16 on the fighting sheet',
          ('GameCube 14e2 compressed z', {'depth_encoding': 'GC_14E2'},
           'depth_fight_pair') in FM.ROWS
          and ('W-buffer, 16-bit fixed (Xbox)',
               {'depth_encoding': 'W_FIXED', 'depth_precision': 16},
               'depth_fight_pair') in FM.ROWS)
    check('GAMECUBE carries GC_14E2 and XBOX W_FIXED (the design\'s preset '
          'decisions)', PRESETS['GAMECUBE']['settings'].get('depth_encoding')
          == 'GC_14E2'
          and PRESETS['XBOX']['settings'].get('depth_encoding') == 'W_FIXED')


# ================================================================ C075


def _mame_wfloat(t):
    """MAME voodoo_render.cpp compute_wfloat, transcribed on the INTEGER
    16.32 iterated w (the oracle of the float road)."""
    import math
    iterw = int(math.floor(t * (1 << 32)))
    if (iterw >> 32) != 0:
        return 0x0000
    if (iterw & 0xffff0000) == 0:
        return 0xffff
    exp = 32 - iterw.bit_length()
    return min(((exp << 12) | ((~iterw >> (19 - exp)) & 0xfff)) + 1, 65535)


def test_c075_voodoo_w16():
    """3dfx's 16-bit floating W: 1/w as a 4-bit octave and 12 inverted
    mantissa bits, constant relative precision per octave, a hard wall
    at the 16th."""
    # (b) unit laws with near a power of two, so t = invw * near is exact
    near = np.float32(0.125)
    wp = raster.WParams(near=near, far=np.float32(100.0), A=np.float32(0.0),
                        B=np.float32(0.0), is_ortho=False)
    o = raster.RasterOpts(enc='VOODOO_W16', wparams=wp)

    def keys_of(w):
        w = np.asarray(w, np.float32)
        invw = (np.float32(1.0) / w).astype(np.float32)
        k, _d = raster.encode_depth(np.zeros_like(invw), invw, o, 24)
        return k
    kk = np.arange(1, 16)
    check('w = near * 2^k (k = 1..15) gives the octave tops k << 12 exactly',
          bool(np.array_equal(keys_of(near * (2.0 ** kk)),
                              (kk << 12).astype(np.float32))))
    check('w = near gives 0, w < near gives 0, w >= 65536 near gives 65535 '
          '(equality included)',
          float(keys_of([near])[0]) == 0.0
          and float(keys_of([near * 0.5])[0]) == 0.0
          and float(keys_of([near * 65536.0])[0]) == 65535.0
          and float(keys_of([near * 70000.0])[0]) == 65535.0)
    w_ramp = near * np.exp(np.linspace(0.0, np.log(70000.0), 20000))
    k_ramp = keys_of(w_ramp)
    check('keys are non-decreasing along a w ramp (monotone across every '
          'octave and the far wall)', bool((np.diff(k_ramp) >= 0).all()))
    e = 5
    j = np.arange(4096)
    t_cells = ((4096.0 + j + 0.5) / float(2 ** (13 + e))).astype(np.float32)
    k_cells = keys_of(near / t_cells)
    check('one octave holds exactly 4096 codes (mid-cell samples, each its '
          'own key, the key falling by one per cell as t rises and w '
          'falls)', len(np.unique(k_cells)) == 4096
          and bool((np.diff(k_cells) == -1).all()))
    # the ORACLE: MAME's integer compute_wfloat at the 16 octave boundaries
    # and 256 interior mid-cell samples
    ts = [2.0 ** -k for k in range(0, 16)]
    for ee in range(16):
        for jj in range(0, 4096, 256):
            ts.append((4096.0 + jj + 0.5) / float(2 ** (13 + ee)))
    ts = np.array(ts, np.float32)
    got = keys_of(near / ts)
    want = np.array([_mame_wfloat(float(t)) for t in ts], np.float32)
    check('the float road pins MAME\'s integer compute_wfloat at the 16 '
          'octave boundaries and 256 mid-cell samples (exact)',
          bool(np.array_equal(got, want)),
          f'{int((got != want).sum())} of {ts.size} differ')
    # (c) constant relative precision per octave
    w0 = near * (2.0 ** (np.arange(1, 16) + 0.37))
    ka = keys_of(w0)
    kb = keys_of(w0 * (1.0 + 2.0 ** -11))
    kc = keys_of(w0 * (1.0 + 2.0 ** -14))
    check('w and w (1 + 2^-11) get distinct keys at EVERY octave (a 16-bit '
          'linear buffer merges them past a few units)',
          bool((ka != kb).all()))
    check('w and w (1 + 2^-14) merge in every octave (under a quarter of a '
          'cell; the spec\'s 2^-13 sits at exactly half a cell at an '
          'octave\'s bottom)', bool((ka == kc).all()))
    z_lin16 = np.round(((np.float32(1.0) - np.float32(0.125) / w0)
                        * np.float32(0.5) + np.float32(0.5))
                       * np.float32(65535.0))
    z_lin16b = np.round(((np.float32(1.0) - np.float32(0.125)
                          / (w0 * (1.0 + 2.0 ** -11)))
                         * np.float32(0.5) + np.float32(0.5))
                        * np.float32(65535.0))
    check('a 16-bit linear z-buffer merges the (1 + 2^-11) pair in the far '
          'octaves', bool((z_lin16 == z_lin16b)[-6:].all()))
    check('the clamp merges the 16th octave\'s top cell with the far wall '
          '(disclosed): the key just under 65536 near is 65535 too',
          float(keys_of([near * 65535.9])[0]) == 65535.0)
    dec = raster.decode_key(np.array([0.0, 4096.0, 65535.0], np.float32),
                            raster.RasterOpts(enc='VOODOO_W16', wparams=raster.WParams(
                                near, np.float32(100.0), np.float32(0.5),
                                np.float32(-0.1), False)), 24)
    check('decode_key: 0 -> -1 (at the near plane), 65535 -> +1 (the far '
          'wall), an octave top decodes through z = A + B / w',
          float(dec[0]) == -1.0 and float(dec[2]) == 1.0
          and abs(float(dec[1]) - (0.5 - 0.1 / (near * 2.0))) < 1e-6)
    # (d) the picture: depth plane on the demo, the fighting sheet
    st = base_settings(W, H)
    sc = demo_scene(st, with_texture=False)
    vp, o_lin = _opts_for(sc, W, H)
    _vp, o_v = _opts_for(sc, W, H, enc='VOODOO_W16')
    g_l = _cpu_gbuf(sc, vp, W, H, o_lin)
    g_16 = _cpu_gbuf(sc, vp, W, H, o_lin, depth_bits=16)
    g_v = _cpu_gbuf(sc, vp, W, H, o_v)
    check('demo scene under VOODOO_W16: the stored depth differs from '
          'LINEAR-16 and LINEAR-24, the covered mask is identical',
          not np.array_equal(g_v.zndc, g_l.zndc)
          and not np.array_equal(g_v.zndc, g_16.zndc)
          and bool(np.array_equal(g_v.tri >= 0, g_l.tri >= 0)))
    floor = g_v.tri >= 0
    floor &= sc.mesh.mat_index[np.where(floor, g_v.tri, 0)] == 0
    octaves = np.unique((g_v.zkey[floor].astype(np.int64) - 1) >> 12)
    check('the demo floor spans three octaves of w (5..14 units over a 0.1 '
          'near plane)', octaves.size >= 3, str(octaves.tolist()))

    def frame(scene_key, **kw):
        st = base_settings(W, H, **kw)
        st.shadows = False
        return np.asarray(R.render(FM.SCENES[scene_key](st), st))
    f_v = frame('depth_fight_pair', depth_encoding='VOODOO_W16')
    f_l = frame('depth_fight_pair')
    f_16 = frame('depth_fight_pair', depth_precision=16)
    check('the fighting-sheet frame under VOODOO_W16 differs from LINEAR-24 '
          'and from LINEAR-16', not np.array_equal(f_v, f_l)
          and not np.array_equal(f_v, f_16))
    # (e) the GPU twin in the simulator
    w3, h3 = 160, 120
    st3 = base_settings(w3, h3)
    sc3 = demo_scene(st3, with_texture=False)
    vp3, o3 = _opts_for(sc3, w3, h3, enc='VOODOO_W16')
    g3, _sim3 = _twin(sc3, vp3, w3, h3, o3, 'C075 VOODOO_W16 on the demo '
                      'scene', zkey_bitwise=True)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
        sc3.mesh, vp3, w3, h3, opts=o3)
    simr = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w3, h3,
                               depth_bits=24, refer=True, opts=o3)
    covered = int((g3.tri >= 0).sum())
    n_mark = int(simr[6].sum())
    check('C075: the referral marks stay under the 10 % bail on the demo '
          'scene', n_mark <= 0.10 * covered, f'{n_mark} of {covered}')
    sc4 = FM.SCENES['depth_fight_pair'](base_settings(w3, h3))
    vp4, o4 = _opts_for(sc4, w3, h3, enc='VOODOO_W16')
    _replay_equals_cpu(sc4, vp4, w3, h3, o4, 'C075 VOODOO_W16 on the '
                       'fighting sheet')
    # a LONE surface crossing octave boundaries is never marked: the
    # demo floor alone (three octaves)
    floor_tris = np.nonzero(sc3.mesh.mat_index == 0)[0].astype(np.int32)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
        sc3.mesh, vp3, w3, h3, subset=floor_tris, opts=o3)
    siml = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w3, h3,
                               depth_bits=24, refer=True, opts=o3)
    check('a LONE surface crossing three octave boundaries is never marked '
          '(the rule needs two candidates)', int(siml[6].sum()) == 0
          and bool((siml[0] >= 0).any()), str(int(siml[6].sum())))
    # (f) the FM row and the preset
    check("the FM row 'Voodoo 16-bit floating W-buffer' runs the fighting "
          'sheet', ('Voodoo 16-bit floating W-buffer',
                    {'depth_encoding': 'VOODOO_W16'}, 'depth_fight_pair')
          in FM.ROWS)
    check('the VOODOO preset carries VOODOO_W16 (GLQuake and most Glide '
          'titles ran the W-buffer)',
          PRESETS['VOODOO']['settings'].get('depth_encoding') == 'VOODOO_W16')


# ---- wave 2 ----
# RAST-A2 (1.90.0): C012 the console vertex formats, C127 REYES jittered
# subsamples, C038 the DS rear-plane depth bitmap, C001 N64 coverage AA
# and the VI scan-out filter. Appended after wave 1's tests; nothing
# above this marker is this section's to edit.


def test_w2_c012_vertex_quantize():
    """C012: `quantize_mesh` rounds the mesh to the console's vertex
    format on a world lattice -- the grid law, the normal grids, the PS1
    UV truncation, 8-bit colours, the recomputed face normals never
    flipped -- monotone crunch, idempotence and re-dialling from the
    source, the demo picture moved once and stable, the GPU twin on the
    quantised mesh (the raster reads the same arrays), the FM rows and
    the presets."""
    st = base_settings(W, H)
    sc = demo_scene(st, with_texture=True)
    m = sc.mesh
    # (a) identity: the module pin (test_a_identity_at_defaults) and the
    # function's own NONE road
    check('C012: mode NONE returns the mesh itself (no copy, no rounding)',
          raster.quantize_mesh(m, 'NONE', 64.0) is m)
    st_d = base_settings(W, H)
    check("C012: the new settings default to NONE / 64.0",
          (st_d.vertex_quantize, st_d.vertex_units) == ('NONE', 64.0))
    # (b) the grid laws
    q = raster.quantize_mesh(m, 'PS1', 64.0)
    check('C012 PS1 at 64 units: every quantised coordinate times 64 is an '
          'integer (the 16-bit lattice)',
          bool(np.array_equal(q.verts * 64.0, np.round(q.verts * 64.0)))
          and q is not m)
    check('C012 PS1: the quantised positions are clipped to the 16-bit '
          'range', bool((np.abs(q.verts * 64.0) <= 32768.0).all()))
    ln = np.linalg.norm(q.normals, axis=1)
    check('C012 PS1: every normal is unit length within 1e-6',
          float(np.abs(ln - 1.0).max()) < 1e-6,
          f'{float(np.abs(ln - 1.0).max()):.2e}')
    n_src = np.round(np.asarray(m.normals, np.float32) * np.float32(4096.0))
    re = n_src / np.float32(4096.0)
    re = re / np.maximum(np.linalg.norm(re, axis=1, keepdims=True), 1e-30)
    check('C012 PS1: normals * 4096 were integral before the normalisation '
          '(re-derived from the source)',
          float(np.abs(re.astype(np.float32) - q.normals).max()) < 1e-6)
    qn = raster.quantize_mesh(m, 'N64', 64.0)
    n8 = np.round(np.asarray(m.normals, np.float32) * np.float32(127.0))
    d8 = n8 / np.maximum(np.linalg.norm(n8, axis=1, keepdims=True), 1e-30)
    check("C012 N64: round(n * 127) reproduces the stored normals' direction",
          float(np.abs(d8.astype(np.float32) - qn.normals).max()) < 1e-6)
    check('C012 PS1: UVs are multiples of 1/256 and never above the source '
          '(truncated to the texel)',
          bool(np.array_equal(q.uvs * 256.0, np.floor(q.uvs * 256.0)))
          and bool((q.uvs <= m.uvs + 1e-7).all()))
    check('C012 N64: UVs untouched (the S10.5 grid needs the texture size)',
          qn.uvs is m.uvs)
    if m.colors is not None:
        check('C012: colours are multiples of 1/255 (all four channels)',
              float(np.abs(q.colors * 255.0
                           - np.round(q.colors * 255.0)).max()) < 1e-4)
    dot = (q.face_normals * m.face_normals).sum(1)
    both = (np.linalg.norm(q.face_normals, axis=1) > 0.5) \
        & (np.linalg.norm(m.face_normals, axis=1) > 0.5)
    check("C012: the recomputed face normals never flip against the export's "
          'winding (dot > 0 wherever both exist)', bool((dot[both] > 0).all())
          and bool(both.any()), f'{int((dot[both] <= 0).sum())} flipped')
    check("C012: face normals of lattice-collapsed polygons keep the "
          "export's (never a zero vector, never NaN)",
          bool(np.isfinite(q.face_normals).all()))
    # (c) monotone crunch
    counts = [len(np.unique(raster.quantize_mesh(m, 'PS1', u).verts, axis=0))
              for u in (64.0, 8.0, 2.0)]
    check('C012: the count of distinct positions is non-increasing across '
          'units 64 -> 8 -> 2', counts[0] >= counts[1] >= counts[2]
          and counts[2] < counts[0], str(counts))
    # (d) idempotence and re-dialling
    q8 = raster.quantize_mesh(m, 'PS1', 8.0)
    check('C012: quantising a quantised mesh with the same dial returns it '
          'unchanged (the tag)', raster.quantize_mesh(q8, 'PS1', 8.0) is q8)
    check('C012: mode NONE on a quantised mesh returns the SOURCE',
          raster.quantize_mesh(q8, 'NONE', 8.0) is m)
    q4 = raster.quantize_mesh(q8, 'PS1', 4.0)
    check('C012: re-dialling starts from the source (units 4 from the units-8 '
          'mesh equals units 4 from the original, bitwise)',
          q4._quant_source is m
          and bool(np.array_equal(q4.verts,
                                  raster.quantize_mesh(m, 'PS1', 4.0).verts)))
    # (e) the picture moves once and stays
    st_q = base_settings(W, H)
    st_q.vertex_quantize, st_q.vertex_units = 'PS1', 8.0
    sc_q = demo_scene(st_q, with_texture=False)
    st_p = base_settings(W, H)
    plain = np.asarray(R.render(demo_scene(st_p, with_texture=False), st_p))
    one = np.asarray(R.render(sc_q, st_q))
    mesh_after = sc_q.mesh
    two = np.asarray(R.render(sc_q, st_q))
    check('C012: the demo picture under PS1 at units 8 differs from NONE',
          not np.array_equal(plain, one))
    check("C012: the second render is bitwise the first and the scene's mesh "
          'was replaced once (the same object)',
          bool(np.array_equal(one, two)) and sc_q.mesh is mesh_after
          and getattr(mesh_after, '_quant_tag', None) == ('PS1', 8.0))
    st_q.vertex_quantize = 'NONE'
    back = np.asarray(R.render(sc_q, st_q))
    check('C012: dialling back to NONE restores the plain picture bitwise '
          '(the source mesh returns)', bool(np.array_equal(back, plain)))
    # (f) the GPU twin on the quantised mesh
    w2, h2 = 160, 120
    st2 = base_settings(w2, h2)
    sc2 = demo_scene(st2, with_texture=False)
    sc2.mesh = raster.quantize_mesh(sc2.mesh, 'PS1', 8.0)
    vp2, o2 = _opts_for(sc2, w2, h2)
    _twin(sc2, vp2, w2, h2, o2, 'C012 PS1 at units 8 on the demo scene')
    # (g) the FM rows and the presets
    for row in (('vertex quantisation (PS1: 16-bit, 8-bit uv)',
                 {'vertex_quantize': 'PS1', 'vertex_units': 8.0}, 'textured'),
                ('vertex quantisation (N64: 8-bit normals)',
                 {'vertex_quantize': 'N64', 'vertex_units': 8.0}, 'demo'),
                ('vertex quantisation (PS1) on the crunch rig',
                 {'vertex_quantize': 'PS1', 'vertex_units': 4.0},
                 'quantized_ship')):
        check(f"the FM row '{row[0]}' exists", row in FM.ROWS)
    for k, mode in (('PSX', 'PS1'), ('PSX_HIRES', 'PS1'), ('N64', 'N64'),
                    ('NDS', 'N64')):
        s_ = PRESETS[k]['settings']
        check(f'the {k} preset carries vertex_quantize {mode} at 64 units',
              s_.get('vertex_quantize') == mode
              and s_.get('vertex_units') == 64.0)


def _dilate3(m):
    out = m.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


def test_w2_c127_jitter():
    """C127: REYES's jittered sample positions -- the offsets law (exact
    (n + 0.5) / 4096 - 0.5, never on a cell edge), determinism and
    freshness per (frame, seed), the locality law (a pixel surrounded by
    one triangle cannot change hands), the GPU twin at the TR:16091 bar
    with the replay equal to the CPU at every marked pixel, the inert
    modes, the FM rows and the RenderMan preset."""
    st = base_settings(W, H)
    check("C127: aa_sample_pattern defaults to GRID",
          st.aa_sample_pattern == 'GRID')
    check('C127: RasterOpts() carries no jitter (bitwise 1.89.0; the module '
          'pin renders it)', raster.RasterOpts().jitter is None
          and raster.jitter_planes(raster.RasterOpts(), 8, 8) is None)
    # (d) the offsets law
    key = raster.jitter_key(1, 0)
    check('C127: the (frame, seed) key is 24-bit (rides a float uniform '
          'exactly)', key == 65537 and raster.jitter_key(2 ** 20, 7) < 2 ** 24
          and float(np.float32(raster.jitter_key(2 ** 20, 7)))
          == float(raster.jitter_key(2 ** 20, 7)))
    JX, JY = raster.jitter_offsets(64, 48, key)
    n_x = (JX + np.float32(0.5)) * np.float32(4096.0) - np.float32(0.5)
    n_y = (JY + np.float32(0.5)) * np.float32(4096.0) - np.float32(0.5)
    check('C127: every offset lies in (-0.5, 0.5) exclusive',
          float(np.abs(JX).max()) < 0.5 and float(np.abs(JY).max()) < 0.5
          and JX.dtype == np.float32 and JX.shape == (48, 64))
    check('C127: every offset is exactly (n + 0.5) / 4096 - 0.5 for an '
          'integer n (exact in float32)',
          bool(np.all(n_x == np.round(n_x))) and bool(np.all(n_y == np.round(n_y))))
    check('C127: the offsets are not degenerate (both axes spread over the '
          'cell)', len(np.unique(JX)) > 1000 and len(np.unique(JY)) > 1000)
    jx1, jy1 = raster.jitter_at(np.uint32(5), np.uint32(7), key)
    check('C127: jitter_at agrees with the planes (one hash, two callers)',
          float(jx1) == float(JX[7, 5]) and float(jy1) == float(JY[7, 5]))
    # (b) determinism and freshness on the demo scene at ss = 2
    def _jit_render(frame, seed, pattern='JITTER', mode='SUPERSAMPLE'):
        s_ = base_settings(W, H)
        s_.aa_mode, s_.aa_samples = mode, 4
        s_.aa_sample_pattern, s_.seed = pattern, seed
        sc_ = demo_scene(s_, with_texture=False)
        sc_.frame = frame
        return np.asarray(R.render(sc_, s_))
    a1 = _jit_render(1, 0)
    a2 = _jit_render(1, 0)
    b_seed = _jit_render(1, 1)
    b_frame = _jit_render(2, 0)
    grid = _jit_render(1, 0, pattern='GRID')
    check('C127: the same (frame, seed) renders bitwise twice',
          bool(np.array_equal(a1, a2)))
    check('C127: seed + 1 changes the picture', not np.array_equal(a1, b_seed))
    check('C127: frame + 1 changes the picture', not np.array_equal(a1, b_frame))
    check('C127: JITTER differs from GRID', not np.array_equal(a1, grid))
    # the inert modes: EDGE / ADAPTIVE / ACCUMULATE run their own logic
    for mode in ('EDGE', 'ACCUMULATE'):
        s_a = base_settings(W, H)
        s_a.aa_mode, s_a.aa_samples, s_a.aa_sample_pattern = mode, 4, 'JITTER'
        s_b = base_settings(W, H)
        s_b.aa_mode, s_b.aa_samples = mode, 4
        ia = np.asarray(R.render(demo_scene(s_a, with_texture=False), s_a))
        ib = np.asarray(R.render(demo_scene(s_b, with_texture=False), s_b))
        check(f'C127: JITTER is inert under aa_mode {mode} (bitwise GRID)',
              bool(np.array_equal(ia, ib)))
    # (c) the locality law on the G-buffer ids
    w2, h2 = 160, 120
    st2 = base_settings(w2, h2)
    sc2 = demo_scene(st2, with_texture=False)
    vp2, o_grid = _opts_for(sc2, w2, h2)
    _vp, o_jit = _opts_for(sc2, w2, h2, jitter=(1, 0))
    g_grid = _cpu_gbuf(sc2, vp2, w2, h2, o_grid)
    g_jit = _cpu_gbuf(sc2, vp2, w2, h2, o_jit)
    ids = g_grid.tri
    edges = np.zeros(ids.shape, bool)
    edges[1:, :] |= ids[1:, :] != ids[:-1, :]
    edges[:-1, :] |= ids[1:, :] != ids[:-1, :]
    edges[:, 1:] |= ids[:, 1:] != ids[:, :-1]
    edges[:, :-1] |= ids[:, 1:] != ids[:, :-1]
    changed = g_jit.tri != ids
    check('C127: locality -- a pixel whose neighbours all share its id keeps '
          'it under JITTER (changed & ~dilate3x3(edges) is empty)',
          not bool((changed & ~_dilate3(edges)).any()),
          f'{int((changed & ~_dilate3(edges)).sum())} escaped')
    check('C127: the jitter moves some edge pixels (non-vacuous)',
          int(changed.sum()) > 0, f'{int(changed.sum())} px')
    # (e) the GPU twin and the replay
    _twin(sc2, vp2, w2, h2, o_jit, 'C127 JITTER (frame 1, seed 0) on the '
          'demo scene')
    # the replay at every marked pixel equals the CPU (the offsets ride
    # into replay_pixels through opts.jitter)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
        sc2.mesh, vp2, w2, h2, opts=o_jit)
    simr = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w2, h2,
                               depth_bits=24, refer=True, opts=o_jit)
    mark = simr[6]
    pys, pxs = np.nonzero(mark)
    _c, _cs, bins, _bs, tiles, tw, _th = CRA.pack_raster_inputs(
        sx, sy, iw, z, bw, src, tmap, w2, h2)
    src_final = np.asarray(src, np.int64) if tmap is None \
        else np.asarray(tmap, np.int64)[src]
    r_tri, r_b, _rlb, _rz, _rf = CRA.replay_pixels(
        pxs, pys, sx, sy, iw, z, bw, src_final, tiles, bins.reshape(-1),
        tw, 'NONE', 24, opts=o_jit)
    check('C127: the referral marks stay under the 10 % bail and the '
          "replay at every marked pixel is the CPU fill's own winner",
          int(mark.sum()) <= 0.10 * int((g_jit.tri >= 0).sum())
          and bool(np.array_equal(r_tri, g_jit.tri[pys, pxs]))
          and (pxs.size == 0 or float(np.abs(r_b - g_jit.bary[pys, pxs]).max()) < 1e-6),
          f'{int(mark.sum())} marked')
    ku = CRA.kernel_uniforms(o_jit, 24, False)
    check('C127: the kernel uniforms carry hal_jit 1 and the 24-bit key',
          ku['hal_jit'] == 1.0 and ku['hal_jit_key'] == float(key))
    for src_name, src in (('FRAGMENT_SOURCE', CRA.FRAGMENT_SOURCE),
                          ('COMPUTE_SOURCE', CRA.COMPUTE_SOURCE),
                          ('COMPUTE_SOURCE_LIN', CRA.COMPUTE_SOURCE_LIN)):
        check(f'C127: {src_name} carries hal_jitter / hal_jit_hash',
              'vec2 hal_jitter(vec2 pix)' in src
              and 'uint hal_jit_hash(uint x, uint y, int k)' in src)
    core = CRA.KERNEL_CORE
    check('C127: hal_jit_hash is a verbatim copy of the GRAIN stage\'s '
          'hal_grain_hash body (two copies of one hash cannot drift)',
          _hash_body(core, 'hal_jit_hash') == _hash_body(
              __import__('halcyon.gpu.stages', fromlist=['GRAIN']).GRAIN,
              'hal_grain_hash'))
    check('C127: the jitter functions sit before hal_rc_fetch (the compute '
          'build keeps them)',
          core.index('vec2 hal_jitter') < core.index('vec4 hal_rc_fetch'))
    # (f) the FM rows and the preset
    for row in (('jittered sample positions (REYES)',
                 {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
                  'aa_sample_pattern': 'JITTER'}, 'demo'),
                ('jittered sample position at 1 sample (a per-pixel offset)',
                 {'aa_mode': 'NONE', 'aa_sample_pattern': 'JITTER'}, 'demo')):
        check(f"the FM row '{row[0]}' exists", row in FM.ROWS)
    check("the RENDERMAN preset carries aa_sample_pattern 'JITTER' (PRMan's "
          'PixelSamples jitter was on by default)',
          PRESETS['RENDERMAN']['settings'].get('aa_sample_pattern') == 'JITTER')


def _hash_body(src, name):
    i = src.index('uint ' + name + '(')
    i = src.index('{', i)
    j = src.index('}', i)
    return src[i:j + 1]


def _backdrop(sc, w, h, img, offset=(0, 0), enc='LINEAR', depth_bits=24):
    """(vp, opts, gbuf) with a backdrop clear applied exactly as render()
    applies it (the derivation block's own lines)."""
    vp, o = _opts_for(sc, w, h, enc=enc)
    o.clear = raster.backdrop_clear(img, offset, w, h, o, depth_bits)
    g = raster.GBuffer(w, h)
    g.clear_depth(o.clear[0], o.clear[1], enc)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g, opts=o,
                     depth_bits=depth_bits)
    return vp, o, g


def test_w2_c038_backdrop_depth():
    """C038: the DS rear-plane depth bitmap -- the occlusion law (a near
    half hides everything, a far half changes nothing), the monotone law
    in the backdrop's distance, the named tie rule (the backdrop wins an
    equal key, both roads), the CLRIMAGE offset, the GPU twin (keys and
    zndc bitwise, the uncovered pixels carrying the clear on both roads),
    the encodings, the render road (the World field, the cache), the FM
    row and the fake-device knock."""
    w, h = 96, 72
    st = base_settings(w, h)
    sc = demo_scene(st, with_texture=False)
    check('C038: World.backdrop_depth defaults to None (every pixel bitwise; '
          'the module pin renders it)',
          getattr(sc.world, 'backdrop_depth', 'missing') is None
          and tuple(getattr(sc.world, 'backdrop_offset', ())) == (0, 0))
    vp, o_plain = _opts_for(sc, w, h)
    g_plain = _cpu_gbuf(sc, vp, w, h, o_plain)
    # (b) the occlusion law
    step = np.full((h, w), 1e6, np.float32)
    step[:, :w // 2] = 0.5
    _vp, o_step, g_step = _backdrop(sc, w, h, step)
    check('C038: a backdrop at 0.5 units over the left half leaves it '
          'uncovered', int((g_step.tri[:, :w // 2] >= 0).sum()) == 0)
    check('C038: the far half (1e6 units) equals the plain render bitwise '
          '(ids, bary, zndc)',
          bool(np.array_equal(g_step.tri[:, w // 2:], g_plain.tri[:, w // 2:]))
          and bool(np.array_equal(g_step.bary[:, w // 2:], g_plain.bary[:, w // 2:]))
          and bool(np.array_equal(g_step.zndc[:, w // 2:], g_plain.zndc[:, w // 2:])))
    check('C038: the uncovered pixels carry the backdrop\'s decoded depth '
          'in `depth` and `zndc`, and stay EMPTY in `tri` (the sky draws them)',
          bool(np.array_equal(g_step.depth[:, :w // 2], o_step.clear[1][:, :w // 2]))
          and bool((g_step.tri[:, :w // 2] == raster.EMPTY).all()))
    key_s, dec_s, sig_s = o_step.clear
    check('C038: under LINEAR the key IS the decoded (quantised) value and the '
          'signature is a content hash',
          bool(np.array_equal(key_s, dec_s)) and isinstance(sig_s, int)
          and sig_s != raster.backdrop_clear(step * 2.0, (0, 0), w, h, o_step, 24)[2])
    # invalid pixels carry no depth
    bad = step.copy()
    bad[:4, :4] = np.nan
    bad[4:8, :4] = -1.0
    k_bad = raster.backdrop_clear(bad, (0, 0), w, h, o_step, 24)[0]
    check('C038: non-finite or non-positive bitmap pixels carry no depth '
          '(+inf key)', bool(np.isinf(k_bad[:8, :4]).all())
          and bool(np.isfinite(k_bad[8:, :4]).all()))
    # (c) monotone in the distance
    counts = []
    for d in (2.0, 5.0, 12.0, 40.0):
        uni = np.full((h, w), d, np.float32)
        _vp, _o, g_d = _backdrop(sc, w, h, uni)
        counts.append(int((g_d.tri >= 0).sum()))
    check('C038: the covered count is non-decreasing in the backdrop\'s '
          'distance over 2, 5, 12, 40 units',
          all(a <= b for a, b in zip(counts, counts[1:]))
          and counts[0] < counts[-1] and counts[-1] == int((g_plain.tri >= 0).sum()),
          str(counts))
    # (d) the tie rule: the clear built DIRECTLY from the plain render's
    # own planes -- the floor stays UNCOVERED on both roads
    key_t = g_plain.depth.copy()
    dec_t = g_plain.depth.copy()
    o_tie = raster.RasterOpts(wparams=o_plain.wparams)
    o_tie.clear = (key_t, dec_t, hash(key_t.tobytes()))
    g_tie = raster.GBuffer(w, h)
    g_tie.clear_depth(key_t, dec_t, 'LINEAR')
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g_tie,
                     opts=o_tie)
    check('C038: THE NAMED TIE RULE -- a clear equal to every fragment\'s key '
          'leaves the frame uncovered (the backdrop wins an equal key)',
          int((g_tie.tri >= 0).sum()) == 0, f'{int((g_tie.tri >= 0).sum())} covered')
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc.mesh, vp, w, h,
                                                        opts=o_tie)
    sim_t = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w, h,
                                depth_bits=24, opts=o_tie)
    check('C038: the kernel road takes the same tie (nothing beats the clear)',
          int((sim_t[0] >= 0).sum()) == 0, f'{int((sim_t[0] >= 0).sum())} covered')
    # (e) the offset shifts the occlusion boundary
    _vp, o_k, g_k = _backdrop(sc, w, h, step, offset=(5, 0))
    col_a = int(np.argmax((g_step.tri >= 0).any(axis=0)))
    col_b = int(np.argmax((g_k.tri >= 0).any(axis=0)))
    check('C038: shifting backdrop_offset by (5, 0) shifts the occlusion '
          'boundary by 5 pixels (the DS\'s CLRIMAGE_OFFSET, wrap-around)',
          col_b == col_a - 5, f'{col_a} -> {col_b}')
    # (f) the GPU twin on the step backdrop: ids, keys and zndc bitwise, the
    # uncovered pixels carrying the clear on both roads
    for enc in ('LINEAR', 'N64_FLOAT18', 'GC_14E2'):
        _vp, o_e, g_e = _backdrop(sc, w, h, step, enc=enc)
        sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(
            sc.mesh, vp, w, h, opts=o_e)
        tri, bary, zndc, front, _b2, _lin, mark = CRA.simulate_raster(
            sx, sy, iw, z, bw, src, tmap, w, h, depth_bits=24, opts=o_e,
            refer=True)
        cov = g_e.tri >= 0
        keyplane = g_e.depth if enc == 'LINEAR' else g_e.zkey
        unc_exp = np.where(np.isfinite(o_e.clear[1]), o_e.clear[0], 1.0)
        check(f'C038 {enc}: the kernel picks the CPU\'s winner at every pixel, '
              'the keys are bitwise and the uncovered pixels report the clear',
              int((tri != g_e.tri).sum()) == 0
              and bool(np.array_equal(zndc[cov], keyplane[cov]))
              and bool(np.array_equal(zndc[~cov], unc_exp[~cov])),
              f'{int((tri != g_e.tri).sum())} ids differ')
        gi = raster.GBuffer(w, h)
        ids_img = np.stack([bary[..., 0], bary[..., 1], np.zeros_like(zndc),
                            tri.astype(np.float32)], 2)
        aux_img = np.stack([zndc, front.astype(np.float32), bary[..., 2],
                            mark.astype(np.float32)], 2)
        CRA.gbuffer_into(gi, ids_img, aux_img, None, opts=o_e, depth_bits=24)
        check(f'C038 {enc}: gbuffer_into writes depth / zndc / zkey bitwise '
              'the CPU G-buffer (the clear\'s planes on the uncovered pixels)',
              bool(np.array_equal(gi.depth, g_e.depth))
              and bool(np.array_equal(gi.zndc, g_e.zndc))
              and (g_e.zkey is None or bool(np.array_equal(gi.zkey, g_e.zkey))))
        if mark.any():
            pys, pxs = np.nonzero(mark)
            _c, _cs, bins, _bs, tiles, tw, _th = CRA.pack_raster_inputs(
                sx, sy, iw, z, bw, src, tmap, w, h)
            src_final = np.asarray(src, np.int64) if tmap is None \
                else np.asarray(tmap, np.int64)[src]
            r_tri, _rb, _rlb, r_z, _rf = CRA.replay_pixels(
                pxs, pys, sx, sy, iw, z, bw, src_final, tiles,
                bins.reshape(-1), tw, 'NONE', 24, opts=o_e)
            check(f'C038 {enc}: the replay at every marked pixel is the CPU\'s '
                  'winner and reports the clear where unbeaten',
                  bool(np.array_equal(r_tri, g_e.tri[pys, pxs]))
                  and bool(np.array_equal(r_z, zndc[pys, pxs])),
                  f'{int(mark.sum())} marked')
    check('C038: the compute source under the clear flag declares hal_rclear '
          'and reads it per pixel; the plain source does not',
          'uniform sampler2D hal_rclear;' in CRA.compute_source(False, True)
          and 'texelFetch(hal_rclear, xy, 0).x' in CRA.compute_source(False, True)
          and 'hal_rclear' not in CRA.COMPUTE_SOURCE
          and CRA.compute_variant_name(False, True) == 'HAL_RASTER_L0C1V0')
    # the render road: the World field, the G-buffer cache, the FM row
    st_r = base_settings(w, h)
    sc_r = FM.SCENES['backdrop_depth'](st_r)
    st_p = base_settings(w, h)
    sc_p = FM.SCENES['backdrop_depth'](st_p)
    sc_p.world.backdrop_depth = None
    R._GBUF_CACHE.clear()
    img_r = np.asarray(R.render(sc_r, st_r))
    img_p = np.asarray(R.render(sc_p, st_p))
    img_r2 = np.asarray(R.render(sc_r, st_r))
    check('C038: the FM scene renders differently with its backdrop than '
          'without, and the second render (a G-buffer cache HIT) is bitwise',
          not np.array_equal(img_r, img_p) and bool(np.array_equal(img_r, img_r2))
          and R._GBUF_STATS['last'] == 'HIT', R._GBUF_STATS['last'])
    check("the FM row 'rear-plane depth bitmap (DS)' exists",
          ('rear-plane depth bitmap (DS)', {}, 'backdrop_depth') in FM.ROWS)
    # the device-switch road (RAST-A1's shape): the GPU device knocks on
    # the raster door WITH the clear and, without a driver, renders
    # bitwise the CPU device (the fake device has no compute road -- its
    # shading runs the simulator, within the shading twin's rounding, so
    # the knock is measured without it)
    from ..gpu import shade as GSH
    seen = {}
    orig = CRA.raster_into_gbuffer

    def _rig(*a, **k):
        seen['clear'] = getattr(k.get('opts'), 'clear', None)
        seen['n'] = seen.get('n', 0) + 1
        return orig(*a, **k)
    CRA.raster_into_gbuffer = _rig
    try:
        st_g = base_settings(w, h)
        st_g.shadows = False
        st_g.render_device, st_g.gpu_raster = 'GPU', True
        sc_g = FM.SCENES['backdrop_depth'](st_g)
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        img_g = np.asarray(R.render(sc_g, st_g))
        st_c = base_settings(w, h)
        st_c.shadows = False
        sc_c = FM.SCENES['backdrop_depth'](st_c)
        R._GBUF_CACHE.clear()
        img_c = np.asarray(R.render(sc_c, st_c))
    finally:
        CRA.raster_into_gbuffer = orig
    check('C038: the GPU device knocks on the raster door with the clear '
          '(opts.clear set) and without a driver renders bitwise the CPU device',
          seen.get('n', 0) >= 1 and seen.get('clear') is not None
          and img_g.shape == img_c.shape and bool(np.array_equal(img_g, img_c)),
          f"knocked {seen.get('n', 0)}")


def _run_stage(name, h, w, uniforms, samplers):
    """The TR:37597 run_stage shape: one registered stage through the
    simulator over (H, W) with the given samplers (arrays)."""
    from ..core.texture import Texture
    from ..gpu import stages as ST
    from ..shaders.compiler import try_compile
    src = ST.STAGES[name].replace('in vec2 vUV;', 'uniform vec2 vUV;')
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, err
    n = h * w
    yy, xx = np.mgrid[0:h, 0:w]
    u = {'vUV': np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h],
                         1).astype(np.float32)}
    for k, v in samplers.items():
        u[k] = Texture(v, colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')
    for k, v in uniforms.items():
        if isinstance(v, (tuple, list)):
            u[k] = np.tile(np.asarray(v, np.float32)[None, :], (n, 1))
        else:
            u[k] = np.full(n, float(v), np.float32)
    outs, _d = prog.run(u, {}, n)
    return np.asarray(outs['Color'], np.float32).reshape(h, w, 4), None


def _rgba(rgb):
    return np.concatenate([np.asarray(rgb, np.float32),
                           np.ones(rgb.shape[:2] + (1,), np.float32)], 2)


def test_w2_c001_n64_coverage_vi():
    """C001: the RDP's 3-bit coverage and the VI's scan-out filter -- the
    coverage laws on the demo G-buffer, the VI laws on a synthetic frame
    (identity on full coverage, a pull never past the neighbours, the
    divot median only on a partial triple, the frame-edge rule,
    determinism), the named skip at ss > 1, the G-buffer cache carrying
    the plane, the GPU twins (raster: ids and the cvg plane equal, the
    extended sliver mark's replay; post: both stages d == 0.0 in the
    simulator and the whole chain function), the refusals by name, the
    FM rows and the N64 preset."""
    from ..core import n64vi as N64VI
    from ..gpu import chain as CH
    from ..gpu import stages as ST
    st = base_settings(W, H)
    check('C001: n64_coverage_aa defaults off and n64_divot on (inert)',
          st.n64_coverage_aa is False and st.n64_divot is True)
    check('C001: RasterOpts() asks for no coverage plane and GBuffer() has '
          'none', raster.RasterOpts().cvg is False
          and raster.GBuffer(4, 4).cvg is None)
    # (b) coverage laws on the demo scene at ss = 1
    w2, h2 = 160, 120
    st2 = base_settings(w2, h2)
    sc2 = demo_scene(st2, with_texture=False)
    vp2, o_c = _opts_for(sc2, w2, h2, cvg=True)
    g = _cpu_gbuf(sc2, vp2, w2, h2, o_c)
    g_plain = _cpu_gbuf(sc2, vp2, w2, h2, raster.RasterOpts(wparams=o_c.wparams))
    check('C001: asking for the coverage plane changes no id, no bary, no '
          'depth (the plane is a side output)',
          bool(np.array_equal(g.tri, g_plain.tri))
          and bool(np.array_equal(g.bary, g_plain.bary))
          and bool(np.array_equal(g.depth, g_plain.depth))
          and g.cvg is not None and g.cvg.dtype == np.uint8)
    ids = g.tri
    same_all = np.ones(ids.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            same_all &= np.roll(np.roll(ids, dy, 0), dx, 1) == ids
    same_all[0, :] = same_all[-1, :] = False
    same_all[:, 0] = same_all[:, -1] = False
    cov = ids >= 0
    check('C001: every covered pixel whose eight neighbours share its id has '
          'cvg 7 (full)', bool((g.cvg[same_all & cov] == 7).all()),
          f'{int((g.cvg[same_all & cov] != 7).sum())} not full')
    partial = g.cvg < 7
    check('C001: partial pixels exist (> 0) and every one lies within one '
          'pixel of an id boundary',
          int(partial.sum()) > 0 and not bool((partial & same_all).any()),
          f'{int(partial.sum())} partial')
    check('C001: uncovered pixels are 7 (the VI blends silhouettes toward the '
          'sky, never the sky)', bool((g.cvg[~cov] == 7).all()))
    # the loop fill and the batched fill agree on the plane
    g_loop = raster.GBuffer(w2, h2)
    raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w2, h2, gbuf=g_loop,
                     opts=o_c, batched=False)
    check('C001: the loop fill and the batched fill write the same coverage '
          'plane', bool(np.array_equal(g_loop.cvg, g.cvg)))
    # coverage_count is the generic grid: n_sub 4 at a fully inside pixel
    # counts 16, and the 3-bit reduction of 16 is 7
    cc = raster.coverage_count(0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 10000.0,
                               np.float32(10.5), np.float32(10.5), 4)
    check('C001: coverage_count is 16 inside a big triangle and coverage16 '
          'reduces it to 7', int(cc) == 16
          and int(raster.coverage16(0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 10000.0,
                                    np.float32(10.5), np.float32(10.5))) == 7)
    check('C001: a pixel centred on an edge counts half its subsamples (8 of '
          '16; the DS-style 8x8 grid is one argument away: 32 of 64)',
          int(raster.coverage_count(0.0, 0.0, 100.0, 0.0, 0.0, 100.0,
                                    10000.0, np.float32(0.0),
                                    np.float32(50.5), 4)) == 8
          and int(raster.coverage_count(0.0, 0.0, 100.0, 0.0, 0.0, 100.0,
                                        10000.0, np.float32(0.0),
                                        np.float32(50.5), 8)) == 32
          and int(raster.coverage16(0.0, 0.0, 100.0, 0.0, 0.0, 100.0,
                                    10000.0, np.float32(0.0),
                                    np.float32(50.5))) == 3)
    # (c) the VI laws on a synthetic frame
    rng = np.random.default_rng(101)
    hs, ws = 24, 40
    rgb = rng.random((hs, ws, 3)).astype(np.float32)
    cvg = rng.integers(0, 8, (hs, ws)).astype(np.int32)
    cvg[::4, ::3] = 7
    c8 = N64VI.quant5(rgb)
    aa = N64VI.vi_aa(c8, cvg)
    full = cvg == 7
    check('C001 VI: pixels with cvg 7 are untouched by the AA pass',
          bool((aa[full] == c8[full]).all()))
    check('C001 VI: quant5 is the RGBA5551 expansion (5 bits replicated, '
          '0..255)', bool(((c8 & 7) == (c8 >> 5)).all()) and c8.max() <= 255)
    # a pull toward the neighbours, never past them: angrylion's list
    # holds the centre first, so penmin <= c <= penmax at EVERY coverage
    lo = np.full(c8.shape, 1000, np.int32)
    hi = np.full(c8.shape, -1, np.int32)
    for dx, dy in N64VI.NEIGHBOURS:
        v = N64VI._nb(c8, dx, dy, 0)
        ok = N64VI._nb(cvg, dx, dy, -1) == 7
        lo = np.where(ok[..., None], np.minimum(lo, v), lo)
        hi = np.where(ok[..., None], np.maximum(hi, v), hi)
    has = hi >= 0
    lo = np.where(has, lo, c8)
    hi = np.where(has, hi, c8)
    lower = np.minimum(c8, lo)
    upper = np.maximum(c8, hi)
    inb = (aa >= lower) & (aa <= upper)
    check('C001 VI: at every coverage the AA output lies within '
          '[min(c, neighbours), max(c, neighbours)] -- a pull toward, never '
          'past (the centre is first in angrylion\'s list, so the 8-bit mask '
          'never wraps)',
          bool(inb.all()), f'{int((~inb).sum())} outside')
    # the formula itself, re-derived per pixel with a scalar loop: a
    # literal transcription of angrylion's video_filter16 (backr[0] = the
    # centre, numoffull = 1, then the full neighbours) and
    # video_max_optimized (the running champions, the previous champion
    # as the penultimate, the scan after the champion's position)

    def _vmo(px_):
        posmax = posmin = 0
        curpenmax = curpenmin = px_[0]
        for i_ in range(1, len(px_)):
            if px_[i_] > px_[posmax]:
                curpenmax = px_[posmax]
                posmax = i_
            elif px_[i_] < px_[posmin]:
                curpenmin = px_[posmin]
                posmin = i_
        mx, mn = px_[posmax], px_[posmin]
        if curpenmax != mx:
            for i_ in range(posmax + 1, len(px_)):
                if px_[i_] > curpenmax:
                    curpenmax = px_[i_]
        if curpenmin != mn:
            for i_ in range(posmin + 1, len(px_)):
                if px_[i_] < curpenmin:
                    curpenmin = px_[i_]
        return curpenmin, curpenmax

    ok_f = True
    for y_ in range(0, hs):
        for x_ in range(0, ws):
            if cvg[y_, x_] >= 7:
                ok_f &= bool((aa[y_, x_] == c8[y_, x_]).all())
                continue
            for ch in range(3):
                c_ = int(c8[y_, x_, ch])
                cand = [c_]
                for dx, dy in N64VI.NEIGHBOURS:
                    qy, qx = y_ + dy, x_ + dx
                    if 0 <= qy < hs and 0 <= qx < ws and cvg[qy, qx] == 7:
                        cand.append(int(c8[qy, qx, ch]))
                pmin, pmax = _vmo(cand)
                d_ = (((pmin + pmax) - 2 * c_) * (7 - int(cvg[y_, x_])) + 4) >> 3
                ok_f &= int(aa[y_, x_, ch]) == ((c_ + d_) & 0xff)
    check('C001 VI: the vectorised AA pass equals a scalar transcription of '
          'angrylion\'s video_filter16 / video_max_optimized (the centre '
          'first in the list) at every pixel of the frame', ok_f)
    flat = np.full((hs, ws, 3), 77, np.int32)
    check('C001 VI: neighbours equal to the centre leave it in place '
          '(identity on a flat field at any coverage)',
          bool(np.array_equal(N64VI.vi_aa(flat, cvg), flat)))
    check('C001 VI: the AA pass moves some partial pixels (non-vacuous)',
          int((aa != c8).sum()) > 0)
    dv = N64VI.vi_divot(aa, cvg)
    la = np.full_like(cvg, 7)
    la[:, 1:] = cvg[:, :-1]
    ra = np.full_like(cvg, 7)
    ra[:, :-1] = cvg[:, 1:]
    trip = (la & cvg & ra) != 7
    lft = aa.copy()
    lft[:, 1:] = aa[:, :-1]
    rgt = aa.copy()
    rgt[:, :-1] = aa[:, 1:]
    med = np.median(np.stack([lft, aa, rgt]), axis=0).astype(np.int32)
    check('C001 VI: the divot changes only pixels whose horizontal triple '
          'holds a partial pixel, and its output is the median of the three',
          bool((dv[~trip] == aa[~trip]).all()) and bool((dv[trip] == med[trip]).all()))
    check('C001 VI: determinism (same inputs, same bits)',
          bool(np.array_equal(N64VI.vi_aa(c8, cvg), aa))
          and bool(np.array_equal(N64VI.vi_divot(aa, cvg), dv)))
    # (d'') the frame's x-edges: partial pixels in column 0 and the last
    # column keep the AA output through the divot (the missing neighbour
    # is the centre) on the CPU and in the simulator alike
    cvg_e = np.full((hs, ws), 7, np.int32)
    cvg_e[:, 0] = 3
    cvg_e[:, -1] = 2
    aa_e = N64VI.vi_aa(c8, cvg_e)
    dv_e = N64VI.vi_divot(aa_e, cvg_e)
    check('C001 VI: at the frame\'s x-edges the divot is the identity (the '
          'missing neighbour IS the centre)',
          bool((dv_e[:, 0] == aa_e[:, 0]).all())
          and bool((dv_e[:, -1] == aa_e[:, -1]).all()))
    # (e) the post twins through the simulator
    uni = {'resolution': (float(ws), float(hs))}
    got_aa, err = _run_stage('N64VI_AA', hs, ws, uni,
                             {'source': _rgba(rgb), 'cvg': N64VI.cvg_image(cvg)})
    ref_aa = aa.astype(np.float32) / np.float32(255.0)
    check('C001 GPU: N64VI_AA bitwise vi_aa(quant5) in the simulator '
          '(d == 0.0), alpha passed through',
          got_aa is not None and float(np.abs(got_aa[..., :3] - ref_aa).max()) == 0.0
          and bool((got_aa[..., 3] == 1.0).all()), str(err) if got_aa is None
          else f'd {float(np.abs(got_aa[..., :3] - ref_aa).max())}')
    got_dv, err = _run_stage('N64VI_DIVOT', hs, ws, uni,
                             {'source': _rgba(ref_aa), 'cvg': N64VI.cvg_image(cvg)})
    ref_dv = dv.astype(np.float32) / np.float32(255.0)
    check('C001 GPU: N64VI_DIVOT bitwise vi_divot over the AA output '
          '(d == 0.0)', got_dv is not None
          and float(np.abs(got_dv[..., :3] - ref_dv).max()) == 0.0,
          str(err) if got_dv is None else f'd {float(np.abs(got_dv[..., :3] - ref_dv).max())}')
    got_e, _err = _run_stage('N64VI_DIVOT', hs, ws, uni,
                             {'source': _rgba(aa_e.astype(np.float32) / 255.0),
                              'cvg': N64VI.cvg_image(cvg_e)})
    check('C001 GPU: the divot\'s frame-edge rule holds in the simulator too '
          '(columns 0 and last untouched)', got_e is not None
          and float(np.abs(got_e[..., :3] - dv_e.astype(np.float32) / 255.0).max()) == 0.0)
    st_v = base_settings(ws, hs)
    st_v.n64_coverage_aa = True
    st_v._n64_cvg = cvg
    whole = N64VI.n64_vi(rgb, st_v)
    check('C001: the chain function n64_vi runs both passes (bitwise the two '
          'stages chained) and returns float32 k / 255',
          bool(np.array_equal(whole, got_dv[..., :3])) and whole.dtype == np.float32)
    st_v.n64_divot = False
    check('C001: n64_divot off leaves the AA pass alone',
          bool(np.array_equal(N64VI.n64_vi(rgb, st_v), ref_aa)))
    # the refusals by name
    st_off = base_settings(W, H)
    check("C001: n64vi_refusal names 'off' when the flag is off",
          'off' in str(CH.n64vi_refusal(st_off)))
    st_on = base_settings(W, H)
    st_on.n64_coverage_aa = True
    st_on._n64_cvg = None
    check("C001: n64vi_refusal names the 'absent' plane (1 sample per pixel)",
          'absent' in str(CH.n64vi_refusal(st_on)))
    check('C001: the two stages are registered EXACT and ENABLED with '
          'sampler + vec2 interfaces',
          ST.VALIDATION['N64VI_AA'][0] == 'EXACT' and 'N64VI_AA' in ST.ENABLED
          and 'N64VI_DIVOT' in ST.ENABLED
          and ST.INTERFACE['N64VI_AA'] == {'samplers': ['source', 'cvg'],
                                            'vec2': ['resolution']})
    # plane_for_frame: the shape rule, printed by name
    said = []
    check('C001: a coverage plane of another size is skipped by name',
          N64VI.plane_for_frame(cvg, (hs + 1, ws), st_on, said.append) is None
          and said and "not the frame's size" in said[0])
    check('C001: an absent plane is skipped by name; the right plane passes',
          N64VI.plane_for_frame(None, (hs, ws), st_on, said.append) is None
          and 'no coverage plane' in said[-1]
          and N64VI.plane_for_frame(cvg, (hs, ws), st_on, said.append) is not None)
    # (d) the named skip at ss > 1: bitwise the plain supersampled frame
    R._N64_SKIP_SAID.clear()
    st_s = base_settings(W, H)
    st_s.aa_mode, st_s.aa_samples, st_s.n64_coverage_aa = 'SUPERSAMPLE', 4, True
    sc_s = demo_scene(st_s, with_texture=False)
    (img_s, out_s) = _capture(lambda: np.asarray(R.render(sc_s, st_s)))
    st_p = base_settings(W, H)
    st_p.aa_mode, st_p.aa_samples = 'SUPERSAMPLE', 4
    img_p = np.asarray(R.render(demo_scene(st_p, with_texture=False), st_p))
    check('C001: with aa_samples 4 the flag renders bitwise the plain '
          'supersampled frame, prints the named skip and leaves no plane',
          bool(np.array_equal(img_s, img_p))
          and 'N64 coverage AA needs 1 sample per pixel' in out_s
          and getattr(sc_s, 'last_cvg', 'missing') is None)
    pst = _post(img_s, st_s)
    check('C001: the post chain over a frame without a plane is the plain '
          'post (the stage never runs)',
          bool(np.array_equal(pst, _post(img_p, st_p))))
    # (d') the G-buffer cache carries the plane: two renders, the second a HIT
    R._GBUF_CACHE.clear()
    st_c = base_settings(W, H)
    st_c.n64_coverage_aa = True
    sc_c = demo_scene(st_c, with_texture=False)
    img1 = np.asarray(R.render(sc_c, st_c))
    cvg1 = None if sc_c.last_cvg is None else sc_c.last_cvg.copy()
    p1 = post.process(img1, st_c, frame=1, seed=0, target_size=(W, H),
                      allow_resize=False, cvg=sc_c.last_cvg)
    img2 = np.asarray(R.render(sc_c, st_c))
    hit = R._GBUF_STATS['last']
    p2 = post.process(img2, st_c, frame=1, seed=0, target_size=(W, H),
                      allow_resize=False, cvg=sc_c.last_cvg)
    check('C001: the second render is a G-buffer cache HIT that restores the '
          'coverage plane (scene.last_cvg present, both post frames bitwise)',
          hit == 'HIT' and cvg1 is not None and sc_c.last_cvg is not None
          and bool(np.array_equal(cvg1, sc_c.last_cvg))
          and bool(np.array_equal(p1, p2)), hit)
    check('C001: the VI stage moves the picture (the post frame differs from '
          'the plain post)',
          not np.array_equal(p1, _post(img1, base_settings(W, H))))
    # (e) the raster twin: ids and the cvg plane equal; the extended
    # sliver mark's replay recomputes coverage16
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc2.mesh, vp2, w2, h2,
                                                        opts=o_c)
    sim = CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w2, h2,
                              depth_bits=24, opts=o_c, refer=True, want_cvg=True)
    check('C001 GPU raster: simulate_raster returns the cvg plane as an '
          'EIGHTH value only when asked', len(sim) == 8
          and len(CRA.simulate_raster(sx, sy, iw, z, bw, src, tmap, w2, h2,
                                      depth_bits=24, opts=o_c)) == 7)
    tri, _bary, _zndc, _front, _b2, _lin, mark, cvg_sim = sim
    check('C001 GPU raster: zero differing ids and the coverage planes equal '
          '(np.array_equal)', int((tri != g.tri).sum()) == 0
          and bool(np.array_equal(cvg_sim, g.cvg)),
          f'{int((cvg_sim != g.cvg).sum())} cvg differ')
    # every pixel one of whose 16 subsamples sits inside the wobble window
    # is marked (the extended sliver mark; its count recorded)
    pys, pxs = np.nonzero(g.tri >= 0)
    _c, _cs, bins, _bs, tiles, tw, _th = CRA.pack_raster_inputs(
        sx, sy, iw, z, bw, src, tmap, w2, h2)
    src_final = np.asarray(src, np.int64) if tmap is None \
        else np.asarray(tmap, np.int64)[src]
    mpy, mpx = np.nonzero(mark)
    rep = CRA.replay_pixels(mpx, mpy, sx, sy, iw, z, bw, src_final, tiles,
                            bins.reshape(-1), tw, 'NONE', 24, opts=o_c,
                            want_cvg=True)
    check('C001 GPU raster: the replay at every marked pixel returns the '
          "CPU's winner and the CPU's coverage16",
          bool(np.array_equal(rep[0], g.tri[mpy, mpx]))
          and bool(np.array_equal(rep[5], g.cvg[mpy, mpx].astype(np.int32))),
          f'{int(mark.sum())} marked (the extended sliver mark)')
    # the sliver law on the CPU: sliver16 at the winners' corners marks a
    # SUBSET of the kernel's marks (the kernel adds its own z / edge windows)
    win_e = np.full(len(pys), -1, np.int64)
    # map each covered pixel's winner to an emitted triangle via src_final
    by_src = {}
    for e_i, s_i in enumerate(src_final):
        by_src.setdefault(int(s_i), e_i)
    for i, (py_, px_) in enumerate(zip(pys, pxs)):
        win_e[i] = by_src.get(int(g.tri[py_, px_]), -1)
    okw = win_e >= 0
    e_ = win_e[okw]
    sl = raster.sliver16(sx[e_, 0], sy[e_, 0], sx[e_, 1], sy[e_, 1],
                         sx[e_, 2], sy[e_, 2],
                         pxs[okw].astype(np.float32) + np.float32(0.5),
                         pys[okw].astype(np.float32) + np.float32(0.5))
    slm = np.zeros(mark.shape, bool)
    slm[pys[okw][sl], pxs[okw][sl]] = True
    check('C001 GPU raster: every pixel whose 16 subsamples touch the wobble '
          'window (CPU sliver16) is marked by the kernel',
          not bool((slm & ~mark).any()), f'{int(slm.sum())} slivers, {int(mark.sum())} marks')
    check('C001 GPU raster: the marks stay under the 10 % referral bail on '
          'the demo scene', int(mark.sum()) <= 0.10 * int((g.tri >= 0).sum()))
    check('C001: the compute source carries the coverage block ONLY under '
          'the cvg flag (V1: imageStore hal_out_cvg; V0: no n16)',
          'imageStore(hal_out_cvg, xy, cvgv);' in CRA.compute_source(False, False, True)
          and 'n16' in CRA.compute_source(False, False, True)
          and 'n16' not in CRA.COMPUTE_SOURCE
          and CRA.compute_variant_name(True, False, True) == 'HAL_RASTER_L1C0V1')
    gi = raster.GBuffer(w2, h2)
    gi.alloc_cvg()
    ids_img = np.stack([_bary[..., 0], _bary[..., 1], np.zeros_like(_zndc),
                        tri.astype(np.float32)], 2)
    aux_img = np.stack([_zndc, _front.astype(np.float32), _bary[..., 2],
                        mark.astype(np.float32)], 2)
    cvg_img = np.zeros((h2, w2, 4), np.float32)
    cvg_img[..., 0] = cvg_sim
    CRA.gbuffer_into(gi, ids_img, aux_img, None, opts=o_c, depth_bits=24,
                     cvg=cvg_img)
    check('C001: gbuffer_into writes the coverage plane (uint8, uncovered 7) '
          'bitwise the CPU G-buffer', bool(np.array_equal(gi.cvg, g.cvg)))
    # (h) the FM rows and the N64 preset
    for row in (('N64 coverage AA + VI filter', {'n64_coverage_aa': True}, 'demo'),
                ('N64 coverage AA, no divot',
                 {'n64_coverage_aa': True, 'n64_divot': False}, 'demo'),
                ('N64 coverage AA on the coverage-edge rig',
                 {'n64_coverage_aa': True}, 'coverage_edge')):
        check(f"the FM row '{row[0]}' exists", row in FM.ROWS)
    n64 = PRESETS['N64']['settings']
    check("the N64 preset drops its supersampler for the coverage blend "
          "(aa_mode NONE, n64_coverage_aa, n64_divot; no aa_filter key)",
          n64.get('aa_mode') == 'NONE' and n64.get('n64_coverage_aa') is True
          and n64.get('n64_divot') is True and 'aa_filter' not in n64)


def test_zz_fix4_gpu_raster_gets_a_coverage_plane():
    """1.90.0 fix pass 4 (found on the driver, 2026-10-05). The kernel
    writes the coverage plane only into a plane that exists, and render()
    never allocated one before the GPU raster call: on the driver a
    GPU-rastered N64 frame had no coverage, the VI filter was skipped on
    the GPU device (5512 of 6912 px wrong) and the plane-less G-buffer was
    cached for the next CPU frame. The fake device cannot build the compute
    raster, so this spy stands at the driver's door."""
    seen = {}
    real = CRA.raster_into_gbuffer

    def spy(mesh, vp, w, h, gbuf, **k):
        seen['plane'] = gbuf.cvg is not None
        seen['asked'] = bool(getattr(k.get('opts'), 'cvg', False))
        return False, 'the test stands in for the driver'
    CRA.raster_into_gbuffer = spy
    try:
        for on in (True, False):
            R._GBUF_CACHE.clear()
            seen.clear()
            st = base_settings(W, H, n64_coverage_aa=on, render_device='GPU',
                               gpu_raster=True)
            st.use_processes = False
            sc = demo_scene(st)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                img = np.asarray(R.render(sc, st))
            check(f'n64_coverage_aa {on}: the GPU raster door is knocked on '
                  '(the gate lets the frame through)', 'plane' in seen,
                  buf.getvalue()[-200:])
            if on:
                check('...with the coverage plane ALLOCATED and asked for, so '
                      'the kernel writes it (want_cvg)',
                      seen.get('plane') is True and seen.get('asked') is True,
                      str(seen))
                check('...and after the stand-in refuses, the CPU fill leaves '
                      'the frame its coverage plane',
                      getattr(sc, 'last_cvg', None) is not None
                      and bool(np.isfinite(img).all()))
            else:
                check('...with NO plane when the frame did not ask for one '
                      '(a default frame pays nothing)',
                      seen.get('plane') is False and seen.get('asked') is False,
                      str(seen))
    finally:
        CRA.raster_into_gbuffer = real
        R._GBUF_CACHE.clear()


def test_zz_fix4_ntsc_dot_crawl_says_why():
    """1.90.0 fix pass 4: the chain records a refused stage as 'refused by
    name (the console says why)'; under dot crawl gpu/chain.ntsc returned
    None and printed nothing (seen on the driver, the VHS preset row)."""
    from ..gpu import chain as CH
    st = base_settings(W, H, composite=True, composite_dot_crawl=0.5)
    CH._WARNED.discard('NTSC on the CPU: dot crawl is frame-dependent and stays on '
                       'the CPU, and a frame using it falls back whole')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got = CH.ntsc(None, st)
    check('ntsc under dot crawl refuses and NAMES dot crawl on the console',
          got is None and 'dot crawl' in buf.getvalue(), buf.getvalue()[:120])
    st0 = base_settings(W, H, composite=False, composite_dot_crawl=0.5)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        got0 = CH.ntsc(None, st0)
    check('with the composite stage off nothing is printed (nothing refused)',
          got0 is None and buf.getvalue() == '')


def main():
    from . import utf8_console
    utf8_console()
    order = [v for k, v in sorted(globals().items())
             if k.startswith('test_')]
    for fn in order:
        print(fn.__name__)
        try:
            fn()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(fn.__name__)
    print()
    print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS) if FAILS
          else 'all R251 raster tests passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
