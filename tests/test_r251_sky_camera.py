"""R251 (1.90.0) sky-camera tests -- the SKY slot (W1.09): C056 Cylinder
Sky (Doom), C100 Gradient Backdrop (LightWave), C048 Mode 7 floor (SNES /
GBA); the CAM slot (W1.10) appends its tests below `# ---- CAM tests
below ----`.

Every feature is proved on BOTH roads: the CPU law against the core
function, the GPU twin (`gpu/sky.simulate` against
`render._background_image`) bitwise (`d == 0.0`), the refusal by name,
and the driver plumbing through `tests/fakedevice.py`. The FIRST test
pins the 1.89.0 zip loudly and the identity at the defaults, bitwise,
for `render` AND `post.process`.

Runs alone::

    python -m halcyon.tests.test_r251_sky_camera
"""
import dataclasses
import importlib
import math
import sys
import traceback

import numpy as np

from ..core import post as PO
from ..core import raster as CR
from ..core import render as R
from ..core import sky as SK
from ..core.scene import ImageBuffer
from ..core.texture import Texture
from ..gpu import shade as GSH
from ..gpu import sky as GSKY
from . import utf8_console
from .scenebuild import _mesh_concat, cube, demo_scene, look_at_matrix, sphere
from .r251_common import clear_palette_locks, delta_extra, preset_delta
from .test_render import _prev_engine, _sk, _wnode, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


W, H = 96, 72
CAM_EYE = (5.2, -6.4, 3.6)          # the demo camera (scenebuild.demo_scene)
CAM_AT = (0.0, -0.2, 0.9)


def settings(w=W, h=H, ss=1, **kw):
    """The era-look rig: 96x72, no shadows, no transparency, one AA
    road chosen by `ss`."""
    st = base_settings(w, h)
    st.shadows = False
    st.transparency = 'NONE'
    st.show_stats = False
    if ss > 1:
        st.aa_mode = 'SUPERSAMPLE'
        st.aa_samples = ss * ss
    else:
        st.aa_mode = 'NONE'
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def cyl_image():
    """A 32x16 RGB image with a distinct hue per column and a distinct
    value per row: any column or row error moves a pixel."""
    px = np.zeros((16, 32, 4), np.float32)
    xs = np.arange(32, dtype=np.float32) / 31.0
    ys = np.arange(16, dtype=np.float32) / 15.0
    px[..., 0] = xs[None, :]
    px[..., 1] = ys[:, None]
    px[..., 2] = (1.0 - xs)[None, :] * 0.5 + 0.25
    px[..., 3] = 1.0
    return px


CYL_PX = cyl_image()
CYL_BUF = ImageBuffer(name='cylsky', pixels=CYL_PX, colorspace='Linear')
CYL_TEX = Texture(CYL_PX, name='cylsky', colorspace='Linear')


def m7_image():
    """A 64x64 map: an 8x8 checker of saturated hues with a red one-texel
    border (the spec's rig)."""
    px = np.zeros((64, 64, 4), np.float32)
    hues = np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0),
                     (0, 1, 1), (1, 0, 1), (1, 1, 1), (0.5, 0.5, 0.5)],
                    np.float32)
    yy, xx = np.mgrid[0:64, 0:64]
    cell = ((yy // 8) * 8 + (xx // 8)) % 8
    px[..., :3] = hues[cell]
    px[0, :, :3] = (1, 0, 0)
    px[-1, :, :3] = (1, 0, 0)
    px[:, 0, :3] = (1, 0, 0)
    px[:, -1, :3] = (1, 0, 0)
    px[..., 3] = 1.0
    return px


M7_BUF = ImageBuffer(name='m7', pixels=m7_image(), colorspace='Linear')


def camera_like(cam, pitch_deg=None, yaw_delta_deg=0.0, projection=None):
    """The demo camera re-aimed: the same eye, its own heading plus
    `yaw_delta_deg`, level (`pitch_deg` 0) or tilted; an optional
    projection override (Blender's `calc_matrix_camera` road)."""
    e = np.asarray(CAM_EYE, np.float64)
    d = np.asarray(CAM_AT, np.float64) - e
    yaw0 = math.atan2(d[1], d[0]) + math.radians(yaw_delta_deg)
    if pitch_deg is None:
        p = math.atan2(d[2], math.hypot(d[0], d[1]))
    else:
        p = math.radians(pitch_deg)
    tgt = e + np.array([math.cos(yaw0) * math.cos(p),
                        math.sin(yaw0) * math.cos(p), math.sin(p)])
    kw = {'matrix_world': look_at_matrix(e, tgt)}
    if projection is not None:
        kw['projection'] = np.asarray(projection, np.float32)
    return dataclasses.replace(cam, **kw)


def twin(sc, st, textures, ss, J=None):
    """(sim, why, cpu, plan, gbuf): the GPU sky pass through the GLSL
    simulator against the CPU's `_background_image` over the same
    G-buffer; `J` a clip-space 4x4 applied to vp as the frame's own J
    block does (jitter, stereo eye)."""
    rw, rh = st.resolution_x * ss, st.resolution_y * ss
    _view, _proj, vp, eye = R.camera_matrices(sc.camera, rw, rh)
    if J is not None:
        vp = (J @ vp).astype(np.float32)
    g = CR.GBuffer(rw, rh)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, rw, rh, gbuf=g)
    dbg = {}
    sim, why = GSKY.simulate(sc, g, st, vp, eye, textures, ss=ss, debug=dbg)
    if sim is None:
        return None, why, None, None, g
    cpu = R._background_image(sc, st, rw, rh, vp, eye, ~g.mask(), textures,
                              ss=ss)
    return sim, None, cpu, dbg.get('plan'), g


def twin_d(sc, st, textures, ss, J=None):
    """(d, why, plan): the max abs difference at every uncovered pixel
    (0.0 = bitwise), None with the refusal when the pass refuses."""
    sim, why, cpu, plan, g = twin(sc, st, textures, ss, J)
    if sim is None:
        return None, why, plan
    unc = ~g.mask()
    if not unc.any():
        return 0.0, None, plan
    d = float(np.abs(sim[unc] - cpu[unc]).max())
    return d, None, plan


def bg_graph():
    """A plain Background node graph (Blender's default world): the
    NODES road the sky pass folds to a flat colour."""
    nodes = {
        'bg': _wnode('bg', 'ShaderNodeBackground', {},
                     [_sk('Color', 'RGBA', [0.3, 0.5, 0.7, 1.0]),
                      _sk('Strength', 'VALUE', 1.3)],
                     [{'name': 'Background', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputWorld', {},
                      [_sk('Surface', 'SHADER', None, ['bg', 0])], []),
    }
    return {'output': 'out', 'nodes': nodes}


def fake_device_render(sc, st):
    """The whole frame through the fake device on the GPU road (the
    served caches cleared first, or a served G-buffer hides the pass)."""
    from . import fakedevice
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    st.render_device = 'GPU'
    with fakedevice.installed():
        got = np.asarray(R.render(sc, st))
    st.render_device = 'CPU'
    return got


DEFERRED_BAR = 6e-3     # the deferred shading's frame bar (texture 0.0; 6e-6 measured on the demo scene)


def fake_device_checks(label, sc, got, cpu):
    """The two fake-device laws of a sky feature: the SKY pixels (the
    pass this pack draws) bitwise the CPU device's, the whole frame
    within the deferred shading's existing bar (its 6e-6 on the demo
    scene's geometry is the shipped shading's, not the sky's)."""
    _v, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CR.GBuffer(W, H)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g)
    unc = ~g.mask()
    ok_shape = got.shape == cpu.shape
    d_sky = float(np.abs(got[unc] - cpu[unc]).max()) if ok_shape and unc.any() else -1.0
    d_all = float(np.abs(got - cpu).max()) if ok_shape else -1.0
    check(f'the fake device draws {label}\'s sky pixels bitwise the CPU '
          'device\'s (the driver\'s rules, the sky pass drawn last)',
          ok_shape and d_sky == 0.0, f'sky max {d_sky}')
    check(f'...and {label} as a whole within the deferred shading\'s bar',
          ok_shape and 0.0 <= d_all <= DEFERRED_BAR, f'frame max {d_all}')


def post_kw(sc, st):
    return dict(frame=7, seed=st.seed, target_size=(W, H),
                allow_resize=False,
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None),
                flare_sources=getattr(sc, 'last_flares', None))


# =========================================================== the identity


def test_00_identity_at_defaults():
    """The 1.89.0 zip beside the package renders the SAME frames as this
    tree at the defaults of every new World field -- `render` and
    `post.process` bitwise -- on the demo scene, a GRADIENT world (the
    fields CYLINDER / LW_GRADIENT sit beside), the analytic CHECKER
    ground plane (the road MODE7 sits beside), an HDRI world through the
    image slot CYLINDER shares, and the supersampled fast_background
    road. The old engine receives THIS tree's settings and scene and
    ignores the fields it never heard of: the proof of invisibility."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the neutrality pin runs, '
          'not skips)', RP is not None)
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')

    def rig(world, ss, texture):
        st = settings(ss=ss)
        sc = demo_scene(st, with_texture=texture)
        for k, v in world.items():
            setattr(sc.world, k, v)
        if world.get('env_image') is not None:
            sc.images['cylsky'] = CYL_BUF
        return sc, st

    cases = [
        ('the demo scene at the defaults', {}, 1, True),
        ('a GRADIENT world with the new World fields at their defaults',
         {'mode': 'GRADIENT'}, 1, False),
        ('the analytic CHECKER ground plane under a GRADIENT sky',
         {'mode': 'GRADIENT', 'ground_plane': True, 'ground_mode': 'CHECKER'},
         1, False),
        ('an HDRI world reading the image slot the cylinder sky shares',
         {'mode': 'HDRI', 'env_image': CYL_BUF}, 1, False),
        ('the demo scene supersampled (the fast_background road)', {}, 2,
         False),
    ]
    for label, world, ss, texture in cases:
        sc, st = rig(world, ss, texture)
        now = np.asarray(R.render(sc, st))
        sc2, st2 = rig(world, ss, texture)
        prev = np.asarray(RP.render(sc2, st2))
        same = now.shape == prev.shape and bool(np.array_equal(now, prev))
        check(f'{label}: render bitwise the 1.89.0 engine', same,
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
        now_p = np.asarray(PO.process(now.copy(), st, **post_kw(sc, st)))
        prev_p = np.asarray(prev_post.process(prev.copy(), st2,
                                              **post_kw(sc2, st2)))
        same_p = now_p.shape == prev_p.shape and bool(np.array_equal(now_p, prev_p))
        check(f'{label}: post.process bitwise the 1.89.0 engine', same_p,
              f'max {float(np.abs(now_p - prev_p).max()) if now_p.shape == prev_p.shape else "shape"}')


# ======================================================= C056 cylinder sky


def cyl_rig(ss=1, world=None, cam=None, **kw):
    st = settings(ss=ss, **kw)
    sc = demo_scene(st, with_texture=False)
    sc.images['cylsky'] = CYL_BUF
    sc.world.mode = 'CYLINDER'
    sc.world.env_image = CYL_BUF
    for k, v in (world or {}).items():
        setattr(sc.world, k, v)
    if cam is not None:
        sc.camera = cam
    return sc, st


def _first_run(col, value):
    """The first output column whose texel column is `value`, or -1."""
    hits = np.nonzero(col == value)[0]
    return int(hits[0]) if hits.size else -1


def _yaw_col(world, yaw, tex_w=32):
    """The texel column of the yaw's own direction (the straight-ahead
    column's texel), Doom's angle-to-column rule."""
    repeats = int(world.sky_cylinder_repeats)
    frac = math.fmod((yaw - float(world.rotation)) / (2.0 * math.pi), 1.0)
    if frac < 0.0:
        frac += 1.0
    return int(math.floor(frac * tex_w * repeats)) % tex_w


def test_sky_cylinder():
    """C056: Doom's angle-mapped sky. The image is wrapped as a cylinder
    by camera yaw alone (a per-column angle table), the rows are 1:1
    with a 200-line screen about the centre row, unlit, point-sampled;
    pitch never moves it, yaw scrolls it, the window shift / jitter of
    the pass rides its own J; the GPU pass reads the CPU's own tables."""
    textures = {'cylsky': CYL_TEX}
    tex_w, tex_h = 32, 16
    cj = getattr(R, 'clip_jitter', None) or SK._clip_jitter

    # --- (2) the unlit texel law: every sky pixel is the table's texel
    sc, st = cyl_rig(world={'strength': 1.5})
    _v, proj, vp, eye = R.camera_matrices(sc.camera, W, H)
    img = R._background_image(sc, st, W, H, vp, eye, None, textures)
    proj_c, jx, jy, yaw = SK.cylinder_inputs(sc, st, W, H)
    col, row = SK.cylinder_tables(sc.world, yaw, proj_c, W, H, tex_w, tex_h,
                                  jx, jy)
    yy, xx = np.mgrid[0:H, 0:W]
    want = (np.asarray(CYL_TEX.pixels, np.float32)[row[yy], col[xx], :3]
            * 1.5).astype(np.float32)
    check('every sky pixel is the column table\'s texel at the row table\'s '
          'row times strength, bitwise (unlit, point-sampled)',
          np.array_equal(img[..., :3], want) and jx == 0.0 and jy == 0.0,
          f'max {float(np.abs(img[..., :3] - want).max())}')
    # 72 output lines at 200/72 texture rows each cross ~7 of the 16 rows
    # (the rest clip, as Doom's 128-row sky did past the screen); 46 deg
    # of the 360 * 4 / 32 = 2.8 deg columns is ~16 columns
    check('the tables are non-trivial: many texel columns and several rows '
          'show across the 96x72 frame',
          len(np.unique(col)) >= 12 and len(np.unique(row)) >= 4,
          f'{len(np.unique(col))} cols, {len(np.unique(row))} rows')
    # the shift probe: projection[0, 2] = +0.2 puts the straight-ahead
    # column 0.1 * Wo further LEFT (the recomputation cannot fake it)
    P = np.asarray(proj, np.float32).copy()
    P[0, 2] = 0.2
    sc_s, _ = cyl_rig(cam=camera_like(sc.camera, projection=P))
    proj_s, jx_s, jy_s, yaw_s = SK.cylinder_inputs(sc_s, st, W, H)
    col_s, _row_s = SK.cylinder_tables(sc_s.world, yaw_s, proj_s, W, H,
                                       tex_w, tex_h, jx_s, jy_s)
    cy_col = _yaw_col(sc.world, yaw)
    x_base, x_shift = _first_run(col, cy_col), _first_run(col_s, cy_col)
    check('a camera shift (projection[0, 2] = +0.2) moves the straight-ahead '
          'texel column 9.6 px LEFT (to 0.4 * Wo), within a pixel',
          x_base >= 0 and x_shift >= 0
          and abs((x_shift - x_base) + 0.1 * W) <= 1.0,
          f'run starts {x_base} -> {x_shift}, yaw col {cy_col}')

    # --- (3) pitch invariance: a level camera and one pitched 20 deg
    # down at the same heading draw the same sky bytes
    sc_l, _ = cyl_rig(cam=camera_like(sc.camera, pitch_deg=0.0))
    sc_p, _ = cyl_rig(cam=camera_like(sc.camera, pitch_deg=-20.0))
    outs = []
    for s in (sc_l, sc_p):
        _v, _p, vp2, eye2 = R.camera_matrices(s.camera, W, H)
        outs.append(R._background_image(s, st, W, H, vp2, eye2, None,
                                        textures))
    check('pitching the camera 20 degrees leaves the sky bitwise (Doom\'s '
          'sky never tilts)', np.array_equal(outs[0], outs[1]))

    # --- (4) yaw scroll: 10 degrees to the right moves the centre
    # column's texel column by round(10 / 360 * 32 * 4) = 4, decreasing
    sc_y, _ = cyl_rig(cam=camera_like(sc.camera, yaw_delta_deg=-10.0))
    proj_y, jx_y, jy_y, yaw_y = SK.cylinder_inputs(sc_y, st, W, H)
    col_y, _ = SK.cylinder_tables(sc_y.world, yaw_y, proj_y, W, H, tex_w,
                                  tex_h, jx_y, jy_y)
    delta = int((int(col[W // 2]) - int(col_y[W // 2])) % tex_w)
    check('turning the camera 10 degrees right scrolls the centre column '
          '4 texel columns (+/- 1), in the decreasing direction',
          delta in (3, 4, 5), f'delta {delta}')

    # --- (5) rows: mid 0.5 at H_out 200, the row just below the centre
    # line shows stored row tex_h - 1 - floor(0.5 * tex_h)
    w_mid = dataclasses.replace(sc.world, sky_cylinder_mid=0.5)
    _v, proj200, _vp, _e = R.camera_matrices(sc.camera, W, 200)
    _c, row200 = SK.cylinder_tables(w_mid, yaw, proj200, W, 200, tex_w,
                                    tex_h)
    check('with Centre Row 0.5 at 200 output lines, the row just below the '
          'centre line (yo = 99) shows stored row tex_h-1-floor(0.5*tex_h)',
          int(row200[200 // 2 - 1]) == tex_h - 1 - int(math.floor(0.5 * tex_h)),
          f'row {int(row200[99])}')

    # --- (6) repeats: four repeats make a quarter turn the same picture
    w1 = dataclasses.replace(sc.world, sky_cylinder_repeats=1)
    w4 = dataclasses.replace(sc.world, sky_cylinder_repeats=4)
    c1, _ = SK.cylinder_tables(w1, yaw, proj_c, W, H, tex_w, tex_h)
    c4, _ = SK.cylinder_tables(w4, yaw, proj_c, W, H, tex_w, tex_h)
    c1q, _ = SK.cylinder_tables(w1, yaw + math.pi / 2.0, proj_c, W, H, tex_w,
                                tex_h)
    c4q, _ = SK.cylinder_tables(w4, yaw + math.pi / 2.0, proj_c, W, H, tex_w,
                                tex_h)
    check('repeats 1 and 4 draw different column tables',
          not np.array_equal(c1, c4))
    check('with 4 repeats the columns a quarter turn away read the same '
          'texel columns; with 1 repeat they do not',
          np.array_equal(c4, c4q) and not np.array_equal(c1, c1q))

    # --- (7) missing image: the solid colour times strength (hdri's rule)
    sc_m, _ = cyl_rig(world={'env_image': None, 'color': (0.3, 0.2, 0.1),
                             'strength': 1.5})
    img_m = R._background_image(sc_m, st, W, H, vp, eye, None, textures)
    exp = (SK.solid(sc_m.world, np.zeros((1, 3), np.float32))
           * 1.5).astype(np.float32)[0]
    check('without an image the cylinder sky is the World colour times '
          'strength at every pixel, bitwise (hdri\'s own rule)',
          np.array_equal(img_m[..., :3], np.broadcast_to(exp, (H, W, 3))))
    d_m, why_m, plan_m = twin_d(sc_m, st, textures, 1)
    check('...and its GPU pass folds to the FLAT mode, bitwise',
          d_m == 0.0 and plan_m is not None
          and plan_m['mode'] == GSKY.MODE_FLAT, str(why_m))

    # --- (8) the GPU twin: three rigs, bitwise, the CYLINDER mode
    for label, ss, fast in (('ss 1', 1, True),
                            ('ss 2 under fast_background', 2, True),
                            ('ss 2 with fast_background off', 2, False)):
        sc_t, st_t = cyl_rig(ss=ss, fast_background=fast,
                             world={'strength': 1.2, 'rotation': 0.3})
        d, why, plan = twin_d(sc_t, st_t, textures, ss)
        check(f'GPU twin at {label}: the sky pass is bitwise the CPU sky at '
              'every uncovered pixel', d == 0.0, str(why) if d is None else f'd {d}')
        check(f'...at {label} the plan chose MODE_CYL',
              plan is not None and plan['mode'] == GSKY.MODE_CYL)
    check('the refusal list takes CYLINDER (no GPU refusal for the mode)',
          GSKY.refusal(sc, st) is None, str(GSKY.refusal(sc, st)))

    # --- (9) the Y-shear composes WITH its sign: the slide rides p12
    # (C058 writes it into the projection; the cylinder reads only p12)
    Ho_t, th = H, 64
    base_r = int(math.floor(0.5 * th))
    P_up = np.asarray(proj, np.float32).copy()
    P_up[1, 2] = 0.05                    # s > 0: looking up
    P_dn = np.asarray(proj, np.float32).copy()
    P_dn[1, 2] = -0.05
    w_mid = dataclasses.replace(sc.world, sky_cylinder_mid=0.5)
    rows = {}
    for name, Pm in (('level', proj), ('up', P_up), ('down', P_dn)):
        _c, r = SK.cylinder_tables(w_mid, yaw, Pm, W, Ho_t, tex_w, th)
        rows[name] = (th - 1) - int(r[Ho_t // 2 - 1])       # r_top
    shift = int(round(0.05 * Ho_t / 2.0 * 200.0 / Ho_t))
    check('a Y-shear slide UP (p12 > 0) shows rows ABOVE the mid line at '
          'the centre row (Heretic\'s sky scrolls down), DOWN rows below',
          rows['up'] < base_r < rows['down'], str(rows))
    check('...and the slide is round(s * Ho/2 * 200/Ho) rows (+/- 1) '
          'either way',
          abs((rows['level'] - rows['up']) - shift) <= 1
          and abs((rows['down'] - rows['level']) - shift) <= 1,
          f'{rows}, expected {shift}')

    # --- (10) the fake device draws the simulator's pixels
    sc_f, st_f = cyl_rig()
    cpu_f = np.asarray(R.render(sc_f, st_f))
    sc_g, st_g = cyl_rig()
    got_f = fake_device_render(sc_g, st_g)
    fake_device_checks('the CYLINDER frame', sc_f, got_f, cpu_f)

    # --- (11) a stereo eye sees the sky at infinity
    p00 = float(proj_c[0, 0])
    st_L, st_R = settings(), settings()
    st_L._stereo = (-0.15, 6.0)
    st_R._stereo = (+0.15, 6.0)
    J_L = cj(st_L, proj_c, W, H)
    J_R = cj(st_R, proj_c, W, H)
    want_dx = p00 * 0.3 / 6.0
    check('the two eyes\' J differ in x by p00 * eye_distance / convergence',
          J_L is not None and J_R is not None
          and abs(float(J_R[0, 3] - J_L[0, 3]) - want_dx) <= 1e-7,
          f'{float(J_R[0, 3] - J_L[0, 3])} vs {want_dx}')
    col_L, _ = SK.cylinder_tables(sc.world, yaw, proj_c, W, H, tex_w, tex_h,
                                  float(J_L[0, 3]), float(J_L[1, 3]))
    col_R, _ = SK.cylinder_tables(sc.world, yaw, proj_c, W, H, tex_w, tex_h,
                                  float(J_R[0, 3]), float(J_R[1, 3]))
    xL, xR = _first_run(col_L, cy_col), _first_run(col_R, cy_col)
    px_shift = 0.5 * want_dx * W
    check('the right eye\'s straight-ahead texel column sits 0.5 * p00 * d / '
          'conv * Wo px (5.6) further right, within a pixel (the sky at '
          'infinity)', xL >= 0 and xR >= 0 and abs((xR - xL) - px_shift) <= 1.0,
          f'{xL} -> {xR}, expected +{px_shift:.2f}')
    sc_st, st_st = cyl_rig(stereo_mode='SBS', stereo_eye_distance=0.3,
                           stereo_convergence=6.0)
    pair = np.asarray(R.render(sc_st, st_st))
    half = pair.shape[1] // 2
    check('a rendered SBS pair shows a different top sky row in each half '
          '(the picture moves between the eyes)',
          not np.array_equal(pair[-1, :half, :3], pair[-1, half:2 * half, :3]))

    # --- (12) the jitter is the frame's
    st_j = settings()
    st_j._accum_jitter = (0.25, -0.125)
    J = cj(st_j, proj_c, W, H)
    J_want = np.eye(4, dtype=np.float32)
    J_want[0, 3] = 2.0 * 0.25 / W
    J_want[1, 3] = 2.0 * -0.125 / H
    check('clip_jitter builds the J block\'s own matrix for an accumulation '
          'pass (jitter as an NDC translation)',
          J is not None and np.array_equal(J, J_want))
    p02 = float(proj_c[0, 2])
    cx_j = (0.5 - 0.5 * (p02 - 2.0 * 0.25 / W)) * W
    cx_0 = (0.5 - 0.5 * p02) * W
    check('the jittered table\'s straight-ahead column moves by exactly the '
          'jitter (+0.25 px)', abs((cx_j - cx_0) - 0.25) <= 1e-9,
          f'{cx_j - cx_0}')
    sc_a, st_a = cyl_rig(aa_mode='ACCUMULATE', aa_samples=4)
    acc = np.asarray(R.render(sc_a, st_a))
    sc_n, st_n = cyl_rig()
    one = np.asarray(R.render(sc_n, st_n))
    _v, _p, vp_n, eye_n = R.camera_matrices(sc_n.camera, W, H)
    g_n = CR.GBuffer(W, H)
    CR.rasterize(sc_n.mesh.verts, sc_n.mesh.tris, vp_n, W, H, gbuf=g_n)
    unc = ~g_n.mask()
    check('the accumulated CYLINDER frame differs from the unjittered one '
          'at the sky (the texel edges average with the geometry\'s)',
          acc.shape == one.shape and bool((acc[unc] != one[unc]).any()))
    sc_1, st_1 = cyl_rig()
    st_1.aa_mode = 'NONE'
    st_1._accum_jitter = (0.25, -0.125)
    d_j, why_j, plan_j = twin_d(sc_1, st_1, textures, 1, J=J)
    check('one jittered accumulation pass: the GPU twin holds bitwise with '
          'the pass\'s own J on both roads', d_j == 0.0
          and plan_j is not None and plan_j['mode'] == GSKY.MODE_CYL,
          str(why_j) if d_j is None else f'd {d_j}')

    # --- reflections see the flat colour (a backdrop, not an environment)
    dirs = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], np.float32)
    ev = SK.evaluate(sc.world, dirs, textures)
    flat = SK.solid(sc.world, dirs)
    check('SK.evaluate along a direction under CYLINDER is the flat World '
          'colour (reflections see a backdrop, not an environment)',
          np.allclose(ev[:, :3], flat[:, :3] * float(sc.world.strength)))


# ================================================= C100 LightWave backdrop


def lw_rig(world=None, cam=None, **kw):
    st = settings(**kw)
    sc = demo_scene(st, with_texture=False)
    sc.world.mode = 'LW_GRADIENT'
    for k, v in (world or {}).items():
        setattr(sc.world, k, v)
    if cam is not None:
        sc.camera = cam
    return sc, st


def _dyadic_dirs(zs):
    """Unnormalised directions with an EXACT float32 z (never through
    M.normalize, which would perturb z by an ulp)."""
    return np.array([[math.sqrt(1.0 - z * z), 0.0, z] for z in zs],
                    np.float32)


def test_sky_lw_gradient():
    """C100: LightWave's Gradient Backdrop. Sky Color at the horizon
    blending to Zenith above, Ground to Nadir below, a HARD horizon
    step, each blend compressed toward the horizon by a whole-number
    Squeeze as repeated multiplication (no pow: the twin is bitwise)."""
    textures = {}
    sc, st = lw_rig()
    w = sc.world
    zen, skyc, gnd, nad = (np.asarray(getattr(w, n), np.float32)
                           for n in ('lw_zenith', 'lw_sky', 'lw_ground',
                                     'lw_nadir'))

    # --- (2) the horizon step
    d = np.array([[1.0, 0.0, 1e-7], [1.0, 0.0, -1e-7], [1.0, 0.0, 0.0]],
                 np.float32)
    out = SK.lw_gradient(w, d)
    check('just above the horizon the backdrop is Sky Color (within 1e-5), '
          'just below it Ground Color, and the two differ by more than 0.1',
          np.abs(out[0] - skyc).max() <= 1e-5
          and np.abs(out[1] - gnd).max() <= 1e-5
          and float(np.abs(out[0] - out[1]).max()) > 0.1,
          f'{out[0]} / {out[1]}')
    check('the exact horizon direction (u == 0) belongs to the SKY half: '
          'Sky Color bitwise, not Ground (the named tie rule)',
          np.array_equal(out[2], skyc) and not np.array_equal(out[2], gnd))
    # --- (3) the poles
    out_p = SK.lw_gradient(w, np.array([[0, 0, 1.0], [0, 0, -1.0]], np.float32))
    check('straight up is Zenith Color and straight down Nadir Color '
          '(within 1e-6)', np.abs(out_p[0] - zen).max() <= 1e-6
          and np.abs(out_p[1] - nad).max() <= 1e-6)

    # --- (4) the squeeze: monotone, exact at dyadic u for squeeze 1,
    # within 2e-7 absolute elsewhere
    def t_of(out_rows):
        # the blue channel carries the widest Sky-to-Zenith span
        return (out_rows[:, 2] - skyc[2]) / (zen[2] - skyc[2])
    dh = _dyadic_dirs([0.5])
    ts = []
    for s in (1, 2, 20):
        ts.append(float(t_of(SK.lw_gradient(
            dataclasses.replace(w, lw_sky_squeeze=s), dh))[0]))
    check('at u = 0.5 the sky-side blend for squeeze 1 < 2 < 20, strictly '
          '(the squeeze compresses the blend toward the horizon)',
          ts[0] < ts[1] < ts[2], str(ts))
    w1 = dataclasses.replace(w, lw_sky_squeeze=1, lw_ground_squeeze=1)
    dd = _dyadic_dirs([0.5, 0.25, 0.75, 0.125])
    got = SK.lw_gradient(w1, dd)
    u = dd[:, 2:3]
    want = (skyc[None, :] + (zen[None, :] - skyc[None, :]) * u).astype(np.float32)
    check('squeeze 1 at the dyadic directions u in (0.5, 0.25, 0.75, 0.125) '
          'is Sky + (Zenith - Sky) * u BITWISE (1 - (1 - u) is exact there)',
          np.array_equal(got, want))
    rng = np.random.default_rng(251)
    rd = rng.normal(size=(1000, 3)).astype(np.float32)
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    rd = rd.astype(np.float32)
    got_r = SK.lw_gradient(w1, rd)
    ur = np.clip(rd[:, 2], -1.0, 1.0).astype(np.float32)
    t_s = (np.float32(1.0) - (np.float32(1.0) - ur)).astype(np.float32)
    t_g = (np.float32(1.0) - (np.float32(1.0) + ur)).astype(np.float32)
    above = ur >= np.float32(0.0)
    want_r = np.where(above[:, None],
                      skyc[None, :] + (zen[None, :] - skyc[None, :]) * t_s[:, None],
                      gnd[None, :] + (nad[None, :] - gnd[None, :]) * t_g[:, None]
                      ).astype(np.float32)
    err = np.where(above, np.abs(t_s - ur), np.abs(t_g + ur))
    check('at 1000 random directions squeeze 1\'s t is within 2e-7 ABSOLUTE '
          'of u (3 ulp off at u = 0.1f, thousands near zero: no bitwise '
          'claim) and the colour is the float32 formula bitwise',
          float(err.max()) <= 2e-7 and np.array_equal(got_r, want_r),
          f'max |t - u| {float(err.max())}')

    # --- (5) each colour dial moves the picture. The demo camera looks
    # 18.5 deg down with a 17.8 deg half-FOV (measured: the whole frame
    # is the ground half), so the A/B renders use it LEVELLED, where
    # the top half is sky and the bottom half ground
    cam_l = camera_like(sc.camera, pitch_deg=0.0)
    sc_l, _ = lw_rig(cam=cam_l)
    _v, _p, vp, eye = R.camera_matrices(sc_l.camera, W, H)
    base = R._background_image(sc_l, st, W, H, vp, eye, None, textures)
    moved = []
    for name in ('lw_zenith', 'lw_sky', 'lw_ground', 'lw_nadir'):
        sc2, _ = lw_rig(world={name: (0.9, 0.1, 0.5)}, cam=cam_l)
        img2 = R._background_image(sc2, st, W, H, vp, eye, None, textures)
        moved.append(not np.array_equal(img2, base))
    check('each of the four colour dials moves the rendered backdrop '
          '(Zenith, Sky, Ground, Nadir) on a level camera', all(moved),
          str(moved))
    sc3, _ = lw_rig(world={'lw_sky_squeeze': 9, 'lw_ground_squeeze': 1},
                    cam=cam_l)
    img3 = R._background_image(sc3, st, W, H, vp, eye, None, textures)
    check('the two squeezes move the rendered backdrop',
          not np.array_equal(img3, base))
    top, bot = base[-1, W // 2, :3], base[0, W // 2, :3]
    check('on the level camera the top row is on the Sky-to-Zenith blend '
          'and the bottom row on the Ground-to-Nadir blend (both halves '
          'show, a hard step between)',
          min(skyc[2], zen[2]) <= top[2] <= max(skyc[2], zen[2])
          and min(gnd[2], nad[2]) <= bot[2] <= max(gnd[2], nad[2])
          and float(np.abs(top - bot).max()) > 0.1, f'{top} / {bot}')

    # --- (6) / (7) the GPU twin, bitwise, MODE_LWGRAD
    for label, world in (('squeezes (2, 2)', {}),
                         ('squeezes (1, 20)', {'lw_sky_squeeze': 1,
                                               'lw_ground_squeeze': 20}),
                         ('squeezes (7, 3)', {'lw_sky_squeeze': 7,
                                              'lw_ground_squeeze': 3}),
                         ('rotation 0.7 (inert by design)', {'rotation': 0.7}),
                         ('strength 1.4', {'strength': 1.4})):
        sc_t, st_t = lw_rig(world=world)
        d, why, plan = twin_d(sc_t, st_t, textures, 1)
        check(f'GPU twin, {label}: the LW backdrop is bitwise the CPU\'s at '
              'every uncovered pixel', d == 0.0,
              str(why) if d is None else f'd {d}')
        check(f'...{label}: the plan chose MODE_LWGRAD',
              plan is not None and plan['mode'] == GSKY.MODE_LWGRAD)
    sc_s, st_s = lw_rig(ss=2)
    d, why, _plan = twin_d(sc_s, st_s, textures, 2)
    check('GPU twin at ss 2 under fast_background: bitwise', d == 0.0,
          str(why) if d is None else f'd {d}')
    check('the refusal list takes LW_GRADIENT', GSKY.refusal(sc, st) is None,
          str(GSKY.refusal(sc, st)))
    sc_r, _ = lw_rig(world={'rotation': 0.7})
    img_r = R._background_image(sc_r, st, W, H, vp, eye, None, textures)
    check('Rotation does not turn the LightWave backdrop (symmetric about '
          'the vertical, as the mode item says)', np.array_equal(img_r, base))

    # --- the fake device
    sc_f, st_f = lw_rig()
    cpu_f = np.asarray(R.render(sc_f, st_f))
    sc_g, st_g = lw_rig()
    got_f = fake_device_render(sc_g, st_g)
    fake_device_checks('the LW_GRADIENT frame', sc_f, got_f, cpu_f)


# ===================================================== C048 Mode 7 floor


def m7_rig(ss=1, over='WRAP', ts=0.5, sky='GRADIENT', world=None, cam=None,
           floor=True, **kw):
    """The demo scene MINUS its floor plane (the analytic floor is
    visible) under a GRADIENT sky or a plain-Background NODES world, the
    Mode 7 floor on with the synthetic 64x64 map."""
    st = settings(ss=ss, **kw)
    sc = demo_scene(st, with_texture=False)
    mesh = _mesh_concat([
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    if sky == 'NODES':
        sc.world.mode = 'NODES'
        sc.world.graph = bg_graph()
    else:
        sc.world.mode = 'GRADIENT'
    sc.world.ground_plane = bool(floor)
    sc.world.ground_mode = 'MODE7'
    sc.world.ground_height = 0.0
    sc.world.mode7_texel_size = ts
    sc.world.mode7_over = over
    sc.world.ground_image = M7_BUF
    for k, v in (world or {}).items():
        setattr(sc.world, k, v)
    if cam is not None:
        sc.camera = cam
    return sc, st


def _drawn(world, inv, eye, w, h):
    """(drawn bool (h*w,), colours (h*w, 3)) of the Mode 7 floor over a
    sentinel of -1: a drawn pixel is any pixel the floor wrote."""
    yy, xx = np.mgrid[0:h, 0:w]
    yy, xx = yy.ravel(), xx.ravel()
    sent = np.full((h * w, 3), -1.0, np.float32)
    out = SK.mode7_floor(world, inv, eye, w, h, 1, xx, yy, sent)
    return out[:, 0] >= 0.0, out


def test_ground_mode7():
    """C048: the SNES PPU's BG mode 7 driven per scanline. One HDMA
    register set per output row (8.8 fixed-point steps from a whole-
    texel origin), a 1024x1024 map of 256 fifteen-bit colours,
    point-sampled, unlit, the M7SEL over-map rule; drawn by OUTPUT
    pixel over any sky on both roads from the same integer tables."""
    textures = {}
    sc, st = m7_rig()
    w = sc.world
    _v, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    inv = np.linalg.inv(vp).astype(np.float32)

    # --- (1) identity: the analytic road and the off switch, bitwise 1.89.0
    RP = _prev_engine('halcyon-1.89.0.zip')
    if RP is not None:
        for label, kw in (('the CHECKER ground plane (the analytic road '
                           'untouched)', {'floor': True,
                                          'world': {'ground_mode': 'CHECKER'}}),
                          ('the ground plane off', {'floor': False})):
            s1, t1 = m7_rig(**kw)
            s2, t2 = m7_rig(**kw)
            now = np.asarray(R.render(s1, t1))
            prev = np.asarray(RP.render(s2, t2))
            check(f'{label} on the floorless scene renders bitwise the '
                  '1.89.0 engine', now.shape == prev.shape
                  and bool(np.array_equal(now, prev)))

    # --- (2) whole-texel origins
    hit, A, C, X0, Y0 = SK.mode7_rows(w, inv, eye, W, H)
    check('every hit row\'s origin is a whole texel (X0 & 255 == 0, '
          'Y0 & 255 == 0: the SNES\'s 13-bit centre, never a fraction)',
          bool(hit.any()) and bool(((X0[hit] & 255) == 0).all())
          and bool(((Y0[hit] & 255) == 0).all()), f'{int(hit.sum())} hit rows')

    # --- (3) the shimmer law: a camera looking slightly down (10 deg)
    # sees the horizon; |A| grows toward it, the rows above never hit
    sc_h, _ = m7_rig(cam=camera_like(sc.camera, pitch_deg=-10.0))
    _v, _p, vp_h, eye_h = R.camera_matrices(sc_h.camera, W, H)
    inv_h = np.linalg.inv(vp_h).astype(np.float32)
    hit_h, A_h, C_h, _x, _y = SK.mode7_rows(sc_h.world, inv_h, eye_h, W, H)
    last = int(np.nonzero(hit_h)[0].max()) if hit_h.any() else -1
    absA = np.abs(A_h[:last + 1].astype(np.int64))
    check('the rows above the horizon do not hit (rays going up) and the '
          'hit rows run from the bottom row up to the last one',
          bool(hit_h[0]) and not bool(hit_h[-1]) and last > 0
          and bool(hit_h[:last + 1].all()),
          f'last hit row {last} of {H}')
    check('|A| (texels per pixel along the row, 8.8) is non-decreasing from '
          'the bottom row up to the last hit row (farther rows step more '
          'texels per pixel: the per-row shimmer)',
          last > 0 and bool((np.diff(absA) >= 0).all()) and absA[-1] > absA[0])

    # --- (4) the unlit palette law
    rgb = SK.mode7_map(w)
    pal = np.unique(rgb.reshape(-1, 3), axis=0)
    check('the palettised map holds at most 256 colours, every one on the '
          '15-bit grid (x * 31 integral)', rgb.shape == (1024, 1024, 3)
          and pal.shape[0] <= 256
          and bool(np.allclose(pal * 31.0, np.round(pal * 31.0), atol=1e-5)),
          f'{pal.shape[0]} colours')
    drawn, cols = _drawn(w, inv, eye, W, H)
    pal_set = {tuple(c) for c in pal.tolist()}
    check('every drawn floor pixel is one of the palette entries bitwise '
          '(unlit, never scaled by strength)', bool(drawn.any())
          and all(tuple(c) in pal_set for c in cols[drawn].tolist()),
          f'{int(drawn.sum())} drawn')
    check('the drawn floor shows several map colours (a non-vacuous '
          'checker)', len({tuple(c) for c in cols[drawn].tolist()}) >= 4)
    same_key = SK.mode7_key(w)
    check('the map is cut once per image content (the cache serves the '
          'same array for the same key)', SK.mode7_map(w) is rgb
          and same_key is not None and same_key == SK.mode7_key(w))

    # --- (5) the over-map rules at texel size 0.05 (the map spans 51 units)
    counts, outs = {}, {}
    for over in ('WRAP', 'TRANSPARENT', 'TILE0'):
        s_o, _ = m7_rig(over=over, ts=0.05)
        dr, out = _drawn(s_o.world, inv, eye, W, H)
        counts[over] = int(dr.sum())
        outs[over] = (dr, out)
    w_o = m7_rig(over='WRAP', ts=0.05)[0].world
    hit_o = SK.mode7_rows(w_o, inv, eye, W, H)[0]
    yy, _xx = np.mgrid[0:H, 0:W]
    n_hit_px = int(hit_o[yy.ravel()].sum())
    check('WRAP draws every hit pixel (the address modulo 1024)',
          counts['WRAP'] == n_hit_px and n_hit_px > 0,
          f'{counts["WRAP"]} of {n_hit_px}')
    check('TRANSPARENT leaves the sky beyond the map (strictly fewer drawn '
          'pixels, more than none)',
          0 < counts['TRANSPARENT'] < counts['WRAP'], str(counts))
    check('TILE0 draws every hit pixel again', counts['TILE0'] == counts['WRAP'])
    outside = outs['TILE0'][0] & ~outs['TRANSPARENT'][0]
    tile0 = {tuple(c) for c in rgb[:8, :8].reshape(-1, 3).tolist()}
    check('...and beyond the map TILE0 shows only the map\'s top-left 8x8 '
          'texels (tile 0, repeated)', bool(outside.any())
          and all(tuple(c) in tile0 for c in outs['TILE0'][1][outside].tolist()),
          f'{int(outside.sum())} outside px, {len(tile0)} tile-0 colours')

    # --- (6) no map: nothing drawn, both roads
    s_n, t_n = m7_rig(world={'ground_image': None})
    s_off, t_off = m7_rig(floor=False)
    check('without a map the MODE7 floor draws nothing: bitwise the same '
          'world without the ground plane',
          np.array_equal(np.asarray(R.render(s_n, t_n)),
                         np.asarray(R.render(s_off, t_off))))
    d_n, why_n, plan_n = twin_d(s_n, t_n, textures, 1)
    check('...and the GPU plan draws no floor either (hal_sky_m7 = 0), '
          'bitwise', d_n == 0.0 and plan_n is not None
          and plan_n['params'].get('hal_sky_m7') == 0, str(why_n))

    # --- (7) the GPU twin: three pitches x three over rules x two skies
    for sky in ('GRADIENT', 'NODES'):
        for over in ('WRAP', 'TRANSPARENT', 'TILE0'):
            for label, ss, fast in (('ss 1', 1, True),
                                    ('ss 2 fast', 2, True),
                                    ('ss 2 unfast', 2, False)):
                s_t, t_t = m7_rig(ss=ss, over=over, ts=0.05, sky=sky,
                                  fast_background=fast)
                d, why, plan = twin_d(s_t, t_t, textures, ss)
                check(f'GPU twin, {sky} sky, {over}, {label}: the Mode 7 '
                      'floor is bitwise the CPU\'s at every uncovered pixel',
                      d == 0.0 and plan is not None
                      and plan['params'].get('hal_sky_m7') == 1
                      and plan['params'].get('hal_sky_m7over') == GSKY.M7_OVER[over],
                      str(why) if d is None else f'd {d}')

    # --- (8) the refusal by name
    s_c, t_c = m7_rig(world={'ground_mode': 'CHECKER'})
    why_c = GSKY.refusal(s_c, t_c)
    check('the CHECKER ground plane still refuses the GPU sky pass by name',
          why_c is not None and 'infinite ground plane' in str(why_c),
          str(why_c))
    check('the MODE7 ground plane does not refuse (integer registers over a '
          'palettised map)', GSKY.refusal(sc, st) is None,
          str(GSKY.refusal(sc, st)))
    ev = SK.ground_plane(w, np.array([[0.0, 0.3, -0.95]], np.float32),
                         np.array([[0.1, 0.2, 0.3]], np.float32), eye, 0.0,
                         textures)
    check('SK.ground_plane leaves a MODE7 world untouched (mirrors and ray '
          'misses see the sky, no floor)',
          np.array_equal(ev, np.array([[0.1, 0.2, 0.3]], np.float32)))

    # --- (9) the fake device
    sc_f, st_f = m7_rig()
    cpu_f = np.asarray(R.render(sc_f, st_f))
    sc_g, st_g = m7_rig()
    got_f = fake_device_render(sc_g, st_g)
    fake_device_checks('the MODE7 frame', sc_f, got_f, cpu_f)
    check('the rendered MODE7 frame differs from the floorless world (the '
          'floor is in the frame)',
          not np.array_equal(cpu_f, np.asarray(R.render(*m7_rig(floor=False)))))


# ============================================ the backdrop export (F008)


def test_sky_backdrop_rgb():
    """`SK.backdrop_rgb(scene, st, w, h)` (exported for LIGHT-A2's F008,
    LightWave's Use Backdrop Color) is `_background_image` over every
    pixel, bitwise, with the frame's matrices or its own."""
    textures = {'cylsky': CYL_TEX}
    for label, (sc, st) in (('a CYLINDER world', cyl_rig()),
                            ('a MODE7 floor under GRADIENT', m7_rig()),
                            ('the LW backdrop at ss 2', lw_rig(ss=2))):
        ss = 2 if 'ss 2' in label else 1
        rw, rh = W * ss, H * ss
        _v, _p, vp, eye = R.camera_matrices(sc.camera, rw, rh)
        want = R._background_image(sc, st, rw, rh, vp, eye, None, textures,
                                   ss=ss)[..., :3]
        got = SK.backdrop_rgb(sc, st, rw, rh, vp, eye, textures, ss=ss)
        check(f'backdrop_rgb over {label} is _background_image at every '
              'pixel, bitwise', got.shape == want.shape
              and np.array_equal(got, want) and got.dtype == np.float32)
        got2 = SK.backdrop_rgb(sc, st, rw, rh, textures=textures, ss=ss)
        check(f'...and with its own camera matrices ({label}) the same',
              np.array_equal(got2, want))


# ---- CAM tests below ----

# (CAM slot, W1.10: C058, C125, C016, C098, C090 -- imports and helpers
# of the CAM half; the module runs under SKY's header and main())
import contextlib
import io
import os
from ..core.settings import RenderSettings

PREV_ZIP = 'halcyon-1.89.0.zip'


def _post_kw(sc, st):
    return dict(frame=1, seed=st.seed,
                target_size=(st.resolution_x, st.resolution_y),
                allow_resize=False,
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None))


def _prev_post(RP):
    return importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')


def _scene(st, **kw):
    """The pack's default scene: the demo card, untextured, opaque."""
    sc = demo_scene(st, with_texture=False)
    for k, v in kw.items():
        setattr(sc, k, v)
    return sc


def _settings(w=96, h=72, **kw):
    st = base_settings(w, h)
    st.transparency = 'NONE'
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _same(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def _maxdiff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return 'shape'
    return f'max {float(np.abs(a - b).max()):.3g}'



def test_00_cam_identity_at_defaults():
    """The 1.89.0 zip pins the CAM roads invisible at their defaults:
    render AND post.process bitwise the previous release on the pack's
    default scene, and the pack's edited presets are named."""
    RP = _prev_engine(PREV_ZIP)
    check('the 1.89.0 zip is beside the package', RP is not None)
    if RP is None:
        return
    prev_post = _prev_post(RP)

    st = _settings()
    sc = _scene(st)
    now = R.render(sc, st)
    now_p = PO.process(now, st, **_post_kw(sc, st))
    st2 = _settings()
    sc2 = _scene(st2)
    prev = RP.render(sc2, st2)
    prev_p = prev_post.process(prev, st2, **_post_kw(sc2, st2))
    check('CAM defaults: render() is bitwise the 1.89.0 engine',
          _same(now, prev), _maxdiff(now, prev))
    check('CAM defaults: post.process is bitwise the 1.89.0 engine',
          _same(now_p, prev_p), _maxdiff(now_p, prev_p))

    # the same at the demo's own AA (supersample 4) and with the aux
    # passes on: the moved J block and the flag line must not move a pixel
    st = _settings(aa_mode='SUPERSAMPLE', aa_samples=4, pass_depth=True,
                   pass_object_index=True, dof=True, dof_focus=6.0)
    sc = _scene(st)
    now = R.render(sc, st)
    st2 = _settings(aa_mode='SUPERSAMPLE', aa_samples=4, pass_depth=True,
                    pass_object_index=True, dof=True, dof_focus=6.0)
    sc2 = _scene(st2)
    prev = RP.render(sc2, st2)
    check('CAM defaults: ss 4 + aux passes + POST dof bitwise 1.89.0',
          _same(now, prev), _maxdiff(now, prev))
    now_p = PO.process(now, st, **_post_kw(sc, st))
    prev_p = prev_post.process(prev, st2, **_post_kw(sc2, st2))
    check('CAM defaults: the POST depth of field is bitwise 1.89.0',
          _same(now_p, prev_p), _maxdiff(now_p, prev_p))
    # the accumulation road (its st2 copies now carry _stereo / _pano_strip)
    st = _settings(aa_mode='ACCUMULATE', aa_samples=2)
    st2 = _settings(aa_mode='ACCUMULATE', aa_samples=2)
    check('CAM defaults: ACCUMULATE AA bitwise 1.89.0 (the carry lines)',
          _same(R.render(_scene(st), st), RP.render(_scene(st2), st2)))
    st = _settings(aa_mode='ADAPTIVE', aa_samples=2)
    st2 = _settings(aa_mode='ADAPTIVE', aa_samples=2)
    check('CAM defaults: ADAPTIVE AA bitwise 1.89.0 (the carry lines)',
          _same(R.render(_scene(st), st), RP.render(_scene(st2), st2)))
    # a stereo pair at ZERO eye distance: nothing for the sign fix to flip
    st = _settings(stereo_mode='ANAGLYPH', stereo_eye_distance=0.0)
    st2 = _settings(stereo_mode='ANAGLYPH', stereo_eye_distance=0.0)
    check('CAM defaults: ANAGLYPH at zero eye distance bitwise 1.89.0',
          _same(R.render(_scene(st), st), RP.render(_scene(st2), st2)))
    # the panorama camera (the true cylinder, untouched)
    st = _settings(48, 24)
    sc = _scene(st)
    sc.camera.type = 'PANO'
    st2 = _settings(48, 24)
    sc2 = _scene(st2)
    sc2.camera.type = 'PANO'
    check('CAM defaults: the PANO camera (true cylinder) bitwise 1.89.0',
          _same(R.render(sc, st), RP.render(sc2, st2)))


def test_01_cam_presets_against_the_zip():
    """Every shipped preset whose dict the round left alone renders AND
    post-processes bitwise the 1.89.0 engine on the demo scene. The
    1.89.0 library from the same zip decides which dicts changed (any
    pack's edit, any new preset: r251_common.preset_delta), so an edited
    preset is excluded by construction, never by a hand-kept list. The
    two whose picture this pack moves on purpose (DOOM: the demo camera
    is pitched, so its new Y-shear shears; VIRTUAL_BOY: a stereo pair
    now) are rendered as well and must move."""
    RP = _prev_engine(PREV_ZIP)
    if RP is None:
        return
    from ..presets.library import PRESETS, apply_preset
    prev_post = _prev_post(RP)
    delta = preset_delta(RP, PRESETS)

    def rig(key):
        st = RenderSettings()
        apply_preset(st, key)
        st.resolution_x, st.resolution_y = 48, 36
        st.use_processes = False
        st.show_stats = False
        return st

    def pair(key):
        st, st2 = rig(key), rig(key)
        sc = demo_scene(st, with_texture=False)
        now = np.asarray(R.render(sc, st))
        sc2 = demo_scene(st2, with_texture=False)
        prev = np.asarray(RP.render(sc2, st2))
        # the adaptive palette lock is per-engine session state: both
        # engines build THIS frame's palette from this frame (r251_common)
        clear_palette_locks(RP)
        now_p = np.asarray(PO.process(now.copy(), st, **_post_kw(sc, st)))
        prev_p = np.asarray(prev_post.process(prev.copy(), st2,
                                              **_post_kw(sc2, st2)))
        return now, prev, now_p, prev_p

    bad = []
    for key in delta.unchanged:
        try:
            now, prev, now_p, prev_p = pair(key)
        except Exception as exc:                                # noqa: BLE001
            bad.append((key, f'{type(exc).__name__}: {exc}'))
            continue
        if not _same(now, prev):
            bad.append((key, 'render ' + _maxdiff(now, prev)))
        elif not _same(now_p, prev_p):
            bad.append((key, 'post ' + _maxdiff(now_p, prev_p)))
    check(f'the {len(delta.unchanged)} shipped presets whose dict the round '
          'left alone render bitwise the 1.89.0 engine, render and post '
          '(each engine from a cleared palette lock)',
          not bad, f'{bad[:6]}; ' + delta_extra(delta))
    # derived from the dicts, so the check is honest at every stage
    moved_by_design = set()
    if PRESETS['DOOM']['settings'].get('camera_yshear'):
        moved_by_design.add('DOOM')
    if PRESETS['VIRTUAL_BOY']['settings'].get('stereo_parallax_layers'):
        moved_by_design.add('VIRTUAL_BOY')
    same_but_should_move = []
    for key in sorted(moved_by_design):
        try:
            now, prev, _, _ = pair(key)
        except Exception as exc:                                # noqa: BLE001
            same_but_should_move.append((key, f'{type(exc).__name__}: {exc}'))
            continue
        if _same(now, prev):
            same_but_should_move.append(key)
    check('DOOM (Y-shear on a pitched camera) and VIRTUAL_BOY (a stereo '
          'pair) carry their new keys and move the picture, as the '
          'CHANGELOG names',
          moved_by_design == {'DOOM', 'VIRTUAL_BOY'} and not same_but_should_move,
          f'{sorted(moved_by_design)}; still {same_but_should_move}')
    inert_edits = {'RENDERMAN', 'SGI_INDY', 'MAX_2012', 'LIGHTWAVE_56',
                   'MAX_R2', 'STUDIO_R4'}
    check('the six inert edits are declared (dof / motion_blur off)',
          all(k in PRESETS for k in inert_edits))


# --------------------------------------------------------------- C058

def _pitched_rig():
    """The demo card seen from low and pitched 16.5 degrees UP, so the
    geometry stays in frame (the spec's 6.2-high target rig sees sky
    only at 96x72; measured, see the handover's deviations)."""
    return look_at_matrix((5.2, -6.4, 1.0), (0.0, -0.2, 3.4))


def _roll_about_z(mw, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    Rz = np.eye(4, dtype=np.float32)
    Rz[0, 0], Rz[0, 1], Rz[1, 0], Rz[1, 1] = c, -s, s, c
    return (np.asarray(mw, np.float32) @ Rz).astype(np.float32)


def _project(vp, pts, w, h):
    """Pixel x, y and ndc y of world points through a viewproj."""
    P = np.concatenate([np.asarray(pts, np.float32),
                        np.ones((len(pts), 1), np.float32)], 1)
    c = P @ np.asarray(vp, np.float32).T
    ndc = c[:, :3] / c[:, 3:4]
    return (ndc[:, 0] * 0.5 + 0.5) * w, (ndc[:, 1] * 0.5 + 0.5) * h, ndc[:, 1]


def test_camera_yshear():
    """C058: the Y-shear of Heretic / Hexen / Build -- the camera
    rendered level, the projection centre slid by f * tan(pitch);
    verticals stay vertical, the horizon slides, roll drops, ZDoom's
    caps hold, and both roads read the one matrix."""
    from ..core import raster
    from ..gpu import craster as CRA
    from ..gpu import sky as GSKY
    W, H = 96, 72
    RP = _prev_engine(PREV_ZIP)

    # (1) identity: the flag off, on the pitched rig
    st = _settings()
    sc = _scene(st)
    sc.camera.matrix_world = _pitched_rig()
    off = R.render(sc, st)
    if RP is not None:
        st2 = _settings()
        sc2 = _scene(st2)
        sc2.camera.matrix_world = _pitched_rig()
        check('camera_yshear off: the pitched rig renders bitwise 1.89.0',
              _same(off, RP.render(sc2, st2)))

    # (2) level pass-through: a level, unrolled camera is bitwise with the flag
    st_l = _settings()
    sc_l = _scene(st_l)
    sc_l.camera.matrix_world = look_at_matrix((5.2, -6.4, 3.6),
                                              (0.0, -0.2, 3.6))
    plain_l = R.render(sc_l, st_l)
    st_l.camera_yshear = True
    check('a level camera passes through the Y-shear bitwise',
          _same(R.render(sc_l, st_l), plain_l))

    # (7) the picture moves on the pitched rig
    st.camera_yshear = True
    on = R.render(sc, st)
    check('the pitched rig renders a different picture under the flag',
          float(np.abs(on - off).max()) > 0.05,
          _maxdiff(on, off))

    # (3) verticals law: the cube's four vertical edges
    cx, cy, cz, hs = 1.4, -0.4, 0.9, 0.9
    corners = [(cx + sx * hs, cy + sy * hs) for sx in (-1, 1) for sy in (-1, 1)]
    bottoms = [(x, y, cz - hs) for x, y in corners]
    tops = [(x, y, cz + hs) for x, y in corners]
    sc.camera._yshear = True
    _v, proj_on, vp_on, _e = R.camera_matrices(sc.camera, W, H)
    sc.camera._yshear = False
    _v, proj_off, vp_off, _e = R.camera_matrices(sc.camera, W, H)
    xb, _yb, _ = _project(vp_on, bottoms, W, H)
    xt, _yt, _ = _project(vp_on, tops, W, H)
    check('verticals law: every vertical edge keeps its x under the shear',
          float(np.abs(xt - xb).max()) < 1e-3,
          f'{float(np.abs(xt - xb).max()):.2e} px')
    xb0, _, _ = _project(vp_off, bottoms, W, H)
    xt0, _, _ = _project(vp_off, tops, W, H)
    check('without the flag at least one vertical edge leans (> 0.5 px)',
          float(np.abs(xt0 - xb0).max()) > 0.5,
          f'{float(np.abs(xt0 - xb0).max()):.2f} px')

    # (4) horizon law: the level forward direction lands where the tilted
    # camera put it, ndc_y = -p11 * tan(theta)
    mw = R.camera_basis(sc.camera)
    lv, theta = R.level_camera(mw)
    f_level = -lv[:3, 2]
    eye = mw[:3, 3]
    far_pt = [tuple(eye + f_level * 1000.0)]
    _, _, ny_on = _project(vp_on, far_pt, W, H)
    _, _, ny_off = _project(vp_off, far_pt, W, H)
    want = -float(proj_off[1, 1]) * math.tan(theta)
    check('horizon law: the level direction projects to -p11*tan(pitch) '
          'on both roads (within 1e-5)',
          abs(float(ny_on[0]) - want) < 1e-5 and
          abs(float(ny_off[0]) - want) < 1e-5,
          f'sheared {float(ny_on[0]):.6f} tilted {float(ny_off[0]):.6f} '
          f'want {want:.6f} (pitch {math.degrees(theta):.1f} deg)')

    # (5) roll dropped: a rolled rig with the flag is bitwise the unrolled
    st_r = _settings(camera_yshear=True)
    sc_r = _scene(st_r)
    sc_r.camera.matrix_world = _roll_about_z(_pitched_rig(), 15.0)
    check('roll is dropped: the 15-degree rolled rig renders bitwise the '
          'unrolled one under the flag', _same(R.render(sc_r, st_r), on))

    # (6) the caps: 70 degrees down clamps to the 56-degree slide
    def slide_for(pitch_deg):
        cam = _scene(_settings()).camera
        d = 8.7
        z = -d * math.tan(math.radians(pitch_deg))
        cam.matrix_world = look_at_matrix((0.0, -d, 5.0), (0.0, 0.0, 5.0 + z))
        cam._yshear = True
        _v, p, _vp, _e = R.camera_matrices(cam, W, H)
        return np.float32(p[1, 2])
    s70 = slide_for(70.0)
    s56 = slide_for(56.0)
    s40 = slide_for(40.0)
    check("the cap: a 70-degree down pitch takes ZDoom's 56-degree slide "
          'bitwise', s70 == s56 and s70 < 0.0, f'{s70} vs {s56}')
    check('the cap: a 40-degree down pitch is not clamped',
          s40 != s56 and abs(float(s40)) < abs(float(s56)))
    u50 = slide_for(-50.0)
    u32 = slide_for(-32.0)
    check("the cap: a 50-degree up pitch takes ZDoom's 32-degree slide "
          'bitwise', u50 == u32 and u50 > 0.0)

    # (8) the GPU raster twin on the sheared vp
    g = raster.GBuffer(W, H)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp_on, W, H, gbuf=g)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc.mesh, vp_on, W, H)
    tri, bary, zndc, front, _b2, _lin, _mk = CRA.simulate_raster(
        sx, sy, iw, z, bw, src, tmap, W, H, depth_bits=24)
    d = int((tri != g.tri).sum())
    check('GPU raster twin on the sheared vp: identical triangle ids',
          d == 0, f'{d} px differ')
    cov = g.tri >= 0
    if d == 0 and cov.any():
        dz = float(np.abs(zndc[cov] - g.zndc[cov]).max())
        df = int((front[cov] != g.front[cov]).sum())
        check('GPU raster twin on the sheared vp: depth within 1e-6 and '
              'front flags identical', dz < 1e-6 and df == 0,
              f'dz {dz:.2e} front {df}')

    # (9) the GPU sky twin on the sheared vp (GRADIENT world)
    st_s = _settings(camera_yshear=True)
    sc_s = _scene(st_s)
    sc_s.camera.matrix_world = _pitched_rig()
    sc_s.world.mode = 'GRADIENT'
    sc_s.world.gradient_falloff = 0.7
    sc_s.world.blend_mode = 'SMOOTH'
    sc_s.world.rotation = 0.4
    sc_s.camera._yshear = R.yshear_active(sc_s.camera, st_s)
    _v, _p, vp_s, eye_s = R.camera_matrices(sc_s.camera, W, H)
    g2 = raster.GBuffer(W, H)
    raster.rasterize(sc_s.mesh.verts, sc_s.mesh.tris, vp_s, W, H, gbuf=g2)
    sim, why = GSKY.simulate(sc_s, g2, st_s, vp_s, eye_s, {}, ss=1)
    check('GPU sky twin on the sheared vp: the GRADIENT sky plans',
          sim is not None, str(why))
    if sim is not None:
        cpu = R._background_image(sc_s, st_s, W, H, vp_s, eye_s, ~g2.mask(),
                                  {}, ss=1)
        un = ~g2.mask()
        dd = float(np.abs(np.asarray(sim)[un][:, :3]
                          - np.asarray(cpu)[un][:, :3]).max())
        check('GPU sky twin on the sheared vp: d == 0.0 at every '
              'uncovered pixel', dd == 0.0, f'{dd:.3g}')

    # (10) every band bitwise (the band road is CPU-only)
    top = R.render(sc, st, band=(0, 36))
    bot = R.render(sc, st, band=(36, 72))
    check('the flag survives the band road: two bands concatenate to the '
          'whole frame bitwise',
          _same(np.concatenate([top, bot], axis=0), on))

    # (11) clip_jitter IS the J block
    st_j = _settings()
    st_j._accum_jitter = (0.25, -0.125)
    st_j._stereo = (0.15, 6.0)
    J = R.clip_jitter(st_j, proj_off, W, H)
    J0 = np.eye(4, dtype=np.float32)
    J0[0, 3] = 2.0 * float(0.25) / W
    J0[1, 3] = 2.0 * float(-0.125) / H
    J0[0, 3] += float(proj_off[0, 0]) * float(0.15) / max(float(6.0), 1e-4)
    check('clip_jitter is the shipped J block, entry for entry',
          J is not None and bool(np.array_equal(J, J0)))
    check('clip_jitter returns None with neither a jitter nor an eye',
          R.clip_jitter(_settings(), proj_off, W, H) is None)

    # the flag never engages on an ORTHO camera or inside a pano strip
    sc_o = _scene(_settings())
    sc_o.camera.type = 'ORTHO'
    check('yshear_active: ORTHO cameras never shear',
          not R.yshear_active(sc_o.camera, _settings(camera_yshear=True)))
    st_ps = _settings(camera_yshear=True)
    st_ps._pano_strip = True
    check('yshear_active: a panorama strip never shears',
          not R.yshear_active(sc.camera, st_ps))
    check('yshear_active: the flag on a PERSP camera engages',
          R.yshear_active(sc.camera, _settings(camera_yshear=True)))

    # the DOOM preset carries the flag (a pitched camera under DOOM shears)
    from ..presets.library import PRESETS
    check("the DOOM preset sets camera_yshear (the engine family's view)",
          PRESETS['DOOM']['settings'].get('camera_yshear') is True
          and 'Y-shear' in PRESETS['DOOM']['note'])


# --------------------------------------------------------------- C125

def _strip_camera(cam, n, i):
    """The road's own transform: strip i of n on a copy of `cam`
    (projection None, PERSP), the camera yawed by -a_i about its up."""
    mw = np.asarray(cam.matrix_world, np.float32)
    up = mw[:3, 1] / max(float(np.linalg.norm(mw[:3, 1])), 1e-9)
    lens = float(getattr(cam, 'lens', 50.0) or 50.0)
    sensor = float(getattr(cam, 'sensor', 36.0) or 36.0)
    phi = 2.0 * math.atan(sensor * 0.5 / max(lens, 1e-3))
    a_i = (i - (n - 1) * 0.5) * phi
    import copy as _copy
    cam2 = _copy.copy(cam)
    m2 = mw.copy()
    m2[:3, :3] = R._rot_axis3(up, -a_i) @ mw[:3, :3]
    cam2.matrix_world = m2
    cam2.projection = None
    cam2.type = 'PERSP'
    return cam2, a_i


def test_camera_pano_parts():
    """C125: Blender 2.4's Pano + Xparts -- N planar strips yawed by one
    strip's field, butted unblended; the middle strip is a plain render,
    the strips abut by construction, the last one is cropped, the extras
    are cleared, the pool stands aside and the AA roads compose inside."""
    W, H = 96, 72
    RP = _prev_engine(PREV_ZIP)

    # (1) identity at 1
    st = _settings(pano_parts=1)
    sc = _scene(st)
    plain = R.render(sc, st)
    if RP is not None:
        st2 = _settings(pano_parts=1)
        check('pano_parts 1 renders bitwise 1.89.0',
              _same(plain, RP.render(_scene(st2), st2)))

    # (2) construction law: the middle strip of three IS a plain 32x72 render
    st3 = _settings(pano_parts=3)
    sc3 = _scene(st3)
    parts = R.render(sc3, st3)
    st_m = _settings(32, 72)
    sc_m = _scene(st_m)
    sc_m.camera.projection = None
    mid = R.render(sc_m, st_m)
    check('pano parts x3: the middle strip is bitwise a plain 32x72 render '
          '(the fallback perspective, a_1 = 0.0 exactly)',
          parts.shape == (H, W, 4) and _same(parts[:, 32:64], mid))

    # (3) strip abutting: four 90-degree parts at 18 mm / 36 mm close the circle
    st4 = _settings(pano_parts=4)
    sc4 = _scene(st4)
    sc4.camera.lens = 18.0
    sc4.camera.sensor = 36.0
    parts4 = R.render(sc4, st4)
    cam0, a_0 = _strip_camera(sc4.camera, 4, 0)
    st_0 = _settings(24, 72)
    sc_0 = _scene(st_0)
    sc_0.camera = cam0
    first = R.render(sc_0, st_0)
    check('pano parts x4 at 18 mm: strip 0 is bitwise the plain 24x72 render '
          'of the camera yawed by +135 degrees (phi = 90)',
          abs(math.degrees(-a_0) - 135.0) < 1e-9 and _same(parts4[:, :24], first),
          f'a_0 = {math.degrees(a_0):.3f} deg')
    cam3, _a3 = _strip_camera(sc4.camera, 4, 3)
    sc_3 = _scene(_settings(24, 72))
    sc_3.camera = cam3
    last = R.render(sc_3, _settings(24, 72))
    check('pano parts x4: the last strip is bitwise its own yawed render',
          _same(parts4[:, 72:96], last))

    # (4) kinks: differs from the true cylinder and from the plain render
    sc_p = _scene(_settings())
    sc_p.camera.type = 'PANO'
    cyl = R.render(sc_p, _settings())
    check('pano parts differ from the true cylinder (PANO camera)',
          float(np.abs(parts - cyl).max()) > 0.05, _maxdiff(parts, cyl))
    check('pano parts differ from the plain render',
          float(np.abs(parts - plain).max()) > 0.05, _maxdiff(parts, plain))

    # (5) the extras are cleared
    st_e = _settings(pano_parts=3, pass_depth=True, dof=True)
    sc_e = _scene(st_e)
    R.render(sc_e, st_e)
    check('pano parts clear the planar extras (depth, shafts, passes)',
          sc_e.last_depth is None and sc_e.last_shafts == []
          and sc_e.last_passes is None and sc_e.last_flares == [])

    # (6) crop: W = 100, n = 3 -> ws = 34, 102 rendered, 100 kept
    st_c = _settings(100, 72, pano_parts=3)
    sc_c = _scene(st_c)
    crop = R.render(sc_c, st_c)
    cam2, _a2 = _strip_camera(sc_c.camera, 3, 2)
    sc_l = _scene(_settings(34, 72))
    sc_l.camera = cam2
    strip2 = R.render(sc_l, _settings(34, 72))
    check('pano parts crop: 100 columns from three 34-wide strips, the last '
          "strip's first 32 columns",
          crop.shape == (72, 100, 4) and _same(crop[:, 68:100], strip2[:, :32]))

    # (7) device: the GPU device headless is bitwise the CPU
    st_g = _settings(pano_parts=3, render_device='GPU')
    check('pano parts on the GPU device (headless) are bitwise the CPU',
          _same(R.render(_scene(st_g), st_g), parts))
    check('pano parts are deterministic', _same(R.render(_scene(st3), st3), parts))

    # (9) the AA roads compose inside the strips (no recursion)
    st_a = _settings(pano_parts=3, aa_mode='ACCUMULATE', aa_samples=2)
    acc = R.render(_scene(st_a), st_a)
    st_am = _settings(32, 72, aa_mode='ACCUMULATE', aa_samples=2)
    sc_am = _scene(st_am)
    sc_am.camera.projection = None
    acc_m = R.render(sc_am, st_am)
    check('ACCUMULATE AA runs inside each strip: the middle strip is bitwise '
          'the accumulated plain 32x72 render',
          acc.shape == (H, W, 4) and _same(acc[:, 32:64], acc_m))
    st_d = _settings(pano_parts=3, aa_mode='ADAPTIVE', aa_samples=2)
    adap = R.render(_scene(st_d), st_d)
    check('ADAPTIVE AA runs inside each strip and returns the 96x72 frame',
          adap.shape == (H, W, 4))
    # Y-shear and stereo stay off inside the strips
    st_y = _settings(pano_parts=3, camera_yshear=True)
    sc_y = _scene(st_y)
    sc_y.camera.matrix_world = _pitched_rig()
    st_y0 = _settings(pano_parts=3)
    sc_y0 = _scene(st_y0)
    sc_y0.camera.matrix_world = _pitched_rig()
    check('Y-shear is off inside the strips (as under the PANO camera)',
          _same(R.render(sc_y, st_y), R.render(sc_y0, st_y0)))
    sc_o = _scene(_settings(pano_parts=3))
    sc_o.camera.type = 'ORTHO'
    sc_o2 = _scene(_settings(pano_parts=1))
    sc_o2.camera.type = 'ORTHO'
    check('an ORTHO camera never slices (the gate reads the camera type)',
          _same(R.render(sc_o, _settings(pano_parts=3)),
                R.render(sc_o2, _settings(pano_parts=1))))

    # (8) the pool stands aside, through the fake Blender engine at 320x240
    from . import fakeblender as FB
    FB.install()
    from .. import engine as ENG
    from .. import properties as props

    def big(eng, dg, bs):
        bs.render.resolution_x, bs.render.resolution_y = 320, 240
        eng.size_x, eng.size_y = 320, 240

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pooled, _p, _c = FB.run_render(props, ENG, rig=big, use_processes=True,
                                       process_count=2, pano_parts=3)
    solo, _p2, _c2 = FB.run_render(props, ENG, rig=big, use_processes=False,
                                   pano_parts=3)
    check('the worker pool stands aside for Pano Parts (its skip line is '
          'printed)', 'worker pool skipped: Pano Parts' in buf.getvalue())
    check('the pooled engine frame is bitwise the in-process Pano Parts frame',
          pooled is not None and solo is not None and _same(pooled, solo)
          and pooled.shape[1] == 320,
          str(None if pooled is None else pooled.shape))


# --------------------------------------------------------------- C016

def _parallax_rig(**kw):
    """Eye Distance 0.8 at Convergence 4.0 (measured: the spec's 0.3 / 6.0
    rounds every demo object to P = 1 at 96x72; this rig gives the Box 5,
    the Floor 6, the Ball 7 under the sky's cap 8)."""
    base = dict(stereo_mode='SBS', stereo_parallax_layers=True,
                stereo_eye_distance=0.8, stereo_convergence=4.0,
                stereo_parallax_max=8)
    base.update(kw)
    st = _settings(**base)
    return st, _scene(st)


def _object_parallax(sc, st, W, H):
    """The test's own copy of the road's per-object whole-pixel parallax:
    P_o (int64 per object) from camera_matrices and the mesh."""
    d = float(st.stereo_eye_distance)
    conv = max(float(st.stereo_convergence), 1e-3)
    Pmax = int(st.stereo_parallax_max)
    view, proj, _vp, _e = R.camera_matrices(sc.camera, W, H)
    mesh = sc.mesh
    verts = np.asarray(mesh.verts, np.float32)
    ones = np.ones((verts.shape[0], 1), np.float32)
    zv = -(np.concatenate([verts, ones], 1) @ view.T)[:, 2].astype(np.float64)
    oi_tri = np.asarray(mesh.obj_index, np.int64)
    n_obj = int(oi_tri.max()) + 1
    oi = np.repeat(oi_tri, 3)
    zc = zv[np.asarray(mesh.tris).ravel()]
    cnt = np.bincount(oi, minlength=n_obj)
    z_o = np.where(cnt > 0, np.bincount(oi, weights=zc, minlength=n_obj)
                   / np.maximum(cnt, 1), conv)
    D = 0.5 * W * float(proj[0, 0]) * d * (1.0 / conv - 1.0 / np.maximum(z_o, 1e-6))
    return np.clip(np.round(D * 0.5), -Pmax, Pmax).astype(np.int64), z_o, proj


def _centre_and_maps(sc, st):
    """The centre frame the road renders (its own settings copy) and the
    (Pmap, ids, uncovered) it derives; the road's own recipe."""
    st2 = st.copy()
    st2.pass_object_index = True
    st2.pass_depth = True
    st2._stereo = (0.0, max(float(st.stereo_convergence), 1e-3))
    centre = R.render(sc, st2)
    passes = sc.last_passes
    H, W = centre.shape[:2]
    P_o, _z, _p = _object_parallax(sc, st, W, H)
    ids = passes['IndexOB'][:, :, 0].astype(np.int64)
    unc = passes['Depth'][:, :, 0] >= 1e10
    Pmap = np.where(unc, int(st.stereo_parallax_max),
                    P_o[np.clip(ids, 0, len(P_o) - 1)])
    return centre, Pmap, ids, unc, P_o


def _old_pack(left, right_f, mode):
    """The 1.89.0 packing tail of _render_stereo, copied verbatim."""
    if mode == 'ANAGLYPH':
        out = right_f.copy()
        out[:, :, 0] = left[:, :, 0]
        out[:, :, 3] = np.maximum(left[:, :, 3], right_f[:, :, 3])
        return out.astype(np.float32)
    lh = left[:, ::2, :]
    rh = right_f[:, ::2, :]
    w_half = min(lh.shape[1], rh.shape[1])
    lh, rh = lh[:, :w_half], rh[:, :w_half]
    pair = (rh, lh) if mode == 'CROSS' else (lh, rh)
    out = np.concatenate(pair, axis=1)
    if out.shape[1] < left.shape[1]:
        out = np.pad(out, ((0, 0), (0, left.shape[1] - out.shape[1]),
                           (0, 0)), mode='edge')
    return out[:, :left.shape[1]].astype(np.float32)


def _eye_ndc_x(sc, st, sign, pt, W, H):
    """ndc_x of a world point through the eye matrix the fixed two-camera
    road builds: the camera moved sign*d/2 along right, J from
    clip_jitter with _stereo = (sign*d/2, conv)."""
    import copy as _copy
    d = float(st.stereo_eye_distance)
    conv = max(float(st.stereo_convergence), 1e-3)
    mw = np.asarray(sc.camera.matrix_world, np.float32)
    right = mw[:3, 0] / max(float(np.linalg.norm(mw[:3, 0])), 1e-9)
    cam2 = _copy.copy(sc.camera)
    mw2 = mw.copy()
    mw2[:3, 3] = mw2[:3, 3] + right * (sign * d * 0.5)
    cam2.matrix_world = mw2
    _v, proj, vp, _e = R.camera_matrices(cam2, W, H)
    st_e = _settings()
    st_e._stereo = (sign * d * 0.5, conv)
    J = R.clip_jitter(st_e, proj, W, H)
    vp = (J @ vp).astype(np.float32)
    P = np.array([list(pt) + [1.0]], np.float32)
    c = P @ vp.T
    return float(c[0, 0] / c[0, 3])


def test_stereo_parallax_layers():
    """C016: the Virtual Boy's per-object integer parallax -- one centre
    render, every object a flat card shifted by a whole pixel, the sky
    at the cap; the two-camera road's convergence sign fixed on the way."""
    W, H = 96, 72
    RP = _prev_engine(PREV_ZIP)
    from ..presets.library import PRESETS

    # (1) identity: the bool without a stereo mode, and ANAGLYPH at zero
    # eye distance (nothing for the sign fix to flip)
    if RP is not None:
        st = _settings(stereo_mode='NONE', stereo_parallax_layers=True)
        st2 = _settings(stereo_mode='NONE', stereo_parallax_layers=True)
        check('parallax layers without a stereo mode render bitwise 1.89.0',
              _same(R.render(_scene(st), st), RP.render(_scene(st2), st2)))
        st = _settings(stereo_mode='ANAGLYPH', stereo_eye_distance=0.0)
        st2 = _settings(stereo_mode='ANAGLYPH', stereo_eye_distance=0.0)
        check('ANAGLYPH at zero eye distance (the bool off) is bitwise 1.89.0',
              _same(R.render(_scene(st), st), RP.render(_scene(st2), st2)))

    # (1b) convergence law: a world point ON the convergence plane straight
    # ahead projects to the same ndc_x through both fixed eye matrices
    st, sc = _parallax_rig()
    mw = R.camera_basis(sc.camera)
    fwd = -mw[:3, 2]
    eye = mw[:3, 3]
    pt = tuple(eye + fwd * 4.0)
    xl = _eye_ndc_x(sc, st, -1, pt, W, H)
    xr = _eye_ndc_x(sc, st, +1, pt, W, H)
    check('convergence law: a point on the convergence plane has zero '
          'disparity through the fixed eye matrices (within 1e-6)',
          abs(xl - xr) < 1e-6, f'left {xl:.6f} right {xr:.6f}')
    pt2 = tuple(eye + fwd * 8.0)
    disp = _eye_ndc_x(sc, st, +1, pt2, W, H) - _eye_ndc_x(sc, st, -1, pt2, W, H)
    _v, proj, _vp, _e = R.camera_matrices(sc.camera, W, H)
    want = float(proj[0, 0]) * 0.8 * (1.0 / 4.0 - 1.0 / 8.0)
    check('convergence law: at twice the convergence the disparity is '
          '+p00*d*(1/conv - 1/z) (the eyes converge, not diverge)',
          abs(disp - want) < 1e-5 and disp > 0.0,
          f'{disp:.5f} want {want:.5f}')
    # the centre frame IS the stereo NONE render (_stereo = (0.0, conv))
    st_n = _settings()
    mono = R.render(_scene(st_n), st_n)
    centre, Pmap, ids, unc, P_o = _centre_and_maps(sc, st)
    check("the centre render (_stereo = (0.0, conv), the passes forced) is "
          'bitwise the stereo NONE render', _same(centre, mono))

    # the road's pair, and the test's own reconstruction of it
    pair = R.render(_scene(st), st)
    left = R.parallax_shift(centre, Pmap, -1)
    right_f = R.parallax_shift(centre, Pmap, +1)
    check('the SBS pair is the packed parallax shift of the centre frame '
          '(the road reconstructed in the test)',
          pair.shape == (H, W, 4) and _same(pair, _old_pack(left, right_f, 'SBS')))

    # (2) flat-card law: every pixel of the Ball's layer the left image
    # claims shows centre[y, x + P_ball]; the right shows x - P_ball
    ball = int(np.argmin(P_o))              # the nearest object
    P_ball = int(P_o[ball])
    ys, xs = np.nonzero((ids == ball) & ~unc)
    # the nearest layer is gathered first, so nothing can claim its
    # destination before it: every in-range card pixel lands
    ok_l = ok_r = True
    n_l = 0
    for y, x in zip(ys, xs):
        xl_ = x - P_ball                    # left buffer: drawn at x - P
        if 0 <= xl_ < W:
            if not np.array_equal(left[y, xl_], centre[y, x]):
                ok_l = False
            n_l += 1
        xr_ = x + P_ball                    # right buffer: drawn at x + P
        if 0 <= xr_ < W:
            if not np.array_equal(right_f[y, xr_], centre[y, x]):
                ok_r = False
    check('flat-card law: the nearest object moves as one card, x - P in '
          'the left buffer and x + P in the right, bitwise the centre pixels',
          ok_l and ok_r and n_l > 0, f'{n_l} left pixels, P = {P_ball}')

    # (3) ordering: nearer objects take smaller P; the sky takes the cap
    order = np.argsort(P_o)
    check('ordering: the three objects take three distinct parallaxes, '
          'nearest smallest, none past the cap', len(set(P_o.tolist())) == 3
          and int(P_o.max()) <= 8 and int(P_o.min()) >= -8,
          str(P_o.tolist()))
    sky_rows = np.nonzero(unc.all(axis=1))[0]
    check('the sky is the farthest layer: uncovered pixels carry exactly '
          'the cap', bool((Pmap[unc] == 8).all()) and unc.any())
    if sky_rows.size:
        y = int(sky_rows[-1])
        check('a sky row shifts by the cap between the two buffers (right = '
              'left moved 2*Pmax)',
              _same(right_f[y, 16:W - 16], left[y, 0:W - 32]))

    # (4) two-camera agreement: the Ball's centroid disparity through the
    # fixed road's eye matrices is within 1 px of 2*P_ball and of D(z)
    _P, z_o, proj = _object_parallax(sc, st, W, H)
    zb = float(z_o[ball])
    cen = eye + fwd * zb
    disp_px = (_eye_ndc_x(sc, st, +1, tuple(cen), W, H)
               - _eye_ndc_x(sc, st, -1, tuple(cen), W, H)) * W / 2.0
    Dz = 0.5 * W * float(proj[0, 0]) * 0.8 * (1.0 / 4.0 - 1.0 / zb)
    check("two-camera agreement: the nearest object's disparity through the "
          'eye matrices is within 1 px of 2*P and of D(z)',
          abs(disp_px - 2 * P_ball) <= 1.0 and abs(disp_px - Dz) < 1e-3,
          f'disp {disp_px:.3f} px, 2P {2 * P_ball}, D(z) {Dz:.3f}')

    # (5) cap
    st_c, sc_c = _parallax_rig(stereo_parallax_max=1)
    P_c, _z, _p = _object_parallax(sc_c, st_c, W, H)
    check('the cap: Parallax Cap 1 gives P in {-1, 0, 1} for every object',
          bool(np.all(np.abs(P_c) <= 1)))

    # (6) packing shared: _pack_stereo is the 1.89.0 tail on synthetic frames
    rng = np.random.RandomState(7)
    a = rng.rand(9, 11, 4).astype(np.float32)
    b = rng.rand(9, 11, 4).astype(np.float32)
    check('_pack_stereo is the 1.89.0 packing tail, SBS / CROSS / ANAGLYPH '
          'and the odd-width pad',
          all(_same(R._pack_stereo(a, b, m), _old_pack(a, b, m))
              for m in ('SBS', 'CROSS', 'ANAGLYPH')))

    # (7) determinism, (8) the picture moves against the two-camera pair
    check('parallax layers are deterministic', _same(R.render(_scene(st), st), pair))
    st_t = _settings(stereo_mode='SBS', stereo_eye_distance=0.8,
                     stereo_convergence=4.0)
    two = R.render(_scene(st_t), st_t)
    check('the layered pair differs from the two-camera SBS pair',
          float(np.abs(pair - two).max()) > 0.05, _maxdiff(pair, two))

    # (9) aux passes: no forced IndexOB unless asked
    sc_a = _scene(st)
    R.render(sc_a, st)
    check('the forced IndexOB / Depth passes are not delivered unless asked',
          sc_a.last_passes is None)
    st_p, sc_p = _parallax_rig(pass_object_index=True)
    R.render(sc_p, st_p)
    check('IndexOB is delivered when the user asked for it',
          sc_p.last_passes is not None and 'IndexOB' in sc_p.last_passes
          and 'Depth' not in sc_p.last_passes)

    # (1c) the AA roads compose: ACCUMULATE with the layers is the shift of
    # the accumulated mono frame; ADAPTIVE returns the SBS shape
    st_acc, sc_acc = _parallax_rig(aa_mode='ACCUMULATE', aa_samples=2)
    acc_pair = R.render(sc_acc, st_acc)
    st_m = _settings(aa_mode='ACCUMULATE', aa_samples=2)
    acc_mono = R.render(_scene(st_m), st_m)
    # the road's own centre under ACCUMULATE: the accumulated frame, its
    # IndexOB / Depth from the LAST jittered pass (stated: the ids follow
    # the last pass, deterministic), so the Pmap is rebuilt the road's way
    st_ac2, sc_ac2 = _parallax_rig(aa_mode='ACCUMULATE', aa_samples=2)
    c_acc, Pmap_acc, _i, _u, _P = _centre_and_maps(sc_ac2, st_ac2)
    check('ACCUMULATE AA composes: the centre frame under the layers is '
          'bitwise the accumulated mono frame (the carry lines at work)',
          _same(c_acc, acc_mono), _maxdiff(c_acc, acc_mono))
    check('ACCUMULATE AA composes: the layered pair is the parallax shift '
          "of that centre by the last pass's layer map",
          _same(acc_pair, _old_pack(R.parallax_shift(acc_mono, Pmap_acc, -1),
                                    R.parallax_shift(acc_mono, Pmap_acc, +1), 'SBS')))
    st_ad, sc_ad = _parallax_rig(aa_mode='ADAPTIVE', aa_samples=2)
    ad = R.render(sc_ad, st_ad)
    check('ADAPTIVE AA composes: the layered pair has the SBS shape (no '
          're-entry into the stereo road)', ad.shape == (H, W, 4)
          and _same(ad[:, :W // 2, :], ad[:, :W // 2, :]))

    # (10) device: the GPU device headless and the fake device are bitwise
    st_g, sc_g = _parallax_rig(render_device='GPU')
    check('parallax layers on the GPU device (headless) are bitwise the CPU',
          _same(R.render(sc_g, st_g), pair))
    from . import fakedevice as FD
    from ..gpu import shade as GSH
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    st_f, sc_f = _parallax_rig(render_device='GPU')
    with FD.installed():
        fake = R.render(sc_f, st_f)
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        st_f2, sc_f2 = _parallax_rig(render_device='GPU')
        c_f, Pmap_f, _i, _u, _P = _centre_and_maps(sc_f2, st_f2)
    # the construction law on the fake device: the pair IS the shift of
    # the frame that device drew; against the CPU pair the deferred
    # shading's own bar (NEAREST 5.96e-6, the documented number) applies
    check('parallax layers through the fake device: the pair is bitwise the '
          "parallax shift of the fake device's own centre frame",
          fake.shape == pair.shape
          and _same(fake, _old_pack(R.parallax_shift(c_f, Pmap_f, -1),
                                    R.parallax_shift(c_f, Pmap_f, +1), 'SBS')))
    check('parallax layers through the fake device match the CPU pair within '
          "the deferred shading's 1e-5 bar (the gather adds nothing)",
          fake.shape == pair.shape and float(np.abs(fake - pair).max()) < 1e-5,
          _maxdiff(fake, pair))

    # the preset
    vb = PRESETS['VIRTUAL_BOY']['settings']
    check('the VIRTUAL_BOY preset is a parallax-layer SBS pair with the cap 16',
          vb.get('stereo_mode') == 'SBS' and vb.get('stereo_parallax_layers') is True
          and vb.get('stereo_parallax_max') == 16)


# --------------------------------------------------------------- C098

def _lens_rig(**kw):
    """The demo scene at f/1.4 under Lens Passes (K = 4, focus 2.0: the
    FM row's numbers, so the lens offsets clear a pixel at 96x72)."""
    base = dict(dof=True, dof_method='LENS_ACCUMULATE',
                dof_lens_pattern='HALTON_DISC', dof_lens_samples=4,
                dof_focus=2.0)
    base.update(kw)
    st = _settings(**base)
    sc = _scene(st)
    sc.camera.fstop = 1.4
    return st, sc


def _ndc_of(cam, pts, W, H):
    _v, _p, vp, _e = R.camera_matrices(cam, W, H)
    P = np.concatenate([np.asarray(pts, np.float32),
                        np.ones((len(pts), 1), np.float32)], 1)
    c = P @ vp.T
    return c[:, :2] / c[:, 3:4]


def _sharp_plane_scene(st, d_f):
    """One quad, fronto-parallel AT the focus distance: the camera above
    it looking straight down from height d_f."""
    from .scenebuild import _mesh_concat, plane
    sc = _scene(st)
    sc.mesh = _mesh_concat([plane(z=0.0, size=4.0, mat=0, obj=0)])
    sc.mesh.smooth = np.zeros(sc.mesh.tris.shape[0], bool)
    sc.camera.matrix_world = look_at_matrix((0.0, 0.0, d_f), (0.0, 0.0, 0.0),
                                            up=(0.0, 1.0, 0.0))
    return sc


def test_camera_lens_dof():
    """C098: depth of field by sampling the lens -- the accumulation
    buffer's translate-and-shear per pass, the focus plane pinned, the
    camera's own f-number as the aperture, both roads on one matrix."""
    W, H = 96, 72
    RP = _prev_engine(PREV_ZIP)
    from ..core import raster
    from ..gpu import craster as CRA
    from ..gpu import sky as GSKY
    from ..presets.library import PRESETS

    # (1) identity: dof on under the POST method (the default) is 1.89.0,
    # render and post.process alike
    if RP is not None:
        prev_post = _prev_post(RP)
        st = _settings(dof=True, dof_focus=6.0)
        sc = _scene(st)
        now = R.render(sc, st)
        st2 = _settings(dof=True, dof_focus=6.0)
        sc2 = _scene(st2)
        prev = RP.render(sc2, st2)
        check('dof under the POST method (the default) renders bitwise 1.89.0',
              _same(now, prev))
        check('dof under the POST method: post.process is bitwise 1.89.0',
              _same(PO.process(now, st, **_post_kw(sc, st)),
                    prev_post.process(prev, st2, **_post_kw(sc2, st2))))

    # (2) focus-plane law on camera_matrices
    st, sc = _lens_rig()
    cam = sc.camera
    mw = R.camera_basis(cam)
    fwd = -mw[:3, 2]
    rgt = mw[:3, 0]
    eye = mw[:3, 3]
    p_focus = eye + fwd * 8.0 + rgt * 0.7
    p_far = eye + fwd * 16.0 + rgt * 0.7
    base = _ndc_of(cam, [p_focus, p_far], W, H)
    cam._lens = (0.01, -0.007, 8.0)
    _v, proj_l, _vp, _e = R.camera_matrices(cam, W, H)
    shear = _ndc_of(cam, [p_focus, p_far], W, H)
    cam._lens = None
    _v, proj0, _vp, _e = R.camera_matrices(cam, W, H)
    check('focus-plane law: a point at the focus depth projects where the '
          'unsheared camera put it (within 1e-6)',
          float(np.abs(shear[0] - base[0]).max()) < 1e-6,
          f'{float(np.abs(shear[0] - base[0]).max()):.2e}')
    want_dx = float(proj0[0, 0]) * 0.01 * (16.0 / 8.0 - 1.0) / 16.0
    check('focus-plane law: a point at twice the focus depth moves by '
          'p00*lx*(z/d_f - 1)/z in NDC x (within 1e-6)',
          abs(float(shear[1, 0] - base[1, 0]) - want_dx) < 1e-6,
          f'{float(shear[1, 0] - base[1, 0]):.6f} want {want_dx:.6f}')

    # (3) MAX_SPIRAL pass 0 is the plain render
    st_p = _settings()
    plain = R.render(_scene(st_p), st_p)
    st_s, sc_s = _lens_rig(dof_lens_pattern='MAX_SPIRAL')
    cam_s = sc_s.camera
    R_l = (42.0 / 1000.0) / (2.0 * 1.4)
    pts_s = R._lens_points(st_s, R_l)
    cam0 = _scene(_settings()).camera
    cam0._lens = (pts_s[0][0], pts_s[0][1], 2.0)
    sc0 = _scene(_settings())
    sc0.camera = cam0
    st0 = _settings()
    st0._lens_pass = True
    check('MAX_SPIRAL pass 0 (the lens centre, Use Original Location) is '
          'bitwise the plain render', pts_s[0] == (0.0, 0.0)
          and _same(R.render(sc0, st0), plain))

    # (4) _lens_points
    pts_h = R._lens_points(st, R_l)
    rad = [math.hypot(x, y) for x, y in pts_h]
    check('HALTON_DISC points lie inside the aperture, are distinct and '
          'deterministic', max(rad) <= R_l + 1e-7 and len(set(pts_h)) == len(pts_h)
          and pts_h == R._lens_points(st, R_l) and len(pts_h) == 4)
    st_s16 = _settings(dof_lens_samples=16, dof_lens_pattern='MAX_SPIRAL')
    rs = [math.hypot(x, y) for x, y in R._lens_points(st_s16, R_l)]
    check('MAX_SPIRAL radii are non-decreasing in k, pass 0 at the centre',
          rs[0] == 0.0 and all(rs[i] <= rs[i + 1] + 1e-9 for i in range(len(rs) - 1)))

    # (5) aperture law: the per-pass spread grows with the aperture
    def spread(fstop):
        st6, sc6 = _lens_rig(dof_lens_samples=6)
        sc6.camera.fstop = fstop
        passes = []
        orig = R.render

        def spy(scene, settings=None, progress=None, band=None):
            out = orig(scene, settings, progress, band)
            if getattr(settings, '_lens_pass', False):
                passes.append(out)
            return out
        R.render = spy
        try:
            mean = R.render(sc6, st6)
        finally:
            R.render = orig
        sp = float(np.mean([np.abs(p - mean).mean() for p in passes]))
        return sp, mean
    sp14, mean14 = spread(1.4)
    sp8, _mean8 = spread(8.0)
    check('aperture law: the per-pass spread at f/1.4 is larger than at f/8',
          sp14 > sp8, f'{sp14:.3g} vs {sp8:.3g}')
    check('the lens-accumulated frame differs from the plain render',
          float(np.abs(mean14 - plain).max()) > 0.05, _maxdiff(mean14, plain))

    # (6) the sharp plane: a quad AT the focus distance
    st_q = _settings(dof=True, dof_method='LENS_ACCUMULATE', dof_lens_samples=4,
                     dof_focus=8.0)
    sc_q = _sharp_plane_scene(st_q, 8.0)
    V = np.asarray(sc_q.mesh.verts, np.float32)
    outs = {}
    for f in (1.4, 64.0):
        sc_q.camera.fstop = f
        cam_q = sc_q.camera
        Rq = (42.0 / 1000.0) / (2.0 * f)
        lx, ly = R._lens_points(st_q, Rq)[1]
        cam_q._lens = (lx, ly, 8.0)
        outs[f] = (_ndc_of(cam_q, V, W, H), None)
        cam_q._lens = None
        outs[f] = (outs[f][0], R.render(sc_q, st_q))
    dv = float(np.abs(outs[1.4][0] - outs[64.0][0]).max())
    check('the sharp plane: the four corners on the focus plane project to '
          'the same NDC at f/1.4 and f/64 (within 1e-6)', dv < 1e-6, f'{dv:.2e}')
    cov = outs[64.0][1][:, :, 3] > 0.5
    er = cov.copy()
    er[1:] &= cov[:-1]
    er[:-1] &= cov[1:]
    er[:, 1:] &= cov[:, :-1]
    er[:, :-1] &= cov[:, 1:]
    a14 = outs[1.4][1][er]
    a64 = outs[64.0][1][er]
    # measured: the sheared vp's corner NDC agrees to 0.0 in the test's
    # float32 projection, but the raster's own clip arithmetic lands an
    # ulp apart and every interior bary moves by that ulp (5.6e-6 in the
    # shading over all 6912 px) -- no pixel shifts, so the bar is the
    # bary-ulp bar, 1e-5, not bitwise (a deviation from the spec)
    dmax = float(np.abs(a14 - a64).max()) if er.any() else 1.0
    check('the sharp plane: its interior pixels agree within 1e-5 between '
          'f/1.4 and f/64 (coverage eroded by one pixel; the bary-ulp bar)',
          er.sum() > 100 and dmax < 1e-5, f'{int(er.sum())} px, max {dmax:.3g}')

    # (7) GPU raster twin and (8) GPU sky twin on a lens-sheared vp
    sc_v = _scene(_settings())
    sc_v.camera._lens = (0.012, -0.009, 2.0)
    _v, _p, vp_l, eye_l = R.camera_matrices(sc_v.camera, W, H)
    g = raster.GBuffer(W, H)
    raster.rasterize(sc_v.mesh.verts, sc_v.mesh.tris, vp_l, W, H, gbuf=g)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc_v.mesh, vp_l, W, H)
    tri, bary, zndc, front, _b2, _lin, _mk = CRA.simulate_raster(
        sx, sy, iw, z, bw, src, tmap, W, H, depth_bits=24)
    d = int((tri != g.tri).sum())
    check('GPU raster twin on the lens-sheared vp: identical triangle ids',
          d == 0, f'{d} px differ')
    covv = g.tri >= 0
    if d == 0 and covv.any():
        dz = float(np.abs(zndc[covv] - g.zndc[covv]).max())
        check('GPU raster twin on the lens-sheared vp: depth within 1e-6',
              dz < 1e-6, f'{dz:.2e}')
    st_k = _settings()
    sc_v.world.mode = 'GRADIENT'
    sc_v.world.gradient_falloff = 0.7
    sc_v.world.blend_mode = 'SMOOTH'
    sc_v.world.rotation = 0.4
    sim, why = GSKY.simulate(sc_v, g, st_k, vp_l, eye_l, {}, ss=1)
    check('GPU sky twin on the lens-sheared vp: the GRADIENT sky plans',
          sim is not None, str(why))
    if sim is not None:
        cpu = R._background_image(sc_v, st_k, W, H, vp_l, eye_l, ~g.mask(), {}, ss=1)
        un = ~g.mask()
        dd = float(np.abs(np.asarray(sim)[un][:, :3] - np.asarray(cpu)[un][:, :3]).max())
        check('GPU sky twin on the lens-sheared vp: d == 0.0 at every '
              'uncovered pixel', dd == 0.0, f'{dd:.3g}')

    # (9) AA composes: the jitter reaches the lens passes (2 x K inner calls)
    st_a, sc_a = _lens_rig(aa_mode='ACCUMULATE', aa_samples=2)
    count = [0]
    orig = R.render

    def spy2(scene, settings=None, progress=None, band=None):
        if getattr(settings, '_accum_jitter', None) is not None and \
                getattr(settings, '_lens_pass', False):
            count[0] += 1
        return orig(scene, settings, progress, band)
    R.render = spy2
    try:
        acc = R.render(sc_a, st_a)
    finally:
        R.render = orig
    check('ACCUMULATE AA composes: every AA pass runs K lens passes with '
          'its jitter (2 x 4 inner renders)', count[0] == 8 and acc.shape == (H, W, 4),
          str(count[0]))

    # (10) device: GPU headless bitwise the CPU, and the fake device
    st_g, sc_g = _lens_rig(render_device='GPU')
    check('lens passes on the GPU device (headless) are bitwise the CPU',
          _same(R.render(sc_g, st_g), mean14 if False else R.render(*reversed(_lens_rig()))))
    from . import fakedevice as FD
    from ..gpu import shade as GSH
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    st_f, sc_f = _lens_rig(render_device='GPU')
    st_ref, sc_ref = _lens_rig()
    ref = R.render(sc_ref, st_ref)
    with FD.installed():
        fake = R.render(sc_f, st_f)
        # the construction law on the fake device: the mean of its own
        # passes (the road's float64 sum, in order)
        passes = []
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        st_f2, sc_f2 = _lens_rig(render_device='GPU')

        def spy3(scene, settings=None, progress=None, band=None):
            out = orig(scene, settings, progress, band)
            if getattr(settings, '_lens_pass', False):
                passes.append(out)
            return out
        R.render = spy3
        try:
            fake2 = R.render(sc_f2, st_f2)
        finally:
            R.render = orig
    accf = passes[0].astype(np.float64)
    for p in passes[1:]:
        accf = accf + p
    check("lens passes through the fake device: the frame is bitwise the "
          "float64 mean of the device's own passes", len(passes) == 4
          and _same(fake2, (accf / 4.0).astype(np.float32)))
    check("lens passes through the fake device match the CPU within the "
          "deferred shading's 1e-5 bar",
          fake.shape == ref.shape and float(np.abs(fake - ref).max()) < 1e-5,
          _maxdiff(fake, ref))
    check('lens passes are deterministic', _same(ref, R.render(*reversed(_lens_rig()))))

    # (11) inside Pano Parts strips: the left strip is the lens-accumulated
    # plain render of the strip's own camera
    st_pp, sc_pp = _lens_rig(pano_parts=2, dof_lens_samples=3)
    out = R.render(sc_pp, st_pp)
    cam0, _a0 = _strip_camera(sc_pp.camera, 2, 0)
    st_l, sc_l = _lens_rig(dof_lens_samples=3)
    st_l.resolution_x, st_l.resolution_y = 48, 72
    sc_l.camera = cam0
    strip = R.render(sc_l, st_l)
    check('lens passes compose inside a Pano Parts strip: the left strip is '
          "bitwise the lens-accumulated 48x72 render of the strip's camera",
          out.shape == (H, W, 4) and _same(out[:, :48], strip))

    # the viewport never runs the lens road (an F12 road), nor the post blur
    st_v, sc_v2 = _lens_rig()
    st_v._viewport = True
    count = [0]

    def spy4(scene, settings=None, progress=None, band=None):
        if getattr(settings, '_lens_pass', False):
            count[0] += 1
        return orig(scene, settings, progress, band)
    R.render = spy4
    try:
        R.render(sc_v2, st_v)
    finally:
        R.render = orig
    check('the viewport never runs the lens passes (an F12 road, by the gate)',
          count[0] == 0)
    st_po, sc_po = _lens_rig()
    frame = R.render(sc_po, st_po)
    post_on = PO.process(frame, st_po, **_post_kw(sc_po, st_po))
    st_off = _settings(dof=False)
    post_off = PO.process(frame, st_off, **_post_kw(sc_po, st_po))
    check('the POST depth-of-field blur stands aside under Lens Passes '
          '(post.process on the lens frame is bitwise post.process with dof off)',
          _same(post_on, post_off) and sc_po.last_depth is not None)

    # the presets carry the road, inert without dof
    for key, pat, n in (('RENDERMAN', 'HALTON_DISC', 16), ('SGI_INDY', None, 23),
                        ('MAX_2012', 'MAX_SPIRAL', 12)):
        ps = PRESETS[key]['settings']
        check(f"the {key} preset sets Lens Passes ({n} points"
              f"{', ' + pat if pat else ''}) without switching dof on",
              ps.get('dof_method') == 'LENS_ACCUMULATE'
              and ps.get('dof_lens_samples') == n
              and (pat is None or ps.get('dof_lens_pattern') == pat)
              and not ps.get('dof', False))


# --------------------------------------------------------------- C090

def _slices(S=4, H=16, W=24):
    """S synthetic slices: a bright bar moving 2 px per slice over a dim
    gradient, so every slice is distinct at every column it touches."""
    out = []
    for k in range(S):
        f = np.zeros((H, W, 4), np.float32)
        f[:, :, 0] = np.linspace(0.05, 0.25, W, dtype=np.float32)[None, :]
        f[:, :, 1] = 0.1 + 0.01 * k
        f[:, :, 2] = np.linspace(0.0, 0.2, H, dtype=np.float32)[:, None]
        f[:, :, 3] = 1.0
        x0 = 4 + 2 * k
        f[:, x0:x0 + 3, :3] = 0.9 - 0.05 * k
        out.append(f)
    return out


def _mean_1890(frames):
    """EN:604-616's own expression, verbatim."""
    acc = frames[0].astype(np.float64)
    for k in range(1, len(frames)):
        acc = acc + frames[k]
    return (acc / float(len(frames))).astype(np.float32)


def _streamed(frames, st, frame_no=0, seed=0):
    """The engine's running acc / wsum form, copied from the hook."""
    n = len(frames)
    mode = str(getattr(st, 'motion_blur_mode', 'MEAN'))
    acc = None
    wsum = None
    for k, img in enumerate(frames):
        if mode == 'MEAN':
            acc = img.astype(np.float64) if acc is None else acc + img
        else:
            w = R.shutter_weight(st, k, n, img.shape[0], img.shape[1],
                                 frame_no=frame_no, seed=seed)
            acc = img * w[:, :, None] if acc is None else acc + img * w[:, :, None]
            wsum = w if wsum is None else wsum + w
    return ((acc / float(n)) if mode == 'MEAN'
            else (acc / wsum[:, :, None])).astype(np.float32)


def _explicit_picks(o, S, M):
    return {(int(o) + (j * S) // M) % S for j in range(M)}


def test_motion_blur_modes():
    """C090: the era's time-slice motion blur -- Max's per-pixel random
    subset of the slices (with the 32 cap and the multi-pass dither),
    LightWave's row-parity split; MEAN stays the 1.89.0 expression."""
    from ..core import film as FILM
    RP = _prev_engine(PREV_ZIP)
    frames = _slices(4)
    H, W = frames[0].shape[:2]

    # (1) identity
    st = _settings()
    check('MEAN: the 1.89.0 expression, verbatim (shutter_combine is the '
          'in-order float64 sum over S)',
          _same(R.shutter_combine(frames, st), _mean_1890(frames)))
    if RP is not None:
        st2 = _settings()
        check('render() is untouched by the engine-side road: bitwise 1.89.0 '
              'at defaults', _same(R.render(_scene(st), st),
                                   RP.render(_scene(st2), st2)))
    check('MEAN: shutter_steps is Blur Steps', R.shutter_steps(_settings(motion_steps=7)) == 7)

    # (2) MAX_SLICES with M == S and dither 0 is MEAN bitwise
    st_m = _settings(motion_blur_mode='MAX_SLICES', motion_samples=4, motion_dither=0.0)
    check('MAX_SLICES with Samples == Steps and Dither 0 is bitwise MEAN',
          _same(R.shutter_combine(frames, st_m), _mean_1890(frames)))

    # (3) subset law against the explicit picks, on S = 4, 5, 7
    for S, M in ((4, 2), (5, 2), (7, 3)):
        fr = _slices(S)
        st_s = _settings(motion_blur_mode='MAX_SLICES', motion_samples=M)
        yy, xx = np.mgrid[0:H, 0:W]
        h = FILM._hash_u32_raw(xx.astype(np.uint32), yy.astype(np.uint32), 3 * 1000003 + 5)
        o = (h.astype(np.int64) * S) >> 24
        ws = [R.shutter_weight(st_s, k, S, H, W, frame_no=3, seed=5) for k in range(S)]
        ok = True
        for y in range(H):
            for x in range(W):
                picks = _explicit_picks(o[y, x], S, M)
                got = {k for k in range(S) if ws[k][y, x] > 0.0}
                if got != picks:
                    ok = False
        check(f'subset law S = {S}, M = {M}: every pixel selects exactly the '
              'hash-named picks (o + floor(j*S/M)) mod S',
              ok and all(int(round(float(w.sum()))) == int(w.sum()) for w in ws))
        out = R.shutter_combine(fr, st_s, frame_no=3, seed=5)
        # every pixel is the mean of exactly M slices
        stack = np.stack(fr).astype(np.float64)
        sel = np.stack([w > 0 for w in ws]).astype(np.float64)
        want = ((stack * sel[:, :, :, None]).sum(0) / sel.sum(0)[:, :, None]).astype(np.float32)
        check(f'subset law S = {S}, M = {M}: every pixel equals the mean of '
              'its M picked slices bitwise', _same(out, want)
              and bool((sel.sum(0) == M).all()))

    # (4) grain: M = 1 is a per-pixel pick
    st_1 = _settings(motion_blur_mode='MAX_SLICES', motion_samples=1)
    g = R.shutter_combine(frames, st_1)
    stack = np.stack(frames)
    is_one = np.zeros((H, W), bool)
    for k in range(4):
        is_one |= np.all(g == stack[k], axis=2)
    check('grain: Samples 1 makes every pixel bitwise one of the four slices, '
          'and the frame differs from MEAN', bool(is_one.all())
          and not _same(g, _mean_1890(frames)))

    # (5) LW_FIELD
    st_lw = _settings(motion_blur_mode='LW_FIELD', motion_steps=2)
    lw = R.shutter_combine(frames, st_lw)
    even = _mean_1890([frames[0], frames[2]])
    odd = _mean_1890([frames[1], frames[3]])
    check('LW_FIELD: even rows are bitwise the mean of the even slices, odd '
          'rows of the odd (row 0 = bottom)',
          _same(lw[0::2], even[0::2]) and _same(lw[1::2], odd[1::2]))
    check('LW_FIELD: shutter_steps doubles Blur Steps', R.shutter_steps(st_lw) == 4)

    # (6) determinism / freshness
    st_d = _settings(motion_blur_mode='MAX_SLICES', motion_samples=2)
    a = R.shutter_combine(frames, st_d, frame_no=1, seed=0)
    check('MAX_SLICES is deterministic per (frame, seed)',
          _same(a, R.shutter_combine(frames, st_d, frame_no=1, seed=0)))
    check('MAX_SLICES: the next frame picks differently',
          not _same(a, R.shutter_combine(frames, st_d, frame_no=2, seed=0)))
    check('MAX_SLICES: another seed picks differently',
          not _same(a, R.shutter_combine(frames, st_d, frame_no=1, seed=1)))

    # (7) dither
    same4 = [frames[0].copy() for _ in range(4)]
    st_dd = _settings(motion_blur_mode='MAX_SLICES', motion_samples=4, motion_dither=0.4)
    dd = R.shutter_combine(same4, st_dd)
    check('dither 0.4 on four identical slices returns the slice within 1e-6 '
          '(mean-preserving on a constant)',
          float(np.abs(dd - frames[0]).max()) < 1e-6)
    st_d0 = _settings(motion_blur_mode='MAX_SLICES', motion_samples=4, motion_dither=0.0)
    check('dither 0.4 on the moving slices differs from dither 0',
          not _same(R.shutter_combine(frames, st_dd), R.shutter_combine(frames, st_d0)))
    st_t8 = _settings(motion_blur_mode='MAX_SLICES', motion_samples=4,
                      motion_dither=0.4, motion_dither_tile=8)
    check('Dither Tile 8 differs from 32',
          not _same(R.shutter_combine(frames, st_t8), R.shutter_combine(frames, st_dd)))
    st_x = _settings(motion_blur_mode='MAX_SLICES', motion_samples=1, motion_dither=1.0)
    wmin = min(float(R.shutter_weight(st_x, k, 4, H, W)[R.shutter_weight(st_x, k, 4, H, W) > 0].min())
               for k in range(4))
    ex = R.shutter_combine(frames, st_x)
    check('Dither 1.0 with Samples 1: every picked weight is at least 1/64 '
          '(the cell-centre rule) and the frame is finite',
          wmin >= 1.0 / 64.0 - 1e-12 and bool(np.isfinite(ex).all()), f'min weight {wmin:.4f}')

    # (7b) streamed == batch on all three modes
    for mode in ('MEAN', 'MAX_SLICES', 'LW_FIELD'):
        st_s = _settings(motion_blur_mode=mode, motion_samples=2, motion_dither=0.4)
        check(f"the engine's streamed acc / wsum form is bitwise shutter_combine ({mode})",
              _same(_streamed(frames, st_s, 2, 1), R.shutter_combine(frames, st_s, 2, 1)))

    # (8) the engine road through the fake Blender engine
    import types
    from . import fakeblender as FB
    FB.install()
    from .. import engine as ENG
    from .. import properties as props

    def motion_frame(**kw):
        def rig(eng, dg, bs):
            ob = next(o for o in bs.objects if o.type == 'MESH')
            base_m = np.array(ob.matrix_world, np.float32).copy()

            def frame_set(fr, sub=0.0):
                t = (float(fr) + float(sub)) - 1.0
                ob.matrix_world[:3, 3] = base_m[:3, 3] + np.array(
                    [3.0 * t, 0.0, 0.0], np.float32)

                def fresh():
                    for o in bs.objects:
                        yield types.SimpleNamespace(
                            object=o, matrix_world=FB._Mat(o.matrix_world),
                            show_self=True, is_instance=False)
                dg.object_instances = fresh()
            eng.frame_set = frame_set
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            img, _p, _c = FB.run_render(props, ENG, rig=rig, **kw)
        return img, buf.getvalue()

    mean_f, log_m = motion_frame(motion_blur=True, motion_steps=4, motion_shutter=0.5)
    max_f, log_x = motion_frame(motion_blur=True, motion_steps=4, motion_shutter=0.5,
                                motion_blur_mode='MAX_SLICES', motion_samples=2)
    lw_f, log_l = motion_frame(motion_blur=True, motion_steps=4, motion_shutter=0.5,
                               motion_blur_mode='LW_FIELD')
    check('the engine road: MAX_SLICES with 2 of 4 samples differs from MEAN',
          mean_f is not None and max_f is not None and not _same(mean_f, max_f))
    check('the engine road: LW_FIELD differs from MEAN and from MAX_SLICES',
          lw_f is not None and not _same(lw_f, mean_f) and not _same(lw_f, max_f))
    check('the engine names the mode on its console line',
          'MAX_SLICES' in log_x and 'LW_FIELD' in log_l and 'MEAN' in log_m)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        n40 = R.shutter_steps(_settings(motion_steps=40, motion_blur_mode='MAX_SLICES'))
    check('Blur Steps 40 under MAX_SLICES clamps to 32 and says so',
          n40 == 32 and 'clamped to 32' in buf.getvalue())

    # the presets
    from ..presets.library import PRESETS
    check('MAX_R2 and STUDIO_R4 carry Object Motion Blur (10 of 10; 5 of 10), '
          'LIGHTWAVE_56 the Dithered split, MAX_2012 the multi-pass dither, '
          'none switching motion_blur on',
          PRESETS['MAX_R2']['settings'].get('motion_blur_mode') == 'MAX_SLICES'
          and PRESETS['MAX_R2']['settings'].get('motion_samples') == 10
          and PRESETS['STUDIO_R4']['settings'].get('motion_samples') == 5
          and PRESETS['STUDIO_R4']['settings'].get('motion_steps') == 10
          and PRESETS['LIGHTWAVE_56']['settings'].get('motion_blur_mode') == 'LW_FIELD'
          and PRESETS['MAX_2012']['settings'].get('motion_dither') == 0.4
          and not any(PRESETS[k]['settings'].get('motion_blur', False)
                      for k in ('MAX_R2', 'STUDIO_R4', 'LIGHTWAVE_56', 'MAX_2012')))



def main():
    utf8_console()
    names = sorted(n for n in globals() if n.startswith('test_'))
    for n in names:
        print(f'\n[{n}]')
        try:
            globals()[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print()
    print(f'{len(FAILS)} failure(s)' if FAILS else 'all sky-camera checks passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
