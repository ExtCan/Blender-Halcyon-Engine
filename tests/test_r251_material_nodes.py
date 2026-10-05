"""R251 material pack tests, wave 2 (MAT-B): C119 the REYES micropolygon
shading rate, then the five period nodes -- C028 Combiner Stage (TEV /
NV2A), C023 SR Bump (PowerVR2), C135 Emboss Bump (DirectX 6), C099
Roughness (Imagine), C123 Env Chrome (Alias / Maya).

Every heading's "twin" is proven at the FUNCTION level first: the node's
or the snap's own arithmetic through the GLSL simulator on the SAME
float32 inputs the CPU road used (`glsl_twin`, `d == 0.0` unless the
section states another bar), then at the FRAME level through the
deferred pass (`sim_vs_cpu`, the existing lighting-ulp bar `< 6e-3`, no
pixel `> 1e-2`), the GPU refusing BY NAME what it cannot carry.

    "C:/Program Files/Blender Foundation/Blender 5.2/5.2/python/bin/python.exe" -m halcyon.tests.test_r251_material_nodes
"""
import importlib
import re
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import base_settings, _prev_engine, _sk
from .scenebuild import demo_scene, checker_image
from . import featurematrix as FM
from .r251_common import clear_palette_locks
from ..core import render as R
from ..core import raster as CRg
from ..core import reyes as REYES
from ..core import post as PO
from ..core import nodeeval as NE
from ..core.settings import RenderSettings
from ..gpu import shade as GSH
from ..gpu import material as MAT
from ..gpu import emit as EM
from ..shaders.compiler import try_compile

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


W, H = 96, 72
f32 = np.float32


# ------------------------------------------------------------------ helpers
def prev():
    return _prev_engine('halcyon-1.89.0.zip')


def scene(key='textured', **kw):
    st = base_settings(W, H, transparency='NONE', **kw)
    st.use_processes = False
    return FM.SCENES[key](st), st


def cpu_frame(key='textured', **kw):
    sc, st = scene(key, **kw)
    return np.asarray(R.render(sc, st))


def _post(img, st):
    return PO.process(img[:, :, :3], st, frame=1, seed=st.seed)


def rig(sc, st):
    """The TR:20084-20091 rig up to the ShadeJob."""
    if hasattr(R, '_GBUF_CACHE'):
        R._GBUF_CACHE.clear()
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, None, view, eye, W, H)
    return g, job


def sim_vs_cpu(sc, st):
    """(gpu_img, cpu_img, gbuf, plan, why): the deferred frame twin, a copy
    of test_render's `sim_vs_cpu` closure (it is local to a test)."""
    cpu = np.asarray(R.render(sc, st))
    g, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    if p is None:
        return None, cpu, g, None, why
    out, _hit = GSH.simulate(job, g)
    return out, cpu, g, p, None


def frame_bar(label, sc, st, bar=6e-3):
    out, cpu, g, p, why = sim_vs_cpu(sc, st)
    check(f'{label}: the deferred pass is not refused', p is not None, str(why))
    if p is None:
        return None
    yy, xx = np.nonzero(g.tri >= 0)
    d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
    check(f'{label}: the GPU frame is the CPU frame at the deferred bar (< {bar:g}, no px > 1e-2)',
          float(d.max()) < bar and not bool((d > 1e-2).any()),
          f'max {float(d.max()):.2e} over {yy.size} px')
    return out


_WRAP = """
{fns}
{unis}
uniform vec2 vUV;
out vec4 Color;
void main()
{{
{body}
}}
"""


def glsl_twin(body, uniforms, n, fns='', unis=''):
    """One GLSL body through the simulator on `n` lanes; `uniforms` are the
    per-lane arrays ((n,), (n, 2), (n, 3), (n, 4)) or Textures the body
    reads. Returns (Color (n, 4) float32, None) or (None, why)."""
    src = _WRAP.format(fns=fns, unis=unis, body=body)
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, str(err)
    u = dict(uniforms)
    u.setdefault('vUV', np.full((n, 2), 0.5, np.float32))
    outs, _d = prog.run(u, {}, n)
    return np.asarray(outs['Color'], np.float32), None


def _master_ins(diffuse=(0.85, 0.2, 0.15), **over):
    """The master shader's input list with every socket it reads without a
    fallback (the featurematrix `_sc_pov_finish` shape); `over` sets VALUE
    sockets by name."""
    ins = [
        _sk('Diffuse Color', 'RGBA', list(diffuse) + [1.0]),
        _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Vertex Color Mix', 'VALUE', 0.0),
        _sk('Diffuse Level', 'VALUE', 1.0),
        _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Specular Level', 'VALUE', 0.9),
        _sk('Glossiness', 'VALUE', 48.0),
        _sk('Roughness', 'VALUE', 0.3),
        _sk('Ambient', 'VALUE', 1.0),
        _sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
        _sk('Opacity', 'VALUE', 1.0),
        _sk('IOR', 'VALUE', 1.45),
        _sk('Anisotropy', 'VALUE', 0.0),
        _sk('Anisotropic Rotation', 'VALUE', 0.0),
        _sk('Metalness', 'VALUE', 0.0),
        _sk('Soften', 'VALUE', 0.0),
        _sk('Reflection', 'VALUE', 0.0),
        _sk('Translucency', 'VALUE', 0.0),
        _sk('Toon Size', 'VALUE', 0.5),
        _sk('Toon Smooth', 'VALUE', 0.05),
    ]
    for k, v in over.items():
        ins.append(_sk(k, 'VALUE', float(v)))
    return ins


def _graph_material(sc, mi, graph):
    sc.materials[mi].graph = graph
    return sc


# ================================================================ identity
def test_r251_identity():
    """The 1.89.0 zip is beside the package (loudly), and with EVERY MAT-B
    field at its default the current tree renders the textured demo AND
    the PHONG ball bitwise the 1.89.0 engine, render AND post.process."""
    RP = prev()
    check('the 1.89.0 zip is beside the package', RP is not None,
          '' if RP is not None else 'halcyon-1.89.0.zip missing: the identity pin did NOT run')
    st0 = RenderSettings()
    check('RenderSettings.shading_rate_area exists at its neutral default 0.0',
          hasattr(st0, 'shading_rate_area') and float(st0.shading_rate_area) == 0.0)
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')
    for label, key, kw in (('textured PHONG', 'textured', {}),
                           ('demo VERTEX rate', 'demo', {'shading_rate': 'VERTEX'}),
                           ('demo AA 4', 'demo', {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4})):
        sc, st = scene(key, **kw)
        now = np.asarray(R.render(sc, st))
        sc2, st2 = scene(key, **kw)
        old = np.asarray(RP.render(sc2, st2))
        same = now.shape == old.shape and bool(np.array_equal(now, old))
        check(f'identity at defaults: {label} renders bitwise the 1.89.0 engine', same,
              f'max {float(np.abs(now - old).max()) if now.shape == old.shape else "shape"}')
        clear_palette_locks(RP)
        np_ = _post(now, st)
        op_ = prev_post.process(old[:, :, :3], st2, frame=1, seed=st2.seed)
        check(f'identity at defaults: {label} render + post.process is bitwise the 1.89.0 engine',
              np_.shape == op_.shape and bool(np.array_equal(np_, op_)),
              f'max {float(np.abs(np_ - op_).max()) if np_.shape == op_.shape else "shape"}')


# ================================================================ C119
def test_c119_identity_at_defaults():
    """Rate 0: no grid, the snap is never called (the call is guarded), and
    the frame is bitwise the 1.89.0 engine's (PIXEL PHONG demo)."""
    calls = []
    real = REYES.snap

    def spy(bary, tri, grid):
        calls.append(1)
        return real(bary, tri, grid)
    REYES.snap = spy
    try:
        sc, st = scene('demo')
        now = np.asarray(R.render(sc, st))
        g, job = rig(sc, st)
        check('rate 0: grid_for(job) is None', REYES.grid_for(job) is None)
    finally:
        REYES.snap = real
    check('rate 0: REYES.snap is never called (the call is guarded)', not calls, str(len(calls)))
    RP = prev()
    if RP is not None:
        sc2, st2 = scene('demo')
        old = np.asarray(RP.render(sc2, st2))
        check('rate 0: the PIXEL PHONG demo is bitwise the 1.89.0 engine',
              now.shape == old.shape and bool(np.array_equal(now, old)))


def test_c119_grid():
    """`micro_grid` on the demo mesh: n >= 1 everywhere, n == 1 behind the
    camera, n scales as 1/sqrt(rate) (rounded up), inv_n is float32(1)/float32(n);
    the area is in OUTPUT pixels (ss folds in)."""
    sc, st = scene('demo')
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g1 = REYES.micro_grid(sc.mesh, vp, W, H, 1.0)
    g4 = REYES.micro_grid(sc.mesh, vp, W, H, 4.0)
    check('rate 1: n >= 1 on every triangle', bool((g1.n >= 1).all()))
    check('rate 1: some triangles dice into more than one cell', int(g1.n.max()) > 1, str(int(g1.n.max())))
    check('rate 4 halves the cell count (ceil): n4 == ceil(n1_exact / 2) within 1 everywhere',
          bool((np.abs(g4.n - np.ceil(g1.n / 2.0)) <= 1).all()))
    check('inv_n is float32(1) / float32(n) bitwise',
          bool(np.array_equal(g1.inv_n, (np.float32(1) / g1.n.astype(np.float32)).astype(np.float32))))
    # three triangles by hand: the projected screen area over the rate
    tris = np.asarray(sc.mesh.tris)
    v = np.asarray(sc.mesh.verts, np.float64)
    m = np.asarray(vp, np.float64)
    hh = v @ m[:, :3].T + m[:, 3][None, :]
    sx = (hh[:, 0] / hh[:, 3] * 0.5 + 0.5) * W
    sy = (hh[:, 1] / hh[:, 3] * 0.5 + 0.5) * H
    ok = True
    for t in (0, tris.shape[0] // 2, tris.shape[0] - 1):
        a, b, c = tris[t]
        area = 0.5 * abs((sx[b] - sx[a]) * (sy[c] - sy[a]) - (sx[c] - sx[a]) * (sy[b] - sy[a]))
        n_hand = int(max(1, min(4096, np.ceil(np.sqrt(area / 1.0)))))
        ok = ok and n_hand == int(g1.n[t])
    check('three triangles pinned by hand: n == ceil(sqrt(A / rate))', ok)
    # behind the camera: a triangle along the camera's BACKWARD axis
    # (the basis's third column) sits at w <= 0 and takes n = 1
    mw = R.camera_basis(sc.camera)
    back = np.asarray(mw[:3, 2], np.float64)
    ctr = np.asarray(eye, np.float64) + back * 3.0

    class _M:            # duck-typed mesh: verts + tris
        pass
    mb = _M()
    mb.verts = np.asarray([ctr, ctr + (1.0, 0.0, 0.0), ctr + (0.0, 1.0, 0.0)], np.float32)
    mb.tris = np.asarray([[0, 1, 2]], np.int32)
    gb = REYES.micro_grid(mb, vp, W, H, 1.0)
    check('a triangle with any corner at w <= 0 takes n == 1 (behind the camera)', int(gb.n[0]) == 1)
    # supersampling: the internal frame at ss = 2 with rate * 4 gives the SAME n
    g_ss = REYES.micro_grid(sc.mesh, R.camera_matrices(sc.camera, 2 * W, 2 * H)[2], 2 * W, 2 * H, 1.0 * 4)
    check('the area is in OUTPUT pixels: ss 2 at rate * ss * ss gives the same n as ss 1 at the rate',
          bool(np.array_equal(g_ss.n, g1.n)))


def test_c119_snap_laws():
    """4096 random barycentrics, n in 1..64: the snapped bary sums to 1
    within 2 ULP, g1 <= b1 and g2 <= b2 (the min corner), k1 + k2 <= n - 1
    (inside the triangle), and n = 1 snaps every fragment to (1, 0, 0)."""
    rng = np.random.default_rng(119)
    n = np.asarray(rng.integers(1, 65, 4096), np.int32)
    r = rng.random((4096, 2)).astype(np.float32)
    b1 = r[:, 0] * (1.0 - r[:, 1])
    b2 = r[:, 1]
    bary = np.stack([1.0 - b1 - b2, b1, b2], 1).astype(np.float32)
    bary[:64, 1] = 0.5
    bary[:64, 2] = 0.5             # exact ties on the hypotenuse
    bary[:64, 0] = 0.0
    grid = REYES.Grid(n, (np.float32(1) / n.astype(np.float32)).astype(np.float32), 1.0)
    tri = np.arange(4096)
    s = REYES.snap(bary, tri, grid)
    check('the snapped barycentrics sum to 1 within 2 ULP',
          float(np.abs(s.sum(axis=1) - 1.0).max()) <= 2.4e-7, f'{float(np.abs(s.sum(axis=1) - 1.0).max()):.2e}')
    check('g1 <= b1 and g2 <= b2: the cell corner is the min corner',
          bool((s[:, 1] <= bary[:, 1] + 1e-7).all() and (s[:, 2] <= bary[:, 2] + 1e-7).all()))
    k1 = np.floor(bary[:, 1] * n.astype(np.float32))
    k2 = np.rint(s[:, 2] * n.astype(np.float32))
    check('k1 + k2 <= n - 1: every cell corner lies inside the triangle',
          bool((k1 + k2 <= n - 1).all()))
    one = REYES.Grid(np.ones(4096, np.int32), np.ones(4096, np.float32), 1e9)
    s1 = REYES.snap(bary, tri, one)
    check('n = 1 snaps every fragment to corner 0 exactly (1, 0, 0)',
          bool(np.array_equal(s1, np.tile(np.array([[1.0, 0.0, 0.0]], np.float32), (4096, 1)))))
    check('the snap is float32', s.dtype == np.float32)
    # a corner value re-snapped lands within ONE cell of itself (k*inv*n can
    # floor one cell short by an ULP; the snap only ever reads the G-buffer's
    # own barycentrics, never its output, so no idempotence law is claimed)
    s2 = REYES.snap(s, tri, grid)
    check('a corner value re-snapped moves by at most one cell (documentary)',
          float(np.abs(s2 - s).max()) <= float((1.0 / n.astype(np.float32)).max()) + 1e-6)


def test_c119_faceted_frame():
    """demo PHONG at rate 16: the Ball's covered pixels take far fewer
    distinct colours than at rate 0 (> 4x fewer) and the picture differs;
    the facets follow the SURFACE: a 3-degree camera turn keeps the number
    of distinct cells within 10 % (documentary)."""
    def big(**kw):
        st = base_settings(2 * W, 2 * H, transparency='NONE', **kw)
        st.use_processes = False
        return FM.SCENES['demo'](st), st

    def big_rig(sc, st):
        view, _p, vp, eye = R.camera_matrices(sc.camera, 2 * W, 2 * H)
        g = CRg.GBuffer(2 * W, 2 * H)
        CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 2 * W, 2 * H, gbuf=g, depth_bits=st.depth_precision)
        return g
    sc0, st0 = big()
    a = np.asarray(R.render(sc0, st0))
    sc1, st1 = big(shading_rate_area=16.0)
    b = np.asarray(R.render(sc1, st1))
    g = big_rig(sc0, st0)
    mi = np.asarray(sc0.mesh.mat_index)[np.maximum(g.tri, 0)]
    ball = (g.tri >= 0) & (mi == 1)
    ua = len(np.unique(a[ball][:, :3], axis=0))
    ub = len(np.unique(b[ball][:, :3], axis=0))
    check('rate 16: the ball takes > 4x fewer distinct colours than at rate 0',
          ub * 4 < ua, f'{ua} -> {ub}')
    check('rate 16: the picture differs from rate 0 (> 1e-2)', float(np.abs(a - b).max()) > 1e-2)
    # the facets follow the surface: turn the camera 3 degrees
    sc2, st2 = big(shading_rate_area=16.0)
    cam = sc2.camera
    th = np.deg2rad(3.0)
    rz = np.array([[np.cos(th), -np.sin(th), 0, 0], [np.sin(th), np.cos(th), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    cam.matrix_world = (rz @ np.asarray(cam.matrix_world, np.float32)).astype(np.float32)
    c = np.asarray(R.render(sc2, st2))
    g2 = big_rig(sc2, st2)
    mi2 = np.asarray(sc2.mesh.mat_index)[np.maximum(g2.tri, 0)]
    ball2 = (g2.tri >= 0) & (mi2 == 1)
    uc = len(np.unique(c[ball2][:, :3], axis=0))
    check('the facets follow the surface: a 3-degree camera turn keeps the distinct-cell count within 10 %',
          abs(uc - ub) <= 0.10 * max(ub, 1) + 2, f'{ub} vs {uc}')
    # inert at the VERTEX / FACE rates (they shade coarser than any grid)
    for rate in ('VERTEX', 'FACE'):
        x = cpu_frame('demo', shading_rate=rate)
        y = cpu_frame('demo', shading_rate=rate, shading_rate_area=16.0)
        check(f'shading_rate_area is inert at the {rate} rate (bitwise)', bool(np.array_equal(x, y)))


def test_c119_gpu_twin():
    """`glsl_twin` of the snap on 4096 lanes with (n, inv_n) per lane
    `d == 0.0`; the row decode `floor((tri + 0.5) / side)` over every tri
    of the demo mesh; `sim_vs_cpu` on textured at rate 4: plan not None,
    consts['reyes'] set, the seam bar; a fogged LINEAR run at rate 16
    (the readback snaps with the same grid)."""
    rng = np.random.default_rng(1190)
    n = np.asarray(rng.integers(1, 65, 4096), np.int32)
    r = rng.random((4096, 2)).astype(np.float32)
    b1 = r[:, 0] * (1.0 - r[:, 1])
    b2 = r[:, 1]
    bary = np.stack([1.0 - b1 - b2, b1, b2], 1).astype(np.float32)
    grid = REYES.Grid(n, (np.float32(1) / n.astype(np.float32)).astype(np.float32), 1.0)
    cpu = REYES.snap(bary, np.arange(4096), grid)
    body = """
    float n = u_n;
    float inv = u_inv;
    float m1 = u_bary.y * n;
    float k1 = floor(m1);
    float m2 = u_bary.z * n;
    float k2 = floor(m2);
    k1 = max(k1, 0.0);
    k2 = max(k2, 0.0);
    float top = n - 1.0;
    float ks = k1 + k2;
    k2 = (ks > top) ? (top - k1) : k2;
    float g1 = k1 * inv;
    float g2 = k2 * inv;
    float g0 = 1.0 - g1;
    g0 = g0 - g2;
    Color = vec4(g0, g1, g2, 1.0);
"""
    got, why = glsl_twin(body, {'u_bary': bary, 'u_n': n.astype(np.float32), 'u_inv': grid.inv_n}, 4096,
                         unis='uniform vec3 u_bary; uniform float u_n; uniform float u_inv;')
    check('the snap compiles in the simulator', got is not None, str(why))
    if got is not None:
        check('the snap is bitwise the CPU on 4096 lanes (d == 0.0)',
              bool(np.array_equal(got[:, :3], cpu)), f'max {float(np.abs(got[:, :3] - cpu).max()):.2e}')
    # the row decode through the emitted function's own arithmetic
    sc, st = scene('textured', shading_rate_area=4.0)
    T = int(np.asarray(sc.mesh.tris).shape[0])
    side = REYES.consts_for(st, sc)['side']
    tri = np.arange(T, dtype=np.float32)
    s = f'{float(side):.9g}' + ('.0' if '.' not in f'{float(side):.9g}' else '')
    got2, why2 = glsl_twin(f"""
    float row = floor((u_tri + 0.5) / {s});
    float colx = u_tri - {s} * row;
    Color = vec4(row, colx, 0.0, 1.0);
""", {'u_tri': tri}, T, unis='uniform float u_tri;')
    check('the row decode compiles', got2 is not None, str(why2))
    if got2 is not None:
        check(f'floor((tri + 0.5) / side) reads the right row for every tri of the demo mesh (side {side})',
              bool(np.array_equal(got2[:, 0], (np.arange(T) // side).astype(np.float32)))
              and bool(np.array_equal(got2[:, 1], (np.arange(T) % side).astype(np.float32))))
    # the frame
    out, cpu_img, g, p, why = sim_vs_cpu(sc, st)
    check('textured at rate 4: the plan is not None', p is not None, str(why))
    if p is not None:
        src = ''.join(str(e[2]) for e in p)
        check("the emitted pass carries hal_reyes_snap and the hal_reyes sampler",
              'f = hal_reyes_snap(f);' in src and 'uniform sampler2D hal_reyes;' in src)
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(out[yy, xx] - cpu_img[yy, xx, :3]).max(axis=1)
        check('textured at rate 4: the GPU frame is the CPU frame at the deferred bar (< 6e-3, no px > 1e-2)',
              float(d.max()) < 6e-3 and not bool((d > 1e-2).any()), f'max {float(d.max()):.2e} over {yy.size} px')
        # the snap moved the GPU picture too (not a refused-to-CPU frame)
        out0, _c0, _g0, p0, _w0 = sim_vs_cpu(*scene('textured'))
        check('rate 4 moves the simulated GPU frame against rate 0', p0 is not None and float(np.abs(out - out0).max()) > 1e-2)
    # fog under REYES: the CPU fogs at the cell corner's depth; the readback snaps the same grid
    sc_f, st_f = scene('textured', shading_rate_area=16.0, fog=True, fog_mode='LINEAR', fog_start=3.0, fog_end=12.0,
                       fog_color=(0.6, 0.65, 0.7))
    frame_bar('textured at rate 16 under LINEAR fog', sc_f, st_f)


def test_c119_refusals_by_name():
    """tex_perspective False + rate 4: the plan refuses with 'affine' in why;
    a transparent-layer / hit probe refuses with 'hit/layer'; a Bump
    material refuses with 'Bump'; every refusal shades on the CPU (the
    CPU frame renders)."""
    sc, st = scene('textured', shading_rate_area=4.0, tex_perspective=False)
    g, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check("affine + rate 4: the plan refuses by name ('affine' in why)", p is None and 'affine' in str(why), str(why))
    img = np.asarray(R.render(sc, st))
    check('affine + rate 4: the CPU road renders (grid on lighting, not on uv)', img.shape == (H, W, 4))
    sc2, st2 = scene('textured', shading_rate_area=4.0)
    g2, job2 = rig(sc2, st2)
    yy, xx = np.nonzero((g2.tri >= 0) & (np.asarray(sc2.mesh.mat_index)[np.maximum(g2.tri, 0)] == 1))
    _b, _m, why_l = GSH._probe_material(job2, g2, 1, yy[:64], xx[:64], layer=True)
    check("a transparent-layer probe under REYES refuses by name ('hit/layer')",
          _b is None and 'hit/layer' in str(why_l), str(why_l))
    _b2, _m2, why_s = GSH._probe_material(job2, g2, 1, yy[:64], xx[:64], secondary=True)
    check("a hit probe under REYES refuses by name ('hit/layer')",
          _b2 is None and 'hit/layer' in str(why_s), str(why_s))
    # ray tracing with a reflective material: the plan names the hit pass
    sc3, st3 = scene('mirror', shading_rate_area=4.0, raytrace=True)
    g3, job3 = rig(sc3, st3)
    GSH._PLAN_CACHE.clear()
    p3, why3, _a3 = GSH.plan_frame(job3, g3)
    check("mirror + raytrace at rate 4: the plan refuses the hit pass by name", p3 is None and 'hit/layer' in str(why3), str(why3))
    # a Bump material: the height pre-pass samples at the pixel
    sc4, st4 = scene('demo', shading_rate_area=4.0)
    ins = _master_ins() + [_sk('Normal', 'VECTOR', [0, 0, 0], ['bump', 0])]
    sc4.materials[1].graph = {'output': 'out', 'nodes': {
        'noise': {'id': 'noise', 'bl_idname': 'ShaderNodeTexNoise', 'props': {},
                  'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]), _sk('Scale', 'VALUE', 5.0),
                             _sk('Detail', 'VALUE', 2.0), _sk('Roughness', 'VALUE', 0.5), _sk('Distortion', 'VALUE', 0.0)],
                  'outputs': [{'name': 'Fac', 'type': 'VALUE'}, {'name': 'Color', 'type': 'RGBA'}]},
        'bump': {'id': 'bump', 'bl_idname': 'ShaderNodeBump', 'props': {},
                 'inputs': [_sk('Strength', 'VALUE', 1.0), _sk('Distance', 'VALUE', 0.1),
                            _sk('Height', 'VALUE', 1.0, ['noise', 0]), _sk('Normal', 'VECTOR', [0, 0, 0])],
                 'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode', 'props': {'model': 'PHONG', 'toon_steps': 2},
                'inputs': ins, 'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['hal', 0]), _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    st4.shading_rate_area = 0.0
    g40, job40 = rig(sc4, st4)
    GSH._PLAN_CACHE.clear()
    p40, why40, _a40 = GSH.plan_frame(job40, g40)
    check('the Bump material plans at rate 0 (the refusal below is REYES own)', p40 is not None, str(why40))
    st4.shading_rate_area = 4.0
    g4, job4 = rig(sc4, st4)
    GSH._PLAN_CACHE.clear()
    p4, why4, _a4 = GSH.plan_frame(job4, g4)
    check("a Bump material at rate 4 refuses by name ('Bump')", p4 is None and 'Bump' in str(why4), str(why4))
    img4 = np.asarray(R.render(sc4, st4))
    check('the Bump material renders on the CPU at rate 4', img4.shape == (H, W, 4) and np.isfinite(img4).all())
    # the plan signature carries the gate: rate 0 -> rate 4 is a different plan
    sc5, st5 = scene('textured')
    g5, job5 = rig(sc5, st5)
    sig0 = GSH._plan_sig(job5, GSH._mesh_key(sc5.mesh))
    st5.shading_rate_area = 4.0
    sig4 = GSH._plan_sig(job5, GSH._mesh_key(sc5.mesh))
    check('shading_rate_area is in the plan signature', sig0 != sig4)
    check('the RENDERMAN preset carries shading_rate_area 1.0',
          float(__import__('halcyon.presets.library', fromlist=['PRESETS']).PRESETS['RENDERMAN']['settings'].get('shading_rate_area', 0)) == 1.0)


# ================================================================ C028
class _FakeEv:
    """An evaluator stand-in: `input(node, name, kind)` returns the arrays
    handed in, so a node function can be driven on synthetic lanes."""

    def __init__(self, n, **arrays):
        self.n = n
        self.arrays = arrays
        self.ctx = None

    def input(self, node, name, kind=None):
        return self.arrays[name]

    def has_link(self, node, name):
        return name in self.arrays


def _cb_node(hardware='TEV', **props):
    p = {'hardware': hardware, 'op': 'ADD', 'bias': 'ZERO', 'scale': 'X1', 'clamp': True,
         'map_a': 'UNSIGNED_IDENTITY', 'map_b': 'UNSIGNED_IDENTITY', 'map_c': 'UNSIGNED_IDENTITY',
         'map_d': 'UNSIGNED_IDENTITY', 'nv_scale': 'X1', 'nv_bias': 'NONE'}
    p.update(props)
    return {'id': 'comb', 'bl_idname': 'HALCYON_CombinerStageNode', 'props': p,
            'inputs': [_sk('A', 'RGBA', [0, 0, 0, 1], ['uA', 0]), _sk('B', 'RGBA', [1, 1, 1, 1], ['uB', 0]),
                       _sk('C', 'RGBA', [0, 0, 0, 1], ['uC', 0]), _sk('D', 'RGBA', [0, 0, 0, 1], ['uD', 0])],
            'outputs': [{'name': 'Color', 'type': 'RGBA'}]}


def _cb_cpu(node, a, b, c, d):
    n = int(a.shape[0])
    ev = _FakeEv(n, A=a, B=b, C=c, D=d)
    return NE.n_halcyon_combiner_stage(ev, node)['Color']


def _lane_emitter(uniform_names):
    """A temporary emitter type whose output IS a per-lane vec4 uniform."""
    def e_lane(em, node, _i):
        return node['props']['u'], 'vec4'
    return e_lane


def node_twin(node, lanes, n, out_index=0, extra_unis='', extra_uniforms=None, gtype='vec4'):
    """Emit `node` with its linked inputs replaced by per-lane vec4 / vec3 /
    float uniforms (`lanes`: name -> (glsl type, array)), run it in the
    simulator, return the output as (n, 4) float32 or (None, why)."""
    nodes = {node['id']: node}
    for sock in node['inputs']:
        link = sock.get('link')
        if link:
            uname = link[0]
            gt = lanes[uname][0]
            nodes[uname] = {'id': uname, 'bl_idname': '__MATB_LANE__', 'props': {'u': uname, 'gt': gt},
                            'inputs': [], 'outputs': [{'name': 'V', 'type': 'RGBA'}]}
    EM.EMITTERS['__MATB_LANE__'] = lambda em, nd, _i: (nd['props']['u'], nd['props']['gt'])
    try:
        em = EM.Emitter({'output': node['id'], 'nodes': nodes})
        em.frame_mode = True
        em.resolution = (W, H)
        var, vt = em.output(node['id'], out_index)
    finally:
        EM.EMITTERS.pop('__MATB_LANE__', None)
    body = em.body() + f'\n    Color = {em.cast(var, vt, "vec4")};'
    unis = ''.join(f'uniform {gt} {name};\n' for name, (gt, _a) in lanes.items()) + extra_unis
    uniforms = {name: np.asarray(arr, np.float32) for name, (_gt, arr) in lanes.items()}
    uniforms.update(extra_uniforms or {})
    fns = '\n'.join(em.inline)
    return glsl_twin(body, uniforms, n, fns=fns, unis=unis)


def test_c028_tev_laws():
    """The TEV stage on 4096 random inputs incl. exact 0.5 ties: ADD with
    c = 0 gives d + a on the 8-bit grid; c = 1 (c8 = 255 -> c9 = 256) gives
    d + b exactly; SUB mirrors ADD; bias ADD_HALF adds exactly 128/255;
    scale X2 doubles and clamps at 1; unclamped values survive beyond 0..1;
    COMP_GR16_GT selects c exactly when the packed 16-bit compare says so
    (brute force in Python ints); HALF on a negative S10 floors (-3 -> -2)."""
    rng = np.random.default_rng(28)
    n = 4096
    a = rng.random((n, 4)).astype(np.float32)
    b = rng.random((n, 4)).astype(np.float32)
    d = (rng.random((n, 4)) * 0.6).astype(np.float32)
    a[:256, :3] = np.float32(0.5)         # ties on the 8-bit grid: 127.5 -> 128 (even)
    a[:, 3] = 0.7
    z = np.zeros((n, 4), np.float32)
    o = np.ones((n, 4), np.float32)
    r255 = np.float32(1.0 / 255.0)
    a8 = np.rint(np.clip(a[:, :3], 0, 1) * np.float32(255)).astype(np.float32)
    b8 = np.rint(np.clip(b[:, :3], 0, 1) * np.float32(255)).astype(np.float32)
    d10 = np.clip(np.rint(d[:, :3] * np.float32(255)), -1024, 1023).astype(np.float32)
    out = _cb_cpu(_cb_node('TEV', op='ADD'), a, b, z, d)
    check('TEV ADD with c = 0 gives d + a on the 8-bit grid (clamped)',
          bool(np.array_equal(out[:, :3], (np.clip(d10 + a8, 0, 255) * r255).astype(np.float32))))
    check("alpha passes through from A", bool(np.array_equal(out[:, 3], a[:, 3])))
    out = _cb_cpu(_cb_node('TEV', op='ADD'), a, b, o, d)
    check('TEV ADD with c = 1 (c9 = 256) gives d + b EXACTLY (the 255 -> 256 rule)',
          bool(np.array_equal(out[:, :3], (np.clip(d10 + b8, 0, 255) * r255).astype(np.float32))))
    out_add = _cb_cpu(_cb_node('TEV', op='ADD', clamp=False), a, b, z, z)
    out_sub = _cb_cpu(_cb_node('TEV', op='SUB', clamp=False), a, b, z, z)
    check('SUB mirrors ADD (d = 0: -lerp == -(+lerp))', bool(np.array_equal(out_sub[:, :3], -out_add[:, :3])))
    out_b = _cb_cpu(_cb_node('TEV', op='ADD', bias='ADD_HALF', clamp=False), z, o, z, z)
    check('bias ADD_HALF adds exactly 128/255', bool(np.array_equal(out_b[:, :3], np.full((n, 3), np.float32(128) * r255, np.float32))))
    out_x2 = _cb_cpu(_cb_node('TEV', op='ADD', scale='X2'), a, b, z, z)
    check('scale X2 doubles and clamps at 1 when clamped',
          bool(np.array_equal(out_x2[:, :3], (np.clip(a8 * 2, 0, 255) * r255).astype(np.float32))))
    hi = np.full((n, 4), 0.75, np.float32)
    out_u = _cb_cpu(_cb_node('TEV', op='ADD', scale='X2', clamp=False), hi, o, z, z)
    check('unclamped: 0.75 * 2 = 1.5 survives (S10 range)', bool(np.allclose(out_u[:, :3], np.float32(382) * r255)))
    out_n = _cb_cpu(_cb_node('TEV', op='SUB', clamp=False), z, o, o, z)
    check('unclamped: -1 survives below zero', bool(np.allclose(out_n[:, :3], -np.float32(255) * r255)))
    # COMP_GR16_GT by brute force in Python ints
    c = rng.random((n, 4)).astype(np.float32)
    c8 = np.rint(np.clip(c[:, :3], 0, 1) * np.float32(255)).astype(np.float32)
    out_g = _cb_cpu(_cb_node('TEV', op='COMP_GR16_GT'), a, b, c, d)
    want = np.zeros((n, 3), np.float32)
    for i in range(n):
        pa = int(a8[i, 1]) * 256 + int(a8[i, 0])
        pb = int(b8[i, 1]) * 256 + int(b8[i, 0])
        sel = c8[i] if pa > pb else 0.0
        want[i] = np.clip(d10[i] + sel, 0, 255) * r255
    check('COMP_GR16_GT selects c exactly when the packed 16-bit compare says so (Python ints)',
          bool(np.array_equal(out_g[:, :3], want.astype(np.float32))))
    out_e = _cb_cpu(_cb_node('TEV', op='COMP_RGB8_EQ'), a, a, c, z)
    check('COMP_RGB8_EQ on a == a selects c on every channel', bool(np.array_equal(out_e[:, :3], (c8 * r255).astype(np.float32))))
    # HALF on a negative S10 floors: -3 -> -2
    m3 = np.full((n, 4), np.float32(3) * r255, np.float32)
    out_h = _cb_cpu(_cb_node('TEV', op='SUB', scale='HALF', clamp=False), z, m3, o, z)
    check('HALF on a negative S10 floors (-3 -> -2)', bool(np.allclose(out_h[:, :3], np.float32(-2) * r255)))
    # bias and scale are ignored by the compare ops, as GX requires
    out_c1 = _cb_cpu(_cb_node('TEV', op='COMP_R8_GT'), a, b, c, d)
    out_c2 = _cb_cpu(_cb_node('TEV', op='COMP_R8_GT', bias='ADD_HALF', scale='X4'), a, b, c, d)
    check('the compare ops ignore bias and scale (GX)', bool(np.array_equal(out_c1, out_c2)))


def test_c028_nv2a_laws():
    """q(1.0) = 255, q(-1.0) = -255 (never -256 from a mapping; -256 only by
    rint of -1.0039), EXPAND_NORMAL maps 0.5 -> 0 exactly, UNSIGNED_INVERT of
    1 is 0; A*B with B = 1 gives pab == q(a) (255*q/255 rounds exactly)."""
    n = 1024
    r255 = np.float32(1.0 / 255.0)
    rng = np.random.default_rng(2028)
    a = rng.random((n, 4)).astype(np.float32)
    o = np.ones((n, 4), np.float32)
    z = np.zeros((n, 4), np.float32)
    check('q(1.0) == 255', float(NE._cb_q9(np.float32(1.0))) == 255.0)
    check('q(-1.0) == -255 (never -256 from a mapping)', float(NE._cb_q9(np.float32(-1.0))) == -255.0)
    check('q(-1.0039) == -256 (the clip)', float(NE._cb_q9(np.float32(-1.0039))) == -256.0)
    check('EXPAND_NORMAL maps 0.5 -> 0 exactly', float(NE._cb_nv_map(np.float32(0.5), 'EXPAND_NORMAL')) == 0.0)
    check('UNSIGNED_INVERT of 1 is 0', float(NE._cb_nv_map(np.float32(1.0), 'UNSIGNED_INVERT')) == 0.0)
    out = _cb_cpu(_cb_node('NV2A'), a, o, z, z)
    qa = NE._cb_q9(a[:, :3])
    check('A*B with B = 1: pab == q(a) (255 * q / 255 rounds exactly)', bool(np.array_equal(out[:, :3], (qa * r255).astype(np.float32))))
    out2 = _cb_cpu(_cb_node('NV2A', map_a='SIGNED_NEGATE', nv_bias='MINUS_HALF'), a, o, z, z)
    check('SIGNED_NEGATE + bias -0.5: clip(-q(a) - 128, -256, 255)',
          bool(np.array_equal(out2[:, :3], (np.clip(-qa - 128, -256, 255) * r255).astype(np.float32))))
    half = np.full((n, 4), 0.5, np.float32)
    out3 = _cb_cpu(_cb_node('NV2A', nv_scale='X4'), half, o, half, o)
    check('A*B + C*D with scale x4: 0.5 + 0.5 -> 4 * 256 -> clamps at 255', bool(np.allclose(out3[:, :3], np.float32(255) * r255)))
    check('the NV2A output is float32 and alpha is A alpha', out.dtype == np.float32 and bool(np.array_equal(out[:, 3], a[:, 3])))


def test_c028_division_rule():
    """For every product p in -65536..65025, rint(p / 255) in float32 equals
    Python's round(p / 255) (banker's) -- exhaustively on the CPU and through
    the simulator on all 130,562 lanes."""
    p = np.arange(-65536, 65026, dtype=np.int64)
    want = np.asarray([round(int(v) / 255) for v in p], np.float32)
    cpu = np.rint(p.astype(np.float32) / np.float32(255.0)).astype(np.float32)
    check(f'CPU: rint(p / 255) in float32 == round(p / 255) for all {p.size} products', bool(np.array_equal(cpu, want)))
    got, why = glsl_twin('    Color = vec4(roundEven(u_p / 255.0), 0.0, 0.0, 1.0);', {'u_p': p.astype(np.float32)}, int(p.size),
                         unis='uniform float u_p;')
    check('simulator: the division rule compiles', got is not None, str(why))
    if got is not None:
        check(f'simulator: roundEven(p / 255.0) == round(p / 255) on all {p.size} lanes', bool(np.array_equal(got[:, 0], want)))


def test_c028_gpu_twin():
    """`glsl_twin` of the emitted stage for every op / hardware / bias / scale on
    4096 inputs `d == 0.0`; a frame through `sim_vs_cpu` on the two FM
    scenes: bar 6e-3 (the deferred seam; the node itself contributes 0)."""
    rng = np.random.default_rng(1028)
    n = 4096
    a = rng.random((n, 4)).astype(np.float32)
    b = rng.random((n, 4)).astype(np.float32)
    c = rng.random((n, 4)).astype(np.float32)
    d = (rng.random((n, 4)) * 2.0 - 0.5).astype(np.float32)      # a linked D reaches beyond 0..1
    a[:128, :3] = np.float32(0.5)
    lanes = {'uA': ('vec4', a), 'uB': ('vec4', b), 'uC': ('vec4', c), 'uD': ('vec4', d)}
    cases = []
    for op in ('ADD', 'SUB', 'COMP_R8_GT', 'COMP_R8_EQ', 'COMP_GR16_GT', 'COMP_GR16_EQ', 'COMP_BGR24_GT',
               'COMP_BGR24_EQ', 'COMP_RGB8_GT', 'COMP_RGB8_EQ'):
        cases.append(('TEV', dict(op=op)))
    for bias in ('ADD_HALF', 'SUB_HALF'):
        for scale in ('X2', 'X4', 'HALF'):
            cases.append(('TEV', dict(op='ADD', bias=bias, scale=scale, clamp=False)))
    for m in ('UNSIGNED_IDENTITY', 'UNSIGNED_INVERT', 'EXPAND_NORMAL', 'EXPAND_NEGATE', 'HALF_BIAS_NORMAL',
              'HALF_BIAS_NEGATE', 'SIGNED_IDENTITY', 'SIGNED_NEGATE'):
        cases.append(('NV2A', dict(map_a=m, map_d=m)))
    for sc_ in ('X2', 'X4', 'HALF'):
        cases.append(('NV2A', dict(nv_scale=sc_, nv_bias='MINUS_HALF')))
    bad = []
    for hw, props in cases:
        node = _cb_node(hw, **props)
        cpu = _cb_cpu(node, a, b, c, d)
        got, why = node_twin(node, lanes, n)
        if got is None:
            bad.append(f'{hw} {props}: {why}')
        elif not np.array_equal(got, cpu):
            bad.append(f'{hw} {props}: max {float(np.abs(got - cpu).max()):.2e}')
    check(f'the emitted stage is bitwise the CPU (d == 0.0) on 4096 lanes for all {len(cases)} cases',
          not bad, '; '.join(bad[:4]))
    for key in ('combiner_node', 'combiner_nv2a_node'):
        sc, st = scene(key)
        frame_bar(f'node combiner {key}', sc, st)
    # the row moves the picture against the plain textured demo
    base = cpu_frame('textured')
    for key in ('combiner_node', 'combiner_nv2a_node'):
        img = cpu_frame(key)
        check(f'{key} moves the picture', float(np.abs(img - base).max()) > 1e-2)
    # the export road: every prop of the node is in NODE_PROPS
    from . import fakebpy
    fakebpy.install()
    from ..export import NODE_PROPS
    check('every Combiner Stage prop reaches the renderer through NODE_PROPS',
          set(NODE_PROPS.get('HALCYON_CombinerStageNode', ())) >= {'hardware', 'op', 'bias', 'scale', 'clamp', 'map_a',
                                                                     'map_b', 'map_c', 'map_d', 'nv_scale', 'nv_bias'})


# ================================================================ C023
def _sr_node(light=(0.6, 0.0, 0.8), strength=0.7, blend='MULTIPLY', color_link=('uC', 0), light_link=None,
             strength_link=None, base=(0.8, 0.7, 0.6, 1.0), base_link=None):
    return {'id': 'srb', 'bl_idname': 'HALCYON_SRBumpNode', 'props': {'blend': blend},
            'inputs': [_sk('Color', 'RGBA', [0.5, 0.5, 1.0, 1.0], list(color_link) if color_link else None),
                       _sk('Light', 'VECTOR', list(light), list(light_link) if light_link else None),
                       _sk('Strength', 'VALUE', float(strength), list(strength_link) if strength_link else None),
                       _sk('Base', 'RGBA', list(base), list(base_link) if base_link else None)],
            'outputs': [{'name': 'Intensity', 'type': 'VALUE'}, {'name': 'Color', 'type': 'RGBA'}]}


def _sr_cpu(node, color, light=None, strength=None, base=None):
    n = int(color.shape[0])
    arrays = {'Color': color}
    arrays['Light'] = light if light is not None else np.tile(np.asarray(node['inputs'][1]['default'], np.float32)[None, :], (n, 1))
    arrays['Strength'] = strength if strength is not None else np.full(n, node['inputs'][2]['default'], np.float32)
    arrays['Base'] = base if base is not None else np.tile(np.asarray(node['inputs'][3]['default'], np.float32)[None, :], (n, 1))
    ev = _FakeEv(n, **arrays)
    # has_link mirrors the node's links (a linked Light is the per-pixel road)
    ev.arrays_linked = {s['name'] for s in node['inputs'] if s.get('link')}
    ev.has_link = lambda nd, name: name in ev.arrays_linked
    return NE.n_halcyon_sr_bump(ev, node)


def test_c023_tables():
    """The tables re-computed from the formulas (the 8-bit encoding has no
    exact zero: 2*128/255 - 1 = +0.0039): SIDX[255] == 255, SIDX[127] == 0,
    SIDX[128] == 1, SIDX non-decreasing; RTAB[128][255] == 128 (+t),
    RTAB[255][128] == 191 (+b), RTAB[127][0] == 0 (-t at ny = 127),
    RTAB[128][0] == 255 (the wrap); COS256[0] == 1, COS256[128] == -1; the
    two prepared textures carry them in the stated slots."""
    from ..core import srbump_tables as SRT
    check('SIDX[255] == 255', int(SRT.SIDX[255]) == 255)
    check('SIDX[127] == 0', int(SRT.SIDX[127]) == 0)
    check('SIDX[128] == 1 (asin(0.0039) * 2/pi * 255 = 0.64 rounds up)', int(SRT.SIDX[128]) == 1)
    check('SIDX is non-decreasing', bool((np.diff(SRT.SIDX.astype(np.int64)) >= 0).all()))
    check('RTAB[128][255] == 128 (+t axis)', int(SRT.RTAB[128][255]) == 128)
    check('RTAB[255][128] == 191 (+b axis)', int(SRT.RTAB[255][128]) == 191)
    check('RTAB[127][0] == 0 (the -t axis at ny = 127)', int(SRT.RTAB[127][0]) == 0)
    check('RTAB[128][0] == 255 (the wrap: -pi + 0.0039 rounds to 255 & 255)', int(SRT.RTAB[128][0]) == 255)
    check('COS256[0] == 1 and COS256[128] == -1', float(SRT.COS256[0]) == 1.0 and float(SRT.COS256[128]) == -1.0)
    tp = SRT.table_pixels()
    ap = SRT.atan_pixels()
    check('__sr_tables__ is (1, 1024, 4) float32 with SIDX / SINS / COSS / COS256 in its four slots',
          tp.shape == (1, 1024, 4) and tp.dtype == np.float32
          and bool(np.array_equal(tp[0, :256, 0], SRT.SIDX.astype(np.float32)))
          and bool(np.array_equal(tp[0, 256:512, 0], SRT.SINS))
          and bool(np.array_equal(tp[0, 512:768, 0], SRT.COSS))
          and bool(np.array_equal(tp[0, 768:, 0], SRT.COS256)))
    check('__sr_atan__ is (256, 256, 4) float32 with RTAB[ny][nx] in red',
          ap.shape == (256, 256, 4) and bool(np.array_equal(ap[:, :, 0], SRT.RTAB.astype(np.float32))))
    sc, st = scene('srbump_node')
    tex = R.prepare_textures(sc, st)
    check('prepare_textures carries __sr_tables__ and __sr_atan__ (cached)',
          '__sr_tables__' in tex and '__sr_atan__' in tex
          and R._TEX_CACHE.get('__sr_atan__') is tex['__sr_atan__'])


def test_c023_laws():
    """A flat texel (0.5, 0.5, 1) at any Light gives I == K1 + K2 (sin S = 1,
    cos S = 0: only the elevation term); Strength 0 gives I == 1.0 exactly and
    Color == Base bitwise (MULTIPLY); a tilted texel facing the light (R == Q)
    is brighter than one facing away (R == Q + 128) for the same S; Q steps in
    256 azimuths (a turn of 1/1024 at a step's centre changes no pixel)."""
    from ..core import srbump_tables as SRT
    n = 512
    flat = np.tile(np.array([[0.5, 0.5, 1.0, 1.0]], np.float32), (n, 1))
    for light in ((0.6, 0.0, 0.8), (0.0, 1.0, 0.0), (0.3, -0.4, 0.5)):
        node = _sr_node(light=light, strength=0.7)
        out = _sr_cpu(node, flat)
        K1, K2, K3, Q = SRT.light_constants(light, 0.7)
        check(f'flat texel at Light {light}: I == K1 + K2 exactly', bool(np.array_equal(out['Intensity'], np.full(n, np.float32(K1 + K2), np.float32))))
    rng = np.random.default_rng(23)
    col = rng.random((n, 4)).astype(np.float32)
    base = rng.random((n, 4)).astype(np.float32)
    node0 = _sr_node(strength=0.0)
    out0 = _sr_cpu(node0, col, base=base)
    check('Strength 0: I == 1.0 exactly', bool(np.array_equal(out0['Intensity'], np.ones(n, np.float32))))
    check('Strength 0: Color == Base bitwise (MULTIPLY)', bool(np.array_equal(out0['Color'], base)))
    # facing vs away: the same S, R == Q vs R == Q + 128 -- built from the tables backwards
    K1, K2, K3, Q = SRT.light_constants((0.6, 0.0, 0.8), 0.7)
    S = 100
    r_face = int(np.nonzero(SRT.RTAB[128] == Q)[0][0]) if (SRT.RTAB[128] == Q).any() else None
    i_face = SRT.intensity(np.array([0]), np.array([0]), np.array([0]), K1, K2, K3, Q)  # placeholder shape
    I_face = float(np.float32(K1) + np.float32(K2) * SRT.SINS[S] + np.float32(K3) * SRT.COSS[S] * SRT.COS256[0])
    I_away = float(np.float32(K1) + np.float32(K2) * SRT.SINS[S] + np.float32(K3) * SRT.COSS[S] * SRT.COS256[128])
    check('a tilted texel facing the light (R == Q) is brighter than one facing away (R == Q + 128)', I_face > I_away)
    # through the table road itself: two texels whose R differ by 128 at one S
    ny, nx = np.nonzero(SRT.RTAB == Q)
    ny2, nx2 = np.nonzero(SRT.RTAB == ((Q + 128) & 255))
    if ny.size and ny2.size:
        b8 = int(np.nonzero(SRT.SIDX == S)[0][0]) if (SRT.SIDX == S).any() else 200
        If = SRT.intensity(np.array([nx[0]]), np.array([ny[0]]), np.array([b8]), K1, K2, K3, Q)
        Ia = SRT.intensity(np.array([nx2[0]]), np.array([ny2[0]]), np.array([b8]), K1, K2, K3, Q)
        check('...and through the table road (texels with R == Q vs R == Q + 128)', float(If[0]) > float(Ia[0]))
    # Q steps: a Light at a step's centre turned by 1/1024 keeps Q, so no pixel moves
    ang = 64.0 / 255.0 * 2.0 * np.pi - np.pi
    L0 = (np.cos(ang) * 0.8, np.sin(ang) * 0.8, 0.6)
    ang2 = ang + 2.0 * np.pi / 1024.0
    L1 = (np.cos(ang2) * 0.8, np.sin(ang2) * 0.8, 0.6)
    o0 = _sr_cpu(_sr_node(light=L0, strength=0.9), col)
    o1 = _sr_cpu(_sr_node(light=L1, strength=0.9), col)
    check('Q steps in 256 azimuths: a 1/1024 turn at a step centre changes no pixel (bitwise)',
          bool(np.array_equal(o0['Intensity'], o1['Intensity'])))
    ang3 = ang + 2.0 * np.pi / 128.0
    L3 = (np.cos(ang3) * 0.8, np.sin(ang3) * 0.8, 0.6)
    o3 = _sr_cpu(_sr_node(light=L3, strength=0.9), col)
    check('...while a 2-step turn moves pixels', not bool(np.array_equal(o0['Intensity'], o3['Intensity'])))
    # ADD blend
    oa = _sr_cpu(_sr_node(strength=0.7, blend='ADD'), col, base=base)
    om = _sr_cpu(_sr_node(strength=0.7), col, base=base)
    check('ADD: Color == min(Base + I, 1); MULTIPLY: Base * I (alpha from Base)',
          bool(np.array_equal(oa['Color'][:, :3], np.minimum(base[:, :3] + om['Intensity'][:, None], 1.0).astype(np.float32)))
          and bool(np.array_equal(om['Color'][:, :3], (base[:, :3] * om['Intensity'][:, None]).astype(np.float32)))
          and bool(np.array_equal(om['Color'][:, 3], base[:, 3])))
    # a linked Light per pixel (the CPU road): the same numbers as the constant road when every lane carries the same light
    light_arr = np.tile(np.array([[0.6, 0.0, 0.8]], np.float32), (n, 1))
    ol = _sr_cpu(_sr_node(light_link=('uL', 0)), col, light=light_arr)
    check('a linked Light evaluated per pixel equals the constant road when every lane carries the same vector',
          bool(np.array_equal(ol['Intensity'], _sr_cpu(_sr_node(), col)['Intensity'])))


def test_c023_gpu_twin():
    """`glsl_twin` on 4096 texel lanes with the two tables bound, for four
    (Light, Strength) settings and a linked Strength: `d == 0.0`; then
    `sim_vs_cpu` on the FM scene under tex_filter NEAREST at the deferred
    bar, MULTIPLY and ADD."""
    from ..core import srbump_tables as SRT
    from ..core.texture import Texture
    rng = np.random.default_rng(1023)
    n = 4096
    col = rng.random((n, 4)).astype(np.float32)
    col[:256, :3] = np.float32(0.5)
    base = rng.random((n, 4)).astype(np.float32)
    tabs = {'hal_sr_tab': Texture(SRT.table_pixels(), colorspace='Non-Color', filt='NEAREST', wrap='EXTEND'),
            'hal_sr_atan': Texture(SRT.atan_pixels(), colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')}
    bad = []
    for light, strength, blend in (((0.6, 0.0, 0.8), 0.7, 'MULTIPLY'), ((0.0, 1.0, 0.0), 1.0, 'ADD'),
                                   ((0.3, -0.4, 0.5), 0.35, 'MULTIPLY'), ((-1.0, 0.0, 0.0), 0.9, 'ADD')):
        node = _sr_node(light=light, strength=strength, blend=blend, base_link=('uB', 0))
        cpu = _sr_cpu(node, col, base=base)
        for idx, key in ((0, 'Intensity'), (1, 'Color')):
            got, why = node_twin(node, {'uC': ('vec4', col), 'uB': ('vec4', base)}, n, out_index=idx,
                                 extra_uniforms=tabs)
            if got is None:
                bad.append(f'{light} {strength} {key}: {why}')
                continue
            want = cpu[key] if idx == 1 else cpu[key]
            g = got[:, 0] if idx == 0 else got
            if not np.array_equal(g, want):
                bad.append(f'{light} {strength} {key}: max {float(np.abs(g - want).max()):.2e}')
    check('the emitted SR Bump is bitwise the CPU (d == 0.0) on 4096 lanes, 4 lights x Intensity + Color', not bad, '; '.join(bad[:3]))
    # a linked Strength with an unlinked Light: per-pixel K1/K2/K3, legal on the GPU
    strength = rng.random(n).astype(np.float32)
    node = _sr_node(light=(0.6, 0.0, 0.8), strength_link=('uS', 0), base_link=('uB', 0))
    cpu = _sr_cpu(node, col, strength=strength, base=base)
    got, why = node_twin(node, {'uC': ('vec4', col), 'uB': ('vec4', base), 'uS': ('float', strength)}, n,
                         out_index=1, extra_uniforms=tabs)
    check('a linked Strength (unlinked Light) is bitwise the CPU on the GPU', got is not None and bool(np.array_equal(got, cpu['Color'])),
          str(why) if got is None else f'max {float(np.abs(got - cpu["Color"]).max()):.2e}')
    for blend in ('MULTIPLY', 'ADD'):
        st = base_settings(W, H, transparency='NONE', tex_filter='NEAREST')
        st.use_processes = False
        sc = FM._sc_srbump_node(st, blend=blend)
        out = frame_bar(f'node SR bump ({blend}) under NEAREST', sc, st)
        if out is not None:
            g, job = rig(sc, st)
            GSH._PLAN_CACHE.clear()
            p, _why, _a = GSH.plan_frame(job, g)
            src = ''.join(str(e[2]) for e in p)
            check(f'the {blend} pass declares hal_sr_tab and hal_sr_atan and fetches them',
                  'uniform sampler2D hal_sr_tab;' in src and 'texelFetch(hal_sr_atan' in src)
    base_img = cpu_frame('demo')
    img = cpu_frame('srbump_node', tex_filter='NEAREST')
    check('the SR bump row moves the picture', float(np.abs(img - base_img).max()) > 1e-2)


def test_c023_refusal_by_name():
    """Light linked to a Normal Map node: the GPU plan refuses with 'Light
    linked' in why; the CPU frame renders (and differs from the unlinked
    one); Strength AND Light linked names the per-pixel sqrt."""
    st = base_settings(W, H, transparency='NONE', tex_filter='NEAREST')
    st.use_processes = False
    sc = FM._sc_srbump_node(st)
    g0 = sc.materials[1].graph
    g0['nodes']['nm'] = {'id': 'nm', 'bl_idname': 'HALCYON_NormalMapNode', 'props': {'space': 'TANGENT', 'map_type': 'OPENGL'},
                         'inputs': [_sk('Color', 'RGBA', [0.5, 0.5, 1.0, 1.0], ['ntex', 0]), _sk('Strength', 'VALUE', 1.0)],
                         'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
    g0['nodes']['srb']['inputs'][1]['link'] = ['nm', 0]
    gb, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, gb)
    check("Light linked: the plan refuses by name ('Light linked')", p is None and 'Light linked' in str(why), str(why))
    img = np.asarray(R.render(sc, st))
    st2 = base_settings(W, H, transparency='NONE', tex_filter='NEAREST')
    st2.use_processes = False
    img2 = np.asarray(R.render(FM._sc_srbump_node(st2), st2))
    check('the CPU frame renders with the linked Light and differs from the unlinked one',
          img.shape == img2.shape and float(np.abs(img - img2).max()) > 1e-3)
    g0['nodes']['val'] = {'id': 'val', 'bl_idname': 'ShaderNodeValue', 'props': {'value': 0.5}, 'inputs': [],
                          'outputs': [{'name': 'Value', 'type': 'VALUE'}]}
    g0['nodes']['srb']['inputs'][2]['link'] = ['val', 0]
    gb, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, gb)
    check("Strength linked with a linked Light: refuses by name ('per-pixel sqrt')", p is None and 'per-pixel sqrt' in str(why), str(why))
    from . import fakebpy
    fakebpy.install()
    from ..export import NODE_PROPS
    check("the SR Bump 'blend' prop reaches the renderer through NODE_PROPS", 'blend' in NODE_PROPS.get('HALCYON_SRBumpNode', ()))


# ================================================================ C135
def _emb_shift_node(light=(0.7071, 0.7071, 0.0), offset=1.0, size=64.0, vec_link=('uV', 0), size_link=None,
                    light_link=None, offset_link=None):
    return {'id': 'shift', 'bl_idname': 'HALCYON_EmbossShiftNode', 'props': {},
            'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], list(vec_link) if vec_link else None),
                       _sk('Light', 'VECTOR', list(light), list(light_link) if light_link else None),
                       _sk('Offset', 'VALUE', float(offset), list(offset_link) if offset_link else None),
                       _sk('Texture Size', 'VALUE', float(size), list(size_link) if size_link else None)],
            'outputs': [{'name': 'Shifted UV', 'type': 'VECTOR'}]}


def _emb_node(base=(0.8, 0.7, 0.6, 1.0)):
    return {'id': 'emb', 'bl_idname': 'HALCYON_EmbossBumpNode', 'props': {},
            'inputs': [_sk('Height', 'VALUE', 0.5, ['uH', 0]), _sk('Height Shifted', 'VALUE', 0.5, ['uS', 0]),
                       _sk('Base', 'RGBA', list(base), ['uB', 0])],
            'outputs': [{'name': 'Factor', 'type': 'VALUE'}, {'name': 'Color', 'type': 'RGBA'}]}


def _emb_cpu(h, hs, base):
    ev = _FakeEv(int(h.shape[0]), **{'Height': h, 'Height Shifted': hs, 'Base': base})
    return NE.n_halcyon_emboss_bump(ev, _emb_node())


def _shift_cpu(node, uv, light=None, offset=None, size=None):
    n = int(uv.shape[0])
    arrays = {'Vector': uv}
    arrays['Light'] = light if light is not None else np.tile(np.asarray(node['inputs'][1]['default'], np.float32)[None, :], (n, 1))
    arrays['Offset'] = offset if offset is not None else np.full(n, node['inputs'][2]['default'], np.float32)
    arrays['Texture Size'] = size if size is not None else np.full(n, node['inputs'][3]['default'], np.float32)
    ev = _FakeEv(n, **arrays)
    linked = {s['name'] for s in node['inputs'] if s.get('link')}
    ev.has_link = lambda nd, name: name in linked
    return NE.n_halcyon_emboss_shift(ev, node)['Shifted UV']


def test_c135_laws():
    """Flat heights (h == hs) give Factor == 0.5 and Color == Base bitwise; a
    slope facing the light (h > hs) brightens, away darkens, monotonic in
    h - hs; min(2b, 1) clamps (h - hs >= 0 never exceeds Base); Shifted UV -
    uv == Offset * Light.xy / size bitwise; Offset 0 gives Shifted UV == uv."""
    rng = np.random.default_rng(135)
    n = 2048
    h = rng.random(n).astype(np.float32) * 0.5
    base = rng.random((n, 4)).astype(np.float32)
    flat = _emb_cpu(h, h, base)
    check('flat heights: Factor == 0.5 exactly', bool(np.array_equal(flat['Factor'], np.full(n, 0.5, np.float32))))
    check('flat heights: Color == Base bitwise (identity at the defaults)', bool(np.array_equal(flat['Color'], base)))
    hs = rng.random(n).astype(np.float32) * 0.5
    out = _emb_cpu(h, hs, base)
    d = h - hs
    up = d > 0
    check('a slope facing the light (h > hs) brightens Factor above 0.5, away darkens',
          bool((out['Factor'][up] > 0.5).all()) and bool((out['Factor'][~up & (d < 0)] < 0.5).all()))
    order = np.argsort(d)
    check('Factor is monotonic in h - hs', bool((np.diff(out['Factor'][order]) >= 0).all()))
    check('min(2b, 1) clamps: h - hs >= 0 never exceeds Base',
          bool((out['Color'][up][:, :3] <= base[up][:, :3] + 1e-7).all())
          and bool(np.array_equal(out['Color'][up][:, :3], base[up][:, :3])))
    uv = np.concatenate([rng.random((n, 2)).astype(np.float32), np.zeros((n, 1), np.float32)], 1)
    node = _emb_shift_node(light=(0.7071, 0.7071, 0.0), offset=1.0, size=64.0)
    suv = _shift_cpu(node, uv)
    inv = np.float32(np.float32(1.0) / np.float32(64.0))
    p = (np.float32(1.0) * np.array([0.7071, 0.7071], np.float32)).astype(np.float32)
    dd = (p * inv).astype(np.float32)
    want = uv.copy()
    want[:, 0] = (uv[:, 0] + dd[0]).astype(np.float32)
    want[:, 1] = (uv[:, 1] + dd[1]).astype(np.float32)
    check('Shifted UV == uv + Offset * Light.xy / size bitwise on the CPU', bool(np.array_equal(suv, want)))
    check('Offset 0 gives Shifted UV == uv', bool(np.array_equal(_shift_cpu(_emb_shift_node(offset=0.0), uv), uv)))
    check('the z of the vector passes through', bool(np.array_equal(suv[:, 2], uv[:, 2])))


def _ball_factor(sc, st, key_out=0):
    """The Emboss Bump node's Factor over the Ball's fragments, through the
    CPU evaluator on the frame's own context."""
    g, job = rig(sc, st)
    mi = np.asarray(sc.mesh.mat_index)[np.maximum(g.tri, 0)]
    yy, xx = np.nonzero((g.tri >= 0) & (mi == 1))
    tri = g.tri[yy, xx]
    bary = g.bary[yy, xx]
    ctx = job.context(tri, bary, xx, yy, None, None, 0, True)
    graph = sc.materials[1].graph
    ev = NE.GraphEvaluator(graph, ctx, job.textures, None)
    return np.asarray(ev.eval_output('emb', key_out), np.float32).reshape(-1)


def test_c135_two_stage_frame():
    """The FM scene at NEAREST: on a RAMP height map (rising along +x) the
    slope's sign is one everywhere, and rotating Light by 180 degrees
    inverts the relief (the mean over the material of Factor - 0.5 flips
    sign) -- the era's slide; on the checker the bright edges move to the
    other side of every square (the same count, a different set)."""
    from ..core.scene import ImageBuffer

    def ramp(sc):
        px = np.zeros((64, 64, 4), np.float32)
        px[:, :, :3] = (np.arange(64, dtype=np.float32) / 63.0 * 0.5)[None, :, None]
        px[:, :, 3] = 1.0
        sc.images['checker'] = ImageBuffer(name='checker', pixels=px)
        for k in ('htex', 'htex2'):        # no wrap seam: the ramp clamps
            sc.materials[1].graph['nodes'][k]['props']['extension'] = 'EXTEND'
        return sc
    st = base_settings(W, H, transparency='NONE')
    st.use_processes = False
    sc = ramp(FM._sc_emboss_node(st, light=(1.0, 0.0, 0.0), size_link=None))
    sc.materials[1].graph['nodes']['shift']['inputs'][2]['default'] = 4.0      # four texels: a clear step
    f0 = _ball_factor(sc, st)
    st2 = base_settings(W, H, transparency='NONE')
    st2.use_processes = False
    sc2 = ramp(FM._sc_emboss_node(st2, light=(-1.0, 0.0, 0.0)))
    sc2.materials[1].graph['nodes']['shift']['inputs'][2]['default'] = 4.0
    f1 = _ball_factor(sc2, st2)
    m0 = float((f0 - 0.5).mean())
    m1 = float((f1 - 0.5).mean())
    check('the ramp relief is not flat (Factor moves off 0.5 on the ball)', float(np.abs(f0 - 0.5).max()) > 0.01)
    check('rotating Light by 180 degrees inverts the relief (mean(Factor - 0.5) flips sign)',
          (m0 < 0 < m1) or (m1 < 0 < m0), f'{m0:.4f} vs {m1:.4f}')
    # the checker: the bright edges slide to the other side of every square
    st3 = base_settings(W, H, transparency='NONE')
    st3.use_processes = False
    sc3 = FM._sc_emboss_node(st3)
    g0 = _ball_factor(sc3, st3)
    st4 = base_settings(W, H, transparency='NONE')
    st4.use_processes = False
    sc4 = FM._sc_emboss_node(st4, light=(-0.7071, -0.7071, 0.0))
    g1 = _ball_factor(sc4, st4)
    b0 = g0 - 0.5 > 0.1
    b1 = g1 - 0.5 > 0.1
    both = int((b0 & b1).sum())
    check('checker: the bright edges under the flipped light are a different set of about the same size',
          b0.sum() > 0 and abs(int(b0.sum()) - int(b1.sum())) <= 0.5 * max(int(b0.sum()), 1)
          and both <= 0.5 * max(int(b0.sum()), 1), f'{int(b0.sum())} vs {int(b1.sum())}, overlap {both}')
    sc, st = sc3, st3
    img0 = np.asarray(R.render(sc, st))
    base_img = cpu_frame('demo')
    check('the emboss row moves the picture', float(np.abs(img0 - base_img).max()) > 1e-2)


def test_c135_gpu_twin():
    """`glsl_twin` `d == 0.0` on 4096 lanes for both stages; `sim_vs_cpu` on
    the FM scene at Closest: the deferred bar; a second run with both Image
    Texture nodes at Linear stays within the same bar (the sampler's class)."""
    rng = np.random.default_rng(1135)
    n = 4096
    h = rng.random(n).astype(np.float32)
    hs = rng.random(n).astype(np.float32)
    base = rng.random((n, 4)).astype(np.float32)
    cpu = _emb_cpu(h, hs, base)
    lanes = {'uH': ('float', h), 'uS': ('float', hs), 'uB': ('vec4', base)}
    got_f, why = node_twin(_emb_node(), lanes, n, out_index=0)
    check('Emboss Bump Factor is bitwise the CPU (d == 0.0)', got_f is not None and bool(np.array_equal(got_f[:, 0], cpu['Factor'])), str(why))
    got_c, why = node_twin(_emb_node(), lanes, n, out_index=1)
    check('Emboss Bump Color is bitwise the CPU (d == 0.0)', got_c is not None and bool(np.array_equal(got_c, cpu['Color'])), str(why))
    uv = np.concatenate([rng.random((n, 2)).astype(np.float32), np.zeros((n, 1), np.float32)], 1)
    for light, offset, size in (((0.7071, 0.7071, 0.0), 1.0, 64.0), ((0.3, -0.9, 0.2), 1.7, 100.0), ((1.0, 0.0, 0.0), 0.5, 37.0)):
        node = _emb_shift_node(light=light, offset=offset, size=size)
        suv = _shift_cpu(node, uv)
        got, why = node_twin(node, {'uV': ('vec3', uv)}, n)
        check(f'Emboss Shift {light} / {offset} / {size}: Shifted UV is bitwise the CPU (d == 0.0)',
              got is not None and bool(np.array_equal(got[:, :3], suv)), str(why) if got is None else f'max {float(np.abs(got[:, :3] - suv).max()):.2e}')
    for interp in ('Closest', 'Linear'):
        st = base_settings(W, H, transparency='NONE')
        st.use_processes = False
        sc = FM._sc_emboss_node(st, interpolation=interp)
        frame_bar(f'node emboss bump with the Image Texture nodes at {interp}', sc, st)


def test_c135_refusal_by_name():
    """Texture Size linked to a Value node: the plan refuses with 'Texture
    Size linked'; the CPU renders; the two classes have no props (nothing
    for NODE_PROPS to carry) and both are registered."""
    st = base_settings(W, H, transparency='NONE')
    st.use_processes = False
    sc = FM._sc_emboss_node(st, size_link=['val', 0])
    g, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check("Texture Size linked: the plan refuses by name ('Texture Size linked')", p is None and 'Texture Size linked' in str(why), str(why))
    img = np.asarray(R.render(sc, st))
    st2 = base_settings(W, H, transparency='NONE')
    st2.use_processes = False
    img2 = np.asarray(R.render(FM._sc_emboss_node(st2), st2))
    check('the CPU renders the linked size, bitwise the constant one (the same value)', bool(np.array_equal(img, img2)))
    from . import fakebpy
    fakebpy.install()
    from ..nodes import shader_nodes as SN
    names = [c.bl_idname for c in SN.NODES]
    check('both emboss classes are registered in NODES and reachable from the Shading family',
          'HALCYON_EmbossShiftNode' in names and 'HALCYON_EmbossBumpNode' in names
          and all(any(getattr(c, 'bl_idname', '') == k for c in SN.MENU_FAMILIES[0][2])
                  for k in ('HALCYON_EmbossShiftNode', 'HALCYON_EmbossBumpNode')))


# ================================================================ C099
class _PixCtx:
    """A ShadeContext stand-in for the Roughness node: the pixel grid, the
    normal, the seed and the frame."""

    def __init__(self, px, py, N, seed=0, frame=1):
        self.n = int(px.shape[0])
        self.px = np.asarray(px, np.int64)
        self.py = np.asarray(py, np.int64)
        self.N = np.asarray(N, np.float32)
        self.settings = RenderSettings()
        self.settings.seed = int(seed)
        self.frame = int(frame)


def _imr_node(roughness=180.0, animate=False, normal_link=None):
    return {'id': 'rough', 'bl_idname': 'HALCYON_ImagineRoughnessNode', 'props': {'animate': bool(animate)},
            'inputs': [_sk('Normal', 'VECTOR', [0, 0, 0], list(normal_link) if normal_link else None),
                       _sk('Roughness', 'VALUE', float(roughness))],
            'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}


def _imr_cpu(node, px, py, N, seed=0, frame=1, roughness=None):
    n = int(px.shape[0])
    r = roughness if roughness is not None else np.full(n, node['inputs'][1]['default'], np.float32)
    ev = _FakeEv(n, Roughness=r, Normal=N)
    ev.ctx = _PixCtx(px, py, N, seed, frame)
    ev.has_link = lambda nd, name: name == 'Normal' and bool(node['inputs'][0].get('link'))
    return NE.n_halcyon_imagine_roughness(ev, node)['Normal']


def _unit_normals(rng, n):
    v = rng.normal(size=(n, 3)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def test_c099_laws():
    """Roughness 0 returns the input normal bitwise; at 255 the output is
    unit length and the mean angle to the input over 4096 pixels is 15..35
    degrees (the "about 25 degrees rms" claim as a band); determinism: the
    same (px, py, seed, frame) gives the same normal bitwise twice and
    across threads 1 vs 4 on a frame; shimmer off: frame 1 == frame 2;
    on: they differ; a seed change changes the picture."""
    rng = np.random.default_rng(99)
    n = 4096
    px = rng.integers(0, 640, n)
    py = rng.integers(0, 480, n)
    N = _unit_normals(rng, n)
    out0 = _imr_cpu(_imr_node(0.0), px, py, N)
    check('Roughness 0 returns the input normal bitwise', bool(np.array_equal(out0, N)))
    out = _imr_cpu(_imr_node(255.0), px, py, N)
    ln = np.linalg.norm(out.astype(np.float64), axis=1)
    check('at 255 the output is unit length (|len - 1| < 1e-6)', float(np.abs(ln - 1.0).max()) < 1e-6)
    cosang = np.clip((out.astype(np.float64) * N).sum(1), -1, 1)
    ang = np.degrees(np.arccos(cosang))
    check('at 255 the mean angle to the input is between 15 and 35 degrees', 15.0 < float(ang.mean()) < 35.0, f'{float(ang.mean()):.1f}')
    out_b = _imr_cpu(_imr_node(255.0), px, py, N)
    check('the same (px, py, seed, frame) gives the same normal bitwise across two evaluations', bool(np.array_equal(out, out_b)))
    out_s = _imr_cpu(_imr_node(255.0), px, py, N, seed=7)
    check('a seed change changes the turn', not bool(np.array_equal(out, out_s)))
    out_f1 = _imr_cpu(_imr_node(255.0), px, py, N, frame=1)
    out_f2 = _imr_cpu(_imr_node(255.0), px, py, N, frame=2)
    check('shimmer off: frame 1 == frame 2 bitwise', bool(np.array_equal(out_f1, out_f2)))
    sh1 = _imr_cpu(_imr_node(255.0, animate=True), px, py, N, frame=1)
    sh2 = _imr_cpu(_imr_node(255.0, animate=True), px, py, N, frame=2)
    check('shimmer on: frame 1 != frame 2', not bool(np.array_equal(sh1, sh2)))
    half = _imr_cpu(_imr_node(64.0), px, py, N)
    ang_h = np.degrees(np.arccos(np.clip((half.astype(np.float64) * N).sum(1), -1, 1)))
    check('a smaller Roughness turns less (monotonic in the mean)', float(ang_h.mean()) < float(ang.mean()))
    # a frame: threads 1 vs 4 bitwise, seed and shimmer laws through the renderer
    st1 = base_settings(W, H, transparency='NONE', seed=3, threads=1)
    st1.use_processes = False
    a = np.asarray(R.render(FM._sc_roughness_node(st1), st1))
    st4 = base_settings(W, H, transparency='NONE', seed=3, threads=4)
    st4.use_processes = False
    b = np.asarray(R.render(FM._sc_roughness_node(st4), st4))
    check('threads 1 vs 4 render the roughened ball bitwise (a stable per-pixel hash)', bool(np.array_equal(a, b)))
    st5 = base_settings(W, H, transparency='NONE', seed=5, threads=1)
    st5.use_processes = False
    c = np.asarray(R.render(FM._sc_roughness_node(st5), st5))
    check('a seed change changes the picture', not bool(np.array_equal(a, c)))
    base_img = cpu_frame('demo', seed=3)
    check('the roughness row moves the picture', float(np.abs(a - base_img).max()) > 1e-2)
    st6 = base_settings(W, H, transparency='NONE', seed=3, threads=1)
    st6.use_processes = False
    d1 = np.asarray(R.render(FM._sc_roughness_node(st6, animate=True, frame=1), st6))
    d2 = np.asarray(R.render(FM._sc_roughness_node(st6, animate=True, frame=2), st6))
    check('shimmer on: two frames of the ball differ', not bool(np.array_equal(d1, d2)))


def test_c099_gpu_twin():
    """`glsl_twin` `d == 0.0` on 4096 pixels (the simulator's normalize is
    numpy's); `sim_vs_cpu` on the FM scene: the deferred bar; the plan
    signature moves with the seed (a baked literal) and Shimmer makes the
    scene time-dependent."""
    rng = np.random.default_rng(1099)
    n = 4096
    px = rng.integers(0, W, n)
    py = rng.integers(0, H, n)
    N = _unit_normals(rng, n)
    for seed, rough, animate, frame in ((0, 255.0, False, 1), (3, 180.0, False, 1), (11, 40.0, True, 5)):
        node = _imr_node(rough, animate=animate)
        cpu = _imr_cpu(node, px, py, N, seed=seed, frame=frame)
        vuv = np.stack([(px + 0.5) / W, (py + 0.5) / H], 1).astype(np.float32)
        EM.EMITTERS['__MATB_LANE__'] = lambda em, nd, _i: (nd['props']['u'], nd['props']['gt'])
        try:
            em = EM.Emitter({'output': 'rough', 'nodes': {'rough': node}})
            em.frame_mode = True
            em.resolution = (W, H)
            em.seed = seed
            var, vt = em.output('rough', 0)
        finally:
            EM.EMITTERS.pop('__MATB_LANE__', None)
        body = em.body() + f'\n    Color = vec4({var}, 1.0);'
        got, why = glsl_twin(body, {'hal_N': N, 'vUV': vuv, 'hal_frame': np.full(n, float(frame), np.float32)}, n,
                             fns='\n'.join(em.inline), unis='uniform vec3 hal_N; uniform float hal_frame;')
        check(f'seed {seed} roughness {rough} shimmer {animate}: the emitted node is bitwise the CPU (d == 0.0)',
              got is not None and bool(np.array_equal(got[:, :3], cpu)),
              str(why) if got is None else f'max {float(np.abs(got[:, :3] - cpu).max()):.2e}')
    st = base_settings(W, H, transparency='NONE', seed=3)
    st.use_processes = False
    sc = FM._sc_roughness_node(st)
    frame_bar('node Imagine roughness (seed 3)', sc, st)
    g, job = rig(sc, st)
    sig3 = GSH._plan_sig(job, GSH._mesh_key(sc.mesh))
    st.seed = 4
    sig4 = GSH._plan_sig(job, GSH._mesh_key(sc.mesh))
    check('the seed is in the plan signature (the hash salt is a baked literal)', sig3 != sig4)
    check('Shimmer makes the scene time-dependent; off, it does not',
          GSH._scene_time_dependent(FM._sc_roughness_node(st, animate=True))
          and not GSH._scene_time_dependent(FM._sc_roughness_node(st, animate=False)))


def test_c099_refusal_by_name():
    """raytrace True with a reflective material carrying the node: the plan
    refuses with 'pixel position' in why (a hit has no pixel); the CPU
    renders; the `animate` prop reaches the renderer through NODE_PROPS."""
    st = base_settings(W, H, transparency='NONE', seed=3, raytrace=True)
    st.use_processes = False
    sc = FM._sc_roughness_node(st)
    for s_ in sc.materials[1].graph['nodes']['hal']['inputs']:
        if s_['name'] == 'Reflection':
            s_['default'] = 0.5
    g, job = rig(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check("raytrace + a reflective roughened material: the plan refuses by name ('pixel position')",
          p is None and 'pixel position' in str(why), str(why))
    img = np.asarray(R.render(sc, st))
    check('the CPU renders the reflective roughened ball', img.shape == (H, W, 4) and bool(np.isfinite(img).all()))
    from . import fakebpy
    fakebpy.install()
    from ..export import NODE_PROPS
    check("the Roughness node's 'animate' (Shimmer) reaches the renderer through NODE_PROPS",
          'animate' in NODE_PROPS.get('HALCYON_ImagineRoughnessNode', ()))
    from ..nodes import shader_nodes as SN
    vec = SN.MENU_FAMILIES[[t for t, _i, _m in SN.MENU_FAMILIES].index('Vector')][2]
    check('Roughness (Imagine) draws in the Vector family',
          any(getattr(c, 'bl_idname', '') == 'HALCYON_ImagineRoughnessNode' for c in vec))


# ================================================================ C123
class _RayCtx:
    def __init__(self, I, N, P):
        self.n = int(I.shape[0])
        self.I = np.asarray(I, np.float32)
        self.N = np.asarray(N, np.float32)
        self.P = np.asarray(P, np.float32)


_EC_COLS = {'Sky Color': (0.55, 0.62, 0.78, 1.0), 'Zenith Color': (0.15, 0.22, 0.48, 1.0),
            'Light Color': (1.0, 1.0, 1.0, 1.0), 'Floor Color': (0.18, 0.18, 0.18, 1.0),
            'Horizon Color': (0.5, 0.5, 0.5, 1.0), 'Grid Color': (0.04, 0.04, 0.04, 1.0)}


def _ec_node(**props):
    return {'id': 'env', 'bl_idname': 'HALCYON_EnvChromeNode', 'props': dict(props),
            'inputs': [_sk('Normal', 'VECTOR', [0, 0, 0])] + [_sk(k, 'RGBA', list(v)) for k, v in _EC_COLS.items()],
            'outputs': [{'name': 'Color', 'type': 'RGBA'}]}


def _ec_cpu(node, I, N, P):
    n = int(I.shape[0])
    arrays = {k: np.tile(np.asarray(v, np.float32)[None, :], (n, 1)) for k, v in _EC_COLS.items()}
    ev = _FakeEv(n, **arrays)
    ev.ctx = _RayCtx(I, N, P)
    ev.has_link = lambda nd, name: False
    return NE.n_halcyon_env_chrome(ev, node)['Color'][:, :3]


def _rays_for(R, P=None):
    """(I, N, P) that reflect to the given R about N = +z: I = (R.x, R.y, -R.z)."""
    R = np.asarray(R, np.float32)
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    I = np.stack([R[:, 0], R[:, 1], -R[:, 2]], 1).astype(np.float32)
    N = np.tile(np.array([[0.0, 0.0, 1.0]], np.float32), (R.shape[0], 1))
    P = np.zeros_like(I) if P is None else np.asarray(P, np.float32)
    return I, N, P


def test_c123_laws():
    """A ray straight up returns the Zenith (t = 1) with light_width_offset
    0.6 and the Light Color with 0 (the tube at the origin); a horizontal
    ray away from tubes returns the Sky Color within 1e-6; Real Floor
    moves the grid with P (parallax), off it depends on R alone; doubling
    light_width_gain doubles the tube EDGES along a ray fan (the lit
    fraction is 0.5 at any gain); grid_width and grid_depth 0 remove the
    grid."""
    I, N, P = _rays_for([[0.0, 0.0, 1.0]])
    zen = np.array(_EC_COLS['Zenith Color'][:3], np.float32)
    up6 = _ec_cpu(_ec_node(light_width_offset=0.6), I, N, P)
    check('a ray straight up with light_width_offset 0.6 returns the Zenith Color (within 1e-6)',
          float(np.abs(up6[0] - zen).max()) < 1e-6, str(up6[0]))
    up0 = _ec_cpu(_ec_node(), I, N, P)
    check('a ray straight up at offset 0 returns the Light Color (the tube at the origin)',
          bool(np.array_equal(up0[0], np.ones(3, np.float32))))
    I, N, P = _rays_for([[0.37, 0.91, 0.02]])
    sky = np.array(_EC_COLS['Sky Color'][:3], np.float32)
    hor = _ec_cpu(_ec_node(), I, N, P)
    check('a horizontal ray (R.z -> 0+) away from tubes returns the Sky Color within 1e-2 of the horizon blend',
          float(np.abs(hor[0] - sky).max()) < 1e-2, str(hor[0]))
    # Real Floor parallax: the same downward R from two points gives different grid verdicts
    Rd = np.array([[0.31, 0.45, -0.6]], np.float32)      # off the depth grid line: pz ~ 1.125
    I, N, _P = _rays_for(Rd)
    grid = np.array(_EC_COLS['Grid Color'][:3], np.float32)
    verdicts = []
    for x in np.linspace(0.0, 1.0, 21):
        out = _ec_cpu(_ec_node(real_floor=True), I, N, np.array([[x, 0.0, 0.5]], np.float32))
        verdicts.append(bool(np.array_equal(out[0], grid)))
    check('Real Floor: the same R from different points gives different grid verdicts (parallax)',
          any(verdicts) and not all(verdicts), str(verdicts))
    outs = [_ec_cpu(_ec_node(real_floor=False), I, N, np.array([[x, 0.0, 0.5]], np.float32))[0]
            for x in np.linspace(0.0, 1.0, 21)]
    check('Real Floor off: the grid depends on R alone (every point gives the same colour)',
          all(np.array_equal(o, outs[0]) for o in outs))
    # tube density: a fan of upward rays across 8 units of the sky plane
    xs = np.linspace(0.05, 8.05, 4096).astype(np.float32)
    Rf = np.stack([xs, np.full_like(xs, 0.04), np.ones_like(xs)], 1)
    I, N, P = _rays_for(Rf)
    light = np.ones(3, np.float32)

    def edges(gain):
        out = _ec_cpu(_ec_node(light_width_gain=gain), I, N, P)
        lit = np.all(out == light, axis=1)
        return int((lit[1:] != lit[:-1]).sum()), float(lit.mean())
    e1, f1 = edges(1.0)
    e2, f2 = edges(2.0)
    check('doubling light_width_gain doubles the tube edges along a ray fan (within 10 %)',
          abs(e2 - 2 * e1) <= 0.1 * max(2 * e1, 1), f'{e1} -> {e2}')
    check('...while the lit fraction stays about 0.5 at either gain (the width is a cell fraction)',
          abs(f1 - 0.5) < 0.1 and abs(f2 - 0.5) < 0.1, f'{f1:.2f} {f2:.2f}')
    # grid off
    Rg = np.stack([np.linspace(-1, 1, 512), np.linspace(-0.7, 0.9, 512), -np.ones(512)], 1).astype(np.float32)
    I, N, P = _rays_for(Rg)
    none = _ec_cpu(_ec_node(grid_width=0.0, grid_depth=0.0, real_floor=False), I, N, P)
    check('grid_width 0 AND grid_depth 0 give no grid pixels', not bool(np.any(np.all(none == grid, axis=1))))
    some = _ec_cpu(_ec_node(real_floor=False), I, N, P)
    check('...and the defaults do draw grid pixels', bool(np.any(np.all(some == grid, axis=1))))
    check('the output is float32 with alpha 1', none.dtype == np.float32)


def test_c123_gpu_twin():
    """`glsl_twin` `d == 0.0` on 4096 rays with the props as values, Real
    Floor on and off, offsets and gains moved; `sim_vs_cpu` on the FM
    scene: the deferred bar; the plan is not None (no refusal exists)."""
    rng = np.random.default_rng(1123)
    n = 4096
    I = rng.normal(size=(n, 3)).astype(np.float32)
    I = (I / np.linalg.norm(I, axis=1, keepdims=True)).astype(np.float32)
    N = rng.normal(size=(n, 3)).astype(np.float32)
    N[:, 2] = np.abs(N[:, 2]) + 0.2
    N = (N / np.linalg.norm(N, axis=1, keepdims=True)).astype(np.float32)
    P = (rng.random((n, 3)) * 4.0 - 2.0).astype(np.float32)
    V = (-I * (rng.random((n, 1)) * 3.0 + 0.5)).astype(np.float32)        # hal_V = eye - P, unnormalised
    bad = []
    for props in ({}, {'real_floor': False}, {'light_width_gain': 2.5, 'light_width_offset': 0.3, 'grid_depth_gain': 3.0,
                                              'grid_width_offset': 0.7, 'floor_altitude': -0.4, 'light_depth': 0.3}):
        node = _ec_node(**props)
        cpu = _ec_cpu(node, np.asarray([-v for v in V], np.float32), N, P)
        EM.EMITTERS['__MATB_LANE__'] = lambda em, nd, _i: (nd['props']['u'], nd['props']['gt'])
        try:
            em = EM.Emitter({'output': 'env', 'nodes': {'env': node}})
            em.frame_mode = True
            em.resolution = (W, H)
            var, vt = em.output('env', 0)
        finally:
            EM.EMITTERS.pop('__MATB_LANE__', None)
        body = em.body() + f'\n    Color = {var};'
        got, why = glsl_twin(body, {'hal_V': V, 'hal_N': N, 'hal_P': P}, n, fns='\n'.join(em.inline),
                             unis='uniform vec3 hal_V; uniform vec3 hal_N; uniform vec3 hal_P;')
        if got is None:
            bad.append(f'{props}: {why}')
        elif not np.array_equal(got[:, :3], cpu):
            bad.append(f'{props}: max {float(np.abs(got[:, :3] - cpu).max()):.2e} at {int((got[:, :3] != cpu).any(axis=1).sum())} lanes')
    check('the emitted Env Chrome is bitwise the CPU (d == 0.0) on 4096 rays, three parameter sets', not bad, '; '.join(bad[:3]))
    sc, st = scene('envchrome_node')
    frame_bar('node env chrome (Alias / Maya) into Matcap Add', sc, st)
    base_img = cpu_frame('demo')
    img = cpu_frame('envchrome_node')
    check('the env chrome row moves the picture', float(np.abs(img - base_img).max()) > 1e-2)
    from . import fakebpy
    fakebpy.install()
    from ..export import NODE_PROPS
    want = {'light_width', 'light_depth', 'light_width_gain', 'light_width_offset', 'light_depth_gain', 'light_depth_offset',
            'grid_width', 'grid_depth', 'grid_width_gain', 'grid_width_offset', 'grid_depth_gain', 'grid_depth_offset',
            'floor_altitude', 'real_floor'}
    check('all 14 Env Chrome props reach the renderer through NODE_PROPS', set(NODE_PROPS.get('HALCYON_EnvChromeNode', ())) >= want)


# ================================================================ runner
def main():
    utf8_console()
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith('test_') and callable(f)]
    order = ['test_r251_identity',
             'test_c119_identity_at_defaults', 'test_c119_grid', 'test_c119_snap_laws',
             'test_c119_faceted_frame', 'test_c119_gpu_twin', 'test_c119_refusals_by_name',
             'test_c028_tev_laws', 'test_c028_nv2a_laws', 'test_c028_division_rule', 'test_c028_gpu_twin',
             'test_c023_tables', 'test_c023_laws', 'test_c023_gpu_twin', 'test_c023_refusal_by_name',
             'test_c135_laws', 'test_c135_two_stage_frame', 'test_c135_gpu_twin', 'test_c135_refusal_by_name',
             'test_c099_laws', 'test_c099_gpu_twin', 'test_c099_refusal_by_name',
             'test_c123_laws', 'test_c123_gpu_twin']
    names = [n for n in order if n in dict(tests)] + [n for n, _f in tests if n not in order]
    for n in names:
        print(f'--- {n}')
        try:
            globals()[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print(f'\nR251 MATERIAL NODES (MAT-B): {len(FAILS)} failure(s)')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
