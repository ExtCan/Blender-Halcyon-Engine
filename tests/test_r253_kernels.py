"""R253 (field): the suite on another machine's NumPy.

A cloud container (Linux, Python 3.13, NumPy 2.5.3 on OpenBLAS 0.3.34,
a Xeon it runs as SkylakeX) failed 20 checks that demand `d == 0.0` of a
GPU sky twin -- 10 to 13 pixels of the 96x72 frame, 4e-08 to 8e-07 --
and raised in one test:

    sky NODES sky_blend (the demo world): ... 10 px differ
    GPU twin, squeezes (7, 3): the LW backdrop is bitwise ...  d 7.15e-07
    the sky pass under Ground Fog is bitwise ...  5.96e-07
    TypeError: only 0-dimensional arrays can be converted to Python scalars

Two causes, neither of them the NumPy VERSION alone.

(1) The CPU sky's ray was a matmul, `pts @ inv.T`, which NumPy hands to
    BLAS `sgemm`. The sky pass states the same sum as four float32
    statements, `((nx*i.x + ny*i.y) + i.z) + i.w`. OpenBLAS's unfused
    cores (Prescott, Nehalem, Sandybridge) sum in that order and round
    after every operation; its Haswell-and-later cores (Haswell, Zen,
    SkylakeX, Cooperlake) fuse the second product into the first,
    `fma(ny, i.y, nx*i.x)` -- one rounding where the chain has two --
    and land an ulp away. Which core runs is the bundled OpenBLAS's
    reading of the CPU: on that Xeon, NumPy 1.26.4 (0.3.23) fell back to
    Prescott and passed, 2.0.2 .. 2.5.3 (0.3.27 .. 0.3.34) picked
    Cooperlake / SkylakeX and failed, and with OPENBLAS_CORETYPE forced
    the verdict follows the core on either NumPy (1.26.4 on Haswell
    fails the same 20; 2.5.3 on Nehalem passes them).
    A one-row product (a frame with ONE uncovered pixel) took another
    routine and another order again, off the chain on every core.
    `mathx.unproject_far` is the chain, statement for statement, on
    NumPy's own multiply and add -- a construction on every build -- and
    `render._background_image` and `sky.mode7_rows` call it.
(2) `raster.jitter_at` answered a pair of SCALAR indices with (1,)
    arrays (its hash needs arrays and the reshape never came back), and
    the test's `float(jx)` is a DeprecationWarning from NumPy 1.25 and a
    TypeError from 2.4. The offsets come back in the indices' own shape.

(3) The fix moves the CPU sky by that ulp wherever the kernel is
    fused, and two kinds of pin hold a CPU frame's bytes: the previous
    releases in their zips (`test_render._prev_engine`), whose ray is
    the matmul, and `test_r253_mapping`'s three frame digests. The
    loader now puts the previous engine's ONE ray line on this tree's
    chain, so a zip pin compares the releases and not the machine's
    BLAS (the same bits as before on an unfused kernel); the digests
    are proven through the old ray and pinned as they render now.

What is pinned, on every machine: the chain against an independent
float64-then-round reference; the twin at `d == 0.0` in four sky modes;
the SAME twins moved, at the container's magnitudes, when a fused ray
(the kernel's arithmetic, emulated) stands in for the chain -- so the
failure is reproduced wherever the suite runs, not only on a fused
BLAS; no matmul left in the two functions; the shape law; the loader's
one-line replacement on the shipped text. This machine's own matmul is
printed as a note, never asserted.

Runs alone::

    python -m halcyon.tests.test_r253_kernels
"""
import inspect
import re
import sys
import traceback
import warnings

import numpy as np

from . import utf8_console
from .scenebuild import demo_scene
from .test_render import base_settings
from ..core import mathx as M
from ..core import raster as CR
from ..core import render as R
from ..core import sky as SK
from ..gpu import sky as GSKY

FAILS = []
f32 = np.float32
f64 = np.float64

W, H = 96, 72


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f'  {extra}' if extra else ''))
    if not ok:
        FAILS.append(name)


def note(text):
    print(f'  note {text}')


# ------------------------------------------------------------------ helpers


def _reference(nx, ny, inv):
    """The chain by another road: every operation in float64, rounded to
    float32 ONCE. For one +, - or * of two float32 values that is the
    correctly rounded float32 result (53 >= 2 * 24 + 2: the double
    rounding is innocuous), so this is IEEE's own answer for the five
    statements, with no float32 arithmetic of NumPy's in it."""
    out = np.empty((nx.shape[0], 4), f32)
    for r in range(4):
        a = (nx.astype(f64) * f64(inv[r, 0])).astype(f32)
        b = (ny.astype(f64) * f64(inv[r, 1])).astype(f32)
        s = (a.astype(f64) + b.astype(f64)).astype(f32)
        s = (s.astype(f64) + f64(inv[r, 2])).astype(f32)
        s = (s.astype(f64) + f64(inv[r, 3])).astype(f32)
        out[:, r] = s
    return out


def _fused(nx, ny, inv):
    """What an FMA `sgemm` computes for the same rows: the first product
    rounded, the SECOND fused into it (the product exact in float64 --
    two 24-bit significands -- added and rounded once), then the two
    exact `1 * i` terms. Measured bitwise OpenBLAS's Haswell, Zen,
    SkylakeX and Cooperlake `sgemm` on 100,000 random rows."""
    out = np.empty((nx.shape[0], 4), f32)
    for r in range(4):
        s = nx * inv[r, 0]
        s = (ny.astype(f64) * f64(inv[r, 1]) + s.astype(f64)).astype(f32)
        s = s + inv[r, 2]
        s = s + inv[r, 3]
        out[:, r] = s
    return out


def _frame(world=None, ss=1, **over):
    """The suite's sky rig (test_gpu_sky_pass): the demo scene at 96x72,
    its G-buffer and the frame's own view-projection."""
    st = base_settings(W, H)
    st.shadows = False
    st.transparency = 'NONE'
    if ss > 1:
        st.aa_mode = 'SUPERSAMPLE'
        st.aa_samples = ss * ss
    for k, v in over.items():
        setattr(st, k, v)
    sc = demo_scene(st, with_texture=False)
    for k, v in (world or {}).items():
        setattr(sc.world, k, v)
    rw, rh = W * ss, H * ss
    _view, _proj, vp, eye = R.camera_matrices(sc.camera, rw, rh)
    g = CR.GBuffer(rw, rh)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, rw, rh, gbuf=g)
    return sc, st, vp, eye, g


def _ndc(g):
    """The uncovered pixels' NDC and count, as _background_image builds
    them."""
    h, w = g.tri.shape
    yy, xx = np.nonzero(~g.mask())
    nx = (xx.astype(f32) + 0.5) / w * 2.0 - 1.0
    ny = (yy.astype(f32) + 0.5) / h * 2.0 - 1.0
    return nx, ny


def _twin(sc, st, vp, eye, g, ss):
    """(pixels that differ, max |d|) between the sky pass in the
    simulator and the CPU sky, at the uncovered pixels."""
    rh, rw = g.tri.shape
    sim, why = GSKY.simulate(sc, g, st, vp, eye, {}, ss=ss)
    if sim is None:
        return None, why
    cpu = R._background_image(sc, st, rw, rh, vp, eye, ~g.mask(), {}, ss=ss)
    unc = ~g.mask()
    n = int((sim[unc] != cpu[unc]).any(axis=1).sum())
    d = float(np.abs(sim[unc].astype(f64) - cpu[unc].astype(f64)).max())
    return n, d


#: the four twins of the report, one per family that failed: the NODES
#: blend, the block road, a `pow`, LightWave's squeezes
MODES = [
    ('NODES sky_blend (the demo world)', {}, 1),
    ('NODES sky_blend at ss 2 under fast_background', {}, 2),
    ('GRADIENT pow 0.7, SMOOTH, the ground shown',
     {'mode': 'GRADIENT', 'gradient_falloff': 0.7, 'blend_mode': 'SMOOTH',
      'show_ground': True, 'horizon_height': 0.15}, 1),
    ('LW_GRADIENT, squeezes (7, 3)',
     {'mode': 'LW_GRADIENT', 'lw_sky_squeeze': 7, 'lw_ground_squeeze': 3}, 1),
]


# -------------------------------------------------------------------- tests


def test_a_the_ray_is_the_chain_by_construction():
    """`mathx.unproject_far` is the sky pass's five statements per row,
    whatever BLAS is installed: bitwise an independent reference on the
    demo frame's own pixels and matrix, and on a random batch."""
    sc, st, vp, eye, g = _frame()
    inv = np.linalg.inv(vp).astype(f32)
    nx, ny = _ndc(g)
    got = M.unproject_far(nx, ny, inv)
    ref = _reference(nx, ny, inv)
    check('unproject_far is (N,4) float32, bitwise the float64-then-round '
          "reference of hal_sky_ray's chain at every uncovered pixel of the "
          'demo frame', got.dtype == f32 and got.shape == (nx.size, 4)
          and np.array_equal(got, ref), f'{nx.size} px')
    rng = np.random.default_rng(253)
    rinv = rng.standard_normal((4, 4)).astype(f32)
    rx = (rng.random(100000).astype(f32) * f32(2.0) - f32(1.0))
    ry = (rng.random(100000).astype(f32) * f32(2.0) - f32(1.0))
    check('...and on 100,000 random rows through a random matrix',
          np.array_equal(M.unproject_far(rx, ry, rinv),
                         _reference(rx, ry, rinv)))
    one = M.unproject_far(nx[:1], ny[:1], inv)
    check('a ONE-pixel batch is the same row (a one-row matmul takes '
          'another BLAS routine and another order again, on fused and '
          'unfused cores alike; the chain has no batch in it)',
          one.shape == (1, 4) and np.array_equal(one[0], got[0]))
    # the fused chain is a DIFFERENT function on these very pixels: this
    # is the pin's teeth, and it does not depend on the BLAS installed
    fz = _fused(nx, ny, inv)
    rows = int((fz != got).any(axis=1).sum())
    rrows = int((_fused(rx, ry, rinv) != M.unproject_far(rx, ry, rinv))
                .any(axis=1).sum())
    check('a fused second product (an FMA `sgemm`) is NOT the chain: it '
          'moves rows of the demo frame and about three in ten of the '
          'random batch', rows > 0 and 20000 < rrows < 40000,
          f'{rows} of {nx.size} rows; {rrows} of 100000')
    # what THIS machine's matmul does -- reported, never asserted: the
    # engine no longer asks it
    pts = np.stack([rx, ry, np.ones(rx.size, f32), np.ones(rx.size, f32)],
                   axis=1)
    mm = pts @ rinv.T
    off = int((mm != M.unproject_far(rx, ry, rinv)).any(axis=1).sum())
    fus = int((mm != _fused(rx, ry, rinv)).any(axis=1).sum())
    kind = ('the chain, bitwise (an unfused kernel)' if off == 0 else
            'the fused chain, bitwise (an FMA kernel)' if fus == 0 else
            'neither the chain nor the fused chain')
    note(f"NumPy {np.__version__}: this machine's float32 matmul is {kind}"
         f' -- {off} of 100000 rows off the chain')


def test_b_the_sky_twins_hold_and_a_fused_ray_moves_them():
    """The four families of the report at `d == 0.0`; then the same
    twins with the fused ray standing in for the chain: every one moves,
    by the report's magnitudes. The second half is the container's
    failure, reproduced on any machine."""
    real = M.unproject_far
    for label, world, ss in MODES:
        sc, st, vp, eye, g = _frame(world, ss=ss)
        n, d = _twin(sc, st, vp, eye, g, ss)
        if n is None:
            check(f'sky {label}: the pass draws', False, str(d))
            continue
        check(f'sky {label}: the GPU pass is bitwise the CPU sky at every '
              'uncovered pixel (d == 0.0)', n == 0 and d == 0.0,
              f'{n} px differ, max {d:g}')
        M.unproject_far = _fused
        try:
            nf, df = _twin(sc, st, vp, eye, g, ss)
        finally:
            M.unproject_far = real
        check(f'...and with a fused ray in the chain\'s place the twin '
              'moves (an ulp or two at a handful of pixels: the '
              "container's failure)", nf is not None and nf > 0
              and 0.0 < df < 4e-6, f'{nf} px differ, max {df:g}')
    check('the stand-in was handed back', M.unproject_far is real)


def test_c_no_matmul_is_left_in_the_sky_ray():
    """The two functions that unproject a pixel for the sky call the
    chain and hold no matmul, and the chain holds none either."""
    for fn, label in ((R._background_image, 'render._background_image'),
                      (SK.mode7_rows, 'sky.mode7_rows')):
        src = inspect.getsource(fn)
        code = '\n'.join(ln.split('#', 1)[0] for ln in src.splitlines())
        check(f'{label} unprojects through mathx.unproject_far and holds no '
              'matmul', 'M.unproject_far(' in code and ' @ ' not in code
              and 'np.dot(' not in code and 'einsum' not in code
              and 'matmul' not in code)
    src = inspect.getsource(M.unproject_far)
    body = src.split('"""')[2]
    code = '\n'.join(ln.split('#', 1)[0] for ln in body.splitlines())
    calls = re.findall(r'np\.(\w+)\(', code)
    check('unproject_far is multiply and add alone, one ufunc call per '
          'statement of the chain (two products, three sums; no @, dot, '
          'einsum or matmul)', ' @ ' not in code and 'dot(' not in code
          and 'einsum' not in code and 'matmul' not in code
          and calls.count('multiply') == 2 and calls.count('add') == 3
          and set(calls) <= {'multiply', 'add', 'asarray', 'empty'},
          str(sorted(set(calls))))
    sky_src = GSKY.SOURCE
    check("the sky pass's hal_sky_ray still states the chain the CPU "
          'transcribes (((nx * i.x + ny * i.y) + i.z) + i.w, four rows)',
          all(f'((nx * i{r}.x + ny * i{r}.y) + i{r}.z) + i{r}.w' in sky_src
              for r in range(4)))


def test_d_mode7_registers_read_the_same_ray():
    """`sky.mode7_rows` builds its per-row registers from the row's first
    and last pixel-centre rays 'exactly as _background_image unprojects
    them': the same function, so the same bits on any kernel."""
    sc, st, vp, eye, g = _frame()
    inv = np.linalg.inv(vp).astype(f32)
    Ho = H
    yo = np.arange(Ho, dtype=f32)
    ny = (yo + f32(0.5)) / f32(Ho) * f32(2.0) - f32(1.0)
    nx0 = np.full(Ho, (0.5 / W) * 2.0 - 1.0, f32)
    seen = []
    real = M.unproject_far

    def spy(nx, ny_, inv_):
        out = real(nx, ny_, inv_)
        seen.append((np.array(nx, f32), np.array(ny_, f32), out.copy()))
        return out

    class _World:
        mode7_texel_size = 0.25
        ground_height = 0.0

    M.unproject_far = spy
    try:
        hit, A, C, X0, Y0 = SK.mode7_rows(_World(), inv, eye, W, Ho)
    finally:
        M.unproject_far = real
    check('mode7_rows unprojects its two edge columns through the chain '
          '(two calls, the first column and the row centres the frame '
          'uses)', len(seen) == 2 and np.array_equal(seen[0][0], nx0)
          and np.array_equal(seen[0][1], ny)
          and np.array_equal(seen[0][2], _reference(nx0, ny, inv)))
    check('...and answers one register set per output row (int32, the '
          'rows the floor can reach marked)', hit.shape == (Ho,)
          and A.dtype == np.int32 and X0.shape == (Ho,) and bool(hit.any()))


def test_e_jitter_at_answers_in_the_shape_it_was_asked():
    """C127's offsets come back in the indices' own (broadcast) shape:
    planes, lists, and 0-d float32 for one pixel's pair of scalars --
    `float()` of that pair warns on no NumPy and raises on none."""
    key = CR.jitter_key(1, 0)
    JX, JY = CR.jitter_offsets(64, 48, key)
    check('the planes are (H, W) float32, as before', JX.shape == (48, 64)
          and JY.shape == (48, 64) and JX.dtype == f32 and JY.dtype == f32)
    for label, ix, iy in (('NumPy uint32 scalars', np.uint32(5), np.uint32(7)),
                          ('Python ints', 5, 7),
                          ('0-d arrays', np.array(5, np.uint32),
                           np.array(7, np.uint32))):
        jx, jy = CR.jitter_at(ix, iy, key)
        with warnings.catch_warnings():
            # NumPy 1.25 .. 2.3 only WARN on float() of a (1,) array:
            # as an error here, so the pin bites on those too
            warnings.simplefilter('error')
            try:
                fx, fy = float(jx), float(jy)
                why = ''
            except Exception as exc:                            # noqa: BLE001
                fx = fy = None
                why = f'{type(exc).__name__}: {exc}'
        check(f'a pair of {label} answers 0-d float32, float() of it is '
              "silent and it is the planes' own value at that pixel",
              np.shape(jx) == () and np.shape(jy) == ()
              and np.asarray(jx).dtype == f32 and np.asarray(jy).dtype == f32
              and fx == float(JX[7, 5]) and fy == float(JY[7, 5]), why)
    xs = np.array([5, 0, 63, 17], np.uint32)
    ys = np.array([7, 0, 47, 30], np.uint32)
    jx, jy = CR.jitter_at(xs, ys, key)
    check('a list of pixels answers a list (the compute raster\'s caller: '
          'it indexes jx[:, None])', jx.shape == (4,) and jy.shape == (4,)
          and np.array_equal(jx, JX[ys, xs]) and np.array_equal(jy, JY[ys, xs])
          and jx[:, None].shape == (4, 1))
    jx, jy = CR.jitter_at(np.array([5], np.uint32), np.array([7], np.uint32),
                          key)
    check('ONE pixel handed as arrays stays a (1,) array (the shape asked '
          'for, not a guess at what was meant)', jx.shape == (1,)
          and jy.shape == (1,) and jx[0] == JX[7, 5] and jy[0] == JY[7, 5])
    jx, jy = CR.jitter_at(np.arange(64, dtype=np.uint32)[None, :],
                          np.arange(48, dtype=np.uint32)[:, None], key)
    check('a row of x against a column of y broadcasts to the planes, '
          'bitwise', jx.shape == (48, 64) and np.array_equal(jx, JX)
          and np.array_equal(jy, JY))
    jx, jy = CR.jitter_at(np.uint32(5), ys, key)
    check('a scalar against a list takes the list\'s shape', jx.shape == (4,)
          and np.array_equal(jx, JX[ys, 5]) and np.array_equal(jy, JY[ys, 5]))


_SHIPPED_RENDER = """import numpy as np


def background(nx, ny, inv):
    pts = np.stack([nx, ny, np.ones(nx.size, np.float32),
                    np.ones(nx.size, np.float32)], axis=1)
    world = pts @ inv.T
    return world
"""

_SHIPPED_SKY = """import numpy as np


def mode7_rows(nx, ny, inv):
    def ray(nx):
        pts = np.stack([nx, ny, np.ones(nx.size, np.float32),
                        np.ones(nx.size, np.float32)], 1)
        wv = pts @ inv.T
        return wv
    return ray(nx)
"""


def test_f_a_previous_engine_is_compared_on_the_same_ray():
    """`test_render._prev_engine` imports a shipped zip to render 'bitwise
    the previous release'. Every release computed the sky ray as the
    matmul, so on a fused BLAS such a pin would measure the kernel, not
    the release: the loader replaces that ONE line in the extracted copy
    with this tree's own chain. Proven here on the two lines as every
    release from 1.32 to 1.92.0 wrote them (no zip needed)."""
    import io
    import os
    import shutil
    import tempfile
    from contextlib import redirect_stdout
    from . import test_render as TR
    sc, st, vp, eye, g = _frame()
    inv = np.linalg.inv(vp).astype(f32)
    nx, ny = _ndc(g)
    want = M.unproject_far(nx, ny, inv)

    def load(path):
        ns = {}
        with open(path, 'r', encoding='utf-8') as fh:
            text = fh.read()
        exec(compile(text, path, 'exec'), ns)                   # noqa: S102
        return ns, text

    root = tempfile.mkdtemp(prefix='halcyon_kern_')
    try:
        os.makedirs(os.path.join(root, 'core'))
        rp = os.path.join(root, 'core', 'render.py')
        sp = os.path.join(root, 'core', 'sky.py')
        with open(rp, 'w', encoding='utf-8') as fh:
            fh.write(_SHIPPED_RENDER)
        with open(sp, 'w', encoding='utf-8') as fh:
            fh.write(_SHIPPED_SKY)
        ns0, _t = load(rp)
        shipped = ns0['background'](nx, ny, inv)
        off = int((shipped != want).any(axis=1).sum())
        buf = io.StringIO()
        with redirect_stdout(buf):
            done = TR._chain_prev_sky_ray(root, 'halcyon-test.zip')
        ns1, t1 = load(rp)
        ns2, t2 = load(sp)
        check('the loader chains both shipped lines (core/render.py and '
              'core/sky.py), silently', done == ['core/render.py', 'core/sky.py']
              and buf.getvalue() == '', f'{done} {buf.getvalue()!r}')
        check("the previous engine's ray is then this tree's chain, bitwise, "
              'in _background_image and in mode7_rows (as shipped it is '
              "this machine's matmul)",
              np.array_equal(ns1['background'](nx, ny, inv), want)
              and np.array_equal(ns2['mode7_rows'](nx, ny, inv), want),
              f'as shipped {off} of {nx.size} rows off the chain here')
        head1 = t1.split('# appended by tests.test_render._prev_engine')[0]
        check('line for line: the shipped text keeps its line numbers (one '
              'line replaced, the chain appended after the last one) and the '
              "appended function is mathx.unproject_far's own source",
              head1.rstrip('\n').count('\n')
              == _SHIPPED_RENDER.rstrip('\n').count('\n')
              and '    world = _hal_chain_ray(nx, ny, inv)\n' in t1
              and '        wv = _hal_chain_ray(nx, ny, inv)\n' in t2
              and inspect.getsource(M.unproject_far).replace(
                  'def unproject_far(', 'def _hal_chain_ray(', 1) in t1)
        # exactly once, or not at all
        with open(rp, 'w', encoding='utf-8') as fh:
            fh.write(_SHIPPED_RENDER + '\n\ndef again(pts, inv):\n'
                     '    world = pts @ inv.T\n    return world\n')
        twice = open(rp, encoding='utf-8').read()
        os.remove(sp)
        TR._PREV_SKY_RAY_NOTED.discard(('halcyon-test.zip', 'core/render.py'))
        buf = io.StringIO()
        with redirect_stdout(buf):
            done = TR._chain_prev_sky_ray(root, 'halcyon-test.zip')
            TR._chain_prev_sky_ray(root, 'halcyon-test.zip')
        check('a file that holds the line twice is left as shipped and named '
              'ONCE (a previous engine is never edited on a guess)',
              done == [] and open(rp, encoding='utf-8').read() == twice
              and buf.getvalue().count('halcyon-test.zip') == 1
              and 'keeps its own ray' in buf.getvalue(), buf.getvalue().strip())
        TR._PREV_SKY_RAY_NOTED.discard(('halcyon-test.zip', 'core/render.py'))
        # a release cut AFTER the fix (the 1.92.0 zip the R254 pins ask
        # for): its ray is the chain already, nothing to replace or say
        after = ('import numpy as np\nfrom . import mathx as M\n\n\n'
                 'def background(nx, ny, inv):\n'
                 '    world = M.unproject_far(nx, ny, inv)\n'
                 '    return world\n')
        with open(rp, 'w', encoding='utf-8') as fh:
            fh.write(after)
        buf = io.StringIO()
        with redirect_stdout(buf):
            done = TR._chain_prev_sky_ray(root, 'halcyon-test.zip')
        check('a release that already states the chain is left as shipped, '
              'silently', done == [] and buf.getvalue() == ''
              and open(rp, encoding='utf-8').read() == after,
              buf.getvalue().strip())
    finally:
        shutil.rmtree(root, True)
    src = inspect.getsource(TR._prev_engine)
    check('_prev_engine chains the ray before it imports the engine, and a '
          'failure there cannot stop the import',
          src.index('_chain_prev_sky_ray(') < src.index('importlib.import_module(')
          and 'stays as shipped' in src)


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
    print('all R253 kernel / NumPy checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
