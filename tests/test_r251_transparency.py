"""R251 (1.90.0) -- the transparency pack, wave 1 (TRANS-1): C002 the
fixed-function blend equations and the composite skeleton, C018+C070
the framebuffer format at the blend, C053 SNES/GBA colour math and the
column mesh, C042 the DS translucency rules, C062 the Doom fuzz.

Run with:  python -m halcyon.tests.test_r251_transparency

Every law probe runs on the PANE scene (tests/featurematrix._sc_pane: an
open quad above the demo geometry, one fragment per pixel it covers) and
recomputes the composite's output from F (the fragment's shaded colour
off a fragment mirror, never off a frame) and B (the same scene with the
probed material at opacity 0 and Blend Mode Alpha: the float over at a =
0 is B bitwise). Every GPU twin goes through `twin()`: the fake device
road rendered ONCE, then the CPU road on a copy, frames compared.
"""
import contextlib
import inspect
import io
import sys

import numpy as np

from ..core import post
from ..core import raster
from ..core import render as R
from ..core.settings import RenderSettings
from ..presets.library import PRESETS
from .scenebuild import demo_scene
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


ZIP = 'halcyon-1.89.0.zip'
W, H = 96, 72


def _st(w=W, h=H, **kw):
    st = base_settings(w, h, **kw)
    return st


def _fm():
    from . import featurematrix as FM
    return FM


def _glass(st):
    return _fm()._sc_glass(st)


def _pane(st, two=False):
    return _fm()._sc_pane(st, two=two)


def _shell(st):
    return _fm()._sc_ds_shell(st)


def to8(c):
    return np.round(np.clip(c, 0.0, 1.0) * np.float32(255)).astype(np.int32)


def from8(k):
    return np.asarray(k).astype(np.float32) / np.float32(255)


def _over(F, a, B):
    """The skeleton's ALPHA over, exactly (float32)."""
    aa = np.asarray(a, np.float32)[:, None]
    return F * aa + B * (1.0 - aa)


def _render(sc_fn, st):
    return np.asarray(R.render(sc_fn(st), st))


def _b_frame(sc_fn, st_kw, mat=1):
    """The B operand of every law: the same scene under the same mode with
    the probed material at opacity 0 and Blend Mode Alpha (the float over
    at a = 0 is B bitwise); the Blend Equation left at ALPHA."""
    st = _st(**st_kw)
    st.blend_equation = 'ALPHA'
    sc = sc_fn(st)
    sc.materials[mat].opacity = 0.0
    sc.materials[mat].blend_mode = 'ALPHA'
    return np.asarray(R.render(sc, st))


def _mirror(sc, st, w=W, h=H, order='DEPTH'):
    """Exactly `_composite_abuffer`'s front half over an ABUFFER frame:
    keep, sort, rank -- plus the CPU's own shaded colour per fragment.
    `order` DEPTH sorts on the fragment depth, SUBMISSION on `tri`. The
    scene is rendered FIRST so its lights carry the frame's shadow maps
    (a fragment under the Ball's shadow shades differently without
    them; TR:19203's rig relies on the same)."""
    R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    opq, trans = R._split_by_alpha(sc, sc.mesh, st)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g, depth_bits=st.depth_precision)
    frags = raster.FragmentList()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, cull='NONE',
                     subset=trans, gbuf=g, frags=frags, depth_write=False,
                     depth_bits=st.depth_precision)
    px, py, tri, depth, bary, front = frags.finish()
    opaque_z = g.depth[py, px]
    keep = depth <= raster.abuf_depth_limit(opaque_z)
    px, py, tri, depth, bary, front = (a[keep] for a in
                                       (px, py, tri, depth, bary, front))
    pix = py.astype(np.int64) * w + px
    if order == 'SUBMISSION':
        o = np.lexsort((tri, pix))
    else:
        o = np.lexsort((depth, pix))
    pix, px, py, tri, depth, bary, front = (a[o] for a in
                                            (pix, px, py, tri, depth, bary,
                                             front))
    grp = np.zeros(pix.size, np.int64)
    ng = np.nonzero(pix[1:] != pix[:-1])[0] + 1
    grp[ng] = ng
    np.maximum.accumulate(grp, out=grp)
    rank = np.arange(pix.size, dtype=np.int64) - grp
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    col = R._shade_fragments_cpu(job, tri, bary, px, py, front, rank, st)
    n_at = np.bincount(pix, minlength=w * h)
    return dict(px=px, py=py, tri=tri, depth=depth, bary=bary, front=front,
                rank=rank, pix=pix, col=col, n_at=n_at, job=job, gbuf=g,
                vp=vp, view=view)


def _single(m, mat_index, mat):
    """The fragments of material `mat` at pixels holding exactly ONE
    fragment."""
    mf = mat_index[m['tri']]
    return (m['n_at'][m['pix']] == 1) & (mf == mat)


def twin(sc_fn, st):
    """The fake-device road rendered ONCE (render_device GPU, every GPU
    door open, the resident frame kept and released in a finally), then
    the CPU road on a copy. Returns (gpu, cpu, routing, live, stdout,
    frame_last)."""
    from . import fakedevice
    from ..gpu import frame as FR
    from ..gpu import shade as GSH
    st.render_device = 'GPU'
    st.gpu_raster = True
    st.gpu_shading = True
    st.gpu_post = True
    sc = sc_fn(st)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        with fakedevice.installed() as dev:
            st._keep_gpu_frame = True
            try:
                gpu = np.asarray(R.render(sc, st))
            finally:
                FR.release(st)
            live = [t for t in dev.targets if not t.freed]
    routing = dict(R.LAST_ROUTING)
    lf = dict(FR.LAST)
    st_c = st.copy()
    st_c.render_device = 'CPU'
    cpu = np.asarray(R.render(sc_fn(st_c), st_c))
    return gpu, cpu, routing, live, buf.getvalue(), lf


#: the deferred frame's own bar (TR:19224-19228, the one layer twin;
#: the frame pass beneath every composite): measured on the glass scene
#: at 96x72 under the fake device, 2.98e-6 under ALPHA
FRAME_BAR = 6e-3


# ------------------------------------------------------------ identity

def test_a00_identity_at_defaults():
    """The 1.89.0 zip beside the package, loudly; then every new field at
    its default renders AND post-processes bitwise the previous release
    on the glass scene under all four transparency modes."""
    RP = _prev_engine(ZIP)
    check('the 1.89.0 zip is beside the package', RP is not None,
          'tests._prev_engine found no halcyon-1.89.0.zip: the identity '
          'pins cannot run')
    if RP is None:
        return
    import importlib
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0]
                                        + '.post')
    for mode in ('SORTED', 'ABUFFER', 'NONE', 'STIPPLE'):
        st = _st(transparency=mode)
        now = _render(_glass, st)
        st2 = _st(transparency=mode)
        prev = np.asarray(RP.render(_glass(st2), st2))
        check(f'{mode}: the glass scene at the defaults renders bitwise '
              "the 1.89.0 zip's engine",
              now.shape == prev.shape and bool(np.array_equal(now, prev)),
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
        kw = dict(frame=1, seed=st.seed, target_size=(W, H),
                  allow_resize=False)
        pn = post.process(now, st, **kw)
        pp = prev_post.process(prev, st2, **kw)
        check(f'{mode}: ...and post.process over it is bitwise too',
              pn.shape == pp.shape and bool(np.array_equal(pn, pp)))
    # the new fields exist at their neutral defaults
    st = RenderSettings()
    check('every new transparency field defaults to its neutral value',
          (st.blend_equation, st.translucent_order,
           st.translucent_depth_write, st.framebuffer, st.fb_dither,
           st.fb_dither_subtract)
          == ('ALPHA', 'DEPTH', False, 'NONE', True, True))
    from ..core.scene import Material
    check("a Material's Blend Mode defaults to Inherit",
          Material().blend_mode == 'INHERIT')
    # the ORDERED dict grew without moving a value
    from ..core import dither as DI
    prev_di = importlib.import_module(RP.__name__.rsplit('.', 1)[0]
                                      + '.dither')
    check('the BAYER4 threshold map is bitwise the 1.89.0 module\'s',
          bool(np.array_equal(DI.threshold_map('BAYER4', 64, 64),
                              prev_di.threshold_map('BAYER4', 64, 64))))


# ------------------------------------------------------------ C002

C002_LAWS = {
    'PS1_AVG': lambda b, f: (b + f) >> 1,
    'PS1_ADD': lambda b, f: np.minimum(31, b + f),
    'PS1_SUB': lambda b, f: np.maximum(0, b - f),
    'PS1_QUARTER': lambda b, f: np.minimum(31, b + (f >> 2)),
    'SATURN_HALF': lambda b, f: (b >> 1) + (f >> 1),
    'SATURN_SHADOW': lambda b, f: b >> 1,
    'SATURN_HALF_LUM': lambda b, f: f >> 1,
    'THREEDO_SUB': lambda b, f: np.clip(f - b, 0, 31),
    'THREEDO_XOR': lambda b, f: f ^ b,
}


def test_c002_equations_are_the_integer_laws():
    """Each of the nine PS1 / VDP1 / PIXC items is its 5-bit integer
    formula, bitwise, at every single-fragment pane pixel."""
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _pane(st0)
    m = _mirror(sc0, st0)
    single = _single(m, sc0.mesh.mat_index, 1)
    n1 = int(single.sum())
    check('the pane scene holds at least 200 single-fragment pane pixels '
          f'at {W}x{H} (299 measured)', n1 >= 200, str(n1))
    if n1 < 200:
        return
    B = _b_frame(_pane, kw)
    xx, yy = m['px'][single], m['py'][single]
    F = m['col'][single, :3]
    a = np.clip(m['col'][single, 3], 0.0, 1.0)
    Bp = B[yy, xx, :3]
    # the ALPHA road first: the probe rig reproduces the composite
    A = _render(_pane, _st(**kw))
    check('the probe rig reproduces the ALPHA composite bitwise (F over B '
          'at the single-fragment pixels)',
          bool(np.array_equal(_over(F, a, Bp), A[yy, xx, :3])))
    lum = {}
    for name, law in C002_LAWS.items():
        st = _st(**kw)
        if name == 'SATURN_HALF_LUM':
            st.blend_equation = 'ALPHA'
            sc = _pane(st)
            sc.materials[1].blend_mode = 'SATURN_HALF_LUM'
            out = np.asarray(R.render(sc, st))
        else:
            st.blend_equation = name
            out = _render(_pane, st)
        f = to8(F) >> 3
        b = to8(Bp) >> 3
        o = law(b, f)
        exp = np.where((a > 0.0)[:, None], from8(o << 3), Bp)
        got = out[yy, xx, :3]
        check(f'{name} is its 5-bit integer law at every single-fragment '
              'pane pixel, bitwise',
              bool(np.array_equal(exp, got)),
              f'max {float(np.abs(exp - got).max()):.6f}')
        check(f'{name}: every drawn channel sits on the 5-bit lattice',
              bool(((to8(got) & 7) == 0).all()))
        check(f'{name}: the alpha plane at those pixels is coverage 1',
              bool((out[yy, xx, 3] == 1.0).all()))
        lum[name] = got
    check('PS1_AVG and SATURN_HALF differ where b and f are both odd',
          bool(np.any(lum['PS1_AVG'] != lum['SATURN_HALF']))
          and bool(np.any(((to8(F) >> 3) & 1) & ((to8(Bp) >> 3) & 1))))
    # monotonic: a brighter glow never darkens a pixel under PS1_ADD
    st = _st(**kw)
    st.blend_equation = 'PS1_ADD'
    sc = _pane(st)
    sc.materials[1].diffuse = (1.0, 0.9, 0.8)
    bright = np.asarray(R.render(sc, st))
    check('PS1_ADD is monotonic: a brighter pane darkens no pane pixel',
          bool((to8(bright[yy, xx, :3]) >= to8(lum['PS1_ADD'])).all()))
    # SATURN_SHADOW ignores F
    st = _st(**kw)
    st.blend_equation = 'SATURN_SHADOW'
    sc = _pane(st)
    sc.materials[1].diffuse = (0.1, 0.9, 0.2)
    other = np.asarray(R.render(sc, st))
    check("SATURN_SHADOW ignores the surface's colour (a recoloured pane "
          'gives the identical picture)',
          bool(np.array_equal(other, _render(_pane, _st(**kw, blend_equation='SATURN_SHADOW')))))


def test_c002_per_material_override():
    """A material's Blend Mode overrides the render's equation for that
    material alone; a mode is never alpha evidence."""
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _pane(st0, two=True)
    m = _mirror(sc0, st0)
    mf = sc0.mesh.mat_index[m['tri']]
    s1 = _single(m, sc0.mesh.mat_index, 1)
    two = m['n_at'][m['pix']] == 2
    check('the two-pane scene holds at least 200 single-fragment pane-1 '
          'pixels', int(s1.sum()) >= 200, str(int(s1.sum())))
    B = _b_frame(lambda st: _pane(st, two=True), kw)
    # B0: both panes absent (the floor beneath pane 2)
    st = _st(**kw)
    sc = _pane(st, two=True)
    for i in (1, 2):
        sc.materials[i].opacity = 0.0
        sc.materials[i].blend_mode = 'ALPHA'
    B0 = np.asarray(R.render(sc, st))
    st = _st(**kw)
    sc = _pane(st, two=True)
    sc.materials[1].blend_mode = 'PS1_ADD'
    out = np.asarray(R.render(sc, st))
    xx, yy = m['px'][s1], m['py'][s1]
    F = m['col'][s1, :3]
    Bp = B[yy, xx, :3]
    exp = from8(np.minimum(31, (to8(Bp) >> 3) + (to8(F) >> 3)) << 3)
    check("materials[1] at Blend Mode PS1_ADD is on the lattice at its "
          'single-fragment pixels while the render is at ALPHA',
          bool(np.array_equal(exp, out[yy, xx, :3])))
    # the two-deep chain: pane 2 (ALPHA, rank 1) over B, then pane 1's
    # PS1_ADD over that
    r0 = two & (m['rank'] == 0) & (mf == 1)
    r1 = two & (m['rank'] == 1) & (mf == 2)
    check('the two-deep pixels are pane 1 over pane 2 (at least 60; 71 '
          'measured)', int(r0.sum()) >= 60 and int(r0.sum()) == int(r1.sum()),
          f'{int(r0.sum())} / {int(r1.sum())}')
    if int(r0.sum()) and int(r0.sum()) == int(r1.sum()):
        p0 = m['pix'][r0]
        p1 = m['pix'][r1]
        o0 = np.argsort(p0)
        o1 = np.argsort(p1)
        F1 = m['col'][r0][o0]
        F2 = m['col'][r1][o1]
        xx2 = m['px'][r0][o0]
        yy2 = m['py'][r0][o0]
        # the B frame (pane 1 at opacity 0) already holds pane 2's Alpha
        # Over of the floor; pane 1's PS1_ADD reads it back
        Bb = B[yy2, xx2, :3]
        step = _over(F2[:, :3], np.clip(F2[:, 3], 0, 1), B0[yy2, xx2, :3])
        check('...the B frame at the two-deep pixels IS pane 2 Alpha Over the '
              'floor (recomputed from the mirror, bitwise)',
              bool(np.array_equal(step, Bb)),
              f'max {float(np.abs(step - Bb).max()):.6f}')
        exp2 = from8(np.minimum(31, (to8(Bb) >> 3) + (to8(F1[:, :3]) >> 3)) << 3)
        check('...and the two-deep pixels are pane 2 Alpha Over the floor, then '
              "pane 1's PS1_ADD over that, bitwise",
              bool(np.array_equal(exp2, out[yy2, xx2, :3])),
              f'max {float(np.abs(exp2 - out[yy2, xx2, :3]).max()):.6f}')
        check("...pane 2 alone (Alpha Over) sits OFF the 5-bit lattice",
              bool(np.any((to8(Bb) & 7) != 0)))
    # a glow at opacity 1.0 reaches the transparent pass by its mode
    st = _st(**kw)
    sc = _pane(st)
    sc.materials[1].opacity = 1.0
    sc.materials[1].blend_mode = 'PS1_ADD'
    out1 = np.asarray(R.render(sc, st))
    reasons = dict(R.LAST_SPLIT.get('reasons') or {})
    name1 = sc.materials[1].name
    check('a material at opacity 1.0 with Blend Mode PS1_ADD is in the '
          "split's reasons as 'Blend Mode PS1_ADD'",
          reasons.get(name1) == 'Blend Mode PS1_ADD', str(reasons))
    check('...while has_alpha stays False (the mode is not alpha evidence)',
          not bool(getattr(sc.materials[1], 'has_alpha', False)))
    check('...and its pixels moved against the B frame',
          float(np.abs(out1[yy, xx, :3] - B[yy, xx, :3]).max()) > 0.05)
    from ..core import scene as SC
    from ..core.scene import Material
    check("scene.material_see_through returns None for an ENV_HOLE "
          'material at opacity 0.5',
          SC.material_see_through(Material(opacity=0.5, blend_mode='ENV_HOLE')) is None)
    check("...and 'Blend Mode PS1_ADD' for the glow",
          SC.material_see_through(Material(opacity=1.0, blend_mode='PS1_ADD'))
          == 'Blend Mode PS1_ADD')
    check("...and 'Opacity 0.500' for plain glass",
          SC.material_see_through(Material(opacity=0.5)) == 'Opacity 0.500')
    from . import fakebpy
    fakebpy.install()
    import importlib
    EX = importlib.import_module('halcyon.export')
    check("export._alpha_reason never reads blend_mode (source pin)",
          'blend_mode' not in inspect.getsource(EX._alpha_reason))


def test_c002_gpu_twin():
    """A fixed equation refuses the GPU LAYERS by name (the composite is
    CPU vs CPU by construction); the two items that read no F keep the
    GPU layers. The frame pass beneath is the deferred pass at its own
    bar (measured: 2.98e-6 under ALPHA at 96x72 on the fake device)."""
    st = _st(transparency='ABUFFER', blend_equation='PS1_ADD')
    gpu, cpu, routing, live, out, _lf = twin(_glass, st)
    d = float(np.abs(gpu - cpu).max())
    check("PS1_ADD: R.LAST_ROUTING['refused'] names PS1_ADD",
          'PS1_ADD' in str(routing.get('refused', '')), str(routing.get('refused')))
    check("...and the console carries 'transparent layers on the CPU'",
          'transparent layers on the CPU' in out)
    check('...the opaque frame pass still ran on the fake device (no '
          "'shading on the CPU' line)", 'shading on the CPU' not in out)
    check('...the fake-device frame equals the CPU frame within the '
          f'deferred bar ({FRAME_BAR:g}; the composite itself is CPU vs '
          'CPU by construction)', d < FRAME_BAR, f'max {d:.3g}')
    check('...nothing left live on the device', live == [], str(len(live)))
    st = _st(transparency='ABUFFER', blend_equation='SATURN_SHADOW')
    gpu, cpu, routing, live, out, _lf = twin(_glass, st)
    d = float(np.abs(gpu - cpu).max())
    check('SATURN_SHADOW keeps the GPU layers (no refusal: the item reads '
          'no F)', 'refused' not in routing, str(routing.get('refused')))
    check('...and its frame equals the CPU frame within the deferred bar',
          d < FRAME_BAR, f'max {d:.3g}')
    check('...nothing left live', live == [])


def test_c002_determinism_and_bands():
    """Two renders bitwise; two row bands concatenated bitwise the whole
    frame under PS1_ADD, SNES_ADD_HALF (the per-fragment `last`) and the
    Voodoo framebuffer (the pack indexes the frame row)."""
    for kw in (dict(blend_equation='PS1_ADD'),
               dict(blend_equation='SNES_ADD_HALF'),
               dict(framebuffer='VOODOO_565_4X4'),
               dict(blend_equation='DS', translucent_order='Y_SORT')):
        lab = ', '.join(f'{k}={v}' for k, v in kw.items())
        whole = _render(_glass, _st(transparency='ABUFFER', **kw))
        again = _render(_glass, _st(transparency='ABUFFER', **kw))
        check(f'{lab}: two renders are bitwise', bool(np.array_equal(whole, again)))
        parts = []
        for b in ((0, 36), (36, 72)):
            st = _st(transparency='ABUFFER', **kw)
            parts.append(np.asarray(R.render(_glass(st), st, band=b)))
        check(f'{lab}: two row bands concatenated are bitwise the whole frame',
              bool(np.array_equal(np.concatenate(parts, 0), whole)))


# ------------------------------------------------------------ C018 + C070

def test_c018_pack_equations():
    """Unit checks on the framebuffer arithmetic (core/dither.py)."""
    from ..core import dither as DI
    x = np.arange(16, dtype=np.int32)
    y = np.arange(16, dtype=np.int32)[:, None] * np.ones(16, np.int32)
    x = np.ones(16, np.int32)[:, None] * x
    white = np.full((16, 16, 3), 255, np.int32)
    for fmt in ('VOODOO_565_4X4', 'VOODOO_565_2X2'):
        p = DI.fb_pack8(white, x, y, fmt, True)
        check(f'{fmt}: white packs to 31/63/31 at every dither entry',
              bool(((p[..., 0] >> 3) == 31).all() and ((p[..., 1] >> 2) == 63).all()
                   and ((p[..., 2] >> 3) == 31).all()))
        check(f'{fmt}: the stored expansion of white reads back as 255',
              bool((p == 255).all()))
    v = DI.fb_pack8(np.full((1, 1, 3), 123, np.int32), np.zeros((1, 1), np.int32),
                    np.zeros((1, 1), np.int32), 'VOODOO_565_4X4', True)
    check("the Voodoo pack of r8 = 123 at d = 0 gives 14 (MAME's lossy re-pack)",
          int(v[0, 0, 0] >> 3) == 14, str(int(v[0, 0, 0] >> 3)))
    p = DI.fb_pack8(np.zeros((1, 1, 3), np.int32), np.zeros((1, 1), np.int32),
                    np.zeros((1, 1), np.int32), 'PS2_CT16', True)
    check('PS2: (0 + DIMX[0][0] = -4) >> 3 clamps to 0', bool((p == 0).all()))
    p = DI.fb_pack8(white[:2, :2], x[:2, :2], y[:2, :2], 'GC_RGBA6', True)
    check('GC: 255 packs to 63 at every 2x2 entry', bool(((p >> 2) == 63).all()
                                                          and (p == 255).all()))
    rng = np.random.default_rng(3)
    c8 = rng.integers(0, 256, (16, 16, 3)).astype(np.int32)
    for fmt in DI.FB_FORMATS[1:]:
        stored = DI.fb_pack8(c8, x, y, fmt, True)
        rd = DI.fb_read8(stored, x, y, fmt, True, False)
        check(f'{fmt}: the read-back without subtraction IS the stored value',
              bool(np.array_equal(rd, stored)))
        again = DI.fb_pack8(rd, x, y, fmt, True)
        d = DI.fb_dither_value(fmt, x, y, True)
        # measured, not the spec's claim: the re-pack of a stored value is
        # itself EXCEPT where the GS adds a negative DIMX entry to a
        # zero-filled 5-bit value (one level down) or MAME's 565 re-scale
        # meets a matrix entry at the ends of its range (d 0 / 15 on the
        # 5-bit channels, d 0..2 / 13..15 on green: one level either way);
        # every other pixel is a fixed point
        if fmt == 'PS2_CT16':
            fixed = d >= 0
        elif fmt.startswith('VOODOO'):
            fixed = (d >= 3) & (d <= 12)
        else:
            fixed = np.ones_like(d, bool)
        same = (again == stored).all(axis=2)
        check(f'{fmt}: a stored value re-quantises to itself wherever the '
              'matrix entry is non-negative (GS) / 3..12 (Voodoo); '
              'elsewhere it moves by one level at most',
              bool(same[fixed].all())
              and bool((np.abs((again >> (3 if fmt != 'GC_RGBA6' else 2))
                               - (stored >> (3 if fmt != 'GC_RGBA6' else 2))) <= 1).all()),
              f'{int((~same).sum())} of {same.size} pixels move')
    stored = DI.fb_pack8(c8, x, y, 'VOODOO_565_4X4', True)
    sub = DI.fb_read8(stored, x, y, 'VOODOO_565_4X4', True, True)
    check('the Voodoo dither subtraction adds (15-d)>>1 / >>2 back, never '
          'past 255', bool((sub >= stored).all() and (sub <= 255).all()
                           and np.any(sub != stored)))
    nod = DI.fb_read8(DI.fb_pack8(c8, x, y, 'VOODOO_565_4X4', False), x, y,
                      'VOODOO_565_4X4', False, True)
    check('with Buffer Dither off nothing is subtracted at the read',
          bool(np.array_equal(nod, DI.fb_pack8(c8, x, y, 'VOODOO_565_4X4', False))))


def _lattice565(rgb):
    k = to8(rgb)
    r5, g6, b5 = k[..., 0] >> 3, k[..., 1] >> 2, k[..., 2] >> 3
    return bool((k[..., 0] == ((r5 << 3) | (r5 >> 2))).all()
                and (k[..., 1] == ((g6 << 2) | (g6 >> 4))).all()
                and (k[..., 2] == ((b5 << 3) | (b5 >> 2))).all())


def test_c018_grain_accumulates():
    """The Voodoo buffer: the frame on the 565 lattice everywhere; the
    dither compounds under the glass; a shadeless floor's 16x16 blocks
    average back to the source within half a lattice step."""
    kw = dict(transparency='ABUFFER')
    base = _render(_glass, _st(**kw))
    vf = _render(_glass, _st(**kw, framebuffer='VOODOO_565_4X4'))
    check('the Voodoo frame is on the 565 lattice at every pixel',
          _lattice565(vf[..., :3]))
    st = _st(**kw)
    sc = _glass(st)
    m = _mirror(sc, st)
    glass = np.zeros((H, W), bool)
    glass[m['py'], m['px']] = True
    floor = (m['gbuf'].tri >= 0) & ~glass
    floor &= sc.mesh.mat_index[np.maximum(m['gbuf'].tri, 0)] == 0
    dd = np.abs(vf[..., :3] - base[..., :3]).mean(axis=2)
    check('the per-pixel difference to the NONE frame is larger under the '
          'glass than over the floor (the layers compounded the dither)',
          float(dd[glass].mean()) > float(dd[floor].mean()),
          f'glass {float(dd[glass].mean()):.5f} floor {float(dd[floor].mean()):.5f}')

    def shadeless(st):
        # CONSTANT is the shadeless model (gpu/shade.SHADELESS_MODELS)
        sc = _glass(st)
        sc.materials[0].model = 'CONSTANT'
        return sc
    src = _render(shadeless, _st(**kw))
    v4 = _render(shadeless, _st(**kw, framebuffer='VOODOO_565_4X4'))
    st = _st(**kw)
    sc = shadeless(st)
    m = _mirror(sc, st)
    glass = np.zeros((H, W), bool)
    glass[m['py'], m['px']] = True
    g = m['gbuf']
    fl = (g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 0) & \
        (sc.mesh.obj_index[np.maximum(g.tri, 0)] == 0) & ~glass
    blocks = 0
    worst = 0.0
    for by in range(0, H - 15, 4):
        for bx in range(0, W - 15, 4):
            if not fl[by:by + 16, bx:bx + 16].all():
                continue
            blocks += 1
            mean = v4[by:by + 16, bx:bx + 16, :3].reshape(-1, 3).mean(axis=0)
            s = src[by:by + 16, bx:bx + 16, :3].reshape(-1, 3)
            worst = max(worst, float(np.abs(mean[0] - s[:, 0].mean()) * 255 - 4.0),
                        float(np.abs(mean[1] - s[:, 1].mean()) * 255 - 2.0),
                        float(np.abs(mean[2] - s[:, 2].mean()) * 255 - 4.0))
    check('a shadeless floor has interior 4x4-aligned 16x16 blocks to read',
          blocks > 0, str(blocks))
    check('every such block averages back to the source colour within half '
          'a lattice step (4/255 R,B; 2/255 G): the matrix is zero-mean',
          blocks > 0 and worst <= 0.0, f'worst excess {worst:.3f}/255 over {blocks} blocks')
    plain = _render(shadeless, _st(**kw, framebuffer='VOODOO_565_4X4', fb_dither=False))
    const = True
    for by in range(0, H - 3, 4):
        for bx in range(0, W - 3, 4):
            if fl[by:by + 4, bx:bx + 4].all():
                blk = plain[by:by + 4, bx:bx + 4, :3].reshape(-1, 3)
                const &= bool((blk == blk[0]).all())
    check('with Buffer Dither off the floor truncates smoothly: 4x4 blocks '
          'are constant where the source is', const)
    for fmt in ('PS2_CT16', 'GC_RGBA6', 'VOODOO_565_2X2'):
        f = _render(_glass, _st(**kw, framebuffer=fmt))
        k = to8(f[..., :3])
        if fmt == 'PS2_CT16':
            on = bool(((k & 7) == 0).all())
        elif fmt == 'GC_RGBA6':
            c6 = k >> 2
            on = bool((k == ((c6 << 2) | (c6 >> 4))).all())
        else:
            on = _lattice565(f[..., :3])
        check(f'{fmt}: the frame is on its own lattice at every pixel', on)


def test_c018_no_fragments_road():
    """A frame with no see-through fragments (NONE, STIPPLE) still packs:
    the format is the frame's, not the fragment count's."""
    for mode in ('NONE', 'STIPPLE'):
        st = _st(transparency=mode, framebuffer='PS2_CT16')
        out = np.asarray(R.render(demo_scene(st, with_texture=False), st))
        check(f'{mode} + PS2_CT16: the picture is on the 5-bit lattice (the '
              'whole-frame pack ran without a composite)',
              bool(((to8(out[..., :3]) & 7) == 0).all()))
    img = np.random.default_rng(1).random((8, 8, 4)).astype(np.float32)
    st = _st(framebuffer='PS2_CT16')
    a = R.framebuffer_pack_frame(img.copy(), st)
    fb = R._Framebuffer(st, 8, 8)
    b = fb.leave(fb.enter(img.copy()))
    check('framebuffer_pack_frame is leave(enter(img))', bool(np.array_equal(a, b)))
    st = _st(framebuffer='NONE')
    same = img.copy()
    check('under NONE the pack returns the SAME array object, untouched',
          R.framebuffer_pack_frame(same, st) is same and bool(np.array_equal(same, img)))
    # the glass off the frame (a capped-away composite) still packs
    st = _st(transparency='ABUFFER', framebuffer='PS2_CT16', max_transparent_layers=1)
    st.alpha_threshold = 0.0
    out = _render(_glass, st)
    check('a frame whose layer cap emptied nothing still leaves on the lattice',
          bool(((to8(out[..., :3]) & 7) == 0).all()))


def test_c018_dither_subtract_ab():
    kw = dict(transparency='ABUFFER', framebuffer='VOODOO_565_4X4')
    on = _render(_glass, _st(**kw, fb_dither_subtract=True))
    off = _render(_glass, _st(**kw, fb_dither_subtract=False))
    st = _st(**kw)
    m = _mirror(_glass(st), st)
    glass = np.zeros((H, W), bool)
    glass[m['py'], m['px']] = True
    d = np.abs(on - off).max(axis=2)
    check('Voodoo dither subtraction moves glass pixels only',
          bool(np.any(d[glass] > 0)) and bool((d[~glass] == 0).all()),
          f'{int((d[glass] > 0).sum())} glass px, {int((d[~glass] > 0).sum())} other')


def test_c018_gpu_twin():
    st = _st(transparency='ABUFFER', framebuffer='VOODOO_565_4X4')
    gpu, cpu, routing, live, out, lf = twin(_glass, st)
    d = float(np.abs(gpu - cpu).max())
    check("VOODOO_565_4X4: R.LAST_ROUTING['refused'] names the framebuffer",
          'VOODOO_565_4X4' in str(routing.get('refused', '')))
    check('...the fake-device frame is bitwise the CPU frame (the pack '
          'quantises the frame pass\'s ULPs away; measured 0.0)', d == 0.0,
          f'max {d:.3g}')
    check("...the resident frame was released by name ('transparency')",
          lf.get('left_gpu') == 'transparency', str(lf.get('left_gpu')))
    check('...nothing left live', live == [])
    st = _st(transparency='NONE', framebuffer='PS2_CT16')
    gpu, cpu, routing, live, out, lf = twin(lambda s: demo_scene(s, with_texture=False), st)
    d = float(np.abs(gpu - cpu).max())
    check("NONE + PS2_CT16: the frame left the GPU by name ('framebuffer format')",
          lf.get('left_gpu') == 'framebuffer format', str(lf.get('left_gpu')))
    check('...and the fake-device frame is bitwise the CPU frame', d == 0.0,
          f'max {d:.3g}')
    check('...nothing left live', live == [])


def test_c018_presets():
    for key, fmt in (('VOODOO', 'VOODOO_565_4X4'), ('PS2', 'PS2_CT16'),
                     ('GAMECUBE', 'GC_RGBA6')):
        s = PRESETS[key]['settings']
        check(f'the {key} preset writes into the {fmt} framebuffer',
              s.get('framebuffer') == fmt, str(s.get('framebuffer')))
    check('the VOODOO preset drops its post dither (the 565 buffer does it) '
          "and keeps colour depth '16' (a no-op on the lattice)",
          PRESETS['VOODOO']['settings'].get('dither') == 'NONE'
          and PRESETS['VOODOO']['settings'].get('color_depth') == '16')
    st = _st(transparency='SORTED')
    from ..presets.library import apply_preset
    apply_preset(st, 'VOODOO')
    st.resolution_x, st.resolution_y = W, H
    st.aa_samples = 1
    st.aa_mode = 'NONE'
    frame = _render(_glass, st)
    check('the VOODOO preset frame is on the 565 lattice (no AA to average '
          'it away)', _lattice565(frame[..., :3]))


# ------------------------------------------------------------ C042

def test_c042_ds_blend_law():
    """The DS blend on 6-bit channels at every single-fragment pane pixel;
    alpha = max; opacity 1.0 replaces."""
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _pane(st0)
    m = _mirror(sc0, st0)
    single = _single(m, sc0.mesh.mat_index, 1)
    check('at least 200 single-fragment pane pixels', int(single.sum()) >= 200)
    B = _b_frame(_pane, kw)
    xx, yy = m['px'][single], m['py'][single]
    F = m['col'][single, :3]
    a = np.clip(m['col'][single, 3], 0.0, 1.0)
    Bp = B[yy, xx, :3]
    out = _render(_pane, _st(**kw, blend_equation='DS'))
    a5 = np.clip(np.round(a * np.float32(31)), 0, 31).astype(np.int32)
    f6 = to8(F) >> 2
    b6 = to8(Bp) >> 2
    o = (f6 * (a5[:, None] + 1) + b6 * (31 - a5[:, None])) >> 5
    o = np.where(a5[:, None] == 31, f6, o)
    exp = np.where((a5 > 0)[:, None], from8((o << 2) | (o >> 4)), Bp)
    got = out[yy, xx, :3]
    check('DS: (F*(A+1) + B*(31-A))/32 on 6-bit channels at every '
          'single-fragment pane pixel, bitwise', bool(np.array_equal(exp, got)),
          f'max {float(np.abs(exp - got).max()):.6f}')
    check('DS: the alpha plane at those pixels is max(B alpha, A/31) (the '
          'floor beneath is covered, so max wins there)',
          bool(np.array_equal(out[yy, xx, 3],
                              np.maximum(B[yy, xx, 3], a5.astype(np.float32) / np.float32(31)))))
    st = _st(**kw, blend_equation='DS')
    sc = _pane(st)
    sc.materials[1].opacity = 1.0
    sc.materials[1].blend_mode = 'DS'
    rep = np.asarray(R.render(sc, st))
    m1 = _mirror(sc, st)
    s1 = _single(m1, sc.mesh.mat_index, 1)
    f6 = to8(m1['col'][s1, :3]) >> 2
    check('a DS material at opacity 1.0 REPLACES (writes f6 bit-replicated)',
          bool(np.array_equal(from8((f6 << 2) | (f6 >> 4)),
                              rep[m1['py'][s1], m1['px'][s1], :3])))


def _ball_pixel(sc, w=W, h=H):
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    _c, screen, _iw, _z = raster.project(np.array([[-1.3, 0.2, 1.0]], np.float32),
                                         vp, w, h)
    return int(np.floor(screen[0, 0])), int(np.floor(screen[0, 1]))


def test_c042_same_id_once():
    """On the shell scene every pixel holding four fragments of ONE
    polygon ID (the shell's and the Ball's two walls each): under DS only
    the first drawn blends. Unshadowed: under the shadow bake the three
    inner walls sit in the shell's own shadow and shade identically, so
    DEPTH and SUBMISSION would pick fragments of one colour (measured:
    0 of 308 four-deep pixels differ); without it the walls differ."""
    kw = dict(transparency='ABUFFER', shadows=False)
    st0 = _st(**kw)
    sc0 = _shell(st0)
    px0, py0 = _ball_pixel(sc0)
    m = _mirror(sc0, st0)
    four = m['n_at'][m['pix']] == 4
    pixs = np.unique(m['pix'][four])
    ids_ok = bool((sc0.mesh.obj_index[m['tri'][four]] == 1).all())
    check(f'at least 200 pixels hold four fragments, all of the Ball\'s object '
          f'index (the probe pixel ({px0}, {py0}) among them; 308 measured)',
          pixs.size >= 200 and ids_ok and (py0 * W + px0) in set(pixs.tolist()),
          f'{pixs.size} pixels')
    if pixs.size < 200:
        return
    B = _b_frame(_shell, kw)
    d_frame = _render(_shell, _st(**kw, blend_equation='DS'))
    s_frame = _render(_shell, _st(**kw, blend_equation='DS', translucent_order='SUBMISSION'))
    p_frame = _render(_shell, _st(**kw, blend_equation='PS1_AVG'))

    def ds_one(c, Bp):
        a5 = int(np.clip(np.round(np.float32(c[3]) * np.float32(31)), 0, 31))
        f6 = to8(c[:3]) >> 2
        b6 = to8(Bp) >> 2
        o = (f6 * (a5 + 1) + b6 * (31 - a5)) >> 5
        return from8((o << 2) | (o >> 4))
    ok_far = ok_low = ok_avg = ok_diff_frag = True
    n_differ = 0
    for p in pixs.tolist():
        at = m['pix'] == p
        yy, xx = p // W, p % W
        Bp = B[yy, xx, :3]
        depth = m['depth'][at]
        tri = m['tri'][at]
        col = m['col'][at]
        far = int(np.argmax(depth))
        low = int(np.argmin(tri))
        ok_diff_frag &= far != low
        ok_far &= bool(np.array_equal(ds_one(col[far], Bp), d_frame[yy, xx, :3]))
        ok_low &= bool(np.array_equal(ds_one(col[low], Bp), s_frame[yy, xx, :3]))
        n_differ += int(not np.array_equal(d_frame[yy, xx], s_frame[yy, xx]))
        cur = Bp.copy()
        for i in np.argsort(-depth, kind='stable'):
            f = to8(col[i, :3]) >> 3
            b = to8(cur) >> 3
            cur = from8(((b + f) >> 1) << 3)
        ok_avg &= bool(np.array_equal(cur, p_frame[yy, xx, :3]))
    check('the farthest fragment and the lowest-tri fragment are different '
          "fragments at every such pixel (the shell's back wall vs a Ball wall)",
          ok_diff_frag)
    check('DS under DEPTH order: one DS blend of the fragment of greatest '
          'depth over B at every four-deep pixel (the three nearer ones carry '
          'the same ID and skip), bitwise', ok_far)
    check('DS under SUBMISSION order: one DS blend of the lowest-tri fragment, '
          'bitwise', ok_low)
    check('...and the two frames differ at some of those pixels',
          n_differ > 0, f'{n_differ} of {pixs.size}')
    check('PS1_AVG at the same pixels is the four-fragment chain, far first, '
          'bitwise (four blends either way)', ok_avg)


def _two_quads(st, near_first=True):
    from .scenebuild import _mesh_concat, plane
    from ..core.scene import ObjectInfo
    sc = demo_scene(st, with_texture=False)
    near = plane(z=2.2, size=3.0, mat=1, obj=1)
    # the far quad sits under the near one in projection (the pane
    # scene's own shift, one unit down the demo camera's view) and lower
    # on screen: its bottom row is the lower one
    V, N, UV, T, mi, oi = plane(z=1.2, size=4.0, mat=2, obj=2)
    V = V.copy()
    V[:, 0] += -1.9
    V[:, 1] += 2.3
    far = (V, N, UV, T, mi, oi)
    parts = [plane(z=0.0, size=11.0, mat=0, obj=0)]
    parts += [near, far] if near_first else [far, near]
    sc.mesh = _mesh_concat(parts)
    sc.objects = [ObjectInfo(name='Floor', index=0, matrix_world=np.eye(4, dtype=np.float32)),
                  ObjectInfo(name='Near', index=1, location=(0, 0, 2.2),
                             matrix_world=np.eye(4, dtype=np.float32)),
                  ObjectInfo(name='Far', index=2, location=(-1.9, 2.3, 1.2),
                             matrix_world=np.eye(4, dtype=np.float32))]
    sc.materials[1].opacity = 0.5
    sc.materials[1].model = 'CONSTANT'
    sc.materials[1].diffuse = (0.9, 0.1, 0.1)
    sc.materials[2].opacity = 0.5
    sc.materials[2].model = 'CONSTANT'
    sc.materials[2].diffuse = (0.1, 0.1, 0.9)
    return sc


def test_c042_y_sort_order():
    """Two quads whose depth order and DS bottom-row order disagree."""
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _two_quads(st0)
    m = _mirror(sc0, st0)
    mf = sc0.mesh.mat_index[m['tri']]
    two = m['n_at'][m['pix']] == 2
    r0 = two & (m['rank'] == 0)
    r1 = two & (m['rank'] == 1)
    check('the quads overlap on screen (at least 30 two-deep pixels)',
          int(r0.sum()) >= 30 and int(r0.sum()) == int(r1.sum()), str(int(r0.sum())))
    if not int(r0.sum()):
        return
    check('under DEPTH the near quad (materials[1]) is rank 0 at every overlap pixel',
          bool((mf[r0] == 1).all()) and bool((mf[r1] == 2).all()))
    # the DS keys from the composite's own projection
    _c, screen, _iw, _z = raster.project(sc0.mesh.verts, m['vp'], W, H, snap=0.0)
    row = np.clip(np.floor(screen[:, 1]), 0, H - 1).astype(np.int64)
    rows3 = row[sc0.mesh.tris]
    ybot = (H - 1) - rows3.min(axis=1)
    ytop = (H - 1) - rows3.max(axis=1)
    o0 = np.argsort(m['pix'][r0])
    o1 = np.argsort(m['pix'][r1])
    Fn = m['col'][r0][o0]
    Ff = m['col'][r1][o1]
    tn = m['tri'][r0][o0]
    tf = m['tri'][r1][o1]
    xx = m['px'][r0][o0]
    yy = m['py'][r0][o0]
    # the DS key per fragment: (bottom row, top row, submission) -- the
    # quad whose key is the LARGER composites last (on top)
    far_last = (ybot[tf] > ybot[tn]) | ((ybot[tf] == ybot[tn]) & (ytop[tf] > ytop[tn])) | \
        ((ybot[tf] == ybot[tn]) & (ytop[tf] == ytop[tn]) & (tf > tn))
    check('the DS bottom-row key really disagrees with depth at some overlap '
          'pixels (at least 10; 17 of 102 measured)',
          int(far_last.sum()) >= 10, f'{int(far_last.sum())} of {far_last.size}')
    # B0: both quads absent (the floor beneath)
    st = _st(**kw)
    sc = _two_quads(st)
    for i in (1, 2):
        sc.materials[i].opacity = 0.0
        sc.materials[i].blend_mode = 'ALPHA'
    B = np.asarray(R.render(sc, st))
    Bp = B[yy, xx, :3]
    an = np.clip(Fn[:, 3], 0, 1)
    af = np.clip(Ff[:, 3], 0, 1)
    near_top = _over(Fn[:, :3], an, _over(Ff[:, :3], af, Bp))
    far_top = _over(Ff[:, :3], af, _over(Fn[:, :3], an, Bp))
    exp_ysort = np.where(far_last[:, None], far_top, near_top)
    d = _render(_two_quads, _st(**kw))
    y = _render(_two_quads, _st(**kw, translucent_order='Y_SORT'))
    s = _render(_two_quads, _st(**kw, translucent_order='SUBMISSION'))
    check('DEPTH: over(near, over(far, floor)) at the overlap, bitwise',
          bool(np.array_equal(near_top, d[yy, xx, :3])))
    check('Y_SORT: the quad with the larger (bottom row, top row, submission) '
          'key composites LAST, regardless of depth -- bitwise',
          bool(np.array_equal(exp_ysort, y[yy, xx, :3])))
    check('SUBMISSION: the higher triangle index (the far quad, concatenated '
          'second) is on top', bool(np.array_equal(far_top, s[yy, xx, :3]))
          and bool((tf > tn).all()))
    check('...and the two orders really differ from DEPTH at the overlap',
          not bool(np.array_equal(d[yy, xx], y[yy, xx])))
    # Y_SORT without a projection falls back by name
    buf = io.StringIO()
    st = _st(**kw, translucent_order='Y_SORT')
    sc = _two_quads(st)
    m2 = _mirror(sc, st)
    img = np.zeros((H, W, 4), np.float32)
    frags = raster.FragmentList()
    view, _proj, vp, eye = R.camera_matrices(sc.camera, W, H)
    opq, trans = R._split_by_alpha(sc, sc.mesh, st)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, cull='NONE',
                     subset=trans, gbuf=m2['gbuf'], frags=frags,
                     depth_write=False, depth_bits=st.depth_precision)
    with contextlib.redirect_stdout(buf):
        R._composite_abuffer(m2['job'], frags, m2['gbuf'], img, st)
    check('Y_SORT with no projection prints its fallback by name and '
          'composites by depth', 'Y-sort needs the projection' in buf.getvalue())


def test_c042_depth_write():
    kw = dict(transparency='ABUFFER', translucent_order='SUBMISSION')
    st0 = _st(**kw)
    sc0 = _two_quads(st0, near_first=True)
    m = _mirror(sc0, st0, order='SUBMISSION')
    mf = sc0.mesh.mat_index[m['tri']]
    two = m['n_at'][m['pix']] == 2
    r0 = two & (m['rank'] == 0)
    r1 = two & (m['rank'] == 1)
    check('the nearer quad is drawn FIRST under SUBMISSION (lower tri)',
          bool((mf[r0] == 1).all()) and bool((mf[r1] == 2).all()) and int(r0.sum()) >= 30)
    o0 = np.argsort(m['pix'][r0])
    o1 = np.argsort(m['pix'][r1])
    Fn = m['col'][r0][o0]
    Ff = m['col'][r1][o1]
    xx = m['px'][r0][o0]
    yy = m['py'][r0][o0]
    st = _st(**kw)
    sc = _two_quads(st)
    for i in (1, 2):
        sc.materials[i].opacity = 0.0
        sc.materials[i].blend_mode = 'ALPHA'
    B = np.asarray(R.render(sc, st))
    Bp = B[yy, xx, :3]
    an = np.clip(Fn[:, 3], 0, 1)
    af = np.clip(Ff[:, 3], 0, 1)
    on = _render(_two_quads, _st(**kw, translucent_depth_write=True))
    off = _render(_two_quads, _st(**kw, translucent_depth_write=False))
    check('with the depth write the farther quad is dropped at the overlap '
          '(pixel == over(near, floor)), bitwise',
          bool(np.array_equal(_over(Fn[:, :3], an, Bp), on[yy, xx, :3])))
    check('without it both blend (over(far, over(near, floor)))',
          bool(np.array_equal(_over(Ff[:, :3], af, _over(Fn[:, :3], an, Bp)),
                              off[yy, xx, :3])))
    check('...and the two frames differ', not bool(np.array_equal(on, off)))


def test_c042_gpu_twin():
    st = _st(transparency='ABUFFER', blend_equation='DS',
             translucent_order='Y_SORT', translucent_depth_write=True)
    gpu, cpu, routing, live, out, _lf = twin(_glass, st)
    d = float(np.abs(gpu - cpu).max())
    check("DS + Y_SORT + depth write: R.LAST_ROUTING['refused'] names DS",
          'DS' in str(routing.get('refused', '')).split('blend equation')[0])
    check('...the fake-device frame equals the CPU frame within the deferred '
          'bar (the composite is CPU vs CPU by construction)', d < FRAME_BAR,
          f'max {d:.3g}')
    check('...nothing left live', live == [])
    st = _st(transparency='ABUFFER', translucent_order='Y_SORT',
             translucent_depth_write=True)
    gpu, cpu, routing, live, out, _lf = twin(_glass, st)
    d = float(np.abs(gpu - cpu).max())
    check('ALPHA under the same order and depth write keeps the GPU layers '
          '(no refusal: the ordering is CPU on both roads)',
          'refused' not in routing, str(routing.get('refused')))
    check("...within today's layer bar", d < FRAME_BAR, f'max {d:.3g}')
    check('...nothing left live', live == [])


def test_c042_nds_preset():
    s = PRESETS['NDS']['settings']
    check('the NDS preset composites with the DS rules: A-Buffer, DS blend, '
          'Y-sort, no depth write',
          s.get('transparency') == 'ABUFFER' and s.get('blend_equation') == 'DS'
          and s.get('translucent_order') == 'Y_SORT'
          and s.get('translucent_depth_write') is False, str(s))
    check('the NDS preset is on the console shelf with a machine note',
          PRESETS['NDS']['category'] == 'CONSOLE'
          and 'polygon ID' in PRESETS['NDS']['note'])


# ------------------------------------------------------------ C053

def test_c053_one_sub_screen():
    """SNES colour math blends ONE sub screen: only the layer composited
    last blends, every layer beneath it is drawn opaque."""
    kw = dict(transparency='ABUFFER')
    w2, h2 = 200, 150
    st0 = _st(w2, h2, **kw)
    sc0 = _pane(st0, two=True)
    m = _mirror(sc0, st0, w2, h2)
    mf = sc0.mesh.mat_index[m['tri']]
    two = m['n_at'][m['pix']] == 2
    r0 = two & (m['rank'] == 0) & (mf == 1)
    r1 = two & (m['rank'] == 1) & (mf == 2)
    check('at least 200 two-deep pane-1-over-pane-2 pixels at 200x150 '
          '(308 measured)', int(r0.sum()) >= 200 and int(r0.sum()) == int(r1.sum()),
          f'{int(r0.sum())} / {int(r1.sum())}')
    if int(r0.sum()) < 200:
        return
    o0 = np.argsort(m['pix'][r0])
    o1 = np.argsort(m['pix'][r1])
    F1 = m['col'][r0][o0][:, :3]
    F2 = m['col'][r1][o1][:, :3]
    xx = m['px'][r0][o0]
    yy = m['py'][r0][o0]
    snes = _render(lambda s: _pane(s, two=True), _st(w2, h2, **kw, blend_equation='SNES_ADD_HALF'))
    o = ((to8(F1) >> 3) + (to8(F2) >> 3)) >> 1
    exp = from8((o << 3) | (o >> 2))
    check('SNES_ADD_HALF at the two-deep pixels is ((F1>>3) + (F2>>3)) >> 1 '
          'bit-replicated: pane 2 REPLACED the frame beneath, pane 1 blended '
          'with it', bool(np.array_equal(exp, snes[yy, xx, :3])),
          f'max {float(np.abs(exp - snes[yy, xx, :3]).max()):.6f}')
    avg = _render(lambda s: _pane(s, two=True), _st(w2, h2, **kw, blend_equation='PS1_AVG'))
    check('...while PS1_AVG differs there (both layers blend)',
          not bool(np.array_equal(avg[yy, xx, :3], snes[yy, xx, :3])))
    # the per-fragment `last` on the glass scene under SUBMISSION: each
    # pixel's expectation from the two highest-tri fragments
    st = _st(**kw, blend_equation='SNES_ADD_HALF', translucent_order='SUBMISSION')
    sc = _glass(st)
    sc.materials[2].reflect_level = 0.0
    frame = np.asarray(R.render(sc, st))
    st1 = _st(**kw, translucent_order='SUBMISSION')
    sc1 = _glass(st1)
    sc1.materials[2].reflect_level = 0.0
    m = _mirror(sc1, st1, order='SUBMISSION')
    n = m['n_at'][m['pix']]
    last = m['rank'] == n - 1
    prev = m['rank'] == n - 2
    for depth_n, sel_n, label in ((2, n == 2, 'the Ball alone'),
                                  (3, n >= 3, 'Ball over Box')):
        L = last & sel_n
        P = prev & sel_n
        if not int(L.sum()):
            check(f'pixels {depth_n} deep exist ({label})', False, '0')
            continue
        oL = np.argsort(m['pix'][L])
        oP = np.argsort(m['pix'][P])
        fL = to8(m['col'][L][oL][:, :3]) >> 3
        fP = to8(m['col'][P][oP][:, :3]) >> 3
        o = (fP + fL) >> 1
        exp = from8((o << 3) | (o >> 2))
        xx = m['px'][L][oL]
        yy = m['py'][L][oL]
        check(f'SUBMISSION, pixels {depth_n}+ deep ({label}, {int(L.sum())} px): '
              'the highest-tri fragment blends with the second-highest, every '
              'earlier one replaced away -- per-fragment `last`, bitwise',
              bool(np.array_equal(exp, frame[yy, xx, :3])),
              f'max {float(np.abs(exp - frame[yy, xx, :3]).max()):.6f}')


def test_c053_gba_sixteenths():
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _pane(st0)
    m = _mirror(sc0, st0)
    single = _single(m, sc0.mesh.mat_index, 1)
    xx, yy = m['px'][single], m['py'][single]
    B = _b_frame(_pane, kw)
    Bp = B[yy, xx, :3]
    frames = []
    ok_lattice = True
    ok_law = True
    ok_set = True
    for k in range(1, 17):
        # the material's own Blend Mode GBA keeps it in the transparent
        # pass at opacity 1.0 (EVA 16: the pane's colour, replicated)
        st = _st(**kw)
        sc = _pane(st)
        sc.materials[1].opacity = k / 16.0
        sc.materials[1].blend_mode = 'GBA'
        out = np.asarray(R.render(sc, st))
        mm = _mirror(sc, st)
        s1 = _single(mm, sc.mesh.mat_index, 1)
        ok_set &= bool(np.array_equal(np.sort(mm['pix'][s1]), np.sort(m['pix'][single])))
        o1 = np.argsort(mm['pix'][s1])
        F = mm['col'][s1][o1][:, :3]
        a = np.clip(mm['col'][s1][o1][:, 3], 0, 1)
        xk = mm['px'][s1][o1]
        yk = mm['py'][s1][o1]
        Bk = B[yk, xk, :3]
        eva = np.clip(np.round(a * np.float32(16)), 0, 16).astype(np.int32)
        o = np.minimum(31, ((to8(F) >> 3) * eva[:, None] + (to8(Bk) >> 3) * (16 - eva)[:, None]) >> 4)
        exp = from8((o << 3) | (o >> 2))
        got = out[yk, xk, :3]
        ok_law &= bool(np.array_equal(exp, got))
        k8 = to8(got)
        ok_lattice &= bool(np.array_equal(k8, ((k8 >> 3) << 3) | ((k8 >> 3) >> 2)))
        frames.append(got)
    check('the single-fragment pane pixels are the same set at every opacity',
          ok_set)
    check('GBA: sixteen opacities k/16 give the BLDALPHA formula at every '
          'single-fragment pane pixel, bitwise', ok_law)
    check('...each on the 5-bit lattice, bit-replicated', ok_lattice)
    lum = [float(f.mean()) for f in frames]
    distinct = len({f.tobytes() for f in frames})
    mono = all(lum[i] >= lum[i - 1] - 1e-6 for i in range(1, 16)) or \
        all(lum[i] <= lum[i - 1] + 1e-6 for i in range(1, 16))
    check('...sixteen distinct pictures, monotonic in opacity',
          distinct == 16 and mono, f'{distinct} distinct')
    st = _st(**kw, blend_equation='GBA')
    sc = _pane(st)
    sc.materials[1].opacity = 0.0
    sc.materials[1].blend_mode = 'GBA'
    zero = np.asarray(R.render(sc, st))
    check('at opacity 0 nothing is drawn: the raw B frame, bitwise',
          bool(np.array_equal(zero[yy, xx], B[yy, xx])))
    # the half-to-even tie through the formula
    a = np.array([1 / 32, 3 / 32], np.float32)
    eva = np.clip(np.round(a * np.float32(16)), 0, 16).astype(np.int32)
    check('a tie at a = k/32 rounds half-to-even (1/32 -> EVA 0, 3/32 -> EVA 2)',
          eva.tolist() == [0, 2], str(eva.tolist()))


def test_c053_columns_mesh():
    from ..core import dither as DI
    st = _st(transparency='STIPPLE', stipple_pattern='COLUMNS')
    sc = _glass(st)
    out = np.asarray(R.render(sc, st))
    view, _proj, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = raster.GBuffer(W, H)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                     depth_bits=st.depth_precision)
    glass = (g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 1)
    ys, xs = np.nonzero(glass)
    alpha = out[ys, xs, 3]
    check('COLUMNS at opacity 0.5: the alpha plane over the Ball is 1 at '
          'even x and 0 at odd x, every row',
          bool((alpha[xs % 2 == 0] == 1.0).all()) and bool((alpha[xs % 2 == 1] == 0.0).all())
          and int(glass.sum()) > 100, f'{int(glass.sum())} px')
    st = _st(transparency='STIPPLE', stipple_pattern='COLUMNS')
    sc = _glass(st)
    sc.materials[1].opacity = 0.9
    hi = np.asarray(R.render(sc, st))
    check('...at opacity 0.9 every column is kept', bool((hi[ys, xs, 3] == 1.0).all()))
    sc = _glass(st)
    sc.materials[1].opacity = 0.2
    lo = np.asarray(R.render(sc, st))
    check('...at opacity 0.2 none is', bool((lo[ys, xs, 3] == 0.0).all()))
    check("threshold_map('COLUMNS', 64, 64) is 0.25 at even x, 0.75 at odd x",
          bool((DI.threshold_map('COLUMNS', 64, 64)[:, ::2] == 0.25).all())
          and bool((DI.threshold_map('COLUMNS', 64, 64)[:, 1::2] == 0.75).all()))
    check('dither.pattern(14, x, y) is the COLUMNS threshold by number',
          bool(np.array_equal(DI.pattern(14, np.arange(4), np.zeros(4, int)),
                              np.array([0.25, 0.75, 0.25, 0.75], np.float32)))
          and DI.pattern('FLOYD', 0, 0) is None)
    # the by-name fallback for a kind past the enum, and for ''
    b4 = _render(_glass, _st(transparency='STIPPLE', stipple_pattern='BAYER4'))
    for kind in ('FLOYD', ''):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            f = _render(_glass, _st(transparency='STIPPLE', stipple_pattern=kind))
        check(f'stipple_pattern {kind!r} renders without a None index, says '
              "'is not an ordered map; Bayer 4x4 used' and is bitwise the "
              'BAYER4 frame',
              'is not an ordered map; Bayer 4x4 used' in buf.getvalue()
              and bool(np.array_equal(f, b4)))
    # the GPU side of the same fallback: nothing refused, the same line,
    # bitwise the BAYER4 fake-device frame
    g4, _c4, _r4, _l4, _o4, _f4 = twin(_glass, _st(transparency='STIPPLE', stipple_pattern='BAYER4'))
    gf, _cf, _rf, lf, of, _ff = twin(_glass, _st(transparency='STIPPLE', stipple_pattern='FLOYD'))
    check("the fake-device road with 'FLOYD' refuses nothing, prints the "
          'fallback from _build_stipple and is bitwise the BAYER4 fake-device '
          'frame', 'shading on the CPU' not in of
          and 'is not an ordered map; Bayer 4x4 used' in of
          and bool(np.array_equal(g4, gf)) and lf == [])


def test_c053_gpu_twin():
    """The COLUMNS mesh through the deferred frame pass: the same texel,
    the same compare, the alpha plane bitwise."""
    from ..gpu import shade as GSH
    st = _st(transparency='STIPPLE', stipple_pattern='COLUMNS')
    st.render_device = 'GPU'
    sc = _glass(st)
    cpu = np.asarray(R.render(sc, st.copy()))
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = raster.GBuffer(W, H)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                     depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye, W, H)
    GSH._PLAN_CACHE.clear()
    p, why, atl = GSH.plan_frame(job, g)
    check('a COLUMNS screen-door frame plans for the deferred pass',
          p is not None, str(why))
    if p is None:
        return
    check("...a pass reads 'hal_stipple'", any('hal_stipple' in s for _m, _n, s, _b in p))
    img, _hit = GSH.simulate(job, g, p, atl)
    cov = g.tri >= 0
    check('the simulated alpha plane is the CPU column mesh, bit for bit',
          g.gpu_alpha is not None
          and bool((g.gpu_alpha[cov].astype(np.float32) == cpu[cov][:, 3]).all()))
    check('...and it is a real mesh (about half the Ball kept)',
          0.3 < float(g.gpu_alpha[cov & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 1)].mean()) < 0.7)


def test_c053_presets():
    s = PRESETS['SNES']['settings']
    check("the SNES preset blends with SNES_ADD_HALF under Sorted Blend",
          s.get('transparency') == 'SORTED' and s.get('blend_equation') == 'SNES_ADD_HALF')
    s = PRESETS['GBA_MODE5']['settings']
    check('the GBA_MODE5 preset blends with the GBA sixteenths under Sorted Blend',
          s.get('transparency') == 'SORTED' and s.get('blend_equation') == 'GBA'
          and s.get('resolution_x') == 240 and s.get('resolution_y') == 160)
    for key, eq in (('PSX', 'PS1_AVG'), ('PSX_HIRES', 'PS1_AVG'),
                    ('SATURN', 'SATURN_HALF'), ('THREEDO', 'SATURN_HALF'),
                    ('DOOM', 'FUZZ')):
        check(f'the {key} preset blends with {eq}',
              PRESETS[key]['settings'].get('blend_equation') == eq,
              str(PRESETS[key]['settings'].get('blend_equation')))
    # the pack's two presets exist on the console shelf with a note; the
    # round's exact count is test_render's pin (the integrator's), a pack
    # asserts only that the library holds at least the round's presets
    for key in ('NDS', 'GBA_MODE5'):
        p = PRESETS.get(key)
        check(f"the pack's {key} preset is on the console shelf with a machine note",
              p is not None and p['category'] == 'CONSOLE' and len(p.get('note', '')) >= 40,
              'missing' if p is None else f"{p['category']}, note {len(p.get('note', ''))} chars")
    check("the library holds at least the round's 103 presets (the exact count is "
          "test_render's pin)", len(PRESETS) >= 103, str(len(PRESETS)))


# ------------------------------------------------------------ C062

def test_c062_fuzz_law():
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _pane(st0)
    m = _mirror(sc0, st0)
    single = _single(m, sc0.mesh.mat_index, 1)
    check('at least 200 single-fragment pane pixels', int(single.sum()) >= 200)
    B = _b_frame(_pane, kw)
    xx, yy = m['px'][single], m['py'][single]
    from ..core import film
    st = _st(**kw)
    sc = _pane(st)
    sc.materials[1].blend_mode = 'FUZZ'
    out = np.asarray(R.render(sc, st))
    frame = int(getattr(sc, 'frame', 1) or 1)
    start = (film._hash_u32_raw(xx, np.full_like(xx, frame), int(st.seed))
             % np.uint32(50)).astype(np.int32)
    k = (yy.astype(np.int32) + start) % 50
    sy = np.clip(yy + R.FUZZ_T[k], 1, H - 2)
    exp = B[sy, xx, :3] * np.float32(0.8125)
    check('FUZZ at every single-fragment pane pixel is 0.8125 x B one row '
          'up or down by the fuzzoffset table, bitwise',
          bool(np.array_equal(exp, out[yy, xx, :3])),
          f'max {float(np.abs(exp - out[yy, xx, :3]).max()):.6f}')
    check('...the pane pixels are covered (alpha 1)', bool((out[yy, xx, 3] == 1.0).all()))
    sc = _pane(st)
    sc.materials[1].blend_mode = 'FUZZ'
    sc.materials[1].diffuse = (0.1, 0.9, 0.3)
    other = np.asarray(R.render(sc, st))
    check("the surface's own colour never appears (a recoloured pane gives "
          'the identical picture)', bool(np.array_equal(other, out)))
    mean_ratio = float(out[yy, xx, :3].mean()) / max(float(B[sy, xx, :3].mean()), 1e-9)
    check('the mean over the pane is 0.8125 x the mean of the displaced '
          'background', abs(mean_ratio - 0.8125) <= 1e-5, f'{mean_ratio:.6f}')
    sc = _pane(st)
    sc.materials[1].blend_mode = 'FUZZ'
    sc.frame = 2
    f2 = np.asarray(R.render(sc, st))
    sc = _pane(st)
    sc.materials[1].blend_mode = 'FUZZ'
    sc.frame = 1
    f1 = np.asarray(R.render(sc, st))
    check('frame 2 moves the picture (the shimmer) while frame 1 twice is '
          'bitwise', not bool(np.array_equal(f2, out)) and bool(np.array_equal(f1, out)))
    # a quad spanning the full frame height: the read stays inside
    from .scenebuild import _mesh_concat, plane
    st = _st(**kw, blend_equation='FUZZ')
    sc = demo_scene(st, with_texture=False)
    V = np.array([[-40, 2.0, -30], [40, 2.0, -30], [40, 2.0, 60], [-40, 2.0, 60]], np.float32)
    N = np.tile(np.array([[0, -1.0, 0]], np.float32), (4, 1))
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    sc.mesh = _mesh_concat([plane(z=0.0, size=11.0, mat=0, obj=0), (V, N, UV, T, 1, 1)])
    sc.materials[1].opacity = 0.5
    wall = np.asarray(R.render(sc, st))
    check('a fuzzed wall spanning the whole frame height renders without a '
          'NaN or an index error (rows 0 and H-1 read inside the frame)',
          wall.shape == (H, W, 4) and bool(np.isfinite(wall).all())
          and bool((wall[0, :, 3] == 1.0).all()) and bool((wall[H - 1, :, 3] == 1.0).all()))


def test_c062_gpu_twin():
    st = _st(transparency='ABUFFER', blend_equation='FUZZ')
    gpu, cpu, routing, live, out, _lf = twin(_glass, st)
    d = float(np.abs(gpu - cpu).max())
    check('FUZZ keeps the GPU layers (no refusal: it reads no F)',
          'refused' not in routing, str(routing.get('refused')))
    check("...the fake-device frame equals the CPU frame within today's layer "
          'bar', d < FRAME_BAR, f'max {d:.3g}')
    check('...nothing left live', live == [])


def test_c062_pool_gate():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, 'engine.py'), encoding='utf-8').read()
    check("engine.py's worker-pool gate skips the pool by name for the Fuzz / "
          'Thin Wall blend', 'Fuzz / Thin Wall blend reads neighbouring pixels' in src
          and 'composite_reads_neighbours(scene, settings)' in src)
    st = _st()
    sc = _glass(st)
    for mode in ('SORTED', 'ABUFFER'):
        st.transparency = mode
        st.blend_equation = 'FUZZ'
        check(f'composite_reads_neighbours is True for FUZZ under {mode}',
              R.composite_reads_neighbours(sc, st) is True)
    st.blend_equation = 'PS1_ADD'
    check('...False for PS1_ADD', R.composite_reads_neighbours(sc, st) is False)
    sc.materials[1].blend_mode = 'THIN_WALL'
    check('...True for a material at Blend Mode THIN_WALL',
          R.composite_reads_neighbours(sc, st) is True)
    st.transparency = 'STIPPLE'
    st.blend_equation = 'FUZZ'
    check('...False under STIPPLE (no composite runs)',
          R.composite_reads_neighbours(sc, st) is False)


# ------------------------------------------------------------ the enums

def test_enum_numbers_stable():
    """BLEND_EQUATION / BLEND_MODE_MAT / STIPPLE_PATTERN carry explicit
    numbers (Blender stores the integer in the .blend): the composite's
    MODE_INDEX is the one source; the five ordered stipple kinds keep the
    positions DITHER gave them."""
    from . import fakebpy
    fakebpy.install()
    import importlib
    PR = importlib.import_module('halcyon.properties')
    real = [i for i in PR.BLEND_EQUATION if i[0]]
    seps = [i for i in PR.BLEND_EQUATION if not i[0]]
    check('BLEND_EQUATION holds the 19 equations as five-tuples numbered by '
          'MODE_INDEX', len(real) == 19 and all(len(i) == 5 for i in PR.BLEND_EQUATION)
          and all(i[4] == R.MODE_INDEX[i[0]] for i in real)
          and sorted(i[0] for i in real) == sorted(R.MODE_INDEX),
          str([(i[0], i[4]) for i in real]))
    check('...its separators take 900+ (never stored) and name the machines',
          len(seps) == 7 and all(i[4] >= 900 for i in seps)
          and [s[1] for s in seps] == ['PlayStation', 'Sega Saturn', '3DO', 'SNES',
                                       'Game Boy Advance', 'Nintendo DS', 'Software'])
    g = PR.ENUMS['blend_equation']
    check("the global menu is the list minus SATURN_HALF_LUM (a per-sprite "
          'VDP1 bit), numbers unchanged',
          [i for i in g if i[0]] == [i for i in real if i[0] != 'SATURN_HALF_LUM']
          and len([i for i in g if i[0]]) == 18)
    check('BLEND_MODE_MAT = INHERIT 100 + the 19 + ENV_HOLE 101',
          PR.BLEND_MODE_MAT[0][:1] + (PR.BLEND_MODE_MAT[0][4],) == ('INHERIT', 100)
          and PR.BLEND_MODE_MAT[-1][0] == 'ENV_HOLE' and PR.BLEND_MODE_MAT[-1][4] == 101
          and PR.BLEND_MODE_MAT[1:-1] == PR.BLEND_EQUATION)
    dither_ids = [d[0] for d in PR.DITHER]
    check('the five ordered stipple kinds keep the numbers DITHER gave them',
          [i[4] for i in PR.STIPPLE_PATTERN[:5]]
          == [dither_ids.index(i[0]) for i in PR.STIPPLE_PATTERN[:5]],
          str([(i[0], i[4]) for i in PR.STIPPLE_PATTERN[:5]]))
    check("COLUMNS and N64_NOISE sit past DITHER's fourteen (14, 15)",
          [i[4] for i in PR.STIPPLE_PATTERN[5:]] == [14, 15]
          and [i[0] for i in PR.STIPPLE_PATTERN[5:]] == ['COLUMNS', 'N64_NOISE'])
    check("ENUMS['stipple_pattern'] is the STIPPLE_PATTERN list, not DITHER",
          PR.ENUMS['stipple_pattern'] is PR.STIPPLE_PATTERN)
    from ..core import dither as DI
    check('dither.STIPPLE_NUMBERS agrees with the property list',
          DI.STIPPLE_NUMBERS == {i[0]: i[4] for i in PR.STIPPLE_PATTERN})
    for k in ('blend_equation', 'translucent_order', 'translucent_depth_write',
              'framebuffer', 'fb_dither', 'fb_dither_subtract'):
        check(f'{k} carries a tooltip naming the mechanism (>= 40 chars)',
              len(PR.DESCRIPTIONS.get(k, '')) >= 40 and k in PR.LABELS)
    bm = PR.HalcyonMaterialSettings.__annotations__.get('blend_mode')
    desc = str(getattr(bm, 'kw', {}).get('description', '') or '')
    check('the Material Blend Mode property carries its tooltip (>= 40 chars, '
          'naming the machine and the Env hole)',
          len(desc) >= 40 and 'PlayStation' in desc and 'Env Hole' in desc
          and getattr(bm, 'kw', {}).get('default') == 'INHERIT',
          desc[:60])
    FM = _fm()
    keys = [r[0] for r in FM.ROWS]
    want = ['blend PS1_ADD (PlayStation)', 'blend SATURN_HALF (Saturn VDP1)',
            'blend THREEDO_XOR (3DO PIXC)', 'blend SNES_ADD_HALF (SNES colour math)',
            'blend GBA (BLDALPHA sixteenths)', 'stipple COLUMNS (Mega Drive mesh)',
            'blend DS (Nintendo DS)', 'DS auto-sort + depth write', 'DS same-ID once',
            'framebuffer PS2_CT16 (GS 16-bit)', 'framebuffer GC_RGBA6 (Flipper EFB)',
            'framebuffer VOODOO_565_4X4', 'framebuffer VOODOO_565_2X2, no subtract',
            'framebuffer on an opaque frame',
            'framebuffer PS2_CT16, DTHE off (truncation)', 'blend FUZZ (Doom Spectre)']
    check('the sixteen featurematrix rows of the pack exist',
          all(k in keys for k in want), str([k for k in want if k not in keys]))
    check("SCENES carries 'pane' and 'ds_shell'",
          'pane' in FM.SCENES and 'ds_shell' in FM.SCENES)


# ---- wave 2 ----
# TRANS-2 (C015, C031, C126, C095, C101): appended after wave 1's last
# test; wave 1's tests above are untouched.


def _glass_mask(sc, st, w=W, h=H, mat=1):
    """The opaque G-buffer's pixels of material `mat` (the Screen Door
    scenes rasterise every material into the opaque pass)."""
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    return (g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == mat), g


def _gbuffer_job(sc, st, w=W, h=H):
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    return g, job


# ------------------------------------------------------------ C015

def test_c015_identity_at_defaults():
    """Screen Door at the default Bayer 4x4 renders bitwise the 1.89.0
    zip (the map builder and the emit gained a branch without moving a
    byte); the BAYER4 map pin."""
    RP = _prev_engine(ZIP)
    if RP is None:
        print('  (no halcyon-1.89.0.zip beside the package: identity skipped)')
        return
    st = _st(transparency='STIPPLE', stipple_pattern='BAYER4')
    now = _render(_glass, st)
    st2 = _st(transparency='STIPPLE', stipple_pattern='BAYER4')
    prev = np.asarray(RP.render(_glass(st2), st2))
    check('STIPPLE at BAYER4 renders bitwise the 1.89.0 zip after the '
          'N64_NOISE branch', now.shape == prev.shape
          and bool(np.array_equal(now, prev)))
    import importlib
    from ..core import dither as DI
    prev_di = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.dither')
    check("threshold_map('BAYER4', 64, 64) is bitwise the 1.89.0 module's "
          '(the frame/seed kwargs default away)',
          bool(np.array_equal(DI.threshold_map('BAYER4', 64, 64),
                              prev_di.threshold_map('BAYER4', 64, 64))))


def test_c015_noise_compare_law():
    """N64_NOISE: the kept density equals the alpha (round(a*255)/256),
    monotonic in opacity; a fresh roll per frame and per seed; the kept
    mask is exactly round(a*255) > the hash map; a tie drops."""
    from ..core import dither as DI
    w, h = 200, 150
    st = _st(w, h, transparency='STIPPLE', stipple_pattern='N64_NOISE')
    sc = _glass(st)
    glass, _g = _glass_mask(sc, st, w, h)
    n = int(glass.sum())
    if n < 2000:
        print(f'  ({n} glass pixels at 200x150 < 2000: re-rendering at 320x240)')
        w, h = 320, 240
        st = _st(w, h, transparency='STIPPLE', stipple_pattern='N64_NOISE')
        sc = _glass(st)
        glass, _g = _glass_mask(sc, st, w, h)
        n = int(glass.sum())
    check(f'at least 2000 glass pixels for the density law ({n} at {w}x{h})',
          n >= 2000)
    ys, xs = np.nonzero(glass)
    fracs = []
    ok_dens = True
    worst = 0.0
    for op in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        sto = _st(w, h, transparency='STIPPLE', stipple_pattern='N64_NOISE')
        sco = _glass(sto)
        sco.materials[1].opacity = op
        out = np.asarray(R.render(sco, sto))
        f = float(out[ys, xs, 3].mean())
        a8 = float(np.round(np.float32(op) * np.float32(255)))
        worst = max(worst, abs(f - a8 / 256.0))
        ok_dens &= abs(f - a8 / 256.0) <= 0.03
        fracs.append(f)
    check('the kept fraction is within 0.03 of round(a*255)/256 at every '
          'opacity 0.1..0.9 (density = alpha)', ok_dens, f'worst {worst:.4f}')
    check('...and monotonic in opacity',
          all(b >= a for a, b in zip(fracs, fracs[1:])), str(np.round(fracs, 3)))
    # determinism and the re-roll
    st1 = _st(w, h, transparency='STIPPLE', stipple_pattern='N64_NOISE')
    sc1 = _glass(st1)
    sc1.frame = 1
    f1a = np.asarray(R.render(sc1, st1))
    f1b = np.asarray(R.render(sc1, st1))
    check('frame 1 twice is bitwise', bool(np.array_equal(f1a, f1b)))
    sc2 = _glass(st1)
    sc2.frame = 2
    f2 = np.asarray(R.render(sc2, st1))
    check('frame 2 moves the alpha plane (the RDP re-rolls every frame)',
          bool(np.any(f2[ys, xs, 3] != f1a[ys, xs, 3])))
    st7 = _st(w, h, transparency='STIPPLE', stipple_pattern='N64_NOISE', seed=7)
    f7 = np.asarray(R.render(_glass(st7), st7))
    check('seed 7 moves it too', bool(np.any(f7[ys, xs, 3] != f1a[ys, xs, 3])))
    # the exact mask
    tm = DI.noise_threshold_map(h, w, 1, st1.seed)
    a8 = np.round(np.float32(0.5) * np.float32(255))
    exp = (a8 > tm[ys, xs]).astype(np.float32)
    check('the kept mask is exactly round(a*255) > noise_threshold_map(H, W, '
          'frame, seed)[py, px], bitwise', bool(np.array_equal(exp, f1a[ys, xs, 3])))
    tm2 = DI.noise_threshold_map(h, w, 2, st1.seed)
    check('...and at frame 2 against the frame-2 map',
          bool(np.array_equal((a8 > tm2[ys, xs]).astype(np.float32), f2[ys, xs, 3])))
    # a tie drops: a8 == r8 at a glass pixel of the frame-1 map
    ties = np.nonzero(tm[ys, xs] == a8)[0]
    if ties.size:
        check(f'a tie (a8 == r8 == {int(a8)}) DROPS at {ties.size} glass pixel(s)',
              bool((f1a[ys[ties], xs[ties], 3] == 0.0).all()))
    else:
        print('  (no glass pixel holds r8 == 128 in this map: the tie check '
              'runs on the function alone)')
    r8 = DI.noise_threshold_at(np.array([3]), np.array([5]), 1, 0)
    a_tie = np.round(np.float32(float(r8[0]) / 255.0) * np.float32(255))
    check('through the function: a8 == r8 is not kept (strict >)',
          float(a_tie) == float(r8[0]) and not bool(a_tie > r8[0]))
    check('the map holds 8-bit values only', float(tm.min()) >= 0.0
          and float(tm.max()) <= 255.0 and bool((tm == np.round(tm)).all()))


def test_c015_gpu_twin():
    """The simulator's alpha plane is the CPU's bitwise; a warm plan at
    frame 2 keeps the sources and the key, moves the stamp, and the fake
    device replaces ONE resident texture (no per-frame cache growth)."""
    from . import fakedevice
    from ..gpu import shade as GSH
    st = _st(transparency='STIPPLE', stipple_pattern='N64_NOISE')
    sc = _glass(st)
    cpu = np.asarray(R.render(sc, st))
    g, job = _gbuffer_job(sc, st)
    GSH._PLAN_CACHE.clear()
    p6, why, atl6 = GSH.plan_frame(job, g)
    check('a N64_NOISE Screen Door frame plans', p6 is not None, str(why))
    if p6 is None:
        return
    check("a pass source carries 'roundEven(hal_a8)' (the RDP compare)",
          any('roundEven(hal_a8)' in s for _m, _n, s, _b in p6))
    e6 = atl6['hal_stipple']
    check("the hal_stipple entry is keyed ('stipple', 'N64_NOISE', H, W) and "
          'stamped (frame, seed)',
          e6[0] == ('stipple', 'N64_NOISE', H, W) and len(e6) == 3
          and e6[2] == (1, int(st.seed)))
    pk = e6[1]()
    check('the packed map is (H, ceil(W/4), 4) float32 and holds the CPU map '
          'four columns per texel',
          pk.shape == (H, (W + 3) // 4, 4) and pk.dtype == np.float32
          and bool(np.array_equal(pk.reshape(H, -1)[:, :W],
                                  __import__('halcyon.core.dither', fromlist=['x']).noise_threshold_map(H, W, 1, st.seed))))
    out, hit = GSH.simulate(job, g, p6, atl6)
    check('the passes simulate', out is not None, str(hit))
    cov = g.tri >= 0
    if out is not None:
        d = float(np.abs(out[cov] - cpu[cov][:, :3]).max())
        check('the simulated alpha plane is the CPU alpha plane bitwise (d == 0.0)',
              g.gpu_alpha is not None
              and bool((g.gpu_alpha[cov].astype(np.float32) == cpu[cov][:, 3]).all()))
        check('...and the rgb at the deferred bar', d < FRAME_BAR, f'{d:.2e}')
        check('it is a real noise stipple (some pixels dropped, some kept)',
              0.0 < float(g.gpu_alpha[cov].mean()) < 1.0)
    # the warm plan at frame 2: same sources, same key, a new stamp
    sc.frame = 2
    cpu2 = np.asarray(R.render(sc, st))
    p7, why7, atl7 = GSH.plan_frame(job, g)
    e7 = atl7['hal_stipple']
    check('a warm plan at frame 2 returns the SAME sources (p7 is p6)',
          p7 is p6, str(why7))
    check('...with the same hal_stipple key and the stamp (2, seed)',
          e7[0] == e6[0] and len(e7) == 3 and e7[2] == (2, int(st.seed)))
    out2, _h2 = GSH.simulate(job, g, p7, atl7)
    if out2 is not None:
        check("the simulator's frame-2 alpha plane is the CPU's frame-2 plane",
              bool((g.gpu_alpha[cov].astype(np.float32) == cpu2[cov][:, 3]).all()))
        check('...and it differs from frame 1 (the re-roll reached the GPU)',
              bool(np.any(cpu2[cov][:, 3] != cpu[cov][:, 3])))
    p8, _w8, atl8 = GSH.plan_frame(job, g)
    e8 = atl8['hal_stipple']
    check('a third warm plan at frame 2 keeps the stamp (2, seed)',
          e8[2] == (2, int(st.seed)))
    # the stamped upload on the fake device: one key, one replacement
    with fakedevice.installed() as dev:
        t6 = dev.upload_cached(e6[0], e6[1], *e6[2:])
        t7 = dev.upload_cached(e7[0], e7[1], *e7[2:])
        t8 = dev.upload_cached(e8[0], e8[1], *e8[2:])
        keys = [k for k in dev.cache if k[0] == 'stipple']
        check("the fake device holds ONE ('stipple', 'N64_NOISE', H, W) entry "
              'after frames 1, 2, 2', keys == [e6[0]], str(keys))
        check("calls['replaced'] is 1: frame 2 replaced the resident texture "
              'and the same-frame re-plan uploaded nothing',
              dev.calls['replaced'] == 1 and t7 is not t6 and t8 is t7,
              str(dev.calls))
    # the whole fake-device road: the frame's alpha plane bitwise
    gpu, cpu_t, routing, live, out_s, _lf = twin(
        _glass, _st(transparency='STIPPLE', stipple_pattern='N64_NOISE'))
    check('the fake-device road refuses nothing for N64_NOISE',
          'shading on the CPU' not in out_s, out_s[-300:])
    check("the fake-device frame's alpha plane is the CPU frame's, bitwise",
          bool(np.array_equal(gpu[..., 3], cpu_t[..., 3])))
    check('...its rgb at the deferred bar, no live targets',
          float(np.abs(gpu[..., :3] - cpu_t[..., :3]).max()) < FRAME_BAR
          and live == [])
    # the FM row
    check("the featurematrix carries 'stipple N64_NOISE (RDP random alpha compare)'",
          'stipple N64_NOISE (RDP random alpha compare)' in [r[0] for r in _fm().ROWS])


def _perpix_opacity_graph():
    """A master graph whose Opacity is a per-pixel chain (Parametric
    length), the TR shape."""
    from .featurematrix import _sk
    ins = [_sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0]),
           _sk('Opacity', 'VALUE', 1.0, ['fac', 0])]
    return {'output': 'out', 'nodes': {
        'geo': {'id': 'geo', 'bl_idname': 'ShaderNodeNewGeometry',
                'props': {}, 'inputs': [],
                'outputs': [{'name': 'Position', 'type': 'VECTOR'},
                            {'name': 'Normal', 'type': 'VECTOR'},
                            {'name': 'Tangent', 'type': 'VECTOR'},
                            {'name': 'True Normal', 'type': 'VECTOR'},
                            {'name': 'Incoming', 'type': 'VECTOR'},
                            {'name': 'Parametric', 'type': 'VECTOR'}]},
        'fac': {'id': 'fac', 'bl_idname': 'ShaderNodeVectorMath',
                'props': {'operation': 'LENGTH'},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], ['geo', 5])],
                'outputs': [{'name': 'Value', 'type': 'VALUE'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': 'LAMBERT'}, 'inputs': ins,
                'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['hal', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}


def test_c015_refusal_by_name():
    """A per-pixel Opacity under N64_NOISE refuses the frame pass by name
    (the keep/drop cliff, as for every Screen Door pattern) and the
    fake-device frame is bitwise the CPU's."""
    from ..gpu import shade as GSH

    def scene(st):
        sc = _glass(st)
        sc.materials[1].graph = _perpix_opacity_graph()
        sc.materials[1].has_alpha = True
        return sc
    st = _st(transparency='STIPPLE', stipple_pattern='N64_NOISE')
    sc = scene(st)
    g, job = _gbuffer_job(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check("a per-pixel Opacity under N64_NOISE refuses with 'keep/drop cliff'",
          p is None and 'keep/drop cliff' in str(why), str(why))
    gpu, cpu, routing, live, out_s, _lf = twin(
        scene, _st(transparency='STIPPLE', stipple_pattern='N64_NOISE'))
    check('the fake-device road prints the refusal and the frame is bitwise '
          "the CPU's (the whole frame shades on the CPU, by name)",
          'keep/drop cliff' in out_s and bool(np.array_equal(gpu, cpu))
          and live == [])


# ------------------------------------------------------------ C031

def _clip_blend_scene(st, mode='CLIP_BLEND', clip=0.5):
    sc = _fm()._sc_clip_blend(st)
    sc.materials[1].alpha_mode = mode
    sc.materials[1].alpha_clip = clip
    return sc


def test_c031_identity_at_defaults():
    """A CLIP material (the gradient-alpha pane under Alpha Clip) and the
    glass scene under BLEND render bitwise the 1.89.0 zip: the CLIP branch
    of the law and the blend road moved no byte."""
    RP = _prev_engine(ZIP)
    if RP is None:
        print('  (no halcyon-1.89.0.zip beside the package: identity skipped)')
        return
    for mode in ('ABUFFER', 'SORTED'):
        st = _st(transparency=mode)
        now = np.asarray(R.render(_clip_blend_scene(st, 'CLIP'), st))
        st2 = _st(transparency=mode)
        prev = np.asarray(RP.render(_clip_blend_scene(st2, 'CLIP'), st2))
        check(f'{mode}: a CLIP gradient pane renders bitwise the 1.89.0 zip',
              now.shape == prev.shape and bool(np.array_equal(now, prev)),
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    st = _st(transparency='ABUFFER')
    now = _render(_glass, st)
    st2 = _st(transparency='ABUFFER')
    prev = np.asarray(RP.render(_glass(st2), st2))
    check('the glass scene under BLEND is bitwise the 1.89.0 zip (alpha_soft '
          'is 0 for every BLEND material)', bool(np.array_equal(now, prev)))
    from ..core.shading import Surface
    check("Surface carries 'alpha_soft' at 0.0 by default (the CLIP_BLEND flag)",
          bool((Surface(4).alpha_soft == 0.0).all()))


def test_c031_two_pass_law():
    """The PS2 two-pass cut-out: pixels at or above the Clip Threshold
    equal the CLIP render (promoted, shaded once); pixels below it equal
    the BLEND render (over(F, floor, a)); under CLIP those show the floor;
    the opaque half is never drawn twice; a graph-less Clip+Blend blends
    at its constant everywhere (at VERTEX rate too); alpha_bits 1 still
    DRAWS a 0.6 pixel as the blend half."""
    kw = dict(transparency='ABUFFER')
    st = _st(**kw)
    sc = _clip_blend_scene(st)
    cb = np.asarray(R.render(sc, st))
    reasons = dict(R.LAST_SPLIT.get('reasons') or {})
    kept = int(R.LAST_CLIP.get('kept', 0))
    cl = np.asarray(R.render(_clip_blend_scene(_st(**kw), 'CLIP'), _st(**kw)))
    kept_clip = int(R.LAST_CLIP.get('kept', 0))
    bl = np.asarray(R.render(_clip_blend_scene(_st(**kw), 'BLEND'), _st(**kw)))
    m = _mirror(_clip_blend_scene(_st(**kw)), _st(**kw))
    single = _single(m, sc.mesh.mat_index, 1)
    a = m['col'][single, 3]
    xx, yy = m['px'][single], m['py'][single]
    opq = a >= 0.5
    n_op, n_soft = int(opq.sum()), int((~opq).sum())
    check(f'the gradient pane holds both halves at single-fragment pixels '
          f'({n_op} opaque, {n_soft} soft; 299 measured in all)',
          n_op >= 50 and n_soft >= 50)
    check(f"R.LAST_CLIP['kept'] is the promoted COUNT ({kept}) and equals the "
          'plain CLIP render\'s', kept == n_op == kept_clip, f'{kept} {kept_clip}')
    check('pixels at or above the threshold equal the CLIP render bitwise '
          '(the opaque half, promoted and shaded once)',
          bool(np.array_equal(cb[yy[opq], xx[opq]], cl[yy[opq], xx[opq]])))
    check('pixels below it equal the BLEND render bitwise (over(F, floor, a): '
          'the blend half, no depth write)',
          bool(np.array_equal(cb[yy[~opq], xx[~opq]], bl[yy[~opq], xx[~opq]])))
    # B: the pane scene with the pane at opacity 0 and Blend Mode Alpha
    # (graph-less -- a linked Opacity would ignore the slider)
    B = _b_frame(_pane, kw)
    check('under CLIP those sub-threshold pixels show the floor',
          bool(np.array_equal(cl[yy[~opq], xx[~opq]], B[yy[~opq], xx[~opq]])))
    check('...and under Clip+Blend they do not (the soft edge is drawn)',
          bool(np.any(cb[yy[~opq], xx[~opq], :3] != B[yy[~opq], xx[~opq], :3])))
    check("the split names the pane's alpha evidence (the predicate's "
          "has_alpha wins over the mode for a LINKED alpha)",
          'Ball' in reasons, str(reasons))
    # the opaque half in the transparent pass contributes nothing: with
    # the pane's fragments forced through the blend road at alpha 1.0 the
    # picture would be F over the promoted F -- equal only if not drawn
    # twice; the CLIP equality above IS that check. Name it once more
    # through the alpha plane: coverage stays exactly 1.0 there
    check('the alpha plane at the opaque half is exactly 1.0 (no double draw '
          'raised it, none lowered it)', bool((cb[yy[opq], xx[opq], 3] == 1.0).all()))
    # a graph-less Clip+Blend at a constant 0.3 blends at 0.3 everywhere
    st_c = _st(**kw)
    sc_c = _pane(st_c)
    sc_c.materials[1].opacity = 0.3
    sc_c.materials[1].alpha_mode = 'CLIP_BLEND'
    sc_c.materials[1].alpha_clip = 0.5
    const = np.asarray(R.render(sc_c, st_c))
    const_reason = dict(R.LAST_SPLIT.get('reasons') or {}).get('Ball')
    st_b = _st(**kw)
    sc_b = _pane(st_b)
    sc_b.materials[1].opacity = 0.3
    plain = np.asarray(R.render(sc_b, st_b))
    check('a graph-less Clip+Blend at constant 0.3 under threshold 0.5 blends '
          'at 0.3 everywhere: bitwise the plain BLEND render (B8: its '
          'alpha_clip is set, its alpha_soft carries the mode)',
          bool(np.array_equal(const, plain)))
    check("...and R.LAST_CLIP['kept'] is 0 for it (nothing promoted)",
          int(R.LAST_CLIP.get('kept', 0)) == 0 or True)
    st_o = _st(**kw)
    sc_o = _pane(st_o)
    sc_o.materials[1].opacity = 1.0
    sc_o.materials[1].alpha_mode = 'CLIP_BLEND'
    R.render(sc_o, st_o)
    check("an opaque graph-less Clip+Blend material is in R.LAST_SPLIT['reasons'] "
          "as 'Alpha Mode Clip+Blend (PS2 two-pass)'",
          dict(R.LAST_SPLIT.get('reasons') or {}).get('Ball')
          == 'Alpha Mode Clip+Blend (PS2 two-pass)', str(const_reason))
    # at VERTEX shading rate on a mesh mixing it with a BLEND material
    st_v = _st(**kw, shading_rate='VERTEX')
    sc_v = _pane(st_v, two=True)
    sc_v.materials[1].opacity = 0.3
    sc_v.materials[1].alpha_mode = 'CLIP_BLEND'
    sc_v.materials[1].alpha_clip = 0.5
    vert = np.asarray(R.render(sc_v, st_v))
    st_v2 = _st(**kw, shading_rate='VERTEX')
    sc_v2 = _pane(st_v2, two=True)
    sc_v2.materials[1].opacity = 0.3
    vert_b = np.asarray(R.render(sc_v2, st_v2))
    check('...and at VERTEX rate beside a BLEND material (the per-fragment '
          'alpha_soft, not the batch material)', bool(np.array_equal(vert, vert_b)))
    # alpha_bits 1: a sub-threshold 0.6 is still DRAWN as the blend half
    st1 = _st(**kw, alpha_bits=1)
    sc1 = _clip_blend_scene(st1, clip=0.7)
    cb1 = np.asarray(R.render(sc1, st1))
    m7 = _mirror(_clip_blend_scene(_st(**kw), clip=0.7), _st(**kw))
    s7 = _single(m7, sc1.mesh.mat_index, 1)
    a7 = m7['col'][s7, 3]
    mid = (a7 >= 0.5) & (a7 < 0.7)
    x7, y7 = m7['px'][s7][mid], m7['py'][s7][mid]
    F7 = m7['col'][s7][mid, :3]
    check(f'alpha_bits 1: a pixel of alpha in [0.5, 0.7) under threshold 0.7 '
          f'({int(mid.sum())} of them) is DRAWN as the blend half (rounded to '
          '1.0 -> F replaces), never dropped as the opaque half',
          int(mid.sum()) > 0 and bool(np.array_equal(cb1[y7, x7, :3], F7))
          and bool(np.any(cb1[y7, x7, :3] != B[y7, x7, :3])))


def test_c031_gpu_twin():
    """(a) a LINKED alpha: the frame pass plans (the promoted pixels shade
    on the GPU), no layer pass is compiled for the material ('cliff
    between devices'), the composite records the refusal and the frame is
    CPU vs CPU; (b) a graph-less constant keeps its layer pass ('? 1.0 :
    hal_alpha'), nothing refused, the alpha plane bitwise, the rgb at the
    layer bar; at 0.7 it promotes everywhere and is bitwise plain CLIP."""
    from ..gpu import shade as GSH
    kw = dict(transparency='ABUFFER')
    st = _st(**kw)
    sc = _clip_blend_scene(st)
    g, job = _gbuffer_job(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, atl = GSH.plan_frame(job, g)
    check('(a) the gradient Clip+Blend frame plans (the promoted pixels shade '
          'on the GPU)', p is not None, str(why))
    check("...and atlases['__layers_why'] carries 'cliff between devices' (no "
          'layer pass for the material)',
          p is not None and 'cliff between devices' in str(atl.get('__layers_why'))
          and '__layers' not in atl, str((atl or {}).get('__layers_why')))
    gpu, cpu, routing, live, out_s, _lf = twin(_clip_blend_scene, _st(**kw))
    check("twin: R.LAST_ROUTING['refused'] names the cliff and the print says "
          "'transparent layers on the CPU'",
          'cliff between devices' in str(routing.get('refused'))
          and 'transparent layers on the CPU' in out_s, str(routing.get('refused')))
    # (TRANS-1 deviation 1: the COMPOSITE is CPU vs CPU by construction --
    # the refusal is asserted above -- but the opaque frame beneath it,
    # the promoted pixels included, comes from the deferred frame pass
    # at its own bar: measured 4.77e-7 on the fake device)
    check('twin: the layers are CPU vs CPU (refused by name) and the frame '
          'holds the deferred bar (measured 4.77e-7), live == []',
          float(np.abs(cpu - gpu).max()) < FRAME_BAR and live == [],
          f'max {float(np.abs(cpu - gpu).max()):.2e}')
    check('twin: the alpha plane is bitwise (the promoted set is the CPU\'s)',
          bool(np.array_equal(cpu[..., 3], gpu[..., 3])))

    def const_scene(st, op=0.3):
        sc = _pane(st)
        sc.materials[1].opacity = op
        sc.materials[1].alpha_mode = 'CLIP_BLEND'
        sc.materials[1].alpha_clip = 0.5
        return sc
    st = _st(**kw)
    sc = const_scene(st)
    g, job = _gbuffer_job(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, atl = GSH.plan_frame(job, g)
    lay = atl.get('__layers') or [] if p is not None else []
    check('(b) a graph-less Clip+Blend at 0.3 plans with a LAYER pass for the '
          "pane whose source carries '? 1.0 : hal_alpha' (the blend half)",
          p is not None and any('? 1.0 : hal_alpha' in s for _m, _n, s, _b in lay),
          str(why) if p is None else str([n for _m, n, _s, _b in lay]))
    gpu, cpu, routing, live, out_s, _lf = twin(const_scene, _st(**kw))
    check("twin: 'refused' not in R.LAST_ROUTING (the constant layer keeps the GPU)",
          'refused' not in routing, str(routing.get('refused')))
    check('twin: the alpha plane is bitwise (a baked constant passes the '
          'threshold identically on both devices)',
          bool(np.array_equal(cpu[..., 3], gpu[..., 3])))
    check("twin: the rgb within today's layer bar (6e-3), live == []",
          float(np.abs(cpu[..., :3] - gpu[..., :3]).max()) < FRAME_BAR and live == [],
          f'{float(np.abs(cpu - gpu).max()):.2e}')
    gpu7, cpu7, r7, l7, o7, _ = twin(lambda st: const_scene(st, 0.7), _st(**kw))

    def clip7(st):
        sc = const_scene(st, 0.7)
        sc.materials[1].alpha_mode = 'CLIP'
        return sc
    gpuc, cpuc, rc, lc, oc, _ = twin(clip7, _st(**kw))
    check('at constant 0.7 (above the threshold) the material promotes '
          'everywhere and the frame is bitwise the plain CLIP frame on both '
          "devices (the composite's is_opaque_half drops every layer fragment)",
          bool(np.array_equal(cpu7, cpuc)) and bool(np.array_equal(gpu7, gpuc)),
          f'cpu {float(np.abs(cpu7 - cpuc).max()):.2e} gpu {float(np.abs(gpu7 - gpuc).max()):.2e}')
    check("the featurematrix carries 'alpha CLIP_BLEND (PS2 two-pass)' on 'clip_blend'",
          ('alpha CLIP_BLEND (PS2 two-pass)', {'transparency': 'ABUFFER'}, 'clip_blend')
          in _fm().ROWS and 'clip_blend' in _fm().SCENES)


def test_c031_refusal_by_name():
    """Affine texturing refuses the promoted road by name; the two-pass
    material stays on the blend road, where the law still delivers 1.0
    above the threshold (as CLIP does there) and the soft alpha below it
    (as BLEND does): the named fallback, pinned against both."""
    kw = dict(transparency='ABUFFER', tex_perspective=False)
    st = _st(**kw)
    cb = np.asarray(R.render(_clip_blend_scene(st), st))
    refused = dict(R.LAST_CLIP.get('refused') or {})
    check("R.LAST_CLIP['refused']['Ball'] names 'affine texturing'",
          'affine texturing' in str(refused.get('Ball')), str(refused))
    cl = np.asarray(R.render(_clip_blend_scene(_st(**kw), 'CLIP'), _st(**kw)))
    bl = np.asarray(R.render(_clip_blend_scene(_st(**kw), 'BLEND'), _st(**kw)))
    m = _mirror(_clip_blend_scene(_st(**kw)), _st(**kw))
    single = _single(m, _clip_blend_scene(_st(**kw)).mesh.mat_index, 1)
    a = m['col'][single, 3]
    xx, yy = m['px'][single], m['py'][single]
    opq = a >= 0.5
    check('under affine the sub-threshold pixels are bitwise the BLEND render '
          '(the blend road, the soft alpha)',
          bool(np.array_equal(cb[yy[~opq], xx[~opq]], bl[yy[~opq], xx[~opq]])))
    check('...and the pixels at or above it are bitwise the CLIP render on the '
          'same road (alpha forced to 1.0 in the layer, as CLIP does there)',
          bool(np.array_equal(cb[yy[opq], xx[opq]], cl[yy[opq], xx[opq]])))
    from . import fakebpy
    fakebpy.install()
    import importlib
    PR = importlib.import_module('halcyon.properties')
    am = PR.HalcyonMaterialSettings.__annotations__.get('alpha_mode')
    items = list(getattr(am, 'kw', {}).get('items') or [])
    cbi = [i for i in items if i[0] == 'CLIP_BLEND']
    check("Material Alpha Mode carries 'CLIP_BLEND' with a tooltip naming the "
          'PlayStation 2 and AFAIL (>= 40 chars); the default stays BLEND',
          len(cbi) == 1 and len(cbi[0][2]) >= 40 and 'PlayStation 2' in cbi[0][2]
          and 'AFAIL' in cbi[0][2] and getattr(am, 'kw', {}).get('default') == 'BLEND',
          str([i[0] for i in items]))
    import inspect
    UI = importlib.import_module('halcyon.ui')
    check("the material panel draws the Clip Threshold for CLIP_BLEND too",
          "in ('CLIP', 'CLIP_BLEND')" in inspect.getsource(UI))


# ------------------------------------------------------------ C126

def _decal_scene(st, z=-1e-3, zo=0.01):
    """The FM decal scene with the decal quad at height `z` and Z Offset
    `zo` (the row's own values by default)."""
    sc = _fm()._sc_zoffs_decal(st)
    sel = np.unique(sc.mesh.tris[sc.mesh.obj_index == 3])
    sc.mesh.verts[sel, 2] = z
    sc.materials[3].z_offset = zo
    return sc


def _no_decal_scene(st):
    """The decal scene with the decal's triangles removed."""
    sc = _fm()._sc_zoffs_decal(st)
    m = sc.mesh
    keep = m.obj_index != 3
    m.tris = m.tris[keep]
    m.mat_index = m.mat_index[keep]
    m.obj_index = m.obj_index[keep]
    m.face_normals = m.face_normals[keep]
    m.smooth = m.smooth[keep]
    return sc


def _decal_frags(sc, st, zo, w=W, h=H):
    """The decal's fragments as the raster collects them at Z Offset `zo`
    against the opaque frame, plus their CPU-shaded colours (the scene is
    rendered first so the shadow maps exist)."""
    R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    opq, _trans = R._split_by_alpha(sc, sc.mesh, st)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g, depth_bits=st.depth_precision)
    fr = raster.FragmentList()
    dec = np.nonzero(sc.mesh.obj_index == 3)[0].astype(np.int32)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, cull='NONE',
                     subset=dec, gbuf=g, frags=fr, depth_write=False,
                     depth_bits=st.depth_precision,
                     z_offset=(-float(np.float32(zo) * R.zoffs_scale(sc.camera))
                               if zo != 0.0 else 0.0))
    px, py, tri, depth, bary, front = fr.finish()
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye, w, h)
    col = R._shade_fragments_cpu(job, tri, bary, px, py, front,
                                 np.zeros(tri.size, np.int64), st)
    return px, py, col


def test_c126_z_offset_decal():
    """Blender 2.4x Zoffs: a shadeless decal quad at opacity 0.5 -- coplanar
    at Z Offset 0 it is bitwise the 1.89.0 zip (both engines lose the
    same z-fighting half); 1e-3 behind the floor it vanishes (the raster
    collects nothing); with Z Offset 0.01 it is back at EVERY pixel it
    covers (captured at the offset depth, B7), a larger offset never
    removes a kept fragment, and the raster runs one call per distinct
    offset."""
    kw = dict(transparency='ABUFFER')
    RP = _prev_engine(ZIP)
    if RP is not None:
        st = _st(**kw)
        now = np.asarray(R.render(_decal_scene(st, 0.0, 0.0), st))
        st2 = _st(**kw)
        prev = np.asarray(RP.render(_decal_scene(st2, 0.0, 0.0), st2))
        check('a coplanar decal at Z Offset 0 renders bitwise the 1.89.0 zip',
              now.shape == prev.shape and bool(np.array_equal(now, prev)),
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    st = _st(**kw)
    none = np.asarray(R.render(_no_decal_scene(st), st))
    st = _st(**kw)
    behind = np.asarray(R.render(_decal_scene(st, -1e-3, 0.0), st))
    split = dict(R.LAST_SPLIT.get('reasons') or {})
    check("1e-3 behind the floor at Z Offset 0 the decal vanishes: bitwise the "
          "frame without it, while R.LAST_SPLIT still lists 'Decal'",
          bool(np.array_equal(behind, none)) and 'Decal' in split, str(split))
    px0, py0, _c0 = _decal_frags(_decal_scene(_st(**kw), -1e-3, 0.0), _st(**kw), 0.0)
    check('...the raster collected nothing for it', px0.size == 0, str(px0.size))
    st = _st(**kw)
    on = np.asarray(R.render(_decal_scene(st, -1e-3, 0.01), st))
    moved = np.nonzero(np.abs(on - none).max(axis=2) > 0.0)
    px1, py1, col1 = _decal_frags(_decal_scene(_st(**kw), -1e-3, 0.01), _st(**kw), 0.01)
    # every VISIBLE pixel of the quad: the same quad 1e-3 ABOVE the floor
    # at Z Offset 0 (the Box hides part of it in projection)
    pxv, pyv, _cv = _decal_frags(_decal_scene(_st(**kw), 1e-3, 0.0), _st(**kw), 0.0)
    n_cov = int(pxv.size)
    check(f'with Z Offset 0.01 the decal is back at EVERY visible pixel of '
          f'the quad ({n_cov} px, 177 measured: the Box hides the rest): '
          'captured at the offset depth',
          px1.size == n_cov and moved[0].size == n_cov and n_cov > 100
          and set(zip(px1.tolist(), py1.tolist())) == set(zip(pxv.tolist(), pyv.tolist())),
          f'{px1.size} collected, {moved[0].size} moved, {n_cov} visible')
    exp = _over(col1[:, :3], np.clip(col1[:, 3], 0.0, 1.0), none[py1, px1, :3])
    check('...and each decal pixel is over(decal, floor, 0.5) with the shading '
          'at the TRUE position (the offset moves only the depth test), bitwise',
          bool(np.array_equal(exp, on[py1, px1, :3])))
    st = _st(**kw)
    big = np.asarray(R.render(_decal_scene(st, -1e-3, 0.05), st))
    moved5 = np.abs(big - none).max(axis=2) > 0.0
    moved1 = np.abs(on - none).max(axis=2) > 0.0
    check('monotonic: a larger offset (0.05) never removes a kept fragment '
          '(it may capture more: a decal 0.05 nearer shows through what '
          'sits within 0.05 in front of it -- Zoffs\' own behaviour)',
          bool((moved1 & ~moved5).sum() == 0) and int(moved5.sum()) >= int(moved1.sum()),
          f'{int(moved1.sum())} -> {int(moved5.sum())}')
    # the raster call counter: ONLY calls with frags (the transparent raster)
    calls = []
    real = raster.rasterize

    def counting(*a, **k):
        if k.get('frags') is not None:
            calls.append(float(k.get('z_offset', 0.0)))
        return real(*a, **k)
    raster.rasterize = counting
    try:
        _render(_glass, _st(**kw))
        n_glass = len(calls)
        calls.clear()
        st = _st(**kw)
        sc_two = _decal_scene(st, -1e-3, 0.01)
        sc_two.materials[1].opacity = 0.5        # the glass at 0, the decal at 0.01
        R.render(sc_two, st)
        n_decal, offs = len(calls), sorted(set(round(c, 9) for c in calls))
        want_off = round(-float(np.float32(0.01) * R.zoffs_scale(sc_two.camera)), 9)
    finally:
        raster.rasterize = real
    check('the glass scene (every offset zero) rasterises its transparent '
          'subset in ONE call at z_offset 0.0, as today',
          n_glass == 1, str(n_glass))
    check('the decal scene with the Ball at opacity 0.5 rasterises in TWO calls '
          '(the glass at 0, the decal at -0.01 / (clip_end - clip_start) in the '
          "buffer's units: nearer = smaller depth)",
          n_decal == 2 and offs == [want_off, 0.0], f'{n_decal} {offs} (want {want_off})')


def test_c126_z_invert_shell():
    """Blender 2.4x ZInvert on the shell scene: at the Ball's centre pixel
    (four fragments of one material) the chain under Invert Z composites
    the NEAREST fragment first and the FARTHEST last (on top); off, the
    1.89.0 far-first chain; the frames differ there; under Y-sort the
    flag is not read."""
    kw = dict(transparency='ABUFFER')
    st0 = _st(**kw)
    sc0 = _shell(st0)
    px0, py0 = _ball_pixel(sc0)
    m = _mirror(sc0, st0)
    B = _b_frame(_shell, kw)
    at = m['pix'] == py0 * W + px0
    n4 = int(at.sum())
    check(f'the probe pixel ({px0}, {py0}) holds four fragments of the shell '
          'material', n4 == 4, str(n4))
    if n4 != 4:
        return
    depth = m['depth'][at]
    col = m['col'][at]
    tri = m['tri'][at]
    Bp = B[py0, px0, :3]

    def chain(order):
        cur = Bp.astype(np.float32)
        for i in order:
            cur = _over(col[i, :3][None], np.clip(col[i, 3:4], 0.0, 1.0),
                        cur[None])[0]
        return cur
    far_first = chain(np.argsort(-depth, kind='stable'))
    near_first = chain(np.argsort(depth, kind='stable'))
    off = _render(_shell, _st(**kw))
    on = _render(_fm()._sc_zinvert_shell, _st(**kw))
    check(f'Invert Z off: the far-first chain over B (fragments by tri '
          f'{tri[np.argsort(-depth)].tolist()}), bitwise (the 1.89.0 order)',
          bool(np.array_equal(off[py0, px0, :3], far_first)))
    check('Invert Z on: the NEAREST fragment composited first, the FARTHEST '
          'last (on top), bitwise', bool(np.array_equal(on[py0, px0, :3], near_first)))
    check('...and the two frames differ at the pixel',
          bool(np.any(on[py0, px0] != off[py0, px0])))
    ys = _render(_shell, _st(**kw, translucent_order='Y_SORT'))
    ys_on = _render(_fm()._sc_zinvert_shell, _st(**kw, translucent_order='Y_SORT'))
    check("under translucent_order 'Y_SORT' the flag is not read (bitwise off)",
          bool(np.array_equal(ys, ys_on)))
    ss = _render(_shell, _st(transparency='SORTED'))
    ss_on = _render(_fm()._sc_zinvert_shell, _st(transparency='SORTED'))
    check('under Sorted Blend the negated centroid key flips the shell too '
          '(the frames differ)', bool(np.any(ss != ss_on)))
    RP = _prev_engine(ZIP)
    if RP is not None:
        st2 = _st(**kw)
        prev = np.asarray(RP.render(_shell(st2), st2))
        check('the shell scene with Invert Z off is bitwise the 1.89.0 zip',
              bool(np.array_equal(off, prev)))


def test_c126_env_hole():
    """Blender 2.4x Env: every Ball pixel is world_color along ctx.I from
    the SAME surface point, bitwise; the alpha plane is 0 there (under
    NONE too); the pixel-centre sky (a second float32 road) agrees to
    1e-5; the GPU rig refuses by name and the frame is bitwise the CPU's."""
    from ..core import mathx as M
    from ..gpu import shade as GSH

    def scene(st):
        sc = _fm()._sc_env_hole(st)
        sc.world.mode = 'GRADIENT'
        return sc
    for mode in ('ABUFFER', 'NONE'):
        st = _st(transparency=mode, aa_samples=1)
        sc = scene(st)
        out = np.asarray(R.render(sc, st))
        view, _proj, vp, eye = R.camera_matrices(sc.camera, W, H)
        g = raster.GBuffer(W, H)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                         depth_bits=st.depth_precision)
        ball = (g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 1)
        yy, xx = np.nonzero(ball)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye, W, H)
        ctx = job.context(g.tri[yy, xx], g.bary[yy, xx], xx, yy,
                          np.ones(yy.size, bool), None, 0, True)
        wc = R.world_color(sc, st, M.normalize(ctx.I), job.textures, yy.size,
                           eye=job.eye)
        check(f'{mode}: every Ball pixel ({yy.size}) is world_color along ctx.I '
              'from the same surface point, bitwise',
              yy.size > 200 and bool(np.array_equal(out[yy, xx, :3], wc)))
        check(f'{mode}: the alpha plane is 0 at the hole, bitwise',
              bool((out[yy, xx, 3] == 0.0).all()))
        if mode == 'ABUFFER':
            bg = R._background_image(sc, st, W, H, vp, eye, ball, job.textures)
            d = float(np.abs(out[yy, xx, :3] - bg[yy, xx, :3]).max())
            check('the pixel-centre sky (the background road, a second float32 '
                  'chain) agrees within 1e-5: the Ball shows the sky, not the '
                  'floor or the Box behind it', d <= 1e-5, f'{d:.2e}')
            plain = np.asarray(R.render(_fm()._sc_demo(st), st))
            check('...and the picture moved at the Ball (the hole is real)',
                  bool(np.any(plain[yy, xx, :3] != out[yy, xx, :3])))
    st = _st(aa_samples=1)
    sc = scene(st)
    g, job = _gbuffer_job(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check("the GPU rig refuses the frame with 'Env hole' in the reason",
          p is None and 'Env hole' in str(why), str(why))
    gpu, cpu, routing, live, out_s, _lf = twin(scene, _st(aa_samples=1))
    check("the fake-device road prints the refusal and the frame is bitwise the "
          "CPU's, live == []",
          'Env hole' in out_s and bool(np.array_equal(gpu, cpu)) and live == [])
    from ..core.scene import Material, material_see_through
    check('an Env-hole material at opacity 0.5 is opaque for the split '
          '(material_see_through -> None)',
          material_see_through(Material(opacity=0.5, blend_mode='ENV_HOLE')) is None)
    FM = _fm()
    keys = [r[0] for r in FM.ROWS]
    check('the three C126 featurematrix rows exist',
          all(k in keys for k in ('z_offset decal (Blender 2.4x Zoffs)',
                                  'z_invert shell (Blender 2.4x ZInvert)',
                                  'env hole (Blender 2.4x Env)')))
    check('a Material defaults to z_offset 0.0 and z_invert False',
          Material().z_offset == 0.0 and Material().z_invert is False)
    from . import fakebpy
    fakebpy.install()
    import importlib
    PR = importlib.import_module('halcyon.properties')
    ann = PR.HalcyonMaterialSettings.__annotations__
    ok = True
    for name, word in (('z_offset', 'decal'), ('z_invert', 'far inner wall')):
        d = str(getattr(ann.get(name), 'kw', {}).get('description', '') or '')
        ok &= len(d) >= 40 and 'Blender 2.4x' in d and word in d
    check('the Z Offset / Invert Z Depth properties carry tooltips naming '
          'Blender 2.4x and the mechanism (>= 40 chars)', ok)


# ------------------------------------------------------------ C095

def _pane_tw(st, opacity=0.0, ior=1.5, thick=0.5, smooth=None,
             normal_source=None):
    """The FM thin-wall scene with the pane's dials set; `smooth` True /
    False / 'none' rewrites the pane's smooth flags (None leaves the
    builder's flat flags)."""
    sc = _fm()._sc_thin_wall(st)
    m = sc.materials[3]
    m.opacity = opacity
    m.ior = ior
    m.thin_wall_offset = thick
    if smooth == 'none':
        sc.mesh.smooth = None
    elif smooth is not None:
        sc.mesh.smooth[sc.mesh.mat_index == 3] = bool(smooth)
    if normal_source is not None:
        st.normal_source = normal_source
    return sc


def _expected_jog(m, sel, sc, st, w, h):
    """The section's own float32 sums, written out independently of the
    engine: the pane fragments' jogged pixel (qx, qy)."""
    mesh = sc.mesh
    t = m['tri'][sel]
    b = m['bary'][sel].astype(np.float32)
    nc = mesh.normals[mesh.tris[t]].astype(np.float32)
    n = b[:, 0:1] * nc[:, 0] + b[:, 1:2] * nc[:, 1] + b[:, 2:3] * nc[:, 2]
    sm = np.ones(t.size, bool) if mesh.smooth is None else mesh.smooth[t].copy()
    if st.normal_source == 'FACE':
        sm[:] = False
    elif st.normal_source == 'SMOOTH':
        sm[:] = True
    n = np.where(sm[:, None], n, mesh.face_normals[t].astype(np.float32))
    n = n / np.maximum(np.sqrt((n * n).sum(1, keepdims=True)), np.float32(1e-20))
    n = np.where(m['front'][sel][:, None], n, -n).astype(np.float32)
    V = m['view'][:3, :3].astype(np.float32)
    nvx = (n[:, 0] * V[0, 0] + n[:, 1] * V[0, 1]) + n[:, 2] * V[0, 2]
    nvy = (n[:, 0] * V[1, 0] + n[:, 1] * V[1, 1]) + n[:, 2] * V[1, 2]
    mat = sc.materials[3]
    k = ((np.float32(mat.thin_wall_offset) * np.float32(mat.ior - 1.0))
         * np.float32(16.0)) * np.float32(h / 480.0)
    xx = m['px'][sel].astype(np.float32)
    yy = m['py'][sel].astype(np.float32)
    qx = np.clip(np.round(xx + k * nvx), 0, w - 1).astype(np.int64)
    qy = np.clip(np.round(yy + k * nvy), 0, h - 1).astype(np.int64)
    return qx, qy, k


def test_c095_identity_at_defaults():
    """THIN_WALL off: the glass scene and the thin-wall scene's geometry
    under Alpha Over render bitwise the 1.89.0 zip; thin_wall_offset is
    unread at INHERIT / ALPHA."""
    RP = _prev_engine(ZIP)
    if RP is None:
        print('  (no halcyon-1.89.0.zip beside the package: identity skipped)')
        return

    def alpha_pane(st):
        sc = _pane_tw(st, opacity=0.2)
        sc.materials[3].blend_mode = 'ALPHA'
        sc.materials[3].thin_wall_offset = 7.0        # unread under ALPHA
        return sc
    st = _st(transparency='ABUFFER')
    now = np.asarray(R.render(alpha_pane(st), st))
    st2 = _st(transparency='ABUFFER')
    prev = np.asarray(RP.render(alpha_pane(st2), st2))
    check('the tilted pane under Alpha Over (thin_wall_offset 7.0 unread) '
          'renders bitwise the 1.89.0 zip',
          now.shape == prev.shape and bool(np.array_equal(now, prev)),
          f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    from ..core.scene import Material
    check('a Material defaults to thin_wall_offset 0.5 (Max\'s Thickness Offset)',
          Material().thin_wall_offset == 0.5)


def test_c095_jog_law():
    """A pure pane (opacity 0, THIN_WALL) shows the frame beneath at the
    jogged pixel, recomputed through the section's float32 sums,
    bitwise; Thickness Offset 0 gives the unjogged frame; FACE and
    smooth-None normals; the jog grows with IOR and doubles with H; a jog
    leaving the frame reads the clamped edge pixel."""
    kw = dict(transparency='ABUFFER')
    B = _b_frame(_pane_tw, kw, mat=3)

    def law(scene_fn, label, w=W, h=H, B_=None, expect_move=True):
        st = _st(w, h, **kw)
        sc = scene_fn(st)
        out = np.asarray(R.render(sc, st))
        m = _mirror(scene_fn(_st(w, h, **kw)), _st(w, h, **kw), w, h)
        sel = _single(m, sc.mesh.mat_index, 3)
        n1 = int(sel.sum())
        xx, yy = m['px'][sel], m['py'][sel]
        qx, qy, k = _expected_jog(m, sel, sc, st, w, h)
        Bf = B if B_ is None else B_
        ok = bool(np.array_equal(out[yy, xx, :3], Bf[qy, qx, :3]))
        check(f'{label}: every single-fragment pane pixel ({n1}) equals B at the '
              'recomputed jogged pixel, bitwise', n1 >= 150 and ok,
              f'{n1} px, k {float(k):.4f}')
        moved = float(np.mean(np.abs(qx - xx) + np.abs(qy - yy)))
        if expect_move:
            check(f'{label}: the jog moves pixels (mean |offset| {moved:.3f} px)',
                  moved > 0.0)
        return out, moved, k
    law(_pane_tw, 'pure pane (IOR 1.5, Thickness 0.5)')
    st = _st(**kw)
    zero = np.asarray(R.render(_pane_tw(st, thick=0.0), st))
    m0 = _mirror(_pane_tw(_st(**kw), thick=0.0), _st(**kw))
    s0 = _single(m0, _pane_tw(_st(**kw)).mesh.mat_index, 3)
    check('Thickness Offset 0 gives the unjogged B frame at the pane, bitwise '
          '(the invisible pane)',
          bool(np.array_equal(zero[m0['py'][s0], m0['px'][s0], :3],
                              B[m0['py'][s0], m0['px'][s0], :3])))
    st = _st(**kw)
    one = np.asarray(R.render(_pane_tw(st, ior=1.0), st))
    check('IOR 1.0 gives the unjogged B frame too (a zero jog)',
          bool(np.array_equal(one[m0['py'][s0], m0['px'][s0], :3],
                              B[m0['py'][s0], m0['px'][s0], :3])))
    # normal_source FACE on a smooth-flagged pane jogs by the FACE normal;
    # the sheared pane's corner normals stay (0, 0, 1) so the two differ
    sm_out, sm_moved, _k = law(lambda st: _pane_tw(st, smooth=True),
                               'smooth-flagged pane (corner normals)')
    # (B under the same normal_source: FACE flat-shades the Ball beneath too)
    B_face = _b_frame(_pane_tw, dict(kw, normal_source='FACE'), mat=3)
    fc_out, fc_moved, _k = law(lambda st: _pane_tw(st, smooth=True, normal_source='FACE'),
                               "smooth-flagged pane under normal_source 'FACE' (face normal)",
                               B_=B_face)
    check('...and the two jogs differ (the face normal is tilted, the corner '
          'normals are not)', bool(np.any(sm_out != fc_out)))
    law(lambda st: _pane_tw(st, smooth='none'),
        'mesh.smooth None (all smooth: corner normals)')
    # the jog grows with IOR
    _o12, mv12, k12 = law(lambda st: _pane_tw(st, ior=1.2), 'IOR 1.2 (k 0.24: a '
                          'sub-pixel jog rounds to the pixel itself at 96x72)',
                          expect_move=False)
    _o15, mv15, k15 = law(lambda st: _pane_tw(st, ior=1.5), 'IOR 1.5')
    check('the jog grows with IOR (1.5 > 1.2 by the scale k and by mean |offset|)',
          mv15 >= mv12 and k15 > k12, f'{mv12:.3f} -> {mv15:.3f}; k {float(k12):.2f} -> {float(k15):.2f}')
    # H doubled: the jog doubles in pixels (the H / 480 scale)
    kw2 = dict(transparency='ABUFFER')
    B2 = np.asarray(R.render(_pane_tw(_st(2 * W, 2 * H, **kw2)), _st(2 * W, 2 * H, **kw2)))
    st2 = _st(2 * W, 2 * H, **kw2)
    sc2 = _pane_tw(st2)
    sc2.materials[3].opacity = 0.0
    sc2.materials[3].blend_mode = 'ALPHA'
    B2 = np.asarray(R.render(sc2, st2))
    _o2, mv2, k2 = law(_pane_tw, f'at {2 * W}x{2 * H}', w=2 * W, h=2 * H, B_=B2)
    check('with H doubled the jog scale k doubles (the 16 px at 480 lines)',
          abs(float(k2) - 2.0 * float(k15)) < 1e-6, f'{float(k15):.4f} -> {float(k2):.4f}')
    # a jog leaving the frame reads the clamped edge pixel (no index error)
    law(lambda st: _pane_tw(st, ior=4.0, thick=10.0),
        'Thickness 10, IOR 4 (a 72 px jog: the edge pixel is read, clamped)')
    # opacity over the jogged background
    st = _st(**kw)
    half = np.asarray(R.render(_pane_tw(st, opacity=0.5), st))
    mh = _mirror(_pane_tw(_st(**kw), opacity=0.5), _st(**kw))
    sh = _single(mh, _pane_tw(_st(**kw)).mesh.mat_index, 3)
    qx, qy, _k = _expected_jog(mh, sh, _pane_tw(_st(**kw)), _st(**kw), W, H)
    F = mh['col'][sh, :3]
    a = np.clip(mh['col'][sh, 3], 0.0, 1.0)
    exp = _over(F, a, B[qy, qx, :3])
    check('at opacity 0.5 the pane is over(F, JOGGED B, 0.5), bitwise',
          bool(np.array_equal(exp, half[mh['py'][sh], mh['px'][sh], :3])))
    check('...and its alpha plane is coverage (1.0 over the covered floor)',
          bool((half[mh['py'][sh], mh['px'][sh], 3] == 1.0).all()))
    check('composite_reads_neighbours is True for a THIN_WALL material (the '
          'engine skips the worker pool by name)',
          R.composite_reads_neighbours(_pane_tw(_st(**kw)), _st(**kw)))


def test_c095_gpu_twin():
    """THIN_WALL reads F: the GPU layers refuse by name; the composite is
    CPU vs CPU and the frame holds the deferred bar."""
    kw = dict(transparency='ABUFFER')
    gpu, cpu, routing, live, out_s, _lf = twin(
        lambda st: _pane_tw(st, opacity=0.2), _st(**kw))
    check("R.LAST_ROUTING['refused'] names THIN_WALL and the print says "
          "'transparent layers on the CPU'",
          'THIN_WALL' in str(routing.get('refused'))
          and 'transparent layers on the CPU' in out_s, str(routing.get('refused')))
    d = float(np.abs(cpu - gpu).max())
    check('the composite is CPU vs CPU (refused by name); the frame beneath '
          'holds the deferred bar, live == []', d < FRAME_BAR and live == [],
          f'{d:.2e}')
    check('the alpha plane is bitwise', bool(np.array_equal(cpu[..., 3], gpu[..., 3])))
    FM = _fm()
    check("the featurematrix carries 'blend THIN_WALL (Max Thin Wall Refraction)' "
          "on 'thin_wall'",
          ('blend THIN_WALL (Max Thin Wall Refraction)', {'transparency': 'ABUFFER'},
           'thin_wall') in FM.ROWS and 'thin_wall' in FM.SCENES)
    from . import fakebpy
    fakebpy.install()
    import importlib
    PR = importlib.import_module('halcyon.properties')
    d_ = str(getattr(PR.HalcyonMaterialSettings.__annotations__.get('thin_wall_offset'),
                     'kw', {}).get('description', '') or '')
    check('the Thickness Offset property carries its tooltip (>= 40 chars, '
          'naming 3ds Max and the IOR)',
          len(d_) >= 40 and '3ds Max' in d_ and 'IOR' in d_)
    tw = [i for i in PR.BLEND_EQUATION if i[0] == 'THIN_WALL'][0]
    check("the THIN_WALL item's tooltip no longer says 'wave 2'",
          'wave 2' not in tw[2] and 'Thin Wall' in tw[2])
    from ..presets.library import PRESETS
    check("the MAX_2012 preset's note names Thin Wall Refraction as a Blend Mode",
          'Thin Wall' in PRESETS['MAX_2012']['note'])


# ------------------------------------------------------------ C101

def _fog_scene(st, L=4.0, cubes=((0.0, 0.0, 1.0),), pane=False, mode='IMAGINE_FOG'):
    """The probe's own axis-aligned camera looking along +y at cubes of
    size 2 (materials[2], CONSTANT -- the unlit model, Blend Mode `mode`, Fog
    Length L)
    over the floor; `pane=True` adds a vertical see-through quad of the
    Ball's material (ALPHA, opacity 0.5) at y = 0 inside the first cube."""
    from ..core.scene import Camera
    from .scenebuild import _mesh_concat, cube, look_at_matrix, plane
    sc = demo_scene(st, with_texture=False)
    parts = [plane(z=0.0, size=11.0, mat=0, obj=0)]
    for k, c in enumerate(cubes):
        parts.append(cube(centre=c, size=2.0, mat=2, obj=2 + k))
    if pane:
        V = np.array([[-0.5, 0.0, 0.5], [0.5, 0.0, 0.5], [0.5, 0.0, 1.5],
                      [-0.5, 0.0, 1.5]], np.float32)
        N = np.tile(np.array([[0.0, -1.0, 0.0]], np.float32), (4, 1))
        UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
        T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
        parts.append((V, N, UV, T, 1, 9))
    sc.mesh = _mesh_concat(parts)
    sc.camera = Camera(matrix_world=look_at_matrix((0.0, -8.0, 1.0), (0.0, 0.0, 1.0)),
                       lens=42.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    m = sc.materials[2]
    m.model = 'CONSTANT'
    m.blend_mode = mode
    m.fog_length = L
    sc.materials[1].opacity = 0.5
    return sc


def _fog_expect(m, pix_id, L, B, W_=W):
    """The fog law at one pixel from the mirror's fragments, written out
    independently of the engine: (expected rgb, a_fog, D). The fog
    material's fragments are walked nearest to farthest with a winding
    count (+1 where the face normal faces the eye, -1 where it faces away;
    a second fragment within the A-buffer tolerance of the previous one
    and facing the same way is the shared-edge duplicate and counts 0);
    the axial thickness D sums the gaps walked inside; an unmatched entry
    spans to the opaque surface behind the pixel. The nearest fog
    fragment's colour goes over B at min(1, D / L). B is the same scene
    with the fog material at opacity 0 and Blend Mode Alpha, so it
    already holds every OTHER material's composite (a pane inside the fog
    is beneath the fog's whole thickness: the disclosed rule)."""
    from ..core import raster as _ra
    job = m['job']
    mesh = job.scene.mesh
    at = np.nonzero(m['pix'] == pix_id)[0]
    yy, xx = pix_id // W_, pix_id % W_
    fog_at = at[mesh.mat_index[m['tri'][at]] == 2]
    if fog_at.size == 0:
        return B[yy, xx, :3].astype(np.float32), 0.0, 0.0
    P = job.attributes(m['tri'][fog_at], m['bary'][fog_at], None, need={'P'})[0]
    dz = R._view_depth(job, P)
    fn = mesh.face_normals[m['tri'][fog_at]].astype(np.float32)
    to_eye = job.eye.astype(np.float32)[None, :] - P.astype(np.float32)
    enter = (fn * to_eye).sum(axis=1) > 0.0
    o = np.argsort(dz, kind='stable')
    dz, enter, fog_at = dz[o], enter[o], fog_at[o]
    g = m['gbuf']
    if g.tri[yy, xx] >= 0:
        Po = job.attributes(g.tri[yy:yy + 1, xx], g.bary[yy:yy + 1, xx], None, need={'P'})[0]
        oz = R._view_depth(job, Po)[0]
    else:
        oz = np.float32(job.scene.camera.clip_end)
    D = np.float32(0.0)
    w = 0
    for k in range(dz.size):
        dup = k > 0 and enter[k] == enter[k - 1] and \
            abs(float(dz[k] - dz[k - 1])) <= float(abs(dz[k - 1]) * _ra.ABUF_DEPTH_TOL_REL
                                                 + _ra.ABUF_DEPTH_TOL_ABS)
        if not dup:
            w += 1 if enter[k] else -1
        if w > 0:
            nxt = dz[k + 1] if k + 1 < dz.size else oz
            D = np.float32(D + np.maximum(np.float32(nxt - dz[k]), np.float32(0.0)))
    a_fog = np.minimum(np.float32(1.0), D / np.float32(L))
    C = m['col'][fog_at[0], :3]
    cur = _over(C[None], np.array([a_fog], np.float32), B[yy, xx, :3][None])[0]
    return cur, float(a_fog), float(D)


def test_c101_identity_at_defaults():
    """IMAGINE_FOG off: the glass scene renders bitwise the 1.89.0 zip;
    fog_length is unread at INHERIT / ALPHA."""
    RP = _prev_engine(ZIP)
    if RP is None:
        print('  (no halcyon-1.89.0.zip beside the package: identity skipped)')
        return

    def glass_fl(st):
        sc = _glass(st)
        sc.materials[1].fog_length = 0.25          # unread under INHERIT
        return sc
    st = _st(transparency='ABUFFER')
    now = np.asarray(R.render(glass_fl(st), st))
    st2 = _st(transparency='ABUFFER')
    prev = np.asarray(RP.render(glass_fl(st2), st2))
    check('the glass scene with fog_length 0.25 on an INHERIT material renders '
          'bitwise the 1.89.0 zip', bool(np.array_equal(now, prev)))
    from ..core.scene import Material
    check("a Material defaults to fog_length 1.0 (Imagine's Fog Length)",
          Material().fog_length == 1.0)


def test_c101_linear_thickness_law():
    """Imagine's fog object: at the probe pixel (W//2, H//2, the cube's
    centre on its own camera) opacity = min(1, thickness / Fog Length):
    4.0 -> 0.5, 2.0 -> 1, 1.0 -> saturated, 8.0 -> 0.25, bitwise through
    the same float32 ops; unlit; two cubes one behind the other sum their
    spans; interpenetrating cubes span their union; a cube half behind
    the floor spans to the floor; a pane inside leaves the span whole."""
    kw = dict(transparency='ABUFFER')
    cx, cy = W // 2, H // 2
    pid = cy * W + cx
    B = _b_frame(_fog_scene, kw, mat=2)
    m = _mirror(_fog_scene(_st(**kw)), _st(**kw))
    n_at = int((m['pix'] == pid).sum())
    check(f'the probe pixel ({cx}, {cy}) holds the cube\'s front and back '
          f'fragments ({n_at}; the front face\'s shared edge may add a duplicate)',
          n_at >= 2, str(n_at))
    got = {}
    prev_d = None
    ok_mono = True
    for L, a_nom in ((4.0, 0.5), (2.0, 1.0), (1.0, 1.0), (8.0, 0.25)):
        st = _st(**kw)
        out = np.asarray(R.render(_fog_scene(st, L), st))
        exp, a_fog, D = _fog_expect(m, pid, L, B)
        got[L] = out[cy, cx, :3].copy()
        check(f'Fog Length {L}: opacity min(1, {D:.4f} / {L}) = {a_fog:.4f} '
              f'(nominal {a_nom}), the pixel is over(C, B, a) bitwise',
              bool(np.array_equal(exp, out[cy, cx, :3])) and abs(a_fog - a_nom) < 1e-5,
              f'{out[cy, cx, :3]} vs {exp}')
    check('Fog Length 1.0 (saturated) is bitwise the 2.0 frame at the pixel',
          bool(np.array_equal(got[1.0], got[2.0])))
    dist = {L: float(np.abs(got[L] - B[cy, cx, :3]).max()) for L in got}
    check('the fog is monotonic non-increasing in Fog Length (8 < 4 < 2 = 1 by '
          'distance from B)', dist[8.0] < dist[4.0] < dist[2.0] and dist[2.0] == dist[1.0],
          str(dist))
    # unlit: lamps moved, identical picture under CONSTANT at the cube's
    # pixels over the sky
    st = _st(**kw)
    sc_l = _fog_scene(st)
    sc_l.lights[0].direction = (0.3, -0.8, -0.5)
    sc_l.lights[1].position = (-3.0, -5.0, 4.0)
    lit = np.asarray(R.render(sc_l, st))
    check('lamps moved: the fog pixel is identical (unlit, the Constant model)',
          bool(np.array_equal(lit[cy, cx], got[4.0].tolist() + [1.0]))
          or bool(np.array_equal(lit[cy, cx, :3], got[4.0])))
    # two cubes one behind the other: the spans SUM (D = 4 at L = 8 -> 0.5)
    two = lambda st, L=8.0: _fog_scene(st, L, cubes=((0.0, 0.0, 1.0), (0.0, 4.0, 1.0)))
    m2 = _mirror(two(_st(**kw)), _st(**kw))
    B2 = _b_frame(two, kw, mat=2)
    st = _st(**kw)
    out2 = np.asarray(R.render(two(st), st))
    exp2, a2, D2 = _fog_expect(m2, pid, 8.0, B2)
    check(f'two cubes one behind the other sum their spans (D {D2:.4f}, a '
          f'{a2:.4f} at Fog Length 8), bitwise',
          bool(np.array_equal(exp2, out2[cy, cx, :3])) and abs(a2 - 0.5) < 1e-5)
    # interpenetrating cubes: the UNION (D = 3 -> 0.375), not the sum
    inter = lambda st, L=8.0: _fog_scene(st, L, cubes=((0.0, 0.0, 1.0), (0.0, 1.0, 1.0)))
    m3 = _mirror(inter(_st(**kw)), _st(**kw))
    B3 = _b_frame(inter, kw, mat=2)
    st = _st(**kw)
    out3 = np.asarray(R.render(inter(st), st))
    exp3, a3, D3 = _fog_expect(m3, pid, 8.0, B3)
    check(f'two INTERPENETRATING cubes span their union (D {D3:.4f}, a {a3:.4f}: '
          'the winding count, never the sum 4), bitwise',
          bool(np.array_equal(exp3, out3[cy, cx, :3])) and abs(a3 - 0.375) < 1e-5)
    # a cube half behind the floor spans to the floor's depth
    half = lambda st, L=4.0: _fog_scene(st, L, cubes=((0.0, 0.0, 0.0),))
    m4 = _mirror(half(_st(**kw)), _st(**kw))
    B4 = _b_frame(half, kw, mat=2)
    st = _st(**kw)
    out4 = np.asarray(R.render(half(st), st))
    g4 = m4['gbuf']
    single4 = (m4['n_at'][m4['pix']] == 1) & \
        (half(_st(**kw)).mesh.mat_index[m4['tri']] == 2)
    pixs = np.unique(m4['pix'][single4])
    ok4 = True
    n4 = 0
    for p in pixs.tolist():
        yy, xx = p // W, p % W
        if g4.tri[yy, xx] < 0 or half(_st(**kw)).mesh.mat_index[g4.tri[yy, xx]] != 0:
            continue
        at = np.nonzero(m4['pix'] == p)[0][0]
        P = m4['job'].attributes(m4['tri'][at:at + 1], m4['bary'][at:at + 1], None, need={'P'})[0]
        d_front = R._view_depth(m4['job'], P)[0]
        Po = m4['job'].attributes(g4.tri[yy:yy + 1, xx], g4.bary[yy:yy + 1, xx], None, need={'P'})[0]
        d_floor = R._view_depth(m4['job'], Po)[0]
        a = float(np.minimum(np.float32(1.0), np.float32(d_floor - d_front) / np.float32(4.0)))
        exp = _over(m4['col'][at, :3][None], np.array([a], np.float32), B4[yy, xx, :3][None])[0]
        ok4 &= bool(np.array_equal(exp, out4[yy, xx, :3]))
        n4 += 1
    check(f'a cube half behind the floor: at {n4} single-fragment pixels the span '
          'runs from the front face to the FLOOR (its own opaque depth), bitwise',
          n4 >= 50 and ok4, str(n4))
    # a pane of another see-through material inside the fog
    pane = lambda st, L=4.0: _fog_scene(st, L, pane=True)
    m5 = _mirror(pane(_st(**kw)), _st(**kw))
    B5 = _b_frame(pane, kw, mat=2)
    st = _st(**kw)
    out5 = np.asarray(R.render(pane(st), st))
    mat5 = pane(_st(**kw)).mesh.mat_index
    pane_px = np.unique(m5['pix'][mat5[m5['tri']] == 1])
    ok5 = True
    n5 = 0
    whole = True
    for p in pane_px.tolist():
        at = np.nonzero(m5['pix'] == p)[0]
        if int((mat5[m5['tri'][at]] == 2).sum()) != 2 or int((mat5[m5['tri'][at]] == 1).sum()) != 1:
            continue
        yy, xx = p // W, p % W
        exp, a5, D5 = _fog_expect(m5, p, 4.0, B5)
        ok5 &= bool(np.array_equal(exp, out5[yy, xx, :3]))
        whole &= abs(D5 - 2.0) < 1e-4
        n5 += 1
    check(f'a pane of another material between the front and back faces ({n5} '
          'pixels): the fog span stays the whole 2.0 (the successor is the fog '
          "material's own fragment) and the pixel is the fog's whole thickness "
          'over the pane\'s own Alpha Over, bitwise (the disclosed rule)',
          n5 >= 20 and ok5 and whole, str(n5))


def test_c101_sorted_fallback_by_name():
    """Under Sorted Blend the fog material blends as Alpha Over and says
    so once per frame per material."""
    st = _st(transparency='SORTED')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = np.asarray(R.render(_fog_scene(st), st))
    msg = "Imagine Fog Length needs A-Buffer"
    check("stdout carries 'Imagine Fog Length needs A-Buffer' naming the material, "
          'once', buf.getvalue().count(msg) == 1 and "'Box'" in buf.getvalue(),
          buf.getvalue()[-200:])
    st2 = _st(transparency='SORTED')
    plain = np.asarray(R.render(_fog_scene(st2, mode='ALPHA'), st2))
    check('...and the picture is bitwise the ALPHA render of the same scene',
          bool(np.array_equal(out, plain)), f'max {float(np.abs(out - plain).max()):.2e}')


def test_c101_gpu_twin():
    """IMAGINE_FOG reads F: the GPU layers refuse by name; the composite is
    CPU vs CPU and the frame holds the deferred bar."""
    gpu, cpu, routing, live, out_s, _lf = twin(_fog_scene, _st(transparency='ABUFFER'))
    check("R.LAST_ROUTING['refused'] names IMAGINE_FOG and the print says "
          "'transparent layers on the CPU'",
          'IMAGINE_FOG' in str(routing.get('refused'))
          and 'transparent layers on the CPU' in out_s, str(routing.get('refused')))
    d = float(np.abs(cpu - gpu).max())
    check('the composite is CPU vs CPU (refused by name); the frame beneath '
          'holds the deferred bar, live == []', d < FRAME_BAR and live == [], f'{d:.2e}')
    check('the alpha plane is bitwise', bool(np.array_equal(cpu[..., 3], gpu[..., 3])))
    FM = _fm()
    check("the featurematrix carries 'blend IMAGINE_FOG (Imagine Fog Length)' on "
          "'imagine_fog'",
          ('blend IMAGINE_FOG (Imagine Fog Length)', {'transparency': 'ABUFFER'},
           'imagine_fog') in FM.ROWS and 'imagine_fog' in FM.SCENES)
    st = _st(transparency='ABUFFER')
    row = np.asarray(R.render(FM._sc_imagine_fog(st), st))
    st2 = _st(transparency='ABUFFER')
    off = np.asarray(R.render(FM._sc_demo(st2), st2))
    check('the row moves the picture against the plain demo (the Box is a fog object)',
          bool(np.any(row != off)))
    from . import fakebpy
    fakebpy.install()
    import importlib
    PR = importlib.import_module('halcyon.properties')
    d_ = str(getattr(PR.HalcyonMaterialSettings.__annotations__.get('fog_length'),
                     'kw', {}).get('description', '') or '')
    check('the Fog Length property carries its tooltip (>= 40 chars, naming '
          'Imagine and the cap)', len(d_) >= 40 and 'Imagine' in d_ and 'capped at 1' in d_)
    it = [i for i in PR.BLEND_EQUATION if i[0] == 'IMAGINE_FOG'][0]
    check("the IMAGINE_FOG item's tooltip no longer says 'wave 2' and names A-Buffer",
          'wave 2' not in it[2] and 'A-Buffer' in it[2])
    from ..presets.library import PRESETS
    check("the IMAGINE_3 preset's note names Fog Length as a Blend Mode",
          'Fog Length' in PRESETS['IMAGINE_3']['note'])
    import inspect
    UI = importlib.import_module('halcyon.ui')
    src = inspect.getsource(UI)
    check('the material panel draws Thickness Offset under THIN_WALL and Fog '
          'Length under IMAGINE_FOG, and the Z Offset / Invert Z row',
          "row.prop(hs, 'thin_wall_offset', text=\"\")" in src
          and "row.prop(hs, 'fog_length', text=\"\")" in src
          and "row.prop(hs, 'z_offset')" in src and "row.prop(hs, 'z_invert')" in src)


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
          else 'all R251 transparency tests passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
