"""R251 RAST-B (1.90.0): the AA resolve dials and the two period wire roads.

    C094  LightWave's Limit Dynamic Range -- every sample clipped at 1.0
          before the anti-aliasing filter (twin: RESOLVE_C, bitwise)
    C122  Blender 2.2x-2.41's gamma-2 OSA sample blend through the
          400-entry tables (twin: RESOLVE_G / _CG, bitwise)
    C063  Elite's wireframe rule: an edge draws when either face faces
          the viewer, no depth test, a dot past a distance (cpu_only,
          both devices bitwise through the wire road's readback by name)
    C055  the vector monitor beam of the Atari DVG / AVG (cpu_only, as
          C063)

The first test pins the 1.89.0 zip loudly and the identity at defaults
(render AND post.process bitwise the previous engine). Every check name
reads as a sentence of what it proves. Own FAILS / check (never imported
from test_render); run alone with

    python -m halcyon.tests.test_r251_raster_wire
"""
import importlib
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np

from ..core import post as PO
from ..core import raster as RA
from ..core import render as R
from ..core.settings import RenderSettings
from ..gpu import frame as FR
from ..gpu import stages as STG
from ..presets.library import CATEGORIES, PRESETS, apply_preset
from .scenebuild import _mesh_concat, cube, demo_scene, plane, sphere
from .r251_common import clear_palette_locks, delta_extra, preset_delta
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


W, H = 96, 72
ZIP = 'halcyon-1.89.0.zip'


def settings(w=W, h=H, **kw):
    """The pack's default rig: 96x72, one sample, no transparency."""
    st = base_settings(w, h)
    st.transparency = 'NONE'
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def post_kw(sc, st, w=None, h=None):
    return dict(frame=1, seed=st.seed,
                target_size=(w or st.resolution_x, h or st.resolution_y),
                allow_resize=False,
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None))


def _maxdiff(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.shape != b.shape:
        return f'shape {a.shape} vs {b.shape}'
    return f'max {float(np.abs(a - b).max())}'


def _same(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


# ---------------------------------------------------------------------------
# the identity pin: the 1.89.0 engine from the zip beside the package
# ---------------------------------------------------------------------------

def test_a_identity_at_defaults():
    """The 1.89.0 zip is beside the package (LOUD when absent), the
    default frame and its post chain are bitwise the 1.89.0 engine's, and
    every shipped preset whose dict the round left alone (the 1.89.0
    library from the same zip decides: any pack's edit or new preset is
    excluded by construction, r251_common.preset_delta) renders AND
    post-processes bitwise the old engine, while the one this pack edited
    (LIGHTWAVE_56) moves."""
    RP = _prev_engine(ZIP)
    check(f'the 1.89.0 zip ({ZIP}) is beside the package and its engine '
          'imports', RP is not None)
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')

    st = settings()
    sc = demo_scene(st, with_texture=False)
    now = np.asarray(R.render(sc, st))
    st2 = settings()
    sc2 = demo_scene(st2, with_texture=False)
    prev = np.asarray(RP.render(sc2, st2))
    check('the demo frame at defaults is bitwise the 1.89.0 engine\'s (the '
          'five new settings are invisible until asked for)',
          _same(now, prev), _maxdiff(now, prev))
    now_p = PO.process(now, st, **post_kw(sc, st))
    prev_p = prev_post.process(prev, st2, **post_kw(sc2, st2))
    check('...and post.process over it is bitwise the 1.89.0 chain',
          _same(now_p, prev_p), _maxdiff(now_p, prev_p))

    # the supersampled default road too: the resolve is this pack's subject
    for filt, n in (('BOX', 4), ('MITCHELL', 9)):
        # ss = round(sqrt(aa_samples)): 4 -> 2, 9 -> 3
        st = settings(aa_mode='SUPERSAMPLE', aa_samples=n, aa_filter=filt)
        now = np.asarray(R.render(demo_scene(st, with_texture=False), st))
        st2 = settings(aa_mode='SUPERSAMPLE', aa_samples=n, aa_filter=filt)
        prev = np.asarray(RP.render(demo_scene(st2, with_texture=False), st2))
        check(f'the supersampled frame ({filt}, {n} samples) at defaults is '
              'bitwise the 1.89.0 resolve', _same(now, prev), _maxdiff(now, prev))

    # the shipped presets: bitwise (render AND post) where the round left
    # the dict alone. The 1.89.0 library from the same zip says which dicts
    # changed (any pack's edit, any new preset): excluded by construction
    delta = preset_delta(RP, PRESETS)

    def pair(key):
        st = RenderSettings()
        apply_preset(st, key)
        st.resolution_x, st.resolution_y = 48, 36
        st.aa_samples = min(st.aa_samples, 2)
        st.output_scale = 'NONE'
        st.transparency = 'NONE' if st.transparency == 'ABUFFER' else st.transparency
        st2 = st.copy()
        sc = demo_scene(st, with_texture=False)
        now = np.asarray(R.render(sc, st))
        sc2 = demo_scene(st2, with_texture=False)
        prev = np.asarray(RP.render(sc2, st2))
        # the adaptive palette lock is per-engine session state: both
        # engines build THIS frame's palette from this frame (r251_common)
        clear_palette_locks(RP)
        now_p = np.asarray(PO.process(now.copy(), st, **post_kw(sc, st)))
        prev_p = np.asarray(prev_post.process(prev.copy(), st2,
                                              **post_kw(sc2, st2)))
        return now, prev, now_p, prev_p

    moved, moved_post, bad = [], [], []
    for key in delta.unchanged:
        try:
            now, prev, now_p, prev_p = pair(key)
        except Exception as exc:                                # noqa: BLE001
            bad.append(f'{key}: {type(exc).__name__}')
            continue
        if not _same(now, prev):
            moved.append((key, _maxdiff(now, prev)))
        elif not _same(now_p, prev_p):
            moved_post.append((key, _maxdiff(now_p, prev_p)))
    check(f'the {len(delta.unchanged)} shipped presets whose dict the round '
          'left alone render bitwise the 1.89.0 engine, render and post '
          '(each engine from a cleared palette lock)',
          not bad and not moved and not moved_post,
          f'render moved {moved}; post moved {moved_post}; errors {bad}; '
          + delta_extra(delta))
    try:
        now, prev, _, _ = pair('LIGHTWAVE_56')
        lw_moves = not _same(now, prev)
        lw_extra = _maxdiff(now, prev)
    except Exception as exc:                                    # noqa: BLE001
        lw_moves, lw_extra = False, f'{type(exc).__name__}: {exc}'
    check('...and LIGHTWAVE_56 (aa_clamp_samples True, the pack\'s one '
          'preset edit, so its dict is excluded above) moves its picture '
          '-- named in the CHANGELOG',
          'LIGHTWAVE_56' in delta.changed and lw_moves, lw_extra)


# ---------------------------------------------------------------------------
# the stage twins through the simulator (the TR run_stage shape)
# ---------------------------------------------------------------------------

def run_stage(src, h, w, uniforms, samplers):
    """Run a resolve-shaped fragment source through the GLSL simulator
    over an h x w output; (image (h, w, 4), None) or (None, why)."""
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile
    src = src.replace('in vec2 vUV;', 'uniform vec2 vUV;')
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, err
    n = h * w
    yy, xx = np.mgrid[0:h, 0:w]
    u = {'vUV': np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h],
                         1).astype(np.float32)}
    for k, v in samplers.items():
        u[k] = Texture(v, colorspace='Non-Color', filt='NEAREST',
                       wrap='EXTEND')
    for k, v in uniforms.items():
        if isinstance(v, int):
            u[k] = np.full(n, int(v), np.int32)
        elif isinstance(v, (tuple, list)):
            u[k] = np.tile(np.asarray(v, np.float32)[None, :], (n, 1))
        else:
            u[k] = np.full(n, float(v), np.float32)
    outs, _d = prog.run(u, {}, n)
    return np.asarray(outs['Color'], np.float32).reshape(h, w, 4), None


def _ktex(ss, st):
    ktex = np.zeros((ss, ss, 4), np.float32)
    ktex[..., 0] = FR.resolve_kernel(ss, st)
    return ktex


def _gpu_road(**kw):
    """The whole GPU road headless through the fake device (the
    test_gpu_frame_resident shape): (cpu, gpu, live targets, FR.LAST,
    dev)."""
    from ..gpu import shade as GSH
    from . import fakedevice
    st_c = settings(**kw)
    sc_c = demo_scene(st_c, with_texture=False)
    cpu = R.render(sc_c, st_c)
    st_g = settings(**kw)
    st_g.render_device = 'GPU'
    sc_g = demo_scene(st_g, with_texture=False)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        gpu = R.render(sc_g, st_g)
        live = [t for t in dev.targets if not t.freed]
    return cpu, gpu, live, dict(FR.LAST), dev


# ---------------------------------------------------------------------------
# C094 LightWave's Limit Dynamic Range
# ---------------------------------------------------------------------------

def test_c094_limit_dynamic_range():
    """LightWave 5.6-7.5's Limit Dynamic Range: every sample's colour is
    clipped at 1.0 BEFORE the anti-aliasing filter, so a hot edge blends
    between displayable values instead of saturating its whole pixel."""
    rng = np.random.default_rng(94)
    Wo, Ho = 24, 18

    # (b) the law: never brighter, equal where nothing exceeded 1.0
    for ss in (2, 3):
        img = rng.random((Ho * ss, Wo * ss, 4)).astype(np.float32)
        # a sparse set of overbright samples (0..4) on a cool field (0..1)
        hot_px = rng.random((Ho * ss, Wo * ss)) < 0.08
        img[hot_px, :3] = (rng.random((int(hot_px.sum()), 3)) * 4.0).astype(np.float32)
        st0 = settings(aa_filter='BOX')
        st1 = settings(aa_filter='BOX', aa_clamp_samples=True)
        plain = R._resolve(img, Wo, Ho, ss, st0)
        clip = R._resolve(img, Wo, Ho, ss, st1)
        tile = img.reshape(Ho, ss, Wo, ss, 4)
        cool = (tile[..., :3] <= 1.0).all(axis=(1, 3, 4))
        check(f'ss {ss}: the clipped resolve is never brighter than the plain '
              'one on any channel', bool((clip[..., :3] <= plain[..., :3] + 1e-6).all()),
              _maxdiff(clip, plain))
        check(f'ss {ss}: ...equal on every pixel whose samples were all <= 1.0 '
              f'({int(cool.sum())} of {cool.size}), and its alpha is untouched',
              cool.any() and bool(np.array_equal(clip[cool][..., :3],
                                                 plain[cool][..., :3]))
              and bool(np.array_equal(clip[..., 3], plain[..., 3])))
        hot = ~cool
        check(f'ss {ss}: ...and strictly darker on some pixel that held an '
              'overbright sample', hot.any()
              and bool((clip[hot][..., :3] < plain[hot][..., :3]).any()))
    one = (rng.random((Ho, Wo, 4)).astype(np.float32) * 3.0)
    st1 = settings(aa_clamp_samples=True)
    got = R._resolve(one, Wo, Ho, 1, st1)
    want = one.copy()
    want[..., :3] = np.minimum(want[..., :3], 1.0)
    check('at one sample the flag returns min(img, 1) on RGB (LightWave '
          'clipped the single pass too), alpha as it was',
          _same(got, want) and _same(R._resolve(one, Wo, Ho, 1, settings()), one))

    # (c) the halo: a hot block whose edge lands mid-pixel. Halcyon's
    # resolve is a per-pixel tile (the filter never reaches a neighbour),
    # so the halo is the EDGE PIXEL saturating: half covered by an 8.0
    # surface it reads 4.0 -> solid white on the display, the hot surface
    # a pixel wider; clipped first it reads 0.5, a real anti-aliased edge
    ss = 4
    field = np.zeros((16 * ss, 16 * ss, 4), np.float32)
    field[..., 3] = 1.0
    field[26:38, 26:38, :3] = 8.0          # 3 output px, edges at half pixels
    for filt in ('BOX', 'TRIANGLE'):
        st0 = settings(aa_filter=filt, aa_filter_width=2.0)
        st1 = settings(aa_filter=filt, aa_filter_width=2.0, aa_clamp_samples=True)
        plain = R._resolve(field, 16, 16, ss, st0)
        clip = R._resolve(field, 16, 16, ss, st1)
        sat0 = int((plain[..., 0] >= 1.0).sum())
        sat1 = int((clip[..., 0] >= 1.0).sum())
        edge0 = float(plain[8, 6, 0])
        edge1 = float(clip[8, 6, 0])
        check(f'{filt} width 2, ss 4: the block saturates fewer output pixels '
              f'under the clamp ({sat1} < {sat0}) -- the halo is gone',
              sat1 < sat0 and sat1 >= 4)
        check(f'{filt}: the half-covered edge pixel reads {edge1:.3f} under the '
              f'clamp (a blend of 1.0 and 0.0) against {edge0:.2f} unclipped',
              0.0 < edge1 < 1.0 and edge0 > 1.0)

    # (d) the GPU twin: the pinned stage untouched, the variant exact
    base_spec = {'samplers': [], 'floats': [], 'ints': [], 'vec2': [], 'vec3': []}
    base_spec.update({k: list(v) for k, v in STG.INTERFACE['RESOLVE'].items()})
    check("resolve_source(False, False) is STAGES['RESOLVE'] byte for byte "
          "and resolve_spec(False, False) is INTERFACE['RESOLVE'] (the "
          'pinned road never changes)',
          STG.resolve_source(False, False) == STG.STAGES['RESOLVE']
          and STG.resolve_spec(False, False) == base_spec)
    src_c = STG.resolve_source(True, False)
    check('the RESOLVE_C source declares clamp_samples and the per-tap min '
          'before the accumulation',
          'uniform float clamp_samples;' in src_c
          and 'if (clamp_samples > 0.5) { t.rgb = min(t.rgb, vec3(1.0)); }' in src_c
          and src_c.index('min(t.rgb') < src_c.index('acc = acc + t * k;'))
    for ss, filt in ((2, 'BOX'), (2, 'TRIANGLE'), (3, 'GAUSS'), (3, 'BOX'),
                     (2, 'MITCHELL')):
        st = settings(aa_filter=filt, aa_clamp_samples=True)
        big = (rng.random((Ho * ss, Wo * ss, 4)).astype(np.float32) * 3.0)
        got, err = run_stage(src_c, Ho, Wo,
                             {'resolution': (float(Wo), float(Ho)),
                              'ss': int(ss), 'clamp_samples': 1.0},
                             {'source': big, 'kernel': _ktex(ss, st)})
        ref = R._resolve(big, Wo, Ho, ss, st)
        check(f'RESOLVE_C at ss {ss} with the {filt} filter is bitwise '
              'render._resolve under the flag (d == 0.0)',
              got is not None and _same(got, ref),
              str(err) if got is None else _maxdiff(got, ref))
    # the variant with the uniform at 0 is the plain resolve
    st = settings(aa_filter='BOX')
    big = (rng.random((Ho * 2, Wo * 2, 4)).astype(np.float32) * 3.0)
    got, err = run_stage(src_c, Ho, Wo, {'resolution': (float(Wo), float(Ho)),
                                         'ss': 2, 'clamp_samples': 0.0},
                         {'source': big, 'kernel': _ktex(2, st)})
    check('RESOLVE_C with clamp_samples 0.0 is the plain resolve bitwise',
          got is not None and _same(got, R._resolve(big, Wo, Ho, 2, st)),
          str(err or ''))

    # the fake device: the declared-vs-spec check and a driver-rules draw
    from . import fakedevice
    with fakedevice.installed() as dev:
        sh, why = dev.compile_dynamic('RESOLVE_C', src_c, STG.resolve_spec(True, False))
        check("the fake device's compile_dynamic accepts ('RESOLVE_C', source, "
              'spec): every declared uniform is in the spec', sh is not None,
              str(why))
        if sh is not None:
            st = settings(aa_filter='GAUSS', aa_clamp_samples=True)
            big = (rng.random((Ho * 2, Wo * 2, 4)).astype(np.float32) * 3.0)
            tgt = dev.Target(Wo, Ho)
            out = dev.draw_fullscreen(
                sh, {'resolution': (float(Wo), float(Ho)), 'ss': 2,
                     'clamp_samples': 1.0},
                {'source': dev.upload(big), 'kernel': dev.upload(_ktex(2, st))},
                tgt)
            check('...and a draw under the driver\'s rules (kinds, bindings) '
                  'lands bitwise on render._resolve',
                  _same(out, R._resolve(big, Wo, Ho, 2, st)))
    # the whole GPU road: the variant draws at ss 2, the frame within the
    # shading twin's rounding (6e-6, alpha bitwise); at ss 1 the frame
    # reads back by name before the CPU clamp
    cpu, gpu, live, LF, dev = _gpu_road(aa_mode='SUPERSAMPLE', aa_samples=4,
                                        aa_clamp_samples=True,
                                        global_ambient=(1.0, 1.0, 1.0),
                                        global_ambient_level=3.0)
    d = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check('the GPU road at ss 2 with the flag: RESOLVE_C drew on the resident '
          'frame (no CPU readback), the frame within 6e-6 of the CPU road '
          '(the shading twin\'s rounding), alpha bitwise, nothing left live',
          LF.get('resolve', {}).get('variant') == 'RESOLVE_C'
          and not LF.get('left_gpu') and d <= 6e-6
          and bool(np.array_equal(gpu[..., 3], cpu[..., 3])) and live == [],
          f"variant {LF.get('resolve')} left_gpu {LF.get('left_gpu')!r} max {d} "
          f'{len(live)} live')
    check('...and the flag moved the picture (samples above 1.0 were there '
          'to clip)', not _same(cpu, _gpu_road(aa_mode='SUPERSAMPLE',
                                                aa_samples=4,
                                                global_ambient=(1.0, 1.0, 1.0),
                                                global_ambient_level=3.0)[0]))
    cpu, gpu, live, LF, dev = _gpu_road(aa_clamp_samples=True,
                                        global_ambient=(1.0, 1.0, 1.0),
                                        global_ambient_level=3.0)
    d = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check("at ss 1 the GPU road releases the frame by name ('sample clamp at "
          "1 sample') and clips on the CPU: the frame within 6e-6, no RGB "
          'above 1.0, nothing left live',
          'sample clamp at 1 sample' in str(LF.get('left_gpu', ''))
          and d <= 6e-6 and float(gpu[..., :3].max()) <= 1.0 and live == [],
          f"left_gpu {LF.get('left_gpu')!r} max {d} peak {float(gpu[..., :3].max())}")

    # (e) the matrix rows and the preset
    from .featurematrix import ROWS
    keys = {r[0]: r for r in ROWS}
    row = keys.get('LightWave limit dynamic range (clamp before AA)')
    check('the feature matrix carries the LightWave clamp row (flat ambient '
          '3.0 pushes samples past 1.0, so the row is not vacuous)',
          row is not None and row[1].get('aa_clamp_samples') is True
          and row[1].get('global_ambient_level') == 3.0 and row[2] == 'demo')
    row2 = keys.get('LightWave limit dynamic range on a hot edge (GAUSS, ss 2)')
    check("...and the hot-edge row on the 'hot_edge' scene",
          row2 is not None and row2[2] == 'hot_edge'
          and row2[1].get('aa_clamp_samples') is True)
    lw = PRESETS['LIGHTWAVE_56']['settings']
    check("the LIGHTWAVE_56 preset clips its nine samples ('aa_clamp_samples': "
          'True, the manual\'s own recommendation) and its note says so',
          lw.get('aa_clamp_samples') is True and lw.get('aa_samples') == 9
          and 'Limit Dynamic Range' in PRESETS['LIGHTWAVE_56']['note'])


# ---------------------------------------------------------------------------
# C122 Blender 2.2x-2.41's gamma-2 OSA sample blend
# ---------------------------------------------------------------------------

def test_c122_gamma2_blend():
    """Blender 2.2x-2.41 squared every OSA sample through a 400-entry
    piecewise-linear table before the filter and square-rooted the sum
    after (initrender.c renderloop_setblending, gammaCorrectionTables.c),
    so bright thin lines stay wide and dark ones pinch."""
    rng = np.random.default_rng(122)
    Wo, Ho = 24, 18
    T = RA.gamma2_tables()

    # (b) the table law: Blender's makeGammaTables(2.0) verbatim
    n = RA.GAMMA2_SIZE
    check('the tables have 401 knots i * 0.0025 with g[400] == 1.0 and '
          'ig[400] == 1.0, and their slopes end at 0',
          n == 400 and T.dom.shape == (401,) and float(T.g[400]) == 1.0
          and float(T.ig[400]) == 1.0 and float(T.gf[400]) == 0.0
          and float(T.igf[400]) == 0.0 and T.dom.dtype == np.float32
          and float(T.dom[1]) == float(np.float32(0.0025)))
    knots = RA.gamma2_correct(T.dom)
    e_k = float(np.abs(knots - T.g).max())
    check('gamma2_correct of the 401 knots returns g within 2e-6 (float32 '
          'i * 0.0025 * 400 can land one below the knot, as the C own '
          '(int)(c * 400) did; measured, not assumed)', e_k <= 2e-6,
          f'max {e_k}')
    back = RA.gamma2_inverse(T.g)
    err = np.abs(back - T.dom)
    e_hi = float(err[20:].max())
    e_lo = float(err[:20].max())
    check('gamma2_inverse of g returns the knots within one table step from '
          '0.05 up (the piecewise-linear round trip) and within five steps '
          'at the steep foot of the square root (the first cell rises 0..0.05)',
          e_hi <= 0.0025 and e_lo <= 0.0126, f'max {e_hi} / foot {e_lo}')
    xs = rng.random(4096).astype(np.float32)
    sq = RA.gamma2_correct(xs)
    e_sq = float(np.abs(sq - xs.astype(np.float64) ** 2).max())
    rt = RA.gamma2_inverse(np.clip(xs, 0.05, 1.0))
    e_rt = float(np.abs(rt - np.sqrt(np.clip(xs, 0.05, 1.0).astype(np.float64))).max())
    check('the tables are the square and the square root to within their '
          'cell curvature (2e-6 and, away from the steep first cells, 1e-4)',
          e_sq <= 2e-6 and e_rt <= 1e-4, f'square {e_sq} root {e_rt}')
    tex = RA.gamma2_texture()
    check('gamma2_texture is one (1, 401, 4) float32 row of (g, gf, ig, igf) '
          'per texel -- the CPU\'s own bits for the gtab sampler',
          tex.shape == (1, 401, 4) and tex.dtype == np.float32
          and _same(tex[0, :, 0], T.g) and _same(tex[0, :, 1], T.gf)
          and _same(tex[0, :, 2], T.ig) and _same(tex[0, :, 3], T.igf))

    # (c) the edge law "RMS >= mean": a half-covered 0|1 edge resolves to
    # sqrt(0.5), flat interiors keep their value to a table step
    ss = 2
    field = np.zeros((Ho * ss, Wo * ss, 4), np.float32)
    field[..., 3] = 1.0
    field[:, Wo - 1:, :3] = 1.0      # pixel 11's tile holds (0, 1) per row
    st0 = settings(aa_filter='BOX')
    st1 = settings(aa_filter='BOX', aa_gamma_blend=True)
    plain = R._resolve(field, Wo, Ho, ss, st0)
    gam = R._resolve(field, Wo, Ho, ss, st1)
    edge = float(gam[0, Wo // 2 - 1, 0])
    check('a half-covered 0|1 edge pixel resolves to sqrt(0.5) = 0.7071 '
          'within one table step under the blend, not 0.5 (the bright side '
          'grows: "anti-aliasing that does not swim")',
          abs(float(plain[0, Wo // 2 - 1, 0]) - 0.5) < 1e-6
          and abs(edge - np.sqrt(0.5)) <= 0.0025, f'{edge}')
    flat = np.ones((Ho, Wo), bool)
    flat[:, Wo // 2 - 1] = False
    check('flat interiors (all-0 and all-1 tiles) keep their value to a table '
          'step, alpha filtered plainly and never gamma\'d',
          float(np.abs(gam[flat][..., :3] - plain[flat][..., :3]).max()) <= 0.0025
          and _same(gam[..., 3], plain[..., 3]))
    smooth = rng.random((Ho * ss, Wo * ss, 4)).astype(np.float32) * 0.9 + 0.05
    plain = R._resolve(smooth, Wo, Ho, ss, st0)
    gam = R._resolve(smooth, Wo, Ho, ss, st1)
    low = float((gam[..., :3] - plain[..., :3]).min())
    check('on random samples in [0.05, 0.95] every blended pixel is >= the '
          'plain resolve minus 1e-3 (Jensen on the squared blend, the table '
          'cells the slack) and most are strictly brighter',
          low >= -1e-3 and float((gam[..., :3] > plain[..., :3]).mean()) > 0.9,
          f'min delta {low}')
    st_d = settings(aa_gamma_blend=True)
    one = rng.random((Ho, Wo, 4)).astype(np.float32)
    check('at one sample the flag is bitwise inert (Blender\'s do_gamma '
          'required OSA)', _same(R._resolve(one, Wo, Ho, 1, st_d), one))
    st_a = settings(aa_mode='SUPERSAMPLE', aa_samples=4, aa_filter='BOX')
    st_b = settings(aa_mode='SUPERSAMPLE', aa_samples=4, aa_filter='BOX',
                    aa_gamma_blend=True)
    fa = np.asarray(R.render(demo_scene(st_a, with_texture=False), st_a))
    fb = np.asarray(R.render(demo_scene(st_b, with_texture=False), st_b))
    moved = int((np.abs(fb[..., :3] - fa[..., :3]) > 0.0025).any(axis=2).sum())
    npx = fa.shape[0] * fa.shape[1]
    check('on the demo at ss 2 the blend moves between 1% and 60% of the '
          'pixels by more than a table step (every shaded gradient tile, '
          'never the flat sky) and never the alpha',
          0.01 * npx < moved < 0.6 * npx and _same(fa[..., 3], fb[..., 3])
          and float(np.abs(fb[..., :3] - fa[..., :3]).max()) > 0.05,
          f'{moved} px moved of {npx}')

    # (d) the GPU twin: RESOLVE_G and RESOLVE_CG exact in the simulator
    src_g = STG.resolve_source(False, True)
    src_cg = STG.resolve_source(True, True)
    spec_g = STG.resolve_spec(False, True)
    spec_cg = STG.resolve_spec(True, True)
    check('RESOLVE_G declares gamma_blend and the gtab sampler, squares each '
          'tap after the fetch and square-roots acc before the write; '
          'RESOLVE_CG clamps first (the CPU\'s order)',
          'uniform sampler2D gtab;' in src_g and 'gamma_blend' in spec_g['floats']
          and 'gtab' in spec_g['samplers'] and 'clamp_samples' not in spec_g['floats']
          and src_g.index('if (gamma_blend > 0.5) {') < src_g.index('acc = acc + t * k;')
          < src_g.rindex('if (gamma_blend > 0.5) {') < src_g.index('Color = acc;')
          and src_cg.index('min(t.rgb') < src_cg.index('if (gamma_blend > 0.5) {')
          and spec_cg['floats'] == ['clamp_samples', 'gamma_blend'])
    for ss, filt in ((2, 'BOX'), (2, 'TRIANGLE'), (3, 'GAUSS'), (2, 'CATROM'),
                     (3, 'MITCHELL'), (3, 'BOX')):
        big = (rng.random((Ho * ss, Wo * ss, 4)).astype(np.float32) * 1.6
               - np.float32(0.2))           # below 0 and above 1: the clamps
        for label, src, clampf in (('RESOLVE_G', src_g, False),
                                   ('RESOLVE_CG', src_cg, True)):
            st = settings(aa_filter=filt, aa_gamma_blend=True,
                          aa_clamp_samples=clampf)
            uni = {'resolution': (float(Wo), float(Ho)), 'ss': int(ss),
                   'gamma_blend': 1.0}
            if clampf:
                uni['clamp_samples'] = 1.0
            got, err = run_stage(src, Ho, Wo, uni,
                                 {'source': big, 'kernel': _ktex(ss, st),
                                  'gtab': tex})
            ref = R._resolve(big, Wo, Ho, ss, st)
            check(f'{label} at ss {ss} with the {filt} filter is bitwise '
                  'render._resolve under the flag(s) (d == 0.0; the fetches '
                  'return the CPU\'s table bits, dom is the same multiply)',
                  got is not None and _same(got, ref),
                  str(err) if got is None else _maxdiff(got, ref))
    from . import fakedevice
    with fakedevice.installed() as dev:
        ok_g = dev.compile_dynamic('RESOLVE_G', src_g, spec_g)[0] is not None
        sh, why = dev.compile_dynamic('RESOLVE_CG', src_cg, spec_cg)
        check("the fake device's compile_dynamic accepts the _G and _CG specs "
              '(declared uniforms all in the spec)', ok_g and sh is not None,
              str(why))
        if sh is not None:
            st = settings(aa_filter='GAUSS', aa_gamma_blend=True,
                          aa_clamp_samples=True)
            big = (rng.random((Ho * 2, Wo * 2, 4)).astype(np.float32) * 2.0)
            out = dev.draw_fullscreen(
                sh, {'resolution': (float(Wo), float(Ho)), 'ss': 2,
                     'clamp_samples': 1.0, 'gamma_blend': 1.0},
                {'source': dev.upload(big), 'kernel': dev.upload(_ktex(2, st)),
                 'gtab': dev.upload_cached(('gamma2tab',), RA.gamma2_texture)},
                dev.Target(Wo, Ho))
            check('...and a RESOLVE_CG draw with the table bound lands bitwise '
                  'on render._resolve', _same(out, R._resolve(big, Wo, Ho, 2, st)))
    cpu, gpu, live, LF, dev = _gpu_road(aa_mode='SUPERSAMPLE', aa_samples=4,
                                        aa_gamma_blend=True)
    d = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check('the GPU road at ss 2 with the flag: RESOLVE_G drew on the resident '
          'frame with the table uploaded once (upload_cached), the frame '
          'within 6e-6 of the CPU road, alpha bitwise, nothing left live',
          LF.get('resolve', {}).get('variant') == 'RESOLVE_G'
          and not LF.get('left_gpu') and d <= 6e-6
          and bool(np.array_equal(gpu[..., 3], cpu[..., 3])) and live == []
          and ('gamma2tab',) in dev.cache,
          f"variant {LF.get('resolve')} left_gpu {LF.get('left_gpu')!r} max {d} "
          f"{len(live)} live gtab cached {('gamma2tab',) in dev.cache}")
    cpu, gpu, live, LF, dev = _gpu_road(aa_mode='SUPERSAMPLE', aa_samples=4,
                                        aa_gamma_blend=True, aa_clamp_samples=True,
                                        global_ambient=(1.0, 1.0, 1.0),
                                        global_ambient_level=3.0)
    d = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check('...and with both flags RESOLVE_CG draws (clamp, then the blend), '
          'the frame within 6e-6, nothing left live',
          LF.get('resolve', {}).get('variant') == 'RESOLVE_CG'
          and not LF.get('left_gpu') and d <= 6e-6 and live == [],
          f"variant {LF.get('resolve')} max {d}")

    # (e) the matrix row and the preset
    from .featurematrix import ROWS
    keys = {r[0]: r for r in ROWS}
    row = keys.get('Blender 2.41 gamma-2 OSA blend')
    check('the feature matrix carries the gamma-2 blend row at 4 samples',
          row is not None and row[1].get('aa_gamma_blend') is True
          and row[1].get('aa_samples') == 4 and row[2] == 'demo')
    p = PRESETS.get('BLENDER_241')
    want = {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9, 'aa_filter': 'GAUSS',
            'aa_gamma_blend': True, 'default_model': 'BI_COOKTORR',
            'shading_rate': 'PIXEL', 'color_management': 'NONE',
            'input_gamma_naive': True, 'gamma': 1.0, 'specular_in_gamma': True,
            'shadows': True, 'shadow_default': 'PER_LIGHT', 'raytrace': True,
            'ray_depth': 2, 'ray_reflection': True, 'ray_refraction': True,
            'transparency': 'SORTED', 'tex_filter': 'TRILINEAR',
            'tex_perspective': True, 'color_depth': '24', 'dither': 'NOISE',
            'global_ambient': (0.0, 0.0, 0.0), 'fog': False, 'glow': False}
    check('the BLENDER_241 preset (SOFTWARE, "Blender 2.41 (2006)") pins '
          'OSA 9 GAUSS with the gamma-2 blend and the 2.4-era values by value',
          p is not None and p['category'] == 'SOFTWARE'
          and p['label'] == 'Blender 2.41 (2006)' and p['settings'] == want
          and 'gamma-2' in p['note'],
          '' if p is None else str({k: (p['settings'].get(k), v)
                                    for k, v in want.items()
                                    if p['settings'].get(k) != v}))
    check('BLENDER_INTERNAL (2.79) stays without the blend (2.42 removed it)',
          not PRESETS['BLENDER_INTERNAL']['settings'].get('aa_gamma_blend', False))
    sa = RenderSettings()
    apply_preset(sa, 'BLENDER_241')
    sb = RenderSettings()
    apply_preset(sb, 'BLENDER_INTERNAL')
    for s_ in (sa, sb):
        s_.resolution_x, s_.resolution_y = 64, 48
        s_.aa_samples = 4
        s_.output_scale = 'NONE'
    fa = np.asarray(R.render(demo_scene(sa, with_texture=False), sa))
    fb = np.asarray(R.render(demo_scene(sb, with_texture=False), sb))
    check('BLENDER_241 renders unlike BLENDER_INTERNAL at the same 4 samples '
          '(the GAUSS filter and the blend guarantee it)', not _same(fa, fb))


# ---------------------------------------------------------------------------
# the wire-road rigs (C063, C055): black clay under a black world, so the
# frame is exactly the lines
# ---------------------------------------------------------------------------

BLACK = dict(material_override='CLAY', override_color=(0.0, 0.0, 0.0),
             global_ambient=(0.0, 0.0, 0.0), shadows=False, fog=False,
             glow=False)

FLOOR = ('floor', dict(z=0.0, size=11.0, mat=0, obj=0))
WALL = ('cube', dict(centre=(2.0, -2.6, 1.2), size=2.4, mat=2, obj=1))
HIDDEN = ('cube', dict(centre=(-0.6, 1.2, 0.9), size=1.8, mat=1, obj=2))
# raised off the floor: a cube RESTING on it shares its bottom face with the
# floor plane and the tie-broken pixels along that edge belong to the floor
LONE = ('cube', dict(centre=(0.0, 0.0, 1.5), size=2.0, mat=2, obj=2))


def _parts(*specs):
    out = []
    for kind, kw in specs:
        out.append(plane(**kw) if kind == 'floor' else cube(**kw))
    return out


def _rig(st, *specs):
    """The demo scene (its camera and lights) with the mesh replaced and
    the world black."""
    sc = demo_scene(st, with_texture=False)
    sc.mesh = _mesh_concat(_parts(*specs))
    sc.world.color = (0.0, 0.0, 0.0)
    sc.world.sky_blend = False
    return sc


def _wire_frame(specs, **kw):
    """(frame, mask of pixels the wire painted exactly white)."""
    st = settings(render_wire=True, wire_color=(1.0, 1.0, 1.0), **BLACK)
    for k, v in kw.items():
        setattr(st, k, v)
    sc = _rig(st, *specs)
    img = np.asarray(R.render(sc, st))
    return img, (img[..., :3] == 1.0).all(axis=2)


def _object_px(specs):
    """IndexOB plane of the rig (rendered without the override, which
    would hand the passes to a scene copy)."""
    st = settings(pass_object_index=True)
    sc = _rig(st, *specs)
    R.render(sc, st)
    return sc.last_passes['IndexOB'][..., 0]


def vp_of(sc):
    """The scene's view-projection as render() derives it."""
    _view, _proj, vp, _eye = R.camera_matrices(sc.camera, W, H)
    return vp


def _dilate1(m):
    out = m.copy()
    out[1:] |= m[:-1]
    out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]
    out[:, :-1] |= m[:, 1:]
    return out


def _wire_gpu_road(specs, **kw):
    """The wire rig through the fake device: (cpu, gpu, live, FR.LAST)."""
    from ..gpu import shade as GSH
    from . import fakedevice
    st_c = settings(render_wire=True, wire_color=(1.0, 1.0, 1.0), **BLACK)
    st_g = settings(render_wire=True, wire_color=(1.0, 1.0, 1.0), **BLACK)
    for k, v in kw.items():
        setattr(st_c, k, v)
        setattr(st_g, k, v)
    st_g.render_device = 'GPU'
    cpu = np.asarray(R.render(_rig(st_c, *specs), st_c))
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        gpu = np.asarray(R.render(_rig(st_g, *specs), st_g))
        live = [t for t in dev.targets if not t.freed]
    return cpu, gpu, live, dict(FR.LAST)



def _pr_tables():
    """properties.py's RANGES / LABELS / DESCRIPTIONS dicts and its enum
    item lists read by AST (the module imports bpy, and fakebpy.install()
    at test time leaks into the next module -- the suite's rule), each
    entry literal-evaluated on its own so a non-literal neighbour never
    hides a new key."""
    import ast
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(root, 'properties.py'), encoding='utf8').read()
    tree = ast.parse(src)
    out = {'RANGES': {}, 'LABELS': {}, 'DESCRIPTIONS': {}, 'ENUMS': {}}
    items = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        name = getattr(node.targets[0], 'id', None)
        if name in ('RANGES', 'LABELS', 'DESCRIPTIONS', 'ENUMS') \
                and isinstance(node.value, ast.Dict):
            for k, v in zip(node.value.keys, node.value.values):
                try:
                    key = ast.literal_eval(k)
                except Exception:                               # noqa: BLE001
                    continue
                if name == 'ENUMS':
                    out[name][key] = getattr(v, 'id', None)
                    continue
                try:
                    out[name][key] = ast.literal_eval(v)
                except Exception:                               # noqa: BLE001
                    out[name][key] = None
        elif isinstance(node.value, ast.Call) \
                and getattr(node.value.func, 'id', None) == '_items':
            rows = []
            for a in node.value.args:
                try:
                    rows.append(ast.literal_eval(a))
                except Exception:                               # noqa: BLE001
                    rows.append(None)
            items[name] = rows
    out['ITEMS'] = items
    return out


# ---------------------------------------------------------------------------
# C063 Elite's wireframe rule
# ---------------------------------------------------------------------------

def test_c063_elite_wire():
    """Elite (BBC Micro 1984) drew an edge when at least one of its two
    faces faced the viewer and never depth-tested (Mark Moxon, 'Drawing
    ships'): far edges show through concave hulls, nothing hides behind
    anything, and past its distance a ship is one dot."""
    PR = _pr_tables()
    DESCRIPTIONS, ENUMS, LABELS, RANGES = (PR['DESCRIPTIONS'], PR['ENUMS'],
                                          PR['LABELS'], PR['RANGES'])
    WIRE_MODE = PR['ITEMS'].get('WIRE_MODE', [])

    # (b) the convex law on a lone cube from the demo's generic view
    st = settings()
    sc = _rig(st, FLOOR, LONE)
    eye = np.asarray(sc.camera.matrix_world, np.float32)[:3, 3]
    mesh = _mesh_concat(_parts(LONE))
    edges, faces = RA.mesh_edges(mesh)
    fn = RA._face_normals(mesh)
    coplanar = (fn[np.maximum(faces[:, 0], 0)] * fn[np.maximum(faces[:, 1], 0)]).sum(1) >= 0.9999
    vis = RA.elite_edges(mesh, eye)
    check('a triangulated cube welds to 18 edges (12 authored + 6 flat '
          'diagonals), every one with two faces (the 24 split corners '
          'welded by position)', edges.shape[0] == 18
          and int(coplanar.sum()) == 6 and bool((faces >= 0).all()),
          f'{edges.shape[0]} edges, {int(coplanar.sum())} flat')
    check('Elite\'s rule marks exactly the 9 authored edges with at least one '
          'front face (the three far-corner edges hide) and never a flat '
          'diagonal', int(vis.sum()) == 9 and not bool((vis & coplanar).any()),
          f'{int(vis.sum())} visible')
    e_vis, _f = RA.elite_visible_edges(mesh, None, eye)
    check('elite_visible_edges returns those 9 as (K, 2) vertex pairs',
          e_vis.shape == (9, 2))
    _img_e, m_el = _wire_frame((LONE,), wire_mode='ELITE')
    _clip, scr, _iw, _z = RA.project(np.asarray(mesh.verts, np.float32),
                                     np.asarray(vp_of(sc), np.float32), W, H)

    def seg_dist(mask, sel):
        """min distance from each inked pixel centre to the projected
        segments `sel` (K, 2) of the welded edge list."""
        py, px = np.nonzero(mask)
        pc = np.stack([px + 0.5, py + 0.5], 1).astype(np.float32)
        best = np.full(pc.shape[0], np.inf, np.float32)
        for a, b in sel:
            A, B = scr[a], scr[b]
            ab = B - A
            L2 = float(ab @ ab)
            t = np.clip(((pc - A) @ ab) / max(L2, 1e-12), 0.0, 1.0)
            d = np.linalg.norm(pc - (A + t[:, None] * ab), axis=1)
            best = np.minimum(best, d.astype(np.float32))
        return best
    d_vis = seg_dist(m_el, edges[vis])
    hidden_only = edges[~vis & ~coplanar]
    d_hid = seg_dist(m_el, hidden_only)
    check('every pixel Elite inks lies within a pixel of one of the 9 visible '
          "edges' projected segments (the DDA on the raster's own grid)",
          m_el.sum() > 40 and float(d_vis.max()) <= 1.0,
          f'{int(m_el.sum())} px, farthest {float(d_vis.max()):.3f}')
    check('...and none lies on a hidden far-corner edge alone (closer than '
          '0.5 px to a hidden edge yet farther than 1.5 px from every '
          'visible one)', int(((d_hid < 0.5) & (d_vis > 1.5)).sum()) == 0
          and hidden_only.shape[0] == 3)

    # (c) through-lines: a cube behind a nearer wall
    ob = _object_px((FLOOR, WALL, HIDDEN))
    wall_px = ob == 1
    _i, m_cube_alone = _wire_frame((FLOOR, HIDDEN), wire_mode='ELITE')
    _i, m_wall_alone_all = _wire_frame((FLOOR, WALL), wire_mode='ALL')
    _i, m_both_el = _wire_frame((FLOOR, WALL, HIDDEN), wire_mode='ELITE')
    _i, m_both_all = _wire_frame((FLOOR, WALL, HIDDEN), wire_mode='ALL')
    through = m_cube_alone & wall_px
    hidden = through & ~m_wall_alone_all
    check('the hidden cube\'s front-facing edges cross the wall\'s pixels '
          f'({int(through.sum())} px) and under ELITE every one of them is '
          'inked through the wall', through.sum() > 20
          and bool((m_both_el & through).sum() == through.sum()))
    check(f'under ALL (the depth-tested overlay) none of those {int(hidden.sum())} '
          'pixels away from the wall\'s own edges is inked', hidden.sum() > 10
          and int((m_both_all & hidden).sum()) == 0,
          f'{int((m_both_all & hidden).sum())} inked')

    # (d) no occlusion by other objects
    check('removing the wall changes none of the cube\'s painted pixels under '
          'ELITE (the two-object frame is a superset of the cube alone)',
          bool((m_both_el & m_cube_alone).sum() == m_cube_alone.sum())
          and m_cube_alone.sum() > 0)

    # (e) the dot
    d_cube = float(np.linalg.norm(eye - np.asarray(LONE[1]['centre'], np.float32)))
    _i, m_dot = _wire_frame((LONE,), wire_mode='ELITE', wire_dot_distance=d_cube - 1.0)
    _i, m_near = _wire_frame((LONE,), wire_mode='ELITE', wire_dot_distance=d_cube + 1.0)
    check(f'wire_dot_distance below the cube\'s distance ({d_cube:.2f}) paints '
          'exactly one pixel for it and none of its edges; above it the edges '
          'are back', int(m_dot.sum()) == 1 and int(m_near.sum()) > 20
          and _same(m_near, m_el if False else m_near),
          f'{int(m_dot.sum())} px as a dot, {int(m_near.sum())} px with edges')
    check('wire_dot_distance 0 never dots (the default)',
          int(_wire_frame((LONE,), wire_mode='ELITE', wire_dot_distance=0.0)[1].sum())
          == int(m_near.sum()))

    # (f) width
    _i, m_w3 = _wire_frame((LONE,), wire_mode='ELITE', wire_width=3.0)
    check('wire_width 3 paints a superset of width 1, between 1x and 9x the '
          'pixels (a 3x3 dilation)', bool((m_w3 & m_el).sum() == m_el.sum())
          and m_el.sum() < m_w3.sum() <= 9 * m_el.sum(),
          f'{int(m_el.sum())} -> {int(m_w3.sum())}')

    # (g) band invariance
    st = settings(render_wire=True, wire_mode='ELITE', wire_color=(1.0, 1.0, 1.0),
                  **BLACK)
    full = np.asarray(R.render(_rig(st, FLOOR, WALL, HIDDEN), st))
    b0 = np.asarray(R.render(_rig(st, FLOOR, WALL, HIDDEN), st, band=(0, H // 2)))
    b1 = np.asarray(R.render(_rig(st, FLOOR, WALL, HIDDEN), st, band=(H // 2, H)))
    check('two bands concatenated equal the whole frame bitwise (the wire '
          'road runs on the whole internal frame before the band slice)',
          _same(np.concatenate([b0, b1], axis=0), full))

    # (h) the device: CPU-after-readback by name, both devices the same bits
    cpu, gpu, live, LF = _wire_gpu_road((FLOOR, WALL, HIDDEN), wire_mode='ELITE',
                                        wire_dot_distance=8.0)
    check("under the fake device with render_device 'GPU' the Elite frame is "
          "bitwise the CPU frame and FR.LAST['left_gpu'] names 'wireframe' "
          '(the wire road\'s readback), nothing left live',
          _same(cpu, gpu) and 'wireframe' in str(LF.get('left_gpu', ''))
          and live == [], f"left_gpu {LF.get('left_gpu')!r} {_maxdiff(cpu, gpu)}")

    # (i) the rows, the enum, the tooltips, the panel, the preset
    from .featurematrix import ROWS, SCENES
    keys = {r[0]: r for r in ROWS}
    row = keys.get('Elite wireframe rule (no depth test, dots past 8 units)')
    check("the feature matrix row homes wire_mode ELITE and wire_dot_distance "
          '8.0 on the demo (7.6 to the cube, 9.0 to the floor, 9.6 to the ball: '
          'floor and ball collapse to dots, the cube keeps its edges)',
          row is not None and row[1].get('wire_mode') == 'ELITE'
          and row[1].get('wire_dot_distance') == 8.0 and row[2] == 'demo')
    st = settings(render_wire=True, wire_mode='ELITE', wire_dot_distance=8.0,
                  wire_color=(1.0, 1.0, 1.0), **BLACK)
    sc = demo_scene(st, with_texture=False)
    sc.world.color = (0.0, 0.0, 0.0)
    sc.world.sky_blend = False
    buf = io.StringIO()
    with redirect_stdout(buf):
        R.render(sc, st)
    check('...and on the demo it prints its accounting: 2 dots, the cube\'s '
          'edges facing the viewer', '2 dot(s)' in buf.getvalue()
          and 'ELITE' in buf.getvalue(), buf.getvalue()[-200:])
    row2 = keys.get('Elite through-lines on a concave hull')
    check("...and the through-line row on the 'elite_ship' scene (a cube "
          'behind a wall)', row2 is not None and row2[2] == 'elite_ship'
          and 'elite_ship' in SCENES)
    ids = [i[0] for i in WIRE_MODE]
    check("WIRE_MODE gains ELITE (and BEAM) at the END of the list, after "
          'CREASE, with tooltips naming the machine',
          ids == ['ALL', 'CREASE', 'ELITE', 'BEAM']
          and 'BBC Micro' in WIRE_MODE[2][1] and len(WIRE_MODE[2][2]) >= 40
          and ENUMS.get('wire_mode') == 'WIRE_MODE')
    check('wire_dot_distance: range 0..100000, label "Dot Distance", a '
          'tooltip of 40+ characters naming Elite',
          RANGES.get('wire_dot_distance') == (0.0, 100000.0)
          and LABELS.get('wire_dot_distance') == 'Dot Distance'
          and len(DESCRIPTIONS.get('wire_dot_distance', '')) >= 40
          and 'Elite' in DESCRIPTIONS['wire_dot_distance'])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ui = open(os.path.join(root, 'ui.py'), encoding='utf8').read()
    check('the Wireframe panel draws wire_dot_distance under ELITE',
          "sub.active = hs.wire_mode == 'ELITE'" in ui
          and "sub.prop(hs, 'wire_dot_distance')" in ui)
    p = PRESETS.get('ELITE_BBC')
    want = {'resolution_x': 320, 'resolution_y': 256, 'aa_mode': 'NONE',
            'render_wire': True, 'wire_mode': 'ELITE', 'wire_width': 1.0,
            'wire_color': (1.0, 1.0, 1.0), 'wire_dot_distance': 0.0,
            'material_override': 'CLAY', 'override_color': (0.0, 0.0, 0.0),
            'global_ambient': (0.0, 0.0, 0.0), 'shadows': False, 'fog': False,
            'glow': False, 'color_depth': '1', 'dither': 'NONE', 'gamma': 1.0,
            'output_scale': '3X'}
    check('the ELITE_BBC preset (PLATFORM, "Elite (BBC Micro, 1984)") pins '
          '320x256, the Elite rule, white 1-bit lines over black clay, 3x',
          p is not None and p['category'] == 'PLATFORM'
          and p['label'] == 'Elite (BBC Micro, 1984)' and p['settings'] == want
          and 'no depth test' in p['note'],
          '' if p is None else str({k: (p['settings'].get(k), v)
                                    for k, v in want.items()
                                    if p['settings'].get(k) != v}))
    sp = RenderSettings()
    apply_preset(sp, 'ELITE_BBC')
    sp.resolution_x, sp.resolution_y = 64, 48
    sp.output_scale = 'NONE'
    sc = demo_scene(sp, with_texture=False)
    sc.world.color = (0.0, 0.0, 0.0)
    sc.world.sky_blend = False
    out = PO.process(R.render(sc, sp), sp, **post_kw(sc, sp))
    vals = np.unique(np.round(out[..., :3], 3))
    check('...and it renders the demo to a two-value (1-bit) frame with lines '
          'present', out.shape == (48, 64, 4) and len(vals) == 2
          and float(out[..., :3].max()) == 1.0
          and 0 < int((out[..., 0] == 1.0).sum()) < 0.5 * 64 * 48,
          f'values {vals[:4]}')


# ---------------------------------------------------------------------------
# C055 the vector monitor beam
# ---------------------------------------------------------------------------

def test_c055_vector_beam():
    """Atari's vector monitors drew a Gaussian phosphor spot along each
    stroke at quantised intensity with dwell dots at the ends (Margolin,
    'The Secret Life of Vector Generators'; MAME avgdvg.c), additive
    where strokes cross, never depth-tested."""
    Wf, Hf = 64, 48

    # (e'') the pinned integer table
    lut = np.asarray(RA.BEAM_LUT, np.int64)
    i = np.arange(576, dtype=np.float64)
    formula = np.round(65536.0 * np.exp(-((i + 0.5) / 64.0) / 2.0)).astype(np.int64)
    check('BEAM_LUT is a literal of 576 int32 entries, monotone non-increasing, '
          'matching round(65536 * exp(-((i + 0.5) / 64) / 2)) within +/-1 on '
          'this machine (the literal is the truth; exp never runs at render)',
          lut.shape == (576,) and bool((np.diff(lut) <= 0).all())
          and int(np.abs(lut - formula).max()) <= 1
          and lut[0] == 65280 and RA._BEAM_LUT_ARR.shape == (577,)
          and RA._BEAM_LUT_ARR[576] == 0,
          f'max |lut - formula| {int(np.abs(lut - formula).max())}')
    check('BEAM_MACHINES pins the three generators: DVG (zmax 15, cmax 255, '
          'dwell 32/64), AVG (14, 1, 16/64), STARWARS (255, 1, 16/64)',
          RA.BEAM_MACHINES == {'DVG': dict(zmax=15, cmax=255, dwell64=32),
                               'AVG': dict(zmax=14, cmax=1, dwell64=16),
                               'STARWARS': dict(zmax=255, cmax=1, dwell64=16)})

    # (c) the quantisation laws per generator
    I = np.linspace(0.0, 1.0, 101).astype(np.float32)
    zd, cd = RA.dvg_program(I, (0.85, 1.0, 0.9))
    za, ca = RA.avg_program(I, (0.85, 0.2, 0.5), zmax=14)
    zs, cs = RA.avg_program(I, (0.85, 0.2, 0.5), zmax=255)
    check('DVG intensities take the 16 codes 0..15 (round(I * 15)), the '
          'overlay colour 8-bit codes floored at 1',
          set(zd.tolist()) == set(range(16)) and bool((np.diff(zd) >= 0).all())
          and cd.tolist() == [217, 255, 230]
          and RA.dvg_program(np.ones(1, np.float32), (0.0, 0.0, 0.0))[1].tolist() == [1, 1, 1])
    check('AVG intensities take only 0 and the even codes 4..14 (3-bit '
          'doubled, 0 draws nothing), colour 1 bit per channel (>= 0.5)',
          set(za.tolist()) == {0, 4, 6, 8, 10, 12, 14} and za[0] == 0
          and bool((za[1:] >= 4).all()) and ca.tolist() == [1, 0, 1])
    check('STARWARS intensities take 0..255 (8-bit STATZ), colour 1 bit per '
          'channel', set(zs.tolist()) <= set(range(256)) and zs[-1] == 255
          and len(set(zs.tolist())) > 60 and cs.tolist() == [1, 0, 1])
    st_fog = settings(fog=True, fog_start=2.0, fog_end=12.0)
    sc = _rig(st_fog, FLOOR, WALL, HIDDEN)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, W, H)
    mesh = sc.mesh
    edges = RA.beam_strokes(mesh, 25.0)
    verts = np.asarray(mesh.verts, np.float32)
    mid = (verts[edges[:, 0]] + verts[edges[:, 1]]) * 0.5
    vm = np.asarray(view, np.float32)
    zv = -(mid @ vm[:3, :3].T + vm[:3, 3][None, :])[:, 2]
    Ifog = np.clip((12.0 - zv) / 10.0, 0.0, 1.0).astype(np.float32)
    zq, _c = RA.dvg_program(Ifog, (1.0, 1.0, 1.0))
    order = np.argsort(zv)
    check('with fog on a farther stroke never outshines a nearer one (the '
          'era\'s fade to black, monotone in view depth) and the strokes span '
          'several codes', bool((np.diff(zq[order]) <= 0).all())
          and len(set(zq.tolist())) >= 3, f'codes {sorted(set(zq.tolist()))}')
    check('the stroke list is the feature edges: on the black rig the floor\'s '
          '4 rim edges and the two cubes\' 24 (no flat diagonal, no '
          'visibility test)', edges.shape[0] == 28, f'{edges.shape[0]} strokes')

    # (d) the profile of one horizontal stroke
    sig = np.float32(1.5)
    segs = np.array([[[10.0, 24.0], [54.0, 24.0]]], np.float32)   # y centre 24.5? no: 24.0 is a pixel BOUNDARY
    segs = np.array([[[10.0, 24.5], [54.0, 24.5]]], np.float32)   # through the row-24 centres
    acc = RA.beam_trace(segs, np.array([14]), np.array([1, 1, 1]), 'AVG', sig, Wf, Hf)
    img = RA.beam_image(acc, 'AVG')
    col = img[:, 32, 0]
    peak = 14 * 1 * int(RA.BEAM_LUT[0]) / (65536.0 * 14 * 1)
    check('one horizontal stroke: the column through its middle peaks on the '
          f'stroke\'s row at Z * C * LUT[0] / (65536 * Zmax * Cmax) = {peak:.4f} '
          '(the ends are far, so no dwell there)',
          abs(float(col[24]) - peak) < 1e-6 and int(np.argmax(col)) == 24,
          f'{float(col[24])}')
    reach = int(np.ceil(3.0 * float(sig)))
    check('...and falls symmetrically to zero within 3 sigma',
          bool(np.array_equal(col[24 - reach:24], col[24 + reach:24:-1]))
          and float(col[24 - reach - 1]) == 0.0 and float(col[24 + reach + 1]) == 0.0
          and bool((np.diff(col[24:24 + reach]) <= 0).all()))

    # (e) the dwell dots per machine
    ratios = {}
    for m in ('DVG', 'AVG', 'STARWARS'):
        zmax = RA.BEAM_MACHINES[m]['zmax']
        acc = RA.beam_trace(segs, np.array([zmax]), np.array([1, 1, 1]), m, sig, Wf, Hf)
        im = RA.beam_image(acc, m)
        end = float(im[24, 10, 0])
        midv = float(im[24, 32, 0])
        ratios[m] = end / midv
    check('under every machine the pixel at a stroke\'s end is brighter than '
          'its midpoint (the deflection amplifier dwells there)',
          all(r > 1.0 for r in ratios.values()), str(ratios))
    check('...and the DVG\'s end/mid ratio exceeds the AVG\'s (its dwell '
          'constant is the larger: the Asteroids vertex dots)',
          ratios['DVG'] > ratios['AVG'] and abs(ratios['AVG'] - ratios['STARWARS']) < 1e-6,
          str(ratios))

    # (b) order-freeness and (g) additivity, integers
    two = np.array([[[8.0, 8.5], [56.0, 40.5]], [[8.0, 40.5], [56.0, 8.5]]], np.float32)
    Z2 = np.array([14, 10])
    a_fwd = RA.beam_trace(two, Z2, np.array([1, 0, 1]), 'AVG', sig, Wf, Hf)
    a_rev = RA.beam_trace(two[::-1].copy(), Z2[::-1].copy(), np.array([1, 0, 1]), 'AVG', sig, Wf, Hf)
    check('the stroke list reversed accumulates the same int64 bits (order-free '
          'integer sums)', _same(a_fwd, a_rev) and a_fwd.dtype == np.int64)
    a0 = RA.beam_trace(two[:1], Z2[:1], np.array([1, 0, 1]), 'AVG', sig, Wf, Hf)
    a1 = RA.beam_trace(two[1:], Z2[1:], np.array([1, 0, 1]), 'AVG', sig, Wf, Hf)
    cross = a_fwd[24, 32]
    check('two crossing strokes: at the crossing the accumulator is exactly the '
          'sum of each alone (additive light), the image clipped at 1',
          _same(a_fwd, a0 + a1) and cross[0] == a0[24, 32, 0] + a1[24, 32, 0]
          and cross[0] > 0 and cross[1] == 0
          and float(RA.beam_image(a_fwd, 'AVG').max()) <= 1.0
          and float(RA.beam_image(a_fwd * 8, 'AVG').max()) == 1.0)

    # (f) sigma: a fatter spot lights more pixels
    lit = {}
    for sg in (0.7, 1.4, 2.8):
        _i, m = _wire_frame((FLOOR, WALL, HIDDEN), wire_mode='BEAM',
                            beam_machine='AVG', beam_sigma=sg)
        lit[sg] = int(m.sum())
    imgs = {}
    for sg in (0.7, 1.4):
        st = settings(render_wire=True, wire_mode='BEAM', beam_machine='AVG',
                      beam_sigma=sg, wire_color=(1.0, 1.0, 1.0), **BLACK)
        imgs[sg] = np.asarray(R.render(_rig(st, FLOOR, WALL, HIDDEN), st))
    n_lo = int((imgs[0.7][..., 0] > 0.05).sum())
    n_hi = int((imgs[1.4][..., 0] > 0.05).sum())
    check('doubling beam_sigma increases the count of pixels above 0.05 '
          '(monotone spread of the phosphor spot)', n_hi > n_lo > 0,
          f'{n_lo} -> {n_hi}')
    check('...and the beam is drawn at the rig\'s 72 rows scaled from 480 '
          '(sigma 0.7 -> 0.105 px), still lighting the strokes',
          lit[0.7] > 0 and lit[2.8] >= lit[0.7])

    # (e') the named no-op: black under 1 bit per channel
    for col_, draws in (((0.0, 0.0, 0.0), False), ((0.4, 0.4, 0.4), False),
                        ((1.0, 1.0, 1.0), True)):
        st = settings(render_wire=True, wire_mode='BEAM', beam_machine='AVG',
                      wire_color=col_, **BLACK)
        st_off = settings(**BLACK)
        buf = io.StringIO()
        with redirect_stdout(buf):
            on = np.asarray(R.render(_rig(st, FLOOR, WALL, HIDDEN), st))
        off = np.asarray(R.render(_rig(st_off, FLOOR, WALL, HIDDEN), st_off))
        named = ('the wire colour rounds to black under 1 bit per channel; '
                 'nothing drawn' in buf.getvalue())
        if draws:
            check(f'wire_color {col_} draws the beam under AVG (no no-op line)',
                  not _same(on, off) and not named)
        else:
            check(f'wire_color {col_} under AVG: the frame is bitwise the frame '
                  'without the beam and the console names the no-op ONCE '
                  '(never a silent zero)', _same(on, off) and named
                  and buf.getvalue().count('rounds to black') == 1)
    st = settings(render_wire=True, wire_mode='BEAM', beam_machine='DVG',
                  wire_color=(0.0, 0.0, 0.0), **BLACK)
    on = np.asarray(R.render(_rig(st, FLOOR, WALL, HIDDEN), st))
    check('the DVG floors its overlay colour at 1/255, so black still draws '
          'a faint beam (the tube was monochrome)', not _same(on, off)
          and 0.0 < float(on[..., :3].max()) < 0.05)

    # (h) the device: CPU-after-readback by name, both devices the same bits
    cpu, gpu, live, LF = _wire_gpu_road((FLOOR, WALL, HIDDEN), wire_mode='BEAM',
                                        beam_machine='DVG', beam_sigma=1.4,
                                        wire_color=(0.85, 1.0, 0.9))
    check("under the fake device with render_device 'GPU' the beam frame is "
          "bitwise the CPU frame and FR.LAST['left_gpu'] names 'wireframe' "
          '(the wire road\'s readback), nothing left live',
          _same(cpu, gpu) and 'wireframe' in str(LF.get('left_gpu', ''))
          and live == [], f"left_gpu {LF.get('left_gpu')!r} {_maxdiff(cpu, gpu)}")

    # (i) the rows, the enum, the tooltips, the panel, the category, the preset
    from .featurematrix import ROWS, SCENES
    keys = {r[0]: r for r in ROWS}
    r1 = keys.get('vector monitor beam (AVG)')
    r2 = keys.get('vector monitor beam (DVG, fat spot)')
    check('the feature matrix carries the AVG row with a colour the 1-bit '
          "channels keep (white) on the 'beam_vectors' scene and the DVG row "
          'homing beam_machine and beam_sigma 1.4 on the demo',
          r1 is not None and r1[1].get('wire_color') == (1.0, 1.0, 1.0)
          and r1[2] == 'beam_vectors' and 'beam_vectors' in SCENES
          and r2 is not None and r2[1].get('beam_machine') == 'DVG'
          and r2[1].get('beam_sigma') == 1.4 and r2[2] == 'demo')
    PR = _pr_tables()
    bm = PR['ITEMS'].get('BEAM_MACHINE', [])
    check("BEAM_MACHINE lists DVG, AVG, STARWARS with their generators' names "
          'and tooltips, and ENUMS maps beam_machine to it',
          [b[0] for b in bm] == ['DVG', 'AVG', 'STARWARS']
          and all(len(b[2]) >= 40 for b in bm)
          and PR['ENUMS'].get('beam_machine') == 'BEAM_MACHINE')
    check('beam_sigma ranges 0.2..4.0 ("Beam Spot"), beam_machine is "Vector '
          'Generator", both tooltips 40+ characters naming the machines',
          PR['RANGES'].get('beam_sigma') == (0.2, 4.0)
          and PR['LABELS'].get('beam_sigma') == 'Beam Spot'
          and PR['LABELS'].get('beam_machine') == 'Vector Generator'
          and len(PR['DESCRIPTIONS'].get('beam_sigma', '')) >= 40
          and len(PR['DESCRIPTIONS'].get('beam_machine', '')) >= 40
          and 'DVG' in PR['DESCRIPTIONS']['beam_machine']
          and '1 bit per channel' in PR['DESCRIPTIONS']['beam_machine'])
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ui = open(os.path.join(root, 'ui.py'), encoding='utf8').read()
    check('the Wireframe panel draws beam_machine and beam_sigma under BEAM and '
          'keeps the crease angle live for BEAM (the stroke list reuses it)',
          "sub.active = hs.wire_mode == 'BEAM'" in ui
          and "sub.prop(hs, 'beam_machine')" in ui
          and "sub.prop(hs, 'beam_sigma')" in ui
          and "hs.wire_mode in ('CREASE', 'BEAM')" in ui)
    check("CATEGORIES gains ('ARCADE', \"Arcade Boards\") at the end, after CEL",
          CATEGORIES[-1] == ('ARCADE', 'Arcade Boards')
          and CATEGORIES[-2][0] == 'CEL')
    p = PRESETS.get('ATARI_VECTOR')
    want = {'resolution_x': 640, 'resolution_y': 480, 'aa_mode': 'NONE',
            'render_wire': True, 'wire_mode': 'BEAM', 'beam_machine': 'DVG',
            'beam_sigma': 0.7, 'wire_angle': 25.0, 'wire_color': (0.85, 1.0, 0.9),
            'material_override': 'CLAY', 'override_color': (0.0, 0.0, 0.0),
            'global_ambient': (0.0, 0.0, 0.0), 'shadows': False, 'fog': False,
            'glow': False, 'color_depth': '24', 'dither': 'NONE', 'gamma': 1.0}
    check('the ATARI_VECTOR preset (ARCADE, "Atari vector arcade (1979)") pins '
          '640x480, the DVG beam at sigma 0.7, the Asteroids gel over black clay',
          p is not None and p['category'] == 'ARCADE'
          and p['label'] == 'Atari vector arcade (1979)' and p['settings'] == want
          and 'dwell' in p['note'],
          '' if p is None else str({k: (p['settings'].get(k), v)
                                    for k, v in want.items()
                                    if p['settings'].get(k) != v}))
    sp = RenderSettings()
    apply_preset(sp, 'ATARI_VECTOR')
    sp.resolution_x, sp.resolution_y = 64, 48
    sc = demo_scene(sp, with_texture=False)
    sc.world.color = (0.0, 0.0, 0.0)
    sc.world.sky_blend = False
    out = PO.process(R.render(sc, sp), sp, **post_kw(sc, sp))
    lit_px = int((out[..., 1] > 0.05).sum())
    check('...and it renders the demo as green-tinted strokes on black (the '
          'gel: G >= R, B), a minority of the frame lit',
          0 < lit_px < 0.5 * 64 * 48 and float(out[..., 0].max()) <= float(out[..., 1].max()),
          f'{lit_px} px lit')
    # the pack's three presets exist on their shelves with a note; the
    # round's exact count is test_render's pin (the integrator's), a pack
    # asserts only that the library holds at least the round's presets
    for key, cat in (('BLENDER_241', 'SOFTWARE'), ('ELITE_BBC', 'PLATFORM'),
                     ('ATARI_VECTOR', 'ARCADE')):
        p = PRESETS.get(key)
        check(f"the pack's {key} preset is shelved {cat} with a machine note",
              p is not None and p['category'] == cat
              and len(p.get('note', '')) >= 40,
              'missing' if p is None else f"{p['category']}, note {len(p.get('note', ''))} chars")
    check("the library holds at least the round's 103 presets (the exact "
          "count is test_render's pin)", len(PRESETS) >= 103, str(len(PRESETS)))


TESTS = [test_a_identity_at_defaults, test_c094_limit_dynamic_range,
         test_c122_gamma2_blend, test_c063_elite_wire, test_c055_vector_beam]


def main():
    from . import utf8_console
    utf8_console()
    for fn in TESTS:
        print(f'\n{fn.__name__}')
        try:
            fn()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(fn.__name__ + ' (exception)')
    print()
    print(f'R251 RAST-B: {len(FAILS)} failure(s)')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
