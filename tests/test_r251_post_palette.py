"""R251 post-palette (PAL-1) tests: P0 the palette snap on the GPU, P1 the
ordered dither to bits on the GPU, C061 palette register depth, C054
Extra Half-Brite, C011 CRY16 (Jaguar), C059 YJK (MSX2+).

Run with:  python -m halcyon.tests.test_r251_post_palette

The FIRST test pins the 1.89.0 zip loudly and the pack's identity at the
defaults (render AND post.process bitwise the previous release); the
SECOND compiles every stage source through the front-end; the THIRD pins
the chain <-> chain_palette import direction. Then one test per feature
in the wave entry's order, each proving: the semantic A/B law on the CPU
function, the GPU twin d == 0.0 in the simulator, the refusal by name,
and the fake-device road (render + post over the same frame, the chain's
record as named, nothing left live).
"""

import importlib
import sys
import traceback

import numpy as np

from ..core import dither as DI
from ..core import palette as PA
from ..core import palette_era as PE
from ..core import post as PO
from ..core import render as R
from ..core.settings import RenderSettings
from ..core.texture import Texture
from ..gpu import chain
from ..gpu import chain_palette as CP
from ..gpu import frame as FR
from ..gpu import shade as GSH
from ..gpu import stages
from ..gpu import stages_palette as SP
from ..presets.library import PRESETS, apply_preset
from ..shaders.compiler import try_compile
from . import fakedevice
from . import featurematrix as FM
from .scenebuild import demo_scene
from .r251_common import clear_palette_locks
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


# ------------------------------------------------------------------ helpers

W, H = 96, 72


def settings(**kw):
    """The TR:37484-37490 shape: 96 x 72, no shadows, no transparency."""
    st = base_settings(W, H)
    st.shadows = False
    st.transparency = 'NONE'
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def preset_settings(key):
    """The TR:37492-37497 shape with the key a parameter: apply_preset sets
    every preset key the settings object has (320x240 included), then the
    96 x 72 / aa_samples 1 reset."""
    st = base_settings(W, H)
    apply_preset(st, key)
    st.resolution_x, st.resolution_y = W, H
    st.aa_samples = 1
    return st


def post_kw(sc, st):
    return dict(frame=7, seed=st.seed, target_size=(W, H),
                allow_resize=False,
                depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None),
                flare_sources=getattr(sc, 'last_flares', None))


def gpu_road(st):
    """TR:37506-37524 copied: render + post on the fake device, then the
    CPU chain over the SAME frame; returns (out_gpu, out_cpu, record,
    live targets, the fake device)."""
    st.render_device = 'GPU'
    sc = demo_scene(st, with_texture=False)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        st._keep_gpu_frame = True
        try:
            img = R.render(sc, st)
            out = PO.process(img, st, **post_kw(sc, st))
            rec = dict(PO.LAST_CHAIN)
        finally:
            FR.release(st)
        live = [t for t in dev.targets if not t.freed]
    st_c = st.copy()
    st_c.render_device = 'CPU'
    out_c = PO.process(img, st_c, **post_kw(sc, st_c))
    return out, out_c, rec, live, dev


def fake_row(label, st, tol, want):
    """One fake-device row in the TR:37526-37576 shape; returns the device
    (its cache is inspected by C061) and the record."""
    out_g, out_c, rec, live, dev = gpu_road(st)
    d = float(np.abs(out_g - out_c).max()) if out_g.shape == out_c.shape \
        else float('inf')
    ok = d <= tol and live == []
    why = [f'max {d}', f'{len(live)} live', str(rec)]
    if 'resident' in want:
        ok &= rec.get('resident') is want['resident']
    if 'stages' in want:
        ok &= rec.get('stages') == want['stages']
    if 'stages_has' in want:
        ok &= all(s in (rec.get('stages') or []) for s in want['stages_has'])
    if 'readbacks' in want:
        ok &= len(rec.get('readbacks') or []) == want['readbacks']
    if 'readback_names' in want:
        rb = rec.get('readbacks') or []
        ok &= len(rb) == len(want['readback_names']) and all(
            nm in r[0] for nm, r in zip(want['readback_names'], rb))
    if 'uploads' in want:
        ok &= rec.get('uploads') == want['uploads']
    check(f'post on the resident frame, {label}: the GPU chain equals the '
          'CPU chain over the same frame '
          + ('bitwise' if tol == 0.0 else f'within {tol:g}')
          + ', its record as named, nothing left live', ok, '; '.join(why))
    return dev, rec


def run_stage(name, h, w, uniforms, samplers):
    """TR:37597-37617 copied: one post stage through the GLSL simulator."""
    src = stages.STAGES[name].replace('in vec2 vUV;', 'uniform vec2 vUV;')
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


def twin(name, ref, h, w, uniforms, samplers):
    """(d, err, alpha_ok): the simulator's stage against the CPU
    reference; d == 0.0 is the bitwise bar."""
    got, err = run_stage(name, h, w, uniforms, samplers)
    if got is None:
        return -1.0, err, False
    d = float(np.abs(got[..., :3] - ref).max())
    if not np.array_equal(got[..., :3], ref):
        d = max(d, np.finfo(np.float32).tiny)
    return d, None, bool((got[..., 3] == 1.0).all())


def palette_inputs(st, pal, h, w):
    """The PALETTE / EHB stage transports the chain builds, from the
    pack's own functions."""
    pal = np.ascontiguousarray(np.asarray(pal, np.float32).reshape(-1, 3))
    kind = str(st.dither)
    m = DI.ORDERED.get(kind)
    tile = CP.tile_image(m) if m is not None else np.zeros((1, 1, 4), np.float32)
    mask = int(m.shape[0] - 1) if m is not None else 0
    uni = {'resolution': (float(w), float(h)),
           'strength': float(np.float32(float(st.dither_strength))),
           'spacing': float(np.float32(DI._palette_spacing(pal))),
           'tile_mask': mask, 'dither_on': int(m is not None)}
    return uni, tile


def members(out, pal):
    """Every pixel of `out` is a row of `pal` (exact membership)."""
    p = np.asarray(pal, np.float32).reshape(-1, 3)
    rows = {r.tobytes() for r in p}
    flat = np.ascontiguousarray(np.asarray(out, np.float32).reshape(-1, 3))
    return all(px.tobytes() in rows for px in flat)


def fm_row(key):
    return any(r[0] == key for r in FM.ROWS)


def _properties():
    """halcyon/properties.py under the fake bpy (the suite's own road)."""
    from . import fakebpy
    fakebpy.install()
    return importlib.import_module(__package__.rsplit('.tests', 1)[0]
                                   + '.properties')


_RNG = np.random.default_rng(251)


def random_image(h=48, w=64):
    return (_RNG.random((h, w, 3)).astype(np.float32) * 0.9 + 0.05
            ).astype(np.float32)


# ------------------------------------------------- 1. identity at defaults


def test_identity_at_defaults():
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the neutrality pin runs)',
          RP is not None,
          '' if RP is not None else 'PIN SKIPPED: keep halcyon-1.89.0.zip beside halcyon/')
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')
    cases = [
        ('the defaults', {}),
        ('VGA256 at 8 bits (P0 road)', {'color_depth': '8',
                                        'palette_mode': 'VGA256'}),
        ('EGA16 + BAYER4 at 4 bits (P0 ordered road)',
         {'color_depth': '4', 'palette_mode': 'EGA16', 'dither': 'BAYER4'}),
        ("16 bits + BAYER4 (P1 road)", {'color_depth': '16',
                                        'dither': 'BAYER4'}),
        ('16 adaptive at 4 bits (the ATARI_ST shape, C061 at NONE)',
         {'color_depth': '4', 'palette_mode': 'ADAPTIVE', 'palette_size': 16}),
        ('HAM6', {'color_depth': 'HAM6'}),
        ('HAM8', {'color_depth': 'HAM8'}),
        ('exposure 0.1', {'exposure': 0.1}),
        ('saturation 2.0', {'saturation': 2.0}),
        ('the CEL_ANIME_MODERN preset', 'preset'),
    ]
    for label, kw in cases:
        # both engines' adaptive palette locks, not only this tree's: the
        # lock is per-engine session state (r251_common.clear_palette_locks)
        clear_palette_locks(RP)
        st = preset_settings('CEL_ANIME_MODERN') if kw == 'preset' \
            else settings(**kw)
        sc = demo_scene(st, with_texture=False)
        now_r = np.asarray(R.render(sc, st))
        now = PO.process(now_r, st, **post_kw(sc, st))
        st2 = preset_settings('CEL_ANIME_MODERN') if kw == 'preset' \
            else settings(**kw)
        sc2 = demo_scene(st2, with_texture=False)
        prev_r = np.asarray(RP.render(sc2, st2))
        prev = prev_post.process(prev_r, st2, **post_kw(sc2, st2))
        check(f'{label}: the render is bitwise the 1.89.0 engine given the '
              'SAME (current) settings object',
              now_r.shape == prev_r.shape and bool(np.array_equal(now_r, prev_r)),
              f'max {float(np.abs(now_r - prev_r).max()) if now_r.shape == prev_r.shape else "shape"}')
        check(f"{label}: post.process is bitwise the 1.89.0 engine's "
              '(each engine from a cleared palette lock)',
              now.shape == prev.shape and bool(np.array_equal(now, prev)),
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    st = settings()
    check('every new setting defaults to the no-op value (palette_bits NONE; '
          "color_depth / palette_mode untouched)",
          st.palette_bits == 'NONE' and st.color_depth == '24'
          and st.palette_mode == 'ADAPTIVE')
    check('the era dispatcher answers None at the defaults and for every '
          '1.89.0 palette mode',
          PE.reduce_depth_era(random_image(8, 8), settings(), 0) is None
          and all(PE.reduce_depth_era(random_image(8, 8),
                                      settings(color_depth='8', palette_mode=m),
                                      0) is None
                  for m in ('ADAPTIVE', 'VGA256', 'MAC256', 'WEB216',
                            'FIXED_666', 'WIN20', 'EGA16', 'CGA4', 'GRAY',
                            'CUSTOM')))


# ------------------------------------------- 2. every stage source compiles


def test_stage_sources_compile():
    for name, src in SP.STAGES.items():
        prog, err = try_compile(src, 'GLSL')
        check(f'the {name} stage source compiles through the GLSL front-end',
              prog is not None, str(err or ''))
        check(f'the {name} stage is registered in gpu/stages and ENABLED '
              '(EXACT grade)',
              stages.STAGES.get(name) == src and name in stages.INTERFACE
              and stages.VALIDATION.get(name, ('?',))[0]
              == ('CLOSE' if name == 'LEGALISE' else 'EXACT')   # wave 2: one sqrt, one division
              and name in stages.ENABLED and name in chain.ENABLED)
        heads = [ln for ln in src.splitlines() if ln.strip().startswith('uniform')]
        check(f'the {name} stage declares one uniform per line (the fake '
              'device reads the first per line)',
              all(ln.count('uniform') == 1 for ln in heads))
        spec = stages.INTERFACE[name]
        declared = {ln.split()[2].rstrip(';') for ln in heads}
        listed = set()
        for kind in ('samplers', 'floats', 'ints', 'vec2', 'vec3'):
            listed |= set(spec.get(kind, []))
        check(f'the {name} INTERFACE names exactly the declared uniforms',
              declared == listed, f'{sorted(declared ^ listed)}')
    check('the unwired DITHER stage is retired (P1): not in STAGES, INTERFACE '
          'or VALIDATION',
          'DITHER' not in stages.STAGES and 'DITHER' not in stages.INTERFACE
          and 'DITHER' not in stages.VALIDATION and 'ORDERED' in stages.STAGES)


# ----------------------------------------- 3. the chain import direction


def test_chain_imports_stand_alone():
    global chain, CP
    pkg = __package__.rsplit('.tests', 1)[0]
    names = [pkg + '.gpu.chain', pkg + '.gpu.chain_palette']
    gpu_pkg = importlib.import_module(pkg + '.gpu')
    for n in names:
        sys.modules.pop(n, None)
        # `from . import x` resolves through the parent package's attribute
        # when it exists: clear it so the fresh import is a real import
        if hasattr(gpu_pkg, n.rsplit('.', 1)[1]):
            delattr(gpu_pkg, n.rsplit('.', 1)[1])
    fresh = importlib.import_module(pkg + '.gpu.chain')
    fresh_cp = importlib.import_module(pkg + '.gpu.chain_palette')
    check('gpu/chain imports fresh from a purged sys.modules and carries '
          'chain_palette as CP (imported at the bottom)',
          getattr(fresh, 'CP', None) is fresh_cp)
    check('chain_palette holds no module-level `chain` attribute (it reads '
          'chain lazily inside each function)',
          not hasattr(fresh_cp, 'chain'))
    check('the fresh chain answers the whole road: quant_refusal on a default '
          'settings object is None',
          fresh.quant_refusal(RenderSettings()) is None)
    check("every chain door the map's shape needs is a callable on "
          'chain_palette (palette, ordered, ehb, cry16, yjk + refusals)',
          all(callable(getattr(fresh_cp, n, None)) for n in
              ('palette', 'palette_refusal', 'ordered', 'ehb', 'ehb_refusal',
               'cry16', 'cry16_refusal', 'yjk', 'yjk_refusal', 'era_refusal',
               'era_road', 'quant_era', 'gate', 'draw')))
    # the rest of this module reads the live modules
    chain, CP = fresh, fresh_cp


# ------------------------------------------------- P0: the palette snap


def test_p0_palette_snap():
    h, w = 48, 64
    rgb = random_image(h, w)

    # (2) the simulator twin against reduce_depth for five palette modes
    # x four dithers at strength 0.7
    custom = tuple((float(r), float(g), float(b)) for r, g, b in
                   ((0.1, 0.1, 0.1), (0.9, 0.2, 0.2), (0.2, 0.8, 0.3),
                    (0.2, 0.3, 0.9), (0.9, 0.9, 0.2), (0.7, 0.2, 0.8),
                    (0.95, 0.95, 0.95)))
    modes = [('VGA256', {'color_depth': '8', 'palette_mode': 'VGA256'}),
             ('EGA16', {'color_depth': '4', 'palette_mode': 'EGA16'}),
             ('GRAY 16', {'color_depth': '4', 'palette_mode': 'GRAY'}),
             ('CUSTOM 7 colours', {'color_depth': '8', 'palette_mode': 'CUSTOM',
                                   'palette_colors': custom}),
             ('ADAPTIVE 256', {'color_depth': '8', 'palette_mode': 'ADAPTIVE'})]
    for label, kw in modes:
        for kind in ('NONE', 'BAYER2', 'BAYER8', 'HALFTONE'):
            PA.clear_caches()
            st = settings(dither=kind, dither_strength=0.7, **kw)
            ref = PO.reduce_depth(rgb, st, 0)
            pal = PO._palette_for(st, CP.palette_size(st), rgb, 0)
            uni, tile = palette_inputs(st, pal, h, w)
            icm = PA.icm_index_image(PA.get_inverse_colormap(pal))
            d, err, a_ok = twin('PALETTE', ref, h, w, uni,
                                {'source': rgb, 'icm': icm,
                                 'pal': CP.pal_image(pal), 'tile': tile})
            check(f'PALETTE {label} + {kind} 0.7: the GPU stage is bitwise '
                  'reduce_depth (the CPU\'s own inverse colormap as a '
                  'texture), alpha 1',
                  d == 0.0 and a_ok, str(err) if err else f'max {d}')

    # (3) tie probe: the 1/64 cell boundaries and the same row one ulp below
    pal = PA.vga256()
    k = (np.arange(64, dtype=np.float32) / np.float32(64)).astype(np.float32)
    for label, row in (('k/64 (the cell boundaries)', k),
                       ('k/64 - 2^-24 (one ulp below)',
                        (k - np.float32(2.0 ** -24)).astype(np.float32))):
        probe = np.stack([row, row[::-1], np.roll(row, 7)], 1)[None]
        probe = np.ascontiguousarray(probe, np.float32)
        st = settings(color_depth='8', palette_mode='VGA256')
        ref = PO.reduce_depth(probe, st, 0)
        uni, tile = palette_inputs(st, pal, 1, 64)
        d, err, _ = twin('PALETTE', ref, 1, 64, uni,
                         {'source': probe,
                          'icm': PA.icm_index_image(PA.get_inverse_colormap(pal)),
                          'pal': CP.pal_image(pal), 'tile': tile})
        check(f'PALETTE tie probe {label}: int(c * 64) truncates the same '
              'cell on both roads, bitwise', d == 0.0, str(err or d))

    # (4) the A/B law: every output pixel is a palette entry; the dither
    # moves the picture and both stay members
    st = settings(color_depth='8', palette_mode='VGA256')
    out0 = PO.reduce_depth(rgb, st, 0)
    st1 = settings(color_depth='8', palette_mode='VGA256', dither='BAYER4',
                   dither_strength=1.0)
    out1 = PO.reduce_depth(rgb, st1, 0)
    st2 = settings(color_depth='8', palette_mode='VGA256', dither='BAYER4',
                   dither_strength=0.0)
    out2 = PO.reduce_depth(rgb, st2, 0)
    check('VGA256 at 8 bits: every output pixel is a palette entry (exact '
          'set membership)', members(out0, pal))
    check('VGA256 + BAYER4 at strength 1.0 differs from strength 0.0 and both '
          'pictures stay palette members',
          not np.array_equal(out1, out2) and members(out1, pal)
          and members(out2, pal))

    # (5) the fake-device rows
    fake_row('VGA256 at 8 bits',
             settings(color_depth='8', palette_mode='VGA256'), 0.0,
             dict(stages=['DISPLAY', 'PALETTE'], readbacks=0, uploads=0))
    fake_row('EGA16 + BAYER4 at 4 bits',
             settings(color_depth='4', palette_mode='EGA16', dither='BAYER4'),
             0.0, dict(stages=['DISPLAY', 'PALETTE'], readbacks=0))
    PA.clear_caches()
    fake_row('adaptive 8-bit, cold lock cache (one palette readback, one '
             'upload)', settings(color_depth='8'), 0.0,
             dict(stages=['DISPLAY', 'PALETTE'], readback_names=['palette'],
                  uploads=1))
    fake_row('adaptive 8-bit, warm lock cache (the peeked table, no '
             'readback)', settings(color_depth='8'), 0.0,
             dict(stages=['DISPLAY', 'PALETTE'], readbacks=0, uploads=0))

    # (6) the refusals, by name
    for label, kw, word in (
            ('Floyd-Steinberg at 8 bits', {'color_depth': '8',
                                           'dither': 'FLOYD'}, 'FLOYD dither'),
            ('Blue Noise at 8 bits', {'color_depth': '8', 'dither': 'NOISE'},
             'NOISE'),
            ('an adaptive palette with Lock Palette off',
             {'color_depth': '8', 'palette_lock': False}, 'Lock Palette'),
            ('a Custom palette without an image',
             {'palette_mode': 'CUSTOM', 'palette_colors': ()}, 'Custom')):
        st = settings(**kw)
        why_p = CP.palette_refusal(st)
        why_c = chain.quant_refusal(st)
        check(f'palette_refusal names {label} as a CPU road and '
              'chain.quant_refusal returns the same string',
              why_p is not None and word in why_p and why_c == why_p,
              f'{why_p!r} / {why_c!r}')
    check('VGA256 at 8 bits is the PALETTE stage: palette_refusal and '
          'quant_refusal are None',
          CP.palette_refusal(settings(color_depth='8', palette_mode='VGA256'))
          is None and chain.quant_refusal(
              settings(color_depth='8', palette_mode='VGA256')) is None)
    check('a Custom palette WITH colours is the PALETTE stage (None), lock '
          'or not',
          chain.quant_refusal(settings(palette_mode='CUSTOM',
                                       palette_colors=custom)) is None
          and chain.quant_refusal(settings(palette_mode='CUSTOM',
                                           palette_colors=custom,
                                           palette_lock=False)) is None)

    # (7) the cached ICM is bitwise the fresh one (the P:360 swap)
    PA.clear_caches()
    check('get_inverse_colormap(pal).lut equals a fresh InverseColormap(pal).lut '
          '(the cached ICM on the NONE road is bitwise)',
          np.array_equal(PA.get_inverse_colormap(pal).lut,
                         PA.InverseColormap(pal).lut))
    img = PA.icm_index_image(PA.get_inverse_colormap(pal))
    check('icm_index_image packs the 64^3 table as 512 x 512 with .r[y, x] = '
          'lut[y * 512 + x] and alpha 1',
          img.shape == (512, 512, 4)
          and np.array_equal(img[..., 0].ravel(),
                             PA.get_inverse_colormap(pal).lut.astype(np.float32))
          and bool((img[..., 3] == 1.0).all()))

    # (8) the FM row
    check("the featurematrix carries the 'VGA 256 + BAYER4 (GPU palette snap)' "
          'row', fm_row('VGA 256 + BAYER4 (GPU palette snap)'))

    # (9) gate before readback: PALETTE disabled -> no palette readback, one
    # readback under the process's own name, the CPU picture bitwise
    saved = chain.ENABLED
    chain.ENABLED = tuple(n for n in saved if n != 'PALETTE')
    try:
        PA.clear_caches()
        out_g, out_c, rec, live, _dev = gpu_road(settings(color_depth='8'))
        rb = rec.get('readbacks') or []
        check("with PALETTE removed from chain.ENABLED the cold adaptive row "
              "engages no PALETTE stage, reads back ONCE under 'quant' "
              "(refused by name), spends no 'palette' readback and no upload, "
              'and equals the CPU road bitwise',
              'PALETTE' not in (rec.get('stages') or []) and len(rb) == 1
              and rb[0][0] == 'quant' and rec.get('uploads') == 0
              and live == [] and np.array_equal(out_g, out_c), str(rec))
    finally:
        chain.ENABLED = saved

    # (10) the fallback after a chain-side readback (P0 B1a)
    saved_draw = CP.draw
    CP.draw = lambda *a, **k: None
    try:
        PA.clear_caches()
        out_g, out_c, rec, live, _dev = gpu_road(settings(color_depth='8'))
        rb = rec.get('readbacks') or []
        check("with CP.draw refusing, the cold adaptive row reads back once "
              "under 'palette', engages no PALETTE stage, and the CPU fallback "
              'runs on the read-back (post-DISPLAY) frame: bitwise the CPU road',
              'PALETTE' not in (rec.get('stages') or []) and len(rb) == 1
              and rb[0][0] == 'palette' and live == []
              and np.array_equal(out_g, out_c), str(rec))
    finally:
        CP.draw = saved_draw


# ------------------------------------------ P1: ordered dither to bits


def test_p1_ordered_bits():
    h, w = 48, 64
    rgb = random_image(h, w)

    # (2) the simulator twin: bits x kinds x strengths
    for bits in ((5, 6, 5), (5, 5, 5), (4, 4, 4), (3, 3, 2)):
        levels = tuple(float((1 << b) - 1) for b in bits)
        lut = chain.quant_lut(bits)
        for kind in ('BAYER2', 'BAYER4', 'BAYER8', 'BAYER16', 'HALFTONE'):
            m = DI.ORDERED[kind]
            for strength in (1.0, 0.6):
                ref = DI.ordered_bits(rgb, bits, kind, strength)
                uni = {'resolution': (float(w), float(h)), 'levels': levels,
                       'strength': float(np.float32(strength)),
                       'tile_mask': int(m.shape[0] - 1)}
                d, err, a_ok = twin('ORDERED', ref, h, w, uni,
                                    {'source': rgb, 'tile': CP.tile_image(m),
                                     'lut': lut})
                check(f'ORDERED {bits} {kind} at strength {strength}: the GPU '
                      'stage is bitwise DI.ordered_bits (tile texture, one '
                      'multiply, one add, roundEven, the level table), alpha 1',
                      d == 0.0 and a_ok, str(err) if err else f'max {d}')

    # (3) the tie probe, per channel
    for bits in ((5, 6, 5), (5, 5, 5), (3, 3, 2)):
        cols = []
        for b in bits:
            lv = np.float32((1 << b) - 1)
            k = np.arange((1 << b) - 1, dtype=np.float32)
            v = ((k + np.float32(0.5)) / lv).astype(np.float32)
            exact = (v * lv).astype(np.float32) == (k + np.float32(0.5))
            cols.append(v[exact])
            check(f'ORDERED {bits}: channel of {b} bits has exact half ties '
                  'to probe (n_t > 0)', int(exact.sum()) > 0, str(int(exact.sum())))
        n_t = min(len(c) for c in cols)
        ties = np.ascontiguousarray(
            np.stack([c[:n_t] for c in cols], 1)[None].astype(np.float32))
        levels = tuple(float((1 << b) - 1) for b in bits)
        ref = DI.ordered_bits(ties, bits, 'BAYER4', 0.0)
        even = all(bool((np.rint(ref[..., ch] * np.float32(levels[ch])) % 2
                         == 0).all()) for ch in range(3))
        uni = {'resolution': (float(n_t), 1.0), 'levels': levels,
               'strength': 0.0, 'tile_mask': 3}
        d, err, _ = twin('ORDERED', ref, 1, n_t, uni,
                         {'source': ties, 'tile': CP.tile_image(DI.BAYER4),
                          'lut': chain.quant_lut(bits)})
        check(f'ORDERED {bits}: {n_t} exact half ties per channel at strength '
              '0.0 round half to even on both roads, bitwise',
              n_t > 0 and even and d == 0.0, str(err or d))

    # (4) the A/B law
    st = settings(color_depth='16', dither='BAYER4', dither_strength=1.0)
    out = PO.reduce_depth(rgb, st, 0)
    on_lattice = all(np.array_equal(out[..., c] * np.float32(lv),
                                    np.round(out[..., c] * np.float32(lv)))
                     for c, lv in enumerate((31.0, 63.0, 31.0)))
    check("BAYER4 at 16 bits differs from the plain 5:6:5 snap and every "
          'output value is on the lattice',
          not np.array_equal(out, PA.snap_bits(rgb, 5, 6, 5)) and on_lattice)

    # (5) the fake-device rows
    fake_row('BAYER4 at 16 bits (the console framebuffer)',
             settings(color_depth='16', dither='BAYER4'), 0.0,
             dict(stages=['DISPLAY', 'ORDERED'], readbacks=0, uploads=0))
    fake_row('BAYER2 at 15 bits', settings(color_depth='15', dither='BAYER2'),
             0.0, dict(stages=['DISPLAY', 'ORDERED'], readbacks=0))
    # The seven console presets, re-measured on the merged 1.90.0 tree
    # (the spec's P1 B1 numbers were the 1.89.0 render's): LIGHT-A1 runs
    # fog inside the material pass, so N64 and DREAMCAST now INHERIT the
    # render's frame (resident, no upload) instead of opening on the fog
    # readback; SIG-1's VI draws after N64's ORDERED dither; design 2.5
    # #32 / #39 moved VOODOO's and PS2's dither into TRANS-1's CPU
    # framebuffer stage (`dither 'NONE'`, `fb_dither`), so their post
    # depth is the plain QUANT snap and the render's frame comes down at
    # 'framebuffer format' (uploaded once at DISPLAY).
    ordered_resident = dict(resident=True, stages=['DISPLAY', 'ORDERED'],
                            readbacks=0, uploads=0)
    for key, how, want in (
            ('PSX', 'its own dither on the ORDERED stage, the render\'s frame '
                    'inherited', ordered_resident),
            ('PSX_HIRES', 'its own dither on the ORDERED stage, the render\'s '
                          'frame inherited', ordered_resident),
            ('THREEDO', 'its own dither on the ORDERED stage, the render\'s '
                        'frame inherited', ordered_resident),
            ('N64', 'BAYER2 on the ORDERED stage then SIG-1\'s VI, the '
                    'render\'s fogged frame inherited (fog on the GPU, no '
                    'readback, no upload)',
             dict(resident=True, stages=['DISPLAY', 'ORDERED', 'VI'],
                  readbacks=0, uploads=0)),
            ('DREAMCAST', 'BAYER2 on the ORDERED stage, the render\'s fogged '
                          'frame inherited (fog on the GPU, no readback, no '
                          'upload)', ordered_resident),
            ('VOODOO', 'the dither in TRANS-1\'s CPU framebuffer stage, so '
                       'the frame comes down at \'framebuffer format\' and '
                       'uploads once; the \'16\' snap is the QUANT stage, '
                       'then SIG-1\'s VIDEO_FILTER; no readback',
             dict(resident=False,
                  stages=['DISPLAY', 'QUANT', 'VIDEO_FILTER'],
                  readbacks=0, uploads=1)),
            ('PS2', 'the DIMX dither in TRANS-1\'s CPU framebuffer stage, so '
                    'the frame comes down at \'framebuffer format\' and '
                    'uploads once; QUANT draws, then the FIELDS interlace '
                    'reads back by name',
             dict(resident=False, stages=['DISPLAY', 'QUANT'],
                  readback_names=['interlace'], uploads=1))):
        fake_row(f'the {key} preset at 96x72 ({how})', preset_settings(key),
                 0.0, want)

    # (6) the refusals
    check("NOISE at 16 bits: quant_refusal names the 'NOISE dither' (a PCG "
          'stream, CPU by name)',
          'NOISE dither' in str(chain.quant_refusal(
              settings(color_depth='16', dither='NOISE'))))
    check("FLOYD at 16 bits: quant_refusal names the 'FLOYD dither'",
          'FLOYD dither' in str(chain.quant_refusal(
              settings(color_depth='16', dither='FLOYD'))))
    check('BAYER4 at 16 bits is the ORDERED stage (quant_refusal None; '
          'ordered_road True; palette_road False)',
          chain.quant_refusal(settings(color_depth='16', dither='BAYER4')) is None
          and CP.ordered_road(settings(color_depth='16', dither='BAYER4'))
          and not CP.palette_road(settings(color_depth='16', dither='BAYER4')))
    check('HAM6 / 1-bit with BAYER4 are NOT the ordered road (their refusals '
          'stand)', not CP.ordered_road(settings(color_depth='HAM6',
                                                 dither='BAYER4'))
          and not CP.ordered_road(settings(color_depth='1', dither='BAYER4'))
          and 'HAM' in str(chain.quant_refusal(settings(color_depth='HAM6',
                                                        dither='BAYER4'))))

    # (7) the existing FM row is the field's home
    check("the featurematrix keeps the 'ordered dither BAYER4' row (dither "
          'has its home; no new row)', fm_row('ordered dither BAYER4'))


# ------------------------------------------ C061: palette register depth


def test_c061_register_depth():
    h, w = 48, 64
    rgb = random_image(h, w)

    # (1) identity at NONE: the same object
    pal = PA.vga256()
    check('snap_registers at palette_bits NONE returns the palette ITSELF '
          '(the same object; the 1.89.0 road untouched)',
          PE.snap_registers(pal, settings()) is pal
          and PE.snap_registers(pal, settings(palette_bits='BOGUS')) is pal)

    # (2) lattice membership, output membership, the picture moves, monotone
    outs = {}
    for bits, mult in (('BITS_3', 7), ('CPC_27', 2), ('BITS_6', 63),
                       ('BITS_4', 15)):
        PA.clear_caches()
        st = settings(color_depth='4', palette_mode='ADAPTIVE', palette_size=16,
                      palette_bits=bits)
        p = PO._palette_for(st, 16, rgb, 0)
        check(f'{bits}: every register of _palette_for times {mult} is an '
              'integer (the DAC lattice)',
              np.array_equal(p * np.float32(mult), np.round(p * np.float32(mult))))
        out = PO.reduce_depth(rgb, st, 0)
        outs[bits] = out
        check(f'{bits}: every output pixel of reduce_depth is one of the '
              'snapped registers', members(out, p))
    PA.clear_caches()
    out_none = PO.reduce_depth(rgb, settings(color_depth='4',
                                             palette_mode='ADAPTIVE',
                                             palette_size=16), 0)
    check('BITS_3 moves the picture against NONE (16 adaptive registers on '
          'the 512-colour lattice)', not np.array_equal(outs['BITS_3'], out_none))
    # monotone on a 64-colour random image
    cols64 = _RNG.random((64, 3)).astype(np.float32)
    img64 = cols64[_RNG.integers(0, 64, size=(24, 32))]
    counts = []
    for bits in ('BITS_1', 'BITS_2', 'BITS_3'):
        PA.clear_caches()
        o = PO.reduce_depth(img64, settings(color_depth='4',
                                            palette_mode='ADAPTIVE',
                                            palette_size=16,
                                            palette_bits=bits), 0)
        counts.append(len({px.tobytes() for px in o.reshape(-1, 3)}))
    # measured [8, 16, 15] on this image: the spec's "monotone" claim is not
    # a law (a coarser lattice can merge registers the frame never used), the
    # lattice bound is -- 2^3 at BITS_1, 4^3 at BITS_2, never more than the 16
    check('the distinct output colours on a 64-colour image are bounded by '
          'the lattice and the 16 registers: <= 8 at BITS_1, <= 16 at BITS_2 '
          'and BITS_3, and BITS_1 <= BITS_3',
          counts[0] <= 8 and counts[1] <= 16 and counts[2] <= 16
          and counts[0] <= counts[2], str(counts))
    check('snap_levels(pal, 3) is snap_bits\'s arithmetic with three levels '
          '(CPC 27: 0, 0.5, 1 per channel, half to even)',
          np.array_equal(PA.snap_levels(np.array([[0.24, 0.25, 0.26],
                                                  [0.74, 0.75, 0.76]],
                                                 np.float32), 3),
                         np.array([[0.0, 0.0, 0.5], [0.5, 1.0, 1.0]],
                                  np.float32)))
    check('a fixed palette is snapped too: EGA16 at BITS_1 collapses to '
          'on/off channels', np.array_equal(
              PE.snap_registers(PA.EGA16, settings(palette_bits='BITS_1')),
              np.round(np.clip(PA.EGA16, 0, 1)).astype(np.float32)))

    # (3) the twin at BITS_3 + BAYER4 and BITS_4 + NONE
    for bits, kind in (('BITS_3', 'BAYER4'), ('BITS_4', 'NONE')):
        PA.clear_caches()
        st = settings(color_depth='4', palette_mode='ADAPTIVE', palette_size=16,
                      palette_bits=bits, dither=kind)
        ref = PO.reduce_depth(rgb, st, 0)
        p = PO._palette_for(st, 16, rgb, 0)
        uni, tile = palette_inputs(st, p, h, w)
        d, err, a_ok = twin('PALETTE', ref, h, w, uni,
                            {'source': rgb,
                             'icm': PA.icm_index_image(PA.get_inverse_colormap(p)),
                             'pal': CP.pal_image(p), 'tile': tile})
        check(f'PALETTE with the {bits} snapped table + {kind}: bitwise '
              'reduce_depth, alpha 1', d == 0.0 and a_ok,
              str(err) if err else f'max {d}')
    PA.clear_caches()
    st_row = settings(color_depth='4', palette_size=16, palette_bits='BITS_3',
                      dither='BAYER4')
    fake_row('16 adaptive registers at 3 bits (Atari ST) + BAYER4, cold',
             st_row, 0.0, dict(stages=['DISPLAY', 'PALETTE'],
                               readback_names=['palette'], uploads=1))
    dev, rec = fake_row('16 adaptive registers at 3 bits (Atari ST) + BAYER4, '
                        'warm (the peeked table snapped by the chain itself)',
                        settings(color_depth='4', palette_size=16,
                                 palette_bits='BITS_3', dither='BAYER4'),
                        0.0, dict(stages=['DISPLAY', 'PALETTE'], readbacks=0,
                                  uploads=0))
    key = ('ADAPTIVE', 16, 'MEDIAN_CUT', 0)
    raw = PE.cached_peek(key)
    snapped = PE.snap_registers(raw, st_row) if raw is not None else None
    check('after the warm frame the chain uploaded the SNAPPED table and not '
          "the raw median cut (dev.cache holds ('pal', snapped) only)",
          raw is not None and ('pal', np.ascontiguousarray(snapped).tobytes())
          in dev.cache and ('pal', np.ascontiguousarray(raw).tobytes())
          not in dev.cache and not np.array_equal(raw, snapped))

    # (4) the refusal
    check("BITS_3 + FLOYD: quant_refusal names the 'FLOYD dither' (the snap "
          'never lifts a diffusion refusal)',
          'FLOYD dither' in str(chain.quant_refusal(
              settings(color_depth='4', palette_bits='BITS_3', dither='FLOYD'))))

    # (5) the FM row, (6) the dead stub
    check("the featurematrix carries the 'palette registers 3-bit (Atari ST)' "
          'row', fm_row('palette registers 3-bit (Atari ST)'))
    check('the dead palette.amiga_ocs() stub is gone (this feature is what it '
          'promised)', not hasattr(PA, 'amiga_ocs'))
    for key_, bits in (('ATARI_ST', 'BITS_3'), ('AMIGA_OCS', 'BITS_4'),
                       ('MSX2', 'BITS_3'), ('PC98', 'BITS_4'),
                       ('TURBO_SILVER', 'BITS_4'), ('SNES', 'BITS_5'),
                       ('NEO_GEO', 'BITS_5'), ('SEGA_32X', 'BITS_5')):
        s = PRESETS[key_]['settings']
        check(f'the {key_} preset carries palette_bits {bits} and a colour '
              'depth that is an item',
              s.get('palette_bits') == bits
              and s.get('color_depth') in ('4', '8'))


# ------------------------------------------ C054: Extra Half-Brite


def test_c054_ehb():
    h, w = 48, 64
    rgb = random_image(h, w)

    # (2) the laws
    PA.clear_caches()
    st = settings(palette_mode='EHB')
    P32 = PE.ehb_fit(rgb, st, 0)
    q32, q64, P64 = PE.ehb_registers(P32)
    check('the 64 EHB registers: rows 32..63 are the exact integer halves of '
          'rows 0..31 (each 4-bit component shifted right one)',
          q64.shape == (64, 3) and np.array_equal(q64[32:], q64[:32] >> 1)
          and np.array_equal(P64, (q64.astype(np.float32) / np.float32(15)
                                   ).astype(np.float32)))
    out = PE.ehb_reduce(rgb, st, 0)
    check('every EHB output pixel is a row of the 64 registers', members(out, P64))
    check('reduce_depth routes EHB through the era dispatcher (bitwise '
          'ehb_reduce)', np.array_equal(PO.reduce_depth(rgb, st, 0), out))
    # the shadow law: lock from A (32 bright lattice colours), reduce B (halves)
    PA.clear_caches()
    lattice = np.array([(r, g, b) for r in (12, 13, 14, 15) for g in (12, 14)
                        for b in (9, 11, 13, 15)], np.int32)
    assert len(lattice) == 32
    halves = lattice >> 1
    coincide = {tuple(x) for x in halves} & {tuple(x) for x in lattice}
    check('the shadow-law fixture: no half coincides with a bright entry',
          not coincide, str(coincide))
    A = (lattice.astype(np.float32) / np.float32(15))[
        _RNG.integers(0, 32, size=(16, 32))].astype(np.float32)
    st_lock = settings(palette_mode='EHB')
    PA._PALETTE_CACHE[PE.ehb_key(st_lock, 0)] = (lattice.astype(np.float32)
                                                 / np.float32(15)).astype(np.float32)
    fit_from_A = PE.ehb_fit(A, st_lock, 0)
    qA32, qA64, PA64 = PE.ehb_registers(fit_from_A)
    check('Lock Palette holds the 32 registers under the P:316 key at size 32 '
          '(the fit is reused, not refitted)',
          np.array_equal(qA32, lattice))
    lut = PE.ehb_lut(qA64)
    codes = (halves[:, 0] << 8) | (halves[:, 1] << 4) | halves[:, 2]
    check("the shadow law: every colour that is an exact half of a bright "
          'register maps to a half-brite index (>= 32)',
          bool((lut[codes] >= 32).all()), str(lut[codes]))
    B = (halves.astype(np.float32) / np.float32(15))[None].astype(np.float32)
    outB = PE.ehb_reduce(B, st_lock, 0)
    check('reducing the halves image under the locked 32 returns the '
          'half-brite rows bitwise', np.array_equal(outB[0], PA64[32:][
              lut[codes] - 32]) and members(outB, PA64[32:]))
    PA.clear_caches()
    st1 = settings(palette_mode='EHB', dither='BAYER4', dither_strength=1.0)
    st0 = settings(palette_mode='EHB', dither='BAYER4', dither_strength=0.0)
    o1 = PE.ehb_reduce(rgb, st1, 0)
    o0 = PE.ehb_reduce(rgb, st0, 0)
    _q, _q64, P64b = PE.ehb_registers(PE.ehb_fit(rgb, st1, 0))
    check('EHB + BAYER4 at strength 1.0 moves the picture against 0.0 and both '
          'stay in the 64', not np.array_equal(o1, o0) and members(o1, P64b)
          and members(o0, P64b))
    stf = settings(palette_mode='EHB', dither='FLOYD')
    of = PE.ehb_reduce(rgb, stf, 0)
    check('EHB under FLOYD (the CPU diffusion road) is a member of the 64 at '
          'every pixel', members(of, P64b) and not np.array_equal(of, o0))
    check('EHB ignores palette_bits (the 4:4:4 snap is its own): BITS_1 gives '
          'the same picture',
          np.array_equal(PE.ehb_reduce(rgb, settings(palette_mode='EHB',
                                                     palette_bits='BITS_1'), 0),
                         PE.ehb_reduce(rgb, settings(palette_mode='EHB'), 0)))

    # (3) the twin
    for kind in ('NONE', 'BAYER4', 'HALFTONE'):
        PA.clear_caches()
        st = settings(palette_mode='EHB', dither=kind, dither_strength=0.8)
        ref = PE.ehb_reduce(rgb, st, 0)
        _q32, q64, P64 = PE.ehb_registers(PE.ehb_fit(rgb, st, 0))
        uni, tile = palette_inputs(st, P64, h, w)
        d, err, a_ok = twin('EHB', ref, h, w, uni,
                            {'source': rgb, 'ehb_lut': PE.ehb_lut_image(q64),
                             'pal': CP.pal_image(P64), 'tile': tile})
        check(f'EHB + {kind} 0.8: the GPU stage is bitwise ehb_reduce (the '
              '64-way integer argmin table), alpha 1', d == 0.0 and a_ok,
              str(err) if err else f'max {d}')
    # the tie probe: (k + 0.5) / 15 where the product is an exact half
    k = np.arange(14, dtype=np.float32)
    v = ((k + np.float32(0.5)) / np.float32(15)).astype(np.float32)
    exact = (v * np.float32(15)).astype(np.float32) == (k + np.float32(0.5))
    v = v[exact]
    n_t = len(v)
    ties = np.ascontiguousarray(np.stack([v, v[::-1], v], 1)[None], np.float32)
    PA.clear_caches()
    st = settings(palette_mode='EHB')
    ref = PE.ehb_reduce(ties, st, 0)
    _q32, q64, P64 = PE.ehb_registers(PE.ehb_fit(ties, st, 0))
    uni, tile = palette_inputs(st, P64, 1, n_t)
    d, err, _ = twin('EHB', ref, 1, n_t, uni,
                     {'source': ties, 'ehb_lut': PE.ehb_lut_image(q64),
                      'pal': CP.pal_image(P64), 'tile': tile})
    check(f'EHB tie probe: {n_t} exact halves of the 4:4:4 lattice round half '
          'to even on both roads, bitwise', n_t > 0 and d == 0.0, str(err or d))
    PA.clear_caches()
    fake_row('Extra Half-Brite, cold (the 32 fitted from one readback)',
             settings(palette_mode='EHB'), 0.0,
             dict(stages=['DISPLAY', 'EHB'], readback_names=['palette'],
                  uploads=1))
    fake_row('Extra Half-Brite, warm (the 32 peeked, no readback)',
             settings(palette_mode='EHB'), 0.0,
             dict(stages=['DISPLAY', 'EHB'], readbacks=0, uploads=0))

    # (4) the refusals
    for label, kw, word in (
            ('FLOYD under EHB', {'palette_mode': 'EHB', 'dither': 'FLOYD'},
             'FLOYD dither'),
            ('Lock Palette off under EHB', {'palette_mode': 'EHB',
                                            'palette_lock': False},
             'Lock Palette')):
        st = settings(**kw)
        why_e = CP.ehb_refusal(st)
        check(f'ehb_refusal names {label} and chain.quant_refusal returns the '
              'same string', why_e is not None and word in why_e
              and chain.quant_refusal(st) == why_e, f'{why_e!r}')
    check('EHB with no dither and the lock on is the EHB stage (quant_refusal '
          'None; era_road True)',
          chain.quant_refusal(settings(palette_mode='EHB')) is None
          and CP.era_road(settings(palette_mode='EHB')))

    # (5) the FM row and the preset
    check("the featurematrix carries the 'Extra Half-Brite (Amiga)' row",
          fm_row('Extra Half-Brite (Amiga)'))
    p = PRESETS.get('AMIGA_EHB')
    check('the AMIGA_EHB preset exists (PLATFORM, palette_mode EHB, dither '
          'NONE, 320x256)', p is not None and p['category'] == 'PLATFORM'
          and p['settings'].get('palette_mode') == 'EHB'
          and p['settings'].get('dither') == 'NONE'
          and (p['settings'].get('resolution_x'),
               p['settings'].get('resolution_y')) == (320, 256))
    check("the PALETTE_MODE enum carries EHB at the END (positional numbering "
          'never moves an item)',
          [i[0] for i in _properties().PALETTE_MODE][-1] == 'EHB')


# ---------------------------------------------------- C011: CRY16 (Jaguar)


def test_c011_cry16():
    h, w = 64, 64
    rgb = random_image(h, w)
    table, div, mul, argmin = PE.cry_tables()

    # (2) the laws
    out = PE.cry16(rgb)
    R8 = np.round(np.clip(rgb, 0, 1) * np.float32(255)).astype(np.int32)
    out8 = np.round(out * np.float32(255)).astype(np.int32)
    check('CRY16: the intensity is exact -- max(r, g, b) of the output equals '
          "the input's at every pixel", np.array_equal(out8.max(2), R8.max(2)))
    ramp = (np.arange(256, dtype=np.float32) / np.float32(255)).astype(np.float32)
    grey = np.ascontiguousarray(np.stack([ramp, ramp, ramp], 1)[None], np.float32)
    check('CRY16: a 1 x 256 grey ramp reproduces bit for bit (cell 0x88 is '
          'white)', np.array_equal(PE.cry16(grey), grey))
    used = {tuple(t) for t in table[argmin[
        np.argmax(R8 == R8.max(2)[..., None], 2),
        div[np.maximum(R8.max(2), 1), np.where(np.argmax(R8 == R8.max(2)[..., None], 2) == 0, R8[..., 1], R8[..., 0])],
        div[np.maximum(R8.max(2), 1), np.where(np.argmax(R8 == R8.max(2)[..., None], 2) == 2, R8[..., 1], R8[..., 2])]]].reshape(-1, 3)}
    check('CRY16: the chroma cells used on the random image are <= 256 and '
          '>= 32 (non-vacuous)', 32 <= len(used) <= 256, str(len(used)))
    check('CRY16 is not RGB565: the picture differs from snap_bits(5, 6, 5)',
          not np.array_equal(out, PA.snap_bits(rgb, 5, 6, 5)))
    check('CRY_TABLE: every cell has max 255 (a true upper-surface point)',
          bool((table.max(1) == 255).all()))
    pins = {0x88: (255, 255, 255), 0x00: (0, 0, 255), 0x0F: (255, 0, 0),
            0xFF: (255, 255, 0), 0xF0: (0, 255, 255), 0xF8: (0, 255, 0),
            0x08: (255, 0, 255)}
    check("CRY_TABLE: white at 0x88, blue 0x00, red 0x0F, yellow 0xFF, cyan "
          '0xF0, green 0xF8, magenta 0x08 (the corners and edge midpoints)',
          all(tuple(int(x) for x in table[c]) == v for c, v in pins.items()),
          str({hex(c): tuple(int(x) for x in table[c]) for c in pins}))
    lut255 = chain.quant_lut((8, 8, 8))
    check('PE.LUT255 equals chain.quant_lut((8, 8, 8)) for every channel (the '
          "CPU's and the GPU's k/255 are the same float32 quotients)",
          all(np.array_equal(PE.LUT255, lut255[0, :, c]) for c in range(3)))
    check('cry_table_diff counts: 0 against the table itself, 256 against the '
          'table xor 2', PE.cry_table_diff(table) == 0
          and PE.cry_table_diff(table ^ 2) == 256)
    check('CRY_DIV row 0 equals row 1 (I == 0 reads as I == 1) and CRY_MUL '
          'row 0 is black', np.array_equal(div[0], div[1])
          and bool((mul[0] == 0).all()))

    # (3) the twin
    imgs = PE.cry_images()
    samp = dict(imgs, lut255=lut255)
    for label, im in (('the random image', rgb), ('the grey ramp', grey)):
        hh, ww = im.shape[:2]
        d, err, a_ok = twin('CRY16', PE.cry16(im), hh, ww,
                            {'resolution': (float(ww), float(hh))},
                            dict(samp, source=im))
        check(f'CRY16 on {label}: the GPU stage is bitwise cry16 (integer end '
              'to end from three constant tables), alpha 1',
              d == 0.0 and a_ok, str(err) if err else f'max {d}')
    k = np.arange(255, dtype=np.float32)
    v = ((k + np.float32(0.5)) / np.float32(255)).astype(np.float32)
    exact = (v * np.float32(255)).astype(np.float32) == (k + np.float32(0.5))
    v = v[exact]
    n_t = len(v)
    ties = np.ascontiguousarray(np.stack([v, v[::-1], np.roll(v, 3)], 1)[None],
                                np.float32)
    d, err, _ = twin('CRY16', PE.cry16(ties), 1, n_t,
                     {'resolution': (float(n_t), 1.0)}, dict(samp, source=ties))
    check(f'CRY16 tie probe: {n_t} exact halves of k/255 round half to even on '
          'both roads, bitwise', n_t > 0 and d == 0.0, str(err or d))
    fake_row('colour depth CRY16 (Jaguar)', settings(color_depth='CRY16'), 0.0,
             dict(stages=['DISPLAY', 'CRY16'], readbacks=0, uploads=0))

    # (4) the refusal and the inert dither
    check('cry16_refusal is None and chain.quant_refusal(CRY16) is None (the '
          'road is integer end to end)',
          CP.cry16_refusal(settings(color_depth='CRY16')) is None
          and chain.quant_refusal(settings(color_depth='CRY16')) is None)
    PE._NOTED.discard('colour depth CRY16: the encode has no dither seat; '
                      'Dither BAYER4 ignored')
    o_b = PO.reduce_depth(rgb, settings(color_depth='CRY16', dither='BAYER4'), 0)
    o_n = PO.reduce_depth(rgb, settings(color_depth='CRY16'), 0)
    check('CRY16 with BAYER4: the dither is inert (bitwise the NONE output) and '
          'the note is recorded once in palette_era._NOTED',
          np.array_equal(o_b, o_n) and any('CRY16' in m and 'BAYER4' in m
                                           for m in PE._NOTED))
    check('CRY16 under FLOYD is still the CRY16 GPU stage (quant_refusal None: '
          'the encode has no dither seat)',
          chain.quant_refusal(settings(color_depth='CRY16', dither='FLOYD'))
          is None)

    # (5) the FM row and the preset
    check("the featurematrix carries the 'CRY16 colour (Jaguar)' row",
          fm_row('CRY16 colour (Jaguar)'))
    check("the JAGUAR preset renders in CRY16 and its note no longer claims a "
          'Gouraud the engine does not do',
          PRESETS['JAGUAR']['settings'].get('color_depth') == 'CRY16'
          and 'CRY' in PRESETS['JAGUAR']['note']
          and 'intensity-only Gouraud is not modelled' in PRESETS['JAGUAR']['note'])
    CD = _properties().COLOR_DEPTH
    check('the COLOR_DEPTH enum carries CRY16 and YJK at the END, after HAM6 '
          '(positional numbering never moves an item)',
          [i[0] for i in CD][-3:] == ['HAM6', 'CRY16', 'YJK'])


# ------------------------------------------------------ C059: YJK (MSX2+)


def test_c059_yjk():
    h, w = 48, 64
    rgb = random_image(h, w)

    # (2) the laws
    cols = _RNG.random((h, w // 4, 3)).astype(np.float32)
    uni = np.repeat(cols, 4, axis=1).astype(np.float32)
    in5 = np.round(np.clip(uni, 0, 1) * np.float32(255)).astype(np.int32) >> 3
    out5 = np.round(np.clip(PE.yjk(uni), 0, 1) * np.float32(255)
                    ).astype(np.int32) >> 3
    check('YJK on a uniform-group image: R and G reproduce the 5-bit input '
          'exactly', np.array_equal(out5[..., 0], in5[..., 0])
          and np.array_equal(out5[..., 1], in5[..., 1]))
    dB = in5[..., 2] - out5[..., 2]
    # measured: in - out is 0 or -1 (the decode's (... + 2) // 4 lands on or
    # one step ABOVE the input); the spec wrote the sign the other way
    check("YJK on a uniform-group image: B is within one 5-bit step (-1 <= in "
          "- out <= 0: Grauw's decode lands on or one above the blue)",
          bool((dB >= -1).all() and (dB <= 0).all()), f'{dB.min()}..{dB.max()}')
    base = PE.yjk(rgb)
    rgb2 = rgb.copy()
    rgb2[10, 6] = np.float32(1.0) - rgb2[10, 6]
    moved = PE.yjk(rgb2)
    mask = np.ones((h, w), bool)
    mask[10, 4:8] = False
    check('YJK locality: changing pixel (10, 6) changes only columns 4..7 of '
          'row 10', np.array_equal(base[mask], moved[mask])
          and not np.array_equal(base[10, 4:8], moved[10, 4:8]))
    two = np.zeros((4, 12, 3), np.float32)
    two[:, :6] = (0.8, 0.3, 0.2)
    two[:, 6:] = (0.2, 0.4, 0.9)
    groups = PE.yjk_groups(two)
    check('yjk_groups is (H, ceil(W/4), 2) int32 and a two-colour image split '
          'at x = 6 gives one (J, K) for group 1 and a different one for group 2',
          groups.shape == (4, 3, 2) and groups.dtype == np.int32
          and not np.array_equal(groups[0, 1], groups[0, 2])
          and np.array_equal(groups[0, 0], groups[0, 0]))
    o2 = np.round(np.clip(PE.yjk(two), 0, 1) * np.float32(255)
                  ).astype(np.int32) >> 3
    y2 = PE._yjk_encode5(two)[0]
    dr = o2[0, 4:8, 0] - y2[0, 4:8]
    dg = o2[0, 4:8, 1] - y2[0, 4:8]
    check('chroma sharing: the decoded (R - y, G - y) is constant across x = '
          '4..7 of the mixed group (one J, K for four pixels)',
          len(set(dr.tolist())) == 1 and len(set(dg.tolist())) == 1,
          f'{dr} {dg}')
    out8 = np.round(base * np.float32(255)).astype(np.int32)
    n_col = len({px.tobytes() for px in out8.reshape(-1, 3)})
    check('YJK colour count on the random image: 1 < distinct triples <= 19,268',
          1 < n_col <= 19268, str(n_col))
    check('YJK is not RGB555: the picture differs from snap_bits(5, 5, 5)',
          not np.array_equal(base, PA.snap_bits(rgb, 5, 5, 5)))
    check('reduce_depth routes YJK through the era dispatcher (bitwise yjk)',
          np.array_equal(PO.reduce_depth(rgb, settings(color_depth='YJK'), 0),
                         base))

    # (3) the twin
    lut255 = chain.quant_lut((8, 8, 8))
    for ww in (64, 63):
        im = np.ascontiguousarray(rgb[:, :ww])
        d, err, a_ok = twin('YJK', PE.yjk(im), h, ww,
                            {'resolution': (float(ww), float(h))},
                            {'source': im, 'lut255': lut255})
        check(f'YJK at width {ww}: the GPU stage is bitwise yjk (four fetches '
              'per aligned group, the right edge repeated), alpha 1',
              d == 0.0 and a_ok, str(err) if err else f'max {d}')
    k = np.arange(255, dtype=np.float32)
    v = ((k + np.float32(0.5)) / np.float32(255)).astype(np.float32)
    exact = (v * np.float32(255)).astype(np.float32) == (k + np.float32(0.5))
    v = v[exact]
    n_t = len(v)
    ties = np.ascontiguousarray(np.stack([v, v[::-1], np.roll(v, 5)], 1)[None],
                                np.float32)
    d, err, _ = twin('YJK', PE.yjk(ties), 1, n_t,
                     {'resolution': (float(n_t), 1.0)},
                     {'source': ties, 'lut255': lut255})
    check(f'YJK tie probe: {n_t} exact halves of k/255 round half to even on '
          'both roads, bitwise', n_t > 0 and d == 0.0, str(err or d))
    # a row whose group sums of j are 2 mod 4: the half-even on the mean
    row = np.zeros((1, 8, 3), np.float32)
    lv = np.float32(31)
    for x, (r5, g5, b5) in enumerate(((20, 10, 10), (20, 10, 10), (21, 10, 10),
                                      (21, 10, 10), (8, 20, 4), (8, 20, 4),
                                      (9, 20, 4), (9, 20, 4))):
        row[0, x] = (r5 / lv, g5 / lv, b5 / lv)
    row = np.ascontiguousarray(row * np.float32(1.0), np.float32)
    _y, j, _k = PE._yjk_encode5(row)
    sums = PE._yjk_group_sums(j)[0]
    d, err, _ = twin('YJK', PE.yjk(row), 1, 8, {'resolution': (8.0, 1.0)},
                     {'source': row, 'lut255': lut255})
    check('YJK on a row whose group sums of j are 2 mod 4 (the mean is an '
          'exact half): half to even on both roads, bitwise',
          bool((sums % 4 == 2).any()) and d == 0.0, f'{sums.tolist()} {err or d}')
    fake_row('colour depth YJK (MSX2+)', settings(color_depth='YJK'), 0.0,
             dict(stages=['DISPLAY', 'YJK'], readbacks=0, uploads=0))

    # (4) the refusal and the inert dither
    check('yjk_refusal is None and chain.quant_refusal(YJK) is None',
          CP.yjk_refusal(settings(color_depth='YJK')) is None
          and chain.quant_refusal(settings(color_depth='YJK')) is None)
    check('YJK with BAYER4: the dither is inert (bitwise the NONE output)',
          np.array_equal(PO.reduce_depth(rgb, settings(color_depth='YJK',
                                                       dither='BAYER4'), 0),
                         base))
    check('the era precedence: CRY16 / YJK own the stage over EHB (a colour '
          'depth beats the palette mode, printed once as ignored)',
          PE.era_mode(settings(color_depth='YJK', palette_mode='EHB')) == 'YJK'
          and PE.era_mode(settings(color_depth='CRY16', palette_mode='EHB'))
          == 'CRY16' and PE.era_mode(settings(palette_mode='EHB')) == 'EHB')

    # (5) the FM row and the preset
    check("the featurematrix carries the 'YJK colour (MSX2+)' row",
          fm_row('YJK colour (MSX2+)'))
    p = PRESETS.get('MSX2_PLUS')
    check('the MSX2_PLUS preset exists (PLATFORM, color_depth YJK, dither '
          'NONE, 256x212)', p is not None and p['category'] == 'PLATFORM'
          and p['settings'].get('color_depth') == 'YJK'
          and p['settings'].get('dither') == 'NONE'
          and (p['settings'].get('resolution_x'),
               p['settings'].get('resolution_y')) == (256, 212))
    # the round's final number is the integrator's (103 after pass 1, and
    # wave 2 adds more); this pack proves its own two keys are shelved with
    # a category and a machine note, and that the library holds the round
    both = [PRESETS.get('AMIGA_EHB'), PRESETS.get('MSX2_PLUS')]
    check("the pack's two presets (AMIGA_EHB, MSX2_PLUS) are in the library "
          'with a category and a note, and the library holds the round '
          '(len(PRESETS) >= 103, the integrator\'s number after pass 1)',
          all(q is not None and str(q.get('category') or '')
              and str(q.get('note') or '') for q in both)
          and len(PRESETS) >= 103, str(len(PRESETS)))


# ------------------------------------------------------------------ main

ORDER = [test_identity_at_defaults, test_stage_sources_compile,
         test_chain_imports_stand_alone, test_p0_palette_snap,
         test_p1_ordered_bits, test_c061_register_depth, test_c054_ehb,
         test_c011_cry16, test_c059_yjk]


# ---- wave 2 ----
#
# PAL-2: C092 Super Black, C093 Video Color Check, C050 attribute cells,
# C060 per-scanline palettes. Appended after wave 1's last test; wave 1's
# helpers are used as they are. The one addition: the engine's post call
# passes the coverage plane like depth= (C092 B3), and the 1.89.0 engine's
# process() has no such kwarg, so the plane rides a second kwargs helper
# instead of wave 1's post_kw.


def post_kw2(sc, st):
    """post_kw plus the coverage plane, the engine's own call shape."""
    return dict(post_kw(sc, st), coverage=getattr(sc, 'last_coverage', None))


def gpu_road_cov(st):
    """gpu_road with the coverage plane passed to BOTH chains (the CPU
    chain runs on a copy of the settings, which drops the private)."""
    st.render_device = 'GPU'
    sc = demo_scene(st, with_texture=False)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        st._keep_gpu_frame = True
        try:
            img = R.render(sc, st)
            out = PO.process(img, st, **post_kw2(sc, st))
            rec = dict(PO.LAST_CHAIN)
        finally:
            FR.release(st)
        live = [t for t in dev.targets if not t.freed]
    st_c = st.copy()
    st_c.render_device = 'CPU'
    out_c = PO.process(img, st_c, **post_kw2(sc, st_c))
    return out, out_c, rec, live, dev


def fake_row_cov(label, st, tol, want):
    """fake_row over gpu_road_cov (the module global is swapped for the
    call, so wave 1's row shape is reused, not copied)."""
    global gpu_road
    keep = gpu_road
    gpu_road = gpu_road_cov
    try:
        return fake_row(label, st, tol, want)
    finally:
        gpu_road = keep


def cpu_frame(st, cov_kw=True):
    """(scene, render, post) on the CPU road, the engine's call shape."""
    sc = demo_scene(st, with_texture=False)
    img = np.asarray(R.render(sc, st))
    out = PO.process(img, st, **(post_kw2(sc, st) if cov_kw
                                 else post_kw(sc, st)))
    return sc, img, out


def _engine():
    """halcyon/engine.py under the fake bpy (hold_store / hold_lookup)."""
    from . import fakebpy
    fakebpy.install()
    return importlib.import_module(__package__.rsplit('.tests', 1)[0]
                                   + '.engine')


def push_saturation(rgb, k=2.5):
    """A random image with its saturation pushed so illegal pixels exist."""
    y = (rgb[..., 0] * np.float32(0.299) + rgb[..., 1] * np.float32(0.587)
         + rgb[..., 2] * np.float32(0.114))[..., None]
    return np.clip(y + np.float32(k) * (rgb - y), 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------- C092: Super Black


def test_c092_super_black():
    thr15 = np.float32(15) / np.float32(255)

    # (1) identity: the defaults are off; threshold 0 is the identity
    st0 = settings()
    check('Super Black defaults: off, threshold 15 (Max\'s default)',
          st0.super_black is False and st0.super_black_threshold == 15)
    _sc, _img, off = cpu_frame(settings(exposure=0.1))
    _sc, _img, t0 = cpu_frame(settings(exposure=0.1, super_black=True,
                                       super_black_threshold=0))
    check('super_black on at threshold 0: render + post bitwise the '
          'super_black off frame (0 changes nothing)', np.array_equal(t0, off))

    # (2) the laws on the demo scene at exposure 0.1 (dark covered pixels)
    st15 = settings(exposure=0.1, super_black=True)
    sc15, img15, out15 = cpu_frame(st15)
    cov = sc15.last_coverage
    check('the coverage plane: scene.last_coverage is a boolean (H, W) plane '
          'with covered AND uncovered pixels, and st._last_coverage is the '
          'same plane',
          cov is not None and cov.shape == (H, W) and cov.dtype == bool
          and 0 < int(cov.sum()) < H * W
          and getattr(st15, '_last_coverage', None) is cov,
          f'{None if cov is None else (cov.shape, int(cov.sum()))}')
    sc_off, _i, _o = cpu_frame(settings(exposure=0.1))
    check('the A1 gate: with super_black off the plane is None (no reduction '
          'is paid for a feature that is off)',
          sc_off.last_coverage is None)
    check('Super Black 15: every channel of every covered pixel is at or '
          'above 15/255',
          float(out15[..., :3][cov].min()) >= float(thr15),
          f'{float(out15[..., :3][cov].min())} vs {float(thr15)}')
    check('Super Black 15: the uncovered background is untouched (bitwise the '
          'off frame there)',
          np.array_equal(out15[~cov], off[~cov]))
    check('Super Black 15 moved the dark demo (covered pixels were below the '
          'floor)', int((out15 != off).any(-1).sum()) > 0,
          str(int((out15 != off).any(-1).sum())))
    st64 = settings(exposure=0.1, super_black=True, super_black_threshold=64)
    _sc, _i, out64 = cpu_frame(st64)
    check('monotone in the threshold: out(15) <= out(64) everywhere, and the '
          '64 picture floors more pixels',
          bool((out15 <= out64).all())
          and int((out64 != off).any(-1).sum())
          > int((out15 != off).any(-1).sum()))
    # the function itself
    rgb = (random_image(20, 100) * np.float32(0.2)).astype(np.float32)
    cvr = _RNG.random((20, 100)) > 0.5
    f15 = PE.super_black(rgb, cvr, settings(super_black=True))
    check('PE.super_black: np.maximum(rgb, T/255) where covered, the input '
          'bits elsewhere, float32',
          f15.dtype == np.float32
          and np.array_equal(f15[cvr], np.maximum(rgb[cvr], thr15))
          and np.array_equal(f15[~cvr], rgb[~cvr]))
    # the transparent film's alpha IS the coverage law (BOX: exact)
    st_t = settings(exposure=0.1, super_black=True, film_transparent=True,
                    aa_filter='BOX')
    sc_t = demo_scene(st_t, with_texture=False)
    img_t = np.asarray(R.render(sc_t, st_t))
    check('under a transparent film at the BOX filter the plane equals '
          'alpha > 0 of the render\'s own frame',
          sc_t.last_coverage is not None
          and np.array_equal(sc_t.last_coverage, img_t[..., 3] > 0.0))
    st_s = settings(exposure=0.1, super_black=True, film_transparent=True,
                    aa_filter='BOX', aa_mode='SUPERSAMPLE', aa_samples=4)
    sc_s = demo_scene(st_s, with_texture=False)
    img_s = np.asarray(R.render(sc_s, st_s))
    cov_s = sc_s.last_coverage
    check('supersampled 2 x 2: the plane is at OUTPUT size, any sample '
          'counting as covered (equals alpha > 0 under BOX), and it covers '
          'at least the single-sample plane\'s interior',
          cov_s is not None and cov_s.shape == (H, W)
          and np.array_equal(cov_s, img_s[..., 3] > 0.0)
          and int(cov_s.sum()) >= int(sc_t.last_coverage.sum()) - 4,
          f'{None if cov_s is None else int(cov_s.sum())}')

    # (3) the twin: W = 100 (not a multiple of 64), a random plane
    for T in (15, 200):
        st = settings(super_black=True, super_black_threshold=T)
        ref = PE.super_black(rgb, cvr, st)
        d, err, a_ok = twin('SUPERBLACK', ref, 20, 100,
                            {'resolution': (100.0, 20.0),
                             'threshold': float(PE.super_black_threshold(st))},
                            {'source': rgb,
                             'coverage': PE.pack_coverage(cvr)})
        check(f'SUPERBLACK twin at threshold {T}, W = 100: the simulator is '
              'bitwise PE.super_black, alpha 1',
              d == 0.0 and a_ok, str(err or d))
    packed = PE.pack_coverage(cvr)
    check('pack_coverage: (H, ceil(W / 64), 4) float32 of exact integers '
          'below 2^16, and unpack_coverage round-trips bitwise',
          packed.shape == (20, 2, 4) and packed.dtype == np.float32
          and float(packed.max()) < 65536.0
          and np.array_equal(packed, np.round(packed))
          and np.array_equal(PE.unpack_coverage(packed, 100), cvr))
    wide = _RNG.random((3, 257)) > 0.3
    check('pack_coverage round-trips at W = 257 and W = 64',
          np.array_equal(PE.unpack_coverage(PE.pack_coverage(wide), 257), wide)
          and np.array_equal(PE.unpack_coverage(
              PE.pack_coverage(wide[:, :64]), 64), wide[:, :64]))
    dev, rec = fake_row_cov('Super Black 15 at exposure 0.1',
                            settings(super_black=True, exposure=0.1), 0.0,
                            dict(stages=['DISPLAY', 'SUPERBLACK', 'QUANT'],
                                 readbacks=0, uploads=0))
    dev0, _rec0 = fake_row_cov('exposure 0.1 without Super Black',
                               settings(exposure=0.1), 0.0,
                               dict(stages=['DISPLAY', 'QUANT'], readbacks=0,
                                    uploads=0))
    check("the packed plane is ONE device.upload per frame (the frame's own "
          'uploads stay 0)',
          dev.calls['upload'] == dev0.calls['upload'] + 1,
          f"{dev.calls['upload']} vs {dev0.calls['upload']}")
    fake_row_cov('Super Black 64 at exposure 0.1',
                 settings(super_black=True, super_black_threshold=64,
                          exposure=0.1), 0.0,
                 dict(stages=['DISPLAY', 'SUPERBLACK', 'QUANT'], readbacks=0,
                      uploads=0))
    out_g, out_c, _rec, _live, _dev = gpu_road_cov(
        settings(super_black=True, exposure=0.1))
    check('the fake-device Super Black frame is not vacuous: it differs from '
          'the off frame and equals the CPU road',
          np.array_equal(out_g, out_c) and not np.array_equal(out_g, off))

    # (4) the named fallbacks and the refusal
    st_n = settings(exposure=0.1, super_black=True)
    sc_n = demo_scene(st_n, with_texture=False)
    img_n = np.asarray(R.render(sc_n, st_n))
    st_n._last_coverage = None
    PO._POST_WARNED.clear()
    out_n = PO.process(img_n, st_n, **post_kw(sc_n, st_n))
    check('no plane under an opaque film: the stage is SKIPPED by name '
          '(bitwise the off frame, the console line recorded)',
          np.array_equal(out_n, off)
          and any('super black: skipped' in m and 'opaque' in m
                  for m in PO._POST_WARNED), str(sorted(PO._POST_WARNED)))
    st_p = st_n.copy()
    st_p.use_processes = True
    PO._POST_WARNED.clear()
    out_p = PO.process(img_n, st_p, **post_kw(sc_n, st_p))
    check('no plane under the process pool: the skip names the pool',
          np.array_equal(out_p, off)
          and any('process pool' in m for m in PO._POST_WARNED),
          str(sorted(PO._POST_WARNED)))
    st_a = st_t.copy()
    out_a = PO.process(img_t, st_a, **post_kw(sc_t, st_a))
    a_cov = img_t[..., 3] > 0.0
    check('no plane under a transparent film: the alpha road engages and '
          'covered pixels are floored',
          float(out_a[..., :3][a_cov].min()) >= float(thr15))
    check('superblack_refusal is None (the no-coverage case is a named skip, '
          'not a refusal) and chain.superblack is the door',
          CP.superblack_refusal(st15) is None
          and chain.superblack is CP.superblack)
    wrong = np.ones((H // 2, W), bool)
    PO._POST_WARNED.clear()
    out_w = PO.process(img_n, st_n, **dict(post_kw(sc_n, st_n),
                                           coverage=wrong))
    check('a plane of another size counts as no plane (skipped by name)',
          np.array_equal(out_w, off) and len(PO._POST_WARNED) == 1)
    plane_a = np.zeros((H, W), bool)
    plane_a[:, :W // 2] = True
    plane_b = ~plane_a
    st_k = settings(exposure=0.1, super_black=True)
    st_k._last_coverage = plane_b
    out_k = PO.process(img_n, st_k, **dict(post_kw(sc_n, st_k),
                                           coverage=plane_a))
    check('the kwarg wins over the private: coverage=plane_a floors by '
          'plane_a although st._last_coverage is plane_b',
          float(out_k[..., :3][plane_a].min()) >= float(thr15)
          and np.array_equal(out_k[plane_b], off[plane_b]))

    # (5) the FM rows, the tooltips
    check("the featurematrix carries 'Super Black 15 (3D Studio / Max)' and "
          "'Super Black 64 (3D Studio / Max)'",
          fm_row('Super Black 15 (3D Studio / Max)')
          and fm_row('Super Black 64 (3D Studio / Max)'))
    PR = _properties()
    check('the two Super Black tooltips are sentences of 40+ characters '
          'naming the mechanism and the machine; the threshold ranges 0..255',
          len(PR.DESCRIPTIONS['super_black']) >= 40
          and '3ds Max' in PR.DESCRIPTIONS['super_black']
          and 'process pool' in PR.DESCRIPTIONS['super_black']
          and len(PR.DESCRIPTIONS['super_black_threshold']) >= 40
          and PR.RANGES['super_black_threshold'] == (0, 255))

    # (6) the roads the frame takes
    def road(**kw):
        st_on = settings(exposure=0.1, super_black=True, **kw)
        sc_on, _img_on, out_on = cpu_frame(st_on)
        _sc2, _img2, out_off = cpu_frame(settings(exposure=0.1, **kw))
        return st_on, sc_on, out_on, out_off

    for mode in ('SBS', 'CROSS'):
        st_on, sc_on, out_on, out_off = road(stereo_mode=mode)
        c = getattr(st_on, '_last_coverage', None)
        ok = c is not None and c.shape == out_on.shape[:2] \
            and bool(c[:, :W // 2].any()) and bool(c[:, W // 2:].any()) \
            and c is sc_on.last_coverage
        check(f'stereo {mode}: the plane has the packed frame\'s shape with '
              'covered pixels in both halves', ok,
              f'{None if c is None else c.shape}')
        check(f'stereo {mode}: the pixel law on the packed frame (covered '
              'floored, uncovered bitwise the off frame)',
              ok and float(out_on[..., :3][c].min()) >= float(thr15)
              and np.array_equal(out_on[~c], out_off[~c]))
    # SBS packs the two eyes' own planes, every other column
    st_e = settings(exposure=0.1, super_black=True, stereo_mode='ANAGLYPH')
    sc_e = demo_scene(st_e, with_texture=False)
    R.render(sc_e, st_e)
    cov_an = st_e._last_coverage
    st_b = settings(exposure=0.1, super_black=True, stereo_mode='SBS')
    sc_b = demo_scene(st_b, with_texture=False)
    R.render(sc_b, st_b)
    cov_sbs = st_b._last_coverage
    planes = []
    for sign in (-1.0, +1.0):
        st_i = settings(exposure=0.1, super_black=True)
        st_i._stereo = (sign * float(st_i.stereo_eye_distance) * 0.5,
                        max(float(st_i.stereo_convergence), 1e-3))
        sc_i = demo_scene(st_i, with_texture=False)
        cam = sc_i.camera
        import copy as _copy
        cam2 = _copy.copy(cam)
        mw = np.asarray(cam.matrix_world, np.float32).copy()
        right = mw[:3, 0] / max(float(np.linalg.norm(mw[:3, 0])), 1e-9)
        mw[:3, 3] = mw[:3, 3] + right * (sign * float(st_i.stereo_eye_distance)
                                         * 0.5)
        cam2.matrix_world = mw
        sc_i.camera = cam2
        R.render(sc_i, st_i)
        planes.append(st_i._last_coverage)
    covL, covR = planes
    check('stereo ANAGLYPH: the plane equals covL | covR of two eye renders '
          'made the way eye_frame makes them',
          cov_an is not None and covL is not None and covR is not None
          and np.array_equal(cov_an, covL | covR))
    check('stereo SBS: the plane is the two eyes\' planes squeezed (every '
          'other column), left | right',
          cov_sbs is not None and covL is not None
          and np.array_equal(cov_sbs, np.concatenate(
              [covL[:, ::2], covR[:, ::2]], axis=1)[:, :W]))
    odd = [np.zeros((2, 5), bool), np.ones((2, 5), bool)]
    check('_pack_stereo_coverage at an odd width pads the seam `edge` to the '
          'picture\'s width; None when an eye has no plane',
          R._pack_stereo_coverage(odd, 'SBS', 5).shape == (2, 5)
          and R._pack_stereo_coverage(odd, 'CROSS', 5)[0].tolist()
          == [True, True, True, False, False]
          and R._pack_stereo_coverage([None, odd[1]], 'SBS', 5) is None)

    st_on, sc_on, out_on, out_off = road(aa_mode='ACCUMULATE', aa_samples=4)
    c_acc = getattr(st_on, '_last_coverage', None)
    passes = []
    for k in range(4):
        st_k2 = settings(exposure=0.1, super_black=True, aa_mode='NONE')
        st_k2._accum_jitter = (R._halton(k + 1, 2) - 0.5,
                               R._halton(k + 1, 3) - 0.5)
        sc_k2 = demo_scene(st_k2, with_texture=False)
        R.render(sc_k2, st_k2)
        passes.append(st_k2._last_coverage)
    check('ACCUMULATE at 4 samples: the plane is the OR of the four passes\' '
          'planes (any sample counts as covered)',
          c_acc is not None and all(p is not None for p in passes)
          and np.array_equal(c_acc, passes[0] | passes[1] | passes[2]
                             | passes[3]) and c_acc is sc_on.last_coverage)
    check('ACCUMULATE: the pixel law holds on the averaged frame',
          c_acc is not None
          and float(out_on[..., :3][c_acc].min()) >= float(thr15)
          and np.array_equal(out_on[~c_acc], out_off[~c_acc]))

    st_pn = settings(exposure=0.1, super_black=True)
    sc_pn = demo_scene(st_pn, with_texture=False)
    sc_pn.camera.type = 'PANO'
    st_pn._last_coverage = np.ones((H, W), bool)       # a stale plane
    img_pn = np.asarray(R.render(sc_pn, st_pn))
    PO._POST_WARNED.clear()
    out_pn = PO.process(img_pn, st_pn, **post_kw2(sc_pn, st_pn))
    st_pf = settings(exposure=0.1)
    sc_pf = demo_scene(st_pf, with_texture=False)
    sc_pf.camera.type = 'PANO'
    img_pf = np.asarray(R.render(sc_pf, st_pf))
    out_pf = PO.process(img_pf, st_pf, **post_kw2(sc_pf, st_pf))
    check('PANO: no honest stitched plane -- the plane is None (a stale one '
          'is cleared), the picture is bitwise super_black off, the skip is '
          'named',
          st_pn._last_coverage is None and sc_pn.last_coverage is None
          and np.array_equal(out_pn, out_pf)
          and any('super black: skipped' in m for m in PO._POST_WARNED))
    st_pp = settings(exposure=0.1, super_black=True, pano_parts=3)
    sc_pp = demo_scene(st_pp, with_texture=False)
    st_pp._last_coverage = np.ones((H, W), bool)
    R.render(sc_pp, st_pp)
    check('Pano Parts: the PANO rule (the plane is None)',
          st_pp._last_coverage is None and sc_pp.last_coverage is None)

    st_ad = settings(exposure=0.1, super_black=True, aa_mode='ADAPTIVE',
                     aa_samples=4)
    if 'ADAPTIVE' in [i[0] for i in _properties().ENUMS['aa_mode']]:
        sc_ad = demo_scene(st_ad, with_texture=False)
        R.render(sc_ad, st_ad)
        st_ba = settings(exposure=0.1, super_black=True)
        sc_ba = demo_scene(st_ba, with_texture=False)
        R.render(sc_ba, st_ba)
        check('ADAPTIVE refine: the BASE frame\'s plane is put back after the '
              'refine passes (scene and settings)',
              sc_ad.last_coverage is not None
              and np.array_equal(sc_ad.last_coverage, sc_ba.last_coverage)
              and st_ad._last_coverage is sc_ad.last_coverage,
              str(R.LAST_ADAPTIVE.get('passes')))

    ENG = _engine()
    ENG._HOLD_CACHE.clear()
    st_h = settings(exposure=0.1, super_black=True, film_hold=2)
    ENG.hold_store('Scene', 1, (W, H), st_h, img15, None, None, None,
                   gpu=False, coverage=cov)
    key, hit = ENG.hold_lookup('Scene', 2, 1, 2, (W, H), st_h)
    check('held frames: hold_store keeps the key frame\'s plane and '
          'hold_lookup returns it under \'coverage\'',
          key == 1 and hit is not None and hit.get('coverage') is cov)
    st_h2 = settings(exposure=0.1, super_black=False, film_hold=2)
    _key, hit2 = ENG.hold_lookup('Scene', 2, 1, 2, (W, H), st_h2)
    check('the hold fingerprint covers super_black (a key frame rendered '
          'with it on is not reused with it off)', hit2 is None)
    ENG._HOLD_CACHE.clear()

    check("the capability row 'super_black' is BOTH and its stage is graded",
          'super_black' in __import__(
              __package__.rsplit('.tests', 1)[0] + '.gpu.capability',
              fromlist=['FEATURES']).FEATURES
          and stages.VALIDATION['SUPERBLACK'][0] == 'EXACT')


# --------------------------------------------- C093: Video Color Check


def _yc(px):
    """(y, c) of an RGB triple with the catalogue's 3-decimal YIQ."""
    y, c2 = PE.legal_terms(np.asarray(px, np.float32).reshape(1, 1, 3))
    return float(y[0, 0]), float(np.sqrt(c2[0, 0]))


def test_c093_video_color_check():
    MODES = ('FLAG_BLACK', 'SCALE_LUMA', 'SCALE_SAT')

    # (1) identity
    st0 = settings()
    check('Video Color Check defaults: NONE, NTSC, IRE_120',
          st0.video_color_check == 'NONE' and st0.video_system == 'NTSC'
          and st0.video_ire_limit == 'IRE_120')
    legal = np.array([[(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.0, 0.0, 0.0),
                       (0.4, 0.5, 0.45)]], np.float32)
    check('legal pixels (mid grey, white, black, a soft tint) are the '
          'identity under all three modes, bitwise',
          all(np.array_equal(PE.video_color_check(
              legal, settings(video_color_check=m)), legal) for m in MODES))
    img = random_image(8, 8)
    check('NONE returns the input object untouched',
          PE.video_color_check(img, settings()) is img)
    hi, lo = PE.legal_limits(settings())
    hi110, _lo = PE.legal_limits(settings(video_ire_limit='IRE_110'))
    hip, lop = PE.legal_limits(settings(video_system='PAL'))
    check('the envelope numbers: NTSC 120 IRE -> 1.2162162 / -0.2972973, 110 '
          'IRE -> 1.1081082, PAL -> 1.2 / -0.2 (float32)',
          hi == np.float32(112.5 / 92.5) and lo == np.float32(-27.5 / 92.5)
          and hi110 == np.float32(102.5 / 92.5) and hip == np.float32(1.2)
          and lop == np.float32(-0.2)
          and abs(float(hi) - 1.2162162) < 1e-6
          and abs(float(lo) + 0.2972973) < 1e-6)

    # (2) the laws on the probe row
    probe = np.array([[(1, 1, 0), (0, 0, 1), (1, 0, 1), (0.5, 0.5, 0.5),
                       (1, 1, 1), (1, 0, 0)]], np.float32)
    fb = PE.video_color_check(probe, settings(video_color_check='FLAG_BLACK'))
    y_y, c_y = _yc((1, 1, 0))
    y_b, c_b = _yc((0, 0, 1))
    check('the catalogue\'s numbers reproduced: yellow y + c = 1.334 (about '
          '131 IRE), blue y - c = -0.334 (about -23 IRE)',
          abs((y_y + c_y) - 1.334) < 2e-3 and abs((y_b - c_b) + 0.334) < 2e-3
          and abs(((y_y + c_y) * 92.5 + 7.5) - 131.0) < 1.0
          and abs(((y_b - c_b) * 92.5 + 7.5) + 23.0) < 1.0,
          f'{y_y + c_y} {y_b - c_b}')
    check('FLAG_BLACK: yellow (a peak) and blue (a trough) turn black; grey '
          'and white stay (100 IRE is legal)',
          not fb[0, 0].any() and not fb[0, 1].any()
          and np.array_equal(fb[0, 3], probe[0, 3])
          and np.array_equal(fb[0, 4], probe[0, 4]))
    fb110 = PE.video_color_check(probe, settings(
        video_color_check='FLAG_BLACK', video_ire_limit='IRE_110'))
    check('FLAG_BLACK at 110 IRE: magenta\'s verdict is the same as at 120 '
          'and yellow stays illegal',
          bool(fb110[0, 2].any()) == bool(fb[0, 2].any())
          and not fb110[0, 0].any())
    rgb = push_saturation(random_image(48, 64))
    y_in, c2_in = PE.legal_terms(rgb)
    n_bad = int(((y_in + np.sqrt(c2_in) > hi) | (y_in - np.sqrt(c2_in) < lo)).sum())
    check('the pushed random image holds more than 100 illegal pixels',
          n_bad > 100, str(n_bad))
    for sysm, lim in (('NTSC', 'IRE_120'), ('PAL', 'IRE_110')):
        kw = dict(video_system=sysm, video_ire_limit=lim)
        h_, l_ = PE.legal_limits(settings(**kw))
        sl = PE.video_color_check(rgb, settings(video_color_check='SCALE_LUMA',
                                                **kw))
        y_o, c2_o = PE.legal_terms(sl)
        check(f'SCALE_LUMA {sysm} {lim}: every output pixel sits under the '
              'peak (y + c <= hi + 1e-5)',
              float((y_o + np.sqrt(c2_o)).max()) <= float(h_) + 1e-5,
              str(float((y_o + np.sqrt(c2_o)).max()) - float(h_)))
        ss_ = PE.video_color_check(rgb, settings(video_color_check='SCALE_SAT',
                                                 **kw))
        y_s, c2_s = PE.legal_terms(ss_)
        c_in = np.sqrt(c2_in)
        # a luma-only violation (a grey above the peak) is Max's own
        # disclosed limit of Scale Saturation; none exists below white
        check(f'SCALE_SAT {sysm} {lim}: the luma is kept (within 4e-6) and '
              'every pixel sits inside the envelope after',
              float(np.abs(y_s - y_in)[c_in > 1e-6].max()) <= 4e-6
              and float((y_s + np.sqrt(c2_s)).max()) <= float(h_) + 1e-5
              and float((y_s - np.sqrt(c2_s)).min()) >= float(l_) - 1e-5,
              f'{float(np.abs(y_s - y_in).max())} '
              f'{float((y_s + np.sqrt(c2_s)).max()) - float(h_)}')
    # SCALE_LUMA keeps the hue: a pure peak pixel is the input times one scalar
    sl = PE.video_color_check(rgb, settings(video_color_check='SCALE_LUMA'))
    a_ = hi - y_in
    b_ = y_in - lo
    peak = (a_ < 0) | (c2_in > a_ * a_)
    trough = c2_in > b_ * b_
    cc_in = np.sqrt(c2_in)
    s_ = hi / (y_in + cc_in)
    only_peak = peak & ~((cc_in * s_) > (y_in * s_ - lo))
    ratio = sl[only_peak] / np.maximum(rgb[only_peak], 1e-9)
    big = rgb[only_peak] > 0.05
    spread = np.where(big, ratio, np.nan)
    check('SCALE_LUMA: a peak-only pixel is the input times ONE scalar (hue '
          'and saturation kept: out / in constant across channels)',
          int(only_peak.sum()) > 20
          and float(np.nanmax(np.nanmax(spread, 1) - np.nanmin(spread, 1)))
          < 1e-5, str(int(only_peak.sum())))
    # PAL versus NTSC on a constructed pixel with y + c in (1.2, 1.216)
    px = None
    for t in np.linspace(0.0, 1.0, 2001):
        cand = np.array([1.0, 1.0, t], np.float32)
        yy_, cc_ = _yc(cand)
        if 1.2005 < yy_ + cc_ < 1.2155:
            px = cand.reshape(1, 1, 3)
            break
    check('PAL versus NTSC: a pixel with y + c between 1.2 and 1.216 is legal '
          'under NTSC and flagged under PAL',
          px is not None
          and np.array_equal(PE.video_color_check(
              px, settings(video_color_check='FLAG_BLACK')), px)
          and not PE.video_color_check(
              px, settings(video_color_check='FLAG_BLACK',
                           video_system='PAL')).any(), str(px))
    # the overlap law: a trough with no peak -- SCALE_LUMA == SCALE_SAT
    only_trough = trough & ~peak
    ss_n = PE.video_color_check(rgb, settings(video_color_check='SCALE_SAT'))
    check('the overlap law: on trough-only pixels SCALE_LUMA equals SCALE_SAT '
          'bitwise (no hidden fourth mode)',
          int(only_trough.sum()) > 5
          and np.array_equal(sl[only_trough], ss_n[only_trough]),
          str(int(only_trough.sum())))
    check('determinism: two calls are bitwise equal, float32, inside 0..1',
          all(np.array_equal(
              PE.video_color_check(rgb, settings(video_color_check=m)),
              PE.video_color_check(rgb, settings(video_color_check=m)))
              for m in MODES) and sl.dtype == np.float32
          and float(sl.min()) >= 0.0 and float(sl.max()) <= 1.0)
    _sc, _i, base = cpu_frame(settings(saturation=2.0))
    outs = {}
    for m in MODES:
        _sc, _i, outs[m] = cpu_frame(settings(saturation=2.0,
                                              video_color_check=m))
        check(f'{m} on the demo scene at saturation 2 changes the picture',
              not np.array_equal(outs[m], base),
              str(int((outs[m] != base).any(-1).sum())))
    _sc, _i, out_pal = cpu_frame(settings(
        saturation=2.0, video_color_check='FLAG_BLACK', video_system='PAL',
        video_ire_limit='IRE_110'))
    black = lambda o: int((o[..., :3] == 0.0).all(-1).sum())
    check('PAL at 110 IRE flags more demo pixels than NTSC at 120 (the FM '
          'row is not vacuous)', black(out_pal) > black(outs['FLAG_BLACK']),
          f'{black(out_pal)} vs {black(outs["FLAG_BLACK"])}')
    check('the stage sits after the display transform and before the '
          'halftone in post.process (source order)',
          (lambda s: s.index("_gpu('display'") < s.index("_gpu('legalise'")
           < s.index("_gpu('superblack'") < s.index("'halftone'"))(
              __import__('inspect').getsource(PO.process)))

    # (3) the twin: three modes x NTSC / PAL x both limits
    for m in MODES:
        for sysm in ('NTSC', 'PAL'):
            for lim in ('IRE_120', 'IRE_110'):
                st = settings(video_color_check=m, video_system=sysm,
                              video_ire_limit=lim)
                ref = PE.video_color_check(rgb, st)
                h_, l_ = PE.legal_limits(st)
                d, err, a_ok = twin('LEGALISE', ref, 48, 64,
                                    {'resolution': (64.0, 48.0),
                                     'mode': PE.LEGAL_MODES[m],
                                     'hi': float(h_), 'lo': float(l_)},
                                    {'source': rgb})
                check(f'LEGALISE twin {m} {sysm} {lim}: the simulator is '
                      'bitwise PE.video_color_check, alpha 1, and the '
                      'picture moved',
                      d == 0.0 and a_ok and not np.array_equal(ref, rgb),
                      str(err or d))
    d, err, _a = twin('LEGALISE',
                      PE.video_color_check(probe, settings(
                          video_color_check='SCALE_SAT')), 1, 6,
                      {'resolution': (6.0, 1.0), 'mode': 3, 'hi': float(hi),
                       'lo': float(lo)}, {'source': probe})
    check('LEGALISE twin on the primaries probe row (SCALE_SAT), bitwise',
          d == 0.0, str(err or d))
    fake_row('Video Color Check FLAG_BLACK at saturation 2',
             settings(video_color_check='FLAG_BLACK', saturation=2.0), 0.0,
             dict(stages=['DISPLAY', 'LEGALISE', 'QUANT'], readbacks=0,
                  uploads=0))
    fake_row('Video Color Check SCALE_SAT at saturation 2',
             settings(video_color_check='SCALE_SAT', saturation=2.0), 0.0,
             dict(stages=['DISPLAY', 'LEGALISE', 'QUANT'], readbacks=0,
                  uploads=0))
    fake_row('Video Color Check SCALE_LUMA, PAL at 110 IRE',
             settings(video_color_check='SCALE_LUMA', saturation=2.0,
                      video_system='PAL', video_ire_limit='IRE_110'), 0.0,
             dict(stages=['DISPLAY', 'LEGALISE', 'QUANT'], readbacks=0,
                  uploads=0))
    fake_row_cov('Video Color Check then Super Black (the stage order)',
                 settings(video_color_check='SCALE_SAT', saturation=2.0,
                          super_black=True), 0.0,
                 dict(stages=['DISPLAY', 'LEGALISE', 'SUPERBLACK', 'QUANT'],
                      readbacks=0, uploads=0))

    # (4) the refusal; a future item never draws the identity silently
    check('legalise_refusal is None and chain.legalise is the door',
          CP.legalise_refusal(settings(video_color_check='SCALE_SAT')) is None
          and chain.legalise is CP.legalise)
    raised = False
    try:
        CP.legalise(chain.Frame(rgb=rgb.copy()), settings())
    except AssertionError:
        raised = True
    check('CP.legalise asserts on an unknown mode (NONE is gated by '
          'post.process; no silent identity draw)', raised)
    check('the LEGALISE grade is CLOSE at 1e-5 (one sqrt, one division on a '
          'driver; bitwise in the simulator)',
          stages.VALIDATION['LEGALISE'] == ('CLOSE', 0.00001)
          and 'LEGALISE' in stages.ENABLED)

    # (5) the FM rows, the tooltips, the presets
    check('the featurematrix carries the three video colour check rows',
          fm_row('video colour check FLAG_BLACK (Max)')
          and fm_row('video colour check SCALE_SAT (Max)')
          and fm_row('video colour check PAL at 110 IRE (Max)'))
    PR = _properties()
    ids = lambda t: [i[0] for i in t]
    check('the three enums and their tooltips (40+ characters, the machine '
          'named)',
          ids(PR.ENUMS['video_color_check'])
          == ['NONE', 'FLAG_BLACK', 'SCALE_LUMA', 'SCALE_SAT']
          and ids(PR.ENUMS['video_system']) == ['NTSC', 'PAL']
          and ids(PR.ENUMS['video_ire_limit']) == ['IRE_120', 'IRE_110']
          and all(len(PR.DESCRIPTIONS[k]) >= 40 for k in
                  ('video_color_check', 'video_system', 'video_ire_limit'))
          and '3ds Max' in PR.DESCRIPTIONS['video_color_check'])
    sg = PRESETS['SGI_BROADCAST']['settings']
    check('SGI_BROADCAST legalises with Scale Saturation (its note promises '
          'broadcast-legal colour); MAX_2012 / MAX_R2 / STUDIO_R4 / '
          'LIGHTWAVE_56 ship the pair off and say so',
          sg.get('video_color_check') == 'SCALE_SAT'
          and all('video_color_check' not in PRESETS[k]['settings']
                  and 'super_black' not in PRESETS[k]['settings']
                  and 'Video Color Check' in PRESETS[k]['note']
                  for k in ('MAX_2012', 'MAX_R2', 'STUDIO_R4',
                            'LIGHTWAVE_56')))


ORDER += [test_c092_super_black, test_c093_video_color_check]



# ------------------------------------------------ C050: attribute cells


def cells_inputs(mode, st, rgb):
    """The CELLS pair's transports, built from the pack's own functions
    the way CP.cells builds them."""
    h, w = rgb.shape[:2]
    bg = PE.cells_bg(rgb, st, 0) if mode == 'C64_MULTI' else 0
    _mach, idx = PE.cells_sets(mode, bg)
    cw, ch = PE.CELL_SIZE[mode]
    cx, cy = -(-w // cw), -(-h // ch)
    kind = PE.cells_dither(st)
    m = DI.ORDERED.get(kind) if kind else None
    tile = CP.tile_image(m) if m is not None \
        else np.zeros((1, 1, 4), np.float32)
    dith = {'strength': float(np.float32(float(st.dither_strength))),
            'spacing': float(np.float32(PE.cells_spacing(mode))),
            'tile_mask': int(m.shape[0] - 1) if m is not None else 0,
            'dither_on': int(m is not None)}
    fit_u = dict(dith, resolution=(float(cx), float(cy)),
                 frame_size=(float(w), float(h)), cell_w=int(cw),
                 cell_h=int(ch), n_sets=int(idx.shape[0]),
                 n_colors=int(idx.shape[1]))
    snap_u = dict(dith, resolution=(float(w), float(h)),
                  cell_shift_x=int(cw.bit_length() - 1),
                  cell_shift_y=int(ch.bit_length() - 1),
                  n_colors=int(idx.shape[1]))
    return (cx, cy), fit_u, snap_u, PE.cells_sets_image(mode, bg), tile


def cells_twin(mode, rgb, **kw):
    """(d, fit_ok, err): CELLS_FIT then CELLS_SNAP through the simulator
    against PE.attribute_cells (and the fit against PE.cells_fit)."""
    PA.clear_caches()
    st = settings(attribute_cells=mode, **kw)
    h, w = rgb.shape[:2]
    ref = PE.attribute_cells(rgb, st, 0)
    (cx, cy), fit_u, snap_u, sets, tile = cells_inputs(mode, st, rgb)
    fit, err = run_stage('CELLS_FIT', cy, cx, fit_u,
                         {'source': rgb, 'sets': sets, 'tile': tile})
    if fit is None:
        return -1.0, False, err
    fit_ok = bool(np.array_equal(fit[..., 0].astype(np.int32),
                                 PE.cells_fit(rgb, st, 0)))
    d, err, a_ok = twin('CELLS_SNAP', ref, h, w, snap_u,
                        {'source': rgb, 'cells': fit, 'sets': sets,
                         'tile': tile, 'lut255': chain.quant_lut((8, 8, 8))})
    return d, fit_ok and a_ok, err


def cell_colours(out, x0, x1, y0, y1):
    blk = np.ascontiguousarray(out[y0:y1, x0:x1].reshape(-1, 3))
    return {px.tobytes() for px in blk}


def test_c050_attribute_cells():
    MODES = ('ZX_SPECTRUM', 'MSX1', 'C64_HIRES', 'C64_MULTI')
    rgb = random_image(32, 48)

    # (1) identity
    st0 = settings()
    check('attribute_cells defaults to NONE and the era dispatcher answers '
          'None there', st0.attribute_cells == 'NONE'
          and PE.reduce_depth_era(rgb, st0, 0) is None
          and PE.era_mode(st0) is None)

    # the tables
    zx = PE.ZX_LEVELS
    check('ZX_LEVELS: black, blue, red, magenta, green, cyan, yellow, white '
          'at 215 (0xD7) normal / 255 bright; black is (0, 0, 0) in both',
          zx.shape == (2, 8, 3) and zx[0, 1].tolist() == [0, 0, 215]
          and zx[0, 2].tolist() == [215, 0, 0]
          and zx[0, 4].tolist() == [0, 215, 0]
          and zx[1, 6].tolist() == [255, 255, 0]
          and zx[1, 7].tolist() == [255, 255, 255]
          and not zx[:, 0].any()
          and len(np.unique(zx.reshape(16, 3), axis=0)) == 15)
    check('TMS9918_15: fifteen colours in VDP order 1..15 (black first, '
          'white last, medium green (33, 200, 66) second)',
          PE.TMS9918_15.shape == (15, 3)
          and PE.TMS9918_15[0].tolist() == [0, 0, 0]
          and PE.TMS9918_15[1].tolist() == [33, 200, 66]
          and PE.TMS9918_15[14].tolist() == [255, 255, 255]
          and len(np.unique(PE.TMS9918_15, axis=0)) == 15)
    shapes = {m: PE.cells_sets(m, 3)[1].shape for m in MODES}
    check('the candidate sets: ZX 128 x 2, MSX1 225 x 2, C64 hires 256 x 2, '
          'C64 multicolour 560 x 4 with the background first',
          shapes == {'ZX_SPECTRUM': (128, 2), 'MSX1': (225, 2),
                     'C64_HIRES': (256, 2), 'C64_MULTI': (560, 4)}
          and bool((PE.cells_sets('C64_MULTI', 3)[1][:, 0] == 3).all())
          and PE.cells_sets('C64_MULTI', 3)[1][0].tolist() == [3, 0, 1, 2]
          and PE.cells_sets('C64_MULTI', 3)[1][-1].tolist() == [3, 13, 14, 15],
          str(shapes))
    zidx = PE.cells_sets('ZX_SPECTRUM')[1]
    check('the shared BRIGHT in the sets: ink and paper of every ZX set come '
          'from the SAME brightness row',
          bool(((zidx[:, 0] // 8) == (zidx[:, 1] // 8)).all()))

    # (2) the laws
    outs = {}
    for m in MODES:
        PA.clear_caches()
        st = settings(attribute_cells=m)
        outs[m] = PO.reduce_depth(rgb, st, 0)
        mach, idx = PE.cells_sets(m, PE.cells_bg(rgb, st, 0)
                                  if m == 'C64_MULTI' else 0)
        pal = PE.LUT255[mach]
        check(f'{m}: every output pixel is a machine colour (membership), '
              'float32, and reduce_depth routes to PE.attribute_cells',
              outs[m].dtype == np.float32 and members(outs[m], pal)
              and np.array_equal(outs[m], PE.attribute_cells(rgb, st, 0)))
        cw, ch = PE.CELL_SIZE[m]
        cap = 4 if m == 'C64_MULTI' else 2
        worst = max(len(cell_colours(outs[m], x, x + cw, y, y + ch))
                    for y in range(0, 32, ch) for x in range(0, 48, cw))
        check(f'{m}: every {cw}x{ch} cell holds at most {cap} colours',
              worst <= cap, str(worst))
        check(f'{m}: the picture differs from the per-pixel nearest colour '
              '(the cell limit is doing something)',
              not np.array_equal(
                  outs[m], pal[PA.nearest_brute(
                      np.round(rgb.reshape(-1, 3) * 255) / 255,
                      pal)].reshape(rgb.shape)))
    # C64 multicolour: the cell's colours are a subset of {bg, a, b, c}
    PA.clear_caches()
    st_m = settings(attribute_cells='C64_MULTI')
    bg = PE.cells_bg(rgb, st_m, 0)
    r8 = np.round(np.clip(rgb, 0, 1) * np.float32(255)).astype(np.int32)
    c64 = np.round(PA.C64_16 * np.float32(255)).astype(np.int32)
    near = np.argmin(((r8[..., None, :] - c64) ** 2).sum(-1), axis=-1)
    want_bg = int(np.argmax(np.bincount(near.ravel(), minlength=16)))
    check('C64 multicolour: bg is the lowest index maximising the count of '
          'nearest-of-16 pixels (recomputed here in one expression)',
          bg == want_bg, f'{bg} vs {want_bg}')
    fit = PE.cells_fit(rgb, st_m, 0)
    midx = PE.cells_sets('C64_MULTI', bg)[1]
    lut_c = PE.LUT255[c64]
    ok = True
    for cy in range(4):
        for cx in range(12):
            allowed = {lut_c[i].tobytes() for i in midx[fit[cy, cx]]}
            ok &= cell_colours(outs['C64_MULTI'], cx * 4, cx * 4 + 4,
                               cy * 8, cy * 8 + 8) <= allowed
    check('C64 multicolour: every cell\'s colours are a subset of its chosen '
          '{bg, a, b, c}', ok)
    other = random_image(16, 16)[..., ::-1].copy()
    check('C64 multicolour under Lock Palette: the background counted from '
          'the first frame is reused for a second, different image',
          PE.cached_peek(PE.cells_bg_key(0)) == bg
          and PE.cells_bg(other, st_m, 0) == bg)
    st_u = settings(attribute_cells='C64_MULTI', palette_lock=False)
    check('with the lock off the background is recounted from every frame',
          PE.cells_bg(other, st_u, 0) == PE.cells_bg_count(other))
    PA.clear_caches()
    # the clash: three saturated colours inside one 8x8 cell come out as two
    clash = np.zeros((16, 16, 3), np.float32)
    clash[:8, :3] = (1.0, 0.0, 0.0)
    clash[:8, 3:6] = (0.0, 1.0, 0.0)
    clash[:8, 6:8] = (0.0, 0.0, 1.0)
    cl = PE.attribute_cells(clash, settings(attribute_cells='ZX_SPECTRUM'), 0)
    check('the attribute clash: red, green and blue inside one ZX cell come '
          'out as TWO colours, and one of the three classes changed colour',
          len(cell_colours(cl, 0, 8, 0, 8)) == 2
          and len(cell_colours(clash, 0, 8, 0, 8)) == 3)
    # the shared BRIGHT: bright red beside normal blue
    br = np.zeros((8, 8, 3), np.float32)
    br[:, :4] = (1.0, 0.0, 0.0)
    br[:, 4:] = (0.0, 0.0, np.float32(215) / np.float32(255))
    bo = PE.attribute_cells(br, settings(attribute_cells='ZX_SPECTRUM'), 0)
    got = np.round(np.unique(bo.reshape(-1, 3), axis=0) * 255).astype(int)
    lv = {int(v) for v in got.ravel()} - {0}
    check('the shared BRIGHT: bright red beside normal blue resolves to two '
          'colours from ONE brightness row (levels all 215 or all 255)',
          len(got) == 2 and len(lv) == 1 and lv <= {215, 255},
          str(got.tolist()))
    # order freedom: the flipped image is the flip of the output
    even = random_image(16, 48)
    for m in ('ZX_SPECTRUM', 'MSX1', 'C64_HIRES'):
        st = settings(attribute_cells=m)
        a = PE.attribute_cells(even, st, 0)
        b = PE.attribute_cells(np.ascontiguousarray(even[:, ::-1]), st, 0)
        # a tie between two sets holding the same two colours in the other
        # order (ink <-> paper) gives the same picture, so the law is exact
        check(f'{m}: order-free -- the output of the horizontally flipped '
              'image is the flip of the output',
              np.array_equal(b, a[:, ::-1]))
    # the ordered dither changes the picture; membership and the limit hold
    st_d = settings(attribute_cells='ZX_SPECTRUM', dither='BAYER8',
                    dither_strength=1.0)
    st_z = settings(attribute_cells='ZX_SPECTRUM', dither='BAYER8',
                    dither_strength=0.0)
    od = PE.attribute_cells(rgb, st_d, 0)
    zpal = PE.LUT255[PE.cells_sets('ZX_SPECTRUM')[0]]
    check('ZX + BAYER8 at strength 1 differs from strength 0 while '
          'membership and the two-colour limit hold; strength 0 is the NONE '
          'picture',
          not np.array_equal(od, outs['ZX_SPECTRUM']) and members(od, zpal)
          and max(len(cell_colours(od, x, x + 8, y, y + 8))
                  for y in range(0, 32, 8) for x in range(0, 48, 8)) <= 2
          and np.array_equal(PE.attribute_cells(rgb, st_z, 0),
                             outs['ZX_SPECTRUM']))
    # partial cells at the top / right edge are the pixels they have
    part = random_image(13, 21)
    op = PE.attribute_cells(part, settings(attribute_cells='C64_HIRES'), 0)
    sub = PE.attribute_cells(np.ascontiguousarray(part[8:, 16:]),
                             settings(attribute_cells='C64_HIRES'), 0)
    check('a partial cell at the top-right corner is fitted from the pixels '
          'it has (equals the fit of that block alone)',
          op.shape == part.shape and np.array_equal(op[8:, 16:], sub))
    check('the precedence: attribute cells own the stage over a scanline '
          'palette, CRY16 / YJK and EHB',
          PE.era_mode(settings(attribute_cells='MSX1',
                               scanline_palette='SHAM', color_depth='CRY16',
                               palette_mode='EHB')) == 'CELLS'
          and PE.era_mode(settings(scanline_palette='SHAM',
                                   color_depth='YJK')) == 'SCANLINE')

    # (3) the twins (the simulator runs K x cell x S iterations per batch)
    d, ok, err = cells_twin('ZX_SPECTRUM', random_image(16, 16))
    check('CELLS twin ZX_SPECTRUM 16x16, no dither: fit and snap bitwise the '
          'CPU, alpha 1', d == 0.0 and ok, str(err or d))
    d, ok, err = cells_twin('ZX_SPECTRUM', random_image(16, 16),
                            dither='BAYER4', dither_strength=0.6)
    check('CELLS twin ZX_SPECTRUM 16x16 + BAYER4 at 0.6: bitwise',
          d == 0.0 and ok, str(err or d))
    d, ok, err = cells_twin('MSX1', random_image(8, 16))
    check('CELLS twin MSX1 16x8 (8x1 runs): bitwise', d == 0.0 and ok,
          str(err or d))
    k = np.arange(255, dtype=np.float32)
    v = ((k + np.float32(0.5)) / np.float32(255)).astype(np.float32)
    v = v[(v * np.float32(255)).astype(np.float32) == (k + np.float32(0.5))]
    n_t = (len(v) // 8) * 8
    ties = np.ascontiguousarray(
        np.stack([v[:n_t], v[:n_t][::-1], np.roll(v[:n_t], 5)], 1)[None],
        np.float32)
    d, ok, err = cells_twin('MSX1', ties)
    check(f'CELLS tie probe: {n_t} exact halves of k/255 round half to even '
          'on both roads (MSX1), bitwise', n_t > 0 and d == 0.0 and ok,
          str(err or d))
    d, ok, err = cells_twin('C64_HIRES', random_image(12, 13),
                            dither='BAYER8', dither_strength=1.0)
    check('CELLS twin C64_HIRES 13x12 + BAYER8 (partial cells at the top and '
          'right edges): bitwise', d == 0.0 and ok, str(err or d))
    d, ok, err = cells_twin('C64_MULTI', random_image(8, 8))
    check('CELLS twin C64_MULTI 8x8 (two 4x8 cells, 560 sets): bitwise',
          d == 0.0 and ok, str(err or d))
    PA.clear_caches()
    fake_row('ZX attribute cells + BAYER8',
             settings(attribute_cells='ZX_SPECTRUM', dither='BAYER8'), 0.0,
             dict(stages=['DISPLAY', 'CELLS'], readbacks=0, uploads=0))
    PA.clear_caches()
    fake_row('C64 multicolour, cold',
             settings(attribute_cells='C64_MULTI'), 0.0,
             dict(stages=['DISPLAY', 'CELLS'], readback_names=['cells'],
                  uploads=1))
    # warm: the background is in the lock cache -- no readback is spent
    # (the draws are recorded, not run: the cold row measured them)
    st_w = settings(attribute_cells='C64_MULTI')
    st_w.render_device, st_w.gpu_post = 'GPU', True
    with fakedevice.installed():
        fr_w = chain.Frame(rgb=random_image(H, W))
        got_w = CP.cells(fr_w, st_w)
        rec_w = (list(fr_w.stages), list(fr_w.readbacks), len(fr_w.pending))
        fr_w.release()
    check('C64 multicolour, warm: the locked background is peeked -- the '
          'CELLS pair is recorded (two draws) with no readback',
          got_w is not None and rec_w == (['CELLS'], [], 2), str(rec_w))
    # a refused gate never costs a readback
    PA.clear_caches()
    keep = chain.ENABLED
    chain.ENABLED = tuple(n for n in keep if n != 'CELLS_SNAP')
    try:
        with fakedevice.installed():
            fr_r = chain.Frame(rgb=random_image(H, W))
            got_r = CP.cells(fr_r, st_w)
            rec_r = (list(fr_r.stages), list(fr_r.readbacks))
            fr_r.release()
    finally:
        chain.ENABLED = keep
    check('the gate order: with CELLS_SNAP not enabled CP.cells returns None '
          'before any readback and before the background is counted',
          got_r is None and rec_r == ([], [])
          and PE.cached_peek(PE.cells_bg_key(0)) is None, str(rec_r))

    # (4) the refusals
    st_l = settings(attribute_cells='C64_MULTI', palette_lock=False)
    why = CP.cells_refusal(st_l)
    check('C64_MULTI with Lock Palette off refuses by name, and '
          'chain.quant_refusal returns that text',
          why is not None and 'Lock Palette' in why
          and chain.quant_refusal(st_l) == why, str(why))
    check('the fixed-set modes never refuse (quant_refusal None; era_road '
          'True), with or without an ordered dither or an inert one',
          all(chain.quant_refusal(settings(attribute_cells=m, dither=dk))
              is None and CP.era_road(settings(attribute_cells=m, dither=dk))
              for m in MODES for dk in ('NONE', 'BAYER4', 'FLOYD')))
    PE._NOTED.clear()
    fl = PE.attribute_cells(rgb, settings(attribute_cells='ZX_SPECTRUM',
                                          dither='FLOYD'), 0)
    check('FLOYD under ZX is inert (bitwise the NONE output), noted once, '
          'and NOT a refusal',
          np.array_equal(fl, outs['ZX_SPECTRUM'])
          and any('attribute cells' in m and 'FLOYD' in m
                  for m in PE._NOTED)
          and CP.cells_refusal(settings(attribute_cells='ZX_SPECTRUM',
                                        dither='FLOYD')) is None)
    st_col = settings(attribute_cells='ZX_SPECTRUM', dither='COLUMNS')
    check('a non-square ordered matrix (COLUMNS, 1 x 2) has no seat in the '
          'cell fit: inert on the CPU and refused by name on the GPU road',
          PE.cells_dither(st_col) is None
          and 'not a square tile' in str(CP.cells_refusal(st_col))
          and np.array_equal(PE.attribute_cells(rgb, st_col, 0),
                             outs['ZX_SPECTRUM']))

    # (5) the FM rows, the tooltip, the presets
    check('the featurematrix carries the two attribute-cell rows',
          fm_row('attribute cells ZX Spectrum')
          and fm_row('attribute cells C64 multicolour'))
    PR = _properties()
    check('ATTRIBUTE_CELLS has the five items in order and a tooltip of 40+ '
          'characters naming the mechanism and the machines',
          [i[0] for i in PR.ENUMS['attribute_cells']]
          == ['NONE', 'ZX_SPECTRUM', 'MSX1', 'C64_HIRES', 'C64_MULTI']
          and len(PR.DESCRIPTIONS['attribute_cells']) >= 40
          and 'ZX Spectrum' in PR.DESCRIPTIONS['attribute_cells']
          and 'least squared error' in PR.DESCRIPTIONS['attribute_cells'])
    zxp = PRESETS['ZX_SPECTRUM']['settings']
    check('ZX_SPECTRUM: attribute_cells ZX_SPECTRUM, colour depth 8 (a real '
          'item now), BAYER8 kept; C64: C64_MULTI at 160x200',
          zxp.get('attribute_cells') == 'ZX_SPECTRUM'
          and zxp.get('color_depth') == '8' and zxp.get('dither') == 'BAYER8'
          and PRESETS['C64']['settings'].get('attribute_cells') == 'C64_MULTI'
          and PRESETS['C64']['settings'].get('resolution_x') == 160)
    p = PRESETS.get('MSX1')
    check('the MSX1 preset exists (PLATFORM, attribute_cells MSX1, BAYER4, '
          '256x192) with a note',
          p is not None and p['category'] == 'PLATFORM'
          and p['settings'].get('attribute_cells') == 'MSX1'
          and p['settings'].get('dither') == 'BAYER4'
          and (p['settings'].get('resolution_x'),
               p['settings'].get('resolution_y')) == (256, 192)
          and 'TMS9918' in p['note'])
    st_p = preset_settings('MSX1')
    sc_p, _i, out_p = cpu_frame(st_p)
    tms = PE.LUT255[PE.TMS9918_15.astype(np.int32)]
    check('the MSX1 preset renders: every pixel a TMS9918 colour, at most '
          'two per eight-pixel run',
          members(out_p[..., :3], tms)
          and max(len(cell_colours(out_p[..., :3], x, x + 8, y, y + 1))
                  for y in range(H) for x in range(0, W, 8)) <= 2)


# --------------------------------------- C060: per-scanline palettes


def row_counts(out, zones=None):
    """The largest count of distinct colours in any row (or any zone)."""
    worst = 0
    for row in out:
        for x0, x1 in (zones or [(0, out.shape[1])]):
            worst = max(worst, len(np.unique(
                np.ascontiguousarray(row[x0:x1]), axis=0)))
    return worst


def test_c060_scanline_palette():
    MODES = ('SPECTRUM_512', 'DYNAMIC_HIRES', 'SHAM')
    rgb = random_image(24, 48)

    # (1) identity
    st0 = settings()
    check('scanline_palette defaults to NONE and the era dispatcher answers '
          'None there', st0.scanline_palette == 'NONE'
          and PE.reduce_depth_era(rgb, st0, 0) is None)

    # (2) the laws
    outs = {}
    for m in MODES:
        st = settings(scanline_palette=m)
        outs[m] = PO.reduce_depth(rgb, st, 0)
        check(f'{m}: reduce_depth routes to PE.scanline_palette, float32, '
              'two calls bitwise equal',
              outs[m].dtype == np.float32
              and np.array_equal(outs[m], PE.scanline_palette(rgb, st, 0))
              and np.array_equal(outs[m], PE.scanline_palette(rgb, st, 0)))
        L = PE.SCANLINE_LEVELS[m]
        lat = outs[m] * np.float32(L - 1)
        check(f'{m}: every output value is on the machine\'s lattice '
              f'({L} levels per channel)',
              float(np.abs(lat - np.round(lat)).max()) < 1e-4)
    z512 = PE.scanline_zones('SPECTRUM_512', 48)
    check('Spectrum 512: three zones a line (the last takes the remainder), '
          'at most 16 colours per third and 48 per line',
          z512 == [(0, 16), (16, 32), (32, 48)]
          and PE.scanline_zones('SPECTRUM_512', 50)[-1] == (33, 50)
          and row_counts(outs['SPECTRUM_512'], z512) <= 16
          and row_counts(outs['SPECTRUM_512']) <= 48
          and row_counts(outs['SPECTRUM_512']) > 16,
          f"{row_counts(outs['SPECTRUM_512'], z512)} / "
          f"{row_counts(outs['SPECTRUM_512'])}")
    check('Dynamic HiRes: at most 16 colours on any line, and more than 16 '
          'across the frame',
          row_counts(outs['DYNAMIC_HIRES']) <= 16
          and len(np.unique(outs['DYNAMIC_HIRES'].reshape(-1, 3), axis=0))
          > 16)
    # SHAM: a row holds its sixteen base colours plus hold-and-modify
    # colours -- more than 16, on the 4:4:4 lattice
    check('Sliced HAM: a line can hold more than its 16 base colours (the '
          'modifies), and the picture differs from Dynamic HiRes',
          row_counts(outs['SHAM']) > 16
          and not np.array_equal(outs['SHAM'], outs['DYNAMIC_HIRES']))
    # the smooth-vertical law on the machine's own lattice
    for m in MODES:
        L = PE.SCANLINE_LEVELS[m]
        ramp = np.zeros((L, 24, 3), np.float32)
        for j in range(L):
            ramp[j] = (np.float32(j) / np.float32(L - 1),
                       np.float32(L - 1 - j) / np.float32(L - 1),
                       np.float32(j // 2) / np.float32(L - 1))
        got = PE.scanline_palette(ramp, settings(scanline_palette=m), 0)
        check(f'{m}: a row-constant vertical ramp on the machine\'s lattice '
              'reproduces bitwise (each line\'s registers hold its colour; a '
              'sky runs smooth top to bottom)',
              np.array_equal(got, PA.snap_levels(ramp, L))
              and np.array_equal(got, ramp))
        full = PE.scanline_palette(rgb, settings(scanline_palette=m), 0)
        crop = PE.scanline_palette(np.ascontiguousarray(rgb[5:13]),
                                   settings(scanline_palette=m), 0)
        check(f'{m}: rows are independent -- rows 5..12 equal the output of '
              'the image cropped to those rows, bitwise',
              np.array_equal(full[5:13], crop))
    one = PO.reduce_depth(rgb, settings(color_depth='4',
                                        palette_mode='ADAPTIVE',
                                        palette_size=16), 0)
    PA.clear_caches()
    check('the picture differs from the single 16-colour adaptive palette '
          '(and holds more colours across the frame)',
          not np.array_equal(outs['DYNAMIC_HIRES'], one)
          and len(np.unique(outs['DYNAMIC_HIRES'].reshape(-1, 3), axis=0))
          > len(np.unique(one.reshape(-1, 3), axis=0)))
    # the fit reads the UNPERTURBED pixels: the tables are the same with
    # and without the dither, the pictures differ, the counts hold
    for m in ('SPECTRUM_512', 'DYNAMIC_HIRES'):
        st_n = settings(scanline_palette=m)
        st_b = settings(scanline_palette=m, dither='BAYER4')
        tn = PE.scanline_palettes(rgb, st_n)
        tb = PE.scanline_palettes(rgb, st_b)
        ob = PE.scanline_palette(rgb, st_b, 0)
        zones = PE.scanline_zones(m, 48)
        check(f'{m} + BAYER4: the register tables are bitwise the NONE '
              'tables (the fit is on the unperturbed pixels), the picture '
              'differs, every zone still holds at most 16 colours',
              tn.shape == (24, len(zones), 16, 3) and tn.dtype == np.float32
              and np.array_equal(tn, tb) and not np.array_equal(ob, outs[m])
              and row_counts(ob, zones) <= 16)
        rows = {r.tobytes() for r in tn[3, 0]}
        check(f'{m}: every pixel of a zone is one of that zone\'s sixteen '
              'registers',
              all(px.tobytes() in rows for px in np.ascontiguousarray(
                  outs[m][3, zones[0][0]:zones[0][1]])))
    PE._NOTED.clear()
    fl = PE.scanline_palette(rgb, settings(scanline_palette='DYNAMIC_HIRES',
                                           dither='FLOYD'), 0)
    check('FLOYD under a scanline palette is inert (bitwise NONE), noted '
          'once', np.array_equal(fl, outs['DYNAMIC_HIRES'])
          and any('scanline palette' in m_ for m_ in PE._NOTED))
    # the HAM refactor: sham_row IS ham_encode's row loop
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the HAM refactor pin runs)',
          RP is not None)
    if RP is not None:
        prev_pal = importlib.import_module(
            RP.__name__.rsplit('.', 1)[0] + '.palette')
        small = random_image(12, 20)
        for bits in (6, 8):
            now_img, now_base = PA.ham_encode(small, bits)
            old_img, old_base = prev_pal.ham_encode(small, bits)
            check(f'PA.ham_encode at HAM{bits} is bitwise the 1.89.0 '
                  'function (output and base) after the sham_row refactor',
                  np.array_equal(now_img, old_img)
                  and np.array_equal(now_base, old_base))
    q = (np.round(rgb[0] * np.float32(15)) / np.float32(15)).astype(np.float32)
    base16 = PA.snap_levels(PA.median_cut(rgb[0], 16), 16)
    row = PA.sham_row(q, base16)
    lat = row * np.float32(15)
    check('PA.sham_row: one line on the 4:4:4 lattice, deterministic, each '
          'pixel either a base colour or one channel away from its left '
          'neighbour',
          row.shape == q.shape and np.array_equal(row, PA.sham_row(q, base16))
          and float(np.abs(lat - np.round(lat)).max()) < 1e-4
          and all((row[x].tobytes() in {b.tobytes() for b in base16})
                  or int((row[x] != row[x - 1]).sum()) <= 1
                  for x in range(1, len(row))))

    # (3) no twin: the GPU road reads back by name and equals the CPU chain
    fake_row('Spectrum 512 (reads back by name)',
             settings(scanline_palette='SPECTRUM_512'), 0.0,
             dict(stages=['DISPLAY'], readback_names=['per-scanline']))

    # (4) the refusal by name
    for m in MODES:
        why = chain.quant_refusal(settings(scanline_palette=m))
        check(f'{m}: chain.quant_refusal names the per-scanline palette and '
              'the CPU' + (' and the HAM encode' if m == 'SHAM' else '')
              + '; era_road is False (no GPU stage attempted)',
              why is not None and 'per-scanline palette' in why
              and 'CPU' in why and m in why
              and (m != 'SHAM' or 'HAM encode' in why)
              and CP.era_road(settings(scanline_palette=m)) is False
              and CP.era_refusal(settings(scanline_palette=m)) == (True, why),
              str(why))
    with fakedevice.installed():
        fr_s = chain.Frame(rgb=rgb.copy())
        st_s = settings(scanline_palette='SHAM')
        st_s.render_device, st_s.gpu_post = 'GPU', True
        got_s = chain.quant(fr_s, st_s)
        rec_s = (list(fr_s.stages), len(fr_s.pending))
        fr_s.release()
    check('chain.quant under a scanline palette returns None and records '
          'no draw', got_s is None and rec_s == ([], 0), str(rec_s))

    # (5) the FM row, the tooltip, the preset
    check("the featurematrix carries 'per-scanline palette (Spectrum 512)'",
          fm_row('per-scanline palette (Spectrum 512)'))
    PR = _properties()
    items = PR.ENUMS['scanline_palette']
    check('SCANLINE_PALETTE has the four items, each naming its cost, and a '
          'tooltip of 40+ characters naming the machines and the CPU',
          [i[0] for i in items] == ['NONE', 'SPECTRUM_512', 'DYNAMIC_HIRES',
                                    'SHAM']
          and all('on the CPU' in i[2] for i in items[1:])
          and len(PR.DESCRIPTIONS['scanline_palette']) >= 40
          and 'Spectrum 512' in PR.DESCRIPTIONS['scanline_palette']
          and 'CPU' in PR.DESCRIPTIONS['scanline_palette'])
    p = PRESETS.get('SPECTRUM_512')
    check('the SPECTRUM_512 preset exists (PLATFORM, scanline_palette '
          'SPECTRUM_512, BAYER4, 320x200) with a note',
          p is not None and p['category'] == 'PLATFORM'
          and p['settings'].get('scanline_palette') == 'SPECTRUM_512'
          and p['settings'].get('dither') == 'BAYER4'
          and (p['settings'].get('resolution_x'),
               p['settings'].get('resolution_y')) == (320, 200)
          and '48 colours a line' in p['note'])
    st_p = preset_settings('SPECTRUM_512')
    _sc, _i, out_p = cpu_frame(st_p)
    zp = PE.scanline_zones('SPECTRUM_512', out_p.shape[1])
    lat = out_p[..., :3] * np.float32(7)
    check('the SPECTRUM_512 preset renders: 3:3:3 values, at most 16 colours '
          'per third of a line',
          float(np.abs(lat - np.round(lat)).max()) < 1e-4
          and row_counts(out_p[..., :3], zp) <= 16)
    check("the pack's wave-2 presets (MSX1, SPECTRUM_512) are in the library "
          'with a category and a note',
          all(PRESETS.get(k) is not None and PRESETS[k].get('category')
              and PRESETS[k].get('note') for k in ('MSX1', 'SPECTRUM_512')))


# ------------------------------------- the wave's remaining seams


def test_wave2_seams():
    """The roads C092's plane takes that its own test did not reach (lens
    passes, parallax layers), the wave-2 stage registry, the self test's
    table through the fake device, and the wave's identity at defaults."""
    thr15 = np.float32(15) / np.float32(255)
    # lens passes: the plane is the OR of every pass's
    st_l = settings(exposure=0.1, super_black=True, dof=True,
                    dof_method='LENS_ACCUMULATE')
    sc_l, _img_l, out_l = cpu_frame(st_l)
    c_l = getattr(st_l, '_last_coverage', None)
    _sc0, _i0, out_l0 = cpu_frame(settings(exposure=0.1, dof=True,
                                           dof_method='LENS_ACCUMULATE'))
    st_1 = settings(exposure=0.1, super_black=True)
    sc_1 = demo_scene(st_1, with_texture=False)
    R.render(sc_1, st_1)
    check('lens passes: the plane is the OR of the passes\' planes (at least '
          'the single-pass plane where the lens samples agree), the pixel '
          'law holds',
          c_l is not None and c_l.shape == (H, W) and c_l is sc_l.last_coverage
          and int(c_l.sum()) >= int(st_1._last_coverage.sum()) - 8
          and float(out_l[..., :3][c_l].min()) >= float(thr15)
          and np.array_equal(out_l[~c_l], out_l0[~c_l]),
          f'{None if c_l is None else int(c_l.sum())}')
    # parallax layers: each eye's plane is the centre's, shifted like the
    # picture, then packed
    st_p = settings(exposure=0.1, super_black=True, stereo_mode='SBS',
                    stereo_parallax_layers=True)
    sc_p, _img_p, out_p = cpu_frame(st_p)
    c_p = getattr(st_p, '_last_coverage', None)
    _s, _i, out_p0 = cpu_frame(settings(exposure=0.1, stereo_mode='SBS',
                                        stereo_parallax_layers=True))
    check('parallax layers: the plane is packed like the picture (covered '
          'pixels in both halves) and the pixel law holds on the pair',
          c_p is not None and c_p.shape == out_p.shape[:2]
          and bool(c_p[:, :W // 2].any()) and bool(c_p[:, W // 2:].any())
          and float(out_p[..., :3][c_p].min()) >= float(thr15)
          and np.array_equal(out_p[~c_p], out_p0[~c_p]),
          f'{None if c_p is None else c_p.shape}')
    # the registry
    want = {'SUPERBLACK', 'LEGALISE', 'CELLS_FIT', 'CELLS_SNAP'}
    check('the four wave-2 stages are in stages_palette.STAGES and merged '
          'into gpu/stages (STAGES, INTERFACE, VALIDATION, ENABLED)',
          want <= set(SP.STAGES) and want <= set(stages.STAGES)
          and want <= set(stages.INTERFACE) and want <= set(stages.ENABLED))
    check('every wave-2 chain door is a callable (legalise, superblack, '
          'cells + refusals, scanline_refusal) and the era table routes '
          'CELLS and SCANLINE',
          all(callable(getattr(CP, n, None)) for n in
              ('legalise', 'legalise_refusal', 'superblack',
               'superblack_refusal', 'cells', 'cells_refusal',
               'scanline_refusal'))
          and CP._ERA['CELLS'][0] is CP.cells
          and CP._ERA['SCANLINE'][0] is None)
    # the self test's stage table, through the fake device
    from . import fakebpy
    fakebpy.install()
    SELF = importlib.import_module(__package__.rsplit('.tests', 1)[0]
                                   + '.selftest')
    out = []
    with fakedevice.installed():
        SELF.gpu_stages(out)
    lines = {str(ln).split()[0]: str(ln) for ln in out if str(ln).strip()}
    for nm in ('SUPERBLACK', 'LEGALISE', 'CELLS'):
        ln = lines.get(nm, '')
        check(f'Run Self Test measures {nm} (through the fake device: '
              'compiled, run, max diff 0.00000 against its CPU reference)',
              ' ok ' in ln and '0.00000' in ln and 'FAILED' not in ln
              and 'SKIPPED' not in ln, ln.strip())
    check('the self test\'s single-draw table skips the CELLS pair (measured '
          'through the orchestrator instead)',
          'CELLS_FIT' not in lines and 'CELLS_SNAP' not in lines)
    # the wave's identity at defaults: every new field at its no-op value
    st = settings()
    check('every wave-2 setting defaults to its no-op value',
          (st.attribute_cells, st.scanline_palette, st.super_black,
           st.super_black_threshold, st.video_color_check, st.video_system,
           st.video_ire_limit)
          == ('NONE', 'NONE', False, 15, 'NONE', 'NTSC', 'IRE_120'))
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the wave-2 neutrality pin '
          'runs)', RP is not None)
    if RP is not None:
        prev_post = importlib.import_module(
            RP.__name__.rsplit('.', 1)[0] + '.post')
        for label, kw in (
                ('film_transparent', {'film_transparent': True}),
                ('SBS stereo at eye distance 0', {'stereo_mode': 'SBS',
                                                  'stereo_eye_distance': 0.0}),
                ('ACCUMULATE at 4 samples', {'aa_mode': 'ACCUMULATE',
                                             'aa_samples': 4}),
                ('SUPERSAMPLE 4 + exposure 0.1 + saturation 2',
                 {'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'exposure': 0.1,
                  'saturation': 2.0})):
            clear_palette_locks(RP)
            st = settings(**kw)
            sc = demo_scene(st, with_texture=False)
            now_r = np.asarray(R.render(sc, st))
            now = PO.process(now_r, st, **post_kw2(sc, st))
            st2 = settings(**kw)
            sc2 = demo_scene(st2, with_texture=False)
            prev_r = np.asarray(RP.render(sc2, st2))
            prev = prev_post.process(prev_r, st2, **post_kw(sc2, st2))
            check(f'{label}: render and post.process (the plane passed, the '
                  'stages off) are bitwise the 1.89.0 engine',
                  now_r.shape == prev_r.shape
                  and np.array_equal(now_r, prev_r)
                  and np.array_equal(now, prev))


ORDER += [test_c050_attribute_cells, test_c060_scanline_palette,
          test_wave2_seams]


# ---- post-pass-2 fixes ----
#
# Item 1 (field v4, C061 ATARI_ST / AMIGA_OCS on the RTX: 390,076 and
# 435,645 of 921,600 px between devices): DISPLAY ran on the driver ahead
# of the colour depth stage's readback for FLOYD, and a sequential error
# diffusion cascades the driver's gamma ulps. The display transform now
# runs on the CPU by name whenever such a kernel follows. The fake
# device's DISPLAY is the simulator's (bitwise the CPU), so the frame was
# bitwise here before the fix too: the RECORD is the check that moves --
# measured before the fix: stages ['DISPLAY'], one readback named
# 'colour depth (the FLOYD dither runs on the CPU)'.


def test_display_before_error_diffusion():
    props = _properties()
    enum = [it[0] for it in props.DITHER]
    seq = [k for k in enum if k in DI.KERNELS]
    check('the DITHER enum\'s error-diffusion kinds are exactly dither.py\'s '
          'KERNELS (FLOYD, JJN, STUCKI, ATKINSON, BURKES, SIERRA, SIERRA_LITE: '
          'each one walks error_diffusion, a sequential scan); NOISE and the '
          'ordered kinds are not among them',
          sorted(seq) == sorted(DI.KERNELS) == sorted(
              ['FLOYD', 'JJN', 'STUCKI', 'ATKINSON', 'BURKES', 'SIERRA',
               'SIERRA_LITE'])
          and 'NOISE' not in DI.KERNELS
          and not any(k in DI.KERNELS for k in DI.ORDERED), str(seq))

    def row(label, st, display_on_cpu):
        PA.clear_caches()
        clear_palette_locks()
        kind = str(st.dither)
        out_g, out_c, rec, live, _dev = gpu_road(st)
        rb = rec.get('readbacks') or []
        names = [r[0] for r in rb]
        stages_ = rec.get('stages') or []
        if display_on_cpu:
            ok = (len(rb) >= 1 and names[0].startswith('display (')
                  and f'the {kind} error-diffusion dither follows' in names[0]
                  and 'cascade' in names[0]
                  and 'DISPLAY' not in stages_
                  and 'display' not in (rec.get('gpu_stages') or [])
                  # the frame is already down when the dither asks: the
                  # colour depth stage adds no second readback
                  and not any(n.startswith('colour depth') for n in names)
                  and rec.get('uploads') == 0)
            what = ('the record names the display readback (the cascade), no '
                    'DISPLAY draw, no second readback, no upload')
        else:
            ok = ('DISPLAY' in stages_
                  and not any(n.startswith('display') for n in names))
            what = 'DISPLAY still draws on the GPU, no display readback'
        check(f'display before error diffusion, {label}: {what}; the frame is '
              'bitwise the CPU chain, nothing left live',
              ok and out_g.shape == out_c.shape
              and bool(np.array_equal(out_g, out_c)) and live == [],
              str(rec))
        return rec

    # the two field presets, as the field ran them
    for key in ('ATARI_ST', 'AMIGA_OCS'):
        st = preset_settings(key)
        check(f'the {key} preset still selects FLOYD (the field row this '
              'fix is for)', str(st.dither) == 'FLOYD')
        row(f'the {key} preset (FLOYD)', st, True)
    # every sequential kernel, on the palette road and on the bit lattice
    for kind in sorted(DI.KERNELS):
        row(f'{kind} at 16 adaptive registers',
            settings(color_depth='4', palette_size=16, dither=kind), True)
    row('FLOYD at 16 bits (the separable lattice road)',
        settings(color_depth='16', dither='FLOYD'), True)
    row('ATKINSON at 1 bit', settings(color_depth='1', dither='ATKINSON'),
        True)

    # the kinds with no carry keep their records: measured BEFORE the fix
    # on this tree (scratchpad probe, 2026-10-04) and pinned whole
    rec = row('BAYER4 at 16 adaptive registers at 3 bits, cold',
              settings(color_depth='4', palette_size=16,
                       palette_bits='BITS_3', dither='BAYER4'), False)
    check('BAYER4 (ordered): the record is the pre-fix one whole -- stages '
          "DISPLAY, PALETTE; one readback named 'palette'; one upload",
          rec.get('stages') == ['DISPLAY', 'PALETTE']
          and len(rec.get('readbacks') or []) == 1
          and 'palette' in rec['readbacks'][0][0]
          and rec.get('uploads') == 1, str(rec))
    rec = row('NOISE at 16 adaptive registers',
              settings(color_depth='4', palette_size=16, dither='NOISE'),
              False)
    check('NOISE (per pixel, no carry): the record is the pre-fix one whole '
          "-- stage DISPLAY; one readback, 'colour depth (the NOISE dither "
          "runs on the CPU)'; no upload",
          rec.get('stages') == ['DISPLAY']
          and [r[0] for r in rec.get('readbacks') or []]
          == ['colour depth (the NOISE dither runs on the CPU)']
          and rec.get('uploads') == 0, str(rec))
    rec = row('the defaults (no dither)', settings(), False)
    check('the defaults: the record is the pre-fix one whole -- stages '
          'DISPLAY, QUANT; no readback; no upload',
          rec.get('stages') == ['DISPLAY', 'QUANT']
          and (rec.get('readbacks') or []) == []
          and rec.get('uploads') == 0, str(rec))
    # FLOYD selected where the colour depth stage does NOT diffuse: the
    # CRY16 encode has no dither seat and draws on the GPU, so nothing
    # sequential follows and the display stays where it was
    rec = row('CRY16 with FLOYD selected (no dither seat: nothing diffuses)',
              settings(color_depth='CRY16', dither='FLOYD'), False)
    check('CRY16 + FLOYD: the record is the pre-fix one whole -- stages '
          'DISPLAY, CRY16; no readback; no upload',
          rec.get('stages') == ['DISPLAY', 'CRY16']
          and (rec.get('readbacks') or []) == []
          and rec.get('uploads') == 0, str(rec))

    # the CPU road never takes the branch: no chain Frame, no record
    PO.LAST_CHAIN.clear()
    st = settings(color_depth='4', palette_size=16, dither='FLOYD')
    sc = demo_scene(st, with_texture=False)
    PO.process(np.asarray(R.render(sc, st)), st, **post_kw(sc, st))
    check('on the CPU device a FLOYD frame opens no chain Frame (LAST_CHAIN '
          'stays empty: the branch is the GPU road\'s alone)',
          dict(PO.LAST_CHAIN) == {})

    # bitwise-neutral: FLOYD and the defaults against the 1.89.0 engine
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (the display-before-diffusion '
          'neutrality pin runs)', RP is not None,
          '' if RP is not None else 'PIN SKIPPED: keep halcyon-1.89.0.zip beside halcyon/')
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.post')
    for label, kw in (('the defaults', {}),
                      ('FLOYD at 16 adaptive registers',
                       {'color_depth': '4', 'palette_size': 16,
                        'dither': 'FLOYD'}),
                      ('FLOYD at 16 bits, exposure 0.1',
                       {'color_depth': '16', 'dither': 'FLOYD',
                        'exposure': 0.1})):
        clear_palette_locks(RP)
        st = settings(**kw)
        sc = demo_scene(st, with_texture=False)
        now_r = np.asarray(R.render(sc, st))
        now = PO.process(now_r, st, **post_kw(sc, st))
        st2 = settings(**kw)
        sc2 = demo_scene(st2, with_texture=False)
        prev_r = np.asarray(RP.render(sc2, st2))
        prev = prev_post.process(prev_r, st2, **post_kw(sc2, st2))
        check(f'{label}: render and post.process are bitwise the 1.89.0 '
              'engine (the CPU road is untouched by the display routing)',
              now_r.shape == prev_r.shape and np.array_equal(now_r, prev_r)
              and now.shape == prev.shape and np.array_equal(now, prev))


ORDER += [test_display_before_error_diffusion]


def main():
    from . import utf8_console
    utf8_console()
    for fn in ORDER:
        print(fn.__name__)
        try:
            fn()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(fn.__name__)
    print()
    print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS) if FAILS
          else 'all post-palette (R251) checks passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
