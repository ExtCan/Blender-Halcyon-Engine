"""R251 lighting pack, LIGHT-A1: the fog twin and the period fog mechanisms.

    python -m halcyon.tests.test_r251_lighting_fog

F000  fog on the GPU road (hal_fog: the four curves, per-vertex 1/8, bands,
      height; `fog_cpu` refusal by name; the hal_fogtab sampler)
F001  GTE hyperbolic 1/z depth cue (PlayStation)
F002  Voodoo 64-entry w-table fog + the Voodoo2 fog dither (3dfx Glide)
F003  PowerVR2 128-entry log fog table (Dreamcast)
F022  DS 32-entry fog table on eye depth (Nintendo DS)
F007  z-fog on post-projection depth (Direct3D 5-7)

Wave 2 (LIGHT-A2), appended after the wave-1 tests under the marker
:
F004  GameCube horizontal fog range adjust (GX_InitFogAdjTable)
F005  per-polygon fog steps (Namco System 21)
F008  backdrop fog: the fog takes the backdrop's colour (LightWave)
F006  per-material burn-through, bias, bank 1 (Model 3 / System 22)
F009  ground fog, the atan integral (POV-Ray fog_type 2)
F010  turbulent fog (POV-Ray)

The first test pins the 1.89.0 zip (`_prev_engine('halcyon-1.89.0.zip')`)
loudly and the identity at defaults, render AND post.process, bitwise.
Every later test: the semantic A/B law on the CPU functions, the GPU twin
`d == 0.0` between the in-shader fog and the CPU's fog over the same
simulated shading, and the refusal by name where one exists.
"""

import contextlib
import importlib
import io
import sys
import traceback

import numpy as np

from ..core import fog as FOG
from ..core import post as PO
from ..core import raster
from ..core import render as R
from ..core.scene import Material
from ..gpu import shade as GSH
from ..presets.library import PRESETS
from . import featurematrix as FM
from .featurematrix import build
from .scenebuild import demo_scene
from .test_render import _prev_engine, base_settings

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


F32 = np.float32
LEGACY_ROWS = ('fog LINEAR', 'fog EXP2', 'fog TABLE16 (fixed-function)',
               'per-vertex fog', 'height fog (ground mist)')
F000_ROWS = LEGACY_ROWS + ('fog EXP', 'fog bands 4')
#: the probe swaps hal_fog's return for (depth after conversions, f, raw
#: eye depth) so the laws read the shader's own numbers
_PROBE_OLD = '    o = o + col * g;\n    return o;\n}'
_PROBE_NEW = '    return vec3(d, f, abs(dz));\n}'
_PROBE_P = '    return P;\n}'
#: the CALL the tail emits (every pass of a fogged plan carries the
#: hal_fog DEFINITION; only the fogged primary passes call it)
_CALL = 'total = hal_fog(total, P);'
#: the rows whose fog quantises (their twin absorbs the one-ULP P of
#: pack_ids' 1-x-y; the smooth curves sit at the measured P-ULP bar)
SMOOTH_ROWS = ('fog LINEAR', 'fog EXP2', 'height fog (ground mist)',
               'fog EXP')
P_ULP_BAR = 4e-6


# ----------------------------------------------------------------- the rig

def _rig(key, mut=None, cpu=True):
    """(scene, settings, cpu image, G-buffer, ShadeJob) for one FM row."""
    sc, st = build(key)
    if mut:
        mut(sc, st)
    img = R.render(sc, st) if cpu else None
    w, h = st.resolution_x, st.resolution_y
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    R._build_shadows(sc, st, sc.mesh)
    bvh = R._cached_bvh(sc, sc.mesh) if st.raytrace else None
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), bvh, view, eye, w, h)
    return sc, st, img, g, job


def _plan(job, g, on=True):
    GSH.FOG_ON_GPU = bool(on)
    GSH._PLAN_CACHE.clear()
    return GSH.plan_frame(job, g)


def _sim(job, g, on=True):
    """(image, hit, passes, atlases, why): the whole simulated frame with
    the module switch `on` (fog in the pass) or off (the CPU readback fog
    inside simulate's _fog_readback)."""
    passes, why, atl = _plan(job, g, on)
    if passes is None:
        return None, None, None, None, why
    img, hit = GSH.simulate(job, g, passes, atl)
    if img is None:
        return None, None, passes, atl, hit
    return img, hit, passes, atl, None


def _probe(job, g, passes, atl, tail=_PROBE_NEW):
    """The probe image: hal_fog's (d, f, raw depth) -- or P with
    tail=_PROBE_P -- at every pixel of a pass that calls hal_fog."""
    pp = []
    for mid, name, src, binds in passes:
        if _CALL in src:
            assert src.count(_PROBE_OLD) == 1, name
            src = src.replace(_PROBE_OLD, tail)
        pp.append((mid, name, src, binds))
    img, hit = GSH.simulate(job, g, pp, atl)
    return img, hit


def _chain_depth(P, eye, view):
    """The sequential float32 depth both roads compute (A1)."""
    dP = np.asarray(P, np.float32) - np.asarray(eye, np.float32)[None, :]
    r = np.asarray(view, np.float32)[2, :3]
    dz = dP[:, 0] * r[0]
    dz += dP[:, 1] * r[1]
    dz += dP[:, 2] * r[2]
    return np.abs(dz).astype(np.float32)


def _legacy_f(depth, st, P):
    """The 1.89.0 apply_fog's transmittance on a depth ramp: rgb = 1 and
    a black fog colour make the blend return f exactly."""
    import copy
    st2 = copy.copy(st)
    st2.fog_color = (0.0, 0.0, 0.0)
    n = np.asarray(depth).shape[0]
    out = FOG.legacy(np.ones((n, 3), np.float32), np.asarray(depth, np.float32),
                     st2, None, P)
    return out[:, 0]


def _arith_twin(key, mut=None):
    """The fog ARITHMETIC twin: hal_fog's f versus the CPU's fog on the
    shader's OWN depth and P (isolating the fog from the G-buffer's P).
    Returns (max |f_gl - f_cpu|, the probe, the rig) or (None, why, rig)."""
    sc, st, cpu, g, job = _rig(key, mut, cpu=False)
    try:
        passes, why, atl = _plan(job, g, True)
        if passes is None:
            return None, why, (sc, st, g, job)
        img, hit = _probe(job, g, passes, atl)
        imgP, _h = _probe(job, g, passes, atl, _PROBE_P)
    finally:
        GSH.FOG_ON_GPU = True
    fogged = hit & (g.tri >= 0) & np.isin(
        _mat_px(sc, g), [int(m) for m, _n, s, _b in passes if _CALL in s])
    py, px = np.nonzero(fogged)
    raw = img[py, px, 2]
    f_gl = img[py, px, 1]
    P = imgP[py, px]
    from types import SimpleNamespace
    ns = SimpleNamespace(px=px, py=py, width=job.width, height=job.height)
    if FOG.extended(st):
        f_cpu = FOG.factor(raw, st, sc, P=P, ctx=ns)
    else:
        f_cpu = _legacy_f(raw, st, P)
    d = float(np.abs(f_gl - f_cpu).max())
    return d, dict(img=img, P=P, raw=raw, f=f_gl, f_cpu=f_cpu, px=px,
                   py=py, passes=passes, atl=atl), (sc, st, g, job)


def _twin(key, mut=None):
    """max |fog in the pass - the CPU's fog over the same shading| on the
    covered pixels, plus the sim-vs-CPU distance and the plan."""
    sc, st, cpu, g, job = _rig(key, mut)
    try:
        on, hit, passes, atl, why = _sim(job, g, True)
        if on is None:
            return None, None, None, None, why
        off, hit2, _p2, _a2, why2 = _sim(job, g, False)
        if off is None:
            return None, None, None, None, why2
    finally:
        GSH.FOG_ON_GPU = True
    cov = (g.tri >= 0) & hit
    d = float(np.abs(on[cov] - off[cov]).max())
    dc = float(np.abs(on[cov] - cpu[cov][:, :3]).max())
    return d, dc, passes, atl, None


def _fog_cpu(passes):
    return {int(m): b.get('fog_cpu') for m, _n, _s, b in passes
            if (b or {}).get('fog_cpu')}


def _sk(name, kind, default, link=None, ident=None):
    return {'name': name, 'identifier': ident or name, 'type': kind,
            'default': default, 'link': link}


def _bi_graph(props, color=(0.55, 0.55, 0.6, 1.0)):
    """A Blender Internal material node graph (test_render's own shape)."""
    p = {'diff_shader': 'LAMBERT', 'spec_shader': 'COOKTORR',
         'shadeless': False}
    p.update(props)
    ins = [_sk('Color', 'RGBA', list(color), ident='Diffuse Color'),
           _sk('Intensity', 'VALUE', 0.8, ident='Diffuse Level'),
           _sk('Specular Intensity', 'VALUE', 0.4, ident='Specular Level'),
           _sk('Hardness', 'VALUE', 50.0, ident='Glossiness'),
           _sk('Emit', 'VALUE', 0.0)]
    return {'output': 'out', 'nodes': {
        'bi': {'id': 'bi', 'bl_idname': 'HALCYON_BIMaterialNode',
               'props': p, 'inputs': ins,
               'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bi', 0])],
                'outputs': []}}}


def _mat_px(sc, g):
    cov = g.tri >= 0
    m = np.full(g.tri.shape, -1, np.int64)
    m[cov] = sc.mesh.mat_index[g.tri[cov]]
    return m


# ------------------------------------------------------------ 0: identity

def test_00_identity_prev_engine():
    """The 1.89.0 zip beside the package, loudly; then every frame with no
    new dial is bitwise the previous release's -- render and post -- at
    the defaults and on the seven F000 rows (A1's sequential depth is the
    only CPU-road change and it is bitwise the matmul)."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (halcyon-1.89.0.zip), so '
          'the identity pins below run against the shipped engine',
          RP is not None)
    st = base_settings(96, 72)
    check('the new fog fields default to the 1.89.0 behaviour (fog_table '
          'NONE, fog_depth W, fog_dither False) and read as not extended',
          str(st.fog_table) == 'NONE' and str(st.fog_depth) == 'W'
          and st.fog_dither is False and not FOG.extended(st))
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0]
                                        + '.post')

    def mk():
        s = base_settings(96, 72)
        return demo_scene(s, with_texture=False), s
    sc, st = mk()
    now = np.asarray(R.render(sc, st))
    sc2, st2 = mk()
    prev = np.asarray(RP.render(sc2, st2))
    check('the demo frame at the defaults renders bitwise the 1.89.0 engine',
          now.shape == prev.shape and bool(np.array_equal(now, prev)),
          f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    pn = PO.process(now, st)
    pp = prev_post.process(prev, st2)
    check('...and post.process of that frame is bitwise the 1.89.0 post',
          pn.shape == pp.shape and bool(np.array_equal(pn, pp)),
          f'max {float(np.abs(pn - pp).max()) if pn.shape == pp.shape else "shape"}')
    for key in F000_ROWS:
        sc, st = build(key)
        now = np.asarray(R.render(sc, st))
        sc2, st2 = build(key)
        prev = np.asarray(RP.render(sc2, st2))
        check(f'{key}: a 1.89.0 field set is not "extended" and renders '
              'bitwise the 1.89.0 engine on the CPU road',
              not FOG.extended(st) and now.shape == prev.shape
              and bool(np.array_equal(now, prev)),
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
    # fog_dither alone is inert (only the Voodoo table reads it)
    sc, st = build('fog LINEAR')
    st.fog_dither = True
    now = np.asarray(R.render(sc, st))
    sc2, st2 = build('fog LINEAR')
    prev = np.asarray(RP.render(sc2, st2))
    check('fog_dither without the Voodoo table is inert: bitwise the 1.89.0 '
          'engine', not FOG.extended(st) and bool(np.array_equal(now, prev)))


# ---------------------------------------------------------------- F000

def test_f000_fog_depth_is_the_cpus():
    """hal_fog's depth is the sequential float32 chain on the fragment's P
    (bitwise, a construction); the fragment's P is the CPU's P up to the
    one ULP of pack_ids' recomputed third barycentric (1 - x - y versus
    the rasteriser's own z: measured, named, the raster pack's line); and
    the fog arithmetic on that depth is the CPU's bit for bit."""
    d, probe, rig = _arith_twin('fog LINEAR')
    sc, st, g, job = rig
    check('the fog LINEAR row plans with the fog in every primary pass',
          d is not None and all(_CALL in s for _m, _n, s, _b
                                in (probe or {}).get('passes', ())),
          str(probe if d is None else ''))
    if d is None:
        return
    chain = _chain_depth(probe['P'], job.eye, job.view)
    n_bad = int((chain != probe['raw']).sum())
    check('hal_fog\'s depth is the sequential chain on its P bitwise (three '
          'products, two sums, abs)', n_bad == 0,
          f'{n_bad} of {chain.size} differ')
    py, px = probe['py'], probe['px']
    ctx = job.context(g.tri[py, px], g.bary[py, px], px, py)
    P_cpu = np.asarray(ctx.P, np.float32)
    dP = float(np.abs(probe['P'] - P_cpu).max())
    n_p = int((probe['P'] != P_cpu).any(axis=1).sum())
    z2 = (1.0 - g.bary[py, px, 0] - g.bary[py, px, 1]).astype(np.float32)
    n_z = int((z2 != g.bary[py, px, 2]).sum())
    check('the fragment\'s P is the CPU\'s P within one ULP -- the residue '
          'is pack_ids\' third barycentric 1-x-y versus the rasteriser\'s '
          'own z (counted; the raster pack\'s line to close)',
          dP <= 2e-6, f'{n_p} of {py.size} px differ, max {dP}; '
          f'bary.z != 1-x-y at {n_z} px')
    dd = float(np.abs(probe['raw'] - ctx.depth).max())
    check('...so the shader\'s depth is ctx.depth within one ULP of P '
          '(bitwise where the barycentrics agree)', dd <= P_ULP_BAR,
          f'max {dd}')
    check('the fog arithmetic on the shader\'s own depth is the CPU\'s '
          'apply_fog bit for bit (LINEAR: one subtraction, one division)',
          d == 0.0, str(d))
    check('...and LINEAR leaves d untouched before the curve (the probe\'s '
          'd equals the raw depth)',
          bool(np.array_equal(probe['img'][py, px, 0], probe['raw'])))


def test_f000_gpu_twin():
    """The seven fog rows. Arithmetic: hal_fog's f on the shader's own
    depth is the CPU's fog bit for bit (d == 0.0, every row). Frame: in-
    shader fog versus the CPU's fog over the same simulated shading is
    bitwise on the quantised rows and within the P-ULP bar (4e-6; the
    G-buffer's 1-x-y, test_f000_fog_depth_is_the_cpus) on the smooth
    curves; and the frame stays within the deferred road's 1e-3 bar."""
    for key in F000_ROWS:
        da, probe, _rig_ = _arith_twin(key)
        check(f'{key}: hal_fog\'s arithmetic is the CPU fog bit for bit on '
              'the shader\'s own depth (d == 0.0)',
              da is not None and da == 0.0, str(probe if da is None else da))
        d, dc, passes, _atl, why = _twin(key)
        bar = P_ULP_BAR if key in SMOOTH_ROWS else 0.0
        check(f'{key}: the frame with fog in the pass equals the CPU fog '
              'over the same simulated shading '
              + ('bitwise' if bar == 0.0 else f'within {bar:g} (P-ULP)'),
              d is not None and d <= bar, str(why if d is None else d))
        check(f'{key}: the deferred frame with fog in the pass matches the '
              'CPU frame within 1e-3', dc is not None and dc < 1e-3,
              str(why if dc is None else dc))
        if passes is not None:
            check(f'{key}: no pass fogs on the CPU (no fog_cpu)',
                  not _fog_cpu(passes), str(_fog_cpu(passes)))


def test_f000_refusal_by_name():
    """(i) the module switch off: every primary pass carries fog_cpu and the
    reason is printed once per plan; (ii) a traced reflection: the
    reflective material's pass carries fog_cpu ('readback'), the others do
    not, and the frame still matches the CPU; (iii) a CPU-evaluated
    environment term (the Bryce world) does the same for the env road."""
    sc, st, cpu, g, job = _rig('fog LINEAR')
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passes, why, atl = _plan(job, g, False)
        fc = _fog_cpu(passes or [])
        prim = [int(m) for m, _n, _s, b in (passes or [])
                if not (b or {}).get('vlight')]
        check('with FOG_ON_GPU off every primary pass carries fog_cpu naming '
              'the module switch', passes is not None and prim
              and all(m in fc and 'module switch' in fc[m] for m in prim),
              str(fc))
        check('...and the reason is printed by name ([Halcyon GPU] fog on '
              'the CPU ...: the module switch is off)',
              '[Halcyon GPU] fog on the CPU' in buf.getvalue()
              and 'module switch' in buf.getvalue())
        check('...and no hal_fog call is emitted while it is off',
              passes is not None and not any(_CALL in s
                                             for _m, _n, s, _b in passes))
        # the switch is a gate the plan reads: flipping it re-plans
        sig_off = GSH._plan_sig(job, ('k',))
        GSH.FOG_ON_GPU = True
        sig_on = GSH._plan_sig(job, ('k',))
        check('the module switch is in the plan signature (a gate the plan '
              'reads must be, B2)', sig_off != sig_on)
        img_off, hit_off = GSH.simulate(job, g, passes, atl)
        cov = (g.tri >= 0) & hit_off
        d_off = float(np.abs(img_off[cov] - cpu[cov][:, :3]).max())
        check('the refused road (CPU fog over the readback) still matches '
              'the CPU frame within 1e-3', d_off < 1e-3, str(d_off))
    finally:
        GSH.FOG_ON_GPU = True

    # (ii) traced reflection under fog (TR:17985's rayfog settings)
    def rayfog(sc, st):
        st.fog = True
        st.fog_mode = 'EXP'
        st.fog_density = 0.08
    sc, st, cpu, g, job = _rig('traced reflection', rayfog)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passes, why, atl = _plan(job, g, True)
        check('the traced reflection row plans under fog', passes is not None,
              str(why))
        if passes is not None:
            refl = set(int(x) for x in (atl.get('__reflect') or {})
                       .get('reflective', ()))
            fc = _fog_cpu(passes)
            check('the reflective material\'s pass carries fog_cpu naming '
                  'the readback; the non-reflective passes carry none',
                  refl and all(m in fc and 'readback' in fc[m] for m in refl)
                  and not any(m in fc for m, _n, _s, _b in passes
                              if int(m) not in refl), f'{refl} {fc}')
            check('...its source has no hal_fog call while the others do',
                  all((_CALL in s) == (int(m) not in refl)
                      for m, _n, s, _b in passes))
            check('...and the reason is printed by name',
                  'fog on the CPU' in buf.getvalue()
                  and 'readback' in buf.getvalue())
            img, hit = GSH.simulate(job, g, passes, atl)
            cov = (g.tri >= 0) & hit
            dr = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
            check('ray hits still fog inside the recursion at their own '
                  'depth: the frame matches the CPU within 1e-3',
                  dr < 1e-3, str(dr))
    finally:
        GSH.FOG_ON_GPU = True

    # (iii) the CPU-env road: FM's 'env reflection (no rays)' on the demo
    # world keeps its baked GLSL env term (no fog_cpu); on the Bryce world
    # the env term is a CPU composite after the readback -> fog_cpu
    def envfog(sc, st):
        st.fog = True
        st.fog_mode = 'EXP'
        st.fog_density = 0.08
    sc, st, cpu, g, job = _rig('env reflection (no rays)', envfog)
    try:
        passes, why, atl = _plan(job, g, True)
        check('env reflection (no rays) on the demo world plans under fog '
              'with the env term in-shader: no pass carries fog_cpu',
              passes is not None and not _fog_cpu(passes),
              str(why if passes is None else _fog_cpu(passes)))
        sc.world.mode = 'BRYCE'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passes, why, atl = _plan(job, g, True)
        if passes is not None:
            prim = set(int(x) for x in ((atl.get('__env') or {})
                                        .get('primary') or {}))
            fc = _fog_cpu(passes)
            check('the same row on the Bryce world: the mirror material\'s '
                  'env term is a CPU composite, so its pass carries fog_cpu '
                  'naming the readback and the others do not',
                  prim and all(m in fc and 'readback' in fc[m] for m in prim)
                  and not any(int(m) in fc for m, _n, _s, _b in passes
                              if int(m) not in prim), f'{prim} {fc}')
        else:
            check('the same row on the Bryce world plans', False, str(why))
    finally:
        GSH.FOG_ON_GPU = True
    # the traced case on the Bryce world
    def envfog_mirror(sc, st):
        envfog(sc, st)
        sc.materials[2].reflect_level = 0.5     # FM's mirror scene's dial
    sc, st, cpu, g, job = _rig('rich world (Bryce) reflection',
                               envfog_mirror)
    try:
        passes, why, atl = _plan(job, g, True)
        if passes is not None:
            refl = set(int(x) for x in (atl.get('__reflect') or {})
                       .get('reflective', ()))
            fc = _fog_cpu(passes)
            check('rich world (Bryce) reflection under fog: the traced '
                  'materials fog on the CPU by name',
                  refl and all(m in fc for m in refl), f'{refl} {fc}')
        else:
            check('rich world (Bryce) reflection under fog plans', False,
                  str(why))
    finally:
        GSH.FOG_ON_GPU = True


def test_f000_vertex_rate_still_in_the_corners():
    """A Gouraud frame under fog: the corners carry the fog (shade_batch's
    LIGHT rate), so no pass carries fog_cpu, no vlight source has the fog
    block, and the frame is bit-level against the CPU."""
    def gouraud(sc, st):
        st.shading_rate = 'VERTEX'
    sc, st, cpu, g, job = _rig('fog LINEAR', gouraud)
    try:
        passes, why, atl = _plan(job, g, True)
        check('a Gouraud fog frame plans', passes is not None, str(why))
        if passes is None:
            return
        check('no vertex-rate pass carries fog_cpu and none emits hal_fog '
              '(the corners already carry the fog)',
              not _fog_cpu(passes)
              and all((b or {}).get('vlight') and _CALL not in s
                      for _m, _n, s, b in passes))
        img, hit = GSH.simulate(job, g, passes, atl)
        cov = (g.tri >= 0) & hit
        d = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('vertex-rate fog stays bit-level against the CPU (d < 1e-6)',
              d < 1e-6, str(d))
    finally:
        GSH.FOG_ON_GPU = True


def test_f000_use_mist_lifted():
    """A BI material with Use Mist off under fog now plans (the 1.89.0
    'opts out of mist' refusal is gone): its pixels are the unfogged CPU
    pixels within the deferred bar while its neighbour is fogged; and every
    pass's sampler list grew by exactly one (hal_fogtab) against the same
    scene unfogged."""
    def make(fog):
        st = base_settings(96, 72)
        st.fog = bool(fog)
        st.fog_mode = 'LINEAR'
        st.fog_start, st.fog_end = 2.0, 12.0
        st.ambient_occlusion = False
        st.shadows = False
        st.transparency = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.materials[0] = Material(name='FloorBI', index=0,
                                   graph=_bi_graph({'use_mist': False}))
        R.collect_exclusive_lights(sc)
        return sc, st

    def rig(fog):
        sc, st = make(fog)
        cpu = R.render(sc, st)
        w, h = st.resolution_x, st.resolution_y
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                         depth_bits=st.depth_precision)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                         eye, w, h)
        return sc, st, cpu, g, job
    try:
        sc0, st0, cpu0, g0, job0 = rig(False)
        p0, why0, a0 = _plan(job0, g0, True)
        sc1, st1, cpu1, g1, job1 = rig(True)
        p1, why1, a1 = _plan(job1, g1, True)
        check('a BI material with Use Mist off plans under fog (no "opts '
              'out of mist" refusal)', p0 is not None and p1 is not None,
              f'{why0} / {why1}')
        if p0 is None or p1 is None:
            return
        s0 = {int(m): list(b.get('samplers') or []) for m, _n, _s, b in p0}
        s1 = {int(m): list(b.get('samplers') or []) for m, _n, _s, b in p1}
        check('every pass\'s samplers grew by exactly one, hal_fogtab, '
              'against the unfogged plan (A5)',
              set(s0) == set(s1) and all(
                  len(s1[m]) == len(s0[m]) + 1 and 'hal_fogtab' in s1[m]
                  and 'hal_fogtab' not in s0[m] for m in s0),
              f'{ {m: (len(s0[m]), len(s1[m])) for m in s0} }')
        check('the Use-Mist-off pass emits no hal_fog call; the others do',
              all((_CALL in s) == (int(m) != 0) for m, _n, s, _b in p1))
        img, hit = GSH.simulate(job1, g1, p1, a1)
        m = _mat_px(sc1, g1)
        cov = (g1.tri >= 0) & hit
        floor = cov & (m == 0)
        other = cov & (m != 0)
        d_floor = float(np.abs(img[floor] - cpu0[floor][:, :3]).max())
        d_all = float(np.abs(img[cov] - cpu1[cov][:, :3]).max())
        moved = float(np.abs(cpu1[other][:, :3] - cpu0[other][:, :3]).max())
        check('its pixels equal the CPU\'s UNFOGGED pixels within the '
              'deferred bar (6e-3) while the frame matches the fogged CPU '
              'frame and the neighbour material moved',
              d_floor < 6e-3 and d_all < 6e-3 and moved > 1e-2,
              f'floor {d_floor} all {d_all} moved {moved}')
    finally:
        GSH.FOG_ON_GPU = True


def test_f000_readback_is_structural():
    """_fog_readback returns its argument by identity and leaves it
    untouched when no pass carries fog_cpu; under the traced row it edits
    exactly the fog_cpu material's pixels (no wall-clock assertion, A6)."""
    sc, st, cpu, g, job = _rig('fog LINEAR')
    try:
        passes, why, atl = _plan(job, g, True)
        if passes is None:
            check('fog LINEAR plans', False, str(why))
            return
        img, hit = GSH.simulate(job, g, passes, atl)
        before = img.copy()
        out2 = GSH._fog_readback(job, g, passes, img, hit)
        check('with no fog_cpu pass the readback fog is structural: the '
              'same array back, no pixel touched',
              out2 is img and bool(np.array_equal(out2, before)))
    finally:
        GSH.FOG_ON_GPU = True

    def rayfog(sc, st):
        st.fog = True
        st.fog_mode = 'EXP'
        st.fog_density = 0.08
    sc, st, cpu, g, job = _rig('traced reflection', rayfog)
    try:
        passes, why, atl = _plan(job, g, True)
        if passes is None:
            check('traced reflection plans', False, str(why))
            return
        fc = _fog_cpu(passes)
        img = np.zeros((g.tri.shape[0], g.tri.shape[1], 3), np.float32)
        img[...] = 0.5
        hit = g.tri >= 0
        before = img.copy()
        out2 = GSH._fog_readback(job, g, passes, img, hit)
        m = _mat_px(sc, g)
        edited = np.any(out2 != before, axis=2)
        check('under the traced row the readback fog edits exactly the '
              'fog_cpu material\'s pixels and no other',
              fc and bool(edited[hit & np.isin(m, list(fc))].all())
              and not bool(edited[hit & ~np.isin(m, list(fc))].any()),
              str(fc))
    finally:
        GSH.FOG_ON_GPU = True


def test_f000_fake_device():
    """The driver road through the fake device: the whole GPU frame under
    fog matches the CPU within the resident road's 6e-6, alpha bitwise;
    hal_fogtab is bound in every pass (the device refuses a declared-but-
    unbound sampler), uploaded once per plan and reused on the second
    frame; the frame stays resident (nothing on the CPU edits it)."""
    from . import fakedevice
    from ..gpu import frame as FR

    def settings(device):
        sc, st = build('fog LINEAR')
        st.shadows = False
        st.transparency = 'NONE'
        st.render_device = device
        return sc, st
    sc_c, st_c = settings('CPU')
    cpu = R.render(sc_c, st_c)
    sc_g, st_g = settings('GPU')
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        gpu = R.render(sc_g, st_g)
        live = [t for t in dev.targets if not t.freed]
        LT = dict(GSH.LAST_TIMINGS)
        LF = dict(FR.LAST)
        keys1 = [k for k in dev.cache if isinstance(k, tuple) and k
                 and k[0] == 'fogtab']
        gpu2 = R.render(sc_g, st_g)
        keys2 = [k for k in dev.cache if isinstance(k, tuple) and k
                 and k[0] == 'fogtab']
    d = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check('the whole GPU road headless under fog: within 6e-6 of the CPU '
          'frame, alpha bitwise, nothing left live',
          d <= 6e-6 and bool(np.array_equal(gpu[..., 3], cpu[..., 3]))
          and live == [], f'max {d}; live {len(live)}')
    check('hal_fogtab was uploaded once (upload_cached by its bytes) and '
          'the second frame reused it', len(keys1) == 1 and len(keys2) == 1
          and bool(np.array_equal(gpu2, gpu)), f'{len(keys1)} {len(keys2)}')
    check('with fog in the pass the frame stays resident on the device '
          '(the fog no longer edits the readback)',
          LT.get('resident') is True and not LF.get('left_gpu'),
          f"resident {LT.get('resident')} {LF}")


# ---------------------------------------------------------------- F001

def _is_int(x, unit, tol=1e-3):
    v = np.asarray(x, np.float64) * unit
    return bool(np.abs(v - np.rint(v)).max() < tol)


def _second_diff_le(vals, unit, lim=1):
    v = np.rint(np.asarray(vals, np.float64) * unit)
    return bool(abs(v[0] - 2.0 * v[1] + v[2]) <= lim)


def test_f001_gte_hyperbolic_depth_cue():
    """The GTE's cue: a factor affine in 1/z, 0 at Fog Start, 1 at Fog End,
    floored to 4.12 (IR0). Laws: the integer floor, monotone transmittance,
    the hyperbola's mid-range density over LINEAR, Per-Vertex Fog's 1/8
    rounding skipped (rule vi); twin d == 0.0 on the per-pixel row, the
    corner bar on the preset's VERTEX row; the PSX presets carry it."""
    st = base_settings(96, 72)
    st.fog, st.fog_mode, st.fog_start, st.fog_end = True, 'GTE_1Z', 3.0, 12.0
    check('GTE_1Z is an extended mode (its own road) and skips the 1/8 '
          'rounding', FOG.extended(st) and not FOG.vertex_quantised(st))
    A, B, A32, B32 = FOG.gte_consts(st)
    check('the GTE constants: A negative, the factor 0 at Fog Start and 1 '
          'at Fog End', A < 0 and abs(A / 3.0 + B) < 1e-12
          and abs(A / 12.0 + B - 1.0) < 1e-12, f'A {A} B {B}')
    ramp = np.linspace(0.5, 20.0, 64).astype(np.float32)
    f = FOG.gte_1z(ramp, A32, B32)
    check('(1 - f) * 4096 is an integer on a 64-point depth ramp (the 4.12 '
          'IR0 floor)', _is_int(1.0 - f, 4096.0))
    check('the transmittance is non-increasing along the ramp',
          bool(np.all(np.diff(f) <= 0)))
    check('f is 1 before Fog Start and 0 at and beyond Fog End',
          bool(np.all(f[ramp <= 3.0] == 1.0))
          and bool(np.all(f[ramp >= 12.0] == 0.0)))
    f_mid = float(FOG.gte_1z(np.asarray([7.5], np.float32), A32, B32)[0])
    check('at the midpoint depth 7.5 the GTE fog amount exceeds LINEAR\'s '
          '(0.8 vs 0.5: the hyperbola is denser mid-range)',
          (1.0 - f_mid) > 0.5 + 0.2 and abs((1.0 - f_mid) - 0.8) < 2e-3,
          f'{1.0 - f_mid:.4f}')
    key_px = 'fog hyperbolic 1/z (PS1 GTE, per pixel)'
    sc_g, st_g = build(key_px)
    img_g = R.render(sc_g, st_g)
    sc_l, st_l = build('fog LINEAR')
    img_l = R.render(sc_l, st_l)
    check('the per-pixel GTE frame differs from the LINEAR frame at the '
          'same Start/End (max > 1e-2)',
          float(np.abs(img_g - img_l).max()) > 1e-2)
    sc_v, st_v = build(key_px)
    st_v.fog_vertex = True
    img_v = R.render(sc_v, st_v)
    check('Per-Vertex Fog under GTE_1Z renders bitwise the same frame (the '
          'mode has its own 12-bit floor, rule vi)',
          bool(np.array_equal(img_v, img_g)))
    da, probe, _r = _arith_twin(key_px)
    check('GTE_1Z: hal_fog\'s arithmetic is the CPU\'s bit for bit on the '
          'shader\'s own depth (d == 0.0)', da is not None and da == 0.0,
          str(probe if da is None else da))
    if da is not None:
        check('...and (1 - f) * 4096 is an integer at every fogged pixel of '
              'the shader', _is_int(1.0 - probe['f'], 4096.0))
    d, dc, passes, _atl, why = _twin(key_px)
    check('GTE_1Z per pixel: the frame with fog in the pass equals the CPU '
          'fog over the same simulated shading bitwise (the floor absorbs '
          'the P ULP)', d is not None and d == 0.0, str(why if d is None else d))
    check('GTE_1Z per pixel: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))
    sc, st, cpu, g, job = _rig('fog hyperbolic 1/z (PS1 GTE)')
    try:
        passes, why, atl = _plan(job, g, True)
        check('the preset\'s look (GTE at the VERTEX rate) plans with the '
              'cue in the CPU-lit corners: no fog_cpu, no hal_fog call',
              passes is not None and not _fog_cpu(passes)
              and not any(_CALL in s for _m, _n, s, _b in passes), str(why))
        if passes is not None:
            img, hit = GSH.simulate(job, g, passes, atl)
            cov = (g.tri >= 0) & hit
            dv = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
            check('...and it is bit-level against the CPU (the corner bar '
                  '1e-6)', dv < 1e-6, str(dv))
    finally:
        GSH.FOG_ON_GPU = True
    import inspect
    check('the Fog End coverage warning (fog_coverage_note) reads GTE_1Z: '
          'it uses Start/End like LINEAR (A13)',
          "'GTE_1Z'" in inspect.getsource(R.fog_coverage_note))
    for k in ('PSX', 'PSX_HIRES'):
        ps = PRESETS[k]['settings']
        check(f'the {k} preset fogs with the GTE cue at its own VERTEX rate '
              '(6..24, no fog_vertex)', ps.get('fog') is True
              and ps.get('fog_mode') == 'GTE_1Z' and ps.get('fog_start') == 6.0
              and ps.get('fog_end') == 24.0 and ps.get('shading_rate') == 'VERTEX'
              and 'fog_vertex' not in ps)


# ---------------------------------------------------------------- F002

MAME_DITHER_4X4 = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]


def _voodoo_index(w):
    """(e, fi, i) for one w through the same ladders, independently."""
    import math
    w = min(max(float(w), 1.0), 65535.0)
    e = int(math.floor(math.log2(w))) if w >= 2.0 else 0
    e = min(e, 15)
    while 2.0 ** (e + 1) <= w and e < 15:
        e += 1
    p = 2.0 ** e
    m = (1.0 - np.float32(p) / np.float32(w)) * 8.0
    fi = int(np.floor(m))
    return e, fi, 4 * e + fi


def test_f002_voodoo_fog_table():
    """Glide's 64-entry w-table with MAME's shift chain and the Voodoo2
    dither: the entry positions, the 6.2 delta, the integer and monotone
    laws, continuity across knots (B4), the dither law, A/B against the
    smooth curve, twin d == 0.0 on both rows, the VOODOO / VOODOO2 presets."""
    check('the dither matrix is MAME\'s dither_matrix_4x4 (16 integers)',
          FOG.M4.tolist() == MAME_DITHER_4X4, str(FOG.M4.tolist()))
    st = base_settings(96, 72)
    st.fog, st.fog_mode, st.fog_density = True, 'EXP', 0.06
    st.fog_table = 'VOODOO64'
    check('VOODOO64 is an extended road that skips the 1/8 rounding',
          FOG.extended(st) and not FOG.vertex_quantised(st))
    T, delta = FOG.voodoo64_table(st)
    diffs = np.diff(T)
    check('64 entries of 0..255, monotone in depth; delta is Glide\'s 6.2 '
          'field: min(diff, 63) << 2', T.shape == (64,)
          and int(T.min()) >= 0 and int(T.max()) <= 255
          and bool(np.all(diffs >= 0))
          and all(int(delta[i]) == min(int(T[min(i + 1, 63)] - T[i]), 63) << 2
                  for i in range(64)))
    c = FOG.fill_curve(st)
    check('entry i sits at 2^(3+i/4)/(8-i%4) (guFogTableIndexToW): entry 0 '
          'at w = 1, entry 63 at 52428.8, each T = round(255 c(w))',
          all(int(T[i]) == int(round(255.0 * c(2.0 ** (3 + (i >> 2))
                                                / (8 - (i & 3)))))
              for i in range(64))
          and abs(2.0 ** 3 / 8 - 1.0) < 1e-12
          and abs(2.0 ** 18 / 5 - 52428.8) < 1e-9)
    ramp = np.linspace(1.0, 60.0, 256).astype(np.float32)
    f = FOG.voodoo64_lookup(ramp, T, delta)
    check('f_op * 256 is an integer on a 256-point depth ramp',
          _is_int(1.0 - f, 256.0))
    check('the fog amount is non-decreasing along the ramp (the fill is '
          'monotone, the interpolation continuous)',
          bool(np.all(np.diff(1.0 - f) >= -1e-7)))
    # continuity across the knots (B4): the last fraction step of interval i
    # lands within 2 of T[i+1]
    bad = []
    for i in range(63):
        e, fi = i >> 2, i & 3
        m = fi + 255.5 / 256.0
        w = np.float32((2.0 ** e) / (1.0 - m / 8.0))
        if float(w) > 65535.0 or _voodoo_index(w)[2] != i:
            continue
        fb = int(round(256.0 * (1.0 - float(
            FOG.voodoo64_lookup(np.asarray([w], np.float32), T, delta)[0]))))
        if abs(fb - int(T[i + 1])) > 2 and int(T[i + 1] - T[i]) <= 63:
            bad.append((i, fb, int(T[i + 1])))
    check('continuity across the knots: at the last fraction step of every '
          'interval the blend lies within 2 of the next entry (a quarter-'
          'interval delta fails this)', not bad, str(bad[:4]))
    # affine within an interval (up to the integer shifts: one unit)
    i = 20
    e, fi = i >> 2, i & 3
    ws = [np.float32((2.0 ** e) / (1.0 - (fi + (k + 0.5) / 256.0) / 8.0))
          for k in (64, 128, 192)]
    fb3 = [float(1.0 - FOG.voodoo64_lookup(np.asarray([w], np.float32), T,
                                          delta)[0]) for w in ws]
    check('within one interval the blend is affine in the fraction up to '
          'the shift chain\'s integer unit (second difference <= 1/256)',
          _second_diff_le(fb3, 256.0, 1))
    # the dither law
    px = np.arange(256, dtype=np.int64)
    py = (np.arange(256, dtype=np.int64) * 7) % 13
    fd = FOG.voodoo64_lookup(ramp, T, delta, px, py, True)
    dd = np.rint(((1.0 - fd) - (1.0 - f)) * 256.0)
    check('the Voodoo2 dither adds 0 or 1 blend unit per pixel and the two '
          'ramps\' means differ by less than 1/256',
          bool(np.all((dd == 0) | (dd == 1)))
          and abs(float(fd.mean() - f.mean())) < 1.0 / 256.0,
          f'units {sorted(set(dd.tolist()))} mean {float(fd.mean() - f.mean())}')
    fn = FOG.voodoo64_lookup(ramp, T, delta, None, None, True)
    check('without a pixel (hits, corners) the dither adds 0 (A10)',
          bool(np.array_equal(fn, f)))
    # A/B on the frame
    key, key_d = 'Voodoo 64-entry fog table', 'Voodoo2 fog dither'
    sc, st = build(key)
    img_t = R.render(sc, st)
    sc2, st2 = build(key)
    st2.fog_table = 'NONE'
    img_s = R.render(sc2, st2)
    dts = float(np.abs(img_t - img_s).max())
    check('the table frame differs from the smooth EXP frame it was filled '
          'from by the table\'s own 8-bit steps (max > 1e-3; measured 0.0087)',
          dts > 1e-3, str(dts))
    da, probe, _r = _arith_twin(key)
    da2, probe2, _r2 = _arith_twin(key_d)
    check('VOODOO64: hal_fog\'s arithmetic is the CPU\'s bit for bit on the '
          'shader\'s own depth and pixel (d == 0.0, plain and dithered)',
          da is not None and da == 0.0 and da2 is not None and da2 == 0.0,
          f'{probe if da is None else da} / {probe2 if da2 is None else da2}')
    if da is not None and da2 is not None:
        du = np.rint((probe['f'] - probe2['f']) * 256.0)
        check('...f_op * 256 is an integer at every fogged pixel and the '
              'dithered frame differs by 0 or 1 unit per pixel',
              _is_int(1.0 - probe['f'], 256.0)
              and bool(np.all((du == 0) | (du == 1))) and bool((du == 1).any()),
              str(sorted(set(du.tolist()))))
    for k in (key, key_d):
        d, dc, passes, _atl, why = _twin(k)
        check(f'{k}: the frame with fog in the pass equals the CPU fog over '
              'the same simulated shading bitwise', d is not None and d == 0.0,
              str(why if d is None else d))
        check(f'{k}: within the deferred bar and no fog_cpu pass',
              dc is not None and dc < 1e-3 and passes is not None
              and not _fog_cpu(passes), str(why if dc is None else dc))
    v = PRESETS['VOODOO']['settings']
    check('the VOODOO preset fogs through Glide\'s table (EXP 0.035 fill, '
          'no TABLE16, no Fog Start/End)', v.get('fog_table') == 'VOODOO64'
          and v.get('fog_mode') == 'EXP' and v.get('fog_density') == 0.035
          and 'fog_start' not in v and 'fog_end' not in v)
    v2 = PRESETS.get('VOODOO2')
    check('the VOODOO2 preset exists (800x600, the fog dither on, CONSOLE, '
          'labelled 1998) and renders a different frame from VOODOO',
          v2 is not None and v2['settings'].get('fog_dither') is True
          and v2['settings'].get('resolution_x') == 800
          and v2['category'] == 'CONSOLE' and '1998' in v2['label'])
    if v2 is not None:
        from ..presets.library import apply_preset
        # both presets shade at the VERTEX rate, where the fog is applied
        # at the CPU-lit corners -- no pixel, so the dither adds 0 (A10):
        # the Voodoo fogged per PIXEL after interpolation; that road is
        # the next cut and is disclosed. At the PIXEL rate the dither shows.
        out = {}
        for k in ('VOODOO', 'VOODOO2'):
            sk_ = base_settings(96, 72)
            apply_preset(sk_, k)
            sk_.resolution_x, sk_.resolution_y = 96, 72
            sk_.shading_rate = 'PIXEL'
            out[k] = R.render(demo_scene(sk_, with_texture=False), sk_)
        dv = float(np.abs(out['VOODOO'] - out['VOODOO2']).max())
        check('...VOODOO2 is not decoration: at the pixel rate its dither '
              'moves the frame by whole blend units (max > 1/256)',
              dv > 1.0 / 256.0, str(dv))
        sv = base_settings(96, 72)
        apply_preset(sv, 'VOODOO2')
        check('...at the preset\'s own VERTEX rate the dither is inert (the '
              'corners have no pixel: disclosed, the per-pixel corner-road '
              'fog is the next cut)', sv.shading_rate == 'VERTEX')


# ---------------------------------------------------------------- F003

def test_f003_pvr_fog_table():
    """The CLX2's 128-entry log table: the density register's 1.7/exp
    quantisation, entry depths per KallistiOS, the integer / monotone /
    far-entry / affine laws, the inert-entry identity (A16), A/B against
    the smooth curve, twin d == 0.0, the DREAMCAST preset."""
    st = base_settings(96, 72)
    st.fog, st.fog_mode, st.fog_density = True, 'EXP', 0.04
    st.fog_end, st.fog_table = 60.0, 'PVR128'
    check('PVR128 is an extended road that skips the 1/8 rounding',
          FOG.extended(st) and not FOG.vertex_quantised(st))
    T, den_q = FOG.pvr128_table(st)
    bits = int(np.float32(den_q).view(np.uint32))
    check('the density register keeps Fog End\'s top 16 bits (sign, '
          'exponent, 7 mantissa bits): den_q <= 60 with the low 16 bits zero',
          float(den_q) <= 60.0 and float(den_q) > 59.0 and (bits & 0xFFFF) == 0,
          f'{float(den_q)}')
    c = FOG.fill_curve(st)
    check('129 entries; entry i at den_q / (2^(i>>4) * ((i&15)+16)/16) '
          '(KallistiOS InverseW_Depth), entry 0 the farthest, monotone',
          T.shape == (129,) and bool(np.all(np.diff(T) <= 0))
          and all(int(T[i]) == int(round(255.0 * c(
              float(den_q) / (2.0 ** (i >> 4) * ((i & 15) + 16) / 16.0))))
              for i in range(129)))
    ramp = np.linspace(0.5, 80.0, 256).astype(np.float32)
    f = FOG.pvr128_lookup(ramp, T, den_q)
    a = 255.0 - np.asarray(f, np.float64) * 255.0
    check('a = 255 - 255 f is an integer 0..255 on a 256-point depth ramp',
          _is_int(a, 1.0) and float(a.min()) >= 0 and float(a.max()) <= 255)
    check('the transmittance is non-increasing with depth (the index runs '
          'far-to-near)', bool(np.all(np.diff(f) <= 1e-7)))
    far = FOG.pvr128_lookup(np.asarray([float(den_q) * 1.5, float(den_q) * 4,
                                        1e4], np.float32), T, den_q)
    afar = np.rint(255.0 - far * 255.0).astype(int)
    check('every depth beyond den_q reads the far entry through the >> 8 '
          'blend at fraction 0: a in {T[0] - 1, T[0]} (the blend\'s own '
          'floor; disclosed)', all(int(x) in (int(T[0]) - 1, int(T[0]))
                                   for x in afar), f'{afar.tolist()} T0 {int(T[0])}')
    st2 = base_settings(96, 72)
    st2.fog, st2.fog_mode, st2.fog_start, st2.fog_end = True, 'LINEAR', 50.0, 60.0
    st2.fog_table = 'PVR128'
    T2, dq2 = FOG.pvr128_table(st2)
    near = FOG.pvr128_lookup(np.linspace(1.0, 10.0, 32).astype(np.float32),
                             T2, dq2)
    check('an inert entry is inert: a = 0 gives f == 1.0 bitwise (the '
          'division by 255.0, not a multiply by 1/255; A16)',
          bool(np.all(near == np.float32(1.0))))
    e, m4 = 3, 5
    ws = [np.float32(float(den_q) / (2.0 ** e * (1.0 + (m4 + (k + 0.5) / 256.0) / 16.0)))
          for k in (64, 128, 192)]
    a3 = [255.0 - 255.0 * float(FOG.pvr128_lookup(np.asarray([w], np.float32),
                                                   T, den_q)[0]) for w in ws]
    check('within one entry the blend is affine in the 8-bit fraction up to '
          'the >> 8 unit', _second_diff_le(a3, 1.0, 1))
    key = 'PowerVR2 128-entry fog table'
    sc, st = build(key)
    img_t = R.render(sc, st)
    sc2, st2 = build(key)
    st2.fog_table = 'NONE'
    img_s = R.render(sc2, st2)
    check('the table frame differs from the smooth EXP frame (max > 1e-2)',
          float(np.abs(img_t - img_s).max()) > 1e-2)
    da, probe, _r = _arith_twin(key)
    check('PVR128: hal_fog\'s arithmetic is the CPU\'s bit for bit on the '
          'shader\'s own depth (d == 0.0)', da is not None and da == 0.0,
          str(probe if da is None else da))
    if da is not None:
        check('...and a is an integer at every fogged pixel of the shader',
              _is_int(255.0 - probe['f'] * 255.0, 1.0))
    d, dc, passes, _atl, why = _twin(key)
    check(f'{key}: the frame with fog in the pass equals the CPU fog over '
          'the same simulated shading bitwise', d is not None and d == 0.0,
          str(why if d is None else d))
    check(f'{key}: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))
    dc_ = PRESETS['DREAMCAST']['settings']
    check('the DREAMCAST preset fogs through the CLX2 table with Fog End 64 '
          'as the density register and EXP 0.02 as the fill',
          dc_.get('fog_table') == 'PVR128' and dc_.get('fog_end') == 64.0
          and dc_.get('fog_mode') == 'EXP' and dc_.get('fog_density') == 0.02)


# ---------------------------------------------------------------- F022

def test_f022_ds_fog_table():
    """The DS's 32-entry density table on eye depth: 7-bit densities at
    equal steps from Fog Start to Fog End, melonDS's 17-bit fraction and
    34-entry tail, 127 read as 128; the laws, A/B, rule (vi), twin."""
    st = base_settings(96, 72)
    st.fog, st.fog_mode, st.fog_start, st.fog_end = True, 'LINEAR', 3.0, 12.0
    st.fog_table = 'DS32'
    check('DS32 is an extended road that skips the 1/8 rounding',
          FOG.extended(st) and not FOG.vertex_quantised(st))
    T, start32, step32 = FOG.ds32_table(st)
    check('34 entries (melonDS\'s T[32] = T[33] = T[31]), entry 0 at Fog '
          'Start = 0, entry 31 at Fog End = 127 under a LINEAR fill, '
          'monotone, step = span / 31',
          T.shape == (34,) and int(T[0]) == 0 and int(T[31]) == 127
          and int(T[32]) == 127 and int(T[33]) == 127
          and bool(np.all(np.diff(T) >= 0))
          and start32 == np.float32(3.0) and step32 == np.float32(9.0 / 31.0))
    ramp = np.linspace(0.0, 20.0, 256).astype(np.float32)
    f = FOG.ds32_lookup(ramp, T, start32, step32)
    check('(1 - f) * 128 is an integer 0..128 on a 256-point depth ramp',
          _is_int(1.0 - f, 128.0) and float(f.min()) >= 0.0
          and float(f.max()) <= 1.0)
    check('the transmittance is non-increasing along the ramp',
          bool(np.all(np.diff(f) <= 1e-7)))
    check('before Fog Start the fog is entry 0 (f = 1); at and beyond Fog '
          'End + one step of float rounding it is EXACTLY the fog colour '
          '(127 read as 128: f == 0.0 bitwise)',
          bool(np.all(f[ramp <= 3.0] == 1.0))
          and bool(np.all(f[ramp >= 12.0 + 1e-3] == 0.0)))
    i = 10
    ds = [np.float32(3.0 + (i + (k + 0.5) / 131072.0 * 0 + k / 4.0) * float(step32))
          for k in (1, 2, 3)]
    d3 = [float(1.0 - FOG.ds32_lookup(np.asarray([x], np.float32), T, start32,
                                       step32)[0]) for x in ds]
    check('within one entry the density is affine in the 17-bit fraction '
          'up to the >> 17 unit', _second_diff_le(d3, 128.0, 1))
    key = 'DS 32-entry fog table'
    sc, st = build(key)
    img_t = R.render(sc, st)
    sc2, st2 = build('fog LINEAR')
    img_l = R.render(sc2, st2)
    dl = float(np.abs(img_t - img_l).max())
    check('the DS frame differs from the LINEAR frame it was filled from '
          '(the 7-bit steps: max > 1e-3)', dl > 1e-3, str(dl))
    sc3, st3 = build(key)
    st3.fog_vertex = True
    check('Per-Vertex Fog under DS32 renders bitwise the same frame (rule '
          'vi)', bool(np.array_equal(R.render(sc3, st3), img_t)))
    da, probe, _r = _arith_twin(key)
    check('DS32: hal_fog\'s arithmetic is the CPU\'s bit for bit on the '
          'shader\'s own depth (d == 0.0)', da is not None and da == 0.0,
          str(probe if da is None else da))
    if da is not None:
        check('...and (1 - f) * 128 is an integer at every fogged pixel',
              _is_int(1.0 - probe['f'], 128.0))
    d, dc, passes, _atl, why = _twin(key)
    check(f'{key}: the frame with fog in the pass equals the CPU fog over '
          'the same simulated shading bitwise', d is not None and d == 0.0,
          str(why if d is None else d))
    check(f'{key}: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))


# ---------------------------------------------------------------- F007

def test_f007_d3d_zfog():
    """Direct3D's z-buffer pixel fog: the depth source becomes the raster's
    own post-projection z (near/far read from that projection, A28), the
    curve unchanged. Identity at W; the near/far law; the A/B (a wall at
    the far stretch, near pixels bitwise untouched); the probe law; twin
    within the P-ULP bar; the orthographic form."""
    from ..core import lights as LI
    sc, st = build('fog LINEAR')
    check('fog_depth W (the default) is not an extended road',
          not FOG.extended(st) and FOG.depth_of(st) == 'W')
    key = 'Direct3D z-fog'
    sc, st = build(key)
    n32, k32, ortho = FOG.projection_consts(sc, 96, 72)
    near, far = float(sc.camera.clip_start), float(sc.camera.clip_end)
    check('the near/far come from the raster\'s projection: n32 = '
          'clip_start, k32 = far / (far - near), perspective',
          not ortho and n32 == np.float32(near)
          and k32 == np.float32(far / (far - near)), f'{n32} {k32} {ortho}')
    sc_c, st_c = build(key)
    aspect = 96 / 72
    sc_c.camera.projection = LI._persp(0.9, aspect, 0.5, 100.0)
    n_c, k_c, o_c = FOG.projection_consts(sc_c, 96, 72)
    check('a custom camera.projection whose near differs from clip_start '
          'is read as the projection\'s (A28)', n_c == np.float32(0.5)
          and k_c == np.float32(100.0 / 99.5) and not o_c, f'{n_c} {k_c}')
    ramp = np.linspace(0.2, 50.0, 256).astype(np.float32)
    zd = FOG.z_from_w(ramp, n32, k32, False)
    dd = np.maximum(ramp, np.float32(1e-6))
    q = n32 / dd
    t = np.float32(1.0) - q
    ref = k32 * t
    check('zd = k32 * (1 - n32 / d) in float32, one op per statement, '
          'bitwise; monotone in d; below 1 everywhere',
          bool(np.array_equal(zd, ref)) and bool(np.all(np.diff(zd) > 0))
          and float(zd.max()) < 1.0)
    # the A/B: thresholds sized from the frame's own z range
    sc0, st0 = build(key)
    st0.fog = False
    img0 = R.render(sc0, st0)
    w, h = 96, 72
    view, _p, vp, eye = R.camera_matrices(sc0.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc0.mesh.verts, sc0.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st0.depth_precision)
    cov = g.tri >= 0
    py, px = np.nonzero(cov)
    job0 = R.ShadeJob(sc0, st0, R.prepare_textures(sc0, st0), None, view,
                      eye, w, h)
    depth = job0.context(g.tri[py, px], g.bary[py, px], px, py).depth
    zpx = FOG.z_from_w(np.ascontiguousarray(depth), n32, k32, False)
    z_start = float(np.percentile(zpx, 40.0))
    z_end = float(zpx.max()) * 1.00001
    sc1, st1 = build(key)
    st1.fog_start, st1.fog_end = z_start, z_end
    img1 = R.render(sc1, st1)
    nearm = zpx < z_start - 1e-6
    farm = zpx > z_start + 1e-4
    check('near pixels (zd below Fog Start) are bitwise the unfogged frame; '
          'far pixels are fogged (the wall at the far stretch)',
          bool(nearm.any()) and bool(farm.any())
          and bool(np.array_equal(img1[py[nearm], px[nearm]],
                                  img0[py[nearm], px[nearm]]))
          and float(np.abs(img1[py[farm], px[farm]]
                           - img0[py[farm], px[farm]]).max()) > 1e-3,
          f'near {int(nearm.sum())} far {int(farm.sum())} '
          f'z {float(zpx.min()):.5f}..{float(zpx.max()):.5f}')
    da, probe, _r = _arith_twin(key)
    check('z-fog: hal_fog\'s arithmetic is the CPU\'s bit for bit on the '
          'shader\'s own depth (d == 0.0)', da is not None and da == 0.0,
          str(probe if da is None else da))
    if da is not None:
        zs = FOG.z_from_w(np.ascontiguousarray(probe['raw']), n32, k32, False)
        check('...the shader\'s converted depth is z_from_w of its raw depth '
              'bitwise', bool(np.array_equal(probe['img'][probe['py'],
                                                          probe['px'], 0], zs)))
    d, dc, passes, _atl, why = _twin(key)
    check(f'{key}: the frame with fog in the pass equals the CPU fog over '
          f'the same simulated shading within {P_ULP_BAR:g} (P-ULP; LINEAR '
          'is smooth)', d is not None and d <= P_ULP_BAR,
          str(why if d is None else d))
    check(f'{key}: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))
    # the orthographic form
    keyo = 'Direct3D z-fog (orthographic)'
    sco, sto = build(keyo)
    no, ko, oo = FOG.projection_consts(sco, 96, 72)
    check('under an ORTHO camera the z-buffer depth is linear: n32 = near, '
          'k32 = 1 / (far - near) (within 1e-4 of the float32 projection)',
          oo and abs(float(no) - sco.camera.clip_start) < 1e-4
          and abs(float(ko) - 1.0 / (sco.camera.clip_end - sco.camera.clip_start)) < 1e-6,
          f'{no} {ko} {oo}')
    da, probe, _r = _arith_twin(keyo)
    check('ortho z-fog: hal_fog\'s arithmetic is the CPU\'s bit for bit '
          '(d == 0.0) and the linear branch is emitted',
          da is not None and da == 0.0
          and all('float tz = d - fp2.z;' in s for _m, _n, s, _b
                  in probe['passes'] if _CALL in s),
          str(probe if da is None else da))
    d, dc, passes, _atl, why = _twin(keyo)
    check(f'{keyo}: the frame with fog in the pass equals the CPU fog over '
          f'the same simulated shading within {P_ULP_BAR:g} and the deferred '
          'bar', d is not None and d <= P_ULP_BAR and dc < 1e-3,
          str(why if d is None else (d, dc)))
    sco2, sto2 = build(keyo)
    sto2.fog = False
    check('the orthographic row is not vacuous: it fogs pixels',
          float(np.abs(R.render(sco2, sto2) - R.render(*build(keyo))).max())
          > 1e-2)




# ---- wave 2 ---- (LIGHT-A2: F004, F005, F008, F006, F009, F010)
#
# Appended after wave 1's last test; wave 1's tests above are untouched.
# Every feature below: the identity at its default against the 1.89.0
# engine, the semantic A/B law on the CPU functions, the GPU twin (d ==
# 0.0 on the shader's own depth; the frame bitwise on a quantised row or
# within the P-ULP bar on a smooth one), the fake-device road where the
# entry asks, and the refusal / named fallback where one exists.

W2_ROWS = ('GameCube fog range adjust', 'per-polygon fog (Namco System 21)',
           'per-material fog burn/bias (Model 3 / System 22)',
           'fog bank 1 (System 22)', 'fog ambient 0.5 (Model 3 fogAmbient)',
           'LightWave backdrop fog', 'POV ground fog (atan integral)',
           'POV ground fog, the sky fogged by elevation',
           'POV turbulent fog', 'POV turbulent fog, depth 1.0')
W2_FIELDS = ('fog_range_adjust', 'fog_face', 'fog_ambient', 'fog_bank1_start',
             'fog_bank1_end', 'fog_color_source', 'fog_ground_offset',
             'fog_ground_alt', 'fog_turbulence', 'fog_turb_depth')
#: the wave-2 calls: a material with a fog dial takes the five-argument
#: hal_fog (F006); every other material keeps wave 1's `_CALL`
_CALL5 = 'total = hal_fog(total, P, '


def _calls(src):
    return src.count(_CALL) + src.count(_CALL5)


def _socket_vals(sc):
    """{material index: (burn, bias, bank)} from each material's master-
    shader graph (the FM builders' socket defaults; the CPU's Surface
    and the GPU's bake read the same numbers)."""
    out = {}
    for i, m in enumerate(sc.materials):
        g = getattr(m, 'graph', None) or {}
        vals = [0.0, 0.0, 0.0]
        for node in (g.get('nodes') or {}).values():
            if node.get('bl_idname') != 'HALCYON_ShaderNode':
                continue
            for ins in node.get('inputs') or ():
                nm = ins.get('name')
                if nm == 'Fog Burn-Through':
                    vals[0] = float(ins.get('default') or 0.0)
                elif nm == 'Fog Bias':
                    vals[1] = float(ins.get('default') or 0.0)
                elif nm == 'Fog Bank':
                    vals[2] = float(ins.get('default') or 0.0)
        out[i] = tuple(vals)
    return out


def _surf_of(sc, g, py, px):
    """A per-pixel `surf` namespace carrying the three fog dials of the
    material under each pixel (F006's CPU inputs)."""
    from types import SimpleNamespace
    m = _mat_px(sc, g)[py, px]
    sv = _socket_vals(sc)
    burn = np.zeros(m.shape, np.float32)
    bias = np.zeros(m.shape, np.float32)
    bank = np.zeros(m.shape, np.float32)
    for i, (b0, b1, b2) in sv.items():
        sel = m == i
        burn[sel], bias[sel], bank[sel] = b0, b1, b2
    return SimpleNamespace(fog_burn=burn, fog_bias=bias, fog_bank=bank)


def _arith_twin_w2(key, mut=None):
    """The fog ARITHMETIC twin for a wave-2 row: hal_fog's f versus the
    CPU's `factor` on the shader's OWN depth and P, with the scene's
    eye / view / triangle ids and the per-pixel material dials handed
    to the CPU (F005 reads the corners, F006 the dials, F009 / F010 the
    eye). Returns (max |f_gl - f_cpu|, probe, rig) or (None, why, rig)."""
    sc, st, cpu, g, job = _rig(key, mut, cpu=False)
    try:
        passes, why, atl = _plan(job, g, True)
        if passes is None:
            return None, why, (sc, st, g, job)
        pp = []
        for mid, name, src, binds in passes:
            if _calls(src):
                assert src.count(_PROBE_OLD) == 1, name
                src = src.replace(_PROBE_OLD, _PROBE_NEW)
            pp.append((mid, name, src, binds))
        img, hit = GSH.simulate(job, g, pp, atl)
        imgP, _h = _probe(job, g, passes, atl, _PROBE_P)
    finally:
        GSH.FOG_ON_GPU = True
    fogged = hit & (g.tri >= 0) & np.isin(
        _mat_px(sc, g), [int(m) for m, _n, s, _b in passes if _calls(s)])
    py, px = np.nonzero(fogged)
    raw = img[py, px, 2]
    f_gl = img[py, px, 1]
    P = imgP[py, px]
    from types import SimpleNamespace
    ns = SimpleNamespace(px=px, py=py, width=job.width, height=job.height,
                         tri=g.tri[py, px], scene=sc, camera_pos=job.eye,
                         view_matrix=job.view, I=None, textures=job.textures,
                         job=job, backdrop=getattr(job, 'backdrop', None))
    surf = _surf_of(sc, g, py, px)
    f_cpu = FOG.factor(raw, st, sc, P=P, ctx=ns, surf=surf)
    d = float(np.abs(f_gl - f_cpu).max()) if py.size else 0.0
    return d, dict(img=img, P=P, raw=raw, f=f_gl, f_cpu=f_cpu, px=px, py=py,
                   passes=passes, atl=atl, d=img[py, px, 0]), (sc, st, g, job)


def _gbuf_of(sc, st):
    w, h = st.resolution_x, st.resolution_y
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    return g, view, vp, eye


def _fake_frame(key, mut=None):
    """(gpu, cpu, live, cache keys) of one row through the fake device."""
    from . import fakedevice

    def settings(device):
        sc, st = build(key)
        st.shadows = False
        st.transparency = 'NONE'
        st.render_device = device
        if mut:
            mut(sc, st)
        return sc, st
    sc_c, st_c = settings('CPU')
    cpu = R.render(sc_c, st_c)
    sc_g, st_g = settings('GPU')
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    with fakedevice.installed() as dev:
        gpu = R.render(sc_g, st_g)
        live = [t for t in dev.targets if not t.freed]
        keys1 = [k for k in dev.cache if isinstance(k, tuple) and k]
        gpu2 = R.render(sc_g, st_g)
        keys2 = [k for k in dev.cache if isinstance(k, tuple) and k]
    return gpu, cpu, live, keys1, keys2, gpu2


# ------------------------------------------------------ wave 2: identity

def test_w2_00_identity_prev_engine():
    """The wave-2 fog fields default to the 1.89.0 behaviour: with every
    one of them written explicitly at its default, a fogged frame reads
    as not extended and renders bitwise the 1.89.0 engine (render and
    post); every wave-2 row moves the frame (none is decoration)."""
    RP = _prev_engine('halcyon-1.89.0.zip')
    check('the 1.89.0 zip is beside the package (halcyon-1.89.0.zip), so '
          'the wave-2 identity pins run against the shipped engine',
          RP is not None)
    st = base_settings(96, 72)
    defaults = {k: getattr(st, k) for k in W2_FIELDS}
    check('the ten wave-2 fields exist with the spec\'s defaults and read '
          'as not extended',
          defaults == {'fog_range_adjust': False, 'fog_face': False,
                       'fog_ambient': 1.0, 'fog_bank1_start': 0.0,
                       'fog_bank1_end': 0.0, 'fog_color_source': 'FIXED',
                       'fog_ground_offset': 0.0, 'fog_ground_alt': 1.0,
                       'fog_turbulence': 0.0, 'fog_turb_depth': 0.5}
          and not FOG.extended(st), str(defaults))
    fs = FOG.structure(base_settings(96, 72))
    check('structure() is None while fog is off; on, it carries the wave-2 '
          'keys at their inert values', fs is None)
    st.fog = True
    fs = FOG.structure(st)
    check('...range_adjust False, face False, source FIXED, bank1 False, '
          'turb False', fs is not None and fs.get('range_adjust') is False
          and fs.get('face') is False and fs.get('source') == 'FIXED'
          and fs.get('bank1') is False and fs.get('turb') is False, str(fs))
    if RP is None:
        return
    prev_post = importlib.import_module(RP.__name__.rsplit('.', 1)[0]
                                        + '.post')
    from .r251_common import clear_palette_locks

    def mk(key):
        sc, s = build(key)
        for k, v in defaults.items():
            setattr(s, k, v)
        return sc, s
    for key in ('fog LINEAR', 'fog EXP', 'height fog (ground mist)'):
        sc, s = mk(key)
        now = np.asarray(R.render(sc, s))
        sc2, s2 = build(key)
        prev = np.asarray(RP.render(sc2, s2))
        check(f'{key} with every wave-2 field at its default renders '
              'bitwise the 1.89.0 engine', not FOG.extended(s)
              and now.shape == prev.shape and bool(np.array_equal(now, prev)),
              f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
        if key == 'fog LINEAR':
            clear_palette_locks(RP)
            pn = PO.process(now, s)
            pp = prev_post.process(prev, s2)
            check('...and post.process of that frame is bitwise the 1.89.0 '
                  'post', pn.shape == pp.shape and bool(np.array_equal(pn, pp)))
    # the gradient world (F008 / F009's scene) at the defaults too
    sc, s = mk('LightWave backdrop fog')
    s.fog_color_source = 'FIXED'
    now = np.asarray(R.render(sc, s))
    sc2, s2 = build('LightWave backdrop fog')
    s2.fog_color_source = 'FIXED'
    prev = np.asarray(RP.render(sc2, s2))
    check('the gradient world under plain LINEAR fog renders bitwise the '
          '1.89.0 engine (the sky tail and the backdrop road are inert at '
          'FIXED / not GROUND)', bool(np.array_equal(now, prev)))
    for key in W2_ROWS:
        sc, s = build(key)
        a = R.render(sc, s)
        sc2, s2 = build(key)
        s2.fog = False
        b = R.render(sc2, s2)
        check(f'{key}: the row fogs pixels (not decoration)',
              float(np.abs(a - b).max()) > 1e-2)


# --------------------------------------------------------------- F004

def test_f004_gc_fog_range_adjust():
    """GX_InitFogAdjTable: ten 4.8 secants at 32-output-pixel steps from
    the viewport centre scale the planar fog depth per column. Laws
    through the shader's own depth probe: symmetric about the centre,
    non-decreasing outward, the edge equals the CPU fill to 0 ulp, the
    two centre columns within half a step of 1; twin d == 0.0 (the
    TABLE16 row is quantised: the frame bitwise); ORTHO is a named inert
    fallback on both roads; the GAMECUBE preset switches it on."""
    key = 'GameCube fog range adjust'
    sc, st = build(key)
    check('fog_range_adjust is an extended road; False is not',
          FOG.extended(st) and FOG.range_adjust_on(st, sc)
          and not FOG.extended(build('fog TABLE16 (fixed-function)')[1]))
    K = FOG.gc_adjust_knots(sc, st, 96, 72)
    r = np.rint(np.asarray(K, np.float64) * 256.0)
    check('the ten knots are 4.8 fixed-point secants: K = r / 256 with r an '
          'integer, K[0] >= 1, non-decreasing outward',
          K.shape == (10,) and bool(np.all(np.abs(K * 256.0 - r) < 1e-9))
          and float(K[0]) >= 1.0 and bool(np.all(np.diff(K) >= 0)),
          str(K.tolist()))
    cx32, k32 = FOG.gc_adjust_consts(st, 96)
    check('cx32 is the internal frame centre and k32 = 1/(32 ss) with ss 1',
          cx32 == np.float32(48.0) and k32 == np.float32(1.0 / 32.0))
    # A/B on the frame
    img_on = R.render(sc, st)
    sc0, st0 = build(key)
    st0.fog_range_adjust = False
    img_off = R.render(sc0, st0)
    dcol = np.abs(img_on - img_off).max(axis=(0, 2))
    check('the adjusted frame differs from the plain one, and at both '
          'frame edges (the columns the secant scales most)',
          float(dcol[:8].max()) > 1e-2 and float(dcol[-8:].max()) > 1e-2,
          f'edges {float(dcol[:8].max()):.4f} / {float(dcol[-8:].max()):.4f}')
    # the laws through the depth probe (d after the adjust vs the raw depth)
    da, probe, rig = _arith_twin_w2(key)
    check('range adjust: hal_fog\'s f is the CPU\'s bit for bit on the '
          'shader\'s own depth (d == 0.0)', da is not None and da == 0.0,
          str(probe if da is None else da))
    if da is not None:
        px, py = probe['px'], probe['py']
        ones = np.ones(px.size, np.float32)
        k_cpu = FOG.range_adjust(ones, px, st, sc, 96, 72)
        k_mir = FOG.range_adjust(ones, 95 - px, st, sc, 96, 72)
        check('k(px) == k(w-1-px) bitwise at every fogged pixel (symmetric '
              'about the centre)', bool(np.array_equal(k_cpu, k_mir)))
        check('the shader\'s adjusted depth is raw * k with the CPU fill\'s '
              'k, bitwise at every fogged pixel (the edge included)',
              bool(np.array_equal(probe['d'],
                                  (probe['raw'] * k_cpu).astype(np.float32))),
              f'max {float(np.abs(probe["d"] - probe["raw"] * k_cpu).max())}')
        kk = k_cpu
        mono = []
        rows = {}
        for x, y, k in zip(px.tolist(), py.tolist(), kk.tolist()):
            rows.setdefault(y, []).append((abs(x + 0.5 - 48.0), k))
        for y, lst in rows.items():
            lst.sort()
            for (d0, k0), (d1, k1) in zip(lst, lst[1:]):
                if d1 > d0 and k1 < k0:
                    mono.append((y, d0, k0, d1, k1))
        check('k is non-decreasing away from the centre along every row',
              not mono, str(mono[:3]))
        centre = kk[np.isin(px, [47, 48])]
        half_step = (float(K[0]) - 1.0) * 0.5 / 32.0
        check('the two centre columns sit half a pixel from the axis: their '
              'k is within (K[0] - 1) / 64 of 1 (not 1 to the ulp: the '
              'column\'s centre is at +0.5, disclosed)',
              centre.size > 0 and float(np.abs(centre - 1.0).max())
              <= half_step + 1e-7, f'{float(np.abs(centre - 1.0).max())}')
    d, dc, passes, _atl, why = _twin(key)
    check(f'{key}: the frame with fog in the pass equals the CPU fog over '
          'the same simulated shading bitwise (TABLE16 quantises)',
          d is not None and d == 0.0, str(why if d is None else d))
    check(f'{key}: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))
    # 2x supersampled: the table is the OUTPUT frame's
    sc2, st2 = build(key)
    st2.aa_mode, st2.aa_samples = 'SUPERSAMPLE', 4
    K2 = FOG.gc_adjust_knots(sc2, st2, 192, 144)
    cx2, k2_ = FOG.gc_adjust_consts(st2, 192)
    check('at 2x supersampling the knots are the same OUTPUT-frame secants '
          'and k32 halves (the 32-pixel step is an output pixel, A19)',
          bool(np.array_equal(K2, K)) and cx2 == np.float32(96.0)
          and k2_ == np.float32(1.0 / 64.0))
    img2 = R.render(sc2, st2)
    check('...and the supersampled frame renders and moves the picture',
          img2.shape == (72, 96, 4)
          and float(np.abs(img2 - img_off).max()) > 1e-3)
    # ORTHO: inert by name on both roads
    def ortho(sc, st):
        sc.camera.type = 'ORTHO'
    sco, sto = build(key)
    ortho(sco, sto)
    sco0, sto0 = build(key)
    ortho(sco0, sto0)
    sto0.fog_range_adjust = False
    check('under an ORTHO camera the adjust is inert: bitwise the frame '
          'without it (k = 1, no branch)',
          not FOG.range_adjust_on(sto, sco)
          and bool(np.array_equal(R.render(sco, sto), R.render(sco0, sto0))))
    notes = FOG.plan_notes(sto, True)
    check('...and the plan\'s notes name the fallback once (orthographic)',
          len(notes) == 1 and 'orthographic' in notes[0], str(notes))
    sc, st, cpu, g, job = _rig(key, ortho)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            passes, why, atl = _plan(job, g, True)
        check('the GPU plan under ORTHO prints the note once and emits no '
              'range-adjust branch', passes is not None
              and buf.getvalue().count('orthographic') == 1
              and not any('float kb = texelFetch' in s
                          for _m, _n, s, _b in passes), str(why))
        if passes is not None:
            img, hit = GSH.simulate(job, g, passes, atl)
            cov = (g.tri >= 0) & hit
            dv = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
            check('...and the ORTHO frame matches the CPU within the '
                  'deferred bar', dv < 1e-3, str(dv))
    finally:
        GSH.FOG_ON_GPU = True
    gc = PRESETS['GAMECUBE']['settings']
    check('the GAMECUBE preset keeps its frame: GXInit leaves the adjust off, '
          'and at the preset\'s VERTEX rate a corner has no pixel column (the '
          'adjust would be inert; the design\'s line was dropped, disclosed)',
          'fog_range_adjust' not in gc and gc.get('shading_rate') == 'VERTEX')
    # the corner road: no pixel, k = 1 (A10) -- bitwise the plain frame
    scv, stv = build(key)
    stv.shading_rate = 'VERTEX'
    scv0, stv0 = build(key)
    stv0.shading_rate = 'VERTEX'
    stv0.fog_range_adjust = False
    check('at the VERTEX rate the adjust is inert (the CPU-lit corners have '
          'no pixel column): bitwise the plain frame',
          bool(np.array_equal(R.render(scv, stv), R.render(scv0, stv0))))


# --------------------------------------------------------------- F005

def test_f005_face_fog():
    """Namco System 21: one fog value per polygon from the mean of its
    three corners' planar depths. Laws through the shader probe: zero
    spread per triangle, at most 17 values under 16 bands, the shader's
    depth equals the CPU corner mean bitwise, Per-Vertex Fog is skipped
    (rule vi); twin d == 0.0 and the frame bitwise; the NAMCO_S21 preset."""
    key = 'per-polygon fog (Namco System 21)'
    sc, st = build(key)
    check('fog_face is an extended road that skips the 1/8 rounding',
          FOG.extended(st) and FOG.face_on(st)
          and not FOG.vertex_quantised(st))
    img_f = R.render(sc, st)
    sc0, st0 = build(key)
    st0.fog_face = False
    img_p = R.render(sc0, st0)
    check('per-polygon fog differs from per-pixel fog at the same bands',
          float(np.abs(img_f - img_p).max()) > 1e-2)
    sc1, st1 = build(key)
    st1.fog_vertex = True
    check('Per-Vertex Fog under per-polygon fog renders bitwise the same '
          'frame (the polygon\'s own step replaces the rounding, rule vi)',
          bool(np.array_equal(R.render(sc1, st1), img_f)))
    da, probe, rig = _arith_twin_w2(key)
    check('face fog: hal_fog\'s f is the CPU\'s bit for bit on the '
          'shader\'s own corners (d == 0.0)', da is not None and da == 0.0,
          str(probe if da is None else da))
    if da is not None:
        sc, st, g, job = rig
        px, py = probe['px'], probe['py']
        tri = g.tri[py, px]
        spread = []
        for t in np.unique(tri):
            fv = probe['f'][tri == t]
            if float(fv.max() - fv.min()) != 0.0:
                spread.append((int(t), float(fv.max() - fv.min())))
        check('every covered pixel of one triangle carries the same f (zero '
              'spread per gbuf.tri)', not spread, str(spread[:3]))
        check('with fog_bands 16 the probe takes at most 17 distinct values',
              len(set(probe['f'].tolist())) <= 17,
              str(len(set(probe['f'].tolist()))))
        from types import SimpleNamespace
        ns = SimpleNamespace(tri=tri, scene=sc, camera_pos=job.eye,
                             view_matrix=job.view)
        dm = np.maximum(FOG.face_depth(ns), 0.0).astype(np.float32)
        check('the shader\'s depth is the CPU corner mean (three sequential '
              'dots, one multiply by float32(1/3)) bitwise',
              bool(np.array_equal(probe['d'], dm)),
              f'max {float(np.abs(probe["d"] - dm).max())}')
    d, dc, passes, _atl, why = _twin(key)
    check(f'{key}: the frame with fog in the pass equals the CPU fog over '
          'the same simulated shading bitwise', d is not None and d == 0.0,
          str(why if d is None else d))
    check(f'{key}: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))
    # the FACE rate: one shade per polygon at its centroid, whose planar
    # depth IS the corner mean (a linear map) -- the flag is redundant
    # there and the preset carries it as the machine's statement
    scf, stf = build(key)
    stf.shading_rate = 'FACE'
    scf0, stf0 = build(key)
    stf0.shading_rate = 'FACE'
    stf0.fog_face = False
    df = float(np.abs(R.render(scf, stf) - R.render(scf0, stf0)).max())
    check('at the FACE rate per-polygon fog is redundant: the centroid\'s '
          'depth is the corner mean up to float rounding (max diff < 1e-3; '
          'the NAMCO_S21 preset\'s flag states the machine, disclosed)',
          df < 1e-3, str(df))
    ns_ = PRESETS['NAMCO_S21']['settings']
    check('the NAMCO_S21 preset fogs per polygon in 16 banks (LINEAR 6..40, '
          'FACE rate, painter\'s sort)', ns_.get('fog_face') is True
          and ns_.get('fog_bands') == 16 and ns_.get('fog_mode') == 'LINEAR'
          and ns_.get('shading_rate') == 'FACE'
          and 'lands in 1.90.0 pass 2' not in PRESETS['NAMCO_S21']['note'])


# --------------------------------------------------------------- F008

def test_f008_backdrop_fog():
    """LightWave's Use Backdrop Color: the fog fades toward the world
    colour at the same pixel, the backdrop itself untouched; points
    without a pixel take the world along their own direction. Laws on
    the CPU, the twin (LINEAR: the P-ULP bar) under a GRADIENT and a
    BRYCE world, the vertex-rate corner bar, the fake-device upload."""
    from ..core import sky as SKY
    key = 'LightWave backdrop fog'
    sc, st = build(key)
    check('BACKDROP is an extended road; FIXED is not',
          FOG.extended(st) and FOG.source_of(st) == 'BACKDROP'
          and not FOG.extended(build('fog LINEAR')[1]))
    st.fog_end = 6.5           # so the demo's far floor is fully fogged
    img = R.render(sc, st)
    w, h = 96, 72
    bd = SKY.backdrop_rgb(sc, st, w, h)
    g, view, vp, eye = _gbuf_of(sc, st)
    cov = g.tri >= 0
    py, px = np.nonzero(cov)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye, w, h)
    depth = job.context(g.tri[py, px], g.bary[py, px], px, py).depth
    full = depth >= st.fog_end + 1e-3
    check('a covered pixel at f == 0 is bitwise the backdrop at that pixel '
          '(the same _background_image call that fills the sky)',
          bool(full.any()) and bool(np.array_equal(
              img[py[full], px[full], :3], bd[py[full], px[full]])),
          f'{int(full.sum())} fully fogged px')
    tg = bd[py, px]
    check('...and the target varies per pixel over the covered pixels (the '
          'gradient sky behind the ball and box; the far floor looks below '
          'the horizon, where a GRADIENT world is one colour)',
          len({tuple(c) for c in tg.tolist()}) > 8)
    sc2, st2 = build(key)
    st2.fog_end = 6.5
    st2.fog_color_source = 'FIXED'
    img_fx = R.render(sc2, st2)
    unc = ~cov
    check('the uncovered pixels equal the FIXED frame\'s (the backdrop '
          'receives no fog)', bool(np.array_equal(img[unc], img_fx[unc])))
    check('the covered pixels moved against the FIXED target',
          float(np.abs(img[cov] - img_fx[cov]).max()) > 1e-2)
    # the twin under the gradient world, then a Bryce world (both the
    # upload road); the print names the upload and its size
    for label, mut in (('GRADIENT', None),
                       ('BRYCE', lambda s, t: setattr(s.world, 'mode', 'BRYCE'))):
        sc_, st_, cpu_, g_, job_ = _rig(key, mut)
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                passes, why, atl = _plan(job_, g_, True)
            check(f'{label}: the plan carries the hal_backdrop atlas and '
                  'prints the upload by name with its size',
                  passes is not None and 'hal_backdrop' in (atl or {})
                  and 'backdrop fog' in buf.getvalue()
                  and 'MB' in buf.getvalue(), str(why))
            if passes is None:
                continue
            check(f'{label}: every pass declares and binds hal_backdrop '
                  '(the definition rides every pass)',
                  all('hal_backdrop' in (b.get('samplers') or [])
                      and 'uniform sampler2D hal_backdrop;' in s
                      for _m, _n, s, b in passes))
            arr = atl['hal_backdrop'][1]()
            check(f'{label}: the atlas is the CPU\'s own backdrop (h, w, 4) '
                  'float32, keyed by a crc', arr.shape == (72, 96, 4)
                  and arr.dtype == np.float32
                  and bool(np.array_equal(arr[..., :3],
                                          FOG.backdrop_of(job_)))
                  and atl['hal_backdrop'][0][0] == 'backdrop')
        finally:
            GSH.FOG_ON_GPU = True
        da, probe, _r = _arith_twin_w2(key, mut)
        check(f'{label}: hal_fog\'s f is the CPU\'s bit for bit (d == 0.0)',
              da is not None and da == 0.0, str(probe if da is None else da))
        d, dc, passes, _atl, why = _twin(key, mut)
        check(f'{label}: the frame with fog in the pass equals the CPU fog '
              f'over the same simulated shading within {P_ULP_BAR:g} '
              '(LINEAR is smooth)', d is not None and d <= P_ULP_BAR,
              str(why if d is None else d))
        check(f'{label}: within the deferred bar and no fog_cpu pass',
              dc is not None and dc < 1e-3 and passes is not None
              and not _fog_cpu(passes), str(why if dc is None else dc))
    # vertex rate: the corners take the direction form (A31)
    def gouraud(sc, st):
        st.shading_rate = 'VERTEX'
    sc_v, st_v, cpu_v, g_v, job_v = _rig(key, gouraud)
    try:
        passes, why, atl = _plan(job_v, g_v, True)
        check('a Gouraud backdrop-fog frame plans with the fog in the '
              'CPU-lit corners (no fog_cpu, no hal_fog call)',
              passes is not None and not _fog_cpu(passes)
              and not any(_calls(s) for _m, _n, s, _b in passes), str(why))
        if passes is not None:
            img_v, hit_v = GSH.simulate(job_v, g_v, passes, atl)
            cov_v = (g_v.tri >= 0) & hit_v
            dv = float(np.abs(img_v[cov_v] - cpu_v[cov_v][:, :3]).max())
            check('...bit-level against the CPU (d < 1e-6, the corner bar; '
                  'a fogged corner is the world along its direction, not '
                  'the pixel\'s backdrop -- disclosed)', dv < 1e-6, str(dv))
    finally:
        GSH.FOG_ON_GPU = True
    # the fake device: the backdrop is uploaded once and reused
    gpu, cpu, live, keys1, keys2, gpu2 = _fake_frame(key)
    bkeys = [k for k in keys2 if k[0] == 'backdrop']
    dd = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check('the whole GPU road headless under backdrop fog: within 6e-6 of '
          'the CPU frame, alpha bitwise, nothing left live',
          dd <= 6e-6 and bool(np.array_equal(gpu[..., 3], cpu[..., 3]))
          and live == [], f'max {dd}; live {len(live)}')
    check('hal_backdrop was uploaded once (upload_cached by its crc) and '
          'the second frame reused it', len(bkeys) == 1
          and bool(np.array_equal(gpu2, gpu)), str(len(bkeys)))
    lw = PRESETS['LIGHTWAVE_56']['settings']
    check('the LIGHTWAVE_56 preset names the backdrop as its fog target '
          '(fog itself off, as LightWave\'s panel defaults)',
          lw.get('fog_color_source') == 'BACKDROP' and not lw.get('fog'))


# --------------------------------------------------------------- F006

def test_f006_material_fog_control():
    """Model 3's burn-through, System 22's bias and second bank, the
    fogAmbient register. Identity at the defaults; the laws on the CPU
    function (burn 1 is inert fog, bias raises the opacity, bank 1 moves
    only the material on it, ambient halves the target); the refusal
    road honours the dials (binds['fog_mat']); a varying socket refuses
    by name; twin d == 0.0 on both rows; the SEGA_MODEL3 preset."""
    from ..core import shading as SH
    st = base_settings(96, 72)
    st.fog = True
    surf = SH.Surface(4)
    check('a default Surface carries no fog dial: not extended, no material '
          'lines', not FOG.material_dials(surf) and not FOG.extended(st, surf)
          and FOG.bank_consts(st, surf) is None)
    ramp = np.linspace(0.5, 15.0, 64).astype(np.float32)
    st.fog_mode, st.fog_start, st.fog_end = 'LINEAR', 3.0, 12.0
    from types import SimpleNamespace
    n = ramp.size
    plain = FOG.factor(ramp, st, None)
    s_burn = SimpleNamespace(fog_burn=np.ones(n, np.float32),
                             fog_bias=np.zeros(n, np.float32),
                             fog_bank=np.zeros(n, np.float32))
    f_burn = FOG.factor(ramp, st, None, surf=s_burn)
    check('burn-through 1 is inert fog: f == 1 bitwise along the ramp',
          bool(np.all(f_burn == np.float32(1.0))))
    s_bias = SimpleNamespace(fog_burn=np.zeros(n, np.float32),
                             fog_bias=np.full(n, 0.5, np.float32),
                             fog_bank=np.zeros(n, np.float32))
    f_bias = FOG.factor(ramp, st, None, surf=s_bias)
    check('bias 0.5 raises the opacity monotonically: f <= the plain f '
          'everywhere and below it before Fog End',
          bool(np.all(f_bias <= plain))
          and bool(np.all(f_bias[ramp < 12.0] < plain[ramp < 12.0])))
    s_neg = SimpleNamespace(fog_burn=np.zeros(n, np.float32),
                            fog_bias=np.full(n, -1.0, np.float32),
                            fog_bank=np.zeros(n, np.float32))
    check('bias -1 clamps to no fog (f == 1)',
          bool(np.all(FOG.factor(ramp, st, None, surf=s_neg) == 1.0)))
    s_zero = SimpleNamespace(fog_burn=np.zeros(n, np.float32),
                             fog_bias=np.zeros(n, np.float32),
                             fog_bank=np.zeros(n, np.float32))
    check('a surface with every dial at 0 leaves f bitwise the plain f (an '
          'inert control is inert: the material lines are not emitted)',
          bool(np.array_equal(FOG.factor(ramp, st, None, surf=s_zero), plain)))
    # bank 1
    st.fog_bank1_start, st.fog_bank1_end = 1.0, 5.0
    s_bank = SimpleNamespace(fog_burn=np.zeros(n, np.float32),
                             fog_bias=np.zeros(n, np.float32),
                             fog_bank=np.ones(n, np.float32))
    f_bank = FOG.factor(ramp, st, None, surf=s_bank)
    st_b = base_settings(96, 72)
    st_b.fog, st_b.fog_mode, st_b.fog_start, st_b.fog_end = True, 'LINEAR', 1.0, 5.0
    f_ref = FOG.factor(ramp, st_b, None, surf=s_zero)
    ref2 = np.float32(1.0) - (np.float32(1.0) - f_ref)   # the dial lines' re-round
    check('bank 1 reads its own Start/End: f equals the scene curve at 1..5 '
          'up to the dial lines\' own re-round (max 1 ulp)',
          float(np.abs(f_bank - f_ref).max()) <= 1.2e-7
          and bool(np.array_equal(f_bank, ref2)),
          f'max {float(np.abs(f_bank - f_ref).max())}')
    st.fog_bank1_end = 0.0
    check('an inert bank (End <= Start) leaves bank-1 points on the scene '
          'curve', not FOG.bank1_valid(st)
          and float(np.abs(FOG.factor(ramp, st, None, surf=s_bank)
                           - plain).max()) <= 1.2e-7)
    st.fog_bank1_end = 5.0
    st.fog_mode = 'EXP'
    st.fog_density = 0.1
    fe = FOG.factor(ramp, st, None, surf=s_bank)
    fe0 = FOG.factor(ramp, st, None, surf=s_zero)
    check('EXP reads density only: the bank is inert there (the tooltip '
          'says so)', FOG.bank_consts(st, s_bank) is None
          and float(np.abs(fe - fe0).max()) <= 1.2e-7)
    # the frames: burn-1 pixels bitwise the unfogged frame; bank 1 moves
    # only the material on it; ambient halves the target
    key, key_b = ('per-material fog burn/bias (Model 3 / System 22)',
                  'fog bank 1 (System 22)')
    sc, st = build(key)
    img = R.render(sc, st)
    sc0, st0 = build(key)
    st0.fog = False
    img0 = R.render(sc0, st0)
    g, view, vp, eye = _gbuf_of(sc, st)
    m = _mat_px(sc, g)
    ball = m == 1
    box = m == 2
    floor = m == 0
    check('the Fog Burn-Through 1 material (the ball) is bitwise the '
          'UNFOGGED frame while the floor is fogged',
          bool(ball.any()) and bool(np.array_equal(img[ball], img0[ball]))
          and float(np.abs(img[floor] - img0[floor]).max()) > 1e-2)
    sv = _socket_vals(sc)
    check('the builder\'s sockets: ball burn 1; box bias 0.5 on bank 1',
          sv[1][0] == 1.0 and sv[2] == (0.0, 0.5, 1.0), str(sv))
    sc_b, st_b = build(key_b)
    img_b = R.render(sc_b, st_b)
    check('bank 1 (1..5) moves the material on bank 1 (the box) and not the '
          'floor or the ball (band-invariance per material)',
          bool(box.any())
          and float(np.abs(img_b[box] - img[box]).max()) > 1e-3
          and bool(np.array_equal(img_b[floor], img[floor]))
          and bool(np.array_equal(img_b[ball], img[ball])))
    sc_a, st_a = build('fog ambient 0.5 (Model 3 fogAmbient)')
    img_a = R.render(sc_a, st_a)
    sc_l, st_l = build('fog LINEAR')
    img_l = R.render(sc_l, st_l)
    py, px = np.nonzero(g.tri >= 0)
    job = R.ShadeJob(sc_l, st_l, R.prepare_textures(sc_l, st_l), None, view,
                     eye, 96, 72)
    depth = job.context(g.tri[py, px], g.bary[py, px], px, py).depth
    f = FOG.factor(depth, st_l, sc_l)
    col = np.asarray(st_l.fog_color, np.float32)[None, :]
    expect = col * 0.5 * (1.0 - f)[:, None]
    got = img_l[py, px, :3] - img_a[py, px, :3]
    check('fog_ambient 0.5 halves the fog colour term exactly: LINEAR - '
          'ambient frame == 0.5 * col * (1 - f) within 2e-6',
          float(np.abs(got - expect).max()) < 2e-6
          and float(np.abs(got).max()) > 1e-2,
          f'max {float(np.abs(got - expect).max())}')
    # the refusal road honours the dials (binds['fog_mat'], A25's road)
    sc, st, cpu, g, job = _rig(key)
    try:
        passes, why, atl = _plan(job, g, False)
        check('with the module switch off every pass carries fog_mat (burn, '
              'bias, bank) beside fog_cpu', passes is not None
              and all((b or {}).get('fog_mat') is not None
                      for _m, _n, _s, b in passes)
              and {int(mm): tuple(b['fog_mat']) for mm, _n, _s, b in passes}
              == _socket_vals(sc), str(why))
        if passes is not None:
            img_c = np.full((72, 96, 3), 0.5, np.float32)
            before = img_c.copy()
            hit = g.tri >= 0
            out2 = GSH._fog_readback(job, g, passes, img_c, hit)
            edited = np.any(out2 != before, axis=2)
            check('the readback fog leaves the burn-through material\'s '
                  'pixels untouched and fogs the others (the refusal road '
                  'honours the dials)', not bool(edited[ball & hit].any())
                  and bool(edited[floor & hit].all()))
    finally:
        GSH.FOG_ON_GPU = True
    # twin: both rows (LINEAR: the P-ULP bar on the frame)
    for k in (key, key_b):
        da, probe, _r = _arith_twin_w2(k)
        check(f'{k}: hal_fog\'s f is the CPU\'s bit for bit on the '
              'shader\'s own depth with the material dials (d == 0.0)',
              da is not None and da == 0.0, str(probe if da is None else da))
        if da is not None:
            srcs = {int(mm): s for mm, _n, s, _b in probe['passes']}
            check(f'{k}: the ball and the box call the five-argument hal_fog '
                  'with their dials as hal_mats values; the floor keeps the '
                  'two-argument call', _CALL5 in srcs[1] and _CALL5 in srcs[2]
                  and _CALL in srcs[0] and 'float fburn' in srcs[1])
        d, dc, passes, _atl, why = _twin(k)
        check(f'{k}: the frame with fog in the pass equals the CPU fog over '
              f'the same simulated shading within {P_ULP_BAR:g}',
              d is not None and d <= P_ULP_BAR, str(why if d is None else d))
        check(f'{k}: within the deferred bar and no fog_cpu pass',
              dc is not None and dc < 1e-3 and passes is not None
              and not _fog_cpu(passes), str(why if dc is None else dc))
    # a Noise-driven Fog Bias refuses by name
    def noisy(sc, st):
        gph = sc.materials[2].graph
        gph['nodes']['noise'] = {
            'id': 'noise', 'bl_idname': 'HALCYON_NoiseNode',
            'props': {'kind': 'RIDGED', 'octaves': 3},
            'inputs': [FM._sk('Vector', 'VECTOR', [0, 0, 0]),
                       FM._sk('Scale', 'VALUE', 4.0),
                       FM._sk('Lacunarity', 'VALUE', 2.0),
                       FM._sk('Gain', 'VALUE', 0.5),
                       FM._sk('Color 1', 'RGBA', [0.1, 0.05, 0.3, 1.0]),
                       FM._sk('Color 2', 'RGBA', [1.0, 0.9, 0.6, 1.0])],
            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                        {'name': 'Fac', 'type': 'VALUE'}]}
        for ins in gph['nodes']['bsdf']['inputs']:
            if ins['name'] == 'Fog Bias':
                ins['link'] = ['noise', 1]
    sc, st, cpu, g, job = _rig(key, noisy)
    try:
        passes, why, atl = _plan(job, g, True)
        check('a Noise-driven Fog Bias socket refuses the GPU by name '
              '(fog_bias varies across the frame)', passes is None
              and 'fog_bias varies' in str(why), str(why))
    finally:
        GSH.FOG_ON_GPU = True
    m3 = PRESETS['SEGA_MODEL3']['settings']
    check('the SEGA_MODEL3 preset carries a live second bank (6..45) for a '
          'material\'s Fog Bank 1 socket', m3.get('fog_bank1_start') == 6.0
          and m3.get('fog_bank1_end') == 45.0)


# --------------------------------------------------------------- F009

def test_f009_pov_ground_fog():
    """POV-Ray's fog_type 2: density 1/(1+Y^2) above the offset integrated
    along the eye ray (the atan mean), the sky fogged by elevation. The
    laws on the CPU functions, the frame (a table and the height layer
    inert, the rounding skipped), determinism, no NumPy warning, the
    twin d == 0.0 (arithmetic) and the sky pass's twin 0.0."""
    import warnings
    key = 'POV ground fog (atan integral)'
    sc, st = build(key)
    check('GROUND is an extended mode that skips the 1/8 rounding',
          FOG.extended(st) and not FOG.vertex_quantised(st)
          and FOG.mode_of(st) == 'GROUND')
    eye = np.asarray([0.0, 0.0, 2.0], np.float32)
    st.fog_ground_offset, st.fog_ground_alt, st.fog_density = 0.0, 1.0, 0.1
    L = 10.0
    low = eye + np.asarray([[L, 0.0, 0.0]], np.float32)
    up = eye + np.asarray([[0.0, 0.0, L]], np.float32)
    f_low = FOG.ground_fog_integral(low, eye, st)
    f_up = FOG.ground_fog_integral(up, eye, st)
    check('two points at equal ray length: the lower one is foggier (the '
          'mean density is non-increasing in height)',
          float(f_low[0]) < float(f_up[0]), f'{float(f_low[0])} {float(f_up[0])}')
    st.fog_ground_offset = 5.0        # the eye and the point below the layer
    P0 = np.asarray([[3.0, 4.0, 1.0]], np.float32)
    f0 = FOG.ground_fog_integral(P0, eye, st)
    dd = np.float32(np.sqrt(np.float32(3.0 * 3.0 + 4.0 * 4.0 + 1.0 * 1.0)))
    e = dd * np.float32(1.0)
    e = e * np.float32(0.1)
    ref = np.exp(-e)
    check('a segment inside the constant layer (y1 <= 0, y2 <= 0) gives '
          'exactly exp(-length * density)', float(f0[0]) == float(ref),
          f'{float(f0[0])} {float(ref)}')
    st.fog_ground_offset = 0.0
    cols = np.ones((3, 3), np.float32)
    dirs = np.asarray([[1.0, 0.0, 1e-6], [0.0, 0.0, 1.0], [0.0, 1.0, -0.2]],
                      np.float32)
    dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    sky = FOG.ground_fog_sky(cols, dirs, st, eye)
    fc = np.asarray(st.fog_color, np.float32)
    check('the sky closed form: a horizon ray (D.z -> 0+) tends to the fog '
          'colour, a falling ray IS the fog colour, the zenith keeps more '
          'than half of the sky (alt 1, density 0.1)',
          float(np.abs(sky[0] - fc).max()) < 1e-4
          and bool(np.array_equal(sky[2], fc))
          and float(sky[1].min()) > 0.5 + float(fc.max()) * 0.0
          and float(np.abs(sky[1] - fc).max()) > 1e-2, str(sky.tolist()))
    # the frame: the uncovered pixels are fogged by elevation (B9)
    keyg = 'POV ground fog, the sky fogged by elevation'
    scg, stg = build(keyg)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        img = R.render(scg, stg)
        img2 = R.render(*build(keyg))
    check('the ground fog frame renders twice bitwise with no NumPy warning '
          'escaping (A32: the discarded candidates divide by zero silently)',
          bool(np.array_equal(img, img2)))
    scd, std = build(key)               # the demo camera looks DOWN
    img_d = R.render(scd, std)
    scu, stu = build(key)
    stu.fog = False
    img_u = R.render(scu, stu)
    g, view, vp, eye_g = _gbuf_of(scd, std)
    unc = ~(g.tri >= 0)
    fc_g = np.asarray(std.fog_color, np.float32)
    check('the demo camera looks down: every sky ray falls, so the sky IS '
          'fogged to the fog colour exactly (POV fogs a falling miss ray '
          'fully) and differs from the unfogged sky',
          bool(unc.any()) and bool(np.all(img_d[unc][:, :3] == fc_g[None, :]))
          and float(np.abs(img_d[unc] - img_u[unc]).max()) > 1e-2)
    # the elevation law needs rising rays: the camera pitched up
    from .scenebuild import look_at_matrix

    def pitch(sc_, st_):
        sc_.camera.matrix_world = look_at_matrix((5.2, -6.4, 1.0),
                                                 (0.0, -0.2, 3.4))
    scp, stp = build(keyg)
    pitch(scp, stp)
    img_p = R.render(scp, stp)
    scp0, stp0 = build(keyg)
    pitch(scp0, stp0)
    stp0.fog = False
    img_p0 = R.render(scp0, stp0)
    gp, view_p, vp_p, eye_p = _gbuf_of(scp, stp)
    unc_p = ~(gp.tri >= 0)
    py_u, px_u = np.nonzero(unc_p)
    inv = np.linalg.inv(vp_p).astype(np.float32)
    nx = (px_u.astype(np.float32) + 0.5) / 96 * 2.0 - 1.0
    ny = (py_u.astype(np.float32) + 0.5) / 72 * 2.0 - 1.0
    pts = np.stack([nx, ny, np.ones_like(nx), np.ones_like(nx)], 1)
    wp = pts @ inv.T
    wp = wp[:, :3] / wp[:, 3:4]
    dz = (wp - eye_p[None, :])
    dz = dz[:, 2] / np.linalg.norm(dz, axis=1)
    moved = np.abs(img_p[py_u, px_u, :3] - img_p0[py_u, px_u, :3]).max(axis=1)
    rising = dz > 0.02
    falling = dz <= 0.0
    q_hi = rising & (dz >= np.percentile(dz[rising], 75))
    q_lo = rising & (dz <= np.percentile(dz[rising], 25))
    check('pitched up: the falling sky rays are the fog colour exactly, '
          'the rising ones are fogged by elevation -- the highest quartile '
          'moved less than the lowest (POV\'s closed form on D.z)',
          bool(rising.sum() > 50) and bool(q_hi.any()) and bool(q_lo.any())
          and (not falling.any() or bool(np.all(
              img_p[py_u[falling], px_u[falling], :3] == fc_g[None, :])))
          and float(moved[q_hi].mean()) < float(moved[q_lo].mean())
          and float(moved[q_lo].mean()) > 1e-2,
          f'rising {int(rising.sum())} hi {float(moved[q_hi].mean()) if q_hi.any() else -1:.4f} '
          f'lo {float(moved[q_lo].mean()) if q_lo.any() else -1:.4f}')
    # inert dials under GROUND: a hardware table, the height layer, the
    # per-vertex rounding
    for field, val, why_ in (('fog_table', 'VOODOO64', 'a hardware table'),
                             ('fog_height', True, 'the height layer'),
                             ('fog_vertex', True, 'Per-Vertex Fog')):
        sci, sti = build(keyg)
        setattr(sti, field, val)
        check(f'{why_} is inert under Ground Fog (bitwise the row)',
              bool(np.array_equal(R.render(sci, sti), img)))
    sti = build(keyg)[1]
    sti.fog_table = 'PVR128'
    notes = FOG.plan_notes(sti, False)
    check('...and the plan\'s notes name the inert table (PVR128 under '
          'Ground Fog) while table_of reads NONE',
          len(notes) == 1 and 'PVR128' in notes[0]
          and FOG.table_of(sti) == 'NONE', str(notes))
    # A/B versus EXP at the same density
    sce, ste = build(key)
    ste.fog_mode = 'EXP'
    check('ground fog differs from EXP at the same density',
          float(np.abs(R.render(sce, ste) - R.render(*build(key))).max())
          > 1e-2)
    # the twins
    for k in (key, keyg):
        da, probe, _r = _arith_twin_w2(k)
        check(f'{k}: hal_fog\'s f is the CPU\'s bit for bit on the '
              'shader\'s own P (d == 0.0; atan / sqrt / exp are NumPy\'s '
              'float32 on both sides)', da is not None and da == 0.0,
              str(probe if da is None else da))
        d, dc, passes, _atl, why = _twin(k)
        check(f'{k}: the frame with fog in the pass equals the CPU fog over '
              f'the same simulated shading within {P_ULP_BAR:g}',
              d is not None and d <= P_ULP_BAR, str(why if d is None else d))
        check(f'{k}: within the deferred bar and no fog_cpu pass',
              dc is not None and dc < 1e-3 and passes is not None
              and not _fog_cpu(passes), str(why if dc is None else dc))
    from . import test_r251_sky_camera as TS
    from ..gpu import sky as GSKY
    scs, sts = build(keyg)
    ds, whys, plan = TS.twin_d(scs, sts, {}, 1)
    check('the sky pass under Ground Fog is bitwise the CPU\'s fogged sky '
          'in the simulator (d == 0.0; the driver bar is 2e-5, A33)',
          ds is not None and ds == 0.0, str(whys if ds is None else ds))
    check('...its params carry the closed form\'s one float32 product and '
          'the fog colour', plan is not None
          and plan['params'].get('hal_sky_gfog', (0, 0, 0))[2] == 1.0
          and 'hal_ground_fog_sky(col, d0)' in GSKY.SOURCE)
    scs2, sts2 = build('LightWave backdrop fog')
    ds2, whys2, _p = TS.twin_d(scs2, sts2, {}, 1)
    check('...and every other mode keeps the sky twin at 0.0 (the tail '
          'returns col untouched)', ds2 is not None and ds2 == 0.0,
          str(whys2 if ds2 is None else ds2))
    # the fake device
    gpu, cpu, live, k1, k2, gpu2 = _fake_frame(keyg)
    scf, stf = build(keyg)
    gf, _v, _vp, _e = _gbuf_of(scf, stf)
    unc_f = ~(gf.tri >= 0)
    d_sky = float(np.abs(gpu[unc_f][:, :3] - cpu[unc_f][:, :3]).max())
    d_geo = float(np.abs(gpu[~unc_f][:, :3] - cpu[~unc_f][:, :3]).max())
    check('the whole GPU road headless under ground fog: the sky drawn in '
          'the burst is bitwise the CPU\'s fogged sky, the geometry within '
          'the deferred bar (6e-3; measured), alpha bitwise, nothing live',
          d_sky == 0.0 and d_geo < 6e-3
          and bool(np.array_equal(gpu[..., 3], cpu[..., 3])) and live == [],
          f'sky {d_sky} geometry {d_geo}; live {len(live)}')
    check('the POVRAY_31 preset\'s note names Ground Fog',
          'Ground Fog' in PRESETS['POVRAY_31']['note'])


# --------------------------------------------------------------- F010

def test_f010_pov_turbulent_fog():
    """POV's turbulent fog: one turbulence read at the fogged segment's
    middle scales the fog distance, faded by exp(-distance * density).
    Identity at 0, A/B, determinism, the thinning law and its near-field
    variance, turb_depth 1 thins more, PRIM_GLSL once per pass, twin
    d == 0.0 on the shader's own P."""
    key, key1 = 'POV turbulent fog', 'POV turbulent fog, depth 1.0'
    sc, st = build(key)
    check('turbulence > 0 is an extended road; 0 is not',
          FOG.extended(st) and FOG.turb_on(st)
          and not FOG.extended(build('fog EXP')[1]))
    img = R.render(sc, st)
    img2 = R.render(*build(key))
    check('two renders are bitwise (the hash has no frame term)',
          bool(np.array_equal(img, img2)))
    sc0, st0 = build(key)
    st0.fog_turbulence = 0.0
    img0 = R.render(sc0, st0)
    check('turbulence 1.0 moves the frame against the plain EXP fog',
          float(np.abs(img - img0).max()) > 1e-2)
    g, view, vp, eye = _gbuf_of(sc, st)
    py, px = np.nonzero(g.tri >= 0)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     96, 72)
    ctx = job.context(g.tri[py, px], g.bary[py, px], px, py)
    f_t = FOG.factor(ctx.depth, st, sc, P=ctx.P, ctx=ctx)
    f_p = FOG.factor(ctx.depth, st0, sc, P=ctx.P, ctx=ctx)
    diff = f_t - f_p
    check('turbulence only shortens the fogged distance: f_turb - f_plain '
          '>= 0 at every pixel, > 0 somewhere', bool(np.all(diff >= 0.0))
          and bool((diff > 0.0).any()), f'min {float(diff.min())}')
    order = np.argsort(ctx.depth)
    third = order.size // 3
    dd_t = FOG.turbulence(ctx.depth, ctx.P, job.eye, st)
    short = (ctx.depth - dd_t) / ctx.depth
    s_near = float(short[order[:third]].mean())
    s_far = float(short[order[-third:]].mean())
    v_near = float(np.var(short[order[:third]]))
    v_far = float(np.var(short[order[-third:]]))
    check('the relative shortening of the fog distance is larger in the '
          'nearest depth third than in the farthest (k = exp(-distance * '
          'density) fades the read: near fog wispy, far fog smooth) and '
          'stays within 0..1', s_near > s_far
          and bool(np.all(short >= 0.0)) and bool(np.all(short <= 1.0)),
          f'near {s_near:.4f} ({v_near:.2e}) far {s_far:.4f} ({v_far:.2e})')
    sc1, st1 = build(key1)
    f_1 = FOG.factor(ctx.depth, st1, sc, P=ctx.P, ctx=ctx)
    check('turb_depth 1.0 thins the fog at least as much as 0.5 (mean f '
          'not lower)', float(f_1.mean()) >= float(f_t.mean()))
    # PRIM_GLSL once per pass: a Noise-graph material and a plain one
    for k, mut, label in ((key, None, 'plain demo materials'),
                          (key, lambda s, t: setattr(
                              s.materials[1], 'graph',
                              FM._sc_noise_node(t).materials[1].graph),
                           'a Noise-graph material')):
        sc_, st_, cpu_, g_, job_ = _rig(k, mut)
        try:
            passes, why, atl = _plan(job_, g_, True)
            check(f'{label}: every pass inlines the pattern library exactly '
                  'once (float hal_pt_turb( count == 1, A36)',
                  passes is not None and all(
                      s.count('float hal_pt_turb(') == 1
                      for _m, _n, s, _b in passes), str(why))
        finally:
            GSH.FOG_ON_GPU = True
    for k in (key, key1):
        da, probe, _r = _arith_twin_w2(k)
        check(f'{k}: hal_fog\'s f is the CPU\'s bit for bit on the '
              'shader\'s own P (d == 0.0; hal_pt_turb is PT.turbulence)',
              da is not None and da == 0.0, str(probe if da is None else da))
        d, dc, passes, _atl, why = _twin(k)
        check(f'{k}: the frame with fog in the pass equals the CPU fog over '
              f'the same simulated shading within {P_ULP_BAR:g} (EXP is '
              'smooth)', d is not None and d <= P_ULP_BAR,
              str(why if d is None else d))
        check(f'{k}: within the deferred bar and no fog_cpu pass',
              dc is not None and dc < 1e-3 and passes is not None
              and not _fog_cpu(passes), str(why if dc is None else dc))
    gpu, cpu, live, k1, k2, gpu2 = _fake_frame(key)
    dd = float(np.abs(gpu[..., :3] - cpu[..., :3]).max())
    check('the whole GPU road headless under turbulent fog: within 6e-6 of '
          'the CPU frame, nothing left live', dd <= 6e-6 and live == [],
          f'max {dd}; live {len(live)}')
    check('the POVRAY_31 preset\'s note names Fog Turbulence',
          'Turbulence' in PRESETS['POVRAY_31']['note'])


# --------------------------------------------------- F015 (the fog half)

def test_f015_spot_fog_half():
    """LIGHT-B1's F015 handed this slot its fog half: Spotlight Fog adds
    the screen spotlights' lobe (colE * en * el) to the fog target
    (Supermodel's spotFogColor x fogAttenuation). Identity without a
    screen-spot lamp; the row moves the frame (pass 1's ledger: 0 px);
    the lobe brightens the fog inside the ellipse only; the refusal
    road recomputes the lobe (A52); twin within the P-ULP bar."""
    key = 'Model 3 spotlight fog lobe'
    RP = _prev_engine('halcyon-1.89.0.zip')
    sc, st = build('fog LINEAR')
    st.fog_spot = 1.0
    check('fog_spot > 0 is an extended road', FOG.extended(st) and FOG.spot_on(st))
    if RP is not None:
        now = R.render(sc, st)
        sc2, st2 = build('fog LINEAR')
        st2.fog_spot = 1.0
        prev = RP.render(sc2, st2)
        check('fog_spot 1.0 without a screen-spot lamp renders bitwise the '
              '1.89.0 engine (no lobe, the target untouched)',
              bool(np.array_equal(np.asarray(now), np.asarray(prev))))
    sc, st = build(key)
    img = R.render(sc, st)
    sc0, st0 = build(key)
    st0.fog_spot = 0.0
    img0 = R.render(sc0, st0)
    d_img = np.abs(img - img0).max(axis=2)
    check('the Model 3 spotlight fog lobe row moves the frame (the pass-1 '
          'ledger measured 0 px: the fog half was not in the tree)',
          float(d_img.max()) > 1e-2, str(float(d_img.max())))
    g, view, vp, eye = _gbuf_of(sc, st)
    cov = g.tri >= 0
    py, px = np.nonzero(cov)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     96, 72)
    depth = job.context(g.tri[py, px], g.bary[py, px], px, py).depth
    sf = FOG.spot_fog_of(job, px, py, depth)
    inside = sf.max(axis=1) > 0.0
    moved = d_img[py, px] > 0.0
    check('the lobe brightens the fog INSIDE the screen ellipse only: every '
          'moved covered pixel has a non-zero lobe, and lobe pixels moved',
          sf is not None and bool(np.all(inside[moved]))
          and bool(moved[inside].sum() > 0.5 * inside.sum()),
          f'moved {int(moved.sum())} inside {int(inside.sum())}')
    check('...and the lobe only ADDS to the target (no channel darkens)',
          bool(np.all(img[py, px, :3] - img0[py, px, :3] >= -1e-6)))
    # the twin: in-pass fog vs the refusal road's recomputed lobe (A52)
    sc_, st_, cpu, g_, job_ = _rig(key)
    try:
        passes, why, atl = _plan(job_, g_, True)
        check('the row plans with the lobe argument on the lit passes '
              '(hal_spotfog) and the fspot parameter in the definition',
              passes is not None and all(
                  'hal_fog(total, P, hal_spotfog)' in s
                  and 'vec3 fspot)' in s for _m, _n, s, _b in passes),
              str(why))
    finally:
        GSH.FOG_ON_GPU = True
    d, dc, passes, _atl, why = _twin(key)
    check(f'{key}: the frame with fog in the pass equals the CPU fog over '
          f'the same simulated shading within {P_ULP_BAR:g} (the refusal '
          'road recomputes the lobe)', d is not None and d <= P_ULP_BAR,
          str(why if d is None else d))
    check(f'{key}: within the deferred bar and no fog_cpu pass',
          dc is not None and dc < 1e-3 and passes is not None
          and not _fog_cpu(passes), str(why if dc is None else dc))


# -------------------------------------------------------------------- main

TESTS = [
    test_00_identity_prev_engine,
    test_f000_fog_depth_is_the_cpus,
    test_f000_gpu_twin,
    test_f000_refusal_by_name,
    test_f000_vertex_rate_still_in_the_corners,
    test_f000_use_mist_lifted,
    test_f000_readback_is_structural,
    test_f000_fake_device,
    test_f001_gte_hyperbolic_depth_cue,
    test_f002_voodoo_fog_table,
    test_f003_pvr_fog_table,
    test_f022_ds_fog_table,
    test_f007_d3d_zfog,
]


# ---- wave 2 ---- the LIGHT-A2 tests join the runner after wave 1's
TESTS.extend([
    test_w2_00_identity_prev_engine,
    test_f004_gc_fog_range_adjust,
    test_f005_face_fog,
    test_f008_backdrop_fog,
    test_f006_material_fog_control,
    test_f009_pov_ground_fog,
    test_f010_pov_turbulent_fog,
    test_f015_spot_fog_half,
])


def test_f007_zfog_advisory():
    """1.90.0 fix pass 3: the Fog End advisory under Fog Depth Z. It
    compared scene-unit distance with a 0..1 End, so it cried PURE fog at
    any End <= 1 and said nothing while a scene-unit Start fogged nothing
    at all (measured: 0 pixels). Now both questions are asked in z."""
    import re
    key = 'Direct3D z-fog'
    w, h = 96, 72
    sc, st = build(key)
    view, proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    py, px = np.nonzero(g.tri >= 0)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye, w, h)
    depth = job.context(g.tri[py, px], g.bary[py, px], px, py).depth
    n32, k32, _o = FOG.projection_consts(sc, w, h)
    zpx = FOG.z_from_w(np.ascontiguousarray(depth), n32, k32, False)
    lo, hi = float(zpx.min()), float(zpx.max())

    def note(**kw):
        _sc, s = build(key)
        for k, v in kw.items():
            setattr(s, k, v)
        return R.fog_coverage_note(proj, g, s)
    n_trap = note(fog_start=5.0, fog_end=40.0)
    check('Z with scene-unit Start/End (5, 40): the advisory says NOTHING is '
          'fogged and names the 0..1 domain and the way out',
          n_trap is not None and 'NOTHING is fogged' in n_trap
          and 'Fog Depth is Z' in n_trap and 'Fog Depth to W' in n_trap,
          str(n_trap)[:80])
    m = re.search(r'span ([0-9.]+)\.\.([0-9.]+)', n_trap or '')
    check("...and the span it prints is the frame's own z range (to 1e-3 of "
          'the shader context\'s depths through z_from_w)',
          m is not None and abs(float(m.group(1)) - lo) < 1e-3
          and abs(float(m.group(2)) - hi) < 1e-3,
          f'{m.groups() if m else None} vs {lo:.4f}..{hi:.4f}')
    sc_a, st_a = build(key)
    st_a.fog_start, st_a.fog_end = 5.0, 40.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        img_a = R.render(sc_a, st_a)
    sc_0, st_0 = build(key)
    st_0.fog = False
    img_0 = R.render(sc_0, st_0)
    check('...and that frame IS bitwise the unfogged frame (the trap is real), '
          'with the line printed by render()',
          bool(np.array_equal(img_a, img_0))
          and 'NOTHING is fogged' in buf.getvalue())
    n_ok = note(fog_start=lo + 0.25 * (hi - lo), fog_end=hi * 1.00001)
    check('Z with Start/End inside the frame\'s span stays silent (it cried '
          'PURE fog before: any End <= 1 lay below every scene distance)',
          n_ok is None, str(n_ok)[:80])
    n_wall = note(fog_start=0.0, fog_end=lo * 0.5)
    check('Z with End below every surface: the PURE fog line, in the z domain',
          n_wall is not None and 'PURE fog colour' in n_wall
          and 'Fog Depth is Z' in n_wall, str(n_wall)[:80])
    n_w = note(fog_depth='W', fog_start=1.0, fog_end=3.0)
    check('W keeps its own wording (scene distances, no z talk)',
          n_w is not None and 'PURE fog colour' in n_w
          and 'scaled parents' in n_w and 'Fog Depth' not in n_w, str(n_w)[:80])
    check('W with Start/End around the subject stays silent',
          note(fog_depth='W', fog_start=5.0, fog_end=40.0) is None)
    check('the EXP curves under Z have no Start to miss: no line',
          note(fog_mode='EXP', fog_start=5.0, fog_end=40.0) is None)
    # an orthographic camera: z is linear there and the same trap holds;
    # the advisory used to return before looking at the fog's domain
    sc_o, st_o = build(key)
    sc_o.camera.type = 'ORTHO'
    st_o.fog_start, st_o.fog_end = 5.0, 40.0
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        img_o = R.render(sc_o, st_o)
    sc_o0, st_o0 = build(key)
    sc_o0.camera.type = 'ORTHO'
    st_o0.fog = False
    check('under an orthographic camera the same frame is bitwise unfogged '
          'and the line is printed too',
          bool(np.array_equal(img_o, R.render(sc_o0, st_o0)))
          and 'NOTHING is fogged' in buf.getvalue(), buf.getvalue()[-160:])


TESTS.append(test_f007_zfog_advisory)


def main():
    from . import utf8_console
    utf8_console()
    for fn in TESTS:
        print(f'\n--- {fn.__name__}')
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            check(f'{fn.__name__} raised {type(exc).__name__}: {exc}', False)
            traceback.print_exc()
        finally:
            GSH.FOG_ON_GPU = True
    print(f'\n{len(FAILS)} failure(s)' if FAILS else '\nall ok')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
