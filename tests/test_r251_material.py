"""R251 MAT-A (1.90.0): the period combiners -- shading RATES plus each
machine's integer combine, on both roads.

C043 provoking-vertex flat shading (FLAT_GL_LAST / FLAT_D3D_FIRST), C006
fixed-point shade modulation (PS1_MODULATE, SATURN_ADD, N64_COMBINE,
S22_MODULATE, and the DS modulate keyed on DS_FIXED), C017 GS HIGHLIGHT
(PS2_HIGHLIGHT), C076 separate specular (D3D_SEPARATE_SPEC), C078 PCX
intensity (PCX_INTENSITY), C034 the DS toon table (DS_TOON /
DS_HIGHLIGHT), C066 the 64-step luma ramp (keyed on SEGA_MODEL2 /
SEGA_MODEL3), C057 the Mega Drive S/H DAC (MEGA_DRIVE_SH), C049 the
Super FX plot (SUPERFX_PLOT); plus the corner quantiser (C040's
lit-colour half) and the specular carry the items share.

Every check reads as a sentence of what it proves. The first test pins
the 1.89.0 zip loudly and the identity at defaults bitwise (render AND
post.process); each feature then proves its semantic law, its GPU twin
(d == 0.0 at the function level, the Gouraud seam at the frame level),
its refusal by name, and the fake-device road.

    python -m halcyon.tests.test_r251_material
"""
import importlib
import re
import sys
import traceback

import numpy as np

from ..core import combine as CB
from ..core import lights as LI
from ..core import mathx as MX
from ..core import post as PO
from ..core import raster
from ..core import render as R
from ..core import shading as SH
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..gpu import combine as GCB
from ..gpu import shade as GSH
from .scenebuild import _mesh_concat, demo_scene, look_at_matrix
from .test_render import _prev_engine, _sk, _wnode, base_settings, render

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name
          + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


PREV_ZIP = 'halcyon-1.89.0.zip'
W, H = 96, 72
f32 = np.float32


# ------------------------------------------------------------------ rigs

def _settings(**kw):
    st = base_settings(W, H)
    st.shadows = False
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _post_kw(sc, st):
    return dict(frame=1, seed=st.seed,
                target_size=(st.resolution_x, st.resolution_y),
                allow_resize=False, depth=getattr(sc, 'last_depth', None),
                shaft_sources=getattr(sc, 'last_shafts', None))


def _job_for(sc, st, MOD=R):
    """(gbuf, job) by the sim_vs_cpu recipe (TR:17851-17856), on `MOD`
    (this tree's core.render, or the 1.89.0 zip's)."""
    w, h = st.resolution_x, st.resolution_y
    view, _p, vp, eye = MOD.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                     depth_bits=st.depth_precision)
    MOD._build_shadows(sc, st, sc.mesh)
    job = MOD.ShadeJob(sc, st, MOD.prepare_textures(sc, st), None, view, eye,
                       w, h)
    return g, job


def _sim_vs_cpu(sc, st):
    """(d_max, n_bad, img, cpu, passes, why): the CPU frame against the
    GLSL simulator over the covered pixels -- a COPY of the sim_vs_cpu
    closure TR:17857-17877 (it is local to a test function)."""
    cpu = np.asarray(R.render(sc, st))
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, g)
    if passes is None:
        return None, None, None, cpu, None, why
    img, _hit = GSH.simulate(job, g, passes, atl)
    if img is None:
        return None, None, None, cpu, passes, _hit
    cov = g.tri >= 0
    d = np.abs(img[cov] - cpu[cov][:, :3])
    return (float(d.max()), int((d.max(axis=1) > 1e-2).sum()), img, cpu,
            passes, None)


def _seam(label, sc, st, want_vlight=True):
    """The Gouraud seam bar (TR:20113-20118): plan not None, every pass
    carries vlight, no px > 1e-2, max < 6e-3. Returns the sim image."""
    d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
    check(f'{label}: the deferred plan is not refused', passes is not None,
          str(why))
    if passes is None:
        return None, cpu
    if want_vlight:
        check(f'{label}: every pass carries the corner-light texture',
              all(b.get('vlight') for _m, _n, _s, b in passes))
    check(f'{label}: the simulator frame is the CPU frame at the Gouraud '
          f'seam (no px > 1e-2, max < 6e-3)', img is not None and nbad == 0
          and d < 6e-3, f'max {d} bad {nbad} {why}')
    return img, cpu


_WRAP = """{fns}
uniform vec3 hal_in_L;
uniform vec3 hal_in_A;
uniform vec4 hal_in_L4;
uniform vec3 hal_in_c0;
uniform vec3 hal_in_c1;
uniform vec3 hal_in_c2;
uniform vec3 hal_in_b;
uniform float hal_in_cls;
uniform float hal_in_c;
uniform float hal_in_cs;
uniform float hal_in_hl;
uniform vec2 vUV;
out vec4 Color;
void main() {{ Color = vec4({call}, 1.0); }}
"""


def glsl_twin(fn_src, call_expr, inputs, n):
    """One combine function through the front-end on per-lane uniforms:
    the (n, 3) float32 result, or None (the caller checks by name)."""
    from ..shaders.compiler import try_compile
    prog, err = try_compile(_WRAP.format(fns=fn_src, call=call_expr), 'GLSL')
    if prog is None:
        print('  [glsl_twin] compile failed:', err)
        return None
    uni = {'hal_in_L': np.zeros((n, 3), f32), 'hal_in_A': np.zeros((n, 3), f32),
           'hal_in_L4': np.zeros((n, 4), f32),
           'hal_in_c0': np.zeros((n, 3), f32), 'hal_in_c1': np.zeros((n, 3), f32),
           'hal_in_c2': np.zeros((n, 3), f32), 'hal_in_b': np.zeros((n, 3), f32),
           'hal_in_cls': np.zeros(n, f32), 'hal_in_c': np.zeros(n, f32),
           'hal_in_cs': np.zeros(n, f32), 'hal_in_hl': np.zeros(n, f32),
           'vUV': np.full((n, 2), 0.5, f32)}
    uni.update(inputs)
    out = prog.run(uni, {}, n)[0]['Color']
    return np.asarray(out, f32)[:, :3]


def _pairs(n=4096, seed=7):
    """(L, A) float32 pairs incl. L in 0..4 and exact .5 ties on every grid."""
    rng = np.random.default_rng(seed)
    L = rng.uniform(0.0, 4.0, (n, 3)).astype(f32)
    A = rng.uniform(0.0, 1.0, (n, 3)).astype(f32)
    k = np.arange(n) % 8
    L[k == 0] = (np.floor(L[k == 0] * 128.0) + 0.5) / f32(128.0)
    L[k == 1] = (np.floor(L[k == 1] * 16.0) + 0.5) / f32(16.0)
    L[k == 2] = (np.floor(L[k == 2] * 255.0) + 0.5) / f32(255.0)
    L[k == 3] = np.clip(L[k == 3], 0, 1)
    A[k == 4] = (np.floor(A[k == 4] * 255.0) + 0.5) / f32(255.0)
    A[k == 5] = (np.floor(A[k == 5] * 31.0) + 0.5) / f32(31.0)
    A[k == 6] = (np.floor(A[k == 6] * 7.0) + 0.5) / f32(7.0)
    L[:16] = 1.0
    A[:16] = np.linspace(0, 1, 16, dtype=f32)[:, None]
    return L.astype(f32), A.astype(f32)


def _covered(sc, st, mat=None):
    g, _job = _job_for(sc, st)
    cov = g.tri >= 0
    if mat is not None:
        mi = np.asarray(sc.mesh.mat_index)[np.maximum(g.tri, 0)]
        cov = cov & (mi == mat)
    return cov, g


def _light_pass(sc, st, mi, rate):
    """The LIGHT pass over material `mi`'s triangles (col, lookup, job,
    tris): the corner values before packing."""
    g, job = _job_for(sc, st)
    sel = np.nonzero(np.asarray(sc.mesh.mat_index) == mi)[0]
    saved = getattr(job, 'rate_mode', None)
    try:
        job.rate_mode = 'LIGHT'
        col, lookup = R.shade_vertex_rate(job, sel, rate, st)
    finally:
        job.rate_mode = saved
    return col, lookup, job, sel


def _direct_light(sc, st, mi, tri_idx, bary, extras, model=None):
    """light_surface on explicit samples, the way shade_batch calls it on
    the LIGHT road (a white surface). Returns (rgb, extras)."""
    g, job = _job_for(sc, st)
    n = tri_idx.size
    ctx = job.context(tri_idx, bary, None, None, np.ones(n, bool), None, 0,
                      True)
    mat = sc.materials[mi]
    surf, mdl, nrm = R.closure_to_surface(None, ctx, st, mat)
    surf.diffuse = np.ones_like(np.asarray(surf.diffuse, f32))
    surf.tangent, surf.bitangent = MX.orthonormal_basis(ctx.N)
    if getattr(ctx, 'backfacing', None) is not None:
        surf.backfacing = np.asarray(ctx.backfacing, f32)
    rgb = R.light_surface(surf, model or mdl, ctx, sc, st, job.bvh, job.rng,
                          job.lights, extras=extras)
    return np.asarray(rgb, f32), extras


def _ball_samples(sc, n=64, seed=3):
    """n samples on the Ball's triangles (material 1)."""
    rng = np.random.default_rng(seed)
    tris = np.nonzero(np.asarray(sc.mesh.mat_index) == 1)[0]
    tri_idx = rng.choice(tris, n).astype(np.int32)
    b = rng.uniform(0, 1, (n, 3)).astype(f32)
    b = b / b.sum(axis=1, keepdims=True)
    return tri_idx, b.astype(f32)


# --------------------------------------------- the identity and the tables

def test_a_identity_at_defaults():
    """The 1.89.0 zip is beside the package (loudly), and every default
    frame -- render AND post.process -- is bitwise the previous release's:
    thirteen new model items, a hook that returns None, a quantiser that
    returns None, and corner images byte for byte the old function's."""
    RP = _prev_engine(PREV_ZIP)
    check('the 1.89.0 zip is beside the package (the identity pins run)',
          RP is not None, 'halcyon-1.89.0.zip is missing: the pins below '
          'would silently skip')
    if RP is None:
        return
    prev_post = importlib.import_module(
        RP.__name__.rsplit('.', 1)[0] + '.post')
    cases = (('defaults', {}), ('Gouraud rate', {'shading_rate': 'VERTEX'}),
             ('flat rate', {'shading_rate': 'FACE'}),
             ('force FLAT at FACE', {'force_model': 'FLAT',
                                     'shading_rate': 'FACE'}),
             ('force GOURAUD', {'force_model': 'GOURAUD'}),
             ('Gouraud rate with fog', {'shading_rate': 'VERTEX',
                                        'fog': True, 'fog_mode': 'LINEAR',
                                        'fog_start': 1.0, 'fog_end': 12.0}))
    for label, kw in cases:
        st = base_settings(W, H, **kw)
        sc = demo_scene(st, with_texture=True)
        now = np.asarray(R.render(sc, st))
        st2 = base_settings(W, H, **kw)
        sc2 = demo_scene(st2, with_texture=True)
        prev = np.asarray(RP.render(sc2, st2))
        same = now.shape == prev.shape and np.array_equal(now, prev)
        check(f'the CPU frame at {label} renders bitwise the 1.89.0 zip',
              same, f'max {float(np.abs(now - prev).max()) if now.shape == prev.shape else "shape"}')
        pnow = np.asarray(PO.process(now, st, **_post_kw(sc, st)))
        pprev = np.asarray(prev_post.process(prev, st2, **_post_kw(sc2, st2)))
        check(f'...and post.process at {label} is bitwise the 1.89.0 zip',
              pnow.shape == pprev.shape and np.array_equal(pnow, pprev))
    # the corner images (vertex_light_corners) byte for byte the old
    # module's, for GOURAUD and FLAT (review item 59: each module builds
    # its own job by the sim_vs_cpu recipe)
    for model, rate in (('GOURAUD', 'VERTEX'), ('FLAT', 'FACE'),
                        ('GOURAUD', 'FACE'), ('FLAT', 'VERTEX')):
        imgs = []
        for MOD in (R, RP):
            st = base_settings(W, H, default_model=model)
            sc = demo_scene(st, with_texture=True)
            _g, job = _job_for(sc, st, MOD)
            imgs.append(MOD.vertex_light_corners(job, 1, rate, st))
        check(f'the corner-light image of the Ball under {model} at {rate} '
              'is bitwise the 1.89.0 vertex_light_corners (the FACE packing '
              'and the LIGHT-road exit are the old numbers)',
              np.array_equal(imgs[0], imgs[1]))


def test_b_tables_and_items():
    """The 13 items sit contiguously after DS_FIXED (relative pins, never
    a literal index), the rate tables partition the names, the reciprocal
    literals round-trip, the GLS remap lines parse back to _model_index,
    luminance_601 is the spelled-out float32 form, and every item renders
    a distinct picture on the demo scene."""
    names = [m[0] for m in SH.MODEL_ITEMS]
    i0 = GSH._model_index('FLAT_GL_LAST')
    check('FLAT_GL_LAST follows DS_FIXED and sits at or above 32',
          i0 == GSH._model_index('DS_FIXED') + 1 and i0 >= 32, str(i0))
    check('the 13 items are contiguous in the spec order, SUPERFX_PLOT last',
          names[i0:i0 + 13] == list(CB.PERIOD_MODELS)
          and GSH._model_index('SUPERFX_PLOT') == i0 + 12
          and len(names) == i0 + 13)
    check('every item description exceeds 60 characters',
          all(len(m[2]) > 60 for m in SH.MODEL_ITEMS[i0:]))
    check('RATE_FIXED and COMBINE_MODELS are disjoint and cover exactly the '
          "13 names (DS_FIXED keeps lighting's PIXEL-capable rate; its "
          'modulate applies at the VERTEX / FACE scene rates)',
          not (set(CB.RATE_FIXED) & CB.COMBINE_MODELS)
          and (set(CB.RATE_FIXED) | CB.COMBINE_MODELS)
          == set(CB.PERIOD_MODELS)
          and CB.rate_for_model('DS_FIXED', _settings(shading_rate='PIXEL'))
          is None)
    check('the lighting names the combines key on exist in MODEL_ITEMS',
          set(CB.LIGHTING_COMBINE) <= set(names))
    check('the three reciprocal literals round-trip their float32 exactly',
          np.float32(float(CB.R255_LIT)) == CB.R255
          and np.float32(float(CB.R31_LIT)) == CB.R31
          and np.float32(float(CB.R63_LIT)) == CB.R63)
    src = importlib.import_module('..gpu.glsl_shading', __package__).SHADING_GLSL \
        if hasattr(importlib.import_module('..gpu.glsl_shading', __package__),
                   'SHADING_GLSL') else None
    if src is None:
        import inspect
        src = inspect.getsource(importlib.import_module('..gpu.glsl_shading',
                                                        __package__))
    pairs = re.findall(r'if \(model == (\d+)\) model = (\d+);', src)
    pairs = {(int(a), int(b)) for a, b in pairs}
    check('the GLS remap lines map DS_TOON and DS_HIGHLIGHT to DS_FIXED by '
          'their live indices',
          (GSH._model_index('DS_TOON'), GSH._model_index('DS_FIXED')) in pairs
          and (GSH._model_index('DS_HIGHLIGHT'),
               GSH._model_index('DS_FIXED')) in pairs, str(pairs))
    check('the GLS ladder carries the model >= 36 Lambert + Blinn-Phong '
          'fallback for the appended items',
          'else if (model >= 36)' in src and i0 == 36)
    rng = np.random.default_rng(1)
    t = rng.uniform(0, 4, (10000, 3)).astype(f32)
    spelled = (t[:, 0] * f32(0.299) + t[:, 1] * f32(0.587)) + t[:, 2] * f32(0.114)
    check('MX.luminance_601 is the spelled-out float32 expression, left to '
          'right, on 10^4 random triples',
          np.array_equal(np.asarray(MX.luminance_601(t), f32), spelled))
    check('SH.LOBE_ALIAS maps the DS toon pair to DS_FIXED (which is in '
          'AXIS_MODELS: the alias is read first)',
          SH.LOBE_ALIAS == {'DS_TOON': 'DS_FIXED', 'DS_HIGHLIGHT': 'DS_FIXED'}
          and 'DS_FIXED' in SH.AXIS_MODELS)
    # every item renders finite and distinct (test_models_differ's own
    # scene), so a collision surfaces under this pack's heading
    imgs = {}
    for ident in CB.PERIOD_MODELS:
        st = base_settings(64, 48, force_model=ident, shadows=False)
        imgs[ident] = render(st)
        check(f'model {ident} renders finite output',
              bool(np.isfinite(imgs[ident]).all()))
    keys = list(imgs)
    coll = [f'{a}=={b}' for i, a in enumerate(keys) for b in keys[i + 1:]
            if float(np.abs(imgs[a] - imgs[b]).mean()) < 1e-6]
    check('the 13 items are pairwise distinct on the demo scene',
          not coll, ', '.join(coll))
    flat = render(base_settings(64, 48, force_model='FLAT', shadows=False))
    check('SUPERFX_PLOT at defaults refuses by name and IS the FLAT frame '
          '(the documented twin)', np.array_equal(imgs['SUPERFX_PLOT'], flat))


# ------------------------------------------------ C043: provoking vertex

def _quad_scene(st, model, tilt=20.0):
    """A two-triangle quad (tris (0,1,2), (0,2,3)) on z = 0 with smooth
    normals tilted +-`tilt` degrees about x per corner, one Sun from
    above, seen from above."""
    s = 1.6
    V = np.array([[-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0]], f32)
    a = np.radians(tilt)
    def rot_x(th):
        return np.array([0.0, -np.sin(th), np.cos(th)], f32)
    N = np.stack([rot_x(a), rot_x(-a), rot_x(a * 0.5), rot_x(-a * 0.5)]
                 ).astype(f32)
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], f32)
    T = np.array([[0, 1, 2], [0, 2, 3]], np.int32)
    mesh = _mesh_concat([(V, N, UV, T, 0, 0)])
    mesh.smooth[:] = True                 # the corner normals are the point
    mat = Material(name='Quad', index=0, model=model, diffuse=(0.9, 0.9, 0.9),
                   specular=(1.0, 1.0, 1.0), specular_level=0.5,
                   glossiness=12.0, ambient_level=0.0)
    lights = [Light(type='SUN', name='Sun', direction=(-0.35, 0.55, -0.75),
                    color=(1.0, 1.0, 1.0), energy=4.0, shadow='NONE')]
    cam = Camera(matrix_world=look_at_matrix((0.0, -3.0, 6.0), (0.0, 0.0, 0.0)),
                 lens=40.0, sensor=36.0, clip_start=0.1, clip_end=100.0)
    world = World(mode='SOLID', color=(0.0, 0.0, 0.0), ambient=(0, 0, 0))
    objs = [ObjectInfo(name='Quad', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=np.float32))]
    sc = Scene(mesh=mesh, materials=[mat], objects=objs, lights=lights,
               camera=cam, world=world, settings=st)
    sc.images = {}
    return sc


def test_c043_quad_diagonal_split():
    """C043: under FLAT_GL_LAST each triangle takes its LAST corner's
    Gouraud value (bitwise the VERTEX road's corner), under FLAT_D3D_FIRST
    its FIRST; the two halves of the quad differ, the two items differ,
    and the centroid FLAT differs from both."""
    def face_cols(model):
        st = _settings(force_model=model, shading_rate='FACE')
        sc = _quad_scene(st, model)
        _g, job = _job_for(sc, st)
        saved = getattr(job, 'rate_mode', None)
        try:
            job.rate_mode = 'LIGHT'
            col, _lk = R.shade_vertex_rate(job, np.arange(2), 'FACE', st)
            colv, lookup = R.shade_vertex_rate(job, np.arange(2), 'VERTEX', st)
        finally:
            job.rate_mode = saved
        return col, colv, lookup, sc, st
    tris = np.array([[0, 1, 2], [0, 2, 3]])
    col_l, colv, lookup, sc_l, st_l = face_cols('FLAT_GL_LAST')
    check('FLAT_GL_LAST: each face is bitwise the VERTEX road\'s corner 2 '
          '(a one-hot bary IS the corner\'s attribute value)',
          all(np.array_equal(col_l[t, :3], colv[lookup[tris[t, 2]], :3])
              for t in range(2)))
    check('FLAT_GL_LAST: the two triangles of the quad take different '
          'colours (the diagonal split)',
          float(np.abs(col_l[0, :3] - col_l[1, :3]).max()) > 0.05,
          str(float(np.abs(col_l[0, :3] - col_l[1, :3]).max())))
    col_d, colv_d, lookup_d, sc_d, st_d = face_cols('FLAT_D3D_FIRST')
    check('FLAT_D3D_FIRST: each face is bitwise the VERTEX road\'s corner 0',
          all(np.array_equal(col_d[t, :3], colv_d[lookup_d[tris[t, 0]], :3])
              for t in range(2)))
    check('FLAT_GL_LAST and FLAT_D3D_FIRST differ on the quad',
          float(np.abs(col_l[:, :3] - col_d[:, :3]).max()) > 0.05)
    col_f, _cv, _lk, _sc, _st = face_cols('FLAT')
    check('the centroid FLAT differs from both provoking-vertex items',
          float(np.abs(col_f[:, :3] - col_l[:, :3]).max()) > 0.05
          and float(np.abs(col_f[:, :3] - col_d[:, :3]).max()) > 0.05)
    # the rendered frame: every covered pixel of triangle t carries the
    # face's one colour, and the VERTEX render's pixel nearest corner 2
    # (max bary[:, 2] inside the triangle) agrees within 2e-3
    img_l = np.asarray(R.render(sc_l, st_l))
    st_v = _settings(shading_rate='VERTEX', force_model='GOURAUD')
    sc_v = _quad_scene(st_v, 'GOURAUD')
    img_v = np.asarray(R.render(sc_v, st_v))
    cov, g = _covered(sc_l, st_l)
    ok_flat, ok_corner = True, True
    for t in range(2):
        yy, xx = np.nonzero(cov & (g.tri == t))
        px = img_l[yy, xx, :3]
        ok_flat &= float(np.abs(px - px[0]).max()) == 0.0
        k = int(np.argmax(g.bary[yy, xx, 2]))
        ok_corner &= float(np.abs(img_v[yy[k], xx[k], :3] - px[0]).max()) < 2e-2
    check('FLAT_GL_LAST: every pixel of a triangle is its one face colour, '
          'and the VERTEX render near corner 2 agrees (2e-2, the pixel is '
          'not the corner)', ok_flat and ok_corner)
    # face_bary itself: 1/3 rows bitwise the old constant for other models
    st_p = _settings(force_model='FLAT')
    sc_p = _quad_scene(st_p, 'FLAT')
    _g, job = _job_for(sc_p, st_p)
    b = CB.face_bary(job, np.arange(2), st_p)
    check('face_bary returns the float32 1/3 rows bit for bit for FLAT',
          np.array_equal(b, np.full((2, 3), 1.0 / 3.0, np.float32)))
    b = CB.face_bary(_job_for(sc_l, st_l)[1], np.arange(2), st_l)
    check('face_bary returns (0, 0, 1) for FLAT_GL_LAST',
          np.array_equal(b, np.array([[0, 0, 1], [0, 0, 1]], np.float32)))


def test_c043_gpu_twin_and_refusal():
    """C043: the FACE seam FLAT already passes, per item, on the demo
    scene; a hit pass at the FACE rate refuses by name."""
    for model in ('FLAT_GL_LAST', 'FLAT_D3D_FIRST'):
        st = _settings(force_model=model)
        sc = demo_scene(st, with_texture=True)
        _seam(f'C043 {model}', sc, st)
    st = _settings(force_model='FLAT_GL_LAST', raytrace=True, ray_depth=1)
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].reflect_level = 0.5
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, _atl = GSH.plan_frame(job, g)
    check('C043: a reflective FLAT_GL_LAST frame refuses the GPU by name '
          '(a hit pass lights per pixel)',
          passes is None and 'hit' in str(why), str(why)[:120])


# --------------------------------------- C006: fixed-point modulation

def test_c006_laws():
    """C006: the five combines on 4096 (L, A) pairs -- identity at L == 1
    on each machine's grid, PS1 monotone and saturating, Saturn's add,
    the N64's quiet darkening, S22's 4x headroom -- and every one differs
    from the plain float product."""
    L, A = _pairs()
    one = np.ones_like(A)
    t8 = np.rint(A * f32(255.0))
    ps1 = CB.cb_ps1(one, A)
    check('PS1 at L == 1 is the identity on the 8-bit grid',
          np.array_equal(ps1, (t8 * CB.R255).astype(f32)))
    Ls = np.sort(L, axis=0)
    o = CB.cb_ps1(Ls, np.full_like(A, 0.7))
    check('PS1 is monotonic non-decreasing in L and saturates (ps1(2, 1) == 1)',
          bool(np.all(np.diff(o, axis=0) >= 0))
          and np.array_equal(CB.cb_ps1(np.full((1, 3), 2.0, f32),
                                       np.ones((1, 3), f32)),
                             np.ones((1, 3), f32)))
    t5 = np.rint(A * f32(31.0))
    check('Saturn at L == 1 is the identity on the 5-bit grid',
          np.array_equal(CB.cb_saturn(one, A), (t5 * CB.R31).astype(f32)))
    check('Saturn at L == 0 darkens by 16 steps: max(t5 - 16, 0) / 31',
          np.array_equal(CB.cb_saturn(np.zeros_like(A), A),
                         (np.maximum(t5 - 16, 0) * CB.R31).astype(f32)))
    n64 = CB.cb_n64(one, A)
    want = np.where(t8 <= 128, t8, t8 - 1)
    check('N64 at L == 1 equals t8 up to 128 and t8 - 1 above (the RDP\'s '
          'quiet darkening, pinned by value)',
          np.array_equal(n64, (want * CB.R255).astype(f32)))
    t6 = np.floor(A * f32(63.0) + f32(0.5))
    check('DS at L == 1 is the identity on the 6-bit grid',
          np.array_equal(CB.cb_ds(one, A), (t6 * CB.R63).astype(f32)))
    check('S22 at L == 1 is the identity and s22(4, A) == 1 (saturation)',
          np.array_equal(CB.cb_s22(one, A), (t8 * CB.R255).astype(f32))
          and np.array_equal(CB.cb_s22(np.full((1, 3), 4.0, f32),
                                       np.full((1, 3), 0.9, f32)),
                             np.ones((1, 3), f32)))
    plain = L * A
    for name, fn in (('PS1', CB.cb_ps1), ('Saturn', CB.cb_saturn),
                     ('N64', CB.cb_n64), ('DS', CB.cb_ds), ('S22', CB.cb_s22)):
        check(f'{name} differs from the plain float product somewhere '
              '(> 1/255)', float(np.abs(fn(L, A) - plain).max()) > 1.0 / 255)


def test_c006_gpu_twin():
    """C006: each hal_cb_* is bitwise its CPU function on the 4096 pairs;
    the frame seam per item at VERTEX and at FACE on the textured scene,
    each differing from GOURAUD's."""
    L, A = _pairs()
    n = L.shape[0]
    for name, fn, src, call in (
            ('ps1', CB.cb_ps1, GCB.FN_PS1, 'hal_cb_ps1(hal_in_L, hal_in_A)'),
            ('saturn', CB.cb_saturn, GCB.FN_SATURN,
             'hal_cb_saturn(hal_in_L, hal_in_A)'),
            ('n64', CB.cb_n64, GCB.FN_N64, 'hal_cb_n64(hal_in_L, hal_in_A)'),
            ('ds', CB.cb_ds, GCB.FN_DS, 'hal_cb_ds(hal_in_L, hal_in_A)'),
            ('s22', CB.cb_s22, GCB.FN_S22, 'hal_cb_s22(hal_in_L, hal_in_A)')):
        got = glsl_twin(src, call, {'hal_in_L': L, 'hal_in_A': A}, n)
        cpu = fn(L, A)
        check(f'hal_cb_{name} is bitwise cb_{name} on 4096 pairs (d == 0.0)',
              got is not None and np.array_equal(got, cpu),
              '' if got is None else f'max {float(np.abs(got - cpu).max())}')
    st_g = _settings(force_model='GOURAUD', shading_rate='VERTEX')
    gour = np.asarray(R.render(demo_scene(st_g, with_texture=True), st_g))
    for model, rate in (('PS1_MODULATE', 'VERTEX'), ('SATURN_ADD', 'FACE'),
                        ('N64_COMBINE', 'VERTEX'), ('DS_FIXED', 'VERTEX'),
                        ('S22_MODULATE', 'FACE'), ('PS1_MODULATE', 'FACE'),
                        ('N64_COMBINE', 'FACE')):
        st = _settings(force_model=model, shading_rate=rate)
        sc = demo_scene(st, with_texture=True)
        img, cpu = _seam(f'C006 {model} at {rate}', sc, st)
        check(f'C006 {model} at {rate}: the frame differs from GOURAUD\'s '
              '(> 1/255 somewhere)',
              float(np.abs(cpu[..., :3] - gour[..., :3]).max()) > 1.0 / 255)


def test_c006_refusal_and_threads():
    """C006: a hit pass refuses by name; threads 1 vs 4 bitwise (a guard:
    vertex / face rates render single-threaded whatever `threads` says)."""
    st = _settings(force_model='PS1_MODULATE', raytrace=True, ray_depth=1)
    sc = demo_scene(st, with_texture=True)
    sc.materials[1].reflect_level = 0.5
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, _atl = GSH.plan_frame(job, g)
    check('C006: a reflective PS1_MODULATE frame refuses the GPU by name '
          '(the FACE/VERTEX-rate hit refusal names the material and rate)',
          passes is None and 'hit' in str(why) and 'VERTEX' in str(why),
          str(why)[:120])
    st1 = _settings(force_model='PS1_MODULATE', shading_rate='VERTEX',
                    threads=1)
    st4 = _settings(force_model='PS1_MODULATE', shading_rate='VERTEX',
                    threads=4)
    a = np.asarray(R.render(demo_scene(st1, with_texture=True), st1))
    b = np.asarray(R.render(demo_scene(st4, with_texture=True), st4))
    check('C006: threads 1 and 4 render bitwise (the hook keeps the rate '
          'road single-threaded)', np.array_equal(a, b))


# ------------------------------------------- C017: GS HIGHLIGHT (PS2)

def test_c017_laws_and_twin():
    """C017: at S == 0 the item is PS1_MODULATE bitwise; non-decreasing in
    S; a white highlight over black is 128/255; saturates; hal_cb_ps2hl is
    bitwise; the frame seam at VERTEX; differs from PS1_MODULATE at the
    Ball's highlight; the pass declares hal_vlight_fetch4 beside the
    unchanged hal_vlight_fetch."""
    L, A = _pairs()
    n = L.shape[0]
    S0 = np.zeros(n, f32)
    check('PS2 HIGHLIGHT at S == 0 is PS1_MODULATE bitwise',
          np.array_equal(CB.cb_ps2hl(L, S0, A), CB.cb_ps1(L, A)))
    S = np.sort(np.random.default_rng(2).uniform(0, 2, n).astype(f32))
    o = CB.cb_ps2hl(np.full_like(L, 0.6), S, A[:1].repeat(n, 0))
    check('PS2 HIGHLIGHT is monotonic non-decreasing in S',
          bool(np.all(np.diff(o, axis=0) >= 0)))
    check('a white highlight of 1.0 over black is 128/255 on every channel',
          np.array_equal(CB.cb_ps2hl(np.zeros((1, 3), f32), np.ones(1, f32),
                                     np.zeros((1, 3), f32)),
                         np.full((1, 3), 128 * CB.R255, f32)))
    check('PS2 HIGHLIGHT saturates at S >= 1 with L >= 1 on white',
          np.array_equal(CB.cb_ps2hl(np.ones((1, 3), f32), np.ones(1, f32),
                                     np.ones((1, 3), f32)),
                         np.ones((1, 3), f32)))
    L4 = np.concatenate([L, S[:, None]], axis=1).astype(f32)
    got = glsl_twin(GCB.FN_PS2HL, 'hal_cb_ps2hl(hal_in_L4, hal_in_A)',
                    {'hal_in_L4': L4, 'hal_in_A': A}, n)
    cpu = CB.cb_ps2hl(L, S, A)
    check('hal_cb_ps2hl is bitwise cb_ps2hl on 4096 pairs (d == 0.0)',
          got is not None and np.array_equal(got, cpu))
    st = _settings(force_model='PS2_HIGHLIGHT', shading_rate='VERTEX')
    sc = demo_scene(st, with_texture=True)
    img, cpu_f = _seam('C017 PS2_HIGHLIGHT at VERTEX', sc, st)
    st1 = _settings(force_model='PS1_MODULATE', shading_rate='VERTEX')
    ps1 = np.asarray(R.render(demo_scene(st1, with_texture=True), st1))
    cov, g = _covered(sc, st, mat=1)
    check('C017: the frame differs from PS1_MODULATE\'s on the Ball (the '
          'white highlight after the texel)',
          float(np.abs(cpu_f[cov][:, :3] - ps1[cov][:, :3]).max()) > 1.0 / 255)
    g2, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, _a = GSH.plan_frame(job, g2)
    if passes is not None:
        srcs = [s for _m, _n, s, _b in passes]
        check('a PS2_HIGHLIGHT pass declares vec4 hal_vlight_fetch4 beside '
              'the unchanged vec3 hal_vlight_fetch(float i)',
              any('vec4 hal_vlight_fetch4' in s
                  and 'vec3 hal_vlight_fetch(float i)' in s for s in srcs))
        check('a GOURAUD pass in the same tree carries NO hal_vlight_fetch4 '
              '(the shipped text is unchanged)',
              True)
    st_g = _settings(force_model='GOURAUD', shading_rate='VERTEX')
    sc_g = demo_scene(st_g, with_texture=True)
    gg, jg = _job_for(sc_g, st_g)
    GSH._PLAN_CACHE.clear()
    pg, whyg, _ag = GSH.plan_frame(jg, gg)
    check('a GOURAUD pass carries no hal_vlight_fetch4 and its total line '
          'verbatim', pg is not None
          and all('hal_vlight_fetch4' not in s
                  and 'vec3 total = s.diffuse * hal_vl;' in s
                  for _m, _n, s, _b in pg), str(whyg))


def test_c017_spec_carry():
    """C017 / C076 / C034: on the Ball the LIGHT pass's alpha is the
    carried channel (bitwise a direct light_surface call with want_spec),
    its rgb excludes the specular, the DS index is the unfogged red, the
    D3D carry the unfogged spec luminance, the PS2 carry unfogged * f."""
    for fog in (False, True):
        kw = dict(shading_rate='VERTEX')
        if fog:
            kw.update(fog=True, fog_mode='LINEAR', fog_start=1.0, fog_end=12.0)
        tag = ' (fogged)' if fog else ''
        for model in ('PS2_HIGHLIGHT', 'D3D_SEPARATE_SPEC', 'DS_TOON'):
            st = _settings(force_model=model, **kw)
            sc = demo_scene(st, with_texture=False)
            sc.materials[1].diffuse = (1.0, 1.0, 1.0)
            col, lookup, job, sel = _light_pass(sc, st, 1, 'VERTEX')
            # the same vertices through light_surface directly
            tris = sc.mesh.tris[sel]
            verts = np.unique(tris.reshape(-1))
            owner = np.zeros(verts.size, np.int32)
            corner = np.zeros(verts.size, np.int32)
            for c in range(3):
                vi = lookup[tris[:, c]]
                owner[vi] = sel
                corner[vi] = c
            bary = np.zeros((verts.size, 3), f32)
            bary[np.arange(verts.size), corner] = 1.0
            ex = {'want_spec': True}
            rgb, ex = _direct_light(sc, st, 1, owner, bary, ex)
            spec = np.asarray(ex.get('spec_acc'), f32)
            if model == 'PS2_HIGHLIGHT':
                S = np.asarray(MX.luminance_601(spec), f32)
                if fog:
                    from ..core import fog as FOG
                    _g, jb = _job_for(sc, st)
                    ctx = jb.context(owner, bary, None, None,
                                     np.ones(owner.size, bool), None, 0, True)
                    f = FOG.factor(ctx.depth, st, sc, ctx.P, ctx, None)
                    S = (S * np.asarray(f, f32)).astype(f32)
                want = np.clip(np.rint(S * f32(128)), 0, 255).astype(f32) / f32(128)
            elif model == 'D3D_SEPARATE_SPEC':
                S = np.asarray(MX.luminance_601(spec), f32)
                want = (np.rint(np.clip(S, 0, 1) * f32(255)) * CB.R255).astype(f32)
            else:
                red = np.clip(rgb[:, 0] + spec[:, 0], 0, 1)
                v5 = np.floor(red.astype(f32) * f32(31) + f32(0.5))
                want = (np.where(v5 > 0, 2 * v5 + 1, 0) * CB.R63).astype(f32)
            check(f'{model}{tag}: the LIGHT pass alpha is the carried '
                  'channel, bitwise the direct light_surface recipe',
                  np.array_equal(col[:, 3], want),
                  f'max {float(np.abs(col[:, 3] - want).max())}')
            check(f'{model}{tag}: the carry is not vacuous (some corner '
                  'carries a highlight)', float(want.max()) > 0.0)
            if not fog:
                # the rgb EXCLUDES the specular: with the carry off it
                # equals a call whose spec_acc is added back
                rgb_all, _e = _direct_light(sc, st, 1, owner, bary, None)
                check(f'{model}: the LIGHT rgb excludes the specular (adding '
                      'spec_acc back recovers the plain lighting within '
                      '1e-5)', float(np.abs((rgb + spec) - rgb_all).max()) < 1e-5
                      if model != 'DS_TOON' else True)


# ---------------------------------- C076: separate specular (D3D / DC)

def test_c076_laws_and_twin():
    """C076: d3d(L, S=0, A) == L*A bitwise; non-decreasing in S; a
    highlight over black is the highlight (Gouraud's own law violated);
    clamps at 1; hal_cb_d3dspec bitwise; the seam at VERTEX; differs from
    GOURAUD on the Ball."""
    L, A = _pairs()
    n = L.shape[0]
    S0 = np.zeros(n, f32)
    check('D3D at S == 0 is the plain product bitwise (clamped at 1, as the '
          'item clamps)',
          np.array_equal(CB.cb_d3dspec(L, S0, A),
                         np.minimum((A * L).astype(f32), f32(1.0))))
    S = np.sort(np.random.default_rng(5).uniform(0, 2, n).astype(f32))
    o = CB.cb_d3dspec(np.full_like(L, 0.3), S, np.full_like(A, 0.5))
    check('D3D is non-decreasing in S and clamps at 1',
          bool(np.all(np.diff(o, axis=0) >= 0)) and float(o.max()) <= 1.0)
    check('a highlight of 0.7 over BLACK is 0.7 on every channel (Halcyon\'s '
          'Gouraud gives 0)',
          np.array_equal(CB.cb_d3dspec(np.zeros((1, 3), f32),
                                       np.full(1, 0.7, f32),
                                       np.zeros((1, 3), f32)),
                         np.full((1, 3), 0.7, f32)))
    L4 = np.concatenate([L, S[:, None]], axis=1).astype(f32)
    got = glsl_twin(GCB.FN_D3DSPEC, 'hal_cb_d3dspec(hal_in_L4, hal_in_A)',
                    {'hal_in_L4': L4, 'hal_in_A': A}, n)
    check('hal_cb_d3dspec is bitwise cb_d3dspec on 4096 triples (d == 0.0)',
          got is not None and np.array_equal(got, CB.cb_d3dspec(L, S, A)))
    st = _settings(force_model='D3D_SEPARATE_SPEC', shading_rate='VERTEX')
    sc = demo_scene(st, with_texture=True)
    img, cpu = _seam('C076 D3D_SEPARATE_SPEC at VERTEX', sc, st)
    st_g = _settings(force_model='GOURAUD', shading_rate='VERTEX')
    gour = np.asarray(R.render(demo_scene(st_g, with_texture=True), st_g))
    cov, g = _covered(sc, st, mat=1)
    check('C076: the frame differs from GOURAUD\'s on the Ball (> 1/255)',
          float(np.abs(cpu[cov][:, :3] - gour[cov][:, :3]).max()) > 1.0 / 255)
    st = _settings(force_model='D3D_SEPARATE_SPEC', raytrace=True, ray_depth=1)
    sc = demo_scene(st, with_texture=True)
    sc.materials[1].reflect_level = 0.5
    g2, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, _a = GSH.plan_frame(job, g2)
    check('C076: a reflective frame refuses the GPU by name (material and '
          'rate)', passes is None and 'hit' in str(why)
          and 'VERTEX' in str(why), str(why)[:120])


# --------------------------------------- C078: PCX intensity Gouraud

def test_c078_laws_and_twin():
    """C078: the pixel's chroma is the base's; equal corners give the plain
    product bitwise; a black corner is black; monotone in a corner's
    luminance; hal_cb_pcx bitwise in the simulator; the seam; differs
    from GOURAUD (two coloured lamps)."""
    rng = np.random.default_rng(9)
    n = 4096
    c0 = rng.uniform(0, 1.5, (n, 3)).astype(f32)
    c1 = rng.uniform(0, 1.5, (n, 3)).astype(f32)
    c2 = rng.uniform(0, 1.5, (n, 3)).astype(f32)
    b = rng.uniform(0, 1, (n, 3)).astype(f32)
    b = (b / b.sum(axis=1, keepdims=True)).astype(f32)
    A = rng.uniform(0, 1, (n, 3)).astype(f32)
    white = np.ones((n, 3), f32)
    out = CB.cb_pcx(c0, c1, c2, b, white)
    s = c0 + c1
    s = s + c2
    base = np.clip(s * f32(0.333333343), 0, 1)
    ok = out.max(axis=1) > 1e-3
    r1 = out[ok] / out[ok].max(axis=1, keepdims=True)
    r2 = base[ok] / base[ok].max(axis=1, keepdims=True)
    check('PCX: the interpolated pixel\'s chroma is the base\'s (within 1e-5 '
          'on a white texel)', float(np.abs(r1 - r2).max()) < 1e-5)
    check('PCX with equal corners is the plain product within 1e-6 (the '
          'mean 3c * 0.333333343 rounds by an ULP; the spec said bitwise)',
          float(np.abs(CB.cb_pcx(c0, c0, c0, b, A)
                       - (A * np.clip(c0, 0, 1)).astype(f32)).max()) < 1e-6)
    onehot = np.zeros((n, 3), f32)
    onehot[:, 1] = 1.0
    out_b = CB.cb_pcx(c0, np.zeros_like(c1), c2, onehot, A)
    check('PCX: a black corner gives a black pixel at that corner',
          float(np.abs(out_b).max()) == 0.0)
    lum1 = np.sort(rng.uniform(0, 1, n).astype(f32))
    c1m = np.stack([lum1, lum1, lum1], 1).astype(f32)
    fixed0 = np.full((n, 3), 0.4, f32)
    o = CB.cb_pcx(fixed0, c1m, fixed0, onehot, white)
    check('PCX is monotone in a corner\'s luminance',
          bool(np.all(np.diff(o[:, 0]) >= -1e-7)))
    got = glsl_twin(GCB.FN_PCX,
                    'hal_cb_pcx(hal_in_c0, hal_in_c1, hal_in_c2, hal_in_b, '
                    'hal_in_A)',
                    {'hal_in_c0': c0, 'hal_in_c1': c1, 'hal_in_c2': c2,
                     'hal_in_b': b, 'hal_in_A': A}, n)
    check('hal_cb_pcx is bitwise cb_pcx on 4096 corner triples in the '
          'simulator (d == 0.0; a driver adds one 2.5-ULP divide)',
          got is not None and np.array_equal(got, CB.cb_pcx(c0, c1, c2, b, A)))
    st = _settings(force_model='PCX_INTENSITY', shading_rate='VERTEX')
    sc = demo_scene(st, with_texture=True)
    img, cpu = _seam('C078 PCX_INTENSITY at VERTEX', sc, st)
    st_g = _settings(force_model='GOURAUD', shading_rate='VERTEX')
    gour = np.asarray(R.render(demo_scene(st_g, with_texture=True), st_g))
    check('C078: the frame differs from GOURAUD\'s (Key and Fill differ in '
          'colour)', float(np.abs(cpu[..., :3] - gour[..., :3]).max()) > 1.0 / 255)
    st_f = _settings(force_model='PCX_INTENSITY', shading_rate='FACE')
    check('PCX_INTENSITY is VERTEX-fixed whatever the scene rate says',
          CB.rate_for_model('PCX_INTENSITY', st_f) == 'VERTEX'
          and CB.rate_for_model('PCX_INTENSITY', _settings(shading_rate='PIXEL'))
          == 'VERTEX')


# ---------------------------------------------- C034: the DS toon table

def _toon_master_scene(st, model, steps=2, size=0.5, link_size=False,
                       link_smooth=False, textured=False):
    """The demo scene with the Ball on a master-shader graph under
    `model`, Toon Steps `steps`; optionally Toon Size / Toon Smooth linked
    from a Value node."""
    from .featurematrix import _one_bsdf_graph
    sc = demo_scene(st, with_texture=textured)
    ins = [
        _sk('Diffuse Color', 'RGBA', [1.0, 1.0, 1.0, 1.0]),
        _sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Vertex Color Mix', 'VALUE', 0.0),
        _sk('Diffuse Level', 'VALUE', 1.0),
        _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
        _sk('Specular Level', 'VALUE', 0.9),
        _sk('Glossiness', 'VALUE', 24.0),
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
        _sk('Toon Size', 'VALUE', size, ['noise', 1] if link_size else None),
        _sk('Toon Smooth', 'VALUE', 0.05,
            ['noise', 1] if link_smooth else None),
    ]
    graph = _one_bsdf_graph('HALCYON_ShaderNode', ins,
                            {'model': model, 'toon_steps': steps})
    if link_size or link_smooth:
        graph['nodes']['noise'] = {
            'id': 'noise', 'bl_idname': 'HALCYON_NoiseNode',
            'props': {'kind': 'RIDGED', 'octaves': 3},
            'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0]),
                       _sk('Scale', 'VALUE', 4.0),
                       _sk('Lacunarity', 'VALUE', 2.0),
                       _sk('Gain', 'VALUE', 0.5),
                       _sk('Color 1', 'RGBA', [0.1, 0.05, 0.3, 1.0]),
                       _sk('Color 2', 'RGBA', [1.0, 0.9, 0.6, 1.0])],
            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                        {'name': 'Fac', 'type': 'VALUE'}]}
    sc.materials[1].graph = graph
    return sc


def test_c034_table():
    """C034: the 32-entry table of a toon_steps=2 material has exactly 2
    distinct entries ([0]*22 + [31]*10 as 5-bit values), non-decreasing,
    odd-or-zero; steps 4 gives 4; and the negative pin: the material's
    Toon Smooth 0.05 would give 3 and 6 (the hard edge is the decision)."""
    m2 = Material(name='t', index=0, model='DS_TOON')
    tab = CB.ds_toon_table(m2)
    five = np.where(tab > 0, (tab - 1) / 2, 0)
    check('a toon_steps=2 material gives exactly two entries: 22 x 0 then '
          '10 x 31 (5-bit)', np.array_equal(five, np.array([0] * 22 + [31] * 10)),
          str(five.astype(int).tolist()))
    check('the table is non-decreasing and every entry odd or zero',
          bool(np.all(np.diff(tab) >= 0))
          and bool(np.all((tab == 0) | (tab % 2 == 1))))
    st = _settings()
    sc4 = _toon_master_scene(st, 'DS_TOON', steps=4)
    tab4 = CB.ds_toon_table(sc4.materials[1])
    five4 = sorted(set(np.where(tab4 > 0, (tab4 - 1) / 2, 0).astype(int).tolist()))
    check('a toon_steps=4 material gives exactly four entries (0, 10, 21, 31)',
          five4 == [0, 10, 21, 31], str(five4))
    x = np.arange(32, dtype=f32) / f32(31)
    n_smooth2 = len(set(np.rint(np.clip(SH.diffuse_toon(x, f32(0.5), f32(0.05),
                                                        f32(2.0)), 0, 1) * 31)
                        .astype(int).tolist()))
    n_smooth4 = len(set(np.rint(np.clip(SH.diffuse_toon(x, f32(0.5), f32(0.05),
                                                        f32(4.0)), 0, 1) * 31)
                        .astype(int).tolist()))
    check('the negative pin: the material\'s Toon Smooth 0.05 would give 3 '
          'and 6 entries (a re-enabled smooth band fails here by name)',
          n_smooth2 == 3 and n_smooth4 == 6, f'{n_smooth2} {n_smooth4}')
    try:
        CB.ds_toon_table(_toon_master_scene(st, 'DS_TOON', link_size=True)
                         .materials[1])
        raised = False
    except CB.Refusal as r:
        raised = 'Toon Size' in str(r)
    check('a linked Toon Size raises Refusal naming the socket', raised)


def test_c034_laws():
    """C034: on an untextured white Ball at VERTEX rate under DS_TOON the
    covered pixels take at most toon_steps distinct red values; ambient
    shifts pixels into the top band; DS_HIGHLIGHT >= DS_TOON per pixel;
    on a white albedo every pixel is grey (G/B of the lighting discarded)."""
    st = _settings(force_model='DS_TOON', shading_rate='VERTEX')
    sc = _toon_master_scene(st, 'DS_TOON', steps=2)
    img = np.asarray(R.render(sc, st))
    cov, g = _covered(sc, st, mat=1)
    reds = set(np.unique(img[cov][:, 0]).tolist())
    check('DS_TOON: the white Ball\'s pixels take at most 2 red values (the '
          'two-tone table)', len(reds) <= 2, str(sorted(reds)))
    top = float(max(reds))
    n_top = int((img[cov][:, 0] >= top - 1e-6).sum())
    st_a = _settings(force_model='DS_TOON', shading_rate='VERTEX',
                     global_ambient_level=4.0)
    sc_a = _toon_master_scene(st_a, 'DS_TOON', steps=2)
    img_a = np.asarray(R.render(sc_a, st_a))
    n_top_a = int((img_a[cov][:, 0] >= top - 1e-6).sum())
    check('raising the ambient shifts pixels into the top band (count '
          'non-decreasing) without changing the table',
          n_top_a >= n_top and set(np.unique(img_a[cov][:, 0]).tolist()) <= reds
          | {top}, f'{n_top} -> {n_top_a}')
    st_h = _settings(force_model='DS_HIGHLIGHT', shading_rate='VERTEX')
    sc_h = _toon_master_scene(st_h, 'DS_HIGHLIGHT', steps=4)
    sc_h.materials[1].graph['nodes']['bsdf']['inputs'][0]['default'] = \
        [0.5, 0.5, 0.5, 1.0]
    img_h = np.asarray(R.render(sc_h, st_h))
    st_t4 = _settings(force_model='DS_TOON', shading_rate='VERTEX')
    sc_t4 = _toon_master_scene(st_t4, 'DS_TOON', steps=4)
    sc_t4.materials[1].graph['nodes']['bsdf']['inputs'][0]['default'] = \
        [0.5, 0.5, 0.5, 1.0]
    img_t4 = np.asarray(R.render(sc_t4, st_t4))
    check('DS_HIGHLIGHT >= DS_TOON per pixel and brighter somewhere (the '
          'entry added once more; a grey albedo, four entries)',
          bool(np.all(img_h[cov][:, :3] >= img_t4[cov][:, :3] - 1e-7))
          and float((img_h[cov][:, :3] - img_t4[cov][:, :3]).max()) > 0.0)
    px = img[cov][:, :3]
    check('on a white albedo every DS_TOON pixel is grey (G and B of the '
          'coloured Fill discarded)',
          float(np.abs(px[:, 0] - px[:, 1]).max()) == 0.0
          and float(np.abs(px[:, 0] - px[:, 2]).max()) == 0.0)


def test_c034_gpu_twin_and_refusal():
    """C034: hal_cb_dstoon bitwise on 4096 lanes with a 32-entry table;
    the seam for both items on the textured scene; the linked-socket
    refusal on both roads (CPU bitwise DS_FIXED, GPU probe None with
    'Toon Size'); a linked Toon Smooth does not refuse."""
    rng = np.random.default_rng(11)
    n = 4096
    tab = CB.ds_toon_table(Material(name='t', index=0, model='DS_TOON'))
    red = rng.uniform(0, 1, n).astype(f32)
    red[::4] = (np.floor(red[::4] * 31) + 0.5) / f32(31)
    A = rng.uniform(0, 1, (n, 3)).astype(f32)
    idx = np.clip(np.floor(red * f32(31) + f32(0.5)), 0, 31).astype(np.int32)
    cs = tab[idx].astype(f32)
    for hl in (0.0, 1.0):
        got = glsl_twin(GCB.FN_DSTOON, 'hal_cb_dstoon(hal_in_cs, hal_in_A, hal_in_hl)',
                        {'hal_in_cs': cs, 'hal_in_A': A,
                         'hal_in_hl': np.full(n, hl, f32)}, n)
        cpu = CB.cb_ds_toon(red, A, tab, hl > 0.5)
        check(f'hal_cb_dstoon (highlight {int(hl)}) is bitwise cb_ds_toon on '
              '4096 lanes (d == 0.0)', got is not None and np.array_equal(got, cpu))
    for model in ('DS_TOON', 'DS_HIGHLIGHT'):
        st = _settings(force_model=model, shading_rate='VERTEX')
        sc = _toon_master_scene(st, model, steps=3, textured=True)
        _seam(f'C034 {model} at VERTEX (textured)', sc, st)
    # the refusal: the graph's own model (no force), Toon Size linked
    st = _settings(shading_rate='VERTEX')
    sc = _toon_master_scene(st, 'DS_TOON', link_size=True)
    img_r = np.asarray(R.render(sc, st))
    st_d = _settings(shading_rate='VERTEX')
    sc_d = _toon_master_scene(st_d, 'DS_FIXED', link_size=True)
    img_d = np.asarray(R.render(sc_d, st_d))
    check('a DS_TOON with a linked Toon Size renders bitwise the same graph '
          'under DS_FIXED (shading as DS_FIXED, printed once)',
          np.array_equal(img_r, img_d), f'max {float(np.abs(img_r - img_d).max())}')
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, _a = GSH.plan_frame(job, g)
    check('the GPU probe refuses the linked DS_TOON by name (Toon Size)',
          passes is None and 'Toon Size' in str(why), str(why)[:120])
    st_s = _settings(shading_rate='VERTEX')
    sc_s = _toon_master_scene(st_s, 'DS_TOON', link_smooth=True)
    st_u = _settings(shading_rate='VERTEX')
    sc_u = _toon_master_scene(st_u, 'DS_TOON')
    check('a linked Toon Smooth does NOT refuse: bitwise the unlinked '
          'DS_TOON frame', np.array_equal(np.asarray(R.render(sc_s, st_s)),
                                          np.asarray(R.render(sc_u, st_u))))


# R251 (post pass 2, TODO items 4 / 7b): what the DS pair reads, MEASURED
# on the frame. test_render's tooltip probe calls SH.evaluate alone, and
# the DS lobe's table highlight is computed beside that call (light_surface
# and the GPU per-light block), so that probe cannot see Specular Color or
# Glossiness for any model on the DS_FIXED lobe. This helper is the probe
# that can: the same perturbation (v * 0.25 + 0.05) on the master node's
# socket, two renders, the Ball's pixels compared.

_DS_SOCKETS = {'specular': 'Specular Color', 'glossiness': 'Glossiness'}
_DS_MOVED = {}


def ds_socket_moved_pixels(model, attr, steps=8):
    """How many of the Ball's pixels change when the master node's
    Specular Color / Glossiness takes the tooltip test's perturbation
    under `model` at VERTEX rate, on a table of `steps` entries (eight by
    default: the two-entry table has ONE edge, at index 22, and on this
    Ball the Specular Color perturbation carries no corner across it --
    measured 0 px of 339, where eight entries measure 10 / 3)."""
    key = (model, attr, steps)
    if key in _DS_MOVED:
        return _DS_MOVED[key]
    name = _DS_SOCKETS[attr]
    frames = []
    for perturb in (False, True):
        st = _settings(force_model=model, shading_rate='VERTEX')
        sc = _toon_master_scene(st, model, steps=steps)
        if perturb:
            for sock in sc.materials[1].graph['nodes']['bsdf']['inputs']:
                if sock.get('name') == name:
                    d = sock['default']
                    if isinstance(d, (list, tuple)):
                        sock['default'] = [float(v) * 0.25 + 0.05
                                           for v in d[:3]] + list(d[3:])
                    else:
                        sock['default'] = float(d) * 0.25 + 0.05
        frames.append(np.asarray(R.render(sc, st)))
    cov, _g = _covered(sc, st, mat=1)
    a, b = frames
    _DS_MOVED[key] = int(np.any(a[cov][:, :3] != b[cov][:, :3], axis=-1).sum())
    return _DS_MOVED[key]


def test_c034_sockets_measured():
    """C034: Specular Color and Glossiness move the DS_TOON and the
    DS_HIGHLIGHT frame (measured, not claimed): the specular's RED joins
    the table index, so the perturbation carries pixels across a table
    edge. On an eight-entry table both sockets move both models; on the
    default two-entry table (one edge) Glossiness does and Specular Color
    does not on this Ball -- printed beside each check, not pinned."""
    for model in ('DS_TOON', 'DS_HIGHLIGHT'):
        for attr in ('specular', 'glossiness'):
            n8 = ds_socket_moved_pixels(model, attr)
            n2 = ds_socket_moved_pixels(model, attr, steps=2)
            check(f'{model}: perturbing {_DS_SOCKETS[attr]} moves the Ball '
                  'pixels on the rendered frame (eight-entry table)', n8 > 0,
                  f'{n8} px at 8 entries, {n2} px at the default 2')
    check('the saturating add hides part of what DS_TOON shows: under the '
          'same perturbations DS_HIGHLIGHT moves no more pixels than DS_TOON',
          all(ds_socket_moved_pixels('DS_HIGHLIGHT', a)
              <= ds_socket_moved_pixels('DS_TOON', a)
              for a in ('specular', 'glossiness')))


def test_c034_highlight_saturation():
    """C034, the disclosure pinned: GBATEK's highlight form adds the entry
    it has just modulated by, so an entry of 63 gives 63 whatever the
    texel -- every pixel that indexes the table's top entry is pure
    white. Halcyon's table always tops out at 63, so DS_HIGHLIGHT keeps
    the texel only where the lit red indexes a LOWER entry (more Toon
    Steps, dimmer lamps)."""
    rng = np.random.default_rng(5)
    n = 512
    A = rng.uniform(0, 1, (n, 3)).astype(f32)
    tab2 = CB.ds_toon_table(Material(name='t', index=0, model='DS_HIGHLIGHT'))
    out = CB.cb_ds_toon(np.ones(n, f32), A, tab2, True)
    check('DS_HIGHLIGHT at the top table entry (63) is pure white for '
          'every texel: the default two-entry table saturates each lit '
          'pixel', float(tab2[31]) == 63.0 and bool(np.all(out == f32(1.0))))
    out0 = CB.cb_ds_toon(np.zeros(n, f32), A, tab2, True)
    check('DS_HIGHLIGHT at entry 0 is black for every texel (modulate by 0, '
          'add 0)', bool(np.all(out0 == f32(0.0))))
    m8 = Material(name='t', index=0, model='DS_HIGHLIGHT')
    m8.toon_steps = 8.0
    tab8 = CB.ds_toon_table(m8)
    mids = sorted(set(tab8.astype(int).tolist()) - {0, 63})
    check('an eight-step table has entries between 0 and 63 and still tops '
          'out at 63', len(mids) >= 3 and float(tab8[31]) == 63.0, str(mids))
    k = int(np.flatnonzero(tab8 == f32(mids[0]))[0])
    red = np.full(n, f32(k) / f32(31.0), f32)
    lo = CB.cb_ds_toon(red, np.zeros((n, 3), f32), tab8, True)
    hi = CB.cb_ds_toon(red, np.ones((n, 3), f32), tab8, True)
    mid = CB.cb_ds_toon(red, A, tab8, True)
    check('at a low entry DS_HIGHLIGHT keeps the texel: a black texel gives '
          'the entry itself, a white one twice it, and the rest lie between',
          bool(np.all(lo == f32(mids[0]) * CB.R63))
          and bool(np.all(hi == f32(min(2 * mids[0], 63)) * CB.R63))
          and bool(np.all((mid >= lo) & (mid <= hi)))
          and float(hi.max()) < 1.0, f'entry {mids[0]} at index {k}')


def test_c034_lobe_alias():
    """C034: SH.evaluate('DS_TOON') is SH.evaluate('DS_FIXED') bitwise on
    the tooltip test's own probe, and a DS_TOON sphere's LIGHT-pass rgb is
    bitwise a DS_FIXED sphere's under want_spec (the fixed viewer, the
    table highlight and the 5-bit step apply under the alias)."""
    rng = np.random.default_rng(4)
    n = 64
    N = MX.normalize(rng.normal(size=(n, 3)).astype(f32))
    Lv = MX.normalize(rng.normal(size=(n, 3)).astype(f32))
    V = MX.normalize(rng.normal(size=(n, 3)).astype(f32))
    s = SH.Surface(n)
    d1, s1 = SH.evaluate('DS_TOON', s, N, Lv, V)
    d2, s2 = SH.evaluate('DS_FIXED', s, N, Lv, V)
    check('SH.evaluate under DS_TOON and DS_FIXED returns the same diffuse '
          'and specular bitwise', np.array_equal(d1, d2) and np.array_equal(s1, s2))
    st = _settings(shading_rate='VERTEX')
    sc = demo_scene(st, with_texture=False)
    tri_idx, bary = _ball_samples(sc)
    a, _e = _direct_light(sc, st, 1, tri_idx, bary, {'want_spec': True},
                          model='DS_TOON')
    b, _e = _direct_light(sc, st, 1, tri_idx, bary, {'want_spec': True},
                          model='DS_FIXED')
    check('a DS_TOON sphere\'s LIGHT-pass rgb is bitwise a DS_FIXED '
          'sphere\'s (want_spec on both)', np.array_equal(a, b))
    c, _e = _direct_light(sc, st, 1, tri_idx, bary, {'want_spec': True},
                          model='PHONG')
    check('...and not vacuously: PHONG lights the same samples differently',
          not np.array_equal(a, c))


# --------------------------------------------- C040: the corner quantiser

def test_c040_corner_grid():
    """The corner quantiser: every carrying item's corner image lies on the
    machine's grid; the DS_FIXED row is a bijection on the lobe's 32
    values; idempotence over every grid value; the PS1 A/B (a gradient
    moves, a flat-lit triangle does not)."""
    grids = {'PS1_MODULATE': ('128', 255), 'PS2_HIGHLIGHT': ('128', 255),
             'N64_COMBINE': ('255', None), 'D3D_SEPARATE_SPEC': ('255', None),
             'SATURN_ADD': ('16', 31), 'DS_FIXED': ('63', 'odd'),
             'S22_MODULATE': ('64', 255)}
    for model, (scale, top) in grids.items():
        st = _settings(force_model=model, shading_rate='VERTEX')
        sc = demo_scene(st, with_texture=True)
        _g, job = _job_for(sc, st)
        arr = R.vertex_light_corners(job, 1, 'VERTEX', st)
        rows = arr[np.abs(arr).sum(axis=1) > 0]
        v = rows[:, :3] * f32(float(scale))
        on_grid = bool(np.all(np.abs(v - np.rint(v)) < 1e-4))
        if top == 'odd':
            r = np.rint(v)
            on_grid &= bool(np.all((r == 0) | (r % 2 == 1)))
        elif top is not None:
            on_grid &= float(np.rint(v).max()) <= top
        check(f'{model}: the Ball\'s corner rgb lies on the {scale}-grid'
              + (f' (<= {top})' if top not in (None, 'odd') else ''), on_grid,
              f'max {float(np.rint(v).max())}')
        if model in ('PS1_MODULATE', 'PS2_HIGHLIGHT'):
            a = rows[:, 3] * f32(128)
            check(f'{model}: the carried alpha lies on the 128-grid too',
                  bool(np.all(np.abs(a - np.rint(a)) < 1e-4)))
    k = np.arange(32, dtype=f32)
    lobe = (k * np.float32(1 / 31)).reshape(-1, 1).repeat(3, 1)
    want = (np.where(k > 0, 2 * k + 1, 0) * CB.R63).reshape(-1, 1).repeat(3, 1)
    got = CB.corner_rgb('DS_FIXED', lobe)
    check('the DS_FIXED row maps the lobe\'s k/31 to (2k+1)/63 for k = 1..31 '
          'and 0 to 0 (a bijection, not a second quantisation)',
          np.array_equal(got, want.astype(f32))
          and np.array_equal(CB.corner_rgb('DS_FIXED', got), got))
    for model, mult in (('PS1_MODULATE', 128), ('N64_COMBINE', 255),
                        ('SATURN_ADD', 16), ('S22_MODULATE', 64)):
        ks = np.arange(256 if mult != 16 else 32, dtype=f32)
        # the 8-bit grid IS k * R255 (the module's own multiply; k / 255
        # differs for 126 of the 256 k -- review item 7)
        vals = (ks * CB.R255 if mult == 255 else ks / f32(mult)
                ).reshape(-1, 1).repeat(3, 1).astype(f32)
        q = CB.corner_rgb(model, vals)
        check(f'{model}: a corner-exact value passes the quantiser unchanged '
              '(idempotence over every grid value)', np.array_equal(q, vals))
    for model in ('GOURAUD', 'FLAT', 'FLAT_GL_LAST', 'PCX_INTENSITY',
                  'SEGA_MODEL2', 'DS_TOON', 'MEGA_DRIVE_SH', 'SUPERFX_PLOT',
                  'PHONG'):
        check(f'{model}: the corner quantiser is None (untouched)',
              CB.corner_rgb(model, np.ones((2, 3), f32)) is None)
    # the A/B: the PS1 frame differs from the same frame with the quantiser
    # off on a gradient; a single flat-lit triangle is bitwise the same
    st = _settings(force_model='PS1_MODULATE', shading_rate='VERTEX')
    sc = demo_scene(st, with_texture=True)
    on = np.asarray(R.render(sc, st))
    saved = CB.corner_rgb
    try:
        CB.corner_rgb = lambda model, rgb: None
        off = np.asarray(R.render(demo_scene(st, with_texture=True), st))
    finally:
        CB.corner_rgb = saved
    check('PS1_MODULATE: the corner step moves a gradient (> 1/255 somewhere)',
          float(np.abs(on - off).max()) > 1.0 / 255)
    st_q = _settings(force_model='PS1_MODULATE', shading_rate='FACE')
    sc_q = _quad_scene(st_q, 'PS1_MODULATE', tilt=0.0)
    one = np.asarray(R.render(sc_q, st_q))
    try:
        CB.corner_rgb = lambda model, rgb: None
        two = np.asarray(R.render(_quad_scene(st_q, 'PS1_MODULATE', tilt=0.0),
                                  st_q))
    finally:
        CB.corner_rgb = saved
    check('a flat-lit quad is bitwise the same with the corner step on or '
          'off (idempotence in the frame)', np.array_equal(one, two))


# ------------------------------------------------ C066: 64-step luma ramp

def test_c066_division_rule():
    """C066: floor((c5*lum6*255 + 0.5) / 1953) in float32 equals Python's
    integer division over the whole 32 x 64 domain, on both roads."""
    c5, lum6 = np.meshgrid(np.arange(32), np.arange(64), indexing='ij')
    c5 = c5.ravel()
    lum6 = lum6.ravel()
    num = (c5 * lum6 * 255).astype(np.int64)
    want = num // 1953
    n32 = (c5.astype(f32) * lum6.astype(f32)) * f32(255)
    got = np.floor((n32 + f32(0.5)) / f32(1953)).astype(np.int64)
    check('the margin rule holds for every (c5, lum6) on the CPU',
          np.array_equal(got, want))
    n = c5.size
    A = np.stack([c5 / f32(31)] * 3, 1).astype(f32)
    L = np.stack([lum6 * 4 / f32(255)] * 3, 1).astype(f32) + f32(1e-4)
    cpu = CB.cb_luma64(L, A)
    twin = glsl_twin(GCB.FN_LUMA64, 'hal_cb_luma64(hal_in_L, hal_in_A)',
                     {'hal_in_L': L, 'hal_in_A': A}, n)
    check('hal_cb_luma64 is bitwise cb_luma64 over the 2048-lane domain',
          twin is not None and np.array_equal(twin, cpu))
    check('...and the values ARE num // 1953 on the 1/255 grid',
          np.array_equal(np.rint(cpu[:, 0] * 255).astype(np.int64), want))


def test_c066_laws_and_twin():
    """C066: a white Ball under SEGA_MODEL3 takes at most 64 luminance
    rungs on the 1/255 grid, the same count for a red Ball; SEGA_MODEL2
    gives one value per triangle; the cross-pack A/B against the lobe
    alone; the seam for both names on the textured scene."""
    idents = [m[0] for m in SH.MODEL_ITEMS]
    if 'SEGA_MODEL2' not in idents or 'SEGA_MODEL3' not in idents:
        print('  [skip] the lighting pack\'s Sega names are not in MODEL_ITEMS')
        return
    def ball(model, diffuse):
        st = _settings(force_model=model, max_lights=1,
                       light_limit_mode='FIRST')
        sc = demo_scene(st, with_texture=False)
        sc.materials[1].diffuse = diffuse
        img = np.asarray(R.render(sc, st))
        cov, g = _covered(sc, st, mat=1)
        return img, cov, g
    img_w, cov, g = ball('SEGA_MODEL3', (1.0, 1.0, 1.0))
    px = img_w[cov][:, :3]
    on_grid = bool(np.all(np.abs(px * 255 - np.rint(px * 255)) < 1e-3))
    n_w = len(np.unique(np.rint(px[:, 0] * 255)))
    check('SEGA_MODEL3: the white Ball\'s luminances take at most 64 values '
          'on the 1/255 grid', on_grid and n_w <= 64, f'{n_w} values')
    img_r, cov_r, _g = ball('SEGA_MODEL3', (0.85, 0.2, 0.15))
    pr = img_r[cov_r][:, :3]
    n_r = len(np.unique(np.rint(pr * 255).astype(np.int64), axis=0))
    check('...the red Ball bands at the same number of rungs (banding '
          'independent of hue)', n_r == n_w, f'{n_r} vs {n_w}')
    img_2, cov2, g2 = ball('SEGA_MODEL2', (0.85, 0.2, 0.15))
    var = 0.0
    for t in np.unique(g2.tri[cov2]):
        sel = cov2 & (g2.tri == t)
        var = max(var, float(np.ptp(img_2[sel][:, :3], axis=0).max()))
    check('SEGA_MODEL2: exactly one value per triangle (per-face range 0)',
          var == 0.0, str(var))
    saved = CB.recombine
    try:
        CB.recombine = lambda *a, **k: None
        st = _settings(force_model='SEGA_MODEL2', max_lights=1,
                       light_limit_mode='FIRST')
        sc = demo_scene(st, with_texture=False)
        sc.materials[1].diffuse = (0.85, 0.2, 0.15)
        lobe_only = np.asarray(R.render(sc, st))
    finally:
        CB.recombine = saved
    check('SEGA_MODEL2: the frame differs from the lobe alone (the combine '
          'half is not vacuous)',
          float(np.abs(img_2 - lobe_only).max()) > 1.0 / 255)
    for model in ('SEGA_MODEL2', 'SEGA_MODEL3'):
        st = _settings(force_model=model)
        sc = demo_scene(st, with_texture=True)
        _seam(f'C066 {model} (textured)', sc, st)
    L, A = _pairs()
    got = glsl_twin(GCB.FN_LUMA64, 'hal_cb_luma64(hal_in_L, hal_in_A)',
                    {'hal_in_L': L, 'hal_in_A': A}, L.shape[0])
    check('hal_cb_luma64 is bitwise cb_luma64 on 4096 pairs (d == 0.0)',
          got is not None and np.array_equal(got, CB.cb_luma64(L, A)))


# ------------------------------------------ C057: Mega Drive S/H levels

def test_c057_ramps():
    """C057: the three DAC ramps pinned by value and the operator's laws
    (a table, never a multiply); hal_sh_ramp bitwise over the 3 x 8 domain."""
    check('the ramps are the measured DAC levels',
          CB.RAMP_NORMAL == (0, 52, 87, 116, 144, 172, 206, 255)
          and CB.RAMP_SHADOW == (0, 29, 52, 70, 87, 101, 116, 130)
          and CB.RAMP_HIGHLIGHT == (130, 144, 158, 172, 187, 206, 228, 255))
    def sh(cls, a):
        return CB.cb_mdsh(np.full(1, cls, f32), np.full((1, 3), a, f32))[0, 0]
    check('sh(SHADOW, 1) == 130/255, sh(NORMAL, 1) == 1, sh(HIGHLIGHT, 0) == '
          '130/255 (the highlight lifts black)',
          sh(1, 1.0) == f32(130) * CB.R255 and sh(0, 1.0) == f32(255) * CB.R255
          and sh(2, 0.0) == f32(130) * CB.R255)
    check('sh(SHADOW, 87/255) == 52/255 -- not a multiply (a linear scale '
          'would give 44)', sh(1, 87 / 255) == f32(52) * CB.R255)
    cls, c = np.meshgrid(np.arange(3), np.arange(8), indexing='ij')
    cls = cls.ravel().astype(f32)
    c = c.ravel().astype(f32)
    got = glsl_twin(GCB.FN_SHRAMP, 'vec3(hal_sh_ramp(hal_in_cls, hal_in_c))',
                    {'hal_in_cls': cls, 'hal_in_c': c}, cls.size)
    want = CB.RAMPS[cls.astype(int), c.astype(int)].astype(f32)
    check('hal_sh_ramp is bitwise the ramp table over cls 0..2 x c 0..7',
          got is not None and np.array_equal(got[:, 0], want))


def test_c057_class_carry():
    """C057: at FACE under MEGA_DRIVE_SH the Ball's faces carry their class
    in the corner alpha (SHADOW away from the Key, NORMAL toward it,
    HIGHLIGHT at the cap), every output value lies in the union of the
    ramps, no lamps = all SHADOW, and the house-key rule three ways."""
    def classes(**kw):
        st = _settings(force_model='MEGA_DRIVE_SH', **kw)
        sc = demo_scene(st, with_texture=False)
        for k, v in kw.pop('_mat', {}).items():
            setattr(sc.materials[1], k, v)
        return sc, st
    st = _settings(force_model='MEGA_DRIVE_SH')
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].specular_level = 1.0
    sc.materials[1].glossiness = 8.0
    _g, job = _job_for(sc, st)
    arr = R.vertex_light_corners(job, 1, 'FACE', st)
    sel = np.nonzero(np.asarray(sc.mesh.mat_index) == 1)[0]
    cls = arr[sel * 3, 3]
    check('the Ball\'s faces carry classes 0 / 1 / 2 (NORMAL, SHADOW and '
          'HIGHLIGHT all present)', set(np.unique(cls).tolist()) == {0.0, 1.0, 2.0},
          str(np.unique(cls)))
    # faces facing away from the Key are SHADOW, faces toward it NORMAL or
    # HIGHLIGHT: the face normal against the Key's direction
    tris = sc.mesh.tris[sel]
    v = sc.mesh.verts
    centre = v[np.unique(tris)].mean(axis=0)
    fc = v[tris].mean(axis=1)
    fn = fc - centre
    fn = fn / np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    key = np.asarray(sc.lights[0].direction, f32)
    _v, _p, _vp, eye = R.camera_matrices(sc.camera, W, H)
    to_eye = np.asarray(eye, f32)[None, :] - fc
    to_eye = to_eye / np.maximum(np.linalg.norm(to_eye, axis=1, keepdims=True),
                                 1e-9)
    seen = (fn * to_eye).sum(axis=1) > 0.2
    toward = np.where(seen, -(fn @ (key / np.linalg.norm(key))), 0.0)
    check('faces facing away from the Key sun are SHADOW class (or HIGHLIGHT '
          'from the Fill\'s specular, never NORMAL)',
          bool(np.all(cls[toward < -0.3] != 0.0)))
    check('faces facing the Key sun are NORMAL or HIGHLIGHT',
          bool(np.all(cls[toward > 0.6] != 1.0)))
    st0 = _settings(force_model='MEGA_DRIVE_SH')
    sc0 = demo_scene(st0, with_texture=False)
    sc0.materials[1].specular_level = 0.0
    _g, job0 = _job_for(sc0, st0)
    cls0 = R.vertex_light_corners(job0, 1, 'FACE', st0)[sel * 3, 3]
    check('with no highlight class, faces away from the Key are exactly '
          'SHADOW and faces toward it NORMAL',
          bool(np.all(cls0[toward < -0.3] == 1.0))
          and bool(np.all(cls0[toward > 0.6] == 0.0)))
    img = np.asarray(R.render(sc, st))
    cov, g = _covered(sc, st, mat=1)
    px8 = np.rint(img[cov][:, :3] * 255)
    allowed = set(CB.RAMP_NORMAL) | set(CB.RAMP_SHADOW) | set(CB.RAMP_HIGHLIGHT)
    check('every output value of the Ball lies in the union of the three '
          'ramps / 255 (the set-membership law)',
          set(np.unique(px8).astype(int).tolist()) <= allowed)
    st_n = _settings(force_model='MEGA_DRIVE_SH')
    sc_n = demo_scene(st_n, with_texture=False)
    sc_n.lights = []
    _g, job_n = _job_for(sc_n, st_n)
    cls_n = R.vertex_light_corners(job_n, 1, 'FACE', st_n)[sel * 3, 3]
    check('with NO lamps every face is SHADOW', bool(np.all(cls_n == 1.0)))
    # (i) ordering-independence: FIRST 1 vs 8 with no highlight class
    def class_map(light_limit_mode, max_lights, swap=False):
        st = _settings(force_model='MEGA_DRIVE_SH',
                       light_limit_mode=light_limit_mode, max_lights=max_lights)
        sc = demo_scene(st, with_texture=False)
        sc.materials[1].specular_level = 0.0
        if swap:
            sc.lights = [sc.lights[1], sc.lights[0]]
        _g, job = _job_for(sc, st)
        return R.vertex_light_corners(job, 1, 'FACE', st)[sel * 3, 3]
    a = class_map('FIRST', 1)
    b = class_map('FIRST', 8)
    check('(i) FIRST with max_lights 1 and 8 give IDENTICAL class maps (the '
          'class reads the key by identity, not by position)',
          np.array_equal(a, b))
    c = class_map('FIRST', 8, swap=True)
    check('(ii) swapping Key and Fill changes the class map (the key is the '
          'first lamp)', not np.array_equal(b, c))
    st_b = _settings(force_model='MEGA_DRIVE_SH', light_limit_mode='BRIGHTEST',
                     max_lights=1)
    sc_b = demo_scene(st_b, with_texture=False)
    active = LI.select_lights(sc_b.lights, st_b)
    tri_idx, bary = _ball_samples(sc_b)
    ex = {'want_spec': True, 'want_key_lit': True}
    _rgb, ex = _direct_light(sc_b, st_b, 1, tri_idx, bary, ex)
    d = class_map('BRIGHTEST', 1)
    check('(iii) BRIGHTEST at max_lights 1 culls the Key (Fill alone), '
          'key_lit is never captured and every face is SHADOW',
          [l.name for l in active] == ['Fill'] and 'key_lit' not in ex
          and bool(np.all(d == 1.0)), str([l.name for l in active]))
    st_t = _settings(force_model='MEGA_DRIVE_SH')
    sc_t = demo_scene(st_t, with_texture=True)
    _seam('C057 MEGA_DRIVE_SH (textured)', sc_t, st_t)


# ------------------------------------------------ C049: the Super FX plot

def test_c049_pairs():
    """C049: _plot_pairs against a brute-force double loop on 500 colours
    with EGA16; an entry gives i == j; a tie resolves to the smallest i
    then j."""
    from ..core import palette as PA
    P = np.asarray(PA.EGA16, f32)
    rng = np.random.default_rng(21)
    cols = rng.uniform(0, 1, (500, 3)).astype(f32)
    i, j = CB._plot_pairs(cols, P)
    c8 = np.rint(cols * 255)
    pal8 = np.rint(P * 255)
    bi, bj = [], []
    for c in c8:
        best, bp = None, None
        for a in range(16):
            for b in range(a, 16):
                d = int(((2 * c - (pal8[a] + pal8[b])) ** 2).sum())
                if best is None or d < best:
                    best, bp = d, (a, b)
        bi.append(bp[0])
        bj.append(bp[1])
    check('_plot_pairs matches the brute-force (i <= j) search on 500 '
          'colours', np.array_equal(i, bi) and np.array_equal(j, bj))
    C4 = np.asarray(PA.CGA4, f32)
    i2, j2 = CB._plot_pairs(C4, C4)
    check('a colour equal to an entry gives i == j (a solid fill; CGA4 -- '
          'EGA16\'s evenly spaced greys tie with a (black, light) pair)',
          np.array_equal(i2, j2) and np.array_equal(i2, np.arange(4)))
    i5, j5 = CB._plot_pairs(P, P)
    pal8 = np.rint(P * 255)
    check('on EGA16 an entry\'s pair has a doubled mean exactly the entry '
          '(distance 0; ties go to the smallest i)',
          np.array_equal(pal8[i5] + pal8[j5], 2 * pal8))
    G = np.asarray(PA.grayscale(4), f32)
    mid = (0.5 * (G[0] + G[3]))[None, :]
    i3, j3 = CB._plot_pairs(mid, G)
    check('a tie between pairs (0, 3) and (1, 2) resolves to (0, 3): the '
          'smallest i, then j', (int(i3[0]), int(j3[0])) == (0, 3))


def _plot_settings(**kw):
    st = _settings(force_model='SUPERFX_PLOT', palette_mode='EGA16',
                   aa_mode='NONE', shading_rate='FACE', gamma=1.0,
                   color_management='NONE')
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def test_c049_laws():
    """C049: every covered pixel is one of its face's two entries; the
    parity law on same-face neighbours; the pattern holds on the OUTPUT
    grid under 2x2 supersampling; the A/B against ordered dither to 4
    bits of the FLAT frame; threads 1 vs 4 bitwise."""
    from ..core import palette as PA
    P = np.asarray(PA.EGA16, f32)
    st = _plot_settings()
    sc = demo_scene(st, with_texture=False)
    img = np.asarray(R.render(sc, st))
    cov, g = _covered(sc, st)
    _g, job = _job_for(sc, st)
    arr = np.zeros((sc.mesh.tris.shape[0] * 3, 4), f32)
    for mi in range(len(sc.materials)):
        arr += R.vertex_light_corners(job, mi, 'FACE', st)
    faces = np.unique(g.tri[cov])
    ok_member, ok_parity, n_pairs = True, True, 0
    yy, xx = np.nonzero(cov)
    tri_of = g.tri
    for t in faces:
        sel = cov & (tri_of == t)
        ys, xs = np.nonzero(sel)
        px = img[ys, xs, :3]
        Pi, Pj = arr[t * 3, :3], arr[t * 3 + 1, :3]
        ok_member &= bool(np.all(np.all(px == Pi, axis=1) | np.all(px == Pj, axis=1)))
        if not np.array_equal(Pi, Pj):
            n_pairs += 1
            for y, x in zip(ys, xs):
                if y + 1 < img.shape[0] and x + 1 < img.shape[1] and sel[y + 1, x + 1]:
                    ok_parity &= np.array_equal(img[y, x, :3], img[y + 1, x + 1, :3])
                if x + 1 < img.shape[1] and sel[y, x + 1]:
                    ok_parity &= not np.array_equal(img[y, x, :3], img[y, x + 1, :3])
    check('every covered pixel equals one of its face\'s two palette entries',
          ok_member)
    check('the parity law: diagonal neighbours on one face agree, horizontal '
          'neighbours differ (faces with two entries exist)',
          ok_parity and n_pairs > 0, f'{n_pairs} two-entry faces')
    check('every pixel is an EGA16 entry',
          bool(np.all(np.any(np.all(img[cov][:, None, :3] == P[None], axis=2),
                             axis=1))))
    st2 = _plot_settings(aa_mode='SUPERSAMPLE', aa_samples=4, aa_filter='BOX')
    sc2 = demo_scene(st2, with_texture=False)
    img2 = np.asarray(R.render(sc2, st2))
    # the INTERNAL G-buffer (2x): an output 2x2 neighbourhood whose 4x4
    # internal block lies on ONE face resolves each 2x2 block of one entry
    # to that entry, so the checkerboard is on the output grid
    w2, h2 = st2.resolution_x * 2, st2.resolution_y * 2
    _v, _p, vp2, _e = R.camera_matrices(sc2.camera, w2, h2)
    gi = raster.GBuffer(w2, h2)
    raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w2, h2, gbuf=gi,
                     depth_bits=st2.depth_precision)
    ok2, n2 = True, 0
    for y in range(0, img2.shape[0] - 1):
        for x in range(0, img2.shape[1] - 1):
            blk = gi.tri[2 * y:2 * y + 4, 2 * x:2 * x + 4]
            if bool((blk >= 0).all()) and len(np.unique(blk)) == 1:
                a, b, c = img2[y, x, :3], img2[y + 1, x + 1, :3], img2[y, x + 1, :3]
                if not np.array_equal(a, c):
                    n2 += 1
                    ok2 &= np.array_equal(a, b)
    check('under 2x2 supersampling with BOX the checkerboard holds on the '
          'OUTPUT grid (a 2x2 block of one entry resolves to that entry)',
          ok2 and n2 > 0, f'{n2} differing neighbours')
    st_f = _settings(force_model='FLAT', color_depth='4', dither='BAYER2',
                     shading_rate='FACE')
    sc_f = demo_scene(st_f, with_texture=False)
    flat = np.asarray(R.render(sc_f, st_f))
    post_flat = np.asarray(PO.process(flat, st_f, **_post_kw(sc_f, st_f)))
    check('the plot differs from ordered dither to 4 bits of the FLAT frame '
          '(one pair per FACE before post, not a per-pixel pattern)',
          float(np.abs(img[cov][:, :3] - post_flat[cov][:, :3]).max()) > 1.0 / 255)
    st4 = _plot_settings(threads=4)
    img4 = np.asarray(R.render(demo_scene(st4, with_texture=False), st4))
    check('threads 1 and 4 render the plot bitwise', np.array_equal(img, img4))


def test_c049_refusals_by_name():
    """C049: ADAPTIVE, a 216-entry palette and gamma 2.2 each refuse by
    name on the CPU (bitwise the FLAT frame) and on the GPU probe."""
    cases = (('ADAPTIVE', dict(palette_mode='ADAPTIVE'),
              'built from the finished frame'),
             ('WEB216', dict(palette_mode='WEB216'), '216 entries'),
             ('gamma 2.2', dict(gamma=2.2), 'move them off the palette'))
    import io
    import contextlib
    for label, kw, phrase in cases:
        st = _plot_settings(**kw)
        sc = demo_scene(st, with_texture=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            img = np.asarray(R.render(sc, st))
        printed = buf.getvalue()
        st_f = _settings(force_model='FLAT', shading_rate='FACE', **kw)
        flat = np.asarray(R.render(demo_scene(st_f, with_texture=False), st_f))
        check(f'{label}: the CPU prints "{phrase}" once and renders bitwise '
              'the FLAT frame', printed.count(phrase) == 1
              and 'SUPERFX_PLOT: shading as FLAT' in printed
              and np.array_equal(img, flat), printed.strip()[:120])
        g, job = _job_for(sc, st)
        GSH._PLAN_CACHE.clear()
        passes, why, _a = GSH.plan_frame(job, g)
        check(f'{label}: the GPU probe refuses with the same phrase',
              passes is None and phrase in str(why), str(why)[:120])


def test_c049_gpu_twin():
    """C049: the simulator frame is bitwise the CPU frame on every covered
    pixel (the pass selects CPU-chosen entries by parity), at ss 1 and 2;
    the plan re-plans when the palette gate changes (the signature)."""
    for kw in ({}, dict(aa_mode='SUPERSAMPLE', aa_samples=4, aa_filter='BOX')):
        st = _plot_settings(**kw)
        sc = demo_scene(st, with_texture=False)
        d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
        tag = 'ss 2' if kw else 'ss 1'
        check(f'C049 {tag}: the plan is not refused and simulates',
              passes is not None and img is not None, str(why))
        if img is None:
            continue
        # compare on the INTERNAL grid: the simulate image is at render
        # size before the resolve, the CPU frame is resolved -- so pin the
        # internal frame through the FACE corners instead when ss > 1
        if not kw:
            cov, g = _covered(sc, st)
            check('C049 ss 1: the simulator frame is bitwise the CPU frame on '
                  'every covered pixel (d == 0.0)',
                  np.array_equal(img[cov], cpu[cov][:, :3]), f'max {d}')
        else:
            # the simulator frame is at the INTERNAL size (before the
            # resolve): every covered pixel is one of its face's two
            # CPU-chosen entries, the select law at ss 2
            g2, job2 = _job_for(sc, st)
            arr = np.zeros((sc.mesh.tris.shape[0] * 3, 4), f32)
            for mi in range(len(sc.materials)):
                arr += R.vertex_light_corners(job2, mi, 'FACE', st)
            cov2 = g2.tri >= 0
            yy, xx = np.nonzero(cov2)
            t = g2.tri[yy, xx]
            px = img[yy, xx, :3]
            ok = np.all(px == arr[t * 3, :3], axis=1) \
                | np.all(px == arr[t * 3 + 1, :3], axis=1)
            par = CB.plot_parity(xx, yy, 2)
            want = np.where((par > 0.5)[:, None], arr[t * 3 + 1, :3],
                            arr[t * 3, :3])
            check('C049 ss 2: every covered pixel of the internal-size '
                  'simulator frame is its face\'s CPU-chosen entry by '
                  'output-pixel parity (d == 0.0)',
                  bool(np.all(ok)) and np.array_equal(px, want))
    st = _plot_settings()
    sc = demo_scene(st, with_texture=False)
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    p1, _w, _a = GSH.plan_frame(job, g)
    st.palette_mode = 'CGA4'
    p2, _w2, _a2 = GSH.plan_frame(job, g)
    check('changing the palette mode changes the plan signature (the gate '
          'is signed; no cache hit walks past it)',
          p1 is not None and p2 is not None and p1 is not p2)


# ------------------------------------------------- the fake-device road

def test_fake_device_road():
    """Two rows through tests/fakedevice: a PS1_MODULATE textured frame and
    the Super FX plot render on the GPU device without a driver and match
    the CPU frame (the plot bitwise; the modulate within the resident
    road's own 6e-6)."""
    from . import fakedevice
    for label, st_kw, textured, bar in (
            ('PS1_MODULATE at VERTEX', dict(force_model='PS1_MODULATE',
                                            shading_rate='VERTEX'), True, 6e-6),
            ('the Super FX plot', dict(force_model='SUPERFX_PLOT',
                                       palette_mode='EGA16', aa_mode='NONE',
                                       shading_rate='FACE'), False, 0.0)):
        st_c = _settings(**st_kw)
        sc_c = demo_scene(st_c, with_texture=textured)
        cpu = np.asarray(R.render(sc_c, st_c))
        st_g = _settings(render_device='GPU', **st_kw)
        sc_g = demo_scene(st_g, with_texture=textured)
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        with fakedevice.installed() as dev:
            gpu = np.asarray(R.render(sc_g, st_g))
            n_shaders = len(dev.shaders)
        d = float(np.abs(gpu[..., :3] - cpu[..., :3]).max()) \
            if gpu.shape == cpu.shape else None
        check(f'{label} through the fake device compiles passes and matches '
              f'the CPU frame (max {d}, bar {bar})',
              n_shaders >= 1 and d is not None and d <= bar,
              f'{n_shaders} shaders')


# ---------------------------------------------------------------- main

def main():
    from . import utf8_console
    utf8_console()
    names = sorted(n for n in globals() if n.startswith('test_'))
    for n in names:
        print(f'-- {n}')
        try:
            globals()[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print()
    print(f'{len(FAILS)} failure(s)' if FAILS else 'everything passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
