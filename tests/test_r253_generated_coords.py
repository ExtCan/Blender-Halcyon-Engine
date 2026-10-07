"""R253 (1.92.0): Generated texture coordinates are measured in the
object's OWN box.

Blender's Generated output is the mesh's texture space -- the
object-space bounding box (auto texspace) or the manual Texture Space
-- so it never changes with the object transform. Halcyon measured it
over the WORLD box of the exported (already transformed) vertices, so a
rotating, scaling or deforming object scrolled through its procedural
textures. The fix takes the world position back into the object's frame
through the per-object inverse matrix both devices already carry (R243)
and normalises it over the object-space box (`export.py` gen_bounds /
`ShadeJob.object_generated_frame`), with `generated_space` = WORLD as
the legacy switch.

Every heading proves one law: the one chain both devices run is
bitwise the simulator's dot (and the einsum it replaces); Generated
follows a moved, turned and scaled object; identity scenes are BITWISE
the 1.91 numbers (contexts and whole frames); the legacy switch
reproduces the scroll bitwise; the GPU twins the CPU on moved objects
(the pass and the bump pre-pass, 6e-3 -- the R243 bar; the generated
line itself bitwise at the function level); the manual texture space is
honoured; the cartoon centre still measures world space; the export
road carries the box; the setting's surface (tooltip, enum items, the
plan signature) and the refusal by name.

    python -m halcyon.tests.test_r253_generated_coords
"""
import re
import sys
import traceback

import numpy as np

from . import utf8_console
from .scenebuild import demo_scene
from .test_render import _cartoon_graph, _sk, _wnode, base_settings
from ..core import mathx as M
from ..core import raster as CR
from ..core import render as R
from ..core.scene import Material, ObjectInfo
from ..core.settings import RenderSettings
from ..gpu import device as DEV
from ..gpu import material as GM
from ..gpu import shade as GSH
from ..shaders import builtins as SB
from ..shaders.compiler import try_compile

FAILS = []
f32 = np.float32
W, H = 96, 72


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


# ------------------------------------------------------------------ rigs

def rot_z(a, t=(0, 0, 0)):
    c, s = np.cos(a), np.sin(a)
    m = np.eye(4, dtype=np.float32)
    m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
    m[:3, 3] = t
    return m


def xform(t=(0, 0, 0), rz=0.0, s=(1, 1, 1)):
    """translate(t) @ rot_z(rz) @ diag(s): the full non-uniform affine
    the fix must undo."""
    S = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
    return (rot_z(rz, t) @ S).astype(np.float32)


#: the demo ball's world-space (= identity-local) bounds: centre
#: (-1.3, 0.2, 1.0), radius 1
BALL_C = np.array([-1.3, 0.2, 1.0], f32)

#: a moved ball that stays in the demo camera's view and clear of the box
MOVE = xform(t=(0.4, -0.3, 0.6), rz=0.7, s=(1.3, 0.8, 1.1))
MOVE_T = xform(t=(0.4, -0.3, 0.6))


def marble_graph(axis='Y', scale=4.0):
    """Diffuse BSDF whose Color is a Marble on an UNLINKED Vector: the
    pattern reads Generated coordinates, as a user wires it."""
    return {'output': 'out', 'nodes': {
        'pat': _wnode('pat', 'HALCYON_MarbleNode',
                      {'octaves': 4, 'axis': axis},
                      [_sk('Vector', 'VECTOR', [0, 0, 0]),
                       _sk('Scale', 'VALUE', scale),
                       _sk('Turbulence', 'VALUE', 1.3),
                       _sk('Veins', 'VALUE', 1.2),
                       _sk('Sharpness', 'VALUE', 0.8),
                       _sk('Color 1', 'RGBA', [0.92, 0.9, 0.86, 1.0]),
                       _sk('Color 2', 'RGBA', [0.14, 0.13, 0.16, 1.0])],
                      [{'name': 'Color', 'type': 'RGBA'},
                       {'name': 'Fac', 'type': 'VALUE'}]),
        'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                    [_sk('Color', 'RGBA', [.8, .8, .8, 1], ['pat', 0]),
                     _sk('Roughness', 'VALUE', 0.),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['b', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}


def object_and_generated_graph():
    """A material reading BOTH the Texture Coordinate Object output (a
    Marble on it) and Generated (a second Marble, unlinked): the two
    hal_obj_rN readers in one pass."""
    g = marble_graph('X', 3.0)
    g['nodes']['tc'] = _wnode(
        'tc', 'ShaderNodeTexCoord', {}, [],
        [{'name': n, 'type': 'VECTOR'} for n in
         ('Generated', 'Normal', 'UV', 'Object', 'Camera', 'Window',
          'Reflection')])
    g['nodes']['pat2'] = _wnode(
        'pat2', 'HALCYON_MarbleNode', {'octaves': 3, 'axis': 'Z'},
        [_sk('Vector', 'VECTOR', [0, 0, 0], ['tc', 3]),
         _sk('Scale', 'VALUE', 5.0),
         _sk('Turbulence', 'VALUE', 0.9),
         _sk('Veins', 'VALUE', 1.0),
         _sk('Sharpness', 'VALUE', 1.0),
         _sk('Color 1', 'RGBA', [0.2, 0.6, 0.9, 1.0]),
         _sk('Color 2', 'RGBA', [0.9, 0.3, 0.1, 1.0])],
        [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Fac', 'type': 'VALUE'}])
    g['nodes']['mix'] = _wnode(
        'mix', 'ShaderNodeMixRGB', {'blend_type': 'MIX'},
        [_sk('Fac', 'VALUE', 0.5), _sk('Color1', 'RGBA', [0, 0, 0, 1], ['pat', 0]),
         _sk('Color2', 'RGBA', [0, 0, 0, 1], ['pat2', 0])],
        [{'name': 'Color', 'type': 'RGBA'}])
    for s in g['nodes']['b']['inputs']:
        if s['name'] == 'Color':
            s['link'] = ['mix', 0]
    return g


def bump_scene(st):
    """The normal-mapped master ball (scenebuild.add_normal_mapped_ball:
    a Generated marble on the Diffuse Color) with a Bump between: the
    marble's Fac into the Bump's Height, the Bump into the master's
    Normal socket (the one socket the GPU bends a normal through) --
    the height chain reads Generated in the GPU's pre-pass."""
    from .scenebuild import add_normal_mapped_ball
    sc = _scene(st)
    add_normal_mapped_ball(sc)
    g = sc.materials[1].graph
    g['nodes']['bump'] = _wnode(
        'bump', 'ShaderNodeBump', {'invert': False},
        [_sk('Strength', 'VALUE', 0.8), _sk('Distance', 'VALUE', 0.6),
         _sk('Height', 'VALUE', 0.5, ['marble', 1]),
         _sk('Normal', 'VECTOR', [0, 0, 0])],
        [{'name': 'Normal', 'type': 'VECTOR'}])
    for s in g['nodes']['hal']['inputs']:
        if s['name'] == 'Normal':
            s['link'] = ['bump', 0]
    return sc


def _settings(**kw):
    st = base_settings(W, H)
    st.transparency = 'NONE'
    st.shadows = False
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _scene(st, graph=None):
    sc = demo_scene(st, with_texture=False)
    if graph is not None:
        sc.materials[1] = Material(name='Pat', index=1, graph=graph)
    return sc


def _normalize(v):
    ln = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.where(ln < 1e-12, 1.0, ln)).astype(f32)


def move_ball(sc, mw, gen_bounds=None, move_verts=True):
    """The demo ball (object 1) under the matrix `mw`: its vertices go
    through the matrix (the exporter hands the renderer WORLD vertices)
    and its ObjectInfo carries the matrix -- what a Blender export of a
    moved object looks like. `move_verts=False` sets the matrix alone
    (the R243 Object-output rig)."""
    mw = np.asarray(mw, f32)
    mesh = sc.mesh
    if move_verts:
        vert_obj = np.zeros(mesh.verts.shape[0], np.int32)
        vert_obj[mesh.tris.reshape(-1)] = np.repeat(mesh.obj_index, 3)
        sel = vert_obj == 1
        V = mesh.verts.copy()
        V[sel] = (V[sel] @ mw[:3, :3].T + mw[:3, 3]).astype(f32)
        mesh.verts = V.astype(f32)
        N = mesh.normals.copy()
        nm = np.linalg.inv(mw[:3, :3]).T
        N[sel] = _normalize(N[sel] @ nm.T)
        mesh.normals = N.astype(f32)
        tsel = mesh.obj_index == 1
        T = mesh.tris[tsel]
        e1 = V[T[:, 1]] - V[T[:, 0]]
        e2 = V[T[:, 2]] - V[T[:, 0]]
        fn = mesh.face_normals.copy()
        fn[tsel] = _normalize(np.cross(e1, e2))
        mesh.face_normals = fn.astype(f32)
    sc.objects[1].matrix_world = mw
    sc.objects[1].location = tuple(float(x) for x in mw[:3, 3])
    sc.objects[1].gen_bounds = gen_bounds
    return sc


def _job(sc, st):
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    return R.ShadeJob(sc, st, {}, None, view, eye, W, H), vp


def ball_ctx(sc, st, need=None):
    """The ball's triangles at bary 1/3, through ShadeJob.context."""
    job, _vp = _job(sc, st)
    sel = np.nonzero(sc.mesh.mat_index == 1)[0]
    bary = np.full((sel.size, 3), 1 / 3.0, f32)
    return job, sel, bary, job.context(sel, bary, need=need)


def gpu_road(sc, st, label, clear=True):
    """The both()-style road: CPU frame, raster, plan, simulate.
    Returns (cpu, sim, mask, passes, why)."""
    cpu = np.asarray(R.render(sc, st))[..., :3]
    _view, _p, vp, _eye = R.camera_matrices(sc.camera, W, H)
    gb = CR.GBuffer(W, H)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=gb)
    job, _vp = _job(sc, st)
    if clear:
        GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, gb)
    m = gb.tri >= 0
    check(f'the GPU takes the {label} road', passes is not None, str(why))
    sim = None
    if passes is not None:
        sim, _c = GSH.simulate(job, gb, passes, atl)
        sim = np.asarray(sim)[..., :3]
    return cpu, sim, m, passes, why


def pass_sources(passes):
    """Every fragment source the plan carries (main passes and bump
    pre-passes), the R252 walk."""
    srcs = []

    def walk(item):
        if isinstance(item, str):
            if 'void main' in item:
                srcs.append(item)
        elif isinstance(item, dict):
            for v in item.values():
                walk(v)
        elif isinstance(item, (tuple, list)):
            for v in item:
                walk(v)
    walk(passes or ())
    return srcs


def defs(t, name):
    return len(re.findall(r'^\s*(?:vec3|vec4)\s+' + re.escape(name)
                          + r'\s*\(float obj\)', t, re.M))


def bits_equal(a, b):
    a = np.ascontiguousarray(np.asarray(a, f32))
    b = np.ascontiguousarray(np.asarray(b, f32))
    return a.shape == b.shape and bool(np.array_equal(a.view(np.uint32),
                                                      b.view(np.uint32)))


# ----------------------------------------------------------------- tests

def test_a_the_one_chain_is_bitwise_the_simulator_dot_and_the_einsum():
    """core/mathx.object_space_points is the sequential float32 chain
    the simulator's dot evaluates for dot(hal_obj_rN(td.y), vec4(P,1))
    -- and the einsum n_tex_coord ran before R253, so no Object-output
    picture moves. 200k points, three matrices gathered per point."""
    rng = np.random.default_rng(1)
    n = 200000
    P = (rng.random((n, 3), f32) * 10 - 5).astype(f32)
    mats = np.stack([np.linalg.inv(MOVE).astype(f32),
                     np.eye(4, dtype=f32),
                     np.linalg.inv(xform((2, -1, 0.5), -1.1, (0.4, 2.5, 1.0)))
                     .astype(f32)])
    idx = rng.integers(0, 3, n)
    g = mats[idx]
    got = M.object_space_points(P, g)
    P4 = np.concatenate([P, np.ones((n, 1), f32)], 1)
    sim = np.stack([SB.dot_(g[:, r, :], P4) for r in range(3)], 1).astype(f32)
    ein = (np.einsum('nij,nj->ni', g[:, :3, :3], P) + g[:, :3, 3]).astype(f32)
    check('object_space_points is BITWISE the simulator dot (np.sum of four '
          'float32 products) on 200k gathered points', bits_equal(got, sim))
    check('...and BITWISE the einsum the Object output used before R253',
          bits_equal(got, ein))
    check('a single (4,4) inverse broadcasts to the same bits',
          bits_equal(M.object_space_points(P[idx == 0], mats[0]), got[idx == 0]))
    check('at the identity it returns P bit for bit (1*x + 0 + 0 + 0)',
          bits_equal(M.object_space_points(P, np.eye(4, dtype=f32)), P))
    # the inverse undoes the matrix to float32 precision on the ball
    L = (rng.random((4096, 3), f32) * 2 - 1).astype(f32) + BALL_C
    Wd = (L @ MOVE[:3, :3].T + MOVE[:3, 3]).astype(f32)
    back = M.object_space_points(Wd, np.linalg.inv(MOVE).astype(f32))
    check('inv(M) . (M . L) returns L within 2e-5 (float32 through both)',
          float(np.abs(back - L).max()) < 2e-5,
          f'{float(np.abs(back - L).max()):.2e}')


def test_b_generated_follows_the_moved_object():
    """The law the user asked for: a ball moved, turned about Z and
    scaled non-uniformly keeps the SAME Generated coordinate at every
    surface point it had at the identity (within float32 through M and
    inv(M)); the exporter's gen_bounds road gives the same; a pure
    translation is equal within 1e-6; the legacy WORLD road reproduces
    the old scrolling numbers bitwise and differs from the fix."""
    st = _settings()
    sc0 = _scene(st)
    _j0, sel0, _b0, c0 = ball_ctx(sc0, st)
    g0 = c0.generated
    span0 = g0.max(0) - g0.min(0)
    check('the identity ball spans the full 0..1 box (the per-object law '
          'of test_generated_coords_are_per_object)', float(span0.min()) > 0.9,
          str(np.round(span0, 3)))
    # the derived road: gen_bounds None, world vertices + matrix
    sc1 = move_ball(_scene(st), MOVE)
    _j1, sel1, _b1, c1 = ball_ctx(sc1, st)
    check('the same triangles are shaded', np.array_equal(sel0, sel1))
    d = float(np.abs(c1.generated - g0).max())
    check('moved + turned + scaled (derived from the vertices): Generated '
          'equals the identity numbers within 2e-5', d < 2e-5, f'{d:.2e}')
    check('...so the box the fragments see is the object-space box, not '
          'the world AABB',
          float(np.abs(c1.generated.max(0) - g0.max(0)).max()) < 2e-5
          and float(np.abs(c1.generated.min(0) - g0.min(0)).max()) < 2e-5)
    # the export road: the box handed in from the untransformed mesh
    vert_obj = np.zeros(sc0.mesh.verts.shape[0], np.int32)
    vert_obj[sc0.mesh.tris.reshape(-1)] = np.repeat(sc0.mesh.obj_index, 3)
    local = sc0.mesh.verts[vert_obj == 1]
    gb = (local.min(0).astype(f32), local.max(0).astype(f32))
    sc2 = move_ball(_scene(st), MOVE, gen_bounds=gb)
    _j2, _s2, _b2, c2 = ball_ctx(sc2, st)
    d2 = float(np.abs(c2.generated - g0).max())
    check('moved + turned + scaled with the exporter\'s gen_bounds: '
          'equal within 2e-5', d2 < 2e-5, f'{d2:.2e}')
    jlo, jspan = _j2.object_generated_frame()
    check('object_generated_frame hands the exported box through verbatim',
          bits_equal(jlo[1], gb[0])
          and bits_equal(jspan[1], np.maximum(gb[1] - gb[0], 1e-6)))
    # pure translation
    sc3 = move_ball(_scene(st), MOVE_T)
    _j3, _s3, _b3, c3 = ball_ctx(sc3, st)
    d3 = float(np.abs(c3.generated - g0).max())
    check('a pure translation: equal within 1e-6', d3 < 1e-6, f'{d3:.2e}')
    # the legacy road on the moved scene: the old numbers, bitwise
    stw = _settings(generated_space='WORLD')
    jw, selw, baryw, cw = ball_ctx(sc1, stw)
    lo_w, span_w = jw.object_bounds()
    oi = sc1.mesh.obj_index[selw]
    P = cw.P
    old = ((P - lo_w[oi]) / span_w[oi]).astype(f32)
    check("generated_space WORLD reproduces the pre-1.92 formula "
          "((P - lo_world) / span_world) BITWISE on the moved ball",
          bits_equal(cw.generated, old))
    dd = float(np.abs(cw.generated - c1.generated).max())
    check('...and the OBJECT road differs from it by more than 0.05 '
          'somewhere (the scroll the user saw)', dd > 0.05, f'{dd:.3f}')
    check('the WORLD road keeps the world bounds on the context too',
          bits_equal(cw.obj_bounds[0], lo_w) and bits_equal(c1.obj_bounds[0], lo_w))
    # the frames differ on a moved ball (the ab row's DIFF, at context level)
    stm = _settings()
    sc4 = move_ball(_scene(stm, marble_graph()), rot_z(0.7, (0.0, 0.0, 0.0)),
                    move_verts=False)
    img_o = np.asarray(R.render(sc4, stm))[..., :3]
    stm_w = _settings(generated_space='WORLD')
    img_w = np.asarray(R.render(sc4, stm_w))[..., :3]
    check("a marble on a ball whose matrix is a rot_z: 'OBJECT' and 'WORLD' "
          "frames differ (the setting's DIFF)",
          float(np.abs(img_o - img_w).max()) > 1e-3)


def test_c_identity_scenes_are_bitwise_neutral():
    """Default-neutrality: with every matrix the identity the new
    ctx.generated is BITWISE the old formula, the Object output is
    bitwise the einsum, and the whole demo frame with a Generated-driven
    marble renders the same bits under OBJECT and WORLD."""
    st = _settings()
    sc = _scene(st, marble_graph())
    job, sel, bary, c = ball_ctx(sc, st)
    lo, span = job.object_bounds()
    oi = sc.mesh.obj_index[sel]
    old = ((c.P - lo[oi]) / span[oi]).astype(f32)
    check('identity: ctx.generated is BITWISE ((P - lo) / span) from '
          'object_bounds()', bits_equal(c.generated, old))
    glo, gspan = job.object_generated_frame()
    check('identity: the object frame IS the world rows (the same bits)',
          bits_equal(glo, lo) and bits_equal(gspan, span))
    check('identity: _object_space hands P back untouched',
          job._object_space(c.P, oi) is c.P)
    # the Object output: the chain against the einsum it replaced
    from ..core import nodeeval as NE
    tc = _wnode('tc', 'ShaderNodeTexCoord', {}, [],
                [{'name': n, 'type': 'VECTOR'} for n in
                 ('Generated', 'Normal', 'UV', 'Object', 'Camera', 'Window',
                  'Reflection')])
    scm = move_ball(_scene(st), MOVE)
    _jm, _sm, _bm, cm = ball_ctx(scm, st)
    inv = cm.object_matrix_inv
    ein = (np.einsum('nij,nj->ni', inv[:, :3, :3], cm.P) + inv[:, :3, 3]).astype(f32)
    out = NE.n_tex_coord(type('E', (), {'ctx': cm})(), tc)
    check('the Object output on a moved ball is BITWISE the einsum it ran '
          'before R253', bits_equal(out['Object'], ein))
    check('...and the Object output is the Generated point before the box '
          '(one chain for both: (Object - lo) / span == Generated bitwise)',
          bits_equal(((out['Object'] - _jm.object_generated_frame()[0][1])
                      / _jm.object_generated_frame()[1][1]).astype(f32),
                     cm.generated))
    # the whole frame: the same bits on both roads at the identity
    img_o = np.asarray(R.render(sc, st))
    sc_w = _scene(_settings(generated_space='WORLD'), marble_graph())
    img_w = np.asarray(R.render(sc_w, _settings(generated_space='WORLD')))
    check('the demo frame with a Generated marble is BITWISE the same under '
          "'OBJECT' and 'WORLD' when no object is transformed",
          bits_equal(img_o, img_w))
    sc_d = demo_scene(_settings(), with_texture=True)
    sc_dw = demo_scene(_settings(generated_space='WORLD'), with_texture=True)
    check('...and so is the textured demo card (an image on Generated)',
          bits_equal(np.asarray(R.render(sc_d, _settings())),
                     np.asarray(R.render(sc_dw, _settings(generated_space='WORLD')))))
    check("RenderSettings defaults generated_space to 'OBJECT' (the fix)",
          RenderSettings().generated_space == 'OBJECT')


def test_d_gpu_twin_on_moved_objects():
    """The GPU road on a moved + turned + scaled ball: the plan is
    taken, the pass bakes hal_lgen_lo / hal_lgen_span and reads the
    hal_obj_rN rows (defined exactly ONCE even when the Object output
    reads them too), the simulated frame twins the CPU within 6e-3 (the
    R243 bar for the three dots on a driver), and the generated LINE,
    compiled standalone on the context's own P and object ids, is
    BITWISE the CPU's ctx.generated. The legacy WORLD road keeps its
    text and twins too, and the flip re-plans."""
    st = _settings()
    sc = move_ball(_scene(st, marble_graph()), MOVE)
    cpu, sim, m, passes, _why = gpu_road(sc, st, 'moved-ball marble')
    if passes is None:
        return
    err = float(np.abs(sim[m] - cpu[m]).max())
    check('...and twins the CPU on it within 6e-3', err < 6e-3, f'{err:.2e}')
    srcs = pass_sources(passes)
    gen_srcs = [s for s in srcs if 'hal_generated' in s]
    check('the pass reads hal_generated', bool(gen_srcs), f'{len(srcs)} sources')
    check('every pass reading Generated defines hal_lgen_lo, hal_lgen_span '
          'and hal_obj_r0..2 exactly ONCE',
          all(defs(s, 'hal_lgen_lo') == 1 and defs(s, 'hal_lgen_span') == 1
              and all(defs(s, f'hal_obj_r{r}') == 1 for r in range(3))
              for s in gen_srcs))
    check('the line is the object-space one (the three dots over the '
          'object box), not the world line',
          all('hal_obj_r0(td.y), vec4(P, 1.0)), dot(hal_obj_r1' in s
              and '- hal_lgen_lo(td.y)) / hal_lgen_span(td.y);' in s
              and '(P - hal_gen_lo(td.y))' not in s for s in gen_srcs))
    check('gpu/device.duplicate_definitions finds nothing to refuse',
          all(DEV.duplicate_definitions(s) == [] for s in srcs))

    # Object output AND Generated in one material: the rows once
    sc2 = move_ball(_scene(st, object_and_generated_graph()), MOVE)
    cpu2, sim2, m2, passes2, _w2 = gpu_road(sc2, st, 'Object-output + Generated')
    if passes2 is not None:
        err2 = float(np.abs(sim2[m2] - cpu2[m2]).max())
        check('...twins the CPU within 6e-3', err2 < 6e-3, f'{err2:.2e}')
        srcs2 = [s for s in pass_sources(passes2) if 'hal_object' in s]
        check('a pass reading hal_object AND hal_generated defines each '
              'hal_obj_rN row exactly once and carries both lines',
              bool(srcs2) and all(
                  all(defs(s, f'hal_obj_r{r}') == 1 for r in range(3))
                  and 'vec3 hal_object = vec3(dot(hal_obj_r0' in s
                  and 'vec3 hal_generated = (vec3(dot(hal_obj_r0' in s
                  and DEV.duplicate_definitions(s) == [] for s in srcs2))

    # the function level: the generated line, standalone, is bitwise
    job, _vp = _job(sc, st)
    consts = {'obj_inv': job.object_matrices(),
              'obj_gen': job.object_generated_frame(),
              'obj_bounds': job.object_bounds(),
              'generated_space': 'OBJECT'}
    fns, line, gerr = GM._generated_frame('hal_generated', consts)
    rows, _ol = GM._object_frame('', consts, need_rows=True)
    check('_generated_frame hands the table and the line, _object_frame '
          'the rows alone when asked', fns and line and gerr is None
          and rows and _ol == '' and 'hal_object' not in rows)
    src = (rows + fns + 'uniform vec3 P;\nuniform vec2 td;\nout vec4 Color;\n'
           'void main() {\n' + line + '    Color = vec4(hal_generated, 1.0);\n}\n')
    prog, perr = try_compile(src, 'GLSL')
    check('the generated line compiles standalone through the front-end',
          prog is not None, str(perr))
    if prog is not None:
        sel = np.nonzero(sc.mesh.mat_index >= 0)[0]
        rng = np.random.default_rng(5)
        bary = rng.random((sel.size, 3)).astype(f32)
        bary /= bary.sum(1, keepdims=True)
        ctx = job.context(sel, bary)
        n = sel.size
        td = np.zeros((n, 2), f32)
        td[:, 1] = sc.mesh.obj_index[sel].astype(f32)
        out = np.asarray(prog.run({'P': ctx.P, 'td': td}, {}, n)[0]['Color'],
                         f32)[:, :3]
        check('...and its output is BITWISE the CPU ctx.generated over every '
              'triangle of the scene (0.0, three objects, one moved)',
              bits_equal(out, ctx.generated),
              f'{float(np.abs(out - ctx.generated).max()):.2e}')

    # the legacy road on the GPU: today's text, and a re-plan on the flip
    stw = _settings(generated_space='WORLD')
    scw = move_ball(_scene(stw, marble_graph()), MOVE)
    jo, _v = _job(sc, st)
    jw, _v = _job(scw, stw)
    mk = GSH._mesh_key(sc.mesh)
    check("the plan signature changes on the flip (R78: a bake the plan "
          "reads MUST be in it)", GSH._plan_sig(jo, mk) != GSH._plan_sig(jw, mk))
    cpuw, simw, mw, passesw, _ww = gpu_road(scw, stw, 'legacy WORLD marble',
                                            clear=False)
    if passesw is not None:
        errw = float(np.abs(simw[mw] - cpuw[mw]).max())
        check('...twins the CPU within 6e-3 on the legacy road', errw < 6e-3,
              f'{errw:.2e}')
        srcw = [s for s in pass_sources(passesw) if 'hal_generated' in s]
        check('the legacy pass carries the pre-1.92 text verbatim: '
              '(P - hal_gen_lo(td.y)) / hal_gen_span(td.y), no object table, '
              'no rows',
              bool(srcw) and all(
                  'vec3 hal_generated = (P - hal_gen_lo(td.y)) / hal_gen_span(td.y);' in s
                  and 'hal_lgen_lo' not in s and 'hal_obj_r0' not in s
                  for s in srcw))
        check('...and the two roads differ on the GPU as on the CPU',
              float(np.abs(simw[mw] - sim[m]).max()) > 1e-3)


def test_e_bump_height_prepass_on_a_moved_object():
    """A Bump whose Height is a Generated-driven Marble on the moved
    ball: the plan is taken, the height pre-pass bakes the object-space
    table (the _assemble_height_pass road), and the frame twins within
    6e-3 (the bump-layer bar)."""
    st = _settings()
    sc = move_ball(bump_scene(st), MOVE)
    cpu, sim, m, passes, _why = gpu_road(sc, st, 'moved-ball bump pre-pass')
    if passes is None:
        return
    err = float(np.abs(sim[m] - cpu[m]).max())
    check('...and twins the CPU within 6e-3', err < 6e-3, f'{err:.2e}')
    srcs = [s for s in pass_sources(passes) if 'hal_generated' in s]
    check('the height pre-pass AND the main pass read Generated (two sources)',
          len(srcs) >= 2, f'{len(srcs)}')
    check('each bakes hal_lgen_lo / hal_lgen_span and the rows exactly once',
          all(defs(s, 'hal_lgen_lo') == 1 and defs(s, 'hal_lgen_span') == 1
              and all(defs(s, f'hal_obj_r{r}') == 1 for r in range(3))
              and DEV.duplicate_definitions(s) == [] for s in srcs))
    # the pre-pass on an identity ball is unchanged text-wise from the
    # world road except for the names: the same bits on the CPU
    st_i = _settings()
    sc_i = bump_scene(st_i)
    sc_w = bump_scene(_settings(generated_space='WORLD'))
    check('the bump frame on the identity ball is BITWISE the same under '
          "'OBJECT' and 'WORLD'",
          bits_equal(np.asarray(R.render(sc_i, st_i)),
                     np.asarray(R.render(sc_w, _settings(generated_space='WORLD')))))


def test_f_manual_texture_space_is_honoured():
    """ObjectInfo.gen_bounds (the exporter's Texture Space road: loc -
    size .. loc + size) is the box, verbatim: a unit ball handed a
    4-unit box about its centre spans 0.25..0.75, on both devices."""
    st = _settings()
    sc = _scene(st, marble_graph())
    sc.objects[1].gen_bounds = ((BALL_C - 2.0).astype(f32), (BALL_C + 2.0).astype(f32))
    job, sel, bary, c = ball_ctx(sc, st)
    g = c.generated
    # bary 1/3 shades triangle centroids, which sit just inside the
    # radius on a 24 x 16 tessellation: 0.25..0.75 within 1e-2
    check('a unit ball in a 4-unit box spans 0.25..0.75 in every axis',
          float(np.abs(g.min(0) - 0.25).max()) < 1e-2
          and float(np.abs(g.max(0) - 0.75).max()) < 1e-2,
          f'{np.round(g.min(0), 3)}..{np.round(g.max(0), 3)}')
    glo, gspan = job.object_generated_frame()
    check('the frame carries the handed box (lo = c - 2, span = 4)',
          bits_equal(glo[1], (BALL_C - 2.0).astype(f32))
          and bits_equal(gspan[1], np.full(3, 4.0, f32)))
    check('...while the other objects keep their own derived boxes',
          bits_equal(glo[0], job.object_bounds()[0][0])
          and bits_equal(glo[2], job.object_bounds()[0][2]))
    cpu, sim, m, passes, _why = gpu_road(sc, st, 'manual texture space')
    if passes is not None:
        err = float(np.abs(sim[m] - cpu[m]).max())
        check('...and the GPU twins it within 6e-3', err < 6e-3, f'{err:.2e}')
        srcs = [s for s in pass_sources(passes) if 'hal_lgen_lo' in s]
        check('the baked table carries the handed box as literals',
              bool(srcs) and all(
                  'vec3 hal_lgen_lo(float obj)' in s and '-3.29999995' in s
                  for s in srcs))
    # a degenerate (flat) axis keeps Halcyon's 1e-6 clamp, not Blender's size-1
    sc2 = _scene(st)
    sc2.objects[1].gen_bounds = (np.array([0, 0, 1.0], f32), np.array([1, 1, 1.0], f32))
    j2, _s, _b, _c = ball_ctx(sc2, st)
    check('a flat box axis keeps the max(hi - lo, 1e-6) clamp (the documented '
          'divergence from Blender, for neutrality on flat planes)',
          bits_equal(j2.object_generated_frame()[1][1],
                     np.array([1, 1, 1e-6], f32)))
    # a hair part / a hand-built object: None derives from the mesh
    check('gen_bounds defaults to None on ObjectInfo (derive from the mesh)',
          ObjectInfo(name='x').gen_bounds is None)


def test_g_cartoon_centre_still_measures_world_space():
    """The cartoon's shape smoothing (and the hair shine's azimuth) read
    the WORLD bounds centre -- a direction from the world point to the
    world centre is correct as-is -- so on a translated + turned ball
    the CPU's cartoon_smooth_normal is BITWISE unchanged by the fix, and
    the GPU pass still carries hal_ccen from hal_gen_lo( (the names
    30116 pins) beside the object-space Generated line."""
    st_o = _settings()
    st_w = _settings(generated_space='WORLD')
    gph = _cartoon_graph(over={'Shadow Smoothing': 0.8})
    sc_o = move_ball(_scene(st_o, gph), rot_z(0.5, (0.3, -0.2, 0.4)))
    sc_w = move_ball(_scene(st_w, gph), rot_z(0.5, (0.3, -0.2, 0.4)))
    _jo, _so, _bo, co = ball_ctx(sc_o, st_o)
    _jw, _sw, _bw, cw = ball_ctx(sc_w, st_w)
    check('ctx.obj_bounds is the WORLD table on both roads (the same bits)',
          bits_equal(co.obj_bounds[0], cw.obj_bounds[0])
          and bits_equal(co.obj_bounds[1], cw.obj_bounds[1]))
    amt = np.full(co.n, 0.8, f32)
    n_o = R.cartoon_smooth_normal(co.N, co, amt)
    n_w = R.cartoon_smooth_normal(cw.N, cw, amt)
    check('cartoon_smooth_normal on the moved ball is BITWISE the legacy '
          'result (it never read Generated)', bits_equal(n_o, n_w))
    check('...while Generated itself moved with the object',
          float(np.abs(co.generated - cw.generated).max()) > 0.05)
    cpu, sim, m, passes, _why = gpu_road(sc_o, st_o, 'cartoon smoothing on a moved ball')
    if passes is not None:
        err = float(np.abs(sim[m] - cpu[m]).max())
        check('...and twins the CPU within 6e-3', err < 6e-3, f'{err:.2e}')
        srcs = pass_sources(passes)
        check("the pass carries 'hal_ccen' from 'hal_gen_lo(' (the world "
              "table, names pinned by test_cartoon_shader_gpu_parity)",
              any('hal_ccen' in s and 'hal_gen_lo(' in s for s in srcs))
        check('...and defines hal_gen_lo / hal_gen_span exactly once beside '
              'the object table', all(
                  defs(s, 'hal_gen_lo') <= 1 and defs(s, 'hal_gen_span') <= 1
                  and DEV.duplicate_definitions(s) == [] for s in srcs))


def test_h_export_carries_the_object_space_box():
    """The export road (bpy-free fakes): _mesh_arrays measures the box
    on the UNTRANSFORMED vertices and returns it as gen_bounds; a manual
    Texture Space (use_auto_texspace False) gives loc - size .. loc +
    size; _info_object lands it on ObjectInfo.gen_bounds (and still
    takes two arguments); export_scene hands a moved object world
    vertices AND its own box."""
    from . import fakeblender as FB
    props, _engine = FB.install()
    from .. import export as EX
    Mx = xform((2.0, -1.0, 0.5), 0.9, (2.0, 0.5, 1.5))
    d = EX._mesh_arrays(FB.cube_mesh(), Mx, 0, 0)
    check('_mesh_arrays returns gen_bounds = the local co.min / co.max '
          '(the unit cube: -1..1)',
          d is not None and 'gen_bounds' in d
          and bits_equal(d['gen_bounds'][0], np.full(3, -1.0, f32))
          and bits_equal(d['gen_bounds'][1], np.full(3, 1.0, f32)))
    check('...while the vertices it returns are WORLD (through the matrix)',
          d is not None and float(np.abs(d['verts'].max(0)
                                         - np.array([1, 1, 1], f32)).max()) > 0.5)

    def tex_mesh():
        me = FB.cube_mesh()
        me.use_auto_texspace = False
        me.texspace_location = (0.5, 0.0, -0.25)
        me.texspace_size = (2.0, 1.5, 1.0)
        return me
    d2 = EX._mesh_arrays(tex_mesh(), np.eye(4, dtype=f32), 0, 0)
    check('with use_auto_texspace False it returns (loc - size, loc + size)',
          d2 is not None
          and bits_equal(d2['gen_bounds'][0], np.array([-1.5, -1.5, -1.25], f32))
          and bits_equal(d2['gen_bounds'][1], np.array([2.5, 1.5, 0.75], f32)))

    def auto_mesh():
        me = FB.cube_mesh()
        me.use_auto_texspace = True
        me.texspace_location = (9.0, 9.0, 9.0)
        me.texspace_size = (9.0, 9.0, 9.0)
        return me
    d3 = EX._mesh_arrays(auto_mesh(), np.eye(4, dtype=f32), 0, 0)
    check('with use_auto_texspace True the manual values are ignored',
          d3 is not None and bits_equal(d3['gen_bounds'][0], np.full(3, -1.0, f32)))
    info = {'name': 'Cube', 'color': (1, 1, 1, 1), 'index': 0, 'random': 0.0,
            'visible_camera': True, 'cast_shadow': True, 'holdout': False,
            'smoothresh': 0.0}
    o2 = EX._info_object(info, np.eye(4, dtype=f32))
    o3 = EX._info_object(info, Mx, d['gen_bounds'])
    check('_info_object still takes two arguments (gen_bounds None)',
          o2.gen_bounds is None and o2.name == 'Cube')
    check('...and lands a handed box on ObjectInfo.gen_bounds',
          o3.gen_bounds is not None and bits_equal(o3.gen_bounds[0], d['gen_bounds'][0])
          and bits_equal(o3.matrix_world, Mx))
    # the whole export of a moved object and a manual-texspace one
    cube = FB.geometry_object('R253Cube', FB.cube_mesh, matrix=Mx, kind='MESH')
    cube2 = FB.geometry_object('R253TexCube', tex_mesh, kind='MESH')
    warn = []
    sc = EX.export_scene(FB.depsgraph_of(props, [cube, cube2]), RenderSettings(), warn)
    check('export_scene exports both', len(sc.objects) == 2, str(warn))
    if len(sc.objects) == 2:
        a, b = sc.objects
        check('the moved cube carries its matrix, WORLD vertices and the '
              'object-space box -1..1',
              bits_equal(a.matrix_world, Mx) and a.gen_bounds is not None
              and bits_equal(a.gen_bounds[0], np.full(3, -1.0, f32))
              and bits_equal(a.gen_bounds[1], np.full(3, 1.0, f32))
              and float(sc.mesh.verts[sc.mesh.obj_index[0] == 0].max()) > 1.5
              if sc.mesh.obj_index is not None else False)
        check('the manual-texspace cube carries loc - size .. loc + size',
              b.gen_bounds is not None
              and bits_equal(b.gen_bounds[0], np.array([-1.5, -1.5, -1.25], f32))
              and bits_equal(b.gen_bounds[1], np.array([2.5, 1.5, 0.75], f32)))
        # and the renderer measures Generated over that box, not the world AABB
        st = _settings()
        view = np.eye(4, dtype=f32)
        job = R.ShadeJob(sc, st, {}, None, view, np.zeros(3, f32), W, H)
        glo, gspan = job.object_generated_frame()
        check('ShadeJob.object_generated_frame reads the exported boxes',
              bits_equal(glo[0], np.full(3, -1.0, f32))
              and bits_equal(gspan[0], np.full(3, 2.0, f32))
              and bits_equal(glo[1], np.array([-1.5, -1.5, -1.25], f32))
              and bits_equal(gspan[1], np.array([4.0, 3.0, 2.0], f32)))
        tri = np.nonzero(sc.mesh.obj_index == 0)[0]
        ctx = job.context(tri, np.full((tri.size, 3), 1 / 3.0, f32))
        check('the moved cube\'s Generated stays inside 0..1 (its own box) '
              'though its world AABB is elsewhere',
              float(ctx.generated.min()) > -1e-4 and float(ctx.generated.max()) < 1 + 1e-4,
              f'{ctx.generated.min():.3f}..{ctx.generated.max():.3f}')


def test_i_the_setting_surface_and_the_refusal():
    """generated_space on the panel: an enum of two items with 12-char
    descriptions, a 40-char tooltip, a label, the plan signature, the
    Textures panel row; the GPU refuses BY NAME when no frame is
    supplied; the setting is read by the engine (render.py context and
    shade.py consts)."""
    import os
    from .. import properties as PR
    items = PR.ENUMS.get('generated_space')
    check("'generated_space' is in properties.ENUMS with OBJECT (default) "
          "and WORLD", items is not None
          and [i[0] for i in items] == ['OBJECT', 'WORLD'])
    check('every item has a description of 12+ characters',
          items is not None and all(len(i[2]) >= 12 for i in items))
    check("its tooltip is 40+ characters and names both boxes",
          len(PR.DESCRIPTIONS.get('generated_space', '')) >= 40
          and 'world' in PR.DESCRIPTIONS['generated_space'].lower()
          and 'object' in PR.DESCRIPTIONS['generated_space'].lower())
    check("it has a label", PR.LABELS.get('generated_space') == 'Generated Space')
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, 'gpu', 'shade.py'), encoding='utf8') as fh:
        shade_src = fh.read()
    with open(os.path.join(root, 'ui.py'), encoding='utf8') as fh:
        ui_src = fh.read()
    with open(os.path.join(root, 'core', 'render.py'), encoding='utf8') as fh:
        render_src = fh.read()
    sig_at = shade_src.index('def _plan_sig')
    sig_body = shade_src[sig_at:shade_src.index('def ', sig_at + 10)]
    check("'generated_space' is in gpu/shade._plan_sig (a bake the plan reads)",
          "'generated_space'" in sig_body)
    check("the Textures panel draws the row", "col.prop(hs, 'generated_space')" in ui_src)
    check("core/render.py reads the setting in context()",
          "getattr(self.settings, 'generated_space', 'OBJECT')" in render_src)
    # refusal by name
    fns, line, err = GM._generated_frame('hal_generated', {'obj_inv': np.eye(4, dtype=f32)[None]})
    check('without a frame the GPU refuses BY NAME',
          fns is None and line is None
          and err == 'generated coordinates need the per-object frame the '
                     'caller did not supply', str(err))
    fns_w, line_w, err_w = GM._generated_frame(
        'hal_generated', {'generated_space': 'WORLD'})
    check('...and the legacy road refuses by its own name without bounds',
          fns_w is None and 'per-object bounds' in str(err_w), str(err_w))
    none = GM._generated_frame('nothing here', {})
    check('a pass reading no Generated gets no table and no line',
          none == ('', '', None))
    rows, oline = GM._object_frame('nothing here', {'obj_inv': None})
    check('_object_frame with nothing to do stays silent', (rows, oline) == ('', ''))
    rows2, oline2 = GM._object_frame('nothing here', {'obj_inv': None}, need_rows=True)
    check('...and asked for rows without matrices refuses by name',
          rows2 is None and 'per-object matrices' in str(oline2))
    # the enum items load from an old file: absent -> the default
    st = RenderSettings()
    check("a settings object without the field resolves to 'OBJECT' through "
          "the getattr default the readers use",
          str(getattr(st, 'generated_space', 'OBJECT')) == 'OBJECT')


def test_j_prewarm_and_no_object_scenes():
    """prewarm() builds the object frame before any worker starts; a
    scene with no ObjectInfo list (a bare mesh) still shades, with the
    world rows (one identity object)."""
    st = _settings()
    sc = move_ball(_scene(st), MOVE)
    job, _vp = _job(sc, st)
    check('the frame is lazy until asked', job._obj_gen is None)
    job.prewarm()
    check('prewarm() builds it', job._obj_gen is not None and job._obj_gen_identity is False)
    sc2 = _scene(st)
    sc2.objects = []
    job2, _vp = _job(sc2, st)
    sel = np.nonzero(sc2.mesh.mat_index == 1)[0]
    ctx = job2.context(sel, np.full((sel.size, 3), 1 / 3.0, f32))
    lo, span = job2.object_bounds()
    oi = np.clip(sc2.mesh.obj_index[sel], 0, lo.shape[0] - 1)
    check('a scene with no objects shades Generated over the world rows '
          '(bitwise the old formula)',
          bits_equal(ctx.generated, ((ctx.P - lo[oi]) / span[oi]).astype(f32)))


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
    print('all R253 generated-coords checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
