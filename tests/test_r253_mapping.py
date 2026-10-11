"""R253 (1.92.0): more mapping types -- Tri-Planar, cube atlas, the sphere
conventions, screen / reflection / normal / tangent texco, object
projection and the hex grid.

The Projector node keeps its four shipped modes byte-for-byte (CPU
digests and the emitted GLSL text are pinned here, the luma64 precedent)
and grows nine appended modes; Tri-Planar Coordinates + Tri-Planar Mix
are the node-graph form of tri-planar texturing, Tri-Planar Texture the
one-node form that owns the image; Object Projection measures a point
in a picked object's frame (local, perspective through its lens, or
parallel); Hex Grid tiles hexagonal cells. The Image Texture's Sphere /
Tube / Box projections finally reach the GPU (they sampled FLAT before).

Every heading proves one law: the function-level twin through the GLSL
simulator on the SAME float32 lanes (bitwise where the design says
bitwise -- abs / max / floor / fract / mul / add only -- and <= 2e-6 for
the transcendental modes), the refusals by name, the tables the census
tests read, the frame-level twin on the demo scene (the deferred bar
6e-3, no pixel > 1e-2), and default neutrality: the shipped Projector
modes and the pre-existing featurematrix pictures are the pre-edit
bytes.

    python -m halcyon.tests.test_r253_mapping
"""
import math
import sys
import traceback
import zlib

import numpy as np

from . import utf8_console
from .scenebuild import checker_image, demo_scene
from .test_render import _ctx, _sk, _wnode, base_settings
from .featurematrix import _one_bsdf_graph
from . import featurematrix as FM
from .test_r251_material_nodes import _master_ins, frame_bar, node_twin
from ..core import mathx as M
from ..core import nodeeval as NE
from ..core import render as R
from ..core import texture as TX
from ..core.nodeeval import GraphEvaluator
from ..core.scene import ImageBuffer
from ..gpu import emit as EM
from ..shaders import builtins as SB

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


# ------------------------------------------------------------ the pins
# Computed on the pre-edit tree (claude-cloud b0cbe1c, the 1.91 code):
# adler32 of n_v16_projector's float32 output on LANE_N random vectors
# (seed 253, -2..2) with Scale 0.7, per mode and axis.
OLD_CPU_DIGESTS = {
    ('PLANAR', 'X'): 3602100279, ('PLANAR', 'Y'): 1516742640,
    ('PLANAR', 'Z'): 1123977831,
    ('CYLINDER', 'X'): 2503073783, ('CYLINDER', 'Y'): 1229782755,
    ('CYLINDER', 'Z'): 2145190063,
    ('SPHERE', 'X'): 3527061326, ('SPHERE', 'Y'): 1703514277,
    ('SPHERE', 'Z'): 2050902072,
    ('BOX', 'X'): 1714368869, ('BOX', 'Y'): 1714368869,
    ('BOX', 'Z'): 1714368869,
}
#: e_v16_projector's emitted body (Vector linked to a vec3 lane named
#: hal_lane, Scale 0.7), the 1.91 text verbatim for axis Z ...
OLD_GLSL_Z = {
    'PLANAR': (
        '    vec3 _v1 = hal_lane;\n'
        '    float _v2 = max(0.7, 1e-9);\n'
        '    float _v3 = _v1.x;\n'
        '    float _v4 = _v1.y;\n'
        '    float _v5 = _v1.z;\n'
        '    vec3 _v6 = vec3(_v3 / _v2, _v4 / _v2, 0.0);'),
    'CYLINDER': (
        '    vec3 _v1 = hal_lane;\n'
        '    float _v2 = max(0.7, 1e-9);\n'
        '    float _v3 = _v1.x;\n'
        '    float _v4 = _v1.y;\n'
        '    float _v5 = _v1.z;\n'
        '    float _v6 = fract(atan(_v4, _v3) / 6.28318530717959 + 1.0);\n'
        '    vec3 _v7 = vec3(_v6 / _v2, _v5 / _v2, 0.0);'),
    'SPHERE': (
        '    vec3 _v1 = hal_lane;\n'
        '    float _v2 = max(0.7, 1e-9);\n'
        '    float _v3 = _v1.x;\n'
        '    float _v4 = _v1.y;\n'
        '    float _v5 = _v1.z;\n'
        '    float _v6 = length(vec3(_v3, _v4, _v5));\n'
        '    float _v7 = fract(atan(_v4, _v3) / 6.28318530717959 + 1.0);\n'
        '    float _v8 = acos(clamp(_v5 / max(_v6, 1e-9), -1.0, 1.0)) '
        '/ 3.14159265358979;\n'
        '    vec3 _v9 = vec3(_v7 / _v2, (1.0 - _v8) / _v2, 0.0);'),
    'BOX': (
        '    vec3 _v1 = hal_lane;\n'
        '    float _v2 = max(0.7, 1e-9);\n'
        '    float _v3 = _v1.x;\n'
        '    float _v4 = _v1.y;\n'
        '    float _v5 = _v1.z;\n'
        '    vec3 _v6 = abs(_v1);\n'
        '    float _v7 = (_v6.x >= max(_v6.y, _v6.z)) ? _v1.y : '
        '((_v6.y >= _v6.z) ? _v1.x : _v1.x);\n'
        '    float _v8 = (_v6.x >= max(_v6.y, _v6.z)) ? _v1.z : '
        '((_v6.y >= _v6.z) ? _v1.z : _v1.y);\n'
        '    vec3 _v9 = vec3(_v7 / _v2, _v8 / _v2, 0.0);'),
}
#: ... and its adler32 for axes X and Y
OLD_GLSL_XY_DIGESTS = {
    ('PLANAR', 'X'): 2693475652, ('PLANAR', 'Y'): 2690460996,
    ('CYLINDER', 'X'): 752500016, ('CYLINDER', 'Y'): 749485360,
    ('SPHERE', 'X'): 2579521321, ('SPHERE', 'Y'): 2576506665,
    ('BOX', 'X'): 1030317123, ('BOX', 'Y'): 1027302467,
}
#: e_tex_image's FLAT text (unlinked Vector, an image assigned)
OLD_TEX_IMAGE_FLAT = ('    vec3 _v1 = vec3(hal_uv, 0.0);\n'
                      '    vec4 _v2 = texture(hal_tex0, _v1.xy);')
#: adler32 of the CPU picture (R.render, 96 x 72, transparency NONE) of
#: three pre-existing featurematrix scenes on the pre-edit tree
OLD_FRAME_DIGESTS = {'textured': 1656485521, 'demo': 1045851408,
                     'mirror': 1324349618}

LANE_N = 4096
NEW_MODES = ('CUBE', 'EQUIRECT', 'MIRROR_BALL', 'ANGULAR', 'CYLINDER_CAP',
             'SCREEN', 'REFLECTION', 'NORMAL', 'TANGENT')
BITWISE_MODES = ('CUBE', 'SCREEN', 'NORMAL')
NEW_IDS = ('HALCYON_TriPlanarNode', 'HALCYON_TriPlanarMixNode',
           'HALCYON_TriPlanarTextureNode', 'HALCYON_ObjectMapNode',
           'HALCYON_HexGridNode')
EXTRA_UNIS = ('uniform vec3 hal_N;\nuniform vec3 hal_V;\n'
              'uniform vec3 hal_P;\nuniform sampler2D hal_tex0;\n')


# ------------------------------------------------------------- helpers
def _bpy():
    """export.py and nodes/shader_nodes.py import bpy: the stub."""
    from . import fakebpy
    return fakebpy.install()


def lanes_rng(seed=253):
    return np.random.default_rng(seed)


def unit(rng, n):
    return M.normalize((rng.random((n, 3)).astype(f32) * 2.0 - 1.0)
                       .astype(f32)).astype(f32)


def lane_ctx(n=LANE_N, seed=253, screen=True):
    """A context with random unit normals / incidents and, with `screen`,
    the pixel grid of a 96 x 72 frame; returns (ctx, V, extra) where V
    are random -2..2 vectors and `extra` the GPU uniforms that stand for
    the same context (hal_N, hal_V, hal_P, vUV)."""
    rng = lanes_rng(seed)
    c = _ctx(n)
    V = (rng.random((n, 3)).astype(f32) * 4.0 - 2.0).astype(f32)
    c.N = unit(rng, n)
    c.I = unit(rng, n)
    c.P = V.copy()
    extra = {'hal_N': c.N, 'hal_V': (-c.I).astype(f32), 'hal_P': c.P}
    if screen:
        c.px = rng.integers(0, W, n)
        c.py = rng.integers(0, H, n)
        c.width, c.height = W, H
        extra['vUV'] = np.stack([((c.px + 0.5) / W).astype(f32),
                                 ((c.py + 0.5) / H).astype(f32)], 1)
    return c, V, extra


def cpu_eval(node, lanes, ctx, out=0, images=None):
    """The CPU evaluator on one node, its linked inputs fed from `lanes`
    (name -> (glsl type, array)), exactly what node_twin feeds the GPU."""
    ev = GraphEvaluator({'output': None, 'nodes': {node['id']: node}}, ctx,
                        images=images)
    for k, (_gt, arr) in lanes.items():
        ev.cache[(k, 0)] = np.asarray(arr, f32)
    r = ev.eval_output(node['id'], out)
    if ev.errors:
        raise RuntimeError(str(ev.errors[0]))
    return r


def twin(node, lanes, n, out, extra):
    g, why = node_twin(node, lanes, n, out_index=out, extra_unis=EXTRA_UNIS,
                       extra_uniforms=extra)
    return g, why


def gpu_cols(g, ref):
    """The simulator's (n, 4) Color trimmed to the CPU output's shape."""
    if ref.ndim == 1:
        return g[:, 0]
    return g[:, :ref.shape[1]]


def projector(mode, axis='Z', scale=0.7, link='lv', aspect=False):
    return _wnode('p', 'HALCYON_ProjectorNode',
                  {'mode': mode, 'axis': axis, 'aspect': aspect},
                  [_sk('Vector', 'VECTOR', [0, 0, 0], [link, 0] if link else None),
                   _sk('Scale', 'VALUE', scale)],
                  [{'name': 'Vector', 'type': 'VECTOR'}])


def emit_body(node, lane_name='hal_lane', secondary=False, resolution=None):
    """e_<node>'s body with the Vector link answered by a vec3 lane."""
    nodes = {node['id']: node}
    for sock in node['inputs']:
        if sock.get('link'):
            nodes[sock['link'][0]] = {
                'id': sock['link'][0], 'bl_idname': '__R253_LANE__',
                'props': {}, 'inputs': [],
                'outputs': [{'name': 'V', 'type': 'VECTOR'}]}
    EM.EMITTERS['__R253_LANE__'] = lambda em, nd, _i: (lane_name, 'vec3')
    try:
        em = EM.Emitter({'output': node['id'], 'nodes': nodes})
        em.secondary = secondary
        if resolution is not None:
            em.resolution = resolution
        try:
            em.output(node['id'], 0)
            err = None
        except EM.Unsupported as exc:
            err = str(exc)
    finally:
        EM.EMITTERS.pop('__R253_LANE__', None)
    return em, err


def index_image(size=64):
    """Every texel a unique colour: R = x / (size-1), G = y / (size-1)."""
    img = np.zeros((size, size, 4), f32)
    yy, xx = np.mgrid[0:size, 0:size]
    img[:, :, 0] = xx / float(size - 1)
    img[:, :, 1] = yy / float(size - 1)
    img[:, :, 2] = 0.5
    img[:, :, 3] = 1.0
    return img


def tri_planar_node(nid='t', scale=0.7, blend=0.5, link_n='ln', link_v='lv'):
    outs = [{'name': k, 'type': 'VECTOR' if k.startswith('UV') else 'VALUE'}
            for k in ('UV X', 'UV Y', 'UV Z', 'Weight X', 'Weight Y',
                      'Weight Z')]
    return _wnode(nid, 'HALCYON_TriPlanarNode', {},
                  [_sk('Vector', 'VECTOR', [0, 0, 0],
                       [link_v, 0] if link_v else None),
                   _sk('Normal', 'VECTOR', [0, 0, 0],
                       [link_n, 0] if link_n else None),
                   _sk('Scale', 'VALUE', scale), _sk('Blend', 'VALUE', blend)],
                  outs)


def tri_texture_node(image='checker', interp='Closest', ext='REPEAT',
                     scale=0.7, blend=0.5, link_n='ln', link_v='lv'):
    props = {'interpolation': interp, 'extension': ext}
    if image:
        props['image'] = image
    return _wnode('tt', 'HALCYON_TriPlanarTextureNode', props,
                  [_sk('Vector', 'VECTOR', [0, 0, 0],
                       [link_v, 0] if link_v else None),
                   _sk('Normal', 'VECTOR', [0, 0, 0],
                       [link_n, 0] if link_n else None),
                   _sk('Scale', 'VALUE', scale), _sk('Blend', 'VALUE', blend)],
                  [{'name': 'Color', 'type': 'RGBA'},
                   {'name': 'Alpha', 'type': 'VALUE'}])


def tex_image_node(proj, image='checker', interp='Closest', link='lv'):
    return _wnode('ti', 'ShaderNodeTexImage',
                  {'image': image, 'interpolation': interp,
                   'projection': proj, 'extension': 'REPEAT'},
                  [_sk('Vector', 'VECTOR', [0, 0, 0], [link, 0] if link else None)],
                  [{'name': 'Color', 'type': 'RGBA'},
                   {'name': 'Alpha', 'type': 'VALUE'}])


def object_map_node(mode, inv12=None, angle=45.0, size=2.0, link='lv',
                    angle_link=None, extra_props=None):
    props = {'mode': mode}
    if inv12 is not None:
        props['object_inv'] = [float(x) for x in inv12]
    props.update(extra_props or {})
    return _wnode('o', 'HALCYON_ObjectMapNode', props,
                  [_sk('Vector', 'VECTOR', [0, 0, 0], [link, 0] if link else None),
                   _sk('Ortho Size', 'VALUE', size),
                   _sk('Angle', 'VALUE', angle,
                       [angle_link, 0] if angle_link else None)],
                  [{'name': 'Vector', 'type': 'VECTOR'},
                   {'name': 'Inside', 'type': 'VALUE'},
                   {'name': 'Depth', 'type': 'VALUE'}])


def hex_node(orient='POINTY', scale=4.0, link='lv'):
    return _wnode('h', 'HALCYON_HexGridNode', {'orient': orient},
                  [_sk('Vector', 'VECTOR', [0, 0, 0], [link, 0] if link else None),
                   _sk('Scale', 'VALUE', scale)],
                  [{'name': 'Vector', 'type': 'VECTOR'},
                   {'name': 'Cell ID', 'type': 'VALUE'},
                   {'name': 'Edge', 'type': 'VALUE'}])


def lit8(x):
    return float(f'{float(x):.8g}')


def trs_matrix():
    """A scaled + rotated + translated object's matrix_world."""
    sx, sy, sz = 2.0, 0.5, 1.5
    a, b = 0.7, -0.4
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0],
                   [0, 0, 1]], np.float64)
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)],
                   [0, math.sin(b), math.cos(b)]], np.float64)
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = rz @ rx @ np.diag([sx, sy, sz])
    m[:3, 3] = [1.5, -2.0, 0.7]
    return m.astype(f32)


def inv12_of(mw):
    inv = np.linalg.inv(np.asarray(mw, f32))
    return [lit8(x) for x in inv[:3, :].reshape(-1)]


# ================================================= the shipped modes
def test_projector_old_modes_frozen():
    """PLANAR / CYLINDER / SPHERE / BOX on 4096 random vectors + Scale
    0.7: the CPU float32 output's adler32 equals the pre-edit digest on
    every axis, and e_v16_projector's emitted text equals the pre-edit
    string (the luma64 precedent, commit 488fa40). Also documents the
    shipped BOX quirk: the face is picked by the COORDINATE's largest
    axis, not the normal -- (0.9, 0.2, 0.3) under N = +Z lands on the X
    face (the (0.2, 0.3) pair), where a normal-picked Box would give the
    Z face's (0.9, 0.2); Cube Atlas and Tri-Planar pick by the normal."""
    c, V, _extra = lane_ctx(screen=False)
    lanes = {'lv': ('vec3', V)}
    for mode in ('PLANAR', 'CYLINDER', 'SPHERE', 'BOX'):
        for axis in ('X', 'Y', 'Z'):
            out = cpu_eval(projector(mode, axis), lanes, c)
            dig = zlib.adler32(np.ascontiguousarray(out, f32).tobytes())
            check(f'{mode} / {axis}: the CPU output is the 1.91 bytes '
                  f'(adler32 {OLD_CPU_DIGESTS[(mode, axis)]})',
                  dig == OLD_CPU_DIGESTS[(mode, axis)], str(dig))
            em, err = emit_body(projector(mode, axis))
            body = em.body()
            if axis == 'Z':
                check(f'{mode} / Z: the emitted GLSL is the 1.91 text verbatim',
                      err is None and body == OLD_GLSL_Z[mode], body)
            else:
                tdig = zlib.adler32(body.encode('utf-8'))
                check(f'{mode} / {axis}: the emitted GLSL is the 1.91 text '
                      f'(adler32 {OLD_GLSL_XY_DIGESTS[(mode, axis)]})',
                      err is None and tdig == OLD_GLSL_XY_DIGESTS[(mode, axis)],
                      str(tdig))
    # the BOX-by-vector quirk, documented not corrected
    c1 = _ctx(1)
    c1.N = np.array([[0.0, 0.0, 1.0]], f32)
    q = cpu_eval(projector('BOX', 'Z', scale=1.0),
                 {'lv': ('vec3', np.array([[0.9, 0.2, 0.3]], f32))}, c1)
    check('BOX picks the face by |vector| (the shipped quirk): (0.9, 0.2, '
          '0.3) under N = +Z gives the X face pair (0.2, 0.3)',
          np.allclose(q[0, :2], [0.2, 0.3], atol=1e-7), str(q[0]))
    tie = cpu_eval(projector('BOX', 'Z', scale=1.0),
                   {'lv': ('vec3', np.array([[0.5, 0.5, 0.5]], f32))}, c1)
    check('the all-equal (0.5, 0.5, 0.5) lane takes the first-max X face',
          np.allclose(tie[0, :2], [0.5, 0.5]) and float(tie[0, 2]) == 0.0)
    cube = cpu_eval(projector('CUBE', 'Z', scale=1.0),
                    {'lv': ('vec3', np.array([[0.9, 0.2, 0.3]], f32))}, c1)
    check('CUBE picks by the NORMAL: the same lane lands on the +Z face '
          '(face index 4)', float(cube[0, 2]) == 4.0, str(cube[0]))
    _bpy()
    from ..nodes import shader_nodes as SN
    spec = next(s for s in SN.VEC_SPECS if s[0] == 'Projector')
    idents = [i[0] for i in spec[5]['mode'][2]]
    check("the Projector enum's first four identifiers are still PLANAR, "
          'CYLINDER, SPHERE, BOX in that order (saved files keep loading)',
          idents[:4] == ['PLANAR', 'CYLINDER', 'SPHERE', 'BOX'], str(idents))
    check('the nine R253 modes are APPENDED after them',
          idents[4:] == list(NEW_MODES), str(idents))


def test_projector_new_modes_twin():
    """Per appended mode and axis, n_v16_projector vs node_twin (hal_N,
    hal_V, vUV, hal_P as the context's uniforms) on 4096 lanes: CUBE /
    SCREEN / NORMAL bitwise, the transcendental modes |delta| <= 2e-6;
    CUBE's z is an integer face 0..5 and its xy stays inside the atlas
    cell; CYLINDER_CAP's z is 1 exactly where |N_axis| > 0.70710678."""
    c, V, extra = lane_ctx()
    lanes = {'lv': ('vec3', V)}
    for mode in NEW_MODES:
        for axis in ('X', 'Y', 'Z'):
            node = projector(mode, axis)
            cpu = cpu_eval(node, lanes, c)
            g, why = twin(node, lanes, LANE_N, 0, extra)
            if not check(f'{mode} / {axis}: the GPU twin runs', g is not None,
                         str(why)):
                continue
            g = gpu_cols(g, cpu)
            d = float(np.abs(cpu.astype(np.float64) - g).max())
            if mode in BITWISE_MODES:
                check(f'{mode} / {axis}: bitwise the CPU (np.array_equal)',
                      bool(np.array_equal(cpu, g)), f'max {d:.2e}')
            else:
                check(f'{mode} / {axis}: |delta| <= 2e-6 on 4096 lanes',
                      d <= 2e-6, f'max {d:.2e}')
            check(f'{mode} / {axis}: finite everywhere',
                  bool(np.isfinite(cpu).all()))
            if mode == 'CUBE':
                z = cpu[:, 2]
                check(f'CUBE / {axis}: z is an integer face index 0..5',
                      bool(np.array_equal(z, np.floor(z)))
                      and float(z.min()) >= 0 and float(z.max()) <= 5)
                ax_i = np.floor(z / 2.0)
                neg = z - ax_i * 2.0
                lo_x = ax_i / 3.0
                hi_x = (ax_i + 1.0) / 3.0
                lo_y = (1.0 - neg) / 2.0
                hi_y = lo_y + 0.5
                check(f'CUBE / {axis}: xy stays inside its atlas cell',
                      bool(((cpu[:, 0] >= lo_x - 1e-6) & (cpu[:, 0] <= hi_x + 1e-6)
                            & (cpu[:, 1] >= lo_y - 1e-6)
                            & (cpu[:, 1] <= hi_y + 1e-6)).all()))
                # every face is reached by random normals
                check(f'CUBE / {axis}: all six faces occur',
                      len(set(np.unique(z).tolist())) == 6)
            if mode == 'CYLINDER_CAP':
                ai = {'X': 0, 'Y': 1, 'Z': 2}[axis]
                lid = np.abs(c.N[:, ai]) > f32(0.70710678)
                check(f'CYLINDER_CAP / {axis}: z is 1 exactly where '
                      '|N_axis| > 0.70710678 and 0 elsewhere',
                      bool(np.array_equal(cpu[:, 2], lid.astype(f32)))
                      and 0 < int(lid.sum()) < LANE_N)
    # NORMAL is N * 0.5 + 0.5 with xy over the scale; the axis is ignored
    node = projector('NORMAL', 'X', scale=1.0)
    cpu = cpu_eval(node, lanes, c)
    check('NORMAL is the world normal * 0.5 + 0.5',
          bool(np.array_equal(cpu, (c.N * f32(0.5) + f32(0.5)).astype(f32))))
    # EQUIRECT is scale-invariant in the direction (no normalize seam)
    e1 = cpu_eval(projector('EQUIRECT', 'Z', scale=1.0), lanes, c)
    e2 = cpu_eval(projector('EQUIRECT', 'Z', scale=1.0),
                  {'lv': ('vec3', (V * f32(3.0)).astype(f32))}, c)
    check('EQUIRECT reads the direction only (|vector| * 3 moves it < 2e-6)',
          float(np.abs(e1 - e2).max()) <= 2e-6)
    # REFLECTION is Tex Coord's Reflection output through the equirect
    refl = M.reflect(M.normalize(c.I), M.normalize(c.N)).astype(f32)
    u, v = TX.env_equirect_uv(refl)
    r = cpu_eval(projector('REFLECTION', 'Z', scale=1.0), lanes, c)
    check("REFLECTION is n_tex_coord's Reflection through env_equirect_uv",
          float(np.abs(r[:, 0] - u).max()) <= 2e-6
          and float(np.abs(r[:, 1] - v).max()) <= 2e-6)
    # TANGENT: the tangent frame is orthonormal, so |out| = |v| at Scale 1
    t = cpu_eval(projector('TANGENT', 'Z', scale=1.0), lanes, c)
    check('TANGENT is a rotation: |out| == |v| within 1e-5',
          float(np.abs(M.length(t) - M.length(V)).max()) <= 1e-5)
    nn = M.normalize(c.N)
    check('TANGENT z is the component along the normal',
          float(np.abs(t[:, 2] - M.dot(V, nn)).max()) <= 1e-6)


def test_projector_screen_is_the_pixel():
    """SCREEN: with ctx.px/py/width/height set the output is ((px+0.5)/w,
    (py+0.5)/h, 0) exactly; a HIT context (px None) reads zeros; an
    em.secondary emitter writes vec3(0.0); Keep Aspect multiplies x by
    w/h on the CPU and refuses BY NAME on an Emitter with no resolution
    (the text lands in em.unsupported), and twins bitwise when
    em.resolution is set."""
    c, V, extra = lane_ctx()
    lanes = {'lv': ('vec3', V)}
    out = cpu_eval(projector('SCREEN', 'Z', scale=1.0), lanes, c)
    want = np.stack([((c.px + 0.5) / W).astype(f32),
                     ((c.py + 0.5) / H).astype(f32),
                     np.zeros(LANE_N, f32)], 1)
    check('SCREEN is ((px + 0.5) / w, (py + 0.5) / h, 0) exactly',
          bool(np.array_equal(out, want)))
    hit = _ctx(LANE_N)
    hit.px = None
    out = cpu_eval(projector('SCREEN', 'Z', scale=1.0), lanes, hit)
    check('a HIT context (px None) reads zeros', not out.any())
    em, err = emit_body(projector('SCREEN'), secondary=True)
    check('em.secondary = True emits vec3(0.0)',
          err is None and 'vec3(0.0)' in em.body(), em.body())
    em, err = emit_body(projector('SCREEN'), secondary=False, resolution=(W, H))
    check('the primary pass reads vUV', err is None and 'vUV' in em.body())
    # Keep Aspect
    outa = cpu_eval(projector('SCREEN', 'Z', scale=1.0, aspect=True), lanes, c)
    check('Keep Aspect multiplies x by w/h on the CPU',
          bool(np.array_equal(outa[:, 0], (want[:, 0] * f32(W / H)).astype(f32)))
          and bool(np.array_equal(outa[:, 1:], want[:, 1:])))
    em, err = emit_body(projector('SCREEN', aspect=True), resolution=None)
    text = 'HALCYON_ProjectorNode (Screen mode with Keep Aspect needs the frame resolution)'
    check('Keep Aspect refuses BY NAME when the Emitter has no resolution',
          err == text and text in em.unsupported, f'{err!r} {em.unsupported}')
    em, err = emit_body(projector('SCREEN', aspect=True), resolution=(W, H))
    check('and emits with a resolution', err is None and not em.unsupported)
    for aspect in (False, True):
        node = projector('SCREEN', 'Z', scale=0.7, aspect=aspect)
        cpu = cpu_eval(node, lanes, c)
        g, why = twin(node, lanes, LANE_N, 0, extra)
        check(f'SCREEN (aspect {aspect}) twins bitwise under node_twin '
              '(resolution set)',
              g is not None and bool(np.array_equal(cpu, gpu_cols(g, cpu))),
              str(why))
    # the aspect literal round-trips float32: the GPU text carries repr(f32)
    check('the Keep Aspect literal is a float32 round-trip',
          EM._f32lit(W / H) == repr(float(f32(W / H))) and '.' in EM._f32lit(2))


# ===================================================== tri-planar
def test_tri_planar_weights():
    """Over 4096 random unit normals and Blend in (0, 0.25, 0.5, 1): each
    weight >= 0, |(wx + wy) + wz - 1| <= 2e-7, an axis-aligned normal
    gives a one-hot weight at every Blend, Blend 1 gives w proportional
    to |n|, the unlinked Normal socket reads ctx.N; the three UV sets
    equal the _box_project pairs / Scale; all six outputs bitwise
    against node_twin."""
    c, V, extra = lane_ctx(screen=False)
    lanes = {'lv': ('vec3', V), 'ln': ('vec3', c.N)}
    for blend in (0.0, 0.25, 0.5, 1.0):
        node = tri_planar_node(blend=blend)
        w = np.stack([cpu_eval(node, lanes, c, i) for i in (3, 4, 5)], 1)
        check(f'Blend {blend}: every weight >= 0', float(w.min()) >= 0.0)
        s = (w[:, 0] + w[:, 1]) + w[:, 2]
        check(f'Blend {blend}: |(wx + wy) + wz - 1| <= 2e-7',
              float(np.abs(s - 1.0).max()) <= 2e-7,
              f'{float(np.abs(s - 1.0).max()):.2e}')
        axes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                         [0, 0, 1], [0, 0, -1]], f32)
        ca = _ctx(6)
        wa = np.stack([cpu_eval(node, {'lv': ('vec3', V[:6]),
                                       'ln': ('vec3', axes)}, ca, i)
                       for i in (3, 4, 5)], 1)
        want = np.abs(axes)
        check(f'Blend {blend}: an axis-aligned normal gives a one-hot weight',
              bool(np.array_equal(wa, want)), str(wa))
        if blend == 1.0:
            a = np.abs(c.N)
            prop = a / a.sum(1, keepdims=True)
            check('Blend 1 gives w proportional to |n|',
                  float(np.abs(w - prop).max()) <= 1e-6)
        if blend == 0.0:
            # t = 0.5: the dominant axis always leads (max |n| >= 1/sqrt3
            # > 0.5 on a unit normal) and a face fades in only past
            # |n_i| = 0.5, so most lanes are one-hot, the near-diagonal
            # ones a two- or three-way blend
            a = np.abs(c.N)
            dom = np.argmax(a, 1)
            second = np.sort(a, 1)[:, 1]
            check('Blend 0: the dominant axis carries the largest weight on '
                  'every lane', bool(np.array_equal(np.argmax(w, 1), dom)))
            check('Blend 0: one-hot wherever the second-largest |n| < 0.5',
                  float(w.max(1)[second < 0.5].min()) == 1.0
                  and int((second < 0.5).sum()) > LANE_N // 4)
    # the unlinked Normal reads ctx.N
    node = tri_planar_node(link_n=None)
    w_unl = np.stack([cpu_eval(node, {'lv': ('vec3', V)}, c, i) for i in (3, 4, 5)], 1)
    w_lnk = np.stack([cpu_eval(tri_planar_node(), lanes, c, i) for i in (3, 4, 5)], 1)
    check('the unlinked Normal socket reads ctx.N (bitwise the linked lane)',
          bool(np.array_equal(w_unl, w_lnk)))
    # the UV sets are the _box_project pairs over Scale
    node = tri_planar_node()
    uvx, uvy, uvz = (cpu_eval(node, lanes, c, i) for i in (0, 1, 2))
    sc = f32(0.7)
    check('UV X is (y, z) / Scale, UV Y (x, z) / Scale, UV Z (x, y) / Scale',
          bool(np.array_equal(uvx[:, :2], np.stack([V[:, 1] / sc, V[:, 2] / sc], 1)))
          and bool(np.array_equal(uvy[:, :2], np.stack([V[:, 0] / sc, V[:, 2] / sc], 1)))
          and bool(np.array_equal(uvz[:, :2], np.stack([V[:, 0] / sc, V[:, 1] / sc], 1)))
          and not uvx[:, 2].any())
    # against _box_project on axis-aligned normals the dominant face's pair
    for ai, uv in ((0, uvx), (1, uvy), (2, uvz)):
        nn = np.zeros((LANE_N, 3), f32)
        nn[:, ai] = 1.0
        bu, bv = NE._box_project(V, nn)
        check(f'face {"XYZ"[ai]} is _box_project\'s pair for that normal',
              bool(np.array_equal(uv[:, 0], (bu / sc).astype(f32)))
              and bool(np.array_equal(uv[:, 1], (bv / sc).astype(f32))))
    # all six outputs bitwise against the GPU
    for blend in (0.0, 0.5, 1.0):
        node = tri_planar_node(blend=blend)
        for i, name in enumerate(('UV X', 'UV Y', 'UV Z', 'Weight X',
                                  'Weight Y', 'Weight Z')):
            cpu = cpu_eval(node, lanes, c, i)
            g, why = twin(node, lanes, LANE_N, i, extra)
            check(f'Blend {blend}: {name} is bitwise the GPU twin',
                  g is not None and bool(np.array_equal(cpu, gpu_cols(g, cpu))),
                  str(why))


def test_tri_planar_mix_twin():
    """(cx*wx + cy*wy) + cz*wz bitwise against node_twin on 4096 lanes;
    Alpha is the blended alpha."""
    rng = lanes_rng(31)
    n = LANE_N
    C = [rng.random((n, 4)).astype(f32) for _ in range(3)]
    Wt = [rng.random(n).astype(f32) for _ in range(3)]
    lanes = {'c0': ('vec4', C[0]), 'c1': ('vec4', C[1]), 'c2': ('vec4', C[2]),
             'w0': ('float', Wt[0]), 'w1': ('float', Wt[1]),
             'w2': ('float', Wt[2])}
    node = _wnode('m', 'HALCYON_TriPlanarMixNode', {},
                  [_sk('Color X', 'RGBA', [0, 0, 0, 1], ['c0', 0]),
                   _sk('Color Y', 'RGBA', [0, 0, 0, 1], ['c1', 0]),
                   _sk('Color Z', 'RGBA', [0, 0, 0, 1], ['c2', 0]),
                   _sk('Weight X', 'VALUE', 0.0, ['w0', 0]),
                   _sk('Weight Y', 'VALUE', 0.0, ['w1', 0]),
                   _sk('Weight Z', 'VALUE', 0.0, ['w2', 0])],
                  [{'name': 'Color', 'type': 'RGBA'},
                   {'name': 'Alpha', 'type': 'VALUE'}])
    c = _ctx(n)
    col = cpu_eval(node, lanes, c, 0)
    want = ((C[0] * Wt[0][:, None] + C[1] * Wt[1][:, None])
            + C[2] * Wt[2][:, None]).astype(f32)
    check('Color is (cx*wx + cy*wy) + cz*wz on all four channels',
          bool(np.array_equal(col, want)))
    alpha = cpu_eval(node, lanes, c, 1)
    check('Alpha is the blended alpha', bool(np.array_equal(alpha, want[:, 3])))
    g, why = twin(node, lanes, n, 0, {})
    check('Color is bitwise the GPU twin',
          g is not None and bool(np.array_equal(col, g)), str(why))
    g, why = twin(node, lanes, n, 1, {})
    check('Alpha is bitwise the GPU twin',
          g is not None and bool(np.array_equal(alpha, g[:, 0])), str(why))
    # the defaults: equal thirds of the socket colours
    node0 = _wnode('m0', 'HALCYON_TriPlanarMixNode', {},
                   [_sk('Color X', 'RGBA', [1, 0, 0, 1]),
                    _sk('Color Y', 'RGBA', [0, 1, 0, 1]),
                    _sk('Color Z', 'RGBA', [0, 0, 1, 1]),
                    _sk('Weight X', 'VALUE', 0.333333),
                    _sk('Weight Y', 'VALUE', 0.333333),
                    _sk('Weight Z', 'VALUE', 0.333333)],
                   [{'name': 'Color', 'type': 'RGBA'},
                    {'name': 'Alpha', 'type': 'VALUE'}])
    col0 = cpu_eval(node0, {}, _ctx(4), 0)
    check('the socket defaults mix equal thirds',
          np.allclose(col0[0], [0.333333, 0.333333, 0.333333, 0.999999], atol=1e-6))


def test_tri_planar_texture_samples_three_faces():
    """An index image (every texel a unique colour) under NEAREST: an
    axis-aligned normal reproduces the plain Image Texture BOX answer
    bitwise; a (1,1,1)/sqrt3 normal at Blend 1 is the exact three-tap
    mean; no image -> opaque black on both roads (and NO sampler); GPU:
    node_twin with the image as the sampler uniform, bitwise under
    NEAREST, <= 2e-3 under Linear; em.samplers holds exactly ONE entry
    with footprint False."""
    c, V, extra = lane_ctx(screen=False)
    px = index_image(64)
    tex = TX.Texture(px, colorspace='Non-Color', filt='NEAREST', wrap='REPEAT')
    images = {'checker': tex}
    for ai in range(3):
        nn = np.zeros((LANE_N, 3), f32)
        nn[:, ai] = 1.0
        lanes = {'lv': ('vec3', V), 'ln': ('vec3', nn)}
        ca = _ctx(LANE_N)
        ca.N = nn
        tri = cpu_eval(tri_texture_node(scale=1.0), lanes, ca, 0, images)
        box = cpu_eval(tex_image_node('BOX'), {'lv': ('vec3', V)}, ca, 0, images)
        check(f'an axis-aligned normal ({"XYZ"[ai]}) reproduces the Image '
              'Texture BOX answer bitwise', bool(np.array_equal(tri, box)))
    nn = np.tile(np.array([[1, 1, 1]], f32) / f32(math.sqrt(3.0)), (LANE_N, 1))
    lanes = {'lv': ('vec3', V), 'ln': ('vec3', nn)}
    tri = cpu_eval(tri_texture_node(scale=1.0, blend=1.0), lanes, c, 0, images)
    taps = [tex.sample(V[:, a], V[:, b], filt='NEAREST', wrap='REPEAT')
            for a, b in ((1, 2), (0, 2), (0, 1))]
    wts = [cpu_eval(tri_planar_node(scale=1.0, blend=1.0), lanes, c, i)
           for i in (3, 4, 5)]
    mean = ((taps[0] * wts[0][:, None] + taps[1] * wts[1][:, None])
            + taps[2] * wts[2][:, None]).astype(f32)
    check('a (1,1,1)/sqrt3 normal at Blend 1 is the exact three-tap mean',
          bool(np.array_equal(tri, mean))
          and float(np.abs(tri - (taps[0] + taps[1] + taps[2]) / 3.0).max()) <= 2e-6)
    check('the three taps differ (the image really is sampled three ways)',
          not np.array_equal(taps[0], taps[1]))
    alpha = cpu_eval(tri_texture_node(scale=1.0, blend=1.0), lanes, c, 1, images)
    check('Alpha is the blended alpha', bool(np.array_equal(alpha, tri[:, 3])))
    # no image: opaque black on both roads, no sampler
    none = cpu_eval(tri_texture_node(image=None), lanes, c, 0, images)
    check('no image -> opaque black on the CPU',
          bool(np.array_equal(none, np.tile([[0, 0, 0, 1]], (LANE_N, 1)).astype(f32))))
    em, err = emit_body(tri_texture_node(image=None))
    check('no image -> opaque black on the GPU and NO sampler',
          err is None and 'vec4(0.0, 0.0, 0.0, 1.0)' in em.body()
          and not em.samplers)
    # the GPU twin
    lanes = {'lv': ('vec3', V), 'ln': ('vec3', c.N)}
    ex = dict(extra)
    ex['hal_tex0'] = tex
    node = tri_texture_node()
    cpu = cpu_eval(node, lanes, c, 0, images)
    g, why = twin(node, lanes, LANE_N, 0, ex)
    check('Tri-Planar Texture is bitwise the GPU twin under NEAREST',
          g is not None and bool(np.array_equal(cpu, g)), str(why))
    a_cpu = cpu_eval(node, lanes, c, 1, images)
    g, why = twin(node, lanes, LANE_N, 1, ex)
    check('its Alpha too',
          g is not None and bool(np.array_equal(a_cpu, g[:, 0])), str(why))
    em, err = emit_body(node)
    check('em.samplers holds exactly ONE entry with footprint False',
          err is None and len(em.samplers) == 1
          and em.samplers[0]['footprint'] is False
          and em.samplers[0]['image'] == 'checker'
          and em.body().count('texture(hal_tex0,') == 3, str(em.samplers))
    # Linear: the simulator's bilinear against the CPU's
    node = tri_texture_node(interp='Linear')
    # the renderer's Texture Filter setting overrides the node on the CPU
    # (n_tex_image's rule, resolve_tex_filter on the GPU): ask for bilinear
    c.settings.tex_filter = 'BILINEAR'
    cpu = cpu_eval(node, lanes, c, 0, images)
    c.settings.tex_filter = 'NEAREST'
    old = SB.ctx()
    SB.set_ctx(SB.Ctx(filt='BILINEAR', wrap='REPEAT'))
    try:
        g, why = twin(node, lanes, LANE_N, 0, ex)
    finally:
        SB.set_ctx(old)
    d = float(np.abs(cpu - g).max()) if g is not None else 1.0
    check('under Linear the twin holds at <= 2e-3', g is not None and d <= 2e-3,
          f'max {d:.2e} {why}')


# ================================================ object projection
def test_object_projection_node():
    """FRAME maps the world-space corners of a scaled + rotated +
    translated object back to its local unit box (bitwise vs the explicit
    float32 chain); PERSPECTIVE: a point on the projector's -Z axis at
    any depth -> (0.5, 0.5), the cone edge (tan_half * depth) -> u = 1,
    behind the lamp Inside = 0, Depth = -local.z; ORTHO: Ortho Size 2
    maps local x in -1..1 to 0..1; export (fakebpy objects LIGHT/SPOT,
    LIGHT/SUN, CAMERA PERSP, CAMERA ORTHO, EMPTY, None) writes
    proj_mode / proj_tan / proj_half / object_inv as specified and warns
    on the unpicked case; CPU vs node_twin bitwise on all three outputs;
    a linked Angle socket refuses BY NAME on the GPU."""
    mw = trs_matrix()
    inv12 = inv12_of(mw)
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                       f32)
    world = (np.c_[corners, np.ones(8, f32)] @ mw.T)[:, :3].astype(f32)
    c8 = _ctx(8)
    c8.P = world
    lanes = {'lv': ('vec3', world)}
    loc = cpu_eval(object_map_node('FRAME', inv12), lanes, c8, 0)
    check('FRAME maps the world corners back to the local unit box (1e-4)',
          float(np.abs(loc - corners).max()) <= 1e-4,
          f'{float(np.abs(loc - corners).max()):.2e}')
    m = [f32(x) for x in inv12]
    px_, py_, pz_ = world[:, 0], world[:, 1], world[:, 2]
    want = np.stack([((m[0] * px_ + m[1] * py_) + m[2] * pz_) + m[3],
                     ((m[4] * px_ + m[5] * py_) + m[6] * pz_) + m[7],
                     ((m[8] * px_ + m[9] * py_) + m[10] * pz_) + m[11]], 1).astype(f32)
    check('and is bitwise the explicit ((m0*x + m1*y) + m2*z) + m3 chain',
          bool(np.array_equal(loc, want)))
    unl = cpu_eval(object_map_node('FRAME', inv12, link=None), {}, c8, 0)
    check('an unlinked Vector reads the world position (ctx.P)',
          bool(np.array_equal(unl, loc)))
    check('FRAME: Inside is 1, Depth is -local.z',
          bool(np.array_equal(cpu_eval(object_map_node('FRAME', inv12), lanes, c8, 1),
                              np.ones(8, f32)))
          and bool(np.array_equal(cpu_eval(object_map_node('FRAME', inv12), lanes, c8, 2),
                                  (-loc[:, 2]).astype(f32))))
    ident = cpu_eval(object_map_node('FRAME'), lanes, c8, 0)
    check('no object_inv -> the identity (world space)',
          bool(np.array_equal(ident, world)))
    # PERSPECTIVE on a lamp at the world origin looking down -Z (identity)
    depths = np.array([0.5, 1.0, 3.0, 10.0], f32)
    n4 = 4
    c4 = _ctx(n4)
    tan_half = lit8(math.tan(math.radians(45.0) * 0.5))
    axis_pts = np.stack([np.zeros(n4, f32), np.zeros(n4, f32), -depths], 1)
    node = object_map_node('PERSPECTIVE')
    out = cpu_eval(node, {'lv': ('vec3', axis_pts)}, c4, 0)
    check('PERSPECTIVE: a point on the -Z axis at any depth -> (0.5, 0.5)',
          bool(np.array_equal(out[:, :2], np.full((n4, 2), 0.5, f32))))
    edge = np.stack([(depths * f32(tan_half)).astype(f32), np.zeros(n4, f32),
                     -depths], 1)
    out = cpu_eval(node, {'lv': ('vec3', edge)}, c4, 0)
    check('the cone edge (tan_half * depth) -> u = 1 within 1e-6',
          float(np.abs(out[:, 0] - 1.0).max()) <= 1e-6)
    inside = cpu_eval(node, {'lv': ('vec3', axis_pts)}, c4, 1)
    behind = cpu_eval(node, {'lv': ('vec3', (-axis_pts).astype(f32))}, c4, 1)
    check('Inside is 1 in front and 0 behind the lamp',
          bool(inside.all()) and not behind.any())
    depth = cpu_eval(node, {'lv': ('vec3', axis_pts)}, c4, 2)
    check('Depth is -local.z', bool(np.array_equal(depth, depths)))
    outside = np.stack([(depths * f32(tan_half) * f32(1.5)).astype(f32),
                        np.zeros(n4, f32), -depths], 1)
    check('a point past the cone edge is outside (Inside 0)',
          not cpu_eval(node, {'lv': ('vec3', outside)}, c4, 1).any())
    # proj_tan from the export overrides the Angle socket
    node2 = object_map_node('PERSPECTIVE', extra_props={'proj_tan': lit8(1.0)})
    out = cpu_eval(node2, {'lv': ('vec3', edge)}, c4, 0)
    check('an exported proj_tan (1.0 = a 90 degree cone) overrides the Angle socket',
          float(np.abs(out[:, 0] - (0.5 * tan_half + 0.5)).max()) <= 1e-6)
    # ORTHO
    xs = np.linspace(-1, 1, 9).astype(f32)
    pts = np.stack([xs, np.zeros(9, f32), -np.ones(9, f32)], 1)
    c9 = _ctx(9)
    out = cpu_eval(object_map_node('ORTHO', size=2.0), {'lv': ('vec3', pts)}, c9, 0)
    check('ORTHO: Ortho Size 2 maps local x in -1..1 to 0..1',
          float(np.abs(out[:, 0] - (xs * 0.5 + 0.5)).max()) <= 1e-6)
    ins = cpu_eval(object_map_node('ORTHO', size=2.0),
                   {'lv': ('vec3', (pts * f32(1.2)).astype(f32))}, c9, 1)
    check('ORTHO: Inside is the window test only (x beyond +-1 is out)',
          bool(ins[1:-1].all()) and not ins[0] and not ins[-1])
    # export
    _bpy()
    from .. import export as EX

    class _Sock:
        def __init__(self, name, value, linked=False):
            self.name, self.default_value, self.is_linked = name, value, linked

    class _Data:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class _Ob:
        def __init__(self, name, kind, data=None, mw=None):
            self.name, self.type, self.data = name, kind, data
            self.matrix_world = np.eye(4, dtype=f32) if mw is None else mw

    class _Node:
        bl_idname = 'HALCYON_ObjectMapNode'

        def __init__(self, ob, mode='AUTO', angle=45.0, size=2.0,
                     angle_linked=False, size_linked=False):
            self.name = 'Object Projection'
            self.object = ob
            self.mode = mode
            self.inputs = [_Sock('Vector', (0, 0, 0)),
                           _Sock('Ortho Size', size, size_linked),
                           _Sock('Angle', angle, angle_linked)]

    spot = _Ob('Spot', 'LIGHT', _Data(type='SPOT', spot_size=1.2), mw)
    sun = _Ob('Sun', 'LIGHT', _Data(type='SUN'))
    cam = _Ob('Cam', 'CAMERA', _Data(type='PERSP', lens=50.0, sensor_width=36.0,
                                     sensor_fit='AUTO'))
    camv = _Ob('CamV', 'CAMERA', _Data(type='PERSP', lens=50.0, sensor_width=36.0,
                                       sensor_height=24.0, sensor_fit='VERTICAL'))
    camo = _Ob('CamO', 'CAMERA', _Data(type='ORTHO', ortho_scale=6.0))
    empty = _Ob('Empty', 'EMPTY', None, mw)
    cases = [
        ('LIGHT/SPOT', spot, 'PERSPECTIVE', lit8(math.tan(0.6)), lit8(1.0)),
        ('LIGHT/SUN', sun, 'ORTHO', lit8(math.tan(math.radians(45.0) * 0.5)), lit8(1.0)),
        ('CAMERA PERSP', cam, 'PERSPECTIVE', lit8(36.0 / 100.0), lit8(1.0)),
        ('CAMERA PERSP (sensor_fit VERTICAL)', camv, 'PERSPECTIVE',
         lit8(24.0 / 100.0), lit8(1.0)),
        ('CAMERA ORTHO', camo, 'ORTHO', lit8(math.tan(math.radians(45.0) * 0.5)),
         lit8(3.0)),
        ('EMPTY', empty, 'FRAME', lit8(math.tan(math.radians(45.0) * 0.5)), lit8(1.0)),
    ]
    for label, ob, want_mode, want_tan, want_half in cases:
        warn = []
        props = EX.object_map_props(_Node(ob), warn)
        check(f'export {label}: proj_mode {want_mode}, proj_tan {want_tan}, '
              f'proj_half {want_half}',
              props['proj_mode'] == want_mode
              and props.get('proj_tan') == want_tan
              and props.get('proj_half') == want_half and not warn,
              str(props) + str(warn))
        want_inv = inv12_of(ob.matrix_world)
        check(f'export {label}: object_inv is rows 0..2 of inv(matrix_world), '
              '8 significant digits', props['object_inv'] == want_inv)
    warn = []
    props = EX.object_map_props(_Node(None), warn)
    check('export None: the identity, proj_mode FRAME and a warning',
          props['object_inv'] == [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                  0.0, 0.0, 1.0, 0.0]
          and props['proj_mode'] == 'FRAME' and len(warn) == 1
          and 'no object picked' in warn[0], str(warn))
    props = EX.object_map_props(_Node(empty, mode='PERSPECTIVE', angle=60.0), [])
    check('an explicit PERSPECTIVE on an empty takes the Angle socket',
          props['proj_mode'] == 'PERSPECTIVE'
          and props['proj_tan'] == lit8(math.tan(math.radians(60.0) * 0.5)))
    props = EX.object_map_props(_Node(spot, mode='PERSPECTIVE', angle=60.0), [])
    check("an explicit PERSPECTIVE on a spot keeps the lamp's own cone",
          props['proj_tan'] == lit8(math.tan(0.6)))
    props = EX.object_map_props(_Node(empty, angle_linked=True, size_linked=True), [])
    check('linked Angle / Ortho Size sockets leave proj_tan / proj_half '
          'unwritten (read per lane at shading time)',
          'proj_tan' not in props and 'proj_half' not in props)
    # the whole-node path through _node_props
    props = EX._node_props(_Node(spot), {}, {}, [])
    check("_node_props carries mode and the baked frame",
          props.get('mode') == 'AUTO' and props.get('proj_mode') == 'PERSPECTIVE'
          and len(props.get('object_inv', ())) == 12)
    # the GPU twin, bitwise on all three outputs
    c, V, extra = lane_ctx(screen=False)
    lanes = {'lv': ('vec3', V)}
    for mode in ('FRAME', 'PERSPECTIVE', 'ORTHO'):
        node = object_map_node(mode, inv12)
        for i, name in enumerate(('Vector', 'Inside', 'Depth')):
            cpu = cpu_eval(node, lanes, c, i)
            g, why = twin(node, lanes, LANE_N, i, extra)
            check(f'{mode} {name} is bitwise the GPU twin',
                  g is not None and bool(np.array_equal(cpu, gpu_cols(g, cpu))),
                  str(why))
        if mode == 'PERSPECTIVE':
            check('PERSPECTIVE Inside is not constant on random lanes',
                  0 < int(cpu_eval(node, lanes, c, 1).sum()) < LANE_N)
    # the exported numbers through the GPU: proj_mode AUTO resolved
    node = object_map_node('AUTO', inv12, extra_props={'proj_mode': 'PERSPECTIVE',
                                                       'proj_tan': lit8(math.tan(0.6)),
                                                       'proj_half': 1.0})
    cpu = cpu_eval(node, lanes, c, 0)
    g, why = twin(node, lanes, LANE_N, 0, extra)
    check('an exported proj_mode / proj_tan twins bitwise',
          g is not None and bool(np.array_equal(cpu, gpu_cols(g, cpu))), str(why))
    # the linked-Angle refusal, by name
    node = object_map_node('PERSPECTIVE', inv12, angle_link='la')
    nodes = {'o': node,
             'lv': {'id': 'lv', 'bl_idname': '__R253_LANE__', 'props': {},
                    'inputs': [], 'outputs': [{'name': 'V', 'type': 'VECTOR'}]},
             'la': {'id': 'la', 'bl_idname': '__R253_LANE__', 'props': {},
                    'inputs': [], 'outputs': [{'name': 'V', 'type': 'VALUE'}]}}
    EM.EMITTERS['__R253_LANE__'] = lambda em, nd, _i: ('hal_lane', 'vec3')
    try:
        em = EM.Emitter({'output': 'o', 'nodes': nodes})
        err = None
        try:
            em.output('o', 0)
        except EM.Unsupported as exc:
            err = str(exc)
    finally:
        EM.EMITTERS.pop('__R253_LANE__', None)
    text = ('HALCYON_ObjectMapNode (a linked Angle needs tan() at shading '
            'time; set the Angle on the node)')
    check('a linked Angle socket refuses BY NAME on the GPU',
          err == text and text in em.unsupported, f'{err!r}')
    ang = (np.full(LANE_N, 30.0, f32))
    cpu = cpu_eval(node, {'lv': ('vec3', V), 'la': ('float', ang)}, c, 0)
    ref = cpu_eval(object_map_node('PERSPECTIVE', inv12, angle=30.0), lanes, c, 0)
    zc = cpu_eval(node, {'lv': ('vec3', V), 'la': ('float', ang)}, c, 2)
    front = zc > 0.1          # the 1e-6 depth guard amplifies the rest
    rel = np.abs(cpu[front] - ref[front]) / np.maximum(np.abs(ref[front]), 1.0)
    check('the CPU shades the linked Angle per lane (30 degrees == the socket at 30, '
          'relative 1e-5 in front of the lens)',
          float(rel.max()) <= 1e-5 and int(front.sum()) > 100, f'{float(rel.max()):.2e}')
    # the matrix literals lift under mark_values (the R174 contract)
    nodes = {'o': object_map_node('FRAME', inv12, link=None)}
    em = EM.Emitter({'output': 'o', 'nodes': nodes})
    em.mark_values = True
    em.output('o', 0)
    check('the twelve matrix floats are hal_MV() literals under mark_values '
          '(a moved projector re-uploads, never recompiles)',
          em.body().count('hal_MV(') >= 12 and 'hal_P' in em.body())


# ======================================================== hex grid
def test_hex_grid_cells():
    """The centre of cell (0,0) maps to Vector (0.5, 0.5) with Edge 1;
    points one cell-step apart share the local UV (within float32, 4e-6);
    the six neighbours of a cell carry six distinct Cell IDs; Edge is 0
    on a shared edge midpoint (within 1e-6) and never negative inside;
    POINTY vs FLAT are the x/y swap; CPU vs node_twin bitwise."""
    sc = 4.0
    SQ3 = 1.7320508
    c1 = _ctx(1)
    out = cpu_eval(hex_node(), {'lv': ('vec3', np.zeros((1, 3), f32))}, c1, 0)
    edge = cpu_eval(hex_node(), {'lv': ('vec3', np.zeros((1, 3), f32))}, c1, 2)
    check('the centre of cell (0,0) maps to Vector (0.5, 0.5) with Edge 1',
          bool(np.array_equal(out[0, :2], [0.5, 0.5])) and float(edge[0]) == 1.0,
          f'{out[0]} {edge[0]}')
    # one cell-step apart: (sqrt3, 0) and (sqrt3/2, 1.5) in cell units
    rng = lanes_rng(11)
    n = 512
    pts = (rng.random((n, 3)).astype(f32) * 0.6 - 0.3).astype(f32)
    pts[:, 2] = 0.0
    base = cpu_eval(hex_node(), {'lv': ('vec3', (pts / f32(sc)).astype(f32))}, _ctx(n), 0)
    for step in ((SQ3, 0.0), (SQ3 * 0.5, 1.5), (-SQ3 * 0.5, -1.5)):
        sh = pts.copy()
        sh[:, 0] += f32(step[0])
        sh[:, 1] += f32(step[1])
        got = cpu_eval(hex_node(), {'lv': ('vec3', (sh / f32(sc)).astype(f32))}, _ctx(n), 0)
        d = float(np.abs(got[:, :2] - base[:, :2]).max())
        check(f'points one cell-step {step} apart share the local UV (4e-6)',
              d <= 4e-6, f'max {d:.2e}')
        check(f'and carry a different Cell ID than the base cell ({step})',
              not np.array_equal(got[:, 2], base[:, 2]))
    # six distinct neighbours
    nb = np.array([[SQ3, 0], [-SQ3, 0], [SQ3 * 0.5, 1.5], [-SQ3 * 0.5, 1.5],
                   [SQ3 * 0.5, -1.5], [-SQ3 * 0.5, -1.5]], f32) / f32(sc)
    nb3 = np.c_[nb, np.zeros(6, f32)].astype(f32)
    ids = cpu_eval(hex_node(), {'lv': ('vec3', nb3)}, _ctx(6), 1)
    ids0 = cpu_eval(hex_node(), {'lv': ('vec3', np.zeros((1, 3), f32))}, c1, 1)
    check('the six neighbours of a cell carry six distinct Cell IDs',
          len(set(ids.tolist())) == 6 and float(ids0[0]) not in set(ids.tolist()))
    check('Cell ID is the TileRandom integer hash (patterns.hash3 at salt 53)',
          float(ids0[0]) == float(NE.PT.hash3(np.int64(0), np.int64(0), np.int64(53))))
    # edge midpoints: (0.866, 0) and (0.433, 0.75) in cell units
    mids = np.array([[0.8660254, 0.0, 0], [-0.8660254, 0.0, 0],
                     [0.4330127, 0.75, 0], [-0.4330127, -0.75, 0]], f32) / f32(sc)
    e = cpu_eval(hex_node(), {'lv': ('vec3', mids.astype(f32))}, _ctx(4), 2)
    check('Edge is 0 on a shared edge midpoint (within 1e-6)',
          float(np.abs(e).max()) <= 1e-6, str(e))
    inside = (rng.random((n, 3)).astype(f32) * 1.6 - 0.8).astype(f32)
    inside[:, 2] = 0.0
    r = np.hypot(inside[:, 0], inside[:, 1])
    inside = inside[r < 0.8]
    e = cpu_eval(hex_node(), {'lv': ('vec3', (inside / f32(sc)).astype(f32))},
                 _ctx(inside.shape[0]), 2)
    check('Edge is never negative inside and > 0 strictly inside',
          float(e.min()) > 0.0 and float(e.max()) <= 1.0)
    vtx = np.array([[0.0, 1.0, 0.0]], f32) / f32(sc)
    e = cpu_eval(hex_node(), {'lv': ('vec3', vtx.astype(f32))}, c1, 2)
    check('Edge is 0 at a vertex too', float(np.abs(e[0])) <= 1e-6)
    # POINTY vs FLAT: the x/y swap
    c, V, extra = lane_ctx(screen=False)
    lanes = {'lv': ('vec3', V)}
    Vs = V.copy()
    Vs[:, 0], Vs[:, 1] = V[:, 1], V[:, 0]
    flat = [cpu_eval(hex_node('FLAT'), lanes, c, i) for i in range(3)]
    pointy_sw = [cpu_eval(hex_node('POINTY'), {'lv': ('vec3', Vs)}, c, i)
                 for i in range(3)]
    check('FLAT(x, y) is POINTY(y, x) with the local UV swapped back',
          bool(np.array_equal(flat[0][:, 0], pointy_sw[0][:, 1]))
          and bool(np.array_equal(flat[0][:, 1], pointy_sw[0][:, 0]))
          and bool(np.array_equal(flat[1], pointy_sw[1]))
          and bool(np.array_equal(flat[2], pointy_sw[2])))
    uv = cpu_eval(hex_node(), lanes, c, 0)
    check('the local UV stays inside 0..1',
          float(uv[:, :2].min()) >= -1e-6 and float(uv[:, :2].max()) <= 1 + 1e-6)
    check('Vector z carries the Cell ID',
          bool(np.array_equal(uv[:, 2], cpu_eval(hex_node(), lanes, c, 1))))
    for orient in ('POINTY', 'FLAT'):
        node = hex_node(orient)
        for i, name in enumerate(('Vector', 'Cell ID', 'Edge')):
            cpu = cpu_eval(node, lanes, c, i)
            g, why = twin(node, lanes, LANE_N, i, extra)
            check(f'{orient} {name} is bitwise the GPU twin',
                  g is not None and bool(np.array_equal(cpu, gpu_cols(g, cpu))),
                  str(why))
    em, err = emit_body(hex_node())
    check('the hex emitter rounds with floor(x + 0.5), the add in its own '
          'statement, and pulls the hash through _need_prims',
          err is None and '+ 0.5;' in em.body() and 'floor(' in em.body()
          and 'roundEven' not in em.body() and '__pt_prims' in em.once)


# ===================================== the Image Texture's projections
def test_image_projection_twins_on_the_gpu():
    """ShaderNodeTexImage with projection SPHERE / TUBE / BOX under
    NEAREST, CPU n_tex_image vs node_twin on a 64 x 64 index image:
    BOX bitwise, SPHERE / TUBE the same texel on >= 99.9% of 4096 lanes
    (the sampled texel ids compare exactly where the UV rounds the
    same) and never more than one texel apart; FLAT stays bitwise the
    pre-edit text."""
    c, V, extra = lane_ctx(screen=False)
    lanes = {'lv': ('vec3', V)}
    tex = TX.Texture(index_image(64), colorspace='Non-Color', filt='NEAREST',
                     wrap='REPEAT')
    images = {'checker': tex}
    ex = dict(extra)
    ex['hal_tex0'] = tex
    for proj in ('FLAT', 'SPHERE', 'TUBE', 'BOX'):
        node = tex_image_node(proj)
        cpu = cpu_eval(node, lanes, c, 0, images)
        g, why = twin(node, lanes, LANE_N, 0, ex)
        if not check(f'{proj}: the GPU twin runs', g is not None, str(why)):
            continue
        same = np.abs(cpu - g).max(1) <= 1e-7
        ids_c = np.rint(cpu[:, :2] * 63).astype(int)
        ids_g = np.rint(g[:, :2] * 63).astype(int)
        apart = np.abs(ids_c - ids_g)
        apart = np.minimum(apart, 64 - apart).max(1)
        if proj in ('FLAT', 'BOX'):
            check(f'{proj}: bitwise the CPU', bool(np.array_equal(cpu, g)))
        else:
            check(f'{proj}: the same texel on >= 99.9% of lanes, never more '
                  'than one texel apart',
                  float(same.mean()) >= 0.999 and int(apart.max()) <= 1,
                  f'{float(same.mean()):.5f} {int(apart.max())}')
        check(f'{proj}: not the FLAT picture (the projection is applied)'
              if proj != 'FLAT' else 'FLAT: samples uv.xy',
              proj == 'FLAT' or not np.array_equal(
                  cpu, cpu_eval(tex_image_node('FLAT'), lanes, c, 0, images)))
    node = tex_image_node('FLAT', link=None)
    em = EM.Emitter({'output': 'ti', 'nodes': {'ti': node}})
    em.output('ti', 0)
    check('the FLAT text is byte-identical to the pre-edit emitter',
          em.body() == OLD_TEX_IMAGE_FLAT, em.body())
    check('and its sampler keeps the footprint', em.samplers[0]['footprint'] is True)
    em, err = emit_body(tex_image_node('BOX'))
    check('BOX reads hal_N (the normal picks the face, as _box_project)',
          err is None and 'abs(hal_N)' in em.body())
    em, err = emit_body(tex_image_node('SPHERE'))
    check('SPHERE is the equirect text (no normalize)',
          err is None and 'atan(' in em.body() and 'normalize' not in em.body()
          and '6.283185307179586' in em.body())
    # the CPU's Image Texture is untouched: the volume Box and the flat
    # road are n_tex_image's own (a FLAT sample equals a direct tex.sample)
    flat = cpu_eval(tex_image_node('FLAT'), lanes, c, 0, images)
    direct = tex.sample(V[:, 0], V[:, 1], filt='NEAREST', wrap='REPEAT')
    check('CPU FLAT is a direct top-level sample (untouched)',
          bool(np.array_equal(flat, direct)))


# =============================================================== tables
def test_mapping_nodes_in_tables():
    """The five new ids and the Projector's nine new identifiers are in
    DISPATCH, EMITTERS, NODE_PROPS (FAMILY_NODE_PROPS or explicit), one
    MENU_FAMILIES family each, verified icons, PROP_TIPS / ITEM_TIPS
    coverage; the hand-written nodes carry the exact callback
    signatures and >= 40-char tooltips on every property."""
    _bpy()
    from ..nodes import shader_nodes as SN
    from ..export import NODE_PROPS
    from .blender_icons import ICONS as VERIFIED_ICONS
    for idn in NEW_IDS:
        check(f'{idn} is in DISPATCH', idn in NE.DISPATCH)
        check(f'{idn} is in EMITTERS (no REFUSED entry)',
              idn in EM.EMITTERS and idn not in EM.REFUSED)
        cls = next((k for k in SN.NODES if k.bl_idname == idn), None)
        check(f'{idn} is in NODES', cls is not None)
        fams = [t for t, _i, m in SN.MENU_FAMILIES if cls in m]
        check(f'{idn} sits in exactly one family', len(fams) == 1, str(fams))
        if cls is not None:
            check(f'{idn} carries a verified icon ({cls.bl_icon})',
                  cls.bl_icon in VERIFIED_ICONS)
            check(f'{idn}: init(self, context) / draw_buttons(self, context, layout)',
                  cls.init.__code__.co_argcount == 2
                  and cls.draw_buttons.__code__.co_argcount == 3)
            for pname, pdef in getattr(cls, '__annotations__', {}).items():
                kw = getattr(pdef, 'kw', {})
                check(f'{idn}.{pname} tooltip >= 40 chars',
                      len(str(kw.get('description') or '')) >= 40)
                items = kw.get('items')
                if items and not callable(items):
                    check(f'{idn}.{pname}: every item line >= 12 chars',
                          all(len(str(i[2])) >= 12 for i in items))
                if getattr(pdef, 'kind', '') in ('EnumProperty', 'BoolProperty',
                                                 'IntProperty', 'FloatProperty'):
                    check(f'{idn}.{pname} is in export.NODE_PROPS',
                          pname in NODE_PROPS.get(idn, ()))
    check('Tri-Planar Texture, Object Projection, Tri-Planar Coordinates and '
          'Hex Grid draw in the Vector family',
          {c.bl_idname for t, _i, m in SN.MENU_FAMILIES if t == 'Vector'
           for c in m if not isinstance(c, tuple)}
          >= {'HALCYON_TriPlanarTextureNode', 'HALCYON_ObjectMapNode',
              'HALCYON_TriPlanarNode', 'HALCYON_HexGridNode'})
    check('Tri-Planar Mix lands in Utilities',
          'HALCYON_TriPlanarMixNode' in {
              c.bl_idname for t, _i, m in SN.MENU_FAMILIES if t == 'Utilities'
              for c in m if not isinstance(c, tuple)})
    check('NODE_PROPS lists the hand-written nodes',
          NODE_PROPS.get('HALCYON_TriPlanarTextureNode') == ('interpolation', 'extension')
          and NODE_PROPS.get('HALCYON_ObjectMapNode') == ('mode',)
          and NODE_PROPS.get('HALCYON_HexGridNode') == ('orient',)
          and NODE_PROPS.get('HALCYON_ProjectorNode') == ('mode', 'axis', 'aspect'))
    # the Projector's enum coverage
    spec = next(s for s in SN.VEC_SPECS if s[0] == 'Projector')
    items = spec[5]['mode'][2]
    check('every appended Projector mode carries >= 20 chars of its own '
          '(the ITEM_TIPS back-fill never replaces it)',
          all(len(i[2]) >= 20 for i in items if i[0] in NEW_MODES))
    tips = SN.ITEM_TIPS[('HALCYON_ProjectorNode', 'mode')]
    check('ITEM_TIPS covers all thirteen Projector modes',
          all(i[0] in tips and len(tips[i[0]]) >= 12 for i in items))
    check('PROP_TIPS covers aspect and the hex orientation',
          len(SN.PROP_TIPS[('HALCYON_ProjectorNode', 'aspect')]) >= 40
          and len(SN.PROP_TIPS[('HALCYON_HexGridNode', 'orient')]) >= 40
          and 'cube' in SN.PROP_TIPS[('HALCYON_ProjectorNode', 'mode')].lower())
    cls = next(k for k in SN.NODES if k.bl_idname == 'HALCYON_ProjectorNode')
    kw = cls.__annotations__['mode'].kw
    check('the registered Projector menu lists thirteen modes, PLANAR first',
          [i[0] for i in kw['items']][:4] == ['PLANAR', 'CYLINDER', 'SPHERE', 'BOX']
          and len(kw['items']) == 13 and kw['default'] == 'PLANAR')
    check("aspect defaults to False (bitwise-neutral)",
          cls.__annotations__['aspect'].kw['default'] is False)
    # the hand-written nodes' sockets, through a recording stub
    class _Socks:
        def __init__(self):
            self.made = []

        def new(self, kind, name):
            s = type('S', (), {})()
            s.kind, s.name = kind, name
            self.made.append(s)
            return s

    for idn, n_in, n_out in (('HALCYON_TriPlanarTextureNode', 4, 2),
                             ('HALCYON_ObjectMapNode', 3, 3)):
        cls = next(k for k in SN.NODES if k.bl_idname == idn)
        node = cls.__new__(cls)
        node.inputs, node.outputs = _Socks(), _Socks()
        cls.init(node, None)
        check(f'{idn}: init makes {n_in} inputs and {n_out} outputs, Vector first',
              len(node.inputs.made) == n_in and len(node.outputs.made) == n_out
              and node.inputs.made[0].name == 'Vector',
              str([s.name for s in node.inputs.made]))
    # the family specs' sockets, through the generated classes
    for name, n_in, n_out in (('HALCYON_TriPlanarNode', 4, 6),
                              ('HALCYON_HexGridNode', 2, 3),
                              ('HALCYON_TriPlanarMixNode', 6, 2)):
        spec = next(s for s in SN.VEC_SPECS + SN.UTIL_SPECS
                    if f'HALCYON_{s[0]}Node' == name)
        check(f'{name}: {n_in} inputs / {n_out} outputs in the spec',
              len(spec[4]) == n_in and len(spec[6]) == n_out
              and len(spec[3]) >= 40)


# ============================================================ frames
def _ball_master(sc, graph_nodes, diffuse_link):
    """The demo Ball on the master shader with Diffuse Color linked."""
    ins = _master_ins()
    for s in ins:
        if s['name'] == 'Diffuse Color':
            s['link'] = diffuse_link
    g = _one_bsdf_graph('HALCYON_ShaderNode', ins, {'model': 'PHONG',
                                                     'toon_steps': 2})
    g['nodes'].update(graph_nodes)
    return g


def test_mapping_frame_twin():
    """A demo-scene material driving the master shader's Diffuse through
    Tri-Planar Texture, and another through Object Projection (FRAME,
    a scaled + turned frame) -> Projector EQUIRECT -> Image Texture,
    under frame_bar (deferred bar 6e-3, no pixel > 1e-2); and the CPU
    picture of three pre-existing featurematrix scenes byte-identical to
    the pre-edit tree (the bitwise-neutral rule)."""
    st = base_settings(W, H, transparency='NONE')
    st.use_processes = False
    sc = demo_scene(st, with_texture=True)
    sc.images['index'] = ImageBuffer(name='index', pixels=index_image(64))
    tri = tri_texture_node(image='index', interp='Closest', scale=0.5, blend=0.5,
                           link_n=None, link_v=None)
    sc.materials[1].graph = _ball_master(sc, {'tt': tri}, ['tt', 0])
    out = frame_bar('Tri-Planar Texture on the Ball', sc, st)
    if out is not None:
        cpu = np.asarray(R.render(sc, st))
        check('the tri-planar ball is not flat (the texture varies)',
              float(cpu[:, :, :3].std()) > 0.05)
    st2 = base_settings(W, H, transparency='NONE')
    st2.use_processes = False
    sc2 = demo_scene(st2, with_texture=True)
    sc2.images['index'] = ImageBuffer(name='index', pixels=index_image(64))
    omap = object_map_node('FRAME', inv12_of(trs_matrix()), link=None)
    omap['id'] = 'om'
    proj = projector('EQUIRECT', 'Z', scale=1.0, link='om')
    proj['id'] = 'pj'
    img = tex_image_node('FLAT', image='index', link='pj')
    img['id'] = 'im'
    sc2.materials[1].graph = _ball_master(sc2, {'om': omap, 'pj': proj, 'im': img},
                                          ['im', 0])
    frame_bar('Object Projection -> Projector EQUIRECT -> Image Texture', sc2, st2)
    # every pre-existing scene: the pre-edit bytes
    for key, want in OLD_FRAME_DIGESTS.items():
        st3 = base_settings(W, H, transparency='NONE')
        st3.use_processes = False
        img3 = np.asarray(R.render(FM.SCENES[key](st3), st3))
        dig = zlib.adler32(np.ascontiguousarray(img3, f32).tobytes())
        check(f"the '{key}' featurematrix scene renders the pre-edit bytes "
              f'(adler32 {want})', dig == want, str(dig))


def main():
    utf8_console()
    FAILS.clear()
    order = ['test_projector_old_modes_frozen', 'test_projector_new_modes_twin',
             'test_projector_screen_is_the_pixel', 'test_tri_planar_weights',
             'test_tri_planar_mix_twin', 'test_tri_planar_texture_samples_three_faces',
             'test_object_projection_node', 'test_hex_grid_cells',
             'test_image_projection_twins_on_the_gpu', 'test_mapping_nodes_in_tables',
             'test_mapping_frame_twin']
    tests = {n: f for n, f in globals().items() if n.startswith('test_') and callable(f)}
    names = [n for n in order if n in tests] + [n for n in sorted(tests) if n not in order]
    for n in names:
        print(f'--- {n}')
        try:
            tests[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print(f'\nR253 MAPPING TYPES: {len(FAILS)} failure(s)')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
