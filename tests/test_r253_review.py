"""R253 review fixes: the adversarial re-read's confirmed findings, each
pinned where it bit.

(1) A singular object matrix (a zero scale on any axis) raised
    LinAlgError out of the whole frame on both Generated roads;
    `core/render.object_inverse` takes the pseudo-inverse and says so once.
(2) A console TYPE can pin the opacity below 1 with its socket hidden at
    1.0 (the Saturn mesh colour calculation): the alpha classifiers never
    saw it, so the half-transparent polygon stayed in the opaque pass.
    `core/console.pinned_alpha_reason` feeds both `material_see_through`
    and `export._alpha_reason`.
(3) The corner road's ALBEDO pass returned its raw fractional opacity:
    no punch-through clip, no Screen Door compare, no Transparency None.
    `ShadeJob._alpha_law` is the one law the pixel road already applied,
    now applied there too.
(4) The PC fixed-function Texture Op was drawn for every shading type
    and honoured by the separate-specular combine alone; D3DTSS_COLOROP
    is texture-stage state, so `core/combine.cb_pc_texop` and its GLSL
    twin run the other three types' product through the same 8-bit ops.
(5) The vertex-colour material sources (GX_SRC_VTX, the N64's unlit
    shade, Model 3's fixed shading, D3D's COLORVERTEX) replaced the
    TEXTURED albedo with the vertex colour; the hardware replaces the
    material colour and the texture still modulates -- a linked Diffuse
    Color is now multiplied, an unlinked one replaced, on both roads.
(6) Super FX and 3DO hide their specular sockets but left the hidden
    socket's 0.5 in the light loop; resolve() pins the level at 0.
(7) A 1.90 Model 2 / Model 3 master picked its specular setting by
    snapping Glossiness; the migrated node took the menu default (P4 /
    P16) and the highlight changed at file load. The migration snaps
    the old sockets the same way.
(8) The Model 2 / Model 3 specular menus had labels under 12 characters.
(9) The GPU plan signature omitted the baked per-object frame (the
    inverse rows, the Generated box): a translation keeps the mesh key,
    so the cache served stale literals. Fingerprinted now.
(10) A TRANSFORMED flat object divided the inverse chain's float32
    residual by the 1e-6 span clamp and read +/-0.5 of noise on its flat
    axis; `_gen_span` gives a flat axis of a transformed object the span
    1.0 (an identity object keeps the 1.91 bits).
"""
import io
import os
import sys
import traceback
import types
from contextlib import redirect_stdout

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from ..core import console as CON
from ..core import render as R
from ..core.scene import material_see_through
from .test_r252_console import _scene as console_scene, console_graph, _render as console_render
from .test_render import _sk
from .test_r251_material import _sim_vs_cpu

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f'  {extra}' if extra and not ok else ''))
    if not ok:
        FAILS.append(name)


def _zero_scale_scene(space):
    st = base_settings(48, 36, shadows=False, generated_space=space)
    sc = demo_scene(st, with_texture=False)
    m = np.eye(4, dtype=f32)
    m[2, 2] = 0.0
    sc.objects[1].matrix_world = m
    return sc, st


def test_singular_matrix_renders():
    R._SINGULAR_SAID.clear()
    for space in ('OBJECT', 'WORLD'):
        sc, st = _zero_scale_scene(space)
        buf = io.StringIO()
        try:
            with redirect_stdout(buf):
                img = R.render(sc, st)
            ok = bool(np.isfinite(img).all())
        except Exception as exc:                                # noqa: BLE001
            img, ok = None, False
            print(f'  raised {type(exc).__name__}: {exc}')
        check(f'a zero-scale object renders a finite frame under generated_space {space}', ok)
        if space == 'OBJECT':
            check('and says so once by name', "object 'Ball': its matrix is singular" in buf.getvalue())
    # the helper itself: identity in, identity out (bitwise), singular -> finite
    eye = np.eye(4, dtype=f32)
    check('object_inverse(identity) is the identity bit for bit', bool(np.array_equal(R.object_inverse(eye), eye)))
    m = np.diag([2.0, 1.0, 0.0, 1.0]).astype(f32)
    inv = R.object_inverse(m, types.SimpleNamespace(name='flat'))
    check('object_inverse of a zero-scale matrix is finite and inverts the live axes',
          bool(np.isfinite(inv).all()) and abs(float(inv[0, 0]) - 0.5) < 1e-6 and abs(float(inv[1, 1]) - 1.0) < 1e-6
          and abs(float(inv[2, 2])) < 1e-6)
    check('None is the identity', bool(np.array_equal(R.object_inverse(None), eye)))


def test_console_pinned_opacity_is_alpha_evidence():
    res = CON.resolve({'console': 'SATURN', 'sat_type': 'MESH'})
    check('resolve pins the Saturn mesh opacity at 0.5', res.get('opacity') == 0.5)
    g = console_graph({'console': 'SATURN', 'sat_type': 'MESH'})
    why = CON.pinned_alpha_reason(g)
    check('pinned_alpha_reason names the machine, the type and the value',
          why is not None and 'Saturn' in why and 'mesh' in why.lower() and '0.500' in why, str(why))
    check('and is None for the Gouraud type and for a non-console graph',
          CON.pinned_alpha_reason(console_graph({'console': 'SATURN', 'sat_type': 'GOURAUD'})) is None
          and CON.pinned_alpha_reason(None) is None and CON.pinned_alpha_reason({'nodes': {}}) is None)
    sc, st = console_scene({'console': 'SATURN', 'sat_type': 'MESH'})
    check('material_see_through reads the pin (the split and the layer plan agree)',
          material_see_through(sc.materials[1]) == why)
    # the exporter's classifier (bpy-free: a material record with the graph)
    try:
        from . import fakebpy as _fb
        _fb.install()
        from .. import export as EX
        m = types.SimpleNamespace(opacity=1.0, graph=g)
        check('export._alpha_reason reads the pin before the hidden sockets',
              EX._alpha_reason(None, m) == why, str(EX._alpha_reason(None, m)))
    except ImportError as exc:
        check('export is importable under fakebpy', False, str(exc))
    # the picture: under Sorted the mesh polygon now blends with what is behind
    st.transparency = 'SORTED'
    mesh = np.asarray(R.render(sc, st))
    sc2, st2 = console_scene({'console': 'SATURN', 'sat_type': 'GOURAUD'})
    st2.transparency = 'SORTED'
    gour = np.asarray(R.render(sc2, st2))
    d = np.abs(mesh[:, :, :3] - gour[:, :, :3]).max(axis=2)
    check('under Sorted the Saturn mesh Ball differs from the Gouraud Ball (it reached the transparent pass)',
          int((d > 1e-3).sum()) > 100, str(int((d > 1e-3).sum())))


def test_corner_road_alpha_law():
    # Screen Door: the corner road's alpha is the compare's 0 / 1, as the pixel road's
    sc, st = console_scene({'console': 'SATURN', 'sat_type': 'MESH'})
    st.transparency = 'STIPPLE'
    img = np.asarray(R.render(sc, st))
    vals = set(np.round(img[:, :, 3].ravel(), 3).tolist())
    check('Screen Door: the Saturn mesh frame alpha is the compare bit (0 or 1), never 0.5',
          vals <= {0.0, 1.0} and 0.0 in vals and 1.0 in vals, str(sorted(vals))[:60])
    # the pixel road's own rule on the same frame, for the record
    st.transparency = 'NONE'
    img = np.asarray(R.render(sc, st))
    check('Transparency None: the corner road writes alpha 1.0', bool(np.all(img[:, :, 3] == 1.0)))
    # an opaque corner material is bitwise unchanged by the law (default neutrality)
    for tr in ('NONE', 'STIPPLE', 'SORTED'):
        sc_a, st_a = console_scene({'console': 'SATURN', 'sat_type': 'GOURAUD'})
        st_a.transparency = tr
        a = np.asarray(R.render(sc_a, st_a))
        check(f'an opaque corner material under {tr} keeps alpha 1.0 everywhere it covers',
              bool(np.all(a[:, :, 3][a[:, :, 3] > 0] == 1.0)))
    # the law is one method, used by both roads
    import inspect
    src = inspect.getsource(R.ShadeJob)
    check('the pixel road and the ALBEDO pass call the same _alpha_law',
          src.count('self._alpha_law(') >= 2 and 'def _alpha_law' in src)


def _textured(props, **kw):
    """A console scene whose Ball and Box carry the checker texture in
    Diffuse Color (the linked-texture road)."""
    sc, st = console_scene(props, inputs=[_sk('Diffuse Color', 'RGBA', [0.85, 0.2, 0.15, 1.0], ['tex', 0])], **kw)
    for m in (sc.materials[1], sc.materials[2]):
        m.graph['nodes']['tex'] = {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                                   'props': {'image': 'checker', 'interpolation': 'Closest', 'extension': 'REPEAT'},
                                   'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                                   'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]}
    return sc, st


def _dmax(a, b):
    return float(np.abs(np.asarray(a)[:, :, :3] - np.asarray(b)[:, :, :3]).max())


def _twin(sc, st):
    st.render_device = 'GPU'
    d, nbad, _img, _cpu, _passes, why = _sim_vs_cpu(sc, st)
    return d, nbad, why


def test_pc_texture_op_on_every_type():
    for t in ('GOURAUD', 'FLAT_GL', 'FLAT_D3D', 'GOURAUD_SEP'):
        a = console_render({'console': 'PC_FIXED', 'pc_type': t, 'pc_texture_op': 'MODULATE'})
        for op in ('MODULATE2X', 'MODULATE4X', 'ADD', 'ADDSIGNED'):
            b = console_render({'console': 'PC_FIXED', 'pc_type': t, 'pc_texture_op': op})
            check(f'PC {t}: Texture Op {op} changes the picture', _dmax(a, b) > 1e-2)
        for op in ('MODULATE4X', 'ADDSIGNED'):
            sc, st = console_scene({'console': 'PC_FIXED', 'pc_type': t, 'pc_texture_op': op}, w=48, h=36, shading_rate='PIXEL')
            d, nbad, why = _twin(sc, st)
            check(f'PC {t} {op}: the simulator frame is the CPU frame', d is not None and d < 6e-3 and nbad == 0, str(why if d is None else d))
    # MODULATE on the three plain types is the plain product (1.90 bits): the recombine hook declines
    from ..core import combine as CB
    sc, st = console_scene({'console': 'PC_FIXED', 'pc_type': 'GOURAUD', 'pc_texture_op': 'MODULATE'})
    check('cb_pc_texop(MOD) is None (the plain product stays)', CB.cb_pc_texop(np.ones((2, 3), f32), np.ones((2, 3), f32), 'MOD') is None)
    check('PC_TEXOP_MODELS names the three non-combine PC models',
          CB.PC_TEXOP_MODELS == frozenset({'GOURAUD', 'FLAT_GL_LAST', 'FLAT_D3D_FIRST'}))


def test_vertex_colour_keeps_a_linked_texture():
    cases = (('GX lit, material source VTX', {'console': 'GAMECUBE', 'gc_material_src': 'VTX'}),
             ('GX vertex', {'console': 'GAMECUBE', 'gc_type': 'VERTEX'}),
             ('N64 vertex (G_LIGHTING off)', {'console': 'N64', 'n64_type': 'VERTEX'}),
             ('Model 3 fixed shading', {'console': 'MODEL3', 'm3_type': 'FIXED'}),
             ('PC colour vertex', {'console': 'PC_FIXED', 'pc_color_vertex': True}))
    for name, props in cases:
        sc_t, st_t = _textured(props)
        sc_u, st_u = console_scene(props)
        check(f'{name}: the linked texture is visible (modulated by the vertex colour, not replaced)',
              _dmax(R.render(sc_t, st_t), R.render(sc_u, st_u)) > 0.1)
        sc_g, st_g = _textured(props, w=48, h=36, shading_rate='PIXEL')
        d, nbad, why = _twin(sc_g, st_g)
        check(f'{name}: the simulator frame is the CPU frame with the texture', d is not None and d < 6e-3 and nbad == 0, str(why if d is None else d))
    # an unlinked Diffuse Color is still REPLACED by the vertex colour (the 1.91 picture)
    a = console_render({'console': 'GAMECUBE', 'gc_type': 'VERTEX'})
    b = console_render({'console': 'GAMECUBE', 'gc_type': 'VERTEX'}, inputs=[_sk('Diffuse Color', 'RGBA', [0.1, 0.1, 0.9, 1.0])])
    check('an unlinked Diffuse Color under a vertex-colour type is replaced (the socket colour does not show)', _dmax(a, b) < 1e-6)


def test_hidden_specular_pins():
    for con in ('SUPERFX', 'THREEDO'):
        kw = dict(palette_mode='EGA16', gamma=1.0, color_management='NONE')
        a = console_render({'console': con}, inputs=[_sk('Specular Level', 'VALUE', 0.0)], **kw)
        b = console_render({'console': con}, inputs=[_sk('Specular Level', 'VALUE', 0.9)], **kw)
        check(f'{con}: the hidden Specular Level socket cannot add a highlight', _dmax(a, b) < 1e-6)
        check(f'{con}: resolve pins spec_level 0', CON.resolve({'console': con}).get('spec_level') == 0.0)


def test_model23_migration_keeps_the_exponent():
    from . import fakebpy
    fakebpy.install()
    from ..nodes import shader_nodes as SN
    from ..core import shading as SH
    N = SN.HALCYON_ConsoleShaderNode

    class _Sock:
        def __init__(self, name, value=None):
            self.name, self.default_value, self.is_linked, self.links, self.hide = name, value, False, [], False

    class _Inputs(list):
        def get(self, name):
            return next((s for s in self if s.name == name), None)

        def new(self, kind, name):
            s = _Sock(name); self.append(s); return s

    class _Node:
        def __init__(self, idn):
            self.bl_idname, self.inputs, self.outputs, self.location, self.label, self.hide, self._raw = idn, _Inputs(), _Inputs(), (0, 0), '', False, {}

        def get(self, k):
            return self._raw.get(k)

    class _Links(list):
        def new(self, a, b):
            self.append((a, b)); b.is_linked = True; return (a, b)

    class _Nodes(list):
        def new(self, idn):
            n = _Node(idn)
            if idn == 'HALCYON_ConsoleShaderNode':
                for kind, name, d in N.SOCKETS:
                    n.inputs.new(kind, name).default_value = d
                n.outputs.new('NodeSocketShader', 'Surface')
                for name, _i, d in CON.ENUM_PROPS:
                    setattr(n, name, d)
                for name, _k, d in CON.SCALAR_PROPS:
                    setattr(n, name, d)
                n.refresh_sockets = lambda: None
            self.append(n); return n

        def remove(self, n):
            if n in self:
                super().remove(n)

    class _Tree:
        def __init__(self):
            self.nodes, self.links = _Nodes(), _Links()

    def migrated(model, gloss, level=0.5):
        tree = _Tree()
        old = _Node('HALCYON_ShaderNode')
        for kind, name, d in SN.HALCYON_ShaderNode.SOCKETS:
            old.inputs.new(kind, name).default_value = d
        old.inputs.get('Glossiness').default_value = gloss
        old.inputs.get('Specular Level').default_value = level
        old.outputs.new('NodeSocketShader', 'Surface')
        old.model = ''
        old._raw['model'] = [m[0] for m in SH.MODEL_ITEMS].index(model)
        tree.nodes.append(old)
        new = SN.migrate_master_node(tree, old)
        return new

    for gloss, want in ((1.0, 'P1'), (2.0, 'P2'), (4.0, 'P4'), (25.0, 'P8')):
        n = migrated('SEGA_MODEL2', gloss)
        check(f'a 1.90 Model 2 master at Glossiness {gloss} migrates to {want}', getattr(n, 'm2_specular', None) == want, str(getattr(n, 'm2_specular', None)))
    for gloss, want in ((8.0, 'P8'), (16.0, 'P16'), (25.0, 'P32'), (64.0, 'P64')):
        n = migrated('SEGA_MODEL3', gloss)
        check(f'a 1.90 Model 3 master at Glossiness {gloss} migrates to {want}', getattr(n, 'm3_specular', None) == want, str(getattr(n, 'm3_specular', None)))
    n = migrated('SEGA_MODEL3', 25.0, level=0.0)
    check('a Specular Level of 0 migrates to OFF', getattr(n, 'm3_specular', None) == 'OFF')
    check('the Model 2 / Model 3 specular labels are 12 characters or more',
          all(len(b) >= 12 for _a, b, _c in CON.M2_SPECULAR_ITEMS + CON.M3_SPECULAR_ITEMS))
    ann = N.__annotations__['pc_color_vertex']
    check('pc_color_vertex refreshes the sockets (an update callback)', ann.kw.get('update') is not None)


def test_generated_frame_signature_and_flat_axis():
    from .test_r253_generated_coords import move_ball, _job
    from ..core import mathx as M
    from ..gpu import shade as GSH
    st = base_settings(64, 48)
    sc0 = demo_scene(st, with_texture=False); j0, _ = _job(sc0, st)
    mw = np.eye(4, dtype=f32); mw[:3, 3] = (0.3, -0.3, 0.0)
    sc1 = demo_scene(st, with_texture=False); move_ball(sc1, mw); j1, _ = _job(sc1, st)
    k0, k1 = GSH._mesh_key(sc0.mesh), GSH._mesh_key(sc1.mesh)
    check('a translation keeps the mesh key (the stale-cache case)', k0 == k1)
    check('...and changes the plan signature (the baked frame is fingerprinted)', GSH._plan_sig(j0, k0) != GSH._plan_sig(j1, k1))
    sc2 = demo_scene(st, with_texture=False); move_ball(sc2, mw)
    sc2.objects[1].gen_bounds = (np.array([-3.3, -3.3, -3.3], f32), np.array([3.3, 3.3, 3.3], f32))
    j2, _ = _job(sc2, st)
    check('a Texture Space edit alone changes the plan signature', GSH._plan_sig(j1, k1) != GSH._plan_sig(j2, GSH._mesh_key(sc2.mesh)))
    # the flat floor, rotated and moved: its flat axis spans 1.0 and reads ~0
    sc3 = demo_scene(st, with_texture=False)
    a = 0.5; c, s_ = np.cos(a), np.sin(a)
    rot = np.eye(4, dtype=f32); rot[1, 1], rot[1, 2], rot[2, 1], rot[2, 2] = c, -s_, s_, c; rot[:3, 3] = (5.0, 5.0, 5.0)
    mesh = sc3.mesh
    vert_obj = np.zeros(mesh.verts.shape[0], np.int32); vert_obj[mesh.tris.reshape(-1)] = np.repeat(mesh.obj_index, 3)
    sel = vert_obj == 0
    V = mesh.verts.copy(); V[sel] = (V[sel] @ rot[:3, :3].T + rot[:3, 3]).astype(f32); mesh.verts = V
    sc3.objects[0].matrix_world = rot
    j3, _ = _job(sc3, st)
    lo, span = j3.object_generated_frame()
    check('a transformed flat object spans 1.0 on its flat axis', float(span[0][2]) == 1.0, str(span[0].tolist()))
    rng = np.random.RandomState(1); P0 = np.zeros((2000, 3), f32); P0[:, :2] = rng.uniform(-5, 5, (2000, 2))
    P = (P0 @ rot[:3, :3].T + rot[:3, 3]).astype(f32)
    g = (M.object_space_points(P, j3.object_matrices()[0]) - lo[0]) / span[0]
    check('and its Generated.z reads within 1e-5 of 0 (no float32 speckle)', float(np.abs(g[:, 2]).max()) < 1e-5, str(float(np.abs(g[:, 2]).max())))
    sc4 = demo_scene(st, with_texture=False); j4, _ = _job(sc4, st); _lo4, span4 = j4.object_generated_frame()
    check('an identity object keeps the 1e-6 clamp (the 1.91 bits)', abs(float(span4[0][2]) - 1e-6) < 1e-9)
    check('_gen_span: identity keeps max(extent, 1e-6), a transform gives a flat axis 1.0',
          bool(np.array_equal(R._gen_span(np.array([2.0, 0.0, 3.0], f32), True), np.array([2.0, 1e-6, 3.0], f32)))
          and bool(np.array_equal(R._gen_span(np.array([2.0, 0.0, 3.0], f32), False), np.array([2.0, 1.0, 3.0], f32))))


def test_field_hardening_f9_and_undo():
    """The field report (every redo panel greyed, undo / redo refused):
    Blender refuses both while a scene-owned job runs or the step is
    gone. The render job can no longer hang on the pipeline lock, the
    pristine-material watcher converts a material once, and a stale
    Set-Shader model no longer makes every Convert button raise."""
    import inspect
    from ..core import convert as CC
    from ..core.shading import MASTER_MODELS
    p = CC.plan('ShaderNodeBsdfPrincipled', {}, set(), '')
    check("plan() with the stale '' model chooses a master model (AUTO)", p['model'] in MASTER_MODELS, str(p['model']))
    p2 = CC.plan('ShaderNodeBsdfPrincipled', {}, set(), 'GX_LIGHT')
    check('plan() with a moved model (GX_LIGHT) reads as AUTO', p2['model'] in MASTER_MODELS, str(p2['model']))
    p3 = CC.plan('ShaderNodeBsdfPrincipled', {}, set(), 'PHONG')
    check('plan() with a master model keeps it', p3['model'] == 'PHONG')
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    esrc = open(os.path.join(root, 'engine.py'), encoding='utf-8').read()
    check('the F12 render acquires the GPU pipeline with a timeout (a render job that never ends greys every redo panel)',
          'PIPELINE.acquire(timeout=' in esrc and 'shades on the CPU' in esrc)
    csrc = open(os.path.join(root, 'convert.py'), encoding='utf-8').read()
    check('the pristine-material watcher converts each material once (an undone conversion stays undone)',
          '_HANDLER_CONVERTED' in csrc and csrc.count('_HANDLER_CONVERTED') >= 3)
    check("convert_material guards the node's model assignment", 'master.model = p[' in csrc and 'except (TypeError, ValueError)' in csrc.split('master.model = p[')[1][:200])
    usrc = open(os.path.join(root, 'ui.py'), encoding='utf-8').read()
    check("the panel never pushes an empty model", "(hal.convert_model or 'AUTO')" in usrc)


def test_field_worker_pool_waits_are_bounded():
    """The field pair again (every F9 panel greyed, undo refused): an F12
    job parked forever in the worker pool. The child's stderr was a pipe
    nobody read while it lived, so a chatty worker eventually blocked in
    its own print and the parent's recv() -- a bare pipe read -- waited
    for the rest of the session. Now: stderr is the worker's log file,
    replies come through a reader thread, and recv() waits in slices
    that answer a cancel (test_break) or a deadline."""
    import queue
    import threading
    import time as _t
    from ..core import parallel as P
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    src = open(os.path.join(root, 'core', 'parallel.py'), encoding='utf-8').read()
    check("a worker's stderr is never a pipe the parent does not read",
          'stderr=subprocess.PIPE' not in src and 'stderr=err' in src)
    check('replies arrive through a reader thread', '_read_loop' in src and 'self._inbox' in src)
    check('the handshake and every band have a deadline',
          'deadline=HANDSHAKE_DEADLINE' in src and src.count('deadline=BAND_DEADLINE') >= 2)
    check('WorkerCancelled is its own outcome, never an in-process retry',
          'except WorkerCancelled:\n        shutdown()\n        raise' in src)
    esrc = open(os.path.join(root, 'engine.py'), encoding='utf-8').read()
    check('the engine hands the pool its test_break',
          'cancel=self.test_break)' in esrc and '_par.WorkerCancelled' in esrc)

    # recv() on a worker that never answers: the cancel ends it within a slice
    w = P.Worker.__new__(P.Worker)
    w._inbox = queue.Queue()
    w.proc = type('P', (), {'poll': staticmethod(lambda: None), 'returncode': None})()
    w.lock = threading.Lock()
    t0 = _t.monotonic()
    try:
        w.recv(cancel=lambda: True)
        got = 'returned'
    except P.WorkerCancelled:
        got = 'cancelled'
    except Exception as exc:                                    # noqa: BLE001
        got = f'{type(exc).__name__}'
    dt = _t.monotonic() - t0
    check('a cancel ends a silent recv() with WorkerCancelled within a slice',
          got == 'cancelled' and dt < 5.0 * P.SLICE + 1.0, f'{got} after {dt:.2f}s')
    t0 = _t.monotonic()
    try:
        w.recv(deadline=0.3)
        got = 'returned'
    except IOError as exc:
        got = 'IOError: ' + str(exc)
    dt = _t.monotonic() - t0
    check('a deadline ends a silent recv() with IOError naming the wait',
          got.startswith('IOError') and 'did not answer' in got and dt < 3.0, f'{got} after {dt:.2f}s')
    import pickle
    w._inbox.put(('msg', pickle.dumps(('ok',))))
    check('a queued reply is returned as the message', w.recv(cancel=lambda: True) == ('ok',))
    w._inbox.put(('eof', 'worker closed the connection'))
    try:
        w.recv()
        got = 'returned'
    except IOError as exc:
        got = str(exc)
    check('a closed connection is an IOError, not a hang', 'closed' in got, got)
    dead = type('P', (), {'poll': staticmethod(lambda: 1), 'returncode': 1})()
    w.proc = dead
    t0 = _t.monotonic()
    try:
        w.recv()
        got = 'returned'
    except IOError as exc:
        got = str(exc)
    check('an exited worker is noticed without a cancel or deadline',
          'exited' in got and _t.monotonic() - t0 < 3.0, got)


def test_field_marshal_pickup_wait_is_cancellable():
    """A GPU burst waiting for the main thread gives up the moment the
    job thread's cancel hook answers true -- Blender's undo and the next
    render JOIN the job thread on the main thread, so the pump could
    not fire and every outstanding burst sat out the whole TIMEOUT."""
    import threading
    import time as _t
    from . import fakebpy
    bpy = fakebpy.install()
    from ..gpu import marshal as M
    while not M._JOBS.empty():
        M._JOBS.get_nowait()
    M._STATE['enabled'] = 0
    M._STATE['timer'] = False
    check('Cancelled passes the stages\' Exception handlers (a BaseException)',
          issubclass(M.Cancelled, BaseException) and not issubclass(M.Cancelled, Exception))
    res = {}

    def worker():
        prev = M.set_cancel(lambda: True)
        try:
            t0 = _t.monotonic()
            try:
                M.run_on_main(lambda: res.setdefault('ran', True), timeout=6.0)
                res['r'] = 'returned'
            except M.Cancelled:
                res['r'] = 'cancelled'
            except BaseException as exc:                        # noqa: BLE001
                res['r'] = type(exc).__name__
            res['dt'] = _t.monotonic() - t0
        finally:
            M.set_cancel(prev)
            res['hook_after'] = M._CANCEL.get(threading.get_ident())

    t = threading.Thread(target=worker)
    t.start()
    t.join(10.0)                     # the main loop never pumps: a join
    check('a cancelled pickup wait raises Cancelled long before the timeout',
          res.get('r') == 'cancelled' and res.get('dt', 99) < 2.0, str(res))
    check('the burst never ran and the hook is restored', 'ran' not in res and res.get('hook_after') is None, str(res))
    res.clear()

    def worker2():
        prev = M.set_cancel(lambda: False)
        try:
            try:
                M.run_on_main(lambda: 1, timeout=0.3)
                res['r'] = 'returned'
            except M.MarshalTimeout:
                res['r'] = 'timeout'
            except BaseException as exc:                        # noqa: BLE001
                res['r'] = type(exc).__name__
        finally:
            M.set_cancel(prev)

    t = threading.Thread(target=worker2)
    t.start()
    t.join(10.0)
    check('a hook answering False leaves the timeout to do its work', res.get('r') == 'timeout', str(res))
    while not M._JOBS.empty():
        M._JOBS.get_nowait()
    # the pump timer flag is checked against Blender, not trusted
    M._STATE['timer'] = True
    bpy.app.timers.unregister(M._pump)
    M._ensure_timer()
    check('_ensure_timer re-registers a pump a file load dropped',
          bpy.app.timers.is_registered(M._pump), str(M._STATE))
    bpy.app.timers.unregister(M._pump)
    M._STATE['timer'] = False
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    esrc = open(os.path.join(root, 'engine.py'), encoding='utf-8').read()
    check('render() installs test_break as the job thread\'s cancel hook and restores it',
          '_marshal.set_cancel(self.test_break)' in esrc and '_marshal.set_cancel(_prev_cancel)' in esrc
          and 'except _marshal.Cancelled:' in esrc)
    msrc = open(os.path.join(root, 'gpu', 'marshal.py'), encoding='utf-8').read()
    check('the pump is registered persistent (a file load no longer drops it)',
          'persistent=True' in msrc)
    check('the pump retires and re-registers under one lock (no orphaned burst)',
          'with _STATE_LOCK:\n        if not _STATE[\'enabled\'] and _JOBS.empty():' in msrc
          and 'def _ensure_timer():\n    with _STATE_LOCK:' in msrc)
    check('the F12 pipeline wait answers test_break in slices',
          'PIPELINE.acquire(timeout=0.25)' in esrc and 'if self.test_break():\n                    return' in esrc)
    check('a thumbnail never touches the driver or the pipeline lock',
          'if preview and gpu_dev:' in esrc and "settings.render_device = 'CPU'\n            gpu_dev = False" in esrc)
    check('every GPU render marshals (the stale bl_use_gpu_context read is gone)',
          'marshalled = gpu_dev and not background' in esrc and 'granted = ' not in esrc)
    check('view_update pins through the transition table, not per update',
          '_pin_display_handler(depsgraph.scene, depsgraph)' in esrc)


def test_field_display_pin_on_transitions_only():
    """The depsgraph handler pins Color Management when a scene's engine
    BECOMES Halcyon and after a file load, never on every update: the
    every-update write landed after each operator's undo push and
    re-applied an undone edit on the next pass."""
    from . import fakebpy
    fakebpy.install()
    import importlib
    import types
    ENG = importlib.import_module('halcyon.engine')
    pinned = []
    orig = ENG._pin_display
    ENG._pin_display = lambda sc: pinned.append(sc.name)
    import sys
    bpy = sys.modules['bpy']
    had = getattr(bpy.data, 'scenes', None)

    def scene(name, engine, uid):
        return types.SimpleNamespace(name=name, session_uid=uid,
                                     render=types.SimpleNamespace(engine=engine))
    a = scene('A', 'HALCYON_RENDER', 11)
    b = scene('B', 'BLENDER_EEVEE', 12)
    try:
        bpy.data.scenes = [a, b]
        ENG._PIN_SEEN.clear()
        ENG._pin_display_handler(a, None)
        ENG._pin_display_handler(a, None)
        ENG._pin_display_handler(a, None)
        check('three depsgraph updates pin the Halcyon scene once', pinned == ['A'], str(pinned))
        b.render.engine = 'HALCYON_RENDER'
        ENG._pin_display_handler(b, None)
        check('a scene switching to Halcyon is pinned on that update', pinned == ['A', 'B'], str(pinned))
        ENG._pin_display_handler(b, None)
        check('and not again on the next', pinned == ['A', 'B'], str(pinned))
        ENG._pin_display_handler('/tmp/x.blend')
        check('a file load pins every Halcyon scene again', pinned == ['A', 'B', 'A', 'B'], str(pinned))
        a.render.engine = 'BLENDER_EEVEE'
        ENG._pin_display_handler(a, None)
        a.render.engine = 'HALCYON_RENDER'
        ENG._pin_display_handler(a, None)
        check('leaving and coming back to Halcyon pins again', pinned[-1] == 'A' and len(pinned) == 5, str(pinned))
    finally:
        ENG._pin_display = orig
        ENG._PIN_SEEN.clear()
        if had is not None:
            bpy.data.scenes = had
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    usrc = open(os.path.join(root, 'ui.py'), encoding='utf-8').read()
    check('the Debug panel names a running Blender job (the F9 / undo refusal has a visible cause)',
          'def _running_jobs' in usrc and 'is_job_running' in usrc and 'Blender job running: ' in usrc)


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
    print('all R253 review-fix checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
