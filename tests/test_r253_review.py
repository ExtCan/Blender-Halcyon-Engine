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
(11) The Coded Shader's language / source update callbacks only flag the
    socket rebuild and a one-shot timer does it -- after Blender has
    pushed the edit's undo step, so the step holds needs_rebuild with the
    OLD sockets and memfile undo brings it back without the callback that
    arms the timer. Persistent undo_post / redo_post / load_post handlers
    (`_rearm_code_rebuilds`) schedule the rebuild on every landing; the
    Compile operator clears the flag in its own step.
(12) The Coded Shader's Auto Compile checkbox was drawn in the sidebar and
    read by nothing: unticked, the node still rebuilt on every road. It is
    retired, not wired -- no rebuild fires while typing (only a Language
    or Text change flags one) and the exporter compiles the live source
    whatever the sockets say, so "off" could only strand the OLD sockets
    under the NEW program. The Inline Source tooltip called the property
    the NAME of a Text datablock; it is the source itself.
(13) A Coded Shader with a Text assigned ran TWO shaders: the exporter
    compiled the Text for the program (the CPU's truth) and copied the
    raw inline `source_text` -- the starter shader `init` leaves there --
    into the graph, which is what the GPU emitter inlines. The deferred
    pass qualified and drew the starter. The editor and the exporter also
    disagreed about an EMPTY assigned Text. One rule now
    (`shaders/compiler.node_source` / `compile_node`): the assigned Text,
    whatever it holds, else the inline source -- compiled by the editor
    and the exporter through one call, shipped by the exporter as the
    graph's source, and the only source the emitter reads.
(14) Compiled Coded Shader programs were filed under the node's NAME in
    one dict handed down the node-group recursion, and names are unique
    per tree only: the material's "Coded Shader" and a group's "Coded
    Shader" (the default name, twice) left ONE program, and one of the two
    nodes ran the other's shader under its own sockets. The graph id is
    the node's PATH now (`export.node_id`: 'Group/Coded Shader'), written
    in one place for the node keys, the links and the programs; the Bump
    pre-pass fields, filed by the same id, stop crossing trees with it.
    The GPU emitter's mangling tag was the id with its punctuation
    flattened, which is many-to-one ('Coded Shader' / 'Coded_Shader'):
    `gpu/emit._code_tag` hands out one tag per node.
"""
import io
import os
import sys
import traceback
import types
from contextlib import contextmanager, redirect_stdout

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import demo_scene
from ..core import console as CON
from ..core import render as R
from ..core.scene import material_see_through
from .test_r252_console import _scene as console_scene, console_graph, _render as console_render
from .test_render import _sk
from .test_r251_material import _sim_vs_cpu, _job_for

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
    # every Halcyon handler survives a file load (Blender drops the rest)
    import importlib as _il
    for modname, fname in (('halcyon.convert', '_halcyon_default_material'),
                           ('halcyon.nodes.shader_nodes', '_migrate_master_sockets'),
                           ('halcyon.engine', '_pin_display_handler')):
        fn = getattr(_il.import_module(modname), fname)
        check(f'{fname} is a persistent handler (it survives File > Open)',
              hasattr(fn, '_bpy_persistent'), str(list(vars(fn).keys())))


_CODE_A = """\
uniform float gain = 1.0;
uniform vec3  tint = vec3(1.0, 0.5, 0.25);

out vec4 Color;

void main() {
    Color = vec4(tint * gain, 1.0);
}
"""

_CODE_B = """\
uniform float  gain    = 1.0;
uniform float3 edge    = float3(0.2, 0.4, 0.8);
uniform float  stripes = 8.0;

in float3 vPosition;

out float4 Color;

void main() {
    float s = step(0.5, frac(vPosition.z * stripes));
    Color = float4(edge * gain * lerp(0.5, 1.0, s), 1.0);
}
"""


class _CodeSock:
    """One node socket: exactly what rebuild_sockets reads and writes."""
    _DEFAULTS = {'NodeSocketFloat': 0.0, 'NodeSocketInt': 0, 'NodeSocketBool': False,
                 'NodeSocketVector': (0.0, 0.0, 0.0), 'NodeSocketColor': (0.0, 0.0, 0.0, 1.0)}
    #: bl_idname -> NodeSocket.type, the two things export.serialize_tree
    #: reads off a socket beyond its name and its links
    _TYPES = {'NodeSocketFloat': 'VALUE', 'NodeSocketInt': 'INT', 'NodeSocketBool': 'BOOLEAN',
              'NodeSocketVector': 'VECTOR', 'NodeSocketColor': 'RGBA', 'NodeSocketShader': 'SHADER'}

    def __init__(self, node, stype, name):
        self.node, self.bl_idname, self.name = node, stype, name
        self.links = []
        self.halcyon_uniform = self.halcyon_key = ''
        self.halcyon_is_image = False
        if stype in self._DEFAULTS:
            self.default_value = self._DEFAULTS[stype]

    @property
    def type(self):
        return self._TYPES[self.bl_idname]

    @property
    def is_linked(self):
        return bool(self.links)


class _CodeSockets(list):
    """node.inputs / node.outputs: new / get / clear and by-name indexing;
    clearing takes the sockets' links with it, as Blender does."""

    def __init__(self, node):
        super().__init__()
        self.node = node

    def new(self, stype, name):
        sock = _CodeSock(self.node, stype, name)
        self.append(sock)
        return sock

    def get(self, name, default=None):
        return next((s for s in self if s.name == name), default)

    def __getitem__(self, key):
        if isinstance(key, str):
            return next(s for s in self if s.name == key)
        return super().__getitem__(key)

    def clear(self):
        for sock in self:
            for link in list(sock.links):
                self.node.id_data.links.remove(link)
        super().clear()


class _CodeLinks(list):
    def new(self, from_socket, to_socket):
        link = types.SimpleNamespace(from_node=from_socket.node, from_socket=from_socket,
                                     to_node=to_socket.node, to_socket=to_socket)
        self.append(link)
        from_socket.links.append(link)
        to_socket.links.append(link)
        return link

    def remove(self, link):
        for sock in (link.from_socket, link.to_socket):
            sock.links[:] = [l for l in sock.links if l is not link]
        self[:] = [l for l in self if l is not link]


class _CodeNodes(list):
    def get(self, name, default=None):
        return next((n for n in self if n.name == name), default)


def _code_plain(tree, name, idname, ins=(), outs=(), **attrs):
    """A stand-in node of any other type, appended to `tree`: its sockets
    as (socket bl_idname, name) pairs, its properties as attributes."""
    node = types.SimpleNamespace(name=name, bl_idname=idname, id_data=tree, **attrs)
    node.inputs, node.outputs = _CodeSockets(node), _CodeSockets(node)
    for stype, sn in ins:
        node.inputs.new(stype, sn)
    for stype, sn in outs:
        node.outputs.new(stype, sn)
    tree.nodes.append(node)
    return node


def _code_coded(SN, tree, name, source, cls=None):
    """The REAL Coded Shader class (or `cls`, a subclass of it) run against
    stand-in sockets: appended to `tree` and built from `source`, a Text,
    exactly as `init` builds a new node."""
    code = (cls or SN.HALCYON_CodeNode)()
    code.name, code.id_data = name, tree
    code.language, code.source, code.source_text = 'GLSL', source, ''
    code.as_surface, code.needs_rebuild = False, False
    code.error = code.warn = ''
    code.inputs, code.outputs = _CodeSockets(code), _CodeSockets(code)
    tree.nodes.append(code)
    code.init(None)
    return code


def _code_tree(SN, source, name='Coded', cls=None):
    """A material whose tree is Value -> Coded Shader -> Material Output,
    the coded node being the REAL class (or `cls`, a subclass of it) run
    against stand-in sockets."""
    tree = types.SimpleNamespace(name=name, nodes=_CodeNodes(), links=_CodeLinks())
    value = _code_plain(tree, 'Value', 'ShaderNodeValue', outs=(('NodeSocketFloat', 'Value'),))
    out = _code_plain(tree, 'Material Output', 'ShaderNodeOutputMaterial', ins=(('NodeSocketShader', 'Surface'),))
    code = _code_coded(SN, tree, 'Coded Shader', source, cls)
    tree.links.new(value.outputs['Value'], code.inputs['Gain'])
    tree.links.new(code.outputs['Color'], out.inputs['Surface'])
    return types.SimpleNamespace(name=name, use_nodes=True, node_tree=tree)


class _MemfileUndo:
    """Blender's memfile undo as far as Python can see it: a push encodes
    the datablocks as they stand at that instant; landing on a step swaps
    in fresh copies of them (new pointers) and fires NOT ONE update
    callback -- undo_post / redo_post are all an add-on hears."""

    def __init__(self, bpy):
        self.bpy, self.steps, self.at = bpy, [], -1

    def push(self):
        import copy
        del self.steps[self.at + 1:]
        self.steps.append(copy.deepcopy(list(self.bpy.data.materials)))
        self.at = len(self.steps) - 1

    def restore(self, index):
        import copy
        self.at = index
        self.bpy.data.materials[:] = copy.deepcopy(self.steps[index])


_CODE_EVENTS = ('undo_post', 'redo_post', 'load_post')


@contextmanager
def _code_bench(bpy, SN):
    """An interactive session with the node module registered and nothing
    else in it: no material, no timer, no handler on the three re-arm
    events. Yields (handlers, timers). The handler lists are the stub's
    own, whatever an earlier test left in their place (test_legacy swaps
    in a one-list namespace and keeps it there). Everything touched in the
    shared stub goes back as it was, and the module is unregistered on the
    way out unless the test already did that itself (the re-arm handler on
    the lists is what tells)."""
    from . import fakebpy
    prior_handlers = bpy.app.handlers
    bpy.app.handlers = fakebpy._handlers
    handlers, timers = bpy.app.handlers, bpy.app.timers
    saved_hooks = {k: list(v) for k, v in vars(handlers).items() if isinstance(v, list)}
    saved_timers = (list(timers.fns), set(timers._persistent))
    saved_mats, saved_bg, saved_q = bpy.data.materials, bpy.app.background, SN._REBUILD_QUEUED[0]
    had_groups = getattr(bpy.data, 'node_groups', None)
    for name in _CODE_EVENTS:
        getattr(handlers, name)[:] = []
    timers.fns[:] = []
    bpy.data.materials = []
    bpy.app.background = False             # an interactive session: timers fire
    SN._REBUILD_QUEUED[0] = False
    try:
        SN.register()
        yield handlers, timers
    finally:
        if any(SN._rearm_code_rebuilds in getattr(handlers, e) for e in _CODE_EVENTS):
            SN.unregister()
        for k, v in saved_hooks.items():
            getattr(handlers, k)[:] = v
        bpy.app.handlers = prior_handlers
        timers.fns[:] = saved_timers[0]
        timers._persistent.clear()
        timers._persistent.update(saved_timers[1])
        bpy.data.materials, bpy.app.background = saved_mats, saved_bg
        SN._REBUILD_QUEUED[0] = saved_q
        if had_groups is None:
            if hasattr(bpy.data, 'node_groups'):
                del bpy.data.node_groups
        else:
            bpy.data.node_groups = had_groups


class _CodeLayout:
    """A UILayout that only writes down what a node's draw asked of it."""

    def __init__(self, log=None):
        self.log = [] if log is None else log
        self.alert = False

    def prop(self, _data, name, **_kw):
        self.log.append(('prop', name))

    def label(self, text='', **_kw):
        self.log.append(('label', text))

    def operator(self, idname, **_kw):
        self.log.append(('operator', idname))
        return types.SimpleNamespace()

    def separator(self, **_kw):
        self.log.append(('separator', ''))

    def row(self, **_kw):
        return _CodeLayout(self.log)

    column = box = row


def test_code_node_props_do_what_their_tooltips_say():
    """Two Coded Shader properties whose tooltips were not true. Auto
    Compile ("off waits for the Build button") was drawn in the sidebar
    and read by nothing: unticked, the node still rebuilt from the update
    callbacks' timer, from the undo / redo / load re-arm and at every
    export. It is RETIRED, not wired: no rebuild fires while typing (only
    a Language or Text change flags one) and the exporter compiles the
    live source whatever the sockets say, so "off" could only strand the
    OLD sockets under a render already running the NEW program. A file
    saved with the box unticked brings the value back as an ID property
    the class no longer declares, and no road may ask for it. Inline
    Source was described as the NAME of a Text datablock; it is the
    source itself, the one that compiles while no Text is assigned."""
    import importlib
    from . import fakebpy
    bpy = fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    EX = importlib.import_module('halcyon.export')
    Node = SN.HALCYON_CodeNode
    scene = types.SimpleNamespace(name='Scene')
    text_a = types.SimpleNamespace(name='shader_a', as_string=lambda: _CODE_A)
    text_b = types.SimpleNamespace(name='shader_b', as_string=lambda: _CODE_B)
    asked = []

    class _SavedUnticked(Node):
        """The node of a file saved with the box unticked, at its most
        reachable: the stored value answers as a plain attribute (Blender
        itself keeps an undeclared ID property out of attribute access)
        and every look at it is written down."""

        @property
        def auto_compile(self):
            asked.append(sys._getframe(1).f_code.co_name)
            return False

    def names(socks):
        return [s.name for s in socks]

    def node():
        return bpy.data.materials[0].node_tree.nodes.get('Coded Shader')

    def queued():
        return timers.is_registered(SN._run_rebuilds)

    def edit(n):
        """The Text and the Language change, as the two widgets make them."""
        n.source = text_b
        Node._source_changed(n, None)
        n.language = 'HLSL'
        Node._lang_changed(n, None)

    def whole(n):
        return (not n.needs_rebuild and names(n.inputs) == ['Gain', 'Edge', 'Stripes']
                and [(l.from_node.name, l.from_socket.name) for l in n.inputs['Gain'].links] == [('Value', 'Value')]
                and [(l.to_node.name, l.to_socket.name) for l in n.outputs['Color'].links]
                == [('Material Output', 'Surface')])

    def stale(n):
        return n.needs_rebuild and n.language == 'HLSL' and names(n.inputs) == ['Gain', 'Tint']

    def state(n):
        return f'flag={n.needs_rebuild} lang={n.language} in={names(n.inputs)} queued={queued()} asked={asked}'

    def uniforms(prog):
        return [u['name'] for u in prog.uniform_schema()] if prog is not None else None

    # ---- the declaration is gone from every table that carried it
    live = set(Node.__annotations__)
    check('the Coded Shader declares no Auto Compile property', 'auto_compile' not in live)
    tipped = {k[1] for k in SN.PROP_TIPS if k[0] == 'HALCYON_CodeNode'}
    check('every Coded Shader tooltip in PROP_TIPS names a live property (none outlives its property)',
          tipped and tipped <= live, f'tips {sorted(tipped)} props {sorted(live)}')
    check('...and every property the node still declares keeps a tooltip at the 40-character bar',
          all(len(str(Node.__annotations__[k].kw.get('description') or '')) >= 40 for k in live),
          str({k: len(str(Node.__annotations__[k].kw.get('description') or '')) for k in live}))
    check('the exporter copies language and as_surface verbatim (the source ships RESOLVED: finding 13)',
          EX.NODE_PROPS['HALCYON_CodeNode'] == ('language', 'as_surface'))
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    here, named = os.path.abspath(__file__), []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for fname in files:
            path = os.path.join(folder, fname)
            if fname.endswith('.py') and os.path.abspath(path) != here:
                if 'auto_compile' in open(path, encoding='utf-8').read():
                    named.append(os.path.relpath(path, root))
    check('no other Python file names the retired property (declaration, draw line, tooltip, census reason)',
          not named, ', '.join(named))

    with _code_bench(bpy, SN) as (_handlers, timers):
        # ---- the node draws no box for it, in the node or in the sidebar
        bpy.data.materials.append(_code_tree(SN, text_a, cls=_SavedUnticked))
        n = node()
        side, face = _CodeLayout(), _CodeLayout()
        n.draw_buttons_ext(None, side)
        n.draw_buttons(None, face)
        check('the sidebar draws no property, only the list of available inputs',
              [name for kind, name in side.log if kind == 'prop'] == []
              and side.log[0] == ('label', 'Available inputs:') and ('label', '  iTime') in side.log
              and all(kind == 'label' for kind, _ in side.log), str(side.log))
        check('the node body draws Language, Text and Output as Surface, and the Compile button',
              [name for kind, name in face.log if kind == 'prop'] == ['language', 'source', 'as_surface']
              and ('operator', 'halcyon.compile_shader') in face.log, str(face.log))

        # ---- a file saved with the box unticked: every road still rebuilds
        check('the node of the saved file starts whole on the GLSL source',
              not n.needs_rebuild and names(n.inputs) == ['Gain', 'Tint'], state(n))
        undo = _MemfileUndo(bpy)
        undo.push()                                              # step 0
        edit(n)
        check('unticked: the update callbacks flag and queue the ONE timer, as for any node',
              stale(n) and timers.fns.count(SN._run_rebuilds) == 1, state(n))
        undo.push()                                              # step 1: flagged, OLD sockets
        timers.pump()
        check('unticked: the timer rebuilds the sockets with no Compile press', whole(n) and not queued(), state(n))
        undo.restore(1)
        fakebpy.fire('undo_post', scene)
        check('unticked: an undo landing on the flagged step schedules the rebuild', queued() and stale(node()),
              state(node()))
        timers.pump()
        check('...and it runs', whole(node()) and not queued(), state(node()))
        undo.restore(1)
        fakebpy.file_load('/tmp/unticked.blend')
        check('unticked: File > Open of the file saved flagged schedules the rebuild', queued() and stale(node()),
              state(node()))
        timers.pump()
        check('...and the loaded node ends whole', whole(node()) and not queued(), state(node()))
        bpy.app.background = True
        undo.restore(1)
        fakebpy.file_load('/tmp/unticked.blend')
        check('unticked: a background load (a render farm, nobody to press Compile) rebuilds at once',
              whole(node()) and not queued(), state(node()))
        bpy.app.background = False

        # ---- the exporter was never the toggle's to hold: it compiles the live source
        undo.restore(1)
        n = node()
        images, programs, warnings = {}, {}, []
        props = EX._node_props(n, images, programs, warnings)
        check('the export of a flagged node carries the NEW program (the picture never waited for a build)',
              uniforms(programs.get(n.name)) == ['gain', 'edge', 'stripes'] and stale(n) and not warnings,
              f'{uniforms(programs.get(n.name))} {warnings} {state(n)}')
        check('...and exactly the three shipped keys', sorted(props) == ['as_surface', 'language', 'source_text'],
              str(sorted(props)))
        op = SN.HALCYON_OT_compile_shader()
        op.node, op.report = n.name, (lambda *_a: None)
        ctx = types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=n.id_data), active_node=None)
        got = op.execute(ctx)
        check('the Compile button still rebuilds on demand and clears the flag', got == {'FINISHED'} and whole(n),
              f'{got} {state(n)}')
        check('through all of it NOT ONE road asked for the saved value', not asked, str(asked))

        # ---- Inline Source is the source, not a datablock's name
        tip = SN.PROP_TIPS[('HALCYON_CodeNode', 'source_text')]
        check('the Inline Source tooltip says it is the source that compiles while no Text is assigned',
              len(tip) >= 40 and 'name of' not in tip.lower() and 'no text datablock' in tip.lower()
              and Node.__annotations__['source_text'].kw.get('description') == tip, tip)
        inline = node()
        inline.source, inline.source_text, inline.language = None, _CODE_A, 'GLSL'
        check('with no Text assigned the node compiles source_text itself, in the editor and at export',
              inline.get_source() == _CODE_A and uniforms(inline.compile_source()) == ['gain', 'tint']
              and uniforms(EX._compile_node(inline)[0]) == ['gain', 'tint'])
        inline.source, inline.language = text_b, 'HLSL'
        check('...and an assigned Text takes its place on both roads',
              inline.get_source() == _CODE_B and uniforms(inline.compile_source()) == ['gain', 'edge', 'stripes']
              and uniforms(EX._compile_node(inline)[0]) == ['gain', 'edge', 'stripes'])


#: the master's sockets at tests/scenebuild.add_coded_floor's values
_MASTER_SOCKETS = (
    ('NodeSocketColor', 'Diffuse Color', (0.55, 0.55, 0.6, 1.0)), ('NodeSocketColor', 'Vertex Color', (1.0, 1.0, 1.0, 1.0)),
    ('NodeSocketFloat', 'Vertex Color Mix', 0.0), ('NodeSocketFloat', 'Diffuse Level', 1.0),
    ('NodeSocketColor', 'Specular Color', (1.0, 1.0, 1.0, 1.0)), ('NodeSocketFloat', 'Specular Level', 0.0),
    ('NodeSocketFloat', 'Glossiness', 12.0), ('NodeSocketFloat', 'Roughness', 0.5),
    ('NodeSocketFloat', 'Ambient', 0.6), ('NodeSocketColor', 'Self-Illumination', (0.0, 0.0, 0.0, 1.0)),
    ('NodeSocketFloat', 'Opacity', 1.0), ('NodeSocketFloat', 'IOR', 1.45), ('NodeSocketFloat', 'Anisotropy', 0.0),
    ('NodeSocketFloat', 'Anisotropic Rotation', 0.0), ('NodeSocketFloat', 'Metalness', 0.0),
    ('NodeSocketFloat', 'Soften', 0.0), ('NodeSocketFloat', 'Reflection', 0.0), ('NodeSocketFloat', 'Translucency', 0.0),
    ('NodeSocketFloat', 'Toon Size', 0.5), ('NodeSocketFloat', 'Toon Smooth', 0.05))


def _code_master_tree(SN, source):
    """_code_tree's material with a master in it: Value -> Coded Shader ->
    Halcyon (Diffuse Color) -> Material Output, the wiring test_render's
    coded GPU twins hold. The Value node reads 0.7, so the linked Gain is
    not the shader's declared default."""
    mat = _code_tree(SN, source)
    tree = mat.node_tree
    code, out = tree.nodes.get('Coded Shader'), tree.nodes.get('Material Output')
    tree.nodes.get('Value').outputs['Value'].default_value = 0.7
    for link in list(code.outputs['Color'].links):
        tree.links.remove(link)
    hal = types.SimpleNamespace(name='Halcyon', bl_idname='HALCYON_ShaderNode', id_data=tree, model='LAMBERT',
                                toon_steps=2)
    hal.inputs, hal.outputs = _CodeSockets(hal), _CodeSockets(hal)
    for stype, name, value in _MASTER_SOCKETS:
        hal.inputs.new(stype, name).default_value = value
    hal.outputs.new('NodeSocketShader', 'Surface')
    tree.nodes.append(hal)
    tree.links.new(code.outputs['Color'], hal.inputs['Diffuse Color'])
    tree.links.new(hal.outputs['Surface'], out.inputs['Surface'])
    return mat


def _code_floor(graph, programs, size=(128, 96)):
    """The demo scene with its floor wearing an exported graph: (scene,
    settings)."""
    from ..core.scene import Material
    st = base_settings(size[0], size[1], shadows=True)
    st.color_depth, st.dither, st.output_scale = '24', 'NONE', 'NONE'
    sc = demo_scene(st, with_texture=False)
    mat = Material(name='Coded', index=0, graph=graph)
    mat.programs = programs
    sc.materials[0] = mat
    return sc, st


def _code_cpu(graph, programs, size=(128, 96)):
    """The CPU's frame of that floor, and nothing else."""
    sc, st = _code_floor(graph, programs, size)
    with redirect_stdout(io.StringIO()):
        return np.asarray(R.render(sc, st))


def _code_twin(graph, programs, keep_plan=False):
    """The demo floor wearing an exported graph: (d_max, n_bad, gpu, cpu,
    why) of the CPU frame against the GLSL simulator's frame of the
    deferred plan, over the covered pixels (_sim_vs_cpu's bar). keep_plan
    leaves the plan cache as the last frame left it: an edit has to re-plan
    by its SIGNATURE, not by a cleared cache."""
    from ..gpu import shade as GSH
    sc, st = _code_floor(graph, programs)
    with redirect_stdout(io.StringIO()):
        cpu = np.asarray(R.render(sc, st))
        g, job = _job_for(sc, st)
        if not keep_plan:
            GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        if passes is None:
            return None, None, None, cpu, str(why)
        img, hit = GSH.simulate(job, g, passes, atlases)
    if img is None:
        return None, None, None, cpu, str(hit)
    cov = g.tri >= 0
    d = np.abs(img[cov] - cpu[cov][:, :3])
    return float(d.max()), int((d.max(axis=1) > 1e-2).sum()), img, cpu, None


def test_code_node_one_source_on_every_road():
    """A Coded Shader with a Text assigned ran two shaders. The exporter
    compiled the Text into the program (what the CPU runs on either
    device) and copied the node's raw inline `source_text` into the
    graph -- the starter shader `init` writes there and nothing clears --
    and the graph's copy is what the GPU emitter inlines. The deferred
    pass QUALIFIED and drew the starter (it refused only when a socket's
    type clashed with a like-named starter uniform). The editor and the
    exporter also read an EMPTY assigned Text differently: the editor
    compiled nothing, the exporter reached past it for the inline source
    ('' did; one blank line did not). One rule now,
    shaders/compiler.node_source: the assigned Text, whatever it holds,
    else the inline source. The editor and the exporter compile through
    one call (compile_node), the exporter ships that call's source as the
    graph's, and the emitter reads nothing else -- and refuses by name a
    graph source that is not its program's own."""
    import ast
    import copy
    import importlib
    import inspect
    import textwrap
    from . import fakebpy
    fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    EX = importlib.import_module('halcyon.export')
    SC = importlib.import_module('halcyon.shaders.compiler')
    from ..gpu import emit as GE
    from ..gpu import shade as GSH
    Node = SN.HALCYON_CodeNode
    STARTER = SC.DEFAULT_GLSL
    # the Text's shader: _CODE_A in a cold tint, so its picture is far from
    # the (orange) starter's and the twin below cannot pass on the wrong one
    TEXT = _CODE_A.replace('vec3(1.0, 0.5, 0.25)', 'vec3(0.1, 0.6, 0.9)')
    held = {'src': TEXT}
    text = types.SimpleNamespace(name='shader_a', as_string=lambda: held['src'])

    def names(socks):
        return [s.name for s in socks]

    def exported(n):
        """The node's whole tree through the real export."""
        images, programs, warnings = {}, {}, []
        graph = EX.serialize_tree(n.id_data, images, programs, warnings)
        return graph, programs, warnings

    def roads(n):
        """What each reader holds for the node: the editor's source, the
        source the exported program was compiled from (None: no program),
        the source the graph carries to the GPU emitter -- and the
        export's warnings."""
        graph, programs, warnings = exported(n)
        return (n.get_source(), getattr(programs.get(n.name), 'source', None),
                graph['nodes'][n.name]['props'].get('source_text'), warnings)

    def short(v):
        return repr(v)[:48]

    # ---- the reported state: the node's own "New Shader Text", then an edit
    n = _code_master_tree(SN, text).node_tree.nodes.get('Coded Shader')
    check('the reported state: a Text assigned, and the starter shader still in source_text (init wrote it)',
          n.source is text and n.source_text == STARTER and TEXT not in (STARTER, _CODE_A)
          and names(n.inputs) == ['Gain', 'Tint'],
          f'{names(n.inputs)} {short(n.source_text)}')
    ed, pr, gr, warns = roads(n)
    check('the editor, the exported program and the graph hold ONE source: the Text',
          ed == pr == gr == TEXT and not warns, f'editor {short(ed)} program {short(pr)} graph {short(gr)} {warns}')
    check('...and the inline property is not what ships while a Text is assigned', gr != n.source_text, short(gr))

    # ---- the picture: the frame pass draws the Text, as the CPU does
    graph_a, programs_a, _w = exported(n)
    d, bad, _gpu, cpu_a, why = _code_twin(graph_a, programs_a)
    check('a Text-sourced coded shader still qualifies for the deferred pass', why is None, str(why))
    check('...and its GPU frame IS the CPU frame (the pass drew the starter shader here)',
          d is not None and bad == 0 and d < 6e-3, f'max {d} bad {bad}')
    # ---- an edit in the Text Editor between two frames, the plan cache kept
    # (it holds the plan of the frame above: the Text before the edit)
    plans = len(GSH._PLAN_CACHE)
    held['src'] = TEXT.replace('tint * gain', 'tint.bgr * gain')
    check('the edit is a new source with the same sockets (nobody pressed Compile)',
          held['src'] != TEXT and names(n.inputs) == ['Gain', 'Tint'])
    graph_b, programs_b, _w = exported(n)
    d, bad, _gpu, cpu_b, why = _code_twin(graph_b, programs_b, keep_plan=True)
    check('the edited Text is a different picture on the CPU', _dmax(cpu_b, cpu_a) > 0.1, str(_dmax(cpu_b, cpu_a)))
    check('...and the GPU frame follows it past the cached plan (the source rides the graph the plan signs)',
          why is None and bad == 0 and d < 6e-3 and len(GSH._PLAN_CACHE) == plans + 1,
          f'{why} max {d} bad {bad} plans {plans} -> {len(GSH._PLAN_CACHE)}')

    # ---- what 1.92.0 shipped: the inline property under the graph's key
    graph_old = copy.deepcopy(graph_a)
    graph_old['nodes'][n.name]['props']['source_text'] = n.source_text
    _d, _b, gpu_starter, _c, why = _code_twin(graph_old, {n.name: SC.compile_shader(STARTER)})
    check('the starter shader is a DIFFERENT picture on this floor (the twins here can tell the two apart)',
          gpu_starter is not None and _dmax(gpu_starter, cpu_a) > 0.3,
          f'{why} {None if gpu_starter is None else _dmax(gpu_starter, cpu_a)}')
    _d, _b, gpu_old, _c, why = _code_twin(graph_old, programs_a)
    check('a graph source that is not its program\'s own is REFUSED by name, never drawn (the CPU renders it)',
          gpu_old is None and 'not the source its program was compiled from' in str(why), str(why))

    # ---- the one state that used to refuse instead: a starter uniform's
    # NAME under another type, here with the out variable renamed too
    held['src'] = ('uniform float baseColor = 0.35;\nuniform vec3 tint = vec3(0.2, 0.9, 0.4);\n\n'
                   'out vec4 Shade;\n\nvoid main() {\n    Shade = vec4(tint * baseColor, 1.0);\n}\n')
    check('the Compile press rebuilds the sockets from the Text (a float Base Color, an out named Shade)',
          n.compile_source() is not None and names(n.inputs) == ['Base Color', 'Tint']
          and n.inputs['Base Color'].type == 'VALUE' and names(n.outputs) == ['Shade'] and not n.error,
          f'{names(n.inputs)} {names(n.outputs)} {n.error}')
    tree = n.id_data
    tree.links.new(n.outputs['Shade'], tree.nodes.get('Halcyon').inputs['Diffuse Color'])
    graph_c, programs_c, _w = exported(n)
    d, bad, _gpu, _cpu, why = _code_twin(graph_c, programs_c)
    check('a Text reusing a starter uniform name under another type, out renamed: qualifies, and the twin holds',
          why is None and bad == 0 and d < 6e-3, f'{why} max {d} bad {bad}')

    # ---- an EMPTY assigned Text is an empty source on every road
    for label, blank in (('empty', ''), ('one blank line', '\n'), ('spaces and newlines', '  \n\n')):
        held['src'] = blank
        before = names(n.inputs)
        got = n.compile_source()
        check(f'an assigned Text that is {label}: the editor compiles nothing, keeps the sockets and says so by name',
              got is None and names(n.inputs) == before and 'shader_a' in n.error and 'empty' in n.error, repr(n.error))
        ed, pr, gr, warns = roads(n)
        check('...and so does the export: no program, the blank source in the graph (NOT the inline starter), '
              'one warning naming the node and the Text',
              ed == gr == blank and pr is None and n.source_text == STARTER and len(warns) == 1
              and n.name in warns[0] and 'shader_a' in warns[0] and 'empty' in warns[0],
              f'editor {short(ed)} program {short(pr)} graph {short(gr)} {warns}')
    op, said = SN.HALCYON_OT_compile_shader(), []
    op.node, op.report = n.name, (lambda kind, msg: said.append((kind, msg)))
    got = op.execute(types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=tree), active_node=None))
    check('the Compile button on an emptied Text is cancelled with that reason (not a bare "Compile failed")',
          got == {'CANCELLED'} and len(said) == 1 and said[0][0] == {'ERROR'} and 'shader_a' in said[0][1], f'{got} {said}')
    held['src'] = ''
    graph_e, programs_e, _w = exported(n)
    d, bad, _gpu, cpu_e, why = _code_twin(graph_e, programs_e)
    check('an emptied Text reads zeros on BOTH roads (the exporter ran the unseen inline starter under it)',
          not programs_e and why is None and bad == 0 and d < 6e-3, f'{sorted(programs_e)} {why} max {d} bad {bad}')

    def gone():
        raise ReferenceError('StructRNA of type Text has been removed')

    n.source = types.SimpleNamespace(name='removed', as_string=gone)
    ed, pr, gr, warns = roads(n)
    check('a Text that cannot be read is an empty source too, on every road',
          ed == gr == '' and pr is None and len(warns) == 1 and 'removed' in warns[0], f'{short(ed)} {short(gr)} {warns}')

    # ---- no Text assigned: the inline source, on every road
    n.source, n.source_text = None, _CODE_A
    check('with no Text the node compiles its inline source', n.compile_source() is not None
          and names(n.inputs) == ['Gain', 'Tint'] and not n.error, f'{names(n.inputs)} {n.error}')
    tree.links.new(n.outputs['Color'], tree.nodes.get('Halcyon').inputs['Diffuse Color'])
    ed, pr, gr, warns = roads(n)
    check('...and the editor, the program and the graph hold that one source', ed == pr == gr == _CODE_A and not warns,
          f'editor {short(ed)} program {short(pr)} graph {short(gr)} {warns}')
    graph_i, programs_i, _w = exported(n)
    d, bad, _gpu, _cpu, why = _code_twin(graph_i, programs_i)
    check('...and the inline source twins on the GPU through the real export', why is None and bad == 0 and d < 6e-3,
          f'{why} max {d} bad {bad}')
    n.source_text = '  \n'
    got = n.compile_source()
    ed, pr, gr, warns = roads(n)
    check('no Text and a blank inline source: nothing compiles and nobody complains (as it always was)',
          got is None and n.error == '' and ed == gr == '  \n' and pr is None and not warns, f'{n.error!r} {warns}')

    # ---- the rule is ONE function, read from the code itself
    def reads(f):
        """Every attribute name and string constant a function mentions."""
        tree_ = ast.parse(textwrap.dedent(inspect.getsource(f)))
        return ({a.attr for a in ast.walk(tree_) if isinstance(a, ast.Attribute)}
                | {c.value for c in ast.walk(tree_) if isinstance(c, ast.Constant) and isinstance(c.value, str)
                   and '\n' not in c.value and ' ' not in c.value})

    def calls(f):
        tree_ = ast.parse(textwrap.dedent(inspect.getsource(f)))
        return {ast.unparse(c.func) for c in ast.walk(tree_) if isinstance(c, ast.Call)}

    check('the node module and the exporter import the SAME two functions from shaders/compiler',
          SN.node_source is SC.node_source and SN.compile_node is SC.compile_node
          and EX.compile_node is SC.compile_node)
    check('get_source is node_source; the editor and the exporter both compile through compile_node',
          calls(Node.get_source) == {'node_source'} and 'compile_node' in calls(Node.compile_source)
          and calls(EX._compile_node) == {'compile_node'} and 'node_source' in calls(SC.compile_node),
          f'{calls(Node.get_source)} {calls(Node.compile_source)} {calls(EX._compile_node)}')
    own = {'source', 'source_text', 'as_string', 'language', 'try_compile'}
    resolve = {f.__qualname__: sorted(own & (reads(f) | calls(f)))
               for f in (Node.get_source, Node.compile_source, EX._compile_node)}
    check('...and none of the three resolves a source, reads the Text or picks the language itself',
          not any(resolve.values()), str(resolve))
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    readers = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('__pycache__', 'tests')]
        for fname in files:
            if fname.endswith('.py') and 'as_string' in open(os.path.join(folder, fname), encoding='utf-8').read():
                readers.append(os.path.relpath(os.path.join(folder, fname), root).replace(os.sep, '/'))
    check('exactly one file reads a Text datablock\'s contents: shaders/compiler.py',
          readers == ['shaders/compiler.py'], str(readers))
    check('the verbatim property table does not carry source_text (it would ship the inline copy again)',
          'source_text' not in EX.NODE_PROPS['HALCYON_CodeNode'])
    emit_tree = ast.parse(textwrap.dedent(inspect.getsource(GE.e_code_node)))
    keys = sorted({c.args[1].value for c in ast.walk(emit_tree)
                   if isinstance(c, ast.Call) and ast.unparse(c.func) == 'prop' and len(c.args) > 1
                   and isinstance(c.args[1], ast.Constant)})
    check('the GPU emitter reads ONE source key, source_text (the \'__source\' nothing ever wrote is gone)',
          keys == ['as_surface', 'language', 'source_text'] and '__source' not in reads(GE.e_code_node), str(keys))

    # ---- the two tooltips state the rule between them
    tip = SN.PROP_TIPS[('HALCYON_CodeNode', 'source')]
    check('the Text tooltip says an assigned Text is the source whatever it holds, and that the sockets wait '
          'for Compile (it promised a recompile "on every edit"; typing in the Text Editor fires nothing on the node)',
          len(tip) >= 40 and 'whatever it holds' in tip and 'empty text' in tip.lower() and 'on compile' in tip.lower()
          and 'every edit' not in tip and Node.__annotations__['source'].kw.get('description') == tip, tip)
    check('...and the Inline Source tooltip still gives the other half: the source while no Text is assigned',
          'no text datablock' in SN.PROP_TIPS[('HALCYON_CodeNode', 'source_text')].lower())


def _code_text(name, src):
    """A Text datablock, as far as shaders/compiler.node_source reads one."""
    return types.SimpleNamespace(name=name, as_string=lambda: src)


def _code_says(rgb, extra=''):
    """A coded shader that says its colour in its BODY. The sockets supply
    every uniform, so two sources that differed in a declared default alone
    would draw one picture under either program -- and hide a swap."""
    return f'uniform float gain = 1.0;\n{extra}\nout vec4 Color;\n\nvoid main() {{\n    Color = vec4({rgb}, 1.0);\n}}\n'


_CODE_RED = _code_says('gain, 0.05, 0.05')
_CODE_BLUE = _code_says('0.05, 0.05, gain')
_CODE_GREEN = _code_says('0.05, gain, 0.05')


def _code_group_node(tree, name, node_tree, gain=1.0):
    """A group node in `tree`: Gain in, Color out, `node_tree` inside."""
    node = _code_plain(tree, name, 'ShaderNodeGroup', ins=(('NodeSocketFloat', 'Gain'),),
                       outs=(('NodeSocketColor', 'Color'),), node_tree=node_tree)
    node.inputs['Gain'].default_value = gain
    return node


def _code_group_tree(SN, source=None, coded='Coded Shader', nested=None):
    """A node group's tree: Group Input (Gain) -> X -> Group Output (Color).
    X is a Coded Shader named `coded`, built from `source` -- or, given
    `nested` (another group's tree), a group node named 'Group' holding it:
    a group in a group."""
    tree = types.SimpleNamespace(name='Group Tree', nodes=_CodeNodes(), links=_CodeLinks())
    gin = _code_plain(tree, 'Group Input', 'NodeGroupInput', outs=(('NodeSocketFloat', 'Gain'),))
    gout = _code_plain(tree, 'Group Output', 'NodeGroupOutput', ins=(('NodeSocketColor', 'Color'),),
                       is_active_output=True)
    mid = _code_group_node(tree, 'Group', nested) if nested is not None else _code_coded(SN, tree, coded, source)
    tree.links.new(gin.outputs['Gain'], mid.inputs['Gain'])
    tree.links.new(mid.outputs['Color'], gout.inputs['Color'])
    return tree


def _code_feed(tree, nodes):
    """The master's Diffuse Color takes the SUM of the `nodes`' Color
    outputs (a chain of Add mixes at factor 1; one node links straight)."""
    hal = tree.nodes.get('Halcyon')
    for link in list(hal.inputs['Diffuse Color'].links):
        tree.links.remove(link)
    acc = nodes[0].outputs['Color']
    for k, node in enumerate(nodes[1:]):
        mix = _code_plain(tree, 'Mix' if not k else f'Mix.{k:03d}', 'ShaderNodeMixRGB',
                          ins=(('NodeSocketFloat', 'Fac'), ('NodeSocketColor', 'Color1'), ('NodeSocketColor', 'Color2')),
                          outs=(('NodeSocketColor', 'Color'),), blend_type='ADD', use_clamp=False)
        mix.inputs['Fac'].default_value = 1.0
        tree.links.new(acc, mix.inputs['Color1'])
        tree.links.new(node.outputs['Color'], mix.inputs['Color2'])
        acc = mix.outputs['Color']
    tree.links.new(acc, hal.inputs['Diffuse Color'])


def test_code_node_same_name_across_trees():
    """Compiled Coded Shader programs were filed under the node's NAME,
    in ONE dict that serialize_tree hands down its node-group recursion --
    and Blender keeps node names unique per tree, not per material. The
    material's "Coded Shader" and a group's "Coded Shader" (the default
    name, twice) left one program, the last compiled; n_halcyon_code looked
    it up by the same bare name, so one of the two nodes ran the OTHER's
    shader under its own sockets, on every road the CPU shades. Which one
    lost depended on the order of the nodes in the tree. The graph id is
    the node's PATH now (export.node_id), the one key the node dicts, the
    links and the programs are written under; the Bump pre-pass fields,
    filed by the same id, stop crossing trees with it. On the GPU the
    mangling tag was the id with its punctuation flattened -- 'Coded
    Shader' and 'Coded_Shader' in one tree shared every GLSL name, and the
    pass qualified and drew one of them twice; gpu/emit._code_tag hands out
    one tag per node. Groups themselves have no emitter: a coded shader
    inside one is refused by name and the CPU -- right, now -- draws it."""
    import ast
    import copy
    import importlib
    import inspect
    import itertools
    import re
    import textwrap
    from . import fakebpy
    fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    EX = importlib.import_module('halcyon.export')
    from ..core import nodeeval as NE
    from ..gpu import emit as GE
    red, blue, green = (_code_text(n, src) for n, src in (('red', _CODE_RED), ('blue', _CODE_BLUE),
                                                          ('green', _CODE_GREEN)))

    def exported(mat):
        images, programs, warnings = {}, {}, []
        graph = EX.serialize_tree(mat.node_tree, images, programs, warnings)
        return graph, programs, warnings

    def cpu(mat, size=(128, 96)):
        graph, programs, _w = exported(mat)
        return _code_cpu(graph, programs, size)

    def same(a, b):
        return bool(np.array_equal(a, b))

    def coded(graph):
        """{id: its graph source} of every Coded Shader at every depth."""
        out = {}
        for nid, node in graph['nodes'].items():
            if node['bl_idname'] == 'HALCYON_CodeNode':
                out[nid] = node['props']['source_text']
            if node.get('group'):
                out.update(coded(node['group']))
        return out

    def whole(graph):
        """Every tree of the graph hangs together: a node sits under its
        own id, every link and the group output name a node of that tree."""
        ok = all(nid == node['id'] and all(not s['link'] or s['link'][0] in graph['nodes'] for s in node['inputs'])
                 and (not node.get('group') or whole(node['group'])) for nid, node in graph['nodes'].items())
        return ok and graph['group_output'] in (None, *graph['nodes'])

    def as_1920(graph, programs):
        """The same export as 1.92.0 wrote it: every node under its bare
        name and the programs filed the same way, in the order
        serialize_tree compiles them (the last one wins)."""
        old = {}

        def bare(g, path):
            nodes = {}
            for nid, node in g['nodes'].items():
                node = dict(node, id=nid[len(path):])
                node['inputs'] = [dict(sk, link=[sk['link'][0][len(path):], sk['link'][1]] if sk['link'] else None)
                                  for sk in node['inputs']]
                if nid in programs:
                    old[node['id']] = programs[nid]
                if node.get('group'):
                    node['group'] = bare(node['group'], nid + EX.ID_SEP)
                nodes[node['id']] = node
            return {'nodes': nodes, 'output': g['output'] and g['output'][len(path):],
                    'group_output': g['group_output'] and g['group_output'][len(path):]}
        return bare(copy.deepcopy(graph), ''), old

    def pair(wire, inner='Coded Shader', top='Coded Shader', group_first=False):
        """The reported material: the master fed by its own tree's Coded
        Shader (RED, its Gain linked to the Value node's 0.7), by a group
        holding a Coded Shader (BLUE, the group's Gain 1.0), or by their
        sum. Both nodes are in the material whatever feeds the master."""
        mat = _code_master_tree(SN, red)
        tree = mat.node_tree
        code = tree.nodes.get('Coded Shader')
        code.name = top
        grp = _code_group_node(tree, 'Group', _code_group_tree(SN, blue, coded=inner))
        if group_first:
            tree.nodes.remove(grp)
            tree.nodes.insert(0, grp)
        _code_feed(tree, {'top': [code], 'group': [grp], 'both': [code, grp]}[wire])
        return mat

    # ---- each shader alone: what "its own picture" means below
    alone_red = cpu(_code_master_tree(SN, red))
    lone = _code_master_tree(SN, blue)
    lone.node_tree.nodes.get('Value').outputs['Value'].default_value = 1.0
    alone_blue = cpu(lone)
    check('the two shaders alone are two pictures (the red one and the blue one)', _dmax(alone_red, alone_blue) > 0.5,
          str(_dmax(alone_red, alone_blue)))

    # ---- the export: one program per node, under its path
    graph, programs, warns = exported(pair('both'))
    check('the reported state: a Coded Shader in the material and one inside a group, both under the default name',
          [n.name for n in pair('both').node_tree.nodes if n.bl_idname == 'HALCYON_CodeNode'] == ['Coded Shader']
          and [n.name for n in pair('both').node_tree.nodes.get('Group').node_tree.nodes
               if n.bl_idname == 'HALCYON_CodeNode'] == ['Coded Shader'])
    check('the export keeps TWO programs, the group\'s under its path (1.92.0 kept one, the last compiled)',
          sorted(programs) == ['Coded Shader', 'Group/Coded Shader'] and not warns, f'{sorted(programs)} {warns}')
    check('...and each is the program of its own node: compiled from the source that node\'s graph entry carries',
          coded(graph) == {'Coded Shader': _CODE_RED, 'Group/Coded Shader': _CODE_BLUE}
          and {k: p.source for k, p in programs.items()} == coded(graph), str(sorted(coded(graph))))
    inner = graph['nodes']['Group']['group']
    check('the group\'s tree is written under the same ids: its node keys, its links and its group output',
          sorted(inner['nodes']) == ['Group/Coded Shader', 'Group/Group Input', 'Group/Group Output']
          and inner['group_output'] == 'Group/Group Output' and whole(graph)
          and inner['nodes']['Group/Coded Shader']['inputs'][0]['link'] == ['Group/Group Input', 0],
          f'{sorted(inner["nodes"])} {inner["group_output"]}')
    check('a node of the material\'s own tree keeps its bare name as its id (every graph[\'nodes\'][node.name] holds)',
          sorted(graph['nodes']) == ['Coded Shader', 'Group', 'Halcyon', 'Material Output', 'Mix', 'Value']
          and graph['output'] == 'Material Output' and EX.node_id('Coded Shader.001') == 'Coded Shader.001')

    # ---- the picture, on the CPU, in both node orders (which node lost its
    # program depended on which was compiled last)
    for first in (False, True):
        order = 'listed first' if first else 'listed last'
        top_only, group_only = cpu(pair('top', group_first=first)), cpu(pair('group', group_first=first))
        check(f'group {order}: the material\'s Coded Shader draws ITS shader -- the frame of that shader alone, '
              'bit for bit', same(top_only, alone_red), str(_dmax(top_only, alone_red)))
        check(f'group {order}: the group\'s Coded Shader draws ITS shader -- the frame of that shader alone, '
              'bit for bit', same(group_only, alone_blue), str(_dmax(group_only, alone_blue)))
        both = cpu(pair('both', group_first=first))
        named = cpu(pair('both', inner='Coded Shader.001', group_first=first))
        check(f'group {order}: summed, the frame is the frame of the same tree with the inner node renamed, bit for '
              'bit -- and neither shader\'s alone', same(both, named) and _dmax(both, alone_red) > 0.3
              and _dmax(both, alone_blue) > 0.3, f'{_dmax(both, named)} {_dmax(both, alone_red)}')
        # what 1.92.0 exported for the same trees: one of the two is the wrong picture
        g_old, p_old = as_1920(*exported(pair('group' if first else 'top', group_first=first))[:2])
        old = _code_cpu(g_old, p_old)
        check(f'group {order}: the 1.92.0 export of it kept one program and drew the OTHER node\'s shader here '
              '(these frames can tell)', len(p_old) == 1 and _dmax(old, group_only if first else top_only) > 0.5,
              f'{sorted(p_old)} {_dmax(old, group_only if first else top_only)}')

    # ---- the same group instanced twice, and a group in a group
    def deep(unique=False):
        mat = _code_master_tree(SN, red)
        tree = mat.node_tree
        tint = _code_group_tree(SN, blue, coded='Coded Shader.001' if unique else 'Coded Shader')
        nest = _code_group_tree(SN, green, coded='Coded Shader.002' if unique else 'Coded Shader')
        nodes = [tree.nodes.get('Coded Shader'), _code_group_node(tree, 'Group', tint, gain=0.6),
                 _code_group_node(tree, 'Group.001', tint, gain=0.25),
                 _code_group_node(tree, 'Outer', _code_group_tree(SN, nested=nest), gain=0.4)]
        _code_feed(tree, nodes)
        return mat

    graph_d, programs_d, warns = exported(deep())
    ids = ['Coded Shader', 'Group.001/Coded Shader', 'Group/Coded Shader', 'Outer/Group/Coded Shader']
    check('one group instanced twice and a group in a group: four Coded Shaders named alike, four programs by path',
          sorted(programs_d) == ids and not warns and whole(graph_d), f'{sorted(programs_d)} {warns}')
    check('...each compiled from its own node\'s source (the two instances of one group: the same source, twice)',
          {k: p.source for k, p in programs_d.items()} == coded(graph_d)
          and [coded(graph_d)[k] for k in ids] == [_CODE_RED, _CODE_BLUE, _CODE_BLUE, _CODE_GREEN])
    frame_d, frame_u = _code_cpu(graph_d, programs_d), cpu(deep(unique=True))
    check('...and the frame is the frame of the same trees under four different names, bit for bit',
          same(frame_d, frame_u), str(_dmax(frame_d, frame_u)))
    old = _code_cpu(*as_1920(graph_d, programs_d))
    check('...where the 1.92.0 export of them ran ONE shader in all four nodes', len(as_1920(graph_d, programs_d)[1]) == 1
          and _dmax(old, frame_d) > 0.3, str(_dmax(old, frame_d)))

    # ---- the id is the path, whatever the names hold
    slashed = pair('both', top='Group/Coded Shader')
    graph_s, programs_s, _w = exported(slashed)
    check('a node NAMED \'Group/Coded Shader\' beside the group\'s Coded Shader: two ids, two programs, each its own',
          sorted(programs_s) == ['Group/Coded Shader', 'Group\\/Coded Shader'] and whole(graph_s)
          and programs_s['Group\\/Coded Shader'].source == _CODE_RED
          and programs_s['Group/Coded Shader'].source == _CODE_BLUE, str(sorted(programs_s)))
    check('...and the frame is the plainly named pair\'s, bit for bit',
          same(_code_cpu(graph_s, programs_s), cpu(pair('both'))))
    names = ['a', 'b', 'a/b', '/', 'a/', '/b', '\\', 'a\\', '\\/b', 'a\\/b', '\\\\', 'Coded Shader', 'Group']
    paths = [p for n in (1, 2, 3) for p in itertools.product(names, repeat=n)]

    def spelled(path):
        nid = ''
        for k, step in enumerate(path):
            nid = EX.node_id(step, nid + EX.ID_SEP if k else '')
        return nid

    check(f'no two of {len(paths)} paths (names holding the separator and the escape, one to three deep) spell one id',
          len({spelled(p) for p in paths}) == len(paths), f'{len({spelled(p) for p in paths})} of {len(paths)}')
    check('a name with neither character is its own id, and a path reads as one',
          all(EX.node_id(n) == n for n in ('Coded Shader', 'Coded Shader.001', 'Señal', 'Mix (old)'))
          and spelled(('Outer', 'Group', 'Coded Shader')) == 'Outer/Group/Coded Shader')

    # ---- a shader that fails to compile is named by its path
    held = {'src': _CODE_BLUE}
    mat = _code_master_tree(SN, red)
    grp = _code_group_node(mat.node_tree, 'Group', _code_group_tree(
        SN, types.SimpleNamespace(name='blue', as_string=lambda: held['src'])))
    _code_feed(mat.node_tree, [mat.node_tree.nodes.get('Coded Shader'), grp])
    held['src'] = _CODE_BLUE.replace('1.0);', '1.0;')
    graph_w, programs_w, warns = exported(mat)
    check('a group\'s Coded Shader that does not compile: the warning names it by its path, and the material\'s '
          'like-named node keeps its own program', sorted(programs_w) == ['Coded Shader'] and len(warns) == 1
          and warns[0].startswith('Group/Coded Shader: ') and programs_w['Coded Shader'].source == _CODE_RED,
          f'{sorted(programs_w)} {warns}')
    broken = _code_cpu(graph_w, programs_w)
    check('...which still draws the red shader (the group\'s output reads zeros: the sum is the red frame)',
          same(broken, alone_red), str(_dmax(broken, alone_red)))

    # ---- the GPU: the material's own coded shader qualifies beside a group
    # that holds its namesake; a group has no emitter, and is refused by name
    d, bad, _gpu, cpu_top, why = _code_twin(*exported(pair('top'))[:2])
    check('the material\'s Coded Shader, a like-named one in a group beside it: qualifies for the deferred pass',
          why is None, str(why))
    check('...and its GPU frame IS the CPU frame, the red one', d is not None and bad == 0 and d < 6e-3
          and same(cpu_top, alone_red), f'max {d} bad {bad}')
    _d, _b, gpu_old, _c, why = _code_twin(*as_1920(*exported(pair('top'))[:2]))
    check('the 1.92.0 export of it was REFUSED there (the graph\'s source against the other node\'s program) while '
          'the CPU drew the wrong shader', gpu_old is None and 'not the source its program was compiled from' in str(why),
          str(why))
    for wire in ('group', 'both'):
        _d, _b, gpu, cpu_g, why = _code_twin(*exported(pair(wire))[:2])
        check(f'a Coded Shader inside a group feeding the master ({wire}): REFUSED by name, never drawn -- groups '
              'have no emitter -- and the CPU frame is the right one', gpu is None and 'ShaderNodeGroup' in str(why)
              and same(cpu_g, alone_blue if wire == 'group' else cpu(pair('both', inner='Coded Shader.001'))),
              str(why))

    # ---- the GPU's own key: the tag a node's GLSL names are mangled under
    def two(name_a, name_b, src_a=_CODE_RED, src_b=_CODE_BLUE):
        """Two Coded Shaders in ONE tree, summed into the master."""
        mat = _code_master_tree(SN, _code_text('a', src_a))
        tree = mat.node_tree
        a = tree.nodes.get('Coded Shader')
        a.name = name_a
        b = _code_coded(SN, tree, name_b, _code_text('b', src_b))
        b.inputs['Gain'].default_value = 0.4
        _code_feed(tree, [a, b])
        return mat

    d, bad, _gpu, cpu_ref, why = _code_twin(*exported(two('Coded Shader', 'Other'))[:2])
    check('two Coded Shaders in one tree under unrelated names: the deferred pass twins the CPU (the reference)',
          why is None and bad == 0 and d < 6e-3, f'{why} max {d} bad {bad}')
    d, bad, _gpu, cpu_t, why = _code_twin(*exported(two('Coded Shader', 'Coded_Shader'))[:2])
    check('\'Coded Shader\' and \'Coded_Shader\' in one tree (one flattens onto the other: the pass drew one shader '
          'twice, 0.80 off the CPU over the whole floor): qualifies, twins, and is the reference frame',
          why is None and bad == 0 and d < 6e-3 and same(cpu_t, cpu_ref), f'{why} max {d} bad {bad}')
    # a tag that is another tag plus '_' plus the start of a name: `B_k` under
    # 'Coded' and `k` under 'Coded B' are both _cnCoded_B_k
    src_a, src_b = (_code_says('gain * B_k, 0.05, 0.05', 'float B_k = 1.0;\n'),
                    _code_says('0.05, 0.05, gain * k', 'float k = 0.5;\n'))
    d, bad, _gpu, cpu_p, why = _code_twin(*exported(two('Coded', 'Coded B', src_a, src_b))[:2])
    check('\'Coded\' declaring B_k beside \'Coded B\' declaring k (one GLSL name between them: 0.57 off): qualifies '
          'and twins', why is None and bad == 0 and d < 6e-3, f'{why} max {d} bad {bad}')
    check('...on the frame the same two shaders draw under unrelated names',
          same(cpu_p, cpu(two('Coded', 'Other', src_a, src_b))))

    em = GE.Emitter({})
    asked = ['Coded Shader', 'Coded Shader.001', 'Coded_Shader', 'Coded.Shader', 'Group/Coded Shader',
             'Group_Coded Shader', 'Group\\/Coded Shader', 'Coded', 'Coded B', '2 Coded Shader', 'Other',
             'Other Shader']
    tags = [GE._code_tag(em, k) for k in asked]
    check('a node alone in its pass, and Blender\'s own .001, keep the tag they always had (the id, flattened)',
          tags[:2] == ['Coded_Shader', 'Coded_Shader_001'] and GE._code_tag(GE.Emitter({}), 'code') == 'code'
          and GE._code_tag(GE.Emitter({}), 'Group/Coded Shader') == 'Group_Coded_Shader', str(tags[:2]))
    check('every later node whose flattened id would meet an earlier tag takes an ordinal in front, counted up '
          'past every tag that begins that way',
          tags[2:] == ['2_Coded_Shader', '3_Coded_Shader', 'Group_Coded_Shader', '2_Group_Coded_Shader',
                       'Group__Coded_Shader', '4_Coded', 'Coded_B', '2_2_Coded_Shader', 'Other', '2_Other_Shader'],
          str(tags))
    check('one tag per node: asked again, every node has the tag it was given, and no two nodes share one',
          [GE._code_tag(em, k) for k in asked] == tags and len(set(tags)) == len(asked), str(tags))
    check('...and no two of them can spell one GLSL name', not any(GE._tags_clash(a, b)
                                                                   for a, b in itertools.combinations(tags, 2)))
    # `once` also holds the emitters' own markers: a node NAMED like one
    graph_m, programs_m, _w = exported(two('__wang', 'Other'))
    em = GE.Emitter(graph_m)
    em.frame_mode, em.resolution, em.programs = True, (128, 96), programs_m
    em.once.add('__wang')                    # the Wang-hash helper's marker, already down
    em.output('__wang', 0)
    check('a Coded Shader named like an emitter marker (\'__wang\', the helper already inlined) is still inlined '
          'and run: the node\'s once-key is its own', len(em.inline) == 1 and '_cn__wang_main' in em.inline[0]
          and any('_cn__wang_main();' in line for line in em.lines), f'{len(em.inline)} {em.lines[-3:]}')
    # the clash rule IS "two tags can spell one name": every pair of tags up
    # to three characters, against every identifier up to four
    alphabet = 'a_1'
    words = [''.join(w) for n in (1, 2, 3, 4) for w in itertools.product(alphabet, repeat=n)]
    idents = [w for w in words if not w[0].isdigit()]
    short = [w for w in words if len(w) <= 3]
    spell = {t: {f'_cn{t}_{n}' for n in idents} for t in short}
    wrong = [(a, b) for a, b in itertools.combinations(short, 2)
             if GE._tags_clash(a, b) != bool(spell[a] & spell[b])]
    check(f'_tags_clash is exactly "two tags can spell one mangled name": all {len(short) * (len(short) - 1) // 2} '
          'pairs of short tags, by brute force', not wrong and GE._tags_clash('a', 'a')
          and GE._tags_clash('Coded', 'Coded_B') and not GE._tags_clash('Coded_Shader', 'Coded_Shader_001'),
          str(wrong[:6]))

    # ---- the Bump pre-pass: the other thing filed by (material, node id)
    bump_ins = (('NodeSocketFloat', 'Strength'), ('NodeSocketFloat', 'Distance'), ('NodeSocketFloat', 'Height'),
                ('NodeSocketVector', 'Normal'))

    def bump(tree, name):
        node = _code_plain(tree, name, 'ShaderNodeBump', ins=bump_ins, outs=(('NodeSocketVector', 'Normal'),),
                           invert=False)
        node.inputs['Strength'].default_value = node.inputs['Distance'].default_value = 1.0
        return node

    def bumped(inner_name):
        """The master's Normal fed by a group whose Bump has a FLAT height
        (it bends nothing); in the material's own tree a Bump over a steep
        ramp, wired to nothing."""
        mat = _code_master_tree(SN, red)
        tree = mat.node_tree
        hal = tree.nodes.get('Halcyon')
        hal.inputs.new('NodeSocketVector', 'Normal')
        hal.inputs.new('NodeSocketFloat', 'Bump Strength').default_value = 1.0
        ramp = _code_plain(tree, 'Gradient Texture', 'ShaderNodeTexGradient', ins=(('NodeSocketVector', 'Vector'),),
                           outs=(('NodeSocketColor', 'Color'), ('NodeSocketFloat', 'Fac')), gradient_type='LINEAR')
        steep = _code_plain(tree, 'Math', 'ShaderNodeMath', ins=(('NodeSocketFloat', 'Value'),) * 2,
                            outs=(('NodeSocketFloat', 'Value'),), operation='MULTIPLY', use_clamp=False)
        steep.inputs[1].default_value = 40.0
        tree.links.new(ramp.outputs['Fac'], steep.inputs[0])
        tree.links.new(steep.outputs['Value'], bump(tree, 'Bump').inputs['Height'])
        gtree = types.SimpleNamespace(name='Flat Bump', nodes=_CodeNodes(), links=_CodeLinks())
        gout = _code_plain(gtree, 'Group Output', 'NodeGroupOutput', ins=(('NodeSocketVector', 'Normal'),),
                           is_active_output=True)
        flat = _code_plain(gtree, 'Value', 'ShaderNodeValue', outs=(('NodeSocketFloat', 'Value'),))
        flat.outputs['Value'].default_value = 0.25
        inner_bump = bump(gtree, inner_name)
        gtree.links.new(flat.outputs['Value'], inner_bump.inputs['Height'])
        gtree.links.new(inner_bump.outputs['Normal'], gout.inputs['Normal'])
        grp = _code_plain(tree, 'Group', 'ShaderNodeGroup', outs=(('NodeSocketVector', 'Normal'),), node_tree=gtree)
        tree.links.new(grp.outputs['Normal'], hal.inputs['Normal'])
        return mat

    built, real = [], R._bump_height_fields

    def spy(*args, **kw):
        out = real(*args, **kw)
        built.append(sorted(key[1] for key in out))
        return out

    R._bump_height_fields = spy
    try:
        big = (320, 240)                      # the floor outgrows one shading batch: the pre-pass runs
        graph_b, programs_b, _w = exported(bumped('Bump'))
        frame_b = _code_cpu(graph_b, programs_b, big)
        fields_b = list(built)
        frame_n = cpu(bumped('Bump.001'), big)
        frame_o = _code_cpu(*as_1920(graph_b, programs_b), big)
    finally:
        R._bump_height_fields = real
    check('a Bump in the material and a Bump in a group, named alike, on a frame whose Bump pre-pass runs: the '
          'field is built for the material\'s own, and the group\'s is a different id',
          fields_b == [['Bump']] and 'Group/Bump' in graph_b['nodes']['Group']['group']['nodes'], str(fields_b))
    check('...the group\'s flat Bump bends nothing: the frame of the same trees with the inner Bump renamed, bit '
          'for bit', same(frame_b, frame_n), str(_dmax(frame_b, frame_n)))
    check('...where under 1.92.0\'s bare ids it took the OTHER Bump\'s gradient field (1.00 off over the floor)',
          _dmax(frame_o, frame_b) > 0.5, str(_dmax(frame_o, frame_b)))

    # ---- the rule, read from the code itself
    def parsed(f):
        return ast.parse(textwrap.dedent(inspect.getsource(f)))

    def calls(f):
        return {ast.unparse(c.func) for c in ast.walk(parsed(f)) if isinstance(c, ast.Call)}

    tree_ = parsed(EX.serialize_tree)
    read = [a for a in ast.walk(tree_) if isinstance(a, ast.Attribute) and a.attr == 'name'
            and isinstance(a.value, ast.Name) and a.value.id in ('node', 'from_node')]
    made = [c.args[0] for c in ast.walk(tree_) if isinstance(c, ast.Call) and ast.unparse(c.func) == 'node_id' and c.args]
    check('serialize_tree reads a node\'s name for ONE purpose, its id: the node key and the link, both through '
          'node_id', len(read) == 2 and all(any(a is m for m in made) for a in read),
          str([ast.unparse(a) for a in read]))
    sub = [c for c in ast.walk(tree_) if isinstance(c, ast.Call) and ast.unparse(c.func) == 'serialize_tree']
    check('...and hands each group its own node\'s id as the path (the recursion passed nothing down)',
          len(sub) == 1 and {k.arg: ast.unparse(k.value) for k in sub[0].keywords}.get('path') == 'nid + ID_SEP',
          str([ast.unparse(c) for c in sub]))
    source = open(os.path.abspath(EX.__file__), encoding='utf-8').read()
    check('one line of the exporter files a program, and it files it under the id it was handed',
          re.findall(r'programs\[[^\]]*\]\s*=', source) == ['programs[key] ='] and 'key = node_id(node.name)' in source,
          str(re.findall(r'programs\[[^\]]*\]\s*=', source)))
    check('the evaluator and the emitter look the program up by the graph id, and the emitter mangles under '
          '_code_tag alone', "node.get('id')" in inspect.getsource(NE.n_halcyon_code)
          and "node.get('id')" in inspect.getsource(GE.e_code_node) and '_code_tag' in calls(GE.e_code_node)
          and 're.sub' not in calls(GE.e_code_node), str(sorted(calls(GE.e_code_node))))


def test_code_node_rebuild_survives_undo_redo_and_load():
    """A coded shader's update callbacks may only FLAG the socket rebuild
    (needs_rebuild) and leave the topology to a one-shot timer. Blender
    pushes the edit's undo step BEFORE that timer runs, so the step holds
    the flag with the OLD sockets, and memfile undo brings it back without
    an update callback -- the only road to the timer. Redo (and an undo
    from any later step) left a node with stale sockets and a flag nobody
    would read; a file saved there carried it into the next session.
    Persistent undo_post / redo_post / load_post handlers re-arm it."""
    import ast
    import importlib
    import inspect
    import textwrap
    from . import fakebpy
    bpy = fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    events = _CODE_EVENTS
    scene = types.SimpleNamespace(name='Scene')
    text_a = types.SimpleNamespace(name='shader_a', as_string=lambda: _CODE_A)
    text_b = types.SimpleNamespace(name='shader_b', as_string=lambda: _CODE_B)

    def names(socks):
        return [s.name for s in socks]

    def node(tree=None):
        return (tree or bpy.data.materials[0].node_tree).nodes.get('Coded Shader')

    def queued():
        return timers.is_registered(SN._run_rebuilds)

    def whole(n, stripes=8.0):
        """The rebuilt node: the new source's sockets, the flag down, the
        links that still have somewhere to go."""
        return (not n.needs_rebuild and names(n.inputs) == ['Gain', 'Edge', 'Stripes']
                and names(n.outputs) == ['Color'] and n.inputs['Stripes'].default_value == stripes
                and [(l.from_node.name, l.from_socket.name) for l in n.inputs['Gain'].links] == [('Value', 'Value')]
                and [(l.to_node.name, l.to_socket.name) for l in n.outputs['Color'].links]
                == [('Material Output', 'Surface')])

    def stale(n):
        return n.needs_rebuild and n.language == 'HLSL' and names(n.inputs) == ['Gain', 'Tint']

    def state(n):
        return f'flag={n.needs_rebuild} lang={n.language} in={names(n.inputs)} queued={queued()}'

    with _code_bench(bpy, SN) as (handlers, timers):
        fn = SN._rearm_code_rebuilds
        check('register() puts the re-arm on undo_post, redo_post and load_post, once each',
              all(getattr(handlers, e).count(fn) == 1 for e in events))
        check('_rearm_code_rebuilds is a persistent handler (it survives File > Open)',
              hasattr(fn, '_bpy_persistent'), str(list(vars(fn).keys())))

        # ---- the edit, in Blender's order: callbacks, undo push, THEN the timer
        bpy.data.materials.append(_code_tree(SN, text_a))
        undo = _MemfileUndo(bpy)
        n = node()
        check('the node starts whole on the GLSL source', not n.needs_rebuild and names(n.inputs) == ['Gain', 'Tint'],
              state(n))
        undo.push()                                              # step 0
        n.source = text_b
        SN.HALCYON_CodeNode._source_changed(n, None)
        n.language = 'HLSL'
        SN.HALCYON_CodeNode._lang_changed(n, None)
        check('the update callbacks only flag: the sockets are untouched and ONE timer is queued',
              stale(n) and timers.fns.count(SN._run_rebuilds) == 1, state(n))
        undo.push()                                              # step 1: flagged, OLD sockets
        timers.pump()
        check('the timer rebuilds outside the step: new sockets, flag down, links kept',
              whole(n) and not queued(), state(n))
        n.inputs['Stripes'].default_value = 12.0
        undo.push()                                              # step 2: a later edit

        # ---- undo from the later step lands on the flagged one
        undo.restore(1)
        n = node()
        check('the restored step IS the flagged one, and the restore alone arms nothing (the 1.92.0 state)',
              stale(n) and not queued(), state(n))
        fakebpy.fire('undo_post', scene)
        check('undo_post schedules the rebuild of the flagged node', queued(), state(n))
        check('...and only schedules it: the handler itself changes no socket', stale(n), state(n))
        timers.pump()
        check('the scheduled rebuild leaves the node whole (the later edit undone, the Gain link kept)',
              whole(node()) and not queued(), state(node()))

        # ---- undo again: a clean step schedules nothing
        undo.restore(0)
        fakebpy.fire('undo_post', scene)
        n = node()
        check('a step with no flagged node schedules nothing',
              not queued() and not n.needs_rebuild and n.language == 'GLSL' and names(n.inputs) == ['Gain', 'Tint'],
              state(n))

        # ---- Ctrl+Shift+Z: the field's symptom
        undo.restore(1)
        fakebpy.fire('redo_post', scene)
        check('redo_post schedules the rebuild of the flagged node', queued() and stale(node()), state(node()))
        timers.pump()
        check('redo onto the flagged step ends whole', whole(node()) and not queued(), state(node()))
        undo.restore(2)
        fakebpy.fire('redo_post', scene)
        check('redo onto the later step keeps its own edit and schedules nothing',
              whole(node(), stripes=12.0) and not queued(), state(node()))

        # ---- File > Open of a file saved flagged, with a rebuild still queued
        undo.restore(1)
        SN._schedule_rebuild()
        check('a rebuild is queued when the load begins', queued() and SN._REBUILD_QUEUED[0])
        probe = lambda *_a: None                                 # noqa: E731
        handlers.load_post.append(probe)
        fakebpy.file_load('/tmp/flagged.blend')
        check('the load dropped the plain handler and kept the persistent re-arm',
              probe not in handlers.load_post and fn in handlers.load_post)
        check('load_post re-schedules although the dropped timer left its queued flag set',
              queued() and stale(node()), f'{state(node())} flag={SN._REBUILD_QUEUED[0]}')
        timers.pump()
        check('the loaded file ends whole', whole(node()) and not queued(), state(node()))

        # ---- a node group's tree is scanned too
        group = _code_tree(SN, text_a, name='Group').node_tree
        g = node(group)
        g.source, g.language, g.needs_rebuild = text_b, 'HLSL', True
        bpy.data.node_groups = [group]
        fakebpy.fire('undo_post', scene)
        check('a flagged node inside a node group is scheduled too', queued() and stale(g), state(g))
        timers.pump()
        check('...and rebuilt', whole(g), state(g))
        bpy.data.node_groups = []

        # ---- background: no main loop will ever fire a timer
        bpy.app.background = True
        undo.restore(1)
        fakebpy.fire('load_post', '/tmp/flagged.blend')
        check('in background the handler rebuilds at once and leaves no timer waiting',
              whole(node()) and not queued(), state(node()))
        bpy.app.background = False

        # ---- the Compile button's own undo step is whole
        undo.restore(1)
        n = node()
        op = SN.HALCYON_OT_compile_shader()
        op.node, op.report = n.name, (lambda *_a: None)
        ctx = types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=n.id_data), active_node=None)
        got = op.execute(ctx)
        check('the Compile operator clears the flag as it rebuilds', got == {'FINISHED'} and whole(n),
              f'{got} {state(n)}')
        undo.push()                                              # the operator's REGISTER|UNDO step
        undo.restore(undo.at)
        fakebpy.fire('redo_post', scene)
        check('landing on the step the operator pushed schedules nothing', not queued() and whole(node()),
              state(node()))

        # ---- the rules the fix stands on, read from the code itself
        def calls(f):
            tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
            return sorted({ast.unparse(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)})

        topo = [c for f in (SN.HALCYON_CodeNode._lang_changed, SN.HALCYON_CodeNode._source_changed)
                for c in calls(f)
                if c.split('.')[-1] in ('new', 'clear', 'remove', 'compile_source', 'rebuild_sockets',
                                        '_run_rebuilds')]
        check('the update callbacks still only flag (no socket is made or cleared inside one)', not topo, str(topo))
        pushes = [c for f in (SN._run_rebuilds, SN._schedule_rebuild, SN._rearm_code_rebuilds,
                              SN.HALCYON_CodeNode.compile_source, SN.HALCYON_CodeNode.rebuild_sockets)
                  for c in calls(f) if 'bpy.ops' in c or 'undo' in c.lower()]
        check('the re-armed rebuild never pushes an undo step (one pushed after an undo would cut the redo '
              'stack and trap Ctrl+Z on the flagged step)', not pushes, str(pushes))
        check('needs_rebuild keeps its >= 40-character tooltip and its census reason',
              len(SN.PROP_TIPS[('HALCYON_CodeNode', 'needs_rebuild')]) >= 40
              and len(str(SN.HALCYON_CodeNode.__annotations__['needs_rebuild'].kw.get('description') or '')) >= 40)

        SN.unregister()
        check('unregister() takes the re-arm off all three lists',
              not any(fn in getattr(handlers, e) for e in events))


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
