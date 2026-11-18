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
(13) A colour, value or vector output wired straight into a SHADER socket
    (the Material Output's Surface, a Mix / Add Shader input) drew two
    pictures: the CPU coerced it to an EMPTY closure and shaded the
    material's fallback surface, the deferred pass cast the same link
    into its colour chain and drew it as the albedo -- qualified, 1.69
    apart. `core/nodeeval.link_closure` reads Blender's implicit
    conversion (an Emission of it, bitwise the explicit node) and
    `gpu/shade._probe_material` refuses that lobe BY NAME.
(14) `closure_to_surface` decided a plain closure's diffuse, specular
    and roughness slots with `np.any(...)` over the BATCH: a pixel with
    no diffuse lobe kept the material's fallback diffuse fully lit when
    its chunk, pool band or render region held no diffuse pixel, and had
    none when one shared the batch -- a hard edge through the glow of a
    masked Mix Shader (Diffuse, Emission) wherever a batch boundary
    fell. Every slot is per pixel now, and a pixel with no diffuse
    weight has no diffuse response (level 0, no ambient share, a white
    albedo): Blender's answer for an Emission, a Glossy, a Glass or a
    Transparent alone.
(15) The same function's other batch reads, and the step (14) left. The
    ambient level carries the diffuse SHARE of the covered surface (a
    lobe at any weight kept the whole ambient term, so a Fac running up
    to 1 dropped it where it arrived). The MASTER branch's mix with
    plain lobes ran for the whole batch when any pixel carried one (a
    pure-master pixel's levels and Reflection were clipped at 1 beside a
    plain pixel) and is per pixel. And what cannot blend -- the model,
    the normal's carrier, the extras -- was named by the heaviest lobe
    over the BATCH: `ShadeJob.lobe_rank` casts that vote once per
    material, over the material's own surface.
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
    # the key read the positions as a strided SUM, which this translation
    # (its components cancel) kept: the CPU's G-buffer cache served the
    # unmoved raster. It carries an exact digest of every position now
    # (core/mathx.content_digest; test_r253_generated_coords, heading d)
    check('a translation whose components cancel changes the mesh key (it kept it: the stale-cache case)', k0 != k1)
    check('...and the plan signature fingerprints the baked frame on its own (the same mesh key handed to both)',
          GSH._plan_sig(j0, k0) != GSH._plan_sig(j1, k0))
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

    def __init__(self, node, stype, name):
        self.node, self.bl_idname, self.name = node, stype, name
        self.links = []
        self.halcyon_uniform = self.halcyon_key = ''
        self.halcyon_is_image = False
        if stype in self._DEFAULTS:
            self.default_value = self._DEFAULTS[stype]


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


def _code_tree(SN, source, name='Coded', cls=None):
    """A material whose tree is Value -> Coded Shader -> Material Output,
    the coded node being the REAL class (or `cls`, a subclass of it) run
    against stand-in sockets."""
    tree = types.SimpleNamespace(name=name, nodes=_CodeNodes(), links=_CodeLinks())

    def plain(nm, idname, ins=(), outs=()):
        node = types.SimpleNamespace(name=nm, bl_idname=idname, id_data=tree)
        node.inputs, node.outputs = _CodeSockets(node), _CodeSockets(node)
        for stype, sn in ins:
            node.inputs.new(stype, sn)
        for stype, sn in outs:
            node.outputs.new(stype, sn)
        tree.nodes.append(node)
        return node

    value = plain('Value', 'ShaderNodeValue', outs=(('NodeSocketFloat', 'Value'),))
    out = plain('Material Output', 'ShaderNodeOutputMaterial', ins=(('NodeSocketShader', 'Surface'),))
    code = (cls or SN.HALCYON_CodeNode)()
    code.name, code.id_data = 'Coded Shader', tree
    code.language, code.source, code.source_text = 'GLSL', source, ''
    code.as_surface, code.needs_rebuild = False, False
    code.error = code.warn = ''
    code.inputs, code.outputs = _CodeSockets(code), _CodeSockets(code)
    tree.nodes.append(code)
    code.init(None)
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
    own for the length of the test, whatever is installed when it starts
    (a test may have swapped the namespace). Everything touched in the
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
    check('the exporter ships language, as_surface and source_text for the node, as before',
          EX.NODE_PROPS['HALCYON_CodeNode'] == ('language', 'as_surface', 'source_text'))
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
        check('...and exactly the three shipped properties', sorted(props) == ['as_surface', 'language', 'source_text'],
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


# ------------------------------- (13) a colour wired into a shader socket

_WIRED_CODE = '''
uniform vec3 baseColor = vec3(0.1, 0.9, 0.3);
in vec3 vPosition;
out vec4 Color;
void main() {
    float s = step(0.5, fract(vPosition.x * 2.0));
    Color = vec4(baseColor * (0.4 + 0.6 * s), 1.0);
}
'''


def _wired_source(kind, rgb=(0.9, 0.3, 0.1)):
    """(nodes, link, programs): one source node and the link off its
    COLOUR, VALUE or VECTOR output (the vector is the interpolated
    normal, so it varies per pixel and goes negative), or a Coded
    Shader's Color output."""
    if kind == 'colour':
        return ({'src': {'id': 'src', 'bl_idname': 'ShaderNodeRGB', 'props': {'value': [rgb[0], rgb[1], rgb[2], 1.0]},
                         'inputs': [], 'outputs': [{'name': 'Color', 'type': 'RGBA'}]}}, ['src', 0], {})
    if kind == 'value':
        return ({'src': {'id': 'src', 'bl_idname': 'ShaderNodeValue', 'props': {'value': 0.4},
                         'inputs': [], 'outputs': [{'name': 'Value', 'type': 'VALUE'}]}}, ['src', 0], {})
    if kind == 'vector':
        return ({'src': {'id': 'src', 'bl_idname': 'ShaderNodeTexCoord', 'props': {}, 'inputs': [],
                         'outputs': [{'name': 'Generated', 'type': 'VECTOR'}, {'name': 'Normal', 'type': 'VECTOR'},
                                     {'name': 'UV', 'type': 'VECTOR'}]}}, ['src', 1], {})
    from ..shaders.compiler import compile_shader
    return ({'src': {'id': 'src', 'bl_idname': 'HALCYON_CodeNode', 'props': {'source_text': _WIRED_CODE, 'language': 'GLSL'},
                     'inputs': [{'name': 'Base Color', 'type': 'VECTOR', 'uniform': 'baseColor',
                                 'default': [0.9, 0.15, 0.6], 'link': None}],
                     'outputs': [{'name': 'Color', 'key': 'Color', 'type': 'RGBA'}]}},
            ['src', 0], {'src': compile_shader(_WIRED_CODE)})


def _wired_out(nodes, surface_link):
    nodes = dict(nodes)
    nodes['out'] = {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                    'inputs': [_sk('Surface', 'SHADER', None, surface_link)], 'outputs': []}
    return {'output': 'out', 'nodes': nodes}


def _wired_emission(nodes, link):
    """The explicit node Blender's conversion stands for: an Emission at
    Strength 1 whose Color takes the same link."""
    nodes = dict(nodes)
    nodes['em'] = {'id': 'em', 'bl_idname': 'ShaderNodeEmission', 'props': {},
                   'inputs': [_sk('Color', 'RGBA', [1.0, 1.0, 1.0, 1.0], link), _sk('Strength', 'VALUE', 1.0)],
                   'outputs': [{'name': 'Emission', 'type': 'SHADER'}]}
    return nodes, ['em', 0]


def _wired_two(nodes, link, idname):
    """A Diffuse BSDF and `link` through a Mix Shader (Fac 0.5) or an Add
    Shader: the second shader socket takes the link."""
    nodes = dict(nodes)
    nodes['dif'] = {'id': 'dif', 'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
                    'inputs': [_sk('Color', 'RGBA', [0.2, 0.3, 0.9, 1.0]), _sk('Roughness', 'VALUE', 0.0)],
                    'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    ins = [_sk('Shader', 'SHADER', None, ['dif', 0]), _sk('Shader', 'SHADER', None, link)]
    if idname == 'ShaderNodeMixShader':
        ins.insert(0, _sk('Fac', 'VALUE', 0.5))
    nodes['two'] = {'id': 'two', 'bl_idname': idname, 'props': {}, 'inputs': ins,
                    'outputs': [{'name': 'Shader', 'type': 'SHADER'}]}
    return nodes, ['two', 0]


def _wired_master(nodes, diffuse=None, emit=None):
    """The road the refusal names: a Halcyon Shader in between, the chain
    in its Diffuse Color or its Self-Illumination."""
    nodes = dict(nodes)
    nodes['hal'] = {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode', 'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': [_sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0], diffuse), _sk('Diffuse Level', 'VALUE', 1.0),
                               _sk('Specular Color', 'RGBA', [1, 1, 1, 1]), _sk('Specular Level', 'VALUE', 0.4),
                               _sk('Glossiness', 'VALUE', 24.0), _sk('Ambient', 'VALUE', 1.0),
                               _sk('Self-Illumination', 'RGBA', [0, 0, 0, 1], emit), _sk('Opacity', 'VALUE', 1.0)],
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]}
    return nodes, ['hal', 0]


def _wired_scene(graph, programs=None, device='CPU'):
    """The demo scene at 128x96 with `graph` on the Ball (material 1)."""
    from ..core.scene import Material
    st = base_settings(128, 96, shadows=True, render_device=device)
    st.color_depth, st.dither, st.output_scale = '24', 'NONE', 'NONE'
    sc = demo_scene(st, with_texture=False)
    mat = Material(name='Wired', index=1, graph=graph)
    mat.programs = programs or {}
    sc.materials[1] = mat
    return sc, st


def _wired_cpu(graph, programs=None):
    sc, st = _wired_scene(graph, programs)
    return np.asarray(R.render(sc, st))


def _wired_plan(graph, programs=None):
    """(passes, why) of the deferred plan for `graph` on the Ball."""
    from ..gpu import shade as GSH
    from .test_r251_material import _job_for
    sc, st = _wired_scene(graph, programs)
    g, job = _job_for(sc, st)
    GSH._PLAN_CACHE.clear()
    passes, why, _atl = GSH.plan_frame(job, g)
    return passes, why


def test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads():
    """A colour (a value, a vector) output wired straight into the
    Material Output's Surface -- no shader node in between -- drew two
    pictures: `coerce(v, SHADER)` handed the CPU an EMPTY closure, so it
    shaded the material's fallback surface as if nothing were wired,
    while `assemble_frame` cast the same link into the pass's one colour
    chain and drew it as the albedo. The plan QUALIFIED: 1.69 apart over
    all 598 pixels of the demo ball. Blender's own reading is the rule
    now (an Emission of the value, Strength 1): the CPU shades it,
    bitwise the explicit node, on every shader socket; the deferred plan
    refuses it by name."""
    from ..core import nodeeval as NE
    from ..gpu import shade as GSH
    from ..gpu import sky as GSKY
    from . import fakedevice
    from .test_r251_material import _job_for

    want = "'Wired': " + GSH.IMPLICIT_EMISSION_REFUSAL
    bare = _wired_cpu(_wired_out({}, None))
    sc0, st0 = _wired_scene(_wired_out({}, None))
    g0, _job0 = _job_for(sc0, st0)
    ball = (g0.tri >= 0) & (sc0.mesh.mat_index[np.maximum(g0.tri, 0)] == 1)
    check('the Ball covers the pixels the finding counted (598)', int(ball.sum()) == 598, str(int(ball.sum())))

    # ---- the three socket types straight into Surface
    for kind in ('colour', 'value', 'vector'):
        nodes, link, _pr = _wired_source(kind)
        wired = _wired_cpu(_wired_out(nodes, link))
        en, el = _wired_emission(nodes, link)
        check(f'a {kind} wired into Surface: the CPU frame is BITWISE the Emission node\'s (Strength 1, the same link in Color)',
              np.array_equal(wired, _wired_cpu(_wired_out(en, el))))
        check(f'a {kind} wired into Surface: no longer the frame of an unwired Surface socket (the 1.92.0 CPU answer)',
              float(np.abs(wired - bare)[ball].max()) > 0.25, str(float(np.abs(wired - bare)[ball].max())))
        check(f'a {kind} wired into Surface: every pixel off the Ball is bitwise the unwired frame',
              np.array_equal(wired[~ball], bare[~ball]))
        passes, why = _wired_plan(_wired_out(nodes, link))
        check(f'a {kind} wired into Surface: the deferred plan REFUSES (it qualified and drew the albedo)', passes is None, str(why))
        check(f'a {kind} wired into Surface: ...by name, word for word', why == want, str(why))
        # the road the refusal names qualifies and twins, in either socket
        for slot in ('emit', 'diffuse'):
            mn, ml = _wired_master(nodes, **{slot: link})
            sc, st = _wired_scene(_wired_out(mn, ml))
            d, nbad, _img, _cpu, mp, mwhy = _sim_vs_cpu(sc, st)
            where = 'Self-Illumination' if slot == 'emit' else 'Diffuse Color'
            check(f'a {kind} through a Halcyon Shader\'s {where}: the plan qualifies and the simulator frame is the CPU frame '
                  f'(max < 6e-3, no px > 1e-2)', mp is not None and d is not None and d < 6e-3 and nbad == 0,
                  str(mwhy if d is None else (d, nbad)))

    # ---- the colour ARRIVES, as light the surface gives off
    nodes, link, _pr = _wired_source('colour')
    lit = _wired_cpu(_wired_out(nodes, link))
    nodes_k, link_k, _pr = _wired_source('colour', rgb=(0.0, 0.0, 0.0))
    dark = _wired_cpu(_wired_out(nodes_k, link_k))
    gain = float(np.abs((lit - dark)[ball][:, :3] - np.array([0.9, 0.3, 0.1], f32)).max())
    check('the wired colour ARRIVES as itself: frame(colour) - frame(black) is the colour at every Ball pixel (1e-6)',
          gain < 1e-6, str(gain))
    under = float(np.abs(dark[ball][:, :3]).max())
    check('...over NOTHING: black wired in is a black Ball (the material\'s fallback diffuse was lit under the glow)',
          under == 0.0, str(under))
    check('...and the alpha stays opaque', float(lit[ball][:, 3].min()) == 1.0 and float(lit[ball][:, 3].max()) == 1.0)

    # ---- the text names what the user wired and where it shades
    for part in ('wired straight into a shader socket', 'an Emission of it', "Blender's implicit conversion",
                 'this material shades on the CPU exactly', 'a Halcyon Shader in between', 'Diffuse Color', 'Self-Illumination'):
        check(f'the refusal says "{part}"', part in GSH.IMPLICIT_EMISSION_REFUSAL)

    # ---- every other shader socket: the Mix and Add Shader inputs, a Reroute on the way, a Coded Shader's output
    for label, idname in (('a Mix Shader input', 'ShaderNodeMixShader'), ('an Add Shader input', 'ShaderNodeAddShader')):
        tn, tl = _wired_two(nodes, link, idname)
        wired = _wired_cpu(_wired_out(tn, tl))
        en, el = _wired_emission(nodes, link)
        xn, xl = _wired_two(en, el, idname)
        check(f'a colour wired into {label}: the CPU frame is bitwise the Emission node\'s in that socket',
              np.array_equal(wired, _wired_cpu(_wired_out(xn, xl))))
        passes, why = _wired_plan(_wired_out(tn, tl))
        check(f'a colour wired into {label}: the plan refuses by the same name', passes is None and why == want, str(why))
    zn, zl = _wired_two(nodes, link, 'ShaderNodeMixShader')
    zn['two']['inputs'][0]['default'] = 0.0
    passes, why = _wired_plan(_wired_out(zn, zl))
    check('a colour wired into a Mix Shader input at Fac 0: refused on the lobe\'s PRESENCE (a weight is sampled, the wiring is a fact)',
          passes is None and why == want, str(why))
    rn = dict(nodes)
    rn['rr'] = {'id': 'rr', 'bl_idname': 'NodeReroute', 'props': {}, 'inputs': [_sk('Input', 'RGBA', [0, 0, 0, 1], link)],
                'outputs': [{'name': 'Output', 'type': 'RGBA'}]}
    check('a colour through a Reroute into Surface: the CPU frame is the direct link\'s, bit for bit',
          np.array_equal(_wired_cpu(_wired_out(rn, ['rr', 0])), lit))
    passes, why = _wired_plan(_wired_out(rn, ['rr', 0]))
    check('a colour through a Reroute into Surface: the plan refuses by the same name', passes is None and why == want, str(why))
    cn, cl_, cprog = _wired_source('coded')
    coded = _wired_cpu(_wired_out(cn, cl_), cprog)
    en, el = _wired_emission(cn, cl_)
    check('a Coded Shader\'s Color wired into Surface: the CPU frame is bitwise the Emission node\'s',
          np.array_equal(coded, _wired_cpu(_wired_out(en, el), cprog)))
    check('...and the shader\'s own stripes reach the Ball', float(np.abs(coded - bare)[ball].max()) > 0.25)
    passes, why = _wired_plan(_wired_out(cn, cl_), cprog)
    check('a Coded Shader\'s Color wired into Surface: the plan refuses by the same name (the wiring, not the node type)',
          passes is None and why == want, str(why))

    # ---- what does NOT become an emission
    check('link_closure: a closure passes through untouched', NE.link_closure(NE.Closure().add('DIFFUSE', 1.0), 4).items[0][0] == 'DIFFUSE')
    check('link_closure: an output with no value stays the empty closure', len(NE.link_closure(None, 4)) == 0)
    one = NE.link_closure(np.full((4, 4), 0.5, f32), 4)
    check('link_closure: a value becomes ONE lobe -- EMISSION, weight 1, strength 1, marked implicit',
          len(one) == 1 and one.items[0][0] == 'EMISSION' and bool(np.all(one.items[0][1] == 1.0))
          and bool(np.all(one.items[0][2]['strength'] == 1.0)) and one.items[0][2].get('implicit') is True)
    ev = NE.GraphEvaluator({'output': 'out', 'nodes': {'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                            'inputs': [_sk('Surface', 'SHADER', [0.9, 0.3, 0.1, 1.0])], 'outputs': []}}}, NE.ShadeContext(4), {}, {})
    surf, _disp = ev.evaluate_surface()
    check('an UNLINKED shader socket stays empty whatever default a hand graph gave it (the link is the conversion)', len(surf) == 0)

    # ---- the controls: the roads on either side of the wiring are untouched
    sc, st = _wired_scene(_wired_out({}, None))
    d, nbad, _img, _cpu, cp, cwhy = _sim_vs_cpu(sc, st)
    check('control, an unwired Surface socket: the plan qualifies and the simulator frame is the CPU frame',
          cp is not None and d is not None and d < 6e-3 and nbad == 0, str(cwhy if d is None else (d, nbad)))
    dn = dict(nodes)
    dn['dif'] = {'id': 'dif', 'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
                 'inputs': [_sk('Color', 'RGBA', [1, 1, 1, 1], link), _sk('Roughness', 'VALUE', 0.0)],
                 'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    sc, st = _wired_scene(_wired_out(dn, ['dif', 0]))
    d, nbad, _img, _cpu, cp, cwhy = _sim_vs_cpu(sc, st)
    check('control, a Diffuse BSDF in between: the plan qualifies and the simulator frame is the CPU frame',
          cp is not None and d is not None and d < 6e-3 and nbad == 0, str(cwhy if d is None else (d, nbad)))
    en, el = _wired_emission(nodes, link)
    passes, why = _wired_plan(_wired_out(en, el))
    check('control, the explicit Emission node keeps its own refusal (a raw EMISSION lobe), not the wiring\'s',
          passes is None and 'a raw EMISSION lobe' in str(why) and 'wired straight' not in str(why), str(why))

    # ---- the device road end to end: the GPU device's frame IS the CPU device's, and says why
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    sc, st = _wired_scene(_wired_out(nodes, link), device='GPU')
    buf = io.StringIO()
    with fakedevice.installed():
        with redirect_stdout(buf):
            gpu = np.asarray(R.render(sc, st))
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    check('the GPU device road draws the CPU frame bit for bit (the fake driver)', np.array_equal(gpu, lit),
          str(float(np.abs(gpu - lit).max())) if gpu.shape == lit.shape else str(gpu.shape))
    check('...and prints the reason once, by name', buf.getvalue().count('[Halcyon GPU] shading on the CPU: ' + want) == 1,
          buf.getvalue()[-400:])

    # ---- the World Output's Surface is a shader socket too
    from ..core.scene import World

    def world_frame(graph):
        stw = base_settings(64, 48)
        scw = demo_scene(stw, with_texture=False)
        scw.world = World()
        scw.world.graph = graph
        return np.asarray(R.render(scw, stw)), scw, stw
    wn = {'src': dict(nodes['src'], props={'value': [0.15, 0.35, 0.8, 1.0]})}
    w_out = {'id': 'w', 'bl_idname': 'ShaderNodeOutputWorld', 'props': {}, 'outputs': []}
    wired_w, scw, stw = world_frame({'output': 'w', 'nodes': dict(wn, w=dict(w_out, inputs=[_sk('Surface', 'SHADER', None, ['src', 0])]))})
    bg = {'id': 'bg', 'bl_idname': 'ShaderNodeBackground', 'props': {},
          'inputs': [_sk('Color', 'RGBA', [0, 0, 0, 1], ['src', 0]), _sk('Strength', 'VALUE', 1.0)],
          'outputs': [{'name': 'Background', 'type': 'SHADER'}]}
    node_w, _s, _t = world_frame({'output': 'w', 'nodes': dict(wn, bg=bg, w=dict(w_out, inputs=[_sk('Surface', 'SHADER', None, ['bg', 0])]))})
    check('a colour wired into the World Output\'s Surface: the CPU frame is bitwise the Background node\'s at Strength 1',
          np.array_equal(wired_w, node_w))
    sky = wired_w[-6:, :, :3].reshape(-1, 3).mean(0)
    check('...and the colour is the sky', bool(np.allclose(sky, [0.15, 0.35, 0.8], atol=0.02)), str(np.round(sky, 3)))
    check('...which the GPU sky pass refuses by name (the node road)',
          GSKY.refusal(scw, stw) == 'the world node graph is evaluated on the CPU', str(GSKY.refusal(scw, stw)))


# ---------------------------------------------------------------------------
# (14) a closure with no diffuse lobe has no diffuse, at every pixel
# ---------------------------------------------------------------------------

_ND_BLUE = (0.2, 0.3, 0.9)
_ND_GLOW = (0.9, 0.3, 0.1)


def _nd_node(kind, color=None):
    """(nodes, link): one plain shader node of `kind`."""
    c4 = lambda c: [c[0], c[1], c[2], 1.0]                      # noqa: E731
    if kind == 'emission':
        return _wired_emission({}, None)[0] | {'em': dict(_wired_emission({}, None)[0]['em'], inputs=[
            _sk('Color', 'RGBA', c4(color or _ND_GLOW)), _sk('Strength', 'VALUE', 1.0)])}, ['em', 0]
    spec = {'diffuse': ('ShaderNodeBsdfDiffuse', [_sk('Color', 'RGBA', c4(color or _ND_BLUE)), _sk('Roughness', 'VALUE', 0.0)]),
            'glossy': ('ShaderNodeBsdfGlossy', [_sk('Color', 'RGBA', c4(color or (0.9, 0.6, 0.2))), _sk('Roughness', 'VALUE', 0.35),
                                                _sk('Anisotropy', 'VALUE', 0.0), _sk('Rotation', 'VALUE', 0.0)]),
            'metallic': ('ShaderNodeBsdfMetallic', [_sk('Base Color', 'RGBA', c4(color or (0.9, 0.6, 0.2))), _sk('Roughness', 'VALUE', 0.3)]),
            'glass': ('ShaderNodeBsdfGlass', [_sk('Color', 'RGBA', c4(color or (0.7, 0.9, 0.8))), _sk('Roughness', 'VALUE', 0.1),
                                              _sk('IOR', 'VALUE', 1.45)]),
            'transparent': ('ShaderNodeBsdfTransparent', [_sk('Color', 'RGBA', c4(color or (1.0, 1.0, 1.0)))]),
            'principled_metal': ('ShaderNodeBsdfPrincipled', [
                _sk('Base Color', 'RGBA', c4(color or (0.8, 0.5, 0.2))), _sk('Metallic', 'VALUE', 1.0), _sk('Roughness', 'VALUE', 0.4),
                _sk('IOR', 'VALUE', 1.45), _sk('Alpha', 'VALUE', 1.0), _sk('Specular IOR Level', 'VALUE', 0.5),
                _sk('Transmission Weight', 'VALUE', 0.0)])}[kind]
    return {kind: {'id': kind, 'bl_idname': spec[0], 'props': {}, 'inputs': spec[1],
                   'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}}, [kind, 0]


def _nd_lone(kind, color=None):
    nodes, link = _nd_node(kind, color)
    return _wired_out(nodes, link)


def _nd_masked(kind):
    """Mix Shader (Fac = the mask, a Diffuse BSDF, `kind`): the mask is
    the interpolated normal's z > 0, so the UPPER side of the Ball is
    `kind` alone, the lower side the Diffuse BSDF alone -- Fac exactly 0
    or 1, the case the batch rule split."""
    nodes, link = _nd_node(kind)
    dn, dl = _nd_node('diffuse')
    nodes = dict(nodes, **dn)
    nodes['tc'] = {'id': 'tc', 'bl_idname': 'ShaderNodeTexCoord', 'props': {}, 'inputs': [],
                   'outputs': [{'name': 'Generated', 'type': 'VECTOR'}, {'name': 'Normal', 'type': 'VECTOR'},
                               {'name': 'UV', 'type': 'VECTOR'}]}
    nodes['sep'] = {'id': 'sep', 'bl_idname': 'ShaderNodeSeparateXYZ', 'props': {},
                    'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0], ['tc', 1])],
                    'outputs': [{'name': 'X', 'type': 'VALUE'}, {'name': 'Y', 'type': 'VALUE'}, {'name': 'Z', 'type': 'VALUE'}]}
    nodes['gt'] = {'id': 'gt', 'bl_idname': 'ShaderNodeMath', 'props': {'operation': 'GREATER_THAN'},
                   'inputs': [_sk('Value', 'VALUE', 0.5, ['sep', 2]), _sk('Value', 'VALUE', 0.0)],
                   'outputs': [{'name': 'Value', 'type': 'VALUE'}]}
    nodes['mix'] = {'id': 'mix', 'bl_idname': 'ShaderNodeMixShader', 'props': {},
                    'inputs': [_sk('Fac', 'VALUE', 0.5, ['gt', 0]), _sk('Shader', 'SHADER', None, dl),
                               _sk('Shader', 'SHADER', None, link)],
                    'outputs': [{'name': 'Shader', 'type': 'SHADER'}]}
    return _wired_out(nodes, ['mix', 0])


def _nd_scene(graph, w=128, h=96, see_through=False, **kw):
    """The demo scene with `graph` on the Ball (material 1; None = a
    graph-less material), at any size and settings."""
    from ..core.scene import Material
    st = base_settings(w, h, **kw)
    st.color_depth, st.dither, st.output_scale = '24', 'NONE', 'NONE'
    sc = demo_scene(st, with_texture=False)
    mat = Material(name='Wired', index=1, graph=graph)
    mat.programs = {}
    if see_through:
        # what the exporter's alpha evidence says of a Transparent BSDF
        mat.has_alpha, mat.alpha_why = True, 'a Transparent BSDF node'
    sc.materials[1] = mat
    return sc, st


def _nd_frame(graph, w=128, h=96, band=None, see_through=False, lamps=True, **kw):
    sc, st = _nd_scene(graph, w, h, see_through, **kw)
    if not lamps:
        sc.lights = []
    R._GBUF_CACHE.clear()
    return np.asarray(R.render(sc, st, band=band))


def _nd_ball(w, h):
    from .test_r251_material import _job_for
    sc, st = _nd_scene(None, w, h)
    g, _job = _job_for(sc, st)
    return (g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 1)


def _nd_closure(kind, fac, n_extra=0):
    """The closure a masked Mix Shader (Diffuse, `kind`) hands on for the
    Fac values `fac`, one pixel each."""
    from ..core import nodeeval as NE
    fac = np.asarray(fac, f32)
    n = fac.size
    col = lambda c: np.tile(np.array([[c[0], c[1], c[2], 1.0]], f32), (n, 1))     # noqa: E731
    cl = NE.Closure()
    cl.add('DIFFUSE', 1.0 - fac, color=col(_ND_BLUE), roughness=np.full(n, 0.6, f32), model='LAMBERT')
    if kind == 'EMISSION':
        cl.add('EMISSION', fac, color=col(_ND_GLOW), strength=np.ones(n, f32))
    elif kind == 'GLOSSY':
        cl.add('GLOSSY', fac, color=col((0.9, 0.6, 0.2)), roughness=np.full(n, 0.35, f32), model='COOK_TORRANCE')
    elif kind in ('GLASS', 'REFRACTION'):
        cl.add(kind, fac, color=col((0.7, 0.9, 0.8)), roughness=np.full(n, 0.1, f32), ior=np.full(n, 1.45, f32))
    elif kind == 'HOLDOUT':
        cl.add('HOLDOUT', fac)
    else:                                                       # TRANSPARENT, TRANSLUCENT
        cl.add(kind, fac, color=col((0.3, 0.8, 0.3)))
    return cl


def _nd_slots(surf, i):
    """Every per-fragment slot of pixel `i`, as bytes."""
    from ..core import shading as SH
    out = {}
    for name in SH.Surface.__slots__:
        v = getattr(surf, name, None)
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == surf.n:
            out[name] = np.ascontiguousarray(v[i]).tobytes()
    return out


def test_a_closure_with_no_diffuse_lobe_has_no_diffuse_at_every_pixel():
    """`closure_to_surface` decided a plain closure's slots with
    `np.any(...)` over the BATCH. A pixel with no diffuse lobe kept the
    material's fallback diffuse fully LIT when no pixel of its batch had
    a diffuse weight, and had no diffuse at all when one did -- and a
    batch is a chunk of the shading loop, a band of the worker pool, a
    render region: a masked Mix Shader (Diffuse, Emission) showed a hard
    edge through its glow wherever one of those boundaries fell (the
    demo Ball at 1280x960, in process: 27929 pixels, 1.93 apart; a
    3-worker pool at 640x480: 3093 pixels, 1.79). The rule is per pixel
    now and it is Blender's: no diffuse lobe, no diffuse -- level 0 and
    no ambient share, over a white albedo, so the corner road and a
    see-through pixel's tint keep the identity. The graph-less material
    and the unwired socket keep the fallback surface."""
    from ..core import nodeeval as NE
    from ..core import parallel as PAR
    from ..core.scene import Material
    from ..gpu import shade as GSH
    from .test_r251_material import _job_for
    glow = np.array(_ND_GLOW, f32)
    mat = Material(name='m', index=0, diffuse=(0.31, 0.47, 0.11), diffuse_level=0.83, ambient_level=0.6,
                   specular=(0.9, 0.8, 0.7), specular_level=0.45, glossiness=19.0, roughness=0.27)
    st0 = base_settings(8, 8)

    def surf_of(kind, fac, material=mat):
        return R.closure_to_surface(_nd_closure(kind, fac), NE.ShadeContext(len(fac)), st0, material)[0]

    # ---- the finding, at the function: one pixel, two batches
    with_neighbour = surf_of('EMISSION', [1.0, 0.0])
    alone = surf_of('EMISSION', [1.0])
    check('a pure-emission pixel: diffuse level 0 with a diffuse neighbour in its batch (it always was)',
          float(with_neighbour.diffuse_level[0]) == 0.0)
    check('a pure-emission pixel: diffuse level 0 ALONE in its batch too (it read the fallback\'s level, fully lit)',
          float(alone.diffuse_level[0]) == 0.0, str(float(alone.diffuse_level[0])))

    # ---- no slot of any pixel reads its batch: alone, or among every other Fac
    facs = [1.0, 0.0, 0.4, 1.0, 0.0, 1e-7, 0.999999]
    for kind in ('EMISSION', 'GLOSSY', 'GLASS', 'REFRACTION', 'TRANSPARENT', 'HOLDOUT', 'TRANSLUCENT'):
        for material, tag in ((mat, 'a material'), (None, 'no material')):
            whole = surf_of(kind, facs, material)
            off = []
            for i, f in enumerate(facs):
                one = _nd_slots(surf_of(kind, [f], material), 0)
                here = _nd_slots(whole, i)
                off += [f'{name}@{f}' for name in one if one[name] != here.get(name)]
            check(f'Mix (Diffuse, {kind}), {tag}: every slot of every pixel is bitwise the same alone and in a mixed batch',
                  not off, ', '.join(off[:6]))

    # ---- the rule: no diffuse weight, no diffuse response
    for kind in ('EMISSION', 'GLOSSY', 'GLASS', 'REFRACTION', 'TRANSPARENT', 'HOLDOUT'):
        s = surf_of(kind, [1.0, 0.0, 0.4])
        check(f'a {kind}-only pixel: diffuse level 0, NO ambient share, a white albedo',
              float(s.diffuse_level[0]) == 0.0 and float(s.ambient[0]) == 0.0 and bool(np.all(s.diffuse[0] == 1.0)),
              f'{s.diffuse_level[0]} {s.ambient[0]} {s.diffuse[0]}')
        # (15): the ambient level carries the diffuse share of the COVERED surface -- a Transparent and a Holdout take none of it
        amb = [0.6, 0.6] if kind in ('TRANSPARENT', 'HOLDOUT') else [0.6, 0.6 * 0.6]
        check(f'...and beside it the diffuse pixels keep the lobe\'s colour, its weight as the level, the material\'s ambient by '
              f'their share of the covered surface',
              bool(np.allclose(s.diffuse[1:], _ND_BLUE, atol=1e-6)) and bool(np.allclose(s.diffuse_level[1:], [1.0, 0.6], atol=1e-6))
              and bool(np.allclose(s.ambient[1:], amb, rtol=0, atol=1e-6)), f'{s.diffuse[1:]} {s.diffuse_level[1:]} {s.ambient[1:]}')
    s = surf_of('EMISSION', [1.0, 0.0])
    check('no glossy weight: specular level 0, the colour the material\'s own (level 0 always was; the colour read its batch)',
          bool(np.all(s.specular_level == 0.0)) and bool(np.allclose(s.specular, (0.9, 0.8, 0.7), atol=1e-6)))
    check('no rough-carrying lobe: the pixel keeps the material\'s roughness and exponent (0 and 8192 in a mixed batch)',
          float(s.roughness[0]) == f32(0.27) and float(s.glossiness[0]) == 19.0 and abs(float(s.roughness[1]) - 0.6) < 1e-6,
          f'{s.roughness} {s.glossiness}')
    s = surf_of('TRANSLUCENT', [1.0])
    check('a Translucent BSDF IS a diffuse lobe (half its weight): level 0.5, the material\'s ambient kept',
          abs(float(s.diffuse_level[0]) - 0.5) < 1e-6 and float(s.ambient[0]) == f32(0.6))

    # ---- what must keep the fallback surface: the graph-less material and the empty closure
    for cl, tag in ((None, 'no closure (a graph-less material)'), (NE.Closure(), 'an EMPTY closure (an unwired Surface socket)')):
        s = R.closure_to_surface(cl, NE.ShadeContext(3), st0, mat)[0]
        check(f'{tag}: the material\'s own diffuse, level, ambient and specular, untouched',
              bool(np.allclose(s.diffuse, (0.31, 0.47, 0.11), atol=1e-6)) and bool(np.all(s.diffuse_level == f32(0.83)))
              and bool(np.all(s.ambient == f32(0.6))) and bool(np.all(s.specular_level == f32(0.45))),
              f'{s.diffuse[0]} {s.diffuse_level[0]} {s.ambient[0]} {s.specular_level[0]}')
    ball = _nd_ball(128, 96)
    bare = _nd_frame(_wired_out({}, None))
    check('the graph-less Ball and the unwired Surface socket are one frame, bit for bit', np.array_equal(_nd_frame(None), bare))
    check('...and it is the LIT fallback surface (the Ball mean is over 1.0 in every channel)',
          float(bare[ball][:, :3].mean(0).min()) > 1.0, str(bare[ball][:, :3].mean(0)))

    # ---- the frame: an Emission node alone is its colour, and nothing under it
    em = _nd_frame(_nd_lone('emission'))
    dev = float(np.abs(em[ball][:, :3] - glow).max())
    check('an Emission node on the Ball: every Ball pixel IS the colour (1e-6) -- the fallback diffuse was lit under it '
          '(the Ball mean was (2.04, 1.44, 1.27))', dev < 1e-6, str(dev))
    check('...every pixel off the Ball is bitwise the fallback frame\'s', np.array_equal(em[~ball], bare[~ball]))
    read = dict(_nd_node('emission')[0])
    read['s2r'] = {'id': 's2r', 'bl_idname': 'ShaderNodeShaderToRGB', 'props': {}, 'inputs': [_sk('Shader', 'SHADER', None, ['em', 0])],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]}
    read['em2'] = {'id': 'em2', 'bl_idname': 'ShaderNodeEmission', 'props': {},
                   'inputs': [_sk('Color', 'RGBA', [1.0, 1.0, 1.0, 1.0], ['s2r', 0]), _sk('Strength', 'VALUE', 1.0)],
                   'outputs': [{'name': 'Emission', 'type': 'SHADER'}]}
    dev = float(np.abs(_nd_frame(_wired_out(read, ['em2', 0]))[ball][:, :3] - glow).max())
    check('Shader to RGB of an Emission reads the colour itself: through a second Emission the Ball is still exactly the colour '
          '(the read lit a default surface under the glow: the Ball mean was (3.19, 2.58, 2.44))', dev < 1e-6, str(dev))
    for rate in ('VERTEX', 'FACE'):
        fr = _nd_frame(_nd_lone('emission'), shading_rate=rate)
        dev = float(np.abs(fr[ball][:, :3] - glow).max())
        check(f'an Emission node at the {rate} rate: the corner road\'s light x albedo still shows the colour (1e-5; a black '
              f'albedo would wipe it, the fallback\'s dimmed and lit it)', dev < 1e-5, str(dev))

    # ---- the frame: the two sides of a masked mix are each their own node's picture
    mixed = _nd_frame(_nd_masked('emission'))
    side = ball & (np.abs(mixed[..., :3] - glow).max(-1) < 1e-6)
    check('the mask splits the Ball (both sides hold pixels)', 100 < int(side.sum()) < int(ball.sum()) - 100,
          f'{int(side.sum())} of {int(ball.sum())}')
    dif = _nd_frame(_nd_lone('diffuse'))
    check('Mix (Diffuse, Emission) by a mask: the emission side is bitwise the Emission node\'s frame there',
          np.array_equal(mixed[side], em[side]))
    check('...and the diffuse side bitwise the Diffuse BSDF\'s', np.array_equal(mixed[ball & ~side], dif[ball & ~side]))

    # ---- the picture no longer reads its batches: bands, the pool, a region, the chunks
    W, H = 160, 120
    ball_b = _nd_ball(W, H)
    full = _nd_frame(_nd_masked('emission'), W, H)
    side_b = ball_b & (np.abs(full[..., :3] - glow).max(-1) < 1e-6)
    rows_all = np.nonzero(ball_b.any(1))[0]
    rows_glow = [int(y) for y in rows_all if bool(np.all(side_b[y][ball_b[y]]))]
    check('the masked Ball has rows wholly on the emission side (the batches the rule split)', len(rows_glow) >= 6, str(len(rows_glow)))
    if len(rows_glow) < 6:
        return              # nothing below has a premise (the checks above already failed by name)
    y0, y1 = min(rows_glow), max(rows_glow) + 1
    contiguous = rows_glow == list(range(y0, y1))
    check('...and they are one run of rows', contiguous, str(rows_glow))
    band = _nd_frame(_nd_masked('emission'), W, H, band=(y0, y1))
    check('a band wholly on the emission side is the full frame\'s rows, bit for bit (its glow sat on the lit fallback)',
          band.shape == full[y0:y1].shape and np.array_equal(band, full[y0:y1]),
          str(float(np.abs(band - full[y0:y1]).max())) if band.shape == full[y0:y1].shape else str(band.shape))
    edges = [0, y0, y1, H]
    joined = np.concatenate([_nd_frame(_nd_masked('emission'), W, H, band=(edges[i], edges[i + 1])) for i in range(3)], 0)
    check('three bands cut at the emission side\'s edges rejoin to the in-process frame, bit for bit', np.array_equal(joined, full))
    # (the four rows nearest the mask's edge: the widest of the cap)
    cols = np.nonzero(side_b[y0:y0 + 4].all(0))[0]
    check('the emission side holds a rectangle of its own (4 rows, 12 columns or more)', cols.size >= 12, str(cols.size))
    x0, x1 = int(cols.min()), int(cols.max()) + 1
    sc_r, st_r = _nd_scene(_nd_masked('emission'), W, H, use_border=True, border_min_x=x0 / W, border_max_x=x1 / W,
                           border_min_y=y0 / H, border_max_y=(y0 + 4) / H)
    rect = R.region_pixels(st_r, W, H)
    R._GBUF_CACHE.clear()
    region = np.asarray(R.render(sc_r, st_r))
    check('...and the region is that rectangle', rect == (x0, y0, x1, y0 + 4), f'{rect} vs {(x0, y0, x1, y0 + 4)}')
    check('a render region wholly on the emission side is the full frame inside its rect, zero outside, bit for bit',
          rect is not None and np.array_equal(region, R._apply_region(full.copy(), rect)), str(rect))
    saved = (R.MIN_CHUNK, R.MAX_CHUNK)
    try:
        R.MIN_CHUNK = R.MAX_CHUNK = 96
        chunked = _nd_frame(_nd_masked('emission'), W, H)
    finally:
        R.MIN_CHUNK, R.MAX_CHUNK = saved
    check('the shading loop in 96-pixel chunks draws the one-chunk frame, bit for bit (chunks of pure glow sat on the lit fallback)',
          np.array_equal(chunked, full), str(float(np.abs(chunked - full).max())))
    for rate in ('VERTEX', 'FACE'):
        full_r = _nd_frame(_nd_masked('emission'), W, H, shading_rate=rate)
        band_r = _nd_frame(_nd_masked('emission'), W, H, band=(y0, y1), shading_rate=rate)
        check(f'...and at the {rate} rate the same band is the full frame\'s rows, bit for bit', np.array_equal(band_r, full_r[y0:y1]))
        lit_side = float(np.abs(full_r[y0:y1][side_b[y0:y1]][:, :3] - glow).max())
        check(f'...where the emission side shows its colour (1e-5; the pixel pass\'s black albedo wiped it)', lit_side < 1e-5, str(lit_side))
    # the pool itself: 3 workers band 320x240 at rows 80 and 160, and the Ball's cap above 160 is pure glow
    WP, HP = 320, 240
    ball_p = _nd_ball(WP, HP)
    ref_p = _nd_frame(_nd_masked('emission'), WP, HP)
    side_p = ball_p & (np.abs(ref_p[..., :3] - glow).max(-1) < 1e-6)
    check('the pool\'s top band (rows 160..240) holds Ball pixels, every one on the emission side',
          int(ball_p[160:].sum()) > 100 and bool(np.all(side_p[160:][ball_p[160:]])), str(int(ball_p[160:].sum())))
    if PAR.find_interpreter() is not None:
        sc_p, st_p = _nd_scene(_nd_masked('emission'), WP, HP)
        img, err = PAR.render_parallel(sc_p, st_p, workers=3, scene_key='r253-no-diffuse')
        PAR.shutdown()
    else:
        img, err = None, 'no worker interpreter'
    if img is None:
        check('the worker pool draws the in-process frame (skipped: unavailable here)', True, str(err))
    else:
        img = np.asarray(img)
        check('the 3-worker pool draws the in-process frame, bit for bit (its top band lit the fallback under the glow)',
              np.array_equal(img, ref_p), str(float(np.abs(img - ref_p).max())) if img.shape == ref_p.shape else str(img.shape))

    # ---- the siblings through the same branch: no lamp, no light -- nothing answers the ambient pool
    for kind in ('glossy', 'metallic', 'principled_metal'):
        dark = _nd_frame(_nd_lone(kind), lamps=False)
        check(f'a {kind} Ball with every lamp removed is BLACK (no diffuse lobe: the ambient pool lit the fallback colour)',
              float(np.abs(dark[ball][:, :3]).max()) == 0.0, str(float(np.abs(dark[ball][:, :3]).max())))
        lit = _nd_frame(_nd_lone(kind))
        mx = _nd_frame(_nd_masked(kind))
        side_k = ball & np.all(mx[..., :3] == lit[..., :3], -1) & ~np.all(mx[..., :3] == dif[..., :3], -1)
        check(f'Mix (Diffuse, {kind}) by a mask: every Ball pixel is bitwise its own side\'s node alone',
              bool(np.all((np.all(mx == lit, -1) | np.all(mx == dif, -1))[ball])) and int(side_k.sum()) > 100, str(int(side_k.sum())))
        if kind == 'glossy' and int(side_k.sum()) >= GSH.PROBE_FRAGMENTS:
            # the deferred probe is a batch too: sixteen fragments, here every one on the glossy side
            # (without a glossy side the check above has already failed by name)
            sc_k, st_k = _nd_scene(_nd_masked(kind))
            g_k, job_k = _job_for(sc_k, st_k)
            ys_k, xs_k = np.nonzero(side_k)
            pick = np.linspace(0, ys_k.size - 1, GSH.PROBE_FRAGMENTS).astype(np.int64)
            bake, _model, pwhy = GSH._probe_material(job_k, g_k, 1, ys_k[pick], xs_k[pick])
            check('the probe is a batch too: sixteen glossy fragments of the MASKED Ball bake diffuse level 0 and ambient 0, the '
                  'frame\'s own rule (they baked the lit fallback for the whole material: a Glossy Ball with a Diffuse island '
                  'qualified 1.93 off the CPU on all 598 pixels)',
                  bake is not None and bake.get('diffuse_level') == 0.0 and bake.get('ambient') == 0.0,
                  str(pwhy if bake is None else (bake.get('diffuse_level'), bake.get('ambient'))))
    check('control, a Diffuse BSDF Ball with every lamp removed is NOT black (its lobe answers the ambient pool)',
          float(_nd_frame(_nd_lone('diffuse'), lamps=False)[ball][:, :3].max()) > 0.01)
    glass_dark = _nd_frame(_nd_lone('glass'), lamps=False, env_reflection=False)
    check('a glass Ball with no lamp and no environment is BLACK', float(np.abs(glass_dark[ball][:, :3]).max()) == 0.0)

    # ---- transparent alone: invisible where alpha composites, black where it cannot
    clear = _nd_frame(_nd_lone('transparent'), see_through=True)
    check('a Transparent BSDF Ball composites to alpha 1 over the floor behind it (the layer road, as before)',
          float(clear[ball][:, 3].min()) == 1.0 and float(clear[ball][:, :3].max()) < 1.0)
    solid = _nd_frame(_nd_lone('transparent'), see_through=True, transparency='NONE')
    check('...and under Transparency None, where nothing composites, it is opaque BLACK (it was the lit fallback surface)',
          float(np.abs(solid[ball][:, :3]).max()) == 0.0 and float(solid[ball][:, 3].min()) == 1.0)

    # ---- both roads: what the deferred plan qualifies it twins, what it cannot carry it refuses by name
    for kind in ('glossy', 'metallic'):
        for rate in ('PIXEL', 'VERTEX', 'FACE'):
            sc, st = _nd_scene(_nd_lone(kind), shading_rate=rate)
            d, nbad, _img, _cpu, passes, why = _sim_vs_cpu(sc, st)
            check(f'a {kind} Ball at the {rate} rate: the plan qualifies and the simulator frame is the CPU frame (max < 6e-3, no px > 1e-2)',
                  passes is not None and d is not None and d < 6e-3 and nbad == 0, str(why if d is None else (d, nbad)))
            g, job = _job_for(sc, st)
            ys, xs = np.nonzero((g.tri >= 0) & (sc.mesh.mat_index[np.maximum(g.tri, 0)] == 1))
            bake, _model, pwhy = GSH._probe_material(job, g, 1, ys[::37], xs[::37])
            check(f'...and the probe bakes diffuse level 0, ambient 0, a white diffuse (the same rule, read through closure_to_surface)',
                  bake is not None and bake.get('diffuse_level') == 0.0 and bake.get('ambient') == 0.0
                  and bake.get('diffuse') == (1.0, 1.0, 1.0),
                  str(pwhy if bake is None else (bake.get('diffuse_level'), bake.get('ambient'), bake.get('diffuse'))))
    for mode in ('NONE', 'STIPPLE'):
        sc, st = _nd_scene(_nd_lone('transparent', (0.9, 0.2, 0.2)), see_through=True, transparency=mode)
        d, nbad, _img, _cpu, passes, why = _sim_vs_cpu(sc, st)
        check(f'a tinted Transparent BSDF Ball under Transparency {mode}: the simulator frame is the CPU frame (the pass lit the '
              f'node\'s colour as an albedo over the CPU\'s fallback: 1.44 off on all 598 pixels)',
              passes is not None and d is not None and d < 6e-3 and nbad == 0, str(why if d is None else (d, nbad)))
    for graph, tag in ((_nd_lone('emission'), 'an Emission node'), (_nd_masked('emission'), 'Mix (Diffuse, Emission) by a mask')):
        sc, st = _nd_scene(graph)
        g, job = _job_for(sc, st)
        GSH._PLAN_CACHE.clear()
        passes, why, _atl = GSH.plan_frame(job, g)
        check(f'{tag}: the deferred plan still refuses the raw EMISSION lobe by name (both devices draw the CPU frame)',
              passes is None and 'a raw EMISSION lobe' in str(why), str(why))


# ---------------------------------------------------------------------------
# (15) the ambient share, the master branch per pixel, the material's vote
# ---------------------------------------------------------------------------


def _nd_master(nid, model='PHONG', color=_ND_BLUE, **sock):
    """(nodes, link): one Halcyon Shader; `sock` overrides socket values
    by name (Diffuse Level, Specular Level, Glossiness, Ambient, Opacity)."""
    v = {'Diffuse Level': 1.0, 'Specular Level': 0.4, 'Glossiness': 24.0, 'Ambient': 1.0, 'Opacity': 1.0}
    v.update(sock)
    return {nid: {'id': nid, 'bl_idname': 'HALCYON_ShaderNode', 'props': {'model': model, 'toon_steps': 2},
                  'inputs': [_sk('Diffuse Color', 'RGBA', [color[0], color[1], color[2], 1.0]),
                             _sk('Diffuse Level', 'VALUE', v['Diffuse Level']), _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
                             _sk('Specular Level', 'VALUE', v['Specular Level']), _sk('Glossiness', 'VALUE', v['Glossiness']),
                             _sk('Ambient', 'VALUE', v['Ambient']), _sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
                             _sk('Opacity', 'VALUE', v['Opacity'])],
                  'outputs': [{'name': 'Surface', 'type': 'SHADER'}]}}, [nid, 0]


def _nd_mix(a, b, fac=None, thr=0.0):
    """Mix Shader of two (nodes, link) pairs: a constant `fac`, or the
    mask (the interpolated normal's z > `thr` is `b` alone)."""
    probe = _nd_masked('emission')['nodes']
    nodes = dict(a[0], **b[0])
    for nid in ('tc', 'sep', 'gt'):
        nodes[nid] = probe[nid]
    nodes['gt'] = dict(nodes['gt'], inputs=[nodes['gt']['inputs'][0], _sk('Value', 'VALUE', thr)])
    nodes['mix'] = {'id': 'mix', 'bl_idname': 'ShaderNodeMixShader', 'props': {},
                    'inputs': [_sk('Fac', 'VALUE', 0.5 if fac is None else fac, ['gt', 0] if fac is None else None),
                               _sk('Shader', 'SHADER', None, a[1]), _sk('Shader', 'SHADER', None, b[1])],
                    'outputs': [{'name': 'Shader', 'type': 'SHADER'}]}
    return _wired_out(nodes, ['mix', 0])


def _nd_master_closure(kind, fac, hot=True, ambient=0.3):
    """Mix Shader (a Halcyon Shader, `kind`) for the Fac values `fac`,
    one pixel each. `hot` gives the master levels over 1 and a low
    opacity: the values a batch-wide clip or a 1 - (1 - o) shows on."""
    from ..core import nodeeval as NE
    fac = np.asarray(fac, f32)
    n = fac.size
    col = lambda c: np.tile(np.array([[c[0], c[1], c[2], 1.0]], f32), (n, 1))     # noqa: E731
    full = lambda v: np.full(n, v, f32)                                        # noqa: E731
    cl = NE.Closure()
    cl.add('HALCYON', 1.0 - fac, color=col(_ND_BLUE), model='PHONG', ambient=full(ambient),
           diffuse_level=full(1.5 if hot else 0.7), spec_level=full(2.0 if hot else 0.4), reflect=full(1.4 if hot else 0.2),
           opacity=full(0.1 if hot else 1.0), emission=col((0.1, 0.2, 0.3)), glossiness=full(31.0), roughness=full(0.2),
           translucency=full(0.3))
    if kind == 'EMISSION':
        cl.add('EMISSION', fac, color=col(_ND_GLOW), strength=np.ones(n, f32))
    elif kind == 'HOLDOUT':
        cl.add('HOLDOUT', fac)
    elif kind in ('GLASS', 'REFRACTION'):
        cl.add(kind, fac, color=col((0.7, 0.9, 0.8)), roughness=full(0.1), ior=full(1.45))
    elif kind in ('DIFFUSE', 'GLOSSY'):
        cl.add(kind, fac, color=col((0.9, 0.6, 0.2)), roughness=full(0.35), model='LAMBERT' if kind == 'DIFFUSE' else 'COOK_TORRANCE')
    elif kind is not None:                                      # TRANSPARENT, TRANSLUCENT
        cl.add(kind, fac, color=col((0.3, 0.8, 0.3)))
    return cl


def test_a_mix_carries_its_ambient_by_share_and_votes_once_per_material():
    """What (14) left in `closure_to_surface`, each seen in a frame
    first. (a) A diffuse lobe at any weight above zero kept the WHOLE
    ambient term, so a Fac running up to 1 dropped the lobe's colour
    times the ambient pool where it arrived ((0.021, 0.036, 0.145) on
    the demo Ball, plain and master alike): the ambient level carries
    the diffuse share of the covered surface now. (b) The master
    branch's mix with plain lobes ran for the WHOLE batch when any pixel
    carried a plain weight: a pure-master pixel's levels and Reflection
    were clipped at 1 beside a plain pixel and its opacity re-derived --
    per pixel now, the master alone bit for bit. A master mixed with an
    Emission by a mask rendered its glow BLACK at the vertex and face
    rates (the weighted colour is black where no master weighs). (c)
    What cannot blend was named by the heaviest lobe over the BATCH: two
    masters mixed by a mask shaded each band with whichever side filled
    it (3089 pixels, 1.51, at 640x480 across three bands). The vote is
    cast once per material, over its own surface."""
    from ..core import nodeeval as NE
    from ..core.scene import Material
    from ..gpu import shade as GSH
    from .test_r251_material import _job_for
    glow = np.array(_ND_GLOW, f32)
    A = 0.6
    mat = Material(name='m', index=0, diffuse=(0.31, 0.47, 0.11), ambient_level=A)
    st0 = base_settings(8, 8)

    def plain(kind, fac):
        return R.closure_to_surface(_nd_closure(kind, fac), NE.ShadeContext(len(fac)), st0, mat)[0]

    def master(kind, fac, **kw):
        return R.closure_to_surface(_nd_master_closure(kind, fac, **kw), NE.ShadeContext(len(fac)), st0, mat)[0]

    # ---- (a) the plain branch: the ambient level is the material's times the diffuse share
    facs = np.array([0.25, 0.5, 0.999], f32)
    want = f32(A) * (f32(1.0) - facs)
    for kind in ('EMISSION', 'GLOSSY', 'GLASS', 'REFRACTION'):
        got = plain(kind, facs).ambient
        check(f'Mix (Diffuse, {kind}) at Fac 0.25 / 0.5 / 0.999: the ambient level is the material\'s x the diffuse share (it was the '
              f'whole level at every Fac under 1)', bool(np.allclose(got, want, rtol=0, atol=1e-7)), f'{got} vs {want}')
    whole = f32(A)
    check('Mix (Diffuse, Transparent): the share is of the COVERED surface -- the ambient level is the material\'s own, bit for bit',
          bool(np.all(plain('TRANSPARENT', [0.25, 0.5, 0.999, 0.3]).ambient == whole)))
    check('Mix (Diffuse, Holdout): the same', bool(np.all(plain('HOLDOUT', [0.25, 0.5, 0.9]).ambient == whole)))
    check('Mix (Diffuse, Translucent): a Translucent answers the pool at its WHOLE weight -- the material\'s own, bit for bit',
          bool(np.all(plain('TRANSLUCENT', [0.0, 0.25, 0.5, 1.0]).ambient == whole)))
    n3 = 3
    col = lambda c: np.tile(np.array([[c[0], c[1], c[2], 1.0]], f32), (n3, 1))     # noqa: E731
    added = NE.Closure().add('DIFFUSE', np.ones(n3, f32), color=col(_ND_BLUE)).add('EMISSION', np.ones(n3, f32), color=col(_ND_GLOW),
                                                                                  strength=np.ones(n3, f32))
    check('Add Shader (Diffuse, Emission): nothing is diluted -- the material\'s own, bit for bit',
          bool(np.all(R.closure_to_surface(added, NE.ShadeContext(n3), st0, mat)[0].ambient == whole)))
    f1, f2 = f32(0.3), f32(0.7)
    nested = NE.Closure().add('DIFFUSE', np.full(n3, (1 - f1) * (1 - f2), f32), color=col(_ND_BLUE))
    nested.add('TRANSPARENT', np.full(n3, f1 * (1 - f2), f32), color=col((1, 1, 1))).add('TRANSPARENT', np.full(n3, f2, f32), color=col((1, 1, 1)))
    check('Mix (Mix (Diffuse, Transparent), Transparent): a share within float noise of the whole IS the whole, bit for bit',
          bool(np.all(R.closure_to_surface(nested, NE.ShadeContext(n3), st0, mat)[0].ambient == whole)))
    three = NE.Closure().add('DIFFUSE', np.full(n3, 0.25, f32), color=col(_ND_BLUE))
    three.add('EMISSION', np.full(n3, 0.25, f32), color=col(_ND_GLOW), strength=np.ones(n3, f32)).add('TRANSPARENT', np.full(n3, 0.5, f32),
                                                                                                    color=col((1, 1, 1)))
    got = R.closure_to_surface(three, NE.ShadeContext(n3), st0, mat)[0].ambient
    check('Mix (Mix (Diffuse, Emission, 0.5), Transparent, 0.5): half of what is COVERED is diffuse -- half the level',
          bool(np.allclose(got, f32(A) * f32(0.5), rtol=0, atol=1e-7)), str(got))
    check('...and still 0 where no diffuse lobe weighs at all', float(plain('EMISSION', [1.0]).ambient[0]) == 0.0)

    # ---- (b) the master branch: no slot of any pixel reads its batch
    facs_b = [0.0, 1.0, 0.4, 0.0, 1e-7]
    for kind in ('EMISSION', 'GLOSSY', 'DIFFUSE', 'GLASS', 'TRANSPARENT', 'TRANSLUCENT', 'HOLDOUT'):
        batch = master(kind, facs_b)
        off = []
        for i, f in enumerate(facs_b):
            one = _nd_slots(master(kind, [f]), 0)
            here = _nd_slots(batch, i)
            off += [f'{name}@{f}' for name in one if one[name] != here.get(name)]
        check(f'Mix (a hot Halcyon Shader, {kind}): every slot of every pixel is bitwise the same alone and in a mixed batch',
              not off, ', '.join(off[:6]))
    alone = _nd_slots(master(None, [0.0]), 0)
    beside = _nd_slots(master('GLOSSY', [0.0, 1.0]), 0)
    check('a pure-master pixel beside a glossy one IS the master alone, bit for bit (Specular Level 2 and Reflection 1.4 were clipped '
          'at 1 there, the opacity re-derived)', alone == beside, ', '.join(k for k in alone if alone[k] != beside.get(k)))
    s = master('GLOSSY', [0.0, 0.5])
    check('...its Specular Level 2 stands; the mixed pixel beside it sums toward 1 as it did',
          float(s.specular_level[0]) == 2.0 and float(s.specular_level[1]) == 1.0, str(s.specular_level))

    def three_way(emit, third, kind):
        emit, third = np.asarray(emit, f32), np.asarray(third, f32)
        cl = _nd_master_closure(None, emit + third)
        k = emit.size
        cl.add('EMISSION', emit, color=np.tile(np.array([[0.9, 0.3, 0.1, 1.0]], f32), (k, 1)), strength=np.ones(k, f32))
        cl.add(kind, third, color=np.tile(np.array([[0.9, 0.6, 0.2, 1.0]], f32), (k, 1)), roughness=np.full(k, 0.35, f32),
               model='COOK_TORRANCE' if kind == 'GLOSSY' else 'LAMBERT')
        return R.closure_to_surface(cl, NE.ShadeContext(k), st0, mat)[0]
    for kind, what in (('GLOSSY', 'Specular Level 2'), ('DIFFUSE', 'Diffuse Level 1.5')):
        one, both = _nd_slots(three_way([0.3], [0.0], kind), 0), _nd_slots(three_way([0.3, 0.0], [0.0, 0.5], kind), 0)
        check(f'the inner tests are per pixel too: a master + Emission pixel with NO plain {kind} weight is bitwise itself beside a '
              f'{kind} pixel (its {what} x 0.7 was clipped at 1 there)', one == both, ', '.join(k for k in one if one[k] != both.get(k)))
    # the master's ambient level: its own by its share, 1.0 for a plain diffuse / the transparent share, nothing for the rest
    fm = np.array([0.25, 0.5, 0.999], f32)
    hw = f32(1.0) - fm
    kept = (1.0 + (np.full(3, 0.3, f32) - 1.0) * hw).astype(f32)
    for kind in ('DIFFUSE', 'TRANSPARENT', 'TRANSLUCENT', 'HOLDOUT'):
        got = master(kind, fm, hot=False).ambient
        check(f'Mix (Halcyon Shader at Ambient 0.3, {kind}): the ambient level is 1 + (0.3 - 1) x the master\'s share, bit for bit '
              f'(as it always was)', bool(np.all(got == kept)), f'{got} vs {kept}')
    for kind in ('EMISSION', 'GLOSSY', 'GLASS'):
        got = master(kind, fm, hot=False).ambient
        check(f'Mix (Halcyon Shader at Ambient 0.3, {kind}): the plain side answers no ambient -- 0.3 x the master\'s share (it counted '
              f'as a diffuse at 1.0)', bool(np.allclose(got, f32(0.3) * hw, rtol=0, atol=1e-6)), f'{got} vs {f32(0.3) * hw}')
    for kind in ('EMISSION', 'DIFFUSE'):
        madd = _nd_master_closure(None, [0.0, 0.0], hot=False)
        if kind == 'EMISSION':
            madd.add('EMISSION', np.ones(2, f32), color=np.tile(np.array([[0.9, 0.3, 0.1, 1.0]], f32), (2, 1)), strength=np.ones(2, f32))
        else:
            madd.add('DIFFUSE', np.ones(2, f32), color=np.tile(np.array([[0.9, 0.3, 0.1, 1.0]], f32), (2, 1)))
        check(f'Add Shader (Halcyon Shader, {kind}): nothing is diluted and nothing over-counted -- the master\'s 0.3',
              bool(np.allclose(R.closure_to_surface(madd, NE.ShadeContext(2), st0, mat)[0].ambient, 0.3, rtol=0, atol=1e-7)))
    for kind in ('EMISSION', 'TRANSPARENT'):
        s = master(kind, [1.0, 0.0], hot=False)
        check(f'Mix (Halcyon Shader, {kind}) at Fac 1: no master and no diffuse lobe -- level 0, no ambient, a white albedo (the plain '
              f'branch\'s rule; the weighted colour is black there)',
              float(s.diffuse_level[0]) == 0.0 and float(s.ambient[0]) == 0.0 and bool(np.all(s.diffuse[0] == 1.0)),
              f'{s.diffuse_level[0]} {s.ambient[0]} {s.diffuse[0]}')

    # ---- (c) the vote: what cannot blend is named once per material
    def two(fac, rank=None):
        fac = np.asarray(fac, f32)
        n = fac.size
        c4 = lambda c: np.tile(np.array([[c[0], c[1], c[2], 1.0]], f32), (n, 1))  # noqa: E731
        up = np.tile(np.array([[0.0, 0.0, 1.0]], f32), (n, 1))
        side = np.tile(np.array([[1.0, 0.0, 0.0]], f32), (n, 1))
        cl = NE.Closure()
        cl.add('HALCYON', 1.0 - fac, color=c4(_ND_BLUE), model='PHONG', normal=up, anime_extras={'who': 'first'})
        cl.add('HALCYON', fac, color=c4(_ND_GLOW), model='TOON', normal=side, anime_extras={'who': 'second'})
        surf, model, nrm = R.closure_to_surface(cl, NE.ShadeContext(n), st0, mat, rank)
        return model, tuple(np.asarray(nrm)[0]), surf.anime_ramp['who'], R.closure_rank(cl, n)
    first, second = ('PHONG', (0.0, 0.0, 1.0), 'first'), ('TOON', (1.0, 0.0, 0.0), 'second')
    check('with no vote handed in the BATCH votes, as before: the same two masters name Phong in one batch and Toon in another '
          '(a bare call; the frame never makes one)', two([0.0, 0.0, 1.0])[:3] == first and two([0.0, 1.0, 1.0])[:3] == second)
    rank = two([0.0, 0.0, 1.0, 1.0, 1.0])[3]
    check('closure_rank: the mean weight of each master lobe in order, the plain glossy side\'s and the masters\' sum\'s',
          bool(np.allclose(rank['masters'], (0.4, 0.6), atol=1e-6)) and rank['spec'] == 0.0 and abs(rank['hw'] - 1.0) < 1e-6, str(rank))
    for fac in ([0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0], [1.0]):
        check(f'with the material\'s vote (0.4 / 0.6) the batch {fac} takes the model, the normal and the extras of the SECOND master',
              two(fac, rank)[:3] == second and two(fac, lambda: rank)[:3] == second, str(two(fac, rank)[:3]))
    check('a vote for another closure\'s shape (three masters) is not ours: the batch votes',
          two([0.0, 0.0, 1.0], {'masters': (0.1, 0.2, 0.7), 'spec': 0.0, 'hw': 1.0})[:3] == first)
    check('a vote that could not be read (None) leaves the batch voting', two([0.0, 0.0, 1.0], lambda: None)[:3] == first)
    all_gloss = _nd_master_closure('GLOSSY', [0.9, 0.9], hot=False)
    light_gloss = {'masters': (0.7,), 'spec': 0.3, 'hw': 0.7}
    check('...and a batch of glossy pixels does not outvote a material whose vote is the master\'s (alone it names the glossy model)',
          R.closure_to_surface(all_gloss, NE.ShadeContext(2), st0, mat)[1] == 'COOK_TORRANCE'
          and R.closure_to_surface(all_gloss, NE.ShadeContext(2), st0, mat, light_gloss)[1] == 'PHONG')
    gl = _nd_master_closure('GLOSSY', [0.0, 0.0], hot=False)
    heavy_gloss = {'masters': (0.3,), 'spec': 0.7, 'hw': 0.3}
    check('a plain glossy side heavier than the masters BY THE MATERIAL\'S VOTE outvotes the model in a batch that holds no glossy '
          'pixel (the batch\'s own vote kept the master\'s there)',
          R.closure_to_surface(gl, NE.ShadeContext(2), st0, mat)[1] == 'PHONG'
          and R.closure_to_surface(gl, NE.ShadeContext(2), st0, mat, heavy_gloss)[1] == 'COOK_TORRANCE')

    # ---- the frames. The mask: normal z > 0.5 is the SECOND shader -- a quarter of the Ball's surface, its cap
    W, H = 160, 120
    ball_b = _nd_ball(W, H)
    cap = ball_b & (np.abs(_nd_frame(_nd_mix(_nd_node('diffuse'), _nd_node('emission'), thr=0.5), W, H)[..., :3] - glow).max(-1) < 1e-6)
    rows_cap = [int(y) for y in np.nonzero(ball_b.any(1))[0] if bool(np.all(cap[y][ball_b[y]]))]
    check('the cap fills whole rows of the Ball (the batches that voted for themselves)', len(rows_cap) >= 6, str(len(rows_cap)))
    if len(rows_cap) < 6:
        return
    y0, y1 = min(rows_cap), max(rows_cap) + 1
    pair = lambda m1, m2, thr=0.5: _nd_mix(_nd_master('a', m1), _nd_master('b', m2, _ND_GLOW), thr=thr)   # noqa: E731
    full = _nd_frame(pair('PHONG', 'TOON'), W, H)
    band = _nd_frame(pair('PHONG', 'TOON'), W, H, band=(y0, y1))
    check('two masters (Phong, Toon) by a mask: a band wholly on the Toon side is the full frame\'s rows, bit for bit (the band '
          'voted Toon, the frame Phong)', np.array_equal(band, full[y0:y1]),
          str(float(np.abs(band - full[y0:y1]).max())) if band.shape == full[y0:y1].shape else str(band.shape))
    saved = (R.MIN_CHUNK, R.MAX_CHUNK)
    try:
        R.MIN_CHUNK = R.MAX_CHUNK = 96
        chunked = _nd_frame(pair('PHONG', 'TOON'), W, H)
    finally:
        R.MIN_CHUNK, R.MAX_CHUNK = saved
    check('...and in 96-pixel chunks the one-chunk frame, bit for bit', np.array_equal(chunked, full))
    check('the vote is the SURFACE\'s: Phong holds three quarters of it, and the frame is bitwise the one whose second master is '
          'Phong too', np.array_equal(full, _nd_frame(pair('PHONG', 'PHONG'), W, H)))
    check('...with the mask turned (z > -0.5: Toon holds three quarters) it is bitwise the one whose FIRST master is Toon too',
          np.array_equal(_nd_frame(pair('PHONG', 'TOON', -0.5), W, H), _nd_frame(pair('TOON', 'TOON', -0.5), W, H)))
    sc, st = _nd_scene(pair('PHONG', 'TOON'), W, H)
    g, job = _job_for(sc, st)
    vote = job.lobe_rank(sc.materials[1])
    check('ShadeJob.lobe_rank: one vote per material, read over its own triangles by their area (3/4 against 1/4 of the sphere, '
          'within 0.05), and cached',
          vote is not None and len(vote['masters']) == 2 and abs(vote['masters'][0] - 0.75) < 0.05
          and abs(sum(vote['masters']) - 1.0) < 1e-5 and job.lobe_rank(sc.materials[1]) is vote, str(vote))
    check('...None for a graph-less material (nothing to vote on)', job.lobe_rank(sc.materials[0]) is None)
    _g2, job2 = _job_for(sc, st)

    def _no_context(*_a, **_k):
        raise RuntimeError('no context here')
    job2.context = _no_context
    said = io.StringIO()
    with redirect_stdout(said):
        unread = (job2.lobe_rank(sc.materials[1]), job2.lobe_rank(sc.materials[1]))
    check('...and a vote that cannot be read is None, said ONCE by name (each batch then votes for itself, as before)',
          unread == (None, None) and said.getvalue().count("'Wired': the lobe vote could not be read") == 1
          and 'each batch votes for itself' in said.getvalue(), said.getvalue()[-300:])
    ys, xs = np.nonzero(cap)
    pick = np.linspace(0, ys.size - 1, GSH.PROBE_FRAGMENTS).astype(np.int64)
    bake, _mi, pwhy = GSH._probe_material(job, g, 1, ys[pick], xs[pick])
    check('the deferred probe is a batch too: sixteen fragments on the Toon cap bake the MATERIAL\'s model, Phong',
          bake is not None and bake.get('__model') == 'PHONG', str(pwhy if bake is None else bake.get('__model')))
    d, nbad, _img, _cpu, passes, why = _sim_vs_cpu(sc, st)
    check('...and the plan qualifies with the simulator frame the CPU frame (max < 6e-3, no px > 1e-2)',
          passes is not None and d is not None and d < 6e-3 and nbad == 0, str(why if d is None else (d, nbad)))
    hot = lambda: _nd_mix(_nd_master('a', **{'Specular Level': 2.0}), _nd_node('glossy'), thr=0.5)       # noqa: E731
    full_h = _nd_frame(hot(), W, H)
    lo = _nd_frame(hot(), W, H, band=(0, y0))
    check('a master at Specular Level 2 mixed with a Glossy BSDF by a mask: the band UNDER the glossy cap is the full frame\'s '
          'rows, bit for bit', np.array_equal(lo, full_h[:y0]),
          str(float(np.abs(lo - full_h[:y0]).max())) if lo.shape == full_h[:y0].shape else str(lo.shape))
    ball = _nd_ball(128, 96)
    side = ball & (np.abs(_nd_frame(_nd_mix(_nd_node('diffuse'), _nd_node('emission')))[..., :3] - glow).max(-1) < 1e-6)
    for rate in ('PIXEL', 'VERTEX', 'FACE'):
        fr = _nd_frame(_nd_mix(_nd_master('a'), _nd_node('emission')), shading_rate=rate)
        dev = float(np.abs(fr[side][:, :3] - glow).max())
        check(f'a master mixed with an Emission by a mask at the {rate} rate: the emission side shows its colour (1e-5; black at '
              f'the corner rates)', int(side.sum()) > 100 and dev < 1e-5, str(dev))
    em = _nd_frame(_nd_lone('emission'))
    for first, tag in ((_nd_node('diffuse'), 'a Diffuse BSDF'), (_nd_master('a'), 'a Halcyon Shader')):
        near = _nd_frame(_nd_mix(first, _nd_node('emission'), fac=0.999))
        step = float(np.abs(near - em)[ball][:, :3].max())
        check(f'{tag} mixed with an Emission at Fac 0.999 is within 0.01 of the Emission alone on every Ball pixel (the lobe kept its '
              f'whole ambient term: 0.146 in the blue channel)', step < 0.01, str(step))
    for first, tag in ((_nd_node('diffuse'), 'Mix (Diffuse, Glossy, 0.5)'), (_nd_master('a'), 'Mix (Halcyon Shader, Glossy, 0.5)')):
        sc, st = _nd_scene(_nd_mix(first, _nd_node('glossy'), fac=0.5))
        g, job = _job_for(sc, st)
        ys, xs = np.nonzero(ball)
        bake, _mi, pwhy = GSH._probe_material(job, g, 1, ys[::37], xs[::37])
        check(f'{tag}: the probe bakes ambient 0.5, the CPU\'s own share (it baked 1.0)',
              bake is not None and abs(bake.get('ambient') - 0.5) < 1e-6, str(pwhy if bake is None else bake.get('ambient')))


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
