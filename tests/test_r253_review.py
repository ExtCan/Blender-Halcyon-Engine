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
(14) A Coded Shader with a Text assigned ran TWO shaders: the exporter
    compiled the Text for the program (the CPU's truth) and copied the
    raw inline `source_text` -- the starter shader `init` leaves there --
    into the graph, which is what the GPU emitter inlines. The deferred
    pass qualified and drew the starter. The editor and the exporter also
    disagreed about an EMPTY assigned Text. One rule now
    (`shaders/compiler.node_source` / `compile_node`): the assigned Text,
    whatever it holds, else the inline source -- compiled by the editor
    and the exporter through one call, shipped by the exporter as the
    graph's source, and the only source the emitter reads.
(15) Compiled Coded Shader programs were filed under the node's NAME in
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
(16) A Coded Shader whose node NAME holds a letter outside ASCII never
    reached the GPU. The tag was the id with every non-WORD character
    flattened, and Python's `\\W` is Unicode-aware: the 'ñ' of 'Señal',
    a Cyrillic or a CJK name went into the pass's identifiers as written.
    The plan qualified; the device's pre-flight refused the text
    ("unexpected character 'ñ'") and the whole frame shaded on the CPU.
    `gpu/emit._code_spell` spells each such letter or digit as its code
    point ('Seu00F1al', 'u8272u5F69') -- not as one more '_', which makes
    names of underscores alone and the '__' GLSL reserves. `_code_tag`
    tells clashing tags apart as before, and a tag that was plain ASCII
    is the tag it was.
(17) The Coded Shader compiler resolved a name when a lane reached it: an
    identifier nothing declares compiled clean, the node's error box
    stayed empty, the exporter filed the program without a warning, and
    the node read zeros at shading time behind an evaluator error -- or
    shaded, when the name sat in a branch no lane took, and the text went
    to the driver as written. `shaders/names.py` resolves every name at
    compile time (identifiers, struct fields, swizzles), so the error is in
    the box on Compile, in the export's warnings, and no program exists to
    run or to inline.
(18) A Bump node INSIDE a node group had no gradient pre-pass. Both
    pre-pass sites (`core/render._bump_height_fields` for the opaque
    frame, the per-rank twin in `_shade_fragments_cpu` for the layers)
    walked the material's own tree, so a group's Bump -- 'Group/Bump'
    since (15) -- found no field and differenced its own shading batch:
    its gradients were cut at every batch boundary once the material
    outgrew one batch (16384 pixels), at every band edge, and mixed
    across the ranks of a see-through material.
    `core/nodeeval.bump_heights` walks every tree and evaluates a
    group-internal height chain in its group's context
    (`group_evaluator`, the binding `n_group` shades with); the field is
    filed under the node's path id.
(19) The raw text of a deferred material pass declared the shadow atlas
    sampler once per shadowed lamp: `_shadow_function` opened each
    `hal_shadow_vis<i>` with `uniform sampler2D hal_shadowpack;`, and
    under two mapped lamps (the demo scene) a strict front-end refuses
    the second as a redefinition. CreateInfo's text is stripped of every
    declaration, so only the legacy `GPUShader(vertex, fragment)`
    fallback handed a driver that text; Halcyon's front-end takes it, the
    pre-flight counts function bodies, and the strict gate planned
    shadows off. `gpu/material.assemble_frame` declares it once per pass
    (`SHADOWPACK_DECL`), ahead of the first function that fetches.
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
    check('the exporter copies language and as_surface verbatim (the source ships RESOLVED: finding 14)',
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


def _code_floor(graph, programs, size=(128, 96), shadows=True):
    """The demo scene with its floor wearing an exported graph: (scene,
    settings)."""
    from ..core.scene import Material
    st = base_settings(size[0], size[1], shadows=shadows)
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


#: a Bump node's sockets
_BUMP_INS = (('NodeSocketFloat', 'Strength'), ('NodeSocketFloat', 'Distance'), ('NodeSocketFloat', 'Height'),
             ('NodeSocketVector', 'Normal'))


def _code_bump(tree, name='Bump'):
    """A Bump node in `tree` at Strength 1 and Distance 1, nothing linked."""
    node = _code_plain(tree, name, 'ShaderNodeBump', ins=_BUMP_INS, outs=(('NodeSocketVector', 'Normal'),),
                       invert=False)
    node.inputs['Strength'].default_value = node.inputs['Distance'].default_value = 1.0
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


def _code_exported(mat):
    """A stand-in material through the REAL exporter: (graph, programs,
    warnings)."""
    import importlib
    EX = importlib.import_module('halcyon.export')
    images, programs, warnings = {}, {}, []
    graph = EX.serialize_tree(mat.node_tree, images, programs, warnings)
    return graph, programs, warnings


def _code_two(SN, name_a, name_b, src_a=None, src_b=None):
    """Two Coded Shaders in ONE tree, summed into the master: the first
    (red unless `src_a` says otherwise) with its Gain linked to the Value
    node's 0.7, the second (blue) at Gain 0.4."""
    mat = _code_master_tree(SN, _code_text('a', _CODE_RED if src_a is None else src_a))
    tree = mat.node_tree
    a = tree.nodes.get('Coded Shader')
    a.name = name_a
    b = _code_coded(SN, tree, name_b, _code_text('b', _CODE_BLUE if src_b is None else src_b))
    b.inputs['Gain'].default_value = 0.4
    _code_feed(tree, [a, b])
    return mat


def _code_pass(graph, programs, shadows=True, size=(128, 96)):
    """The deferred plan's pass for the demo floor wearing an exported
    graph: (source, binds, why) -- the text the device is handed, or the
    reason the plan gives for refusing the frame."""
    from ..gpu import shade as GSH
    sc, st = _code_floor(graph, programs, size, shadows)
    with redirect_stdout(io.StringIO()):
        g, job = _job_for(sc, st)
        GSH._PLAN_CACHE.clear()
        passes, why, _atlases = GSH.plan_frame(job, g)
    if passes is None:
        return None, None, str(why)
    _mi, _name, src, binds = next(ps for ps in passes if ps[1] == 'Coded')
    return src, binds, None


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
        return _code_two(SN, name_a, name_b, src_a, src_b)

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
    bump = _code_bump

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
        del built[:]
        frame_o = _code_cpu(*as_1920(graph_b, programs_b), big)
        fields_o = list(built)
    finally:
        R._bump_height_fields = real
    check('a Bump in the material and a Bump in a group, named alike, on a frame whose Bump pre-pass runs: each '
          'has a field of its own, the group\'s under its path id',
          fields_b == [['Bump', 'Group/Bump']] and 'Group/Bump' in graph_b['nodes']['Group']['group']['nodes'],
          str(fields_b))
    check('...the group\'s flat Bump bends nothing: the frame of the same trees with the inner Bump renamed, bit '
          'for bit', same(frame_b, frame_n), str(_dmax(frame_b, frame_n)))
    check('...where under 1.92.0\'s bare ids the two share ONE field, the material\'s own, and the group\'s Bump '
          'takes it (1.00 off over the floor)', fields_o == [['Bump']] and _dmax(frame_o, frame_b) > 0.5,
          f'{fields_o} {_dmax(frame_o, frame_b)}')

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


def test_code_node_non_ascii_name_reaches_the_gpu():
    """A Coded Shader's GLSL names are mangled under a tag made from the
    node's graph id, and the tag was the id with every non-WORD character
    flattened to '_' -- Python's word, which is Unicode's: `\\W` keeps the
    'ñ' of 'Señal', every Cyrillic and CJK letter, every digit of every
    script. The letter went into the pass (`_cnSeñal_main`), a GLSL
    identifier is [A-Za-z0-9_] and nothing else, and the device's
    pre-flight refused the text before the driver ("unexpected character
    'ñ'"): the CPU drew the right picture and the frame never reached the
    GPU, for nothing but one node's name. gpu/emit._code_spell spells each
    such letter as its code point ('Seu00F1al', 'u8272u5F69'). Not as one
    more '_': a name of such letters alone would be underscores alone, two
    of them one tag, and GLSL reserves every identifier holding '__'; a
    code point is letters and digits and adds no underscore. Like the
    flattening it is many-to-one (a node NAMED 'Seu00F1al'), and _code_tag
    tells clashing tags apart as it did. A tag that was plain ASCII is the
    tag it was."""
    import ast
    import importlib
    import inspect
    import itertools
    import re
    import textwrap
    from . import fakebpy
    fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    from ..gpu import device as DV
    from ..gpu import emit as GE
    ident = re.compile(r'[A-Za-z_][A-Za-z0-9_]*\Z')        # what GLSL calls an identifier

    def one(name):
        """The master fed by ONE Coded Shader, the red one, named `name`."""
        mat = _code_master_tree(SN, _code_text('a', _CODE_RED))
        mat.node_tree.nodes.get('Coded Shader').name = name
        return mat

    def two(*args):
        return _code_two(SN, *args)

    def twin(mat):
        return _code_twin(*_code_exported(mat)[:2])

    def handed(mat):
        return _code_pass(*_code_exported(mat)[:2])

    def same(a, b):
        return a is not None and b is not None and bool(np.array_equal(a, b))

    def mangled(src):
        """Every name of a text that a coded shader's tag went into."""
        return sorted(set(re.findall(r'\b_cn\w+', src or '')))

    def flattened(key):
        """The whole tag as 1.92.0 spelled it."""
        return re.sub(r'\W', '_', str(key))

    # ---- the references: the same trees under names GLSL could always spell
    d, bad, gpu_one, cpu_one, why = twin(one('Coded Shader'))
    check('one Coded Shader under its default name: the deferred pass twins the CPU (the reference)',
          why is None and bad == 0 and d < 6e-3, f'{why} max {d} bad {bad}')
    d, bad, gpu_two, cpu_two, why = twin(two('Coded Shader', 'Other'))
    check('two Coded Shaders under unrelated ASCII names: the same (the second reference)',
          why is None and bad == 0 and d < 6e-3 and not same(cpu_two, cpu_one), f'{why} max {d} bad {bad}')

    # ---- the reported node: one non-ASCII letter in its name
    d, bad, gpu, cpu, why = twin(one('Señal'))
    check('a Coded Shader named \'Señal\', alone in its material: QUALIFIES for the deferred pass (it was refused: '
          '"would not compile: unexpected character \'ñ\'")', why is None, str(why))
    check('...and its GPU frame twins the CPU frame', d is not None and bad == 0 and d < 6e-3, f'max {d} bad {bad}')
    check('...both of them the frames the same node draws under its default name, bit for bit (a name is not a '
          'picture)', same(cpu, cpu_one) and same(gpu, gpu_one))
    d, bad, gpu, cpu, why = twin(two('Señal', 'Other'))
    check('\'Señal\' beside a second Coded Shader, \'Other\', summed into the master: qualifies and twins',
          why is None and bad == 0 and d < 6e-3, f'{why} max {d} bad {bad}')
    check('...on the frames the same pair draws under ASCII names, bit for bit',
          same(cpu, cpu_two) and same(gpu, gpu_two))

    # ---- no ASCII letter at all; one letter each; a name spelled like the other's tag
    for name_a, name_b, what in (
            ('Шейдер', '色彩', 'Cyrillic beside CJK, not one ASCII letter between them'),
            ('色', '彩', 'one letter each: flattened, both are \'_\''),
            ('Señal', 'Seu00F1al', 'a node NAMED as the other is spelled: one tag between them')):
        d, bad, gpu, cpu, why = twin(two(name_a, name_b))
        check(f'{name_a!r} beside {name_b!r} ({what}): qualifies and twins', why is None and bad == 0 and d < 6e-3,
              f'{why} max {d} bad {bad}')
        check('...on the reference pair\'s frames, bit for bit', same(cpu, cpu_two) and same(gpu, gpu_two))
    # a tag that is another tag plus '_' plus the start of a name: `B_k` under
    # 'Señal' and `k` under 'Señal B' would both be _cnSeu00F1al_B_k
    src_a, src_b = (_code_says('gain * B_k, 0.05, 0.05', 'float B_k = 1.0;\n'),
                    _code_says('0.05, 0.05, gain * k', 'float k = 0.5;\n'))
    d, bad, _gpu, cpu_p, why = twin(two('Señal', 'Señal B', src_a, src_b))
    check('\'Señal\' declaring B_k beside \'Señal B\' declaring k (one GLSL name between them, unless the tags are '
          'told apart): qualifies and twins', why is None and bad == 0 and d < 6e-3, f'{why} max {d} bad {bad}')
    check('...on the frame the same two shaders draw under unrelated names',
          same(cpu_p, _code_cpu(*_code_exported(two('Coded', 'Other', src_a, src_b))[:2])))

    # ---- the text the device is handed
    src, binds, why = handed(two('Señal', 'Other'))
    check('the pass of \'Señal\' beside \'Other\' is plain ASCII, and runs each shader under its own tag',
          src is not None and src.isascii() and '    _cnSeu00F1al_main();' in src and '    _cnOther_main();' in src,
          str(why))
    check('...every name a tag went into is a GLSL identifier (letters, digits, underscores), none holding the '
          'reserved \'__\'', mangled(src) == ['_cnOther_Color', '_cnOther_gain', '_cnOther_main', '_cnSeu00F1al_Color',
                                              '_cnSeu00F1al_gain', '_cnSeu00F1al_main']
          and all(ident.match(n) and '__' not in n for n in mangled(src)), str(mangled(src)))
    declared = [n for key in ('samplers', 'frame_uniforms') for n in (binds or {}).get(key) or ()]
    check('...and so is every sampler and uniform the device declares for the pass',
          all(ident.match(n) for n in declared), str(declared))
    check('...which the device\'s pre-flight -- the wall that refused it, before the driver -- lets through',
          src is not None and DV.PREFLIGHT and DV.preflight('Coded', src) is None,
          str(src is not None and DV.preflight('Coded', src)))
    src_cjk, _binds, why = handed(two('色', '彩'))
    check('\'色\' beside \'彩\': two tags with no ordinal and no underscore of their own (`_cnu8272_main`, '
          '`_cnu5F69_main`)', mangled(src_cjk) == ['_cnu5F69_Color', '_cnu5F69_gain', '_cnu5F69_main', '_cnu8272_Color',
                                                    '_cnu8272_gain', '_cnu8272_main']
          and src_cjk.isascii() and DV.preflight('Coded', src_cjk) is None, f'{why} {mangled(src_cjk)}')

    # ---- the tag as 1.92.0 spelled it: these checks can tell
    real = GE._code_spell
    GE._code_spell = flattened
    try:
        _d, _b, gpu_old, cpu_old, why_old = twin(one('Señal'))
        src_old, _binds, _why = handed(one('Señal'))
    finally:
        GE._code_spell = real
    check('the tag as 1.92.0 spelled it (the flattening alone) put the letter into the pass -- `_cnSeñal_main`, a '
          'text that is not ASCII -- and the plan QUALIFIED it', src_old is not None and not src_old.isascii()
          and '    _cnSeñal_main();' in src_old and not all(ident.match(n) for n in mangled(src_old)),
          str(mangled(src_old)))
    check('...the pre-flight refused that text by name, so no driver saw it and the whole frame shaded on the CPU',
          'would not compile' in str(DV.preflight('Coded', src_old or ''))
          and "unexpected character 'ñ'" in str(DV.preflight('Coded', src_old or '')),
          str(DV.preflight('Coded', src_old or '')))
    check('...as the simulator did: no GPU frame, and the CPU frame the right one all along', gpu_old is None
          and "unexpected character 'ñ'" in str(why_old) and same(cpu_old, cpu_one), str(why_old))

    # ---- the spelling itself
    spelled = {'Señal': 'Seu00F1al', 'Шейдер': 'u0428u0435u0439u0434u0435u0440', '色': 'u8272', '彩': 'u5F69',
               '色彩': 'u8272u5F69', 'Mi señal 2': 'Mi_seu00F1al_2', 'Group/Señal': 'Group_Seu00F1al',
               '\U0001d6fc glow': 'u1D6FC_glow', '層３': 'u5C64uFF13', 'Capa ٣': 'Capa_u0663', 'x²': 'xu00B2',
               '½': 'u00BD'}
    got = {k: GE._code_spell(k) for k in spelled}
    check('a letter or digit GLSL cannot hold (a full-width 3, an Arabic-Indic one, a superscript) is its code '
          'point -- u and four or more hex digits, beyond the BMP included -- and the rest of the name is flattened '
          'as it was', got == spelled, str(got))
    pins = {'code': 'code', 'code2': 'code2', 'x': 'x', 'Coded Shader': 'Coded_Shader',
            'Coded Shader.001': 'Coded_Shader_001', 'Group/Coded Shader': 'Group_Coded_Shader',
            'Group\\/Coded Shader': 'Group__Coded_Shader', '__wang': '__wang'}
    got = {k: GE._code_tag(GE.Emitter({}), k) for k in pins}
    check('the tags test_render and the same-name test pin are where they were', got == pins, str(got))
    alphabet = ('a', 'B', '7', '_', ' ', '.', '/', '\\', '—', 'ñ', '色')
    keys = [''.join(w) for n in (1, 2, 3, 4) for w in itertools.product(alphabet, repeat=n)]
    plain = [k for k in keys if flattened(k).isascii()]
    moved = [k for k in plain if GE._code_spell(k) != flattened(k)]
    check(f'a tag that was plain ASCII is the tag it was: all {len(plain)} of {len(keys)} names up to four characters '
          '(punctuation, both separators, an em dash) that the flattening left no letter in',
          not moved and 0 < len(plain) < len(keys), str(moved[:6]))
    check(f'...and all {len(keys)} now spell a tag of letters, digits and underscores',
          all(re.fullmatch(r'[A-Za-z0-9_]+', GE._code_spell(k)) for k in keys))
    grew = [k for k in keys if re.findall(r'_+', GE._code_spell(k)) != re.findall(r'_+', flattened(k))]
    check('...with exactly the underscores the flattening put there: no run of them is made or lengthened',
          not grew, str(grew[:6]))
    letters = ('a', 'Z', '7', 'ñ', 'Ш', '色', '\U0001d6fc')
    words = [''.join(w) for n in (1, 2, 3) for w in itertools.product(letters, repeat=n)]
    tags = [GE._code_spell(w) for w in words]
    check(f'{len(words)} names of letters and digits alone (four scripts, one beyond the BMP) spell {len(words)} tags '
          'and not one underscore (flattened to ASCII, \'色\' and \'彩\' are one tag: \'_\')',
          len(set(tags)) == len(words) and not any('_' in t for t in tags), f'{len(set(tags))} tags')

    # ---- one tag per node still: what the spelling cannot tell apart, _code_tag does
    em = GE.Emitter({})
    asked = ['Señal', 'Seu00F1al', 'Señal.001', 'Señal B', 'Señal_B', '色', '彩', '色彩', '色 彩',
             'Group/Señal', 'Group Señal']
    tags = [GE._code_tag(em, k) for k in asked]
    check('a node whose spelled id meets an earlier tag takes the ordinal, as a flattened one does; Blender\'s own '
          '.001 never needs it', tags == ['Seu00F1al', '2_Seu00F1al', 'Seu00F1al_001', '3_Seu00F1al_B', '4_Seu00F1al_B',
                                          'u8272', 'u5F69', 'u8272u5F69', '2_u8272_u5F69', 'Group_Seu00F1al',
                                          '2_Group_Seu00F1al'], str(tags))
    check('...one tag per node, asked again or not; no two can spell one GLSL name; every one in the alphabet '
          '_tags_clash is exact over', [GE._code_tag(em, k) for k in asked] == tags and len(set(tags)) == len(asked)
          and not any(GE._tags_clash(a, b) for a, b in itertools.combinations(tags, 2))
          and all(re.fullmatch(r'[A-Za-z0-9_]+', t) for t in tags), str(tags))

    # ---- every name the port hands on, a sampler's included (the device
    # declares that one by name)
    inline, uniforms, ins, outs, samplers = GE._port_code_node(
        'uniform float gain = 1.0;\nuniform sampler2D img;\nin vec3 vPosition;\nout vec4 Color;\n'
        'float lift = 0.25;\nfloat half_of(float x) { return x * 0.5; }\n'
        'void main() { Color = texture(img, vPosition.xy) * half_of(gain) + vec4(lift); }\n', GE._code_spell('色彩'))
    names = [u[2] for u in uniforms] + [i[1] for i in ins] + [o[0] for o in outs.values()] + [m for _n, m in samplers]
    check('a uniform, a varying, an out, a sampler, a global, a function and the entry point under the tag of '
          '\'色彩\': seven ASCII identifiers, in an ASCII text',
          names == ['_cnu8272u5F69_gain', '_cnu8272u5F69_vPosition', '_cnu8272u5F69_Color', '_cnu8272u5F69_img']
          and mangled(inline) == sorted(names + ['_cnu8272u5F69_half_of', '_cnu8272u5F69_lift', '_cnu8272u5F69_main'])
          and inline.isascii() and all(ident.match(n) for n in mangled(inline)), f'{names} {mangled(inline)}')

    # ---- the rule, read from the code
    def calls(f):
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        return {ast.unparse(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}

    check('_code_tag spells through _code_spell and flattens nothing itself; the emitter mangles under _code_tag '
          'alone', '_code_spell' in calls(GE._code_tag) and 're.sub' not in calls(GE._code_tag)
          and '_code_tag' in calls(GE.e_code_node) and not {'re.sub', '_code_spell'} & calls(GE.e_code_node),
          str(sorted(calls(GE._code_tag))))


def _bump_chain(tree, scale=None, normal=None, tag=''):
    """The height chain of the group-Bump probe, in `tree`: Gradient Texture
    (QUADRATIC_SPHERE) -> Math MULTIPLY -> Bump, a dome that bends the whole
    floor. The multiplier is 6.0, or the output socket `scale`; the Bump's
    Normal takes the output socket `normal`. `tag` ends the three names (a
    second chain in one tree). Returns the Bump."""
    ramp = _code_plain(tree, 'Gradient Texture' + tag, 'ShaderNodeTexGradient', ins=(('NodeSocketVector', 'Vector'),),
                       outs=(('NodeSocketColor', 'Color'), ('NodeSocketFloat', 'Fac')),
                       gradient_type='QUADRATIC_SPHERE')
    steep = _code_plain(tree, 'Math' + tag, 'ShaderNodeMath', ins=(('NodeSocketFloat', 'Value'),) * 2,
                        outs=(('NodeSocketFloat', 'Value'),), operation='MULTIPLY', use_clamp=False)
    steep.inputs[1].default_value = 6.0
    bump = _code_bump(tree, 'Bump' + tag)
    tree.links.new(ramp.outputs['Fac'], steep.inputs[0])
    tree.links.new(steep.outputs['Value'], bump.inputs['Height'])
    if scale is not None:
        tree.links.new(scale, steep.inputs[1])
    if normal is not None:
        tree.links.new(normal, bump.inputs['Normal'])
    return bump


#: the sockets a group of the probe takes (`_bump_group_tree` with inputs)
_BUMP_GROUP_INS = (('NodeSocketFloat', 'Scale'), ('NodeSocketVector', 'Normal'))


def _bump_group_tree(inputs=False, reads=True, nested=None):
    """A node group's tree around `_bump_chain`: [Group Input (Scale,
    Normal)] -> chain -> Group Output (Normal). With `inputs` the chain's
    multiplier and its Bump's Normal are the group's inputs (`reads` False:
    the Group Input is there and nothing reads it); given `nested` (another
    such tree) a group node 'Inner' stands where the chain would, handed
    both inputs: a group in a group."""
    tree = types.SimpleNamespace(name='Bump Group', nodes=_CodeNodes(), links=_CodeLinks())
    gout = _code_plain(tree, 'Group Output', 'NodeGroupOutput', ins=(('NodeSocketVector', 'Normal'),),
                       is_active_output=True)
    gin = _code_plain(tree, 'Group Input', 'NodeGroupInput', outs=_BUMP_GROUP_INS) if inputs else None
    if nested is not None:
        mid = _code_plain(tree, 'Inner', 'ShaderNodeGroup', ins=_BUMP_GROUP_INS,
                          outs=(('NodeSocketVector', 'Normal'),), node_tree=nested)
        for name in ('Scale', 'Normal'):
            tree.links.new(gin.outputs[name], mid.inputs[name])
    else:
        fed = inputs and reads
        mid = _bump_chain(tree, gin.outputs['Scale'] if fed else None, gin.outputs['Normal'] if fed else None)
    tree.links.new(mid.outputs['Normal'], gout.inputs['Normal'])
    return tree


def test_group_bump_gathers_from_a_whole_material_field():
    """A Bump node INSIDE a node group had no gradient pre-pass. The
    pre-pass exists because n_bump differences its height toward the +x /
    +y neighbours shaded in the SAME batch: once a material covers more
    than one shading batch (MIN_CHUNK, 16384 pixels) the gradients are cut
    at every batch boundary, so the height chains render to whole-material
    fields first and every batch gathers from those. Both pre-pass sites
    (core/render._bump_height_fields for the opaque frame, the per-rank
    twin in _shade_fragments_cpu for the layers) walked the material's OWN
    tree; since the path ids a group's Bump is 'Group/Bump', found no field
    and differenced its batch. The same chain drew one picture in the
    material's tree and another inside a group, on any frame large enough:
    0.35 apart on two rows at 320x240, 0.17 at 640x480, nothing at 128x96
    where no pre-pass runs. core/nodeeval.bump_heights walks every
    tree now and evaluates a group-internal chain in its group's context --
    group_evaluator, the binding n_group shades with -- and the field is
    filed under the node's path id, so two instances of one group keep two
    fields."""
    import ast
    import importlib
    import inspect
    import textwrap
    from . import fakebpy, fakedevice
    fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    EX = importlib.import_module('halcyon.export')
    from ..core import nodeeval as NE
    from ..gpu import shade as GSH
    big = (320, 240)                          # the floor outgrows one shading batch: the pre-pass runs

    def master():
        """_code_master_tree's material and its master, given the Normal
        socket the probe feeds."""
        mat = _code_master_tree(SN, _code_text('red', _CODE_RED))
        hal = mat.node_tree.nodes.get('Halcyon')
        hal.inputs.new('NodeSocketVector', 'Normal')
        hal.inputs.new('NodeSocketFloat', 'Bump Strength').default_value = 1.0
        return mat, mat.node_tree, hal

    def plain(tree, name, idname, value=None, outs=(('NodeSocketFloat', 'Value'),)):
        node = _code_plain(tree, name, idname, outs=outs)
        if value is not None:
            node.outputs['Value'].default_value = value
        return node

    def feeds(tree, scale, tag=''):
        """(Scale, Normal): a Value node at `scale` and the Geometry node's
        Normal, the two sockets a fed chain or group takes."""
        geo = tree.nodes.get('Geometry') or plain(tree, 'Geometry', 'ShaderNodeNewGeometry', outs=(
            ('NodeSocketVector', 'Position'), ('NodeSocketVector', 'Normal')))
        return plain(tree, 'Scale' + tag, 'ShaderNodeValue', scale).outputs['Value'], geo.outputs['Normal']

    def group(tree, name, node_tree, scale=None, normal=None):
        """A group node holding `node_tree`, its two inputs linked when given."""
        fed = scale is not None
        node = _code_plain(tree, name, 'ShaderNodeGroup', ins=_BUMP_GROUP_INS if fed else (),
                           outs=(('NodeSocketVector', 'Normal'),), node_tree=node_tree)
        if fed:
            tree.links.new(scale, node.inputs['Scale'])
            tree.links.new(normal, node.inputs['Normal'])
        return node

    def built_as(make):
        """A material whose master's Normal takes what `make(tree)` returns."""
        mat, tree, hal = master()
        tree.links.new(make(tree).outputs['Normal'], hal.inputs['Normal'])
        return mat

    # the probe's pair: the chain in the material's own tree, and inside a group
    def top():
        return built_as(_bump_chain)

    def grouped():
        return built_as(lambda tree: group(tree, 'Group', _bump_group_tree()))

    # the group's context: the multiplier and the Normal are the group's inputs
    def top_fed(scale=3.5):
        return built_as(lambda tree: _bump_chain(tree, *feeds(tree, scale)))

    def grouped_fed(scale=3.5, reads=True, nested=False):
        def make(tree):
            inner = _bump_group_tree(inputs=True, reads=reads)
            if nested:
                return group(tree, 'Outer', _bump_group_tree(inputs=True, nested=inner), *feeds(tree, scale))
            return group(tree, 'Group', inner, *feeds(tree, scale))
        return built_as(make)

    # two chains, the second bending the first's normal: at the top, and as ONE group tree instanced twice
    def top_two():
        def make(tree):
            first = _bump_chain(tree, *feeds(tree, 6.0))
            return _bump_chain(tree, feeds(tree, 2.0, '.001')[0], first.outputs['Normal'], tag='.001')
        return built_as(make)

    def grouped_two():
        def make(tree):
            inner = _bump_group_tree(inputs=True)
            first = group(tree, 'Group', inner, *feeds(tree, 6.0))
            return group(tree, 'Group.001', inner, feeds(tree, 2.0, '.001')[0], first.outputs['Normal'])
        return built_as(make)

    def frame(mat, size=big, glass=False, band=None, device='CPU'):
        """(frame, printed) of the demo floor wearing `mat`'s export. `glass`
        sets the master's Opacity to 0.6 and flags the material see-through,
        as the exporter flags such a master: its fragments leave the opaque
        pass for the layered road."""
        if glass:
            mat.node_tree.nodes.get('Halcyon').inputs['Opacity'].default_value = 0.6
        images, programs, warnings = {}, {}, []
        graph = EX.serialize_tree(mat.node_tree, images, programs, warnings)
        sc, st = _code_floor(graph, programs, size)
        sc.materials[0].has_alpha = glass
        st.render_device = device
        said = io.StringIO()
        with redirect_stdout(said):
            img = np.asarray(R.render(sc, st, band=band))
        return img, said.getvalue()

    def same(a, b):
        return bool(np.array_equal(a, b))

    def rows_off(a, b):
        """(max difference, the rows holding one)."""
        d = np.abs(a.astype(np.float64) - b.astype(np.float64)).max(axis=-1)
        return float(d.max()), sorted(set(np.nonzero(d > 0)[0].tolist()))

    # ---- the spies: every field the pre-pass builds, what it evaluates on the way, the layered road
    built, value_reads, layered, gathered = [], [], [], []
    real_fields, real_layers, real_chunked = R._bump_height_fields, R._shade_fragments_cpu, R._shade_chunked
    real_value = NE.DISPATCH['ShaderNodeValue']
    state = {'prepass': False, 'own_tree_only': False}

    def spy_fields(*args, **kw):
        state['prepass'] = True
        try:
            out = real_fields(*args, **kw)
        finally:
            state['prepass'] = False
        if state['own_tree_only']:            # the 1.92.0 pre-pass: no field for a node below the material's tree
            out = {key: field for key, field in out.items() if EX.ID_SEP not in key[1]}
        built.append(out)
        return out

    def spy_value(ev, node):
        if state['prepass']:
            value_reads.append(node.get('id'))
        return real_value(ev, node)

    def spy_layers(job, tri, *args, **kw):
        layered.append(int(tri.size))
        return real_layers(job, tri, *args, **kw)

    def spy_chunked(job, *args, **kw):
        gathered.append(sorted(key[1] for key in job.bump_fields))
        return real_chunked(job, *args, **kw)

    def keys():
        """The node ids of each pre-pass run since the last reset."""
        return [sorted(key[1] for key in out) for out in built]

    def reset():
        for log in (built, value_reads, layered, gathered):
            del log[:]

    def shot(mat, **kw):
        """frame()'s picture, with the spies' logs of that frame alone."""
        reset()
        return frame(mat, **kw)[0]

    R._bump_height_fields, R._shade_fragments_cpu, R._shade_chunked = spy_fields, spy_layers, spy_chunked
    NE.DISPATCH['ShaderNodeValue'] = spy_value
    keep_chunks = R.MIN_CHUNK, R.MAX_CHUNK
    try:
        # ---- the reported pair, on a frame whose pre-pass runs
        frame_top = shot(top())
        check('the probe\'s chain in the material\'s own tree at 320x240: the pre-pass runs and builds its Bump\'s '
              'field (the reference)', keys() == [['Bump']], str(keys()))
        frame_grp = shot(grouped())
        check('the same chain inside a group: the pre-pass RUNS for it too, the field filed under the node\'s path '
              'id (it built nothing, and the group\'s Bump differenced its own batch)', keys() == [['Group/Bump']],
              str(keys()))
        check('...and the frame is the top-level frame, bit for bit (0.35 off on two rows)', same(frame_grp, frame_top),
              str(rows_off(frame_grp, frame_top)))
        R.MIN_CHUNK = R.MAX_CHUNK = 1 << 30
        whole = shot(top())
        whole_keys = keys()
        R.MIN_CHUNK, R.MAX_CHUNK = keep_chunks
        check('both are the frame of the material shaded in ONE batch -- no pre-pass at all, n_bump differencing '
              'the whole floor itself', whole_keys == [] and same(whole, frame_top) and same(whole, frame_grp),
              f'{whole_keys} {rows_off(whole, frame_grp)}')
        state['own_tree_only'] = True
        cut = shot(grouped())
        state['own_tree_only'] = False
        off, rows = rows_off(cut, frame_top)
        check('these frames can tell: with the group\'s field withheld (the 1.92.0 pre-pass, the material\'s own '
              'tree only) the group frame is cut along the batch boundary -- a row or two, more than 0.1 off',
              keys() == [[]] and off > 0.1 and 1 <= len(rows) <= 4, f'{keys()} {off} {rows}')
        small_top, small_grp = shot(top(), size=(128, 96)), shot(grouped(), size=(128, 96))
        check('at 128x96 the floor fits one batch: no pre-pass runs, and the two frames agree as they always did',
              keys() == [] and same(small_grp, small_top), f'{keys()} {rows_off(small_grp, small_top)}')
        large_top, large_grp = shot(top(), size=(640, 480)), shot(grouped(), size=(640, 480))
        check('at 640x480 (the floor in several batches, 0.17 off): bit for bit',
              keys() == [['Group/Bump']] and same(large_grp, large_top), f'{keys()} {rows_off(large_grp, large_top)}')

        # ---- the group's context: a height that reads the group's inputs
        fed_top = shot(top_fed())
        fed_grp = shot(grouped_fed())
        fed_reads = list(value_reads)
        check('a chain that reads its group\'s inputs (Scale from a Value node, Normal from Geometry): the field is '
              'evaluated with them bound, as n_group binds them -- the frame of the same chain at the top level',
              keys() == [['Group/Bump']] and same(fed_grp, fed_top), f'{keys()} {rows_off(fed_grp, fed_top)}')
        check('...which is another picture than the 6.0 chain\'s: the height really is the group input\'s',
              rows_off(fed_top, frame_top)[0] > 0.05, str(rows_off(fed_top, frame_top)[0]))
        unread = shot(grouped_fed(reads=False))
        check('a chain that reads NO group input: the pre-pass leaves what feeds the group alone (the Value at its '
              'Scale is not evaluated there), where the fed chain\'s pre-pass binds it',
              'Scale' not in value_reads and 'Scale' in fed_reads and keys() == [['Group/Bump']]
              and same(unread, frame_top), f'{value_reads} {fed_reads} {rows_off(unread, frame_top)}')
        two_top = shot(top_two())
        two_grp = shot(grouped_two())
        fields = built[-1] if built else {}
        gx_a, gx_b = (fields.get((0, nid), (np.zeros(1),))[0] for nid in ('Group/Bump', 'Group.001/Bump'))
        check('ONE group instanced twice at two scales: two fields, each under its own instance\'s path',
              keys() == [['Group.001/Bump', 'Group/Bump']] and gx_a.shape == gx_b.shape and bool(gx_b.any())
              and float(np.abs(gx_a - 3.0 * gx_b).max()) < 1e-5 and float(np.abs(gx_a - gx_b).max()) > 1e-3,
              f'{keys()} {float(np.abs(gx_a).max())} {float(np.abs(gx_b).max())}')
        check('...and the frame is the frame of the two chains at the top level, bit for bit',
              same(two_grp, two_top), str(rows_off(two_grp, two_top)))
        nested = shot(grouped_fed(nested=True))
        check('a group in a group, the Scale handed down two levels: the field is filed under the whole path and '
              'the frame is the top-level chain\'s, bit for bit',
              keys() == [['Outer/Inner/Bump']] and same(nested, fed_top), f'{keys()} {rows_off(nested, fed_top)}')

        # ---- the layered road: the same floor, see-through
        glass_top = shot(top(), glass=True)
        check('the floor flagged see-through leaves the opaque pass for the layered road, whose per-rank fields '
              'the top-level Bump gathers from (the reference)',
              len(layered) == 1 and keys() == [] and ['Bump'] in gathered and rows_off(glass_top, frame_top)[0] > 0.1,
              f'{layered} {keys()} {gathered}')
        glass_grp = shot(grouped(), glass=True)
        check('the group\'s Bump on the layered road: a field per rank under its path id, and the top-level frame '
              'bit for bit (no field at all: one mixed batch per chunk, 0.21 off)',
              len(layered) == 1 and ['Group/Bump'] in gathered and same(glass_grp, glass_top),
              f'{layered} {gathered} {rows_off(glass_grp, glass_top)}')
        R.MAX_CHUNK = 20000                   # that road evaluates its heights MAX_CHUNK fragments at a time
        glass_two = shot(grouped(), glass=True)
        R.MIN_CHUNK, R.MAX_CHUNK = keep_chunks
        check('...and with the layer\'s heights evaluated in two chunks (MAX_CHUNK lowered under the floor\'s '
              'fragment count): the same frame, bit for bit',
              layered and 20000 < layered[0] <= 40000 and same(glass_two, glass_grp),
              f'{layered} {rows_off(glass_two, glass_grp)}')

        # ---- bands: the fields come from the G-buffer's full coverage (job.bump_field_source)
        reset()
        height = big[1]
        bands = [(0, height // 3), (height // 3, height // 2 + 7), (height // 2 + 7, height)]
        joined = np.concatenate([frame(grouped(), band=band)[0] for band in bands], axis=0)
        check('three bands of the group frame, joined: each band builds the group Bump\'s whole-frame field, and '
              'the rows are the whole frame\'s, bit for bit (0.53 off)',
              keys() == [['Group/Bump']] * 3 and same(joined, frame_grp), f'{keys()} {rows_off(joined, frame_grp)}')

        # ---- the GPU device: a group has no emitter, so the CPU shades the frame -- pre-passed
        reset()
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        with fakedevice.installed():
            on_gpu, said = frame(grouped(), device='GPU')
        R._GBUF_CACHE.clear()
        GSH._PLAN_CACHE.clear()
        check('on the GPU device the material is refused by name (a node group has no emitter) and the CPU shades '
              'the frame: the same pre-passed frame, bit for bit',
              'no GLSL emitter for ShaderNodeGroup' in said and keys() == [['Group/Bump']] and same(on_gpu, frame_grp),
              f'{keys()} {rows_off(on_gpu, frame_grp)} {said[-300:]}')
    finally:
        R._bump_height_fields, R._shade_fragments_cpu, R._shade_chunked = real_fields, real_layers, real_chunked
        NE.DISPATCH['ShaderNodeValue'] = real_value
        R.MIN_CHUNK, R.MAX_CHUNK = keep_chunks

    # ---- the rule, read from the code itself
    def calls(f):
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        return {ast.unparse(c.func) for c in ast.walk(tree) if isinstance(c, ast.Call)}

    check('both pre-pass sites take their heights from ONE walk (bump_heights), and the layered road finds its '
          'Bump materials at any depth (has_bump)',
          'bump_heights' in calls(R._bump_height_fields) and 'bump_heights' in calls(R._shade_fragments_cpu)
          and 'has_bump' in calls(R._bump_height_fields) and 'has_bump' in calls(R._bump_materials_of),
          str(sorted(calls(R._bump_materials_of))))
    check('the shading pass and the pre-pass bind a group through ONE function (group_evaluator), the only place '
          'a group\'s inputs are bound',
          'group_evaluator' in calls(NE.n_group) and 'group_evaluator' in calls(NE.bump_heights)
          and inspect.getsource(NE).count('.group_inputs = ') == 1
          and '.group_inputs = ' in inspect.getsource(NE.group_evaluator))


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
    check('the wired colour is ADDED over the lit fallback surface: frame(colour) - frame(black) is the colour at every Ball pixel (1e-6)',
          gain < 1e-6, str(gain))
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


#: the reported source: `nothing` is declared nowhere
_CODE_UNDECLARED = """\
uniform float gain = 1.0;

out vec4 Color;

void main() {
    Color = vec4(nothing, 0.05, gain, 1.0);
}
"""


class _Layout:
    """node.draw_buttons' layout: every label it draws, in order."""

    def __init__(self, seen=None):
        self.seen = [] if seen is None else seen
        self.alert = False

    def row(self, **_kw):
        return _Layout(self.seen)

    box = column = row

    def prop(self, *_a, **_kw):
        pass

    def operator(self, *_a, **_kw):
        return types.SimpleNamespace()

    def label(self, text='', icon='NONE'):
        self.seen.append((text, icon))


def test_code_node_name_errors_are_compile_errors():
    """An identifier nothing declares was accepted by the Coded Shader
    compiler and failed only when the program ran: the node's error box
    stayed empty on Compile, the exporter filed the program and warned of
    nothing, and at shading time the evaluator caught the exception and the
    node's outputs read zeros. On the GPU road the plan's probe refused the
    whole frame when a probed lane reached the name -- and qualified it
    when none did, handing the driver the text as written. The name is a
    compile error now (shaders/names.py): in the box, in the export's
    warnings, and with no program there is nothing to run and nothing to
    inline."""
    import importlib
    from . import fakebpy
    fakebpy.install()
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    from .. import export as EX
    from ..core import raster
    from ..core.nodeeval import GraphEvaluator
    from ..gpu import device as DV
    from ..gpu import shade as GSH
    from ..shaders.compiler import try_compile
    from .scenebuild import add_coded_floor
    from .test_render import _ctx

    want = "line 6: undeclared identifier 'nothing'"
    good = types.SimpleNamespace(name='good', as_string=lambda: _CODE_A)
    bad = types.SimpleNamespace(name='bad', as_string=lambda: _CODE_UNDECLARED)
    mended = types.SimpleNamespace(name='mended',
                                   as_string=lambda: _CODE_UNDECLARED.replace('nothing', 'gain * 0.5'))

    def names(socks):
        return [s.name for s in socks]

    def links(n):
        return ([(l.from_node.name, l.from_socket.name) for l in n.inputs['Gain'].links],
                [(l.to_node.name, l.to_socket.name) for l in n.outputs['Color'].links])

    def boxed(n):
        lay = _Layout()
        n.draw_buttons(None, lay)
        return [text for text, icon in lay.seen if icon == 'ERROR']

    # ---- the node: Compile puts the error in the box and keeps the sockets
    tree = _code_tree(SN, good).node_tree
    n = tree.nodes.get('Coded Shader')
    wired = links(n)
    check('the node starts whole: no error, the good source\'s sockets, both links',
          n.error == '' and boxed(n) == [] and names(n.inputs) == ['Gain', 'Tint']
          and wired == ([('Value', 'Value')], [('Material Output', 'Surface')]), f'{n.error!r} {names(n.inputs)}')
    n.source = bad
    said = []
    op = SN.HALCYON_OT_compile_shader()
    op.node, op.report = n.name, (lambda kind, text: said.append((kind, text)))
    ctx = types.SimpleNamespace(space_data=types.SimpleNamespace(edit_tree=tree), active_node=None)
    got = op.execute(ctx)
    check('Compile on a source with an undeclared identifier is cancelled and reports it by name and line',
          got == {'CANCELLED'} and said == [({'ERROR'}, want)], f'{got} {said}')
    check('the node\'s error box shows that line', n.error == want and boxed(n) == [want], f'{n.error!r} {boxed(n)}')
    check('...and the sockets are left as they were, links and all (no program, nothing to rebuild from)',
          names(n.inputs) == ['Gain', 'Tint'] and names(n.outputs) == ['Color'] and links(n) == wired,
          f'{names(n.inputs)} {links(n)}')
    check('the rebuild timer\'s road (compile_source) answers the same', n.compile_source() is None and n.error == want)

    # ---- the export: no program is filed, the warning names node, name and line
    twin = _code_coded(SN, tree, 'Good Shader', good)
    programs, warnings = {}, []
    graph = EX.serialize_tree(tree, {}, programs, warnings)
    check('serialize_tree files the good node\'s program and none for the node that does not compile',
          sorted(programs) == ['Good Shader'] and twin.error == '', str(sorted(programs)))
    check('...and the export warns, naming the node, the identifier and the line',
          warnings == [f'Coded Shader: {want}'], str(warnings))

    # ---- shading time: the node reads zeros with NOTHING for the evaluator to catch
    ev = GraphEvaluator(graph, _ctx(), programs=programs)
    col = ev.eval_output('Coded Shader', 0)
    check('at shading time the node reads zeros and the evaluator records no error (it used to catch '
          'ShaderError: unknown identifier per batch)',
          col is not None and not np.any(col) and not ev.unsupported and not ev.errors,
          f'{sorted(ev.unsupported)} {ev.errors}')

    # ---- mending the source clears the box and rebuilds
    n.source = mended
    said[:] = []
    got = op.execute(ctx)
    check('the mended source compiles: the box is empty and the sockets follow the new declarations',
          got == {'FINISHED'} and n.error == '' and boxed(n) == [] and names(n.inputs) == ['Gain']
          and links(n) == wired, f'{got} {n.error!r} {names(n.inputs)} {links(n)}')

    # ---- the GPU road. A name behind a branch the sixteen probe points miss used to qualify the frame
    # and ride into the pass as written (here `vUV`, which the frame shader itself declares: a driver
    # compiles that text and draws the full-screen quad's UV where the CPU zeroes the node)
    strip = ('in vec2 vScreenUV;\nout vec4 Color;\nvoid main() {\n    Color = vec4(0.2, 0.6, 0.9, 1.0);\n'
             '    if (vScreenUV.x > 0.5 && vScreenUV.x < 0.53) { Color = vec4(vUV, 0.0, 1.0); }\n}\n')
    prog, err = try_compile(strip)
    check('a varying read without its declaration, behind a branch, is a compile error that says what to write',
          prog is None and err == "line 5: undeclared identifier 'vUV' (declare it: in vec2 vUV;)", repr(err))
    w, h = 96, 72
    st = base_settings(w, h, shadows=True)
    st.color_depth, st.dither, st.output_scale = '24', 'NONE', 'NONE'
    sc = add_coded_floor(demo_scene(st, with_texture=False))
    code = sc.materials[0].graph['nodes']['code']
    code['props']['source_text'], code['inputs'] = strip, []
    sc.materials[0].programs = {}                # what the exporter files for a source that does not compile
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('the frame still qualifies for the GPU: the node is zeros there, as on the CPU', passes is not None, str(why))
    if passes is not None:
        texts = [x for ps in passes for x in ps if isinstance(x, str) and 'void main' in x]
        check('no pass carries the coded text (the unresolved name never reaches a driver)',
              bool(texts) and not any('vScreenUV' in t or '_cncode_' in t for t in texts))
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        d = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('...and the simulated GPU frame is the CPU frame', d < 6e-3, f'max {d:.5f}')
    GSH._PLAN_CACHE.clear()

    # ---- the pre-flight every dynamic pass crosses is the same front-end: a pass text that names
    # something undeclared in a branch is refused before the driver, with the name and its line
    stray = 'in vec2 vUV;\nout vec4 fragColor;\nvoid main() {\n    fragColor = vec4(vUV, 0.0, 1.0);\n' \
            '    if (vUV.x > 2.0) { fragColor = vec4(hal_missing); }\n}\n'
    pre = DV.preflight('Stray', stray)
    check('device.preflight refuses a pass that names something undeclared',
          pre is not None and "line 5: undeclared identifier 'hal_missing'" in pre and 'refused before the driver' in pre,
          str(pre))
    check('...and lets the same pass through once the name is declared',
          DV.preflight('Stray', 'uniform float hal_missing;\n' + stray) is None)


#: the shadow atlas sampler's declaration, as a line of a pass
_SHADOWPACK_LINE = 'uniform sampler2D hal_shadowpack;'


def _shadow_plan(edit=None, size=(48, 36), **kw):
    """The demo scene's deferred plan with shadows ON -- its SUN and its
    POINT both cast, both MAP unless `edit(scene)` or the settings say
    otherwise: (passes, why, job, gbuf, atlases, scene, settings). The BVH
    rides on the job by render()'s own rule for ray shadows, so a RAY
    lamp plans its traversal."""
    from ..gpu import shade as GSH
    st = base_settings(size[0], size[1], **dict({'shadows': True}, **kw))
    st.color_depth, st.dither, st.output_scale = '24', 'NONE', 'NONE'
    sc = demo_scene(st, with_texture=False)
    if edit is not None:
        edit(sc)
    traced = st.shadows and (st.shadow_default == 'RAY' or R._per_light_ray_shadows(sc, st))
    with redirect_stdout(io.StringIO()):
        g, job = _job_for(sc, st, bvh=R._cached_bvh(sc, sc.mesh) if traced else None)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
    GSH._PLAN_CACHE.clear()
    return passes, why, job, g, atlases, sc, st


def _shadow_fns(src, ray=False):
    """The lamps whose shadow function a pass DEFINES: the mapped ones
    (three arguments, the atlas fetch), or with `ray` the traced ones
    (a fourth, the distance)."""
    import re
    tail = r', float dist\)' if ray else r'\)'
    return [int(i) for i in re.findall(r'^float hal_shadow_vis(\d+)\(vec3 P, vec3 N, vec3 L' + tail + r'\s*$', src, re.M)]


def _declared_twice(src):
    """The names a text declares as a global `uniform` more than once --
    what a strict GLSL front-end calls a redefinition (the text-level
    stand-in for glslang, which the suite does not have). Whole lines,
    comments aside, any number of declarations to a line: the same lines
    `device.strip_declarations` reads."""
    import re
    names = []
    for line in str(src).splitlines():
        names += re.findall(r'(?:^|;)\s*uniform\s+\w+\s+(\w+)\s*(?=;)', line.split('//', 1)[0])
    return sorted({n for n in names if names.count(n) > 1})


def _as_1_92_declared(src):
    """A pass as 1.92.0 wrote it: no declaration ahead of the shadow
    functions, one opening EACH mapped lamp's own."""
    import re
    body = src.replace(_SHADOWPACK_LINE + '\n\n', '', 1)
    return re.sub(r'^(float hal_shadow_vis\d+\(vec3 P, vec3 N, vec3 L\))\s*$', _SHADOWPACK_LINE + r'\n\1', body, flags=re.M)


def _driver_handover(src, binds, createinfo):
    """What `gpu/device._compile_dynamic_miss` hands a driver for a
    material pass, over a stand-in `gpu` module: (error, fragment text,
    [(slot, sampler)]). With `createinfo` the module has
    GPUShaderCreateInfo (every current Blender); without, the legacy
    `GPUShader(vertex, fragment)` constructor is the only road left. The
    spec is the one `gpu/shade` builds for a material pass."""
    from ..gpu import device as DV
    got = {'fragment': None, 'samplers': []}

    class _Info:
        def vertex_in(self, *_a):
            pass

        vertex_out = fragment_out = push_constant = vertex_source = vertex_in

        def sampler(self, slot, _kind, name):
            got['samplers'].append((slot, name))

        def fragment_source(self, text):
            got['fragment'] = text

    def _legacy(_vertex, fragment):
        got['fragment'] = fragment
        return object()

    gpu = types.ModuleType('gpu')
    gpu.types = types.SimpleNamespace(GPUShader=_legacy)
    gpu.shader = types.SimpleNamespace(create_from_info=lambda _info: object())
    if createinfo:
        gpu.types.GPUShaderCreateInfo = _Info
        gpu.types.GPUStageInterfaceInfo = lambda _name: types.SimpleNamespace(smooth=lambda *_a: None)
    spec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris'] + list(binds.get('samplers', ())),
            'floats': ['hal_attr_side', 'hal_slot_count', 'hal_tri_side'] + list(binds.get('frame_uniforms', ())),
            'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
    key = ('HAL_SHADOWPACK_HANDOVER', hash(src))
    absent = object()
    saved_mod = sys.modules.get('gpu', absent)
    saved = {k: DV._STATE[k] for k in ('checked', 'available', 'reason')}
    try:
        sys.modules['gpu'] = gpu
        DV._STATE.update(checked=True, available=True, reason='a stand-in driver')
        _shader, err = DV._compile_dynamic_miss(key, 'HAL_TMAT', src, spec)
    finally:
        DV._STATE.update(saved)
        DV._STATE['shaders'].pop(key, None)
        if saved_mod is absent:
            sys.modules.pop('gpu', None)
        else:
            sys.modules['gpu'] = saved_mod
    return err, got['fragment'], got['samplers']


def test_shadow_atlas_sampler_is_declared_once_per_pass():
    """Every mapped lamp reads ONE combined shadow atlas, `hal_shadowpack`,
    and `_shadow_function` opened each lamp's `hal_shadow_vis<i>` with its
    own `uniform sampler2D hal_shadowpack;` -- so the raw text of a pass
    under two shadowed lamps (the demo scene: a SUN and a POINT, both
    MAP) declared the sampler twice, and a strict front-end refuses the
    second as a redefinition (glslang 15.1: `'hal_shadowpack' :
    redefinition`). `strip_declarations` takes every such line off on the
    CreateInfo road, so current Blender builds never saw it; the legacy
    `GPUShader(vertex, fragment)` fallback hands the driver the raw text.
    Halcyon's own front-end accepts the redeclaration, the pre-flight
    counts function bodies only, and the strict gate planned every scene
    with shadows off. The assembler declares it once now
    (`gpu/material.SHADOWPACK_DECL`), ahead of the first function that
    fetches from it, however many lamps cast."""
    import ast
    import inspect
    import re
    import textwrap
    from ..core.scene import Light
    from ..gpu import device as DV
    from ..gpu import material as GM
    from ..gpu import shade as GSH
    from ..shaders.compiler import try_compile

    def spot(k):
        return Light(type='SPOT', name=f'Spot {k}', position=(0.5 - k, -3.0, 5.0), direction=(-0.1, 0.5, -0.85),
                     color=(1.0, 0.9, 0.8), energy=300.0, shadow='MAP', spot_size=1.0)

    def more(n):
        def edit(sc):
            sc.lights += [spot(k) for k in range(n)]
        return edit

    def point_traced(sc):
        sc.lights[1].shadow = 'RAY'

    def sun_traced(sc):
        sc.lights[0].shadow = 'RAY'

    def texts(passes):
        return [(name, src, binds) for _mi, name, src, binds in passes]

    def count(src):
        return len(re.findall(r'^\s*uniform\s+sampler2D\s+hal_shadowpack\s*;', src, re.M))

    decl = getattr(GM, 'SHADOWPACK_DECL', None)
    check('the assembler has the declaration to write, and it is that one line (`gpu/material.SHADOWPACK_DECL`)',
          decl == _SHADOWPACK_LINE + '\n', repr(decl))

    # ---- the report: the demo scene, shadows on -- two mapped lamps
    passes, why, job, g, atlases, sc, st = _shadow_plan()
    check('the demo scene with shadows on qualifies for the deferred road', passes is not None, str(why))
    if passes is None:
        return
    demo = texts(passes)
    check('it plans its three material passes, each defining a mapped shadow function for BOTH lamps (the SUN and the POINT)',
          [n for n, _s, _b in demo] == ['Floor', 'Ball', 'Box'] and all(_shadow_fns(s) == [0, 1] for _n, s, _b in demo),
          str([(n, _shadow_fns(s)) for n, s, _b in demo]))
    check('every pass declares the atlas sampler exactly ONCE (it was once per shadowed lamp: twice here)',
          [count(s) for _n, s, _b in demo] == [1, 1, 1], str([count(s) for _n, s, _b in demo]))
    check('...as the line `uniform sampler2D hal_shadowpack;`, alone on its line',
          all(s.splitlines().count(_SHADOWPACK_LINE) == 1 for _n, s, _b in demo))
    check('...ahead of the first function that fetches from it (declaration before use, in file order)',
          all(0 < s.find(_SHADOWPACK_LINE) < s.find('float hal_shadow_vis0(') < s.find('texelFetch(hal_shadowpack')
              for _n, s, _b in demo))
    check('...and both lamps\' functions still fetch from it',
          all(s.count('texelFetch(hal_shadowpack') >= 2 for _n, s, _b in demo))
    check('the sampler is listed once in each pass\'s binds, as before',
          all(list(b['samplers']).count('hal_shadowpack') == 1 for _n, _s, b in demo),
          str([list(b['samplers']) for _n, _s, b in demo]))
    check('no pass declares ANY uniform twice (the text-level reading of a strict front-end\'s "redefinition")',
          all(_declared_twice(s) == [] for _n, s, _b in demo), str([_declared_twice(s) for _n, s, _b in demo]))
    check('the frame still uploads the one packed atlas under that name',
          sorted(k for k in atlases if k.startswith('hal_shadow')) == ['hal_shadowpack'], str(sorted(atlases)))

    # ---- however many lamps cast, and whatever else shadows them
    for label, edit, kw, mapped, traced in (
            ('three mapped lamps', more(1), {}, [0, 1, 2], []),
            ('eight mapped lamps', more(6), {}, list(range(8)), []),
            ('one mapped lamp beside a traced one (Per Light)', point_traced, {'shadow_default': 'PER_LIGHT'}, [0], [1]),
            ('a traced lamp ahead of the one mapped lamp (Per Light)', sun_traced, {'shadow_default': 'PER_LIGHT'}, [1], [0]),
            ('two traced lamps (Ray)', None, {'shadow_default': 'RAY'}, [], [0, 1])):
        p2, why2 = _shadow_plan(edit, size=(24, 18), **kw)[:2]
        check(f'{label}: the plan qualifies', p2 is not None, str(why2))
        if p2 is None:
            continue
        t2 = texts(p2)
        check(f'{label}: every pass defines exactly those shadow functions',
              all(_shadow_fns(s) == mapped and _shadow_fns(s, ray=True) == traced for _n, s, _b in t2),
              str([(_shadow_fns(s), _shadow_fns(s, ray=True)) for _n, s, _b in t2]))
        want = 1 if mapped else 0
        check(f'{label}: the sampler is declared {"once" if want else "nowhere"} and listed {"once" if want else "nowhere"}',
              all(count(s) == want and list(b['samplers']).count('hal_shadowpack') == want for _n, s, b in t2),
              str([(count(s), list(b['samplers']).count('hal_shadowpack')) for _n, s, b in t2]))
        if mapped:
            check(f'{label}: ...ahead of the first fetch, whichever lamp makes it',
                  all(0 < s.find(_SHADOWPACK_LINE) < s.find('texelFetch(hal_shadowpack') for _n, s, _b in t2))
        check(f'{label}: no uniform is declared twice', all(_declared_twice(s) == [] for _n, s, _b in t2),
              str([_declared_twice(s) for _n, s, _b in t2]))
        if not mapped:
            check(f'{label}: the text never names the atlas', all('hal_shadowpack' not in s for _n, s, _b in t2))
    for label, kw in (('at the vertex rate (the corners carry the shadowed light)', {'shading_rate': 'VERTEX'}),
                      ('with shadows off', {'shadows': False})):
        p3, why3 = _shadow_plan(size=(24, 18), **kw)[:2]
        check(f'{label}: the plan qualifies and no pass names the atlas, in its text or in its binds',
              p3 is not None and all('hal_shadowpack' not in s and 'hal_shadowpack' not in b['samplers']
                                     and _shadow_fns(s) == [] for _n, s, b in texts(p3)), str(why3))

    # ---- the 1.92.0 text put back, so the checks can tell -- and so the picture is known not to move
    old = [(n, _as_1_92_declared(s), b) for n, s, b in demo]
    check('the 1.92.0 text put back declares it once per lamp, and the text-level check names it',
          [count(s) for _n, s, _b in old] == [2, 2, 2] and all(_declared_twice(s) == ['hal_shadowpack'] for _n, s, _b in old),
          str([(count(s), _declared_twice(s)) for _n, s, _b in old]))
    check('...which the pre-flight every pass crosses lets through (function bodies are all it counts; Halcyon\'s own '
          'front-end takes a redeclared uniform) -- why no headless check saw it',
          all(DV.duplicate_definitions(s) == [] and DV.preflight(n, s) is None for n, s, _b in old))
    lines = [[ln for ln in DV.strip_declarations(s).splitlines() if ln.strip()] for _n, s, _b in demo]
    check('the CreateInfo road\'s text is unmoved: stripped, each pass is line for line the stripped 1.92.0 text',
          lines == [[ln for ln in DV.strip_declarations(s).splitlines() if ln.strip()] for _n, s, _b in old])
    check('...and holds no declaration at all (the interface declares the sampler)',
          all(not re.search(r'^\s*(?:uniform|in|out)\s+\w', DV.strip_declarations(s), re.M) for _n, s, _b in demo))
    with redirect_stdout(io.StringIO()):
        cpu = np.asarray(R.render(sc, st))
        img, hit = GSH.simulate(job, g, passes, atlases)
        img_old, hit_old = GSH.simulate(job, g, [(mi, n, s, b) for (mi, _n, _s, _b), (n, s, b) in zip(passes, old)], atlases)
    check('the simulator draws the once-declared passes', img is not None, str(hit))
    check('...and the 1.92.0 text\'s frame is that frame bit for bit (a declaration moves no value)',
          img is not None and img_old is not None and np.array_equal(img, img_old), str(hit_old))
    if img is not None:
        cov = g.tri >= 0
        d = np.abs(img[cov] - cpu[cov][:, :3])
        check('...which is the CPU frame, shadows and all (max < 6e-3, no px > 1e-2)',
              float(d.max()) < 6e-3 and int((d.max(axis=1) > 1e-2).sum()) == 0, f'max {float(d.max()):.5f}')

    # ---- what each road hands the driver
    for name, src, binds in demo[:1]:
        err, frag, _samplers = _driver_handover(src, binds, createinfo=False)
        check('the legacy constructor is handed the planned text itself -- the raw text, one declaration',
              err is None and frag == src and count(frag) == 1, str(err))
        err, frag, samplers = _driver_handover(src, binds, createinfo=True)
        names = [nm for _slot, nm in samplers]
        check('CreateInfo is handed the stripped text: the fetches, and no declaration of the sampler',
              err is None and frag == DV.strip_declarations(src) and count(frag) == 0
              and frag.count('texelFetch(hal_shadowpack') == src.count('texelFetch(hal_shadowpack'), str(err))
        check('...and the sampler once, in the slot its place in the list gives it (the three G-buffer samplers first)',
              names == ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris'] + list(binds['samplers'])
              and names.count('hal_shadowpack') == 1 and [slot for slot, _nm in samplers] == list(range(len(names))),
              str(samplers))
        err, frag, _samplers = _driver_handover(_as_1_92_declared(src), binds, createinfo=False)
        check('the 1.92.0 text reached the legacy constructor with BOTH declarations (the text glslang refuses)',
              err is None and count(frag) == 2)

    # ---- the strict front-end itself, where there is one (tests/glsl_strict.py is the gate; the suite has no glslang)
    from . import glsl_strict as GS
    if GS._glslang() is None:
        print('  note HALCYON_GLSLANG is not set: the strict front-end did not run on the shadowed passes')
    else:
        said = [GS.compile_source(s) for _n, s, _b in demo]
        check('glslang compiles each shadowed pass as raw text (strict GLSL 4.30)', all(ok for ok, _msg in said),
              '; '.join(msg[:160] for ok, msg in said if not ok))
        said = [GS.compile_source(s) for _n, s, _b in old]
        check('...and refuses the 1.92.0 text of each as a redefinition of the sampler',
              all(not ok and "'hal_shadowpack' : redefinition" in msg for ok, msg in said),
              '; '.join(msg[:160] for _ok, msg in said))

    # ---- the rule, read from the code
    fn = ast.parse(textwrap.dedent(inspect.getsource(GM._shadow_function)))
    said = [c.value for c in ast.walk(fn) if isinstance(c, ast.Constant) and isinstance(c.value, str)][1:]
    check('`_shadow_function` writes no declaration of its own: no string in it says `uniform` (its docstring aside)',
          bool(said) and not any('uniform' in t for t in said), str([t for t in said if 'uniform' in t]))
    asm = inspect.getsource(GM.assemble_frame)
    check('`assemble_frame` appends SHADOWPACK_DECL in one place, ahead of the loop that writes the functions, and lists the '
          'sampler in one place',
          asm.count('SHADOWPACK_DECL') == 1 and asm.count("'hal_shadowpack'") == 1
          and 0 < asm.find('SHADOWPACK_DECL') < asm.find('_shadow_function(i, smeta, consts)'))
    one = 'uniform sampler2D hal_shadowpack;\nuniform sampler2D hal_other; uniform float hal_k;  // note\n' \
          'float hal_f() { return hal_k; }\n'
    check('the text-level check reads whole lines the way `strip_declarations` does: comments aside, any number to a line',
          _declared_twice(one) == [] and _declared_twice(one + '  uniform sampler2D hal_other;  // again\n') == ['hal_other']
          and _declared_twice(one + 'uniform float hal_x; uniform float hal_k;\n') == ['hal_k']
          and _declared_twice(one + '// uniform float hal_k;\n') == [])
    check('...and Halcyon\'s front-end does take the redeclared sampler (the simulator could never have caught it)',
          try_compile('uniform sampler2D hal_a;\nuniform sampler2D hal_a;\nuniform vec2 vUV;\nout vec4 Color;\n'
                      'void main() { Color = texelFetch(hal_a, ivec2(0, 0), 0); }\n', 'GLSL')[0] is not None)


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
