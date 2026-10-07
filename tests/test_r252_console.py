"""R252 (1.91.0): the Console Emulation Shader, the trimmed master menu
and the RenderWare pipelines.

The node (nodes/shader_nodes.HALCYON_ConsoleShaderNode) carries the
period machines that used to be rows of the master shader's model menu,
with the SHADER TYPES each machine's polygon attribute word could select
and the options it carried (core/console.py). Every heading here is
proven at the function level first (the combine's own arithmetic, the
resolver's table), then at the frame level on the demo scene (the CPU
road), then through the GLSL simulator where the item has a twin (the
corner road: the pass interpolates the CPU's own corners and runs the
combine's GLSL twin, `d == 0.0` at every covered pixel unless a section
states another bar).

    python -m halcyon.tests.test_r252_console
"""
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import base_settings, _sk
from .scenebuild import demo_scene
from .featurematrix import _one_bsdf_graph
from .test_r251_material import _sim_vs_cpu, glsl_twin
from ..core import combine as CB
from ..core import console as CON
from ..core import render as R
from ..core import shading as SH
from ..core.scene import Light
from ..gpu import combine as GCB
from ..gpu import shade as GSH
from ..shaders.compiler import try_compile

FAILS = []
f32 = np.float32


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def console_graph(props, inputs=None, diffuse=(0.85, 0.2, 0.15)):
    """A single Console Emulation Shader graph, as the exporter carries it."""
    ins = [_sk('Diffuse Color', 'RGBA', list(diffuse) + [1.0]),
           _sk('Diffuse Level', 'VALUE', 1.0),
           _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
           _sk('Specular Level', 'VALUE', 0.9),
           _sk('Glossiness', 'VALUE', 48.0),
           _sk('Ambient', 'VALUE', 1.0),
           _sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
           _sk('Opacity', 'VALUE', 1.0),
           _sk('Toon Size', 'VALUE', 0.5),
           _sk('Vertex Color', 'RGBA', [0.3, 0.9, 0.4, 1.0]),
           _sk('Vertex Color Mix', 'VALUE', 0.0),
           _sk('Prelit Color', 'RGBA', [0.5, 0.5, 0.5, 1.0]),
           _sk('Night Color', 'RGBA', [0.1, 0.1, 0.3, 1.0]),
           _sk('Night Blend', 'VALUE', 0.0)]
    names = {s['name'] for s in (inputs or [])}
    ins = [s for s in ins if s['name'] not in names] + list(inputs or [])
    return _one_bsdf_graph('HALCYON_ConsoleShaderNode', ins, props)


def _scene(props, inputs=None, w=64, h=48, energy=0.45, vcol=False, **kw):
    """The demo scene with the Ball and the Box as console materials. The
    lamps are dimmed (`energy`) so the 8-bit units do not saturate the
    whole Ball; `vcol` False strips the mesh's (white) colour attribute
    so the vertex-colour types read the socket."""
    st = base_settings(w, h, shadows=False, **kw)
    sc = demo_scene(st)
    for l in sc.lights:
        l.energy = float(l.energy) * energy
    if not vcol and getattr(sc.mesh, 'colors', None) is not None:
        sc.mesh.colors = None            # an unpainted mesh
    sc.materials[1].graph = console_graph(props, inputs)
    sc.materials[2].graph = console_graph(props, inputs, diffuse=(0.2, 0.45, 0.85))
    return sc, st


def _render(props, inputs=None, **kw):
    sc, st = _scene(props, inputs, **kw)
    return np.asarray(R.render(sc, st))


def _ball_box(sc, st):
    """Boolean masks of the Ball and the Box materials in the frame."""
    from ..core import raster
    w, h = st.resolution_x, st.resolution_y
    _v, _p, vp, _e = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    cov = g.tri >= 0
    mi = np.full((h, w), -1, np.int32)
    mi[cov] = sc.mesh.mat_index[g.tri[cov]]
    return mi == 1, mi == 2


# ------------------------------------------------------------ the tables

def test_tables():
    """The engine table grew ten items, appended; the master's menu lost
    the moved ones and numbers its items by the engine index; every
    console type resolves to a real model; the resolver's defaults are
    the pre-R252 behaviour; the exporter names every node property."""
    names = [m[0] for m in SH.MODEL_ITEMS]
    i0 = names.index('SUPERFX_PLOT') + 1
    check('the ten R252 items follow SUPERFX_PLOT, in the spec order',
          names[i0:] == list(CB.PERIOD_MODELS_R252), str(names[i0:]))
    check('every R252 item has a substantial description naming its '
          'machine and year', all(len(m[2]) > 60 and '(' in m[1]
                                   for m in SH.MODEL_ITEMS[i0:]))
    check('the master menu offers 22 classic models and none of the moved '
          'ones', len(SH.MASTER_MODELS) == 22
          and not (set(SH.MASTER_MODELS) & set(SH.MOVED_MODELS)))
    items = SH.master_model_items()
    check('each master item carries its MODEL_ITEMS index as its number',
          all(len(it) == 5 and SH.MODEL_ITEMS[it[4]][0] == it[0]
              for it in items) and [it[0] for it in items] == list(SH.MASTER_MODELS))
    check('every moved model names a registered node',
          set(SH.MOVED_MODELS.values()) == {
              'HALCYON_AnimeShaderNode', 'HALCYON_CartoonNode',
              'HALCYON_MaxStandardNode', 'HALCYON_ConsoleShaderNode'})
    check('the master menu plus the moved models is the engine table '
          'minus the R252 items', set(SH.MASTER_MODELS) | set(SH.MOVED_MODELS)
          == set(names[:i0]))
    # every console type resolves to a model of the table
    bad = []
    for con, _l, _d in CON.CONSOLE_ITEMS:
        panel = CON.PANEL.get(con, ())
        tprop = panel[0] if panel else None
        items_t = next((it for nm, it, _d in CON.ENUM_PROPS if nm == tprop), ())
        for it in items_t or ((None,),):
            props = {'console': con}
            if tprop and it[0]:
                props[tprop] = it[0]
            m = CON.model_of(props)
            if m not in names or m not in CON.CONSOLE_MODELS:
                bad.append(f'{con}/{it[0]} -> {m}')
    check('every machine / type pair resolves to a CONSOLE_MODELS member '
          'of the engine table', not bad, ', '.join(bad))
    check('the combine dict carries non-default options only, and the '
          'defaults table names every key the hook reads',
          CON.combine_opts({'console': 'PC_FIXED'}) == {}
          and CON.combine_opts({'console': 'SUPERFX', 'sfx_type': 'SOLID'})
          == {'sfx_dither': False}
          and CON.combine_opts({'console': 'MODEL2', 'luma_gamma': 0.5})
          == {'luma_gamma': 0.5}
          and set(CON.COMBINE_DEFAULTS) >= {'luma_gamma', 'saturn_half',
                                            'texop', 'pcx_first', 'md_class',
                                            'sfx_dither', 'rw_matfx', 'rw_dual'})
    check('MIGRATE covers every console model a master node could carry '
          'and each entry resolves back to that model',
          set(CON.MIGRATE) == {k for k, v in SH.MOVED_MODELS.items()
                               if v == 'HALCYON_ConsoleShaderNode'}
          and all(CON.model_of(dict(v)) == k for k, v in CON.MIGRATE.items()))
    d = CON.resolve({})
    check('the resolver at the node defaults is the PlayStation Gouraud '
          'with nothing forced', d['model'] == 'PS1_MODULATE'
          and d['fixed_shade'] == 0.0 and d['light_limit'] == 0
          and d['axis_viewer'] == 0.0 and d['vmix'] is None
          and d['gloss'] is None and d['spec_level'] is None
          and d['rate'] == 'VERTEX' and d['prelit'] is None)
    from ..export import NODE_PROPS
    declared = {n for n, _i, _d in CON.ENUM_PROPS} | {n for n, _k, _d in CON.SCALAR_PROPS}
    check('export.NODE_PROPS names every console property',
          set(NODE_PROPS['HALCYON_ConsoleShaderNode']) == declared,
          str(declared ^ set(NODE_PROPS['HALCYON_ConsoleShaderNode'])))
    check('every panel property is declared', all(
        p in declared for ps in CON.PANEL.values() for p in ps))
    s = SH.Surface(3)
    check('the Surface defaults are inert: no fixed shading, no limit, the '
          'scene viewer, white prelight off, GX clamp / spec, the sun '
          'clamped, continuous alpha',
          np.all(s.fixed_shade == 0) and np.all(s.light_limit == 0)
          and np.all(s.axis_viewer == 0) and np.all(s.prelit == 1)
          and np.all(s.prelit_mode == 0) and np.all(s.gx_diff_fn == 0)
          and np.all(s.gx_attn_fn == 0) and np.all(s.sun_clamp == 1)
          and np.all(s.alpha_steps == 0))


# --------------------------------------------------------- the combines

def _rng(n, seed=0):
    g = np.random.default_rng(seed)
    L = g.uniform(0.0, 1.6, (n, 3)).astype(f32)
    A = g.uniform(0.0, 1.0, (n, 3)).astype(f32)
    a = g.uniform(0.0, 1.0, n).astype(f32)
    return L, A, a


def test_combine_laws():
    """Each R252 combine holds its machine's identities on exact inputs:
    a full-lit texel is itself under the TEV and the 8-bit modulate
    (the N64's own rounding leaves it one step short), the add saturates,
    the decals take the texel at alpha 1 and the light at alpha 0, the
    Jaguar keeps hue, the 3DO never drops below an eighth, the luma ramp
    at gamma 1 is the pre-R252 path bitwise."""
    L, A, a = _rng(512)
    one = np.ones((512, 3), f32)
    t8 = CB._t8(A)
    check('TEV: a full-lit texel is exactly itself (255 -> 256 trick)',
          np.array_equal(np.rint(CB.cb_tev(one, A) * 255), t8))
    check('8-bit modulate: a full-lit texel is exactly itself',
          np.array_equal(np.rint(CB.cb_mod8(one, A) * 255), t8))
    n64 = np.rint(CB.cb_n64(one, A) * 255)
    check("N64: full light leaves bright texels one step short (the RDP's "
          'quiet darkening the TEV does not have)',
          np.all(n64 <= t8) and np.any(n64 < t8))
    check('TEV never exceeds the 8-bit modulate by more than one step on '
          'random inputs',
          float(np.abs(CB.cb_tev(L, A) - CB.cb_mod8(L, A)).max() * 255) <= 1.0 + 1e-3)
    check('MOD2X doubles the product, saturating; MOD4X quadruples',
          np.array_equal(CB.cb_mod8_k(L, A, 2.0),
                         np.minimum(np.rint(CB.cb_mod8(L, A) * 255) * 2, 255) / f32(255.0)
                         * f32(255.0) * CB.R255)
          and np.all(CB.cb_mod8_k(L, A, 4.0) >= CB.cb_mod8_k(L, A, 2.0) - 1e-6))
    check('ADD8 saturates: texel + light at 255, never past it',
          np.all(CB.cb_add8(L, A) <= 1.0 + 1e-6)
          and np.array_equal(np.rint(CB.cb_add8(np.zeros_like(L), A) * 255), t8))
    check('ADDSIGNED8 at half light is the texel itself',
          np.array_equal(np.rint(CB.cb_addsigned8(np.full_like(L, 128.0 / 255.0), A) * 255),
                         t8))
    zero = np.zeros(512, f32)
    full = np.ones(512, f32)
    check('DECAL8: alpha 1 is the texel, alpha 0 the lit colour (8-bit)',
          np.array_equal(np.rint(CB.cb_decal8(L, A, full) * 255), t8)
          and np.array_equal(np.rint(CB.cb_decal8(L, A, zero) * 255), CB._s8(L)))
    ds1 = CB.cb_ds_decal(L, A, full)
    ds0 = CB.cb_ds_decal(L, A, zero)
    t6 = np.floor(np.clip(A, 0, 1) * 63 + 0.5)
    l6 = np.floor(np.clip(L, 0, 1) * 63 + 0.5)
    check("DS decal: alpha 1 is the 6-bit texel through GBATEK's /64 and "
          'alpha 0 the 6-bit lit colour through the same /64 (each one step '
          "short of itself: the mode-1 rule's own arithmetic)",
          np.array_equal(np.rint(ds1 * 63), np.floor(t6 * 63.0 / 64.0))
          and np.array_equal(np.rint(ds0 * 63), np.floor(l6 * 63.0 / 64.0)))
    nb1 = np.rint(CB.cb_n64_blend(L, A, full) * 255)
    check("N64 blend: alpha 1 is the texel within the combiner's own "
          'rounding (one step), alpha 0 the shade exactly',
          np.all(np.abs(nb1 - t8) <= 1.0) and np.any(nb1 == t8)
          and np.array_equal(np.rint(CB.cb_n64_blend(L, A, zero) * 255), CB._s8(L)))
    jag = CB.cb_jaguar(L, A)
    ratio = jag / np.maximum(A, 1e-6)
    hue_kept = np.abs(ratio - ratio[:, :1]).max(axis=1)
    check('Jaguar CRY: the three channels scale by ONE intensity (hue kept, '
          'within the 8-bit floor)', float(hue_kept[A.min(axis=1) > 0.2].max()) < 0.03,
          str(float(hue_kept[A.min(axis=1) > 0.2].max())))
    tdo = CB.cb_threedo(np.zeros_like(L), A)
    lit = A.min(axis=1) > 0.1
    check('3DO PIXC: no light still leaves 1/8 of the cel (the multiplier '
          'is 1..8), on the 5-bit cel grid',
          np.all(tdo[lit] > 0) and np.allclose(
              tdo, np.floor(np.floor(np.rint(np.clip(A, 0, 1) * 31) * 255 / 31) / 8) / 255,
              atol=1e-6))
    check('luma64 at gamma 1 is the pre-R252 path bitwise',
          np.array_equal(CB.cb_luma64(L, A), CB.cb_luma64(L, A, 1.0))
          and CB.luma_remap_table(1.0) is None)
    tab = CB.luma_remap_table(0.5)
    check('the luma remap table is monotone, 64 integers, ends pinned',
          tab.shape == (64,) and tab[0] == 0 and tab[63] == 63
          and np.all(np.diff(tab) >= 0) and np.array_equal(tab, np.floor(tab)))
    check('Saturn half luminance halves the texel before the add',
          np.all(CB.cb_saturn(L, A, half=True) <= CB.cb_saturn(L, A) + 1e-6)
          and not np.array_equal(CB.cb_saturn(L, A, half=True), CB.cb_saturn(L, A)))
    c0, c1, c2 = (np.random.default_rng(k).uniform(0, 1, (512, 3)).astype(f32)
                  for k in (3, 4, 5))
    bary = np.full((512, 3), 1.0 / 3.0, f32)
    check('PCX first-corner base differs from the mean base and is the '
          'corner where the corners agree',
          not np.array_equal(CB.cb_pcx(c0, c1, c2, bary, A),
                             CB.cb_pcx(c0, c1, c2, bary, A, first=True))
          and np.allclose(CB.cb_pcx(c0, c0, c0, bary, A, first=True),
                          CB.cb_pcx(c0, c0, c0, bary, A), atol=1e-6))
    S = np.random.default_rng(6).uniform(0, 0.5, 512).astype(f32)
    check('the D3D texture op under MOD is cb_d3dspec verbatim',
          np.array_equal(CB.cb_d3dspec_op(L, S, A, 'MOD'), CB.cb_d3dspec(L, S, A)))
    check('the Mega Drive forced class rides light_alpha',
          np.all(CB.light_alpha('MEGA_DRIVE_SH', L, {'md_class': 2.0}) == 2.0)
          and np.all(CB.light_alpha('MEGA_DRIVE_SH', L, {'md_class': 1.0}) == 1.0))


_WRAP = """
{fns}
{unis}
out vec4 Color;
void main() {{
{body}
}}
"""


def _twin(fns, call, L, A, a=None, extra=None):
    n = L.shape[0]
    unis = 'uniform vec3 L; uniform vec3 A; uniform float a;'
    body = f'    Color = vec4({call}, 1.0);'
    src = _WRAP.format(fns=fns, unis=unis, body=body)
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        return None, str(err)
    u = {'L': L, 'A': A, 'a': a if a is not None else np.ones(n, f32),
         'vUV': np.full((n, 2), 0.5, f32)}
    outs, _d = prog.run(u, {}, n)
    return np.asarray(outs['Color'], f32)[:, :3], None


def test_combine_gpu_twins():
    """Every R252 combine's GLSL text evaluates bitwise the CPU's (the
    simulator on the same float32 inputs, d == 0.0): the TEV, the 8-bit
    modulate and its 2x / 4x, add, add signed, the two decals, the N64
    blend, the Jaguar byte, the 3DO eighths, the luma ramp table, the
    Saturn half and the PCX first-corner base."""
    L, A, a = _rng(2048, 7)
    cases = [
        ('TEV', GCB.FN_TEV, 'hal_cb_tev(L, A)', CB.cb_tev(L, A)),
        ('8-bit modulate', GCB.FN_MOD8, 'hal_cb_mod8(L, A)', CB.cb_mod8(L, A)),
        ('MOD2X', GCB.FN_MOD8K, 'hal_cb_mod8k(L, A, 2.0)', CB.cb_mod8_k(L, A, 2.0)),
        ('MOD4X', GCB.FN_MOD8K, 'hal_cb_mod8k(L, A, 4.0)', CB.cb_mod8_k(L, A, 4.0)),
        ('ADD8', GCB.FN_ADD8, 'hal_cb_add8(L, A)', CB.cb_add8(L, A)),
        ('ADDSIGNED8', GCB.FN_ADDSIGNED8, 'hal_cb_addsigned8(L, A)',
         CB.cb_addsigned8(L, A)),
        ('DECAL8', GCB.FN_DECAL8, 'hal_cb_decal8(L, A, a)', CB.cb_decal8(L, A, a)),
        ('DS decal', GCB.FN_DSDECAL, 'hal_cb_dsdecal(L, A, a)', CB.cb_ds_decal(L, A, a)),
        ('N64 blend', GCB.FN_N64BLEND, 'hal_cb_n64blend(L, A, a)', CB.cb_n64_blend(L, A, a)),
        ('Jaguar', GCB.FN_JAGUAR, 'hal_cb_jaguar(L, A)', CB.cb_jaguar(L, A)),
        ('3DO', GCB.FN_THREEDO, 'hal_cb_threedo(L, A)', CB.cb_threedo(L, A)),
        ('luma64 gamma 0.5', GCB.fn_luma_remap(0.5) + GCB.FN_LUMA64,
         'hal_cb_luma64(L, A)', CB.cb_luma64(L, A, 0.5)),
        ('luma64 gamma 1 (identity remap)', GCB.fn_luma_remap(1.0) + GCB.FN_LUMA64,
         'hal_cb_luma64(L, A)', CB.cb_luma64(L, A)),
        ('Saturn half', GCB.FN_SATURN_HALF, 'hal_cb_saturn_half(L, A)',
         CB.cb_saturn(L, A, half=True)),
    ]
    for name, fns, call, cpu in cases:
        out, why = _twin(fns, call, L, A, a)
        if out is None:
            check(f'{name}: the GLSL twin compiles', False, why)
            continue
        d = float(np.abs(out - cpu).max())
        check(f'{name}: the GLSL twin is bitwise the CPU (d == 0.0)', d == 0.0,
              f'd={d}')
    # the PCX first-corner twin takes corners
    c0, c1, c2 = (np.random.default_rng(k).uniform(0, 1, (2048, 3)).astype(f32)
                  for k in (3, 4, 5))
    bary = np.random.default_rng(8).dirichlet((1, 1, 1), 2048).astype(f32)
    src = _WRAP.format(fns=GCB.FN_PCX_FIRST,
                       unis='uniform vec3 c0; uniform vec3 c1; uniform vec3 c2; '
                            'uniform vec3 b; uniform vec3 A;',
                       body='    Color = vec4(hal_cb_pcx1(c0, c1, c2, b, A), 1.0);')
    prog, err = try_compile(src, 'GLSL')
    if prog is None:
        check('PCX first-corner twin compiles', False, str(err))
    else:
        outs, _d = prog.run({'c0': c0, 'c1': c1, 'c2': c2, 'b': bary, 'A': A,
                             'vUV': np.full((2048, 2), 0.5, f32)}, {}, 2048)
        o = np.asarray(outs['Color'], f32)[:, :3]
        cpu = CB.cb_pcx(c0, c1, c2, bary, A, first=True)
        d = float(np.abs(o - cpu).max())
        check('PCX first-corner: the GLSL twin matches the CPU (the one '
              'non-power-of-two divide: 1 ulp)', d < 2e-6, f'd={d}')


# ------------------------------------------------------------ the frames

def test_types_render_distinct():
    """Every machine / type pair renders finite on the demo scene, and the
    types that resolve to different (model, flags) pairs render
    different pictures; the fixed-shading types are the albedo exactly
    (std 0 over the Ball); the vertex-colour types read the socket on
    an unpainted mesh."""
    imgs = {}
    keyed = {}
    for con, _l, _d in CON.CONSOLE_ITEMS:
        panel = CON.PANEL.get(con, ())
        tprop = panel[0] if panel else None
        items_t = next((it for nm, it, _d in CON.ENUM_PROPS if nm == tprop), ())
        for it in items_t or ((None,),):
            props = {'console': con}
            if tprop and it[0]:
                props[tprop] = it[0]
            img = _render(props)
            name = f'{con}/{it[0]}'
            imgs[name] = img
            if not np.isfinite(img).all():
                check(f'{name} renders finite', False)
            r = CON.resolve(props)
            if con == 'SUPERFX' or r['model'] == 'N64_SHADE':
                # the plot refuses on the demo scene's adaptive palette
                # (both fills are the FLAT frame); G_CC_SHADE differs
                # only under a linked texture
                continue
            if r['fixed_shade'] > 0.5 or r['model'] in ('DECAL_ALPHA', 'DS_DECAL'):
                # fixed shading is the albedo whatever the machine; a
                # decal at texel alpha 1 is the texel too
                key = ('FIXED', r['vmix'], r['prelit'], r['opacity'])
            else:
                key = (r['model'], r['vmix'], r['rate'], r['prelit'],
                       r['opacity'], tuple(sorted(r['combine'].items())))
            keyed.setdefault(key, []).append(name)
    check(f'all {len(imgs)} machine / type pairs render finite',
          all(np.isfinite(v).all() for v in imgs.values()))
    reps = [v[0] for v in keyed.values()]
    coll = [f'{a}=={b}' for i, a in enumerate(reps) for b in reps[i + 1:]
            if float(np.abs(imgs[a] - imgs[b]).mean()) < 1e-6]
    check(f'the {len(reps)} distinct resolutions render pairwise distinct '
          'pictures', not coll, ', '.join(coll))
    sc, st = _scene({'console': 'PS1', 'ps1_type': 'RAW'})
    ball, _box = _ball_box(sc, st)
    for name in ('PS1/RAW', 'PS2/DECAL', 'DREAMCAST/DECAL', 'SATURN/REPLACE',
                 'N64/DECAL', 'PSP/REPLACE', 'MODEL2/FIXED', 'SYSTEM22/FIXED',
                 'RENDERWARE/UNLIT', 'PSP/DECAL', 'DS/DECAL'):
        px = imgs[name][ball][:, :3]
        check(f'{name}: fixed shading is flat over the Ball',
              float(px.std(axis=0).max()) < 2e-3, str(px.std(axis=0).max()))
    rw = imgs['RENDERWARE/UNLIT'][ball][:, :3].mean(axis=0)
    check('RenderWare unlit prelit: the grey prelight halves the colour on '
          'the GS grid (0.5 -> 64/128)', np.allclose(rw, np.array([0.85, 0.2, 0.15]) * 0.5, atol=0.012),
          str(rw))
    gcv = imgs['GAMECUBE/VERTEX'][ball][:, :3].mean(axis=0)
    check('GX vertex colour on an unpainted mesh reads the socket (the '
          'green swatch), flat', np.allclose(gcv, (0.3, 0.9, 0.4), atol=0.02)
          and float(imgs['GAMECUBE/VERTEX'][ball][:, :3].std(axis=0).max()) < 2e-3, str(gcv))
    n64s = imgs['N64/SHADE'][ball][:, :3]
    check('N64 shade only keeps the material colour (red) and the light',
          n64s[:, 0].mean() > 2 * n64s[:, 1].mean() and n64s[:, 0].std() > 0.05)


def test_options_change_the_picture():
    """Each option moves the frame it claims to: GX's diffuse function, the
    attenuation function's highlight gate, Model 2's specular bits and the
    luma gamma, Model 3's exponent / sun clamp / alpha steps, the DS table,
    the PS1 light limit, the D3D texture ops and viewer, the Mega Drive
    class, the Saturn half, the Super FX fill, the RenderWare platform,
    night blend, env map, specular plugin and dual texture."""
    def differs(a, b, tag, want=True, tol=1e-6):
        d = float(np.abs(_render(a) - _render(b)).mean()) if isinstance(a, dict) \
            else float(np.abs(a - b).mean())
        return check(tag, (d > tol) == want, f'mean |d| = {d:.5f}')

    gx = {'console': 'GAMECUBE'}
    differs(gx, dict(gx, gc_diffuse_fn='SIGN'), 'GX_DF_SIGN differs from CLAMP')
    differs(gx, dict(gx, gc_diffuse_fn='NONE'), 'GX_DF_NONE differs from CLAMP')
    differs(gx, dict(gx, gc_attn_fn='SPOT'), 'GX_AF_SPOT drops the highlight')
    sc, st = _scene(gx)
    ball, _b = _ball_box(sc, st)
    sp = _render(dict(gx, gc_attn_fn='SPEC'))[ball]
    no = _render(dict(gx, gc_attn_fn='NONE'))[ball]
    check('under GX_AF_NONE no pixel is brighter than under GX_AF_SPEC '
          '(the highlight is the only difference)', np.all(no[:, :3] <= sp[:, :3] + 1e-5))
    m2 = {'console': 'MODEL2'}
    differs(dict(m2, m2_specular='OFF'), dict(m2, m2_specular='P8'),
            'Model 2 specular OFF differs from power 8')
    differs(dict(m2, m2_specular='P1'), dict(m2, m2_specular='P8'),
            'Model 2 power 1 differs from power 8')
    differs(m2, dict(m2, luma_gamma=0.5), 'the luma ramp gamma moves the frame')
    m3 = {'console': 'MODEL3'}
    differs(dict(m3, m3_specular='P8'), dict(m3, m3_specular='P64'),
            'Model 3 exponent 8 differs from 64')
    differs(m3, dict(m3, m3_sun_clamp=False), 'the sun clamp off goes darker '
            'than ambient somewhere')
    img_c = _render(m3)
    img_u = _render(dict(m3, m3_sun_clamp=False))
    check('sun clamp off never brightens a pixel', np.all(img_u[..., :3] <= img_c[..., :3] + 1e-5))
    op = [_sk('Opacity', 'VALUE', 0.3)]
    a_c = _render(m3, op)[..., 3]
    a_s = _render(dict(m3, m3_alpha_steps=True), op)[..., 3]
    check('alpha steps: the Ball alpha lands on a 1/32 step (0.3 -> 10/32)',
          np.allclose(np.unique(np.round(a_s[ball], 4)), np.unique(np.round(a_s[ball], 4)))
          and abs(float(a_s[ball].mean()) - 10.0 / 32.0) < 1e-3
          and abs(float(a_c[ball].mean()) - 0.3) < 1e-3, f'{a_s[ball].mean()} {a_c[ball].mean()}')
    ds = {'console': 'DS'}
    differs(dict(ds, ds_table='LINEAR'), dict(ds, ds_table='PIN'),
            'the DS table shape moves the highlight')
    differs(dict(ds, ds_type='TOON', toon_steps=2), dict(ds, ds_type='TOON', toon_steps=5),
            'the toon step count moves the DS table')
    # the light limit: three extra lamps, the limit drops them
    def with_extra(props):
        sc, st = _scene(props)
        for i, (c, d) in enumerate((((1, 0.2, 0.2), (0.6, -0.5, -0.5)),
                                    ((0.2, 1, 0.2), (-0.2, -0.8, -0.4)),
                                    ((0.2, 0.2, 1), (0.3, 0.7, -0.6)))):
            sc.lights.append(Light(type='SUN', name=f'extra{i}', direction=d,
                                   color=c, energy=3.0, shadow='NONE'))
        return np.asarray(R.render(sc, st))
    ps1 = {'console': 'PS1'}
    base = with_extra(ps1)
    lim = with_extra(dict(ps1, light_limit=True))
    plain = _render(ps1)
    check("the PS1's three-light limit drops the extra lamps: the limited "
          'frame differs from the five-lamp one',
          float(np.abs(base - lim).mean()) > 1e-4)
    check('...and with two scene lamps plus three extras, the first THREE '
          'light it (the limited frame is not the two-lamp frame either)',
          float(np.abs(plain - lim).mean()) > 1e-4)
    pc = {'console': 'PC_FIXED'}
    for opn in ('MODULATE2X', 'MODULATE4X', 'ADD', 'ADDSIGNED'):
        differs(pc, dict(pc, pc_texture_op=opn), f'D3D texture op {opn} differs from MODULATE')
    differs(pc, dict(pc, pc_local_viewer=False), 'the local viewer off (camera axis) moves the highlight')
    differs(pc, dict(pc, pc_color_vertex=True), 'colour vertex takes the vertex colour as the material')
    md = {'console': 'MEGADRIVE'}
    differs(dict(md, md_type='SHADOW'), dict(md, md_type='HIGHLIGHT'), 'the forced S/H classes differ')
    differs(dict(md, md_type='AUTO'), dict(md, md_type='HIGHLIGHT'), 'the automatic class differs from HIGHLIGHT forced')
    sat = {'console': 'SATURN'}
    differs(sat, dict(sat, sat_type='HALF_LUM'), 'Saturn half luminance darkens')
    pcx = {'console': 'PCX'}
    differs(pcx, dict(pcx, pcx_base='FIRST'), 'the PCX first-corner base differs from the mean')
    rw = {'console': 'RENDERWARE'}
    differs(dict(rw, rw_platform='PS2'), dict(rw, rw_platform='PC'), 'RenderWare PS2 differs from PC (the overbright grid)')
    dgp = float(np.abs(_render(dict(rw, rw_platform='GC')) - _render(dict(rw, rw_platform='PC'))).max())
    check('RenderWare GC and PC agree within one 8-bit step (the TEV rounding '
          'bound)', dgp <= 1.0 / 255.0 + 1e-6, f'max |d| = {dgp:.5f}')
    differs(rw, dict(rw, rw_specular='PLUGIN'), 'the GTA specular plugin adds a highlight')
    night = [_sk('Night Blend', 'VALUE', 0.8)]
    check('night blend darkens the prelit frame',
          float(_render(rw, night)[..., :3].mean()) < float(_render(rw)[..., :3].mean()))
    env = [_sk('Env Map', 'RGBA', [0.6, 0.7, 0.9, 1.0]), _sk('Env Map Coefficient', 'VALUE', 0.5)]
    check('the env map adds by its coefficient under ENVMAP and nothing under NONE',
          float(np.abs(_render(dict(rw, rw_matfx='ENVMAP'), env) - _render(rw, env)).mean()) > 1e-4
          and float(np.abs(_render(rw, env) - _render(rw)).mean()) < 1e-6)
    sfx = {'console': 'SUPERFX'}
    st_kw = dict(palette_mode='EGA16', gamma=1.0, color_management='NONE')
    try:
        a_d = _render(sfx, **st_kw)
        a_s = _render(dict(sfx, sfx_type='SOLID'), **st_kw)
        check('the Super FX solid fill differs from the dither pair under a '
              'fixed 16-colour palette', float(np.abs(a_d - a_s).mean()) > 1e-6)
    except Exception as exc:                                    # noqa: BLE001
        check('the Super FX gates accept the fixed-palette settings', False, str(exc))


def test_rate_override():
    """The node's Rate menu decides per material (Machine / Scene / Vertex /
    Face) for the models the hardware tables do not pin; the Sega boards
    and the rate-fixed items keep theirs; a PS1 flat primitive is the FACE
    road beside a Gouraud one in the same scene."""
    st = base_settings(32, 24, shading_rate='PIXEL')
    sc = demo_scene(st)

    def rate(props):
        sc.materials[1].graph = console_graph(props)
        m = R.material_model(sc.materials[1], st)
        return CB.rate_for_model(m, st, sc.materials[1]) or R.RATE_FOR_MODEL.get(m, st.shading_rate)
    check('Machine: the DS lights per vertex', rate({'console': 'DS'}) == 'VERTEX')
    check('Scene: the DS follows a pixel-rate scene',
          rate({'console': 'DS', 'rate': 'SCENE'}) == 'PIXEL')
    check('Face: a GX material lights per face on request',
          rate({'console': 'GAMECUBE', 'rate': 'FACE'}) == 'FACE')
    check('a PS1 flat primitive is FACE whatever the Rate menu says',
          rate({'console': 'PS1', 'ps1_type': 'FLAT', 'rate': 'VERTEX'}) == 'FACE')
    check('the Jaguar blitter fills per vertex, or flat on request',
          rate({'console': 'JAGUAR'}) == 'VERTEX'
          and rate({'console': 'JAGUAR', 'jag_type': 'FLAT'}) == 'FACE')
    check('Model 2 stays FACE and Model 3 VERTEX under any Rate menu',
          rate({'console': 'MODEL2', 'rate': 'VERTEX'}) == 'FACE'
          and rate({'console': 'MODEL3', 'rate': 'FACE'}) == 'VERTEX')
    check('the Mega Drive and the 3DO stay FACE, RenderWare VERTEX',
          rate({'console': 'MEGADRIVE', 'rate': 'VERTEX'}) == 'FACE'
          and rate({'console': 'THREEDO', 'rate': 'VERTEX'}) == 'FACE'
          and rate({'console': 'RENDERWARE', 'rate': 'FACE'}) == 'VERTEX')
    check('a combine under Scene on a pixel-rate scene lights per vertex',
          rate({'console': 'N64', 'rate': 'SCENE'}) == 'VERTEX')
    check('rate_for_model without a material is the pre-R252 answer',
          CB.rate_for_model('PS1_MODULATE', st) == 'VERTEX'
          and CB.rate_for_model('DS_FIXED', st) is None)


def test_gpu_parity():
    """Through the GLSL simulator: a console material at its machine's
    rate qualifies for the GPU and the frame is the CPU's (the corner
    road's bar: max < 6e-3, no pixel > 1e-2) for the RenderWare three,
    the decals, the N64 blend, the Jaguar, the 3DO and the D3D ops;
    the pixel-rate refusals say their reason by name."""
    cases = [
        ('RenderWare PS2', {'console': 'RENDERWARE', 'rw_platform': 'PS2'}),
        ('RenderWare GC', {'console': 'RENDERWARE', 'rw_platform': 'GC'}),
        ('RenderWare PC', {'console': 'RENDERWARE', 'rw_platform': 'PC'}),
        ('RenderWare unlit', {'console': 'RENDERWARE', 'rw_type': 'UNLIT'}),
        ('DS decal', {'console': 'DS', 'ds_type': 'DECAL'}),
        ('PSP decal', {'console': 'PSP', 'psp_type': 'DECAL'}),
        ('PSP add', {'console': 'PSP', 'psp_type': 'ADD'}),
        ('N64 blend', {'console': 'N64', 'n64_type': 'BLENDRGBA'}),
        ('N64 shade', {'console': 'N64', 'n64_type': 'SHADE'}),
        ('Jaguar', {'console': 'JAGUAR'}),
        ('3DO', {'console': 'THREEDO'}),
        ('D3D MODULATE2X', {'console': 'PC_FIXED', 'pc_texture_op': 'MODULATE2X'}),
        ('D3D ADDSIGNED', {'console': 'PC_FIXED', 'pc_texture_op': 'ADDSIGNED'}),
        ('Model 2 gamma 0.5', {'console': 'MODEL2', 'luma_gamma': 0.5}),
        ('Saturn half', {'console': 'SATURN', 'sat_type': 'HALF_LUM'}),
        ('PCX first', {'console': 'PCX', 'pcx_base': 'FIRST'}),
        ('GX SIGN at vertex rate', {'console': 'GAMECUBE', 'gc_diffuse_fn': 'SIGN'}),
        ('GX SPOT at pixel rate', {'console': 'GAMECUBE', 'gc_attn_fn': 'SPOT',
                                   'rate': 'SCENE'}),
        ('GX NONE at pixel rate', {'console': 'GAMECUBE', 'gc_diffuse_fn': 'NONE',
                                   'rate': 'SCENE'}),
        ('PS1 raw at pixel rate', {'console': 'PS1', 'ps1_type': 'RAW',
                                   'rate': 'SCENE'}),
        ('D3D axis viewer at pixel rate', {'console': 'PC_FIXED', 'pc_type': 'GOURAUD',
                                           'pc_local_viewer': False, 'rate': 'SCENE'}),
    ]
    for name, props in cases:
        sc, st = _scene(props, w=48, h=36, shading_rate='PIXEL')
        st.render_device = 'GPU'
        d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
        if d is None:
            check(f'{name}: the frame qualifies for the GPU', False, str(why))
            continue
        check(f'{name}: the simulator frame is the CPU frame (max {d:.2e}, '
              f'{nbad} px > 1e-2)', d < 6e-3 and nbad == 0)
    # the refusals by name
    sc, st = _scene({'console': 'DS', 'light_limit': True, 'rate': 'SCENE'},
                    w=32, h=24, shading_rate='PIXEL')
    st.render_device = 'GPU'
    d, nbad, img, cpu, passes, why = _sim_vs_cpu(sc, st)
    check('a light limit at the pixel rate refuses the GPU by name',
          why is not None and 'light limit' in str(why), str(why))


def test_node_and_migration():
    """The node registers, sits in the Shading family, documents every
    socket, hides what each type ignores; a master node saved with a
    moved model (its raw enum integer) is rebuilt as the right node at
    load with its links and values; the master's own menu holds."""
    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    import importlib
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    N = SN.HALCYON_ConsoleShaderNode
    check('the Console node registers in NODES and the Shading family',
          N in SN.NODES and N in SN.MENU_FAMILIES[0][2])
    names = [s[1] for s in N.SOCKETS]
    miss = [n for n in names if len(SN.CONSOLE_SOCKET_DOCS.get(n, '')) < 40]
    check(f'the node documents every one of its {len(names)} sockets',
          not miss, ', '.join(miss))
    extra = [n for con in CON.MACHINE_SOCKETS for n in CON.MACHINE_SOCKETS[con]
             if n not in names]
    check('every machine socket the table names exists on the node',
          not extra, ', '.join(extra))
    ann = N.__annotations__
    declared = {n for n, _i, _d in CON.ENUM_PROPS} | {n for n, _k, _d in CON.SCALAR_PROPS}
    check('the node declares exactly the table\'s properties',
          set(ann) == declared, str(set(ann) ^ declared))
    thin = [k for k, v in ann.items() if len(str(v.kw.get('description') or '')) < 40]
    check('every property carries a real tooltip', not thin, ', '.join(thin))
    # the master menu
    items = SN.HALCYON_ShaderNode.__annotations__['model'].kw['items']
    check('the master offers the 22 classics only, numbered by the engine '
          'table', [it[0] for it in items] == list(SH.MASTER_MODELS)
          and all(SH.MODEL_ITEMS[it[4]][0] == it[0] for it in items))
    check('the master draws Toon Steps for Toon only and no console row '
          'remains in RELEVANT', 'DS_TOON' not in SN.HALCYON_ShaderNode.RELEVANT
          and 'CARTOON' not in SN.HALCYON_ShaderNode.RELEVANT
          and 'TOON' in SN.HALCYON_ShaderNode.RELEVANT)

    # ---- the migration, on a fake tree
    class _Sock:
        def __init__(self, name, value=None):
            self.name = name
            self.default_value = value
            self.is_linked = False
            self.links = []
            self.hide = False

    class _Link:
        def __init__(self, a, b):
            self.from_socket, self.to_socket = a, b

    class _Node:
        def __init__(self, idn):
            self.bl_idname = idn
            self.inputs = []
            self.outputs = []
            self.location = (0, 0)
            self.label = ''
            self.hide = False
            self._raw = {}

        def get(self, k):
            return self._raw.get(k)

    class _Inputs(list):
        def get(self, name):
            return next((s for s in self if s.name == name), None)

        def new(self, kind, name):
            s = _Sock(name)
            self.append(s)
            return s

    class _Tree:
        def __init__(self):
            self.nodes = _Nodes(self)
            self.links = _Links()

    class _Links(list):
        def new(self, a, b):
            ln = _Link(a, b)
            self.append(ln)
            b.is_linked = True
            b.links = [ln]
            a.links = getattr(a, 'links', []) + [ln]
            return ln

    class _Nodes(list):
        def __init__(self, tree):
            super().__init__()
            self.tree = tree

        def new(self, idn):
            n = _Node(idn)
            n.inputs = _Inputs()
            n.outputs = _Inputs()
            if idn == 'HALCYON_ConsoleShaderNode':
                for kind, name, d in N.SOCKETS:
                    n.inputs.new(kind, name).default_value = d
                n.outputs.new('NodeSocketShader', 'Surface')
                for name, _i, d in CON.ENUM_PROPS:
                    setattr(n, name, d)
                for name, _k, d in CON.SCALAR_PROPS:
                    setattr(n, name, d)
                n.refresh_sockets = lambda: None
            elif idn == 'HALCYON_MaxStandardNode':
                n.inputs.new('NodeSocketColor', 'Diffuse Color')
                n.outputs.new('NodeSocketShader', 'Surface')
            else:
                n.inputs.new('NodeSocketColor', 'Diffuse Color')
                n.outputs.new('NodeSocketShader', 'Surface')
            self.append(n)
            return n

        def remove(self, n):
            if n in self:
                super().remove(n)

    tree = _Tree()
    old = _Node('HALCYON_ShaderNode')
    old.inputs = _Inputs()
    old.outputs = _Inputs()
    for kind, name, d in SN.HALCYON_ShaderNode.SOCKETS:
        old.inputs.new(kind, name).default_value = d
    old.inputs.get('Diffuse Color').default_value = (0.1, 0.2, 0.3, 1.0)
    old.inputs.get('Toon Size').default_value = 0.7
    old.outputs.new('NodeSocketShader', 'Surface')
    old.toon_steps = 5
    old.model = ''                               # the stale enum reads empty
    old._raw['model'] = [m[0] for m in SH.MODEL_ITEMS].index('DS_TOON')
    tree.nodes.append(old)
    tex = _Node('ShaderNodeTexImage')
    tex.outputs = _Inputs()
    tex.outputs.new('NodeSocketColor', 'Color')
    tree.nodes.append(tex)
    out = _Node('ShaderNodeOutputMaterial')
    out.inputs = _Inputs()
    out.inputs.new('NodeSocketShader', 'Surface')
    tree.nodes.append(out)
    tree.links.new(tex.outputs[0], old.inputs.get('Diffuse Color'))
    tree.links.new(old.outputs[0], out.inputs[0])
    check('saved_master_model reads the raw enum integer through the engine '
          'table', SN.saved_master_model(old) == 'DS_TOON')
    new = SN.migrate_master_node(tree, old)
    check('a DS_TOON master becomes a Console node set to the DS toon table '
          'with its Toon Steps', new is not None
          and new.bl_idname == 'HALCYON_ConsoleShaderNode'
          and new.console == 'DS' and new.ds_type == 'TOON'
          and new.ds_table == 'CUSTOM' and new.toon_steps == 5
          and new.rate == 'SCENE')
    check('the texture link and the socket values travel, the output '
          'relinks, the old node is gone',
          new.inputs.get('Diffuse Color').is_linked
          and new.inputs.get('Diffuse Color').links[0].from_socket is tex.outputs[0]
          and new.inputs.get('Toon Size').default_value == 0.7
          and out.inputs[0].links[0].from_socket is new.outputs[0]
          and old not in tree.nodes)
    for model, idn, attr in (('ANIME', 'HALCYON_AnimeShaderNode', None),
                             ('CARTOON', 'HALCYON_CartoonNode', None),
                             ('MAX_METAL', 'HALCYON_MaxStandardNode', 'METAL'),
                             ('GX_LIGHT', 'HALCYON_ConsoleShaderNode', 'GAMECUBE')):
        o = _Node('HALCYON_ShaderNode')
        o.inputs = _Inputs()
        o.outputs = _Inputs()
        o.inputs.new('NodeSocketColor', 'Diffuse Color')
        o.outputs.new('NodeSocketShader', 'Surface')
        o.model = model
        tree.nodes.append(o)
        n2 = SN.migrate_master_node(tree, o)
        ok = n2 is not None and n2.bl_idname == idn
        if ok and attr and idn == 'HALCYON_MaxStandardNode':
            ok = n2.shader_type == attr
        if ok and attr and idn == 'HALCYON_ConsoleShaderNode':
            ok = n2.console == attr
        check(f'a {model} master becomes a {idn}', ok)
    keep = _Node('HALCYON_ShaderNode')
    keep.inputs = _Inputs()
    keep.outputs = _Inputs()
    keep.model = 'PHONG'
    tree.nodes.append(keep)
    check('a Phong master is left alone', SN.migrate_master_node(tree, keep) is None
          and keep in tree.nodes)


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
    print('all R252 console checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
