"""Strict GLSL check of every shader Halcyon can hand a driver -- a TOOL,
not a suite member: it needs the Khronos glslang front-end, which no CI
runner ships.

    HALCYON_GLSLANG=/path/to/glslang python3 -m halcyon.tests.glsl_strict

(build glslang from https://github.com/KhronosGroup/glslang: cmake -G
Ninja -DENABLE_OPT=OFF -DENABLE_HLSL=OFF .. && ninja glslang-standalone;
the binary is StandAlone/glslang). Without the variable the tool prints
one line saying it did not run and exits 0.

Why it exists (1.91.0, the first field log): the simulator that proves
every twin bitwise accepts a GLSL text a driver rejects -- a function
defined twice ('function already has a body'), the Modulate 4x crash --
so the strict front-end is the one check that stands in for the driver.
It compiles, as desktop GLSL 4.30 fragments: every material pass the
deferred road plans for the demo scene under every engine model at every
shading rate, every Console Emulation Shader type and option at the
machine's and the scene's rate, the anime / cartoon / BI / Max masters,
the post stages (gpu/stages*.py), the sky pass source and the
rasteriser's fragment road. Refusals BY NAME are not failures (they never
reach a driver). R253 (field): the compute RASTER is covered too, as a
4.30 compute shader in all eight variants -- its text alone is not a unit
(the device's CreateInfo declares the interface), so the check writes the
header the CreateInfo writes: the images, the samplers and
`craster.RASTER_FLOATS`, over `device.strip_declarations(source)`. A push
constant the dispatch names and the text lost, or the reverse, fails here.
R253 (review): the Coded Shader's pass too, by NODE NAME -- the node's
graph id is the tag its GLSL names are mangled under (gpu/emit._code_tag),
the one place a name the user typed reaches the driver. A name in another
script ('Señal', Cyrillic, CJK) went into the identifiers letter for
letter; the tag spells those as code points now, and the pass of each
compiles here beside the plain names and the two kinds of clash.
R253 (review): SHADOWS are covered. Every sweep above planned its scene
with shadows off, so no pass carrying a shadow term had ever been
compiled here -- and the RAW text of one declared `uniform sampler2D
hal_shadowpack;` once per mapped lamp: under two (the demo scene's SUN
and POINT) a redefinition, on the one road that hands a driver the raw
text (the legacy constructor; CreateInfo's text is stripped). The
material sweeps run twice now, shadows off and then ON with the demo
scene's two mapped lamps: the engine models at every rate, the console
types and options, the cel masters, the coded shaders. And
`check_shadow_kinds` plans the demo scene under the lamps' other shadow
kinds -- traced (hard; soft for a SUN's disc, a POINT's sphere, an AREA
lamp's rectangle and its disk), a mapped lamp beside a traced one (Per
Light), the MIDPOINT map, a SPOT's single map beside the POINT's cube,
one to sixteen PCF taps, eight mapped lamps on the one atlas -- with the
BVH render() builds for traced shadows on the job, and holds each pass to
the shadow functions its kind names. A shadowed sweep in which NO pass
carries a shadow function fails by name: it checked nothing, which is how
shadows stayed unseen. A text glslang has judged is not handed over
again (the Floor's pass is one text under every console option), so the
doubled sweeps cost their planning and little else.
Not covered: the ray kernels (gpu/rtrace, their push constants are listed
at each call), the ink pass (built per frame from a job), and the lamps
render() itself adds to a frame before it plans -- a caustics cookie, the
World Light -- which the job recipe here does not (in a pass they are a
cookie function and one more mapped SUN).
R253 (review): the EMISSION SLOT ROUTING. A closure of emission lobes
alone (a colour wired straight into Surface, an Emission node, a Mix /
Add Shader of those) rides the pass's emission term since the route was
ported, and its text had been judged by the simulator alone:
check_emission_route compiles ten such graphs at every shading rate,
shadows off and on, the bare-socket literal included.
"""
import os
import re
import subprocess
import sys
import tempfile

import numpy as np

from . import utf8_console
from .test_render import base_settings, _sk
from .scenebuild import demo_scene
from .featurematrix import _one_bsdf_graph
from .test_r251_material import _job_for
from .test_r252_console import _scene as console_scene
from ..core import console as CON
from ..core import render as R
from ..core import shading as SH
from ..gpu import shade as GSH

FAILS = []


def _glslang():
    path = os.environ.get('HALCYON_GLSLANG', '')
    return path if path and os.path.exists(path) else None


def compile_source(src, stage='frag'):
    """(ok, message) from glslang on `src` with a 4.30 version line.

    Bytes both ways: the text is written as UTF-8 and the message decoded
    leniently. A source holding a letter outside ASCII is exactly what
    this tool exists to report, and glslang quotes ONE byte of it in its
    error -- decoding that strictly raised out of the check instead of
    failing the shader by name."""
    with tempfile.NamedTemporaryFile('w', suffix=f'.{stage}', delete=False,
                                     encoding='utf-8') as fh:
        fh.write('#version 430 core\n' + src)
        path = fh.name
    try:
        r = subprocess.run([_glslang(), '-S', stage, path], capture_output=True)
        return r.returncode == 0, (r.stdout + r.stderr).decode('utf-8', 'replace').strip()
    finally:
        os.unlink(path)


#: label suffix of a sweep's shadow state (off keeps the 1.92.0 labels)
_SHADOWS = {False: '', True: ', shadows on'}

#: (stage, text) -> glslang's (ok, message): a text is judged once
_JUDGED = {}


def _frame_bvh(sc, st):
    """The BVH render() builds for a frame whose shadows are traced
    (Shadow Method Ray, or Per Light with a Ray lamp), else None. Without
    one on the job a RAY lamp plans NO shadow term at all
    (gpu/shade._shadow_meta mirrors the CPU's fully-lit answer), and the
    pass compiled here would be the unshadowed one."""
    from ..core import render as R
    if st.shadows and (st.shadow_default == 'RAY'
                       or R._per_light_ray_shadows(sc, st)):
        return R._cached_bvh(sc, sc.mesh)
    return None


def _shadow_fns(src, traced=False):
    """How many lamps' shadow functions a pass DEFINES: the mapped ones
    (three arguments, the atlas fetch), or the traced ones (a fourth, the
    distance)."""
    tail = r', float dist\)' if traced else r'\)'
    return len(re.findall(r'^float hal_shadow_vis\d+\(vec3 P, vec3 N, vec3 L'
                          + tail + r'\s*$', src, re.M))


def _shadowed(srcs):
    """The sources that carry a shadow term, of either kind."""
    return sum(1 for src in (srcs or ())
               if _shadow_fns(src) or _shadow_fns(src, traced=True))


def _shadows_seen(what, n):
    """A shadowed sweep whose passes carry no shadow function checked
    nothing -- say so, and fail: the gate planned shadows off for two
    rounds and nobody could tell from its output."""
    print(f'      {n} of those passes carry a shadow term')
    if n == 0:
        FAILS.append(f'{what}: shadows on, yet no pass carries a shadow '
                     'function -- the sweep checked nothing')
        print(f'  FAIL {what}: shadows on, yet no pass carries a shadow '
              'function')


def _pass_sources(sc, st):
    """The fragment sources of every material pass the plan makes, or
    (None, why) when the plan refuses the frame by name."""
    st.render_device = 'GPU'
    g, job = _job_for(sc, st, bvh=_frame_bvh(sc, st))
    GSH._PLAN_CACHE.clear()
    passes, why, _atl = GSH.plan_frame(job, g)
    if passes is None:
        return None, why
    out = []
    for ps in passes:
        src = next((x for x in ps if isinstance(x, str) and 'void main' in x), None)
        if src is not None:
            out.append(src)
    return out, None


def _record(label, srcs, why, stage='frag'):
    if srcs is None:
        return 'refused'
    for src in srcs:
        key = (stage, src)
        if key not in _JUDGED:
            _JUDGED[key] = compile_source(src, stage)
        ok, msg = _JUDGED[key]
        if not ok:
            FAILS.append(f'{label}: {msg[:400]}')
            print(f'  FAIL {label}\n       {msg[:300]}')
            return 'bad'
    return 'ok'


def check_models():
    for shadows in (False, True):
        counts = {'ok': 0, 'refused': 0, 'bad': 0}
        lit = 0
        for ident, _l, _d in SH.MODEL_ITEMS:
            for rate in ('PIXEL', 'VERTEX', 'FACE'):
                st = base_settings(24, 18, shadows=shadows, force_model=ident,
                                   shading_rate=rate)
                srcs, why = _pass_sources(demo_scene(st), st)
                lit += _shadowed(srcs)
                counts[_record(f'model {ident} at {rate}{_SHADOWS[shadows]}',
                               srcs, why)] += 1
        print(f'  engine models{_SHADOWS[shadows]}: {counts}')
        if shadows:
            _shadows_seen('engine models', lit)


def check_console():
    for shadows in (False, True):
        counts = {'ok': 0, 'refused': 0, 'bad': 0}
        lit = 0
        for con, _l, _d in CON.CONSOLE_ITEMS:
            for prop in CON.PANEL.get(con, ()):
                items = next((it for nm, it, _d in CON.ENUM_PROPS if nm == prop), None)
                vals = [it[0] for it in items] if items else [None]
                for v in vals:
                    props = {'console': con}
                    if v is not None:
                        props[prop] = v
                    else:
                        props[prop] = {'luma_gamma': 0.5, 'm3_sun_clamp': False,
                                       'm3_alpha_steps': True, 'light_limit': True,
                                       'pc_color_vertex': True, 'pc_local_viewer': False,
                                       'toon_steps': 4}.get(prop, True)
                    for rate in ('MACHINE', 'SCENE'):
                        for shading in ('VERTEX', 'PIXEL'):
                            sc, st = console_scene(dict(props, rate=rate), w=24, h=18,
                                                   shading_rate=shading)
                            # (the console scene is built with shadows off;
                            # its lamps are the demo scene's two mapped ones)
                            st.shadows = shadows
                            srcs, why = _pass_sources(sc, st)
                            lit += _shadowed(srcs)
                            counts[_record(f'console {con} {prop}={props[prop]} '
                                           f'{rate}/{shading}{_SHADOWS[shadows]}',
                                           srcs, why)] += 1
        print(f'  console types and options{_SHADOWS[shadows]}: {counts}')
        if shadows:
            _shadows_seen('console types and options', lit)


def check_masters():
    graphs = {
        'anime': _one_bsdf_graph('HALCYON_AnimeShaderNode',
                                 [_sk('Diffuse Color', 'RGBA', [0.8, 0.3, 0.3, 1]),
                                  _sk('Line Art', 'RGBA', [1, 1, 1, 1]),
                                  _sk('Opacity', 'VALUE', 1.0),
                                  _sk('Emission Strength', 'VALUE', 0.0)],
                                 {'compat': 'GENERIC', 'tones': 'TWO'}),
        'cartoon': _one_bsdf_graph('HALCYON_CartoonNode',
                                   [_sk('Paint Color', 'RGBA', [0.8, 0.3, 0.3, 1]),
                                    _sk('Opacity', 'VALUE', 1.0),
                                    _sk('Emission Strength', 'VALUE', 0.0)],
                                   {'shadow_mode': 'PAINTED'}),
    }
    for shadows in (False, True):
        counts = {'ok': 0, 'refused': 0, 'bad': 0}
        lit = 0
        for name, graph in graphs.items():
            for rate in ('PIXEL', 'VERTEX'):
                st = base_settings(24, 18, shadows=shadows, shading_rate=rate)
                sc = demo_scene(st)
                sc.materials[1].graph = graph
                srcs, why = _pass_sources(sc, st)
                lit += _shadowed(srcs)
                counts[_record(f'{name} master at {rate}{_SHADOWS[shadows]}',
                               srcs, why)] += 1
        print(f'  cel masters{_SHADOWS[shadows]}: {counts}')
        if shadows:
            _shadows_seen('cel masters', lit)


def _spot(k=0):
    from ..core.scene import Light
    return Light(type='SPOT', name=f'Spot {k}', position=(0.5 - k, -3.0, 5.0),
                 direction=(-0.1, 0.5, -0.85), color=(1.0, 0.9, 0.8),
                 energy=300.0, shadow='MAP', spot_size=1.0)


def _area(shape):
    from ..core.scene import Light
    return Light(type='AREA', name='Area', position=(0.0, -1.0, 5.0),
                 direction=(0.0, 0.0, -1.0), energy=500.0, shadow='RAY',
                 area_size=(1.5, 1.0), area_shape=shape)


def _kind_soft(sc):
    sc.lights[0].radius = 0.05          # a SUN's size is an angle
    sc.lights[1].radius = 0.3           # a POINT's a sphere


def _kind_point_traced(sc):
    sc.lights[1].shadow = 'RAY'


#: (label, settings, scene edit, mapped functions, traced functions, soft)
#: -- the demo scene's SUN and POINT under each kind of shadow a lamp can
#: cast into a deferred pass, and what every pass must then define
SHADOW_KINDS = (
    ('two mapped lamps, one PCF tap', {'shadow_samples': 1}, None, 2, 0, False),
    ('two mapped lamps, nine PCF taps', {'shadow_samples': 9}, None, 2, 0, False),
    ('two mapped lamps, sixteen PCF taps', {'shadow_samples': 16}, None, 2, 0, False),
    ('two mapped lamps, softness 0', {'shadow_softness': 0.0}, None, 2, 0, False),
    ('the MIDPOINT map', {'shadow_map_depth': 'MIDPOINT'}, None, 2, 0, False),
    ("a SPOT's single map beside the SUN's and the POINT's cube", {},
     lambda sc: sc.lights.append(_spot()), 3, 0, False),
    ('eight mapped lamps on the one atlas', {},
     lambda sc: sc.lights.extend(_spot(k) for k in range(6)), 8, 0, False),
    ('two traced lamps, hard', {'shadow_default': 'RAY'}, None, 0, 2, False),
    ("two traced lamps, soft (a SUN's disc, a POINT's sphere)",
     {'shadow_default': 'RAY', 'shadow_samples': 4}, _kind_soft, 0, 2, True),
    ("a traced AREA lamp, soft over its rectangle",
     {'shadow_default': 'RAY', 'shadow_samples': 4},
     lambda sc: sc.lights.append(_area('SQUARE')), 0, 3, True),
    ("a traced AREA lamp, soft over its disk",
     {'shadow_default': 'RAY', 'shadow_samples': 4},
     lambda sc: sc.lights.append(_area('DISK')), 0, 3, True),
    ('a mapped SUN beside a traced POINT (Per Light)',
     {'shadow_default': 'PER_LIGHT'}, _kind_point_traced, 1, 1, False),
    ("a mapped SUN and SPOT beside a soft traced POINT (Per Light)",
     {'shadow_default': 'PER_LIGHT', 'shadow_samples': 4},
     lambda sc: (_kind_point_traced(sc), _kind_soft(sc),
                 sc.lights.append(_spot())), 2, 1, True),
)


def check_shadow_kinds():
    """R253 (review): the demo scene under each kind of shadow a lamp
    casts into a deferred pass (SHADOW_KINDS), at the pixel rate -- a
    vertex-rate pass carries no shadow term, the corners do. Each pass
    must DEFINE the functions its kind names: a kind that planned
    something else (a RAY lamp with no BVH on the job plans nothing) is
    a failure here, not a quiet pass. A refusal is a failure too: every
    one of these is a frame the deferred road draws."""
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
    for label, kw, edit, mapped, traced, soft in SHADOW_KINDS:
        st = base_settings(24, 18, shadows=True, shading_rate='PIXEL', **kw)
        sc = demo_scene(st)
        if edit is not None:
            edit(sc)
        srcs, why = _pass_sources(sc, st)
        label = 'shadows: ' + label
        got = _record(label, srcs, why)
        if got == 'refused':
            FAILS.append(f'{label}: refused ({why})')
            print(f'  FAIL {label}\n       refused: {why}')
        elif got == 'ok':
            seen = sorted({(_shadow_fns(src), _shadow_fns(src, traced=True),
                            'hal_smp_hash3(sx' in src) for src in srcs})
            if seen != [(mapped, traced, soft)]:
                got = 'bad'
                FAILS.append(f'{label}: the passes define {seen} (mapped, '
                             f'traced, soft), not {(mapped, traced, soft)}')
                print(f'  FAIL {label}\n       the passes define {seen}, '
                      f'not {(mapped, traced, soft)}')
        counts[got] += 1
    print(f'  shadow kinds: {counts}')


def check_stages():
    """The post stages' RAW sources (gpu/stages*.py STAGES, with their own
    declarations -- body() strips those for CreateInfo)."""
    from ..gpu import stages as STG
    counts = {'ok': 0, 'bad': 0}
    for name in sorted(STG.STAGES):
        counts[_record(f'stage {name}', [STG.STAGES[name]], None)] += 1
    print(f'  post stages: {counts}')


def check_sky_and_kernels():
    """The sky pass source as the device receives it (gpu/sky.SOURCE, the
    accessors filled) and the rasteriser's fragment sources (gpu/craster;
    its compute variants are check_compute_raster's)."""
    from ..gpu import sky as GSKY
    from ..gpu import craster as CR
    counts = {'ok': 0, 'bad': 0}
    counts[_record('sky pass', [GSKY.SOURCE], None)] += 1
    for cvg in (False, True):
        counts[_record(f'craster fragment cvg={cvg}', [CR.fragment_source(cvg)],
                       None)] += 1
    print(f'  sky and raster fragments: {counts}')


def compute_raster_unit(lin, clear, cvg):
    """The compute raster as the driver compiles it: the interface the
    device's CreateInfo declares (compile_compute: 8x8 groups, RGBA32F
    write images and samplers by index, float push constants -- plain
    uniforms on the OpenGL backend) over the stripped source."""
    from ..gpu import craster as CR
    from ..gpu import device as DV
    images = ('hal_out_ids', 'hal_out_aux') \
        + (('hal_out_lin',) if lin else ()) \
        + (('hal_out_cvg',) if cvg else ())
    samplers = ('hal_rc', 'hal_rbins', 'hal_rtiles') \
        + (('hal_rclear',) if clear else ())
    head = ['layout(local_size_x = 8, local_size_y = 8) in;']
    head += [f'layout(binding = {i}, rgba32f) uniform writeonly image2D {nm};'
             for i, nm in enumerate(images)]
    head += [f'layout(binding = {i}) uniform sampler2D {nm};'
             for i, nm in enumerate(samplers)]
    head += [f'uniform float {nm};' for nm in CR.RASTER_FLOATS]
    return '\n'.join(head) + '\n' + DV.strip_declarations(
        CR.compute_source(lin, clear, cvg))


def check_compute_raster():
    """R253 (field): the compute raster in every variant, as a compute
    shader with exactly the interface raster_on_device declares."""
    from ..gpu import craster as CR
    counts = {'ok': 0, 'bad': 0}
    for lin in (False, True):
        for clear in (False, True):
            for cvg in (False, True):
                counts[_record(
                    f'craster compute {CR.compute_variant_name(lin, clear, cvg)}',
                    [compute_raster_unit(lin, clear, cvg)], None,
                    stage='comp')] += 1
    print(f'  compute raster: {counts}')


def check_coded():
    """R253 (review): the Coded Shader's pass under the names a node can
    carry -- the real node class through the real exporter
    (test_r253_review's stand-ins), two coded shaders summed into the
    master. A refusal is a failure HERE: every one of these names is
    meant to reach the driver. Shadows off and on: the coded floor's own
    pass carried the twice-declared atlas sampler under the demo lamps,
    which is why this check planned with shadows off until the
    declaration was fixed. Runs last: it installs the bpy stub."""
    import importlib
    from . import fakebpy
    fakebpy.install()
    from . import test_r253_review as RV
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    for shadows in (False, True):
        counts = {'ok': 0, 'refused': 0, 'bad': 0}
        lit = 0
        for pair in (('Coded Shader', 'Other'), ('Coded Shader', 'Coded_Shader'), ('Coded', 'Coded B'),
                     ('Señal', 'Other'), ('Señal', 'Seu00F1al'), ('Шейдер', '色彩'), ('色', '彩'),
                     ('層３', '\U0001d6fc glow')):
            graph, programs, _warns = RV._code_exported(RV._code_two(SN, *pair))
            src, _binds, why = RV._code_pass(graph, programs, shadows=shadows, size=(24, 18))
            label = 'coded shaders %r beside %r' % pair + _SHADOWS[shadows]
            srcs = None if src is None else [src]
            lit += _shadowed(srcs)
            got = _record(label, srcs, why)
            if got == 'refused':
                FAILS.append(f'{label}: refused ({why})')
                print(f'  FAIL {label}\n       refused: {why}')
            counts[got] += 1
        print(f'  coded shaders by node name{_SHADOWS[shadows]}: {counts}')
        if shadows:
            _shadows_seen('coded shaders by node name', lit)


def check_emission_route():
    """R253 (review): the EMISSION SLOT ROUTING's passes -- a colour, a
    value, a vector, a Coded Shader's Color and an Image Texture wired
    straight into Surface, an Emission node, and the Mix / Add Shader
    over emissions with a linked and with a BARE side (the one new
    literal the route writes) -- at every shading rate, shadows off and
    on. A refusal is a failure HERE: each of these is meant to reach the
    driver, where the simulator alone had judged its text. Runs after
    check_coded (the review module's builders want the bpy stub)."""
    from . import fakebpy
    fakebpy.install()
    from ..core.scene import Material
    from . import test_r253_review as RV

    def cases():
        for kind in ('colour', 'value', 'vector', 'coded'):
            nodes, link, prog = RV._wired_source(kind)
            yield f'a {kind} wired into Surface', RV._wired_out(nodes, link), prog, False
        yield ('an Image Texture wired into Surface', RV._wired_out(
            {'src': {'id': 'src', 'bl_idname': 'ShaderNodeTexImage',
                     'props': {'image': 'checker', 'interpolation': 'Linear', 'extension': 'REPEAT'},
                     'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0])],
                     'outputs': [{'name': 'Color', 'type': 'RGBA'}, {'name': 'Alpha', 'type': 'VALUE'}]}},
            ['src', 0]), {}, True)
        e1 = RV._em_node('e1', (0.9, 0.3, 0.1), 2.0)
        e2 = RV._em_node('e2', (0.1, 0.8, 0.3), 0.5)
        yield 'an Emission node', RV._wired_out(dict(e1), ['e1', 0]), {}, False
        for idname in ('ShaderNodeMixShader', 'ShaderNodeAddShader'):
            yield (f'{idname} over two Emissions', RV._wired_out(
                dict(e1, **e2, **RV._shader_pair(idname, ['e1', 0], ['e2', 0], 0.3)), ['two', 0]), {}, False)
            yield (f'{idname} over an Emission and a bare socket', RV._wired_out(
                dict(e1, **RV._shader_pair(idname, ['e1', 0], None, 0.3)), ['two', 0]), {}, False)

    for shadows in (False, True):
        counts = {'ok': 0, 'refused': 0, 'bad': 0}
        lit = 0
        for what, graph, programs, textured in cases():
            for rate in ('PIXEL', 'VERTEX', 'FACE'):
                st = base_settings(24, 18, shadows=shadows, shading_rate=rate)
                sc = demo_scene(st, with_texture=textured)
                mat = Material(name='Wired', index=1, graph=graph)
                mat.programs = programs
                sc.materials[1] = mat
                srcs, why = _pass_sources(sc, st)
                label = f'emission route: {what} at {rate}{_SHADOWS[shadows]}'
                lit += _shadowed(srcs)
                got = _record(label, srcs, why)
                if got == 'refused':
                    FAILS.append(f'{label}: refused ({why})')
                    print(f'  FAIL {label}\n       refused: {why}')
                counts[got] += 1
        print(f'  emission slot routing{_SHADOWS[shadows]}: {counts}')
        if shadows:
            _shadows_seen('emission slot routing', lit)


def mapped_shadow_cases():
    """(label, light, settings overrides): one mapped lamp per kind, with
    and without a size -- a sized lamp's term is the per-fragment
    penumbra lookup (gpu/softmap.py), a lamp of size 0 the constant PCF."""
    from ..core.scene import Light
    cases = []
    for kind, sizes in (('POINT', (0.0, 0.3)), ('SPOT', (0.0, 0.3)),
                        ('SUN', (0.0, 0.05)), ('AREA', (0.5,))):
        for size in sizes:
            kw = dict(type=kind, name='L', position=(1.5, 2.5, 4.0),
                      direction=(-0.3, -0.5, -0.8), energy=3.0 if kind == 'SUN'
                      else 900.0, shadow='MAP')
            if kind == 'AREA':
                l = Light(area_size=(size, size * 0.5), area_x=(1, 0, 0),
                          area_y=(0, 1, 0), **kw)
            else:
                l = Light(radius=size, **kw)
            if kind == 'SPOT':
                l.spot_size = 1.2
            cases.append((f'{kind} size {size}', l, {}))
            if size > 0.0:
                cases.append((f'{kind} size {size} midpoint, 16 samples', l,
                              {'shadow_map_depth': 'MIDPOINT',
                               'shadow_samples': 16}))
            else:
                # R254 map-gpu: Softness above a texel sends a size-0
                # lamp through the sized lookup with no lamp term
                cases.append((f'{kind} size 0, Softness 4', l,
                              {'shadow_softness': 4.0}))
    return cases


def traced_shadow_cases():
    """(label, light): the traced shadow functions and the Area form
    factor, whose sizes ride hal_lights texels since R254 map-gpu."""
    from ..core.scene import Light
    cases = []
    for kind, size, extra in (('POINT', 0.0, {'shadow_density': 0.6}),
                              ('POINT', 0.3, {}), ('SPOT', 0.3, {}),
                              ('SUN', 0.1, {}), ('SUN', 2.0, {}),
                              ('AREA', 0.8, {}),
                              ('AREA', 0.8, {'area_shape': 'DISK',
                                             'area_gamma': 0.8}),
                              ('POINT', 0.3, {'shadow_color':
                                              (0.2, 0.1, 0.0),
                                              'shadow_density': 0.5})):
        kw = dict(type=kind, name='L', position=(1.5, 2.5, 4.0),
                  direction=(-0.3, -0.5, -0.8), energy=3.0 if kind == 'SUN'
                  else 900.0, shadow='RAY')
        if kind == 'AREA':
            l = Light(area_size=(size, size * 0.5), area_x=(1, 0, 0),
                      area_y=(0, 1, 0), **kw)
        else:
            l = Light(radius=size, **kw)
        if kind == 'SPOT':
            l.spot_size = 1.2
        for k, v in extra.items():
            setattr(l, k, v)
        cases.append((f'{kind} size {size} {extra or ""}'.strip(), l))
    return cases


def _both_value_roads(label, plan):
    """Compile the plan `plan()` makes on the texel road (the shadow
    values fetched from hal_lights) and on the literal road (the same
    float32 values printed), the suite's A/B of gpu/shade.SHADOW_TEXELS."""
    out = []
    for flag, road in ((True, 'texels'), (False, 'literals')):
        old = GSH.SHADOW_TEXELS
        GSH.SHADOW_TEXELS = flag
        try:
            srcs, why = plan()
        finally:
            GSH.SHADOW_TEXELS = old
        out.append(_record(f'{label} ({road})', srcs, why))
    return out


def check_mapped_shadows():
    """The shadow-map terms (the model sweep above plans with shadows
    off, so no shadow function reached the front-end before R254), on
    both value roads."""
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
    for label, light, over in mapped_shadow_cases():
        st = base_settings(24, 18, shadows=True, shadow_default='MAP',
                           shadow_map_size=64, **over)
        sc = demo_scene(st, with_texture=False)
        sc.lights = [light]
        R.render(sc, st)                    # bakes the map the plan packs
        for got in _both_value_roads(f'mapped shadow, {label}',
                                     lambda: _pass_sources(sc, st)):
            counts[got] += 1
    print(f'  mapped shadows: {counts}')


def check_traced_shadows():
    """The traced shadow functions (hard, soft, a Sun's two laws, an
    Area's rectangle and ellipse) and the Area form factor, on both
    value roads."""
    from ..core.bvh import BVH
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
    for label, light in traced_shadow_cases():
        st = base_settings(24, 18, shadows=True, shadow_default='PER_LIGHT')
        sc = demo_scene(st, with_texture=False)
        sc.lights = [light]

        def plan(sc=sc, st=st):
            st.render_device = 'GPU'
            g, job = _job_for(sc, st)
            job = R.ShadeJob(sc, st, job.textures,
                             BVH.cached(sc.mesh.verts, sc.mesh.tris),
                             job.view, job.eye, job.width, job.height)
            GSH._PLAN_CACHE.clear()
            passes, why, _atl = GSH.plan_frame(job, g)
            if passes is None:
                return None, why
            return [x for ps in passes for x in ps
                    if isinstance(x, str) and 'void main' in x], None
        for got in _both_value_roads(f'traced shadow, {label}', plan):
            counts[got] += 1
    print(f'  traced shadows: {counts}')


def main():
    utf8_console()
    FAILS.clear()
    _JUDGED.clear()
    if _glslang() is None:
        print('glsl_strict: HALCYON_GLSLANG not set or not found -- the strict '
              'GLSL check did NOT run (build glslang and point the variable at '
              'StandAlone/glslang)')
        return 0
    print(f'glsl_strict: {_glslang()}')
    for fn in (check_models, check_console, check_masters, check_shadow_kinds,
               check_stages, check_sky_and_kernels, check_mapped_shadows,
               check_traced_shadows, check_compute_raster, check_coded,
               check_emission_route):
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILS.append(f'{fn.__name__} raised {exc}')
    print()
    if FAILS:
        print(f'{len(FAILS)} shader(s) a strict GLSL front-end rejects:')
        for f in FAILS:
            print('  ' + f)
        return 1
    print(f'every shader compiles under strict GLSL 4.30 ({len(_JUDGED)} '
          'distinct texts)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
