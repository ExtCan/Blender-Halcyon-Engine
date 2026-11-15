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
Not covered: the ray kernels (gpu/rtrace, their push constants are listed
at each call) and the ink pass (built per frame from a job).
"""
import os
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


def _pass_sources(sc, st):
    """The fragment sources of every material pass the plan makes, or
    (None, why) when the plan refuses the frame by name."""
    st.render_device = 'GPU'
    g, job = _job_for(sc, st)
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
        ok, msg = compile_source(src, stage)
        if not ok:
            FAILS.append(f'{label}: {msg[:400]}')
            print(f'  FAIL {label}\n       {msg[:300]}')
            return 'bad'
    return 'ok'


def check_models():
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
    for ident, _l, _d in SH.MODEL_ITEMS:
        for rate in ('PIXEL', 'VERTEX', 'FACE'):
            st = base_settings(24, 18, shadows=False, force_model=ident,
                               shading_rate=rate)
            srcs, why = _pass_sources(demo_scene(st), st)
            counts[_record(f'model {ident} at {rate}', srcs, why)] += 1
    print(f'  engine models: {counts}')


def check_console():
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
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
                        srcs, why = _pass_sources(sc, st)
                        counts[_record(f'console {con} {prop}={props[prop]} '
                                       f'{rate}/{shading}', srcs, why)] += 1
    print(f'  console types and options: {counts}')


def check_masters():
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
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
    for name, graph in graphs.items():
        for rate in ('PIXEL', 'VERTEX'):
            st = base_settings(24, 18, shadows=False, shading_rate=rate)
            sc = demo_scene(st)
            sc.materials[1].graph = graph
            srcs, why = _pass_sources(sc, st)
            counts[_record(f'{name} master at {rate}', srcs, why)] += 1
    print(f'  cel masters: {counts}')


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
    meant to reach the driver. Runs last: it installs the bpy stub."""
    import importlib
    from . import fakebpy
    fakebpy.install()
    from . import test_r253_review as RV
    SN = importlib.import_module('halcyon.nodes.shader_nodes')
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
    for pair in (('Coded Shader', 'Other'), ('Coded Shader', 'Coded_Shader'), ('Coded', 'Coded B'),
                 ('Señal', 'Other'), ('Señal', 'Seu00F1al'), ('Шейдер', '色彩'), ('色', '彩'),
                 ('層３', '\U0001d6fc glow')):
        graph, programs, _warns = RV._code_exported(RV._code_two(SN, *pair))
        src, _binds, why = RV._code_pass(graph, programs, shadows=False, size=(24, 18))
        label = 'coded shaders %r beside %r' % pair
        got = _record(label, None if src is None else [src], why)
        if got == 'refused':
            FAILS.append(f'{label}: refused ({why})')
            print(f'  FAIL {label}\n       refused: {why}')
        counts[got] += 1
    print(f'  coded shaders by node name: {counts}')


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
    return cases


def check_mapped_shadows():
    """The shadow-map terms (the model sweep above plans with shadows
    off, so no shadow function reached the front-end before R254)."""
    counts = {'ok': 0, 'refused': 0, 'bad': 0}
    for label, light, over in mapped_shadow_cases():
        st = base_settings(24, 18, shadows=True, shadow_default='MAP',
                           shadow_map_size=64, **over)
        sc = demo_scene(st, with_texture=False)
        sc.lights = [light]
        R.render(sc, st)                    # bakes the map the plan packs
        srcs, why = _pass_sources(sc, st)
        counts[_record(f'mapped shadow, {label}', srcs, why)] += 1
    print(f'  mapped shadows: {counts}')


def main():
    utf8_console()
    FAILS.clear()
    if _glslang() is None:
        print('glsl_strict: HALCYON_GLSLANG not set or not found -- the strict '
              'GLSL check did NOT run (build glslang and point the variable at '
              'StandAlone/glslang)')
        return 0
    print(f'glsl_strict: {_glslang()}')
    for fn in (check_models, check_console, check_masters, check_stages,
               check_sky_and_kernels, check_mapped_shadows,
               check_compute_raster, check_coded):
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
    print('every shader compiles under strict GLSL 4.30')
    return 0


if __name__ == '__main__':
    sys.exit(main())
