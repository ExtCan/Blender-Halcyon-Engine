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
reach a driver). Not covered: the compute kernels (their uniforms are
declared by the device's CreateInfo, so the text alone is not a unit) and
the ink pass (built per frame from a job).
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
from ..core import shading as SH
from ..gpu import shade as GSH

FAILS = []


def _glslang():
    path = os.environ.get('HALCYON_GLSLANG', '')
    return path if path and os.path.exists(path) else None


def compile_source(src, stage='frag'):
    """(ok, message) from glslang on `src` with a 4.30 version line."""
    with tempfile.NamedTemporaryFile('w', suffix=f'.{stage}', delete=False) as fh:
        fh.write('#version 430 core\n' + src)
        path = fh.name
    try:
        r = subprocess.run([_glslang(), '-S', stage, path], capture_output=True,
                           text=True)
        return r.returncode == 0, (r.stdout + r.stderr).strip()
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
                                  _sk('Line Art', 'RGBA', [1, 1, 1, 1])],
                                 {'compat': 'GENERIC', 'tones': 'TWO'}),
        'cartoon': _one_bsdf_graph('HALCYON_CartoonNode',
                                   [_sk('Paint Color', 'RGBA', [0.8, 0.3, 0.3, 1])],
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
    accessors filled) and the rasteriser's fragment and compute sources
    in every variant (gpu/craster)."""
    from ..gpu import sky as GSKY
    from ..gpu import craster as CR
    counts = {'ok': 0, 'bad': 0}
    counts[_record('sky pass', [GSKY.SOURCE], None)] += 1
    for cvg in (False, True):
        counts[_record(f'craster fragment cvg={cvg}', [CR.fragment_source(cvg)],
                       None)] += 1
    # (the compute kernels are built for CreateInfo with their uniforms
    # declared by the device, so their text alone is not a compilable
    # unit; they are field-proven on the driver and not checked here)
    print(f'  sky and raster fragments: {counts}')


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
               check_sky_and_kernels):
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
