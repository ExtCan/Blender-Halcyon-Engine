"""R253 (1.92.0): the compositing passes -- Mist, Env, Beauty, the
Blender Internal light split (Diffuse, Spec, Ambient, Emit, Shadow, AO,
Color) and the per-lamp passes (Light00..Light07).

Every heading is proven on the demo scene through core.render first (the
sum identity, the Shadow / AO / Mist / Color contents, the no-new-
arithmetic proof with every pass on against every pass off), then at the
device seam (gpu.shade.plan_frame refuses the light split BY NAME, the
plan cache cannot walk past the flag, the GPU device lands on the CPU's
bits), then through the fake engine (Beauty delivered as the linear
frame beside a palette-quantised Combined), and finally at the text
level (the panel note, the capability table, the declared pass table).

    python -m halcyon.tests.test_r253_compositing
"""
import dataclasses
import sys
import traceback

import numpy as np

from . import utf8_console
from .test_render import base_settings, _engine_module
from .scenebuild import demo_scene
from ..core import post
from ..core import raster
from ..core import render as R
from ..core.scene import Material
from ..core.settings import RenderSettings
from ..gpu import capability as CAP
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32

#: the four that sum to the beauty, in BI's own names
SPLIT = ('pass_diffuse', 'pass_specular', 'pass_ambient', 'pass_emission')
ALL_NEW = ('pass_mist', 'pass_environment', 'pass_beauty', 'pass_diffuse',
           'pass_specular', 'pass_ambient', 'pass_emission', 'pass_shadow',
           'pass_ao', 'pass_color', 'pass_lights')


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def _settings(w=96, h=72, **kw):
    """The demo scene's settings with the identity's preconditions: no
    fog, no per-lamp clamp, no traced reflections (each is a stated
    non-separable term)."""
    st = base_settings(w, h, **kw)
    st.fog = False
    st.light_clamp = 0.0
    st.raytrace = False
    st.output_scale = 'NONE'
    return st


def _sk(name, kind, default, link=None):
    return {'name': name, 'type': kind, 'default': default, 'link': link}


def _bi_graph(props, ins_extra=None, color=(0.55, 0.55, 0.6, 1.0)):
    """A Blender Internal material node graph, exactly the exporter's shape
    (the shape tests/test_render's BI panel round uses)."""
    p = {'diff_shader': 'LAMBERT', 'spec_shader': 'COOKTORR',
         'shadeless': False}
    p.update(props)
    ins = [_sk('Color', 'RGBA', list(color)),
           _sk('Intensity', 'VALUE', 0.8),
           _sk('Specular Intensity', 'VALUE', 0.4),
           _sk('Hardness', 'VALUE', 50.0),
           _sk('Emit', 'VALUE', 0.0)]
    for e in (ins_extra or []):
        ins.append(e)
    return {'output': 'out', 'nodes': {
        'bi': {'id': 'bi', 'bl_idname': 'HALCYON_BIMaterialNode',
               'props': p, 'inputs': ins,
               'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['bi', 0])],
                'outputs': []}}}


def _covered(passes):
    return passes['Depth'][..., 0] < 1e9


# --------------------------------------------------------- the sum identity


def test_light_passes_sum_to_the_beauty():
    """Diffuse + Spec + Ambient + Emit is the linear beauty at every
    covered pixel (the split reads the very arrays `out` receives), and
    with Env added the identity holds at EVERY pixel -- sky included,
    through the supersample filter, because the colour passes resolve
    through the beauty's own kernel."""
    for aa in (1, 4):
        st = _settings(aa_samples=aa)
        for k in SPLIT + ('pass_environment', 'pass_depth'):
            setattr(st, k, True)
        sc = demo_scene(st)
        img = R.render(sc, st)
        p = sc.last_passes
        missing = [n for n in ('Diffuse', 'Spec', 'Ambient', 'Emit', 'Env',
                               'Depth') if n not in p]
        check(f'aa={aa}: the split and Env are produced', not missing,
              ', '.join(missing))
        if missing:
            continue
        cov = _covered(p)
        total = p['Diffuse'] + p['Spec'] + p['Ambient'] + p['Emit']
        # fully covered: no sky sample under the filter (Env exactly 0)
        full = cov & (np.abs(p['Env']).max(axis=2) == 0.0)
        e_cov = float(np.abs(total - img[..., :3])[full].max())
        check(f'aa={aa}: Diffuse + Spec + Ambient + Emit == beauty at '
              'covered pixels', full.any() and e_cov < 1e-4,
              f'max {e_cov:.2e} over {int(full.sum())} px')
        e_all = float(np.abs(total + p['Env'] - img[..., :3]).max())
        check(f'aa={aa}: ... and + Env == beauty at every pixel',
              e_all < 1e-4, f'max {e_all:.2e}')
        if aa == 1:
            check('Env is the frame at uncovered pixels',
                  float(np.abs(p['Env'] - img[..., :3])[~cov].max()) == 0.0
                  and float(p['Env'][~cov].max()) > 0.01)
            check('Env is black under geometry',
                  float(np.abs(p['Env'][cov]).max()) == 0.0)
        check(f'aa={aa}: the passes are at the output resolution',
              all(v.shape[:2] == (st.resolution_y, st.resolution_x)
                  for v in p.values()),
              str({k: v.shape for k, v in p.items()}))
        check(f'aa={aa}: the split carries real light',
              float(p['Diffuse'][cov].max()) > 0.1
              and float(p['Spec'][cov].max()) > 0.05
              and float(p['Ambient'][cov].max()) > 0.001)


def test_light_passes_are_bitwise_neutral():
    """The no-new-arithmetic proof: the frame with EVERY new pass on is
    bit for bit the frame with them off -- on the plain demo scene, on a
    variant with traced soft shadows, ambient occlusion and raytracing,
    and on BI materials whose extras dict the split now shares (a RESULT
    ramp, Transparency > Specular, a Shadows Only catcher)."""
    variants = {
        'demo': dict(),
        'soft shadows + AO + raytrace': dict(
            shadow_default='RAY', shadow_samples=3, ambient_occlusion=True,
            ao_samples=4, raytrace=True, ray_depth=1),
    }
    for label, kw in variants.items():
        st_off = _settings(64, 48, **kw)
        st_on = st_off.copy()
        for k in ALL_NEW:
            setattr(st_on, k, True)
        off = R.render(demo_scene(st_off), st_off)
        sc = demo_scene(st_on)
        on = R.render(sc, st_on)
        check(f'{label}: every pass on == every pass off, bitwise',
              bool(np.array_equal(off, on)),
              f'max {float(np.abs(off - on).max()):.3e}')
        check(f'{label}: ... and the passes exist',
              bool(sc.last_passes) and 'Diffuse' in sc.last_passes)

    # the BI extras seam: materials that already asked for an extras dict
    # (spectra, shadows only) and one that keeps diff_acc / spec_acc
    # apart (a RESULT ramp) -- the split rides the same dict
    ramp = {'use_ramp_dif': True, 'ramp_dif_input': 'RESULT',
            'ramp_dif_blend': 'MIX', 'ramp_dif_factor': 1.0,
            'ramp_dif_stops': [(0.0, 0.1, 0.0, 0.3, 1.0),
                               (1.0, 1.0, 0.9, 0.6, 1.0)]}
    spectra = ({'use_transparency': True, 'transp_mode': 'Z_TRANSPARENCY'},
               [_sk('Opacity', 'VALUE', 0.7),
                _sk('Transp Fresnel', 'VALUE', 0.0),
                _sk('Transp Blend', 'VALUE', 1.25),
                _sk('Transp Specular', 'VALUE', 0.8)])
    catcher = {'shadow_only': True}
    for label, floor, box in (('RESULT ramp', ramp, None),
                              ('Transparency > Specular', None, spectra),
                              ('Shadows Only catcher', catcher, None)):
        frames = []
        for flip in (False, True):
            st = _settings(64, 48)
            if flip:
                for k in ALL_NEW:
                    setattr(st, k, True)
            sc = demo_scene(st, with_texture=False)
            if floor is not None:
                sc.materials[0] = Material(name='FloorBI', index=0,
                                           graph=_bi_graph(floor))
            if box is not None:
                sc.materials[2] = Material(name='BoxBI', index=2,
                                           graph=_bi_graph(*box))
            frames.append(R.render(sc, st))
            if flip:
                p = sc.last_passes or {}
        check(f'BI {label}: every pass on == off, bitwise',
              bool(np.array_equal(frames[0], frames[1])),
              f'max {float(np.abs(frames[0] - frames[1]).max()):.3e}')
        check(f'BI {label}: the split is produced', 'Diffuse' in p
              and bool(np.isfinite(p['Diffuse']).all()))

    # and the defaults ask for nothing: an old scene's settings carry no
    # new pass, wanted_passes is empty, and a preset dict without the
    # fields leaves them off (apply ignores what it does not know)
    st = RenderSettings()
    check('every new pass defaults off',
          not any(getattr(st, k) for k in ALL_NEW)
          and R.wanted_passes(st) == () and R.light_pass_names(st) == ())
    st.apply({'pass_depth': True})
    check('an older preset dict leaves the new passes off',
          R.wanted_passes(st) == ('Depth',))


# ------------------------------------------------------------- the contents


def test_shadow_pass_content():
    """1 lit, 0 in shadow, averaged over the lamps that shadow; three
    equal channels; all 1.0 with Shadows off; the ball's shadow on the
    floor reads dark."""
    st = _settings(pass_shadow=True, pass_depth=True, pass_object_index=True)
    sc = demo_scene(st)
    R.render(sc, st)
    p = sc.last_passes
    sh = p['Shadow']
    cov = _covered(p)
    check('Shadow values sit in 0..1',
          float(sh.min()) >= 0.0 and float(sh.max()) <= 1.0,
          f'{float(sh.min()):.3f}..{float(sh.max()):.3f}')
    check('the three channels are equal',
          bool(np.array_equal(sh[..., 0], sh[..., 1]))
          and bool(np.array_equal(sh[..., 1], sh[..., 2])))
    floor = cov & (p['IndexOB'][..., 0] == 0)
    check('the floor under the ball holds shadow (< 0.5)',
          floor.any() and float(sh[..., 0][floor].min()) < 0.5,
          f'min {float(sh[..., 0][floor].min()):.3f}')
    check('lit floor reads 1.0',
          float(sh[..., 0][floor].max()) >= 0.999)
    check('the sky reads lit (1.0)', float(sh[~cov].min()) == 1.0)

    st2 = _settings(pass_shadow=True, pass_depth=True, shadows=False)
    sc2 = demo_scene(st2)
    R.render(sc2, st2)
    check('with Shadows off the pass is all 1.0',
          float(sc2.last_passes['Shadow'].min()) == 1.0)


def test_mist_pass_content():
    """BI's mist: the scene Fog curve on the camera distance -- 1 at the
    sky, monotone with Depth, and read from the curve, not the toggle."""
    st = _settings(pass_mist=True, pass_depth=True)
    st.fog_start, st.fog_end = 4.0, 12.0
    sc = demo_scene(st)
    R.render(sc, st)
    p = sc.last_passes
    m = p['Mist'][..., 0]
    d = p['Depth'][..., 0]
    cov = _covered(p)
    check('Mist is one channel at the output size',
          p['Mist'].shape == (72, 96, 1))
    check('uncovered pixels read 1.0', float(m[~cov].min()) == 1.0)
    check('covered values sit in 0..1',
          float(m[cov].min()) >= 0.0 and float(m[cov].max()) <= 1.0)
    check('the near edge is clearer than the far edge',
          float(m[cov].min()) < 0.5 and float(m[cov].max()) > 0.5,
          f'{float(m[cov].min()):.3f}..{float(m[cov].max()):.3f}')
    order = np.argsort(d[cov], kind='stable')
    ms = m[cov][order]
    check('Mist is non-decreasing in Depth',
          float(np.diff(ms).min()) >= -1e-6 if ms.size > 1 else True)
    # the curve itself: LINEAR -> (d - start) / (end - start), clipped
    expect = np.clip((d[cov] - 4.0) / 8.0, 0.0, 1.0)
    check('LINEAR mist is (d - start) / (end - start), clipped',
          float(np.abs(m[cov] - expect).max()) < 1e-5,
          f'max {float(np.abs(m[cov] - expect).max()):.2e}')

    st2 = st.copy()
    st2.fog = True
    sc2 = demo_scene(st2)
    R.render(sc2, st2)
    check('Fog on vs off produces the same Mist bytes',
          bool(np.array_equal(sc2.last_passes['Mist'], p['Mist'])))

    st3 = st.copy()
    st3.fog_mode = 'EXP'
    st3.fog_density = 0.15
    sc3 = demo_scene(st3)
    R.render(sc3, st3)
    m3 = sc3.last_passes['Mist'][..., 0]
    expect3 = 1.0 - np.exp(-0.15 * d[cov])
    check('EXP mist follows the EXP curve',
          float(np.abs(m3[cov] - expect3).max()) < 1e-4)


def test_ao_and_color_passes():
    """AO is the ambient-occlusion factor itself (white when AO is off);
    Color is the material colour before any light."""
    st = _settings(pass_ao=True, pass_color=True, pass_depth=True,
                   pass_material_index=True)
    sc = demo_scene(st)
    R.render(sc, st)
    p = sc.last_passes
    cov = _covered(p)
    check('with Ambient Occlusion off AO == 1.0 everywhere',
          float(p['AO'].min()) == 1.0 and float(p['AO'].max()) == 1.0)
    ball = cov & (p['IndexMA'][..., 0] == 1)
    col = p['Color'][ball]
    check("Color at the Ball's pixels is its diffuse (0.85, 0.2, 0.15)",
          ball.any() and float(np.abs(col - np.array(
              [0.85, 0.2, 0.15], f32)[None, :]).max()) < 1e-5,
          str(col[:1]))
    check('Color is black at the sky', float(np.abs(p['Color'][~cov]).max())
          == 0.0)

    st2 = _settings(pass_ao=True, pass_depth=True, ambient_occlusion=True,
                    ao_samples=6)
    sc2 = demo_scene(st2)
    img = R.render(sc2, st2)
    p2 = sc2.last_passes
    ao = p2['AO']
    cov2 = _covered(p2)
    check('with AO on the factor darkens somewhere and never exceeds 1',
          float(ao[cov2].min()) < 0.98 and float(ao.max()) <= 1.0,
          f'{float(ao[cov2].min()):.3f}..{float(ao.max()):.3f}')
    check('the sky reads open (1.0)', float(ao[~cov2].min()) == 1.0)
    check('the frame still renders finite with AO on',
          bool(np.isfinite(img).all()))


def test_per_light_passes_sum_to_the_lamps():
    """Light00 + Light01 == Diffuse + Spec on the two-lamp demo (the lamp
    slots take the same pre-clamp sums); Light02..07 stay black; PASS_SPEC
    carries exactly LIGHT_PASS_SLOTS Light names."""
    st = _settings(pass_lights=True, pass_diffuse=True, pass_specular=True,
                   pass_depth=True)
    sc = demo_scene(st)
    R.render(sc, st)
    p = sc.last_passes
    names = [f'Light{i:02d}' for i in range(R.LIGHT_PASS_SLOTS)]
    check('all eight lamp passes are produced',
          all(n in p for n in names))
    cov = _covered(p)
    two = p['Light00'] + p['Light01']
    e = float(np.abs(two - p['Diffuse'] - p['Spec'])[cov].max())
    check('Light00 + Light01 == Diffuse + Spec', e < 1e-5, f'max {e:.2e}')
    check('Light00 (the key) and Light01 (the fill) both carry light',
          float(p['Light00'][cov].max()) > 0.1
          and float(p['Light01'][cov].max()) > 0.01)
    check('Light02..Light07 are black',
          all(float(np.abs(p[n]).max()) == 0.0 for n in names[2:]))
    ENG = _engine_module()
    lights = [n for n, _c, _i, _k in ENG.PASS_SPEC if n.startswith('Light')]
    check('PASS_SPEC carries exactly LIGHT_PASS_SLOTS Light names',
          lights == names, str(lights))
    check('the lamp order is the selection order',
          R.light_pass_names(st)[-R.LIGHT_PASS_SLOTS:] == tuple(names))


def test_rate_shaded_and_adaptive_frames_keep_the_identity():
    """A vertex-rate material reports its whole lit colour as Diffuse (the
    corners have no per-lobe split to interpolate); the adaptive AA
    refine passes average the colour passes with the frame."""
    st = _settings(64, 48, force_model='GOURAUD')
    for k in SPLIT + ('pass_depth', 'pass_shadow', 'pass_color'):
        setattr(st, k, True)
    sc = demo_scene(st)
    img = R.render(sc, st)
    p = sc.last_passes
    cov = _covered(p)
    check('GOURAUD: Diffuse is the whole lit colour',
          float(np.abs(p['Diffuse'] - img[..., :3])[cov].max()) == 0.0)
    check('GOURAUD: Spec / Ambient / Emit are black, Shadow open',
          float(p['Spec'].max()) == 0.0 and float(p['Ambient'].max()) == 0.0
          and float(p['Emit'].max()) == 0.0
          and float(p['Shadow'][cov].min()) == 1.0)
    check('GOURAUD: Color carries the albedo',
          float(p['Color'][cov].max()) > 0.1)

    st = _settings(64, 48, aa_mode='ADAPTIVE', aa_samples=4)
    for k in SPLIT + ('pass_environment', 'pass_depth'):
        setattr(st, k, True)
    sc = demo_scene(st)
    img = R.render(sc, st)
    p = sc.last_passes
    total = p['Diffuse'] + p['Spec'] + p['Ambient'] + p['Emit'] + p['Env']
    e = float(np.abs(total - img[..., :3]).max())
    check('ADAPTIVE: the refined colour passes still sum to the refined '
          'beauty', e < 1e-4 and int(R.LAST_ADAPTIVE.get('pixels', 0)) > 0,
          f'max {e:.2e}, {R.LAST_ADAPTIVE.get("pixels")} px refined')


# ------------------------------------------------------------ the GPU seam


def test_light_passes_refuse_the_gpu_by_name():
    """plan_frame refuses any light-component pass by name; the plan cache
    cannot reuse a valid plan once the flag flips (the flags are in the
    signature); the GPU device lands on the CPU's bits, passes included."""
    w, h = 64, 48
    st = _settings(w, h, shadows=False)
    sc = demo_scene(st, with_texture=False)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, _a = GSH.plan_frame(job, g)
    check('the plain demo frame plans on the GPU', passes is not None,
          str(why))
    for field, name in (('pass_diffuse', 'Diffuse'), ('pass_shadow', 'Shadow'),
                        ('pass_lights', 'Light00')):
        setattr(st, field, True)
        passes, why, _a = GSH.plan_frame(job, g)      # the cache is warm
        check(f'{field} refuses the GPU by name (through a warm cache)',
              passes is None and 'light-component passes' in str(why)
              and name in str(why), str(why))
        setattr(st, field, False)
    passes, why, _a = GSH.plan_frame(job, g)
    check('with the flags off the same frame plans again',
          passes is not None, str(why))
    for field in ('pass_mist', 'pass_environment', 'pass_beauty',
                  'pass_depth'):
        setattr(st, field, True)
    passes, why, _a = GSH.plan_frame(job, g)
    check('the frame passes (Mist, Env, Beauty, Depth) need no refusal',
          passes is not None, str(why))

    # the whole road: GPU device, no driver -> the same bytes, pass
    # buffers included; the verdict records that the GPU did not engage
    st_c = _settings(w, h)
    for k in ALL_NEW + ('pass_depth',):
        setattr(st_c, k, True)
    sc_c = demo_scene(st_c)
    cpu = R.render(sc_c, st_c)
    pc = sc_c.last_passes
    st_g = st_c.copy()
    st_g.render_device = 'GPU'
    sc_g = demo_scene(st_g)
    gpu = R.render(sc_g, st_g)
    pg = sc_g.last_passes
    check('the GPU device draws the CPU frame bit for bit',
          bool(np.array_equal(cpu, gpu)))
    check('... and every pass buffer bit for bit',
          set(pc) == set(pg)
          and all(bool(np.array_equal(pc[k], pg[k])) for k in pc))
    check('the verdict says the GPU did not engage',
          R.LAST_GPU_VERDICT.get('wanted') is True
          and R.LAST_GPU_VERDICT.get('engaged') is False
          and bool(R.LAST_GPU_VERDICT.get('why')))


def test_pass_panel_and_table_text():
    """The capability table names both tiers, the scene features name the
    split as blocking, the UI note no longer claims mist / light
    components are not produced, and Blender's own Passes panel stays
    unforced (Vector and Denoising Data are still not produced)."""
    import inspect
    import os

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, 'ui.py'), encoding='utf8') as fh:
        ui_src = fh.read()
    i = ui_src.index('class HALCYON_PT_passes')
    panel = ui_src[i:i + 4000]
    check('the Passes panel offers every new toggle',
          all(f"'{k}'" in panel for k in ALL_NEW))
    check('the panel note no longer claims mist / light components are '
          'not produced',
          'mist, vectors, light components -- is not' not in panel
          and 'Vector' in panel and 'Denoising' in panel)

    check("FEATURES has 'render_passes' as BOTH with a measured number",
          CAP.FEATURES.get('render_passes', (None,))[0] == CAP.BOTH
          and 'measured' in CAP.FEATURES['render_passes'][1]
          and any(ch.isdigit() for ch in CAP.FEATURES['render_passes'][1]))
    check("FEATURES has 'light_passes' as NOT_YET naming the MRT twin",
          CAP.FEATURES.get('light_passes', (None,))[0] == CAP.NOT_YET
          and 'MRT' in CAP.FEATURES['light_passes'][1])
    check("'light_passes' is BLOCKING so the device panel says CPU for now",
          'light_passes' in CAP.BLOCKING)
    st = _settings(pass_diffuse=True)
    sc = demo_scene(st)
    used = CAP.scene_features(sc, st)
    check('scene_features names light_passes and render_passes when asked',
          'light_passes' in used and 'render_passes' in used)
    used0 = CAP.scene_features(sc, _settings())
    check('... and neither when no pass is on',
          'light_passes' not in used0 and 'render_passes' not in used0)

    ENG = _engine_module()
    names = [n for n, _c, _i, _k in ENG.PASS_SPEC]
    for want in ('Mist', 'Env', 'Beauty', 'Diffuse', 'Spec', 'Ambient',
                 'Emit', 'Shadow', 'AO', 'Color', 'Light07'):
        check(f'PASS_SPEC declares {want}', want in names)
    spec = {n: (c, i, k) for n, c, i, k in ENG.PASS_SPEC}
    check('Mist is one channel Z VALUE, Beauty four RGBA COLOR',
          spec['Mist'] == (1, 'Z', 'VALUE')
          and spec['Beauty'] == (4, 'RGBA', 'COLOR')
          and spec['Diffuse'] == (3, 'RGB', 'COLOR'))
    st_all = RenderSettings()
    for f in dataclasses.fields(RenderSettings):
        if f.name.startswith('pass_'):
            setattr(st_all, f.name, True)
    check('the declared passes are exactly the ones wanted',
          set(names) == set(R.wanted_passes(st_all)),
          str(set(names) ^ set(R.wanted_passes(st_all))))
    check("Blender's own Passes panel stays unforced",
          'VIEWLAYER_PT_layer_passes' not in ENG.FORCED_PANELS)
    src = inspect.getsource(ENG)
    check('the engine comment names Vector / Denoising as the remaining gap',
          'Vector' in src and 'Denoising Data' in src)

    from .. import properties as PR
    for k in ALL_NEW:
        check(f'{k} carries a label and a real tooltip',
              k in PR.LABELS and len(PR.DESCRIPTIONS.get(k, '')) >= 40)
    check("the light tooltips say 'CPU'",
          all('CPU' in PR.DESCRIPTIONS[k] for k in SPLIT
              + ('pass_shadow', 'pass_ao', 'pass_color', 'pass_lights')))


# ---------------------------------------------------------- the engine end


def test_beauty_pass_is_the_linear_frame():
    """Through the fake engine: Beauty is registered as 4-channel RGBA and
    delivered as the linear frame beside Combined; under the EGA preset
    (16 colours, Stucki) the Combined is palette-quantised while Beauty
    is not; at default post the two agree."""
    from . import fakeblender as FB
    from ..presets.library import PRESETS, apply_preset
    props, engine = FB.install()

    img, extra, cap = FB.run_render(props, engine, pass_beauty=True,
                                    aa_samples=1)
    check('Beauty is registered as a 4-channel pass',
          ('Beauty', 4) in cap['registered'], str(cap['registered']))
    check('Beauty is delivered at the frame size',
          'Beauty' in extra and extra['Beauty'].shape == img.shape,
          str({k: v.shape for k, v in extra.items()}))
    if 'Beauty' in extra:
        d = float(np.abs(extra['Beauty'][..., :3] - img[..., :3]).max())
        check('at default post Beauty agrees with Combined (8-bit depth '
              'quantisation apart)', d < 5e-3, f'max {d:.2e}')
        check('Beauty is finite', bool(np.isfinite(extra['Beauty']).all()))

    ega = dict(PRESETS['EGA']['settings'])
    for k in ('resolution_x', 'resolution_y', 'pixel_aspect_x',
              'pixel_aspect_y', 'aa_mode', 'aa_samples'):
        ega.pop(k, None)
    ega['aa_samples'] = 1
    img2, extra2, _cap2 = FB.run_render(props, engine, pass_beauty=True,
                                        **ega)

    def uniq(a):
        q = np.round(np.asarray(a[..., :3], np.float64) * 1000.0)
        return len(np.unique(q.reshape(-1, 3), axis=0))

    check('under EGA the engine still delivers Beauty',
          'Beauty' in extra2 and extra2['Beauty'].shape == img2.shape)
    if 'Beauty' in extra2:
        d2 = float(np.abs(extra2['Beauty'][..., :3] - img2[..., :3]).max())
        check('under EGA Beauty differs from the quantised Combined',
              d2 > 0.02, f'max {d2:.3f}')

    # the renderer end, on the demo scene: the frame post is handed vs the
    # EGA Combined -- > 64 linear colours against <= 16
    st = RenderSettings()
    apply_preset(st, 'EGA')
    st.resolution_x, st.resolution_y = 96, 72
    st.aa_samples = 1
    st.output_scale = 'NONE'
    # square pixels: the 1.2 aspect resample blends palette entries
    st.pixel_aspect_x = st.pixel_aspect_y = 1.0
    sc = demo_scene(st)
    beauty = R.render(sc, st)
    combined = post.process(beauty, st, target_size=(96, 72))
    check('the linear frame holds > 64 colours while the EGA Combined '
          'holds <= 16', uniq(beauty) > 64 and uniq(combined) <= 16,
          f'{uniq(beauty)} vs {uniq(combined)}')


def test_passes_survive_the_sub_render_roads():
    """The roads that carry last_passes generically: the panorama clears
    them, the stereo parallax road keeps only the wanted ones, motion
    blur / accumulate hand back the centre's."""
    st = _settings(64, 48, pass_diffuse=True, pass_mist=True)
    sc = demo_scene(st)
    sc.camera.type = 'PANO'
    R.render(sc, st)
    check('a panorama clears the new passes like the old ones',
          sc.last_passes is None)

    st = _settings(64, 48, pass_diffuse=True, pass_mist=True,
                   stereo_mode='PARALLAX')
    sc = demo_scene(st)
    img = R.render(sc, st)
    p = sc.last_passes or {}
    check('the parallax stereo road keeps the wanted passes and drops the '
          'forced ones', set(p) == {'Diffuse', 'Mist'}
          and bool(np.isfinite(img).all()), str(sorted(p)))

    st = _settings(64, 48, pass_diffuse=True, pass_environment=True,
                   aa_mode='ACCUMULATE', aa_samples=2)
    sc = demo_scene(st)
    img = R.render(sc, st)
    p = sc.last_passes or {}
    check('the accumulate road hands back the colour passes',
          'Diffuse' in p and 'Env' in p
          and p['Diffuse'].shape[:2] == img.shape[:2])


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
    print('all R253 compositing checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
