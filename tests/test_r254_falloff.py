"""R254: the light falloff family and the Light Limit.

Every check below fails on the tree before this round and passes now (the
module runs against the old tree too: the helpers fall back to the calls
it had), or pins something that must NOT have moved.

    (1) Falloff Start is a Custom Range value: the inverse laws do not
        shift by a leftover Start (it put a pole at d = Start)
    (2) Scene Default = Custom Range: the GPU ramps over End - Start
    (3) Custom Range + Sphere + Start: the Sphere clips at Falloff End on
        the GPU too (hal_lights texel 5.w)
    (4) Blender Internal's 0.001 snap through Scene Default
    (5) the volumetric beam's reach through Scene Default
    (6) the Light Limit: Brightest ranks on what a lamp delivers, Nearest
        measures a distance, one selection (LI.frame_lights) for every
        reader, survivors in scene order (the GPU plan's signature)
    (7) the Light panel draws exactly the dials the law in use reads --
        Scene Default lamps, the Blender Internal Distance, Sphere, no
        falloff rows on a Sun, the Area lamp's Distance note
    (8) the family at random on both roads, within the deferred bar

    python -m halcyon.tests.test_r254_falloff
"""
import ast
import os
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import plane, _mesh_concat, look_at_matrix
from .test_r251_lighting_lamps import _run_glsl
from ..core import lights as LI
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
#: the strip rig: a Lambert floor under an orthographic camera looking
#: straight down, W x H pixels over SPAN world units, the lamp at height
#: HGT off-centre -- every pixel of the middle row is a known distance
W, H = 160, 8
SPAN = 30.0
HGT = 2.0
X0 = -3.0
ALB = 0.6
E0 = 37.0
BAR = 6e-3

LAWS = ('NONE', 'INVERSE', 'INVERSE_SQUARE', 'CUSTOM', 'BI_LINEAR',
        'BI_SQUARE', 'BI_SLIDERS', 'GL_3TERM', 'POV_FADE_LINEAR',
        'POV_FADE_SQUARE', 'GX_GENTLE', 'GX_MEDIUM', 'GX_STEEP',
        'NOMINAL_INVERSE', 'NOMINAL_INV_SQUARE')
DIALS = ('decay_start', 'decay_end', 'decay_ld1', 'decay_ld2', 'gx_ref_brite')


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def note(text):
    print(f'  note {text}')


# ------------------------------------------------------------------ the rig


def _settings(**kw):
    st = base_settings(W, H, shadow_default='PER_LIGHT',
                       global_ambient=(0.0, 0.0, 0.0))
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _lamp(kind='POINT', **kw):
    base = dict(type=kind, name='L', position=(X0, 0.0, HGT),
                direction=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0),
                energy=E0, shadow='NONE')
    if kind == 'SPOT':
        base.update(spot_size=3.05, spot_blend=0.0)
    base.update(kw)
    return Light(**base)


def _scene(lights, st, floor=90.0):
    if isinstance(lights, Light):
        lights = [lights]
    mats = [Material(name='Floor', index=0, model='LAMBERT',
                     diffuse=(ALB,) * 3, specular_level=0.0,
                     ambient_level=0.0)]
    mesh = _mesh_concat([plane(z=0.0, size=floor, mat=0, obj=0)])
    mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
    objs = [ObjectInfo(name='Floor', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=f32))]
    cam = Camera(matrix_world=look_at_matrix((0.0, 0.0, 40.0),
                                             (0.0, 0.0, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    cam.type = 'ORTHO'
    cam.ortho_scale = SPAN * H / W
    sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=list(lights),
               camera=cam, world=World(color=(0.0, 0.0, 0.0),
                                       ambient=(0.0, 0.0, 0.0)),
               settings=st)
    sc.images = {}
    return sc


def _cpu(lights, st=None, sc=None):
    st = st or _settings()
    LI.clear_shadow_cache()
    return np.asarray(R.render(sc or _scene(lights, st), st))[..., :3]


def _gpu(lights, st=None, sc=None):
    """The deferred pass through the GLSL simulator twin (plan_frame +
    simulate on a fresh plan), or None with the refusal printed."""
    st = st or _settings()
    LI.clear_shadow_cache()
    sc = sc or _scene(lights, st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     W, H)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    if p is None:
        note(f'the GPU plan refused: {why}')
        return None
    out, hit = GSH.simulate(job, g)
    if out is None:
        note(f'the GPU simulator refused: {hit}')
        return None
    return np.asarray(out)


def _roads(label, lights, st=None, bar=BAR):
    """(cpu, gpu) with the two-road parity check at the deferred bar."""
    c = _cpu(lights, st)
    g = _gpu(lights, st)
    if g is None:
        check(f'{label}: the deferred pass plans and simulates', False)
        return c, None
    d = float(np.abs(c - g).max())
    check(f'{label}: the GPU twin is the CPU frame within the deferred bar',
          d < bar, f'max |CPU - GPU| {d:.2e} (frame peak {float(c.max()):.3f})')
    return c, g


def _dist(pos=(X0, 0.0, HGT)):
    """Distance from `pos` to the middle row's pixels."""
    x = ((np.arange(W) + 0.5) / W - 0.5) * SPAN
    y = -(((H // 2 + 0.5) / H) - 0.5) * SPAN * H / W
    return np.sqrt((x - pos[0]) ** 2 + (y - pos[1]) ** 2 + pos[2] ** 2)


def _row(img):
    return np.asarray(img)[H // 2, :, :3].mean(axis=1).astype(np.float64)


def _frame_lights(sc, st):
    """LI.frame_lights; on a tree without it, the bare call every reader
    used to make."""
    fn = getattr(LI, 'frame_lights', None)
    return fn(sc, st) if fn else LI.select_lights(sc.lights, st)


def _job_ns(st):
    """What pack_light_texels needs of a job when no lamp is a screen
    spot: its settings."""
    return types.SimpleNamespace(settings=st)


def _old_attenuate_d(dist, start):
    """The distance the inverse laws divided by before this round."""
    return np.maximum(dist - start, LI.EPS) if start > 0 \
        else np.maximum(dist, LI.EPS)


# ------------------------------------------- 0. what must not have moved


def test_a_start_zero_and_under_the_limit_are_the_old_statements():
    """Default neutrality at the function level: with Falloff Start 0 the
    attenuation, its GLSL text and the packed texels are the statements
    the old tree ran, and a scene at or under the Light Limit never
    reaches the new ranking (frames: docs-dev/r254/falloff.md, 360 frames
    bitwise against the fork point on both roads)."""
    dist = np.linspace(0.05, 40.0, 257).astype(f32)
    st = _settings()
    bad = []
    for law in ('INVERSE', 'INVERSE_SQUARE'):
        lt = _lamp(decay=law)
        d = _old_attenuate_d(dist, 0.0)
        want = (1.0 / d if law == 'INVERSE' else 1.0 / (d * d)).astype(f32)
        if not np.array_equal(LI.attenuate(lt, dist, st), want):
            bad.append(law)
    check('Inverse and Inverse Square at Start 0 are bitwise the old '
          'statements 1/max(d, eps)^n', not bad, ', '.join(bad))
    refs = {'D': 'hal_ltex(3).y', 'start': 'hal_ltex(3).x',
            'ld1': 'hal_ltex(3).z', 'ld2': 'hal_ltex(3).w',
            'sphere': 'hal_ltex(5).w'}
    lt = _lamp(decay='INVERSE_SQUARE')
    check('the emitted Inverse Square at Start 0 is the text it was, on the '
          'literal and the texel road',
          GM._attenuation(lt, 'NONE') == GM._attenuation(lt, 'NONE', refs=refs)
          == '(1.0 / (max(dist, 1e-06) * max(dist, 1e-06)))',
          GM._attenuation(lt, 'NONE'))
    sph = _lamp(decay='BI_LINEAR', bi_sphere=True)
    check('a Sphere outside Custom Range still divides by texel 3.y (its '
          'GLSL is the text it was)',
          GM._attenuation(sph, 'NONE', refs=refs)
          == '((max(dist, 0.0) < hal_ltex(3).y) ? ((hal_ltex(3).y / '
             '(hal_ltex(3).y + max(dist, 0.0))) * ((hal_ltex(3).y - '
             'max(dist, 0.0)) / hal_ltex(3).y)) : 0.0)')
    # the packer at Start 0: resolving Scene Default moves no byte
    lamps = [_lamp(decay='DEFAULT', decay_end=6.5),
             _lamp('SPOT', decay='DEFAULT', decay_end=6.5),
             _lamp(decay='CUSTOM', decay_end=6.5),
             _lamp(decay='BI_SQUARE', decay_end=6.5, bi_sphere=True),
             Light(type='SUN', direction=(0.2, 0.1, -1.0), energy=2.0)]
    t_plain = GM.pack_light_texels(lamps)
    same = all(np.array_equal(GM.pack_light_texels(lamps, _job_ns(_settings(
        light_falloff_default=law))), t_plain)
        for law in ('CUSTOM', 'NONE', 'INVERSE', 'BI_LINEAR'))
    check('at Start 0 the packed texels do not depend on the render\'s '
          'default (End - 0 is End: resolving Scene Default moves no byte), '
          'and texel 5.w is 0 on every lamp', same
          and float(np.abs(t_plain[0, 5::GM.LIGHT_TEXEL_STRIDE, 3]).max())
          == 0.0)
    # at or under the limit: the very list, and no bounds are measured
    calls = []
    orig = LI.scene_bounds
    LI.scene_bounds = lambda v: (calls.append(1), orig(v))[1]
    try:
        pts = [_lamp(name=f'P{i}', position=(i * 2.0 - 7.0, 0.0, 2.0))
               for i in range(7)]
        sun = Light(type='SUN', name='Sun', direction=(0.2, 0.1, -1.0),
                    energy=3.0, shadow='NONE')
        ok = True
        for mode in ('BRIGHTEST', 'NEAREST', 'FIRST'):
            st8 = _settings(light_limit_mode=mode)
            sc = _scene(pts + [sun], st8)
            got = _frame_lights(sc, st8)
            ok = ok and len(got) == 8 and all(
                a is b for a, b in zip(got, sc.lights))
            st0 = _settings(light_limit_mode=mode, max_lights=0)
            sc0 = _scene(pts + [sun, _lamp(name='P7'), _lamp(name='P8')], st0)
            got0 = _frame_lights(sc0, st0)
            ok = ok and len(got0) == 10 and all(
                a is b for a, b in zip(got0, sc0.lights))
        stf = _settings(light_limit_mode='FIRST', max_lights=2)
        scf = _scene(pts + [sun], stf)
        ok_first = [l.name for l in _frame_lights(scf, stf)] == ['P0', 'P1']
    finally:
        LI.scene_bounds = orig
    check('eight lamps under the factory limit of 8, and any number under '
          'limit 0, come back as the scene\'s own list in its order',
          ok)
    check('First keeps the first lamps in scene order, as it did', ok_first)
    check('...and none of those selections measured the scene (the old '
          'path, untouched)', not calls, f'{len(calls)} bounds calls')
    over = pts + [sun, _lamp(name='P7', energy=5.0), _lamp(name='P8',
                                                           energy=90.0)]
    st8 = _settings()
    key = [float(l.energy) * float(np.mean(l.color)) for l in over]
    want = [over[i].name for i in np.argsort(key)[::-1][:8]]
    check('a bare lamp list (no scene to measure against) keeps the raw '
          'ranking select_lights always had',
          [l.name for l in LI.select_lights(over, st8)] == want)


# ----------------------------------- 1. Falloff Start is Custom Range only


def test_b_falloff_start_is_a_custom_range_value():
    """decay-start-hidden-pole: the panel draws Falloff Start for Custom
    Range only, but Inverse / Inverse Square / Scene Default read a
    leftover value as 1/(d - Start)^n: 1/EPS inside Start."""
    dist = np.asarray([1.0, 3.0, 3.5, 5.0, 12.0], f32)
    st = _settings()
    for law, sd in (('INVERSE', 'NONE'), ('INVERSE_SQUARE', 'NONE'),
                    ('DEFAULT', 'INVERSE_SQUARE'), ('DEFAULT', 'INVERSE')):
        st_l = _settings(light_falloff_default=sd)
        a0 = LI.attenuate(_lamp(decay=law), dist, st_l)
        a3 = LI.attenuate(_lamp(decay=law, decay_start=3.0), dist, st_l)
        check(f'{law}{" under " + sd if law == "DEFAULT" else ""}: a '
              'leftover Falloff Start 3 changes nothing (bitwise Start 0)',
              np.array_equal(a0, a3),
              f'at d = 1, 3, 3.5, 5: {[float(v) for v in a3[:4]]}')
    reads = []
    for law in LAWS:
        kw = dict(decay_end=6.5, decay_ld1=0.35, decay_ld2=0.8)
        a0 = LI.attenuate(_lamp(decay=law, **kw), dist, st)
        a3 = LI.attenuate(_lamp(decay=law, decay_start=2.0, **kw), dist, st)
        if not np.array_equal(a0, a3):
            reads.append(law)
    check('of the fifteen laws only Custom Range reads Falloff Start',
          reads == ['CUSTOM'], ', '.join(reads))
    c = LI.attenuate(_lamp(decay='CUSTOM', decay_start=2.0, decay_end=6.0),
                     dist, st)
    want = np.clip(1.0 - (dist - 2.0) / 4.0, 0.0, 1.0).astype(f32)
    check('Custom Range still ramps from Start to End',
          float(np.abs(c - want).max()) < 1e-6)
    # the emitted text, both value roads
    refs = {'D': 'hal_ltex(3).y', 'start': 'hal_ltex(3).x',
            'ld1': 'hal_ltex(3).z', 'ld2': 'hal_ltex(3).w',
            'sphere': 'hal_ltex(5).w'}
    for law in ('INVERSE', 'INVERSE_SQUARE'):
        l0 = _lamp(decay=law)
        l3 = _lamp(decay=law, decay_start=3.0)
        check(f'the {law} GLSL does not mention Start, literal or texel',
              GM._attenuation(l3, 'NONE') == GM._attenuation(l0, 'NONE')
              and GM._attenuation(l3, 'NONE', refs=refs)
              == GM._attenuation(l0, 'NONE', refs=refs),
              GM._attenuation(l3, 'NONE', refs=refs))
    # the frame: a POINT and a SPOT, both roads
    for kind in ('POINT', 'SPOT'):
        for law, sd in (('INVERSE_SQUARE', 'NONE'),
                        ('DEFAULT', 'INVERSE_SQUARE')):
            st_l = _settings(light_falloff_default=sd)
            c0, g0 = _roads(f'{kind} {law}, Start 0', _lamp(kind, decay=law),
                            st_l)
            c3, g3 = _roads(f'{kind} {law}, leftover Start 3',
                            _lamp(kind, decay=law, decay_start=3.0), st_l)
            inside = _dist() < 3.0
            hot = int((_row(c3)[inside] >= 63.9).sum())
            check(f'{kind} {law}: the frame with a leftover Start 3 is '
                  'bitwise the Start 0 frame on the CPU road (no pixel at '
                  'the 64.0 ceiling inside Start)',
                  np.array_equal(c0, c3) and hot == 0,
                  f'max |d| {float(np.abs(c0 - c3).max()):.3g}, '
                  f'{hot} of {int(inside.sum())} px at the ceiling')
            if g0 is not None and g3 is not None:
                check(f'{kind} {law}: ...and on the GPU twin',
                      np.array_equal(g0, g3),
                      f'max |d| {float(np.abs(g0 - g3).max()):.3g}')


# --------------------------- 2 + 3. Scene Default = Custom Range, the Sphere


def test_c_scene_default_custom_range_on_the_gpu():
    """scene-default-custom-gpu-ramp: the packer chose texel 3.y from the
    lamp's own unresolved 'DEFAULT', so a Scene Default lamp under a
    Custom Range render ramped over End on the GPU and End - Start on
    the CPU."""
    st = _settings(light_falloff_default='CUSTOM')
    lt_d = _lamp(decay='DEFAULT', decay_start=2.5, decay_end=6.5,
                 energy=400.0)
    lt_c = _lamp(decay='CUSTOM', decay_start=2.5, decay_end=6.5,
                 energy=400.0)
    t_d = GM.pack_light_texels([lt_d], _job_ns(st))
    t_c = GM.pack_light_texels([lt_c], _job_ns(_settings()))
    check('the packer hands a Scene Default lamp under Custom Range the '
          'ramp width End - Start (4.0), as for the law chosen on the lamp',
          float(t_d[0, 3, 1]) == 4.0 and np.array_equal(t_d, t_c),
          f'texel 3.y {float(t_d[0, 3, 1])}')
    c, g = _roads('Scene Default = Custom Range 2.5..6.5', lt_d, st)
    d = _dist()
    cos = HGT / d
    law = np.clip(1.0 - (d - 2.5) / 4.0, 0.0, 1.0)
    want = ALB / np.pi * 400.0 / (4.0 * np.pi) * law * cos
    check('the CPU frame is the label\'s law clip(1 - (d - Start) / (End - '
          'Start))', float(np.abs(_row(c) - want).max()) < 2e-5,
          f'{float(np.abs(_row(c) - want).max()):.2e}')
    if g is not None:
        g_c = _gpu(lt_c, _settings())
        check('the GPU frame through Scene Default is bitwise the GPU frame '
              'of Custom Range chosen on the lamp',
              g_c is not None and np.array_equal(g, g_c))
        check('...and it reaches zero at Falloff End (no light past 6.5)',
              float(_row(g)[d > 6.55].max()) == 0.0,
              f'{float(_row(g)[d > 6.55].max()):.4f} past End')
    for kind in ('POINT', 'SPOT'):
        for law_d in ('GX_MEDIUM', 'POV_FADE_SQUARE', 'BI_SLIDERS',
                      'NOMINAL_INVERSE'):
            _roads(f'{kind} on Scene Default = {law_d}',
                   _lamp(kind, decay='DEFAULT', decay_start=1.5,
                         decay_end=4.0, decay_ld1=0.3, decay_ld2=0.2),
                   _settings(light_falloff_default=law_d))


def test_d_custom_range_sphere_start_on_the_gpu():
    """custom-sphere-start-gpu: under Custom Range texel 3.y is End -
    Start, and the Sphere wrap divided by it: the GPU clipped at End -
    Start, the CPU at Falloff End."""
    d = _dist()
    for route in ('lamp', 'default'):
        for kind in ('POINT', 'SPOT'):
            for s0, end in ((0.0, 6.5), (2.5, 6.5), (4.0, 2.6)):
                lt = _lamp(kind, decay=('CUSTOM' if route == 'lamp'
                                        else 'DEFAULT'),
                           decay_start=s0, decay_end=end, bi_sphere=True,
                           energy=400.0)
                st = _settings(light_falloff_default=(
                    'CUSTOM' if route == 'default' else 'INVERSE_SQUARE'))
                tag = (f'Custom Range {s0}..{end} + Sphere, {kind}, '
                       + ('on the lamp' if route == 'lamp'
                          else 'via Scene Default'))
                c, g = _roads(tag, lt, st)
                if g is None:
                    continue
                rc, rg = _row(c), _row(g)
                lc = float(d[rc > 0].max()) if (rc > 0).any() else 0.0
                lg = float(d[rg > 0].max()) if (rg > 0).any() else 0.0
                check(f'{tag}: both roads light out to the same distance',
                      abs(lc - lg) < 1e-9, f'CPU {lc:.2f} GPU {lg:.2f}')
    # the value road: the float32 attenuate divides by, for this pairing only
    st_c = _settings(light_falloff_default='CUSTOM')
    lamps = [_lamp(decay='CUSTOM', decay_start=2.5, decay_end=6.5,
                   bi_sphere=True),
             _lamp('SPOT', decay='DEFAULT', decay_start=2.5, decay_end=6.5,
                   bi_sphere=True),
             _lamp(decay='CUSTOM', decay_start=2.5, decay_end=6.5),
             _lamp(decay='BI_LINEAR', decay_end=6.5, bi_sphere=True),
             _lamp(decay='CUSTOM', decay_start=0.0, decay_end=6.5,
                   bi_sphere=True)]
    tex = GM.pack_light_texels(lamps, _job_ns(st_c))
    w5 = [float(tex[0, i * GM.LIGHT_TEXEL_STRIDE + 5, 3]) for i in range(5)]
    y3 = [float(tex[0, i * GM.LIGHT_TEXEL_STRIDE + 3, 1]) for i in range(5)]
    check('texel 5.w carries Falloff End for a Custom Range + Sphere lamp '
          '(chosen or through Scene Default) and stays 0 for every other',
          w5 == [6.5, 6.5, 0.0, 0.0, 6.5], str(w5))
    check('at Start 0 texel 5.w is the float32 texel 3.y was (the Sphere '
          'divides by the number it always did)', w5[4] == y3[4] == 6.5)
    # statement level: the emitted expression on the packer's texels
    dist = np.linspace(0.05, 9.0, 181).astype(f32)
    worst = 0.0
    for lt in lamps[:2] + lamps[4:]:
        t = GM.pack_light_texels([lt], _job_ns(st_c))
        expr = GM._attenuation(lt, 'CUSTOM', refs={
            'D': 'u_d', 'start': 'u_s', 'ld1': 'u_a', 'ld2': 'u_b',
            'sphere': 'u_p'})
        src = ('uniform float dist;\nuniform float u_d;\nuniform float u_s;'
               '\nuniform float u_a;\nuniform float u_b;\nuniform float u_p;'
               '\nout vec4 Color;\nvoid main()\n{\n    float att = '
               f'{expr};\n    Color = vec4(att, 0.0, 0.0, 1.0);\n}}\n')
        got, err = _run_glsl(src, dist.size, dist=dist, u_s=t[0, 3, 0],
                             u_d=t[0, 3, 1], u_a=t[0, 3, 2], u_b=t[0, 3, 3],
                             u_p=t[0, 5, 3])
        if got is None:
            check('the Custom Range + Sphere expression compiles standalone',
                  False, str(err))
            continue
        worst = max(worst, float(np.abs(
            got[:, 0] - LI.attenuate(lt, dist, st_c)).max()))
    check('the emitted Custom Range + Sphere expression on the packed '
          'texels equals LI.attenuate (2e-6)', worst < 2e-6, f'{worst:.2e}')


# ----------------------- 4 + 5. the BI snap and the beam via Scene Default


def test_e_bi_snap_through_scene_default():
    """bi-snap-skips-scene-default: lamp_get_visibility's 0.001 cut-off
    was applied on the lamp's own unresolved law name, so a Scene Default
    lamp under a Blender Internal render default kept its tail."""
    d = _dist()
    for kind in ('POINT', 'SPOT'):
        for law, end, cut in (('BI_SQUARE', 0.02, 4.5), ('BI_LINEAR', 0.004,
                                                         4.2)):
            kw = dict(decay_end=end, energy=5.0e4)
            st_d = _settings(light_falloff_default=law)
            c_l, g_l = _roads(f'{kind} {law} on the lamp',
                              _lamp(kind, decay=law, **kw), _settings())
            c_d, g_d = _roads(f'{kind} {law} via Scene Default',
                              _lamp(kind, decay='DEFAULT', **kw), st_d)
            lit = int((_row(c_d)[d > cut] > 0).sum())
            check(f'{kind} {law}: through Scene Default the CPU frame is '
                  'bitwise the lamp-chosen frame (nothing lit past the '
                  'cut-off)', np.array_equal(c_l, c_d) and lit == 0,
                  f'{lit} px lit past d = {cut}')
            if g_l is not None and g_d is not None:
                check(f'{kind} {law}: ...and the GPU twin',
                      np.array_equal(g_l, g_d)
                      and int((_row(g_d)[d > cut] > 0).sum()) == 0)
    consts = {'light_texels': True, 'falloff_default': 'BI_SQUARE',
              'cookies': {}, 'light_links': {}, 'resolution': (W, H),
              'specular_in_gamma': True, 'light_clamp': 0.0,
              'shadow_default': 'PER_LIGHT'}
    snap = 'att = (att <= 0.001) ? 0.0 : att;'
    blk = '\n'.join(GM._one_light_source(0, _lamp(decay='DEFAULT'), consts))
    blk_sq = '\n'.join(GM._one_light_source(
        0, _lamp(decay='DEFAULT'), dict(consts,
                                        falloff_default='INVERSE_SQUARE')))
    check('the emitted block snaps a Scene Default lamp under a Blender '
          'Internal default, and only then', snap in blk
          and snap not in blk_sq)
    P = np.asarray([[X0 + 9.0, 0.0, 0.0]], f32)
    r_l = LI.sample(_lamp(decay='BI_SQUARE', decay_end=0.02), P,
                    _settings())[1]
    r_d = LI.sample(_lamp(decay='DEFAULT', decay_end=0.02), P,
                    _settings(light_falloff_default='BI_SQUARE'))[1]
    check('LI.sample past the cut-off: zero both ways',
          float(np.abs(r_l).max()) == 0.0 and float(np.abs(r_d).max()) == 0.0,
          f'on the lamp {float(r_l.max()):.3g}, via Scene Default '
          f'{float(r_d.max()):.3g}')


def test_f_beam_reach_through_scene_default():
    """beam-reach-scene-default: a volumetric lamp's beam stops at Falloff
    End under Custom Range; through Scene Default it ran the full reach
    (the unresolved law name again, core/render._light_volumes)."""
    bw, bh = 96, 72

    def beam(decay, sd, end):
        lt = Light(type='SPOT', name='B', position=(-6.0, 0.0, 8.0),
                   direction=(1.0, 0.0, -0.25), color=(1, 1, 1),
                   energy=400.0, shadow='NONE', spot_size=0.5,
                   spot_blend=0.1, decay=decay, decay_start=0.0,
                   decay_end=end, volumetric=1.0)
        st = _settings(light_falloff_default=sd, spot_cones=True)
        st.resolution_x, st.resolution_y = bw, bh
        sc = _scene(lt, st, floor=60.0)
        sc.camera = Camera(matrix_world=look_at_matrix((0.0, -14.0, 4.0),
                                                       (0.0, 0.0, 4.0)),
                           lens=28.0, sensor=36.0, clip_start=0.1,
                           clip_end=200.0)
        LI.clear_shadow_cache()
        return np.asarray(R.render(sc, st))[..., :3]

    a = beam('CUSTOM', 'INVERSE_SQUARE', 3.0)
    b = beam('DEFAULT', 'CUSTOM', 3.0)
    far = beam('CUSTOM', 'INVERSE_SQUARE', 60.0)
    check('the beam of a Scene Default lamp under Custom Range stops at '
          'Falloff End: the frame is the lamp-chosen frame',
          np.array_equal(a, b),
          f'frame sums {float(a.sum()):.1f} / {float(b.sum()):.1f}')
    check('...and Falloff End really is the beam\'s reach (End 60 is a '
          'longer beam)', float(far.sum()) > 4.0 * float(a.sum()),
          f'{float(far.sum()):.1f} vs {float(a.sum()):.1f}')


# ------------------------------------------------------ 6. the Light Limit


def _nine_and_a_sun():
    pts = [Light(type='POINT', name=f'P{i}', position=(-13.0 + 3.2 * i, 0.0,
                                                       2.5),
                 color=(1, 1, 1), energy=10.0, shadow='NONE')
           for i in range(9)]
    sun = Light(type='SUN', name='Sun', direction=(0.2, 0.1, -1.0),
                color=(1, 1, 1), energy=3.0, shadow='NONE')
    return pts, sun


def test_g_light_limit_ranks_on_what_a_lamp_delivers():
    """light-limit-raw-energy / limit-brightest-01: Brightest compared a
    Sun's strength (3) with a point lamp's watts (10) -- at the factory
    limit of 8 a Sun plus nine 10 W lamps rendered without the Sun."""
    pts, sun = _nine_and_a_sun()
    full = _cpu(pts + [sun], _settings(max_lights=0))
    for order, ls in (('the Sun listed last', pts + [sun]),
                      ('the Sun listed first', [sun] + pts)):
        st = _settings()
        sc = _scene(ls, st)
        kept = [l.name for l in _frame_lights(sc, st)]
        check(f'factory Light Limit 8, Brightest, {order}: the Sun is lit',
              'Sun' in kept, str(kept))
        check(f'...the survivors are in scene order ({order})',
              kept == [l.name for l in ls if l.name in kept], str(kept))
        c, g = _roads(f'a Sun and nine 10 W lamps, {order}', ls, st)
        check(f'...and the frame keeps its light ({order}; the two weakest '
              'point lamps go, not the Sun)',
              float(c.mean()) > 0.95 * float(full.mean()),
              f'frame mean {float(c.mean()):.4f} of {float(full.mean()):.4f}'
              ' unlimited')
    if not hasattr(LI, 'limit_strength'):
        check('LI.limit_strength exists (the number Brightest ranks on)',
              False)
        return
    # the number is the radiance the light loop delivers at the scene
    st = _settings()
    sc = _scene(pts + [sun], st)
    centre, radius = LI.limit_bounds(sc)
    far = Light(type='POINT', name='far', position=(40.0, 0.0, 30.0),
                color=(0.5, 1.0, 0.25), energy=900.0, shadow='NONE')
    got = LI.limit_strength(far, centre, radius, st)
    rad = float(np.mean(LI.sample(far, np.asarray([centre], f32), st)[1]))
    check('a point lamp outside the scene is ranked on exactly the '
          'radiance LI.sample delivers at the scene\'s centre',
          abs(got - rad) < 1e-6 * max(rad, 1e-12), f'{got:.6g} vs {rad:.6g}')
    check('a Sun is ranked on its strength times its mean colour',
          LI.limit_strength(sun, centre, radius, st) == 3.0)
    at_c = Light(type='POINT', name='c', position=tuple(centre),
                 energy=10.0, shadow='NONE')
    half = Light(type='POINT', name='h', energy=10.0, shadow='NONE',
                 position=(float(centre[0]) + 0.5 * radius, float(centre[1]),
                           float(centre[2])))
    s_c = LI.limit_strength(at_c, centre, radius, st)
    s_h = LI.limit_strength(half, centre, radius, st)
    check('a lamp ON the centre is measured at half the scene\'s radius, '
          'not at the pole of its own curve',
          np.isfinite(s_c) and abs(s_c - s_h) < 1e-5 * s_h
          and abs(s_c - 10.0 / (4.0 * np.pi * (0.5 * radius) ** 2))
          < 1e-5 * s_c, f'{s_c:.6g} vs {s_h:.6g}')
    neg = Light(type='POINT', name='n', position=(40.0, 0.0, 30.0),
                color=(0.5, 1.0, 0.25), energy=-900.0, shadow='NONE')
    check('a negative Power ranks by its magnitude',
          LI.limit_strength(neg, centre, radius, st) == got)
    none = Light(type='POINT', name='k', position=(40.0, 0.0, 30.0),
                 energy=10.0, decay='NONE', shadow='NONE')
    check('the lamp\'s own falloff law is what is ranked (Falloff None: '
          'Power / 4 pi at any distance)',
          abs(LI.limit_strength(none, centre, radius, st)
              - 10.0 / (4.0 * np.pi)) < 1e-6)
    area = Light(type='AREA', name='a', position=(40.0, 0.0, 30.0),
                 direction=(-0.8, 0.0, -0.6), energy=10.0, decay_end=25.0,
                 area_size=(0.3, 0.3), area_x=(0.6, 0.0, -0.8),
                 area_y=(0.0, 1.0, 0.0), shadow='NONE')
    check('an Area lamp is ranked on its form factor\'s far field, 2 '
          '(Distance / d)^2 (its Distance is a gain)',
          abs(LI.limit_strength(area, centre, radius, st)
              - 10.0 * 2.0 * (25.0 / 50.0) ** 2 / (4.0 * np.pi)) < 1e-5)
    # the style gain is not in the ranking: a strobe keeps the selection
    strobe = [Light(type='POINT', name=f'S{i}', position=(40.0, i, 30.0),
                    energy=900.0 + i, shadow='NONE') for i in range(3)]
    st2 = _settings(max_lights=2)
    sc2 = _scene(strobe, st2)
    k0 = [l.name for l in _frame_lights(sc2, st2)]
    strobe[2].style_gain = 0.0
    k1 = [l.name for l in _frame_lights(sc2, st2)]
    check('a lamp whose light style is on a dark step keeps its slot (the '
          'ranking reads the raw Power)', k0 == k1 == ['S1', 'S2'], str(k1))
    # nothing parked: the answer is a function of (scene, settings) alone
    st3 = _settings()
    sc3 = _scene(pts + [sun], st3)
    before = (set(vars(st3)), set(vars(sc3)))
    _frame_lights(sc3, st3)
    check('frame_lights parks nothing on the settings or the scene',
          before == (set(vars(st3)), set(vars(sc3))))
    line = LI.limit_report(sc3, st3)
    check('the lamps left unlit are named (the limit used to drop them in '
          'silence)', bool(line) and 'P7' in line and 'P8' in line
          and 'Sun' not in line.split('(', 2)[-1] and '2 of 10' in line,
          str(line))
    check('...and nothing is reported while every lamp is lit',
          LI.limit_report(_scene(pts[:7] + [sun], st3), st3) is None)


def test_h_light_limit_nearest_measures_a_distance():
    """limit-nearest-01: no caller ever handed select_lights the point to
    measure from, so Nearest ran the Brightest ranking."""
    near = [Light(type='POINT', name='far-stronger', position=(-70.0, 0.0,
                                                               40.0),
                  color=(1, 1, 1), energy=950.0, shadow='NONE'),
            Light(type='POINT', name='near-weak', position=(0.0, 0.0, 1.0),
                  color=(1, 1, 1), energy=5.0, shadow='NONE'),
            Light(type='POINT', name='far-strong', position=(60.0, 0.0, 40.0),
                  color=(1, 1, 1), energy=900.0, shadow='NONE'),
            Light(type='POINT', name='mid-weak', position=(4.0, 0.0, 1.0),
                  color=(1, 1, 1), energy=6.0, shadow='NONE')]
    st_n = _settings(light_limit_mode='NEAREST', max_lights=2)
    st_b = _settings(light_limit_mode='BRIGHTEST', max_lights=2)
    k_n = [l.name for l in _frame_lights(_scene(near, st_n), st_n)]
    k_b = [l.name for l in _frame_lights(_scene(near, st_b), st_b)]
    check('Nearest keeps the two lamps nearest the middle of the scene',
          k_n == ['near-weak', 'mid-weak'], str(k_n))
    check('...which is not what Brightest keeps on this rig',
          k_b == ['far-stronger', 'far-strong'], str(k_b))
    c, g = _roads('Nearest, Light Limit 2', near, st_n)
    alone = _cpu([near[1], near[3]], _settings(max_lights=0))
    check('the Nearest frame is the frame of the two near lamps alone',
          np.array_equal(c, alone),
          f'max |d| {float(np.abs(c - alone).max()):.3g}')
    sun = Light(type='SUN', name='Sun', direction=(0.2, 0.1, -1.0),
                energy=0.01, shadow='NONE')
    st1 = _settings(light_limit_mode='NEAREST', max_lights=1)
    check('a Sun has no position and reaches everything: Nearest always '
          'counts it',
          [l.name for l in _frame_lights(_scene(near + [sun], st1), st1)]
          == ['Sun'])
    calls = []
    orig = LI.select_lights

    def spy(lights, settings, centre=None, *a, **k):
        calls.append(centre is not None)
        return orig(lights, settings, centre, *a, **k)

    LI.select_lights = spy
    try:
        _cpu(near, st_n)
        _gpu(near, st_n)
    finally:
        LI.select_lights = orig
    check('every selection of an over-limit frame, CPU and GPU, is handed '
          'the scene\'s centre', bool(calls) and all(calls),
          f'{sum(calls)} of {len(calls)} calls')


def test_i_one_selection_for_both_roads():
    """The frame's lamps come from LI.frame_lights everywhere, and the
    survivors' order is the scene's -- the order the GPU plan's selection
    signature fingerprints."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bare = []
    for sub in ('', 'core', 'gpu', 'nodes'):
        folder = os.path.join(root, sub)
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith('.py'):
                continue
            path = os.path.join(folder, fn)
            with open(path, encoding='utf-8') as fh:
                tree = ast.parse(fh.read())
            owner = {}
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for ch in ast.walk(node):
                        owner.setdefault(id(ch), node.name)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else \
                    getattr(f, 'id', '')
                if name != 'select_lights':
                    continue
                if sub == 'core' and fn == 'lights.py' and \
                        owner.get(id(node)) == 'frame_lights':
                    continue
                bare.append(f'{os.path.join(sub, fn)}:{node.lineno}')
    check('no reader selects lamps from a bare list: every call of '
          'select_lights is frame_lights\' own (a caller that needs the '
          'frame\'s lamps calls LI.frame_lights(scene, settings))',
          not bare, ', '.join(bare))
    # the order flip: a Power edit swaps the RANK of the two kept lamps
    st = _settings(max_lights=2)
    a = Light(type='POINT', name='A', position=(-6.0, 0.0, 2.0),
              color=(1, 1, 1), energy=100.0, shadow='NONE')
    b = Light(type='SPOT', name='B', position=(5.0, 0.0, 2.0),
              direction=(0.0, 0.0, -1.0), spot_size=1.2, spot_blend=0.0,
              color=(1, 1, 1), energy=90.0, shadow='NONE')
    c = Light(type='POINT', name='C', position=(0.0, 0.0, 2.0),
              color=(1, 1, 1), energy=1.0, shadow='NONE')
    sc = _scene([a, b, c], st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    tex = R.prepare_textures(sc, st)
    GSH._PLAN_CACHE.clear()

    def frame():
        LI.clear_shadow_cache()
        job = R.ShadeJob(sc, st, tex, None, view, eye, W, H)
        p, _why, atl = GSH.plan_frame(job, g)
        out, _hit = GSH.simulate(job, g, p, atl)
        cpu = np.asarray(R.render(sc, st))[..., :3]
        return p, [l.name for l in job.lights], \
            float(np.abs(np.asarray(out) - cpu).max())

    p1, k1, d1 = frame()
    b.energy = 110.0
    p2, k2, d2 = frame()
    check('the kept lamps come back in scene order before and after the '
          'Power edit that swaps their rank', k1 == k2 == ['A', 'B'],
          f'{k1} then {k2}')
    check('...so the plan-cache HIT that follows repacks each lamp into '
          'its own block: the GPU twin is still the CPU frame',
          p2 is p1 and max(d1, d2) < BAR,
          f'max |CPU - GPU| {d1:.2e} then {d2:.2e}')
    eq = [Light(type='POINT', name=f'E{i}', position=(i - 4.0, 0.0, 2.0),
                energy=25.0, shadow='NONE') for i in range(9)]
    st8 = _settings()
    check('equal lamps keep their scene order (the first eight; ties used '
          'to fall to the sort\'s whim)',
          [l.name for l in _frame_lights(_scene(eq, st8), st8)]
          == [f'E{i}' for i in range(8)])


# ------------------------------------------------------------ 7. the panel


class _Layout:
    """A recording layout: (property, label) rows and the note texts."""

    def __init__(self, rec):
        self.rec = rec
        self.active = True
        self.use_property_split = False

    def prop(self, data, name, **kw):
        self.rec.append((name, kw.get('text')))

    def column(self, **kw):
        return _Layout(self.rec)

    row = column
    box = column

    def separator(self, **kw):
        pass

    def label(self, **kw):
        self.rec.append(('label', str(kw.get('text', ''))))

    def template_ID(self, data, name, **kw):
        self.rec.append((name, None))

    def operator(self, idn, **kw):
        self.rec.append(('op:' + idn, None))
        return types.SimpleNamespace(scope=None)


def _draw(UI, defaults, kind, scene_default=None, **fields):
    hs = types.SimpleNamespace(**defaults)
    hs.cookie = None
    for k, v in fields.items():
        setattr(hs, k, v)
    light = types.SimpleNamespace(type=kind, color=(1, 1, 1), energy=10.0,
                                  halcyon=hs)
    scene = None if scene_default is None else types.SimpleNamespace(
        halcyon=types.SimpleNamespace(light_falloff_default=scene_default))
    rec = []
    panel = UI.HALCYON_PT_light()
    panel.layout = _Layout(rec)
    UI.HALCYON_PT_light.draw(panel, types.SimpleNamespace(
        engine=UI.ENGINE, light=light, scene=scene, object=None))
    return rec


def _reads(law, route):
    """The dials LI.attenuate reads under `law` (chosen on the lamp, or
    through Scene Default): each one moved alone."""
    dist = np.linspace(0.3, 30.0, 97).astype(f32)
    base = dict(decay_start=1.5, decay_end=6.0, decay_ld1=0.3, decay_ld2=0.2,
                gx_ref_brite=0.4)
    st = _settings(light_falloff_default=(law if route == 'default'
                                          else 'NONE'))
    dec = law if route == 'lamp' else 'DEFAULT'
    a0 = LI.attenuate(_lamp(decay=dec, **base), dist, st)
    out = set()
    for dial, v in (('decay_start', 0.5), ('decay_end', 9.0),
                    ('decay_ld1', 0.7), ('decay_ld2', 0.6),
                    ('gx_ref_brite', 0.8)):
        if not np.array_equal(a0, LI.attenuate(
                _lamp(decay=dec, **dict(base, **{dial: v})), dist, st)):
            out.add(dial)
    return out


def test_j_panel_draws_what_the_law_reads():
    """scene-default-hides-sliders, bi-distance-not-drawn,
    bi-sphere-not-drawn, sun-falloff-menu-dead and the wording of
    area-distance-gain: the Light panel's falloff rows are exactly the
    dials the law in use reads."""
    from . import fakebpy
    fakebpy.install()
    from .. import properties as PR
    from .. import ui as UI
    ann = PR.HalcyonLightSettings.__annotations__
    defaults = {k: (tuple(v.kw.get('default')) if isinstance(
        v.kw.get('default'), (tuple, list)) else v.kw.get('default'))
        for k, v in ann.items()}
    wrong = []
    for kind in ('POINT', 'SPOT'):
        for law in LAWS:
            for route in ('lamp', 'default'):
                rec = _draw(UI, defaults, kind,
                            scene_default=(law if route == 'default'
                                           else 'INVERSE_SQUARE'),
                            decay=(law if route == 'lamp' else 'DEFAULT'))
                drawn = {n for n, _t in rec if n in DIALS}
                read = _reads(law, route)
                if drawn != read:
                    wrong.append(f'{kind} {law} ({route}): drawn '
                                 f'{sorted(drawn)} read {sorted(read)}')
    check('for every law, chosen on the lamp or arriving through Scene '
          'Default, the panel draws exactly the dials LI.attenuate reads '
          '(Sphere off)', not wrong,
          f'{len(wrong)} of 60 rows differ; ' + '; '.join(wrong[:4]))
    rec = _draw(UI, defaults, 'POINT', scene_default='CUSTOM')
    check('a lamp on Scene Default under a Custom Range render shows its '
          'own Falloff Start and Falloff End, with a note saying why',
          ('decay_start', None) in rec and ('decay_end', None) in rec
          and any(n == 'label' and 'render' in t for n, t in rec), str(rec[5:11]))
    for law in ('BI_LINEAR', 'BI_SQUARE', 'BI_SLIDERS'):
        rec = _draw(UI, defaults, 'POINT', decay=law)
        check(f'{law} draws its Distance (Falloff End, the half-strength '
              'distance of the law)', ('decay_end', 'Distance') in rec)
    no_sphere, no_dist = [], []
    for kind in ('POINT', 'SPOT'):
        for law in LAWS:
            rec = _draw(UI, defaults, kind, decay=law, bi_sphere=True)
            if not any(n == 'bi_sphere' for n, _t in rec):
                no_sphere.append(f'{kind} {law}')
            if not any(n == 'decay_end' for n, _t in rec):
                no_dist.append(f'{kind} {law}')
    check('Sphere is drawn for a Point and a Spot lamp under every law',
          not no_sphere, ', '.join(no_sphere[:5]))
    check('...and while it is on, the distance it clips at is drawn under '
          'every law', not no_dist, ', '.join(no_dist[:5]))
    rec = _draw(UI, defaults, 'POINT', decay='INVERSE_SQUARE')
    check('with Sphere off, Inverse Square draws no distance at all',
          not any(n in DIALS for n, _t in rec))
    family = set(DIALS) | {'decay', 'bi_sphere', 'decay_near_start',
                           'decay_near_end'}
    for hemi in (False, True):
        seen = set()
        for law in ('DEFAULT',) + LAWS:
            rec = _draw(UI, defaults, 'SUN', decay=law, hemi=hemi,
                        bi_sphere=True)
            seen |= {n for n, _t in rec if n in family}
        check(f'a Sun (hemi {hemi}) draws no falloff row under any law -- a '
              'directional lamp never attenuates',
              not seen, ', '.join(sorted(seen)))
        rec = _draw(UI, defaults, 'SUN', hemi=hemi)
        check(f'...and keeps the rest of its panel (hemi {hemi}: the '
              'Hemisphere flag and its ground colour)',
              any(n == 'hemi' for n, _t in rec)
              and any(n == 'hemi_ground' for n, _t in rec))
    rec = _draw(UI, defaults, 'AREA')
    names = [n for n, _t in rec]
    i_d = rec.index(('decay_end', 'Distance')) \
        if ('decay_end', 'Distance') in rec else -1
    check('an Area lamp draws Distance, a note that brightness scales with '
          'its square, then Gamma -- and no falloff menu',
          i_d >= 0 and rec[i_d + 1][0] == 'label'
          and 'Distance squared' in rec[i_d + 1][1]
          and rec[i_d + 2][0] == 'area_gamma' and 'decay' not in names
          and 'bi_sphere' not in names, str(rec[i_d:i_d + 3]))
    try:
        _draw(UI, defaults, 'POINT', scene_default=None, decay='DEFAULT')
        ok = True
    except Exception:                                           # noqa: BLE001
        ok = False
    check('the panel draws without a scene (Scene Default reads as Inverse '
          'Square, the render setting\'s own default)', ok)
    # the wording
    tip = {n: str(ann[n].kw.get('description') or '') for n in ann}
    check('Falloff End\'s tooltip names every role of the value, the Area '
          'lamp\'s gain included',
          all(w in tip['decay_end'] for w in (
              'Custom Range', 'Blender Internal', 'POV', 'GX', 'Nominal',
              'Sphere', 'Area', 'squared')), tip['decay_end'][:60])
    check('Falloff Start\'s tooltip says it is a Custom Range value',
          'Custom Range only' in tip['decay_start'])
    check('the Lin / Quad sliders\' tooltips name OpenGL 1.1 as well as '
          'Blender Internal, and an OpenGL coefficient above 1 can be typed '
          '(hard maximum 100, the slider still 0..1)',
          all('OpenGL' in tip[n] and 'Blender Internal' in tip[n]
              and float(ann[n].kw.get('max')) >= 100.0
              and float(ann[n].kw.get('soft_max')) == 1.0
              for n in ('decay_ld1', 'decay_ld2')))
    thin = [n for n in ('decay', 'decay_start', 'decay_end', 'decay_ld1',
                        'decay_ld2', 'bi_sphere', 'area_gamma')
            if len(tip[n]) < 40]
    thin += [it[0] for it in PR.LIGHT_LIMIT if len(it[2]) < 40]
    thin += [n for n in ('light_limit_mode', 'max_lights',
                         'light_falloff_default')
             if len(str(PR.HalcyonSettings.__annotations__[n].kw.get(
                 'description') or '')) < 40]
    check('every tooltip this round touched is 40+ characters', not thin,
          ', '.join(thin))
    check('the Light Limit items say what they measure (delivered light; '
          'the middle of the scene)',
          'deliver' in PR.LIGHT_LIMIT[0][2] and 'middle' in PR.LIGHT_LIMIT[1][2])
    check('the enum numbering is untouched (saved files hold the numbers)',
          [it[0] for it in PR.FALLOFF] == list(LAWS)
          and [it[0] for it in PR.LIGHT_LIMIT] == ['BRIGHTEST', 'NEAREST',
                                                   'FIRST'])
    d = Light()
    check('no default moved: the panel\'s and the dataclass\'s falloff '
          'defaults agree',
          all(defaults[n] == getattr(d, n) for n in DIALS + ('bi_sphere',))
          and defaults['decay'] == d.decay == 'DEFAULT')


# ----------------------------------------- 8. the family on both roads


def test_k_family_on_both_roads():
    """The verifier's sweep as a fixed grid: law x route x lamp kind x
    Start x Sphere x Near pair, CPU road against the GPU twin (the two
    GPU-only defects of this round sat in the Start > 0 cells)."""
    bad = []
    worst = 0.0
    n = 0
    rng = np.random.default_rng(254)
    for law in ('CUSTOM', 'INVERSE_SQUARE', 'INVERSE', 'BI_SQUARE',
                'GX_MEDIUM', 'NOMINAL_INVERSE', 'NONE', 'GL_3TERM'):
        for route in ('lamp', 'default'):
            for start in (0.0, 1.5):
                for sphere in (False, True):
                    kind = 'POINT' if rng.random() < 0.5 else 'SPOT'
                    near = (0.0, 0.0) if rng.random() < 0.5 else (2.0, 6.0)
                    lt = _lamp(kind, decay=(law if route == 'lamp'
                                            else 'DEFAULT'),
                               decay_start=start,
                               decay_end=float(rng.choice([0.5, 4.0, 7.0])),
                               decay_ld1=0.3, decay_ld2=0.2,
                               gx_ref_brite=0.4, bi_sphere=sphere,
                               decay_near_start=near[0],
                               decay_near_end=near[1],
                               energy=float(rng.choice([37.0, -37.0, 4000.0])))
                    st = _settings(light_falloff_default=(
                        law if route == 'default' else 'INVERSE_SQUARE'))
                    c = _cpu(lt, st)
                    g = _gpu(lt, st)
                    n += 1
                    if g is None or not (np.isfinite(c).all()
                                         and np.isfinite(g).all()):
                        bad.append(f'{kind} {law} {route} (refused / '
                                   'non-finite)')
                        continue
                    d = float(np.abs(c - g).max())
                    rel = d / max(float(np.abs(c).max()), 1e-12)
                    if rel > 1e-3 or d >= BAR:
                        bad.append(f'{kind} {law} {route} start {start} '
                                   f'sphere {sphere}: {d:.2e}')
                    else:
                        worst = max(worst, rel)
    check(f'{n} lamps of the falloff family: the GPU twin agrees with the '
          'CPU within 1e-3 of the frame peak and the deferred bar',
          not bad, f'{len(bad)} diverge ({"; ".join(bad[:3])}); worst of the '
          f'rest {worst:.1e}')


def test_l_strict_glsl_of_the_falloff_blocks():
    """The changed emitter shapes through the strict front-end, when
    HALCYON_GLSLANG points at glslang (tests/glsl_strict.py is the gate;
    its scenes carry no falloff variants)."""
    from . import glsl_strict as GS
    if GS._glslang() is None:
        note('HALCYON_GLSLANG is not set: the strict front-end did not run '
             'on the falloff blocks')
        return
    cases = [('Inverse Square, leftover Start', dict(decay='INVERSE_SQUARE',
                                                     decay_start=3.0), 'NONE'),
             ('Custom Range + Sphere + Start', dict(
                 decay='CUSTOM', decay_start=2.5, decay_end=6.5,
                 bi_sphere=True), 'NONE'),
             ('Scene Default = Custom Range + Sphere', dict(
                 decay='DEFAULT', decay_start=2.5, decay_end=6.5,
                 bi_sphere=True), 'CUSTOM'),
             ('Scene Default = BI Square (the snap)', dict(
                 decay='DEFAULT', decay_end=0.02), 'BI_SQUARE'),
             ('Scene Default = GX Medium + Sphere', dict(
                 decay='DEFAULT', decay_end=4.0, bi_sphere=True), 'GX_MEDIUM')]
    bad = []
    n = 0
    for label, kw, sd in cases:
        for kind in ('POINT', 'SPOT'):
            st = _settings(light_falloff_default=sd)
            sc = _scene(_lamp(kind, **kw), st)
            view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
            g = CRg.GBuffer(W, H)
            CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                          depth_bits=st.depth_precision)
            job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                             eye, W, H)
            GSH._PLAN_CACHE.clear()
            passes, why, _atl = GSH.plan_frame(job, g)
            if passes is None:
                bad.append(f'{label} ({kind}): refused, {why}')
                continue
            for ps in passes:
                src = next((x for x in ps if isinstance(x, str)
                            and 'void main' in x), None)
                if src is None:
                    continue
                n += 1
                ok, msg = GS.compile_source(src)
                if not ok:
                    bad.append(f'{label} ({kind}): {msg[:200]}')
    check(f'{n} planned passes of the changed falloff blocks compile under '
          'strict GLSL 4.30', not bad and n > 0, '; '.join(bad[:2]))


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
    print('all R254 falloff and Light Limit checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
