"""R254: where the light packages meet.

Each package of the light round has its own module; these checks hold the
seams between them, which no single package could test:

    (a) the visible beam's unit changed (spot-beam: core/cones.beam_gain)
        and the load-time sort of older files (native: light_units.py)
        keeps an older Halcyon file's beams at the brightness they had:
        Beam Density x 64 pi^2 for the scenes that draw beams, and an
        Area lamp's Volumetric over 2 D^2 (its beam reads Distance now)
    (b) an Area lamp's object scale is folded into its Size (ray-road)
        under both power units (native): in watts the power is kept, on
        Blender Internal units the gain of scale_x * scale_y is kept --
        and either way the shadow softens with the scale
    (c) the hal_lights row carries TWO tails past the per-lamp texels:
        the ambient pool (flags) and the lamps' shadow values (map-gpu).
        Each package laid its own at n * STRIDE; together the ambient
        tail sits first and the shadow blocks follow it, on the plan
        build and on every repack, and the frame is the CPU's

    python -m halcyon.tests.test_r254_integration
"""
import math
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_r254_native import _Group, _bpy
from ..core import cones as CONES
from ..core.scene import Light

FAILS = []


def check(name, ok, detail=''):
    print(('  ok   ' if ok else '  FAIL ') + name
          + (f'  {detail}' if detail else ''))
    if not ok:
        FAILS.append(name)


_UID = [9000]


def _lamp(name, kind, **stored):
    _UID[0] += 1
    return types.SimpleNamespace(
        name=name, type=kind, library=None, session_uid=_UID[0],
        halcyon=_Group(('area_power', 'decay_end', 'volumetric', 'shadow'),
                       **stored))


def _bscene(name, engine='HALCYON_RENDER', **stored):
    return types.SimpleNamespace(
        name=name, library=None,
        render=types.SimpleNamespace(engine=engine),
        halcyon=_Group(('light_units_version', 'spot_cones',
                        'spot_cone_density', 'max_lights'), **stored))


def _data(scenes, lights):
    return types.SimpleNamespace(scenes=list(scenes), lights=list(lights),
                                 filepath='')


def _old_beam(kind, energy, density, volumetric):
    """The beam's radiance per unit of marched scatter BEFORE the round:
    density x volumetric x energy / pi, whatever the lamp kind."""
    return density * volumetric * energy / math.pi


def _new_beam(kind, energy, density, volumetric, distance=25.0):
    lt = Light(type=kind, energy=energy)
    lt.decay_end = distance
    return density * volumetric * CONES.beam_gain(lt)


def test_a_old_file_beams_keep_their_brightness():
    """An older Halcyon file that draws beams opens with the beams it
    had: the unit moved by 64 pi^2, and the sort moves the dials back."""
    _bpy()
    from .. import light_units as LU
    check('the gain is the old unit over the new one for a spot or point '
          'lamp (energy / pi over energy / (64 pi^3))',
          abs(LU.BEAM_UNIT_GAIN - 64.0 * math.pi ** 2) < 1e-9
          and abs(_old_beam('SPOT', 1000.0, 1.0, 1.0)
                  / _new_beam('SPOT', 1000.0, 1.0, 1.0)
                  / LU.BEAM_UNIT_GAIN - 1.0) < 1e-6)

    spot = _lamp('Spot', 'SPOT', volumetric=0.03)
    point = _lamp('Bulb', 'POINT', volumetric=0.01)
    area = _lamp('Panel', 'AREA', volumetric=0.05, decay_end=25.0)
    dark = _lamp('NoBeam', 'SPOT')
    sc = _bscene('Stage', spot_cones=True, spot_cone_density=0.5)
    off = _bscene('NoCones', spot_cone_density=2.0)
    data = _data([sc, off], [spot, point, area, dark])
    LU.sort_file(data)
    d_new = float(sc.halcyon.spot_cone_density)
    check('a scene that draws beams has its Beam Density multiplied by '
          '64 pi^2', abs(d_new / (0.5 * LU.BEAM_UNIT_GAIN) - 1.0) < 1e-9,
          f'{d_new:.3f}')
    check('...a scene whose Volumetric Lights switch is off keeps its '
          'Density (it drew no beam to keep)',
          float(off.halcyon.spot_cone_density) == 2.0)
    for la, kind in ((spot, 'SPOT'), (point, 'POINT')):
        v0 = {'Spot': 0.03, 'Bulb': 0.01}[la.name]
        was = _old_beam(kind, 800.0, 0.5, v0)
        now = _new_beam(kind, 800.0, d_new, float(la.halcyon.volumetric))
        check(f'the {kind.title()} beam is as bright as it was, and its '
              'Volumetric was not touched',
              abs(now / was - 1.0) < 1e-6
              and float(la.halcyon.volumetric) == v0,
              f'x{now / was:.6f}')
    was = _old_beam('AREA', 2.0, 0.5, 0.05)
    now = _new_beam('AREA', 2.0, d_new, float(area.halcyon.volumetric), 25.0)
    check("the Area lamp's beam is as bright as it was: its Volumetric "
          'was divided by 2 Distance^2 (the beam reads Distance now)',
          abs(now / was - 1.0) < 1e-6
          and abs(float(area.halcyon.volumetric) / (0.05 / 1250.0) - 1.0)
          < 1e-9, f'x{now / was:.6f}')
    check('...and the lamp is on Blender Internal units, where Distance '
          'is the dial the panel shows',
          area.halcyon.area_power == 'BI')
    check('the file is stamped: a second sort changes nothing',
          sc.halcyon.light_units_version == LU.UNITS_VERSION
          and LU.sort_file(data) == []
          and float(sc.halcyon.spot_cone_density) == d_new
          and abs(float(area.halcyon.volumetric) / (0.05 / 1250.0) - 1.0)
          < 1e-9)

    # nothing to keep: no lamp with a Volumetric
    sc2 = _bscene('Plain', spot_cones=True, spot_cone_density=1.0)
    LU.sort_file(_data([sc2], [_lamp('S', 'SPOT')]))
    check('an older file with the switch on but no lamp scattering keeps '
          'its Density: its first beam uses the new unit, where the '
          "slider's 0..4 is the working range",
          float(sc2.halcyon.spot_cone_density) == 1.0)
    # a file saved since: untouched
    sc3 = _bscene('Saved', spot_cones=True, spot_cone_density=3.0,
                  light_units_version=LU.UNITS_VERSION)
    LU.sort_file(_data([sc3], [_lamp('S', 'SPOT', volumetric=1.0)]))
    check('a file saved since the change is never rescaled',
          float(sc3.halcyon.spot_cone_density) == 3.0)
    # a file with no sign of Halcyon: untouched
    sc4 = _bscene('Eevee', engine='BLENDER_EEVEE')
    LU.sort_file(_data([sc4], [_lamp('S', 'SPOT')]))
    check('a file with no sign of Halcyon is not rescaled',
          not sc4.halcyon.is_property_set('spot_cone_density'))
    # the top of the range
    sc5 = _bscene('Thick', spot_cones=True, spot_cone_density=1000.0)
    LU.sort_file(_data([sc5], [_lamp('S', 'SPOT', volumetric=0.5)]))
    check("a Density past the dial's range stops at its top",
          float(sc5.halcyon.spot_cone_density) == LU.BEAM_DENSITY_MAX)
    from .. import properties as PR
    check('...which is the range the render setting declares',
          tuple(PR.RANGES['spot_cone_density'])[1] == LU.BEAM_DENSITY_MAX)


def test_b_scaled_area_lamp_under_both_units():
    """The fold of the object scale into the Size meets the power units:
    watts keep the power, Blender Internal keeps its scale gain."""
    _bpy()
    from .. import export as EX

    def exported(units, scale, size=0.5, gamma=1.0):
        mw = np.eye(4, dtype=np.float32)
        mw[:3, 0] *= scale[0]
        mw[:3, 1] *= scale[1]
        mw[:3, 3] = (0.0, 0.0, 4.0)
        hs = _Group(('area_power', 'area_gamma', 'decay', 'decay_start',
                     'decay_end', 'shadow', 'shadow_map_size', 'shadow_bias',
                     'shadow_softness', 'shadow_samples', 'shadow_density',
                     'negative', 'diffuse_only', 'specular_only',
                     'ambient_only', 'hotspot', 'volumetric', 'flare',
                     'caustics'),
                    area_power=units, area_gamma=gamma, decay='DEFAULT',
                    decay_end=25.0, shadow='MAP', shadow_softness=1.0,
                    shadow_density=1.0, shadow_color=(0.0, 0.0, 0.0),
                    negative=False, diffuse_only=False, specular_only=False,
                    ambient_only=False)
        data = types.SimpleNamespace(
            type='AREA', color=(1.0, 1.0, 1.0), energy=10.0, halcyon=hs,
            shape='RECTANGLE', size=size, size_y=size, shadow_soft_size=0.0,
            use_shadow=True)
        return EX.export_light(types.SimpleNamespace(name='L', data=data),
                               mw)
    for units in ('WATTS', 'BI'):
        try:
            one = exported(units, (1.0, 1.0))
            big = exported(units, (2.0, 3.0))
        except Exception as exc:                                # noqa: BLE001
            check(f'{units}: a scaled Area lamp exports', False, repr(exc))
            continue
        check(f'{units}: scale (2, 3) is folded into the Size on unit axes '
              '(the face the shadow softens by)',
              np.allclose(big.area_size, (1.0, 1.5), atol=1e-6)
              and abs(float(np.linalg.norm(big.area_x)) - 1.0) < 1e-6
              and abs(float(np.linalg.norm(big.area_y)) - 1.0) < 1e-6
              and tuple(one.area_size) == (0.5, 0.5))
        k = float(big.energy) / float(one.energy)
        if units == 'WATTS':
            check("in watts the lamp's power is kept (Blender's rule: the "
                  'face grows, its radiance drops)', abs(k - 1.0) < 1e-6,
                  f'x{k:.6f}')
        else:
            check('on Blender Internal units the gain of scale_x * scale_y '
                  'is kept (an older file\'s scaled lamp is as bright as '
                  'it was)', abs(k - 6.0) < 1e-5, f'x{k:.6f}')
    try:
        g = exported('BI', (2.0, 3.0), gamma=0.5)
        o = exported('BI', (1.0, 1.0), gamma=0.5)
        k = float(g.energy) / float(o.energy)
        check('...shaped by Gamma as the form factor is (6 ^ 0.5)',
              abs(k - 6.0 ** 0.5) < 1e-5, f'x{k:.6f}')
    except Exception as exc:                                    # noqa: BLE001
        check('a scaled Area lamp with a Gamma exports', False, repr(exc))


def test_c_the_two_tails_of_the_light_row():
    """An ambient lamp, a mapped sized lamp and an Area lamp in one
    frame: the ambient tail and the shadow tail share the hal_lights
    row without overlapping, and the GPU twin draws the CPU's frame --
    before and after an edit that only repacks the row."""
    import copy
    from . import test_r254_map_soft as MS
    from . import test_r254_map_gpu as MG
    from ..core import lights as LI
    from ..core import render as R
    from ..gpu import material as GM
    from ..gpu import shade as GSH

    def build(amb_energy=0.2, radius=0.3):
        st = MS._settings(shadow_samples=4)
        key = MS._lamp('POINT', radius, 'MAP')
        sc = MS._wall(key, st)
        area = MS._lamp('AREA', 1.0, 'MAP', pos=(-3.0, -4.0, 5.0))
        area.energy = 2.0
        amb = Light(type='POINT', name='Amb', position=(0.0, 0.0, 3.0),
                    color=(0.4, 0.6, 1.0), energy=amb_energy, shadow='NONE')
        amb.ambient_only = True
        sc.lights = [key, area, amb]
        return sc, st

    sc, st = build()
    cpu, img, cov, passes, atlases, why = MG._sim(sc, st)
    check('a frame with an ambient lamp, a mapped sized lamp and an Area '
          'lamp plans', img is not None, str(why))
    if img is None:
        return
    d = float(np.abs(img[cov][:, :3] - cpu[cov][:, :3]).max())
    check('...and the GPU twin draws the CPU frame (the 6e-3 bar)',
          d < 6e-3, f'max {d:.2e}')
    lights = LI.frame_lights(sc, st)
    n = len(lights)
    job, _g = MG._job(sc, st)
    metas = [GSH._shadow_meta(l, st, None, want_atlas=False)[0]
             for l in lights]
    tail = GM.light_tail_layout(lights, metas, ambient=True)
    base = GM.ambient_tail_base(n)
    first = min(list(tail['area'].values()) + list(tail['map'].values()))
    check('the ambient tail sits at n * STRIDE and every shadow block '
          'starts past its three texels',
          base == n * GM.LIGHT_TEXEL_STRIDE
          and first >= base + GM.AMBIENT_TAIL_TEXELS
          and len(tail['map']) >= 1 and len(tail['area']) == 1,
          f'ambient at {base}, first block at {first}')
    row = GM.pack_light_texels(lights, job, shadows=metas)
    pool = LI.ambient_light(sc, st)
    check('the packed row is as wide as the layout says and holds the '
          'ambient pool in its tail',
          row.shape[1] == tail['width']
          and np.array_equal(row[0, base, :3], pool)
          and float(np.abs(pool).max()) > 0.0)
    bare = GM.light_tail_layout(lights, metas)
    check('without an ambient lamp the shadow blocks start at n * STRIDE '
          '(the layout map-gpu ships alone)',
          min(list(bare['area'].values()) + list(bare['map'].values()))
          == n * GM.LIGHT_TEXEL_STRIDE)
    # an edit of the ambient lamp AND of the mapped lamp's size: a
    # plan-cache hit that repacks the row, and still the CPU's frame
    job, g = MG._job(sc, st)
    GSH._PLAN_CACHE.clear()
    p0 = GSH.plan_frame(job, g)
    sc.lights[2].energy = 0.6
    sc.lights[0].radius = 0.6
    stats = getattr(GSH, 'PLAN_STATS', None)
    before = dict(stats) if isinstance(stats, dict) else None
    job2, g2 = MG._job(sc, st)
    passes2, why2, atl2 = GSH.plan_frame(job2, g2)
    img2, _hit = GSH.simulate(job2, g2, passes2, atl2)
    cpu2 = MG._cpu_frame(sc, st)
    cov2 = g2.tri >= 0
    d2 = float(np.abs(img2[cov2][:, :3] - cpu2[cov2][:, :3]).max())
    check('after an ambient Power edit and a size edit the repacked row '
          'still draws the CPU frame', d2 < 6e-3, f'max {d2:.2e}')
    check('...and the two frames differ (the edit was drawn)',
          float(np.abs(img2 - img).max()) > 1e-3)
    if before is not None:
        check('...on the same plan: the edit was a cache hit',
              int(stats.get('miss', 0)) == int(before.get('miss', 0)),
              f"misses {before.get('miss')} -> {stats.get('miss')}")
    del p0, copy


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
    print('all R254 integration checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
