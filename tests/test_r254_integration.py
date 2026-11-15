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
