"""R254: Blender's OWN light options under Halcyon.

Every check below fails on the tree before this round and passes now, or
pins something that must NOT have moved.

    (a) a default lamp of each type exports the fields it always did
    (b) Exposure, Temperature (the Colour as its Tint) and Normalize are
        honoured at export with Blender's own meaning, on both roads
    (c) an Area lamp's Power is Blender's watts for a new lamp; a lamp on
        'Blender Internal (Distance)' is bitwise the lamp it was
    (d) the light-units sort: which lamps of which files keep the old law
        (File > Open, Append, Link, the add-on enabled late, undo, save)
    (e) a Spot's object scale shapes its cone: the round case folded into
        the exported angles, the elliptical case on both roads
    (f) scene time honours fps_base and the subframe
    (g) Blender's hidden switches that have an exact equal here: the
        Square flag, a lobe factor of 0, the lamp object's ray visibility
    (h) the Light panel: one Halcyon-owned panel with the rows the engine
        reads, Cast Shadow among them; the stock panels are not forced
    (i) an Area lamp mirrored across one face axis keeps shining

    python -m halcyon.tests.test_r254_native
"""
import gzip
import math
import os
import sys
import tempfile
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings
from .scenebuild import plane, cube, _mesh_concat, look_at_matrix
from .test_r251_lighting_lamps import _run_glsl
from ..core import cones as CONES
from ..core import lights as LI
from ..core import raster as CRg
from ..core import render as R
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
W, H = 120, 90
BAR = 6e-3

#: Blender 5.2's own RNA defaults for a lamp (every type's fields on one
#: fake datablock; the Radius is the factory-startup lamp's 0.1)
NATIVE = dict(
    color=(1.0, 1.0, 1.0), energy=1000.0, use_temperature=False,
    temperature=6500.0, temperature_color=(1.0, 1.0, 1.0), exposure=0.0,
    normalize=True, shadow_soft_size=0.1, use_soft_falloff=True,
    angle=math.radians(0.526), spot_size=math.radians(45.0),
    spot_blend=0.15, show_cone=False, use_square=False, shape='SQUARE',
    size=1.0, size_y=1.0, use_shadow=True, diffuse_factor=1.0,
    specular_factor=1.0)
#: bpy's own light.temperature_color at 2000 K (Blender 5.2.2)
TEMP_2000 = (2.5201, 0.6469, 0.0204)
POS = (2.5, -1.5, 5.0)
AIM = (0.0, 0.0, 0.8)

#: export_light of the default fake lamp on the tree BEFORE this round
#: (0d638aa), field by field as the repr of plain Python values: only the
#: fields that differ from the Light() dataclass default are listed,
#: every other old field must still equal that default
_OLD_POSE = {
    'name': "'L'",
    'position': '(2.5, -1.5, 5.0)',
    'direction':
        '(-0.4889756441116333, 0.29338538646698, -0.8214790225028992)',
    'frame_x': '(0.5144957304000854, 0.8574929237365723, 0.0)',
    'frame_y':
        '(-0.7044124603271484, 0.4226474463939667, 0.5702387094497681)',
}
OLD_EXPORT = {
    'POINT': dict(_OLD_POSE, radius='0.1'),
    'SUN': dict(_OLD_POSE, type="'SUN'", radius='0.009180431865490174'),
    'SPOT': dict(_OLD_POSE, type="'SPOT'", radius='0.1',
                 spot_size='0.7853981633974483'),
    'AREA': dict(_OLD_POSE, type="'AREA'", radius='0.1',
                 area_x=_OLD_POSE['frame_x'], area_y=_OLD_POSE['frame_y']),
}


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


def _bpy():
    from . import fakebpy
    return fakebpy.install()


def _hs(**kw):
    """A Halcyon light-settings group at the PropertyGroup defaults, an
    Area lamp on the OLD power units unless the test says otherwise."""
    _bpy()
    from .. import properties as PR
    ann = PR.HalcyonLightSettings.__annotations__
    d = {k: (tuple(v.kw.get('default')) if isinstance(
        v.kw.get('default'), (tuple, list)) else v.kw.get('default'))
        for k, v in ann.items()}
    d['cookie'] = None
    d['exclude_collection'] = None
    d['area_power'] = 'BI'
    d.update(kw)
    return types.SimpleNamespace(**d)


def _matrix(scale=(1.0, 1.0, 1.0), pos=POS, aim=AIM):
    m = np.array(look_at_matrix(pos, aim), np.float64)
    m[:3, :3] = m[:3, :3] @ np.diag(scale)
    return m.astype(f32)


def _down(pos=(0.0, 0.0, 4.0), scale=(1.0, 1.0, 1.0)):
    """A lamp matrix pointing straight down (-Z), its X / Y the world's."""
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = np.diag(scale)
    m[:3, 3] = pos
    return m.astype(f32)


def _export(kind, scale=(1.0, 1.0, 1.0), hs=None, ob=None, matrix=None,
            **over):
    """export_light of a fake Blender lamp carrying every native default."""
    _bpy()
    from .. import export as EX
    props = dict(NATIVE)
    props.update(over)
    data = types.SimpleNamespace(type=kind, halcyon=hs or _hs(), **props)
    o = types.SimpleNamespace(name='L', data=data, **(ob or {}))
    return EX.export_light(o, _matrix(scale) if matrix is None else matrix)


def _settings(**kw):
    st = base_settings(W, H, shadow_default='PER_LIGHT',
                       global_ambient=(0.0, 0.0, 0.0))
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _scene(lights, st, box=True, spec=0.0):
    if isinstance(lights, Light):
        lights = [lights]
    mats = [Material(name='Floor', index=0, model='LAMBERT',
                     diffuse=(0.8, 0.8, 0.8), specular_level=spec,
                     ambient_level=0.0),
            Material(name='Box', index=1, model='LAMBERT',
                     diffuse=(0.8, 0.3, 0.2), specular_level=spec,
                     ambient_level=0.0)]
    parts = [plane(z=0.0, size=14.0, mat=0, obj=0)]
    objs = [ObjectInfo(name='Floor', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=f32))]
    if box:
        parts.append(cube(centre=(0.0, 0.0, 1.6), size=1.2, mat=1, obj=1))
        objs.append(ObjectInfo(name='Box', index=1, location=(0, 0, 1.6),
                               matrix_world=np.eye(4, dtype=f32)))
    mesh = _mesh_concat(parts)
    mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
    cam = Camera(matrix_world=look_at_matrix((0.0, -7.5, 6.5),
                                             (0.0, 0.6, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=list(lights),
               camera=cam, world=World(color=(0.0, 0.0, 0.0),
                                       ambient=(0.0, 0.0, 0.0)),
               settings=st)
    sc.images = {}
    return sc


def _cpu(lights, st=None, **kw):
    st = st or _settings()
    LI.clear_shadow_cache()
    R._GBUF_CACHE.clear()
    return np.asarray(R.render(_scene(lights, st, **kw), st))[..., :3] \
        .astype(np.float64)


def _gpu(lights, st=None, **kw):
    """The deferred pass through the GLSL simulator twin (plan_frame +
    simulate on a fresh plan), or None with the refusal printed."""
    st = st or _settings()
    LI.clear_shadow_cache()
    sc = _scene(lights, st, **kw)
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    # the frame's lamps, their maps and style gains, as R.render readies
    # them before either road shades
    LI.apply_light_styles(sc)
    LI.build_shadow_maps(sc, st)
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
    return np.asarray(out)[..., :3].astype(np.float64)


def _roads(label, lights, st=None, bar=BAR, **kw):
    """(cpu, gpu) with the two-road parity check at the deferred bar."""
    c = _cpu(lights, st, **kw)
    g = _gpu(lights, st, **kw)
    if g is None:
        check(f'{label}: the deferred pass plans and simulates', False)
        return c, None
    d = float(np.abs(c - g).max())
    check(f'{label}: the GPU twin is the CPU frame within the deferred bar',
          d < bar, f'max |CPU - GPU| {d:.2e} (frame peak {float(c.max()):.3f})')
    return c, g


def _lit_ratio(img, ref, floor=0.01):
    """Per-channel median of img / ref over the pixels ref lights."""
    m = ref.min(axis=2) > floor
    if not m.any():
        return np.zeros(3)
    return np.median(img[m] / ref[m], axis=0)


def _canon(v):
    """A field value as plain Python (a float32 widened exactly), so its
    repr does not depend on the NumPy version."""
    if isinstance(v, (tuple, list)):
        return tuple(_canon(x) for x in v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _fields(lt):
    import dataclasses
    return {f.name: repr(_canon(getattr(lt, f.name)))
            for f in dataclasses.fields(lt)}


# ---------------------------------------------- (a) the default lamp's export


def test_a_default_lamp_exports_as_before():
    """A lamp that moves none of the options this round reads exports
    field for field what it exported before (repr-exact against the
    frozen export of the tree this round started from)."""
    import dataclasses
    d0 = Light()
    for kind in ('POINT', 'SUN', 'SPOT', 'AREA'):
        got = _fields(_export(kind))
        want = OLD_EXPORT[kind]
        bad = []
        for f in dataclasses.fields(d0):
            if f.name == 'spot_scale':
                continue                      # the one field this round adds
            exp = want.get(f.name, repr(_canon(getattr(d0, f.name))))
            if got[f.name] != exp:
                bad.append(f'{f.name}: {got[f.name]} != {exp}')
        check(f'a default {kind} lamp exports every old field as before '
              '(repr-exact)', not bad, '; '.join(bad[:4]))
        check(f'...and the new field is at its inert default ({kind})',
              got['spot_scale'] == '(1.0, 1.0)')
    # uniform object scale moves no cone, no energy and no colour
    for kind in ('POINT', 'SUN', 'SPOT'):
        a = _fields(_export(kind))
        b = _fields(_export(kind, scale=(3.0, 3.0, 3.0)))
        moved = [n for n in ('energy', 'color', 'radius', 'spot_size',
                             'spot_blend', 'spot_scale', 'hotspot',
                             'decay', 'decay_end', 'affect_diffuse',
                             'affect_specular', 'spot_square')
                 if a[n] != b[n]]
        check(f'a uniformly scaled {kind} lamp (x3) exports the same cone, '
              'energy and colour', not moved, ', '.join(moved))
    # the frame: both roads render the default lamps as one picture
    for kind, e in (('POINT', 600.0), ('SUN', 2.0), ('SPOT', 2500.0),
                    ('AREA', 0.05)):
        _roads(f'a default {kind} lamp',
               [_export(kind, energy=e, hs=_hs(shadow='MAP'))])


# ------------------------------- (b) Exposure, Temperature, Normalize


def test_b_exposure_temperature_normalize():
    """native-exposure-dead, native-temperature-dead, native-normalize-dead:
    the three rows drawn under Power changed nothing on any road."""
    _bpy()
    from .. import export as EX
    for kind in ('POINT', 'SUN', 'SPOT', 'AREA'):
        base = _export(kind)
        up = _export(kind, exposure=1.0)
        dn = _export(kind, exposure=-2.0)
        check(f'{kind}: Exposure +1 doubles the exported energy, -2 '
              'quarters it (BKE_light_power = energy * 2^exposure)',
              up.energy == base.energy * 2.0 and dn.energy == base.energy * 0.25,
              f'{base.energy} -> {up.energy}, {dn.energy}')
        warm = _export(kind, use_temperature=True, temperature=2000.0,
                       temperature_color=TEMP_2000, color=(1.0, 0.5, 0.25))
        want = tuple(c * t for c, t in zip((1.0, 0.5, 0.25), TEMP_2000))
        check(f'{kind}: Temperature multiplies the Colour (the Tint) by '
              "Blender's own blackbody colour",
              np.allclose(warm.color, want, rtol=0, atol=1e-12),
              str(warm.color))
        off = _export(kind, use_temperature=False, temperature=2000.0,
                      temperature_color=TEMP_2000, color=(1.0, 0.5, 0.25))
        check(f'{kind}: with the Temperature box off the Kelvin value is '
              'not read', off.color == (1.0, 0.5, 0.25))
    # Normalize off: Blender's BKE_light_area, type by type
    r = 0.1
    cases = [
        ('POINT', {}, 4.0 * math.pi * r * r),
        ('POINT', {'shadow_soft_size': 0.0}, 4.0),
        ('SPOT', {}, 4.0 * math.pi * r * r),
        ('SUN', {}, math.pi * math.sin(math.radians(0.526) / 2.0) ** 2),
        ('SUN', {'angle': 0.0}, 1.0),
        ('AREA', {'size': 0.25}, 0.0625),
        ('AREA', {'shape': 'RECTANGLE', 'size': 2.0, 'size_y': 0.5}, 1.0),
        ('AREA', {'shape': 'DISK', 'size': 2.0}, math.pi),
        ('AREA', {'shape': 'ELLIPSE', 'size': 2.0, 'size_y': 0.5},
         math.pi / 4.0),
    ]
    for kind, over, area in cases:
        a = _export(kind, **over)
        b = _export(kind, normalize=False, **over)
        check(f'{kind} {over or "(defaults)"}: Normalize off multiplies the '
              f'energy by BKE_light_area = {area:.6g}',
              abs(b.energy - a.energy * area) <= 1e-9 * abs(a.energy * area),
              f'{b.energy / a.energy:.6g}')
    a = _export('AREA', size=0.5, scale=(2.0, 3.0, 1.0), normalize=False,
                hs=_hs(area_power='WATTS'))
    b = _export('AREA', size=0.5, scale=(2.0, 3.0, 1.0),
                hs=_hs(area_power='WATTS'))
    check("an Area lamp's object scale counts in the face Normalize off "
          'multiplies by (Size x scale, both axes)',
          abs(a.energy / b.energy - 0.5 * 2.0 * 0.5 * 3.0) < 1e-6,
          f'{a.energy / b.energy:.6g}')
    check('bke_light_area is exported for the panel and the tests',
          callable(getattr(EX, 'bke_light_area', None)))
    # the frames, both roads
    for kind, e in (('POINT', 600.0), ('SUN', 2.0), ('SPOT', 2500.0),
                    ('AREA', 0.05)):
        ref = _roads(f'{kind} base', [_export(kind, energy=e)])
        got = _roads(f'{kind} Exposure +1', [_export(kind, energy=e,
                                                     exposure=1.0)])
        for road, a, b in (('CPU', got[0], ref[0]), ('GPU twin', got[1],
                                                     ref[1])):
            if a is None or b is None:
                continue
            k = _lit_ratio(a, b)
            check(f'{kind} Exposure +1 lights the frame twice as bright '
                  f'({road})', float(np.abs(k - 2.0).max()) < 2e-3,
                  f'x({k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f})')
        got = _roads(f'{kind} at 2000 K', [_export(
            kind, energy=e, use_temperature=True, temperature=2000.0,
            temperature_color=TEMP_2000)])
        for road, a, b in (('CPU', got[0], ref[0]), ('GPU twin', got[1],
                                                     ref[1])):
            if a is None or b is None:
                continue
            k = _lit_ratio(a, b)
            check(f'{kind} at 2000 K is the blackbody colour per channel '
                  f'({road})',
                  float(np.abs(k - np.asarray(TEMP_2000)).max()) < 4e-3,
                  f'x({k[0]:.4f}, {k[1]:.4f}, {k[2]:.4f})')
    ref = _cpu([_export('POINT', energy=600.0)])
    got = _roads('POINT Normalize off',
                 [_export('POINT', energy=600.0, normalize=False)])
    k = _lit_ratio(got[0], ref)
    check('a Point lamp of Radius 0.1 with Normalize off is 4 pi r^2 = '
          '0.1257 as bright', float(np.abs(k - 4.0 * math.pi * 0.01).max())
          < 2e-4, f'x{k[0]:.5f}')


# ------------------------------------------ (c) the Area lamp's power units


def _nadir(img):
    """The floor straight under a lamp hung over it and pointing down:
    the brightest pixel of the frame (the lamp's own peak)."""
    return float(np.asarray(img).mean(axis=2).max())


def test_c_area_power_units():
    """area-power-01 = native-area-power-units = area-distance-gain: a new
    Area lamp is in Blender's watts; the old law is one menu item away
    and bitwise the lamp it was."""
    _bpy()
    from .. import export as EX
    from .. import properties as PR
    check('the watts Distance is sqrt(2) = 1.4142136',
          abs(EX.AREA_WATTS_DISTANCE - math.sqrt(2.0)) < 1e-15)
    ann = PR.HalcyonLightSettings.__annotations__['area_power']
    items = ann.kw.get('items')
    check("Power Units is a lamp property, 'WATTS' by default (what a new "
          "lamp gets) with 'BI' the second item",
          ann.kw.get('default') == 'WATTS'
          and [it[0] for it in items] == ['WATTS', 'BI'])
    check('its tooltip and both items explain themselves (40 / 12 chars)',
          len(str(ann.kw.get('description') or '')) >= 40
          and all(len(it[1]) >= 12 and len(it[2]) >= 12 for it in items))
    # export
    bi = _export('AREA', hs=_hs(area_power='BI', decay_end=7.0,
                                decay='CUSTOM'))
    check("'Blender Internal (Distance)' exports the Distance dial and the "
          'falloff menu as they are', bi.decay_end == 7.0
          and bi.decay == 'CUSTOM')
    wt = _export('AREA', hs=_hs(area_power='WATTS', decay_end=7.0,
                                decay='CUSTOM'))
    check("'Blender Power (Watts)' exports the one Distance at which the "
          'form factor is watts, whatever the dial says',
          wt.decay_end == EX.AREA_WATTS_DISTANCE)
    st_c = _settings(light_falloff_default='CUSTOM')
    wd = _export('AREA', hs=_hs(area_power='WATTS'))
    check('...and never resolves to Custom Range, on the lamp or through '
          'Scene Default (the beam would be cut at 1.41 units)',
          LI.resolved_decay(wt, st_c) != 'CUSTOM'
          and LI.resolved_decay(wd, st_c) != 'CUSTOM'
          and LI.resolved_decay(bi, st_c) == 'CUSTOM')
    old_hs = _hs()
    del old_hs.area_power
    old = _export('AREA', hs=old_hs)
    check('a settings group without the option (a fake, an older panel) '
          "is the lamp it was: 'BI'",
          _fields(old) == _fields(_export('AREA', hs=_hs(area_power='BI'))))
    for kind in ('POINT', 'SUN', 'SPOT'):
        check(f'the option is read by Area lamps only ({kind} untouched)',
              _fields(_export(kind, hs=_hs(area_power='WATTS')))
              == _fields(_export(kind, hs=_hs(area_power='BI'))))
    # the radiometry: Blender's Area / Point ratio at equal Power is pi
    # (albedo P cos cos' / (4 pi r^2) against albedo P cos / (4 pi^2 r^2))
    st = _settings()
    hgt = 4.0
    pt = _export('POINT', energy=400.0, matrix=_down((0.0, 0.0, hgt)),
                 hs=_hs(decay='INVERSE_SQUARE', shadow='NONE'))
    ar_w = _export('AREA', energy=400.0, size=0.2,
                   matrix=_down((0.0, 0.0, hgt)),
                   hs=_hs(area_power='WATTS', shadow='NONE'))
    ar_b = _export('AREA', energy=400.0, size=0.2,
                   matrix=_down((0.0, 0.0, hgt)),
                   hs=_hs(area_power='BI', shadow='NONE'))
    n_pt = _nadir(_cpu(pt, st, box=False))
    n_w = _nadir(_cpu(ar_w, st, box=False))
    st_hi = _settings()
    n_b = _nadir(_cpu(_export(
        'AREA', energy=400.0 / 400.0, size=0.2,
        matrix=_down((0.0, 0.0, hgt)),
        hs=_hs(area_power='BI', shadow='NONE')), st_hi, box=False)) * 400.0
    check("in watts an Area lamp is 4 times the Point lamp of the same "
          "Power under its nadir, Blender's own ratio (a face of flux P "
          "radiates P / pi along its axis, a point P / 4 pi)",
          abs(n_w / n_pt / 4.0 - 1.0) < 0.01,
          f'Area / Point = {n_w / n_pt:.4f}')
    check('on Blender Internal units at the factory Distance 25 the same '
          'lamp is 25^2 / 2 = 312.5 times that',
          abs(n_b / n_w / 312.5 - 1.0) < 0.01, f'{n_b / n_w:.1f}')
    check('area_blender_watts says so: Power x Distance^2 / 2',
          EX.area_blender_watts(1.0, 25.0, 'BI') == 312.5
          and EX.area_blender_watts(10.0, 25.0, 'WATTS') == 10.0)
    # a new lamp at Blender's Add > Light > Area defaults (10 W, Size 1)
    new = _export('AREA', energy=10.0, size=1.0,
                  matrix=_matrix(pos=(2.0, -1.5, 4.0)),
                  hs=_hs(area_power='WATTS', shadow='MAP'))
    was = _export('AREA', energy=10.0, size=1.0,
                  matrix=_matrix(pos=(2.0, -1.5, 4.0)),
                  hs=_hs(area_power='BI', shadow='MAP'))
    f_new, g_new = _roads('a new 10 W Area lamp', [new])
    f_was = _cpu([was])
    check('a new 10 W Area lamp no longer whites out the frame',
          float((f_was >= 1.0).mean()) > 0.5
          and float((f_new >= 1.0).mean()) == 0.0
          and 0.005 < float(f_new.mean()) < 0.5,
          f'clipped {float((f_was >= 1.0).mean()) * 100:.1f}% -> '
          f'{float((f_new >= 1.0).mean()) * 100:.1f}%, mean '
          f'{float(f_was.mean()):.3f} -> {float(f_new.mean()):.4f}')
    # the Distance dial is not read in watts; Gamma still is
    a = _cpu([_export('AREA', energy=40.0,
                      hs=_hs(area_power='WATTS', decay_end=25.0))])
    b = _cpu([_export('AREA', energy=40.0,
                      hs=_hs(area_power='WATTS', decay_end=3.0))])
    c = _cpu([_export('AREA', energy=40.0,
                      hs=_hs(area_power='WATTS', area_gamma=0.6))])
    check('in watts the Distance dial changes no pixel, Gamma still '
          'shapes the field', np.array_equal(a, b)
          and not np.array_equal(a, c))
    # object scale: Blender keeps the POWER (the old law gains scale^2)
    far = dict(size=0.2, energy=400.0, shadow='NONE')

    def scaled(units, scale):
        return _nadir(_cpu(_export(
            'AREA', energy=far['energy'] if units == 'WATTS' else 1.0,
            size=far['size'], matrix=_down((0.0, 0.0, hgt), scale),
            hs=_hs(area_power=units, shadow='NONE')), st, box=False))
    kw = scaled('WATTS', (2.0, 3.0, 1.0)) / scaled('WATTS', (1.0, 1.0, 1.0))
    kb = scaled('BI', (2.0, 3.0, 1.0)) / scaled('BI', (1.0, 1.0, 1.0))
    check('scaling an Area lamp object (2, 3) keeps its power in watts '
          '(the far field within 2%)', abs(kw - 1.0) < 0.02, f'x{kw:.4f}')
    check('...and is still the gain of scale_x * scale_y on Blender '
          'Internal units', abs(kb / 6.0 - 1.0) < 0.02, f'x{kb:.4f}')
    # the importer keeps a 2.79 lamp on the old law
    from ..core import blend279_map as BM
    from .. import legacy_import as LEG
    parsed = {'kind': 'AREA', 'name': 'A', 'r': 1.0, 'g': 1.0, 'b': 1.0,
              'energy': 0.5, 'dist': 12.0, 'falloff_type': 1, 'mode': 0,
              'area_size': 1.0, 'area_sizey': 1.0, 'area_shape': 0}
    try:
        lm = BM.lamp_map(parsed, 279)
    except Exception:                                           # noqa: BLE001
        lm = {'type': 'AREA', 'name': 'A', 'color': (1.0, 1.0, 1.0),
              'energy': 0.5, 'distance': 12.0, 'falloff': 'INVERSE_LINEAR',
              'warnings': []}
    fake = types.SimpleNamespace(
        color=None, energy=0.0, halcyon=_hs(area_power='WATTS'))
    LEG._apply_lamp_bi(fake, dict(lm, type='AREA'))
    check("the legacy importer sets an Area lamp's units to Blender "
          'Internal (its energy and Distance are 2.79\'s)',
          fake.halcyon.area_power == 'BI')


# ------------------------------------------------ (d) the light-units sort


class _Group:
    """A property group that knows which of its values are stored."""

    def __init__(self, names, **stored):
        object.__setattr__(self, '_names', tuple(names))
        object.__setattr__(self, '_stored', {})
        object.__setattr__(self, '_defaults', {'area_power': 'WATTS',
                                               'light_units_version': 0})
        for k, v in stored.items():
            self._stored[k] = v

    @property
    def bl_rna(self):
        return types.SimpleNamespace(properties=[
            types.SimpleNamespace(identifier=n) for n in self._names])

    def is_property_set(self, name):
        return name in self._stored

    def property_unset(self, name):
        self._stored.pop(name, None)

    def __getattr__(self, name):
        st = object.__getattribute__(self, '_stored')
        if name in st:
            return st[name]
        df = object.__getattribute__(self, '_defaults')
        if name in df:
            return df[name]
        if name in object.__getattribute__(self, '_names'):
            return 0
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self._stored[name] = value


_L_NAMES = ('area_power', 'decay_end', 'shadow', 'area_gamma')
_S_NAMES = ('light_units_version', 'max_lights', 'shadows')
_UID = [100]


def _lamp(name, kind='AREA', library=None, **stored):
    _UID[0] += 1
    return types.SimpleNamespace(name=name, type=kind, library=library,
                                 session_uid=_UID[0],
                                 halcyon=_Group(_L_NAMES, **stored))


def _bscene(engine='BLENDER_EEVEE', library=None, **stored):
    return types.SimpleNamespace(
        library=library, render=types.SimpleNamespace(engine=engine),
        halcyon=_Group(_S_NAMES, **stored))


def _data(scenes, lights):
    return types.SimpleNamespace(scenes=list(scenes), lights=list(lights))


def _blend(marks, wrap=None):
    """A stand-in .blend on disk holding the given byte marks."""
    fd, path = tempfile.mkstemp(suffix='.blend', prefix='halcyon_units_')
    os.close(fd)
    body = b'BLENDER17-01' + b'\x00' * 300 + b''.join(
        m + b'\x00' * 77 for m in marks) + b'ENDB'
    if wrap == 'gzip':
        body = gzip.compress(body)
    elif wrap == 'zstd':
        import zstandard
        body = zstandard.ZstdCompressor().compress(body)
    with open(path, 'wb') as fh:
        fh.write(body)
    return path


def test_d_light_units_sort():
    """The migration: an Area lamp without a stored value keeps Blender
    Internal units when the file it was saved in is an older Halcyon
    file -- by every route a lamp can arrive."""
    bpy = _bpy()
    from .. import light_units as LU
    eng, stamp = LU.ENGINE.encode(), b'light_units_version'

    # ---- File > Open
    old_a, old_p = _lamp('KeyArea'), _lamp('Fill', kind='POINT')
    chosen = _lamp('Chosen', area_power='WATTS')
    data = _data([_bscene('HALCYON_RENDER')], [old_a, old_p, chosen])
    kept = LU.sort_file(data)
    check('an older Halcyon file: its Area lamp is set to Blender Internal '
          'units, named in the report',
          kept == ['KeyArea'] and old_a.halcyon.is_property_set('area_power')
          and old_a.halcyon.area_power == 'BI')
    check('...a Point lamp is not written to, a lamp whose units someone '
          'chose is left alone',
          not old_p.halcyon.is_property_set('area_power')
          and chosen.halcyon.area_power == 'WATTS')
    check('...and every local scene now carries the stamp',
          data.scenes[0].halcyon.light_units_version == LU.UNITS_VERSION)
    new_a = _lamp('NewArea')
    data.lights.append(new_a)
    check('a lamp added to the sorted file stays unset (watts), sort after '
          'sort', LU.sort_file(data) == []
          and not new_a.halcyon.is_property_set('area_power')
          and new_a.halcyon.area_power == 'WATTS')
    ev_a, ev_t = _lamp('Soft'), _lamp('Tuned', decay_end=4.0)
    data = _data([_bscene('BLENDER_EEVEE')], [ev_a, ev_t])
    kept = LU.sort_file(data)
    check('a file with no sign of Halcyon (EEVEE, no setting moved): its '
          'Area lamp opens in watts, as it was made',
          not ev_a.halcyon.is_property_set('area_power'))
    check('...but a lamp whose own Halcyon dials were moved was lit for '
          'Halcyon and keeps the old law', kept == ['Tuned'])
    wb = _lamp('PlayblastSaved')
    data = _data([_bscene('BLENDER_WORKBENCH', max_lights=4)], [wb])
    check('a Halcyon file saved on another engine is still a Halcyon file '
          '(a moved render setting says so)', LU.sort_file(data)
          == ['PlayblastSaved'])
    flag = LU.HALCYON_FILES_ONLY
    try:
        LU.HALCYON_FILES_ONLY = False
        ev2 = _lamp('Any')
        LU.sort_file(_data([_bscene('BLENDER_EEVEE')], [ev2]))
        check('HALCYON_FILES_ONLY = False keeps EVERY unsorted Area lamp '
              'on the old law (the stricter reading, one switch)',
              ev2.halcyon.area_power == 'BI')
    finally:
        LU.HALCYON_FILES_ONLY = flag

    # ---- library files, read from their bytes
    p_old = _blend([eng])
    p_new = _blend([eng, stamp])
    p_eev = _blend([b'BLENDER_EEVEE'])
    check('file_marks: (shows Halcyon, is sorted) from a plain file',
          LU.file_marks(p_old) == (True, False)
          and LU.file_marks(p_new) == (True, True)
          and LU.file_marks(p_eev) == (False, False))
    p_gz = _blend([eng], wrap='gzip')
    check('...through gzip', LU.file_marks(p_gz) == (True, False))
    try:
        import zstandard                                  # noqa: F401
        p_zs = _blend([eng, stamp], wrap='zstd')
        check('...through zstd (Blender 3.0+ compressed saves)',
              LU.file_marks(p_zs) == (True, True))
    except ImportError:
        note('zstandard is not importable here; Blender bundles it')
    check('...and None for a file that cannot be read',
          LU.file_marks(os.path.join(tempfile.gettempdir(),
                                     'halcyon_units_missing.blend')) is None)
    big = tempfile.mkstemp(suffix='.blend')[1]
    with open(big, 'wb') as fh:
        fh.write(b'\x00' * ((1 << 22) - 5) + eng + b'\x00' * 9)
    check('a mark lying across a read boundary is found',
          LU.file_marks(big) == (True, False))

    def lib(path):
        return types.SimpleNamespace(filepath=path, library=None)

    # linked lamps are judged by THEIR file at every load, whatever the
    # main file is
    l_old, l_new, l_eev = (_lamp('LibOld', library=lib(p_old)),
                           _lamp('LibNew', library=lib(p_new)),
                           _lamp('LibEevee', library=lib(p_eev)))
    main = _data([_bscene('HALCYON_RENDER',
                          light_units_version=LU.UNITS_VERSION)],
                 [l_old, l_new, l_eev])
    kept = LU.sort_file(main)
    check('a lamp linked from an older Halcyon library keeps the old law; '
          'from a sorted library, or one never used with Halcyon, watts',
          kept == ['LibOld'] and not l_new.halcyon.is_property_set(
              'area_power') and not l_eev.halcyon.is_property_set(
              'area_power'))

    # ---- Append / Link
    def item(idb, path, id_type='OBJECT'):
        return types.SimpleNamespace(
            id=idb, id_type=id_type,
            source_libraries=[types.SimpleNamespace(filepath=path)],
            source_library=None)

    a_old, a_new, a_eev, a_lost = (_lamp('A1'), _lamp('A2'), _lamp('A3'),
                                   _lamp('A4', shadow='RAY'))
    ob = types.SimpleNamespace(name='A1', type='LIGHT', data=a_old,
                               library=None)
    mesh = types.SimpleNamespace(name='Cube', type='MESH',
                                 data=types.SimpleNamespace(name='Cube'))
    kept = LU.sort_import([item(ob, p_old), item(a_old, p_old, 'LIGHT'),
                           item(mesh, p_old), item(a_new, p_new, 'LIGHT'),
                           item(a_eev, p_eev, 'LIGHT'),
                           item(a_lost, '/nowhere/gone.blend', 'LIGHT'),
                           item(None, p_old)])
    check('an appended Area lamp is judged by its source file: an older '
          'Halcyon file keeps the old law (once, though its object and '
          'its datablock both arrive), a sorted or an EEVEE file does not',
          kept.count('A1') == 1 and 'A2' not in kept and 'A3' not in kept
          and a_old.halcyon.area_power == 'BI')
    check('...and when the source cannot be read the lamp\'s own moved '
          'Halcyon dials decide', 'A4' in kept)
    hooks = bpy.app.handlers
    had_init = hasattr(hooks, 'render_init')
    if not had_init:
        hooks.render_init = []
    saved = {n: list(getattr(hooks, n)) for n in (
        'load_post', 'blend_import_post', 'undo_post', 'redo_post',
        'save_pre', 'render_init')}
    real_data = bpy.data
    try:
        # ---- the handlers, as Blender fires them
        k1, k2 = _lamp('Reg1'), _lamp('Reg2', kind='POINT')
        bpy.data = _data([_bscene('HALCYON_RENDER')], [k1, k2])
        LU.register()
        check('enabling the add-on over an open older file sorts it at once '
              'when bpy.data can be read',
              k1.halcyon.area_power == 'BI'
              and bpy.data.scenes[0].halcyon.light_units_version
              == LU.UNITS_VERSION)
        check('register hooks load_post, blend_import_post, undo_post, '
              'redo_post, save_pre and render_init, each marked persistent',
              all(any(getattr(fn, '__module__', '').endswith('light_units')
                      and hasattr(fn, '_bpy_persistent')
                      for fn in getattr(hooks, n)) for n in saved))
        # Blender swaps bpy.data for a stub while an add-on registers;
        # the real data is behind its restricted context
        LU.unregister()
        from . import fakebpy
        real_ctx = bpy.context
        r1 = _lamp('Unsaved')
        behind = _data([_bscene('HALCYON_RENDER')], [r1])
        bpy.data = types.SimpleNamespace()            # the stub
        bpy.context = types.SimpleNamespace(_real_data=behind)
        LU.register()
        made_after = _lamp('MadeAfter')
        behind.lights.append(made_after)
        bpy.data = behind
        bpy.app.timers.pump()
        fakebpy.fire('render_init', None)
        check('with bpy.data swapped out during register the open file is '
              'still sorted at once (a lamp never saved included), nothing '
              'is left pending, and a lamp a script makes afterwards is a '
              'new lamp',
              r1.halcyon.area_power == 'BI' and LU._PENDING[0] is False
              and not made_after.halcyon.is_property_set('area_power'))
        LU.unregister()
        # a build with no road to the data: the sort waits for the main
        # loop's next turn (a timer) or render_init, and then trusts
        # only the file on disk
        on_disk = _blend([eng, b'LAWait\x00', b'LAOther\x00'])
        for route in ('the timer', 'render_init'):
            w1, w2 = _lamp('Wait'), _lamp('MadeMeanwhile')
            waiting = _data([_bscene('HALCYON_RENDER')], [w1, w2])
            waiting.filepath = on_disk
            bpy.data = types.SimpleNamespace()
            bpy.context = types.SimpleNamespace()
            LU.register()
            wrote_early = w1.halcyon.is_property_set('area_power')
            bpy.data = waiting
            if route == 'the timer':
                bpy.app.timers.pump()
            else:
                fakebpy.fire('render_init', None)
            check(f'with no road to the data during register the sort is '
                  f'left pending and runs from {route}, once: the lamp the '
                  'file on disk holds keeps the old law, a lamp made in '
                  'the meantime is new',
                  not wrote_early and w1.halcyon.area_power == 'BI'
                  and not w2.halcyon.is_property_set('area_power')
                  and LU._PENDING[0] is False,
                  f'{wrote_early} {w1.halcyon.area_power}')
            bpy.app.timers.pump()
            LU.unregister()
            check(f'...and no timer is left behind ({route})',
                  not bpy.app.timers.is_registered(LU._sort_pending))
        check('file_holds_lamp reads a Light datablock\'s name from the '
              'file (code, name, terminator): Wait yes, Wai no',
              LU.file_holds_lamp(on_disk, 'Wait') is True
              and LU.file_holds_lamp(on_disk, 'Wai') is False
              and LU.file_holds_lamp('/nowhere/x.blend', 'Wait') is None)
        os.remove(on_disk)
        bpy.context = real_ctx
        k1, k2 = _lamp('Reg1'), _lamp('Reg2', kind='POINT')
        bpy.data = _data([_bscene('HALCYON_RENDER')], [k1, k2])
        LU.register()
        # undo to a state from before the add-on was enabled
        k1.halcyon.property_unset('area_power')
        fakebpy.fire('undo_post', None)
        check('an undo that restores the lamp without its value gets the '
              "'BI' written again", k1.halcyon.is_property_set('area_power')
              and k1.halcyon.area_power == 'BI')
        user = _lamp('UserNew')
        bpy.data.lights.append(user)
        fakebpy.fire('redo_post', None)
        check('...and never touches a lamp this session did not sort',
              not user.halcyon.is_property_set('area_power'))
        # save
        fakebpy.fire('save_pre', None)
        check('before a save every local Area lamp stores the units it is '
              'using (a lamp appended from this file later says what it is)',
              user.halcyon.is_property_set('area_power')
              and user.halcyon.area_power == 'WATTS'
              and k1.halcyon.area_power == 'BI'
              and not k2.halcyon.is_property_set('area_power'))
        # File > Open through the handler list
        o1 = _lamp('Opened')
        bpy.data = _data([_bscene('HALCYON_RENDER')], [o1])
        fakebpy.file_load('older.blend')
        check('File > Open runs the sort (the handler survives the load)',
              o1.halcyon.area_power == 'BI')
        # an import through the handler
        i1 = _lamp('Imported')
        ctx = types.SimpleNamespace(process_stage='DONE', options=set(),
                                    import_items=[item(i1, p_old, 'LIGHT')])
        fakebpy.fire('blend_import_post', ctx)
        check('File > Append runs the import sort',
              i1.halcyon.area_power == 'BI')
        LU.unregister()
        check('unregister removes every hook',
              not any(getattr(fn, '__module__', '').endswith('light_units')
                      for n in saved for fn in getattr(hooks, n)))
    finally:
        bpy.data = real_data
        if 'real_ctx' in locals():
            bpy.context = real_ctx
        for n, lst in saved.items():
            getattr(hooks, n)[:] = lst
        if not had_init:
            del hooks.render_init
        for p in (p_old, p_new, p_eev, p_gz, big):
            try:
                os.remove(p)
            except OSError:
                pass
    import halcyon as HPKG
    check('halcyon registers the light_units module',
          any(getattr(m, '__name__', '') == 'halcyon.light_units'
              for m in HPKG._import_modules()))


# ----------------------------------------------- (e) a Spot's object scale


def _blender_spotmask(d, spot_size, blend, sx, sy, sz):
    """Blender's light_spot_attenuation (eevee_light_lib) for directions
    d (light -> surface, lamp space, -Z forward), float64."""
    x, y, z = d[:, 0], d[:, 1], -d[:, 2]
    inv = (sz / sx, sz / sy)
    zz = np.maximum(z, 1e-30)
    ell = 1.0 / np.sqrt(1.0 + (x * inv[0] / zz) ** 2 + (y * inv[1] / zz) ** 2)
    ci = math.cos(spot_size * 0.5)
    t = np.clip((ell - ci) / ((1.0 - ci) * blend), 0.0, 1.0)
    m = t * t * (3.0 - 2.0 * t)
    return np.where(z > 0.0, m, 0.0)


def _halcyon_soft(lt, d):
    """The engine's cone factor without its cosine: support and plateau."""
    L = (-d).astype(f32)
    ax = np.asarray(lt.direction, f32)
    z = -(L @ (ax / np.linalg.norm(ax)))
    inpr = LI.scaled_cos(lt, L, z) if LI.spot_scale_on(lt) else z
    fac = LI.spot_falloff(lt, L).astype(np.float64)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(inpr > 0.0, fac / np.maximum(inpr, 1e-30), 0.0)


def _dirs(n=6000, half=1.45, seed=5):
    rng = np.random.default_rng(seed)
    th = rng.uniform(0.0, half, n)
    ph = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                     -np.cos(th)], axis=1)


def test_e_spot_object_scale():
    """native-spot-object-scale: Blender shapes a Spot's cone by its object
    scale (scale.z / scale.xy); nothing of it reached the engine."""
    _bpy()
    from .. import export as EX
    base = _export('SPOT')
    wide = _export('SPOT', scale=(2.0, 2.0, 1.0))
    narrow = _export('SPOT', scale=(1.0, 1.0, 2.0))
    check('scale (2, 2, 1) folds into the angles: Beam 45 -> 79.28 deg, '
          'Blend 0.15 -> 0.1190, no elliptical field',
          abs(math.degrees(wide.spot_size) - 79.28) < 0.01
          and abs(wide.spot_blend - 0.1190) < 1e-4
          and tuple(wide.spot_scale) == (1.0, 1.0),
          f'{math.degrees(wide.spot_size):.2f} deg, {wide.spot_blend:.4f}')
    check('scale (1, 1, 2) narrows it: 23.40 deg',
          abs(math.degrees(narrow.spot_size) - 23.40) < 0.01,
          f'{math.degrees(narrow.spot_size):.2f}')
    ell = _export('SPOT', scale=(2.0, 1.0, 1.0))
    check('scale (2, 1, 1): the wider axis folds (79.28 deg), the narrower '
          'one rides spot_scale = (1, 2)',
          abs(ell.spot_size - wide.spot_size) < 1e-12
          and np.allclose(ell.spot_scale, (1.0, 2.0), atol=1e-6),
          str(ell.spot_scale))
    hot = _export('SPOT', scale=(2.0, 2.0, 1.0),
                  hs=_hs(spot_law='POV', hotspot=0.4))
    check('the POV hotspot angle moves with the cone',
          abs(hot.hotspot - 2.0 * math.atan(2.0 * math.tan(0.2))) < 1e-9)
    full = _export('SPOT', scale=(2.0, 1.0, 1.0), spot_size=math.pi)
    check('a 180 degree cone scaled stays below 180 (89.9 a side)',
          full.spot_size <= math.radians(179.8) + 1e-9)
    # the edges land where Blender puts them, round and elliptical
    d = _dirs()
    for label, scale in (('(2, 2, 1)', (2.0, 2.0, 1.0)),
                         ('(1, 1, 2)', (1.0, 1.0, 2.0)),
                         ('(2, 1, 1)', (2.0, 1.0, 1.0)),
                         ('(1, 3, 2)', (1.0, 3.0, 2.0)),
                         ('(0.5, 1, 1)', (0.5, 1.0, 1.0))):
        lt = _export('SPOT', scale=scale, matrix=_down((0, 0, 4.0), scale))
        ref = _blender_spotmask(d, NATIVE['spot_size'], NATIVE['spot_blend'],
                                *scale)
        got = _halcyon_soft(lt, d)
        # compare away from the two edges themselves (a hair of slack)
        lit_b, lit_h = ref > 1e-4, got > 1e-4
        full_b, full_h = ref > 1.0 - 1e-4, got > 1.0 - 1e-4
        edge = (np.abs(ref - 1e-4) < 3e-3) | (np.abs(got - 1e-4) < 3e-3) \
            | (np.abs(ref - 1.0) < 3e-3) | (np.abs(got - 1.0) < 3e-3)
        ok = bool(np.all((lit_b == lit_h)[~edge])
                  and np.all((full_b == full_h)[~edge]))
        check(f'scale {label}: the cone lights the directions Blender\'s '
              'lights, at full strength where Blender\'s is',
              ok and int(lit_b.sum()) > 50 and int(full_b.sum()) > 20,
              f'{int(lit_b.sum())} lit, {int(full_b.sum())} full of '
              f'{d.shape[0]}; band max diff '
              f'{float(np.abs(ref - got).max()):.3f}')
    # scaled_cos against float64, and behind the lamp
    lt = _export('SPOT', scale=(2.0, 1.0, 1.0),
                 matrix=_down((0, 0, 4.0), (2.0, 1.0, 1.0)))
    L = (-d).astype(f32)
    L[-9:, 2] = -np.abs(L[-9:, 2])               # behind the lamp
    z = -(L @ np.asarray(lt.direction, f32))     # the unit axis here
    got = LI.scaled_cos(lt, L, z)
    s, u, _f = LI.cookie_frame(lt)
    x = -(L.astype(np.float64) @ s.astype(np.float64)) * lt.spot_scale[0]
    y = -(L.astype(np.float64) @ u.astype(np.float64)) * lt.spot_scale[1]
    zz = np.maximum(z.astype(np.float64), 1e-20)
    want = np.where(z > 0.0, 1.0 / np.sqrt(1.0 + (x / zz) ** 2
                                           + (y / zz) ** 2), 0.0)
    check('scaled_cos == 1/sqrt(1 + (sx x/z)^2 + (sy y/z)^2) within 1e-6 '
          'of float64, 0 behind the lamp, float32',
          float(np.abs(got - want).max()) < 1e-6
          and bool(np.all(got[z <= 0.0] == 0.0)) and got.dtype == np.float32,
          f'max {float(np.abs(got - want).max()):.2e}')
    sq = _export('SPOT', scale=(2.0, 1.0, 1.0),
                 matrix=_down((0, 0, 4.0), (2.0, 1.0, 1.0)),
                 hs=_hs(spot_square=True))
    gs = LI.scaled_cos(sq, L, z)
    m = np.maximum(np.abs(x / zz), np.abs(y / zz))
    want_s = np.where(z > 0.0, 1.0 / np.sqrt(1.0 + m * m), 0.0)
    check('with Square the wider squeezed side is taken (a rectangular '
          'pyramid)', float(np.abs(gs - want_s).max()) < 1e-6)
    # ---- the GLSL twin
    consts = {'light_texels': False, 'falloff_default': 'INVERSE_SQUARE',
              'cookies': {}, 'light_links': {}, 'resolution': (W, H),
              'specular_in_gamma': True, 'light_clamp': 0.0,
              'shadow_default': 'PER_LIGHT'}
    for label, lamp, ref_cos in (('elliptical', lt, got),
                                 ('elliptical + Square', sq, gs)):
        lines = GM._one_light_source(0, lamp, consts)
        blk, on = [], False
        for ln in lines:
            if 'float sc_x' in ln:
                on = True
            if on:
                blk.append(ln)
            if on and 'cosang = (cosang > 0.0)' in ln:
                break
        check(f'the {label} block is emitted after cosang, and the plain '
              'Square block is not emitted beside it',
              len(blk) >= 13 and not any('sq_x' in ln for ln in lines),
              str(len(blk)))
        n = L.shape[0]
        head = ('uniform vec3 L;\nuniform float cosang_in;\nout vec4 Color;'
                '\nvoid main()\n{\n    float cosang = cosang_in;\n')
        src = head + '\n'.join(blk) \
            + '\n    Color = vec4(cosang, 0.0, 0.0, 1.0);\n}\n'
        with np.errstate(over='ignore', invalid='ignore'):
            out, err = _run_glsl(src, n, L=L, cosang_in=z)
        if out is None:
            check(f'the {label} GLSL compiles standalone', False, str(err))
            continue
        dd = float(np.abs(out[:, 0] - ref_cos).max())
        check(f'the {label} block (with its dot products) equals '
              'scaled_cos within 2e-6', dd < 2e-6, f'd {dd:.3g}')
        tail = [ln for ln in blk if 'dot(' not in ln]
        xs = (-(L @ s.astype(f32))).astype(f32)
        ys = (-(L @ u.astype(f32))).astype(f32)
        src2 = ('uniform float sc_x_in;\nuniform float sc_y_in;\n'
                'uniform float cosang_in;\nout vec4 Color;\nvoid main()\n{\n'
                '    float cosang = cosang_in;\n    float sc_x = sc_x_in;\n'
                '    float sc_y = sc_y_in;\n' + '\n'.join(tail)
                + '\n    Color = vec4(cosang, 0.0, 0.0, 1.0);\n}\n')
        with np.errstate(over='ignore', invalid='ignore'):
            out2, err2 = _run_glsl(src2, n, sc_x_in=xs, sc_y_in=ys,
                                   cosang_in=z)
        if out2 is None:
            check(f'the {label} tail compiles standalone', False, str(err2))
        else:
            d2 = float(np.abs(out2[:, 0] - ref_cos).max())
            check(f'the {label} arithmetic equals scaled_cos bitwise '
                  '(d == 0.0)', d2 == 0.0, f'd {d2:.3g}')
    rnd = _export('SPOT', scale=(2.0, 2.0, 1.0))
    check('a round (folded) spot emits no elliptical block at all',
          not any('sc_x' in ln for ln in GM._one_light_source(0, rnd,
                                                              consts)))
    # ---- structure and texels
    e2 = _export('SPOT', scale=(3.0, 1.0, 1.5),
                 matrix=_down((0, 0, 4.0), (3.0, 1.0, 1.5)))
    e3 = _export('SPOT', scale=(4.0, 1.0, 2.0),
                 matrix=_down((0, 0, 4.0), (4.0, 1.0, 2.0)))
    r0 = _export('SPOT', matrix=_down((0, 0, 4.0)))
    check('an elliptical cone is light-signature structure (plan MISS '
          'against the round lamp); its squeeze is a texel (dragging the '
          'scale is a plan HIT)',
          GSH._light_sig(e2) != GSH._light_sig(r0)
          and GSH._light_sig(e2) == GSH._light_sig(e3))
    tx_r = GM.pack_light_texels([r0])
    tx_e = GM.pack_light_texels([e2])
    check('texels 10.w / 11.w carry the squeeze; a round lamp writes '
          'zeros there, as before',
          tx_r[0, 10, 3] == 0.0 and tx_r[0, 11, 3] == 0.0
          and tx_e[0, 10, 3] == f32(e2.spot_scale[0])
          and tx_e[0, 11, 3] == f32(e2.spot_scale[1]),
          f'{tx_e[0, 10, 3]}, {tx_e[0, 11, 3]}')
    # ---- the frames, both roads
    pos = (0.5, -1.0, 5.0)
    for label, scale, mode in (('round fold (2, 2, 1)', (2.0, 2.0, 1.0),
                                'MAP'),
                               ('elliptical (2, 1, 1), map shadow',
                                (2.0, 1.0, 1.0), 'MAP'),
                               ('elliptical (1, 2.5, 1), no shadow',
                                (1.0, 2.5, 1.0), 'NONE'),
                               ('elliptical + Square (2, 1, 1)',
                                (2.0, 1.0, 1.0), 'NONE')):
        hs = _hs(shadow=mode, spot_square='Square' in label)
        lamp = _export('SPOT', energy=2500.0, scale=scale, hs=hs,
                       matrix=_matrix(scale, pos=pos, aim=(0, 0, 0)))
        _roads(f'spot {label}', [lamp])
    one = _cpu([_export('SPOT', energy=2500.0, hs=_hs(shadow='NONE'),
                        matrix=_matrix(pos=pos, aim=(0, 0, 0)))])
    two = _cpu([_export('SPOT', energy=2500.0, scale=(2.0, 2.0, 1.0),
                        hs=_hs(shadow='NONE'),
                        matrix=_matrix((2.0, 2.0, 1.0), pos=pos,
                                       aim=(0, 0, 0)))])
    half = _cpu([_export('SPOT', energy=2500.0, scale=(1.0, 1.0, 2.0),
                         hs=_hs(shadow='NONE'),
                         matrix=_matrix((1.0, 1.0, 2.0), pos=pos,
                                        aim=(0, 0, 0)))])
    n1, n2, nh = (int((f.sum(2) > 1e-3).sum()) for f in (one, two, half))
    check('scale (2, 2, 1) lights a wider patch of the floor, (1, 1, 2) a '
          'narrower one (they rendered identically before)',
          n2 > 2.0 * n1 and nh < 0.5 * n1, f'{nh} < {n1} < {n2} px')
    ex = _cpu([_export('SPOT', energy=2500.0, scale=(2.0, 1.0, 1.0),
                       hs=_hs(shadow='NONE'),
                       matrix=_matrix((2.0, 1.0, 1.0), pos=pos,
                                      aim=(0, 0, 0)))])
    ne = int((ex.sum(2) > 1e-3).sum())
    check('scale (2, 1, 1) lights an ellipse: more than the unit cone, '
          'less than the round (2, 2, 1) one that circumscribes it',
          n1 < ne < n2 and float((ex - two).max()) < 1e-6,
          f'{n1} < {ne} < {n2} px')
    # ---- the shadow buffer covers the wide axis without a new rule
    st = _settings()
    a = _export('SPOT', energy=2500.0, scale=(2.0, 1.0, 1.0),
                hs=_hs(shadow='MAP'),
                matrix=_matrix((2.0, 1.0, 1.0), pos=pos, aim=(0, 0, 0)))
    b = _export('SPOT', energy=2500.0, scale=(2.0, 2.0, 1.0),
                hs=_hs(shadow='MAP'),
                matrix=_matrix((2.0, 2.0, 1.0), pos=pos, aim=(0, 0, 0)))
    for lamp in (a, b):
        LI.clear_shadow_cache()
        sc = _scene([lamp], st)
        LI.build_shadow_maps(sc, st)
    check("the elliptical lamp's shadow buffer is the circumscribing round "
          "cone's (same extent), so every lit texel has a depth",
          a.shadow_map is not None and b.shadow_map is not None
          and float(a.shadow_map.extent) == float(b.shadow_map.extent),
          f'{float(a.shadow_map.extent):.4f}')
    # ---- the slide is squeezed with the cone
    ck = np.zeros((8, 8, 3), f32)
    ck[:, :4] = 1.0                               # left half bright
    hs = _hs(shadow='NONE')
    lamp = _export('SPOT', energy=2500.0, scale=(1.0, 2.0, 1.0), hs=hs,
                   matrix=_down((0, 0, 4.0), (1.0, 2.0, 1.0)))
    lamp.cookie = ck
    lamp.cookie_filter = 'CLOSEST'
    t = math.tan(lamp.spot_size * 0.5)
    # points 0.9 of the way to the cone's edge along the squeezed X axis
    P = np.array([[-0.9 * t * 4.0 / lamp.spot_scale[0], 0.0, 0.0],
                  [0.9 * t * 4.0 / lamp.spot_scale[0], 0.0, 0.0]], f32)
    Lp = (np.asarray(lamp.position, f32)[None, :] - P)
    Lp = Lp / np.linalg.norm(Lp, axis=1, keepdims=True)
    cf = LI.cookie_factor(lamp, P, Lp.astype(f32))
    check('the projected slide is squeezed with the elliptical cone: its '
          'left and right edges sit on the cone\'s edges',
          float(cf[0].mean()) > 0.9 and float(cf[1].mean()) < 0.1,
          f'{cf[0].mean():.2f} / {cf[1].mean():.2f}')
    lamp2 = _export('SPOT', energy=2500.0, scale=(1.0, 2.0, 1.0),
                    hs=_hs(shadow='NONE'),
                    matrix=_matrix((1.0, 2.0, 1.0), pos=pos, aim=(0, 0, 0)))
    lamp2.cookie = ck
    _roads('an elliptical spot projecting a slide', [lamp2])
    r_blur = _export('SPOT', matrix=_down((0, 0, 4.0)))
    e_blur = _export('SPOT', scale=(2.0, 1.0, 1.0),
                     matrix=_down((0, 0, 4.0), (2.0, 1.0, 1.0)))
    bu, bv = LI.cookie_blur_uv(e_blur)
    check("the slide's focus blur grows across the squeezed axis only",
          LI.cookie_blur_uv(r_blur)[0] == LI.cookie_blur_uv(r_blur)[1]
          and abs(bv / bu - 2.0) < 1e-6, f'{bu:.4f}, {bv:.4f}')
    # ---- the beam is the same elliptical cone
    lamp = _export('SPOT', energy=10.0, scale=(1.0, 2.5, 1.0),
                   hs=_hs(shadow='NONE', volumetric=1.0),
                   matrix=_down((0.0, 0.0, 4.0), (1.0, 2.5, 1.0)))
    rng = np.random.default_rng(11)
    n = 400
    org = np.tile(np.array([[0.0, -9.0, 2.0]], f32), (n, 1))
    tgt = np.stack([rng.uniform(-4, 4, n), rng.uniform(-2, 2, n),
                    rng.uniform(0.2, 3.5, n)], axis=1)
    rays = tgt - org
    rays = (rays / np.linalg.norm(rays, axis=1, keepdims=True)).astype(f32)
    depth = np.full(n, 30.0, f32)
    fast = CONES.spot_cone(org, rays, depth, lamp, samples=256)
    # a brute-force march that shares no code with the segment solve
    apex = np.asarray(lamp.position, np.float64)
    s, u, fw = (np.asarray(v, np.float64) for v in LI.cookie_frame(lamp))
    ch = math.cos(lamp.spot_size * 0.5)
    inner = ch + (1.0 - ch) * min(max(lamp.spot_blend, 0.0), 0.999)
    steps = 6000
    tot = np.zeros(n)
    dt = 30.0 / steps
    for i in range(steps):
        p = org.astype(np.float64) + rays.astype(np.float64) * ((i + 0.5) * dt)
        dv = p - apex[None, :]
        xq = (dv @ s) * lamp.spot_scale[0]
        yq = (dv @ u) * lamp.spot_scale[1]
        zq = dv @ fw
        cq = zq / np.sqrt(np.maximum(xq * xq + yq * yq + zq * zq, 1e-12))
        a_ = np.clip((cq - ch) / max(inner - ch, 1e-6), 0.0, 1.0)
        a_ = a_ * a_ * (3.0 - 2.0 * a_)
        dist = np.sqrt(np.maximum((dv * dv).sum(1), 1e-6))
        tot += np.where(cq > ch, a_ / np.maximum(dist ** 2.0, 1e-6), 0.0) * dt
    live = tot > 1e-4
    rel = np.abs(fast[live] - tot[live]) / tot[live]
    check('the volumetric beam is the same elliptical cone: the segment '
          'solve matches a brute-force march through the squeezed cone',
          int(live.sum()) > 40 and float(np.median(rel)) < 0.02
          and float(np.abs(fast[~live]).max(initial=0.0)) < 2e-3,
          f'{int(live.sum())} rays in the beam, median rel err '
          f'{float(np.median(rel)):.1e}, worst {float(rel.max()):.1e}')
    rl = _export('SPOT', energy=10.0, hs=_hs(shadow='NONE', volumetric=1.0),
                 matrix=_down((0.0, 0.0, 4.0)))
    rl2 = _export('SPOT', energy=10.0, hs=_hs(shadow='NONE', volumetric=1.0),
                  matrix=_down((0.0, 0.0, 4.0)))
    rl2.spot_scale = (1.0, 1.0)
    check('a round lamp marches the round cone it always did',
          np.array_equal(CONES.spot_cone(org, rays, depth, rl, samples=24),
                         CONES.spot_cone(org, rays, depth, rl2, samples=24)))
    check('spot_scale_fold is exported for the tests and the docs',
          callable(getattr(EX, 'spot_scale_fold', None)))
    # ---- the strict front-end, where one is installed (the driver's
    # stand-in: tests/glsl_strict.py compiles the demo scene, which has
    # no squeezed lamp)
    from . import glsl_strict as GS
    if GS._glslang() is None:
        note('HALCYON_GLSLANG is not set: the strict GLSL compile of the '
             'elliptical block did not run here')
        return
    for label, square, slide in (('elliptical', False, False),
                                 ('elliptical + Square', True, False),
                                 ('elliptical with a slide', False, True)):
        lamp = _export('SPOT', energy=2500.0, scale=(2.0, 1.0, 1.0),
                       hs=_hs(shadow='MAP', spot_square=square),
                       matrix=_matrix((2.0, 1.0, 1.0), pos=pos,
                                      aim=(0, 0, 0)))
        if slide:
            lamp.cookie = ck
        st = _settings()
        srcs, why = GS._pass_sources(_scene([lamp], st), st)
        if srcs is None:
            check(f'strict GLSL: the {label} frame plans', False, str(why))
            continue
        bad = [msg for ok, msg in (GS.compile_source(src) for src in srcs)
               if not ok]
        check(f'strict GLSL 4.30 accepts every pass of the {label} frame '
              '(the block is in the text)',
              not bad and any('sc_x' in src for src in srcs),
              bad[0][:200] if bad else f'{len(srcs)} passes')


# --------------------------------------------------------- (f) scene time


def _bs(frame, fps=24, base=1.0, sub=0.0):
    return types.SimpleNamespace(
        frame_current=frame, frame_subframe=sub,
        render=types.SimpleNamespace(fps=fps, fps_base=base))


def test_f_scene_time():
    """style-time-fpsbase-01: scene time ignored the frame-rate base and
    the subframe."""
    _bpy()
    from .. import export as EX
    for fr in (0, 1, 3, 7, 250):
        check(f'fps 24, base 1, frame {fr}: the old expression, bitwise',
              EX.scene_time(_bs(fr)) == fr / 24
              and repr(EX.scene_time(_bs(fr))) == repr(fr / max(24, 1)))
    check('fps 48 / base 2 is fps 24 / base 1 (the real frame rate)',
          abs(EX.scene_time(_bs(5, fps=48, base=2.0)) - 5 / 24) < 1e-12,
          f'{EX.scene_time(_bs(5, fps=48, base=2.0)):.6f}')
    check('fps 30 / base 1.001 (NTSC): 1001 frames are 33.4 s, not 33.37',
          abs(EX.scene_time(_bs(1001, fps=30, base=1.001)) - 33.4000334)
          < 1e-6)
    check('the subframe counts: frame 2.5 at 25 fps is 0.1 s',
          abs(EX.scene_time(_bs(2, fps=25, sub=0.5)) - 0.1) < 1e-12)
    # an accumulated motion blur: FAST_STROBE at frame 3, a 0.5 shutter
    # in 4 steps lies inside ONE dark step of the style
    lamp = Light(type='POINT', light_style='FAST_STROBE',
                 light_style_rate=10.0)
    offs = np.linspace(-0.25, 0.25, 4)
    gains = []
    for off in offs:
        b = int(np.floor(3 + off))
        gains.append(LI.style_gain(lamp, EX.scene_time(
            _bs(b, sub=float(3 + off - b)))))
    check('a strobe that is dark across the whole shutter stays dark '
          'under motion blur (each step read a whole frame: 0.5 before)',
          max(gains) == 0.0, str(gains))
    gains = []
    for off in offs:
        b = int(np.floor(5 + off))
        gains.append(LI.style_gain(lamp, EX.scene_time(
            _bs(b, sub=float(5 + off - b)))))
    check('...and frame 5, whose shutter crosses the 0.2 s step once, is '
          'three quarters lit', abs(float(np.mean(gains)) - 0.75) < 1e-12,
          str(gains))
    # the viewport's lights-only export reads the same clock
    parked = Scene(lights=[], time=99.0)
    dg = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            frame_current=6, frame_subframe=0.0,
            render=types.SimpleNamespace(fps=48, fps_base=2.0),
            unit_settings=types.SimpleNamespace(scale_length=1.0)),
        object_instances=[])
    nxt = EX.export_lights_into(parked, dg)
    check('the lights-only export stamps the same time on its scene',
          abs(nxt.time - 6 / 24) < 1e-12 and parked.time == 99.0,
          f'{nxt.time}')


# ------------------------------------- (g) Blender's hidden light switches


def test_g_hidden_switches():
    """native-hidden-unexported and the lamp object's ray visibility: what
    has an exact equal here is honoured, the rest stays unread."""
    sq = _export('SPOT', use_square=True)
    check("Blender's own Square flag (a 2.79 file opened directly keeps "
          'it) exports the Square cone',
          sq.spot_square is True and _export('SPOT').spot_square is False)
    for label, kw, ob, want in (
            ('Diffuse factor 0', {'diffuse_factor': 0.0}, {},
             (False, True)),
            ('Glossy factor 0', {'specular_factor': 0.0}, {},
             (True, False)),
            ('the lamp object hidden from diffuse rays', {},
             {'visible_diffuse': False}, (False, True)),
            ('the lamp object hidden from glossy rays', {},
             {'visible_glossy': False}, (True, False)),
            ('factors of 0.5 (no exact equal: unread)',
             {'diffuse_factor': 0.5, 'specular_factor': 0.5}, {},
             (True, True))):
        lt = _export('POINT', ob=ob, **kw)
        check(f'{label}: affect_diffuse, affect_specular = {want}',
              (lt.affect_diffuse, lt.affect_specular) == want)
    # the frames: a lamp with its diffuse off lights no Lambert floor
    ref, _g = _roads('a Point lamp', [_export('POINT', energy=600.0)],
                     spec=0.5)
    off, _g = _roads('a Point lamp hidden from diffuse rays',
                     [_export('POINT', energy=600.0,
                              ob={'visible_diffuse': False})], spec=0.5)
    only, _g = _roads('a Point lamp with Diffuse Only set',
                      [_export('POINT', energy=600.0,
                               hs=_hs(specular_only=True))], spec=0.5)
    check('a lamp hidden from diffuse rays renders as Halcyon\'s own '
          'Specular Only lamp (it rendered in full before)',
          np.array_equal(off, only) and float(ref.mean()) > 4.0
          * float(off.mean()),
          f'mean {float(ref.mean()):.4f} -> {float(off.mean()):.4f}')
    for name in ('use_custom_distance', 'cutoff_distance',
                 'transmission_factor', 'volume_factor',
                 'shadow_filter_radius', 'use_shadow_jitter', 'spread',
                 'use_soft_falloff'):
        a = _fields(_export('POINT'))
        b = _fields(_export('POINT', **{name: 0.37}))
        if a != b:
            check(f'{name} has no equal here and stays unread', False)
            break
    else:
        check('Custom Distance, the Transmission / Volume factors, the '
              'EEVEE shadow dials, Spread and Soft Falloff stay unread',
              True)


def test_i_mirrored_area_lamp():
    """Negative object scale (a mirrored rig): an Area lamp mirrored
    across one face axis lit nothing; Blender keeps its handedness."""
    pos = (1.5, -1.0, 4.0)
    frames = {}
    for label, scale in (('unit', (1.0, 1.0, 1.0)),
                         ('mirrored in X', (-1.0, 1.0, 1.0)),
                         ('mirrored in Y', (1.0, -1.0, 1.0)),
                         ('mirrored in X and Y', (-1.0, -1.0, 1.0)),
                         ('mirrored in Z', (1.0, 1.0, -1.0))):
        lamp = _export('AREA', energy=40.0, scale=scale,
                       hs=_hs(area_power='WATTS', shadow='NONE'),
                       matrix=_matrix(scale, pos=pos, aim=(0, 0, 0)))
        if label == 'unit':
            check('an unmirrored Area lamp exports the axes it always did',
                  _fields(lamp)['area_y'] == _fields(_export(
                      'AREA', matrix=_matrix(pos=pos, aim=(0, 0, 0))))[
                          'area_y'])
        frames[label] = _roads(f'an Area lamp {label}', [lamp])[0]
    ref = float(frames['unit'].mean())
    for label, was in (('mirrored in X', 'it rendered black before'),
                       ('mirrored in Y', 'it rendered black before'),
                       ('mirrored in X and Y', 'a half turn: it always did')):
        d = float(np.abs(frames[label] - frames['unit']).max())
        check(f'an Area lamp {label} lights the scene as the unmirrored '
              f'lamp does ({was})',
              ref > 0.01 and d < 2e-5, f'max diff {d:.2e}, mean {ref:.4f}')
    check('an Area lamp mirrored in Z points the other way and lights '
          'nothing here, as in Blender',
          float(frames['mirrored in Z'].max()) == 0.0)
    sp = {}
    for label, scale in (('unit', (1.0, 1.0, 1.0)),
                         ('mirrored in X', (-1.0, 1.0, 1.0))):
        sp[label] = _cpu([_export('SPOT', energy=2500.0, scale=scale,
                                  hs=_hs(shadow='NONE'),
                                  matrix=_matrix(scale, pos=pos,
                                                 aim=(0, 0, 0)))])
    check('a Spot mirrored in X keeps its cone (it always did)',
          np.array_equal(sp['unit'], sp['mirrored in X']))


# ------------------------------------------------------ (h) the Light panel


class _Layout:
    """A recording layout: enough of bpy's UILayout for a panel draw."""

    def __init__(self, rec, active=None):
        object.__setattr__(self, 'rec', rec)
        object.__setattr__(self, '_active', [True] if active is None
                           else active)
        object.__setattr__(self, 'use_property_split', False)
        object.__setattr__(self, 'use_property_decorate', True)

    def __setattr__(self, k, v):
        if k == 'active':
            self._active[0] = bool(v)
        else:
            object.__setattr__(self, k, v)

    def prop(self, data, name, **kw):
        self.rec.append((name, kw.get('text'), self._active[0]))

    def column(self, **kw):
        return _Layout(self.rec, [self._active[0]])

    row = column
    box = column

    def separator(self, **kw):
        pass

    def label(self, **kw):
        self.rec.append(('label', str(kw.get('text', '')), self._active[0]))

    def template_ID(self, data, name, **kw):
        self.rec.append((name, None, self._active[0]))

    def operator(self, idn, **kw):
        self.rec.append(('op:' + idn, None, self._active[0]))
        return types.SimpleNamespace(scope=None)


def _draw(panel_cls, light, ob=None):
    rec = []
    panel = panel_cls()
    panel.layout = _Layout(rec)
    panel_cls.draw(panel, types.SimpleNamespace(
        engine='HALCYON_RENDER', light=light, scene=None, object=ob))
    return rec


def test_h_light_panel():
    """native-light-panel-duplicate, native-soft-falloff-dead,
    native-use-shadow-hidden (= use-shadow-hidden-01): one Light panel
    that draws what the engine reads."""
    _bpy()
    from .. import engine as ENG
    from .. import ui as UI
    forced = set(ENG.FORCED_PANELS)
    check('the stock Light panels are no longer forced on (two were '
          'titled Light; DATA_PT_area never existed)',
          not forced & {'DATA_PT_light', 'DATA_PT_EEVEE_light',
                        'DATA_PT_spot', 'DATA_PT_area'})
    check('...and the two that carry the generic marker are excluded',
          {'DATA_PT_light', 'DATA_PT_spot'} <= set(ENG.EXCLUDED_PANELS))
    check("the lamp's datablock selector is still shown",
          'DATA_PT_context_light' in forced)
    P = UI.HALCYON_PT_light_native
    check('Halcyon owns a panel titled Light, registered above Halcyon '
          'Light',
          P.bl_label == 'Light' and P.bl_context == 'data'
          and UI.CLASSES.index(P) < UI.CLASSES.index(UI.HALCYON_PT_light))
    lamp = types.SimpleNamespace(type='POINT', color=(1, 1, 1), energy=10.0)
    ctx = types.SimpleNamespace(engine='HALCYON_RENDER', light=lamp)
    check('it polls for a lamp under Halcyon only',
          P.poll(ctx) and not P.poll(types.SimpleNamespace(
              engine='BLENDER_EEVEE', light=lamp))
          and not P.poll(types.SimpleNamespace(engine='HALCYON_RENDER',
                                               light=None)))
    common = ['type', 'use_temperature', 'temperature', 'color', 'energy',
              'exposure', 'normalize']
    want = {
        'POINT': common + ['shadow_soft_size', 'use_shadow'],
        'SUN': common + ['angle', 'use_shadow'],
        'SPOT': common + ['shadow_soft_size', 'spot_size', 'spot_blend',
                          'show_cone', 'use_shadow'],
        'AREA': common + ['area_power', 'shape', 'size', 'use_shadow'],
    }
    for kind in ('POINT', 'SUN', 'SPOT', 'AREA'):
        la = types.SimpleNamespace(type=kind, halcyon=_hs(
            area_power='WATTS'), **NATIVE)
        rec = _draw(P, la)
        names = [n for n, _t, _a in rec if n != 'label']
        check(f'{kind}: the Light panel draws exactly the native rows the '
              'engine reads, in order', names == want[kind], str(names))
        check(f'{kind}: no Soft Falloff row (the lamp lights from its '
              'centre; the option has no meaning here)',
              'use_soft_falloff' not in names)
        check(f'{kind}: Cast Shadow is drawn (Blender\'s own switch vetoes '
              'the Shadows menu and was drawn nowhere)',
              ('use_shadow', 'Cast Shadow', True) in rec)
        # the stub's four-field lamp draws too
        bare = types.SimpleNamespace(type=kind, color=(1, 1, 1),
                                     energy=10.0, halcyon=_hs())
        try:
            _draw(P, bare)
            ok = True
        except Exception as e:                                  # noqa: BLE001
            ok = repr(e)
        check(f'{kind}: a lamp carrying only type, color, energy and '
              'halcyon draws (every read is defensive)', ok is True, str(ok))
        hal = [n for n, _t, _a in _draw(UI.HALCYON_PT_light, la)]
        check(f'{kind}: Halcyon Light no longer repeats Colour and Power',
              'color' not in hal and 'energy' not in hal)
    # Temperature relabels the Colour
    la = types.SimpleNamespace(type='POINT', halcyon=_hs(), **dict(
        NATIVE, use_temperature=True))
    rec = _draw(P, la)
    check('with Temperature on the Colour row reads Tint and the Kelvin '
          'field is live; off, Color and greyed',
          ('color', 'Tint', True) in rec
          and ('temperature', '', True) in rec
          and ('color', 'Color', True) in _draw(P, types.SimpleNamespace(
              type='POINT', halcyon=_hs(), **NATIVE))
          and ('temperature', '', False) in _draw(
              P, types.SimpleNamespace(type='POINT', halcyon=_hs(),
                                       **NATIVE)))
    # the Area lamp's units
    la_b = types.SimpleNamespace(type='AREA', halcyon=_hs(
        area_power='BI', decay_end=25.0), **dict(NATIVE, energy=0.05))
    rec = _draw(P, la_b)
    names = [n for n, _t, _a in rec]
    i = names.index('area_power')
    check('an Area lamp on Blender Internal units draws Distance under '
          'Power Units and says what the lamp gives in Blender\'s watts',
          rec[i + 1][:2] == ('decay_end', 'Distance')
          and rec[i + 2][0] == 'label' and '15.62 W' in rec[i + 2][1],
          str(rec[i:i + 3]))
    la_w = types.SimpleNamespace(type='AREA', halcyon=_hs(
        area_power='WATTS'), **NATIVE)
    check('...in watts no Distance is drawn anywhere (it is not read)',
          'decay_end' not in [n for n, _t, _a in _draw(P, la_w)]
          and 'decay_end' not in [n for n, _t, _a in _draw(
              UI.HALCYON_PT_light, la_w)])
    hal = [n for n, _t, _a in _draw(UI.HALCYON_PT_light, la_b)]
    check('Gamma stays in Halcyon Light, with no falloff menu',
          'area_gamma' in hal and 'decay' not in hal)
    rect = types.SimpleNamespace(type='AREA', halcyon=_hs(), **dict(
        NATIVE, shape='RECTANGLE'))
    check('a Rectangle draws Size X and Y',
          ('size', 'Size X', True) in _draw(P, rect)
          and ('size_y', 'Y', True) in _draw(P, rect))
    # the rows that appear only while a hidden switch is in force
    sqr = types.SimpleNamespace(type='SPOT', halcyon=_hs(), **dict(
        NATIVE, use_square=True))
    check("Blender's Square flag is drawn while it is set",
          'use_square' in [n for n, _t, _a in _draw(P, sqr)])
    gl = types.SimpleNamespace(type='SPOT', halcyon=_hs(spot_law='GL11'),
                               **NATIVE)
    check('Blend is greyed under a cone law that does not read it',
          ('spot_blend', 'Blend', False) in _draw(P, gl))
    dim = types.SimpleNamespace(type='POINT', halcyon=_hs(), **dict(
        NATIVE, diffuse_factor=0.0))
    ob = types.SimpleNamespace(data=dim, visible_diffuse=True,
                               visible_glossy=False)
    rec = _draw(P, dim, ob)
    check('a lobe switched off by Blender\'s Influence settings is named '
          'and its switch drawn, so nothing hidden darkens a lamp',
          ('diffuse_factor', 'Diffuse Factor', True) in rec
          and ('visible_glossy', 'Glossy Ray Visibility', True) in rec
          and any(n == 'label' and 'Influence' in t for n, t, _a in rec))
    check('...and a default lamp shows none of those rows',
          not any(n in ('diffuse_factor', 'specular_factor',
                        'visible_diffuse', 'visible_glossy', 'use_square')
                  for n, _t, _a in _draw(P, types.SimpleNamespace(
                      type='SPOT', halcyon=_hs(), **NATIVE))))


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
    print('all R254 native light-option checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
