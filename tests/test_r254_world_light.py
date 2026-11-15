"""R254: World Light -- the world's own luminary as a scene light.

With `World.world_light` on, the sun or moon the sky DRAWS becomes one
ordinary SUN Light at render entry (core/worldlight.py): the Bryce sun or
moon, the Physical sky's sun, a Sky Texture node's sun, the brightest
region of an HDRI / cube map (or a direction typed in by hand).

    neutrality   off by default; a scene without the toggle is not touched
                 (same list object, bitwise the same frame whatever the
                 other dials hold); select_lights is the old function for
                 every list that carries no world light
    luminaries   each mode's direction is where the PICTURE shows the
                 body (the sky sampled and its brightest point found),
                 under a World rotation too; moon phase, horizon, strength
    image search every mapping, layout, convention and turn; the refusals
    the frame    a lamp-less scene is lit and shadowed from the luminary's
                 side; bitwise the frame of a hand-made Sun lamp
    lamp-ness    the light limit never drops it, shafts, re-sync after a
                 lamp-only export, a world edit re-lights
    GPU twin     the deferred pass at the frame bar for every mode, the
                 energy texel, the fake device through render()
    wiring       properties, tooltips, presets, panel, feature matrix

    python -m halcyon.tests.test_r254_world_light
"""
import copy
import dataclasses
import importlib
import inspect
import io
import os
import sys
import traceback
from contextlib import redirect_stdout

import numpy as np

from . import utf8_console
from . import featurematrix as FM
from .scenebuild import demo_scene
from .test_render import _sk, _wnode, base_settings
from ..core import lights as LI
from ..core import raster as CRg
from ..core import render as R
from ..core import sky as SK
from ..core import worldlight as WL
from ..core.scene import ImageBuffer, Light, World
from ..core.texture import Texture
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
W, H = 96, 72
FIELDS = ('world_light', 'world_light_strength', 'world_light_shadow',
          'world_light_shafts', 'world_light_find', 'world_light_elevation',
          'world_light_azimuth', 'world_light_color')
#: a sun behind the ball, away from the camera: its shadow falls on the
#: open floor in front of the ball
SUN_EL, SUN_AZ = 0.7, 2.4
BALL = np.array([-1.3, 0.2, 1.0], f32)


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


# ------------------------------------------------------------------ helpers


def _settings(**kw):
    st = base_settings(W, H, shadows=True, transparency='NONE')
    st.show_stats = False
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _nolamp(st, mode='BRYCE', **world):
    """The demo scene with no lamps under a `mode` world, World Light on."""
    sc = demo_scene(st, with_texture=False)
    sc.lights = []
    sc.world.mode = mode
    sc.world.clouds = False
    sc.world.sun_elevation = SUN_EL
    sc.world.sun_rotation = SUN_AZ
    sc.world.world_light = True
    for k, v in world.items():
        setattr(sc.world, k, v)
    return sc


def _render(sc, st):
    with redirect_stdout(io.StringIO()):
        return np.asarray(R.render(sc, st)).copy()


def _quiet(fn, *a, **kw):
    with redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def _sphere(n=200000, seed=5):
    d = np.random.default_rng(seed).normal(size=(n, 3)).astype(f32)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cone(axis, half, n=4000, seed=9):
    """Unit directions within `half` radians of `axis`."""
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    t = np.cross(axis, [0.0, 0.0, 1.0] if abs(axis[2]) < 0.9 else [1.0, 0, 0])
    t /= np.linalg.norm(t)
    b = np.cross(axis, t)
    r = np.tan(half) * np.sqrt(rng.random(n))
    a = rng.random(n) * 2.0 * np.pi
    d = axis[None, :] + t[None, :] * (r * np.cos(a))[:, None] \
        + b[None, :] * (r * np.sin(a))[:, None]
    return (d / np.linalg.norm(d, axis=1, keepdims=True)).astype(f32)


def _lum(rgb):
    return np.asarray(rgb, f32)[:, :3] @ np.array([0.2126, 0.7152, 0.0722], f32)


def _angle(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.arccos(np.clip(a @ b / np.linalg.norm(a)
                                   / np.linalg.norm(b), -1.0, 1.0)))


def _brightest(sample, toward, half):
    """The brightest direction of a sky sampler over the whole sphere plus
    a dense cone around `toward` (so a small disc cannot be missed)."""
    d = np.concatenate([_sphere(), _cone(toward, half)])
    return d[int(np.argmax(_lum(sample(d))))]


def _sky_node_world(elev=0.5, rot=1.1, intensity=1.5, link_vector=False,
                    size=0.03):
    w = World()
    w.graph = {'output': 'w', 'nodes': {
        's': _wnode('s', 'ShaderNodeTexSky',
                    {'sky_type': 'PREETHAM', 'sun_elevation': elev,
                     'sun_rotation': rot, 'sun_intensity': intensity,
                     'turbidity': 2.5, 'sun_disc': True, 'sun_size': size},
                    [_sk('Vector', 'VECTOR', [0, 0, 0],
                         ['m', 0] if link_vector else None)],
                    [{'name': 'Color', 'type': 'RGBA'}]),
        'bg': _wnode('bg', 'ShaderNodeBackground', {},
                     [_sk('Color', 'RGBA', [0.05, 0.05, 0.05, 1.0], ['s', 0]),
                      _sk('Strength', 'VALUE', 1.0)],
                     [{'name': 'Background', 'type': 'SHADER'}]),
        'w': _wnode('w', 'ShaderNodeOutputWorld', {},
                    [_sk('Surface', 'SHADER', None, ['bg', 0])], [])}}
    return w


def _hdri_world(px, **kw):
    """(world, textures) for an image world over `px`."""
    w = World(mode='HDRI')
    buf = ImageBuffer(name='wl', pixels=px, colorspace='Linear')
    w.env_image = buf
    w.env_filter = 'NEAREST'
    for k, v in kw.items():
        setattr(w, k, v)
    return w, {'wl': Texture(px, name='wl', colorspace='Linear')}


def _cube_strip(S=32, face=2, at=(20, 9), hot=30.0):
    """A 6:1 strip of S x S faces: flat blue, one 3x3 hot block on `face`."""
    px = np.zeros((S, 6 * S, 4), f32)
    px[:, :, :3] = (0.2, 0.3, 0.5)
    px[:, :, 3] = 1.0
    x, y = face * S + at[0], at[1]
    px[y - 1:y + 2, x - 1:x + 2, :3] = np.array([1.0, 0.8, 0.6], f32) * hot
    return px


def _floor_pixel(sc, p):
    """(row, col) of world point `p` on this camera's frame."""
    _v, _p, vp, _eye = R.camera_matrices(sc.camera, W, H)
    clip = np.append(np.asarray(p, f32), 1.0).astype(f32) @ vp.T
    ndc = clip[:3] / clip[3]
    return (int((ndc[1] * 0.5 + 0.5) * H), int((ndc[0] * 0.5 + 0.5) * W))


def _twin(sc, st):
    """(cpu, sim, covered, why): the deferred pass through the GLSL
    simulator against the CPU frame (the R251 lamps module's rig). The
    CPU render runs first: it is render() that syncs the world light and
    builds its shadow map, exactly as the GPU road finds them."""
    cpu = _render(sc, st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     W, H)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    if p is None:
        return cpu, None, None, str(why)
    out, hit = GSH.simulate(job, g)
    if out is None:
        return cpu, None, None, str(hit)
    return cpu, out, g.tri >= 0, None


def _frame_bar(label, cpu, sim, mask, why, bar=6e-3):
    check(f'{label}: the deferred pass plans and simulates', sim is not None,
          str(why))
    if sim is None:
        return
    yy, xx = np.nonzero(mask)
    d = np.abs(sim[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
    check(f'{label}: the simulator matches the CPU within the deferred bar',
          int((d > 1e-2).sum()) == 0 and float(d.max()) < bar,
          f'{int((d > 1e-2).sum())} of {yy.size} px >0.01, '
          f'max {float(d.max()):.6f}')


def _old_select_lights(lights, settings, centre=None):
    """lights.select_lights as it stood before R254, verbatim."""
    active = [l for l in lights if l.type != 'AMBIENT']
    limit = int(settings.max_lights)
    if limit <= 0 or len(active) <= limit:
        return active
    if settings.light_limit_mode == 'FIRST':
        return active[:limit]
    if settings.light_limit_mode == 'NEAREST' and centre is not None:
        key = [float(np.linalg.norm(np.asarray(l.position, f32) - centre))
               for l in active]
        order = np.argsort(key)
    else:
        key = [float(l.energy) * float(np.mean(l.color)) for l in active]
        order = np.argsort(key)[::-1]
    return [active[i] for i in order[:limit]]


def _many_lamps(n=11, seed=2):
    rng = np.random.default_rng(seed)
    return [Light(type='POINT', name=f'P{i}',
                  position=tuple(rng.normal(size=3) * 4.0),
                  color=tuple(rng.random(3)), energy=float(100 + 900 * rng.random()),
                  shadow='NONE') for i in range(n)] + \
        [Light(type='AMBIENT', name='Amb', energy=0.1)]


# ============================================================== neutrality


def test_a_neutrality():
    """Off by default and inert when off: the scene is not touched, the
    frame is bitwise the same whatever the other dials hold, and
    select_lights is the pre-R254 function for every unmarked list."""
    w = World()
    check('a World() carries the eight fields at their neutral defaults',
          w.world_light is False and w.world_light_strength == 1.0
          and w.world_light_shadow == 'MAP' and w.world_light_shafts == 0.0
          and w.world_light_find == 'DETECT'
          and w.world_light_color == (1.0, 1.0, 1.0)
          and sorted(f.name for f in dataclasses.fields(World)
                     if f.name.startswith('world_light')) == sorted(FIELDS))
    st = _settings()
    sc = demo_scene(st, with_texture=False)
    held = sc.lights
    members = list(held)
    check('sync on a scene without the toggle returns None and leaves '
          'scene.lights the same list of the same lamps',
          WL.sync(sc) is None and sc.lights is held
          and all(a is b for a, b in zip(sc.lights, members))
          and len(sc.lights) == len(members))
    ref = _render(sc, st)
    check('...and render() leaves it so too',
          sc.lights is held and len(held) == len(members)
          and not any(WL.is_world_light(l) for l in held))
    loud = {'world_light_strength': 7.0, 'world_light_shadow': 'RAY',
            'world_light_shafts': 2.0, 'world_light_find': 'MANUAL',
            'world_light_elevation': -0.4, 'world_light_azimuth': 2.0,
            'world_light_color': (1.0, 0.1, 0.1)}
    px = FM.world_light_sky()
    buf = ImageBuffer(name='wl_sky', pixels=px, colorspace='Linear')
    for mode in ('NODES', 'BRYCE', 'PHYSICAL', 'HDRI', 'GRADIENT'):
        frames = []
        for dials in ({}, loud):
            st_ = _settings()
            sc_ = demo_scene(st_, with_texture=False)
            sc_.world.mode = mode
            if mode == 'HDRI':
                sc_.images['wl_sky'] = buf
                sc_.world.env_image = buf
            for k, v in dials.items():
                setattr(sc_.world, k, v)
            frames.append(_render(sc_, st_))
        check(f'a {mode} frame is bitwise the same with every World Light '
              'dial changed and the toggle off',
              np.array_equal(frames[0], frames[1]),
              f'max {float(np.abs(frames[0] - frames[1]).max())}')
    # the toggle on a world with no luminary: said once, nothing changes
    st_g = _settings()
    sc_g = demo_scene(st_g, with_texture=False)
    sc_g.world.mode = 'GRADIENT'
    a = _render(sc_g, st_g)
    sc_g.world.world_light = True
    WL._PRINTED.clear()
    out = io.StringIO()
    with redirect_stdout(out):
        b = np.asarray(R.render(sc_g, st_g)).copy()
        np.asarray(R.render(sc_g, st_g))
    check('World Light on a GRADIENT world adds no light, changes no pixel '
          'and says why once in the console',
          np.array_equal(a, b) and len(sc_g.lights) == 2
          and out.getvalue().count('[Halcyon] world light:') == 1
          and 'GRADIENT world has no sun or moon' in out.getvalue())
    for mode in ('SOLID', 'GRADIENT', 'BANDS', 'STARFIELD', 'PAINTED',
                 'CYLINDER', 'LW_GRADIENT'):
        lum, why = WL.luminary(World(mode=mode, world_light=True))
        check(f'{mode} offers no luminary, by name',
              lum is None and mode in str(why), str(why))
    check('MODES_WITH_LUMINARY and the rest partition sky.MODES',
          set(WL.MODES_WITH_LUMINARY) | {'SOLID', 'GRADIENT', 'BANDS',
                                         'STARFIELD', 'PAINTED', 'CYLINDER',
                                         'LW_GRADIENT'} == set(SK.MODES))
    # the light limit: the old function for every list without the mark
    same = True
    for n in (3, 8, 9, 11):
        for limit in (0, 1, 4, 8):
            for mode in ('BRIGHTEST', 'FIRST', 'NEAREST'):
                lamps = _many_lamps(n)
                st_l = _settings(max_lights=limit, light_limit_mode=mode)
                for centre in (None, np.zeros(3, f32)):
                    got = LI.select_lights(lamps, st_l, centre)
                    want = _old_select_lights(lamps, st_l, centre)
                    same = same and len(got) == len(want) and \
                        all(x is y for x, y in zip(got, want))
    check('select_lights returns the pre-R254 selection, object for '
          'object, for every lamp list without a world light', same)
    check('the demo frame is unchanged by all of the above', np.array_equal(
        ref, _render(demo_scene(_settings(), with_texture=False),
                     _settings())))


# ============================================================== luminaries


def test_b_bryce_and_physical():
    """The light comes from where the picture shows the body."""
    for mode in ('BRYCE', 'PHYSICAL'):
        for rot in (0.0, 0.9, -2.2):
            w = World(mode=mode, clouds=False, sun_elevation=0.5,
                      sun_rotation=1.1, rotation=rot, world_light=True)
            lum, why = WL.luminary(w)
            top = _brightest(lambda d: SK.evaluate(w, d), lum['dir'],
                             3.0 * w.sun_size)
            check(f'{mode} under World rotation {rot}: the luminary lies '
                  "inside the sun disc the sky draws (the picture's "
                  'brightest point)',
                  lum is not None and _angle(top, lum['dir']) <= w.sun_size,
                  f'{np.degrees(_angle(top, lum["dir"])):.3f} deg')
            lt = WL.make_light(w)
            check(f'...and the Light is a SUN aimed FROM it: direction = '
                  '-luminary, energy pi x intensity, radius 2 x Sun Size, '
                  'the sun colour',
                  lt.type == 'SUN' and WL.is_world_light(lt)
                  and lt.name == WL.NAME
                  and np.allclose(lt.direction, -lum['dir'], atol=1e-6)
                  and abs(lt.energy - np.pi * w.sun_intensity) < 1e-6
                  and abs(lt.radius - 2.0 * w.sun_size) < 1e-9
                  and np.allclose(lt.color, w.sun_color))
    w = World(mode='BRYCE', sun_elevation=0.5, sun_intensity=2.0,
              strength=1.5, world_light=True, world_light_strength=0.5)
    check('energy = pi x Sun Intensity x World Strength x World Light '
          'Strength', abs(WL.make_light(w).energy
                          - np.pi * 2.0 * 1.5 * 0.5) < 1e-6)
    check('a facing white surface under intensity 1 reads its own colour: '
          'UNIT_ENERGY is pi', WL.UNIT_ENERGY == float(np.pi))
    # the horizon
    size = 0.03
    es = (-0.2, -size, -size / 2, 0.0, size / 2, size, 0.4)
    en = []
    for e in es:
        lt = _quiet(WL.make_light, World(mode='BRYCE', sun_elevation=e,
                                         sun_size=size, world_light=True))
        en.append(0.0 if lt is None else lt.energy)
    check('the light fades out as the disc sets: none with the disc under '
          'the horizon, half with its centre on it, full with it clear',
          en[0] == 0.0 and en[1] == 0.0 and abs(en[3] - np.pi / 2) < 1e-5
          and abs(en[5] - np.pi) < 1e-5 and abs(en[6] - np.pi) < 1e-5
          and all(b >= a for a, b in zip(en, en[1:])), str(np.round(en, 4)))
    check('zero Sun Intensity, zero World Strength or zero World Light '
          'Strength make no light',
          _quiet(WL.make_light, World(mode='BRYCE', sun_intensity=0.0)) is None
          and _quiet(WL.make_light, World(mode='BRYCE', strength=0.0)) is None
          and _quiet(WL.make_light, World(
              mode='BRYCE', world_light_strength=0.0)) is None)
    wide = WL.make_light(World(mode='BRYCE', sun_size=0.4))
    thin = WL.make_light(World(mode='BRYCE', sun_size=0.005))
    check('the shadow softness follows the disc up to MAX_ANGLE (0.06, the '
          'default Bryce disc) and no further: a cinema sun still casts a '
          'map shadow', wide.radius == WL.MAX_ANGLE == 0.06
          and abs(thin.radius - 0.01) < 1e-9)
    # the dials reach the Light
    lt = WL.make_light(World(mode='PHYSICAL', world_light_shadow='RAY',
                             world_light_shafts=0.8))
    check('Shadow and Light Shafts land on the Light (shadow, volumetric)',
          lt.shadow == 'RAY' and lt.volumetric == f32(0.8)
          or (lt.shadow == 'RAY' and abs(lt.volumetric - 0.8) < 1e-9))
    check('the default Shadow is MAP and the default Shafts 0',
          WL.make_light(World(mode='BRYCE')).shadow == 'MAP'
          and WL.make_light(World(mode='BRYCE')).volumetric == 0.0
          and WL.make_light(World(mode='BRYCE',
                                  world_light_shadow='NONE')).shadow == 'NONE')
    check('the panel line names the body and where it stands',
          'sun' in WL.LAST_REPORT['text'] and 'elevation' in WL.LAST_REPORT['text'])


def test_c_moon():
    """Bryce's one body, swapped: same direction, the moon's colour, a
    quarter of the sun's strength scaled by the phase the disc shows."""
    base = dict(mode='BRYCE', clouds=False, sun_elevation=0.6,
                sun_rotation=0.4, rotation=0.7, world_light=True)
    sun = WL.make_light(World(**base))
    full = WL.make_light(World(celestial='MOON', moon_phase=0.5,
                               moon_earthshine=0.0, **base))
    check('the moon lights from the direction the sun would',
          np.allclose(sun.direction, full.direction))
    wm = World(celestial='MOON', moon_phase=0.5, **base)
    lum, _w = WL.luminary(wm)
    inside = SK.evaluate(wm, _cone(lum['dir'], wm.moon_size * 0.5, 64))
    wm2 = World(celestial='MOON', moon_phase=0.5, **base)
    wm2.moon_size = 1e-4
    gone = SK.evaluate(wm2, _cone(lum['dir'], wm.moon_size * 0.5, 64))
    check('...which is where the sky draws the moon disc (shrinking Moon '
          'Size to nothing darkens exactly those directions)',
          float((_lum(inside) - _lum(gone)).min()) > 0.1)
    w0 = World(**base)
    small = WL.make_light(World(celestial='MOON', moon_phase=0.5,
                                moon_size=0.02, **base))
    check('a full moon with no earthshine is MOON_LIGHT (0.25) of the sun, '
          'in the moon colour, softened by 2 x Moon Size up to MAX_ANGLE',
          abs(full.energy - sun.energy * 0.25) < 1e-6
          and WL.MOON_LIGHT == 0.25
          and np.allclose(full.color, w0.moon_color)
          and 2.0 * w0.moon_size > WL.MAX_ANGLE
          and full.radius == WL.MAX_ANGLE
          and abs(small.radius - 0.04) < 1e-9)
    en = []
    for ph in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
        lt = _quiet(WL.make_light, World(celestial='MOON', moon_phase=ph,
                                         moon_earthshine=0.0, **base))
        en.append(0.0 if lt is None else lt.energy / sun.energy / 0.25)
    check('the phase scales it by the lit fraction: new 0, quarter 1/2, '
          'full 1, waning quarter 1/2, new 0',
          abs(en[0]) < 1e-6 and abs(en[2] - 0.5) < 1e-6
          and abs(en[3] - 1.0) < 1e-6 and abs(en[4] - 0.5) < 1e-6
          and abs(en[5]) < 1e-6 and 0.0 < en[1] < 0.5, str(np.round(en, 4)))
    shine = WL.make_light(World(celestial='MOON', moon_phase=0.0,
                                moon_earthshine=0.2, **base))
    check('earthshine is the new moon\'s floor (sky._moon\'s own lit * '
          '(1 - shine) + shine)',
          abs(shine.energy - sun.energy * 0.25 * 0.2) < 1e-6
          and abs(WL.moon_lit_fraction(0.25, 0.2) - (0.5 * 0.8 + 0.2)) < 1e-12)
    check('PHYSICAL has no moon: celestial MOON still gives the sun there',
          np.allclose(WL.make_light(World(
              mode='PHYSICAL', celestial='MOON')).color, World().sun_color))


def test_d_sky_texture_node():
    w = _sky_node_world(elev=0.5, rot=1.1, intensity=1.5)
    w.world_light = True
    lum, why = WL.luminary(w)
    st = _settings()
    sc = demo_scene(st, with_texture=False)
    sc.world = w
    top = _brightest(lambda d: R.world_color(sc, st, d, {}), lum['dir'], 0.09)
    check("NODES: the luminary lies inside the Sky Texture node's own sun "
          'disc as the world graph draws it',
          lum is not None and _angle(top, lum['dir']) <= 0.03,
          f'{np.degrees(_angle(top, lum["dir"])):.3f} deg')
    lt = WL.make_light(w)
    check("...energy pi x the node's Sun Intensity, the node's disc colour, "
          "radius 2 x its Sun Size",
          abs(lt.energy - np.pi * 1.5) < 1e-5
          and tuple(lt.color) == WL.NODE_SUN_COLOR
          and abs(lt.radius - 0.06) < 1e-9)
    w.rotation = 1.0
    check('the node graph does not read the World rotation, and neither '
          'does its sun', np.allclose(WL.make_light(w).direction, lt.direction))
    lum2, why2 = WL.luminary(_sky_node_world(link_vector=True))
    check('a Sky Texture mapped through its Vector input is refused by name',
          lum2 is None and 'Vector' in str(why2), str(why2))
    lum3, why3 = WL.luminary(World(mode='NODES'))
    wn = World(mode='NODES')
    wn.graph = {'output': 'w', 'nodes': {'w': _wnode(
        'w', 'ShaderNodeOutputWorld', {}, [], [])}}
    lum4, why4 = WL.luminary(wn)
    check('a node world without a Sky Texture offers none, by name',
          lum3 is None and lum4 is None and 'Sky Texture' in str(why4))
    low = _quiet(WL.make_light, _sky_node_world(elev=-0.3))
    check("the node's sun under the horizon makes no light", low is None)


# ============================================================ image search


def _reads_hot(world, tex, toward, hot_floor, spread):
    """The sky reads the hot texels AT `toward` and the plain sky
    `spread` radians away on every side."""
    at = _lum(SK.evaluate(world, np.asarray(toward, f32).reshape(1, 3), tex))
    ring = _cone(toward, spread, 400)
    ring = ring[[_angle(r, toward) > spread * 0.8 for r in ring]]
    around = _lum(SK.evaluate(world, ring, tex))
    return float(at[0]) > hot_floor and float(around.max()) < hot_floor * 0.2, \
        f'at {float(at[0]):.2f}, ring max {float(around.max()):.2f}'


def test_e_image_sun_hdri():
    px = FM.world_light_sky()
    for mapping in ('EQUIRECT', 'MIRRORBALL'):
        for filt in ('NEAREST', 'BILINEAR'):
            for rot in (0.0, 1.3):
                WL._FIND_CACHE.clear()
                w, tex = _hdri_world(px, env_mapping=mapping, env_filter=filt,
                                     rotation=rot, world_light=True)
                lum, why = WL.luminary(w, tex)
                ok, said = (False, str(why)) if lum is None else \
                    _reads_hot(w, tex, lum['dir'], 10.0, 0.25)
                check(f'HDRI {mapping} / {filt} / rotation {rot}: the sky '
                      'reads the hot texels along the found direction and '
                      'plain sky all round it', ok, said)
    WL._FIND_CACHE.clear()
    w, tex = _hdri_world(px, world_light=True, strength=2.0,
                         world_light_strength=1.5)
    lt = WL.make_light(w, tex)
    check('the Light: energy pi x World Strength x World Light Strength, '
          'the found colour normalised to its largest channel',
          abs(lt.energy - np.pi * 2.0 * 1.5) < 1e-5
          and np.allclose(lt.color, (1.0, 0.9, 0.7), atol=0.02)
          and 0.0 < lt.radius <= WL.MAX_ANGLE, f'{lt.color} r {lt.radius:.4f}')
    check('the panel line says the sun was found in the image',
          'found in the image' in WL.LAST_REPORT['text'])
    w.env_tint = (0.5, 1.0, 1.0)
    check('Env Tint tints the found colour as it tints the picture',
          WL.make_light(w, tex).color[0] < 0.6 * max(WL.make_light(w, tex).color))
    # textures may arrive as a callable (render entry prepares them lazily)
    calls = []

    def lazy():
        calls.append(1)
        return tex
    w.env_tint = (1.0, 1.0, 1.0)
    check('textures may be a callable, called for an image world only',
          WL.make_light(w, lazy) is not None and len(calls) == 1
          and WL.make_light(World(mode='BRYCE'), lazy) is not None
          and len(calls) == 1)
    # the cache
    WL._FIND_CACHE.clear()
    r1 = WL.find_image_sun(w, tex)
    r2 = WL.find_image_sun(w, tex)
    px2 = FM.world_light_sky(sun_px=(200, 80))
    w2, tex2 = _hdri_world(px2, world_light=True)
    r3 = WL.find_image_sun(w2, tex2)
    check('the search runs once per image (a second call is the cached '
          'object) and again for another image',
          r1 is r2 and r3 is not r1
          and _angle(r3[0]['dir'], r1[0]['dir']) > 0.5
          and len(WL._FIND_CACHE) <= WL.FIND_CACHE_MAX)
    # an 8-bit sky: one clipped blob is found...
    ldr = np.zeros((128, 256, 4), f32)
    ldr[:, :, :3] = 0.5
    ldr[:, :, 3] = 1.0
    one = ldr.copy()
    one[90:96, 60:66, :3] = 1.0
    wl, tl = _hdri_world(one, world_light=True)
    lum, why = WL.luminary(wl, tl)
    ok, said = (False, str(why)) if lum is None else \
        _reads_hot(wl, tl, lum['dir'], 0.9, 0.3)
    check('an 8-bit sky with one clipped sun is found', ok or (
        lum is not None and float(_lum(SK.evaluate(
            wl, lum['dir'].reshape(1, 3), tl))[0]) > 0.9), said)
    # ...two equally bright regions are refused, and so is a flat sky
    two = one.copy()
    two[90:96, 180:186, :3] = 1.0
    wt, tt = _hdri_world(two, world_light=True)
    lum_t, why_t = WL.luminary(wt, tt)
    check('two equally bright regions are refused by name and point at '
          'Manual', lum_t is None and 'more than one' in str(why_t)
          and 'Manual' in str(why_t), str(why_t))
    wf, tf = _hdri_world(ldr, world_light=True)
    lum_f, why_f = WL.luminary(wf, tf)
    check('a sky with nothing standing out is refused by name',
          lum_f is None and 'stands out' in str(why_f), str(why_f))
    check('an image world without its image offers none',
          WL.luminary(World(mode='HDRI', world_light=True), {})[0] is None
          and WL.luminary(World(mode='CUBEMAP', world_light=True), {})[0] is None)
    # Manual: nothing is read from the image
    wm, tm = _hdri_world(two, world_light=True, world_light_find='MANUAL',
                         world_light_elevation=0.3, world_light_azimuth=-1.0,
                         world_light_color=(0.9, 0.5, 0.2), rotation=0.8)
    ltm = WL.make_light(wm, None)
    want = SK._rotate_z(SK._sun_vector(0.3, -1.0).reshape(1, 3), 0.8)[0]
    check('Manual: the direction is the Bryce sun rule on Elevation / '
          'Azimuth turned by the World rotation, the colour as typed, no '
          'image needed',
          np.allclose(ltm.direction, -want, atol=1e-6)
          and np.allclose(ltm.color, (0.9, 0.5, 0.2))
          and abs(ltm.energy - np.pi) < 1e-6
          and ltm.radius == WL.MANUAL_ANGLE)


def test_f_image_sun_cubemap():
    for conv in ('OPENGL', 'QUAKE2'):
        for face in range(6):
            for filt, rot in (('NEAREST', 0.0), ('BILINEAR', 2.0)):
                WL._FIND_CACHE.clear()
                px = _cube_strip(face=face)
                w = World(mode='CUBEMAP', cube_convention=conv,
                          cube_filter=filt, rotation=rot, world_light=True,
                          cube_face_rot=(1, 0, 2, 3, 0, 1),
                          cube_face_flip=(False, True, False, False, True,
                                          False))
                w.env_image = ImageBuffer(name='wl', pixels=px,
                                          colorspace='Linear')
                tex = {'wl': Texture(px, name='wl', colorspace='Linear')}
                lum, why = _quiet(WL.luminary, w, tex)
                ok, said = (False, str(why)) if lum is None else \
                    _reads_hot(w, tex, lum['dir'], 8.0, 0.3)
                check(f'CUBEMAP {conv} face {face} / {filt} / rotation '
                      f'{rot} with turned and mirrored faces: the sky reads '
                      'the hot texels along the found direction', ok, said)
    # the six-image road
    WL._FIND_CACHE.clear()
    S = 32
    w = World(mode='CUBEMAP', cube_source='SIX', world_light=True)
    tex = {}
    for i, slot in enumerate(('px', 'nx', 'py', 'ny', 'pz', 'nz')):
        pic = np.zeros((S, S, 4), f32)
        pic[:, :, :3] = (0.2, 0.3, 0.5)
        pic[:, :, 3] = 1.0
        if slot == 'nz':
            pic[4:7, 9:12, :3] = 25.0
        buf = ImageBuffer(name=f'wl_{slot}', pixels=pic, colorspace='Linear')
        setattr(w, f'cube_image_{slot}', buf)
        tex[buf.name] = Texture(pic, name=buf.name, colorspace='Linear')
    lum, why = WL.luminary(w, tex)
    ok, said = (False, str(why)) if lum is None else \
        _reads_hot(w, tex, lum['dir'], 8.0, 0.3)
    check('CUBEMAP from six images: found, and the sky reads it there',
          ok, said)


# =============================================================== the frame


def test_g_frame_lit_and_shadowed():
    """A lamp-less scene is lit and shadowed from the luminary's side on
    every world type that has one, and the frame is bitwise the frame of
    the same Sun lamp made by hand."""
    st = _settings()
    px = FM.world_light_sky()
    buf = ImageBuffer(name='wl_sky', pixels=px, colorspace='Linear')
    strip = _cube_strip(face=0, at=(16, 22))
    cbuf = ImageBuffer(name='wl_cube', pixels=strip, colorspace='Linear')
    rigs = (
        ('the Bryce sun', 'BRYCE', {}),
        ('the Bryce moon', 'BRYCE', {'celestial': 'MOON', 'moon_phase': 0.5,
                                     'world_light_strength': 3.0}),
        ('the Physical sky', 'PHYSICAL', {}),
        ('a Sky Texture node', 'NODES', {}),
        ('an HDRI sun', 'HDRI', {'env_image': buf, 'env_filter': 'NEAREST'}),
        ('a cube-map sun', 'CUBEMAP', {'env_image': cbuf}),
        ('a Manual HDRI direction', 'HDRI', {
            'env_image': buf, 'world_light_find': 'MANUAL',
            'world_light_elevation': SUN_EL, 'world_light_azimuth': SUN_AZ}),
    )
    for label, mode, dials in rigs:
        WL._FIND_CACHE.clear()
        sc = _nolamp(st, mode, **dials)
        if mode == 'NODES':
            sc.world = _sky_node_world(elev=SUN_EL, rot=SUN_AZ)
            sc.world.world_light = True
        for b in (buf, cbuf):
            sc.images[b.name] = b
        if mode in WL.IMAGE_MODES and 'world_light_find' not in dials:
            # turn the World so the found sun stands at SUN_AZ: the
            # shadow then falls on the open floor, and the frame proves
            # the rotation reaches the light
            found, _why = WL.luminary(sc.world, R.prepare_textures(sc, st))
            sc.world.rotation = SUN_AZ - float(np.arctan2(found['dir'][1],
                                                          found['dir'][0]))
        on = _render(sc, st)
        lt = [l for l in sc.lights if WL.is_world_light(l)]
        check(f'{label}: render() put exactly one world light in '
              'scene.lights', len(lt) == 1 and len(sc.lights) == 1)
        if not lt:
            continue
        toward = -np.asarray(lt[0].direction, f32)
        sc.world.world_light_shadow = 'NONE'
        flat = _render(sc, st)
        sc.world.world_light = False
        off = _render(sc, st)
        check(f'{label}: switching it off removes the light again',
              len(sc.lights) == 0)
        g = CRg.GBuffer(W, H)
        CRg.rasterize(sc.mesh.verts, sc.mesh.tris,
                      R.camera_matrices(sc.camera, W, H)[2], W, H, gbuf=g)
        cov = g.tri >= 0
        check(f'{label}: the lamp-less scene is lit (covered pixels '
              'brighter than with the toggle off)',
              float(on[cov][:, :3].mean()) > float(off[cov][:, :3].mean())
              + 0.02, f'{float(on[cov][:, :3].mean()):.4f} vs '
                      f'{float(off[cov][:, :3].mean()):.4f}')
        check(f'{label}: the luminary stands above the horizon at the '
              "rig's azimuth", toward[2] > 0.15 and abs(np.arctan2(
                  toward[1], toward[0]) - SUN_AZ) < 0.05,
              str(np.round(toward, 3)))
        if toward[2] > 0.15:
            # the ball's shadow lands on the floor directly down-light of it
            p = BALL - toward * (BALL[2] / toward[2])
            r, c = _floor_pixel(sc, p)
            inside = 0 <= r < H and 0 <= c < W and g.tri[r, c] >= 0 and \
                int(sc.mesh.mat_index[g.tri[r, c]]) == 0
            if inside:
                check(f'{label}: the floor down-light of the ball is in '
                      'shadow (darker than the same light with Shadow None)',
                      float(flat[r, c, :3].mean())
                      - float(on[r, c, :3].mean()) > 0.03,
                      f'{float(on[r, c, :3].mean()):.4f} vs '
                      f'{float(flat[r, c, :3].mean()):.4f} at {r},{c}')
            else:
                check(f'{label}: the shadow point is on the visible floor',
                      False, f'{r},{c}')
        # a hand-made Sun lamp with the same numbers: the same frame
        sc.world.world_light_shadow = 'MAP'
        sc.lights = [Light(type='SUN', name=WL.NAME,
                           direction=tuple(lt[0].direction),
                           color=tuple(lt[0].color), energy=lt[0].energy,
                           radius=lt[0].radius, shadow='MAP')]
        hand = _render(sc, st)
        check(f'{label}: bitwise the frame of the same Sun lamp made by '
              'hand (past sync it IS a lamp)', np.array_equal(on, hand),
              f'max {float(np.abs(on - hand).max())}')
    # ray shadows
    st_r = _settings(shadow_default='PER_LIGHT')
    sc = _nolamp(st_r, 'BRYCE', world_light_shadow='RAY')
    ray = _render(sc, st_r)
    sc.world.world_light_shadow = 'NONE'
    none = _render(sc, st_r)
    toward = SK._sun_vector(SUN_EL, SUN_AZ)
    r, c = _floor_pixel(sc, BALL - toward * (BALL[2] / toward[2]))
    check('Shadow Ray Traced shadows the same floor point',
          float(none[r, c, :3].mean()) - float(ray[r, c, :3].mean()) > 0.03)
    # the sun moved to the other side moves the shadow away from it
    sc = _nolamp(st, 'BRYCE')
    a = _render(sc, st)
    sc.world.sun_rotation = SUN_AZ - np.pi
    b = _render(sc, st)
    check('a world-only edit re-lights: turning the sun half round lights '
          'the floor point that was in shadow',
          float(b[r, c, :3].mean()) - float(a[r, c, :3].mean()) > 0.03
          and len(sc.lights) == 1
          and np.allclose(sc.lights[0].direction,
                          -SK._sun_vector(SUN_EL, SUN_AZ - np.pi), atol=1e-6))


def test_h_behaves_like_a_lamp():
    st = _settings()
    # beside real lamps: appended last, the lamps untouched
    sc = demo_scene(st, with_texture=False)
    lamps = list(sc.lights)
    sc.world.mode = 'BRYCE'
    sc.world.world_light = True
    both = _render(sc, st)
    check('beside real lamps the world light is appended LAST and the '
          'lamps keep their places',
          len(sc.lights) == len(lamps) + 1
          and all(a is b for a, b in zip(sc.lights, lamps))
          and WL.is_world_light(sc.lights[-1]))
    _render(sc, st)
    _render(sc, st)
    check('render after render keeps exactly one (sync is idempotent)',
          sum(WL.is_world_light(l) for l in sc.lights) == 1
          and len(sc.lights) == len(lamps) + 1)
    sc.world.world_light = False
    alone = _render(sc, st)
    check('...and it adds its light to theirs (the frame is brighter)',
          float(both[..., :3].mean()) > float(alone[..., :3].mean()) + 0.01)
    # a lamp-only re-export (export_lights_into) rebuilds scene.lights from
    # the lamps alone: the next render re-lights
    sc.world.world_light = True
    _render(sc, st)
    nxt = copy.copy(sc)
    nxt.lights = list(lamps)
    again = _render(nxt, st)
    check('a scene whose lights were rebuilt from the lamps alone is '
          're-lit at the next render, to the same frame',
          sum(WL.is_world_light(l) for l in nxt.lights) == 1
          and np.array_equal(again, both))
    # the light limit never drops it
    for mode in ('BRIGHTEST', 'FIRST', 'NEAREST'):
        lamps11 = _many_lamps(11)
        wl = WL.make_light(World(mode='BRYCE', world_light_strength=0.01))
        st_l = _settings(max_lights=8, light_limit_mode=mode)
        for centre in (None, np.zeros(3, f32)):
            sel = LI.select_lights(lamps11 + [wl], st_l, centre)
            rest = _old_select_lights(
                lamps11, _settings(max_lights=7, light_limit_mode=mode), centre)
            check(f'light limit 8 / {mode}'
                  + ('' if centre is None else ' with a centre')
                  + ': the world light keeps a slot however weak, and the '
                  'lamps are ranked for the seven left',
                  len(sel) == 8 and sel[-1] is wl
                  and all(a is b for a, b in zip(sel, rest)))
    check('light limit 1 keeps the world light alone',
          LI.select_lights(_many_lamps(4) + [wl],
                           _settings(max_lights=1)) == [wl])
    check('GM.MAX_LIGHTS is still the 8 the limit defaults to',
          GM.MAX_LIGHTS == 8 and _settings().max_lights == 8)
    # shafts
    sc = _nolamp(st, 'BRYCE', sun_elevation=0.25, sun_rotation=2.2)
    _render(sc, st)
    _v, _p, vp, _e = R.camera_matrices(sc.camera, W, H)
    check('Light Shafts 0: the world light is no shaft source',
          R.shaft_sources(sc, st, vp) == [])
    sc.world.world_light_shafts = 0.7
    _render(sc, st)
    src = R.shaft_sources(sc, st, vp)
    toward = SK._sun_vector(0.25, 2.2)
    clip = np.append(toward, 0.0).astype(f32) @ vp.T
    check('Light Shafts 0.7: one source of that strength at the sun\'s own '
          'vanishing point on screen',
          len(src) == 1 and abs(src[0][1] - 0.7) < 1e-6
          and clip[3] > 0 and np.allclose(src[0][0], clip[:2] / clip[3],
                                          atol=1e-5), str(src))
    # light styles and the ambient pool see an ordinary lamp
    sc = _nolamp(st, 'BRYCE')
    _render(sc, st)
    check('apply_light_styles gave it the NORMAL gain and the ambient pool '
          'does not count it',
          float(getattr(sc.lights[0], 'style_gain', 1.0)) == 1.0
          and np.array_equal(LI.ambient_light(sc, st), LI.ambient_light(
              demo_scene(st, with_texture=False), st) * 0
              + LI.ambient_light(_off(sc), st)))


def _off(sc):
    c = copy.copy(sc)
    c.lights = []
    return c


# ================================================================ GPU twin


def test_i_gpu_twin():
    st = _settings()
    px = FM.world_light_sky()
    buf = ImageBuffer(name='wl_sky', pixels=px, colorspace='Linear')
    strip = _cube_strip(face=2)
    cbuf = ImageBuffer(name='wl_cube', pixels=strip, colorspace='Linear')
    for label, mode, dials in (
            ('the Bryce sun', 'BRYCE', {}),
            ('the Bryce moon', 'BRYCE', {'celestial': 'MOON',
                                         'moon_phase': 0.4}),
            ('the Physical sky', 'PHYSICAL', {}),
            ('an HDRI sun', 'HDRI', {'env_image': buf,
                                     'env_filter': 'NEAREST'}),
            ('a cube-map sun', 'CUBEMAP', {'env_image': cbuf})):
        WL._FIND_CACHE.clear()
        sc = _nolamp(st, mode, **dials)
        for b in (buf, cbuf):
            sc.images[b.name] = b
        cpu, sim, mask, why = _twin(sc, st)
        _frame_bar(f'GPU twin, {label} as the only light', cpu, sim, mask, why)
    # the texels: the world light packs as any SUN
    sc = _nolamp(st, 'BRYCE', sun_intensity=1.7)
    _render(sc, st)
    sel = LI.select_lights(sc.lights, st)
    job = R.ShadeJob(sc, st, {}, None,
                     *R.camera_matrices(sc.camera, W, H)[::3], W, H)
    texels = GM.pack_light_texels(sel, job)
    check('hal_lights carries it: the energy texel is np.float32(pi x 1.7)',
          len(sel) == 1 and float(texels[0, 0, 3]) == float(f32(sel[0].energy))
          and abs(sel[0].energy - np.pi * 1.7) < 1e-6,
          f'{float(texels[0, 0, 3])}')
    # structure: the toggle changes the plan's light signature
    sig_on = tuple(GSH._light_sig(l) for l in sc.lights)
    sc.world.world_light = False
    _render(sc, st)
    check('the plan signature follows the toggle (one light signature with '
          'it, none without)',
          len(sig_on) == 1
          and tuple(GSH._light_sig(l) for l in sc.lights) == ())
    # beside lamps, and after a world edit, parity holds
    sc = demo_scene(st, with_texture=False)
    sc.world.mode = 'PHYSICAL'
    sc.world.sun_elevation = SUN_EL
    sc.world.sun_rotation = SUN_AZ
    sc.world.world_light = True
    cpu, sim, mask, why = _twin(sc, st)
    _frame_bar('GPU twin, the world light beside the demo lamps', cpu, sim,
               mask, why)
    sc.world.sun_rotation = SUN_AZ - 1.2
    cpu2, sim2, mask2, why2 = _twin(sc, st)
    _frame_bar('GPU twin after a world-only edit', cpu2, sim2, mask2, why2)
    check('...and both frames moved with the sun',
          sim is not None and sim2 is not None
          and float(np.abs(sim2 - sim).max()) > 0.02
          and float(np.abs(cpu2 - cpu).max()) > 0.02)
    # the feature-matrix row, and the whole frame through the fake device
    sc_r, st_r = FM.build('world light (HDRI sun, no lamps)')
    st_r.transparency = 'NONE'
    st_r.show_stats = False
    cpu, sim, mask, why = _twin(sc_r, st_r)
    _frame_bar('the feature-matrix row', cpu, sim, mask, why)
    from .test_r251_sky_camera import fake_device_render
    sc_f, st_f = FM.build('world light (HDRI sun, no lamps)')
    st_f.transparency = 'NONE'
    st_f.show_stats = False
    cpu_f = _render(sc_f, st_f)
    got = _quiet(fake_device_render, sc_f, st_f)
    check('render() on the GPU device through the fake driver: the world '
          'light reached the pass (engaged, one light, the CPU frame within '
          'the deferred bar)',
          bool(R.LAST_GPU_VERDICT.get('engaged'))
          and sum(WL.is_world_light(l) for l in sc_f.lights) == 1
          and got.shape == cpu_f.shape
          and float(np.abs(got - cpu_f).max()) < 6e-3,
          f"{R.LAST_GPU_VERDICT.get('why')} max "
          f'{float(np.abs(got - cpu_f).max()) if got.shape == cpu_f.shape else -1}')


# ================================================================== wiring


def test_j_wiring():
    """fakebpy: every field mirrors into the settings group with a real
    tooltip, the presets leave it alone, the panel is registered and says
    so when a world has no luminary, render() syncs at both entries."""
    from . import fakebpy
    from .blender_icons import ICONS as VERIFIED_ICONS
    import re
    fakebpy.install()
    P = importlib.import_module('halcyon.properties')
    importlib.reload(P)
    ui = importlib.import_module('halcyon.ui')
    from ..presets import skies as SKP
    ann = P.HalcyonWorldSettings.__annotations__
    check('every world_light* World field has a HalcyonWorldSettings '
          'property (the generic export loop carries it)',
          all(f in ann for f in FIELDS), str([f for f in FIELDS if f not in ann]))
    short = [f for f in FIELDS
             if len(str(ann[f].kw.get('description', '') or '')) < 40]
    check('every one carries a >= 40-char tooltip', not short, str(short))
    w = World()
    bad = [f for f in FIELDS if ann[f].kw.get('default') != getattr(w, f)
           and tuple(ann[f].kw.get('default', ())
                     if isinstance(getattr(w, f), tuple) else ()) != getattr(w, f)]
    check('every property default is the World dataclass default (off, 1, '
          'MAP, 0, DETECT...)', not bad, str(bad))
    enum_short = [f'{f}.{i[0]}' for f in ('world_light_shadow',
                                          'world_light_find')
                  for i in ann[f].kw['items'] if len(i[2]) < 12]
    check('every enum item carries a description, and the items are the '
          'values the core reads',
          not enum_short
          and [i[0] for i in ann['world_light_shadow'].kw['items']]
          == ['NONE', 'MAP', 'RAY']
          and [i[0] for i in ann['world_light_find'].kw['items']]
          == ['DETECT', 'MANUAL'], str(enum_short))
    check('presets.skies.EXCLUDED holds every world_light* field: no sky '
          'preset or sky file carries or resets the toggle',
          all(f in SKP.EXCLUDED for f in FIELDS)
          and not any(f.startswith('world_light') for f in SKP.sky_fields()))
    wp = World(mode='BRYCE', world_light=True, world_light_strength=2.5)
    SKP.apply_sky(wp, 'BRYCE_DEFAULT')
    check('applying a sky preset leaves World Light as the user set it',
          wp.world_light is True and wp.world_light_strength == 2.5)
    pan = getattr(ui, 'HALCYON_PT_world_light', None)
    check('HALCYON_PT_world_light is registered under the World panel and '
          'polls for every world type',
          pan is not None and pan in ui.CLASSES
          and pan.bl_parent_id == 'HALCYON_PT_world'
          and 'mode' not in inspect.getsource(pan.poll))
    src = inspect.getsource(pan)
    check('the panel draws every dial, the toggle in its header',
          all(f"'{f}'" in src for f in FIELDS)
          and "'world_light', text=\"\"" in inspect.getsource(pan.draw_header))
    check('...says so when the world type has no luminary instead of '
          'showing a dead control, shows the last report and warns of a '
          'Sun lamp lighting the scene as well',
          'MODES_WITH_LUMINARY' in src and 'no sun or moon' in src
          and 'LAST_REPORT' in src and 'Sun lamps light it as well' in src)
    icons = re.findall(r"icon='(\w+)'", src)
    check("the panel's icons are on the vetted list",
          icons and all(i in VERIFIED_ICONS for i in icons), str(icons))
    rsrc = inspect.getsource(R)
    check('render() and the second shading entry both sync the world light '
          'before collect_exclusive_lights reads the lamps',
          rsrc.count('_WL.sync(scene, lambda: prepare_textures(scene, st))')
          == 2
          and rsrc.index('_WL.sync(scene')
          < rsrc.index('    collect_exclusive_lights(scene)\n    mesh = '))
    # the real export fast path: lamps only, then render re-lights
    export = importlib.import_module('halcyon.export')

    class _DG:
        object_instances = ()

        class scene:
            class unit_settings:
                scale_length = 1.0
    st = _settings()
    parked = _nolamp(st, 'BRYCE')
    _render(parked, st)
    nxt = export.export_lights_into(parked, _DG())
    check('export_lights_into hands back the lamps alone (no synthetic '
          'light is exported)...', nxt.lights == [] and nxt.world is parked.world)
    _render(nxt, st)
    check('...and the next render lights it from the world again',
          len(nxt.lights) == 1 and WL.is_world_light(nxt.lights[0]))
    check("the feature matrix has the row and its scene",
          ('world light (HDRI sun, no lamps)', {}, 'world_light') in FM.ROWS
          and 'world_light' in FM.SCENES)
    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    frag = os.path.join(root, 'docs-dev', 'r254', 'world-light.md')
    text = open(frag, encoding='utf-8').read() if os.path.exists(frag) else ''
    check('docs-dev/r254/world-light.md carries the three sections',
          all(h in text for h in ('## CHANGELOG', '## README',
                                  '## Blender-only')))


def main():
    utf8_console()
    FAILS.clear()
    names = sorted(n for n in globals() if n.startswith('test_'))
    for n in names:
        print(f'\n[{n}]')
        try:
            globals()[n]()
        except Exception:                                       # noqa: BLE001
            traceback.print_exc()
            FAILS.append(n + ' (exception)')
    print()
    print(f'{len(FAILS)} failure(s)' if FAILS
          else 'all R254 world light checks passed')
    for f in FAILS:
        print('  FAIL', f)
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
