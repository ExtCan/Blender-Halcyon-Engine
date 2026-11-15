"""R254: what the rendered viewport (and the next F12) shows follows what was
changed.

A light option that the renderer honours but that does not reach the rendered
viewport, or reaches it through a stale cache, looks exactly like an option
that does not work. Each test below fails on the tree before this round:

    (A) the update classifier     a 5.x light edit arrives with the light's own
                                  node tree and is a lights-only export; a
                                  scene-level dial is no export at all
    (B) viewport visibility       H, the eye on a collection, Local View: the
                                  hidden lamp or mesh leaves the viewport
                                  export; F12 keeps its own rule
    (C) update callbacks          every Halcyon setting has one, so a script or
                                  operator write is what a panel drag is
    (D) the lights-only export    reads the clock, stands down for a material
                                  that bakes a lamp, and a Light Group resolves
    (E) the F12 mesh cache        a newly built RENDER depsgraph lists every ID:
                                  objects the main depsgraph already reported
                                  are not dirtied by it
    (F) the Shoot On held cel     an edit of the scene retires it
    (G) the image pixel cache     a repainted or reloaded image is read again
    (H) the mesh key              every array the mesh uploads pack is digested
                                  exactly, memoised on identity
    (I) neutrality                a scene nobody edited exports and renders the
                                  same bits, and its keys still hit

    python -m halcyon.tests.test_r254_viewport
"""
import ast
import contextlib
import copy
import inspect
import io
import os
import sys
import tempfile
import traceback
import types

import numpy as np

from . import utf8_console
from . import fakebpy
from . import fakeblender as FB
from . import fakedevice
from .scenebuild import (_mesh_concat, cube, demo_scene, look_at_matrix, plane,
                         sphere)
from ..core import lights as LI
from ..core import render as R
from ..core.scene import (Camera, Light, Material, ObjectInfo, Scene, World)
from ..core.settings import RenderSettings
from ..gpu import shade as GSH

FAILS = []
NS = types.SimpleNamespace
W, H = 96, 72


def check(name, cond, extra=''):
    ok = bool(cond)
    print(f'  {"ok  " if ok else "FAIL"} {name}  {extra}' if extra
          else f'  {"ok  " if ok else "FAIL"} {name}')
    if not ok:
        FAILS.append(name)
    return ok


def note(text):
    print(f'  note {text}')


# ------------------------------------------------------------------ helpers


def _mods():
    """(properties, engine, export, compat, preview) under the fake bpy."""
    import importlib
    props, eng = FB.install()
    return (props, eng, importlib.import_module('halcyon.export'),
            importlib.import_module('halcyon.compat'),
            importlib.import_module('halcyon.preview'))


class _Up:
    """One depsgraph update."""

    def __init__(self, uid, geo=False, xform=False):
        self.id = uid
        self.is_updated_geometry = geo
        self.is_updated_transform = xform


def _id(type_name, **attrs):
    """A datablock stand-in whose class carries Blender's type name."""
    ob = type(type_name, (), {})()
    for k, v in attrs.items():
        setattr(ob, k, v)
    return ob


def _updates(*ups, mode='VIEWPORT'):
    return NS(updates=list(ups), mode=mode)


def _lamp_object(name='Lamp', uid=900, kind='SUN', energy=7.0):
    mw = np.eye(4, dtype=np.float32)
    mw[:3, 3] = (1.0, 2.0, 3.0)
    return NS(type='LIGHT', name=name, session_uid=uid,
              data=NS(type=kind, color=(1.0, 0.5, 0.25), energy=energy,
                      halcyon=None, shadow_soft_size=0.1),
              matrix_world=mw)


def _mesh_object(name, uid, matrix=None, factory=FB.cube_mesh):
    ob = FB.geometry_object(name, factory, kind='MESH', matrix=matrix)
    ob.session_uid = uid
    ob.data = NS(session_uid=uid * 10)
    return ob


def _depsgraph(props, objects, updates=(), mode='VIEWPORT', space=None,
               frame=1):
    """A depsgraph whose instances can be walked any number of times (the
    exporter walks them five times), with a view layer that answers the
    two per-viewport visibility questions from `space.hidden`."""
    dg = FB.depsgraph_of(props, objects)
    dg.scene.frame_current = frame
    dg.object_instances = [
        NS(object=o, matrix_world=FB._Mat(o.matrix_world), show_self=True,
           is_instance=False) for o in objects]
    dg.updates = list(updates)
    dg.mode = mode
    for o in objects:
        if not hasattr(o, 'visible_get'):
            o.visible_get = (lambda view_layer=None, viewport=None, _o=o:
                             _o.name not in getattr(viewport, 'hidden', ()))
            o.visible_in_viewport_get = (
                lambda sp, _o=o: _o.name not in getattr(sp, 'hidden', ()))
    dg.view_layer = NS(name='ViewLayer', objects=list(objects))
    return dg


def _engine_self(vp):
    """What HalcyonRenderEngine.view_update needs of `self`."""
    return NS(_viewport=lambda: vp, tag_redraw=lambda: None)


def _view_update(EN, vp, dg, space):
    ctx = NS(region=NS(width=W, height=H), space_data=space)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        EN.HalcyonRenderEngine.view_update(_engine_self(vp), ctx, dg)
    return buf.getvalue()


# ------------------------------------------------------- (A) the classifier


def test_a_update_classifier():
    """A 5.x light owns an embedded node tree that Blender tags with every
    edit of the light, so the update list is [Object, PointLight,
    ShaderNodeTree] and the tree sent each tick down the FULL export (real
    Blender: 7 ticks of a Power drag = 7 full exports, 59-118 ms each on 152
    meshes, against 0.33-0.46 ms through the lights-only path)."""
    _props, EN, EX, _compat, _pv = _mods()
    tree = _id('ShaderNodeTree', session_uid=41)
    lamp = _id('PointLight', node_tree=tree)
    lamp_ob = _id('Object', type='LIGHT')
    got = EN._classify_updates(_updates(_Up(lamp_ob, geo=True), _Up(lamp),
                                        _Up(tree, geo=True, xform=True)))
    check("the 5.x list of a light edit -- [Object(LIGHT), PointLight, the "
          "light's own ShaderNodeTree] -- is 'lights'", got == 'lights', got)
    mat_tree = _id('ShaderNodeTree', session_uid=77)
    check("a material's ShaderNodeTree alone is still 'full' (the halo ramp "
          "widget's road)",
          EN._classify_updates(_updates(_Up(mat_tree))) == 'full')
    check("a light edit beside ANOTHER node tree is 'full'",
          EN._classify_updates(_updates(_Up(lamp), _Up(tree), _Up(mat_tree)))
          == 'full')
    # the evaluated copies: both sides resolve to the original's uid
    orig = _id('ShaderNodeTree', session_uid=55)
    ev_tree = _id('ShaderNodeTree', session_uid=9001, original=orig)
    ev_lamp = _id('SpotLight', node_tree=_id('ShaderNodeTree', session_uid=9002,
                                             original=orig))
    check('evaluated copies are matched through .original',
          EN._classify_updates(_updates(_Up(ev_lamp), _Up(ev_tree)))
          == 'lights')
    bare = type('SunLight', (), {})()
    check("a light with no node_tree attribute (the pinned stub) is still "
          "'lights'", EN._classify_updates(_updates(_Up(bare))) == 'lights')
    # the six pins of test_render.test_light_edit_fast_paths, restated
    mesh_ob = _id('Object', type='MESH')
    pins = [
        (EN._classify_updates(_updates(_Up(bare))), 'lights'),
        (EN._classify_updates(_updates(_Up(lamp_ob, xform=True))), 'lights'),
        (EN._classify_updates(_updates(_Up(mesh_ob, xform=True))), 'full'),
        (EN._classify_updates(_updates(_Up(bare), _Up(_id('Mesh'), geo=True))),
         'full'),
        (EN._classify_updates(_updates(_Up(_id('Screen')))), 'none'),
        (EN._classify_updates(_updates()), 'full')]
    check('the six pins of test_light_edit_fast_paths hold',
          all(a == b for a, b in pins), str(pins))

    # ---- a scene-level dial: a Scene update carries the geometry and
    # transform flags exactly when one of the scene's ID properties was
    # written (measured in 5.2.2), and that value travels in the settings
    scene_prop = _Up(_id('Scene'), geo=True, xform=True)
    got = EN._classify_updates(_updates(scene_prop))
    check("a Scene update with geometry / transform flags alone (a "
          "scene.halcyon dial) is 'none': the cheap settings half applies it",
          got == 'none', got)
    check("...and beside a mesh object it is 'full' as ever",
          EN._classify_updates(_updates(scene_prop, _Up(mesh_ob))) == 'full')
    check("an unknown ID with a transform flag is still 'full'",
          EN._classify_updates(_updates(_Up(_id('Odd'), xform=True))) == 'full')

    # ---- the mesh cache's list of harmless IDs
    want = {'PointLight', 'SunLight', 'SpotLight', 'AreaLight',
            'FreestyleLineStyle'}
    check('the four light subclasses and the Freestyle line style cannot '
          'dirty a cached mesh', want <= set(EX._IRRELEVANT_IDS),
          str(sorted(want - set(EX._IRRELEVANT_IDS))))
    armed, EX._CACHE_ARMED = EX._CACHE_ARMED, True
    try:
        EX._DIRTY_ALL = False
        EX._DIRTY_ALL_BY = None
        EX._watch_updates(None, NS(updates=[NS(id=lamp), NS(id=tree)]))
        check("a light edit does not flush the mesh cache (the split line "
              "read 'dirt ALL(PointLight)')", not EX._DIRTY_ALL,
              str(EX._DIRTY_ALL_BY))
        EX._watch_updates(None, NS(updates=[NS(id=_id('Key'))]))
        check('an unrecognised ID still flushes it whole',
              EX._DIRTY_ALL and EX._DIRTY_ALL_BY == 'Key')
    finally:
        EX._DIRTY_ALL = False
        EX._DIRTY_ALL_BY = None
        EX._CACHE_ARMED = armed


# --------------------------------------------------- (B) viewport visibility


def test_b_viewport_visibility():
    """H, Alt+H, the outliner's eye on an object or a collection, Local
    View and the viewport's object-type filter report one bare Scene update.
    The classifier reads that as 'none', and for the collection eye and
    Local View even a full export kept the object, because the only test
    was the instance's show_self (real Blender, 14 switches: 0 px before,
    the forced full export 0 px for six of them; all 14 arrive now)."""
    props, EN, EX, compat, PV = _mods()
    a = _mesh_object('A', 21)
    m2 = np.eye(4, dtype=np.float32)
    m2[0, 3] = 3.0
    b = _mesh_object('B', 22, matrix=m2)
    lamp = _lamp_object()
    objs = [a, b, lamp]
    space = NS(hidden=set())
    scene_poke = [_Up(_id('Scene'))]
    vp = PV.Viewport()
    said = _view_update(EN, vp, _depsgraph(props, objs, space=space), space)
    check('the first view_update exports the scene whole',
          vp.scene is not None and [o.name for o in vp.scene.objects]
          == ['A', 'B'] and len(vp.scene.lights) == 1 and not vp._said,
          said[-300:])
    v0 = vp.version

    # ---- a poke that changes nothing visible costs nothing
    _view_update(EN, vp, _depsgraph(props, objs, scene_poke), space)
    check('a Scene poke that hides nothing keeps the parked export',
          vp.version == v0)

    # ---- H on the lamp
    space.hidden.add('Lamp')
    _view_update(EN, vp, _depsgraph(props, objs, scene_poke), space)
    check('hiding the lamp (a bare Scene update) re-exports, and the lamp '
          'is gone from the viewport scene',
          vp.version == v0 + 1 and vp.scene.lights == []
          and [o.name for o in vp.scene.objects] == ['A', 'B'],
          f'version {v0}->{vp.version}, lights {len(vp.scene.lights)}')
    space.hidden.discard('Lamp')
    _view_update(EN, vp, _depsgraph(props, objs, scene_poke), space)
    check('un-hiding it brings it back', len(vp.scene.lights) == 1
          and vp.version == v0 + 2)

    # ---- a mesh
    space.hidden.add('B')
    _view_update(EN, vp, _depsgraph(props, objs, scene_poke), space)
    check('hiding a mesh takes it out of the viewport scene',
          [o.name for o in vp.scene.objects] == ['A'])
    space.hidden.discard('B')
    _view_update(EN, vp, _depsgraph(props, objs, scene_poke), space)
    check('...and un-hiding returns it',
          [o.name for o in vp.scene.objects] == ['A', 'B'])

    # ---- the fingerprint itself
    calls = []
    for o in objs:
        real = o.visible_get
        o.visible_get = (lambda view_layer=None, viewport=None, _r=real:
                         (calls.append(1), _r(view_layer, viewport))[1])
    dg = _depsgraph(props, objs, scene_poke)
    s0 = EN._visible_sig(dg, space)
    space.hidden.add('A')
    s1 = EN._visible_sig(dg, space)
    space.hidden.discard('A')
    check('the fingerprint moves when an object is hidden and returns when '
          'it is shown again',
          s0 is not None and s1 != s0 and EN._visible_sig(dg, space) == s0)
    calls.clear()
    lamp_tick = _depsgraph(props, objs, [_Up(_id('PointLight'))])
    check('a poke that names no Scene keeps the known fingerprint and asks '
          'no object anything (every visibility switch reports a Scene)',
          EN._visible_sig(lamp_tick, space, s0) == s0 and not calls)
    check('...with nothing known yet it is computed all the same',
          EN._visible_sig(lamp_tick, space, None) == s0 and len(calls) == 3)
    check('no space (headless, F12): no fingerprint, nothing compared',
          EN._visible_sig(dg, None) is None)

    # ---- the export rule: the viewport's space decides, F12 does not ask
    space.hidden.update(('B', 'Lamp'))
    st = RenderSettings()
    vp_sc = EX.export_scene(_depsgraph(props, objs), st, [], space=space)
    f12_sc = EX.export_scene(_depsgraph(props, objs, mode='RENDER'), st, [],
                             space=space)
    bare_sc = EX.export_scene(_depsgraph(props, objs), st, [])
    check('a VIEWPORT export leaves out what the space hides',
          [o.name for o in vp_sc.objects] == ['A'] and vp_sc.lights == [])
    check('a RENDER export keeps its own rule (show_self: hide_render '
          'decides), whatever space is handed in',
          [o.name for o in f12_sc.objects] == ['A', 'B']
          and len(f12_sc.lights) == 1)
    check('...and so does an export with no space',
          [o.name for o in bare_sc.objects] == ['A', 'B']
          and len(bare_sc.lights) == 1)
    parked = bare_sc
    lo = EX.export_lights_into(parked, _depsgraph(props, objs), space=space)
    check('the lights-only export leaves a hidden lamp out as well',
          lo is not None and lo.lights == [] and lo.mesh is parked.mesh)
    check('compat.in_view: no space shows everything; an object that cannot '
          'answer is shown',
          compat.in_view(NS(object=NS()), None)
          and compat.in_view(NS(object=NS()), space))
    space.hidden.clear()

    # ---- a lamp edit and a visibility change on one poke is a full export
    vp2 = PV.Viewport()
    _view_update(EN, vp2, _depsgraph(props, objs), space)
    space.hidden.add('B')
    lamp_and_scene = [_Up(_id('Object', type='LIGHT'), xform=True),
                      _Up(_id('Scene'))]
    _view_update(EN, vp2, _depsgraph(props, objs, lamp_and_scene), space)
    check("'lights' with a changed fingerprint takes the full export (the "
          'mesh hidden on the same poke leaves too)',
          [o.name for o in vp2.scene.objects] == ['A'])
    space.hidden.clear()


# ------------------------------------------------------ (C) update callbacks


def test_c_settings_carry_an_update_callback():
    """Blender calls a property's update -- and tags its owner for the
    depsgraph -- from Python only when the property HAS an update function.
    None of HalcyonLightSettings had one: `light.halcyon.shadow_softness = 8`
    from a script, and the panel's own 'Use Render Shadow Settings' button,
    never reached the rendered viewport (real Blender: view_update +0, 0 px;
    978 px arrive now, through the lights-only path at 0.2 ms)."""
    props, _EN, _EX, _compat, _pv = _mods()
    for cls in (props.HalcyonLightSettings, props.HalcyonSettings,
                props.HalcyonWorldSettings, props.HalcyonMaterialSettings):
        ann = cls.__annotations__
        missing = [k for k, p in ann.items()
                   if getattr(p, 'kind', '') != 'CollectionProperty'
                   and not callable((getattr(p, 'kw', None) or {})
                                    .get('update'))]
        check(f'every property of {cls.__name__} ({len(ann)}) has an update '
              'callback', not missing, ', '.join(missing[:6]))
    light = props.HalcyonLightSettings.__annotations__
    check('the dials the report names are among them (Softness, the Shadows '
          'menu, Shadow Map Size the panel button writes, the cookie)',
          all(light[k].kw['update'] is props._settings_written
              for k in ('shadow_softness', 'shadow', 'shadow_map_size',
                        'cookie', 'volumetric_occlusion')))
    scene_ann = props.HalcyonSettings.__annotations__
    world_ann = props.HalcyonWorldSettings.__annotations__
    check('the two properties that write other dials keep their own '
          'callbacks',
          scene_ann['render_device'].kw['update'].__name__ == '_device_flip'
          and world_ann['paint_look'].kw['update']
          is props._painted_look_changed)
    check('the callback does nothing (Blender does the tagging): it returns '
          'None', props._settings_written(None, None) is None)
    fn = ast.parse(inspect.getsource(props._settings_written)).body[0]
    body = [n for n in fn.body
            if not (isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant))]
    check('...and its body is one bare return: no write can start another '
          'callback',
          len(body) == 1 and isinstance(body[0], ast.Return)
          and not any(isinstance(n, (ast.Call, ast.Assign, ast.Attribute))
                      for n in ast.walk(fn)))


# ------------------------------------------------- (D) the lights-only export


def _style_scene(st):
    sc = demo_scene(st, with_texture=False)
    sc.lights = [Light(type='POINT', name='Key', position=(2.0, -3.0, 6.0),
                       energy=900.0, shadow='NONE', light_style='FLICKER_A')]
    sc.frame, sc.fps, sc.time = 1, 24, 1.0 / 24.0
    return sc


def test_d_lights_only_export():
    """The lights-only export kept the parked frame number, so a keyframed
    lamp scrubbed through the timeline animated its Power and position while
    its Light Style and Caustics stood still (real Blender: 92150, 6330 and
    18950 px against a full export of the same frame; 0 px now)."""
    props, EN, EX, _compat, PV = _mods()
    st = RenderSettings()
    st.resolution_x, st.resolution_y = W, H
    st.aa_mode = 'NONE'
    st.output_scale = 'NONE'
    parked = _style_scene(st)
    lamp = NS(type='LIGHT', name='Key', session_uid=5,
              data=NS(type='POINT', color=(1.0, 1.0, 1.0), energy=900.0,
                      halcyon=FB.live(props.HalcyonLightSettings,
                                      light_style='FLICKER_A'),
                      shadow_soft_size=0.0, use_shadow=False),
              matrix_world=None)
    mw = np.eye(4, dtype=np.float32)
    mw[:3, 3] = (2.0, -3.0, 6.0)

    def dg(frame, objects=None):
        return NS(scene=NS(unit_settings=NS(scale_length=1.0),
                           frame_current=frame, render=NS(fps=24)),
                  object_instances=[NS(object=o, matrix_world=mw,
                                       show_self=True)
                                    for o in (objects or [lamp])])
    out = EX.export_lights_into(parked, dg(13))
    check('the lights-only export carries the depsgraph frame, fps and time '
          '(13, 24, 13/24), not the parked frame 1',
          out.frame == 13 and out.fps == 24
          and abs(out.time - 13.0 / 24.0) < 1e-12,
          f'frame {out.frame} time {out.time:.4f}')
    check('...and the parked scene keeps its own clock',
          parked.frame == 1 and abs(parked.time - 1.0 / 24.0) < 1e-12)
    full = _style_scene(st)
    full.lights = out.lights
    full.frame, full.fps, full.time = 13, 24, 13.0 / 24.0
    LI.clear_shadow_cache()
    a = R.render(out, st)
    b = R.render(full, st)
    stale = copy.copy(out)
    stale.frame, stale.time = 1, 1.0 / 24.0
    c = R.render(stale, st)
    check('the lights-only frame of frame 13 IS the full export\'s (bitwise)',
          np.array_equal(a, b))
    check('...and differs from the same lamps under the parked clock (the '
          'flicker is at another letter)',
          int((np.abs(a - c)[..., :3].max(axis=2) > 1.0 / 255.0).sum()) > 100,
          f'{int((np.abs(a - c)[..., :3].max(axis=2) > 1.0 / 255.0).sum())} px')
    no_clock = NS(scene=NS(unit_settings=NS(scale_length=1.0)),
                  object_instances=dg(1).object_instances)
    kept = EX.export_lights_into(parked, no_clock)
    check('a stub depsgraph with no clock keeps the parked one',
          kept.frame == parked.frame and kept.time == parked.time)

    # ---- a material that bakes a lamp's frame: Object Projection
    spot = NS(name='Spot', type='LIGHT', data=NS(type='SPOT', spot_size=1.2),
              matrix_world=np.eye(4, dtype=np.float32))
    empty = NS(name='Empty', type='EMPTY', data=None,
               matrix_world=np.eye(4, dtype=np.float32))
    node = lambda ob: NS(name='Object Projection', object=ob, mode='AUTO',
                         inputs=[])
    p_lamp = EX.object_map_props(node(spot), [])
    p_empty = EX.object_map_props(node(empty), [])
    check('object_map_props marks a node that projects from a LAMP',
          p_lamp.get('object_is_light') is True)
    check('...and writes nothing new for any other object (its props are '
          'what they were)',
          'object_is_light' not in p_empty
          and set(p_empty) == {'object_inv', 'proj_mode', 'proj_tan',
                               'proj_half'})
    mat = Material(name='Floor')
    mat.graph = {'nodes': {'n1': {'bl_idname': 'HALCYON_ObjectMapNode',
                                  'props': p_lamp}}}
    check('_lamp_bound reads it off the exported materials',
          EX._lamp_bound([mat]) == (True, False)
          and EX._lamp_bound([Material(name='Plain')]) == (False, False))
    bound = copy.copy(parked)
    bound.lamp_bound = (True, False)
    check('a parked scene whose material projects from a lamp takes NO '
          'lights-only export (None: the caller exports fully) -- the '
          'projection stayed at the old pose while the lamp moved '
          '(8525 / 10755 px in the real viewport)',
          EX.export_lights_into(bound, dg(1)) is None)

    # ---- a Light Group names its lamps at material export
    grouped = copy.copy(parked)
    grouped.lamp_bound = (False, True)
    same = EX.export_lights_into(grouped, dg(1))
    check('a Light Group with the same lamp names keeps the fast path',
          same is not None and same.mesh is parked.mesh)
    renamed = copy.copy(lamp)
    renamed.name = 'Key Lamp'
    check('...a renamed lamp is a full export (a rename reports the lamp '
          'OBJECT alone, which classifies as a lights-only edit)',
          EX.export_lights_into(grouped, dg(1, [renamed])) is None)
    second = copy.copy(lamp)
    second.name = 'Fill'
    check('...and so is a lamp that appeared',
          EX.export_lights_into(grouped, dg(1, [lamp, second])) is None)
    with_wl = copy.copy(grouped)
    wl = Light(type='SUN', name='World Light')
    wl.world_light = True
    with_wl.lights = list(parked.lights) + [wl]
    check("the world's own light (made at render entry, never exported) is "
          'not a name that moved',
          EX.export_lights_into(with_wl, dg(1)) is not None)

    # ---- view_update takes the full export when the fast path stands down
    a_ob = _mesh_object('A', 31)
    lamp_ob = _lamp_object()
    space = None
    vp = PV.Viewport()
    _view_update(EN, vp, _depsgraph(props, [a_ob, lamp_ob]), space)
    first = vp.scene
    lamp_move = [_Up(_id('Object', type='LIGHT'), xform=True)]
    _view_update(EN, vp, _depsgraph(props, [a_ob, lamp_ob], lamp_move), space)
    check('a lamp move on an ordinary scene shares the parked mesh '
          '(lights-only)', vp.scene is not first and vp.scene.mesh is first.mesh)
    vp.scene.lamp_bound = (True, False)
    held = vp.scene
    _view_update(EN, vp, _depsgraph(props, [a_ob, lamp_ob], lamp_move), space)
    check('...and on a scene with a lamp-bound material it is a full export '
          '(a new mesh object, the materials exported again)',
          vp.scene is not held and vp.scene.mesh is not held.mesh
          and not vp._said)

    # ---- the Light Group itself: resolved through bpy at material export
    import bpy
    coll = NS(all_objects=[NS(type='LIGHT', name='Key'),
                           NS(type='MESH', name='Box'),
                           NS(type='LIGHT', name='Rim')])
    had = getattr(bpy.data, 'collections', None)
    bpy.data.collections = NS(get=lambda n: coll if n == 'Stage' else None)
    try:
        bi = NS(name='BI Material', bl_idname='HALCYON_BIMaterialNode',
                light_group='Stage', inputs=[], outputs=[])
        warn = []
        got = EX._node_props(bi, {}, {}, warn)
        check("a BI Material's Light Group resolves to the lamps in its "
              'collection (export.py had no bpy in scope there: the NameError '
              'was swallowed, every group named no lamps and the material '
              'took every lamp -- 0 px for the group in the real viewport, '
              '1677 px now)',
              got.get('light_group_lights') == ['Key', 'Rim'] and not warn,
              f"{got.get('light_group_lights')} {warn}")
        warn = []
        bi.light_group = 'Nowhere'
        got = EX._node_props(bi, {}, {}, warn)
        check('...and a group that names no collection still says so',
              got.get('light_group_lights') == [] and len(warn) == 1)
    finally:
        if had is None:
            del bpy.data.collections
        else:
            bpy.data.collections = had


# ------------------------------------------------------ (E) the F12 mesh cache


class _MainGraph:
    """The view layer's VIEWPORT depsgraph, as _heard_objects reads it."""
    bl_rna = NS(functions={'id_type_updated': NS(parameters={'id_type': NS(
        enum_items=[NS(identifier=t)
                    for t in ('OBJECT', 'MESH', 'LIGHT', 'SCENE')])})})
    mode = 'VIEWPORT'

    def __init__(self):
        self.pending = set()

    def id_type_updated(self, kind):
        return kind in self.pending


def _layer(objects, hide_viewport=False, exclude=False, children=()):
    return NS(collection=NS(hide_viewport=hide_viewport, objects=objects),
              exclude=exclude, children=list(children))


def test_e_f12_mesh_cache():
    """A newly built RENDER depsgraph lists every ID it holds, and the
    same-frame frame_change of every F12 walked that list into the dirt: no
    F12 ever reused a cached mesh (real Blender, nothing edited: 'meshes
    live 63 cached 0'; 'live 0 cached 63' now, the interactive job and the
    blocking one alike)."""
    props, _EN, EX, _compat, _pv = _mods()
    import bpy

    # ---- which objects the main depsgraph holds and reports
    def tok(name, hide=False):
        return NS(name=name, session_uid=None, hide_viewport=hide)
    o = {n: tok(n) for n in 'abcdefg'}
    o['b'].hide_viewport = True
    tree = _layer([o['a'], o['b']], children=[
        _layer([o['c']], hide_viewport=True,
               children=[_layer([o['d']])]),
        _layer([o['e']], exclude=True, children=[_layer([o['f']])]),
        _layer([o['g']])])
    got = EX._viewport_enabled(NS(layer_collection=tree))
    check("_viewport_enabled: an object's own Disable in Viewports removes "
          "it, a collection's is inherited by everything under it, Exclude "
          "is each layer collection's own",
          got == {('n', 'a'), ('n', 'f'), ('n', 'g')}, str(sorted(got)))

    # ---- when the main depsgraph's reports may be trusted
    main = _MainGraph()
    a = _mesh_object('A', 21)
    b = _mesh_object('B', 22)
    a.hide_viewport = b.hide_viewport = False
    vl = NS(name='ViewLayer', depsgraph=main,
            layer_collection=_layer([a, b]))
    bscene = NS(name='Scene')
    render_dg = NS(mode='RENDER', scene=bscene, view_layer=vl, updates=[])
    had_wm = getattr(bpy.data, 'window_managers', None)
    hooks = bpy.app.handlers
    foreign = lambda *a_: None
    foreign.__module__ = 'another_addon.handlers'
    own = lambda *a_: None
    own.__module__ = EX.__name__.rpartition('.')[0] + '.engine'
    try:
        bpy.data.window_managers = [NS(windows=[NS(scene=bscene,
                                                   view_layer=vl)])]
        check('a still F12 of the scene a window shows, nothing pending, no '
              'foreign handler: the viewport-enabled objects are heard',
              EX._heard_objects(render_dg) == {21, 22},
              str(EX._heard_objects(render_dg)))
        check('a VIEWPORT depsgraph is never taken for one (its list is '
              'real changes)',
              EX._heard_objects(NS(mode='VIEWPORT', scene=bscene,
                                   view_layer=vl)) is None)
        main.pending.add('MESH')
        check('a tag the main depsgraph has not reported yet: nothing is '
              'assumed (the whole list is walked, as before)',
              EX._heard_objects(render_dg) is None)
        main.pending.clear()
        hooks.render_pre.append(foreign)
        check('a foreign render_pre handler: nothing is assumed (it can move '
              'a vertex only the render depsgraph sees)',
              EX._heard_objects(render_dg) is None)
        hooks.render_pre.remove(foreign)
        hooks.render_pre.append(own)
        check("the add-on's own handlers do not count as foreign",
              EX._heard_objects(render_dg) == {21, 22})
        hooks.render_pre.remove(own)
        other = NS(name='Other')
        check('a scene no window shows (rendered by name from a script): '
              'nothing is assumed -- nobody evaluates its main depsgraph '
              '(2875 px of stale mesh under a rule without this test)',
              EX._heard_objects(NS(mode='RENDER', scene=other,
                                   view_layer=vl)) is None)
        b.hide_viewport = True
        check('an object disabled in viewports but enabled in renders is '
              'not heard: nobody reports its edits (145 px of stale mesh '
              'under a rule without this test)',
              EX._heard_objects(render_dg) == {21})
        b.hide_viewport = False

        # ---- the walk itself
        def fresh_graph(*objs, extra=()):
            ups = [NS(id=_id('Object', session_uid=x.session_uid))
                   for x in objs]
            ups += [NS(id=_id(t, session_uid=900 + i))
                    for i, t in enumerate(('Scene', 'PointLight', 'Mesh',
                                           'Key') + tuple(extra))]
            return NS(mode='RENDER', scene=bscene, view_layer=vl, updates=ups)
        armed, EX._CACHE_ARMED = EX._CACHE_ARMED, True
        fake_scene = NS(name='R254 E', frame_current=7, frame_subframe=0.0)

        def reset():
            EX._DIRTY_OBJS.clear()
            EX._DIRTY_DATA.clear()
            EX._DIRTY_ALL = False
            EX._DIRTY_ALL_BY = None
        EX._on_frame_change(fake_scene, fresh_graph())      # records frame 7
        reset()
        EX._on_frame_change(fake_scene, fresh_graph(a, b))
        check("a still F12's RENDER depsgraph that lists every object, a "
              'Mesh and a Key dirties NOTHING the main depsgraph reports',
              not EX._DIRTY_OBJS and not EX._DIRTY_DATA and not EX._DIRTY_ALL,
              f'{EX._DIRTY_OBJS} {EX._DIRTY_DATA} {EX._DIRTY_ALL_BY}')
        reset()
        b.hide_viewport = True
        EX._on_frame_change(fake_scene, fresh_graph(a, b))
        check('...an object the main depsgraph does not hold is dirtied as '
              'always, and only it', EX._DIRTY_OBJS == {22}
              and not EX._DIRTY_ALL, str(EX._DIRTY_OBJS))
        b.hide_viewport = False
        reset()
        hooks.render_pre.append(foreign)
        EX._on_frame_change(fake_scene, fresh_graph(a, b))
        check('with a foreign handler the whole list is walked exactly as '
              'before (every object dirty, the unknown ID flushes)',
              EX._DIRTY_OBJS == {21, 22} and EX._DIRTY_ALL
              and EX._DIRTY_ALL_BY == 'Key',
              f'{EX._DIRTY_OBJS} {EX._DIRTY_ALL_BY}')
        hooks.render_pre.remove(foreign)
        reset()
        fake_scene.frame_current = 8
        EX._on_frame_change(fake_scene, fresh_graph(a, b))
        check('a real frame step still flushes whole',
              EX._DIRTY_ALL and EX._DIRTY_ALL_BY == 'frame')
        reset()
        vp_same = NS(mode='VIEWPORT', updates=[NS(id=_id('Object',
                                                         session_uid=21))])
        EX._on_frame_change(fake_scene, vp_same)
        check("a VIEWPORT depsgraph's same-frame list is still walked",
              EX._DIRTY_OBJS == {21})
        reset()

        # ---- end to end: two F12 exports with the frame_change between
        mat = FB.halcyon_material('E Mat')
        old_mats = bpy.data.materials
        bpy.data.materials = [mat]
        EX._MESH_CACHE.clear()

        def f12_export():
            dg = _depsgraph(props, [a, b], mode='RENDER')
            return EX.export_scene(dg, RenderSettings(), [])
        try:
            f12_export()
            first = dict(EX.EXPORT_SPLIT)
            EX._on_frame_change(fake_scene, fresh_graph(a, b))
            sc2 = f12_export()
            second = dict(EX.EXPORT_SPLIT)
            check('the second F12 of an unedited scene serves every mesh '
                  'from the cache (it read every one live before)',
                  first['meshes'] == 2 and second['meshes'] == 0
                  and second['cached'] == 2, f'{first} {second}')
            EX._MESH_CACHE.clear()
            EX._DIRTY_ALL = True
            cold = f12_export()
            check('...and the cached export is the live one, array for array',
                  all(np.array_equal(getattr(sc2.mesh, n),
                                     getattr(cold.mesh, n))
                      for n in ('verts', 'normals', 'tris', 'mat_index',
                                'obj_index', 'face_normals', 'smooth')))
            # an edit the main depsgraph reported is still honoured
            EX._watch_updates(None, NS(updates=[NS(id=_id(
                'Object', session_uid=22))]))
            EX._on_frame_change(fake_scene, fresh_graph(a, b))
            f12_export()
            third = dict(EX.EXPORT_SPLIT)
            check('an object the main depsgraph reported is exported live, '
                  'the other one stays cached',
                  third['meshes'] == 1 and third['cached'] == 1, str(third))
        finally:
            bpy.data.materials = old_mats
            EX._MESH_CACHE.clear()
            reset()
            EX._CACHE_ARMED = armed
    finally:
        for fn in (foreign, own):
            if fn in hooks.render_pre:
                hooks.render_pre.remove(fn)
        if had_wm is None:
            del bpy.data.window_managers
        else:
            bpy.data.window_managers = had_wm


# -------------------------------------------------------- (F) the held cel


def test_f_film_hold_follows_edits():
    """Shoot On twos keeps the key frame's picture for the held frames. The
    entry was keyed on the settings only, so an F12 on a held frame after
    the lamp was moved or re-powered returned the old picture and printed
    'holds frame 1 (not rendered again)' (real Blender: 0 px after Power x4
    and a red lamp; 5702 px now, and a sequence render keeps its holds)."""
    _props, EN, _EX, _compat, _pv = _mods()
    st = RenderSettings()
    st.film_hold = 2
    img = np.zeros((4, 4, 4), np.float32)

    def stored():
        EN._HOLD_CACHE.clear()
        EN.hold_store('Scene', 1, (4, 4), st, img, None, None, None)

    def held():
        return EN.hold_lookup('Scene', 2, 1, 2, (4, 4), st)[1] is not None
    scene = NS(name='Scene')
    lamp_edit = _updates(_Up(_id('Object', type='LIGHT'), geo=True),
                         _Up(_id('PointLight')))
    stored()
    check('a held frame finds its key frame', held())
    EN._drop_holds_handler(scene, lamp_edit)
    check('an edit of a lamp retires the held cel', not held())
    stored()
    EN._drop_holds_handler(scene, _updates(_Up(_id('Object', type='MESH'),
                                               xform=True)))
    check('...and so does a moved mesh', not held())
    stored()
    EN._drop_holds_handler(scene, _updates(_Up(_id('Screen'))))
    EN._drop_holds_handler(scene, _updates(_Up(_id('Scene'))))
    check('interface churn and a bare Scene poke (a selection, a hide) keep '
          'it', held())
    lamp_edit.mode = 'RENDER'
    EN._drop_holds_handler(scene, lamp_edit)
    check("a RENDER depsgraph's updates keep it (the frame steps of a "
          'sequence are not edits)', held())
    lamp_edit.mode = 'VIEWPORT'
    real = EN._render_job_running
    EN._render_job_running = lambda: True
    try:
        EN._drop_holds_handler(scene, lamp_edit)
        check("...and nothing is dropped while Blender's render job runs",
              held())
    finally:
        EN._render_job_running = real
    EN._drop_holds_handler('/path/to/another.blend')
    check('load_post drops it (the next file\'s Scene is another picture)',
          not held())
    EN._HOLD_CACHE.clear()
    EN._drop_holds_handler(scene, lamp_edit)       # an empty cache: no-op
    src = inspect.getsource(EN.register)
    check('register() hangs the handler on depsgraph_update_post and '
          'load_post, unregister() takes it off',
          '_drop_holds_handler' in src
          and '_drop_holds_handler' in inspect.getsource(EN.unregister))


# -------------------------------------------------- (G) the image pixel cache


class _Image:
    """What compat.image_pixels reads of a bpy Image."""

    def __init__(self, name, px, source='GENERATED', path='', dirty=False):
        self.name_full = name
        self.source = source
        self.filepath_raw = path
        self.is_dirty = dirty
        self.has_data = True
        self.packed_file = None
        self.library = None
        self.colorspace_settings = NS(name='sRGB')
        self.reads = 0
        self.set(px)

    def set(self, px):
        self._px = np.asarray(px, np.float32)
        self.size = (self._px.shape[1], self._px.shape[0])
        owner = self

        class _P:
            def foreach_get(self, buf):
                owner.reads += 1
                buf[:] = owner._px.reshape(-1)
        self.pixels = _P()


def test_g_image_pixel_cache():
    """The pixel cache was keyed on the name, size, source, path, the dirty
    flag and the colour space. None of those moves when a file is rewritten
    and reloaded, or when an image that is already dirty is painted again: a
    projector's gobo stayed as first read (real Blender viewport: 0 px after
    two repaints and after a reload; 9581, 11596 and 10456 px now)."""
    _props, _EN, _EX, compat, _pv = _mods()
    import bpy
    had_path = getattr(bpy, 'path', None)
    bpy.path = NS(abspath=lambda p, library=None: p)
    rng = np.random.RandomState(3)
    a, b, c = (rng.rand(4, 6, 4).astype(np.float32) for _ in range(3))
    try:
        compat.clear_image_cache()
        img = _Image('Gobo', a, dirty=True)
        p0 = compat.image_pixels(img)
        check('a dirty image is read', np.array_equal(p0, a))
        img.set(b)
        p1 = compat.image_pixels(img)
        check('...repainted while already dirty, it is read AGAIN (the old '
              'key could not move: the first read was served forever)',
              np.array_equal(p1, b))
        img.set(c)
        check('...and again after the next stroke',
              np.array_equal(compat.image_pixels(img), c))
        n = img.reads
        p3 = compat.image_pixels(img)
        p4 = compat.image_pixels(img)
        check('an unchanged dirty image is still read (Blender reports no '
              'pixel write), and the cached ARRAY is handed back, so the '
              'identity-keyed caches downstream stay warm',
              img.reads == n + 2 and p4 is p3)
        check('one generation per image is kept',
              sum(1 for k in compat._PIXEL_CACHE if k[0] == 'Gobo') == 1
              and len(compat._PIXEL_ORDER) == len(compat._PIXEL_CACHE))

        # ---- a clean FILE image is keyed on the file itself
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'gobo.png')
            with open(path, 'wb') as fh:
                fh.write(b'x' * 100)
            st0 = os.stat(path)
            fimg = _Image('File', a, source='FILE', path=path)
            f0 = compat.image_pixels(fimg)
            f1 = compat.image_pixels(fimg)
            check('a clean file image is read once and served from the '
                  'cache', fimg.reads == 1 and f1 is f0)
            # rewritten with the same mtime but another size, then reloaded
            with open(path, 'wb') as fh:
                fh.write(b'y' * 164)
            os.utime(path, ns=(st0.st_atime_ns, st0.st_mtime_ns))
            fimg.set(b)
            check('the file rewritten (same mtime, another size) and '
                  'reloaded is read again',
                  np.array_equal(compat.image_pixels(fimg), b)
                  and fimg.reads == 2)
            os.utime(path, ns=(st0.st_atime_ns, st0.st_mtime_ns + 1))
            fimg.set(c)
            check('...and so is one whose mtime moved by a nanosecond at the '
                  'same size (a rewrite within the same second)',
                  np.array_equal(compat.image_pixels(fimg), c)
                  and fimg.reads == 3)
            check('...with the older generations dropped',
                  sum(1 for k in compat._PIXEL_CACHE if k[0] == 'File') == 1)
            # painted in Blender: dirty wins over the file
            fimg.is_dirty = True
            fimg.set(a)
            check('a file image painted in Blender (dirty) is read fresh',
                  np.array_equal(compat.image_pixels(fimg), a))
            fimg.is_dirty = False               # reverted / saved
            fimg.set(c)
            check('...and after it is clean again the file decides once more '
                  '(no entry from before the paint is served)',
                  np.array_equal(compat.image_pixels(fimg), c))
            packed = _Image('Packed', a, source='FILE', path=path)
            packed.packed_file = object()
            check('a packed image has no file stamp (its pixels are in the '
                  '.blend)', compat._file_stamp(packed) is None
                  and compat._file_stamp(fimg) is not None)
        seq = _Image('Seq', a, source='SEQUENCE')
        compat.image_pixels(seq)
        compat.image_pixels(seq)
        check('a SEQUENCE is never cached, as before', seq.reads == 2
              and not any(k[0] == 'Seq' for k in compat._PIXEL_CACHE))
        compat.clear_image_cache()
        check('clear_image_cache empties both tables',
              not compat._PIXEL_CACHE and not compat._PIXEL_ORDER)
    finally:
        compat.clear_image_cache()
        if had_path is None:
            del bpy.path
        else:
            bpy.path = had_path


# ---------------------------------------------------------- (H) the mesh key


def _big_scene(st, variant):
    """Floor + a 52,488-vertex ball + a 24-vertex box (the LAST part: with
    53,003 vertices the key's stride is 103 and not one box vertex, box
    triangle or box material index is sampled)."""
    mats = [Material(name='Floor', index=0, model='LAMBERT',
                     diffuse=(0.8, 0.8, 0.8), specular_level=0.0),
            Material(name='Ball', index=1, model='LAMBERT',
                     diffuse=(0.2, 0.4, 0.8), specular_level=0.0),
            Material(name='Box', index=2, model='LAMBERT',
                     diffuse=(0.8, 0.3, 0.2), specular_level=0.0)]
    box = cube(centre=(0.0, -0.5, 1.6), size=0.9,
               mat=1 if variant == 'material' else 2, obj=2)
    parts = [plane(z=0.0, size=14.0, mat=0, obj=0),
             sphere(centre=(-2.5, 1.5, 1.0), radius=1.0, mat=1, obj=1,
                    segs=324, rings=162), box]
    mesh = _mesh_concat(parts)
    mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
    if variant == 'normals':
        n = np.array(mesh.normals, np.float32)
        nb, nt = box[0].shape[0], box[3].shape[0]
        v = mesh.verts[-nb:] - np.array([0.0, -0.5, 1.6], np.float32)
        n[-nb:] = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True),
                                 1e-9)
        mesh.normals = n
        mesh.smooth[-nt:] = True
    objs = [ObjectInfo(name=n_, index=i, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=np.float32))
            for i, n_ in enumerate(('Floor', 'Ball', 'Box'))]
    cam = Camera(matrix_world=look_at_matrix((0.0, -7.5, 6.5), (0.0, 0.6, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    lamp = Light(type='SPOT', name='L', position=(1.0, -2.0, 7.0),
                 direction=(-0.1, 0.25, -0.96), energy=1500.0, spot_size=1.3,
                 spot_blend=0.2, shadow='MAP', radius=0.0)
    sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=[lamp],
               camera=cam, world=World(color=(0, 0, 0),
                                       ambient=(0.02, 0.02, 0.02)),
               settings=st)
    sc.images = {}
    sc.frame, sc.fps, sc.time = 1, 24, 1 / 24
    return sc


def _clear_render_caches():
    LI.clear_shadow_cache()
    R._GBUF_CACHE.clear()
    R._TEX_CACHE.clear()
    R._BVH_CACHE.clear()
    GSH._PLAN_CACHE.clear()


def _px(a, b):
    d = np.abs(a[..., :3] - b[..., :3]).max(axis=2)
    return int((d > 1.0 / 255.0).sum()), float(d.max())


def test_h_mesh_key_digests_every_array():
    """gpu/shade._mesh_key keys the plan and the attribute / triangle
    uploads. Its position term is an exact digest; every other term was a
    sum over about 512 samples, so a change between the samples kept the
    key: a 24-vertex box among 53,003 vertices switched from flat to smooth
    kept its old normals on the device (245 px wrong), and the same box
    given another material kept the old triangle table, so the device
    refused every frame."""
    _props, _EN, _EX, _compat, PV = _mods()
    st = RenderSettings()
    st.shadows = True
    st.shadow_default = 'PER_LIGHT'
    st.preview_scale = 1
    st.render_device = 'GPU'
    st = PV.shape_settings(st, W, H)
    base = _big_scene(st, 'base')
    n = base.mesh.verts.shape[0]
    stride = max(1, n // 512)
    tstride = max(1, base.mesh.tris.shape[0] // 512)
    box_v = np.arange(n - 24, n)
    box_t = np.arange(base.mesh.tris.shape[0] - 12, base.mesh.tris.shape[0])
    check('the premise: not one vertex, triangle or material index of the '
          'box is among the samples the sums read',
          not (box_v % stride == 0).any() and not (box_t % tstride == 0).any()
          and not (box_t % stride == 0).any(),
          f'{n} verts, stride {stride}, tri stride {tstride}')
    k0 = GSH._mesh_key(base.mesh)
    for variant, what in (('normals', 'switched from flat to smooth'),
                          ('material', 'given another material')):
        other = _big_scene(st, variant)
        check(f'the box {what} (no vertex moved): the mesh key moves',
              GSH._mesh_key(other.mesh) != k0)
    check('the same content in fresh arrays is the same key (a re-export of '
          'an unedited scene still hits every cache)',
          GSH._mesh_key(_big_scene(st, 'base').mesh) == k0)
    check('the old terms are all still there, in their places, and the '
          'digests ride one tuple at the end: the key only got longer',
          k0[:2] == (n, base.mesh.tris.shape[0])
          and not any(isinstance(x, tuple) for x in k0[:-1])
          and isinstance(k0[-1], tuple)
          and len(k0[-1]) == len(GSH._MESH_KEY_ARRAYS)
          and ('gb_attrs',) + k0 != ('gb_tris',) + k0)

    # ---- the memo
    counted = []
    real = GSH._content_digest
    GSH._content_digest = lambda a_: (counted.append(1), real(a_))[1]
    try:
        m = _big_scene(st, 'base').mesh
        GSH._mesh_key(m)
        first = len(counted)
        GSH._mesh_key(m)
        GSH._mesh_key(m)
        check('the attribute digests are computed once per mesh: a draft, a '
              'refine and a lights-only export (the same MeshData) pay only '
              'the position digest that every call always paid',
              first == 1 + sum(1 for nm in GSH._MESH_KEY_ARRAYS
                               if getattr(m, nm, None) is not None)
              and len(counted) == first + 2, f'{first}, then {len(counted)}')
        before = GSH._mesh_key(m)
        m.mat_index = m.mat_index.copy()
        m.mat_index[-12:] = 0
        check('an attribute REPLACED by another array is digested again',
              GSH._mesh_key(m) != before)
        shallow = copy.copy(m)
        shallow.normals = shallow.normals.copy()
        shallow.normals[-24:] *= -1.0
        check('...also on a copy that carried the memo along',
              GSH._mesh_key(shallow) != GSH._mesh_key(m))
    finally:
        GSH._content_digest = real

    # ---- the GPU road, through the device's own upload cache
    for variant, what in (('normals', 'flat -> smooth'),
                          ('material', 'another material')):
        _clear_render_caches()
        with fakedevice.installed():
            def frame(scene):
                vp = PV.Viewport()
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    vp._render(None, scene, st, None, W, H, ('k',), 0, False)
                if vp.frame is None:
                    raise RuntimeError(buf.getvalue()[-800:])
                v = R.LAST_GPU_VERDICT
                return vp.frame.copy(), bool(v.get('engaged')), \
                    str(v.get('why') or '')
            f_a, on_a, _ = frame(_big_scene(st, 'base'))
            f_b, on_b, why_b = frame(_big_scene(st, variant))
        _clear_render_caches()
        with fakedevice.installed():
            f_c, on_c, _ = frame(_big_scene(st, variant))
        eff = _px(f_a, f_c)
        stale = _px(f_b, f_c)
        check(f'GPU road, box {what}: the warm frame is the cold frame and '
              'the device stays engaged',
              stale[0] == 0 and on_a and on_b and on_c and eff[0] > 50,
              f'effect {eff[0]} px, warm-vs-cold {stale[0]} px (max '
              f'{stale[1]:.3f}), engaged {on_b} {why_b[:60]}')
    _clear_render_caches()


def test_h2_an_object_turned_about_its_own_centre():
    """The perf cluster's verify-stale-sum-blind-10: a rotation about the
    centroid keeps sum(verts) and a symmetric object's summed normals, so a
    key made of sums could not move even at stride 1; the second F12 kept
    the old silhouette (6627 px on a cube). The position digest that landed
    covers it; this holds it there."""
    st = RenderSettings()
    st.resolution_x, st.resolution_y = W, H
    st.shadows = True
    st.shadow_default = 'MAP'
    st.aa_mode = 'NONE'
    st.output_scale = 'NONE'

    def scene(angle):
        box = list(cube(centre=(0.0, 0.0, 1.0), size=1.6, mat=1, obj=1))
        c, s = np.cos(angle), np.sin(angle)
        rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
        centre = np.array([0.0, 0.0, 1.0], np.float32)
        box[0] = ((box[0] - centre) @ rot.T + centre).astype(np.float32)
        box[1] = (box[1] @ rot.T).astype(np.float32)
        mesh = _mesh_concat([plane(z=0.0, size=10.0, mat=0, obj=0), tuple(box)])
        fn = getattr(mesh, 'face_normals', None)
        if fn is not None:
            nt = box[3].shape[0]
            fn = np.array(fn, np.float32)
            fn[-nt:] = fn[-nt:] @ rot.T
            mesh.face_normals = fn
        mats = [Material(name='Floor', index=0, model='LAMBERT',
                         diffuse=(0.8, 0.8, 0.8), specular_level=0.0),
                Material(name='Box', index=1, model='LAMBERT',
                         diffuse=(0.8, 0.3, 0.2), specular_level=0.0)]
        objs = [ObjectInfo(name=n_, index=i, location=(0, 0, 0),
                           matrix_world=np.eye(4, dtype=np.float32))
                for i, n_ in enumerate(('Floor', 'Box'))]
        cam = Camera(matrix_world=look_at_matrix((0.0, -6.5, 5.0),
                                                 (0.0, 0.0, 0.8)),
                     lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
        lamp = Light(type='SUN', name='Sun', direction=(-0.4, 0.3, -0.85),
                     energy=3.0, shadow='MAP')
        sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=[lamp],
                   camera=cam, world=World(color=(0, 0, 0),
                                           ambient=(0.05, 0.05, 0.05)),
                   settings=st)
        sc.images = {}
        sc.frame, sc.fps, sc.time = 1, 24, 1 / 24
        return sc
    a = scene(0.0)
    b = scene(np.radians(30.0))
    check('the premise: the turn keeps the vertex sum (and the count)',
          a.mesh.verts.shape == b.mesh.verts.shape
          and abs(float(a.mesh.verts.sum()) - float(b.mesh.verts.sum())) < 1e-3
          and not np.array_equal(a.mesh.verts, b.mesh.verts))
    check('the mesh key moves', GSH._mesh_key(a.mesh) != GSH._mesh_key(b.mesh))
    check('the shadow base key moves',
          LI._shadow_base_signature(a, st, None)
          != LI._shadow_base_signature(b, st, None))
    _clear_render_caches()
    with contextlib.redirect_stdout(io.StringIO()):
        f_a = R.render(a, st)
        warm = R.render(b, st)
        _clear_render_caches()
        cold = R.render(scene(np.radians(30.0)), st)
    eff = _px(f_a, cold)
    check('the second render after the turn IS a render with every cache '
          'cleared (bitwise), and the turn shows',
          np.array_equal(warm, cold) and eff[0] > 100, f'effect {eff[0]} px')
    _clear_render_caches()


# ------------------------------------------------------------ (I) neutrality


def test_i_an_unedited_scene_is_the_same_bits():
    """Keys only get stricter and a scene nobody edited is untouched: the
    same arrays out of the exporter with or without a space that hides
    nothing, the same frame from warm caches as from cold ones, and every
    cache still hit."""
    props, EN, EX, _compat, PV = _mods()
    a = _mesh_object('A', 21)
    m2 = np.eye(4, dtype=np.float32)
    m2[0, 3] = 3.0
    b = _mesh_object('B', 22, matrix=m2, factory=FB.glyph_mesh)
    lamp = _lamp_object()
    objs = [a, b, lamp]
    st = RenderSettings()
    plain = EX.export_scene(_depsgraph(props, objs), st, [])
    spaced = EX.export_scene(_depsgraph(props, objs), st, [],
                             space=NS(hidden=set()))
    names = ('verts', 'normals', 'uvs', 'colors', 'tris', 'mat_index',
             'obj_index', 'face_normals', 'smooth')
    check('an export for a viewport that hides nothing is the export with '
          'no space, array for array',
          all(np.array_equal(getattr(plain.mesh, n_), getattr(spaced.mesh, n_))
              for n_ in names)
          and [o.name for o in plain.objects] == [o.name for o in spaced.objects]
          and len(plain.lights) == len(spaced.lights) == 1)
    check('...and an ordinary scene bakes nothing of a lamp into its '
          'materials', plain.lamp_bound == (False, False))
    lo = EX.export_lights_into(plain, _depsgraph(props, objs))
    check('the lights-only export of the same lamps is the same lamps',
          lo is not None and len(lo.lights) == 1
          and vars(lo.lights[0]) is not vars(plain.lights[0])
          and all(np.array_equal(np.asarray(v), np.asarray(
              vars(plain.lights[0])[k])) for k, v in vars(lo.lights[0]).items()
              if isinstance(v, (int, float, str, tuple, bool))))
    check('...on the same frame the clock it writes is the parked one',
          (lo.frame, lo.fps, lo.time) == (plain.frame, plain.fps, plain.time))

    # ---- frames: warm equals cold, and the keys hit
    rs = RenderSettings()
    rs.resolution_x, rs.resolution_y = W, H
    rs.shadows = True
    rs.aa_mode = 'NONE'
    rs.output_scale = 'NONE'
    sc = demo_scene(rs, with_texture=False)
    _clear_render_caches()
    with contextlib.redirect_stdout(io.StringIO()):
        cold = R.render(sc, rs)
        k1 = GSH._mesh_key(sc.mesh)
        maps1 = [l.shadow_map for l in sc.lights]
        warm = R.render(sc, rs)
        maps2 = [l.shadow_map for l in sc.lights]
        again = demo_scene(rs, with_texture=False)
        k2 = GSH._mesh_key(again.mesh)
    check('a second render of the untouched scene is bitwise the first',
          np.array_equal(cold, warm))
    check('...its shadow maps came from the cache (the same objects)',
          all(m1 is m2_ for m1, m2_ in zip(maps1, maps2)))
    check('...and the same scene exported again (fresh arrays, the same '
          'bytes) has the same mesh key', k1 == k2)
    _clear_render_caches()


def main():
    utf8_console()
    fakebpy.install()
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
    print('all R254 viewport checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
