"""R254: the per-lamp contribution flags and light linking.

Every check below fails on the tree this round forked from and passes now
(the module runs against that tree too: the helpers fall back to what it
had), or pins something that must NOT have moved.

    (1) Ambient Only REMOVES the lamp: one predicate (LI.is_ambient_lamp)
        at every reader of the lamp list -- no diffuse, no highlight, no
        shadow, no beam / shaft / flare, no slot under the Light Limit,
        no scatter in a volume; what is left is flat ambient
    (2) the ambient pool on the GPU road is a VALUE (the ambient tail of
        the hal_lights row): an edit of an ambient lamp is a plan-cache
        hit with the right ambient
    (3) light linking resolves every instance of a linked object, the
        collection that holds the instancers, the hair of a linked
        object; 'Only' with nothing resolved lights nothing
    (4) volume containers, the visible beam, the shafts, the flare and
        the shaded halo honour the contribution flags and the Light Limit
    (5) the Light panel: what a style does on this frame, Diffuse Only +
        Specular Only, what Energy means on an Ambient Only lamp
    (6) the flags on the other roads: VERTEX / FACE shading, the console
        models, material light groups, Only Shadow with Negative

    python -m halcyon.tests.test_r254_flags
"""
import sys
import traceback
import types

import numpy as np

from . import utf8_console
from .test_render import base_settings, _vol_graph
from .scenebuild import cube, sphere, _mesh_concat, look_at_matrix
from ..core import lights as LI
from ..core import lines as LN
from ..core import raster as CRg
from ..core import render as R
from ..core import volume as VOL
from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
from ..gpu import material as GM
from ..gpu import shade as GSH

FAILS = []
f32 = np.float32
W, H = 96, 72
BAR = 6e-3
ALB = 0.7                       # the floor's albedo
KEY_POS = (-2.0, -3.0, 6.0)


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
    st = base_settings(W, H, shadows=True, transparency='NONE',
                       global_ambient=(0.0, 0.0, 0.0), fog=False)
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _grid(z=0.0, size=30.0, n=1, mat=0, obj=0):
    """A floor of n x n quads (n = 1 is scenebuild.plane's two triangles;
    a tessellated one gives the VERTEX and FACE rates corners inside a
    shadow)."""
    h = size * 0.5
    ax = np.linspace(-h, h, n + 1, dtype=f32)
    xx, yy = np.meshgrid(ax, ax)
    V = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, z, f32)], 1)
    N = np.tile(np.array([[0, 0, 1.0]], f32), (V.shape[0], 1))
    UV = (V[:, :2] / size + 0.5).astype(f32) * 4.0
    T = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            T += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return (V.astype(f32), N, UV, np.asarray(T, np.int32), mat, obj)


def _key(**kw):
    base = dict(type='POINT', name='Key', position=KEY_POS,
                color=(1.0, 1.0, 1.0), energy=800.0, shadow='MAP')
    base.update(kw)
    return Light(**base)


def _fill(**kw):
    base = dict(type='POINT', name='Fill', position=(4.0, -5.0, 5.0),
                color=(1.0, 1.0, 1.0), energy=400.0, shadow='NONE')
    base.update(kw)
    return Light(**base)


def _beam(**kw):
    """A spot whose visible beam crosses the frame."""
    base = dict(type='SPOT', name='Beam', position=(0.0, 4.5, 2.6),
                direction=(0.0, -1.0, -0.2), color=(1.0, 1.0, 1.0),
                energy=800.0, spot_size=0.7, spot_blend=0.2, volumetric=1.0,
                shadow='NONE')
    base.update(kw)
    return Light(**base)


def _shaft_sun(**kw):
    """A low Sun in front of the camera: its light shafts have a
    vanishing point (a Sun draws no beam, so it throws shafts whatever
    Volumetric Lights says)."""
    base = dict(type='SUN', name='ShaftSun', direction=(0.0, -0.8, -0.2),
                color=(1.0, 1.0, 1.0), energy=0.5, volumetric=0.5,
                shadow='NONE')
    base.update(kw)
    return Light(**base)


def _sun_shafts(sc, st, vp):
    """How many shaft sources the _shaft_sun lamps of a scene give (told
    from a beam lamp's by their Volumetric of exactly 0.5: a spot with a
    beam also threw shafts before the beam took its own unit)."""
    return len([1 for _xy, vol in R.shaft_sources(sc, st, vp)
                if vol == 0.5])


def _flare_lamp(**kw):
    """A point lamp in the middle of the frame, above the floor."""
    base = dict(type='POINT', name='Flare', position=(0.3, 0.8, 1.5),
                color=(1.0, 1.0, 1.0), energy=50.0, flare=0.8,
                shadow='NONE')
    base.update(kw)
    return Light(**base)


def _sun(**kw):
    base = dict(type='SUN', name='Sun', direction=(0.25, 0.35, -0.9),
                color=(1.0, 1.0, 1.0), energy=2.0, shadow='MAP')
    base.update(kw)
    return Light(**base)


def _scene(lights, st, tess=1, fog=None, mats=None):
    """A grey Lambert floor (object 0), a glossy Phong ball (1) and a
    Lambert slab held above the floor as the caster (2); `fog` adds a
    volume container cube (object 3, material 3) with that density.

    The ball's triangles are flagged smooth, as an export flags the
    polygons whose corner normals it wrote smooth. (Left flat with their
    shared vertex normals -- a pairing only a hand-built mesh has -- the
    per-vertex rate lights the ball's corners differently on the two
    roads, 0.13 with or without any flag: not this package's subject.)"""
    if isinstance(lights, Light):
        lights = [lights]
    materials = [
        Material(name='Floor', index=0, model='LAMBERT',
                 diffuse=(ALB, ALB, ALB), specular_level=0.0),
        Material(name='Ball', index=1, model='PHONG', diffuse=(0.3, 0.5, 0.2),
                 specular=(1.0, 1.0, 1.0), specular_level=1.0,
                 glossiness=40.0),
        Material(name='Slab', index=2, model='LAMBERT',
                 diffuse=(0.6, 0.2, 0.2), specular_level=0.0)]
    if mats:
        for i, m in mats.items():
            materials[i] = m
    parts = [_grid(n=tess),
             sphere(centre=(1.6, 0.0, 0.8), radius=0.8, mat=1, obj=1),
             cube(centre=(-0.6, -0.4, 2.2), size=1.2, mat=2, obj=2)]
    objs = [ObjectInfo(name='Floor', index=0, location=(0, 0, 0),
                       matrix_world=np.eye(4, dtype=f32)),
            ObjectInfo(name='Ball', index=1, location=(1.6, 0.0, 0.8),
                       matrix_world=np.eye(4, dtype=f32)),
            ObjectInfo(name='Slab', index=2, location=(-0.6, -0.4, 2.2),
                       matrix_world=np.eye(4, dtype=f32))]
    if fog is not None:
        materials.append(Material(name='Fog', index=3,
                                  graph=_vol_graph(Density=float(fog))))
        parts.append(cube(centre=(0.3, 0.3, 2.0), size=5.0, mat=3, obj=3))
        objs.append(ObjectInfo(name='FogBox', index=3,
                               location=(0.3, 0.3, 2.0),
                               matrix_world=np.eye(4, dtype=f32)))
    mesh = _mesh_concat(parts)
    mesh.smooth = np.asarray(mesh.obj_index) == 1
    cam = Camera(matrix_world=look_at_matrix((0.5, -8.0, 7.0),
                                             (0.3, 0.8, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    sc = Scene(mesh=mesh, materials=materials, objects=objs,
               lights=list(lights), camera=cam,
               world=World(color=(0.0, 0.0, 0.0), ambient=(0.0, 0.0, 0.0)),
               settings=st)
    sc.images = {}
    return sc


def _gbuf(sc, st):
    view, _p, vp, eye = R.camera_matrices(sc.camera, W, H)
    g = CRg.GBuffer(W, H)
    CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, W, H, gbuf=g,
                  depth_bits=st.depth_precision)
    return g, view, vp, eye


def _objects(sc, st):
    """The object index under every pixel (-1 where nothing is)."""
    g = _gbuf(sc, st)[0]
    tri = np.asarray(g.tri)
    return np.where(tri >= 0,
                    np.asarray(sc.mesh.obj_index)[np.clip(tri, 0, None)], -1)


def _cpu(sc, st, clear=True):
    if clear:
        LI.clear_shadow_cache()
    return np.asarray(R.render(sc, st))[..., :3].astype(np.float64)


def _twin(sc, st, keep=False, bvh=None):
    """The deferred pass through the GLSL simulator twin: (frame, passes,
    atlases, why). With `keep` the plan cache is left as it stands (what a
    second F12 or a viewport refine sees)."""
    R.render(sc, st)                      # the maps the plan packs
    g, view, _vp, eye = _gbuf(sc, st)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), bvh, view, eye,
                     W, H)
    if not keep:
        GSH._PLAN_CACHE.clear()
    p, why, atl = GSH.plan_frame(job, g)
    if p is None:
        return None, None, None, why
    out, hit = GSH.simulate(job, g, p, atl)
    if out is None:
        return None, p, atl, hit
    return np.asarray(out).astype(np.float64)[..., :3], p, atl, None


def _device(sc, st):
    """The whole GPU road (shade_frame, uploads, the CPU passes that
    follow it) through the fake device, which applies the driver's rules:
    a declared sampler must be bound, a declared uniform set."""
    from . import fakedevice
    st.render_device = 'GPU'
    with fakedevice.installed():
        img = np.asarray(R.render(sc, st))[..., :3].astype(np.float64)
    st.render_device = 'CPU'
    return img


def _roads(label, sc, st, cov=None, bar=BAR):
    """CPU frame and its twin with the two-road check at the deferred
    bar, over the covered pixels."""
    c = _cpu(sc, st)
    g, _p, _a, why = _twin(sc, st)
    if g is None:
        check(f'{label}: the deferred pass plans and simulates', False,
              str(why))
        return c, None
    if cov is None:
        cov = _objects(sc, st) >= 0
    d = float(np.abs(c[cov] - g[cov]).max())
    check(f'{label}: the GPU twin is the CPU frame within the deferred bar',
          d < bar, f'max |CPU - GPU| {d:.2e}')
    return c, g


def _mean(img, mask):
    return float(np.asarray(img)[mask].mean()) if mask.any() else float('nan')


_REG = {}


def _regions():
    """floor_lit / floor_core / ball / slab masks from the default key
    lamp with and without its shadow (ratios of renders, never the
    engine's own Shadow pass)."""
    if _REG:
        return _REG
    st = _settings()
    sc = _scene(_key(), st)
    obj = _objects(sc, st)
    a = _cpu(sc, st).mean(axis=2)
    b = _cpu(_scene(_key(shadow='NONE'), st), st).mean(axis=2)
    floor = obj == 0
    ratio = np.where(b > 1e-6, a / np.maximum(b, 1e-6), 1.0)
    _REG.update(floor=floor, ball=obj == 1, slab=obj == 2, cov=obj >= 0,
                floor_lit=floor & (ratio > 0.999) & (b > 0.5 * b[floor].max()),
                floor_core=floor & (ratio < 0.001) & (b > 1e-4))
    return _REG


def _frame_lights(sc, st):
    fn = getattr(LI, 'frame_lights', None)
    return fn(sc, st) if fn else LI.select_lights(sc.lights, st)


def _is_ambient(l):
    fn = getattr(LI, 'is_ambient_lamp', None)
    return fn(l) if fn else l.type == 'AMBIENT'


# ------------------------------------------- 0. what must not have moved


def test_a_a_scene_without_the_features_reads_what_it_read():
    """Default neutrality at the function level: without an ambient lamp,
    a light link or a flagged lamp every list is the scene's own lamps in
    their order, the light row carries no tail, the plan signature entry
    is None and the passes bake the ambient literal they always did
    (frames: docs-dev/r254/flags.md, the feature matrix bitwise against
    the fork point on both roads)."""
    st = _settings(spot_cones=True)
    lamps = [_key(), _fill(), _beam(), _shaft_sun(), _flare_lamp()]
    sc = _scene(lamps, st, fog=0.3)
    fl = _frame_lights(sc, st)
    check('the frame\'s lamps are the scene\'s own, in their order',
          len(fl) == len(lamps) and all(a is b for a, b in zip(fl, lamps)))
    if hasattr(VOL, 'scatter_lights'):
        sl = VOL.scatter_lights(sc, st)
        check('a volume scatters the same lamps in the same order, each '
              'with sign +1 (the march keys its random streams on the '
              'position)',
              [l for l, _s in sl] == [l for l in lamps]
              and all(s == 1.0 for _l, s in sl)
              and all(a is b for (a, _s), b in zip(sl, lamps)))
        check('the beam list is the Volumetric lamps of a kind that marches',
              [l.name for l in R._beam_lamps(sc, st)] == ['Beam'])
    g, _view, vp, _eye = _gbuf(_scene(_fill(), st), st)
    check('the shaft and flare sources still find the volumetric Sun and '
          'the flare lamp', _sun_shafts(sc, st, vp) == 1
          and len(R._flare_sources(sc, st, g, vp)) == 1,
          f'{_sun_shafts(sc, st, vp)} Sun shaft(s), '
          f'{len(R._flare_sources(sc, st, g, vp))} flare(s)')
    job = types.SimpleNamespace(settings=st, scene=sc)
    t0 = GM.pack_light_texels(fl)
    t1 = GM.pack_light_texels(fl, job)
    check('the light row carries no ambient tail (n x STRIDE texels, the '
          'bytes it had)',
          t1.shape == (1, len(fl) * GM.LIGHT_TEXEL_STRIDE, 4)
          and bool(np.array_equal(t0, t1)))
    if hasattr(GSH, '_ambient_sig'):
        check('the plan signature\'s ambient entry is None',
              GSH._ambient_sig(sc, st) is None)
    st2 = _settings(global_ambient=(0.2, 0.1, 0.05))
    sc2 = _scene([_key(), _fill()], st2)
    _img, passes, atl, why = _twin(sc2, st2)
    check('a scene with Global Ambient and no ambient lamp plans', passes
          is not None, str(why))
    if passes is not None:
        srcs = [s for _m, _n, s, _b in passes]
        lit = 'vec3 total = s.diffuse * (vec3(0.200000003, 0.100000001, ' \
              '0.0500000007)'
        tail = f'hal_ltex({2 * GM.LIGHT_TEXEL_STRIDE})'
        check('its passes bake the ambient literal they always did and '
              'read no tail texel',
              all(lit in s for s in srcs)
              and not any(tail in s for s in srcs),
              next((ln.strip() for s in srcs for ln in s.splitlines()
                    if 'vec3 total = s.diffuse' in ln), '')[:90])
        lt = atl.get('hal_lights')
        check('and the uploaded row is 2 x STRIDE texels', lt is not None
              and np.asarray(lt[1]()).shape
              == (1, 2 * GM.LIGHT_TEXEL_STRIDE, 4))
    if hasattr(GM, 'link_ladder'):
        check('a link to scattered objects emits the comparisons it always '
              'did', GM.link_ladder((1, 4)) ==
              '((abs(td.y - 1.0) < 0.5) ? 1.0 : 0.0) + '
              '((abs(td.y - 4.0) < 0.5) ? 1.0 : 0.0)',
              GM.link_ladder((1, 4)))


# ------------------------------------------- 1. Ambient Only removes the lamp


def test_b_ambient_only_is_only_ambient():
    """ambient-only-01: the flag used to ADD colour x Energy to the
    ambient pool and remove nothing -- a Sun kept its directional light
    and its shadow on top of the ambient."""
    rg = _regions()
    st = _settings(pass_diffuse=True, pass_specular=True, pass_shadow=True,
                   pass_ambient=True)

    def amb_sun(**kw):
        return _sun(name='Amb', energy=0.5, ambient_only=True,
                    direction=(-0.24, -0.34, -0.91), **kw)
    sc = _scene(amb_sun(), st)
    c = _cpu(sc, st)
    lit, core = _mean(c, rg['floor_lit']), _mean(c, rg['floor_core'])
    floor = c[rg['floor']]
    check('an Ambient Only Sun of strength 0.5 leaves a FLAT floor at '
          'albedo x 0.5 (it used to keep its sun light and its shadow on '
          'top)', abs(float(floor.max()) - ALB * 0.5) < 1e-6
          and abs(float(floor.min()) - ALB * 0.5) < 1e-6,
          f'floor min {floor.min():.4f} max {floor.max():.4f} '
          f'(lit {lit:.4f}, shadow core {core:.4f})')
    ps = getattr(sc, 'last_passes', None) or {}
    dif = np.asarray(ps.get('Diffuse', np.zeros(1)))[..., :3]
    spc = np.asarray(ps.get('Spec', np.zeros(1)))[..., :3]
    shd = np.asarray(ps.get('Shadow', np.ones(1)))[..., :3]
    check('its Diffuse and Specular passes are empty and its Shadow pass '
          'is 1 everywhere (no lamp shadows)',
          float(np.abs(dif).max()) == 0.0 and float(np.abs(spc).max()) == 0.0
          and float(shd[rg['cov']].min()) == 1.0,
          f'Diffuse max {float(np.abs(dif).max()):.4f}, Spec max '
          f'{float(np.abs(spc).max()):.4f}, Shadow min '
          f'{float(shd[rg["cov"]].min()):.2f}')
    amb = np.asarray(ps.get('Ambient', np.zeros((H, W, 3))))[..., :3]
    check('the Ambient pass is the pool: albedo x colour x Energy, the '
          'formula it was (Energy is a flat level, 1.0 = white)',
          abs(_mean(amb, rg['floor']) - ALB * 0.5) < 1e-6
          and bool(np.array_equal(LI.ambient_light(sc, st),
                                  np.asarray([0.5, 0.5, 0.5], f32))))
    st = _settings()
    # no position, no direction: moving the lamp changes no pixel
    for kind, a_kw, b_kw in (
            ('POINT', dict(position=(0.0, -2.0, 4.0)),
             dict(position=(2.5, 2.0, 1.5))),
            ('SUN', dict(direction=(0.2, 0.3, -1.0)),
             dict(direction=(-0.8, 0.1, -0.3))),
            ('SPOT', dict(position=(0.0, -2.0, 4.0), direction=(0, 0.3, -1),
                          spot_size=0.9),
             dict(position=(2.5, 2.0, 1.5), direction=(0.5, 0.0, -1),
                  spot_size=0.4))):
        for fog in (None, 0.35):
            fr = []
            for kw in (a_kw, b_kw):
                lt = Light(type=kind, name='Amb', energy=0.3, shadow='MAP',
                           ambient_only=True, **kw)
                fr.append(_cpu(_scene([_fill(), lt], st, fog=fog), st))
            d = float(np.abs(fr[0] - fr[1]).max())
            check(f'moving an Ambient Only {kind} lamp changes no pixel'
                  + (' in a scene with a volume container' if fog else ''),
                  d == 0.0, f'max change {d:.5f}')
    ga, _p, _a, why = _twin(_scene([_fill(), Light(
        type='POINT', name='Amb', energy=0.3, ambient_only=True,
        position=(0.0, -2.0, 4.0))], st), st)
    gb, _p, _a, _w = _twin(_scene([_fill(), Light(
        type='POINT', name='Amb', energy=0.3, ambient_only=True,
        position=(2.5, 2.0, 1.5))], st), st)
    check('...nor on the GPU twin', ga is not None and gb is not None
          and float(np.abs(ga - gb).max()) == 0.0, str(why))
    _roads('fill + an Ambient Only Sun', _scene([_fill(), amb_sun()], st), st)
    # every reader of the lamp list
    amb = Light(type='SPOT', name='Amb', position=(0.0, 4.5, 2.6),
                direction=(0.0, -1.0, -0.2), energy=0.3, spot_size=0.7,
                shadow='RAY', ambient_only=True, volumetric=1.0)
    a, b = _fill(name='A'), _fill(name='B', position=(-4.0, -5.0, 5.0))
    st_l = _settings(max_lights=2, light_limit_mode='FIRST', spot_cones=True,
                     shadow_default='PER_LIGHT')
    sc = _scene([amb, a, b], st_l, fog=0.3)
    kept = [l.name for l in _frame_lights(sc, st_l)]
    check('it takes no slot under the Light Limit (limit 2 over [ambient, '
          'A, B] keeps A and B)', kept == ['A', 'B'], str(kept))
    check('it is not the key lamp of the line', LN.key_light(sc) is a,
          getattr(LN.key_light(sc), 'name', None))
    check('set to Ray Traced under Per Light it asks for no BVH',
          not R._per_light_ray_shadows(sc, st_l))
    if hasattr(VOL, 'scatter_lights'):
        check('it scatters in no volume',
              [l.name for l, _s in VOL.scatter_lights(sc, st_l)]
              == ['A', 'B'])
    st_m = _settings(spot_cones=True)
    g, _view, vp, _eye = _gbuf(_scene(_fill(), st_m), st_m)
    for flag in (False, True):
        sc_m = _scene([_fill(), _beam(ambient_only=flag),
                       _shaft_sun(ambient_only=flag),
                       _flare_lamp(ambient_only=flag)], st_m)
        got = (bool(R._beams_active(sc_m, st_m)),
               _sun_shafts(sc_m, st_m, vp),
               len(R._flare_sources(sc_m, st_m, g, vp)))
        check('a beam spot, a shaft Sun and a flare lamp '
              + ('with Ambient Only: no beam, no shaft, no flare' if flag
                 else 'draw their beam, the Sun\'s shafts and one flare'),
              got == ((False, 0, 0) if flag else (True, 1, 1)), str(got))
    if hasattr(LI, 'limit_report'):
        rep = LI.limit_report(_scene([amb, a, b], _settings(max_lights=2)),
                              _settings(max_lights=2))
        check('and the Light Limit report does not count it as a lamp left '
              'unlit', rep is None, str(rep))
    # what Energy means, said where the render is read
    if hasattr(LI, 'ambient_report'):
        hot = LI.ambient_report(_scene([_fill(), Light(
            type='POINT', name='Amb', energy=800.0, ambient_only=True)], st))
        check('a lamp left at 800 W with Ambient Only is reported by name '
              'with its level and the unit (1.0 = white ambient)',
              hot is not None and "'Amb' at level 800" in hot
              and '1.0 = white ambient' in hot, str(hot))
        check('...and nothing is reported at a usable level, or without an '
              'ambient lamp',
              LI.ambient_report(_scene([_fill(), amb_sun()], st)) is None
              and LI.ambient_report(_scene([_fill(), _key()], st)) is None)
        from . import fakeblender as FB
        props, eng = FB.install()

        def rig(_eng, dg, bscene, flag=True):
            # (the stand-in's instances are a one-shot generator: a list,
            # so the exporter's lamp walk still finds the lamp)
            dg.object_instances = list(dg.object_instances)
            lamp_ob = [o for o in bscene.objects if o.type == 'LIGHT'][0]
            lamp_ob.data.type = 'POINT'
            lamp_ob.data.energy = 800.0
            lamp_ob.data.halcyon = FB.live(props.HalcyonLightSettings,
                                           ambient_only=flag)
        try:
            _i, _p, cap_e = FB.run_render(props, eng, rig=rig)
            said = [w for w in (cap_e.get('reports') or [])
                    if 'Ambient Only' in str(w)]
            _i, _p, cap_p = FB.run_render(
                props, eng, rig=lambda e, d, b: rig(e, d, b, False))
            quiet = [w for w in (cap_p.get('reports') or [])
                     if 'Ambient Only' in str(w)]
            check('the F12 says so in its report (and says nothing for the '
                  'same lamp without the flag)', len(said) >= 1 and not quiet,
                  str((said or ['no line'])[0])[:90])
        except Exception as exc:                                # noqa: BLE001
            traceback.print_exc()
            check('the F12 says so in its report', False, repr(exc)[:160])
    old_type = Light(type='AMBIENT', name='Old', color=(0.2, 0.4, 0.6),
                     energy=0.5)
    flagged = Light(type='POINT', name='New', color=(0.2, 0.4, 0.6),
                    energy=0.5, ambient_only=True)
    check('one predicate: the AMBIENT type and a lamp with Ambient Only '
          'are the same ambient light',
          _is_ambient(old_type) and _is_ambient(flagged)
          and not _is_ambient(_key())
          and bool(np.array_equal(_cpu(_scene([_fill(), old_type], st), st),
                                  _cpu(_scene([_fill(), flagged], st), st))))


# ------------------------------------- 2. the ambient pool on the GPU plan


def test_c_the_ambient_pool_is_a_value_on_the_gpu():
    """ambient-gpu-stale-01 = gpu-ambient-only-plan-stale: the plan baked
    the pool as a literal while its signature held nothing of the lamps
    that feed it -- a Power, colour or style edit of an ambient lamp was
    a cache hit that drew the old ambient (0.21 .. 0.33 off)."""
    rg = _regions()
    cov = rg['cov']
    st = _settings()
    amb = _sun(name='Amb', energy=0.1, ambient_only=True, shadow='NONE')
    sc = _scene([_fill(), amb], st)

    def pair(keep=True):
        c = _cpu(sc, st)
        g, p, atl, why = _twin(sc, st, keep=keep)
        return c, g, p, atl, why
    c0, g0, p0, atl0, why = pair(keep=False)
    check('a scene with an Ambient Only lamp plans', p0 is not None, str(why))
    if p0 is None:
        return
    tail = GM.ambient_tail_base(1) if hasattr(GM, 'ambient_tail_base') else 16
    srcs = [s for _m, _n, s, _b in p0]
    check('its passes read the pool from the ambient tail of the light row '
          '(no ambient literal)',
          all(f'vec3 total = s.diffuse * (hal_ltex({tail}).xyz' in s
              for s in srcs),
          next((ln.strip() for s in srcs for ln in s.splitlines()
                if 'vec3 total = s.diffuse' in ln), '')[:80])
    row = np.asarray(atl0['hal_lights'][1]())
    eng, wrld = LI.ambient_light_split(sc, st)
    check('the tail is the CPU\'s own float32 triples: the pool, its '
          'engine part, its world part',
          row.shape[1] == tail + 3
          and bool(np.array_equal(row[0, tail, :3], LI.ambient_light(sc, st)))
          and bool(np.array_equal(row[0, tail + 1, :3], eng))
          and bool(np.array_equal(row[0, tail + 2, :3], wrld)),
          f'row {row.shape}')
    check('the first frame matches', float(np.abs(c0[cov] - g0[cov]).max())
          < BAR)

    def step(label, edit, want_hit):
        edit()
        c, g, p, atl, why_ = pair()
        if g is None:
            check(f'{label}: plans', False, str(why_))
            return
        hit = p is p0
        d = float(np.abs(c[cov] - g[cov]).max())
        check(f'{label}: a plan-cache {"HIT" if want_hit else "MISS"} and '
              'the twin is the CPU frame', hit == want_hit and d < BAR,
              f'{"hit" if hit else "miss"}, max |CPU - GPU| {d:.2e}, CPU '
              f'floor {_mean(c, rg["floor"]):.4f}')
    step('Energy 0.1 -> 0.4', lambda: setattr(amb, 'energy', 0.4), True)
    step('colour -> (0.2, 0.6, 1.0)',
         lambda: setattr(amb, 'color', (0.2, 0.6, 1.0)), True)

    def strobe():
        amb.light_style = 'FAST_STROBE'
        sc.time = 0.1
    step('FAST_STROBE at 0.1 s (gain 0)', strobe, True)
    step('the next style step (0.2 s, gain 1)',
         lambda: setattr(sc, 'time', 0.2), True)
    step('Ambient Only off (the lamp joins the light loop: structure)',
         lambda: setattr(amb, 'ambient_only', False), False)
    # the whole road through the fake device, warm cache, an edit between
    st_d = _settings()
    amb_d = Light(type='POINT', name='Amb', energy=0.15, color=(1.0, 0.8, 0.6),
                  ambient_only=True, position=(0, 0, 3))
    sc_d = _scene([_key(shadow='NONE'), amb_d], st_d)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    from . import fakedevice
    st_d.render_device = 'GPU'
    with fakedevice.installed():
        d1 = np.asarray(R.render(sc_d, st_d))[..., :3].astype(np.float64)
        n_plans = len(GSH._PLAN_CACHE)
        amb_d.energy = 0.45
        d2 = np.asarray(R.render(sc_d, st_d))[..., :3].astype(np.float64)
        n_plans2 = len(GSH._PLAN_CACHE)
    st_d.render_device = 'CPU'
    c2 = _cpu(sc_d, st_d)
    e2 = float(np.abs(d2 - c2).max())
    check('through the device road: the second frame on a warm plan cache '
          'after a Power edit is the CPU frame, with no new plan',
          e2 < BAR and n_plans2 == n_plans
          and float(np.abs(d2 - d1).max()) > 0.1,
          f'max |CPU - GPU| {e2:.2e}; plans {n_plans} -> {n_plans2}; the '
          f'frame moved by {float(np.abs(d2 - d1).max()):.3f}')
    # no lamp left in the loop: the pass still declares and binds the row
    st_o = _settings()
    only = Light(type='POINT', name='Amb', energy=0.3, ambient_only=True)
    sc_o = _scene([only], st_o)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    try:
        d_o = _device(sc_o, st_o)
        c_o = _cpu(sc_o, st_o)
        e_o = float(np.abs(d_o - c_o).max())
        check('a scene lit by an ambient lamp ALONE shades on the device '
              'road (the row is declared and bound with no lamp in the '
              'loop)', e_o < BAR and abs(_mean(c_o, rg['floor']) - ALB * 0.3)
              < 1e-6, f'max |CPU - GPU| {e_o:.2e}')
    except Exception as exc:                                    # noqa: BLE001
        check('a scene lit by an ambient lamp ALONE shades on the device '
              'road', False, repr(exc)[:160])
    # Blender Internal's flat world ambient reads the pool's two halves
    st_b = _settings(global_ambient=(0.1, 0.05, 0.02))
    amb_b = Light(type='POINT', name='Amb', energy=0.2, ambient_only=True)
    sc_b = _scene([_fill(), amb_b], st_b, mats={0: Material(
        name='FloorBI', index=0, graph=_bi_graph({}))})
    cb, gb, pb, _ab, why_b = (_cpu(sc_b, st_b),) + _twin(sc_b, st_b)
    if gb is None:
        check('a Blender Internal material beside an ambient lamp plans',
              False, str(why_b))
    else:
        t2 = GM.ambient_tail_base(1) if hasattr(GM, 'ambient_tail_base') \
            else 16
        check('a Blender Internal material reads the engine and the world '
              'half from the tail and matches',
              float(np.abs(cb[cov] - gb[cov]).max()) < BAR
              and any(f'hal_ltex({t2 + 1}).xyz' in s
                      and f'hal_ltex({t2 + 2}).xyz' in s
                      for _m, _n, s, _b in pb),
              f'max |CPU - GPU| {float(np.abs(cb[cov] - gb[cov]).max()):.2e}')
        amb_b.energy = 0.6
        cb2 = _cpu(sc_b, st_b)
        gb2, pb2, _a2, _w2 = _twin(sc_b, st_b, keep=True)
        check('...and after a Power edit on the warm cache',
              gb2 is not None and pb2 is pb
              and float(np.abs(cb2[cov] - gb2[cov]).max()) < BAR,
              f'max |CPU - GPU| '
              f'{float(np.abs(cb2[cov] - gb2[cov]).max()) if gb2 is not None else -1:.2e}')
    # the radiosity gather still bakes its sky term: the value is in the
    # signature there, so an edit is a new plan (and the right picture)
    if hasattr(GSH, '_ambient_sig'):
        st_r = _settings(radiosity=True)
        amb_r = Light(type='POINT', name='Amb', energy=0.2, ambient_only=True)
        sc_r = _scene([_fill(), amb_r], st_r)
        s1 = GSH._ambient_sig(sc_r, st_r)
        amb_r.energy = 0.5
        s2 = GSH._ambient_sig(sc_r, st_r)
        st_n = _settings()
        s3 = GSH._ambient_sig(sc_r, st_n)
        amb_r.energy = 0.9
        check('under Radiosity (whose gather bakes the pool) the pool\'s '
              'value is in the plan signature; without it only the flag is',
              s1 != s2 and s3 == GSH._ambient_sig(sc_r, st_n)
              and s3 is not None, f'{s1} / {s2} / {s3}')


# ------------------------------------------------- 3. light linking resolves


def _row(name, instancer='', link_name=''):
    o = ObjectInfo(name=name, matrix_world=np.eye(4, dtype=f32))
    o.instancer = instancer
    o.link_name = link_name
    return o


def _resolve(lights, objects):
    """export._resolve_links; on a tree without it, the statements the
    two export sites ran inline."""
    from . import fakeblender as FB
    FB.install()
    from .. import export as EX
    fn = getattr(EX, '_resolve_links', None)
    if fn is not None:
        return fn(lights, objects)
    name_to_index = {o.name: i for i, o in enumerate(objects)}
    for lt in lights:
        names = getattr(lt, 'exclude_names', None)
        if names:
            lt.exclude_objects = tuple(sorted(
                name_to_index[n] for n in names if n in name_to_index))


def _linked(names, mode='EXCLUDE'):
    lt = Light(type='POINT', name='L')
    lt.exclude_mode = mode
    lt.exclude_names = None if names is None else set(names)
    return lt


def _fake_export(kinds):
    """export_scene (and the viewport's export_lights_into) over a fake
    depsgraph: a ground, three instances of one mesh generated by three
    instancers, and one lamp per entry of `kinds` = [(collection object
    names, mode)]. Returns (scene, lights-only scene)."""
    from . import fakeblender as FB
    from ..core.settings import RenderSettings
    props, _eng = FB.install()
    from .. import export as EX

    def obj(name, at=(0.0, 0.0, 0.0)):
        m = np.eye(4, dtype=f32)
        m[:3, 3] = at
        return FB.geometry_object(name, FB.cube_mesh, matrix=m, kind='MESH')
    ground = obj('Ground')
    gem = obj('Gem')
    by_name = {'Ground': ground, 'Gem': gem}
    insts = [types.SimpleNamespace(object=ground,
                                   matrix_world=FB._Mat(ground.matrix_world),
                                   show_self=True, is_instance=False,
                                   parent=None)]
    for i in range(3):
        empty = types.SimpleNamespace(name=f'Inst{i}', name_full=f'Inst{i}',
                                      type='EMPTY')
        by_name[f'Inst{i}'] = empty
        m = np.eye(4, dtype=f32)
        m[:3, 3] = (3.0 * i, 0.0, 0.0)
        insts.append(types.SimpleNamespace(
            object=gem, matrix_world=FB._Mat(m), show_self=True,
            is_instance=True, parent=empty))
    for j, (names, mode) in enumerate(kinds):
        hs = FB.live(props.HalcyonLightSettings)
        hs.exclude_mode = mode
        hs.exclude_collection = None if names is None else \
            types.SimpleNamespace(all_objects=[by_name[n] for n in names])
        lob = types.SimpleNamespace(
            type='LIGHT', name=f'Lamp{j}',
            data=types.SimpleNamespace(
                type='POINT', color=(1.0, 1.0, 1.0), energy=100.0,
                halcyon=hs, shadow_soft_size=0.0, use_shadow=True))
        m = np.eye(4, dtype=f32)
        m[:3, 3] = (0.0, 0.0, 5.0)
        insts.append(types.SimpleNamespace(
            object=lob, matrix_world=FB._Mat(m), show_self=True,
            is_instance=False, parent=None))
    dg = FB.depsgraph_of(props, [ground, gem])
    dg.object_instances = insts
    warn = []
    sc = EX.export_scene(dg, RenderSettings(), warn)
    dg2 = FB.depsgraph_of(props, [ground, gem])
    dg2.object_instances = insts
    lights_only = EX.export_lights_into(sc, dg2)
    return sc, lights_only


def test_d_light_links_resolve_every_instance():
    """link-instances-01 and link-only-empty-01: the resolver kept one
    index per name (of three instances only the last was affected, a
    collection of instancer empties reached nothing) and wrote () for an
    'Only' link that resolved to nothing, which both roads read as no
    link at all."""
    rows = [_row('Ground'), _row('Gem', 'Inst0'), _row('Gem', 'Inst1'),
            _row('Gem', 'Inst2'), _row('Box')]

    def res(names, mode='EXCLUDE', objects=rows):
        lt = _linked(names, mode)
        _resolve([lt], objects)
        return tuple(lt.exclude_objects)
    check('a link to the instanced object reaches all three instances',
          res({'Gem'}) == (1, 2, 3), str(res({'Gem'})))
    check('a link to the collection holding the instancer empties reaches '
          'all three', res({'Inst0', 'Inst1', 'Inst2'}) == (1, 2, 3),
          str(res({'Inst0', 'Inst1', 'Inst2'})))
    check('a link to one instancer reaches its instance alone',
          res({'Inst1'}) == (2,), str(res({'Inst1'})))
    check('names and instancers join (an object and somebody else\'s '
          'instancer)', res({'Box', 'Inst0'}) == (1, 4),
          str(res({'Box', 'Inst0'})))
    check("'Only' with a collection that resolves to nothing writes (-1,): "
          'no object carries that index',
          res(set(), 'ONLY') == (-1,) and res({'Gone'}, 'ONLY') == (-1,),
          f"{res(set(), 'ONLY')} {res({'Gone'}, 'ONLY')}")
    check("'Exclude' with nothing resolved stays unlinked, and a lamp "
          'without a collection is untouched',
          res(set(), 'EXCLUDE') == () and res(None, 'ONLY') == ())
    clash = [_row('Rock', link_name='Rock [props.blend]'),
             _row('Rock', link_name='Rock')]
    check('a library object does not answer for a local object of the same '
          'name (the link name is library-qualified)',
          res({'Rock'}, objects=clash) == (1,)
          and res({'Rock [props.blend]'}, objects=clash) == (0,),
          f"{res({'Rock'}, objects=clash)}")
    check('a row built without a link name (a hand-built scene) matches on '
          'its name', res({'Box'}) == (4,))
    # through the exporter and the viewport's lights-only export
    try:
        sc, lo = _fake_export([(['Gem'], 'EXCLUDE'),
                               (['Inst0', 'Inst1', 'Inst2'], 'EXCLUDE'),
                               (['Inst2'], 'ONLY'), ([], 'ONLY'),
                               ([], 'EXCLUDE'), (None, 'ONLY')])
        want = [(1, 2, 3), (1, 2, 3), (3,), (-1,), (), ()]
        got = [tuple(l.exclude_objects) for l in sc.lights]
        check('export_scene: three instances of one mesh export as three '
              'rows that carry their instancer',
              [(o.name, getattr(o, 'instancer', None)) for o in sc.objects]
              == [('Ground', ''), ('Gem', 'Inst0'), ('Gem', 'Inst1'),
                  ('Gem', 'Inst2')],
              str([(o.name, getattr(o, 'instancer', None))
                   for o in sc.objects]))
        check('export_scene resolves the six links (object, instancers, one '
              "instancer under Only, empty Only, empty Exclude, no link)",
              got == want, str(got))
        got2 = [tuple(l.exclude_objects) for l in lo.lights]
        check('the viewport\'s lights-only export resolves them the same '
              '(one resolver for both sites)', got2 == want, str(got2))
    except Exception as exc:                                    # noqa: BLE001
        traceback.print_exc()
        check('export_scene over a fake depsgraph of instances', False,
              repr(exc)[:200])
    # the renderer: (-1,) lights nothing, a run of indices is one term
    rg = _regions()
    st = _settings()
    c, g = _roads("'Only' (-1,)", _scene(_key(exclude_objects=(-1,),
                                              exclude_mode='ONLY'), st), st)
    check("an 'Only' lamp whose collection is empty lights nothing",
          float(np.abs(c).max()) == 0.0
          and (g is None or float(np.abs(g).max()) == 0.0),
          f'CPU max {float(np.abs(c).max()):.4f}')
    c_un = _cpu(_scene(_key(), st), st)
    c_ex, _g = _roads("'Exclude' (-1,)", _scene(_key(
        exclude_objects=(-1,), exclude_mode='EXCLUDE'), st), st)
    check("...and under 'Exclude' the same list changes nothing",
          bool(np.array_equal(c_ex, c_un)))
    sc_run = _scene(_key(exclude_objects=(1, 2), exclude_mode='EXCLUDE'), st)
    c_run, g_run = _roads('a link to two neighbouring objects', sc_run, st)
    check('the ball and the slab are unlit, the floor keeps its light',
          _mean(c_run, rg['ball']) == 0.0 and _mean(c_run, rg['slab']) == 0.0
          and abs(_mean(c_run, rg['floor_lit'])
                  - _mean(c_un, rg['floor_lit'])) < 1e-9)
    if hasattr(GM, 'link_ladder'):
        _i, passes, _a, _w = _twin(sc_run, st)
        check('consecutive indices are ONE range test in the pass source',
              GM.link_runs((5, 1, 2, 3, 9)) == [(1, 3), (5, 5), (9, 9)]
              and GM.link_ladder((1, 2))
              == '((td.y > 0.5 && td.y < 2.5) ? 1.0 : 0.0)'
              and passes is not None
              and any('(td.y > 0.5 && td.y < 2.5)' in s
                      for _m, _n, s, _b in passes), GM.link_ladder((1, 2)))
    sc_many = _scene(_key(exclude_objects=tuple(range(1, 301)),
                          exclude_mode='EXCLUDE'), st)
    c_m = _cpu(sc_many, st)
    g_m, _p, _a, why = _twin(sc_many, st)
    check('a link of 300 instances side by side stays on the GPU (it '
          'refused above 64 objects) and matches',
          g_m is not None and float(np.abs(
              c_m[rg['cov']] - g_m[rg['cov']]).max()) < BAR, str(why))


# -------------------------- 4. volumes, beams, halos: flags and the limit


def _fog_scene(lights, st):
    """The volume container alone against a black world: every pixel of
    its silhouette shows scatter and nothing else (object 0 = the box)."""
    mats = [Material(name='Fog', index=0, graph=_vol_graph(Density=0.35))]
    mesh = _mesh_concat([cube(centre=(0.3, 0.3, 2.0), size=5.0, mat=0,
                              obj=0)])
    objs = [ObjectInfo(name='FogBox', index=0, location=(0.3, 0.3, 2.0),
                       matrix_world=np.eye(4, dtype=f32))]
    cam = Camera(matrix_world=look_at_matrix((0.5, -8.0, 7.0),
                                             (0.3, 0.8, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=200.0)
    sc = Scene(mesh=mesh, materials=mats, objects=objs, lights=list(lights),
               camera=cam, world=World(color=(0.0, 0.0, 0.0),
                                       ambient=(0.0, 0.0, 0.0)),
               settings=st)
    sc.images = {}
    return sc


def test_e_media_honour_the_flags_and_the_limit():
    """flags-volume-beam-01: the volume march took every lamp that was
    not of type AMBIENT and the beam every volumetric lamp, flags and
    Light Limit unread -- a Negative lamp brightened the fog, an Only
    Shadow lamp lit it, a lamp the limit had dropped still lit it."""
    st = _settings(shadows=False)
    pos = (-1.0, -1.5, 5.0)

    def lamp(**kw):
        d = dict(type='POINT', name='L', position=pos, energy=1500.0,
                 shadow='NONE')
        d.update(kw)
        return Light(**d)

    def fog(lights, **kw):
        s = _settings(shadows=False, **kw)
        return _cpu(_fog_scene(lights, s), s)
    plain = fog([lamp()])
    reg = plain.sum(axis=2) > 1e-4
    x = _mean(plain, reg)
    check('a plain lamp lights the fog', x > 0.05 and int(reg.sum()) > 500,
          f'+{x:.4f} over {int(reg.sum())} px')
    neg = fog([lamp(negative=True)])
    check('a Negative lamp takes that light away (it added it)',
          float(np.abs(neg + plain).max()) < 1e-6,
          f'{_mean(neg, reg):+.4f} against the plain lamp\'s {x:+.4f}')
    for tag, kw in (('Only Shadow', dict(only_shadow=True)),
                    ('Specular Only', dict(specular_only=True)),
                    ('Diffuse Only + Specular Only',
                     dict(diffuse_only=True, specular_only=True)),
                    ('Diffuse off (affect_diffuse False)',
                     dict(affect_diffuse=False))):
        f = fog([lamp(**kw)])
        check(f'{tag}: the lamp lights no fog',
              float(np.abs(f).max()) == 0.0, f'{_mean(f, reg):+.4f}')
    f = fog([lamp(diffuse_only=True)])
    check('Diffuse Only alone lights it as the plain lamp does',
          bool(np.array_equal(f, plain)))
    f = fog([Light(type='POINT', name='Z', position=(0, 0, 9), energy=0.0,
                   shadow='NONE'), lamp()], max_lights=1,
            light_limit_mode='FIRST')
    check('a lamp the Light Limit dropped lights no fog',
          float(np.abs(f).max()) == 0.0, f'{_mean(f, reg):+.4f}')
    two = [lamp(name='A', energy=900.0), lamp(name='B', position=(2, 2, 4))]
    check('a fog scene over the limit is the scene without the dropped lamp',
          bool(np.array_equal(
              fog(two, max_lights=1, light_limit_mode='FIRST'),
              fog(two[:1]))))
    # light linking reaches the container (object 0 here)
    for tag, kw, lit in (
            ("'Exclude' the container", dict(exclude_objects=(0,)), False),
            ("'Only' another object", dict(exclude_objects=(5,),
                                           exclude_mode='ONLY'), False),
            ("'Only' the container", dict(exclude_objects=(0,),
                                          exclude_mode='ONLY'), True),
            ("'Only' nothing", dict(exclude_objects=(-1,),
                                    exclude_mode='ONLY'), False)):
        f = fog([lamp(**kw)])
        check(f'light linking, {tag}: the fog is '
              + ('lit' if lit else 'not lit'),
              bool(np.array_equal(f, plain)) if lit
              else float(np.abs(f).max()) == 0.0, f'{_mean(f, reg):+.4f}')
    # with the lamp's shadow on: an Only Shadow lamp still gives no light
    st_s = _settings()
    f_os = _cpu(_scene([_fill(), _key(only_shadow=True)], st_s, fog=0.3),
                st_s)
    f_no = _cpu(_scene([_fill()], st_s, fog=0.3), st_s)
    obj = _objects(_scene([_fill()], st_s, fog=0.3), st_s)
    sky = obj == 3
    f_k = _cpu(_scene([_fill(), _key()], st_s, fog=0.3), st_s)
    check('in a full scene the fog reads an Only Shadow lamp as no lamp '
          '(the floor under it keeps the shadow the lamp subtracts)',
          float((f_os - f_no)[sky].max()) <= 1e-9
          and float((f_k - f_no)[sky].max()) > 0.02,
          f'Only Shadow adds at most {float((f_os - f_no)[sky].max()):+.5f} '
          f'in the box; the plain key adds up to '
          f'{float((f_k - f_no)[sky].max()):+.4f}')
    # the device road draws the same fog (the march is one CPU pass)
    st_g = _settings()
    sc_g = _scene([_fill(), _key(negative=True, energy=300.0)], st_g, fog=0.3)
    R._GBUF_CACHE.clear()
    GSH._PLAN_CACHE.clear()
    try:
        d_g = _device(sc_g, st_g)
        c_g = _cpu(sc_g, st_g)
        e = float(np.abs(d_g - c_g).max())
        check('a Negative lamp in fog: the device road is the CPU frame',
              e < BAR, f'max |CPU - GPU| {e:.2e}')
    except Exception as exc:                                    # noqa: BLE001
        check('a Negative lamp in fog: the device road is the CPU frame',
              False, repr(exc)[:160])
    # the lamp's own visible beam
    st_b = _settings(shadows=False, spot_cones=True)

    def beam(first=(), limit=None, **kw):
        s = _settings(shadows=False, spot_cones=True)
        if limit:
            s.max_lights, s.light_limit_mode = limit, 'FIRST'
        a = _cpu(_scene(list(first) + [_beam(**kw)], s), s)
        b = _cpu(_scene(list(first) + [_beam(volumetric=0.0, **kw)], s), s)
        return a - b                    # what the Volumetric dial adds
    b0 = beam()
    regb = b0.sum(axis=2) > 1e-7
    xb = _mean(b0, regb)
    check('a plain spot draws its beam', xb > 0.0 and int(regb.sum()) > 300,
          f'+{xb:.5f} over {int(regb.sum())} px')
    bn = beam(negative=True)
    check('a Negative lamp\'s beam subtracts (it added)',
          abs(_mean(bn, regb) + xb) < 1e-5 * max(xb, 1.0)
          and _mean(bn, regb) < 0.0, f'{_mean(bn, regb):+.5f}')
    for tag, kw in (('Only Shadow', dict(only_shadow=True)),
                    ('Ambient Only', dict(ambient_only=True))):
        bb = beam(**kw)
        check(f'{tag}: no beam', float(np.abs(bb).max()) == 0.0,
              f'{_mean(bb, regb):+.5f}')
    bb = beam(first=[Light(type='POINT', name='Z', position=(0, 0, 9),
                           energy=0.0, shadow='NONE')], limit=1)
    check('a lamp the Light Limit dropped draws no beam',
          float(np.abs(bb).max()) == 0.0, f'{_mean(bb, regb):+.5f}')
    bb = beam(specular_only=True)
    check('Specular Only leaves the beam alone (it is neither lobe; the '
          'tooltip says so)', abs(_mean(bb, regb) - xb) < 1e-6 * max(xb, 1.0),
          f'{_mean(bb, regb):+.5f}')
    del st_b
    # shafts and flares
    st_m = _settings(spot_cones=False)
    g, _view, vp, _eye = _gbuf(_scene(_fill(), st_m), st_m)

    def sources(first=(), limit=None, **kw):
        s = _settings(spot_cones=False)
        if limit:
            s.max_lights, s.light_limit_mode = limit, 'FIRST'
        sc = _scene(list(first) + [_shaft_sun(**kw), _flare_lamp(**kw)], s)
        return (_sun_shafts(sc, s, vp), len(R._flare_sources(sc, s, g, vp)))
    zero = Light(type='POINT', name='Z', position=(0, 0, 9), energy=0.0,
                 shadow='NONE')
    got = {'plain': sources(), 'Only Shadow': sources(only_shadow=True),
           'Negative': sources(negative=True),
           'dropped': sources(first=[zero], limit=1)}
    check('shafts and flares: a plain and a Negative lamp have theirs, an '
          'Only Shadow lamp and one the Light Limit dropped have none',
          got == {'plain': (1, 1), 'Only Shadow': (0, 0),
                  'Negative': (1, 1), 'dropped': (0, 0)}, str(got))
    # the shaded halo is a diffuse receiver
    st_h = _settings(shadows=False)

    def halo(lights, **kw):
        s = _settings(shadows=False, **kw)
        sc = _scene(lights, s)
        spec = {'size': 0.6, 'hardness': 50, 'add': 0.5, 'alpha': 1.0,
                'color': (1.0, 0.7, 0.3), 'seed': 3, 'rings': 0, 'lines': 0,
                'star_points': 0, 'xalpha': False, 'soft': False,
                'shaded': True}
        m = Material(name='G', index=len(sc.materials), halo=spec)
        sc.materials.append(m)
        sc.halos = [{'mat': m.index,
                     'pos': np.array([[0.0, 0.6, 2.6]], f32)}]
        with_h = _cpu(sc, s)
        sc.halos = None
        return with_h - _cpu(sc, s)     # what the halo draws
    hl = lamp(position=(0.0, 0.6, 5.0), energy=400.0)
    h0 = halo([hl])
    regh = np.abs(h0).sum(axis=2) > 1e-4
    xh = _mean(h0, regh)
    check('a shaded halo takes the lamp\'s light', xh > 0.01
          and int(regh.sum()) > 20, f'+{xh:.4f} over {int(regh.sum())} px')
    two_h = lamp(name='M', position=(0.0, 0.6, 5.0), energy=400.0)
    for tag, lights, kw, want in (
            ('Only Shadow', [lamp(position=(0.0, 0.6, 5.0), energy=400.0,
                                  only_shadow=True)], {}, 0.0),
            ('Specular Only', [lamp(position=(0.0, 0.6, 5.0), energy=400.0,
                                    specular_only=True)], {}, 0.0),
            ('Ambient Only at level 0', [lamp(position=(0.0, 0.6, 5.0),
                                              energy=0.0, ambient_only=True)],
             {}, 0.0),
            ('dropped by the Light Limit', [zero, hl],
             dict(max_lights=1, light_limit_mode='FIRST'), 0.0),
            ('a plain lamp and its Negative twin', [hl, lamp(
                name='N', position=(0.0, 0.6, 5.0), energy=400.0,
                negative=True)], {}, 0.0),
            ('two plain lamps', [hl, two_h], {}, 2.0)):
        hh = halo(lights, **kw)
        got_h = _mean(hh, regh)
        check(f'the shaded halo under {tag}: {want:g} x the lamp',
              abs(got_h - want * xh) < 0.02 * xh, f'{got_h:+.4f} (the plain '
              f'lamp gives {xh:+.4f})')
    del st_h
    # the Light Meter node reads what a lamp gives a diffuse surface
    from ..core import nodeeval as NE

    def meter(lights, time=0.0):
        s = _settings()
        sc = _scene(lights, s)
        sc.time = time
        LI.apply_light_styles(sc)
        P = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.0]], f32)
        N = np.tile(np.array([[0.0, 0.0, 1.0]], f32), (2, 1))
        ev = types.SimpleNamespace(
            ctx=types.SimpleNamespace(scene=sc, settings=s, n=2, N=N, P=P,
                                      bvh=None),
            input=lambda node, name, kind: np.ones(2, f32))
        out = NE.n_u16_light_meter(ev, {
            'bl_idname': 'HALCYON_LightMeterNode', 'props': {},
            'inputs': [{'name': 'Exposure', 'default': 1.0}]})
        return np.asarray(out['Color'])[:, :3].astype(np.float64)
    m0 = meter([lamp()])
    got_m = {
        'negative': meter([lamp(negative=True)]),
        'only shadow': meter([lamp(only_shadow=True)]),
        'specular only': meter([lamp(specular_only=True)]),
        'ambient only': meter([lamp(ambient_only=True)]),
        'dark style step': meter([lamp(light_style='FAST_STROBE')], 0.1)}
    check('the Light Meter node: a Negative lamp reads minus the plain lamp; '
          'an Only Shadow, a Specular Only and an Ambient Only lamp read '
          'nothing; a strobe on its dark step reads nothing',
          float(m0.min()) > 0.0
          and float(np.abs(got_m['negative'] + m0).max()) < 1e-6
          and all(float(np.abs(got_m[k]).max()) == 0.0
                  for k in ('only shadow', 'specular only', 'ambient only',
                            'dark style step')),
          f'plain {m0[0, 0]:.3f}; '
          + ', '.join(f'{k} {v[0, 0]:+.3f}' for k, v in got_m.items()))


# ----------------------------------------------------------- 5. the panel


class _Layout:
    """A recording layout: (kind, name / text, text / icon, alert)."""

    def __init__(self, rec):
        self.rec = rec
        self.active = True
        self.alert = False
        self.use_property_split = False

    def prop(self, data, name, **kw):
        self.rec.append(('prop', name, kw.get('text'), self))

    def column(self, **kw):
        return _Layout(self.rec)

    row = column
    box = column

    def separator(self, **kw):
        pass

    def label(self, **kw):
        self.rec.append(('label', str(kw.get('text', '')), kw.get('icon'),
                         self))

    def template_ID(self, data, name, **kw):
        self.rec.append(('prop', name, None, self))

    def operator(self, idn, **kw):
        self.rec.append(('op', idn, None, self))
        return types.SimpleNamespace(scope=None)


def _draw_light(UI, defaults, kind='POINT', scene=None, energy=10.0, **fields):
    hs = types.SimpleNamespace(**defaults)
    hs.cookie = None
    for k, v in fields.items():
        setattr(hs, k, v)
    light = types.SimpleNamespace(type=kind, color=(1, 1, 1), energy=energy,
                                  halcyon=hs)
    rec = []
    panel = UI.HALCYON_PT_light()
    panel.layout = _Layout(rec)
    ctx = types.SimpleNamespace(engine=UI.ENGINE, light=light, scene=scene,
                                object=None)
    UI.HALCYON_PT_light.draw(panel, ctx)
    return rec


def _labels(rec):
    return [r[1] for r in rec if r[0] == 'label']


def test_f_the_panel_says_what_the_flags_do():
    """style-still-ui-01, both-only-ui-01 and what Energy means on an
    Ambient Only lamp: three notes, no pixel changed."""
    from . import fakeblender as FB
    props, _eng = FB.install()
    from .. import ui as UI
    from .. import export as EX
    from ..core.settings import RenderSettings
    ann = props.HalcyonLightSettings.__annotations__
    defaults = {k: (tuple(v.kw.get('default')) if isinstance(
        v.kw.get('default'), (tuple, list)) else v.kw.get('default'))
        for k, v in ann.items()}

    def bscene(frame=1, fps=24, base=1.0, sub=0.0):
        return types.SimpleNamespace(
            frame_current=frame, frame_subframe=sub,
            render=types.SimpleNamespace(fps=fps, fps_base=base),
            halcyon=types.SimpleNamespace(
                light_falloff_default='INVERSE_SQUARE'))
    rec = _draw_light(UI, defaults, scene=bscene())
    check('a steady lamp draws no style line, no warning and no ambient '
          'note', not any('Animated' in t or 'lights no surface' in t
                          or 'ambient' in t.lower() for t in _labels(rec)),
          str([t for t in _labels(rec)][:4]))
    # the style on this frame
    for style, frame, dark in (('SLOW_STROBE', 1, True),
                               ('SLOW_STRONG_PULSE', 1, True),
                               ('FAST_STROBE', 1, False),
                               ('SLOW_STROBE', 24, False),
                               ('GENTLE_PULSE', 1, False)):
        sc_b = bscene(frame)
        rec = _draw_light(UI, defaults, scene=sc_b, light_style=style)
        line = [r for r in rec if r[0] == 'label' and 'Animated' in r[1]]
        gain = LI.style_gain(types.SimpleNamespace(
            light_style=style, light_style_rate=10.0), frame / 24.0)
        ok = len(line) == 1 and (
            ('DARK' in line[0][1] and line[0][2] == 'ERROR') if dark
            else (f'x{gain:.2f}' in line[0][1] and line[0][2] != 'ERROR'))
        check(f'{style} at frame {frame}: the panel says '
              + ('the lamp is dark' if dark else f'Power x{gain:.2f}'),
              ok and (gain == 0.0) == dark,
              line[0][1] if line else 'no line')
    rec = _draw_light(UI, defaults, scene=None, light_style='SLOW_STROBE')
    check('without a scene to read the clock from the line is left out '
          '(the panel still draws)',
          not any('Animated' in t for t in _labels(rec))
          and any(r[1] == 'light_style_rate' for r in rec))
    # the panel's clock is the exporter's
    if hasattr(UI, 'scene_clock'):
        ob = FB.geometry_object('C', FB.cube_mesh, kind='MESH')
        bad = []
        for frame, fps, base, sub in ((1, 24, 1.0, 0.0), (37, 30, 1.0, 0.0),
                                      (7, 48, 2.0, 0.0), (3, 24, 1.0, 0.5),
                                      (5, 30, 1.001, 0.25)):
            dg = FB.depsgraph_of(props, [ob])
            dg.scene.frame_current = frame
            dg.scene.frame_subframe = sub
            dg.scene.render.fps = fps
            dg.scene.render.fps_base = base
            sc = EX.export_scene(dg, RenderSettings(), [])
            if float(UI.scene_clock(dg.scene)) != float(sc.time):
                bad.append(f'frame {frame}+{sub} at {fps}/{base}: panel '
                           f'{UI.scene_clock(dg.scene)} export {sc.time}')
        check('the status line reads the seconds the exporter writes to '
              'Scene.time (whole and fractional frames, any frame-rate '
              'base)', not bad, '; '.join(bad))
    # both Only boxes
    for d_on, s_on in ((True, True), (True, False), (False, True)):
        rec = _draw_light(UI, defaults, diffuse_only=d_on, specular_only=s_on)
        warn = [r for r in rec if r[0] == 'label'
                and 'lights no surface' in r[1]]
        check(f'Diffuse Only {d_on} + Specular Only {s_on}: '
              + ('a warning row' if d_on and s_on else 'no warning'),
              (len(warn) == 1 and warn[0][2] == 'ERROR') if d_on and s_on
              else not warn)
    # Ambient Only: what Energy means
    rec = _draw_light(UI, defaults, energy=0.25, ambient_only=True)
    lvl = [r for r in rec if r[0] == 'prop' and r[1] == 'energy'
           and r[2] == 'Ambient Level']
    notes = [r for r in rec if r[0] == 'label' and '1.0 = white ambient'
             in r[1]]
    check('Ambient Only draws the lamp\'s Energy again as Ambient Level '
          'and says its unit (1.0 = white ambient)',
          len(lvl) == 1 and not lvl[0][3].alert and len(notes) == 1
          and notes[0][2] == 'INFO',
          str([n[1] for n in notes]))
    rec = _draw_light(UI, defaults, energy=800.0, ambient_only=True)
    lvl = [r for r in rec if r[0] == 'prop' and r[1] == 'energy'
           and r[2] == 'Ambient Level']
    notes = [r for r in rec if r[0] == 'label' and '1.0 = white ambient'
             in r[1]]
    check('at a point lamp\'s 800 W the row is red and the note is a '
          'warning that names the level',
          len(lvl) == 1 and lvl[0][3].alert and len(notes) == 1
          and notes[0][2] == 'ERROR' and '800' in notes[0][1],
          str([n[1] for n in notes]))
    # the tooltips say what the options do
    table = (getattr(props, 'GROUP_DOCS', None) or {}).get(
        'HalcyonLightSettings', {})

    def tip(name):
        return str(ann[name].kw.get('description') or table.get(name) or '')
    thin = [n for n in ('diffuse_only', 'specular_only', 'ambient_only',
                        'exclude_mode', 'exclude_collection', 'only_shadow')
            if len(tip(n)) < 40]
    check('every touched tooltip is 40+ characters', not thin, str(thin))
    check('the Ambient Only tooltip says INSTEAD and gives the unit; the '
          'Linking tooltip says what an empty Only does; Light Linking '
          'names instances',
          '1.0' in tip('ambient_only') and 'white' in tip('ambient_only')
          and 'lights nothing' in tip('exclude_mode')
          and 'instance' in tip('exclude_collection').lower(),
          tip('ambient_only')[:60])


# ------------------------------------------- 6. the flags on the other roads


def test_g_flags_on_the_other_roads():
    """The cluster's untested corners: per-vertex and per-face shading,
    the fixed-function console models, a material's light group under
    the Light Limit, Only Shadow with Negative, a screen spot that
    leaves the frame's lamps."""
    for rate in ('VERTEX', 'FACE'):
        st = _settings(shading_rate=rate)

        def fr(lights, st=st):
            return _cpu(_scene(lights, st, tess=30), st)
        base = fr([_fill()])
        plain = fr([_fill(), _key()]) - base
        obj = _objects(_scene(_fill(), st, tess=30), st)
        floor = obj == 0
        ns = fr([_fill(), _key(shadow='NONE')]) - base
        lit = floor & (ns.mean(axis=2) > 0.5 * ns[floor].max()) \
            & (np.abs(plain - ns).max(axis=2) < 1e-7)
        core = floor & (ns.mean(axis=2) > 0.25 * ns[floor].max()) \
            & (plain.mean(axis=2) < 0.02 * ns[floor].max())
        neg = fr([_fill(), _key(negative=True)]) - base
        both = fr([_fill(), _key(diffuse_only=True, specular_only=True)])
        osh = fr([_fill(), _key(only_shadow=True)]) - base
        amb = fr([_fill(), _key(ambient_only=True, energy=0.25)]) - base
        lnk = fr([_fill(), _key(exclude_objects=(0,))]) - base
        check(f'{rate} rate: a Negative lamp is minus the plain lamp; both '
              'Only boxes light nothing; Only Shadow darkens the shadow and '
              'leaves the lit floor; Ambient Only is flat; an excluded '
              'floor is unlit',
              float(np.abs(neg + plain).max()) < 2e-5
              and bool(np.array_equal(both, base))
              and int(core.sum()) > 10
              and _mean(osh, core) < -0.5 * _mean(ns, core)
              and abs(_mean(osh, lit)) < 1e-6
              and float(np.abs(amb[floor] - ALB * 0.25).max()) < 1e-6
              and float(np.abs(lnk[floor]).max()) == 0.0,
              f'negative + plain {float(np.abs(neg + plain).max()):.1e}; '
              f'core {int(core.sum())} px, Only Shadow there '
              f'{_mean(osh, core):+.4f} (the lamp gives {_mean(ns, core):+.4f}'
              f'), on the lit floor {_mean(osh, lit):+.1e}; ambient '
              f'{_mean(amb, floor):.4f}')
        for tag, lights in (
                ('negative', [_fill(), _key(negative=True)]),
                ('only shadow', [_fill(), _key(only_shadow=True)]),
                ('ambient only', [_fill(), _key(ambient_only=True,
                                                 energy=0.25)]),
                ('a light link', [_fill(), _key(exclude_objects=(1, 2))])):
            _roads(f'{rate} rate, {tag}', _scene(lights, st, tess=30), st)
    # the fixed-function console models keep the flags on both roads
    ran = []
    for model in ('GX_LIGHT', 'DS_FIXED', 'PS1_MODULATE', 'N64_COMBINE',
                  'SEGA_MODEL2', 'SATURN_ADD'):
        st = _settings(force_model=model)
        lights = [_fill(), _key(negative=True, energy=300.0),
                  _sun(name='Amb', energy=0.2, ambient_only=True),
                  _fill(name='Os', position=(-3.0, -2.0, 5.0),
                        only_shadow=True, shadow='MAP'),
                  _fill(name='Lk', position=(0.0, -6.0, 3.0),
                        exclude_objects=(0,))]
        sc = _scene(lights, st, tess=12)
        c = _cpu(sc, st)
        g, _p, _a, why = _twin(sc, st)
        if g is None:
            note(f'{model}: the plan refuses by name ({str(why)[:70]}) -- '
                 'both devices shade it on the CPU')
            continue
        cov = _objects(sc, st) >= 0
        d = float(np.abs(c[cov] - g[cov]).max())
        ran.append(model)
        check(f'{model} with a Negative, an Ambient Only, an Only Shadow and '
              'a linked lamp: the twin is the CPU frame', d < BAR,
              f'max |CPU - GPU| {d:.2e}')
    check('at least four console models planned', len(ran) >= 4, str(ran))
    # a material's light group under the Light Limit
    st = _settings(max_lights=2, light_limit_mode='FIRST')
    a = _fill(name='A')
    b = _key(name='B', shadow='NONE')
    c_ = _fill(name='C', position=(-4.0, -5.0, 5.0))
    grp = {0: Material(name='FloorBI', index=0, graph=_bi_graph(
        {'light_group_lights': ['B', 'C']}))}
    sc = _scene([a, b, c_], st, mats=grp)
    R.collect_exclusive_lights(sc)
    cg, gg = _roads('a light group [B, C] under Light Limit 2 (keeps A, B)',
                    sc, st)
    st1 = _settings()
    sc1 = _scene([_key(name='B', shadow='NONE')], st1, mats=grp)
    only_b = _cpu(sc1, st1)
    floor = _objects(sc, st) == 0
    check('the grouped floor is lit by B alone: its group\'s lamps that '
          'survive the limit (C is dropped, A is not in the group)',
          float(np.abs(cg[floor] - only_b[floor]).max()) < 1e-6,
          f'max difference {float(np.abs(cg[floor] - only_b[floor]).max()):.1e}')
    # a console material's own light limit under the scene's Light Limit
    from .test_r252_console import console_graph
    con = {1: Material(name='BallCon', index=1, graph=console_graph(
        {'console': 'SYSTEM22', 'light_limit': True}))}
    weak = _fill(name='Weak', energy=2.0)
    b1 = _key(name='B', shadow='NONE', energy=120.0)
    c1 = _fill(name='C', position=(-4.0, -5.0, 5.0), energy=110.0)
    st_c = _settings(shadows=False, max_lights=2, light_limit_mode='BRIGHTEST')
    sc_c = _scene([weak, b1, c1], st_c, mats=con)
    if sc_c.mesh.colors is not None:
        sc_c.mesh.colors = None
    kept = [l.name for l in _frame_lights(sc_c, st_c)]
    cc, gc = _roads('a System 22 material (one lamp) under Light Limit 2 '
                    '(Brightest keeps B, C)', sc_c, st_c)
    st_1 = _settings(shadows=False)
    sc_1 = _scene([_key(name='B', shadow='NONE', energy=120.0)], st_1,
                  mats=con)
    if sc_1.mesh.colors is not None:
        sc_1.mesh.colors = None
    ball = _objects(sc_c, st_c) == 1
    one_b = _cpu(sc_1, st_1)
    check('the machine\'s one lamp is the FIRST survivor of the scene\'s '
          'limit in scene order (B), not the first lamp of the scene',
          kept == ['B', 'C']
          and float(np.abs(cc[ball] - one_b[ball]).max()) < 1e-6,
          f'kept {kept}; ball against the B-only frame '
          f'{float(np.abs(cc[ball] - one_b[ball]).max()):.1e}')
    # Only Shadow with Negative: Blender Internal's arithmetic, -(-x)
    rg = _regions()
    st = _settings()
    base = _cpu(_scene([_fill()], st), st)
    osh, g1 = _roads('Only Shadow', _scene([_fill(), _key(only_shadow=True)],
                                           st), st)
    osn, g2 = _roads('Only Shadow + Negative', _scene(
        [_fill(), _key(only_shadow=True, negative=True)], st), st)
    check('Only Shadow + Negative gives back exactly what Only Shadow takes '
          '(the shadow brightens: Blender Internal\'s negative energy in '
          'the same subtraction)',
          float(np.abs((osn - base) + (osh - base)).max()) < 1e-6
          and _mean(osh - base, rg['floor_core']) < -0.01,
          f'core {_mean(osh - base, rg["floor_core"]):+.4f} / '
          f'{_mean(osn - base, rg["floor_core"]):+.4f}')
    # a screen spot outside the frame's lamps, with Spotlight Fog on
    from . import featurematrix as FM
    for tag, edit in (
            ('dropped by the Light Limit',
             lambda s, t: (setattr(t, 'max_lights', len(s.lights) - 1),
                           setattr(t, 'light_limit_mode', 'FIRST'))),
            ('with Ambient Only',
             lambda s, t: setattr(s.lights[-1], 'ambient_only', True))):
        sc_f, st_f = FM.build('Model 3 spotlight fog lobe')
        st_f.transparency = 'NONE'
        edit(sc_f, st_f)
        wf, hf = st_f.resolution_x, st_f.resolution_y
        LI.clear_shadow_cache()
        c_f = np.asarray(R.render(sc_f, st_f))[..., :3]
        view, _p, vp, eye = R.camera_matrices(sc_f.camera, wf, hf)
        g_f = CRg.GBuffer(wf, hf)
        CRg.rasterize(sc_f.mesh.verts, sc_f.mesh.tris, vp, wf, hf, gbuf=g_f)
        job = R.ShadeJob(sc_f, st_f, R.prepare_textures(sc_f, st_f), None,
                         view, eye, wf, hf)
        GSH._PLAN_CACHE.clear()
        try:
            p_f, why_f, atl_f = GSH.plan_frame(job, g_f)
            img_f, _h = GSH.simulate(job, g_f, p_f, atl_f) \
                if p_f is not None else (None, None)
            cov_f = g_f.tri >= 0
            d = float(np.abs(np.asarray(img_f)[cov_f]
                             - c_f[cov_f]).max()) if img_f is not None else -1
            check(f'a Model 3 screen spot {tag}, Spotlight Fog on: the fog '
                  'pass compiles and matches (it read an accumulator no '
                  'pass declared)', img_f is not None and d < BAR,
                  f'max |CPU - GPU| {d:.2e} {why_f or ""}')
        except Exception as exc:                                # noqa: BLE001
            check(f'a Model 3 screen spot {tag}, Spotlight Fog on: the fog '
                  'pass compiles and matches', False, repr(exc)[:120])


# ------------------------------------------------------ 7. strict GLSL


def test_h_strict_glsl_of_the_changed_passes():
    """The changed emitter shapes through the strict front-end, when
    HALCYON_GLSLANG points at glslang (tests/glsl_strict.py is the gate;
    its scenes carry no ambient lamp and no light link)."""
    from . import glsl_strict as GS
    if GS._glslang() is None:
        note('HALCYON_GLSLANG is not set: the strict front-end did not run '
             'on the ambient-tail and link-ladder passes')
        return
    st = _settings()
    amb = Light(type='POINT', name='Amb', energy=0.2, ambient_only=True)
    bi = {0: Material(name='FloorBI', index=0, graph=_bi_graph({}))}
    cases = [('the ambient tail', _scene([_fill(), amb], st)),
             ('the ambient tail with no lamp in the loop', _scene([amb], st)),
             ('the ambient tail under a Blender Internal material',
              _scene([_fill(), amb], st, mats=bi)),
             ('a link ladder with a run', _scene(
                 [_fill(), _key(exclude_objects=(1, 2))], st)),
             ("an 'Only' link that resolved to nothing", _scene(
                 [_fill(), _key(exclude_objects=(-1,),
                                exclude_mode='ONLY')], st)),
             ('an Only Shadow lamp with a run and a lone index', _scene(
                 [_fill(), _key(only_shadow=True,
                                exclude_objects=(0, 1, 5))], st))]
    bad = []
    n = 0
    for label, sc in cases:
        _img, passes, _atl, why = _twin(sc, st)
        if passes is None:
            bad.append(f'{label}: refused, {why}')
            continue
        for ps in passes:
            src = next((x for x in ps if isinstance(x, str)
                        and 'void main' in x), None)
            if src is None:
                continue
            n += 1
            ok, msg = GS.compile_source(src)
            if not ok:
                bad.append(f'{label}: {msg[:200]}')
    check(f'{n} planned passes of the ambient tail and the link ladder '
          'compile under strict GLSL 4.30', not bad and n > 0,
          '; '.join(bad[:2]))


def _bi_graph(props, color=(ALB, ALB, ALB, 1.0)):
    def sk(name, kind, default, link=None, ident=None):
        return {'name': name, 'identifier': ident or name, 'type': kind,
                'default': default, 'link': link}
    p = {'diff_shader': 'LAMBERT', 'spec_shader': 'COOKTORR',
         'shadeless': False}
    p.update(props)
    ins = [sk('Color', 'RGBA', list(color), ident='Diffuse Color'),
           sk('Intensity', 'VALUE', 0.8, ident='Diffuse Level'),
           sk('Specular Intensity', 'VALUE', 0.0, ident='Specular Level'),
           sk('Hardness', 'VALUE', 50.0, ident='Glossiness'),
           sk('Emit', 'VALUE', 0.0)]
    return {'output': 'out', 'nodes': {
        'bi': {'id': 'bi', 'bl_idname': 'HALCYON_BIMaterialNode',
               'props': p, 'inputs': ins,
               'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [sk('Surface', 'SHADER', None, ['bi', 0])],
                'outputs': []}}}


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
    print('all R254 flag and light-link checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
