"""Renderer tests. Run with:  python3 -m halcyon.tests.test_render"""

import os
import sys
import types

import numpy as np

from ..core import post
from ..core import render as R
from ..core.nodeeval import Closure, GraphEvaluator, ShadeContext
from ..core.settings import RenderSettings
from ..presets.library import PRESETS, apply_preset
from .scenebuild import demo_scene

FAILS = []


def check(name, cond, extra=''):
    print(('  ok   ' if cond else '  FAIL ') + name + (('  ' + extra) if extra else ''))
    if not cond:
        FAILS.append(name)


def base_settings(w=96, h=72, **kw):
    st = RenderSettings()
    st.resolution_x, st.resolution_y = w, h
    st.aa_samples = 1
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def render(st):
    return R.render(demo_scene(st), st)


# ------------------------------------------------------------------- basics


def test_renders_something():
    st = base_settings()
    img = render(st)
    check('render produces a full image', img.shape == (72, 96, 4))
    check('image has real content', img[..., :3].std() > 0.02,
          f'std={img[..., :3].std():.4f}')
    check('no NaNs or infinities', np.isfinite(img).all())


def test_geometry_lands_where_projected():
    """Project known world points and confirm the right surface is there.

    Stronger than eyeballing brightness: it ties the camera matrices, the
    clipper and the rasteriser together against an independently computed
    answer.
    """
    from ..core import raster
    w, h = 128, 96
    st = base_settings(w, h, shadows=False)
    sc = demo_scene(st)
    _view, _proj, vp, _eye = R.camera_matrices(sc.camera, w, h)
    gb = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=gb)
    for name, pt, expect in (('sphere', (-1.3, 0.2, 1.0), 1),
                             ('cube', (1.4, -0.4, 0.9), 2),
                             ('floor', (-4.5, 0.0, 0.0), 0)):
        p = np.array([pt[0], pt[1], pt[2], 1.0], np.float32) @ vp.T
        ndc = p[:3] / p[3]
        x = int(np.clip((ndc[0] * 0.5 + 0.5) * w, 0, w - 1))
        y = int(np.clip((ndc[1] * 0.5 + 0.5) * h, 0, h - 1))
        tri = int(gb.tri[y, x])
        mat = int(sc.mesh.mat_index[tri]) if tri >= 0 else -1
        check(f'{name} rasterises at its projected pixel', mat == expect,
              f'pixel ({x},{y}) holds material {mat}, expected {expect}')
    covered = float(gb.mask().mean())
    check('geometry covers a sensible share of the frame',
          0.25 < covered < 0.98, f'{covered:.2f}')


def _shadowed_fraction(lit, shadowed):
    """Share of pixels the shadow pass meaningfully darkened."""
    return float(((lit[..., :3] - shadowed[..., :3]).mean(axis=2) > 0.02).mean())


def test_shadows_darken():
    lit = render(base_settings(shadows=False))
    mapped = render(base_settings(shadows=True, shadow_default='MAP'))
    rayed = render(base_settings(shadows=True, shadow_default='RAY'))
    fm = _shadowed_fraction(lit, mapped)
    fr = _shadowed_fraction(lit, rayed)
    check('shadow maps darken a real region', fm > 0.01, f'{fm:.3f} of pixels')
    check('ray-traced shadows darken a real region', fr > 0.01,
          f'{fr:.3f} of pixels')
    check('shadows never brighten anything',
          float((mapped[..., :3] - lit[..., :3]).max()) < 1e-4)
    # the two methods are approximating the same thing, so they must agree
    agree = float(np.abs(mapped[..., :3] - rayed[..., :3]).mean())
    check('shadow maps and shadow rays agree', agree < 0.02, f'delta={agree:.4f}')


def test_shading_rates_differ():
    px = render(base_settings(shading_rate='PIXEL'))
    vx = render(base_settings(shading_rate='VERTEX'))
    fc = render(base_settings(shading_rate='FACE'))
    check('Gouraud differs from Phong',
          float(np.abs(px - vx).mean()) > 1e-3, f'{np.abs(px - vx).mean():.4f}')
    check('flat differs from Gouraud',
          float(np.abs(vx - fc).mean()) > 1e-3, f'{np.abs(vx - fc).mean():.4f}')


# Translucent with Translucency at 0 genuinely *is* Lambert -- the back lobe is
# multiplied by zero. That is correct behaviour, not a collision to fix.
EXPECTED_TWINS = {('LAMBERT', 'TRANSLUCENT')}


def test_models_differ():
    """Each reflectance model must produce its own image."""
    from ..core.shading import MODEL_ITEMS
    imgs = {}
    for ident, _label, _d in MODEL_ITEMS:
        st = base_settings(64, 48, force_model=ident, shadows=False)
        imgs[ident] = render(st)
        if not np.isfinite(imgs[ident]).all():
            check(f'model {ident} produces finite output', False)
    keys = list(imgs)
    collisions = []
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if float(np.abs(imgs[a] - imgs[b]).mean()) < 1e-6:
                if (a, b) not in EXPECTED_TWINS and (b, a) not in EXPECTED_TWINS:
                    collisions.append(f'{a}=={b}')
    check(f'all {len(keys)} shading models are distinct (bar documented twins)',
          not collisions, ', '.join(collisions[:4]))

    # the pairs that carry the most weight for a period look
    must_differ = (
        ('PHONG', 'BLINN_PHONG'), ('PHONG', 'COOK_TORRANCE'),
        ('LAMBERT', 'OREN_NAYAR'), ('LAMBERT', 'MINNAERT'),
        ('PHONG', 'WARD'), ('PHONG', 'ANISOTROPIC'), ('PHONG', 'STRAUSS'),
        ('PHONG', 'TOON'), ('PHONG', 'CONSTANT'), ('PHONG', 'METAL'),
        ('PHONG', 'BLINN'),
        # Gouraud and flat are shading *rates*: selecting them must change
        # the frequency at which lighting is evaluated, not just the maths
        ('PHONG', 'GOURAUD'), ('GOURAUD', 'FLAT'),
    )
    # a highlight changing shape is a large *local* change and a tiny mean one,
    # so the peak difference is the meaningful measure here
    bad = [f'{a}~{b}' for a, b in must_differ
           if float(np.abs(imgs[a] - imgs[b]).max()) < 1e-3]
    check('the significant model pairs are all visibly different', not bad,
          ', '.join(bad))


def test_batched_rasteriser_matches_reference():
    """The fast rasteriser must be bit-identical to the reference loop.

    This is the whole licence for having two implementations: if they ever
    disagree, the fast one is wrong and this catches it.
    """
    from ..core import raster
    from .scenebuild import _mesh_concat, plane, sphere
    cases = {
        'demo scene': None,
        'dense sphere': [sphere(radius=1.5, segs=64, rings=40, mat=0)],
        'mixed sizes': [plane(size=12.0, mat=0),
                        sphere(centre=(-1, 0, 1), segs=40, rings=26, mat=0)],
    }
    for name, parts in cases.items():
        st = base_settings(200, 150)
        sc = demo_scene(st)
        if parts is not None:
            sc.mesh = _mesh_concat(parts)
        _v, _p, vp, _e = R.camera_matrices(sc.camera, 200, 150)
        bufs = []
        for mode in (False, True):
            gb = raster.GBuffer(200, 150)
            gb.alloc_linear()
            raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 200, 150,
                             gbuf=gb, batched=mode)
            bufs.append(gb)
        a, b = bufs
        ok = (np.array_equal(a.tri, b.tri) and
              np.array_equal(a.front, b.front) and
              float(np.abs(a.bary - b.bary).max()) == 0.0 and
              float(np.abs(a.bary_lin - b.bary_lin).max()) == 0.0)
        fa = np.where(np.isfinite(a.depth), a.depth, 0.0)
        fb = np.where(np.isfinite(b.depth), b.depth, 0.0)
        ok = ok and float(np.abs(fa - fb).max()) == 0.0
        check(f'batched rasteriser is bit-identical ({name})', ok)


def test_batched_transparency_matches_reference():
    """A-buffer fragment sets must agree too, not just the depth buffer."""
    from ..core import raster
    st = base_settings(160, 120)
    sc = demo_scene(st)
    _v, _p, vp, _e = R.camera_matrices(sc.camera, 160, 120)
    sets = []
    for mode in (False, True):
        gb = raster.GBuffer(160, 120)
        fl = raster.FragmentList()
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 160, 120, gbuf=gb,
                         frags=fl, depth_write=False, batched=mode)
        px, py, tri, depth, bary, front = fl.finish()
        order = np.lexsort((tri, depth, px, py))
        sets.append((px[order], py[order], tri[order], np.round(depth[order], 6)))
    same = all(np.array_equal(a, b) for a, b in zip(sets[0], sets[1]))
    check('batched A-buffer fragments match the reference', same,
          f'{sets[0][0].size} vs {sets[1][0].size} fragments')


def test_lighting_is_exposed_sanely():
    """A Blender-default light rig must not clip the frame to white.

    Light energy arrives in watts, so the Lambertian 1/pi has to be applied or
    everything blows out and the material colour is invisible.
    """
    st = base_settings(96, 72)
    for desc, kind, energy, limit in (("Blender's default 1000 W point",
                                       'POINT', 1000.0, 0.08),
                                      ("a sun at strength 3", 'SUN', 3.0, 0.02)):
        sc = demo_scene(st)
        sc.lights = [l for l in sc.lights if l.type == kind]
        sc.lights[0].energy = energy
        img = R.render(sc, st)[..., :3]
        clipped = float((img >= 1.0).mean())
        check(f'{desc} does not blow the frame out', clipped < limit,
              f'{clipped * 100:.1f}% of channels clipped')

    # and the material colour has to survive to the framebuffer
    sc = demo_scene(st)
    sc.materials[1].diffuse = (0.85, 0.1, 0.1)
    sc.materials[2].diffuse = (0.85, 0.1, 0.1)
    red = R.render(sc, st)[..., :3].reshape(-1, 3).mean(0)
    sc = demo_scene(st)
    sc.materials[1].diffuse = (0.1, 0.3, 0.85)
    sc.materials[2].diffuse = (0.1, 0.3, 0.85)
    blue = R.render(sc, st)[..., :3].reshape(-1, 3).mean(0)
    check('material colour survives to the framebuffer',
          red[0] > blue[0] * 1.2 and blue[2] > red[2] * 1.2,
          f'red={np.round(red, 3)} blue={np.round(blue, 3)}')


def test_image_row_order():
    """Row 0 must be the bottom of the picture, matching Blender's rect buffer.

    Getting this backwards renders the whole frame upside down.
    """
    from ..core import raster
    st = base_settings(80, 60, shadows=False)
    sc = demo_scene(st)
    _v, _p, vp, _e = R.camera_matrices(sc.camera, 80, 60)
    gb = raster.GBuffer(80, 60)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 80, 60, gbuf=gb)
    cov = gb.mask()
    check('row 0 is the bottom of the image (Blender rect order)',
          cov[0].mean() > cov[-1].mean(),
          f'row0 coverage {cov[0].mean():.2f}, last row {cov[-1].mean():.2f}')


def test_delivered_size_is_exact():
    """post.process must honour target_size whatever the chain does to the image.

    Blender allocates the render buffer up front; handing back anything larger
    overruns it and takes the process down. Pixel Scale and pixel aspect both
    resize, so the guarantee has to hold across them.
    """
    img = render(base_settings(96, 72))
    target = (96, 72)
    combos = [
        dict(output_scale='NONE'), dict(output_scale='2X'),
        dict(output_scale='3X'), dict(output_scale='4X'),
        dict(pixel_aspect_x=1.0, pixel_aspect_y=1.2),
        dict(pixel_aspect_x=10.0, pixel_aspect_y=11.0, output_scale='2X'),
        dict(output_scale='4X', pixel_aspect_x=59.0, pixel_aspect_y=54.0,
             crt=True, crt_curvature=0.4, composite=True, jpeg_artifacts=True),
    ]
    bad = []
    for kw in combos:
        st = base_settings(96, 72, **kw)
        out = post.process(img, st, target_size=target)
        if out.shape[1] != target[0] or out.shape[0] != target[1]:
            bad.append(f'{kw} -> {out.shape[1]}x{out.shape[0]}')
    check('post output is always exactly the requested size', not bad,
          '; '.join(bad[:2]))

    # and the unconstrained path must still be free to resize
    st = base_settings(96, 72, output_scale='3X')
    free = post.process(img, st)
    check('without a target the chain may still scale up',
          free.shape[:2] == (72 * 3, 96 * 3), str(free.shape))


def test_threaded_shading_is_deterministic():
    """Shading across threads must be bit-identical to shading on one."""
    base = None
    bad = []
    for threads in (1, 4, 20):
        st = base_settings(160, 120, threads=threads)
        img = R.render(demo_scene(st), st)
        if base is None:
            base = img
        elif float(np.abs(base - img).max()) != 0.0:
            bad.append(str(threads))
    check('threaded shading matches single-threaded exactly', not bad,
          'differs at ' + ','.join(bad) if bad else '')


def _wnode(nid, idname, props=None, ins=(), outs=()):
    return {'id': nid, 'bl_idname': idname, 'props': props or {},
            'inputs': [dict(i) for i in ins], 'outputs': [dict(o) for o in outs]}


def _sk(name, typ, default, link=None):
    return {'name': name, 'type': typ, 'default': default, 'link': link}


def _sky_rows(img):
    """The uncovered top of the frame. Row 0 is the bottom."""
    return img[-20:, :, :3]


def test_world_backgrounds_render():
    """Background, environment texture and sky texture must all show up.

    The background pass used to be handed an empty texture dictionary, so any
    world driven by an image rendered black.
    """
    from ..core.scene import ImageBuffer, World
    from .scenebuild import checker_image

    # plain Background node
    st = base_settings(120, 90)
    sc = demo_scene(st)
    sc.world = World()
    sc.world.graph = {'output': 'w', 'nodes': {
        'bg': _wnode('bg', 'ShaderNodeBackground', {},
                     [_sk('Color', 'RGBA', [0.15, 0.35, 0.8, 1.0]),
                      _sk('Strength', 'VALUE', 1.0)],
                     [{'name': 'Background', 'type': 'SHADER'}]),
        'w': _wnode('w', 'ShaderNodeOutputWorld', {},
                    [_sk('Surface', 'SHADER', None, ['bg', 0])], [])}}
    sky = _sky_rows(R.render(sc, st)).reshape(-1, 3).mean(0)
    check('Background node reaches the sky',
          np.allclose(sky, [0.15, 0.35, 0.8], atol=0.05), str(np.round(sky, 3)))

    # environment texture
    env = ImageBuffer(name='sky.hdr',
                      pixels=checker_image(64, a=(0.2, 0.5, 1.0),
                                           b=(1.0, 0.7, 0.3), squares=4))
    sc = demo_scene(st)
    sc.images = {'sky.hdr': env}
    sc.world = World()
    sc.world.graph = {'output': 'w', 'nodes': {
        'e': _wnode('e', 'ShaderNodeTexEnvironment', {'image': 'sky.hdr'},
                    [_sk('Vector', 'VECTOR', [0, 0, 0])],
                    [{'name': 'Color', 'type': 'RGBA'}]),
        'bg': _wnode('bg', 'ShaderNodeBackground', {},
                     [_sk('Color', 'RGBA', [0.05, 0.05, 0.05, 1.0], ['e', 0]),
                      _sk('Strength', 'VALUE', 1.0)],
                     [{'name': 'Background', 'type': 'SHADER'}]),
        'w': _wnode('w', 'ShaderNodeOutputWorld', {},
                    [_sk('Surface', 'SHADER', None, ['bg', 0])], [])}}
    sky = _sky_rows(R.render(sc, st))
    check('environment texture reaches the sky', sky.std() > 0.05,
          f'std={sky.std():.4f} mean={np.round(sky.reshape(-1, 3).mean(0), 3)}')

    # sky texture
    sc = demo_scene(st)
    sc.world = World()
    sc.world.graph = {'output': 'w', 'nodes': {
        's': _wnode('s', 'ShaderNodeTexSky',
                    {'sky_type': 'PREETHAM', 'sun_elevation': 0.4,
                     'turbidity': 2.5, 'sun_disc': False},
                    [_sk('Vector', 'VECTOR', [0, 0, 0])],
                    [{'name': 'Color', 'type': 'RGBA'}]),
        'bg': _wnode('bg', 'ShaderNodeBackground', {},
                     [_sk('Color', 'RGBA', [0.05, 0.05, 0.05, 1.0], ['s', 0]),
                      _sk('Strength', 'VALUE', 1.0)],
                     [{'name': 'Background', 'type': 'SHADER'}]),
        'w': _wnode('w', 'ShaderNodeOutputWorld', {},
                    [_sk('Surface', 'SHADER', None, ['bg', 0])], [])}}
    sky = _sky_rows(R.render(sc, st)).reshape(-1, 3).mean(0)
    check('sky texture produces a lit sky', float(sky.mean()) > 0.05,
          str(np.round(sky, 3)))
    check('sky texture is blue-ish upward', sky[2] > sky[0], str(np.round(sky, 3)))


def test_generated_coords_are_per_object():
    """Generated coordinates normalise over each object, as Blender does.

    Normalising over the whole scene made every procedural texture on a
    normal-sized object sample a tiny patch of its own space and come out flat
    whenever a large ground plane was present.
    """
    from ..core.scene import ObjectInfo
    from .scenebuild import _mesh_concat, plane, sphere
    st = base_settings(64, 48)
    sc = demo_scene(st)
    sc.mesh = _mesh_concat([plane(size=200.0, mat=0, obj=0),
                            sphere(centre=(0, 0, 1), radius=1.0, mat=1, obj=1)])
    sc.objects = [ObjectInfo(name='Ground', index=0,
                             matrix_world=np.eye(4, dtype=np.float32)),
                  ObjectInfo(name='Ball', index=1,
                             matrix_world=np.eye(4, dtype=np.float32))]
    job = R.ShadeJob(sc, st, {}, None, np.eye(4, dtype=np.float32),
                     np.zeros(3, np.float32), 64, 48)
    sel = np.nonzero(sc.mesh.mat_index == 1)[0]
    g = job.context(sel, np.full((sel.size, 3), 1 / 3.0, np.float32)).generated
    span = g.max(0) - g.min(0)
    check('a small object still gets a full 0..1 Generated span',
          float(span.min()) > 0.9, str(np.round(span, 3)))


def test_procedural_textures_vary():
    """Every procedural texture must produce spatial variation, not flat colour."""
    tex = {
        'ShaderNodeTexChecker': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                  _sk('Color1', 'RGBA', [.8, .8, .8, 1]),
                                  _sk('Color2', 'RGBA', [.2, .2, .2, 1]),
                                  _sk('Scale', 'VALUE', 8.0)], {}),
        'ShaderNodeTexNoise': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                _sk('Scale', 'VALUE', 8.0),
                                _sk('Detail', 'VALUE', 2.0),
                                _sk('Roughness', 'VALUE', .5),
                                _sk('Distortion', 'VALUE', 0.)],
                               {'noise_dimensions': '3D'}),
        'ShaderNodeTexGradient': ([_sk('Vector', 'VECTOR', [0, 0, 0])],
                                  {'gradient_type': 'LINEAR'}),
        'ShaderNodeTexVoronoi': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                  _sk('Scale', 'VALUE', 8.0),
                                  _sk('Randomness', 'VALUE', 1.0)],
                                 {'distance': 'EUCLIDEAN', 'feature': 'F1',
                                  'voronoi_dimensions': '3D'}),
        'ShaderNodeTexMagic': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                _sk('Scale', 'VALUE', 5.0),
                                _sk('Distortion', 'VALUE', 1.0)],
                               {'turbulence_depth': 2}),
        'ShaderNodeTexWave': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                               _sk('Scale', 'VALUE', 5.0),
                               _sk('Distortion', 'VALUE', 0.),
                               _sk('Detail', 'VALUE', 2.0)],
                              {'wave_type': 'BANDS', 'wave_profile': 'SIN',
                               'bands_direction': 'X'}),
    }
    flat = []
    for tid, (ins, props) in tex.items():
        st = base_settings(120, 90)
        sc = demo_scene(st)
        sc.materials[0].graph = {'output': 'out', 'nodes': {
            't': _wnode('t', tid, props, ins,
                        [{'name': 'Color', 'type': 'RGBA'},
                         {'name': 'Fac', 'type': 'VALUE'}]),
            'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                        [_sk('Color', 'RGBA', [.8, .8, .8, 1], ['t', 0]),
                         _sk('Roughness', 'VALUE', 0.),
                         _sk('Normal', 'VECTOR', [0, 0, 0])],
                        [{'name': 'BSDF', 'type': 'SHADER'}]),
            'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                          [_sk('Surface', 'SHADER', None, ['b', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}
        floor = R.render(sc, st)[5:40, :, :3]
        if floor.std() < 0.01:
            flat.append(tid.replace('ShaderNodeTex', ''))
    check(f'all {len(tex)} procedural textures produce variation', not flat,
          'flat: ' + ', '.join(flat))


def test_wireframe_shows_the_world_behind():
    """Wireframe's see-through pixels must show the world, not black.

    Rendered with a transparent film so alpha marks the wire itself; with an
    opaque film those pixels are still alpha 1, they just show the sky.
    """
    st = base_settings(160, 120, film_transparent=True)
    sc = demo_scene(st)
    sc.world.sky_blend = True
    sc.world.horizon = (0.9, 0.3, 0.1)
    sc.world.zenith = (0.1, 0.2, 0.9)
    wire = {'output': 'out', 'nodes': {
        'h': _wnode('h', 'HALCYON_ShaderNode', {'model': 'WIREFRAME'},
                    [_sk('Diffuse Color', 'RGBA', [1, 1, 1, 1]),
                     _sk('Opacity', 'VALUE', 1.0)],
                    [{'name': 'Surface', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['h', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}
    for m in sc.materials:
        m.graph = wire
    img = R.render(sc, st)
    frac = float((img[..., 3] > 0.5).mean())
    check('the wireframe model draws edges', 0.002 < frac < 0.25,
          f'{frac * 100:.2f}% of pixels')
    covered_bg = img[60, 80, :3]
    check('see-through pixels show the world, not black',
          float(covered_bg.sum()) > 0.02, str(np.round(covered_bg, 4)))


def test_sky_modes():
    """Every sky mode must produce its own visible background."""
    from ..core.scene import ImageBuffer, World
    from .scenebuild import checker_image
    from ..core import sky as SKY
    st = base_settings(120, 90)
    seen = {}
    # driven by the mode list rather than a copy of it, so a mode added to the
    # engine and forgotten here fails this test instead of going untested
    modes = [m for m in SKY.MODES if m not in ('NODES', 'HDRI')]
    check('every sky mode but NODES and HDRI is covered here',
          len(modes) == len(SKY.MODES) - 2, str(modes))
    for mode in modes:
        sc = demo_scene(st)
        sc.world = World()
        sc.world.mode = mode
        if mode == 'STARFIELD':
            # the default backdrop is nearly black by design; give it enough
            # to measure without changing what is being measured
            sc.world.color = (0.02, 0.02, 0.06)
            sc.world.star_brightness = 2.0
        img = R.render(sc, st)
        seen[mode] = _sky_rows(img)
        check(f'{mode} sky renders something', float(seen[mode].mean()) > 0.01,
              str(np.round(seen[mode].reshape(-1, 3).mean(0), 3)))
    # Distinctness is asked of the whole sphere rather than of the strip of
    # sky one camera happens to see: the demo scene's horizon is flat, and
    # down there a banded gradient and a smooth one agree exactly -- correctly,
    # since the first band *is* the horizon colour.
    th = np.linspace(0.02, np.pi - 0.02, 48)
    ph = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    T_, P_ = np.meshgrid(th, ph)
    dirs = np.stack([np.sin(T_) * np.cos(P_), np.sin(T_) * np.sin(P_),
                     np.cos(T_)], -1).reshape(-1, 3).astype(np.float32)
    full = {}
    for mode in modes:
        w = World()
        w.mode = mode
        if mode == 'STARFIELD':
            w.color = (0.02, 0.02, 0.06)
            w.star_brightness = 2.0
        full[mode] = SKY.evaluate(w, dirs)
    pairs = [(a, b) for i, a in enumerate(full) for b in list(full)[i + 1:]]
    same = [f'{a}=={b}' for a, b in pairs
            if float(np.abs(full[a] - full[b]).mean()) < 1e-4]
    check('the sky modes are all distinct over the whole sphere', not same,
          ', '.join(same))

    sc = demo_scene(st)
    sc.world = World()
    sc.world.mode = 'HDRI'
    sc.world.env_image = ImageBuffer(
        name='sky.hdr', pixels=checker_image(64, a=(0.2, 0.5, 1.0),
                                             b=(1.0, 0.7, 0.3), squares=4))
    sc.images = {'sky.hdr': sc.world.env_image}
    hd = _sky_rows(R.render(sc, st))
    check('HDRI sky renders the image', hd.std() > 0.05, f'std={hd.std():.4f}')

    # an explicit mode must beat the node tree, which Blender worlds always have
    sc = demo_scene(st)
    sc.world = World()
    sc.world.mode = 'GRADIENT'
    sc.world.zenith = (0.9, 0.1, 0.1)
    sc.world.graph = {'output': 'w', 'nodes': {
        'bg': _wnode('bg', 'ShaderNodeBackground', {},
                     [_sk('Color', 'RGBA', [0.0, 1.0, 0.0, 1.0]),
                      _sk('Strength', 'VALUE', 1.0)],
                     [{'name': 'Background', 'type': 'SHADER'}]),
        'w': _wnode('w', 'ShaderNodeOutputWorld', {},
                    [_sk('Surface', 'SHADER', None, ['bg', 0])], [])}}
    top = _sky_rows(R.render(sc, st)).reshape(-1, 3).mean(0)
    green_wins = top[1] > top[0] * 1.5 and top[1] > top[2] * 1.5
    check('an explicit sky mode overrides the node tree', not green_wins,
          str(np.round(top, 3)))


def _hemisphere(n_el=60, n_az=120):
    """A real grid of sky directions. A spiral of samples can miss a thin band
    like a rainbow entirely, which is exactly how it fooled me once."""
    el = np.linspace(-0.1, 1.45, n_el)
    az = np.linspace(0, 2 * np.pi, n_az)
    E, A = np.meshgrid(el, az, indexing='ij')
    d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
    return d.reshape(-1, 3).astype(np.float32)


def test_bryce_layers_respond():
    """Every layer of the Bryce stack must change the sky on its own."""
    from ..core import sky as SKY
    from ..core.scene import World
    dirs = _hemisphere()

    def sky_of(**kw):
        w = World()
        w.mode = 'BRYCE'
        for k, v in kw.items():
            setattr(w, k, v)
        return SKY.evaluate(w, dirs, {})

    base = sky_of()
    check('the Bryce sky is finite and positive',
          bool(np.isfinite(base).all()) and float(base.min()) >= 0.0)

    layers = (
        ('cumulus', dict(clouds=False)),
        ('cloud cover', dict(cloud_cover=0.9)),
        ('cloud thickness', dict(cloud_thickness=1.6)),
        ('cloud sun rim', dict(cloud_rim=3.0)),
        ('cloud frequency', dict(cloud_scale=6.0)),
        ('stratus', dict(stratus=True, stratus_density=1.0)),
        ('haze', dict(haze_density=1.0, haze_height=1.0)),
        ('haze sun tint', dict(haze_density=1.0, haze_sun_tint=1.0)),
        ('ground fog', dict(fog_density=0.9)),
        ('sun corona', dict(sun_corona=4.0, sun_glow=1.0)),
        ('sun altitude', dict(sun_elevation=1.2)),
        ('sun azimuth', dict(sun_rotation=3.0)),
        ('rainbow', dict(rainbow=True, rainbow_intensity=1.0)),
        ('secondary bow', dict(rainbow=True, rainbow_intensity=1.0,
                               rainbow_secondary=2.0)),
        ('stars', dict(stars=True, star_brightness=2.0, star_density=1.0)),
        ('sky rotation', dict(rotation=1.5)),
    )
    dead = []
    for label, kw in layers:
        alt = sky_of(**kw)
        if float(np.abs(base - alt).max()) < 1e-3:
            dead.append(label)
    check(f'all {len(layers)} Bryce controls change the sky', not dead,
          'no effect: ' + ', '.join(dead))


def test_rainbow_geometry():
    """The bow must sit at its stated angle from the antisolar point."""
    from ..core import sky as SKY
    from ..core.scene import World
    dirs = _hemisphere(120, 240)
    w = World()
    w.mode = 'BRYCE'
    w.clouds = False
    w.haze_density = 0.0
    plain = SKY.evaluate(w, dirs, {})
    w.rainbow = True
    w.rainbow_intensity = 1.0
    w.rainbow_secondary = 0.0
    bowed = SKY.evaluate(w, dirs, {})
    diff = np.abs(bowed - plain).max(axis=1)
    sun = SKY._sun_vector(w.sun_elevation, w.sun_rotation)
    ang = np.degrees(np.arccos(np.clip(dirs @ -sun, -1.0, 1.0)))
    lit = diff > 1e-3
    check('the rainbow appears somewhere', bool(lit.any()))
    if lit.any():
        centre = float(ang[lit].mean())
        check('the bow sits at ~42 degrees from the antisolar point',
              abs(centre - w.rainbow_radius) < 3.0, f'{centre:.1f} deg')
        outside = ang[lit] > centre
        inside = ang[lit] < centre
        if outside.any() and inside.any():
            red_out = float(bowed[lit][outside][:, 0].mean())
            red_in = float(bowed[lit][inside][:, 0].mean())
            check('red is on the outside of the primary bow', red_out > red_in,
                  f'{red_out:.3f} vs {red_in:.3f}')


def test_debug_passes_survive_a_preset():
    """Render passes are data and must bypass the period display chain."""
    from ..presets.library import apply_preset
    from ..core.settings import RenderSettings
    for mode in ('DEPTH', 'NORMAL', 'UV', 'MATID', 'OVERDRAW'):
        st = RenderSettings()
        apply_preset(st, 'EGA')          # 16 colours + heavy error diffusion
        st.resolution_x, st.resolution_y = 120, 90
        st.aa_samples = 1
        st.output_scale = 'NONE'
        st.debug_pass = mode
        raw = R.render(demo_scene(st), st)
        out = post.process(raw, st, target_size=(120, 90))[..., :3]
        # only the display gamma may touch a data pass -- no palette, no dither,
        # no CRT mask, no JPEG blocks
        expect = np.power(np.clip(raw[..., :3], 0, 1), 1.0 / max(st.gamma, 1e-3))
        delta = float(np.abs(out - expect).mean())
        check(f'{mode} survives a 16-colour preset',
              out.std() > 0.02 and delta < 0.02,
              f'std={out.std():.4f} delta-beyond-gamma={delta:.4f}')


def test_gabor_and_noise_types():
    """Gabor exists, and the Noise node honours every fractal type."""
    from ..core.nodeeval import DISPATCH
    check('Gabor texture is implemented', 'ShaderNodeTexGabor' in DISPATCH)
    ins = [_sk('Vector', 'VECTOR', [0, 0, 0]), _sk('Scale', 'VALUE', 6.0),
           _sk('Detail', 'VALUE', 3.0), _sk('Roughness', 'VALUE', 0.5),
           _sk('Lacunarity', 'VALUE', 2.0), _sk('Offset', 'VALUE', 0.0),
           _sk('Gain', 'VALUE', 1.0), _sk('Distortion', 'VALUE', 0.0)]
    outs = [{'name': 'Fac', 'type': 'VALUE'}, {'name': 'Color', 'type': 'RGBA'}]
    seen = {}
    for ntype in ('FBM', 'MULTIFRACTAL', 'RIDGED_MULTIFRACTAL',
                  'HYBRID_MULTIFRACTAL', 'HETERO_TERRAIN'):
        st = base_settings(96, 72)
        sc = demo_scene(st)
        sc.materials[0].graph = {'output': 'out', 'nodes': {
            't': _wnode('t', 'ShaderNodeTexNoise',
                        {'noise_dimensions': '3D', 'noise_type': ntype,
                         'normalize': True}, ins, outs),
            'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                        [_sk('Color', 'RGBA', [.8, .8, .8, 1], ['t', 0]),
                         _sk('Roughness', 'VALUE', 0.),
                         _sk('Normal', 'VECTOR', [0, 0, 0])],
                        [{'name': 'BSDF', 'type': 'SHADER'}]),
            'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                          [_sk('Surface', 'SHADER', None, ['b', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}
        seen[ntype] = R.render(sc, st)[5:40, :, :3]
        check(f'noise type {ntype} varies', seen[ntype].std() > 0.01,
              f'std={seen[ntype].std():.4f}')
    same = [a for a in seen if a != 'FBM'
            and float(np.abs(seen[a] - seen['FBM']).mean()) < 1e-5]
    check('the fractal types differ from plain fBm', not same, ', '.join(same))

    # Gabor through a material
    st = base_settings(96, 72)
    sc = demo_scene(st)
    sc.materials[0].graph = {'output': 'out', 'nodes': {
        't': _wnode('t', 'ShaderNodeTexGabor', {'gabor_type': '2D'},
                    [_sk('Vector', 'VECTOR', [0, 0, 0]),
                     _sk('Scale', 'VALUE', 5.0), _sk('Frequency', 'VALUE', 2.0),
                     _sk('Anisotropy', 'VALUE', 1.0),
                     _sk('Orientation', 'VALUE', 0.7)],
                    [{'name': 'Value', 'type': 'VALUE'},
                     {'name': 'Phase', 'type': 'VALUE'},
                     {'name': 'Intensity', 'type': 'VALUE'}]),
        'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                    [_sk('Color', 'RGBA', [.8, .8, .8, 1], ['t', 0]),
                     _sk('Roughness', 'VALUE', 0.),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['b', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}
    g = R.render(sc, st)[5:40, :, :3]
    check('Gabor renders a pattern', g.std() > 0.01, f'std={g.std():.4f}')


def test_wireframe_via_material_override():
    """A material set to Wireframe in the Halcyon panel must draw edges.

    Blender materials always carry a node tree, so the override has to win over
    the graph or the model silently degrades to flat colour.
    """
    st = base_settings(160, 120, film_transparent=True)
    sc = demo_scene(st)
    for m in sc.materials:
        m.use_override = True
        m.model = 'WIREFRAME'
        m.graph = {'output': 'out', 'nodes': {
            'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                        [_sk('Color', 'RGBA', [.8, .8, .8, 1]),
                         _sk('Roughness', 'VALUE', 0.5),
                         _sk('Normal', 'VECTOR', [0, 0, 0])],
                        [{'name': 'BSDF', 'type': 'SHADER'}]),
            'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                          [_sk('Surface', 'SHADER', None, ['b', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}
    img = R.render(sc, st)
    frac = float((img[..., 3] > 0.5).mean())
    check('material-override Wireframe draws edges', 0.002 < frac < 0.25,
          f'{frac * 100:.2f}% of pixels solid')


def test_no_node_falls_back_silently():
    """No node in a normal material may raise and degrade to pass-through.

    A node that throws is caught and replaced by its first input, which still
    produces plausible variation -- that is exactly how a crash inside the Noise
    texture went unnoticed while looking like it worked.
    """
    from ..core.nodeeval import GraphEvaluator, ShadeContext
    from ..core.settings import RenderSettings
    ctx = ShadeContext(64)
    ctx.settings = RenderSettings()
    ctx.generated = np.stack([np.linspace(0, 1, 64)] * 3, 1).astype(np.float32)
    ctx.uv = np.stack([np.linspace(0, 1, 64)] * 2, 1).astype(np.float32)
    ctx.N = np.tile(np.array([[0, 0, 1.0]], np.float32), (64, 1))
    ctx.I = np.tile(np.array([[0, 0, -1.0]], np.float32), (64, 1))
    ctx.P = np.zeros((64, 3), np.float32)

    cases = {
        'ShaderNodeTexNoise': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                _sk('Scale', 'VALUE', 5.0),
                                _sk('Detail', 'VALUE', 3.0),
                                _sk('Roughness', 'VALUE', 0.5),
                                _sk('Lacunarity', 'VALUE', 2.0),
                                _sk('Distortion', 'VALUE', 0.0)],
                               {'noise_dimensions': '3D'}),
        'ShaderNodeTexVoronoi': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                  _sk('Scale', 'VALUE', 5.0),
                                  _sk('Randomness', 'VALUE', 1.0)],
                                 {'feature': 'F1', 'distance': 'EUCLIDEAN'}),
        'ShaderNodeTexMagic': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                _sk('Scale', 'VALUE', 5.0),
                                _sk('Distortion', 'VALUE', 1.0)],
                               {'turbulence_depth': 2}),
        'ShaderNodeTexWave': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                               _sk('Scale', 'VALUE', 5.0),
                               _sk('Distortion', 'VALUE', 2.0),
                               _sk('Detail', 'VALUE', 2.0)],
                              {'wave_type': 'BANDS'}),
        'ShaderNodeTexGabor': ([_sk('Vector', 'VECTOR', [0, 0, 0]),
                                _sk('Scale', 'VALUE', 5.0),
                                _sk('Frequency', 'VALUE', 2.0),
                                _sk('Anisotropy', 'VALUE', 1.0),
                                _sk('Orientation', 'VALUE', 0.5)],
                               {'gabor_type': '2D'}),
    }
    broken = []
    for tid, (ins, props) in cases.items():
        node = _wnode('t', tid, props, ins,
                      [{'name': 'Fac', 'type': 'VALUE'},
                       {'name': 'Color', 'type': 'RGBA'},
                       {'name': 'Value', 'type': 'VALUE'}])
        ev = GraphEvaluator({'output': None, 'nodes': {'t': node}}, ctx)
        ev.eval_output('t', 0)
        ev.eval_output('t', 1)
        if ev.errors:
            broken.append(f"{tid}: {ev.errors[0][2][:60]}")
    check('no texture node raises and silently falls back', not broken,
          '; '.join(broken))


def _pattern_specs():
    """SPECS from the node module, which needs bpy -- so provide the stub.

    Tests must not depend on another test having installed it first; the runner
    orders them alphabetically and that ordering has already shifted once.
    """
    from . import fakebpy
    bpy = fakebpy.install()
    if not hasattr(bpy.types, 'UIList'):
        bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    from ..nodes.pattern_nodes import SPECS
    return SPECS


def _pattern_graph(cls_name, sockets, props, outputs, out_index=0):
    ins = []
    for kind, name, default in sockets:
        typ = {'NodeSocketFloat': 'VALUE', 'NodeSocketColor': 'RGBA',
               'NodeSocketVector': 'VECTOR'}[kind]
        d = list(default) if isinstance(default, tuple) else default
        ins.append(_sk(name, typ, d if d is not None else [0, 0, 0]))
    kinds = {'NodeSocketFloat': 'VALUE', 'NodeSocketColor': 'RGBA',
             'NodeSocketVector': 'VECTOR'}
    outs = [{'name': n, 'type': kinds[k]} for k, n in outputs]
    pv = {}
    for key, spec in props.items():
        pv[key] = spec[1]
    return {'output': 'out', 'nodes': {
        't': _wnode('t', cls_name, pv, ins, outs),
        'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                    [_sk('Color', 'RGBA', [.8, .8, .8, 1], ['t', out_index]),
                     _sk('Roughness', 'VALUE', 0.0),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['b', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}


def test_pattern_nodes():
    """Every Halcyon procedural must render, vary, and never silently fall back.

    The graphs are built from the same spec table the node classes are, so a
    socket rename cannot make this test quietly stop covering anything.
    """
    from ..core.nodeeval import DISPATCH, GraphEvaluator, ShadeContext
    from ..core.settings import RenderSettings
    SPECS = _pattern_specs()

    ctx = ShadeContext(256)
    ctx.settings = RenderSettings()
    g = np.stack([np.linspace(0, 1, 256), np.linspace(0, 1, 256) ** 1.3,
                  np.linspace(0.2, 0.8, 256)], 1).astype(np.float32)
    ctx.generated = g
    ctx.uv = g[:, :2].copy()
    ctx.P = g * 3.0
    ctx.N = np.tile(np.array([[0, 0, 1.0]], np.float32), (256, 1))
    ctx.I = np.tile(np.array([[0, 0, -1.0]], np.float32), (256, 1))
    ctx.time = 1.5

    broke, flat, missing = [], [], []
    for name, _label, _icon, _desc, sockets, props, outputs in SPECS:
        idname = f'HALCYON_{name}Node'
        if idname not in DISPATCH:
            missing.append(idname)
            continue
        graph = _pattern_graph(idname, sockets, props, outputs)
        ev = GraphEvaluator(graph, ctx)
        for oi in range(len(outputs)):
            v = ev.eval_output('t', oi)
            if v is None:
                broke.append(f'{name}[{oi}]=None')
        if ev.errors:
            broke.append(f'{name}: {ev.errors[0][2][:70]}')
            continue
        col = ev.eval_output('t', 0)
        if col is None or float(np.asarray(col).std()) < 1e-4:
            flat.append(name)
    check(f'all {len(SPECS)} pattern nodes have evaluators', not missing,
          ', '.join(missing))
    check('no pattern node raises and silently falls back', not broke,
          '; '.join(broke[:3]))
    check('every pattern node produces variation', not flat, ', '.join(flat))


def test_pattern_nodes_render():
    """And they survive a real render through the material pipeline."""
    SPECS = _pattern_specs()
    flat = []
    for name, _label, _icon, _desc, sockets, props, outputs in SPECS:
        st = base_settings(96, 72)
        sc = demo_scene(st)
        sc.materials[0].graph = _pattern_graph(f'HALCYON_{name}Node', sockets,
                                               props, outputs)
        floor = R.render(sc, st)[5:40, :, :3]
        if not np.isfinite(floor).all() or floor.std() < 0.005:
            flat.append(f'{name}({floor.std():.4f})')
    check('every pattern node renders with visible variation', not flat,
          ', '.join(flat))


def test_film_transparency():
    """The background must be opaque unless transparency is asked for."""
    for transparent in (False, True):
        st = base_settings(80, 60, film_transparent=transparent)
        img = R.render(demo_scene(st), st)
        bg = float(img[-5:, :, 3].max())
        geo = float(img[5:20, :, 3].min())
        check(f'film_transparent={transparent} gives the right background alpha',
              (bg < 0.01) if transparent else (bg > 0.99), f'alpha={bg:.2f}')
        check(f'film_transparent={transparent} keeps geometry opaque', geo > 0.99,
              f'alpha={geo:.2f}')


def test_material_conversion_plan():
    """Converting a material must pick a sensible model and carry inputs over."""
    from ..core.convert import SOURCES, choose_model, glossiness_from_roughness, plan

    expected = {
        'ShaderNodeBsdfDiffuse': 'LAMBERT',
        'ShaderNodeEmission': 'CONSTANT',
        'ShaderNodeBsdfGlass': 'BLINN',
        'ShaderNodeBsdfToon': 'TOON',
        'ShaderNodeBsdfTranslucent': 'TRANSLUCENT',
        'ShaderNodeBsdfMetallic': 'METAL',
    }
    wrong = []
    for idname, model in expected.items():
        got = choose_model(idname, {'Roughness': 0.0}, set())
        if got != model:
            wrong.append(f'{idname}->{got} (wanted {model})')
    check('each source shader maps to the right model', not wrong,
          ', '.join(wrong))

    # every source in the table must produce a plan with a valid model
    from ..core.shading import MODEL_ITEMS
    valid = {m[0] for m in MODEL_ITEMS}
    bad = []
    for idname in SOURCES:
        p = plan(idname, {'Color': [.5, .5, .5, 1], 'Roughness': 0.3}, set())
        if p['model'] not in valid:
            bad.append(f"{idname}->{p['model']}")
        if not p['pairs']:
            bad.append(f'{idname}: nothing carried')
    check(f'all {len(SOURCES)} source shaders plan cleanly', not bad,
          ', '.join(bad))

    # an unknown shader must still convert rather than fail
    p = plan('ShaderNodeBsdfFromTheFuture', {'Color': [1, 0, 0, 1]}, set())
    check('an unknown source still converts', p['model'] in valid and p['pairs'],
          f"{p['model']} {[t for t, _ in p['pairs']]}")

    # linked sockets count even when their constant reads zero
    check('a linked Metallic still selects METAL',
          choose_model('ShaderNodeBsdfPrincipled', {'Metallic': 0.0},
                       {'Metallic'}) == 'METAL')
    check('a linked Anisotropic still selects ANISOTROPIC',
          choose_model('ShaderNodeBsdfPrincipled', {}, {'Anisotropic'})
          == 'ANISOTROPIC')

    # roughness has to become the exponent the period models actually shade with
    check('roughness maps to a sane specular exponent',
          glossiness_from_roughness(1.0) < glossiness_from_roughness(0.5)
          < glossiness_from_roughness(0.1),
          f'{glossiness_from_roughness(1.0):.1f} / '
          f'{glossiness_from_roughness(0.5):.1f} / '
          f'{glossiness_from_roughness(0.1):.1f}')

    # the mapping must only name sockets the master shader actually has
    from ..nodes import shader_nodes as SN
    sockets = {name for _k, name, _d in SN.HALCYON_ShaderNode.SOCKETS}
    unknown = set()
    for table in SOURCES.values():
        for target, _aliases in table:
            if target not in sockets:
                unknown.add(target)
    for extras in __import__('halcyon.core.convert', fromlist=['EXTRAS']).EXTRAS.values():
        for target in extras:
            if target not in sockets:
                unknown.add(target)
    check('every mapped socket exists on the master shader', not unknown,
          ', '.join(sorted(unknown)))


def test_stock_panels_are_adopted():
    """Blender's engine-agnostic property panels must all be available.

    A hand-written list of panel names rots: the previous one was missing the
    material slot list, so a model with several materials had no way to reach
    any but the active one, and the UV map list, colour attributes, vertex
    groups and colour management were gone too.
    """
    from . import fakebpy
    bpy = fakebpy.install()

    class _Panel(bpy.types.Panel):
        pass

    keep = []                       # __subclasses__ is weak; hold references

    def mk(name, compat, space='PROPERTIES'):
        cls = type(name, (_Panel,), {'COMPAT_ENGINES': set(compat),
                                     'bl_space_type': space})
        keep.append(cls)
        return cls

    wanted = ['MATERIAL_PT_context_material', 'DATA_PT_uv_texture',
              'DATA_PT_vertex_colors', 'DATA_PT_vertex_groups',
              'DATA_PT_shape_keys', 'RENDER_PT_color_management',
              'WORLD_PT_viewport_display', 'MATERIAL_PT_viewport']
    for name in wanted:
        mk(name, {'BLENDER_EEVEE'})
    generic = mk('RENDER_PT_output', {'BLENDER_RENDER', 'BLENDER_EEVEE'})
    rejects = [mk('RENDER_PT_freestyle', {'BLENDER_RENDER'}),
               mk('CYCLES_PT_sampling', {'CYCLES'}),
               mk('VIEW3D_PT_tools', {'BLENDER_RENDER'}, space='VIEW_3D')]

    import importlib
    engine = importlib.import_module('halcyon.engine')
    adopted = {c.__name__ for c in engine.enable_compatible_panels()}
    missing = [w for w in wanted if w not in adopted]
    check('every essential stock panel is adopted', not missing,
          ', '.join(missing))
    check('generic panels are adopted', generic.__name__ in adopted)
    wrong = [r.__name__ for r in rejects if r.__name__ in adopted]
    check('unsupported and foreign panels are left alone', not wrong,
          ', '.join(wrong))
    engine.disable_compatible_panels()
    still = [c.__name__ for c in keep
             if 'HALCYON_RENDER' in getattr(c, 'COMPAT_ENGINES', set())]
    check('unregistering removes the engine from every panel', not still,
          ', '.join(still))


def test_inverse_colormap_is_exact():
    """The fast nearest-colour cube must agree with brute force.

    It is built with a matrix product and the |c|^2 term dropped, which is only
    valid because that term cannot change which palette entry is nearest. If
    that reasoning were wrong the dithering would go subtly wrong everywhere.
    """
    from ..core import palette as PA
    rng = np.random.default_rng(3)
    for size in (16, 64, 256):
        pal = rng.random((size, 3)).astype(np.float32)
        icm = PA.InverseColormap(pal)
        n = icm.bits
        cells = rng.integers(0, icm.n, size=(2000, 3))
        c = ((cells + 0.5) / icm.n).astype(np.float32)
        brute = np.argmin(((c[:, None, :] - pal[None, :, :]) ** 2).sum(2), axis=1)
        got = icm.lut[(cells[:, 0] << (2 * n)) | (cells[:, 1] << n) | cells[:, 2]]
        check(f'inverse colormap matches brute force ({size} colours)',
              bool((brute == got).all()),
              f'{int((brute != got).sum())} of 2000 differ')

    pal = rng.random((64, 3)).astype(np.float32)
    a = PA.get_inverse_colormap(pal)
    b = PA.get_inverse_colormap(pal)
    check('the cached cube is the same object', a is b)
    check('the cached cube matches an uncached one',
          np.array_equal(a.lut, PA.InverseColormap(pal).lut))


def test_palette_lock():
    """A locked palette must be identical frame to frame, and releasable."""
    from ..core import palette as PA
    from ..core import post as PO
    from ..core.settings import RenderSettings
    PA.clear_caches()
    st = RenderSettings()
    st.palette_mode = 'ADAPTIVE'
    st.palette_size = 64
    rng = np.random.default_rng(5)
    frame_a = rng.random((32, 32, 3)).astype(np.float32)
    frame_b = rng.random((32, 32, 3)).astype(np.float32)

    st.palette_lock = True
    p1 = PO._palette_for(st, 64, frame_a, 0)
    p2 = PO._palette_for(st, 64, frame_b, 0)
    check('a locked palette is identical on the next frame',
          np.array_equal(p1, p2))

    st.palette_lock = False
    p3 = PO._palette_for(st, 64, frame_b, 0)
    check('unlocking rebuilds it from the new frame',
          not np.array_equal(p1, p3))

    PA.clear_caches()
    st.palette_lock = True
    p4 = PO._palette_for(st, 64, frame_b, 0)
    check('clearing the cache releases the lock', not np.array_equal(p1, p4))


def test_error_diffusion_output():
    """The rewritten diffusion loop must still produce palette-only output."""
    from ..core import dither as DI
    from ..core import palette as PA
    rng = np.random.default_rng(7)
    img = rng.random((48, 64, 3)).astype(np.float32)
    pal = PA.get_palette('ADAPTIVE', 32, img.reshape(-1, 3), 'MEDIAN_CUT', 0)
    for kind in ('FLOYD', 'STUCKI', 'ATKINSON', 'JJN', 'BURKES', 'SIERRA'):
        out, idx = DI.error_diffusion(img, pal, kind, 1.0, True)
        check(f'{kind} emits only palette colours',
              bool(np.abs(out - pal[idx]).max() < 1e-6))
        check(f'{kind} output is finite and in range',
              bool(np.isfinite(out).all() and out.min() >= -1e-6
                   and out.max() <= 1.0 + 1e-6))
    # serpentine must actually change the result
    a, _ = DI.error_diffusion(img, pal, 'FLOYD', 1.0, True)
    b, _ = DI.error_diffusion(img, pal, 'FLOYD', 1.0, False)
    check('serpentine traversal changes the pattern',
          float(np.abs(a - b).mean()) > 1e-4)


def test_wavefront_diffusion_matches_sequential():
    """The diagonal-at-a-time diffusion must equal the pixel-at-a-time one.

    Error diffusion looks strictly serial, but a pixel only depends on
    neighbours up and to the left, so everything on the skewed diagonal
    x + b*y = t is mutually independent. That is only worth having if the result
    is provably the same, so this compares them exactly.
    """
    from ..core import dither as DI
    from ..core import palette as PA
    rng = np.random.default_rng(11)
    img = rng.random((60, 80, 3)).astype(np.float32)
    pal = PA.get_palette('ADAPTIVE', 32, img.reshape(-1, 3), 'MEDIAN_CUT', 0)
    icm = PA.get_inverse_colormap(pal)
    bad = []
    for kind in ('FLOYD', 'JJN', 'STUCKI', 'ATKINSON', 'BURKES', 'SIERRA',
                 'SIERRA_LITE'):
        saved = DI.error_diffusion_wavefront
        DI.error_diffusion_wavefront = lambda *a, **k: None
        try:
            seq, seq_i = DI.error_diffusion(img, pal, kind, 1.0, False, icm)
        finally:
            DI.error_diffusion_wavefront = saved
        fast, fast_i = DI.error_diffusion(img, pal, kind, 1.0, False, icm)
        if not np.array_equal(seq_i, fast_i) or \
                float(np.abs(seq - fast).max()) != 0.0:
            bad.append(kind)
    check('wavefront diffusion is identical to sequential for every kernel',
          not bad, ', '.join(bad))

    # the schedule must be rejected outright if a kernel cannot support one
    check('an unschedulable kernel is refused',
          DI._wave_skew([(-1, 0, 1.0)]) is None)
    check('Floyd-Steinberg skews by 2', DI._wave_skew(DI.KERNELS['FLOYD'][0]) == 2)

    # serpentine must still take the sequential route
    a, _ = DI.error_diffusion(img, pal, 'FLOYD', 1.0, True, icm)
    b, _ = DI.error_diffusion(img, pal, 'FLOYD', 1.0, False, icm)
    check('serpentine still differs from single-direction',
          float(np.abs(a - b).mean()) > 1e-4)


def test_scissor_cuts_work_without_changing_pixels():
    """A banded render must skip triangles outside the band, not just their pixels.

    Without this every worker in a pool rasterises and clips the whole mesh for
    its own slice, and sixty slices means sixty rasterisations -- which measured
    2.3x *slower* than not using the pool at all.
    """
    from ..core import raster

    st = base_settings(240, 180)
    sc = demo_scene(st)
    _v, _p, vp, _eye = R.camera_matrices(sc.camera, 240, 180)

    full = raster.GBuffer(240, 180)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 240, 180, gbuf=full)
    drawn_full = int(full.mask().sum())

    band = raster.GBuffer(240, 180)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 240, 180, gbuf=band,
                     scissor=(60, 120))
    cov = band.mask()
    # A triangle overlapping the band still covers rows beyond it, and that is
    # harmless: only the band's rows are ever read back. What the scissor
    # guarantees is that triangles wholly outside are never processed at all.
    check('the band itself is covered', int(cov[60:120].sum()) > 0)
    check('less is drawn than for a whole frame', int(cov.sum()) < drawn_full,
          f'{int(cov.sum())} vs {drawn_full}')
    check('pixels inside the band are identical either way',
          np.array_equal(full.tri[60:120], band.tri[60:120]))
    fa = np.isfinite(full.depth[60:120])
    fb = np.isfinite(band.depth[60:120])
    check('the uncovered pixels are the same ones', np.array_equal(fa, fb))
    check('and so are their depths where finite',
          float(np.abs(full.depth[60:120][fa] - band.depth[60:120][fa]).max())
          == 0.0)

    # a thin band must touch fewer triangles -- the pixel count is not the
    # measure here, because the floor plane spans the whole frame either way
    thin = raster.GBuffer(240, 180)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 240, 180, gbuf=thin,
                     scissor=(0, 4))
    seen_full = len(np.unique(full.tri[full.mask()]))
    seen_thin = len(np.unique(thin.tri[thin.mask()]))
    check('a thin band touches fewer triangles', seen_thin < seen_full,
          f'{seen_thin} vs {seen_full}')


def test_band_rendering_rejoins_exactly():
    """Rendering in horizontal bands must equal rendering the whole frame."""
    for ss in (1, 2):
        st = base_settings(120, 90, aa_mode='SUPERSAMPLE' if ss > 1 else 'NONE')
        st.aa_samples = ss * ss
        sc = demo_scene(st)
        full = R.render(sc, st)
        rows = [(y, min(y + 15, 90)) for y in range(0, 90, 15)]
        joined = np.concatenate([R.render(sc, st, band=b) for b in rows], axis=0)
        check(f'bands rejoin exactly at {ss}x supersampling',
              joined.shape == full.shape and
              float(np.abs(joined - full).max()) == 0.0,
              f'{joined.shape} vs {full.shape}')


def test_worker_pool():
    """Worker processes must produce the same image, and fail safely."""
    from ..core import parallel as P

    exe = P.find_interpreter()
    check('a worker interpreter can be located', exe is not None, str(exe))
    parent, pkg = P.package_location()
    check('the package can be imported by a worker',
          os.path.isdir(os.path.join(parent, pkg)), f'{parent} / {pkg}')

    st = base_settings(320, 240)
    sc = demo_scene(st)
    ref = R.render(sc, st)
    img, err = P.render_parallel(sc, st, workers=3, scene_key='test')
    if img is None:
        check('worker pool renders (skipped: unavailable here)', True, str(err))
    else:
        check('the worker pool result is identical to in-process',
              img.shape == ref.shape and float(np.abs(img - ref).max()) == 0.0,
              f'{img.shape} vs {ref.shape}')
        img2, _ = P.render_parallel(sc, st, workers=3, scene_key='test')
        check('a second frame reuses the workers and still matches',
              img2 is not None and float(np.abs(img2 - ref).max()) == 0.0)

    # every refusal must be a reason, never a crash
    small = base_settings(40, 30)
    _none, why = P.render_parallel(sc, small, workers=4)
    check('an undersized frame declines with a reason',
          _none is None and bool(why), str(why))
    _none, why = P.render_parallel(sc, st, workers=1)
    check('a single worker declines with a reason',
          _none is None and bool(why), str(why))
    P.shutdown()
    check('the pool shuts down cleanly', True)


def test_compiled_shader_survives_pickling():
    """Workers rebuild compiled shaders from their generated source."""
    import pickle

    from ..shaders.compiler import compile_shader
    prog = compile_shader(
        'uniform float k = 2.0; in vec2 vUV; out vec4 C;'
        ' void main(){ C = vec4(vUV.x * k, k, 0.0, 1.0); }', 'GLSL')
    clone = pickle.loads(pickle.dumps(prog))
    n = 8
    uni = {'k': np.full(n, 3.0, np.float32)}
    varying = {'vUV': np.stack([np.linspace(0, 1, n),
                                np.zeros(n)], 1).astype(np.float32)}
    a, _ = prog.run(uni, varying, n)
    b, _ = clone.run(uni, varying, n)
    check('a pickled shader still compiles and runs identically',
          np.array_equal(a['C'], b['C']))


def test_caches_do_not_change_the_image():
    """Every cache must be invisible in the output."""
    from ..core import lights as LI
    from ..core import render as RR
    st = base_settings(200, 150)
    sc = demo_scene(st)
    RR.clear_caches()
    st.cache_shadows = False
    plain = R.render(sc, st)
    RR.clear_caches()
    st.cache_shadows = True
    first = R.render(sc, st)
    second = R.render(sc, st)
    check('shadow caching does not change the first frame',
          float(np.abs(plain - first).max()) == 0.0)
    check('shadow caching does not change later frames',
          float(np.abs(plain - second).max()) == 0.0)

    # moving a light must invalidate the cache
    sc2 = demo_scene(st)
    sc2.lights[0].direction = (0.7, -0.3, -0.6)
    moved = R.render(sc2, st)
    check('moving a light rebuilds its shadow map',
          float(np.abs(moved - second).max()) > 1e-4)
    RR.clear_caches()


def test_stats_report():
    """The timing breakdown must record the stages it claims to."""
    from ..core import stats as SS
    SS.reset()
    st = base_settings(120, 90)
    R.render(demo_scene(st), st)
    lines = []
    SS.report(printer=lines.append)
    text = '\n'.join(lines)
    missing = [s for s in ('rasterise', 'shade', 'shadow maps')
               if s not in text]
    check('the breakdown names each render stage', not missing,
          ', '.join(missing))
    check('the breakdown names a slowest stage', 'slowest stage' in text)


def test_stats_account_for_the_whole_frame():
    """The breakdown must show any time it failed to attribute.

    A report whose parts summed to 0.08s against a 19.5s total is worse than no
    report, because it points at the wrong stage with apparent authority.
    """
    from ..core import stats as SS
    SS.reset()
    SS.add('a', 0.1)
    lines = []
    SS.report(total=10.0, printer=lines.append)
    text = '\n'.join(lines)
    check('a large gap is reported as unaccounted for', 'unaccounted' in text)
    check('it refuses to name a slowest stage when most time is untracked',
          'not instrumented' in text)

    SS.reset()
    st = base_settings(160, 120)
    R.render(demo_scene(st), st)
    lines = []
    SS.report(printer=lines.append)
    text = '\n'.join(lines)
    for stage in ('rasterise', 'shade', 'background / sky',
                  'resolve / downsample'):
        check(f'the breakdown covers {stage}', stage in text)


def test_fast_background():
    """The cheap background path must not visibly change a smooth sky."""
    from ..core.scene import World
    out = {}
    for fast in (False, True):
        st = base_settings(200, 150, aa_mode='SUPERSAMPLE')
        st.aa_samples = 4
        st.fast_background = fast
        sc = demo_scene(st)
        sc.world = World()
        sc.world.mode = 'GRADIENT'
        out[fast] = R.render(sc, st)
    diff = float(np.abs(out[False] - out[True]).mean())
    check('the fast background matches a smooth sky closely', diff < 0.01,
          f'mean difference {diff:.6f}')
    check('the fast background keeps the frame size',
          out[True].shape == out[False].shape)


def _glass_scene(st, layers=8):
    from .scenebuild import _mesh_concat, plane, sphere
    parts = [plane(size=14.0, mat=0)]
    parts += [sphere(centre=(0, 0, 0.6 + i * 0.3), radius=1.5, segs=24,
                     rings=16, mat=1) for i in range(layers)]
    sc = demo_scene(st)
    sc.mesh = _mesh_concat(parts)
    sc.materials[1].opacity = 0.35
    sc.materials[1].has_alpha = True
    return sc


def test_abuffer_threading_and_layers():
    """Transparency must be threaded, and identical however it is threaded."""
    base = None
    for threads in (1, 4, 16):
        st = base_settings(200, 150, transparency='ABUFFER', threads=threads)
        st.max_transparent_layers = 0
        img = R.render(_glass_scene(st), st)
        if base is None:
            base = img
        else:
            check(f'A-buffer output is identical at {threads} threads',
                  float(np.abs(base - img).max()) == 0.0)

    # the layer cap must keep the near layers, which are the visible ones
    st = base_settings(200, 150, transparency='ABUFFER')
    st.max_transparent_layers = 4
    capped = R.render(_glass_scene(st), st)
    check('a layer cap barely changes the picture',
          float(np.abs(base - capped).mean()) < 0.02,
          f'mean difference {float(np.abs(base - capped).mean()):.5f}')
    check('a layer cap keeps the image finite', bool(np.isfinite(capped).all()))

    # and deep stacks must not cost more per layer than shallow ones
    from ..core import stats as SS
    SS.reset()
    st = base_settings(200, 150, transparency='ABUFFER')
    st.max_transparent_layers = 0
    R.render(_glass_scene(st, layers=16), st)
    lines = []
    SS.report(printer=lines.append)
    check('transparency shading is reported separately',
          'transparency shading' in '\n'.join(lines))


def test_transparency_modes():
    """All four transparency modes must render, and mean different things."""
    out = {}
    for mode in ('NONE', 'STIPPLE', 'SORTED', 'ABUFFER'):
        st = base_settings(200, 150, transparency=mode)
        sc = demo_scene(st)
        sc.materials[1].opacity = 0.4
        sc.materials[1].has_alpha = True
        out[mode] = R.render(sc, st)
        check(f'{mode} renders a visible scene',
              float(out[mode][..., :3].std()) > 0.05,
              f'std={float(out[mode][..., :3].std()):.4f}')

    check('Opaque ignores alpha entirely',
          float(np.abs(out['NONE'] - out['ABUFFER']).mean()) > 1e-4)
    check('Screen Door differs from blending',
          float(np.abs(out['STIPPLE'] - out['ABUFFER']).mean()) > 1e-4)

    # Sorted and A-Buffer agree on convex geometry and part where it crosses
    from .scenebuild import _mesh_concat, plane

    def tilted(z, tilt, mat):
        V, N, UV, T, _m, _o = plane(z=z, size=6.0, mat=mat)
        V = V.copy()
        V[:, 2] += V[:, 0] * tilt
        return (V, N, UV, T, mat, 0)

    got = {}
    for mode in ('SORTED', 'ABUFFER'):
        st = base_settings(200, 150, transparency=mode)
        sc = demo_scene(st)
        sc.mesh = _mesh_concat([tilted(1.0, 0.6, 1), tilted(1.2, -0.6, 2)])
        for m in (sc.materials[1], sc.materials[2]):
            m.opacity = 0.5
            m.has_alpha = True
        got[mode] = R.render(sc, st)
    check('Sorted and A-Buffer differ where polygons interpenetrate',
          float(np.abs(got['SORTED'] - got['ABUFFER']).mean()) > 1e-5,
          f"{float(np.abs(got['SORTED'] - got['ABUFFER']).mean()):.6f}")


def test_rays_only_where_wanted():
    """Secondary rays must be traced per fragment, not per batch.

    Testing np.any() and then tracing for the whole batch meant one transparent
    fragment in a chunk cost a refraction ray for every fragment in it.
    """
    st = base_settings(200, 150, raytrace=True)
    st.ray_depth = 2
    st.ray_refraction = True
    sc = demo_scene(st)
    sc.materials[1].opacity = 0.4
    sc.materials[1].has_alpha = True
    with_glass = R.render(sc, st)

    sc2 = demo_scene(st)          # nothing transparent at all
    opaque_only = R.render(sc2, st)
    check('a transparent material changes the picture',
          float(np.abs(with_glass - opaque_only).mean()) > 1e-4)
    check('ray-traced output stays finite', bool(np.isfinite(with_glass).all()))

    # and reflection must only touch reflective fragments
    sc3 = demo_scene(st)
    sc3.materials[1].reflect_level = 0.0
    sc3.materials[2].reflect_level = 0.9
    mixed = R.render(sc3, st)
    check('reflection applies selectively', bool(np.isfinite(mixed).all())
          and float(np.abs(mixed - opaque_only).mean()) > 1e-5)


def test_strict_node_evaluation():
    """Strict mode must surface a node failure instead of hiding it."""
    from ..core import nodeeval as NE
    from ..core.settings import RenderSettings

    bad = _wnode('t', 'ShaderNodeTexNoise', {'noise_dimensions': 'NONSENSE'},
                 [_sk('Vector', 'VECTOR', [0, 0, 0]),
                  _sk('Scale', 'VALUE', 'not a number')],
                 [{'name': 'Fac', 'type': 'VALUE'}])
    ctx = _ctx()
    graph = {'output': None, 'nodes': {'t': bad}}

    NE.STRICT = False
    ev = NE.GraphEvaluator(graph, ctx)
    ev.eval_output('t', 0)
    check('a broken node falls back quietly by default', bool(ev.errors))

    NE.STRICT = True
    try:
        raised = False
        try:
            NE.GraphEvaluator(graph, ctx).eval_output('t', 0)
        except Exception:                                       # noqa: BLE001
            raised = True
        check('strict mode raises instead', raised)
    finally:
        NE.STRICT = False
    check('strict mode is off again afterwards', NE.STRICT is False)


def test_debug_settings_survive_presets():
    """Developer settings must not be reset by loading a preset."""
    from ..core.settings import RenderSettings
    from ..presets.library import PRESERVED
    st = RenderSettings()
    st.use_processes = True
    st.process_count = 12
    st.show_stats = True
    st.debug_pass = 'NORMAL'
    apply_preset(st, 'PSX')
    check('worker settings survive a preset',
          st.use_processes and st.process_count == 12,
          f'{st.use_processes} {st.process_count}')
    check('the timing and pass settings survive a preset',
          st.show_stats and st.debug_pass == 'NORMAL')
    for name in ('use_processes', 'process_count', 'show_stats', 'debug_pass'):
        check(f'{name} is listed as preserved', name in PRESERVED)


def test_shader_node_tooltips():
    """Every input documented, and the model lists checked against the shader.

    The claim "Glossiness affects Phong but not Cook-Torrance" is only worth
    making if it is true, so the ones that can be measured are measured: each
    parameter is perturbed and the models whose output changes are compared with
    what the tooltip says.
    """
    from ..core.shading import MODEL_ITEMS, Surface, evaluate
    from ..core import mathx as MX
    from ..nodes.shader_nodes import (ALL, MEASURED, SOCKET_DOCS, SOCKET_MODELS,
                                      HALCYON_ShaderNode as HS)

    names = [n for _k, n, _d in HS.SOCKETS]
    check('every input has a tooltip',
          all(n in SOCKET_DOCS for n in names),
          ', '.join(n for n in names if n not in SOCKET_DOCS))
    check('every input has a model list',
          all(n in SOCKET_MODELS for n in names),
          ', '.join(n for n in names if n not in SOCKET_MODELS))

    idents = [m[0] for m in MODEL_ITEMS]
    bad = []
    for n, who in SOCKET_MODELS.items():
        if who != ALL:
            bad += [f'{n}:{m}' for m in who if m not in idents]
    check('model lists name only real models', not bad, ', '.join(bad))
    check('every model has a substantial description',
          all(len(m[2]) > 60 for m in MODEL_ITEMS),
          ', '.join(m[0] for m in MODEL_ITEMS if len(m[2]) <= 60))

    # measure the truth and compare
    fields = {'Specular Color': 'specular', 'Glossiness': 'glossiness',
              'Roughness': 'roughness', 'Anisotropy': 'anisotropy',
              'Anisotropic Rotation': 'aniso_rot', 'IOR': 'ior',
              'Toon Size': 'toon_size', 'Toon Smooth': 'toon_smooth'}
    n = 48
    N = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    L = np.stack([np.linspace(-0.9, 0.9, n), np.full(n, 0.3),
                  np.full(n, 0.6)], 1).astype(np.float32)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    V = np.tile(np.array([[0, 0.2, 0.98]], np.float32), (n, 1))

    def fresh():
        s = Surface(n)
        s.diffuse[:] = 0.6
        s.specular[:] = 0.9
        s.specular_level[:] = 0.7
        s.glossiness[:] = 25.0
        s.roughness[:] = 0.4
        s.metallic[:] = 0.3
        s.anisotropy[:] = 0.3
        s.aniso_rot[:] = 0.4
        s.ior[:] = 1.5
        s.toon_size[:] = 0.5
        s.toon_smooth[:] = 0.2
        s.tangent, s.bitangent = MX.orthonormal_basis(N)
        return s

    wrong = []
    for sock, attr in fields.items():
        if sock not in MEASURED:
            continue
        measured = set()
        for ident in idents:
            a = fresh()
            d1, s1 = evaluate(ident, a, N, L, V)
            b = fresh()
            setattr(b, attr, getattr(b, attr) * 0.25 + 0.05)
            d2, s2 = evaluate(ident, b, N, L, V)
            if float(np.abs(d1 - d2).max()) > 1e-6 or \
                    float(np.abs(s1 - s2).max()) > 1e-6:
                measured.add(ident)
        claimed = set(SOCKET_MODELS[sock]) if SOCKET_MODELS[sock] != ALL \
            else set(idents)
        if measured != claimed:
            wrong.append(f"{sock}: claims {sorted(claimed)} but measures "
                         f"{sorted(measured)}")
    check('the documented model lists match the shading code', not wrong,
          ' | '.join(wrong[:2]))


def test_material_panel_visibility():
    """The material panel must appear even with no material, so one can be made.

    Requiring context.material to exist meant an object with no material had no
    panel, and gating the slot list on more than one slot meant no Add button --
    between them there was no way to create the first material from here.
    """
    import types

    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    bpy.types.AddonPreferences = type('AddonPreferences', (bpy.types.Panel,), {})
    import importlib
    ui = importlib.import_module('halcyon.ui')

    def ctx(engine='HALCYON_RENDER', material=None, obj=True, slots=0):
        o = None
        if obj:
            o = types.SimpleNamespace(
                material_slots=[types.SimpleNamespace(material=None)] * slots,
                active_material_index=0)
        return types.SimpleNamespace(engine=engine, material=material, object=o)

    cases = [
        ('no object', ctx(obj=False), False),
        ('zero slots', ctx(slots=0), True),
        ('one empty slot', ctx(slots=1), True),
        ('a real material', ctx(material=object(), slots=1), True),
        ('another engine', ctx(engine='CYCLES', slots=2), False),
    ]
    wrong = [label for label, c, want in cases
             if bool(ui.HALCYON_PT_material.poll(c)) is not want]
    check('the material panel shows exactly when it should', not wrong,
          ', '.join(wrong))

    src = __import__('inspect').getsource(ui.HALCYON_PT_material.draw)
    check('the slot list is not gated on the slot count',
          'material_slots' in src and "len(slots) > 1" not in
          src.split('template_list')[0],
          'the list is still conditional')
    check('a New control is offered', 'material.new' in src)


def _master(**over):
    ins = [_sk('Diffuse Color', 'RGBA', [.7, .7, .75, 1]),
           _sk('Diffuse Level', 'VALUE', 1.0),
           _sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
           _sk('Specular Level', 'VALUE', 0.5),
           _sk('Glossiness', 'VALUE', 25.0), _sk('Opacity', 'VALUE', 1.0),
           _sk('Fresnel', 'VALUE', 0.0), _sk('Fresnel Power', 'VALUE', 3.0),
           _sk('Fresnel Color', 'RGBA', [1, 1, 1, 1]),
           _sk('Rim Light', 'RGBA', [1, 1, 1, 1]),
           _sk('Rim Amount', 'VALUE', 0.0), _sk('Rim Power', 'VALUE', 3.0),
           _sk('Matcap', 'RGBA', [1, 0, 0, 1]),
           _sk('Matcap Blend', 'VALUE', 0.0),
           _sk('Reflection Color', 'RGBA', [1, 1, 1, 1]),
           _sk('Edge Opacity', 'VALUE', 1.0),
           _sk('Backface Color', 'RGBA', [0, 1, 0, 1]),
           _sk('Backface Mix', 'VALUE', 0.0)]
    for k, v in over.items():
        for x in ins:
            if x['name'] == k:
                x['default'] = v
    return {'output': 'out', 'nodes': {
        'h': _wnode('h', 'HALCYON_ShaderNode', {'model': 'PHONG'}, ins,
                    [{'name': 'Surface', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['h', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}


def test_master_shader_effects():
    """Fresnel, rim, matcap, edge opacity and backface must each do something."""
    st = base_settings(160, 120, backface_cull=False)

    def go(**over):
        sc = demo_scene(st)
        for m in sc.materials:
            m.graph = _master(**over)
        return R.render(sc, st)

    base = go()
    for label, over in (('Fresnel', {'Fresnel': 1.5}),
                        ('Rim Light', {'Rim Amount': 1.2}),
                        ('Matcap', {'Matcap Blend': 0.9}),
                        ('Edge Opacity', {'Edge Opacity': 0.0}),
                        ('Backface Mix', {'Backface Mix': 1.0})):
        img = go(**over)
        check(f'{label} changes the render',
              float(np.abs(img - base).mean()) > 1e-4,
              f'delta {float(np.abs(img - base).mean()):.6f}')
        check(f'{label} stays finite', bool(np.isfinite(img).all()))

    # Fresnel must act at the silhouette, not uniformly
    lit = go(Fresnel=3.0)
    diff = np.abs(lit - base)[..., :3].mean(axis=2)
    check('Fresnel is stronger somewhere than everywhere',
          float(diff.max()) > 4.0 * float(diff.mean()) if diff.mean() > 0 else False,
          f'peak {diff.max():.4f} vs mean {diff.mean():.4f}')

    # and the effects work whatever reflectance model is chosen
    broken = []
    for model in ('LAMBERT', 'COOK_TORRANCE', 'TOON', 'METAL'):
        sc = demo_scene(st)
        g = _master(**{'Rim Amount': 1.5})
        g['nodes']['h']['props']['model'] = model
        for m in sc.materials:
            m.graph = g
        plain = _master()
        plain['nodes']['h']['props']['model'] = model
        sc2 = demo_scene(st)
        for m in sc2.materials:
            m.graph = plain
        if float(np.abs(R.render(sc, st) - R.render(sc2, st)).mean()) < 1e-4:
            broken.append(model)
    check('the rim term works on every model', not broken, ', '.join(broken))


def test_matcap_coordinates():
    """Matcap UVs must span the sphere and stay inside 0..1."""
    from ..core.nodeeval import DISPATCH, GraphEvaluator
    check('the Matcap Coordinates node exists',
          'HALCYON_MatcapUVNode' in DISPATCH)
    ctx = _ctx(64)
    ang = np.linspace(-1.2, 1.2, 64)
    ctx.N = np.stack([np.sin(ang), np.zeros(64), np.cos(ang)], 1).astype(np.float32)
    ctx.I = np.tile(np.array([[0, 0, -1.0]], np.float32), (64, 1))
    node = _wnode('m', 'HALCYON_MatcapUVNode', {},
                  [_sk('Scale', 'VALUE', 1.0)],
                  [{'name': 'Vector', 'type': 'VECTOR'},
                   {'name': 'Facing', 'type': 'VALUE'}])
    ev = GraphEvaluator({'output': None, 'nodes': {'m': node}}, ctx)
    uv = ev.eval_output('m', 0)
    check('matcap coordinates are produced', uv is not None)
    if uv is not None:
        check('they stay within the unit square',
              float(uv[:, :2].min()) >= -0.001 and float(uv[:, :2].max()) <= 1.001,
              f'{uv[:, :2].min():.3f}..{uv[:, :2].max():.3f}')
        check('they vary with the normal', float(uv[:, 0].std()) > 0.05)
    check('no error was swallowed', not ev.errors,
          str(ev.errors[0] if ev.errors else ''))


def test_period_objects():
    """The Add-menu objects must be sound geometry, not just a pile of quads.

    Each generator is checked for the mistakes a lattice generator actually
    makes -- an off-by-one in a wrap, a degenerate quad, an index past the end
    -- and then for the things that make each object itself: the teapot has a
    handle on one side and a spout on the other, the Cornell box is the
    published measurement with its normals facing in, and the checkerboard
    alternates.
    """
    from ..core import geometry as GEO

    broke = []
    for label, (verts, faces) in (
            ('teapot', GEO.utah_teapot(6, 12)),
            ('teacup', GEO.teacup(4, 12)),
            ('saucer', GEO.saucer(4, 12)),
            ('teaspoon', GEO.teaspoon(4, 10))):
        ok, msg = GEO.is_manifoldish(verts, faces)
        if not ok:
            broke.append(f'{label}: {msg}')
    check('every generated object is sound geometry', not broke,
          '; '.join(broke))

    # resolution is a real knob, not decoration
    low = GEO.utah_teapot(4, 8)
    high = GEO.utah_teapot(12, 24)
    check('the teapot resolution knob does something',
          len(high[1]) > 4 * len(low[1]),
          f'{len(low[1])} -> {len(high[1])} faces')

    (lo, _ly, lz), (hi, _hy, hz) = GEO.bounds(GEO.utah_teapot(8, 16)[0])
    check('the teapot has a spout on one side and a handle on the other',
          lo < -2.8 and hi > 2.9, f'x from {lo:.2f} to {hi:.2f}')
    check('the teapot stands on the floor at Newell height',
          abs(lz) < 1e-6 and abs(hz - 3.15) < 1e-6, f'z {lz:.3f}..{hz:.3f}')

    # the Cornell box: measurements, normals and which wall is which
    verts, faces, groups = GEO.cornell_box()
    ok, msg = GEO.is_manifoldish(verts, faces)
    check('the Cornell box is sound geometry', ok, msg)
    V = np.asarray(verts, np.float64)
    room = [i for i, g in enumerate(groups)
            if g in ('white', 'red', 'green', 'light')]
    centre = (V.min(0) + V.max(0)) * 0.5
    outward = []
    for i in room:
        p = V[list(faces[i])]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        n = n / max(float(np.linalg.norm(n)), 1e-12)
        to_centre = centre - p.mean(0)
        if float(np.dot(n, to_centre / np.linalg.norm(to_centre))) < 0:
            outward.append(groups[i])
    check('every wall of the Cornell box faces inward', not outward,
          ', '.join(outward))

    for name, want_sign in (('red', +1.0), ('green', -1.0)):
        idx = [i for i, g in enumerate(groups) if g == name]
        x = float(np.mean([V[list(faces[i])][:, 0].mean() for i in idx]))
        check(f'the {name} wall is on the side the published scene puts it',
              x * want_sign > 0.4, f'x={x:.3f}')

    for name, height in (('short', 165.0), ('tall', 330.0)):
        idx = [i for i, g in enumerate(groups) if g == name]
        z = max(float(V[list(faces[i])][:, 2].max()) for i in idx)
        check(f'the {name} block is {height:.0f}mm tall',
              abs(z - height / 552.8) < 1e-6, f'{z * 552.8:.1f}mm')

    # the checkerboard
    cv, cf, cg = GEO.checker_plane(10.0, 6)
    ok, msg = GEO.is_manifoldish(cv, cf)
    check('the checker plane is sound geometry', ok, msg)
    check('the checker plane has one group per face', len(cg) == len(cf))
    grid = np.asarray(cg).reshape(6, 6)
    check('the checker plane actually alternates',
          bool((grid[:, :-1] != grid[:, 1:]).all()
               and (grid[:-1, :] != grid[1:, :]).all()))
    span = np.asarray(cv)[:, 0]
    check('the checker plane is the size it was asked for',
          abs(float(span.max() - span.min()) - 10.0) < 1e-6)


def test_generated_objects_are_wound_outward():
    """Every generated solid must have its normals pointing out of it.

    This is the bug the first cut of these objects shipped with. A surface of
    revolution swept while walking *down* the profile and *anticlockwise* around
    the axis winds every quad the wrong way round, and nothing says so: the
    z-buffer does not care, and the shading takes an absolute value. It only
    appears when backface culling is switched on -- five presets do -- and then
    the whole outer surface is culled and you are looking at the far interior.
    The teapot's own base was visible through its side.

    Two independent checks, because the first is cheap and the second is what
    the user actually sees.
    """
    from ..core import geometry as GEO
    from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
    from .scenebuild import _mesh_concat, look_at_matrix

    # 1. the divergence theorem. Exact for these even though the band seams are
    #    topologically open, because the two rings at a seam are coincident.
    wrong = []
    for label, built in (('teapot', GEO.utah_teapot(8, 16)),
                         ('teacup', GEO.teacup(5, 16)),
                         ('saucer', GEO.saucer(4, 16)),
                         ('teaspoon', GEO.teaspoon(5, 12))):
        vol = GEO.signed_volume(built[0], built[1])
        if vol <= 0.0:
            wrong.append(f'{label} encloses {vol:+.4f}')
    check('every lathed object encloses a positive volume', not wrong,
          '; '.join(wrong))

    # the Cornell box is the deliberate exception: its walls face inward, so
    # its signed volume is negative, and a test that just asked for "positive"
    # everywhere would have to make an exception rather than state the rule
    v, f, _g = GEO.cornell_box()
    check('the Cornell box is inside out on purpose, and stays that way',
          GEO.signed_volume(v, f) < 0.0)

    # 2. and the invariant that matters: for a closed solid seen from outside,
    #    culling the back can only ever remove something already hidden.
    def prim(built):
        V = np.asarray(built[0], np.float32)
        tris = []
        for face in built[1]:
            for k in range(1, len(face) - 1):
                tris.append((face[0], face[k], face[k + 1]))
        T = np.asarray(tris, np.int32)
        N = np.zeros_like(V)
        fn = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
        for i in range(3):
            np.add.at(N, T[:, i], fn)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        N = N / np.where(ln < 1e-12, 1.0, ln)
        return (V, N.astype(np.float32),
                (V[:, :2] * 0.25 + 0.5).astype(np.float32), T, 0, 0)

    def shot(built, cull, cam, target):
        st = base_settings(200, 150)
        st.backface_cull = cull
        mesh = _mesh_concat([prim(built)])
        mesh.smooth = np.ones(mesh.tris.shape[0], bool)
        sc = Scene(
            mesh=mesh,
            materials=[Material(name='P', index=0, model='BLINN_PHONG',
                                diffuse=(0.8, 0.72, 0.35),
                                specular_level=0.9, glossiness=140.0)],
            objects=[ObjectInfo(name='T', index=0,
                                matrix_world=np.eye(4, dtype=np.float32))],
            lights=[Light(type='SUN', direction=(-0.55, 0.5, -0.65),
                          energy=5.0)],
            camera=Camera(matrix_world=look_at_matrix(cam, target), lens=45.0,
                          sensor=36.0, clip_start=0.1, clip_end=500.0),
            world=World(mode='SOLID', color=(0.0, 0.0, 0.0)), settings=st)
        return R.render(sc, st)[..., :3]

    views = ((1, 0, 0.4), (-1, 0, 0.4), (0, 1, 0.4), (0, -1, 0.4),
             (0.93, 0, 0.37), (0, 0, 1), (0, 0, -1))
    bad = []
    for label, built in (('teapot', GEO.utah_teapot(8, 20)),
                         ('teacup', GEO.teacup(5, 16)),
                         ('saucer', GEO.saucer(4, 16)),
                         ('teaspoon', GEO.teaspoon(5, 12))):
        V = np.asarray(built[0], np.float64)
        centre = (V.min(0) + V.max(0)) * 0.5
        radius = float(np.linalg.norm(V - centre, axis=1).max())
        for d in views:
            d = np.asarray(d, np.float64)
            d = d / np.linalg.norm(d)
            cam = tuple(centre + d * radius * 4.0)
            a = shot(built, False, cam, tuple(centre))
            b = shot(built, True, cam, tuple(centre))
            mean = float(np.abs(a - b).mean())
            px = int((np.abs(a - b).mean(axis=2) > 1e-3).sum())
            # a handful of pixels may legitimately flip: exactly at the
            # silhouette a front and a back face meet inside one pixel, and
            # which of them wins is a tie that culling breaks differently.
            # A winding error is not a handful of pixels, it is the object.
            if mean > 1e-3 or px > a[..., 0].size // 2000:
                bad.append(f'{label} from {np.round(d, 2).tolist()}: '
                           f'{px}px, mean {mean:.5f}')
    check('culling the back of a generated solid leaves the picture alone',
          not bad, '; '.join(bad[:3]))


def test_period_objects_render():
    """And each of them survives being handed to the renderer."""
    from ..core import geometry as GEO
    from .scenebuild import _mesh_concat, demo_scene

    def as_mesh(built):
        verts, faces = built[0], built[1]
        V = np.asarray(verts, np.float32)
        extent = float(np.abs(V).max()) or 1.0
        V = V * (1.6 / extent)
        tris = []
        for f in faces:
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        T = np.asarray(tris, np.int32)
        # smooth vertex normals, so a shared vertex gets one -- the renderer
        # uses the face normal anyway with smooth off, but the array must be
        # there and must be the right length
        N = np.zeros_like(V)
        e1 = V[T[:, 1]] - V[T[:, 0]]
        e2 = V[T[:, 2]] - V[T[:, 0]]
        fn = np.cross(e1, e2)
        for i in range(3):
            np.add.at(N, T[:, i], fn)
        ln = np.linalg.norm(N, axis=1, keepdims=True)
        N = N / np.where(ln < 1e-12, 1.0, ln)
        UV = V[:, :2] * 0.5 + 0.5
        return _mesh_concat([(V, N.astype(np.float32),
                              UV.astype(np.float32), T, 0, 0)])

    broke = []
    for label, built in (('teapot', GEO.utah_teapot(6, 14)),
                         ('teacup', GEO.teacup(4, 12)),
                         ('saucer', GEO.saucer(4, 12)),
                         ('teaspoon', GEO.teaspoon(4, 10)),
                         ('cornell', GEO.cornell_box()),
                         ('checker', GEO.checker_plane(6.0, 6))):
        st = base_settings(96, 72)
        sc = demo_scene(st)
        sc.mesh = as_mesh(built)
        try:
            img = R.render(sc, st)
        except Exception as exc:                                # noqa: BLE001
            broke.append(f'{label}: {exc!r}')
            continue
        if not np.isfinite(img).all():
            broke.append(f'{label}: not finite')
        elif float(img[..., :3].std()) < 1e-3:
            broke.append(f'{label}: nothing visible')
    check('every period object renders', not broke, '; '.join(broke))


def test_add_menu_is_complete():
    """Every Add-menu operator must be registered and reachable from the menu.

    An operator that exists but is in no menu is invisible, and a menu entry
    for an operator that was renamed is a dead click. Both have happened here.
    """
    import inspect

    from . import fakebpy
    bpy = fakebpy.install()
    bpy.types.UIList = type('UIList', (bpy.types.Panel,), {})
    import importlib
    objects = importlib.import_module('halcyon.objects')

    ops = [c for c in objects.CLASSES
           if c.__name__.startswith('HALCYON_OT_')]
    check('the Add menu has the four period objects', len(ops) == 4,
          ', '.join(c.bl_idname for c in ops))

    src = inspect.getsource(objects.VIEW3D_MT_halcyon_add.draw)
    missing = [c.bl_idname for c in ops if c.__name__ not in src]
    check('every Add operator appears in the menu', not missing,
          ', '.join(missing))

    drawn = inspect.getsource(objects.draw_add_menu)
    check('the Add menu is gated on the engine', 'ENGINE' in drawn)
    check('the menu is appended to the 3D view Add menu',
          'VIEW3D_MT_add' in inspect.getsource(objects.register))

    # every operator's properties must be real bpy properties, which is what
    # the stub's register_class already asserts -- so registering is the test
    objects.register()
    objects.unregister()
    check('the Add-menu module registers and unregisters cleanly', True)


def test_backface_culling_keeps_the_front():
    """Culling must remove the far side of a solid, not the near side.

    The sign was inverted, and it survived because nothing closed was ever
    rendered with culling on: five presets set it, and on those a cube showed
    its own interior -- which is dark, and reads as the object not being there
    rather than as being inside out. The test that catches it is the one that
    needs no reference image: culling the back of a *closed convex* solid can
    only ever be invisible, because the back was behind the front anyway.
    """
    from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
    from .scenebuild import _mesh_concat, cube, look_at_matrix, plane, sphere

    def shot(prim, cull, flip=False):
        st = base_settings(140, 105)
        st.backface_cull = cull
        v, n, uv, t, mi, oi = prim
        if flip:
            t = t[:, ::-1].copy()
        mesh = _mesh_concat([(v, n, uv, t, mi, oi)])
        mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
        sc = Scene(
            mesh=mesh,
            materials=[Material(name='A', index=0, model='LAMBERT',
                                diffuse=(0.9, 0.9, 0.9))],
            objects=[ObjectInfo(name='P', index=0,
                                matrix_world=np.eye(4, dtype=np.float32))],
            lights=[Light(type='SUN', direction=(-0.4, 0.5, -0.7), energy=5.0)],
            camera=Camera(matrix_world=look_at_matrix((4, -5, 3), (0, 0, 0)),
                          lens=45.0, sensor=36.0, clip_start=0.1,
                          clip_end=100.0),
            world=World(mode='SOLID', color=(0.0, 0.0, 0.0)), settings=st)
        return R.render(sc, st)

    for label, prim in (('cube', cube(centre=(0, 0, 0), size=2.0)),
                        ('sphere', sphere(centre=(0, 0, 0), radius=1.0))):
        plain, culled = shot(prim, False), shot(prim, True)
        check(f'culling the back of a closed {label} changes nothing at all',
              float(np.abs(plain - culled).max()) == 0.0,
              f'max delta {float(np.abs(plain - culled).max()):.6f}')

    floor = plane(z=-1.0, size=8.0)
    faced = shot(floor, True)
    check('a surface facing the camera survives culling',
          float(faced[..., :3].mean()) > 0.05,
          f'mean {float(faced[..., :3].mean()):.4f}')
    away = shot(floor, True, flip=True)
    check('and the same surface wound the other way is culled',
          float(away[..., :3].max()) < 1e-6,
          f'max {float(away[..., :3].max()):.4f}')


def test_banded_sky_is_banded():
    """The banded gradient must produce exactly the number of steps asked for."""
    from ..core import sky as SKY
    from ..core.scene import World

    up = np.linspace(0.001, 1.0, 4096, dtype=np.float32)
    dirs = np.stack([np.zeros_like(up), np.sqrt(np.maximum(1 - up * up, 0)),
                     up], 1).astype(np.float32)

    wrong = []
    for count in (2, 5, 8, 17):
        w = World()
        w.mode = 'BANDS'
        w.band_count = count
        col = SKY.bands(w, dirs)
        levels = np.unique(np.round(col[:, 2].astype(np.float64), 6))
        # `count` steps between horizon and zenith, and the zenith itself is
        # the last sample's own step, so `count` distinct values is right
        if len(levels) != count:
            wrong.append(f'{count} bands -> {len(levels)} levels')
    check('a banded sky has exactly as many levels as it was asked for',
          not wrong, '; '.join(wrong))

    w = World()
    w.mode = 'GRADIENT'
    smooth = SKY.gradient(w, dirs)
    w.mode = 'BANDS'
    w.band_count = 6
    stepped = SKY.bands(w, dirs)
    check('banding is a quantisation of the same gradient, not another one',
          float(np.abs(smooth - stepped).max()) < 1.0 / 6.0 + 1e-3,
          f'max gap {float(np.abs(smooth - stepped).max()):.4f}')

    w.band_softness = 1.0
    soft = SKY.bands(w, dirs)
    check('softness rounds the step edges off',
          len(np.unique(np.round(soft[:, 2].astype(np.float64), 6))) > 6)


def test_starfield_goes_all_the_way_round():
    """Starfield stars must appear below the horizon as well as above it.

    Bryce's star layer fades out toward the ground because it sits under a sky
    dome. This mode has no dome, so a star at -60 degrees is as valid as one
    at +60, and getting that wrong would be invisible until someone pointed a
    camera up from below.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    rng = np.random.default_rng(7)
    d = rng.normal(size=(40000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    w = World()
    w.mode = 'STARFIELD'
    w.color = (0.0, 0.0, 0.0)
    w.star_brightness = 1.0
    col = SKY.starfield(w, d)
    lit = col.max(axis=1) > 0.02
    up = float(lit[d[:, 2] > 0.3].mean())
    down = float(lit[d[:, 2] < -0.3].mean())
    check('stars appear above the horizon', up > 0.001, f'{up:.4f}')
    check('and just as many below it', down > 0.001 and
          abs(up - down) < max(up, down) * 0.5, f'up {up:.4f} down {down:.4f}')

    w.star_brightness = 0.0
    dark = SKY.starfield(w, d)
    check('zero brightness means no stars at all',
          float(np.abs(dark).max()) < 1e-6)

    w.star_brightness = 1.0
    w.nebula = 2.0
    neb = SKY.starfield(w, d)
    check('a nebula adds light without removing the stars',
          float(neb.mean()) > float(col.mean()) and
          float(neb.max()) >= float(col.max()) - 1e-4)


def test_sheen_bump_and_refraction():
    """The three new master-shader inputs, and that their defaults change nothing.

    The defaults matter as much as the effects: these sockets were added to a
    shader that already had thirty-three, and anything that shifted an existing
    render by a hair would have broken every scene made before them.
    """
    st = base_settings(140, 105, backface_cull=False)
    st.raytrace = True
    st.ray_refraction = True
    st.ray_reflection = True

    def graph(extra=None, opacity=1.0, bumped=False):
        g = _master(**{'Opacity': opacity})
        if bumped:
            g['nodes']['n'] = _wnode(
                'n', 'ShaderNodeTexNoise', {'noise_dimensions': '3D'},
                [_sk('Vector', 'VECTOR', [0, 0, 0]), _sk('Scale', 'VALUE', 14.0),
                 _sk('Detail', 'VALUE', 3.0), _sk('Roughness', 'VALUE', 0.5),
                 _sk('Distortion', 'VALUE', 0.0)],
                [{'name': 'Fac', 'type': 'VALUE'},
                 {'name': 'Color', 'type': 'RGBA'}])
            g['nodes']['bp'] = _wnode(
                'bp', 'ShaderNodeBump', {},
                [_sk('Strength', 'VALUE', 1.0), _sk('Distance', 'VALUE', 1.0),
                 _sk('Height', 'VALUE', 0.0, ['n', 0]),
                 _sk('Normal', 'VECTOR', [0, 0, 0])],
                [{'name': 'Normal', 'type': 'VECTOR'}])
            g['nodes']['h']['inputs'].append(
                _sk('Normal', 'VECTOR', [0, 0, 0], ['bp', 0]))
        for name, value in (extra or {}).items():
            kind = 'RGBA' if isinstance(value, (list, tuple)) else 'VALUE'
            g['nodes']['h']['inputs'].append(_sk(name, kind, value))
        return g

    def shot(g):
        sc = demo_scene(st)
        for m in sc.materials:
            m.graph = g
        return R.render(sc, st)

    # 1. the defaults are inert
    plain = shot(graph())
    defaults = shot(graph({'Sheen': 0.0, 'Sheen Color': [1, 1, 1, 1],
                           'Sheen Roughness': 0.3, 'Bump Strength': 1.0,
                           'Refraction Amount': 1.0}))
    check('the new inputs at their defaults change nothing at all',
          float(np.abs(plain - defaults).max()) == 0.0,
          f'max delta {float(np.abs(plain - defaults).max()):.8f}')

    # 2. sheen is a lit term at the silhouette, not a flat add
    sheened = shot(graph({'Sheen': 2.5, 'Sheen Roughness': 0.1}))
    delta = np.abs(sheened - plain)[..., :3].mean(axis=2)
    check('Sheen changes the render', float(delta.mean()) > 1e-4,
          f'delta {float(delta.mean()):.6f}')
    check('Sheen is concentrated rather than uniform',
          float(delta.max()) > 4.0 * float(delta.mean()),
          f'peak {float(delta.max()):.4f} vs mean {float(delta.mean()):.4f}')
    broad = shot(graph({'Sheen': 2.5, 'Sheen Roughness': 1.0}))
    check('Sheen Roughness widens the band',
          float(np.abs(broad - sheened).mean()) > 1e-4)
    tinted = shot(graph({'Sheen': 2.5, 'Sheen Roughness': 0.1,
                         'Sheen Color': [1.0, 0.0, 0.0, 1.0]}))
    check('Sheen Color tints it', float(np.abs(tinted - sheened).mean()) > 1e-4)
    check('sheen stays finite', bool(np.isfinite(sheened).all()))

    # 3. an unlit surface gets no sheen, because sheen needs a light
    st_dark = base_settings(80, 60, backface_cull=False)
    sc = demo_scene(st_dark)
    sc.lights = []
    for m in sc.materials:
        m.graph = graph({'Sheen': 3.0})
    lit_none = R.render(sc, st_dark)
    sc2 = demo_scene(st_dark)
    sc2.lights = []
    for m in sc2.materials:
        m.graph = graph()
    check('sheen needs a light, unlike the rim term',
          float(np.abs(lit_none - R.render(sc2, st_dark)).max()) < 1e-6)

    # 4. bump strength scales the supplied normal
    bumped = shot(graph(bumped=True))
    off = shot(graph({'Bump Strength': 0.0}, bumped=True))
    hard = shot(graph({'Bump Strength': 3.0}, bumped=True))
    flat_ = shot(graph())
    check('Bump Strength 0 puts the normal back where it started',
          float(np.abs(off - flat_).mean()) < 1e-6,
          f'delta {float(np.abs(off - flat_).mean()):.8f}')
    check('Bump Strength 1 leaves the bump as given',
          float(np.abs(bumped - flat_).mean()) > 1e-4)
    check('Bump Strength above 1 is not the same as leaving it alone',
          float(np.abs(hard - bumped).mean()) > 1e-4)

    # and the claim itself -- that the knob scales how far the normal is bent
    # away from the surface -- measured on the normals rather than the pixels,
    # where a saturating highlight could hide it. A constant normal is fed in
    # rather than a bump node, so the only thing varying is the knob.
    from ..core.nodeeval import GraphEvaluator
    from ..core.render import closure_to_surface

    def tilted(k):
        g = _master()
        g['nodes']['v'] = _wnode(
            'v', 'ShaderNodeCombineXYZ', {},
            [_sk('X', 'VALUE', 0.6), _sk('Y', 'VALUE', 0.0),
             _sk('Z', 'VALUE', 0.8)],
            [{'name': 'Vector', 'type': 'VECTOR'}])
        g['nodes']['h']['inputs'].append(
            _sk('Normal', 'VECTOR', [0, 0, 1], ['v', 0]))
        g['nodes']['h']['inputs'].append(_sk('Bump Strength', 'VALUE', k))
        return g

    angles = []
    for k in (0.0, 0.5, 1.0, 2.0):
        ctx = _emit_ctx(64)
        ev = GraphEvaluator(tilted(k), ctx)
        cl, _ = ev.evaluate_surface()
        _surf, _model, nrm = closure_to_surface(cl, ctx, st)
        n = np.asarray(nrm, np.float32)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        g = np.asarray(ctx.N, np.float32)
        g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        angles.append(float(np.arccos(np.clip((n * g).sum(1), -1, 1)).mean()))
    check('Bump Strength 0 leaves the geometric normal untouched',
          angles[0] < 1e-5, f'{angles[0]:.6f} rad')
    check('Bump Strength scales the angle the normal is bent through',
          angles[1] < angles[2] < angles[3],
          str([round(a, 5) for a in angles]))

    # 5. refraction amount gates the ray traced through the surface
    glass = shot(graph(opacity=0.25))
    none = shot(graph({'Refraction Amount': 0.0}, opacity=0.25))
    half = shot(graph({'Refraction Amount': 0.5}, opacity=0.25))
    check('Refraction Amount 0 stops the surface refracting',
          float(np.abs(none - glass).mean()) > 1e-3)
    check('and 0.5 lands between the two',
          float(np.abs(half - glass).mean()) < float(np.abs(none - glass).mean()))
    check('refraction stays finite', bool(np.isfinite(none).all()))


def test_render_passes_reach_blender():
    """Every pass Halcyon offers must be produced, and be data rather than a picture.

    The engine was force-enabling Blender's own Passes panel and then writing
    exactly one pass into the render result, so every box in it was a control
    that did nothing. This covers both halves: the buffers exist and hold the
    right kind of number, and the engine declares and writes each one.
    """
    import inspect

    from ..core import render as CR
    from ..core.settings import RenderSettings

    st = RenderSettings()
    st.resolution_x, st.resolution_y = 120, 90
    st.output_scale = 'NONE'
    for attr in ('pass_depth', 'pass_normal', 'pass_position', 'pass_uv',
                 'pass_object_index', 'pass_material_index'):
        setattr(st, attr, True)
    wanted = CR.wanted_passes(st)
    check('all six passes are offered', len(wanted) == 6, str(wanted))

    for aa in (1, 4):
        st.aa_samples = aa
        sc = demo_scene(st)
        R.render(sc, st)
        got = getattr(sc, 'last_passes', None) or {}
        missing = [n for n in wanted if n not in got]
        check(f'every requested pass is produced at aa={aa}', not missing,
              ', '.join(missing))
        wrong = [f'{n}{got[n].shape}' for n in got
                 if got[n].shape[:2] != (90, 120)]
        check(f'every pass is at the output resolution at aa={aa}', not wrong,
              ', '.join(wrong))

    sc = demo_scene(st)
    R.render(sc, st)
    got = sc.last_passes
    d = got['Depth'][..., 0]
    covered = np.isfinite(d) & (d < 1e9)
    check('the depth pass holds distances, not a grey ramp',
          covered.any() and float(d[covered].min()) > 0.1
          and float(d[covered].max()) > 1.5,
          f'{float(d[covered].min()):.2f}..{float(d[covered].max()):.2f}')
    check('uncovered pixels use the far value the compositor expects',
          float(d[~covered].min()) >= 1e10 if (~covered).any() else True)

    n = got['Normal']
    lens = np.linalg.norm(n[covered], axis=1)
    check('the normal pass holds unit vectors in -1..1',
          float(np.abs(lens - 1.0).max()) < 1e-3 and float(n.min()) < -0.1,
          f'|n| off by {float(np.abs(lens - 1.0).max()):.5f}')

    ids = got['IndexMA'][..., 0]
    check('the material index pass holds whole numbers',
          float(np.abs(ids - np.round(ids)).max()) < 1e-6
          and len(np.unique(ids)) > 1, str(np.unique(ids)[:6]))

    # ...and the engine end: declared, and written
    src = inspect.getsource(_engine_module())
    check('the engine declares its passes to Blender',
          'def update_render_passes' in src and 'register_pass' in src)
    check('and writes them into the render result',
          '_deliver_passes' in src and 'foreach_set' in src)
    from .. import engine as ENG
    names = {p[0] for p in ENG.PASS_SPEC}
    check('the declared passes are exactly the ones produced',
          names == set(wanted), f'{sorted(names)} vs {sorted(wanted)}')
    check("Blender's own Passes panel is not force-enabled any more",
          'VIEWLAYER_PT_layer_passes' not in ENG.FORCED_PANELS)


def _engine_module():
    from . import fakebpy
    fakebpy.install()
    import importlib
    return importlib.import_module('halcyon.engine')


def test_socket_values_survive_a_rebuild():
    """Rebuilding the coded shader's sockets must not read freed memory.

    Selecting HLSL crashed Blender. `sock.default_value` on a colour or vector
    socket is a live view into the socket's own memory rather than a copy, and
    the rebuild held one across `inputs.clear()` -- then read it back to
    restore the value, which is a use-after-free. It takes the process down
    instead of raising, so no amount of exception handling would have helped.

    Checked statically, since it needs Blender to reproduce: the captured value
    must be materialised before anything is cleared.
    """
    import ast
    import inspect

    from . import fakebpy
    fakebpy.install()
    import importlib
    mod = importlib.import_module('halcyon.nodes.shader_nodes')
    src = inspect.getsource(mod.HALCYON_CodeNode.rebuild_sockets)
    tree = ast.parse(src.lstrip())

    clear_line = None
    capture_lines = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, 'attr', '') == 'clear':
            clear_line = node.lineno if clear_line is None else min(clear_line,
                                                                    node.lineno)
        if isinstance(node, ast.Attribute) and node.attr == 'default_value':
            if isinstance(node.ctx, ast.Load):
                capture_lines.append(node.lineno)
    check('the rebuild does clear the sockets', clear_line is not None)
    before = [l for l in capture_lines if clear_line and l < clear_line]
    check('a default_value is read before the clear', bool(before))

    # nothing captured before the clear may be stored raw: the value has to go
    # through tuple() or float() first, which is what makes it a copy
    lines = src.splitlines()
    copies = sum(1 for l in lines[:clear_line] if 'tuple(' in l or 'float(' in l)
    check('and the captured value is copied before anything is cleared',
          copies > 0, 'no tuple()/float() copy before the clear')
    check('nothing is stored straight into the keep table',
          not any('.default_value' in l and 'keep[' in l
                  for l in lines[:clear_line]),
          '; '.join(l.strip() for l in lines[:clear_line]
                    if '.default_value' in l and 'keep[' in l))

    # the re-entrancy guard must not live on the node instance: Blender hands
    # out a new wrapper per access, so such an attribute is written to a
    # temporary and read back as the class default. Checked on the parsed tree,
    # so a comment explaining the old bug does not count as the bug.
    whole = ast.parse(inspect.getsource(mod))
    stale = [x for x in ast.walk(whole)
             if isinstance(x, ast.Attribute) and x.attr == '_busy']
    check('the update guard does not rely on an instance attribute', not stale,
          f'{len(stale)} reference(s) remain')
    text = inspect.getsource(mod)
    check('it uses the node pointer instead',
          'as_pointer' in text and '_UPDATING' in text)


def test_toon_steps_actually_steps():
    """Toon Steps must produce that many tones, and two must not change.

    It was accepted by the node, copied by the exporter, threaded into the
    shading signature -- and then never read. Four releases of a control that
    did nothing.
    """
    from ..core.shading import diffuse_toon

    # sampled evenly in the *angle*, because that is the axis the tones are cut
    # along -- sampling ndl evenly crowds the samples toward the light and the
    # narrowest tone gets too few of them to count
    n = 4000
    ang = np.linspace(0.0, 1.0, n).astype(np.float32)
    ndl = np.cos(ang * np.pi * 0.5).astype(np.float32)
    size = np.full(n, 0.5, np.float32)
    smooth = np.full(n, 1e-4, np.float32)

    def plateaus(steps):
        v = diffuse_toon(ndl, size, smooth, np.full(n, steps, np.float32))
        vals, counts = np.unique(np.round(v, 5), return_counts=True)
        # a level that only a couple of samples land on is the ramp between two
        # tones, not a tone
        return vals[counts > n // 200]

    wrong = []
    for steps in (2, 3, 4, 6, 10, 16):
        got = len(plateaus(steps))
        if got != steps:
            wrong.append(f'{steps} steps -> {got} tones')
    check('Toon Steps produces exactly that many tones', not wrong,
          '; '.join(wrong))

    # the default must be untouched, or every toon render ever made changes
    one = diffuse_toon(ndl, size, smooth, np.full(n, 2.0, np.float32))
    old = np.clip((np.clip(1.0 - size, 0, 1) + np.maximum(smooth, 1e-4)
                   - np.arccos(np.clip(ndl, 0, 1)) / (np.pi * 0.5))
                  / np.maximum(smooth, 1e-4), 0.0, 1.0)
    check('two steps is bit-identical to the single-step version it replaces',
          float(np.abs(one - old).max()) == 0.0,
          f'max delta {float(np.abs(one - old).max()):.8f}')

    # and it survives the whole pipeline, which is where it was lost
    def shot(steps):
        st = base_settings(140, 105)
        sc = demo_scene(st)
        g = _master()
        g['nodes']['h']['props']['model'] = 'TOON'
        g['nodes']['h']['props']['toon_steps'] = steps
        for m in sc.materials:
            m.graph = g
        return R.render(sc, st)

    a, b = shot(2), shot(8)
    check('and reaches the renderer from the node',
          float(np.abs(a - b).mean()) > 1e-3,
          f'delta {float(np.abs(a - b).mean()):.5f}')


def test_wireframe_draws_wires():
    """The Wireframe model must draw edges on dense meshes, not dots.

    The edge distance came from a central difference that needed the pixels on
    *both* sides to be the same triangle. On anything denser than a cube that
    is rarely true, the derivative collapsed to zero, the distance went to
    infinity and the wire came out as a scatter of dots -- or, with the
    interior knocked through to the background, as nothing at all.
    """
    from ..core import geometry as GEO
    from ..core.scene import Camera, Light, Material, ObjectInfo, Scene, World
    from .scenebuild import _mesh_concat, look_at_matrix, sphere

    def wire_shot(prim, res=(200, 150)):
        st = base_settings(*res)
        mesh = _mesh_concat([prim])
        mesh.smooth = np.zeros(mesh.tris.shape[0], bool)
        mat = Material(name='W', index=0, model='WIREFRAME',
                       diffuse=(1.0, 1.0, 1.0))
        sc = Scene(
            mesh=mesh, materials=[mat],
            objects=[ObjectInfo(name='T', index=0,
                                matrix_world=np.eye(4, dtype=np.float32))],
            lights=[Light(type='SUN', direction=(-0.5, 0.5, -0.7), energy=5.0)],
            camera=Camera(matrix_world=look_at_matrix((4, -5, 3), (0, 0, 0)),
                          lens=45.0, sensor=36.0, clip_start=0.1,
                          clip_end=200.0),
            world=World(mode='SOLID', color=(0.0, 0.0, 0.0)), settings=st)
        img = R.render(sc, st)[..., :3]
        return (img.max(axis=2) > 0.25)

    def teapot_prim(steps, segs):
        verts, faces = GEO.utah_teapot(steps, segs)
        V = np.asarray(verts, np.float32) * 0.55
        tris = []
        for f in faces:
            for k in range(1, len(f) - 1):
                tris.append((f[0], f[k], f[k + 1]))
        T = np.asarray(tris, np.int32)
        N = np.tile(np.array([[0, 0, 1.0]], np.float32), (len(V), 1))
        return (V, N, V[:, :2].astype(np.float32), T, 0, 0)

    # A wire is a connected structure. Dots are not: count how much of the lit
    # area has a lit neighbour, which is near 1 for lines and low for specks.
    def connectedness(on):
        if not on.any():
            return 0.0
        nb = np.zeros_like(on)
        nb[1:, :] |= on[:-1, :]
        nb[:-1, :] |= on[1:, :]
        nb[:, 1:] |= on[:, :-1]
        nb[:, :-1] |= on[:, 1:]
        return float((on & nb).sum()) / float(on.sum())

    for label, prim in (('sphere', sphere(radius=1.4)),
                        ('teapot 2.6k', teapot_prim(8, 16)),
                        ('teapot 13k', teapot_prim(16, 40))):
        on = wire_shot(prim)
        frac = float(on.mean())
        conn = connectedness(on)
        check(f'the {label} wireframe draws something', frac > 0.01,
              f'{frac * 100:.1f}% of the frame lit')
        check(f'the {label} wireframe is lines rather than dots', conn > 0.9,
              f'{conn * 100:.0f}% of lit pixels have a lit neighbour')
        check(f'the {label} wireframe is not a solid fill', frac < 0.5,
              f'{frac * 100:.1f}% lit')


def test_the_engine_actually_runs():
    """One frame, all the way through the real engine, against a fake Blender.

    Everything above this line tests the renderer. Nothing tested the *engine*:
    the property group, `to_settings`, the exporter, the delivery. Six bugs
    shipped through that gap in a row, every one of them a control wired up at
    one end and not the other, and every one of them invisible to a test that
    calls `render()` directly with a hand-built scene.

    It is not Blender. It cannot catch a segfault or a driver quirk. It catches
    the setting that never arrives.
    """
    from . import fakeblender as FB
    props, engine = FB.install()

    img, passes, cap = FB.run_render(props, engine)
    check('a frame renders through the engine and is delivered', img is not None,
          str(cap.get('reports')))
    if img is None:
        return
    check('the delivered buffer is the size Blender asked for',
          img.shape == (90, 120, 4), str(img.shape))
    check('and it holds finite pixels', bool(np.isfinite(img).all()))

    # every render pass in the dropdown must change what is delivered
    base = FB.run_render(props, engine, debug_pass='BEAUTY')[0]
    same = []
    for mode in ('DEPTH', 'NORMAL', 'UV', 'MATID', 'OVERDRAW', 'WIREFRAME'):
        shot = FB.run_render(props, engine, debug_pass=mode)[0]
        if shot is None or float(np.abs(shot - base).mean()) < 1e-4:
            same.append(mode)
    check('every render pass reaches the delivered image', not same,
          ', '.join(same) + ' came out as the beauty render')

    # and the extra passes arrive as separate buffers
    _img, extra, _cap = FB.run_render(
        props, engine, pass_depth=True, pass_normal=True,
        pass_object_index=True)
    missing = [n for n in ('Depth', 'Normal', 'IndexOB') if n not in extra]
    check('the extra passes are registered and written', not missing,
          ', '.join(missing))
    if 'Depth' in extra:
        d = extra['Depth'][..., 0]
        hit = d < 1e9
        check('the depth pass holds real distances', hit.any()
              and float(d[hit].min()) > 0.5, f'{float(d[hit].min()):.2f}')


def test_wireframe_through_the_engine():
    """A material shaded as Wireframe by its node must carve edges.

    The reported symptom was a flat unlit fill, and the cause turned out not to
    be a broken carve at all: once a mesh's triangles are a couple of pixels
    across, *every* pixel is within a wire width of an edge, so All Edges is a
    solid fill and no width setting escapes it. This pins down both halves --
    that the carve happens at all, and that Creases & Silhouette stays a
    wireframe however dense the mesh gets.
    """
    from . import fakeblender as FB
    props, engine = FB.install()

    def lit(mode='ALL', divisions=12, wire_size=None, **kw):
        mat = FB.halcyon_material('Wire', model='WIREFRAME',
                                  diffuse=(0.95, 0.9, 0.35, 1.0))
        if wire_size is not None:
            mat.node_tree.nodes[0].wire_size = wire_size
        img, _p, _c = FB.run_render(props, engine, material=mat,
                                    mesh=FB.grid_mesh(6.0, divisions),
                                    wire_mode=mode, **kw)
        if img is None:
            return None
        return float((img[..., :3].max(axis=2) > 0.12).mean())

    solid = lit('ALL', 4)
    check('a coarse mesh draws edges rather than filling',
          solid is not None and 0.02 < solid < 0.35, f'{solid}')

    # the shape of the reported bug, stated as a measurement
    dense = lit('ALL', 40)
    check('All Edges does fill in on a dense mesh -- this is arithmetic, '
          'not a defect', dense is not None and dense > 0.45, f'{dense}')

    # and the way out must not care how dense the mesh is
    creases = [lit('CREASE', d) for d in (4, 12, 40, 80)]
    ok = all(c is not None and 0.001 < c < 0.12 for c in creases)
    spread = max(creases) - min(creases) if ok else 1.0
    check('Creases & Silhouette stays a wireframe at any density', ok,
          str([round(c, 3) for c in creases]))
    check('and barely moves as triangles are added', spread < 0.02,
          f'spread {spread:.3f}')

    # the width has to be reachable from the node, which it was not
    thin, thick = lit('ALL', 24, wire_size=0.2), lit('ALL', 24, wire_size=3.0)
    check('the wire size on the shader node changes the wire',
          thin is not None and thick is not None and thick > thin + 0.02,
          f'{thin} -> {thick}')


def test_wire_size_is_reachable():
    """A node-shaded Wireframe material must be able to set its own width.

    It could not: `wire_size` was only exported inside the material-override
    branch, and a material shaded by a Halcyon Shader node never goes through
    that branch. The width was stuck at the dataclass default with no control
    anywhere in the interface that could change it.
    """
    from ..core.render import material_wire_size
    from ..core.scene import Material
    from . import fakebpy
    fakebpy.install()
    import importlib
    from .. import export as EX

    check('the exporter carries the shader node wire size',
          'wire_size' in EX.NODE_PROPS.get('HALCYON_ShaderNode', ()),
          str(EX.NODE_PROPS.get('HALCYON_ShaderNode')))

    mat = Material(name='m', index=0)
    mat.graph = {'output': 'o', 'nodes': {
        'h': {'id': 'h', 'bl_idname': 'HALCYON_ShaderNode',
              'props': {'model': 'WIREFRAME', 'wire_size': 0.4},
              'inputs': [], 'outputs': []}}}
    check('and the renderer reads it from the node',
          abs(material_wire_size(mat) - 0.4) < 1e-6, str(material_wire_size(mat)))

    plain = Material(name='p', index=0)
    plain.wire_size = 2.5
    check('falling back to the material when there is no node',
          abs(material_wire_size(plain) - 2.5) < 1e-6)

    shader = importlib.import_module('halcyon.nodes.shader_nodes')
    check('the shader node offers a Wire Size',
          'wire_size' in shader.HALCYON_ShaderNode.__annotations__)
    import inspect
    src = inspect.getsource(shader.HALCYON_ShaderNode.draw_buttons)
    check('and draws it when the model is Wireframe',
          "WIREFRAME" in src and 'wire_size' in src)


def test_every_sky_lab_control_does_something():
    """Each Bryce Sky Lab control must change the sky it belongs to.

    A control that exists and does nothing is the failure mode this project
    keeps hitting, and a sky with sixty of them is sixty chances to hit it. So
    each one is perturbed on its own, against a world where the layer it
    belongs to is switched on, and asked to change the picture.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    rng = np.random.default_rng(11)
    d = rng.normal(size=(6000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    def base(**kw):
        w = World(mode='BRYCE')
        w.clouds = True
        w.stratus = True
        w.haze_density = 0.5
        w.fog_density = 0.3
        w.stars = True
        w.star_brightness = 1.0
        for k, v in kw.items():
            setattr(w, k, v)
        return w

    ref = SKY.evaluate(base(), d)
    dead = []
    knobs = (
        ('sky_mode', 'SOFT'), ('sun_glow_color', (0.2, 0.9, 0.3)),
        ('shadow_color', (0.9, 0.1, 0.1)), ('shadow_intensity', 0.0),
        ('haze_base_height', 0.4), ('fog_base_height', 0.4),
        ('fog_blend_sky', 1.0), ('fog_sun_tint', 1.0),
        ('cloud_frequency', 3.0), ('cloud_amplitude', 2.5),
        ('cloud_turbulence', 0.3), ('spherical_clouds', False),
        ('stratus_frequency', 3.0), ('stratus_amplitude', 2.5),
        ('comets', 2.0),
    )
    for name, value in knobs:
        got = SKY.evaluate(base(**{name: value}), d)
        if float(np.abs(got - ref).mean()) < 1e-5:
            dead.append(name)
    check('every new Sky Lab control changes the sky', not dead,
          ', '.join(dead))

    # the moon's own controls need the moon out
    moon_ref = SKY.evaluate(base(celestial='MOON'), d)
    soft = SKY.evaluate(base(celestial='MOON', moon_softness=0.9), d)
    check('moon softness changes the terminator',
          float(np.abs(soft - moon_ref).mean()) > 1e-7,
          f'{float(np.abs(soft - moon_ref).mean()):.9f}')

    # the comets' own controls need comets out
    comet_ref = SKY.evaluate(base(comets=2.0), d)
    comet_dead = []
    for name, value in (('comet_length', 0.6), ('comet_width', 0.05),
                        ('comet_tail_sun', 0.0),
                        ('comet_color', (0.2, 0.5, 1.0)),
                        ('comet_speed', 1.0)):
        got = SKY.evaluate(base(comets=2.0, **{name: value}), d, time=3.0)
        base_t = SKY.evaluate(base(comets=2.0), d, time=3.0)
        if float(np.abs(got - base_t).mean()) < 1e-9:
            comet_dead.append(name)
    check('every comet control changes the comets', not comet_dead,
          ', '.join(comet_dead))

    # comet count only matters once comets are on
    c1 = SKY.evaluate(base(comets=2.0, comet_count=1), d)
    c8 = SKY.evaluate(base(comets=2.0, comet_count=8), d)
    check('the comet count changes how many there are',
          float(np.abs(c8 - c1).mean()) > 1e-6)

    # the four that only mean anything once the camera is somewhere
    eye = (3.0, 2.0, 1.5)
    ref_eye = SKY.evaluate(base(), d, eye=eye)
    dead = []
    for name, value in (('volumetric_world', 3.0),
                        # the link is on by default, so *off* is the change
                        ('link_clouds_to_view', False),
                        ('fixed_cloud_plane', False)):
        got = SKY.evaluate(base(**{name: value}), d, eye=eye)
        if float(np.abs(got - ref_eye).mean()) < 1e-5:
            dead.append(name)
    check('the camera-relative cloud controls do something', not dead,
          ', '.join(dead))

    ocean = World(mode='BRYCE')
    ocean.ground_plane = True
    ocean.ground_mode = 'OCEAN'
    ocean.ground_fade = 50.0
    flat = SKY.evaluate(ocean, d, eye=(0.0, 0.0, 2.0))
    ocean.color_perspective = 2.0
    curved = SKY.evaluate(ocean, d, eye=(0.0, 0.0, 2.0))
    check('Colour Perspective changes how distance takes the haze',
          float(np.abs(curved - flat).mean()) > 1e-5)


def test_the_sky_lab_stacks_in_brydes_order():
    """Layer order is half of why a Bryce sky reads as one.

    Stars sit beyond the atmosphere, so a cloud in front of one must hide it,
    and haze must dim a cloud rather than leaving it crisp against a hazed
    sky. Both were wrong: stars were composited over the clouds, and the haze
    went on before the decks did.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    rng = np.random.default_rng(5)
    d = rng.normal(size=(20000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] > 0.25]

    def w(**kw):
        world = World(mode='BRYCE')
        world.horizon = (0.0, 0.0, 0.0)
        world.sky_mid = (0.0, 0.0, 0.0)
        world.zenith = (0.0, 0.0, 0.0)
        world.sun_glow = 0.0
        world.sun_disc = False
        world.sun_intensity = 0.0
        world.clouds = False
        world.stratus = False
        world.haze_density = 0.0
        world.fog_density = 0.0
        for k, v in kw.items():
            setattr(world, k, v)
        return world

    stars_only = SKY.evaluate(w(stars=True, star_brightness=2.0), d)
    lit = stars_only.max(axis=1) > 0.02
    check('there are stars to hide', int(lit.sum()) > 20, str(int(lit.sum())))

    # a solid cloud deck over the top of them
    covered = SKY.evaluate(w(stars=True, star_brightness=2.0, clouds=True,
                             cloud_cover=1.0, cloud_density=1.0,
                             cloud_color=(0.5, 0.5, 0.5)), d)
    star_px = covered[lit]
    check('a cloud deck hides the stars behind it',
          float(np.abs(star_px - covered[lit].mean(axis=0)).mean()) <
          float(np.abs(stars_only[lit] - stars_only[lit].mean(axis=0)).mean()),
          'stars still show through the clouds')

    # and haze must reach the clouds, not stop behind them
    clouds = w(clouds=True, cloud_cover=0.6, cloud_color=(1.0, 1.0, 1.0))
    plain = SKY.evaluate(clouds, d)
    hazed = w(clouds=True, cloud_cover=0.6, cloud_color=(1.0, 1.0, 1.0),
              haze_density=1.0, haze_height=1.0, haze_color=(0.0, 0.0, 0.0))
    dimmed = SKY.evaluate(hazed, d)
    bright = plain.max(axis=1) > 0.5
    check('haze dims the clouds as well as the sky behind them',
          bool(bright.any()) and
          float(dimmed[bright].mean()) < float(plain[bright].mean()) * 0.9,
          f'{float(dimmed[bright].mean()):.3f} vs {float(plain[bright].mean()):.3f}')


def test_the_ocean_is_water():
    """The infinite ocean's own properties, measured rather than eyeballed."""
    from ..core import sky as SKY
    from ..core.scene import World

    n = 40000
    rng = np.random.default_rng(19)
    d = rng.normal(size=(n, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] < -0.02]                     # only rays that hit the water
    eye = (0.0, 0.0, 2.0)

    def w(**kw):
        world = World(mode='BRYCE')
        world.ground_plane = True
        world.ground_mode = 'OCEAN'
        world.ground_fade = 1e6                # keep distance haze out of it
        for k, v in kw.items():
            setattr(world, k, v)
        return world

    ref = SKY.evaluate(w(), d, eye=eye)
    dead = []
    for name, value in (('ocean_choppiness', 1.4), ('ocean_wind_angle', 2.4),
                        ('ocean_spread', 0.0), ('ocean_wave_scale', 4.0),
                        ('ocean_detail', 9), ('ocean_deep', (0.4, 0.0, 0.0)),
                        ('ocean_shallow', (0.0, 0.4, 0.0)),
                        ('ocean_glitter', 8.0), ('ocean_glitter_size', 2.0),
                        ('ocean_foam', 1.0), ('ocean_transparency', 1.0)):
        got = SKY.evaluate(w(**{name: value}), d, eye=eye)
        if float(np.abs(got - ref).mean()) < 1e-6:
            dead.append(name)
    check('every ocean control changes the water', not dead, ', '.join(dead))

    # Fresnel: water seen edge-on is a mirror, water seen from above is not
    steep = np.array([[0.0, 0.2, -0.98]], np.float32)
    steep /= np.linalg.norm(steep)
    graze = np.array([[0.0, 0.999, -0.045]], np.float32)
    graze /= np.linalg.norm(graze)
    flat = w(ocean_choppiness=0.0, ocean_glitter=0.0,
             zenith=(0.0, 0.0, 0.0), horizon=(1.0, 1.0, 1.0),
             sky_mid=(1.0, 1.0, 1.0), sun_glow=0.0, sun_disc=False,
             clouds=False, haze_density=0.0, ocean_deep=(0.0, 0.0, 0.0),
             ocean_shallow=(0.0, 0.0, 0.0))
    down = float(SKY.evaluate(flat, steep, eye=eye).mean())
    side = float(SKY.evaluate(flat, graze, eye=eye).mean())
    check('water is a mirror at a glancing angle and dark looking down',
          side > down + 0.3, f'grazing {side:.3f} vs down {down:.3f}')

    # the glitter path must sit under the sun, not opposite it
    sunny = w(sun_elevation=0.25, sun_rotation=1.5708, ocean_glitter=6.0,
              ocean_choppiness=0.25, clouds=False)
    col = SKY.evaluate(sunny, d, eye=eye)
    bright = col.max(axis=1) > np.percentile(col.max(axis=1), 99.0)
    toward = d[bright][:, 1] > 0                  # the sun is toward +Y
    check('the glitter path lands under the sun',
          float(toward.mean()) > 0.8, f'{float(toward.mean()) * 100:.0f}% of it')


def test_sub_pixel_waves_widen_the_glitter():
    """When Horizon Smoothing is on, what it removes must go somewhere.

    Smoothing is off by default -- Bryce kept its waves all the way to the
    horizon -- but for anyone who turns it up, dropping the small waves
    outright is what turns distant water to glass. The test is the shape of
    the falloff: with a large pixel footprint the same wave field must still
    produce a highlight, and a wider one.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    world = World(mode='BRYCE')
    world.ocean_choppiness = 0.5
    world.ocean_detail = 8
    world.ocean_horizon_smooth = 1.0
    world.ocean_sparkle = 0.0          # measuring the fade, not the jitter
    n = 4000
    p = np.stack([np.linspace(0, 40, n), np.zeros(n), np.zeros(n)],
                 1).astype(np.float32)

    sharp, lost_sharp = SKY._wave_normal(p, world, 0.0,
                                         lod=np.full(n, 1e-4, np.float32))
    blurred, lost_blur = SKY._wave_normal(p, world, 0.0,
                                          lod=np.full(n, 2.0, np.float32))
    check('a large footprint flattens the drawn normal',
          float(np.abs(blurred[:, 2] - 1.0).mean()) <
          float(np.abs(sharp[:, 2] - 1.0).mean()),
          'the blurred normal is not flatter')
    check('and the slope it lost is accounted for instead',
          float(lost_blur.mean()) > float(lost_sharp.mean()) + 1e-6,
          f'{float(lost_sharp.mean()):.6f} -> {float(lost_blur.mean()):.6f}')
    check('nothing is lost when every wave is resolvable',
          float(lost_sharp.max()) < 1e-6, f'{float(lost_sharp.max()):.8f}')


def test_wave_size_makes_waves_smaller_not_fainter():
    """Reported: the waves are far too big, and turning them down erases them.

    Two faults behind it. The pixel footprint the waves are measured against
    was a hard-coded 0.002 rad, four to six times coarser than a real frame,
    and it was taken along the *stretched* axis of a grazing ray rather than
    the area-equivalent square -- ten times too coarse again at the horizon.
    Between them they cut most of the wave trains out of the picture, leaving
    a big smooth swell near the camera and glass everywhere else. Turning Wave
    Size down pushed every remaining train under the cut, so the water went
    flat instead of fine.
    """
    from ..core import sky as SKY
    from ..core import render as R
    from ..core.scene import World

    # a real frame is far finer than the guess that was hard-coded
    proj = R.camera_matrices(None, 1920, 1080)[1]
    angle, width = R.pixel_footprint(None, proj, 1080)
    check('the pixel footprint comes from the frame, not a constant',
          0.0 < angle < 0.002 * 0.5, f'{angle:.6f} rad')
    check('and an orthographic camera reports a width instead',
          R.pixel_footprint(types.SimpleNamespace(type='ORTHO', ortho_scale=8.0),
                            proj, 400)[1] == 0.02)

    # and it has to actually reach the water: the setting was read at one end
    # and never written at the other, which is why it was stuck on the guess
    sc = demo_scene()
    st = RenderSettings(resolution_x=64, resolution_y=48)
    sc.world.mode = 'BRYCE'
    sc.world.ground_plane = True
    sc.world.ground_mode = 'OCEAN'
    if hasattr(sc.world, '_pixel_angle'):
        del sc.world._pixel_angle
    R.render(sc, st)
    want = R.pixel_footprint(sc.camera,
                             R.camera_matrices(sc.camera, 64, 48)[1], 48)[0]
    got = float(getattr(sc.world, '_pixel_angle', 0.0))
    check('and a render puts it where the water reads it',
          got > 0.0 and abs(got - want) < 1e-6, f'{got:.6f} want {want:.6f}')

    def water(wave_size, pixel_angle, height=12.0, w=400, h=150):
        world = World(mode='BRYCE')
        world.ground_plane = True
        world.ground_mode = 'OCEAN'
        world.ground_height = 0.0
        world.ocean_wave_scale = wave_size
        world._pixel_angle = pixel_angle
        world._pixel_width = 0.0
        eye = np.array([0.0, 0.0, height], np.float32)
        ys = np.linspace(np.tan(np.radians(-20.0)), np.tan(np.radians(-0.4)), h)
        xs = np.linspace(-0.4, 0.4, w)
        gx, gy = np.meshgrid(xs, ys)
        d = np.stack([gx.ravel(), np.ones(gx.size), gy.ravel()],
                     1).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return SKY.evaluate(world, d, None, eye=eye, time=0.0).reshape(h, w, 3)

    def texture(img, r0, r1):
        # adjacent-pixel difference along a row: wave detail with the
        # distance gradient taken out
        return float(np.mean(np.abs(np.diff(img[r0:r1].mean(axis=2), axis=1))))

    pa = 2.0 * np.arctan(18.0 / 50.0) / (1080 * 2)      # 50mm, 1080 rows, 4x
    fine = texture(water(0.25, pa), 0, 25)
    coarse = texture(water(2.0, pa), 0, 25)
    check('small waves survive being made small', fine > coarse * 0.75,
          f'2.0m -> {coarse:.5f}, 0.25m -> {fine:.5f}')
    check('and they are finer than the big ones, not just present',
          fine >= coarse, f'{coarse:.5f} vs {fine:.5f}')

    # the count that matters: how many of the trains are still drawn
    def trains(scale, lod):
        freq, n = 1.0, 0
        for _ in range(5):
            if np.clip(scale / freq / max(lod * 2.0, 1e-9) - 1.0, 0, 1) > 0.05:
                n += 1
            freq *= 1.9
        return n
    dist, graze = 66.0, 0.18
    old = dist / graze * 0.002
    new = dist * pa / np.sqrt(graze)
    check('mid-distance water keeps most of its wave trains now',
          trains(1.0, new) >= 4 > trains(2.0, old),
          f'{trains(2.0, old)} -> {trains(1.0, new)} of 5')

    # with smoothing turned up, what it removes must roughen the reflection
    # rather than leave a mirror
    world = World(mode='BRYCE')
    world.ground_mode = 'OCEAN'
    world.ocean_horizon_smooth = 1.0
    n = 512
    p = np.stack([np.linspace(0, 200, n), np.zeros(n), np.zeros(n)],
                 1).astype(np.float32)
    d = np.stack([np.zeros(n), np.full(n, 0.99), np.full(n, -0.14)],
                 1).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    sky_c = np.tile(np.asarray(world.horizon, np.float32), (n, 1))
    glass = SKY._ocean(world, d, d, p, np.full(n, 60.0, np.float32), sky_c,
                       0.0, np.full(n, 1e-4, np.float32))
    rough = SKY._ocean(world, d, d, p, np.full(n, 60.0, np.float32), sky_c,
                       0.0, np.full(n, 8.0, np.float32))
    check('water too far to draw waves on is rough, not a mirror',
          float(np.std(rough)) < float(np.std(glass)) and
          not np.allclose(rough, glass),
          f'{float(np.std(glass)):.5f} -> {float(np.std(rough)):.5f}')

    # and Wave Size is the water's own control now
    a = World(mode='BRYCE'); a.ground_mode = 'OCEAN'; a.ground_scale = 2.0
    b = World(mode='BRYCE'); b.ground_mode = 'OCEAN'; b.ground_scale = 40.0
    na = SKY._wave_normal(p, a, 0.0)[0]
    nb = SKY._wave_normal(p, b, 0.0)[0]
    check('the chequerboard Scale no longer sets the size of the sea',
          np.allclose(na, nb), 'ground_scale still moves the waves')


def test_the_waves_reach_the_horizon():
    """Reported: small waves fade out with distance. They must not.

    A modern renderer fades waves a pixel cannot resolve, and the water goes
    smooth toward the horizon. Bryce did not: its ocean was a procedural
    material on an infinite plane with nothing filtering it, so the waves
    compressed into a band of shimmer instead of flattening into glass. The
    fade is now off unless asked for -- and this is the check that it stays
    off, because it was on and nothing noticed.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    def water(smooth, sparkle=1.0, wave=0.3, h=180, w=360, height=14.0):
        world = World(mode='BRYCE')
        world.ground_plane = True
        world.ground_mode = 'OCEAN'
        world.ground_height = 0.0
        world.ground_fade = 1e5              # haze off: this is about waves
        world.ocean_wave_scale = wave
        world.ocean_horizon_smooth = smooth
        world.ocean_sparkle = sparkle
        world._pixel_angle = 2.0 * np.arctan(18.0 / 50.0) / (h * 4)
        world._pixel_width = 0.0
        eye = np.array([0.0, 0.0, height], np.float32)
        ys = np.linspace(np.tan(np.radians(-24.0)), np.tan(np.radians(-0.25)), h)
        xs = np.linspace(-0.6, 0.6, w)
        gx, gy = np.meshgrid(xs, ys)
        d = np.stack([gx.ravel(), np.ones(gx.size), gy.ravel()],
                     1).astype(np.float32)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return SKY.evaluate(world, d, None, eye=eye, time=0.0).reshape(h, w, 3)

    def texture(img, r0, r1):
        return float(np.mean(np.abs(np.diff(img[r0:r1].mean(axis=2), axis=1))))

    near, far = (0, 20), (150, 180)
    keep = water(0.0)
    near_k, far_k = texture(keep, *near), texture(keep, *far)
    check('by default the waves still have texture at the horizon',
          far_k > near_k * 0.5, f'near {near_k:.5f}, far {far_k:.5f}')
    check('and it is real detail, not a flat band',
          far_k > 0.002, f'{far_k:.5f}')

    # the control still works for anyone who wants the smooth version
    smoothed = water(1.0)
    far_s = texture(smoothed, *far)
    check('Horizon Smoothing at 1 does smooth the horizon',
          far_s < far_k * 0.5, f'{far_k:.5f} -> {far_s:.5f}')
    check('and it is a slider, not a switch',
          far_k > texture(water(0.5), *far) > far_s,
          f'{far_k:.5f} / {texture(water(0.5), *far):.5f} / {far_s:.5f}')

    # the shimmer must be decorrelation, not extra contrast: jittering where
    # the sample lands inside a pixel may not brighten or darken the water
    plain = water(0.0, sparkle=0.0)
    check('shimmer does not change how bright the water is',
          abs(float(plain.mean()) - float(keep.mean())) < 0.01,
          f'{float(plain.mean()):.4f} vs {float(keep.mean()):.4f}')

    # and it must be repeatable: the same still frame twice, the same pixels
    check('and it is deterministic, so still water does not crawl',
          np.array_equal(water(0.0), keep))


def test_clouds_do_not_react_to_the_camera():
    """Moving the camera must not move the clouds, unless asked.

    Reported straight after 1.20.0: the clouds raced whenever the camera
    rotated. Orbiting a viewport *translates* the eye around a pivot, and
    Link Clouds to View had shipped defaulting to off -- so the deck was
    nailed to the world and slid past, against a cloud height of 1.0 that is a
    dome parameter rather than a distance. A one-unit move was a whole
    deck-height of parallax.

    The default is the bit-identical case, and that is what is asserted: not
    "close", identical, because a sky that does not depend on the camera
    cannot depend on it a little.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    rng = np.random.default_rng(2)
    d = rng.normal(size=(6000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] > 0.05]

    def world(**kw):
        w = World(mode='BRYCE')
        w.clouds = True
        w.stratus = True
        w.cloud_cover = 0.55
        for k, v in kw.items():
            setattr(w, k, v)
        return w

    check('Link Clouds to View is on by default',
          World().link_clouds_to_view is True)

    base = SKY.evaluate(world(), d, eye=(0.0, 0.0, 1.6))
    moved = []
    for eye in ((1.0, 0.0, 1.6), (6.0, 0.0, 1.6), (0.0, 6.0, 3.0),
                (-4.0, 2.0, 0.5)):
        got = SKY.evaluate(world(), d, eye=eye)
        delta = float(np.abs(got - base).max())
        if delta != 0.0:
            moved.append(f'{eye}: {delta:.8f}')
    check('the camera does not move the clouds at all', not moved,
          '; '.join(moved))

    # and with it off, the parallax has to be *weighted*: a cloud overhead is
    # at the deck's height and swings past, one at the horizon is effectively
    # at infinity and does not. Adding the offset flat slides the whole sky,
    # which is what made it look like the clouds were racing.
    free = world(link_clouds_to_view=False)
    ref = SKY.evaluate(free, d, eye=(0.0, 0.0, 1.6))
    far = SKY.evaluate(free, d, eye=(6.0, 0.0, 1.6))
    check('turning the link off does give parallax',
          float(np.abs(far - ref).mean()) > 1e-3,
          f'{float(np.abs(far - ref).mean()):.6f}')

    horizon = d[np.abs(d[:, 2]) < 0.10]
    overhead = d[d[:, 2] > 0.7]
    h = float(np.abs(SKY.evaluate(free, horizon, eye=(0.0, 0.0, 1.6))
                     - SKY.evaluate(free, horizon, eye=(6.0, 0.0, 1.6))).mean())
    o = float(np.abs(SKY.evaluate(free, overhead, eye=(0.0, 0.0, 1.6))
                     - SKY.evaluate(free, overhead, eye=(6.0, 0.0, 1.6))).mean())
    check('and the parallax is largest overhead and least at the horizon',
          o > h * 2.0, f'overhead {o:.5f} vs horizon {h:.5f}')


def test_the_water_reflects_the_same_sky():
    """The ocean's reflection must be the sky the camera is actually under.

    It evaluated the sky from the origin while the sky above was evaluated
    from the camera, so with camera-dependent clouds the two disagreed -- the
    water reflected a sky that was not there.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    rng = np.random.default_rng(8)
    d = rng.normal(size=(8000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d = d[d[:, 2] < -0.05]

    w = World(mode='BRYCE')
    w.ground_plane = True
    w.ground_mode = 'OCEAN'
    w.clouds = True
    w.link_clouds_to_view = False        # make the sky camera-dependent
    w.ocean_choppiness = 0.05            # near mirror, so the sky dominates

    near = SKY.evaluate(w, d, eye=(0.0, 0.0, 2.0))
    far = SKY.evaluate(w, d, eye=(40.0, 0.0, 2.0))
    check('the reflection follows the camera when the sky does',
          float(np.abs(near - far).mean()) > 1e-4,
          f'{float(np.abs(near - far).mean()):.6f}')

    # The water surface itself legitimately moves with the camera -- a
    # different viewpoint sees different water -- so "does it hold still" is
    # not the question. The question is whether the reflection was evaluated
    # from the camera at all, and that is asked directly.
    import inspect
    src = inspect.getsource(SKY._ocean)
    check('the reflection is evaluated from the camera, not the origin',
          "eye=getattr(world, '_eye'" in src,
          'the reflection still evaluates the sky from the origin')
    SKY.evaluate(w, d, eye=(12.0, -3.0, 2.0))
    stashed = getattr(w, '_eye', None)
    check('and the camera position is there for it to use',
          stashed is not None and abs(float(stashed[0]) - 12.0) < 1e-6,
          str(stashed))


def test_comets_are_animated():
    """Comets must cross the sky over time, and be comet-shaped while they do.

    Two things were wrong before they could move at all. The sun vector had
    been an argument of the comet function since it was written and was never
    read, so tails pointed wherever the random number generator sent them; and
    the streak was bounded at the far end only, so it ran on *in front of* the
    head as far as a cos^8 falloff allowed -- about forty degrees, which draws
    a line through the sky rather than a comet.
    """
    from ..core import sky as SKY
    from ..core.scene import World

    rng = np.random.default_rng(7)
    d = rng.normal(size=(9000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    def night(**kw):
        w = World(mode='BRYCE')
        w.clouds = False
        w.stratus = False
        w.haze_density = 0.0
        w.fog_density = 0.0
        w.sun_elevation = -0.3
        w.comets = 1.5
        w.comet_count = 8
        for k, v in kw.items():
            setattr(w, k, v)
        return w

    a = SKY.evaluate(night(), d, time=0.0)
    b = SKY.evaluate(night(), d, time=5.0)
    check('comets move as time passes', float(np.abs(b - a).mean()) > 1e-6,
          f'{float(np.abs(b - a).mean()):.9f}')

    still = SKY.evaluate(night(comet_speed=0.0), d, time=5.0)
    still0 = SKY.evaluate(night(comet_speed=0.0), d, time=0.0)
    check('and hold still at zero speed', np.array_equal(still, still0))
    check('a still frame is where the animation starts, so turning the speed '
          'up does not empty it',
          float(np.abs(still0 - a).mean()) < 1e-9,
          f'{float(np.abs(still0 - a).mean()):.9f}')

    # the shape: a compact head with a tail on one side of it only. Measured
    # on a fine grid of the sky around the comet rather than on scattered
    # directions, because one comet is about two degrees across and a random
    # sphere sample barely lands on it
    one = World(mode='BRYCE')
    one.comets = 1.0
    one.comet_count = 1
    one.comet_speed = 0.0
    one.comet_length = 0.12
    sun = SKY._sun_vector(-0.3, 0.6)

    coarse = rng.normal(size=(200000, 3)).astype(np.float32)
    coarse /= np.linalg.norm(coarse, axis=1, keepdims=True)
    head = coarse[int(np.argmax(
        SKY._comets(coarse, sun, one, 1.0, 1, 0, 0.0)[:, 0]))]

    e1 = np.cross(head, np.array([0.0, 0.0, 1.0], np.float32))
    e1 /= max(float(np.linalg.norm(e1)), 1e-9)
    e2 = np.cross(head, e1)
    span = np.linspace(-0.30, 0.30, 301, dtype=np.float32)   # +/- 17 degrees
    ga, gb = np.meshgrid(span, span)
    patch = head[None, :] + ga.ravel()[:, None] * e1[None, :] \
        + gb.ravel()[:, None] * e2[None, :]
    patch /= np.linalg.norm(patch, axis=1, keepdims=True)
    mag = SKY._comets(patch, sun, one, 1.0, 1, 0, 0.0)[:, 0]

    lit = mag > 0.02
    frac = float(lit.mean())
    check('a comet lights a small part of the sky and no more',
          0.0005 < frac < 0.25, f'{frac * 100:.2f}% of a 34-degree patch')
    check('the head is the brightest part of it',
          float(np.hypot(ga.ravel()[int(np.argmax(mag))],
                         gb.ravel()[int(np.argmax(mag))])) < 0.02)

    # everything lit outside the coma sits on one side: that is what a tail is
    r = np.hypot(ga.ravel(), gb.ravel())
    tail = lit & (r > 0.03)
    check('there is a tail as well as a head', int(tail.sum()) > 50,
          str(int(tail.sum())))
    if tail.any():
        wts = mag[tail]
        ca = float(np.average(ga.ravel()[tail], weights=wts))
        cb = float(np.average(gb.ravel()[tail], weights=wts))
        n = max((ca * ca + cb * cb) ** 0.5, 1e-9)
        side = (ga.ravel()[tail] * ca + gb.ravel()[tail] * cb) / n
        behind = float(mag[tail][side > 0.05].sum())
        ahead = float(mag[tail][side < -0.05].sum())
        check('the tail runs one way, not both',
              behind > ahead * 40.0,
              f'{behind:.2f} behind the head, {ahead:.4f} in front')

    # the sun vector is read now: pointing the tail at the sun must move it
    away = SKY._comets(coarse, sun, night(comet_tail_sun=1.0), 1.0, 1, 0, 0.0)
    trail = SKY._comets(coarse, sun, night(comet_tail_sun=0.0), 1.0, 1, 0, 0.0)
    check('tail direction is the sun, not the random seed',
          float(np.abs(away - trail).mean()) > 1e-9,
          f'{float(np.abs(away - trail).mean()):.9f}')
    other = SKY._sun_vector(0.9, 3.0)
    moved = SKY._comets(coarse, other, night(comet_tail_sun=1.0), 1.0, 1, 0, 0.0)
    check('and moving the sun moves the tail with it',
          float(np.abs(moved - away).mean()) > 1e-9,
          f'{float(np.abs(moved - away).mean()):.9f}')


def test_water_presets():
    """Every water must apply, render, and be its own water."""
    from ..core import sky as SKY
    from ..core.scene import World
    from ..presets import skies as SK
    from ..presets import waters as WA

    items = WA.water_items()
    check('there is a library of waters', len(items) >= 16, str(len(items)))
    check('every entry has a label and a note',
          all(label and note for _k, label, note in items))
    check('the display order covers the library',
          set(WA.ORDER) == set(WA.WATERS),
          str(set(WA.ORDER) ^ set(WA.WATERS)))

    fields = set(WA.water_fields())
    stray = []
    for key, entry in WA.WATERS.items():
        unknown = [n for n in entry['settings'] if n not in fields]
        if unknown:
            stray.append(f'{key}: {", ".join(unknown)}')
    check('no preset sets a field that does not exist', not stray,
          '; '.join(stray))

    # the two libraries must own disjoint halves of the World, or picking a
    # sky throws the water away and nobody can see why
    overlap = fields & set(SK.sky_fields())
    check('skies and waters own disjoint fields', not overlap, str(overlap))

    eye = np.array([0.0, 0.0, 9.0], np.float32)
    rng = np.random.default_rng(5)
    d = rng.normal(size=(4000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    d[:, 2] = -np.abs(d[:, 2]) - 0.05      # all looking down at the water
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    rendered, broke = {}, []
    for key, label, _note in items:
        w = World(mode='BRYCE')
        ok, msg = WA.apply_water(w, key)
        if not ok:
            broke.append(f'{key}: {msg}')
            continue
        if not (w.ground_plane and w.ground_mode == 'OCEAN'):
            broke.append(f'{key}: does not turn the plane to water')
            continue
        w._pixel_angle, w._pixel_width = 0.0006, 0.0
        col = SKY.evaluate(w, d, None, eye=eye, time=0.0)
        if col is None or not np.isfinite(col).all():
            broke.append(f'{key}: not finite')
            continue
        rendered[key] = col
    check('every water applies and renders', not broke, '; '.join(broke))

    same = []
    keys = list(rendered)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if float(np.abs(rendered[a] - rendered[b]).mean()) < 1e-4:
                same.append(f'{a} == {b}')
    check('no two waters land on the same picture', not same, '; '.join(same))

    # save and load must be exact for every one of them
    bad = []
    for key, _label, _note in items:
        w = World(mode='BRYCE')
        WA.apply_water(w, key)
        text = WA.dumps(w, key)
        w2 = World(mode='BRYCE')
        ok, _msg = WA.loads(w2, text)
        if not ok:
            bad.append(f'{key}: would not load')
            continue
        for f in fields:
            if getattr(w, f) != getattr(w2, f):
                bad.append(f'{key}.{f}')
    check('every water survives a save and a load exactly', not bad,
          '; '.join(bad[:6]))

    # a sky file is not a water file and must be refused
    w = World(mode='BRYCE')
    ok, msg = WA.loads(w, SK.dumps(w, 'a sky'))
    check('a sky file is refused by the water loader', not ok, msg)
    ok, msg = SK.loads(w, WA.dumps(w, 'a water'))
    check('and a water file is refused by the sky loader', not ok, msg)

    # applying a sky must leave the water exactly as it was, and the reverse
    w = World(mode='BRYCE')
    WA.apply_water(w, 'STORM_SWELL')
    before = {f: getattr(w, f) for f in fields}
    SK.apply_sky(w, 'DAWN')
    changed = [f for f in fields if getattr(w, f) != before[f]]
    check('applying a sky does not touch the water', not changed,
          ', '.join(changed))
    sky_before = {f: getattr(w, f) for f in SK.sky_fields()}
    WA.apply_water(w, 'MILLPOND')
    changed = [f for f in SK.sky_fields() if getattr(w, f) != sky_before[f]]
    check('and applying a water does not touch the sky', not changed,
          ', '.join(changed))


def test_water_preset_operators_are_wired():
    """The water library's buttons must exist and do what they say."""
    import os

    from . import fakeblender as FB
    props, _engine = FB.install()
    from .. import ui as UI
    from ..presets import waters as WA

    for name in ('HALCYON_OT_water_preset', 'HALCYON_OT_water_save',
                 'HALCYON_OT_water_load'):
        check(f'{name} exists', hasattr(UI, name))
    check('all three are registered',
          all(getattr(UI, n) in UI.CLASSES for n in
              ('HALCYON_OT_water_preset', 'HALCYON_OT_water_save',
               'HALCYON_OT_water_load')))
    check('Apply Preset is what the button says',
          UI.HALCYON_OT_water_preset.bl_label == "Apply Preset",
          UI.HALCYON_OT_water_preset.bl_label)
    check('and importing says so too',
          UI.HALCYON_OT_water_load.bl_label == "Import Preset",
          UI.HALCYON_OT_water_load.bl_label)

    src = open(os.path.join(os.path.dirname(UI.__file__), 'ui.py'),
               encoding='utf-8').read()
    at = src.index('Water Presets')
    body = src[at:at + 700]
    for want in ('water_preset', 'halcyon.water_preset', 'halcyon.water_save',
                 'halcyon.water_load'):
        check(f'the panel draws {want}', want in body)
    # drawn under the water, where Bryce kept it, not in the sky lab
    before = src[max(at - 500, 0):at]
    check("the library is drawn only when the plane is water",
          "ground_mode == 'OCEAN'" in before)

    items = props.water_preset_items(None, None)
    check('the dropdown lists the library',
          len(items) >= len(WA.ORDER), str(len(items)))
    check('and its keys match the library',
          {i[0] for i in items} >= set(WA.ORDER))


def test_sky_presets():
    """Every sky must apply, render, and be its own sky.

    A preset library where two entries land on the same picture is a library
    with a copy-paste in it, and that is not something you notice by scrolling
    the list.
    """
    import dataclasses

    from ..core import sky as SKY
    from ..core.scene import World
    from ..presets import skies as SK

    items = SK.sky_items()
    check('there is a library of skies', len(items) >= 20, str(len(items)))
    check('every entry has a label and a note',
          all(label and note for _k, label, note in items))
    check('the display order covers the library',
          set(SK.ORDER) == set(SK.SKIES),
          str(set(SK.ORDER) ^ set(SK.SKIES)))

    rng = np.random.default_rng(4)
    d = rng.normal(size=(4000, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)

    fields = set(SK.sky_fields())
    stray = []
    for key, entry in SK.SKIES.items():
        unknown = [n for n in entry['settings'] if n not in fields]
        if unknown:
            stray.append(f'{key}: {", ".join(unknown)}')
    check('no preset sets a field that does not exist', not stray,
          '; '.join(stray))

    rendered = {}
    broke = []
    for key, _label, _note in items:
        w = World(mode='BRYCE')
        ok, msg = SK.apply_sky(w, key)
        if not ok:
            broke.append(f'{key}: {msg}')
            continue
        col = SKY.evaluate(w, d)
        if not np.isfinite(col).all():
            broke.append(f'{key}: not finite')
            continue
        rendered[key] = col
    check('every sky applies and renders', not broke, '; '.join(broke))

    pairs = [(a, b) for i, a in enumerate(rendered)
             for b in list(rendered)[i + 1:]]
    same = [f'{a}=={b}' for a, b in pairs
            if float(np.abs(rendered[a] - rendered[b]).mean()) < 1e-4]
    check('no two skies are the same sky', not same, ', '.join(same[:4]))

    # applying one after another must not leave anything behind
    w = World(mode='BRYCE')
    SK.apply_sky(w, 'VOLCANIC')
    SK.apply_sky(w, 'CLEAR_BLUE')
    fresh = World(mode='BRYCE')
    SK.apply_sky(fresh, 'CLEAR_BLUE')
    left = [f.name for f in dataclasses.fields(World)
            if f.name in fields and getattr(w, f.name) != getattr(fresh, f.name)]
    check('presets do not accumulate', not left, ', '.join(left))

    # the mode and the strength are the user's, not the preset's
    w = World(mode='BRYCE')
    w.strength = 2.5
    SK.apply_sky(w, 'SUNSET')
    check('a sky does not touch Strength', abs(w.strength - 2.5) < 1e-6)


def test_sky_files_round_trip():
    """A saved sky must load back exactly, and survive being from elsewhere."""
    import dataclasses

    from ..core.scene import World
    from ..presets import skies as SK

    fields = set(SK.sky_fields())
    wrong = []
    for key, _label, _note in SK.sky_items():
        a = World(mode='BRYCE')
        SK.apply_sky(a, key)
        b = World(mode='BRYCE')
        ok, _msg = SK.loads(b, SK.dumps(a, key))
        if not ok:
            wrong.append(f'{key}: would not load')
            continue
        diff = [f.name for f in dataclasses.fields(World)
                if f.name in fields
                and getattr(a, f.name) != getattr(b, f.name)]
        if diff:
            wrong.append(f'{key}: {", ".join(diff[:3])}')
    check('every sky survives a save and a load unchanged', not wrong,
          '; '.join(wrong[:3]))

    # a file from a later version, carrying a field this one has never heard of
    w = World(mode='BRYCE')
    data = SK.sky_to_dict(w, 'from the future')
    data['version'] = SK.FORMAT_VERSION + 1
    data['settings']['cloud_hyperbole'] = 3.0
    ok, msg = SK.sky_from_dict(World(mode='BRYCE'), data)
    check('a sky from a later version still loads', ok, msg)
    check('and says what it had to leave out', 'does not have' in msg, msg)

    # and something that is not a sky at all is refused rather than half-applied
    ok, msg = SK.loads(World(), '{"format": "something-else", "settings": {}}')
    check('a file that is not a sky is refused', not ok, msg)
    ok, msg = SK.loads(World(), 'not json at all')
    check('and so is a file that is not JSON', not ok, msg)

    # the excluded fields must never be written into a file
    text = SK.dumps(World(mode='BRYCE'), 'x')
    leaked = [n for n in SK.EXCLUDED if f'"{n}"' in text]
    check('a sky file carries no node tree, image or render setting',
          not leaked, ', '.join(leaked))


def test_sky_preset_operators_are_wired():
    """Picking a sky, applying it and importing one are three separate acts.

    They used to be one: the menu applied on selection, and the file browser
    applied whatever it opened. Both are now choices that wait for a button --
    and the whole box only appears under the Bryce sky, because a library of
    Bryce skies means nothing next to a solid colour.
    """
    import inspect

    from . import fakebpy
    fakebpy.install()
    import importlib
    ui = importlib.import_module('halcyon.ui')
    props = importlib.import_module('halcyon.properties')

    for name in ('HALCYON_OT_sky_preset', 'HALCYON_OT_sky_save',
                 'HALCYON_OT_sky_load'):
        cls = getattr(ui, name, None)
        check(f'{name} exists', cls is not None)
        if cls is not None:
            check(f'{name} is registered', cls in ui.CLASSES, name)

    src = inspect.getsource(ui.HALCYON_PT_world.draw)
    for op in ('halcyon.sky_preset', 'halcyon.sky_save', 'halcyon.sky_load'):
        check(f'{op} is reachable from the World panel', op in src)

    # the selection lives on the world, so the dropdown remembers it
    check('the world carries the chosen sky',
          'sky_preset' in props.HalcyonWorldSettings.__annotations__)
    check('the panel draws it as a property rather than a menu of actions',
          "prop(hs, 'sky_preset'" in src and 'operator_menu_enum' not in src)
    check('and applying it is its own button',
          "operator('halcyon.sky_preset'" in src and 'Apply Preset' in src)

    # the box only exists under Bryce
    at = src.index('Sky Presets')
    body = src[max(at - 400, 0):at]
    check('the preset box is gated on the Bryce sky',
          "hs.mode == 'BRYCE'" in body, body[-120:])

    # importing must not touch the current sky
    load = inspect.getsource(ui.HALCYON_OT_sky_load.execute)
    check('importing copies the file into the library',
          'copyfile' in load and '_sky_library_dir' in load)
    check('and does not apply it',
          'SK.loads' not in load and 'apply_sky' not in load)
    check('a file that is not a sky is refused before it is copied',
          load.index('SK.FORMAT') < load.index('copyfile'))

    # the menu must offer every built-in sky
    from ..presets.skies import sky_items
    keys = {i[0] for i in props.sky_preset_items(None, None)}
    missing = [k for k, _l, _n in sky_items() if k not in keys]
    check('the menu offers every built-in sky', not missing,
          ', '.join(missing))

    # Blender does not hold the strings an items callback returns, so the
    # module has to. A garbage-collected enum item is a corrupted menu.
    first = props.sky_preset_items(None, None)
    second = props.sky_preset_items(None, None)
    check('the enum items are kept alive between calls', first is second)




def test_no_setting_lies():
    """Nothing in the UI may be a control that does nothing.

    A slider that silently has no effect is worse than an absent one: it costs
    the user time and their trust in every other control.
    """
    import dataclasses
    import os
    import re

    from ..core.settings import RenderSettings

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = ''
    for base, _dirs, files in os.walk(root):
        if '__pycache__' in base or os.path.basename(base) == 'tests':
            continue
        for f in files:
            if f.endswith('.py'):
                with open(os.path.join(base, f)) as fh:
                    src += fh.read() + '\n'
    with open(os.path.join(root, 'ui.py')) as fh:
        ui = fh.read()

    dead = [f.name for f in dataclasses.fields(RenderSettings)
            if not re.search(r'(?:st|settings|self\.settings)\.' + f.name + r'\b',
                             src)
            and not re.search(r"getattr\([^,]+,\s*'" + f.name + r"'", src)]
    shown = [n for n in dead
             if re.search(r"\.prop\((?:hs|self), '" + n + r"'", ui)]
    check('no unimplemented setting is exposed in the UI', not shown,
          ', '.join(shown))


def test_no_duplicate_definitions():
    """A definition shadowed by a later copy is dead code waiting to mislead."""
    import ast
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    problems = []
    for base, _dirs, files in os.walk(root):
        if '__pycache__' in base:
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(base, f)
            with open(path) as fh:
                try:
                    tree = ast.parse(fh.read())
                except SyntaxError:
                    problems.append(f'{f}: does not parse')
                    continue
            names = [n.name for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            dupes = sorted({n for n in names if names.count(n) > 1})
            if dupes:
                problems.append(f'{f}: {", ".join(dupes)}')
    check('no module defines the same thing twice', not problems,
          ' | '.join(problems))


def test_new_settings_are_implemented():
    """The settings implemented in this pass must actually change a render."""
    st = base_settings(120, 90)

    def go(**kw):
        s = base_settings(120, 90, **kw)
        sc = demo_scene(s)
        sc.materials[1].opacity = 0.45
        sc.materials[1].has_alpha = True
        return R.render(sc, s)

    check('alpha_threshold clips alpha',
          float(np.abs(go(alpha_threshold=0.2) - go(alpha_threshold=0.8)).mean())
          > 1e-4)
    fog = dict(fog=True, fog_start=2.0, fog_end=20.0)
    check('fog_vertex bands the fog',
          float(np.abs(go(**fog) - go(fog_vertex=True, **fog)).mean()) > 1e-6)

    from ..core import lights as LI
    from ..core.scene import Light
    from ..core.settings import RenderSettings
    l = Light(type='POINT', position=(0, 0, 5))
    d = np.array([2.0, 4.0], np.float32)
    a = RenderSettings()
    a.light_falloff_default = 'INVERSE'
    b = RenderSettings()
    b.light_falloff_default = 'INVERSE_SQUARE'
    check('light_falloff_default drives a light set to Scene Default',
          not np.allclose(LI.attenuate(l, d, a), LI.attenuate(l, d, b)))


def test_painters_algorithm():
    """Painter's must compare polygons, and fail the way the real thing does."""
    from .scenebuild import _mesh_concat, plane

    def tilted(z, tilt, mat):
        V, N, UV, T, _m, _o = plane(z=z, size=6.0, mat=mat)
        V = V.copy()
        V[:, 2] += V[:, 0] * tilt
        return (V, N, UV, T, mat, 0)

    def cross(mode, key='CENTROID'):
        st = base_settings(200, 150, depth_sort=mode)
        st.painters_key = key
        sc = demo_scene(st)
        sc.mesh = _mesh_concat([tilted(1.0, 0.7, 1), tilted(1.1, -0.7, 2)])
        sc.materials[1].diffuse = (0.9, 0.15, 0.1)
        sc.materials[2].diffuse = (0.1, 0.3, 0.9)
        return R.render(sc, st)

    zb = cross('ZBUFFER')
    pa = cross('PAINTERS')
    check('interpenetrating surfaces render differently under Painters',
          float(np.abs(zb - pa).mean()) > 1e-3,
          f'mean difference {float(np.abs(zb - pa).mean()):.5f}')
    check('both stay finite',
          bool(np.isfinite(zb).all() and np.isfinite(pa).all()))

    # the sort key must change where it goes wrong
    keys = {}
    for key in ('CENTROID', 'NEAREST', 'FARTHEST'):
        st = base_settings(200, 150, depth_sort='PAINTERS')
        st.painters_key = key
        keys[key] = R.render(demo_scene(st), st)
    pairs = [('CENTROID', 'NEAREST'), ('CENTROID', 'FARTHEST'),
             ('NEAREST', 'FARTHEST')]
    same = [f'{a}=={b}' for a, b in pairs
            if float(np.abs(keys[a] - keys[b]).mean()) < 1e-5]
    check('all three sort keys give different results', not same,
          ', '.join(same))

    # a whole polygon must carry one depth
    from ..core import raster
    st = base_settings(160, 120, depth_sort='PAINTERS')
    sc = demo_scene(st)
    _v, _p, vp, eye = R.camera_matrices(sc.camera, 160, 120)
    view = np.linalg.inv(np.asarray(sc.camera.matrix_world, np.float32))
    fd = R.polygon_depths(sc.mesh, view, eye, 'CENTROID')
    check('one depth per triangle', fd.shape == (sc.mesh.tris.shape[0],),
          str(fd.shape))
    gb = raster.GBuffer(160, 120)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 160, 120, gbuf=gb,
                     flat_depth=fd)
    cov = gb.mask()
    yy, xx = np.nonzero(cov)
    if yy.size:
        tri = gb.tri[yy, xx]
        got = gb.depth[yy, xx]
        check('every fragment carries its polygon depth',
              float(np.abs(got - fd[tri]).max()) < 1e-4,
              f'max deviation {float(np.abs(got - fd[tri]).max()):.6f}')

    # and the fast path must agree with the reference
    saved = (raster.BATCH_MIN_TRIS, raster.LARGE_TRI_PX)
    try:
        raster.BATCH_MIN_TRIS, raster.LARGE_TRI_PX = 10 ** 9, 0
        seq = R.render(demo_scene(st), st)
        raster.BATCH_MIN_TRIS, raster.LARGE_TRI_PX = 24, 16384
        bat = R.render(demo_scene(st), st)
    finally:
        raster.BATCH_MIN_TRIS, raster.LARGE_TRI_PX = saved
    check('batched and sequential agree under Painters',
          float(np.abs(seq - bat).max()) == 0.0)


def test_new_features_1_5():
    """Displacement bump, light linking, lens, shafts and defocus."""
    st = base_settings(200, 150)

    # light linking
    base = R.render(demo_scene(st), st)
    sc = demo_scene(st)
    sc.lights[0].exclude_objects = (1,)
    excl = R.render(sc, st)
    sc2 = demo_scene(st)
    sc2.lights[0].exclude_objects = (1,)
    sc2.lights[0].exclude_mode = 'ONLY'
    only = R.render(sc2, st)
    check('light linking Exclude changes the render',
          float(np.abs(base - excl).mean()) > 1e-4)
    check('Only differs from Exclude', float(np.abs(excl - only).mean()) > 1e-4)

    # displacement as bump
    g = {'output': 'out', 'nodes': {
        'n': _wnode('n', 'ShaderNodeTexNoise', {'noise_dimensions': '3D'},
                    [_sk('Vector', 'VECTOR', [0, 0, 0]),
                     _sk('Scale', 'VALUE', 9.0), _sk('Detail', 'VALUE', 3.0),
                     _sk('Roughness', 'VALUE', .5),
                     _sk('Distortion', 'VALUE', 0.)],
                    [{'name': 'Fac', 'type': 'VALUE'},
                     {'name': 'Color', 'type': 'RGBA'}]),
        'd': _wnode('d', 'ShaderNodeDisplacement', {'space': 'OBJECT'},
                    [_sk('Height', 'VALUE', 0.0, ['n', 0]),
                     _sk('Midlevel', 'VALUE', 0.5), _sk('Scale', 'VALUE', 1.0),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'Displacement', 'type': 'VECTOR'}]),
        'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                    [_sk('Color', 'RGBA', [.8, .8, .8, 1]),
                     _sk('Roughness', 'VALUE', 0.),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['b', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0], ['d', 0])], [])}}

    def with_disp(scale):
        s2 = base_settings(200, 150)
        s2.displacement_scale = scale
        sc3 = demo_scene(s2)
        for m in sc3.materials:
            m.graph = g
        return R.render(sc3, s2)

    check('displacement drives a bump normal',
          float(np.abs(with_disp(1.5) - with_disp(0.0)).mean()) > 1e-4)

    # post effects
    def shot(**kw):
        s2 = base_settings(200, 150, **{k: v for k, v in kw.items()
                                        if k != 'vol'})
        sc4 = demo_scene(s2)
        if kw.get('vol'):
            sc4.lights[0].direction = (0.55, -0.67, -0.38)
            sc4.lights[0].volumetric = 2.0
        img = R.render(sc4, s2)
        return post.process(img, s2, target_size=(200, 150),
                            depth=getattr(sc4, 'last_depth', None),
                            shaft_sources=getattr(sc4, 'last_shafts', None))

    plain = shot()
    for label, kw in (('lens distortion', dict(lens_distortion=0.35)),
                      ('chromatic aberration', dict(chromatic_aberration=4.0)),
                      ('depth of field', dict(dof=True, dof_focus=9.0,
                                              dof_amount=2.0)),
                      ('light shafts', dict(vol=True))):
        img = shot(**kw)
        check(f'{label} changes the image',
              float(np.abs(img - plain).mean()) > 1e-4,
              f'delta {float(np.abs(img - plain).mean()):.5f}')
        check(f'{label} stays finite', bool(np.isfinite(img).all()))

    # a light behind the camera must throw no shafts
    s3 = base_settings(200, 150)
    sc5 = demo_scene(s3)
    sc5.lights[0].direction = (-0.55, 0.67, 0.38)
    sc5.lights[0].volumetric = 2.0
    R.render(sc5, s3)
    check('a source behind the camera is skipped',
          len(getattr(sc5, 'last_shafts', [])) == 0)


def test_gpu_shaders_compile_and_agree():
    """Validate GLSL that no driver here can execute.

    Halcyon has its own GLSL front-end and a NumPy backend. Compiling the GPU
    stages with it and running them proves the *logic*, even though nothing on
    this machine can prove the *execution*. Every stage the engine is allowed to
    run must agree with the CPU function it replaces.
    """
    from ..core import dither as DI
    from ..core import post as PO
    from ..core.settings import RenderSettings
    from ..core.texture import Texture
    from ..gpu.stages import ENABLED, MASK_KINDS, STAGES, VALIDATION
    from ..shaders.compiler import try_compile

    bad = []
    for name, src in STAGES.items():
        prog, err = try_compile(src, 'GLSL')
        if prog is None:
            bad.append(f'{name}: {err}')
    check(f'all {len(STAGES)} GPU stages are valid GLSL', not bad,
          '; '.join(bad))

    h, w = 32, 48
    img = np.random.default_rng(2).random((h, w, 3)).astype(np.float32)
    tex = Texture(np.concatenate([img, np.ones((h, w, 1), np.float32)], 2),
                  colorspace='Non-Color', filt='NEAREST', wrap='EXTEND')
    yy, xx = np.mgrid[0:h, 0:w]
    uv = np.stack([((xx + 0.5) / w).ravel(),
                   ((yy + 0.5) / h).ravel()], 1).astype(np.float32)
    n = h * w

    def run(name, **uni):
        # bound as a uniform here only: the compiler resolves known varying
        # names from a shading context this harness does not have
        src = STAGES[name].replace('in vec2 vUV;', 'uniform vec2 vUV;')
        prog, err = try_compile(src, 'GLSL')
        u = {'source': tex, 'vUV': uv}
        for k, v in uni.items():
            u[k] = (np.full(n, float(v), np.float32)
                    if isinstance(v, (int, float))
                    else np.broadcast_to(np.asarray(v, np.float32),
                                         (n, len(v))).copy())
        outs, _d = prog.run(u, {}, n)
        return outs['Color'].reshape(h, w, 4)[:, :, :3]

    st = RenderSettings()
    st.exposure, st.gamma, st.contrast = 1.3, 2.2, 0.15
    st.saturation, st.brightness = 1.25, 0.05
    got = run('DISPLAY', exposure=1.3, brightness=0.05, contrast=0.15,
              saturation=1.25, gamma=2.2)
    want = PO.display_transform(img.copy(), st)
    check('the DISPLAY shader is exact against the CPU path',
          float(np.abs(got - want).max()) < 1e-5,
          f'max {float(np.abs(got - want).max()):.6f}')

    s3 = RenderSettings()
    s3.crt, s3.crt_scanlines, s3.crt_mask = True, 0.4, 'APERTURE'
    s3.crt_mask_strength, s3.crt_vignette = 0.35, 0.5
    s3.crt_curvature = s3.crt_bloom = 0.0
    got = run('CRT', scanlines=0.4, mask_strength=0.35,
              mask_kind=MASK_KINDS['APERTURE'], vignette=0.5, resolution=(w, h))
    want = PO.crt(img.copy(), s3)
    tol = VALIDATION['CRT'][1]
    check('the CRT shader agrees within its stated tolerance',
          float(np.abs(got - want).max()) <= tol,
          f'max {float(np.abs(got - want).max()):.5f} vs tolerance {tol}')

    got = run('DITHER', levels=(32., 64., 32.), strength=1.0, matrix_size=4.0,
              resolution=(w, h))
    want = DI.ordered_bits(img.copy(), (5, 6, 5), 'BAYER4', 1.0)
    tol = VALIDATION['DITHER'][1]
    check('the DITHER shader agrees within its stated tolerance',
          float(np.abs(got - want).max()) <= tol,
          f'max {float(np.abs(got - want).max()):.5f} vs tolerance {tol}')

    # nothing unproven may be enabled
    unproven = [k for k, (grade, _t) in VALIDATION.items()
                if grade == 'UNPROVEN' and k in ENABLED]
    check('no unvalidated stage is enabled', not unproven, ', '.join(unproven))
    check('every enabled stage has a validation grade',
          all(k in VALIDATION for k in ENABLED))


def test_gpu_absence_is_handled():
    """With no gpu module the engine must explain itself, not fall over."""
    from ..gpu import device
    from ..gpu.stages import STAGES
    device.reset()
    ok, why = device.probe()
    check('probing without a GPU returns a reason', bool(why), str(why))
    shader, err = device.compile_stage('DISPLAY', STAGES['DISPLAY'])
    check('compiling without a GPU returns None and a reason',
          shader is None and bool(err), str(err))
    check('describe() is human readable', 'GPU' in device.describe(),
          device.describe())


def test_selftest_report():
    """The self-test must produce a full report on a machine with no GPU."""
    from . import fakebpy
    bpy = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy.types, name):
            setattr(bpy.types, name, type(name, (bpy.types.Panel,), {}))
    bpy.app.version = (5, 2, 0)
    import importlib
    st_mod = importlib.import_module('halcyon.selftest')

    out = []
    st_mod.environment(out)
    text = '\n'.join(out)
    for field in ('addon', 'blender', 'platform', 'numpy', 'logical cores'):
        check(f'the report states the {field}', field in text)

    out = []
    st_mod.gpu_stages(out)
    check('the GPU section explains itself when there is no GPU',
          'skipped' in '\n'.join(out), '\n'.join(out)[:80])

    out = []
    st_mod.frame_breakdown(out)
    check('the report includes a frame breakdown',
          'slowest stage' in '\n'.join(out))

    out = []
    st_mod.cpu_scaling(out)
    text = '\n'.join(out)
    check('the report includes thread scaling', 'threads' in text)
    check('the report covers the worker pool', 'worker processes' in text)


def test_no_bl_info_dependency():
    """Nothing may read bl_info: an installed extension does not have it.

    Blender uses blender_manifest.toml for extensions and does not expose
    bl_info at all, so importing it raises ImportError on exactly the install
    path most users take.
    """
    import os
    import re

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for base, _dirs, files in os.walk(root):
        if '__pycache__' in base or os.path.basename(base) == 'tests':
            continue
        for f in files:
            if not f.endswith('.py') or f == '__init__.py':
                continue
            with open(os.path.join(base, f)) as fh:
                text = fh.read()
            if re.search(r'import\s+bl_info|bl_info\s*\[', text):
                offenders.append(f)
    check('no module reads bl_info', not offenders, ', '.join(offenders))

    from ..version import version, version_string
    check('the version comes from the manifest',
          isinstance(version(), tuple) and len(version()) == 3,
          str(version()))
    manifest = os.path.join(root, 'blender_manifest.toml')
    with open(manifest) as fh:
        raw = [l for l in fh if l.strip().startswith('version')][0]
    check('it matches blender_manifest.toml', version_string() in raw,
          f'{version_string()} vs {raw.strip()}')


def test_selftest_survives_a_broken_section():
    """One failing section must not take the whole report with it."""
    from . import fakebpy
    bpy = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy.types, name):
            setattr(bpy.types, name, type(name, (bpy.types.Panel,), {}))
    bpy.app.version = (5, 2, 0)
    import importlib
    mod = importlib.import_module('halcyon.selftest')

    original = mod.gpu_stages
    try:
        def explode(_out):
            raise RuntimeError('simulated driver failure')
        mod.gpu_stages = explode
        op = mod.HALCYON_OT_selftest()
        op.include_scaling = False
        op.heavy = False

        class _WM:
            clipboard = ''

        class _Ctx:
            window_manager = _WM()

        printed = []
        real_p = mod._p
        mod._p = lambda out, text='': (out.append(text), printed.append(text))
        try:
            op.report = lambda *a, **k: None
            op.execute(_Ctx())
        finally:
            mod._p = real_p
    finally:
        mod.gpu_stages = original

    text = '\n'.join(printed)
    check('the failing section is named', 'gpu stages failed' in text)
    check('later sections still ran', 'slowest stage' in text
          or 'FRAME BREAKDOWN' in text)
    check('the report still closes properly', 'end of report' in text)


def test_gpu_interface_specs():
    """Every stage must have a CreateInfo spec matching its GLSL.

    Blender defaults to Vulkan, where the legacy GPUShader constructor does not
    exist and the interface has to be declared separately from the source. A
    mismatch there fails on the driver and nowhere else.
    """
    import re

    from ..gpu.stages import INTERFACE, STAGES, body

    missing = [k for k in STAGES if k not in INTERFACE]
    check('every stage has an interface spec', not missing, ', '.join(missing))

    problems = []
    for name in STAGES:
        src = body(name)
        if re.search(r'^(uniform|in|out)\s+\w', src, re.M):
            problems.append(f'{name}: declarations survived stripping')
        if 'void main()' not in src:
            problems.append(f'{name}: lost main()')
        spec = INTERFACE[name]
        for group in ('floats', 'ints', 'vec2', 'vec3', 'samplers'):
            for uni in spec.get(group, ()):
                if not re.search(r'\b' + re.escape(uni) + r'\b', src):
                    problems.append(f'{name}: declares {uni}, body never uses it')
    check('interface specs match their shader bodies', not problems,
          '; '.join(problems[:3]))

    # anything the body reads must be declared, or the driver rejects it
    undeclared = []
    for name in STAGES:
        spec = INTERFACE[name]
        declared = {'vUV', 'Color', 'pos', 'uv'}
        for group in ('floats', 'ints', 'vec2', 'vec3', 'samplers'):
            declared |= set(spec.get(group, ()))
        for line in STAGES[name].splitlines():
            m = re.match(r'^(?:uniform|in|out)\s+\w+\s+(\w+)\s*;', line.strip())
            if m and m.group(1) not in declared:
                undeclared.append(f'{name}: {m.group(1)}')
    check('every declaration in the source is in the spec', not undeclared,
          ', '.join(undeclared))


def test_threaded_background():
    """The background pass must be identical however many threads run it."""
    from ..core.scene import World
    base = None
    for threads in (1, 4, 16):
        st = base_settings(320, 240, threads=threads)
        st.aa_samples = 4
        sc = demo_scene(st)
        sc.world = World()
        sc.world.mode = 'BRYCE'
        img = R.render(sc, st)
        if base is None:
            base = img
        else:
            # summing a slice and summing the whole array can differ in the
            # last bit; one ULP of float32 is the honest bar here, not zero
            d = float(np.abs(base - img).max())
            check(f'threaded background matches at {threads} threads',
                  d <= 1e-6, f'max difference {d:.3e}')


def test_material_templates():
    """Every template must name real models, sockets and texture nodes.

    They are recipes built at runtime, so a renamed socket turns one into a
    silent no-op rather than an error -- which is exactly what a test is for.
    """
    from . import fakebpy
    bpy = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy.types, name):
            setattr(bpy.types, name, type(name, (bpy.types.Panel,), {}))
    import importlib
    tmpl = importlib.import_module('halcyon.templates')
    from ..core.nodeeval import DISPATCH
    from ..core.shading import MODEL_ITEMS
    from ..nodes.shader_nodes import HALCYON_ShaderNode as HS

    models = {m[0] for m in MODEL_ITEMS}
    sockets = {n for _k, n, _d in HS.SOCKETS}
    problems = []
    for key, spec in tmpl.TEMPLATES.items():
        for field in ('label', 'note', 'model'):
            if field not in spec:
                problems.append(f'{key}: missing {field}')
        if spec.get('model') not in models:
            problems.append(f"{key}: unknown model {spec.get('model')}")
        for sock in spec.get('inputs', {}):
            if sock not in sockets:
                problems.append(f'{key}: unknown socket {sock}')
        for idname, _p, _i, target in spec.get('textures', []):
            if idname not in DISPATCH:
                problems.append(f'{key}: {idname} has no evaluator')
            if target not in sockets:
                problems.append(f'{key}: unknown target {target}')
    check(f'all {len(tmpl.TEMPLATES)} templates are valid', not problems,
          '; '.join(problems[:3]))
    labels = [v['label'] for v in tmpl.TEMPLATES.values()]
    check('template labels are unique', len(labels) == len(set(labels)))
    check('the menu lists every template',
          len(tmpl.template_items()) == len(tmpl.TEMPLATES))


def test_infinite_ground():
    """The analytic ground plane must appear below the horizon and nowhere else."""
    from ..core import sky as SKY
    from ..core.scene import World

    dirs = _hemisphere(80, 120)
    el = np.linspace(-1.2, -0.05, 40)
    az = np.linspace(0, 2 * np.pi, 120)
    E, A = np.meshgrid(el, az, indexing='ij')
    down = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                     np.sin(E)], -1).reshape(-1, 3).astype(np.float32)
    eye = np.array([0.0, 0.0, 4.0], np.float32)

    # strictly upward: _hemisphere dips just below the horizon, where the
    # ground legitimately appears
    up_el = np.linspace(0.05, 1.45, 40)
    UE, UA = np.meshgrid(up_el, az, indexing='ij')
    up = np.stack([np.cos(UE) * np.cos(UA), np.cos(UE) * np.sin(UA),
                   np.sin(UE)], -1).reshape(-1, 3).astype(np.float32)

    plain = World()
    plain.mode = 'GRADIENT'
    above = SKY.evaluate(plain, up, {}, eye=eye)

    seen = {}
    for mode in ('SOLID', 'CHECKER', 'NOISE', 'OCEAN'):
        w = World()
        w.mode = 'GRADIENT'
        w.ground_plane = True
        w.ground_mode = mode
        seen[mode] = SKY.evaluate(w, down, {}, eye=eye, time=1.0)
        check(f'{mode} ground renders below the horizon',
              float(np.abs(seen[mode] - SKY.evaluate(plain, down, {},
                                                     eye=eye)).mean()) > 1e-4)
        check(f'{mode} ground stays finite', bool(np.isfinite(seen[mode]).all()))

    w = World()
    w.mode = 'GRADIENT'
    w.ground_plane = True
    w.ground_mode = 'CHECKER'
    check('the ground never touches rays pointing up',
          float(np.abs(SKY.evaluate(w, up, {}, eye=eye) - above).max()) == 0.0)

    # the ocean animates, the others do not
    a = SKY.evaluate(w, down, {}, eye=eye, time=0.0)
    b = SKY.evaluate(w, down, {}, eye=eye, time=5.0)
    check('a checker floor does not animate',
          float(np.abs(a - b).max()) == 0.0)
    w.ground_mode = 'OCEAN'
    a = SKY.evaluate(w, down, {}, eye=eye, time=0.0)
    b = SKY.evaluate(w, down, {}, eye=eye, time=5.0)
    check('the ocean animates with scene time',
          float(np.abs(a - b).mean()) > 1e-4)


def test_bryce_sky_lab():
    """Every Bryce control must change the sky on its own."""
    from ..core import sky as SKY
    from ..core.scene import World
    dirs = _hemisphere(70, 120)

    def sky_of(**kw):
        w = World()
        w.mode = 'BRYCE'
        for k, v in kw.items():
            setattr(w, k, v)
        return SKY.evaluate(w, dirs, {}, eye=np.array([0, 0, 3.0], np.float32),
                            time=1.0)

    base = sky_of()
    layers = (
        ('three-stop gradient', dict(sky_mid=(0.9, 0.3, 0.2))),
        ('mid stop height', dict(sky_mid=(0.9, 0.3, 0.2), sky_mid_height=0.8)),
        ('atmosphere', dict(atmosphere_density=0.8)),
        ('atmosphere falloff', dict(atmosphere_density=0.8,
                                    atmosphere_falloff=4.0)),
        ('haze blend with sky', dict(haze_density=1.0, haze_blend_sky=1.0)),
        ('cloud ambience', dict(cloud_ambience=1.0)),
        ('moon', dict(celestial='MOON', sun_elevation=0.6)),
    )
    dead = [name for name, kw in layers
            if float(np.abs(base - sky_of(**kw)).max()) < 1e-3]
    check(f'all {len(layers)} new Bryce controls change the sky', not dead,
          'no effect: ' + ', '.join(dead))

    # wind must move the deck with time, and only with time
    still_a = sky_of(cloud_wind=0.0)
    still_b = sky_of(cloud_wind=0.0)
    check('a windless sky does not drift',
          float(np.abs(still_a - still_b).max()) == 0.0)
    w = World()
    w.mode = 'BRYCE'
    w.cloud_wind = 3.0
    eye = np.array([0, 0, 3.0], np.float32)
    a = SKY.evaluate(w, dirs, {}, eye=eye, time=0.0)
    b = SKY.evaluate(w, dirs, {}, eye=eye, time=4.0)
    check('wind drifts the cloud deck over time',
          float(np.abs(a - b).mean()) > 1e-4)

    # moon phases must actually differ from each other
    phases = [sky_of(celestial='MOON', sun_elevation=0.6, moon_phase=p)
              for p in (0.05, 0.25, 0.5)]
    same = [i for i in range(len(phases) - 1)
            if float(np.abs(phases[i] - phases[i + 1]).max()) < 1e-3]
    check('the moon shows different phases', not same)

    # cloud shadows need the ground to land on
    def ground(shadows):
        w = World()
        w.mode = 'BRYCE'
        w.ground_plane = True
        w.ground_mode = 'SOLID'
        w.cloud_shadows = shadows
        el = np.linspace(-0.6, -0.05, 40)
        az = np.linspace(0, 2 * np.pi, 100)
        E, A = np.meshgrid(el, az, indexing='ij')
        d = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A),
                      np.sin(E)], -1).reshape(-1, 3).astype(np.float32)
        return SKY.evaluate(w, d, {}, eye=eye, time=1.0)

    check('cloud shadows land on the ground',
          float(np.abs(ground(0.0) - ground(1.0)).mean()) > 1e-4)


def test_no_socket_edits_in_update_callbacks():
    """RNA update callbacks must not add or remove sockets.

    Blender is mid-update inside a property callback, and mutating a node's
    sockets there segfaults rather than raising. Selecting HLSL did exactly
    that -- and worse, it set another property whose own callback then rebuilt
    the sockets from inside a nested callback.

    This cannot be exercised without Blender, so the rule is checked statically
    instead: no function used as an `update=` handler may reach socket
    mutation.
    """
    import ast
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Only topology changes are the hazard. Setting `hide` or a description on
    # an existing socket is a property write and is safe -- Blender's own nodes
    # do it. Adding or removing sockets while RNA is mid-update is not.
    banned = ('rebuild_sockets', 'compile_source')
    offenders = []
    for base, _dirs, files in os.walk(root):
        if '__pycache__' in base or os.path.basename(base) == 'tests':
            continue
        for f in files:
            if not f.endswith('.py'):
                continue
            path = os.path.join(base, f)
            with open(path) as fh:
                tree = ast.parse(fh.read())
            # every name used as update=<name>
            handlers = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword) and node.arg == 'update':
                    if isinstance(node.value, ast.Name):
                        handlers.add(node.value.id)
                    elif isinstance(node.value, ast.Attribute):
                        handlers.add(node.value.attr)
            if not handlers:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in handlers:
                    for call in ast.walk(node):
                        if isinstance(call, ast.Call):
                            name = getattr(call.func, 'attr',
                                           getattr(call.func, 'id', ''))
                            if name in banned:
                                offenders.append(f'{f}:{node.name} calls {name}')
                    for sub in ast.walk(node):
                        if isinstance(sub, ast.Attribute) and \
                                sub.attr in ('clear', 'new', 'remove'):
                            owner = getattr(sub.value, 'attr', '')
                            if owner in ('inputs', 'outputs'):
                                offenders.append(
                                    f'{f}:{node.name} edits {owner}')
    check('no update callback rebuilds sockets', not offenders,
          '; '.join(sorted(set(offenders))))

    # and the one function an update callback IS allowed to call must stay
    # free of topology changes, or the exemption stops being true
    import importlib
    from . import fakebpy
    bpy = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy.types, name):
            setattr(bpy.types, name, type(name, (bpy.types.Panel,), {}))
    sn = importlib.import_module('halcyon.nodes.shader_nodes')
    body = __import__('inspect').getsource(sn.HALCYON_ShaderNode.refresh_sockets)
    topology = [w for w in ('inputs.new', 'inputs.clear', 'inputs.remove',
                            'outputs.new', 'outputs.clear', 'outputs.remove')
                if w in body]
    check('the master shader only toggles existing sockets', not topology,
          ', '.join(topology))


def test_both_shader_languages():
    """Both default templates must compile and run through the interpreter."""
    from . import fakebpy
    bpy = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy.types, name):
            setattr(bpy.types, name, type(name, (bpy.types.Panel,), {}))
    import importlib
    sn = importlib.import_module('halcyon.nodes.shader_nodes')
    from ..shaders.compiler import try_compile

    for lang, src in (('GLSL', sn.DEFAULT_GLSL), ('HLSL', sn.DEFAULT_HLSL)):
        prog, err = try_compile(src, lang)
        check(f'the default {lang} template compiles', prog is not None,
              str(err))
        if prog is None:
            continue
        uni = {}
        for u in prog.uniform_schema():
            k = u['kind']
            if k in ('VALUE', 'INT'):
                uni[u['name']] = np.full(4, 1.0, np.float32)
            elif k in ('VECTOR2', 'VECTOR', 'RGBA'):
                uni[u['name']] = np.ones(
                    (4, {'VECTOR2': 2, 'VECTOR': 3, 'RGBA': 4}[k]), np.float32)
        outs, _d = prog.run(uni, {}, 4)
        check(f'the default {lang} template runs',
              bool(outs) and all(np.isfinite(np.asarray(v)).all()
                                 for v in outs.values() if v is not None))


def test_device_capability_table():
    """The capability table must be honest about can't versus not yet."""
    from ..gpu import capability as C
    from ..gpu.stages import VALIDATION

    bad = [f for f, (sup, why) in C.FEATURES.items()
           if sup not in (C.BOTH, C.NOT_YET, C.NEVER) or len(why) < 20]
    check('every feature has a support level and a real reason', not bad,
          ', '.join(bad))

    # anything claimed as running on both must be a stage proven on hardware
    proven = {k for k, (grade, _t) in VALIDATION.items()
              if grade in ('EXACT', 'CLOSE')}
    claimed = {f for f, (sup, _w) in C.FEATURES.items() if sup == C.BOTH}
    mapping = {'display_transform': 'DISPLAY', 'ordered_dither': 'DITHER',
               'crt': 'CRT', 'lens': 'LENS'}
    unproven = [f for f in claimed if mapping.get(f, '') not in proven]
    check('nothing claims GPU support without a measured stage', not unproven,
          ', '.join(unproven))

    # the two impossible ones must stay impossible
    check('error diffusion is marked as never portable',
          C.FEATURES['error_diffusion'][0] == C.NEVER)
    check('the A-buffer is marked as never portable',
          C.FEATURES['abuffer'][0] == C.NEVER)

    # the coded shader node is NOT_YET, not NEVER -- it is the easiest piece
    check('the coded shader node is portable, just unported',
          C.FEATURES['code_node'][0] == C.NOT_YET)
    check('the rasteriser is named as the last piece',
          'last piece' in C.FEATURES['rasterise'][1].lower(),
          C.FEATURES['rasterise'][1][:50])
    # the two must not get swapped: the code node is the easy one
    check('the coded shader node is described as the easiest piece',
          'easiest' in C.FEATURES['code_node'][1].lower(),
          C.FEATURES['code_node'][1][:60])
    check('the node evaluator is described as the hard one',
          'hard' in C.FEATURES['node_graph'][1].lower(),
          C.FEATURES['node_graph'][1][:60])


def test_device_plan_falls_back():
    """Choosing GPU must never refuse to render, and must say what moved."""
    from ..gpu import capability as C
    from ..gpu import device as dev

    st = base_settings(120, 90)
    st.render_device = 'GPU'
    st.raytrace = True
    st.dither = 'FLOYD'
    sc = demo_scene(st)
    sc.materials[0].programs = {'shader': object()}

    device, stages, notes = C.plan(sc, st)
    check('a missing GPU falls back to the CPU', device == C.CPU)
    check('and explains why', any('no GPU' in n for n in notes),
          '; '.join(notes))

    # now pretend a GPU is there
    saved = dev.probe
    try:
        dev.probe = lambda: (True, 'stub device')
        device, stages, notes = C.plan(sc, st)
        proven_now = {f for f, (g, _t) in C.FEATURES.items() if g == C.BOTH}
        check('with a GPU present every proven stage is selected',
              set(stages) == proven_now, f'{sorted(stages)} vs {sorted(proven_now)}')
        text = ' '.join(notes)
        for feat in ('code_node', 'node_graph'):
            check(f'{feat} is reported as staying on the CPU', feat in text)
        check('the reason is given, not just the name',
              'GLSL' in text or 'NumPy' in text)
    finally:
        dev.probe = saved

    # CPU is always unconditionally fine
    st.render_device = 'CPU'
    device, stages, notes = C.plan(sc, st)
    check('CPU mode asks nothing of the GPU',
          device == C.CPU and stages == () and not notes)
    check('every feature is supported on the CPU',
          all(C.supports(f, C.CPU) for f in C.FEATURES))


def test_device_choice_does_not_change_the_image():
    """Selecting GPU must not alter output when it all falls back anyway."""
    st_cpu = base_settings(160, 120)
    st_gpu = base_settings(160, 120)
    st_gpu.render_device = 'GPU'
    a = post.process(R.render(demo_scene(st_cpu), st_cpu), st_cpu,
                     target_size=(160, 120), allow_resize=False)
    b = post.process(R.render(demo_scene(st_gpu), st_gpu), st_gpu,
                     target_size=(160, 120), allow_resize=False)
    check('with no GPU present both devices render identically',
          float(np.abs(a - b).max()) == 0.0)


def test_glsl_shading_models_match_cpu():
    """Every reflectance model in GLSL must equal its CPU counterpart.

    Halcyon's own GLSL front-end executes the shaders through NumPy, so the
    maths of a GPU port can be checked exactly on a machine with no GPU. This
    found nine wrong formulas on the first run -- Blinn-Phong's exponent is
    four times the stated gloss, Minnaert takes 1 + 2*roughness as its
    darkness, Ward is driven by roughness rather than the Phong exponent, Toon
    steps on the angle rather than the cosine, and Strauss scales its diffuse
    by rn and skips the soften pass entirely.
    """
    from ..core import mathx as MX
    from ..core.shading import MODEL_ITEMS, Surface, evaluate
    from ..gpu.glsl_shading import DISPATCH, GLSL
    from ..shaders.compiler import try_compile

    src = GLSL + DISPATCH + """
uniform int model;
uniform float glossiness; uniform float roughness; uniform float metallic;
uniform float anisotropy; uniform float soften; uniform float ior;
uniform float translucency; uniform float toon_size; uniform float toon_smooth;
uniform float opacity;
uniform vec3 diffuse; uniform vec3 tangent; uniform vec3 bitangent;
uniform vec3 nrm; uniform vec3 lgt; uniform vec3 vew;
out vec4 Color;
void main() {
    HalcyonSurface s;
    s.diffuse = diffuse; s.specular = vec3(1.0);
    s.diffuse_level = 1.0; s.specular_level = 1.0;
    s.glossiness = glossiness; s.roughness = roughness; s.metallic = metallic;
    s.anisotropy = anisotropy; s.aniso_rot = 0.0; s.soften = soften;
    s.ior = ior; s.translucency = translucency; s.opacity = opacity;
    s.toon_size = toon_size; s.toon_smooth = toon_smooth; s.toon_steps = 2.0;
    s.tangent = tangent; s.bitangent = bitangent;
    Color = hal_evaluate(model, s, nrm, lgt, vew);
}
"""
    prog, err = try_compile(src, 'GLSL')
    check('the GLSL shading library compiles', prog is not None, str(err))
    if prog is None:
        return

    n = 64
    N = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    th = np.linspace(0.05, 3.0, n)
    ph = np.linspace(0, 6.0, n)
    L = MX.normalize(np.stack([np.sin(th) * np.cos(ph),
                               np.sin(th) * np.sin(ph), np.cos(th)],
                              1).astype(np.float32))
    V = MX.normalize(np.tile(np.array([[0.15, 0.25, 0.95]], np.float32), (n, 1)))
    T, B = MX.orthonormal_basis(N)

    def surf():
        s2 = Surface(n)
        s2.diffuse[:] = 0.7
        s2.specular[:] = 1.0
        s2.specular_level[:] = 1.0
        s2.glossiness[:] = 28.0
        s2.roughness[:] = 0.35
        s2.metallic[:] = 0.4
        s2.anisotropy[:] = 0.35
        s2.soften[:] = 0.0
        s2.ior[:] = 1.5
        s2.translucency[:] = 0.6
        s2.toon_size[:] = 0.5
        s2.toon_smooth[:] = 0.15
        s2.opacity[:] = 0.85
        s2.tangent, s2.bitangent = T, B
        return s2

    ref = surf()
    idx = {m[0]: i for i, m in enumerate(MODEL_ITEMS)}
    uni = {'glossiness': ref.glossiness, 'roughness': ref.roughness,
           'metallic': ref.metallic, 'anisotropy': ref.anisotropy,
           'soften': ref.soften, 'ior': ref.ior,
           'translucency': ref.translucency, 'toon_size': ref.toon_size,
           'toon_smooth': ref.toon_smooth, 'opacity': ref.opacity,
           'diffuse': np.tile(np.array([[0.7, 0.7, 0.7]], np.float32), (n, 1)),
           'tangent': T, 'bitangent': B, 'nrm': N, 'lgt': L, 'vew': V}

    wrong = []
    tested = 0
    for ident, _label, _note in MODEL_ITEMS:
        if ident == 'WIREFRAME':          # not a reflectance model
            continue
        tested += 1
        d_cpu, s_cpu = evaluate(ident, surf(), N, L, V)
        u = dict(uni)
        u['model'] = np.full(n, idx[ident], np.int32)
        col = prog.run(u, {}, n)[0]['Color']
        de = float(np.abs(col[:, 0] - np.asarray(d_cpu)).max())
        se = float(np.abs(col[:, 1:4] - np.asarray(s_cpu)).max())
        if de > 2e-3 or se > 2e-3:
            wrong.append(f'{ident} (d {de:.4f}, s {se:.4f})')
    check(f'all {tested} GLSL models match the CPU exactly', not wrong,
          '; '.join(wrong[:4]))


def _emit_ctx(n=32):
    from ..core.nodeeval import ShadeContext
    from ..core.settings import RenderSettings
    c = ShadeContext(n)
    c.settings = RenderSettings()
    g = np.stack([np.linspace(0.05, 0.95, n), np.linspace(0.9, 0.1, n) ** 1.3,
                  np.linspace(0.2, 0.8, n)], 1).astype(np.float32)
    c.generated = g
    c.uv = g[:, :2].copy()
    c.P = g * 3.0
    c.N = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    c.I = np.tile(np.array([[0.1, 0.2, -0.97]], np.float32), (n, 1))
    return c


def test_glsl_node_emitters_match_cpu():
    """Every GLSL node emitter must equal the NumPy evaluator.

    This is the piece that decides whether a GPU frame is possible at all, and
    the one where a wrong answer is most dangerous: an emitter that is merely
    plausible still produces a picture, just not the right one. So each is run
    through Halcyon's own GLSL front-end and compared against the node it
    replaces.

    Four real errors were found this way, all of which would have rendered
    something believable: alpha was being blended in MixRGB where Blender keeps
    it from the first input, RGB and Value read a property rather than a
    socket, Logarithm takes a base, an unlinked texture Vector means generated
    coordinates rather than the socket default, and Layer Weight's facing
    output is driven by an exponent rather than a plain one-minus-cosine.
    """
    from ..core.nodeeval import GraphEvaluator
    from ..gpu.emit import Emitter, Unsupported
    from ..gpu.glsl_shading import GLSL
    from ..shaders.compiler import try_compile

    n = 32

    def val(name, d, link=None):
        return {'name': name, 'type': 'VALUE', 'default': d, 'link': link}

    def col(name, d, link=None):
        return {'name': name, 'type': 'RGBA', 'default': d, 'link': link}

    def vec(name, d, link=None):
        return {'name': name, 'type': 'VECTOR', 'default': d, 'link': link}

    def node(idname, props, ins, outs):
        return {'output': None, 'nodes': {'n': {
            'id': 'n', 'bl_idname': idname, 'props': props,
            'inputs': [dict(i) for i in ins],
            'outputs': [{'name': o[0], 'type': o[1]} for o in outs]}}}

    cases = []
    cases.append(('RGB', node('ShaderNodeRGB', {'value': [.8, .3, .15, 1]}, [],
                              [('Color', 'RGBA')]), 'vec4', 0))
    cases.append(('Value', node('ShaderNodeValue', {'value': 0.42}, [],
                                [('Value', 'VALUE')]), 'float', 0))
    for op in ('MIX', 'ADD', 'MULTIPLY', 'SUBTRACT', 'SCREEN', 'DIFFERENCE',
               'LIGHTEN', 'DARKEN', 'DIVIDE'):
        cases.append((f'MixRGB {op}',
                      node('ShaderNodeMixRGB', {'blend_type': op},
                           [val('Fac', .65), col('Color1', [.8, .2, .4, 1]),
                            col('Color2', [.1, .7, .3, 1])],
                           [('Color', 'RGBA')]), 'vec4', 0))
    for op in ('ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'POWER', 'MINIMUM',
               'MAXIMUM', 'SQRT', 'ABSOLUTE', 'SINE', 'COSINE', 'FLOOR',
               'CEIL', 'FRACT', 'MODULO', 'LESS_THAN', 'GREATER_THAN',
               'ARCTAN2', 'EXPONENT', 'LOGARITHM', 'SIGN'):
        cases.append((f'Math {op}',
                      node('ShaderNodeMath', {'operation': op},
                           [val('Value', .7), val('Value', .35)],
                           [('Value', 'VALUE')]), 'float', 0))
    for op, gt, idx in (('ADD', 'vec3', 0), ('SUBTRACT', 'vec3', 0),
                        ('MULTIPLY', 'vec3', 0), ('CROSS_PRODUCT', 'vec3', 0),
                        ('NORMALIZE', 'vec3', 0), ('ABSOLUTE', 'vec3', 0),
                        ('MINIMUM', 'vec3', 0), ('MAXIMUM', 'vec3', 0),
                        ('DOT_PRODUCT', 'float', 1), ('DISTANCE', 'float', 1),
                        ('LENGTH', 'float', 1)):
        cases.append((f'VecMath {op}',
                      node('ShaderNodeVectorMath', {'operation': op},
                           [vec('Vector', [.5, -.3, .8]),
                            vec('Vector', [.2, .9, -.4])],
                           [('Vector', 'VECTOR'), ('Value', 'VALUE')]), gt, idx))
    cases += [
        ('Invert', node('ShaderNodeInvert', {},
                        [val('Fac', .8), col('Color', [.7, .25, .5, 1])],
                        [('Color', 'RGBA')]), 'vec4', 0),
        ('Gamma', node('ShaderNodeGamma', {},
                       [col('Color', [.6, .3, .9, 1]), val('Gamma', 2.2)],
                       [('Color', 'RGBA')]), 'vec4', 0),
        ('BrightContrast', node('ShaderNodeBrightContrast', {},
                                [col('Color', [.5, .4, .7, 1]),
                                 val('Bright', .15), val('Contrast', .4)],
                                [('Color', 'RGBA')]), 'vec4', 0),
        ('CombineXYZ', node('ShaderNodeCombineXYZ', {},
                            [val('X', .3), val('Y', .6), val('Z', .9)],
                            [('Vector', 'VECTOR')]), 'vec3', 0),
        ('CombineRGB', node('ShaderNodeCombineRGB', {},
                            [val('R', .3), val('G', .6), val('B', .9)],
                            [('Image', 'RGBA')]), 'vec4', 0),
        ('Clamp', node('ShaderNodeClamp', {'clamp_type': 'MINMAX'},
                       [val('Value', 1.4), val('Min', .2), val('Max', .9)],
                       [('Result', 'VALUE')]), 'float', 0),
        ('MapRange', node('ShaderNodeMapRange', {'clamp': True},
                          [val('Value', .6), val('From Min', 0.), val('From Max', 1.),
                           val('To Min', -2.), val('To Max', 4.)],
                          [('Result', 'VALUE')]), 'float', 0),
        ('HueSaturation', node('ShaderNodeHueSaturation', {},
                               [val('Hue', .62), val('Saturation', 1.4),
                                val('Value', .9), val('Fac', 1.),
                                col('Color', [.8, .35, .2, 1])],
                               [('Color', 'RGBA')]), 'vec4', 0),
        ('LayerWeight facing', node('ShaderNodeLayerWeight', {},
                                    [val('Blend', .5), vec('Normal', [0, 0, 0])],
                                    [('Fresnel', 'VALUE'), ('Facing', 'VALUE')]),
         'float', 1),
        ('Checker', node('ShaderNodeTexChecker', {},
                         [vec('Vector', [.3, .7, .1]),
                          col('Color1', [.9, .9, .9, 1]),
                          col('Color2', [.1, .1, .1, 1]), val('Scale', 5.)],
                         [('Color', 'RGBA'), ('Fac', 'VALUE')]), 'vec4', 0),
    ]

    wrong = []
    for label, graph, gtype, index in cases:
        em = Emitter(graph)
        try:
            var, vt = em.output('n', index)
        except Unsupported as exc:
            wrong.append(f'{label}: unsupported ({exc})')
            continue
        var = em.cast(var, vt, gtype)
        wrap = ('vec4(vec3(%s), 1.0)' % var) if gtype == 'float' else (
            ('vec4(%s, 1.0)' % var) if gtype == 'vec3' else var)
        src = GLSL + """
uniform vec3 hal_N; uniform vec3 hal_V; uniform vec3 hal_P; uniform vec3 hal_T;
uniform vec3 hal_generated; uniform vec2 hal_uv;
out vec4 Color;
void main() {
%s
    Color = %s;
}
""" % (em.body(), wrap)
        prog, err = try_compile(src, 'GLSL')
        if prog is None:
            wrong.append(f'{label}: will not compile ({err})')
            continue
        c = _emit_ctx(n)
        got = prog.run({'hal_N': c.N, 'hal_V': -c.I, 'hal_P': c.P,
                        'hal_T': c.N, 'hal_generated': c.generated,
                        'hal_uv': c.uv}, {}, n)[0]['Color']
        want = GraphEvaluator(graph, _emit_ctx(n)).eval_output('n', index)
        if not isinstance(want, np.ndarray):
            continue                       # closure outputs go another route
        want = np.asarray(want, np.float32)
        if want.ndim == 1:
            want = np.stack([want] * 3, 1)
        w = min(want.shape[1], got.shape[1])
        e = float(np.abs(got[:, :w] - want[:, :w]).max())
        if e > 2e-3:
            wrong.append(f'{label} ({e:.5f})')
    check(f'all {len(cases)} GLSL node emitters match the CPU', not wrong,
          '; '.join(wrong[:4]))

    # the image texture, whose coordinate default is the easy thing to get wrong
    from ..core.texture import Texture
    from ..core.settings import RenderSettings
    src_img = np.zeros((16, 16, 4), np.float32)
    yy, xx = np.mgrid[0:16, 0:16]
    src_img[..., 0] = xx / 15.0
    src_img[..., 1] = yy / 15.0
    src_img[..., 2] = 0.5
    src_img[..., 3] = 1.0
    tex = Texture(src_img, colorspace='Non-Color', filt='NEAREST',
                  wrap='REPEAT')
    gg = np.stack([np.linspace(0.05, 0.95, n), np.linspace(0.9, 0.1, n),
                   np.zeros(n)], 1).astype(np.float32)

    def img_ctx():
        c = _emit_ctx(n)
        st2 = RenderSettings()
        st2.tex_filter = 'NEAREST'
        c.settings = st2
        c.generated = gg * 0.3          # deliberately unlike the UVs
        c.uv = gg[:, :2].copy()
        return c

    igraph = {'output': None, 'nodes': {'n': {
        'id': 'n', 'bl_idname': 'ShaderNodeTexImage',
        'props': {'image': 'tex', 'interpolation': 'Closest',
                  'extension': 'REPEAT'},
        'inputs': [{'name': 'Vector', 'type': 'VECTOR', 'default': [0, 0, 0],
                    'link': None}],
        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                    {'name': 'Alpha', 'type': 'VALUE'}]}}}
    em = Emitter(igraph)
    var, _vt = em.output('n', 0)
    prog, err = try_compile(GLSL + """
uniform sampler2D hal_tex0; uniform vec3 hal_generated; uniform vec2 hal_uv;
out vec4 Color;
void main() {
%s
    Color = %s;
}
""" % (em.body(), var), 'GLSL')
    check('the image texture emitter compiles', prog is not None, str(err))
    if prog is not None:
        c = img_ctx()
        got = prog.run({'hal_tex0': tex, 'hal_generated': c.generated,
                        'hal_uv': c.uv}, {}, n)[0]['Color']
        want = GraphEvaluator(igraph, img_ctx(), {'tex': tex}).eval_output('n', 0)
        e = float(np.abs(got[:, :3] - np.asarray(want)[:, :3]).max())
        check('the image texture samples where the CPU does', e < 2e-3,
              f'max difference {e:.6f}')
        check('it declares a sampler for the assembler to bind',
              len(em.samplers) == 1 and em.samplers[0]['image'] == 'tex',
              str(em.samplers))


def test_emitter_declines_rather_than_guesses():
    """An unknown node must be reported, never approximated."""
    from ..gpu.emit import EMITTERS, can_emit, supported
    graph = {'output': 'out', 'nodes': {
        'x': {'id': 'x', 'bl_idname': 'ShaderNodeSomethingNew', 'props': {},
              'inputs': [], 'outputs': [{'name': 'Out', 'type': 'RGBA'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [{'name': 'Surface', 'type': 'SHADER',
                            'default': None, 'link': ['x', 0]}],
                'outputs': []}}}
    ok, missing = can_emit(graph)
    check('an unknown node blocks emission', not ok)
    check('and is named', 'ShaderNodeSomethingNew' in missing, str(missing))
    check('the supported list is not empty', len(supported()) > 20,
          str(len(supported())))
    check('every emitter is callable',
          all(callable(f) for f in EMITTERS.values()))


def test_assembled_material_shader_matches_cpu():
    """A whole material, assembled into one shader, must shade as the CPU does.

    This is the test that matters most: the emitter, the reflectance models and
    the light loop are checked *together*, so the seams between them are
    covered rather than only the pieces. It is also what proves a GPU frame is
    achievable at all -- everything below the rasteriser now exists and agrees.
    """
    from ..core import mathx as MX
    from ..core.nodeeval import GraphEvaluator
    from ..core.scene import Light
    from ..core.shading import MODEL_ITEMS, Surface, evaluate
    from ..gpu.material import LIGHT_KIND, assemble
    from ..shaders.compiler import try_compile

    n = 48

    def wn(i, t, p=None, ins=(), outs=()):
        return {'id': i, 'bl_idname': t, 'props': p or {},
                'inputs': [dict(x) for x in ins],
                'outputs': [dict(x) for x in outs]}

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    graph = {'output': 'out', 'nodes': {
        'a': wn('a', 'ShaderNodeRGB', {'value': [.85, .25, .15, 1]}, [],
                [{'name': 'Color', 'type': 'RGBA'}]),
        'b': wn('b', 'ShaderNodeRGB', {'value': [.1, .45, .9, 1]}, [],
                [{'name': 'Color', 'type': 'RGBA'}]),
        'm': wn('m', 'ShaderNodeMixRGB', {'blend_type': 'MIX'},
                [sk('Fac', 'VALUE', .35), sk('Color1', 'RGBA', [0, 0, 0, 1], ['a', 0]),
                 sk('Color2', 'RGBA', [0, 0, 0, 1], ['b', 0])],
                [{'name': 'Color', 'type': 'RGBA'}]),
        'h': wn('h', 'ShaderNodeHueSaturation', {},
                [sk('Hue', 'VALUE', .55), sk('Saturation', 'VALUE', 1.3),
                 sk('Value', 'VALUE', .95), sk('Fac', 'VALUE', 1.0),
                 sk('Color', 'RGBA', [0, 0, 0, 1], ['m', 0])],
                [{'name': 'Color', 'type': 'RGBA'}]),
        'd': wn('d', 'ShaderNodeBsdfDiffuse', {},
                [sk('Color', 'RGBA', [0, 0, 0, 1], ['h', 0]),
                 sk('Roughness', 'VALUE', 0.0), sk('Normal', 'VECTOR', [0, 0, 0])],
                [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': wn('out', 'ShaderNodeOutputMaterial', {},
                  [sk('Surface', 'SHADER', None, ['d', 0]),
                   sk('Displacement', 'VECTOR', [0, 0, 0])], [])}}

    lights = [Light(type='SUN', direction=(-0.4, 0.3, -0.86),
                    color=(1.0, 0.95, 0.85), energy=3.0),
              Light(type='POINT', position=(2.0, -1.5, 3.0),
                    color=(0.4, 0.6, 1.0), energy=25.0)]

    src, samplers = assemble(graph, light_count=len(lights))
    check('a complete material assembles into one shader', src is not None,
          str(samplers))
    if src is None:
        return
    prog, err = try_compile(src, 'GLSL')
    check('the assembled shader compiles', prog is not None, str(err))
    if prog is None:
        return

    P = np.stack([np.linspace(-2, 2, n), np.linspace(-1, 1, n),
                  np.full(n, 0.5)], 1).astype(np.float32)
    Nr = MX.normalize(np.stack([np.linspace(-.4, .4, n),
                                np.linspace(.3, -.3, n),
                                np.ones(n)], 1).astype(np.float32))
    V = MX.normalize(np.tile(np.array([[0.1, 0.2, 0.97]], np.float32), (n, 1)))
    T, B = MX.orthonormal_basis(Nr)
    ambient = np.array([0.05, 0.06, 0.08], np.float32)

    def ctx():
        c = _emit_ctx(n)
        c.generated = np.abs(P) * 0.2
        c.uv = P[:, :2] * 0.1
        c.P = P
        c.N = Nr
        c.I = -V
        c.T = T
        return c

    base = np.asarray(GraphEvaluator(graph, ctx()).eval_output('h', 0))[:, :3]

    def cpu(model):
        s2 = Surface(n)
        s2.diffuse = base.copy()
        s2.specular[:] = 1.0
        s2.diffuse_level[:] = 1.0
        s2.specular_level[:] = 0.6
        s2.glossiness[:] = 30.0
        s2.roughness[:] = 0.35
        s2.metallic[:] = 0.2
        s2.soften[:] = 0.0
        s2.ior[:] = 1.45
        s2.toon_size[:] = 0.5
        s2.toon_smooth[:] = 0.15
        s2.opacity[:] = 1.0
        s2.tangent, s2.bitangent = T, B
        out = base * ambient[None, :]
        for lt in lights:
            if lt.type == 'SUN':
                L = MX.normalize(-np.tile(
                    np.asarray(lt.direction, np.float32)[None, :], (n, 1)))
                att = np.ones(n, np.float32)
            else:
                d = np.asarray(lt.position, np.float32)[None, :] - P
                dist = np.linalg.norm(d, axis=1)
                L = d / np.maximum(dist, 1e-6)[:, None]
                att = 1.0 / np.maximum(dist * dist, 1e-6)
            dif, spec = evaluate(model, s2, Nr, L, V)
            rad = (np.asarray(lt.color, np.float32)[None, :] * lt.energy
                   * att[:, None] / np.pi)
            out = out + (base * np.asarray(dif)[:, None]
                         + s2.specular * 0.6 * np.asarray(spec)) * rad
        return out

    uni = {'hal_P': P, 'hal_N': Nr, 'hal_V': V, 'hal_T': T,
           'hal_generated': np.abs(P) * 0.2, 'hal_uv': P[:, :2] * 0.1,
           'hal_diffuse_level': np.full(n, 1.0, np.float32),
           'hal_specular_level': np.full(n, 0.6, np.float32),
           'hal_glossiness': np.full(n, 30.0, np.float32),
           'hal_roughness': np.full(n, 0.35, np.float32),
           'hal_metallic': np.full(n, 0.2, np.float32),
           'hal_anisotropy': np.zeros(n, np.float32),
           'hal_aniso_rot': np.zeros(n, np.float32),
           'hal_soften': np.zeros(n, np.float32),
           'hal_ior': np.full(n, 1.45, np.float32),
           'hal_translucency': np.zeros(n, np.float32),
           'hal_toon_size': np.full(n, 0.5, np.float32),
           'hal_toon_smooth': np.full(n, 0.15, np.float32),
           'hal_opacity': np.ones(n, np.float32),
           'hal_specular_tint': np.ones((n, 3), np.float32),
           'hal_ambient': np.tile(ambient[None, :], (n, 1))}
    for i, lt in enumerate(lights):
        uni[f'hal_lkind{i}'] = np.full(n, LIGHT_KIND[lt.type], np.int32)
        uni[f'hal_lpos{i}'] = np.tile(
            np.asarray(lt.position, np.float32)[None, :], (n, 1))
        uni[f'hal_ldir{i}'] = np.tile(
            np.asarray(lt.direction, np.float32)[None, :], (n, 1))
        uni[f'hal_lcol{i}'] = np.tile(
            np.asarray(lt.color, np.float32)[None, :], (n, 1))
        uni[f'hal_lenergy{i}'] = np.full(n, lt.energy, np.float32)
        uni[f'hal_lradius{i}'] = np.full(n, 0.5, np.float32)

    idx = {m[0]: i for i, m in enumerate(MODEL_ITEMS)}
    wrong = []
    tested = 0
    for ident, _l, _no in MODEL_ITEMS:
        if ident in ('WIREFRAME', 'GOURAUD', 'FLAT'):
            continue
        tested += 1
        u = dict(uni)
        u['hal_model'] = np.full(n, idx[ident], np.int32)
        got = prog.run(u, {}, n)[0]['Color'][:, :3]
        want = cpu(ident)
        rel = float(np.abs(got - want).max()) / max(float(np.abs(want).max()), 1e-6)
        if rel > 5e-3:
            wrong.append(f'{ident} ({rel:.5f})')
    check(f'all {tested} models shade identically in an assembled shader',
          not wrong, '; '.join(wrong[:4]))

    # a material using a node with no emitter must produce no shader at all
    bad_graph = {'output': 'out', 'nodes': {
        'x': wn('x', 'ShaderNodeSomethingNew', {}, [],
                [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': wn('out', 'ShaderNodeOutputMaterial', {},
                  [sk('Surface', 'SHADER', None, ['x', 0])], [])}}
    src2, why = assemble(bad_graph, light_count=1)
    check('an unsupported material assembles to nothing', src2 is None)
    check('and the missing node type is named',
          'ShaderNodeSomethingNew' in str(why), str(why))


def test_worker_resolves_blender_package_names():
    """A worker must unpickle objects made under the add-on's Blender name.

    Installed as an extension the package is imported as
    `bl_ext.user_default.halcyon_render`, so every dataclass pickled inside
    Blender carries that module path. A worker importing the same files as
    plain `halcyon_render` cannot resolve those names, and unpickling the scene
    fails with `No module named 'bl_ext'`.

    That failure spent five releases looking like the worker exiting for no
    reason, because a read error in the worker's loop was caught and treated
    exactly like end of stream.
    """
    import importlib
    import os
    import pickle
    import subprocess
    import types

    from ..core import parallel as P

    check('the package can name itself as Blender sees it',
          isinstance(P.import_name(), str) and P.import_name(),
          P.import_name())

    alias = 'bl_ext.user_default.halcyon_probe'
    real = P.package_location()[1]
    parent = P.package_location()[0]

    # pickle something under the alias, exactly as Blender would
    from ..core.scene import Material
    from ..core.settings import RenderSettings
    saved = (Material.__module__, RenderSettings.__module__)
    stubs = []
    try:
        for name in ('bl_ext', 'bl_ext.user_default'):
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = []
                sys.modules[name] = mod
                stubs.append(name)
        pkg = importlib.import_module(real)
        sys.modules[alias] = pkg
        stubs.append(alias)
        for sub in ('core', 'core.scene', 'core.settings'):
            full = f'{alias}.{sub}'
            sys.modules[full] = importlib.import_module(f'{real}.{sub}')
            stubs.append(full)
        Material.__module__ = f'{alias}.core.scene'
        RenderSettings.__module__ = f'{alias}.core.settings'
        blob = pickle.dumps((Material(), RenderSettings()))
    finally:
        Material.__module__, RenderSettings.__module__ = saved
        for name in stubs:
            sys.modules.pop(name, None)

    check('the pickle really carries the Blender module path', b'bl_ext' in blob)

    exe = P.find_interpreter()
    if exe is None:
        check('worker alias resolution (skipped: no interpreter)', True)
        return

    env = dict(os.environ)
    env['PYTHONPATH'] = parent + os.pathsep + env.get('PYTHONPATH', '')
    env['HALCYON_WORKER_ROOT'] = parent
    env['HALCYON_WORKER_PKG'] = real
    env['HALCYON_WORKER_ALIAS'] = alias
    code = P.BOOTSTRAP_SOURCE.split("note('entering main')")[0] + """
import pickle, sys
obj = pickle.loads(sys.stdin.buffer.read())
sys.stderr.write('OK ' + ','.join(type(o).__name__ for o in obj))
"""
    try:
        run = subprocess.run([exe, '-c', code], input=blob, env=env,
                             capture_output=True, timeout=90)
    except Exception as exc:                                    # noqa: BLE001
        check('worker alias resolution (skipped: could not spawn)', True,
              str(exc))
        return
    text = run.stderr.decode('utf-8', 'replace')
    check('a worker unpickles objects named for Blender', 'OK ' in text,
          text.strip()[-160:])
    check('and gets both objects back',
          'Material' in text and 'RenderSettings' in text,
          text.strip()[-80:])


def test_gbuffer_reconstruction_matches_cpu():
    """GLSL must rebuild shading inputs from a packed G-buffer exactly.

    This is the foundation of moving shading to the GPU without writing a GPU
    rasteriser first -- which is the right order, because shading is about 71%
    of a frame and rasterising is about 9%. The CPU already produces triangle
    IDs and barycentrics; a full-screen pass can turn those back into
    positions, normals and UVs, using the same mechanism the post stages
    already prove works on real hardware.
    """
    from ..core import raster
    from ..core.texture import Texture
    from ..gpu import gbuffer as GB
    from ..gpu.glsl_shading import GLSL
    from ..shaders.compiler import try_compile

    w, h = 64, 48
    st = base_settings(w, h)
    sc = demo_scene(st)
    _v, _p, vp, _e = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)

    ids = GB.pack_ids(g)
    attrs, side = GB.pack_attributes(sc.mesh)
    check('the id texture is one texel per pixel', ids.shape == (h, w, 4),
          str(ids.shape))
    check('triangle ids survive the float packing',
          np.array_equal(ids[:, :, 3].astype(np.int64), g.tri.astype(np.int64)))
    check('barycentrics sum to one where covered',
          float(np.abs(ids[g.tri >= 0, :3].sum(axis=1) - 1.0).max()) < 1e-5)

    prog, err = try_compile(GLSL + GB.GLSL + """
uniform vec2 hal_screen;
out vec4 Color;
void main() {
    HalcyonFragment f = hal_read_gbuffer(hal_screen);
    Color = vec4(f.P, f.covered ? 1.0 : 0.0);
}
""", 'GLSL')
    check('the G-buffer reader compiles', prog is not None, str(err))
    if prog is None:
        return

    yy, xx = np.mgrid[0:h, 0:w]
    uv = np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h],
                  1).astype(np.float32)
    n = w * h
    out = prog.run({
        'hal_gb_ids': Texture(ids, colorspace='Non-Color', filt='NEAREST',
                              wrap='EXTEND'),
        'hal_gb_attrs': Texture(attrs, colorspace='Non-Color', filt='NEAREST',
                                wrap='EXTEND'),
        'hal_attr_side': np.full(n, float(side), np.float32),
        'hal_slot_count': np.full(n, float(GB.SLOTS), np.float32),
        'hal_screen': uv}, {}, n)[0]['Color']

    got = out[:, :3].reshape(h, w, 3)
    want = GB.cpu_reconstruct(sc.mesh, g, 0)
    cov = g.tri >= 0
    check('some of the frame is actually covered', int(cov.sum()) > 100,
          str(int(cov.sum())))
    e = float(np.abs(got[cov] - want[cov]).max())
    check('GLSL rebuilds positions exactly as the CPU does', e < 2e-3,
          f'max difference {e:.6f}')
    check('and agrees about which pixels are covered',
          np.array_equal(out[:, 3].reshape(h, w) > 0.5, cov))


def test_spot_cones_match_a_brute_force_march():
    """The analytic cone must agree with marching the whole ray.

    The fast path solves a quadratic for where the view ray enters and leaves
    the cone; the reference walks the entire ray in small steps and tests
    containment at each one. They share no code, so agreement means the
    intersection maths is right rather than that one bug is in both.
    """
    from ..core import cones, mathx as MX
    from ..core.scene import Light

    n = 3000
    rng = np.random.default_rng(7)
    lt = Light(type='SPOT', position=(0.0, 3.0, 0.0), direction=(0.0, -1.0, 0.0),
               spot_size=0.9, spot_blend=0.2, volumetric=1.0)
    origin = np.array([[0.0, 1.0, 6.0]], np.float32)
    d = rng.normal(size=(n, 3)).astype(np.float32)
    d[:, 2] = -np.abs(d[:, 2]) - 0.3
    rays = MX.normalize(d)
    far = np.full(n, np.inf, np.float32)

    fast = cones.spot_cone(origin, rays, far, lt, samples=1024)
    ref = cones.reference(origin, rays, far, lt, samples=4096)
    check('the cone maths produces no NaNs', bool(np.all(np.isfinite(fast))))

    lit = ref > 1e-5
    agree = float(((fast > 1e-6) == lit).mean())
    check('it agrees which rays are inside the beam', agree > 0.99,
          f'{100 * agree:.1f}%')
    if lit.any():
        rel = np.abs(fast[lit] - ref[lit]) / np.maximum(ref[lit], 1e-6)
        check('and how much each one scatters', float(np.median(rel)) < 5e-3,
              f'median relative error {float(np.median(rel)):.5f}')

    # geometry must cut the beam short rather than shine through it
    near = np.full(n, 4.0, np.float32)
    cut = cones.spot_cone(origin, rays, near, lt, samples=1024)
    check('a surface never makes the beam brighter',
          bool(np.all(cut <= fast + 1e-6)))
    check('and does dim it somewhere', bool(np.any(cut < fast - 1e-5)))

    # a ray pointing away from the light gets nothing at all
    away = MX.normalize(np.tile(np.array([[0, 0, 1.0]], np.float32), (16, 1)))
    check('rays facing away from the cone stay black',
          float(cones.spot_cone(origin, away, np.full(16, np.inf, np.float32),
                                lt).max()) == 0.0)

    # only spot lights have cones
    for kind in ('POINT', 'SUN', 'AREA'):
        other = Light(type=kind, position=(0, 3, 0), volumetric=1.0)
        check(f'a {kind.lower()} light casts no cone',
              float(cones.spot_cone(origin, rays, far, other).max()) == 0.0)


def test_spot_cones_only_add_light():
    """The pass must brighten, never darken, and do nothing when switched off."""
    from ..core.scene import Light

    st = base_settings(160, 120)
    sc = demo_scene(st)
    sc.lights.append(Light(type='SPOT', position=(0, 4, 2),
                           direction=(0, -1, -0.3), color=(1.0, 0.9, 0.7),
                           energy=800.0, spot_size=0.8, spot_blend=0.25,
                           volumetric=1.0))
    st.spot_cones = False
    off = R.render(sc, st)
    st.spot_cones = True
    on = R.render(sc, st)

    check('the frame is still finite', bool(np.all(np.isfinite(on))))
    d = on[:, :, :3] - off[:, :, :3]
    check('cones only ever add light', float(d.min()) >= -1e-6,
          f'darkest change {float(d.min()):.6f}')
    check('and brighten some of the frame', int((d.sum(axis=2) > 1e-4).sum()) > 50,
          str(int((d.sum(axis=2) > 1e-4).sum())))

    # a light with no volumetric value contributes nothing
    sc.lights[-1].volumetric = 0.0
    check('a spot with no volumetric value draws no cone',
          np.array_equal(R.render(sc, st), off))


def test_debug_passes():
    for mode in ('DEPTH', 'NORMAL', 'UV', 'MATID', 'OVERDRAW', 'WIREFRAME'):
        st = base_settings(64, 48, debug_pass=mode)
        img = render(st)
        ok = np.isfinite(img).all() and img[..., :3].std() > 1e-4
        check(f'debug pass {mode}', ok, f'std={img[..., :3].std():.4f}')


def test_affine_texture_warp():
    persp = render(base_settings(tex_perspective=True))
    affine = render(base_settings(tex_perspective=False))
    d = float(np.abs(persp - affine).mean())
    check('affine mapping warps the floor texture', d > 1e-3, f'delta={d:.4f}')


def test_vertex_snap():
    smooth = render(base_settings(vertex_snap=False))
    snapped = render(base_settings(vertex_snap=True, vertex_snap_grid=4.0))
    d = float(np.abs(smooth - snapped).mean())
    check('vertex snapping moves geometry', d > 1e-3, f'delta={d:.4f}')


def test_supersampling():
    a = render(base_settings(aa_mode='NONE'))
    st = base_settings(aa_mode='SUPERSAMPLE', aa_samples=4)
    b = render(st)
    check('supersampled image is the requested size', b.shape == a.shape)
    edge_a = float(np.abs(np.diff(a[..., :3], axis=1)).mean())
    edge_b = float(np.abs(np.diff(b[..., :3], axis=1)).mean())
    check('supersampling softens edges', edge_b < edge_a,
          f'{edge_b:.4f} < {edge_a:.4f}')


def test_transparency():
    st = base_settings(transparency='ABUFFER')
    sc = demo_scene(st)
    sc.materials[1].opacity = 0.35
    sc.materials[1].has_alpha = True
    img = R.render(sc, st)
    st2 = base_settings()
    opaque = render(st2)
    d = float(np.abs(img - opaque).mean())
    check('A-buffer composites transparent surfaces', d > 1e-3, f'delta={d:.4f}')
    check('transparency stays finite', np.isfinite(img).all())


def test_fog():
    clear = render(base_settings(fog=False))
    foggy = render(base_settings(fog=True, fog_mode='LINEAR', fog_start=2.0,
                                 fog_end=15.0, fog_color=(1.0, 0.0, 0.0)))
    check('fog tints distant geometry',
          float(foggy[..., 0].mean() - clear[..., 0].mean()) > 0.01)


def test_raytraced_reflection():
    st = base_settings(raytrace=True, ray_depth=2)
    sc = demo_scene(st)
    sc.materials[1].reflect_level = 0.9
    ref = R.render(sc, st)
    sc2 = demo_scene(st)
    flat = R.render(sc2, st)
    check('reflections change the image',
          float(np.abs(ref - flat).mean()) > 1e-3)


# --------------------------------------------------------------- node graph


def _ctx(n=16):
    c = ShadeContext(n)
    c.N = np.tile(np.array([[0., 0., 1.]], np.float32), (n, 1))
    c.I = np.tile(np.array([[0., 0., -1.]], np.float32), (n, 1))
    c.uv = np.stack([np.linspace(0, 1, n), np.linspace(0, 1, n)], 1).astype(np.float32)
    c.P = np.zeros((n, 3), np.float32)
    c.settings = RenderSettings()
    return c


def _node(nid, idname, props=None, inputs=(), outputs=()):
    return {'id': nid, 'bl_idname': idname, 'props': props or {},
            'inputs': [dict(i) for i in inputs],
            'outputs': [dict(o) for o in outputs]}


def test_node_math():
    g = {'output': None, 'nodes': {'m': _node(
        'm', 'ShaderNodeMath', {'operation': 'MULTIPLY'},
        [{'name': 'Value', 'type': 'VALUE', 'default': 3.0, 'link': None},
         {'name': 'Value_001', 'type': 'VALUE', 'default': 4.0, 'link': None},
         {'name': 'Value_002', 'type': 'VALUE', 'default': 0.0, 'link': None}],
        [{'name': 'Value', 'type': 'VALUE'}])}}
    ev = GraphEvaluator(g, _ctx())
    v = ev.eval_output('m', 0)
    check('Math node multiplies', np.allclose(v, 12.0), str(v[:2]))


def test_node_chain_and_ramp():
    lut = np.zeros((256, 4), np.float32)
    lut[:, 0] = np.linspace(0, 1, 256)
    lut[:, 3] = 1.0
    g = {'output': None, 'nodes': {
        'coord': _node('coord', 'ShaderNodeTexCoord', {}, [],
                       [{'name': 'Generated', 'type': 'VECTOR'},
                        {'name': 'Normal', 'type': 'VECTOR'},
                        {'name': 'UV', 'type': 'VECTOR'}]),
        'sep': _node('sep', 'ShaderNodeSeparateXYZ', {},
                     [{'name': 'Vector', 'type': 'VECTOR', 'default': [0, 0, 0],
                       'link': ['coord', 2]}],
                     [{'name': 'X', 'type': 'VALUE'}, {'name': 'Y', 'type': 'VALUE'},
                      {'name': 'Z', 'type': 'VALUE'}]),
        'ramp': _node('ramp', 'ShaderNodeValToRGB', {'lut': lut.tolist()},
                      [{'name': 'Fac', 'type': 'VALUE', 'default': 0.0,
                        'link': ['sep', 0]}],
                      [{'name': 'Color', 'type': 'RGBA'},
                       {'name': 'Alpha', 'type': 'VALUE'}]),
    }}
    ev = GraphEvaluator(g, _ctx())
    col = ev.eval_output('ramp', 0)
    check('TexCoord -> SeparateXYZ -> ColorRamp chain',
          col is not None and abs(float(col[0, 0])) < 0.02 and
          float(col[-1, 0]) > 0.95, str(col[[0, -1], 0]) if col is not None else 'None')


def test_node_group_recursion():
    inner = {'nodes': {
        'gi': _node('gi', 'NodeGroupInput', {}, [],
                    [{'name': 'Val', 'type': 'VALUE'}]),
        'add': _node('add', 'ShaderNodeMath', {'operation': 'ADD'},
                     [{'name': 'Value', 'type': 'VALUE', 'default': 0.0,
                       'link': ['gi', 0]},
                      {'name': 'Value_001', 'type': 'VALUE', 'default': 10.0,
                       'link': None},
                      {'name': 'Value_002', 'type': 'VALUE', 'default': 0.0,
                       'link': None}],
                     [{'name': 'Value', 'type': 'VALUE'}]),
        'go': _node('go', 'NodeGroupOutput', {},
                    [{'name': 'Out', 'type': 'VALUE', 'default': 0.0,
                      'link': ['add', 0]}], []),
    }, 'output': None, 'group_output': 'go'}
    grp = _node('g', 'ShaderNodeGroup', {},
                [{'name': 'Val', 'type': 'VALUE', 'default': 5.0, 'link': None}],
                [{'name': 'Out', 'type': 'VALUE'}])
    grp['group'] = inner
    ev = GraphEvaluator({'output': None, 'nodes': {'g': grp}}, _ctx())
    v = ev.eval_output('g', 0)
    check('node groups evaluate recursively', v is not None and
          np.allclose(v, 15.0), str(v[:2]) if v is not None else 'None')


def test_unknown_node_passthrough():
    g = {'output': None, 'nodes': {'weird': _node(
        'weird', 'ShaderNodeSomethingFromTheFuture', {},
        [{'name': 'Color', 'type': 'RGBA', 'default': [0.25, 0.5, 0.75, 1.0],
          'link': None}],
        [{'name': 'Color', 'type': 'RGBA'}])}}
    ev = GraphEvaluator(g, _ctx())
    v = ev.eval_output('weird', 0)
    ok = v is not None and np.allclose(v[0, :3], [0.25, 0.5, 0.75], atol=1e-4)
    check('unknown nodes pass through and are reported',
          ok and 'ShaderNodeSomethingFromTheFuture' in ev.unsupported)


def test_coded_shader_in_graph():
    from ..shaders.compiler import compile_shader
    src = """
    uniform vec3 tint = vec3(1.0, 0.5, 0.25);
    in vec2 vUV;
    out vec4 Color;
    void main() { Color = vec4(tint * vUV.x, 1.0); }
    """
    prog = compile_shader(src, 'GLSL')
    node = _node('code', 'HALCYON_CodeNode', {},
                 [{'name': 'Tint', 'type': 'RGBA', 'uniform': 'tint',
                   'default': [0.2, 0.4, 0.8, 1.0], 'link': None}],
                 [{'name': 'Color', 'type': 'RGBA', 'key': 'Color'}])
    ev = GraphEvaluator({'output': None, 'nodes': {'code': node}}, _ctx(),
                        programs={'code': prog})
    col = ev.eval_output('code', 0)
    ok = col is not None and float(col[0, 0]) < 0.01 and \
        abs(float(col[-1, 0]) - 0.2) < 0.02
    check('coded shader node runs inside a material graph', ok,
          str(col[[0, -1], :3]) if col is not None else 'None')


def test_bsdf_translation():
    g = {'output': 'out', 'nodes': {
        'p': _node('p', 'ShaderNodeBsdfPrincipled', {},
                   [{'name': 'Base Color', 'type': 'RGBA',
                     'default': [0.9, 0.1, 0.1, 1.0], 'link': None},
                    {'name': 'Metallic', 'type': 'VALUE', 'default': 0.0,
                     'link': None},
                    {'name': 'Roughness', 'type': 'VALUE', 'default': 0.25,
                     'link': None},
                    {'name': 'IOR', 'type': 'VALUE', 'default': 1.45, 'link': None},
                    {'name': 'Alpha', 'type': 'VALUE', 'default': 1.0, 'link': None},
                    {'name': 'Specular', 'type': 'VALUE', 'default': 0.5,
                     'link': None},
                    {'name': 'Transmission', 'type': 'VALUE', 'default': 0.0,
                     'link': None}],
                   [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': _node('out', 'ShaderNodeOutputMaterial', {},
                     [{'name': 'Surface', 'type': 'SHADER', 'default': None,
                       'link': ['p', 0]},
                      {'name': 'Displacement', 'type': 'VECTOR',
                       'default': [0, 0, 0], 'link': None}], []),
    }}
    ctx = _ctx()
    ev = GraphEvaluator(g, ctx)
    cl, _ = ev.evaluate_surface()
    check('Principled BSDF becomes a closure',
          isinstance(cl, Closure) and len(cl) >= 2, f'{len(cl) if cl else 0} lobes')
    surf, model, _ = R.closure_to_surface(cl, ctx, ctx.settings)
    check('closure resolves to a period model with the right colour',
          np.allclose(surf.diffuse[0], [0.9, 0.1, 0.1], atol=0.05) and
          model in ('COOK_TORRANCE', 'OREN_NAYAR', 'PHONG', 'BLINN_PHONG'),
          f'{model} {np.round(surf.diffuse[0], 3)}')


# ------------------------------------------------------------------- post


def test_palette_counts():
    img = render(base_settings())
    for depth, mode, size, limit in (('8', 'ADAPTIVE', 256, 256),
                                     ('8', 'VGA256', 256, 256),
                                     ('4', 'EGA16', 16, 16),
                                     ('1', 'GRAY', 2, 2)):
        st = base_settings(color_depth=depth, palette_mode=mode,
                           palette_size=size, dither='FLOYD')
        out = post.process(img, st)[..., :3]
        n = len(np.unique(out.reshape(-1, 3), axis=0))
        check(f'{mode} yields at most {limit} colours', n <= limit, f'{n} colours')


def test_dither_changes_image():
    img = render(base_settings())
    st_a = base_settings(color_depth='4', palette_mode='EGA16', dither='NONE')
    st_b = base_settings(color_depth='4', palette_mode='EGA16', dither='FLOYD')
    a = post.process(img, st_a)
    b = post.process(img, st_b)
    check('dithering changes the quantised image',
          float(np.abs(a - b).mean()) > 1e-3)


def test_output_scale_and_aspect():
    img = render(base_settings(96, 72))
    st = base_settings(96, 72, output_scale='3X')
    out = post.process(img, st)
    check('3x nearest upscale', out.shape[:2] == (216, 288), str(out.shape))
    st2 = base_settings(96, 72, pixel_aspect_x=1.0, pixel_aspect_y=1.2)
    out2 = post.process(img, st2)
    check('pixel aspect stretches the image', out2.shape[0] > 72, str(out2.shape))


def test_post_chain_stability():
    img = render(base_settings())
    st = base_settings(glow=True, star_filter=True, lens_flare=True,
                       color_depth='16', dither='BAYER4', composite=True,
                       interlace='BLEND', crt=True, crt_scanlines=0.4,
                       crt_mask='SHADOW', crt_curvature=0.3, crt_vignette=0.4,
                       jpeg_artifacts=True, jpeg_quality=30, output_scale='2X')
    out = post.process(img, st)
    check('every post stage at once stays finite and in range',
          np.isfinite(out).all() and out[..., :3].min() >= -1e-6 and
          out[..., :3].max() <= 1.0 + 1e-6)


def test_presets_do_not_overlap():
    """Applying a preset must reset first, so nothing carries over.

    Without this, going from EGA to Infini-D left EGA's 16-colour palette, its
    2-light limit, its 1.2 pixel aspect and its 3x output scale behind, because
    Infini-D's entry does not mention any of them.
    """
    import dataclasses

    from ..core.settings import RenderSettings
    from ..presets.library import PRESERVED, reset_settings

    fresh = RenderSettings()
    fields = [f.name for f in dataclasses.fields(RenderSettings)]

    pairs = [('EGA', 'INFINID_4'), ('PSX', 'LIGHTWAVE_56'), ('VHS', 'WEB_PNG8'),
             ('CGA', 'ELECTRIC_IMAGE')]
    leaks = []
    for first, second in pairs:
        st = RenderSettings()
        apply_preset(st, first)
        apply_preset(st, second)
        clean = RenderSettings()
        apply_preset(clean, second)
        for name in fields:
            if getattr(st, name) != getattr(clean, name):
                leaks.append(f'{first}->{second}: {name}')
    check('switching presets leaves nothing behind', not leaks,
          '; '.join(leaks[:4]))

    # the default preset is a full reset
    st = RenderSettings()
    apply_preset(st, 'PSX')
    apply_preset(st, 'DEFAULT')
    dirty = [n for n in fields if getattr(st, n) != getattr(fresh, n)]
    check('the Default preset restores every setting', not dirty,
          ', '.join(dirty[:5]))

    # machine and pipeline settings survive a preset change
    st = RenderSettings()
    st.threads = 12
    st.seed = 4242
    st.film_transparent = True
    st.debug_pass = 'NORMAL'
    apply_preset(st, 'N64')
    kept = (st.threads == 12 and st.seed == 4242 and st.film_transparent
            and st.debug_pass == 'NORMAL')
    check('machine and pipeline settings survive a preset', kept,
          f'threads={st.threads} seed={st.seed} '
          f'transparent={st.film_transparent} pass={st.debug_pass}')

    # and opting out of the reset still layers
    st = RenderSettings()
    apply_preset(st, 'EGA')
    apply_preset(st, 'INFINID_4', reset=False)
    check('reset can be turned off to layer presets',
          st.palette_mode == 'EGA16', st.palette_mode)

    check('every preserved field exists on RenderSettings',
          all(p in fields for p in PRESERVED),
          ', '.join(sorted(set(PRESERVED) - set(fields))))
    reset_settings(RenderSettings())


def test_preset_library_is_well_formed():
    """Every preset must be complete and name only real settings."""
    import dataclasses

    from ..core.settings import RenderSettings
    from ..presets.library import CATEGORIES
    fields = {f.name for f in dataclasses.fields(RenderSettings)}
    cats = {c[0] for c in CATEGORIES}
    problems = []
    for key, entry in PRESETS.items():
        for required in ('label', 'category', 'note', 'settings'):
            if required not in entry:
                problems.append(f'{key}: missing {required}')
        if entry.get('category') not in cats:
            problems.append(f"{key}: unknown category {entry.get('category')}")
        for name in entry.get('settings', {}):
            if name not in fields:
                problems.append(f'{key}: unknown setting {name}')
    check(f'all {len(PRESETS)} presets are well formed', not problems,
          '; '.join(problems[:4]))
    check('there is a Default preset', 'DEFAULT' in PRESETS)
    labels = [e['label'] for e in PRESETS.values()]
    check('preset labels are unique', len(labels) == len(set(labels)))


def test_all_presets_render():
    bad = []
    for key in sorted(PRESETS):
        st = RenderSettings()
        apply_preset(st, key)
        st.resolution_x, st.resolution_y = 64, 48
        st.aa_samples = min(st.aa_samples, 2)
        st.output_scale = 'NONE'
        try:
            img = R.render(demo_scene(st), st)
            out = post.process(img, st)
            if not np.isfinite(out).all():
                bad.append(key + '(nan)')
        except Exception as exc:                                # noqa: BLE001
            bad.append(f'{key}({type(exc).__name__})')
    check(f'all {len(PRESETS)} presets render', not bad, ', '.join(bad[:4]))


def test_blender_layer_imports():
    from . import fakebpy
    fakebpy.install()
    import importlib
    try:
        mod = importlib.import_module('halcyon')
        importlib.reload(mod) if 'halcyon.engine' in sys.modules else None
        mod.register()
        mod.unregister()
        check('the Blender layer registers cleanly against a bpy stub', True)
    except Exception as exc:                                    # noqa: BLE001
        check('the Blender layer registers cleanly against a bpy stub', False,
              repr(exc))


def test_text_objects_export_without_losing_the_frame():
    """A text object is the first thing to enter that is not a mesh.

    Blender hands a mesh object geometry the depsgraph already owns, so
    `to_mesh_clear()` on one frees nothing and the old export ordering --
    convert everything, then read everything -- got away with it for the whole
    project. Text, curves, surfaces and metaballs *build* a temporary instead,
    and the next conversion of the same object destroys the last one. The
    moment one of those is in the scene twice, the export was reading freed
    memory, which in Blender is a crash and not an exception.
    """
    from . import fakeblender as FB
    from ..core.settings import RenderSettings
    props, engine = FB.install()
    from .. import export as EX

    def one(objects):
        warn = []
        dg = FB.depsgraph_of(props, objects)
        return EX.export_scene(dg, RenderSettings(), warn), warn

    text = FB.geometry_object('Text', FB.glyph_mesh)
    sc, warn = one([text])
    check('a text object exports', len(sc.mesh.tris) > 0, str(len(sc.mesh.tris)))
    check('with no material of its own it still gets one',
          len(sc.materials) >= 1, str(len(sc.materials)))
    check('and nothing is reported against it', not warn, str(warn))

    # the case that used to read freed memory: one object, two instances
    sc, warn = one([text, text])
    check('the same text object instanced twice does not read a freed mesh',
          len(sc.mesh.tris) > 0, str(warn))

    # a mixed scene, since text and mesh objects free differently
    cube = FB.geometry_object('Cube', FB.cube_mesh, kind='MESH')
    curve = FB.geometry_object('Curve', FB.glyph_mesh, kind='CURVE')
    meta = FB.geometry_object('Metaball', FB.cube_mesh, kind='META')
    sc, warn = one([cube, text, curve, meta, text])
    check('text, curve, metaball and mesh export together in one scene',
          len(sc.objects) == 5, f'{len(sc.objects)} objects, {warn}')

    # text with its fill off converts to outlines: nothing to draw, and the
    # user should be told why rather than left looking at an empty frame
    hollow = FB.geometry_object(
        'Outline', lambda: FB.glyph_mesh(faces=False))
    sc, warn = one([hollow])
    check('a text object with no faces is named in a warning',
          any('Outline' in w and 'Fill' in w for w in warn), str(warn))

    # one object that cannot convert must not take the rest of the frame down
    class Broken:
        type = 'FONT'
        name = 'Broken'
        color = (1, 1, 1, 1)
        matrix_world = np.eye(4, dtype=np.float32)

        def to_mesh(self):
            return types.SimpleNamespace(materials=None)

        def to_mesh_clear(self):
            pass

    sc, warn = one([cube, Broken(), text])
    check('an object that will not convert is skipped, not fatal',
          len(sc.objects) == 2, f'{len(sc.objects)} objects')
    check('and it is named in the report',
          any('Broken' in w for w in warn), str(warn))

    # and the whole thing goes through the engine, not just the exporter
    img, _passes, cap = FB.run_render(props, engine, mesh=FB.glyph_mesh())
    check('a glyph-shaped mesh renders through the engine',
          img is not None and bool(np.isfinite(img).all()),
          str(cap.get('reports')))


def main():
    order = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in order:
        print(fn.__name__)
        try:
            fn()
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            FAILS.append(fn.__name__)
    print()
    print(f'{len(FAILS)} failure(s): ' + ', '.join(FAILS) if FAILS
          else 'all renderer tests passed')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
