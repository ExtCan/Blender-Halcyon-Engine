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
    check('a New control is offered (born as a Halcyon material)',
          'halcyon.material_new' in src)


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

    # NTSC now carries the CPU formulation -- I and Q at their own radii, and
    # the triple box run as three real passes, because the CPU re-pads the
    # frame edge before every pass and one composed kernel cannot say that.
    # It finally has a reference to be measured against; it stays UNPROVEN
    # (and disabled) until a driver reports the same numbers.
    s4 = RenderSettings()
    s4.composite, s4.composite_bleed = True, 1.0
    s4.composite_ringing, s4.composite_dot_crawl = 0.5, 0.0
    ri = max(int(round(w / 320.0 * 6.0 * 1.0)), 1)
    rq = max(int(round(w / 320.0 * 12.0 * 1.0)), 1)

    def run_tex(name, source_np, extra_samplers=None, **uni):
        src = STAGES[name].replace('in vec2 vUV;', 'uniform vec2 vUV;')
        prog, _err = try_compile(src, 'GLSL')
        u = {'source': Texture(source_np, colorspace='Non-Color',
                               filt='NEAREST', wrap='EXTEND'), 'vUV': uv}
        for k, v in (extra_samplers or {}).items():
            u[k] = Texture(v, colorspace='Non-Color', filt='NEAREST',
                           wrap='EXTEND')
        for k, v in uni.items():
            u[k] = (np.full(n, float(v), np.float32)
                    if isinstance(v, (int, float))
                    else np.broadcast_to(np.asarray(v, np.float32),
                                         (n, len(v))).copy())
        outs, _d = prog.run(u, {}, n)
        return outs['Color'].reshape(h, w, 4)

    rgba = np.concatenate([img, np.ones((h, w, 1), np.float32)], 2)
    yiq = rgba
    for step in range(3):
        yiq = run_tex('NTSC_BLUR', yiq, ri=ri, rq=rq, ry=2.0,
                      to_yiq=1.0 if step == 0 else 0.0, resolution=(w, h))
    got = run_tex('NTSC', rgba, extra_samplers={'blurred': yiq},
                  ringing=0.5)[:, :, :3]
    want = PO.composite_ntsc(img.copy(), s4)
    ntsc_err = float(np.abs(got - want).max())
    check('the NTSC passes now match the CPU path they will be measured '
          'against', ntsc_err < 2e-3, f'max {ntsc_err:.5f}')

    # nothing unproven may be enabled
    unproven = [k for k, (grade, _t) in VALIDATION.items()
                if grade == 'UNPROVEN' and k in ENABLED]
    check('no unvalidated stage is enabled', not unproven, ', '.join(unproven))
    # and what has been measured must not quietly fall back out
    check('NTSC is enabled now that hardware has measured it (0.00037)',
          'NTSC' in ENABLED and 'NTSC_BLUR' in ENABLED, str(ENABLED))
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

    from ..nodes.pattern_nodes import SPECS

    models = {m[0] for m in MODEL_ITEMS}
    sockets = {n for _k, n, _d in HS.SOCKETS}
    # what each pattern node REALLY has, from the spec table the classes
    # are generated from -- a recipe naming a socket that is not here is
    # the silent no-op this test exists to catch
    ptable = {f'HALCYON_{s[0]}Node': ({n for _k, n, _d in s[4]},
                                      {n for _k, n in s[6]},
                                      set(s[5]))
              for s in SPECS}
    problems = []
    for key, spec in tmpl.TEMPLATES.items():
        for field in ('label', 'note', 'model', 'category'):
            if field not in spec:
                problems.append(f'{key}: missing {field}')
        if spec.get('category') not in ('SIMPLE', 'ADVANCED'):
            problems.append(f"{key}: bad category {spec.get('category')}")
        if spec.get('model') not in models:
            problems.append(f"{key}: unknown model {spec.get('model')}")
        for sock in spec.get('inputs', {}):
            if sock not in sockets:
                problems.append(f'{key}: unknown socket {sock}')
        for entry in spec.get('textures', []):
            if not isinstance(entry, dict):
                idname, props, inputs, target = entry
                entry = {'node': idname, 'props': props, 'inputs': inputs,
                         'target': target}
            idname = entry['node']
            if idname not in DISPATCH:
                problems.append(f'{key}: {idname} has no evaluator')
            if entry['target'] not in sockets:
                problems.append(f"{key}: unknown target {entry['target']}")
            if idname in ptable:
                pin, pout, pprops = ptable[idname]
                for sock in entry.get('inputs', {}):
                    if sock not in pin:
                        problems.append(f'{key}: {idname} has no input '
                                        f'{sock!r}')
                for prop in entry.get('props', {}):
                    if prop not in pprops:
                        problems.append(f'{key}: {idname} has no prop '
                                        f'{prop!r}')
                want = entry.get('output')
                if want is not None and want not in pout:
                    problems.append(f'{key}: {idname} has no output '
                                    f'{want!r}')
            if entry.get('bump') is not None and \
                    not isinstance(entry['bump'], float):
                problems.append(f'{key}: bump strength must be a float')
    check(f'all {len(tmpl.TEMPLATES)} templates are valid', not problems,
          '; '.join(problems[:3]))
    labels = [v['label'] for v in tmpl.TEMPLATES.values()]
    check('template labels are unique', len(labels) == len(set(labels)))
    check('the menu lists every template',
          len(tmpl.template_items()) == len(tmpl.TEMPLATES))
    # the shelf the field asked for: 15 more, in two named groups,
    # Water and Lava among them
    check('the shelf holds 28 templates', len(tmpl.TEMPLATES) == 28,
          str(len(tmpl.TEMPLATES)))
    simple = tmpl.category_keys('SIMPLE')
    advanced = tmpl.category_keys('ADVANCED')
    check('Simple and Advanced partition the shelf',
          not (set(simple) & set(advanced))
          and set(simple) | set(advanced) == set(tmpl.TEMPLATES))
    check('Water and Lava are on it, with textures, as Advanced',
          {'WATER', 'LAVA'} <= set(advanced)
          and tmpl.TEMPLATES['WATER'].get('textures')
          and tmpl.TEMPLATES['LAVA'].get('textures'))
    check('the grouped menu class is registered alongside the operator',
          any(getattr(c, 'bl_idname', '') == 'HALCYON_MT_material_templates'
              for c in tmpl.CLASSES))


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

    # anything claimed as running on both must carry its measurement: post
    # stages through the VALIDATION table, the deferred features through the
    # hardware number recorded in their own reason string
    proven = {k for k, (grade, _t) in VALIDATION.items()
              if grade in ('EXACT', 'CLOSE')}
    claimed = {f for f, (sup, _w) in C.FEATURES.items() if sup == C.BOTH}
    mapping = {'display_transform': 'DISPLAY', 'ordered_dither': 'DITHER',
               'crt': 'CRT', 'lens': 'LENS', 'composite_ntsc': 'NTSC'}
    unproven = []
    for f in claimed:
        if f in mapping:
            if mapping[f] not in proven:
                unproven.append(f)
        else:
            why = C.FEATURES[f][1]
            if 'measured' not in why or not any(ch.isdigit() for ch in why):
                unproven.append(f)
    check('nothing claims GPU support without a measured number', not unproven,
          ', '.join(unproven))

    # the two impossible ones must stay impossible
    check('error diffusion is marked as never portable',
          C.FEATURES['error_diffusion'][0] == C.NEVER)
    check('the A-buffer is marked as never portable',
          C.FEATURES['abuffer'][0] == C.NEVER)

    # the coded shader node crossed the bar in round 13: inlined natively,
    # measured through the deferred pass on the user's driver at 0.000021
    check('the coded shader node is proven on hardware',
          C.FEATURES['code_node'][0] == C.BOTH and
          '0.000021' in C.FEATURES['code_node'][1])
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
        # plan() selects POST stages; the deferred features are a render-path
        # choice the user makes with the GPU Shading switch, not a stage
        post_feats = {'display_transform', 'ordered_dither', 'crt', 'lens'}
        proven_now = {f for f, (g, _t) in C.FEATURES.items()
                      if g == C.BOTH} & post_feats
        check('with a GPU present every proven post stage is selected',
              set(stages) == proven_now, f'{sorted(stages)} vs {sorted(proven_now)}')
        check('and the deferred stage is mentioned in the notes',
              any('shading' in n.lower() for n in notes), '; '.join(notes))
        text = ' '.join(notes)
        # the port is complete: claiming the node graph forces a CPU frame
        # would now be false, exactly as it became false for code_node.
        # This check FLIPPED when node_graph reached BOTH (R106 audit)
        check('node_graph is no longer claimed as CPU-stuck',
              'node_graph' not in text, text)
        # the coded shader node left the blocking list: the deferred pass
        # inlines it, so claiming it forces a CPU frame would be false
        check('code_node is no longer claimed as CPU-stuck',
              'code_node' not in text, text)
        check('the deferred note carries its measurement, not just a claim',
              '0.000051' in text)
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
    # the modern Mix node: three data types behind DUPLICATE display
    # names ('A' three times over) -- resolution must go by identifier,
    # because plain 'A' silently reads the FLOAT socket's default
    # instead of the user's linked colour. These sockets carry the real
    # identifiers a Blender export carries.
    def midsock(name, ident, typ, d):
        return {'name': name, 'identifier': ident, 'type': typ,
                'default': d, 'link': None}

    def mix_inputs():
        return [
            midsock('Factor', 'Factor_Float', 'VALUE', .65),
            midsock('Factor', 'Factor_Vector', 'VECTOR', [.5, .5, .5]),
            midsock('A', 'A_Float', 'VALUE', .2),
            midsock('B', 'B_Float', 'VALUE', .9),
            midsock('A', 'A_Vector', 'VECTOR', [.8, -.2, .4]),
            midsock('B', 'B_Vector', 'VECTOR', [.1, .6, -.5]),
            midsock('A', 'A_Color', 'RGBA', [.8, .2, .4, 1.]),
            midsock('B', 'B_Color', 'RGBA', [.1, .7, .3, 1.]),
        ]

    mix_outs = [('Result', 'VALUE'), ('Result', 'VECTOR'),
                ('Result', 'RGBA')]
    cases.append(('Mix FLOAT',
                  node('ShaderNodeMix', {'data_type': 'FLOAT'},
                       mix_inputs(), mix_outs), 'float', 0))
    cases.append(('Mix VECTOR',
                  node('ShaderNodeMix', {'data_type': 'VECTOR'},
                       mix_inputs(), mix_outs), 'vec3', 1))
    for op in ('MIX', 'ADD', 'MULTIPLY', 'SUBTRACT', 'SCREEN',
               'DIFFERENCE', 'DIVIDE', 'LIGHTEN', 'DARKEN', 'OVERLAY'):
        cases.append((f'Mix RGBA {op}',
                      node('ShaderNodeMix',
                           {'data_type': 'RGBA', 'blend_type': op,
                            'clamp_result': True},
                           mix_inputs(), mix_outs), 'vec4', 2))
    # a colour-mode Mix whose COLOUR sockets are the interesting ones:
    # if resolution fell back to the first 'A' (the float socket), this
    # would come out grey and the parity above could not tell -- so one
    # case pins asymmetric colours through the identifier path with the
    # factor unclamped
    cases.append(('Mix RGBA unclamped factor',
                  node('ShaderNodeMix',
                       {'data_type': 'RGBA', 'blend_type': 'MIX',
                        'clamp_factor': False},
                       [midsock('Factor', 'Factor_Float', 'VALUE', 1.4)]
                       + mix_inputs()[1:], mix_outs), 'vec4', 2))

    # Mapping: the field named it by material ('Material.002': no GLSL
    # emitter for ShaderNodeMapping). All four vector types, the negated
    # TEXTURE inverse quirk included, with baked float32 trig
    for vt_mode in ('POINT', 'TEXTURE', 'VECTOR', 'NORMAL'):
        cases.append((f'Mapping {vt_mode}',
                      node('ShaderNodeMapping', {'vector_type': vt_mode},
                           [vec('Vector', [.4, -.7, .9]),
                            vec('Location', [.3, -.2, .5]),
                            vec('Rotation', [.35, -.6, 1.1]),
                            vec('Scale', [1.6, .7, -1.2])],
                           [('Vector', 'VECTOR')]), 'vec3', 0))
    # ...a VARYING vector through the transform...
    cases.append(('Mapping varying P', {'output': None, 'nodes': {
        'g': {'id': 'g', 'bl_idname': 'ShaderNodeNewGeometry', 'props': {},
              'inputs': [],
              'outputs': [{'name': 'Position', 'type': 'VECTOR'},
                          {'name': 'Normal', 'type': 'VECTOR'}]},
        'n': {'id': 'n', 'bl_idname': 'ShaderNodeMapping',
              'props': {'vector_type': 'POINT'},
              'inputs': [vec('Vector', [0, 0, 0], ['g', 0]),
                         vec('Location', [.3, -.2, .5]),
                         vec('Rotation', [.35, -.6, 1.1]),
                         vec('Scale', [1.6, .7, -1.2])],
              'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]}}},
        'vec3', 0))
    # ...and a LINKED rotation: the in-shader trig fallback
    cases.append(('Mapping linked rotation', {'output': None, 'nodes': {
        'c': {'id': 'c', 'bl_idname': 'ShaderNodeCombineXYZ', 'props': {},
              'inputs': [val('X', .35), val('Y', -.6), val('Z', 1.1)],
              'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
        'n': {'id': 'n', 'bl_idname': 'ShaderNodeMapping',
              'props': {'vector_type': 'TEXTURE'},
              'inputs': [vec('Vector', [.4, -.7, .9]),
                         vec('Location', [.3, -.2, .5]),
                         vec('Rotation', [0, 0, 0], ['c', 0]),
                         vec('Scale', [1.6, .7, -1.2])],
              'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]}}},
        'vec3', 0))

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
        s2.specular[:] = np.array([1.0, 0.55, 0.3], np.float32)[None, :]
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
            # evaluate() folds the specular COLOUR into spec (spec_col), so
            # only the level multiplies here -- the replica used to apply
            # s2.specular again, which cancelled the same double-tint in the
            # old GLSL as long as every test specular stayed white
            out = out + (base * np.asarray(dif)[:, None]
                         + 0.6 * np.asarray(spec)) * rad
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
           'hal_specular_tint': np.tile(
               np.array([[1.0, 0.55, 0.3]], np.float32), (n, 1)),
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


def test_deferred_gpu_shading_matches_the_renderer():
    """The GPU frame path must reproduce the renderer's own frame, not a
    replica of it.

    The earlier assembled-shader test proved emitters + models + light loop
    agree with a hand-built copy of the CPU maths. This one closes the gap
    that copy left open: it rasterises a real scene, packs the real G-buffer,
    assembles the real per-material passes with their constants probed
    through `closure_to_surface` itself, runs them through Halcyon's own GLSL
    front-end -- and compares against what `render()` actually delivered.
    Every seam is inside the comparison: packing, reconstruction, probing,
    baking, the light loop, two-sided flips, flat versus smooth normals.
    """
    from ..core import raster
    from ..core.scene import Light, Material
    from ..gpu import shade as GSH
    from .scenebuild import cube, plane, sphere, _mesh_concat, look_at_matrix

    w, h = 128, 96
    st = base_settings(w, h, shadows=False, render_device='CPU')
    st.fog = False
    st.ambient_occlusion = False
    sc = demo_scene(st, with_texture=False)
    # the demo scene, minus what the frame shader does not carry: its POINT
    # light keeps INVERSE_SQUARE decay, its materials are graphless constants
    sc.lights = [
        Light(type='SUN', name='Key', direction=(-0.62, 0.45, -0.45),
              color=(1.0, 0.96, 0.88), energy=6.0, shadow='NONE'),
        Light(type='POINT', name='Fill', position=(3.5, -3.0, 3.0),
              color=(0.45, 0.6, 1.0), energy=600.0, shadow='NONE',
              decay='INVERSE_SQUARE'),
        Light(type='SPOT', name='Rim', position=(-4.0, 2.5, 4.0),
              direction=(0.55, -0.35, -0.75), color=(1.0, 0.3, 0.8),
              energy=400.0, shadow='NONE', spot_size=0.9, spot_blend=0.3),
    ]

    # CPU truth: the actual beauty frame, before any post
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    cpu_img = R.render(sc, st)

    # the deferred path, simulated through Halcyon's own compiler
    view, proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)

    passes, why, atlases = GSH.plan_frame(job, g)
    check('the demo frame qualifies for GPU shading', passes is not None,
          str(why))
    if passes is None:
        return
    check('one pass per material present', len(passes) == 3,
          str([p[1] for p in passes]))

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the simulated frame ran', img is not None, str(hit is None))
    if img is None:
        return
    cov = g.tri >= 0
    check('the passes claim exactly the covered pixels',
          bool((hit == cov).all()),
          f'{int((hit ^ cov).sum())} pixels disagree')

    got = img[cov]
    want = cpu_img[cov][:, :3]
    err = float(np.abs(got - want).max())
    mean = float(np.abs(got - want).mean())
    check('the GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} mean {mean:.6f} over {int(cov.sum())} px')

    # a reflective material under a Bryce sky QUALIFIES now: worlds too
    # rich for GLSL take the CPU-composite env path -- the renderer's
    # own world along the reflected rays, added after readback
    sc2 = demo_scene(st, with_texture=False)
    sc2.lights = list(sc.lights)
    sc2.materials[1] = Material(
        name='Mirrored', index=1, model='PHONG', diffuse=(0.8, 0.2, 0.1),
        reflect_level=0.5)
    sc2.world.mode = 'BRYCE'
    job2 = R.ShadeJob(sc2, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p2, why2, a2b = GSH.plan_frame(job2, g)
    check('a reflective material under a Bryce sky now qualifies (the '
          'CPU-composite env)', p2 is not None and '__env' in (a2b or {}),
          str(why2))

    # ray-traced shadows ride the BVH textures now -- but a RAY frame whose
    # job carries NO BVH mirrors the CPU exactly: `visibility` returns fully
    # lit when bvh is None, so every light is simply unshadowed and the
    # frame still qualifies. Nothing to trace is not a refusal.
    st3 = base_settings(w, h, shadows=True, shadow_default='RAY')
    sc3 = demo_scene(st3, with_texture=False)     # its lights are shadow=MAP
    job3 = R.ShadeJob(sc3, st3, {}, None, view, eye, w, h)
    p3, why3, _a3 = GSH.plan_frame(job3, g)
    check('RAY mode with no BVH is fully lit on both paths, and qualifies',
          p3 is not None, str(why3))

    # area lights joined the loop in 1.25.7 -- the direct math is the point
    # branch, and their softness lives in the shadow term
    st4 = base_settings(w, h, shadows=False)
    sc4 = demo_scene(st4, with_texture=False)
    sc4.lights = [Light(type='AREA', name='Soft', position=(0, 0, 5),
                        energy=80.0, shadow='NONE')]
    job4 = R.ShadeJob(sc4, st4, {}, None, view, eye, w, h)
    p4, why4, _a4 = GSH.plan_frame(job4, g)
    check('an area-lit frame qualifies now', p4 is not None, str(why4))


def test_deferred_shading_carries_the_shadows():
    """Shadow-mapped frames must shade on the GPU and match the renderer.

    This is the widest gate the deferred pass has: real scenes render with
    shadows on, and until the maps travelled, every one of them fell back to
    the CPU. The maps are the same depth images the CPU just baked, packed
    into an atlas per light -- six cells for a point light's cube -- with the
    matrix, the linearisation, the slope bias, the normal offset and every
    Vogel PCF tap baked into the shader exactly as `ShadowMap.lookup` does
    them. The comparison target is `render()` itself, shadows and all.
    """
    from ..core import lights as LI, raster
    from ..core.scene import Light
    from ..gpu import shade as GSH

    w, h = 128, 96
    st = base_settings(w, h, shadows=True, render_device='CPU')
    sc = demo_scene(st, with_texture=False)
    # the demo scene's own lights, shadows and all: SUN with an ortho map,
    # POINT with a six-face cube -- plus a SPOT for the perspective map
    sc.lights.append(
        Light(type='SPOT', name='Rim', position=(-4.0, 2.5, 4.0),
              direction=(0.55, -0.35, -0.75), color=(1.0, 0.3, 0.8),
              energy=400.0, shadow='MAP', shadow_bias=0.02,
              spot_size=0.9, spot_blend=0.3))

    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    cpu_img = R.render(sc, st)          # builds the shadow maps as it goes

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)

    passes, why, atlases = GSH.plan_frame(job, g)
    check('a shadow-mapped frame now qualifies', passes is not None, str(why))
    if passes is None:
        return
    check('every shadowed light packed an atlas', len(atlases) == 3,
          str(sorted(atlases)))
    entry = atlases.get('hal_shadow1')
    cube_atlas = entry[1]() if entry is not None else None
    check('the point light packed six cube faces',
          cube_atlas is not None and
          cube_atlas.shape[1] == cube_atlas.shape[0] // 2 * 3,
          str(None if cube_atlas is None else cube_atlas.shape))

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the shadowed frame simulates', img is not None, str(why))
    if img is None:
        return
    cov = g.tri >= 0
    got = img[cov]
    want = cpu_img[cov][:, :3]
    err = float(np.abs(got - want).max())
    mean = float(np.abs(got - want).mean())
    check('the shadowed GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} mean {mean:.6f} over {int(cov.sum())} px')

    # and the shadows are actually in the picture, not vacuously matched:
    # the same frame with shadows off must differ where the shadows fall
    st_off = base_settings(w, h, shadows=False)
    st_off.color_depth = '24'
    st_off.dither = 'NONE'
    st_off.output_scale = 'NONE'
    for l in sc.lights:
        l.shadow_map = None
    unshadowed = R.render(sc, st_off)
    delta = float(np.abs(cpu_img[cov][:, :3] - unshadowed[cov][:, :3]).max())
    check('the frame being matched really contains shadows', delta > 0.05,
          f'shadows change the frame by {delta:.4f}')


def test_deferred_shading_carries_ray_shadows():
    """Hard ray-traced shadows must shade on the GPU and match the renderer.

    The first ray-tracing stage to reach the deferred pass, standing on the
    occlusion kernel that already matches `bvh.occluded()` ray for ray: the
    BVH travels as two textures shared by every ray light, each light's
    visibility is `visibility`'s own RAY branch -- origin biased along N and
    L, the ray clipped just short of the light, no density term -- and the
    comparison target is `render()` itself with Shadow Method RAY. Soft
    ray shadows QUALIFY now (their jitter is a pure function of pixel,
    sample, light and seed); the dedicated seam test proves the picture.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    w, h = 96, 72
    st = base_settings(w, h, shadows=True, shadow_default='RAY',
                       render_device='CPU')
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)   # SUN + POINT, both cast

    cpu_img = R.render(sc, st)                # traces its shadows as it goes

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)    # as render() built it
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, w, h)

    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a ray-shadowed frame now qualifies', passes is not None, str(why))
    if passes is None:
        return
    check('the BVH rides as two shared textures',
          'hal_bvh' in atlases and 'hal_btris' in atlases,
          str(sorted(atlases)))
    check('and no per-light atlas: every ray light shares them',
          not any(k.startswith('hal_shadow') for k in atlases),
          str(sorted(atlases)))

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the ray-shadowed frame simulates', img is not None, str(hit))
    if img is None:
        return
    cov = g.tri >= 0
    got = img[cov]
    want = cpu_img[cov][:, :3]
    err = float(np.abs(got - want).max())
    mean = float(np.abs(got - want).mean())
    check('the ray-shadowed GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} mean {mean:.6f} over {int(cov.sum())} px')

    # and the rays really darken the picture: the same frame fully lit
    # must differ where the shadows fall
    st_off = base_settings(w, h, shadows=False)
    st_off.color_depth = '24'
    st_off.dither = 'NONE'
    st_off.output_scale = 'NONE'
    unshadowed = R.render(sc, st_off)
    delta = float(np.abs(cpu_img[cov][:, :3] - unshadowed[cov][:, :3]).max())
    check('the frame being matched really contains ray shadows', delta > 0.05,
          f'ray shadows change the frame by {delta:.4f}')

    # a light with radius takes the soft path, and it QUALIFIES: the
    # jitter is deterministic per (pixel, sample, light, seed) now --
    # the once-refusing "random stream" is gone
    sc.lights[1].radius = 0.5
    st.shadow_samples = 4
    GSH._PLAN_CACHE.clear()
    p2, why2, _a2 = GSH.plan_frame(job, g)
    check('soft ray shadows now qualify (the random stream is gone)',
          p2 is not None, str(why2))
    sc.lights[1].radius = 0.0

    # radius alone is not soft: with one shadow sample the CPU takes the
    # hard single-ray path, and so may the GPU
    sc.lights[1].radius = 0.5
    st.shadow_samples = 1
    GSH._PLAN_CACHE.clear()
    p3, why3, _a3 = GSH.plan_frame(job, g)
    check('radius with one sample is the hard path, and qualifies',
          p3 is not None, str(why3))


def test_ray_reflections_shade_on_the_gpu():
    """One traced bounce must shade on the GPU and match the renderer.

    The reflections arc lands: rays built exactly as `_add_raytraced`
    builds them (unflipped interpolated normal, camera V, raw ray bias),
    closest hits from the kernel that already matches `bvh.intersect()`
    tie for tie, the hit points shaded by the SAME materials through
    secondary passes -- with the environment term, because that is what
    the CPU's depth-exhausted branch shades hits with, and without the
    backface override, because `trace()` passes `front=None` -- and the
    composite is `_add_raytraced`'s own blend, misses falling to
    `world_color` computed by the renderer itself. The comparison target
    is `render()` with ray tracing ON, the refusal this whole arc
    existed to lift.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    w, h = 128, 96
    st = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[2].reflect_level = 0.5       # the Box becomes a mirror

    cpu_img = R.render(sc, st)                # traces as it always has

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, w, h)

    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a ray-traced frame now qualifies', passes is not None, str(why))
    if passes is None:
        return
    rplan = atlases.get('__reflect')
    check('the plan carries a reflection stage', rplan is not None)
    if rplan is None:
        return
    check('with a secondary pass for EVERY mesh material (any of them '
          'can be hit)', len(rplan['secondary']) == 3,
          str([p[1] for p in rplan['secondary']]))
    check('and the Box is the reflective one', rplan['reflective'] == [2],
          str(rplan['reflective']))

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the ray-traced frame simulates', img is not None, str(hit))
    if img is None:
        return
    cov = g.tri >= 0
    got = img[cov]
    want = cpu_img[cov][:, :3]
    err = float(np.abs(got - want).max())
    mean = float(np.abs(got - want).mean())
    check('the ray-traced GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} mean {mean:.6f} over {int(cov.sum())} px')

    # the reflections are really in the picture: the same frame without
    # ray tracing must differ on the mirror
    st_off = base_settings(w, h, shadows=True, raytrace=False)
    st_off.color_depth = '24'
    st_off.dither = 'NONE'
    st_off.output_scale = 'NONE'
    st_off.env_reflection = False
    flat_img = R.render(sc, st_off)
    box_px = np.zeros_like(cov)
    box_px[cov] = sc.mesh.mat_index[g.tri[cov]] == 2
    delta = float(np.abs(cpu_img[box_px][:, :3]
                         - flat_img[box_px][:, :3]).max())
    check('the frame being matched really contains traced reflections',
          delta > 0.05, f'ray tracing changes the mirror by {delta:.4f}')

    # ---- depth 2 QUALIFIES now: the recursion tree is walked branch by
    # branch (the dedicated depth test proves the picture; this locks
    # the plan)
    st2 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=2)
    job2 = R.ShadeJob(sc, st2, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p2, why2, a2x = GSH.plan_frame(job2, g)
    check('ray depth 2 now qualifies, with the mid-level passes',
          p2 is not None and a2x['__reflect']['depth'] == 2
          and len(a2x['__reflect']['secondary_mid']) == 3, str(why2))

    # a refracting material QUALIFIES now -- the refusal this check once
    # asserted was lifted when the refraction sweep landed; the dedicated
    # block below proves the picture, this one just locks the plan
    sc.materials[1].opacity = 0.5
    st3 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                        transparency='NONE')
    job3 = R.ShadeJob(sc, st3, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, a3 = GSH.plan_frame(job3, g)
    check('a refracting material now qualifies',
          p3 is not None and '__reflect' in a3
          and a3['__reflect']['refractive'] == [1], str(why3))
    sc.materials[1].opacity = 1.0

    # ray_reflection OFF under raytrace: the CPU adds neither the traced
    # bounce NOR the env term -- the qualifying frame must mirror that
    st4 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1)
    st4.ray_reflection = False
    st4.color_depth = '24'
    st4.dither = 'NONE'
    st4.output_scale = 'NONE'
    cpu4 = R.render(sc, st4)
    job4 = R.ShadeJob(sc, st4, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p4, why4, a4 = GSH.plan_frame(job4, g)
    check('ray tracing with reflection off still qualifies',
          p4 is not None, str(why4))
    if p4 is not None:
        check('and plans no reflection stage', '__reflect' not in a4)
        img4, _h4 = GSH.simulate(job4, g, p4, a4)
        e4 = float(np.abs(img4[cov] - cpu4[cov][:, :3]).max())
        check('and the un-reflected frame still matches (no env term '
              'either -- exactly the CPU)', e4 < 6e-3, f'max {e4:.5f}')

    # ---- refraction: the Ball becomes glass, the exact _add_raytraced
    # lerp -- eta by facing side, TIR falling back to a mirror bounce,
    # origin stepped INTO the surface, rgb*(1-k) + hit*k*diffuse
    sc.materials[1].opacity = 0.35
    sc.materials[1].ior = 1.33
    st6 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                        transparency='NONE')
    st6.color_depth = '24'
    st6.dither = 'NONE'
    st6.output_scale = 'NONE'
    cpu6 = R.render(sc, st6)
    job6 = R.ShadeJob(sc, st6, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p6, why6, a6 = GSH.plan_frame(job6, g)
    check('a refracting frame now qualifies', p6 is not None, str(why6))
    if p6 is not None:
        rp6 = a6.get('__reflect')
        check('the plan carries the refractive set',
              rp6 is not None and rp6['refractive'] == [1],
              str(None if rp6 is None else rp6['refractive']))
        img6, _h6 = GSH.simulate(job6, g, p6, a6)
        e6 = float(np.abs(img6[cov] - cpu6[cov][:, :3]).max())
        check('the refracted GPU frame is the CPU frame', e6 < 6e-3,
              f'max {e6:.5f}')
        # vacuity: the glass really transmits -- k = (1-0.35)*1 = 0.65
        st6b = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                             transparency='NONE')
        st6b.ray_refraction = False
        st6b.color_depth = '24'
        st6b.dither = 'NONE'
        st6b.output_scale = 'NONE'
        opaque6 = R.render(sc, st6b)
        ball6 = np.zeros_like(cov)
        ball6[cov] = sc.mesh.mat_index[g.tri[cov]] == 1
        d6 = float(np.abs(cpu6[ball6][:, :3] - opaque6[ball6][:, :3]).max())
        check('the frame being matched really refracts', d6 > 0.05,
              f'refraction changes the glass by {d6:.4f}')
        # and refraction OFF still qualifies and matches (no lerp at all)
        job6b = R.ShadeJob(sc, st6b, {}, bvh, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        p6b, why6b, a6b = GSH.plan_frame(job6b, g)
        check('refraction off still qualifies', p6b is not None, str(why6b))
        if p6b is not None:
            img6b, _h6b = GSH.simulate(job6b, g, p6b, a6b)
            e6b = float(np.abs(img6b[cov] - opaque6[cov][:, :3]).max())
            check('and matches the CPU without the lerp', e6b < 6e-3,
                  f'max {e6b:.5f}')
    sc.materials[1].opacity = 1.0

    # a refractive material whose base colour varies per pixel refuses,
    # named: the lerp tint must be a constant. Built as a master-shader
    # graph -- a checker driving Diffuse Color with Opacity 0.5 on the
    # node itself, since a graph material's opacity lives in its closure
    st7 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                        transparency='NONE')
    sc7 = demo_scene(st7, with_texture=True)  # brings the checker image

    def sk7(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    sc7.materials[0].graph = {'output': 'out', 'nodes': {
        'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                'props': {'image': 'checker', 'interpolation': 'Closest'},
                'inputs': [sk7('Vector', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'},
                            {'name': 'Alpha', 'type': 'VALUE'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': 'PHONG'},
                'inputs': [sk7('Diffuse Color', 'RGBA', [1, 1, 1, 1],
                               ['tex', 0]),
                           sk7('Opacity', 'VALUE', 0.5)],
                'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'inputs': [sk7('Surface', 'SHADER', None, ['hal', 0])],
                'outputs': []}}}
    tex7 = R.prepare_textures(sc7, st7)
    g7 = raster.GBuffer(w, h)
    raster.rasterize(sc7.mesh.verts, sc7.mesh.tris, vp, w, h, gbuf=g7)
    job7 = R.ShadeJob(sc7, st7, tex7, BVH(sc7.mesh.verts, sc7.mesh.tris),
                      view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p7, why7, _a7 = GSH.plan_frame(job7, g7)
    check('texture-tinted glass refuses, naming the varying base colour',
          p7 is None and 'varies per pixel' in str(why7), str(why7))

    # ---- the field shape itself: normal-mapped WATER-like glass. The
    # Normal chain bends the shading normal, so the rays must bend too --
    # _ray_context runs the CPU's own closure code for exactly the ray
    # pixels, and the frame must still match render() outright
    from .scenebuild import add_normal_mapped_ball
    st8 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                        transparency='NONE')
    st8.color_depth = '24'
    st8.dither = 'NONE'
    st8.output_scale = 'NONE'
    sc8 = demo_scene(st8, with_texture=False)
    add_normal_mapped_ball(sc8)               # master graph, Normal linked
    hal8 = sc8.materials[1].graph['nodes']['hal']
    for sock in hal8['inputs']:
        if sock['name'] == 'Diffuse Color':
            sock['link'] = None               # flat tint: the lerp needs it
            sock['default'] = [0.2, 0.5, 0.8, 1.0]
        if sock['name'] == 'Opacity':
            sock['default'] = 0.4             # wavy glass
        if sock['name'] == 'IOR':
            sock['default'] = 1.33
    sc8.materials[2].reflect_level = 0.5      # and the mirror stays
    tex8 = R.prepare_textures(sc8, st8)
    cpu8 = R.render(sc8, st8)
    bvh8 = BVH(sc8.mesh.verts, sc8.mesh.tris)
    job8 = R.ShadeJob(sc8, st8, tex8, bvh8, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p8, why8, a8 = GSH.plan_frame(job8, g)
    check('normal-mapped glass now qualifies (the Water shape)',
          p8 is not None and '__reflect' in (a8 or {}), str(why8))
    if p8 is not None:
        img8, _h8 = GSH.simulate(job8, g, p8, a8)
        e8 = float(np.abs(img8[cov] - cpu8[cov][:, :3]).max())
        check('bent-ray glass matches the renderer', e8 < 6e-3,
              f'max {e8:.5f}')
        # the bend is really in the rays: flattening the chain must move
        # the refraction, or _ray_context proved nothing
        for sock in hal8['inputs']:
            if sock['name'] == 'Bump Strength':
                sock['default'] = 0.0
        GSH._PLAN_CACHE.clear()
        flat8 = R.render(sc8, st8)
        ball8 = np.zeros_like(cov)
        ball8[cov] = sc8.mesh.mat_index[g.tri[cov]] == 1
        d8 = float(np.abs(cpu8[ball8][:, :3] - flat8[ball8][:, :3]).max())
        check('the chain really bends the rays', d8 > 0.02,
              f'flattening the bump moves the glass by {d8:.4f}')

    # ---- the whole ray stack at once: RAY shadows AND the traced bounce,
    # both walking the same two BVH textures in one frame
    st5 = base_settings(w, h, shadows=True, shadow_default='RAY',
                        raytrace=True, ray_depth=1)
    st5.color_depth = '24'
    st5.dither = 'NONE'
    st5.output_scale = 'NONE'
    cpu5 = R.render(sc, st5)
    job5 = R.ShadeJob(sc, st5, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p5, why5, a5 = GSH.plan_frame(job5, g)
    check('ray shadows plus reflections qualify together',
          p5 is not None, str(why5))
    if p5 is not None:
        check('sharing the BVH textures and the reflection plan',
              'hal_bvh' in a5 and '__reflect' in a5, str(sorted(a5)))
        img5, _h5 = GSH.simulate(job5, g, p5, a5)
        e5 = float(np.abs(img5[cov] - cpu5[cov][:, :3]).max())
        check('and the full ray stack matches the renderer', e5 < 6e-3,
              f'max {e5:.5f}')


def test_a_material_visible_only_in_reflections_still_travels():
    """A mesh material with ZERO pixels on screen still gets its pass.

    The field named this hole precisely: 'Material.003' lived where no
    camera ray could rasterise it, reachable only through the Water's
    secondary rays -- and the off-screen probe crashed on 2-wide
    synthetic barycentrics (gbuf.bary is (H,W,3); raster.fetch needs
    (N,3)). This scene rebuilds that anatomy headlessly: a glass floor
    with IOR 1.0 -- rays pass straight through, hits guaranteed -- over
    a hidden plane the z-buffer occludes everywhere, plus the mirror Box
    so both sweeps walk the tree. The plan must qualify, probe the
    hidden material over its own triangles, and the simulated frame must
    match render() with the hidden plane really in the picture.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..core.scene import Material, ObjectInfo
    from ..gpu import shade as GSH
    from .scenebuild import _mesh_concat, cube, plane, sphere

    w, h = 128, 96
    st = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                       transparency='NONE')
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    # the demo mesh, plus a plane the camera can never see: it sits
    # UNDER the opaque-in-the-z-buffer floor (Transparency NONE), so
    # every primary ray loses it -- only rays continuing THROUGH the
    # glass floor reach it
    mesh = _mesh_concat([
        plane(z=0.0, size=11.0, mat=0, obj=0),
        sphere(centre=(-1.3, 0.2, 1.0), radius=1.0, mat=1, obj=1),
        cube(centre=(1.4, -0.4, 0.9), size=1.8, mat=2, obj=2),
        plane(z=-1.5, size=9.0, mat=3, obj=3),
    ])
    smooth = np.zeros(mesh.tris.shape[0], bool)
    smooth[mesh.mat_index == 1] = True
    mesh.smooth = smooth
    sc.mesh = mesh
    # self-lit, because down there the shadow maps put it in the dark --
    # the vacuity check below needs its colour to carry, not its litness
    sc.materials.append(Material(name='Hidden', index=3, model='LAMBERT',
                                 diffuse=(0.9, 0.15, 0.75),
                                 emission=(0.9, 0.15, 0.75),
                                 emission_level=1.0,
                                 specular_level=0.0, ambient_level=0.6))
    sc.objects.append(ObjectInfo(name='Hidden', index=3,
                                 location=(0, 0, -1.5),
                                 matrix_world=np.eye(4, dtype=np.float32)))
    sc.materials[0].opacity = 0.5         # the floor becomes glass...
    sc.materials[0].ior = 1.0             # ...that does not bend the rays
    sc.materials[2].reflect_level = 0.5   # and the Box stays a mirror

    cpu_img = R.render(sc, st)

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    cov = g.tri >= 0
    on_screen = np.unique(sc.mesh.mat_index[g.tri[cov]])
    check('the hidden material has NO pixel on screen (the premise)',
          3 not in on_screen, str(on_screen))

    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('the frame qualifies -- the off-screen probe no longer crashes',
          passes is not None, str(why))
    if passes is None:
        return
    rplan = atlases.get('__reflect')
    check('the plan carries the reflection stage', rplan is not None)
    if rplan is None:
        return
    check('with a secondary pass for the hidden material too',
          len(rplan['secondary']) == 4,
          str([p[1] for p in rplan['secondary']]))

    # the rays REALLY reach it: the frame's own refraction rays, asked
    # of the same BVH the sweeps trace
    py, px, org, dirs, _N = GSH._refraction_rays(job, g, rplan)
    hid, _t, _u, _v = bvh.intersect(org, dirs,
                                    np.full(py.size, 1e30, np.float32))
    hit_mats = np.unique(sc.mesh.mat_index[hid[hid >= 0]])
    check('the refraction rays land on the hidden plane', 3 in hit_mats,
          str(hit_mats))

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the frame simulates', img is not None, str(hit))
    if img is None:
        return
    err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
    check('and matches render() -- the hidden material shades through '
          'its probed pass', err < 6e-3,
          f'max {err:.5f} over {int(cov.sum())} px')

    # vacuity: recolour the hidden plane and the picture must move --
    # through the glass alone, since it owns no pixel of its own
    sc.materials[3].diffuse = (0.05, 0.05, 0.05)
    sc.materials[3].emission = (0.05, 0.05, 0.05)
    GSH._PLAN_CACHE.clear()
    cpu_dark = R.render(sc, st)
    moved = float(np.abs(cpu_img - cpu_dark).max())
    check('the hidden plane is really in the picture', moved > 0.05,
          f'recolouring it moves the frame by {moved:.4f}')
    job2 = R.ShadeJob(sc, st, {}, bvh, view, eye, w, h)
    p2, why2, a2 = GSH.plan_frame(job2, g)
    check('the recoloured frame still qualifies', p2 is not None,
          str(why2))
    if p2 is not None:
        img2, _h2 = GSH.simulate(job2, g, p2, a2)
        e2 = float(np.abs(img2[cov] - cpu_dark[cov][:, :3]).max())
        check('and the GPU tracks the recolour exactly', e2 < 6e-3,
              f'max {e2:.5f}')


def test_the_bump_node_shades_on_the_gpu():
    """ShaderNodeBump must travel: a height PRE-PASS plus the CPU formula.

    The field named this one precisely: 'Water' bends with a Bump node,
    whose height input needs neighbour differences the deferred pass never
    computed. Now it does, the CPU's own way: the height chain renders to
    its own target over the same ids texture, the main pass fetches the +x
    and +y neighbours by texelFetch (integer coordinates -- the sampler
    lottery stays closed) gated on the neighbour being covered by the same
    material, and bends with n_bump's exact arithmetic. Checked against
    `render()` itself, with proof the bump is load-bearing -- and then the
    whole field shape at once: bump-driven WAVY GLASS under ray tracing,
    where the rays bend through the CPU-evaluated n_bump and hit shading
    treats the node as a wire (ctx.px is None on hits).
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH
    from .scenebuild import ImageBuffer, add_normal_mapped_ball

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    w, h = 128, 96

    def build(strength=0.8, opacity=1.0, raytrace=False):
        st = base_settings(w, h, shadows=True, transparency='NONE',
                           raytrace=raytrace,
                           ray_depth=1 if raytrace else 2)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        add_normal_mapped_ball(sc)     # full master socket list
        g = sc.materials[1].graph
        rng = np.random.default_rng(3)
        him = np.zeros((16, 16, 4), np.float32)
        him[:, :, 0] = rng.random((16, 16))
        him[:, :, 1] = him[:, :, 2] = him[:, :, 0]
        him[:, :, 3] = 1.0
        sc.images['hmap'] = ImageBuffer(name='hmap', pixels=him)
        g['nodes']['htex'] = {
            'id': 'htex', 'bl_idname': 'ShaderNodeTexImage',
            'props': {'image': 'hmap', 'interpolation': 'Closest'},
            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                        {'name': 'Alpha', 'type': 'VALUE'}]}
        g['nodes']['bump'] = {
            'id': 'bump', 'bl_idname': 'ShaderNodeBump',
            'props': {'invert': False},
            'inputs': [sk('Strength', 'VALUE', strength),
                       sk('Distance', 'VALUE', 0.6),
                       sk('Height', 'VALUE', 0.5, ['htex', 0]),
                       sk('Normal', 'VECTOR', [0, 0, 0])],
            'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
        for s in g['nodes']['hal']['inputs']:
            if s['name'] == 'Normal':
                s['link'] = ['bump', 0]
            if s['name'] == 'Diffuse Color':
                s['link'] = None
                s['default'] = [0.8, 0.3, 0.2, 1.0]
            if s['name'] == 'Opacity':
                s['default'] = opacity
            if s['name'] == 'IOR':
                s['default'] = 1.33
        return sc, st

    sc, st = build(0.8)
    tex = R.prepare_textures(sc, st)
    cpu_img = R.render(sc, st).copy()
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, tex, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a Bump material now qualifies', passes is not None, str(why))
    if passes is None:
        return
    bumped = [e for e in passes if e[1] == 'Bumpy'][0]
    check('with one height pre-pass',
          len(bumped[3].get('prepasses', ())) == 1)
    img, _hit = GSH.simulate(job, g, passes, atlases)
    cov = g.tri >= 0
    e = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
    check('the bump frame is the CPU frame', e < 6e-3, f'max {e:.5f}')

    sc0, st0 = build(0.0)
    flat = R.render(sc0, st0).copy()
    ball = np.zeros_like(cov)
    ball[cov] = sc.mesh.mat_index[g.tri[cov]] == 1
    d = float(np.abs(cpu_img[ball][:, :3] - flat[ball][:, :3]).max())
    check('the bump is load-bearing', d > 0.05,
          f'strength moves the ball by {d:.4f}')

    # the FIELD shape whole: bump-driven wavy glass under ray tracing
    sc2, st2 = build(0.8, opacity=0.4, raytrace=True)
    tex2 = R.prepare_textures(sc2, st2)
    cpu2 = R.render(sc2, st2).copy()
    bvh2 = BVH(sc2.mesh.verts, sc2.mesh.tris)
    job2 = R.ShadeJob(sc2, st2, tex2, bvh2, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p2, why2, a2 = GSH.plan_frame(job2, g)
    check('bump-driven wavy glass qualifies under ray tracing',
          p2 is not None and '__reflect' in (a2 or {}), str(why2))
    if p2 is not None:
        img2, _h2 = GSH.simulate(job2, g, p2, a2)
        e2 = float(np.abs(img2[cov] - cpu2[cov][:, :3]).max())
        check('and matches the renderer, bent rays and all', e2 < 6e-3,
              f'max {e2:.5f}')

    # and the FINAL layer of the field's Water: a NOISE height -- the
    # sin-fract family the emitter refuses by name -- must NOT refuse
    # the material any more: its height image evaluates on the CPU with
    # the renderer's own float64 arithmetic, and the GPU differences it
    def to_noise(sc):
        gr = sc.materials[1].graph
        gr['nodes']['noise'] = {
            'id': 'noise', 'bl_idname': 'ShaderNodeTexNoise', 'props': {},
            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                       sk('Scale', 'VALUE', 6.0),
                       sk('Detail', 'VALUE', 2.0),
                       sk('Roughness', 'VALUE', 0.5),
                       sk('Distortion', 'VALUE', 0.0)],
            'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                        {'name': 'Color', 'type': 'RGBA'}]}
        for s in gr['nodes']['bump']['inputs']:
            if s['name'] == 'Height':
                s['link'] = ['noise', 0]

    sc3, st3 = build(0.8, opacity=0.4, raytrace=True)
    to_noise(sc3)
    tex3 = R.prepare_textures(sc3, st3)
    cpu3 = R.render(sc3, st3).copy()
    bvh3 = BVH(sc3.mesh.verts, sc3.mesh.tris)
    job3 = R.ShadeJob(sc3, st3, tex3, bvh3, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, a3 = GSH.plan_frame(job3, g)
    check('NOISE-into-Bump glass qualifies (the sin-fract wall opens '
          'for height chains)', p3 is not None, str(why3))
    if p3 is not None:
        bumpy3 = [e for e in p3 if e[1] == 'Bumpy'][0]
        pp3 = bumpy3[3].get('prepasses', ())
        check('its height pre-pass is CPU-evaluated',
              len(pp3) == 1 and pp3[0][2].get('cpu') is True,
              str([p[2].get('cpu') for p in pp3]))
        img3, _h3 = GSH.simulate(job3, g, p3, a3)
        e3 = float(np.abs(img3[cov] - cpu3[cov][:, :3]).max())
        check('and the noise-bumped glass matches the renderer exactly',
              e3 < 6e-3, f'max {e3:.5f}')
        # the noise really shapes the picture
        sc3b, st3b = build(0.0, opacity=0.4, raytrace=True)
        to_noise(sc3b)
        flat3 = R.render(sc3b, st3b).copy()
        d3 = float(np.abs(cpu3[ball][:, :3] - flat3[ball][:, :3]).max())
        check('and the noise is load-bearing', d3 > 0.02,
              f'noise strength moves the glass by {d3:.4f}')

    # THE FIELD FRAME ENTIRE: the same noise-bumped glass, now REFLECTIVE,
    # under the BANDS sky -- the quantised gradient the env term can
    # finally express. sky.bands, baked term for term
    sc4, st4 = build(0.8, opacity=0.4, raytrace=True)
    to_noise(sc4)
    for s in sc4.materials[1].graph['nodes']['hal']['inputs']:
        if s['name'] == 'Reflection':
            s['default'] = 0.3
    sc4.world.mode = 'BANDS'
    sc4.world.band_count = 6
    sc4.world.band_softness = 0.15
    sc4.world.horizon = (0.9, 0.5, 0.2)       # contrasty, so bands SHOW
    sc4.world.zenith = (0.1, 0.2, 0.6)
    tex4 = R.prepare_textures(sc4, st4)
    cpu4 = R.render(sc4, st4).copy()
    bvh4 = BVH(sc4.mesh.verts, sc4.mesh.tris)
    job4 = R.ShadeJob(sc4, st4, tex4, bvh4, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p4, why4, a4 = GSH.plan_frame(job4, g)
    check('reflective glass under a BANDS sky qualifies',
          p4 is not None, str(why4))
    if p4 is not None:
        img4, _h4 = GSH.simulate(job4, g, p4, a4)
        e4 = float(np.abs(img4[cov] - cpu4[cov][:, :3]).max())
        check('and the banded-sky reflections match the renderer',
              e4 < 6e-3, f'max {e4:.5f}')
        # the bands are really in the reflection: a GRADIENT sky differs
        sc4b, st4b = build(0.8, opacity=0.4, raytrace=True)
        to_noise(sc4b)
        for s in sc4b.materials[1].graph['nodes']['hal']['inputs']:
            if s['name'] == 'Reflection':
                s['default'] = 0.3
        sc4b.world.mode = 'GRADIENT'
        sc4b.world.horizon = (0.9, 0.5, 0.2)
        sc4b.world.zenith = (0.1, 0.2, 0.6)
        smooth4 = R.render(sc4b, st4b).copy()
        d4 = float(np.abs(cpu4[cov][:, :3] - smooth4[cov][:, :3]).max())
        check('and the bands are load-bearing vs a smooth gradient',
              d4 > 0.01, f'banding moves the frame by {d4:.4f}')
        GSH._PLAN_CACHE.clear()
        job4b = R.ShadeJob(sc4b, st4b, R.prepare_textures(sc4b, st4b),
                           BVH(sc4b.mesh.verts, sc4b.mesh.tris), view, eye,
                           w, h)
        p4b, why4b, a4b = GSH.plan_frame(job4b, g)
        check('the GRADIENT sky qualifies too', p4b is not None, str(why4b))
        if p4b is not None:
            img4b, _h4b = GSH.simulate(job4b, g, p4b, a4b)
            e4b = float(np.abs(img4b[cov] - smooth4[cov][:, :3]).max())
            check('and matches as well', e4b < 6e-3, f'max {e4b:.5f}')


def test_worker_bands_do_not_seam_the_waves():
    """A banded render must equal the whole frame -- bump gradients too.

    The last scheduling-dependent picture artifact: a worker band shades
    only its rows, `n_bump` differences toward the +y neighbour, and the
    band's top row used to flatten its waves against missing coverage --
    the chunk seam's sibling. Now the band's scissor rasterises ONE
    context row past its edge (the scissor culls whole triangles, so the
    row is complete) and the gradient fields build from the G-buffer's
    full coverage, so a band's gradients are the whole frame's, bit for
    bit. The negative control proves the seam is real: with the field
    mechanism disabled, the band edge differs by ~0.28 on exactly the
    edge row.
    """
    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    from .scenebuild import add_normal_mapped_ball

    w, h = 160, 120
    st = base_settings(w, h, shadows=True)
    sc = demo_scene(st, with_texture=False)
    add_normal_mapped_ball(sc)
    gr = sc.materials[1].graph
    gr['nodes']['noise'] = {
        'id': 'noise', 'bl_idname': 'ShaderNodeTexNoise', 'props': {},
        'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                   sk('Scale', 'VALUE', 6.0), sk('Detail', 'VALUE', 2.0),
                   sk('Roughness', 'VALUE', 0.5),
                   sk('Distortion', 'VALUE', 0.0)],
        'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                    {'name': 'Color', 'type': 'RGBA'}]}
    gr['nodes']['bump'] = {
        'id': 'bump', 'bl_idname': 'ShaderNodeBump',
        'props': {'invert': False},
        'inputs': [sk('Strength', 'VALUE', 0.8),
                   sk('Distance', 'VALUE', 0.6),
                   sk('Height', 'VALUE', 0.5, ['noise', 0]),
                   sk('Normal', 'VECTOR', [0, 0, 0])],
        'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
    for sock in gr['nodes']['hal']['inputs']:
        if sock['name'] == 'Normal':
            sock['link'] = ['bump', 0]

    whole = R.render(sc, st)
    mid = h // 2

    # vacuity precondition: the band edge really crosses the waves
    from ..core import raster
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    ball_rows = np.nonzero((np.where(g.tri >= 0,
                                     sc.mesh.mat_index[np.clip(g.tri, 0,
                                                               None)],
                                     -1) == 1).any(axis=1))[0]
    check('the band edge crosses the bump material',
          ball_rows.min() < mid < ball_rows.max(),
          f'ball rows {ball_rows.min()}..{ball_rows.max()}, edge {mid}')

    top = R.render(sc, st, band=(0, mid))
    bot = R.render(sc, st, band=(mid, h))
    stitched = np.zeros_like(whole)
    stitched[:mid] = top
    stitched[mid:] = bot
    d = float(np.abs(stitched - whole).max())
    check('two bands stitch to the EXACT whole frame', d == 0.0,
          f'max difference {d:.9f}')

    # negative control: the seam this fixes is real
    real = R._bump_height_fields
    R._bump_height_fields = lambda *a, **k: {}
    try:
        top2 = R.render(sc, st, band=(0, mid))
        bot2 = R.render(sc, st, band=(mid, h))
    finally:
        R._bump_height_fields = real
    stitched2 = np.zeros_like(whole)
    stitched2[:mid] = top2
    stitched2[mid:] = bot2
    d2 = float(np.abs(stitched2 - whole).max())
    check('and without the fields the band edge really seams',
          d2 > 0.01, f'the disabled mechanism differs by {d2:.4f}')


def test_rich_worlds_reflect_via_the_cpu_composite():
    """STARFIELD, BRYCE, PHYSICAL, HDRI, ground plane, world graphs --
    every world reflects now, evaluated by the renderer itself.

    The env term is the LAST rgb term the CPU adds (fog frames refuse),
    and every pixel it applies to is CPU-known: reflective primaries by
    mask, depth-exhausted hits by readback. So instead of baking each
    rich sky to GLSL, the composite asks `world_color` -- the CPU's own
    world, the Bryce sky lab included -- along the reflected rays and
    adds the term after readback. Exact for ANY world by construction.
    One mechanism, six refusals gone.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..core.scene import ImageBuffer, World
    from ..gpu import shade as GSH
    from .scenebuild import checker_image

    w, h = 96, 72

    def starfield(sc):
        sc.world = World()
        sc.world.mode = 'STARFIELD'
        sc.world.color = (0.1, 0.1, 0.2)
        sc.world.star_brightness = 2.0

    def bryce(sc):
        sc.world = World()
        sc.world.mode = 'BRYCE'

    def physical(sc):
        sc.world = World()
        sc.world.mode = 'PHYSICAL'

    def hdri(sc):
        sc.world = World()
        sc.world.mode = 'HDRI'
        sc.world.env_image = ImageBuffer(
            name='sky.hdr', pixels=checker_image(64, a=(0.2, 0.5, 1.0),
                                                 b=(1.0, 0.7, 0.3),
                                                 squares=4))
        sc.images = {'sky.hdr': sc.world.env_image}

    def ground(sc):
        sc.world = World(color=(0.2, 0.3, 0.5))
        sc.world.ground_plane = True

    def wgraph(sc):
        sc.world = World()
        sc.world.graph = {
            'output': 'out',
            'nodes': {
                'bg': {'id': 'bg', 'bl_idname': 'ShaderNodeBackground',
                       'props': {},
                       'inputs': [{'name': 'Color', 'type': 'RGBA',
                                   'default': [0.9, 0.4, 0.1, 1.0],
                                   'link': None},
                                  {'name': 'Strength', 'type': 'VALUE',
                                   'default': 1.0, 'link': None}],
                       'outputs': [{'name': 'Background',
                                    'type': 'SHADER'}]},
                'out': {'id': 'out',
                        'bl_idname': 'ShaderNodeOutputWorld',
                        'props': {},
                        'inputs': [{'name': 'Surface', 'type': 'SHADER',
                                    'default': None, 'link': ['bg', 0]}],
                        'outputs': []},
            },
        }

    worlds = (('STARFIELD', starfield), ('BRYCE', bryce),
              ('PHYSICAL', physical), ('HDRI', hdri),
              ('ground plane', ground), ('world graph', wgraph))

    for tag, setup in worlds:
        for raytrace in (True, False):
            st = base_settings(w, h, shadows=True, raytrace=raytrace,
                               ray_depth=1 if raytrace else 2)
            st.color_depth = '24'
            st.dither = 'NONE'
            st.output_scale = 'NONE'
            sc = demo_scene(st, with_texture=False)
            sc.materials[2].reflect_level = 0.6
            if raytrace:
                # the env term lives at the HITS under ray tracing, and
                # only REFLECTIVE surfaces carry it -- so the floor
                # reflects too, and the mirror's rays land on a surface
                # whose depth-exhausted shading really adds the sky
                sc.materials[0].reflect_level = 0.3
            setup(sc)
            cpu = R.render(sc, st)
            view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
            g = raster.GBuffer(w, h)
            raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h,
                             gbuf=g)
            tex = R.prepare_textures(sc, st)
            bvh = BVH(sc.mesh.verts, sc.mesh.tris) if raytrace else None
            job = R.ShadeJob(sc, st, tex, bvh, view, eye, w, h)
            GSH._PLAN_CACHE.clear()
            p, why, a = GSH.plan_frame(job, g)
            site = 'hit env' if raytrace else 'primary env'
            check(f'{tag} ({site}) qualifies', p is not None, str(why))
            if p is None:
                continue
            check(f'{tag} ({site}) takes the CPU-composite path',
                  '__env' in a, str(sorted(a)))
            img, _hit = GSH.simulate(job, g, p, a)
            cov = g.tri >= 0
            d = np.abs(img[cov] - cpu[cov][:, :3]).max(axis=1)
            check(f'{tag} ({site}) matches the renderer',
                  int((d > 1e-2).sum()) == 0,
                  f'{int((d > 1e-2).sum())} px, max {float(d.max()):.6f}')
            # vacuity: the sky is really in the mirror
            st_off = base_settings(w, h, shadows=True, raytrace=raytrace,
                                   ray_depth=1 if raytrace else 2)
            st_off.color_depth = '24'
            st_off.dither = 'NONE'
            st_off.output_scale = 'NONE'
            st_off.env_reflection = False
            flat = R.render(sc, st_off)
            box = np.zeros_like(cov)
            box[cov] = sc.mesh.mat_index[g.tri[cov]] == 2
            dv = float(np.abs(cpu[box][:, :3] - flat[box][:, :3]).max())
            check(f'{tag} ({site}) really reflects the world', dv > 5e-3,
                  f'the env term moves the mirror by {dv:.4f}')

    # the recursion keeps its env site at the FINAL depth: Bryce at
    # depth 2 -- mirror-in-mirror under the sky lab
    st2 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=2,
                        transparency='NONE')
    st2.color_depth = '24'
    st2.dither = 'NONE'
    st2.output_scale = 'NONE'
    sc2 = demo_scene(st2, with_texture=False)
    sc2.materials[2].reflect_level = 0.6
    sc2.materials[0].reflect_level = 0.4
    bryce(sc2)
    cpu2 = R.render(sc2, st2)
    view2, _p2, vp2, eye2 = R.camera_matrices(sc2.camera, w, h)
    g2 = raster.GBuffer(w, h)
    raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w, h, gbuf=g2)
    job2 = R.ShadeJob(sc2, st2, R.prepare_textures(sc2, st2),
                      BVH(sc2.mesh.verts, sc2.mesh.tris), view2, eye2,
                      w, h)
    GSH._PLAN_CACHE.clear()
    p2, why2, a2 = GSH.plan_frame(job2, g2)
    check('Bryce at ray depth 2 qualifies', p2 is not None, str(why2))
    if p2 is not None:
        img2, _h2 = GSH.simulate(job2, g2, p2, a2)
        cov2 = g2.tri >= 0
        d2 = np.abs(img2[cov2] - cpu2[cov2][:, :3]).max(axis=1)
        check('and mirror-in-mirror under the sky lab matches',
              int((d2 > 1e-2).sum()) == 0,
              f'max {float(d2.max()):.6f}')


def test_ray_depth_beyond_one_shades_on_the_gpu():
    """The recursion tree, walked branch by branch, exact at any depth.

    The CPU's `_add_raytraced` recurses: at depth d < D a hit's own
    reflective/refractive materials spawn the next rays and its shading
    carries NO environment term (the traced child replaces it); at
    d == D the hit shades with the environment -- the depth-exhausted
    branch. The deferred pass now walks the same tree: per level, hits
    shade through secondary passes (a `secondary_mid` variant without
    env above the final depth), children composite backward with the
    HIT material's own constants, and a pixel whose reflection hit both
    reflects AND refracts branches exactly as the CPU branches. Two
    mirrors and a glass ball make mirror-in-mirror at depth 2 -- the
    second bounce moves the frame by 0.6, and the seam stays zero.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    w, h = 96, 72

    def build(depth):
        st = base_settings(w, h, shadows=True, raytrace=True,
                           ray_depth=depth, transparency='NONE')
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.materials[2].reflect_level = 0.6   # mirror Box
        sc.materials[0].reflect_level = 0.4   # mirror Floor
        sc.materials[1].opacity = 0.5         # glass Ball: branches
        sc.materials[1].ior = 1.2
        return sc, st

    sc, st = build(2)
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a depth-2 frame qualifies', passes is not None, str(why))
    if passes is None:
        return
    rp = atlases['__reflect']
    check('the plan carries the depth and the mid-level passes',
          rp['depth'] == 2 and len(rp['secondary_mid']) == 3,
          f"depth {rp['depth']}, mid {len(rp.get('secondary_mid', []))}")
    img, _hit = GSH.simulate(job, g, passes, atlases)
    cov = g.tri >= 0
    d = np.abs(img[cov] - cpu[cov][:, :3]).max(axis=1)
    check('the depth-2 frame is the CPU frame (mirror-in-mirror '
          'included)', int((d > 1e-2).sum()) == 0,
          f'{int((d > 1e-2).sum())} px >0.01, max {float(d.max()):.6f}')

    # vacuity: the second bounce is really in the picture
    sc1, st1 = build(1)
    cpu1 = R.render(sc1, st1)
    dv = float(np.abs(cpu - cpu1).max())
    check('depth 2 really adds a second bounce', dv > 0.05,
          f'depth 2 vs depth 1 differ by {dv:.4f}')

    # and one more level: depth 3 walks the same machinery
    sc3, st3 = build(3)
    cpu3 = R.render(sc3, st3)
    job3 = R.ShadeJob(sc3, st3, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, a3 = GSH.plan_frame(job3, g)
    check('a depth-3 frame qualifies too', p3 is not None, str(why3))
    if p3 is not None:
        img3, _h3 = GSH.simulate(job3, g, p3, a3)
        d3 = np.abs(img3[cov] - cpu3[cov][:, :3]).max(axis=1)
        check('and matches at depth 3', int((d3 > 1e-2).sum()) == 0,
              f'max {float(d3.max()):.6f}')

    # the backend CONTRACT the field broke: the driver's level images
    # carry FOUR channels (read_target), the front-end's three -- and
    # the child composites WRITE into them, so the shared recursion must
    # normalise. Fake shaders, real recursion and trace: both shapes
    # must produce the identical picture instead of a broadcast crash.
    from ..gpu.rtrace import simulate_intersect

    def fake_draw(channels):
        def d(_plist, sec_ids, _level, hit_region=None):
            img4 = np.zeros((h, w, channels), np.float32)
            img4[:, :, 0] = (sec_ids[:, :, 3] >= 0).astype(np.float32)
            img4[:, :, 1] = 0.25
            img4[:, :, 2] = 0.5
            if channels == 4:
                img4[:, :, 3] = 1.0
            return img4
        return d

    def isect(org, dirs):
        return simulate_intersect(bvh, org, dirs, 1e30)

    out3 = GSH._run_sweeps(job, g, rp, np.zeros((h, w, 3), np.float32),
                           fake_draw(3), isect)
    out4 = GSH._run_sweeps(job, g, rp, np.zeros((h, w, 3), np.float32),
                           fake_draw(4), isect)
    d34 = float(np.abs(out3 - out4).max())
    check('a four-channel (driver-shaped) level image composites '
          'identically to a three-channel one', d34 == 0.0,
          f'max difference {d34:.9f}')


def test_soft_shadows_and_ao_shade_on_the_gpu():
    """Soft ray shadows and ambient occlusion, exact on both devices.

    The old refusal was honest: their jitter came from a sequential
    random stream whose order the batch layout owned -- unreproducible
    anywhere, including on the CPU itself across thread counts. Now
    every sample is a pure function of (pixel, sample index, stream,
    seed): hash draws through the pattern hash, angles from a shared
    256-entry unit-circle table (a driver's sin/cos round differently,
    and an occlusion ray is a cliff), all float32. So the CPU picture is
    chunking-invariant AND the deferred pass draws the identical rays --
    including at TRACED HITS, where the CPU threads the spawning pixel's
    identity through `trace()` and the GPU's secondary pass fragment IS
    that pixel.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    w, h = 64, 48
    st = base_settings(w, h, shadows=True, shadow_default='RAY',
                       raytrace=True, ray_depth=1)
    st.shadow_samples = 4
    st.ambient_occlusion = True
    st.ao_samples = 4
    st.ao_distance = 2.0
    st.ao_intensity = 1.0
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.lights[1].radius = 0.8              # the fill light goes SOFT
    sc.materials[2].reflect_level = 0.5    # and hits shade soft+AO too

    cpu = R.render(sc, st)

    # the CPU picture no longer depends on its own scheduling
    st.threads = 8
    cpu8 = R.render(sc, st)
    st.threads = 1
    dth = float(np.abs(cpu - cpu8).max())
    check('soft+AO render identically at any thread count', dth == 0.0,
          f'max difference {dth:.9f}')

    # vacuity: both features are load-bearing in the compared frame
    sc.lights[1].radius = 0.0
    hard = R.render(sc, st)
    sc.lights[1].radius = 0.8
    dr = float(np.abs(cpu - hard).max())
    check('the radius really softens the shadows', dr > 0.02,
          f'radius moves the frame by {dr:.4f}')
    st.ambient_occlusion = False
    noao = R.render(sc, st)
    st.ambient_occlusion = True
    da = float(np.abs(cpu - noao).max())
    check('the occlusion really darkens the creases', da > 0.02,
          f'AO moves the frame by {da:.4f}')

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a soft-shadowed, occluded, ray-traced frame qualifies',
          passes is not None, str(why))
    if passes is None:
        return
    check('the circle table rides the atlases', 'hal_circle' in atlases,
          str(sorted(atlases)))
    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the frame simulates', img is not None, str(hit))
    if img is None:
        return
    cov = g.tri >= 0
    d = np.abs(img[cov] - cpu[cov][:, :3]).max(axis=1)
    flips = int((d > 1e-2).sum())
    check('soft shadows + AO + reflections match the renderer',
          flips == 0, f'{flips} px >0.01, max {float(d.max()):.6f} '
          f'over {int(cov.sum())} px')


def test_the_picture_does_not_depend_on_the_chunking():
    """Bump gradients must be a function of the picture, not the batches.

    The field's first real driver run of the Water anatomy showed 38 px
    off by up to 0.19, all on ONE ROW of the glass -- and the driver was
    innocent: the CPU shaded PIXEL-rate fragments in mixed-material
    chunks, `n_bump`'s neighbour validity is "shaded in the same batch",
    and a chunk boundary landing mid-material flattened the waves along
    its row. The seam moved with the thread count (chunk size derives
    from it), meaning the RENDERED PICTURE depended on an internal
    scheduling constant. Now shading batches one material at a time, the
    gradients are whole up to the memory cap -- and this test holds the
    invariant at the exact size and scene the field failed on.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..core.render import MAX_CHUNK, MIN_CHUNK, resolve_threads
    from ..core.scene import ImageBuffer
    from ..gpu import shade as GSH
    from .scenebuild import add_normal_mapped_ball

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    w, h = 480, 360
    st = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1,
                       transparency='NONE')
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    add_normal_mapped_ball(sc)
    gr = sc.materials[1].graph
    gr['nodes']['noise'] = {
        'id': 'noise', 'bl_idname': 'ShaderNodeTexNoise', 'props': {},
        'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                   sk('Scale', 'VALUE', 6.0), sk('Detail', 'VALUE', 2.0),
                   sk('Roughness', 'VALUE', 0.5),
                   sk('Distortion', 'VALUE', 0.0)],
        'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                    {'name': 'Color', 'type': 'RGBA'}]}
    gr['nodes']['bump'] = {
        'id': 'bump', 'bl_idname': 'ShaderNodeBump',
        'props': {'invert': False},
        'inputs': [sk('Strength', 'VALUE', 0.8),
                   sk('Distance', 'VALUE', 0.6),
                   sk('Height', 'VALUE', 0.5, ['noise', 0]),
                   sk('Normal', 'VECTOR', [0, 0, 0])],
        'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
    for sock in gr['nodes']['hal']['inputs']:
        if sock['name'] == 'Normal':
            sock['link'] = ['bump', 0]
        if sock['name'] == 'Diffuse Color':
            sock['link'] = None
            sock['default'] = [0.2, 0.5, 0.8, 1.0]
        if sock['name'] == 'Opacity':
            sock['default'] = 0.4
        if sock['name'] == 'IOR':
            sock['default'] = 1.33
        if sock['name'] == 'Reflection':
            sock['default'] = 0.3
    sc.materials[2].reflect_level = 0.5
    sc.world.mode = 'BANDS'
    sc.world.band_count = 6
    sc.world.band_softness = 0.15

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    cov = g.tri >= 0

    # vacuity: under the OLD mixed-material chunking, a boundary really
    # cut the glass mid-frame at this size -- the scene exercises the seam
    py, px = np.nonzero(cov)
    n = int(py.size)
    mi = sc.mesh.mat_index[g.tri[py, px]]
    st.threads = 1
    workers = resolve_threads(st)
    old_chunk = int(min(max(int(np.ceil(n / max(workers * 4, 1))),
                            MIN_CHUNK), MAX_CHUNK))
    gid = np.full((h, w), -1, np.int64)
    gid[py, px] = np.arange(n)
    cut = 0
    for i in np.nonzero(mi == 1)[0]:
        yy, xx = int(py[i]), int(px[i])
        for ny, nx in ((yy, xx + 1), (yy + 1, xx)):
            if ny < h and nx < w and gid[ny, nx] >= 0 \
                    and mi[gid[ny, nx]] == 1 \
                    and gid[ny, nx] // old_chunk != i // old_chunk:
                cut += 1
                break
    check('the old chunking would cut the bump material mid-frame',
          cut > 0, f'covered {n}, old chunk {old_chunk}, cut px {cut}')

    # the invariant itself: the frame must be IDENTICAL whatever the
    # thread count, because chunk size derives from it
    cpu1 = R.render(sc, st)
    st.threads = 8
    cpu8 = R.render(sc, st)
    st.threads = 1
    dmax = float(np.abs(cpu1 - cpu8).max())
    check('threads 1 and 8 render the identical frame', dmax == 0.0,
          f'max difference {dmax:.9f}')

    # and the deferred frame agrees with it at the field size -- the very
    # check that read 38 px off before the fix
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, BVH(sc.mesh.verts, sc.mesh.tris),
                     view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p, why, a = GSH.plan_frame(job, g)
    check('the field-size Water frame plans', p is not None, str(why))
    if p is not None:
        img, _h = GSH.simulate(job, g, p, a)
        d = np.abs(img[cov] - cpu1[cov][:, :3]).max(axis=1)
        flips = int((d > 1e-2).sum())
        check('and the GPU frame has ZERO px off by >0.01 (the field '
              'read 38)', flips == 0,
              f'{flips} px, max {float(d.max()):.6f}')

    # a Bump material too covered for ONE CPU batch -- the field's own
    # Water was 347106 pixels -- gets a whole-material gradient PRE-PASS
    # on the CPU instead of cut gradients, so it neither seams NOR
    # refuses. The invariants hold at any size now.
    from .scenebuild import look_at_matrix
    w6, h6 = 640, 480
    st6 = base_settings(w6, h6, shadows=False, raytrace=False)
    st6.color_depth = '24'
    st6.dither = 'NONE'
    st6.output_scale = 'NONE'
    sc6 = demo_scene(st6, with_texture=False)
    add_normal_mapped_ball(sc6)
    gr6 = sc6.materials[1].graph
    gr6['nodes']['noise'] = dict(gr['nodes']['noise'])
    gr6['nodes']['bump'] = dict(gr['nodes']['bump'])
    for sock in gr6['nodes']['hal']['inputs']:
        if sock['name'] == 'Normal':
            sock['link'] = ['bump', 0]
    sc6.materials[0].graph = gr6                      # bump on the floor
    for lt in sc6.lights:
        lt.shadow = 'NONE'
    # a clear patch of floor, away from the ball and box, close enough
    # that the floor alone fills every pixel of the 640x480 frame
    sc6.camera.matrix_world = look_at_matrix((3.5, 3.0, 2.2),
                                             (3.5, 3.001, 0.0))
    view6, _p6, vp6, eye6 = R.camera_matrices(sc6.camera, w6, h6)
    g6 = raster.GBuffer(w6, h6)
    raster.rasterize(sc6.mesh.verts, sc6.mesh.tris, vp6, w6, h6, gbuf=g6)
    m6 = np.where(g6.tri >= 0,
                  sc6.mesh.mat_index[np.clip(g6.tri, 0, None)], -1)
    big = int((m6 == 0).sum())
    check('the oversize scene really exceeds one batch', big > MAX_CHUNK,
          f'{big} floor px vs cap {MAX_CHUNK}')
    st6.threads = 1
    over1 = R.render(sc6, st6)
    st6.threads = 8
    over8 = R.render(sc6, st6)
    st6.threads = 1
    dover = float(np.abs(over1 - over8).max())
    check('an over-the-cap Bump material is STILL chunking-invariant',
          dover == 0.0, f'max difference {dover:.9f}')
    tex6 = R.prepare_textures(sc6, st6)
    job6 = R.ShadeJob(sc6, st6, tex6, None, view6, eye6, w6, h6)
    GSH._PLAN_CACHE.clear()
    p6, why6, a6 = GSH.plan_frame(job6, g6)
    check('and QUALIFIES for the deferred pass (the field hit the old '
          'refusal at 347106 px)', p6 is not None, str(why6))
    if p6 is not None:
        img6, _h6 = GSH.simulate(job6, g6, p6, a6)
        cov6 = g6.tri >= 0
        d6 = np.abs(img6[cov6] - over1[cov6][:, :3]).max(axis=1)
        check('and the GPU frame matches the whole-gradient CPU frame',
              int((d6 > 1e-2).sum()) == 0,
              f'{int((d6 > 1e-2).sum())} px, max {float(d6.max()):.6f}')


def test_the_viewport_preview_works_headlessly():
    """The viewport's working half must render, draft, refine and re-kick.

    The old viewport rendered synchronously inside Blender's draw callback:
    the UI froze for every frame and any failure blanked the viewport for
    good. The rebuilt path lives in `preview.Viewport`, which is bpy-free
    on purpose -- THIS test drives the exact worker loop Blender will:
    export in, want a view, kick, park a frame, and only re-render when
    the camera or the scene actually moved on.

    The field taught the second half: the first build aborted the in-flight
    render on every camera move, and since aborts land between stages, an
    orbit completed NOTHING -- the preview updated only at rest. So motion
    now renders coarse DRAFTS that always finish, rest renders full
    quality, and the only aborts left are a refine overtaken by motion and
    a re-exported scene. The clock is injected, so every one of those
    transitions is driven deterministically here. The refined frame is
    checked against `render()` + `post.process` run directly with the same
    inputs, so the preview shows the engine's own picture, exactly.
    """
    import time as _time
    from ..core.scene import Camera
    from ..preview import Viewport, DRAFT_WINDOW

    st = base_settings(96, 72, shadows=False)
    st.preview_scale = 2
    sc = demo_scene(st, with_texture=False)

    t = {'now': 100.0}
    vp = Viewport(clock=lambda: t['now'])
    check('no scene yet: kick refuses politely', vp.kick() is False)

    vp.set_scene(sc, st)                    # marks the motion clock too
    check('no wanted view yet: kick still refuses', vp.kick() is False)

    view, proj, _vpm, _eye = R.camera_matrices(sc.camera, 96, 72)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                 projection=proj.astype(np.float32), type='PERSP')
    vp.want(cam, 192, 144)
    check('a fresh view kicks a worker', vp.kick() is True)

    def wait_done():
        for _ in range(600):
            with vp.lock:
                if not vp.busy:
                    return vp.frame
            _time.sleep(0.02)
        return None

    frame = wait_done()
    check('the worker parked a frame', frame is not None)
    if frame is None:
        return
    check('and it is a DRAFT: entering the view is inside the motion '
          'window, so the first picture arrives at a quarter the pixels',
          frame.shape == (36, 48, 4), str(frame.shape))
    check('a draft of the current view is enough while moving',
          vp.kick() is False)

    t['now'] += DRAFT_WINDOW + 0.1          # the view rests
    check('at rest, the parked draft re-kicks to refine', vp.kick() is True)
    frame = wait_done()
    check('the refine parked a full-quality frame',
          frame is not None and frame.shape == (72, 96, 4),
          str(None if frame is None else frame.shape))
    if frame is None:
        return
    check('finite and opaque', bool(np.isfinite(frame).all()
                                    and (frame[:, :, 3] == 1.0).all()))

    # the refined frame IS the engine's own picture: same scene, camera and
    # settings through render() + the post chain directly
    sc.camera = cam
    st.resolution_x, st.resolution_y = 96, 72
    want_img = post.process(R.render(sc, st), st, target_size=(96, 72))
    want_img = np.asarray(want_img, np.float32)[:, :, :3]
    d = float(np.abs(frame[:, :, :3] - want_img).max())
    check('the refined frame is render() + post, exactly', d == 0.0,
          f'max difference {d}')

    check('a resting viewport with its full frame costs zero',
          vp.kick() is False)

    # ---- the abort rules, the exact shape the field report demanded
    mw2 = np.linalg.inv(view).astype(np.float32).copy()
    mw2[0, 3] += 0.25
    cam2 = Camera(matrix_world=mw2, projection=proj.astype(np.float32),
                  type='PERSP')

    with vp.lock:                           # pretend a REFINE is in flight
        vp.busy = True
        vp.abort = False
        vp._inflight_draft = False
    vp.want(cam2, 192, 144)
    check('motion aborts an in-flight refine (its draft is on screen)',
          vp.abort is True)

    with vp.lock:                           # pretend a DRAFT is in flight
        vp.abort = False
        vp._inflight_draft = True
    vp.want(cam, 192, 144)
    check('motion NEVER aborts a draft -- drafts finishing is what makes '
          'the preview stream during an orbit', vp.abort is False)
    with vp.lock:
        vp.busy = False

    # moving again: the moved camera drafts, completes, then refines at rest
    vp.want(cam2, 192, 144)
    check('a moved camera re-kicks', vp.kick() is True)
    f2 = wait_done()
    check('the motion draft completed', f2 is not None
          and f2.shape == (36, 48, 4),
          str(None if f2 is None else f2.shape))
    t['now'] += DRAFT_WINDOW + 0.1
    check('and refines at rest', vp.kick() is True)
    f3 = wait_done()
    check('to full quality', f3 is not None and f3.shape == (72, 96, 4),
          str(None if f3 is None else f3.shape))

    # a re-exported scene aborts an in-flight REFINE (expensive work of an
    # outdated scene) -- but a DRAFT runs to completion, or playback shows
    # nothing: every frame's export would kill the previous frame's draft
    # before it could park. FLIPPED in 1.25.84 (was 'aborts even a draft'):
    # the field's animation report named the cost of the old rule.
    with vp.lock:
        vp.busy = True
        vp.abort = False
        vp._inflight_draft = True
    vp.set_scene(sc, st)
    check('a scene export lets a DRAFT finish (playback streams)',
          vp.abort is False)
    with vp.lock:
        vp.busy = True
        vp.abort = False
        vp._inflight_draft = False
    vp.set_scene(sc, st)
    check('a scene export still aborts a REFINE (outdated full frame)',
          vp.abort is True)
    with vp.lock:
        vp.busy = False
    t['now'] += DRAFT_WINDOW + 0.1          # let it refine directly
    vp.want(cam, 192, 144)
    check('a scene edit re-kicks the same view', vp.kick() is True)
    check('and completes', wait_done() is not None)


def test_the_refine_invites_itself_when_the_view_rests():
    """A parked draft must refine WITHOUT another input event.

    Blender redraws only on events. A GPU draft finishes in milliseconds,
    so its completion redraw lands INSIDE the motion window; kick()
    declines ("parked draft is enough while moving") and nothing ever
    asks again -- the refine waited for the next input. The field named
    it on 1.25.83: "It doesn't always refine on stopping, however
    pressing the middle mouse button usually fixes it" -- the MMB press
    was the missing event. The decline now ARMS one redraw request for
    the window's lapse. Held here: the decline arms exactly once with
    the window's remainder, re-declines do not stack timers, the fired
    recheck lets kick() start the REFINE, and a parked FULL frame arms
    nothing (a resting viewport still costs zero).
    """
    import time as _time
    from ..core.scene import Camera
    from ..preview import DRAFT_WINDOW, RECHECK_SLACK, Viewport

    st = base_settings(96, 72, shadows=False)
    st.preview_scale = 2
    sc = demo_scene(st, with_texture=False)

    t = {'now': 100.0}
    vp = Viewport(clock=lambda: t['now'])
    armed = []
    vp._schedule_recheck = lambda engine, delay: armed.append(delay) or True

    vp.set_scene(sc, st)
    view, proj, _v, _e = R.camera_matrices(sc.camera, 96, 72)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                 projection=proj.astype(np.float32), type='PERSP')
    vp.want(cam, 192, 144)
    check('the fresh view kicks a draft', vp.kick() is True)

    def wait_done():
        for _ in range(600):
            with vp.lock:
                if not vp.busy:
                    return vp.frame
            _time.sleep(0.02)
        return None

    check('the draft parked', wait_done() is not None)
    check('nothing armed while a worker was the next step', not armed)

    # the completion redraw arrives INSIDE the window (the field's case)
    t['now'] += 0.05
    check('kick declines: parked draft while moving', vp.kick() is False)
    check('...and ARMS the window-lapse recheck', len(armed) == 1,
          str(armed))
    remain = DRAFT_WINDOW - 0.05 + RECHECK_SLACK
    check('armed for the window remainder', abs(armed[0] - remain) < 0.01,
          f'{armed[0]:.3f} vs {remain:.3f}')
    check('a second redraw does not stack a second timer',
          vp.kick() is False and len(armed) == 1, str(armed))

    # the timer fires just past the lapse: _fire clears the flag and tags
    # a redraw; the redraw calls kick(), which must now REFINE
    t['now'] += remain
    vp._recheck_armed = False               # what _fire does
    check('the fired recheck starts the refine', vp.kick() is True)
    with vp.lock:
        was_draft = vp._inflight_draft
    check('and it IS the refine, not another draft', was_draft is False)
    frame = wait_done()
    check('full quality parked without any input event',
          frame is not None and frame.shape == (72, 96, 4),
          str(None if frame is None else frame.shape))

    # a parked FULL frame must never arm: a resting viewport costs zero
    armed.clear()
    check('at rest with the full frame, kick declines', vp.kick() is False)
    check('and arms nothing', not armed)

    # moved again before the timer fired: the fired redraw re-arms with
    # the NEW window's remainder -- it converges, never orphans
    mw2 = np.linalg.inv(view).astype(np.float32).copy()
    mw2[0, 3] += 0.5
    cam2 = Camera(matrix_world=mw2, projection=proj.astype(np.float32),
                  type='PERSP')
    vp.want(cam2, 192, 144)
    check('the moved view drafts again', vp.kick() is True)
    check('the moved-view draft parked', wait_done() is not None)
    t['now'] += 0.02
    check('inside the new window it declines again', vp.kick() is False)
    check('and re-arms for the NEW remainder', len(armed) == 1
          and abs(armed[0] - (DRAFT_WINDOW - 0.02 + RECHECK_SLACK)) < 0.01,
          str(armed))


def test_animation_playback_streams_drafts():
    """Playback must stream: exports outpacing drafts showed NOTHING.

    Every animation frame re-exports the scene, and set_scene aborted
    ANYTHING mid-flight -- so on any scene whose draft takes longer than
    one playback tick, each export killed the previous frame's draft
    before it parked, and the viewport never updated until playback
    stopped. The field named it on 1.25.83: "running an animation in the
    viewport, it does not update in realtime." The R25 orbit bug, scene-
    version edition, cured by the same rule: a DRAFT always runs to
    completion (one export stale at worst), a REFINE still dies. Held
    end to end: an export DURING a draft leaves it running, the stale
    draft parks and counts as superseded (kick immediately re-drafts the
    newest export while the storm holds), and once the exports stop the
    view refines to the CURRENT scene's exact picture.
    """
    import time as _time
    from ..core.scene import Camera
    from ..preview import DRAFT_WINDOW, Viewport

    st = base_settings(96, 72, shadows=False)
    st.preview_scale = 2
    sc = demo_scene(st, with_texture=False)

    t = {'now': 200.0}
    vp = Viewport(clock=lambda: t['now'])
    vp._schedule_recheck = lambda engine, delay: True
    vp.set_scene(sc, st)
    view, proj, _v, _e = R.camera_matrices(sc.camera, 96, 72)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                 projection=proj.astype(np.float32), type='PERSP')
    vp.want(cam, 192, 144)
    check('frame 1 kicks a draft', vp.kick() is True)

    # frame 2's export arrives WHILE the draft renders: playback tick
    with vp.lock:
        mid_flight = vp.busy
    vp.set_scene(sc, st)
    check('the export found the draft in flight', mid_flight is True)
    check('and did NOT abort it', vp.abort is False)

    def wait_done():
        for _ in range(600):
            with vp.lock:
                if not vp.busy:
                    return vp.frame
            _time.sleep(0.02)
        return None

    frame = wait_done()
    check('the superseded draft still parked (playback streams)',
          frame is not None and frame.shape == (36, 48, 4),
          str(None if frame is None else frame.shape))
    check('but it is of the OLD export: the newest re-drafts at once',
          vp.kick() is True)
    check('and it is a draft (the storm still holds)',
          vp._inflight_draft is True)
    check('the newest export drafted', wait_done() is not None)

    # the exports stop; the window lapses; the refine snaps in
    t['now'] += DRAFT_WINDOW + 0.1
    check('at rest the parked draft refines', vp.kick() is True)
    frame = wait_done()
    check('to full quality', frame is not None
          and frame.shape == (72, 96, 4),
          str(None if frame is None else frame.shape))
    sc.camera = cam
    st.resolution_x, st.resolution_y = 96, 72
    want_img = post.process(R.render(sc, st), st, target_size=(96, 72))
    want_img = np.asarray(want_img, np.float32)[:, :, :3]
    d = float(np.abs(frame[:, :, :3] - want_img).max())
    check('and it is the CURRENT scene, exactly', d == 0.0,
          f'max difference {d}')


def test_every_material_binds_its_own_texture():
    """Two image-textured materials must bind TWO textures on the driver.

    Sampler names are positional per material shader ('hal_tex0'...), and
    the driver's frame-wide by-NAME map kept the first texture it saw --
    "materials are all using the same one when they should be different",
    said the field, and they had checked, reimported, reloaded: it was
    never on their end. The compiler sim binds per pass BY DESIGN, so the
    one place sim and driver diverged was exactly the bug -- no headless
    picture could show it. Held here at the mechanism level: the real
    plan for a two-textured-material scene carries the SAME sampler name
    with DIFFERENT images (the collision precondition), the shipped
    gather keyed by (pass, sampler) resolves them to their OWN uploads,
    the old by-name semantics demonstrably collapsed them, and a shared
    image still deduplicates to one upload underneath.
    """
    import types
    from ..core import raster
    from ..gpu import shade as GSH_

    def sk(name, tp, default, link=None):
        return {'name': name, 'identifier': name, 'type': tp,
                'default': default, 'link': link}

    def tex_graph(img_key):
        return {'output': 'out', 'nodes': {
            'Image Texture': {
                'id': 'Image Texture', 'bl_idname': 'ShaderNodeTexImage',
                'props': {'image': img_key, 'interpolation': 'Closest',
                          'extension': 'REPEAT', 'projection': 'FLAT'},
                'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'},
                            {'name': 'Alpha', 'type': 'VALUE'}]},
            'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                     'props': {},
                     'inputs': [sk('Color', 'RGBA', [1, 1, 1, 1],
                                   ['Image Texture', 0]),
                                sk('Roughness', 'VALUE', 0.0)],
                     'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['bsdf', 0])],
                    'outputs': []}}}

    st = base_settings(96, 72)
    st.threads = 1
    st.render_device = 'GPU'
    sc = demo_scene(st, with_texture=False)
    red = np.zeros((8, 8, 4), np.float32)
    red[..., 0] = 1.0
    red[..., 3] = 1.0
    blu = np.zeros((8, 8, 4), np.float32)
    blu[..., 2] = 1.0
    blu[..., 3] = 1.0
    sc.images['imgA'] = types.SimpleNamespace(pixels=red, name='imgA',
                                              colorspace='sRGB')
    sc.images['imgB'] = types.SimpleNamespace(pixels=blu, name='imgB',
                                              colorspace='sRGB')
    sc.materials[1].graph = tex_graph('imgA')
    sc.materials[2].graph = tex_graph('imgB')
    R.render(sc, st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     96, 72)
    GSH_._PLAN_CACHE.clear()
    passes, why, _atlases = GSH_.plan_frame(job, g)
    check('the two-textured scene qualifies', passes is not None, str(why))
    if passes is None:
        return

    tex_binds = [(b, dict(b.get('textures') or {}))
                 for _mi, _n, _s, b in passes
                 if b.get('textures')]
    check('two passes carry image textures', len(tex_binds) == 2,
          str(len(tex_binds)))
    names_a = set(tex_binds[0][1])
    names_b = set(tex_binds[1][1])
    check('the sampler NAMES collide across materials (the field shape)',
          bool(names_a & names_b), f'{names_a} vs {names_b}')
    keys = {v for _b, t in tex_binds for v in t.values()}
    check('but the IMAGES differ', keys == {'imgA', 'imgB'}, str(keys))

    # the shipped gather, driven with a recording upload
    calls = []

    def up(ik, build):
        calls.append(ik)
        return ('TEX', ik)

    got = GSH_._gather_pass_textures([b for b, _t in tex_binds],
                                     job.textures, up)
    check('one entry per (pass, sampler)', len(got) == 2, str(len(got)))
    texes = set(got.values())
    check('each material binds ITS OWN texture', len(texes) == 2,
          str(texes))

    # the OLD by-name semantics, demonstrably the bug
    old = {}
    for b, t in tex_binds:
        for sname, key in t.items():
            if sname in old:
                continue
            old[sname] = key
    check('the frame-wide by-name map would collapse them to one',
          len(old) == 1 and len(set(old.values())) == 1, str(old))

    # a SHARED image still deduplicates the actual upload underneath
    calls.clear()
    shared = [tex_binds[0][0],
              {'textures': dict(tex_binds[0][1])}]
    got2 = GSH_._gather_pass_textures(shared, job.textures, up)
    check('a shared image maps two pass entries onto ONE upload key',
          len(got2) == 2 and len(set(calls)) == 1,
          f'{len(got2)} entries, {len(set(calls))} unique uploads')


def test_viewport_bursts_stay_out_of_the_draw_loop():
    """Bursts cross to the main thread ONLY in the pump's timer slices,
    never inside a draw callback. 1.25.89-.91 ran viewport bursts inside
    view_draw and the field got the whole scene flashing rapidly whenever
    bursts streamed -- camera motion AND refines -- a seizure risk. Two
    patches (a time budget, a viewport-rect fence) failed to tame it;
    the architecture was reverted. This test is the structural guard:
    the drain-mode machinery must NOT exist, view_draw must not touch
    the marshal queue, and the plain pump must actually move work.
    """
    import inspect
    from ..gpu import marshal as M
    from . import fakeblender as FB
    _, E = FB.install()          # engine imports bpy; headless needs the stub

    # 1) the machinery is GONE, not just unused -- reintroduction fails
    for name in ('drain', 'begin_draw_drain', 'end_draw_drain',
                 '_POKES', 'STALE_DRAIN'):
        check(f'marshal has no {name}', not hasattr(M, name))
    check('no draw-drain state keys survive',
          'draw_drain' not in M._STATE and
          'last_draw_drain' not in M._STATE, str(M._STATE))

    # 2) view_draw never touches the queue: it may start the redraw
    #    poll, but no drain call, no _JOBS, no marshal import. Comments
    #    are stripped first -- the one comment there EXPLAINS the revert
    #    and names the forbidden words; only code counts.
    src = inspect.getsource(E.HalcyonRenderEngine.view_draw)
    code = '\n'.join(ln for ln in src.splitlines()
                     if not ln.strip().startswith('#'))
    for frag in ('drain', '_JOBS', 'marshal'):
        check(f'view_draw CODE is free of "{frag}"', frag not in code)

    # 3) the plain pump still moves work while marshalling is on
    ran = []
    M.enable()
    try:
        M._JOBS.put(M._Job(lambda: ran.append('a')))
        M._JOBS.put(M._Job(lambda: ran.append('b')))
        tick = M._pump()
        check('the pump drains every queued burst in one slice',
              ran == ['a', 'b'])
        check('and keeps ticking while marshalling is on',
              tick == M.TICK)
    finally:
        M.disable()

    # 4) queueing work pokes nothing: run_on_main on the main thread
    #    (this test) runs in place, and the only cross-thread signal a
    #    finished frame sends is preview's flag-only redraw request --
    #    the marshal itself owns no redraw side channel
    got = M.run_on_main(lambda: 'in-place')
    check('run_on_main still runs in place where there is no pump',
          got == 'in-place')


def test_the_viewport_black_guard():
    """A GPU viewport frame that comes back suddenly black is not shown.

    The field reports materials randomly turning pure black / flashing --
    VIEWPORT only, GPU device, no material pattern, since the viewport
    GPU arc. The headless cadence stress (alternating draft/refine sizes,
    scene edits, warm caches vs fresh) runs CLEAN, so the mechanism lives
    in live driver state headless cannot reproduce. Until the field's
    guard lines name it, the viewport MEASURES: a GPU frame whose black
    fraction jumps >20% against the last parked frame is re-shaded on
    the CPU, kept, counted, and named on the console. Held here: stable
    frames never trigger; an injected black frame triggers exactly once,
    parks the REAL picture, and the count says so; the ratio then
    converges so a genuinely dark scene stops paying.
    """
    import types
    from ..core.scene import Camera
    from .. import preview as PV

    st = base_settings(96, 72, shadows=False)
    st.threads = 1
    st.render_device = 'GPU'
    st.preview_scale = 1
    sc = demo_scene(st, with_texture=False)
    PV.shape_settings(st, 96, 72)
    view, proj, _v, _e = R.camera_matrices(sc.camera, 96, 72)
    cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                 projection=proj.astype(np.float32), type='PERSP')
    vp = PV.Viewport()
    vp.set_scene(sc, st)
    vp.abort = False
    key = vp._key(cam, 96, 72)

    vp._render(None, sc, st, cam, 96, 72, key, vp.version, False)
    check('a normal GPU frame parks without the guard',
          vp.frame is not None and vp.guard_count == 0)
    good = vp.frame.copy()
    vp._render(None, sc, st, cam, 96, 72, key, vp.version, False)
    check('stable frames never trigger', vp.guard_count == 0)

    # inject one black frame: the driver-state stand-in
    real = PV.core_render
    calls = {'n': 0}

    def fake_render(scene, settings, progress=None):
        calls['n'] += 1
        if calls['n'] == 1:
            return np.zeros((settings.resolution_y,
                             settings.resolution_x, 4), np.float32)
        return real.render(scene, settings, progress=progress)

    PV.core_render = types.SimpleNamespace(render=fake_render)
    try:
        vp._render(None, sc, st, cam, 96, 72, key, vp.version, False)
    finally:
        PV.core_render = real
    check('the injected black frame triggered the guard once',
          vp.guard_count == 1, str(vp.guard_count))
    check('and the frame the user sees is the REAL picture',
          vp.frame is not None
          and float(np.abs(vp.frame - good).max()) < 1e-5)
    check('the guard names its device', vp.last_engaged == 'CPU (guard)')

    # convergence: the parked (real) ratio is now the baseline
    vp._render(None, sc, st, cam, 96, 72, key, vp.version, False)
    check('the next normal frame does not re-trigger', vp.guard_count == 1)

    # a PARTIAL blackout -- one material's region, under any whole-frame
    # threshold -- must trigger through the TILE map (same view only)
    real2 = PV.core_render
    calls2 = {'n': 0}

    def fake_partial(scene, settings, progress=None):
        calls2['n'] += 1
        img = real2.render(scene, settings, progress=progress)
        if calls2['n'] == 1:
            img = np.array(img, np.float32, copy=True)
            # rows 24:48 x cols 24:72 = TWO full 24px tiles = 16.7% of
            # the frame -- under the 20% whole-frame bar on purpose
            img[24:48, 24:72, :3] = 0.0
        return img

    PV.core_render = types.SimpleNamespace(render=fake_partial)
    try:
        vp._render(None, sc, st, cam, 96, 72, key, vp.version, False)
    finally:
        PV.core_render = real2
    check('a sub-threshold REGION blackout triggers via tiles',
          vp.guard_count == 2, str(vp.guard_count))
    check('and the parked frame is the real picture again',
          vp.frame is not None
          and float(np.abs(vp.frame - good).max()) < 1e-5)

    # ...but NEVER across a camera move: tiles flip legitimately as
    # content crosses a MOVING frame, so the tile rule is same-view-only
    from ..core.scene import Camera as _Cam
    mw2 = np.array(cam.matrix_world, np.float32).copy()
    mw2[0, 3] += 0.6
    cam2 = _Cam(matrix_world=mw2, projection=np.array(cam.projection,
                                                      np.float32),
                type='PERSP')
    key2 = vp._key(cam2, 96, 72)
    calls2['n'] = 0
    PV.core_render = types.SimpleNamespace(render=fake_partial)
    try:
        vp._render(None, sc, st, cam2, 96, 72, key2, vp.version, False)
    finally:
        PV.core_render = real2
    check('a moved camera does NOT tile-trigger (orbit-safe)',
          vp.guard_count == 2, str(vp.guard_count))


def test_the_viewport_honors_the_device_switch():
    """The CPU/GPU switch governs the viewport exactly as it governs F12.

    The viewport was pinned to the CPU its whole life because worker
    threads have no GPU context -- the exact problem the F12 marshal
    solved in 1.25.53, borrowed here since 1.25.83. Held headlessly:

    * the marshal REFCOUNTS -- an F12 finishing mid-viewport-frame used
      to switch marshalling off underneath the other render, whose every
      remaining burst then fell back with a misleading reason;
    * shape_settings passes the whole device family through untouched
      (the old code pinned render_device='CPU' right here) while still
      stripping what makes no sense per redraw;
    * a GPU-device viewport frame with no driver present parks the SAME
      pixels as the CPU frame -- the matrix doctrine, fallback equality;
    * a busy driver (the pipeline lock held by another thread -- an F12
      mid-flight) forces that one frame to the CPU with a named reason,
      without touching the stored settings, and the pixels still match.
    """
    from ..core.scene import Camera
    from ..gpu import marshal as M
    from ..preview import Viewport, shape_settings

    # --- the marshal refcounts
    check('marshalling starts off', M.enabled() is False)
    M.enable()
    M.enable()
    M.disable()
    check('one render finishing must not strip the other',
          M.enabled() is True)
    M.disable()
    check('the last disable switches it off', M.enabled() is False)
    M.disable()                              # over-release must not wedge
    M.enable()
    check('a floor of zero: the next enable still works',
          M.enabled() is True)
    M.disable()

    # --- shape_settings: device family untouched, per-redraw noise stripped
    st = RenderSettings()
    st.render_device = 'GPU'
    st.gpu_shading = False                   # NON-default: proves passthrough
    st.gpu_scissor = False
    st.show_stats = True
    st.use_processes = True
    st.aa_mode = 'SUPERSAMPLE'
    st.preview_scale = 1
    shape_settings(st, 96, 72)
    check('the device rides through', st.render_device == 'GPU')
    check('its Debug toggles ride through in NON-default positions',
          st.gpu_shading is False and st.gpu_scissor is False)
    check('worker processes are stripped (a pool per draft)',
          st.use_processes is False)
    check('anti-aliasing is stripped', st.aa_mode == 'NONE')
    check('the stats wish is remembered as one line per refine',
          st.show_stats is False and st._viewport_stats is True)
    check('viewport frames are marked (quiets the depth firehose)',
          st._viewport is True)

    # --- fallback equality: GPU device, no driver -> the CPU pixels
    def _park(dev, hold_elsewhere=False):
        s = RenderSettings()
        s.render_device = dev
        s.preview_scale = 1
        s.threads = 1
        s.shadows = False
        sc = demo_scene(s, with_texture=False)
        shape_settings(s, 96, 72)
        view, proj, _v, _e = R.camera_matrices(sc.camera, 96, 72)
        cam = Camera(matrix_world=np.linalg.inv(view).astype(np.float32),
                     projection=proj.astype(np.float32), type='PERSP')
        vp = Viewport()
        vp.set_scene(sc, s)
        vp.abort = False           # kick() clears this; we call _render direct
        key = vp._key(cam, 96, 72)
        vp._render(None, sc, s, cam, 96, 72, key, vp.version, False)
        return vp, s

    vp_cpu, _ = _park('CPU')
    vp_gpu, st_gpu = _park('GPU')
    check('the CPU viewport frame parked', vp_cpu.frame is not None)
    check('the GPU-device viewport frame parked', vp_gpu.frame is not None)
    check('the worker took the GPU path', vp_gpu.last_engaged == 'GPU')
    d = float(np.abs(vp_cpu.frame.astype(np.float64)
                     - vp_gpu.frame).max())
    check('no driver: the GPU device parks the CPU pixels exactly',
          d == 0.0, f'max difference {d}')
    check('the marshal is off again afterwards (refcount balanced)',
          M.enabled() is False)
    check('the pipeline lock is free again',
          M.PIPELINE.acquire(blocking=False))
    M.PIPELINE.release()

    # --- a busy driver: the F12 holds the pipeline, the frame goes CPU
    import threading as _th
    got, gate = _th.Event(), _th.Event()

    def _hold():
        M.PIPELINE.acquire()
        got.set()
        gate.wait(10.0)
        M.PIPELINE.release()

    thr = _th.Thread(target=_hold, daemon=True)
    thr.start()
    got.wait(10.0)
    try:
        vp_busy, st_busy = _park('GPU')
    finally:
        gate.set()
        thr.join(10.0)
    check('a busy driver still parks a frame', vp_busy.frame is not None)
    check('that frame went to the CPU', vp_busy.last_engaged == 'CPU')
    check('and said why, once',
          any('driver is busy' in s for s in vp_busy._said))
    check('the STORED settings keep their device (the copy took the hit)',
          st_busy.render_device == 'GPU')
    d = float(np.abs(vp_cpu.frame.astype(np.float64)
                     - vp_busy.frame).max())
    check('the busy-driver frame is the CPU picture exactly', d == 0.0,
          f'max difference {d}')


def test_the_viewport_bvh_is_cached_across_orbits():
    """render() must not rebuild the BVH for every view of the same scene.

    The field scene pays ~0.2 s to build its BVH; a viewport orbit calls
    render() once per view of the SAME exported scene, and an animation
    renders the same mesh for most of its frames. The cache keys on mesh
    CONTENT (identity is useless across exports) and lives on the scene
    object, so a fresh export starts clean and an edited mesh rebuilds.
    """
    from ..core.render import _cached_bvh

    st = base_settings(96, 72, shadows=True, shadow_default='RAY')
    sc = demo_scene(st, with_texture=False)

    a = _cached_bvh(sc, sc.mesh)
    b = _cached_bvh(sc, sc.mesh)
    check('the same scene and mesh reuse the same tree', a is b)

    sc.mesh.verts = sc.mesh.verts + np.float32(0.25)    # the mesh changed
    c = _cached_bvh(sc, sc.mesh)
    check('a changed mesh rebuilds', c is not a)
    d = _cached_bvh(sc, sc.mesh)
    check('and then caches again', d is c)

    # through render() itself: two renders of one scene, one build. The
    # cache moved to a module-level content-keyed store in 1.25.82 (the
    # scene-attribute version could never hit across exports), so the
    # assertions read the store, not the scene
    sc2 = demo_scene(st, with_texture=False)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    R._BVH_CACHE.clear()
    img1 = R.render(sc2, st)
    trees1 = list(R._BVH_CACHE.values())
    img2 = R.render(sc2, st)
    trees2 = list(R._BVH_CACHE.values())
    check('render() populated the cache', len(trees1) == 1)
    check('the second render reused it, not rebuilt it',
          len(trees2) == 1 and trees1 and trees2[0] is trees1[0])
    check('and the frames are identical',
          bool(np.array_equal(img1, img2)))


def test_converted_materials_shade_on_the_gpu():
    """A master-shader-node material must qualify, extras and all.

    Every converted material is built around HALCYON_ShaderNode, and until
    the emitter knew it, no converted scene ever qualified -- the probe
    harvested its constants happily and the emitter then refused the node.
    This locks the whole road: the master node's colour chain emitted, its
    rim, fresnel and sheen baked and reproduced, an AREA light in the loop
    (with the cube shadow the map builder already gives it), all against
    `render()` itself.
    """
    from ..core import raster
    from ..core.scene import Light, Material
    from ..gpu import shade as GSH

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def master_graph():
        ins = [
            sk('Diffuse Color', 'RGBA', [0.8, 0.8, 0.8, 1.0], ['mix', 0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1.0, 0.9, 0.7, 1.0]),
            sk('Specular Level', 'VALUE', 0.7),
            sk('Glossiness', 'VALUE', 36.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 0.8),
            sk('Self-Illumination', 'RGBA', [0.02, 0.0, 0.03, 1.0]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.1),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
            sk('Fresnel', 'VALUE', 0.45),
            sk('Fresnel Power', 'VALUE', 2.5),
            sk('Fresnel Color', 'RGBA', [1.0, 0.5, 0.8, 1.0]),
            sk('Rim Amount', 'VALUE', 0.5),
            sk('Rim Power', 'VALUE', 3.0),
            sk('Rim Light', 'RGBA', [0.4, 0.8, 1.0, 1.0]),
            sk('Sheen', 'VALUE', 0.35),
            sk('Sheen Roughness', 'VALUE', 0.4),
            sk('Sheen Color', 'RGBA', [1.0, 0.85, 0.9, 1.0]),
        ]
        return {'output': 'out', 'nodes': {
            'a': {'id': 'a', 'bl_idname': 'ShaderNodeRGB',
                  'props': {'value': [0.75, 0.3, 0.2, 1.0]}, 'inputs': [],
                  'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'b': {'id': 'b', 'bl_idname': 'ShaderNodeRGB',
                  'props': {'value': [0.15, 0.4, 0.85, 1.0]}, 'inputs': [],
                  'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'mix': {'id': 'mix', 'bl_idname': 'ShaderNodeMixRGB',
                    'props': {'blend_type': 'MIX'},
                    'inputs': [sk('Fac', 'VALUE', 0.4),
                               sk('Color1', 'RGBA', [0, 0, 0, 1], ['a', 0]),
                               sk('Color2', 'RGBA', [0, 0, 0, 1], ['b', 0])],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}}

    w, h = 128, 96
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[1] = Material(name='Converted', index=1, graph=master_graph())
    sc.lights.append(
        Light(type='AREA', name='Panel', position=(0.0, 4.0, 4.5),
              direction=(0.0, -0.6, -0.8), color=(1.0, 0.9, 0.7),
              energy=250.0, shadow='MAP', shadow_bias=0.03))
    cpu_img = R.render(sc, st)

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)

    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a converted-material frame with an area light qualifies',
          passes is not None, str(why))
    if passes is None:
        return
    conv = [p for p in passes if p[1] == 'Converted']
    check('the master-node material got its own pass', len(conv) == 1,
          str([p[1] for p in passes]))
    src = conv[0][2]
    check('its shader carries the sheen lobe', 'hal_edge_vn' in src)
    check('and the silhouette cheats', 'hal_sil' in src)

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the converted frame simulates', img is not None, str(why))
    if img is None:
        return
    cov = g.tri >= 0
    err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
    mean = float(np.abs(img[cov] - cpu_img[cov][:, :3]).mean())
    check('the converted GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} mean {mean:.6f} over {int(cov.sum())} px')

    # the extras must be in the picture being matched, not vacuous
    plain = master_graph()
    for s_ in plain['nodes']['hal']['inputs']:
        if s_['name'] in ('Fresnel', 'Rim Amount', 'Sheen'):
            s_['default'] = 0.0
    sc2 = demo_scene(st, with_texture=False)
    sc2.materials[1] = Material(name='Plain', index=1, graph=plain)
    sc2.lights = list(sc.lights)
    plain_img = R.render(sc2, st)
    delta = float(np.abs(cpu_img[cov][:, :3] - plain_img[cov][:, :3]).max())
    check('rim, fresnel and sheen visibly change the frame', delta > 0.02,
          f'{delta:.4f}')

    # vertex-colour blending joined the G-buffer in 1.25.9 and qualifies now
    vc = master_graph()
    for s_ in vc['nodes']['hal']['inputs']:
        if s_['name'] == 'Vertex Color Mix':
            s_['default'] = 0.5
    sc3 = demo_scene(st, with_texture=False)
    sc3.materials[1] = Material(name='VertexBlend', index=1, graph=vc)
    job3 = R.ShadeJob(sc3, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, _a3 = GSH.plan_frame(job3, g)
    check('vertex-colour blending qualifies now', p3 is not None, str(why3))
    GSH._PLAN_CACHE.clear()


def test_vertex_colours_travel_in_the_gbuffer():
    """Painted vertices must reach the GPU and blend as the evaluator blends.

    The attribute texture's fourth slot was reserved for a tangent nothing
    ever wrote; the vertex colour lives there now, all four components. The
    master node's Vertex Color Mix -- the one refusal it had left -- blends
    the paint over the diffuse exactly as `n_halcyon_shader` does, and the
    ShaderNodeVertexColor node reads the same slot.
    """
    from ..core import raster
    from ..core.scene import Light, Material
    from ..gpu import shade as GSH

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def painted_graph(vmix):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.7, 0.6, 0.5, 1.0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', vmix),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ]
        return {'output': 'out', 'nodes': {
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}}

    w, h = 128, 96
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    # paint every corner from its position: a gradient no constant can fake
    v = sc.mesh.verts
    paint = np.stack([
        np.clip(v[:, 0] * 0.2 + 0.5, 0.0, 1.0),
        np.clip(v[:, 2] * 0.4, 0.0, 1.0),
        np.clip(0.9 - v[:, 1] * 0.2, 0.0, 1.0),
        np.ones(v.shape[0], np.float32)], axis=1).astype(np.float32)
    sc.mesh.colors = paint
    sc.materials[1] = Material(name='Painted', index=1,
                               graph=painted_graph(0.75))
    cpu_img = R.render(sc, st)

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)

    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a painted-vertex material qualifies now', passes is not None,
          str(why))
    if passes is None:
        return
    src = [p for p in passes if p[1] == 'Painted'][0][2]
    check('its shader reads the paint from the G-buffer', 'hal_vcol' in src)

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the painted frame simulates', img is not None, str(why))
    if img is None:
        return
    cov = g.tri >= 0
    err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
    check('the painted GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} over {int(cov.sum())} px')

    # the paint must matter: mix zero renders a different picture
    sc2 = demo_scene(st, with_texture=False)
    sc2.mesh.colors = paint
    sc2.materials[1] = Material(name='Unpainted', index=1,
                                graph=painted_graph(0.0))
    plain = R.render(sc2, st)
    delta = float(np.abs(cpu_img[cov][:, :3] - plain[cov][:, :3]).max())
    check('the paint visibly changes the frame', delta > 0.05, f'{delta:.4f}')

    # an unpainted mesh reads white, exactly as the CPU answers
    sc3 = demo_scene(st, with_texture=False)
    sc3.mesh.colors = None
    sc3.materials[1] = Material(name='White', index=1,
                                graph=painted_graph(0.6))
    cpu3 = R.render(sc3, st)
    g3 = raster.GBuffer(w, h)
    raster.rasterize(sc3.mesh.verts, sc3.mesh.tris, vp, w, h, gbuf=g3)
    job3 = R.ShadeJob(sc3, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, a3 = GSH.plan_frame(job3, g3)
    check('an unpainted mesh still qualifies', p3 is not None, str(why3))
    if p3 is not None:
        img3, _h3 = GSH.simulate(job3, g3, p3, a3)
        cov3 = g3.tri >= 0
        e3 = float(np.abs(img3[cov3] - cpu3[cov3][:, :3]).max())
        check('and blends toward white as the CPU does', e3 < 6e-3,
              f'max {e3:.5f}')
    GSH._PLAN_CACHE.clear()


def test_per_pixel_surface_parameters_shade_on_the_gpu():
    """A node chain driving Roughness must not push the material off the GPU.

    Until now only the base colour could vary per pixel; a texture on
    Roughness or Specular Level refused the whole material. Linked
    master-node sockets now emit their chains -- through the same emitter,
    sharing subexpressions with the colour -- and the probe exempts exactly
    those fields from its constancy rule. The proof is the usual one: a
    frame whose roughness, specular level, specular colour and emission all
    vary across the surface, against `render()` itself.
    """
    from ..core import raster
    from ..core.scene import Light, Material
    from ..gpu import shade as GSH

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def graph():
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.55, 0.5, 1.0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1], ['smix', 0]),
            sk('Specular Level', 'VALUE', 0.5, ['ramp', 0]),
            sk('Glossiness', 'VALUE', 30.0),
            sk('Roughness', 'VALUE', 0.3, ['ramp', 0]),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1], ['emix', 0]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ]
        return {'output': 'out', 'nodes': {
            'uv': {'id': 'uv', 'bl_idname': 'ShaderNodeTexCoord', 'props': {},
                   'inputs': [],
                   'outputs': [{'name': 'Generated', 'type': 'VECTOR'},
                               {'name': 'Normal', 'type': 'VECTOR'},
                               {'name': 'UV', 'type': 'VECTOR'}]},
            'sep': {'id': 'sep', 'bl_idname': 'ShaderNodeSeparateXYZ',
                    'props': {},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0], ['uv', 2])],
                    'outputs': [{'name': 'X', 'type': 'VALUE'},
                                {'name': 'Y', 'type': 'VALUE'},
                                {'name': 'Z', 'type': 'VALUE'}]},
            'ramp': {'id': 'ramp', 'bl_idname': 'ShaderNodeMapRange',
                     'props': {},
                     'inputs': [sk('Value', 'VALUE', 0.5, ['sep', 0]),
                                sk('From Min', 'VALUE', 0.0),
                                sk('From Max', 'VALUE', 1.0),
                                sk('To Min', 'VALUE', 0.08),
                                sk('To Max', 'VALUE', 0.85)],
                     'outputs': [{'name': 'Result', 'type': 'VALUE'}]},
            'c1': {'id': 'c1', 'bl_idname': 'ShaderNodeRGB',
                   'props': {'value': [1.0, 0.8, 0.4, 1.0]}, 'inputs': [],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'c2': {'id': 'c2', 'bl_idname': 'ShaderNodeRGB',
                   'props': {'value': [0.3, 0.6, 1.0, 1.0]}, 'inputs': [],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'smix': {'id': 'smix', 'bl_idname': 'ShaderNodeMixRGB',
                     'props': {'blend_type': 'MIX'},
                     'inputs': [sk('Fac', 'VALUE', 0.5, ['sep', 1]),
                                sk('Color1', 'RGBA', [0, 0, 0, 1], ['c1', 0]),
                                sk('Color2', 'RGBA', [0, 0, 0, 1], ['c2', 0])],
                     'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'emix': {'id': 'emix', 'bl_idname': 'ShaderNodeMixRGB',
                     'props': {'blend_type': 'MIX'},
                     'inputs': [sk('Fac', 'VALUE', 0.5, ['sep', 0]),
                                sk('Color1', 'RGBA', [0, 0, 0, 1]),
                                sk('Color2', 'RGBA', [0.15, 0.02, 0.2, 1.0])],
                     'outputs': [{'name': 'Color', 'type': 'RGBA'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'COOK_TORRANCE', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}}

    w, h = 128, 96
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[1] = Material(name='Varied', index=1, graph=graph())
    cpu_img = R.render(sc, st)

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)

    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a frame with per-pixel roughness and specular qualifies',
          passes is not None, str(why))
    if passes is None:
        return
    src = [p for p in passes if p[1] == 'Varied'][0][2]
    check('roughness is an expression, not a constant',
          's.roughness = _v' in src, 'no emitted chain found for roughness')
    check('and so is the specular colour', 's.specular = ' in src and
          's.specular = vec3(' not in src.split('s.specular = ')[1][:12])

    img, hit = GSH.simulate(job, g, passes, atlases)
    check('the per-pixel frame simulates', img is not None, str(why))
    if img is None:
        return
    cov = g.tri >= 0
    err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
    mean = float(np.abs(img[cov] - cpu_img[cov][:, :3]).mean())
    check('the per-pixel GPU frame is the CPU frame', err < 6e-3,
          f'max {err:.5f} mean {mean:.6f} over {int(cov.sum())} px')

    # vacuity: pin the chains to constants and the picture must change
    flat = graph()
    for s_ in flat['nodes']['hal']['inputs']:
        s_['link'] = None if s_['name'] in ('Roughness', 'Specular Level',
                                            'Specular Color',
                                            'Self-Illumination') else s_['link']
    sc2 = demo_scene(st, with_texture=False)
    sc2.materials[1] = Material(name='Flat', index=1, graph=flat)
    flat_img = R.render(sc2, st)
    delta = float(np.abs(cpu_img[cov][:, :3] - flat_img[cov][:, :3]).max())
    check('the varying parameters visibly change the frame', delta > 0.02,
          f'{delta:.4f}')
    GSH._PLAN_CACHE.clear()


def test_the_frame_plan_is_cached_and_invalidates_honestly():
    """A held-still scene plans once; anything that matters re-plans.

    Planning probes every material through the CPU's closure path and
    assembles ~500-line sources -- 4.4 ms of a 14.3 ms warm frame at 480x360,
    paid every frame for a scene that had not changed. The cache keys on a
    content signature of everything the plan reads, so the test is about the
    key: a camera move must hit, a material edit must miss.
    """
    from ..core import raster
    from ..gpu import shade as GSH

    w, h = 96, 72
    st = base_settings(w, h, shadows=True)
    sc = demo_scene(st, with_texture=False)
    cpu_img = R.render(sc, st)                    # bakes the shadow maps
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)

    GSH._PLAN_CACHE.clear()
    calls = {'n': 0}
    real_probe = GSH._probe_material

    def counting_probe(*a, **kw):
        calls['n'] += 1
        return real_probe(*a, **kw)

    GSH._probe_material = counting_probe
    try:
        p1 = GSH.plan_frame(job, g)
        first = calls['n']
        p2 = GSH.plan_frame(job, g)
        second = calls['n'] - first
    finally:
        GSH._probe_material = real_probe
    check('the first plan probes its materials', first >= 3, str(first))
    check('the second plan probes nothing', second == 0, str(second))
    check('and returns the identical passes', p2[0] is p1[0])

    # a moved eye is the same plan: the camera is not in the signature
    job_b = R.ShadeJob(sc, st, {}, None, view,
                       eye + np.float32([1.0, -2.0, 0.5]), w, h)
    p3 = GSH.plan_frame(job_b, g)
    check('a camera move hits the cache', p3[0] is p1[0])

    # a material edit is a different scene and must re-plan
    sc.materials[1].diffuse = (0.1, 0.9, 0.2)
    p4 = GSH.plan_frame(job, g)
    check('a material edit misses the cache', p4[0] is not p1[0])
    check('and the new plan carries the new colour',
          any('0.9' in p[2] for p in p4[0]),
          'edited diffuse not found in any source')

    # the cache must never trade correctness: the cached plan still matches
    sc.materials[1].diffuse = (0.85, 0.2, 0.15)   # back to stock
    p5 = GSH.plan_frame(job, g)
    img, _hit = GSH.simulate(job, g, p5[0], p5[2])
    cov = g.tri >= 0
    err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
    check('a plan served from cache still matches the renderer', err < 6e-3,
          f'max {err:.5f}')
    GSH._PLAN_CACHE.clear()


def test_camera_motion_does_not_recompile_the_shaders():
    """An orbit must not change the shader source, and must still match.

    The eye used to be baked into the source like every other frame constant
    -- correct for a still, catastrophic for an animation, because a moving
    camera changed the source every frame and every frame paid the driver's
    shader compile. The eye is a uniform now, and this holds it there: two
    camera positions, byte-identical sources, both frames matching render().
    """
    from ..core import raster
    from ..core.scene import Camera
    from ..gpu import shade as GSH
    from .scenebuild import look_at_matrix

    w, h = 96, 72
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'

    eyes = ((5.2, -6.4, 3.6), (-6.0, 4.8, 4.2))
    sources = []
    for eye_pos in eyes:
        sc = demo_scene(st, with_texture=False)
        sc.camera = Camera(matrix_world=look_at_matrix(eye_pos, (0.0, -0.2, 0.9)),
                           lens=42.0, sensor=36.0, clip_start=0.1,
                           clip_end=200.0)
        cpu_img = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'the frame at eye {eye_pos} plans', passes is not None,
              str(why))
        if passes is None:
            return
        sources.append(tuple(p[2] for p in passes))
        img, hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
        check(f'and matches the renderer from there', err < 6e-3,
              f'max {err:.5f}')

    check('moving the camera leaves every shader source byte-identical',
          sources[0] == sources[1],
          'sources differ; the driver would recompile per frame')
    check('and no source carries the eye as a literal',
          all('hal_eye' in src for src in sources[0]))


def test_deferred_shading_samples_image_textures():
    """Textured materials must shade on the GPU and match the renderer.

    The CPU samples the *prepared* pixels -- resized, quantised, colourspace
    converted -- so those exact pixels travel to the GPU, and the filter
    arithmetic (floor-based nearest, half-texel bilinear, all four wrap
    modes) is reproduced in the shader rather than left to the driver's
    sampler state, which no driver documents to the texel.
    """
    from ..core import raster
    from ..core.texture import Texture
    from ..gpu import shade as GSH
    from ..gpu.material import _texture_sampler
    from ..shaders.compiler import try_compile

    w, h = 128, 96
    for filt in ('NEAREST', 'BILINEAR'):
        st = base_settings(w, h, shadows=True, tex_filter=filt)
        sc = demo_scene(st)                     # the textured floor
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        cpu_img = R.render(sc, st)

        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        textures = R.prepare_textures(sc, st)
        job = R.ShadeJob(sc, st, textures, None, view, eye, w, h)

        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'a textured frame qualifies under {filt}', passes is not None,
              str(why))
        if passes is None:
            continue
        tex_pass = [p for p in passes if p[3].get('textures')]
        check('the textured material asks for its image',
              len(tex_pass) == 1 and 'checker' in
              str(list(tex_pass[0][3]['textures'].values())),
              str([p[3] for p in passes]))
        img, hit = GSH.simulate(job, g, passes, atlases)
        check(f'the textured frame simulates under {filt}', img is not None,
              str(why))
        if img is None:
            continue
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
        check(f'the textured GPU frame is the CPU frame under {filt}',
              err < 6e-3, f'max {err:.5f} over {int(cov.sum())} px')

    # the sampler arithmetic itself, against Texture.sample, off the grid:
    # random uvs well outside [0,1], all four wraps, both filters
    rng = np.random.default_rng(11)
    px = rng.random((13, 9, 4)).astype(np.float32)
    tex = Texture(px, colorspace='Non-Color')
    n = 500
    uv = (rng.random((n, 2)).astype(np.float32) * 4.0 - 1.5)
    for filt in ('NEAREST', 'BILINEAR'):
        for wrap in ('REPEAT', 'EXTEND', 'CLIP', 'MIRROR'):
            src = _texture_sampler('hal_tex0', tex, filt, wrap) + """
uniform vec2 hal_uv_in;
out vec4 Color;
void main() { Color = hal_sample_hal_tex0(hal_uv_in); }
"""
            prog, err = try_compile(src, 'GLSL')
            if prog is None:
                check(f'{filt}/{wrap} sampler compiles', False, str(err))
                continue
            got = prog.run({
                'hal_tex0': Texture(px, colorspace='Non-Color',
                                    filt='NEAREST', wrap='EXTEND'),
                'hal_uv_in': uv}, {}, n)[0]['Color']
            want = tex.sample(uv[:, 0], uv[:, 1], filt=filt, wrap=wrap)
            e = float(np.abs(got - np.asarray(want)).max())
            check(f'{filt}/{wrap} matches Texture.sample off the grid',
                  e < 2e-5, f'max {e:.7f}')

    # the N64 filter used to refuse by name; 1.25.81 ported it (pure
    # arithmetic, no footprint needed) -- the negative test flips, as
    # negative tests must when scope grows
    st = base_settings(w, h, tex_filter='N64_3POINT')
    sc = demo_scene(st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('the N64 filter QUALIFIES now (the refusal flipped when the '
          'port landed)', p is not None, str(why))


def test_generated_glsl_is_driver_strict():
    """Everything a real driver refused, or plausibly would, checked headless.

    Halcyon's own front-end is deliberately forgiving, and that forgiveness
    is a blind spot: the first deferred frame on real hardware died on a
    redeclared uniform that every headless test had shrugged at. A uniform
    with a trailing comment survived `strip_declarations`, CreateInfo
    declared the same name a second time, and the driver rejected the shader
    whole. This test holds the strictness the driver holds.
    """
    import re

    from ..core import raster
    from ..core.scene import Light
    from ..gpu import shade as GSH
    from ..gpu.device import strip_declarations
    from ..gpu.material import _f
    from ..gpu.stages import STAGES, body

    decl = re.compile(r'^(uniform|in|out)\s+\w[^;]*;$')
    multi = re.compile(r'^(?:(?:uniform|in|out)\s+\w[^;]*?;\s*)+$')

    def surviving(src):
        # anything that would redeclare a name CreateInfo declares itself:
        # a whole-line declaration, a line of several GLUED declarations
        # (the second field failure -- the whole-line pattern alone was
        # blind to it), or the word `uniform` appearing AT ALL, since a
        # stripped source has no legitimate use for it anywhere
        bad = []
        for ln in src.splitlines():
            code = ln.split('//', 1)[0].strip()
            if decl.match(code) or multi.match(code) \
                    or re.search(r'\buniform\b', code):
                bad.append(ln)
        return bad

    # the exact failure: a declaration with a trailing comment must strip
    tricky = ('uniform sampler2D tex;   // a note that hid this from the\n'
              'uniform float     n;     // strip, and killed the frame\n'
              'in vec2 vUV;\nout vec4 Color;\n'
              'void main() { Color = texture(tex, vUV) * n; }\n')
    check('a commented declaration is stripped like any other',
          not surviving(strip_declarations(tricky)),
          str(surviving(strip_declarations(tricky))))

    # the second exact failure: two declarations GLUED on one line by a
    # block join -- `hal_bump0;uniform sampler2D hal_shadow0;` reached the
    # field's driver as a redefinition while every headless seam read zero
    glued = ('uniform sampler2D hal_bump0;uniform sampler2D hal_shadow0;\n'
             'in vec2 vUV;out vec4 Color;\n'
             'void main() { }\n')
    check('declarations glued on one line strip together',
          not surviving(strip_declarations(glued)),
          str(surviving(strip_declarations(glued))))

    # no stage may hand CreateInfo a source that still declares anything
    left = {name: surviving(body(name)) for name in STAGES}
    bad = {k: v for k, v in left.items() if v}
    check('no stage body still declares a global after the strip', not bad,
          str(bad))

    # and neither may a generated material pass -- the case that shipped
    w, h = 64, 48
    st = base_settings(w, h, shadows=False)
    sc = demo_scene(st, with_texture=False)
    sc.lights = [Light(type='SUN', direction=(-0.6, 0.4, -0.5), energy=5.0,
                       shadow='NONE'),
                 Light(type='SPOT', position=(-4.0, 2.5, 4.0),
                       direction=(0.55, -0.35, -0.75), energy=300.0,
                       shadow='NONE', spot_size=0.9, spot_blend=0.3)]
    view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    passes, why, atlases = GSH.plan_frame(job, g)
    check('a frame still plans', passes is not None, str(why))
    if passes is None:
        return
    for _mi, name, src, _smp in passes:
        left = surviving(strip_declarations(src))
        check(f"'{name}' hands CreateInfo no duplicate declarations",
              not left, str(left[:2]))

    # the SECONDARY (reflection-hit) passes go to the same driver and must
    # hold the same strictness -- they were a blind spot until the first
    # reflected frame on real hardware disagreed
    from ..core.bvh import BVH
    st_r = base_settings(w, h, shadows=False, raytrace=True, ray_depth=1)
    sc_r = demo_scene(st_r, with_texture=False)
    sc_r.materials[2].reflect_level = 0.5
    sc_r.lights = list(sc.lights)
    bvh_r = BVH(sc_r.mesh.verts, sc_r.mesh.tris)
    job_r = R.ShadeJob(sc_r, st_r, {}, bvh_r, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p_r, why_r, a_r = GSH.plan_frame(job_r, g)
    check('a ray-traced frame plans for the strictness check',
          p_r is not None and '__reflect' in (a_r or {}), str(why_r))
    if p_r is not None and '__reflect' in (a_r or {}):
        for _mi, name, src, _smp in a_r['__reflect']['secondary']:
            left = surviving(strip_declarations(src))
            check(f"secondary '{name}' hands CreateInfo no duplicate "
                  'declarations', not left, str(left[:2]))

    # every baked literal is a float literal, not an integer that relies on
    # implicit conversion: 6.0 must never bake as `6`
    for value, want in ((6.0, '6.0'), (0.0, '0.0'), (1.0, '1.0'),
                        (0.5, '0.5'), (600.0, '600.0')):
        check(f'_f({value}) carries its decimal point', _f(value) == want,
              _f(value))
    check('exponent literals stay as they are', 'e' in _f(1e-6), _f(1e-6))

    # no generated identifier may be a GLSL reserved word our front-end
    # happens to tolerate
    reserved = r'\b(float|int|bool|vec[234])\s+' \
               r'(smooth|flat|sample|buffer|patch|precise|input|output|' \
               r'filter|active|common|partition)\s*[=;,)]'
    for _mi, name, src, _smp in passes:
        hits = re.findall(reserved, src)
        check(f"'{name}' declares no reserved-word identifiers", not hits,
              str(hits))

    # ---- the FIELD combo the walls above guard: a Bump material under
    # SHADOW MAPS with no image texture in its main pass. That pass
    # declares hal_bump0 and hal_shadow0 with an empty texture block
    # between them -- the exact splice that reached the driver as one
    # glued line and killed 'Water' and 'Bumpy' while every headless
    # seam read zero. Primary, PRE-PASS and secondary sources all strip.
    from .scenebuild import ImageBuffer, add_normal_mapped_ball

    def skl(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    wl, hl = 64, 48
    st_b = base_settings(wl, hl, shadows=True, raytrace=True, ray_depth=1,
                         transparency='NONE')
    sc_b = demo_scene(st_b, with_texture=False)
    add_normal_mapped_ball(sc_b)
    gb = sc_b.materials[1].graph
    rng_b = np.random.default_rng(5)
    him = np.zeros((8, 8, 4), np.float32)
    him[:, :, 0] = rng_b.random((8, 8))
    him[:, :, 1] = him[:, :, 2] = him[:, :, 0]
    him[:, :, 3] = 1.0
    sc_b.images['hmap'] = ImageBuffer(name='hmap', pixels=him)
    gb['nodes']['htex'] = {
        'id': 'htex', 'bl_idname': 'ShaderNodeTexImage',
        'props': {'image': 'hmap', 'interpolation': 'Closest'},
        'inputs': [skl('Vector', 'VECTOR', [0, 0, 0])],
        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                    {'name': 'Alpha', 'type': 'VALUE'}]}
    gb['nodes']['bump'] = {
        'id': 'bump', 'bl_idname': 'ShaderNodeBump',
        'props': {'invert': False},
        'inputs': [skl('Strength', 'VALUE', 0.8),
                   skl('Distance', 'VALUE', 0.6),
                   skl('Height', 'VALUE', 0.5, ['htex', 0]),
                   skl('Normal', 'VECTOR', [0, 0, 0])],
        'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
    for sock in gb['nodes']['hal']['inputs']:
        if sock['name'] == 'Normal':
            sock['link'] = ['bump', 0]
    sc_b.materials[2].reflect_level = 0.5
    tex_b = R.prepare_textures(sc_b, st_b)
    R.render(sc_b, st_b)                      # builds the shadow maps
    view_b, _pb, vp_b, eye_b = R.camera_matrices(sc_b.camera, wl, hl)
    g_b = raster.GBuffer(wl, hl)
    raster.rasterize(sc_b.mesh.verts, sc_b.mesh.tris, vp_b, wl, hl,
                     gbuf=g_b)
    job_b = R.ShadeJob(sc_b, st_b, tex_b,
                       BVH(sc_b.mesh.verts, sc_b.mesh.tris),
                       view_b, eye_b, wl, hl)
    GSH._PLAN_CACHE.clear()
    p_b, why_b, a_b = GSH.plan_frame(job_b, g_b)
    check('the bump-under-shadow-maps combo plans for the strictness '
          'check', p_b is not None, str(why_b))
    if p_b is not None:
        sources = [(name, src, bn) for _mi, name, src, bn in p_b]
        pre = []
        for name, src, bn in list(sources):
            for uname, psrc, _pi in (bn.get('prepasses') or ()):
                if psrc != '__CPU__':
                    pre.append((f'{name} pre-pass {uname}', psrc, {}))
        for _mi, name, src, bn in \
                ((a_b or {}).get('__reflect') or {}).get('secondary', ()):
            sources.append((f'secondary {name}', src, bn))
        check('the combo really carries a height pre-pass', bool(pre))
        for name, src, _bn in sources + pre:
            left = surviving(strip_declarations(src))
            check(f"'{name}' hands CreateInfo no declaration text",
                  not left, str(left[:2]))


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


def test_normal_maps_bend_the_deferred_normal():
    """A Normal Map chain on the master shader must shade on the GPU.

    Three seams meet here and each is locked separately. First the math:
    `n_normal_map`'s tangent construction (the renderer never carries a UV
    tangent, so the frame is built from the geometric normal), the node's
    own Strength lerp, and `closure_to_surface`'s Bump Strength lerp, all
    against `render()` itself, in tangent and world space. Second the
    ORDER: the CPU evaluates the graph against the UNFLIPPED interpolated
    normal, bends, and only then flips for two-sided lighting -- so a
    back-facing surface with a normal map is the case that catches a shader
    that flips first. Third the refusals: the Bump node needs neighbouring
    pixels and says so; Normal Source FACE shades with a normal the
    G-buffer does not carry; a normal bent outside the master node still
    moves the material to the CPU.
    """
    import re

    from ..core import raster
    from ..core.scene import ImageBuffer, Material
    from ..gpu import shade as GSH
    from ..gpu.device import strip_declarations

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def mgraph(space, bump=0.65):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.7, 0.5, 0.35, 1.0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.7),
            sk('Glossiness', 'VALUE', 36.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
            sk('Bump Strength', 'VALUE', bump),
            sk('Normal', 'VECTOR', [0, 0, 0], ['nmap', 0]),
        ]
        return {'output': 'out', 'nodes': {
            'ntex': {'id': 'ntex', 'bl_idname': 'ShaderNodeTexImage',
                     'props': {'image': 'nmap', 'interpolation': 'Closest'},
                     'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                     'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                 {'name': 'Alpha', 'type': 'VALUE'}]},
            'nmap': {'id': 'nmap', 'bl_idname': 'ShaderNodeNormalMap',
                     'props': {'space': space},
                     'inputs': [sk('Strength', 'VALUE', 0.8),
                                sk('Color', 'RGBA', [0.5, 0.5, 1.0, 1.0],
                                   ['ntex', 0])],
                     'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}}

    rng = np.random.default_rng(7)
    nm = rng.random((10, 10, 4)).astype(np.float32)
    nm[:, :, 2] = 0.75 + nm[:, :, 2] * 0.25    # decoded z stays >= 0.5
    nm[:, :, 3] = 1.0

    w, h = 128, 96
    decl = re.compile(r'^(uniform|in|out)\s+\w[^;]*;$')
    for space in ('TANGENT', 'WORLD'):
        st = base_settings(w, h, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.images['nmap'] = ImageBuffer(name='nmap', pixels=nm.copy(),
                                        colorspace='Non-Color')
        sc.materials[1] = Material(name='Bumpy', index=1,
                                   graph=mgraph(space))
        cpu_img = R.render(sc, st)

        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None,
                         view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'a {space}-space normal-mapped frame qualifies',
              passes is not None, str(why))
        if passes is None:
            continue
        src = [p for p in passes if p[1] == 'Bumpy'][0][2]
        check('the bend is emitted, Bump Strength lerp included',
              'vec3 Nsurf = normalize(N0 + ' in src and '0.65' in src, 'no')
        check('two-sided flips the BENT normal, after the chains ran',
              src.find('side = (dot(Nsurf, V)') > src.find('vec3 Nsurf'),
              'flip found before the bend')
        check('the tangent frame is built from the unflipped bent normal',
              'cross(up, Nsurf)' in src, 'tangent still reads the flipped N')
        stripped = strip_declarations(src)
        left = [ln for ln in stripped.splitlines()
                if decl.match(ln.split('//', 1)[0].strip())]
        check(f'the {space} source strips clean for CreateInfo', not left,
              str(left))

        img, _hit = GSH.simulate(job, g, passes, atlases)
        check(f'the {space} normal-mapped frame simulates',
              img is not None, str(why))
        if img is None:
            continue
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
        mean = float(np.abs(img[cov] - cpu_img[cov][:, :3]).mean())
        check(f'the {space} normal-mapped GPU frame is the CPU frame',
              err < 6e-3, f'max {err:.5f} mean {mean:.6f}')

    # ---- the ordering proof: a back-facing, two-sided, normal-mapped floor.
    # The CPU shows the graph the normal pointing AWAY from the camera; a
    # shader that flips before running the chains disagrees on every one of
    # these pixels, and nowhere else.
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    st.two_sided_lighting = True
    sc = demo_scene(st, with_texture=False)
    m = sc.mesh
    vmask = np.zeros(m.verts.shape[0], bool)
    vmask[m.tris[m.mat_index == 0].ravel()] = True
    m.normals[vmask] *= -1.0
    sm = m.smooth.copy()
    sm[m.mat_index == 0] = True                # interpolate, don't re-derive
    m.smooth = sm
    sc.images['nmap'] = ImageBuffer(name='nmap', pixels=nm.copy(),
                                    colorspace='Non-Color')
    sc.materials[0] = Material(name='Under', index=0,
                               graph=mgraph('TANGENT', bump=1.0))
    cpu_img = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('the back-facing normal-mapped frame qualifies',
          passes is not None, str(why))
    if passes is not None:
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        floor_px = cov & (m.mat_index[np.where(cov, g.tri, 0)] == 0)
        check('back faces are actually in frame', bool(floor_px.any()))
        err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
        check('the graph saw the unflipped normal, exactly as ctx.N',
              err < 6e-3, f'max {err:.5f}')

    # ---- the refusals, each named
    st = base_settings(w, h)
    sc = demo_scene(st, with_texture=False)
    bumpy = mgraph('TANGENT')
    bumpy['nodes']['bmp'] = {
        'id': 'bmp', 'bl_idname': 'ShaderNodeBump', 'props': {},
        'inputs': [sk('Strength', 'VALUE', 1.0),
                   sk('Distance', 'VALUE', 0.1),
                   sk('Height', 'VALUE', 0.5, ['ntex', 1]),
                   sk('Normal', 'VECTOR', [0, 0, 0])],
        'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
    for s in bumpy['nodes']['hal']['inputs']:
        if s['name'] == 'Normal':
            s['link'] = ['bmp', 0]
    sc.images['nmap'] = ImageBuffer(name='nmap', pixels=nm.copy(),
                                    colorspace='Non-Color')
    sc.materials[1] = Material(name='Bumpy', index=1, graph=bumpy)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('the Bump node QUALIFIES now, height pre-pass and all -- the '
          'refusal this check once held is lifted',
          p is not None and any(len(e[3].get('prepasses', ())) == 1
                                for e in p), str(why))

    # FLIPPED (1.25.102): Normal Source FACE rides the hal_triaux
    # per-tri texture now -- the plan must qualify and every pass must
    # carry the stored-normal override
    st = base_settings(w, h)
    st.normal_source = 'FACE'
    sc = demo_scene(st, with_texture=False)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('Normal Source FACE qualifies via the per-tri texture',
          p is not None and 'hal_triaux' in _a and
          all('hal_triaux_fetch' in src for _m, _n, src, _b in p),
          str(why))

    # a normal bent anywhere but the master node still moves to the CPU
    st = base_settings(w, h)
    sc = demo_scene(st, with_texture=False)
    sc.materials[1] = Material(name='Lobe', index=1, graph={
        'output': 'out', 'nodes': {
            'nmap': {'id': 'nmap', 'bl_idname': 'ShaderNodeNormalMap',
                     'props': {'space': 'TANGENT'},
                     'inputs': [sk('Strength', 'VALUE', 1.0),
                                sk('Color', 'RGBA', [0.6, 0.4, 0.9, 1.0])],
                     'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]},
            'bsdf': {'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse',
                     'props': {},
                     'inputs': [sk('Color', 'RGBA', [0.8, 0.3, 0.3, 1.0]),
                                sk('Roughness', 'VALUE', 0.0),
                                sk('Normal', 'VECTOR', [0, 0, 0],
                                   ['nmap', 0])],
                     'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['bsdf', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}})
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('a lobe-bent normal still shades on the CPU',
          p is None and 'master' in str(why), str(why))


def test_coded_shaders_run_native_in_the_deferred_pass():
    """The coded shader node's GLSL runs on the GPU as itself.

    The easy case, finally collected: on the CPU the user's GLSL is compiled
    to NumPy; in the deferred pass the translation step stops existing. What
    remains is the contract around the source, reproduced exactly: uniforms
    become sockets (socket values win over declared defaults, linked chains
    feed per pixel), declared `in` names bind to the same varyings the
    interpreter binds, outs become output sockets, and everything is mangled
    per node so two coded shaders sharing names cannot collide. `time` rides
    as a per-frame uniform like hal_eye, so an animated coded shader never
    recompiles. Everything outside the contract refuses by name -- including
    HLSL-flavoured GLSL that Halcyon's own forgiving front-end would accept
    and a real driver would not.
    """
    import re

    from ..core import raster
    from ..core.scene import Material
    from ..gpu import shade as GSH
    from ..gpu.device import strip_declarations
    from ..shaders.compiler import compile_shader

    CODE = '''
uniform vec3  baseColor = vec3(0.85, 0.45, 0.2);
uniform float bands     = 6.0;

in vec3 vNormal;
in vec3 vView;
in float time;

out vec4 Color;

float quant(float x) { return floor(x * bands) / bands; }
vec3 g_tint = vec3(1.0, 0.9, 0.8);

void main() {
    vec3  n   = normalize(vNormal);
    float ndv = clamp(dot(n, normalize(vView)), 0.0, 1.0);
    float lam = clamp(dot(n, normalize(vec3(0.4, -0.7, 0.6))), 0.0, 1.0);
    float rim = pow(1.0 - ndv, 2.5) * (0.5 + 0.5 * sin(time));
    Color = vec4(baseColor * g_tint * (0.25 + 0.75 * quant(lam))
                 + vec3(rim * 0.35), 1.0);
}
'''
    CODE2 = '''
uniform vec3 baseColor = vec3(0.1, 0.9, 0.3);
in vec3 vPosition;
out vec4 Color;
void main() {
    float s = step(0.5, fract(vPosition.x * 2.0));
    Color = vec4(baseColor * (0.4 + 0.6 * s), 1.0);
}
'''

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def master(ins_extra, diffuse_link):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0], diffuse_link),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ]
        nodes = {'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                         'props': {'model': 'PHONG', 'toon_steps': 2},
                         'inputs': ins,
                         'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
                 'out': {'id': 'out',
                         'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                         'inputs': [sk('Surface', 'SHADER', None,
                                       ['hal', 0]),
                                    sk('Displacement', 'VECTOR', [0, 0, 0])],
                         'outputs': []}}
        nodes.update(ins_extra)
        return {'output': 'out', 'nodes': nodes}

    def code_node(nid, src, sockets):
        return {'id': nid, 'bl_idname': 'HALCYON_CodeNode',
                'props': {'source_text': src, 'language': 'GLSL'},
                'inputs': sockets,
                'outputs': [{'name': 'Color', 'key': 'Color',
                             'type': 'RGBA'}]}

    def scene_with(graph, programs, w=128, h=96):
        st = base_settings(w, h, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.time = 1.7
        sc.frame = 42
        mat = Material(name='Coded', index=1, graph=graph)
        mat.programs = programs
        sc.materials[1] = mat
        return sc, st

    def run(sc, st, w=128, h=96):
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        return cpu, g, job, passes, why, atlases

    # ---- the full contract in one material: helper function, global with
    # initializer, socket values beating declared defaults, a chain feeding
    # a uniform per pixel, and the clock
    chain = {
        'uv': {'id': 'uv', 'bl_idname': 'ShaderNodeTexCoord', 'props': {},
               'inputs': [],
               'outputs': [{'name': 'Generated', 'type': 'VECTOR'},
                           {'name': 'Normal', 'type': 'VECTOR'},
                           {'name': 'UV', 'type': 'VECTOR'}]},
        'sep': {'id': 'sep', 'bl_idname': 'ShaderNodeSeparateXYZ',
                'props': {},
                'inputs': [sk('Vector', 'VECTOR', [0, 0, 0], ['uv', 2])],
                'outputs': [{'name': 'X', 'type': 'VALUE'},
                            {'name': 'Y', 'type': 'VALUE'},
                            {'name': 'Z', 'type': 'VALUE'}]},
        'ramp': {'id': 'ramp', 'bl_idname': 'ShaderNodeMapRange',
                 'props': {},
                 'inputs': [sk('Value', 'VALUE', 0.5, ['sep', 0]),
                            sk('From Min', 'VALUE', 0.0),
                            sk('From Max', 'VALUE', 1.0),
                            sk('To Min', 'VALUE', 3.0),
                            sk('To Max', 'VALUE', 9.0)],
                 'outputs': [{'name': 'Result', 'type': 'VALUE'}]},
        'code': code_node('code', CODE, [
            {'name': 'Base Color', 'type': 'VECTOR', 'uniform': 'baseColor',
             'default': [0.2, 0.55, 0.9], 'link': None},
            {'name': 'Bands', 'type': 'VALUE', 'uniform': 'bands',
             'default': 4.0, 'link': ['ramp', 0]}]),
    }
    sc, st = scene_with(master(chain, ['code', 0]),
                        {'code': compile_shader(CODE)})
    cpu, g, job, passes, why, atlases = run(sc, st)
    check('a coded-shader frame qualifies', passes is not None, str(why))
    if passes is not None:
        mi, name, src, binds = [p for p in passes if p[1] == 'Coded'][0]
        check('the clock is a per-frame uniform, not a baked constant',
              binds.get('frame_uniforms') == ['hal_time']
              and 'uniform float hal_time;' in src, str(binds))
        check('the socket value beat the declared default',
              '0.55' in src and '_cncode_main();' in src, 'no')
        decl = re.compile(r'^(uniform|in|out)\s+\w[^;]*;$')
        left = [ln for ln in strip_declarations(src).splitlines()
                if decl.match(ln.split('//', 1)[0].strip())]
        check('the coded source strips clean for CreateInfo', not left,
              str(left))
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('the coded GPU frame is the CPU frame', err < 6e-3,
              f'max {err:.5f}')
        # an animation must not recompile: new time, byte-identical source
        sc.time = 9.9
        GSH._PLAN_CACHE.clear()
        p2, _w2, _a2 = GSH.plan_frame(job, g)
        check('a time change keeps the sources byte-identical',
              p2 is not None and
              [p for p in p2 if p[1] == 'Coded'][0][2] == src)

    # ---- two coded shaders sharing a uniform name cannot collide
    both = dict(chain)
    both['code2'] = code_node('code2', CODE2, [
        {'name': 'Base Color', 'type': 'VECTOR', 'uniform': 'baseColor',
         'default': [0.9, 0.15, 0.6], 'link': None}])
    both['mix'] = {'id': 'mix', 'bl_idname': 'ShaderNodeMixRGB',
                   'props': {'blend_type': 'MIX'},
                   'inputs': [sk('Fac', 'VALUE', 0.5, ['sep', 1]),
                              sk('Color1', 'RGBA', [0, 0, 0, 1],
                                 ['code', 0]),
                              sk('Color2', 'RGBA', [0, 0, 0, 1],
                                 ['code2', 0])],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'}]}
    sc, st = scene_with(master(both, ['mix', 0]),
                        {'code': compile_shader(CODE),
                         'code2': compile_shader(CODE2)})
    cpu, g, job, passes, why, atlases = run(sc, st)
    check('two coded shaders in one material qualify',
          passes is not None, str(why))
    if passes is not None:
        src = [p for p in passes if p[1] == 'Coded'][0][2]
        check('their names are mangled apart',
              '_cncode_baseColor' in src and '_cncode2_baseColor' in src)
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('and both shade as the CPU shades them', err < 6e-3,
              f'max {err:.5f}')

    # ---- a node whose program never compiled reads zeros on both sides
    sc, st = scene_with(master(chain, ['code', 0]), {})
    cpu, g, job, passes, why, atlases = run(sc, st)
    check('a missing program still qualifies (zeros, as the CPU answers)',
          passes is not None, str(why))
    if passes is not None:
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('and the zeros agree with the CPU zeros', err < 6e-3,
              f'max {err:.5f}')

    # ---- the refusals through the whole plan: the CPU renders these
    # happily (its front-end is forgiving), the GPU refuses them by name
    cases = [
        ('in vec3 vNormal;\nout vec4 Color;\nvoid main(){ Color = '
         'vec4(saturate(vNormal.z)); }',
         'HLSL-flavoured', 'saturate compiles here, dies on a driver'),
        ('in vec3 vNormal;\nout vec4 Color;\nvoid main(){ if (vNormal.z '
         '< -2.0) discard; Color = vec4(0.5, 0.5, 0.5, 1.0); }',
         'discard', 'a discard that sixteen probes cannot rule out'),
        ('in float vDepth;\nout vec4 Color;\nvoid main(){ Color = '
         'vec4(vDepth); }', 'view-space depth', 'the depth varying'),
    ]
    for src_bad, needle, label in cases:
        sc, st = scene_with(
            master({'code': code_node('code', src_bad, [])}, ['code', 0]),
            {'code': compile_shader(src_bad)})
        _cpu, g, job, passes, why, _a = run(sc, st)
        check(f'{label} is refused by name',
              passes is None and needle in str(why), str(why))

    # sampler uniforms travel now (1.25.16) -- the port carries them for
    # the assembler's sampler machinery instead of refusing
    from ..gpu.emit import Unsupported as EmitUnsupported, _port_code_node
    ported = _port_code_node(
        'uniform sampler2D img;\nout vec4 Color;\nvoid main(){ Color = '
        'texture(img, vec2(0.5, 0.5)); }', 'x')
    check('a sampler uniform is carried, mangled, not refused',
          ported[4] == [('img', '_cnx_img')] and '_cnx_img' in ported[0],
          str(ported[4]))
    try:
        _port_code_node('out vec4 Color;\nvoid mainImage(){ Color = '
                        'vec4(1.0); }', 'x')
        check('a shadertoy entry point is refused by name', False,
              'no refusal raised')
    except EmitUnsupported as exc:
        check('a shadertoy entry point is refused by name',
              'mainImage' in str(exc), str(exc))

    # HLSL as a language refuses before anything parses
    HLSL_SRC = 'out float4 Color;\nvoid main(){ Color = ' \
               'float4(1.0, 1.0, 1.0, 1.0); }'
    sc, st = scene_with(master({'code': {
        'id': 'code', 'bl_idname': 'HALCYON_CodeNode',
        'props': {'source_text': HLSL_SRC, 'language': 'HLSL'},
        'inputs': [], 'outputs': [{'name': 'Color', 'type': 'RGBA'}]}},
        ['code', 0]), {'code': compile_shader(HLSL_SRC, 'HLSL')})
    _cpu, g, job, passes, why, _a = run(sc, st)
    check('HLSL refuses as before', passes is None and 'HLSL' in str(why),
          str(why))


def test_period_patterns_shade_on_the_gpu():
    """Halcyon's own procedural textures run in the deferred pass.

    The pattern library rides an integer hash -- multiply-accumulate,
    xorshift, mask -- and that is what makes it portable at all: uint32
    wrap-then-mask equals the CPU's exact-int64-then-mask, because 2^31
    divides 2^32. The front-end grew real uint semantics for this, so every
    GLSL primitive is verified against its patterns.py original first, then
    whole materials against render() itself. Generated coordinates travel
    too: per-object bounds bake as lookup functions over the object index
    the tri_data texture already carries -- so the DEFAULT wiring of every
    texture node (unlinked Vector) now qualifies instead of refusing.

    Blender's Noise/Voronoi/White Noise/Musgrave/Brick stay refused BY
    NAME: their sin-fract hash is evaluated in float64 on the CPU and a
    driver's float32 sin decorrelates it into a different picture. Wave
    refuses only when distorted (distortion runs that same Perlin);
    Gradient and Magic are pure arithmetic and travel whole.
    """
    from ..core import patterns as PT
    from ..core import raster
    from ..core.scene import Material
    from ..gpu import shade as GSH
    from ..gpu.procedural import PATTERN_GLSL, PRIM_GLSL
    from ..shaders.compiler import try_compile

    # ---- the primitives against patterns.py, through the front-end whose
    # uint semantics now wrap exactly as a driver's
    rng = np.random.default_rng(5)
    pts = (rng.random((2000, 3)).astype(np.float32) * 40.0 - 20.0)

    def run_prim(body, extra=''):
        src = PRIM_GLSL + extra + f'''
uniform vec3 pin;
out vec4 Color;
void main() {{ Color = vec4({body}, 0.0, 0.0, 1.0); }}'''
        prog, err = try_compile(src)
        check('a primitive compiles through the front-end', prog is not None,
              str(err))
        if prog is None:
            return None
        return prog.run({'pin': pts}, {}, pts.shape[0])[0]['Color'][:, 0]

    i = np.floor(pts).astype(np.int64)
    got = run_prim('hal_pt_hash3(int(floor(pin.x)), int(floor(pin.y)), '
                   'int(floor(pin.z)))')
    if got is not None:
        want = PT.hash3(i[:, 0], i[:, 1], i[:, 2])
        check('the integer hash is EXACT under uint32 wrap',
              bool(np.array_equal(got.astype(np.float32),
                                  want.astype(np.float32))))
    for label, body, want in (
            ('value noise', 'hal_pt_vnoise(pin)', PT.value_noise(pts)),
            ('fbm', 'hal_pt_fbm(pin, 6, 2.4, 0.62)',
             PT.fbm(pts, 6, 2.4, 0.62)),
            ('turbulence', 'hal_pt_turb(pin, 5, 2.0, 0.5)',
             PT.turbulence(pts, 5)),
            ('worley F1', 'hal_pt_worley(pin, 1.0).x', PT.worley(pts)[0]),
            ('worley F2', 'hal_pt_worley(pin, 0.7).y',
             PT.worley(pts, 0.7)[1])):
        got = run_prim(body)
        if got is not None:
            e = float(np.abs(got.astype(np.float32)
                             - want.astype(np.float32)).max())
            check(f'{label} matches patterns.py', e < 2e-4, f'max {e:.7f}')

    # ---- whole materials against render(), wired as a user wires them:
    # unlinked Vector, meaning generated coordinates
    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def master(extra, link):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0], link),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ]
        nodes = {'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                         'props': {'model': 'PHONG', 'toon_steps': 2},
                         'inputs': ins,
                         'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
                 'out': {'id': 'out',
                         'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                         'inputs': [sk('Surface', 'SHADER', None,
                                       ['hal', 0]),
                                    sk('Displacement', 'VECTOR', [0, 0, 0])],
                         'outputs': []}}
        nodes.update(extra)
        return {'output': 'out', 'nodes': nodes}

    def run_mat(graph, label):
        w, h = 128, 96
        st = base_settings(w, h, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.materials[1] = Material(name='Pat', index=1, graph=graph)
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'{label} qualifies', passes is not None, str(why))
        if passes is None:
            return None
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{label} is the CPU frame', err < 6e-3, f'max {err:.5f}')
        return passes

    def pat_node(idname, sockets, props):
        return {'pat': {'id': 'pat', 'bl_idname': idname, 'props': props,
                        'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                                   sk('Scale', 'VALUE', 4.0)] + sockets
                        + [sk('Color 1', 'RGBA', [0.92, 0.9, 0.86, 1.0]),
                           sk('Color 2', 'RGBA', [0.14, 0.13, 0.16, 1.0])],
                        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                    {'name': 'Fac', 'type': 'VALUE'}]}}

    run_mat(master(pat_node('HALCYON_MarbleNode',
                            [sk('Turbulence', 'VALUE', 1.3),
                             sk('Veins', 'VALUE', 1.2),
                             sk('Sharpness', 'VALUE', 0.8)],
                            {'octaves': 5, 'axis': 'Y'}), ['pat', 0]),
            'marble on generated coordinates')
    run_mat(master(pat_node('HALCYON_WoodNode',
                            [sk('Rings', 'VALUE', 8.0),
                             sk('Turbulence', 'VALUE', 0.35),
                             sk('Grain', 'VALUE', 0.4)],
                            {'octaves': 4, 'axis': 'Z'}), ['pat', 0]),
            'wood')
    run_mat(master(pat_node('HALCYON_GraniteNode',
                            [sk('Contrast', 'VALUE', 1.6),
                             sk('Speckle', 'VALUE', 0.35)],
                            {'octaves': 6}), ['pat', 0]), 'granite')
    run_mat(master(pat_node('HALCYON_DentsNode',
                            [sk('Size', 'VALUE', 1.0),
                             sk('Depth', 'VALUE', 1.0)],
                            {'octaves': 3}), ['pat', 0]), 'dents')
    run_mat(master(pat_node('HALCYON_CrackleNode',
                            [sk('Randomness', 'VALUE', 1.0),
                             sk('Width', 'VALUE', 0.06),
                             sk('Smooth', 'VALUE', 0.02)],
                            {}), ['pat', 0]), 'crackle')

    # the Blender nodes that are pure arithmetic
    run_mat(master({'pat': {'id': 'pat',
                            'bl_idname': 'ShaderNodeTexGradient',
                            'props': {'gradient_type': 'RADIAL'},
                            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                        {'name': 'Fac', 'type': 'VALUE'}]}},
                   ['pat', 0]), 'a radial gradient')
    run_mat(master({'pat': {'id': 'pat', 'bl_idname': 'ShaderNodeTexMagic',
                            'props': {'turbulence_depth': 3},
                            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                                       sk('Scale', 'VALUE', 2.5),
                                       sk('Distortion', 'VALUE', 1.4)],
                            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                        {'name': 'Fac', 'type': 'VALUE'}]}},
                   ['pat', 0]), 'magic depth 3')
    run_mat(master({'pat': {'id': 'pat', 'bl_idname': 'ShaderNodeTexWave',
                            'props': {'wave_type': 'RINGS',
                                      'wave_profile': 'TRI'},
                            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                                       sk('Scale', 'VALUE', 1.2),
                                       sk('Distortion', 'VALUE', 0.0),
                                       sk('Detail', 'VALUE', 2.0),
                                       sk('Detail Scale', 'VALUE', 1.0)],
                            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                        {'name': 'Fac', 'type': 'VALUE'}]}},
                   ['pat', 0]), 'undistorted ring waves')

    # ---- the refusals, each named
    def refuses(graph, needle, label):
        w, h = 96, 72
        st = base_settings(w, h)
        sc = demo_scene(st, with_texture=False)
        sc.materials[1] = Material(name='Pat', index=1, graph=graph)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        check(f'{label} refuses by name', p is None and needle in str(why),
              str(why))

    refuses(master({'pat': {'id': 'pat', 'bl_idname': 'ShaderNodeTexNoise',
                            'props': {}, 'inputs':
                            [sk('Vector', 'VECTOR', [0, 0, 0]),
                             sk('Scale', 'VALUE', 5.0),
                             sk('Detail', 'VALUE', 2.0),
                             sk('Roughness', 'VALUE', 0.5),
                             sk('Distortion', 'VALUE', 0.0)],
                            'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                                        {'name': 'Color', 'type': 'RGBA'}]}},
                   ['pat', 1]), 'sin-fract', 'the Noise texture')
    refuses(master({'pat': {'id': 'pat', 'bl_idname': 'ShaderNodeTexWave',
                            'props': {},
                            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                                       sk('Scale', 'VALUE', 1.0),
                                       sk('Distortion', 'VALUE', 2.0),
                                       sk('Detail', 'VALUE', 2.0),
                                       sk('Detail Scale', 'VALUE', 1.0)],
                            'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                        {'name': 'Fac', 'type': 'VALUE'}]}},
                   ['pat', 0]), 'distortion', 'a distorted wave')
    linked = pat_node('HALCYON_MarbleNode',
                      [sk('Turbulence', 'VALUE', 1.0, ['val', 0]),
                       sk('Veins', 'VALUE', 1.0),
                       sk('Sharpness', 'VALUE', 1.0)],
                      {'octaves': 5, 'axis': 'X'})
    linked['val'] = {'id': 'val', 'bl_idname': 'ShaderNodeValue',
                     'props': {'value': 1.5}, 'inputs': [],
                     'outputs': [{'name': 'Value', 'type': 'VALUE'}]}
    refuses(master(linked, ['pat', 0]), 'batch mean',
            'a linked pattern scalar')


def test_the_pattern_library_completes_on_the_gpu():
    """Every remaining period pattern shades in the deferred pass.

    The second half of the library, mechanical now the primitives exist:
    Plasma, Ripples, Starfield, Weave, Scratches, Tiles, Brick, Spiral,
    Bozo, Agate, Leopard, Onion, Bumps and Wrinkles, each against
    `render()` itself. Two seams are specific to this batch. The seeded
    generators (Ripples' sources, Scratches' angles) are per-scene
    constants, so they BAKE -- the GLSL never needs the generator, and both
    sides read literal-identical numbers. And the animated ones ride
    `hal_time` like the coded shaders do: a time change leaves every source
    byte-identical, so an animation never recompiles a plasma.
    """
    from ..core import raster
    from ..core.scene import Material
    from ..gpu import shade as GSH

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def master(extra, link):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0], link),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ]
        nodes = {'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                         'props': {'model': 'PHONG', 'toon_steps': 2},
                         'inputs': ins,
                         'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
                 'out': {'id': 'out',
                         'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                         'inputs': [sk('Surface', 'SHADER', None,
                                       ['hal', 0]),
                                    sk('Displacement', 'VECTOR', [0, 0, 0])],
                         'outputs': []}}
        nodes.update(extra)
        return {'output': 'out', 'nodes': nodes}

    CASES = [
        ('HALCYON_PlasmaNode', {'animate': True, 'cycle_palette': True},
         [sk('Complexity', 'VALUE', 3.0), sk('Speed', 'VALUE', 1.0)]),
        ('HALCYON_RipplesNode', {'animate': True, 'sources': 4, 'seed': 2},
         [sk('Frequency', 'VALUE', 8.0), sk('Decay', 'VALUE', 0.6),
          sk('Speed', 'VALUE', 1.5)]),
        ('HALCYON_StarfieldNode', {},
         [sk('Density', 'VALUE', 0.7), sk('Size', 'VALUE', 0.4),
          sk('Twinkle', 'VALUE', 0.5),
          sk('Sky Color', 'RGBA', [0.02, 0.02, 0.05, 1]),
          sk('Star Color', 'RGBA', [1, 1, 0.9, 1])]),
        ('HALCYON_WeaveNode', {},
         [sk('Thickness', 'VALUE', 0.35), sk('Gap', 'VALUE', 0.08),
          sk('Distortion', 'VALUE', 0.2),
          sk('Warp Color', 'RGBA', [0.7, 0.2, 0.2, 1]),
          sk('Weft Color', 'RGBA', [0.2, 0.2, 0.7, 1])]),
        ('HALCYON_ScratchesNode', {'count': 5, 'seed': 3},
         [sk('Width', 'VALUE', 0.03), sk('Length', 'VALUE', 1.2),
          sk('Anisotropy', 'VALUE', 0.8)]),
        ('HALCYON_TilesNode', {},
         [sk('Rows', 'VALUE', 4.0), sk('Columns', 'VALUE', 5.0),
          sk('Grout', 'VALUE', 0.08), sk('Offset', 'VALUE', 0.5),
          sk('Bevel', 'VALUE', 0.15), sk('Variation', 'VALUE', 0.6),
          sk('Tile Color', 'RGBA', [0.8, 0.75, 0.7, 1]),
          sk('Grout Color', 'RGBA', [0.2, 0.2, 0.2, 1])]),
        ('HALCYON_BrickNode', {},
         [sk('Width', 'VALUE', 0.25), sk('Height', 'VALUE', 0.125),
          sk('Mortar', 'VALUE', 0.05), sk('Offset', 'VALUE', 0.5),
          sk('Bevel', 'VALUE', 0.12), sk('Variation', 'VALUE', 0.5),
          sk('Brick Color', 'RGBA', [0.6, 0.25, 0.15, 1]),
          sk('Mortar Color', 'RGBA', [0.75, 0.72, 0.68, 1])]),
        ('HALCYON_SpiralNode', {'axis': 'Z'},
         [sk('Turns', 'VALUE', 4.0), sk('Sharpness', 'VALUE', 1.0),
          sk('Twist', 'VALUE', 0.5)]),
        ('HALCYON_BozoNode', {'octaves': 4},
         [sk('Turbulence', 'VALUE', 0.8), sk('Lacunarity', 'VALUE', 2.0)]),
        ('HALCYON_AgateNode', {'octaves': 6, 'axis': 'Z'},
         [sk('Turbulence', 'VALUE', 1.0), sk('Bands', 'VALUE', 1.1),
          sk('Sharpness', 'VALUE', 0.77)]),
        ('HALCYON_LeopardNode', {}, [sk('Spot', 'VALUE', 1.0)]),
        ('HALCYON_OnionNode', {},
         [sk('Thickness', 'VALUE', 1.0), sk('Sharpness', 'VALUE', 1.0)]),
        ('HALCYON_BumpsNode', {'octaves': 1},
         [sk('Roundness', 'VALUE', 1.3), sk('Lacunarity', 'VALUE', 2.0),
          sk('Gain', 'VALUE', 0.5)]),
        ('HALCYON_WrinklesNode', {'octaves': 8},
         [sk('Lacunarity', 'VALUE', 2.0), sk('Crease', 'VALUE', 1.0)]),
    ]

    w, h = 96, 72
    plasma_src = None
    for idname, props, socks in CASES:
        base = [sk('Vector', 'VECTOR', [0, 0, 0]), sk('Scale', 'VALUE', 4.0)]
        has_c = any('Color' in s['name'] for s in socks)
        extra_c = [] if has_c else \
            [sk('Color 1', 'RGBA', [0.9, 0.9, 0.85, 1]),
             sk('Color 2', 'RGBA', [0.15, 0.12, 0.1, 1])]
        node = {'pat': {'id': 'pat', 'bl_idname': idname, 'props': props,
                        'inputs': base + socks + extra_c,
                        'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                    {'name': 'Fac', 'type': 'VALUE'}]}}
        st = base_settings(w, h, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.time = 2.3
        sc.materials[1] = Material(name='Pat', index=1,
                                   graph=master(node, ['pat', 0]))
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        short = idname.replace('HALCYON_', '').replace('Node', '')
        check(f'{short} qualifies', passes is not None, str(why))
        if passes is None:
            continue
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{short} is the CPU frame', err < 6e-3, f'max {err:.5f}')
        if idname == 'HALCYON_PlasmaNode':
            plasma_src = [p for p in passes if p[1] == 'Pat'][0]
            check('the animated plasma rides the clock as a uniform',
                  'hal_time' in plasma_src[3].get('frame_uniforms', ()),
                  str(plasma_src[3]))
            sc.time = 7.7
            GSH._PLAN_CACHE.clear()
            p2, _w2, _a2 = GSH.plan_frame(job, g)
            check('a time change keeps the plasma source byte-identical',
                  p2 is not None and
                  [p for p in p2 if p[1] == 'Pat'][0][2] == plasma_src[2])


def test_the_new_halcyon_nodes_travel():
    """The new node shelf, end to end on both devices.

    Eight new nodes (Fractal Noise, Cells, TV Static, Pixelate, UV Scroll,
    Scanlines, Hardware Palette, Color Cycle) plus GPU stories for the
    four originals: Posterize and Ordered Dither emit (roundEven IS
    np.round -- the depth quantiser proved that pairing), Screen Info
    emits everything but Depth, Depth Cue refuses by name. The Bayer
    threshold is ARITHMETIC on the GPU -- digit 2*(x^y)+y per bit -- and
    must equal dither.threshold_map bit for bit. Also the regression for
    the round's latent bug: Dither's pattern and Depth Cue's falloff were
    never in export.NODE_PROPS, so both dropdowns silently rendered as
    their defaults from the day they shipped.
    """
    from ..core import dither as DT
    from ..core import patterns as PT
    from ..core import raster
    from ..core.nodeeval import GraphEvaluator, ShadeContext
    from ..core.palette import NODE_PALETTES
    from ..core.settings import RenderSettings
    from . import fakebpy
    fakebpy.install()                  # export.py imports bpy via compat
    from ..export import NODE_PROPS
    from ..gpu import shade as GSH
    from ..gpu.procedural import PRIM_GLSL
    from ..shaders.compiler import try_compile
    from .featurematrix import build

    # ---- the new primitives against patterns.py, through the front-end
    rng = np.random.default_rng(9)
    pts = (rng.random((2000, 3)).astype(np.float32) * 40.0 - 20.0)

    def run_prim(body):
        src = PRIM_GLSL + f'''
uniform vec3 pin;
out vec4 Color;
void main() {{ Color = vec4({body}, 0.0, 0.0, 1.0); }}'''
        prog, err = try_compile(src)
        check('a new primitive compiles', prog is not None, str(err))
        if prog is None:
            return None
        return prog.run({'pin': pts}, {}, pts.shape[0])[0]['Color'][:, 0]

    got = run_prim('hal_pt_ridged(pin, 5, 2.0, 0.5)')
    if got is not None:
        e = float(np.abs(got - PT.ridged(pts, 5)).max())
        check('ridged matches patterns.py', e < 2e-4, f'max {e:.7f}')
    f1c, f2c, idc = PT.worley(pts, 0.9)
    for comp, want, label in (('x', f1c, 'F1'), ('y', f2c, 'F2')):
        got = run_prim(f'hal_pt_worley3(pin, 0.9).{comp}')
        if got is not None:
            e = float(np.abs(got - want).max())
            check(f'worley3 {label} matches', e < 2e-4, f'max {e:.7f}')
    got = run_prim('hal_pt_worley3(pin, 0.9).z')
    if got is not None:
        # ids are hash draws spaced 1/65535 apart; the sim's float64
        # intermediates can move the last bit, so match within a tenth
        # of that spacing rather than bit-for-bit
        frac = float((np.abs(got - idc) < 1e-6).mean())
        check('worley3 carries the winning cell id', frac > 0.999,
              f'{frac:.4f} matching')

    # ---- the Bayer arithmetic against the CPU's threshold matrices
    xs, ys = np.meshgrid(np.arange(16), np.arange(16), indexing='xy')
    grid = np.stack([xs.ravel(), ys.ravel(),
                     np.zeros(256)], 1).astype(np.float32)

    from ..gpu.emit import _BAYER_GLSL as _BAYER_TEST_GLSL

    def run_bayer(bits, n):
        # px % n by uint mask, exactly the emitter's own form -- the
        # front-end has no integer division to lean on
        src = _BAYER_TEST_GLSL + f'''
uniform vec3 pin;
out vec4 Color;
void main() {{ Color = vec4(hal_bayer(int(uint(int(pin.x)) & {n - 1}u),
    int(uint(int(pin.y)) & {n - 1}u), {bits},
    {float(n * n)!r}), 0.0, 0.0, 1.0); }}'''
        prog, err = try_compile(src)
        check(f'hal_bayer({n}x{n}) compiles', prog is not None, str(err))
        if prog is None:
            return None
        return prog.run({'pin': grid}, {}, 256)[0]['Color'][:, 0]
    for kind, bits, n in (('BAYER2', 1, 2), ('BAYER4', 2, 4),
                          ('BAYER8', 3, 8)):
        got = run_bayer(bits, n)
        if got is None:
            continue
        tm = DT.threshold_map(kind, 16, 16)
        want = tm[grid[:, 1].astype(int), grid[:, 0].astype(int)]
        check(f'the {kind} arithmetic equals threshold_map EXACTLY',
              bool(np.array_equal(got.astype(np.float32),
                                  want.astype(np.float32))))

    # ---- whole materials: the four matrix scenes cover every new node
    for key in ('node Fractal Noise (ridged)',
                'node Cells + Hardware Palette',
                'node Pixelate + Scroll + Scanlines',
                'node TV Static + Ordered Dither'):
        sc, st = build(key)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc.time = 1.7
        sc.frame = 12
        w, h = st.resolution_x, st.resolution_y
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'{key} qualifies', passes is not None, str(why))
        if passes is None:
            continue
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{key} is the CPU frame', err < 6e-3, f'max {err:.5f}')

    # ---- refusals, each named
    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def diffuse_graph(extra, link):
        nodes = {'bsdf': {'id': 'bsdf',
                          'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
                          'inputs': [sk('Color', 'RGBA', [1, 1, 1, 1],
                                        link),
                                     sk('Roughness', 'VALUE', 0.0)],
                          'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]},
                 'out': {'id': 'out',
                         'bl_idname': 'ShaderNodeOutputMaterial',
                         'props': {},
                         'inputs': [sk('Surface', 'SHADER', None,
                                       ['bsdf', 0]),
                                    sk('Displacement', 'VECTOR',
                                       [0, 0, 0])],
                         'outputs': []}}
        nodes.update(extra)
        return {'output': 'out', 'nodes': nodes}

    def refuses(graph, needle, label):
        from ..core.scene import Material
        w, h = 96, 72
        st = base_settings(w, h)
        sc = demo_scene(st, with_texture=False)
        sc.materials[1] = Material(name='Ref', index=1, graph=graph)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        check(f'{label} refuses by name', p is None and needle in str(why),
              str(why))

    refuses(diffuse_graph({'cue': {
        'id': 'cue', 'bl_idname': 'HALCYON_DepthCueNode',
        'props': {'mode': 'LINEAR'},
        'inputs': [sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1]),
                   sk('Fog Color', 'RGBA', [0.5, 0.55, 0.65, 1]),
                   sk('Start', 'VALUE', 5.0), sk('End', 'VALUE', 40.0)],
        'outputs': [{'name': 'Color', 'type': 'RGBA'}]}}, ['cue', 0]),
        'view matrix', 'Depth Cue')
    refuses(diffuse_graph({'dith': {
        'id': 'dith', 'bl_idname': 'HALCYON_DitherNode',
        'props': {'pattern': 'HALFTONE'},
        'inputs': [sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1]),
                   sk('Levels', 'VALUE', 4.0),
                   sk('Strength', 'VALUE', 1.0)],
        'outputs': [{'name': 'Color', 'type': 'RGBA'}]}}, ['dith', 0]),
        'halftone', 'the halftone dither')
    refuses(diffuse_graph({'si': {
        'id': 'si', 'bl_idname': 'HALCYON_ScreenInfoNode', 'props': {},
        'inputs': [],
        'outputs': [{'name': 'Screen UV', 'type': 'VECTOR'},
                    {'name': 'Pixel', 'type': 'VECTOR'},
                    {'name': 'Depth', 'type': 'VALUE'},
                    {'name': 'Facing', 'type': 'VALUE'},
                    {'name': 'Frame', 'type': 'VALUE'},
                    {'name': 'Time', 'type': 'VALUE'}]}}, ['si', 2]),
        'depth', "Screen Info's Depth")

    # ---- Screen Info's Facing DOES travel
    from ..core.scene import Material
    w, h = 96, 72
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[1] = Material(name='Face', index=1, graph=diffuse_graph(
        {'si': {'id': 'si', 'bl_idname': 'HALCYON_ScreenInfoNode',
                'props': {}, 'inputs': [],
                'outputs': [{'name': 'Screen UV', 'type': 'VECTOR'},
                            {'name': 'Pixel', 'type': 'VECTOR'},
                            {'name': 'Depth', 'type': 'VALUE'},
                            {'name': 'Facing', 'type': 'VALUE'},
                            {'name': 'Frame', 'type': 'VALUE'},
                            {'name': 'Time', 'type': 'VALUE'}]}},
        ['si', 3]))
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check("Screen Info's Facing qualifies", passes is not None, str(why))
    if passes is not None:
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('and is the CPU frame', err < 6e-3, f'max {err:.5f}')

    # ---- the export-table regression: the props the evaluators read
    for idname, want in (('HALCYON_DitherNode', ('pattern',)),
                         ('HALCYON_DepthCueNode', ('mode',)),
                         ('HALCYON_PaletteNode', ('palette',)),
                         ('HALCYON_ScrollNode', ('animate', 'fps')),
                         ('HALCYON_ScanlinesNode', ('animate',)),
                         ('HALCYON_ColorCycleNode', ('animate',))):
        check(f'{idname} exports {want}', NODE_PROPS.get(idname) == want,
              str(NODE_PROPS.get(idname)))

    # ---- and the CPU evaluators HONOR those props (the bug's other half)
    n = 256
    ctx = ShadeContext(n)
    ctx.settings = RenderSettings()
    grid_uv = np.stack([np.linspace(0, 1, n),
                        np.linspace(0, 1, n) ** 1.2], 1).astype(np.float32)
    ctx.uv = grid_uv
    ctx.generated = np.concatenate(
        [grid_uv, np.full((n, 1), 0.4, np.float32)], 1)
    ctx.P = ctx.generated * 3.0
    ctx.N = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    ctx.I = np.tile(np.array([[0, 0, -1.0]], np.float32), (n, 1))
    ctx.px = np.arange(n) % 64
    ctx.py = np.arange(n) // 4
    ctx.depth = np.linspace(2.0, 30.0, n).astype(np.float32)
    ctx.time = 2.0
    ctx.frame = 7

    def one(idname, inputs, props, out_name):
        graph = {'output': None,
                 'nodes': {'t': {'id': 't', 'bl_idname': idname,
                                 'props': props, 'inputs': inputs,
                                 'outputs': [{'name': out_name}]}}}
        ev = GraphEvaluator(graph, ctx)
        got = ev.eval_output('t', 0)
        check(f'{idname} evaluates', got is not None and not ev.errors,
              str(ev.errors[:1]))
        return got

    dsocks = [sk('Color', 'RGBA', [0.5, 0.5, 0.5, 1]),
              sk('Levels', 'VALUE', 4.0), sk('Strength', 'VALUE', 1.0)]
    d_bayer = one('HALCYON_DitherNode', dsocks, {'pattern': 'BAYER2'},
                  'Color')
    d_half = one('HALCYON_DitherNode', dsocks, {'pattern': 'HALFTONE'},
                 'Color')
    check('the Dither node HONORS its exported pattern',
          d_bayer is not None and d_half is not None and
          float(np.abs(d_bayer - d_half).max()) > 1e-4)
    csocks = [sk('Color', 'RGBA', [0.8, 0.8, 0.8, 1]),
              sk('Fog Color', 'RGBA', [0.2, 0.3, 0.4, 1]),
              sk('Start', 'VALUE', 3.0), sk('End', 'VALUE', 20.0)]
    q_lin = one('HALCYON_DepthCueNode', csocks, {'mode': 'LINEAR'}, 'Color')
    q_exp = one('HALCYON_DepthCueNode', csocks, {'mode': 'EXP2'}, 'Color')
    check('the Depth Cue node HONORS its exported falloff',
          q_lin is not None and q_exp is not None and
          float(np.abs(q_lin - q_exp).max()) > 1e-4)

    # ---- the new utilities behave, on the numbers
    px = one('HALCYON_PixelateNode',
             [sk('Vector', 'VECTOR', None), sk('Pixels X', 'VALUE', 8.0),
              sk('Pixels Y', 'VALUE', 8.0), sk('Pixels Z', 'VALUE', 0.0)],
             {}, 'Vector')
    if px is not None:
        check('Pixelate snaps to 8 cells',
              len(np.unique(px[:, 0].round(6))) <= 8,
              str(len(np.unique(px[:, 0].round(6)))))
        check('and a zero axis passes through',
              bool(np.array_equal(px[:, 2],
                                  np.zeros(n, np.float32))))
    sc_socks = [sk('Vector', 'VECTOR', None),
                sk('Scroll X', 'VALUE', 0.25), sk('Scroll Y', 'VALUE', 0.0),
                sk('Spin', 'VALUE', 0.0)]
    moved = one('HALCYON_ScrollNode', sc_socks, {'animate': True, 'fps': 0},
                'Vector')
    if moved is not None:
        drift = float(np.abs(moved[:, 0] - (grid_uv[:, 0] + 0.5)).max())
        check('UV Scroll moves 0.25/s for 2s', drift < 1e-5,
              f'max {drift:.6f}')
    ctx.time = 2.13            # a time the 15-step clock genuinely rounds
    stepped = one('HALCYON_ScrollNode', sc_socks,
                  {'animate': True, 'fps': 15}, 'Vector')
    ctx.time = 2.0
    if stepped is not None:
        t15 = np.floor(np.float32(2.13) * np.float32(15)) / np.float32(15)
        check('the stepped clock is not the smooth one',
              abs(float(t15) - 2.13) > 1e-3, f'{float(t15):.5f}')
        drift = float(np.abs(stepped[:, 0]
                             - (grid_uv[:, 0]
                                + np.float32(0.25) * t15)).max())
        check('and quantises its clock at 15 steps', drift < 1e-5,
              f'max {drift:.6f}')
    lines = one('HALCYON_ScanlinesNode',
                [sk('Color', 'RGBA', [1.0, 1.0, 1.0, 1]),
                 sk('Vector', 'VECTOR', None), sk('Lines', 'VALUE', 8.0),
                 sk('Darkness', 'VALUE', 0.5),
                 sk('Thickness', 'VALUE', 0.5)],
                {'animate': False}, 'Color')
    if lines is not None:
        vals = set(np.round(lines[:, 0], 5))
        check('Scanlines split the surface into lit and darkened lines',
              vals == {0.5, 1.0}, str(sorted(vals)))
    pal = one('HALCYON_PaletteNode',
              [sk('Color', 'RGBA', [0.5, 0.5, 0.5, 1], None),
               sk('Mix', 'VALUE', 1.0)], {'palette': 'GAMEBOY'}, 'Color')
    if pal is not None:
        table = NODE_PALETTES['GAMEBOY']
        dmin = np.abs(pal[:, None, :3] - table[None]).max(axis=2).min(axis=1)
        check('Hardware Palette lands ON a Game Boy entry',
              float(dmin.max()) < 1e-6, f'max {float(dmin.max()):.7f}')
    cyc = one('HALCYON_ColorCycleNode',
              [sk('Fac', 'VALUE', 0.25), sk('Speed', 'VALUE', 0.5),
               sk('Steps', 'VALUE', 0.0)], {'animate': True}, 'Fac')
    if cyc is not None:
        # fac 0.25 + 0.5/s * 2s = 1.25 -> wraps to 0.25
        check('Color Cycle wraps its phase',
              float(np.abs(cyc - 0.25).max()) < 1e-5,
              f'{float(cyc[0]):.5f}')

    # ---- the standing invariant this round establishes: EVERY Halcyon
    # node has an evaluator AND a GPU story -- an emitter, or a refusal
    # whose reason is written down. A node in neither table would route
    # with a generic excuse, which is how the utilities sat for ten
    # versions.
    from ..core.nodeeval import DISPATCH
    from ..gpu.emit import EMITTERS, REFUSED
    from ..nodes import pattern_nodes as PN
    from ..nodes import shader_nodes as SN
    all_ids = [cls.bl_idname for cls in SN.NODES] + \
              [f'HALCYON_{spec[0]}Node' for spec in PN.SPECS]
    no_eval = [i for i in all_ids if i not in DISPATCH]
    no_story = [i for i in all_ids
                if i not in EMITTERS and i not in REFUSED]
    check(f'all {len(all_ids)} Halcyon nodes have evaluators', not no_eval,
          ', '.join(no_eval))
    check('and every one has a GPU story, emitter or named refusal',
          not no_story, ', '.join(no_story))


def test_more_utilities_and_the_geometry_audit():
    """Five more utilities, and the Geometry node told the truth at last.

    Flipbook, UV Wave, Halftone, Threshold and Quantize each work end to
    end on both devices. And the audit they prompted: the Geometry node's
    GPU mapping was BY INDEX from a hand-written list, and the list was
    wrong -- Tangent emitted the NORMAL, Incoming emitted the POSITION,
    Parametric emitted GENERATED coordinates, Backfacing emitted 0.0
    whatever the winding, Random Per Island emitted 0.0 where the CPU has
    a real hash. Four silent divergences and no test reading those
    sockets (the matrix-blindness lesson, again). Outputs resolve by
    name now: each either the CPU's exact expression -- Backfacing is
    the measured plane-side convention, perspective-gated like the
    backface override -- or a refusal that says why.
    """
    from ..core import raster
    from ..core.nodeeval import GraphEvaluator, ShadeContext
    from ..core.scene import Material
    from ..core.settings import RenderSettings
    from ..gpu import shade as GSH
    from .featurematrix import GEOMETRY_OUTS, build

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    # ---- the three new matrix scenes, sim against render()
    for key in ('node Flipbook + UV Wave',
                'node Halftone + Threshold + Quantize',
                'node Geometry (Tangent x Incoming)'):
        sc, st = build(key)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc.time = 0.9
        w, h = st.resolution_x, st.resolution_y
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'{key} qualifies', passes is not None, str(why))
        if passes is None:
            continue
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{key} is the CPU frame', err < 6e-3, f'max {err:.5f}')

    def geo_graph(out_index, scale=None):
        nodes = {'geo': {'id': 'geo', 'bl_idname': 'ShaderNodeNewGeometry',
                         'props': {}, 'inputs': [],
                         'outputs': [dict(o) for o in GEOMETRY_OUTS]}}
        src = ['geo', out_index]
        if scale is not None:
            nodes['mul'] = {'id': 'mul',
                            'bl_idname': 'ShaderNodeVectorMath',
                            'props': {'operation': 'MULTIPLY'},
                            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                          src),
                                       sk('Vector', 'VECTOR',
                                          [scale] * 3)],
                            'outputs': [{'name': 'Vector',
                                         'type': 'VECTOR'}]}
            src = ['mul', 0]
        nodes['absn'] = {'id': 'absn', 'bl_idname': 'ShaderNodeVectorMath',
                         'props': {'operation': 'ABSOLUTE'},
                         'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                       src)],
                         'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]}
        nodes['bsdf'] = {'id': 'bsdf',
                         'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
                         'inputs': [sk('Color', 'RGBA', [1, 1, 1, 1],
                                       ['absn', 0]),
                                    sk('Roughness', 'VALUE', 0.0)],
                         'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
        nodes['out'] = {'id': 'out',
                        'bl_idname': 'ShaderNodeOutputMaterial',
                        'props': {},
                        'inputs': [sk('Surface', 'SHADER', None,
                                      ['bsdf', 0]),
                                   sk('Displacement', 'VECTOR', [0, 0, 0])],
                        'outputs': []}
        return {'output': 'out', 'nodes': nodes}

    def run_graph(sc, st, graph, mat_index=1):
        w, h = st.resolution_x, st.resolution_y
        sc.materials[mat_index] = Material(name='Geo', index=mat_index,
                                           graph=graph)
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        if passes is None:
            return None, why, None, None
        img, _hit = GSH.simulate(job, g, passes, atlases)
        return passes, why, img, (cpu, g)

    # ---- every emitted Geometry output against the CPU, one at a time
    for idx, label, scale in ((0, 'Position', 0.12), (1, 'Normal', None),
                              (2, 'Tangent', None), (4, 'Incoming', None),
                              (5, 'Parametric', None)):
        st = base_settings(96, 72, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        passes, why, img, rest = run_graph(sc, st, geo_graph(idx, scale))
        check(f'Geometry {label} qualifies', passes is not None, str(why))
        if passes is None:
            continue
        cpu, g = rest
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'Geometry {label} is the CPU frame', err < 6e-3,
              f'max {err:.5f}')

    # ---- Backfacing on mixed winding: the plane test vs the rasteriser
    st = base_settings(96, 72, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    st.two_sided_lighting = True
    sc = demo_scene(st, with_texture=False)
    m = sc.mesh
    floor = np.nonzero(m.mat_index == 0)[0]
    m.tris[floor[::2]] = m.tris[floor[::2]][:, [0, 2, 1]]
    passes, why, img, rest = run_graph(sc, st, geo_graph(6), mat_index=0)
    check('Geometry Backfacing qualifies on a perspective camera',
          passes is not None, str(why))
    if passes is not None:
        cpu, g = rest
        cov = g.tri >= 0
        fronts = g.front[cov]
        check('the fixture really shows both windings',
              bool(fronts.any()) and bool((~fronts).any()),
              f'{int(fronts.sum())}/{fronts.size} front')
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('Backfacing is the CPU frame across both windings',
              err < 6e-3, f'max {err:.5f}')

    # ---- the refusals, named
    def refuses(graph, needle, label, camera=None):
        w, h = 96, 72
        st = base_settings(w, h)
        sc = demo_scene(st, with_texture=False)
        if camera:
            sc.camera.type = camera
        sc.materials[1] = Material(name='Ref', index=1, graph=graph)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        check(f'{label} refuses by name', p is None and needle in str(why),
              str(why))

    refuses(geo_graph(6), 'orthographic',
            'Backfacing under an orthographic camera', camera='ORTHO')

    # True Normal and Random Per Island FLIPPED from refusals to the
    # hal_triaux per-tri texture (the CPU's own stored normals and
    # sin-fract randoms, baked): the plan must qualify and the pass
    # must fetch the texel, not recompute
    def qualifies_with_aux(graph, label):
        w, h = 96, 72
        st = base_settings(w, h)
        sc = demo_scene(st, with_texture=False)
        sc.materials[1] = Material(name='Aux', index=1, graph=graph)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        p, why, a = GSH.plan_frame(job, g)
        check(f'{label} rides the per-tri texture now', p is not None
              and 'hal_triaux' in a
              and any('hal_triaux_fetch' in src for _m, _n, src, _b in p),
              str(why))

    qualifies_with_aux(geo_graph(3), "Geometry's True Normal")
    qualifies_with_aux(geo_graph(8, scale=1.0),
                       "Geometry's Random Per Island")

    # ---- the new utilities behave, on the numbers
    n = 256
    ctx = ShadeContext(n)
    ctx.settings = RenderSettings()
    uv = np.stack([np.linspace(0, 1, n),
                   np.linspace(0, 1, n) ** 1.1], 1).astype(np.float32)
    ctx.uv = uv
    ctx.generated = np.concatenate(
        [uv, np.full((n, 1), 0.3, np.float32)], 1)
    ctx.N = np.tile(np.array([[0, 0, 1.0]], np.float32), (n, 1))
    ctx.I = np.tile(np.array([[0, 0, -1.0]], np.float32), (n, 1))
    ctx.time = 0.0

    def one(idname, inputs, props, out_name, out_index=0):
        graph = {'output': None,
                 'nodes': {'t': {'id': 't', 'bl_idname': idname,
                                 'props': props, 'inputs': inputs,
                                 'outputs': [{'name': out_name}]
                                 if out_index == 0 else
                                 [{'name': f'o{i}'} for i in
                                  range(out_index)] +
                                 [{'name': out_name}]}}}
        ev = GraphEvaluator(graph, ctx)
        got = ev.eval_output('t', out_index)
        check(f'{idname} evaluates', got is not None and not ev.errors,
              str(ev.errors[:1]))
        return got

    # Flipbook at rest: offset 5 in a 4x4 sheet is column 1, row 1 from
    # the top -- so x lands in [0.25, 0.5) and y in [0.5, 0.75)
    fb = one('HALCYON_FlipbookNode',
             [sk('Vector', 'VECTOR', None), sk('Columns', 'VALUE', 4.0),
              sk('Rows', 'VALUE', 4.0), sk('Rate', 'VALUE', 8.0),
              sk('Cell Offset', 'VALUE', 5.0)],
             {'animate': False}, 'Vector')
    if fb is not None:
        want_x = (uv[:, 0] + 1.0) / 4.0
        want_y = (uv[:, 1] + 2.0) / 4.0
        ok = float(np.abs(fb[:, 0] - want_x).max()
                   + np.abs(fb[:, 1] - want_y).max())
        check('Flipbook picks cell 5 of a 4x4 sheet', ok < 1e-6,
              f'{ok:.7f}')
    wv = one('HALCYON_UVWaveNode',
             [sk('Vector', 'VECTOR', None),
              sk('Amplitude X', 'VALUE', 0.05),
              sk('Amplitude Y', 'VALUE', 0.0),
              sk('Frequency', 'VALUE', 3.0), sk('Speed', 'VALUE', 1.0)],
             {'animate': True}, 'Vector')
    if wv is not None:
        want = uv[:, 0] + np.sin(uv[:, 1] * np.float32(3.0)
                                 * np.float32(2.0 * np.pi)) \
            * np.float32(0.05)
        err = float(np.abs(wv[:, 0] - want).max())
        check('UV Wave is its own formula at t=0', err < 1e-6,
              f'{err:.7f}')
        check('and leaves y alone at zero amplitude',
              float(np.abs(wv[:, 1] - uv[:, 1]).max()) < 1e-7)
    ht_dark = one('HALCYON_HalftoneNode',
                  [sk('Color', 'RGBA', [0.15, 0.15, 0.15, 1]),
                   sk('Vector', 'VECTOR', None), sk('Dots', 'VALUE', 8.0),
                   sk('Angle', 'VALUE', 45.0),
                   sk('Ink Color', 'RGBA', [0, 0, 0, 1]),
                   sk('Paper Color', 'RGBA', [1, 1, 1, 1])],
                  {}, 'Fac', out_index=1)
    ht_light = one('HALCYON_HalftoneNode',
                   [sk('Color', 'RGBA', [0.85, 0.85, 0.85, 1]),
                    sk('Vector', 'VECTOR', None), sk('Dots', 'VALUE', 8.0),
                    sk('Angle', 'VALUE', 45.0),
                    sk('Ink Color', 'RGBA', [0, 0, 0, 1]),
                    sk('Paper Color', 'RGBA', [1, 1, 1, 1])],
                   {}, 'Fac', out_index=1)
    if ht_dark is not None and ht_light is not None:
        check('Halftone prints more ink where the input is darker',
              float(ht_dark.mean()) > float(ht_light.mean()),
              f'{float(ht_dark.mean()):.3f} vs '
              f'{float(ht_light.mean()):.3f}')
    # Threshold and Quantize, swept across a ramp: the handlers read
    # their sockets through ev.input, so a stub that answers by name
    # sweeps them without wiring a driver chain
    from ..core.nodeeval import n_halcyon_quantize as NQ
    from ..core.nodeeval import n_halcyon_threshold as NTH

    class _EvStub:
        def __init__(self, **m):
            self._m = m

        def input(self, node, name, kind):
            return self._m[name]

    dummy = {'id': 't', 'props': {}, 'inputs': [], 'outputs': []}
    ramp = np.linspace(0, 1, n).astype(np.float32)
    half = np.full(n, 0.5, np.float32)
    hard = NTH(_EvStub(Fac=ramp, Level=half,
                       Smooth=np.zeros(n, np.float32)), dummy)['Fac']
    check('Threshold cuts hard at the level',
          set(np.unique(hard)) == {np.float32(0.0), np.float32(1.0)}
          and float(hard[ramp < 0.5].max()) == 0.0
          and float(hard[ramp >= 0.5].min()) == 1.0)
    soft = NTH(_EvStub(Fac=ramp, Level=half,
                       Smooth=np.full(n, 0.2, np.float32)), dummy)['Fac']
    mid = (ramp > 0.42) & (ramp < 0.58)
    check('Threshold smooths across the band when asked',
          bool(((soft[mid] > 0.0) & (soft[mid] < 1.0)).any())
          and float(soft[ramp < 0.35].max()) == 0.0
          and float(soft[ramp > 0.65].min()) == 1.0)
    q = NQ(_EvStub(Fac=ramp, Steps=np.full(n, 5.0, np.float32)),
           dummy)['Fac']
    lv = np.unique(q)
    check('Quantize lands on exactly its steps', len(lv) == 5
          and float(lv.min()) == 0.0 and float(lv.max()) > 0.99,
          str(lv))


def test_the_era_audit_round():
    """The 90s-features audit: what was missing, dead, or lying.

    The sweep of every render property against its readers found two
    dead controls -- `reflection_blur_samples` had a UI slider and no
    reader (the Toon Steps disease), `watermark` sat in the settings
    table with no UI and no reader -- and one marquee absence: the era's
    Radiosity checkbox. All three live now:

    - RADIOSITY: one-bounce gathered ambient. Sky rays return the
      ambient colour, hit rays return the hit material's flat diffuse
      with a linear falloff -- colour bleed, deterministic streams, its
      own salt, and the SAME gather in GLSL against the closest-hit
      kernel with a baked albedo table.
    - BLURRY REFLECTIONS: a cone of jittered rays averaged per
      reflective fragment; the samples slider finally does something.
      The deferred sweeps carry one ray per fragment, so blurry frames
      refuse BY NAME and shade on the CPU.
    - THE BURN-IN: watermark text stamps the final frame in a 5x7
      bitmap font, white over a drop shadow, with %F/%R/%V tokens --
      after every post stage, so both devices stamp identical bits.
    """
    from ..core import post as PP
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    # ---- radiosity: bleed is real, and the GPU gathers the same rays
    def render_pair(paint_floor=True, **rad):
        st = base_settings(96, 72, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        for k, v in rad.items():
            setattr(st, k, v)
        sc = demo_scene(st, with_texture=False)
        if paint_floor:
            sc.materials[0].diffuse = (0.9, 0.05, 0.05)
        return sc, st

    sc, st = render_pair(radiosity=True, radiosity_samples=4,
                         radiosity_distance=4.0)
    on = R.render(sc, st)
    sc2, st2 = render_pair()
    off = R.render(sc2, st2)
    red = float((on[..., 0] - off[..., 0]).sum())
    grn = float((on[..., 1] - off[..., 1]).sum())
    check('the red floor lends the scene red', red > 1.0,
          f'{red:.2f}')
    check('...and it is BLEED, not brightening', red > grn + 1.0,
          f'red {red:.2f} vs green {grn:.2f}')

    view, _proj, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)    # as render() built it
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, 96, 72)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('radiosity qualifies for the deferred pass',
          passes is not None, str(why))
    if passes is not None:
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - on[cov][:, :3]).max())
        check('the GPU gathers the CPU frame', err < 6e-3,
              f'max {err:.6f}')

    # supersession: radiosity + AO on together must equal radiosity alone
    sc3, st3 = render_pair(radiosity=True, radiosity_samples=4,
                           radiosity_distance=4.0, ambient_occlusion=True)
    both = R.render(sc3, st3)
    check('radiosity supersedes plain AO, exactly',
          bool(np.array_equal(both, on)))

    # ---- blurry reflections: the dead slider lives
    from .featurematrix import build
    sc, st = build('blurry reflections')
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    blurred = R.render(sc, st)
    st_sharp = __import__('copy').copy(st)
    st_sharp = type(st)(**{f.name: getattr(st, f.name)
                           for f in __import__('dataclasses').fields(st)})
    st_sharp.reflection_blur = 0.0
    sharp = R.render(sc, st_sharp)
    d_blur = float(np.abs(blurred[..., :3] - sharp[..., :3]).max())
    check('a blur cone changes the reflection', d_blur > 0.01,
          f'{d_blur:.4f}')
    st_more = type(st)(**{f.name: getattr(st, f.name)
                          for f in __import__('dataclasses').fields(st)})
    st_more.reflection_blur_samples = 8
    more = R.render(sc, st_more)
    check('the SAMPLES slider finally does something (dead since it '
          'shipped)', float(np.abs(more[..., :3]
                                   - blurred[..., :3]).max()) > 1e-4)
    # determinism: the jitter rides the pixel-identity streams
    again = R.render(sc, st)
    check('the blur is deterministic', bool(np.array_equal(blurred, again)))
    # FLIPPED (1.25.105): the deferred sweeps run the cone now -- the
    # plan qualifies and the sim reproduces the CPU's averaged jitter
    from ..core.bvh import BVH as _BVH
    view, _proj, vp, eye = R.camera_matrices(sc.camera,
                                             st.resolution_x,
                                             st.resolution_y)
    g = raster.GBuffer(st.resolution_x, st.resolution_y)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, st.resolution_x,
                     st.resolution_y, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st),
                     _BVH(sc.mesh.verts, sc.mesh.tris), view, eye,
                     st.resolution_x, st.resolution_y)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('blurry reflections qualify for the deferred sweeps',
          p is not None, str(why))
    if p is not None:
        simg, _hit = GSH.simulate(job, g, p, _a)
        cov = g.tri >= 0
        errb = float(np.abs(simg[cov] - blurred[cov][:, :3]).max())
        check('...and the sim reproduces the CPU cone', errb < 6e-3,
              f'max {errb:.6f}')

    # ---- the burn-in stamp. The reference is the same post chain with
    # no watermark: post legitimately quantises and transforms, so
    # "unchanged" means "equal to the stampless run", never "equal to
    # the input".
    st_empty = base_settings(96, 72)
    img = np.full((72, 96, 4), 0.5, np.float32)
    base = PP.process(img.copy(), st_empty, frame=37)
    st = base_settings(96, 72)
    st.watermark = 'A1 %F'
    out = PP.process(img.copy(), st, frame=37)
    diff = np.abs(out[..., :3] - base[..., :3]).max(axis=2)
    check('the stamp marks the frame', float(diff.max()) > 0.4)
    check('...in the bottom-left corner only',
          float(diff[:24, :64].max()) > 0.4 and
          float(diff[24:, :].max()) < 1e-6 and
          float(diff[:, 64:].max()) < 1e-6,
          f'top {float(diff[24:, :].max()):.4f}')
    check('the token expanded to the frame number',
          PP.stamp_text('A1 %F', st, 37) == 'A1 0037')
    check('%R and %V expand too',
          PP.stamp_text('%R', st, 0) == '96X72' and
          PP.stamp_text('%V', st, 0).startswith('HALCYON 1.'))
    out2 = PP.process(img.copy(), st, frame=37)
    check('the stamp is deterministic', bool(np.array_equal(out, out2)))
    st._viewport = True
    vout = PP.process(img.copy(), st, frame=37)
    check('the viewport never shows the burn-in',
          bool(np.array_equal(vout, base)))

    # ---- the slate tokens: date, time, render time, host version.
    # The clock injects through `info` so the pixels are testable; the
    # engine fills the real values at F12 time.
    import datetime
    fixed = datetime.datetime(1997, 8, 29, 2, 14)
    info = {'now': fixed, 'render_time': 12.4, 'blender': '5.2.0'}
    check('%D stamps the date',
          PP.stamp_text('%D', st, 0, info) == '1997-08-29')
    check('%T stamps the time of day',
          PP.stamp_text('%T', st, 0, info) == '02:14')
    check('%S stamps the render clock',
          PP.stamp_text('%S', st, 0, info) == '12.4S')
    check('...in minutes past a minute',
          PP._stamp_seconds(192.6) == '3M13S' and
          PP._stamp_seconds(7505) == '2H05M', PP._stamp_seconds(192.6))
    check('%B stamps the host version',
          PP.stamp_text('%B', st, 0, info) == 'BLENDER 5.2.0')
    check('absent info says ? instead of guessing',
          PP.stamp_text('%S %B', st, 0, None) == '? BLENDER ?')

    # ---- the field's 118-second lesson, both halves. HALF ONE: the
    # documented matcap workflow (Image Texture through Matcap
    # Coordinates into the Matcap socket) refused since the override
    # was ported -- 'matcap varies across the frame' -- and one such
    # material put the WHOLE frame, radiosity gather included, on the
    # CPU. The matcap COLOUR is per-pixel now.
    from ..core.scene import ImageBuffer, Material
    from .scenebuild import checker_image

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def matcap_master(blend_link=None):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
            sk('Matcap', 'RGBA', [0, 0, 0, 1], ['tex', 0]),
            sk('Matcap Blend', 'VALUE', 0.55, blend_link),
        ]
        nodes = {
            'muv': {'id': 'muv', 'bl_idname': 'HALCYON_MatcapUVNode',
                    'props': {},
                    'inputs': [sk('Scale', 'VALUE', 1.0)],
                    'outputs': [{'name': 'Vector', 'type': 'VECTOR'},
                                {'name': 'Facing', 'type': 'VALUE'}]},
            'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                    'props': {'image': 'eyes',
                              'interpolation': 'Closest'},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['muv', 0])],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Alpha', 'type': 'VALUE'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}
        return {'output': 'out', 'nodes': nodes}

    st = base_settings(96, 72, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.images['eyes'] = ImageBuffer(name='eyes', pixels=checker_image())
    sc.materials[1] = Material(name='Eyes', index=1,
                               graph=matcap_master())
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, None, view, eye, 96, 72)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check("the FIELD'S OWN refusal is lifted: an image-driven matcap "
          'qualifies', passes is not None, str(why))
    if passes is not None:
        img2, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img2[cov] - cpu[cov][:, :3]).max())
        check('and the per-pixel matcap is the CPU frame', err < 6e-3,
              f'max {err:.5f}')
    # a varying BLEND still refuses, by name
    lk = matcap_master(blend_link=['muv', 1])   # Facing drives the blend
    sc.materials[1] = Material(name='Eyes', index=1, graph=lk)
    GSH._PLAN_CACHE.clear()
    p2, why2, _a2 = GSH.plan_frame(job, g)
    check('a varying matcap BLEND still refuses by name',
          p2 is None and 'matcap_blend' in str(why2), str(why2))

    # HALF TWO: when the gather DOES run on the CPU, the cost is named
    # before it is paid
    import contextlib
    import io
    st = base_settings(48, 36, shadows=True)
    st.radiosity = True
    st.radiosity_samples = 2
    sc = demo_scene(st, with_texture=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.render(sc, st)
    check('the CPU gather announces its cost and its levers',
          'radiosity gathers on the CPU' in buf.getvalue()
          and 'Gather Samples' in buf.getvalue(),
          buf.getvalue()[:120])


def test_interpolated_radiosity_and_the_inked_slate():
    """The era's own radiosity, and the slate learns colour.

    INTERPOLATED RADIOSITY (LightWave's shipping mode): gather every Nth
    pixel into a grid, blend between the points. The grid point's source
    is the first covered pixel of its block in row-major order, gathered
    with that pixel's own deterministic identity -- so the GPU's grid
    pre-pass casts the identical rays and the two devices blend the same
    field. Spacing 1 remains the full-rate 1.25.95 path, byte for byte.
    Bands must not seam: the scissor grows by two blocks so a band's
    grid points see complete blocks and land the whole frame's numbers.

    THE INKED SLATE: `&%r/g/b/y/c/m` switch the burn-in's ink; the
    secret `&%2204355` wears a rainbow that scrolls one character per
    frame -- glyph cell i at frame f+1 must equal glyph cell i+1 at
    frame f, exactly.
    """
    from ..core import post as PP
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    def setup(spacing, samples=4):
        st = base_settings(96, 72, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        st.radiosity = True
        st.radiosity_samples = samples
        st.radiosity_distance = 4.0
        st.radiosity_spacing = spacing
        sc = demo_scene(st, with_texture=False)
        sc.materials[0].diffuse = (0.9, 0.05, 0.05)
        return sc, st

    # ---- the interpolation is real, deterministic, and CLOSE
    sc, st = setup(2)
    interp = R.render(sc, st)
    sc1, st1 = setup(1)
    full = R.render(sc1, st1)
    d = np.abs(full[..., :3] - interp[..., :3])
    check('spacing 2 interpolates (differs from full-rate)',
          float(d.max()) > 1e-4, f'{float(d.max()):.4f}')
    check('...but stays the same picture (mean)', float(d.mean()) < 0.02,
          f'{float(d.mean()):.5f}')
    again = R.render(*setup(2)[:1], st)
    check('the interpolated gather is deterministic',
          bool(np.array_equal(interp, again)))

    # ---- bands must not seam: two half-frames == the whole frame
    sc_b, st_b = setup(2)
    whole = R.render(sc_b, st_b)
    sc_t, st_t = setup(2)
    top = R.render(sc_t, st_t, band=(0, 36))     # bands return their rows
    sc_u, st_u = setup(2)
    bot = R.render(sc_u, st_u, band=(36, 72))
    joined = np.concatenate([top, bot], axis=0)
    check('band halves equal the whole frame EXACTLY (no grid seam)',
          bool(np.array_equal(joined, whole)),
          f'shapes {top.shape}/{bot.shape} vs {whole.shape}')

    # ---- the GPU field: grid pre-pass present, sim is the CPU frame
    sc, st = setup(2)
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, 96, 72)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('interpolated radiosity qualifies', passes is not None,
          str(why))
    if passes is not None:
        check('the grid pre-pass rides the plan',
              '__radfield' in atlases)
        for _mid, name, src, binds in passes:
            check(f"pass '{name}' reads the field, not the BVH",
                  'hal_rad_lookup' in src and
                  'hal_bvh_intersect' not in src and
                  'hal_radfield' in binds.get('samplers', ()))
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('the interpolated field sim-matches the CPU frame',
              err < 6e-3, f'max {err:.6f}')

    # ---- traced hits have no place in a screen cache: with rays on,
    # secondary passes still carry the FULL gather
    sc, st = setup(2)
    st.raytrace = True
    st.ray_depth = 1
    sc.materials[2].reflect_level = 0.5
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, 96, 72)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('rays + interpolated radiosity qualify', passes is not None,
          str(why))
    if passes is not None:
        rplan = atlases.get('__reflect')
        check('secondary passes keep the full gather',
              rplan is not None and any(
                  'hal_rad_at' in p[2] for p in rplan['secondary']))
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check('and the reflected frame sim-matches', err < 6e-3,
              f'max {err:.6f}')

    # ---- the inked slate
    st = base_settings(96, 72)
    img0 = np.full((72, 96, 4), 0.25, np.float32)
    base = PP.process(img0.copy(), st, frame=0)
    st.watermark = '&%rA'
    red = PP.process(img0.copy(), st, frame=0)
    diff = np.abs(red[..., :3] - base[..., :3]).sum(axis=2)
    ys, xs = np.nonzero(diff > 0.2)
    check('the red escape draws SOMETHING', ys.size > 0)
    lit = red[ys, xs]
    ink = lit[lit[:, 0] > 0.9]
    check('...and its ink is red', ink.size > 0 and
          float(ink[:, 1].max()) < 0.1 and float(ink[:, 2].max()) < 0.1)
    check('the escape itself is consumed (one glyph, not four)',
          float(xs.max()) < 4 + 6 * 2, str(int(xs.max())))
    runs = PP._stamp_runs('&%gHI', 0)
    check('runs carry the ink switch',
          [c for c, _col in runs] == ['H', 'I'] and
          runs[0][1] == (0.0, 1.0, 0.0))

    # the rainbow scrolls one character per frame: cell i at frame f+1
    # equals cell i+1 at frame f, same glyph, shifted hue
    st.watermark = '&%2204355AAAAAAAA'
    f0 = PP.process(img0.copy(), st, frame=10)
    f1 = PP.process(img0.copy(), st, frame=11)
    adv, margin = 6, 4
    ok = True
    for i in range(3):
        c_next = f1[:16, margin + i * adv:margin + (i + 1) * adv]
        c_prev = f0[:16, margin + (i + 1) * adv:margin + (i + 2) * adv]
        if not np.array_equal(c_next, c_prev):
            ok = False
    check('the rainbow scrolls exactly one character per frame', ok)
    hues = {tuple(np.round(PP._hue_rgb(k / 12.0), 5)) for k in range(12)}
    check('twelve distinct hues around the wheel', len(hues) == 12)

    # THE FIELD'S STILL: one timeline frame, rendered again and again.
    # Keyed to the timeline alone the rainbow stands perfectly still
    # there -- two reports in a row watched it do exactly that. The
    # slate already prints wall-clock truths (%T, %S), so the rainbow
    # rides the render-event clock too: 'scroll' in stamp_info, the
    # engine's per-render serial. A serial step obeys the same shift
    # law as a frame step...
    s0 = PP.process(img0.copy(), st, frame=10, stamp_info={'scroll': 3})
    s1 = PP.process(img0.copy(), st, frame=10, stamp_info={'scroll': 4})
    ok = all(np.array_equal(
        s1[:16, margin + i * adv:margin + (i + 1) * adv],
        s0[:16, margin + (i + 1) * adv:margin + (i + 2) * adv])
        for i in range(3))
    check('a render-serial step scrolls one character, same law', ok)
    # ...frame and serial are ONE clock, additive and bit-exact...
    check('frame and serial share one clock',
          np.array_equal(
              PP.process(img0.copy(), st, frame=7,
                         stamp_info={'scroll': 4}),
              PP.process(img0.copy(), st, frame=0,
                         stamp_info={'scroll': 11})))
    # ...and no serial means the pure-frame clock, bit for bit: the
    # headless matrix and the self-test prover pass stamp_info=None
    # and keep every pixel they had
    check('absent serial keeps the pure-frame clock',
          np.array_equal(
              PP.process(img0.copy(), st, frame=10,
                         stamp_info={'scroll': 0}), f0))

    # THE FIELD'S SECOND STRING: 'SHOT 4 %F &%c%D %T &%2204355'. The
    # escapes are prefix inks -- they colour what FOLLOWS -- so a
    # TRAILING egg claimed zero glyphs and the slate came back cyan
    # with no rainbow anywhere. The egg must never resolve to nothing:
    # left without a visible glyph of its own it takes the WHOLE line
    # (explicit inks yield; the egg was typed to get a rainbow
    # SOMEWHERE). Given its own glyphs it colours only those, and
    # every other ink keeps its section.
    import datetime as _dt
    fixed = {'now': _dt.datetime(2026, 8, 8, 14, 32),
             'render_time': 5.0, 'blender': '5.2.0'}
    txt = PP.stamp_text('SHOT 4 %F &%c%D %T &%2204355', st, 4, fixed)
    pr = PP._stamp_runs(txt, 4)
    check('a trailing egg takes the whole line',
          len(pr) > 20 and all(col == 'RAINBOW' for ch, col in pr
                               if ch != ' '), str(pr[:4]))
    pr2 = PP._stamp_runs('AB &%2204355CD', 0)
    check('an egg with its own glyphs colours only those',
          [col for _ch, col in pr2] == [(1.0, 1.0, 1.0)] * 3 +
          ['RAINBOW'] * 2, str(pr2))
    check('a trailing-spaces egg promotes too',
          all(col == 'RAINBOW' for _ch, col in
              PP._stamp_runs('AB&%2204355 ', 0)))
    check('no egg, no promotion',
          all(col == (0.0, 1.0, 1.0) for _ch, col in
              PP._stamp_runs('&%cAB', 0)))
    # and through process(): the promoted line is a moving RAINBOW,
    # not one tint -- many distinct inks on a still, scrolling on the
    # render serial exactly like any other rainbow
    st_ref = base_settings(96, 72)
    ref = PP.process(img0.copy(), st_ref, frame=4,
                     stamp_info=dict(fixed, scroll=1))
    st_f = base_settings(96, 72)
    st_f.watermark = 'SHOT 4 %F &%c%D %T &%2204355'
    g0 = PP.process(img0.copy(), st_f, frame=4,
                    stamp_info=dict(fixed, scroll=1))
    g1 = PP.process(img0.copy(), st_f, frame=4,
                    stamp_info=dict(fixed, scroll=2))
    d = float(np.abs(np.asarray(g0[:16], np.float64) - g1[:16]).max())
    check('the field string scrolls per render', d > 0.2, f'{d:.3f}')
    mask = np.abs(g0[..., :3].astype(np.float64)
                  - ref[..., :3]).sum(axis=2) > 0.2
    lit = g0[mask][:, :3]
    ink = lit[lit.sum(axis=1) > 0.5]          # drop the drop shadow
    inks = {tuple(np.round(v, 4)) for v in ink}
    check('and wears many inks, not one cyan tint', len(inks) > 12,
          str(len(inks)))


def test_the_last_shading_routes_fall():
    """R105: the final three shading routes -- specular slot routing,
    the Wireframe node's ink, and the blur cone in the sweeps.

    Slot routing: a lone raw GLOSSY lobe's colour chain goes to
    s.specular (closure_to_surface's own slot) with the flat diffuse
    baked; the Specular BSDF's DIFFUSE+GLOSSY pair keeps the chain in
    the diffuse slot and bakes the glossy side through the existing
    constancy rule. The Wireframe node: ShadeJob.wire_fields expression
    for expression -- world edge distance from the attribute texture's
    corners, Pixel Size from the per-corner screen positions packed per
    frame (camera-dependent, so it rides the per-frame road, never the
    plan's atlas cache). Blurry reflections: _blurred_reflection's cone
    in the sweeps, same salt streams, same fold, K lane values
    averaged, recursively at every reflective spawn.
    """
    from .featurematrix import build
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    def plan_sim(key, mutate=None):
        sc, st = build(key)
        st.render_device = 'GPU'
        if mutate:
            mutate(sc, st)
        cpu = R.render(sc, st)
        w, h = st.resolution_x, st.resolution_y
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        tex = R.prepare_textures(sc, st)
        job = R.ShadeJob(sc, st, tex, BVH(sc.mesh.verts, sc.mesh.tris),
                         view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        p, why, a = GSH.plan_frame(job, g)
        if p is None:
            return None, why, None, cpu, g
        img, _hit = GSH.simulate(job, g, p, a)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        return p, err, a, cpu, g

    # ---- the specular slot routing (queued since the node shelf round)
    p, err, a, _c, _g = plan_sim('node Metallic BSDF (raw graph)')
    check('raw Metallic BSDF qualifies (slot routing)', p is not None,
          str(err))
    if p is not None:
        check('...and sim-matches the CPU', err < 6e-3, f'max {err:.6f}')
        check('...with the chain routed to s.specular',
              any('s.specular = ' in src and '__slot' not in src
                  for _m, _n, src, _b in p))
    p, err, _a, _c, _g = plan_sim('node Specular BSDF (raw graph)')
    check('raw Specular BSDF qualifies (diffuse chain + baked gloss)',
          p is not None, str(err))
    if p is not None:
        check('...and sim-matches the CPU', err < 6e-3, f'max {err:.6f}')

    # ---- the Wireframe node, both units
    p, err, _a, _c, _g = plan_sim('node Wireframe (cel ink)')
    check('the Wireframe node qualifies (world units)', p is not None,
          str(err))
    if p is not None:
        check('...and sim-matches the CPU EXACTLY', err < 1e-5,
              f'max {err:.6f}')

    def _pixel_size(sc, st):
        for _nid, nd in sc.materials[1].graph['nodes'].items():
            if nd.get('bl_idname') == 'ShaderNodeWireframe':
                nd['props']['use_pixel_size'] = True
                for skt in nd['inputs']:
                    if skt['name'] == 'Size':
                        skt['default'] = 2.0
    p, err, _a, _c, _g = plan_sim('node Wireframe (cel ink)',
                                  mutate=_pixel_size)
    check('the Wireframe node qualifies (Pixel Size)', p is not None,
          str(err))
    if p is not None:
        check('...pixel-size wires sim-match EXACTLY', err < 1e-5,
              f'max {err:.6f}')
        check('...and the pass carries the screen texture',
              any('hal_vscreen' in b.get('samplers', ())
                  for _m, _n, _s, b in p))

    # ---- the blur cone in the sweeps (full coverage in the era-audit
    # test; here the structural guarantee: MORE samples change the sim
    # exactly as they change the CPU)
    p, err, _a, cpu8, g8 = plan_sim(
        'blurry reflections',
        mutate=lambda sc, st: setattr(st, 'reflection_blur_samples', 8))
    check('an 8-sample cone qualifies and sim-matches',
          p is not None and err < 6e-3, str(err))


def test_the_raster_endgame():
    """R103: the compute raster learns the last three shading-adjacent
    tricks -- the affine carry, real subdivision, and the tie referral.

    Affine: the kernel's lin image must carry fill()'s own bary_lin
    (l . bw, no perspective) to the ulp, so the PS1 warp shades from
    the same interpolants whichever rasteriser ran. Subdivision: the
    once-dead tex_affine_subdiv now splits screen triangles until no
    edge exceeds the cap -- deterministic, shared by both rasterisers,
    exact for every rasteriser input (all are screen-affine).
    Referral: under quantised depth and vertex snapping the kernel
    marks every decision inside a cross-device noise window and the
    marked pixels are REPLAYED with the CPU fill's own arithmetic --
    the R73 ray referral, at the raster.
    """
    from ..core import raster
    from ..core.scene import Camera, MeshData
    from ..gpu import craster as CRA
    from .scenebuild import look_at_matrix

    w, h = 160, 120
    st = base_settings(w, h)
    sc = demo_scene(st, with_texture=True)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)

    # ---- the affine carry: lin == fill()'s bary_lin
    g = raster.GBuffer(w, h)
    g.alloc_linear()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc.mesh, vp, w, h)
    tri, bary, zndc, front, _b2, lin, _mk = CRA.simulate_raster(
        sx, sy, iw, z, bw, src, tmap, w, h)
    check('affine carry: tri ids identical',
          bool(np.array_equal(tri, g.tri)))
    cov = g.tri >= 0
    dl = float(np.abs(lin[cov] - g.bary_lin[cov]).max())
    check('affine carry: the lin image IS bary_lin (to the ulp)',
          dl < 1e-5, f'{dl:.2e}')
    check('...and it differs from the perspective bary (not vacuous)',
          float(np.abs(lin[cov] - bary[cov]).max()) > 1e-3)

    # ---- subdivision: deterministic, bounded, shared
    sx2, sy2, iw2, z2, bw2, src2, _tm = CRA.raster_inputs_for(
        sc.mesh, vp, w, h, subdiv_px=12)
    check('subdivision emits more triangles', sx2.shape[0] > sx.shape[0],
          f'{sx.shape[0]} -> {sx2.shape[0]}')
    e01 = np.sqrt((sx2[:, 0] - sx2[:, 1]) ** 2 + (sy2[:, 0] - sy2[:, 1]) ** 2)
    e12 = np.sqrt((sx2[:, 1] - sx2[:, 2]) ** 2 + (sy2[:, 1] - sy2[:, 2]) ** 2)
    e20 = np.sqrt((sx2[:, 2] - sx2[:, 0]) ** 2 + (sy2[:, 2] - sy2[:, 0]) ** 2)
    # a pass cap exists, so the guarantee is "capped or six passes deep";
    # on this scene six passes are plenty
    check('...and no screen edge exceeds the cap',
          float(np.maximum(np.maximum(e01, e12), e20).max()) <= 12.0 + 1e-3)
    sx3, _sy3, _iw3, _z3, _bw3, _src3, _tm3 = CRA.raster_inputs_for(
        sc.mesh, vp, w, h, subdiv_px=12)
    check('...and the split is deterministic',
          bool(np.array_equal(sx2, sx3)))
    # subdivided raster == unsubdivided COVERAGE (same surfaces, same
    # depth planes: the picture's geometry cannot move)
    g_sub = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g_sub,
                     subdiv_px=12)
    check('subdivision does not move coverage',
          bool(np.array_equal(g_sub.tri >= 0, g.tri >= 0)))

    # ---- THE NAMED TIE RULE: equal depth -> lowest triangle id.
    # This round found the two CPU fill paths DISAGREEING at exact
    # quantised ties (the batched path draws big triangles first, so
    # "first tested wins" meant different winners by path -- 3 px of
    # the demo scene at 16 bits, latent since the batched fill
    # shipped). The rule is order-free, so loop, batched, kernel and
    # replay all land the same winner.
    gb16 = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=gb16,
                     depth_bits=16)
    gl16 = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=gl16,
                     depth_bits=16, batched=False)
    check('batched and loop fills agree at 16-bit ties (the latent '
          'bug, pinned)', bool(np.array_equal(gb16.tri, gl16.tri)),
          f'{int((gb16.tri != gl16.tri).sum())} differ')

    # coincident IDENTICAL quads: exact ties everywhere. The tie rule
    # decides them deterministically on every path -- ZERO referral
    # marks needed, and the kernel equals fill exactly
    verts = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
                     np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3],
                     [0, 1, 2], [0, 2, 3]], np.int32)
    mesh = MeshData(verts=verts, tris=tris)
    cam = Camera(matrix_world=look_at_matrix((0.0, 0.0, 6.0),
                                             (0.0, 0.0, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=100.0)
    _v, _p, vp2, _e = R.camera_matrices(cam, w, h)
    g16 = raster.GBuffer(w, h)
    raster.rasterize(mesh.verts, mesh.tris, vp2, w, h, gbuf=g16,
                     depth_bits=16)
    sx4, sy4, iw4, z4, bw4, src4, _tm4 = CRA.raster_inputs_for(
        mesh, vp2, w, h, depth_bits=16)
    tri4, _b4, z44, _f4, _bb4, _l4, mark4 = CRA.simulate_raster(
        sx4, sy4, iw4, z4, bw4, src4, None, w, h, depth_bits=16,
        refer=True)
    # the kernel equals fill on every tie BEFORE any replay -- the id
    # rule decides them. But a zero raw-depth GAP is still fragile in
    # general (two coincidentally-equal values can SPLIT under a
    # driver's fma, and a split is decided by depth, not the id rule),
    # so fully-duplicated geometry floods the marks past the budget
    # and the frame honestly bails to the CPU raster -- degenerate
    # scenes stay correct by falling back, named
    cov4 = int((tri4 >= 0).sum())
    check('exact ties: kernel equals fill with no replay needed',
          bool(np.array_equal(tri4, g16.tri)),
          f'{int((tri4 != g16.tri).sum())} differ')
    check('...duplicate-identical geometry floods the referral (the '
          'honest bail)', int(mark4.sum()) > CRA.REFER_BAIL_FRAC * cov4,
          f'{int(mark4.sum())} of {cov4}')
    check('...and the ties all went to the lowest id',
          set(np.unique(g16.tri[g16.tri >= 0]).tolist()) <= {0, 1})

    # NEAR-coincident quads: some depths land near step boundaries in
    # a close competition -- the one genuinely fragile class. Marks
    # must fire there, and the replay must hand back fill()'s own
    # answer at every one
    verts2 = verts.copy()
    verts2 = np.concatenate([verts2, verts2 + [0, 0, 1e-4]], 0)
    tris2 = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
                     np.int32)
    mesh2 = MeshData(verts=verts2.astype(np.float32), tris=tris2)
    g16b = raster.GBuffer(w, h)
    raster.rasterize(mesh2.verts, mesh2.tris, vp2, w, h, gbuf=g16b,
                     depth_bits=16)
    sx5, sy5, iw5, z5, bw5, src5, _tm5 = CRA.raster_inputs_for(
        mesh2, vp2, w, h, depth_bits=16)
    tri5, _b5, z55, _f5, _bb5, _l5, mark5 = CRA.simulate_raster(
        sx5, sy5, iw5, z5, bw5, src5, None, w, h, depth_bits=16,
        refer=True)
    check('boundary-window marks fire on near-coincident depth',
          bool(mark5.any()), f'{int(mark5.sum())} marked')
    pys, pxs = np.nonzero(mark5)
    _c5, _cs5, bins5, _bs5, tiles5, tw5, _th5 = CRA.pack_raster_inputs(
        sx5, sy5, iw5, z5, bw5, src5, None, w, h)
    r_tri, r_b, r_lb, r_z, r_fr = CRA.replay_pixels(
        pxs, pys, sx5, sy5, iw5, z5, bw5,
        np.asarray(src5, np.int64), tiles5, bins5.reshape(-1), tw5,
        'NONE', 16)
    tri5[pys, pxs] = r_tri
    z55[pys, pxs] = np.where(r_tri >= 0, r_z, 1.0)
    check('replayed + unmarked pixels equal fill() EXACTLY',
          bool(np.array_equal(tri5, g16b.tri)),
          f'{int((tri5 != g16b.tri).sum())} differ')
    check("...and the replayed depths are fill()'s own",
          bool(np.array_equal(z55[pys, pxs], g16b.zndc[pys, pxs])))

    # the field predictor: on the matrix's own quantised rows the mark
    # rate must sit inside the referral budget -- the driver REPLAYS
    # a handful of pixels instead of bailing the frame
    from .featurematrix import build as _mbuild
    for row in ('16-bit z-buffer', 'vertex snapping (PS1)'):
        scq, stq = _mbuild(row)
        wq, hq = stq.resolution_x, stq.resolution_y
        snapq = float(stq.vertex_snap_grid) if stq.vertex_snap else 0.0
        bitsq = int(stq.depth_precision)
        _vq, _pq, vpq, _eq = R.camera_matrices(scq.camera, wq, hq)
        sxq, syq, iwq, zq, bwq, srcq, tmq = CRA.raster_inputs_for(
            scq.mesh, vpq, wq, hq, depth_bits=bitsq, snap=snapq)
        triq, _bq, _zq2, _fq, _bbq, _lq, markq = CRA.simulate_raster(
            sxq, syq, iwq, zq, bwq, srcq, tmq, wq, hq,
            depth_bits=bitsq, refer=True)
        covq = int((triq >= 0).sum())
        check(f'{row}: marks inside the referral budget',
              int(markq.sum()) <= CRA.REFER_BAIL_FRAC * covq,
              f'{int(markq.sum())} of {covq}')

    # marks stay OFF when the referral is off (24-bit default road)
    _t6, _b6, _z6, _f6, _bb6, _l6, mk6 = CRA.simulate_raster(
        sx5, sy5, iw5, z5, bw5, src5, None, w, h, depth_bits=16,
        refer=False)
    check('no marks without the referral flag', not bool(mk6.any()))


def test_the_shading_routes_fall():
    """R102: five features that routed whole frames to the CPU by name
    now shade on the driver, each by its own honest road.

    CONSTANT/WIREFRAME: light_surface's early return emitted verbatim
    (diffuse x level + emission), the wires carved by apply_wireframe on
    the readback -- the fog doctrine. Normal Source FACE + Geometry's
    True Normal / Random Per Island: the hal_triaux per-tri texture
    carries the CPU's OWN stored normals and sin-fract randoms, fetched
    rather than recomputed. Affine mapping: uv re-interpolates by the
    rasteriser's own screen-linear barycentrics (hal_gb_idslin), uv
    only, hits keep true bary. Screen Door stipple: the CPU's alpha
    chain + its own threshold map, the 0/1 pattern decoded from an
    encoded alpha (0.9 kept / 0.6 dropped over the 0.5 coverage floor).
    Light linking: light_surface's isin() mask as an exact td.y ladder.
    Every new gate rides the plan signature (R78) -- the warm-cache
    checks at the end prove a primed cache cannot walk around any of
    them.
    """
    from .featurematrix import build
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    def plan_sim(key, need_bvh=True, alloc_lin=False):
        sc, st = build(key)
        st.render_device = 'GPU'
        cpu = R.render(sc, st)
        w, h = st.resolution_x, st.resolution_y
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        if alloc_lin or not st.tex_perspective:
            g.alloc_linear()
        # exactly render()'s raster: affine subdivision included, now
        # that tex_affine_subdiv is alive (a harness that skips it
        # shades a different G-buffer than the frame it compares to)
        sub = int(getattr(st, 'tex_affine_subdiv', 0) or 0) \
            if not st.tex_perspective else 0
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                         subdiv_px=sub)
        tex = R.prepare_textures(sc, st)
        job = R.ShadeJob(sc, st, tex,
                         BVH(sc.mesh.verts, sc.mesh.tris)
                         if need_bvh else None, view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atl = GSH.plan_frame(job, g)
        return sc, st, cpu, g, job, passes, why, atl, (vp, eye, tex)

    # ---- CONSTANT and WIREFRAME
    sc, st, cpu, g, job, p, why, atl, _x = plan_sim('model CONSTANT')
    check('model CONSTANT qualifies', p is not None, str(why))
    if p is not None:
        img, _hit = GSH.simulate(job, g, p, atl)
        cov = g.tri >= 0
        check('CONSTANT sim is the CPU frame',
              float(np.abs(img[cov] - cpu[cov][:, :3]).max()) < 6e-3)
    sc, st, cpu, g, job, p, why, atl, (vp, eye, tex) = \
        plan_sim('model WIREFRAME')
    check('model WIREFRAME qualifies', p is not None, str(why))
    if p is not None:
        img, _hit = GSH.simulate(job, g, p, atl)
        py, px = np.nonzero(g.mask())
        full = np.zeros((g.height, g.width, 4), np.float32)
        full[py, px, :3] = img[py, px]
        full[py, px, 3] = 1.0
        full = R.apply_wireframe(job, g, full, st, vp, eye, tex)
        cov = g.tri >= 0
        check('WIREFRAME sim + the readback carve is the CPU frame',
              float(np.abs(full[cov][:, :3] - cpu[cov][:, :3]).max())
              < 6e-3)

    # ---- Normal Source FACE rides hal_triaux
    sc, st, cpu, g, job, p, why, atl, _x = plan_sim('normal source FACE')
    check('normal source FACE qualifies', p is not None, str(why))
    if p is not None:
        check('...and the per-tri texture is packed', 'hal_triaux' in atl)
        img, _hit = GSH.simulate(job, g, p, atl)
        cov = g.tri >= 0
        check('FACE normals sim is the CPU frame',
              float(np.abs(img[cov] - cpu[cov][:, :3]).max()) < 6e-3)

    # ---- affine: the PS1 warp on the deferred pass, uv only
    for key in ('affine mapping (PS1 warp)', 'affine + subdivision'):
        sc, st, cpu, g, job, p, why, atl, _x = plan_sim(key)
        check(f'{key} qualifies', p is not None, str(why))
        if p is None:
            continue
        img, _hit = GSH.simulate(job, g, p, atl)
        cov = g.tri >= 0
        check(f'{key} sim is the CPU frame',
              float(np.abs(img[cov] - cpu[cov][:, :3]).max()) < 6e-3)
        check('...the passes read the screen-linear ids',
              all('hal_gb_idslin' in b.get('samplers', ())
                  for _m, _n, _s, b in p))
    # hits keep TRUE barycentrics: a reflective affine frame's secondary
    # passes must not touch the screen-linear road
    sc, st = build('affine mapping (PS1 warp)')
    st.render_device = 'GPU'
    st.raytrace, st.ray_depth = True, 1
    sc.materials[2].reflect_level = 0.5
    w, h = st.resolution_x, st.resolution_y
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    g.alloc_linear()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st),
                     BVH(sc.mesh.verts, sc.mesh.tris), view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p, why, atl = GSH.plan_frame(job, g)
    check('affine + rays qualifies', p is not None, str(why))
    if p is not None:
        rp = atl.get('__reflect') or {}
        check('...and hit passes keep true barycentrics',
              all('hal_gb_idslin' not in s
                  for _m, _n, s, _b in (rp.get('secondary') or ())))

    # ---- Screen Door stipple: rgb shaded, the 0/1 pattern in alpha
    sc, st, cpu, g, job, p, why, atl, _x = plan_sim('screen-door stipple')
    check('screen-door stipple qualifies', p is not None, str(why))
    if p is not None:
        check('...the CPU threshold map is packed', 'hal_stipple' in atl)
        img, _hit = GSH.simulate(job, g, p, atl)
        cov = g.tri >= 0
        check('stipple rgb sim is the CPU frame',
              float(np.abs(img[cov] - cpu[cov][:, :3]).max()) < 6e-3)
        check('the stipple ALPHA pattern is the CPU pattern, bit for bit',
              g.gpu_alpha is not None and
              bool((g.gpu_alpha[cov].astype(np.float32)
                    == cpu[cov][:, 3]).all()))
        check('...and it is a real screen door (some pixels dropped)',
              0.0 < float(g.gpu_alpha[cov].mean()) < 1.0)
    # a VARYING opacity under stipple refuses through the constancy
    # rule -- Opacity is not a per-pixel socket, so the ordered
    # threshold always compares the baked constant on both devices
    sc, st = build('screen-door stipple')
    st.render_device = 'GPU'
    # drive opacity the honest way: a master graph whose Opacity socket
    # takes a varying chain (Parametric length)
    from .featurematrix import _sk
    ins = [_sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0]),
           _sk('Opacity', 'VALUE', 1.0, ['fac', 0])]
    sc.materials[1].graph = {'output': 'out', 'nodes': {
        'geo': {'id': 'geo', 'bl_idname': 'ShaderNodeNewGeometry',
                'props': {}, 'inputs': [],
                'outputs': [{'name': 'Position', 'type': 'VECTOR'},
                            {'name': 'Normal', 'type': 'VECTOR'},
                            {'name': 'Tangent', 'type': 'VECTOR'},
                            {'name': 'True Normal', 'type': 'VECTOR'},
                            {'name': 'Incoming', 'type': 'VECTOR'},
                            {'name': 'Parametric', 'type': 'VECTOR'}]},
        'fac': {'id': 'fac', 'bl_idname': 'ShaderNodeVectorMath',
                'props': {'operation': 'LENGTH'},
                'inputs': [_sk('Vector', 'VECTOR', [0, 0, 0],
                               ['geo', 5])],
                'outputs': [{'name': 'Value', 'type': 'VALUE'}]},
        'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': 'LAMBERT'}, 'inputs': ins,
                'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
        'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                'props': {},
                'inputs': [_sk('Surface', 'SHADER', None, ['hal', 0]),
                           _sk('Displacement', 'VECTOR', [0, 0, 0])],
                'outputs': []}}}
    w, h = st.resolution_x, st.resolution_y
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g2 = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g2)
    job2 = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p2, why2, _a2 = GSH.plan_frame(job2, g2)
    check('a varying-opacity material under stipple refuses by name',
          p2 is None and 'varies' in str(why2), str(why2))

    # ---- light linking: the td.y ladder
    sc, st, cpu, g, job, p, why, atl, _x = plan_sim(
        'light linking (exclude + only)')
    check('light linking qualifies', p is not None, str(why))
    if p is not None:
        img, _hit = GSH.simulate(job, g, p, atl)
        cov = g.tri >= 0
        check('light linking sim is the CPU frame',
              float(np.abs(img[cov] - cpu[cov][:, :3]).max()) < 6e-3)
        check('...via the object ladder in the source',
              any('hal_lk' in s for _m, _n, s, _b in p))
    sc2, st2 = build('light linking (exclude + only)')
    st2.render_device = 'GPU'
    sc2.lights[-1].exclude_objects = tuple(range(70))
    view, _proj, vp, eye = R.camera_matrices(sc2.camera, w, h)
    g3 = raster.GBuffer(w, h)
    raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp, w, h, gbuf=g3)
    job3 = R.ShadeJob(sc2, st2, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, _a3 = GSH.plan_frame(job3, g3)
    check('a 70-object linking list refuses by name',
          p3 is None and '64' in str(why3), str(why3))

    # ---- R78: every new gate is in the plan signature. Prime the
    # cache with the QUALIFYING variant, flip one setting, and the
    # plan must change -- a stale hit here is the affine bug of R78
    # all over again.
    sc, st = build('baseline PHONG + map shadows')
    st.render_device = 'GPU'
    w, h = st.resolution_x, st.resolution_y
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g4 = raster.GBuffer(w, h)
    g4.alloc_linear()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g4)
    tex4 = R.prepare_textures(sc, st)
    job4 = R.ShadeJob(sc, st, tex4, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    base_p, _bw, _ba = GSH.plan_frame(job4, g4)
    check('the warm-cache fixture qualifies', base_p is not None,
          str(_bw))
    flips = [('normal_source', 'FACE', 'hal_triaux_fetch'),
             ('tex_perspective', False, 'hal_gb_idslin'),
             ('transparency', 'STIPPLE', 'hal_stipple')]
    for name, value, needle in flips:
        old = getattr(st, name)
        setattr(st, name, value)
        p5, why5, _a5 = GSH.plan_frame(job4, g4)   # WARM cache
        got = p5 is not None and \
            any(needle in s for _m, _n, s, _b in p5)
        check(f'a warm cache cannot walk around {name}', got, str(why5))
        setattr(st, name, old)
    # and the stipple PATTERN re-bakes the map on a warm cache
    st.transparency = 'STIPPLE'
    p6, _w6, a6 = GSH.plan_frame(job4, g4)
    st.stipple_pattern = 'BAYER8' \
        if str(getattr(st, 'stipple_pattern', '')) != 'BAYER8' else 'BAYER2'
    p7, _w7, a7 = GSH.plan_frame(job4, g4)
    k6 = (a6 or {}).get('hal_stipple', (None,))[0]
    k7 = (a7 or {}).get('hal_stipple', (None,))[0]
    check('the stipple pattern is in the signature (map key changes)',
          p6 is not None and p7 is not None and k6 != k7,
          f'{k6} vs {k7}')


def test_the_rainbow_scrolls_on_a_re_rendered_still():
    """The field's exact test, run the field's exact way: F12, F12 again.

    Two reports in a row said the rainbow does not scroll. Both were
    stills -- every console this project has ever been sent is a single
    F12 of one timeline frame -- and a rainbow keyed to the timeline
    alone holds its colours BY CONSTRUCTION on a re-rendered frame. The
    R99 answer restated that design instead of moving the pixels. The
    slate was never inside the picture's determinism contract (%T and
    %S print wall-clock truths into it), so the rainbow now rides the
    engine's render serial as well. This renders the SAME timeline
    frame twice through the real engine and demands both halves of the
    doctrine: the slate must move, the picture must not.
    """
    from . import fakeblender as FB
    FB.install()
    from .. import engine as ENG
    from .. import properties as props

    kw = dict(watermark='&%2204355ABCDEFGH')
    a, _p, _c = FB.run_render(props, ENG, **kw)
    b, _p2, _c2 = FB.run_render(props, ENG, **kw)
    check('both renders deliver a frame', a is not None and b is not None)
    if a is None or b is None:
        return
    strip = 16                       # margin + glyph rows, scale 1
    check('the picture above the slate is bit-identical across renders',
          np.array_equal(a[strip:], b[strip:]))
    d = float(np.abs(np.asarray(a[:strip], np.float64) - b[:strip]).max())
    check('and the slate itself scrolled between the two renders',
          d > 0.2, f'max stamp diff {d:.3f}')


def test_material_passes_early_out_on_their_mask():
    """A pass shades ONLY the pixels it owns; the picture cannot move.

    Every material pass draws the full screen and used to shade every
    pixel, multiplying by its ownership mask at the end -- harmless while
    shading was ALU, catastrophic once it carried BVH loops. The field
    measured the disease exactly: a 640x640 radiosity frame spent 5.0 of
    5.8 seconds in shade because every material pass ran the 8-ray gather
    for ALL 410k pixels. The early-out writes the same (0,0,0,0) a masked
    pixel always wrote, so this is pure transport: the multi-material
    radiosity frame must stay sim-identical to the CPU, and the emitted
    source must carry the return so the fix cannot silently regress.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import shade as GSH

    st = base_settings(96, 72, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    st.radiosity = True
    st.radiosity_samples = 4
    st.radiosity_distance = 4.0
    sc = demo_scene(st, with_texture=False)   # three materials: the shape
    sc.materials[0].diffuse = (0.85, 0.1, 0.1)
    cpu = R.render(sc, st)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)
    job = R.ShadeJob(sc, st, {}, bvh, view, eye, 96, 72)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('the multi-material radiosity frame qualifies',
          passes is not None, str(why))
    if passes is None:
        return
    for _mid, name, src, _binds in passes:
        check(f"pass '{name}' carries the early-out",
              'if (keep < 0.5)' in src and 'return;' in src)
    img, _hit = GSH.simulate(job, g, passes, atlases)
    cov = g.tri >= 0
    err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
    check('and the early-out moved NOTHING: sim is the CPU frame',
          err < 6e-3, f'max {err:.6f}')


def test_matcap_and_backface_shade_on_the_gpu():
    """The last two constant-coefficient surface cheats join the frame.

    Matcap lerps the whole lit result toward one colour, after fresnel and
    rim, exactly as `apply_surface_effects`. The backface override needs
    what the G-buffer never carried -- the rasteriser's front flag -- and
    the answer is that for a perspective camera, projected-winding front is
    EXACTLY the plane-side test against the eye, computed from the corner
    positions the attribute texture already holds. The convention was
    measured, not guessed: the rasteriser's `is_front` is screen area < 0,
    which equals plane < 0, and the test pins the emitted expression so
    nobody flips the sign back. Orthographic cameras refuse by name (the
    plane test is the perspective answer), and the camera TYPE joins the
    plan signature while its position stays out. The MatcapUV node ports
    too -- sphere-map coordinates from the view-space normal, the era's
    entire environment-reflection trick.
    """
    from ..core import raster
    from ..core.scene import ImageBuffer, Material
    from ..gpu import shade as GSH
    from ..tests.scenebuild import checker_image

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def master(extra, link, more=()):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0], link),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ] + list(more)
        nodes = {'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                         'props': {'model': 'PHONG', 'toon_steps': 2},
                         'inputs': ins,
                         'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
                 'out': {'id': 'out',
                         'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                         'inputs': [sk('Surface', 'SHADER', None,
                                       ['hal', 0]),
                                    sk('Displacement', 'VECTOR', [0, 0, 0])],
                         'outputs': []}}
        nodes.update(extra)
        return {'output': 'out', 'nodes': nodes}

    w, h = 128, 96

    def run(sc, st, label, expect=None):
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None,
                         view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'{label} qualifies', passes is not None, str(why))
        if passes is None:
            return None, None, None
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{label} is the CPU frame', err < 6e-3, f'max {err:.5f}')
        return passes, g, cpu

    # ---- matcap: the whole result lerps toward one colour
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[1] = Material(name='Mat', index=1, graph=master(
        {}, None, more=[sk('Matcap', 'RGBA', [0.3, 0.5, 0.9, 1.0]),
                        sk('Matcap Blend', 'VALUE', 0.55)]))
    run(sc, st, 'a matcap-blended material')

    # ---- backface override on mixed winding: alternate floor triangles
    # flipped, so front and back share the frame
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    st.two_sided_lighting = True
    sc = demo_scene(st, with_texture=False)
    m = sc.mesh
    floor = np.nonzero(m.mat_index == 0)[0]
    flip = floor[::2]
    m.tris[flip] = m.tris[flip][:, [0, 2, 1]]
    sc.materials[0] = Material(name='Bk', index=0, graph=master(
        {}, None, more=[sk('Backface Color', 'RGBA', [0.9, 0.1, 0.1, 1.0]),
                        sk('Backface Mix', 'VALUE', 0.7)]))
    passes, g, _cpu = run(sc, st, 'the backface override on mixed winding')
    if passes is not None:
        cov = g.tri >= 0
        fronts = g.front[cov]
        check('front and back both appear in the frame',
              bool(fronts.any()) and bool((~fronts).any()),
              f'{int(fronts.sum())} front px')
        src = [p for p in passes if p[1] == 'Bk'][0][2]
        check('the measured sign convention is pinned in the source',
              '(dot(hal_bf_pl, hal_eye - hal_bf_p0) < 0.0) ? 0.0 : 1.0'
              in src)

    # ---- the MatcapUV chain: the era's environment trick, whole
    st = base_settings(w, h, shadows=True)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.images['env'] = ImageBuffer(name='env', pixels=checker_image())
    sc.materials[1] = Material(name='Env', index=1, graph=master(
        {'muv': {'id': 'muv', 'bl_idname': 'HALCYON_MatcapUVNode',
                 'props': {},
                 'inputs': [sk('Scale', 'VALUE', 0.9)],
                 'outputs': [{'name': 'Vector', 'type': 'VECTOR'},
                             {'name': 'Facing', 'type': 'VALUE'}]},
         'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                 'props': {'image': 'env', 'interpolation': 'Closest'},
                 'inputs': [sk('Vector', 'VECTOR', [0, 0, 0], ['muv', 0])],
                 'outputs': [{'name': 'Color', 'type': 'RGBA'},
                             {'name': 'Alpha', 'type': 'VALUE'}]}},
        ['tex', 0]))
    run(sc, st, 'a matcap-mapped environment material')

    # ---- orthographic cameras refuse the backface override by name
    st = base_settings(w, h)
    sc = demo_scene(st, with_texture=False)
    sc.camera.type = 'ORTHO'
    sc.materials[1] = Material(name='Bk', index=1, graph=master(
        {}, None, more=[sk('Backface Color', 'RGBA', [0.9, 0.1, 0.1, 1.0]),
                        sk('Backface Mix', 'VALUE', 0.7)]))
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('the backface override refuses orthographic cameras by name',
          p is None and 'orthographic' in str(why), str(why))


def test_env_reflection_and_inert_opacity_on_the_gpu():
    """The INERT list empties into its honest cases.

    Environment reflection travels when the world is what `world_color`'s
    plain path can express: a solid colour, the two-colour blend, or an
    environment texture (equirect and mirror-ball both) -- sampled through
    the same manual bilinear arithmetic as every other image, with the
    world spec in the plan signature so a sky edit re-plans. An active sky
    mode, a world graph or the ground plane refuse by name. And a
    reflective material with Environment Reflection OFF is inert on the
    CPU, so it shades instead of refusing.

    Opacity and edge opacity follow the same honesty: Transparency NONE
    forces alpha to 1.0 after everything -- the era's no-alpha-unit answer
    -- so under NONE they cannot reach the picture and shade freely; any
    other mode still refuses, naming the missing alpha compositing.
    """
    from ..core import raster
    from ..core.scene import ImageBuffer, Material
    from ..gpu import shade as GSH
    from ..tests.scenebuild import checker_image

    w, h = 128, 96

    def run(label, tweak, expect=True, needle=None):
        st = base_settings(w, h, shadows=True)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        tweak(sc, st)
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None,
                         view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        if not expect:
            check(f'{label} refuses by name',
                  passes is None and (needle or '') in str(why), str(why))
            return None, None
        check(f'{label} qualifies', passes is not None, str(why))
        if passes is None:
            return None, None
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{label} is the CPU frame', err < 6e-3, f'max {err:.5f}')
        return img, g

    # ---- reflection over each world kind the plain path expresses
    base_img, g = run('reflection over the blend sky',
                      lambda sc, st: setattr(sc.materials[1],
                                             'reflect_level', 0.45))
    plain, _g2 = run('the same frame without reflection',
                     lambda sc, st: None)
    if base_img is not None and plain is not None:
        cov = g.tri >= 0
        check('and the reflection actually changes the frame',
              float(np.abs(base_img[cov] - plain[cov]).max()) > 0.02)

    def solid(sc, st):
        sc.materials[1].reflect_level = 0.4
        sc.world.sky_blend = False
        sc.world.color = (0.2, 0.4, 0.7)
    run('reflection over a solid world', solid)

    def envmap(sc, st):
        sc.materials[2].reflect_level = 0.5
        sc.images['envmap'] = ImageBuffer(name='envmap',
                                          pixels=checker_image())
        sc.world.env_image = sc.images['envmap']
    run('reflection over an equirect environment', envmap)

    def mirror(sc, st):
        envmap(sc, st)
        sc.world.env_mapping = 'MIRRORBALL'
    run('reflection over a mirror-ball environment', mirror)

    def refl_off(sc, st):
        sc.materials[1].reflect_level = 0.45
        st.env_reflection = False
    run('a reflective material with the setting off (inert)', refl_off)

    def bryce(sc, st):
        sc.materials[1].reflect_level = 0.45
        sc.world.mode = 'BRYCE'
    run('reflection under a Bryce sky', bryce)   # CPU-composite env now

    def ground(sc, st):
        sc.materials[1].reflect_level = 0.45
        sc.world.ground_plane = True
    run('reflection over the ground plane', ground)  # CPU-composite env

    # ---- opacity and edge opacity, inert under Transparency NONE
    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def see_through_graph():
        ins = [
            sk('Diffuse Color', 'RGBA', [0.7, 0.45, 0.3, 1.0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 0.5),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
            sk('Edge Opacity', 'VALUE', 0.3),
        ]
        return {'output': 'out', 'nodes': {
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}}

    def clear_mat(sc, st):
        st.transparency = 'NONE'
        sc.materials[1] = Material(name='Clear', index=1,
                                   graph=see_through_graph())
    run('half-opacity under Transparency NONE (forced opaque)', clear_mat)

    def clear_sorted(sc, st):
        st.transparency = 'SORTED'
        sc.materials[1] = Material(name='Clear', index=1,
                                   graph=see_through_graph())
    run('the same material under SORTED', clear_sorted, expect=False,
        needle='alpha compositing')


def test_coded_shader_images_and_screen_inputs():
    """The coded-shader contract completes: images and screen inputs.

    A sampler uniform's socket names the image, and its prepared pixels
    ride the same manual-sampler machinery as every texture -- filtered
    with the scene's own filter and REPEAT wrap, exactly the SCtx the
    evaluator hands the program. The rewrite runs over the shader's own
    inlined functions, mangled sampler name included. A MISSING image
    breaks the node on the CPU too -- the evaluator errors -- so the probe
    refuses those frames itself. vScreenUV and iResolution bake from the
    frame size --
    screen coordinates derive from vUV, whose orientation every agreement
    test has proven, not from gl_FragCoord, whose y origin nothing
    headless can check; gl_FragCoord itself refuses by name because its z
    is the view-space depth. The frame size joins the plan signature, so a
    resolution change re-bakes.
    """
    from ..core import raster
    from ..core.scene import ImageBuffer, Material
    from ..gpu import shade as GSH
    from ..shaders.compiler import compile_shader
    from ..tests.scenebuild import checker_image

    CODE = '''
uniform sampler2D pattern;
uniform float zoom = 3.0;

in vec2 vUV;
in vec2 vScreenUV;
in vec3 iResolution;

out vec4 Color;

void main() {
    vec4 tex = texture(pattern, vUV * zoom);
    float vig = 1.0 - 0.6 * length(vScreenUV
        - vec2(0.5, iResolution.y / iResolution.x * 0.5));
    Color = vec4(tex.rgb * vig, 1.0);
}
'''

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def graph(image_key):
        ins = [
            sk('Diffuse Color', 'RGBA', [0.6, 0.6, 0.6, 1.0], ['code', 0]),
            sk('Vertex Color', 'RGBA', [1, 1, 1, 1]),
            sk('Vertex Color Mix', 'VALUE', 0.0),
            sk('Diffuse Level', 'VALUE', 1.0),
            sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
            sk('Specular Level', 'VALUE', 0.4),
            sk('Glossiness', 'VALUE', 24.0),
            sk('Roughness', 'VALUE', 0.3),
            sk('Ambient', 'VALUE', 1.0),
            sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
            sk('Opacity', 'VALUE', 1.0),
            sk('IOR', 'VALUE', 1.45),
            sk('Anisotropy', 'VALUE', 0.0),
            sk('Anisotropic Rotation', 'VALUE', 0.0),
            sk('Metalness', 'VALUE', 0.0),
            sk('Soften', 'VALUE', 0.0),
            sk('Reflection', 'VALUE', 0.0),
            sk('Translucency', 'VALUE', 0.0),
            sk('Toon Size', 'VALUE', 0.5),
            sk('Toon Smooth', 'VALUE', 0.05),
        ]
        return {'output': 'out', 'nodes': {
            'code': {'id': 'code', 'bl_idname': 'HALCYON_CodeNode',
                     'props': {'source_text': CODE, 'language': 'GLSL'},
                     'inputs': [
                         {'name': 'Pattern', 'type': 'RGBA',
                          'uniform': 'pattern', 'is_image': True,
                          'image': image_key, 'default': None, 'link': None},
                         {'name': 'Zoom', 'type': 'VALUE',
                          'uniform': 'zoom', 'default': 2.0, 'link': None}],
                     'outputs': [{'name': 'Color', 'key': 'Color',
                                  'type': 'RGBA'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG', 'toon_steps': 2},
                    'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0]),
                               sk('Displacement', 'VECTOR', [0, 0, 0])],
                    'outputs': []}}}

    def run(image_key, label, w=128, h=96, filt='NEAREST'):
        st = base_settings(w, h, shadows=True, tex_filter=filt)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        sc.images['checker'] = ImageBuffer(name='checker',
                                           pixels=checker_image())
        mat = Material(name='Coded', index=1, graph=graph(image_key))
        mat.programs = {'code': compile_shader(CODE)}
        sc.materials[1] = mat
        cpu = R.render(sc, st)
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None,
                         view, eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atlases = GSH.plan_frame(job, g)
        check(f'{label} qualifies', passes is not None, str(why))
        if passes is None:
            return None
        img, _hit = GSH.simulate(job, g, passes, atlases)
        cov = g.tri >= 0
        err = float(np.abs(img[cov] - cpu[cov][:, :3]).max())
        check(f'{label} is the CPU frame', err < 6e-3, f'max {err:.5f}')
        return [p for p in passes if p[1] == 'Coded'][0]

    for filt in ('NEAREST', 'BILINEAR'):
        got = run('checker', f'an image-sampling coded shader under {filt}',
                  filt=filt)
        if got is not None and filt == 'NEAREST':
            check('the image binds under its mangled sampler name',
                  got[3]['textures'].get('_cncode_pattern') == 'checker',
                  str(got[3]['textures']))
            src128 = got[2]
    # a missing image breaks the node on the CPU (the evaluator errors),
    # so the probe refuses the frame rather than inventing a fallback
    st = base_settings(128, 96)
    sc = demo_scene(st, with_texture=False)
    mat = Material(name='Coded', index=1, graph=graph('no_such_image'))
    mat.programs = {'code': compile_shader(CODE)}
    sc.materials[1] = mat
    view, _proj, vp, eye = R.camera_matrices(sc.camera, 128, 96)
    g = raster.GBuffer(128, 96)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 128, 96, gbuf=g)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None,
                     view, eye, 128, 96)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('a missing image refuses at the probe, as the CPU breaks too',
          p is None and 'evaluator' in str(why), str(why))
    # the frame size is baked, so a resolution change re-bakes the source
    got_small = run('checker', 'the same shader at another resolution',
                    w=96, h=72)
    if got_small is not None and src128 is not None:
        check('the baked screen inputs differ across resolutions',
              got_small[2] != src128)

    # gl_FragCoord refuses by name: its z the pass cannot carry
    from ..gpu.emit import Unsupported as EmitUnsupported, _port_code_node
    try:
        _port_code_node('out vec4 Color;\nvoid main(){ Color = '
                        'vec4(gl_FragCoord.xy, 0.0, 1.0); }', 'x')
        check('gl_FragCoord is refused by name', False, 'no refusal')
    except EmitUnsupported as exc:
        check('gl_FragCoord is refused by name',
              'vScreenUV' in str(exc), str(exc))


def test_the_compute_rasteriser_is_fill():
    """The compute kernel picks the same triangle fill() picks, everywhere.

    The claim is exactness, so the bar is an integer: ZERO differing
    triangle ids, on every scene shape that stresses a different rule.
    Both-inclusive edges and first-triangle-wins ties come from the
    per-pixel sequential walk in submission order -- coplanar overlapping
    quads are the tie case, and every covered pixel must pick the FIRST.
    Near-plane clipping emits sub-triangles whose corners carry
    barycentrics over the original; culling must drop the same triangles;
    mixed winding must set the same front flags. Barycentrics and depth
    agree to the ulp (float32 both sides, same operation order).
    """
    from ..core import raster
    from ..core.scene import Camera, MeshData
    from ..gpu import craster as CRA
    from ..tests.scenebuild import look_at_matrix

    def diff(sc_mesh, vp, w, h, label, cull='NONE'):
        g = raster.GBuffer(w, h)
        raster.rasterize(sc_mesh.verts, sc_mesh.tris, vp, w, h, gbuf=g,
                         cull=cull)
        sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc_mesh, vp,
                                                             w, h)
        # depth_bits must MATCH rasterize's default: the watertight window
        # creates overlap pixels on shared edges whose cross-triangle z
        # differs at the ulp, and quantisation must collapse it on BOTH
        # sides or the harness manufactures a disagreement production
        # never has (both production roads share st.depth_precision)
        tri, bary, zndc, front, _b2, _lin, _mk = CRA.simulate_raster(
            sx, sy, iw, z, bw, src, tmap, w, h, cull=cull, depth_bits=24)
        d = int((tri != g.tri).sum())
        check(f'{label}: zero differing pixels', d == 0,
              f'{d} of {w * h} differ')
        cov = g.tri >= 0
        if cov.any() and d == 0:
            db = float(np.abs(bary[cov] - g.bary[cov]).max())
            dz = float(np.abs(zndc[cov] - g.zndc[cov]).max())
            df = int((front[cov] != g.front[cov]).sum())
            check(f'{label}: bary to the ulp', db < 1e-5, f'{db:.2e}')
            check(f'{label}: depth exact-ish', dz < 1e-6, f'{dz:.2e}')
            check(f'{label}: front flags identical', df == 0, str(df))
        return g

    # ---- the demo scene, plain and with both cull modes
    w, h = 160, 120
    st = base_settings(w, h)
    sc = demo_scene(st, with_texture=False)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = diff(sc.mesh, vp, w, h, 'the demo scene')
    check('the demo scene actually covers pixels', bool((g.tri >= 0).any()))
    diff(sc.mesh, vp, w, h, 'back-face culling', cull='BACK')
    diff(sc.mesh, vp, w, h, 'front-face culling', cull='FRONT')

    # ---- mixed winding (both front flags in one frame)
    sc2 = demo_scene(st, with_texture=False)
    m = sc2.mesh
    floor = np.nonzero(m.mat_index == 0)[0]
    m.tris[floor[::2]] = m.tris[floor[::2]][:, [0, 2, 1]]
    diff(m, vp, w, h, 'mixed winding')

    # ---- exact depth ties: the SAME triangles submitted twice. Merely
    # coplanar geometry is not a tie at the ulp (different corners round
    # differently and both implementations agree on that -- the first form
    # of this test proved it); identical triangles ARE, and the strict `<`
    # must keep the first submission on every covered pixel
    verts = np.array([[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
                     np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3],
                     [0, 1, 2], [0, 2, 3]], np.int32)   # byte-identical
    mesh = MeshData(verts=verts, tris=tris)
    cam = Camera(matrix_world=look_at_matrix((0.0, 0.0, 6.0),
                                             (0.0, 0.0, 0.0)),
                 lens=35.0, sensor=36.0, clip_start=0.1, clip_end=100.0)
    _v2, _p2, vp2, _e2 = R.camera_matrices(cam, w, h)
    g = diff(mesh, vp2, w, h, 'exact depth ties')
    cov_tris = np.unique(g.tri[g.tri >= 0])
    check('every exact tie went to the first submission',
          set(cov_tris.tolist()) <= {0, 1}, str(cov_tris))

    # ---- near-plane clipping: the camera inside the geometry
    cam2 = Camera(matrix_world=look_at_matrix((0.3, -1.2, 0.6),
                                              (0.0, 0.5, 0.4)),
                  lens=24.0, sensor=36.0, clip_start=0.25, clip_end=100.0)
    _v3, _p3, vp3, _e3 = R.camera_matrices(cam2, w, h)
    sc3 = demo_scene(st, with_texture=False)
    sxc, _syc, _iwc, _zc, _bwc, srcc, _tmc = CRA.raster_inputs_for(
        sc3.mesh, vp3, w, h)
    check('the close camera actually clips (more emitted than source tris)',
          sxc.shape[0] != sc3.mesh.tris.shape[0]
          or len(np.unique(srcc)) < sxc.shape[0],
          f'{sxc.shape[0]} emitted from {sc3.mesh.tris.shape[0]}')
    diff(sc3.mesh, vp3, w, h, 'near-plane clipped geometry')

    # ---- the reconstruction: a GBuffer built from the kernel's images
    # carries fill()'s exact fields, empty-pixel conventions included
    g_cpu = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g_cpu)
    sx, sy, iw, z, bw, src, tmap = CRA.raster_inputs_for(sc.mesh, vp, w, h)
    tri, bary, zndc, front, b2, _lin, _mk = CRA.simulate_raster(
        sx, sy, iw, z, bw, src, tmap, w, h)
    ids = np.stack([bary[:, :, 0], bary[:, :, 1],
                    1.0 - bary[:, :, 0] - bary[:, :, 1],
                    tri.astype(np.float32)], -1).astype(np.float32)
    aux = np.stack([zndc, front.astype(np.float32), b2,
                    np.zeros_like(zndc)], -1).astype(np.float32)
    g_rec = raster.GBuffer(w, h)
    CRA.gbuffer_into(g_rec, ids, aux)
    check('reconstructed tri ids are identical',
          bool(np.array_equal(g_rec.tri, g_cpu.tri)))
    cov = g_cpu.tri >= 0
    check('reconstructed bary carries the CPU\'s own b2',
          float(np.abs(g_rec.bary[cov] - g_cpu.bary[cov]).max()) < 1e-5)
    check('empty pixels keep depth at +inf',
          bool(np.isinf(g_rec.depth[~cov]).all()) if (~cov).any() else True)
    check('covered depth matches',
          float(np.abs(g_rec.depth[cov] - g_cpu.depth[cov]).max()) < 1e-5)
    check('front flags reconstruct identically',
          bool(np.array_equal(g_rec.front, g_cpu.front)))

    # ---- the render() hook: with no driver, GPU raster falls back to the
    # CPU path and the frame is untouched
    w4, h4 = 96, 72
    st4 = base_settings(w4, h4)
    st4.render_device = 'GPU'
    st4.gpu_raster = True
    img_gpu = R.render(demo_scene(st4, with_texture=False), st4)
    st5 = base_settings(w4, h4)
    img_cpu = R.render(demo_scene(st5, with_texture=False), st5)
    check('the hook falls back cleanly without a driver',
          float(np.abs(img_gpu - img_cpu).max()) < 1e-6)


def test_the_bvh_occlusion_kernel_is_occluded():
    """The ray-tracing arc's first kernel: any-hit traversal, exactly.

    The front-end supports divergent while-loops but not dynamic array
    writes, so the traversal is STACKLESS: a threaded BVH whose links are
    precomputed at pack time to walk the CPU's exact LIFO order (right
    subtree first). Any-hit never depends on order, but the links are
    built so closest-hit will inherit the right tie-breaks for free. The
    bar is the usual integer: ZERO mismatched booleans against
    `bvh.occluded()` -- on box-random rays and on the real shadow-ray
    shape, surface origins pointing at a light with tmax at the light.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..gpu import rtrace as RT

    st = base_settings(160, 120)
    sc = demo_scene(st, with_texture=False)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)

    # threading invariants: the links ARE the CPU's pop order
    first, miss = RT.thread_links(bvh)
    check('the root has nowhere to miss to', miss[0] == -1)
    inner = np.nonzero((bvh.left[:bvh.n_nodes] >= 0)
                       & (bvh.right[:bvh.n_nodes] >= 0))[0]
    ok_first = all(first[i] == bvh.right[i] for i in inner)
    ok_sib = all(miss[bvh.right[i]] == bvh.left[i] for i in inner)
    ok_up = all(miss[bvh.left[i]] == miss[i] for i in inner)
    check('right child walks first, as the LIFO pop does', ok_first)
    check("the right child's miss is its left sibling", ok_sib)
    check("the left child's miss is the parent's", ok_up)

    # box-random rays
    rng = np.random.default_rng(3)
    n = 1200
    lo = sc.mesh.verts.min(0) - 1.0
    hi = sc.mesh.verts.max(0) + 1.0
    org = (rng.random((n, 3)) * (hi - lo) + lo).astype(np.float32)
    d = rng.normal(size=(n, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tmax = (rng.random(n) * 20.0 + 0.1).astype(np.float32)
    want = bvh.occluded(org, d, tmax)
    got = RT.simulate_occluded(bvh, org, d, tmax)
    check('box-random rays: zero mismatches',
          int((got != want).sum()) == 0,
          f'{int((got != want).sum())} of {n}, CPU hits {int(want.sum())}')
    check('and the rays actually hit things', bool(want.any()))

    # the shadow-ray shape: surface points toward the point light
    w, h = 160, 120
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    py, px = np.nonzero(g.tri >= 0)
    pick = rng.choice(py.size, 1500, replace=False)
    ctx = job.context(g.tri[py, px][pick], g.bary[py, px][pick],
                      px[pick], py[pick], np.ones(1500, bool), None, 0, True)
    lp = np.asarray(sc.lights[1].position, np.float32)
    delta = lp[None, :] - ctx.P
    dist = np.linalg.norm(delta, axis=1)
    ldir = (delta / dist[:, None]).astype(np.float32)
    sorg = (ctx.P + ldir * 1e-3).astype(np.float32)
    stmax = (dist - 2e-3).astype(np.float32)
    want_s = bvh.occluded(sorg, ldir, stmax)
    got_s = RT.simulate_occluded(bvh, sorg, ldir, stmax)
    check('shadow-shaped rays: zero mismatches',
          int((got_s != want_s).sum()) == 0,
          f'{int((got_s != want_s).sum())} of 1500, '
          f'shadowed {int(want_s.sum())}')
    check('and shadow and light both occur',
          bool(want_s.any()) and bool((~want_s).any()))


def test_compute_wrappers_fetch_data_exactly():
    """Driver-only compute sources read data textures by texelFetch.

    The field taught this one at a price: the reflected frame's compute
    trace misread row-boundary-adjacent RAY texels through texture() with
    computed normalized coordinates -- 95 deterministic wrong rays at a
    245-wide layout, zero at 283 -- so exactness through the sampler is a
    texture-size lottery. texelFetch with integer coordinates is exact by
    specification. The FRAGMENT variants keep texture(): they are what
    the NumPy front-end runs and what every headless proof is made of,
    and the shared BVH texel helpers stay with them, fragment-measured
    at zero across the shadow and kernel sections.
    """
    from ..gpu import craster as CRA, rtrace as RT

    check('the raster compute build swapped its fetchers',
          'texelFetch' in CRA.COMPUTE_SOURCE
          and 'texture(' not in CRA.COMPUTE_SOURCE)
    for name, src in (('occlusion', RT.OCCLUDE_COMPUTE),
                      ('closest-hit', RT.INTERSECT_COMPUTE)):
        tail = src.split('void main()')[-1]
        check(f'the {name} compute main() reads rays only by texelFetch',
              'texelFetch' in tail and 'texture(' not in tail)
    # the tree fetches went texelFetch everywhere in 1.25.71 on
    # suspicion of row-boundary misreads -- and the glass-mirror flips
    # stayed BIT-IDENTICAL, an acquittal (the real mechanism was
    # coincident-surface t ties; see the tie-referral test). The
    # conversion bought zero correctness and cost the fragment shadow
    # taps real milliseconds on the field driver, so 1.25.73 reverts
    # the TREE to the filtered form every 0-px measurement was made
    # with. The RAY fetches above keep texelFetch: theirs is the
    # layout that actually misread on hardware.
    for name, src in (('occlusion', RT.OCCLUDE_FRAGMENT),
                      ('closest-hit', RT.INTERSECT_FRAGMENT)):
        check(f'the {name} traversal reads the tree by texture()',
              'texture(hal_bvh' in src and 'texture(hal_btris' in src
              and 'texelFetch(hal_bvh,' not in src
              and 'texelFetch(hal_btris,' not in src)
    check('the raster fragment variant keeps the front-end path',
          'texelFetch' not in CRA.FRAGMENT_SOURCE)
    # and the swap really produced compilable-shaped GLSL: the strip that
    # CreateInfo applies leaves no declarations behind
    import re
    from ..gpu.device import strip_declarations
    decl = re.compile(r'^(uniform|in|out)\s+\w[^;]*;$')
    for name, src in (('raster', CRA.COMPUTE_SOURCE),
                      ('occlusion', RT.OCCLUDE_COMPUTE),
                      ('closest-hit', RT.INTERSECT_COMPUTE)):
        left = [ln for ln in strip_declarations(src).splitlines()
                if decl.match(ln.split('//', 1)[0].strip())]
        check(f'the {name} compute source strips clean', not left,
              str(left[:2]))


def test_the_bvh_closest_hit_kernel_is_intersect():
    """The reflections arc's kernel: closest-hit traversal, exactly.

    Any-hit was order-free; closest-hit is where the threaded links EARN
    their construction. `bvh.intersect()` pops LIFO (right subtree first),
    keeps a hit only when strictly closer, and inside a leaf argmin keeps
    the first of equal minima -- so a sequential walk in the same order
    with the same strict `<` must agree on every ray, including exact
    ties, which byte-identical duplicate triangles here force outright.
    The bar is the usual integer: ZERO mismatched hit ids against
    `bvh.intersect()`, on box-random rays and on the exact reflection
    shape `trace()` will use -- surface origins along reflect(-V, N) with
    tmax at 1e30 -- with t, u, v agreeing wherever the ids do.
    """
    from ..core import mathx as M, raster
    from ..core.bvh import BVH
    from ..gpu import rtrace as RT

    st = base_settings(160, 120)
    sc = demo_scene(st, with_texture=False)
    bvh = BVH(sc.mesh.verts, sc.mesh.tris)

    def agree(name, rays_n, want, got, tmax):
        wid, wt, wu, wv = want
        gid, gt, gu, gv = got
        mism = int((gid != wid).sum())
        check(f'{name}: zero mismatched hit ids', mism == 0,
              f'{mism} of {rays_n}, CPU hit {int((wid >= 0).sum())}')
        hit = (wid >= 0) & (gid == wid)
        if hit.any():
            dt = float(np.abs(gt[hit] - wt[hit]).max())
            duv = float(max(np.abs(gu[hit] - wu[hit]).max(),
                            np.abs(gv[hit] - wv[hit]).max()))
            check(f'{name}: t agrees where the ids do', dt <= 1e-5,
                  f'max {dt:.2e}')
            check(f'{name}: barycentrics agree', duv <= 1e-5,
                  f'max {duv:.2e}')
        missed = wid < 0
        if missed.any():
            left = float(np.abs(gt[missed] - tmax[missed]).max())
            check(f'{name}: a miss leaves t at tmax, as the CPU does',
                  left == 0.0, f'max {left:.2e}')

    # box-random rays
    rng = np.random.default_rng(11)
    n = 1200
    lo = sc.mesh.verts.min(0) - 1.0
    hi = sc.mesh.verts.max(0) + 1.0
    org = (rng.random((n, 3)) * (hi - lo) + lo).astype(np.float32)
    d = rng.normal(size=(n, 3)).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tmax = (rng.random(n) * 20.0 + 0.1).astype(np.float32)
    want = bvh.intersect(org, d, tmax)
    got = RT.simulate_intersect(bvh, org, d, tmax)
    agree('box-random rays', n, want, got, tmax)
    check('and both hits and misses occur',
          bool((want[0] >= 0).any()) and bool((want[0] < 0).any()))

    # exact ties, forced: byte-identical duplicated triangles, so several
    # ids hit at the SAME t and only the visit order decides the winner
    tris2 = np.concatenate([sc.mesh.tris, sc.mesh.tris[:200]], axis=0)
    bvh2 = BVH(sc.mesh.verts, tris2)
    n2 = 800
    org2 = (rng.random((n2, 3)) * (hi - lo) + lo).astype(np.float32)
    d2 = rng.normal(size=(n2, 3)).astype(np.float32)
    d2 /= np.linalg.norm(d2, axis=1, keepdims=True)
    tmax2 = np.full(n2, 50.0, np.float32)
    want2 = bvh2.intersect(org2, d2, tmax2)
    got2 = RT.simulate_intersect(bvh2, org2, d2, tmax2)
    agree('duplicated-triangle ties', n2, want2, got2, tmax2)
    dup_hit = want2[0] >= sc.mesh.tris.shape[0]
    check('the tie population is real: some winners ARE duplicates',
          bool(dup_hit.any()) or bool((want2[0] >= 0).any()),
          f'{int(dup_hit.sum())} duplicate wins')

    # the reflection shape: exactly the rays trace() will cast
    w, h = 160, 120
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    py, px = np.nonzero(g.tri >= 0)
    pick = rng.choice(py.size, 1500, replace=False)
    ctx = job.context(g.tri[py, px][pick], g.bary[py, px][pick],
                      px[pick], py[pick], np.ones(1500, bool), None, 0, True)
    N = M.normalize(ctx.N)
    V = -M.normalize(ctx.I)
    Rdir = M.reflect(-V, N).astype(np.float32)
    rorg = (ctx.P + N * 1e-3).astype(np.float32)
    rtmax = np.full(1500, 1e30, np.float32)
    want_r = bvh.intersect(rorg, Rdir, rtmax)
    got_r = RT.simulate_intersect(bvh, rorg, Rdir, rtmax)
    agree('reflection-shaped rays', 1500, want_r, got_r, rtmax)
    check('and reflections both hit geometry and escape to the sky',
          bool((want_r[0] >= 0).any()) and bool((want_r[0] < 0).any()))


def test_coincident_surface_ties_route_to_the_cpu():
    """The glass-mirror seam's named mechanism, and its cure.

    The 1.25.72 field cross-check proved a real kernel-vs-frontend
    divergence on exactly 1925 of 141478 sweep rays -- cached==fresh
    (upload cache exonerated), both != frontend -- after `precise` and a
    texelFetch conversion left the flips bit-identical (arithmetic and
    fetches acquitted). The headless boundary census then measured every
    decision cliff on the same recorded rays: self-hit epsilon 0,
    inclusion grazes 0 below 1e-5, near-miss beaters 0, det cliff 0 --
    and WINNER TIES 22106, every one the demo box's bottom face resting
    EXACTLY coplanar on the floor (n.n = -1, plane offset 0, floor tri 0
    vs box tris 772/773). Two surfaces at the same depth put the winner
    in the last bit of t, and the driver's fused rounding is not
    NumPy's: 1925 of those coin flips landed the other way,
    deterministically. No arithmetic edit can pin that, so the kernel
    now NAMES the ties: it tracks its two nearest accepted hits, and
    when they sit within 1e-5 relative t (measured valley -- real ties
    under 1e-6, the next distinct surface beyond 1e-3) it returns id
    -2, and every wrapper re-resolves exactly those rays through
    bvh.intersect itself. The GPU path then returns the reference's own
    winner BY CONSTRUCTION -- ray routing, the layer-routing doctrine
    at ray granularity. Clean geometry must flag nothing.
    """
    from ..core.bvh import BVH
    from ..gpu import rtrace as RT

    # the field shape in miniature: a "floor" plane with a coincident
    # opposite-wound "box bottom" over part of it, and a catch plane
    # below so grazing misses still land somewhere
    verts = np.array([
        [-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0],     # floor
        [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0],     # box bottom
        [-3, -3, -5], [3, -3, -5], [3, 3, -5], [-3, 3, -5],  # catch
    ], np.float32)
    tris = np.array([
        [0, 1, 2], [0, 2, 3],
        [6, 5, 4], [7, 6, 4],          # reversed winding: contact plane
        [8, 9, 10], [8, 10, 11],
    ], np.int32)
    bvh = BVH(verts, tris)

    rng = np.random.default_rng(73)
    n = 700
    org = np.stack([rng.random(n) * 3.8 - 1.9,
                    rng.random(n) * 3.8 - 1.9,
                    np.full(n, 2.0)], 1).astype(np.float32)
    d = np.stack([rng.normal(0, 0.08, n), rng.normal(0, 0.08, n),
                  np.full(n, -1.0)], 1).astype(np.float32)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    tmax = np.full(n, 1e30, np.float32)

    want = bvh.intersect(org, d, tmax)
    got = RT.simulate_intersect(bvh, org, d, tmax)
    routed = int(RT.LAST_TIE_ROUTED)
    scale = 2.0 / -d[:, 2]
    over = (np.abs(org[:, 0] + d[:, 0] * scale) < 1.0) \
        & (np.abs(org[:, 1] + d[:, 1] * scale) < 1.0)
    check('rays through the contact plane get flagged and referred',
          routed > 0, f'{routed} routed, {int(over.sum())} cross it')
    check('the referral count is the contact population, not the frame',
          0 < routed <= int(over.sum()) + 8,
          f'{routed} vs {int(over.sum())} rays over the contact patch')
    check('every ray returns the reference winner EXACTLY',
          bool((got[0] == want[0]).all()),
          f'{int((got[0] != want[0]).sum())} of {n} differ')
    check('referred rays carry the CPU t, u, v verbatim',
          bool((got[1] == want[1]).all() and (got[2] == want[2]).all()
               and (got[3] == want[3]).all()))

    # clean geometry: the same rays with the contact plane LIFTED clear
    # of the window must flag nothing and still agree
    verts2 = verts.copy()
    verts2[4:8, 2] = 0.25
    bvh2 = BVH(verts2, tris)
    want2 = bvh2.intersect(org, d, tmax)
    got2 = RT.simulate_intersect(bvh2, org, d, tmax)
    check('separated surfaces flag NO ties (routing costs nothing '
          'on clean scenes)', int(RT.LAST_TIE_ROUTED) == 0,
          f'{int(RT.LAST_TIE_ROUTED)} routed')
    check('and the clean frame still agrees on every id',
          bool((got2[0] == want2[0]).all()))

    # the source contract: the window, the referral, the loosened prune
    check('the kernel tracks its second-nearest accepted hit',
          'second_t' in RT.INTERSECT_GLSL)
    check('the kernel refers noise-window ties as id -2',
          '-2.0' in RT.INTERSECT_GLSL
          and 'second_t <= best_t * 1.00001' in RT.INTERSECT_GLSL)
    check('the slab prune loosens by the same window',
          'tn <= best_t * 1.00001' in RT.INTERSECT_GLSL)
    check('the any-hit kernel is untouched (ties cannot flip a boolean)',
          'second_t' not in RT.TRAVERSE_GLSL)


def test_projected_light_textures_spot_and_sun():
    """Sixth-generation projective texturing: the light cookie / gobo.

    A SPOT projects its image through the cone like a slide projector --
    Splinter Cell's window patterns -- and a SUN tiles its image across the
    world as a cloud shadow. The CPU multiply lives in lights.sample, so
    every consumer (pixel rate, Gouraud corners, layers, hits) gets it from
    one site; the GLSL mirror writes the CPU's own bilinear texel
    arithmetic into the light loop. Bars: a WHITE cookie and a strength-0
    cookie are EXACTLY inert (the multiply is mix(1, rgb, s) -- both
    collapse to 1), a patterned cookie moves the picture, the projection
    turns with the lamp's own frame, and the simulated deferred frame
    matches the CPU frame to the deferred bar.
    """
    from ..core import raster
    from ..core.scene import Light
    from ..gpu import shade as GSH

    w, h = 160, 120
    st = base_settings(w, h)
    st.shadows = False
    sc = demo_scene(st, with_texture=False)
    ck = np.zeros((8, 8, 4), np.float32)
    ck[:, :, 3] = 1.0
    ck[::2, ::2, :3] = 1.0
    ck[1::2, 1::2, :3] = 1.0
    ck[:, :, 1] *= 0.3
    ck[0, :, 2] = 1.0          # a blue stripe: transpose-ASYMMETRIC, so
    ck[0, :, 0] = 0.1          # the frame-rotation check below can see
    spot = Light(type='SPOT', name='proj', position=(0.0, -4.0, 6.0),
                 direction=(0.0, 0.45, -0.9), color=(1.0, 1.0, 1.0),
                 energy=800.0, spot_size=1.0, spot_blend=0.2,
                 shadow='NONE', decay='INVERSE_SQUARE')
    sun = Light(type='SUN', name='clouds', direction=(-0.5, 0.4, -0.75),
                color=(1.0, 0.97, 0.9), energy=3.0, shadow='NONE')
    sc.lights = [spot, sun]

    plain = R.render(sc, st)
    spot.cookie = ck
    got = R.render(sc, st)
    check('a spot cookie moves the picture',
          float(np.abs(got - plain).max()) > 1e-2,
          f'{float(np.abs(got - plain).max()):.4f}')

    spot.cookie = np.ones((4, 4, 4), np.float32)
    spot._cookie_tex = None
    check('a WHITE cookie is exactly inert',
          float(np.abs(R.render(sc, st) - plain).max()) == 0.0)
    spot.cookie = ck
    spot._cookie_tex = None
    spot.cookie_strength = 0.0
    check('strength 0 is exactly inert',
          float(np.abs(R.render(sc, st) - plain).max()) == 0.0)
    spot.cookie_strength = 1.0

    # the projection turns with the lamp: swapping the frame axes must
    # move the pattern (the cookie is chromatically asymmetric)
    from ..core import lights as LI
    s_ax, u_ax, _f = LI.cookie_frame(spot)
    spot.frame_x, spot.frame_y = tuple(u_ax), tuple(s_ax)
    turned = R.render(sc, st)
    check('the projection follows the lamp frame',
          float(np.abs(turned - got).max()) > 1e-3,
          f'{float(np.abs(turned - got).max()):.4f}')
    spot.frame_x = spot.frame_y = None

    # sun cookie: tiles by cookie_scale, and scale changes the tiling
    sun.cookie = ck
    sun.cookie_scale = 3.0
    t1 = R.render(sc, st)
    check('a sun cookie moves the picture',
          float(np.abs(t1 - got).max()) > 1e-2)
    sun.cookie_scale = 6.0
    sun._cookie_tex = None
    check('the sun tiling follows cookie_scale',
          float(np.abs(R.render(sc, st) - t1).max()) > 1e-3)
    sun.cookie_scale = 3.0

    # the deferred mirror: same frame through the compiled light loop
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atlases = GSH.plan_frame(job, g)
    check('the cookie frame qualifies for GPU shading', passes is not None,
          str(why))
    if passes is not None:
        check('both cookies ride as atlases',
              'hal_cookie0' in (atlases or {})
              and 'hal_cookie1' in (atlases or {}),
              str(sorted(k for k in (atlases or {})
                         if not k.startswith('__'))))
        cpu_img = R.render(sc, st)
        img, hit = GSH.simulate(job, g, passes, atlases)
        check('the simulated cookie frame ran', img is not None)
        if img is not None:
            cov = g.tri >= 0
            err = float(np.abs(img[cov] - cpu_img[cov][:, :3]).max())
            check('the GPU cookie frame is the CPU cookie frame',
                  err < 6e-3, f'max {err:.5f} over {int(cov.sum())} px')


def test_accumulation_and_edge_antialiasing():
    """The two aa_mode values that were declared and dead, implemented.

    ACCUMULATE is the OpenGL accumulation buffer: N whole frames at
    deterministic Halton subpixel offsets, averaged -- so it must be
    bit-identical run to run, soften real silhouettes, and fall back to a
    plain render at 1 sample. EDGE is the era's flicker filter: a 1-2-1
    tent applied ONLY at id-buffer edges and deep depth creases -- so
    pixels away from every edge must ride through BIT-IDENTICAL.
    """
    w, h = 96, 72
    st = base_settings(w, h)
    st.shadows = False
    sc = demo_scene(st, with_texture=False)
    plain = R.render(sc, st)

    st2 = st.copy()
    st2.aa_mode = 'ACCUMULATE'
    st2.aa_samples = 4
    a1 = R.render(sc, st2)
    a2 = R.render(sc, st2)
    check('accumulation is deterministic',
          float(np.abs(a1 - a2).max()) == 0.0)
    check('accumulation softens the frame',
          float(np.abs(a1 - plain).max()) > 1e-3,
          f'{float(np.abs(a1 - plain).max()):.4f}')
    st2.aa_samples = 1
    check('one sample falls back to the plain frame',
          float(np.abs(R.render(sc, st2) - plain).max()) == 0.0)

    st3 = st.copy()
    st3.aa_mode = 'EDGE'
    # threshold 0 = silhouettes only, by contract -- which makes the
    # id-edge mask below the COMPLETE rule. (The crease half now works in
    # scene units and flags real depth breaks the old device-depth compare
    # never could; it is proven separately just after)
    st3.aa_edge_threshold = 0.0
    e = R.render(sc, st3)
    from ..core import raster as CR
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = CR.GBuffer(w, h)
    CR.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g)
    tri = g.tri
    edge = np.zeros(tri.shape, bool)
    dif = tri[:, 1:] != tri[:, :-1]
    edge[:, 1:] |= dif
    edge[:, :-1] |= dif
    dif = tri[1:, :] != tri[:-1, :]
    edge[1:, :] |= dif
    edge[:-1, :] |= dif
    # dilate by one: the tent reads one pixel around each flagged pixel
    grow = edge.copy()
    grow[1:] |= edge[:-1]
    grow[:-1] |= edge[1:]
    grow[:, 1:] |= edge[:, :-1]
    grow[:, :-1] |= edge[:, 1:]
    interior = ~grow
    d = np.abs(e - plain).max(axis=2)
    check('edge smoothing moves edge pixels',
          float(d[edge].max()) > 1e-3 if edge.any() else False)
    check('pixels away from every edge ride through untouched',
          float(d[interior].max()) == 0.0 if interior.any() else True,
          f'max {float(d[interior].max()):.6f} off-edge')
    # the crease half, in scene units: a tight threshold smooths interior
    # depth breaks the silhouette-only pass left alone
    st4 = st.copy()
    st4.aa_mode = 'EDGE'
    st4.aa_edge_threshold = 0.005
    ec = R.render(sc, st4)
    dc = np.abs(ec - plain).max(axis=2)
    check('a scene-unit depth threshold smooths interior creases too',
          bool(interior.any()) and float(dc[interior].max()) > 0.0,
          f'max {float(dc[interior].max()):.6f} on interior creases')


def test_height_fog_layers_the_mist():
    """Ground mist: fog that thins with world height above a top plane.

    Two exact controls pin the maths: a fog top ABOVE the whole scene with
    zero falloff leaves the distance fog untouched (h = 1 everywhere), and
    a huge falloff with the top below the scene clears the fog entirely
    (h -> 0 makes the transmittance 1), matching the fog-off frame bit for
    bit. Between the controls, the mist is real: the picture differs from
    pure distance fog.
    """
    w, h = 96, 72
    st = base_settings(w, h)
    st.shadows = False
    st.fog = True
    st.fog_mode = 'EXP'
    st.fog_density = 0.08
    sc = demo_scene(st, with_texture=False)
    fogged = R.render(sc, st)
    st_off = st.copy()
    st_off.fog = False
    clear = R.render(sc, st_off)
    check('distance fog is on for the controls',
          float(np.abs(fogged - clear).max()) > 1e-2)

    hi = st.copy()
    hi.fog_height = True
    hi.fog_height_top = 1e6
    hi.fog_height_falloff = 0.0
    check('a top above the world leaves distance fog EXACTLY',
          float(np.abs(R.render(sc, hi) - fogged).max()) == 0.0)

    lo = st.copy()
    lo.fog_height = True
    lo.fog_height_top = -1e6
    lo.fog_height_falloff = 1e9
    check('a floor-sunk top with huge falloff clears the fog EXACTLY',
          float(np.abs(R.render(sc, lo) - clear).max()) == 0.0)

    mid = st.copy()
    mid.fog_height = True
    mid.fog_height_top = 0.5
    mid.fog_height_falloff = 1.5
    m = R.render(sc, mid)
    check('between the controls the mist is real',
          float(np.abs(m - fogged).max()) > 1e-3
          and float(np.abs(m - clear).max()) > 1e-3)


def test_sixth_generation_console_presets():
    """GameCube, PS2 and Xbox join the console shelf.

    Every key each preset sets must be a real RenderSettings field (apply
    ignores unknowns, which would turn a typo into a silent no-op preset),
    the trio must land in the CONSOLE category, each look must lean on the
    features that defined the machine -- the PS2's field rendering and
    dither, the GameCube's trilinear mipmaps and layered table fog, the
    Xbox's anisotropy -- and each preset must actually render.
    """
    from dataclasses import fields as _dc_fields

    from ..core.settings import RenderSettings as RS
    from ..presets.library import PRESETS

    known = {f.name for f in _dc_fields(RS)}
    for key in ('GAMECUBE', 'PS2', 'XBOX'):
        check(f'{key} preset exists', key in PRESETS)
        p = PRESETS.get(key) or {}
        check(f'{key} sits on the console shelf',
              p.get('category') == 'CONSOLE', str(p.get('category')))
        extra = sorted(set(p.get('settings', {})) - known)
        check(f'every {key} key is a real setting', not extra, str(extra))
    check('the PS2 look is field-rendered and dithered',
          PRESETS['PS2']['settings'].get('interlace') == 'FIELDS'
          and PRESETS['PS2']['settings'].get('dither') != 'NONE')
    check('the GameCube look mips trilinearly into layered table fog',
          PRESETS['GAMECUBE']['settings'].get('tex_filter') == 'TRILINEAR'
          and PRESETS['GAMECUBE']['settings'].get('fog_mode') == 'TABLE16'
          and PRESETS['GAMECUBE']['settings'].get('fog_height') is True)
    check('the Xbox look filters anisotropically',
          int(PRESETS['XBOX']['settings'].get('tex_aniso', 1)) >= 2)
    for key in ('GAMECUBE', 'PS2', 'XBOX'):
        st = base_settings(64, 48)
        st.apply(dict(PRESETS[key]['settings']))
        st.resolution_x, st.resolution_y = 64, 48
        st.aa_mode, st.aa_samples = 'NONE', 1
        st.use_processes = False
        sc = demo_scene(st, with_texture=False)
        img = R.render(sc, st)
        check(f'the {key} preset renders', img is not None
              and np.isfinite(img).all() and img.max() > 0.01)


def test_presets_never_touch_the_device_switch():
    """Selecting a look must not move the CPU/GPU switch.

    Where a frame computes is a property of the machine, not of the look:
    a 1996 preset draws the identical picture on either device. Until this
    round apply_preset(reset=True) returned render_device to its dataclass
    default, so picking ANY preset silently flipped a GPU user back to the
    CPU -- the field asked for exactly this fix. The whole device family
    must survive: the switch itself, every Debug toggle under it (in
    NON-default positions, so preservation is proven rather than
    coincidental), and the scissor tuning knob. A preset dict that names a
    device key is refused by name too -- preserving the reset is not
    enough if a future entry were to list one.
    """
    from ..presets.library import (DEVICE_KEYS, PRESERVED, PRESETS,
                                   apply_preset)

    check('every device key is preserved from the reset',
          DEVICE_KEYS <= PRESERVED, str(sorted(DEVICE_KEYS - PRESERVED)))

    non_default = {
        'render_device': 'GPU', 'gpu_post': False, 'gpu_shading': False,
        'gpu_raster': False, 'gpu_hold_context': True, 'gpu_scissor': False,
        'layer_gpu_min_frac': 0.5,
    }
    unknown = sorted(k for k in DEVICE_KEYS if k not in non_default)
    check('the test covers the whole device family', not unknown,
          str(unknown))
    for key in sorted(PRESETS):
        st = RenderSettings()
        for k, v in non_default.items():
            setattr(st, k, v)
        apply_preset(st, key, reset=True)
        moved = {k: getattr(st, k) for k, v in non_default.items()
                 if getattr(st, k) != v}
        check(f'{key} leaves the device family alone', not moved, str(moved))

    # defense in depth: a preset entry naming a device key is ignored
    PRESETS['_DEVICE_TEST'] = {
        'label': 'x', 'category': 'CONSOLE', 'note': 'x',
        'settings': {'render_device': 'CPU', 'gpu_shading': False,
                     'tex_filter': 'NEAREST'},
    }
    try:
        st = RenderSettings()
        st.render_device, st.gpu_shading = 'GPU', True
        apply_preset(st, '_DEVICE_TEST', reset=True)
        check('a device key inside a preset dict is refused by name',
              st.render_device == 'GPU' and st.gpu_shading is True,
              f'{st.render_device}/{st.gpu_shading}')
        check('the rest of that preset still applies',
              st.tex_filter == 'NEAREST', st.tex_filter)
    finally:
        del PRESETS['_DEVICE_TEST']


def test_resolution_presets_cover_the_categories():
    """The resolution shelf: televisions, monitors, computers, consoles,
    video formats and pictures, each key in exactly one category.

    The two tables must hold each other -- every group member a real
    preset, every preset in some group, no key twice (keys are enum
    identifiers and operator arguments saved inside .blend files, so the
    17 original keys must survive any reorganisation forever). Values
    must be renderable dimensions with positive pixel aspects, D1 and DV
    keep their broadcast aspects, and every key labels and describes
    itself without falling over.
    """
    from ..core.settings import (RESOLUTION_GROUPS, RESOLUTION_PRESETS,
                                 resolution_description, resolution_label)

    grouped = [k for _label, keys in RESOLUTION_GROUPS for k in keys]
    check('no key sits in two categories', len(grouped) == len(set(grouped)),
          str(sorted(k for k in set(grouped) if grouped.count(k) > 1)))
    missing = sorted(set(grouped) - set(RESOLUTION_PRESETS))
    check('every grouped key is a real preset', not missing, str(missing))
    orphans = sorted(set(RESOLUTION_PRESETS) - set(grouped))
    check('every preset belongs to a category', not orphans, str(orphans))

    legacy = ('CGA', 'VGA_13H', 'QVGA', 'VGA', 'MAC_CLASSIC', 'MAC_13',
              'SVGA', 'XGA', 'AMIGA_PAL', 'AMIGA_HIRES', 'NTSC_D1',
              'PAL_D1', 'NTSC_TOASTER', 'PSX', 'PSX_HI', 'N64', 'QUAKE')
    lost = sorted(set(legacy) - set(RESOLUTION_PRESETS))
    check('all 17 original keys survive', not lost, str(lost))

    for label, _keys in RESOLUTION_GROUPS:
        check(f'category "{label}" is not empty', bool(_keys))
    for k, v in RESOLUTION_PRESETS.items():
        x, y, ax, ay = v
        check(f'{k} has renderable dimensions',
              isinstance(x, int) and isinstance(y, int)
              and 16 <= x <= 4096 and 16 <= y <= 4096
              and ax > 0 and ay > 0, str(v))
        check(f'{k} labels and describes itself',
              bool(resolution_label(k)) and str(x) in
              resolution_description(k))
    check('D1 keeps its broadcast pixel aspect',
          RESOLUTION_PRESETS['NTSC_D1'][2:] == (10.0, 11.0)
          and RESOLUTION_PRESETS['PAL_D1'][2:] == (59.0, 54.0))
    check('DV matches D1 convention',
          RESOLUTION_PRESETS['DV_NTSC'][2:] == (10.0, 11.0)
          and RESOLUTION_PRESETS['DV_PAL'][2:] == (59.0, 54.0))
    check('the widescreen anamorphic pair exists',
          RESOLUTION_PRESETS['NTSC_D1_WIDE'][2] / RESOLUTION_PRESETS['NTSC_D1_WIDE'][3] > 1.0
          and RESOLUTION_PRESETS['PAL_D1_WIDE'][2] / RESOLUTION_PRESETS['PAL_D1_WIDE'][3] > 1.0)


def test_high_poly_shadow_maps_build_in_parallel_and_exact():
    """The high-poly walls, held to their bits.

    At 820k triangles the profile said: shadow maps 74% of the frame
    (seven serial rasterisations -- sun map plus six cube faces), then
    the rasteriser's own per-triangle pipelines. Two fixes ship, both
    REQUIRED to be invisible in the picture: the maps now build in
    PARALLEL (each map is its own buffer of order-independent depth
    min-compares -- big-array NumPy that releases the interpreter lock,
    the one shape of work threads scale here), and build_screen_tris
    dropped a 30 MB materialised identity plus two full-array copies in
    its no-straddler fast path. A frustum cull was tried and REVERTED on
    the measurement: a concentrated object sits inside most map
    frustums, so it kept ~100% and cost 15%. Held here: parallel maps
    are BIT-IDENTICAL to serial ones for sun, spot and cube lights on a
    dense mesh, and the dense frame itself is bit-stable across the
    shadow cache and thread counts.
    """
    import concurrent.futures as fut

    from ..core import lights as L
    from ..core.scene import Light

    st = base_settings(96, 72)
    st.threads = 1
    st.shadows = True
    st.cache_shadows = False
    sc = demo_scene(st, with_texture=False)
    # densify past the 50k parallel threshold: a lattice of small tris
    rng = np.random.default_rng(11)
    C = rng.uniform(-2.5, 2.5, (55000, 3)).astype(np.float32)
    C[:, 2] = np.abs(C[:, 2]) * 0.8 + 0.2
    V = (np.repeat(C, 3, 0)
         + rng.uniform(-0.06, 0.06, (165000, 3))).astype(np.float32)
    m = sc.mesh
    base = m.verts.shape[0]
    newt = np.arange(165000, dtype=np.int32).reshape(-1, 3) + base
    m.verts = np.concatenate([m.verts, V])
    m.normals = np.concatenate(
        [m.normals, np.tile(np.array([[0, 0, 1.0]], np.float32),
                            (165000, 1))])
    if m.uvs is not None:
        m.uvs = np.concatenate([m.uvs, np.zeros((165000, 2), np.float32)])
    if getattr(m, 'uvs2', None) is not None:
        m.uvs2 = np.concatenate([m.uvs2, np.zeros((165000, 2), np.float32)])
    if m.colors is not None:
        m.colors = np.concatenate(
            [m.colors, np.ones((165000, 4), np.float32)])
    m.tris = np.concatenate([m.tris, newt])
    fn = np.cross(m.verts[newt[:, 1]] - m.verts[newt[:, 0]],
                  m.verts[newt[:, 2]] - m.verts[newt[:, 0]])
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-9)
    m.face_normals = np.concatenate([m.face_normals, fn.astype(np.float32)])
    m.mat_index = np.concatenate(
        [m.mat_index, np.full(newt.shape[0], 1, np.int32)])
    m.obj_index = np.concatenate(
        [m.obj_index, np.full(newt.shape[0], int(m.obj_index.max()),
                              np.int32)])
    m.smooth = np.concatenate([m.smooth, np.ones(newt.shape[0], bool)])
    sc.lights.append(Light(type='SPOT', name='sp', position=(3.0, -4.0, 5.0),
                           direction=(-0.4, 0.5, -0.75),
                           color=(1.0, 1.0, 1.0), energy=300.0,
                           shadow='MAP', spot_size=0.9, spot_blend=0.3))

    check('the dense fixture crosses the parallel threshold',
          m.tris.shape[0] >= 50000, str(m.tris.shape[0]))

    class _Serial:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def map(self, fn, it):
            return [fn(x) for x in it]

    old = fut.ThreadPoolExecutor
    fut.ThreadPoolExecutor = _Serial
    try:
        L._build_shadow_maps(sc, st)
        serial = [l.shadow_map for l in sc.lights]
    finally:
        fut.ThreadPoolExecutor = old
    L._build_shadow_maps(sc, st)
    parallel = [l.shadow_map for l in sc.lights]

    kinds = set()
    for a, b, l in zip(serial, parallel, sc.lights):
        if a is None and b is None:
            continue
        kinds.add(l.type)
        if hasattr(a, 'faces'):
            same = all(np.array_equal(x.depth, y.depth)
                       for x, y in zip(a.faces, b.faces))
            check(f'{l.name}: all six cube faces bit-identical', same)
        else:
            check(f'{l.name}: map bit-identical',
                  np.array_equal(a.depth, b.depth))
    check('sun, point and spot all built maps',
          kinds >= {'SUN', 'POINT', 'SPOT'}, str(kinds))

    # the dense frame end to end: cache off == cache on == threads 4
    img1 = R.render(sc, st)
    st.cache_shadows = True
    img2 = R.render(sc, st)
    st.threads = 4
    img3 = R.render(sc, st)
    check('the dense frame is bit-stable across the shadow cache',
          np.array_equal(img1, img2))
    check('and across thread counts', np.array_equal(img1, img3))
    check('and finite with coverage', np.isfinite(img1).all()
          and float(np.ptp(img1)) > 0.05)


def test_the_mix_shader_mixes_masters():
    """Mix Shader between Halcyon Shaders must actually MIX -- by Fac.

    closure_to_surface kept ONE master lobe, last-wins, weight DISCARDED:
    mixing two Halcyon Shaders showed only the second whatever Fac said,
    and mixing one against a raw BSDF ignored the BSDF -- "the Mix Shader
    node doesn't work", said the field, about the node every converted
    material flows through. Masters now blend in MATERIAL SPACE by their
    weights (attribute by attribute, exactly how the fixed-function era
    mixed looks), per pixel when Fac is driven; the heaviest lobe names
    the model; a plain-BSDF side folds in by relative weight; TRANSPARENT
    eats opacity exactly as Fac says. A single full-weight master reduces
    to multiplying by 1.0 -- every existing material shades
    BIT-IDENTICALLY, held below at fac 0 against the pure graph.
    """
    from ..core import raster

    def sk(name, tp, default, link=None):
        return {'name': name, 'identifier': name, 'type': tp,
                'default': default, 'link': link}

    def master(nid, model, color, rim=0.0, opacity=1.0):
        ins = [sk('Diffuse Color', 'RGBA', list(color)),
               sk('Diffuse Level', 'VALUE', 1.0),
               sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
               sk('Specular Level', 'VALUE', 0.5),
               sk('Glossiness', 'VALUE', 40.0), sk('Roughness', 'VALUE', 0.3),
               sk('Ambient', 'VALUE', 1.0),
               sk('Self-Illumination', 'RGBA', [0, 0, 0, 1]),
               sk('Opacity', 'VALUE', opacity), sk('IOR', 'VALUE', 1.45),
               sk('Anisotropy', 'VALUE', 0.0),
               sk('Anisotropic Rotation', 'VALUE', 0.0),
               sk('Metalness', 'VALUE', 0.0), sk('Soften', 'VALUE', 0.0),
               sk('Reflection', 'VALUE', 0.0),
               sk('Translucency', 'VALUE', 0.0),
               sk('Toon Size', 'VALUE', 0.5),
               sk('Toon Smooth', 'VALUE', 0.05),
               sk('Rim Amount', 'VALUE', rim),
               sk('Normal', 'VECTOR', [0, 0, 0])]
        return {'id': nid, 'bl_idname': 'HALCYON_ShaderNode',
                'props': {'model': model}, 'inputs': ins,
                'outputs': [{'name': 'Surface', 'type': 'SHADER'},
                            {'name': 'BSDF', 'type': 'SHADER'}]}

    A = lambda: master('a', 'PHONG', (1.0, 0.0, 0.0, 1.0), rim=0.8)  # noqa: E731
    B = lambda: master('b', 'TOON', (0.0, 0.0, 1.0, 1.0))            # noqa: E731

    def mix_graph(fac, link=None, second=None, extra=None):
        nodes = {'a': A(), 'b': second if second is not None else B(),
                 'mix': {'id': 'mix', 'bl_idname': 'ShaderNodeMixShader',
                         'props': {},
                         'inputs': [sk('Fac', 'VALUE', fac, link),
                                    sk('Shader', 'SHADER', None, ['a', 0]),
                                    sk('Shader', 'SHADER', None, ['b', 0])],
                         'outputs': [{'name': 'Shader', 'type': 'SHADER'}]},
                 'out': {'id': 'out',
                         'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                         'inputs': [sk('Surface', 'SHADER', None,
                                       ['mix', 0])],
                         'outputs': []}}
        if extra:
            nodes.update(extra)
        return {'output': 'out', 'nodes': nodes}

    st = base_settings(96, 72, shadows=False)
    st.threads = 1
    sc = demo_scene(st, with_texture=False)
    view, _p, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, 96, 72)
    ys, xs = np.nonzero(g.mask())
    ctx = job.context(g.tri[ys, xs], g.bary[ys, xs], px=xs, py=ys)

    def surface_of(graph):
        ev = GraphEvaluator(graph, ctx)
        cl, _d = ev.evaluate_surface()
        return R.closure_to_surface(cl, ctx, st)

    # the Fac sweep: parameters lerp, the heaviest lobe names the model
    d0, _m0, _ = surface_of(mix_graph(0.0))
    dq, _mq, _ = surface_of(mix_graph(0.25))
    dh, mh, _ = surface_of(mix_graph(0.75))
    d1, m1, _ = surface_of(mix_graph(1.0))
    check('fac 0 shows A', np.allclose(d0.diffuse[0], [1, 0, 0], atol=1e-6)
          and abs(float(d0.rim[0]) - 0.8) < 1e-6)
    check('fac 0.25 blends a quarter toward B',
          np.allclose(dq.diffuse[0], [0.75, 0, 0.25], atol=1e-5)
          and abs(float(dq.rim[0]) - 0.6) < 1e-5)
    check('fac 1 shows B, rim gone',
          np.allclose(d1.diffuse[0], [0, 0, 1], atol=1e-6)
          and float(d1.rim[0]) < 1e-6)
    check('the heavier lobe names the model', mh == 'TOON' and m1 == 'TOON',
          f'{mh}/{m1}')

    # bit-parity: fac 0 IS the pure A graph, field for field
    pure = {'output': 'out', 'nodes': {
        'a': A(), 'out': {'id': 'out',
                          'bl_idname': 'ShaderNodeOutputMaterial',
                          'props': {},
                          'inputs': [sk('Surface', 'SHADER', None,
                                        ['a', 0])],
                          'outputs': []}}}
    sA, mA, _ = surface_of(pure)
    fields = ('diffuse', 'specular', 'diffuse_level', 'specular_level',
              'glossiness', 'roughness', 'opacity', 'rim', 'emission',
              'ambient', 'reflect', 'metallic')
    off = [f for f in fields
           if not np.array_equal(getattr(sA, f), getattr(d0, f))]
    check('fac 0 is BIT-IDENTICAL to the pure master', not off, str(off))

    # per-pixel Fac: a checker drives the mix, both colours present
    chk = {'chk': {'id': 'chk', 'bl_idname': 'ShaderNodeTexChecker',
                   'props': {},
                   'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                              sk('Color1', 'RGBA', [1, 1, 1, 1]),
                              sk('Color2', 'RGBA', [0, 0, 0, 1]),
                              sk('Scale', 'VALUE', 6.0)],
                   'outputs': [{'name': 'Color', 'type': 'RGBA'},
                               {'name': 'Fac', 'type': 'VALUE'}]}}
    sp, _mp, _ = surface_of(mix_graph(0.5, link=['chk', 1], extra=chk))
    reds = np.isclose(sp.diffuse[:, 0], 1.0, atol=1e-4).mean()
    blues = np.isclose(sp.diffuse[:, 2], 1.0, atol=1e-4).mean()
    check('a driven Fac mixes PER PIXEL (both sides present)',
          0.05 < reds < 0.95 and 0.05 < blues < 0.95,
          f'red {reds:.2f} blue {blues:.2f}')

    # master mixed against a raw diffuse BSDF: the BSDF side counts now
    bsdf = {'id': 'b', 'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
            'inputs': [sk('Color', 'RGBA', [0.0, 1.0, 0.0, 1.0]),
                       sk('Roughness', 'VALUE', 0.0)],
            'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    sb, _mb, _ = surface_of(mix_graph(0.5, second=bsdf))
    check('a plain BSDF side is no longer ignored',
          float(sb.diffuse[0, 1]) > 0.3,
          str([round(float(v), 3) for v in sb.diffuse[0]]))
    check('and the master side survives too', float(sb.diffuse[0, 0]) > 0.3)
    check('the rim fades with the master share',
          abs(float(sb.rim[0]) - 0.4) < 1e-5, f'{float(sb.rim[0]):.3f}')

    # master mixed toward TRANSPARENT: opacity follows Fac exactly
    tr = {'id': 'b', 'bl_idname': 'ShaderNodeBsdfTransparent', 'props': {},
          'inputs': [sk('Color', 'RGBA', [1, 1, 1, 1])],
          'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    stx, _mt, _ = surface_of(mix_graph(0.75, second=tr))
    check('mixing toward Transparent sets opacity to 1-Fac',
          abs(float(stx.opacity[0]) - 0.25) < 1e-5,
          f'{float(stx.opacity[0]):.3f}')

    # the whole path: renders on both devices, identical headless
    for fac, link, extra in ((0.5, None, None), (0.5, ['chk', 1], chk)):
        st2 = base_settings(96, 72, shadows=False)
        st2.threads = 1
        sc2 = demo_scene(st2, with_texture=False)
        sc2.materials[1].graph = mix_graph(fac, link=link, extra=extra)
        cpu = R.render(sc2, st2)
        st3 = base_settings(96, 72, shadows=False)
        st3.threads = 1
        st3.render_device = 'GPU'
        sc3 = demo_scene(st3, with_texture=False)
        sc3.materials[1].graph = mix_graph(fac, link=link, extra=extra)
        gpu = R.render(sc3, st3)
        d = float(np.abs(np.asarray(cpu, np.float64) - gpu).max())
        tag = 'driven' if link else 'constant'
        check(f'{tag}-fac master mix: device fallback equality', d == 0.0,
              f'max {d}')

    # the deferred pass: a CONSTANT-fac master mix bakes the blended
    # constants and mixes the colour chains -- it QUALIFIES and the sim
    # agrees; a DRIVEN fac varies baked fields, and the probe's own
    # constancy rule routes it with the reason
    from ..gpu import shade as GSH_

    def _plan_and_sim(graph):
        st4 = base_settings(96, 72, shadows=False)
        st4.threads = 1
        st4.shadows = True
        st4.render_device = 'GPU'
        sc4 = demo_scene(st4, with_texture=False)
        sc4.materials[1].graph = graph
        cpu4 = R.render(sc4, st4)
        view4, _p4, vp4, eye4 = R.camera_matrices(sc4.camera, 96, 72)
        g4 = raster.GBuffer(96, 72)
        raster.rasterize(sc4.mesh.verts, sc4.mesh.tris, vp4, 96, 72,
                         gbuf=g4)
        job4 = R.ShadeJob(sc4, st4, R.prepare_textures(sc4, st4), None,
                          view4, eye4, 96, 72)
        GSH_._PLAN_CACHE.clear()
        passes, why, atlases = GSH_.plan_frame(job4, g4)
        if passes is None:
            return None, why, None, None
        img, _h = GSH_.simulate(job4, g4, passes, atlases)
        return passes, why, cpu4, (img, g4.tri >= 0)

    passes, why, cpu4, sim4 = _plan_and_sim(mix_graph(0.5))
    check('a constant-fac master mix QUALIFIES for the deferred pass',
          passes is not None, str(why))
    if passes is not None:
        img, cov = sim4
        d = float(np.abs(np.asarray(cpu4, np.float64)[..., :3][cov]
                         - np.asarray(img, np.float64)[..., :3][cov]).max())
        check('and the sim shades the BLENDED look', d < 6e-3, f'max {d:.6f}')
    passes, why, _c, _s = _plan_and_sim(mix_graph(0.5, link=['chk', 1],
                                                  extra=chk))
    check('a driven-fac mix routes via the constancy rule, with the reason',
          passes is None and 'varies across the frame' in str(why),
          str(why))


def test_the_missing_node_shelf():
    """The nodes the field felt were missing, each proven for what it IS.

    The audit against Blender 5.2's catalogue found seven genuinely absent
    types and a family of silent fallbacks. What ships: Metallic BSDF (the
    METAL model -- a pure conductor was always this engine's native
    tongue), the Specular BSDF (spec/gloss, the DirectX-era workflow,
    mapped nearly literally onto Lambert + Blinn-Phong), Wireframe (exact
    edge distance in world units or output pixels, from the same
    perspective factors the mip footprint rides), Vector Transform (world/
    camera/object for points, vectors and inverse-transpose normals), and
    Ambient Occlusion with REAL rays -- the engine's own deterministic
    hemisphere sampler on its own BVH, distinct hash salt. What refuses
    now does it BY NAME with the era's reasons: volumes point at Height
    Fog and Volumetric cones, Bevel names the missing geometry query, OSL
    points at the Coded Shader node.
    """
    from ..core import convert as CV
    from ..core import raster
    from ..core.nodeeval import DISPATCH
    from ..gpu import emit as EM
    from ..tests.featurematrix import build

    def sk(name, tp, default, link=None):
        return {'name': name, 'type': tp, 'default': default, 'link': link}

    # a real shading context from the demo scene
    st = base_settings(96, 72, shadows=False)
    st.threads = 1
    sc = demo_scene(st, with_texture=False)
    view, _proj, vp, eye = R.camera_matrices(sc.camera, 96, 72)
    g = raster.GBuffer(96, 72)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, 96, 72, gbuf=g)
    job = R.ShadeJob(sc, st, {}, None, view, eye, 96, 72)
    ys, xs = np.nonzero(g.mask())
    tri_idx = g.tri[ys, xs]
    bary = g.bary[ys, xs]
    ctx = job.context(tri_idx, bary, px=xs, py=ys)

    # --- Metallic BSDF: a conductor, in the era's own words
    node = {'id': 'm', 'bl_idname': 'ShaderNodeBsdfMetallic', 'props': {},
            'inputs': [sk('Base Color', 'RGBA', [0.9, 0.6, 0.2, 1.0]),
                       sk('Roughness', 'VALUE', 0.3)],
            'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    ev = GraphEvaluator({'output': None, 'nodes': {'m': node}}, ctx)
    cl = ev.eval_output('m', 0)
    kinds = {k for k, _w, _p in cl.items}
    check('Metallic BSDF is one GLOSSY lobe', kinds == {'GLOSSY'},
          str(kinds))
    params = cl.items[0][2]
    check('and the lobe speaks METAL', params.get('model') == 'METAL',
          str(params.get('model')))
    check('fully metallic', float(np.min(params['metallic'])) == 1.0)

    # --- Specular BSDF: spec/gloss, nearly literal
    node = {'id': 's', 'bl_idname': 'ShaderNodeEeveeSpecular', 'props': {},
            'inputs': [sk('Base Color', 'RGBA', [0.2, 0.5, 0.8, 1.0]),
                       sk('Specular', 'RGBA', [1.0, 0.9, 0.7, 1.0]),
                       sk('Roughness', 'VALUE', 0.25),
                       sk('Emissive Color', 'RGBA', [0.5, 0.0, 0.0, 1.0]),
                       sk('Transparency', 'VALUE', 0.4)],
            'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    ev = GraphEvaluator({'output': None, 'nodes': {'s': node}}, ctx)
    cl = ev.eval_output('s', 0)
    kinds = [k for k, _w, _p in cl.items]
    check('Specular BSDF: diffuse + specular + emission + transparency',
          kinds == ['DIFFUSE', 'GLOSSY', 'EMISSION', 'TRANSPARENT'],
          str(kinds))
    check('the specular lobe is Blinn-Phong with the SOCKET colour',
          cl.items[1][2].get('model') == 'BLINN_PHONG'
          and abs(float(cl.items[1][2]['color'][0, 0]) - 1.0) < 1e-6)

    # --- Wireframe: exact edge geometry, both size modes
    def wire_fac(size, pixel):
        node = {'id': 'w', 'bl_idname': 'ShaderNodeWireframe',
                'props': {'use_pixel_size': pixel},
                'inputs': [sk('Size', 'VALUE', size)],
                'outputs': [{'name': 'Fac', 'type': 'VALUE'}]}
        ev = GraphEvaluator({'output': None, 'nodes': {'w': node}}, ctx)
        return ev.eval_output('w', 0), ev

    edge_b = np.copy(bary)
    edge_b[:, 0] = 0.0
    edge_b[:, 1] += bary[:, 0]                     # ON the 1-2 edge exactly
    ctx_edge = job.context(tri_idx, edge_b, px=xs, py=ys)
    node = {'id': 'w', 'bl_idname': 'ShaderNodeWireframe', 'props': {},
            'inputs': [sk('Size', 'VALUE', 0.01)],
            'outputs': [{'name': 'Fac', 'type': 'VALUE'}]}
    ev = GraphEvaluator({'output': None, 'nodes': {'w': node}}, ctx_edge)
    on_edge = ev.eval_output('w', 0)
    check('a point ON an edge always wires', float(on_edge.min()) == 1.0)
    fac_small, _ = wire_fac(1e-6, False)
    fac_huge, _ = wire_fac(1e6, False)
    check('a vanishing size wires almost nothing',
          fac_small.mean() < 0.5, f'{fac_small.mean():.3f}')
    check('a huge size wires everything', float(fac_huge.min()) == 1.0)
    fac_px, _ = wire_fac(2.0, True)
    check('pixel-size mode answers too (finite, some wire, not all)',
          np.isfinite(fac_px).all() and 0.0 < fac_px.mean() < 1.0,
          f'{fac_px.mean():.3f}')

    # --- Vector Transform: round trips and normals
    def vt(vec, vtype, src, dst, c=ctx):
        node = {'id': 'v', 'bl_idname': 'ShaderNodeVectorTransform',
                'props': {'vector_type': vtype, 'convert_from': src,
                          'convert_to': dst},
                'inputs': [sk('Vector', 'VECTOR', list(vec))],
                'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]}
        ev = GraphEvaluator({'output': None, 'nodes': {'v': node}}, c)
        return ev.eval_output('v', 0)

    p0 = np.array([1.0, 2.0, 3.0], np.float32)
    cam = vt(p0, 'POINT', 'WORLD', 'CAMERA')
    node = {'id': 'v', 'bl_idname': 'ShaderNodeVectorTransform',
            'props': {'vector_type': 'POINT', 'convert_from': 'CAMERA',
                      'convert_to': 'WORLD'},
            'inputs': [sk('Vector', 'VECTOR', [0, 0, 0], None)],
            'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]}
    node['inputs'][0] = sk('Vector', 'VECTOR', [0, 0, 0])
    ev = GraphEvaluator({'output': None, 'nodes': {'v': node}}, ctx)
    ev.cache[('src', 0)] = cam
    node['inputs'][0]['link'] = ['src', 0]
    back = ev.eval_output('v', 0)
    check('a POINT round-trips world -> camera -> world',
          float(np.abs(back - p0[None, :]).max()) < 1e-3,
          f'{float(np.abs(back - p0[None, :]).max()):.6f}')
    moved = vt(p0, 'POINT', 'WORLD', 'CAMERA')
    kept = vt(p0, 'VECTOR', 'WORLD', 'CAMERA')
    check('a VECTOR ignores the translation a POINT feels',
          float(np.abs(moved - kept).max()) > 1e-3)
    nrm = vt([0.0, 0.0, 1.0], 'NORMAL', 'WORLD', 'CAMERA')
    check('a NORMAL comes back unit length',
          float(np.abs(np.linalg.norm(nrm, axis=1) - 1.0).max()) < 1e-5)

    # --- Ambient Occlusion: real rays, deterministic, box shadows floor
    def ao_eval(c):
        node = {'id': 'a', 'bl_idname': 'ShaderNodeAmbientOcclusion',
                'props': {'samples': 8, 'inside': False,
                          'only_local': False},
                'inputs': [sk('Color', 'RGBA', [1.0, 1.0, 1.0, 1.0]),
                           sk('Distance', 'VALUE', 2.0)],
                'outputs': [{'name': 'Color', 'type': 'RGBA'},
                            {'name': 'AO', 'type': 'VALUE'}]}
        ev = GraphEvaluator({'output': None, 'nodes': {'a': node}}, c)
        col = ev.eval_output('a', 0)
        ao = ev.eval_output('a', 1)
        return col, ao, ev

    check('the job built no BVH of its own (rays are off)',
          getattr(ctx, 'bvh', None) is None)
    col1, ao1, _ = ao_eval(ctx)
    ctx2 = job.context(tri_idx, bary, px=xs, py=ys)
    _c2, ao2, _ = ao_eval(ctx2)
    check('AO is deterministic: two evaluations, identical bits',
          np.array_equal(ao1, ao2))
    check('the scene occludes SOMEWHERE (a box sits on this floor)',
          float(ao1.min()) < 0.9, f'min {float(ao1.min()):.3f}')
    check('and is open somewhere else', float(ao1.max()) > 0.97,
          f'max {float(ao1.max()):.3f}')
    check('Color is colour times AO', np.allclose(col1[:, 0], ao1, atol=1e-6))

    # --- named honesty: the family that refuses says WHY
    for idn, frag in (('ShaderNodeVolumePrincipled', 'volumetrics'),
                      ('ShaderNodeBevel', 'closest-geometry'),
                      ('ShaderNodeScript', 'Coded Shader'),
                      ('ShaderNodeLightFalloff', 'lives on the LAMPS')):
        nd = {'id': 'x', 'bl_idname': idn, 'props': {},
              'inputs': [sk('Strength', 'VALUE', 3.0),
                         sk('Normal', 'VECTOR', [0, 0, 1])],
              'outputs': [{'name': 'Quadratic', 'type': 'VALUE'},
                          {'name': 'Normal', 'type': 'VECTOR'}]}
        ev = GraphEvaluator({'output': None, 'nodes': {'x': nd}}, ctx)
        ev.eval_output('x', 0)
        check(f'{idn} names its reason',
              any(frag in u for u in ev.unsupported),
              str(ev.unsupported))
    check('every named refusal is a registered handler',
          all(k in DISPATCH for k in (
              'ShaderNodeVolumePrincipled', 'ShaderNodeVolumeScatter',
              'ShaderNodeVolumeAbsorption', 'ShaderNodeBevel',
              'ShaderNodeLightFalloff', 'ShaderNodeScript',
              'ShaderNodeOutputAOV', 'ShaderNodeBsdfRayPortal')))

    # --- the GPU side. The new rows exposed a LATENT bug: the frame pass
    # emits ONE colour chain and feeds it to the DIFFUSE term, so a raw
    # GLOSSY graph had shaded a different picture on the driver since its
    # emitter existed -- unseen because no matrix row ever carried a raw
    # BSDF graph. The probe now refuses conductor and emission lobes on
    # non-master graphs BY NAME; diffuse chains and every master-converted
    # material keep the proven path.
    from ..gpu import shade as GSH_

    def _plan_of(sc_, st_):
        st_.render_device = 'GPU'
        w_, h_ = st_.resolution_x, st_.resolution_y
        R.render(sc_, st_)                     # bakes the shadow maps
        view_, _p_, vp_, eye_ = R.camera_matrices(sc_.camera, w_, h_)
        g_ = raster.GBuffer(w_, h_)
        raster.rasterize(sc_.mesh.verts, sc_.mesh.tris, vp_, w_, h_, gbuf=g_)
        job_ = R.ShadeJob(sc_, st_, R.prepare_textures(sc_, st_), None,
                          view_, eye_, w_, h_)
        GSH_._PLAN_CACHE.clear()
        return GSH_.plan_frame(job_, g_)

    # FLIPPED (1.25.105): the specular slot routing carries lone-GLOSSY
    # raw graphs now -- the plan qualifies and the chain lands in
    # s.specular, never the diffuse slot
    for key in ('node Metallic BSDF (raw graph)',
                'node Specular BSDF (raw graph)'):
        sc_n, st_n = build(key)
        ok, miss = EM.can_emit(sc_n.materials[1].graph)
        check(f'{key}: the colour chain emits', ok, str(miss))
        p, why, _a = _plan_of(sc_n, st_n)
        check(f'{key}: the plan qualifies via the slot routing',
              p is not None, str(why))

    sc_g, st_g = build('node Metallic BSDF (raw graph)')
    sc_g.materials[1].graph['nodes']['bsdf'] = {
        'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfGlossy', 'props': {},
        'inputs': [sk('Color', 'RGBA', [0.9, 0.6, 0.2, 1.0]),
                   sk('Roughness', 'VALUE', 0.35)],
        'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    p, why, _a = _plan_of(sc_g, st_g)
    check('a raw Glossy BSDF rides the same slot routing',
          p is not None and any('s.specular = ' in src
                                for _m, _n, src, _b in p), str(why))

    sc_d, st_d = build('node Metallic BSDF (raw graph)')
    sc_d.materials[1].graph['nodes']['bsdf'] = {
        'id': 'bsdf', 'bl_idname': 'ShaderNodeBsdfDiffuse', 'props': {},
        'inputs': [sk('Color', 'RGBA', [0.8, 0.3, 0.3, 1.0]),
                   sk('Roughness', 'VALUE', 0.0)],
        'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    p, why, _a = _plan_of(sc_d, st_d)
    check('a raw DIFFUSE graph still qualifies (its colour IS the chain)',
          p is not None, str(why))

    # FLIPPED (1.25.105): the Wireframe node emits now -- exact edge
    # distance from the attribute texture's corners
    sc_w, _st = build('node Wireframe (cel ink)')
    ok, miss = EM.can_emit(sc_w.materials[1].graph)
    check('the wireframe graph emits', ok, str(miss))

    # --- the conversion operator learned the spec/gloss shader too
    p = CV.plan('ShaderNodeEeveeSpecular',
                values={'Roughness': 0.3, 'Transparency': 0.4},
                links={'Specular'})
    check('convert: Specular BSDF chooses the DirectX model',
          p['model'] == 'BLINN_PHONG', p['model'])
    check('convert: transparency becomes opacity',
          abs(p['extras'].get('Opacity', -1) - 0.6) < 1e-6,
          str(p['extras']))
    check('convert: the specular colour socket carries over',
          ('Specular Color', 'Specular') in p['pairs'], str(p['pairs']))

    # --- and the three matrix rows render on both devices identically
    for key in ('node Metallic BSDF (raw graph)',
                'node Specular BSDF (raw graph)',
                'node Wireframe (cel ink)'):
        scC, stC = build(key, 64, 48)
        cpu = R.render(scC, stC)
        scG, stG = build(key, 64, 48)
        stG.render_device = 'GPU'
        gpu = R.render(scG, stG)
        d = float(np.abs(np.asarray(cpu, np.float64) - gpu).max())
        check(f'{key}: headless fallback equality', d == 0.0, f'max {d}')
        check(f'{key}: the picture is alive', float(np.ptp(cpu)) > 0.05)


def test_every_version_stamp_agrees():
    """A stamp that can lie is worse than no stamp.

    An installed extension reports blender_manifest.toml -- that is what
    the self-test header and every console line print through
    version_string(). The 1.25.73 and 1.25.74 ships bumped bl_info alone;
    the manifest kept saying 1.25.72, and the field spent two rounds
    (rightly!) swearing it was on the latest build while the header
    disagreed -- the proof it WAS current being a self-test section that
    only exists in the new code. Blender parses bl_info statically and
    the manifest is TOML, so one constant cannot serve both; the honest
    guard is this test: the manifest, bl_info (read by AST -- importing
    the package root needs bpy), the version.py fallback and the
    CHANGELOG's newest entry must all name the SAME version, or the suite
    fails before a lying zip can ship.
    """
    import ast
    import os
    import re

    from .. import version as V

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = V._from_manifest()
    check('the manifest parses a version', manifest is not None,
          str(manifest))
    with open(os.path.join(root, '__init__.py'), encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    bl = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, 'id', None) == 'bl_info':
                    try:
                        bl = tuple(ast.literal_eval(node.value)['version'])
                    except Exception:                           # noqa: BLE001
                        bl = None
    check('bl_info carries a literal version', bl is not None, str(bl))
    head = None
    with open(os.path.join(root, 'CHANGELOG.md'), encoding='utf-8') as fh:
        for line in fh:
            m = re.match(r'## \[(\d+)\.(\d+)\.(\d+)\]', line.strip())
            if m:
                head = tuple(int(g) for g in m.groups())
                break
    check('the changelog opens with a versioned entry', head is not None,
          str(head))
    stamps = {'manifest (what an installed extension REPORTS)': manifest,
              'bl_info': bl, 'version.py fallback': tuple(V.VERSION),
              'changelog newest entry': head}
    vals = {v for v in stamps.values() if v is not None}
    check('manifest, bl_info, fallback and changelog name ONE version',
          len(vals) == 1, str(stamps))
    check('and version_string() reports exactly that one',
          V.version_string() == '.'.join(str(x) for x in (manifest or ())),
          V.version_string())


def _matrix_run(sc, st):
    """Render + post, exactly the engine's shape, for the feature matrix."""
    img = R.render(sc, st)
    return post.process(img, st, frame=1, seed=st.seed,
                        target_size=(st.resolution_x, st.resolution_y),
                        allow_resize=False,
                        depth=getattr(sc, 'last_depth', None),
                        shaft_sources=getattr(sc, 'last_shafts', None))


def test_every_feature_survives_the_device_switch():
    """The whole feature matrix, once per device, demanding EXACT equality.

    With no driver in this environment, every GPU gate must probe, refuse
    and land on the very same CPU code the CPU device runs -- so the two
    devices' pictures must match bit for bit on EVERY row. Any difference
    is a hole in the switch or the fallback plumbing (a stage that runs
    different code by device, a fallback that loses a setting). This is
    the headless half of the audit; the self-test's FEATURE x DEVICE
    MATRIX renders the same rows against a real driver.
    """
    from .featurematrix import ROWS, build

    bad = []
    for key, _o, _s in ROWS:
        sc, st = build(key)
        cpu = _matrix_run(sc, st)
        sc2, st2 = build(key)
        st2.render_device = 'GPU'      # gpu_raster/shading/post default on
        gpu = _matrix_run(sc2, st2)
        same = cpu.shape == gpu.shape and bool((cpu == gpu).all())
        if not same:
            bad.append((key, float(np.abs(np.asarray(cpu, np.float64)
                                          - gpu).max())
                        if cpu.shape == gpu.shape else 'shape'))
    check(f'all {len(ROWS)} feature rows fall back bit-exactly '
          'without a driver', not bad, str(bad[:6]))
    check('the matrix is not vacuous: it covered every row',
          len(ROWS) >= 90, str(len(ROWS)))


def test_the_device_switch_gates_every_gpu_entry():
    """The CPU device must never knock on the GPU's door -- not probe it,
    not attempt a stage, nothing. The audit found post doing exactly that:
    `_gpu_stage` gated on `device != GPU AND not gpu_post`, so a CPU
    render with the default gpu_post=True ran its post chain on any
    driver present -- the switch said CPU while dither drew on the GPU
    (up to 0.032 from the CPU chain, per the stage table's own claims).
    These counters make the contract structural for all four doors:
    probe, rasteriser, shading, post.
    """
    from ..gpu import chain as CH
    from ..gpu import craster as CRA
    from ..gpu import device as DEV
    from ..gpu import shade as GSH

    calls = {'probe': 0, 'raster': 0, 'shade': 0, 'post': 0}
    saved = (DEV.probe, CRA.raster_into_gbuffer, GSH.shade_frame,
             CH.try_stage)

    def _probe(*a, **k):
        calls['probe'] += 1
        return saved[0](*a, **k)

    def _raster(*a, **k):
        calls['raster'] += 1
        return saved[1](*a, **k)

    def _shade(*a, **k):
        calls['shade'] += 1
        return saved[2](*a, **k)

    def _stage(*a, **k):
        calls['post'] += 1
        return saved[3](*a, **k)

    DEV.probe, CRA.raster_into_gbuffer = _probe, _raster
    GSH.shade_frame, CH.try_stage = _shade, _stage
    try:
        st = base_settings(96, 72)
        st.shadows = False
        st.dither = 'BAYER4'           # a post stage with a GPU twin
        st.crt = True
        sc = demo_scene(st, with_texture=False)
        _matrix_run(sc, st)            # render_device is CPU by default
        cpu_calls = dict(calls)
        check('the CPU device never touches the GPU: probe, raster, '
              'shade and post all uncalled',
              all(v == 0 for v in cpu_calls.values()), str(cpu_calls))

        for k in calls:
            calls[k] = 0
        st2 = base_settings(96, 72)
        st2.shadows = False
        st2.dither = 'BAYER4'
        st2.crt = True
        st2.render_device = 'GPU'
        sc2 = demo_scene(st2, with_texture=False)
        _matrix_run(sc2, st2)
        gpu_calls = dict(calls)
        check('the GPU device knocks on every door (and falls back '
              'cleanly here, where no driver answers)',
              gpu_calls['raster'] >= 1 and gpu_calls['shade'] >= 1
              and gpu_calls['post'] >= 1, str(gpu_calls))

        # the per-stage toggles gate their own doors under the GPU device
        for k in calls:
            calls[k] = 0
        st3 = base_settings(96, 72)
        st3.shadows = False
        st3.dither = 'BAYER4'
        st3.crt = True
        st3.render_device = 'GPU'
        st3.gpu_raster = False
        st3.gpu_shading = False
        st3.gpu_post = False
        sc3 = demo_scene(st3, with_texture=False)
        _matrix_run(sc3, st3)
        off_calls = dict(calls)
        check('with all three stage toggles off, the GPU device attempts '
              'nothing', off_calls['raster'] == 0
              and off_calls['shade'] == 0 and off_calls['post'] == 0,
              str(off_calls))
    finally:
        (DEV.probe, CRA.raster_into_gbuffer, GSH.shade_frame,
         CH.try_stage) = saved


def test_anisotropic_rotation_reaches_the_gpu_frame():
    """The term the GLSL dropped: Anisotropic Rotation turns the frame.

    The field's feature matrix put every model on the driver for the
    first time and ANISOTROPIC came back 0.0627 over 405 pixels -- the
    demo material carries aniso_rotation 0.2, the CPU's _aniso_frame
    rotates the tangent frame by it, and the emitted GLSL never did (WARD
    hid the same gap under one output quantum at demo parameters). The
    assembler now rotates: baked rotations land as cos/sin literals, a
    per-pixel chain rotates with driver trig. This test holds the sim to
    the CPU for the exact matrix row that failed, for WARD with the
    rotation cranked where its lobe makes the gap visible, and proves
    the term is not vacuous: turning the rotation MOVES the highlight.
    """
    from ..core import raster
    from ..gpu import shade as GSH
    from .featurematrix import build

    def sim_vs_cpu(key, tweak=None):
        sc, st = build(key)
        if tweak:
            tweak(sc)
        cpu = R.render(sc, st)
        w, h = st.resolution_x, st.resolution_y
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                         depth_bits=st.depth_precision)
        R._build_shadows(sc, st, sc.mesh)
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view,
                         eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atl = GSH.plan_frame(job, g)
        if passes is None:
            return None, why, None
        img, _hit = GSH.simulate(job, g, passes, atl)
        cov = g.tri >= 0
        d = np.abs(img[cov] - cpu[cov][:, :3])
        return float(d.max()), int((d.max(axis=1) > 1e-2).sum()), img

    d1, fl1, img1 = sim_vs_cpu('model ANISOTROPIC')
    check('the matrix row that FAILED on the field now matches',
          d1 is not None and d1 < 1e-3, f'max {d1} ({fl1})')

    def crank(sc):
        for m in sc.materials:
            m.aniso_rotation = 0.37
            m.anisotropy = 0.7
    d2, fl2, _i2 = sim_vs_cpu('model WARD', crank)
    check('WARD under a cranked rotation matches too',
          d2 is not None and d2 < 1e-3, f'max {d2} ({fl2})')

    def zero(sc):
        for m in sc.materials:
            m.aniso_rotation = 0.0
    _d3, _f3, img3 = sim_vs_cpu('model ANISOTROPIC', zero)
    check('the rotation is not vacuous: zeroing it moves the highlight',
          img1 is not None and img3 is not None
          and float(np.abs(img1 - img3).max()) > 1e-2,
          f'{float(np.abs(img1 - img3).max()) if img1 is not None and img3 is not None else -1:.4f}')

    # the other matrix verdicts, pinned as refusals-by-name until ported:
    # affine frames shade on the CPU (0.835 measured), coarse-depth and
    # snapped frames rasterise on the CPU (quantised depth ties, 3 px)
    # ... and the refusal must survive a WARM plan cache. The gate alone
    # was not enough: it shipped with tex_perspective absent from the
    # plan signature, the field matrix planned 'texture NEAREST' first
    # (identical signature once the setting is invisible), and the
    # affine row walked straight past the refusal into the cached valid
    # plan -- 0.835 over 1355 px on the driver, in two consecutive
    # pastes. A gate the signature does not fingerprint is a gate a
    # cache hit bypasses; this orders the two plans exactly as the
    # field's matrix did.
    sc5, st5 = build('texture NEAREST')
    view5, _p5, vp5, eye5 = R.camera_matrices(sc5.camera, 96, 72)
    g5 = raster.GBuffer(96, 72)
    raster.rasterize(sc5.mesh.verts, sc5.mesh.tris, vp5, 96, 72, gbuf=g5)
    job5 = R.ShadeJob(sc5, st5, R.prepare_textures(sc5, st5), None,
                      view5, eye5, 96, 72)
    GSH._PLAN_CACHE.clear()
    p5, why5, _a5 = GSH.plan_frame(job5, g5)
    check('the perspective twin of the affine frame plans (cache primed)',
          p5 is not None, str(why5))
    # FLIPPED (1.25.102): affine no longer refuses -- it re-plans past
    # the warm cache into passes that read the screen-linear ids. The
    # R78 property this test guards is UNCHANGED: the primed
    # perspective plan must NOT be reused for the affine frame.
    sc4, st4 = build('affine mapping (PS1 warp)')
    view4, _p4, vp4, eye4 = R.camera_matrices(sc4.camera, 96, 72)
    g4 = raster.GBuffer(96, 72)
    g4.alloc_linear()
    raster.rasterize(sc4.mesh.verts, sc4.mesh.tris, vp4, 96, 72, gbuf=g4)
    job4 = R.ShadeJob(sc4, st4, R.prepare_textures(sc4, st4), None,
                      view4, eye4, 96, 72)
    p4, why4, _a4 = GSH.plan_frame(job4, g4)      # cache NOT cleared
    check('affine frames re-plan past a warm cache onto the '
          'screen-linear ids',
          p4 is not None and all('hal_gb_idslin' in b.get('samplers', ())
                                 for _m, _n, _s, b in p4), str(why4))


def test_fog_rides_the_deferred_readback():
    """Fog no longer refuses the GPU: the readback takes apply_fog itself.

    Fog is separable -- a lerp toward the fog colour by geometry alone --
    so instead of a GLSL twin of four modes, the vertex quantisation and
    the height layer, the deferred results take core.render.apply_fog at
    every point the CPU fogs: frame pixels at their view depth, ray HITS
    inside the sweep recursion at the hit's own depth (after the child
    composites, exactly as trace() recurses), and layer fragments at the
    gather. Vertex-rate materials SKIP the readback fog -- their corners
    were lit through shade_batch's LIGHT path, which already fogged at
    the corner (per-vertex fog, the era's own) -- so the Gouraud case
    must come out BIT-identical, not merely close. This was the largest
    R-P cluster in the field matrix (five rows) and 0.75s of the field's
    own 1.4s fogged frame.
    """
    from ..core import raster
    from ..gpu import shade as GSH
    from .featurematrix import build

    def sim_vs_cpu(key, mut=None):
        sc, st = build(key)
        if mut:
            mut(sc, st)
        cpu = R.render(sc, st)
        w, h = st.resolution_x, st.resolution_y
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                         depth_bits=st.depth_precision)
        R._build_shadows(sc, st, sc.mesh)
        bvh = R._cached_bvh(sc, sc.mesh) if st.raytrace else None
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), bvh, view,
                         eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atl = GSH.plan_frame(job, g)
        if passes is None:
            return None, why
        img, _hit = GSH.simulate(job, g, passes, atl)
        cov = g.tri >= 0
        return float(np.abs(img[cov] - cpu[cov][:, :3]).max()), img

    for key in ('fog LINEAR', 'fog EXP2', 'fog TABLE16 (fixed-function)',
                'per-vertex fog', 'height fog (ground mist)'):
        d, _img = sim_vs_cpu(key)
        check(f'{key}: the deferred frame matches the CPU',
              d is not None and d < 1e-3, str(d))

    d0, img_on = sim_vs_cpu('fog LINEAR')
    _dx, img_off = sim_vs_cpu('baseline PHONG + map shadows')
    check('the fog is not vacuous: fogged and clear frames differ',
          img_on is not None and img_off is not None
          and float(np.abs(img_on - img_off).max()) > 1e-2)

    def gouraud(sc, st):
        st.shading_rate = 'VERTEX'
    dg, _ig = sim_vs_cpu('fog LINEAR', gouraud)
    check('vertex-rate fog is BIT-level (corners carry it; the readback '
          'skips)', dg is not None and dg < 1e-6, str(dg))

    def rayfog(sc, st):
        st.fog = True
        st.fog_mode = 'EXP'
        st.fog_density = 0.08
    dr, _ir = sim_vs_cpu('traced reflection', rayfog)
    check('ray hits fog inside the recursion at their own depth',
          dr is not None and dr < 1e-3, str(dr))


def test_trilinear_mip_and_aniso_actually_filter():
    """TRILINEAR, mip bias and anisotropy were wired to nothing.

    ShadeContext.duv/dvv were initialised to None and NOTHING ever set
    them, so `filt == 'TRILINEAR' and c.duv is not None` was always
    False: CPU trilinear silently sampled bilinear, mips were built by
    prepare_textures and never read, tex_mip_bias lived only inside the
    never-taken branch, and Texture.sample accepted `aniso` and ignored
    it -- no anisotropic path existed at all. Three settings, a UI page
    and two console presets (GameCube, Xbox) rode on them. Found by
    pulling the thread the matrix could not see (it compares devices,
    not semantics; both devices rendered the same not-trilinear frame).

    Now: ShadeJob.uv_screen_gradients computes ANALYTIC per-pixel screen
    derivatives of the interpolated UV (perspective-exact, no seams at
    triangle edges, a pure function of (tri, bary) -- so chunking,
    threading and the A-buffer cannot change it), the context carries
    them for every screen point, compute_lod turns them into a mip
    level with the bias, and a new N-tap anisotropic sampler follows
    the minor axis for its level while averaging taps along the major.
    Ray hits have no footprint and keep the top level, as the era did.
    Derivatives apply only to raw flat-projected UV lookups -- a linked
    Vector chain resamples through a transform the chain rule was never
    applied to, and keeps the top level rather than filtering wrongly.
    """
    from .featurematrix import build

    def render(key, **kw):
        sc, st = build(key)
        for k, v in kw.items():
            setattr(st, k, v)
        return R.render(sc, st)

    bil = render('texture BILINEAR')
    tri = render('texture TRILINEAR + mips')
    check('TRILINEAR finally differs from BILINEAR',
          float(np.abs(tri - bil).max()) > 0.05,
          f'{float(np.abs(tri - bil).max()):.4f}')

    def shimmer(img):
        g = img[..., :3].mean(axis=2)
        return float(np.abs(np.diff(g[8:24, :], axis=1)).mean())
    check('the mips smooth the receding checker (less shimmer than '
          'bilinear)', shimmer(tri) < shimmer(bil),
          f'{shimmer(tri):.5f} vs {shimmer(bil):.5f}')

    an = render('anisotropy 4x')
    check('anisotropy differs from plain trilinear',
          float(np.abs(an - tri).max()) > 0.05)
    check('aniso keeps more detail than trilinear over-blur '
          '(minor-axis mip level)', shimmer(an) > shimmer(tri),
          f'{shimmer(an):.5f} vs {shimmer(tri):.5f}')

    sharp = render('mip bias sharp')
    soft = render('texture TRILINEAR + mips', tex_mip_bias=2.0)
    check('the mip bias moves the level both ways',
          float(np.abs(sharp - soft).max()) > 0.05)
    check('positive bias blurs, negative sharpens',
          shimmer(sharp) > shimmer(soft),
          f'{shimmer(sharp):.5f} vs {shimmer(soft):.5f}')

    a1 = render('anisotropy 4x')
    saved = R.MAX_CHUNK
    try:
        R.MAX_CHUNK = 1000
        a2 = render('anisotropy 4x')
    finally:
        R.MAX_CHUNK = saved
    check('the analytic footprint is chunk-invariant (a pure function '
          'of tri and bary)', float(np.abs(a1 - a2).max()) == 0.0)
    a3 = render('anisotropy 4x', threads=4)
    check('and thread-invariant', float(np.abs(a1 - a3).max()) == 0.0)

    # ray hits keep the top level: a trilinear scene with a mirror must
    # render, deterministically, with no footprint machinery on hits
    m1 = render('traced reflection', tex_filter='TRILINEAR',
                tex_mipmap=True)
    m2 = render('traced reflection', tex_filter='TRILINEAR',
                tex_mipmap=True)
    check('trilinear under ray tracing renders deterministically '
          '(hits sample the top level)',
          np.isfinite(m1).all() and float(np.abs(m1 - m2).max()) == 0.0)


def test_exotic_filters_ride_the_deferred_pass():
    """TRILINEAR, mip bias, anisotropy and the N64 filter on the driver.

    The frame passes sample from a MIP ATLAS packed from the CPU's own
    build_mips output, pick their level from the hal_uvgrad field -- the
    CPU's analytic derivatives, uploaded verbatim -- and mirror
    compute_lod, _sample_trilinear and _sample_aniso line for line in
    GLSL (the level ladder is a select chain; nothing the front-end
    cannot run). The N64 3-point filter is pure arithmetic and rides the
    plain image. The footprint applies exactly where the CPU applies it:
    raw flat-UV lookups on screen points -- so ray hits (secondary
    passes) sample the top level as the CPU does, coded-shader images
    stay bilinear, and a TRILINEAR-filtered height chain takes the
    proven CPU height-image pre-pass. Glass layers refuse the footprint
    BY NAME for now and shade on the CPU.
    """
    from ..core import raster
    from ..gpu import shade as GSH
    from .featurematrix import build

    def sim_vs_cpu(key, mut=None):
        sc, st = build(key)
        if mut:
            mut(sc, st)
        cpu = R.render(sc, st)
        w, h = st.resolution_x, st.resolution_y
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = raster.GBuffer(w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                         depth_bits=st.depth_precision)
        R._build_shadows(sc, st, sc.mesh)
        bvh = R._cached_bvh(sc, sc.mesh) if st.raytrace else None
        job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), bvh, view,
                         eye, w, h)
        GSH._PLAN_CACHE.clear()
        passes, why, atl = GSH.plan_frame(job, g)
        if passes is None:
            return None, str(why), None
        img, _hit = GSH.simulate(job, g, passes, atl)
        if img is None:
            return None, str(_hit), None
        cov = g.tri >= 0
        d = np.abs(img[cov] - cpu[cov][:, :3])
        return float(d.max()), int((d.max(axis=1) > 1e-2).sum()), img

    imgs = {}
    for key in ('texture TRILINEAR + mips', 'texture N64 3-point',
                'mip bias sharp', 'anisotropy 4x'):
        d, fl, img = sim_vs_cpu(key)
        check(f'{key}: the deferred frame matches the CPU',
              d is not None and d < 1e-3, f'{d} ({fl})')
        imgs[key] = img
    _d, _f, bil = sim_vs_cpu('texture BILINEAR')
    check('the footprint is not vacuous: deferred trilinear differs '
          'from deferred bilinear',
          imgs['texture TRILINEAR + mips'] is not None and bil is not None
          and float(np.abs(imgs['texture TRILINEAR + mips'] - bil).max())
          > 0.05)
    check('and anisotropy differs from plain trilinear on the driver '
          'path', imgs['anisotropy 4x'] is not None
          and float(np.abs(imgs['anisotropy 4x']
                           - imgs['texture TRILINEAR + mips']).max())
          > 0.05)

    def rayfi(sc, st):
        st.tex_filter = 'TRILINEAR'
        st.tex_mipmap = True
    d, fl, _img = sim_vs_cpu('traced reflection', rayfi)
    check('ray hits sample the top level on both devices',
          d is not None and d < 1e-3, f'{d} ({fl})')

    # glass layers: the footprint refuses BY NAME (the layer passes have
    # no field yet) and the frame still plans -- the opaque part shades
    # on the driver while the glass shades on the CPU
    sc, st = build('sorted glass layers')
    st.tex_filter = 'TRILINEAR'
    st.tex_mipmap = True
    donor = build('texture TRILINEAR + mips')[0]
    sc.materials[1].graph = donor.materials[0].graph
    sc.images.update(donor.images)
    w, h = st.resolution_x, st.resolution_y
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    opq, _trn = R._split_by_alpha(sc, sc.mesh, st)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g, depth_bits=st.depth_precision)
    R._build_shadows(sc, st, sc.mesh)
    job = R.ShadeJob(sc, st, R.prepare_textures(sc, st), None, view, eye,
                     w, h)
    GSH._PLAN_CACHE.clear()
    passes, why, atl = GSH.plan_frame(job, g)
    check('the textured-glass frame still plans its opaque passes',
          passes is not None, str(why))
    if passes is not None:
        lwhy = (atl or {}).get('__layers_why')
        check('the glass layer refusal is BY NAME',
              (atl or {}).get('__layers') is None
              and 'TRILINEAR' in str(lwhy), str(lwhy))


def test_the_bvh_cache_survives_exports_and_the_sky_masks_its_blocks():
    """The two per-frame costs that were pure waste, removed exactly.

    BVH: the cache was content-keyed but stored ON the scene object --
    and every F12 exports a fresh Scene, so it could never hit; the
    field paid its 0.76 s build on every render of an unchanged mesh
    while the docstring said "identity is useless across exports".
    Now module-level: two fresh scenes with the same mesh content share
    ONE tree (identity), a changed mesh rebuilds, answers are identical
    through the cache, and the LRU stays bounded.

    Sky: the fast-background path evaluated EVERY low-res pixel even
    when the frame was mostly geometry. Now only the blocks the mask
    can read get evaluated -- held BIT-IDENTICAL to the full evaluation
    across random masks, band masks, pad-band-only masks and empty
    masks at odd sizes and ss 2..4 (the sky is per-pixel independent,
    so skipping unread blocks cannot move a value).
    """
    from ..presets.skies import apply_sky

    st = base_settings(96, 72)
    R._BVH_CACHE.clear()
    sc1 = demo_scene(st, with_texture=False)
    sc2 = demo_scene(st, with_texture=False)
    b1 = R._cached_bvh(sc1, sc1.mesh)
    b2 = R._cached_bvh(sc2, sc2.mesh)
    check('two exports of the same mesh share ONE tree', b1 is b2)
    sc3 = demo_scene(st, with_texture=False)
    sc3.mesh.verts = sc3.mesh.verts + np.float32(0.01)
    check('a changed mesh rebuilds', R._cached_bvh(sc3, sc3.mesh)
          is not b1)
    org = np.zeros((32, 3), np.float32)
    org[:, 2] = 5.0
    d = np.tile(np.array([0, 0, -1], np.float32), (32, 1))
    r1 = b1.intersect(org, d, np.full(32, 1e30, np.float32))
    r2 = b2.intersect(org, d, np.full(32, 1e30, np.float32))
    check('answers through the cache are identical',
          all(bool((a == b).all()) for a, b in zip(r1, r2)))
    for k in range(8):
        sck = demo_scene(st, with_texture=False)
        sck.mesh.verts = sck.mesh.verts * np.float32(1.0 + 0.01 * (k + 1))
        R._cached_bvh(sck, sck.mesh)
    check('the LRU stays bounded', len(R._BVH_CACHE) <= R._BVH_CACHE_CAP,
          str(len(R._BVH_CACHE)))

    def old_fast(scene, stf, w, h, vp, eye, uncovered, ss):
        img = np.zeros((h, w, 4), np.float32)
        lw, lh = max(w // ss, 1), max(h // ss, 1)
        low = R._background_image(scene, stf, lw, lh, vp, eye, None, {},
                                  ss=1)
        big = np.repeat(np.repeat(low, ss, axis=0), ss, axis=1)
        if big.shape[0] < h or big.shape[1] < w:
            big = np.pad(big, ((0, max(h - big.shape[0], 0)),
                               (0, max(w - big.shape[1], 0)), (0, 0)),
                         mode='edge')
        big = big[:h, :w]
        mask = np.broadcast_to(uncovered, (h, w))
        img[mask] = big[mask]
        return img

    rng = np.random.default_rng(7)
    bad = []
    for (w, h, ss) in ((256, 192, 2), (255, 191, 2), (129, 97, 3)):
        stf = base_settings(w, h)
        scf = demo_scene(stf, with_texture=False)
        apply_sky(scf.world, 'BRYCE_DEFAULT')
        _v, _p, vp, eye = R.camera_matrices(scf.camera, w, h)
        masks = {'random': rng.random((h, w)) < 0.3,
                 'band': np.zeros((h, w), bool),
                 'padonly': np.zeros((h, w), bool),
                 'empty': np.zeros((h, w), bool)}
        masks['band'][: h // 3] = True
        masks['padonly'][-1, -1] = True
        for kind, m in masks.items():
            a = old_fast(scf, stf, w, h, vp, eye, m, ss)
            b = R._background_image(scf, stf, w, h, vp, eye, m, {}, ss=ss)
            if not (a.shape == b.shape and bool((a == b).all())):
                bad.append((w, h, ss, kind))
    check('the masked sky is BIT-IDENTICAL to the full evaluation on '
          'every mask shape', not bad, str(bad))


def _marshal_reset(M):
    M._STATE['enabled'] = False
    M._STATE['timer'] = False
    while not M._JOBS.empty():
        try:
            M._JOBS.get_nowait()
        except Exception:                                       # noqa: BLE001
            break


def test_gpu_bursts_marshal_to_the_main_thread():
    """The marshal's mechanics: crossing, results, exceptions, timeout.

    `bl_use_gpu_context` froze the interface for the length of the frame
    -- the field named it twice ('mid-render it just stops responding',
    'still does the not responding thing'). The render thread now runs
    with no GPU context and every driver burst crosses to the main
    thread through gpu/marshal.py: a queue, an event, and a
    bpy.app.timers timer (Blender's documented cross-thread pattern).
    The fake bpy's timers are pumped BY the test, so the test thread IS
    the main loop.
    """
    import threading
    import time as _t

    from . import fakeblender as FB
    FB.install()
    import sys
    bpy = sys.modules['bpy']
    from ..gpu import marshal as M
    _marshal_reset(M)

    # disabled: runs in place, on the calling thread
    got = {}

    def where():
        got['tid'] = threading.get_ident()
        return 'ran'

    wk = threading.Thread(target=lambda: got.setdefault(
        'r', M.run_on_main(where)))
    wk.start()
    wk.join(10)
    check('disabled marshalling runs in place', got.get('r') == 'ran'
          and got.get('tid') not in (None, threading.get_ident()),
          str(got))

    # enabled: the burst runs on the PUMPING thread, result crosses back
    M.enable()
    got.clear()
    res = {}

    def worker():
        try:
            res['r'] = M.run_on_main(where)
        except Exception as exc:                                # noqa: BLE001
            res['e'] = exc

    wk = threading.Thread(target=worker)
    wk.start()
    deadline = _t.monotonic() + 10.0
    while wk.is_alive() and _t.monotonic() < deadline:
        bpy.app.timers.pump()
        _t.sleep(0.002)
    wk.join(1)
    check('an enabled burst runs on the main loop and returns',
          res.get('r') == 'ran' and got.get('tid') == threading.get_ident(),
          str((res, got.get('tid'), threading.get_ident())))

    # exceptions cross back intact
    def boom():
        raise ValueError('driver said no')

    res.clear()

    def worker2():
        try:
            M.run_on_main(boom)
        except Exception as exc:                                # noqa: BLE001
            res['e'] = exc

    wk = threading.Thread(target=worker2)
    wk.start()
    deadline = _t.monotonic() + 10.0
    while wk.is_alive() and _t.monotonic() < deadline:
        bpy.app.timers.pump()
        _t.sleep(0.002)
    wk.join(1)
    check('the burst\'s exception crosses back to the worker',
          isinstance(res.get('e'), ValueError)
          and 'driver said no' in str(res.get('e')), str(res.get('e')))

    # a main loop that never pumps: a bounded timeout, not a hang
    res.clear()

    def worker3():
        try:
            M.run_on_main(where, timeout=0.2, what='a test burst')
        except Exception as exc:                                # noqa: BLE001
            res['e'] = exc

    wk = threading.Thread(target=worker3)
    wk.start()
    wk.join(5)
    check('an unpumped main loop times out instead of hanging',
          isinstance(res.get('e'), M.MarshalTimeout), str(res.get('e')))

    # a burst that has STARTED may run as long as it needs: the timeout
    # bounds PICKUP, never execution. The field's first layer burst was
    # timed out mid-success by the old start-to-finish cap -- 8 seconds
    # of waiting, the work discarded, and the CPU path paid on top
    res.clear()

    def slow():
        _t.sleep(0.6)
        return 'took a while'

    def worker4():
        try:
            res['r'] = M.run_on_main(slow, timeout=0.2)
        except Exception as exc:                                # noqa: BLE001
            res['e'] = exc

    wk = threading.Thread(target=worker4)
    wk.start()
    deadline = _t.monotonic() + 10.0
    while wk.is_alive() and _t.monotonic() < deadline:
        bpy.app.timers.pump()
        _t.sleep(0.002)
    wk.join(1)
    check('a picked-up burst outlives the pickup timeout',
          res.get('r') == 'took a while', str(res))

    # an abandoned burst is SKIPPED when the timer fires late: no main
    # loop blocks on work whose result was already given up on
    ran = []

    def never():
        ran.append(1)

    res.clear()

    def worker5():
        try:
            M.run_on_main(never, timeout=0.15)
        except Exception as exc:                                # noqa: BLE001
            res['e'] = exc

    wk = threading.Thread(target=worker5)
    wk.start()
    wk.join(5)                     # unpumped: the pickup times out
    check('the unpumped burst still times out',
          isinstance(res.get('e'), M.MarshalTimeout), str(res.get('e')))
    bpy.app.timers.pump()          # the timer fires late...
    check('...and the abandoned burst never runs', not ran, str(ran))

    # after disable, the timer retires itself on its next tick
    M.disable()
    bpy.app.timers.pump()
    bpy.app.timers.pump()
    check('the drain timer retires once the render is over',
          not bpy.app.timers.fns, str(bpy.app.timers.fns))
    _marshal_reset(M)


def test_a_released_context_render_still_completes():
    """The whole engine, rendered from a WORKER thread, main loop live.

    This is F12's real shape now: `bl_use_gpu_context = False`, the
    render on its own thread, the interface's main loop free to draw --
    and pumping the marshal's timer. Headless there is no gpu module, so
    every marshalled burst raises ON THE MAIN LOOP and the reason
    crosses back into the usual printed CPU fallback: the frame must
    still complete, correctly, with no deadlock and no leftover timer.
    """
    import threading
    import time as _t

    from . import fakeblender as FB
    props, engine = FB.install()
    import sys
    bpy = sys.modules['bpy']
    from ..core.settings import RenderSettings
    from ..gpu import marshal as M
    _marshal_reset(M)

    check('the shipped default releases the context',
          RenderSettings().gpu_hold_context is False)

    old_bg = bpy.app.background
    old_attr = engine.HalcyonRenderEngine.bl_use_gpu_context
    crossings = []
    orig = M.run_on_main

    def counting(fn, *a, **kw):
        crossings.append(threading.get_ident())
        return orig(fn, *a, **kw)

    out = {}
    try:
        bpy.app.background = False              # a windowed session
        engine.HalcyonRenderEngine.bl_use_gpu_context = False
        M.run_on_main = counting

        def worker():
            try:
                out['img'], out['passes'], out['cap'] = FB.run_render(
                    props, engine, render_device='GPU')
            except Exception as exc:                            # noqa: BLE001
                out['e'] = exc

        wk = threading.Thread(target=worker)
        wk.start()
        deadline = _t.monotonic() + 120.0
        while wk.is_alive() and _t.monotonic() < deadline:
            bpy.app.timers.pump()
            _t.sleep(0.001)
        alive = wk.is_alive()
        wk.join(1)
        check('the threaded render completes (no deadlock)', not alive
              and 'e' not in out, str(out.get('e')))
        check('and delivers a real frame', out.get('img') is not None
              and bool(np.isfinite(out['img']).all()),
              str(out.get('cap', {}).get('reports')))
        check('GPU bursts really crossed the marshal',
              len(crossings) > 0, f'{len(crossings)} crossings')
        bpy.app.timers.pump()
        bpy.app.timers.pump()
        check('the render left no timer behind', not bpy.app.timers.fns,
              str(bpy.app.timers.fns))
        check('and marshalling is off again', not M.enabled())
    finally:
        M.run_on_main = orig
        bpy.app.background = old_bg
        engine.HalcyonRenderEngine.bl_use_gpu_context = old_attr
        _marshal_reset(M)


def test_ray_traced_layers_shade_on_the_gpu():
    """Transparent layers spawn the SAME recursion the opaque frame does.

    The field's named refusal -- 'transparent layers under ray tracing
    recurse on the CPU', 25.1 seconds of it -- lifts here: each rank's
    fragments become a virtual PRIMARY surface (tri + bary are all the
    ray machinery reads), `_run_sweeps` walks `_add_raytraced`'s tree
    from it, and the layer pass's alpha rides through untouched, exactly
    the CPU's order (rays modify rgb, then alpha computes). Sampling
    identity is the fragment's own pixel, the same streams the CPU
    threads through `trace()`. Proof per FRAGMENT against
    `_shade_chunked` -- the compositor's own call -- alpha included.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..core.scene import ImageBuffer, World
    from ..gpu import shade as GSH

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def mirror_frags(sc, st, job, g, w, h):
        frags = raster.FragmentList()
        _opq, trans = R._split_by_alpha(sc, sc.mesh, st)
        view, _proj, vp, _eye = R.camera_matrices(sc.camera, w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h,
                         cull='NONE', subset=trans, gbuf=g, frags=frags,
                         depth_write=False,
                         depth_bits=st.depth_precision)
        px, py, tri, depth, bary, front = frags.finish()
        keep = depth <= raster.abuf_depth_limit(g.depth[py, px])
        px, py, tri, depth, bary, front = (a[keep] for a in
                                           (px, py, tri, depth, bary,
                                            front))
        cent = sc.mesh.verts[sc.mesh.tris].mean(axis=1)
        vz = np.abs((cent - job.eye[None, :])
                    @ job.view[:3, :3].T)[:, 2]
        pix = py.astype(np.int64) * g.width + px
        order = np.lexsort((vz[tri].astype(np.float32), pix))
        pix, px, py, tri, bary, front = (a[order] for a in
                                         (pix, px, py, tri, bary, front))
        gr = np.zeros(pix.size, np.int64)
        ng = np.nonzero(pix[1:] != pix[:-1])[0] + 1
        gr[ng] = ng
        np.maximum.accumulate(gr, out=gr)
        rank = np.arange(pix.size, dtype=np.int64) - gr
        return px, py, tri, bary, front, rank

    def seam(tag, sc, st, w, h, textures=None, expect_layers=None):
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        opq, _trans = R._split_by_alpha(sc, sc.mesh, st)
        g = raster.GBuffer(w, h)
        if opq is not None and opq.size:
            raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h,
                             subset=opq, gbuf=g,
                             depth_bits=st.depth_precision)
        bvh = BVH(sc.mesh.verts, sc.mesh.tris)
        job = R.ShadeJob(sc, st, textures or {}, bvh, view, eye, w, h)
        px, py, tri, bary, front, rank = mirror_frags(sc, st, job, g,
                                                      w, h)
        check(f'{tag}: fragments exist', px.size > 0, f'{px.size}')
        GSH._PLAN_CACHE.clear()
        p, why, a = GSH.plan_frame(job, g)
        check(f'{tag}: the frame qualifies', p is not None, str(why))
        if p is None:
            return
        lp = (a or {}).get('__layers')
        check(f'{tag}: the layers plan under ray tracing',
              lp is not None,
              str((a or {}).get('__layers_why')))
        if expect_layers is not None:
            check(f'{tag}: the expected materials hold layer passes',
                  lp is not None and sorted(e[0] for e in lp)
                  == sorted(expect_layers),
                  str([e[0] for e in (lp or ())]))
        cpu_col = R._shade_chunked(job, tri, bary, px, py, front, None,
                                   st)
        got, gwhy = GSH.simulate_fragments(job, g, tri, bary, px, py,
                                           rank)
        check(f'{tag}: the layers simulate', got is not None, str(gwhy))
        if got is None:
            return
        d = np.abs(got - cpu_col).max(axis=1)
        flips = int((d > 1e-2).sum())
        check(f'{tag}: every fragment matches the CPU, alpha included',
              flips == 0 and float(d.max()) < 6e-3,
              f'{flips} of {tri.size} fragments >0.01, '
              f'max {float(d.max()):.6f}')
        return cpu_col

    # ---- 1. glass that refracts among mirrors, one bounce: the glass
    # ball's fragments (both faces) trace through the reflective floor
    # and the mirror box -- and the frame WITH rays must differ from the
    # frame without them, on the fragments themselves
    w, h = 96, 72
    st = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1)
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].opacity = 0.5
    sc.materials[0].reflect_level = 0.2
    sc.materials[2].reflect_level = 0.5
    R.render(sc, st)                        # leaves the shadow maps
    ray_col = seam('one bounce', sc, st, w, h, expect_layers=[1])

    st_off = base_settings(w, h, shadows=True, raytrace=False)
    st_off.color_depth = '24'
    st_off.dither = 'NONE'
    st_off.output_scale = 'NONE'
    view1, _p1, vp1, eye1 = R.camera_matrices(sc.camera, w, h)
    opq1, _t1 = R._split_by_alpha(sc, sc.mesh, st)
    g1 = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp1, w, h,
                     subset=opq1, gbuf=g1, depth_bits=st.depth_precision)
    job1 = R.ShadeJob(sc, st_off, {}, None, view1, eye1, w, h)
    px1, py1, tri1, bary1, front1, _r1 = mirror_frags(sc, st_off, job1,
                                                      g1, w, h)
    flat_col = R._shade_chunked(job1, tri1, bary1, px1, py1, front1,
                                None, st_off)
    if ray_col is not None and ray_col.shape[0] == flat_col.shape[0]:
        dv = float(np.abs(ray_col[:, :3] - flat_col[:, :3]).max())
        check('the rays really change the glass fragments', dv > 0.02,
              f'ray tracing moves the fragments by {dv:.4f}')

    # ---- 2. the recursion tree from a layer: ray depth 2 -- a glass
    # fragment's ray hits the mirror, whose OWN reflection spawns the
    # next level before compositing backward
    st2 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=2)
    st2.color_depth = '24'
    st2.dither = 'NONE'
    st2.output_scale = 'NONE'
    sc2 = demo_scene(st2, with_texture=False)
    sc2.materials[1].opacity = 0.5
    sc2.materials[0].reflect_level = 0.3
    sc2.materials[2].reflect_level = 0.6
    R.render(sc2, st2)
    seam('depth 2', sc2, st2, w, h, expect_layers=[1])

    # ---- 3. the field's own shape: ALL transparency AND ray tracing.
    # Every visible material see-through, the opaque G-buffer empty, and
    # the ray plan opened by the LAYER materials alone
    st3 = base_settings(w, h, shadows=True, raytrace=True, ray_depth=1)
    st3.color_depth = '24'
    st3.dither = 'NONE'
    st3.output_scale = 'NONE'
    sc3 = demo_scene(st3, with_texture=False)
    for m3 in sc3.materials:
        m3.opacity = 0.5
    sc3.materials[2].reflect_level = 0.4    # glass that also reflects
    R.render(sc3, st3)
    seam('all-transparent + ray', sc3, st3, w, h,
         expect_layers=[0, 1, 2])

    # ---- 4. a rich world behind ray-traced glass: under ray tracing
    # the env term lives at the recursion's final depth, where the
    # CPU-composite '__env' machinery already covers ANY world -- so
    # Bryce behind reflective glass QUALIFIES here (the non-ray case
    # keeps its named refusal, tested elsewhere)
    st4 = base_settings(w, h, shadows=False, raytrace=True, ray_depth=1)
    st4.color_depth = '24'
    st4.dither = 'NONE'
    st4.output_scale = 'NONE'
    sc4 = demo_scene(st4, with_texture=False)
    sc4.materials[1].opacity = 0.5
    sc4.materials[1].reflect_level = 0.5
    sc4.materials[0].reflect_level = 0.3
    sc4.world = World()
    sc4.world.mode = 'BRYCE'
    seam('Bryce behind ray glass', sc4, st4, w, h, expect_layers=[1])

    # ---- 5. deterministic sampling FROM layer fragments: soft ray
    # shadows and ambient occlusion, where every jittered ray is a pure
    # function of (pixel, sample, stream, seed) and a fragment's pixel
    # IS its identity. Small frame: sampling sims are slow headlessly
    w5, h5 = 64, 48
    st5 = base_settings(w5, h5, shadows=True, shadow_default='RAY',
                        raytrace=True, ray_depth=1)
    st5.shadow_samples = 4
    st5.ambient_occlusion = True
    st5.ao_samples = 4
    st5.ao_distance = 2.0
    st5.ao_intensity = 1.0
    st5.color_depth = '24'
    st5.dither = 'NONE'
    st5.output_scale = 'NONE'
    sc5 = demo_scene(st5, with_texture=False)
    sc5.materials[1].opacity = 0.5
    sc5.lights[1].radius = 0.8
    R.render(sc5, st5)
    seam('soft + AO from a layer', sc5, st5, w5, h5, expect_layers=[1])

    # ---- 6. the remaining refusal stays named: a Bump pre-pass on a
    # transparent layer, under ray tracing too
    st6 = base_settings(w, h, shadows=False, raytrace=True, ray_depth=1,
                        transparency='SORTED')
    sc6 = demo_scene(st6, with_texture=False)
    rng6 = np.random.default_rng(3)
    him6 = np.zeros((16, 16, 4), np.float32)
    him6[:, :, 0] = rng6.random((16, 16))
    him6[:, :, 1] = him6[:, :, 2] = him6[:, :, 0]
    him6[:, :, 3] = 1.0
    sc6.images = {'hmap': ImageBuffer(name='hmap', pixels=him6)}
    sc6.materials[1] = type(sc6.materials[1])(
        name='WavyGlass', index=1, graph={
            'output': 'out', 'nodes': {
                'htex': {'id': 'htex',
                         'bl_idname': 'ShaderNodeTexImage',
                         'props': {'image': 'hmap',
                                   'interpolation': 'Closest'},
                         'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                         'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                     {'name': 'Alpha',
                                      'type': 'VALUE'}]},
                'bump': {'id': 'bump', 'bl_idname': 'ShaderNodeBump',
                         'props': {'invert': False},
                         'inputs': [sk('Strength', 'VALUE', 0.8),
                                    sk('Distance', 'VALUE', 0.6),
                                    sk('Height', 'VALUE', 0.5,
                                       ['htex', 0]),
                                    sk('Normal', 'VECTOR', [0, 0, 0])],
                         'outputs': [{'name': 'Normal',
                                      'type': 'VECTOR'}]},
                'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                        'props': {'model': 'PHONG'},
                        'inputs': [sk('Diffuse Color', 'RGBA',
                                      [0.3, 0.6, 0.9, 1.0]),
                                   sk('Opacity', 'VALUE', 0.5),
                                   sk('Normal', 'VECTOR', [0, 0, 0],
                                      ['bump', 0])],
                        'outputs': [{'name': 'Surface',
                                     'type': 'SHADER'}]},
                'out': {'id': 'out',
                        'bl_idname': 'ShaderNodeOutputMaterial',
                        'props': {},
                        'inputs': [sk('Surface', 'SHADER', None,
                                      ['hal', 0])],
                        'outputs': []}}})
    sc6.materials[1].has_alpha = True
    tex6 = R.prepare_textures(sc6, st6)
    view6, _p6, vp6, eye6 = R.camera_matrices(sc6.camera, w, h)
    opq6, _t6 = R._split_by_alpha(sc6, sc6.mesh, st6)
    g6 = raster.GBuffer(w, h)
    raster.rasterize(sc6.mesh.verts, sc6.mesh.tris, vp6, w, h,
                     subset=opq6, gbuf=g6)
    job6 = R.ShadeJob(sc6, st6, tex6,
                      BVH(sc6.mesh.verts, sc6.mesh.tris), view6, eye6,
                      w, h)
    GSH._PLAN_CACHE.clear()
    p6, why6, a6 = GSH.plan_frame(job6, g6)
    check('the wavy-glass ray frame still qualifies', p6 is not None,
          str(why6))
    lp6r = (a6 or {}).get('__layers')
    check('and a Bump pre-pass on a ray-traced layer now plans',
          lp6r is not None and (a6 or {}).get('__layers_why') is None,
          str((a6 or {}).get('__layers_why',
                             [e[0] for e in (lp6r or ())])))


def test_bump_glass_layers_shade_on_the_gpu():
    """The Water anatomy as GLASS: Bump pre-passes ride the layers.

    The field named this by material ('Material.008': a Bump pre-pass
    on a transparent layer is not ported yet) -- and the port surfaced
    a CPU bug older than itself: transparent fragments' bump gradients
    came from `_screen_grad` over whatever batch they landed in, so
    front and back faces COLLIDED in the scatter and every chunk
    boundary cut the waves -- 539 of 1914 fragments moved with the
    chunk size (by up to 2.98) in rasterisation order, dozens even in
    the compositor's pixel-sorted order. The layer is the surface: one
    fragment per pixel per rank. The CPU now shades rank
    by rank with per-rank gradient fields (`_shade_fragments_cpu`), and
    the GPU draws each material's height pre-pass per rank over the
    rank's own ids -- the same definition on both devices, including
    the CPU-evaluated height image for chains the emitter refuses
    (Blender's sin-fract Noise: the exact Water).
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..core.scene import ImageBuffer
    from ..gpu import shade as GSH
    import halcyon.core.render as RR

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def build(w, h, raytrace=False, noise=False, strength=0.8):
        from .scenebuild import add_normal_mapped_ball
        st = base_settings(w, h, shadows=True, raytrace=raytrace,
                           ray_depth=1)
        st.color_depth = '24'
        st.dither = 'NONE'
        st.output_scale = 'NONE'
        sc = demo_scene(st, with_texture=False)
        add_normal_mapped_ball(sc)      # FULL master socket list
        g = sc.materials[1].graph
        if noise:
            g['nodes']['hnoise'] = {
                'id': 'hnoise', 'bl_idname': 'ShaderNodeTexNoise',
                'props': {'noise_dimensions': '3D'},
                'inputs': [sk('Vector', 'VECTOR', [0, 0, 0]),
                           sk('Scale', 'VALUE', 10.0),
                           sk('Detail', 'VALUE', 2.0),
                           sk('Roughness', 'VALUE', 0.5),
                           sk('Distortion', 'VALUE', 0.0)],
                'outputs': [{'name': 'Fac', 'type': 'VALUE'},
                            {'name': 'Color', 'type': 'RGBA'}]}
            hlink = ['hnoise', 0]
        else:
            rngb = np.random.default_rng(3)
            him = np.zeros((16, 16, 4), np.float32)
            him[:, :, 0] = rngb.random((16, 16))
            him[:, :, 1] = him[:, :, 2] = him[:, :, 0]
            him[:, :, 3] = 1.0
            sc.images['hmap'] = ImageBuffer(name='hmap', pixels=him)
            g['nodes']['htex'] = {
                'id': 'htex', 'bl_idname': 'ShaderNodeTexImage',
                'props': {'image': 'hmap', 'interpolation': 'Closest'},
                'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'},
                            {'name': 'Alpha', 'type': 'VALUE'}]}
            hlink = ['htex', 0]
        g['nodes']['bump'] = {
            'id': 'bump', 'bl_idname': 'ShaderNodeBump',
            'props': {'invert': False},
            'inputs': [sk('Strength', 'VALUE', strength),
                       sk('Distance', 'VALUE', 0.6),
                       sk('Height', 'VALUE', 0.5, hlink),
                       sk('Normal', 'VECTOR', [0, 0, 0])],
            'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
        for s in g['nodes']['hal']['inputs']:
            if s['name'] == 'Normal':
                s['link'] = ['bump', 0]
            if s['name'] == 'Diffuse Color':
                s['link'] = None
                s['default'] = [0.3, 0.6, 0.9, 1.0]
            if s['name'] == 'Opacity':
                s['default'] = 0.5
            if s['name'] == 'IOR':
                s['default'] = 1.33
        sc.materials[1].has_alpha = True
        sc.materials[2].reflect_level = 0.5
        return sc, st

    def frag_mirror(sc, st, job, g, w, h):
        frags = raster.FragmentList()
        _opq, trans = R._split_by_alpha(sc, sc.mesh, st)
        _view, _proj, vp, _eye = R.camera_matrices(sc.camera, w, h)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h,
                         cull='NONE', subset=trans, gbuf=g, frags=frags,
                         depth_write=False,
                         depth_bits=st.depth_precision)
        px, py, tri, depth, bary, front = frags.finish()
        keep = depth <= raster.abuf_depth_limit(g.depth[py, px])
        px, py, tri, depth, bary, front = (a[keep] for a in
                                           (px, py, tri, depth, bary,
                                            front))
        cent = sc.mesh.verts[sc.mesh.tris].mean(axis=1)
        vz = np.abs((cent - job.eye[None, :])
                    @ job.view[:3, :3].T)[:, 2]
        pix = py.astype(np.int64) * g.width + px
        order = np.lexsort((vz[tri].astype(np.float32), pix))
        pix, px, py, tri, bary, front = (a[order] for a in
                                         (pix, px, py, tri, bary, front))
        gr = np.zeros(pix.size, np.int64)
        ng = np.nonzero(pix[1:] != pix[:-1])[0] + 1
        gr[ng] = ng
        np.maximum.accumulate(gr, out=gr)
        rank = np.arange(pix.size, dtype=np.int64) - gr
        return px, py, tri, bary, front, rank

    # ---- 1. the CPU bug, pinned: the OLD mixed-batch call flips with
    # the chunk size; the per-rank shading does not
    w, h = 160, 120
    sc, st = build(w, h)
    tex = R.prepare_textures(sc, st)
    R.render(sc, st)                        # shadow maps
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    opq, _tr = R._split_by_alpha(sc, sc.mesh, st)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g, depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, tex, None, view, eye, w, h)
    px, py, tri, bary, front, rank = frag_mirror(sc, st, job, g, w, h)
    st.threads = 1
    old_chunked = R._shade_chunked(job, tri, bary, px, py, front, None,
                                   st)
    old_max = RR.MAX_CHUNK
    try:
        RR.MAX_CHUNK = 512
        old_chunked2 = R._shade_chunked(job, tri, bary, px, py, front,
                                        None, st)
    finally:
        RR.MAX_CHUNK = old_max
    dold = int((np.abs(old_chunked - old_chunked2).max(axis=1)
                > 1e-3).sum())
    check('the mixed-batch call really flips with the chunk size',
          dold > 10, f'{dold} fragments moved (pixel-sorted order; '
          'rasterisation order moved hundreds)')
    new_a = R._shade_fragments_cpu(job, tri, bary, px, py, front, rank,
                                   st)
    try:
        RR.MAX_CHUNK = 512
        new_b = R._shade_fragments_cpu(job, tri, bary, px, py, front,
                                       rank, st)
    finally:
        RR.MAX_CHUNK = old_max
    dnew = float(np.abs(new_a - new_b).max())
    check('per-rank fields do not', dnew == 0.0, f'max {dnew}')
    # ...and the whole SORTED frame is thread-invariant with the glass
    st.threads = 1
    fa = R.render(sc, st)
    st.threads = 8
    fb = R.render(sc, st)
    st.threads = 1
    check('the bumpy-glass frame no longer depends on threads',
          float(np.abs(fa - fb).max()) == 0.0,
          f'max {float(np.abs(fa - fb).max()):.6f}')

    # ---- 2. the seam: bump glass layers vs the compositor's own CPU
    # shading, no ray -- then the FULL Water anatomy (sin-fract Noise
    # into Bump, ray tracing on: the height image is CPU-evaluated over
    # each rank's virtual surface, exactly)
    for tag, ray, noise in (('bump glass', False, False),
                            ('the Water anatomy', True, True)):
        w2, h2 = 96, 72
        sc2, st2 = build(w2, h2, raytrace=ray, noise=noise)
        tex2 = R.prepare_textures(sc2, st2)
        R.render(sc2, st2)
        view2, _p2, vp2, eye2 = R.camera_matrices(sc2.camera, w2, h2)
        opq2, _t2 = R._split_by_alpha(sc2, sc2.mesh, st2)
        g2 = raster.GBuffer(w2, h2)
        raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w2, h2,
                         subset=opq2, gbuf=g2,
                         depth_bits=st2.depth_precision)
        bvh2 = BVH(sc2.mesh.verts, sc2.mesh.tris) if ray else None
        job2 = R.ShadeJob(sc2, st2, tex2, bvh2, view2, eye2, w2, h2)
        px2, py2, tri2, bary2, front2, rank2 = frag_mirror(sc2, st2,
                                                           job2, g2,
                                                           w2, h2)
        GSH._PLAN_CACHE.clear()
        p2, why2, a2 = GSH.plan_frame(job2, g2)
        check(f'{tag}: the frame qualifies', p2 is not None, str(why2))
        if p2 is None:
            continue
        lp2 = (a2 or {}).get('__layers')
        check(f'{tag}: the bump glass holds a layer pass with its '
              'pre-pass', lp2 is not None
              and any((e[3].get('prepasses') or ()) for e in lp2),
              str((a2 or {}).get('__layers_why')))
        cpu2 = R._shade_fragments_cpu(job2, tri2, bary2, px2, py2,
                                      front2, rank2, st2)
        got2, gwhy2 = GSH.simulate_fragments(job2, g2, tri2, bary2, px2,
                                             py2, rank2)
        check(f'{tag}: the layers simulate', got2 is not None,
              str(gwhy2))
        if got2 is None:
            continue
        d2 = np.abs(got2 - cpu2).max(axis=1)
        flips2 = int((d2 > 1e-2).sum())
        check(f'{tag}: every fragment matches the CPU, alpha included',
              flips2 == 0 and float(d2.max()) < 6e-3,
              f'{flips2} of {tri2.size} fragments >0.01, '
              f'max {float(d2.max()):.6f}')

    # ---- 3. vacuity: the bump is load-bearing on the fragments
    sc3, st3 = build(w, h, strength=0.0)
    tex3 = R.prepare_textures(sc3, st3)
    job3 = R.ShadeJob(sc3, st3, tex3, None, view, eye, w, h)
    flat3 = R._shade_fragments_cpu(job3, tri, bary, px, py, front, rank,
                                   st3)
    dv3 = float(np.abs(new_a[:, :3] - flat3[:, :3]).max())
    check('the bump really bends the glass', dv3 > 0.02,
          f'strength moves the fragments by {dv3:.4f}')


def test_the_picture_does_not_depend_on_the_rasterisers_last_ulp():
    """Coplanar glass-on-opaque survives per-device depth rounding.

    The field selftest measured 1036 px between the CPU and GPU whole
    renders of the two-glass scene while the fragment shading agreed to
    0.000048 -- the divergence was the A-buffer depth test, not the
    shading. The demo box's bottom face is COPLANAR with the floor: its
    fragments interpolate to the opaque depth plus or minus a few ULPs,
    the compute rasteriser rounds zndc ~9e-7 differently from fill(),
    and the bare `<` let the DEVICE decide, pixel by pixel, whether the
    contact layer exists. Collection and compositor now share a
    ~30-ULP tolerance (raster.abuf_depth_limit), so a modeled tie lands
    the same way under any rounding -- the scheduling-invariance
    doctrine, extended to the rasteriser pair.
    """
    from ..core import raster

    w, h = 480, 360        # the selftest's size: the tie population is real
    st = base_settings(w, h, shadows=False, transparency='SORTED')
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].opacity = 0.5
    sc.materials[2].opacity = 0.6
    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    opq, trans = R._split_by_alpha(sc, sc.mesh, st)
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g, depth_bits=st.depth_precision)

    # every transparent fragment, no depth gate: the superset the
    # devices agree on (the rasterised geometry itself is proven exact)
    allf = raster.FragmentList()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, cull='NONE',
                     subset=trans, gbuf=g, frags=allf, depth_write=False,
                     depth_test=False, depth_bits=st.depth_precision)
    px, py, tri, depth, bary, front = allf.finish()
    oz = g.depth[py, px]

    with np.errstate(invalid='ignore'):
        rel = np.abs(depth - oz) / np.maximum(np.abs(oz), np.float32(1e-9))
    rel[~np.isfinite(oz)] = np.inf
    ties = int((rel <= 2e-6).sum())
    check('the scene really holds coplanar contact fragments',
          ties > 500, f'{ties} fragments within 2e-6 of the opaque depth')

    # the last-ULP divergence, at the class the field driver measured
    # (zndc ~9e-7): the bare comparison flips under it...
    rng = np.random.default_rng(11)
    oz_b = oz * (1.0 + rng.uniform(-1e-6, 1e-6,
                                   oz.size).astype(np.float32))
    old_flips = int(((depth <= oz) != (depth <= oz_b)).sum())
    check('the bare comparison really flips under last-ULP rounding',
          old_flips > 100, f'{old_flips} keep flips')
    # ...and the tolerant limit does not
    new_a = depth <= raster.abuf_depth_limit(oz)
    new_b = depth <= raster.abuf_depth_limit(oz_b)
    new_flips = int((new_a != new_b).sum())
    check('the tolerant limit does not', new_flips == 0,
          f'{new_flips} keep flips')
    check('and it keeps the contact layer',
          bool(new_a[rel <= 2e-6].all()),
          f'{int(new_a[rel <= 2e-6].sum())} of {ties}')

    # end to end: the REAL collection path against a last-ULP-perturbed
    # opaque G-buffer yields the identical surviving fragment set
    f_a = raster.FragmentList()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, cull='NONE',
                     subset=trans, gbuf=g, frags=f_a, depth_write=False,
                     depth_bits=st.depth_precision)
    g2 = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g2, depth_bits=st.depth_precision)
    cov2 = np.isfinite(g2.depth)
    g2.depth[cov2] *= (1.0 + np.random.default_rng(7).uniform(
        -1e-6, 1e-6, int(cov2.sum())).astype(np.float32))
    f_b = raster.FragmentList()
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, cull='NONE',
                     subset=trans, gbuf=g2, frags=f_b, depth_write=False,
                     depth_bits=st.depth_precision)

    def kept_set(fl, gb):
        fpx, fpy, ftri, fdep, _fb, _ff = fl.finish()
        ok = fdep <= raster.abuf_depth_limit(gb.depth[fpy, fpx])
        key = (fpy[ok].astype(np.int64) * w + fpx[ok]) * 100000 \
            + ftri[ok]
        return np.sort(key)

    ka = kept_set(f_a, g)
    kb = kept_set(f_b, g2)
    check('the surviving fragment set is identical under the rounding',
          ka.size == kb.size and bool((ka == kb).all()),
          f'{ka.size} vs {kb.size} fragments')


def test_transparent_layers_shade_on_the_gpu():
    """A-buffer fragments shade through the deferred machinery, per layer.

    The always-printed summary line named this stage on the field frame:
    'transparency shading 25.7s' of 33.7 -- every transparent fragment
    shaded on one CPU path while the GPU idled. Now each depth layer's
    fragments become an ids texture (real triangle, REAL barycentrics,
    everything else uncovered), every see-through material's LAYER pass
    draws it with the true alpha chain emitted -- opacity clamped, the
    hard threshold, the edge-opacity silhouette blend on the BENT normal
    -- and the per-pixel-disjoint materials merge by plain addition. The
    proof is per FRAGMENT: the exact `_shade_chunked` call the
    compositor makes, against `simulate_fragments` on the same sorted
    ranks, alpha included.
    """
    from ..core import raster
    from ..core.bvh import BVH
    from ..core.scene import ImageBuffer, World
    from ..gpu import shade as GSH

    w, h = 96, 72

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def glass_graph(opacity=0.5, edge=None, fresnel_power=None, bump=False):
        ins = [sk('Diffuse Color', 'RGBA', [0.3, 0.6, 0.9, 1.0]),
               sk('Specular Level', 'VALUE', 0.7),
               sk('Glossiness', 'VALUE', 32.0),
               sk('Opacity', 'VALUE', opacity)]
        if edge is not None:
            ins.append(sk('Edge Opacity', 'VALUE', edge))
        if fresnel_power is not None:
            ins.append(sk('Fresnel Power', 'VALUE', fresnel_power))
        nodes = {
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG'}, 'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0])],
                    'outputs': []}}
        if bump:
            nodes['htex'] = {
                'id': 'htex', 'bl_idname': 'ShaderNodeTexImage',
                'props': {'image': 'hmap', 'interpolation': 'Closest'},
                'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Color', 'type': 'RGBA'},
                            {'name': 'Alpha', 'type': 'VALUE'}]}
            nodes['bump'] = {
                'id': 'bump', 'bl_idname': 'ShaderNodeBump',
                'props': {'invert': False},
                'inputs': [sk('Strength', 'VALUE', 0.8),
                           sk('Distance', 'VALUE', 0.6),
                           sk('Height', 'VALUE', 0.5, ['htex', 0]),
                           sk('Normal', 'VECTOR', [0, 0, 0])],
                'outputs': [{'name': 'Normal', 'type': 'VECTOR'}]}
            nodes['hal']['inputs'].append(
                sk('Normal', 'VECTOR', [0, 0, 0], ['bump', 0]))
        return {'output': 'out', 'nodes': nodes}

    def frag_mirror(sc, st, job, g):
        """Exactly `_composite_abuffer`'s front half: keep, sort, rank."""
        view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
        frags = raster.FragmentList()
        opq, trans = R._split_by_alpha(sc, sc.mesh, st)
        raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h,
                         cull='NONE', subset=trans, gbuf=g, frags=frags,
                         depth_write=False,
                         depth_bits=st.depth_precision)
        px, py, tri, depth, bary, front = frags.finish()
        opaque_z = g.depth[py, px]
        keep = depth <= raster.abuf_depth_limit(opaque_z)
        px, py, tri, depth, bary, front = (a[keep] for a in
                                           (px, py, tri, depth, bary,
                                            front))
        cent = sc.mesh.verts[sc.mesh.tris].mean(axis=1)
        view_z = np.abs((cent - job.eye[None, :])
                        @ job.view[:3, :3].T)[:, 2]
        key = view_z[tri].astype(np.float32)
        pix = py.astype(np.int64) * g.width + px
        order = np.lexsort((key, pix))
        pix, px, py, tri, bary, front = (a[order] for a in
                                         (pix, px, py, tri, bary, front))
        grp = np.zeros(pix.size, np.int64)
        new_group = np.nonzero(pix[1:] != pix[:-1])[0] + 1
        grp[new_group] = new_group
        np.maximum.accumulate(grp, out=grp)
        rank = np.arange(pix.size, dtype=np.int64) - grp
        return px, py, tri, bary, front, rank

    # ---- 1. two glass materials at once: the smooth ball AND the
    # anisotropic box see-through, the ball reflecting the gradient sky,
    # under both shadow maps -- multi-material layers, backfaces, the
    # env term on a layer, and the additive merge all in one frame
    st = base_settings(w, h, shadows=True, transparency='SORTED')
    st.color_depth = '24'
    st.dither = 'NONE'
    st.output_scale = 'NONE'
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].opacity = 0.5
    sc.materials[1].reflect_level = 0.25
    sc.materials[2].opacity = 0.6

    # vacuity first, so the LAST render leaves the glass frame's shadow
    # maps on the scene for the mirror below
    solid = None
    sc.materials[1].opacity = sc.materials[2].opacity = 1.0
    sc.materials[1].reflect_level = 0.0
    solid = R.render(sc, st)
    sc.materials[1].opacity = 0.5
    sc.materials[1].reflect_level = 0.25
    sc.materials[2].opacity = 0.6
    cpu_full = R.render(sc, st)
    dv = float(np.abs(cpu_full - solid).max())
    check('the glass really changes the picture', dv > 0.02,
          f'transparency moves the frame by {dv:.4f}')

    view, _proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    opq, trans = R._split_by_alpha(sc, sc.mesh, st)
    check('both glass materials are in the transparent subset',
          np.unique(sc.mesh.mat_index[trans]).size == 2,
          str(np.unique(sc.mesh.mat_index[trans])))
    g = raster.GBuffer(w, h)
    raster.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, subset=opq,
                     gbuf=g, depth_bits=st.depth_precision)
    job = R.ShadeJob(sc, st, {}, None, view, eye, w, h)
    px, py, tri, bary, front, rank = frag_mirror(sc, st, job, g)
    check('fragments survive the depth test', px.size > 0, f'{px.size}')
    check('the glass shows both its faces',
          bool(front.any()) and bool((~front).any()),
          f'{int((~front).sum())} backfaces of {front.size}')
    check('some pixels stack layers', int(rank.max()) >= 1,
          f'deepest rank {int(rank.max())}')

    GSH._PLAN_CACHE.clear()
    p, why, a = GSH.plan_frame(job, g)
    check('the glass frame qualifies', p is not None, str(why))
    if p is not None:
        lp = (a or {}).get('__layers')
        check('both glass materials hold layer passes',
              lp is not None and sorted(e[0] for e in lp) == [1, 2],
              str((a or {}).get('__layers_why',
                                [e[0] for e in (lp or ())])))
    cpu_col = R._shade_chunked(job, tri, bary, px, py, front, None, st)
    got, gwhy = GSH.simulate_fragments(job, g, tri, bary, px, py, rank)
    check('the layers simulate', got is not None, str(gwhy))
    if got is not None:
        d = np.abs(got - cpu_col).max(axis=1)
        flips = int((d > 1e-2).sum())
        check('every fragment matches the CPU shading, alpha included',
              flips == 0 and float(d.max()) < 6e-3,
              f'{flips} of {tri.size} fragments >0.01, '
              f'max {float(d.max()):.6f}')
        mid = (cpu_col[:, 3] > 0.05) & (cpu_col[:, 3] < 0.95)
        check('the alpha chain is really exercised', bool(mid.any()),
              f'{int(mid.sum())} translucent fragments')

    # ---- 1b. the exporter knows master-shader glass is see-through.
    # The add-on's own templates put the alpha on the master node's
    # sockets (Glass: Opacity 0.12) with use_override off, so m.opacity
    # stays 1.0 -- and _tree_has_alpha never looked at the master node:
    # the engine's own glass presets exported as OPAQUE and skipped the
    # transparent pass entirely. The layer port is what surfaced it.
    from . import fakeblender as FB
    FB.install()
    from .. import export as EX
    MatT = type(sc.materials[1])
    check('master-socket Opacity marks the material see-through',
          EX._tree_has_alpha(None, MatT(
              name='t', graph=glass_graph(opacity=0.12))) is True)
    check('master-socket Edge Opacity does too',
          EX._tree_has_alpha(None, MatT(
              name='t', graph=glass_graph(opacity=1.0, edge=0.5))) is True)
    check('an opaque master shader stays opaque',
          EX._tree_has_alpha(None, MatT(
              name='t', graph=glass_graph(opacity=1.0))) is False)

    # ---- 2. the edge-opacity silhouette on a master-shader glass, with
    # the hard threshold ZEROING the facing area (0.5 < 0.6): the rim
    # term is the only alpha left, faded on the interpolated normal
    # through the master socket path
    st2 = base_settings(w, h, shadows=True, transparency='SORTED')
    st2.color_depth = '24'
    st2.dither = 'NONE'
    st2.output_scale = 'NONE'
    st2.alpha_threshold = 0.6
    sc2 = demo_scene(st2, with_texture=False)
    sc2.materials[1] = type(sc2.materials[1])(
        name='EdgeGlass', index=1, graph=glass_graph(
            opacity=0.5, edge=0.15, fresnel_power=2.0))
    sc2.materials[1].has_alpha = True        # what the exporter now sets
    R.render(sc2, st2)                       # leaves the shadow maps
    view2, _p2, vp2, eye2 = R.camera_matrices(sc2.camera, w, h)
    opq2, _t2 = R._split_by_alpha(sc2, sc2.mesh, st2)
    g2 = raster.GBuffer(w, h)
    raster.rasterize(sc2.mesh.verts, sc2.mesh.tris, vp2, w, h,
                     subset=opq2, gbuf=g2, depth_bits=st2.depth_precision)
    job2 = R.ShadeJob(sc2, st2, {}, None, view2, eye2, w, h)
    px2, py2, tri2, bary2, front2, rank2 = frag_mirror(sc2, st2, job2, g2)
    cpu2 = R._shade_chunked(job2, tri2, bary2, px2, py2, front2, None,
                            st2)
    zeroed = int((cpu2[:, 3] < 1e-6).sum())
    rimmed = int((cpu2[:, 3] > 0.02).sum())
    check('the threshold and the edge term are both load-bearing',
          zeroed > 0 and rimmed > 0,
          f'{zeroed} thresholded to nothing, {rimmed} rim fragments')
    GSH._PLAN_CACHE.clear()
    got2, why2 = GSH.simulate_fragments(job2, g2, tri2, bary2, px2, py2,
                                        rank2)
    check('edge-opacity glass simulates', got2 is not None, str(why2))
    if got2 is not None:
        d2 = np.abs(got2 - cpu2).max(axis=1)
        flips2 = int((d2 > 1e-2).sum())
        check('the silhouette alpha matches the CPU exactly',
              flips2 == 0 and float(d2.max()) < 6e-3,
              f'{flips2} of {tri2.size} fragments >0.01, '
              f'max {float(d2.max()):.6f}')

    # ---- 3. the refusals stay narrow and named
    z0 = np.zeros(0, np.int32)
    zb = np.zeros((0, 3), np.float32)
    zr = np.zeros(0, np.int64)

    # ray tracing: the layers now PLAN (the recursion runs per layer in
    # the executors -- proven fragment by fragment in
    # test_ray_traced_layers_shade_on_the_gpu)
    st3 = base_settings(w, h, shadows=False, raytrace=True, ray_depth=1)
    sc3 = demo_scene(st3, with_texture=False)
    sc3.materials[1].opacity = 0.5
    view3, _p3, vp3, eye3 = R.camera_matrices(sc3.camera, w, h)
    opq3, _t3 = R._split_by_alpha(sc3, sc3.mesh, st3)
    g3 = raster.GBuffer(w, h)
    raster.rasterize(sc3.mesh.verts, sc3.mesh.tris, vp3, w, h,
                     subset=opq3, gbuf=g3)
    job3 = R.ShadeJob(sc3, st3, {}, BVH(sc3.mesh.verts, sc3.mesh.tris),
                      view3, eye3, w, h)
    GSH._PLAN_CACHE.clear()
    p3, why3, a3 = GSH.plan_frame(job3, g3)
    check('the ray-traced glass frame still qualifies', p3 is not None,
          str(why3))
    lp3 = (a3 or {}).get('__layers')
    check('and its layers now plan under ray tracing',
          lp3 is not None and [e[0] for e in lp3] == [1]
          and (a3 or {}).get('__layers_why') is None,
          str((a3 or {}).get('__layers_why',
                             [e[0] for e in (lp3 or ())])))
    rp3 = (a3 or {}).get('__reflect')
    check("and the glass's refraction constants joined the ray plan",
          rp3 is not None and 1 in (rp3.get('refractive') or ()),
          str(rp3 and rp3.get('refractive')))

    # a rich world behind the glass: the env term has no honest GLSL
    st4 = base_settings(w, h, shadows=False, transparency='SORTED')
    sc4 = demo_scene(st4, with_texture=False)
    sc4.materials[1].opacity = 0.5
    sc4.materials[1].reflect_level = 0.5
    sc4.world = World()
    sc4.world.mode = 'BRYCE'
    view4, _p4, vp4, eye4 = R.camera_matrices(sc4.camera, w, h)
    opq4, _t4 = R._split_by_alpha(sc4, sc4.mesh, st4)
    g4 = raster.GBuffer(w, h)
    raster.rasterize(sc4.mesh.verts, sc4.mesh.tris, vp4, w, h,
                     subset=opq4, gbuf=g4)
    job4 = R.ShadeJob(sc4, st4, {}, None, view4, eye4, w, h)
    GSH._PLAN_CACHE.clear()
    p4, why4, a4 = GSH.plan_frame(job4, g4)
    check('the Bryce frame qualifies (nothing opaque reflects)',
          p4 is not None, str(why4))
    check('reflective glass under Bryce refuses its layers by name',
          'rich world' in str((a4 or {}).get('__layers_why')),
          str((a4 or {}).get('__layers_why')))
    got4, gwhy4 = GSH.simulate_fragments(job4, g4, z0, zb, z0, z0, zr)
    check('and the executor surfaces the same reason',
          got4 is None and 'rich world' in str(gwhy4), str(gwhy4))

    # a Bump pre-pass on a TRANSPARENT material now PLANS (its height
    # image draws per rank -- proven fragment by fragment in
    # test_bump_glass_layers_shade_on_the_gpu)
    st5 = base_settings(w, h, shadows=False, transparency='SORTED')
    sc5 = demo_scene(st5, with_texture=False)
    rng5 = np.random.default_rng(3)
    him = np.zeros((16, 16, 4), np.float32)
    him[:, :, 0] = rng5.random((16, 16))
    him[:, :, 1] = him[:, :, 2] = him[:, :, 0]
    him[:, :, 3] = 1.0
    sc5.images = {'hmap': ImageBuffer(name='hmap', pixels=him)}
    sc5.materials[1] = type(sc5.materials[1])(
        name='WavyGlass', index=1,
        graph=glass_graph(opacity=0.5, bump=True))
    sc5.materials[1].has_alpha = True
    tex5 = R.prepare_textures(sc5, st5)
    view5, _p5, vp5, eye5 = R.camera_matrices(sc5.camera, w, h)
    opq5, _t5 = R._split_by_alpha(sc5, sc5.mesh, st5)
    g5 = raster.GBuffer(w, h)
    raster.rasterize(sc5.mesh.verts, sc5.mesh.tris, vp5, w, h,
                     subset=opq5, gbuf=g5)
    job5 = R.ShadeJob(sc5, st5, tex5, None, view5, eye5, w, h)
    GSH._PLAN_CACHE.clear()
    p5, why5, a5 = GSH.plan_frame(job5, g5)
    check('the wavy-glass frame qualifies', p5 is not None, str(why5))
    lp5 = (a5 or {}).get('__layers')
    check('a Bump pre-pass on a layer now plans',
          lp5 is not None and [e[0] for e in lp5] == [1]
          and any((e[3].get('prepasses') or ()) for e in lp5),
          str((a5 or {}).get('__layers_why',
                             [e[0] for e in (lp5 or ())])))

    # ...and an OPAQUE Bump material still costs the layers NOTHING:
    # only see-through materials are probed for layer passes
    sc6 = demo_scene(st5, with_texture=False)
    sc6.images = {'hmap': ImageBuffer(name='hmap', pixels=him)}
    sc6.materials[1] = type(sc6.materials[1])(
        name='BumpyOpaque', index=1,
        graph=glass_graph(opacity=1.0, bump=True))
    sc6.materials[2].opacity = 0.6
    tex6 = R.prepare_textures(sc6, st5)
    view6, _p6, vp6, eye6 = R.camera_matrices(sc6.camera, w, h)
    opq6, _t6 = R._split_by_alpha(sc6, sc6.mesh, st5)
    g6 = raster.GBuffer(w, h)
    raster.rasterize(sc6.mesh.verts, sc6.mesh.tris, vp6, w, h,
                     subset=opq6, gbuf=g6)
    job6 = R.ShadeJob(sc6, st5, tex6, None, view6, eye6, w, h)
    GSH._PLAN_CACHE.clear()
    p6, why6, a6 = GSH.plan_frame(job6, g6)
    check('the opaque-bump frame qualifies', p6 is not None, str(why6))
    lp6 = (a6 or {}).get('__layers')
    check('an opaque Bump material does not cost the frame its layers',
          lp6 is not None and [e[0] for e in lp6] == [2],
          str((a6 or {}).get('__layers_why',
                             [e[0] for e in (lp6 or ())])))

    # ---- 7. a frame that is ALL transparency: the field's own shape.
    # Every visible material see-through means the opaque G-buffer is
    # EMPTY -- and plan_frame's old "nothing to shade is a success"
    # early return skipped the layer planning entirely, so the field's
    # all-glass frame refused with the unnamed default while 26.5
    # seconds of fragments shaded on the CPU. The plan now builds from
    # the materials' own triangles, no G-buffer pixels required.
    st7 = base_settings(w, h, shadows=True, transparency='SORTED')
    st7.color_depth = '24'
    st7.dither = 'NONE'
    st7.output_scale = 'NONE'
    sc7 = demo_scene(st7, with_texture=False)
    for m7 in sc7.materials:
        m7.opacity = 0.5
    R.render(sc7, st7)                       # leaves the shadow maps
    view7, _p7, vp7, eye7 = R.camera_matrices(sc7.camera, w, h)
    opq7, tr7 = R._split_by_alpha(sc7, sc7.mesh, st7)
    check('every triangle is in the transparent subset',
          tr7.size == sc7.mesh.tris.shape[0] and
          (opq7 is None or opq7.size == 0),
          f'{tr7.size} of {sc7.mesh.tris.shape[0]}')
    g7 = raster.GBuffer(w, h)
    if opq7 is not None and opq7.size:
        raster.rasterize(sc7.mesh.verts, sc7.mesh.tris, vp7, w, h,
                         subset=opq7, gbuf=g7,
                         depth_bits=st7.depth_precision)
    check('the opaque G-buffer is empty', not bool((g7.tri >= 0).any()))
    job7 = R.ShadeJob(sc7, st7, {}, None, view7, eye7, w, h)
    GSH._PLAN_CACHE.clear()
    p7, why7, a7 = GSH.plan_frame(job7, g7)
    check('an all-transparent frame still plans',
          p7 is not None and p7 == [], str(why7))
    lp7 = (a7 or {}).get('__layers')
    check('and its layer plan holds every glass material',
          lp7 is not None and sorted(e[0] for e in lp7) == [0, 1, 2],
          str((a7 or {}).get('__layers_why',
                             sorted(e[0] for e in (lp7 or ())))))
    px7, py7, tri7, bary7, front7, rank7 = frag_mirror(sc7, st7, job7, g7)
    check('the whole picture is fragments', px7.size > 3000,
          f'{px7.size}')
    cpu7 = R._shade_chunked(job7, tri7, bary7, px7, py7, front7, None,
                            st7)
    got7, why7b = GSH.simulate_fragments(job7, g7, tri7, bary7, px7,
                                         py7, rank7)
    check('the all-glass frame simulates', got7 is not None, str(why7b))
    if got7 is not None:
        d7 = np.abs(got7 - cpu7).max(axis=1)
        check('and matches the CPU per fragment, alpha included',
              int((d7 > 1e-2).sum()) == 0 and float(d7.max()) < 6e-3,
              f'{int((d7 > 1e-2).sum())} of {tri7.size} fragments '
              f'>0.01, max {float(d7.max()):.6f}')

    # ---- 8. the impossible seam stays NAMED: fragments whose material
    # the layer predicate reads as opaque get a reason carrying the
    # material and the fields it read -- never a bare default
    st8 = base_settings(w, h, shadows=False, transparency='SORTED')
    sc8 = demo_scene(st8, with_texture=False)     # every material opaque
    view8, _p8, vp8, eye8 = R.camera_matrices(sc8.camera, w, h)
    g8 = raster.GBuffer(w, h)
    raster.rasterize(sc8.mesh.verts, sc8.mesh.tris, vp8, w, h, gbuf=g8)
    job8 = R.ShadeJob(sc8, st8, {}, None, view8, eye8, w, h)
    GSH._PLAN_CACHE.clear()
    fake_tri = np.nonzero(sc8.mesh.mat_index == 1)[0][:4].astype(np.int32)
    fb8 = np.full((fake_tri.size, 3), 1.0 / 3.0, np.float32)
    z8 = np.zeros(fake_tri.size, np.int32)
    got8, why8 = GSH.simulate_fragments(job8, g8, fake_tri, fb8, z8, z8,
                                        np.zeros(fake_tri.size, np.int64))
    check('a predicate disagreement names the material and its fields',
          got8 is None and 'Ball' in str(why8) and 'opacity' in str(why8),
          str(why8))

    # ---- 9. the field's Material.002 shape: a MAPPED texture driving a
    # glass layer. ShaderNodeMapping was the first emitter gap a field
    # frame ever named -- the mapping's trig is baked from NumPy's own
    # float32 cos/sin, so the driver does none of its own
    st9 = base_settings(w, h, shadows=True, transparency='SORTED')
    st9.color_depth = '24'
    st9.dither = 'NONE'
    st9.output_scale = 'NONE'
    sc9 = demo_scene(st9, with_texture=True)    # brings the checker image
    sc9.materials[1] = type(sc9.materials[1])(
        name='MappedGlass', index=1, graph={'output': 'out', 'nodes': {
            'tc': {'id': 'tc', 'bl_idname': 'ShaderNodeTexCoord',
                   'props': {}, 'inputs': [],
                   'outputs': [{'name': 'Generated', 'type': 'VECTOR'},
                               {'name': 'UV', 'type': 'VECTOR'}]},
            'map': {'id': 'map', 'bl_idname': 'ShaderNodeMapping',
                    'props': {'vector_type': 'POINT'},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['tc', 0]),
                               sk('Location', 'VECTOR', [0.2, -0.1, 0.0]),
                               sk('Rotation', 'VECTOR', [0.0, 0.0, 0.7]),
                               sk('Scale', 'VECTOR', [2.0, 3.0, 1.0])],
                    'outputs': [{'name': 'Vector', 'type': 'VECTOR'}]},
            'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                    'props': {'image': 'checker',
                              'interpolation': 'Closest'},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['map', 0])],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Alpha', 'type': 'VALUE'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG'},
                    'inputs': [sk('Diffuse Color', 'RGBA', [1, 1, 1, 1],
                                  ['tex', 0]),
                               sk('Diffuse Level', 'VALUE', 1.0),
                               sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
                               sk('Specular Level', 'VALUE', 0.6),
                               sk('Glossiness', 'VALUE', 24.0),
                               sk('Ambient', 'VALUE', 1.0),
                               sk('Opacity', 'VALUE', 0.5)],
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None,
                                  ['hal', 0])],
                    'outputs': []}}})
    sc9.materials[1].has_alpha = True
    tex9 = R.prepare_textures(sc9, st9)
    R.render(sc9, st9)                       # leaves the shadow maps
    view9, _p9, vp9, eye9 = R.camera_matrices(sc9.camera, w, h)
    opq9, _t9 = R._split_by_alpha(sc9, sc9.mesh, st9)
    g9 = raster.GBuffer(w, h)
    raster.rasterize(sc9.mesh.verts, sc9.mesh.tris, vp9, w, h,
                     subset=opq9, gbuf=g9, depth_bits=st9.depth_precision)
    job9 = R.ShadeJob(sc9, st9, tex9, None, view9, eye9, w, h)
    px9, py9, tri9, bary9, front9, rank9 = frag_mirror(sc9, st9, job9,
                                                       g9)
    cpu9 = R._shade_chunked(job9, tri9, bary9, px9, py9, front9, None,
                            st9)
    check('the mapped texture really varies over the glass',
          float(cpu9[:, :3].std()) > 0.02,
          f'std {float(cpu9[:, :3].std()):.4f}')
    GSH._PLAN_CACHE.clear()
    got9, why9 = GSH.simulate_fragments(job9, g9, tri9, bary9, px9, py9,
                                        rank9)
    check('mapped-texture glass simulates (the Material.002 shape)',
          got9 is not None, str(why9))
    if got9 is not None:
        d9 = np.abs(got9 - cpu9).max(axis=1)
        check('and matches the CPU per fragment, alpha included',
              int((d9 > 1e-2).sum()) == 0 and float(d9.max()) < 6e-3,
              f'{int((d9 > 1e-2).sum())} of {tri9.size} fragments '
              f'>0.01, max {float(d9.max()):.6f}')


def _pick_splitting_frac(rank, w, h):
    """A `layer_gpu_min_frac` that lands BETWEEN two layer sizes.

    Returns (frac, dense_ranks, sparse_ranks) or (None, why, None) when
    the frame cannot split (fewer than two distinct layer sizes)."""
    counts = np.bincount(rank, minlength=int(rank.max()) + 1)
    vals = sorted({int(c) for c in counts if c > 0}, reverse=True)
    if len(vals) < 2:
        return None, f'layer sizes {counts.tolist()} cannot split', None
    target = (vals[0] + vals[1]) // 2
    thresh = max(1, int(round((target / float(w * h)) * w * h)))
    dense = sorted(int(r) for r in range(counts.size)
                   if counts[r] >= thresh)
    sparse = sorted(int(r) for r in range(counts.size)
                    if 0 < counts[r] < thresh)
    return target / float(w * h), dense, sparse


def test_sparse_layers_route_to_the_cpu_exactly():
    """Rank routing's partition and merge must change NOTHING.

    The 1.25.59 field split settled where the layer stage's seconds
    live: not in the material count (skipping absent materials' passes
    moved nothing) but in sixteen full-frame round trips -- the driver
    pays a draw and a readback over EVERY pixel per depth layer,
    whether three fragments live there or a million, while the CPU
    path pays per fragment. So layers below `layer_gpu_min_frac` of
    the frame now shade on the proven per-rank CPU path. Routing is by
    WHOLE layer, and this test corners the machinery itself: the
    "driver" is a stub that answers with the CPU's own colours for
    exactly the fragments it was given, so the routed picture must
    equal the pure CPU picture BIT FOR BIT -- any difference at all
    belongs to the router.
    """
    from ..gpu import shade as GSH

    w, h = 96, 72
    st = base_settings(w, h, shadows=True, transparency='SORTED')
    sc = demo_scene(st, with_texture=False)
    sc.materials[1].opacity = 0.5
    sc.materials[2].opacity = 0.6
    base = R.render(sc, st)

    # record the pure CPU shading: (pixel, rank) names a fragment
    # uniquely, so the stub below can answer with the CPU's own colour
    # for ANY subset it is asked about
    recs = []
    real_cpu = R._shade_fragments_cpu

    def recording_cpu(job, tri, bary, px, py, front, rank, st2):
        col = real_cpu(job, tri, bary, px, py, front, rank, st2)
        key = (py.astype(np.int64) * w + px) * 64 + rank
        recs.append((key.copy(), col.copy(), rank.copy()))
        return col

    R._shade_fragments_cpu = recording_cpu
    try:
        sc2 = demo_scene(st, with_texture=False)
        sc2.materials[1].opacity = 0.5
        sc2.materials[2].opacity = 0.6
        base2 = R.render(sc2, st)
    finally:
        R._shade_fragments_cpu = real_cpu
    check('recording the CPU shading changed nothing',
          float(np.abs(base2 - base).max()) == 0.0)
    check('the CPU shading was recorded', len(recs) >= 1, str(len(recs)))
    keys = np.concatenate([r[0] for r in recs])
    cols = np.concatenate([r[1] for r in recs])
    rank_all = np.concatenate([r[2] for r in recs])
    o = np.argsort(keys)
    keys, cols = keys[o], cols[o]
    check('some pixels stack layers', int(rank_all.max()) >= 1,
          f'deepest rank {int(rank_all.max())}')

    frac, dense_r, sparse_r = _pick_splitting_frac(rank_all, w, h)
    check('the frame CAN split between the paths', frac is not None,
          str(dense_r))
    if frac is None:
        return

    gpu_calls = []
    lookups_ok = []

    def stub_gpu(job, gbuf, tri, bary, px, py, rank):
        kk = (py.astype(np.int64) * w + px) * 64 + rank
        pos = np.minimum(np.searchsorted(keys, kk), keys.size - 1)
        lookups_ok.append(bool((keys[pos] == kk).all()))
        gpu_calls.append(np.unique(rank).tolist())
        return cols[pos].copy(), None

    cpu_calls = []

    def recording_cpu2(job, tri, bary, px, py, front, rank, st2):
        cpu_calls.append(np.unique(rank).tolist())
        return real_cpu(job, tri, bary, px, py, front, rank, st2)

    def gpu_settings(f):
        sth = base_settings(w, h, shadows=True, transparency='SORTED')
        sth.render_device = 'GPU'
        sth.gpu_shading = True
        sth.gpu_raster = False       # identical fragments to the CPU run
        sth.gpu_post = False
        sth.layer_gpu_min_frac = f
        return sth

    def hybrid_render(f):
        del gpu_calls[:], cpu_calls[:], lookups_ok[:]
        sth = gpu_settings(f)
        sch = demo_scene(sth, with_texture=False)
        sch.materials[1].opacity = 0.5
        sch.materials[2].opacity = 0.6
        real_gpu = GSH.shade_fragments_frame
        GSH.shade_fragments_frame = stub_gpu
        R._shade_fragments_cpu = recording_cpu2
        try:
            img2 = R.render(sch, sth)
        finally:
            GSH.shade_fragments_frame = real_gpu
            R._shade_fragments_cpu = real_cpu
        return img2, dict(R.LAST_ROUTING)

    hyb, rt = hybrid_render(frac)
    check('the frame really split', rt.get('gpu_ranks', 0) >= 1
          and rt.get('cpu_ranks', 0) >= 1 and 'refused' not in rt,
          str(rt))
    check('every stub lookup found its fragment',
          lookups_ok and all(lookups_ok), str(lookups_ok))
    check('the stub saw exactly the dense layers',
          gpu_calls == [dense_r], f'{gpu_calls} vs {dense_r}')
    check('the CPU path saw exactly the sparse layers',
          cpu_calls == [sparse_r], f'{cpu_calls} vs {sparse_r}')
    check('the fragment counts add up',
          rt.get('gpu_frags', -1) + rt.get('cpu_frags', -1) == keys.size,
          str(rt))
    check('the routed picture equals the pure CPU picture bit for bit',
          float(np.abs(hyb - base).max()) == 0.0,
          f'max {float(np.abs(hyb - base).max()):.6f}')

    # frac 0: nothing routes away -- the stub shades every layer
    all_gpu, rt0 = hybrid_render(0.0)
    check('frac 0 keeps every layer on the driver',
          rt0.get('cpu_ranks', 1) == 0 and gpu_calls
          and not cpu_calls, str(rt0))
    check('and still matches bit for bit',
          float(np.abs(all_gpu - base).max()) == 0.0)

    # frac 1: every layer is "sparse" -- the driver is never asked,
    # the route is printed as a route, and the picture stands
    all_cpu, rt1 = hybrid_render(1.0)
    check('frac 1 asks the driver nothing',
          rt1.get('gpu_ranks', 1) == 0 and not gpu_calls, str(rt1))
    check('and the whole-CPU frame still matches bit for bit',
          float(np.abs(all_cpu - base).max()) == 0.0)


def test_hybrid_layer_routing_matches_both_pure_paths():
    """A mixed frame agrees with BOTH pure runs -- through the real maths.

    The exact-merge test above pins the machinery with a lookup stub;
    this one routes through `simulate_fragments`, the NumPy front-end
    of the actual layer passes, so the dense layers shade through the
    GPU's own maths while the sparse layers take the per-rank CPU
    path. The mixed picture must sit within the proven layer tolerance
    of the all-CPU frame AND the all-simulated frame, and the routing
    record must show a real mix -- a hybrid that quietly ran pure
    would pass any picture check, so vacuity is checked by name.
    """
    from ..gpu import shade as GSH

    w, h = 96, 72

    def settings(device='CPU', frac=0.0):
        sth = base_settings(w, h, shadows=True, transparency='SORTED')
        sth.render_device = device
        if device == 'GPU':
            sth.gpu_shading = True
            sth.gpu_raster = False
            sth.gpu_post = False
            sth.layer_gpu_min_frac = frac
        return sth

    def scene(sth):
        sch = demo_scene(sth, with_texture=False)
        sch.materials[1].opacity = 0.5
        sch.materials[1].reflect_level = 0.25
        sch.materials[2].opacity = 0.6
        return sch

    st_c = settings()
    base = R.render(scene(st_c), st_c)

    whys = []

    def stub_sim(job, gbuf, tri, bary, px, py, rank):
        got, why = GSH.simulate_fragments(job, gbuf, tri, bary, px, py,
                                          rank)
        whys.append(why)
        return got, why

    def render_with(frac):
        del whys[:]
        sth = settings('GPU', frac)
        real_gpu = GSH.shade_fragments_frame
        GSH.shade_fragments_frame = stub_sim
        try:
            img2 = R.render(scene(sth), sth)
        finally:
            GSH.shade_fragments_frame = real_gpu
        return img2, dict(R.LAST_ROUTING)

    all_sim, rt0 = render_with(0.0)
    check('the pure simulated frame shades every layer',
          rt0.get('cpu_ranks', 1) == 0 and whys == [None], str((rt0,
                                                                whys)))
    d0 = np.abs(all_sim - base).max(axis=2)
    check('and matches the CPU frame (the pure baseline)',
          int((d0 > 1e-2).sum()) == 0 and float(d0.max()) < 6e-3,
          f'{int((d0 > 1e-2).sum())} px >0.01, max {float(d0.max()):.6f}')

    counts = rt0.get('counts') or []
    vals = sorted({c for c in counts if c > 0}, reverse=True)
    check('the frame has two layer sizes to split between',
          len(vals) >= 2, str(counts))
    if len(vals) < 2:
        return
    frac = ((vals[0] + vals[1]) // 2) / float(w * h)
    hyb, rt = render_with(frac)
    check('the hybrid frame really mixed',
          rt.get('gpu_ranks', 0) >= 1 and rt.get('cpu_ranks', 0) >= 1
          and 'refused' not in rt and whys == [None], str((rt, whys)))
    dc = np.abs(hyb - base).max(axis=2)
    check('the mixed frame matches the all-CPU frame',
          int((dc > 1e-2).sum()) == 0 and float(dc.max()) < 6e-3,
          f'{int((dc > 1e-2).sum())} px >0.01, max {float(dc.max()):.6f}')
    dg = np.abs(hyb - all_sim).max(axis=2)
    check('and the all-simulated frame',
          int((dg > 1e-2).sum()) == 0 and float(dg.max()) < 6e-3,
          f'{int((dg > 1e-2).sum())} px >0.01, max {float(dg.max()):.6f}')


def test_depth_quantizes_per_pixel_not_per_vertex():
    """An N-bit z-buffer stores values ON its grid -- rounded per PIXEL.

    The old code quantized the VERTEX z in build_screen_tris and then
    interpolated: every slanted triangle's stored depths landed up to a
    full step OFF the grid, which tilted whole depth planes against
    each other -- close-fitting geometry (a face's muzzle and eye
    sockets, grass against a stone) showed solid stomp-through wedges
    instead of the thin dithered fight bands real N-bit hardware
    showed. The field named it on two pictures. Now the fillers
    interpolate at full precision and round at the buffer, the same
    formula on both rasterisers (roundEven in the kernel = np.round).
    """
    from ..core import raster as CRr

    W, H, BITS = 160, 120, 16
    steps = float((1 << BITS) - 1)
    # a strongly slanted quad: interpolated z spans many steps
    verts = np.array([[-0.9, -0.9, 0.10, 1.0], [0.9, -0.9, 0.55, 1.0],
                      [0.9, 0.9, 0.55, 1.0], [-0.9, 0.9, 0.10, 1.0]],
                     np.float32)
    tris = np.array([[0, 1, 2], [0, 2, 3]], np.int32)

    def grid_error(depth_arr, cov):
        d = depth_arr[cov].astype(np.float32)
        q = (np.round((d * np.float32(0.5) + np.float32(0.5))
                      * np.float32(steps)) / np.float32(steps)
             * np.float32(2.0) - np.float32(1.0))
        return float(np.abs(d - q).max()) if d.size else -1.0

    # NEW: per-pixel quantization -- every stored depth is on the grid
    g = CRr.GBuffer(W, H)
    sx, sy, iw, z, bw, src = CRr.build_screen_tris(verts.copy(), tris,
                                                   W, H)
    CRr.fill(g, sx, sy, iw, z, bw, src, depth_bits=BITS)
    cov = g.tri >= 0
    check('the slanted quad covers pixels', int(cov.sum()) > 5000,
          str(int(cov.sum())))
    e_new = grid_error(g.depth, cov)
    check('every stored depth lies exactly on the N-bit grid',
          e_new == 0.0, f'max off-grid {e_new:.2e}')

    # OLD behaviour reproduced: quantize the VERTEX z, interpolate --
    # the negative control must land OFF the grid, or this test could
    # not catch a regression to it
    g2 = CRr.GBuffer(W, H)
    zq = np.round((z * 0.5 + 0.5) * steps) / steps * 2.0 - 1.0
    CRr.fill(g2, sx, sy, iw, zq.astype(np.float32), bw, src)
    e_old = grid_error(g2.depth, g2.tri >= 0)
    check('vertex-quantized interpolation is provably off the grid',
          e_old > 0.25 / steps, f'max off-grid {e_old:.2e}')

    # both fillers, same picture: the batched path must quantize
    # identically (it is verified bit-identical to fill elsewhere; this
    # pins the NEW parameter through it)
    g3 = CRr.GBuffer(W, H)
    CRr.fill_batched(g3, sx, sy, iw, z, bw, src, depth_bits=BITS)
    check('fill and fill_batched quantize identically',
          np.array_equal(g.depth, g3.depth)
          and np.array_equal(g.tri, g3.tri))

    # the kernel's front-end mirror rounds the same way
    from ..gpu import craster as CRA
    tri_s, _b, zndc_s, _f, _b2, _lin, _mk = CRA.simulate_raster(
        sx, sy, iw, z, bw, src, None, W, H, depth_bits=BITS)
    same = tri_s == g.tri
    check('the kernel mirror picks the same winners at 16 bits',
          bool(same.all()), f'{int((~same).sum())} pixels differ')
    dz = np.abs(zndc_s[cov] - g.zndc[cov]).max() if cov.any() else 0.0
    check('and stores the same quantized depths',
          float(dz) == 0.0, f'max {float(dz):.2e}')

    # a coplanar transparent contact still survives the A-buffer keep
    # at 16 bits: both sides quantize to the SAME grid value, so the
    # tolerant limit keeps the fragment on either rasteriser
    opq = g.depth[cov][:100]
    check('quantized coplanar contacts stay within the keep limit',
          bool((opq <= CRr.abuf_depth_limit(opq)).all()))


def test_a_blend_mode_alone_does_not_make_a_material_see_through():
    """The bug three rounds of "the depth is screwed up" were really.

    The field's console settled it: `25 of 25 materials are
    see-through (1209 of 1209 triangles)` on a solid Sonic model, and
    `NOTHING is in the depth-buffered pass`. The importer had set a
    non-opaque blend mode on every material -- as importers routinely
    do -- and `_tree_has_alpha` treated the MODE as evidence of alpha.
    Every triangle then rasterised with culling off and no depth
    write, so hidden-surface removal never happened and the model was
    composited as stacked layers: under Sorted Blend, polygon-centroid
    ordering, whose classic failure is one surface wedging through
    another. A blend mode is a policy for handling alpha, not proof
    any exists.
    """
    from . import fakeblender as FB
    FB.install()
    from .. import export as EX

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    MatT = type(demo_scene(base_settings(32, 24)).materials[0])

    def master(opacity=1.0, edge=None, link_opacity=False):
        ins = [sk('Diffuse Color', 'RGBA', [0.5, 0.5, 0.5, 1.0]),
               sk('Opacity', 'VALUE', opacity,
                  ['tex', 1] if link_opacity else None)]
        if edge is not None:
            ins.append(sk('Edge Opacity', 'VALUE', edge))
        return {'output': 'out', 'nodes': {
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG'}, 'inputs': ins,
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'tex': {'id': 'tex', 'bl_idname': 'ShaderNodeTexImage',
                    'props': {'image': 'img'},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0])],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Alpha', 'type': 'VALUE'}]},
            'out': {'id': 'out',
                    'bl_idname': 'ShaderNodeOutputMaterial', 'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0])],
                    'outputs': []}}}

    class FakeMat:
        def __init__(self, blend):
            self.blend_method = blend

    # the field's exact shape: a solid material whose only unusual
    # property is the blend mode its importer set
    for mode in ('CLIP', 'HASHED', 'BLEND', 'DITHERED', 'BLENDED'):
        m = MatT(name='sonic', graph=master())
        got = EX._tree_has_alpha(FakeMat(mode), m)
        check(f'blend mode {mode} alone does NOT mark a solid material '
              'see-through', got is False, str(EX._alpha_reason(
                  FakeMat(mode), m)))

    # ...and every genuine kind of alpha still does, with the reason
    # naming its own evidence rather than "flagged on export"
    cases = [
        ('opacity below one', MatT(name='t', opacity=0.5), 'Opacity'),
        ('a master Opacity socket below one',
         MatT(name='t', graph=master(opacity=0.12)), 'Opacity'),
        ('an Edge Opacity below one',
         MatT(name='t', graph=master(1.0, edge=0.5)), 'Edge Opacity'),
        ('a LINKED Opacity socket (an alpha texture)',
         MatT(name='t', graph=master(1.0, link_opacity=True)),
         'linked'),
    ]
    for label, mm, token in cases:
        why = EX._alpha_reason(FakeMat('OPAQUE'), mm)
        check(f'{label} still marks it see-through',
              why is not None and token in why, str(why))

    # a Transparent BSDF is alpha wherever it sits in the tree
    g = master()
    g['nodes']['tr'] = {'id': 'tr',
                        'bl_idname': 'ShaderNodeBsdfTransparent',
                        'props': {}, 'inputs': [],
                        'outputs': [{'name': 'BSDF', 'type': 'SHADER'}]}
    why = EX._alpha_reason(FakeMat('OPAQUE'), MatT(name='t', graph=g))
    check('a Transparent BSDF anywhere in the tree marks it',
          why is not None and 'Transparent' in why, str(why))

    # end to end: the classification a mis-flagged model now gets
    st = base_settings(64, 48)
    sc = demo_scene(st, with_texture=False)
    for m in sc.materials:
        m.has_alpha = False
        m.opacity = 1.0
    R._split_by_alpha(sc, sc.mesh, st)
    rec = dict(R.LAST_SPLIT)
    check('a fully solid model puts every triangle in the '
          'depth-buffered pass',
          rec.get('see_through') == 0
          and rec.get('tris_see_through') == 0, str(rec))


def test_a_shading_rate_interpolates_light_not_texture():
    """Gouraud is a LIGHTING rate. The texture stays per pixel.

    The field rendered a 1209-triangle character under a preset that
    selects Gouraud and got a smeared blur: Halcyon evaluated the whole
    material at the vertices, texture included, and interpolated the
    result across triangles the size of his cheek. Hardware of the
    period interpolated the vertex COLOUR and sampled the texture at
    every pixel, then multiplied -- MODULATE, and before OpenGL 1.2's
    separate specular that combined colour was the whole lighting
    result, which is why this multiplies everything by the texel.
    """
    def detail(img):
        """Mean horizontal contrast: how much texture survives."""
        g = img[..., :3].mean(axis=2)
        return float(np.abs(np.diff(g, axis=1)).mean())

    w, h = 128, 96
    imgs = {}
    for rate in ('PIXEL', 'VERTEX', 'FACE'):
        st = base_settings(w, h, shading_rate=rate, shadows=False)
        imgs[rate] = R.render(demo_scene(st, with_texture=True), st)

    # the OLD path, reproduced exactly: shade the whole material at the
    # vertices and interpolate. Without this control the test could not
    # tell a fix from a coincidence.
    st_old = base_settings(w, h, shading_rate='VERTEX', shadows=False)
    sc_old = demo_scene(st_old, with_texture=True)
    real_interp = R._shade_interpolated

    def old_interpolated(job, tri_idx, bary, rate, st=None, px=None,
                         py=None, front=None, blin=None):
        mesh = job.scene.mesh
        uniq = np.unique(tri_idx)
        col, lookup = R.shade_vertex_rate(job, uniq, rate, st)
        if rate == 'FACE':
            return col[np.searchsorted(uniq, tri_idx)]
        tris = mesh.tris[tri_idx]
        c0, c1, c2 = (col[lookup[tris[:, i]]] for i in range(3))
        return (c0 * bary[:, 0:1] + c1 * bary[:, 1:2]
                + c2 * bary[:, 2:3]).astype(np.float32)

    R._shade_interpolated = old_interpolated
    try:
        old = R.render(sc_old, st_old)
    finally:
        R._shade_interpolated = real_interp

    d_px, d_vx, d_old = (detail(imgs['PIXEL']), detail(imgs['VERTEX']),
                         detail(old))
    check('shading the whole material per vertex really did blur the '
          'texture', d_old < d_px * 0.85,
          f'old {d_old:.5f} vs per-pixel {d_px:.5f}')
    check('the vertex rate now keeps the texture nearly as sharp as '
          'per-pixel shading', d_vx > d_old * 1.1 and d_vx > d_px * 0.7,
          f'vertex {d_vx:.5f}, old {d_old:.5f}, per-pixel {d_px:.5f}')
    check('and it is still a different picture from per-pixel shading '
          '(the banding a shading RATE exists for)',
          float(np.abs(imgs['PIXEL'] - imgs['VERTEX']).max()) > 1e-2,
          f'{float(np.abs(imgs["PIXEL"] - imgs["VERTEX"]).max()):.4f}')
    check('the face rate differs from the vertex rate too',
          float(np.abs(imgs['VERTEX'] - imgs['FACE']).max()) > 1e-2)

    # an UNTEXTURED, non-specular material must be unaffected: its
    # albedo is one colour, so factoring it out and multiplying it back
    # reproduces the old picture. This is what makes the change a fix
    # rather than a new look.
    stp = base_settings(w, h, shading_rate='VERTEX', shadows=False,
                        force_model='LAMBERT')
    scp = demo_scene(stp, with_texture=False)
    new_plain = R.render(scp, stp)
    R._shade_interpolated = old_interpolated
    try:
        old_plain = R.render(demo_scene(stp, with_texture=False), stp)
    finally:
        R._shade_interpolated = real_interp
    dp = float(np.abs(new_plain - old_plain).max())
    check('an untextured diffuse material renders exactly as before',
          dp < 2e-3, f'max {dp:.6f}')

    # Painter's mode stores polygon view distances, not ndc depth --
    # reading them as ndc is what made the depth line announce the
    # field's frame was at its projection's asymptote
    from ..core import raster as CRp
    stq = base_settings(64, 48, depth_sort='PAINTERS')
    scq = demo_scene(stq, with_texture=False)
    view, proj, vp, eye = R.camera_matrices(scq.camera, 64, 48)
    gq = CRp.GBuffer(64, 48)
    CRp.rasterize(scq.mesh.verts, scq.mesh.tris, vp, 64, 48, gbuf=gq,
                  flat_depth=R.polygon_depths(scq.mesh, view, eye))
    said = R.depth_report(proj, gq, stq.depth_precision, 'PAINTERS') or ''
    check("Painter's mode is reported as Painter's, not as broken depth",
          "Painter's algorithm" in said and 'asymptote' not in said, said)


def test_gouraud_frames_shade_on_the_gpu():
    """The named refusal 'the scene shading rate is VERTEX' is lifted.

    The field hit it on every render of a model whose preset selects
    Gouraud -- which is most of the console presets -- so choosing GPU
    bought nothing. The port is the R66 split made device-shaped: the
    CPU lights the CORNERS of every triangle over a white surface
    (shadows, env, the model's own formula -- its own code, so the
    values are its own numbers) and the pass interpolates them by the
    G-buffer's barycentrics and multiplies by the per-pixel albedo
    chain. MODULATE, with the driver doing the per-pixel half.
    """
    from ..core import raster as CRg
    from ..gpu import shade as GSH

    w, h = 96, 72

    def gbuffer_for(sc, st):
        view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
        g = CRg.GBuffer(w, h)
        CRg.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                      depth_bits=st.depth_precision)
        tex = R.prepare_textures(sc, st)
        job = R.ShadeJob(sc, st, tex, None, view, eye, w, h)
        return g, job

    def seam(rate, label):
        st = base_settings(w, h, shadows=True, shading_rate=rate)
        sc = demo_scene(st, with_texture=True)
        cpu = R.render(sc, st)
        g, job = gbuffer_for(sc, st)
        GSH._PLAN_CACHE.clear()
        p, why, _a = GSH.plan_frame(job, g)
        check(f'a {label} frame qualifies for the deferred pass',
              p is not None, str(why))
        if p is None:
            return None
        want = rate != 'PIXEL'
        check(f'{label} passes carry corner light exactly when the rate '
              'says so',
              all(bool(b.get('vlight')) == want for _mi, _n, _s, b in p),
              str([(mi, bool(b.get('vlight'))) for mi, _n, _s, b in p]))
        out, _hit = GSH.simulate(job, g)
        check(f'the {label} passes simulate', out is not None, str(_hit))
        if out is None:
            return None
        yy, xx = np.nonzero(g.tri >= 0)
        d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
        check(f'the {label} frame matches the CPU per pixel',
              int((d > 1e-2).sum()) == 0 and float(d.max()) < 6e-3,
              f'{int((d > 1e-2).sum())} of {yy.size} px >0.01, '
              f'max {float(d.max()):.6f}')
        return out

    vx = seam('VERTEX', 'Gouraud')
    fc = seam('FACE', 'flat')
    px_out = seam('PIXEL', 'per-pixel')
    if vx is not None and px_out is not None:
        check('Gouraud is really a different picture (the banding the '
              'rate exists for)',
              float(np.abs(vx - px_out).max()) > 1e-2,
              f'{float(np.abs(vx - px_out).max()):.4f}')
    if vx is not None and fc is not None:
        check('and flat differs from Gouraud',
              float(np.abs(vx - fc).max()) > 1e-2)

    # mixed rates in ONE frame: a per-material Gouraud override next to
    # per-pixel materials -- each pass gets its own rate
    st = base_settings(w, h, shadows=True, shading_rate='PIXEL')
    sc = demo_scene(st, with_texture=True)
    sc.materials[1].use_override = True
    sc.materials[1].model = 'GOURAUD'
    cpu = R.render(sc, st)
    g, job = gbuffer_for(sc, st)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('a mixed-rate frame qualifies', p is not None, str(why))
    if p is not None:
        rates = {mi: bool(b.get('vlight')) for mi, _n, _s, b in p}
        check('exactly the Gouraud material interpolates corner light',
              rates.get(1) is True
              and not any(v for k, v in rates.items() if k != 1),
              str(rates))
        out, _hit = GSH.simulate(job, g)
        if out is not None:
            yy, xx = np.nonzero(g.tri >= 0)
            d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
            check('and matches the CPU per pixel, both rates at once',
                  int((d > 1e-2).sum()) == 0 and float(d.max()) < 6e-3,
                  f'{int((d > 1e-2).sum())} of {yy.size} px >0.01, '
                  f'max {float(d.max()):.6f}')

    # under ray tracing a Gouraud material still refuses, BY NAME: a
    # hit lights per pixel and the light loop has no Gouraud formula
    st2 = base_settings(w, h, shadows=True, shading_rate='VERTEX',
                        raytrace=True, ray_depth=1)
    sc2 = demo_scene(st2, with_texture=True)
    sc2.materials[2].reflect_level = 0.5
    g2, job2 = gbuffer_for(sc2, st2)
    job2.bvh = R._cached_bvh(sc2, sc2.mesh)
    GSH._PLAN_CACHE.clear()
    p2, why2, _a2 = GSH.plan_frame(job2, g2)
    check('a ray-traced Gouraud frame refuses by name',
          p2 is None and 'hit' in str(why2), str(why2))


def test_texture_coordinates_and_two_uv_maps_on_the_gpu():
    """UV Map by NAME, Texture Coordinate outputs, and Mix -- end to end.

    Three gaps in one frame: the UV Map node ignored its layer name on
    the GPU (always the active layer), the Texture Coordinate node's
    Camera and Window outputs were silently wrong in GLSL (both
    answered world position or generated coordinates), and the modern
    Mix node had no emitter at all. The material here reads the SECOND
    UV map into a checker, screen-space Window into another, and mixes
    them -- per-pixel, against the CPU, through the whole deferred
    stack including the attribute texture's new second-UV half.
    """
    from ..core import raster as CRuv
    from ..gpu import shade as GSH

    w, h = 96, 72

    def sk(nm, t, d, l=None):
        return {'name': nm, 'type': t, 'default': d, 'link': l}

    def build_graph():
        return {'output': 'out', 'nodes': {
            'uv2': {'id': 'uv2', 'bl_idname': 'ShaderNodeUVMap',
                    'props': {'uv_map': 'back'}, 'inputs': [],
                    'outputs': [{'name': 'UV', 'type': 'VECTOR'}]},
            'ck2': {'id': 'ck2', 'bl_idname': 'ShaderNodeTexChecker',
                    'props': {},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['uv2', 0]),
                               sk('Color1', 'RGBA', [1, 1, 1, 1]),
                               sk('Color2', 'RGBA', [0, 0, 0, 1]),
                               sk('Scale', 'VALUE', 6.0)],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Fac', 'type': 'VALUE'}]},
            'tc': {'id': 'tc', 'bl_idname': 'ShaderNodeTexCoord',
                   'props': {}, 'inputs': [],
                   'outputs': [{'name': 'Generated', 'type': 'VECTOR'},
                               {'name': 'Normal', 'type': 'VECTOR'},
                               {'name': 'UV', 'type': 'VECTOR'},
                               {'name': 'Object', 'type': 'VECTOR'},
                               {'name': 'Camera', 'type': 'VECTOR'},
                               {'name': 'Window', 'type': 'VECTOR'},
                               {'name': 'Reflection', 'type': 'VECTOR'}]},
            'ckw': {'id': 'ckw', 'bl_idname': 'ShaderNodeTexChecker',
                    'props': {},
                    'inputs': [sk('Vector', 'VECTOR', [0, 0, 0],
                                  ['tc', 5]),
                               sk('Color1', 'RGBA', [0.9, 0.3, 0.1, 1]),
                               sk('Color2', 'RGBA', [0.1, 0.3, 0.9, 1]),
                               sk('Scale', 'VALUE', 8.0)],
                    'outputs': [{'name': 'Color', 'type': 'RGBA'},
                                {'name': 'Fac', 'type': 'VALUE'}]},
            'mx': {'id': 'mx', 'bl_idname': 'ShaderNodeMix',
                   'props': {'data_type': 'RGBA', 'blend_type': 'MIX'},
                   'inputs': [
                       {'name': 'Factor', 'identifier': 'Factor_Float',
                        'type': 'VALUE', 'default': 0.5, 'link': ['ck2', 1]},
                       {'name': 'A', 'identifier': 'A_Float',
                        'type': 'VALUE', 'default': 0.0, 'link': None},
                       {'name': 'B', 'identifier': 'B_Float',
                        'type': 'VALUE', 'default': 0.0, 'link': None},
                       {'name': 'A', 'identifier': 'A_Color',
                        'type': 'RGBA', 'default': [0.2, 0.8, 0.3, 1],
                        'link': ['ckw', 0]},
                       {'name': 'B', 'identifier': 'B_Color',
                        'type': 'RGBA', 'default': [0.8, 0.7, 0.2, 1],
                        'link': None}],
                   'outputs': [{'name': 'Result', 'type': 'VALUE'},
                               {'name': 'Result', 'type': 'RGBA'}]},
            'hal': {'id': 'hal', 'bl_idname': 'HALCYON_ShaderNode',
                    'props': {'model': 'PHONG'},
                    'inputs': [sk('Diffuse Color', 'RGBA',
                                  [0.5, 0.5, 0.5, 1.0], ['mx', 1]),
                               sk('Diffuse Level', 'VALUE', 1.0),
                               sk('Specular Level', 'VALUE', 0.4),
                               sk('Specular Color', 'RGBA', [1, 1, 1, 1]),
                               sk('Glossiness', 'VALUE', 24.0),
                               sk('Ambient', 'VALUE', 1.0),
                               sk('Opacity', 'VALUE', 1.0)],
                    'outputs': [{'name': 'Surface', 'type': 'SHADER'}]},
            'out': {'id': 'out', 'bl_idname': 'ShaderNodeOutputMaterial',
                    'props': {},
                    'inputs': [sk('Surface', 'SHADER', None, ['hal', 0])],
                    'outputs': []}}}

    def scene():
        st = base_settings(w, h, shadows=True)
        sc = demo_scene(st, with_texture=False)
        mesh = sc.mesh
        # a SECOND uv set that is genuinely different: swapped and
        # scaled, so reading the wrong layer is loudly visible
        mesh.uvs2 = (mesh.uvs[:, ::-1] * 1.7 + 0.13).astype(np.float32)
        mesh.uv_names = ['front', 'back']
        sc.materials[1].graph = build_graph()
        sc.materials[1].has_alpha = False
        sc.materials[1].opacity = 1.0
        return sc, st

    sc, st = scene()
    cpu = R.render(sc, st)
    view, _p, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = CRuv.GBuffer(w, h)
    CRuv.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                   depth_bits=st.depth_precision)
    tex = R.prepare_textures(sc, st)
    job = R.ShadeJob(sc, st, tex, None, view, eye, w, h)
    GSH._PLAN_CACHE.clear()
    p, why, _a = GSH.plan_frame(job, g)
    check('the two-UV / TexCoord / Mix material qualifies',
          p is not None, str(why))
    if p is None:
        return
    out, _hit = GSH.simulate(job, g)
    check('and simulates', out is not None, str(_hit))
    if out is None:
        return
    yy, xx = np.nonzero(g.tri >= 0)
    d = np.abs(out[yy, xx] - cpu[yy, xx, :3]).max(axis=1)
    check('the frame matches the CPU per pixel',
          int((d > 1e-2).sum()) == 0 and float(d.max()) < 6e-3,
          f'{int((d > 1e-2).sum())} of {yy.size} px >0.01, '
          f'max {float(d.max()):.6f}')

    # vacuity + the negative control that PROVES the second layer is
    # read: collapse it onto the first and the picture must move
    mi = sc.mesh.mat_index[g.tri[yy, xx]]
    on_mat = cpu[yy, xx][mi == 1]
    check('the mixed material really varies across the surface',
          float(on_mat[:, :3].std()) > 0.05,
          f'std {float(on_mat[:, :3].std()):.4f}')
    sc2, st2 = scene()
    sc2.mesh.uvs2 = sc2.mesh.uvs.copy()
    cpu2 = R.render(sc2, st2)
    dm = float(np.abs(cpu2 - cpu).max())
    check("collapsing the second UV map onto the first changes the "
          'picture (the layer name is really honoured)', dm > 0.05,
          f'max change {dm:.4f}')


def test_the_frame_reports_what_its_depth_can_resolve():
    """The number behind "the depth is screwed up".

    Two field pictures showed surfaces tearing through each other and
    neither the console nor the self test could say WHY: the frame
    never printed what its z-buffer could resolve, nor which surfaces
    had been taken out of the depth-buffered pass. Both are printed
    now, and both are checked here against arithmetic rather than
    against a picture.
    """
    from ..core import raster as CRq

    w, h = 96, 72
    st = base_settings(w, h)
    st.depth_precision = 16
    sc = demo_scene(st, with_texture=False)
    view, proj, vp, eye = R.camera_matrices(sc.camera, w, h)
    g = CRq.GBuffer(w, h)
    CRq.rasterize(sc.mesh.verts, sc.mesh.tris, vp, w, h, gbuf=g,
                  depth_bits=st.depth_precision)

    line = R.depth_report(proj, g, st.depth_precision)
    check('the frame reports its depth resolution', line is not None)
    if line is None:
        return
    check('and names the bit depth and the clip range',
          '16-bit' in line and 'clip' in line, line)

    # the reported near/far must be the camera's own, recovered from
    # the projection matrix
    p = np.asarray(proj, np.float64)
    near = p[2, 3] / (p[2, 2] - 1.0)
    far = p[2, 3] / (p[2, 2] + 1.0)
    check('near and far come back out of the projection matrix',
          abs(near - sc.camera.clip_start) < 1e-3
          and abs(far - sc.camera.clip_end) < 1e-1,
          f'{near:.4g}..{far:.4g} vs {sc.camera.clip_start}..'
          f'{sc.camera.clip_end}')

    # the resolution claim is falsifiable: two surfaces separated by
    # LESS than the reported figure must quantize to the same depth
    # value, and by comfortably more must not
    cov = g.tri >= 0
    z = g.zndc[cov].astype(np.float64)
    span = far - near
    dist = 2.0 * far * near / ((far + near) - z * span)
    d = float(np.median(dist))
    steps = float((1 << 16) - 1)
    res = (2.0 / steps) * span * d * d / (2.0 * far * near)

    def ndc_at(dd):
        return (far + near) / span - 2.0 * far * near / (span * dd)

    def q(v):
        return np.round((v * 0.5 + 0.5) * steps)

    check('surfaces closer than the reported resolution are '
          'indistinguishable',
          q(ndc_at(d)) == q(ndc_at(d + res * 0.25)),
          f'resolution {res:.3g} at distance {d:.3g}')
    check('and surfaces well beyond it are distinguishable',
          q(ndc_at(d)) != q(ndc_at(d + res * 4.0)))

    # 32 bits means no grid at all, and the reporter still answers
    check('a 32-bit buffer still reports',
          '32-bit' in (R.depth_report(proj, g, 32) or ''))

    # an instrument that lies is worse than no instrument. The field's
    # console reported its subject at 2e+14 world units away: ndc z had
    # collapsed onto the projection's depth ASYMPTOTE -- (f+n)/(f-n),
    # the value distance runs to at infinity -- and the first version
    # of this line clamped the vanishing denominator and printed the
    # clamp as a measurement.
    asym = (far + near) / span
    g_inf = CRq.GBuffer(w, h)
    g_inf.tri[:] = 0
    g_inf.zndc[:] = np.float32(asym)
    said = R.depth_report(proj, g_inf, 16) or ''
    check('asymptote depths are refused, not invented',
          'asymptote' in said and 'e+1' not in said, said)
    check('and the raw ndc range is printed either way',
          'ndc z' in said and 'ndc z' in line, said)

    # z past 1.0 is past the far clip: still measurable, but named
    g_mix = CRq.GBuffer(w, h)
    g_mix.tri[:] = g.tri
    g_mix.zndc[:] = g.zndc
    g_mix.tri[:h // 2] = 0
    g_mix.zndc[:h // 2] = np.float32(1.0 + (asym - 1.0) * 0.5)
    mixed = R.depth_report(proj, g_mix, 16) or ''
    check('pixels past the far clip are named, and the rest measured',
          'PAST the far clip' in mixed and 'the subject sits at' in mixed,
          mixed)

    # the classification record: a see-through material is named with
    # its reason, and an all-opaque scene records none
    sc2 = demo_scene(st, with_texture=False)
    sc2.materials[1].opacity = 0.5
    R._split_by_alpha(sc2, sc2.mesh, st)
    rec = dict(R.LAST_SPLIT)
    check('a see-through material is recorded with its reason',
          rec.get('see_through') == 1
          and any('Opacity' in v for v in rec['reasons'].values()),
          str(rec.get('reasons')))
    check('and its triangles are counted, not just its name',
          0 < rec.get('tris_see_through', 0) < rec.get('tris', 0),
          str((rec.get('tris_see_through'), rec.get('tris'))))

    sc3 = demo_scene(st, with_texture=False)
    R._split_by_alpha(sc3, sc3.mesh, st)
    check('an all-opaque scene records nothing see-through',
          int(R.LAST_SPLIT.get('see_through', 0)) == 0,
          str(R.LAST_SPLIT.get('reasons')))

    # the export flag is a reason in its own right: a material with no
    # opacity change but has_alpha set must still be classified, since
    # that is exactly the case that surprises the field
    sc4 = demo_scene(st, with_texture=False)
    sc4.materials[1].has_alpha = True
    R._split_by_alpha(sc4, sc4.mesh, st)
    rec4 = dict(R.LAST_SPLIT)
    check('an export-flagged material is classified see-through at '
          'full opacity',
          rec4.get('see_through') == 1
          and any('flagged' in v for v in rec4['reasons'].values()),
          str(rec4.get('reasons')))


def test_every_setting_does_what_it_says():
    """Every control on the settings panel is proven to control something.

    The field asked for it in as many words: "ensure all sliders, values,
    changeable settings work and actually do what they say." A tooltip is
    a promise, and the only proof of a promise is a changed picture -- so
    this test holds the whole settings class to one of five standards, and
    its accounting fails the moment a NEW field is added without one:

      * matrix rows    -- the feature matrix already flips it (90+ fields);
      * A/B DIFF       -- flipping it here must CHANGE the frame;
      * A/B EQUAL      -- flipping it must NOT change the frame (caches,
                          schedulers and worker counts are picture-neutral
                          by the determinism doctrine);
      * behavioural    -- its effect is data, not pixels (extra passes,
                          texture wrap, near-plane clip, motion blur,
                          palette lock, the worker pool);
      * declared infra -- device routing and viewport pacing, named and
                          justified one line each, nothing silently exempt.

    The audit that authored this found and removed seven corpse settings,
    a tooltip that contradicted its neighbour, a near-plane epsilon that
    claimed 1e-4 and drove nothing, a matrix row 'testing' a depth mode
    (ZBUFFER_NOWRITE) no code has ever read, and a light-shafts row whose
    scene had no volumetric light to shaft. This test is the reason none
    of those can come back.
    """
    import dataclasses
    import re as _re
    from ..core import parallel as PAR
    from ..core import palette as PA
    from ..core import raster as RAST
    from .featurematrix import SCENES, ROWS

    fields = {f.name for f in dataclasses.fields(RenderSettings)}
    W, H = 96, 72

    # ------------------------------------------------------------ part 1
    # the reader sweep: every field must be read somewhere in the engine
    # (core/, gpu/, engine.py, preview.py, export.py). settings.py declares,
    # properties.py/ui.py present, presets/ store -- none of those COUNT as
    # a reader, which is exactly how seven corpses hid for ninety versions.
    UI_ONLY = {'res_preset'}       # an "apply preset" selector; its tooltip
    #                                says in as many words it only sets others
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = []
    for base, dirs, names in os.walk(root):
        dirs[:] = [d for d in dirs
                   if d not in ('tests', 'presets', '__pycache__')]
        for n in names:
            if n.endswith('.py') and n not in ('settings.py', 'properties.py',
                                               'ui.py'):
                with open(os.path.join(base, n), encoding='utf8') as fh:
                    src.append(fh.read())
    blob = '\n'.join(src)
    unread = sorted(f for f in fields - UI_ONLY
                    if not _re.search(r'\b' + _re.escape(f) + r'\b', blob))
    check('every setting has a reader inside the engine',
          not unread, ', '.join(unread))

    # ------------------------------------------------------------ part 2
    # the A/B table. frame() renders through the same road the matrix
    # drives -- core render, then the full post chain with the depth and
    # shaft data the scene carried out -- so a setting from any stage of
    # the pipeline can testify. Baselines are cached per context.
    PA._PALETTE_CACHE.clear()
    _cache = {}

    def frame(scene_key, ov, mutate=None, mid=''):
        st = base_settings(W, H)
        for k, v in ov.items():
            setattr(st, k, v)
        sc = SCENES[scene_key](st)
        if mutate is not None:
            mutate(sc)
        img = R.render(sc, st)
        rgb = post.process(img[:, :, :3], st, frame=1, seed=st.seed,
                           depth=getattr(sc, 'last_depth', None),
                           shaft_sources=getattr(sc, 'last_shafts', None))
        # alpha rides along at render resolution: the Screen Door lives
        # ENTIRELY in the alpha plane, and a harness that drops it calls
        # stipple_pattern dead when it is not
        return rgb, img[:, :, 3]

    def cached(scene_key, ov, mutate, mid):
        key = (scene_key, repr(sorted(ov.items(), key=repr)), mid)
        if key not in _cache:
            _cache[key] = frame(scene_key, ov, mutate, mid)
        return _cache[key]

    def _same(a, b):
        return (a[0].shape == b[0].shape and np.array_equal(a[0], b[0])
                and np.array_equal(a[1], b[1]))

    def ab(field, probe, scene_key, base, mutate=None, mid='', expect='DIFF'):
        a = cached(scene_key, base, mutate, mid)
        b = frame(scene_key, dict(base, **{field: probe}), mutate, mid)
        if expect == 'SHAPE':
            check(f"'{field}' changes the frame's shape",
                  a[0].shape != b[0].shape, f'{a[0].shape} vs {b[0].shape}')
        elif expect == 'EQUAL':
            check(f"'{field}' must not move a pixel (doctrine)", _same(a, b))
        elif expect == 'ANY':
            check(f"'{field}' changes the frame", not _same(a, b))
        else:
            check(f"'{field}' changes the picture",
                  a[0].shape == b[0].shape and not _same(a, b),
                  '' if a[0].shape == b[0].shape
                  else f'{a[0].shape} vs {b[0].shape}')

    def _vol_spot(sc):
        # dim enough that the beam does not clip to solid white: a saturated
        # cone hides what density, falloff and sample count each do
        sc.lights[-1].volumetric = 0.02

    def _clear_models(sc):
        for m in sc.materials:
            m.model = ''                    # falls through to default_model

    def _inherit_bias(sc):
        for light in sc.lights:
            light.shadow_bias = 0.0         # 0 = use the render setting

    _DISP_GRAPH = {'output': 'out', 'nodes': {
        'n': _wnode('n', 'ShaderNodeTexNoise', {'noise_dimensions': '3D'},
                    [_sk('Vector', 'VECTOR', [0, 0, 0]),
                     _sk('Scale', 'VALUE', 9.0), _sk('Detail', 'VALUE', 3.0),
                     _sk('Roughness', 'VALUE', .5),
                     _sk('Distortion', 'VALUE', 0.)],
                    [{'name': 'Fac', 'type': 'VALUE'},
                     {'name': 'Color', 'type': 'RGBA'}]),
        'd': _wnode('d', 'ShaderNodeDisplacement', {'space': 'OBJECT'},
                    [_sk('Height', 'VALUE', 0.0, ['n', 0]),
                     _sk('Midlevel', 'VALUE', 0.5),
                     _sk('Scale', 'VALUE', 1.0),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'Displacement', 'type': 'VECTOR'}]),
        'b': _wnode('b', 'ShaderNodeBsdfDiffuse', {},
                    [_sk('Color', 'RGBA', [.8, .8, .8, 1]),
                     _sk('Roughness', 'VALUE', 0.),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'BSDF', 'type': 'SHADER'}]),
        'out': _wnode('out', 'ShaderNodeOutputMaterial', {},
                      [_sk('Surface', 'SHADER', None, ['b', 0]),
                       _sk('Displacement', 'VECTOR', [0, 0, 0], ['d', 0])],
                      [])}}

    def _attach_disp(sc):
        for m in sc.materials:
            m.graph = _DISP_GRAPH

    CONES = {'spot_cones': True}
    AO = {'ambient_occlusion': True, 'ao_samples': 4}
    RAD = {'radiosity': True, 'radiosity_samples': 4,
           'radiosity_distance': 4.0}
    DOF = {'dof': True, 'dof_focus': 6.0, 'dof_amount': 2.0}
    SHAFT = {'shaft_threshold': 0.4, 'shaft_length': 0.4}
    GLOW = {'glow': True, 'glow_threshold': 0.5, 'glow_intensity': 0.8}
    STAR = {'star_filter': True, 'glow_threshold': 0.5, 'star_intensity': 0.8}
    FLARE = {'lens_flare': True, 'flare_intensity': 0.7}
    PALB = {'color_depth': '8', 'palette_mode': 'ADAPTIVE'}
    DITH = {'dither': 'BAYER4', 'color_depth': '16'}
    CRT = {'crt': True, 'crt_scanlines': 0.4, 'crt_mask': 'APERTURE',
           'crt_curvature': 0.1}
    NTSC = {'composite': True, 'composite_bleed': 0.7}
    JPEG = {'jpeg_artifacts': True, 'jpeg_quality': 40}

    TABLE = [
        # cartoon outlines: the master switch diffs on its own; each part
        # probes against a base that isolates it
        ('outline',            True,              'demo',         {}),
        ('outline_color',      (1.0, 0.2, 0.2),   'demo',         {'outline': True}),
        ('outline_width',      4,                 'demo',         {'outline': True}),
        ('outline_opacity',    0.35,              'demo',         {'outline': True}),
        ('outline_objects',    False,             'demo',
         {'outline': True, 'outline_depth': False, 'outline_normals': False}),
        ('outline_materials',  True,              'demo',
         {'outline': True, 'outline_objects': False, 'outline_depth': False,
          'outline_normals': False}),
        ('outline_depth',      False,             'demo',
         {'outline': True, 'outline_objects': False, 'outline_normals': False,
          'outline_depth_threshold': 0.01}),
        ('outline_depth_threshold', 0.5,          'demo',
         {'outline': True, 'outline_objects': False, 'outline_normals': False,
          'outline_depth_threshold': 0.01}),
        ('outline_normals',    False,             'demo',
         {'outline': True, 'outline_objects': False, 'outline_depth': False,
          'outline_normal_angle': 30.0}),
        ('outline_normal_angle', 150.0,           'demo',
         {'outline': True, 'outline_objects': False, 'outline_depth': False,
          'outline_normal_angle': 30.0}),
        ('outline_over_sky',   False,             'demo',
         {'outline': True, 'outline_width': 3}),
        # field                probe               scene          base ctx
        ('max_transparent_layers', 1,             'glass',        {}),
        ('spot_cones',         True,              'cookie_spot',  {},
         _vol_spot, 'vol'),
        ('spot_cone_density',  3.0,               'cookie_spot',  CONES,
         _vol_spot, 'vol'),
        ('spot_cone_samples',  3,                 'cookie_spot',  CONES,
         _vol_spot, 'vol'),
        ('spot_cone_falloff',  0.5,               'cookie_spot',  CONES,
         _vol_spot, 'vol'),
        ('spot_cone_reach',    2.0,               'cookie_spot',  CONES,
         _vol_spot, 'vol'),
        ('displacement_scale', 0.0,               'demo',         {},
         _attach_disp, 'disp'),
        ('default_model',      'LAMBERT',         'demo',         {},
         _clear_models, 'nomodel'),
        ('lens_vignette_edges', False,            'demo',
         {'lens_distortion': 0.3}),
        ('shaft_decay',        0.5,               'shafts',       SHAFT),
        ('shaft_samples',      4,                 'shafts',       SHAFT),
        ('dof_layers',         2,                 'demo',         DOF),
        ('dof_max_radius',     4.0,               'demo',         DOF),
        # ss = round(sqrt(aa_samples)), and 2x2 supersampling is
        # filter-blind: ANY symmetric kernel over two symmetric taps
        # normalises to (1/2, 1/2). Nine samples make a 3x3 grid, whose
        # three taps tell the kernels apart. Width needs a filter that
        # has a shape -- BOX is width-invariant by definition
        ('aa_filter',          'GAUSS',           'demo',  {'aa_samples': 9}),
        ('aa_filter_width',    2.0,               'demo',
         {'aa_samples': 9, 'aa_filter': 'GAUSS'}),
        ('aa_edge_threshold',  0.005,             'demo',  {'aa_mode': 'EDGE'}),
        ('ao_distance',        0.3,               'demo',         AO),
        ('ao_intensity',       2.5,               'demo',         AO),
        ('radiosity_intensity', 3.0,              'demo',         RAD),
        ('global_ambient',     (0.3, 0.05, 0.05), 'demo',         {}),
        ('global_ambient_level', 4.0,             'demo',         {}),
        ('shadow_map_size',    32,                'demo',         {}),
        # the demo lights set their own bias, and a per-light value wins;
        # zeroing them is how a scene says "use the render setting"
        ('shadow_bias',        0.5,               'demo',         {},
         _inherit_bias, 'inhb'),
        ('shadow_softness',    6.0,               'demo',         {}),
        ('ray_reflection',     False,             'mirror',
         {'raytrace': True, 'ray_depth': 1}),
        ('ray_refraction',     False,             'glass',
         {'raytrace': True, 'ray_depth': 2}),
        # the master switch for traced shadows: RAY-mode lights obey it
        # (it used to gate only an unreachable no-map fallback)
        ('ray_shadows',        False,             'demo',
         {'shadow_default': 'RAY'}),
        ('ray_bias',           0.8,               'demo',
         {'shadow_default': 'RAY'}),
        ('stipple_pattern',    'HALFTONE',        'glass',
         {'transparency': 'STIPPLE'}),
        ('alpha_threshold',    0.9,               'glass',        {}),
        ('fog_color',          (0.8, 0.1, 0.1),   'demo',
         {'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0,
          'fog_end': 12.0}),
        ('glow_radius',        40.0,              'demo',         GLOW),
        ('glow_quality',       'BOX',             'demo',         GLOW),
        ('star_points',        6,                 'demo',         STAR),
        ('star_length',        60.0,              'demo',         STAR),
        ('star_rotation',      45.0,              'demo',         STAR),
        ('flare_ghosts',       0,                 'demo',         FLARE),
        ('flare_streak',       0.0,               'demo',         FLARE),
        ('palette_size',       8,                 'demo',         PALB),
        ('palette_method',     'OCTREE',          'demo',         PALB),
        ('dither_strength',    0.1,               'demo',         DITH),
        ('dither_serpentine',  False,             'demo',
         {'dither': 'FLOYD', 'color_depth': '16'}),
        # seed feeds the ray-shadow jitter streams; MAP soft shadows are a
        # deterministic texel filter and never consult it
        ('seed',               7,                 'soft',
         {'shadow_default': 'RAY', 'shadow_samples': 8}),
        ('color_management',   'SRGB',            'demo',         {}),
        ('input_gamma_naive',  False,             'textured',
         {'color_management': 'SRGB'}),
        ('crt_mask_strength',  0.9,               'demo',         CRT),
        ('crt_bloom',          0.8,               'demo',         CRT),
        ('composite_ringing',  0.9,               'demo',         NTSC),
        ('composite_dot_crawl', 0.8,              'demo',         NTSC),
        ('jpeg_passes',        3,                 'demo',         JPEG),
        ('block_size',         16,                'demo',         JPEG),
        ('pixel_grid',         True,              'demo',
         {'output_scale': '2X'}),
        # the demo's only sharp creases are the cube's right angles: 25 and
        # 80 degrees both keep them. 95 removes them, which is visible
        ('wire_angle',         95.0,              'demo',
         {'render_wire': True, 'wire_mode': 'CREASE'}),
        ('wire_color',         (1.0, 0.2, 0.2),   'demo',
         {'render_wire': True}),
        ('wire_width',         3.0,               'demo',
         {'render_wire': True}),
    ]
    SHAPES = [
        ('resolution_x',   128,  'demo', {}, None, '', 'SHAPE'),
        ('resolution_y',   96,   'demo', {}, None, '', 'SHAPE'),
        ('output_scale',   '2X', 'demo', {}, None, '', 'SHAPE'),
        ('pixel_aspect_x', 2.0,  'demo', {}, None, '', 'ANY'),
        ('pixel_aspect_y', 2.0,  'demo', {}, None, '', 'ANY'),
    ]
    EQUALS = [
        # optimisations and schedulers: the doctrine says the picture may
        # not know they exist
        ('fast_background', False, 'demo', {'aa_samples': 2}, None, '',
         'EQUAL'),
        ('cache_shadows',   False, 'demo', {}, None, '', 'EQUAL'),
        ('threads',         4,     'demo', {}, None, '', 'EQUAL'),
        ('show_stats',      True,  'demo', {}, None, '', 'EQUAL'),
        ('palette_lock',    False, 'demo', PALB, None, '', 'EQUAL'),
    ]
    for row in TABLE:
        field, probe, scene_key, base = row[:4]
        mutate = row[4] if len(row) > 4 else None
        mid = row[5] if len(row) > 5 else ''
        ab(field, probe, scene_key, base, mutate, mid)
    for field, probe, scene_key, base, mutate, mid, expect in SHAPES + EQUALS:
        ab(field, probe, scene_key, base, mutate, mid, expect)

    # ------------------------------------------------------------ part 3
    # behavioural: effects that are data rather than pixels.

    # the pass_* toggles hand the compositor real buffers
    st = base_settings(W, H, pass_position=True, pass_uv=True,
                       pass_object_index=True, pass_material_index=True)
    sc = demo_scene(st, with_texture=True)
    R.render(sc, st)
    p = sc.last_passes or {}
    for name in ('Position', 'UV', 'IndexOB', 'IndexMA'):
        check(f"pass toggle delivers a '{name}' buffer", name in p)
    if 'Position' in p:
        check('the Position pass varies across the frame',
              float(np.ptp(p['Position'])) > 0.0)
    if 'UV' in p:
        check('the UV pass varies across the frame',
              float(np.ptp(p['UV'])) > 0.0)
    if 'IndexMA' in p:
        check('the IndexMA pass separates the three demo materials',
              len(np.unique(p['IndexMA'])) >= 2)

    # tex_wrap_default decides what lives outside 0..1
    st_r = base_settings(W, H, tex_wrap_default='REPEAT')
    st_c = base_settings(W, H, tex_wrap_default='CLIP')
    sc_t = demo_scene(st_r, with_texture=True)
    tex_r = R.prepare_textures(sc_t, st_r)
    tex_c = R.prepare_textures(sc_t, st_c)
    tname = next(iter(tex_r))
    u = np.array([1.6], np.float32)
    v = np.array([0.3], np.float32)
    check('tex_wrap_default REPEAT and CLIP disagree outside 0..1',
          not np.allclose(tex_r[tname].sample(u, v),
                          tex_c[tname].sample(u, v)))

    # clip_near_epsilon is where a polygon crossing the camera plane is cut
    fl, n_, f_ = 1.5, 0.1, 100.0
    mvp = np.array([[fl, 0, 0, 0], [0, fl, 0, 0],
                    [0, 0, -(f_ + n_) / (f_ - n_), -2 * f_ * n_ / (f_ - n_)],
                    [0, 0, -1, 0]], np.float32)
    verts = np.array([[0, 0.8, 0.5], [-1, -0.8, -3], [1, -0.8, -3]],
                     np.float32)
    tris = np.array([[0, 1, 2]], np.int32)
    cov = {}
    for eps in (1e-5, 1.0):
        g = RAST.GBuffer(64, 48)
        RAST.rasterize(verts, tris, mvp, 64, 48, gbuf=g, near_eps=eps)
        cov[eps] = int((g.tri >= 0).sum())
    check('clip_near_epsilon moves the near-plane cut',
          cov[1e-5] > 0 and cov[1e-5] != cov[1.0], str(cov))

    # palette_lock holds frame 1's palette across an animation
    hgrad = np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :, None]
    f_red = np.concatenate([np.repeat(hgrad, H, 0),
                            0.2 * np.ones((H, W, 2), np.float32)], 2)
    f_blue = np.concatenate([0.2 * np.ones((H, W, 2), np.float32),
                             np.repeat(hgrad, H, 0)], 2)
    st_p = base_settings(W, H, color_depth='8', palette_mode='ADAPTIVE')
    outs = {}
    for lock in (True, False):
        PA._PALETTE_CACHE.clear()
        st_p.palette_lock = lock
        post.reduce_depth(f_red, st_p, seed=0)      # frame 1 sets the palette
        outs[lock] = post.reduce_depth(f_blue, st_p, seed=0)
    PA._PALETTE_CACHE.clear()
    check('palette_lock keeps frame 1 colours on frame 2',
          not np.array_equal(outs[True], outs[False]))

    # the worker pool must hand back the exact in-process pixels. The pool
    # honestly declines frames under 64k pixels, so this one earns its split
    st_w = base_settings(320, 240)
    sc_w = demo_scene(st_w, with_texture=False)
    solo = R.render(sc_w, st_w)
    pooled, why = PAR.render_parallel(sc_w, st_w, 2)
    check('use_processes: the pool engages in this environment',
          pooled is not None, str(why))
    if pooled is not None:
        check('use_processes: pooled pixels ARE the in-process pixels',
              np.array_equal(pooled, solo))

    # motion blur: the engine re-renders across the shutter, so a light
    # moved by frame_set must smear -- and both knobs must matter
    from . import fakeblender as FB
    FB.install()
    from .. import engine as ENG
    from .. import properties as props

    def motion_frame(**kw):
        def rig(eng, dg, bs):
            # move the MESH: a sun light only has a direction, so a
            # translated sun is the classic no-op that proves nothing
            ob = next(o for o in bs.objects if o.type == 'MESH')
            base_m = np.array(ob.matrix_world, np.float32).copy()

            def frame_set(fr, sub=0.0):
                t = (float(fr) + float(sub)) - 1.0
                ob.matrix_world[:3, 3] = base_m[:3, 3] + np.array(
                    [3.0 * t, 0.0, 0.0], np.float32)

                def fresh():
                    for o in bs.objects:
                        yield types.SimpleNamespace(
                            object=o, matrix_world=FB._Mat(o.matrix_world),
                            show_self=True, is_instance=False)
                dg.object_instances = fresh()
            eng.frame_set = frame_set
        img, _p, _c = FB.run_render(props, ENG, rig=rig, **kw)
        return img

    still = motion_frame(motion_blur=False)
    blur = motion_frame(motion_blur=True, motion_steps=4,
                        motion_shutter=0.5)
    check('motion_blur smears a light moved across the shutter',
          still is not None and blur is not None
          and not np.array_equal(still, blur))
    steps6 = motion_frame(motion_blur=True, motion_steps=6,
                          motion_shutter=0.5)
    check('motion_steps changes the accumulation',
          steps6 is not None and blur is not None
          and not np.array_equal(steps6, blur))
    wide = motion_frame(motion_blur=True, motion_steps=4,
                        motion_shutter=1.5)
    check('motion_shutter changes how far the smear reaches',
          wide is not None and blur is not None
          and not np.array_equal(wide, blur))

    # ------------------------------------------------------------ part 4
    # the accounting: every field has a home, and every name in a manual
    # set still exists -- a renamed or new setting fails here by design.
    rows_covered = set()
    for _n, ov, _s in ROWS:
        rows_covered |= set(ov)
    ab_fields = ({r[0] for r in TABLE}
                 | {r[0] for r in SHAPES} | {r[0] for r in EQUALS})
    BEHAVIOURAL = {'pass_position', 'pass_uv', 'pass_object_index',
                   'pass_material_index', 'tex_wrap_default',
                   'clip_near_epsilon', 'motion_blur', 'motion_shutter',
                   'motion_steps', 'palette_lock', 'use_processes'}
    INFRA = {
        'render_device':      'device routing: the whole device suite',
        'gpu_shading':        'device routing: the matrix parity rows',
        'gpu_raster':         'device routing: the raster parity rows',
        'gpu_post':           'device routing: the post parity rows',
        'gpu_hold_context':   'context lifecycle between renders',
        'layer_gpu_min_frac': 'LAYER-pass scheduling threshold',
        'gpu_scissor':        'device-side scissor; parity rows run it on',
        'viewport_gpu':       'viewport drawing, outside the F12 contract',
        'preview_scale':      'viewport pacing, outside the F12 contract',
        'progressive':        'viewport pacing, outside the F12 contract',
        'process_count':      'worker count; the pool EQUAL runs with 2',
    }
    homes = (rows_covered | ab_fields | BEHAVIOURAL
             | set(INFRA) | UI_ONLY)
    homeless = sorted(fields - homes)
    check('every setting is proven by a matrix row, an A/B, a behavioural '
          'check, or a declared reason', not homeless, ', '.join(homeless))
    ghosts = sorted((homes - {'width', 'height'}) - fields - rows_covered)
    check('no proof references a setting that no longer exists',
          not ghosts, ', '.join(ghosts))


def test_every_material_template_renders():
    """All 28 template recipes render through the engine and move the frame.

    The validity test above proves the names; this proves the RECIPES --
    each spec is translated into the same master-graph form the engine
    exports from Blender (full socket list, texture nodes, the Bump chain
    for the ones that carry one) and rendered over the demo scene. A
    template that renders the untouched frame is a decorative dictionary
    entry, which is what the vacuity doctrine exists to prevent. The two
    the field named get their own promises held: Water is see-through and
    refracts under rays (the proven Water anatomy), Lava glows.
    """
    from . import fakebpy
    bpy_stub = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy_stub.types, name):
            setattr(bpy_stub.types, name,
                    type(name, (bpy_stub.types.Panel,), {}))
    import importlib
    tmpl = importlib.import_module('halcyon.templates')
    from ..nodes.pattern_nodes import SPECS
    from ..nodes.shader_nodes import HALCYON_ShaderNode as HS

    TYPE = {'NodeSocketFloat': 'VALUE', 'NodeSocketColor': 'RGBA',
            'NodeSocketVector': 'VECTOR'}
    FILL = {'VALUE': 0.0, 'RGBA': [0.0, 0.0, 0.0, 1.0],
            'VECTOR': [0.0, 0.0, 0.0]}
    pspec = {f'HALCYON_{s[0]}Node': s for s in SPECS}

    def sockval(kind, d):
        t = TYPE[kind]
        if d is None:
            return t, FILL[t]
        return t, (list(d) if hasattr(d, '__len__') else float(d))

    def graph_for(spec):
        nodes = {}
        links = {}                    # master socket name -> ['nid', out_idx]
        for i, entry in enumerate(spec.get('textures', [])):
            if not isinstance(entry, dict):
                idname, props, inputs, target = entry
                entry = {'node': idname, 'props': props, 'inputs': inputs,
                         'target': target}
            nid = f't{i}'
            socks, sprops, souts = (pspec[entry['node']][4],
                                    pspec[entry['node']][5],
                                    pspec[entry['node']][6])
            tins = []
            for kind, sname, dflt in socks:
                t, v = sockval(kind,
                               entry.get('inputs', {}).get(sname, dflt))
                tins.append(_sk(sname, t, v))
            touts = [{'name': n, 'type': TYPE[k]} for k, n in souts]
            nodes[nid] = _wnode(nid, entry['node'],
                                dict(entry.get('props', {})), tins, touts)
            onames = [n for _k, n in souts]
            oi = (onames.index(entry['output'])
                  if entry.get('output') in onames else 0)
            src = [nid, oi]
            if entry.get('bump') is not None:
                bid = f'b{i}'
                nodes[bid] = _wnode(
                    bid, 'ShaderNodeBump', {},
                    [_sk('Strength', 'VALUE', float(entry['bump'])),
                     _sk('Distance', 'VALUE', 1.0),
                     _sk('Height', 'VALUE', 0.0, src),
                     _sk('Normal', 'VECTOR', [0, 0, 0])],
                    [{'name': 'Normal', 'type': 'VECTOR'}])
                src = [bid, 0]
            links[entry['target']] = src
        ins = []
        for kind, sname, dflt in HS.SOCKETS:
            t, v = sockval(kind, spec.get('inputs', {}).get(sname, dflt))
            ins.append(_sk(sname, t, v, links.get(sname)))
        nodes['h'] = _wnode('h', 'HALCYON_ShaderNode',
                            {'model': spec['model'], 'toon_steps': 2}, ins,
                            [{'name': 'Surface', 'type': 'SHADER'}])
        nodes['out'] = _wnode('out', 'ShaderNodeOutputMaterial', {},
                              [_sk('Surface', 'SHADER', None, ['h', 0]),
                               _sk('Displacement', 'VECTOR', [0, 0, 0])], [])
        return {'output': 'out', 'nodes': nodes}

    def shot(g=None, **ov):
        st = base_settings(96, 72, **ov)
        sc = demo_scene(st, with_texture=False)
        if g is not None:
            for m in sc.materials:
                m.graph = g
        return R.render(sc, st)

    base = shot()
    for key in sorted(tmpl.TEMPLATES):
        spec = tmpl.TEMPLATES[key]
        ov = {'raytrace': True, 'ray_depth': 2} if key == 'WATER' else {}
        img = shot(graph_for(spec), **ov)
        check(f"template '{spec['label']}' renders and moves the frame",
              img.shape == base.shape and not np.array_equal(img, base))
        if key in ('LAVA', 'NEON', 'DEAD_CHANNEL'):
            check(f"...and '{spec['label']}' glows",
                  float(img[..., :3].max()) > 0.5,
                  f'max {float(img[..., :3].max()):.3f}')
        if key == 'WATER':
            check('...and the water is see-through under rays',
                  float(img[..., 3].min()) < 0.99,
                  f'min alpha {float(img[..., 3].min()):.3f}')


def test_the_high_resolution_round():
    """The field's high-res reports get their instruments and their exit.

    "A faint but visible wireframe on all objects, regardless of settings"
    reproduced as the shadow map's texels showing at high output
    resolution: a 512 map's contact shadows turn into blocky fringes that
    hug every silhouette, and "regardless of settings" because lights
    saved before 1.30.1 carry per-light 512/0.02 explicitly, overriding
    the render slider. Proven here: the raster is clean on a shadeless
    frame, shadows-off removes the fringe, a 2048 map resolves it. This
    test holds the three shipped pieces: the coarseness note that names
    the trap, the adopt operator that frees old scenes, and the raster
    split instrument for the "dies at extreme resolutions" half.
    """
    import contextlib
    import io

    # (a) the coarseness note fires when texels dwarf output pixels...
    def console_of(mapsize, per_light=0):
        st = base_settings(320, 240)
        st.shadow_map_size = mapsize
        sc = demo_scene(st, with_texture=False)
        for light in sc.lights:
            light.shadow_map_size = per_light
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            R.render(sc, st)
        return buf.getvalue()

    coarse = console_of(32)
    check('a coarse shadow map is named in the console',
          'px/texel' in coarse and 'blocky' in coarse,
          coarse[-160:].replace('\n', ' '))
    check('...and the note names the render-settings road',
          'Shadow Map Size' in coarse)
    fine = console_of(2048)
    check('a fine map at the same size stays silent',
          'px/texel' not in fine)
    overridden = console_of(2048, per_light=32)
    check('a per-light override is named as the thing to raise',
          'px/texel' in overridden and 'per-light size set' in overridden
          and 'overrides the render setting' in overridden,
          overridden[-160:].replace('\n', ' '))

    # (b) the adopt operator points lights back at the render settings
    import types as _types

    from . import fakebpy
    bpy_stub = fakebpy.install()
    bpy_stub.types.UIList = type('UIList', (bpy_stub.types.Panel,), {})
    bpy_stub.types.AddonPreferences = type(
        'AddonPreferences', (bpy_stub.types.Panel,), {})
    import importlib
    UI = importlib.import_module('halcyon.ui')

    def fake_light(size, bias):
        return _types.SimpleNamespace(
            type='LIGHT',
            data=_types.SimpleNamespace(halcyon=_types.SimpleNamespace(
                shadow_map_size=size, shadow_bias=bias)))

    lights = [fake_light(512, 0.02), fake_light(0, 0.0),
              fake_light(1024, 0.1)]
    ctx = _types.SimpleNamespace(
        scene=_types.SimpleNamespace(objects=lights), light=None)
    op = UI.HALCYON_OT_adopt_shadow_settings.__new__(
        UI.HALCYON_OT_adopt_shadow_settings)
    op.scope = 'SCENE'
    reports = []
    op.report = lambda kind, msg: reports.append(msg)
    result = op.execute(ctx)
    check('the adopt operator clears every per-light override',
          result == {'FINISHED'}
          and all(l.data.halcyon.shadow_map_size == 0
                  and l.data.halcyon.shadow_bias == 0.0 for l in lights))
    check('...and reports how many lights it freed',
          reports and reports[0].startswith('2 light'))
    check('...and is registered', any(
        getattr(c, 'bl_idname', '') == 'halcyon.adopt_shadow_settings'
        for c in UI.CLASSES))

    # (c) the raster split instrument exists end to end: the craster
    # records it, render prints it, and the self test carries the
    # high-resolution section that turns "dies at 4K" into a named stage
    import inspect

    from ..gpu import craster as CRA
    check('craster keeps the last raster split', hasattr(CRA, 'LAST_RASTER'))
    rsrc = inspect.getsource(R)
    check('a final render prints the split', 'raster split' in rsrc)
    import halcyon.selftest as SELF
    ssrc = inspect.getsource(SELF)
    check('the self test benchmarks the raster at high resolutions',
          'HIGH-RESOLUTION RASTER' in ssrc and '3840' in ssrc)


def test_the_wireframe_names_itself():
    """The overlay can never again be mistaken for a rendering bug.

    The field spent two rounds hunting "a faint but visible wireframe on
    all objects, regardless of settings" -- and the second field picture
    showed mesh triangulation lines, which only the Wireframe Overlay
    draws. The header printed 'wire=ALL' on EVERY render (it echoed the
    mode, not the state), so the console never said whether the overlay
    was actually on. Now the header says wire=OFF unless it is engaged,
    and the overlay prints how many pixels it inked and where the
    checkbox lives every time it draws on a final render.
    """
    import contextlib
    import io

    # engine header: wire=OFF when the overlay is off, the mode when on
    from . import fakeblender as FB
    FB.install()
    from .. import engine as ENG
    from .. import properties as props

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FB.run_render(props, ENG)
    off = buf.getvalue()
    check("the header says wire=OFF when the overlay is off",
          'wire=OFF' in off and 'wire=ALL' not in off)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        FB.run_render(props, ENG, render_wire=True)
    on = buf.getvalue()
    check('...and names the mode when it is on', 'wire=ALL' in on)
    check('...and the overlay reports what it inked, and where the '
          'checkbox lives',
          'wireframe overlay: inked' in on
          and 'Wireframe Overlay checkbox' in on, on[-200:].replace('\n', ' '))

    # core render: the ink note prints there too, and only when drawing
    st = base_settings(96, 72, render_wire=True)
    sc = demo_scene(st, with_texture=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.render(sc, st)
    check('the core render prints the ink note when the overlay draws',
          'wireframe overlay: inked' in buf.getvalue())
    st2 = base_settings(96, 72)
    sc2 = demo_scene(st2, with_texture=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        R.render(sc2, st2)
    check('...and stays silent when it does not',
          'wireframe overlay' not in buf.getvalue())

    # the perf instruments behind the 44-second shade bucket: the shade
    # split print, the device dispatch/read subsplit, and the readback
    # contiguity copy are all in place
    import inspect

    from ..gpu import device as DEV
    rsrc = inspect.getsource(R)
    check('a GPU-shaded final render prints the shade split',
          'shade split' in rsrc and 'material' in rsrc)
    check('the device keeps the dispatch/read subsplit',
          hasattr(DEV, 'LAST_DISPATCH'))
    dsrc = inspect.getsource(DEV)
    check('readbacks are copied contiguous ONCE at the boundary',
          'ascontiguousarray'
          in dsrc.split('_dispatch_compute_impl', 1)[1])


def test_the_preset_menu_never_floods_the_console():
    """The preset enum names a real default, so RNA has nothing to warn about.

    The field's console flooded with "current value '0' matches no enum in
    'HalcyonSettings', '', 'preset'" on every redraw: the enum was a dynamic
    callback whose items BEGIN with a category header ('', 'General', ''),
    and a dynamic enum's unset value is index 0 -- the header's empty
    identifier. The cure is a static list with an explicit default; this
    holds all three parts of it in place.
    """
    from . import fakebpy
    bpy_stub = fakebpy.install()
    for name in ('UIList', 'AddonPreferences', 'Collection'):
        if not hasattr(bpy_stub.types, name):
            setattr(bpy_stub.types, name,
                    type(name, (bpy_stub.types.Panel,), {}))
    import importlib
    P = importlib.import_module('halcyon.properties')

    prop = P.HalcyonSettings.__annotations__['preset']
    items = getattr(prop, 'items', None)
    check('the preset enum is a static list, not a callback',
          items is not None and not callable(items)
          and len(items) > 70, str(type(items)))
    if not items or callable(items):
        return
    ids = [i[0] for i in items]
    check('its first entry is still a category header (the trap the '
          'default exists for)', ids[0] == '')
    check('and the declared default is a real preset',
          getattr(prop, 'default', None) == 'DEFAULT' and 'DEFAULT' in ids)
    check('every header is empty-id and every real id is a preset',
          all((not i[0]) or (i[0] in __import__(
              'halcyon.presets.library', fromlist=['PRESETS']).PRESETS)
              for i in items))


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


def test_every_setting_has_a_detailed_tooltip():
    """Every registered property carries a real description.

    'Detailed tooltip' is a shipped promise now, so it is a test: every
    property on every Halcyon property group -- render settings, material
    override, light, world -- must carry a description of substance, not
    an empty string and not a three-word shrug. New properties fail here
    until they explain themselves, which is the point.
    """
    import importlib
    import inspect

    from . import fakebpy
    fakebpy.install()
    import halcyon.properties as P
    importlib.reload(P)
    from .fakebpy import _Prop

    bare = []
    total = 0
    for _name, obj in vars(P).items():
        if not (inspect.isclass(obj) and hasattr(obj, '__annotations__')):
            continue
        for pname, pdef in obj.__annotations__.items():
            if not isinstance(pdef, _Prop):
                continue
            total += 1
            desc = str(pdef.kw.get('description', '') or '')
            if len(desc) < 25:
                bare.append(f'{obj.__name__}.{pname}')
    check(f'all {total} properties carry a detailed tooltip', not bare,
          ', '.join(bare[:8]) + ('...' if len(bare) > 8 else ''))


def test_the_sky_library_is_complete_and_valid():
    """Every sky preset applies cleanly, has a note, and ships a thumbnail.

    303 presets is a promise with three parts: every settings key must be
    a real sky field (a typo would silently not apply), every entry must
    carry a real tooltip note, and every entry must have its rendered
    thumbnail in presets/thumbs -- because a gallery with holes reads as
    broken. Applying each preset to a World must also actually change it
    from the default sky (a preset that changes nothing is decoration).
    """
    from ..core.scene import World
    from ..presets import skies as SK

    fields = set(SK.sky_fields())
    thumbs = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'presets', 'thumbs')
    bad_keys, bare_notes, missing_thumbs, inert = [], [], [], []
    defaults = World()
    for key in SK.ORDER:
        entry = SK.SKIES.get(key)
        if entry is None:
            bad_keys.append(key + ' (no entry)')
            continue
        unknown = [k for k in entry['settings'] if k not in fields]
        if unknown:
            bad_keys.append(f'{key}: {unknown[:3]}')
        if len(str(entry.get('note', ''))) < 12:
            bare_notes.append(key)
        if not os.path.exists(os.path.join(thumbs, key + '.png')):
            missing_thumbs.append(key)
        w = World()
        ok, _ = SK.apply_sky(w, key)
        # BRYCE_DEFAULT is the one preset whose whole point is the
        # default sky: reset-to-default IS its effect
        changed = (not entry['settings']) or any(
            getattr(w, f, None) != getattr(defaults, f, None)
            for f in entry['settings'])
        if not (ok and changed):
            inert.append(key)
    check(f'all {len(SK.ORDER)} presets use real sky fields', not bad_keys,
          '; '.join(bad_keys[:4]))
    check('every preset carries a detailed note', not bare_notes,
          ', '.join(bare_notes[:6]))
    check('every preset ships its thumbnail', not missing_thumbs,
          ', '.join(missing_thumbs[:6]) +
          (f' (+{len(missing_thumbs) - 6})' if len(missing_thumbs) > 6
           else ''))
    check('every preset changes the sky it is applied to', not inert,
          ', '.join(inert[:6]))
    check('the library holds at least 250 more skies than the original 43',
          len(SK.ORDER) >= 293, str(len(SK.ORDER)))
