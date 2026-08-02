"""The renderer: rasterise, shade, composite.

Pipeline order, which is the period-correct one rather than the path-traced one:

    supersample -> rasterise opaque z-buffer -> reconstruct fragment attributes
    -> evaluate node graph per material -> resolve closure to a reflectance
    model -> light it -> ray-traced reflection/refraction -> A-buffer
    transparency -> fog -> downsample

Everything here is bpy-free so the whole thing can be exercised headlessly.
"""

import os

import numpy as np

from . import lights as LI
from . import stats as ST
from . import mathx as M
from . import raster
from . import shading as SH
from .bvh import BVH
from .nodeeval import Closure, GraphEvaluator, ShadeContext, to_color, to_value
from .texture import Texture

EMPTY = -1


# ------------------------------------------------------------------ camera


def camera_matrices(camera, width, height):
    """(view, proj, viewproj, eye) from a Camera."""
    if camera is None or camera.matrix_world is None:
        mw = np.eye(4, dtype=np.float32)
        mw[2, 3] = 8.0
    else:
        mw = np.asarray(camera.matrix_world, np.float32)
    view = np.linalg.inv(mw).astype(np.float32)
    eye = mw[:3, 3].astype(np.float32)
    if camera is not None and camera.projection is not None:
        proj = np.asarray(camera.projection, np.float32)
    else:
        near = camera.clip_start if camera else 0.1
        far = camera.clip_end if camera else 1000.0
        aspect = width / max(height, 1)
        if camera is not None and camera.type == 'ORTHO':
            half = (camera.ortho_scale or 6.0) * 0.5
            proj = LI._ortho(half, near, far)
            proj[0, 0] = 1.0 / (half * aspect)
            proj[1, 1] = 1.0 / half
        else:
            lens = camera.lens if camera else 50.0
            sensor = camera.sensor if camera else 36.0
            fov_x = 2.0 * np.arctan(sensor * 0.5 / max(lens, 1e-3))
            fov_y = 2.0 * np.arctan(np.tan(fov_x * 0.5) / max(aspect, 1e-6))
            proj = LI._persp(fov_y, aspect, near, far)
    return view, proj, (proj @ view).astype(np.float32), eye


def pixel_footprint(camera, proj, height):
    """How much of the world one rendered pixel covers.

    Returns (radians, metres): the angle a pixel subtends for a perspective
    camera, or the constant width it covers for an orthographic one. The
    infinite water plane needs this to know which waves are small enough that
    drawing them would only alias -- and it was reading a hard-coded guess of
    0.002 rad, which is roughly four times coarser than a 480-row frame at a
    normal lens and is why the waves stopped being drawn so close in.

    `height` is the *rendered* height, so supersampling is already in it: a
    frame rendered at 4x resolves waves a quarter the size, which it should.
    """
    if camera is not None and getattr(camera, 'type', '') == 'ORTHO':
        return 0.0, float(getattr(camera, 'ortho_scale', 6.0)) / max(height, 1)
    f = abs(float(proj[1, 1])) if proj is not None else 1.0
    fov_y = 2.0 * np.arctan(1.0 / max(f, 1e-6))
    return float(fov_y) / max(height, 1), 0.0


# ---------------------------------------------------------------- textures


_TEX_CACHE = {}


def clear_caches():
    _TEX_CACHE.clear()
    LI.clear_shadow_cache()


def prepare_textures(scene, settings):
    """ImageBuffers -> sampling-ready Textures, with the era's limits applied.

    Mip building and colour quantisation are not free, and neither depends on
    anything that changes between frames, so the result is cached per image and
    per the settings that affect it.
    """
    out = {}
    sig = (settings.tex_filter, settings.tex_wrap_default, settings.tex_max_size,
           settings.tex_quantize, settings.tex_mipmap, settings.color_management,
           settings.input_gamma_naive)
    images = getattr(scene, 'images', None) or {}
    for key, buf in images.items():
        if buf is None:
            continue
        px = getattr(buf, 'pixels', None)
        if px is None:
            continue
        ckey = (key, id(px), px.shape, sig)
        cached = _TEX_CACHE.get(ckey)
        if cached is not None:
            out[key] = cached
            continue
        tex = Texture(px, name=getattr(buf, 'name', ''),
                      colorspace=getattr(buf, 'colorspace', 'sRGB'),
                      wrap=settings.tex_wrap_default, filt=settings.tex_filter)
        if settings.color_management != 'NONE' and not settings.input_gamma_naive:
            tex.to_linear()
        if settings.tex_max_size:
            tex.clamp_size(settings.tex_max_size)
        if settings.tex_quantize:
            tex.quantize(settings.tex_quantize)
        if settings.tex_mipmap:
            tex.build_mips()
        if len(_TEX_CACHE) > 32:
            _TEX_CACHE.clear()
        _TEX_CACHE[ckey] = tex
        out[key] = tex
    return out


# ----------------------------------------------------------- closure resolve


def closure_to_surface(cl, ctx, settings, material=None):
    """Collapse a node-graph closure into one reflectance model + parameters.

    Cycles-style closures are additive lobes; a 1990s renderer has one shader
    with a diffuse and a specular term. This is the honest translation: sum the
    weighted lobes into those slots and pick the model the tree implies.
    """
    n = ctx.n
    surf = SH.Surface(n)
    model = None
    if material is not None:
        surf.diffuse[:] = np.asarray(material.diffuse, np.float32)[None, :]
        surf.specular[:] = np.asarray(material.specular, np.float32)[None, :]
        surf.glossiness[:] = material.glossiness
        surf.specular_level[:] = material.specular_level
        surf.diffuse_level[:] = material.diffuse_level
        surf.ambient[:] = material.ambient_level
        surf.opacity[:] = material.opacity
        surf.ior[:] = material.ior
        surf.roughness[:] = material.roughness
        surf.metallic[:] = material.metallic
        surf.anisotropy[:] = material.anisotropy
        surf.aniso_rot[:] = material.aniso_rotation
        surf.reflect[:] = material.reflect_level
        surf.emission[:] = np.asarray(material.emission, np.float32)[None, :] * \
            material.emission_level
        model = material.model

    if not isinstance(cl, Closure) or not cl.items:
        chosen = model or settings.default_model
        if settings.force_model != 'NONE':
            chosen = settings.force_model
        surf.model = chosen
        return surf, chosen, None

    diff = np.zeros((n, 3), np.float32)
    diff_w = np.zeros(n, np.float32)
    spec = np.zeros((n, 3), np.float32)
    spec_w = np.zeros(n, np.float32)
    emis = np.zeros((n, 3), np.float32)
    transp = np.zeros(n, np.float32)
    refr = np.zeros(n, np.float32)
    refr_col = np.zeros((n, 3), np.float32)
    rough_acc = np.zeros(n, np.float32)
    rough_w = np.zeros(n, np.float32)
    normal = None
    halcyon = None
    gloss_model = None
    diff_model = None

    for kind, w, p in cl.items:
        w = np.clip(np.asarray(w, np.float32).reshape(-1), 0.0, None)
        if w.shape[0] != n:
            w = np.broadcast_to(w, (n,)).copy()
        col = p.get('color')
        rgb = to_color(col, n)[:, :3] if col is not None else np.ones((n, 3), np.float32)
        if p.get('normal') is not None:
            normal = p['normal']
        if kind == 'HALCYON':
            halcyon = p
            continue
        if kind == 'DIFFUSE':
            diff += rgb * w[:, None]
            diff_w += w
            diff_model = p.get('model', diff_model)
            r = p.get('roughness')
            if r is not None:
                rough_acc += to_value(r, n) * w
                rough_w += w
        elif kind == 'GLOSSY':
            spec += rgb * w[:, None]
            spec_w += w
            gloss_model = p.get('model', gloss_model)
            r = p.get('roughness')
            if r is not None:
                rough_acc += to_value(r, n) * w
                rough_w += w
            if p.get('metallic') is not None:
                surf.metallic = to_value(p['metallic'], n)
            if p.get('anisotropy') is not None:
                surf.anisotropy = to_value(p['anisotropy'], n)
            if p.get('rotation') is not None:
                surf.aniso_rot = to_value(p['rotation'], n)
        elif kind == 'EMISSION':
            st = p.get('strength')
            emis += rgb * w[:, None] * (to_value(st, n)[:, None] if st is not None else 1.0)
        elif kind == 'TRANSPARENT':
            transp += w
        elif kind in ('GLASS', 'REFRACTION'):
            refr += w
            refr_col += rgb * w[:, None]
            spec += rgb * w[:, None] * 0.5
            spec_w += w * 0.5
            if p.get('ior') is not None:
                surf.ior = to_value(p['ior'], n)
            gloss_model = gloss_model or 'COOK_TORRANCE'
        elif kind == 'TRANSLUCENT':
            surf.translucency = np.maximum(surf.translucency, w)
            diff += rgb * w[:, None] * 0.5
            diff_w += w * 0.5
        elif kind == 'HOLDOUT':
            transp += w

    if halcyon is not None:
        p = halcyon
        surf.diffuse = to_color(p['color'], n)[:, :3]
        surf.diffuse_level = to_value(p['diffuse_level'], n)
        surf.specular = to_color(p['spec_color'], n)[:, :3]
        surf.specular_level = to_value(p['spec_level'], n)
        surf.glossiness = np.maximum(to_value(p['glossiness'], n), 0.5)
        surf.roughness = np.clip(to_value(p['roughness'], n), 0.0, 1.0)
        surf.ambient = to_value(p['ambient'], n)
        surf.emission = to_color(p['emission'], n)[:, :3]
        surf.opacity = np.clip(to_value(p['opacity'], n), 0.0, 1.0)
        surf.ior = np.maximum(to_value(p['ior'], n), 1.0)
        surf.anisotropy = to_value(p['anisotropy'], n)
        surf.aniso_rot = to_value(p['rotation'], n)
        surf.metallic = to_value(p['metallic'], n)
        surf.soften = to_value(p['soften'], n)
        surf.reflect = to_value(p['reflect'], n)
        surf.translucency = to_value(p['translucency'], n)
        surf.toon_size = to_value(p['toon_size'], n)
        surf.toon_smooth = to_value(p['toon_smooth'], n)
        if p.get('toon_steps') is not None:
            surf.toon_steps = to_value(p['toon_steps'], n)
        for key, attr, kind in (
                ('fresnel', 'fresnel', 'v'), ('fresnel_power', 'fresnel_power', 'v'),
                ('fresnel_color', 'fresnel_color', 'c'), ('rim', 'rim', 'v'),
                ('rim_power', 'rim_power', 'v'), ('rim_color', 'rim_color', 'c'),
                ('matcap', 'matcap', 'c'), ('matcap_blend', 'matcap_blend', 'v'),
                ('reflect_color', 'reflect_color', 'c'),
                ('edge_opacity', 'edge_opacity', 'v'),
                ('backface_color', 'backface_color', 'c'),
                ('backface_mix', 'backface_mix', 'v'),
                ('sheen', 'sheen', 'v'), ('sheen_color', 'sheen_color', 'c'),
                ('sheen_roughness', 'sheen_roughness', 'v'),
                ('refraction', 'refraction', 'v')):
            if p.get(key) is not None:
                setattr(surf, attr, to_color(p[key], n)[:, :3] if kind == 'c'
                        else to_value(p[key], n))
        model = p.get('model', model)
        if p.get('normal') is not None:
            normal = p['normal']
            # Bump Strength scales how far the supplied normal is allowed to
            # bend away from the surface it sits on. Done here rather than in
            # the node graph because the geometric normal is only known once
            # the closure has been collapsed against a fragment.
            bs = p.get('bump_strength')
            if bs is not None:
                k = to_value(bs, n)[:, None]
                if np.any(np.abs(k - 1.0) > 1e-4):
                    geo = M.normalize(np.asarray(ctx.N, np.float32))
                    normal = geo + (M.normalize(np.asarray(normal, np.float32))
                                    - geo) * k
    else:
        w = np.maximum(diff_w, 1e-6)
        if np.any(diff_w > 1e-6):
            surf.diffuse = diff / w[:, None]
            surf.diffuse_level = np.clip(diff_w, 0.0, 1.0)
        if np.any(spec_w > 1e-6):
            surf.specular = spec / np.maximum(spec_w, 1e-6)[:, None]
            surf.specular_level = np.clip(spec_w, 0.0, 1.0)
        else:
            surf.specular_level = np.zeros(n, np.float32)
        if np.any(rough_w > 1e-6):
            r = np.clip(rough_acc / np.maximum(rough_w, 1e-6), 0.0, 1.0)
            surf.roughness = r
            # Blender roughness -> a Phong/Blinn exponent, the classic mapping
            surf.glossiness = np.maximum(2.0 / np.maximum(r * r * r * r, 1e-5) - 2.0,
                                         0.5).astype(np.float32)
            surf.glossiness = np.minimum(surf.glossiness, 8192.0)
        surf.emission = emis
        surf.opacity = np.clip(1.0 - transp, 0.0, 1.0)
        surf.reflect = np.clip(refr, 0.0, 1.0)
        if model is None:
            model = gloss_model or diff_model

    if settings.force_model != 'NONE':
        model = settings.force_model
    if model is None:
        model = settings.default_model
    surf.model = model
    return surf, model, normal


# ----------------------------------------------------------------- lighting


def light_surface(surf, model, ctx, scene, settings, bvh=None, rng=None,
                  active_lights=None):
    """Direct lighting + ambient + emission for a batch of points."""
    n = ctx.n
    N = M.normalize(ctx.N)
    V = -M.normalize(ctx.I)
    if settings.two_sided_lighting:
        flip = M.dot(N, V) < 0.0
        N = np.where(flip[:, None], -N, N)

    if model in ('CONSTANT', 'WIREFRAME'):
        return surf.diffuse * surf.diffuse_level[:, None] + surf.emission

    # Lambertian BRDF normalisation. Light energy arrives in Blender's watt-based
    # units, and Cycles divides reflected radiance by pi; without this every
    # surface renders pi times too bright and clips to white, which hides the
    # material colour entirely. Applied to both lobes so the diffuse/specular
    # balance -- and the period-correct unnormalised highlight shape -- is
    # untouched. This is a units conversion, not a change to the models.
    inv_pi = np.float32(1.0 / np.pi)

    amb_col = LI.ambient_light(scene, settings)
    out = surf.diffuse * surf.ambient[:, None] * amb_col[None, :]

    spx = getattr(ctx, 'spx', None)
    spy = getattr(ctx, 'spy', None)
    have_id = spx is not None and spy is not None

    if settings.ambient_occlusion and bvh is not None:
        out *= ambient_occlusion(ctx.P, N, bvh, settings, rng,
                                 sample_xy=(spx, spy) if have_id
                                 else None)[:, None]

    # the sheen lobe's falloff, computed once rather than per light
    sheen_exp = None
    if np.any(surf.sheen > 1e-4):
        r = np.clip(surf.sheen_roughness, 0.0, 1.0)
        sheen_exp = 1.0 + (1.0 - r) * 15.0
        edge_vn = np.clip(1.0 - np.abs(M.dot(N, V)), 0.0, 1.0)

    lights = active_lights if active_lights is not None else \
        LI.select_lights(scene.lights, settings)
    clamp = float(settings.light_clamp)

    obj_idx = getattr(ctx, 'object_index_raw', None)
    for li, light in enumerate(lights):
        lit_mask = None
        excl = getattr(light, 'exclude_objects', None)
        if excl and obj_idx is not None:
            inside = np.isin(obj_idx, np.asarray(list(excl), np.int32))
            lit_mask = inside if light.exclude_mode == 'ONLY' else ~inside
            if not lit_mask.any():
                continue
        L, rad, dist = LI.sample(light, ctx.P, settings)
        ndl = M.dot(N, L)
        if not np.any(ndl > 0.0) and not np.any(surf.translucency > 0):
            continue
        vis = LI.visibility(light, ctx.P, N, L, dist, settings, bvh, rng,
                            sample_xy=(spx, spy, li) if have_id else None)
        if not np.any(vis > 0.0):
            continue
        dif, spec = SH.evaluate(model, surf, N, L, V)
        if light.negative:
            rad = -rad
        contrib = np.zeros((n, 3), np.float32)
        if light.affect_diffuse and not light.specular_only:
            contrib += (dif[:, None] * surf.diffuse *
                        surf.diffuse_level[:, None]) * rad
        if light.affect_specular and not light.diffuse_only:
            sp = spec
            if not settings.specular_in_gamma:
                sp = np.power(np.maximum(sp, 0.0), 2.2)
            contrib += sp * surf.specular_level[:, None] * rad
            if sheen_exp is not None:
                # velvet: light scattered back at grazing angles, so the lobe
                # lives at the silhouette and vanishes face-on. It still needs
                # a light -- unlike the rim term, which is the cheat version of
                # the same look and needs none.
                sh = (np.power(edge_vn, sheen_exp) *
                      np.maximum(ndl, 0.0) * surf.sheen)
                contrib += surf.sheen_color * sh[:, None] * rad
        contrib *= inv_pi
        contrib *= vis[:, None]
        if lit_mask is not None:
            contrib *= lit_mask[:, None]
        if clamp > 0.0:
            contrib = np.minimum(contrib, clamp)
        out += contrib

    if settings.clamp_specular:
        out = np.minimum(out, 64.0)
    out = out + surf.emission
    return apply_surface_effects(out, surf, N, V)


def apply_surface_effects(out, surf, N, V):
    """Fresnel, rim, matcap, reflection tint and backface override.

    These sit outside the reflectance model on purpose: they are the artistic
    cheats every package of the era offered on top of whichever shader you
    picked, so they behave the same on Lambert as on Cook-Torrance.
    """
    facing = np.clip(np.abs(M.dot(N, V)), 0.0, 1.0)
    edge = 1.0 - facing

    if np.any(surf.fresnel > 1e-4):
        f = np.power(edge, np.maximum(surf.fresnel_power, 0.01)) * surf.fresnel
        out = out + surf.fresnel_color * (f * surf.specular_level)[:, None]

    if np.any(surf.rim > 1e-4):
        r = np.power(edge, np.maximum(surf.rim_power, 0.01)) * surf.rim
        out = out + surf.rim_color * r[:, None]

    if np.any(surf.matcap_blend > 1e-4):
        k = np.clip(surf.matcap_blend, 0.0, 1.0)[:, None]
        out = out * (1.0 - k) + surf.matcap * k

    if np.any(surf.backface_mix > 1e-4) and surf.backfacing is not None:
        k = (np.clip(surf.backface_mix, 0.0, 1.0) *
             np.clip(surf.backfacing, 0.0, 1.0))[:, None]
        out = out * (1.0 - k) + surf.backface_color * k
    return out


def ambient_occlusion(P, N, bvh, settings, rng=None, sample_xy=None):
    """1 = open sky, falling toward 0 in creases, over `ao_distance`.

    With `sample_xy` (integer pixel identity), the cosine-weighted
    hemisphere directions are a pure function of (pixel, sample, seed):
    the same picture whatever the batch order or thread count, and
    exactly reproducible by the deferred pass -- hash draws for the
    radius, the shared unit-circle table for the angle (a driver rounds
    sin/cos differently, and an occlusion ray is a cliff). Without an
    identity, the legacy sequential stream still runs.
    """
    n = P.shape[0]
    samples = max(int(settings.ao_samples), 1)
    t, b = M.orthonormal_basis(N)
    occ = np.zeros(n, np.float32)
    dist = float(settings.ao_distance)
    origin = P + N * max(settings.ray_bias, 1e-4)
    if sample_xy is not None:
        from . import patterns as PT
        spx, spy = sample_xy
        seed = int(getattr(settings, 'seed', 0) or 0)
        for k in range(samples):
            z = 2 * k + 8389 + 7919 * seed
            u1 = PT.sample_u(spx, spy, z)
            ca, sa = PT.sample_circle(PT.sample_u(spx, spy, z + 1))
            r = np.sqrt(u1)
            d = M.normalize(t * (r * ca)[:, None] +
                            b * (r * sa)[:, None] +
                            N * np.sqrt(np.maximum(np.float32(1.0) - u1,
                                                   np.float32(0.0)))[:, None])
            occ += bvh.occluded(origin, d,
                                np.full(n, dist, np.float32)).astype(
                                    np.float32)
        ao = 1.0 - (occ / samples) * float(settings.ao_intensity)
        return np.clip(ao, 0.0, 1.0)
    rng = rng or np.random.default_rng(settings.seed)
    for _ in range(samples):
        u1 = rng.random(n).astype(np.float32)
        u2 = rng.random(n).astype(np.float32)
        r = np.sqrt(u1)
        th = 2.0 * np.pi * u2
        d = M.normalize(t * (r * np.cos(th))[:, None] +
                        b * (r * np.sin(th))[:, None] +
                        N * np.sqrt(np.maximum(1.0 - u1, 0.0))[:, None])
        occ += bvh.occluded(origin, d, np.full(n, dist, np.float32)).astype(np.float32)
    ao = 1.0 - (occ / samples) * float(settings.ao_intensity)
    return np.clip(ao, 0.0, 1.0)


def shade_closure_flat(cl, ctx):
    """Shader-to-RGB: evaluate a closure as a colour. Legal for a rasteriser."""
    scene = getattr(ctx, 'scene', None)
    settings = ctx.settings
    if scene is None or settings is None:
        col = np.zeros((ctx.n, 4), np.float32)
        col[:, 3] = 1.0
        return col
    surf, model, nrm = closure_to_surface(cl, ctx, settings)
    if nrm is not None:
        ctx = _with_normal(ctx, nrm)
    rgb = light_surface(surf, model, ctx, scene, settings,
                        getattr(ctx, 'bvh', None))
    return np.concatenate([rgb, surf.opacity[:, None]], axis=1).astype(np.float32)


def _with_normal(ctx, nrm):
    import copy
    c = copy.copy(ctx)
    c.N = M.normalize(nrm)
    return c


# ---------------------------------------------------------- world / horizon


def world_color(scene, settings, dirs, textures, n=None, eye=None):
    """Background radiance along `dirs` (N,3)."""
    world = scene.world
    n = dirs.shape[0] if n is None else n
    if world is None:
        return np.zeros((n, 3), np.float32)

    # An explicit sky mode wins over the node tree. Blender worlds always have a
    # node tree, so without this the Halcyon sky settings could never take
    # effect -- which is exactly what "the sky doesn't work" looked like.
    from . import sky as SKY
    time = getattr(scene, 'time', 0.0)

    def _ground(col):
        # applied after whichever sky mode ran, node tree included: an infinite
        # floor has nothing to do with how the sky above it is coloured
        if getattr(world, 'ground_plane', False) and eye is not None:
            return SKY.ground_plane(world, dirs, col.astype(np.float32), eye,
                                    time, textures)
        return col

    chosen = SKY.evaluate(world, dirs, textures, eye=eye, time=time)
    if chosen is not None:
        return chosen            # evaluate() already applied the ground

    if world.graph:
        ctx = ShadeContext(n)
        ctx.I = M.normalize(dirs)
        ctx.N = -ctx.I
        ctx.P = M.normalize(dirs) * 1e6
        ctx.generated = M.normalize(dirs)
        ctx.settings = settings
        ev = GraphEvaluator(world.graph, ctx, textures, {})
        cl, _ = ev.evaluate_surface()
        if isinstance(cl, Closure) and cl.items:
            acc = np.zeros((n, 3), np.float32)
            for kind, w, p in cl.items:
                col = to_color(p.get('color'), n)[:, :3]
                st = p.get('strength')
                s = to_value(st, n)[:, None] if st is not None else 1.0
                wv = np.asarray(w, np.float32).reshape(-1)
                if wv.shape[0] != n:
                    wv = np.broadcast_to(wv, (n,))
                acc += col * s * wv[:, None]
            return _ground(acc)
    if world.env_image is not None:
        tex = (textures.get(getattr(world.env_image, 'name', None)) or
               textures.get('world_env'))
        if tex is not None:
            from .texture import env_equirect_uv, env_sphere_uv
            d = M.normalize(dirs)
            u, v = (env_sphere_uv(d) if world.env_mapping == 'MIRRORBALL'
                    else env_equirect_uv(d))
            return _ground(tex.sample(u, v, filt='BILINEAR',
                                      wrap='EXTEND')[:, :3])
    if world.sky_blend:
        d = M.normalize(dirs)
        t = np.clip(d[:, 2] * 0.5 + 0.5, 0.0, 1.0)[:, None]
        hor = np.asarray(world.horizon, np.float32)[None, :]
        zen = np.asarray(world.zenith, np.float32)[None, :]
        return _ground((hor + (zen - hor) * t).astype(np.float32))
    return _ground(np.broadcast_to(np.asarray(world.color, np.float32)[None, :],
                                   (n, 3)).copy())


# ------------------------------------------------------------------- fog


def apply_fog(rgb, depth, settings, scene, vertex_rate=False):
    """Distance fog. `fog_vertex` evaluates it per vertex and interpolates,
    which is how fixed-function hardware did it -- and it shows, because the
    fog band follows the tessellation rather than the surface."""
    if not settings.fog:
        return rgb
    if settings.fog_vertex and not vertex_rate:
        # quantise the depth so the blend steps rather than sweeps, which is
        # what interpolating a per-vertex factor looks like on coarse geometry
        depth = np.round(depth * 8.0) / 8.0
    mode = settings.fog_mode
    d = np.maximum(depth, 0.0)
    if mode == 'LINEAR':
        f = (settings.fog_end - d) / max(settings.fog_end - settings.fog_start, 1e-5)
    elif mode == 'EXP':
        f = np.exp(-settings.fog_density * d)
    elif mode == 'EXP2':
        f = np.exp(-((settings.fog_density * d) ** 2))
    else:                                   # TABLE16 -- the fixed-function LUT
        t = np.clip((d - settings.fog_start) /
                    max(settings.fog_end - settings.fog_start, 1e-5), 0.0, 1.0)
        f = 1.0 - np.floor(t * 16.0) / 16.0
    f = np.clip(f, 0.0, 1.0)[:, None]
    col = np.asarray(settings.fog_color, np.float32)[None, :]
    return (rgb * f + col * (1.0 - f)).astype(np.float32)


# --------------------------------------------------------- fragment shading


class ShadeJob:
    """Everything needed to shade an arbitrary set of surface points."""

    def __init__(self, scene, settings, textures, bvh, view, eye, width, height):
        self.scene = scene
        self.settings = settings
        self.textures = textures
        self.bvh = bvh
        self.view = view
        self.eye = eye
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(settings.seed)
        self.lights = LI.select_lights(scene.lights, settings)
        self.unsupported = set()
        self._obj_matrices = None
        self._bounds = None
        self._obj_bounds = None
        #: (mat_index, bump-node id) -> (gx, gy) full-frame gradient grids,
        #: filled by _shade_all for materials whose chunking would otherwise
        #: cut n_bump's screen gradients mid-material
        self.bump_fields = {}

    def object_bounds(self):
        """Per-object bounding boxes, for Generated texture coordinates.

        Blender normalises Generated coordinates over each object's own bounding
        box. Normalising over the whole scene instead makes every procedural
        texture on a normal-sized object sample a tiny patch of its own space
        and come out flat -- which is exactly what a big ground plane in the
        scene used to do to everything else.
        """
        if self._obj_bounds is not None:
            return self._obj_bounds
        mesh = self.scene.mesh
        n_obj = max(len(self.scene.objects), 1)
        lo = np.zeros((n_obj, 3), np.float32)
        hi = np.ones((n_obj, 3), np.float32)
        if mesh is not None and mesh.verts is not None and mesh.verts.size:
            if mesh.obj_index is not None and mesh.tris is not None:
                # a vertex belongs to whichever object owns its triangles
                vert_obj = np.zeros(mesh.verts.shape[0], np.int32)
                vert_obj[mesh.tris.reshape(-1)] = np.repeat(mesh.obj_index, 3)
                for i in range(n_obj):
                    sel = vert_obj == i
                    if sel.any():
                        lo[i] = mesh.verts[sel].min(0)
                        hi[i] = mesh.verts[sel].max(0)
            else:
                lo[:] = mesh.verts.min(0)
                hi[:] = mesh.verts.max(0)
        self._obj_bounds = (lo, np.maximum(hi - lo, 1e-6))
        return self._obj_bounds

    def prewarm(self):
        """Build the lazily-cached tables before any worker thread starts.

        Populating them from several threads at once is a benign race in CPython
        but wastes the work; doing it up front also keeps the workers pure
        readers of shared state, which is what keeps the split safe.
        """
        mesh = self.scene.mesh
        if mesh is not None and mesh.verts is not None and mesh.verts.size:
            if self._bounds is None:
                self._bounds = (mesh.verts.min(0), mesh.verts.max(0))
        self.object_bounds()
        if self._obj_matrices is None and self.scene.objects:
            mats = []
            for o in self.scene.objects:
                m = o.matrix_world
                mats.append(np.linalg.inv(np.asarray(m, np.float32))
                            if m is not None else np.eye(4, np.float32))
            self._obj_matrices = np.stack(mats)

    # ..................................................... attribute fetch
    def attributes(self, tri_idx, bary, bary_lin=None):
        mesh = self.scene.mesh
        tris = mesh.tris
        P = raster.fetch(mesh.verts, tris, tri_idx, bary)
        smooth = mesh.smooth[tri_idx] if mesh.smooth is not None else None
        if mesh.normals is not None:
            Ns = raster.fetch(mesh.normals, tris, tri_idx, bary)
        else:
            Ns = mesh.face_normals[tri_idx]
        Ng = mesh.face_normals[tri_idx] if mesh.face_normals is not None else Ns
        if smooth is not None:
            Ns = np.where(smooth[:, None], Ns, Ng)
        ub = bary_lin if (bary_lin is not None and not self.settings.tex_perspective) \
            else bary
        uv = raster.fetch(mesh.uvs, tris, tri_idx, ub) if mesh.uvs is not None \
            else np.zeros((tri_idx.size, 2), np.float32)
        uv2 = raster.fetch(mesh.uvs2, tris, tri_idx, ub) \
            if getattr(mesh, 'uvs2', None) is not None else uv
        col = raster.fetch(mesh.colors, tris, tri_idx, bary) \
            if mesh.colors is not None else np.ones((tri_idx.size, 4), np.float32)
        return P, M.normalize(Ns), M.normalize(Ng), uv, uv2, col

    def context(self, tri_idx, bary, px=None, py=None, front=None, bary_lin=None,
                ray_depth=0, is_camera=True):
        mesh = self.scene.mesh
        n = tri_idx.size
        P, Ns, Ng, uv, uv2, col = self.attributes(tri_idx, bary, bary_lin)
        ctx = ShadeContext(n)
        ctx.P = P
        ctx.N = Ns
        ctx.Ng = Ng
        ctx.uv = uv
        ctx.uv2 = uv2
        ctx.vcol = col
        ctx.I = M.normalize(P - self.eye[None, :])
        ctx.px = px
        ctx.py = py
        ctx.width = self.width
        ctx.height = self.height
        ctx.tri = tri_idx
        ctx.camera_pos = self.eye
        ctx.settings = self.settings
        ctx.time = self.scene.time
        ctx.frame = self.scene.frame
        ctx.ray_depth = ray_depth
        ctx.is_camera_ray = is_camera
        ctx.scene = self.scene
        ctx.bvh = self.bvh
        if front is not None:
            ctx.backfacing = (~front).astype(np.float32)
        view_p = (P - self.eye[None, :]) @ self.view[:3, :3].T
        ctx.depth = np.abs(view_p[:, 2]).astype(np.float32)
        obj_idx = mesh.obj_index[tri_idx] if mesh.obj_index is not None else \
            np.zeros(n, np.int32)
        self._fill_object(ctx, obj_idx)
        ctx.object_index_raw = obj_idx
        lo, span = self.object_bounds()
        oi_c = np.clip(obj_idx, 0, lo.shape[0] - 1)
        ctx.generated = ((P - lo[oi_c]) / span[oi_c]).astype(np.float32)
        ctx.random = _hash1(tri_idx.astype(np.float32))
        ctx.bump_fields = getattr(self, 'bump_fields', None)
        # the deterministic-sampling identity: the screen pixel where one
        # exists. Traced hits overwrite these with the pixel that spawned
        # their ray (ShadeJob.shade's sample_xy), so a hit's soft shadows
        # and AO draw the same streams either device would.
        ctx.spx = np.asarray(px, np.int64) if px is not None else None
        ctx.spy = np.asarray(py, np.int64) if py is not None else None
        return ctx

    def _fill_object(self, ctx, obj_idx):
        objs = self.scene.objects
        if not objs:
            return
        n = ctx.n
        locs = np.array([o.location for o in objs], np.float32)
        cols = np.array([o.color for o in objs], np.float32)
        idxs = np.array([o.index for o in objs], np.float32)
        rnds = np.array([o.random for o in objs], np.float32)
        oi = np.clip(obj_idx, 0, len(objs) - 1)
        ctx.object_loc = locs[oi]
        ctx.object_color = cols[oi]
        ctx.object_index = idxs[oi]
        ctx.object_random = rnds[oi]
        if self._obj_matrices is None:
            mats = []
            for o in objs:
                m = o.matrix_world
                mats.append(np.linalg.inv(np.asarray(m, np.float32))
                            if m is not None else np.eye(4, np.float32))
            self._obj_matrices = np.stack(mats)
        ctx._obj_mats = self._obj_matrices
        ctx._obj_idx = oi

    # .......................................................... the shading
    def shade(self, tri_idx, bary, px=None, py=None, front=None, bary_lin=None,
              ray_depth=0, is_camera=True, rng=None, sample_xy=None):
        """RGBA for a set of surface samples, batched by material.

        `sample_xy` is (spx, spy): the SAMPLING identity for surface
        points that have no screen pixel of their own -- a traced hit
        carries the pixel that spawned its ray, so its soft shadows and
        ambient occlusion draw the same deterministic streams the
        primary surface drew.
        """
        n = tri_idx.size
        out = np.zeros((n, 4), np.float32)
        if n == 0:
            return out
        mesh = self.scene.mesh
        mat_idx = mesh.mat_index[tri_idx] if mesh.mat_index is not None else \
            np.zeros(n, np.int32)
        spx, spy = sample_xy if sample_xy is not None else (None, None)
        for mi in np.unique(mat_idx):
            sel = np.nonzero(mat_idx == mi)[0]
            mat = self.scene.materials[int(mi)] if int(mi) < len(self.scene.materials) \
                else None
            sub = self.context(tri_idx[sel], bary[sel],
                               px[sel] if px is not None else None,
                               py[sel] if py is not None else None,
                               front[sel] if front is not None else None,
                               bary_lin[sel] if bary_lin is not None else None,
                               ray_depth, is_camera)
            if spx is not None:
                sub.spx = np.asarray(spx, np.int64)[sel]
                sub.spy = np.asarray(spy, np.int64)[sel]
            out[sel] = self.shade_batch(sub, mat, ray_depth, rng)
        return out

    def shade_batch(self, ctx, mat, ray_depth=0, rng=None):
        st = self.settings
        n = ctx.n
        cl = None
        discard = None
        disp = None
        if mat is not None and mat.graph:
            ev = GraphEvaluator(mat.graph, ctx, self.textures, mat.programs)
            cl, disp = ev.evaluate_surface()
            self.unsupported.update(ev.unsupported)
            discard = ev.cache.get('__discard')
        if mat is not None and mat.shadeless:
            surf, model, nrm = closure_to_surface(cl, ctx, st, mat)
            rgb = surf.diffuse * surf.diffuse_level[:, None] + surf.emission
            return np.concatenate([rgb, surf.opacity[:, None]], 1).astype(np.float32)

        surf, model, nrm = closure_to_surface(cl, ctx, st, mat)
        # Gouraud/flat split (see _shade_interpolated): hardware of the
        # period interpolated the LIGHTING between vertices and still
        # sampled the TEXTURE at every pixel -- `texel x vertex colour`,
        # the MODULATE combiner. ALBEDO returns the per-pixel half,
        # LIGHT shades the lighting half over a white surface so the two
        # multiply back together with the texture at full resolution.
        rate_mode = getattr(self, 'rate_mode', None)
        if rate_mode == 'ALBEDO':
            alb = np.asarray(surf.diffuse, np.float32)
            return np.concatenate(
                [alb, np.clip(surf.opacity, 0.0, 1.0)[:, None]],
                axis=1).astype(np.float32)
        if rate_mode == 'LIGHT':
            surf.diffuse = np.ones_like(np.asarray(surf.diffuse, np.float32))
        if nrm is not None:
            ctx.N = M.normalize(nrm)
        if st.normal_source == 'FACE':
            ctx.N = ctx.Ng
        if disp is not None and st.displacement_scale > 0.0:
            d = np.asarray(disp, np.float32)
            h = d if d.ndim == 1 else (d[:, 2] if d.shape[1] >= 3
                                       else d.mean(axis=1))
            bumped = bump_from_height(ctx, h, st.displacement_scale)
            if bumped is not None:
                ctx.N = bumped
                surf.tangent, surf.bitangent = M.orthonormal_basis(ctx.N)
        surf.tangent, surf.bitangent = M.orthonormal_basis(ctx.N)
        # the context knows which fragments face away; the surface never did,
        # so the backface override had nothing to key off
        if getattr(ctx, 'backfacing', None) is not None:
            surf.backfacing = np.asarray(ctx.backfacing, np.float32)
        rgb = light_surface(surf, model, ctx, self.scene, st, self.bvh,
                            rng if rng is not None else self.rng, self.lights)

        if st.raytrace and ray_depth < st.ray_depth:
            rgb = self._add_raytraced(rgb, surf, ctx, ray_depth)
        elif st.env_reflection and np.any(surf.reflect > 1e-4):
            V = -M.normalize(ctx.I)
            R = M.reflect(-V, M.normalize(ctx.N))
            env = world_color(self.scene, st, R, self.textures, n,
                              eye=self.eye)
            rgb = rgb + env * surf.reflect[:, None] * surf.specular * \
                surf.reflect_color

        rgb = apply_fog(rgb, ctx.depth, st, self.scene)
        alpha = np.clip(surf.opacity, 0.0, 1.0)
        if st.alpha_threshold > 0.0:
            # a hard cutoff rather than a blend: cheaper, and what hardware
            # without an alpha unit actually did with cut-out textures
            alpha = np.where(alpha >= st.alpha_threshold, alpha, 0.0)
        if np.any(np.abs(surf.edge_opacity - 1.0) > 1e-4):
            facing = np.clip(np.abs(M.dot(M.normalize(ctx.N),
                                          -M.normalize(ctx.I))), 0.0, 1.0)
            t = np.power(1.0 - facing, np.maximum(surf.fresnel_power, 0.01))
            alpha = np.clip(alpha * (1.0 - t) + surf.edge_opacity * t, 0.0, 1.0)
        if st.transparency == 'STIPPLE' and ctx.px is not None:
            # keep or drop each pixel outright against an ordered threshold --
            # no blending, exactly as hardware without an alpha unit managed it
            from .dither import threshold_map
            tm = threshold_map(st.stipple_pattern
                               if st.stipple_pattern != 'NONE' else 'BAYER4',
                               64, 64)
            thr = tm[np.asarray(ctx.py) % tm.shape[0],
                     np.asarray(ctx.px) % tm.shape[1]]
            alpha = np.where(alpha > np.asarray(thr, np.float32), 1.0, 0.0)
        elif st.transparency == 'NONE':
            alpha = np.ones_like(alpha)
        if discard is not None:
            alpha = np.where(np.asarray(discard).reshape(-1)[:alpha.size], 0.0, alpha)
        return np.concatenate([rgb, alpha[:, None]], axis=1).astype(np.float32)

    def _add_raytraced(self, rgb, surf, ctx, ray_depth):
        """Secondary rays, traced only for the fragments that actually want them.

        This used to test `np.any(...)` and then trace for the whole batch: one
        transparent fragment anywhere in a chunk of a quarter of a million meant
        a refraction ray for every one of them, recursively to the ray depth. A
        scene with a single sheet of glass paid for refracting the entire frame.
        """
        st = self.settings
        if self.bvh is None:
            return rgb
        N = M.normalize(ctx.N)
        V = -M.normalize(ctx.I)
        spx = getattr(ctx, 'spx', None)
        spy = getattr(ctx, 'spy', None)

        def _sxy(sel):
            if spx is None or spy is None:
                return None
            return (np.asarray(spx, np.int64)[sel],
                    np.asarray(spy, np.int64)[sel])

        if st.ray_reflection:
            want = np.nonzero(surf.reflect > 1e-4)[0]
            if want.size:
                R = M.reflect(-V[want], N[want])
                hit = self.trace(ctx.P[want] + N[want] * st.ray_bias, R,
                                 ray_depth + 1, sample_xy=_sxy(want))
                rgb[want] += (hit * surf.reflect[want, None] *
                              surf.specular[want] * surf.reflect_color[want])

        if st.ray_refraction:
            want = np.nonzero(surf.opacity < 0.999)[0]
            if want.size:
                Nw, Vw = N[want], V[want]
                ior = np.maximum(surf.ior[want], 1e-3)
                eta = np.where(M.dot(Nw, Vw) < 0, ior, 1.0 / ior)
                T = M.refract(-Vw, Nw, eta)
                bad = (T * T).sum(1) < 1e-9
                T = np.where(bad[:, None], M.reflect(-Vw, Nw), T)
                hit = self.trace(ctx.P[want] - Nw * st.ray_bias, T,
                                 ray_depth + 1, sample_xy=_sxy(want))
                k = ((1.0 - np.clip(surf.opacity[want], 0.0, 1.0)) *
                     np.clip(surf.refraction[want], 0.0, 1.0))[:, None]
                rgb[want] = rgb[want] * (1.0 - k) + hit * k * surf.diffuse[want]
        return rgb

    def trace(self, origin, dirs, ray_depth, sample_xy=None):
        """Shade whatever a secondary ray hits (background if nothing).

        `sample_xy` carries the spawning pixels' identity, so the hit's
        deterministic sampling (soft shadows, AO) follows the ray."""
        n = origin.shape[0]
        if self.bvh is None or ray_depth > self.settings.ray_depth:
            return world_color(self.scene, self.settings, dirs, self.textures, n)
        tmax = np.full(n, 1e30, np.float32)
        tid, t, u, v = self.bvh.intersect(origin, dirs, tmax)
        hit = tid >= 0
        out = np.zeros((n, 3), np.float32)
        miss = np.nonzero(~hit)[0]
        if miss.size:
            out[miss] = world_color(self.scene, self.settings, dirs[miss],
                                    self.textures, miss.size, eye=self.eye)
        if not np.any(hit):
            return out
        idx = np.nonzero(hit)[0]
        bary = np.stack([1.0 - u[idx] - v[idx], u[idx], v[idx]], axis=1).astype(np.float32)
        col = self.shade(tid[idx].astype(np.int32), bary, ray_depth=ray_depth,
                         is_camera=False,
                         sample_xy=(sample_xy[0][idx], sample_xy[1][idx])
                         if sample_xy is not None else None)
        out[idx] = col[:, :3]
        return out


def bump_from_height(ctx, height, strength=1.0):
    """Perturb the normal from a height field, using screen derivatives.

    The node graph's Displacement output was computed and thrown away. Actually
    displacing geometry means tessellating it, which no 1990s scanline renderer
    did either -- they turned the height into a normal perturbation and called
    it bump mapping, which is what this does.
    """
    if ctx.px is None or ctx.py is None or ctx.width is None:
        return None
    h = np.asarray(height, np.float32).reshape(-1)
    if h.size != ctx.n or float(h.std()) < 1e-9:
        return None
    px = np.asarray(ctx.px, np.int64)
    py = np.asarray(ctx.py, np.int64)
    w, hh = int(ctx.width), int(ctx.height)
    grid = np.zeros((hh, w), np.float32)
    seen = np.zeros((hh, w), bool)
    grid[py, px] = h
    seen[py, px] = True
    # one-sided differences where the neighbour is missing, so silhouettes do
    # not invent a gradient out of empty space
    right = np.zeros_like(grid)
    right[:, :-1] = grid[:, 1:]
    ok_r = np.zeros_like(seen)
    ok_r[:, :-1] = seen[:, 1:]
    up = np.zeros_like(grid)
    up[:-1, :] = grid[1:, :]
    ok_u = np.zeros_like(seen)
    ok_u[:-1, :] = seen[1:, :]
    dx = np.where(ok_r, right - grid, 0.0)[py, px]
    dy = np.where(ok_u, up - grid, 0.0)[py, px]

    N = M.normalize(ctx.N)
    t, b = M.orthonormal_basis(N)
    k = float(strength) * 8.0
    return M.normalize(N - t * (dx * k)[:, None] - b * (dy * k)[:, None])


def _hash1(x):
    h = np.sin(x * 12.9898) * 43758.5453
    return (h - np.floor(h)).astype(np.float32)


# ---------------------------------------------------- shading-rate dispatch


def shade_vertex_rate(job, tri_subset, rate, st=None):
    """Gouraud / flat: shade at vertices or face centres, interpolate after.

    Historically these are *shading rates*, not reflectance models -- the same
    Phong maths evaluated less often. Doing it properly is what gives the
    faceted, banded look rather than a fake approximation of it.
    """
    mesh = job.scene.mesh
    tris = mesh.tris[tri_subset]
    if rate == 'FACE':
        bary = np.full((tris.shape[0], 3), 1.0 / 3.0, np.float32)
        idx = tri_subset.astype(np.int32)
        col = (_shade_chunked(job, idx, bary, None, None, None, None, st)
               if st is not None else job.shade(idx, bary))
        return col, None
    verts = np.unique(tris.reshape(-1))
    lookup = np.full(mesh.verts.shape[0], -1, np.int64)
    lookup[verts] = np.arange(verts.size)
    owner = np.zeros(verts.size, np.int32)
    corner = np.zeros(verts.size, np.int32)
    for c in range(3):
        vi = lookup[tris[:, c]]
        owner[vi] = tri_subset
        corner[vi] = c
    bary = np.zeros((verts.size, 3), np.float32)
    bary[np.arange(verts.size), corner] = 1.0
    col = (_shade_chunked(job, owner, bary, None, None, None, None, st)
           if st is not None else job.shade(owner, bary))
    return col, lookup


# --------------------------------------------------------------- main entry


def _build_shadows(scene, st, mesh):
    if not (st.shadows and st.shadow_default in ('MAP', 'PER_LIGHT')):
        return
    cast = None
    if mesh is not None and mesh.obj_index is not None and scene.objects:
        keep = np.array([o.cast_shadow for o in scene.objects], bool)
        if not keep.all():
            cast = np.nonzero(keep[np.clip(mesh.obj_index, 0,
                                           len(scene.objects) - 1)])[0]
    LI.build_shadow_maps(scene, st, cast)


def shaft_sources(scene, settings, vp):
    """Screen positions of lights that scatter, for the light-shaft pass."""
    out = []
    for light in scene.lights:
        vol = float(getattr(light, 'volumetric', 0.0))
        if vol <= 0.0:
            continue
        if light.type == 'SUN':
            # a directional light has no position: its shafts converge on the
            # vanishing point, which is the direction projected with w = 0
            d = M.normalize(np.asarray(light.direction, np.float32))
            clip = np.append(-d, 0.0).astype(np.float32) @ vp.T
        else:
            pos = np.asarray(light.position, np.float32)
            clip = np.append(pos, 1.0).astype(np.float32) @ vp.T
        if clip[3] <= 1e-6:
            continue                       # behind the camera
        ndc = clip[:3] / clip[3]
        # a source outside the frame still throws shafts into it, so the bound
        # is generous; only rule out lights nowhere near the view
        if abs(ndc[0]) > 6.0 or abs(ndc[1]) > 6.0:
            continue
        out.append(((float(ndc[0]), float(ndc[1])), vol))
    return out


def render(scene, settings=None, progress=None, band=None):
    """Render `scene`. Returns a linear (H,W,4) float32 image.

    `band` is an optional (y0, y1) range of output rows; only those rows are
    shaded and returned. Used by the worker pool to split a frame across
    processes.

    Row 0 is the BOTTOM of the picture: the rasteriser maps NDC y = -1 to row 0.
    That matches Blender's render-result buffer, so the engine hands it over
    without flipping. Anything writing a PNG (PIL, most image libraries) treats
    row 0 as the top and must flip first.
    """
    st = settings or scene.settings
    mesh = scene.mesh
    W = max(int(st.resolution_x), 1)
    H = max(int(st.resolution_y), 1)
    ss = 1
    if st.aa_mode == 'SUPERSAMPLE':
        ss = max(int(np.round(np.sqrt(max(st.aa_samples, 1)))), 1)
    rw, rh = W * ss, H * ss

    view, proj, vp, eye = camera_matrices(scene.camera, rw, rh)
    # the water needs to know how big a pixel is before it can decide which
    # waves are too small to draw
    if getattr(scene, 'world', None) is not None:
        try:
            pa, pw = pixel_footprint(scene.camera, proj, rh)
            scene.world._pixel_angle = pa
            scene.world._pixel_width = pw
        except Exception:                                       # noqa: BLE001
            pass
    with ST.track('prepare textures'):
        textures = prepare_textures(scene, st)

    need_bvh = st.raytrace or st.ambient_occlusion or \
        (st.shadows and st.shadow_default == 'RAY')
    bvh = None
    if need_bvh and mesh is not None and mesh.tris is not None and mesh.tris.size:
        with ST.track('build BVH'):
            bvh = _cached_bvh(scene, mesh)

    with ST.track('shadow maps'):
        _build_shadows(scene, st, mesh)
    if False:
        pass

    if progress:
        progress(0.05, 'Rasterising')

    gbuf = raster.GBuffer(rw, rh)
    if not st.tex_perspective:
        gbuf.alloc_linear()
    frags = raster.FragmentList() if st.transparency in ('SORTED', 'ABUFFER') else None

    job = ShadeJob(scene, st, textures, bvh, view, eye, rw, rh)

    if mesh is None or mesh.tris is None or mesh.tris.size == 0:
        img = _background_image(scene, st, rw, rh, vp, eye, None, textures)
        if band is not None:
            y0, y1 = max(band[0], 0), min(band[1], H)
            return _resolve(img[y0 * ss:y1 * ss], W, y1 - y0, ss, st)
        return _resolve(img, W, H, ss, st)

    opaque, transparent = _split_by_alpha(scene, mesh, st)
    snap = st.vertex_snap_grid if st.vertex_snap else 0.0
    cull = 'BACK' if st.backface_cull else 'NONE'

    # when only a band is wanted, the rasteriser is told so: otherwise every
    # worker in a pool rasterises the whole mesh for its own slice
    scissor = None
    has_bump = any('ShaderNodeBump' in
                   str((getattr(m, 'graph', None) or {}).get('nodes', {}))
                   for m in getattr(scene, 'materials', ()) or ())
    if band is not None:
        scissor = (max(band[0], 0) * ss, min(band[1], H) * ss)
        if has_bump:
            # one CONTEXT row past the band's top: n_bump differences
            # toward the +y neighbour, and without that row a band's
            # last shaded row flattened its waves -- the same seam the
            # chunk fix killed, band edition. The scissor culls whole
            # triangles, so one extra row keeps the neighbour coverage
            # complete; shading stays band-masked below.
            scissor = (scissor[0], min(scissor[1] + 1, rh))

    flat_depth = None
    if st.depth_sort == 'PAINTERS':
        flat_depth = polygon_depths(mesh, view, eye, st.painters_key)

    want_overdraw = st.debug_pass == 'OVERDRAW'
    # The compute rasteriser: the CPU's own fill rules on the GPU, measured
    # at ZERO differing pixels on hardware. Strictly opt-in and strictly
    # qualified -- anything it does not reproduce rasterises on the CPU
    # exactly as before, with the reason printed. Whole-frame only.
    rastered_on_gpu = False
    if str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
            getattr(st, 'gpu_raster', False) and band is None and \
            flat_depth is None and not want_overdraw:
        if not getattr(st, 'tex_perspective', True):
            print('[Halcyon GPU] rasterising on the CPU: affine texture '
                  'mode needs screen-linear barycentrics the compute '
                  'raster does not carry yet')
        else:
            from ..gpu import craster as _craster
            with ST.track('rasterise (GPU)'):
                # the GPU crossings live at the device boundary now
                # (gpu/device.py _main): the CPU halves of this call
                # never block the interface, the driver halves cross
                # as millisecond bursts
                try:
                    ok_r, why_r = _craster.raster_into_gbuffer(
                        mesh, vp, rw, rh, gbuf, cull=cull, snap=snap,
                        depth_bits=st.depth_precision, subset=opaque)
                except Exception as exc:                        # noqa: BLE001
                    ok_r, why_r = False, str(exc)
            if ok_r:
                rastered_on_gpu = True
            else:
                print(f'[Halcyon GPU] rasterising on the CPU: {why_r}')
    if not rastered_on_gpu:
        with ST.track('rasterise'):
            raster.rasterize(mesh.verts, mesh.tris, vp, rw, rh, cull=cull,
                             snap=snap, depth_bits=st.depth_precision,
                             subset=opaque, gbuf=gbuf,
                             count_overdraw=want_overdraw,
                             flat_depth=flat_depth, scissor=scissor,
                             batched=False if want_overdraw else None)

    if band is None:
        # the two numbers behind "the depth is screwed up": what the
        # z-buffer can resolve on THIS frame, and which surfaces were
        # taken out of the depth-buffered pass altogether
        _rep = depth_report(proj, gbuf, st.depth_precision,
                            getattr(st, 'depth_sort', 'ZBUFFER'))
        if _rep:
            print(_rep)
        _sp = LAST_SPLIT
        if _sp.get('see_through'):
            named = '; '.join(f'{n}: {w}' for n, w
                              in list(_sp['reasons'].items())[:6])
            more = len(_sp['reasons']) - 6
            print(f"[Halcyon] transparency: {_sp['see_through']} of "
                  f"{_sp['materials']} materials are see-through "
                  f"({_sp['tris_see_through']} of {_sp['tris']} "
                  f'triangles) -- {named}'
                  + (f' (+{more} more)' if more > 0 else ''))
            if opaque is not None and opaque.size == 0:
                print('[Halcyon] transparency: NOTHING is in the '
                      'depth-buffered pass -- every surface stacks as '
                      'A-buffer layers instead, so solid geometry can '
                      'show its own back faces. If these materials are '
                      'meant to be solid, set their blend mode to '
                      'Opaque (or Transparency to None) and the frame '
                      'goes back through the z-buffer')

    if progress:
        progress(0.35, 'Shading')

    covered = gbuf.mask()
    if band is not None:
        keep = np.zeros(rh, bool)
        keep[max(band[0], 0) * ss:min(band[1], H) * ss] = True
        covered &= keep[:, None]
    with ST.track('background / sky'):
        img = _background_image(scene, st, rw, rh, vp, eye,
                                (~covered) & (keep[:, None] if band is not None
                                              else True), textures, ss=ss)
    _spot_cones(img, scene, st, gbuf, vp, eye, rw, rh)

    py, px = np.nonzero(covered)
    if py.size:
        # Deferred GPU shading: the G-buffer just rasterised is shaded in a
        # full-screen pass per material, on the same mechanism the post
        # stages run on. Strictly opt-in, and strictly qualified -- a frame
        # using anything the GLSL does not reproduce shades on the CPU
        # exactly as before, with the reason printed rather than guessed
        # around. Whole-frame only: a worker band re-splits the arithmetic
        # the GPU would do in one pass anyway.
        shaded_on_gpu = False
        if str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
                st.gpu_shading and band is None:
            from ..gpu import shade as _gpu_shade
            with ST.track('shade (GPU)'):
                try:
                    got, why = _gpu_shade.shade_frame(job, gbuf)
                except Exception as exc:                        # noqa: BLE001
                    got, why = None, str(exc)
            if got is None:
                print(f'[Halcyon GPU] shading on the CPU: {why}')
            else:
                img[py, px, :3] = got[py, px]
                img[py, px, 3] = 1.0
                shaded_on_gpu = True
        if not shaded_on_gpu:
            tri_idx = gbuf.tri[py, px]
            bary = gbuf.bary[py, px]
            blin = gbuf.bary_lin[py, px] if gbuf.bary_lin is not None else None
            front = gbuf.front[py, px]
            if band is not None and has_bump:
                # a band shades only its rows, but n_bump's gradients
                # need the CONTEXT row the extended scissor rasterised:
                # hand the field builder the G-buffer's FULL coverage,
                # so a band's gradients equal the whole frame's
                fpy, fpx = np.nonzero(gbuf.mask())
                job.bump_field_source = (
                    gbuf.tri[fpy, fpx], gbuf.bary[fpy, fpx], fpx, fpy,
                    gbuf.front[fpy, fpx],
                    gbuf.bary_lin[fpy, fpx]
                    if gbuf.bary_lin is not None else None)
            with ST.track('shade'):
                col = _shade_all(job, tri_idx, bary, px, py, front, blin, st,
                                 progress=progress)
            img[py, px, :3] = col[:, :3]
            img[py, px, 3] = np.maximum(img[py, px, 3], col[:, 3])

    with ST.track('wireframe'):
        img = apply_wireframe(job, gbuf, img, st, vp, eye, textures)

    if transparent is not None and transparent.size and frags is not None:
        if progress:
            progress(0.65, 'Transparency')
        raster.rasterize(mesh.verts, mesh.tris, vp, rw, rh, cull='NONE', snap=snap,
                         depth_bits=st.depth_precision, subset=transparent,
                         gbuf=gbuf, frags=frags, depth_write=False,
                         flat_depth=flat_depth, scissor=scissor)
        with ST.track('transparency'):
            img = _composite_abuffer(job, frags, gbuf, img, st, band=band)
    elif transparent is not None and transparent.size:
        raster.rasterize(mesh.verts, mesh.tris, vp, rw, rh, cull=cull, snap=snap,
                         depth_bits=st.depth_precision, subset=transparent,
                         gbuf=gbuf, flat_depth=flat_depth, scissor=scissor)

    if st.debug_pass != 'BEAUTY':
        img = _debug_pass(job, gbuf, img, st)

    if progress:
        progress(0.85, 'Resolving')
    # hand the post chain what it needs for defocus and shafts
    # Extra passes come off the same G-buffer the beauty image did, before
    # anything downsamples or quantises it.
    depth_m = None
    if st.dof or 'Depth' in wanted_passes(st):
        with ST.track('linear depth'):
            depth_m = linear_depth(job, gbuf, eye)
    scene.last_passes = build_aux_passes(job, gbuf, st, depth_m)
    if scene.last_passes and ss > 1:
        # data, not colour: averaging a normal or an object index across
        # samples produces a value that was never on any surface, so the
        # top-left sample of each output pixel is taken instead
        scene.last_passes = {k: v[::ss, ::ss]
                             for k, v in scene.last_passes.items()}

    scene.last_depth = None
    if st.dof and depth_m is not None:
        # metres, so Focus Distance in the UI is the distance it says it is.
        # It was normalised device depth, which put every focus value the
        # slider allows far behind the whole scene and blurred the lot.
        scene.last_depth = depth_m[::ss, ::ss] if ss > 1 else depth_m
    scene.last_shafts = shaft_sources(scene, st, vp)

    with ST.track('resolve / downsample'):
        if band is not None:
            y0 = max(band[0], 0)
            y1 = min(band[1], H)
            return _resolve(img[y0 * ss:y1 * ss], W, y1 - y0, ss, st)
        return _resolve(img, W, H, ss, st)


def material_model(mat, settings):
    """The model a material will resolve to, without evaluating its graph.

    Only needs to be right for materials that name a model explicitly; graphs
    whose model falls out of a closure fall back to the scene shading rate.
    """
    if settings.force_model != 'NONE':
        return settings.force_model
    if mat is None:
        return settings.default_model
    # A material-level override is authoritative even when a node tree exists,
    # because Blender materials always have one. Missing this meant a material
    # set to Wireframe in the Halcyon panel shaded as flat colour and never got
    # its edges carved out.
    if getattr(mat, 'use_override', False) and getattr(mat, 'model', None):
        return mat.model
    if getattr(mat, 'graph', None):
        for node in mat.graph.get('nodes', {}).values():
            if node.get('bl_idname') == 'HALCYON_ShaderNode':
                return node.get('props', {}).get('model', settings.default_model)
        return getattr(mat, 'model', None) if getattr(mat, 'model', None) else None
    return getattr(mat, 'model', None) or settings.default_model


def material_wire_size(mat, default=1.0):
    """The wire width for a material, from its node if it has one.

    Same reasoning as `material_model`: for a material shaded as Wireframe by a
    Halcyon Shader node, the node is where the user set it, so the node is
    where it is read from.
    """
    if mat is None:
        return default
    graph = getattr(mat, 'graph', None)
    if graph:
        for node in graph.get('nodes', {}).values():
            if node.get('bl_idname') == 'HALCYON_ShaderNode':
                v = node.get('props', {}).get('wire_size')
                if v is not None:
                    return max(float(v), 0.05)
    return max(float(getattr(mat, 'wire_size', default) or default), 0.05)


RATE_FOR_MODEL = {'GOURAUD': 'VERTEX', 'FLAT': 'FACE', 'WIREFRAME': 'PIXEL'}


def _cached_bvh(scene, mesh):
    """The scene's BVH, rebuilt only when the mesh content changed.

    A viewport orbit calls render() once per view of the SAME scene, and an
    animation renders the same mesh for most of its frames -- rebuilding the
    tree every call was a fifth of a second a frame on a real field scene.
    The cache lives on the scene object (a fresh export starts clean) behind
    a strided content fingerprint, the same idiom every GPU upload cache
    uses: identity is useless across exports, content is not.
    """
    v = mesh.verts
    stride = max(1, v.shape[0] // 512)
    key = (int(v.shape[0]), int(mesh.tris.shape[0]),
           round(float(v[::stride].sum()), 3),
           round(float(np.abs(v[::stride]).sum()), 3))
    hit = getattr(scene, '_bvh_cache', None)
    if hit is not None and hit[0] == key:
        return hit[1]
    bvh = BVH(mesh.verts, mesh.tris)
    try:
        scene._bvh_cache = (key, bvh)
    except Exception:                                           # noqa: BLE001
        pass                       # a scene that refuses attributes still renders
    return bvh


def _bump_height_fields(job, mat, mi, tri_m, bary_m, px_m, py_m, front_m,
                        blin_m, chunkn):
    """(mi, node_id) -> (gx, gy): whole-material bump gradients, pre-passed.

    `n_bump` differences its height chain toward the +x/+y neighbours,
    gated on the neighbour being shaded in the same batch -- so when a
    material is too covered for one batch, its gradients used to cut at
    every chunk boundary. This is the CPU's own height PRE-PASS, the
    same idea the deferred pass proved: evaluate the height chain over
    the material's FULL frame pixels (in bounded chunks -- heights are
    per-pixel independent, so chunking here cannot cut anything),
    scatter to a frame grid exactly as `_screen_grad` would, difference
    ONCE, and let every shading chunk gather its own pixels. Bitwise the
    same arithmetic as a whole-material batch: float32 grid, the same
    forward differences, the same validity, the same gather.
    """
    from .nodeeval import VALUE, GraphEvaluator
    graph = getattr(mat, 'graph', None)
    nodes = (graph or {}).get('nodes', {})
    bumps = [n for n in nodes.values()
             if n.get('bl_idname') == 'ShaderNodeBump']
    if not bumps:
        return {}
    H, W = job.height, job.width
    fields = {}
    for node in bumps:
        img = np.zeros((H, W), np.float32)
        valid = np.zeros((H, W), bool)
        for s in range(0, int(tri_m.size), int(chunkn)):
            e = min(s + int(chunkn), int(tri_m.size))
            ctx = job.context(tri_m[s:e], bary_m[s:e], px_m[s:e],
                              py_m[s:e], front_m[s:e]
                              if front_m is not None else None,
                              blin_m[s:e] if blin_m is not None else None,
                              0, True)
            ev = GraphEvaluator(graph, ctx, job.textures,
                                getattr(mat, 'programs', None))
            h = np.asarray(ev.input(node, 'Height', VALUE),
                           np.float32).reshape(-1)
            img[py_m[s:e], px_m[s:e]] = h
            valid[py_m[s:e], px_m[s:e]] = True
        gx = np.zeros_like(img)
        gy = np.zeros_like(img)
        gx[:, :-1] = np.where(valid[:, 1:] & valid[:, :-1],
                              img[:, 1:] - img[:, :-1], 0.0)
        gy[:-1, :] = np.where(valid[1:, :] & valid[:-1, :],
                              img[1:, :] - img[:-1, :], 0.0)
        fields[(int(mi), node.get('id'))] = (gx, gy)
    return fields


def _shade_all(job, tri_idx, bary, px, py, front, blin, st, progress=None):
    """Dispatch fragments to the right shading rate, per material.

    Gouraud and flat are shading *rates*, not reflectance models, so a material
    that asks for either is shaded at vertex or face frequency rather than
    having its lighting faked at pixel frequency.
    """
    scene = job.scene
    mesh = scene.mesh
    n = tri_idx.size
    out = np.zeros((n, 4), np.float32)
    mat_idx = mesh.mat_index[tri_idx] if mesh.mat_index is not None else \
        np.zeros(n, np.int32)
    rates = {}
    for i, mat in enumerate(scene.materials):
        model = material_model(mat, st)
        rates[i] = RATE_FOR_MODEL.get(model, st.shading_rate) if model \
            else st.shading_rate
    per_frag = np.array([rates.get(int(m), st.shading_rate) for m in
                         range(max(len(scene.materials), 1))], dtype=object)
    frag_rate = per_frag[np.clip(mat_idx, 0, per_frag.size - 1)]
    for rate in ('PIXEL', 'VERTEX', 'FACE'):
        sel = np.nonzero(frag_rate == rate)[0]
        if sel.size == 0:
            continue
        if rate == 'PIXEL':
            # ONE MATERIAL PER CHUNKED CALL. n_bump's neighbour validity
            # is "shaded in the same batch", so a chunk boundary used to
            # cut a material's screen gradients mid-frame: one row of the
            # field's water shaded with flattened waves where fragment
            # 79917 of a 480x360 frame happened to land -- the same row
            # on every machine whose settings chunked there, and a row
            # the GPU (whole-material pre-pass) correctly did NOT
            # flatten. Shading per material makes the gradients a
            # function of the picture, not of the chunk size: any
            # material that fits MAX_CHUNK shades in one batch, and the
            # deferred plan refuses Bump materials too covered to (they
            # would still cut). The memory bound chunking exists for is
            # untouched -- chunks are still capped, just never across a
            # material's interior unless the material alone exceeds the
            # cap.
            sel_mats = mat_idx[sel]
            done = 0
            total = int(sel.size)
            workers = resolve_threads(st)
            for m in np.unique(sel_mats):
                sm = sel[sel_mats == m]
                nm = int(sm.size)
                # a material too covered for ONE batch would have its
                # n_bump gradients cut at every chunk boundary -- so its
                # height chains render to whole-material gradient fields
                # FIRST (chunked themselves: heights are per-pixel
                # independent), and every shading chunk gathers from
                # those. Materials that fit one batch never pay this.
                mchunk = int(min(max(int(np.ceil(nm / max(workers * 4, 1))),
                                     MIN_CHUNK), MAX_CHUNK))
                mat = job.scene.materials[int(m)] \
                    if int(m) < len(job.scene.materials) else None
                src = getattr(job, 'bump_field_source', None)
                if src is not None and mat is not None \
                        and getattr(mat, 'graph', None) \
                        and int(m) not in {k[0] for k in job.bump_fields}:
                    # banded shading: the fields come from the G-buffer's
                    # FULL coverage (context row included), so the band's
                    # gradients equal the whole frame's
                    ftri, fbary, fpx, fpy, ffront, fblin = src
                    fm = job.scene.mesh.mat_index[ftri] \
                        if job.scene.mesh.mat_index is not None \
                        else np.zeros(ftri.size, np.int32)
                    fs = np.nonzero(fm == m)[0]
                    if fs.size:
                        fchunk = int(min(max(
                            int(np.ceil(fs.size / max(workers * 4, 1))),
                            MIN_CHUNK), MAX_CHUNK))
                        job.bump_fields.update(_bump_height_fields(
                            job, mat, int(m), ftri[fs], fbary[fs],
                            fpx[fs], fpy[fs],
                            ffront[fs] if ffront is not None else None,
                            fblin[fs] if fblin is not None else None,
                            fchunk))
                elif mchunk < nm and mat is not None \
                        and getattr(mat, 'graph', None):
                    job.bump_fields.update(_bump_height_fields(
                        job, mat, int(m), tri_idx[sm], bary[sm], px[sm],
                        py[sm], front[sm] if front is not None else None,
                        blin[sm] if blin is not None else None, mchunk))
                if progress is not None:
                    def _prog(v, msg, _d=done, _nm=nm):
                        # remap this material's local shade fraction into
                        # the frame's global one, keeping the bar (and the
                        # preview's abort ticks) monotonic
                        local = min(max((v - 0.35) / 0.30, 0.0), 1.0)
                        progress(0.35 + 0.30 * ((_d + local * _nm)
                                                / max(total, 1)), msg)
                else:
                    _prog = None
                out[sm] = _shade_chunked(job, tri_idx[sm], bary[sm],
                                         px[sm], py[sm], front[sm],
                                         blin[sm] if blin is not None
                                         else None,
                                         st, progress=_prog)
                done += nm
        else:
            out[sel] = _shade_interpolated(
                job, tri_idx[sel], bary[sel], rate, st,
                px[sel] if px is not None else None,
                py[sel] if py is not None else None,
                front[sel] if front is not None else None,
                blin[sel] if blin is not None else None)
    return out


MIN_CHUNK = 16384            # below this, per-chunk overhead dominates
SMALL_CHUNK = 2048           # ...but an idle core costs more than that overhead
MAX_CHUNK = 262144           # above this, one chunk's temporaries get large


def resolve_threads(settings):
    n = int(getattr(settings, 'threads', 0) or 0)
    if n <= 0:
        try:
            n = len(os.sched_getaffinity(0))
        except AttributeError:
            n = os.cpu_count() or 1
    return max(1, min(int(n), 64))


def _shade_chunked(job, tri_idx, bary, px, py, front, blin, st, progress=None):
    """Shade in bounded chunks, across threads when there are cores to use.

    `progress` ticks between chunks. Beyond a better progress bar, it is
    what makes a viewport render ABORTABLE mid-shade: the preview's tick
    raises when a newer view supersedes this one, and before this the
    abort could only land between whole stages -- which for shade meant
    after the expensive part had already run to completion.

    Deferred shading makes every fragment independent, so this is a clean split.
    The workers only touch bpy-free code and read-only shared state, and NumPy
    drops the GIL for the array work, so the threads do real parallel work
    rather than taking turns.

    Chunking matters on its own even single-threaded: a 3440x1440 frame at 4x
    supersampling is 79 million fragments, and building the full shading context
    for all of them at once would need tens of gigabytes. Bounded chunks make
    the peak memory a function of chunk size and thread count, not resolution.
    """
    n = int(tri_idx.size)
    out = np.zeros((n, 4), np.float32)
    if n == 0:
        return out

    def slice_of(a, s, e):
        return None if a is None else a[s:e]

    workers = resolve_threads(st)
    chunk = int(np.ceil(n / max(workers * 4, 1)))
    chunk = int(min(max(chunk, MIN_CHUNK), MAX_CHUNK))
    # Subdividing below this floor to give every worker a chunk was tried and
    # measured *worse* on a 20-core machine -- 1.00x became 0.87x -- because
    # the shading path is not actually thread-limited by chunk count. See the
    # threading note in the README: NumPy releases the interpreter lock only
    # for large operations, and the node evaluator is dominated by Python
    # dispatch between small ones. More chunks bought only more overhead.
    starts = list(range(0, n, chunk))

    if workers == 1 or len(starts) == 1:
        for s in starts:
            if progress and s:
                progress(0.35 + 0.30 * (s / n), 'Shading')
            e = min(s + chunk, n)
            out[s:e] = job.shade(tri_idx[s:e], bary[s:e], slice_of(px, s, e),
                                 slice_of(py, s, e), slice_of(front, s, e),
                                 slice_of(blin, s, e))
        return out

    # Threads are not permitted in a Blender extension, so chunks run in
    # sequence here and real parallelism comes from the worker processes
    # instead. The chunking still earns its place: it is what bounds peak
    # memory, which is otherwise a function of resolution.
    job.prewarm()
    seed = int(getattr(st, 'seed', 0) or 0)
    for i, s in enumerate(starts):
        if progress and s:
            progress(0.35 + 0.30 * (s / n), 'Shading')
        e = min(s + chunk, n)
        out[s:e] = job.shade(tri_idx[s:e], bary[s:e], slice_of(px, s, e),
                             slice_of(py, s, e), slice_of(front, s, e),
                             slice_of(blin, s, e),
                             rng=np.random.default_rng(seed + i * 7919))
    return out


def _shade_interpolated(job, tri_idx, bary, rate, st=None,
                        px=None, py=None, front=None, blin=None):
    """Shade at vertex or face rate, then interpolate to the fragments.

    THE LIGHTING is what these rates interpolate -- not the texture.
    Hardware that shaded per vertex still sampled the texture at every
    pixel and multiplied: `texel x vertex colour`, the MODULATE
    combiner every fixed-function pipeline of the era implemented, and
    the reason a Gouraud-shaded PlayStation model shows soft banded
    light over a SHARP texture. Halcyon evaluated the whole material
    at the vertices instead, texture included, so a textured model
    came out smeared across its own triangles -- the field photographed
    exactly that on a 1209-triangle character whose preset selects
    Gouraud. So: lighting over a WHITE surface at the vertex or face
    rate, interpolated; albedo and alpha at the PIXEL rate; multiplied.

    An untextured material is unaffected -- its albedo is one colour,
    so multiplying it back in reproduces the old result exactly. The
    banding and the missed highlights, which are the point of a
    shading RATE, are untouched: they live in the lighting term.

    The vertex and face passes went through job.shade() directly, which meant
    they never used the thread pool -- so every preset with a Gouraud or flat
    shading rate, which is most of the console and home-computer ones, rendered
    single-threaded no matter what the thread count said.
    """
    mesh = job.scene.mesh
    uniq = np.unique(tri_idx)
    saved = getattr(job, 'rate_mode', None)
    try:
        job.rate_mode = 'LIGHT'
        col, lookup = shade_vertex_rate(job, uniq, rate, st)
    finally:
        job.rate_mode = saved
    if rate == 'FACE':
        order = np.searchsorted(uniq, tri_idx)
        light = col[order]
    else:
        tris = mesh.tris[tri_idx]
        c0 = col[lookup[tris[:, 0]]]
        c1 = col[lookup[tris[:, 1]]]
        c2 = col[lookup[tris[:, 2]]]
        light = (c0 * bary[:, 0:1] + c1 * bary[:, 1:2]
                 + c2 * bary[:, 2:3]).astype(np.float32)
    try:
        job.rate_mode = 'ALBEDO'
        alb = _shade_chunked(job, tri_idx, bary, px, py, front, blin, st) \
            if st is not None else \
            job.shade(tri_idx, bary, px, py, front, blin)
    finally:
        job.rate_mode = saved
    out = np.empty_like(light)
    out[:, :3] = light[:, :3] * alb[:, :3]
    # alpha comes from the PIXEL pass: a cut-out texture's edge is the
    # one thing that must never be interpolated between vertices
    out[:, 3] = alb[:, 3]
    return out


def polygon_depths(mesh, view, eye, mode='CENTROID'):
    """One depth per triangle, for Painter's algorithm.

    Painter's does not compare fragments, it compares whole polygons: the one
    whose chosen depth is nearest wins the pixel outright. Giving every fragment
    of a triangle that single depth turns the existing z-buffer into exactly
    that comparison, and reproduces the algorithm's real failures -- surfaces
    that interpenetrate meet along a polygon edge instead of their true
    intersection, and a large polygon can be occluded by a small nearer one it
    actually passes in front of.
    """
    v = mesh.verts[mesh.tris]                       # (T, 3, 3)
    # depth of each corner, then reduce -- taking the min or max of the
    # positions instead would pick a corner per axis and mean nothing
    rel = v - eye[None, None, :]
    per_vertex = np.abs(rel @ view[:3, :3].T)[:, :, 2]
    if mode == 'NEAREST':
        depth = per_vertex.min(axis=1)
    elif mode == 'FARTHEST':
        depth = per_vertex.max(axis=1)
    else:
        depth = per_vertex.mean(axis=1)
    # matches the z-buffer's convention: smaller is nearer
    return depth.astype(np.float32)


#: how the last frame classified its materials: which went to the
#: A-buffer and WHY. A surface in the transparent pass is rasterised
#: with cull NONE and no depth write, so its back faces and everything
#: behind it stack as depth layers -- correct for glass, and a very
#: convincing "the depth is broken" for a solid character whose
#: materials were merely FLAGGED see-through. The printed line names
#: the flag, so the field never has to guess which pass a surface is in.
LAST_SPLIT = {}


def _split_by_alpha(scene, mesh, st=None):
    """Triangle indices for the opaque and the transparent passes.

    Opaque and Screen Door do not use a separate pass at all: their geometry
    belongs in the depth-buffered pass with everything else. Splitting it out
    and then never shading it is what made both modes render nothing.
    """
    LAST_SPLIT.clear()
    if mesh.mat_index is None:
        return None, np.zeros(0, np.int32)
    if st is not None and st.transparency in ('NONE', 'STIPPLE'):
        return None, np.zeros(0, np.int32)
    see_through = np.zeros(max(len(scene.materials), 1), bool)
    reasons = {}
    for i, m in enumerate(scene.materials):
        why = None
        if m.opacity < 0.999:
            why = f'Opacity {float(m.opacity):.3f}'
        elif getattr(m, 'has_alpha', False):
            # export.py's _alpha_reason names the specific evidence --
            # a transparent node, or which Alpha socket. A vague reason
            # is what let a whole mis-flagged character hide for three
            # rounds behind "flagged on export".
            why = str(getattr(m, 'alpha_why', None)
                      or 'flagged see-through on export')
        see_through[i] = why is not None
        if why is not None:
            reasons[str(getattr(m, 'name', None) or f'material {i}')] = why
    mi = np.clip(mesh.mat_index, 0, see_through.size - 1)
    t = see_through[mi]
    LAST_SPLIT.update(reasons=reasons, materials=int(see_through.size),
                      see_through=int(see_through.sum()),
                      tris=int(mesh.mat_index.size),
                      tris_see_through=int(t.sum()))
    return np.nonzero(~t)[0].astype(np.int32), np.nonzero(t)[0].astype(np.int32)


def depth_report(proj, gbuf, depth_bits, depth_sort='ZBUFFER'):
    """What this frame's z-buffer can actually resolve, in world units.

    "The depth is wrong" is a picture; this is the number behind it. The
    near and far planes come back out of the projection matrix, the
    frame's own covered depths say where the subject sits, and the
    N-bit grid step converts to the smallest world separation two
    surfaces can have and still be told apart THERE. Depth in a
    perspective frame is hyperbolic: resolution falls with the SQUARE
    of distance, so a near plane set very close spends the whole buffer
    on empty air in front of the subject -- the classic cause of
    surfaces tearing through each other at a normal bit depth. Returns
    a one-line string, or None if the frame has no covered pixels.
    """
    cov = gbuf.tri >= 0
    if proj is None or not cov.any():
        return None
    if str(depth_sort).upper() == 'PAINTERS':
        # Painter's does not store ndc depth at all: every fragment of a
        # polygon carries that polygon's single VIEW distance, so the
        # buffer holds distances, not ndc values. Reading them as ndc is
        # what made this line announce that the field's frame sat at its
        # projection's depth asymptote -- it was reporting a scene 4.5
        # to 5.5 units from the camera.
        d = gbuf.depth[cov].astype(np.float64)
        d = d[np.isfinite(d)]
        rng = f'{d.min():.4g}..{d.max():.4g}' if d.size else 'empty'
        return ('[Halcyon] depth: Painter\'s algorithm -- ONE depth per '
                f'polygon (view distance {rng}), no per-pixel z-buffer, '
                f'so the {int(depth_bits)}-bit setting does not apply. '
                'Polygons that interpenetrate meet along an edge '
                'instead of their true intersection, and a large '
                'polygon can be hidden by a small nearer one: that is '
                'the algorithm, not a fault. Depth Method -> Z-Buffer '
                'resolves per pixel')
    p = np.asarray(proj, np.float64)
    a, b = float(p[2, 2]), float(p[2, 3])
    denom_n, denom_f = a - 1.0, a + 1.0
    if abs(denom_n) < 1e-12 or abs(denom_f) < 1e-12:
        return None                     # orthographic: depth is linear
    near, far = b / denom_n, b / denom_f
    if not (0.0 < near < far):
        return None
    z = gbuf.zndc[cov].astype(np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return None
    zlo, zhi = float(z.min()), float(z.max())
    span = far - near
    head = (f'[Halcyon] depth: {int(depth_bits)}-bit z-buffer, clip '
            f'{near:.4g}..{far:.4g}, ndc z {zlo:.6f}..{zhi:.6f}')
    # A covered pixel at or past ndc z = 1 sits AT or BEYOND the far
    # plane. Far clipping is off by design -- period renderers drew it
    # -- but no distance can be recovered from such a value, and the
    # first version of this line clamped the vanishing denominator and
    # reported the subject at 2e+14 world units away. An instrument
    # that lies is worse than no instrument: it must say "I cannot
    # measure this, and here is why".
    # ndc z reaches 1 exactly AT the far plane and approaches
    # (f+n)/(f-n) as distance runs to infinity, so a value past 1 is
    # past the far clip (drawn anyway -- period renderers did) and a
    # value at the asymptote carries no recoverable distance at all:
    # in float32 that is what an enormous scene scale collapses to.
    denom = (far + near) - z * span
    good = denom > 1e-9
    past = int((z > 1.0).sum())
    if not good.any():
        return (head + '; every covered pixel sits at this '
                'projection\'s depth asymptote, where NO distance can '
                'be recovered -- the geometry is effectively infinitely '
                'far for this clip range, so the far clip needs to come '
                'in (or the scene scale down) before depth means '
                'anything')
    dist = 2.0 * far * near / denom[good]
    d_near, d_med = float(dist.min()), float(np.median(dist))
    steps = float((1 << int(max(2, min(int(depth_bits), 32)))) - 1)
    step_ndc = 2.0 / steps

    def res_at(d):
        # d(ndc z)/d(distance) = 2*f*n / ((f-n) * d^2)
        return step_ndc * span * d * d / (2.0 * far * near)

    tail = ''
    if past:
        tail += (f'; {100.0 * past / z.size:.0f}% of covered pixels are '
                 'PAST the far clip plane (drawn anyway, but their '
                 'depth is outside the range the buffer was set up for)')
    if not good.all():
        tail += (f'; {100.0 * float((~good).sum()) / good.size:.0f}% sit '
                 'at the depth asymptote and cannot be measured at all')
    return (head + f'; the subject sits at {d_near:.3g}..{d_med:.3g}, '
            f'where the buffer resolves {res_at(d_near):.3g}..'
            f'{res_at(d_med):.3g} world units -- surfaces closer '
            'together than that cannot be told apart' + tail)


def _spot_cones(img, scene, st, gbuf, vp, eye, w, h):
    """Add the visible beam of each spot light over the whole frame.

    Unlike the background this runs on every pixel, not just uncovered ones: a
    beam is in front of whatever it crosses, and stops at it rather than behind
    it. The depth buffer is what cuts it short.
    """
    if not getattr(st, 'spot_cones', False):
        return
    lights = [l for l in (scene.lights or ())
              if str(getattr(l, 'type', '')).upper() == 'SPOT'
              and float(getattr(l, 'volumetric', 0.0)) > 0.0]
    if not lights:
        return
    from . import cones as CONES
    with ST.track('spot cones'):
        inv = np.linalg.inv(vp).astype(np.float32)
        yy, xx = np.mgrid[0:h, 0:w]
        nx = (xx.ravel().astype(np.float32) + 0.5) / w * 2.0 - 1.0
        ny = (yy.ravel().astype(np.float32) + 0.5) / h * 2.0 - 1.0
        one = np.ones(nx.size, np.float32)
        world = np.stack([nx, ny, one, one], axis=1) @ inv.T
        world = world[:, :3] / np.where(np.abs(world[:, 3:4]) < 1e-9, 1e-9,
                                        world[:, 3:4])
        dirs = M.normalize(world - eye[None, :])

        depth = gbuf.depth.reshape(-1).astype(np.float32)
        # the z-buffer holds view depth; the beam needs distance along the ray
        forward = M.normalize(np.asarray(scene.camera.forward, np.float32)
                              [None, :])[0] if hasattr(scene.camera, 'forward') \
            else None
        if forward is not None:
            cosang = np.abs(np.einsum('ij,j->i', dirs, forward))
            dist = np.where(np.isfinite(depth),
                            depth / np.maximum(cosang, 1e-4), np.inf)
        else:
            dist = np.where(np.isfinite(depth), depth, np.inf)

        reach = float(getattr(st, 'spot_cone_reach', 64.0))
        add = np.zeros((nx.size, 3), np.float32)
        for light in lights:
            scatter = CONES.spot_cone(
                eye, dirs, dist, light,
                samples=int(getattr(st, 'spot_cone_samples', 12)),
                density=float(getattr(st, 'spot_cone_density', 1.0))
                * float(light.volumetric),
                falloff=float(getattr(st, 'spot_cone_falloff', 2.0)),
                max_distance=reach)
            if scatter.any():
                col = np.asarray(light.color, np.float32)[None, :]
                add += scatter[:, None] * col * float(light.energy) / np.pi
        img[:, :, :3] += add.reshape(h, w, 3)


def _background_image(scene, st, w, h, vp, eye, uncovered=None,
                      textures=None, ss=1):
    """World colour through the camera rays that miss geometry.

    Shading only the uncovered pixels matters: on a full-frame scene this is
    nearly the whole cost of the background pass, and it is entirely wasted
    work when something is drawn over the top of it.
    """
    img = np.zeros((h, w, 4), np.float32)
    # An opaque film is the default; Blender's Film > Transparent (and the
    # matching Halcyon toggle) is what makes the background punch through.
    bg_alpha = 0.0 if getattr(st, 'film_transparent', False) else 1.0

    # Supersampling multiplies the number of background rays by ss squared, and
    # a sky is smooth almost everywhere -- at 4x that is sixteen evaluations of
    # a procedural sky or an HDRI lookup for one output pixel. Evaluating it at
    # output resolution and expanding costs a barely visible amount of detail on
    # a sun disc and saves the other fifteen sixteenths.
    if ss > 1 and getattr(st, 'fast_background', True):
        lw, lh = max(w // ss, 1), max(h // ss, 1)
        low = _background_image(scene, st, lw, lh, vp, eye, None, textures, ss=1)
        big = np.repeat(np.repeat(low, ss, axis=0), ss, axis=1)
        if big.shape[0] < h or big.shape[1] < w:
            big = np.pad(big, ((0, max(h - big.shape[0], 0)),
                               (0, max(w - big.shape[1], 0)), (0, 0)), mode='edge')
        big = big[:h, :w]
        if uncovered is None:
            return big.copy()
        mask = np.broadcast_to(uncovered, (h, w))
        img[mask] = big[mask]
        return img
    if uncovered is None:
        yy, xx = np.mgrid[0:h, 0:w]
        yy = yy.ravel()
        xx = xx.ravel()
    else:
        yy, xx = np.nonzero(uncovered)
        if yy.size == 0:
            return img
    img[yy, xx, 3] = bg_alpha
    inv = np.linalg.inv(vp).astype(np.float32)
    nx = (xx.astype(np.float32) + 0.5) / w * 2.0 - 1.0
    ny = (yy.astype(np.float32) + 0.5) / h * 2.0 - 1.0
    pts = np.stack([nx, ny, np.ones(nx.size, np.float32),
                    np.ones(nx.size, np.float32)], axis=1)
    world = pts @ inv.T
    world = world[:, :3] / np.where(np.abs(world[:, 3:4]) < 1e-9, 1e-9, world[:, 3:4])
    dirs = M.normalize(world - eye[None, :])
    # the background is a large independent job and was the third biggest
    # stage on a real frame, so it is chunked like the shading
    total = dirs.shape[0]
    if total > 4 * MIN_CHUNK:
        # chunked to bound memory on a large frame, not for parallelism
        cols = np.empty((total, 3), np.float32)
        step = MAX_CHUNK
        for lo in range(0, total, step):
            hi = min(lo + step, total)
            cols[lo:hi] = world_color(scene, st, dirs[lo:hi], textures or {},
                                      hi - lo, eye=eye)
        img[yy, xx, :3] = cols
    else:
        img[yy, xx, :3] = world_color(scene, st, dirs, textures or {}, total,
                                      eye=eye)
    return img


def _bump_materials_of(job, tri):
    """{mi: material} for fragment materials carrying a Bump node."""
    mesh = job.scene.mesh
    if mesh.mat_index is None or tri.size == 0:
        return {}
    out = {}
    for mi in np.unique(mesh.mat_index[tri]):
        mi = int(mi)
        mat = job.scene.materials[mi] \
            if mi < len(job.scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        nodes = (graph or {}).get('nodes', {})
        if any(n.get('bl_idname') == 'ShaderNodeBump'
               for n in nodes.values()):
            out[mi] = mat
    return out


def _shade_fragments_cpu(job, tri, bary, px, py, front, rank, st):
    """The A-buffer fragments' CPU shading, scheduling-invariant.

    A Bump node's screen gradients must be a function of the SURFACE,
    not of the batch layout -- and for transparent fragments the surface
    is the LAYER: one fragment per pixel per rank. Shading all ranks in
    one mixed array made `_screen_grad`'s scatter collide (front and
    back faces at the same pixel, last write winning by sort order) and
    let every chunk boundary cut the waves: 539 of 1914 fragments moved
    with the chunk size in the repro scene, by up to 2.98. So when any
    fragment's material carries a Bump node, the fragments shade RANK
    BY RANK with whole-material gradient fields built from each rank's
    own fragments -- `_bump_height_fields`, the same pre-pass the opaque
    frame runs, applied per layer. Frames without a Bump material take
    the old single call: nothing else reads screen gradients here, and
    per-fragment shading is proven chunking-invariant.
    """
    bump_mats = _bump_materials_of(job, tri)
    if not bump_mats:
        return _shade_chunked(job, tri, bary, px, py, front, None, st)
    from .nodeeval import VALUE, GraphEvaluator
    mesh = job.scene.mesh
    mat_f = mesh.mat_index[tri] if mesh.mat_index is not None \
        else np.zeros(tri.size, np.int32)

    # heights are per-fragment pure, so each material's chain evaluates
    # ONCE over ALL its fragments (chunked for memory); each rank then
    # scatters ITS OWN subset and differences on the frame grid -- the
    # same field definition, at a fraction of the evaluator cost the
    # first per-rank version paid (the field's 33-second frame grew to
    # 43 on exactly that; the waves' noise chain was re-running once
    # per depth layer)
    H, W = job.height, job.width
    heights = {}
    mat_idx_of = {}
    for mi, mat in bump_mats.items():
        idx = np.nonzero(mat_f == mi)[0]
        if idx.size == 0:
            continue
        mat_idx_of[mi] = idx
        graph = getattr(mat, 'graph', None)
        nodes = (graph or {}).get('nodes', {})
        for node in nodes.values():
            if node.get('bl_idname') != 'ShaderNodeBump':
                continue
            hv = np.empty(idx.size, np.float32)
            for s in range(0, int(idx.size), int(MAX_CHUNK)):
                e = min(s + int(MAX_CHUNK), int(idx.size))
                sub = idx[s:e]
                ctx = job.context(tri[sub], bary[sub], px[sub], py[sub],
                                  front[sub] if front is not None
                                  else None, None, 0, True)
                ev = GraphEvaluator(graph, ctx, job.textures,
                                    getattr(mat, 'programs', None))
                hv[s:e] = np.asarray(ev.input(node, 'Height', VALUE),
                                     np.float32).reshape(-1)
            heights[(mi, node.get('id'))] = hv

    col = np.zeros((tri.size, 4), np.float32)
    stash = dict(getattr(job, 'bump_fields', {}) or {})
    try:
        top = int(rank.max()) if rank.size else -1
        # one stable sort finds every layer; sixteen `rank == r` scans
        # over millions of fragments used to. Stable argsort keeps equal
        # ranks in original index order, so each slice is bit-identical
        # to nonzero's ascending indices.
        rorder = np.argsort(rank, kind='stable')
        rbounds = np.searchsorted(rank[rorder], np.arange(top + 2))
        for r in range(top + 1):
            sel = rorder[rbounds[r]:rbounds[r + 1]]
            if sel.size == 0:
                continue
            fields = dict(stash)
            for (mi, node_id), hv in heights.items():
                idx = mat_idx_of[mi]
                m = rank[idx] == r
                if not m.any():
                    continue
                img = np.zeros((H, W), np.float32)
                valid = np.zeros((H, W), bool)
                img[py[idx[m]], px[idx[m]]] = hv[m]
                valid[py[idx[m]], px[idx[m]]] = True
                gx = np.zeros_like(img)
                gy = np.zeros_like(img)
                gx[:, :-1] = np.where(valid[:, 1:] & valid[:, :-1],
                                      img[:, 1:] - img[:, :-1], 0.0)
                gy[:-1, :] = np.where(valid[1:, :] & valid[:-1, :],
                                      img[1:, :] - img[:-1, :], 0.0)
                fields[(mi, node_id)] = (gx, gy)
            job.bump_fields = fields
            col[sel] = _shade_chunked(job, tri[sel], bary[sel], px[sel],
                                      py[sel],
                                      front[sel] if front is not None
                                      else None, None, st)
    finally:
        job.bump_fields = stash
    return col


#: how the last GPU-gated A-buffer frame routed its depth layers --
#: per-layer fragment counts, the threshold, and who shaded what. The
#: printed routing line reads from it; the tests assert on it so a
#: "hybrid" run can never silently be a pure one.
LAST_ROUTING = {}


def _fmt_frags(n):
    """1234567 -> '1.2M': the routing line is read off a console."""
    n = int(n)
    if n >= 1000000:
        return f'{n / 1e6:.1f}M'
    if n >= 1000:
        return f'{n / 1e3:.1f}k'
    return str(n)


def _composite_abuffer(job, frags, gbuf, img, st, band=None):
    """True A-buffer: shade every fragment, sort per pixel, composite.

    Two things used to make this the slowest stage in the renderer by a wide
    margin:

    The shading called job.shade directly, so every transparent fragment in the
    frame was shaded on one thread no matter what the thread count said. It goes
    through the chunked pool now, like the opaque pass.

    The compositing walked depth layers with `rank == r`, which is a full scan
    and a full gather over *every* fragment in the frame for *every* layer. A
    pixel a hundred deep in glass therefore cost a hundred passes over millions
    of fragments. Sorting by layer first makes each layer a contiguous slice, so
    the whole composite is one pass in total.
    """
    px, py, tri, depth, bary, front = frags.finish()
    if px.size == 0:
        return img
    opaque_z = gbuf.depth[py, px]
    # the SAME tolerant limit the collection used (raster.abuf_depth_limit):
    # a coplanar contact must not flip on which rasteriser rounded the
    # opaque depth's last ULP
    keep = depth <= raster.abuf_depth_limit(opaque_z)
    if not np.any(keep):
        return img
    px, py, tri, depth, bary, front = (a[keep] for a in
                                       (px, py, tri, depth, bary, front))

    # Sorted Blend orders whole polygons by their centroid, which is what a
    # renderer without per-fragment lists could manage -- and it shows the
    # classic sorting errors where surfaces interpenetrate. A-Buffer sorts every
    # fragment on its own depth and is correct through any arrangement.
    if st.transparency == 'SORTED':
        mesh = job.scene.mesh
        cent = mesh.verts[mesh.tris].mean(axis=1)
        view_z = np.abs((cent - job.eye[None, :]) @ job.view[:3, :3].T)[:, 2]
        key = view_z[tri].astype(np.float32)
    else:
        key = depth

    pix = py.astype(np.int64) * gbuf.width + px
    order = np.lexsort((key, pix))
    pix = pix[order]
    px = px[order]
    py = py[order]
    tri = tri[order]
    bary = bary[order]
    front = front[order]

    grp_start = np.zeros(pix.size, np.int64)
    new_group = np.nonzero(pix[1:] != pix[:-1])[0] + 1
    grp_start[new_group] = new_group
    np.maximum.accumulate(grp_start, out=grp_start)
    rank = np.arange(pix.size, dtype=np.int64) - grp_start

    limit = int(getattr(st, 'max_transparent_layers', 0) or 0)
    if limit > 0 and rank.size and int(rank.max()) >= limit:
        within = rank < limit
        # a silent truncation reads as a rendering bug: the dropped
        # layers simply are not drawn, so wherever nothing opaque sits
        # behind them the BACKGROUND shows through -- black holes in
        # the middle of solid-looking geometry. Say it, with the number
        # and the setting that fixes it.
        cut = int((~within).sum())
        hit = int(np.unique(py[~within].astype(np.int64) * gbuf.width
                            + px[~within]).size)
        print(f'[Halcyon] transparency: the {limit}-layer cap dropped '
              f'{cut} fragments at {hit} pixels -- those layers are '
              'not drawn, and where nothing opaque sits behind them '
              'the background shows through. Raise Max Transparent '
              'Layers to draw them')
        px, py, tri, bary, front, rank = (a[within] for a in
                                          (px, py, tri, bary, front, rank))
        if px.size == 0:
            return img

    # Deferred GPU shading of the layers themselves: each depth layer's
    # fragments become an ids texture and every transparent material's
    # LAYER pass (real alpha out) draws it, exactly the opaque frame's
    # mechanism. Same gate as the opaque pass -- opt-in, whole-frame only
    # -- and any refusal keeps this shading on the CPU, with the reason
    # printed. On the field frame this stage was 25.7 s of 33.7.
    #
    # RANK ROUTING: the driver pays full-frame FIXED costs per layer --
    # a draw and a readback cover every pixel whether the layer holds
    # three fragments or a million -- while the CPU path pays per
    # FRAGMENT. The 1.25.59 field split proved it: skipping absent
    # materials' passes moved nothing, because the cost was never the
    # material count, it was sixteen full-frame round trips. So each
    # layer goes to whichever path is cheaper for IT: layers below
    # `layer_gpu_min_frac` of the frame's pixels shade on the proven
    # per-rank CPU path. Routing is by WHOLE layer -- a rank is never
    # split -- so both paths build their per-rank fields from complete
    # layers and each fragment's colour is exactly what the pure run
    # would have given it. The routing prints with per-layer counts:
    # the field names its own distribution.
    col = None
    LAST_ROUTING.clear()
    if str(getattr(st, 'render_device', 'CPU')).upper() == 'GPU' and \
            getattr(st, 'gpu_shading', False) and band is None:
        from ..gpu import shade as _gpu_shade
        counts = np.bincount(rank, minlength=int(rank.max()) + 1)
        frac = float(getattr(st, 'layer_gpu_min_frac', 0.0) or 0.0)
        thresh = max(1, int(round(frac * gbuf.width * gbuf.height))) \
            if frac > 0.0 else 1
        dense = counts >= thresh
        gsel = dense[rank]
        n_gpu_r = int((dense & (counts > 0)).sum())
        n_cpu_r = int((~dense & (counts > 0)).sum())
        LAST_ROUTING.update(
            counts=[int(c) for c in counts], thresh=int(thresh),
            gpu_ranks=n_gpu_r, cpu_ranks=n_cpu_r,
            gpu_frags=int(gsel.sum()),
            cpu_frags=int(rank.size - int(gsel.sum())))
        if not gsel.any():
            # every layer sits below the break-even: nothing for the
            # driver. That is a ROUTE, not a refusal -- say so calmly,
            # and only when a driver was actually there to be skipped
            from ..gpu import device as _gdev
            ok, _dwhy = _gdev.probe()
            if ok:
                print('[Halcyon GPU] layer routing: all '
                      f'{n_cpu_r} layers below the GPU break-even '
                      f'({thresh} fragments); shaded on the CPU '
                      '(routed, not refused)')
        else:
            with ST.track('transparency shading (GPU)'):
                try:
                    got, why = _gpu_shade.shade_fragments_frame(
                        job, gbuf, tri[gsel], bary[gsel], px[gsel],
                        py[gsel], rank[gsel])
                except Exception as exc:                        # noqa: BLE001
                    got, why = None, str(exc)
            if got is None:
                # the partition was recorded above but never ACTED on:
                # say so in the record, or a "hybrid" that quietly fell
                # back whole would look like a mix to the tests
                LAST_ROUTING['refused'] = str(why)
                print('[Halcyon GPU] transparent layers on the CPU: '
                      f'{why}')
            else:
                col = np.zeros((rank.size, 4), np.float32)
                col[gsel] = got
                cpu_s = 0.0
                if n_cpu_r:
                    import time as _time
                    csel = ~gsel
                    t0 = _time.perf_counter()
                    with ST.track('transparency shading (routed CPU)'):
                        col[csel] = _shade_fragments_cpu(
                            job, tri[csel], bary[csel], px[csel],
                            py[csel], front[csel], rank[csel], st)
                    cpu_s = _time.perf_counter() - t0
                LAST_ROUTING['cpu_s'] = cpu_s
                lt = dict(getattr(_gpu_shade, 'LAST_LAYER_TIMINGS', {})
                          or {})
                if lt:
                    # the split that names the next perf target: where
                    # the layer stage's seconds actually went. Buckets
                    # are disjoint and `other` is the printed remainder
                    # -- a cost the line refuses to hide.
                    wait = max(lt.get('wall_ms', 0.0)
                               - lt.get('exec_ms', 0.0), 0.0)
                    scz = 100.0 * lt.get('scissor_px', 0.0) \
                        / max(lt.get('frame_px', 0.0), 1.0)
                    print('[Halcyon GPU] transparent split: '
                          f"plan {lt.get('plan_ms', 0.0) / 1e3:.1f}s, "
                          'compile '
                          f"{lt.get('compile_ms', 0.0) / 1e3:.1f}s, "
                          'uploads '
                          f"{lt.get('upload_ms', 0.0) / 1e3:.1f}s "
                          f"({lt.get('upload_mb', 0.0):.0f} MB), "
                          f"draws {lt.get('draw_ms', 0.0) / 1e3:.1f}s, "
                          'reads+sync '
                          f"{lt.get('read_ms', 0.0) / 1e3:.1f}s, "
                          f"sweeps {lt.get('sweep_ms', 0.0) / 1e3:.1f}s "
                          '(of which CPU ray build '
                          f"{lt.get('ray_build_ms', 0.0) / 1e3:.1f}s), "
                          f"other {lt.get('other_ms', 0.0) / 1e3:.1f}s "
                          f"of {lt.get('total_ms', 0.0) / 1e3:.1f}s; "
                          f'scissor {scz:.0f}% of full frames; '
                          f"{lt.get('ranks', 0)} of "
                          f'{n_gpu_r + n_cpu_r} layers on the GPU; '
                          f"marshal {lt.get('crossings', 0)} crossings, "
                          f'{wait / 1e3:.1f}s waiting')
                print('[Halcyon GPU] layer routing: GPU '
                      f"{n_gpu_r} layers "
                      f"({_fmt_frags(LAST_ROUTING['gpu_frags'])}), CPU "
                      f"{n_cpu_r} layers "
                      f"({_fmt_frags(LAST_ROUTING['cpu_frags'])}, "
                      f'{cpu_s:.1f}s); per-layer frags '
                      + ' '.join(_fmt_frags(c) for c in counts))
    if col is None:
        with ST.track('transparency shading'):
            col = _shade_fragments_cpu(job, tri, bary, px, py, front,
                                       rank, st)
    if st.alpha_bits < 8:
        levels = float(2 ** max(st.alpha_bits, 1) - 1)
        col[:, 3] = np.round(col[:, 3] * levels) / levels

    # group by layer so each one is a contiguous run, then composite the
    # farthest kept layer first
    layer_order = np.argsort(rank, kind='stable')
    rank_s = rank[layer_order]
    bounds = np.searchsorted(rank_s, np.arange(int(rank_s[-1]) + 2))
    for r in range(int(rank_s[-1]), -1, -1):
        lo, hi = bounds[r], bounds[r + 1]
        if hi <= lo:
            continue
        sel = layer_order[lo:hi]
        yy, xx = py[sel], px[sel]
        a = np.clip(col[sel, 3], 0.0, 1.0)[:, None]
        img[yy, xx, :3] = col[sel, :3] * a + img[yy, xx, :3] * (1.0 - a)
        img[yy, xx, 3] = np.clip(img[yy, xx, 3] + a[:, 0], 0.0, 1.0)
    return img


def edge_factor(gbuf, width=1.0):
    """Screen-space distance to the nearest triangle edge, in pixels.

    Barycentrics fall to zero at the edges, so dividing by their screen-space
    derivative converts them into a width that stays constant however far away
    the triangle is -- the standard barycentric wireframe, done properly rather
    than by thresholding raw barycentrics.

    The derivative is the whole difficulty. A central difference needs the
    pixels on *both* sides to belong to the same triangle, and on anything
    denser than a cube that is rarely true: the derivative collapses to zero,
    the distance goes to infinity, and the wire comes out as a scatter of dots
    with most of it missing. One-sided differences are used instead -- a pixel
    needs only one neighbour along each axis, in either direction -- which is
    the difference between a wireframe and a dotted line.

    A pixel with no same-triangle neighbour on either axis is a triangle about
    one pixel across. There is no derivative to measure and no interior to
    speak of, so it is treated as being *on* the edge rather than infinitely
    far from it. That is what a wireframe of a dense mesh looks like, and the
    alternative was those triangles vanishing entirely.
    """
    b = gbuf.bary
    tri = gbuf.tri

    def one_sided(axis):
        # neighbouring pixels along `axis` that share a triangle
        if axis == 1:
            same = tri[:, 1:] == tri[:, :-1]
            diff = b[:, 1:] - b[:, :-1]
        else:
            same = tri[1:, :] == tri[:-1, :]
            diff = b[1:, :] - b[:-1, :]
        d = np.zeros_like(b)
        have = np.zeros(tri.shape, bool)
        if axis == 1:
            d[:, :-1] = np.where(same[:, :, None], diff, 0.0)   # forward
            have[:, :-1] = same
            back = np.zeros_like(b)
            back[:, 1:] = np.where(same[:, :, None], diff, 0.0)  # backward
            hb = np.zeros(tri.shape, bool)
            hb[:, 1:] = same
        else:
            d[:-1, :] = np.where(same[:, :, None], diff, 0.0)
            have[:-1, :] = same
            back = np.zeros_like(b)
            back[1:, :] = np.where(same[:, :, None], diff, 0.0)
            hb = np.zeros(tri.shape, bool)
            hb[1:, :] = same
        # prefer the forward difference, fall back to the backward one
        out = np.where(have[:, :, None], d, back)
        return out, (have | hb)

    dx, has_x = one_sided(1)
    dy, has_y = one_sided(0)
    fw = np.abs(dx) + np.abs(dy)
    measurable = has_x | has_y
    dist = b / np.maximum(fw, 1e-6)
    dist = dist.min(axis=2)
    return np.where(measurable, dist, 0.0)


def edge_distance_exact(gbuf, mesh, vp, snap=0.0):
    """Distance in pixels from each covered pixel to its triangle's nearest edge.

    Computed from the triangle's own projected vertices rather than from a
    finite difference of the barycentrics. The difference matters more than it
    sounds: a derivative needs neighbouring pixels that belong to the same
    triangle, and on a mesh whose triangles are a few pixels across there are
    none -- which is how a wireframe ends up as a scatter of dots, or as
    nothing at all.

    For a point with linear barycentrics l0,l1,l2 in a triangle of screen area
    A, the distance to the edge opposite corner i is |l_i| * 2A / |e_i|. Exact,
    at any triangle size, at any resolution, with no neighbours involved.

    Returns None if it cannot be computed, so the caller can fall back.
    """
    if mesh is None or mesh.tris is None or mesh.verts is None:
        return None
    from . import raster as _raster
    try:
        _clip, screen, _invw, _z = _raster.project(
            np.asarray(mesh.verts, np.float32), vp, gbuf.width, gbuf.height,
            snap=snap)
    except Exception:                                           # noqa: BLE001
        return None

    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    out = np.full((gbuf.height, gbuf.width), 1e9, np.float32)
    if py.size == 0:
        return out
    tri = np.asarray(mesh.tris, np.int32)[gbuf.tri[py, px]]
    p0 = screen[tri[:, 0]]
    p1 = screen[tri[:, 1]]
    p2 = screen[tri[:, 2]]
    qx = px.astype(np.float32) + 0.5
    qy = py.astype(np.float32) + 0.5

    area2 = ((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
             - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    safe = np.where(np.abs(area2) < 1e-9, 1e-9, area2)
    l0 = ((p1[:, 1] - p2[:, 1]) * (qx - p2[:, 0])
          + (p2[:, 0] - p1[:, 0]) * (qy - p2[:, 1])) / safe
    l1 = ((p2[:, 1] - p0[:, 1]) * (qx - p2[:, 0])
          + (p0[:, 0] - p2[:, 0]) * (qy - p2[:, 1])) / safe
    l2 = 1.0 - l0 - l1

    e0 = np.linalg.norm(p1 - p2, axis=1)
    e1 = np.linalg.norm(p2 - p0, axis=1)
    e2 = np.linalg.norm(p0 - p1, axis=1)
    a2 = np.abs(area2)
    d0 = np.abs(l0) * a2 / np.maximum(e0, 1e-6)
    d1 = np.abs(l1) * a2 / np.maximum(e1, 1e-6)
    d2 = np.abs(l2) * a2 / np.maximum(e2, 1e-6)
    d = np.minimum(np.minimum(d0, d1), d2)
    # a triangle straddling the near plane projects a corner to nonsense; those
    # keep the fallback rather than inventing a distance
    bad = ~np.isfinite(d)
    d = np.where(bad, 0.0, d)
    out[py, px] = d.astype(np.float32)
    return out


def crease_edges(gbuf, mesh, angle_deg=25.0):
    """Pixels on a silhouette or on an edge where the surface turns.

    A wireframe of every triangle edge is what these renderers drew, and on a
    dense mesh at 320x240 it is also a solid fill: once triangles are a couple
    of pixels across, every pixel is within a pixel of an edge. That is
    arithmetic rather than a bug, and no width setting escapes it.

    This is the way out. An edge is kept only where the surface genuinely
    turns -- against the background, against another object, or across a
    face-normal difference wider than `angle_deg`. The result is one pixel
    wide whatever the triangle count behind it.
    """
    cov = gbuf.mask()
    out = np.zeros(cov.shape, bool)
    if not cov.any():
        return out
    tri = gbuf.tri
    fn = getattr(mesh, 'face_normals', None)
    obj = getattr(mesh, 'obj_index', None)
    cos_lim = float(np.cos(np.radians(max(min(angle_deg, 179.0), 0.0))))

    normals = None
    if fn is not None:
        fn = np.asarray(fn, np.float32)
        normals = np.zeros(cov.shape + (3,), np.float32)
        normals[cov] = fn[tri[cov]]

    for dy, dx in ((0, 1), (1, 0)):
        a = (slice(None, -dy or None), slice(None, -dx or None))
        b = (slice(dy, None), slice(dx, None))
        both = cov[a] & cov[b]
        # one side covered and the other not: a silhouette
        edge = cov[a] ^ cov[b]
        diff = both & (tri[a] != tri[b])
        if obj is not None and diff.any():
            oi = np.asarray(obj, np.int32)
            diff_obj = diff & (oi[tri[a]] != oi[tri[b]])
        else:
            diff_obj = np.zeros_like(diff)
        turn = np.zeros_like(diff)
        if normals is not None and diff.any():
            dot = (normals[a] * normals[b]).sum(axis=2)
            turn = diff & (dot < cos_lim)
        hit = edge | diff_obj | turn
        out[a] |= hit
        out[b] |= hit
    return out & cov


def _thicken(mask, radius):
    """Grow a boolean mask by `radius` pixels, so a wire can be more than one."""
    r = int(max(round(radius - 1.0), 0))
    if r <= 0:
        return mask
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def apply_wireframe(job, gbuf, img, st, vp, eye, textures):
    """Draw the Wireframe shading model, and the global wire overlay."""
    scene = job.scene
    mesh = scene.mesh
    cov = gbuf.mask()
    if not cov.any():
        return img
    wire_mats = {i for i, m in enumerate(scene.materials)
                 if material_model(m, st) == 'WIREFRAME'}
    if not wire_mats and not st.render_wire:
        # Nothing to carve. If a *surface* nonetheless shaded as Wireframe the
        # user is looking at a flat unlit fill and has no way to know why, so
        # the mismatch is recorded rather than left silent.
        scene.wireframe_note = None
        return img
    scene.wireframe_note = sorted(wire_mats)
    crease = None
    if str(getattr(st, 'wire_mode', 'ALL')) == 'CREASE':
        crease = crease_edges(gbuf, mesh, float(getattr(st, 'wire_angle', 25.0)))
        dist = None
    else:
        dist = edge_distance_exact(
            gbuf, mesh, vp, snap=st.vertex_snap_grid if st.vertex_snap else 0.0)
        if dist is None:
            dist = edge_factor(gbuf)
    py, px = np.nonzero(cov)
    d = dist[py, px] if dist is not None else None
    if st.render_wire:
        w = max(float(st.wire_width), 0.1)
        on = (d < w) if d is not None else \
            _thicken(crease, w)[py, px]
        col = np.asarray(st.wire_color, np.float32)
        yy, xx = py[on], px[on]
        img[yy, xx, :3] = col[None, :]
        img[yy, xx, 3] = 1.0
    if wire_mats:
        mat_idx = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
            else np.zeros(py.size, np.int32)
        is_wire = np.isin(mat_idx, list(wire_mats))
        if is_wire.any():
            widths = np.array([material_wire_size(m)
                               for m in scene.materials] or [1.0], np.float32)
            w = widths[np.clip(mat_idx, 0, widths.size - 1)]
            if d is not None:
                off = is_wire & (d >= w)
            else:
                thick = _thicken(crease, float(np.max(w)))
                off = is_wire & ~thick[py, px]
            if off.any():
                yy, xx = py[off], px[off]
                # these pixels see straight through the surface, so they get the
                # world behind them -- shaded here because the background pass
                # deliberately skipped every covered pixel
                clear = np.zeros((gbuf.height, gbuf.width), bool)
                clear[yy, xx] = True
                behind = _background_image(scene, st, gbuf.width, gbuf.height,
                                           vp, eye, clear, textures)
                img[yy, xx, :3] = behind[yy, xx, :3]
                img[yy, xx, 3] = 0.0 if getattr(st, 'film_transparent', False) \
                    else 1.0
    return img


def _debug_pass(job, gbuf, img, st):
    mesh = job.scene.mesh
    h, w = gbuf.height, gbuf.width
    out = np.zeros((h, w, 4), np.float32)
    out[:, :, 3] = 1.0
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    mode = st.debug_pass
    if mode == 'OVERDRAW':
        od = gbuf.overdraw.astype(np.float32)
        lo, hi = float(od.min()), float(od.max())
        if hi <= lo:
            # uniform coverage: show the count rather than one flat colour
            out[:, :, :3] = _heat(np.full_like(od, 0.0 if hi <= 0 else 0.5))[:, :, :3]
            return out
        out[:, :, :3] = _heat((od - lo) / (hi - lo))[:, :, :3]
        return out
    if py.size == 0:
        return out
    tri = gbuf.tri[py, px]
    bary = gbuf.bary[py, px]
    if mode == 'DEPTH':
        d = gbuf.depth[py, px]
        finite = np.isfinite(d)
        lo, hi = (d[finite].min(), d[finite].max()) if finite.any() else (0.0, 1.0)
        v = 1.0 - np.clip((d - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        out[py, px, :3] = v[:, None]
    elif mode == 'NORMAL':
        ctx = job.context(tri, bary, px, py)
        out[py, px, :3] = ctx.N * 0.5 + 0.5
    elif mode == 'UV':
        ctx = job.context(tri, bary, px, py)
        out[py, px, 0] = ctx.uv[:, 0] % 1.0
        out[py, px, 1] = ctx.uv[:, 1] % 1.0
    elif mode == 'MATID':
        mi = mesh.mat_index[tri] if mesh.mat_index is not None else tri * 0
        out[py, px, :3] = _idcolor(mi)
    elif mode == 'WIREFRAME':
        edge = (bary.min(axis=1) < 0.02).astype(np.float32)
        out[py, px, :3] = edge[:, None]
    else:
        return img
    return out


def linear_depth(job, gbuf, eye):
    """Distance from the camera, in scene units, with NaN where nothing was hit.

    `gbuf.depth` is normalised device depth: it runs 0..1 and crowds almost
    everything into the last few thousandths, which is exactly what a depth
    buffer is for and exactly wrong for anything that wants a distance. A Z
    pass in NDC is unusable in a comp, and a depth-of-field focus expressed in
    metres compared against it is not comparing anything.
    """
    h, w = gbuf.height, gbuf.width
    out = np.full((h, w), np.nan, np.float32)
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    if py.size == 0:
        return out
    ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py)
    out[py, px] = np.linalg.norm(ctx.P - np.asarray(eye, np.float32)[None, :],
                                 axis=1)
    return out


def wanted_passes(st):
    """Which extra passes this render is being asked for.

    Written out one setting at a time rather than looped over a table of
    strings, so that a reader -- and the test that hunts for controls nothing
    reads -- can see each one actually being used.
    """
    names = []
    if getattr(st, 'pass_depth', False):
        names.append('Depth')
    if getattr(st, 'pass_normal', False):
        names.append('Normal')
    if getattr(st, 'pass_position', False):
        names.append('Position')
    if getattr(st, 'pass_uv', False):
        names.append('UV')
    if getattr(st, 'pass_object_index', False):
        names.append('IndexOB')
    if getattr(st, 'pass_material_index', False):
        names.append('IndexMA')
    return tuple(names)


def build_aux_passes(job, gbuf, st, depth_m=None):
    """Raw data buffers for the compositor, straight off the G-buffer.

    These are *data*, so nothing here is display-transformed, dithered or
    filtered: a depth in metres stays a depth in metres, and an object index
    stays an integer. Uncovered pixels get the conventions Blender's own
    engines use -- 1e10 for depth, zero for everything else -- so a Z pass
    composites the same way a Cycles one does.
    """
    names = wanted_passes(st)
    if not names:
        return {}
    mesh = job.scene.mesh
    h, w = gbuf.height, gbuf.width
    cov = gbuf.mask()
    py, px = np.nonzero(cov)
    out = {}
    ctx = None
    if py.size and ({'Normal', 'Position', 'UV'} & set(names)):
        ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py)
    for name in names:
        if name == 'Depth':
            d = depth_m if depth_m is not None else linear_depth(
                job, gbuf, getattr(job, 'eye', (0.0, 0.0, 0.0)))
            d = np.where(np.isfinite(d), d, 1e10)
            out[name] = d[:, :, None].astype(np.float32)
            continue
        chans = 1 if name in ('IndexOB', 'IndexMA') else 3
        buf = np.zeros((h, w, chans), np.float32)
        if py.size:
            if name == 'Normal':
                buf[py, px] = ctx.N
            elif name == 'Position':
                buf[py, px] = ctx.P
            elif name == 'UV':
                buf[py, px, 0] = ctx.uv[:, 0]
                buf[py, px, 1] = ctx.uv[:, 1]
                buf[py, px, 2] = 1.0
            elif name == 'IndexMA':
                mi = (mesh.mat_index[gbuf.tri[py, px]]
                      if mesh.mat_index is not None else np.zeros(py.size))
                buf[py, px, 0] = np.asarray(mi, np.float32)
            elif name == 'IndexOB':
                oi = (mesh.obj_index[gbuf.tri[py, px]]
                      if mesh.obj_index is not None else np.zeros(py.size))
                buf[py, px, 0] = np.asarray(oi, np.float32)
        out[name] = buf
    return out


def _heat(t):
    t = np.clip(t, 0, 1)[..., None]
    a = np.array([0.0, 0.0, 0.3], np.float32)
    b = np.array([1.0, 0.2, 0.0], np.float32)
    c = np.array([1.0, 1.0, 0.6], np.float32)
    lo = a + (b - a) * np.clip(t * 2, 0, 1)
    hi = b + (c - b) * np.clip(t * 2 - 1, 0, 1)
    return np.where(t < 0.5, lo, hi)


def _idcolor(i):
    i = i.astype(np.float32)
    return np.stack([_hash1(i * 1.7 + 0.3), _hash1(i * 3.1 + 1.9),
                     _hash1(i * 5.3 + 4.1)], axis=1)


# ----------------------------------------------------------- AA / downsample

FILTERS = {
    'BOX': lambda x: np.ones_like(x),
    'TRIANGLE': lambda x: np.maximum(1.0 - np.abs(x), 0.0),
    'GAUSS': lambda x: np.exp(-2.0 * x * x),
    'CATROM': lambda x: _catrom(x),
    'MITCHELL': lambda x: _mitchell(x),
}


def _catrom(x):
    a = np.abs(x)
    a2 = a * a
    a3 = a2 * a
    w = np.where(a < 1.0, 1.5 * a3 - 2.5 * a2 + 1.0,
                 np.where(a < 2.0, -0.5 * a3 + 2.5 * a2 - 4.0 * a + 2.0, 0.0))
    return w


def _mitchell(x, b=1 / 3, c=1 / 3):
    a = np.abs(x)
    a2 = a * a
    a3 = a2 * a
    w1 = ((12 - 9 * b - 6 * c) * a3 + (-18 + 12 * b + 6 * c) * a2 + (6 - 2 * b)) / 6
    w2 = ((-b - 6 * c) * a3 + (6 * b + 30 * c) * a2 + (-12 * b - 48 * c) * a +
          (8 * b + 24 * c)) / 6
    return np.where(a < 1, w1, np.where(a < 2, w2, 0.0))


def _resolve(img, W, H, ss, st):
    if ss <= 1:
        return img
    fn = FILTERS.get(st.aa_filter, FILTERS['BOX'])
    off = (np.arange(ss, dtype=np.float32) + 0.5) / ss - 0.5
    wx = fn(off * 2.0 * st.aa_filter_width)
    wy = wx
    kern = np.outer(wy, wx).astype(np.float32)
    s = kern.sum()
    kern = kern / (s if abs(s) > 1e-8 else 1.0)
    tile = img[:H * ss, :W * ss].reshape(H, ss, W, ss, 4)
    return np.einsum('hiwjc,ij->hwc', tile, kern).astype(np.float32)
