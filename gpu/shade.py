"""Deferred shading of the CPU's G-buffer, on the GPU.

The route the capability table has pointed at all along: shading is about 71%
of a frame and rasterising about 9%, so the win is moving *shading* -- and the
CPU's G-buffer already holds everything shading needs. This module packs it
into textures, assembles one full-screen shader per material with every frame
constant baked in, and draws -- the same upload/draw mechanism the post stages
have proven on real hardware.

The honesty rules, same as every GPU stage:

* Qualification is explicit. A frame runs here only when everything it needs
  is inside what the shaders reproduce -- and when it is not, the frame stays
  on the CPU and the *reason* is a string somebody can read, not a silent
  wrong picture.
* The surface constants are not re-derived. Each material is probed through
  the renderer's own `closure_to_surface` on fragments taken from the actual
  frame, so what gets baked is what the CPU would have used, by construction.
* The whole thing is verifiable without a GPU: `simulate` runs the same
  sources through Halcyon's own GLSL front-end. The only part that needs a
  driver is the driver.
"""

import numpy as np

#: how many fragments to probe per material when checking its surface is
#: constant across the frame. Spread over the frame, so a texture feeding
#: roughness has to be constant everywhere to slip through, not just flat
#: across one triangle.
PROBE_FRAGMENTS = 16

#: surface fields that must be inert for the frame shader to be the whole
#: story: each is a term light_surface or shade_batch applies that the GLSL
#: does not carry. (name, inert value)
#: surface fields that must be inert UNLESS the settings make them so on the
#: CPU too. Transparency NONE forces alpha to 1.0 after everything -- the
#: era's no-alpha-unit behaviour -- so opacity and edge opacity are inert
#: there by construction and shade freely. Any other mode still refuses.
INERT_FIELDS = (('edge_opacity', 1.0), ('opacity', 1.0))

#: master-shader extras the frame shader carries when their coefficients are
#: constant: (scalar fields...), (colour fields...). Rim and fresnel are the
#: era's silhouette cheats and sheen is the velvet lobe -- all three are on
#: nearly every converted material, so refusing them refused real scenes.
EXTRA_SCALARS = ('fresnel', 'fresnel_power', 'rim', 'rim_power',
                 'sheen', 'sheen_roughness', 'matcap_blend', 'backface_mix',
                 'reflect', 'refraction', 'edge_opacity')
EXTRA_COLORS = ('fresnel_color', 'rim_color', 'sheen_color', 'matcap',
                'backface_color', 'reflect_color')

#: models the GLSL dispatch reproduces at pixel rate. WIREFRAME and CONSTANT
#: return early on the CPU (no light loop), GOURAUD and FLAT are shading
#: rates, and WARD/ANISOTROPIC are fine -- the tangent frame is emitted.
UNSUPPORTED_MODELS = frozenset({'WIREFRAME', 'CONSTANT', 'GOURAUD', 'FLAT'})


def _model_index(model):
    from ..core.shading import MODEL_ITEMS
    for i, item in enumerate(MODEL_ITEMS):
        if item[0] == model:
            return i
    return None


def _constant(arr, tol=1e-5):
    a = np.asarray(arr, np.float32)
    if a.size == 0:
        return True, 0.0
    spread = float(np.ptp(a, axis=0).max()) if a.ndim > 1 else float(np.ptp(a))
    return spread <= tol, spread


def _probe_material(job, gbuf, mi, py, px, frags=None, layer=False,
                    secondary=False):
    """Run the CPU's own closure path on real fragments of material `mi`.

    Returns (bake, model, why). `bake` holds every constant assemble_frame
    needs, harvested from the same code that would have shaded the frame.

    `frags=(tri_idx, bary)` probes explicit surface samples instead of
    frame pixels -- how a material with no pixel on screen is probed when
    a reflection ray could still hit it. The constancy rule holds the
    same way: sixteen samples spread over the material's own triangles.
    `layer=True` probes for a TRANSPARENT-LAYER pass: the alpha fields
    are the point there, so the inertness rule does not apply.
    `secondary=True` probes for a SECONDARY (hit) pass: every consumer of
    a hit colour reads rgb only, so the alpha fields cannot reach the
    picture and are inert by construction -- without this, a see-through
    material anywhere in the mesh refused the whole ray plan under
    Sorted/A-Buffer ('visible only in reflections' + the alpha message).
    """
    from ..core.render import closure_to_surface, RATE_FOR_MODEL
    from ..core.nodeeval import GraphEvaluator

    st = job.settings
    mat = job.scene.materials[mi] if mi < len(job.scene.materials) else None

    if frags is not None:
        tri_idx, bary = frags
        ctx = job.context(tri_idx, bary, None, None,
                          np.ones(tri_idx.size, bool), None, 0, True)
    else:
        tri_idx = gbuf.tri[py, px]
        bary = gbuf.bary[py, px]
        ctx = job.context(tri_idx, bary, px, py,
                          np.ones(px.size, bool), None, 0, True)
    cl = None
    if mat is not None and mat.graph:
        ev = GraphEvaluator(mat.graph, ctx, job.textures, mat.programs)
        cl, disp = ev.evaluate_surface()
        if ev.unsupported:
            return None, None, ('nodes the evaluator itself does not know: '
                                + ', '.join(sorted(ev.unsupported)))
        if ev.cache.get('__discard') is not None:
            return None, None, 'the material discards fragments'
        if disp is not None and st.displacement_scale > 0.0:
            return None, None, 'displacement moves the shading normal'
        # the frame pass emits ONE per-pixel colour chain and feeds it to
        # the DIFFUSE term; on a MASTER graph that is exactly the Diffuse
        # Color socket, but a raw BSDF graph's colour belongs to its own
        # lobe. A raw GLOSSY, GLASS or EMISSION lobe's colour would land
        # in the wrong slot and shade a DIFFERENT picture -- the worst
        # outcome there is. Found by the sim the round the Metallic and
        # Specular BSDF nodes arrived: raw glossy graphs had shaded wrong
        # on the driver since the emitters existed, unseen because no
        # matrix row ever carried one. Refuse BY NAME; the pictures stay
        # the CPU's own until the specular slot routing is ported.
        from .material import master_node
        if getattr(cl, 'items', None) and \
                master_node(mat.graph) is None:
            # HALCYON lobes stay allowed: their colour IS the diffuse
            # chain (e_halcyon_shader emits exactly that, and a Mix
            # Shader of masters blends those chains), and every other
            # parameter bakes from the closure -- which now carries the
            # WEIGHTED blend. The probe's own constancy rule still routes
            # any mix whose baked fields vary per pixel.
            offside = sorted({k for k, w, _p in cl.items
                              if k not in ('DIFFUSE', 'TRANSPARENT',
                                           'HALCYON')
                              and bool(np.any(np.asarray(w) > 1e-6))})
            if offside:
                return None, None, (
                    'a raw ' + '/'.join(offside) + ' lobe rides the '
                    'specular or emission slot, and the frame pass emits '
                    'only the diffuse chain -- this material shades on '
                    'the CPU exactly (Convert to Halcyon Shader puts it '
                    'on the driver)')
    if mat is not None and getattr(mat, 'shadeless', False):
        return None, None, 'shadeless materials take the early CPU path'

    surf, model, nrm = closure_to_surface(cl, ctx, st, mat)
    if nrm is not None:
        # a bent normal qualifies exactly when the frame shader will bend
        # it identically: the master node's Normal chain, which the
        # assembler emits. Any other source of one still shades on the CPU.
        from .material import master_normal_linked
        if not master_normal_linked(getattr(mat, 'graph', None)
                                    if mat is not None else None):
            return None, None, ('the graph bends the shading normal outside '
                                "the master shader's Normal socket")
    rate = str(RATE_FOR_MODEL.get(model, st.shading_rate))
    if rate not in ('PIXEL', 'VERTEX', 'FACE'):
        return None, None, f'unknown shading rate {rate}'
    if rate != 'PIXEL' and (layer or secondary):
        which = 'hit' if secondary else 'transparent-layer'
        return None, None, (f'{model or "the material"} shades at {rate} '
                            f'rate, and a {which} pass lights per pixel '
                            f'-- the light loop has no {model} formula')
    if rate == 'PIXEL':
        if model in UNSUPPORTED_MODELS:
            return None, None, \
                f'the {model} model shades outside the light loop'
        idx = _model_index(model)
        if idx is None:
            return None, None, f'unknown model {model}'
    else:
        # a vertex- or face-rate pass never LIGHTS in GLSL: the CPU
        # computes the corner lighting -- shadows, rays, env, the
        # model's own formula -- and the pass interpolates it, so the
        # model needs no GLSL dispatch entry at all
        idx = 0

    for name, inert in INERT_FIELDS:
        if layer or secondary:
            # a transparent-layer pass EMITS the alpha chain (these
            # fields are its whole point), and a secondary pass's alpha
            # is coverage only (hit composites read rgb) -- not a leak
            # either way
            continue
        if str(getattr(st, 'transparency', 'SORTED')) == 'NONE':
            # alpha is forced to 1.0 after everything on the CPU, so these
            # fields cannot reach the picture; the GPU writes 1.0 too
            continue
        field = getattr(surf, name, None)
        if field is None:
            continue
        if float(np.abs(np.asarray(field) - inert).max()) > 1e-4:
            return None, None, f'the material uses {name}, which needs the ' \
                               f'alpha compositing the deferred target ' \
                               f'does not do (Transparency NONE shades it)'

    bake = {'__rate': rate}
    from .material import BAKE_FIELDS, per_pixel_fields
    # fields a linked master-node socket will compute per pixel are exempt
    # from the constancy rule -- varying is their whole point, and the
    # assembler emits the same chain the evaluator just ran
    perpix = set(per_pixel_fields(getattr(mat, 'graph', None)
                                  if mat is not None else None))
    for name in BAKE_FIELDS:
        if name in perpix:
            continue
        ok, spread = _constant(getattr(surf, name))
        if not ok:
            return None, None, f'{name} varies across the frame ' \
                               f'(spread {spread:.4g}); only the base ' \
                               f'colour may vary per pixel'
        arr = np.asarray(getattr(surf, name))
        bake[name] = float(arr.reshape(arr.shape[0], -1)[0, 0])
    for name in ('specular', 'emission') + EXTRA_COLORS:
        if name in perpix:
            continue
        field = getattr(surf, name, None)
        if field is None:
            continue
        ok, spread = _constant(field)
        if not ok:
            return None, None, f'{name} varies across the frame'
        bake[name] = tuple(float(v) for v in np.asarray(field)[0])
    for name in EXTRA_SCALARS:
        field = getattr(surf, name, None)
        if field is None:
            continue
        ok, spread = _constant(field)
        if not ok:
            return None, None, f'{name} varies across the frame; only the '                                f'base colour may vary per pixel'
        arr = np.asarray(field)
        bake[name] = float(arr.reshape(-1)[0])
    ok, _spread = _constant(surf.ambient)
    if not ok:
        return None, None, 'ambient level varies across the frame'
    bake['ambient'] = float(np.asarray(surf.ambient)[0])
    if mat is not None and not mat.graph:
        bake['diffuse'] = tuple(float(v) for v in np.asarray(surf.diffuse)[0])
    elif mat is None:
        bake['diffuse'] = tuple(float(v) for v in np.asarray(surf.diffuse)[0])
    # when the base colour happens to be flat -- graph or not -- remember
    # it: the refraction blend (rgb*(1-k) + hit*k*diffuse) needs the
    # PRIMARY pixel's diffuse as a constant, and a flat one qualifies a
    # material the general rule could not
    ok_d, _sd = _constant(surf.diffuse)
    if ok_d and surf.diffuse.shape[0]:
        bake['diffuse_flat'] = tuple(float(v)
                                     for v in np.asarray(surf.diffuse)[0])
    return bake, idx, None


def _env_world(job):
    """(spec, why) for the environment-reflection term's world.

    The spec mirrors `world_color`'s own branch order for the plain NODES
    path: env texture, then the two-colour blend, then the solid colour.
    Anything richer -- an active sky mode, a world node graph, the ground
    plane -- returns a reason instead, and a reflective material under it
    shades on the CPU. A missing world reflects black on the CPU, which is
    the same as emitting nothing.
    """
    world = getattr(job.scene, 'world', None)
    if world is None:
        return None, None                     # env is zeros: emit nothing
    mode = str(getattr(world, 'mode', 'NODES'))
    if mode in ('SOLID', 'GRADIENT', 'BANDS'):
        # the simple sky modes are closed-form arithmetic on the ray's z --
        # exactly `sky.solid/gradient/bands`, portable to the shader with
        # every constant baked. Rotation is a no-op here (it spins x and y,
        # these read only z), strength multiplies at the end as evaluate()
        # does, and the ground PLANE (a traced surface) still refuses.
        if getattr(world, 'ground_plane', False):
            return None, 'the ground plane is not in the deferred pass yet'
        strength = float(getattr(world, 'strength', 1.0))
        if mode == 'SOLID':
            return ('SKY_SOLID',
                    tuple(float(c) * strength
                          for c in getattr(world, 'color', (0, 0, 0)))), None
        spec = {
            'horizon': tuple(float(v) for v in world.horizon),
            'zenith': tuple(float(v) for v in world.zenith),
            'ground': tuple(float(v) for v in
                            getattr(world, 'ground_color', (0, 0, 0))),
            'height': float(getattr(world, 'horizon_height', 0.0)),
            'falloff': max(float(getattr(world, 'gradient_falloff', 1.0)),
                           0.01),
            'blend': str(getattr(world, 'blend_mode', 'LINEAR')),
            'show_ground': bool(getattr(world, 'show_ground', False)),
            'strength': strength,
        }
        if mode == 'BANDS':
            spec['steps'] = max(int(getattr(world, 'band_count', 8)), 1)
            spec['soft'] = float(np.clip(getattr(world, 'band_softness',
                                                 0.0), 0.0, 1.0))
        return ('SKY_BANDS' if mode == 'BANDS' else 'SKY_GRAD', spec), None
    if mode in ('STARFIELD', 'BRYCE', 'PHYSICAL', 'HDRI'):
        # rich skies take the CPU-composite path: the env term is the
        # LAST rgb term the CPU adds (fog frames refuse), and every
        # pixel it applies to is CPU-known -- so the renderer evaluates
        # its own world along the reflected rays and the composite adds
        # it after readback. Exact for ANY world, by construction.
        return ('CPU',), None
    if getattr(world, 'graph', None):
        return ('CPU',), None
    if getattr(world, 'ground_plane', False):
        return ('CPU',), None
    env_img = getattr(world, 'env_image', None)
    if env_img is not None:
        key = getattr(env_img, 'name', None)
        tex = (job.textures or {}).get(key) or \
            (job.textures or {}).get('world_env')
        if tex is None:
            return ('CPU',), None
        tkey = key if key in (job.textures or {}) else 'world_env'
        kind = 'MIRRORBALL' if str(getattr(world, 'env_mapping', '')) == \
            'MIRRORBALL' else 'EQUIRECT'
        return (kind, tkey), None
    if getattr(world, 'sky_blend', False):
        return ('BLEND', tuple(float(v) for v in world.horizon),
                tuple(float(v) for v in world.zenith)), None
    return ('SOLID', tuple(float(v)
                           for v in getattr(world, 'color', (0, 0, 0)))), None


def _shadow_meta(light, st, bvh=None):
    """(meta, atlas, why) for one light's shadow term.

    (None, None, None) means the light casts no shadow and needs no code --
    which must mean the CPU also treats it as fully lit, or the frame does
    not qualify. The meta carries everything `_shadow_function` bakes; for a
    mapped light the atlas is the depth data, one cell per cube face. A ray
    light needs no atlas of its own -- the BVH textures are shared and the
    caller packs them once for the frame.
    """
    from ..core.lights import CubeShadow, ShadowMap

    if not getattr(st, 'shadows', True) or \
            getattr(light, 'shadow', 'MAP') == 'NONE':
        return None, None, None
    mode = light.shadow if st.shadow_default == 'PER_LIGHT' else \
        st.shadow_default
    if mode == 'NONE':
        return None, None, None
    sm = getattr(light, 'shadow_map', None)
    if mode == 'RAY' or (sm is None and bvh is not None
                         and getattr(st, 'ray_shadows', True)):
        # exactly `visibility`'s RAY branch, decided per light
        if bvh is None:
            # the CPU only builds a BVH for shadow_default RAY (or the ray
            # features the plan has already refused), and without one the
            # RAY branch returns fully lit. Mirror that: no shadow term.
            return None, None, None
        samples = max(1, int(getattr(st, 'shadow_samples', 1))) \
            if float(getattr(light, 'radius', 0.0)) > 0.0 else 1
        # soft ray shadows travel now: the jitter is a pure function of
        # (pixel, sample, light, seed) through the pattern hash and the
        # shared unit-circle table, so both devices draw the SAME rays
        meta = {'ray': True,
                'bias': max(float(getattr(st, 'ray_bias', 1e-3)), 1e-4),
                'radius': float(getattr(light, 'radius', 0.0)),
                'samples': int(samples)}
        return meta, None, None
    if sm is None:
        # the CPU treats a missing map as lit (the ray fallback above
        # already had its chance)
        return None, None, None
    if isinstance(sm, CubeShadow):
        faces_sm = list(sm.faces)
        grid = (3, 2)
        origin = sm.origin
    elif isinstance(sm, ShadowMap):
        faces_sm = [sm]
        grid = (1, 1)
        origin = sm.origin
    else:
        return None, None, f'unknown shadow map type {type(sm).__name__}'
    first = faces_sm[0]
    size = int(first.size)

    # packed lazily, behind a content fingerprint: the CPU caches these maps
    # across frames, and re-packing + re-uploading an unchanged 33 MB cube
    # atlas every frame was most of the warm frame's cost
    key = ('shadow', size, grid,
           tuple(round(float(f.depth[::23].sum()), 3) for f in faces_sm))

    def build(_faces=faces_sm, _size=size, _grid=grid):
        atlas = np.zeros((_size * _grid[1], _size * _grid[0], 4), np.float32)
        for fi, f in enumerate(_faces):
            cy, cx = (fi // _grid[0]) * _size, (fi % _grid[0]) * _size
            atlas[cy:cy + _size, cx:cx + _size, 0] = f.depth
        return atlas

    bias = float(getattr(light, 'shadow_bias', 0.0) or st.shadow_bias)
    soft = max(float(getattr(light, 'shadow_softness', 1.0))
               * float(st.shadow_softness), 0.0)
    meta = {
        'faces': [{'vp': np.asarray(f.vp, np.float32)} for f in faces_sm],
        'size': size, 'near': float(first.near), 'far': float(first.far),
        'persp': bool(first.persp), 'extent': float(first.extent),
        'origin': tuple(float(v) for v in origin), 'grid': grid,
        'bias': bias, 'softness': soft,
        'density': float(getattr(light, 'shadow_density', 1.0)),
    }
    return meta, (key, build), None


def _light_sig(l):
    t = lambda v: tuple(round(float(x), 6) for x in v)
    return (l.type, t(getattr(l, 'position', (0, 0, 0))),
            t(getattr(l, 'direction', (0, 0, -1))),
            t(getattr(l, 'color', (1, 1, 1))),
            round(float(getattr(l, 'energy', 1.0)), 6),
            getattr(l, 'decay', 'DEFAULT'),
            round(float(getattr(l, 'decay_start', 0.0)), 6),
            round(float(getattr(l, 'decay_end', 40.0)), 6),
            round(float(getattr(l, 'spot_size', 0.0)), 6),
            round(float(getattr(l, 'spot_blend', 0.0)), 6),
            getattr(l, 'shadow', 'MAP'), bool(getattr(l, 'negative', False)),
            round(float(getattr(l, 'shadow_bias', 0.0)), 6),
            round(float(getattr(l, 'shadow_softness', 1.0)), 6),
            round(float(getattr(l, 'shadow_density', 1.0)), 6),
            round(float(getattr(l, 'radius', 0.0)), 6),
            # a projected texture bakes its frame, strength and size into
            # the pass, and its pixels ride the upload cache: swap or edit
            # the image and the plan must rebuild
            (None if getattr(l, 'cookie', None) is None else
             (round(float(getattr(l, 'cookie_strength', 1.0)), 6),
              round(float(getattr(l, 'cookie_scale', 10.0)), 6),
              _cookie_sig(l))))


def _cookie_sig(l):
    px = getattr(getattr(l, 'cookie', None), 'pixels',
                 getattr(l, 'cookie', None))
    try:
        px = np.asarray(px, np.float32)
        return (px.shape[1], px.shape[0], round(float(px[::7, ::7].sum()), 3))
    except Exception:                                           # noqa: BLE001
        return None


def _mat_sig(m):
    if m is None:
        return None
    t = lambda v: tuple(round(float(x), 6) for x in v)
    graph = getattr(m, 'graph', None)
    return (m.name, getattr(m, 'model', None), t(m.diffuse), t(m.specular),
            round(float(m.diffuse_level), 6), round(float(m.specular_level), 6),
            round(float(m.glossiness), 6), round(float(m.roughness), 6),
            round(float(m.ambient_level), 6), round(float(m.opacity), 6),
            round(float(getattr(m, 'reflect_level', 0.0)), 6),
            round(float(getattr(m, 'emission_level', 0.0)), 6),
            round(float(getattr(m, 'ior', 1.45)), 6),
            hash(repr(graph)) if graph else None)


def _plan_sig(job, mkey):
    """Everything the plan depends on, cheap enough to compute per frame.

    The camera is deliberately absent: the sources no longer contain it, and
    the whole point of caching the plan is that an orbit re-plans nothing.
    """
    st = job.settings
    scene = job.scene
    st_sig = tuple(getattr(st, n, None) for n in (
        'raytrace', 'ambient_occlusion', 'fog', 'shadows', 'shadow_default',
        'force_model', 'shading_rate', 'two_sided_lighting',
        # a GATE the plan reads MUST be in this signature, or a cache hit
        # walks straight past the refusal: the affine gate shipped outside
        # it, the field matrix ran 'texture NEAREST' first (same signature
        # once tex_perspective is invisible), and the affine row reused
        # the cached valid plan -- 0.835 over 1355 px, twice
        'tex_perspective',
        # the filter trio bakes into the mip samplers (R78's lesson: a
        # gate or bake the plan reads MUST be in this signature)
        'tex_aniso', 'tex_mip_bias', 'tex_mipmap',
        'transparency', 'env_reflection',
        'specular_in_gamma', 'clamp_specular', 'light_clamp',
        'light_falloff_default', 'shadow_samples', 'shadow_bias',
        'shadow_softness', 'tex_filter', 'max_lights', 'light_limit_mode',
        'global_ambient', 'global_ambient_level', 'default_model',
        'displacement_scale', 'normal_source', 'ray_shadows', 'ray_bias',
        'ray_reflection', 'ray_refraction', 'ray_depth',
        'ao_samples', 'ao_distance', 'ao_intensity', 'seed',
        # the radiosity gather bakes ALL of these (and its albedo table
        # follows the materials, which _mat_sig already fingerprints);
        # the blur pair GATES the plan (R78: a gate the plan reads must
        # be in the signature)
        'radiosity', 'radiosity_samples', 'radiosity_distance',
        'radiosity_intensity', 'radiosity_spacing', 'reflection_blur',
        'reflection_blur_samples',
        # the transparent-layer alpha chain bakes this cutoff
        'alpha_threshold'))
    world = getattr(scene, 'world', None)
    world_sig = None
    if world is not None:
        world_sig = (tuple(getattr(world, 'ambient', (0, 0, 0))),
                     float(getattr(world, 'ambient_level', 0.0)),
                     # the environment-reflection term reads these; a sky
                     # edit must re-plan the reflective materials
                     str(getattr(world, 'mode', 'NODES')),
                     bool(getattr(world, 'sky_blend', False)),
                     tuple(getattr(world, 'color', (0, 0, 0))),
                     tuple(getattr(world, 'horizon', (0, 0, 0))),
                     tuple(getattr(world, 'zenith', (0, 0, 0))),
                     getattr(getattr(world, 'env_image', None), 'name', None),
                     str(getattr(world, 'env_mapping', '')),
                     bool(getattr(world, 'ground_plane', False)),
                     bool(getattr(world, 'graph', None)),
                     # the simple sky modes bake these into the env term
                     tuple(getattr(world, 'ground_color', (0, 0, 0))),
                     round(float(getattr(world, 'horizon_height', 0.0)), 6),
                     round(float(getattr(world, 'gradient_falloff', 1.0)), 6),
                     str(getattr(world, 'blend_mode', 'LINEAR')),
                     bool(getattr(world, 'show_ground', False)),
                     int(getattr(world, 'band_count', 8)),
                     round(float(getattr(world, 'band_softness', 0.0)), 6),
                     round(float(getattr(world, 'strength', 1.0)), 6))
    shadow_sig = tuple(
        (None if getattr(l, 'shadow_map', None) is None else
         ('cube', tuple(round(float(f.depth[::23].sum()), 3)
                        for f in l.shadow_map.faces))
         if hasattr(getattr(l, 'shadow_map', None), 'faces') else
         ('map', round(float(l.shadow_map.depth[::23].sum()), 3)))
        for l in scene.lights)
    return (mkey, st_sig, world_sig, shadow_sig,
            # whether a BVH exists decides the RAY branch (lit vs traced),
            # and its content is the mesh's, which mkey already fingerprints
            getattr(job, 'bvh', None) is not None,
            # the camera's POSITION stays out (an orbit re-plans nothing),
            # but its TYPE gates the backface override, so it is in
            str(getattr(getattr(scene, 'camera', None), 'type', 'PERSP')),
            # coded shaders may bake the frame size (vScreenUV/iResolution)
            (int(job.width), int(job.height)),
            tuple(_light_sig(l) for l in scene.lights),
            tuple(_mat_sig(m) for m in scene.materials),
            tuple(sorted(job.textures or {})))


#: the last few plans, keyed by scene signature. A plan re-probes materials
#: through the CPU's closure path and assembles ~500-line sources; a held
#: still scene was paying that every frame, and it measured 4.4 ms of a
#: 14.3 ms warm frame at 480x360.
_PLAN_CACHE = {}


def plan_frame(job, gbuf, use_cache=True):
    """Decide whether this frame can shade on the GPU, and build its passes.

    Returns (passes, why, shadow_atlases). `passes` is a list of
    (mat_id, name, source, binds) -- one full-screen pass per material
    present in the frame -- or None with the first disqualifying reason.
    The reasons are the interface: they are what the console says instead of
    rendering the wrong picture.

    Plans are cached on a content signature of everything they read -- mesh,
    materials, lights, shadow maps, the relevant settings -- so an animation
    re-plans only what changed. The one trade this makes: the constancy
    probes (is roughness flat across the frame?) run on the first frame of a
    scene state rather than on every frame. A surface parameter that is
    constant on frame one and varying on frame two without anything else
    changing would slip through -- and would also have slipped past frame
    one's sixteen probe points, so the cache does not lower the bar.
    """
    from ..core import lights as LI
    from .material import assemble_frame

    sig = None
    if use_cache:
        try:
            sig = _plan_sig(job, _mesh_key(job.scene.mesh))
        except Exception:                                       # noqa: BLE001
            sig = None
        hit = _PLAN_CACHE.get(sig) if sig is not None else None
        if hit is not None:
            return hit

    st = job.settings
    scene = job.scene

    # ambient occlusion travels: hash-driven hemisphere rays through the
    # shared traversal, sampling identical to the CPU's by construction.
    # Without a BVH the CPU's AO term is silently skipped -- mirror that.
    ao_on = bool(getattr(st, 'ambient_occlusion', False)) and \
        getattr(job, 'bvh', None) is not None
    # the radiosity gather travels the same way: closest-hit rays, a
    # baked per-material albedo table, the gather's own hash salt. It
    # supersedes plain AO exactly as light_surface does.
    rad_on = bool(getattr(st, 'radiosity', False)) and \
        getattr(job, 'bvh', None) is not None
    if rad_on:
        ao_on = False
    if bool(getattr(st, 'raytrace', False)) and \
            float(getattr(st, 'reflection_blur', 0.0)) > 1e-3:
        # blurry reflections average N jittered rays per reflective
        # fragment; the deferred sweeps carry exactly one ray per
        # fragment, so the whole frame shades where the cone is real
        return None, 'blurry reflections trace a cone of jittered rays ' \
                     'per fragment; the deferred sweeps carry one -- ' \
                     'the frame shades on the CPU', {}
    # ray tracing at ANY depth: the CPU's recursion is a tree -- at depth
    # d < D a hit's own reflective/refractive materials spawn the next
    # rays (and no env term), at d == D the hit shades with the
    # environment, exactly the depth-exhausted branch. The deferred pass
    # walks the same tree branch by branch, compositing backward.
    ray_on = bool(getattr(st, 'raytrace', False))
    ray_depth = max(int(getattr(st, 'ray_depth', 1)), 1) if ray_on else 1
    # fog no longer refuses: it is separable, and the readback takes the
    # CPU's own apply_fog (see _fog_readback) -- same modes, same vertex
    # quantisation, same height layer, same order, by construction
    if not getattr(st, 'tex_perspective', True):
        # the feature matrix measured this one instead of guessing: the
        # deferred pass shaded an affine frame at 0.835 max difference
        # over 1355 pixels. Affine mode interpolates SCREEN-LINEAR
        # coordinates -- the CPU carries the warp through its whole
        # shading path (and affine subdivision re-splits triangles), and
        # the frame shader reads perspective barycentrics. Until the
        # affine carry is ported, these frames shade where they are
        # correct.
        return None, 'affine texture mapping interpolates screen-linear ' \
                     'coordinates the deferred pass does not carry ' \
                     '(the PS1 warp shades on the CPU)', {}
    if str(getattr(st, 'force_model', 'NONE')) in ('WIREFRAME', 'CONSTANT'):
        # GOURAUD and FLAT left this list when their rates were ported:
        # each material's probe now decides its rate, and vertex-rate
        # passes interpolate CPU-lit corners instead of lighting in GLSL
        return None, f'Force Model is {st.force_model}', {}
    if str(getattr(st, 'shading_rate', 'PIXEL')) not in ('PIXEL', 'VERTEX',
                                                         'FACE'):
        return None, f'the scene shading rate is {st.shading_rate}', {}
    if str(getattr(st, 'normal_source', 'AUTO')) == 'FACE':
        # FACE replaces the shading normal with the face normal AFTER the
        # graph runs -- the graph still sees the smooth one -- so the frame
        # would need both normals per fragment, and the G-buffer carries one
        return None, 'Normal Source FACE shades with face normals the ' \
                     'G-buffer does not carry (the graph still reads the ' \
                     'interpolated one)', {}

    lights = LI.select_lights(scene.lights, st)
    for l in lights:
        if l.type not in ('SUN', 'POINT', 'SPOT', 'AREA'):
            return None, f'{l.type} lights are not in the GLSL light loop', {}
        if getattr(l, 'exclude_objects', None):
            return None, 'light linking needs per-object masks', {}

    # shadow maps: the same maps the CPU just baked, packed for the GPU
    shadows, atlases = [], {}
    for i, l in enumerate(lights):
        meta, atlas, why = _shadow_meta(l, st, getattr(job, 'bvh', None))
        if why is not None:
            return None, f"light '{getattr(l, 'name', i)}': {why}", {}
        shadows.append(meta)
        if atlas is not None:
            atlases[f'hal_shadow{i}'] = atlas

    # projected light textures (gobos): the image rides along per light,
    # in the same cached-upload idiom, and the light loop samples it with
    # the CPU's own bilinear texel arithmetic written out in GLSL
    cookies = {}
    for i, l in enumerate(lights):
        ck = getattr(l, 'cookie', None)
        if ck is None or l.type not in ('SPOT', 'SUN'):
            continue
        px = np.asarray(getattr(ck, 'pixels', ck), np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        if px.shape[2] < 4:
            px = np.concatenate(
                [px] + [px[:, :, :1]] * (3 - px.shape[2])
                + [np.ones(px.shape[:2] + (1,), np.float32)], axis=2) \
                if px.shape[2] == 1 else np.concatenate(
                    [px, np.ones(px.shape[:2] + (1,), np.float32)], axis=2)
        ckey = ('cookie', str(getattr(l, 'name', i)), px.shape[1],
                px.shape[0], float(px[::7, ::7].sum()))
        atlases[f'hal_cookie{i}'] = (ckey, (lambda p=px: p))
        s_ax, u_ax, f_ax = LI.cookie_frame(l)
        cookies[i] = {
            'kind': l.type,
            'side': tuple(float(v) for v in s_ax),
            'up': tuple(float(v) for v in u_ax),
            'fwd': tuple(float(v) for v in f_ax),
            'tanh': float(max(np.tan(float(getattr(l, 'spot_size', 1.2))
                                     * 0.5), 1e-6)),
            'scale': float(max(getattr(l, 'cookie_scale', 10.0), 1e-6)),
            'strength': float(np.clip(getattr(l, 'cookie_strength', 1.0),
                                      0.0, 1.0)),
            'w': int(px.shape[1]), 'h': int(px.shape[0]),
        }

    # ray shadows: the BVH rides along as two textures shared by every ray
    # light, in the same cached-upload idiom as the map atlases, with the
    # texture sides baked into the traversal as literals (the plan signature
    # fingerprints the mesh, so a changed BVH re-plans anyway)
    bvh_sides = None
    if ao_on or rad_on or any(m is not None and m.get('ray')
                              for m in shadows):
        from .rtrace import bvh_atlas_entries
        entries, bvh_sides = bvh_atlas_entries(job.bvh)
        atlases.update(entries)

    # the unit-circle table: 256 cos/sin pairs shared by every soft-shadow
    # and AO sample, as a texture so every device reads the SAME float32
    # values -- a driver's own sin/cos round differently, and an occlusion
    # ray is a cliff
    soft_any = any(m is not None and m.get('ray')
                   and int(m.get('samples', 1)) > 1 for m in shadows)
    if ao_on or rad_on or soft_any:
        from ..core.patterns import CIRCLE256

        def _build_circle():
            img = np.zeros((1, 256, 4), np.float32)
            img[0, :, 0] = CIRCLE256[:, 0]
            img[0, :, 1] = CIRCLE256[:, 1]
            return img

        atlases['hal_circle'] = (('circle256', 1), _build_circle)

    covered = gbuf.tri >= 0
    if covered.any():
        mat_ids = np.unique(
            scene.mesh.mat_index[gbuf.tri[covered]]
            if scene.mesh.mat_index is not None else np.zeros(1, np.int32))
    else:
        # nothing OPAQUE to shade is a success for the deferred pass --
        # but a frame can be ALL transparency (every visible material
        # see-through, the whole picture in the A-buffer: the field's
        # 26.5-second frame was exactly this), and its LAYER plan must
        # still build. Empty mat_ids skips the opaque loop; the layer
        # loop below probes each see-through material over its own
        # triangles, no G-buffer pixels required. The old early return
        # here skipped the layer planning entirely, and the refusal
        # came out as the unnamed default.
        mat_ids = np.zeros(0, np.int64)

    from ..core import render as _CR
    consts = {
        'ambient_color': tuple(float(v) for v in LI.ambient_light(scene, st)),
        'two_sided': bool(getattr(st, 'two_sided_lighting', True)),
        'specular_in_gamma': bool(getattr(st, 'specular_in_gamma', True)),
        'clamp_specular': bool(getattr(st, 'clamp_specular', True)),
        'light_clamp': float(getattr(st, 'light_clamp', 0.0)),
        'falloff_default': str(getattr(st, 'light_falloff_default',
                                       'INVERSE_SQUARE')),
        'shadow_samples': int(getattr(st, 'shadow_samples', 4)),
        'tex_filter': str(getattr(st, 'tex_filter', 'NEAREST')),
        'tex_aniso': int(getattr(st, 'tex_aniso', 1) or 1),
        'tex_mip_bias': float(getattr(st, 'tex_mip_bias', 0.0) or 0.0),
        # per-object bounds for Generated coordinates: derived from the mesh,
        # which the plan signature already fingerprints
        'obj_bounds': job.object_bounds(),
        # the frame size, for coded shaders reading vScreenUV/iResolution
        'resolution': (float(job.width), float(job.height)),
        # the camera TYPE (already in the plan signature): the emitters'
        # perspective-only answers -- Geometry's Backfacing plane test --
        # gate on it, exactly as the backface override does
        'camera': str(getattr(scene.camera, 'type', 'PERSP')).upper(),
        # for vertex-rate passes: the corner-light texture indexes by
        # original triangle id, so its side bakes from the mesh's count
        # (the plan signature fingerprints the mesh already)
        'tri_count': int(scene.mesh.tris.shape[0])
        if getattr(scene.mesh, 'tris', None) is not None else 0,
        # the mesh's named UV layers, for the UV Map node's resolution
        'uv_names': tuple(getattr(scene.mesh, 'uv_names', None) or ()),
        # the BVH texture sides, baked into the traversal source when any
        # light shadows by ray; None otherwise
        'bvh_sides': bvh_sides,
        # deterministic sampling: the seed every hash stream mixes in, and
        # the AO spec when the frame occludes ambient light
        'seed': int(getattr(st, 'seed', 0) or 0),
        # the transparent-layer alpha chain's hard cutoff
        'alpha_threshold': float(getattr(st, 'alpha_threshold', 0.0)),
        'ao': {
            'samples': max(int(getattr(st, 'ao_samples', 8)), 1),
            'distance': float(getattr(st, 'ao_distance', 1.0)),
            'intensity': float(getattr(st, 'ao_intensity', 1.0)),
            'bias': max(float(getattr(st, 'ray_bias', 1e-3)), 1e-4),
        } if ao_on else None,
        # the one-bounce gather: samples/distance/intensity, the scene's
        # ambient colour for sky rays, and the flat-albedo table BOTH
        # devices bleed from (the CPU builds it; the shader bakes it)
        'radiosity': None if not rad_on else {
            'samples': max(int(getattr(st, 'radiosity_samples', 8)), 1),
            'distance': max(float(getattr(st, 'radiosity_distance', 3.0)),
                            1e-4),
            'intensity': float(getattr(st, 'radiosity_intensity', 1.0)),
            'bias': max(float(getattr(st, 'ray_bias', 1e-3)), 1e-4),
            'salt': _CR.RADIOSITY_SALT,
            'ambient': tuple(float(v)
                             for v in LI.ambient_light(scene, st)),
            'albedo': tuple(tuple(float(c) for c in row)
                            for row in _CR.radiosity_albedos(scene)),
            # the interpolated mode: gather every Nth pixel into a grid
            # PRE-PASS, blend in the material passes -- the CPU field's
            # exact twin (same sources, same identities, same rays)
            'spacing': max(int(getattr(st, 'radiosity_spacing', 1) or 1),
                           1),
            'grid': ((int(job.width) + max(int(getattr(
                st, 'radiosity_spacing', 1) or 1), 1) - 1)
                // max(int(getattr(st, 'radiosity_spacing', 1) or 1), 1),
                (int(job.height) + max(int(getattr(
                    st, 'radiosity_spacing', 1) or 1), 1) - 1)
                // max(int(getattr(st, 'radiosity_spacing', 1) or 1), 1)),
        },
        # projected light textures: per-light frame/size/strength for the
        # loop's GLSL, keyed by light index (empty dict = none in the frame)
        'cookies': cookies,
    }

    from .material import per_pixel_fields

    def _mat_name(mi):
        return scene.materials[mi].name if mi < len(scene.materials) \
            else f'material {mi}'

    def _env_for(bake):
        """The env spec for one material, or (None, why) when it refuses."""
        if float(bake.get('reflect', 0.0)) > 1e-4 and \
                getattr(st, 'env_reflection', True):
            return _env_world(job)
        return None, None

    def _one_material(mi, bake, model_idx, secondary=False, mid=False,
                      layer=False):
        """(entry, why): assemble one material's pass from its probe.

        `mid` builds the INTERMEDIATE-depth secondary variant: a hit
        that will spawn deeper rays shades with NO environment term (the
        traced child replaces it), exactly the CPU's d < D branch.
        `layer` builds the TRANSPARENT-LAYER variant: real alpha out."""
        mat = scene.materials[mi] if mi < len(scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        vrate = str(bake.get('__rate', 'PIXEL'))
        if vrate != 'PIXEL' and (secondary or layer):
            # the on-screen loop reuses the PRIMARY probe's bake for its
            # secondary entry, so the probe's own hit refusal never ran
            # for it -- gate here too, or a Gouraud material would get a
            # hit pass that interpolates camera-surface corner light
            # while the CPU lights every hit with the model's formula
            which = 'hit' if secondary else 'transparent-layer'
            return None, (f"'{_mat_name(mi)}' shades at {vrate} rate, "
                          f'and a {which} pass lights per pixel -- the '
                          'light loop has no formula for it')
        # under ray tracing the CPU never takes the env branch at depth 0
        # (the traced bounce replaces it); the depth-exhausted SECONDARY
        # shade is exactly where the env branch lives -- so only the
        # passes that will EMIT the env term get to refuse over the world.
        # A vertex-rate pass emits no env term at all: the CPU's corner
        # lighting already added it (by the renderer's own code, so ANY
        # world qualifies -- Bryce lab included), and the pass multiplies
        # the whole lit result by albedo exactly as the CPU does.
        env_spec = None
        if vrate == 'PIXEL' and ((secondary and not mid) or not ray_on):
            env_spec, env_why = _env_for(bake)
            if env_why is not None:
                return None, f"'{_mat_name(mi)}' reflects the world: " \
                             f'{env_why}'
            if env_spec is not None and env_spec[0] == 'CPU':
                if layer:
                    # the CPU-composite env adds at G-buffer pixels; a
                    # layer's fragments are not those. Narrow and named.
                    return None, f"'{_mat_name(mi)}': a rich world " \
                                 'behind transparent layers stays on ' \
                                 'the CPU'
                # a world richer than the baked GLSL paths: the pass
                # emits NO env term, and the composite adds the
                # renderer's own -- record this material's constants
                sc_env = (float(bake['reflect'])
                          * np.asarray(bake.get('specular', (1, 1, 1)),
                                       np.float32)
                          * np.asarray(bake.get('reflect_color',
                                                (1, 1, 1)), np.float32))
                cpu_env['hit' if secondary else 'primary'][mi] = \
                    tuple(float(x) for x in sc_env)
                env_spec = None
        c = dict(consts)
        c['env'] = env_spec
        src, info = assemble_frame(graph, mi, model_idx, bake, lights,
                                   c, shadows, textures=job.textures,
                                   programs=getattr(mat, 'programs', None)
                                   if mat is not None else None,
                                   secondary=secondary, layer=layer,
                                   vertex_rate=(vrate if vrate != 'PIXEL'
                                                else None))
        if src is None:
            kind = ' (as a reflection)' if secondary else \
                (' (as a transparent layer)' if layer else '')
            return None, f"'{_mat_name(mi)}'{kind}: {info}"
        return (mi, getattr(mat, 'name', f'mat{mi}') if mat else f'mat{mi}',
                src, info), None

    def _ray_gate(mi, bake, graph):
        """Why one material keeps a ray-traced frame off the GPU, or None."""
        # a master Normal chain no longer refuses: _ray_context evaluates
        # the bend with the CPU's own closure code for the ray pixels
        refracts = float(bake.get('opacity', 1.0)) < 0.999 and \
            getattr(st, 'ray_refraction', True) and job.bvh is not None
        if refracts:
            if 'ior' in per_pixel_fields(graph):
                return f"'{_mat_name(mi)}' refracts through a per-pixel " \
                       'IOR on the CPU'
            if 'diffuse_flat' not in bake:
                return f"'{_mat_name(mi)}' tints its refraction by a " \
                       'base colour that varies per pixel'
        if float(bake.get('reflect', 0.0)) > 1e-4:
            if 'specular' in per_pixel_fields(graph):
                return f"'{_mat_name(mi)}' scales its reflection by a " \
                       'per-pixel Specular Color on the CPU'
        return None

    def _collect_ray_terms(mi, bake):
        """The material's ray constants, exactly _add_raytraced's.

        One writer for all three loops -- the on-screen materials, the
        mesh-wide secondary loop, and the transparent-layer loop -- so a
        material's reflect scale and refraction lerp are the same
        numbers whichever surface a ray leaves from.
        """
        if not ray_on or job.bvh is None:
            return
        if getattr(st, 'ray_reflection', True) and \
                float(bake.get('reflect', 0.0)) > 1e-4:
            reflective[mi] = (
                float(bake['reflect'])
                * np.asarray(bake.get('specular', (1.0, 1.0, 1.0)),
                             np.float32)
                * np.asarray(bake.get('reflect_color', (1.0, 1.0, 1.0)),
                             np.float32))
        if getattr(st, 'ray_refraction', True) and \
                float(bake.get('opacity', 1.0)) < 0.999:
            op = min(max(float(bake['opacity']), 0.0), 1.0)
            refractive[mi] = {
                'k': (1.0 - op)
                * min(max(float(bake.get('refraction', 1.0)), 0.0), 1.0),
                'diffuse': tuple(float(v) for v in bake['diffuse_flat']),
                'ior': max(float(bake.get('ior', 1.45)), 1e-3),
            }

    passes = []
    secondary = []
    secondary_mid = []
    reflective = {}
    refractive = {}
    cpu_env = {'primary': {}, 'hit': {}}
    any_screen = False
    rng = np.random.default_rng(19)
    py, px = np.nonzero(covered)
    for mi in mat_ids:
        mi = int(mi)
        m = scene.mesh.mat_index[gbuf.tri[py, px]] \
            if scene.mesh.mat_index is not None else np.zeros(py.size, np.int32)
        mine = np.nonzero(m == mi)[0]
        if mine.size == 0:
            continue
        pick = mine if mine.size <= PROBE_FRAGMENTS else \
            mine[rng.choice(mine.size, PROBE_FRAGMENTS, replace=False)]
        bake, model_idx, why = _probe_material(job, gbuf, mi,
                                               py[pick], px[pick])
        if bake is None:
            return None, f"'{_mat_name(mi)}': {why}", {}
        if float(bake.get('backface_mix', 0.0)) > 1e-4 and \
                str(getattr(scene.camera, 'type', 'PERSP')).upper() != \
                'PERSP':
            # the shader decides backfacing with a plane-side test against
            # the eye, which is the rasteriser's answer only in perspective
            return None, f"'{_mat_name(mi)}': the backface override " \
                         'under an orthographic camera is not in the ' \
                         'deferred pass yet', {}
        mat = scene.materials[mi] if mi < len(scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        if ray_on:
            gate = _ray_gate(mi, bake, graph)
            if gate is not None:
                return None, gate, {}
        entry, why = _one_material(mi, bake, model_idx)
        if entry is None:
            return None, why, {}
        passes.append(entry)
        any_screen = any_screen or bool(entry[3].get('uses_screen'))
        # exactly _add_raytraced's k and tint, per material: the gate
        # above already held every constant this needs
        _collect_ray_terms(mi, bake)
        if ray_on:
            entry2, why2 = _one_material(mi, bake, model_idx,
                                         secondary=True)
            if entry2 is None:
                return None, why2, {}
            secondary.append(entry2)
            any_screen = any_screen or bool(entry2[3].get('uses_screen'))
            if ray_depth >= 2:
                entry3, why3 = _one_material(mi, bake, model_idx,
                                             secondary=True, mid=True)
                if entry3 is None:
                    return None, why3, {}
                secondary_mid.append(entry3)

    # transparent LAYERS: under SORTED/ABUFFER the A-buffer's fragments
    # shade per layer through the same machinery, with the REAL alpha
    # chain emitted. Under RAY TRACING the layers spawn the same
    # recursion the opaque frame does: their materials pass the ray gate
    # and their ray constants join the plan, so this runs BEFORE the ray
    # plan assembles -- the sweeps then run per layer over a virtual
    # surface in the executors. Best-effort throughout: a layer refusal
    # keeps the transparent shading on the CPU, named, without costing
    # the opaque plan.
    lwhy = None
    lpasses = []
    layer_bakes = []
    if str(getattr(st, 'transparency', 'NONE')) in ('SORTED', 'ABUFFER'):
        all_mats_l = np.unique(scene.mesh.mat_index) \
            if scene.mesh.mat_index is not None else np.zeros(1, np.int32)
        m_all = scene.mesh.mat_index[gbuf.tri[py, px]] \
            if scene.mesh.mat_index is not None \
            else np.zeros(py.size, np.int32)
        mats_l = scene.materials or []
        for mi in all_mats_l:
            mi = int(mi)
            # only a see-through material can rasterise into the
            # A-buffer -- `_split_by_alpha`'s own predicate -- so an
            # opaque material whose layer variant would refuse (a Bump
            # pre-pass, say) cannot cost the frame its GPU layers
            m_l = mats_l[mi] if 0 <= mi < len(mats_l) else None
            if m_l is not None and not (
                    float(getattr(m_l, 'opacity', 1.0)) < 0.999
                    or getattr(m_l, 'has_alpha', False)):
                continue
            mine_l = np.nonzero(m_all == mi)[0]
            try:
                if mine_l.size:
                    pick_l = mine_l if mine_l.size <= PROBE_FRAGMENTS \
                        else mine_l[rng.choice(mine_l.size,
                                               PROBE_FRAGMENTS,
                                               replace=False)]
                    bake, model_idx, why = _probe_material(
                        job, gbuf, mi, py[pick_l], px[pick_l],
                        layer=True)
                else:
                    tri_pool = np.nonzero(
                        scene.mesh.mat_index == mi)[0] \
                        if scene.mesh.mat_index is not None \
                        else np.arange(1)
                    pick_t = tri_pool \
                        if tri_pool.size <= PROBE_FRAGMENTS else \
                        tri_pool[rng.choice(tri_pool.size,
                                            PROBE_FRAGMENTS,
                                            replace=False)]
                    fb = np.full((pick_t.size, 3), 1.0 / 3.0,
                                 np.float32)
                    bake, model_idx, why = _probe_material(
                        job, gbuf, mi, None, None,
                        frags=(pick_t.astype(np.int32), fb),
                        layer=True)
            except Exception as exc:                            # noqa: BLE001
                bake, model_idx = None, None
                why = f'probing failed ({exc})'
            if bake is None:
                lwhy = f"'{_mat_name(mi)}' (as a transparent " \
                       f'layer): {why}'
                break
            if ray_on:
                # the same constants-must-hold gate the opaque loop
                # runs: a layer fragment's rays lerp and scale by
                # per-material constants
                gate = _ray_gate(mi, bake, getattr(m_l, 'graph', None)
                                 if m_l is not None else None)
                if gate is not None:
                    lwhy = gate
                    break
            entry, whyL = _one_material(mi, bake, model_idx,
                                        layer=True)
            if entry is None:
                lwhy = whyL
                break
            lpasses.append(entry)
            layer_bakes.append((mi, bake))
    if lwhy is None and lpasses:
        atlases['__layers'] = lpasses
        # the layers' own ray terms open and join the ray plan below:
        # an all-glass frame has NO opaque material to open it
        for mi_l, bake_l in layer_bakes:
            _collect_ray_terms(mi_l, bake_l)
    elif lwhy is not None:
        atlases['__layers_why'] = lwhy

    if rad_on and consts['radiosity']['spacing'] > 1:
        # the interpolated gather's grid pre-pass: one source, both
        # executors -- the driver draws it into a grid-sized target,
        # the simulator runs it over grid lanes, and every material
        # pass binds the result as hal_radfield
        from .material import radiosity_field_pass
        atlases['__radfield'] = radiosity_field_pass(
            consts['radiosity'], consts, bvh_sides or {})

    rplan = None
    if ray_on and (reflective or refractive):
        # a secondary ray can hit a material with no pixel on screen, so
        # every material the MESH carries needs a secondary pass -- probed
        # over its own triangles, since the frame has none of its fragments
        seen = {p[0] for p in secondary}
        all_mats = np.unique(scene.mesh.mat_index) \
            if scene.mesh.mat_index is not None else np.zeros(1, np.int32)
        for mi in all_mats:
            mi = int(mi)
            if mi in seen:
                continue
            tri_pool = np.nonzero(scene.mesh.mat_index == mi)[0] \
                if scene.mesh.mat_index is not None else np.arange(1)
            pick_t = tri_pool if tri_pool.size <= PROBE_FRAGMENTS else \
                tri_pool[rng.choice(tri_pool.size, PROBE_FRAGMENTS,
                                    replace=False)]
            # FULL three-component barycentrics: gbuf.bary is (H,W,3) and
            # raster.fetch requires (N,3) -- the field found the 2-wide
            # version with a broadcast crash on the first scene that had a
            # material visible only in reflections
            frag_bary = np.full((pick_t.size, 3), 1.0 / 3.0, np.float32)
            try:
                bake, model_idx, why = _probe_material(
                    job, gbuf, mi, None, None,
                    frags=(pick_t.astype(np.int32), frag_bary),
                    secondary=True)
            except Exception as exc:                            # noqa: BLE001
                bake, model_idx = None, None
                why = f'probing it off-screen failed ({exc})'
            if bake is None:
                return None, f"'{_mat_name(mi)}' (visible only in " \
                             f'reflections): {why}', {}
            gate = _ray_gate(mi, bake, getattr(
                scene.materials[mi], 'graph', None)
                if mi < len(scene.materials) else None)
            if gate is not None:
                return None, gate, {}
            # a hidden material's constants matter at depth >= 2: a ray
            # can hit it, and ITS hits spawn the next level with ITS
            # reflect scale and refraction lerp
            _collect_ray_terms(mi, bake)
            entry2, why2 = _one_material(mi, bake, model_idx,
                                         secondary=True)
            if entry2 is None:
                return None, why2, {}
            secondary.append(entry2)
            any_screen = any_screen or bool(entry2[3].get('uses_screen'))
            if ray_depth >= 2:
                entry3, why3 = _one_material(mi, bake, model_idx,
                                             secondary=True, mid=True)
                if entry3 is None:
                    return None, why3, {}
                secondary_mid.append(entry3)
        if any_screen:
            # ctx.px is None for hit points on the CPU: screen-space
            # inputs have no honest value there, on either device
            return None, 'screen-space shader inputs shade reflection ' \
                         'hits on the CPU (a hit point has no screen ' \
                         'position)', {}
        rplan = {'secondary': secondary,
                 'secondary_mid': secondary_mid,
                 'depth': int(ray_depth),
                 'scale': {mi: tuple(float(x) for x in sc)
                           for mi, sc in reflective.items()},
                 'reflective': sorted(reflective),
                 'refract': refractive,
                 'refractive': sorted(refractive),
                 'bias': float(getattr(st, 'ray_bias', 1e-3))}
        atlases['__reflect'] = rplan

    if cpu_env['primary'] or cpu_env['hit']:
        atlases['__env'] = cpu_env

    if sig is not None:
        if len(_PLAN_CACHE) > 4:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[sig] = (passes, None, atlases)
    return passes, None, atlases


def _textures(job, gbuf):
    """The three packed G-buffer textures and their sizes."""
    from . import gbuffer as GB
    ids = GB.pack_ids(gbuf)
    attrs, side = GB.pack_attributes(job.scene.mesh, respect_smooth=True)
    tris, tside = GB.pack_tri_data(job.scene.mesh)
    return ids, attrs, side, tris, tside


def _gather_pass_textures(bind_dicts, job_textures, up):
    """Driver textures for every pass's samplers, keyed by (PASS, name).

    Sampler names are POSITIONAL per material shader ('hal_tex0',
    'hal_tex1'...), so the frame-wide by-name map this replaces handed
    every material the FIRST material's texture -- "materials are all
    using the same one when they should be different", said the field,
    exactly, on any scene with two image-textured materials. The
    compiler sim binds per pass by design and could never see it: the
    ONE divergence between sim and driver was the bug. Keying by
    (id(pass binds), sampler) makes the driver bind per pass too; the
    content-keyed upload cache underneath still deduplicates the actual
    uploads, so a shared image costs one GPU texture either way.
    """
    out = {}
    for binds in bind_dicts:
        for sname, key in (binds.get('textures') or {}).items():
            if (id(binds), sname) in out:
                continue
            tx = job_textures[key]
            ik = ('img', key, tx.width, tx.height,
                  round(float(tx.pixels[::13].sum()), 3))
            out[(id(binds), sname)] = up(ik, lambda _t=tx: _t.pixels)
        # mip atlases: the CPU's OWN build_mips output, packed as a
        # vertical stack -- the driver filters the very texels the CPU
        # filters
        for sname, key in (binds.get('textures_mip') or {}).items():
            if (id(binds), sname) in out:
                continue
            tx = job_textures[key]
            mk = ('mipatlas', key, tx.width, tx.height,
                  round(float(tx.pixels[::13].sum()), 3))
            from .material import mip_atlas as _mip_atlas
            out[(id(binds), sname)] = up(mk, lambda _t=tx: _mip_atlas(_t)[0])
    return out


def _mesh_key(mesh):
    """A cheap content fingerprint of the mesh's packable attributes.

    The attribute and triangle-data textures depend only on the mesh, and a
    mesh holds still for most of an animation -- but a fresh export makes
    fresh arrays, so identity is useless as a key. A strided sum is not: it
    costs microseconds and changes when the data does.
    """
    v = mesh.verts
    stride = max(1, v.shape[0] // 512)
    parts = [int(v.shape[0]), int(mesh.tris.shape[0]),
             round(float(v[::stride].sum()), 3)]
    nrm = getattr(mesh, 'normals', None)
    if nrm is not None:
        parts.append(round(float(np.asarray(nrm)[::stride].sum()), 3))
    uvs = getattr(mesh, 'uvs', None)
    if uvs is not None:
        parts.append(round(float(np.asarray(uvs)[::stride].sum()), 3))
    uvs2 = getattr(mesh, 'uvs2', None)
    if uvs2 is not None:
        # the second UV set rides the attribute texture too: a changed
        # layer must miss the upload cache exactly like the first
        parts.append(round(float(np.asarray(uvs2)[::stride].sum()), 3))
    mats = getattr(mesh, 'mat_index', None)
    if mats is not None:
        parts.append(int(np.asarray(mats)[::stride].sum()))
    return tuple(parts)


#: where the last driver frame's milliseconds went, for the self-test
LAST_TIMINGS = {}

#: where the last transparent-layer frame's milliseconds went -- the
#: printed split that lets a field paste name the next perf target
LAST_LAYER_TIMINGS = {}

#: scissor the per-layer passes and readbacks to each depth layer's own
#: bounding box. Module-level so the self test can A/B it against the
#: proven full-frame path on the driver itself; the Debug toggle
#: (settings.gpu_scissor) is the field's switch. Both must be on for
#: regions to apply.
REGION_DRAWS = True

#: one row resets a fragment's ids texel: bary zeroed, triangle -1
_IDS_CLEAR = np.array([0.0, 0.0, 0.0, -1.0], np.float32)


def _mat_eval(job, gbuf, mi):
    """(py, px, ctx, ev): one material's frame pixels, context and
    evaluator -- built ONCE per frame, shared by every CPU slice.

    Three consumers used to each build their own: the height image, the
    reflection rays and the refraction rays -- for the water that meant
    three context builds and three runs of the noise chain per frame.
    The context and the evaluator are pure functions of (frame,
    material), so they live on the job now (one job is one frame),
    keyed by the G-buffer's identity. Sharing the EVALUATOR is the
    point: its per-node cache means the noise heights evaluate once and
    every later ask -- the bend, the pre-pass image -- is a lookup.
    `ev` is None for a material with no node graph.
    """
    import time as _time
    from ..core.nodeeval import GraphEvaluator
    cache = getattr(job, '_hal_mat_eval', None)
    if cache is None or cache.get('__gbuf') is not gbuf:
        cache = {'__gbuf': gbuf}
        job._hal_mat_eval = cache
    mi = int(mi)
    got = cache.get(mi)
    if got is not None:
        return got
    t0 = _time.perf_counter()
    mesh = job.scene.mesh
    covered = gbuf.tri >= 0
    m = np.where(covered, mesh.mat_index[gbuf.tri], -1) \
        if mesh.mat_index is not None else \
        np.where(covered, 0, -1)
    py, px = np.nonzero(m == mi)
    if py.size == 0:
        got = (py, px, None, None)
    else:
        ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py,
                          np.ones(py.size, bool), None, 0, True)
        mat = job.scene.materials[mi] \
            if mi < len(job.scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        ev = GraphEvaluator(graph, ctx, job.textures,
                            getattr(mat, 'programs', None)) \
            if graph else None
        got = (py, px, ctx, ev)
    cache[mi] = got
    _RAY_BUILD[0] += (_time.perf_counter() - t0) * 1000.0
    return got


def _cpu_height_image(job, gbuf, mat_id, node_id):
    """A Bump height chain, evaluated by the renderer itself, as an image.

    The pre-pass is only a picture of the height over the frame -- so a
    chain the GLSL emitter refuses (Blender's sin-fract Noise family above
    all) does not have to refuse the material: the CPU evaluator produces
    the image with its own float64 arithmetic, exactly, and the GPU takes
    its neighbour differences from that. (h, 0, 0, keep), zeros outside
    the material's pixels, exactly the GPU pre-pass's output shape. The
    evaluator comes from the frame's shared per-material cache, so when
    the ray sweeps already ran the chain (or will), the heights are
    computed once.
    """
    from ..core.nodeeval import VALUE
    h, w = gbuf.tri.shape
    img = np.zeros((h, w, 4), np.float32)
    py, px, _ctx, ev = _mat_eval(job, gbuf, mat_id)
    if py.size == 0 or ev is None:
        return img
    mat = job.scene.materials[mat_id] \
        if mat_id < len(job.scene.materials) else None
    graph = getattr(mat, 'graph', None) if mat is not None else None
    node = (graph or {}).get('nodes', {}).get(node_id)
    hval = np.asarray(ev.input(node, 'Height', VALUE),
                      np.float32).reshape(-1)
    img[py, px, 0] = hval
    img[py, px, 3] = 1.0
    return img


def _reflective_mask(job, gbuf, rplan):
    """Which pixels want a reflection ray: covered, and reflective."""
    mesh = job.scene.mesh
    covered = gbuf.tri >= 0
    if mesh.mat_index is None:
        want = 0 in rplan['reflective']
        return covered if want else np.zeros_like(covered)
    mat_pix = np.where(covered, mesh.mat_index[gbuf.tri], -1)
    return np.isin(mat_pix, np.asarray(rplan['reflective'], np.int64))


def _ray_context(job, gbuf, py, px):
    """(ctx, N): the ray pixels' context and the normal the rays bend off.

    N is `normalize(ctx.N)` AFTER the master Normal chain -- evaluated
    with the CPU's own closure code (`GraphEvaluator` through
    `closure_to_surface`, the same calls `shade_batch` makes) for exactly
    the materials that bend, so a normal-mapped water builds the same
    rays on either device, by construction. This is CPU work proportional
    to the RAY count, not the frame: the one slice of a ray-traced frame
    that still runs the evaluator, and the price of exactness until a
    bend pre-pass earns its way in with a zero of its own.
    """
    from ..core import mathx as M
    from ..core.nodeeval import GraphEvaluator
    from ..core.render import closure_to_surface
    from .material import master_normal_linked

    ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py,
                      np.ones(py.size, bool), None, 0, True)
    N = M.normalize(ctx.N)
    mesh = job.scene.mesh
    m = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
        else np.zeros(py.size, np.int32)
    for mi in np.unique(m):
        mat = job.scene.materials[int(mi)] \
            if int(mi) < len(job.scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        if not master_normal_linked(graph):
            continue
        sel = np.nonzero(m == mi)[0]
        sub = job.context(gbuf.tri[py[sel], px[sel]],
                          gbuf.bary[py[sel], px[sel]], px[sel], py[sel],
                          np.ones(sel.size, bool), None, 0, True)
        ev = GraphEvaluator(graph, sub, job.textures,
                            getattr(mat, 'programs', None))
        cl, _disp = ev.evaluate_surface()
        _surf, _model, nrm = closure_to_surface(cl, sub, job.settings, mat)
        if nrm is not None:
            N[sel] = M.normalize(nrm)
    return ctx, N


#: milliseconds spent building ray blocks (context + bent normals) since
#: the last reset -- the sweep loop reads it into LAST_TIMINGS so the
#: self-test can show how much of 'sweeps' is CPU-side ray construction
_RAY_BUILD = [0.0]


def _ray_blocks(job, gbuf, mats):
    """(py, px, P, I, N) for the pixels of `mats`, cached per material.

    The expensive halves of ray building -- `job.context` over the ray
    pixels and the evaluator run that bends the normal -- are pure
    functions of (frame, material). The water REFLECTS and REFRACTS, so
    both sweeps used to pay them over the same pixels; now each material
    pays once per frame and every caller gathers from the cache. The
    cache lives on the job (one job is one frame) and is keyed by the
    G-buffer's identity, so a re-rasterised frame never reuses stale
    pixels. Values are EXACTLY `_ray_context`'s: the per-material blocks
    are the same per-pixel arithmetic the mixed selection ran, and every
    consumer scatters by (py, px), so block order cannot change a pixel.
    """
    import time as _time
    from ..core import mathx as M
    from ..core.render import closure_to_surface
    from .material import master_normal_linked
    cache = getattr(job, '_hal_ray_blocks', None)
    if cache is None or cache.get('__gbuf') is not gbuf:
        cache = {'__gbuf': gbuf}
        job._hal_ray_blocks = cache
    blocks = []
    for mi in mats:
        mi = int(mi)
        blk = cache.get(mi)
        if blk is None:
            py, px, ctx, ev = _mat_eval(job, gbuf, mi)
            if py.size == 0:
                blk = (py, px, None, None, None)
            else:
                t0 = _time.perf_counter()
                # exactly _ray_context's bend: the master Normal chain
                # through the CPU's own closure code where one exists,
                # normalize(ctx.N) otherwise -- run through the SHARED
                # evaluator, so its node cache (the noise heights above
                # all) serves the pre-pass image too
                N = M.normalize(ctx.N)
                mat = job.scene.materials[mi] \
                    if mi < len(job.scene.materials) else None
                graph = getattr(mat, 'graph', None) \
                    if mat is not None else None
                if ev is not None and master_normal_linked(graph):
                    cl, _disp = ev.evaluate_surface()
                    _surf, _model, nrm = closure_to_surface(
                        cl, ctx, job.settings, mat)
                    if nrm is not None:
                        N = M.normalize(nrm)
                blk = (py, px, np.asarray(ctx.P), np.asarray(ctx.I), N)
                _RAY_BUILD[0] += (_time.perf_counter() - t0) * 1000.0
            cache[mi] = blk
        blocks.append(blk)
    blocks = [b for b in blocks if b[0].size]
    if not blocks:
        z = np.zeros(0, np.int64)
        return z, z, None, None, None
    return (np.concatenate([b[0] for b in blocks]),
            np.concatenate([b[1] for b in blocks]),
            np.concatenate([b[2] for b in blocks]),
            np.concatenate([b[3] for b in blocks]),
            np.concatenate([b[4] for b in blocks]))


def _reflection_rays(job, gbuf, rplan):
    """(py, px, org, dirs): exactly `_add_raytraced`'s ray construction.

    N is the SHADING normal -- bent by the master Normal chain where one
    exists, UNFLIPPED -- V looks at the camera, the origin steps off the
    surface by the RAW ray bias (no floor, unlike the shadow branch), and
    the directions mirror V about N.
    """
    from ..core import mathx as M
    py, px, P, I, N = _ray_blocks(job, gbuf, rplan['reflective'])
    if py.size == 0:
        return py, px, None, None
    V = -M.normalize(I)
    dirs = M.reflect(-V, N).astype(np.float32)
    org = (P + N * rplan['bias']).astype(np.float32)
    return py, px, org, dirs


def _secondary_ids(h, w, py, px, tid, u, v):
    """The hit points as an ids texture, aligned with the primary pixels.

    trace() fetches with bary [1-u-v, u, v]; pack_ids stores (b0, b1, b2,
    tri), so b0 = 1-u-v, b1 = u, b2 = v. Missed and non-reflective pixels
    keep tri -1 -- uncovered, so no secondary pass writes there.
    """
    sec = np.zeros((h, w, 4), np.float32)
    sec[:, :, 3] = -1.0
    hit = tid >= 0
    if hit.any():
        sec[py[hit], px[hit], 0] = 1.0 - u[hit] - v[hit]
        sec[py[hit], px[hit], 1] = u[hit]
        sec[py[hit], px[hit], 2] = v[hit]
        sec[py[hit], px[hit], 3] = tid[hit].astype(np.float32)
    return sec, hit


def _composite_reflections(job, gbuf, rplan, out, py, px, dirs, hit,
                           sec_img):
    """rgb += hit_colour * reflect * specular * reflect_color, in place.

    Hits take the secondary pass's colour; misses take `world_color` for
    the ray direction -- computed with the renderer's own function, so ANY
    world is exact here, however rich. The scale is a per-material
    constant (per-pixel specular was refused at plan time).
    """
    from ..core.render import world_color
    add = np.zeros((py.size, 3), np.float32)
    if hit.any():
        add[hit] = sec_img[py[hit], px[hit], :3]
    miss = ~hit
    if miss.any():
        wc = world_color(job.scene, job.settings, dirs[miss], job.textures,
                         int(miss.sum()), eye=job.eye)
        add[miss] = np.asarray(wc, np.float32)[:, :3]
    mesh = job.scene.mesh
    m = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
        else np.zeros(py.size, np.int32)
    scale = np.zeros((py.size, 3), np.float32)
    for mi, sc in rplan['scale'].items():
        scale[m == mi] = np.asarray(sc, np.float32)
    out[py, px] = out[py, px] + add * scale
    return out


def _no_layer_plan_why(job, atlases, tri):
    """The reason there is no layer plan, NAMED -- never the bare default.

    '__layers_why' carries a planning refusal verbatim. With neither key
    present, the plan found no see-through material at all -- yet the
    caller holds fragments, so the transparent subset and the layer
    predicate disagree. Name the first fragment material and what the
    predicate read from it, so the field console says the mechanism."""
    why = (atlases or {}).get('__layers_why')
    if why is not None:
        return why
    mats = job.scene.materials or []
    mat_index = getattr(job.scene.mesh, 'mat_index', None)
    if mat_index is not None and mat_index.size and tri.size:
        mi = int(mat_index[int(tri[0])])
        m = mats[mi] if 0 <= mi < len(mats) else None
        name = getattr(m, 'name', f'mat{mi}') if m is not None else f'mat{mi}'
        return (f"the layer plan is empty, yet fragments arrived from "
                f"'{name}' (opacity "
                f"{float(getattr(m, 'opacity', 1.0)):.3f}, has_alpha "
                f'{bool(getattr(m, "has_alpha", False))} read as opaque '
                'to the layer predicate)')
    return 'the layer plan is empty and no fragment names a material'


def _layer_coverage(job, lpasses, tri):
    """Why some A-buffer fragment has NO layer pass, or None.

    The plan probes the materials `_split_by_alpha`'s predicate says can
    rasterise transparent fragments. This is the check that the mirror
    held for the actual fragments -- a discrepancy refuses by name
    instead of shading those fragments to nothing."""
    mat_index = getattr(job.scene.mesh, 'mat_index', None)
    fmats = (np.unique(mat_index[np.clip(tri, 0, mat_index.size - 1)])
             if mat_index is not None and mat_index.size else
             np.zeros(1, np.int64))
    have = {int(e[0]) for e in lpasses}
    for m in fmats:
        if int(m) not in have:
            mats = job.scene.materials or []
            name = getattr(mats[int(m)], 'name', f'mat{int(m)}') \
                if 0 <= int(m) < len(mats) else f'mat{int(m)}'
            return f"'{name}' rasterised transparent fragments but has " \
                   'no layer pass'
    return None


def _fragment_ids(h, w, py, px, tri, bary):
    """One transparency layer as an ids texture: (b0, b1, b2, tri).

    Exactly `_secondary_ids`' shape, from A-buffer fragments instead of
    ray hits -- the layer's pixels carry their own triangle and REAL
    barycentrics, everything else stays uncovered (tri -1)."""
    sec = np.zeros((h, w, 4), np.float32)
    sec[:, :, 3] = -1.0
    sec[py, px, 0] = bary[:, 0]
    sec[py, px, 1] = bary[:, 1]
    sec[py, px, 2] = bary[:, 2]
    sec[py, px, 3] = tri.astype(np.float32)
    return sec


def _layer_gbuf(h, w, py, px, tri, bary):
    """One layer's fragments as a virtual G-buffer surface.

    The ray machinery reads exactly two things from a surface -- `.tri`
    and `.bary` (the context builder takes the pixels' coordinates
    alongside) -- so a rank's fragments scattered into a GBuffer make a
    PRIMARY surface the sweeps can spawn rays from, with the fragment
    pixel as the sampling identity, exactly the CPU's."""
    from ..core import raster as CR
    vg = CR.GBuffer(w, h)
    vg.tri[py, px] = tri
    vg.bary[py, px] = bary
    return vg


def shade_fragments_frame(job, gbuf, tri, bary, px, py, rank):
    """A-buffer fragments shaded by the driver, layer by layer.

    Each depth layer's fragments become an ids texture; every
    see-through material's LAYER pass (real alpha out) draws over it,
    the per-pixel-disjoint materials merge under the proven blend, and
    -- under ray tracing -- the layer's fragments then spawn the SAME
    recursion the opaque frame runs, through `_run_sweeps` over a
    virtual surface built from the rank's own triangles and
    barycentrics. The gather returns per-fragment RGBA in the caller's
    order: rgb from the composited image, alpha from the layer pass.
    Returns (col (N,4), why): why non-None means the caller shades on
    the CPU as before. `LAST_LAYER_TIMINGS` holds where the
    milliseconds went, for the printed split.
    """
    import time as _time

    from . import device
    from . import marshal as _marshal

    _marshal.acct_reset()
    _RAY_BUILD[0] = 0.0
    # DISJOINT buckets: a millisecond lands in exactly one, and
    # `other_ms` is the honest remainder (total minus every bucket) --
    # the field's 1.25.59 split hid ~1.8 s of worker-side NumPy
    # (rank scans, scatters, gathers, copies) in exactly that gap, and
    # a remainder nobody prints is a cost nobody attacks
    tm = {'plan_ms': 0.0, 'compile_ms': 0.0, 'upload_ms': 0.0,
          'upload_mb': 0.0, 'draw_ms': 0.0, 'read_ms': 0.0,
          'sweep_ms': 0.0, 'other_ms': 0.0, 'total_ms': 0.0, 'ranks': 0,
          'scissor_px': 0.0, 'frame_px': 0.0}
    LAST_LAYER_TIMINGS.clear()
    t_all = _time.perf_counter()

    ok, why = device.probe()
    if not ok:
        return None, why
    t0 = _time.perf_counter()
    passes, why, atlases = plan_frame(job, gbuf)
    tm['plan_ms'] = (_time.perf_counter() - t0) * 1000.0
    if passes is None:
        return None, why
    lpasses = (atlases or {}).get('__layers')
    if not lpasses:
        return None, _no_layer_plan_why(job, atlases, tri)
    why = _layer_coverage(job, lpasses, tri)
    if why is not None:
        return None, why
    rplan = (atlases or {}).get('__reflect')
    env_plan = (atlases or {}).get('__env')
    h, w = gbuf.tri.shape
    from . import gbuffer as GB
    mesh = job.scene.mesh
    mkey = _mesh_key(mesh)
    holder = {}

    def build_attrs():
        arr, sd = GB.pack_attributes(mesh, respect_smooth=True)
        holder['side'] = sd
        return arr

    def build_tris():
        arr, sd = GB.pack_tri_data(mesh)
        holder['tside'] = sd
        return arr

    sec_lists = []
    if rplan is not None:
        sec_lists = list(rplan.get('secondary') or ()) \
            + list(rplan.get('secondary_mid') or ())
    t0 = _time.perf_counter()
    try:
        tex_attrs = device.upload_cached(('gb_attrs',) + mkey, build_attrs)
        tex_tris = device.upload_cached(('gb_tris',) + mkey, build_tris)
        tex_shadows = {sname: device.upload_cached(entry[0], entry[1])
                       for sname, entry in atlases.items()
                       if not sname.startswith('__')}
        srcs_all = []
        for _mi, _n, _s, binds in list(lpasses) + sec_lists:
            srcs_all.append(binds)
            srcs_all.extend(p[2] for p in (binds.get('prepasses') or ()))
        tex_images = _gather_pass_textures(srcs_all, job.textures,
                                           device.upload_cached)
    except Exception as exc:                                    # noqa: BLE001
        return None, f'uploading the layer textures failed: {exc}'
    tm['upload_ms'] += (_time.perf_counter() - t0) * 1000.0
    side = holder.get('side', int(tex_attrs.width))
    tside = holder.get('tside', int(tex_tris.width))
    uni = {'hal_attr_side': float(side), 'hal_slot_count': 4.0,
           'hal_tri_side': float(tside),
           'hal_eye': tuple(float(v) for v in job.eye)}

    radfield = (atlases or {}).get('__radfield')
    if radfield is not None:
        # layer fragments read the interpolated field like frame pixels
        # do (a transparent surface is still a SCREEN pixel). The grid
        # pass draws over the OPAQUE frame's ids; the executor reads the
        # small grid back and re-uploads it as a plain texture, so no
        # target outlives this block -- pure transport, tiny (the grid).
        try:
            ids_f, _a2, _s2, _t2, _ts2 = _textures(job, gbuf)
            tex_ids_f = device.upload(ids_f)
            rsrc, rbinds = radfield
            rspec = {'samplers': list(rbinds.get('samplers', ())),
                     'floats': ['hal_attr_side', 'hal_slot_count',
                                'hal_tri_side'],
                     'vec3': ['hal_eye']}
            rshader, rerr = device.compile_dynamic('HAL_RADFIELD', rsrc,
                                                   rspec)
            if rshader is None:
                return None, f'the driver rejected the radiosity grid ' \
                             f'pass (layers): {rerr}'
            rbind = {'hal_gb_ids': tex_ids_f, 'hal_gb_attrs': tex_attrs,
                     'hal_gb_tris': tex_tris}
            for sname in rbinds.get('samplers', ()):
                if sname not in rbind:
                    if sname not in tex_shadows:
                        return None, f'the radiosity grid pass wants ' \
                                     f'{sname} but nothing was packed'
                    rbind[sname] = tex_shadows[sname]
            gw_r, gh_r = rbinds['size']
            rtgt = device.Target(int(gw_r), int(gh_r))
            try:
                grid = device.draw_fullscreen(rshader, uni, rbind, rtgt,
                                              read=True, blend='NONE',
                                              clear=True)
            finally:
                rtgt.free()
            tex_shadows['hal_radfield'] = device.upload(
                np.asarray(grid, np.float32))
        except Exception as exc:                                # noqa: BLE001
            return None, f'the radiosity grid pass failed (layers): {exc}'

    built = []
    prepass_built = {}          # (mat_id, uname) -> ('gpu', shader, pbinds)
    #                             or ('cpu', node_id)
    t_compile = _time.perf_counter()
    for mat_id, name, src, binds in lpasses:
        prepasses = binds.get('prepasses') or ()
        pre_unames = {p[0] for p in prepasses}
        spec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris']
                + list(binds.get('samplers', ())),
                'floats': ['hal_attr_side', 'hal_slot_count',
                           'hal_tri_side']
                + list(binds.get('frame_uniforms', ())),
                'vec3': ['hal_eye']}
        shader, err = device.compile_dynamic(f'HAL_TMAT_{mat_id}', src,
                                             spec)
        if shader is None:
            return None, f"the driver rejected '{name}' (layer): {err}"
        bind = {'hal_gb_attrs': tex_attrs, 'hal_gb_tris': tex_tris}
        for sname in binds.get('samplers', ()):
            if sname in pre_unames:
                continue            # a height image, drawn PER RANK below
            if sname in tex_shadows:
                bind[sname] = tex_shadows[sname]
            elif (id(binds), sname) in tex_images:
                bind[sname] = tex_images[(id(binds), sname)]
            else:
                return None, f"'{name}' (layer) wants {sname} but " \
                             'nothing was packed for it'
        extra = {}
        for u in binds.get('frame_uniforms', ()):
            if u == 'hal_time':
                extra[u] = float(getattr(job.scene, 'time', 0.0))
            elif u == 'hal_frame':
                extra[u] = float(getattr(job.scene, 'frame', 0))
        # the Bump height pre-passes: the same shaders the opaque frame
        # compiles (same names, same sources -- cache hits), drawn per
        # RANK below so the neighbour differences ride each layer's own
        # surface, exactly the CPU's per-rank gradient fields
        for uname, psrc, pbinds in prepasses:
            if pbinds.get('cpu'):
                prepass_built[(mat_id, uname)] = ('cpu', pbinds['node'],
                                                  None)
                continue
            pspec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs',
                                  'hal_gb_tris']
                     + list(pbinds.get('samplers', ())),
                     'floats': ['hal_attr_side', 'hal_slot_count',
                                'hal_tri_side']
                     + list(pbinds.get('frame_uniforms', ())),
                     'vec3': ['hal_eye']}
            pshader, perr = device.compile_dynamic(
                f'HAL_BUMP_{mat_id}_{uname}', psrc, pspec)
            if pshader is None:
                return None, f"the driver rejected '{name}' height " \
                             f'pass: {perr}'
            pbind = {'hal_gb_attrs': tex_attrs, 'hal_gb_tris': tex_tris}
            for sname in pbinds.get('samplers', ()):
                if (id(pbinds), sname) in tex_images:
                    pbind[sname] = tex_images[(id(pbinds), sname)]
                elif sname in tex_shadows:
                    pbind[sname] = tex_shadows[sname]
                else:
                    return None, f"'{name}' height pass wants {sname} " \
                                 'but nothing was packed for it'
            prepass_built[(mat_id, uname)] = ('gpu', pshader, pbind)
        built.append((mat_id, name, shader, bind, extra, prepasses))

    # the secondary (hit) passes, compiled ONCE for every layer's sweeps
    # -- the same names and sources shade_frame compiles, so a frame that
    # already ray-traced its opaque half pays nothing here
    sec_draws = {'secondary': [], 'secondary_mid': []}
    if rplan is not None:
        for which2 in ('secondary', 'secondary_mid'):
            tag2 = 'HAL_RMAT' if which2 == 'secondary' else 'HAL_RMATM'
            for mat_id, name, src, binds in (rplan.get(which2) or ()):
                spec2 = {'samplers': ['hal_gb_ids', 'hal_gb_attrs',
                                      'hal_gb_tris']
                         + list(binds.get('samplers', ())),
                         'floats': ['hal_attr_side', 'hal_slot_count',
                                    'hal_tri_side']
                         + list(binds.get('frame_uniforms', ())),
                         'vec3': ['hal_eye']}
                shader, err = device.compile_dynamic(f'{tag2}_{mat_id}',
                                                     src, spec2)
                if shader is None:
                    return None, f"the driver rejected '{name}' " \
                                 f'(hit): {err}'
                bind = {'hal_gb_attrs': tex_attrs,
                        'hal_gb_tris': tex_tris}
                for sname in binds.get('samplers', ()):
                    if sname in tex_shadows:
                        bind[sname] = tex_shadows[sname]
                    elif (id(binds), sname) in tex_images:
                        bind[sname] = tex_images[(id(binds), sname)]
                    else:
                        return None, f"'{name}' (hit) wants {sname} " \
                                     'but nothing was packed for it'
                extra = {}
                for u in binds.get('frame_uniforms', ()):
                    if u == 'hal_time':
                        extra[u] = float(getattr(job.scene, 'time', 0.0))
                    elif u == 'hal_frame':
                        extra[u] = float(getattr(job.scene, 'frame', 0))
                sec_draws[which2].append((name, shader, bind, extra))
    tm['compile_ms'] = (_time.perf_counter() - t_compile) * 1000.0

    out = np.zeros((tri.size, 4), np.float32)
    top = int(rank.max()) if rank.size else -1
    need_vg = rplan is not None or any(
        kind == 'cpu' for kind, _a, _b in prepass_built.values())
    # a frame 16 layers deep pays the per-rank loop 16 times, and its
    # DEEP ranks are sparse -- the field split showed ~2s of nothing
    # but fresh 50 MB buffers. One ids buffer and one virtual surface
    # live for the whole loop; each rank scatters its fragments in and
    # scatters them back out (a reset proportional to the RANK, not the
    # frame). The evaluator caches key on surface identity, so reusing
    # the object gets FRESH cache dicts per rank instead.
    ids = np.zeros((h, w, 4), np.float32)
    ids[:, :, 3] = -1.0
    vg = None
    if need_vg:
        from ..core import raster as CR
        vg = CR.GBuffer(w, h)
    # one stable sort instead of a full `rank == r` scan per layer: a
    # 16-deep frame paid sixteen passes over every fragment in the
    # frame just to FIND each layer. Stable argsort keeps equal ranks
    # in original order, so each slice is bit-identical to nonzero's
    # ascending indices -- the same pattern the compositor's own layer
    # loop has always used.
    rorder = np.argsort(rank, kind='stable')
    rbounds = np.searchsorted(rank[rorder], np.arange(top + 2))
    # targets live for the WHOLE loop: the 1.25.60 field frame allocated
    # and freed a ~50 MB offscreen per layer (plus one per sweep level
    # and one per height pre-pass) -- driver allocations at that size
    # are milliseconds each, sixteen layers deep. Every pass full-clears
    # before its first draw, so a reused target is semantically a fresh
    # one; every rank draws at least one pass (coverage guaranteed it),
    # so a read can never see a stale layer.
    pools = {'layer': None, 'sec': None, 'pre': {}}

    def _free_pools():
        if pools['layer'] is not None:
            pools['layer'].free()
        if pools['sec'] is not None:
            pools['sec'].free()
        for _tgt, _tex in pools['pre'].values():
            _tgt.free()
        pools['layer'] = pools['sec'] = None
        pools['pre'] = {}

    def _fail(msg):
        _free_pools()
        return None, msg

    # scissoring: a depth layer's passes and readbacks cost its own
    # bounding box, not the frame. Pure transport -- the same pixels
    # shade either way (the self test proves the two paths identical on
    # the driver) -- and the Debug toggle turns it off if a driver ever
    # disagrees about the newer region-read path.
    region_on = REGION_DRAWS and \
        bool(getattr(getattr(job, 'settings', None), 'gpu_scissor', True))
    for r in range(top + 1):
        sel = rorder[rbounds[r]:rbounds[r + 1]]
        if sel.size == 0:
            continue
        tm['ranks'] += 1
        spy, spx = py[sel], px[sel]
        ids[spy, spx, :3] = bary[sel]
        ids[spy, spx, 3] = tri[sel].astype(np.float32)
        if region_on:
            x0 = int(spx.min())
            y0 = int(spy.min())
            region = (x0, y0, int(spx.max()) + 1 - x0,
                      int(spy.max()) + 1 - y0)
            tm['scissor_px'] += float(region[2]) * float(region[3])
        else:
            x0 = y0 = 0
            region = None
            tm['scissor_px'] += float(w) * float(h)
        tm['frame_px'] += float(w) * float(h)
        # only the materials PRESENT in this rank draw: a pass with no
        # keep pixels writes nothing, so skipping it is bit-identical
        # -- and a deep, sparse rank usually holds one material, not
        # the whole scene's list
        rank_mats = set(
            int(m) for m in np.unique(mesh.mat_index[tri[sel]])) \
            if mesh.mat_index is not None else {0}
        t0 = _time.perf_counter()
        try:
            tex_ids = device.upload(ids)
        except Exception as exc:                                # noqa: BLE001
            return _fail(f'uploading layer {r} failed: {exc}')
        tm['upload_ms'] += (_time.perf_counter() - t0) * 1000.0
        tm['upload_mb'] += ids.nbytes / 1e6
        if vg is not None:
            vg.tri[spy, spx] = tri[sel]
            vg.bary[spy, spx] = bary[sel]
            # fresh per-rank evaluator caches: the caches key on the
            # surface OBJECT, and this object now holds a new rank
            job._hal_mat_eval = {'__gbuf': vg}
            job._hal_ray_blocks = {'__gbuf': vg}
        # the rank's height pre-passes: drawn over THIS layer's ids (or
        # CPU-evaluated over its virtual surface), so the neighbour
        # differences ride the rank's own coverage -- the same
        # definition the CPU's per-rank gradient fields use. Absent
        # materials' pre-passes skip with their materials.
        pre_tex = {}
        rd0 = tm['read_ms']
        t0 = _time.perf_counter()
        try:
            for key, entry in prepass_built.items():
                if key[0] not in rank_mats:
                    continue
                kind = entry[0]
                if kind == 'cpu':
                    himg = _cpu_height_image(job, vg, key[0], entry[1])
                    pre_tex[key] = device.upload(himg)
                else:
                    _kind, pshader, pbind = entry
                    pooled = pools['pre'].get(key)
                    if pooled is None:
                        tgt = device.Target(w, h)
                        pooled = (tgt, device.target_texture(tgt))
                        pools['pre'][key] = pooled
                    tgt, tex_handle = pooled
                    pb = dict(pbind)
                    pb['hal_gb_ids'] = tex_ids
                    device.draw_fullscreen(pshader, uni, pb, tgt,
                                           read=False, blend='NONE',
                                           clear=True, region=region)
                    pre_tex[key] = tex_handle
        except Exception as exc:                                # noqa: BLE001
            return _fail(f'layer {r} height pass failed: {exc}')
        if pools['layer'] is None:
            pools['layer'] = device.Target(w, h)
        target = pools['layer']
        first_draw = True
        try:
            for mat_id, name, shader, bind, extra, prepasses in built:
                if mat_id not in rank_mats:
                    continue
                b = dict(bind)
                b['hal_gb_ids'] = tex_ids
                for uname, _ps, _pb in prepasses:
                    b[uname] = pre_tex[(mat_id, uname)]
                # ALPHA_PREMULT -- the SAME blend every proven
                # multi-material one-target merge uses (the opaque frame,
                # the secondary sweeps). Materials are disjoint per pixel,
                # so out = src + dst*(1-src.a) IS the plain sum the
                # front-end's `acc += got` does: at a material's own
                # pixels dst is 0; everywhere else src is vec4(0) with
                # src.a 0, leaving dst intact. The first field run of
                # this path used ADDITIVE_PREMULT -- the engine's only
                # never-proven blend state -- and mismatched 1036 px;
                # never stand new driver state under a new feature when
                # a proven state computes the same thing.
                device.draw_fullscreen(shader, {**uni, **extra} if extra
                                       else uni, b, target, read=False,
                                       blend='ALPHA_PREMULT',
                                       clear=first_draw, region=region)
                first_draw = False
            tr = _time.perf_counter()
            img = device.read_target(target, region)
            tm['read_ms'] += (_time.perf_counter() - tr) * 1000.0
        except Exception as exc:                                # noqa: BLE001
            return _fail(f'layer {r} draw failed: {exc}')
        tm['draw_ms'] += (_time.perf_counter() - t0) * 1000.0 \
            - (tm['read_ms'] - rd0)
        if rplan is not None:
            # the recursion, from THIS layer's surface: the rank's
            # fragments spawn the same tree the opaque frame does
            from . import rtrace as RT

            def draw_secondary(plist, sec_ids, level,
                               _region=region, _x0=x0, _y0=y0):
                tu = _time.perf_counter()
                try:
                    tex_sec = device.upload(sec_ids)
                except Exception as exc:                        # noqa: BLE001
                    raise _SweepFail(f'uploading the layer level-{level} '
                                     f'ray buffer failed: {exc}')
                tm['upload_ms'] += (_time.perf_counter() - tu) * 1000.0
                tm['upload_mb'] += sec_ids.nbytes / 1e6
                drawn = sec_draws['secondary'] \
                    if plist is rplan['secondary'] \
                    else sec_draws['secondary_mid']
                if pools['sec'] is None:
                    pools['sec'] = device.Target(w, h)
                t2 = pools['sec']
                try:
                    for i2, (nm2, sh2, bd2, ex2) in enumerate(drawn):
                        b2 = dict(bd2)
                        b2['hal_gb_ids'] = tex_sec
                        device.draw_fullscreen(sh2, {**uni, **ex2}
                                               if ex2 else uni, b2, t2,
                                               read=False,
                                               blend='ALPHA_PREMULT',
                                               clear=(i2 == 0),
                                               region=_region)
                    tr2 = _time.perf_counter()
                    sec_r = device.read_target(t2, _region)
                    tm['read_ms'] += (_time.perf_counter() - tr2) * 1000.0
                except _SweepFail:
                    raise
                except Exception as exc:                        # noqa: BLE001
                    raise _SweepFail(f'the layer level-{level} ray '
                                     f'passes failed: {exc}')
                if _region is None:
                    return sec_r
                # the sweeps composite over full-frame arrays; outside
                # the rank's box no ray was spawned, so zeros there are
                # exactly what the full-frame read's cleared background
                # held
                sec_img = np.zeros((h, w, 4), np.float32)
                sec_img[_y0:_y0 + _region[3],
                        _x0:_x0 + _region[2]] = sec_r
                return sec_img

            def isect(org, dirs):
                got_h, why_r = RT.intersect_frame(job.bvh, org, dirs)
                if got_h is None:
                    raise _SweepFail(f'the layer ray trace failed: '
                                     f'{why_r}')
                return got_h

            if region is None:
                rgb = np.ascontiguousarray(img[:, :, :3], np.float32)
            else:
                # the sweeps work the full frame; the layer's colours
                # sit in its box and the rest never spawned a ray
                rgb = np.zeros((h, w, 3), np.float32)
                rgb[y0:y0 + region[3], x0:x0 + region[2]] = img[:, :, :3]
            # the sweep bucket must stay DISJOINT from uploads and
            # reads: draw_secondary uploads and reads inside this
            # window, and double-counted milliseconds would make the
            # printed remainder lie small
            up0, rr0 = tm['upload_ms'], tm['read_ms']
            t0 = _time.perf_counter()
            try:
                rgb = _run_sweeps(job, vg, rplan, rgb, draw_secondary,
                                  isect, env=env_plan)
            except _SweepFail as sf:
                return _fail(str(sf))
            except Exception as exc:                            # noqa: BLE001
                return _fail(f'the layer {r} sweeps failed: '
                             f'{type(exc).__name__}: {exc}')
            tm['sweep_ms'] += max(
                (_time.perf_counter() - t0) * 1000.0
                - (tm['upload_ms'] - up0) - (tm['read_ms'] - rr0), 0.0)
            out[sel, :3] = rgb[spy, spx]
            out[sel, 3] = img[spy - y0, spx - x0, 3]
        else:
            out[sel] = img[spy - y0, spx - x0, :4]
        # scatter back OUT: the reset costs the rank, not the frame
        ids[spy, spx] = _IDS_CLEAR
        if vg is not None:
            vg.tri[spy, spx] = -1
    _free_pools()
    if vg is not None:
        # the per-rank evaluator caches keyed on this vg object; leave
        # nothing dangling for whatever shades next on this job
        job._hal_mat_eval = {}
        job._hal_ray_blocks = {}
    tm['ray_build_ms'] = float(_RAY_BUILD[0])
    tm['total_ms'] = (_time.perf_counter() - t_all) * 1000.0
    tm['other_ms'] = max(
        tm['total_ms'] - tm['plan_ms'] - tm['compile_ms']
        - tm['upload_ms'] - tm['draw_ms'] - tm['read_ms']
        - tm['sweep_ms'], 0.0)
    tm.update(_marshal.acct())
    LAST_LAYER_TIMINGS.update(tm)
    if getattr(job.settings, 'fog', False):
        # the CPU fogs every layer fragment inside shade_batch; the
        # gathered driver colours take the same apply_fog on the way out
        # (vertex-rate materials never reach layer passes -- refused)
        from ..core.render import fog_for_points
        out[:, :3] = fog_for_points(job, tri, bary, out[:, :3])
    return out, None


def simulate_fragments(job, gbuf, tri, bary, px, py, rank):
    """The layer passes through the front-end: the headless proof.

    Same layers, same sources, same additive merge (a sum in NumPy),
    same gather -- returns (col (N,4), why)."""
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile

    passes, why, atlases = plan_frame(job, gbuf)
    if passes is None:
        return None, why
    lpasses = (atlases or {}).get('__layers')
    if not lpasses:
        return None, _no_layer_plan_why(job, atlases, tri)
    why = _layer_coverage(job, lpasses, tri)
    if why is not None:
        return None, why
    h, w = gbuf.tri.shape
    _ids0, attrs, side, tris_t, tside = _textures(job, gbuf)
    yy, xx = np.mgrid[0:h, 0:w]
    uv = np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h],
                  1).astype(np.float32)
    n = h * w
    base = {
        'hal_gb_attrs': Texture(attrs, colorspace='Non-Color',
                                filt='NEAREST', wrap='EXTEND'),
        'hal_gb_tris': Texture(tris_t, colorspace='Non-Color',
                               filt='NEAREST', wrap='EXTEND'),
    }
    for sname, entry in (atlases or {}).items():
        if sname.startswith('__'):
            continue
        _key, build = entry
        base[sname] = Texture(build(), colorspace='Non-Color',
                              filt='NEAREST', wrap='EXTEND')
    if (atlases or {}).get('__radfield') is not None:
        ids_f, _a2, _s2, _t2, _ts2 = _textures(job, gbuf)
        rtex, rwhy = _sim_radfield(atlases['__radfield'], job, ids_f,
                                   base, side, tside)
        if rtex is None:
            return None, rwhy
        base['hal_radfield'] = rtex
    rplan = (atlases or {}).get('__reflect')
    env_plan = (atlases or {}).get('__env')

    def uni_for(binds2, ids_arr):
        uni2 = dict(base)
        uni2['hal_gb_ids'] = Texture(ids_arr, colorspace='Non-Color',
                                     filt='NEAREST', wrap='EXTEND')
        for sname, key in (binds2.get('textures') or {}).items():
            uni2[sname] = Texture(job.textures[key].pixels,
                                  colorspace='Non-Color',
                                  filt='NEAREST', wrap='EXTEND')
        uni2['hal_attr_side'] = np.full(n, float(side), np.float32)
        uni2['hal_slot_count'] = np.full(n, 4.0, np.float32)
        uni2['hal_tri_side'] = np.full(n, float(tside), np.float32)
        uni2['hal_eye'] = np.tile(np.asarray(job.eye,
                                             np.float32)[None, :],
                                  (n, 1))
        uni2['hal_time'] = np.full(n, float(getattr(job.scene, 'time',
                                                    0.0)), np.float32)
        uni2['hal_frame'] = np.full(n, float(getattr(job.scene,
                                                     'frame', 0)),
                                    np.float32)
        uni2['vUV'] = uv
        return uni2

    progs = []
    pre_progs = {}              # (mat_id, uname) -> ('gpu', prog, pbinds)
    #                             or ('cpu', node_id, None)
    for mat_id, name, src, binds in lpasses:
        sim_src = src.replace('in vec2 vUV;', 'uniform vec2 vUV;')
        prog, err = try_compile(sim_src, 'GLSL')
        if prog is None:
            return None, f"'{name}' (layer) would not compile: {err}"
        for uname, psrc, pbinds in (binds.get('prepasses') or ()):
            if pbinds.get('cpu'):
                pre_progs[(mat_id, uname)] = ('cpu', pbinds['node'], None)
                continue
            pp = psrc.replace('in vec2 vUV;', 'uniform vec2 vUV;')
            pprog, perr = try_compile(pp, 'GLSL')
            if pprog is None:
                return None, f"'{name}' height pass would not " \
                             f'compile: {perr}'
            pre_progs[(mat_id, uname)] = ('gpu', pprog, pbinds)
        progs.append((mat_id, name, prog, binds))

    sec_progs = {'secondary': [], 'secondary_mid': []}
    if rplan is not None:
        for which2 in ('secondary', 'secondary_mid'):
            for mat_id, name, src, binds in (rplan.get(which2) or ()):
                pp = src.replace('in vec2 vUV;', 'uniform vec2 vUV;')
                prog, err = try_compile(pp, 'GLSL')
                if prog is None:
                    return None, f"'{name}' (hit) would not " \
                                 f'compile: {err}'
                sec_progs[which2].append((mat_id, name, prog, binds))

    out = np.zeros((tri.size, 4), np.float32)
    top = int(rank.max()) if rank.size else -1
    need_vg = rplan is not None or any(
        k == 'cpu' for k, _a, _b in pre_progs.values())
    for r in range(top + 1):
        sel = np.nonzero(rank == r)[0]
        if sel.size == 0:
            continue
        ids = _fragment_ids(h, w, py[sel], px[sel], tri[sel], bary[sel])
        vg = _layer_gbuf(h, w, py[sel], px[sel], tri[sel], bary[sel]) \
            if need_vg else None
        # the driver's own skip, mirrored: only the materials PRESENT
        # in this rank run (an absent material adds zeros -- skipping
        # is bit-identical)
        mesh_l = job.scene.mesh
        rank_mats = set(
            int(m) for m in np.unique(mesh_l.mat_index[tri[sel]])) \
            if mesh_l.mat_index is not None else {0}
        # the rank's height pre-pass images, over ITS OWN surface
        pre_imgs = {}
        for key, entry in pre_progs.items():
            if key[0] not in rank_mats:
                continue
            if entry[0] == 'cpu':
                pre_imgs[key] = _cpu_height_image(job, vg, key[0],
                                                  entry[1])
            else:
                _k, pprog, pbinds = entry
                pgot = pprog.run(uni_for(pbinds, ids), {}, n)[0]['Color']
                pre_imgs[key] = pgot.reshape(h, w, 4)
        acc = np.zeros((n, 4), np.float32)
        for mat_id, name, prog, binds in progs:
            if mat_id not in rank_mats:
                continue
            u = uni_for(binds, ids)
            for uname, _ps, _pb in (binds.get('prepasses') or ()):
                u[uname] = Texture(pre_imgs[(mat_id, uname)],
                                   colorspace='Non-Color',
                                   filt='NEAREST', wrap='EXTEND')
            got = prog.run(u, {}, n)[0]['Color']
            acc += got                       # additive: keeps are disjoint
        img = acc.reshape(h, w, 4)
        if rplan is not None:
            from .rtrace import simulate_intersect

            def draw_secondary(plist, sec_ids, _level):
                plist_progs = sec_progs['secondary'] \
                    if plist is rplan['secondary'] \
                    else sec_progs['secondary_mid']
                acc3 = np.zeros((n, 3), np.float32)
                for _mi2, name2, prog2, binds2 in plist_progs:
                    got2 = prog2.run(uni_for(binds2, sec_ids), {},
                                     n)[0]['Color']
                    keep2 = got2[:, 3] > 0.5
                    acc3[keep2] = got2[keep2, :3]
                return acc3.reshape(h, w, 3)

            def isect(org, dirs):
                return simulate_intersect(job.bvh, org, dirs, 1e30)

            rgb = np.ascontiguousarray(img[:, :, :3], np.float32)
            try:
                rgb = _run_sweeps(job, vg, rplan, rgb, draw_secondary,
                                  isect, env=env_plan)
            except _SweepFail as sf:
                return None, str(sf)
            out[sel, :3] = rgb[py[sel], px[sel]]
            out[sel, 3] = img[py[sel], px[sel], 3]
        else:
            out[sel] = img[py[sel], px[sel], :4]
    if getattr(job.settings, 'fog', False):
        # the front-end mirror of the gather fog above
        from ..core.render import fog_for_points
        out[:, :3] = fog_for_points(job, tri, bary, out[:, :3])
    return out, None


def _refraction_rays(job, gbuf, rplan):
    """(py, px, org, dirs): the refraction half of `_add_raytraced`.

    Per-fragment eta chosen by which side of the surface the camera sees
    (dot(N, V) < 0 means exiting), GLSL refract with the total-internal-
    reflection fallback to a mirror bounce, and the origin stepped INTO
    the surface by the raw ray bias.
    """
    from ..core import mathx as M
    mesh = job.scene.mesh
    py, px, P, I, N = _ray_blocks(job, gbuf, rplan['refractive'])
    if py.size == 0:
        return py, px, None, None
    V = -M.normalize(I)
    m = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
        else np.zeros(py.size, np.int32)
    ior = np.ones(py.size, np.float32)
    for mi, spec in rplan['refract'].items():
        ior[m == mi] = spec['ior']
    eta = np.where(M.dot(N, V) < 0, ior, 1.0 / ior)
    T = M.refract(-V, N, eta)
    bad = (T * T).sum(1) < 1e-9
    T = np.where(bad[:, None], M.reflect(-V, N), T).astype(np.float32)
    org = (P - N * rplan['bias']).astype(np.float32)
    return py, px, org, T


class _SweepFail(Exception):
    """A sweep stage failed; the message is the caller-facing reason."""


def _hit_surface(job, tris, bary):
    """(P, I, N, m): a hit surface's shading frame, the CPU's own way.

    ctx.px is None at a hit, so the Bump node is a wire and only the
    Normal Map chain bends: the same normal the CPU's hit shading uses.
    I is P - eye (ctx.I always is), so V stays the camera direction.
    """
    from ..core import mathx as M
    from ..core.nodeeval import GraphEvaluator
    from ..core.render import closure_to_surface
    from .material import master_normal_linked

    tris = np.asarray(tris, np.int32)
    ctx = job.context(tris, bary, None, None,
                      np.ones(tris.size, bool), None, 0, True)
    N = M.normalize(ctx.N)
    mesh = job.scene.mesh
    m = mesh.mat_index[tris] if mesh.mat_index is not None \
        else np.zeros(tris.size, np.int32)
    for mi in np.unique(m):
        mat = job.scene.materials[int(mi)] \
            if int(mi) < len(job.scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        if not master_normal_linked(graph):
            continue
        sel = np.nonzero(m == mi)[0]
        sub = job.context(tris[sel], bary[sel], None, None,
                          np.ones(sel.size, bool), None, 0, True)
        ev = GraphEvaluator(graph, sub, job.textures,
                            getattr(mat, 'programs', None))
        cl, _disp = ev.evaluate_surface()
        _surf, _model, nrm = closure_to_surface(cl, sub, job.settings, mat)
        if nrm is not None:
            N[sel] = M.normalize(nrm)
    return np.asarray(ctx.P), np.asarray(ctx.I), N, m


def _hit_rays(job, tris, bary, which, rplan):
    """(org, dirs): rays FROM hit surfaces -- the CPU's recursion step.

    Reflection steps OFF the surface, refraction INTO it, both by the
    raw bias, exactly `_add_raytraced` at a hit batch.
    """
    from ..core import mathx as M
    P, I, N, m = _hit_surface(job, tris, bary)
    V = -M.normalize(I)
    if which == 'reflective':
        dirs = M.reflect(-V, N).astype(np.float32)
        org = (P + N * rplan['bias']).astype(np.float32)
        return org, dirs
    ior = np.ones(np.asarray(tris).size, np.float32)
    for mi, spec in rplan['refract'].items():
        ior[m == mi] = spec['ior']
    eta = np.where(M.dot(N, V) < 0, ior, 1.0 / ior)
    T = M.refract(-V, N, eta)
    bad = (T * T).sum(1) < 1e-9
    dirs = np.where(bad[:, None], M.reflect(-V, N), T).astype(np.float32)
    org = (P - N * rplan['bias']).astype(np.float32)
    return org, dirs


def _apply_cpu_env_primary(job, gbuf, env_scales, out):
    """out[material px] += world(R) * scale: the renderer's OWN env term.

    For worlds richer than the baked GLSL paths -- the Bryce sky lab,
    STARFIELD, PHYSICAL, HDRI, a world node graph, the ground plane --
    the environment reflection is evaluated by `world_color` itself
    along the reflected rays and added AFTER readback. Exact for any
    world by construction: the CPU adds this term last (fog refuses),
    and the reflected rays use the same bent, unflipped normals the
    proven ray machinery uses.
    """
    from ..core import mathx as M
    from ..core.render import world_color
    if not env_scales:
        return out
    for mi, scale in env_scales.items():
        py, px, P, I, N = _ray_blocks(job, gbuf, [int(mi)])
        if py.size == 0:
            continue
        V = -M.normalize(I)
        R = M.reflect(-V, N).astype(np.float32)
        env = np.asarray(world_color(job.scene, job.settings, R,
                                     job.textures, int(py.size),
                                     eye=job.eye), np.float32)[:, :3]
        out[py, px] = out[py, px] + env * np.asarray(scale, np.float32)
    return out


def _apply_cpu_env_hits(job, rplan, env_scales, img, py, px, tid, u, v):
    """img[hit px] += world(R) * scale, at the recursion's final depth.

    The depth-exhausted hits are exactly where the CPU's env branch
    lives; their surfaces are readback-known, so the same CPU-composite
    trick covers them: reflect the camera direction about the hit's
    bent normal, ask the renderer's own world, scale by the HIT
    material's constants.
    """
    from ..core import mathx as M
    from ..core.render import world_color
    if not env_scales:
        return
    hit = tid >= 0
    if not hit.any():
        return
    mesh = job.scene.mesh
    m = mesh.mat_index[np.clip(tid, 0, None)] \
        if mesh.mat_index is not None else np.zeros(tid.size, np.int32)
    want = hit & np.isin(m, np.asarray(sorted(env_scales), np.int64))
    sel = np.nonzero(want)[0]
    if sel.size == 0:
        return
    bary = np.stack([1.0 - u[sel] - v[sel], u[sel], v[sel]],
                    axis=1).astype(np.float32)
    P, I, N, mh = _hit_surface(job, tid[sel], bary)
    V = -M.normalize(I)
    R = M.reflect(-V, N).astype(np.float32)
    env = np.asarray(world_color(job.scene, job.settings, R, job.textures,
                                 int(sel.size), eye=job.eye),
                     np.float32)[:, :3]
    scale = np.zeros((sel.size, 3), np.float32)
    for mi, sc in env_scales.items():
        scale[mh == int(mi)] = np.asarray(sc, np.float32)
    img[py[sel], px[sel]] = img[py[sel], px[sel]] + env * scale


def _child_reflect(job, rplan, img, py, px, dirs, child_hit, child_img,
                   mat_px):
    """img[hit px] += child colour * the HIT material's reflect scale.

    The recursion's backward composite: exactly `_add_raytraced`'s
    reflection add, with the constants of the material the PARENT ray
    hit, and `world_color` along the child rays where they missed.
    """
    from ..core.render import world_color
    add = np.zeros((py.size, 3), np.float32)
    hit = np.asarray(child_hit, bool)      # 1-D, aligned with these rays
    if hit.any():
        add[hit] = child_img[py[hit], px[hit], :3]
    miss = ~hit
    if miss.any():
        wc = world_color(job.scene, job.settings, dirs[miss], job.textures,
                         int(miss.sum()), eye=job.eye)
        add[miss] = np.asarray(wc, np.float32)[:, :3]
    scale = np.zeros((py.size, 3), np.float32)
    for mi, sc in rplan['scale'].items():
        scale[mat_px == mi] = np.asarray(sc, np.float32)
    img[py, px] = img[py, px] + add * scale


def _child_refract(job, rplan, img, py, px, dirs, child_hit, child_img,
                   mat_px):
    """img[hit px] = img*(1-k) + child*k*diffuse, by the HIT material."""
    from ..core.render import world_color
    add = np.zeros((py.size, 3), np.float32)
    hit = np.asarray(child_hit, bool)      # 1-D, aligned with these rays
    if hit.any():
        add[hit] = child_img[py[hit], px[hit], :3]
    miss = ~hit
    if miss.any():
        wc = world_color(job.scene, job.settings, dirs[miss], job.textures,
                         int(miss.sum()), eye=job.eye)
        add[miss] = np.asarray(wc, np.float32)[:, :3]
    k = np.zeros((py.size, 1), np.float32)
    dif = np.ones((py.size, 3), np.float32)
    for mi, spec in rplan['refract'].items():
        sel = mat_px == mi
        k[sel] = np.float32(spec['k'])
        dif[sel] = np.asarray(spec['diffuse'], np.float32)
    img[py, px] = img[py, px] * (1.0 - k) + add * k * dif


def _uvgrad_field(job, gbuf):
    """(H, W, 4) float32: the CPU's analytic UV screen derivatives.

    (du/dx, du/dy, dv/dx, dv/dy) at every covered pixel, zeros elsewhere
    -- computed by ShadeJob.uv_screen_gradients, the very function the
    CPU's own trilinear reads through the context. Same numbers, one
    upload, shared by every footprint-filtered sampler in the frame.
    """
    from ..core import raster as _raster
    h, w = gbuf.tri.shape
    field = np.zeros((h, w, 4), np.float32)
    cov = gbuf.tri >= 0
    if cov.any():
        mesh = job.scene.mesh
        tri = gbuf.tri[cov]
        bary = gbuf.bary[cov]
        uv = _raster.fetch(mesh.uvs, mesh.tris, tri, bary) \
            if mesh.uvs is not None \
            else np.zeros((tri.size, 2), np.float32)
        du, dv = job.uv_screen_gradients(tri, bary, uv)
        field[cov, 0] = du[:, 0]
        field[cov, 1] = du[:, 1]
        field[cov, 2] = dv[:, 0]
        field[cov, 3] = dv[:, 1]
    return field


def _fog_readback(job, gbuf, passes, out, hit):
    """The CPU's own fog over the deferred readback, pixel-rate only.

    Fog is separable -- a lerp toward the fog colour by geometry alone --
    so instead of a GLSL twin of four fog modes, the vertex quantisation
    and the height layer, the readback takes core.render.apply_fog with
    the same P and view depth ctx.depth carries. Order matches the CPU
    exactly: lighting, traced composites and the env term are already in
    `out`; fog is the CPU's LAST rgb operation, and it is the last here.

    Vertex-rate materials SKIP: shade_batch lit their corners with
    rate_mode LIGHT, which runs apply_fog at the corner -- per-vertex
    fog, the era's own -- and the interpolated product already carries
    it. Fogging again would double-attenuate exactly those materials.
    """
    st = job.settings
    if not getattr(st, 'fog', False) or not hit.any():
        return out
    from ..core.render import fog_for_points
    vrate = sorted(mat_id for mat_id, _n, _s, binds in passes
                   if (binds or {}).get('vlight'))
    py, px = np.nonzero(hit)
    tri = gbuf.tri[py, px]
    mesh = job.scene.mesh
    mi = mesh.mat_index[tri] if mesh.mat_index is not None \
        else np.zeros(tri.size, np.int32)
    sel = ~np.isin(mi, np.asarray(vrate, np.int64)) if vrate \
        else np.ones(tri.size, bool)
    if not sel.any():
        return out
    out[py[sel], px[sel]] = fog_for_points(
        job, tri[sel], gbuf.bary[py, px][sel], out[py[sel], px[sel]])
    return out


def _run_sweeps(job, gbuf, rplan, out, draw_secondary, intersect,
                env=None):
    """The ray recursion both backends share: `_add_raytraced`'s tree.

    At each level the hits shade through the secondary passes -- WITH
    the environment term at the final depth, WITHOUT it above (a traced
    child replaces it, exactly the CPU's `d < D` branch) -- then any hit
    on a reflective/refractive material spawns the next level, and the
    child's colours composite backward with the HIT material's
    constants. Depth 1 reduces to the flat sweep this generalises.
    `env` is the plan's CPU-composite env spec (atlases['__env']): its
    'hit' scales apply at the final depth, where the CPU's env branch
    lives, evaluated by the renderer's own `world_color`.

    `draw_secondary(pass_list, sec_ids, level)` returns the (H, W, 3)
    image of those passes over that ids texture; `intersect(org, dirs)`
    returns (tid, t, u, v). Either raises `_SweepFail` with the reason.
    """
    h, w = gbuf.tri.shape
    depth = max(int(rplan.get('depth', 1)), 1)
    mesh = job.scene.mesh
    env_hit = (env or {}).get('hit') or {}

    def shade_level(level, py, px, org, dirs):
        tid, _t, u, v = intersect(org, dirs)
        sec_ids, hitm = _secondary_ids(h, w, py, px, tid, u, v)
        plist = rplan['secondary'] if level >= depth \
            else rplan['secondary_mid']
        img = draw_secondary(plist, sec_ids, level)
        # the CONTRACT, enforced where both backends meet: a level image
        # is (H, W, 3). The front-end adapter returned 3 channels and the
        # driver's read-back returned 4 -- readable by either composite,
        # but the child composites WRITE into the level image, and the
        # field found the 4-channel one with a broadcast crash the
        # headless path could never reach
        img = np.ascontiguousarray(np.asarray(img)[:, :, :3], np.float32)
        if level >= depth and env_hit:
            # the depth-exhausted env term, for worlds the GLSL cannot
            # bake: the renderer's own sky along the hits' reflections
            _apply_cpu_env_hits(job, rplan, env_hit, img, py, px,
                                tid, u, v)
        if level < depth:
            hit = tid >= 0
            m = mesh.mat_index[np.clip(tid, 0, None)] \
                if mesh.mat_index is not None \
                else np.zeros(tid.size, np.int32)
            for which in ('reflective', 'refractive'):
                mats = rplan.get(which) or ()
                if len(mats) == 0:
                    continue
                sel = np.nonzero(hit & np.isin(
                    m, np.asarray(mats, np.int64)))[0]
                if sel.size == 0:
                    continue
                bary2 = np.stack([1.0 - u[sel] - v[sel], u[sel], v[sel]],
                                 axis=1).astype(np.float32)
                org2, dirs2 = _hit_rays(job, tid[sel], bary2, which, rplan)
                cpy, cpx = py[sel], px[sel]
                child_img, child_hitm = shade_level(level + 1, cpy, cpx,
                                                    org2, dirs2)
                if which == 'reflective':
                    _child_reflect(job, rplan, img, cpy, cpx, dirs2,
                                   child_hitm, child_img, m[sel])
                else:
                    _child_refract(job, rplan, img, cpy, cpx, dirs2,
                                   child_hitm, child_img, m[sel])
        if getattr(job.settings, 'fog', False):
            # the CPU fogs every hit inside its recursion (shade_batch at
            # the hit point runs apply_fog at the HIT's own view depth,
            # after the child composites) -- mirror it here, after the
            # children, before this level returns to its parent. Misses
            # take world colour and stay unfogged, exactly as trace()
            # leaves the sky. Secondary passes refuse vertex-rate
            # materials, so every hit here fogs at the pixel rate.
            selF = np.nonzero(tid >= 0)[0]
            if selF.size:
                from ..core.render import fog_for_points
                baryF = np.stack([1.0 - u[selF] - v[selF], u[selF],
                                  v[selF]], axis=1).astype(np.float32)
                img[py[selF], px[selF]] = fog_for_points(
                    job, tid[selF], baryF, img[py[selF], px[selF]])
        return img, hitm

    for which, rays_fn, composite_fn in SWEEPS:
        if not rplan.get(which):
            continue
        py, px, org, dirs = rays_fn(job, gbuf, rplan)
        if py.size == 0:
            continue
        img1, hitm1 = shade_level(1, py, px, org, dirs)
        out = composite_fn(job, gbuf, rplan, out, py, px, dirs, hitm1,
                           img1)
    return out


def _composite_refractions(job, gbuf, rplan, out, py, px, dirs, hit,
                           sec_img):
    """rgb = rgb*(1-k) + hit_colour*k*diffuse, in place.

    The LERP the CPU applies AFTER the reflection add -- k and the tint
    are per-material constants ((1-opacity)*refraction and the flat base
    colour, both held constant by the plan's gates). Misses fall to
    `world_color` along the transmitted ray, exactly as `trace()` does.
    """
    from ..core.render import world_color
    add = np.zeros((py.size, 3), np.float32)
    if hit.any():
        add[hit] = sec_img[py[hit], px[hit], :3]
    miss = ~hit
    if miss.any():
        wc = world_color(job.scene, job.settings, dirs[miss], job.textures,
                         int(miss.sum()), eye=job.eye)
        add[miss] = np.asarray(wc, np.float32)[:, :3]
    mesh = job.scene.mesh
    m = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
        else np.zeros(py.size, np.int32)
    k = np.zeros((py.size, 1), np.float32)
    dif = np.zeros((py.size, 3), np.float32)
    for mi, spec in rplan['refract'].items():
        sel = m == mi
        k[sel] = spec['k']
        dif[sel] = np.asarray(spec['diffuse'], np.float32)
    out[py, px] = out[py, px] * (1.0 - k) + add * k * dif
    return out


#: the two secondary-ray sweeps, in the CPU's application order:
#: reflections ADD first, then refractions LERP over the result
SWEEPS = (('reflective', _reflection_rays, _composite_reflections),
          ('refractive', _refraction_rays, _composite_refractions))


def _vlight_image(job, spec):
    """The corner-light texture for one vertex-rate pass, padded square.

    The VALUES are the CPU's own: `vertex_light_corners` runs the full
    lighting at the corners (rate_mode LIGHT), so the driver's picture
    stands on the same numbers the CPU picture stands on and the seam
    is the interpolation arithmetic alone.
    """
    from ..core.render import vertex_light_corners
    arr = vertex_light_corners(job, int(spec['mat']), str(spec['rate']),
                               job.settings)
    side = int(spec['side'])
    img = np.zeros((side * side, 4), np.float32)
    img[:arr.shape[0]] = arr
    return img.reshape(side, side, 4)


def _sim_radfield(radfield, job, ids_arr, tex_by_name, side, tside):
    """The grid pre-pass through the front-end, shared by both sims.

    Returns (Texture, None) or (None, why). `tex_by_name` supplies the
    packed attribute/tri/BVH/circle textures the pass samples; the ids
    come in as the FRAME's array (a layer sim runs over per-rank ids,
    but the gather reads the opaque frame).
    """
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile
    rsrc, rbinds = radfield
    gw_r, gh_r = rbinds['size']
    rprog, rerr = try_compile(
        rsrc.replace('in vec2 vUV;', 'uniform vec2 vUV;'), 'GLSL')
    if rprog is None:
        return None, f'the radiosity grid pass would not compile: {rerr}'
    gyy, gxx = np.mgrid[0:gh_r, 0:gw_r]
    gn = int(gw_r) * int(gh_r)
    guv = np.stack([(gxx.ravel() + 0.5) / gw_r,
                    (gyy.ravel() + 0.5) / gh_r], 1).astype(np.float32)
    runi = {name: tex_by_name[name]
            for name in rbinds.get('samplers', ())
            if name in tex_by_name}
    runi['hal_gb_ids'] = Texture(ids_arr, colorspace='Non-Color',
                                 filt='NEAREST', wrap='EXTEND')
    runi['hal_attr_side'] = np.full(gn, float(side), np.float32)
    runi['hal_slot_count'] = np.full(gn, 4.0, np.float32)
    runi['hal_tri_side'] = np.full(gn, float(tside), np.float32)
    runi['hal_eye'] = np.tile(np.asarray(job.eye, np.float32)[None, :],
                              (gn, 1))
    runi['vUV'] = guv
    try:
        rout = rprog.run(runi, {}, gn)[0]['Color']
    except Exception as exc:                                    # noqa: BLE001
        return None, f'the radiosity grid pass failed in the ' \
                     f'front-end: {exc}'
    return Texture(np.asarray(rout, np.float32).reshape(int(gh_r),
                                                        int(gw_r), 4),
                   colorspace='Non-Color', filt='NEAREST',
                   wrap='EXTEND'), None


def simulate(job, gbuf, passes=None, atlases=None):
    """Run the frame passes through Halcyon's own GLSL front-end.

    This is the proof that does not need a GPU: the same sources the driver
    would compile, executed by the NumPy backend against the same packed
    textures -- shadow atlases included. Returns (image (H,W,3), covered
    mask) or (None, why).
    """
    from ..core.texture import Texture
    from ..shaders.compiler import try_compile

    if passes is None:
        passes, why, atlases = plan_frame(job, gbuf)
        if passes is None:
            return None, why
    h, w = gbuf.tri.shape
    ids, attrs, side, tris, tside = _textures(job, gbuf)
    yy, xx = np.mgrid[0:h, 0:w]
    uv = np.stack([(xx.ravel() + 0.5) / w, (yy.ravel() + 0.5) / h],
                  1).astype(np.float32)
    n = h * w
    tex = {
        'hal_gb_ids': Texture(ids, colorspace='Non-Color', filt='NEAREST',
                              wrap='EXTEND'),
        'hal_gb_attrs': Texture(attrs, colorspace='Non-Color', filt='NEAREST',
                                wrap='EXTEND'),
        'hal_gb_tris': Texture(tris, colorspace='Non-Color', filt='NEAREST',
                               wrap='EXTEND'),
    }
    for sname, entry in (atlases or {}).items():
        if sname.startswith('__'):
            continue               # plans and specs, not atlases
        _key, build = entry
        tex[sname] = Texture(build(), colorspace='Non-Color', filt='NEAREST',
                             wrap='EXTEND')

    if (atlases or {}).get('__radfield') is not None:
        # the interpolated gather's grid pre-pass, over grid lanes: the
        # simulator's twin of the driver's grid draw. Its output joins
        # `tex` by name, exactly as the driver's target joins the binds.
        rtex, rwhy = _sim_radfield(atlases['__radfield'], job, ids,
                                   tex, side, tside)
        if rtex is None:
            return None, rwhy
        tex['hal_radfield'] = rtex

    def fill_base(uni, ids_texture, binds):
        uni['hal_gb_ids'] = ids_texture
        if binds.get('vlight'):
            uni['hal_vlight'] = Texture(_vlight_image(job, binds['vlight']),
                                        colorspace='Non-Color',
                                        filt='NEAREST', wrap='EXTEND')
        for sname, key in (binds.get('textures') or {}).items():
            # the manual sampler fetches texel centres, so the binding's
            # own filter and wrap never fire; the arithmetic is in the
            # shader
            uni[sname] = Texture(job.textures[key].pixels,
                                 colorspace='Non-Color', filt='NEAREST',
                                 wrap='EXTEND')
        for sname, key in (binds.get('textures_mip') or {}).items():
            from .material import mip_atlas as _mip_atlas
            uni[sname] = Texture(_mip_atlas(job.textures[key])[0],
                                 colorspace='Non-Color', filt='NEAREST',
                                 wrap='EXTEND')
        if binds.get('needs_uvgrad'):
            uni['hal_uvgrad'] = Texture(_uvgrad_field(job, gbuf),
                                        colorspace='Non-Color',
                                        filt='NEAREST', wrap='EXTEND')
        uni['hal_attr_side'] = np.full(n, float(side), np.float32)
        uni['hal_slot_count'] = np.full(n, 4.0, np.float32)
        uni['hal_tri_side'] = np.full(n, float(tside), np.float32)
        uni['hal_eye'] = np.tile(np.asarray(job.eye, np.float32)[None, :],
                                 (n, 1))
        # per-frame scalars a coded shader may read; unused are inert
        uni['hal_time'] = np.full(n, float(getattr(job.scene, 'time', 0.0)),
                                  np.float32)
        uni['hal_frame'] = np.full(n, float(getattr(job.scene, 'frame', 0)),
                                   np.float32)
        uni['vUV'] = uv
        return uni

    def run_passes(pass_list, ids_texture):
        got_out = np.zeros((n, 3), np.float32)
        got_hit = np.zeros(n, bool)
        for mat_id, name, src, binds in pass_list:
            sim_src = src.replace('in vec2 vUV;', 'uniform vec2 vUV;')
            prog, err = try_compile(sim_src, 'GLSL')
            if prog is None:
                return None, None, f"'{name}' would not compile: {err}"
            uni = fill_base(dict(tex), ids_texture, binds)
            # bump height pre-passes: render each chain over the same ids,
            # hand the image to the main pass as its neighbour texture --
            # or, for a chain the emitter refused, evaluate it with the
            # renderer's own CPU code and hand over that image instead
            for uname, psrc, pbinds in (binds.get('prepasses') or ()):
                if pbinds.get('cpu'):
                    himg = _cpu_height_image(job, gbuf, mat_id,
                                             pbinds['node'])
                    uni[uname] = Texture(himg, colorspace='Non-Color',
                                         filt='NEAREST', wrap='EXTEND')
                    continue
                pp = psrc.replace('in vec2 vUV;', 'uniform vec2 vUV;')
                pprog, perr = try_compile(pp, 'GLSL')
                if pprog is None:
                    return None, None, \
                        f"'{name}' height pass would not compile: {perr}"
                puni = fill_base(dict(tex), ids_texture, pbinds)
                pgot = pprog.run(puni, {}, n)[0]['Color']
                uni[uname] = Texture(pgot.reshape(h, w, 4),
                                     colorspace='Non-Color', filt='NEAREST',
                                     wrap='EXTEND')
            got = prog.run(uni, {}, n)[0]['Color']
            keep = got[:, 3] > 0.5
            got_out[keep] = got[keep, :3]
            got_hit |= keep
        return got_out, got_hit, None

    out, hit, why = run_passes(passes, tex['hal_gb_ids'])
    if out is None:
        return None, why
    out = out.reshape(h, w, 3)
    hit = hit.reshape(h, w)

    env_plan = (atlases or {}).get('__env')
    out = _apply_cpu_env_primary(job, gbuf,
                                 (env_plan or {}).get('primary'), out)

    rplan = (atlases or {}).get('__reflect')
    if rplan is not None:
        from .rtrace import simulate_intersect

        def draw_secondary(plist, sec_ids, _level):
            sec_out, _sec_hit, why = run_passes(
                plist, Texture(sec_ids, colorspace='Non-Color',
                               filt='NEAREST', wrap='EXTEND'))
            if sec_out is None:
                raise _SweepFail(str(why))
            return sec_out.reshape(h, w, 3)

        def isect(org, dirs):
            return simulate_intersect(job.bvh, org, dirs, 1e30)

        try:
            out = _run_sweeps(job, gbuf, rplan, out, draw_secondary, isect,
                              env=env_plan)
        except _SweepFail as sf:
            return None, str(sf)
    out = _fog_readback(job, gbuf, passes, out, hit)
    return out, hit


def shade_frame(job, gbuf):
    """The driver path: upload the G-buffer, draw each material's pass.

    Returns (image (H,W,3), covered mask) or (None, why). Every failure --
    no gpu module, a driver that rejects a shader, anything -- is a reason,
    and the caller shades on the CPU as it always has.
    """
    from . import device

    import time as _time
    ok, why = device.probe()
    if not ok:
        return None, why
    t_p = _time.perf_counter()
    passes, why, atlases = plan_frame(job, gbuf)
    t_plan = _time.perf_counter() - t_p
    if passes is None:
        return None, why
    h, w = gbuf.tri.shape
    if not passes:
        return np.zeros((h, w, 3), np.float32), np.zeros((h, w), bool)

    import time as _time
    from . import gbuffer as GB
    t_all = _time.perf_counter()
    t0 = _time.perf_counter()
    mesh = job.scene.mesh
    mkey = _mesh_key(mesh)
    ids = GB.pack_ids(gbuf)                    # camera-dependent: every frame
    side_holder = {}

    def build_attrs():
        arr, sd = GB.pack_attributes(mesh, respect_smooth=True)
        side_holder['side'] = sd
        return arr

    def build_tris():
        arr, sd = GB.pack_tri_data(mesh)
        side_holder['tside'] = sd
        return arr

    rplan = atlases.get('__reflect')
    prepass_tex = {}               # (mat_id, sampler name) -> height texture
    try:
        tex_ids = device.upload(ids)
        tex_attrs = device.upload_cached(('gb_attrs',) + mkey, build_attrs)
        tex_tris = device.upload_cached(('gb_tris',) + mkey, build_tris)
        tex_shadows = {sname: device.upload_cached(entry[0], entry[1])
                       for sname, entry in atlases.items()
                       if not sname.startswith('__')}
        all_binds = [b for _mi, _n2, _s2, b in
                     (passes + (rplan['secondary'] if rplan else []))]
        for b in list(all_binds):
            all_binds.extend(p[2] for p in (b.get('prepasses') or ()))
        tex_images = _gather_pass_textures(all_binds, job.textures,
                                           device.upload_cached)
        # the footprint field: the CPU's analytic UV derivatives for every
        # covered pixel, one RGBA32F upload shared by every filtered
        # sampler (du/dx, du/dy, dv/dx, dv/dy)
        tex_uvgrad = None
        if any((b or {}).get('needs_uvgrad') for b in all_binds):
            tex_uvgrad = device.upload(_uvgrad_field(job, gbuf))
        # vertex-rate passes: the CPU lights the corners (worker side --
        # cheap, that is the point of the rate) and the values cross as
        # one small texture per material. Fresh each frame: the corners
        # move with the lights, and caching them is a later economy
        vlight_tex = {}
        for _vmi, _vn, _vs, binds in passes:
            spec = binds.get('vlight')
            if spec:
                vlight_tex[int(_vmi)] = device.upload(
                    _vlight_image(job, spec))
    except Exception as exc:                                    # noqa: BLE001
        return None, f'uploading the G-buffer failed: {exc}'
    # the packers only ran on a cache miss; on a hit the sides come from the
    # texture itself (attribute textures are square)
    side = side_holder.get('side', int(tex_attrs.width))
    tside = side_holder.get('tside', int(tex_tris.width))
    t_upload = _time.perf_counter() - t0

    # compile and gather bindings for every pass before anything draws
    def build_draws(pass_list, ids_tex, tag='HAL_MAT'):
        built = []
        for mat_id, name, src, binds in pass_list:
            samplers = binds.get('samplers', ())
            spec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs', 'hal_gb_tris']
                    + list(samplers),
                    'floats': ['hal_attr_side', 'hal_slot_count',
                               'hal_tri_side']
                    + list(binds.get('frame_uniforms', ())),
                    'vec3': ['hal_eye']}
            shader, err = device.compile_dynamic(f'{tag}_{mat_id}', src,
                                                 spec)
            if shader is None:
                return None, f"the driver rejected '{name}': {err}"
            bind = {'hal_gb_ids': ids_tex, 'hal_gb_attrs': tex_attrs,
                    'hal_gb_tris': tex_tris}
            for sname in samplers:
                if sname == 'hal_vlight' and int(mat_id) in vlight_tex:
                    bind[sname] = vlight_tex[int(mat_id)]
                elif sname == 'hal_uvgrad' and tex_uvgrad is not None:
                    bind[sname] = tex_uvgrad
                elif sname in tex_shadows:
                    bind[sname] = tex_shadows[sname]
                elif (id(binds), sname) in tex_images:
                    bind[sname] = tex_images[(id(binds), sname)]
                elif (mat_id, sname) in prepass_tex:
                    bind[sname] = prepass_tex[(mat_id, sname)]
                else:
                    return None, f"'{name}' wants {sname} but nothing " \
                                 f'was packed for it'
            # per-frame scalars only the passes that declare them may
            # receive -- the driver refuses unknown uniform names
            extra = {}
            for u in binds.get('frame_uniforms', ()):
                if u == 'hal_time':
                    extra[u] = float(getattr(job.scene, 'time', 0.0))
                elif u == 'hal_frame':
                    extra[u] = float(getattr(job.scene, 'frame', 0))
            built.append((name, shader, bind, extra))
        return built, None

    uni = {'hal_attr_side': float(side), 'hal_slot_count': 4.0,
           'hal_tri_side': float(tside),
           'hal_eye': tuple(float(v) for v in job.eye)}

    # bump height pre-passes draw FIRST, each into its own target, so
    # build_draws can bind their colour textures into the main passes.
    # Fragment renders one, fragment samples it: no stage crossing.
    prepass_targets = []
    for mat_id, name, _src, binds in passes:
        for uname, psrc, pbinds in (binds.get('prepasses') or ()):
            if pbinds.get('cpu'):
                # the emitter refused this height chain; the renderer's
                # own evaluator produces the image instead, exactly
                try:
                    himg = _cpu_height_image(job, gbuf, mat_id,
                                             pbinds['node'])
                    prepass_tex[(mat_id, uname)] = device.upload(himg)
                except Exception as exc:                        # noqa: BLE001
                    for t in prepass_targets:
                        t.free()
                    return None, f"'{name}' CPU height pass failed: {exc}"
                continue
            spec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs',
                                 'hal_gb_tris']
                    + list(pbinds.get('samplers', ())),
                    'floats': ['hal_attr_side', 'hal_slot_count',
                               'hal_tri_side']
                    + list(pbinds.get('frame_uniforms', ())),
                    'vec3': ['hal_eye']}
            shader, err = device.compile_dynamic(
                f'HAL_BUMP_{mat_id}_{uname}', psrc, spec)
            if shader is None:
                return None, f"the driver rejected '{name}' height " \
                             f'pass: {err}'
            bind = {'hal_gb_ids': tex_ids, 'hal_gb_attrs': tex_attrs,
                    'hal_gb_tris': tex_tris}
            for sname in pbinds.get('samplers', ()):
                if (id(pbinds), sname) in tex_images:
                    bind[sname] = tex_images[(id(pbinds), sname)]
                else:
                    return None, f"'{name}' height pass wants {sname} " \
                                 'but nothing was packed for it'
            extra = {}
            for u in pbinds.get('frame_uniforms', ()):
                if u == 'hal_time':
                    extra[u] = float(getattr(job.scene, 'time', 0.0))
                elif u == 'hal_frame':
                    extra[u] = float(getattr(job.scene, 'frame', 0))
            tgt = device.Target(w, h)
            prepass_targets.append(tgt)
            try:
                device.draw_fullscreen(shader, {**uni, **extra} if extra
                                       else uni, bind, tgt, read=False,
                                       blend='NONE', clear=True)
            except Exception as exc:                            # noqa: BLE001
                for t in prepass_targets:
                    t.free()
                return None, f"'{name}' height pass failed: {exc}"
            prepass_tex[(mat_id, uname)] = device.target_texture(tgt)

    radfield = atlases.get('__radfield')
    if radfield is not None:
        # the interpolated gather's grid pre-pass: drawn ONCE at grid
        # resolution before any material pass, its texture bound to all
        # of them by name through tex_shadows -- the same by-name road
        # the shadow atlases ride
        rsrc, rbinds = radfield
        rspec = {'samplers': list(rbinds.get('samplers', ())),
                 'floats': ['hal_attr_side', 'hal_slot_count',
                            'hal_tri_side'],
                 'vec3': ['hal_eye']}
        rshader, rerr = device.compile_dynamic('HAL_RADFIELD', rsrc, rspec)
        if rshader is None:
            for t in prepass_targets:
                t.free()
            return None, f'the driver rejected the radiosity grid ' \
                         f'pass: {rerr}'
        rbind = {'hal_gb_ids': tex_ids, 'hal_gb_attrs': tex_attrs,
                 'hal_gb_tris': tex_tris}
        for sname in rbinds.get('samplers', ()):
            if sname in rbind:
                continue
            if sname in tex_shadows:
                rbind[sname] = tex_shadows[sname]
            else:
                for t in prepass_targets:
                    t.free()
                return None, f'the radiosity grid pass wants {sname} ' \
                             'but nothing was packed for it'
        gw_r, gh_r = rbinds['size']
        rtgt = device.Target(int(gw_r), int(gh_r))
        prepass_targets.append(rtgt)
        try:
            device.draw_fullscreen(rshader, uni, rbind, rtgt, read=False,
                                   blend='NONE', clear=True)
        except Exception as exc:                                # noqa: BLE001
            for t in prepass_targets:
                t.free()
            return None, f'the radiosity grid pass failed: {exc}'
        tex_shadows['hal_radfield'] = device.target_texture(rtgt)

    plan_draw, err = build_draws(passes, tex_ids)
    if plan_draw is None:
        for t in prepass_targets:
            t.free()
        return None, err

    t_draw = 0.0
    target = device.Target(w, h)
    try:
        # every pass blends into the one target -- each material writes only
        # where its alpha is one, premultiplied blending leaves the rest
        # alone -- and the frame reads back once, however many materials.
        # Per-material readbacks plus a NumPy merge measured 5.9 ms of a
        # 14.3 ms warm frame; this is that line item, removed.
        t1 = _time.perf_counter()
        try:
            for i, (name, shader, bind, extra) in enumerate(plan_draw):
                device.draw_fullscreen(shader, {**uni, **extra} if extra
                                       else uni, bind, target, read=False,
                                       blend='ALPHA_PREMULT', clear=(i == 0))
            got = device.read_target(target)
        except Exception as exc:                                # noqa: BLE001
            # a driver that objects to the blend path gets the readback
            # path, not a CPU frame: slower is better than absent
            print(f'[Halcyon GPU] blended compositing fell back to per-pass '
                  f'readback: {type(exc).__name__}: {exc}')
            got = None
        if got is not None:
            t_draw = _time.perf_counter() - t1
            hit = got[:, :, 3] > 0.5
            # no masking needed: the target was cleared to zero and the
            # blend leaves untouched pixels at zero, so the colour planes
            # are already exactly what a mask would have produced. The
            # where() this replaces was most of the composite slice
            out = np.ascontiguousarray(got[:, :, :3], np.float32)
        else:
            out = np.zeros((h, w, 3), np.float32)
            hit = np.zeros((h, w), bool)
            for name, shader, bind, extra in plan_draw:
                t1 = _time.perf_counter()
                try:
                    frame = device.draw_fullscreen(shader, {**uni, **extra}
                                                   if extra else uni, bind,
                                                   target)
                except Exception as exc:                        # noqa: BLE001
                    return None, f"drawing '{name}' failed: {exc}"
                t_draw += _time.perf_counter() - t1
                keep = frame[:, :, 3] > 0.5
                out[keep] = frame[keep, :3]
                hit |= keep
    finally:
        target.free()
        for t in prepass_targets:
            t.free()

    # the CPU-composite environment term: for worlds richer than the
    # baked GLSL paths, the renderer's own world_color along the
    # reflected rays, added exactly where the CPU adds it (last)
    env_plan = atlases.get('__env')
    try:
        out = _apply_cpu_env_primary(job, gbuf,
                                     (env_plan or {}).get('primary'), out)
    except Exception as exc:                                    # noqa: BLE001
        return None, f'the environment composite failed: {exc}'

    # the traced bounces: rays off the reflective then refractive pixels,
    # closest hits shaded by the SAME materials through their secondary
    # passes, each blend composited exactly as _add_raytraced does it,
    # in _add_raytraced's order
    t_reflect = 0.0
    _RAY_BUILD[0] = 0.0
    if rplan is not None:
        t1 = _time.perf_counter()
        from . import rtrace as RT

        def draw_secondary(plist, sec_ids, level):
            try:
                tex_sec = device.upload(sec_ids)
            except Exception as exc:                            # noqa: BLE001
                raise _SweepFail(f'uploading the level-{level} ray '
                                 f'buffer failed: {exc}')
            tag = 'HAL_RMAT' if plist is rplan['secondary'] else 'HAL_RMATM'
            sec_draw, err = build_draws(plist, tex_sec, tag=tag)
            if sec_draw is None:
                raise _SweepFail(str(err))
            target2 = device.Target(w, h)
            try:
                for i2, (name, shader, bind, extra) in enumerate(sec_draw):
                    device.draw_fullscreen(shader, {**uni, **extra}
                                           if extra else uni, bind,
                                           target2, read=False,
                                           blend='ALPHA_PREMULT',
                                           clear=(i2 == 0))
                sec_img = device.read_target(target2)
            except _SweepFail:
                raise
            except Exception as exc:                            # noqa: BLE001
                raise _SweepFail(f'the level-{level} ray passes '
                                 f'failed: {exc}')
            finally:
                target2.free()
            return sec_img

        def isect(org, dirs):
            got_hits, why_r = RT.intersect_frame(job.bvh, org, dirs)
            if got_hits is None:
                raise _SweepFail(f'the ray trace failed: {why_r}')
            return got_hits

        try:
            out = _run_sweeps(job, gbuf, rplan, out, draw_secondary, isect,
                              env=env_plan)
        except _SweepFail as sf:
            return None, str(sf)
        except Exception as exc:                                # noqa: BLE001
            # shade_frame's contract is that EVERY failure is a reason
            # and the caller shades on the CPU -- the field's depth-2
            # section died whole because a ValueError escaped this loop
            # instead of becoming one
            return None, (f'the ray sweeps failed: '
                          f'{type(exc).__name__}: {exc}')
        t_reflect = _time.perf_counter() - t1

    total = _time.perf_counter() - t_all
    LAST_TIMINGS.clear()
    LAST_TIMINGS.update(
        plan_ms=t_plan * 1000.0,
        pack_upload_ms=t_upload * 1000.0,
        draw_read_ms=t_draw * 1000.0,
        reflect_ms=t_reflect * 1000.0,
        ray_build_ms=_RAY_BUILD[0],
        composite_ms=max(total - t_upload - t_draw - t_reflect, 0.0)
        * 1000.0,
        passes=len(passes))
    out = _fog_readback(job, gbuf, passes, out, hit)
    return out, hit
