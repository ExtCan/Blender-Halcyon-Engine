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
                 # R251 lighting: the period finish dials (F006, F019-F021)
                 'fog_burn', 'fog_bias', 'fog_bank', 'brilliance', 'crand',
                 'pov_metallic',
                 'fresnel_blend', 'rim_blend', 'matcap_mode',
                 'reflect', 'refraction', 'edge_opacity',
                 # the BI panel round's CPU-consumed ray constants
                 'ray_ior', 'bi_ray_filter', 'bi_mir_fresnel',
                 'bi_mir_blend', 'use_mist',
                 # R211 punch-through: >= 0 is the CLIP threshold; the
                 # layer/stipple chains emit the CPU's hard 0/1 law
                 'alpha_clip')
EXTRA_COLORS = ('anime_shine_color2',
                # R243: the Max Multi-Layer's second highlight colour, the
                # Max Translucent's colour
                'specular2', 'translucent_color',
                'fresnel_color', 'rim_color', 'sheen_color', 'matcap',
                'backface_color', 'reflect_color',
                # the anime tone colours (R218)
                'anime_shadow1', 'anime_shadow2',
                # the cartoon paint tones (R228)
                'cartoon_shadow', 'cartoon_hl_color',
                # the 80s anime additions (R229)
                'anime_shine_color', 'anime_air_color',
                # R238: the cel's key, in its own frame
                'cel_dir')

#: models the GLSL dispatch reproduces at pixel rate. GOURAUD and FLAT are
#: shading rates (the corner-light road carries them); WIREFRAME and
#: CONSTANT are shadeless -- light_surface returns early, so the pass
#: emits diffuse x level + emission and no lighting support at all
#: (apply_wireframe carves the wires on the readback either way).
from ..core import shading as _SHM
from ..core import reyes as REYES        # R251 C119 (MAT-B)
# R251 (LIGHT-B2): SEGA_MODEL2 / SEGA_MODEL3 are their rates (FACE /
# VERTEX, render.RATE_FOR_MODEL): the corner road is the machine on
# both devices and a PIXEL-rate request refuses here by name
UNSUPPORTED_MODELS = frozenset({'GOURAUD', 'FLAT'}) | _SHM.UNSUPPORTED_MODELS
SHADELESS_MODELS = frozenset({'WIREFRAME', 'CONSTANT'})


def _plot_ss(st):
    from ..core import combine as _CBS
    return int(_CBS.plot_ss(st))


def _model_index(model):
    from ..core.shading import MODEL_ITEMS
    if isinstance(model, str) and model.startswith('BI_MATRIX_'):
        # the BI material node: 100 + diffuse*10 + spec, decoded by the
        # GLSL dispatch's matrix branch
        try:
            return 100 + int(model[10]) * 10 + int(model[12])
        except (ValueError, IndexError):
            return None
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


def _cel_field_of(job, gbuf):
    """R238: the frame's cel field -- the render's own lazy builder when
    the job carries one, else computed here from the job's camera (a
    job built by hand, as the parity harness does), None when no
    material reads it."""
    lazy = getattr(job, 'cel_lazy', None)
    if lazy is not None:
        return lazy()
    from ..core import celfield as CF
    if not CF.field_on(job.scene, job.settings):
        return None
    from ..core.render import camera_matrices
    view, proj, vp, _eye = camera_matrices(job.scene.camera, job.width,
                                           job.height)
    field = CF.compute(job.scene, job.scene.mesh, gbuf, view, proj, vp,
                       getattr(job.scene, 'camera', None), job.settings)
    job.cel_lazy = lambda f=field: f
    return field


def _cel_atlas_entry(field):
    """R238: the cel field as an upload-cached texture entry: .r the
    screen shadow, .g the depth rim, keyed on the field's own bytes
    (a frame's field is per-frame data, like the lamps' values)."""
    import zlib
    ss = np.asarray(field['ss'], np.float32)
    rim = np.asarray(field['rim'], np.float32)
    key = ('celfield', int(ss.shape[1]), int(ss.shape[0]),
           int(zlib.adler32(ss.tobytes())), int(zlib.adler32(rim.tobytes())))

    def _build(_ss=ss, _rim=rim):
        img = np.zeros(_ss.shape + (4,), np.float32)
        img[..., 0] = _ss
        img[..., 1] = _rim
        img[..., 3] = 1.0
        return img

    return key, _build


def _cam_uniforms(job):
    """R238: the camera's (right, up, back) axes as frame uniforms --
    a key fixed to the camera composes its direction on them in the
    shader, so an orbit changes three uniforms and re-plans nothing."""
    from ..core.celfield import camera_axes
    r, u, b = camera_axes(getattr(job.scene, 'camera', None))
    return {'hal_cam_right': tuple(float(v) for v in r),
            'hal_cam_up': tuple(float(v) for v in u),
            'hal_cam_back': tuple(float(v) for v in b)}


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
    route_slot = None
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
            kinds = {k for k, w, _p in cl.items
                     if bool(np.any(np.asarray(w) > 1e-6))}
            offside = sorted(kinds - {'DIFFUSE', 'TRANSPARENT',
                                      'HALCYON'})
            if offside == ['GLOSSY']:
                # THE SPECULAR SLOT ROUTING (queued since the node
                # shelf round): a lone GLOSSY lobe's colour chain IS
                # the specular colour -- closure_to_surface puts it in
                # surf.specular -- so the pass routes the emitted chain
                # to s.specular and bakes the untouched flat diffuse.
                # With a DIFFUSE lobe alongside (the Specular BSDF's
                # spec/gloss pair), the emitted chain is the DIFFUSE
                # lobe's own colour and stays in its slot; the glossy
                # side's colour must then bake, which the specular
                # constancy rule below enforces by name.
                route_slot = 'diffuse' if 'DIFFUSE' in kinds \
                    else 'specular'
            elif offside:
                return None, None, (
                    'a raw ' + '/'.join(offside) + ' lobe rides the '
                    'specular or emission slot, and the frame pass emits '
                    'only the diffuse chain -- this material shades on '
                    'the CPU exactly (Convert to Halcyon Shader puts it '
                    'on the driver)')
    if mat is not None and getattr(mat, 'shadeless', False):
        return None, None, 'shadeless materials take the early CPU path'
    from ..core.render import bi_sss_params
    _sssp = bi_sss_params(mat) if mat is not None else None
    if _sssp is not None and getattr(st, 'sss', True):
        # the octree gather HAS a GLSL twin (hal_sss_sample: the tree
        # as a data texture, stackless hit/miss traversal in the C's
        # own visit order -- measured at float32 exactness against the
        # CPU tree). The pass needs the pre-pass tree; without one the
        # frame refuses by name rather than silently skipping the
        # scatter.
        entry = (getattr(job, 'sss_trees', None) or {}).get(int(mi))
        if entry is None:
            return None, None, ('uses Blender Internal subsurface '
                                'scattering but the pre-pass tree is '
                                'missing; the material shades on the '
                                'CPU')

    surf, model, nrm = closure_to_surface(cl, ctx, st, mat)
    if nrm is not None:
        # a bent normal qualifies exactly when the frame shader will bend
        # it identically: the master node's Normal chain, which the
        # assembler emits. Any other source of one still shades on the CPU.
        from .material import master_faceted, master_normal_linked
        _g = getattr(mat, 'graph', None) if mat is not None else None
        # R242: Max's Faceted bends it to the stored face normal, which
        # the assembler substitutes (hal_triaux) exactly as the CPU does
        if not master_normal_linked(_g) and not master_faceted(_g):
            return None, None, ('the graph bends the shading normal outside '
                                "the master shader's Normal socket")
    from ..core import combine as _CBR
    rate = str(_CBR.rate_for_model(model, st, mat)
               or RATE_FOR_MODEL.get(model, st.shading_rate))
    if rate == 'PIXEL' and np.any(surf.light_limit > 0.5):
        # R252: the machine's light limit is the corner road's rule (the
        # CPU lights the first N lamps); a pixel-rate pass emits every
        # lamp, so it refuses by name
        return None, None, ("the material's console light limit is a "
                            'per-vertex rule; at the pixel rate the frame '
                            'shades on the CPU, by name')
    if rate not in ('PIXEL', 'VERTEX', 'FACE'):
        return None, None, f'unknown shading rate {rate}'
    if rate != 'PIXEL' and (layer or secondary):
        which = 'hit' if secondary else 'transparent-layer'
        return None, None, (f'{model or "the material"} shades at {rate} '
                            f'rate, and a {which} pass lights per pixel '
                            f'-- the light loop has no {model} formula')
    if (layer or secondary) and REYES.rate_of(st) > 0.0:
        # R251 C119 (MAT-B): only the camera G-buffer carries the
        # barycentrics the snap reads
        return None, None, REYES.REFUSE_PASS
    if rate == 'PIXEL':
        if model in SHADELESS_MODELS:
            idx = 0            # never dispatched: the pass emits no lights
        elif model in UNSUPPORTED_MODELS:
            return None, None, \
                f'the {model} model shades outside the light loop'
        else:
            idx = _model_index(model)
            if idx is None:
                return None, None, f'unknown model {model}'
    else:
        # a vertex- or face-rate pass never LIGHTS in GLSL: the CPU
        # computes the corner lighting -- shadows, rays, env, the
        # model's own formula -- and the pass interpolates it, so the
        # model needs no GLSL dispatch entry at all
        idx = 0

    if mat is not None and \
            str(getattr(mat, 'blend_mode', 'INHERIT')) == 'ENV_HOLE':
        # R251 C126 (Blender 2.4x Env): the hole needs the world
        # evaluator per COVERED pixel, which only the CPU sky road
        # provides today (gpu/sky.py draws uncovered pixels only; the
        # env-reflection road applies the world on the CPU over the
        # readback). cpu_only, by name -- lifted the day the sky pass
        # draws through holes (a coverage-mask edit in gpu/sky.py)
        return None, None, (f"material '{getattr(mat, 'name', mi)}' is "
                            'an Env hole (the world along the view ray, '
                            'alpha 0); the frame shades on the CPU')
    stipple_mode = str(getattr(st, 'transparency', 'SORTED')) == 'STIPPLE'
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
        if name == 'opacity' and mat is not None:
            from ..core.scene import clip_road
            if clip_road(mat)[0] is not None:
                # R211/R213 punch-through: this material's visibility
                # is resolved in the z-pass BEFORE shading and the law
                # forces alpha to 1 at every promoted pixel, so the
                # opacity chain cannot reach the picture through the
                # frame pass -- it is inert here by construction
                continue
        if stipple_mode and name == 'opacity':
            # Screen Door: opacity feeds the ordered threshold, not a
            # blend -- the frame pass emits the CPU's own chain (clamp,
            # hard cutoff, then keep-or-drop against the very threshold
            # map the CPU tiles). A CONSTANT opacity bakes; the
            # constancy loop below still refuses a varying one unless
            # the graph computes it per pixel, which stays refused here
            continue
        field = getattr(surf, name, None)
        if field is None:
            continue
        if float(np.abs(np.asarray(field) - inert).max()) > 1e-4:
            return None, None, f'the material uses {name}, which needs the ' \
                               f'alpha compositing the deferred target ' \
                               f'does not do (Transparency NONE shades it)'
    # (R213: Opacity is a per-pixel socket now -- a linked chain is
    # emitted and both devices threshold the same per-pixel alpha; an
    # opacity varying WITHOUT a granted chain still refuses through the
    # BAKE_FIELDS constancy rule below)

    bake = {'__rate': rate, '__model': model}
    if rate != 'PIXEL':
        # R251 (MAT-A): the period items' CPU gates (a linked DS Toon
        # Size, the Super FX palette / gamma gates) refuse the pass by
        # name with the CPU's own text; the DS table rides the bake
        from ..core import combine as _CB
        try:
            _tab = _CB.probe_gate(model, mat, st)
        except _CB.Refusal as _r:
            return None, None, f'{model}: {_r}'
        if _tab is not None:
            bake['__ds_table'] = tuple(float(v) for v in _tab)
    if rate == 'PIXEL' and model in SHADELESS_MODELS:
        bake['__shadeless'] = True
    if rate == 'PIXEL' and np.any(surf.fixed_shade > 0.5):
        # R252: fixed shading at the pixel rate IS the shadeless pass
        # (diffuse x level + emission, light_surface's early return);
        # the RenderWare prelight never reaches here (rate-fixed VERTEX)
        if not np.all(surf.fixed_shade > 0.5):
            return None, None, ('fixed shading varies across the frame; '
                                'the frame shades on the CPU')
        bake['__shadeless'] = True
    # R252: the Console node's combine-stage options (the texture op,
    # the Saturn half, the luma ramp, the PCX base, the plot's dither)
    # ride the bake for gpu/combine.recombine_lines
    bake['__console'] = _CBR.console_opts(mat) if mat is not None else {}
    if mat is not None and \
            str(getattr(mat, 'alpha_mode', 'BLEND')) == 'CLIP_BLEND':
        # R251 C031: the layer emit keeps the sub-threshold alpha
        bake['__clip_blend'] = True
    from .material import BAKE_FIELDS, per_pixel_fields
    # fields a linked master-node socket will compute per pixel are exempt
    # from the constancy rule -- varying is their whole point, and the
    # assembler emits the same chain the evaluator just ran
    perpix = set(per_pixel_fields(getattr(mat, 'graph', None)
                                  if mat is not None else None))
    for name in BAKE_FIELDS:
        if name == 'opacity' and name in perpix and stipple_mode:
            # R213 scope: the layer road takes the per-pixel Opacity
            # chain (compositing is continuous; an ulp is an ulp), but
            # Screen Door thresholds it against the ordered map -- a
            # keep/drop CLIFF where a driver's last-bit rounding flips
            # whole pixels. Cliffs refuse by name here.
            return None, None, ('a per-pixel Opacity under Screen Door '
                                'is a keep/drop cliff; the frame shades '
                                'on the CPU, by name')
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
        if name == 'specular' and route_slot == 'specular':
            # the routed chain IS the specular colour: the pass emits
            # it per pixel, so its variation is the whole point
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
    if route_slot == 'specular':
        # the specular slot routing: the emitted chain becomes
        # s.specular, so s.diffuse must carry the CPU's own untouched
        # flat value as a constant -- and it must BE flat, or the pass
        # cannot represent the surface
        if not ok_d:
            return None, None, ('a glossy-routed graph varies its '
                                'diffuse per pixel; the material '
                                'shades on the CPU')
        bake['diffuse'] = tuple(float(v)
                                for v in np.asarray(surf.diffuse)[0])
        bake['__slot'] = 'specular'
    if _sssp is not None and getattr(st, 'sss', True):
        # the assembler binds the tree texture and emits the diffuse
        # replacement (shade_lamp_loop's block); texfac shapes the col
        bake['__sss'] = {'texfac': float(_sssp.get('texfac', 0.0)),
                         'key': f'__sss_tree_{int(mi)}__'}
    # R164: the terminator fix and the object-colour modulation
    from ..core.render import bi_node_props as _bnp
    _bip = _bnp(mat) if mat is not None else None
    if _bip:
        if _bip.get('raybias') and st.shadows and \
                any(getattr(l, 'shadow', 'NONE') == 'RAY'
                    or str(getattr(st, 'shadow_default', '')) == 'RAY'
                    for l in job.lights):
            # R167: the RAYBIAS terminator fix HAS its GLSL twin now --
            # the per-face smooth proxy reads the stored face normal
            # from hal_triaux, the object's Auto Smooth threshold rides
            # a per-tri hal_sres texel, and the curve is the sbias
            # branch's, guarded like the CPU's _pcurve. (This refusal
            # was the field's 8:48 frame: THREE materials in
            # a 2013-era file carry MA_RAYBIAS, and the whole frame's
            # shading fell back to the CPU for a correction the GPU can
            # run exactly.) Secondary passes still refuse in the
            # assembler: ray hits carry no G-buffer triangle id to
            # fetch the texels by.
            if getattr(job.scene.mesh, 'face_normals', None) is None:
                return None, None, ('Ray Bias terminator correction '
                                    'needs stored face normals this '
                                    'mesh does not carry; the material '
                                    'shades on the CPU')
            bake['__raybias'] = True
        if _bip.get('use_obcolor'):
            _sss_done = _sssp is not None and (
                (getattr(job, 'sss_trees', None) or {}).get(int(mi))
                is not None or not getattr(st, 'sss', True))
            if not _sss_done:
                if any(nd.get('props', {}).get('use_transparency')
                       for nd in (mat.graph or {}).get('nodes',
                                                       {}).values()
                       if nd.get('bl_idname')
                       == 'HALCYON_BIMaterialNode'):
                    return None, None, ('Object Color with '
                                        'transparency modulates the '
                                        'alpha per object; the '
                                        'material shades on the CPU')
                colored = [
                    (i, tuple(np.clip(np.asarray(
                        getattr(o, 'color', (1, 1, 1, 1)),
                        np.float32)[:3], 0.0, None)))
                    for i, o in enumerate(
                        getattr(job.scene, 'objects', ()) or ())
                    if tuple(np.round(np.asarray(
                        getattr(o, 'color', (1, 1, 1, 1)),
                        np.float32)[:3], 6)) != (1.0, 1.0, 1.0)]
                if len(colored) > 24:
                    return None, None, (f'Object Color with '
                                        f'{len(colored)} coloured '
                                        'objects exceeds the ladder; '
                                        'the material shades on the '
                                        'CPU')
                if colored:
                    bake['__obcolor'] = colored
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
    if mode in ('STARFIELD', 'BRYCE', 'PHYSICAL', 'HDRI', 'PAINTED',
                'CUBEMAP'):          # R253: the skybox is an environment
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
    if getattr(light, 'type', '') == 'HEMI':
        # BI never shadowed hemi lamps and the CPU's visibility()
        # returns unconditional ones for them -- a tap here referenced
        # variables the HEMI branch never declares (latent compile
        # break, found by the R164 phongcorr test)
        return None, None, None
    mode = light.shadow if st.shadow_default == 'PER_LIGHT' else \
        st.shadow_default
    # R251 C052: PLANAR before the ray fallback, the CPU's `visibility`
    # gate exactly (the polygon is a readback edit, not a lighting term)
    if mode in ('NONE', 'PLANAR'):
        return None, None, None
    sm = getattr(light, 'shadow_map', None)
    # the CPU's gate exactly: ray_shadows is the master switch for traced
    # shadows, RAY mode included (see lights.shadow)
    if getattr(st, 'ray_shadows', True) and (
            mode == 'RAY' or (sm is None and bvh is not None)):
        # R208: the caster mask (Material.cast_shadow and friends) is a
        # BVH cast filter on the CPU; the GLSL any-hit has no filter,
        # so a frame with excluded casters routes its ray shadows to
        # the CPU by name rather than shadowing with the wrong casters
        if getattr(st, '_shadow_cast_tri', None) is not None:
            return None, None, ('a material or object opts out of '
                                'shadow casting; ray shadows honour '
                                'the caster mask on the CPU')
        # exactly `visibility`'s RAY branch, decided per light
        if bvh is None:
            # the CPU only builds a BVH for shadow_default RAY (or the ray
            # features the plan has already refused), and without one the
            # RAY branch returns fully lit. Mirror that: no shadow term.
            return None, None, None
        kind = str(getattr(light, 'type', 'POINT')).upper()
        area = None
        if kind == 'AREA':
            asz = getattr(light, 'area_size', (0.0, 0.0))
            hx, hy = float(asz[0]) * 0.5, float(asz[1]) * 0.5
            soft_on = hx > 0.0 or hy > 0.0
            if soft_on:
                ax = np.asarray(getattr(light, 'area_x', (1, 0, 0)),
                                np.float32)
                ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
                ay = np.asarray(getattr(light, 'area_y', (0, 1, 0)),
                                np.float32)
                ay = ay / max(float(np.linalg.norm(ay)), 1e-9)
                area = {'pos': tuple(float(v) for v in light.position),
                        'ax': tuple(float(v) for v in ax),
                        'ay': tuple(float(v) for v in ay),
                        'hx': hx, 'hy': hy,
                        'disk': str(getattr(light, 'area_shape', 'SQUARE'))
                        in ('DISK', 'ELLIPSE')}
        else:
            soft_on = float(getattr(light, 'radius', 0.0)) > 0.0
        samples = max(1, int(getattr(st, 'shadow_samples', 1))) \
            if soft_on else 1
        # soft ray shadows travel now: the jitter is a pure function of
        # (pixel, sample, light, seed) through the pattern hash and the
        # shared unit-circle table, so both devices draw the SAME rays.
        # SUN carries its size as an ANGLE and AREA as its rectangle --
        # the same per-kind meaning visibility() gives them on the CPU
        meta = {'ray': True,
                'bias': max(float(getattr(st, 'ray_bias', 1e-3)), 1e-4),
                'radius': float(getattr(light, 'radius', 0.0)),
                'kind': kind, 'area': area,
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
    # R251 C117: Woo's MIDPOINT map -- the texels are the halfway point
    # between the two nearest casters, and the compare carries no bias
    # and no normal offset (the CPU's `visibility` rule, one resolve)
    from ..core.lights import midpoint_map as _midpoint_map
    mid = bool(_midpoint_map(light, st))

    # packed lazily, behind a content fingerprint: the CPU caches these maps
    # across frames, and re-packing + re-uploading an unchanged 33 MB cube
    # atlas every frame was most of the warm frame's cost
    key = ('shadow', size, grid, 'MID' if mid else 'CLS',
           tuple(round(float(f.depth[::23].sum()), 3) for f in faces_sm))

    def build(_faces=faces_sm, _size=size, _grid=grid):
        atlas = np.zeros((_size * _grid[1], _size * _grid[0], 4), np.float32)
        for fi, f in enumerate(_faces):
            cy, cx = (fi // _grid[0]) * _size, (fi % _grid[0]) * _size
            atlas[cy:cy + _size, cx:cx + _size, 0] = f.depth
        return atlas

    bias = float(getattr(light, 'shadow_bias', 0.0) or st.shadow_bias)
    if mid:
        bias = 0.0
    soft = max(float(getattr(light, 'shadow_softness', 1.0))
               * float(st.shadow_softness), 0.0) \
        + float(getattr(sm, 'soft_extra', 0.0) or 0.0)
    meta = {
        'faces': [{'vp': np.asarray(f.vp, np.float32)} for f in faces_sm],
        'size': size, 'near': float(first.near), 'far': float(first.far),
        'persp': bool(first.persp), 'extent': float(first.extent),
        'origin': tuple(float(v) for v in origin), 'grid': grid,
        'bias': bias, 'softness': soft,
        'density': float(getattr(light, 'shadow_density', 1.0)),
        'midpoint': mid,
    }
    return meta, (key, build), None


def _light_sig(l):
    """One light's contribution to the plan signature.

    R169: the light's tweakable VALUES -- position, direction, colour,
    energy, spot cone, decay distances, the Negative sign -- ride the
    hal_lights texture now and are deliberately ABSENT here, so a lamp
    edit is a plan-cache HIT plus a texel re-upload instead of the
    recompile storm the field measured at 45 seconds per viewport
    refine. What stays is STRUCTURE: everything that changes the
    emitted source. Two exceptions keep their values: a MAP-shadowed
    light bakes its light-space matrices into the shadow function
    (moving it re-plans -- its map rebuilt anyway), and a cookie light
    bakes its projection axes from the direction.
    """
    import numpy as np
    t = lambda v: tuple(round(float(x), 6) for x in v)
    spotsi = float(np.cos(float(getattr(l, 'spot_size', 0.0)) * 0.5))
    spotbl = (1.0 - spotsi) * float(getattr(l, 'spot_blend', 0.0))
    baked_map = getattr(l, 'shadow_map', None) is not None
    cookie = getattr(l, 'cookie', None)
    sig = (l.type,
           getattr(l, 'decay', 'DEFAULT'),
           # attenuation STRUCTURE: which factors exist (their widths
           # and distances are texels)
           float(getattr(l, 'decay_start', 0.0)) > 0.0,
           float(getattr(l, 'decay_ld1', 0.0) or 0.0) > 0.0,
           float(getattr(l, 'decay_ld2', 0.0) or 0.0) > 0.0,
           bool(getattr(l, 'bi_sphere', False)),
           spotbl != 0.0,
           getattr(l, 'shadow', 'MAP'),
           # per-lobe gates emit different blocks (and were MISSING
           # from the old signature -- a latent stale-plan bug, fixed
           # in passing)
           bool(getattr(l, 'affect_diffuse', True)),
           bool(getattr(l, 'affect_specular', True)),
           bool(getattr(l, 'specular_only', False)),
           bool(getattr(l, 'diffuse_only', False)),
           # R251 LIGHT-B1: the only-shadow block (F014), the cone law
           # and whether its exponent is nonzero (F012), the screen
           # spot (F015) each emit a different block -- structure
           bool(getattr(l, 'only_shadow', False)),
           str(getattr(l, 'spot_law', 'BLENDER') or 'BLENDER'),
           float(getattr(l, 'spot_exponent', 0.0) or 0.0) != 0.0,
           bool(getattr(l, 'screen_spot', False)),
           # light linking bakes a per-light object ladder into the
           # pass source (R78: a bake the plan reads MUST be in the
           # signature)
           tuple(sorted(getattr(l, 'exclude_objects', ()) or ())),
           str(getattr(l, 'exclude_mode', 'EXCLUDE')),
           # the soft-ray shadow function bakes the radius and sample
           # unroll; the shadow colour folds into every light block
           round(float(getattr(l, 'radius', 0.0)), 6),
           t(getattr(l, 'shadow_color', (0, 0, 0))))
    if str(getattr(l, 'type', '')).upper() == 'AREA':
        # the AREA soft-ray twin bakes the rectangle -- centre, axes,
        # half-sizes, shape -- as literals (the radius precedent), so
        # all of them are structure
        sig += ('area',
                t(getattr(l, 'area_size', (0.0, 0.0))),
                t(getattr(l, 'area_x', (1, 0, 0))),
                t(getattr(l, 'area_y', (0, 1, 0))),
                t(getattr(l, 'position', (0, 0, 0))),
                str(getattr(l, 'area_shape', 'SQUARE')),
                # the form factor bakes Gamma (and emits no pow at the
                # 1.0 default); its distance and direction are texels
                round(float(getattr(l, 'area_gamma', 1.0) or 1.0), 6))
    if baked_map:
        sig += ('map',
                t(getattr(l, 'position', (0, 0, 0))),
                t(getattr(l, 'direction', (0, 0, -1))),
                round(float(getattr(l, 'spot_size', 0.0)), 6),
                round(float(getattr(l, 'shadow_bias', 0.0)), 6),
                round(float(getattr(l, 'shadow_softness', 1.0)), 6),
                round(float(getattr(l, 'shadow_density', 1.0)), 6),
                # R251 C117: the per-lamp Map Depth override changes
                # the emitted offset line
                str(getattr(l, 'shadow_map_depth', 'INHERIT')))
    if cookie is not None:
        # a projected texture bakes its frame, strength and size into
        # the pass, and its pixels ride the upload cache: swap or edit
        # the image and the plan must rebuild. R219: the roll axes are
        # baked literals too (rolling a lamp about its beam kept the
        # direction and served a stale frame -- fixed in passing), and
        # the extension/filter choices change the emitted lookup
        sig += ('cookie',
                t(getattr(l, 'direction', (0, 0, -1))),
                t(getattr(l, 'frame_x', None) or (0.0,)),
                t(getattr(l, 'frame_y', None) or (0.0,)),
                round(float(getattr(l, 'cookie_strength', 1.0)), 6),
                round(float(getattr(l, 'cookie_scale', 10.0)), 6),
                str(getattr(l, 'cookie_extend', 'AUTO') or 'AUTO'),
                str(getattr(l, 'cookie_filter', 'BILINEAR')
                    or 'BILINEAR'),
                _cookie_sig(l))
    return sig


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
            # R251 (transparency pack): the per-material blend equation
            # and the alpha road are plan gates (Env hole, Clip+Blend)
            str(getattr(m, 'blend_mode', 'INHERIT')),
            str(getattr(m, 'alpha_mode', 'BLEND')),
            hash(repr(graph)) if graph else None)


#: R216: node types whose output moves with the CLOCK. A material
#: carrying one bakes probed constants (a rim colour fed by an
#: Oscillator, a matcap crossfaded by Frame Blend) that are only right
#: for the frame that probed them -- so the plan signature gains the
#: frame for such scenes, values re-probe, and the SOURCES stay
#: byte-identical (they carry texels, not literals): fresh values,
#: zero recompiles. The field's report: "fresnel/rimlight/matcap
#: colors don't update every frame".
_TIME_NODES = frozenset({
    'HALCYON_TimerNode', 'HALCYON_OscillatorNode', 'HALCYON_CounterNode',
    'HALCYON_PulseNode', 'HALCYON_WobbleNode', 'HALCYON_FrameBlendNode',
    'HALCYON_OnFrameNode', 'HALCYON_ColorCycleNode',
    'HALCYON_FlipbookNode', 'HALCYON_StepTimeNode', 'HALCYON_WaveNode',
    'HALCYON_ScrollNode'})
#: these honour an 'animate' prop (default on)
_TIME_NODES_ANIMATE = frozenset({
    'HALCYON_StaticNode', 'HALCYON_WaterNode', 'HALCYON_CausticsNode',
    'HALCYON_PlasmaNode', 'HALCYON_RipplesNode', 'HALCYON_UVWaveNode',
    'HALCYON_RippleWarpNode', 'HALCYON_WaveWarpNode',
    'HALCYON_OrbitNode', 'HALCYON_SpinNode',
    # R251 C099 (MAT-B): Shimmer re-randomises per frame (its prop is
    # named `animate` so this check reads it as it stands)
    'HALCYON_ImagineRoughnessNode'})


def _scene_time_dependent(scene):
    for m in (getattr(scene, 'materials', ()) or ()):
        g = getattr(m, 'graph', None)
        if not g:
            continue
        for nd in (g.get('nodes', {}) or {}).values():
            bid = nd.get('bl_idname', '')
            if bid in _TIME_NODES:
                return True
            if bid in _TIME_NODES_ANIMATE and \
                    (nd.get('props') or {}).get('animate', True):
                return True
    return False


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
        # R251 texture pack: the prep-time CONTENT laws (belt and braces
        # beside tex_sig's strided sum) and every sample-time dial that
        # sample_opts reads (gates of the emission site and of the aniso
        # rule on both roads)
        'tex_format', 'tex_tmem_format', 'tex_compress', 'tex_frac_bits',
        'tex_clamp_mode', 'tex_colorkey', 'tex_colorkey_range',
        'tex_mip_select', 'tex_lod_sharpen', 'tex_lod_source', 'tex_lod_k',
        'tex_lod_l',
        # R253: the Generated-coordinate road is a BAKE the plan reads
        # (the object-space line or the legacy world line); a moved
        # object re-plans through _mesh_key already
        'generated_space',
        # R251 material pack (MAT-B): the REYES snap is a gate AND a
        # bake (C119), and the keys the pack found missing -- the AA
        # pair (the internal size the grid is sized for), the palette
        # trio the Palette node bakes and the display pair the
        # specular-in-gamma road reads
        'shading_rate_area', 'aa_mode', 'aa_samples',
        'palette_mode', 'palette_size', 'palette_colors', 'gamma',
        'color_management',
        'transparency', 'env_reflection',
        # the Screen Door threshold map bakes from the pattern (R78: a
        # bake the plan reads MUST be in this signature)
        'stipple_pattern',
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
        'alpha_threshold',
        # R251 C117: the map depth changes the emitted offset line (a
        # bake the plan reads MUST be in this signature)
        'shadow_map_depth',
        # R251 (lighting): the fog block's STRUCTURE -- which hal_fog
        # branches a pass emits -- and the hal_fogtab sampler gate
        # (specular_viewer AXIS reads the same texture, F011)
        'fog_mode', 'fog_vertex', 'fog_height', 'fog_table', 'fog_dither',
        'fog_depth', 'specular_viewer',
        # R251 (LIGHT-B2 F020): crand's hash folds the frame in
        # (structure: the hal_frame uniform)
        'crand_per_frame',
        # R251 (MAT-A C049): the Super FX plot's supersample bake and
        # its palette / gamma GATES (a gate the plan reads MUST be here)
        'aa_mode', 'aa_samples', 'palette_mode', 'palette_size',
        'palette_colors', 'gamma', 'color_management')) + (
        int(getattr(st, 'fog_bands', 0) or 0) >= 2,
        # the module switch IS a gate the plan reads (B2)
        bool(FOG_ON_GPU))
    # R251 LIGHT-B1 (F015): whether the fog reads the screen-spot lobe
    st_sig = st_sig + (float(getattr(st, 'fog_spot', 0.0) or 0.0) > 0.0,)
    # R251 LIGHT-A2 (F004 / F005 / F008 / F006 / F010): the wave-2 fog
    # STRUCTURE -- which hal_fog branches a pass emits (core/fog.structure)
    st_sig = st_sig + (
        bool(getattr(st, 'fog_range_adjust', False)),
        bool(getattr(st, 'fog_face', False)),
        str(getattr(st, 'fog_color_source', 'FIXED')),
        float(getattr(st, 'fog_bank1_end', 0.0) or 0.0)
        > float(getattr(st, 'fog_bank1_start', 0.0) or 0.0),
        float(getattr(st, 'fog_turbulence', 0.0) or 0.0) > 0.0)
    world = getattr(scene, 'world', None)
    world_sig = None
    if world is not None:
        world_sig = (tuple(getattr(world, 'ambient', (0, 0, 0))),
                     float(getattr(world, 'ambient_level', 0.0)),
                     # exposure bakes linfac/logfac into every pass
                     float(getattr(world, 'exposure', 0.0) or 0.0),
                     float(getattr(world, 'exposure_range', 1.0)
                           or 1.0),
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
    obcol_sig = tuple(
        tuple(np.round(np.asarray(getattr(o, 'color', (1, 1, 1, 1)),
                                  np.float32), 5))
        for o in (getattr(scene, 'objects', ()) or ()))
    # R216: image CONTENT joins the signature (a strided sum per image)
    # -- an image sequence advancing, or a repainted texture, must
    # re-plan so the mip atlases re-upload; a name alone hid both
    tex_sig = ()
    try:
        tex_sig = tuple(sorted(
            (str(k), float(np.asarray(v.pixels)[::173, ::173].sum()))
            for k, v in (getattr(job, 'textures', {}) or {}).items()
            if getattr(v, 'pixels', None) is not None))
    except Exception:                                           # noqa: BLE001
        tex_sig = ()
    clock = (int(getattr(scene, 'frame', 0)),
             round(float(getattr(scene, 'time', 0.0)), 6)) \
        if _scene_time_dependent(scene) else None
    return (mkey, st_sig, world_sig, shadow_sig, obcol_sig, tex_sig,
            clock,
            # whether a BVH exists decides the RAY branch (lit vs traced),
            # and its content is the mesh's, which mkey already fingerprints
            getattr(job, 'bvh', None) is not None,
            # the camera's POSITION stays out (an orbit re-plans nothing),
            # but its TYPE gates the backface override, so it is in
            str(getattr(getattr(scene, 'camera', None), 'type', 'PERSP')),
            # coded shaders may bake the frame size (vScreenUV/iResolution)
            (int(job.width), int(job.height)),
            tuple(_light_sig(l) for l in scene.lights),
            # R169: WHICH lights the max_lights cap selects (and their
            # order) is structure -- selection ranks by energy, and the
            # values left the per-light signatures, so the chosen SET
            # must be fingerprinted on its own or an energy tweak that
            # flips the cap's choice would reuse a plan indexed for
            # different lamps
            _selection_sig(scene, st),
            tuple(_mat_sig(m) for m in scene.materials),
            tuple(sorted(job.textures or {})))


def _selection_sig(scene, st):
    from ..core import lights as LI
    try:
        sel = LI.select_lights(scene.lights, st)
        ids = {id(l) for l in sel}
        return tuple(i for i, l in enumerate(scene.lights)
                     if id(l) in ids)
    except Exception:                                           # noqa: BLE001
        return tuple(range(len(scene.lights or ())))


#: the last few plans, keyed by scene signature. A plan re-probes materials
#: through the CPU's closure path and assembles ~500-line sources; a held
#: still scene was paying that every frame, and it measured 4.4 ms of a
#: 14.3 ms warm frame at 480x360.
#: R174: material-value texels on/off (the test suite A/Bs this; there is
#: no user option -- the values are the same float32 bits either way)
MATERIAL_TEXELS = True


def _sampler_limit():
    """The driver's fragment-sampler limit, through the device layer."""
    from . import device
    return device.max_fragment_samplers()


def _pool_tag(tag, src, spec):
    """One shader NAME per distinct (source, interface) -- the pool.

    device.compile_dynamic caches on (name, hash(source)); with material
    values lifted into texels, same-structure materials produce
    byte-identical sources, and giving them the same NAME makes the
    whole pool one compile. The interface is hashed in defensively --
    equal sources imply equal declared uniforms, but a hash is cheaper
    than the proof staying true forever.

    R183: the digest is a STABLE hash, not Python's session-salted
    hash(). The salted name meant the same shader arrived under a NEW
    name every Blender session -- and the field paid the driver's full
    compile (~20 s) and pipeline build (~14 s) twice running for
    byte-identical sources. Whatever a driver or Blender keys its disk
    caches on, a name that never changes can only help -- and it makes
    console lines comparable across sessions."""
    import hashlib
    h = hashlib.blake2b(digest_size=8)
    h.update(src.encode('utf-8', 'replace'))
    for part in tuple(spec.get('samplers', ())) + ('|',) + \
            tuple(spec.get('floats', ())):
        h.update(str(part).encode('utf-8', 'replace'))
    return f'{tag}_P{h.hexdigest()}'


_PLAN_CACHE = {}


def _pack_dstab(job, rows):
    """R251 (LIGHT-B2 F018): the DS shininess tables as a data texture
    -- row = material index, 128 texels of T[i] in .r (float32 integers
    0..255), the CPU's own SH.ds_shininess_table per Glossiness."""
    n = max([len(job.scene.materials)] + [int(mi) + 1 for mi, _g in rows])
    arr = np.zeros((n, 128, 4), np.float32)
    for mi, gloss in rows:
        arr[int(mi), :, 0] = _SHM.ds_shininess_table(float(gloss))
    return arr


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
    present_now = _present_materials(job.scene.mesh, gbuf)
    if use_cache:
        try:
            sig = _plan_sig(job, _mesh_key(job.scene.mesh))
        except Exception:                                       # noqa: BLE001
            sig = None
        hit = _PLAN_CACHE.get(sig) if sig is not None else None
        if hit is not None and hit[0] is not None and \
                not present_now.issubset({int(p[0]) for p in hit[0]}):
            # R248: THE UNSHADED-BLACK MATERIAL. A plan holds one pass
            # per material, and it used to hold passes only for the
            # materials on screen when it was built -- while its
            # signature, by design, holds no camera. An orbit that
            # brought a material into view for the first time hit the
            # cached plan, found no pass for it, and the pass loop left
            # its pixels at the cleared target's zero: pure black, no
            # shading, no refusal -- "in refined but not in orbit, or
            # vice versa" because drafts and refines are different
            # sizes with different plans, each built from whatever was
            # visible at ITS first frame. A plan now covers every
            # material the mesh carries (below), and a hit that still
            # lacks one on screen re-plans instead of serving it
            hit = None
        if hit is not None:
            h_passes, h_why, h_atlases = hit
            if h_atlases and 'hal_lights' in h_atlases:
                # R169: the light VALUES are per-frame data, not part
                # of the plan -- a cache hit reuses every source and
                # every other atlas, but the hal_lights texture repacks
                # from the CURRENT lamps (this is the whole point: a
                # lamp edit lands here, as a hit plus one tiny upload,
                # instead of a 25-shader recompile storm)
                from ..core import lights as _LI2
                from .material import pack_light_texels as _plt
                _arr = _plt(_LI2.select_lights(job.scene.lights,
                                               job.settings), job)
                h_atlases = dict(h_atlases)
                h_atlases['hal_lights'] = (('lights', _arr.tobytes()),
                                           lambda a=_arr: a)
                hit = (h_passes, h_why, h_atlases)
                _PLAN_CACHE[sig] = hit
            if h_atlases and 'hal_fogtab' in h_atlases:
                # R251: the fog values are per-frame data too (the view
                # row rides them): a hit repacks from THIS frame
                from ..core.fog import pack_fog_texels as _pft2
                _fa = _pft2(job)
                h_atlases = dict(h_atlases)
                h_atlases['hal_fogtab'] = (('fogtab', _fa.tobytes()),
                                           lambda a=_fa: a)
                hit = (h_passes, h_why, h_atlases)
                _PLAN_CACHE[sig] = hit
            if h_atlases and 'hal_backdrop' in h_atlases:
                # R251 LIGHT-A2 (F008): the backdrop follows the camera
                from ..core.fog import backdrop_atlas as _bda2
                h_atlases = dict(h_atlases)
                h_atlases['hal_backdrop'] = _bda2(job)
                hit = (h_passes, h_why, h_atlases)
                _PLAN_CACHE[sig] = hit
            if h_atlases and 'hal_stipple' in h_atlases and \
                    len(h_atlases['hal_stipple']) == 3:
                # R251 C015: the N64 noise map is per-frame data (the
                # frame and seed are its stamp, never in the plan
                # signature -- a re-plan per frame is the cost this
                # idiom avoids): a hit repacks it for THIS frame and
                # the stamped upload replaces the one resident texture
                _fs = (int(getattr(job.scene, 'frame', 1) or 1),
                       int(getattr(job.settings, 'seed', 0) or 0))
                if h_atlases['hal_stipple'][2] != _fs:
                    h_atlases = dict(h_atlases)
                    h_atlases['hal_stipple'] = _stipple_atlas_entry(
                        'N64_NOISE', job)
                    hit = (h_passes, h_why, h_atlases)
                    _PLAN_CACHE[sig] = hit
            if h_atlases and 'hal_celfield' in h_atlases:
                # R238: the cel field is per-frame data too (the camera
                # and the frame's own G-buffer shape it): a hit
                # recomputes it from THIS frame and re-uploads
                _cf = _cel_field_of(job, gbuf)
                if _cf is not None:
                    h_atlases = dict(h_atlases)
                    h_atlases['hal_celfield'] = _cel_atlas_entry(_cf)
                    hit = (h_passes, h_why, h_atlases)
                    _PLAN_CACHE[sig] = hit
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
    # blurry reflections no longer refuse: the sweeps run the cone --
    # every reflective spawn (primary and recursive) expands to the
    # CPU's own K jittered directions (same BLUR_SALT streams, same
    # tangent frame, same below-surface fold) and the K traced lane
    # values AVERAGE before the composite, exactly
    # _blurred_reflection's mean. reflection_blur and its sample count
    # are in the plan signature (R78).
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
    # affine texture mapping no longer refuses the SHADING: the frame
    # passes re-interpolate uv from hal_gb_idslin, the rasteriser's OWN
    # screen-linear barycentrics packed as a second ids texture -- the
    # CPU's attributes() picks bary_lin for uv exactly when
    # tex_perspective is off, and only for uv (P, N and colour stay
    # perspective-correct; ray HITS have no screen-linear bary on
    # either device and keep true barycentrics). The matrix once
    # measured the ungated version at 0.835 over 1355 px -- the warp
    # was missing; now the warp rides the CPU's own numbers. The
    # RASTER of an affine frame still runs on the CPU (craster does
    # not carry bary_lin yet), which render.py prints by name.
    affine = not getattr(st, 'tex_perspective', True)
    # R251 C119 (MAT-B): under affine mapping the uv rides a second
    # grid (hal_gb_idslin) the snap does not cover -- refuse by name
    if affine and REYES.rate_of(st) > 0.0:
        return None, REYES.REFUSE_AFFINE, {}
    # WIREFRAME and CONSTANT left the refusal list with the shadeless
    # emit: the pass writes diffuse x level (+ emission) and
    # apply_wireframe carves the wires on the readback, the CPU's own
    # separable stage -- the fog doctrine again. GOURAUD/FLAT left it
    # earlier when their rates were ported to the corner-light road.
    if str(getattr(st, 'shading_rate', 'PIXEL')) not in ('PIXEL', 'VERTEX',
                                                         'FACE'):
        return None, f'the scene shading rate is {st.shading_rate}', {}
    if str(getattr(st, 'normal_source', 'AUTO')) == 'FACE' and \
            getattr(scene.mesh, 'face_normals', None) is None:
        # FACE left the refusal list with the hal_triaux texture: the
        # STORED face normals ride a per-tri texel (the CPU's own
        # normalized values, same bits), fetched AFTER the graph ran
        # against the interpolated normal -- the CPU's exact order.
        # Only a mesh with no stored face normals still refuses: there
        # ctx.Ng falls back to the interpolated normal per FRAGMENT,
        # which no per-tri texel can carry.
        return None, 'Normal Source FACE needs stored face normals ' \
                     'this mesh does not carry', {}

    lights = LI.select_lights(scene.lights, st)
    for l in lights:
        if l.type not in ('SUN', 'HEMI', 'POINT', 'SPOT', 'AREA'):
            return None, f'{l.type} lights are not in the GLSL light loop', {}
        if len(getattr(l, 'exclude_objects', ()) or ()) > 64:
            # linking emits a per-light object ladder against td.y (an
            # exact integer float) -- the same isin() decision the CPU
            # takes, no texture, no cliff. A pathological list is the
            # one honest refusal left.
            return None, ('light linking with more than 64 linked '
                          'objects shades on the CPU'), {}

    # shadow maps: the same maps the CPU just baked, packed for the GPU
    shadows, atlases = [], {}
    shadow_parts = []
    for i, l in enumerate(lights):
        meta, atlas, why = _shadow_meta(l, st, getattr(job, 'bvh', None))
        if why is not None:
            return None, f"light '{getattr(l, 'name', i)}': {why}", {}
        shadows.append(meta)
        if atlas is not None:
            shadow_parts.append((i, atlas))
    if shadow_parts:
        # ONE combined atlas for every mapped light, stacked by rows.
        # Eight per-light samplers put the field's heavier materials AT
        # the 16-sampler fragment cliff (12 material samplers + the
        # G-buffer's three), and the real driver rejected the compile:
        # 'the driver rejected <material>: HAL_MAT_1: CreateInfo failed' --
        # the refusal milestone that ended five rounds of guessing.
        # Every light's cell arithmetic is already baked as literals,
        # so packing costs one added row offset per light and frees
        # seven sampler slots on an eight-light scene.
        voff = 0
        keys, packs = [], []
        for i, (akey, abuild) in shadow_parts:
            m = shadows[i]
            ph = int(m['size']) * int(m['grid'][1])
            pw = int(m['size']) * int(m['grid'][0])
            m['voff'] = voff
            keys.append(akey)
            packs.append((voff, ph, pw, abuild))
            voff += ph
        total_h = voff
        total_w = max(pw for _v, _h, pw, _b in packs)

        def build_pack(_packs=tuple(packs), _th=total_h, _tw=total_w):
            out = np.zeros((_th, _tw, 4), np.float32)
            for v, ph, pw, b in _packs:
                px = b()
                out[v:v + ph, :pw] = px[:, :pw]
            return out

        atlases['hal_shadowpack'] = (('shadowpack',) + tuple(keys),
                                     build_pack)

    # projected light textures (gobos): the image rides along per light,
    # in the same cached-upload idiom, and the light loop samples it with
    # the CPU's own bilinear texel arithmetic written out in GLSL
    cookies = {}
    for i, l in enumerate(lights):
        ck = getattr(l, 'cookie', None)
        if ck is None:
            continue
        # R219: the PREPARED pixels -- source-size blur already baked in
        # by the shared road the CPU sampler reads, so both devices
        # sample identical texels; and every lamp kind projects now
        px = LI.cookie_pixels(l)
        if px is None:
            continue
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
        try:
            a_sx, a_sy = (float(v) for v in
                          getattr(l, 'area_size', (1.0, 1.0))[:2])
        except Exception:                                       # noqa: BLE001
            a_sx = a_sy = 1.0
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
            'extend': LI.cookie_extend(l),
            'filter': LI.cookie_filter(l),
            'sx': max(a_sx, 1e-6), 'sy': max(a_sy, 1e-6),
        }

    # R221: the anime Shadow Ramps -- bake (idempotent, cache-keyed) and
    # pack every ramped material's LUT into one atlas, offsets by scene
    # material index order. The lamp loop samples the SAME texels the
    # CPU's _anime_ramp_sample reads, so the devices cannot disagree.
    anime_ramps = {}
    ramp_luts = []
    from ..core.nodeeval import bake_anime_ramp
    for mi_, m in enumerate(getattr(job.scene, 'materials', ()) or ()):
        g = getattr(m, 'graph', None)
        if not g or 'HALCYON_AnimeShaderNode' not in str(
                g.get('nodes', {})):
            continue
        try:
            spec = bake_anime_ramp(g, job.textures, job.settings)
        except Exception:                                       # noqa: BLE001
            spec = None
        if spec is None:
            continue
        lut = spec['lut']
        anime_ramps[mi_] = {'v0': sum(l.shape[0] for l in ramp_luts),
                            'w': int(lut.shape[1]),
                            'h': int(lut.shape[0])}
        ramp_luts.append(lut)
    if ramp_luts:
        rkey = ('animeramp', len(ramp_luts),
                tuple(round(float(l[::5, ::7].sum()), 4)
                      for l in ramp_luts))

        def build_ramps(_ls=tuple(ramp_luts)):
            stack = np.concatenate(_ls, axis=0)
            return np.concatenate(
                [stack, np.ones(stack.shape[:2] + (1,), np.float32)],
                axis=2)

        atlases['hal_animeramp'] = (rkey, build_ramps)

    # R239: the SDF face maps -- bake (idempotent, cache-keyed) and
    # pack every face material's LUT into one atlas; the face's frame
    # (from the first object wearing the material) bakes as literals,
    # exactly the values render.face_frame hands the CPU lamp loop
    anime_faces = {}
    face_luts = []
    from ..core.nodeeval import bake_face_sdf, face_frame
    for mi_, m in enumerate(getattr(job.scene, 'materials', ()) or ()):
        g = getattr(m, 'graph', None)
        if not g or 'HALCYON_AnimeShaderNode' not in str(
                g.get('nodes', {})):
            continue
        try:
            fsp = bake_face_sdf(g, job.textures, job.settings)
        except Exception:                                       # noqa: BLE001
            fsp = None
        if fsp is None:
            continue
        nd_f = next((nd for nd in g.get('nodes', {}).values()
                     if nd.get('bl_idname') == 'HALCYON_AnimeShaderNode'),
                    {})
        p_f = nd_f.get('props', {})
        fr = face_frame(job.scene, mi_,
                        str(p_f.get('face_forward', 'NEG_Y')),
                        str(p_f.get('face_up', 'POS_Z')))
        if fr is None:
            continue
        lut_f = fsp['lut']
        anime_faces[mi_] = {
            'v0': sum(l.shape[0] for l in face_luts),
            'w': int(lut_f.shape[1]), 'h': int(lut_f.shape[0]),
            'fwd': tuple(float(v) for v in fr[0]),
            'up': tuple(float(v) for v in fr[1]),
            'right': tuple(float(v) for v in fr[2])}
        face_luts.append(lut_f)
    if face_luts:
        fkey = ('facesdf', len(face_luts),
                tuple(round(float(l[::5, ::7].sum()), 4)
                      for l in face_luts),
                tuple(sorted((k, sp['fwd'], sp['up'])
                             for k, sp in anime_faces.items())))

        def build_faces(_ls=tuple(face_luts)):
            stack = np.concatenate(_ls, axis=0)
            img = np.zeros(stack.shape + (4,), np.float32)
            img[..., 0] = stack
            img[..., 3] = 1.0
            return img

        atlases['hal_facesdf'] = (fkey, build_faces)

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

    # R238: the cel field (the cel materials' screen shadow and depth
    # rim), computed on the CPU from this frame's G-buffer -- the
    # material passes read it per pixel as hal_celfield. The scene's
    # key lamp (lines.key_light: the first non-ambient) carries the
    # screen shadow under Scene Lamps
    _cel_field = _cel_field_of(job, gbuf)
    if _cel_field is not None:
        atlases['hal_celfield'] = _cel_atlas_entry(_cel_field)
    from ..core.lines import key_light as _key_light
    _klight = _key_light(scene)
    _cel_key_index = next((i for i, l in enumerate(lights)
                           if l is _klight), -1)

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
    _amb_eng, _amb_wrld = LI.ambient_light_split(scene, st)
    consts = {
        'ambient_color': tuple(float(v) for v in LI.ambient_light(scene, st)),
        # the split the BI material node's flat-ambient rule needs:
        # engine Global Ambient stays diffuse-tinted, only the WORLD
        # part flat-adds (see light_surface's BI branch)
        'ambient_engine': tuple(float(v) for v in _amb_eng),
        'ambient_world': tuple(float(v) for v in _amb_wrld),
        'two_sided': bool(getattr(st, 'two_sided_lighting', True)),
        'specular_in_gamma': bool(getattr(st, 'specular_in_gamma', True)),
        'clamp_specular': bool(getattr(st, 'clamp_specular', True)),
        'light_clamp': float(getattr(st, 'light_clamp', 0.0)),
        'falloff_default': str(getattr(st, 'light_falloff_default',
                                       'INVERSE_SQUARE')),
        # lamp names claimed by materials' EXCLUSIVE light groups,
        # collected by render.collect_exclusive_lights at render entry
        'exclusive_lights': tuple(sorted(
            getattr(scene, 'exclusive_lights', None) or ())),
        'shadow_samples': int(getattr(st, 'shadow_samples', 4)),
        # the per-lamp shadow MODE resolution the CPU runs (smode =
        # light.shadow under PER_LIGHT, the override otherwise): the
        # RAYBIAS terminator branch fires only for smode == 'RAY'
        # lamps, exactly light_surface's gate. Already in the plan
        # signature's settings fingerprint.
        'shadow_default': str(getattr(st, 'shadow_default', 'PER_LIGHT')),
        'tex_filter': str(getattr(st, 'tex_filter', 'NEAREST')),
        'tex_aniso': int(getattr(st, 'tex_aniso', 1) or 1),
        'tex_mip_bias': float(getattr(st, 'tex_mip_bias', 0.0) or 0.0),
        # R251 texture pack: the sample-time dials material._tex_opts
        # hands to core.texture.sample_opts (one function, both devices)
        'tex_mipmap': bool(getattr(st, 'tex_mipmap', False)),
        'tex_frac_bits': str(getattr(st, 'tex_frac_bits', 'FLOAT')),
        'tex_clamp_mode': str(getattr(st, 'tex_clamp_mode', 'EDGE')),
        'tex_colorkey': bool(getattr(st, 'tex_colorkey', False)),
        'tex_colorkey_range': int(getattr(st, 'tex_colorkey_range', 0)),
        'tex_mip_select': str(getattr(st, 'tex_mip_select', 'FILTER')),
        'tex_lod_sharpen': bool(getattr(st, 'tex_lod_sharpen', False)),
        'tex_lod_source': str(getattr(st, 'tex_lod_source', 'DERIVATIVE')),
        'tex_lod_k': float(getattr(st, 'tex_lod_k', 0.0)),
        'tex_lod_l': int(getattr(st, 'tex_lod_l', 0)),
        # per-object bounds for Generated coordinates: derived from the mesh,
        # which the plan signature already fingerprints
        'obj_bounds': job.object_bounds(),
        # R243: the per-object inverse matrices, for Object coordinates
        # (the Texture Coordinate node's Object output, Max's Object XYZ)
        'obj_inv': job.object_matrices(),
        # R253: the OBJECT-space box Generated coordinates span (the
        # mesh's texture space) and which road the bake takes; the
        # setting is in _plan_sig (a bake the plan reads MUST be)
        'obj_gen': job.object_generated_frame(),
        'generated_space': str(getattr(st, 'generated_space', 'OBJECT')),
        # the frame size, for coded shaders reading vScreenUV/iResolution
        'resolution': (float(job.width), float(job.height)),
        # R238: the cel field rides this frame; the scene's key lamp
        'cel_field': _cel_field is not None,
        'cel_key_index': int(_cel_key_index),
        # the world->camera transform, baked into SSS passes: the
        # scatter tree lives in camera space, where shi->co lived
        'view_rows': tuple(
            float(v) for v in
            np.asarray(job.view, np.float32)[:3, :4].ravel()),
        # BI's world Exposure (wrld_exposure_correct): active when
        # exp != 0 or range != 1; the assembler bakes linfac/logfac
        'world_exposure': (
            (float(getattr(scene.world, 'exposure', 0.0) or 0.0),
             float(getattr(scene.world, 'exposure_range', 1.0) or 1.0))
            if getattr(scene, 'world', None) is not None
            and (float(getattr(scene.world, 'exposure', 0.0) or 0.0)
                 != 0.0
                 or float(getattr(scene.world, 'exposure_range', 1.0)
                          or 1.0) != 1.0)
            else None),
        # the camera TYPE (already in the plan signature): the emitters'
        # perspective-only answers -- Geometry's Backfacing plane test --
        # gate on it, exactly as the backface override does
        'camera': str(getattr(scene.camera, 'type', 'PERSP')).upper(),
        # Normal Source FACE: every pass overrides the shading normal
        # with the STORED face normal from the hal_triaux texture --
        # the CPU's own normalize(mesh.face_normals), fetched per tri,
        # AFTER the graph ran against the interpolated one (the CPU's
        # exact order). normal_source is in the plan signature.
        'normal_face':
            str(getattr(st, 'normal_source', 'AUTO')).upper() == 'FACE',
        # affine texture mode: frame passes re-interpolate uv from the
        # hal_gb_idslin texture (the rasteriser's own screen-linear
        # barycentrics). tex_perspective is in the plan signature.
        'affine': affine,
        # Screen Door stipple: the frame pass emits the CPU's own alpha
        # chain and keeps-or-drops against the hal_stipple threshold
        # texture (the CPU's threshold_map, uploaded verbatim).
        # transparency and stipple_pattern are in the plan signature.
        'stipple': ({'pattern': str(getattr(st, 'stipple_pattern', 'NONE'))
                     if str(getattr(st, 'stipple_pattern', 'NONE'))
                     != 'NONE' else 'BAYER4'}
                    if str(getattr(st, 'transparency', 'NONE')) == 'STIPPLE'
                    else None),
        # light linking: per-light object ladders against td.y --
        # light_surface's isin() mask, unrolled. In _light_sig.
        'light_links': {
            i: {'objects': tuple(sorted(l.exclude_objects)),
                'mode': str(getattr(l, 'exclude_mode', 'EXCLUDE'))}
            for i, l in enumerate(lights)
            if getattr(l, 'exclude_objects', None)},
        # for vertex-rate passes: the corner-light texture indexes by
        # original triangle id, so its side bakes from the mesh's count
        # (the plan signature fingerprints the mesh already)
        'tri_count': int(scene.mesh.tris.shape[0])
        if getattr(scene.mesh, 'tris', None) is not None else 0,
        # the mesh's named UV layers, for the UV Map node's resolution
        'uv_names': tuple(getattr(scene.mesh, 'uv_names', None) or ()),
        'color_name': str(getattr(scene.mesh, 'color_name', None) or ''),
        'has_vcol': getattr(scene.mesh, 'colors', None) is not None,
        # the BVH texture sides, baked into the traversal source when any
        # light shadows by ray; None otherwise
        'bvh_sides': bvh_sides,
        # deterministic sampling: the seed every hash stream mixes in, and
        # the AO spec when the frame occludes ambient light
        'seed': int(getattr(st, 'seed', 0) or 0),
        # R251 LIGHT-B1 (F011): PIXEL | AXIS -- the fixed camera-axis
        # viewer reads hal_fogtab texel 227 (section 0's map)
        'specular_viewer': str(getattr(st, 'specular_viewer', 'PIXEL')),
        # R251 (LIGHT-B2 F020): crand folds the frame number into its
        # hash (structure: the pass declares hal_frame; in st_sig)
        'crand_per_frame': bool(getattr(st, 'crand_per_frame', False)),
        # R251 (MAT-A C049): the output-pixel pitch of the Super FX plot
        'plot_ss': _plot_ss(st),
        # R251 (MAT-A C049): the output-pixel pitch of the Super FX plot
        'plot_ss': _plot_ss(st),
        # R251 C119 (MAT-B): the REYES snap's padded-square side, None
        # at rate 0 (the snap function is emitted only when set)
        'reyes': REYES.consts_for(st, scene),
        # the transparent-layer alpha chain's hard cutoff
        'alpha_threshold': float(getattr(st, 'alpha_threshold', 0.0)),
        # R251 (lighting): fog STRUCTURE for hal_fog (core/fog.structure,
        # plus the camera type for the z-fog branch) and the ONE rule
        # for the hal_fogtab sampler (lighting.md section 0, A9)
        'fog': _fog_structure(st, scene),
        'fogtab': _fogtab_needed(st, scene),
        # R174: material VALUES ride the hal_mats texture (assemble_frame
        # marks them, lifts them, and hands the row back in
        # info['mat_values']). Same-structure materials share one
        # compiled shader; a slider drag re-uploads a texel row. The
        # module flag exists for the A/B in the test suite.
        '__mark_values': MATERIAL_TEXELS,
        # R181: the pass guard checks against the DRIVER'S sampler
        # limit, not the spec floor (a session constant; 16 headless)
        'max_samplers': _sampler_limit(),
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
        # R221: packed anime Shadow Ramp offsets, keyed by material index
        'anime_ramps': anime_ramps,
        'anime_faces': anime_faces,
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
        if (secondary or layer) and consts.get('reyes'):
            # R251 C119 (MAT-B): the on-screen loop reuses the primary
            # probe's bake, so the probe's refusal never ran -- gate here
            return None, f"'{_mat_name(mi)}': {REYES.REFUSE_PASS}"
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
        # R251 (lighting, B17): fog stays on the CPU, BY NAME, for the
        # materials whose composites land AFTER the readback -- traced
        # reflections / refractions and a CPU-evaluated env term (the
        # CPU fogs base + r*hit; in-shader fog would give fog(base) +
        # r*hit) -- and while the module switch is off. Decided HERE,
        # before the source exists; the tail reads consts['fog_cpu'],
        # assemble_frame echoes it as info['fog_cpu'], _fog_readback
        # fogs exactly those ids. Printed once per plan (a cache hit
        # never re-enters this function). A material with Use Mist off
        # is fogged by neither road and carries no reason.
        fog_cpu = None
        if bool(getattr(st, 'fog', False)) and vrate == 'PIXEL' \
                and not secondary and not layer \
                and float(bake.get('use_mist', 1.0)) >= 0.5:
            if not FOG_ON_GPU:
                fog_cpu = 'the module switch is off'
            elif ((ray_on and job.bvh is not None
                   and ((getattr(st, 'ray_reflection', True)
                         and float(bake.get('reflect', 0.0)) > 1e-4)
                        or (getattr(st, 'ray_refraction', True)
                            and float(bake.get('opacity', 1.0))
                            < 0.999)))
                  or mi in cpu_env['primary']):
                fog_cpu = ('traced/env composites land after the '
                           'readback; fog stays on the CPU')
            if fog_cpu:
                print(f"[Halcyon GPU] fog on the CPU for "
                      f"'{_mat_name(mi)}': {fog_cpu}")
        c = dict(consts)
        c['env'] = env_spec
        c['fog_cpu'] = fog_cpu
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
            if abs(float(bake.get('bi_transp_fresnel', 0.0))) > 0.0:
                # fresnel alpha makes the refraction lerp view-dependent
                return f"'{_mat_name(mi)}' refracts through a Fresnel " \
                       'alpha on the CPU'
        if float(bake.get('reflect', 0.0)) > 1e-4:
            if 'specular' in per_pixel_fields(graph):
                return f"'{_mat_name(mi)}' scales its reflection by a " \
                       'per-pixel Specular Color on the CPU'
            if abs(float(bake.get('bi_mir_fresnel', 0.0))) > 0.0 and \
                    ray_depth >= 2:
                # the composite applies the view factor at the PRIMARY
                # surface; a bounce chain needs it at every hit
                return f"'{_mat_name(mi)}' mirrors through Fresnel at " \
                       f'ray depth {int(ray_depth)} on the CPU (the ' \
                       'view factor composes per bounce)'
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
            if abs(float(bake.get('bi_mir_fresnel', 0.0))) > 0.0:
                # Mirror > Fresnel: view-dependent, applied per pixel at
                # the composite (the CPU applies it to surf.reflect
                # before _add_raytraced -- same maths, same place in
                # the chain)
                mir_fres[mi] = (float(bake['bi_mir_fresnel']),
                                float(bake.get('bi_mir_blend', 1.25)))
        if getattr(st, 'ray_refraction', True) and \
                float(bake.get('opacity', 1.0)) < 0.999:
            op = min(max(float(bake['opacity']), 0.0), 1.0)
            # BI's Filter: 0 passes light untinted, 1 tints by the base
            # colour -- exactly _add_raytraced's lerp toward white
            f = min(max(float(bake.get('bi_ray_filter', 1.0)), 0.0), 1.0)
            dif = tuple(1.0 + f * (float(v) - 1.0)
                        for v in bake['diffuse_flat'])
            refractive[mi] = {
                'k': (1.0 - op)
                * min(max(float(bake.get('refraction', 1.0)), 0.0), 1.0),
                'diffuse': dif,
                'ior': max(float(bake.get('ray_ior',
                                          bake.get('ior', 1.45))), 1e-3),
            }

    passes = []
    secondary = []
    secondary_mid = []
    reflective = {}
    refractive = {}
    mir_fres = {}
    cpu_env = {'primary': {}, 'hit': {}}
    any_screen = False
    rng = np.random.default_rng(19)
    py, px = np.nonzero(covered)
    # R248: a pass for EVERY material the mesh carries, not only those
    # on screen this frame -- the plan's signature holds no camera, so
    # a plan must serve any view. A material on screen is probed on its
    # own fragments exactly as before (a refusal there refuses the
    # frame, by name, as before); a material off screen is probed over
    # its own triangles -- the road the ray plan already walks for
    # materials 'visible only in reflections' -- and one that cannot be
    # probed there is left UNPLANNED, named: it never costs the frame
    # its GPU, and the first frame that shows it re-plans with it on
    # screen (the cache-hit rule above). The draw side compiles and
    # draws only the passes on screen, so an off-screen pass costs the
    # plan a probe and a source, never a driver compile.
    m_of = scene.mesh.mat_index[gbuf.tri[py, px]] \
        if scene.mesh.mat_index is not None else np.zeros(py.size, np.int32)
    all_ids = np.unique(scene.mesh.mat_index) \
        if scene.mesh.mat_index is not None else np.zeros(1, np.int32)
    unplanned = {}
    for mi in all_ids:
        mi = int(mi)
        on_screen = mi in present_now
        mine = np.nonzero(m_of == mi)[0] if on_screen else np.zeros(0, np.int64)
        if on_screen and mine.size == 0:
            continue
        if on_screen:
            pick = mine if mine.size <= PROBE_FRAGMENTS else \
                mine[rng.choice(mine.size, PROBE_FRAGMENTS, replace=False)]
            bake, model_idx, why = _probe_material(job, gbuf, mi,
                                                   py[pick], px[pick])
        else:
            tri_pool = np.nonzero(scene.mesh.mat_index == mi)[0]
            if tri_pool.size == 0:
                continue
            pick_t = tri_pool if tri_pool.size <= PROBE_FRAGMENTS else \
                tri_pool[rng.choice(tri_pool.size, PROBE_FRAGMENTS,
                                    replace=False)]
            frag_bary = np.full((pick_t.size, 3), 1.0 / 3.0, np.float32)
            try:
                bake, model_idx, why = _probe_material(
                    job, gbuf, mi, None, None,
                    frags=(pick_t.astype(np.int32), frag_bary))
            except Exception as exc:                            # noqa: BLE001
                bake, model_idx = None, None
                why = f'probing it off-screen failed ({exc})'
        refuse = None
        if bake is None:
            refuse = f"'{_mat_name(mi)}': {why}"
        # the BI panel round's honest CPU-only flags
        elif float(bake.get('cast_only', 0.0)) > 0.5:
            refuse = f"'{_mat_name(mi)}' is Cast Only: peeling the " \
                     'camera surface shades on the CPU'
        elif float(bake.get('shadows_only', 0.0)) > 0.5:
            refuse = f"'{_mat_name(mi)}' is Shadows Only: the " \
                     'shadow catcher shades on the CPU'
        # R251: 'opts out of mist' no longer refuses -- the fog block
        # honours the bake (a Use-Mist-off pass emits no hal_fog call)
        elif float(bake.get('backface_mix', 0.0)) > 1e-4 and \
                str(getattr(scene.camera, 'type', 'PERSP')).upper() != \
                'PERSP':
            # the shader decides backfacing with a plane-side test against
            # the eye, which is the rasteriser's answer only in perspective
            refuse = f"'{_mat_name(mi)}': the backface override " \
                     'under an orthographic camera is not in the ' \
                     'deferred pass yet'
        mat = scene.materials[mi] if mi < len(scene.materials) else None
        graph = getattr(mat, 'graph', None) if mat is not None else None
        if refuse is None and ray_on:
            refuse = _ray_gate(mi, bake, graph)
        entry = entry2 = entry3 = None
        if refuse is None:
            entry, why = _one_material(mi, bake, model_idx)
            if entry is None:
                refuse = why
        if refuse is None and ray_on:
            entry2, why2 = _one_material(mi, bake, model_idx,
                                         secondary=True)
            if entry2 is None:
                refuse = why2
            elif ray_depth >= 2:
                entry3, why3 = _one_material(mi, bake, model_idx,
                                             secondary=True, mid=True)
                if entry3 is None:
                    refuse = why3
        if refuse is not None:
            if on_screen:
                return None, refuse, {}
            # off screen: the frame keeps its GPU; the material is
            # named, and re-probed on its own fragments the first
            # frame it appears
            unplanned[mi] = str(refuse)
            continue
        passes.append(entry)
        any_screen = any_screen or bool(entry[3].get('uses_screen'))
        # exactly _add_raytraced's k and tint, per material: the gate
        # above already held every constant this needs
        _collect_ray_terms(mi, bake)
        if ray_on:
            secondary.append(entry2)
            any_screen = any_screen or bool(entry2[3].get('uses_screen'))
            if ray_depth >= 2:
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
    if affine and str(getattr(st, 'transparency', 'NONE')) in (
            'SORTED', 'ABUFFER'):
        # the per-layer ids textures pack perspective barycentrics; an
        # affine layer would need its own screen-linear set per rank.
        # Until that carry exists the transparent stage shades on the
        # CPU, named -- the opaque frame keeps its GPU passes.
        lwhy = ('affine texture mapping: the layer ids textures carry '
                'no screen-linear barycentrics yet')
    elif str(getattr(st, 'transparency', 'NONE')) in ('SORTED', 'ABUFFER'):
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
            if m_l is not None:
                # R251: the ONE predicate `_split_by_alpha` uses
                # (core/scene.py `material_see_through`: opacity, the
                # export's alpha evidence, a per-material Blend Mode, the
                # PS2 Clip+Blend; an Env hole is opaque), so the layer
                # plan and the split can never disagree
                from ..core.scene import material_see_through
                if material_see_through(m_l) is None:
                    continue
            if m_l is not None and \
                    str(getattr(m_l, 'alpha_mode', 'BLEND')) == 'CLIP_BLEND':
                # R251 C031 (PS2 AFAIL): the opaque half is decided on
                # the CPU by _promote_clip, and the composite drops a
                # layer fragment whose alpha passes the threshold. A
                # layer pass would decide `hal_alpha >= threshold` per
                # fragment in the driver's own chain arithmetic (held to
                # 6e-3, not 0): within an ulp of the threshold one device
                # promotes where the other still blends -- the keep/drop
                # cliff class. A baked CONSTANT passes the threshold
                # identically on both devices and keeps its layer pass.
                from .material import per_pixel_fields as _ppf
                if getattr(m_l, 'has_alpha', False) or \
                        'opacity' in _ppf(getattr(m_l, 'graph', None)):
                    lwhy = (f"'{_mat_name(mi)}' is Clip+Blend with a "
                            'per-pixel alpha: the blend half\'s '
                            'keep/drop against the Clip Threshold is '
                            'a cliff between devices; the layers '
                            'shade on the CPU, by name')
                    break
            if m_l is not None:
                # R211/R213 punch-through: a material on the clip road
                # (Alpha Mode Clip, or a Blend chain that provably
                # yields only 0/1) never reaches the A-buffer -- its
                # pixels resolve in the z-pass and shade with the
                # OPAQUE frame -- so a layer pass for it would compile
                # for nothing. One that could NOT lift its alpha out
                # (refused, by name, in the render log) still blends:
                # keep its layer pass.
                from ..core.scene import clip_road
                if clip_road(m_l)[0] is not None and \
                        str(getattr(m_l, 'alpha_mode', 'BLEND')) != 'CLIP_BLEND':
                    # (R251 C031: a Clip+Blend material's layer pass IS
                    # its blend half -- never skipped)
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
        # R167: a Ray Bias material's hit pass is a STUB (the
        # correction has no triangle id on hits). Stubs may EXIST --
        # the planner builds hit passes whenever ray tracing is on --
        # but the moment something reflective or refractive would
        # actually run one, the whole plan refuses by name, exactly
        # the pre-twin behaviour for mirrored scenes.
        for p in list(secondary) + list(secondary_mid):
            if (p[3] or {}).get('raybias_secondary_stub'):
                return None, (f"'{p[1]}' (as a reflection): Ray Bias "
                              'terminator correction has no triangle '
                              'id on ray hits; the material shades on '
                              'the CPU'), {}
        rplan = {'secondary': secondary,
                 'secondary_mid': secondary_mid,
                 'depth': int(ray_depth),
                 'scale': {mi: tuple(float(x) for x in sc)
                           for mi, sc in reflective.items()},
                 'reflective': sorted(reflective),
                 'refract': refractive,
                 'refractive': sorted(refractive),
                 'mir_fres': dict(mir_fres),
                 'bias': float(getattr(st, 'ray_bias', 1e-3))}
        atlases['__reflect'] = rplan

    if cpu_env['primary'] or cpu_env['hit']:
        atlases['__env'] = cpu_env

    # the per-triangle auxiliary texture: STORED face normals (the CPU's
    # own normalize(mesh.face_normals), same bits) and the per-tri
    # random (_hash1 of the tri index, the CPU's own sin-fract values
    # baked rather than recomputed by a driver that decorrelates).
    # Packed once, served by name to Normal Source FACE, the Geometry
    # node's True Normal, and Random Per Island.
    def _pass_binds():
        for _mid, _nm, _src, b in passes:
            yield b
        for _mid, _nm, _src, b in (atlases.get('__layers') or ()):
            yield b
        rp = atlases.get('__reflect') or {}
        for lst in (rp.get('secondary') or (), rp.get('secondary_mid')
                    or ()):
            for _mid, _nm, _src, b in lst:
                yield b
    if any('hal_triaux' in (b.get('samplers') or ())
           for b in _pass_binds()):
        from . import gbuffer as GB
        mesh = scene.mesh
        if getattr(mesh, 'face_normals', None) is None:
            return None, 'the per-tri texture needs stored face ' \
                         'normals this mesh does not carry', {}
        akey = ('triaux',) + _mesh_key(mesh)
        atlases['hal_triaux'] = (akey,
                                 lambda m=mesh: GB.pack_tri_aux(m)[0])

    if any('hal_sres' in (b.get('samplers') or ())
           for b in _pass_binds()):
        # R167: each triangle's object Auto Smooth threshold, for the
        # RAYBIAS terminator twin -- per-tri because the object count
        # is unbounded (the field file carries 109), where a uniform
        # ladder like obcolor's would cap out. Values join the cache
        # key: Auto Smooth edits must repack without a mesh edit.
        from . import gbuffer as GB2
        mesh2 = scene.mesh
        sres_obj = np.array(
            [float(getattr(o, 'smoothresh', 0.0) or 0.0)
             for o in (getattr(scene, 'objects', None) or ())],
            np.float32)
        if sres_obj.size == 0:
            sres_obj = np.zeros(1, np.float32)
        oidx = np.clip(np.asarray(
            getattr(mesh2, 'obj_index', None)
            if getattr(mesh2, 'obj_index', None) is not None
            else np.zeros(mesh2.tris.shape[0], np.int64),
            np.int64), 0, sres_obj.size - 1)
        per_tri = sres_obj[oidx]
        akey = ('sres', per_tri.tobytes()) + _mesh_key(scene.mesh)
        atlases['hal_sres'] = (akey,
                               lambda v=per_tri: GB2.pack_tri_scalar(v))

    if any('hal_lights' in (b.get('samplers') or ())
           for b in _pass_binds()):
        # R169: the per-light VALUE texture. The cache key carries the
        # packed bytes, so a lamp edit re-uploads this one tiny texture
        # -- and nothing else: the pass sources no longer contain the
        # values, so the plan above was a cache HIT.
        from .material import pack_light_texels
        _lt_arr = pack_light_texels(lights, job)
        atlases['hal_lights'] = (('lights', _lt_arr.tobytes()),
                                 lambda a=_lt_arr: a)
    if consts.get('fogtab'):
        # R251: the fog VALUE texture (core/fog.pack_fog_texels): like
        # hal_lights, per-frame data keyed by its bytes and repacked on
        # a plan-cache hit (a slider drag re-uploads 8 KB, never recompiles)
        from ..core.fog import pack_fog_texels as _pft
        _ft_arr = _pft(job)
        atlases['hal_fogtab'] = (('fogtab', _ft_arr.tobytes()),
                                 lambda a=_ft_arr: a)
    if consts.get('fogtab') and \
            (consts.get('fog') or {}).get('source') == 'BACKDROP':
        # R251 LIGHT-A2 (F008): the CPU's own backdrop at every pixel,
        # uploaded as the fog target (keyed by a crc of its bytes, not
        # the bytes themselves); repacked on a plan-cache hit below
        from ..core.fog import backdrop_atlas as _bda
        atlases['hal_backdrop'] = _bda(job, announce=True)

    # R174: the per-MATERIAL value texture. Every pass that lifted values
    # gets a row (hal_mrow rides its binds as 'mat_row'); the walk order
    # here is the pass construction order, deterministic per plan, and
    # the same binds objects live in the cached plan -- a HIT returns
    # rows and texture together, values already proven unchanged by the
    # signature.
    _mat_rows = []
    for b in _pass_binds():
        mv = b.get('mat_values')
        if mv:
            b['mat_row'] = len(_mat_rows)
            _mat_rows.append(mv)
    if _mat_rows:
        _mw = max(len(r) for r in _mat_rows)
        _ma = np.zeros((len(_mat_rows), _mw, 4), np.float32)
        for _ri, _row in enumerate(_mat_rows):
            _ma[_ri, :len(_row), 0] = np.asarray(_row, np.float32)
        atlases['hal_mats'] = (('mats', _ma.tobytes()),
                               lambda a=_ma: a)

    _ds_rows = sorted({tuple(b['dstab']) for b in _pass_binds()
                       if b.get('dstab')})
    if _ds_rows:
        # R251 (LIGHT-B2 F018): the DS materials' shininess tables --
        # the key carries the (material, Glossiness) pairs; Glossiness
        # is in _mat_sig, so a change re-plans and repacks
        _da = _pack_dstab(job, _ds_rows)
        atlases['hal_dstab'] = (('dstab', _da.tobytes()), lambda a=_da: a)
    if any('hal_circle' in (b.get('samplers') or ())
           for b in _pass_binds()) and 'hal_circle' not in atlases:
        # R251 (LIGHT-B2 F020, B13): a crand material appends the
        # sampling primitives (SAMPLING_GLSL declares hal_circle) with
        # no soft shadow, AO or radiosity in the frame -- the table is
        # registered here, after the bakes exist
        from ..core.patterns import CIRCLE256 as _C256

        def _build_circle_r251():
            img = np.zeros((1, 256, 4), np.float32)
            img[0, :, 0] = _C256[:, 0]
            img[0, :, 1] = _C256[:, 1]
            return img
        atlases['hal_circle'] = (('circle256', 1), _build_circle_r251)
    if any('hal_recip256' in (b.get('samplers') or ())
           for b in _pass_binds()) and 'hal_recip256' not in atlases:
        # R251 C088 (TEX-2): the CPU's own float32 reciprocals 1/1..1/256
        # as a 1-row texture -- both roads multiply by the SAME table,
        # never divide (the summed-area box)
        from ..core.texture import RECIP256 as _R256

        def _build_recip():
            img = np.zeros((1, 256, 4), np.float32)
            img[0, :, 0] = _R256
            return img
        atlases['hal_recip256'] = (('recip256', 1), _build_recip)
    if any('hal_stipple' in (b.get('samplers') or ())
           for b in _pass_binds()):
        # the CPU's own 64x64 threshold map, uploaded verbatim: both
        # devices compare the SAME baked opacity against the SAME
        # threshold values, so the keep-or-drop decision is identical
        pat = (consts.get('stipple') or {}).get('pattern', 'BAYER4')
        atlases['hal_stipple'] = _stipple_atlas_entry(pat, job)

    if unplanned:
        atlases['__unplanned'] = unplanned
    if sig is not None:
        if len(_PLAN_CACHE) > 4:
            _PLAN_CACHE.clear()
        _PLAN_CACHE[sig] = (passes, None, atlases)
    return passes, None, atlases


def _stipple_atlas_entry(pat, job):
    """The hal_stipple atlas entry for pattern `pat`: the CPU's own
    threshold map, uploaded verbatim so both devices compare the SAME
    baked opacity against the SAME threshold values.

    R251 C015 (N64 RDP dither_alpha_en): for N64_NOISE the map is the
    CPU's FULL-FRAME 8-bit random (core/dither.noise_threshold_map),
    packed FOUR pixel columns per RGBA32F texel (pixel x in texel
    x >> 2, channel x & 3: 3.7 MB at 720p, the CPU array's own size)
    under a key STABLE across frames, ('stipple', 'N64_NOISE', H, W),
    with the frame and seed carried as the upload STAMP (a 3-tuple
    entry): one noise texture lives at a time, replaced through the
    graveyard when the stamp moves, and a re-plan of the same frame
    uploads nothing. The ordered kinds keep the 64x64 tile in .r and
    the 2-tuple entry; a kind with no map (an error-diffusion kind,
    '' for a stored number no item carries) bakes Bayer 4x4 by name,
    exactly shade_batch's fallback."""
    p = str(pat)
    if p == 'N64_NOISE':
        H, W = int(job.height), int(job.width)
        frame = int(getattr(job.scene, 'frame', 1) or 1)
        seed = int(getattr(job.settings, 'seed', 0) or 0)

        def _build_noise(h=H, w=W, f=frame, s=seed):
            from ..core.dither import threshold_map
            tm = np.asarray(threshold_map('N64_NOISE', h, w, frame=f,
                                          seed=s), np.float32)
            pad = (-w) % 4
            if pad:
                tm = np.pad(tm, ((0, 0), (0, pad)), mode='edge')
            return np.ascontiguousarray(tm.reshape(h, -1, 4))
        return (('stipple', 'N64_NOISE', H, W), _build_noise,
                (frame, seed))

    def _build_stipple(p=p):
        from ..core.dither import ORDERED, threshold_map
        if ORDERED.get(str(p)) is None:
            # R251: the CPU's own by-name fallback (shade_batch):
            # an error-diffusion kind, or '' for a stored number
            # no item carries, bakes Bayer 4x4 on both devices
            print(f'[Halcyon] transparency: stipple pattern '
                  f'{p!r} is not an ordered map; Bayer 4x4 '
                  'used (an error-diffusion kind, or a scene '
                  'saved before 1.90.0 with Screen Door at None, '
                  'reads back as this)')
            p = 'BAYER4'
        tm = threshold_map(p, 64, 64)
        out = np.zeros((64, 64, 4), np.float32)
        out[:, :, 0] = np.asarray(tm, np.float32)
        return out
    return (('stipple', p), _build_stipple)


def _material_name(job, mi):
    mats = getattr(job.scene, 'materials', None) or []
    return getattr(mats[mi], 'name', None) or f'material {mi}' \
        if 0 <= int(mi) < len(mats) else f'material {mi}'


def _present_materials(mesh, gbuf):
    """The material ids with at least one pixel in this frame's G-buffer."""
    try:
        covered = gbuf.tri >= 0
        if not covered.any() or getattr(mesh, 'mat_index', None) is None:
            return set()
        return set(int(m) for m in np.unique(mesh.mat_index[gbuf.tri[covered]]))
    except Exception:                                           # noqa: BLE001
        return set()


def _textures(job, gbuf):
    """The three packed G-buffer textures and their sizes."""
    from . import gbuffer as GB
    ids = GB.pack_ids(gbuf)
    attrs, side = GB.pack_attributes(job.scene.mesh, respect_smooth=True)
    tris, tside = GB.pack_tri_data(job.scene.mesh)
    return ids, attrs, side, tris, tside


def _pack_vscreen(job):
    """Per-corner screen positions for the Wireframe node's Pixel Size.

    (sx, sy, w) per triangle corner from the job's OWN sgrad cache --
    the same projection the mip footprints and wire_fields ride -- so
    the shader's world-per-pixel arithmetic runs on the CPU's exact
    numbers. CAMERA-DEPENDENT: packed per frame like the ids texture
    and the footprint field, never through the plan's atlas cache (an
    orbit must not serve stale screen positions -- the R78 lesson wears
    many costumes).
    """
    mesh = job.scene.mesh
    if getattr(job, '_sgrad_cache', None) is None:
        # the same lazy ensure wire_fields itself uses
        job.uv_screen_gradients(np.zeros(1, np.int32),
                                np.full((1, 3), 1.0 / 3.0, np.float32),
                                np.zeros((1, 2), np.float32))
    sx, sy, w = job._sgrad_cache
    tris = np.asarray(mesh.tris, np.int32)
    n = tris.shape[0] * 3
    side = int(np.ceil(np.sqrt(max(n, 1))))
    out = np.zeros((side * side, 4), np.float32)
    idx = tris.reshape(-1)
    out[:n, 0] = np.asarray(sx, np.float32)[idx]
    out[:n, 1] = np.asarray(sy, np.float32)[idx]
    out[:n, 2] = np.asarray(w, np.float32)[idx]
    return out.reshape(side, side, 4)


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
                  round(float(tx.pixels[::13].sum()), 3),
                  getattr(tx, 'prep', None))              # R251 A12.13
            out[(id(binds), sname)] = up(ik, lambda _t=tx: _t.pixels)
        # mip atlases: the CPU's OWN build_mips output, packed as a
        # vertical stack -- the driver filters the very texels the CPU
        # filters
        for sname, key in (binds.get('textures_mip') or {}).items():
            if (id(binds), sname) in out:
                continue
            tx = job_textures[key]
            mk = ('mipatlas', key, tx.width, tx.height,
                  round(float(tx.pixels[::13].sum()), 3),
                  getattr(tx, 'prep', None))              # R251 A12.13
            from .material import mip_atlas as _mip_atlas
            out[(id(binds), sname)] = up(mk, lambda _t=tx: _mip_atlas(_t)[0])
        # R251 C088 (TEX-2): the summed-area atlases, keyed like the mips
        for sname, key in (binds.get('textures_sat') or {}).items():
            if (id(binds), sname) in out:
                continue
            tx = job_textures[key]
            sk = ('satatlas', key, tx.width, tx.height,
                  round(float(tx.pixels[::13].sum()), 3),
                  getattr(tx, 'prep', None))
            from .material import sat_atlas as _sat_atlas
            out[(id(binds), sname)] = up(sk, lambda _t=tx: _sat_atlas(_t))
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
    tris = np.asarray(mesh.tris)
    tstride = max(1, tris.shape[0] // 512)
    parts = [int(v.shape[0]), int(tris.shape[0]),
             round(float(v[::stride].sum()), 3),
             # ORDER-SENSITIVE triangle fingerprint (R170): a winding
             # flip permutes each triangle's indices and a plain sum
             # never notices -- the G-buffer cache served a culled
             # frame for its mirror, and the tri/attr upload caches
             # carried the same latent hole. Weighting the three
             # corners differently makes any reorder change the key.
             int((tris[::tstride].astype(np.int64)
                  * np.array([[1, 3, 7]], np.int64)).sum())]
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
    # R251 C012: the vertex-format tag (a fine lattice can move the
    # strided sums by less than the 3-decimal rounding)
    parts.append(getattr(mesh, '_quant_tag', None))
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

    An ADAPTIVE REFINE pass evaluates only the flagged pixels and their
    one-pixel surround. The bump emitter fetches this image at exactly
    three texels per shaded pixel -- the pixel and its +x / +y
    neighbours -- and a refine pass shades only inside its mask, so
    every fetched texel lies inside the mask grown by one. Every texel
    outside it is never read; computing it was the single biggest cost
    of the field's first fast adaptive frame (a full height evaluation
    per material, per pass: 1.5 seconds times three passes of a
    14-second frame). The values at the computed texels are the full
    image's values bit for bit, so the picture cannot move.
    """
    from ..core.nodeeval import VALUE
    h, w = gbuf.tri.shape
    img = np.zeros((h, w, 4), np.float32)
    rm = getattr(job.settings, '_refine_mask', None)
    if rm is None:
        py, px, _ctx, ev = _mat_eval(job, gbuf, mat_id)
        if py.size == 0 or ev is None:
            return img
    else:
        grow = rm.copy()
        grow[1:] |= rm[:-1]
        grow[:-1] |= rm[1:]
        grow[:, 1:] |= rm[:, :-1]
        grow[:, :-1] |= rm[:, 1:]
        mesh = job.scene.mesh
        covered = gbuf.tri >= 0
        m = np.where(covered, mesh.mat_index[gbuf.tri], -1) \
            if mesh.mat_index is not None else \
            np.where(covered, 0, -1)
        py, px = np.nonzero((m == int(mat_id)) & grow)
        if py.size == 0:
            return img
        from ..core.nodeeval import GraphEvaluator
        ctx = job.context(gbuf.tri[py, px], gbuf.bary[py, px], px, py,
                          np.ones(py.size, bool), None, 0, True)
        mat0 = job.scene.materials[mat_id] \
            if mat_id < len(job.scene.materials) else None
        graph0 = getattr(mat0, 'graph', None) if mat0 is not None else None
        ev = GraphEvaluator(graph0, ctx, job.textures,
                            getattr(mat0, 'programs', None)) \
            if graph0 else None
        if ev is None:
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
        return py, px, None, None, None
    V = -M.normalize(I)
    dirs = M.reflect(-V, N).astype(np.float32)
    org = (P + N * rplan['bias']).astype(np.float32)
    return py, px, org, dirs, N.astype(np.float32)


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
    fac = _mirror_fresnel_factors(job, gbuf, rplan, py, px, m)
    out[py, px] = out[py, px] + add * scale * fac[:, None]
    return out


def _mirror_fresnel_factors(job, gbuf, rplan, py, px, mat_px):
    """Per-pixel Mirror > Fresnel multipliers for the reflect composite.

    Ones where no fresnel-mirrored material sits. The normal is the
    SHADING normal through `_ray_context` -- the same bent N the CPU
    scales surf.reflect with -- so a normal-mapped fresnel mirror
    matches by construction. Depth >= 2 plans refuse fresnel mirrors
    outright (a bounce chain would need the factor at every hit), so
    this only ever runs against the primary surface."""
    mf = rplan.get('mir_fres') or {}
    fac = np.ones(py.size, np.float32)
    if not mf:
        return fac
    from ..core import mathx as M
    from ..core.shading import bi_fresnel_fac
    for mi, (fres, blend) in mf.items():
        sel = np.nonzero(mat_px == int(mi))[0]
        if sel.size == 0:
            continue
        ctx, N = _ray_context(job, gbuf, py[sel], px[sel])
        ndv = (N * (-M.normalize(ctx.I))).sum(1)
        f = bi_fresnel_fac(-ndv, np.full(sel.size, blend, np.float32),
                           np.full(sel.size, fres, np.float32))
        fac[sel] = np.clip(f, 0.0, 1.0)
    return fac


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
                f'{bool(getattr(m, "has_alpha", False))}, blend_mode '
                f"{getattr(m, 'blend_mode', 'INHERIT')} read as opaque "
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
        tex_shadows = {sname: device.upload_cached(entry[0], entry[1],
                                                   *entry[2:])
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
           'hal_eye': tuple(float(v) for v in job.eye),
           **_cam_uniforms(job)}

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
                     'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
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
                'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
        shader, err = device.compile_dynamic(
            _pool_tag('HAL_TMAT', src, spec), src, spec)
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
            elif u == 'hal_mrow':
                extra[u] = float(binds.get('mat_row', 0))
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
                     'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
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
                         'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
                shader, err = device.compile_dynamic(
                    _pool_tag(tag2, src, spec2), src, spec2)
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
                    elif u == 'hal_mrow':
                        extra[u] = float(binds.get('mat_row', 0))
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
        try:
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
            # R175: the rank's passes and its readback in ONE crossing.
            _rank_burst = []
            for mat_id, name, shader, bind, extra, prepasses in built:
                if mat_id not in rank_mats:
                    continue
                b = dict(bind)
                b['hal_gb_ids'] = tex_ids
                for uname, _ps, _pb in prepasses:
                    b[uname] = pre_tex[(mat_id, uname)]
                _rank_burst.append(
                    (shader, {**uni, **extra} if extra else uni, b,
                     target, 'ALPHA_PREMULT', not _rank_burst, region))
            # the fused burst counts under draw_ms (the outer window);
            # read_ms keeps only reads that still cross alone
            img = device.draw_many(_rank_burst, read=target,
                                   read_region=region)
        except Exception as exc:                                # noqa: BLE001
            return _fail(f'layer {r} draw failed: {exc}')
        tm['draw_ms'] += (_time.perf_counter() - t0) * 1000.0 \
            - (tm['read_ms'] - rd0)
        if rplan is not None:
            # the recursion, from THIS layer's surface: the rank's
            # fragments spawn the same tree the opaque frame does
            from . import rtrace as RT

            def draw_secondary(plist, sec_ids, level, hit_region=None,
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
                    # R175: the sweep's passes + readback, one crossing
                    _sec_burst = []
                    for i2, (nm2, sh2, bd2, ex2) in enumerate(drawn):
                        b2 = dict(bd2)
                        b2['hal_gb_ids'] = tex_sec
                        _sec_burst.append(
                            (sh2, {**uni, **ex2} if ex2 else uni, b2,
                             t2, 'ALPHA_PREMULT', i2 == 0, _region))
                    sec_r = device.draw_many(_sec_burst, read=t2,
                                             read_region=_region)
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
        for _cn, _cv in _cam_uniforms(job).items():
            uni2[_cn] = np.tile(np.asarray(_cv, np.float32)[None, :],
                                (n, 1))
        uni2['hal_time'] = np.full(n, float(getattr(job.scene, 'time',
                                                    0.0)), np.float32)
        uni2['hal_frame'] = np.full(n, float(getattr(job.scene,
                                                     'frame', 0)),
                                    np.float32)
        uni2['hal_mrow'] = np.full(n, float(binds2.get('mat_row', 0)),
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

            def draw_secondary(plist, sec_ids, _level, hit_region=None):
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
        return py, px, None, None, None
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
    return py, px, org, T, None


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
    """(org, dirs, N): rays FROM hit surfaces -- the CPU's recursion step.

    Reflection steps OFF the surface, refraction INTO it, both by the
    raw bias, exactly `_add_raytraced` at a hit batch. N rides along
    for the blur cone's tangent frame.
    """
    from ..core import mathx as M
    P, I, N, m = _hit_surface(job, tris, bary)
    V = -M.normalize(I)
    if which == 'reflective':
        dirs = M.reflect(-V, N).astype(np.float32)
        org = (P + N * rplan['bias']).astype(np.float32)
        return org, dirs, N.astype(np.float32)
    ior = np.ones(np.asarray(tris).size, np.float32)
    for mi, spec in rplan['refract'].items():
        ior[m == mi] = spec['ior']
    eta = np.where(M.dot(N, V) < 0, ior, 1.0 / ior)
    T = M.refract(-V, N, eta)
    bad = (T * T).sum(1) < 1e-9
    dirs = np.where(bad[:, None], M.reflect(-V, N), T).astype(np.float32)
    org = (P - N * rplan['bias']).astype(np.float32)
    return org, dirs, N.astype(np.float32)


def _cone_jitter(R, N, px, py, k, seed, blur_deg):
    """Sample k of the blur cone: `_blurred_reflection`, ray for ray.

    The same deterministic streams (BLUR_SALT, the pixel identity that
    follows a ray through the whole recursion), the same tangent-disk
    construction, the same fold to the mirror direction for rays bent
    below the surface -- so the cone's rays are the CPU's own on either
    backend.
    """
    from ..core import mathx as M
    from ..core import patterns as PT
    from ..core.render import ShadeJob
    half = np.float32(np.tan(np.radians(max(blur_deg, 0.0)) * 0.5))
    t, b = M.orthonormal_basis(R)
    z = 2 * k + ShadeJob.BLUR_SALT + 7919 * seed
    sx = np.asarray(px, np.int64)
    sy = np.asarray(py, np.int64)
    u1 = PT.sample_u(sx, sy, z)
    ca, sa = PT.sample_circle(PT.sample_u(sx, sy, z + 1))
    r = np.sqrt(u1) * half
    d = M.normalize(R + t * (r * ca)[:, None] + b * (r * sa)[:, None])
    below = M.dot(d, N) <= 0.0
    return np.where(below[:, None], R, d).astype(np.float32)


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
    _child_reflect_add(job, rplan, img, py, px, add, mat_px)


def _child_reflect_add(job, rplan, img, py, px, add, mat_px):
    """The reflect composite's scale-and-accumulate tail, add precomputed.

    The blur road lands here with the AVERAGE of K jittered lane values
    -- exactly `_blurred_reflection`'s mean -- and the single-ray road
    with one; the composite is identical either way.
    """
    scale = np.zeros((py.size, 3), np.float32)
    for mi, sc in rplan['scale'].items():
        scale[mat_px == mi] = np.asarray(sc, np.float32)
    img[py, px] = img[py, px] + add * scale


def _composite_reflections_add(job, gbuf, rplan, out, py, px, add):
    """The primary reflection composite's tail, lane values precomputed."""
    mesh = job.scene.mesh
    m = mesh.mat_index[gbuf.tri[py, px]] if mesh.mat_index is not None \
        else np.zeros(py.size, np.int32)
    scale = np.zeros((py.size, 3), np.float32)
    for mi, sc in rplan['scale'].items():
        scale[m == mi] = np.asarray(sc, np.float32)
    out[py, px] = out[py, px] + add * scale
    return out


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


def _field_uv(job, gbuf, st, cov):
    """R251 (TEX-2, A12.5): (tri, bary, uv) of the covered pixels with uv
    fetched by the CPU's OWN affine rule -- attributes() R:2286 reads uv
    from the screen-linear barycentrics exactly when tex_perspective is
    off, and context() hands THAT uv to uv_screen_gradients with the
    true barycentrics. The GPU's sampler already reads the affine uv
    (hal_gb_idslin); the fields were the odd ones out."""
    from ..core import raster as _raster
    mesh = job.scene.mesh
    tri = gbuf.tri[cov]
    bary = gbuf.bary[cov]
    ub = gbuf.bary_lin[cov] if (gbuf.bary_lin is not None
                                and not getattr(st, 'tex_perspective', True)) else bary
    uv = _raster.fetch(mesh.uvs, mesh.tris, tri, ub) \
        if mesh.uvs is not None \
        else np.zeros((tri.size, 2), np.float32)
    return tri, bary, uv


def _uvgrad_field(job, gbuf):
    """(H, W, 4) float32: the CPU's analytic UV screen derivatives.

    (du/dx, du/dy, dv/dx, dv/dy) at every covered pixel, zeros elsewhere
    -- computed by ShadeJob.uv_screen_gradients, the very function the
    CPU's own trilinear reads through the context. Same numbers, one
    upload, shared by every footprint-filtered sampler in the frame.
    R251 (TEX-2): built ONCE per job and G-buffer (fill_base runs per
    pass, A12.8) and fetching uv by the CPU's affine rule (A12.5).
    """
    cache = getattr(job, '_uvgrad_fields', None)
    if cache is None:
        cache = job._uvgrad_fields = {}
    hit = cache.get(id(gbuf))
    if hit is not None and hit[0] is gbuf:
        return hit[1]
    h, w = gbuf.tri.shape
    field = np.zeros((h, w, 4), np.float32)
    cov = gbuf.tri >= 0
    if cov.any():
        tri, bary, uv = _field_uv(job, gbuf, job.settings, cov)
        du, dv = job.uv_screen_gradients(tri, bary, uv)
        field[cov, 0] = du[:, 0]
        field[cov, 1] = du[:, 1]
        field[cov, 2] = dv[:, 0]
        field[cov, 3] = dv[:, 1]
    cache[id(gbuf)] = (gbuf, field)        # the G-buffer rides along: its id cannot be reused while cached
    return field


def _lod_field(job, gbuf, st, key):
    """(H, W, 4) float32, channel 0 = the CPU's own mip LOD at every covered
    pixel, zeros elsewhere: gs_lod16 over the view depth (key (0, 0), the
    PlayStation 2 road, C022), the per-triangle table (TRIANGLE, C079),
    or compute_lod on the analytic derivatives for a (w, h) texture (a
    Mip Level Select on the derivative road, C072). The same functions
    on the same attributes() output the CPU's sampler reads; one upload
    per distinct key per frame, like hal_uvgrad, and one BUILD per key
    per job (A12.8: fill_base runs per pass)."""
    from ..core import texture as TX
    cache = getattr(job, '_lod_fields', None)
    if cache is None:
        cache = job._lod_fields = {}
    hit = cache.get((id(gbuf), key))
    if hit is not None and hit[0] is gbuf:
        return hit[1]
    h, w = gbuf.tri.shape
    field = np.zeros((h, w, 4), np.float32)
    cov = gbuf.tri >= 0
    if cov.any():
        tri = gbuf.tri[cov]
        bary = gbuf.bary[cov]
        opts = TX.sample_opts(st)
        bias = float(getattr(st, 'tex_mip_bias', 0.0) or 0.0)
        if opts['lod_source'] == 'GS_Q':
            P = job.attributes(tri, bary, need={'P'})[0]                  # A0.5: P alone
            field[cov, 0] = TX.gs_lod16(job.view_depth(P), opts['lod_k16'], opts['lod_l'])
        elif opts['lod_source'] == 'TRIANGLE':
            field[cov, 0] = job.tri_lod(key[0], key[1], bias)[tri]         # C079
        else:
            tri, bary, uv = _field_uv(job, gbuf, st, cov)
            du, dv = job.uv_screen_gradients(tri, bary, uv)
            field[cov, 0] = TX.compute_lod(du, dv, key[0], key[1], bias)    # C072 on the derivative road
    cache[(id(gbuf), key)] = (gbuf, field)
    return field


#: R251: fog runs INSIDE the deferred material pass (`hal_fog`,
#: gpu/material.FOG_GLSL) -- bitwise the CPU's apply_fog in the simulator.
#: The module switch exists for the A/B in the test suite (the
#: MATERIAL_TEXELS precedent) and, unlike that precedent, it is IN the plan
#: signature: a gate the plan reads must be, or a cache hit walks straight
#: past the refusal. Off, every primary pass carries `fog_cpu` and the
#: 1.89.0 readback fog runs below.
FOG_ON_GPU = True


def _fog_structure(st, scene):
    """`consts['fog']`: the branches hal_fog emits (core/fog.structure)
    plus the camera type, which decides the z-fog form (F007)."""
    from ..core import fog as _FOG
    fs = _FOG.structure(st)
    if fs:
        fs['ortho'] = str(getattr(getattr(scene, 'camera', None), 'type',
                                  'PERSP')) == 'ORTHO'
        # R251 LIGHT-A2: the plan's fog notes, once per plan (the
        # ORTHO fallback of the GC range adjust, a table inert under
        # Ground Fog); the CPU road prints the same lines once
        for _note in _FOG.plan_notes(st, fs['ortho']):
            print(_note)
        # F015's fog half (LIGHT-A2): the lobe joins the target only when
        # a screen-spot lamp exists AND Spotlight Fog > 0 (both in the
        # signature: _light_sig's screen_spot, st_sig's fog_spot gate)
        fs['spot'] = _FOG.spot_on(st) and any(
            getattr(l, 'screen_spot', False)
            for l in (getattr(scene, 'lights', ()) or ()))
    return fs


def _fogtab_needed(st, scene):
    """`consts['fogtab']`, lighting.md section 0's ONE rule: iff true,
    every pass declares, appends and binds the hal_fogtab sampler."""
    if bool(getattr(st, 'fog', False)):
        return True
    if str(getattr(st, 'specular_viewer', 'PIXEL')) == 'AXIS':
        return True
    # R251 LIGHT-B1 (F015): a screen-spot lamp reads the view row from
    # hal_fogtab texel 3 (integrator: LIGHT-B1's clause of the one rule)
    if any(getattr(l, 'screen_spot', False)
           for l in (getattr(scene, 'lights', ()) or ())):
        return True
    from ..core import shading as _SH
    axis = getattr(_SH, 'AXIS_MODELS', None)
    if axis:
        from ..core.render import material_model
        return any(material_model(m, st) in axis
                   for m in (getattr(scene, 'materials', ()) or ()))
    return False


def _fog_readback(job, gbuf, passes, out, hit):
    """The CPU's own fog over the deferred readback, for the materials the
    planner named `fog_cpu` -- and only those.

    R251: fog is in the material pass for every primary pixel-rate pass
    (`hal_fog`); this road remains for the materials whose composites land
    AFTER the readback (traced reflections and refractions, a CPU-evaluated
    environment term: the CPU fogs `base + r*hit`, so in-shader fog would
    give `fog(base) + r*hit`) and for the module switch. `_one_material`
    decides `binds['fog_cpu']` before the source exists and prints the
    reason once per plan; here the selection is by material id, exactly
    the 1.89.0 body over those ids, now handing the screen pixels to
    `fog_for_points` (the Voodoo dither reads them).

    Returns `out` by IDENTITY when nothing was fogged on the CPU (the
    tests assert structure, never wall-clock). Vertex-rate materials SKIP
    as before: shade_batch lit their corners with rate_mode LIGHT, which
    runs apply_fog at the corner, and the interpolated product already
    carries it.
    """
    st = job.settings
    if not getattr(st, 'fog', False) or not hit.any():
        return out
    cpu_ids = sorted({int(mat_id) for mat_id, _n, _s, binds in passes
                      if (binds or {}).get('fog_cpu')
                      and not (binds or {}).get('vlight')})
    if not cpu_ids:
        return out
    from ..core.render import fog_for_points
    py, px = np.nonzero(hit)
    tri = gbuf.tri[py, px]
    mesh = job.scene.mesh
    mi = mesh.mat_index[tri] if mesh.mat_index is not None \
        else np.zeros(tri.size, np.int32)
    sel = np.isin(mi, np.asarray(cpu_ids, np.int64))
    if not sel.any():
        return out
    pys, pxs = py[sel], px[sel]
    # R251 C119 (MAT-B): the CPU fogs at the cell corner's depth (its
    # ctx.depth derives from the snapped P), so the readback snaps with
    # the same grid object before fogging
    _fb = gbuf.bary[py, px][sel]
    _rg = REYES.grid_for(job)
    if _rg is not None:
        _fb = REYES.snap(_fb, tri[sel], _rg)
    # R251 LIGHT-A2 (F006): the fog_cpu materials' own dials, per
    # pixel from binds['fog_mat'] (burn, bias, bank), so the refusal
    # road honours them exactly as shade_batch's surf does
    from ..core.fog import readback_surf as _rbs
    surf_rb = _rbs(passes, mi[sel])
    out[pys, pxs] = fog_for_points(
        job, tri[sel], _fb, out[pys, pxs],
        px=pxs, py=pys, surf=surf_rb)
    return out


#: the reflect stage's sub-split, reset per frame by the callers that
#: publish LAST_TIMINGS: where the sweep seconds actually go -- the GPU
#: trace, the secondary draws+reads, the CPU sky along miss rays -- plus
#: how many levels ran, how many were all-miss skips, and the ray count
_SWEEP_STATS = {'trace_ms': 0.0, 'draw_ms': 0.0, 'env_ms': 0.0,
                'levels': 0, 'skips': 0, 'rays': 0}


def _reset_sweep_stats():
    _SWEEP_STATS.update(trace_ms=0.0, draw_ms=0.0, env_ms=0.0,
                        levels=0, skips=0, rays=0)


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
    blur = float(getattr(job.settings, 'reflection_blur', 0.0))
    bsamples = max(int(getattr(job.settings, 'reflection_blur_samples',
                               1)), 1)
    bseed = int(getattr(job.settings, 'seed', 0) or 0)

    def lane_add(img, hitm, dirs, py, px):
        """Per-lane traced value: the hit's colour, or the sky along
        the ray -- exactly what `trace()` returns per ray."""
        import time as _time
        from ..core.render import world_color
        add = np.zeros((py.size, 3), np.float32)
        hitb = np.asarray(hitm, bool)
        if hitb.any():
            add[hitb] = img[py[hitb], px[hitb], :3]
        miss = ~hitb
        if miss.any():
            t0 = _time.perf_counter()
            wc = world_color(job.scene, job.settings, dirs[miss],
                             job.textures, int(miss.sum()), eye=job.eye)
            _SWEEP_STATS['env_ms'] += (_time.perf_counter() - t0) * 1000.0
            add[miss] = np.asarray(wc, np.float32)[:, :3]
        return add

    def traced_add(level, py, px, org, dirs, N):
        """One reflective spawn's lane values -- the cone when blur is on.

        `_blurred_reflection`, sweep edition: K jittered directions from
        the same deterministic streams, each traced through the SAME
        recursion (children blur again, exactly the CPU's tree), the K
        lane values averaged. No blur reduces to the single ray.
        """
        if blur <= 1e-3 or bsamples < 1 or N is None:
            img, hitm = shade_level(level, py, px, org, dirs)
            return lane_add(img, hitm, dirs, py, px)
        acc = np.zeros((py.size, 3), np.float32)
        for k in range(bsamples):
            d = _cone_jitter(dirs, N, px, py, k, bseed, blur)
            imgK, hitK = shade_level(level, py, px, org, d)
            acc += lane_add(imgK, hitK, d, py, px)
        return acc / np.float32(bsamples)

    def shade_level(level, py, px, org, dirs):
        import time as _time
        t0 = _time.perf_counter()
        tid, _t, u, v = intersect(org, dirs)
        _SWEEP_STATS['trace_ms'] += (_time.perf_counter() - t0) * 1000.0
        _SWEEP_STATS['levels'] += 1
        _SWEEP_STATS['rays'] += int(py.size)
        sec_ids, hitm = _secondary_ids(h, w, py, px, tid, u, v)
        plist = rplan['secondary'] if level >= depth \
            else rplan['secondary_mid']
        if not hitm.any():
            # every ray of this level missed: no secondary pass owns a
            # pixel, so the level image is exactly zero -- skip the
            # draws (a depth-8 sweep used to pay full-screen passes and
            # an 800 MB readback per level to paint nothing)
            _SWEEP_STATS['skips'] += 1
            img = np.zeros((h, w, 3), np.float32)
        else:
            hy, hx = py[hitm], px[hitm]
            hreg = (int(hx.min()), int(hy.min()),
                    int(hx.max() - hx.min() + 1),
                    int(hy.max() - hy.min() + 1))
            t0 = _time.perf_counter()
            img = draw_secondary(plist, sec_ids, level, hit_region=hreg)
            _SWEEP_STATS['draw_ms'] += (_time.perf_counter() - t0) * 1000.0
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
                org2, dirs2, N2 = _hit_rays(job, tid[sel], bary2, which,
                                            rplan)
                cpy, cpx = py[sel], px[sel]
                if which == 'reflective':
                    # the cone applies at EVERY reflective spawn, as the
                    # CPU's recursion does; the average composites with
                    # the hit material's constants exactly like one ray
                    add2 = traced_add(level + 1, cpy, cpx, org2, dirs2,
                                      N2)
                    _child_reflect_add(job, rplan, img, cpy, cpx, add2,
                                       m[sel])
                else:
                    child_img, child_hitm = shade_level(level + 1, cpy,
                                                        cpx, org2, dirs2)
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
        py, px, org, dirs, Nspawn = rays_fn(job, gbuf, rplan)
        if py.size == 0:
            continue
        if which == 'reflective':
            # the blur cone (when on) expands and averages here; a
            # sharp mirror reduces to one ray through the same road
            add1 = traced_add(1, py, px, org, dirs, Nspawn)
            out = _composite_reflections_add(job, gbuf, rplan, out,
                                             py, px, add1)
        else:
            img1, hitm1 = shade_level(1, py, px, org, dirs)
            out = composite_fn(job, gbuf, rplan, out, py, px, dirs,
                               hitm1, img1)
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
    for _cn, _cv in _cam_uniforms(job).items():
        runi[_cn] = np.tile(np.asarray(_cv, np.float32)[None, :], (gn, 1))
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
    # R248: the driver's own rule, mirrored -- a material on screen
    # without a pass refuses by name, and only the passes on screen run
    present = _present_materials(job.scene.mesh, gbuf)
    planned_ids = {int(p[0]) for p in passes}
    missing = sorted(present - planned_ids)
    if missing:
        unp = (atlases or {}).get('__unplanned') or {}
        return None, (f'{len(missing)} material(s) on screen have no GPU '
                      'pass: ' + '; '.join(
                          f"'{_material_name(job, mi)}'"
                          + (f' ({unp[mi]})' if mi in unp else
                             ' (no pass in the plan)')
                          for mi in missing[:4]))
    passes = [p for p in passes if int(p[0]) in present]
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
        _key, build = entry[0], entry[1]    # R251 C015: a stamped 3-tuple
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

    if not getattr(job.settings, 'tex_perspective', True):
        # affine: the rasteriser's own screen-linear barycentrics join
        # the sim's textures by name, exactly as the driver binds them
        from . import gbuffer as GB
        tex['hal_gb_idslin'] = Texture(GB.pack_ids_lin(gbuf),
                                       colorspace='Non-Color',
                                       filt='NEAREST', wrap='EXTEND')

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
        # R251 (TEX-2): the summed-area atlases (C088) and the CPU-decided
        # LOD fields (C022 / C079 / C072), by name as the driver binds them
        for sname, key in (binds.get('textures_sat') or {}).items():
            from .material import sat_atlas as _sat_atlas
            uni[sname] = Texture(_sat_atlas(job.textures[key]),
                                 colorspace='Non-Color', filt='NEAREST',
                                 wrap='EXTEND')
        for uniform, key in (binds.get('textures_lod') or {}).items():
            uni[uniform] = Texture(_lod_field(job, gbuf, job.settings, key),
                                   colorspace='Non-Color', filt='NEAREST',
                                   wrap='EXTEND')
        if binds.get('needs_uvgrad'):
            uni['hal_uvgrad'] = Texture(_uvgrad_field(job, gbuf),
                                        colorspace='Non-Color',
                                        filt='NEAREST', wrap='EXTEND')
        if binds.get('needs_wirescreen'):
            uni['hal_vscreen'] = Texture(_pack_vscreen(job),
                                         colorspace='Non-Color',
                                         filt='NEAREST', wrap='EXTEND')
        if 'hal_reyes' in (binds.get('samplers') or ()):
            # R251 C119 (MAT-B): the per-triangle (n, inv_n) pair, the
            # CPU's own numbers, beside hal_vlight
            uni['hal_reyes'] = Texture(REYES.image(job),
                                       colorspace='Non-Color',
                                       filt='NEAREST', wrap='EXTEND')
        uni['hal_attr_side'] = np.full(n, float(side), np.float32)
        uni['hal_slot_count'] = np.full(n, 4.0, np.float32)
        uni['hal_tri_side'] = np.full(n, float(tside), np.float32)
        uni['hal_eye'] = np.tile(np.asarray(job.eye, np.float32)[None, :],
                                 (n, 1))
        for _cn, _cv in _cam_uniforms(job).items():
            uni[_cn] = np.tile(np.asarray(_cv, np.float32)[None, :], (n, 1))
        # per-frame scalars a coded shader may read; unused are inert
        uni['hal_time'] = np.full(n, float(getattr(job.scene, 'time', 0.0)),
                                  np.float32)
        uni['hal_frame'] = np.full(n, float(getattr(job.scene, 'frame', 0)),
                                   np.float32)
        # R174: this pass's row in the hal_mats value texture
        uni['hal_mrow'] = np.full(n, float(binds.get('mat_row', 0)),
                                  np.float32)
        uni['vUV'] = uv
        return uni

    def run_passes(pass_list, ids_texture):
        got_out = np.zeros((n, 3), np.float32)
        got_hit = np.zeros(n, bool)
        got_stip = np.zeros(n, bool)
        for mat_id, name, src, binds in pass_list:
            sim_src = src.replace('in vec2 vUV;', 'uniform vec2 vUV;')
            prog, err = try_compile(sim_src, 'GLSL')
            if prog is None:
                return None, None, None, \
                    f"'{name}' would not compile: {err}"
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
                    return None, None, None, \
                        f"'{name}' height pass would not compile: {perr}"
                puni = fill_base(dict(tex), ids_texture, pbinds)
                pgot = pprog.run(puni, {}, n)[0]['Color']
                uni[uname] = Texture(pgot.reshape(h, w, 4),
                                     colorspace='Non-Color', filt='NEAREST',
                                     wrap='EXTEND')
            got = prog.run(uni, {}, n)[0]['Color']
            keep = got[:, 3] > 0.5
            np.copyto(got_out, got[:, :3], where=keep[:, None])
            got_hit |= keep
            got_stip |= got[:, 3] > 0.75
        return got_out, got_hit, got_stip, None

    out, hit, stip01, why = run_passes(passes, tex['hal_gb_ids'])
    if out is None:
        return None, why
    out = out.reshape(h, w, 3)
    hit = hit.reshape(h, w)
    # R248: the coverage law, mirrored from shade_frame
    _cov = gbuf.tri >= 0
    _mesh = job.scene.mesh
    if _mesh.mat_index is not None:
        _mpx = np.where(_cov, _mesh.mat_index[np.maximum(gbuf.tri, 0)], -1)
        _owed = _cov & np.isin(_mpx, list(planned_ids)) & ~hit
    else:
        _mpx = None
        _owed = _cov & ~hit
    if _owed.any():
        _names = [] if _mpx is None else [
            f"'{_material_name(job, int(mi))}'"
            for mi in np.unique(_mpx[_owed])[:4]]
        return None, (f'{int(_owed.sum())} covered pixel(s) came back '
                      'unshaded from the material passes '
                      f'({", ".join(_names) or "?"}); the frame shades on '
                      'the CPU')
    if str(getattr(job.settings, 'transparency', 'NONE')) == 'STIPPLE':
        # the encoded Screen Door bit, decoded exactly as the driver
        # path decodes its readback and carried the same way
        gbuf.gpu_alpha = stip01.reshape(h, w)

    env_plan = (atlases or {}).get('__env')
    out = _apply_cpu_env_primary(job, gbuf,
                                 (env_plan or {}).get('primary'), out)

    rplan = (atlases or {}).get('__reflect')
    if rplan is not None:
        from .rtrace import simulate_intersect

        def draw_secondary(plist, sec_ids, _level, hit_region=None):
            # the front-end shades the full frame; outside the hit box
            # every pass keeps zero, which is what the region read
            # returns on the driver -- the consumed pixels agree
            sec_out, _sec_hit, _sec_stip, why = run_passes(
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
    # R250: the sky pass, mirrored -- the same source over the same ids,
    # its colour at the uncovered pixels (covered ones discard); the
    # verdict rides the G-buffer as the driver road's does
    _vp = getattr(job, 'vp', None)
    try:
        gbuf.gpu_sky = False
        gbuf.gpu_sky_why = ''
    except AttributeError:
        pass
    if _vp is not None:
        from . import sky as GSKY
        sky4, sky_why = GSKY.simulate(job.scene, gbuf, job.settings, _vp,
                                      job.eye, job.textures,
                                      ss=int(getattr(job, 'ss', 1) or 1))
        if sky4 is not None:
            unc = gbuf.tri < 0
            out[unc] = sky4[unc][:, :3]
            try:
                gbuf.gpu_sky = True
                gbuf.sim_sky = sky4
            except AttributeError:
                pass
        else:
            try:
                gbuf.gpu_sky_why = str(sky_why)
            except AttributeError:
                pass
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
    _compile_before = device.compile_stats()
    t0 = _time.perf_counter()
    mesh = job.scene.mesh
    mkey = _mesh_key(mesh)
    # R248: only the passes ON SCREEN compile and draw -- a plan carries
    # a pass for every material the mesh has, and a pass with no pixel
    # writes nothing, so skipping it is bit-identical and saves a
    # driver compile per hidden material. A material on screen with NO
    # pass is the field's unshaded-black defect: it never draws black
    # again -- the frame refuses, by name, and shades on the CPU
    present = _present_materials(mesh, gbuf)
    planned_ids = {int(p[0]) for p in passes}
    missing = sorted(present - planned_ids)
    if missing:
        unp = atlases.get('__unplanned') or {}
        why = '; '.join(
            f"'{_material_name(job, mi)}'"
            + (f' ({unp[mi]})' if mi in unp else ' (no pass in the plan)')
            for mi in missing[:4])
        return None, (f'{len(missing)} material(s) on screen have no GPU '
                      f'pass: {why}')
    all_passes = passes
    passes = [p for p in passes if int(p[0]) in present]
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
        # R249: the ink pass reads this frame's ids texture too (the
        # G-buffer holds it; a later drop is safe -- every draw that
        # reads it is followed by a readback that proves it executed)
        try:
            gbuf.gpu_ids_texture = tex_ids
        except AttributeError:
            pass
        tex_attrs = device.upload_cached(('gb_attrs',) + mkey, build_attrs)
        tex_tris = device.upload_cached(('gb_tris',) + mkey, build_tris)
        tex_shadows = {sname: device.upload_cached(entry[0], entry[1],
                                                   *entry[2:])
                       for sname, entry in atlases.items()
                       if not sname.startswith('__')}
        if not getattr(job.settings, 'tex_perspective', True):
            # affine texture mode: the rasteriser's own screen-linear
            # barycentrics, per frame like the ids texture itself, bound
            # by name to every pass and height pre-pass that wants them
            tex_shadows['hal_gb_idslin'] = device.upload(
                GB.pack_ids_lin(gbuf))
        all_binds = [b for _mi, _n2, _s2, b in
                     (passes + (rplan['secondary'] if rplan else []))]
        for b in list(all_binds):
            all_binds.extend(p[2] for p in (b.get('prepasses') or ()))
        if any(b.get('needs_wirescreen') for b in all_binds):
            # per-corner screen positions for the Wireframe node's
            # Pixel Size: camera-dependent, packed per frame like the
            # footprint field, bound by name through tex_shadows
            tex_shadows['hal_vscreen'] = device.upload(_pack_vscreen(job))
        if any('hal_reyes' in (b.get('samplers') or ()) for b in all_binds):
            # R251 C119 (MAT-B): a VALUE (it depends on the camera, which
            # the plan signature excludes): uploaded per frame beside
            # hal_vlight, bound by name to every pass that declares it
            tex_shadows['hal_reyes'] = device.upload(REYES.image(job))
        tex_images = _gather_pass_textures(all_binds, job.textures,
                                           device.upload_cached)
        # the footprint field: the CPU's analytic UV derivatives for every
        # covered pixel, one RGBA32F upload shared by every filtered
        # sampler (du/dx, du/dy, dv/dx, dv/dy)
        tex_uvgrad = None
        if any((b or {}).get('needs_uvgrad') for b in all_binds):
            tex_uvgrad = device.upload(_uvgrad_field(job, gbuf))
        # R251 (TEX-2): the CPU-decided LOD fields (hal_lodq /
        # hal_lod_WxH), one upload per distinct key per frame
        tex_lod = {}
        for b in all_binds:
            for _un, _lk in ((b or {}).get('textures_lod') or {}).items():
                if _un not in tex_lod:
                    tex_lod[_un] = device.upload(
                        _lod_field(job, gbuf, job.settings, _lk))
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
        # R250: an unexpected exception here is a reason AND a traceback
        # (the field's 'object() takes no arguments' named nothing)
        import traceback as _tb
        print('[Halcyon GPU] the G-buffer upload raised: '
              + _tb.format_exc())
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
                    'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
            shader, err = device.compile_dynamic(_pool_tag(tag, src, spec),
                                                 src, spec)
            if shader is None:
                return None, f"the driver rejected '{name}': {err}"
            bind = {'hal_gb_ids': ids_tex, 'hal_gb_attrs': tex_attrs,
                    'hal_gb_tris': tex_tris}
            for sname in samplers:
                if sname == 'hal_vlight' and int(mat_id) in vlight_tex:
                    bind[sname] = vlight_tex[int(mat_id)]
                elif sname == 'hal_uvgrad' and tex_uvgrad is not None:
                    bind[sname] = tex_uvgrad
                elif sname in tex_lod:
                    bind[sname] = tex_lod[sname]              # R251 TEX-2
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
                elif u == 'hal_mrow':
                    extra[u] = float(binds.get('mat_row', 0))
            built.append((name, shader, bind, extra))
        return built, None

    uni = {'hal_attr_side': float(side), 'hal_slot_count': 4.0,
           'hal_tri_side': float(tside),
           'hal_eye': tuple(float(v) for v in job.eye),
           **_cam_uniforms(job)}

    # bump height pre-passes draw FIRST, each into its own target, so
    # build_draws can bind their colour textures into the main passes.
    # Fragment renders one, fragment samples it: no stage crossing.
    # R183: this block was UNATTRIBUTED -- it ran between the upload and
    # draw windows and its cost (a CPU height evaluation is seconds at a
    # supersampled resolution) landed in the composite residual as
    # 'other'. It owns its milliseconds now, cpu-vs-gpu counted.
    _t_pre = _time.perf_counter()
    _pre_n = _pre_cpu = 0
    _pre_whys = []
    prepass_targets = []
    for mat_id, name, _src, binds in passes:
        for uname, psrc, pbinds in (binds.get('prepasses') or ()):
            if pbinds.get('cpu'):
                # the emitter refused this height chain; the renderer's
                # own evaluator produces the image instead, exactly.
                # WHO and WHY are recorded for the split: the field's
                # composite cost is these passes, and naming the
                # refusing ingredient is what makes it fixable
                try:
                    _pre_n += 1
                    _pre_cpu += 1
                    _why = str(pbinds.get('why', '') or '')
                    _why = _why.replace(' evaluates on the CPU into the '
                                        'height pre-pass', '')
                    _pre_whys.append(f"'{name}': {_why}"[:90])
                    himg = _cpu_height_image(job, gbuf, mat_id,
                                             pbinds['node'])
                    prepass_tex[(mat_id, uname)] = device.upload(himg)
                except Exception as exc:                        # noqa: BLE001
                    for t in prepass_targets:
                        t.free()
                    return None, f"'{name}' CPU height pass failed: {exc}"
                continue
            _pre_n += 1
            spec = {'samplers': ['hal_gb_ids', 'hal_gb_attrs',
                                 'hal_gb_tris']
                    + list(pbinds.get('samplers', ())),
                    'floats': ['hal_attr_side', 'hal_slot_count',
                               'hal_tri_side']
                    + list(pbinds.get('frame_uniforms', ())),
                    'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
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
                elif sname in tex_shadows:
                    # the affine screen-linear ids, and any future
                    # by-name texture a height pre-pass reads
                    bind[sname] = tex_shadows[sname]
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
    _pre_ms = (_time.perf_counter() - _t_pre) * 1000.0

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
                 'vec3': ['hal_eye', 'hal_cam_right', 'hal_cam_up', 'hal_cam_back']}
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

    # R250: the sky / background, drawn LAST in the same burst -- blend
    # NONE, covered pixels discard, so the material passes' texels stand
    # and every uncovered texel is the world's colour whatever a pass
    # left there. The readback is then the whole frame. Refusals name
    # themselves and the CPU draws the sky as it did before
    from . import sky as GSKY
    _sky_draw, _sky_plan, _sky_why = None, None, None
    _job_vp = getattr(job, 'vp', None)
    if _job_vp is not None:
        try:
            _sky_draw, _sky_plan = GSKY.prepare(
                job.scene, gbuf, job.settings, _job_vp, job.eye,
                job.textures, int(getattr(job, 'ss', 1) or 1), tex_ids)
            if _sky_draw is None:
                _sky_why = _sky_plan
                _sky_plan = None
        except Exception as exc:                                # noqa: BLE001
            _sky_draw, _sky_plan = None, None
            _sky_why = f'the sky pass failed to prepare: {exc}'
    else:
        _sky_why = 'the frame carries no view-projection for the sky pass'
    _sky_drawn = False

    t_draw = 0.0
    _burst_snap = {}
    _c_own = _c_env = 0.0
    target = device.Target(w, h)
    try:
        # every pass blends into the one target -- each material writes only
        # where its alpha is one, premultiplied blending leaves the rest
        # alone -- and the frame reads back once, however many materials.
        # Per-material readbacks plus a NumPy merge measured 5.9 ms of a
        # 14.3 ms warm frame; this is that line item, removed.
        t1 = _time.perf_counter()
        try:
            # R175: every pass and the readback in ONE marshal crossing
            # -- the same commands in the same order, none of the
            # per-pass queue sleeps (was ~14 ms of latency per pass)
            _draws = [(shader, {**uni, **extra} if extra else uni, bind,
                       target, 'ALPHA_PREMULT', i == 0, None)
                      for i, (name, shader, bind, extra)
                      in enumerate(plan_draw)]
            if _sky_draw is not None:
                _sk_sh, _sk_uni, _sk_bind = _sky_draw
                _draws.append((_sk_sh, _sk_uni, _sk_bind, target, 'NONE',
                               False, None))
            got = device.draw_many(_draws, read=target)
            _sky_drawn = _sky_draw is not None
            # snapshot NOW: the sweeps and layer ranks run their own
            # bursts before LAST_TIMINGS is written, and this pair must
            # describe the OPAQUE frame's burst
            _burst_snap = dict(device.LAST_BURST)
        except Exception as exc:                                # noqa: BLE001
            # a driver that objects to the blend path gets the readback
            # path, not a CPU frame: slower is better than absent
            print(f'[Halcyon GPU] blended compositing fell back to per-pass '
                  f'readback: {type(exc).__name__}: {exc}')
            got = None
        stip = str(getattr(job.settings, 'transparency', 'NONE')) == \
            'STIPPLE'
        if got is not None:
            t_draw = _time.perf_counter() - t1
            _tc0 = _time.perf_counter()
            # R250: `hit` means 'a material pass wrote this covered
            # pixel'. The sky draw writes alpha 1 at UNCOVERED pixels,
            # so the mask is taken over the G-buffer's coverage -- the
            # fog readback indexes triangles through it, and a sky pixel
            # (tri -1) must never reach that road
            _covered_px = gbuf.tri >= 0
            hit = (got[:, :, 3] > 0.5) & _covered_px
            # the colour planes are exactly what a mask would have
            # produced at covered pixels (the target was cleared to zero
            # and the blend leaves untouched pixels at zero); at
            # uncovered pixels they are the sky when it drew
            out = np.ascontiguousarray(got[:, :, :3], np.float32)
            if stip:
                # the encoded Screen Door bit (see the pass composite):
                # 0.9 = kept, 0.6 = dropped; decoded here and carried
                # out of band on the G-buffer for the frame's alpha
                gbuf.gpu_alpha = (got[:, :, 3] > 0.75) & _covered_px
            _c_own = (_time.perf_counter() - _tc0) * 1000.0
        else:
            out = np.zeros((h, w, 3), np.float32)
            hit = np.zeros((h, w), bool)
            _sky_drawn = False
            _sky_why = 'the blended burst fell back to per-pass readbacks'
            _covered_px = gbuf.tri >= 0
            if stip:
                gbuf.gpu_alpha = np.zeros((h, w), bool)
            for name, shader, bind, extra in plan_draw:
                t1 = _time.perf_counter()
                try:
                    frame = device.draw_fullscreen(shader, {**uni, **extra}
                                                   if extra else uni, bind,
                                                   target)
                except Exception as exc:                        # noqa: BLE001
                    return None, f"drawing '{name}' failed: {exc}"
                t_draw += _time.perf_counter() - t1
                keep = (frame[:, :, 3] > 0.5) & _covered_px
                # masked copy instead of boolean fancy indexing: the same
                # pixels move, but no index lists are materialised -- at a
                # supersampled frame this was most of the composite bucket
                # (measured 3.7x at 7200^2, byte-identical)
                np.copyto(out, frame[:, :, :3], where=keep[:, :, None])
                hit |= keep
                if stip:
                    gbuf.gpu_alpha |= frame[:, :, 3] > 0.75
    finally:
        for t in prepass_targets:
            t.free()
        # R250: the frame stays on the GPU when nothing on the CPU will
        # touch its readback -- the sky drew (every pixel is the frame's),
        # no Screen Door (its alpha codes differ from the frame's alpha),
        # no fog, no CPU environment composite, no reflection sweep. The
        # target is then kept for the ink, the resolve and the post chain
        # (gpu/frame.py owns its lifetime); otherwise it is pooled as before
        _keep_target = bool(
            got is not None and _sky_drawn and not stip
            # R251: fog runs in the pass; only a fog_cpu material
            # (traced / env composite, or the module switch) edits it
            and not any((b or {}).get('fog_cpu')
                        for _fm, _fn, _fs, b in passes)
            and not (atlases.get('__env') or {}).get('primary')
            and atlases.get('__reflect') is None)
        if not _keep_target:
            target.free()

    def _fail(why):
        if _keep_target:
            target.free()
        return None, why

    # R248: every covered pixel whose material has a pass must have been
    # written (alpha one, or the Screen Door's encoded bit). A pixel
    # left at the cleared zero is the unshaded-black defect on the
    # driver's side -- a pass that compiled and wrote nothing -- and
    # the frame refuses by name rather than show it
    try:
        covered = gbuf.tri >= 0
        if mesh.mat_index is not None:
            mat_px = np.where(covered, mesh.mat_index[np.maximum(gbuf.tri, 0)], -1)
            owed = covered & np.isin(mat_px, list(planned_ids)) & ~hit
        else:
            owed = covered & ~hit
        n_owed = int(owed.sum())
    except Exception:                                           # noqa: BLE001
        n_owed, owed, mat_px = 0, None, None
    if n_owed:
        names = []
        try:
            for mi in np.unique(mat_px[owed])[:4]:
                names.append(f"'{_material_name(job, int(mi))}'")
        except Exception:                                       # noqa: BLE001
            pass
        return _fail(f'{n_owed} covered pixel(s) came back unshaded from '
                     f'the material passes ({", ".join(names) or "?"}); '
                     'the frame shades on the CPU')

    # the CPU-composite environment term: for worlds richer than the
    # baked GLSL paths, the renderer's own world_color along the
    # reflected rays, added exactly where the CPU adds it (last)
    env_plan = atlases.get('__env')
    _tc1 = _time.perf_counter()
    try:
        out = _apply_cpu_env_primary(job, gbuf,
                                     (env_plan or {}).get('primary'), out)
    except Exception as exc:                                    # noqa: BLE001
        return _fail(f'the environment composite failed: {exc}')
    _c_env = (_time.perf_counter() - _tc1) * 1000.0

    # the traced bounces: rays off the reflective then refractive pixels,
    # closest hits shaded by the SAME materials through their secondary
    # passes, each blend composited exactly as _add_raytraced does it,
    # in _add_raytraced's order
    t_reflect = 0.0
    _RAY_BUILD[0] = 0.0
    _reset_sweep_stats()
    if rplan is not None:
        t1 = _time.perf_counter()
        from . import rtrace as RT

        def draw_secondary(plist, sec_ids, level, hit_region=None):
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
                # scissor to the hit box: a secondary pass can only own
                # hit pixels, and past level 1 the hits huddle in a
                # corner of a frame the pass used to cover whole.
                # R175: all sweeps and the readback in one crossing.
                _sparse = hit_region is not None and \
                    hit_region[2] * hit_region[3] < 0.5 * w * h
                sec_r = device.draw_many(
                    [(shader, {**uni, **extra} if extra else uni, bind,
                      target2, 'ALPHA_PREMULT', i2 == 0, hit_region)
                     for i2, (name, shader, bind, extra)
                     in enumerate(sec_draw)],
                    read=target2,
                    read_region=hit_region if _sparse else None)
                if _sparse:
                    # sparse hits: the box crossed the bus, not the
                    # frame -- the composite only ever looks at hit
                    # pixels, and the rest of a cleared target is zero
                    rx, ry, rw2, rh2 = hit_region
                    sec_img = np.zeros((h, w, 4), np.float32)
                    sec_img[ry:ry + rh2, rx:rx + rw2] = sec_r
                else:
                    sec_img = sec_r
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
            return _fail(str(sf))
        except Exception as exc:                                # noqa: BLE001
            # shade_frame's contract is that EVERY failure is a reason
            # and the caller shades on the CPU -- the field's depth-2
            # section died whole because a ValueError escaped this loop
            # instead of becoming one
            return _fail(f'the ray sweeps failed: '
                         f'{type(exc).__name__}: {exc}')
        t_reflect = _time.perf_counter() - t1

    total = _time.perf_counter() - t_all
    # driver compilations that happened INSIDE this frame: the field's
    # 33.9s cold frame reported 'composite 19656 ms' -- which was the
    # driver compiling 25 material shaders in the untimed gap where
    # build_draws runs. Named now, and subtracted from composite so
    # that bucket means what it says.
    _cn1, _cms1 = device.compile_stats()
    compile_ms = max(_cms1 - _compile_before[1], 0.0)
    compile_n = max(_cn1 - _compile_before[0], 0)
    LAST_TIMINGS.clear()
    LAST_TIMINGS.update(
        plan_ms=t_plan * 1000.0,
        pack_upload_ms=t_upload * 1000.0,
        draw_read_ms=t_draw * 1000.0,
        # R175b: measured INSIDE the burst crossing -- submission vs
        # the readback wait (= the GPU actually executing). This pair
        # names whether a slow draw+read is main-thread overhead or
        # real driver work. Snapshotted at the opaque burst, before the
        # sweeps run bursts of their own.
        burst_draw_ms=float(_burst_snap.get('draw_ms', 0.0)),
        burst_read_ms=float(_burst_snap.get('read_ms', 0.0)),
        reflect_ms=t_reflect * 1000.0,
        ray_build_ms=_RAY_BUILD[0],
        compile_ms=compile_ms,
        compile_n=compile_n,
        composite_ms=max((total - t_upload - t_draw - t_reflect) * 1000.0
                         - compile_ms, 0.0),
        passes=len(passes),
        # the reflect sub-split: where those milliseconds went and how
        # much of the recursion was all-miss levels skipped outright
        reflect_trace_ms=_SWEEP_STATS['trace_ms'],
        reflect_draw_ms=_SWEEP_STATS['draw_ms'],
        reflect_env_ms=_SWEEP_STATS['env_ms'],
        reflect_levels=_SWEEP_STATS['levels'],
        reflect_skips=_SWEEP_STATS['skips'],
        reflect_rays=_SWEEP_STATS['rays'])
    # R182: the composite bucket is a RESIDUAL (total minus the named
    # stages), and the field's 16x-supersampled frame put 23 seconds in
    # it. Name the measured parts so the residual's own residual is the
    # suspect list, not the whole bucket.
    LAST_TIMINGS['c_own_ms'] = _c_own
    LAST_TIMINGS['c_env_ms'] = _c_env
    LAST_TIMINGS['prepass_ms'] = _pre_ms
    LAST_TIMINGS['prepass_n'] = _pre_n
    LAST_TIMINGS['prepass_cpu'] = _pre_cpu
    LAST_TIMINGS['prepass_whys'] = '; '.join(
        sorted(set(_pre_whys)))[:220]
    LAST_TIMINGS['c_other_ms'] = max(
        float(LAST_TIMINGS.get('composite_ms', 0.0)) - _c_own - _c_env
        - _pre_ms, 0.0)
    _tcf = _time.perf_counter()
    out = _fog_readback(job, gbuf, passes, out, hit)
    LAST_TIMINGS['c_fog_ms'] = (_time.perf_counter() - _tcf) * 1000.0
    # R250: the sky's verdict and the frame's residency, for render()
    # and the field test. gpu_frame_rgba is the readback itself when it
    # IS the frame (nothing on the CPU touched it): rgb whole, the sky
    # in the uncovered pixels; render() writes only the alpha plane
    if _sky_drawn and _sky_plan is not None:
        LAST_TIMINGS['sky'] = int(_sky_plan['mode'])
    else:
        LAST_TIMINGS['sky'] = -1
    LAST_TIMINGS['sky_why'] = '' if _sky_drawn else str(_sky_why or '')
    LAST_TIMINGS['resident'] = bool(_keep_target)
    try:
        gbuf.gpu_sky = bool(_sky_drawn)
        gbuf.gpu_sky_why = '' if _sky_drawn else str(_sky_why or '')
        if _keep_target:
            from .frame import Resident as _Resident
            gbuf.gpu_frame = _Resident(target, w, h, 'shade')
            gbuf.gpu_frame_rgba = got
        else:
            gbuf.gpu_frame = None
            gbuf.gpu_frame_rgba = None
    except AttributeError:
        if _keep_target:
            target.free()
    return out, hit
