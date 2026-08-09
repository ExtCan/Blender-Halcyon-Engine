"""Which features can run on which device, and why.

Cycles has the same problem and solves it the same way: some features simply do
not exist on some devices, so the device is a choice and the UI says what that
choice costs. Open Shading Language was CPU-only in Cycles for years for exactly
this shape of reason.

The table below distinguishes two things that are easy to conflate:

    NEVER    the algorithm cannot run on a GPU at all
    NOT_YET  it could, nobody has written it
    BOTH     ported, and measured against the CPU path on real hardware

That distinction is the point. "Error diffusion is CPU-only" and "the node
evaluator is CPU-only" are true for completely different reasons, and only one
of them will ever change. Collapsing them into one flag would hide the roadmap.
"""

CPU = 'CPU'
GPU = 'GPU'

BOTH = 'BOTH'
NOT_YET = 'NOT_YET'
NEVER = 'NEVER'

#: feature -> (support, one-line reason)
FEATURES = {
    # --- proven on hardware
    'display_transform': (BOTH, "exposure, gamma, contrast and saturation"),
    'ordered_dither': (BOTH, "Bayer patterns and bit-depth quantisation"),
    'crt': (BOTH, "phosphor mask, scanlines and vignette"),

    # --- portable, simply not written yet
    'code_node': (BOTH,
                  "the coded shader node is already GLSL, so this was the "
                  "easiest piece of the port rather than the hardest -- the "
                  "deferred pass inlines it natively, mangled per node, "
                  "sockets baked or chain-fed, the clock as a per-frame "
                  "uniform. measured through the deferred pass on real "
                  "hardware at 0.000021 (RTX 5060 Ti, Vulkan); image "
                  "inputs, vScreenUV and iResolution travel now, and HLSL "
                  "still refuses"),
    'node_graph': (BOTH,
                   "78 node types carry a GLSL emitter, each measured "
                   "against the NumPy evaluator to 0.000006 in simulation "
                   "and 115 of 116 matrix rows on real hardware. The "
                   "evaluator was the hard piece of the port, and the "
                   "emitters retired it -- Normal Map bends the "
                   "shading normal, Bump renders its height chain to a "
                   "pre-pass and takes the CPU's own neighbour differences "
                   "by texelFetch, coded shaders inline natively, the "
                   "Wireframe node draws exact edge distance, and ALL "
                   "nineteen period pattern textures ride their integer "
                   "hash bit-exactly. What refuses does so PER MATERIAL, "
                   "by name (Blender's sin-fract Noise family above all: "
                   "a driver's float32 sin decorrelates it), and that one "
                   "material shades on the CPU exactly -- a node graph no "
                   "longer moves the frame"),
    'gbuffer_upload': (BOTH,
                      "measured: the deferred pass reproduces the CPU frame "
                      "to 0.000051 max difference on real hardware (RTX 5060 "
                      "Ti, Vulkan). Shadow maps ride along as atlases, image "
                      "textures as their prepared pixels with the filter "
                      "arithmetic in the shader, and unchanged uploads are "
                      "cached across frames behind content fingerprints"),
    'shading_glsl': (BOTH,
                     "all 18 reflectance models, measured through the "
                     "deferred pass on real hardware at 0.000051 -- sun, "
                     "point and spot lights, two-sided lighting, flat and "
                     "smooth normals"),
    'rasterise': (BOTH,
                  "the last piece, built as a COMPUTE rasteriser -- the "
                  "CPU's own fill rules, one thread per pixel over binned "
                  "tiles, because hardware rasterisation could never match "
                  "this renderer at triangle edges. measured: 0 differing "
                  "pixels of 76800 on real hardware (RTX 5060 Ti, Vulkan), "
                  "barycentrics to 3e-7, and 7x faster than the CPU at "
                  "working size -- the kernel IS fill(). Opt in as GPU "
                  "Rasteriser in the Debug panel; affine frames carry "
                  "their screen-linear barycentrics and quantised-depth "
                  "frames run the tie referral; what it cannot reproduce "
                  "(Painter's ordered fill, the overdraw instrument, "
                  "worker bands) rasterises on the CPU and says why"),
    'shading_models': (BOTH,
                       "all 18 reflectance models dispatch in GLSL, "
                       "measured through the deferred pass at 0.000051 on "
                       "real hardware; CONSTANT and WIREFRAME emit "
                       "light_surface's shadeless early return, and the "
                       "Gouraud/flat rates interpolate CPU-lit corners"),
    'raytrace': (BOTH,
                 "COMPLETE, measured on real hardware (RTX 5060 Ti, "
                 "Vulkan): hard ray shadows (0.000048, 0 px), SOFT ray "
                 "shadows and ambient occlusion (0.000047, 0 px -- "
                 "hash-jittered identically on both devices), one "
                 "traced bounce (0.000048, 0 px), refraction through "
                 "bent noise-and-bump normals (0.000028, 0 px), and "
                 "the full recursion tree at ray depth beyond 1 -- "
                 "mirror-in-mirror at 0.000024, 0 px of 172800. Hits "
                 "spawn their own rays and composite backward with the "
                 "hit material's constants, exactly as trace() "
                 "recurses. Rays whose two nearest surfaces tie within "
                 "float noise (coincident contact geometry -- a box "
                 "resting on a floor) are flagged by the kernel and "
                 "re-resolved on the CPU intersector, so the driver's "
                 "last-bit rounding can never pick a different surface "
                 "than the reference"),
    'lens': (BOTH, "barrel distortion and chromatic aberration, agreeing "
                   "with the CPU path to 0.004 on hardware"),
    'composite_ntsc': (BOTH, "three blur draws and a combine, exactly the "
                             "CPU's triple-box shape; 0.00037 measured on "
                             "hardware. Dot crawl is frame-dependent and "
                             "keeps a frame using it on the CPU"),

    # --- genuinely impossible
    'error_diffusion': (NEVER,
                        "Floyd-Steinberg and its relatives are sequential by "
                        "construction. The diagonal wavefront helps on a CPU "
                        "but there is no GPU formulation that keeps the "
                        "result"),
    'abuffer': (NEVER,
                "an unbounded per-pixel fragment list needs depth peeling or "
                "linked lists, which is a different algorithm rather than a "
                "port of this one"),
}

#: features that force the whole frame onto the CPU when a scene uses them.
#: code_node left this list when the deferred pass learned to inline it --
#: a coded-shader scene is no longer forced anywhere
BLOCKING = ('node_graph', 'rasterise', 'shading_models')


def supports(feature, device):
    support, _why = FEATURES.get(feature, (NOT_YET, "unknown feature"))
    if device == CPU:
        return True
    return support == BOTH


def reason(feature):
    return FEATURES.get(feature, (NOT_YET, "unknown feature"))[1]


def material_shader(mat, light_count=0):
    """The complete GLSL for one material, or (None, why)."""
    from .material import assemble
    graph = getattr(mat, 'graph', None)
    if not graph:
        return None, 'no node graph'
    return assemble(graph, light_count=light_count)


def material_can_emit(mat):
    """(ok, missing) for one material's graph, without generating code."""
    from .emit import can_emit
    graph = getattr(mat, 'graph', None)
    if not graph:
        return True, set()          # no graph is trivially emittable
    return can_emit(graph)


def emittable_materials(scene):
    """How many of a scene's materials could be emitted as GLSL today."""
    mats = list(getattr(scene, 'materials', ()) or ())
    ok = 0
    missing = set()
    for mat in mats:
        good, miss = material_can_emit(mat)
        if good:
            ok += 1
        else:
            missing |= set(miss)
    return ok, len(mats), missing


def scene_features(scene, settings):
    """Which relevant features a given scene and settings actually use."""
    used = set()
    for mat in getattr(scene, 'materials', ()) or ():
        if getattr(mat, 'programs', None):
            used.add('code_node')
        if getattr(mat, 'graph', None):
            used.add('node_graph')
    used.add('rasterise')
    used.add('shading_models')
    if getattr(settings, 'raytrace', False):
        used.add('raytrace')
    if getattr(settings, 'transparency', 'NONE') in ('SORTED', 'ABUFFER'):
        used.add('abuffer')
    if str(getattr(settings, 'dither', 'NONE')) in (
            'FLOYD', 'JJN', 'STUCKI', 'ATKINSON', 'BURKES', 'SIERRA',
            'SIERRA_LITE'):
        used.add('error_diffusion')
    if getattr(settings, 'crt', False):
        used.add('crt')
    if getattr(settings, 'composite', False):
        used.add('composite_ntsc')
    return used


def plan(scene, settings):
    """What the requested device can actually deliver for this scene.

    Returns (effective_device, gpu_stages, notes). The engine never refuses --
    an unsupported feature moves that work to the CPU and says so, because a
    render that is slower than hoped beats a render that does not happen.
    """
    requested = str(getattr(settings, 'render_device', CPU)).upper()
    used = scene_features(scene, settings)
    notes = []

    if requested != GPU:
        return CPU, (), notes

    from . import device as dev
    ok, why = dev.probe()
    if not ok:
        return CPU, (), [f'no GPU available: {why}']

    blocked = sorted(f for f in used if f in BLOCKING
                     and FEATURES[f][0] != BOTH)
    for f in blocked:
        notes.append(f'{f} runs on the CPU: {reason(f)}')

    stages = tuple(f for f in ('display_transform', 'ordered_dither', 'crt',
                               'lens')
                   if FEATURES[f][0] == BOTH)
    if getattr(settings, 'gpu_shading', False):
        notes.append('deferred shading is on: measured at 0.000051 against '
                     'the CPU frame -- shadow maps, ray-traced shadows '
                     'hard and SOFT, ambient occlusion, ray reflections '
                     'and refraction at ANY depth, image textures, '
                     'converted master-shader materials, rim/fresnel/'
                     'sheen and area lights included; FOG rides the '
                     'readback (the CPU\'s own apply_fog over the '
                     'driver\'s pixels -- all four modes, per-vertex '
                     'quantisation and the height layer, fogged at each '
                     'point\'s own depth including ray hits); TRILINEAR, '
                     'mip bias, anisotropy and the N64 3-point filter '
                     'sample from the CPU\'s own mip atlases with its '
                     'own footprint field (glass layers keep the '
                     'footprint on the CPU for now, by name); anything '
                     'else outside its scope shades on the CPU and says '
                     'why -- node chains may drive the '
                     'surface parameters per pixel, a Normal Map chain '
                     'on the master shader bends the normal itself, '
                     'coded shader nodes run their GLSL natively, the '
                     'matcap and backface overrides ride along, and '
                     'environment reflections travel for EVERY world -- '
                     'the simple modes as baked GLSL, and the rich ones '
                     '(STARFIELD, BRYCE, PHYSICAL, HDRI, world graphs, '
                     'the ground plane) evaluated by the renderer '
                     'itself along the reflected rays')
    else:
        notes.append('GPU Shading is OFF (Debug panel): frames shade on '
                     'the CPU. Flip the device switch again, or tick GPU '
                     'Shading, to shade on your driver')
    # rasterisation stays a CPU job; shading and the proven post stages move
    return (GPU if stages else CPU), stages, notes


def summary():
    """Rows for the UI: (feature, support, reason), proven first."""
    order = {BOTH: 0, NOT_YET: 1, NEVER: 2}
    rows = [(k, v[0], v[1]) for k, v in FEATURES.items()]
    rows.sort(key=lambda r: (order[r[1]], r[0]))
    return rows
