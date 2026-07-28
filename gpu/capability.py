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
    'code_node': (NOT_YET,
                  "the coded shader node is already GLSL, so this is the "
                  "easiest piece of the port rather than the hardest -- today "
                  "it compiles to NumPy, which on a GPU stops being necessary"),
    'node_graph': (NOT_YET,
                   "the hard one, and the reason a full GPU frame is still out "
                   "of reach: 29 of 106 node types now have a GLSL emitter, "
                   "each verified against the NumPy one. A material built only "
                   "from those can be emitted; anything else keeps the whole "
                   "material on the CPU rather than guessing"),
    'gbuffer_upload': (NOT_YET,
                      "the CPU G-buffer packs into textures and GLSL rebuilds "
                      "positions, normals and UVs from it exactly -- the route "
                      "to GPU shading that does not need a GPU rasteriser "
                      "first, since shading is 71% of a frame and rasterising "
                      "is 9%. The upload and the draw remain unwritten"),
    'shading_glsl': (NOT_YET,
                     "all 17 reflectance models are written in GLSL and match "
                     "the CPU exactly, and a whole material now assembles into "
                     "one shader that shades identically -- but nothing calls "
                     "it until the rasteriser exists"),
    'rasterise': (NOT_YET,
                  "the last piece. Draw the mesh into a G-buffer of triangle "
                  "IDs and barycentrics; there is a bit-identical CPU "
                  "reference to diff against, but the draw itself cannot be "
                  "checked without a GPU"),
    'shading_models': (NOT_YET,
                       "18 formulas already written down, mechanical to "
                       "translate"),
    'raytrace': (NOT_YET, "BVH traversal on the GPU is a project of its own"),
    'lens': (BOTH, "barrel distortion and chromatic aberration, agreeing "
                   "with the CPU path to 0.004 on hardware"),
    'composite_ntsc': (NOT_YET, "written, but blurs I and Q with one radius "
                                "where the CPU path uses two"),

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

#: features that force the whole frame onto the CPU when a scene uses them
BLOCKING = ('code_node', 'node_graph', 'rasterise', 'shading_models')


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
    # the frame is still a CPU frame; only the proven post stages move across
    return (GPU if stages else CPU), stages, notes


def summary():
    """Rows for the UI: (feature, support, reason), proven first."""
    order = {BOTH: 0, NOT_YET: 1, NEVER: 2}
    rows = [(k, v[0], v[1]) for k, v in FEATURES.items()]
    rows.sort(key=lambda r: (order[r[1]], r[0]))
    return rows
