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
    'display_transform': (BOTH, "exposure, the view-transform curve "
                          "(sRGB/Filmic/Reinhard), gamma, contrast and "
                          "saturation"),
    'ordered_dither': (BOTH, "Bayer and halftone matrices to a bit depth, "
                       "drawn from the CPU's own matrix and level table "
                       "(R251), and bit-depth quantisation"),
    # R251 (post-palette): the VALIDATION grades of PALETTE and CRY16
    'palette_snap': (BOTH, "the inverse-colormap palette snap and its "
                     "ordered dithers, drawn from the CPU's own 64^3 index "
                     "table; error diffusion and Blue Noise keep the CPU "
                     "by name"),
    'era_colour_roads': (BOTH, "CRY16, YJK, Extra Half-Brite and attribute "
                         "cells draw from CPU-built integer tables; "
                         "per-scanline palettes (a median cut per line) "
                         "stay on the CPU by name"),
    # R251 (post-palette, wave 2): the VALIDATION grades of SUPERBLACK
    # and LEGALISE
    'super_black': (BOTH, "the keying floor on geometry-covered pixels, "
                    "the coverage plane packed 64 pixels a texel"),
    'video_color_check': (BOTH, "the composite-envelope legaliser; "
                          "flag-black root-free, the scale modes within "
                          "1e-5 (one sqrt, one division)"),
    'crt': (BOTH, "phosphor mask, scanlines and vignette"),
    # R251 post-signal (SIG-1)
    'signal_era': (BOTH, "the era's scan-out stages (the GameCube copy "
                   "filter, the PS2 CRTC blend, the 3dfx scan-out filter): "
                   "integer arithmetic on the frame's own 8-bit values, hashes "
                   "for every 'random', LUTs for every root; measured bitwise "
                   "(0.0) the CPU in the simulator and through the fake "
                   "device; the field test reads the driver's number. The "
                   "3DO and GBA resamplers run after the final readback on "
                   "both roads"),
    'signal_codec': (BOTH, "the codecs the display showed: MPEG-1 intra "
                     "blocks (five draws at the padded size: the pack's "
                     "float32 DCT table, the quantiser's reciprocal "
                     "fetched from the CPU's table) and Smacker 4x4 "
                     "blocks (integers end to end over the frame palette, "
                     "one named readback when the palette lock is cold); "
                     "measured bitwise (0.0) the CPU in the simulator and "
                     "through the fake device; under a driver's FMA the "
                     "MPEG passes are graded CLOSE at 1/255 until the self "
                     "test prints its number"),
    'matte_glow': (BOTH, "the Tron printer's backlit mattes: the gel plane "
                   "uploaded once, two blur draws per diffusion pass and "
                   "one sum (7 samplers, under the guaranteed 16); "
                   "measured bitwise (0.0) the CPU in the simulator and "
                   "through the fake device, graded CLOSE at 5e-4 under a "
                   "driver's FMA; a diffusion radius past 160 taps, a "
                   "missing gel plane and the viewport refuse by name"),
    'n64_vi': (BOTH, "the N64 Video Interface's scan-out half: the dither "
               "filter (+-1 toward 8 neighbours on the 8-bit expansion) and "
               "the gamma / gamma-dither square-root LUTs; measured bitwise "
               "(0.0) the CPU in the simulator on 15- and 16-bit frames; a "
               "24-bit frame refuses by name on both roads. The coverage "
               "half (R251 C001): the RDP's 3-bit coverage of the winning "
               "polygon from the rasteriser and the VI's coverage blend and "
               "divot as two integer passes over the CPU's own coverage "
               "plane, measured bitwise (0.0) the CPU in the simulator; on "
               "the driver a marked pixel's coverage is the CPU's replay"),
    # R251 texture pack (TEX-2): the sample-time roads
    'texture_sample_roads': (BOTH,
                             "measured: GL_CLAMP border taps, the chroma key "
                             "after the filter, the mip level select (N64 "
                             "blend, nearest level, Voodoo dither), the PS2 "
                             "LOD from Q, the per-polygon level, N64 sharpen "
                             "and the summed-area box are emitted in the "
                             "CPU's statement order on CPU-decided LOD fields "
                             "(0.0 over 500 off-grid uvs at the sampler and "
                             "0.0 on the demo floor's own inputs; the level "
                             "blends carry 1.89.0's pre-existing 1-ulp "
                             "trilinear fraction, 5.96e-8); the frame sits "
                             "at the deferred lighting bar (< 6e-3)"),
    # R251 post-signal (SIG-2)
    'signal_tape': (BOTH, "the tape path (FIR + out draws), the S-Video and "
                    "RF cables, the PAL receiver and the digital formats' "
                    "chroma siting: integer FIRs summing to 65536, integer "
                    "matrices, hashes for every noise, a sine table for "
                    "every phase; measured bitwise (0.0) the CPU in the "
                    "simulator and through the fake device; RF over the "
                    "composite stage inherits that stage's 0.001 and sits "
                    "within 4/255 (measured, 2% of pixels); a low-pass wider than 96 px (a deep dub "
                    "or an S-Video Q band on a wide frame) refuses by name "
                    "and runs on the CPU"),
    # R251 texture pack (TEX-1)
    'texture_storage_laws': (BOTH,
                             "measured: the card's texel format, the N64 "
                             "TMEM budget and block compression (DXT1, NV2A, "
                             "CMPR, Dreamcast VQ) run once at prep and both "
                             "devices sample one array -- the frame sampler "
                             "twin is 0.0 on the demo floor's covered pixels; "
                             "the Voodoo 4/8-bit texel fraction and POV-Ray's "
                             "normalised-distance filter are emitted in the "
                             "CPU's statement order (0.0 over 500 off-grid "
                             "uvs, four wraps; the driver's `/` within 2.5 ULP "
                             "for the POV weights)"),

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
                  "frames run the tie referral; Painter's ordered fill reads "
                  "one depth per polygon here since 1.90.0; what it "
                  "cannot reproduce (the overdraw instrument, worker "
                  "bands) rasterises on the CPU and says why"),
    'shading_models': (BOTH,
                       "all 18 reflectance models dispatch in GLSL, "
                       "measured through the deferred pass at 0.000051 on "
                       "real hardware; CONSTANT and WIREFRAME emit "
                       "light_surface's shadeless early return, and the "
                       "Gouraud/flat rates interpolate CPU-lit corners"),
    # R251 (LIGHT-B2)
    'console_light_units': (BOTH,
                            "GX_LIGHT and DS_FIXED dispatch in GLSL bitwise "
                            "the CPU in the simulator (the axis viewer and "
                            "the DS shininess tables ride data textures; "
                            "the driver's normalize, one division and the "
                            "8-bit / 5-bit tie are the only moves); "
                            "SEGA_MODEL2 / SEGA_MODEL3 ARE their rates and "
                            "light on the CPU's corner road on both "
                            "devices; Brilliance and Metallic (POV) are "
                            "driver pow / acos within the deferred bar, "
                            "Crand is the integer hash, bitwise (measured d == 0.0 in the "
                            "simulator on both models; the Sega corner road within 7.2e-7)"),
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
    # R251 (RAST-B): the two wire roads draw on the CPU on either
    # device through the wire road's readback by name ('wireframe')
    # R251 shadow pack (SHADOW)
    'shadow_masks': (BOTH, "planar polygons (Blinn / Model 1), Dreamcast "
                     "modifier volumes and DS shadow polygons are decided once "
                     "on the CPU at the frame's own raster parameters and "
                     "applied as one float32 numpy rule over the finished "
                     "frame on both roads -- bitwise by construction, 0 px on "
                     "the demo rows (measured 0 px); the midpoint shadow map rides the same "
                     "atlas and compare as the classic map at its 6e-3 bar "
                     "(0.000006 in the simulator)"),
    # R251 transparency pack (TRANS-1)
    'blend_equation': (BOTH, "the blend unit's fixed equations and the "
                       "framebuffer formats run in the composite on the CPU on "
                       "both roads; every item that reads F's rgb (all but "
                       "ALPHA, SATURN_SHADOW, FUZZ) and every framebuffer "
                       "format refuses the GPU LAYER passes by name; the opaque "
                       "frame pass stays on the GPU; SATURN_SHADOW and FUZZ "
                       "keep the GPU layers (they read only a > 0, a baked "
                       "constant); measured: composite bitwise CPU-vs-CPU; "
                       "frame at the deferred bar (2.98e-6 on the fake "
                       "device); framebuffer roads 0.0. R251 wave 2: "
                       "THIN_WALL and IMAGINE_FOG read F and refuse the GPU "
                       "LAYER passes by name (the same print), measured "
                       "4.77e-6 / 2.98e-7 at the deferred bar; the N64 "
                       "RDP's random alpha compare (Screen Door N64_NOISE) "
                       "runs on both roads from the CPU's full-frame 8-bit "
                       "map uploaded under a (frame, seed) stamp, measured "
                       "alpha plane bitwise (0.0), rgb 5.96e-6; Clip+Blend "
                       "promotes its opaque half on the CPU and keeps a "
                       "LAYER pass only for a baked-constant alpha (a "
                       "per-pixel alpha refuses the layers by name), "
                       "measured alpha plane bitwise, rgb 4.77e-7; Z Offset "
                       "/ Z Invert are the CPU composite on both roads "
                       "(measured 0.0 by construction); an Env hole "
                       "material shades the frame on the CPU by name "
                       "(measured: the fake-device frame bitwise the CPU's)"),
    'vector_beam': (NOT_YET,
                    "the vector monitor beam (Atari DVG / AVG) and Elite's "
                    "wire rule are per-stroke scatters into an integer "
                    "accumulator / a DDA over an edge list; on the GPU "
                    "device the frame reads back by name ('wireframe') "
                    "and the same NumPy draws the same bits, so both "
                    "devices agree exactly. A GPU twin is a per-pixel "
                    "gather over binned strokes with the same pinned "
                    "table -- feasible, not written"),

    # --- construction roads (R251): whole render() calls per pass,
    # combined on the CPU over the read-back frames, so both devices
    # produce the same bytes by construction
    'lens_dof': (BOTH, "lens-sampled depth of field (REYES / the SGI "
                       "accumulation buffer / 3ds Max multi-pass): K whole "
                       "renders on whichever device, averaged on the CPU in "
                       "float64 in pass order; measured 0 differing pixels "
                       "between the GPU device and the CPU at 96x72 and the "
                       "fake device's frame bitwise the mean of its own 4 "
                       "passes (test_r251_sky_camera)"),
    'slice_blur': (BOTH, "Max Object Motion Blur's random slices and "
                         "LightWave's row-parity Dithered blur: an integer "
                         "selection and a fixed-order float64 sum over the "
                         "engine's read-back slices, where the plain mean "
                         "already runs; measured 0 differing pixels between "
                         "the engine's streamed combine and shutter_combine "
                         "on 4 slices, all 3 modes (test_r251_sky_camera)"),

    # --- genuinely impossible
    'error_diffusion': (NEVER,
                        "Floyd-Steinberg and its relatives are sequential by "
                        "construction. The diagonal wavefront helps on a CPU "
                        "but there is no GPU formulation that keeps the "
                        "result"),
    # R251 (MAT-A)
    'period_combiners': (BOTH,
                         "the fixed-function combiners (PS1 x/128, Saturn "
                         "add, N64 combiner, DS modulate, System 22, GS "
                         "HIGHLIGHT, D3D separate specular, PCX intensity, "
                         "the DS toon table, the Mega Drive S/H ramps, the "
                         "64-step luma ramp) run on the corner road: the "
                         "CPU lights and quantises the corners, the pass "
                         "interpolates and applies the machine's integer "
                         "rule in floats with floor / roundEven -- bitwise "
                         "the CPU at the function level in the simulator "
                         "(measured d == 0.0 on 4096 pairs per item), the "
                         "frame at the Gouraud seam (one 8-bit level at a "
                         "boundary pixel, driver only); the Super FX plot "
                         "selects CPU-chosen palette entries by parity, "
                         "bitwise (measured d == 0.0); the provoking-vertex "
                         "flat items are the FACE road itself"),
    'abuffer': (NEVER,
                "an unbounded per-pixel fragment list needs depth peeling or "
                "linked lists, which is a different algorithm rather than a "
                "port of this one"),
    # R253: the compositing passes
    'render_passes': (BOTH,
                      "Depth, Normal, Position, UV, IndexOB, IndexMA, Mist, "
                      "Env and Beauty come off the CPU-reconstructed G-buffer "
                      "and the frame itself on both roads (the GPU rasteriser "
                      "reconstructs the GBuffer, the shading burst only writes "
                      "the frame) -- measured 0.0 (bitwise) by construction in "
                      "the fake-device matrix rows"),
    'light_passes': (NOT_YET,
                     "the BI light split (Diffuse, Spec, Ambient, Emit, Shadow, "
                     "AO, Color, per-lamp Light00..07) reads light_surface's "
                     "own accumulators; a frame asking for any of them shades "
                     "on the CPU by name (plan_frame refuses), so both devices "
                     "draw the same bits; a GLSL twin is MRT, one target per "
                     "lobe -- feasible, not written"),
    # R251 material pack, wave 2 (MAT-B)
    'period_material_nodes': (BOTH,
                              "measured: the REYES micropolygon snap, the TEV / NV2A "
                              "combiner stage, the PowerVR2 (S,R) bump tables, the "
                              "DirectX 6 emboss and the Alias Env Chrome showroom "
                              "are bitwise (0.0) function twins in the simulator on "
                              "4096-lane inputs and sit on the deferred frame bar "
                              "(< 6e-3) as whole passes; Imagine Roughness is 0.0 in "
                              "the simulator and a 2-ULP normalize on a driver; the "
                              "GPU refuses by name what it cannot carry (affine or "
                              "hit / layer passes under REYES, a linked SR Bump Light, "
                              "a linked Emboss Texture Size, Roughness off the grid)"),
}

#: features that force the whole frame onto the CPU when a scene uses them.
#: code_node left this list when the deferred pass learned to inline it --
#: a coded-shader scene is no longer forced anywhere
BLOCKING = ('node_graph', 'rasterise', 'shading_models',
            # R253: any light-component pass puts the frame's shading on
            # the CPU (plan_frame refuses by name); the device panel and
            # plan()'s notes say 'Light Passes -- CPU for now'
            'light_passes')


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
    # R253: the passes a frame asks for
    from ..core.render import light_pass_names, wanted_passes
    if wanted_passes(settings):
        used.add('render_passes')
    if light_pass_names(settings):
        used.add('light_passes')
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
        # R251 LIGHT-B1
        notes.append('the R251 lamp roads -- the fixed camera-axis viewer, '
                     'the Only Shadow lamp, the OpenGL / POV-Ray / GX cone '
                     'laws, the GL / POV / GX decay laws and the Model 3 '
                     'screen spotlight -- run in-shader, bitwise the CPU in '
                     'the simulator (their values ride the hal_lights '
                     'texture, 8 texels per lamp; the driver\'s own sums of '
                     'products at 1 ULP, its pow at the deferred bar)')
        notes.append('deferred shading is on: measured at 0.000051 against '
                     'the CPU frame -- shadow maps, ray-traced shadows '
                     'hard and SOFT, ambient occlusion, ray reflections '
                     'and refraction at ANY depth, image textures, '
                     'converted master-shader materials, rim/fresnel/'
                     'sheen and area lights included; FOG runs in the '
                     'material pass (the four curves, per-vertex '
                     'quantisation, bands, height, the GTE 1/z cue, the '
                     'Voodoo / PowerVR2 / DS tables, z-buffer depth, the '
                     'GameCube column secants, per-polygon fog, the '
                     'material burn-through / bias / bank, the POV ground '
                     'integral and turbulence) and '
                     'is bitwise the CPU in the simulator; a BACKDROP fog '
                     'target uploads the CPU backdrop as one more '
                     'sampler (hal_backdrop, keyed by a crc); the ground '
                     'fog atan / sqrt / exp and the turbulence exp '
                     'are driver transcendentals (the field test reads '
                     'the number of the driver); under Ground Fog the sky '
                     'pass fogs by elevation (the closed form), bitwise '
                     'in the simulator; traced, '
                     'refracted and CPU-env materials keep the CPU fog '
                     'over the readback, by name; a fogged pass costs '
                     'one more sampler (hal_fogtab); TRILINEAR, '
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
