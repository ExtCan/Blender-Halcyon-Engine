# R253 -- cubemap-world (Cube Map world mode, CUBEMAP)

Branch `r253/cubemap-world`. The integrator owns CHANGELOG.md, README.md,
EXTENSION_LISTING.md, version.py, `__init__.py` and blender_manifest.toml;
the fragments below are written in their voice. Version stays 1.91.0 in
this worktree.

## CHANGELOG

- **What was asked.** A cube map world: the 1990s skybox, six square faces
  around the camera, loaded from one packed image (a cross or a strip) or
  from six files named the OpenGL or the Quake / Half-Life way, with the
  Blender Z-up mapping, a per-face turn / mirror, nearest or bilinear
  sampling INSIDE a face and never across a seam, the world's Rotation,
  Tint and Strength -- on both devices.
- **The mode.** `CUBEMAP` is appended at the END of `core/sky.MODES` and of
  `HalcyonWorldSettings.mode` (positional numbering: every saved file keeps
  its mode). `core/sky.py` carries the whole law in one `# R253 cube map`
  section: `CUBE_LAYOUTS` (horizontal cross 4x3 with +Y on top, -X +Z +X -Z
  across, -Y below; the HDRShop vertical cross 3x4 with -Z at the bottom
  turned 180 degrees; the 6:1 and 1:6 strips in GL face order),
  `cube_layout_auto` (4:3 / 3:4 / 6:1 / 1:6 by aspect), `CUBE_CONVENTIONS`
  (slot -> GL face, quarter turns, mirror: OpenGL is the identity; Quake 2
  is derived from ref_gl/gl_warp.c's suffix order {rt, bk, lf, ft, up, dn},
  its `st_to_vec` table and MakeSkyVec's `t = 1 - t` -- rt at +X, lf at +Y,
  up at +Z, dn at -Z, bk at -X, ft at -Y in Blender's Z-up frame, every
  Quake face mirrored against GL's because GL's faces are defined from
  outside the cube and Quake's are painted for the inside, up and dn turned),
  `cube_atlas` (the six faces as ONE (6S, S) float32 texture in GL face
  order, each moved into the canonical GL orientation by `np.rot90` /
  `np.fliplr` -- exact texel permutations, no arithmetic -- cached four deep
  by source identity, shape, prep tag and every dial; a hit re-checks the
  source arrays by identity so a recycled id never serves a stale atlas),
  `cube_face_uv` (OpenGL 4.6 core profile section 8.13, table 8.19, on the
  direction swapped to GL axes as gl = (x, z, -y): Blender +X -X +Y -Y +Z -Z
  land on GL faces 0 1 5 4 2 3; ties X over Y over Z), `cube_sample`
  (`Texture._sample_nearest` / `_sample_bilinear` with every tap clamped to
  the face's own [0, S-1] texels and offset by face * S -- the seam law:
  NEAREST shows the era's hard edge, BILINEAR a half-texel gutter, never a
  blend of two faces), `cubemap` (the texel times `env_tint`, exactly as
  `hdri()` tints; the solid colour without an atlas, hdri's rule) and
  `cube_set_from_path` (one face file -> the six paths and the convention:
  `rt..ft` with or without an underscore, `_px.._nz`, `posx..negz`,
  `_right.._back`, the file's own case kept). The single image rides the
  shared `env_image` slot exactly as the Cylinder Sky's does; the six-image
  mode uses six new `World.cube_image_*` slots, exported on the `env_image`
  road (Linear) through one `export._world_image` helper. `evaluate()`
  dispatches `elif mode == 'CUBEMAP'` after LW_GRADIENT: an environment
  like HDRI -- reflections and ray misses see the cube texel along any
  direction (unlike the Cylinder backdrop). The world's existing Rotation
  spins it (no second dial), Strength multiplies after the tint.
- **The GPU road** (`gpu/sky.py`): `MODE_CUBE = 9`, one PARAMS texel
  `hal_sky_cube = (S, filter, 0)` appended LAST, the atlas uploaded through
  the existing `hal_sky_env` sampler by `plan()`'s own env block (no new
  sampler, no new push constant -- the pinned interface holds). The GLSL
  `hal_sky_sample_cube` is the op-for-op transcription of `cube_face_uv` +
  `cube_sample`, dispatched in `main()` after the mode-8 branch and before
  the strength multiply; the ground-fog tail is untouched. Bitwise in the
  simulator at the function level (4006 directions, both filters) and on
  the frame for NEAREST under every rig and rotation; BILINEAR on the frame
  is within 4e-6 (the ray build's own ulps under the suite's NumPy, which
  the shipped HDRI / GRADIENT twins of `test_gpu_sky_pass` show at the
  same 10-13 pixels on the pristine tree; a wrong face or a crossed seam
  would be a whole texel). The fake device draws the SINGLE and the SIX
  frames' sky pixels bitwise the CPU device's. Refused by name, printed
  once through `gpu/sky._warn` at the console line, when 6 * S would
  exceed the driver's texture height (`CUBE_ATLAS_MAX_HEIGHT` 16384, the
  field's GL_MAX_TEXTURE_SIZE floor: faces above 2730 px stay on the CPU);
  a missing image folds both roads to the flat colour x strength (MODE_FLAT,
  bitwise) and a malformed one (a 5x3-cell cross, six faces of two sizes,
  a non-square face, an empty slot) prints ONE `[Halcyon] cube map: ...`
  line naming the size or the slot and draws the flat colour -- not a
  refusal, the frame still draws. `gpu/shade._env_world` names CUBEMAP
  among the rich modes: a reflective material under a skybox takes the
  exact CPU-composite road (`('CPU',)`). The console line names the mode
  (`_sky_names` 7 cylinder sky, 8 gradient backdrop, 9 cube map).
- **Pre-existing, fixed on the way (gpu/sky.py `plan`).** The GPU HDRI sky
  spun the OTHER way under a non-zero World Rotation since 1.89.0: the
  shader's `x*c - y*s, x*s + y*c` is `sky._rotate_z`'s text for the angle
  -rotation, but the params texel carried (cos(rot), sin(rot)); every
  uncovered GPU pixel of a rotated HDRI differed from the CPU's (GRADIENT /
  BANDS / LW_GRADIENT read only d.z, so the sign never showed; no direction
  mode had been twinned with a rotation). The texel now carries
  (cos(-rot), sin(-rot)) and the HDRI NEAREST twin under rotation 0.3 is
  bitwise (`test_cube_gpu_twin` pins it; it was 0.71 at every pixel).
- **The panel** (`ui.HALCYON_PT_world.draw`): Source (Single Image / Six
  Images); under Single the Image slot and the Layout (Auto by Aspect, the
  two crosses, the two strips); under Six a `Load Six Faces...` button
  (`HALCYON_OT_cube_load_six`, the sky-file browser pattern: pick ONE face
  file and the other five are found by suffix, the convention set from the
  family, the missing ones reported by name) and six Image slots labelled
  in the convention's own words (+X .. -Z or rt .. ft); then Convention,
  Filter, Tint, and a Face Orientation box with a quarter-turn and a
  Mirror per slot (`cube_face_rot` IntVectorProperty(6), `cube_face_flip`
  BoolVectorProperty(6) -- sizes that are now a file-format promise). Not
  in the show_ground list: an environment, like HDRI. Every new property
  carries a >= 40-char tooltip, every enum item a >= 12-char line.
- **Presets.** `presets/skies.EXCLUDED` gains every `cube_*` field (the six
  image slots foremost): a skybox belongs to its images, a sky file never
  carries an ImageBuffer, and applying a Bryce preset never clears a loaded
  skybox.
- **Nodes.** No node is added; `export.NODE_PROPS` is untouched (the node
  rules are vacuous for this feature).
- **Neutrality.** Every new World field defaults inert (`cube_source`
  SINGLE, `cube_layout` AUTO, `cube_convention` OPENGL, `cube_filter`
  NEAREST, zero turns, no mirrors, six empty slots); `hal_sky_cube` is the
  last PARAMS texel and no pre-existing mode sets it; an HDRI frame is
  bitwise `hdri()`'s own evaluation along `_background_image`'s rays; a
  GRADIENT / HDRI / SOLID frame with every cube dial changed and six slots
  filled is bitwise the default frame (`test_cube_neutrality`).
- **Stand-ins and open items.** The Quake 2 / Half-Life orientation table
  is derived from gl_warp.c and pinned by gradient corners headless; only a
  real `*rt/lf/up/dn/bk/ft` set in Blender proves it is not mirrored or
  turned on one face. An 8-bit TGA skybox decodes through `prepare_textures`
  as env_image does (Linear ImageBuffer), the same brightness HDRI mode gives
  such a file. A 2048-px face set is a 403 MB float32 atlas beside its
  sources. The field's nearest-texel cliffs: a ray an ulp off on the driver
  flips a whole texel, HDRI's stance -- counted by name, not a seam bug.
  Adjacent and out of scope: `_env_world` still sends a CYLINDER /
  LW_GRADIENT world with an env_image down the NODES fall-through (the
  image reflected as EQUIRECT where the CPU reflects the flat colour).
- **Tests.** `tests/test_r253_cubemap.py` (registered after the R252 block
  in `tests/run_all.py`): eleven tests, 140 checks -- the face rule and the
  function-level GLSL twin, the four layouts and AUTO, the six-image road
  under both conventions with the gl_warp.c corners, the per-face
  orientation and the cache, the seams (a 401-sample sweep across a seam
  reads only the two edge texels; BILINEAR clamps), rotation / tint /
  strength, the GPU twin (three rigs x two filters bitwise at rotation 0,
  three rotations, the SIX road, MODE_FLAT without an image, the height
  refusal sentence, the fake device on both roads, the content key), the
  reflections, the wiring under fakebpy, the neutrality of every
  default, and the console-shader driver-strictness sweep of the deep
  re-read below (151 combos). `tests/test_render.py::test_sky_modes` excludes CUBEMAP beside
  HDRI (no image -> the solid colour by design; `- 3`).

## README

**Twelve sky modes** -- the eleven of 1.90 and now (1.92) the **Cube Map**,
the 1990s skybox: six square faces around the camera from one packed image
(a horizontal or vertical cross, a 6:1 or 1:6 strip, read by aspect) or
from six files named the OpenGL way (+X -X +Y -Y +Z -Z) or the Quake /
Half-Life way (rt lf up dn bk ft -- pick one file and the other five load
by suffix), point-sampled for the hard seam of the era or bilinear inside
each face and never across one, with a quarter-turn and a mirror per face
for a set that loads sideways, the world's Rotation, Tint and Strength.
It is an environment like the HDRI: reflections see it, and it draws in
the GPU sky pass bitwise the CPU's.

## Blender-only

- The six template_ID slots, their convention labels and the Face
  Orientation rows drawing in the World panel (fakebpy has no layout).
- `HALCYON_OT_cube_load_six`'s file browser, `bpy.data.images.load(path,
  check_existing=True)` of the six files, and that the loaded images reach
  the exporter with pixels (compat.image_pixels).
- A real Quake 2 / Half-Life skybox set rendering the right way round (the
  gl_warp.c-derived orientation table: every face mirrored, up / dn turned).
- The brightness of an 8-bit skybox under Blender's colour management
  (ImageBuffer colorspace 'Linear' like env_image).
- The real driver's division / floor roundings at nearest-texel cliffs, the
  upload of a 6S-tall atlas near GL_MAX_TEXTURE_SIZE on the field's cards,
  and the VRAM of a 2048-face skybox.
- The console lines `[Halcyon GPU] sky: cube map drawn in the shading burst`,
  the once-printed `[Halcyon] cube map: ...` layout / slot messages and the
  height-limit refusal appearing in Blender's system console.
- The GPU HDRI sky under a non-zero Rotation now matching the CPU's on a
  real driver (the pre-existing sign fix).

## Deep re-read: the Console Emulation Shader road (the 1.91.0 crash)

Asked for alongside the feature: a deep re-read of the new game-emulation
shader code after everything, because the field crashed on changing the
texture op to Modulate 4x in the node's PC fixed-function mode. Read: the
tables and the resolver (`core/console.py`), every GLSL combine text and
the recombine lines (`gpu/combine.py`), the CPU combines (`core/combine.py`
`cb_mod8_k` / `cb_add8` / `cb_addsigned8` / `cb_tev` / `cb_decal8` against
their GLSL twins, op for op), the node (`nodes/shader_nodes.py`), the
albedo emitter against `nodeeval.console_albedo` (vertex-colour source
rule, the dual-texture blends), the per-light finish lines (the GX
attenuation gate on both roads, the DS table row), the light-limit /
fixed-shade gates in `gpu/shade.py`, and the device's compile path.

- **Instrument.** An exhaustive sweep (scratch, 654 combos: every machine's
  defaults, every option of every machine one at a time, light limit, the
  Rate menu, every PC fixed-function shading x texture op, local viewer off
  with colour vertex -- each at the PIXEL, VERTEX and FACE rates) through
  the GPU plan on the demo scene, every assembled pass source checked for
  the whole crash class (a function defined twice, an HLSL-ism, a
  reserved-word identifier, a declaration left for CreateInfo to redeclare,
  an int / int division) and the frame held to the Gouraud-seam bar.
  Result: 0 problems in 630 planned combos; the 24 refusals are by name and
  the CPU's own (Saturn MESH: opacity on the deferred target; Super FX under
  the scene's ADAPTIVE palette). A compact form (the PC grid, every
  machine at two rates, every option once) is
  `test_zz_console_shader_driver_strictness` in the R253 module.
- **Found and changed (`gpu/device._compile_dynamic_miss`, both the dynamic
  and the static site).** After a CreateInfo refusal the code went on to the
  legacy `gpu.types.GPUShader(VERTEX, fragment)` constructor with the SAME
  text -- the field log reads "CreateInfo failed: Shader Compile Error ...;
  legacy constructor also failed: cannot create 'GPUShader' instances" and
  the application went down right after. That is a second GPU object asked
  of a source the driver just refused, the double hand-over R193's rule
  forbids; a CreateInfo refusal now returns at once (the legacy constructor
  remains for a Blender without CreateInfo). Whether the second attempt was
  the fatal step cannot be proved headless (Blender-only); the duplicate
  guard already keeps the first refusal from happening.
- **Pre-existing, noted, not changed.** The crash session's earlier access
  violation inside the preview thread's `_background_image` (the
  `world_color` write, right after a Super FX refusal) is the OPEN item the
  1.91.0 entry names: `compat.image_pixels` reads through `foreach_get` into
  a fresh NumPy buffer (no Blender-owned memory reaches the renderer), and
  nothing in the console road touches that write. Not reproduced headless.
