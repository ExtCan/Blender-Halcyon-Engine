# R254 -- ray-road (traced shadows and the shadow controls around them)

Work package `ray-road` of the light audit. The integrator owns CHANGELOG.md,
README.md, version.py, `__init__.py`, blender_manifest.toml and
tests/run_all.py; nothing here touches them. What they need from this
package is listed under "For the integrator" at the end.

The user's report: "Fix the lights so that changing the size of the lights
actually works. The lights don't get softer/harder with size. Also, I think
some of the general options just don't work properly." This package is the
traced half: the `size-ray` audit cluster (twelve findings and the verifier's
new one) and the traced / panel findings of `hal-shadow`
(perlight-ray-bvh-01, samples-dead-01, density-ray-01, menu-ui-01,
adopt-partial-01, rayshadows-greyed-01, planarz-hidden-01 and the verifier's
lamp-dials-greying-01).

Roads checked for every fix: the CPU road (core/render.py, headless and real
Blender 5.2.2 F12 through the add-on), the GPU road through its simulator
twin (gpu/shade.plan_frame + simulate, and for the accumulation road the
fake device, whose draws run the driver's own uniform plumbing; the machine
has no real GPU driver, so nothing here was drawn by one), and the rendered
viewport's worker driven headless (preview.Viewport) for the two the user
would see first. "Before" numbers are from a frozen copy of commit 87489e1
(before any light fix) unless a line says "fork point" (7e9a718, which
already carried sections 1 and 2's engine halves).

The rig of most numbers is the audit's: a floor, a plate whose straight edge
shadows it and one lamp, seen from straight above, so frame / unshadowed
frame is the visibility and a penumbra is a profile along x. "Grain" is the
RMS of that visibility against a 20000-sample Monte-Carlo of the lamp's own
geometry, inside the penumbra.

## 1. A lamp on Ray Traced under Per Light casts no shadow

- **What the user saw.** Shadow Method = Per Light, the lamp's own Shadows
  menu on Ray Traced: no shadow at all, unless Ray Tracing or Ambient
  Occlusion happened to be on.
- **What was wrong.** `core/render.py` built the BVH for the render-wide Ray
  Traced method but not for a Per Light lamp that traces;
  `core/lights.visibility` then found no BVH and returned fully lit. Fixed on
  the branch this package forked from (`render._per_light_ray_shadows`).
- **This package.** The tests (section A of tests/test_r254_ray_road.py),
  and one more clause: under Shadow Method Ray Traced with Ray Shadows OFF no
  BVH is built either (`visibility` returns before it reads one; the frame is
  bitwise the frame with a BVH in reach).
- **Pictures that change.** None beyond the fix already on the branch.
- **Proof.** POINT / SPOT / SUN / AREA: umbra darkness 0.000 -> 1.000, the
  frame bitwise the render-wide Ray Traced frame, GLSL twin within 6.0e-4. A
  mapped and a traced lamp under Per Light both cast (umbra vis 0.427 ->
  0.000). No BVH is built for: render Shadow Maps; Per Light with only mapped
  lamps; Per Light + lamp Ray Traced with Ray Shadows or Shadows off; a hemi
  lamp; render Ray Traced with Ray Shadows off. Real Blender F12: Per Light +
  lamp Ray Traced frame mean 0.35678 (= the lamp-None frame) -> 0.34248; the
  rendered viewport's worker: umbra 0.000 -> 1.000 in draft and refine.

## 2. The lamp's own Samples dial is read by nothing

- **What the user saw.** The lamp's Samples slider moved no pixel, mapped or
  traced.
- **What was wrong.** Exported and never read. Fixed on the branch
  (`lights.shadow_sample_count`: the lamp's value, 0 = the render's).
- **This package.** The tests (section B), and the two places the engine fix
  left behind: `halcyon.adopt_shadow_settings` ("All Lights Use These
  Settings") now also resets a lamp's Samples to 0 and its Map Depth to
  Inherit (adopt-partial-01), and the lamp panel's "Overrides the render
  setting" note counts both.
- **Pictures that change.** None from this package. Note for the changelog
  (the verifier's): a lamp whose Samples slider was ever touched in an
  existing file now really uses the stored value.
- **Proof.** Traced and mapped: lamp 16 under render 4 is render 16 bitwise;
  a lamp at the render's own count is the inherited frame bitwise; GLSL twin
  0.000000. Real Blender F12: lamp Samples 16 takes 1.60 s against 0.46 s
  (it was 0.18 s either way). The operator: three of four fake lamps carried
  an override (Samples 32; Map Depth Midpoint; Map Size 512 + Bias + Samples
  8) and all are reset; in real Blender `bpy.ops.halcyon.
  adopt_shadow_settings(scope='SCENE')` leaves samples 0, map_depth INHERIT,
  map_size 0 and keeps softness 3.0.

## 3. Traced soft shadows are white noise at the default 4 rays

- **What the user saw.** A sized lamp's traced penumbra is salt and pepper:
  five grey levels scattered across it.
- **What was wrong.** `core/lights.visibility` (and its GLSL twin in
  `gpu/material._shadow_function`) drew every ray's two uniforms as
  independent hashes of (pixel, ray index, light, seed). Nothing stratified
  the N rays of a pixel, so the error was sqrt(p(1 - p) / N).
- **What it does now.** One stratified pattern per pixel, the same
  arithmetic on both roads: two hashes for the pixel (a radial shift `h1` and
  a turn `h2`), and ray k of N takes `u1 = (k + h1) / N` (N equal-area rings
  with one shared shift) and the turn `turn_k + h2`, where `turn_k` is the
  base-2 radical inverse of k when N is a power of two (the rays are then a
  (0,2)-net) and k golden fractions otherwise
  (`lights.soft_ray_turns`). The disc shapes (POINT, SPOT, SUN, Area DISK /
  ELLIPSE) read the turn as an integer step through the shared 256-entry
  circle table; the rectangle reads it as its second coordinate. The audit
  proposed golden fractions for every count; measured on half-plane occluders
  over twelve orientations, the radical inverse is better at the counts users
  pick (disc, RMS: 4 rays 0.114 against 0.136, 8 rays 0.056 against 0.066)
  and golden fractions at the others (5 rays 0.100 against 0.116), so the
  rule takes each where it wins. The frozen-to-the-screen grain is the Seed
  tooltip's promise and stays.
- **Pictures that change.** Every traced shadow of a lamp with a size, at
  Samples 2 or more. A size-0 lamp (and any lamp at Samples 1) never reaches
  the sampler and renders bitwise.
- **Proof.** Rig, 240x96, 4 rays, grain before -> after: POINT 0.200 ->
  0.114, SPOT 0.200 -> 0.114, SUN 0.196 -> 0.117, Area SQUARE 0.207 -> 0.105,
  Area DISK 0.200 -> 0.108. Grey levels: 5 before and 5 after (four binary
  rays cannot give more); what changes is how the penumbra uses them -- the
  share of its pixels more than one level off the truth falls from 21.1% to
  1.2% (POINT, 160x64). By count (POINT): 2 / 4 / 5 / 8 / 16 rays 0.283 /
  0.193 / 0.177 / 0.141 / 0.102 -> 0.204 / 0.109 / 0.099 / 0.064 / 0.040.
  Bias (worst column mean against the truth) 0.089 -> 0.027. 10-90 width
  against the truth: POINT 0.391 / 0.385, SUN 0.341 / 0.346, Area 0.458 /
  0.448. Every pixel puts exactly one ray in each ring (99.9% of pixels; the
  rest sit on a ring boundary). Cost: 97.4 -> 99.5 ms at 4 rays, 241.5 ->
  233.6 ms at 16 (two hashes a pixel instead of two a ray). CPU against GLSL
  twin: 0.000000 to 0.000001 for the five shapes at 4 rays and for 2, 5, 8
  and 16 rays. Deterministic: same seed same picture, another seed moves it,
  1 and 3 threads identical; four bands stitched and two worker processes
  give the whole frame's bytes, under Accumulate too. Real Blender F12
  (blui/spec_*_RAY.json, 320x240):
  size 0 moved 0 pixels for POINT, SPOT and SUN; high-frequency grain of the
  penumbra pixels (RMS of luminance minus its four neighbours) POINT 0.0906
  -> 0.0502, 0.0812 -> 0.0426, 0.0647 -> 0.0372; SPOT 0.1327 -> 0.0737,
  0.1191 -> 0.0624, 0.0945 -> 0.0539; SUN 0.1598 -> 0.1095, 0.1575 ->
  0.0897, 0.1607 -> 0.0874. Viewport worker: grain 0.201 / 0.200 -> 0.115 /
  0.114 (draft / refine).

## 4. Density does nothing to a traced shadow

- **What the user saw.** Shadow Density lightened mapped and planar shadows
  and left a traced one black.
- **What was wrong.** The RAY branch of `core/lights.visibility` returned
  before the density line, and the GLSL mirrored it on purpose.
- **What it does now.** The traced visibility goes through the map tail's
  own expression, `1 - (1 - v) * density`, on both roads -- skipped at the
  default 1.0, because `1 - (1 - v)` is not bitwise `v` in float32. The ray
  meta carries the density and `gpu/shade._light_sig` carries it for every
  lamp (the return line is baked).
- **Pictures that change.** Traced shadows of lamps with Density below 1.
- **Proof.** POINT / SPOT / SUN / AREA, hard and soft: umbra vis at Density
  0.4 0.000 -> 0.600; GLSL twin within 8e-7. The same Density gives the same
  umbra traced and mapped (0.7500 / 0.7500 at 0.25). Density 0 is the
  unshadowed frame. At 1.0 `visibility` returns exactly 0.0 / 1.0 and the
  emitted function is the old text to the character. Real Blender F12:
  traced frame mean 0.34248, Density 0.4 0.35107 (it was 0.35678 = no shadow
  for both, the lamp being Per Light + Ray Traced).

## 5. Scaling an Area lamp makes it brighter, not bigger

- **What the user saw.** S on an Area lamp in the viewport: the light gets
  much brighter and the shadow does not soften.
- **What was wrong.** `export.export_light` exported the Size alone and the
  raw matrix columns as the lamp's axes. The lit contour
  (`lights.area_corners`) multiplied by those unnormalised axes while the
  emitting area and the shadow rays read the unscaled Size: energy x scale
  squared, penumbra unchanged.
- **What it does now.** A scaled lamp exports Size x scale on unit axes
  (EEVEE's `area.size *= scale.xy`); a non-uniform scale makes a rectangle or
  an ellipse. An unscaled lamp (columns within 1e-5 of unit) takes the old
  lines untouched. The optional guard inside `area_corners` was left out, as
  the verifier advised (the GLSL builds its own rectangle from the axes).
- **Pictures that change.** Area lamps whose object is scaled.
- **Proof.** Size 0.25 scaled x4 against Size 1.0: lit floor x16.000 ->
  x1.000, traced 10-90 width x0.250 -> x1.000, the two frames 21.1 apart ->
  3.6e-7; GLSL twin 1e-6. Size 0.5 scaled (4, 1, 1) is the 2.0 x 0.5
  rectangle (frames 4.15 apart -> 0.0). Real Blender F12
  (audit spec_area_scale.json): frame mean 0.18497 -> 0.04425 against 0.04424
  for Size 1.0; the two frames 0.717 apart -> 0.00003.

## 6. An Area lamp of Size 0 lights the scene with noise

- **What the user saw.** Size 0 (Blender lets it be typed): the floor is
  white noise, and the GPU and CPU pictures disagree.
- **What was wrong.** `lights.area_emit_area` and `area_corners` clamped the
  sides at 1e-6, so the Stokes contour summed rounding noise and multiplied
  it by dist^2 / 1e-12. The GLSL rectangle (`gpu/material._area_function`)
  clamped at its own 1e-6 and broke fifty times sooner in float32.
- **What it does now.** One constant, `lights.AREA_MIN_SIDE = 0.02`, is the
  shortest side an area lamp LIGHTS with, read by all three sites through
  `lights.area_lit_sides`. The far-field energy does not depend on the size
  (BI's area lamp is total-power-like), so the clamped face is the small
  lamp, drawn cleanly. The traced shadow still reads the true Size: a Size 0
  lamp casts a hard shadow. The audit's fix named two sites; the verifier
  found the third (the GLSL rectangle), and with only two raised a 0.01 lamp
  would have gone from 0.9% to 75% off.
- **Pictures that change.** Area lamps with a side under 0.02. Lamps of 0.02
  or more build the same face bitwise.
- **Proof.** Lit floor mean, relative noise, worst relative |GLSL - CPU|:
  SQUARE 0: 26.75 / 1.17 / 6.4e7 -> 0.3416 / 0.1294 / 0.0023 (the 0.25 lamp:
  0.3415 / 0.1293 / 0.0000); SQUARE 1e-5: 3.10 / 1.64 -> 0.3416 / 0.1294;
  SQUARE 0.001: GLSL 84% off -> 0.23%; SQUARE 0.005: 3.3% -> 0.23%; DISK 0:
  31.57 -> 0.3416 (GLSL 0.66%); RECTANGLE 1 x 0: GLSL 302% off -> 0.00%.
  Real Blender F12: Size 0 frame mean 0.36878 -> 0.04400 (the 0.25 lamp:
  0.04400), DISK 0 0.40805 -> 0.04400.

## 7. A very wide sun crowds its rays at the rim

- **What the user saw.** A sun with an Angle near 180 degrees shades like a
  ring at the horizon.
- **What was wrong.** The rays were drawn uniform in a tangent-plane disc of
  radius tan(angle / 2), which is uniform in solid angle only for small
  angles; at pi the tangent is 1.6e16.
- **What it does now.** `lights.sun_soft_law`: the angle is clamped to 179.9
  degrees (EEVEE's ceiling), and above 0.5 rad the rays are uniform in the
  solid angle of the sun's cap (`cos t = 1 - u1 (1 - cos(a / 2))`), in the
  CPU sampler and in the GLSL (which bakes 1 - cos(a / 2) in place of the
  tangent). Up to 0.5 rad the disc lines stand.
- **Pictures that change.** Traced shadows of suns wider than 0.5 rad.
- **Proof.** Share of the rays inside half the angular radius (a uniform cap
  in brackets): 0.6 rad 0.240 -> 0.251 (0.251); 1.0 rad 0.220 -> 0.254
  (0.254); 2.0 rad 0.123 -> 0.266 (0.266); 3.0 rad 0.004 -> 0.289 (0.289);
  pi 0.000 -> 0.293 (0.293), widest tilt 1.5708 -> 1.5699 rad. A 0.2 rad sun
  fires `normalize(L + disc jitter)` on the stratified draws, bitwise against
  the formula written out in the test. GLSL twin 0.000000 at 0.2, 2.0 and pi.
  Real Blender F12: Angle pi frame mean 0.26877 -> 0.42176.

## 8. A large Seed makes the GPU refuse the frame

- **What the user saw.** With Seed above 542357 and any sized traced lamp
  (or occlusion, or radiosity) the driver rejects the material pass and the
  frame shades on the CPU.
- **What was wrong.** `gpu/material.py` baked the salt `2k + 131 i + 7919
  seed` as a decimal integer literal into the soft-ray, occlusion and
  radiosity functions; past 2**32 - 1 no GLSL compiler accepts it (and past
  2**31 - 1 it is no int). The simulator twin accepts anything, so no
  headless test could see it. The CPU road raised a NumPy overflow warning at
  the same seeds.
- **What it does now.** `patterns.wrap_salt` keeps the low 31 bits, and
  `patterns.sample_u` wraps its salt itself. hash3 reads only those bits
  (`iz = q 2**31 + r` adds `r C mod 2**31`), so every picture is the same
  bit for bit. The three GLSL sites emit the wrapped number.
- **Pictures that change.** None.
- **Proof.** Largest integer literal in the planned passes: seed 271188
  2147546168 -> 104729, seed 600000 4751408396 -> 456441100, seed 2**30
  8502961512652 -> 1073750220 (soft rays + occlusion), 1073751804 with the
  full-rate radiosity gather. `sample_u` on a wrapped salt equals an
  exact-integer hash of the bare salt for six salts up to 8.5e12, with
  warnings raised as errors. Occlusion at seed 600000, full-rate radiosity at
  seed 2**30 and blurred reflections at seed 600000 render bitwise the old
  tree. The strict glslang front-end compiles 28 soft-ray sources (five
  shapes, the cap law, Density, seeds 0 and 2**30, 4 and 5 rays).
- **Not covered here.** `gpu/ink.py` pushes `seed * 7919` as an int uniform
  (`hal_ink_salt`); that is the ink pass, outside this package (see "For the
  integrator").

## 9. Sixteen accumulation passes leave the penumbra as grainy as one

- **What the user saw.** Anti-aliasing by the Accumulation Buffer, or depth
  of field by Lens Passes, smooths everything but the soft-shadow grain.
- **What was wrong.** Every road that averages whole frames rendered each
  pass with the same seed, and the shadow hash had no pass term: N passes
  traced the same rays N times.
- **What it does now.** A pass index, `_sample_pass`, joins the salt
  (`lights.soft_ray_salt`, x 104729). The audit's version set a private flag
  in the accumulation road; the verifier showed it does not survive the
  engine's settings copies (the lens road copies again and its carry list did
  not name it). Here `RenderSettings.copy()` itself carries that one flag --
  every other private flag is still dropped -- so it reaches the innermost
  render through any road, and `RenderSettings.sample_pass_for(k)` appends
  one base-65 digit (k + 1) per road, so a lens pass inside an accumulation
  pass, the refines of either and the frame outside all differ. Set by:
  `_render_accumulated`, `_render_lens_accumulated`, the adaptive refines
  (core/render.py) and the motion-blur slices (`engine.
  _render_motion_accumulated`, which puts the index back afterwards). On the
  GPU road the index rides a per-draw scalar, `hal_spass`, declared by any
  pass with a soft traced lamp and fed at the driver's four uniform sites and
  the simulator's two: the passes of a frame share ONE plan and one compiled
  shader. Nothing has to be refused.
- **Pictures that change.** Frames rendered through those roads with a sized
  traced lamp. Pass index 0 (every other frame) is the plain pattern.
- **Proof.** Grain as a share of one frame's (ideal in brackets): Accumulate
  x4 0.97 -> 0.48 (0.50), x16 0.97 -> 0.25 (0.25), Lens Passes x16 1.00 ->
  0.25, Accumulate x4 + Lens x4 0.97 -> 0.25; Supersample x4 0.48 -> 0.50
  (it always worked). With section 3 the x16 penumbra goes from 0.194 to
  0.028. Three accumulation passes of two lens passes shade with six
  different indices (66, 67, 131, 132, 196, 197), a plain frame with 0, the
  adaptive refines with 2, 3, 4 beside the base frame's 0. GPU road through
  the fake device, Accumulate x4: gpu-shaded, max |GPU - CPU| 0.000000, one
  plan in the cache; with the scalar forced to 0 the same frame is 0.119
  away, so the plumbing is what makes the roads agree. Real Blender F12,
  Accumulate x8: penumbra grain 0.93 -> 0.39 of one frame's (ideal 0.35).
- **Not done.** The occlusion, radiosity and blurred-reflection salts have
  the same shape and still redraw the same rays under these roads (the
  verifier's closing remark). They are not shadows; the recipe is the same
  one line per site plus the `hal_spass` term in `_ao_function` /
  `_rad_function`.

## 10. The panels

Pixels unchanged. Rows are checked through a recording layout that follows
the `active` flag up the layout chain, and in real Blender 5.2.2 against the
real RNA (four lamp kinds x five render methods x four lamp menus draw; both
operators run through bpy.ops).

- **Halcyon Light (size-ray-03, menu-ui-01, lamp-dials-greying-01,
  size-ray-09, planarz-hidden-01).** The shadow rows grey by the method that
  will RUN (`ui.light_shadow_method`: the render's Shadow Method unless it is
  Per Light; None, a hemi sun, Blender's own Shadow switch and the Shadows
  master veto), not by the lamp's menu. Before, render Shadow Maps + lamp
  Ray Traced drew the live Clip planes greyed, and render Ray Traced drew
  seven dead rows active. When the lamp's menu names a method the render
  overrides, two note rows say so and a button sets the render to Per Light
  (`halcyon.shadows_per_light`); at the factory state (both on Shadow Map)
  nothing is added. A traced lamp says "Ray Shadows is off: this light casts
  nothing" or "Samples is 1: traced shadows are hard, size is ignored"; a
  planar lamp shows Floor Plane Z; a hemi sun and a lamp vetoed by Blender's
  own Shadow switch each get one explaining row.
- **Shadows (size-ray-08, planarz-hidden-01).** The map dials grey under Ray
  Traced / Planar / None; Ray Shadows and Ray Bias are drawn under Ray Traced
  and Per Light, with an error row when the master is off under Ray Traced;
  Floor Plane Z is drawn under Per Light too.
- **Ray Tracing (rayshadows-greyed-01 as MODIFIED).** Ray Shadows is live
  while the Shadow Method lets a light trace, Ray Bias while anything traces
  (reflections, traced shadows, occlusion, radiosity). Neither hangs on the
  panel's master; neither is drawn live when nothing reads it.
- **Tooltips.** Shadow Method ("The shadow method EVERY light uses ... Per
  Light is the one setting under which each light's own Shadows menu
  decides"; it said "when a light does not choose"), Ray Shadows and Ray Bias
  (independent of the Ray Tracing switch), the lamp's Shadows (only read
  under Per Light), Samples ("At 1 a traced shadow is one ray at the light's
  centre: hard") and Density (mapped, traced or planar). All at least 40
  characters. The RENDER setting's Samples tooltip belongs to the map-soft
  package and was not touched; size-ray-09 asks it to say "1 = hard traced
  shadows: a light's size is not read".

## Not done: size-ray-11 (a size tick recompiles the shader)

Left as it was, on purpose. The verifier measured 320-595 ms per slider tick
on a two-material scene and named the fix: carry radius, tan(angle / 2), the
half-sizes and the unit axes in hal_lights texels for the soft-ray function
AND the area form factor (and the mapped road's blur literal). The "falloff"
package merged while this one was being written moved the texel stride to 16
and reserved texels 12-15; this branch forked before it, so a layout change
here would have collided with it. The recipe on top of the merged tree:
texel 12 = (radius, sun law constant, hx, hy), 13 / 14 = the unit axes;
`_shadow_function` and `_area_function` read them through `_lref`; drop
radius, area_size, area_x, area_y and (for an unshadowed-map lamp) position
from `_light_sig`, keeping only structure (size > 0, kind, shape, sample
count, the sun law). The first step the audit proposed -- dropping radius
from the signature when no shadow function is emitted -- saves only the 5 ms
re-plan (the verifier's measurement) and was not worth a change.

## Tests

tests/test_r254_ray_road.py, `python3 -m halcyon.tests.test_r254_ray_road`:
eleven test functions, 228 checks, 70 to 100 s. Each section fails on the
tree before its fix: sections C to J on the fork point (70 failures and the
helpers that did not exist), A and B on 87489e1 (the Per Light lamp's umbra
0.000, the lamp's Samples moving the frame by 0.0000). `test_0` pins this
package's own roads bitwise against `halcyon-1.92.0.zip` (the tree before
R254) when it sits beside the package and notes its absence otherwise: hard
traced shadows of every lamp kind, occlusion, radiosity and blurred
reflections at small and large seeds, area lamps of ordinary size, shadows
off. Mapped shadows are the map-soft package's this round and are not pinned
there. With HALCYON_GLSLANG set, section H compiles the soft-ray passes under
the strict front-end.

Neutrality beyond the module: the 312 rows of tests/featurematrix.ROWS render
bitwise the old tree on the CPU road except one, "soft RAY shadows" (the
pattern of section 3).

No existing test pinned an old behaviour; none was changed.

## For the integrator

- Register `tests/test_r254_ray_road.py` in tests/run_all.py.
- Changelog lines: sections 3 to 10 above, and for 2 the verifier's note
  that a lamp's stored Samples is read now.
- Shared lines touched outside the traced road: `gpu/shade._light_sig` gains
  `round(shadow_density, 6)` in its common tuple (the 'map' tail still
  carries it too); `RenderSettings.copy()` carries `_sample_pass`;
  `patterns.sample_u` wraps its salt; `gpu/shade.py` gains `_sample_pass(job)`
  and an `elif u == 'hal_spass'` branch at each of the four frame-uniform
  sites and a line at each of the two simulator sites.
- A trial merge with claude-cloud at 0d638aa (read-only, `git merge-tree`)
  is textually clean, and this module, test_r253_uniforms,
  test_r253_light_options, test_r253_review and the strict gate pass on the
  merged tree. One interplay was found and fixed here: the falloff package's
  panel test draws `HALCYON_PT_light` with scene settings that carry only its
  own fields, so the lamp panel reads the render's shadow state through
  `ui.render_shadow_state` with the renderer's defaults.
- The map-soft package: the lamp panel now greys Softness (and the map
  dials) by the effective method; the render's Samples tooltip should say
  that 1 is a hard traced shadow (size-ray-09).
- Follow-ups seen and not started: the pass index for occlusion, radiosity
  and blurred reflections (section 9); `gpu/ink.py`'s `hal_ink_salt` int
  uniform at large seeds (section 8); size-ray-11 (above); a soft traced
  frame can differ from its GLSL twin by one ray at a pixel whose ray grazes
  an occluder's edge within float noise, a tie either road may round either
  way (seen once in every run of this package: a 2.0 x 0.5 rectangle at 16
  rays, one pixel of 23040, 0.031 = one ray of sixteen; the ray crosses the
  plate's plane 4e-4 from its edge before the origin bias moves it onto it).
  The old pattern has the same exposure; the acceptance cases above are free
  of it.
