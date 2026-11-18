# R254 -- falloff (the light falloff family and the Light Limit)

Work package `falloff` of the light audit. The integrator owns CHANGELOG.md,
README.md, version.py, `__init__.py`, blender_manifest.toml and
tests/run_all.py; nothing here touches them. What they need from this
package is listed under "For the integrator" at the end.

The user's report: "The lights don't get softer/harder with size. Also, I
think some of the general options just don't work properly." This package
is the second half for the falloff menu, its distance dials and the Light
Limit: nine of the ten findings of the `hal-decay` audit cluster (the
tenth, the dead lamp's shadow bake, is the bake-speed package's) and its
verifier's three new ones, the Light Limit findings of `hal-flags`
(limit-brightest-01, limit-nearest-01 and the report and tooltip of
limit-first-order-ui-01), and one defect found while fixing them.

Roads checked for every fix: the CPU road (core/render.py, headless and
real Blender 5.2.2 F12), the GPU road through its simulator twin
(gpu/shade.plan_frame + simulate; the machine has no real GPU driver), and
the rendered viewport under xvfb for the two the user would see first.
"Fork point" below is commit 41c2521, a frozen copy of which every
before-number was measured on.

## 1. Falloff Start whites out an Inverse Square lamp

- **What the user saw.** A Point or Spot lamp on Inverse Square (or
  Inverse, or left on Scene Default) lit a blown-out white disc under
  itself. Nothing in its panel explained it: the lamp had once been on
  Custom Range, where Falloff Start was typed, and the slider is drawn for
  Custom Range only.
- **What was wrong.** `core/lights.attenuate` built `d = max(dist - start,
  EPS)` before the law switch whenever Start was above zero, and the two
  inverse laws divided by it: 1/(d - Start)^n, a pole at d = Start and
  1/EPS inside it. `gpu/material._attenuation_curve` emitted the same
  shift, so both roads agreed and were both wrong.
- **What it does now.** Falloff Start is a Custom Range value and nothing
  else, on both roads: `d = max(dist, EPS)` (the Custom Range branch keeps
  reading Start). The GLSL of the inverse laws no longer mentions Start.
- **Pictures that change.** Point and Spot lamps with Falloff Start above 0
  under Inverse or Inverse Square, chosen on the lamp or arriving through
  Scene Default. A lamp at Start 0 (the default) runs the statement it
  always ran.
- **Proof.** `LI.attenuate`, Inverse Square, Start 3, at d = 1, 3, 3.5, 5:
  before [1e12, 1e12, 4.0, 0.25], after [1.0, 0.111, 0.0816, 0.04] --
  bitwise the Start 0 values. Frame: 24 of 24 pixels inside Start sat at
  the 64.0 ceiling on the CPU road and on the GPU twin; now the frame is
  bitwise the Start 0 frame on both. Real Blender F12, 1000 W Point lamp,
  Shadow Map: frame mean 0.598 with 43.1% of the frame white -> 0.308 and
  0.0%, 0 codes from the Start 0 frame (was 238 codes). Rendered viewport:
  Start 0 -> 3 moved the frame by 0.9765; now 0.0000 while the export
  still carries start=3.

## 2. Scene Default = Custom Range ramps over the wrong width on the GPU

- **What the user saw.** With the render's Light Falloff Default on Custom
  Range, a lamp left on Scene Default with a Falloff Start reached further
  on the GPU than on the CPU and never went dark at Falloff End.
- **What was wrong.** `gpu/material.pack_light_texels` chose what hal_lights
  texel 3.y carries from the lamp's own unresolved `'DEFAULT'`, so it
  packed Falloff End where the emitted ramp divides by (End - Start).
- **What it does now.** The packer resolves Scene Default before choosing.
  CPU road untouched.
- **Pictures that change.** GPU road only: a Scene Default lamp under a
  Custom Range render with Start above 0. It now equals the CPU frame.
- **Proof.** Start 2.5, End 6.5: texel 3.y 6.5 -> 4.0; max |CPU - GPU|
  7.2e-1 of a 6.07 peak (11.8%) -> 5.3e-6; light past Falloff End on the
  GPU 0.683 -> 0; the GPU frame is now bitwise the GPU frame of Custom
  Range chosen on the lamp. Real Blender F12: Scene Default = Custom Range
  3..9 is 0 codes from the lamp-chosen frame.

## 3. Custom Range + Sphere + Falloff Start on the GPU

- **What the user saw.** Nothing yet: Sphere had no row. Once it has one
  (fix 7) the GPU would clip such a lamp at End - Start.
- **What was wrong.** `gpu/material._attenuation` took the Sphere's
  Distance from texel 3.y, which under Custom Range is the ramp's width.
- **What it does now.** The Sphere of a Custom Range lamp reads its own
  value, `max(End, eps)`, the very float32 `LI.attenuate` divides by, from
  hal_lights texel 5.w. The packer writes that texel for this pairing only
  (Point / Spot, resolved Custom Range, Sphere on); every other lamp's
  texels are the bytes they were, and every other Sphere still reads 3.y
  through the text it always had.
- **Why a texel and not `3.y + 3.x`.** The sum (End - Start) + Start is
  the one-line fix, and it is wrong when End is below Start, which the
  panel allows: the ramp's end is then clamped to Start + eps, so the sum
  is Start, while the CPU clips at End. Statement level, emitted expression
  on the packer's texels against `LI.attenuate`: Custom Range 4.0..2.6 +
  Sphere: texel 5.w 0.0, the sum 3.5e-1 (probe f08).
- **Pictures that change.** GPU road only: Custom Range + Sphere with Start
  above 0. At Start 0 texel 5.w is the float32 texel 3.y was: bitwise.
- **Proof.** Custom Range 2.5..6.5 + Sphere: max |CPU - GPU| 1.17 of a
  4.20 peak (27.9%) -> 4.5e-6; last lit pixel CPU 6.24 / GPU 3.84 -> 6.24
  / 6.24. Point and Spot, on the lamp and through Scene Default, Start 0,
  2.5 and the End-below-Start corner: twelve frames at or under 7.2e-6.

## 4. Blender Internal's cut-off through Scene Default

- **What was wrong.** `lamp_get_visibility`'s "visifac <= 0.001 -> 0" was
  applied when `light.decay` started with `BI_` -- the unresolved name --
  in `core/lights.sample` and in `gpu/material._one_light_source`. A lamp
  on Scene Default under a Blender Internal render default kept its tail.
- **What it does now.** Both roads test the resolved law
  (`LI.resolved_decay`, new: the one place a law's NAME is resolved).
- **Pictures that change.** A Scene Default lamp under a BI_* render
  default loses values at or below 0.001 x Power / (4 pi). At the factory
  Distance of 25 that starts 158 units out (BI Square).
- **Proof.** BI Square, Distance 0.02: pixels lit past the cut-off through
  Scene Default 118 -> 0, CPU and GPU twin, Point and Spot; the frame is
  bitwise the lamp-chosen frame on each road; the emitted block carries the
  snap line under a BI default and not under Inverse Square.

## 5. The volumetric beam's reach through Scene Default

- **What was wrong.** `core/render._light_volumes` capped the beam at
  Falloff End when `light.decay == 'CUSTOM'` -- the unresolved name again.
- **What it does now.** `LI.resolved_decay(light, st) == 'CUSTOM'`. The
  beam pass composites on the CPU whichever device shaded: one place.
- **Pictures that change.** The beam of a volumetric lamp left on Scene
  Default under a Custom Range render stops at its Falloff End.
- **Proof.** Spot, Volumetric 1, End 3 (probe f03, 160x120): frame sum on
  the lamp 2810.29, through Scene Default 97249.39 -> 2810.29, max
  difference 484.6 -> 0.0. This is the one line of this package inside the
  volumetric pass; the beam's brightness scale two statements below is
  another package's and is untouched.

## 6. The Light Limit

- **What the user saw.** At the factory settings (Light Limit 8, Brightest)
  a scene with a Sun and nine lamps rendered without the Sun, and said
  nothing. Nearest gave the same picture as Brightest.
- **What was wrong.** `core/lights.select_lights` ranked on `energy x
  mean(colour)` across lamp kinds: a Sun's strength (3) is a radiance, a
  Point lamp's Power (10 W) is divided by 4 pi and by its falloff before it
  lights anything. And its Nearest branch needed a `centre` that none of
  its nine callers passed, so Nearest ran the Brightest ranking.
- **Two verdicts, one implementation.** The finding was audited twice
  (hal-decay light-limit-raw-energy, hal-flags limit-brightest-01 and
  limit-nearest-01) and the two verified texts differ on where a lamp is
  measured. The hal-flags verdict arrived second, names the difference and
  asks for one; this is it:
  - `LI.frame_lights(scene, settings)` is the one selection (both
    verdicts). All nine readers call it: `core/render.light_surface` (its
    fallback), `_stash_ground_light`, `ShadeJob`; `core/fog.spot_fog_of`;
    `core/nodeeval.n_u16_light_meter`; `core/shadowmask.build`;
    `gpu/shade._selection_sig`, the plan-cache hit's texel repack, and the
    plan. Nothing is parked on the scene or the settings:
    `LI.limit_reference(scene)` is measured inside the call, lazily, only
    when the limit actually drops a lamp.
  - **Brightest** ranks on `LI.limit_strength`, what the lamp delivers in
    the light loop's own units (the hal-flags text). Sun / hemi: strength x
    mean colour. Point / Spot: |Power| x mean colour / (4 pi) x the lamp's
    own falloff law, the mean over the 3 x 3 x 3 grid on the scene's
    bounds, no distance taken below 5% of the bounds' diagonal. Area: its
    form factor (`area_inp`, Distance and Gamma included) on a surface
    facing it, the same mean -- zero behind its face. A Spot ranks as the
    point lamp it is inside its cone (the cone is not read; the tooltip
    says so). The raw Power, never the light style's gain: a strobing lamp
    keeps its slot, and the GPU plan with it. The camera is not read: an
    orbit never changes the set.
  - **Nearest** sorts on the distance from the camera's position (the
    hal-flags text: it is what "closest" means to the viewer, and what the
    games did; a bounds centre keeps the lamps in the middle of a level
    whatever the camera looks at). A Sun or hemi counts as distance 0. The
    cost, stated: the set follows the camera, so a move that changes it
    re-plans the GPU frame through `_selection_sig` -- for scenes over the
    limit with Nearest chosen, and only on the frames the set changes.
  - Ties keep scene order (stable sort; they used to fall to an unstable
    sort reversed), and the survivors come back in scene order (both).
  - First is untouched. A lamp list at or under the limit comes back as the
    very list it was, and no vertex is read.
  - The lamps left unlit are named, for all three modes
    (`LI.limit_report`): an F12 prints the line, reports it as a warning,
    and the Lighting panel draws it under the Light Limit rows ("Last
    render: 2 of 10 lamps not lit" and the names) while the limit and the
    mode are still the ones that render used.
- **Why not at the scene's centre alone** (the hal-decay text: distance to
  the bounds' centre, floored at 1e-6). A lamp that happens to sit near the
  middle of the bounds is then ranked near the pole of its own curve.
  Probe f06, a room-sized floor, a Sun of strength 3 and a chandelier of
  nine 100 W bulbs 1.62 from the centre, Light Limit 8: at the centre a
  bulb ranks 3.05 against the Sun's 3.00, the Sun is dropped and the frame
  keeps 55.8% of its light; over the grid a bulb ranks 0.64, two bulbs go
  and the frame keeps 86.0%.
- **A defect found on the way: the kept lamps' order was not in the GPU
  plan's signature.** `gpu/shade._selection_sig` fingerprints the kept SET
  in scene order, but the plan's light loop and the cache-hit repack ran in
  RANK order. A Power edit that swapped the rank of two kept lamps of
  different kinds was a plan-cache hit whose repack filled a spot's texels
  into a point lamp's block. Probe f01 (Point 100 W, Spot 90 -> 110 W,
  Point 1 W, Light Limit 2): max |CPU - GPU| 9.2e-7 on the first frame,
  3.8e-1 (90% of the frame peak) after the edit; now 1.2e-6. Returning the
  survivors in scene order makes the set the order.
- **Pictures that change.** Scenes with MORE lamps than the Light Limit
  under Brightest or Nearest: which lamps are lit, and their order (the
  per-material light limit's "first N of the selection", the float
  summation order). Scenes at or under the limit, limit 0, and First are
  bitwise.
- **Proof.** A Sun of strength 3 and nine 10 W Point lamps, factory limit:
  kept the nine lamps, frame mean 0.0301 of 0.5925 (5.1%) -> keeps the Sun
  and seven lamps, 0.5860 (98.9%), CPU = GPU twin to 1.2e-7, whichever way
  the list is ordered. An Area lamp of 60 W against a Point lamp of 300 W
  at the same place: keys 16.90 against 0.073 (x230), their solo frames
  11.59 against 0.057 (x204); Light Limit 1 keeps the Area lamp; the same
  lamp facing away from the scene ranks 0. Real Blender F12 (a Sun of 1.5
  and nine 60 W lamps, Shadow Map): the frame keeps 40.3% -> 89.4% of the
  unlimited frame's light, the console and the report say `Light Limit 8
  (Brightest): 2 of 10 lamps are not lit (X1, X8). ...`, Nearest drops the
  two lamps furthest from the camera instead (X2, X3) and differs from
  Brightest by 45 codes (0 before). Real Blender panel
  (bl_panel_limit.py): after the F12 the Lighting panel draws `Last
  render: 2 of 10 lamps not lit` / `Lamp0, Lamp8`; switching the mode
  removes the line until the next render; an F12 with the limit off clears
  it. Rendered viewport, eleven lamps: raising the Sun from 0 to 3 changed
  nothing (0.0000); now the frame mean goes 0.018 -> 0.779. Nearest on a
  rig where nearest is not brightest: kept the two far lamps (frame mean
  0.00269) -> keeps the two nearest the camera, and the frame is bitwise
  the frame of those two alone (0.00954); with the camera moved over the
  far end it keeps the lamp that hangs there, while Brightest keeps the
  same lamps wherever the camera stands.

## 7. The Light panel draws the dials the law in use reads

No pixel moves; `ui.HALCYON_PT_light.draw` and tooltips only.

- **Scene Default hid every slider** (scene-default-hides-sliders). A lamp
  left on Scene Default reads its own Start / End / Lin / Quad / Ref
  Brightness for whatever law the render chose, and drew none. The rows
  now follow the law in use (`ui.light_falloff_in_use`), with a note
  "This lamp's dials for the render's falloff".
- **The Blender Internal laws never drew their Distance**
  (bi-distance-not-drawn). All three are scaled by Falloff End; it is drawn
  as "Distance".
- **Sphere had no row** (bi-sphere-not-drawn), though it was exported and
  clipped the lamp on every road; a 2.79 import with Sphere on could not
  be inspected or cleared. Point and Spot lamps draw it; while it is on,
  the laws that draw no distance of their own (None, Inverse, Inverse
  Square, OpenGL 1.1) draw "Sphere Distance".
- **A Sun drew the Falloff menu and its sliders**
  (sun-falloff-menu-dead); none changed a pixel (a directional lamp never
  reaches `attenuate`). A Sun or hemi draws no falloff rows.
- **An Area lamp's Distance is a gain** (area-distance-gain, the wording
  half): a note under the row, "Brightness scales with Distance squared",
  and a Falloff End tooltip that names every role of the value. The Area
  lamp's brightness law is untouched (another package decides it).
- **Lin / Quad sliders** (the verifier's lin-quad-tooltip-names-bi-only):
  the tooltips named Blender Internal only although OpenGL 1.1 reads them
  too, and the hard maximum of 1.0 capped OpenGL's coefficients. Tooltips
  name both readers; hard maximum 100, slider (soft) maximum still 1.
- **One table.** `ui.FALLOFF_ROWS` maps each law to the dials it reads.
  The test holds it against the engine: for all fifteen laws, chosen on
  the lamp or through Scene Default, on Point and Spot, the set of dials
  the panel draws equals the set `LI.attenuate` reads when each is moved
  alone. Before: 38 of 60 rows differed. Now 0.
- **Proof in real Blender** (the audit's bl_decay.py through this tree,
  real RNA, recording layout): `draw POINT BI_LINEAR: color, energy,
  decay, decay_end[Distance], bi_sphere, decay_near_start,
  decay_near_end`; `draw SUN CUSTOM: color, energy`; `draw AREA DEFAULT:
  color, energy, decay_end[Distance], area_gamma`. Its F12 sections on
  this tree: all fifteen laws on Point and Spot within 0.50 codes of the
  label's formula, every law through Scene Default 0.00 codes from the law
  chosen on the lamp.
- Tooltips reworded (all 40+ characters): `decay`, `decay_start`,
  `decay_end`, `decay_ld1`, `decay_ld2`, `bi_sphere`, `area_gamma` on the
  lamp; `light_falloff_default`, `light_limit_mode`, `max_lights` on the
  scene; the three Light Limit items. No property was added, no enum item
  moved, no default changed.

## Default neutrality

- **360 frames bitwise against the fork point**, probe v07: 180 lamps
  (fifteen laws x on the lamp / Scene Default x Point / Spot x plain /
  Sphere / Near pair, at Start 0) on the CPU road and on the GPU twin. 0
  frames differ.
- **The whole feature matrix** (tests/featurematrix.ROWS, 96x72), probe
  f09: sha1 of the CPU frame, of every GLSL pass source the plan emits and
  of the packed hal_lights texels, this tree against the fork point.
  312 rows: 0 CPU frames differ, 0 of the 290 planned rows emit a
  different GLSL text (22 refuse by name in both trees), 0 texel arrays
  differ.
- **Real Blender F12**, before against after: Inverse Square at Start 0,
  Custom Range 3..9, Scene Default = Custom Range 3..9 and the ten-lamp
  scene with the limit off: 0 codes each.
- **The suite's own pins** at the function level
  (`test_a_start_zero_and_under_the_limit_are_the_old_statements`): the
  inverse laws at Start 0 are bitwise the old statements and emit the old
  text; a Sphere outside Custom Range emits the old text; at Start 0 the
  packed texels do not depend on the render's default and texel 5.w is 0
  on every lamp; eight lamps under limit 8, any number under limit 0, and
  First return the scene's own list without measuring the scene once; a
  bare lamp list keeps the raw ranking.

## Two roads

- Every changed case in the suite: max |CPU - GPU twin| 8.3e-6 (bar 6e-3).
- The verifier's sweep, probe v11, 240 random lamps of the family (law on
  the lamp or through Scene Default, Start, End, Lin, Quad, Ref
  Brightness, Sphere, Near pair, Power sign, Point and Spot): fork point
  229 agree within 1e-3 of the peak and 11 diverge (10 x Scene Default =
  Custom Range with Start, 5 x Custom Range + Sphere + Start); this tree
  240 agree, 0 diverge, worst 1.4e-5 of the peak, no NaN or inf.
- Strict GLSL gate (tests/glsl_strict.py under glslang): engine models
  174 ok / 3 refused by name, console 396 / 12, masters 4, post stages 38,
  sky and raster 3, compute raster 8, 0 bad: "every shader compiles under
  strict GLSL 4.30". The gate's scenes carry no falloff variants, so
  `test_l_strict_glsl_of_the_falloff_blocks` compiles the changed shapes
  when HALCYON_GLSLANG is set: 10 planned passes, all accepted.
- Plan signature: nothing new is baked. Texel 5.w is a value (a drag
  re-uploads); whether the Sphere reads it follows `decay` and
  `bi_sphere`, both already in `_light_sig`, and the render's default,
  already in the plan signature. The `decay_start > 0` bit of `_light_sig`
  is left as it was (it now costs one re-plan when Start crosses zero and
  nothing else).

## Tests

`tests/test_r254_falloff.py`, 12 functions, 153 checks (one of them the
strict front-end, which runs when HALCYON_GLSLANG is set), 25 s. Run
against the fork point (the helpers fall back to the calls it had) 78
checks fail, one or more per defect, and 55 pass (the neutrality pins); on
this tree all pass.

Pin changed: `tests/test_r251_material.py::test_c057_class_carry` (iii).
- Old: `[l.name for l in LI.select_lights(sc_b.lights, st_b)] == ['Fill']`
  on the stock demo -- "BRIGHTEST at max_lights 1 culls the Key (Fill
  alone)". That is the defect: a Sun of strength 6 losing to a 600 W Point
  lamp five units away on raw energy (5.68 against 410; delivered, 5.68
  against 0.96).
- Now: (iii-a) `LI.frame_lights` keeps `['Key']` on the stock demo (5.68
  against the Fill's 0.96 over the grid); (iii) the culled-key case the
  check is about runs with the Key at strength 0.5 (0.47 against 0.96):
  `['Fill']`, `key_lit` never captured, every face SHADOW -- the same
  three assertions.

Existing tests, this tree against the fork point (same counts unless
said): test_r253_light_options 195 ok; test_r251_lighting_lamps 106 ok;
test_r251_lighting_models 87 ok; test_r251_material 243 ok (242 + the new
iii-a); 23 functions of test_render.py that touch attenuate / the packer /
the limit / the panel and tooltip audits 638 ok; test_legacy lamp mapping
30 ok; test_r251_shadow planar + midpoint 17 ok; console presets 21 ok;
enum numbers 17 ok; spot fog 7 ok. The only failures on either tree are
the missing-zip pins and the Super FX fake-device 6e-8 pin of
baseline_fails.txt.

## Not done here, on purpose

- The Area lamp's x1250 against a Point lamp of the same Power, and any
  power-unit option: another package decides.
- dead-lamp-shadow-bake (a lamp nothing reads still bakes its map): the
  bake-speed package.
- A scene with no geometry at all (lamps over the analytic ground plane
  only) has no bounds for Brightest to average over and keeps the raw
  ranking; Nearest still measures from its camera.
- The viewport does not print or remember the limit's report (it would
  repeat on every redraw); the F12 does, and the panel line is that F12's.
- The volume march and the visible beam do not read the Light Limit at all
  (hal-flags flags-volume-beam-01): volumetrics, another package.

## For the integrator

- **tests/run_all.py**: add `test_r254_falloff`.
- **CHANGELOG.md:897-902** lists "the 3ds Max decay Start" among the dials
  already shipped. That was this shift, and it is gone: 3ds Max's Decay
  with a Start is the two Nominal Distance laws (Falloff End is the Start
  R0). The entry needs rewording with this round's.
- **Anything that needs the frame's lamps calls
  `LI.frame_lights(scene, settings)`.** `test_i_one_selection_for_both_
  roads` walks core/, gpu/, nodes/ and the package root with `ast` and
  fails on any `select_lights(` call outside `frame_lights`, naming file
  and line. A bake that keeps "the lamps the limit selected" from a bare
  `select_lights(scene.lights, settings)` would rank on raw energy and
  disagree with the frame.
- `select_lights`'s first two lines (the docstring and the `active = `
  candidate filter) are textually untouched, and `frame_lights` /
  `limit_report` derive their candidates from `select_lights` itself, so
  a change to which lamps are candidates (Ambient Only, say) lands in that
  one line. Its third parameter is `reference` now (it was `centre`, which
  nothing passed).
- Nearest reads `scene.camera.matrix_world`. A pass that renders the scene
  from another camera sees the set that camera gives.
- hal_lights texel 5.w is taken (the Sphere's Distance of a Custom Range
  lamp). Texels 12-15 are still reserved zeros.
- Adjacent text other packages may touch: properties.py GROUP_DOCS
  `decay_end` sits directly above `'shadow'`; ui.py's falloff rows end
  above the Spot rows, the Shadows rows are untouched; the Lighting panel
  gained the report line between Light Limit Mode and Light Falloff
  Default; engine.py gained thirteen lines between the export split print
  and the device plan.

### CHANGELOG fragment

- **Falloff Start no longer whites out an Inverse Square lamp.** Falloff
  Start is a Custom Range value; the inverse laws used to divide by
  (d - Start), so a value left over from Custom Range (the only law that
  draws the slider) put the whole disc inside Start at the ceiling. Both
  roads. Lamps at Start 0 are bitwise.
- **The Light Limit keeps the Sun.** Brightest ranks on what a lamp
  delivers -- a Sun's strength against a lamp's watts after 4 pi and its
  own falloff, an Area lamp by its form factor, averaged over the scene --
  instead of the raw numbers; Nearest really measures a distance (from the
  camera; a Sun always counts); ties and survivors keep scene order; the
  lamps left unlit are named in the console, the render's report and the
  Lighting panel. One function, `LI.frame_lights`,
  for the CPU loop, the GPU plan, the fog, the ground plane, the light
  meter and the planar shadows. Scenes at or under the limit are bitwise.
- **Scene Default follows the law everywhere.** On the GPU a Scene Default
  lamp under a Custom Range render ramps over End - Start as the CPU does;
  Blender Internal's 0.001 cut-off and the volumetric beam's reach apply
  through Scene Default as they do with the law chosen on the lamp; Custom
  Range + Sphere + Start clips at Falloff End on the GPU.
- **The Light panel shows what the lamp reads.** A lamp on Scene Default
  shows the dials of the law the render chose; the Blender Internal laws
  show their Distance; Sphere has a row; a Sun shows no falloff rows; an
  Area lamp's Distance says it scales brightness with its square; the Lin
  / Quad sliders take OpenGL coefficients above 1.

## Probes

All under
`/tmp/claude-0/-home-user-Blender-Halcyon-Engine/1c5e906d-b1ee-5f5f-923e-25adb13f5e76/scratchpad/audit/falloff/`
(run as `python3 <script>`; `HAL_ROOT=<dir holding halcyon/>` points one at
another tree; saved outputs in `out/`): v07_fix_check.py (360 reference
frames), v09_sphere_fix.py, v11_random.py (the verifier's own, unchanged);
f01_limit_order_plan.py, f02_limit.py, f03_beam_reach.py, f04_text.py,
f06_floor.py, f08_sphere_value.py, f09_matrix_neutral.py; real Blender:
spec_start.json and spec_limit.json through blui/bl_harness.py with
f05_bl_sheet.py (numbers and `out/bl_before_after.png`),
bl_panel_limit.py (the two real panels on a recording layout, the
tooltips and ranges from RNA), bl_viewport_falloff.py (the rendered
viewport, made by mk_viewport_probe.py), and the audit's bl_decay.py
(`out/bl_decay_after.txt`).
