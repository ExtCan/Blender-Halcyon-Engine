# R254 -- native (Blender's own light options under Halcyon)

Work package `native` of the light audit. The integrator owns CHANGELOG.md,
README.md, version.py, blender_manifest.toml, `__init__.py` bl_info and
tests/run_all.py; nothing here touches them. What they need from this
package is under "For the integrator" at the end.

The user's report: "Fix the lights so that changing the size of the lights
actually works. The lights don't get softer/harder with size. Also, I think
some of the general options just don't work properly." This package is the
second sentence for Blender's OWN Light panel: the `native-main` audit
cluster (except the map and ray softness items, which belong to the
`map-soft` and `ray-road` packages), `area-power-01` (= native-area-power-
units = area-distance-gain), `style-time-fpsbase-01`, the verifier's missed
options, and three defects found on the way (a mirrored Area lamp, the
Cornell Box operator's lamp, the watts constant of the audit itself).

Roads checked: the CPU road and the GPU road through its simulator twin
(gpu/shade.plan_frame + simulate) headless; real Blender 5.2.2 F12 through
the add-on with Cycles' direct light on the same scene as Blender's own
answer; the rendered viewport and the Properties editor in a live UI
session under xvfb. "Old tree" below is the frozen snapshot of 87489e1
(before any light fix); "base" is 0d638aa, the commit this branch forked
from. Sheets: `native_before_after.png` (F12, old tree | this tree |
Cycles) and `native_light_panel.png` (X screenshots of the Light tab), both
in the scratch directory `wt/native/out/`.

## 1. Exposure, Temperature and Normalize did nothing

- **What the user saw.** Three rows directly under Power in the Light
  panel -- Exposure, the Temperature box with its Kelvin field, Normalize
  -- changed no pixel on any road. With Temperature ticked the stock panel
  relabelled the colour "Tint" while Halcyon Light drew it again as
  "Color".
- **What was wrong.** `export.export_light` built the lamp from
  `la.color` and `la.energy` alone; nothing in the add-on read `exposure`,
  `use_temperature`, `temperature` or `normalize`.
- **What it does now.** `export._native_light_options`, called at the end
  of export_light, folds them with Blender's own meaning: energy x
  2^exposure (BKE_light_power); colour x `light.temperature_color`
  (Blender's own blackbody value, so no table); with Normalize off, energy
  x `export.bke_light_area` -- BKE_light_area line for line (an Area
  lamp's face with the object scale, pi/4 of it for Disk and Ellipse; 4 pi
  r^2 for a Point or Spot, 4 at Radius 0; pi sin^2(angle/2) for a Sun, 1
  at Angle 0). Energy and colour already travel on both roads (the CPU's
  `sample`, hal_lights texels 0.w and 2.rgb), so nothing in core/ or gpu/
  changed for these three. Every statement is gated on the option being
  off its default.
- **Pictures that change.** A lamp whose Exposure is not 0, whose
  Temperature box is ticked, or whose Normalize is unticked: a file in
  which someone moved a dead dial changes look. Every other lamp exports
  the fields it exported (section 9).
- **Proof.** Real F12, median response on unclipped lit pixels, old tree
  -> this tree (Cycles): Point Exposure +1 x1.000 -> x1.999 (2.000), -2
  x1.000 -> x0.251 (0.250); Point 2000 K x(1, 1, 1) -> x(2.514, 0.648,
  0.026) (2.520, 0.647, 0.020); 12000 K -> x(0.827, 1.000, 1.564) (0.826,
  0.995, 1.566); Point Normalize off x1.000 -> x0.127 (0.126; 4 pi r^2 =
  0.1257); Spot the same three: 1.999 / (2.518, 0.647, 0.025) / 0.127.
  Halcyon / Cycles at equal settings: 1.002 with Exposure +1 (was 0.502),
  1.012 with Normalize off (was 7.97). Headless, all four types, CPU and
  GPU twin: x2.0000 and x(2.5201, 0.6469, 0.0204) on both, max |CPU -
  GPU| 6.7e-5 at a frame peak of 8.1. Rendered viewport: exposure_p1 moved
  0.00% of pixels, now 62.48% (the same frame as Power x2); temperature
  and Normalize likewise. Keyframed Exposure (0 -> 1, frame 3) x1.999,
  keyframed Temperature (6500 -> 2000 K) x(2.411, 0.659, 0.024) against
  Cycles' x(2.417, 0.657, 0.020), a driver on Power x1.999: animation
  works because the fold reads the evaluated lamp.

## 2. Two panels titled Light, dead rows, and a hidden Shadow switch

- **What the user saw.** The Light tab showed two panels both titled
  "Light" (the type row twice), Colour and Power a second time in Halcyon
  Light, and four rows the engine never read (Exposure, Temperature,
  Normalize -- until section 1 -- and Soft Falloff). Blender's own Shadow
  checkbox switched a lamp's shadow off under Halcyon while the Halcyon
  Shadows menu went on saying Shadow Map, and no panel drew it.
- **What was wrong.** `engine.FORCED_PANELS` forced DATA_PT_light,
  DATA_PT_EEVEE_light, DATA_PT_spot and DATA_PT_area (which does not exist
  in Blender 5.2); `ui.HALCYON_PT_light.draw` repeated `color` and
  `energy`; `use_shadow` lives in the header of DATA_PT_EEVEE_light_shadow,
  an EEVEE-only panel.
- **What it does now.** `ui.HALCYON_PT_light_native` (titled "Light",
  registered above Halcyon Light) draws the native rows the engine reads
  and nothing else: type; Temperature and Colour (labelled Tint while
  Temperature is on); Power or Strength, Exposure, Normalize; an Area
  lamp's Power Units and, on Blender Internal units, its Distance with
  what the lamp gives in Blender's watts; Radius, Angle or Shape / Size /
  Size Y; a Spot's Angle, Blend (greyed under a cone law that does not read
  it) and Show Cone; Cast Shadow. No Soft Falloff: a lamp lights from its
  centre here, so the option has no meaning (it moves a Cycles frame by
  x1.0003). Every lamp attribute is read through getattr, so the test
  stub's four-field lamp draws. The four stock names left FORCED_PANELS;
  DATA_PT_light and DATA_PT_spot, which carry the generic marker, joined
  EXCLUDED_PANELS. Halcyon Light no longer draws Colour and Power.
- **Pictures that change.** None (UI only).
- **Proof.** Real Blender, census of every registered Properties 'data'
  panel under engine Halcyon: panels titled Light, old tree
  [DATA_PT_EEVEE_light, DATA_PT_light] -> [HALCYON_PT_light_native], for
  all four lamp types. Live UI session, draw() calls recorded through
  Panel.append: the editor drew DATA_PT_context_light,
  HALCYON_PT_light_native and HALCYON_PT_light for a Point, a Sun, a Spot,
  an old Area and a new Area lamp. Under EEVEE, Cycles and Workbench the
  census is identical to the old tree's line for line (12 lines), and the
  live session drew DATA_PT_EEVEE_light / CYCLES_LIGHT_PT_light /
  DATA_PT_light and no Halcyon panel. Screenshots:
  `native_light_panel.png`.

## 3. A new Area lamp whited out the frame

- **What the user saw.** Add > Light > Area (10 W, Size 1) rendered a
  white frame. The lamp's Power was not Blender's watts: the engine runs
  Blender Internal's form factor, normalised at the lamp's Distance dial,
  and at the factory Distance 25 that is 312 times Blender's lamp. The
  Distance row's tooltip described the Custom ramp.
- **What was wrong.** Nothing in the law (it is 2.79's, by design, and
  2.79 imports depend on it); what a lamp made in Blender 5 received.
- **What it does now.** Every Area lamp carries
  `halcyon.area_power`: **Blender Power (Watts)**, the RNA default and so
  what a new lamp gets, or **Blender Internal (Distance)**, today's law.
  Under Watts `export._native_light_options` exports
  `decay_end = AREA_WATTS_DISTANCE = sqrt(2)` (and `decay =
  'INVERSE_SQUARE'`, so neither the lamp's hidden Falloff menu nor a
  Custom Range render default can cut its volumetric beam at 1.41 units;
  an Area lamp never attenuates on either road). Both roads read
  `decay_end` as before; core/ and gpu/ are untouched. Under Watts the
  object's scale keeps the POWER, as in Blender (energy / (scale_x
  scale_y)^gamma takes Blender Internal's "matrix size works on energy"
  gain out); on Blender Internal units a scaled lamp is the lamp it was.
  The panel draws Distance only on Blender Internal units, with "Gives N W
  in Blender's watts" beside it (Power x Distance^2 / 2).
- **The constant is sqrt(2), not the audit's sqrt(pi/2).** The verifier of
  area-power-01 derived 2 D^2 / pi from "Cycles gives albedo P cos cos' /
  (4 pi r^2)". Blender's Area lamp is a one-sided Lambertian face of flux
  P: radiance P / (pi A) (EEVEE `Light::shape_radiance_get`: `M_1_PI /
  area`), four times a Point lamp's P / (4 pi) on its axis, not pi times.
  Measured through real Blender against Cycles, 200 W, unclipped, 60 575
  pixels: at Distance sqrt(pi/2) Halcyon / Cycles = 0.788 (= pi/4); at
  sqrt(2) 1.003, 1.004 at Size 3, 1.004 at object scale 3. This is the
  native-main auditor's own number (D^2 / 2 = 312.5, his measured 313.8).
- **Old files.** A lamp saved before the option existed has no value
  stored and would read Watts and render 312 times darker. `light_units.py`
  keeps those lamps: wherever an Area lamp arrives without a stored value
  it is judged by the file it was last saved in, and 'BI' is written onto
  it when that file is an older Halcyon file (a scene renders with
  Halcyon, or carries a moved Halcyon setting, or the lamp's own Halcyon
  dials were moved). A file with no sign of Halcyon -- authored for EEVEE
  or Cycles -- opens in watts, as its lamps were made
  (`HALCYON_FILES_ONLY`; False gives the stricter reading, 'BI' for every
  unstamped file). The file stamp is the hidden
  `scene.halcyon.light_units_version`. Routes:
  - File > Open, Revert, Recover, the startup file: `load_post`, before
    the undo stack's first state is taken.
  - Linked lamps: judged at every load and at link time by THEIR library
    file, read from its bytes (the engine's name in a scene's RenderData,
    the stamp's property name; through zstd or gzip) -- the library's
    scenes are not loaded.
  - File > Append / Link, drag and drop, the asset browser,
    `bpy.data.libraries.load`: `blend_import_post`, the same search of the
    source file; lamps that arrive as dependencies of an appended Scene,
    Collection or Object are in the import items and are sorted too.
  - The add-on enabled or updated over an open file: sorted inside
    `register` (the real data behind Blender's restricted context), after
    the engine class exists; where a build offers no such road the sort
    waits for a zero-delay timer or `render_init` and then touches only the
    lamps the file on disk holds.
  - Undo: a value written outside the undo stack (the late-enable case) is
    written again by `undo_post` / `redo_post` for the lamps this session
    sorted.
  - A lamp made by a script or an operator has no stored value: watts.
    `save_pre` stores what each local Area lamp is using and stamps every
    scene, so a lamp appended later from that file says what it is.
  - The legacy importer and the append watch set 'BI' with the rest of a
    2.79 lamp's values (`legacy_import._apply_lamp_bi`).
  Not covered: a compressed library when Blender's bundled zstandard
  module is missing (the lamp's own dials decide alone); a lamp made while
  the add-on was disabled, in an unstamped Halcyon file (sorted with the
  old ones); "Reset to Default Value" on a migrated lamp's Power Units
  followed by an undo in the same session (the 'BI' returns; a choice from
  the menu is never overridden); an Area lamp appended from a pre-1.93
  Halcyon file that is zstd-unreadable AND has every Halcyon dial at its
  default (arrives in watts).
- **Pictures that change.** A lamp CREATED after this version (watts); a
  file never used with Halcyon whose Area lamps were at the factory
  Distance (they were white, now they are Blender's lamp). Area lamps of
  older Halcyon files: none.
- **Proof.** New 10 W lamp, real F12: old tree frame mean 0.7945 with
  72.9% of the frame white -> 0.0447, 0.0% white (lit mean 0.0108, Cycles
  0.0100; the frame is 8-bit stepped). Size 3 and object scale 3 keep the
  power (frame means 0.0446 / 0.0444). Old file: a .blend saved by the old
  tree through real Blender (two Area lamps tuned for the old law, one of
  them with every Halcyon dial at its default, and a Point lamp), plain
  and zstd-compressed, opened and rendered by this tree: both Area lamps
  stamped 'BI', frame **bitwise** the old tree's (max diff 0.000e+00). The
  same file with the add-on enabled only after it is open: bitwise again
  (and through the pending road, forced: bitwise, with a lamp the script
  added in between left in watts). Live UI session: the undo that restores
  the loaded state brings the lamps back without their value
  ([('Fill', False), ('Key', False)] before the add-on's handler), and
  they read 'BI', stored, after it. Append: Key (old file) 'BI'; Fill
  (zstd copy) 'BI'; a Light datablock from an old library with default
  dials 'BI'; the EEVEE file's lamp unset; a lamp from a file re-saved by
  this tree keeps its stored Watts. Link: the same three verdicts, and
  again after the linking file is saved and reopened. The EEVEE-authored
  file under Halcyon: mean 0.0511, 0.0% white. Headless: Area / Point at
  equal Power under the lamp 3.996 (Blender: 4); Blender Internal at
  Distance 25 / Watts 312.5; CPU against the GPU twin 3.5e-8.

## 4. A Spot's object scale was ignored

- **What the user saw.** Scaling a Spot object reshapes the cone Blender
  draws in the viewport (wider, narrower, oval); the render kept the
  unscaled round cone.
- **What was wrong.** export_light exported `spot_size` and `spot_blend`
  only; the matrix supplied position and direction. Blender shapes the
  cone by scale.z / scale.xy (eevee_light.cc `spot_size_inv`).
- **What it does now.** `export.spot_scale_fold`: with kx = |X| / |Z| and
  ky = |Y| / |Z| of the object matrix and k the larger, the half-angle
  becomes atan(k tan(half)) (kept below 89.9 degrees) and Blend is
  re-derived so the full-strength edge lands where Blender puts it; the
  POV hotspot angle moves the same way. That is the whole of the round
  case (kx = ky): every reader of the cone on both roads -- the cone laws,
  the shadow buffer, the volumetric beam, the projected slide and its
  focus blur, the light meter node -- follows the exported angles. When kx
  and ky differ the cone is an ellipse: the new `Light.spot_scale = (k /
  kx, k / ky)` squeezes the narrower axis inside the folded round cone.
  `core/lights.scaled_cos` (float32, one operation per statement; with
  Square the wider squeezed side, a rectangular pyramid) replaces the
  cosine every cone law reads; its GLSL twin is the `sc_*` block in
  `gpu/material._one_light_source`, the squeeze riding hal_lights texels
  10.w / 11.w (a scale drag is a plan-cache hit; whether a squeeze exists
  is one `_light_sig` entry appended only when it does). The slide is
  squeezed with the cone (`lights.cookie_spot_axes`, the CPU lookup's axes
  and the GPU's baked literals), its focus blur grows across the squeezed
  axis, and the beam is the same elliptical cone (`cones.spot_cone` solves
  the segment in the squeezed frame). The shadow buffer needs no new rule:
  the folded angle is the round cone that circumscribes the ellipse.
  Unit and uniform scale run none of it.
- **Pictures that change.** A Spot whose object scale is not uniform.
- **Proof.** Real F12 beside Cycles, share of the frame the cone lights:
  scale (2, 2, 1) 17.5% -> 58.0% (Cycles 58.1%); (2, 1, 1) 17.5% -> 38.1%
  (38.4%); (1, 1, 2) 17.5% -> 3.7% (4.4%); (3, 3, 3) 17.5% -> 17.5%, the
  frame bitwise the old tree's. Folded numbers: (2, 2, 1) on a 45 degree
  cone gives 79.28 degrees and Blend 0.1190. Brightness inside the cone,
  Halcyon / Cycles median: 0.889, 0.851 and 0.993 against 0.958 for the
  unscaled lamp -- the engine's Blender Internal cone multiplies by the
  cosine to the axis, which a wider cone shows more of (a native 79 degree
  Halcyon spot differs from Blender's the same way). Headless: on 6000
  random directions the cone lights exactly the directions Blender's
  `light_spot_attenuation` lights, at full strength where Blender's is,
  for five scales; `scaled_cos` within 1.0e-7 of float64; the emitted
  block equals it within 2e-6 and its arithmetic bitwise (d = 0.0), plain
  and with Square; CPU against the GPU twin 8.5e-6 (round fold), 8.8e-6
  (elliptical, map shadow), 1.0e-5 (no shadow), 9.1e-6 (with Square),
  1.1e-5 (with a slide), frame peaks about 6.1 -- under the 6e-3 bar by
  two orders. The beam against a brute-force march through the squeezed
  cone: median relative error 4.4e-7, worst 2.8e-6, over 88 rays. Strict GLSL 4.30
  (glslang) accepts every pass of the elliptical, elliptical + Square and
  elliptical-with-slide frames. Rendered viewport: scale (2, 2, 1) moved
  0.00% of pixels, now 23.86%.

## 5. Scene time ignored the frame-rate base and the subframe

- **What the user saw.** A light style (and every other clocked feature:
  caustics, the time nodes) ran at the wrong speed when the frame rate
  base was not 1, and under motion blur a strobing lamp was averaged over
  two whole frames instead of the shutter.
- **What was wrong.** `export.export_scene`: `time = frame_current / fps`.
- **What it does now.** `export.scene_time`: (frame + subframe) x fps_base
  / fps, the old expression kept when the subframe is 0 and the base is 1
  (those frames are the value they were, bitwise). The viewport's
  lights-only export stamps the same clock on its scene.
- **Pictures that change.** Scenes with a clocked feature at a frame-rate
  base other than 1, and every motion-blurred frame of such a scene.
- **Proof.** Real F12, FAST_STROBE, lit mean old tree -> this tree: fps 48
  / base 2, frame 3 (0.125 s, the style says dark) 0.3097 -> 0.0000; frame
  5 (lit) 0.0000 -> 0.3097; fps 24 / base 2, frame 3 (lit) 0.0000 ->
  0.3097; fps 24 / base 1 unchanged. Motion blur, 4 steps over a 0.5
  shutter: frame 3, dark across the whole shutter, 0.1552 -> 0.0000; frame
  5, one step of four before the 0.2 s boundary, 0.1552 -> 0.2325 (0.75 of
  0.3097).

## 6. Blender's hidden light switches

- **What the user saw.** A file authored in EEVEE or Cycles lost values
  Halcyon has an exact equal for, with nothing on screen saying so.
- **What it does now.** Three are honoured, each drawn in the Light panel
  only while it is in force (so a default lamp shows no extra row and no
  hidden switch shapes or darkens a lamp):
  - Blender's own `use_square` (a 2.79 file opened directly keeps it)
    exports the Square cone, Blender Internal's exactly. Real F12: lit
    share 17.1% -> 22.7%.
  - A Diffuse or Glossy factor of exactly 0 switches that lobe off
    (`affect_diffuse` / `affect_specular`, which both roads already read).
    Other factor values have no exact equal and stay unread.
  - The lamp OBJECT's ray visibility, Diffuse and Glossy (EEVEE multiplies
    each lobe by it, eevee_light.cc `Light::sync`; in Cycles
    visible_diffuse = False switches the lamp off, x0.0000): real F12 lit
    x1.000 -> x0.000, as Cycles. Headless the frame is bitwise Halcyon's
    own Specular Only lamp on both roads.
- **Left unread** (no equal here): Custom Distance (EEVEE's window is not
  Blender Internal's Sphere), the Transmission and Volume Scatter factors,
  every EEVEE shadow dial, Cycles' Spread, Soft Falloff, the lamp's node
  tree.
- **README-ready sentence.** "Under Halcyon a lamp's own Light panel reads
  its type, Colour, Temperature, Power or Strength, Exposure, Normalize,
  Radius / Angle / Shape and Size, a Spot's Angle and Blend, Cast Shadow
  and the object's scale; it does not read Soft Falloff, Custom Distance,
  the Transmission and Volume Scatter factors (Diffuse and Glossy only
  where they are 0), EEVEE's shadow settings (Jitter, Overblur, Filter,
  Resolution Limit, Absolute Resolution, the buffer clip start, a Sun's
  cascade count, exponent, fade and max distance), Cycles' Spread, or the
  lamp's node tree (a Cycles Emission node's colour and strength)."

## 7. The verifier's missed options, each through real Blender

| option | old tree | this tree | Blender (Cycles) |
| --- | --- | --- | --- |
| lamp node tree, Emission strength 5, red | x1.000 | x1.000 (unread, listed) | x(5, 1, 1) |
| object visible_diffuse = False | x1.000 | x0.000 | x0.000 |
| object visible_glossy / camera / shadow = False | x1.000 | x1.000 (diffuse scene) | x1.000 |
| diffuse_factor = 0 | x1.000 | x0.000 (EEVEE's meaning) | x1.000 (Cycles ignores it) |
| hide_render | the energy-0 frame, bitwise | same | same |
| lamp in an excluded collection | the energy-0 frame, bitwise | same | same |
| lamp reached through a collection instance | the base frame, bitwise | same | same |
| Spot scale (-1, -1, -1) and (1, 1, -1) | dark | dark | dark |
| Spot scale (-1, 1, 1) | unchanged | unchanged | unchanged |
| Area scale (-1, 1, 1) | **dark** | lit, the unmirrored frame | lit, x0.9999 |
| keyframed Exposure / Temperature | x1.000 | x1.999 / x(2.41, 0.66, 0.02) | x2.000 / x(2.42, 0.66, 0.02) |
| driver on Power | x1.999 | x1.999 | x2.000 |

Not run: DATA_PT_preview (the lamp thumbnail; the panel is listed and
closed, no thumbnail under the software rasteriser) and a real GPU F12
(llvmpipe; the GPU road is proven through the simulator twins).

## 8. Two defects found on the way

- **An Area lamp mirrored across one face axis lit nothing.** Object
  scale -1 in X or in Y (half of a mirrored rig) winds the lamp's face the
  other way and the form factor of a face seen from behind is zero.
  Blender keeps the handedness (eevee_light.cc negates the Y axis).
  `_native_light_options` negates `area_y` when cross(area_x, area_y)
  points against the object's Z; an unmirrored lamp runs nothing. Proof:
  real F12 lit mean x0.0000 -> the unmirrored frame (Cycles x0.9999);
  headless max diff 0.0 against the unmirrored lamp, CPU against the GPU
  twin 2.5e-7. Picture change: such a lamp was black and is now lit, on
  either power units.
- **The Cornell Box operator's lamp pointed at the ceiling.**
  `objects.HALCYON_OT_add_cornell_box` turned its Area lamp half a turn
  about X; an Area lamp shines along its own -Z, so the box stayed dark
  ("the box is lit the moment it is added"). The lamp now keeps its rest
  rotation and carries 10 W per unit of Size squared in Blender's watts
  (it was 100 W per unit squared, neither law's number). Proof: the
  operator at its defaults through real Blender, frame mean 0.083 with
  only the ceiling lit -> 0.097 with the walls, the blocks and the floor
  lit and 0.2% of the frame white.

## 9. What did not move

- export_light of a default lamp of each type: every field the old export
  had is repr-identical to the base tree's (the test freezes the table);
  against the old tree (87489e1) the one difference is `shadow_samples` 4
  -> 0, the already merged Samples dial. The new field `spot_scale` is at
  its inert (1, 1).
- Frames and shader text, base tree against this tree, nine lamp setups
  (the four defaults, a uniformly scaled Spot, a Square GL11 Spot, three
  Area lamps on the old law including a scaled one and a Disk with Gamma)
  under Shadow Maps and Ray Traced: 54 of 54 artefacts identical -- CPU
  frame bitwise, GPU twin frame bitwise, emitted GLSL the same text -- and
  the demo scene bitwise.
- Real Blender F12 of the default Point, Spot (also at uniform scale 3)
  and Sun: bitwise the old tree's frames.
- tests/glsl_strict: every shader compiles under strict GLSL 4.30 (174
  model passes, 396 console passes, 4 cel masters, 38 stages, the sky and
  the rasteriser).

## Tests

`tests/test_r254_native.py`, nine functions, 221 checks (three of them the
strict GLSL compile, run where HALCYON_GLSLANG is set; the zstd file check
runs where the zstandard module imports, as in Blender); 52 fail on the
base tree (every function fails or raises there). Existing modules run
green: test_r253_light_options, test_r254_falloff (one pin changed, below),
test_legacy, and the test_render functions that touch the changed code
(panels, export_light, the area form factor and shapes, the gobo, the spot
cones, the tooltip census).

Pin changed, `tests/test_r254_falloff.py` test_j_panel_draws_what_the_law_
reads: was "an Area lamp draws Distance, a note that brightness scales with
its square, then Gamma -- and no falloff menu" (drawn by HALCYON_PT_light);
now "Halcyon Light draws an Area lamp's Gamma and no falloff menu", "the
Light panel draws an Area lamp's Distance under Blender Internal power
units, with what the lamp gives in Blender's watts beside it (10 W at
Distance 25: 3125 W)" and "no Distance in watts". The Distance row moved
beside the Power it scales and is drawn only where it is read.

## For the integrator

- **Version strings.** Two tooltips and one console line say "before
  1.93" (properties.py AREA_POWER and `area_power`, light_units.py); the
  module docstring says "since 1.93". Change them if this round ships
  under another number.
- **Shadows rows of HALCYON_PT_light** (the ray-road package's lines): now
  that Cast Shadow is drawn in the Light panel, the Shadows menu and its
  sub-rows should grey when it is off:
  `sub.active = bool(getattr(light, 'use_shadow', True)) and hs.shadow !=
  'NONE'`, and the `col.prop(hs, 'shadow')` row itself in a row whose
  `active` is `bool(getattr(light, 'use_shadow', True))`.
- **ray-road's Area scale fold** (the AREA branch of export_light, not
  touched here). `_native_light_options` runs after that branch and reads
  the object matrix itself: under Watts it divides the energy by (|X|
  |Y|)^gamma, which cancels the fold's Blender Internal compensation
  (`energy *= (nx ny)^gamma`) and, on this tree's unfolded axes, the
  form factor's own gain -- either way the power is constant, as in
  Blender. If the fold lands with a different compensation, this divisor
  must match it. On Blender Internal units nothing here touches a scaled
  lamp. The mirrored-lamp fix negates `lt.area_y` after the branch,
  whatever the branch wrote. `bke_light_area` reads `la.size` and the
  matrix norms, not the folded `lt.area_size`.
- **map-soft.** The elliptical cone needs nothing in the bake: the
  exported `spot_size` is the circumscribing round cone, so the buffer's
  FOV and `_light_shadow_signature` are right as they are.
- **hal_lights texels 10.w and 11.w are taken** (the elliptical cone's
  squeeze, written for a squeezed Spot only).
- **`__init__._import_modules`**: light_units is registered after engine
  (its sort of an already-open file reads each scene's render engine).
- **A decision taken here, one switch to reverse.** Files with no sign of
  Halcyon open in watts (`light_units.HALCYON_FILES_ONLY = True`). The
  brief's wording, "every Area lamp of a file saved before this version",
  is the False setting: then an EEVEE-authored file keeps whiting out
  until each lamp's Power Units is switched by hand.
- **CHANGELOG, look changes to name:** lamps with a moved Exposure,
  Temperature or Normalize; non-uniformly scaled Spots; clocked features
  at a frame-rate base other than 1 and under motion blur; lamps whose
  Diffuse / Glossy influence was 0 or whose object was hidden from diffuse
  or glossy rays; Spots carrying Blender's own Square flag; Area lamps
  mirrored across one axis; new Area lamps and the Area lamps of files
  never used with Halcyon (watts); the Cornell Box operator's lamp.
- **README:** the sentence in section 6; the Area lamp's two power units
  (Distance sqrt(2) = watts, the factory 25 = 312.5 x).
