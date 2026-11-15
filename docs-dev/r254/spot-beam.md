# R254 -- spot-beam (Spot / Area / Sun / Hemi options and the per-lamp effects)

Work package `spot-beam` of the light audit. The integrator owns CHANGELOG.md,
README.md, version.py, `__init__.py`, blender_manifest.toml and
tests/run_all.py; nothing here touches them. What they need from this
package is under "For the integrator" at the end.

The user's report: "The lights don't get softer/harder with size. Also, I
think some of the general options just don't work properly." This package
is the second half for the options a lamp carries beside its shadow and
its falloff: the audit clusters `hal-spot-area-hemi` and `hal-effects`
(both verified), minus the items other packages own (sun-angle-map-01,
area-power-01, the four shadow-bake findings, the contribution flags).

Roads checked for every fix: the CPU road (core/render.py, headless and
real Blender 5.2.2 F12), the GPU road through its simulator twin
(gpu/shade.plan_frame + simulate) and, where a whole frame matters, the
whole GPU road through tests/fakedevice; the rendered viewport under xvfb
for the Sun's shafts. The machine has no real GPU driver. "Before" is
claude-cloud at 0d638aa, a frozen copy of which every before-number was
measured on.

Sheet (real Blender, before | after):
`/tmp/claude-0/-home-user-Blender-Halcyon-Engine/1c5e906d-b1ee-5f5f-923e-25adb13f5e76/scratchpad/wt/spot-beam/out/spot_beam_before_after.png`

## 1. Hotspot does something under the default cone law

- **What the user saw.** The Hotspot slider of every spot lamp moved no
  pixel. Only the POV cone law read it; the default law (Blender
  Internal), OpenGL and the six GX laws ignored it, and the panel drew it
  for all nine.
- **What was wrong.** `core/lights.spot_law_coeffs` was the only reader
  (POV's radius). `core/lights._spot_blender` took its band from Blender's
  Blend alone, and so did the three GPU sites.
- **What it does now.** Under the Blender Internal law Hotspot is the
  inner angle: full strength (times the cosine, as always) inside
  cos(hotspot/2), the same smoothstep out to the Spot Size, zero past it.
  One helper, `core/lights.spot_blend_band(light, spotsi)`, returns the
  band's width in cosine units -- `(1 - spotsi) * Blend` at Hotspot 0 (the
  caller's own expression, so such a lamp moves no bit), `cos(min(hotspot,
  size)/2) - cos(size/2)` above it -- and every site calls it:
  `_spot_blender`, `gpu/material.pack_light_texels` (hal_lights texel
  2.w), the baked literal in `gpu/material._one_light_source`, the
  "a blend exists" bit of `gpu/shade._light_sig`, and the visible beam's
  edge (`core/cones.beam_edge`). It is a VALUE on both roads: dragging
  Hotspot re-uploads one texel. A Hotspot at or past the cone closes the
  band (the hard-edge block; that crossing is structure and re-plans).
  The panel draws Hotspot under Blender Internal and POV only (the OpenGL
  and GX laws have no inner angle) and says "Hotspot replaces Blend (Beam
  Shape)" while it does.
- **Pictures that change.** A spot on the Blender Internal law whose
  Hotspot is above zero -- including any file where the dead slider had
  been moved. tests/featurematrix.py's cone-law rows carry hotspot 0.5
  under their BLENDER A/B twin; those tests compare relatively and pass
  unchanged. Hotspot 0: bitwise.
- **Proof.** Function: for Hotspot 0.2 / 0.5 / 0.8 on a 1.0 rad cone the
  factor is exactly cos inside the inner angle and the smoothstep over
  cos(h/2) - cos(size/2) within 2e-5; Hotspot 1.0 / 1.7 / 3.1 are the
  Blend 0 cone bitwise. The CPU's float32 band equals the packed texel
  bitwise (0.0913299 at Hotspot 0.5). The emitted statements, standalone
  through the GLSL front-end: d = 0 against the CPU law. Frames (floor +
  box, 128x96): max |CPU - GPU twin| 2.5e-06 / 5.4e-06 / 7.2e-07 at
  Hotspot 0.3 / 0.7 / 1.4. A Hotspot drag with the plan cache warm is a
  HIT and the frame is the CPU's (5.4e-06). Ortho floor probe
  (probes/q03_hotspot.py): the flat radius is 0.281 / 0.531 / 0.781 for 15
  / 30 / 45 deg (wanted 0.263 / 0.536 / 0.828, pixel pitch 0.0625). Real
  Blender F12, default spot (45 deg, Blend 0.15): Hotspot 20 deg changes
  4629 of 76800 px, 40 deg 1606 px; before, 0 px each.

## 2. The visible beam's unit

- **What the user saw.** With the beam switch on, Volumetric 1 on a
  Blender-stock lamp drew a solid white wedge (a 1000 W spot: 35.5% of
  the real-Blender frame white; a 1000 W point: 100%). The usable slider
  band was 0.001 .. 0.02 on a 0..4, two-decimal slider.
- **What was wrong.** `core/render._light_volumes` multiplied the marched
  scatter by `energy / pi`. A spot or point lamp lights its surfaces with
  `energy / (4 pi)` (lights.sample), and an isotropic medium scatters
  1/(4 pi) of that toward the eye: the composite was 16 pi^2 above the
  physical in-scatter before the slider's own unit is counted. An Area
  lamp's beam read the Power and not the Distance its surfaces are scaled
  by (the form factor's `dist^2 / A`).
- **The two proposals, measured.** The two verifiers proposed `energy /
  (16 pi^3)` and `energy / (64 pi^3)`. probes/q01_beam_unit.py, share of
  the frame over white at Volumetric 1 / 4 (and the beam's 95th
  percentile):

  | lamp | 16 pi^3 | 64 pi^3 |
  |---|---|---|
  | spot 1000 W, 4.7 m | 4.0% / 14.0% (p95 0.87 / 3.49) | 0.6% / 4.0% (p95 0.22 / 0.87) |
  | spot 1000 W, side-on | 1.2% / 15.9% | 0.0% / 1.2% |
  | point 1000 W, 5.8 m | 71.6% / 100% | 10.2% / 71.6% |
  | point 1000 W, factory position | 25.1% / 93.5% | 0.0% / 25.1% |

  16 pi^3 gives a spot a slightly livelier unit value and makes a point
  lamp unusable above 0.25; 64 pi^3 is the only one of the two that
  leaves both kinds a range.
- **What it does now.** `core/cones.beam_gain(light)`, called by
  `_light_volumes` (and the unused `add_cones`):
  - Spot, Point: `energy / (64 pi^3)` -- the lamp's own energy / (4 pi),
    the isotropic phase 1 / (4 pi), and Density x Volumetric counted as
    the haze's scattering coefficient in units of 1 / (4 pi) per world
    unit. Volumetric 1 at Density 1 is a mean free path of 12.6 units, a
    thick stage haze; 4 is smoke.
  - Area: `energy * Distance^2 / (32 pi^3)` -- the same scale through the
    form factor's far field (2 D^2 times a point lamp's). The beam follows
    Distance exactly as the lamp's surfaces do, so it is right under BI's
    Distance 25 and stays right when the native package expresses the
    same lamp in Blender's watts (Distance sqrt(pi/2)): the two give the
    same surfaces (2.6e-07 of peak) and the same beam (peaks 0.3435 /
    0.3435; they were 398x apart).
- **The working range, honestly.** Spot: the whole slider -- at 1000 W,
  Volumetric 0.25 is a hint (p95 0.05), 1 a clear beam (p95 0.22 on a
  floor of median 0.33), 4 a strong one (p95 0.87, 3.9% of the frame over
  white). Area, a lamp exposed to light its floor near 0.75: beam peak
  0.05 / 0.20 / 0.82 at 0.25 / 1 / 4. Point: a point lamp fills the whole
  frame with glow, so at 1000 W seen from 3 to 6 units the range that
  reads is 0 .. 1 (0.25: glow median 0.06; 1: 0.23 with 15% of the frame
  over white; 4: 72% over white, in real Blender 70.9%); at 100 .. 300 W
  or further away the whole slider works. One physical unit cannot give
  a cone and an all-round glow the same headroom; the Density dial is the
  per-scene trim.
- **Which scenes get dimmer beams, and by how much.** Every Spot and Point
  beam: 64 pi^2 = 631.65 times dimmer at the same Volumetric and Density
  (multiply Beam Density by 631.65 to restore a beam exactly; the dial's
  new range tops out at 100000). Area beams: `Distance^2 / (32 pi^2)`
  times today's -- 1.979 times BRIGHTER at the default Distance 25, equal
  at Distance 17.8, dimmer below. Lamps with Volumetric 0, and every
  frame with the switch off, are bitwise.
- **Proof.** tests `test_c_beam_unit`: on the old tree the same checks
  read beam p95 138.0 at Volumetric 1 (floor median 0.326), 40.4% of the
  frame over white at 4, the point lamp 100% white. Whole GPU road against
  CPU with the beam: 1.6e-05 (one composite, both devices). Real Blender
  F12, 1000 W spot: 35.5% / 36.6% of the frame white at Volumetric 1 / 4
  before, 0.4% / 2.2% after; 1000 W point: 100% / 100% before, 21.3% /
  70.9% after.

## 3. One lamp, one volumetric effect

- **What the user saw (after fix 2, in my own first picture).** A spot at
  Volumetric 1 with its beam drawn also grew banded streaks below its own
  pool, and the box under it turned into a staircase.
- **What was wrong.** `core/render.shaft_sources` lists every lamp with
  Volumetric above zero, so the one slider fed the in-air beam and the
  screen-space Light Shafts at once. Nobody saw it: under the old unit a
  usable beam sat at 0.001 .. 0.02, where the shafts' strength is nothing.
  On the lamp's own radiance scale a beam is set near 1, and the same 1
  smeared every bright pixel away from the lamp in the march's 24 steps.
  The verifier had weighed dropping beam lamps from the shaft sources
  and left it out because it changes F12 pictures; with the unit fixed it
  is the unit change that would otherwise change them for the worse.
- **What it does now.** A Spot, Point or Area lamp is a shaft source only
  while the render does not draw its beam (Volumetric Lights off); a Sun
  or a hemi has no beam and always throws shafts. The lamp's tooltip, the
  render switch's tooltip and the Light Shafts group say so, and the lamp
  panel's "No beam" row adds "(the lamp throws Light Shafts only)".
- **Pictures that change.** F12 frames with the beam switch ON and a
  Spot / Point / Area lamp's Volumetric above zero lose that lamp's
  shafts (those frames change by fix 2 anyway). Switch off: bitwise, the
  lamp throws shafts as before (the 1000 W spot's 320x240 frame hashes
  the same on both trees).
- **A side effect that matters.** Lamps set up for beams no longer pay the
  shaft stage at all, the largest cost the verifier measured (four beam
  spots at 640x480: post 5.81 s of a 8.5 s frame).
- **Proof.** `test_c_beam_unit`: sources per kind with the switch off /
  on -- SPOT 1 / 0, POINT 1 / 0, AREA 1 / 0, SUN 1 / 1, HEMI 1 / 1 (old
  tree: 1 / 1 everywhere).

## 4. The lamp panel says what Volumetric needs

- **What the user saw.** Volumetric on a Spot, Point or Area lamp drew no
  beam until an unrelated-looking switch -- the header checkbox of a
  closed sub-panel called "Spot Cones" under Render > Lighting -- was
  ticked. A Sun's Volumetric did nothing unless the sun was ahead of the
  camera. The panel said neither.
- **What it does now (UI and text only).**
  - `ui.HALCYON_PT_light`: while Volumetric is above zero and the render
    switch is off, the lamp panel draws "No beam: Volumetric Lights is off
    in Render > Lighting" and the switch itself ("Enable Volumetric Lights
    (Render)").
  - The render sub-panel is titled "Volumetric Lights" (it gates three
    lamp kinds); its dials are labelled Beam Density / Samples / Falloff /
    Reach; the property identifiers are unchanged, so files keep loading.
    `spot_cone_density` has a range (0 .. 100000; it had none).
  - A Sun with Volumetric above zero gets a live status line
    (`ui.shaft_anchor_status`, fed by `ui.lamp_anchor_view`, the flare's
    own anchor code factored out): behind the camera / far outside the
    frame / "Light Shafts converge on the sun's place in the sky". Its
    bounds are `shaft_sources`' own (|ndc| 6 = a frame fraction outside
    -2.5 .. 3.5): 0 disagreements on 400 random sun directions.
  - Tooltips: Volumetric, the switch, Beam Density (its unit).
- **Proof.** `test_l_panel_says_what_each_dial_needs`. Real Blender 5.2.2
  with a UI (bl/bl_panel_states.py): the Properties editor drew the panel
  27 times across ten states (spot with the switch off / on, Hotspot, POV,
  GX Ring 2 + Screen Spotlight, area with a cookie, point, sun ahead of
  and behind the camera, hemi) with 0 exceptions, and the rows through
  the real RNA are the ones listed here -- `shadow_soft_size` exists on
  an Area lamp, the Sun's line reads "Light Shafts converge on the sun's
  place in the sky" ahead of the camera and "Sun is BEHIND the render
  camera -- no shafts" behind it.

## 5. The beam follows the lamp's cone law

- **What the user saw.** Cone Law, Hotspot and Cone Exponent shaped the
  pool on the floor and left the visible beam alone; under POV the beam
  kept following Blend, which the pool never read.
- **What was wrong.** `core/cones.spot_cone` applied its own smoothstep
  over Blend and never asked `lights.spot_law_factor`.
- **What it does now.** `core/cones.beam_angular`: the Blender Internal
  beam keeps the smoothstep it always drew (bitwise at Hotspot 0, and its
  edge is the Hotspot's band above it); every other law calls
  `spot_law_factor` itself. The verifier's correction is in: GX Ring 2 is
  at full strength AT the Spot Size and fades to zero at `c - (1 - c) /
  sqrt(2)`, so its cone is marched out to that zero
  (`core/cones.beam_cos_out`) instead of ending in a cut rim. The
  brute-force twin `core/cones.reference` takes the same law statement.
  A Square cone's beam stays round, as its tooltip says.
- **Pictures that change.** Beams of spots on a law other than Blender
  Internal, and of Blender Internal spots with a Hotspot.
- **Proof.** Beam sum against the Blender Internal beam (every row was
  1.000): POV 0.521, POV + Hotspot 30 deg 0.824, GL11 1.058 (a hard rim),
  GX Ring 1 0.713 (hollow), Blender Internal + Hotspot 10 deg 0.554. GX
  Ring 2: 5293 beam px against 3799, edge / peak 0.019 (the naive fix
  measured 0.440). Analytic cone against the 4096-step brute-force march
  under eight law cases: median relative error under 1%. Whole GPU road:
  the same beam to 3e-08 (probes/q04_beam_law.py).

## 6. A point lamp's glow is the integral

- **Found while measuring the beam dials on Point beams (a missed
  option).** `core/cones.point_glow` marched twelve evenly spaced
  midpoints through a sphere of radius Beam Reach (64): the segment is
  tens of units long and 1 / d^2 peaks within a unit or two of the ray's
  closest approach, so the samples mostly stepped over the peak. The
  frame's glow summed to 1.96x the converged value at 12 samples, 0.31x
  at 2 and 0.47x at 64 (probes/q08_missed.py, old numbers), drawn as
  rings.
- **What it does now.** The samples are spaced by ANGLE about the lamp (t
  = tc + D tan(theta)); dt = d^2 / D dtheta cancels the inverse square,
  so the sum is smooth and within a percent at any count. Beam Occlusion
  still tests every sample.
- **Pictures that change.** Point lamp glows (they change by fix 2
  anyway): no rings, a clean hot core at the lamp.
- **Proof.** 2 / 12 / 64 samples against 4096: x1.023 / x1.007 / x1.001
  (old tree, same rig: x0.482 / x1.285 / x0.995); per ray against the
  closed form of the integral of (1 - d^2/R^2) / d^2: max 0.8% (old: off
  by up to 3.7x). Spot and Area beams converge within 2% at 12 samples
  and keep their midpoint march (the banding the module documents).

## 7. An Area lamp's hidden Custom Range cut its beam

- **What was wrong.** An Area lamp draws no Falloff menu and never
  attenuates, but `_light_volumes` still read a Custom Range left on it
  (a converted Point lamp) and cut the beam at Falloff End -- which on an
  Area lamp is the panel's Distance, a gain.
- **What it does now.** The Custom Range reach applies to the lamps that
  attenuate (`... == 'CUSTOM' and kind != 'AREA'`). This also keeps the
  beam whole when the native package exports Distance sqrt(pi/2).
- **Proof.** Area, Distance 3: 1800 beam px under DEFAULT and under the
  hidden CUSTOM, bitwise equal (old tree: 1814 / 916 px). A Point lamp
  keeps the rule (3885 of 12288 px at Custom Range end 3).

## 8. Blend, and the cone dials that are not read

- **What the user saw.** Blender's own Beam Shape > Blend stays live under
  every cone law and under Screen Spotlight, and moves no pixel under
  eight laws.
- **What it does now (UI only).** `ui.spot_cone_notes`: under a law other
  than Blender Internal the panel says "Blend (Beam Shape) is not read by
  this cone law" (and under POV "Hotspot sets the soft edge"); a Screen
  Spotlight says its cone dials are not read. The rows under Cone Law are
  the dials that law reads. Blender's own panel cannot be greyed from the
  add-on.

## 9. Light shafts: the viewport, the cost, the worker pool

- **What the user saw.** A Sun's Volumetric drew nothing in the rendered
  viewport (F12 drew it). With Worker Processes on, F12 dropped the
  shafts too. And each shaft source cost about 1.2 s at 640x480.
- **What was wrong.** `preview.py` `_post` passed the flare sources and
  not the shaft sources the render had computed. `core/parallel.
  render_parallel` returns pixels only, so `scene.last_shafts` was never
  set on the pooled road. `core/post.light_shafts` ran 24 whole-frame
  bilinear resamples per source through 2-D fancy indexing.
- **What it does now.**
  - `core/post.light_shafts` is the same sums in the same order -- F12 is
    bitwise what it was -- computed by `_shaft_march`: the four taps
    gathered with `np.take` through flat indices, each step visiting only
    the window of the frame its samples can find a bright pixel from
    (`_shaft_window`; everything else adds an exact zero), the steps of
    one source resampled on up to four threads and accumulated in step
    order.
  - `light_shafts(..., proxy_pixels=N)` / `post.process(...,
    shaft_proxy=N)`: a box-averaged proxy of at most N pixels is marched
    and the smear upsampled. F12 passes 0.
  - `preview._post` passes the sources on REFINE frames
    (`SHAFT_PROXY_PX = 48000`), never on a draft, and carries the anchors
    through the camera-frame crop (`preview.crop_shaft_sources`, the
    flares' own mapping).
  - `core/parallel.render_parallel` derives the anchors in the parent
    after the join (with the Y-shear flag render() sets).
- **Pictures that change.** The rendered viewport's refine frames of
  scenes with shaft sources; F12 with Worker Processes on. In-process F12:
  bitwise.
- **Proof.** Bitwise against the old stage (kept verbatim in the test) on
  five frames x three source sets and at Shaft Length 1. Timings, one
  source, machine load about 2 (probes/q06_shaft_new.py): 640x480 noise
  1159 -> 453 ms; 1409x577 noise 3158 -> 1863 ms; 1409x577 with a lamp
  pool 3004 -> 78 ms; 640x480 with a bright edge 1069 -> 26 ms. Proxy:
  6000 px on a 76800 px frame, mean error 0.0020 with shafts of 0.751.
  Viewport (headless worker): a Sun's Volumetric moves the refine by a
  mean of 0.1407 (0 before), the frame is render() + post with the
  sources bitwise, a draft carries none, camera-frame mode equals post
  over the cropped render with the anchors moved. Real Blender rendered
  viewport, Sun ahead of the camera: frame mean 0.8121 -> 0.8121 ->
  0.8121 at Volumetric 0 / 1 / 4 before ("did not change"), 0.8121 ->
  0.9419 -> 0.9983 after (16210 and 11270 of 45326 px). Pool: the pooled
  frame is the in-process frame and `last_shafts` matches (it was None).

## 10. A Hemi's shafts, flare and caustics

- **What was wrong.** export turns a Sun with Hemisphere into type
  'HEMI'; `shaft_sources` and `_flare_sources` tested for 'SUN' and sent a
  hemi down the positional branch (the anchor chased the lamp object);
  `_synth_caustic_cookies` skipped the kind, though the panel draws the
  Caustics sliders for it.
- **What it does now.** All three treat HEMI as the directional lamp it
  is; a caustic hemi tiles the web at Caustic Scale world units like its
  Sun.
- **Pictures that change.** A hemi with Volumetric, Flare or Caustics
  above zero.
- **Proof.** Anchors of a hemi at (0,6,6) and at (3,2,3) with one
  rotation: (0.0, 1.572) both, its Sun's (they were (0.0, 0.643) and
  (0.606, 0.231); the frame moved on 5531 px for the shafts, 530 px for
  the flare, now 0). Caustics 0 -> 1 on a hemi: 12231 of 12288 px (0
  before); CPU against GPU twin 4.3e-05.

## 11. Caustic Scale above 8 on a Spot

- **What was wrong.** The bake's 8-cell period exists so a Sun tiles
  seamlessly; a spot never tiles, yet `min(scale, 8)` clamped it. The
  slider runs to 64.
- **What it does now (the verifier's correction).** Above 8 a spot's cone
  is cut from ONE 64-cell lattice at 32 texels a cell up to a 512 tile,
  so the web zooms as the slider moves (a single re-seed at 8; at 8 and
  under the web is the one it was).
- **Proof.** 8 -> 16 -> 32 -> 64 change 2702 / 2801 / 2916 px (0 each
  before). At Scale 4 the tile is the old web times one constant (ratio
  1.39885 everywhere, see 12). Scale 16 is the first quarter of Scale
  32's web: mean difference 0.028 against 0.553 for an unrelated web. CPU
  against GPU twin at Scale 64: 4.8e-05. Real Blender F12: Scale 8 -> 32
  changes 0 of 76800 px before, 3880 after (sheet, row 4: "Scale 32"
  drew the 8-cell web).

## 12. Caustics no longer dims the lamp

- **What was wrong.** `1 + (web - 0.35) * 1.6 * c` pivots on 0.35; the
  web's mean is 0.17. The bake's own comment promised "mean stays near
  1".
- **The choice.** The verifier measured two cures: move the pivot to 0.17
  (adds a constant, so the crest-to-cell contrast drops from 4.6 to 3.2
  at strength 1) or divide by the mean (the same web, uniformly
  brighter). Dividing keeps what a scene was tuned to look like, and it
  stays neutral where the floor at zero bites (strength 1.8 and up), so
  that is what ships: `core/render.caustic_factor` is the old expression,
  `caustic_mean(c)` its mean over a fixed lattice (one constant per
  strength, so nothing breathes as the web writhes), and the tile is
  factor / mean.
- **Existing caustic lamps get brighter by** 1.166x at Caustics 0.5,
  1.399x at 1, 2.030x at 2, 1.570x at 4 (1 / 0.8574, 0.7149, 0.4926,
  0.6370). The pattern does not move. Caustics 0: untouched.
- **Proof.** A Sun's seamless tile averages 1.000 / 0.999 / 0.998 / 0.997
  at 0.5 / 1 / 2 / 4 over four frame times (0.857 / 0.714 / 0.492 /
  0.635 before); the sampled cookie factor over one tile has the tile's
  mean (0.99873 / 0.99873). A floor under a caustic Sun with several
  tiles in view: mean luminance x0.9965 of the plain lamp (x0.7124
  before), crests still over 2x. Real Blender (display-referred means):
  caustic spot x0.838 of the plain lamp before, x0.943 after.

## 13. Cookies

- **The Area lamp's blur (UI).** The projection's focus blur reads
  `Light.shadow_soft_size`, which Blender draws for Point and Spot lamps
  only. The cookie box now draws it for an Area lamp as "Projection
  Blur", and the note names the slider each kind has (Radius / Angle /
  Projection Blur). No engine change. The row shows Blender's own
  tooltip for that property.
- **Spot Size on the GPU road (verifier's new finding).** A spot's cookie
  lookup bakes 2 tan(spot_size / 2) and its atlas carries the size blur,
  but the Spot Size was in `_light_sig` only for a MAP-shadowed lamp. It
  is in the cookie part of the signature now. Warm plan, Spot Size 1.2 ->
  0.7: max |CPU - GPU| 0.330 (no shadow), 0.330 (ray traced), 0.491
  (caustic spot), 0.099 (Radius 0.1) before; 5.3e-06 / - / 1.4e-05 /
  3.1e-06 after. A plain spot's Spot Size drag stays a plan-cache HIT.
- **The image fingerprint (verifier's new finding).** The cookie's plan
  signature, its upload key and the CPU's preparation key were the sum of
  every 7th texel. `core/lights.cookie_fingerprint` is the crc32 of every
  texel (4 ms for 1024x1024 RGBA), remembered on the lamp against the
  pixel array it was taken from, taken fresh once per render() and on
  every plan. Warm plan, a 64 px slide mirrored: max |CPU - GPU| 0.288
  (Spot) / 0.360 (Sun) / 1.609 (Area) before, 4.0e-06 / 5.5e-06 / 1.4e-05
  after. The CPU road had the same hole through its blurred preparation:
  the same lamp with its array mirrored in place rendered the old frame
  (0.000 moved; 0.273 off a fresh lamp); now it is the fresh lamp's frame
  bitwise.
- **Colour and alpha (a missed option, documented).** A coloured slide
  tints per channel ((1, 0.5, 0.25) to four decimals; GPU twin 2e-06).
  The alpha channel is not read (an RGBA slide with alpha 0 on one half
  changes 0 px), and the image multiplies as stored (an 8-bit grey 128
  halves the light: 0.5020, not its sRGB decode 0.2158). The cookie
  tooltip now says both.

## 14. A Screen Spotlight honours Specular Only

- **What was wrong.** The screen-spot block of `core/render.light_surface`
  adds its diffuse lobe before the lamp's diffuse gates and never
  consulted them; `gpu/material._one_light_source` emitted the same
  unconditional add.
- **What it does now.** Both roads add the lobe only when
  `affect_diffuse and not specular_only` (the ordinary lamp path's own
  condition; both flags were plan structure already). The fog lobe -- the
  lamp in the air -- stays.
- **Proof.** Lambert scene: lit sum 13702.6 -> 0.0 with Specular Only (it
  was 13702.6 -> 13702.6); whole GPU road equal to the CPU (0.0); the
  emitted block differs from the plain screen spot's by exactly the one
  `total += hal_sst;` line. Diffuse Only: bitwise the plain ellipse.

## The missed options, measured

| option | result |
|---|---|
| Beam Density / Samples / Falloff / Reach on POINT and AREA beams | all four move both. Density is linear (x2.000, x0). Point: see fix 6. Area: Reach also sets its soft start (5% of Reach), so Reach 4 gives 2.2x the beam sum of Reach 64 -- as designed, reported |
| Area lamp highlight against Size and Shape | by design (BI): the specular is the lamp centre's highlight times the form factor. Half-max box of the highlight on a glossy floor: 56x18 px at Size 0.25 and 1, 64x20 at Size 4, 66x18 for a 4 x 0.5 rectangle, 52x18 for 0.5 x 4, 62x19 disk; CPU = GPU twin to 5e-07. The highlight does not take the lamp's shape. Not changed |
| Area Distance with the hidden Falloff = Custom | fixed (7) |
| flare anchor under the viewport camera-frame crop | works: the flare core lands within 0.2 px of the lamp's projection minus the crop origin for two rects (probes/q11_stereo_crop.py). The shaft anchors take the same mapping (9) |
| effects under a render border | the beam inside the rect is the full frame's bitwise; shaft sources are the full frame's |
| effects under stereo | the beam renders per eye (7918 px mono, 7936 anaglyph, 7918 side-by-side). NOT fixed: flare and shaft anchors are the LEFT eye's in mono-frame ndc, so in a side-by-side or cross pair the flare is drawn once, at x = 210..213 of a 320 px pair where the two halves want it near 106 and 266. Right needs the post chain per eye before the pack |
| effects under Pano Parts | the beam renders in each strip; shaft and flare sources are dropped (0) |
| Film Transparent | the beam adds RGB (up to 0.478 on 1666 px in the probe) where alpha stays 0: premultiplied emission, lost by a straight-alpha file. Same as every other additive stage; reported |
| several effect lamps at once | cookie + beam + Hotspot spot, caustic Sun, caustic spot at Scale 16, glowing point with a flare, area slab: whole GPU road against CPU 4.5e-05, either lamp order |
| cookie colour and alpha | measured and documented (13) |
| cookie and beam on RECTANGLE / DISK / ELLIPSE area lamps, Beam Occlusion under other shadow settings | the verifier's own measurements stand (works); not re-run |
| Sun Angle under Planar shadows and on a Hemi, Square with wide cones under maps, Area size under object scale | the map-soft / ray-road / bake packages' ground |

## Default neutrality

- probes/q10_neutral.py hashes 158 results on both trees -- for 39 lamps
  (every kind x shadow NONE / MAP / RAY, Blend 0 and 1, Square, every
  cone law, GX and GL with a Hotspot they never read, screen spot,
  screen spot + Diffuse Only, Volumetric 1 with the switch off,
  Volumetric 0 with it on, cookies on all four kinds, flares, Negative,
  Only Shadow): the CPU frame before post, the F12 frame after post, the
  emitted GLSL and the GPU twin's frame; plus Sun shafts through F12's
  post (the rewritten march), a 1000 W spot's shafts with the switch off
  at two sizes, and the demo scene under both shadow defaults. 158 of
  158 identical.
- In the test module: the Blender Internal cone at Hotspot 0 is the old
  body bitwise for twelve Blend x Spot Size pairs; texel 2.w is the old
  float32; the emitted block equals the block of a lamp without the
  field; the beam's angular term, edge and cut are the old statements;
  Volumetric 0 / switch off / Caustics 0 frames are the plain lamp's
  bitwise; `light_shafts` is the old stage bitwise.
- Real Blender: the base spot and the base point frames are 0 codes apart
  between the trees.

## Two roads

- Changed GLSL-visible cases against the CPU, all under the deferred bar
  of 6e-3: Hotspot frames 2.5e-06 .. 5.4e-06 (and the standalone
  statements at 0); caustic hemi 4.3e-05; caustic spot at Scale 64
  4.8e-05; cookie keys warm 3.1e-06 .. 1.4e-05; screen spot + Specular
  Only 0.0; five effect lamps at once 4.5e-05.
- The beam, the shafts, the flares and the caustic bake are single CPU
  passes shared by both devices; the whole GPU road (fakedevice) equals
  the CPU with a beam to 1.6e-05.
- Signature: Hotspot rides texel 2.w (not in the signature; only "a band
  exists" is, as before); the cookie's Spot Size and fingerprint are in
  it because the plan bakes them.
- Strict GLSL gate (tests/glsl_strict): green. `test_m` compiles 14
  planned passes of the changed blocks under strict GLSL 4.30.

## Tests

`tests/test_r254_spot_beam.py`, 16 functions, 115 checks (114 without
HALCYON_GLSLANG). On the frozen old tree 40 pass (the neutrality pins and
the options that already worked) and 63 fail with the old numbers, at
least one in every fix's function (logs:
`.../wt/spot-beam/logs/final/r254_new.log` and `r254_on_base.log`).

Existing tests run green with no pin changed: tests/test_r253_light_options
(195 checks), tests/test_r254_falloff (152), tests/test_r251_lighting_lamps
and tests/test_r251_lighting_fog (their only failure is the missing
1.89.0 zip of baseline_fails.txt), tests/test_r253_region, and 27
functions of tests/test_render.py (cones, volumes, flares, caustics,
gobos, the light texel cache, the worker pool, test_no_setting_lies,
test_every_setting_does_what_it_says, the GLSL strictness pair).

## Not done here, on purpose

- sun-angle-map-01, wide-spot-map-01, ring2-map-reach-01,
  hemi-cube-bake-01, screen-spot-map-bake-01, area-power-01: other
  packages.
- Stereo side-by-side flare / shaft anchors (above): needs the post chain
  per eye.
- An animated caustic lamp still re-plans on the GPU every frame (its
  cookie's pixels are in the signature); the auditor's observation, not
  a finding.
- The flare still does not read the lamp's Power (as coded).

## For the integrator

- **Shared function.** `core/render._light_volumes`: this package changed
  exactly two statements -- the reach condition (`if
  LI.resolved_decay(light, st) == 'CUSTOM' and kind != 'AREA':` plus its
  comment) and the composite line (`add += scatter[:, None] * col *
  CONES.beam_gain(light)`). The flags package edits the same function
  (flags-volume-beam-01); nothing else in it moved.
- `core/render.shaft_sources` gained the "beam lamps are not sources
  while beams are drawn" rule and the HEMI branch; `_flare_sources` the
  HEMI branch; the screen-spot block of `light_surface` (and of
  `gpu/material._one_light_source`) the diffuse gate. If the flags
  package rewrote the flag conditions of the ordinary lamp path, the
  screen spot should take whatever condition it settled on.
- `preview.py`: only `_post` (three statements), `SHAFT_PROXY_PX` and the
  module function `crop_shaft_sources`.
- area-power-01 (native package): the Area beam needs nothing from it --
  it reads `decay_end`, so `Distance = sqrt(pi/2)` gives the watts-scale
  beam, and the hidden Custom Range no longer cuts it.
- `properties.LABELS` renames four labels (identifiers unchanged);
  `RANGES['spot_cone_density'] = (0, 100000)` clamps a negative Density
  (never a designed feature) to 0 on load.
- README: the beam section should say Volumetric Lights, the unit, and
  that stereo pairs keep mono flare / shaft anchors.

### CHANGELOG fragment

- **Hotspot works under the default cone law.** It is the inner angle of
  the Blender Internal cone (full strength inside, the smoothstep out to
  the Spot Size) and replaces Blend while above zero; it was read by POV
  alone. A value on both roads. Spots that carry a non-zero Hotspot under
  the default law change.
- **Volumetric beams are on the lamp's own scale.** Spot and Point beams
  are 632 times dimmer at the same numbers (multiply Beam Density by 632
  to restore one); Volumetric 0..4 is now a working range at Blender's
  lamp powers. An Area lamp's beam follows its Distance (1.98 times
  brighter at Distance 25). The beam follows the Cone Law, Hotspot and
  exponent; a Point lamp's glow no longer depends on the sample count; an
  Area lamp's hidden Custom Range no longer cuts its beam.
- **One lamp, one volumetric effect.** With Volumetric Lights on, a Spot,
  Point or Area lamp draws its beam and no longer also throws
  screen-space Light Shafts; with it off (and for a Sun) the value is the
  shafts' strength, as before.
- **"Spot Cones" is "Volumetric Lights"**, and the lamp panel shows the
  switch when it is the reason there is no beam; a Sun gets a live
  status line for its shafts.
- **Light Shafts draw in the rendered viewport** (refine frames) and
  survive Worker Processes; the stage is 1.7 times cheaper on a frame
  that is bright everywhere and up to 40 times on a lamp pool, and
  bitwise the same in F12.
- **A Hemi's shafts, flare and caustics follow its direction.**
- **Caustics keeps the lamp's light** (existing caustic lamps are 1.17x /
  1.40x / 2.03x / 1.57x brighter at strength 0.5 / 1 / 2 / 4, same web),
  and Caustic Scale 8..64 works on a Spot.
- **Cookies:** an Area lamp's Projection Blur is in the panel; the GPU
  road follows Spot Size and every edit of the image; alpha is not read
  (documented).
- **A Screen Spotlight honours Specular Only.**

## Probes

In `/tmp/claude-0/-home-user-Blender-Halcyon-Engine/1c5e906d-b1ee-5f5f-923e-25adb13f5e76/scratchpad/wt/spot-beam/`
(`probes/` headless, run with `python3 qNN_*.py`; `HE_PKG=<pkg dir>`
points the `he_common` rigs at another tree; `bl/` real Blender):

- q01_beam_unit.py, q02_area_beam_unit.py -- the unit, the candidates
- q03_hotspot.py -- the hotspot on both roads
- q04_beam_law.py -- the beam under each law
- q06_shaft_new.py -- the march against the old stage, timings, proxy
- q07_effects.py -- hemi, caustics, cookie keys, screen spot (out/q07_old.txt,
  out/q07_new.txt)
- q08_missed.py, q11_stereo_crop.py -- the missed options
- q10_neutral.py -- the 158-hash battery (out/q10_old.json, out/q10_new.json)
- bl/run_f12.sh, bl/run_vp.sh, bl/analyze.py, bl/make_sheet.py -- real
  Blender F12 and the rendered viewport, before and after
