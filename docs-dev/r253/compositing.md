# R253 -- compositing: more render passes for Blender's compositor

Branch `r253/compositing`. The version stays 1.91.0 in this worktree; the
integrator owns the stamps, CHANGELOG.md, README.md and the listing.

## CHANGELOG

- **What was asked.** "More Compositing Features and Types": more of the
  buffers Blender's compositor can read beside Combined. 'Compositing' in
  Halcyon is the Passes panel (`ui.HALCYON_PT_passes`), the engine's
  `update_render_passes` / `PASS_SPEC` and `core/render.build_aux_passes`;
  the Composite Video chain (`HALCYON_PT_composite`, `core/signal_*`) is
  the NTSC/PAL scan-out stage and is untouched.
- **Eighteen new passes, Blender Internal 2.79's own names** (`render_
  result.c`: `Mist`, `Env`, `Diffuse`, `Spec`, `Ambient`, `Emit`,
  `Shadow`, `AO`, `Color`), so a comp built for BI drops in exactly as
  the six existing data passes follow Blender's (`Depth`, `Normal`,
  `Position`, `UV`, `IndexOB`, `IndexMA`). Eleven new settings, all
  `False` by default (`core/settings.py` `pass_mist`, `pass_environment`,
  `pass_beauty`, `pass_diffuse`, `pass_specular`, `pass_ambient`,
  `pass_emission`, `pass_shadow`, `pass_ao`, `pass_color`,
  `pass_lights`); `pass_lights` opens eight fixed slots
  `Light00..Light07` (`core/render.LIGHT_PASS_SLOTS`), the first eight
  selected lamps in lamp order. `core/render.wanted_passes` reads each
  field by name; `light_pass_names` is the one predicate `render()`, the
  GPU plan, the capability table and the engine share; `engine.PASS_SPEC`
  declares every name (Mist 1ch `Z` VALUE, Beauty 4ch `RGBA` COLOR, the
  rest 3ch `RGB` COLOR); `properties.LABELS` / `DESCRIPTIONS` carry a
  label and a tooltip of 40+ characters each; the Passes panel gains
  Frame and Light columns and a corrected note.
- **Three frame passes, bitwise twins by construction.** `Mist` is BI's
  mist (`shadeoutput.c mistfactor`) as the scene Fog curve
  (`core/fog.legacy_curve`: LINEAR / EXP / EXP2 / TABLE16 on Fog Start /
  End / Density, the dials `legacy_import` maps BI mist onto) on the
  camera distance -- 0 near, 1 at the sky, read from the curve whether or
  not Fog is drawn (`build_aux_passes`). `Env` is the world where the
  camera saw it: the frame at uncovered pixels, black under geometry,
  snapshotted in `render()` right after the shadow mask and BEFORE the
  wires, ink, halos and volumetric lights draw over the sky. `Beauty` is
  the linear frame `engine.render` hands to `post.process`, delivered
  beside the finished Combined (palette, dither, CRT and signal stages
  untouched). All three come off the CPU-reconstructed G-buffer and the
  frame itself on both roads (`gpu/craster.raster_into_gbuffer`
  reconstructs the GBuffer; the GPU shading burst only writes the frame),
  recorded as `capability.FEATURES['render_passes']` = BOTH, measured 0.0
  by construction in the fake-device matrix row `aux passes (frame: mist
  + env + beauty)`.
- **The BI light split, read from `light_surface`'s own accumulators.**
  `ShadeJob.pass_sink` (one `(rh, rw, 3)` float32 plane per wanted name,
  allocated by `render()` for whole-frame roads) is written by
  `shade_batch` at the opaque frame's camera fragments only
  (`pass_sink_armed` around `_shade_all`: never a ray hit, a transparent
  layer, a vertex corner or the SSS point pass). `light_surface` fills
  `extras['split']` from SEPARATE arrays fed the very `dcontrib` /
  `scontrib` `out` receives (after vis, shadow colour and lit_mask,
  before the per-lamp clamp) at the ANIME, generic, cel-key, screen-spot
  and Only-Shadow sites; `Ambient` is `out` before the first lamp (the
  ambient term, radiosity, AO, the prelight); `AO` the
  `ambient_occlusion()` factor captured from the ONE existing call (white
  when AO is off); `Emit` the emission as it joins; `Color` the diffuse
  times its level; `Shadow` 1 lit / 0 dark averaged over the lamps that
  shadow (2.79 `shade_only_shadow`'s accum / ir shape), reading the `vis`
  each lamp already traced and 1.0 where `need` was False -- zero extra
  rays, the existing skip logic untouched. The tracked road (RESULT
  ramps, SSS, world exposure) reports the rewritten `diff_acc` /
  `spec_acc`; the non-separable tails (cartoon paint, cel bands and hair
  shine, fixed-shade mixes, the GX / DS saturation) report
  `out - Ambient - Spec` as Diffuse so the identity holds; vertex- and
  face-rate materials (`_shade_interpolated`) report their WHOLE lit
  colour as Diffuse (and Light00), their albedo as Color. Colour passes
  (Env and the split) resolve through `_resolve`, the CPU's own AA
  kernel, so **Diffuse + Spec + Ambient + Emit + Env == the linear beauty
  at every pixel through the supersample filter** (measured 2.4e-7); the
  data passes keep the top-left-sample rule (`DATA_PASSES`). Adaptive AA
  refine passes average the colour passes with the frame
  (`_adaptive_refine`), so the identity holds at `aa_mode ADAPTIVE` too.
- **Device story.** Any light-component pass puts the frame's shading on
  the CPU BY NAME: `gpu/shade.plan_frame` returns `light-component
  passes (<names>) accumulate on the CPU: the GLSL loop keeps no per-lobe
  sums`, printed per frame by `render()` as `[Halcyon GPU] shading on
  the CPU: ...` and recorded in `LAST_GPU_VERDICT`; the eleven flags join
  `_plan_sig` so a cached valid plan cannot walk past the refusal (the
  R78 lesson, pinned through a warm cache). `capability.FEATURES
  ['light_passes']` = NOT_YET names the GLSL twin (MRT, one target per
  lobe); `scene_features` adds `light_passes` and `BLOCKING` lists it, so
  `capability.plan`'s notes and the device panel say "Light Passes -- CPU
  for now". Both devices draw the same bits (matrix row `aux passes
  (light split)`).
- **Bitwise-neutral at the defaults, proven.** Every pass on vs every
  pass off is `np.array_equal` on the demo scene, on a variant with
  traced soft shadows + ambient occlusion + raytracing, and on BI
  materials that already share the extras dict (a RESULT ramp,
  Transparency > Specular, a Shadows Only catcher). A non-None extras
  dict is neutral by itself: every reader keys off a surf flag or a key
  only written when its own flag asked.
- **Out of scope, by name.** Vector / motion (no previous-frame
  transforms are exported), Cryptomatte (the manifest contract; IndexOB /
  IndexMA + ID Mask cover masks), Ink and Halo passes (the composites
  return only the edited frame), DiffInd / GlossInd (radiosity replaces
  the ambient term and is reported inside Ambient), Reflect (traced
  reflections join the beauty only). The identity holds for the classic
  models with Light Clamp 0, Fog off, no traced reflections and no
  Gamma-2 sample blend; Shadow reads 1.0 where a lamp would not light
  the point (BI's `shade_only_shadow` traced every point); IndexOB stays
  the export order (unchanged, two R251 tests depend on it); the
  per-lamp sink costs eight full-frame planes at render resolution.
- **Stale claims corrected.** The `debug_pass` comment in
  `core/settings.py` (DIFFUSE | SPECULAR | AMBIENT | SHADOW were never
  DEBUG_PASS items), the engine's `VIEWLAYER_PT_layer_passes` note and
  the Passes panel note (Mist and the light components ARE produced now;
  Vector and Denoising Data are the remaining reasons the panel stays
  hidden).
- **Tests.** `tests/test_r253_compositing.py` (11 tests, 107 checks):
  `test_light_passes_sum_to_the_beauty` (aa 1 and 4), `test_light_passes_
  are_bitwise_neutral` (5 scene variants + the defaults),
  `test_shadow_pass_content`, `test_mist_pass_content` (LINEAR exact, EXP,
  Fog on == off), `test_ao_and_color_passes`, `test_per_light_passes_sum_
  to_the_lamps`, `test_rate_shaded_and_adaptive_frames_keep_the_identity`,
  `test_light_passes_refuse_the_gpu_by_name` (warm cache, the GPU device
  bit for bit with every pass buffer), `test_pass_panel_and_table_text`,
  `test_beauty_pass_is_the_linear_frame` (fake engine, EGA),
  `test_passes_survive_the_sub_render_roads` (panorama, parallax stereo,
  accumulate). Updated: `test_render_passes_reach_blender` (every
  `pass_*` field by introspection, 17 fields / 24 names), `test_the_
  engine_actually_runs` (Beauty, Mist, Diffuse through fakeblender); two
  new `featurematrix.ROWS`.

## README

**Render passes.** Beside Combined the engine writes, on request,
Blender Internal's own pass set for the compositor's Render Layers node:
the data passes Depth, Normal, Position, UV, IndexOB, IndexMA and Mist
(the scene Fog curve on the camera distance, 1 at the sky, whether or not
Fog is drawn), the frame passes Env (the world where the camera saw it)
and Beauty (the linear frame before the palette, dither, CRT and signal
stages), and the light split Diffuse, Spec, Ambient, Emit, Shadow, AO and
Color -- named exactly as BI 2.79 named them, so a comp built for Blender
Internal drops in. Diffuse + Spec + Ambient + Emit (+ Env at the sky) is
the linear beauty for the classic models, through the supersample filter
and the adaptive AA refine; Shadow is 1 lit / 0 dark averaged over the
lamps that cast; AO is the occlusion factor itself; Per-Lamp Passes adds
Light00..Light07, one per selected lamp in lamp order. The data and frame
passes come off the same G-buffer and frame on both devices, bit for bit;
any light pass shades the frame on the CPU by name (the GLSL loop keeps
no per-lobe sums -- the console says so, the device panel reads "Light
Passes -- CPU for now"). Every pass defaults off: an old scene renders
the same bytes. Vector, Cryptomatte, Ink and Halo passes are not
produced; the compositor's ID Mask node on IndexOB / IndexMA covers
masks.

## Blender-only

- `RenderEngine.register_pass` accepting the new names and channel ids
  (Mist 1ch 'Z', Beauty 4ch 'RGBA', Env / the split 3ch 'RGB', the eight
  LightNN) and the Render Layers node showing one socket per registered
  pass.
- `rect.foreach_set` sizes for 1-, 3- and 4-channel passes on a real
  RenderResult (fakeblender's FakePass only mimics the shape).
- The Image Editor's pass dropdown showing Beauty as untouched linear
  colour next to a palette-quantised Combined under an EGA preset.
- The `[Halcyon GPU] shading on the CPU: light-component passes (...)`
  line in the real console on a GPU-device F12 with a driver present
  (headless, the device probe refuses first and the verdict carries the
  probe's reason; the plan's refusal is pinned through `plan_frame`
  directly), and the device panel row "Light Passes -- CPU for now".
- Per-lamp pass memory at production resolution and AA (eight
  `(rh, rw, 3)` float32 planes) on a real machine.
- Material preview renders (`is_preview`) with passes enabled on the
  scene -- `update_render_passes` is called with the preview scene and
  must not throw.
- Animation renders: passes delivered every frame with the plan cache
  warm across frames; the Beauty buffer's `post.fit_to` under a border
  or a resized output.
