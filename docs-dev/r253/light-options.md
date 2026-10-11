# R253 -- light-options (More Light options: eight period lamp dials)

Branch `r253/light-options-v2`. The integrator owns CHANGELOG.md, README.md,
EXTENSION_LISTING.md, version.py, `__init__.py` and blender_manifest.toml;
the fragments below are written in their voice. Version stays 1.91.0 in
this worktree.

## CHANGELOG

- **What was asked.** More per-lamp dials in the Halcyon Light panel, in
  the period spirit of the existing ones. The audit against the code found
  most of the candidate list already shipped (falloff start/end, spot halo
  and beam occlusion, the cookie on every lamp kind with its filter and
  extension, sun angle and point/spot radius soft shadows, the area shapes,
  diffuse / specular / ambient-only lamps and light linking, BI shadow
  colour, negative, only-shadow, per-lamp flares and caustics, the hemi
  flag, the 3ds Max decay Start and LightWave's linear falloff, the R252
  per-material light limit); eight were missing and are built here, every
  one a new `Light` field (core/scene.py) that is bitwise inert at its
  default -- `tests/test_r253_light_options.py::test_a_identity_at_defaults`
  renders the demo, gobo, negative, soft, area and hemi scenes bitwise the
  1.91.0 tree under every pre-existing FALLOFF item, and proves the GLSL
  the plan emits at the defaults carries no R253 token at all.
- **(A) Square spot** (`Light.spot_square`, panel "Square"). Blender
  Internal's `LA_SQUARE`, verbatim from 2.79 shadeoutput.c
  `lamp_get_visibility`: the view vector in lamp space, `x = max(|x/z|,
  |y/z|)`, `inpr = 1/sqrt(1 + x^2)`, zero behind the lamp
  (`core/lights.square_cos`, float32 one op per statement, the lamp frame
  from `cookie_frame` so a projected slide fills the square corner to
  corner). `spot_falloff` swaps the cosine before the cone law, so every
  law -- BI blend, GL11, POV, the six GX functions -- gets a square version.
  The shadow buffer widens to the square's diagonal (`_build_shadow_maps`,
  `tan(size/2) * sqrt(2)`); the flag is in `_light_shadow_signature`. GPU:
  the `sq_*` statements after `cosang` in `gpu/material._one_light_source`,
  the lamp's side / up axes on hal_lights texels 10 / 11; the arithmetic
  tail is bitwise through the GLSL front-end (`d == 0.0`), the full block
  with its two `dot()`s within 2e-6, the frame at the deferred bar 6e-3.
  The 2.79 importer stops warning "square spot cones are not imported":
  `blend279_map.lamp_map` carries `square`, `legacy_import._apply_lamp_bi`
  sets `hs.spot_square`. The volumetric beam of a square spot stays ROUND
  (core/cones.py marches a cone; BI's own halo was round too) -- named in
  the tooltip.
- **(B) Specular Colour** (`Light.specular_color`, white). OpenGL 1.1's
  `glLight GL_SPECULAR` / Direct3D 7's `D3DLIGHT7.dcvSpecular`: a second
  colour per light, here a MULTIPLIER on the lamp colour for the classic
  specular lobes only (`core/render.light_surface`: `rad_s = rad * tint`
  in the two `scontrib` lines; the sheen term, the ramp ENERGY luminance
  and the cel roads CARTOON / ANIME keep the plain `rad`). GPU: texel 8.rgb,
  `vec3 rad_s = rad * hal_ltex(8).rgb` substituted into the two
  `hal_scon +=` lines ONLY when the tint is not white (structure in
  `_light_sig`), so an old scene's text is the 1.91 text. A tint drag is a
  plan-cache HIT, on / off a MISS; the red channel of a (1, 0.2, 0.2) tint
  is bitwise the untinted frame (x 1.0 is exact).
- **(C) Hemi ground colour** (`Light.hemi_ground`, black). Rost, OpenGL
  Shading Language (2004) 9.1 Hemisphere Lighting, `mix(ground, sky,
  0.5 + 0.5 N.L)`, split into the sky term (today's BI wrap) plus a
  GROUND term weighted `1 - wrap`, at the lamp's sign x energy x style
  gain, NO cookie on the ground (the slide is a projector, the ground a
  bounce -- the sky term keeps it). A floor facing away from the hemi
  shades exactly `diffuse * level * ground * energy / pi`; a face toward
  it is bitwise untouched; a negative hemi negates both terms. GPU: texel
  9.rgb x texel 0.w, one `hal_dcon +=` line in the HEMI path when the
  ground is non-black (structure); the hemi frame at 6e-3.
- **(D) Nominal-distance decay** (`FALLOFF` gains `NOMINAL_INVERSE` and
  `NOMINAL_INV_SQUARE`, appended at the END; `core/lights.DECAY_LAWS_R253`).
  `(D/d)^n` with D = Falloff End (the panel relabels it "Nominal Distance"
  under the two modes): 3ds Max 3's Decay Inverse / Inverse Square with
  Start R0 (`Lum = (R0/R)^n`), LightWave 6/7's Inverse Distance /
  Inverse Distance^2 with Range. Exactly 1.0 at d = D (float32 D/D), 0.5
  and 0.25 at 2D, brighter inside, finite at the lamp (d floored at 1e-6);
  the Sphere clamp composes outside it as for every law. `decay_law_glsl`
  emits the same two or three statements -- bitwise standalone, the frame
  at the bar; the Scene Default road (`light_falloff_default`) reaches it
  on both devices. Every pre-existing FALLOFF position is pinned.
- **(E) Near Attenuation** (`Light.decay_near_start / decay_near_end`,
  0 / 0). 3ds Max's Attenuation Parameters rollout: the light FADES IN
  from nothing at Start to full at End (the far pair is today's Custom
  Range); active only while End > Start; never on SUN / HEMI / AREA (Max's
  directional lights carry no attenuation; the area lamp's distance speaks
  only through its form factor). Max's manual draws the ramp and names no
  curve: it is LINEAR here, an assumption named in the tooltip.
  `core/lights.attenuate` lost its NONE early return so the shared tail
  (Sphere wrap, then `near_attenuation`, then the BI 0.001 snap in
  `sample()`) sees every curve -- the NONE + Sphere arithmetic is the same
  statements and is pinned bitwise. GPU: four statements after `att` is
  final and before the snap, Start on texel 8.w, the span on 9.w --
  bitwise standalone, the frame at the bar; the pair being live is
  structure, the distances a texel (HIT).
- **(F) Shadow buffer clip** (`Light.shadow_clip_start / shadow_clip_end`,
  0 / 0). Blender Internal's la->clipsta / la->clipend as the map's near
  and far planes (`core/lights.shadow_clip_planes`, SPOT buffer and the
  POINT cube alike); 0 keeps the automatic plane bitwise (the same python
  expressions), an End inside Start falls back to the automatic far plane.
  A ClipEnd short of the floor drops the floor's shadow (930 px brighter,
  none darker on the probe scene). The pair is in `_light_shadow_signature`
  (one lamp's map rebuilds, the other rides the cache). No GLSL: the GPU
  shadow meta reads near / far / vp off the built ShadowMap, so the
  clipped map is one texture on both devices (pinned: meta == map). The
  2.79 Lamp parser now reads `clipend` beside `clipsta`
  (`core/blend279.py`), `lamp_map` carries `clip_start / clip_end` for a
  buffer-shadowed spot only, `_apply_lamp_bi` sets them when the shadow
  is MAP.
- **(G) Cookie Offset / Tiling / Angle** (`Light.cookie_offset /
  cookie_tile / cookie_rotation`, (0, 0) / (1, 1) / 0). 3ds Max's
  projector-map Coordinates rollout, the 2.4x lamp texture's ofs / size: a
  rotation about the slide's centre, then the tiling, then the offset, on
  the cookie uv of every lamp kind right before the lookup
  (`core/lights.cookie_xform`, float32 one op per statement, applied ONLY
  when `cookie_xform_on`); the CLIP gate and the focus blur stay in texture
  space after it. GPU: `_cookie_function` opens with the fourteen `ck_*`
  statements as literals (the cookie road's convention: side / up /
  strength are literals too) before the CLIP test, the values in the cookie
  plan signature and the cookies meta (`'xform'`). A 90-degree Angle swaps
  the stripe axis, Tiling 2 doubles the stripes, Offset 0.5 slides the
  slide by half; parity under the gobo bar for SPOT, SUN, POINT and AREA
  under REPEAT / BILINEAR and CLIP / CLOSEST.
- **(H) Light Style** (`Light.light_style` NORMAL, `light_style_rate` 10).
  Quake (1996) QuakeC world.qc's twelve worldspawn lightstyle strings
  (`core/lights.LIGHT_STYLES`, 'a' = 0, 'm' = normal, 'z' = double),
  advanced `rate` characters per second of `scene.time` as R_AnimateLight
  did (`floor(time * 10) % len`). Quake's own step is 22/256 per letter
  (its 'm' is 1.03); Halcyon normalises 'm' to EXACTLY 1.0 so NORMAL is
  bitwise inert. `apply_light_styles(scene)` writes the runtime
  `Light.style_gain` before the shadow maps build (and on ShadeJob
  construction for the plan rigs); every energy reader multiplies by it:
  `lights.sample`, `ambient_light_split`, the screen spot (render.py and
  fog.py's lobe), the beams, the lens-flare intensity (`_flare_sources`,
  which never read the energy, so the gain lands on the flare dial) and
  the GPU packer -- `pack_light_texels` scales the SAME python double
  before `np.float32(sign * energy / 4pi)`, so the energy texel is
  bitwise the CPU's scale and a flicker frame is a plan-cache HIT. A
  FAST_STROBE lamp at time 0.1 ('a') is bitwise the lamp at energy 0 at
  every site.
- **The texel texture.** `gpu/material.LIGHT_TEXEL_STRIDE` 8 -> 16: texel 8
  = (specular_color.rgb, near start), 9 = (hemi_ground.rgb, near span),
  10 / 11 = the SPOT frame's side / up, 12-15 reserved zeros. Every reader
  goes through `_lref` and the symbol, so texels 0-7 move no bit (the R251
  spot-law, decay-law and screen-spot pins still read them where they were).
- **The panel** (`ui.HALCYON_PT_light`): Specular Color and Light Style (+
  Style Rate, active off NORMAL) under Energy; "Nominal Distance" for the
  two new falloffs and Near Attenuation Start / End for non-hemi lamps in
  the decay block; Square after Hotspot; Shadow Clip Start / End (active
  under MAP, "the map's near and far planes; 0 = automatic") for Spot and
  Point; Projection Offset / Tiling / Angle under the cookie; Ground Color
  under Hemi. Every property carries a >= 40-char tooltip, every
  LIGHT_STYLE / FALLOFF item a >= 12-char line (audited by the new module;
  test_tooltips_everywhere audits node props only). `export.export_light`
  reads every field through `getattr` with the dataclass default, so an
  older fake hs (the gobo test's) and `halcyon=None` export the inert lamp.
- **Nodes.** No node is added; `export.NODE_PROPS` is untouched.
- **Stand-ins and open items.** The Near Attenuation ramp is linear (Max
  states no curve). Max's Decay R0 is its Start field; Halcyon reuses
  Falloff End as the nominal distance, as POV and GX already do. The
  square spot's beam stays round. The ground term ignores the cookie by
  design. Specular Colour leaves the cel roads, the sheen and the ramp
  ENERGY input on the plain lamp colour. The identity pin needs the 1.91.0
  zip beside the package (`halcyon-1.91.0.zip`, a `git archive` of the
  pre-feature tree): without it the test prints a note and runs the
  in-tree proofs only (the R251 modules fail loudly on a missing zip; this
  one does not, since the suite's own environment lacks it).
- **Tests.** `tests/test_r253_light_options.py` (registered after the R253
  compositing block in `tests/run_all.py`): ten tests, 210 checks -- the
  identity (19 bitwise frames against the 1.91.0 tree, the attenuate
  restructure, the token-free GLSL, the stride), and per option the law,
  the A/B, the standalone GLSL (bitwise where stated), the frame twin at
  6e-3, the plan HIT / MISS structure, the import and the export; plus the
  light-panel tooltip audit, the dataclass / PropertyGroup default
  agreement and the panel drawing every dial for every lamp kind under a
  recording layout. Nine rows join `tests/featurematrix.py` (ROWS, lights
  section, append-only). Constraining tests re-run green:
  test_r251_lighting_lamps (every check but the pre-existing missing
  1.89.0 zip), test_r251_shadow (same), test_light_texel_cache,
  test_light_edit_fast_paths, test_gobo_every_lamp_kind,
  test_gobo_gpu_parity_kinds, test_gobo_extend_and_filter,
  test_gobo_focus_blur, test_hemi_light,
  test_hemi_is_a_proper_hemisphere_lamp, test_bi_lamp_math,
  test_bi_lamp_loop_tail, test_lighting_is_exposed_sanely,
  test_light_sizes_actually_work, test_spot_cones_match_a_brute_force_march,
  test_area_lamp_shapes, test_tooltips_everywhere,
  test_every_node_prop_reaches_the_renderer, test_period_lens_flares,
  test_light_volumes_layer_and_stop, test_no_duplicate_definitions,
  test_every_version_stamp_agrees, test_no_bl_info_dependency,
  test_legacy::test_lamp_world_mapping.

## README

**Eight more lamp dials (1.92)**, each from a named period source and each
silent until you touch it: Blender Internal's **Square** spot (the cone
becomes a pyramid, every cone law included, the shadow buffer widened to
the corners; a 2.79 file's square spots now import square); a per-lamp
**Specular Color** (OpenGL's `GL_SPECULAR`, Direct3D 7's `dcvSpecular`)
that tints the highlight and nothing else; the hemi's **Ground Color** (the
two-tone hemisphere of Rost's *OpenGL Shading Language*, 9.1); two more
falloffs, **Inverse** and **Inverse Square from Nominal Distance** (3ds
Max's Decay with Start R0, LightWave's Inverse Distance with Range: exactly
full strength at the Falloff End, brighter inside); 3ds Max's **Near
Attenuation** Start / End fade-in; Blender Internal's **ClipSta / ClipEnd**
on the shadow buffer; the projected slide's **Offset / Tiling / Angle**
(3ds Max's projector-map Coordinates, the 2.4x lamp texture's ofs / size);
and Quake's twelve **Light Styles** -- the torch flicker, the candles, the
strobes, the fluorescent tube -- stepped ten characters per second of scene
time, on the surfaces, the beam, the flare and the fog lobe alike. Every
dial has a GPU twin (the laws bitwise in the simulator, the frames at the
deferred bar) and rides the lamp-edit fast path: a value drag re-uploads a
texel, only a flag re-plans.

## Blender-only

- The Light data panel's layout in the Properties editor: the property
  split, the sub-column activity (Style Rate off NORMAL, the clip pair under
  MAP, Ground Color under Hemi), the "Nominal Distance" relabel, the ANGLE
  subtype on Projection Angle, the COLOR pickers for Specular Color and
  Ground Color -- fakebpy records property keywords and the recording
  layout only the prop names.
- A real 2.79 .blend with a Square spot and ClipSta / ClipEnd appended
  through the legacy importer / append watch (headless covers `lamp_map`,
  the fixture DNA and `_apply_lamp_bi` on a fake namespace, not a live
  append).
- Lamp-drag responsiveness in the live viewport: toggling Square / a tint /
  a ground re-plans once; dragging the near distances, the tint, the ground,
  the clips and a flickering style across frames is a plan-cache HIT plus a
  texel upload (headless pins the plan identity, not the wall time).
- The real driver compiling the new GLSL statements -- the square-spot
  block, the near-attenuation lines, the cookie transform, the hemi ground
  line -- in a frame with eight lamps at stride 16 (the simulator and
  `gpu/device.duplicate_definitions` / `lint_declaration_order` prove the
  text; `tests/glsl_strict` needs a glslang binary, absent here).
- Light styles over a real animation playback (`scene.time` from
  `frame_current / fps` in export.py), the flares and beams flickering in
  sync, and the viewport's frame-change invalidation of the light texels.
- The visual read of a square spot's widened shadow buffer at the field's
  map sizes (corner coverage, no clipped corners) and of a ClipEnd cutting
  a receiver mid-floor.
