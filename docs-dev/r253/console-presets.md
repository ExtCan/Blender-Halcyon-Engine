# R253 -- console-presets (Render Presets for the consoles that got shaders)

Branch `r253/console-presets-v2`, written against the `claude-cloud` tip
(1.91.0 + the six merged R253 features). The version stays 1.91.0 in the
worktree; the integrator bumps it once for the round and re-sums the
preset count (121 on this branch) if another feature adds presets.

## CHANGELOG

### Render presets for the consoles that got shaders (R253)

- **What was asked.** "One-click looks for every machine the Console
  Emulation Shader emulates." The 1.91.0 node resolves to 27 engine
  models, but the preset shelf (`presets/library.py`) predated it: the
  PSP, Namco System 22, the Super FX, the Wii, the Naomi, RenderWare on
  its three platforms and the GTA-era PS2 look had no preset at all, and
  several console presets still named a stand-in model (JAGUAR ->
  GOURAUD, THREEDO -> FLAT) or a light count the node's own
  `core/console.LIGHT_LIMIT` contradicts (PSX 4 vs the GTE's 3, PS2 8 vs
  the VU1 microcode's 3, N64 4 vs the RSP's 7).
- **Nine new presets** (112 -> 121), each a complete look with its
  sources and its stand-ins named in the note: **PSP** (CONSOLE: 480x272
  at 2x, PS1_MODULATE per vertex for the GU's modulate function, four
  lights in order, RGB565 (GU_PSM_5650), Bayer 4x4 standing in for
  sceGuSetDither's own matrix, sceGuFog's linear near/far -- the sceGu
  reference); **NAMCO_S22** (ARCADE: 640x480, S22_MODULATE per vertex,
  one parallel light, point-sampled textures (Super System 22 added
  bilinear), 24-bit, Linear fog standing in for MAME namcos22's czram
  per-polygon-z depth-cue table); **SUPERFX** (CONSOLE: 256x224 at the
  SNES's 8:7 pixel, SUPERFX_PLOT per face, 4 bpp over the new
  **SUPERFX16** fixed palette at BITS_5, gamma 1.0 and colour management
  NONE so the plot's gates in `core/combine.plot_gate` pass and the plot
  draws on both roads; the SNES preset itself is untouched -- its
  adaptive 256-entry palette cannot pass the gates, pinned by
  `test_r251_post_palette`); **WII** (CONSOLE: the GAMECUBE dict at the
  Wii's numbers -- 640x480 progressive with an ANAMORPHIC 16:9 pixel
  (4:3 pixel aspect, visible in Blender's Format panel), GX_ANISO_4,
  1024 textures and shadow maps, the deflicker copy, the XFB 4:2:2 --
  the GX API reference / Dolphin); **NAOMI** (ARCADE: the DREAMCAST dict
  at 24 bits with no dither and no supersampler (31 kHz VGA), VQ textures
  to 1024, PVR128 fog kept; the eight lights are CPU-lit, as the SH-4
  did, disclosed); **RENDERWARE_PS2** (CONSOLE: the PS2 dict at 512x448
  FIELDS into PS2_CT16 with DTHE, RENDERWARE_PS2 per vertex, max_lights 0
  (RpWorld has no hardware light count: 0 = no limit), linear fog
  (RwCameraSetFogDistance), no bloom, no shadows, VU1's near reject; the
  generic 512 width is a stand-in for any one title); **RENDERWARE_GC**
  (the GAMECUBE dict with RENDERWARE_GC, linear fog without the height
  layer, no shadows, max_lights 0); **RENDERWARE_XBOX** (the XBOX dict
  with RENDERWARE_PC per vertex, no bloom, no shadows, linear fog,
  max_lights 0); **GTA_PS2** (CONSOLE: RENDERWARE_PS2 at 640x448 FIELDS
  with `crtc_blend 'PREVIOUS_FRAME'` at `crtc_alpha 128` -- the GS PMODE
  RC2 / ALP circuit behind GTA III / Vice City's "Trails" option, visible
  over an animation; a warm linear fog, 256 textures from Q; the note
  names the node's Specular = Plugin for San Andreas vehicles).
- **The existing console presets corrected, nothing else moved**: PSX
  and PSX_HIRES `max_lights` 4 -> 3 under `light_limit_mode 'FIRST'` (the
  GTE's three-light colour matrix, LLM / LCM -- nocash), PS2 8 -> 3
  FIRST (the common VU1 microcode's three directional lights), N64 4 -> 7
  FIRST (gbi.h's seven directional lights plus ambient), GAMECUBE / NDS /
  SEGA_MODEL2 / SEGA_MODEL3 / MEGA_DRIVE / D3D_RETAIL_1997 gain FIRST at
  their existing counts; JAGUAR `default_model` GOURAUD -> JAGUAR_CRY (the
  blitter's intensity-only Gouraud exists since R252; the note now
  describes GOURD mode instead of disclaiming it) and THREEDO FLAT ->
  THREEDO_PIXC (the cel engine's n/8 multiplier; shading_rate FACE kept,
  the model is rate-fixed there). These are preset-VALUE changes, not
  default changes: a saved scene keeps its settings; re-applying those
  presets renders identically on the two-lamp demo (the limit never
  bites; the plot of every changed preset is still proven distinct from
  the default frame) but will differ on a scene with more lamps than the
  limit, or with Jaguar / 3DO materials that have no model of their own.
  Every R251 pin on the shelf is re-asserted in one place
  (`test_old_shelf_unchanged_where_pinned`).
- **The `console_node` hint.** A console / arcade preset may carry a
  fifth, optional top-level key: the `HALCYON_ConsoleShaderNode`
  property values (only `core/console.ENUM_PROPS` / `SCALAR_PROPS`
  names) that reproduce the preset's `default_model` on the node --
  `presets/library.console_hint(key)` returns a copy or None,
  `CONSOLE_HINTED` is the set of keys carrying one (27: the nine new and
  18 existing; System 21, Model 1, the SNES and the handheld bitmaps
  carry none). `apply_preset` never reads it, so a preset with the hint
  renders exactly as one without. It reaches Blender in three places:
  `ui.HALCYON_OT_apply_preset` points the scene's Convert to Game
  **Console** menu (`convert_console`, UI state, never a render setting)
  at the hint's machine; `ui.HALCYON_PT_presets` shows "Game conversion:
  Sony PlayStation / Gouraud, textured" under the note; and the Game
  convert block (`ui._draw_convert_target`) grows a fourth **From
  Preset** button (icon IMPORT, this material only) with a "Preset:
  <machine / type>" line whenever the active preset carries a hint --
  the convert-buttons counts hold at the default preset. The operator
  (`convert.HALCYON_OT_convert_to_node`) gains `from_preset`; set, it
  reads the hint into the machine and into `choice['console_props']`,
  which `core/convert.plan_for`'s CONSOLE branch merges after
  `props['console']` (which stays the first key) and before the
  conversion's own decisions (the vertex-colour switch wins over a
  hinted type); undeclared keys are dropped and `console` itself never
  overrides the menu's.
- **SUPERFX16** (`core/palette.SUPERFX16`, `FIXED_PALETTES['SUPERFX16']`,
  `properties.PALETTE_MODE` item "Star Fox 16 (Super FX)" appended AFTER
  EHB, positional): a 16-entry STAND-IN for Star Fox's 4 bpp polygon
  CGRAM (no published dump is cited; the note and the tooltip say so),
  luma-ordered, every entry on the SNES's 5-bit lattice so BITS_5 snaps
  the table to itself and the plot's pairs land on the post stage's own
  entries. The CPU palette stage and the GPU PALETTE stage pick it up
  generically through `post._palette_for` (no GLSL change); the
  `test_r251_post_palette` EHB-is-last pin becomes "EHB keeps index 10,
  SUPERFX16 follows".
- **GPU story.** A preset is a dict of settings; nothing new is computed
  per pixel. Every model the presets name has its R252 road: the nine
  new presets run through the GLSL simulator in
  `test_console_presets_on_the_gpu` and their shading (model, light limit
  at the vertex / face rate, fog) is the CPU's at the corner-road bar
  (max < 6e-3, no pixel > 1e-2 -- measured 1e-7 .. 2.6e-5) once the
  raster / frame stages the shading simulator does not run are
  neutralised BY NAME, each with its own R251 road: the GPU rasteriser's
  painter's sort (SUPERFX), its depth encodings GC_14E2 / W_FIXED and the
  INTEGER_D3D pixel centre (WII, RENDERWARE_GC, RENDERWARE_XBOX), the
  composite's framebuffer formats PS2_CT16 / GC_RGBA6 (CPU on both roads
  by construction, as the capability table states), the edge pass after
  shading (the PS2 family). The Super FX plot is drawn in GLSL as the
  CPU's twin under the preset (gates pass) and refuses by name ("the plot
  writes palette entries, and gamma 2.2 ...") when the user raises gamma,
  the CPU shading that material as FLAT. The post chain's depth / palette
  stage (`gpu/chain.quant_refusal`) reports no refusal for any of the
  nine; SUPERFX16 is a GPU palette stage for free. GTA_PS2's
  PREVIOUS_FRAME blend and FIELDS, the Wii's deflicker and 4:2:2, are the
  R251 stages unchanged (on a single still the CRTC blend reads BGCOLOR
  and says so). Default neutrality: `RenderSettings()` is untouched (no
  new field), `PRESETS['DEFAULT']['settings']` stays `{}`, the enum item
  is appended, so an old scene's indices and frame are what they were
  (proved field-for-field in `test_console_shelf_exists_and_is_well_formed`).
- **Tests.** `tests/test_r253_console_presets.py` (113 checks): the shelf
  (the nine keys, categories, real settings, unique labels, the 121
  pin, the optional key, default neutrality), the hints' coherence
  against `core/console` (declared names, legal values, `resolve` ->
  the preset's model, the light counts under FIRST, the machine's rate
  with the named exemptions SATURN / D3D_RETAIL_1997 / XBOX, all 18
  machines covered, the RenderWare zero), the CPU frames (every new and
  changed preset finite and distinct from the default; the plot's gates,
  `effective_model`, every post pixel on the palette, the Ball's render
  pixels palette entries before post), the GLSL simulator (above),
  SUPERFX16 (shape, lattice, enum position, `reduce_depth` membership,
  the GPU stage), the hint through `plan_for` and the headless
  operators (Apply Preset PSX -> convert_console PS1, INFINID_4 leaves it;
  Convert to Game From Preset on WII -> GAMECUBE / LIT / light limit on,
  on a hintless preset the menu's machine), the presets panel and the
  Game block on a recording layout (four operators with a hint, three
  without, the R253 convert-buttons fixture unchanged), and the R251
  pins re-asserted. Constraining tests re-run green:
  `test_preset_library_is_well_formed`, `test_all_presets_render`,
  `test_presets_do_not_overlap`, `test_presets_never_touch_the_device_switch`,
  `test_sixth_generation_console_presets`,
  `test_the_preset_menu_never_floods_the_console`,
  `test_tooltips_everywhere`, `test_every_setting_has_a_detailed_tooltip`,
  `test_max_shelf_completion` (121), `test_every_version_stamp_agrees`,
  `test_blender_layer_imports`, and the modules `test_r251_post_palette`
  (two pins updated with the reason), `test_r251_lighting_fog`,
  `test_r251_post_signal`, `test_r251_texture`, `test_r251_transparency`,
  `test_r251_raster`, `test_r251_raster_wire`, `test_r251_post_tape`,
  `test_r251_post_codec`, `test_r252_console`, `test_r253_convert_buttons`.
  `tests/run_all.py --images` adds SUPERFX, GTA_PS2 and PSP to the
  contact sheet (now four rows).

## README

**Render presets for every console that got a shader (1.92).** The
preset shelf now has a look for each machine the Console Emulation
Shader emulates: the **PSP** (480x272, the GU's four lights, RGB565 with
the GE's dither), **Namco System 22** (Ridge Racer's point-sampled
640x480 and its depth cue), the **Super FX** (Star Fox's 4 bpp plot into
a fixed sixteen-colour palette -- the new *Star Fox 16* palette mode),
the **Wii** (the GameCube at 480p with an anamorphic 16:9 pixel, 4x
anisotropy and 1024 textures), the **Naomi** (the Dreamcast's PowerVR2
at 24 bits on a VGA monitor), **RenderWare** on the PlayStation 2, the
GameCube and the Xbox (Criterion's prelit pipelines, linear fog, no
hardware light count) and a **GTA-era PS2** look with the CRTC's
previous-frame "Trails" blend. The older console presets now carry
their machines' real light counts (the PlayStation's three, the PS2's
three, the N64's seven), the Jaguar preset shades with the blitter's
intensity-only CRY Gouraud and the 3DO preset with the cel engine's PIXC
multiplier -- the models the Console Emulation Shader brought. Each of
these presets also remembers which machine it is: applying one points
the Material panel's Convert to Game **Console** menu at it, the presets
panel says "Game conversion: Sony PlayStation / Gouraud, textured", and a
**From Preset** button next to the three Game buttons converts this
material to the preset's machine, shader type and light limit in one
click. 121 render presets.

## Blender-only

- `HALCYON_OT_apply_preset.execute` writing `scene.render.resolution_x/y`,
  `resolution_percentage` and the pixel aspect for the new presets (the
  Wii's 4:3 pixel, the PSP's 480x272 at 2x), and the `property_unset`
  reset road accepting `'SUPERFX16'` as a `palette_mode` enum value in the
  real RNA (the headless fake does not validate enum membership).
- `hs.convert_console` taking the hint's machine after Apply Preset in a
  real scene (the EnumProperty setter), the presets panel's "Game
  conversion: ..." label and the Game block's "From Preset" button drawn
  in the real layout (icons rendered, wrapping, enabled state, the
  Preset line's width).
- Convert to Game "From Preset" on a real material: `rw_platform` /
  `gc_type` / `light_limit` set through `setattr` on the registered
  `HALCYON_ConsoleShaderNode` (the `console` update callback firing
  `refresh_sockets`), the machine's unused sockets hidden, and the
  material rendering under the preset in the viewport.
- The Super FX plot on a real driver (the GLSL text compiled by the GPU,
  not the simulator) with the SUPERFX16 palette texture in the PALETTE
  stage, and the CUSTOM-without-image road staying refused by name on
  the driver.
- GTA_PS2's PREVIOUS_FRAME CRTC blend over an animation (the trails) in
  the viewport and in a rendered sequence; FIELDS on real frame pairs.
- The Wii preset's anamorphic pixel aspect in Blender's camera frame and
  the final 16:9 output; the PSP preset's 480x272 with the 2x pixel scale
  in the image editor.
- Saved .blend files from 1.91.0 whose `scene.halcyon.preset` is any old
  key still resolving (the enum is static and every old id survives); a
  file saved with `palette_mode EHB` keeping its index after SUPERFX16 is
  appended.

## Deviations from the design

- The GPU test cannot assert "twin or refusal" on the raw preset through
  `_sim_vs_cpu`: that helper compares the simulated SHADING passes
  against `R.render`, which also runs the raster / frame stages
  (painter's sort, depth encodings, pixel centre, framebuffer formats,
  the edge pass) that have their own GPU roads or named CPU stages but
  are not part of the shading simulation. The test neutralises exactly
  those stages BY NAME (listed with their road) and then requires the
  twin; the capability table's statement that framebuffer formats run
  in the composite on the CPU on both roads is asserted from the source.
  Measured: every new preset is a twin at 1e-7 .. 2.6e-5 once neutralised.
- The hint coherence test's rate rule exempts three presets by name
  (the design named SATURN only): D3D_RETAIL_1997 shades FACE because
  FLAT_D3D_FIRST is rate-fixed at FACE (`core/combine.RATE_FIXED`), and
  XBOX keeps its BLINN per pixel (the design's own "default kept").
  VOODOO / VOODOO2 need no model exemption: GOURAUD is a console model
  and PC_FIXED / GOURAUD resolves to it.
- The "every rendered pixel before post is a palette entry" check is
  made on the untextured Ball (and the Box under a z-buffer): the
  textured Floor keeps its texel over the plotted pair (the combine's
  modulate, as the R252 plot already behaves) and the sky is not a
  polygon; the post-stage check covers every pixel as designed.
- `tests/run_all.py`'s contact sheet grows to as many rows as the picks
  need (the design added three picks to a fixed 3x3 sheet).
- README.md is not edited (impl_common.md: the integrator owns it); the
  paragraph and the 112 -> 121 counts are above.
- `convert.HALCYON_OT_convert_to_node.from_preset` applies only when the
  target is CONSOLE (the design implied it; made explicit).

## Limitations

- SUPERFX16 is a stand-in palette (documented in the note, the tooltip
  and the module); so are the System 22 fog curve, the PSP dither matrix,
  the RenderWare PS2 width and the Saturn preset's FACE rate against the
  node's per-vertex Machine rate.
- On a single still GTA_PS2's PREVIOUS_FRAME blend has no predecessor
  and reads BGCOLOR (printed by `core/signal_era`); the trails exist only
  over a sequence.
- The light-limit corrections (PSX / PSX_HIRES / PS2 / N64) move no pixel
  on the two-lamp demo, so their effect on a many-lamp scene is proved
  only by `core/lights.select_lights`' existing tests, not by a frame
  here.
- The convert-buttons `test_panel_and_scene_props` source-count pins
  (`op.model = _model` x3, `convert_to_bi` x3) hold; the Game block's
  recorded count is unchanged at the default preset because
  `_draw_convert_target` reads `hal.preset` through `getattr` and draws
  the fourth button only for a hinted preset.
