# Halcyon — roadmap

## Where the road has been

**1.30.0 — the GPU port, complete.** Every feature either runs on the GPU
(proven on real hardware against the CPU frame), refuses by name (the frame
still renders correctly on the CPU and the console says why), or is impossible
by construction and documented as such. The proof is the FEATURE × DEVICE
MATRIX in the self test — 116 rows, rendered by both devices and diffed. The
field verdict on the release hardware (RTX 5060 Ti, Vulkan):

    116 rows: 114 raster+shade on the driver and matched,
    2 partially routed to the CPU by name and matched, 0 FAILED

**1.31–1.52 — the look rounds and the public release.** The halo energy
kit, the gradient family, weather, the terrain generator, the
image-as-palette road, the Bryce 1995 `.mat` decode (all 46 library presets
from the stored numbers), the lit infinite grounds, the metric-honesty
rounds, and the first tagged release with its documentation rewritten
against live counts.

**1.53–1.62 — the Blender Internal arc and the Anime Shader.** Any
2.79-or-earlier .blend appends with its Blender Internal materials rebuilt
one to one (the BI Material node, the ported BI texture engine, lamps at
their 2.79 energies, the old world), a plain File ▸ Append fixing its lamps
automatically; the Anime Shader with its game-texture compatibility modes;
116 resolution presets.

**1.63–1.69 — the stylized road.** Per-material ink and marked edges, the
Shadow Ramp road, real volumes (eight scattering models, mesh containers,
voxels, smoke grids), round area lamps, the Add-menu shelves.

**1.70–1.73 — the cel arc.** The ink style pack, the Cartoon Shader (paint,
not light, six eras), the 80s additions to the Anime Shader, and the era
looks: film stocks, weave, dust, grain, flicker, holds on twos and threes.

**1.74–1.80 — the 40s arc,** in the order chosen from the Imitation Study:
the drawn line, the 2D media, the painted backdrop and the painted
background road, the inker's line, media 2, the colour process (the
three-strip camera, the timed print, impure dyes, registration, the silver
key), and the print's wear.

**1.81–1.84 — the cel's light, the Xrd study, the preset expansion, the
hair pass.** Camera and world keys, screen shadows, the depth rim; Arc
System Works' line control and the SDF face shadow; the Anime Style menu
and fourteen Cel & Film presets; the hair shine's shapes; 140 resolution
presets; a written tooltip on every option, suite-enforced.

**1.85–1.86 — the Max study.** 3ds Max's material and map library under a
3DS Max submenu; then, from the field's own Max 2012 shader library, every
map rebuilt on Max's own algorithm on Halcyon's own tables (no Autodesk code
or data ships), the eight Standard shaders as master models, the Standard
and Raytrace materials as nodes 1:1 with their rollouts, the last maps, the
object frame on the GPU, and a 3ds Max 2012 render preset. 1.86.1 fixed the
icon that stopped 1.86.0 enabling, and taught the suite Blender's icon enum.

**1.87 — Sparking! ZERO from the game's own export.** The SPARKING
compatibility mode rebuilt on the field's FModel export and its
reconstruction of the master material (the tone strip read down its height
by the half-Lambert cosine, the floor-to-white tone, the Mask1 multiply), a
material importer for the exported .json instances, the Cel Sparking! ZERO
Skin template, Separate/Combine Color on the GPU, the colour layer by name
on both devices. 1.87.1 read the field's own render of it: the colours are
now taken as exported (Unreal's linear values — the swatch the artist
chose — not the reconstruction's 2.2 on top of them), the game's outline
shell is recognised and made invisible (Halcyon inks), the highlight
parameters are carried but not lit, the shared instances are found under
the export's Common folder (1.87.2: and rebuilt on every import). 1.87.2
also closed the viewport's unshaded-black material at its root: the GPU
plan now carries a pass for every material the mesh has, a cache hit that
lacks one on screen re-plans, and a pixel no pass wrote refuses the frame
by name.

**1.88 — the ink pass on the GPU.** The remaining CPU passes of a GPU frame,
measured first: the outline bucket was 514 ms of a 720p frame (the chamfer
237, the crease test 193) against under 150 for everything else. The line
now draws as fragment passes on the G-buffer the shading uploaded — seeds,
the mask roads as one diamond search, the chamfer as an iterated relaxation
exact within the band, the style maths dial for dial, boil and pencil, the
composite — bitwise the CPU line on the field frame; the CPU road itself
halved (the crease test's reduction, the chamfer on one 64-bit key); the
nearest-seed rule made order-free and the near-silhouette rule made
explicit on both roads. Left for 1.89: the sky / background (70 ms), the
film and post stages (60 ms), the cel field, and the frame kept on the GPU
between them without a readback.

**1.89 — the frame stays on the GPU.** Measured first at the field's own F12
shape (1280×720 under CEL_ANIME_MODERN, 2× supersampled inside): after the
ink, a GPU frame's CPU time was the sky (179 ms — 109 at 1× for a flat
colour world, the camera rays and their normalisations), the supersample
resolve (107), the post (the grain 68, the display 27, the bit depth 15)
and 112 ms of copies and masks between them. The sky now draws as the last
draw of the shading's own burst (the solid sky, the node world's flat colour,
sky blend and environment image, a plain Background node, the gradient, the
bands, HDRI; covered pixels discard, so the readback is the whole frame),
the frame stays on the GPU between the stages (the ink reads it in place,
the resolve draws it at output size, the post chain inherits it; any CPU
stage that edits it releases it by name), and the post chain runs resident
— no upload when the render's frame is inherited (one more after each
active CPU-only stage that hands the frame back), one readback at the end,
the grain (to 1.2 px) with
the flicker and the bit depth without a dither as stages of their own. In
the simulator every new road is bitwise the CPU (the grain within 1.2e-7);
the driver's numbers are the field test's. Left: the Bryce and painted
skies (2.8 s and 1.9 s a frame on the CPU — the biggest single win
remaining), the starfield and physical skies, the film beyond grain and
flicker, the cel field, the ink's depth upload.

**1.90 — the period features, on both roads.** The field's ask: "Add some
more 3D software/console period features, as many as possible that works
with both GPU and CPU modes". Measured first, from 1.89's own ledger: a
fogged GPU frame still paid about 550 ms of CPU fog over the readback, so
the round's first item was the fog itself — it now runs inside the deferred
material pass (the four curves, the per-vertex rounding, the bands, the
height layer), the frame stays resident under it, and only materials whose
traced or environment composites land after the readback keep the CPU's
fog, by name. On that road, 109 mechanisms in nine packs, each from a named
machine or package: shadows (4: the midpoint map, planar shadows, Dreamcast
modifier volumes, DS shadow polygons), lighting (23: the fog tables and
cues of the PlayStation, Voodoo, PowerVR2, DS, Direct3D, GameCube, System
21, Model 3, LightWave and POV-Ray; the lamp laws of OpenGL, POV-Ray and
GX; the GX, Sega and DS light units; POV-Ray's finish), materials (15: the
period combiners in thirteen models, five node features, the REYES rate),
raster (14: pixel centres, near rejection, the ordering table, five depth
encodings, vertex formats, jitter, the DS rear plane, N64 coverage AA,
LightWave's and Blender 2.41's sample blends, the Elite and vector-beam
wires), textures (12: texel formats, the TMEM budget, block compression,
coarse weights, GL_CLAMP, the chroma key, POV-Ray's and 3D Studio's
filters, the mip roads of the N64, the Voodoo, the Riva and the PS2),
transparency (10: the consoles' blend equations and framebuffer formats,
DS translucency, fuzz, the N64 noise compare, the PS2's two-pass alpha,
Blender 2.4x's Zoffs, Max's thin wall, Imagine's fog object), sky and
camera (8: Y-shear, Doom's cylinder sky, LightWave's backdrop, Mode 7, Pano
parts, Virtual Boy stereo, lens-pass depth of field, time-slice blur),
palette (10: the palette snap and the ordered dither on the GPU, register
depth, Extra Half-Brite, CRY, YJK, attribute cells, per-scanline palettes,
Super Black, the video legaliser) and signal (13: the N64 VI, the GameCube
copy filter, the PS2 CRTC, the 3dfx filters, the 3DO and GBA stretches,
tape, cables, the PAL receiver, chroma siting, MPEG-1, Smacker, Tron's
matte glow). 22 presets (112, with an Arcade Boards category), 17 shading
models (49), six nodes (249 evaluated), 105 settings (396), the feature
matrix from 116 rows to 299, fifteen test modules, 9503 checks.
Three refusals lifted on the way: the ordered dither and the palette snap
draw on the GPU (the old DITHER stage, never wired, retired), and Painter's
algorithm rides the GPU rasteriser because a named tie rule made it
order-free. Every default is bitwise the 1.89.0 engine, rendered from its
zip; every twin is held to the CPU in the simulator and through the fake
device. On the driver: the run of record on the final tree, after fix pass
4 (2026-10-05; `docs-dev/r251_field4_report.txt`, per row in
`docs-dev/r251_field4_verdicts.txt`, against the earlier run in
`docs-dev/r251_field4_vs_field3.txt`), 212 variants at 1280x720 — both
waves' 206, four error-diffusion controls and two frames of the field's own
export — in seventeen fresh processes (sixteen of thirteen, one of four),
none died. In all 212 the GPU device's first frame rasterised on the GPU;
in the earlier run it was served the CPU's raster in 188 of 208 (the cache
item under "Open after 1.90"). 43 rows have no pixel over 1e-6; 34 sit
within one 8-bit level at 1 to 28 of 921,600 pixels and 13 within two
levels at 2 to 4; 56 have the parked shadow-map compare at (76, 688) as
their only pixel over 1e-2, with 1 to 28 over 1e-3 (446 and 440 under an
HDRI world); 60 have something else, in 58 of them 1 to 42 pixels over 1e-3
and 1 to 8 over 1e-2 (the other two: 120 and 42 on the VHS preset, 200; 39
and 38 on the field frame at ss 2, 212); 6 diverged. Three of the six are
presets whose dither is Floyd-Steinberg error diffusion — ATARI_ST 351,922
pixels, AMIGA_OCS 67,372, Summed Area x4 on STUDIO_R4 354,017 — with the
display already on the CPU by name ahead of the dither: that routing is
active on the driver and did not make the frames agree (the first item
under "Open after 1.90"). The other three are signal stages over the bars
their rows state: the RF modulator over composite (23,542 pixels over 1e-3,
795 over 1e-2, max 3.53e-2; bar 4/255), MPEG-1 intra blocks (4,774 and 63,
max 1.41e-1 at the parked pixel; bar 1/255) and the VIDEO_CD preset (9,266
over 1e-3, none over 1e-2, max 8.32e-3; bar 1/255). New with the GPU
rasteriser engaged: a few pixels differ by large amounts, read as
rasteriser edges (the second item under "Open after 1.90"). Of the second
wave's 88 variants (119-206): 16 with no pixel over 1e-6, 12 within one
level, 8 within two, 21 on the parked pixel alone, 27 others, and 4 of the
6 diverged. Read back to the CPU by name in those rows' post lines: the
display ahead of the error-diffusion dither (176), the per-scanline
palette's colour depth (190), interlace (126, 173, 185, 194) and the NTSC
cable, refused (200); in ten rows the frame left the GPU at the framebuffer
format before post (126, 139, 166-168, 170-173, 193); the RENDERMAN
preset's two rows (137, 158) print the sky and the post on the CPU and take
as long on the GPU device as on the CPU device, and the report does not
print why. The self test's feature matrix on the driver (96x72, the first
process's copy): 299 rows, 293 matched, 4 routed to the CPU by name and
matched, 2 FAILED; the earlier run printed 291 and 4 FAILED, two of them
the N64 coverage defect, fixed in fix pass 4. The wave-1-only run of
2026-09-27 is superseded: presets were edited after it. Left: everything
under "Open after 1.90" below.

## Open after 1.90 (disclosed)

Wrong, undecided or unverified as the round stands — none of these is
fixed:

- **Error diffusion gives different pictures on the two devices; the
  decision is the user's and is open.** On the driver three presets whose
  dither is Floyd-Steinberg error diffusion differ between the CPU and
  GPU devices: ATARI_ST at 351,922 of 921,600 pixels (max 1.00), AMIGA_OCS
  at 67,372 (max 0.533), Summed Area x4 on STUDIO_R4 at 354,017 (max
  0.484). On the CPU's raster the earlier run counted 289,708, 364,620 and
  312,123: the count moves with the frame beneath the dither. In all three
  the post line shows the display on the CPU by name and no post stage on
  the GPU. Controls on the driver (207-210): ATARI_ST with the dither off 0
  pixels, with BAYER4 0 pixels; AMIGA_OCS with the dither off 1 pixel
  (2.00e-1 at (37, 789): the next item); STUDIO_R4 with the dither off and
  Summed Area x4 kept 4 pixels (max 3.23e-2). So neither the summed-area
  footprint nor the palette is the cause. The reading: the GPU's shading
  differs from the CPU's in the last bits (2.3e-5 on the self test's shaded
  frame on this driver) and error diffusion amplifies that into a different
  dither pattern, so under such a dither the devices give different,
  equally dithered pictures. That breaks the determinism doctrine, holds
  for any error-diffusion preset, and predates this round (never measured
  before). To decide: render such frames on the CPU by name, or keep the
  GPU's speed with the difference printed. Since fix pass 4 the routing's
  console line says the pattern still differs.
- **The GPU rasteriser's edges differ from the CPU's at a few pixels of a
  720p frame; cause not diagnosed, open.** That is the reading; measured is
  this. In the run of 2026-10-05, the first in which every 720p row
  rasterised on the GPU, ten rows that were 0 pixels on the CPU's raster
  differ: nine at 1 to 6 pixels (107, 119, 135, 137, 139, 158, 190, 206,
  209) and the VHS preset (200) at 128, 120 over 1e-3 and 42 over 1e-2.
  Variant 119 (D3D_RETAIL_1997) is one pixel, 1.90e-1 at (414, 573), with
  the sky alone and the post chain alone both 0. The field frame at ss 2
  (212) went from 3 pixels (2 over 1e-2, max 7.06e-2) to 39 (38 over 1e-2,
  max 9.80e-1 at (359, 11)). The same places recur: (37, 789) is the
  maximum in eight rows (18, 46, 49, 127, 135, 202, 206, 209), (431, 572)
  in three (87, 92, 107), (414, 573) in two (31, 119). The console prints
  the raster tie referral replaying pixels with the CPU's arithmetic, and
  these are not among them. The rasteriser and its referral predate this
  round; in the earlier field run only the 20 Painter's rows rasterised on
  the GPU, because the G-buffer cache served the CPU's raster to the other
  188 (the cache item below). It is not one of this round's features. It
  breaks the determinism doctrine at those pixels, and the GPU rasteriser
  is on by default on the GPU device. What would close it: a wider
  referral window or an exact edge function, and a 720p driver row per
  release with the cache cleared.
- **DS Highlight whitens every lit surface at the default toon table**
  (61% of the test frame pure white; a narrower band or more steps do not
  help, the generated table always tops out at full). The hardware rule is
  undecided between GBATEK (as built: modulate by the table entry, add the
  entry) and melonDS (modulate by the vertex red as grey, add the entry);
  the arithmetic waits for that decision (the tooltip now says the default
  table saturates), and the catalogue cites melonDS for a rule it
  does not support.
- **The chroma key does not reach** the texel formats whose alpha is not
  the source's (I8 and the N64's I4: intensity; Model 2's I4: its white
  marker), block compression after an alpha-less format (the conversion
  runs first: neighbours move, VQ can lose holes) and NCC (the table is
  fitted before the key); no preset combines them. Under formats with no
  alpha plane the key works since fix pass 3, headless; the driver's two
  rows on it (variants 167, 168) are 0 pixels between the devices, which
  shows the devices agree, not that a cut-out was keyed.
- **The display-before-error-diffusion routing is active on the driver
  and does not close the difference** (the first item): the three rows
  print it by name, draw no post stage on the GPU, and still differ.
- **The wider cascade class is not routed**: the matte glow and film grain
  before the display, and the legaliser, Super Black, chroma siting, MPEG-1
  and Smacker between the display and an error-diffusion colour depth,
  still draw on the driver. The full run shows routing them by name would
  not be enough: with no post stage on the GPU at all the three rows
  still differ. It waits on the decision in the first item.

- **Two preview-collage rows move no rgb pixels** (the Mega Drive column
  mesh and the N64 noise compare: alpha-plane features compared on a scene
  where the colour is identical). The rows should show the alpha plane or a
  scene where the colour moves. Also weak: the TEV Combiner Stage row and
  its matrix row compare white with white (the scene's registers
  saturate; the emitter's twin is tested without saturation elsewhere),
  and the Emboss Bump and Env Chrome rows have not been judged.
- **The texture cache is keyed by `id()` of the pixel array** — the same
  line as in 1.89.0, so it predates the round. A stale prepared texture was
  measured once, headless (worked around in the collage tool only); whether
  it can occur inside Blender is not known.
- **Blender dies silently after a few dozen 720p variants in one process**
  (twice in the driver test, no crash file; each variant passes alone). The
  test runs in fresh processes of thirteen (the run of record's seventeen
  all finished); the cause — VRAM growth in the
  pools or cached uploads across plans is a guess — is not found.
- **The feature matrix prints 2 FAILED rows of 299 on the driver** (the
  self test at 96x72): Summed Area filtering (0.2784 at 1 pixel, a whole
  texel at one texel edge; its 720p row shows 23 pixels with 2 over 1e-2)
  and Floyd-Steinberg error diffusion (0.0323, 724 pixels: the first
  item). The first run printed four: the two N64 coverage rows were a
  defect (no coverage plane under the GPU rasteriser, so the filter was
  skipped on the GPU device), fixed in fix pass 4 and rerun at 0 pixels.
- **The earlier run's 720p rows shared the CPU's raster; the run of
  record's do not.** The G-buffer cache key carries no device, so in 188 of
  the earlier run's 208 rows (2026-10-04) the GPU runs were served the CPU
  run's G-buffer; only the 20 Painter's rows rasterised on the GPU. The
  field script now clears the cache before the GPU runs and prints the
  raster's device on every GPU line: in the run of 2026-10-05 the first GPU
  frame of all 212 rows rasterised on the GPU (3 to 375 ms, median 32), and
  frames 2 and 3 were served that G-buffer in 192 rows (in the other 20
  every frame rasterised on the GPU). The matrix clears the cache itself
  and was the rasteriser's evidence in both runs. The cache key itself
  still carries no device.
- **Three signal rows diverged on the driver, over their own stated
  bars**: the RF modulator over composite (the post chain alone: max
  1.96e-2, 23,507 pixels over 1e-3, 787 over 1e-2; bar 4/255), MPEG-1
  intra blocks (the post chain alone: max 1.18e-2, 4,712 over 1e-3, 3
  over 1e-2; bar 1/255) and the VIDEO_CD preset (max 8.32e-3, 9,266 over
  1e-3 and 921,589 over 1e-6; bar 1/255). Not investigated.
- **The RENDERMAN preset gets no GPU speed on the driver** (variants 137
  and 158: sky and post on the CPU; 5696-5760 and 26,831-26,979 ms warm
  on the GPU device against 6024 and 28,210 on the CPU device). The
  report does not print the reason.
- **Field-script labels the driver does not meet.** Rows labelled bitwise
  that are not 0 pixels, among them Painter's on the GPU raster (20
  pixels, 3 over 1e-2), the three planar-shadow rows (7 to 22 pixels at
  one 8-bit level) and the PS1 / N64 vertex-format rows (17 to 20, one
  of them with 1 over 1e-2); the two midpoint shadow-map rows, at 2 and
  5 pixels over 1e-2 against a stated bar of 6e-3; and the matte-glow
  row, whose post chain alone is one 8-bit level off at 5 pixels against
  a stated bar of 5e-4. Reword the label or fix the row: not decided.

- **No edited preset is compared with its 1.89.0 self by the suite.** 39
  of the 90 shipped presets have edited settings, 22 are new, 51 are
  unchanged (`docs-dev/r251_presets.log`, line 1); the identity pin covers
  only the 51. `tools/r251_preset_sheet.py` renders the comparison on the
  demo scene at 320x240 on the CPU road — a look, not a test, not a driver
  measurement. Six edited presets move no pixel there (MAX_2012, MAX_R2,
  STUDIO_R4, POVRAY_2, POVRAY_31, SATURN) and have not been rendered on a
  scene where their edit could show. Needs: a pin, or a stated reason per
  preset.
- **A preset's Default Model reaches only materials that name no model**
  (`core/render.py:6337`, `material_model`: Force Model, the material's
  own model and a Halcyon node's model come first). Seven presets changed
  it this round (DREAMCAST, GAMECUBE, N64, PS2, PSX, PSX_HIRES, SATURN);
  SATURN's change moves 0 px on both test scenes, whose materials all name
  a model. How Blender materials reach this rule in a real scene was not
  read. Needs a reading of the exporter, then a decision.
- **Two preset choices await the user:** DOOM now sets Y-shear (the
  projection of a pitched camera changes: 31,022 of 76,800 px against
  1.89.0 on the demo scene); VIRTUAL_BOY now renders a side-by-side stereo
  pair with parallax layers (48,393 px), and renders grey in both
  versions, not red.
- **Register-depth palettes are built, then snapped**
  (`core/post.py:323-325`), so entries collide: AMIGA_OCS 32 -> 24
  distinct colours, ATARI_ST 16 -> 11 on the demo frame. Build in register
  space, or dedupe and refill: undecided.
- **Three preset keys of the design are left out on purpose**, each with
  what would make it right. GAMECUBE `fog_range_adjust`: inert at the
  preset's VERTEX rate (pinned bitwise); the hardware fogged per pixel
  under per-vertex lighting, so a per-pixel fog under the VERTEX rate
  would make it live. POWERVR_PCX2 `tex_format 'RGB5550'`: one global
  alpha-less format turns every cut-out opaque; the card chose per
  texture (5:5:5 opaque, 4:4:4:4 with alpha), so a format picked by each
  texture's own alpha would. D3D_RETAIL_1997 `fog_depth 'Z'`: the Fog
  switch then fogs nothing at scene-unit Start/End; the user's decision,
  and an advisory that also reaches the viewport would soften it.
- **CPU cost of the palette and cell roads is measured at 320x240 only**
  (three consecutive frames, a busy machine): a first-frame cost that did not repeat on the next two frames (not measured:
  whether it returns when a cache is evicted or a scene changes)
  of about 0.67 s for CRY16, 0.5 s for VGA256 with BAYER4, 0.9 s in the
  render for the Dreamcast VQ rows, 0.3 s in post for SNES / NEO_GEO /
  SEGA_32X; and per frame about 570 ms for C64 multicolour cells, 190 for
  MSX1 cells, 127 for MSX2, 124 for TURBO_SILVER, 95 for ZX_SPECTRUM, 30
  for VGA256 with BAYER4. At 720p the earlier driver run (run 3) gives the
  post stage on both devices: the C64 preset 8619 ms on the CPU device against 95 to
  112 on the GPU device, C64 hires cells 2466 against 31 to 37,
  ZX_SPECTRUM 1303 against 99 to 104, MSX1 1048 against 91 to 128. On
  the CPU device these roads are against the performance rule.
- **The adaptive palette lock is keyed without the picture**
  (`core/post.py:317`: size, method, seed) — older than the round, and by
  design for animations. `core/palette.clear_caches()` clears it;
  `core/render.clear_caches()` (`render.py:257-259`) does not. In one
  process a later frame inherited an earlier scene's palette (measured
  headless in the collage tool, fixed there only). Whether a second scene
  in one Blender session inherits the first scene's palette was not read.
- The G-buffer's third barycentric is packed as `1 - x - y`, one ULP off
  the rasteriser's own on most pixels: headless, the fog stage over the
  same simulated shading sits 2.4e-7 from the CPU on the smooth curves
  (bitwise on the quantised modes), and the whole fogged frame 8.3e-7 to
  1.1e-5 from the CPU frame across the fog rows the suite checks
  (`docs-dev/suite_1.90.0_fix2.log:12824-13500`). Two lines (the kernel's
  and `pack_ids`'s) that must move together.
- N64 coverage AA: the RDP's CLAMP accumulate and the VI's 2x bilinear
  scale are unbuilt (half of the "N64 blur"); the stage makes the whole
  frame 15-bit (its tooltip says so since fix pass 3).
- Unverified stand-ins: SR Bump's azimuth origin against KallistiOS's
  `pvr_pack_bump`; the Jaguar's CRY chroma square against the manual's
  tables; LightWave's squeeze curve against a real render; the VHS preset
  at one generation where its spec said three.

## Named refusals that could someday lift

Each prints its reason and shades on the CPU correctly today:

- Vertex-rate materials behind glass
- TRILINEAR footprints in LAYER passes
- Per-pixel opacity under stipple
- Ortho Backfacing (needs a mixed-winding ortho fixture first)
- Light linking past 64 objects
- Rich worlds behind NON-ray layers
- The Light Meter, Falloff's Shadow/Light and Shader to RGB (they read the
  lamp loop at shade time; the GPU's tone is a Facing or a Layer Weight)
- The Blur node's re-evaluation (it re-runs the upstream chain at shifted
  points)
- DepthCue per-material fog and Vector Transform's camera half (want
  per-frame uniforms at material rate)
- Gabor noise and Sky Texture GLSL emitters (CPU evaluators exist)
- Colour layers other than the active one (the G-buffer carries one; since
  1.87 a node naming THAT layer resolves on both devices)
- A linked percentage on the 3ds Max material nodes (the evaluator converts
  the unit; the raw chain earns no per-pixel grant)
- The ink's stroke road (Smooth, Pressure, Overshoot), isophote weight,
  Surface anchor, Form / Shadow / Tone line sources and vertex-colour line
  control (chain and surface walks on the CPU; the rest of the line is on
  the GPU since 1.88)
- The Bryce, painted, starfield and physical skies, world node graphs
  beyond a plain Background node, and the ground plane (the sky pass draws
  the simple skies and HDRI since 1.89; the starfield and physical twins
  want a stated bar against the CPU's float64-rounded chains, the Bryce and
  painted skies a round of their own)
- The film stages beyond grain and flicker (dust, soften, grade, the colour
  process, weave, hairs, scratches, cue marks, halftone; grains over 1.2 px
  and clumps) — per-pixel twins unbuilt; the sparse painters are a
  different algorithm
- What is left of the bit depth's dithers and palettes (the ordered
  dithers and the palette snap draw on the GPU since 1.90, on the CPU's own
  matrix, level table and inverse colormap): the Noise and Blue Noise
  dithers (a PCG stream), an adaptive palette with Lock Palette off, a
  Custom palette without an image, the 1-bit depth (declined: a three-term
  float sum before a strict compare, not bitwise-able without `precise`)
- Fog over the readback for materials with traced reflections, refractions
  or a CPU environment term (fog runs in the material pass since 1.90;
  their composites land after it); a per-pixel fog road for vertex-rate
  passes (the Voodoo, PowerVR and DS fogged per pixel — the Voodoo2 fog
  dither and the GameCube range adjust both need a pixel); the material
  fog dials on reflection hits; the sky pass as the backdrop-fog target
  (drops a 14.7 MB upload at 720p); a fog socket driven per pixel
- The Sega Model 2 and Model 3 light units (no GLSL: the CPU's corner road
  on both devices; a pixel-rate request refuses)
- The fixed-function blend equations, Thin Wall and the Imagine fog object
  in the GPU's layer passes (they read the frame beneath on a 5- or 6-bit
  lattice); a per-pixel alpha under the PS2's two-pass alpha; an Env-hole
  material's frame pass (until the sky pass draws through holes)
- The REYES shading rate under affine texturing, for a Bump-node material,
  and on hit and layer passes
- A linked Toon Size under the DS toon tables, a per-pixel Glossiness under
  the DS light unit, a per-pixel Brilliance, a linked Light on SR Bump
- The planar-shadow, modifier-volume and DS shadow-polygon mask and the
  framebuffer-format pack (CPU edits over the readback; the GLSL mask block
  was written and dropped for its order against the traced composites)
- The Elite and vector-beam wires (a GPU line pass over the same welded
  edge list; a per-pixel gather over binned strokes)
- Per-scanline palettes (a per-row compute port) and C64 multicolour cells
  with Lock Palette off
- By size: a matte-glow diffusion past 160 taps (the downsampled ladder is
  the next cut), the NTSC S-Video cable past 1,685 pixels of width, a tape
  low-pass wider than 96 pixels
- The depth encodings past their referral budget (a frame marking more
  than a tenth of its pixels rasterises on the CPU whole), the W-buffers
  under an orthographic camera, N64 coverage under supersampling
- A reflective material under a Mode 7 floor; Super Black's coverage plane
  and depth of field from the worker pool

## Impossible by construction (documented, not planned)

- Error diffusion on the GPU — sequential by definition (since 1.90 the
  display transform ahead of it runs on the CPU by name; on the driver
  that did not make the devices diffuse one picture: first under "Open
  after 1.90")
- A-buffer fragment *collection* on the GPU — the shading of those fragments
  is already there; the unbounded per-pixel list is a different algorithm
- Dot crawl (NTSC) — frame-dependent state

Filed here until 1.90 and lifted: Painter's algorithm on the GPU. It was
"ordered submission fill"; with one depth per polygon and a named tie rule
on equal keys it is order-free, and the compute rasteriser draws it.

## New machinery under consideration

- What 1.90's packs named as their next cuts: the NES / Mega Drive / SNES /
  Game Boy Color sub-palette cells (a per-frame fitted set of
  sub-palettes); the Jaguar's intensity-only Gouraud and the MSX2+ YAE
  screens; Cinepak's per-strip codebooks and MPEG-1's inter-frame road; the
  PAL-shaped composite bleed under the PAL receiver; the machines'
  write-time dither matrices (the PS1's signed 4x4, the RDP's magic
  square); alpha-to-coverage on the N64 coverage plane and DS edge
  coverage; the NV2A combiner's mux, dot-product and final stages and TEV's
  konst registers; REYES "smooth" interpolation; a highlight-suited DS
  table; a coverage plane that includes blended geometry; colormap
  lighting (Doom / Quake) once a pack owns the unlit albedo per pixel
- Sparking! ZERO's undecoded terms: the reflection sphere map, the top
  light, the FaceLighting masks, ShadowStep, the highlight's gate
  (SpecularColor's alpha, M_Gloss, SpecularShininess, SpecularSmooth —
  carried on the node, not lit), the outline shell's push (the inverted
  hull the game's vertex shader extrudes by COL0; Halcyon inks instead) —
  the reconstruction the field brought does not carry them either; the
  game's own master graph would
- Generalising the Shadow Ramp road past the Anime Shader
- A GPU march for the cel field, if the viewport's refine asks for it
- The Imitation Study's unbuilt ranks: stroke coherence across frames, tonal
  art map scale nesting, the watercolour law as its own media node, the
  successive-exposure false-colour paint palette
- 3ds Max leftovers: Matte/Shadow, Raytrace's extended parameters, the maps
  still without a node (Bitmap, Combustion, Particle Age, Flat Mirror,
  Reflect/Refract, Thin Wall Refraction, Camera Map Per Pixel)
- Region/border render with exact pixel-identity
- Hybrid opaque frame (split one frame's materials across devices)
- QTVR strip panorama camera
- In-graph Bevel
- Cast-filtered shadows
- Baking to texture
- More of the Bryce library's texture components mapped one-to-one as the
  decoder's understanding of the component blocks deepens
