# R254 map-gpu: the Softness dial, what the device keeps, what an edit costs

Work-package key `map-gpu`, forked from 4e655f6 (the map-soft merge). The
integrator owns CHANGELOG.md / README.md / the version stamps; the fragments
below are in their house voice. Findings covered: soft-erode-01 (hal-shadow)
= size-map-v01, the size-0 half; gpu-atlas-stale-01 (hal-shadow) =
gpu-shadow-atlas-key-row-sum (viewport-update); perf-gpu-lamp-move-recompile-04
(perf); the GPU half of vp-size-drag-rebake-recompile (viewport-update);
size-ray-11 (size-ray); the GPU half of perf-cache-bytes-01. Not done, with
the measurements and the design: the no-caster table and the conservative
search (map-soft's follow-ups), perf-gpu-device-cpu-bake-02.

Every number was measured on this tree against the fork point, same machine,
back to back. The probes are in `<scratch>/wt/map-gpu/aud/` (`soft0.py`,
`preset_mass.py`, `box_mass.py`, `gpu_par.py`, `neutral_dump.py` /
`neutral_cmp.py`, `counters.py`, `drag_check.py`, `timings.py`,
`vis_time.py`, `misc_numbers.py`, `skip_proto.py`), their outputs in its
`out/`, the real-Blender scripts and frames in `<scratch>/wt/map-gpu/real/`.

## CHANGELOG

- **What was reported.** "The lights don't get softer / harder with size ...
  some of the general options just don't work properly", and "Shadow map is
  slower than ray traced shadows" (an RTX 5060 Ti, the GPU render device, the
  rendered viewport).

- **Softness shrank the shadow it was blurring, and at 16 erased it.** Under
  Shadow Maps the render's Softness (times the lamp's own) is a blur radius
  in texels. For a lamp of size 0 the lookup was four to twelve hard taps
  spread over that radius, and to keep so wide a kernel from shadowing its
  own receiver the lookup point was lifted off the surface by the same
  factor: `offset = texel * (1.5 + 2.5 sin) * max(1, Softness)`
  (`core/lights.visibility`, mirrored as a literal by
  `gpu/material._shadow_function`). At Softness 8 on a 512 sun map the point
  sat a world unit above the floor, past the caster. Wall on a floor, POINT
  of size 0, 512 map, Softness 0 / 1 / 2 / 4 / 8 / 16 / 32: area darker than
  half 7.48 / 7.39 / 6.87 / 5.93 / 4.38 / 2.26 / 0.05 (the traced hard
  shadow: 7.90); a Sun: 3.15 / 3.12 / 2.88 / 2.54 / 1.72 / 0.28 / 0.00; five
  grey levels across every edge.

- **What it does now.** A lamp of size 0 whose Softness is above one texel,
  at Samples above 1, takes the per-receiver lookup map-soft built for sized
  lamps (`core/softmap.lookup`, GLSL twin `gpu/softmap.shadow_function`) with
  no lamp term (`softmap.zero_shape`: A = Bv = 0): the 13-tap search decides
  lit / umbra, and what is left is filtered by `F = clamp(4 Samples, 16, 64)`
  bilinear compares over the Softness radius, each against the receiver
  PLANE's depth at its own texel. The normal offset carries no blur factor
  (`softmap.offset_scale`, the sized lamp's rule). The probe stage is
  skipped where the footprint has no lamp term -- it would read the search's
  own taps -- on both roads (`amax > 0`). Same scene, Softness 0 / 1 / 2 / 4
  / 8 / 16 / 32: POINT area 7.48 / 7.39 / 7.85 / 7.84 / 7.74 / 7.81 / 7.08,
  mass 7.48 / 7.47 / 7.85 / 7.84 / 7.81 / 8.07 / 8.16; SPOT mass 7.34 / 7.34
  / 7.79 / 7.77 / 7.67 / 7.84 / 8.21; SUN mass 3.15 / 3.16 / 3.28 / 3.28 /
  3.26 / 3.46 / 2.98. The 10-90 % width of the edge grows in proportion
  (POINT: 0.064 / 0.127 / 0.257 / 0.515 / 0.989 world units for 2 / 4 / 8 /
  16 / 32 texels), in 28 to 540 grey levels with no step above 0.055, and
  the half-dark line stays on the hard edge (it walked 0.04 inwards at
  Softness 4 and 0.07 at 8).

- **What stays bit for bit.** Softness at or below 1 on a size-0 lamp (the
  default is 1) and every lamp at Samples 1 keep the constant lookup
  (`softmap.lookup_shape`). A sized lamp's shadow is the map-soft frame.

- **The device kept the old shadow atlas when a caster moved between the
  rows its fingerprint sampled.** The atlas upload key and the plan
  signature identified a map by `round(depth[::23].sum(), 3)` per face: the
  sum of every 23rd ROW. A small caster between two sampled rows, or one
  sliding along the rows, kept the key; `device.upload_cached` returned the
  resident texture and the GPU road drew the shadow where the caster used
  to be while the CPU road drew it right (0.33 over 32 px in the audit's
  case; here a plate under a spot at five positions along x: one key,
  80740.805). The atlas now has two identities: its KEY is the pack's layout
  (per lamp: map size, grid, depth rule) and its STAMP an exact content
  digest of every face (`gpu/shade.map_digest`: the CRC-32 of the depth
  bytes, `core/mathx.content_digest`, taken once per baked map and parked
  on `ShadowMap.digest`). `upload_cached(key, build, stamp)` -- the R251
  C015 rule -- replaces the one resident pack when the stamp moved, so the
  device can neither miss a changed map nor pile old ones up (it kept seven
  after six nudges, 168 MB at a 512 point lamp, and evicted by count at 96).

- **A lamp edit recompiled every material.** The shadow function printed
  the lamp's values into its text -- each face's light-space matrix, the
  origin, the clip planes, Bias, Softness (as unrolled tap offsets), Density,
  a sized lamp's radius, a traced lamp's radius or rectangle -- and the
  light block printed the Shadow Colour, so `_light_sig` and `_plan_sig`
  carried all of them and one tick of any of them was a plan miss and one
  driver compile per material pass, where moving a traced lamp compiled
  nothing. Those values ride `hal_lights` now: texels 12-15 of each lamp
  (Shadow Colour and Density; the traced size and Area Gamma; an Area
  lamp's unit axes and half sizes) and a tail past the lamp texels laid out
  by `material.light_tail_layout` (an Area lamp's face; per mapped lamp five
  header texels and nine per face, `material.map_texels`). Each texel is
  np.float32 of the expression the emitter printed, which is the float32 the
  CPU computes with, and the shader's arithmetic and its order are the
  CPU's: a tap offset is `vogel * Softness` in a statement of its own ahead
  of the add, as `_pcf_offsets(taps) * radius` is. `_light_sig` keeps what
  each block's EXISTENCE depends on, `_plan_sig` adds each lamp's shadow as
  structure (`_shadow_struct`: method, map size / grid / projection / depth
  rule, tap unroll, which lookup, a traced Sun's law, an Area's shape), and
  a plan-cache hit rebuilds the shadow metas, the pack entry and the texels
  from the current lamps and maps -- the idiom the hit path already used
  for the light texels. The render's Softness and Bias left the settings
  signature for the same reason.

- **The pack is one channel.** It was RGBA32F with depth in .r and zeros in
  three channels; it builds as a 2D float32 array and `gpu/device` uploads a
  2D array as R32F (falling back to RGBA32F with the values in .r where a
  build cannot create the format). The fake device and the simulator's
  sampler keep one channel too (`gpu/shade.RedTexture`; `texelFetch` on it
  returns (r, 0, 0, 1)). The map's nearest depth, which map-soft kept in the
  G channel of the lamp's first atlas texel, is tail texel h4.x.

- **The plan cache.** It emptied itself past four entries; with every lamp
  nudge a new entry, a drag threw the draft's and the refine's plans away
  every fifth tick. It holds eight plans and drops the least recently used
  one (`PLAN_CACHE_MAX`); a lamp edit is no longer an entry at all, so the
  two plans a viewport alternates between -- different frame sizes, and
  different map sizes once drafts bake smaller maps -- are always the two
  most recently used. A parked plan keeps the pack's key only: its build
  closure held the lamps' depth maps (100 MB for a 2048 cube).

## The numbers

### Softness on a lamp of size 0 (`soft0.py`, wall on a floor, 512 map, Samples 4)

| lamp | Softness | before: area / mass / levels | after: area / mass / width / levels |
|---|---|---|---|
| POINT (traced 7.90) | 1 | 7.39 / 7.47 / 5 | 7.39 / 7.47 / 0.025 / 5 (bitwise) |
| | 2 | 6.87 / 7.00 / 5 | 7.85 / 7.85 / 0.064 / 28 |
| | 4 | 5.93 / 6.15 / 5 | 7.84 / 7.84 / 0.127 / 53 |
| | 8 | 4.38 / 4.73 / 5 | 7.74 / 7.81 / 0.257 / 105 |
| | 16 | 2.26 / 2.67 / 5 | 7.81 / 8.07 / 0.515 / 208 |
| | 32 | 0.05 / 0.33 / none | 7.08 / 8.16 / 0.989 / 398 |
| SPOT (7.90) | 4 / 8 / 16 | 5.51 / 3.75 / 1.52 area | 7.73 / 7.51 / 7.23 area, 7.77 / 7.67 / 7.84 mass |
| SUN (3.24) | 4 / 8 / 16 / 32 | 2.54 / 1.72 / 0.28 / 0.00 area | 3.23 / 3.16 / 3.14 / 0.18 area, 3.28 / 3.26 / 3.46 / 2.98 mass |

At 32 texels the blur is wider than the wall's shadow, so little of it is
darker than half (a Sun: 0.18) while the mass stays (2.98 of 3.24).

### Every shipped preset with Softness above 1 under Shadow Maps (`preset_mass.py`)

Eleven presets (a twelfth, Bryce still, is traced). Mass behind the wall
under the preset's own Softness / Samples / Map Size, before -> after:

| preset | Softness, Samples, map | POINT size 0 (traced 7.90) | SUN size 0 (traced 3.24) | POINT Radius 0.25 |
|---|---|---|---|---|
| INFINID_4 | 1.5, 8, 512 | 7.25 -> 7.84 | 3.09 -> 3.28 | 7.78, bitwise |
| MAX_R2 | 2.0, 8, 512 | 7.03 -> 7.84 | 3.03 -> 3.28 | bitwise |
| MAX_2012 | 4.0, 8, 512 | 6.19 -> 7.85 | 2.73 -> 3.28 | bitwise |
| BRYCE_2, ANIMATION_MASTER | 3.0, 12, 512 | 6.58 -> 7.84 | 2.86 -> 3.28 | bitwise |
| CINEMA4D_4 | 2.5, 8, 512 | 6.81 -> 7.85 | 2.94 -> 3.28 | bitwise |
| VUE_DESPRIT | 4.0, 12, 512 | 6.17 -> 7.85 | 2.71 -> 3.28 | bitwise |
| LIGHTSCAPE | 4.0, 4, 512 | 6.15 -> 7.84 | 2.69 -> 3.28 | bitwise |
| GAMECUBE, WII | 2.0, 8, 1024 | 7.47 -> 7.90 | 3.10 -> 3.22 | bitwise |
| XBOX | 1.5, 8, 1024 | 7.60 -> 7.90 | 3.13 -> 3.22 | bitwise |

Grey levels across the edge: 5 to 11 before, 12 to 94 after.

### Real Blender 5.2.2, F12 through this tree (`real/bl_preset.py`, `real/analyze.py`)

The harness scene (floor, floating box, ball), the preset applied by the
add-on's own operator, 320x240, a POINT lamp of Radius 0 and a SUN of Angle
0; the shadow term is the frame over the same frame with shadows off (the
unshadowed frames are bitwise equal between the two trees). Sheet:
`<scratch>/wt/map-gpu/real/out/presets_before_after.png` (left before, right
after; rows MAX_2012 POINT, SUN, VUE_DESPRIT POINT, SUN).

| preset, lamp | darker than half, px | mass | levels in the penumbra |
|---|---|---|---|
| MAX_2012, POINT | 3040 -> 3518 | 3051.5 -> 3515.1 | 328 -> 447 |
| MAX_2012, SUN | 1848 -> 1664 | 1873.9 -> 1673.5 | 113 -> 206 |
| VUE_DESPRIT, POINT | 0 -> 0 (Density) | 1158.2 -> 1263.9 | 409 -> 426 |
| VUE_DESPRIT, SUN | 984 -> 696 | 893.4 -> 793.3 | 278 -> 324 |

The Sun's mass goes DOWN here and that is the fix as well: under a
floating caster the lifted lookup did not shorten the shadow, it displaced
it onto the box's and the ball's own faces. The same scene headless
(`box_mass.py`, Samples 8, no ambient): the traced hard shadow is 1636 and
the mapped one 1711 at Softness 0; at Softness 4 it was 1860.8 and is
1660.9; at 8, 1755.5 -> 1692.5. POINT: traced 3558, Softness 0 3392; at 4,
3027.2 -> 3489.4; at 8, 2661.2 -> 3489.1.

### The device and the edits (`counters.py`, fake device, 96x72, three materials, 512 maps, six ticks each)

| edit | before: plan misses / compiles / pack uploads / packs resident | after |
|---|---|---|
| Power (control) | 0 / 0 / 0 / 1 (24 MB) | 0 / 0 / 0 / 1 (6 MB) |
| mapped point lamp nudged | 6 / 18 / 6 x 24 MB / 7 (168 MB) | 0 / 0 / 6 x 6 MB / 1 (6 MB) |
| mapped spot nudged | 6 / 18 / 6 x 4 MB / 7 (28 MB) | 0 / 0 / 6 x 1 MB / 1 (1 MB) |
| traced lamp nudged (control) | 0 / 0 / - / - | 0 / 0 / - / - |
| Radius dragged, mapped | 6 / 18 / 0 / 1 | 0 / 0 / 0 / 1 |
| Radius dragged, traced | 6 / 18 | 0 / 0 |
| Sun Angle dragged, mapped and traced | 6 / 18 each | 0 / 0 each |
| Area Size dragged, mapped and traced | 6 / 18 each | 0 / 0 each |
| lamp Softness dragged (size 0, sized) | 6 / 18 each | 0 / 0 each |
| lamp Bias dragged | 6 / 18 | 0 / 0 |
| lamp Density dragged, mapped and traced | 6 / 18 each | 0 / 0 each |
| Shadow Colour dragged | 6 / 18 | 0 / 0 |
| render Softness, render Bias dragged | 6 / 18 each | 0 / 0 each |
| a caster moved, mapped | 6 / 0 / 6 x 24 MB / 7 (168 MB) | 6 / 0 / 6 x 6 MB / 1 (6 MB) |
| a caster moved, traced (control) | 6 / 0 / - / - | 6 / 0 / - / - |

First frame: 4 compiles on both trees. Every edit after it now re-uploads
the light texels (1,200 bytes for one point lamp: 16 lamp texels and a
59-texel tail) and, when a map changed, the pack. A moved caster is still a
plan miss -- the mesh key is in the signature -- and still compiles nothing
(the sources are identical). `drag_check.py`: after each of these edits on
a kept device the GPU frame is the CPU frame of the edited scene, worst
2.3e-3 (an Area lamp's own lighting term), 1e-6 elsewhere.

Host side of the GPU road, one POINT lamp (`timings.py host`):

| map | pack bytes | pack build | plan + pack build after a nudge | digest, six faces |
|---|---|---|---|---|
| 512 | 24 MB -> 6 MB | 20.5 -> 14.1 ms | 18.5 -> 6.2 ms | 1.7 ms |
| 1024 | 96 -> 24 MB | 1207 -> 32 ms | 669 -> 21.7 ms | 7.5 ms |
| 2048 | 384 -> 96 MB | 3397 -> 92 ms | 1758 -> 86 ms | 33.9 ms |

The RGBA pack allocated the per-lamp RGBA atlas and the pack, 2 x 384 MB of
mostly zeros at 2048, on every nudge. `_plan_sig`: 0.10 to 0.57 ms -> 0.06
ms.

### Real Blender 5.2.2 with a window (`real/bl_gpu.py`; xvfb, Mesa llvmpipe, `--gpu-backend opengl`)

The GPU device engaged for all three frames (no fallback printed; `engaged
True`), F12 at 160x120, a 256 point map: size 0 at Softness 1, size 0 at
Softness 4, Radius 0.3. Against the CPU device's frame: 0.003822 /
0.003906 / 0.003906, 0 px above 6e-3 (one step of the 8-bit output stage).
This build creates an R32F texture and reads it back exactly, and
`device.upload` of a 2D array returns one (`R32F`; an RGBA array:
`RGBA32F`). The third frame -- a lamp gaining a size after the Softness-4
frame -- compiled NOTHING here (plan 9.7 ms, draw + read 6.9 ms: the sized
lamp's text is the size-0 Softness text, the size is a texel); on the fork
point the same frame compiled one new shader (956 ms and a 17.5 s first
draw under llvmpipe). No real GPU driver was run.

### Both roads

`gpu_par.py` (160x120, wall scene, the simulator against the CPU): size-0
lamps at Softness 2 / 8 / 16, Samples 16, a MIDPOINT map, per-lamp Softness
/ Bias / Density: 1e-6 to 6e-6. Sized lamps (unchanged arithmetic): POINT /
SPOT / SUN 1e-6 to 6e-6, AREA 0.4e-3 to 2.8e-3 (its lighting term, as in
map-soft). Traced: POINT / SPOT / SUN 0 to 1e-6, AREA 0.5e-3 to 1.2e-3. An
AREA lamp of size 0 is degenerate input -- its own lighting term differs
between the roads with shadows OFF, 1.08 on 2328 px, before and after -- so
its shadow is judged as a term (each road's frame over its own unshadowed
frame): 7e-6 over 5215 px; on the CPU it is the size-0 point lamp's term
bitwise. Blender never exports one (Area sides are clamped at 0.02).

`tests/glsl_strict.py`: mapped shadows 28 of 28 (the 14 cases on the texel
road and on the literal road, size-0 lamps at Softness 4 added), traced
shadows and the Area form factor 16 of 16 (new: `check_traced_shadows`),
everything else as before; "every shader compiles under strict GLSL 4.30".

## Which pictures change

- A lamp of size 0 under Shadow Maps with Softness (lamp x render) above 1
  and Samples above 1: its shadow, on both roads. That is the eleven presets
  above when a lamp has no size (a legacy import's buffer spots, a Radius
  set to 0).
- Nothing else. `neutral_dump.py` / `neutral_cmp.py`: 181 arrays -- CPU
  frames, simulator frames and depth packs of POINT / SPOT / SUN / AREA at
  size 0 with Softness 0 / 0.5 / 1, Samples 1 (also at Softness 4), 7 and
  16, a MIDPOINT map, per-lamp Softness x render Softness = 1 with Bias,
  Density and Shadow Colour, a 256 map; no shadow; traced hard and soft;
  SIZED lamps at Softness 1 and 4; Area discs with Gamma; a wide Sun; the
  demo scene mapped, traced and with shadows off -- are bitwise the fork
  point's. 16 arrays of the fix cases differ. Of 74 GLSL texts 66 changed
  (values became texels) and 8 are identical (no shadow function and no
  Area lamp in them).
- The emitted GLSL of any pass with a mapped or soft-traced shadow, a
  Shadow Colour or an Area lamp: the same statements over texel fetches.
  `gpu/shade.SHADOW_TEXELS = False` prints the literals again (the suite's
  A/B: fourteen lamp cases, texel frame == literal frame bitwise).

## Cost

- Depth fetches per floor point for a size-0 lamp (`misc_numbers.py`): the
  constant lookup took as many as Samples (4 / 8 / 12); the filter road
  takes 13 to 25 for the search on every point and 4 F on the 5 to 8 % in
  the penumbra: 22.4 / 26.7 / 40.4 on average at Softness 4 and Samples
  4 / 8 / 12 (POINT), 21.4 / 25.6 / 38.8 (SUN).
- The shadow term alone over 307,200 floor points (`vis_time.py`, best of
  7, two runs of each tree; the machine was at load 7 and unchanged paths
  moved by up to 30 % between runs): POINT size 0, Softness 4, Samples 4 /
  8 / 12: 261-300 / 294-379 / 378-428 ms before, 389-737 / 447-654 /
  598-877 ms after; SUN, Softness 4, Samples 8: 252-265 -> 320-478 ms. So
  about 1.2 to 1.6 times the constant lookup on the quieter run, for a
  shadow that is where it should be. Whole warm frames (`timings.py cpu`)
  were inside the machine's noise and are not quoted.
- Softness at or below 1 and sized lamps: the same code as before.

## Not done

- **Samples 1 with Softness above 1 still shrinks the hard shadow** (by
  instruction: Samples 1 stays bitwise). One tap cannot blur, and the
  offset still carries `max(1, Softness)`: POINT 7.48 / 6.20 / 4.78 / 2.71
  at Softness 1 / 4 / 8 / 16, SUN 3.15 / 2.69 / 2.11 / 0.95. No shipped
  preset pairs Samples 1 with a Softness above 1 (the hard-shadow presets
  set 0 or leave 1). The fix is one line on each road -- the factor 1 when
  the lamp's sample count is 1 -- and it would make Softness inert at
  Samples 1, which is what the tooltip says ("It needs Samples above 1").
- **Leaving size 0 at Softness 1 (the default) is still a step.** Above
  Softness 1 it is gone: size 0 against the smallest size, POINT, Softness
  4: mass 7.84 -> 7.84, largest change of the term 0.0001. At Softness 1
  the size-0 lamp keeps the constant lookup by rule: mass 7.47 -> 7.84 and
  887 of 71,093 floor points change by more than 0.25 (a Sun: 295).
- **The no-caster table and the conservative search** (map-soft's
  follow-ups). Measured, not built. On a frame's worth of floor points 73
  to 87 % are fully lit and each pays the 13-tap search (`skip_proto.py`:
  POINT 0.3: search on 307k points, probe on 16 %, filter on 8.5 %; SUN
  0.05: 11.6 % / 6.3 %), about half the lookup's fetches. The exact skip is
  `min depth over the search's footprint >= d_r - slope` (the plane depth
  is never deeper than d_r, so no tap can be a blocker); a min PYRAMID per
  face answers it in four fetches at the level whose tile covers the
  footprint, independent of Softness and Samples, and the same pyramid is
  what the thin-caster limit needs (a search that found nothing where the
  pyramid says a nearer texel exists must not report lit). The pack is one
  channel now, so the pyramid has to be extra rows of each lamp's band; a
  footprint that leaves a cube face keeps the search.
- **perf-gpu-device-cpu-bake-02** (the maps baked on the GPU device). Not
  attempted in code: it was to come last and the item above was not done.
  What this round adds to the audit's design: the GPU device DOES run in
  real Blender here (xvfb, llvmpipe, OpenGL backend; see above), so a
  kernel bake can be exercised without a GPU. The hook, against the bake
  as it stands: `core/lights._render_depth(verts, tris, vp, size, cull,
  midpoint)` is the one place a depth image is produced. A module-level
  `LI.DEPTH_BAKER = None`; when set and `midpoint` is false,
  `_render_depth` returns `DEPTH_BAKER(verts, tris, vp, size, cull)` unless
  that returns None (a refusal, printed by name as '[Halcyon GPU] shadow
  bake on the CPU: <why>'). `core/render._build_shadows` sets it for the
  frame when the render device is GPU and the device probe passed, and
  clears it in a finally. The baker is a depth-only variant of the compute
  rasteriser (`gpu/craster`: `compute_variant_name` gains a flag; ONE
  image, .x = z where a triangle won and 1.0 elsewhere, read back through
  `device._main`), dispatched serially: `_build_shadow_maps` must not take
  its thread pool while the hook is set. The read-back array becomes the
  `ShadowMap` both roads read, so a CPU fallback inside the frame sees the
  same texels. `_shadow_base_signature` needs the baker's identity so a
  CPU-device frame never reads a GPU-baked map from the cache. Expect a
  gain only where the bake is texel-bound (the verifier: 0.25 s against
  2.30 s of host work per 2048 cube at 55k triangles, nothing at 885k).
- **A lamp with a projected texture still re-plans on a size tick.** Its
  blurred image is prepared from the source size and held by the plan
  (`LI.cookie_blur_uv`), so the Radius stays a value in the cookie group of
  `_light_sig`; the sources are identical, so the re-plan uploads a new
  image and compiles nothing.
- **One resident pack per LAYOUT, not per scene.** Two scenes with the same
  lamp layout rendered in turn on one device (a viewport and an F12 at the
  same map size) replace each other's pack: one re-upload per switch.
- **A mapped lamp's Density leaving exactly 1.0 re-plans once** (no
  compile): the bit stays in `_light_sig` because a traced shadow emits its
  Density term only off 1.0 (pinned by test_r254_ray_road).
- **Roads.** CPU F12 and the GPU simulator twin were measured headless;
  real Blender 5.2.2 F12 was rendered on the CPU device (two presets, both
  trees) and on the GPU device under llvmpipe; the rendered viewport was
  not driven live; no real GPU driver ran, so compile and upload times on
  the reporter's card are not measured.

## Pins changed

- `tests/test_r254_map_soft.py test_a_size_zero_is_the_old_lookup`: the
  list of settings under which a size-0 lamp "is the classic PCF lookup,
  bitwise" held `{'shadow_softness': 4.0, 'shadow_samples': 9}`; it holds
  `{'shadow_softness': 1.0, 'shadow_samples': 9}`, and a new check says a
  size-0 lamp at Softness 4 is NOT the classic lookup. `key, build = atlas`
  reads `atlas[0], atlas[1]`; `float(np.abs(px[..., 1:]).max()) == 0.0`
  became `px.ndim == 2`.
- same file, `test_h`: `'a lamp of size 0 at Softness 8 is still the
  classic lookup (its own decision)'` (array_equal to the classic lookup)
  became `'a lamp of size 0 at Softness 8 keeps its mass too'` (within 10 %
  of the Softness-1 mass, and not the classic lookup).
- same file, `test_j`: `'the nearest depth of the map rides the G channel
  of the lamp's first atlas texel'` (`px[0, 0, 1] == f32(d_min)`) became
  `"... rides the lamp's tail block in hal_lights, and the pack is one
  channel"` (`hal_lights[0, 16 + 4, 0] == f32(d_min)`, `px.ndim == 2`); `'a
  nearer caster re-keys the atlas'` (`k0 != k1`) became `'re-stamps the
  atlas under the same key'` (`k0 == k1 and s0 != s1`); `"a mapped lamp's
  radius / angle / Size Y is light-signature structure"` (`!=`) became `'is
  a texel value: the light signature does not move'` (`==`); the shape
  stays `!=`.
- `tests/test_r253_light_options.py`: `'texels 12-15 are reserved zeros
  ...'` (`np.all(tex[0, 12:16] == 0.0)`) became `'texels 12-15 hold the
  shadow values at their defaults ...'` (12 = (0, 0, 0, 1), 13 = (0, 1, 0,
  0), 14 and 15 zero).
- `tests/test_render.py test_deferred_shading_carries_the_shadows`:
  `pack.ndim == 3 and pack.shape[2] == 4` became `pack.ndim == 2`.
- `tests/glsl_strict.py`: `mapped_shadow_cases` gains a size-0 Softness-4
  case per kind; `check_mapped_shadows` compiles both value roads;
  `check_traced_shadows` is new.

`tests/test_r254_ray_road.py` 'mapped: the lamp's Samples moves the frame'
fails at the fork point as it does here (max 0.0137 against a 0.02 bar: the
map-soft filter is continuous at any tap count); claude-cloud repaired the
pin after this branch forked (f6484e3). It was not touched.

## Tests

`tests/test_r254_map_gpu.py` (`python -m halcyon.tests.test_r254_map_gpu`):
eight functions, a to h, 141 checks, all passing here. On the fork point
with only this file dropped in: 33 failures (27 failed checks and six
functions that stop on a missing API), every function failing. Re-run on
this tree: test_r254_map_soft (168 ok), test_r254_ray_road (the one fork-
point failure above), test_r254_falloff, test_r254_spot_beam,
test_r253_light_options (195 ok), test_r253_uniforms, test_r251_shadow and
test_r251_lighting_lamps (only the 1.89.0 zip pin of the baseline list), 34
functions of test_render that touch plans, shadows, lamps and the device
(`test_every_feature_survives_the_device_switch` among them; only
`test_gpu_sky_pass`'s eight baseline sky pins fail), and the strict GLSL
gate.

## Files

- `core/softmap.py`: `zero_shape`, `lookup_shape(light, samples, soft)`,
  the probe skipped where the footprint has no lamp term.
- `core/lights.py`: `visibility` hands the Softness to `lookup_shape`;
  `ShadowMap.__slots__` gains `digest` (the slot only).
- `gpu/softmap.py`: every value a literal or a texel fetch (`mb`), the
  probe branch, the nearest depth as a value.
- `gpu/material.py`: `light_tail_layout`, `area_face_texels`, `map_texels`,
  `pack_light_texels(lights, job, shadows)`, the constant lookup, the
  traced functions, the Area form factor and the Shadow Colour over texels,
  the `hal_ltex` prototype ahead of the shadow functions.
- `gpu/shade.py`: `_shadow_meta` (layout key, stamp, one-channel build,
  `want_atlas`), `map_digest`, `_shadow_struct`, `_shadow_pack`,
  `_light_sig`, `_plan_sig`, the hit path, `_plan_store`, `PLAN_CACHE_MAX`,
  `PLAN_STATS`, `SHADOW_TEXELS`, `RedTexture`, `sim_texture`.
- `gpu/device.py`: `_upload_impl` uploads a 2D array as R32F.
- `shaders/builtins.py`: `texelFetch` on a one-channel sampler.
- `tests/fakedevice.py`: `FakeTexture.fmt`, the R32F sampler.
- `tests/test_r254_map_gpu.py` (new), `tests/glsl_strict.py`, the pins
  above.

## Notes for the merge

- `ShadowMap.digest` is filled lazily by `gpu/shade.map_digest` and never
  by the bake. A bake that rewrites `depth` IN PLACE on a cached map must
  set `digest = None`; one that builds a new `ShadowMap` needs nothing.
- `_light_sig`, `_plan_sig` and the hit path of `plan_frame` changed
  shape; the "viewport" package's `_mesh_key` work and the "flags"
  package's ambient pool touch the same function bodies.
- Draft-sized maps from the bake package are a different `_shadow_struct`
  (the map size is structure) and therefore a different plan and a
  different pack key: both stay cached and resident.
- `hal_lights` texels 12-15 are taken, and the texture is wider than
  n x 16 whenever a frame has a mapped or an Area lamp.
