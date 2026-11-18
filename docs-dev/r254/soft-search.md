# R254 soft-search: the mapped soft shadow's blocker search and filter

Work-package key `soft-search`, forked from dd9d034 (claude-cloud after the
map-gpu merge and the integration commit). The integrator owns CHANGELOG.md /
README.md / the version stamps; the fragments below are in their house
voice. Findings covered: the independent review's SP-1 (high), SP-2 (medium)
and SP-3 (low) of the mapped soft shadow, the no-caster skip the map-soft
and map-gpu write-ups list as a follow-up, and the "Samples 1 with Softness
above 1" decision map-gpu left open. Not fixed, with the measurements: thin
casters under wide lamps, and two casters at different depths over one
receiver (both under Limits). Every finding was reproduced on the fork point
first; the numbers below were measured on this tree against it, same
machine, back to back. The probes are in `<scratch>/wt/soft-search/` --
`probes/` (the reviewer's own, pointed at this tree), `probes_base/` (the
same against the fork point), `aud/` (this package's), `ms/` and `mg/` (the
map-soft and map-gpu packages' census, width, neutrality and timing tools)
-- and their outputs in each one's `logs/` or `out/`.

## CHANGELOG

- **What was reported.** "Fix the lights so that changing the size of the
  lights actually works. The lights don't get softer / harder with size",
  and "Shadow map is slower than ray traced shadows". The review of the
  merged per-receiver lookup found three places where the size still did
  not shape the shadow, all reproduced here: a sized Sun over a large
  ground (SP-1), a strip Area lamp (SP-2), a scene modelled at centimetre
  size (SP-3).

- **A Sun's shadow depended on how large the ground was (SP-1).** The same
  Sun (0.2 rad), the same objects, the same world size of a shadow texel:
  on a ground of 20 units the shadow's mass was 366.3 with 388 penumbra
  pixels (traced: 375.6 and 320); on 80 and 160 units 317.7 / 317.3 with
  203 / 201, and 78 pixels moved by more than 0.25. The penumbra stayed at
  198 / 203 / 225 / 203 pixels for an Angle of 0.00918 / 0.05 / 0.1 / 0.2
  while the traced one grows 17 / 88 / 174 / 320: the Angle no longer
  softened anything. *What was wrong:* `core/softmap._face` (twin:
  `gpu/softmap.shadow_function`) bounded the blocker search by `t(d_min)`
  with `d_min` the nearest depth of the WHOLE map. Under a tilted Sun that
  is the near corner of the ground, tens of units nearer than any caster,
  so the search ran at its cap -- a sixteenth of the map, 4.6 world units on
  the 80-unit ground -- and its 13 taps stood units apart and stepped over
  character-sized casters. A point lamp had the same fault in a milder
  form: a plate 0.3 above the lamp, shadowing nothing in view, is the
  nearest depth of the cube and moved 13 to 24 floor pixels by more than
  0.25.

- **What it does now.** Each map carries a REACH PYRAMID
  (`softmap.map_pyramid`, built once from the depth and parked on the map:
  the `pyr` slot of `ShadowMap` / `CubeShadow`). Per level a grid of tiles
  4, 8, 16 .. texels wide; each cell holds the smallest depth over the
  3 x 3 tiles around it, so ONE texel answers "the nearest depth within r
  texels of (u, v)" for any r up to the tile's width. The search's reach
  starts at the cap and is tightened against it: `r = t_s (1 + 4/F) amax +
  S + 3`, `d_near = max(d_near, the pyramid at r)`, `t_s = t(d_near)`, up
  to six times, until a step no longer moves the bound. Every step is
  conservative -- no caster inside the current reach is nearer than its
  minimum, so none matters beyond the next, smaller reach -- and it ends
  at the reach of the casters that stand near this receiver. On open
  ground under the Sun above the reach goes 164, 53, 13, 6, 5 texels (a
  2048 map); beside a caster it stops at that caster's own footprint.
  The 3 x 3 neighbourhood of a cube's tile is continued across the face's
  seams: the finest level of each face is extended by a ring of tiles read
  off the neighbouring faces (`softmap._seam_grids`: the tile's image on
  the neighbour, its stored depths divided by the largest divisor
  `_CubeFetch` would apply), wide enough for the cap's reach, and every
  coarser level pools that. The grid stands one texel off the map's own so
  that the middle of a map -- where a lamp aimed at a centred scene puts
  whole lines of receivers -- is not a tile border.
  Same Sun, grounds of 20 / 80 / 160 units: mass 373.5 / 371.8 / 371.7
  (traced 375.6), penumbra 442 / 440 / 440, 0 pixels moved; penumbra at
  0.00918 / 0.05 / 0.1 / 0.2 rad on 160 units: 198 / 219 / 271 / 440. The
  plate above the point lamp moves 0 floor pixels by more than 0.25.

- **A receiver with no caster in reach skips the search.** 73 to 87 % of a
  frame's floor points are fully lit, and each paid the 13-tap search --
  about half of the lookup's texel fetches, and most of what a sized
  mapped lamp cost over a size-0 one. Two exact tests, both read from the
  pyramid, both on both roads:
  1. *Nothing nearer.* The receiver plane's depth is never deeper than the
     receiver's own (`plane` floors it there), so when the nearest depth
     within reach is at or past `d_r - sl` no tap can be a blocker: lit.
     This serves receivers that face the map's axis -- a floor straight
     under a point lamp.
  2. *Nothing above* (a LEVEL receiver: a floor, a ground, a table top).
     Under a slanted light the receiver's own plane is the nearest depth
     within any reach, so test 1 never passes. The pyramid's second part
     holds, over the same tiles, the greatest world HEIGHT a texel's point
     reaches. Where the receiver's plane agrees with the level plane
     through the same point over the tile's whole window -- their largest
     disagreement there is carried as a margin, so a normal a hair off
     level still passes and one a degree off passes only on a small reach
     -- a blocker has to stand above that level, and nothing within reach
     does: lit. A cube's height part knows nothing past a face's edge, so
     a reach that crosses a seam is searched.
  Both change no value: with the skips switched off the lookup returns the
  same float32 on every point of 48 lamp / map size / Bias configurations
  -- level floors and a table top, a plate just above the floor, normals
  tilted by 1e-7 rad to 20 degrees (`aud/level_check.py`: 0 of 1.7
  million points differ) -- and a lamp of size 0 under the user's
  Softness is bitwise the fork point's frame.

- **A strip Area lamp's thin side did not sharpen the shadow (SP-2).** A
  0.2 x 2.0 Area lamp with its long side along a wall's shadow edge: the
  edge was 0.145 / 0.250 / 0.270 world units wide at three floor lines
  against the traced 0.045 / 0.085 / 0.115, and the thin Size from 0.05
  to 1.0 moved it by a factor 1.9 where the traced edge grows 4.1 times.
  *What was wrong:* `softmap._filter` took the bilinear lattice's stride
  from the kernel's LONG radius, and a tap's ramp is as wide across the
  kernel's thin side as along its long one. *What it does now:* the
  kernel's blur radius is taken along each axis of the map (`rx`, `ry`);
  from a ratio of 1.5 the stride ACROSS the short axis is divided by the
  largest of 2, 4, 8, 16 not above the ratio, cross-faded to the next like
  the stride itself (`ANISO_FROM`, `ANISO_STEPS`). The edge is 0.090 /
  0.130 / 0.145 -- what a stride of one texel gives (0.085 / 0.125 /
  0.140), so the lattice no longer adds to it -- and the Size series grows
  2.9 times. A round footprint (a disc's reaches a ratio of 1.41 at a cube
  face's edge) keeps its one stride: POINT, SPOT, SUN and square / disc
  Area lamps are bitwise what they are without the change.
  *The review's own proposal was measured and not taken*: a stride from
  the short radius alone (`|A x B| / amax`) gives the same widths and
  leaves 48 % of a ball's penumbra FLAT under the same strip at Samples 4
  (16 % with the strip turned by a quarter) -- the stripes the lattice was
  widened to remove. The per-axis stride leaves 0.00 %.

- **One Sample shrank under a Softness it cannot draw.** One Sample is the
  hard single-tap lookup for every lamp, and its normal offset carried
  `max(1, Softness)`: a point lamp's hard shadow kept 7.48 / 6.20 / 4.78 /
  2.71 of its mass at Softness 1 / 4 / 8 / 16 (a Sun 3.15 / 2.69 / 2.11 /
  0.95). The constant lookup's kernel is never wider than a texel -- a
  wider Softness goes through the plane-biased filter, and at one Sample
  there is one tap and nothing to blur -- so its offset carries no
  Softness factor (`core/lights.visibility`; the factor's texel `h2.y`
  and its literal are 1, the emitted text is unchanged). 7.53 at every
  Softness, bit for bit; at Softness 1 or less the same bits as before.
  The decision was taken by the integrator; the tooltip already said the
  Softness "needs Samples above 1".

- **A scene modelled at centimetre size lost every mapped shadow (SP-3).**
  The Bias is a world length, 0.02 by default; at a hundredth of its size
  the test scene's casters stand nearer to their receivers than that and
  the shadow mass was 0.0 for every lamp, sized or not (traced: 888). The
  maps now carry the radius of the scene they were baked for
  (`softmap.stamp_scene`, called by `core/render._build_shadows` right
  after the bake: the `scene_radius` slot), and under a radius of one unit
  the lookups of both roads run with `Bias x radius`
  (`softmap.scene_bias`): the default is then 2 % of the radius, where the
  shadow is whole (880.6). At a radius of one unit or more the Bias is the
  dial's own value, bit for bit, whatever it is set to. One console line
  says so beside the texel warning: "[Halcyon] shadows: the scene is 0.2
  units across, smaller than the shadow Bias is made for (a world length);
  it is scaled to the scene: 'L' Bias 0.02 -> 0.002. ...". The render
  Bias tooltip gains "In a scene smaller than one unit across its radius
  it scales down with the scene". (The review proposed `min(Bias, 0.02 x
  radius)`; that moves every picture whose Bias is raised above 2 % of its
  scene's radius, a scene of 5 units at Bias 0.2 for one. `Bias x min(1,
  radius)` is the same number at the default and leaves every scene of a
  unit or more alone.)

## The numbers

### SP-1: a Sun over a growing ground (`probes/probe_dmin.py`, 200x150)

The same Sun, objects and world texel (0.058): ground 20 on a 512 map, 80 on
2048, 160 on 4096. Umbra / penumbra pixels / mass of the mapped shadow, and
pixels that moved by more than 0.25 against the 20-unit ground. Traced, 64
rays: 371 / 17 / 382.1 at 0.00918 rad, 331 / 88 / 377.0 at 0.05, 295 / 174 /
375.8 at 0.1, 231 / 320 / 375.6 at 0.2.

| Angle, ground | before | after |
|---|---|---|
| 0.00918, 20 / 80 / 160 | 281/194/373.8, 280/198/372.5, 280/198/372.5 | the same (4 px moved on both trees: the 512 map against the 2048 one) |
| 0.05, 20 | 272 / 218 / 373.9 | 271 / 219 / 373.9 |
| 0.05, 80 | 273 / 203 / 369.5, 6 px moved | 271 / 219 / 372.6, 0 moved |
| 0.05, 160 | 273 / 186 / 365.0, 13 moved | 271 / 219 / 372.5, 0 moved |
| 0.1, 20 | 252 / 261 / 374.2 | 250 / 267 / 374.1 |
| 0.1, 80 | 253 / 225 / 363.4, 14 moved | 247 / 271 / 372.8, 0 moved |
| 0.1, 160 | 250 / 128 / 335.1, 65 moved | 247 / 271 / 372.7, 0 moved |
| 0.2, 20 | 188 / 388 / 366.3 | 184 / 442 / 373.5 |
| 0.2, 80 | 185 / 203 / 317.7, 78 moved | 183 / 440 / 371.8, 0 moved |
| 0.2, 160 | 184 / 201 / 317.3, 78 moved | 183 / 440 / 371.7, 0 moved |

`probe_dmin.py big` (ground 50 on a 512 map against 400 on a 4096 one, texel
0.145): mass at 0.02 / 0.05 / 0.1 rad 365.9 -> 347.4, 361.2 -> 301.2, 342.6
-> 305.3 before (58 / 196 / 120 pixels moved by more than 0.1); 366.7 ->
366.5, 366.8 -> 366.5, 367.0 -> 366.8 after, 0 moved (traced 379.9 / 377.0 /
375.8: within 3.5 %). `probe_ground.py`, the same figure on 20 / 100 / 400
units at 0.2 rad, mass against the traced 375.6:

| ground, map | texel | before | after |
|---|---|---|---|
| 20, 512 | 0.058 | 366.3 | 373.5 |
| 20, 2048 | 0.015 | 370.7 | 374.4 |
| 100, 2048 | 0.073 | 325.7 | 383.0 |
| 100, 512 | 0.29 | 243.5 | 358.1 |
| 400, 2048 | 0.29 | 238.7 | 357.8 |
| 400, 512 | 1.2 | 169.5 | 213.5 |

Within 5 % of the traced mass on every ground whose texel resolves the
figure (0.29 units and finer); on the 400-unit ground at the default 512 map
one texel is 1.2 units, twice the figure's width, and the console's texel
warning stands. POINT and SPOT were independent of the ground and are
(POINT 0.5: 334 / 601 / 603.1 -> 336 / 602 / 603.0 on all three).

### Real Blender 5.2.2, F12 through this tree (`probes/run_bl.sh`, `bl_sheet.py`)

Add > Light > Sun with its Angle at 0.2, the reviewer's scene, 320x240; and
an Area lamp of shape RECTANGLE over a wall, orthographic camera above.
Sheet: `<scratch>/wt/soft-search/out/soft_search_before_after.png` (left the
traced reference, middle the fork point, right this tree).

| case | traced, 64 rays | before | after |
|---|---|---|---|
| Sun, ground 20 (umbra / penumbra / mass) | 586 / 857 / 969.3 | 479 / 1024 / 949.7 | 463 / 1151 / 966.9 |
| Sun, ground 80 | 586 / 857 / 969.4 | 467 / 515 / 815.6, 224 px moved | 461 / 1141 / 962.3, 0 moved |
| Sun, ground 160 | 586 / 857 / 969.3 | 465 / 538 / 821.3, 205 px moved | 461 / 1141 / 962.1, 0 moved |
| Area 0.2 x 2.0, edge width at three lines | 0.050 0.100 0.125 | 0.162 0.250 0.275 | 0.088 0.138 0.150 |
| Area 2.0 x 0.2 | 0.313 0.550 0.800 | 0.312 0.562 0.763 | 0.325 0.550 0.763 |
| Area 1 x 1 | 0.163 0.288 0.425 | 0.175 0.300 0.412 | 0.175 0.300 0.412 |

### SP-2: strip Area lamps (`probes/probe_strip.py`, `probe_strip_size.py`, `aud/stripes.py`)

10-90 % width of the wall's side edge at three floor lines, world units, 512
map, Samples 4 (the traced road: 256 rays).

| lamp | traced | before | after | a stride of one texel |
|---|---|---|---|---|
| RECTANGLE 0.2 x 2.0 | 0.045 0.085 0.115 | 0.145 0.250 0.270 | 0.090 0.130 0.145 | 0.085 0.125 0.140 |
| the same, 2048 map | | 0.135 0.185 0.210 | 0.075 0.125 0.135 | 0.075 0.115 0.135 |
| RECTANGLE 0.1 x 1.0 | 0.025 0.040 0.060 | 0.065 0.140 0.180 | 0.055 0.075 0.095 | 0.055 0.075 0.095 |
| RECTANGLE 0.5 x 1.5 | 0.085 0.145 0.205 | 0.145 0.225 0.295 | 0.115 0.175 0.245 | |
| strip 2.0 x 0.2 (long side across) | 0.305 0.555 0.800 | 0.315 0.550 0.765 | 0.315 0.550 0.765 | |
| SQUARE 1 x 1, DISK 1.0 | | 0.170 0.295 0.410, 0.150 0.265 0.370 | the same | |

Size X from 0.05 to 1.0 at Size Y 2.0, middle line: traced 0.070 -> 0.285
(x 4.1); before 0.200 -> 0.375 (x 1.9); after 0.105 -> 0.300 (x 2.9). The
far line is within 26 % of the traced width; the two nearer lines carry what
a 0.2 SQUARE lamp carries alone on the same map (0.055 0.070 0.095 against
0.035 0.060 0.080): the texel and the user Softness, not the lattice.

Curved receiver, share of a ball's penumbra that is flat / largest neighbour
step (`aud/stripes.py`, a ball in a box's penumbra, Samples 4):

| lamp | before | the review's stride (short radius) | after |
|---|---|---|---|
| AREA 0.2 x 2.0 | 0.00 %, 0.089 | 48.19 %, 0.125 | 0.00 %, 0.089 |
| AREA 2.0 x 0.2 | 0.00 %, 0.091 | 16.21 %, 0.134 | 0.00 %, 0.092 |
| POINT 0.5 / SPOT 0.5 / SUN 0.1 / AREA 1.0 / disc 1.0 | 0.00 % | 0.00 %, steps up to 0.056 from 0.050 | 0.00 %, the steps they had |

### SP-3: the scene at a fraction of its size (`probes/probe_scale_bias.py`, default Bias)

| scale | traced mass | before: sized / size 0 | after: sized / size 0 |
|---|---|---|---|
| 0.01, POINT | 888.2 | 0.0 / 0.0 | 880.6 / 853.8 |
| 0.01, SUN | 410.5 | 0.0 / 0.0 | 414.9 / 433.0 |
| 0.02, POINT | 890.0 | 637.2 / 608.0 | 880.6 / 853.8 |
| 0.03, POINT | 890.6 | 829.7 / 795.0 | 880.6 / 853.8 |
| 0.05, POINT | 891.7 | 879.3 / 853.0 | 880.6 / 853.8 |
| 0.10 and 1.00, POINT | 892.7 | 880.6 / 853.8 | 880.6 / 853.8 |

### Samples 1 under a Softness (`tests/test_r254_soft_search.py` h; mass behind the wall)

| lamp at Samples 1 | Softness 1 / 4 / 8 / 16 before | after |
|---|---|---|
| POINT, size 0 | 7.48 / 6.20 / 4.78 / 2.71 | 7.53 at each, one array |
| SUN, size 0 | 3.15 / 2.69 / 2.11 / 0.95 | 3.25 at each |
| SPOT, size 0 | | 7.39 at each |
| POINT, Radius 0.25 | the size-0 shadow, shrinking alike | 7.53 at each |

(The "before" row is the map-gpu write-up's, on its own floor grid; the
Softness-1 frame itself is bitwise the fork point's.) No shipped preset
pairs mapped shadows at Samples 1 with a Softness above 1: of 121 presets,
the six at Samples 1 are four traced ones and two mapped ones at Softness 0.

### The review's other probes, before and after

Pixels where the mapped term is lighter (leak) / darker (dark) than 64 traced
rays by more than 0.25.

| probe, case | before: leak / dark | after: leak / dark |
|---|---|---|
| interior: POINT 0.1 in the room (of 29214) | 17 / 22 | 16 / 19 |
| interior: POINT 0.25 | 3 / 28 | 0 / 19 |
| interior: POINT 0.5 | 31 / 51 | 31 / 51 |
| interior: SPOT 0.25 (of 12555) | 8 / 4 | 3 / 4 |
| interior: AREA 1.0 on the ceiling (of 23005) | 38 / 0 | 32 / 0 |
| interior: SUN 0.05 through the window, SUN at Blender's default | 7 / 4, 23 / 40 | 7 / 4, 23 / 40 |
| interior: POINT 0.25 outside the window | 35 / 3 | 35 / 3 |
| curved: POINT 0.1, 512 / 2048 (of 23492) | 134 / 111, 37 / 47 | 136 / 111, 37 / 44 |
| curved: POINT 0.5 | 31 / 299, 23 / 265 | 31 / 298, 23 / 265 |
| curved: SPOT 0.25 | 77 / 177, 47 / 121 | 77 / 177, 47 / 121 |
| curved: SUN at the default, SUN 0.1 | 91 / 198, 40 / 156 | 91 / 198, 40 / 154 |
| curved: AREA 1.0 | 38 / 313, 30 / 288 | 38 / 315, 27 / 288 |
| seams: floor R 0.1 / 0.25 / 0.5, the seam band (315 px) | 2/0, 2/1, 2/97 | 2/0, 2/1, 2/97 |
| seams: floor, elsewhere (28690 px) | 10/3, 16/14, 51/392 | 11/2, 12/16, 51/392 |
| seams: wall, elsewhere (26460 px) | 5/2, 11/18, 24/249 | 3/1, 5/21, 24/249 |
| seams: ceiling, elsewhere (27891 px) | 6/1, 16/8, 36/271 | 5/0, 16/8, 36/271 |
| thin walls: POINT outside / SUN / POINT inside / AREA | 90/0, 357/9, 0/25, 154/0 | 90/0, 357/9, 0/17, 154/0 |
| options, POINT 0.25: Softness 0 / 4 / 8 | 77/8, 157/66, 274/182 | 77/8, 157/67, 274/183 |
| options, POINT 0.25: Samples 2 / 16, Map 256 / 2048 | 82/9, 90/8, 186/63, 6/0 | 83/9, 90/8, 186/64, 6/0 |
| options, POINT 0.25: Bias 0.0001 / 0.2 / 0.5 | 81/26, 136/4, 440/1 | 82/26, 138/4, 442/1 |
| multi (four sized lamps): mean abs error of the frame | 0.0827 | 0.0725 |

The four-lamp frame is still the sum of the four single-lamp frames exactly
(max difference 0). Where a count moves it moves by one to three pixels in
23,000 either way, except the seam probe's wall at R 0.25 (dark 18 -> 21
beside leak 11 -> 5); the cases that are two pixels worse are the curved
scene's POINT 0.1 on the 512 map (leak 134 -> 136), its AREA 1.0 (dark 313
-> 315) and the options scene at Bias 0.2 and 0.5 (leak +2). The fixture
probe (a plate 0.3 above a point lamp), floor pixels moved by more than
0.05 / 0.25:

| lamp, map | before | after |
|---|---|---|
| R 0.1, 512 / 2048 | 67 / 13, 82 / 10 | 0 / 0, 0 / 0 |
| R 0.25, 512 / 2048 | 189 / 24, 172 / 15 | 2 / 0, 0 / 0 |
| R 0.5, 512 / 2048 | 13 / 0, 13 / 0 | 2 / 0, 9 / 0 |

### The map-soft acceptance scene and wall (which sized pictures move, and by how much)

`aud/census2.py mass` (320x240, floor + box + ball, 512 map, Samples 4):
umbra / penumbra pixels / mass.

| lamp | before | after |
|---|---|---|
| POINT 0.1 | 2840 / 433 / 3052.0 | 2840 / 433 / 3052.0 |
| POINT 0.5 | 1967 / 2341 / 3086.6 | 1969 / 2340 / 3086.9 |
| POINT 2.0 | 1219 / 4866 / 3200.5 | the same |
| SPOT 0.1 | 2851 / 453 / 3070.4 | 2851 / 453 / 3070.6 |
| SPOT 0.5 | 2077 / 2081 / 3061.4 | 2076 / 2089 / 3062.3 |
| SPOT 2.0 | 647 / 6887 / 3170.5 | the same |
| SUN 0.05 | 1428 / 423 / 1633.1 | 1426 / 424 / 1633.0 |
| SUN 0.2 | 1071 / 1113 / 1616.8 | 1066 / 1131 / 1616.3 |
| SUN 0.6 | 326 / 3083 / 1656.7 | 273 / 3282 / 1663.1 |
| AREA 0.25 | 2818 / 488 / 3055.3 | 2819 / 486 / 3055.3 |
| AREA 1.0 | 1940 / 2405 / 3069.9 | 1940 / 2408 / 3070.2 |
| AREA 3.0 | 1323 / 4608 / 3160.1 | the same |
| POINT / SPOT / SUN 0, AREA 0.05 | 2964.5, 2988.8, 1675.0, 3057.6 | the same |

The wall's edge widths at four floor lines (`ms/widths.py`, 512 map; traced:
256 rays): POINT 0.1 / 0.25 / 0.5 / 1.0 / 2.0 and AREA 0.5 / 2.0 are the
fork point's to the third decimal; the lines that move:

| lamp | traced | before | after |
|---|---|---|---|
| SPOT 1.0 | 0.230 0.597 0.966 1.324 | 0.242 0.604 0.913 1.165 | 0.235 0.602 0.954 1.200 |
| SPOT 0.5 | 0.117 0.298 0.479 0.647 | 0.130 0.315 0.461 0.650 | 0.130 0.315 0.461 0.655 |
| SUN 0.1 | 0.051 0.153 0.253 | 0.087 0.171 0.261 | 0.088 0.173 0.263 |
| SUN 0.2 | 0.103 0.307 0.516 | 0.143 0.332 0.510 | 0.142 0.333 0.531 |
| SUN 0.6 | 0.324 0.954 | 0.451 0.893 | 0.451 0.931 |
| AREA 1.0 | 0.135 0.343 0.553 0.740 | 0.135 0.340 0.547 0.744 | 0.140 0.340 0.547 0.744 |

## Which pictures change

- A lamp WITH a size under Shadow Maps at Samples above 1: its shadow, by
  the per-receiver bound of the search. On small scenes by little (the
  tables above: a mass within 0.4 %, a width within 0.04; the map-gpu
  neutrality scenes move by at most 0.002 of a pixel value, an Area lamp
  by 0.28 in a frame clamped at 64); on a large ground, or with a caster
  near the lamp, by what was wrong.
- A strip footprint (an Area lamp, or any lamp whose kernel is 1.5 times or
  more longer along one map axis than the other): the edge across its thin
  side.
- A lamp at Samples 1 under a Softness above 1: its hard shadow no longer
  shrinks.
- Any mapped shadow in a scene whose radius is under one unit: the Bias.
- Nothing else. The map-gpu neutrality dump (`mg/neutral_dump.py`,
  `neutral_cmp.py`: CPU frames, simulator frames and depth packs) against
  the fork point: of 181 arrays 32 differ, each one named above -- the four
  kinds at Samples 1 under Softness 4 (8 frames), the sized lamps at
  Softness 1 and 4 (16 frames), and the eight depth packs of the sized
  lamps (they gain their pyramid's rows). The other 149 are bitwise the
  fork point's: lamps of size 0 at Softness 0 / 0.5 / 1, Samples 1 / 7 /
  16, a MIDPOINT map, the per-lamp dials, a 256 map, no shadow, traced hard
  and soft, Area discs, a wide Sun, the demo scene mapped, traced and with
  shadows off. So are the 16 frames of a size-0 lamp under a Softness
  above 1 (the skips are exact; their packs gain the rows). Of 74 GLSL
  texts 58 are identical and 16 changed: the per-receiver lookup's.

## The GPU road

`gpu/softmap.shadow_function` emits the same statements: `hal_sm_cell{i}`
(one cell of the pyramid: the level by a comparison ladder, the tile by a
division by a power of two, exact on both roads), `hal_sm_near{i}` (the
nearest depth within r, with a cube's floor), the up-to-six tightening
steps as nested branches, the level-receiver test, the per-axis lattice.
The pyramid is rows of the depth pack: a lamp on the per-receiver lookup
carries, under its depth cells, one block per face on the faces' own grid
(`softmap.pyr_layout`: a block is its depth part and its height part side
by side, 388 x 135 texels for a 512 face, 1540 x 521 for a 2048 one;
`gpu/shade._shadow_meta` builds the band, `_shadow_pack` stacks it): 26 %
more texels for that lamp, a flat map or a cube. That the rows exist is
layout -- the lamp's pack
key gains `'PYR'` -- and their content is derived from the depth texels
alone, so the stamp is still the faces' digests: one resident pack per
layout, replaced when a map changed. No new value texel (the pad and the
nearest depth of the whole map were there); the Samples-1 factor `h2.y` is
1; the Bias texel `h0.w` is the scene-bounded Bias.

The simulator against the CPU (`probes/probe_gpu.py`, 160x120, the review's
twelve cases): 5e-6 to 1.9e-4 on eleven and 4.2e-3 with four lamps (a pixel
value of 47.8), as on the fork point. This package's cases
(`tests/test_r254_soft_search.py` e, g, h, i, j; `aud/par_strip.py`), both
value roads (`gpu/shade.SHADOW_TEXELS` on and off): SUN 0.2 on an 80-unit
ground 1.8e-5; POINT 0.25 with the plate (six faces, seams) 4e-6; SPOT at
Samples 16 1e-6; a lamp of size 0 at Softness 4 1e-6; the scene at scale
0.01 1e-6 to 5e-6; Samples 1 at Softness 8 0; slanted SUN / SPOT / POINT
over a floor 1e-6 to 2e-6; Area lamps 7e-4 to 3.0e-3 (strips 1.2e-3 and
1.6e-3), which is the Area lamp's own lighting term as before. All under
6e-3. The level-receiver branch itself cannot show in a frame (it returns
what the search would); a copy of the pass text that paints it does: it
decides 4871 of 5285 pixels under the slanted Sun, 3091 under the Spot,
1134 under the Point lamp (whose floor mostly passes the depth test).
`tests/glsl_strict.py`: "every shader compiles under strict GLSL 4.30 (526
distinct texts)", mapped shadows 28 of 28 on the two value roads.

One CPU / simulator pin moved (below): the bound is piecewise constant over
tiles, so a receiver within a thousandth of a texel of a tile border can
read the neighbouring cell on the road whose position differs in its last
bits. Both cells are valid bounds; the search then stands at a slightly
different reach. One pixel of 5285 at 1.0e-4 in one case.

Depth-pack fetches per point looked up, counted on the CPU road's own
stages over the acceptance frame (`aud/census2.py fetch`; the two roads run
the same branches):

| lamp | Samples 4: before -> after | Samples 16 | points that skip the search |
|---|---|---|---|
| POINT 0.1 | 14.9 -> 7.1 | 38.1 -> 14.0 | 81 % |
| POINT 0.5 | 21.0 -> 15.8 | 56.1 -> 37.1 | 61 % |
| POINT 2.0 | 29.3 -> 27.8 | 93.5 -> 85.7 | 31 % |
| SPOT 0.1 | 12.3 -> 6.9 | 33.7 -> 14.1 | 68 % |
| SPOT 0.5 | 18.3 -> 14.8 | 49.6 -> 33.8 | 54 % |
| SPOT 2.0 | 32.9 -> 30.6 | 107.8 -> 98.0 | 38 % |
| SUN 0.05 | 14.5 -> 7.0 | 39.3 -> 14.0 | 87 % |
| SUN 0.2 | 17.6 -> 11.2 | 45.6 -> 21.8 | 78 % |
| SUN 0.6 | 23.0 -> 21.4 | 63.4 -> 50.0 | 38 % |
| AREA 0.25 | 15.0 -> 7.4 | 39.0 -> 14.8 | 81 % |
| AREA 1.0 | 21.2 -> 16.3 | 58.9 -> 40.9 | 58 % |
| AREA 3.0 | 27.8 -> 26.2 | 88.3 -> 80.3 | 31 % |

Per fragment: 2.4 to 4.7 texels of the pyramid (the tightening steps and
the height), then nothing for a skipped one; 13 for the search, 13 for the
probe and 64 per lattice as before for the rest. With the per-receiver
bound alone (before the level-receiver test) a Sun paid MORE than on the
fork point -- 17.3 against 14.5, the bound's texels on top of a search that
its own ground never let it skip -- which is why the height part exists.

Text of one sized lamp's shadow function (`ms/glsl_size.py`, the literal
road): POINT 0.3 17,882 -> 22,404 bytes at Samples 4 and 35,463 -> 40,191
at 16; SPOT 13,249 -> 17,430 and 29,810 -> 34,197; SUN 12,689 -> 16,685 and
29,250 -> 33,452; AREA 18,469 -> 22,991 and 36,050 -> 40,778. A lamp of
size 0 on the constant lookup: 4,161 and 7,353 bytes, unchanged.
