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
