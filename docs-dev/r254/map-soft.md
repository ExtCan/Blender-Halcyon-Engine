# R254 map-soft: under Shadow Maps a lamp's size shapes its shadow

Work-package key `map-soft`. The integrator owns CHANGELOG.md / README.md /
the version stamps; the fragments below are in their house voice. Findings
covered: size-map-01, -02, -03, -04, -05, -07 (the map-cache half), -08, -09,
and the size-driven part of sun-angle-map-01, native-size-under-maps and
soft-erode-01. Every number below was measured on this tree; the probes are
in `<scratch>/wt/map-soft/` (`aud/widths.py`, `aud/stress.py`,
`aud/stress2.py`, `aud/seam.py`, `aud/gpu_par.py`, `aud/perf.py`,
`aud/census.py`, `aud/zero_dump.py`, `soft_metrics.py`), the outputs in its
`out/`.

## CHANGELOG

- **What was reported.** "Fix the lights so that changing the size of the
  lights actually works. The lights don't get softer / harder with size."
  Under Shadow Maps (the render default) the size of a lamp -- POINT / SPOT
  Radius, SUN Angle, AREA Size and Size Y -- did not soften its shadow: it
  shrank it, split its edge into four hard copies and, for a Sun, erased it.
  Shadow mass on the test floor (320x240, floor + box + ball), radius
  0 / 0.1 / 0.5 / 2: POINT 2964 -> 2701 -> 1957 -> 1247; a Sun at 0.2 rad:
  0 pixels of shadow; three grey levels in every penumbra; the same lamp
  softer on a 512 map than on a 2048 one.

- **What was wrong** (all in `core/lights.py`, mirrored statement for
  statement by `gpu/shade._shadow_meta` and `gpu/material._shadow_function`).
  `_build_shadow_maps._stamp_soft` turned the lamp's size into ONE number per
  map, `soft_extra`: a world radius (`_soft_world`: the radius, half the MEAN
  of an Area's two sizes, `tan(angle / 2)` times the ortho camera's stand-off
  for a Sun) divided by the texel at one representative distance -- the
  scene's bounding-box centre -- halved and clamped to 24 TEXELS.
  `visibility` added it to the PCF radius, where `ShadowMap.lookup` spread
  the render's Samples (4) nearest-texel binary taps over it, the same four
  offsets for every pixel: four displaced hard copies. And the normal offset
  that keeps a wide binary kernel from self-shadowing was multiplied by the
  same number, `texel * (1.5 + 2.5 sin) * max(1, soft)`: at 24 texels the
  lookup point sat 1 to 3 world units above the floor, over the caster, so
  the floor read lit. No caster-to-receiver distance entered anywhere.

- **What it does now.** A sized lamp at more than one Sample takes a
  per-receiver lookup, `core/softmap.lookup` (the module docstring is the
  algorithm, statement by statement), and `gpu/softmap.shadow_function`
  emits the same statements as GLSL:
  1. the receiver PLANE as a depth function of the texel offset (1/depth is
     linear under a perspective map, depth under an ortho one), never deeper
     than the receiver itself; every compare below is made against it at the
     tap's own texel centre, so a wide kernel needs no grown bias and no
     grown offset;
  2. the lamp as two texel vectors per unit of `t = 1/d - 1/d_r`
     (perspective) or `d_r - d` (ortho): a disc of the Radius facing the
     receiver, a Sun's `tan(angle / 2)`, an Area's two half-axes projected
     through the map face (a rectangle as the disc of equal area);
  3. a 13-tap BLOCKER SEARCH over the largest footprint any caster could
     give (bounded by the nearest depth the map holds and by a cap that is
     an ANGLE at the lamp: tangent 1/8, i.e. a sixteenth of a cube face and
     the same angle for a Spot's cone of any width; a sixteenth of an ortho
     map): no blocker -> lit, all blocked -> umbra;
  4. the penumbra by similar triangles from the mean blocker depth, a PROBE
     with the same taps at that radius (none / all -> lit / umbra), and the
     estimate once more from the casters the probe found;
  5. a FILTER of `F = clamp(4 * Samples, 16, 64)` taps, each a bilinear
     compare on a lattice whose stride follows the radius (two strides
     cross-faded), so the ramp is continuous at any tap count. The lamp's
     penumbra and the user Softness add in quadrature.
  A cube map's kernel crosses face seams by reading the neighbouring face
  along the same direction at lookup (`softmap._CubeFetch`), with the stored
  depth re-expressed along this face's axis. Nothing is added to the baked
  map: no guard ring, no second copy of the depth, no bake cost.
  `ShadowMap` stores two things at bake that the lookup reads -- its three
  axes and its nearest depth (`axes`, `d_min`; `CubeShadow.d_min`) --
  neither of which depends on the lamp's size.

- **Samples 1 stays hard for every lamp** (`softmap.lookup_shape`): one
  Sample is the constant single-tap lookup whatever the lamp's size, the
  rule the traced road already had (one sample, one hard ray). The "hard
  shadow maps" presets ask for exactly that with Samples 1 on lamps that
  carry Blender's default sizes; before, such a lamp had a hard edge that
  still shrank (POINT radius 0.25: area 7.48 -> 5.34), now it is bitwise the
  size-0 lamp's shadow. The lamp's own Samples dial (0 = the render's)
  decides per lamp through `lights.shadow_sample_count`.

- **A size edit re-bakes nothing** (size-map-07). The lamp's size left
  `lights._light_shadow_signature`: the map carries nothing derived from it.
  A Radius drag on a 2048 point lamp went from 6 rasterisations and 2.0 to
  2.7 s per step to 0 rasterisations and 0.19 to 0.29 s (`aud/perf.py drag`),
  and the per-size copies of the maps the cache used to keep (up to 65) are
  gone with the key.

- **Where this departs from the audit's prototype, and why** (each measured;
  the verifier's corrections are marked V):
  - *No guard ring.* The prototype padded every cube face with 57 / 213
    texels resampled from its neighbours (2 to 5 s per bake at 2048, +140 MB
    of depth, a 2048 cube atlas growing from 384 to 560 MB). Reading the
    neighbour at lookup gives the same seam (largest step along a seam-hugging
    shadow edge 0.005 to 0.012, rms against 256 rays 0.012 to 0.032; clamped
    at the face edge the step is 1.0) for nothing: `aud/seam.py`.
  - *The plane depth is clamped at the receiver's own.* A plane continued
    past a concave corner runs behind the neighbouring surface and reads it
    as a caster. Wall face above a floor, 512 map, darkest term: Sun 0.05
    0.57 -> 0.94, Sun 0.2 0.61 -> 0.83, Spot 0.5 0.80 -> 0.99 (0.95 to 1.00
    with the rest of this list); flat receivers unchanged to the digit.
  - *Slope clamp at N.L 0.1 instead of 0.25.* A lit ramp 82 degrees from the
    light under a point lamp of radius 1: mean false shadow 0.082 -> 0.000.
  - *The normal offset shrinks where the plane is exact*
    (`softmap.offset_scale`: a quarter of the classic offset for N.L above
    0.3, the whole of it at 0.1 and below, where the slope clamp bites). The
    lifted lookup point is what displaced the shadow and striped a thin
    caster lying close to the floor (the point landed inside it). Against the
    traced road on three smooth spheres: mean term error 0.0126 -> 0.0040,
    pixels darker by 0.25: 84 -> 31 of 5496 (a size-0 lamp: 79); two-layer
    line, 512 map: rms 0.248 -> 0.102, largest step 0.318 -> 0.057; area
    darker than half behind the wall, POINT radius 0.25, 512 map: 7.43 ->
    7.78 (traced 7.80). A lit ramp at 60 to 82 degrees still carries no
    false shadow; Bias 0 stays clean.
  - *The estimate is taken once more from the probe's own casters.* Free (the
    probe already fetches). Near the foot of the wall, 2048 map, against the
    traced width: POINT radius 1: 0.281 -> 0.243 (traced 0.230); Sun 0.2:
    0.164 -> 0.149 (traced 0.103); Sun 0.6: 0.790 -> 0.456 (traced 0.324).
  - *Penumbra and Softness add in quadrature* instead of plainly. The 512 and
    the 2048 map now agree with each other and with the traced road: POINT
    radius 0.25, four floor lines, 512: 0.089 / 0.180 / 0.270 / 0.363 ->
    0.069 / 0.153 / 0.244 / 0.331; 2048: 0.059 / 0.146 / 0.237 / 0.325;
    traced: 0.059 / 0.150 / 0.242 / 0.326.
  - V *No softness factor in the offset.* A sized lamp no longer erodes at
    any Softness (below).
  - V *The plain mean of the blocker depths on both roads* (the prototype's
    CPU side kept a leftover of a dropped trial; it was the whole of its
    CPU / GPU gap).
  - V *Samples 1 = the hard lookup* (above), and F / B come from
    `shadow_sample_count`.
  - V *Real branches in the shader* (below).
  - V, modified: *the cap.* The verifier measured a narrow spot going dead
    under size/16 texels (cone 0.35 rad: radius 0.5 and 1.0 both 0.320) and
    proposed size/4 for Spot and Sun maps. An angle at the lamp is the rule
    that makes a Point and a Spot of one Radius agree: cone 0.35 rad now
    gives 0.474 / 0.973 for radius 0.5 / 1.0 (traced 0.479 / 0.966). For a
    Sun size/4 was measured and NOT taken: at 0.6 rad the twelve search taps
    step over casters (largest step 0.31 to 0.36 against 0.03 to 0.07 at
    size/16). Doubling the angle for cubes was measured and not taken either:
    far lines of radius 1 get closer (0.94 / 1.26 against traced 0.97 / 1.32,
    from 0.82 / 1.05) but the umbra erodes (821 px against traced 1796, from
    1197) and the estimate turns patchy.

- **Softness on a sized lamp** (soft-erode-01, the sized half). Shadow mass
  behind the wall at Softness 1 / 4 / 8 / 16, 512 map: POINT radius 0.25:
  7.89 / 7.90 / 7.88 / 7.88 (the prototype kept `max(1, S)` in the offset:
  area 7.43 / 6.07 / 4.58 / 2.53); Sun 0.05: 3.29 / 3.29 / 3.28 / 3.14. The
  edge still widens with the dial. A lamp of size 0 keeps its arithmetic
  bitwise in this package, erosion included (POINT 7.47 -> 2.67 at 16): that
  is the other decision the audit names.

- **The tooltips** (size-map-05, -09). Render Softness: "A constant blur of
  mapped shadows, in shadow-map texels, added to the penumbra a lamp's own
  size (Radius, Angle, Size) casts. 0 adds none. It needs Samples above 1".
  Render Samples: "Quality of soft shadows. 1 is a hard edge for every lamp.
  Above 1, mapped: a lamp with a size filters 4 taps per sample (16 to 64)
  into a smooth penumbra and a lamp of size 0 takes this many taps of the
  Softness blur; traced: rays per pixel from a sized lamp. More is smoother
  and slower". Light Softness: "Multiplies the render's Softness for this
  light's mapped shadow: 1 is the render setting, 0 adds no blur. The
  penumbra itself comes from the lamp's size" (it was never texels). Light
  Samples gains "1 is a hard edge".

## The numbers

Wall standing on a floor, lamp 7 units away; 10 to 90 % width of one shadow
edge in WORLD units at four floor lines behind the wall (the caster-to-
receiver distance grows left to right); traced = the lamp's true shape, 256
stratified rays (`aud/widths.py`).

| lamp | traced | map 512 | map 2048 | old, 512 |
|---|---|---|---|---|
| POINT 0.1 | 0.024 0.061 0.095 0.132 | 0.040 0.073 0.103 0.136 | 0.025 0.060 0.096 0.131 | 0.078 0.070 0.081 0.093 |
| POINT 0.25 | 0.059 0.150 0.242 0.326 | 0.069 0.153 0.244 0.331 | 0.059 0.146 0.237 0.325 | 0.116 0.140 0.166 none |
| POINT 0.5 | 0.117 0.298 0.479 0.647 | 0.131 0.304 0.479 0.599 | 0.118 0.295 0.471 0.641 | 0.228 0.256 none none |
| POINT 1.0 | 0.230 0.597 0.966 1.324 | 0.243 0.590 0.824 1.045 | 0.238 0.573 0.816 1.043 | 0.398 none none none |
| SPOT 0.25 | 0.059 0.150 0.242 0.326 | 0.076 0.152 0.251 0.320 | 0.057 0.146 0.233 0.314 | 0.183 0.198 0.213 none |
| SPOT 0.5 | 0.117 0.298 0.479 0.647 | 0.130 0.308 0.473 0.613 | 0.115 0.293 0.483 0.611 | 0.285 0.308 none none |
| SUN 0.05 | 0.026 0.077 0.127 | 0.068 0.101 0.140 | 0.035 0.084 0.135 | 0.363 0.286 none |
| SUN 0.1 | 0.051 0.153 0.253 | 0.087 0.171 0.260 | 0.066 0.169 0.263 | 0.536 none none |
| SUN 0.2 | 0.103 0.307 0.516 | 0.143 0.339 0.516 | 0.135 0.337 0.527 | none none none |
| AREA 0.5 | 0.067 0.171 0.275 0.376 | 0.075 0.176 0.271 0.360 | 0.066 0.163 0.263 0.354 | 0.116 0.140 0.166 none |
| AREA 1.0 | 0.135 0.343 0.553 0.740 | 0.141 0.349 0.534 0.745 | 0.132 0.335 0.527 0.711 | 0.228 0.256 none none |

Levels across a penumbra: 18 to 390 (old: 5); largest single step on a line
sampled every 0.0025: 0.02 to 0.06 for the sizes above at 512 (old: 0.25).
A Sun's 512 map spans the whole scene (texel 0.041 here), which is the
0.03 to 0.04 its nearest line is wider than the traced one.

The acceptance scene (`soft_metrics.py`, 320x240, floor + box + ball,
512 map, Samples 4): umbra / penumbra pixels, shadow mass, grey levels,
neighbour roughness.

| lamp, size | old mass | new: umbra | penumbra | mass | levels | rough | traced mass |
|---|---|---|---|---|---|---|---|
| POINT 0 | 2964.5 | 2914 | 121 | 2964.5 | 3 | 0.119 | 3127 |
| POINT 0.1 | 2701.0 | 2845 | 432 | 3054.6 | 275 | 0.070 | 3098 |
| POINT 0.5 | 1957.0 | 1944 | 2336 | 3058.1 | 760 | 0.023 | 3126 |
| POINT 2.0 | 1247.2 | 1192 | 4819 | 3157.4 | 875 | 0.011 | 3860 |
| SPOT 0.1 | 2726.8 | 2851 | 453 | 3070.4 | 342 | 0.080 | 3098 |
| SPOT 0.5 | 2000.8 | 2079 | 2101 | 3060.2 | 759 | 0.021 | 3126 |
| SPOT 2.0 | 1411.8 | 677 | 6984 | 3187.3 | 887 | 0.009 | 3860 |
| SUN 0 | 1675.0 | 1557 | 230 | 1675.0 | 3 | 0.095 | 1634 |
| SUN 0.05 | 1610.0 | 1428 | 423 | 1633.1 | 328 | 0.064 | 1625 |
| SUN 0.2 | 0.0 | 1081 | 1115 | 1625.2 | 623 | 0.024 | 1616 |
| SUN 0.6 | 0.0 | 321 | 3104 | 1629.4 | 846 | 0.020 | 1563 |
| AREA 0.05 | 2899.0 | 2933 | 238 | 3057.6 | 199 | 0.159 | 3115 |
| AREA 0.25 | 2644.2 | 2820 | 490 | 3055.2 | 312 | 0.050 | 3096 |
| AREA 1.0 | 1957.0 | 1932 | 2390 | 3050.3 | 749 | 0.019 | 3122 |
| AREA 3.0 | 1247.2 | 1328 | 4518 | 3121.1 | 857 | 0.011 | 3431 |

The mass of a sized lamp sits 3 % above its size-0 mass because the size-0
lookup keeps the classic normal offset (which costs it 5 % against the
traced 3127) and the sized one does not; both are within 7 % of the traced
road until the lamp is very large. The roughness column is the metric's
own reading of a NARROW edge (a traced edge of the same width reads 0.11 to
0.14): it is under 0.03 wherever the penumbra is wider than a few pixels.

GPU simulator against the CPU (`aud/gpu_par.py`, 160x120): POINT 0.25 / 1.0
1e-6, SPOT 3e-6, SUN 6e-6, also at Samples 16, Softness 0 / 4 and on a
MIDPOINT map; AREA 0.4e-3 to 2.8e-3, which is the Area lamp's own lighting
term (the same gap, on the same pixels, with the lamp's shadow set to NONE:
0.002762 over 455 px against 456 px with it).

No leak, no acne: floor under the wall's foot, darkest reference 0: POINT
0.25 / 0.5, SPOT 0.5, SUN 0.1, AREA 1.0 all under 0.02; lit ramp at 60 / 75 /
82 degrees: false shadow 0.0000; concave corner, wall face: min 0.95 to 1.00
(prototype 0.57 to 0.96); Bias 0: false shadow 0.00000 on the open floor;
low-poly smooth spheres (12 and 8 segments), 512 map: 0 to 4 lit-side pixels
darker than the traced road by 0.1.

## Which pictures change

- A lamp WITH a size under Shadow Maps at Samples above 1: its shadow. That
  is every Blender lamp at its defaults (point 0.1 / 0.25, Sun 0.526 deg,
  Area 0.25) and every `Light` built with `area_size` left at its (1, 1).
- A sized lamp at Samples 1: its shadow is now the size-0 lamp's (it used to
  shrink with the size and keep a hard edge).
- Nothing else. A lamp of size 0 takes `ShadowMap.lookup` through the old
  statements: 112 arrays (CPU frames, simulator frames and packed atlases of
  POINT / SPOT / SUN / AREA at size 0 over two scenes, Softness 0 / 1 / 4,
  Samples 1 / 4 / 16, a MIDPOINT map, per-lamp softness / samples / bias /
  density, a 256 map, the demo scene on both shadow methods) and 74 texts
  (every pass's GLSL and every atlas key) are identical between the fork
  point (7e9a718) and this tree: `aud/zero_dump.py`, `aud/zero_cmp.py`,
  "112 arrays, 74 texts compared, 0 differ". Real Blender 5.2.2 F12, size 0:
  POINT, SPOT and SUN frames bitwise the old tree's.

## The GPU road

`gpu/shade._shadow_meta` adds `meta['soft']` (the shape and the map's axes)
for a sized lamp at Samples above 1 and nothing otherwise;
`gpu/material._shadow_function` hands such a meta to
`gpu/softmap.shadow_function` and keeps its text for every other. What is
baked as a literal is structure: the size (radius, an Area's rectangle,
axes and shape) joins the `'map'` group of `_light_sig`. What rides a texel
is not: the map's nearest depth sits in the G channel of the lamp's first
atlas texel (the atlas key gains `('sized', d_min)` for such a lamp only),
so a caster moving nearer re-uploads the atlas and leaves the text alone
(pinned: two scenes, `d_min` 3.85 -> 3.10, same text, different key). The
atlas keeps its size and layout: no padded cells.

Texel fetches per fragment at Samples 4 (16 filter taps, 12 + 1 search
taps); the three stages are real branches, which the simulator mirrors by
evaluating both sides and selecting:

| fragment | fetches |
|---|---|
| a lamp of size 0 (unchanged) | 4 |
| lit or umbra by the search | 14 |
| decided by the probe | 27 |
| penumbra, one lattice | 91 |
| penumbra, two lattices cross-faded | 155 |
| the prototype, every fragment | 154 |

Counted over whole frames of the acceptance scene (`aud/census.py`): 15 to
27 fetches per fragment on average (POINT 0.1: 15.6; 0.5: 22.0; 2.0: 23.1;
SUN 0.05: 15.7; 0.6: 26.3; AREA 1.0: 22.0), 38 to 79 at Samples 16.
`tests/glsl_strict.py` gains `check_mapped_shadows` (the sweep planned with
shadows off, so no shadow function had ever reached the strict front-end):
11 of 11 mapped cases compile, sized and not, MIDPOINT and Samples 16
included.

## Cost

640x480, two lamps (one Sun), six 96-segment spheres and a floor (55k
triangles), Samples 4, best of three warm frames, shared machine
(`aud/perf.py time`):

| lamps | old cold / warm | new cold / warm | traced, 4 rays |
|---|---|---|---|
| POINT 0.3, 512 | 1.36 / 0.51 | 1.83 / 0.82 | 6.23 / 5.15 |
| POINT 0.3, 2048 | 4.37 / 0.56 | 5.59 / 0.95 | |
| SPOT 0.3, 512 | 0.75 / 0.50 | 0.98 / 0.71 | |
| SPOT 0.3, 2048 | 2.58 / 0.49 | 2.73 / 0.70 | |
| SUN 0.05, 512 | 0.56 / 0.44 | 0.55 / 0.42 | 2.42 / 1.98 |
| SUN 0.05, 2048 | 1.22 / 0.45 | 1.13 / 0.42 | |
| AREA 1.0, 512 | 1.47 / 0.84 | 1.76 / 1.13 | |
| AREA 1.0, 2048 | 5.64 / 0.80 | 5.09 / 1.26 | |
| POINT 0, 512 / 2048 | 1.60 / 0.53, 4.71 / 0.50 | 1.48 / 0.62, 6.08 / 0.56 | |

A warm frame costs 0.95 to 1.7 times the old one and a fifth of the traced
road at the same Samples; a cold frame pays the lookup and nothing at bake
(the cold spread between the two columns at 2048 is the shared machine: the
bake is the same code). The prototype's cold 2048 point frame was 11.3 s.

## Limits, as measured

- **Very large lamps.** One mean caster depth cannot describe a lamp several
  units wide. POINT radius 2 at 7 units: the far penumbra stops at the cap
  (0.76 / 0.82 / 1.05 against traced 1.18 / 1.75 / 2.06) and the shadow is
  darker and tighter than the traced one (umbra 1192 px against 783); it
  stays whole and smooth. A Sun past about 0.3 rad shows a blotchy core at
  Samples 4 (the twelve search taps); Samples 16 gives a clean one.
- **Thin casters under wide lamps.** A caster much narrower than the lamp's
  footprint can fall between the search taps: poles 0.06 to 0.25 wide under
  POINT radius 0.25 are clean (largest step 0.05 to 0.09), under radius 0.5
  a handful of steps of 0.25 appear on a line of 2800 samples, under radius
  1.0 up to 0.44 (0.26 at Samples 16). The cure is a conservative depth
  pyramid for the search, not more taps; see the follow-up.
- **Resolution.** Every edge carries the map's own texel and the user
  Softness (1 texel at the default): a Sun over a 14-unit scene on a 512 map
  cannot be sharper than about 0.06 at the contact.
- **Leaving size 0.** Size 0 keeps the classic four-tap lookup and its normal
  offset; the smallest non-zero size takes the new one. The edge turns from
  four levels to a ramp and the shadow moves by the offset it no longer
  carries (1.5 to 4 texels towards the caster). It is a step towards the
  traced picture, and it is a step.
- **Roads.** CPU F12 and the GPU simulator twin were measured; real Blender
  5.2.2 F12 was rendered through this tree; the rendered viewport runs the
  same `render` (not driven live here); nothing ran on a real GPU driver.

## Pins changed

- `tests/test_render.py test_light_sizes_actually_work`: was
  `check('the size-derived blur is stamped on the built map',
  hasattr(sm, 'soft_extra'))`; now `check('nothing derived from the size is
  stamped on the built map, and a size edit is served the cached map',
  not hasattr(sm, 'soft_extra') and sm is sm_hard)`. The neighbouring pin
  "radius zero maps are the old picture bitwise" is kept (its label loses
  "(cache keyed on size)").

## Tests

`tests/test_r254_map_soft.py` (run as `python -m
halcyon.tests.test_r254_map_soft`): fourteen functions, a to n, one law
each. On the OLD engine (the fork point's tree with only this test file and
`core/softmap.py` dropped in) the neutrality test passes 20 of 20 and every
other function fails: 80 failures. Existing modules re-run on this tree:
`test_r251_shadow` (only the 1.89.0 zip pin of the baseline list fails),
`test_r253_light_options` (all pass), and 28 functions of `test_render`
that touch lamps, shadows, tooltips and the device switch (all pass).

## Files

- `core/softmap.py` (new): the sized lookup and the constants both roads
  read.
- `gpu/softmap.py` (new): its GLSL twin.
- `core/lights.py`: `ShadowMap` / `CubeShadow` (`axes`, `d_min`; no
  `soft_extra`), `_light_shadow_signature` (no size), `_build_shadow_maps`
  (no `_soft_world` / `_stamp_soft`), the MAP branch of `visibility`.
- `gpu/shade.py`: the map part of `_shadow_meta`, the `'map'` group of
  `_light_sig`.
- `gpu/material.py`: `_shadow_function` hands a sized meta to the twin.
- `properties.py`: four tooltips.
- `tests/test_r254_map_soft.py` (new), `tests/glsl_strict.py`
  (`check_mapped_shadows`), `tests/test_render.py` (one pin).

## Follow-ups for the integrator

- **A conservative search.** A dilated tile-minimum of each map (it fits the
  unused G channel of the atlas cell, so no layout change) would let a
  fragment with no caster within reach skip the search exactly: 14 fetches
  -> 1 for most of a frame, and most of the 1.3 to 1.7 warm-frame factor. A
  min pyramid on top of it is what the thin-caster limit needs.
- **Softness on a lamp of size 0** still erodes the shadow (size-map-v01);
  routing it through the same plane-biased filter would also remove the
  step at size 0. Not done here by instruction.
- **The size as texels.** The size is baked as literals (the Area and ray
  precedent), so a size drag on the GPU road is a plan miss and a new text,
  as before. `d_min` shows the way: the atlas G channel has room.
- **The map cache is still bounded by 64 entries, not by bytes**
  (size-map-v03's second half).
