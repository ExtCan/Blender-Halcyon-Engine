# R254 bake-speed: the shadow-map road bakes less, and bakes it faster

Work-package key `bake-speed`. The integrator owns CHANGELOG.md / README.md /
the version stamps; the fragments below are in their house voice. Findings
covered: perf-depth-only-bake-05 = perf-bake-01, verify-highpoly-triangle-
bound-14, dead-lamp-shadow-bake, limit-maps-perf-01, perf-limit-bake-01,
verify-light-limit-bakes-12, hemi-cube-bake-01, screen-spot-map-bake-01,
verify-key-overreach-11, perf-viewport-fullsize-bake-06 = perf-draft-bake-01,
verify-object-move-layers-13, perf-thread-gate-09, perf-cache-hoard-07 =
perf-cache-bytes-01 = size-map-v03 (second half), perf-worker-pool-rebake-08,
wide-spot-map-01 = v-spot-wide-beam-maps, ring2-map-reach-01, clip-area-01,
note-coarse-01, stale-map-same-scene-01, perf-gpu-device-cpu-bake-02.

The package was built in two sittings. The first forked from 4e655f6 and was
cut off with its write-up unfinished; the second merged it over claude-cloud
as it stood at dd9d034 (the viewport, native, flags and map-gpu packages and
the integration commit on top), re-proved every item on the merged tree,
measured everything below again and added the device bake. Every number in
this file was measured on the merged tree against dd9d034 ("before"), same
machine, back to back -- two frozen copies of the trees, `<scratch>/wt/
bake-speed2/base/` and `new/`. The probes are in `<scratch>/wt/bake-speed2/
work/` (`t1_depth.py`, `t3_threads.py`, `t4_pool.py`, `n1_dump.py` +
`n2_cmp.py` + `n_all.sh`, `tm_table.py` + `tm_all.sh` + `tm_drive.sh` +
`tm_print.py`, `p1_draft_soft.py`, `p2_vol.py`, `g1_fake.py`, `g2_marks.py`,
`g3_host_cost.py`, `accept.sh`), the real-Blender scripts in `<scratch>/wt/
bake-speed2/bl/`, outputs in its `out/`, `logs/` and `bl/out/`.

## CHANGELOG

- **What was reported.** "Shadow map is slower than ray traced shadows."
  Per pixel it never was: a mapped lookup is a handful of texel fetches, a
  traced shadow a BVH walk. The map road lost on every frame that BAKED,
  and it baked far more than anything read, at full size, through the
  beauty rasteriser. On the 55,000-triangle test scene with two point lamps
  on 2048 maps (640x480, CPU): a cold frame TM_P2048_COLD_OLD s (the bake
  TM_P2048_COLDMAPS_OLD s of it) against TM_P2048_RAYCOLD s traced; one
  lamp nudged TM_P2048_NUDGE_OLD s against TM_P2048_RAYNUDGE s; a small
  object moved TM_P2048_OBJ_OLD s against TM_P2048_RAYOBJ s; a 160x120
  viewport draft while the lamp is dragged TM_P2048_DRAFT_OLD s against
  TM_P2048_RAYDRAFT s; with four worker processes TM_P2048_POOL_OLD s.

- **The bake is depth only** (perf-depth-only-bake-05, perf-bake-01;
  `core/lights._render_depth`, `core/raster.rasterize_depth`). A map keeps
  one number per texel. `_render_depth` allocated a full G-buffer (triangle
  ids, three barycentrics, depth, zndc, front, overdraw: 29 bytes a texel,
  122 MB for a 17 MB plane at 2048) and ran the beauty rasteriser:
  perspective barycentrics and an attribute einsum per fragment, a
  three-key sort to resolve them, only to return `gb.zndc`. The depth-only
  fill runs the SAME clipper, boxes, size classes, edge functions, wobble
  window and z expression as float32 operations on the same operands, and
  nothing else: an edge function is a row term minus a column term, so both
  factors are computed once per row and per column and one subtraction runs
  per texel; a large triangle (the floor: most of a map) resolves densely,
  strip by strip, with no gather at all; the small ones resolve with a
  per-texel minimum instead of a sort. The texel is `min(z)`, which is what
  the fills' named tie rule left in zndc (the rule only picks the triangle
  ID among equal depths). A NaN or +inf fragment never passed their strict
  `<` against an empty texel and is dropped before the minimum (the
  verifier's NaN condition, and +inf with it: the MIDPOINT sweep would
  otherwise have marked a second surface the old sweep never saw).
  One cube / one map, serial, 55k triangles (`t1_depth.py`): POINT 512
  T1_P512; POINT 2048 T1_P2048; SPOT 512 T1_S512; SPOT 2048 T1_S2048; SUN
  512 T1_U512; SUN 2048 T1_U2048; MIDPOINT POINT 512 T1_PM512. At 885,000
  triangles: POINT 512 T1B_P512, POINT 2048 T1B_P2048, SPOT 512 T1B_S512,
  SUN 512 T1B_U512 (there the cost is the per-triangle setup; see Limits).
  Bitwise in every case.

- **A per-map reject in front of the clipper** (verify-highpoly-triangle-
  bound-14; `core/raster.frustum_keep`). Every triangle of the scene went
  through `build_screen_tris` for every map -- a 48-byte gather and a clip
  classification each -- and a cube face threw most of them away only at
  the fill's box test (the clipper's own cull is a 4-SCREEN guard band).
  Now one byte per vertex and one gather per triangle drop a triangle
  whose corners are all behind the near plane, or -- if it is wholly in
  front of it -- all outside the same side of the map by two texels. A
  triangle that straddles the near plane is always kept: the clipper makes
  new corners for it and the reject must not depend on how those round
  (this is narrower than the verifier's prototype, which also dropped
  straddlers by their side bits; it costs nothing measurable and needs no
  rounding argument). A cube face of the test scene keeps 0 to 36% of the
  mesh. A face that keeps nothing, and a face the casters' bounding box
  cannot reach (`lights._box_misses`, eight corners, used for the layer of
  a moved object), return a BLANK plane: no allocation, and
  `ShadowMap` shares one read-only plane among all blank maps of a size
  and value (the texels are the ones a plane of ones gave; the atlas keys
  and plan signatures that sum them come out the same). The R87 note in
  `_render_depth` ('a cull kept ~100% of triangles and cost 15%') was
  measured on a sun and a spot map, which enclose the scene; it is
  replaced by this one.

- **A map is baked for the lamps something looks up** (dead-lamp-shadow-
  bake, limit-maps-perf-01, perf-limit-bake-01, verify-light-limit-bakes-12,
  hemi-cube-bake-01, screen-spot-map-bake-01; `core/lights.map_roles`).
  `_build_shadow_maps` walked `scene.lights` and baked every lamp whose
  Shadows menu resolved to a map. The rule now has three answers per lamp:
  - *no shadow term*: the old gate (`wants_shadow_map`), and a HEMI --
    `visibility` answers ones for one before it looks at a map and the GPU
    plan emits no function, yet ticking Hemisphere on a Sun left at the
    panel's default baked a six-face cube nothing read (the export sets
    `shadow = 'NONE'` and the panel block then overwrites it);
  - *bake*: a lamp of `LI.frame_lights` whose shadow a surface may read;
  - *defer*: a lamp that would get a map but that no surface is known to
    read -- a lamp the Light Limit dropped (12 point lamps at the default
    limit of 8: 72 faces baked, 48 read), an Ambient Only lamp
    (`is_ambient_lamp`: `frame_lights` holds none since the flags
    package), and, on a frame without a BVH, a Screen Spotlight ('no
    shadows, no cone') and a lamp that adds no light (Power 0, a black
    colour, both Only flags or neither lobe) while nothing reads a shadow
    apart from its lamp's light.
  A deferred lamp is not skipped: it holds a PROMISE (`_Deferred`), and the
  first `visibility` call that reaches for its map bakes it, once, through
  the cache, whichever thread asks. So the rule needs to know who certainly
  reads, never who might: the volume march and the ground plane light from
  a Screen Spotlight as from a spot and shadow it through `visibility`;
  both get exactly the map they always got, on demand. When the first
  sitting wrote the rule the march still lit a container from EVERY lamp,
  limit or not, and a dropped lamp's map was baked by its first lookup;
  the flags package made the march read `frame_lights`, and the same
  promise is now simply never called (12 lamps under limit 8 beside a
  volume: 48 faces, was 72). The march also no longer asks a lamp of no
  radiance for a shadow (`core/volume.march`: it adds zeros whatever the
  shadow says, and the lookup was what baked the promised map: 6 faces ->
  0 beside a volume).
  "Nothing reads a shadow apart from the light" is
  `render.shadow_read_apart_from_light`, one clause per reader in
  `light_surface` that takes `vis` without multiplying by the lamp's
  radiance: a wanted light pass (the Shadow pass's average), a Shadows
  Only material, a cel's fixed key (the lamps then only cast), the paint
  model's lit verdict, the Mega Drive class's key-lit fraction; anything
  it cannot tell from a graph answers yes. With it true every lamp of the
  frame is baked as before.
  In the surface loop a lamp that adds nothing is not ASKED for a shadow
  where nothing reads it apart (`light_surface`: no Shadows Only sum, no
  light split, not the paint, not the key lamp) -- exact, it added zeros --
  which is what keeps its promise unbaked, and removes the traced road's
  half of the same waste (a Power 0 lamp under Ray Traced traced a full
  set of shadow rays). The ground plane's lamp loop skips a lamp of no
  radiance the same way.
  Measured (`tests/test_r254_bake_speed.py` c, faces rasterised): 12 point
  lamps under limit 8: 72 -> 48; + a Hemi: 12 -> 6; + a Screen Spotlight:
  7 -> 6; + three dead lamps: 19 -> 6. Frames identical in every row.

- **A lamp's cache key holds what its own bake reads** (verify-key-
  overreach-11; `_light_shadow_signature`, `_shadow_base_signature`).
  Every lamp was keyed on position, direction and Spot Size whatever its
  kind, and every key on the raw Shadow Method, Map Size and Map Depth of
  the render. A Sun's frame is built from the scene's bounds and its
  DIRECTION: dragging a sun lamp re-rasterised its map for identical
  texels. A Point or Area lamp's cube is built from its POSITION: aiming an
  Area lamp re-baked six faces. Now: SUN -> direction; SPOT -> position,
  direction, cone (Spot Size, Square, and whether the law is GX Ring 2);
  everything else -> position; ClipSta / ClipEnd for Spot and Point; the
  RESOLVED Map Size and Map Depth (the lamp's own or the render's). The
  base key is the geometry and the caster set alone, each a full content
  digest (the caster set was a count and an index sum; the triangle list
  was not in the key at all). Rasterisations per edit, was -> now: a Sun
  translated 1 -> 0; a Point lamp rotated 6 -> 0; an Area lamp aimed 6 ->
  0; a non-spot's Spot Size 6 -> 0; a lamp's Map Depth set to the render's
  own value 6 -> 0; Shadow Maps -> Per Light with every lamp on maps all ->
  0; the render's Map Size changed under lamps that set their own all -> 0.

- **A moved object re-rasterises itself** (verify-object-move-layers-13;
  `_bake_lamps`). With an honest geometry digest every vertex change
  missed every lamp, and every map re-rasterised every caster. A CLASSIC
  texel is the minimum z over the fragments on it, and a fragment's z
  depends on its own triangle and the map's frame only, so `depth(all) =
  linearise(min(zndc(static), zndc(moved)))` bit for bit (`_linearise`
  clips z to [-1, 1]: an empty texel's 1.0 and a fragment past the far
  plane land on one float). Each cached map keeps the per-object digests
  of its last bake (one CRC per object's vertex block). On a miss with the
  same lamp, casters, topology, object index and PLAN (every matrix, near
  and far, byte for byte): the first frame bakes the layer of everything
  that did not change since -- a full bake's cost -- and from then on only
  the changed objects are rasterised, each face is one `np.minimum`, and a
  face they do not reach is its static map as it stands. Whole bakes
  remain for: a MIDPOINT map (its second surface is not a plain minimum),
  a mesh without an object index or whose objects are not each one block
  of vertices, more than a third of the casters moving, and any change of
  the scene's bounding box (it moves a Sun's frame and every far plane --
  so an object that IS the bound is never layered while it moves).

- **A viewport draft bakes draft-sized maps** (perf-viewport-fullsize-
  bake-06, perf-draft-bake-01; `preview.ViewportRenderer._render`,
  `lights._map_size`, `RenderSettings.copy`). A draft shrank the picture
  and baked full-size maps for it. The draft's settings copy now carries
  `_shadow_map_div`: 2 when the draft's pixels are twice the refine's, 4
  at four times or more; a map's side is divided by it (never under 64,
  never over its own size). The Softness is in TEXELS, so the same copy
  divides it (the verifier's correction: undivided, the same count of
  texels on a quarter-size map is a blur four times as wide; penumbra area
  of a Sun's shadow at Softness 8, refine / draft at 2 / draft at 8: 2105 /
  2202 / 6376 floor points, the shadow's mass 1635 / 1637). The resolved
  size is in each lamp's key, so draft and refine maps are separate entries
  and the refine at rest is served its own. `RenderSettings.copy` carries
  the divisor as it carries `_sample_pass`, so every road inside a draft
  frame bakes the draft's size. F12 and refines never see it. On the GPU
  road the map size is plan structure, so a draft and a refine are two
  plans; the plan cache (eight entries since map-gpu) keeps both and a drag
  that alternates them re-plans neither (the test counts 2 misses and 6
  hits over four ticks).

- **The cache is bounded by bytes** (perf-cache-hoard-07, perf-cache-
  bytes-01, size-map-v03; `_ShadowCache`). One map per lamp pose (and one
  per lamp per frame of moving geometry) was parked until 65 entries
  existed -- 6 GB of 2048 cubes -- and then everything was dropped, the
  maps of the lamps that had not moved included. Entries now leave oldest
  first while the depth held exceeds 512 MB (`SHADOW_CACHE_BYTES`) plus
  what the frame in hand uses, static layers counted, and an entry the
  frame uses never leaves.

- **The bake's threads follow texels too** (perf-thread-gate-09;
  `_run_depth_jobs`): in threads when two or more maps are due and the
  casters are 50,000 or the texels 2,000,000 (one 640 cube), where it was
  triangles alone. The pool's size is the machine's, as it was (the
  Threads dial is the shading loop's and defaults to 1; capping the bake
  by it would have made every default scene bake in line).

- **A worker pool's parent bakes once** (perf-worker-pool-rebake-08;
  `core/parallel.render_parallel`, `render.prebake_shadows`,
  `lights.adopt_shadow_maps`, `core/worker.py`). Every worker process
  baked the whole set for its own band. The parent now bakes before the
  upload, each lamp carries its map and the key it was baked for, the
  scene is pickled ONCE for all workers (it was once per worker) and sent
  to all before any is waited for; a worker parks the maps under their
  keys and its own build finds them. A set over 1 GB (the message length
  is 32 bits) is not baked by the parent and the workers bake as before.

- **A wide Spot bakes a cube; a GX Ring 2 Spot's map reaches its ring**
  (wide-spot-map-01, v-spot-wide-beam-maps, ring2-map-reach-01;
  `lights.spot_map_fov`, `_plan_map`). A Spot always baked ONE perspective
  map of `min(Spot Size x 1.15, 3.0)` rad: at 150 degrees 18 texels per
  unit tangent where a 60 degree cone has 372, and the texel-scaled normal
  offset lifted the receiver out of its own shadow. `spot_map_fov` is the
  one place the field of view is computed (the cone and its 15%, a Square
  cone's diagonal, and under GX Ring 2 the angle at which the ring's light
  ENDS: libogc's RING2 is centred on the cutoff and reaches zero at
  `cos = c - (1 - c) / sqrt 2`); up to 2.2 rad the Spot keeps its one map,
  frame for frame; above it bakes the Point lamp's cube, of which the
  faces wholly outside the cone stay blank (nothing the Spot lights looks
  through them -- as everything outside the single map always read lit).
  That is a round cone wider than 109 degrees, a Square one wider than 90,
  a GX Ring 2 one wider than about 83.

- **ClipSta / ClipEnd on an Area lamp** (clip-area-01; `_plan_map`). The
  pair is drawn for Spot and Point lamps and the tooltips say so, but an
  Area lamp's cube honoured a value left on it (a Spot switched to Area
  kept clipping, unseen). An Area lamp keeps the automatic planes.

- **The coarse-map note measures from the camera** (note-coarse-01;
  `render._shadow_coarseness_note`). It read a `position` no Camera
  carries, so the distance was the scene's from the world ORIGIN: a camera
  9 units from a scene 1.4 from the origin was told a 512 map spanned ~10
  pixels a texel (it spanned 1.5) and to raise Shadow Map Size -- which
  multiplies the bake. Console text only.

- **A frame that builds no map leaves none behind** (stale-map-same-
  scene-01; `render._build_shadows`, `lights.drop_shadow_maps`). Switching
  a Scene that had been rendered with maps to Ray Traced with Ray Shadows
  off kept the earlier frame's mapped shadow (`visibility` falls through
  to `light.shadow_map` once the traced branch is off).

## The numbers

TIMING_SECTION

## Which pictures change

- A **Spot wider than about 109 degrees** (Square: 90, GX Ring 2: 83)
  under shadow maps: its shadow, which was lost (and falsely cast
  elsewhere). Real Blender 5.2.2 F12, 150 degrees, against the traced
  frame's 3582 shadow pixels: 1553 missing and 941 spurious -> 137 missing
  and 43 spurious. Both roads (the GPU reads the same cube; simulator
  against CPU 3e-6).
- A **GX Ring 2 Spot** under shadow maps: the shadow inside the lit ring
  beyond the cutoff, which the map did not reach. Real Blender F12, a 40
  degree cone overhead: 107 of 369 shadow pixels missing -> 0.
- An **Area lamp carrying ClipSta / ClipEnd** values: its map is no longer
  clipped by them.
- A Scene object **re-rendered after a switch to Ray Traced with Ray
  Shadows off**: no stale mapped shadow.
- **Viewport drafts** (never refines, never F12) under shadow maps: maps of
  half or a quarter of the side while the view moves.
- Nothing else. `n1_dump.py` renders 71 cases on both trees -- every kind
  hard and sized, CLASSIC and MIDPOINT, Softness, Samples 1, Square,
  clips, own sizes, odd sizes, a lamp inside the geometry, a grazing spot,
  Per Light mixed, Ray Traced, a caster subset, the Light Limit, dark /
  no-lobe / negative / only-shadow lamps, a Hemi, a Screen Spotlight,
  volumes, the ground plane, the Shadow pass, the demo scene, and a
  twelve-step edit sequence through the cache (lamps moved, rotated,
  sized, the ball moved four times, back, another object, out of bounds,
  the settings) -- and dumps every CPU frame, every simulated GPU frame,
  every map texel, vp, near / far / extent / d_min, every shadow pack, and
  the text of every pass and pack key: 1280 arrays in both dumps, 1277
  identical; the three that differ are the GPU shadow packs of the dark,
  no-lobe and Screen Spotlight scenes, which no longer carry the maps
  nothing read (their simulated frames are identical). 168 arrays exist
  only at the fork point: the maps of lamps nothing looked up. 44 texts,
  38 identical; the six that differ are the pass sources and pack keys of
  those same three scenes (a shadow function fewer each). Real Blender
  F12, a 60 degree spot: bitwise the fork point's frame.

## The GPU road

Nothing under `gpu/` was edited. The plan reads the maps the CPU baked, so
it sees every change above through them: fewer lamps in the shadow pack, a
cube for a wide Spot (`_shadow_meta` already switches on the map's type),
blank faces as ordinary far-plane cells. Parity is held by the simulator
twin in the tests (a dark lamp, a Screen Spotlight, a Hemi, a 150 degree
Spot: max 3e-6) and by the dump above.

For the implementer of the GPU atlas and plan cache (see Merge notes):
`_shadow_meta` reads a frame lamp without a map as a request for traced
shadows whenever a BVH exists, and refuses the frame under a caster mask.
That is why `map_roles` defers a lamp of the frame only on a frame WITHOUT
a BVH. `LI.map_deferred(light)` is the predicate for an early
`return None, None, None` there; with it in place the `alone` condition in
`map_roles` can go and frames with a BVH save the same bakes.

## Where a device bake stands

`core/lights._render_depth(verts, tris, vp, size, cull='NONE',
midpoint=False, bounds=None)`: vertex positions, caster triangles, a
view-projection and a size in; the map's NDC depth plane out (1.0 where
nothing landed; `raster.blank_depth` when nothing landed at all), or the
MIDPOINT triple. Every bake goes through the module's name for it -- a
whole map, a static layer, the layer of the moved objects, a map baked on
its first lookup, the maps a pool is handed -- from `_run_depth_jobs`, the
one caller. A device bake (perf-gpu-device-cpu-bake-02) replaces what that
name returns for CLASSIC and leaves the rest to it; `_run_depth_jobs` is
where its serial dispatch would replace the thread pool.

## Limits, as measured

LIMITS_SECTION

## Pins changed

- `tests/test_r251_shadow.py test_c117_midpoint_per_lamp_override`: was
  `check('the base signature differs between CLASSIC and MIDPOINT',
  LI._shadow_base_signature(sc_c, st_c, None) !=
  LI._shadow_base_signature(sc_m, st_m, None))`; now `check('the resolved
  lamp signature differs between a CLASSIC and a MIDPOINT render (the
  cache cannot serve across the setting)', LI._light_shadow_signature(a,
  st_c) != LI._light_shadow_signature(a, st_m) and
  LI._shadow_base_signature(sc_c, st_c, None) ==
  LI._shadow_base_signature(sc_m, st_m, None) and
  LI._light_shadow_signature(b, st_c) == LI._light_shadow_signature(b,
  st_m))`. The render's Map Depth left the base key and entered each
  lamp's (resolved): the cache still cannot serve across the setting, and
  a lamp that overrides it no longer misses when the render's changes.

No other pin moved. `_light_shadow_signature(light)` still takes one
argument (the lamp's own unresolved values) for the tests that call it so;
`_render_depth` keeps its positional signature and its name (two test
modules call it, one patches it to count bakes); `LI._SHADOW_CACHE.clear()`
still empties everything.

## Tests

TESTS_SECTION

## Files

- `core/raster.py`: the depth-only section (`blank_depth`, `is_blank`,
  `clip_positions`, `frustum_keep`, `depth_fragments`, `rasterize_depth`).
  Nothing above it is touched: the beauty fills are the reference.
- `core/lights.py`: the map-baking section rewritten (`_render_depth`,
  `_box_misses`, `spot_map_fov`, `_map_size`, `_MapPlan`, `_plan_map`,
  `_light_shadow_signature`, `_shadow_base_signature`, `map_roles` and its
  predicates, `_Deferred`, `_ShadowCache`, `_Layer`, `_Frame`,
  `_run_depth_jobs`, `_bake_lamps`, `build_shadow_maps`,
  `_build_shadow_maps`, `adopt_shadow_maps`, `drop_shadow_maps`,
  `planned_map_bytes`); `ShadowMap.__init__` (a blank map); five lines in
  `visibility` (a deferred map is baked on its first lookup). The lookup
  arithmetic is untouched.
- `core/render.py`: `_build_shadows`, `prebake_shadows`,
  `shadow_read_apart_from_light`, the skip in `light_surface` and in the
  ground plane's lamp loop, the eye of `_shadow_coarseness_note`.
- `core/parallel.py`, `core/worker.py`: the hand-over.
- `core/settings.py`: `RenderSettings.copy` carries `_shadow_map_div`.
- `preview.py`: the draft's divisor and Softness.
- `tests/test_r254_bake_speed.py` (new), `tests/test_r251_shadow.py` (one
  pin).

## Merge notes

For the **map-gpu** implementer (gpu/shade.py, gpu/material.py, the plan
cache), nothing of yours was edited:
- `_shadow_meta`: add `if LI.map_deferred(light): return None, None, None`
  before the ray fallback (a frame lamp is deferred only when no surface
  reads its shadow on either road). Then delete the `alone` condition in
  `LI.map_roles` and `st._shadow_untraced` in `render._build_shadows`:
  frames with a BVH then skip the same bakes, and no dark lamp or Screen
  Spotlight can turn a mapped frame into a traced one or a refusal.
- Draft maps: the divisor is `settings._shadow_map_div` (2 or 4, absent
  otherwise; `RenderSettings.copy` carries it). The same draft copy has
  `shadow_softness / div`, which `_plan_sig` already reads, and the maps'
  size and content sit in `shadow_sig`, so a draft plan and a refine plan
  have different signatures today. If `shadow_sig` becomes structure
  (kind, size, midpoint) that still holds; if Softness becomes a texel,
  name the divisor in the settings part of the signature. `_PLAN_CACHE`
  clears itself past its handful of entries: a drag now alternates a
  draft plan and a refine plan per mapped configuration -- raise the cap
  or key drafts apart, or the alternation evicts itself (the verifier's
  note on perf-viewport-fullsize-bake-06).
- Stamps: map objects are shared more than before. One `ShadowMap` /
  `CubeShadow` is served to every frame whose key matches; a cube face a
  moved object does not reach is the SAME face object in successive cubes
  of a layered lamp; all blank maps of one size and value share one depth
  array (`LI._BLANK_PLANES`). A bake serial belongs on the map object
  (set it in `ShadowMap.__init__` / `CubeShadow.__init__`), never on the
  depth array.
- `shadow_sig` walks `scene.lights`: a lamp the Light Limit dropped now
  reads None there, so moving one no longer re-plans.
- The device bake: `LI._render_depth` (see Where a device bake stands).

For the **flags** implementer (volumes, the beam, `is_ambient_lamp`):
- `map_roles` picks `is_ambient_lamp` up by name from core/lights when it
  exists. If `select_lights` drops ambient lamps the clause is redundant
  (a lamp outside `frame_lights` is deferred anyway); either way an
  Ambient Only lamp stops baking the day it stops shading.
- The volume march and the beam need nothing from this package: while the
  march reads every lamp, a dropped lamp's map is baked on its first
  lookup; when it reads `frame_lights` the promise is simply never called.
  One line would save a bake there: the march asks `visibility` for a
  lamp of no radiance (`rad * vis`, rad 0), which bakes its deferred map;
  `if LI.lamp_is_dark(light): continue` before the call is exact.
- `light_surface`: the block this package adds sits directly above the
  `need` mask (`LI.lamp_adds_nothing`). `LI.lamp_is_dark`,
  `LI.lamp_has_no_lobe` are there for the other readers.

For the **viewport** implementer: `preview.Viewport._render` gained the
divisor lines inside `if draft:`, after the pixel-budget loop. A lights-
only export that reuses Light objects needs nothing: every build resets
`shadow_map` and the two private attributes it sets on a lamp
(`_shadow_defer`, `_shadow_key`); `LI.drop_shadow_maps(scene)` is the
reset for a frame that builds no map.

For the **native** implementer: an Area lamp's `shadow_clip_start` /
`shadow_clip_end` no longer reach its map (clip-area-01).

For the integrator:
- `tests/run_all.py`: add `test_r254_bake_speed`.
- core/lights.py: the whole map-baking section (from `# --- map baking` to
  `# --- light sampling`) is rewritten; a branch that edited
  `_build_shadow_maps`, `build_shadow_maps` or the two signatures there
  will conflict -- take this side and re-apply the other's intent to
  `_plan_map` / `map_roles` / `_light_shadow_signature`.
- `tests/test_r254_ray_road.py` 'mapped: the lamp's Samples moves the
  frame' (max 0.0137 under a 0.02 bar) fails at the fork point with the
  same number; it is not in baseline_fails.txt.
- Not done here, by ownership: the Shadow Method tooltip still calls maps
  'soft, fast' and the Map Size tooltip offers size as the only dial
  (properties.py). The verifier's wording: maps are re-rendered on the CPU
  whenever a lamp or an object moves while traced rays are not; on a heavy
  mesh the bake time follows the triangle count rather than the size.
