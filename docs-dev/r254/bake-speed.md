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
  on 2048 maps (640x480, CPU), before -> after, against the traced frame:
  a cold frame 4.90 -> 1.64 s (the bake 4.06 -> 0.92 s of it) against 2.37 s traced
  (2.00 s once its tree is on disk); one lamp nudged 2.45 -> 1.05 s against 1.45 s;
  a small object moved 4.59 -> 0.99 s against 2.08 s (its first step
  2.02 s); a 160x120 viewport draft while the lamp is dragged 1.88 -> 0.14 s
  against 0.10 s; with four worker processes 14.81 -> 3.48 s against 1.45 s.
  In the real viewport and on the GPU device the reporter's sentence is
  only partly answered: see Limits.

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
  One cube or one map, serial, 55,000 triangles (`t1_depth.py`), before ->
  after: POINT 512 0.26 -> 0.08 s; POINT 2048 2.38 -> 0.48 s; SPOT 512
  0.09 -> 0.05 s; SPOT 2048 1.23 -> 0.26 s; SUN 512 0.08 -> 0.05 s; SUN
  2048 0.47 -> 0.12 s; a MIDPOINT POINT 512 0.57 -> 0.17 s. At 885,000
  triangles: POINT 512 2.73 -> 1.12 s, POINT 2048 5.07 -> 1.70 s, SPOT 2048
  2.18 -> 1.45 s -- and SPOT 512 1.04 -> 1.07 s, SUN 512 0.91 -> 0.93 s:
  no gain where a small map encloses a high-poly scene (the cost there is
  the per-triangle setup both fills share; see Limits). Bitwise in every
  case.

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
  promise is now simply never called (a Sun and a point lamp beside a
  volume under Light Limit 1: 0 faces, was 6). The march also no longer
  asks a lamp of no
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
  bake-06, perf-draft-bake-01; `preview.Viewport._render`,
  `lights.draft_map_divisor`, `lights._map_size`, `RenderSettings.copy`). A
  draft shrank the picture and baked full-size maps for it. The draft's
  settings copy now carries `_shadow_map_div`: 2 when the draft's pixels
  are twice the refine's, 4 at four times or more; a map's side is divided
  by it (never under 64, never over its own size). Only where that makes
  the draft's bake cheaper: before, the refine was served the full-size
  map its draft had baked, and now it bakes its own, so on a mesh whose
  bake follows the TRIANGLE count -- a smaller map costs the same there --
  the divisor would add a bake and save nothing (the real viewport, 65,000
  triangles, a 512 cube: draft plus refine 0.57 s before, 0.63 s with the
  divisor, 0.49 s without it). `draft_map_divisor` takes it when the
  frame's maps hold at least 64 texels per scene triangle and mapped lamp
  (a 2048 cube on that mesh: the first picture after 0.39 s instead of
  1.81 s). The Softness is in TEXELS, so the same copy
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

- **On the GPU device the maps are baked by the device** (perf-gpu-device-
  cpu-bake-02; `gpu/craster.DepthBaker` / `dispatch_depth`, `core/lights.
  build_shadow_maps(baker=)`, `core/render._shadow_baker`). Under the GPU
  render device a traced shadow never left the shader, while every mapped
  lamp's map was baked by the NumPy fill on every lamp or object edit --
  the reporter's configuration. A GPU-device frame now bakes its CLASSIC
  maps with a depth-only variant of the compute rasteriser
  (`HAL_RASTER_DEPTH`): the camera raster's own pixel walk
  (`hal_raster_pixel`, the same text), ONE image holding the winner's z --
  one channel (R32F) where the build creates it, RGBA32F otherwise -- read
  back into the same `ShadowMap` the GLSL lookup and any CPU fallback of
  the frame read. The host half (the clipper both bakes share, the corner
  pack and the tile bins) runs in the bake's threads; the dispatches
  follow in line, a device being one queue. A texel whose decision sits
  inside the cross-device noise window (its centre on a triangle edge)
  comes back marked and is decided on the host with the fill's own
  statements (`replay_depth`).
  What stays with the fill: a MIDPOINT map (its second surface), a map
  baked on its first lookup (the lookup comes from a shading thread), a
  worker's band, the pool parent's prebake, the texture baker -- and every
  face the device would not bake faster (`device_pays`: under 512 x 512
  texels, or under 16 texels per clipped triangle; both bakes pay the same
  clipper and about half a microsecond per clipped triangle after it, and
  what the device removes is the fill's 40 to 80 ns per TEXEL: at 885,000
  triangles the kernel's host side is the whole CPU bake). That routing is
  silent. A refusal is named on the console and the fill bakes the face
  from the clipper output the device already had: a driver that compiles
  the kernel in neither image format; a dispatch that raises (that format
  is retired for the session, the next is tried); a face with more than a
  tenth of its covered texels marked; and THE TRIAL -- the first device
  bakes of a session are each compared with the CPU bake of the same
  face, texel for texel and on the clock. A device whose map differs
  (more than 1e-4 of the texels beyond 1e-5 in NDC z; a fused multiply-add
  moves a z by 1e-7) or that takes more than half the fill's time over two
  timed maps bakes nothing more that session and says so once. A device's
  maps are keyed apart from the fill's in the cache (on a driver the two
  agree to an ulp of z, not bit for bit; one ulp is up to 1.3e-4 world
  units at the far end of a spot's range, the map's own resolution
  there). The CPU device and a GPU device with GPU Rasteriser off are
  untouched. **No real GPU was available: what it gains on one is not
  measured** (see Limits).

## The numbers

The machine has four cores and was shared with other jobs throughout (load 3
to 13). Every figure is the fastest of several frames, the two trees back
to back per configuration. The tables of POINT 512, POINT 2048 and SPOT 512
hold two passes merged row by row to the fastest frame seen (each header
says how many); the others one -- a second pass of them was abandoned when
the load passed 13. Rows that bake nothing on either tree (warm, size
dragged: the same shading code) still differ by up to a factor of two
between the trees in a single pass; that is the machine. The 'shadow maps'
stage in brackets is the steadier number, and the ratios mean more than
the second decimals.

### Frames, CPU road, 640x480, two lamps of a kind, size 0 (`tm_table.py`)

The scene of `shadow_perf_probe.py` (a floor and six 9,000-triangle
spheres) plus a 1,000-triangle ball: 56,322 triangles. Seconds per frame,
the 'shadow maps' stage in brackets. *cold*: every cache empty; *lamp
nudged*: one lamp moves; *size dragged*: both lamps sized, one size steps
(traced: soft rays); *object moved*: the ball moves, first step and later
steps; *draft drag*: a 160x120 viewport draft per lamp nudge, as preview.py
makes it (after: maps / 4 and Softness / 4); *pool*: four worker processes,
a cold frame. "before" is claude-cloud at dd9d034, "after" this package at
e9c24fa (the CPU bake did not change after that commit; the viewport's
divisor rule did, which this probe does not go through: it sets the
divisor itself), "traced" the same frames under Ray Traced (the fastest of
either tree: the traced road is the same code).

**POINT, 512 maps** (56322 triangles; 2 runs, load at their ends from 5.7)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 1.40 (0.63) | 1.03 (0.21) | 1.4 | 2.25 | 0.46 |
| cold, tree on disk | | | | 1.85 | |
| warm | 0.60 | 0.56 | 1.1 | 1.65 | 0.34 |
| lamp nudged | 0.91 (0.35) | 0.68 (0.09) | 1.3 | 1.61 | 0.42 |
| size dragged | 0.82 | 0.93 | 0.9 | 5.39 | 0.17 |
| object moved, first step | 1.42 (0.62) | 1.04 (0.23) | 1.4 | 2.28 | 0.46 |
| object moved, next steps | 1.37 (0.62) | 0.70 (0.03) | 2.0 | 2.38 | 0.29 |
| draft drag | 0.32 (0.28) | 0.11 (0.06) | 2.8 | 0.09 | 1.23 |
| pool, cold | 2.75 | 0.65 | 4.3 | 1.20 | 0.54 |

**POINT, 2048 maps** (56322 triangles; 2 runs, load at their ends from 5.4)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 4.90 (4.06) | 1.64 (0.92) | 3.0 | 2.37 | 0.69 |
| cold, tree on disk | | | | 2.00 | |
| warm | 0.55 | 0.62 | 0.9 | 1.55 | 0.40 |
| lamp nudged | 2.45 (1.88) | 1.05 (0.55) | 2.3 | 1.45 | 0.72 |
| size dragged | 0.92 | 1.16 | 0.8 | 5.28 | 0.22 |
| object moved, first step | 4.46 (3.77) | 2.02 (1.29) | 2.2 | 2.18 | 0.93 |
| object moved, next steps | 4.59 (3.94) | 0.99 (0.17) | 4.6 | 2.08 | 0.48 |
| draft drag | 1.88 (1.83) | 0.14 (0.10) | 13.5 | 0.10 | 1.46 |
| pool, cold | 14.81 | 3.48 | 4.3 | 1.45 | 2.40 |

**SPOT, 512 maps** (56322 triangles; 2 runs, load at their ends from 6.5)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 1.55 (0.39) | 1.49 (0.15) | 1.0 | 2.60 | 0.57 |
| cold, tree on disk | | | | 2.15 | |
| warm | 1.09 | 1.08 | 1.0 | 1.78 | 0.61 |
| lamp nudged | 1.19 (0.11) | 1.07 (0.07) | 1.1 | 1.81 | 0.59 |
| size dragged | 0.81 | 1.39 | 0.6 | 5.54 | 0.25 |
| object moved, first step | 1.62 (0.30) | 1.44 (0.16) | 1.1 | 2.54 | 0.57 |
| object moved, next steps | 0.96 (0.17) | 1.30 (0.01) | 0.7 | 2.48 | 0.52 |
| draft drag | 0.17 (0.13) | 0.10 (0.06) | 1.7 | 0.09 | 1.13 |
| pool, cold | 1.30 | 1.31 | 1.0 | 2.32 | 0.57 |

**SPOT, 2048 maps** (56322 triangles; 1 runs, load at their ends from 5.3)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 5.66 (4.14) | 1.29 (0.52) | 4.4 | 2.29 | 0.56 |
| cold, tree on disk | | | | 1.88 | |
| warm | 1.32 | 0.49 | 2.7 | 1.62 | 0.30 |
| lamp nudged | 2.88 (1.44) | 0.77 (0.27) | 3.7 | 1.69 | 0.46 |
| size dragged | 1.75 | 0.77 | 2.3 | 5.53 | 0.14 |
| object moved, first step | 5.81 (4.12) | 1.30 (0.55) | 4.5 | 2.35 | 0.55 |
| object moved, next steps | 4.92 (3.65) | 1.17 (0.11) | 4.2 | 2.33 | 0.50 |
| draft drag | 1.12 (1.08) | 0.09 (0.06) | 12.2 | 0.11 | 0.84 |
| pool, cold | 7.59 | 1.30 | 5.8 | 1.49 | 0.87 |

**SUN, 512 maps** (56322 triangles; 1 runs, load at their ends from 4.4)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 1.32 (0.15) | 0.82 (0.08) | 1.6 | 1.96 | 0.42 |
| cold, tree on disk | | | | 1.44 | |
| warm | 1.04 | 0.44 | 2.4 | 1.19 | 0.37 |
| lamp nudged | 1.04 (0.07) | 0.46 (0.04) | 2.3 | 1.15 | 0.40 |
| size dragged | 1.24 | 0.69 | 1.8 | 3.92 | 0.17 |
| object moved, first step | 1.35 (0.16) | 0.88 (0.08) | 1.5 | 2.04 | 0.43 |
| object moved, next steps | 1.44 (0.15) | 0.74 (0.01) | 2.0 | 1.73 | 0.42 |
| draft drag | 0.09 (0.07) | 0.07 (0.04) | 1.3 | 0.07 | 1.00 |
| pool, cold | 1.82 | 0.53 | 3.4 | 0.93 | 0.58 |

**SUN, 2048 maps** (56322 triangles; 1 runs, load at their ends from 5.9)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 1.53 (0.66) | 1.59 (0.34) | 1.0 | 1.99 | 0.80 |
| cold, tree on disk | | | | 1.61 | |
| warm | 0.46 | 1.26 | 0.4 | 1.39 | 0.91 |
| lamp nudged | 1.06 (0.51) | 1.25 (0.18) | 0.8 | 1.35 | 0.93 |
| size dragged | 0.60 | 0.99 | 0.6 | 4.09 | 0.24 |
| object moved, first step | 1.17 (0.58) | 1.89 (0.52) | 0.6 | 1.68 | 1.12 |
| object moved, next steps | 1.16 (0.55) | 1.58 (0.23) | 0.7 | 1.81 | 0.87 |
| draft drag | 0.55 (0.51) | 0.08 (0.05) | 6.6 | 0.07 | 1.17 |
| pool, cold | 3.00 | 1.56 | 1.9 | 1.45 | 1.07 |

### The verifier's 885,000-triangle scene, 512 maps (`tm_table.py KIND 512 384`)

**POINT, 512 maps** (885762 triangles; 1 runs, load at their ends from 5.8)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 12.83 (9.87) | 4.20 (2.43) | 3.1 | 11.10 | 0.38 |
| cold, tree on disk | | | | 3.77 | |
| warm | 1.14 | 0.77 (0.03) | 1.5 | 2.61 | 0.30 |
| lamp nudged | 5.70 (4.97) | 2.82 (2.06) | 2.0 | 2.54 | 1.11 |
| object moved, first step | 9.23 (6.97) | 5.72 (4.01) | 1.6 | 11.32 | 0.51 |
| object moved, next steps | 8.52 (6.42) | 2.19 (0.45) | 3.9 | 11.09 | 0.20 |

**SPOT, 512 maps** (885762 triangles; 1 runs, load at their ends from 4.6)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 4.83 (2.50) | 5.93 (3.05) | 0.8 | 13.16 | 0.45 |
| cold, tree on disk | | | | 5.48 | |
| warm | 0.99 | 1.79 (0.04) | 0.6 | 2.25 | 0.80 |
| lamp nudged | 2.30 (1.22) | 2.12 (1.20) | 1.1 | 2.09 | 1.01 |
| object moved, first step | 6.01 (2.88) | 4.66 (2.35) | 1.3 | 13.30 | 0.35 |
| object moved, next steps | 5.52 (2.80) | 2.59 (0.31) | 2.1 | 10.40 | 0.25 |

**SUN, 512 maps** (885762 triangles; 1 runs, load at their ends from 3.4)

| frame | maps, before | maps, after | x | traced | after / traced |
|---|---|---|---|---|---|
| cold | 2.65 (0.97) | 2.54 (0.95) | 1.0 | 10.45 | 0.24 |
| cold, tree on disk | | | | 2.75 | |
| warm | 0.61 | 0.60 (0.02) | 1.0 | 1.30 | 0.46 |
| lamp nudged | 1.40 (0.87) | 1.31 (0.73) | 1.1 | 1.50 | 0.87 |
| object moved, first step | 2.98 (1.13) | 3.06 (1.15) | 1.0 | 12.18 | 0.25 |
| object moved, next steps | 2.84 (1.16) | 1.84 (0.18) | 1.5 | 10.25 | 0.18 |

### The bake alone, one cube or one map, serial (`t1_depth.py`)

The G-buffer road against the depth-only fill, the same faces, bitwise in
every row:

| triangles | lamp, map | before | after | x |
|---|---|---|---|---|
| 55,298 | POINT 512 | 0.263 s | 0.079 s | 3.3 |
| | POINT 512, MIDPOINT | 0.570 s | 0.166 s | 3.4 |
| | POINT 2048 | 2.380 s | 0.477 s | 5.0 |
| | SPOT 512 | 0.090 s | 0.050 s | 1.8 |
| | SPOT 512, MIDPOINT | 0.208 s | 0.109 s | 1.9 |
| | SPOT 2048 | 1.229 s | 0.255 s | 4.8 |
| | SUN 512 | 0.076 s | 0.050 s | 1.5 |
| | SUN 512, MIDPOINT | 0.143 s | 0.085 s | 1.7 |
| | SUN 2048 | 0.473 s | 0.116 s | 4.1 |
| 884,738 | POINT 512 | 2.725 s | 1.122 s | 2.4 |
| | POINT 2048 | 5.071 s | 1.699 s | 3.0 |
| | SPOT 512 | 1.044 s | 1.071 s | 1.0 |
| | SPOT 2048 | 2.180 s | 1.445 s | 1.5 |
| | SUN 512 | 0.913 s | 0.933 s | 1.0 |
| | SUN 2048 | 1.712 s | 1.157 s | 1.5 |

Threads (`t3_threads.py`, two lamps, 2048, the machine at load 3.4): POINT,
twelve faces, in line 1.28 s, three threads 0.84 s; SUN, two maps, 0.32 s
-> 0.22 s.

### Faces rasterised (`tests/test_r254_bake_speed.py`, exact counts)

| case | before | after |
|---|---|---|
| 12 point lamps under Light Limit 8 | 72 | 48 |
| a point lamp and a Hemi left on Shadow Map | 12 | 6 |
| a point lamp and a Screen Spotlight | 7 | 6 |
| a point lamp and three lamps that add no light | 19 | 6 |
| a Power 0 lamp beside a volume container | 6 | 0 |
| a lamp the Light Limit dropped beside a volume | 6 | 0 |
| a Sun translated; a Point lamp rotated; an Area lamp aimed | 1; 6; 6 | 0 |
| Shadow Maps -> Per Light, every lamp on maps | all | 0 |
| a moved ball, from its second step on (casters rasterised) | all | the ball's |

### The worker pool, where a cold frame goes (`t4_pool.py`, two point lamps, 2048, four workers)

| | before | after |
|---|---|---|
| the parent's bake | - | 1.02 s |
| one pickle of the scene | 3 MB | 154 MB, 0.10 s |
| upload, four workers | 0.18 s | 1.06 s |
| the bands | 9.29 s (each worker baked both cubes) | 0.48 s |
| the frame | 9.48 s | 2.66 s |

### What a device bake leaves on the host (`g3_host_cost.py`, serial, the machine at load 6)

`front` is the clipper both bakes run; `fill` what the CPU bake adds to it;
`pack` what the device bake adds on the host. Upload, dispatch and read are
the driver's and are not in the table.

| triangles | lamp, map | CPU bake = front + fill | device host = front + pack | uploaded, read back |
|---|---|---|---|---|
| 55,298 | POINT 512 | 129 = 29 + 100 ms | 29 + 26 ms | 5.6 MB, 3 MB |
| | POINT 2048 | 554 = 27 + 527 ms | 27 + 28 ms | 6.8 MB, 48 MB |
| | SPOT 2048 | 355 = 21 + 334 ms | 21 + 23 ms | 5.9 MB, 16 MB |
| | SUN 2048 | 160 = 14 + 146 ms | 14 + 22 ms | 5.7 MB, 16 MB |
| 884,738 | POINT 512 | 2893 = 2337 + 556 ms | 2337 + 291 ms | 87 MB, 3 MB |
| | SPOT 512 | 1038 = 529 + 509 ms | 529 + 530 ms | 85 MB, 1 MB |
| | SUN 512 | 1135 = 646 + 489 ms | 646 + 455 ms | 85 MB, 1 MB |
| | POINT 2048 | 2937 = 1522 + 1415 ms | 1522 + 274 ms | 88 MB, 48 MB |

At 55,000 triangles and 2048 the fill is 91 to 95% of the CPU bake and it
is the part the device replaces; at 885,000 triangles and 512 the device's
own host work equals the fill it would replace, before a byte is uploaded.
That is what `device_pays` encodes.

### Real Blender 5.2.2

F12 through the add-on, background mode, CPU device (`bl/run_cold.sh`,
`bl/spec_cold.json`: the harness scene, two point lamps and a sun on 2048
maps, 640x480). The engine's own line:

| frame | before | after |
|---|---|---|
| cold | `4.1s -- top stages: shadow maps 3.3s, shade 0.7s` | `1.5s -- top stages: shadow maps 0.8s, shade 0.7s` |
| the same frame again | 0.7 s | 0.9 s |
| Ray Traced | `1.5s -- top stages: shade 1.5s` | 1.5 s |

The four frames (512 warm, 2048 cold, 2048 again, traced) are bitwise equal
between the two trees.

The rendered viewport, CPU device, a window under xvfb (`bl/run_viewport.sh`,
the hal-shadow verifier's script: a floor, a 65,000-triangle sphere, a box;
the lamp is nudged six times and the viewport worker's `render` calls are
timed; drafts are 131x86, refines 262x173). Seconds per draft / per refine
(the bake's share in brackets):

| lamp, map | Shadow Maps, before | Shadow Maps, after | Ray Traced |
|---|---|---|---|
| POINT 512 | 0.39 (0.35) / 0.17 | 0.30 (0.26) / 0.19 | 0.07 to 0.13 / 0.19 to 0.22 |
| POINT 2048 | 1.81 (1.77) / 0.16 | 0.39 (0.35) / 0.71 (0.63) | 0.11 to 0.16 / 0.17 to 0.27 |
| SPOT 2048 | 0.91 (0.88) / 0.12 | 0.29 (0.24) / 0.40 (0.33) | 0.10 to 0.18 / 0.19 to 0.31 |

Before, the draft baked the full-size map and the refine was served it.
After, a draft of a texel-bound frame bakes half-size maps (POINT 2048: the
first picture after 0.39 s instead of 1.81 s) and the refine bakes its own;
at 512 on this mesh the bake follows the triangles, the draft keeps the
full size and the refine is served it as before (`draft_map_divisor`; with
the divisor taken there the pair cost 0.63 s, more than before). The moved
box in the same script: the viewport frame after the move is the frame of a
session with every cache cleared, 0 pixels differ, on both trees.

The GPU device with a window (`bl/run_gpu.sh`, `bl/bl_gpu_bake.py`; xvfb,
Mesa llvmpipe, `--gpu-backend opengl`: 'OPENGL on llvmpipe (LLVM 20.1.2,
256 bits)'), F12 at 160x120, a point lamp and a spot, every face the device
bakes held against the CPU bake of the same triangles:

| run | what the log says |
|---|---|
| 512 and 2048, the trial's clock off | 22 faces dispatched (two cold frames and two lamp nudges): **0 texels differ** from the CPU bake on every face; 158 / 34 / 615 / 140 texels decided on the host; the image format settled on R32F; CPU-device frame against GPU-device frame 0.0041 / 0.0039, 0 pixels above 6e-3 (one step of the 8-bit output stage); no kernel uniform stripped |
| 1024, as shipped | five faces baked and compared (0 texels differ), then `[Halcyon GPU] shadow bake on the CPU: the device bakes a map no faster than the CPU fills it (171 ms against 58 ms over 2 maps)`; the next frame `[Halcyon GPU] shadow maps bake on the CPU this session: ...` and the nudged lamp's cube is the fill's (F12 0.21 s; with the software device baking it, 2.44 s at 2048) |

llvmpipe runs the kernel on the CPU, so its dispatch times say nothing
about a GPU (2048: 1.03 s for six faces the fill bakes in 0.39 s) -- which
is exactly the case the trial's clock exists for.

## Which pictures change

- A **Spot wider than about 109 degrees** (Square: 90, GX Ring 2: 83)
  under shadow maps: its shadow, which was lost (and falsely cast
  elsewhere). Real Blender 5.2.2 F12 (`bl/run_shape.sh`), 150 degrees,
  against the traced frame's 3582 shadow pixels: 1553 missing (43%) and 941
  spurious -> 137 missing (3.8%) and 43 spurious. Both roads (the GPU reads
  the same cube; simulator against CPU 3e-6). Sheet: `<scratch>/wt/
  bake-speed2/bl/out/shape_sheet.png` (rows: 150 degrees, GX Ring 2;
  columns: before, after, traced).
- A **GX Ring 2 Spot** under shadow maps: the shadow inside the lit ring
  beyond the cutoff, which the map did not reach. Real Blender F12, a 40
  degree cone overhead: 107 of 369 shadow pixels missing (29%) -> 0 (30
  spurious, 9 before: the map road's soft edge against the traced hard
  one).
- An **Area lamp carrying ClipSta / ClipEnd** values: its map is no longer
  clipped by them.
- A Scene object **re-rendered after a switch to Ray Traced with Ray
  Shadows off**: no stale mapped shadow.
- **Viewport drafts** (never refines, never F12) under shadow maps, where
  the frame's maps hold 64 texels per triangle and lamp or more: maps of
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
  the text of every pass, pack key and pack stamp. Against claude-cloud at
  dd9d034, on the final tree (the device bake in): 1244 arrays in both
  dumps, 1241 identical; the three that differ are the GPU shadow packs of
  the dark, no-lobe and Screen Spotlight scenes, which no longer carry the
  maps nothing read (their simulated frames are identical). 204 arrays
  exist only before: the maps of lamps nothing looked up. 66 texts, 57
  identical; the nine that differ are the pass sources, pack keys and pack
  stamps of those same three scenes (a shadow function fewer each). Real
  Blender F12, a 60 degree spot and the four frames of the cold run:
  bitwise the frames of dd9d034.

## The GPU road

- The plan (`gpu/shade.py`, `gpu/material.py`) is not edited. It reads the
  maps the bake left on the lamps, so it sees every change above through
  them: fewer lamps in the shadow pack, a cube for a wide Spot, blank
  faces as ordinary far-plane cells, a draft's smaller maps as another
  plan. Parity is held by the simulator twin in the tests (a dark lamp, a
  Screen Spotlight, a Hemi, a 150 degree Spot: 1e-6 to 3e-6) and by the
  dump above.
- `gpu/craster.py` gains the depth variant and its host side (the section
  'R254: the shadow-map bake'): `depth_compute_source` /
  `depth_fragment_source` (the kernel core cut before its resolve, the
  same core text the camera raster compiles), `depth_uniforms`,
  `simulate_depth` (the twin, over the packed arrays a dispatch uploads),
  `replay_depth`, `depth_plane`, `dispatch_depth`, `bake_depth`,
  `DepthBaker`, `depth_baker`, `device_pays`. `compute_variant_name` /
  `compute_source` take `depth=True`. The strict GLSL gate compiles the
  variant in both image formats (`tests/glsl_strict.depth_bake_unit`;
  compute raster 10 of 10, 'every shader compiles under strict GLSL 4.30').
- `gpu/device.py`: `compile_compute` and `dispatch_compute` take
  `image_format` (RGBA32F unless named: every existing caller is
  unchanged); a one-channel image reads back (H, W); `reset()` forgets
  the bake's verdict.
- `tests/fakedevice.py`: the compute door, on request
  (`installed(compute=True, compute_formats=...)`): `compile_compute`
  under the driver's rules (the format must be one the device creates,
  every declared uniform in the CreateInfo, every CreateInfo name READ by
  the text, every image stored to) and `dispatch_compute` for the depth
  kernel, run through the GLSL front-end over the arrays the dispatch
  uploaded and the push constants it set. Without `compute=True` the fake
  is what it was: the real doors refuse for want of a driver, as they
  always did under it, and `depth_baker()` then says once per process that
  the maps bake on the CPU.

## Limits, as measured

Where the map road is still slower than the traced road, and why:

- **While a lamp is dragged in the rendered viewport** (CPU device, the
  table above): per nudge, draft plus refine, 0.49 s against 0.26 s traced
  at a 512 cube, 1.10 s against 0.43 s at a 2048 cube, 0.69 s against
  0.49 s at a 2048 spot (before: 0.57, 1.97, 1.03 s). A moved lamp is a
  new map whatever is cached, and on that 65,000-triangle mesh a cube
  costs a quarter of a second at any size (the clipper and the per-
  triangle fill); a traced viewport frame of 45,000 pixels costs 0.19 s.
  This is the reporter's sentence and on the CPU device it still holds
  for this gesture; the gap went from 2.1x-4.6x to 1.4x-2.6x.
- **A 160x120 draft** in the headless table: POINT 512 0.11 s against 0.09 s traced (before 0.32 s); POINT 2048 0.14 s against 0.10 s traced (before 1.88 s); SPOT 512 0.10 s against 0.09 s traced (before 0.17 s); SPOT 2048 0.09 s against 0.11 s traced (before 1.12 s); SUN 512 0.07 s against 0.07 s traced (before 0.09 s); SUN 2048 0.08 s against 0.07 s traced (before 0.55 s). The draft's
  maps are a quarter of the side and still a bake per nudge; a traced
  draft of 19,200 pixels is nearly free.
- **The first frame an object moves** costs about two later ones, and at
  2048 as much as the traced frame: POINT 2.02 s against 2.18 s traced (before 4.46 s); SPOT 1.30 s against 2.35 s traced (before 5.81 s); SUN 1.89 s against 1.68 s traced (before 1.17 s). It
  bakes the layer of everything that did not move (a full bake's cost)
  and the moved layer; from the second step on the object alone is
  rasterised.
- **The worker pool, a cold frame at 2048**: POINT 3.48 s against 1.45 s traced (before 14.81 s; in-process 1.64 s); SPOT 1.30 s against 1.49 s traced (before 7.59 s; in-process 1.29 s); SUN 1.56 s against 1.45 s traced (before 3.00 s; in-process 1.59 s). The parent's
  bake (1.0 s) and 154 MB of maps crossing four pipes (1.1 s) are still
  more than a traced pool frame; in-process the same cold frame is faster
  than both. The maps would have to be shared, not copied (a mapped file
  or shared memory, with the lifetime rules that needs on each platform):
  not done.
- **A high-poly scene on a small map** (885,000 triangles, 512): a spot
  or sun map bakes no faster than before (one map, serial: 1.07 s against
  1.04 s, 0.93 against 0.91): the map encloses the scene, nothing is
  rejected, and the time is the clipper's and the fill's per-triangle
  setup, which the depth-only fill shares with the beauty fill. In the
  frame table's single pass the two-spot cold frame even reads slower
  (4.83 -> 5.93 s; the machine was at load 5, and the bake alone says
  equal). A cube gains 2.4x there (the reject). A nudged lamp at that
  size is a full bake and costs as much as the traced frame or more
  (POINT 2.82 s against 2.54 s, SPOT 2.12 against 2.09, SUN 1.31 against
  1.50); a moved object is 4 to 5 times cheaper than traced from its
  second step on, because the traced road rebuilds its BVH (10 to 11 s)
  for every vertex edit. The cold traced frame builds the tree too; the
  'cold, tree on disk' row is that frame once the tree is cached.
- **The SUN 2048 table** was taken once, at load 6: its warm row -- no
  bake on either tree, the same shading code -- reads 0.46 s against
  1.26 s, and every row under it carries that offset. Its 'shadow maps'
  stage is the usable column there (cold 0.66 -> 0.34 s, a nudge 0.51 ->
  0.18 s, a moved object 0.55 -> 0.23 s).

What is not measured, and what was not done:

- **The device bake on a GPU.** No GPU was available. What is proven: the
  kernel's texels are the fill's bit for bit in the simulator and, on
  llvmpipe, in real Blender (22 faces, 0 texels differ); every refusal
  falls back by name; the trial switches a device off that is not faster.
  What is not: upload, dispatch and read times on a driver, and therefore
  the gain. From the host numbers the fill is 0.5 s of a 0.55 s cube at
  2048 and 55,000 triangles and the device's own host work 0.03 s, so the
  bake should shrink to its dispatch and a 48 MB read; `device_pays` and
  the trial's 0.5 are set from host measurements and are the first
  numbers to revisit with the `[Halcyon GPU] shadow bake: ...` line of a
  real session. On a driver the device's z differs from the fill's by
  ulps (fused multiply-add); the marked texels are replayed, the rest are
  used as read, and the two devices' maps are cached apart.
- **The device bakes in line.** The dispatches cross to the main thread
  one at a time; the host halves are threaded. A frame of many small maps
  pays a crossing per face.
- **A frame with a BVH keeps the maps of its dark lamps and Screen
  Spotlights.** `gpu/shade._shadow_meta` reads a frame lamp without a map
  as a request for traced shadows when a BVH exists (and refuses the
  frame under a caster mask), so `map_roles` defers a lamp OF THE FRAME
  only on a frame without one. `LI.map_deferred(light)` is the predicate
  for an early `return None, None, None` there; with it the `alone`
  condition in `map_roles` and `st._shadow_untraced` can go. Not done:
  that function is being edited for the mapped soft shadow's search.
- **A map promised to a later lookup is the fill's**, on any device (the
  lookup comes from a shading thread; the device is the frame's).
- **The large-triangle fill tests every texel of a triangle's box.** The
  floor's two triangles each sweep the whole face of a map; a span fill
  would halve that, and has to stay bitwise at the edges. Not attempted.
- **Roads.** CPU F12 and the GPU simulator twin headless; real Blender
  5.2.2 F12 on the CPU device (both trees) and on the GPU device under
  llvmpipe; the rendered viewport on the CPU device, driven by script
  under xvfb (both trees). The rendered viewport on the GPU device was
  not driven, and no real GPU driver ran.

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
- `tests/test_r254_world_light.py` (the panel test): was `rsrc.count(
  '_WL.sync(scene, lambda: prepare_textures(scene, st))') == 2 and
  rsrc.index('_WL.sync(scene') < rsrc.index('    collect_exclusive_lights(
  scene)\n    mesh = ')`; now the count is `== 3`, `'def prebake_shadows('`
  must sit above `def render(` with exactly one sync before it, and the
  order is read from `render()`'s own first line on (`rsrc.index(
  '_WL.sync(scene', rnd) < ...`). The pool parent's `prebake_shadows`
  syncs the world's light the same way before it bakes for it; counted as
  it was, the old order check would have been satisfied by that earlier
  line whatever `render()` did.

No other pin moved. `_light_shadow_signature(light)` still takes one
argument (the lamp's own unresolved values) for the tests that call it so;
`_render_depth` keeps its positional signature and its name (two test
modules call it, one patches it to count bakes); `LI._SHADOW_CACHE.clear()`
still empties everything.

## Tests

`tests/test_r254_bake_speed.py` (`python -m halcyon.tests.
test_r254_bake_speed`): twelve functions, a to l, 134 checks, all
passing. On claude-cloud at dd9d034 with only this file and
`tests/fakedevice.py` dropped in: 26 failures -- ten functions stop on a
missing API (a, b, c, d, e, f, g, i, j, l) and the other two fail their
checks (h: the thread gate; k: the note and the stale map) -- every
function fails.

- a: the depth-only bake against the G-buffer road, zndc and built map
  (every kind, CLASSIC and MIDPOINT, a lamp inside the geometry, grazing,
  odd sizes, a caster subset, clip planes, degenerate and coincident
  triangles, NaN and infinite depths); no G-buffer is allocated.
- b: the reject drops only what cannot land (straddlers at three scales,
  the box test).
- c: who gets a map (the Light Limit, a Hemi, a Screen Spotlight, dark and
  no-lobe lamps; the Shadow pass, a Shadows Only material and the paint
  read a dark lamp's shadow; a volume and the ground plane under a Screen
  Spotlight bake on demand; a dark lamp beside a volume bakes nothing; a
  promise through pickle and deep copy; both roads; a frame with a BVH;
  the traced road).
- d: sixteen edits against the faces they rasterise, each frame a cold
  frame's.
- e: draft sizes, separate entries, the settings copy, the Softness (the
  penumbra's area), the two GPU plans of a drag, `draft_map_divisor`.
- f: the static layer, bitwise the full bake over a sequence of moves.
- g: the byte bound and the frame's own entries.
- h: the thread gate; threaded faces bitwise in-line ones.
- i: the pool's parent bakes once, a worker adopts; `planned_map_bytes`.
- j: `spot_map_fov`, the cube above 2.2 rad, the skipped face, the Area
  lamp's clips; both roads.
- k: the coarse-map note; no map left behind.
- l: the device bake -- the kernel's text; `device_pays` and the silent
  routing; a GPU-device frame through the fake device with the compute
  door (one dispatch per face, R32F, maps bitwise the CPU bake's, keys
  apart, served again, a nudged lamp, a moved object's layers, MIDPOINT,
  the CPU device beside it); the CPU device, GPU Rasteriser off and a
  worker's band compile nothing; the refusals (no R32F, neither format, a
  dispatch that raises, a first map that differs, a one-ulp driver that is
  believed, the trial's clock); marked texels replayed bitwise; the
  referral budget; a blank face; `replay_depth` against the fill at every
  texel of three cube faces.

Re-run on the final tree (`work/accept.sh`, logs in `logs/accept_final/`):
@@ACCEPT_PENDING@@

## Files

- `core/raster.py`: the depth-only section (`blank_depth`, `is_blank`,
  `clip_positions`, `frustum_keep`, `depth_fragments`,
  `depth_screen_tris`, `rasterize_depth`). Nothing above it is touched:
  the beauty fills are the reference.
- `core/lights.py`: the map-baking section rewritten (`DEPTH_BAKER`,
  `_render_depth`, `_box_misses`, `spot_map_fov`, `_map_size`,
  `draft_map_divisor`, `_MapPlan`, `_plan_map`,
  `_light_shadow_signature`, `_shadow_base_signature`, `map_roles` and
  its predicates, `_Deferred`, `_ShadowCache`, `_Layer`, `_Frame`,
  `_run_depth_jobs`, `_bake_lamps`, `build_shadow_maps`,
  `_build_shadow_maps`, `adopt_shadow_maps`, `drop_shadow_maps`,
  `planned_map_bytes`); `ShadowMap.__init__` (a blank map); five lines in
  `visibility` (a deferred map is baked on its first lookup). The lookup
  arithmetic is untouched.
- `core/render.py`: `_shadow_baker`, `_build_shadows`, `prebake_shadows`,
  `shadow_read_apart_from_light`, the skip in `light_surface` and in the
  ground plane's lamp loop, the eye of `_shadow_coarseness_note`.
- `core/volume.py`: the march does not ask a lamp of no radiance for a
  shadow (one `continue`).
- `core/parallel.py`, `core/worker.py`: the hand-over.
- `core/settings.py`: `RenderSettings.copy` carries `_shadow_map_div`.
- `preview.py`: the draft's divisor and Softness.
- `gpu/craster.py`, `gpu/device.py`: the device bake (see The GPU road).
- `tests/test_r254_bake_speed.py` (new), `tests/fakedevice.py` (the
  compute door), `tests/glsl_strict.py` (the depth variant),
  `tests/test_r251_shadow.py` and `tests/test_r254_world_light.py` (one
  pin each).

## Merge notes

For the engineer of the **mapped soft shadow's search** (core/softmap.py,
gpu/softmap.py, the pack's layout in gpu/shade.py), none of which is
edited here:
- No bake rewrites `depth` in place. Every bake builds NEW `ShadowMap`
  objects and nothing writes to a cached one, so there is no derived slot
  to reset anywhere in this package: `digest`, `d_min` and a pyramid slot
  beside them are filled once per object and stay valid for its life.
- Map objects are SHARED more than before, read-only: one `ShadowMap` /
  `CubeShadow` is served to every frame whose key matches; a cube face a
  moved object does not reach is the SAME face object in successive cubes
  of a layered lamp; all blank maps of one size and value share one
  read-only depth array (`LI._BLANK_PLANES`, `flags.writeable` False).
  Anything cached per map belongs on the map object, never on the depth
  array; a pyramid built from a blank map's plane must not write to it.
- `ShadowMap.__slots__` is as map-gpu left it; a new slot needs no change
  here (`ShadowMap.__init__` sets the slots it always set; the lazy ones
  are read with `getattr(..., None)`).
- A draft's maps are smaller AND its Softness is divided with them
  (`settings._shadow_map_div`, 2 or 4; `RenderSettings.copy` carries it):
  a search radius in texels shrinks with the map, one in world units does
  not.
- A device-baked map is an ordinary `ShadowMap` (float32 depth, the same
  `_linearise`); on a real driver its texels may sit an ulp of NDC z from
  the fill's.
- `_shadow_meta`: `if LI.map_deferred(light): return None, None, None`
  before the ray fallback would let `map_roles` drop its `alone` condition
  (see Limits).

For the integrator:
- `tests/run_all.py`: add `test_r254_bake_speed`.
- Merged cleanly over the viewport, native, flags and map-gpu packages
  (git reported no conflict in core/lights.py; the merged file was read
  through and the two tests that pinned the old readers were rewritten:
  the volume march reads `frame_lights`, a Softness above one texel is a
  filter that keeps the shadow's mass).
- Two failures of `tests/test_r254_flags.py` ('its passes bake the
  ambient literal they always did and read no tail texel', 'and the
  uploaded row is 2 x STRIDE texels') are on claude-cloud at dd9d034 as
  they are here; `tests/test_r251_shadow.py` fails only the 1.89.0 zip
  pin of the baseline list.
- Under the plain fake device (no compute door) the first GPU-device
  frame of a process prints one line, `[Halcyon GPU] shadow maps bake on
  the CPU this session: HAL_RASTER_DEPTH_RGBA32F: compute CreateInfo
  failed: ...` -- the same refusal the camera raster already prints there
  every frame.
- Not done here, by ownership: the Shadow Method tooltip still calls maps
  'soft, fast' and the Map Size tooltip offers size as the only dial
  (properties.py). The verifier's wording: maps are re-rendered whenever
  a lamp or an object moves while traced rays are not; on a heavy mesh
  the bake time follows the triangle count rather than the size.
