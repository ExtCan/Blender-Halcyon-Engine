# R253 -- terrain-tool (the expanded Terrain tool: pictures, erosion, live dials, Export Heightmap)

Branch `r253/terrain-tool-v2`, written against the `claude-cloud` tip
(b85b6ab: 1.91.0 + the field fix + the six merged R253 features). The
version stays 1.91.0 in the worktree; the integrator bumps it once for
the round.

## CHANGELOG

### Expanded Terrain: pictures, erosion, live dials, Export Heightmap (R253)

- **What was asked.** "Expand the Terrain creation tool with map support":
  the Bryce Terrain Editor idiom -- a terrain GROWN from a picture
  (heightmap) as well as from fractals, the classic post-processing dials
  (erosion, terracing, smoothing, raise / zero edges, spikes, fractal
  detail, sea level with a water plane, symmetry / tiling), higher
  resolution, the heights back out as an image, and the ability to come
  back later and re-dial a placed terrain. Road / path carving and
  planet wrap are out of scope; "bake a procedural texture node to a
  heightmap" is NOT built this round (it would need the material
  evaluator fed a synthetic P grid on both roads -- a follow-up).
- **The geometry is split, not changed** (`core/geometry.py`):
  `terrain()` is now `terrain_heights()` (the stage pipeline, returning
  the normalised `(n1, n1)` float32 field) + `heights_to_mesh()` (the
  R201 tuple / quad loops), over `_terrain_grid` / `_terrain_domain` /
  `_terrain_base` -- the 1.91 statements lifted out verbatim, every
  float32 cast and op in its old order. **Proven bitwise**: the pinned
  pre-change sha1s of the vertex arrays for all seven kinds at 32
  divisions, the operator defaults, a dialled case (island / terraces /
  sea / ridge / seed) and the faces all reproduce, and every new stage
  dial passed explicitly at its default is still that mesh.
- **Four landforms appended** after PLATEAU (never reordered; a saved
  file's enum keeps its meaning): **Picture (Heightmap)** -- the heights
  ARE an image (`heightmap` takes a `(n1,n1)` field or raw `(H,W[,4])`
  pixels, sampled by `sample_heightmap`: channel LUMA (Rec.709 weights) /
  R / G / B / A, BILINEAR with edge clamp or NEAREST, optional `tile`
  wrap so a seamless picture makes a seamless terrain; `heightmap_scale`
  / `heightmap_offset`; no image is a flat plane, never an exception);
  **Archipelago** (fBm through a soft threshold at 0.55 -- the sea is
  exactly 0 -- with a ridged crest on each island); **Badlands** (ridged
  fractal under a low-frequency envelope, pre-terraced into nine strata
  with a quintic riser, three talus passes); **River Valley** (rolling
  fBm with one smoothstep channel along `direction`, its centre line
  wandering with a noise of the along-valley coordinate only, the floor
  a flat shelf). Row order: Blender's pixel buffer is bottom row first
  (`compat.image_pixels`) and the grid's row 0 is -Y, so a picture lies
  on the ground as seen from above -- no flip, and Export writes the
  same order.
- **The stages**, each a pure NumPy function on the `(n1, n1)` float32
  field, bpy-free, each guarded by its default: `fractal_detail` (fine
  fBm octaves on any base, pictures included), `symmetrize_heights` (X /
  Y / XY mirrors; RADIAL is the mean of the four rotations summed so it
  is its OWN rotation bitwise), `thermal_erosion` (Musgrave, Kolb & Mace
  1989's talus slide on the 4-neighbour grid: each drop above the talus
  moves rate/4 of itself downhill, the fraction at which a cell fed by
  all four neighbours can never overshoot them -- so mass is conserved,
  the maximum never rises and the minimum never falls, for every rate in
  [0, 1]; material that would leave the grid stays put),
  `hydraulic_erosion` (the water-table approximation: rain, dissolution
  capped at the rock, water + sediment flowing down the rock + water
  drop, evaporation, deposition above the carrying capacity, suspended
  sediment settled at the end -- rock + sediment conserved to rounding),
  `smooth_heights` (the 3x3 cross kernel (4,1,1,1,1)/8 with edge padding,
  clipped into the input's own range), `spike_heights` (Bryce's Spikes:
  thresholded value noise added as needles), `island_mask` (RADIAL is
  the 1.91 ring expression; SQUARE the Chebyshev ring so all four edges
  reach exactly zero), `edge_shape` (ZERO forces the rim to 0.0, RAISE
  lifts the outer eighth toward `edge_height`), `seamless_heights` (the
  core cross-faded with its half-period roll, the last row / column then
  copies of the first, bitwise), `heights_to_pixels` (the export road:
  `(n1,n1,4)` bottom row first, RGB = height, A = 1 -- `sample_heightmap`
  of it at the grid size is the identity bitwise), `water_quad`. Stage
  order: base -> clip >= 0 -> detail -> symmetry -> thermal -> hydraulic
  -> smooth -> spikes -> island ring / edges -> terraces -> sea ->
  seamless (erosion before the ring, so "the island zeroes the rim"
  holds after erosion). The erosion and smoothing passes run elementwise
  in float64 and come back float32 -- no reductions over the field, so
  the same seed slides the same way on every machine.
- **The operator** (`objects.HALCYON_OT_add_terrain`): Divisions up to
  **1024** (1,050,625 vertices; the tooltip and an ERROR label in the
  panel name the count and the pause), the Picture chosen by NAME
  through `prop_search` over `bpy.data.images` (Blender refuses ID
  pointers on operators), and every stage dial by the same name as the
  geometry kwarg (`TERRAIN_FIELDS`), from ONE table (`_terrain_dials`)
  shared with the per-object group so defaults, ranges and menus cannot
  drift; every default equals the geometry default, so an old
  redo-recorded call builds the 1.91 mesh. A grouped `draw()` redo panel
  (Landform / Picture / Fractal / Erosion / Shape / Sea / Shading). The
  Add menu keeps the Terrain line and adds **Terrain From Image** (the
  same operator with `kind = 'PICTURE'`).
- **Re-dial later** (the R211 fur pattern exactly): `execute` copies its
  dials onto `ob.halcyon_terrain` (`HalcyonTerrainSettings`, with the
  real `heightmap` Image pointer) under the `_TERRAIN_BUSY` guard; a
  panel in **Object Properties > Halcyon Terrain** draws the same boxes
  and re-dials the mesh live through the ONE builder `_terrain_realize`
  (which swaps `ob.data` in place, removes the orphan mesh, keeps the
  name -- `from_pydata` only, the R214 rule); Divisions above 512 are
  applied by the **Rebuild Terrain** button only, never by a slider
  tick. **Export Heightmap** reads the mesh's CURRENT vertex Z (sculpts
  export), refuses a non-square mesh by name, divides by the stored
  Height (or normalises to 0..1), writes a float Non-Color image (new,
  or resized when the size differs), and clears the engine's pixel cache
  so a re-import reads the fresh buffer.
- **The water plane** (`water_plane`, default off): a child quad of the
  terrain's size at `sea_level * height` wearing a reflective Halcyon
  PHONG (Reflection 0.3, Specular Level 0.8, Glossiness 200, the
  `water_color` diffuse); re-fitted when the sea moves, its colour a
  node edit, removed when toggled off; remembered by name on the group
  (`water_child`) so a renamed terrain still finds it.
- **GPU story.** Nothing in `gpu/` changes and nothing new reaches the
  renderer: the terrain and its water quad export as ordinary meshes, so
  both devices rasterise them identically by construction. The one
  renderer piece the tool wires, Altitude & Slope, already has its twin
  (`core/nodeeval.n_halcyon_altitude_slope` /
  `gpu/emit.e_halcyon_altitude_slope`); the new test renders a pictured,
  eroded terrain and an eroded Badlands under the Bryce material through
  the CPU road and `_r202_gpu_parity` (max 1e-6 and 3e-6 against the
  6e-3 bar).
- **Tests.** `tests/test_r253_terrain_tool.py` (175 checks): the pinned
  1.91 hashes; pictures (bitwise at the grid size, NEAREST / BILINEAR,
  R / LUMA / G / A, non-square resampling, the bilinear midpoint, edge
  clamp, tiling, the flat plane, row order, mesh Z = image * scale *
  height + offset * height); thermal mass / max / min / flattening /
  determinism and a modelled uphill slide that conserves mass yet breaks
  the max law; hydraulic mass / NaN / determinism; the island and edge
  laws (SQUARE exact zero, ZERO, RAISE, erosion + island, RADIAL bitwise
  the 1.91 ring at three falloffs); smoothing, spikes, symmetry (exact),
  seamless (bitwise, odd counts too), detail; the four landforms (clean
  lattices, deterministic, pairwise distinct from the seven old kinds,
  the Valley turning, the Archipelago's floor majority, the Badlands'
  strata); 1024 divisions; the export round trip (also through the fake
  Image and `compat.image_pixels`); the bpy surface under `fakebpy`
  (registration, the shared dial table, defaults == geometry defaults,
  tooltips >= 40 / items >= 12, the source pins, every icon verified,
  and the one builder on fake mesh / object factories: build, re-build
  in place, the water plane on / re-fit / off, `execute` with a stale
  name warning by name and building a flat plane, with a real picture,
  the live update, the > 512 gate, the busy flag, Export and Normalize,
  the non-square refusal); both-roads parity; the water quad.
  `tests/fakebpy.py` gains `data.images` (`_FakeImage` with a
  `foreach_get` / `foreach_set` pixel buffer, `scale`, `update`) and
  `_Collection.get` / `__iter__` / `__len__`. Constraining tests re-run
  green: `test_terrain_generator`, `test_terrain_types`,
  `test_icon_census`, `test_fur_object_panel`, `test_fur_tufts`,
  `test_every_version_stamp_agrees`,
  `test_every_node_prop_reaches_the_renderer`, `test_tooltips_everywhere`,
  `test_every_setting_has_a_detailed_tooltip`, `test_animation_eta`, and
  the R253 modules `test_r253_convert_buttons` (holding
  `test_old_scenes_render_identically`), `test_r253_cubemap`,
  `test_r253_halo_node`.

## README

**Terrain, expanded (1.92).** *Add > Halcyon > Terrain* grows eleven
landforms now: the seven of before plus **Archipelago**, **Badlands**,
**River Valley** and **Picture (Heightmap)** -- a terrain whose heights
are an image (*Add > Halcyon > Terrain From Image* opens it on that
type). Pick any image in the redo panel; choose the plane it reads
(luminance, or one channel), bilinear or nearest sampling, tiling for a
seamless picture, and a scale and offset. The picture lies on the ground
the way you see it from above (its bottom row toward -Y); set it to
Non-Color so the heights arrive untouched. Under the Landform box sit the
Bryce Terrain Editor's dials: fractal **Detail** octaves on any base
(a picture gets procedural grit), **Thermal** and **Hydraulic** erosion
passes (the talus slide and the rain-and-sediment model of Musgrave's
1989 paper), **Smoothing**, **Spikes**, **Symmetry** (mirror or radial),
a round or **Square** island, **Zero / Raise Edges**, a **Seamless** tile
and the terraces, with Divisions up to 1024 (a million vertices -- the
panel warns, and Blender pauses). **Sea Level** can now put a **Water
Plane** under the mountain: a reflective quad at the sea height, a child
of the terrain, re-fitted as you dial. Every terrain the operator builds
carries its dials in **Object Properties > Halcyon Terrain**, so you can
come back a week later and re-dial it live (above 512 divisions the
**Rebuild Terrain** button applies the change), and **Export Heightmap**
writes the current heights -- sculpts included -- into a float image the
Picture landform reads back exactly. The terrain and its water export as
ordinary meshes, so both devices draw them the same.

## Blender-only

- Executing `HALCYON_OT_add_terrain` end to end on a real Mesh
  (`from_pydata` at 1024 divisions, the F9 redo panel's `prop_search`
  over `bpy.data.images`, the grouped boxes, the > 512 ERROR label).
- The dial tables reaching RNA: `HALCYON_OT_add_terrain.__annotations__`
  and `HalcyonTerrainSettings.__annotations__` are assigned after the
  class bodies (one table for both); `fakebpy.register_class` validates
  them, real `bpy.utils.register_class` has not been run on them here.
- The Object Properties > Halcyon Terrain panel: live `ob.data` swaps
  under slider drags, the `_TERRAIN_BUSY` guard against re-entry, the
  > 512 gate and its INFO label, undo of a re-dial.
- `layout.operator(...).kind = 'PICTURE'` on the Add menu's
  OperatorProperties.
- Export Heightmap on a real `bpy.data.images.new(float_buffer=True)`
  datablock: `scale` on a size mismatch, `colorspace_settings.name =
  'Non-Color'`, the image opening in the Image editor; reading the
  mesh's CURRENT Z after a sculpt; `compat.clear_image_cache` making a
  re-export visible to the next Picture read.
- The water plane: parenting (`child.parent = ob` with the child's local
  frame the terrain's), the PHONG material's reflection in the preview,
  removal when toggled off, undo of the pair, and a user-deleted child.
- Image colour management: an sRGB-tagged heightmap read through
  `compat.image_pixels` arrives display-linear -- the tooltips say to
  set the image to Non-Color.
- The real-time cost of 1024 divisions in `from_pydata` and the
  viewport (3.9 s for the field and 1.3 s for the tuple lists headless;
  Blender's own build on top).
- The CLASSES order against real RNA (`HalcyonTerrainSettings` registers
  before `Object.halcyon_terrain` is assigned -- the fur precedent).

## Deviations from the design

- Test module is `tests/test_r253_terrain_tool.py` and this note is
  `docs-dev/r253/terrain-tool.md` (the protocol's `<feature-key>`
  naming), not the design's `test_r253_terrain.py` / `terrain.md`.
- CHANGELOG.md, README.md, EXTENSION_LISTING.md, version.py,
  `__init__.py` and blender_manifest.toml are untouched (impl_common.md
  overrides the design's version / docs edits; the text lives above).
- `_terrain_base` takes the feature scale `fs` as well (DUNES and VALLEY
  divide by it); `edge_shape(h, mode, edge_height)` needs no `n1`;
  `spike_heights(h, p, amount, scale, slope_only)` takes the noise
  domain `p` rather than a seed (the slope-only variant exists on the
  function and is not exposed as a dial).
- `terrain_heights` / `terrain` also accept `heightmap_channel`,
  `heightmap_sampling` and `heightmap_tile`, and `heightmap` may be raw
  `(H,W[,4])` pixels: `_terrain_params` hands `compat.image_pixels`
  straight through and the geometry samples at the grid size (one
  sampling road, testable without bpy).
- Thermal erosion moves `rate / 4` of each drop per neighbour, not
  `rate / 2`: the quarter is the bound at which the max / min laws are
  PROVABLE for every rate in [0, 1] with four neighbours (at a half, a
  cell fed by all four could overshoot). The dial's range is 0..1.
- Erosion and smoothing compute in float64 elementwise and return
  float32 (mass error 3e-10 / 5e-9 relative instead of ~1e-6).
- BADLANDS pre-terraces with a quintic riser and scales the field by 1.6
  (clipped to 1) so at least five strata show at the default amplitude.
- The seven kind items moved verbatim into the module tuple
  `TERRAIN_KIND_ITEMS` (the operator's `items=` now names it) so the
  group reads the same menu; the new dials are appended to the
  operator's annotations from the one `_terrain_dials` / `_terrain_extras`
  table rather than declared twice.
- The water dials (`water_plane`, `water_color`) update through their
  own `_terrain_water_update` (re-fit the plane alone) instead of a full
  terrain rebuild.
- `fakebpy._Collection.remove` accepts keyword arguments
  (`do_unlink=True`).

## Limitations

- No `flip_y` for pictures (the orientation is stated in the tooltips and
  above); add one only if the field asks.
- `seamless` with an island falloff above 0 cross-fades the zero rim into
  the middle -- the tooltip says to set Island Falloff to 0 for a tile.
- Erosion costs O(passes x n1^2): 200 hydraulic passes at 1024 divisions
  take minutes headless; the live panel rebuilds the whole mesh per dial
  change at or below 512 divisions.
- Hydraulic erosion has no max / min law (deposition legitimately builds
  fans above a cell's neighbours); it is held to mass conservation,
  determinism and no NaN.
- "Bake a procedural Halcyon / Blender texture node to a heightmap" is a
  follow-up (it needs an evaluator entry point on both roads); the
  Detail octaves are the procedural-on-a-picture road this round.
- The Export operator's `invoke` uses `invoke_props_dialog` (Image name,
  Normalize); a bare `execute` from a script writes `<object> Heightmap`.
