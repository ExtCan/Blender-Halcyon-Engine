# R254 -- viewport (what was changed reaches the rendered viewport, and the next F12)

Work package `viewport` of the light audit. The integrator owns CHANGELOG.md,
README.md, version.py, `__init__.py`, blender_manifest.toml and
tests/run_all.py; nothing here touches them. What they need from this
package is under "For the integrator" at the end.

The user's report: "Fix the lights so that changing the size of the lights
actually works. The lights don't get softer/harder with size. Also, I
think some of the general options just don't work properly." A light
option that the renderer honours but that does not reach the rendered
viewport, reaches it late, or is undone by a stale cache looks exactly like
an option that does not work. This package is that half: the audit cluster
`viewport-update` (ten findings) with its verifier's corrections, new
findings and missed options, and one item of cluster `perf`.

Roads checked. The rendered viewport: real Blender 5.2.2 under xvfb
(Rendered shading, camera view, a 524 x 346 frame, the CPU device --
llvmpipe gives the add-on no usable GPU). F12: real Blender, the blocking
render in background mode and the interactive render job under xvfb. The
GPU road: the driver road through tests/fakedevice (the GLSL front-end
simulator plus the device's own upload cache). "Before" in every
real-Blender number is the frozen tree at commit 87489e1 (before any light
fix); "the tree before this round" for the test module is 8de170d, the
commit this branch starts from.

Everything measured is in one log, `<scratch>/wt/viewport/out/viewport_before_after.txt`
(`<scratch>` is the audit's scratch root; the probes that wrote the log are
beside it in `../probes/`, and `make_log.sh` runs all of them on both
trees). In the viewport rows "px" is the number of
frame pixels that moved by more than 1/255 out of 181,304, and "forced full
export" means: tag an unrelated mesh so that a full export of the very same
state runs, and count the pixels that moves -- anything above 0 there is
something the fast path left stale. The Blender sections of the log were
run at commit 39670bb; the commits after it add the node-group walk of
`export._lamp_bound` (pinned headless), comments, tests and this file, and
the headless sections (13) were run on the final tree.

No picture of a scene nobody edited changes. No GLSL is emitted or changed
by this package. No existing test pin was changed.

## 1. Hiding a lamp or a mesh did not take it out of the rendered viewport

- **What the user saw.** H on a lamp: the lamp kept lighting the frame.
  Alt+H: a lamp that had been hidden stayed dark. The same for a mesh. With
  the outliner's eye on a COLLECTION, with Local View and with the
  viewport's own Object Types filter it was worse: the object stayed in the
  picture even after other edits re-exported the scene.
- **What was wrong.** Two things. (a) Every one of those switches reports
  one bare `Scene` update, which `engine._classify_updates` reads as
  'none', so `view_update` kept the parked export. (b) The exporter's only
  visibility test was the depsgraph instance's `show_self`
  (`export.export_scene`, `export.export_lights_into`,
  `compat.evaluated_meshes`, `compat.unconvertible`, the hair and halo
  collectors). A collection's eye, Local View, a local collection and the
  type filter do not remove an object from the depsgraph: they clear the
  per-viewport visibility that Blender's own engines ask for
  (`BKE_object_is_visible_in_viewport`).
- **What it does now.** `engine.view_update` takes the view's own space
  (`context.space_data`) and compares a fingerprint of what that viewport
  shows -- the view layer's objects that pass
  `Object.visible_get(view_layer, viewport=space)`, plus the type switches
  -- with the parked export's (`engine._visible_sig`); a difference is a
  full export. The viewport export skips what the space hides
  (`compat.in_view`), in every instance loop. F12 passes no space and keeps
  its own rule (`show_self` of the RENDER depsgraph: Disable in Renders
  decides). A depsgraph poke that names no Scene does not pay for the
  fingerprint.
- **Instances follow their instancer.** Blender hands its engines a copy of
  an instanced object that carries the INSTANCER's visibility; Python gets
  the source's own answer on `inst.object` (measured: False for an instance
  whose source collection has its eye closed). Asking it hid every instance
  of a library collection kept in the scene with its eye closed -- found by
  probing exactly that set-up, fixed before it shipped: `compat.in_view`
  asks `inst.parent`, and reads the type switch for the instanced object's
  own type (an Empty hidden by the Empties switch still shows what it
  instances, as in Blender).
- **Pictures that change.** Rendered viewport only: an object hidden for
  that viewport is no longer in it. F12: none.
- **Proof.** Real viewport, 14 switches (H and Alt+H on the lamp and on the
  cube, the eye on the lamp's and on the cube's collection, Local View in
  and out, the Lights filter off and on, H and Alt+H on an instancing
  Empty). Before: 0 px for all 14; the forced full export then moved
  97,848 px (lamp), 3,609 px (cube), 567 px (instancer) for the H cases and
  STILL 0 px for the collection eye, Local View and the filter. After:
  97,848 / 3,609 / 99,952 (Local View) / 567 px on the switch itself, and
  0 px on every forced full export. Instances (vp_inst.py): eye closed on
  the source collection leaves the instance and its lamp in (1,687 px, the
  original alone goes), H on the Empty takes them out (82,752 px), Empties
  filter 0 px, Lights filter 99,371 px, Local View on the Empty alone
  leaves exactly its instances. Cost of the fingerprint: 0.23 ms per Scene
  poke at 154 objects, 3.7 ms at 2,154; a lamp tick (no Scene named) pays
  0.01 ms, 0.07 ms at 2,154.
  `test_b_viewport_visibility`.

## 2. A script's or an operator's write to a Halcyon setting never reached the viewport

- **What the user saw.** `light.halcyon.shadow_softness = 8` from a script,
  a preset script, or the panel's own "Use Render Shadow Settings" / "All
  Lights Use These Settings" button: the number in the panel changed and
  the rendered viewport did not, until some unrelated edit touched the lamp.
  A cookie or an exclude collection assigned from a script did not even
  arrive with the next full export.
- **What was wrong.** Blender runs a property's update -- and tags the
  datablock that owns it for the depsgraph -- from the interface always,
  but from Python only when the property HAS an update function. No
  property of `HalcyonLightSettings` had one (the whole add-on had two), so
  a Python write fired no depsgraph update and the EVALUATED copy the
  exporter reads kept the old value.
- **What it does now.** `properties._apply_update_tags` gives every
  property of the four groups (`HalcyonLightSettings`, `HalcyonSettings`,
  `HalcyonWorldSettings`, `HalcyonMaterialSettings`) the callback
  `_settings_written`, which does nothing: having a callback is what makes
  Blender tag the owner. The two properties with a callback of their own
  keep it. The operator needed no edit -- its writes are now what a panel
  drag is. Nothing is stored differently; old files are untouched.
- **Scene-level dials.** With a callback a scene dial written from a script
  now fires its Scene update. That update carries the geometry and
  transform flags (an ID-property write always does; measured), which the
  classifier read as 'full': the same was true of every DRAG of a
  scene-level dial in the interface -- 7 ticks of a Shadow Softness drag
  were 7 full exports (7.6-19.5 ms each on 152 meshes, the mesh cache
  warm) for a value that travels in the settings. `_classify_updates` no longer lets a Scene
  update force an export by its flags; the dial is applied by
  `view_update`'s cheap half (no export at all). Every other scene edit
  measured (unit scale, camera, world, Freestyle, fps, resolution,
  Simplify, a hide, a selection) reports the Scene bare, and what it can
  change in the export arrives as the Object / World / Collection update
  beside it.
- **Pictures that change.** None; values arrive that did not.
- **Proof.** Real viewport, each write followed by nothing else. Before:
  view_update +0 and 0 px for Softness, Density, Shadow Colour, Negative,
  Shadow Map Size, the panel button, both scene dials and the world dial; a
  hand-made `update_tag()` afterwards moved 978 / 1,426 / 1,426 / 99,952 /
  1,426 / 1,426 px for the six lamp writes. After: the same pixel counts
  arrive on the write itself, through the lights-only path at 0.2-0.3 ms,
  and the hand-made tag afterwards moves 0 px. Scene dials: 1,458 and
  1,426 px on the write, no export. The world dial of that probe (Colour
  under a node world) moves no pixel on either tree, but its update now
  arrives; World Light Strength written from a script moves 85,009 px.
  In the real bpy 55 of 55, 417 of 417, 215 of 215 and 95 of 95 properties
  of the four groups carry the callback, and the evaluated copy reads the
  written value (before: 1.0 after writing 7.0).
  A lamp's exclude collection set from a script: before, the cube linked
  into it stayed lit even through forced full exports (the evaluated lamp
  never got the pointer); after, 1,702 px. `test_c_settings_carry_an_update_callback`.

## 3. Every light slider tick ran the full scene export

- **What the user saw.** Dragging Power, Radius, Colour, Spot Size or any
  Halcyon light slider in a scene with many objects was heavy: the
  viewport lagged behind the mouse.
- **What was wrong.** A 5.x light owns an embedded node tree
  (`Light.node_tree`) that Blender tags with every edit of the light, so a
  tick arrives as [Object, PointLight, ShaderNodeTree].
  `engine._classify_updates` listed 'ShaderNodeTree' as a full export, and
  `export._IRRELEVANT_IDS` named 'Light' where the update's type is the
  subclass, so the whole mesh cache was flushed as well ('dirt
  ALL(PointLight)').
- **What it does now.** A node tree that belongs to a light updated in the
  same list is that light's edit ('lights'); any other tree is still a full
  export. The four light subclasses and the Freestyle line style are named
  in `_IRRELEVANT_IDS`. Landed together with items 4 and 5, which keep the
  lights-only path right now that data edits take it.
- **Pictures that change.** None: the lights-only export builds the same
  `Light` objects.
- **Proof.** Real mouse drags on the probe's own three-row panel, 152 mesh
  objects (the machine was shared; the times are per tick as logged).
  Power: before 7 ticks = 7 full exports, 43-83 ms each; after 7
  lights-only exports, 0.39-0.57 ms each with one at 2.2 ms. Halcyon
  Shadow Density: 6 x 48-142 ms -> 6 x 0.36-0.57 ms. Scene Shadow
  Softness: 7 x 7.6-19.5 ms -> no export. The frames match on both trees
  (918, 452 and 2,137 px), and through the whole of vp_misc.py (two lamps,
  type changes, Volumetric Occlusion) every forced full export after a
  lights-only frame is 0 px.
  `test_a_update_classifier`; the six pins of
  `test_render.test_light_edit_fast_paths` pass unchanged.

## 4. Scrubbing a keyframed lamp froze its Light Style and Caustics

- **What the user saw.** A lamp with keyframed Power or position that also
  flickers (Light Style) or carries Caustics: on the timeline the Power and
  the position animated and the flicker stood still.
- **What was wrong.** A frame step whose only updates are a lamp's goes
  through `export.export_lights_into`, which shallow-copies the parked
  scene -- frame, fps and time included. `lights.apply_light_styles` and
  the caustic cookies read `scene.time`.
- **What it does now.** `export_lights_into` writes the three clock values
  with the expressions `export_scene` uses. A stub depsgraph with no clock
  keeps the parked one.
- **Pictures that change.** Viewport frames of a keyframed, styled lamp
  while scrubbing or playing: they are now the full export's frames.
- **Proof.** Real viewport, FLICKER_A, frames 13 / 44 / 30. Before: the
  exported gain was the previous frame's (1.000 / 1.167 / 1.333 where the
  true ones are 1.167 / 1.333 / 1.167), scene.frame lagged one step, and
  the forced full export moved 92,150 / 6,330 / 18,950 px. After: the true
  gain and frame, 0 px. Headless: the lights-only frame of frame 13 is
  `np.array_equal` to the full export's and 3,874 px from the stale-clock
  one. `test_d_lights_only_export`.
- **The verifier's residue** (an image SEQUENCE on the floor beside an
  animated lamp): measured. A frame step reports [Action, Object,
  PointLight] -- no Image -- and takes the lights-only path, and the forced
  full export of the same frame is 0 px on both trees: the full export
  reads the same pixels, i.e. an image sequence does not advance in the
  viewport on either path. Not a lights-only defect; not in this package.

## 5. A material that bakes a lamp kept the old lamp

- **What the user saw.** An Object Projection node that projects from a
  lamp: moving or turning the lamp left the projection where it was.
- **What was wrong.** `export.object_map_props` bakes the picked object's
  inverse matrix (and a spot's cone) into the MATERIAL graph;
  `export_lights_into` shares the parked materials.
- **What it does now.** `object_map_props` marks a node whose object is a
  lamp (`object_is_light`, written for lamps only -- every other node's
  props are what they were); `export_scene` records what the materials
  bake of the lamps (`Scene.lamp_bound`); `export_lights_into` returns None
  for such a scene and `view_update` exports fully. The same guard covers a
  Light Group: its lamps are resolved to NAMES at material export, and a
  rename reports the lamp object alone (measured), which classifies as a
  lights-only edit -- the lights-only export stands down when the set of
  lamp names differs from the parked one. Scenes without either keep the
  fast path.
- **Proof.** Real viewport: lamp moved / turned, before 8,525 / 10,755 px
  stale on the forced full export, after 0 px ('lights->full' in the log);
  Spot Size and Power on that scene: 0 px. Renamed lamp in a group: the
  material's names follow ('Key Lamp'), forced full 0 px.
  `test_d_lights_only_export`.

### 5b. Found on the way: a Light Group never resolved in Blender

- **What the user saw.** A BI Material's Light Group did nothing: the
  material was lit by every lamp, and the console said the group's
  collection had no lamps.
- **What was wrong.** `export._node_props` looks the collection up with
  `bpy.data.collections.get(...)`, and export.py has no module-level `bpy`.
  The NameError was swallowed by the guard around it, so every group
  resolved to an empty list (which the core reads as "no group"). It has
  been so since the first commit of the file; the core's tests hand the
  names in directly and never saw it.
- **What it does now.** The lookup imports bpy.
- **Pictures that change.** Blender renders (viewport and F12) of a
  material with a Light Group set on the node: it is now lit by that
  group's lamps only, which is what the option says. Legacy imports that
  carried `light_group_lights` in directly are unchanged.
- **Proof.** Real viewport, a cube whose BI Material names collection
  'Key', a second lamp outside it: clearing and setting the group moved
  0 px before (exported names [], with the warning) and 1,677 px after
  (names ['L'], no warning). `test_d_lights_only_export` (the last two
  checks).

## 6. No F12 ever reused a cached mesh

- **What the user saw.** Every F12 exported every mesh again, edited or
  not ("meshes 63x ... (0 cached)"); relighting a heavy scene paid the
  whole export at each render.
- **What was wrong.** `export._on_frame_change` forwards the depsgraph's
  update list to `_watch_updates` when the frame did not move. On an F12
  that depsgraph is the RENDER depsgraph Blender has just built, and a
  newly built depsgraph lists every ID it holds: every object was dirtied,
  and the first unlisted type (PointLight, FreestyleLineStyle) flushed the
  pool.
- **What it does now.** Edits do not need that list: they arrive through
  `depsgraph_update_post` of the main depsgraph before the render starts.
  `export._heard_objects` names the objects for which that is true, and
  the walk skips them; every other object is dirtied exactly as before. An
  object is "heard" only when ALL of these hold (the pixel counts are the
  stale mesh measured with that one test switched off):
  * the depsgraph is a RENDER one and the frame did not move;
  * no foreign function sits in `render_init`, `render_pre`,
    `frame_change_pre` or `frame_change_post` (the verifier's 207 px for
    a handler that moves a vertex; a handler that tags what it edits is
    also caught by the next test);
  * a window shows this scene and view layer -- a scene rendered by name
    from a script has no main depsgraph anyone evaluates (2,875 px);
  * that main depsgraph holds no tag it has not reported yet
    (`Depsgraph.id_type_updated`, measured: set when an ID is tagged,
    cleared by the evaluation that reports it). This one closes a case
    that was reasoned, not produced: an edit and the F12 key handled in
    one event batch;
  * the object is enabled in viewports (`_viewport_enabled`, Blender's
    layer sync read off the original data and checked against
    `Depsgraph.ids` in ten configurations): a render-only object is in the
    render depsgraph alone (145 px);
  * its mesh cannot change behind the depsgraph's back (`_reads_outside`):
    no driver, no texture-reading modifier, no node modifier, no file
    cache -- pixels a script writes fire no update at all (2,106 px).
  And an Image / Texture / ImageTexture update empties the RENDER pool:
  Blender reports the texture of a reloaded displacement image and does not
  re-evaluate the object (2,106 px). The VIEWPORT pool still agrees with
  the viewport depsgraph and is left alone.
- **Pictures that change.** None: the cache stores the arrays a live
  export produces.
- **Proof.** Background F12s, 25 meshes, 21 steps: before "live 25 cached
  0" at every step; after "live 2 cached 23" with nothing edited (the two
  are the render-only object and the instanced one, never heard), the same
  after a lamp's Power, a Halcyon softness written by a script, a scene
  dial, a moved lamp; "live 3" after one mesh moved, scaled or edited;
  "live 25" after a modifier or a material slot was added, a frame step,
  Disable in Renders. At all 21 steps the warm F12 equals an F12 with the
  export cache flushed: 0 px. The interactive render job (the F12 key's):
  "live 63 cached 0" -> "live 0 cached 63", "live 1" after a mesh edit.
  With a foreign render_pre handler, a scene no window shows, a reloaded or
  script-painted displacement image: every mesh live, 0 px, on both trees.
  `test_e_f12_mesh_cache`.
- **What is not covered.** An object reached only through an instanced
  collection, a geometry-nodes object, a driven object: exported live at
  every F12, as all were. See "Not proven" for the one risk this rule
  cannot rule out.

## 7. A repainted or reloaded projector image was served stale

- **What the user saw.** A gobo (Projected Texture) repainted in Blender,
  or rewritten on disk and reloaded, kept projecting the first picture --
  in the viewport and on F12, until Blender was restarted. Any material
  texture behaved the same.
- **What was wrong.** `compat.image_pixels` keyed its cache on the name,
  size, source, path, the dirty flag and the colour space. None of those
  moves on a reload or on the second edit of an already dirty image.
- **What it does now.** A clean file image is also keyed on its file's
  (mtime in nanoseconds, size). A DIRTY image is read at every export --
  Blender fires nothing when pixels change, so reading them is the only
  way to know -- and when the fresh read equals the cached copy the cached
  ARRAY is returned, so identity-keyed caches downstream stay warm. One
  generation per image is kept.
- **Cost.** One `os.stat` per file image per export; one pixel read per
  dirty image per export (and per lamp tick for a dirty gobo).
- **Proof.** Real viewport, a Spot with a 64 x 64 gobo, each change
  followed by one update of the lamp: repaint to a checker / to rings /
  file rewritten and reloaded -- before 0 / 0 / 0 px, after 9,581 /
  11,596 / 10,456 px. A pixel write alone fires no update on either tree
  (view_update +0): the viewport shows it at the next update of anything.
  `test_g_image_pixel_cache`.

## 8. F12 on a held frame ignored edits (Shoot On twos / threes)

- **What the user saw.** With Shoot On 2s, an F12 on an even frame after
  moving or re-powering a lamp returned the old picture and printed
  "holds frame 1 (not rendered again)".
- **What was wrong.** The held cel (`engine._HOLD_CACHE`) is keyed on the
  scene's name, the key frame, the size and the settings; scene content
  never invalidated it.
- **What it does now.** `engine._drop_holds_handler`
  (depsgraph_update_post) drops the cel on any update of the main depsgraph
  that would re-export the viewport. A RENDER depsgraph's updates and
  anything during Blender's render job are ignored, so the holds inside a
  sequence render survive; load_post drops it too.
- **Proof.** Background: lamp Power x4 and red, F12 on frame 2 again:
  before 0 px, after 5,702 px; a mesh moved: 0 -> 1,403 px; a selection
  keeps the cel; an animation 1..4 on twos with the lamp keyframed still
  holds (frame 2 vs 1 and 4 vs 3: 0 px, 3 vs 1: 5,817 px). The interactive
  render job: 0 -> 5,692 px, and the same holds. `test_f_film_hold_follows_edits`.

## 9. The mesh key read 512 samples of everything but the positions

- **What the user saw (GPU road).** A small object in a large scene
  switched from flat to smooth kept its old shading; given another
  material, the GPU refused every frame and the CPU drew it.
- **What was wrong.** `gpu/shade._mesh_key` keys the plan and the
  attribute / triangle uploads. Its position term is an exact digest (it
  landed before this round); its normals, uvs, uvs2, material-index and
  triangle terms were sums over about 512 samples.
- **What it does now.** `_attr_digests` appends the exact digest
  (`mathx.content_digest`) of every other array those uploads pack: tris,
  normals, uvs, uvs2, colors, mat_index, obj_index, face_normals, smooth.
  Memoised on the mesh and reused only while each attribute IS the array it
  digested, so a draft, a refine and a lights-only export pay nothing; the
  old terms stay in their places.
- **Pictures that change.** None for a scene nobody edited (the key of the
  same bytes is the same key). GPU road: the two stale cases above are now
  the cold frame.
- **Proof.** The verifier's probe on this tree (53,003 vertices, a
  24-vertex box): key equal=False for both edits; GPU warm-vs-cold 245 px
  -> 0 px, "REFUSED" -> engaged. `test_h_mesh_key_digests_every_array`
  (GPU road through the fake device's upload cache).
- **An object turned about its own centre** (cluster perf,
  verify-stale-sum-blind-10): covered by the position digest that landed;
  held by `test_h2_an_object_turned_about_its_own_centre` (the key moves,
  the second render is bitwise a render with every cache cleared; it
  passes on 8de170d as well, as a pin should).

## 10. The verifier's missed options, each driven in the real viewport

| Option | Before | After |
| --- | --- | --- |
| Second lamp: Power, move, type change | right picture, full export each | right picture, lights-only 0.3 ms; forced full 0 px |
| Second lamp: H / Alt+H | 0 px (stale 90,867 px) | 90,867 px |
| Second lamp deleted | 90,867 px | the same |
| Undo after a Power edit | 9,917 px, full export | the same |
| Object.data swapped to another Light and back | 17,611 px | the same |
| Scene-level light dials | UI drag: a full export per tick; script: nothing | no export; arrives on the write |
| Lamp renamed inside a Light Group | group dead (item 5b) | names follow, forced full 0 px |
| Volumetric Occlusion (a box inside a Spot's beam, Spot Cones on) | 5,741 px, full export | 5,741 px, lights-only; forced full 0 px |
| Blender's Exposure / Temperature / Normalize | update arrives ('full'), 0 px | update arrives ('lights'), 0 px: the export does not read them yet (another package) |
| A lamp's exclude collection | script write dead; link 0 px | 1,702 px |
| The world's own light beside a lamp | (not in the old tree) | survives every lights-only export (the lamp list is rebuilt, `worldlight.sync` re-lights it at render entry); its dials written from a script arrive (85,009 px) |

Volumetric Occlusion is live, not inert: the audit's rig had no geometry
inside a beam.

## Neutrality

- The feature matrix, all 313 rows, CPU road (`R.render`): the frame
  digests of this tree and of 8de170d are identical, row for row.
- The same 313 rows through the viewport worker (CPU device), two frames
  each from one scene object (the second rides every cache): identical.
- Every sixth row (53) on the driver road through tests/fakedevice, two
  frames each: identical digests, the same 50 rows engaged.
- An export for a viewport that hides nothing is the export with no space,
  array for array; the same content in fresh arrays has the same mesh key
  (`test_i_an_unedited_scene_is_the_same_bits`).

## Pins changed

None. Existing tests run and green on this tree: the 16 functions of
tests/test_render.py that touch the changed functions (one of them,
`test_a_released_context_render_still_completes`, fails in that run order
on 8de170d as well and passes alone on both trees), the tooltip and
update-callback censuses, tests/test_r253_review.py,
test_r253_mapping.py, test_r253_terrain_tool.py,
test_r253_light_options.py, test_r254_falloff.py,
test_r254_world_light.py.

tests/test_r254_viewport.py: 124 checks pass here. On 8de170d 25 checks
fail and 8 of the 10 tests stop on a function that does not exist there;
the turn-about-centre test passes there too.

claude-cloud has moved since this branch was cut (the map-soft, ray-road
and spot-beam packages). `git merge-tree` of this branch with f6484e3
reports no conflict, and on that merged tree this module and
test_r254_map_soft, test_r254_ray_road, test_r254_spot_beam,
test_r254_falloff, test_r254_world_light all pass.

## Not proven, and what is left

- **The GPU road in a real viewport.** llvmpipe gives no usable device;
  every GPU statement here is the driver road through tests/fakedevice.
- **Brush strokes.** The image cache was driven with script pixel writes
  and reloads. A real paint stroke in the Image Editor was not simulated;
  it is expected to report an Image update (a full export, which now reads
  the dirty pixels), but that was not measured.
- **The F12 cache and state no depsgraph reports.** The rule in item 6
  rests on "an edit of a heard object reaches `depsgraph_update_post`
  before the render". Six ways that is false were found and closed. What
  cannot be ruled out is a seventh of the same kind: an evaluated mesh that
  depends on something outside the depsgraph through a route not listed in
  `_reads_outside`. The failure would be one stale object in an F12 at an
  unchanged frame; a frame step, an undo or a file load clears it.
- **Two event-loop cases reasoned, not measured.** An edit and an F12 key
  press handled in the same event batch (the pending-tag test is what
  catches it); a main-depsgraph rebuild racing the render job's read of
  `id_type_updated` (a stale read can only say "pending" or "not pending"
  one evaluation late; the first is safe, the second is the same batch
  case).
- **Left as found.** A custom node's property written from a script (the
  Light Group string, for one) fires no update: the nodes module has the
  same gap item 2 closed for the four property groups. A scene rendered by
  name whose camera was moved by a script without an update renders through
  the old camera matrix (the export reads the camera of the original
  scene). Tiled (UDIM) images have no file stamp. An image sequence does
  not advance in the viewport. `preview._post` passes no depth (the
  spot-beam package edits that function).

## For the integrator

- **Files.** engine.py (`view_update`, `_classify_updates`, `_visible_sig`,
  `_names_scene`, `_drop_holds_handler`, `_render_job_running`, register /
  unregister), export.py (`_IRRELEVANT_IDS`, `_RENDER_FEEDS`,
  `_watch_updates`, `_on_frame_change`, `_heard_objects` and its helpers,
  `_cache_scrub`, `export_lights_into`, `_lamp_bound`, `_lamp_names`, the
  `space` argument of `export_scene` and of the hair / halo collectors, one
  key in `object_map_props`, the bpy import in `_node_props`), compat.py
  (`in_view`, the `space` argument of `evaluated_meshes` and
  `unconvertible`, `image_pixels`), properties.py (`_settings_written`,
  `_apply_update_tags`), gpu/shade.py (`_mesh_key`, `_attr_digests`),
  tests/test_r254_viewport.py, this file.
- **Signatures other packages may meet.** `export.export_scene(depsgraph,
  settings, warnings=None, space=None)`;
  `export.export_lights_into(parked, depsgraph, space=None)` may return
  None (the caller exports fully); `compat.evaluated_meshes(depsgraph,
  skip=None, space=None)`; `export._watch_updates(scene, depsgraph,
  heard=None)`; `export._cache_scrub(mode=None)`.
- **Not touched.** `export_light`, the shadow maps and their keys,
  `preview.py`, ui.py (the adopt-shadow-settings operator needed no edit:
  the update callbacks deliver its writes), CHANGELOG / README / version /
  manifest / run_all.
- **run_all.** Add `test_r254_viewport` beside `test_r254_world_light`
  (about 12 s).
- **CHANGELOG, in the user's words.** Hiding a lamp, a mesh or a collection
  now hides it in the rendered viewport (H, the eye icons, Local View, the
  viewport's object-type filter). Halcyon settings changed from a script or
  by the panel's own buttons update the rendered viewport at once. Dragging
  a light slider no longer re-exports the scene (0.4 ms a tick instead of
  40-140 ms on 152 objects); dragging a scene-level dial exports nothing.
  A keyframed lamp's Light Style and Caustics follow the timeline in the
  viewport. An Object Projection from a lamp follows the lamp. A BI
  Material's Light Group works in Blender (it never resolved). F12 reuses
  cached meshes when nothing that could change them was edited. A repainted
  or reloaded image (a gobo, any texture) is read again. Shoot On twos: an
  edit re-renders the held frame. GPU: a small object's smooth flag or
  material slot change in a large scene is no longer served from a stale
  upload.
