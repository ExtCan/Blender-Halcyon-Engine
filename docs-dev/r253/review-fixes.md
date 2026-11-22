# R253 review fixes: what the adversarial re-read found and what changed

The round's review (three lenses per feature, three refuters per finding)
and the console-shader deep re-read the field log asked for. Each item
below was reproduced headless before it was fixed and is pinned in
`tests/test_r253_review.py` (registered in `run_all`); the Coded
Shader's name resolution is pinned in `tests/test_shaders.py` as well,
and its matrices, indexed writes, array fields, integer division and
float vectors built from ints there alone.

## CHANGELOG

- **The review's confirmed findings, fixed.** Every finding below was
  reproduced with a probe on the merged tree before the fix and is pinned
  in `tests/test_r253_review.py` (8 tests), with the GPU twin re-proven
  where a road changed. (1) *A singular object matrix crashed the frame.*
  A zero scale on any axis (a plane flattened with Scale Z 0, an animated
  scale passing through 0) made `np.linalg.inv` raise `LinAlgError` out of
  the whole render on both Generated roads -- `core/render.object_inverse`
  takes the pseudo-inverse (the point lands on the collapsed plane) and
  prints once by name. (2) *A console type's pinned opacity was invisible
  to the alpha classifiers.* The Saturn mesh colour calculation pins
  Opacity 0.5 in `console.resolve` with its socket hidden at 1.0, so
  `export._alpha_reason` and `material_see_through` classified the
  material opaque and the half-transparent polygon never reached the
  transparent pass; `core/console.pinned_alpha_reason` feeds both, on
  both roads. (3) *The corner road skipped the alpha law.* The ALBEDO
  pass of every vertex- and face-rate material (every console machine)
  returned its raw fractional opacity: no punch-through clip, no Screen
  Door compare, no Transparency None -- the pixel road's law alone.
  `ShadeJob._alpha_law` is that law factored into one method, applied on
  both roads (an opaque corner material is bitwise unchanged). (4) *The
  PC fixed-function Texture Op reached one shading type.* The node drew
  Modulate / 2x / 4x / Add / Add Signed for all four D3D shading types but
  only the separate-specular combine honoured it; D3DTSS_COLOROP is
  texture-stage state, orthogonal to the shade mode, so `core/combine.
  cb_pc_texop` and its GLSL twin run the Gouraud and the two flat types'
  product through the same 8-bit ops (Modulate stays the plain float
  product, the 1.90 bits; the simulator twin d = 1.8e-7). (5) *The
  vertex-colour material sources dropped the texture.* GX_SRC_VTX, the
  N64's unlit shade (G_LIGHTING off), Model 3's fixed shading and D3D's
  COLORVERTEX replaced the TEXTURED albedo with the vertex colour -- the
  hardware replaces the material colour and the texture still modulates
  it. A linked Diffuse Color is now multiplied by the vertex colour
  (`base * (1 + (vcol - 1) * mix)`, one op per statement on both roads),
  an unlinked one replaced as before; the N64 Modulate control is
  untouched. (6) *Super FX and 3DO lit a hidden highlight.* Their
  specular sockets are hidden (`MACHINE_SOCKETS`) but the hidden socket's
  0.5 still fed the light loop; `resolve` pins the level at 0. (7) *A
  1.90 Model 2 / Model 3 master changed its highlight at file load.* The
  old master snapped its Glossiness to the board's four specular settings
  (`sega_model2_spec` / `sega_model3_spec`); the migrated node took the
  menu default (Power 4 / Exponent 16). `migrate_master_node` now snaps the
  old sockets with the same thresholds (Specular Level 0 -> Off). (8) The
  Model 2 / Model 3 specular menus carried labels under the house rule's
  12 characters ("Off", "Power 4"); named now. The PC node's Colour Vertex
  toggle gained the update callback that refreshes the Vertex Color Mix
  socket's visibility. (9) *The GPU plan cache served stale object
  frames.* `_plan_sig` fingerprints the mesh, the settings, the lights and
  the object colours, but not the per-object inverse rows and Generated
  box the R253 road BAKES into the pass as literals -- a translation keeps
  the mesh key, so a moved object (or a Texture Space edit) re-used the
  cached plan with the old box. The frame and the matrices are in the
  signature now (pinned: a translation changes it, a Texture Space edit
  alone changes it). (10) *A transformed flat object read noise on its
  flat axis.* The 1e-6 span clamp divided the inverse chain's float32
  residual (about 1e-7 of |P|) into +/-0.5 of speckle on a rotated plane's
  Generated.z; `_gen_span` gives a flat axis of a TRANSFORMED object the
  span 1.0 (it reads within 1e-6 of 0) while an identity object keeps the
  1.91 clamp bit for bit.
- **The field's greyed redo panels and refused undo / redo (hardened).**
  Blender's own rule (`WM_operator_check_ui_enabled`, `ed_undo_step_pre`):
  every Adjust Last Operation panel is disabled and undo / redo are
  refused while a non-background job owned by the scene is running, or
  while the operator's undo step is gone. Three things in the add-on
  could feed that and are closed: (1) the F12 render acquired the GPU
  pipeline lock with NO bound, so a viewport frame that never released
  it parked the render job for the whole session -- `engine.render`
  waits 30 s, then shades on the CPU and says so; (2) the
  pristine-material watcher (`convert._halcyon_default_material`) runs
  on the depsgraph pass AFTER the operator's undo push, so Ctrl+Z
  restored the pristine tree and the watcher converted it straight back
  (undo looked broken) -- it converts each material ONCE per session
  (`_HANDLER_CONVERTED`); (3) a 1.90 scene saved with Detection=Set and
  a model that moved off the master menu reads the model back as '',
  which `core/convert.plan` passed through verbatim and `master.model =
  ''` raised inside every Convert button, leaving an unlinked master
  node per click -- anything but a master model is AUTO now, the node's
  model assignment is guarded, and the panel never pushes an empty
  model (the review's confirmed finding). The field's diagnostic is the
  jobs list: `bpy.app.is_job_running` for RENDER / COMPOSITE /
  OBJECT_BAKE names the job that holds the panel.
- **The render job can no longer be parked (the second pass over the
  same field pair).** A five-lens hunt over Blender 5.2's own sources
  (`wm_jobs.cc`, `ed_undo.cc`, `wm_operators.cc`, `render_internal.cc`)
  and the add-on found the roads that could keep the F12 job alive for
  the rest of a session, and closed every one. (1) *The worker pool's
  stderr pipe.* Each worker was started with `stderr=PIPE` and that pipe
  was read only after a failure -- yet the renderer prints inside every
  band (the worker routes `print()` to stderr to keep the protocol
  stream clean), so a persistent worker filled the pipe after enough
  frames (64 KiB on Linux, a few KiB on Windows), blocked inside its own
  print, never answered, and the parent's `recv()` -- a bare pipe read --
  waited for the rest of the session; an F12 job that never ends greys
  every redo panel and refuses undo / redo, and Ctrl+Z (which kills every
  job by JOINING its thread) then froze the application. Now the child's
  stderr is its log file, replies arrive through a reader thread, and
  `Worker.recv` waits in 0.25 s slices that answer Blender's
  `test_break` (Esc, the undo kill, the next render) with
  `WorkerCancelled` -- the one pool outcome that is not an in-process
  retry -- or a deadline (60 s for the handshake, an hour per band) with
  a named IOError and the in-process road. Reproduced headless before
  the fix: 900 frames of a scene whose band road prints one note per
  call (a W encoding under an orthographic camera), two workers -- the
  1.92.0 pool wedged with the worker in `anon_pipe_write` and the
  parent's render thread in `anon_pipe_read` once the pipe filled; the
  fixed pool finished all 900 with 112 KiB of worker output per worker
  in its log. (2) *The GPU marshal's pickup
  wait.* A burst waiting for the main-thread pump sat out the whole 8 s
  TIMEOUT per burst while Blender's undo or the next render joined the
  job thread ON the main thread (the pump could not fire): an 8 s freeze
  per outstanding GPU stage. `render()` installs `test_break` as the job
  thread's cancel hook (`gpu/marshal.set_cancel`, per thread, restored in
  `finally`); the wait looks at it every 50 ms and raises
  `marshal.Cancelled` -- a BaseException on purpose, so the stages'
  `except Exception` CPU fallbacks do not keep a cancelled render alive
  -- which the engine catches like its own `_Cancelled`. The pump's
  registration is now checked against `bpy.app.timers.is_registered`
  instead of a flag a file load could leave stale (after which every
  burst timed out and fell back to the CPU for the session). (3) *The
  display pin fought undo.* `_pin_display_handler` rewrote Color
  Management on EVERY depsgraph update, which lands after each
  operator's undo push -- an undone Color Management edit was re-applied
  by the next pass and undo looked broken there. It pins on transitions
  only (a scene's engine becoming Halcyon, and every file load);
  `render()` and `view_update` still pin the frames they draw. (4) *The
  cause is visible.* The Debug panel names any running Blender job
  (`bpy.app.is_job_running`) in red with the rule, so a render that
  never finished is seen as what it is instead of as a broken interface.
  Reproduced in a real Blender 5.2.2 (UI under Xvfb, `ed.undo_redo.poll`
  being Blender's exact redo-panel test): with the add-on on, the CPU
  road, the GPU road, the rendered viewport, the material tab, a
  Halcyon material and three F12 renders all leave the panel enabled and
  undo / redo working, as EEVEE does; the parked-job roads above are the
  ones that differ from that baseline in the field, and each is pinned
  headless (`test_field_worker_pool_waits_are_bounded`,
  `test_field_marshal_pickup_wait_is_cancellable`,
  `test_field_display_pin_on_transitions_only`). The same hunt's second
  batch: (5) *the F12 pipeline wait answers test_break* -- the 30 s
  acquire is taken in 0.25 s slices and gives up the moment Blender
  stops the job, so a kill no longer holds its join for the bound; (6)
  *thumbnails never touch the driver* -- a material / world / icon
  preview renders on the CPU with no pipeline lock, because Blender
  joins the preview job's thread on the main thread for every
  node-editor operator, every undo and every F12 start, and a thumbnail
  parked behind a viewport frame's lock froze those; (7) *every GPU
  render marshals* -- Blender copies `bl_use_gpu_context` into the
  engine type once at registration (False) and a later class-attribute
  write never reaches it, so the old `granted` read of that stale
  attribute switched the marshal OFF under Hold GPU Context: bursts
  still queued, the pump retired after each one, and the pickup race
  cost 8 s timeouts (the Hold toggle keeps its background meaning only;
  its tooltip says so); (8) *the pump is persistent and retires under
  the lock it registers under* -- a file load dropped the non-persistent
  timer, and a burst queued between the pump's unlocked "queue empty"
  look and its flag clear waited out the whole timeout for a pump that
  had just left; (9) *view_update pins through the transition table*
  too, so a rendered viewport no longer re-applies undone Color
  Management edits on every depsgraph poke.
- **The Coded Shader's socket rebuild survives undo, redo and File >
  Open.** The node's Language and Text update callbacks may only FLAG the
  rebuild (`needs_rebuild` -- sockets are never made or cleared inside an
  RNA update) and a one-shot `bpy.app.timers` function, `_run_rebuilds`,
  does it on the main loop's next pass. Blender pushes the edit's undo
  step before that timer runs (5.2's own order: `ui_apply_but_funcs_after`
  pushes inside the event handlers, `wm_event_do_notifiers` runs the
  timers after them), so the step holds `needs_rebuild` True with the
  OLD sockets and the rebuild lands outside any step; memfile undo then
  restores datablocks without firing one update callback -- the only
  road to `_schedule_rebuild`. Ctrl+Shift+Z onto such a step
  (or Ctrl+Z onto it from a later one) brought back a node with stale
  sockets, the flag up and nothing to read it; the next Coded Shader
  edit anywhere rebuilt it by accident, and a file saved in between
  carried the state into the next session. (1) *Every landing re-arms.*
  `_rearm_code_rebuilds`, a persistent handler on `undo_post`,
  `redo_post` and `load_post` (once each, and run for the already-open
  file at registration), scans every material, world and node-group
  tree for a flagged Coded Shader and schedules the rebuild exactly as a
  fresh edit would. In background mode no main loop will ever fire a
  timer (`-b` never enters `WM_main`) and a handler is not an update
  callback, so it rebuilds at once: a render farm opening a file saved
  flagged exports the right sockets. (2) *The queue is checked against
  Blender.* `_schedule_rebuild` trusted a module flag alone; a File >
  Open that dropped the pending non-persistent timer
  (`wm_file_read_pre` -> `BLI_timer_on_file_load`) left it set, and no
  rebuild could be scheduled again that session (the marshal pump's
  disease). The flag counts only while `bpy.app.timers.is_registered`
  agrees. (3) *The Compile
  button's own step is whole.* `halcyon.compile_shader` (REGISTER |
  UNDO) rebuilt the sockets and left the flag up, so the step it pushed
  re-armed a second rebuild on every landing; it clears the flag before
  it compiles. (4) *Considered and left: the timer's rebuild as an undo
  step of its own.* Sending it through the operator gives two steps per
  edit, and the first is still the flagged one -- an undo onto it must
  rebuild again, and a step pushed there lands on top of the state the
  undo just restored: the redo stack is cut and Ctrl+Z can never get
  past the flagged step. (A timer also runs with a window and no
  editor, so the operator's `space_data.edit_tree` has nothing to find.)
  The sockets are derived data, so the flag is what the stack carries
  and they are re-derived on arrival, un-pushed (pinned: nothing on the
  rebuild road calls an operator or an undo push). By Blender's own rule
  -- `BKE_undosys_step_undo` loads the step before the active one -- one
  Ctrl+Z still takes the whole edit back and one Ctrl+Shift+Z brings it
  back whole. Pinned in
  `test_code_node_rebuild_survives_undo_redo_and_load`: the real node
  class against a stand-in memfile stack that swaps in fresh copies of
  each step and fires no update callback, through undo, redo, a file
  load with a rebuild still queued, a node group, background mode and
  the Compile operator (the fake bpy gained `fire` -- Blender's
  one-or-two-argument handler call -- `file_load` and persistent
  timers). Seven mutations of the fix each fail it.
- **The Coded Shader's Auto Compile checkbox is retired, and Inline
  Source says what it is.** `HALCYON_CodeNode.auto_compile` ("Auto
  Compile", on by default) was drawn in the node's sidebar and named in
  four places -- its declaration, `draw_buttons_ext`, its `PROP_TIPS`
  tooltip and the census's `NOT_SHADING` reason -- and read in none. The
  tooltip promised "off waits for the Build button (steadier while
  typing long shaders)"; with the box unticked the node rebuilt exactly
  as with it ticked: `_lang_changed` / `_source_changed` flag and
  schedule, `_rearm_code_rebuilds` schedules every flagged node on undo,
  redo and load, `_run_rebuilds` rebuilds every one, and
  `export._compile_node` compiles the source at every export. The choice
  was to wire it or to retire it; it is retired, on three findings. (1)
  *Nothing rebuilds while typing.* The only automatic rebuild is the one
  a Language or a Text-pointer change flags. `source_text` has no update
  callback and no widget, and typing into the assigned Text fires
  nothing on the node (seen in a real Blender 5.2.2: `Text.write` /
  `clear` / `write` leave `needs_rebuild` down and the sockets as they
  were) -- after an edit in the Text Editor the sockets have always
  waited for the Compile button, with or without the box. (2) *The
  picture never waited.* The exporter compiles the live source whatever
  the sockets say, so a wired box could hold back the SOCKETS alone: the
  old uniforms' sockets under a render already running the new program,
  its new uniforms with no socket to set them and a renamed output with
  none to carry it -- the stale state the entry above was written to
  end, now as a mode. (3) *It would be saved.* `needs_rebuild` is
  written to the .blend (seen: a node saved flagged comes back flagged),
  so an unticked, flagged node would open under `-b` with nobody to
  press Compile. What changed: the property, its sidebar line and the
  separator under it, its `PROP_TIPS` entry and its `NOT_SHADING` reason
  are gone; no rebuild road is touched and the callbacks still only
  flag. *Files saved with the property still load.* Run in Blender 5.2.2
  (the `bpy` module, background): the 1.92.0 add-on saved a file with
  the box unticked on one Coded Shader, ticked on a second and unticked
  on a third saved flagged (Language switched, the old sockets); a fresh
  process with the property gone opened it -- `open_mainfile` FINISHED,
  the three nodes whole, `hasattr(node, 'auto_compile')` False,
  `compile_source` clean, and the stored 0 / 1 still among the node's
  system ID properties (`bl_system_properties_get()`, not the
  custom-property keys), where nothing reads it and a re-save carries it
  along. The flagged node arrived rebuilt by `load_post`. A later
  property of that name would inherit the stored values, so the name
  stays retired. *The Inline Source tooltip.* `PROP_TIPS` called
  `source_text` "the name of the Text datablock holding this shader's
  source"; it is the source itself -- the starter shader `init` writes,
  and what `get_source()` and `export._compile_node` compile while no
  Text datablock is assigned. Reworded to say so (the registered RNA
  description read back from the same Blender). Pinned in
  `test_code_node_props_do_what_their_tooltips_say` (22 checks): the
  real node class, subclassed so the stored value answers False and
  every look at it is written down, goes through the callbacks' timer,
  an undo landing, File > Open, a background load, the export of a
  flagged node and the Compile operator -- every road rebuilds and not
  one asks; the sidebar draws no property; no other Python file names
  the retired property; every Coded Shader key in `PROP_TIPS` names a
  live property; `source_text` compiles as the source with no Text
  assigned and yields to an assigned Text, in the editor and at export.
  Ten mutations each fail it: a gate on the value in `_run_rebuilds`,
  `_pending_code_nodes`, `_source_changed` or `export._compile_node`;
  the declaration, the draw line, the tooltip or the census reason
  restored; the old Inline Source wording; `get_source` ignoring
  `source_text`. The two coded-shader tests now share one bench
  (`_code_bench`).
- **The Coded Shader compiles one source on every road (a node with a
  Text assigned drew the starter shader on the GPU).** A Coded Shader
  keeps its shader in one of two places: the Text datablock assigned to
  it (`source`) or the inline `source_text`, which `init` fills with the
  starter shader and which assigning a Text -- the node's own "New
  Shader Text" button does it -- never clears. Three readers answered
  "which one" three ways. `get_source()` (the editor compile, the
  sockets) took the Text whenever one was assigned.
  `export._compile_node` (the program in `programs[node.name]`, what
  `n_halcyon_code` runs wherever the CPU shades) took the Text but
  reached past an empty one for the inline source. The GPU emitter
  resolved nothing: `e_code_node` inlines `props.source_text`, which
  `export._node_props` copied verbatim off the node (`NODE_PROPS` listed
  it), "or" a `__source` prop that nothing in the tree ever wrote. So a
  GLSL node with a Text assigned ran the Text on the CPU and whatever
  its inline source last held in the deferred pass. *Wrong picture or
  fallback: the wrong picture.* The probe put the real node class
  (stand-in sockets) through the real `serialize_tree` and rendered
  `R.render` against `gpu/shade.simulate` at 128x96. With a Text of
  other uniforms the plan QUALIFIED and the frame was the starter at its
  declared defaults -- the CPU's own frame of the starter to four
  decimals, 1.43 off the CPU's frame of the Text over all 598 pixels of
  the material. With the starter's uniforms kept and the body rewritten
  (the "+" button, then an edit) it qualified and drew the starter under
  the sockets' values (1.58); with the out variable renamed it qualified
  and drew zeros (2.02). Only a Text reusing a starter uniform's NAME
  under another type refused ("socket 'Base Color' is VALUE but the
  uniform is vec3") and fell back to the CPU. The same in a real Blender
  5.2.2 (the `bpy` module, background; the registered node,
  `export_material`): a fresh node twinned at 1e-5, and the same node
  with a Text assigned and edited qualified 1.46 off over the 598
  pixels. (1) *One rule.* `shaders/compiler.node_source(node)`: the
  assigned Text, whatever it holds, else the inline source.
  `compile_node(node)` compiles it and returns (program, error, source).
  `get_source` is `node_source`; `compile_source` and
  `export._compile_node` are each that one `compile_node` call; no other
  file reads a Text's contents. (2) *The export ships what it compiled.*
  `_node_props` writes the call's source under `props.source_text` and
  is the key's only writer -- `source_text` left `NODE_PROPS`, the table
  of properties copied verbatim, and the census's `NOT_SHADING` says why
  -- so the GLSL the emitter inlines is the string the program was
  compiled from. Riding the graph also puts the Text in the GPU plan's
  signature (`gpu/shade._mat_sig` hashes the graph, not the programs):
  an edit in the Text Editor between two frames re-plans instead of
  hitting the cache. (3) *The emitter reads one key and holds it to the
  program.* `e_code_node` reads `source_text` alone (`__source` is
  gone), and a graph source that is not `programs[id].source` refuses by
  name -- "a coded shader whose graph source is not the source its
  program was compiled from" -- so a hand-built graph or a later
  regression falls back to the CPU instead of drawing another shader.
  (4) *The empty Text: the editor's rule, on both.* In Blender 5.2.2 a
  new or cleared Text reads `''` and one holding a blank line reads
  `'\n'`. The editor compiled nothing for either and said nothing; the
  exporter ran the inline source under `''` (the starter at its declared
  defaults, beneath the sockets the last Text left, on both devices) and
  nothing under `'\n'`. An assigned Text is now the source whatever it
  holds, for three reasons: the inline source has no widget, so the
  fallback rendered a shader the user could neither see nor edit while
  the Text in front of them was blank; the fallback was itself two
  rules; and the Inline Source tooltip already said it ("while the node
  has no Text datablock assigned"). An emptied Text compiles nothing,
  reads zeros on both roads and says so by name -- "the assigned Text
  'x' is empty -- nothing to compile, the outputs read zeros" in the
  node's error box, from the Compile button and as the export's warning
  in the F12 report; a Text that cannot be read is treated the same. No
  Text and a blank inline source stays silent, as before. *One picture
  changes*: a file whose Coded Shader holds an EMPTY Text rendered the
  inline source and now renders zeros, with that warning. (5) *The Text
  tooltip.* It promised "edit it in the Text Editor and the node
  recompiles on every edit"; typing in the Text fires nothing on the
  node (seen again in 5.2.2: `needs_rebuild` down, the sockets as they
  were, the next export compiling the new source). It now states the
  rule, and that the sockets follow on Compile. Pinned in
  `test_code_node_one_source_on_every_road` (33 checks): the real node
  class with a Text assigned and the starter still in `source_text`,
  through the real `serialize_tree` (the stand-in sockets gained `type`
  and `is_linked`; a master stands between the node and the output) --
  the editor's source, the program's and the graph's are one string; the
  simulator's frame is the CPU's (under 6e-3, no pixel over 1e-2) for
  the Text, for an edit made between two frames with the plan cache
  kept, for the Text that used to refuse, for the emptied Text and for
  the inline source; the 1.92.0 graph is refused by name; the three
  kinds of empty Text and the unreadable one on every road; and the rule
  read from the code (one resolver, one reader of a Text, one emitter
  key). Fourteen mutations each fail it: the export shipping the raw
  inline property, shipping nothing, or resolving for itself again;
  `source_text` back in the verbatim table; `node_source` falling back
  under an empty or an unreadable Text, or turned upside down;
  `get_source` or `compile_source` resolving alone; the emitter's second
  key restored, its hold on the program removed, or the emitter inlining
  the program's source while the graph keeps the inline copy (the cached
  plan then serves the old GLSL after an edit); the emptied Text made
  silent; the old tooltip. *Seen from the same probe*: a colour wired
  STRAIGHT into the Material Output's Surface (a plain RGB node does it,
  no coded shader needed) qualified for the deferred pass and the
  simulator's frame was 1.69 off the CPU's over the material, identical
  on the tree before this fix -- closed since by its own entry below ("A
  colour wired straight into a shader socket is an Emission on both
  roads"); and `serialize_tree` handed one `programs` dict down its
  node-group recursion keyed by `node.name`, which is unique per tree
  only, so a Coded Shader in the material and one inside a group that
  shared a name (the default one) left ONE program, the last compiled
  (seen at the export) -- closed since by the entry below ("Every Coded
  Shader runs its own program").
- **Every Coded Shader runs its own program: the graph id is the node's
  path (two that shared a name across trees shared one program).** A
  material's compiled Coded Shader programs live in one dict,
  `programs`, which `export.serialize_tree` hands down its node-group
  recursion. `_node_props` filed each program under `node.name`, and
  `n_halcyon_code` looks its program up by the node's graph id, which
  was that same bare name. Blender keeps node names unique per TREE, and
  a material is several trees: its own and one per group node. So the
  material's "Coded Shader" and a group's "Coded Shader" -- the default
  name, twice; nobody has to rename anything to get here -- left ONE
  program, the last compiled, and the other node ran it under its own
  sockets wherever the CPU shades. The entry above saw this at the
  export and left it for its own fix. *Wrong picture or fallback: the
  wrong picture.* The probe put the real node class (stand-in sockets)
  through the real `serialize_tree` and rendered `R.render` at 128x96,
  with two shaders that differ in their BODY: every uniform has a
  socket, so two sources that differ in a declared default alone draw
  one picture under either program -- the probe's first pair did, and
  read 0.0000. With the group node after the material's node in the
  tree, the material's node drew the group's shader: 1.48 off the frame
  of the same trees with the inner node renamed, over all 4848 pixels of
  the floor. With the group node first it was the group's node that drew
  the material's shader (2.17). The GPU device hid none of it: the
  deferred pass refuses a graph source that is not its program's own
  (the entry above) and a node group has no emitter at all, so the frame
  fell to the CPU and its wrong program. The same in a real Blender
  5.2.2 (the `bpy` module, background; a real node group, the registered
  node, `export_material`): Blender itself named both nodes 'Coded
  Shader', the export kept one program, and the frame was 1.52 off.
  (1) *The id is the path.* `export.node_id(name, path)`: a node of the
  material's own tree keeps its bare name (every
  `graph['nodes'][node.name]` still holds); a node inside the group node
  'Group' is 'Group/Coded Shader', inside a second instance of the same
  group 'Group.001/Coded Shader', two levels down 'Outer/Group/Coded
  Shader'. Each step is escaped (`/` and `\`), so no two paths spell one
  id whatever the names hold -- a node NAMED 'Group/Coded Shader' beside
  that group included. `serialize_tree` writes that one id as the node
  key, the entry's `id`, every link and `group_output`, hands the id
  plus the separator to the recursion as its path, and hands the id to
  `_node_props`, whose `programs[key]` is the only line that files a
  program. Nothing changed in the evaluator: `n_halcyon_code`, and the
  sub-evaluator `n_group` builds over the same `programs`, already
  looked up by graph id. The compile warning names the node by the same
  id ("Group/Coded Shader: ..."), so two same-named nodes that fail can
  be told apart. (2) *The Bump pre-pass, the other thing filed by node
  id.* `core/render._bump_height_fields` files a material's whole-frame
  gradient fields under (material, node id) for the Bump nodes of the
  material's own tree, and `n_bump` gathers by (material, node id)
  wherever it sits. A group's "Bump" beside the material's "Bump" took
  the OTHER node's field on any frame whose pre-pass runs (a material
  covering more than one shading batch, 16384 pixels): 1.00 off over
  29164 pixels at 320x240, nothing at 128x96 where no pre-pass runs. The
  path id ends it with no line changed in the renderer: the group's Bump
  is 'Group/Bump', finds no field and differences its own batch, as a
  group's Bump under any other name always did. (3) *The GPU's key: one
  mangling tag per node.* `e_code_node` inlines each coded shader under
  `_cn<tag>_<name>`, and the tag was the id with every non-word
  character flattened to '_', which is many-to-one. 'Coded Shader' and
  'Coded_Shader' (or 'Coded.Shader') in ONE tree are two programs on the
  CPU and one set of GLSL names: the plan QUALIFIED, one shader's text
  was renamed onto the other's, and the simulator's frame was 0.80 off
  the CPU's over the 4848 pixels. Two tags also meet without being
  equal: 'Coded' declaring `B_k` and 'Coded B' declaring `k` are both
  `_cnCoded_B_k` (0.57 off). `gpu/emit._code_tag(em, key)` hands out one
  tag per node id per emitter: the flattened id while it meets no tag
  already handed out, else the same with an ordinal in front
  ('2_Coded_Shader'), counted up until nothing begins that way.
  `_tags_clash` is the test: equal, or one is the other plus '_' plus a
  rest that can begin an identifier. A rest that begins with a digit
  cannot, which is why Blender's own 'Coded Shader' / 'Coded Shader.001'
  never met and keep their tags; a node alone in its pass keeps exactly
  the names it had (`_cncode_main` and the other spellings `test_render`
  pins are untouched). The emitter's once-set, which the node id shared
  with the emitters' own helper markers ('__wang', ...), takes the node
  under a marker of its own, so a node named like one of them is still
  inlined. A coded shader INSIDE a group is not emitted:
  groups have no emitter, the material is refused by name ("no GLSL
  emitter for ShaderNodeGroup") and the CPU -- right, now -- draws it.
  *Pictures that change*: a material whose trees hold same-named Coded
  Shaders (or same-named Bump nodes, on a pre-passed frame) drew one
  node with another's program or field, and now draws each with its
  own; two coded shaders in one tree whose names flatten alike drew one
  shader twice on the GPU device and now twin the CPU. Nothing else
  moves: the ids of a material's own tree are the names they were (a
  name holding `/` or `\` gains its escape, in the node key and its
  links together). Pinned in `test_code_node_same_name_across_trees` (46
  checks), through the real `serialize_tree`: the two programs under
  their paths, each compiled from its own node's graph source, and the
  group's tree written under the same ids; in BOTH node orders the
  material's node and the group's node each drawing the frame of its
  shader alone, bit for bit, and their sum the frame of the same trees
  with the inner node renamed; one group instanced twice and a group in
  a group (four nodes named alike, four programs, the frame of four
  different names); the 1.92.0 export of each rebuilt by hand and shown
  to be another picture, so the frames can tell; a name holding the
  separator; 2379 paths of awkward names spelling 2379 ids; the warning
  by path; on the GPU the material's node qualifying and twinning beside
  a group that holds its namesake (under 6e-3, no pixel over 1e-2), the
  1.92.0 export refused there, a group-fed master refused by name with
  the right CPU frame; the two tag clashes twinning on the frame the
  same shaders draw under unrelated names; the tags themselves, and
  `_tags_clash` against brute force over all 741 pairs of short tags
  ("can these two spell one mangled name"); the two Bump nodes on a
  320x240 frame whose pre-pass is seen to run; and the rule read from
  the code (one reader of a node's name, one line that files a program).
  Nineteen mutations each fail it: the program filed by name; the node
  keys, the links or the recursion's path left bare; the path built from
  the group node's name instead of its id (a nested path loses its
  parents); `node_id` escaping nothing, the separator alone or the
  escape alone; the warning by bare name; `_node_props` ignoring its
  key; the emitter's old tag; `_code_tag` without its check or its
  memory; `_tags_clash` reduced to equality, without the digit
  exception, without the swap, or without the joining underscore; the
  once-set taking the bare id; the group evaluator handed an empty
  `programs`. *Seen from the same probes
  and left for their own fixes*: a Coded Shader whose node name holds a
  non-ASCII letter puts that letter into its GLSL names (`\W` keeps
  every Unicode letter) and the plan's pre-flight refuses the material
  -- "would not compile: unexpected character" -- so it shades on the
  CPU, correctly, but never on the GPU; the shader compiler accepts an
  undeclared identifier (`vec4(nothing, ...)` compiles with no error and
  no warning and raises "unknown identifier" only when it runs: the node
  reads zeros and the export warns of nothing); and the Bump pre-pass
  still covers the material's own tree only, so a group's Bump is cut at
  shading-batch boundaries on a frame whose pre-pass runs (the same
  chain in the material's own tree and in a group: 0.35 apart along two
  rows at 320x240, nothing at 128x96).
- **A Coded Shader named in any script reaches the GPU: a letter GLSL
  cannot hold is spelled as its code point (a node called 'Señal' kept
  its whole frame on the CPU).** `e_code_node` inlines each coded shader
  under `_cn<tag>_<name>`, and the tag is the node's graph id with every
  non-word character flattened to `_` (the entry above). "Word" was
  Python's, not GLSL's: `\W` is Unicode-aware, so the 'ñ' of a node
  named 'Señal', a Cyrillic or a CJK name, the letters and digits of
  every script stayed in the tag and went into the pass as written --
  `_cnSeñal_main`. A GLSL identifier is `[A-Za-z0-9_]` and nothing else.
  The entry above saw this from its probes and left it for its own fix.
  *Wrong picture or fallback: the fallback.* The plan QUALIFIED the
  material; the device's pre-flight (`gpu/device.preflight`: Halcyon's
  own GLSL front-end, on the render worker, before the driver is handed
  anything) refused the text -- "would not compile the pass (line 1209:
  unexpected character 'ñ'); refused before the driver" -- and a refused
  pass sends its whole frame to the CPU. So the picture was right, and a
  scene holding one such node never shaded on the GPU, for nothing but a
  name any keyboard outside English types. The probe put the real node
  class (stand-in sockets) through the real `serialize_tree`: 'Señal'
  alone feeding the master, and 'Señal' beside a second coded shader,
  'Other', the two summed into the master's Diffuse Color. Both were
  refused, Cyrillic and CJK names the same, and glslang refuses the same
  text ("unexpected token") where the simulator does. The same in a real
  Blender 5.2.2 (the `bpy` module, background; the registered node,
  `export_material`): Blender keeps each name as it was typed, and all
  four materials -- 'Señal' alone, beside 'Other', 'Шейдер' beside '色彩',
  '色' beside '彩' -- were refused with that line. (1) *The spelling.*
  `gpu/emit._code_spell(key)` is the flattening as it was, then every
  character still outside ASCII -- by then letters and digits only --
  written as its code point: `u` and four or more upper-case hex digits.
  'Señal' is 'Seu00F1al', 'Шейдер' is 'u0428u0435u0439u0434u0435u0440',
  '色彩' is 'u8272u5F69', a full-width '３' is 'uFF13', a letter beyond the
  BMP 'u1D6FC'. `_code_tag` asks it for the tag and keeps its clash
  check; nothing else spells one. (2) *Why the code point and not one
  more underscore.* Flattening to ASCII ('ñ' to `_`) was tried first,
  and it qualifies and twins as well. It was not shipped: a name of such
  letters alone becomes underscores alone -- '色' and '彩' are both `_`,
  one tag between them, and beside each other they are `_cn__main` and
  `_cn2___main` -- and GLSL reserves every identifier that holds `__`.
  glslang 15.1 takes those with a warning (`identifiers containing
  consecutive underscores ("__") are reserved`) as desktop 4.30 and
  under Vulkan rules alike, and refuses them below ES 3.00; no driver
  was asked. A code point is letters and digits, so it adds no
  underscore at all and can neither make a run nor lengthen one. That is
  also why it carries no delimiter: `_u00F1_` makes a run between two
  such letters, a leading `_` makes one after any punctuation, a
  trailing one against the `_` that joins tag and name. (3) *Still one
  tag per node.* The spelling is many-to-one, as the flattening is: a
  node NAMED 'Seu00F1al' beside 'Señal' spells the same tag, and 'Señal
  B' is 'Señal' plus `_` plus the start of a name (`B_k` under the one
  and `k` under the other are both `_cnSeu00F1al_B_k`). `_code_tag`
  settles both as it settles 'Coded Shader' beside `Coded_Shader` -- the
  ordinal in front, `2_Seu00F1al` -- with no line of `_tags_clash`
  changed: every tag is `[A-Za-z0-9_]` now, the alphabet its brute-force
  check is over. A code point begins with a letter, so '色 彩' beside '色'
  takes the ordinal (rightly: `u5F69_k` in the one is `k` in the other)
  and 'Señal.001' beside 'Señal' does not, as Blender's own suffix never
  did. (4) *What does not move.* A tag that was plain ASCII is the tag
  it was, to the letter: the first substitution is the old line, and the
  second touches only what no driver could read. That is every name the
  pre-flight ever let through, punctuation in other scripts included
  ('A—B' is still `A_B`), and with them `_cncode_main`,
  `_cncode_baseColor`, `_cncode2_baseColor`, `_cncode_pattern` and the
  tags the same-name test pins. It also means the double underscores
  some ASCII names already make (`Group\/Coded Shader` is
  `Group__Coded_Shader`, and a name with two spaces in a row makes one
  too) are still made: pinned, on the footing they were, and not this
  entry's to move. (5) *The strict gate.* `tests/glsl_strict.py` (the
  tool, glslang) compiled no coded shader at all. `check_coded` compiles
  the pass of eight pairs of names through the real exporter -- plain
  ones, the two kinds of clash, 'Señal' beside 'Other' and beside a node
  named 'Seu00F1al', Cyrillic beside CJK, one letter each, a full-width
  digit beside a letter beyond the BMP -- as GLSL 4.30: eight of eight,
  where the old flattening fails five by name. A refusal counts as a
  failure there, because each of those names is meant to reach the
  driver. `compile_source` now writes UTF-8 and decodes glslang's answer
  leniently: glslang quotes ONE byte of the offending letter, and
  decoding that strictly raised out of the check instead of failing the
  shader by name. *Pictures that change*: none on the CPU. A frame
  holding such a node shaded on the CPU and now shades on the GPU
  wherever the GPU device is selected: 2.0e-5 off the CPU frame on the
  simulator, the number the ASCII-named reference twins at, and both
  frames bit for bit the frames of the same tree under ASCII names. In
  the real Blender 5.2.2 the four materials above read the same: 2.0e-5,
  bit for bit the reference's frames, the pre-flight silent. Pinned in
  `test_code_node_non_ascii_name_reaches_the_gpu` (33 checks), through
  the real `serialize_tree`: 'Señal' alone and beside 'Other' qualifying
  and twinning (under 6e-3, no pixel over 1e-2), the CPU's frame and the
  simulator's each bit for bit that of the same tree under ASCII names;
  Cyrillic beside CJK, '色' beside '彩' and 'Señal' beside a node named
  'Seu00F1al' the same; the `B_k` / `k` pair under 'Señal' / 'Señal B'
  on the frame of unrelated names; the pass text plain ASCII, its six
  mangled names exact, each an identifier and none holding `__`, the
  device's sampler and uniform names with them, and the pre-flight
  letting it through; the 1.92.0 flattening put back -- the plan
  qualifying a text that is not ASCII, the pre-flight and the simulator
  each refusing it by name, the CPU frame right all along -- so the
  checks can tell; the spellings (a full-width digit, an Arabic-Indic
  one, a superscript, a letter beyond the BMP); the pinned tags where
  they were; over 16104 names up to four characters, all 7380 whose
  flattening was plain ASCII unmoved, every tag in the alphabet, and the
  runs of underscores the flattening's own, one for one; 399 names of
  letters and digits alone spelling 399 tags without an underscore; the
  clashing tags by name, one per node, none able to spell another's;
  every name the port hands on under such a tag, a sampler's included;
  and the rule read from the code (`_code_tag` spells through
  `_code_spell` alone). Seventeen mutations each fail it: the old
  flattening, in `_code_spell` or left behind in `_code_tag`; the
  ASCII-only flatten; the code point between underscores, with a leading
  one or with a trailing one; unpadded, lower-case or without its
  letter; cut to 16 bits; one fixed letter for every such character;
  every non-ASCII character spelled, punctuation too (tags the
  pre-flight let through move); Latin-1 alone, the BMP alone, or letters
  without digits; `_code_tag` without its clash check or without its
  memory. *Seen from the same probes and left for their own fixes*: with
  shadows on, the RAW text of a material pass declares `uniform
  sampler2D hal_shadowpack;` once per shadowed lamp (beside each
  `hal_shadow_vis<i>`), and with two such lamps -- the demo scene's --
  glslang refuses it as a redefinition (declared once it compiles;
  `strip_declarations` takes every such line off on the CreateInfo road,
  so only the legacy constructor would hand a driver the raw text) --
  the strict gate plans every scene with shadows off and has never seen
  it, which is why `check_coded` plans that way too; and a comment in
  another script inside the user's own GLSL travels in the pass as
  written (the front-end and glslang both skip it; no driver was asked).
- **A Bump inside a node group gathers from a whole-material field (its
  gradients were cut at every shading-batch boundary).** `n_bump`
  differences its height toward the +x / +y neighbours shaded in the
  SAME batch, so a material that covers more than one shading batch
  (`MIN_CHUNK`, 16384 pixels) has its height chains rendered to
  whole-material gradient fields first, and every batch gathers from
  those. Both sites that build the fields --
  `core/render._bump_height_fields` for the opaque frame, the per-rank
  twin inside `_shade_fragments_cpu` for the layers -- walked
  `graph['nodes']`, the material's OWN tree. Since the entry above, a
  group's Bump has its path for an id ('Group/Bump'): it found no field
  and went back to differencing its own batch, which is what a group's
  Bump under any name had always done. The entry above saw it and left
  it for its own fix. *Wrong picture or fallback: the wrong picture.*
  The probe (the demo floor through the real `serialize_tree`,
  `R.render` on the CPU) put ONE chain -- Gradient Texture (Quadratic
  Sphere) -> Math Multiply 6.0 -> Bump -> the master's Normal -- once in
  the material's own tree and once inside a group whose output feeds the
  master's Normal. The two frames are identical at 128x96, where no
  pre-pass runs. At 320x240 they are 0.35 apart over 230 pixels on rows
  53-54, the batch boundary; at 640x480, 0.17 apart over 1472 pixels on
  six rows with one thread and 0.31 over 3172 pixels on fourteen rows
  with four, because the batch size follows the thread count: the
  group's bump was a function of the machine. The top-level frame is the
  reference, and not by convention: it is, bit for bit, the frame of the
  material shaded in ONE batch (the batch size raised past the floor: no
  pre-pass at all). Three more roads had the same hole. *See-through*:
  the layered road shades rank by rank from per-rank fields, and found
  its Bump materials in the top tree too, so the group's material took
  the old single call -- every rank in one mixed array, cut per chunk
  (0.21 off). *Bands*: `job.bump_field_source` hands the builder the
  G-buffer's FULL coverage so that a band's last row keeps its +y
  neighbour, and the group's Bump had no field to read it from: 0.53 off
  along each band's last row (three bands of a 320x240 frame), and at
  any frame size, since a banded frame always pre-passes (0.54 at
  128x96). *The GPU device*: a node group has no emitter, the material
  is refused by name and the frame falls to the same CPU road, cut
  included. (1) *One walk.* `core/nodeeval.bump_heights(ev)` hands out
  (node id, heights) for every Bump node below an evaluator: its own
  tree's, then each group node's, to any depth. Both pre-pass sites take
  their heights from it, and `nodeeval.has_bump(graph)` is the any-depth
  test both ask (`_bump_materials_of` for the layered road, which is
  what moves a group-Bump material onto the rank-by-rank road at all).
  (2) *The group's context.* A group-internal chain is evaluated by the
  evaluator of its group's tree with the group node's inputs bound:
  `nodeeval.group_evaluator(ev, node)`, factored out of `n_group`, which
  is now its first caller. The shading pass and the pre-pass bind
  through ONE function, taken step by step down the path, so a height
  that reads a Group Input reads what the shading pass will read -- two
  levels down included. The pre-pass asks for the binding lazily: the
  group's inputs are evaluated when the inner tree first reads a Group
  Input socket, so a chain that reads none does not pay for whatever
  feeds the group. `n_group` binds up front exactly as before; nothing
  moves in the shading pass. (3) *Filed under the path.* The key is
  (material index, node id) as it was, and the id is the node's path, so
  `n_bump`'s lookup is the line it was and two instances of ONE group --
  two bindings, two heights -- keep two fields ('Group/Bump',
  'Group.001/Bump'). An id is handed out once, a tree's own nodes before
  its groups': a hand-built graph that repeats an id (the 1.92.0 export,
  rebuilt for the entry above's test) leaves the material's own node the
  field, as it always did. (4) *One context per chunk.*
  `_bump_height_fields` built a shading context and an evaluator per
  Bump node per chunk; it builds one per chunk, every Bump of the
  material shares it and its cache, and the coverage mask -- one mask,
  whichever node -- is scattered once. The arithmetic of a field is the
  lines it was: every top-level frame of the probes (one Bump and two
  chained; opaque, see-through and banded; 128x96, 320x240 and 640x480)
  is bit for bit the frame the tree drew before. *Pictures that change*:
  a material whose Bump node sits inside a node group, where that
  material covers more than 16384 pixels of the frame, on a banded
  frame, or see-through. Its bump is the whole-material one now -- the
  frame the same nodes draw outside the group. Nothing else moves.
  Pinned in `test_group_bump_gathers_from_a_whole_material_field` (20
  checks), through the real `serialize_tree`, with a spy on the pre-pass
  so every frame says whether it ran and for which ids: the chain at the
  top level building 'Bump' and inside a group building 'Group/Bump' on
  a 320x240 frame, the two frames bit for bit, and both the one-batch
  frame; the group's field withheld (the 1.92.0 pre-pass) cutting a row
  or two by more than 0.1, so the frames can tell; 128x96 (no pre-pass,
  the two agree as they always did) and 640x480; a chain that reads its
  group's Scale and Normal inputs against the same chain fed at the top
  level, and a chain that reads none leaving the Value at the group's
  Scale unevaluated in the pre-pass; ONE group tree instanced twice at
  two scales (two fields, one three times the other, and the frame of
  two chains at the top level); a group in a group ('Outer/Inner/Bump');
  the see-through floor on the layered road, with the fields each rank
  gathers from and with the layer's heights in two chunks; three bands
  joined against the whole frame; the GPU device on the stand-in driver
  (refused by name, the CPU's pre-passed frame bit for bit); and the
  rule read from the code (one walk, one function that binds a group).
  The entry above's test keeps its two Bumps and gains the group's field
  ('Bump' and 'Group/Bump' built side by side; under the 1.92.0 ids one
  field, the material's own). Eighteen mutations each fail the two
  tests: the walk not entering groups, entering one level only, or
  entering them before the tree's own Bumps; the group's tree evaluated
  with nothing bound, with a binding that never binds, or bound in the
  walk's root evaluator instead of its parent's; the pre-pass binding
  eagerly; an id handed out once per NAME, or again each time it is met;
  `has_bump` looking at the own tree only; the field filed, or asked
  for, under the bare name; one height image for every node, or the
  first node's height alone; the coverage mask of the first chunk only;
  the layered road finding top-level Bump materials only, keeping the
  own tree's heights only, or writing a chunk's heights at the front of
  the layer. *Seen from the same probes and left for their own fixes*:
  the master's Bump Height socket is sugar that `desugar_master_bump`
  rewrites into a Bump node for the nodes of the material's own tree
  only, so a master INSIDE a group ignores its linked Bump Height
  altogether (0.89 off over 4435 pixels of the floor at 128x96 -- no
  pre-pass involved); and a Bump whose HEIGHT is fed by another Bump's
  normal is still cut where the pre-pass chunks, in the material's own
  tree too (0.84 off over 461 pixels on rows 52-54 at 320x240, the same
  before and after this entry): while the fields are being built the
  upstream Bump has none yet and differences the chunk.
- **A uniform the driver stripped never fails a pass (the llvmpipe field
  log).** A real Blender 5.2.2 session on the OpenGL backend (Mesa
  llvmpipe) printed, for every GPU frame, `rasterising on the CPU: raster
  dispatch failed: ValueError: GPUShader.uniform_float: uniform
  hal_rtiles_w not found`, `blended compositing fell back to per-pass
  readback: ValueError: GPUShader.uniform_float: uniform hal_cam_right not
  found` and `shading on the CPU: drawing 'Material.001' failed:
  GPUShader.uniform_float: uniform hal_cam_right not found` -- the whole
  session on the CPU road with the GPU device selected. One rule is behind
  all three: a linked GL program keeps only the uniforms that reach its
  output, Blender's `GPUShader.uniform_float` / `uniform_int` /
  `uniform_bool` raise `ValueError` for a name the program no longer has,
  and Halcyon set every uniform of a pass unconditionally -- so ONE
  stripped name failed the compute raster, then the blended burst, then
  the per-pass road behind it. (1) *One door.*
  `gpu/device.set_uniform(shader, kind, name, value)` is the only place
  under `gpu/` that calls a GPUShader's `uniform_sampler` / `uniform_bool`
  / `uniform_int` / `uniform_float`. The two sites that did call them now
  call it: `_draw_in_bound`, which every full-screen pass of every family
  crosses (material, layer, bump height, radiosity grid, sky, ink,
  resolve, the post chain), and `_dispatch_compute_impl`, which the
  compute raster and the two ray kernels cross; a scan over `gpu/*.py`
  pins that nothing else does. The `ValueError` whose text says the name
  was *not found* is a skip, and the name is remembered per shader (a
  program's active set never changes after the link), so the driver is
  asked ONCE and every later draw of that shader skips the set without
  raising at all. Every other error -- a `ValueError` about the value, a
  `TypeError`, anything the driver raises -- propagates exactly as before,
  the draw's `finally` still restores blend and scissor, and a kernel's
  output IMAGE is never skipped (a kernel without its output has nothing
  to read back). `engine.py`'s viewport blit sets `image` on Blender's
  built-in IMAGE shader, which is not a Halcyon pass, and stays a plain
  call. (2) *The raster's dead pair is gone.* `hal_rtiles_w` /
  `hal_rtiles_h` were declared by the compute shell and read by nothing in
  any of its eight variants -- the tiles fetch has been a `texelFetch`
  with integer coordinates since the exact-fetch round -- so they left the
  text, the CreateInfo and the dispatch together: `craster.RASTER_FLOATS`
  is the one list both use, 15 push constants (60 bytes), each read by
  every variant. (3) *The camera axes are skipped, not pruned.*
  `hal_cam_right` / `hal_cam_up` / `hal_cam_back` are read only by a key
  fixed to the camera (`cel_light` 1: the cel key block and the Anime
  shine's follow); every other material and layer pass, every bump height
  pre-pass and the radiosity grid pass carry them unread. They ride the
  frame's ONE uniform dict and the spec of every pass family (seven sites
  in `gpu/shade.py`), which is what keeps the source, the CreateInfo and
  the dict in step; pruning them per variant would be a hand-kept copy of
  what the linker already computed, so the door takes the skip -- three
  names per material shader, each asked once. (4) *The Debug breakdown
  says so.* With Timing Breakdown on, an F12 prints under its table
  `[Halcyon GPU] N uniforms stripped by the driver in K shaders (names):
  ...`, counted once per shader so the number is stable from the second
  frame on; the self test's frame breakdown carries the same line, and
  `device.reset()` forgets it with the rest of the driver world. Pinned in
  `tests/test_r253_uniforms.py` (10 tests, 72 checks, registered in
  `run_all`): the door's rule and its loud errors; the real
  `draw_fullscreen`, `draw_many` and `dispatch_compute` over a shader that
  refuses one name; `raster_on_device` sending exactly what its CreateInfo
  declares; and the demo frame on `tests/fakedevice.installed_gl()` -- the
  REAL device draw functions over a stand-in driver that strips every push
  constant a pass never reads and raises Blender's own `ValueError` for it
  -- which shades on the GPU road bit for bit the frame of a driver that
  strips nothing, and with the old unconditional set prints the field's
  two shading lines word for word. A one-off scan put all 121 shipped
  presets through that driver (the demo frame, the render and the GPU post
  chain): every one is bitwise the plain fake's, across 775 compiled
  shaders in 41 pass families, and the only names stripped are the three
  camera axes, in each of the 356 material shaders. The strict gate
  (`tests/glsl_strict.py`, glslang 15.1.0) now compiles the compute raster
  too, as a 4.30 compute shader in all eight variants under the header the
  CreateInfo writes: 623 sources compile, 15 are refused by name before
  the driver, none is rejected.
- **Two handler-stub leftovers in the suite, closed.** (1)
  `test_mesh_export_cache` proved "without bpy handlers the cache never
  arms" by leaning on a stub that had no `bpy.app.handlers` at all; the
  stub has modelled them since the persistent-handler fix, so
  `export.register()` armed and the check failed in every run. The test
  takes the handlers away for that one call and restores them. (2)
  `test_append_watch` installed a one-list `bpy.app.handlers` and never
  returned it, so every module after LEGACY in `run_all` ran against a
  namespace with no `persistent` and no undo / load lists (the Coded
  Shader test above raised on it in the full run); the stub is lent
  inside a try / finally and restored, and the test checks the hand-back.
- **A colour wired straight into a shader socket is an Emission on both
  roads (the deferred pass qualified and drew it as the albedo).** A
  Material Output whose Surface socket is fed by a COLOUR output -- no
  shader node in between, the quick way to look at a texture -- drew two
  pictures of one graph. *Which road was right: neither.* The CPU read
  the socket through `nodeeval.coerce(v, SHADER)`, which hands anything
  that is not a closure an EMPTY one, so `closure_to_surface` shaded the
  material's fallback surface (the viewport colour under the default
  model) exactly as if nothing were wired. The deferred probe saw that
  same empty closure -- no lobe to refuse -- and
  `gpu/material.assemble_frame` casts whatever feeds Surface into the
  pass's one colour chain, the albedo. Reproduced headless before the
  fix, on the demo Ball at 128x96 (`R.render` against
  `gpu/shade.simulate`): the plan QUALIFIED and the simulator frame was
  1.69 off the CPU's, more than 1e-2 off on all 598 pixels of the
  material (CPU mean (1.17, 1.17, 1.20), which is the frame of an
  unwired socket; simulator (1.31, 0.45, 0.17)). A value read 0.96 off
  and a vector 3.38; a Coded Shader's Color 1.68 (the test's striped
  shader; the report's own read 1.50) -- the wiring, not the node type;
  and the same split sat on the Mix Shader's (0.51) and the Add
  Shader's (2.04) shader inputs. Blender's own rule is neither
  picture: `ntree_shader_implicit_closure_cast` puts an Emission node on
  every link from a non-shader socket into a shader socket before EEVEE
  compiles the tree, and Cycles' `ShaderGraph::connect` adds the same
  node as its automatic conversion (colour one, Strength 1, a float into
  Strength). (1) *The CPU reads Blender's conversion.*
  `core/nodeeval.link_closure` is what a LINK hands a shader socket: a
  closure passes through, and a colour, a value or a vector becomes ONE
  EMISSION lobe -- `to_color` of the value, weight 1, Strength 1, the
  very lobe an Emission node hands on, so the frame is the explicit
  node's bit for bit. `GraphEvaluator.input` calls it for every linked
  SHADER read, which is every shader socket there is: the Material and
  World Output's Surface, the Mix and Add Shader inputs, the Max blends'
  sub-materials, a group's shader sockets, Shader to RGB. An output with
  no value (a renamed socket, a node that raised) and an UNLINKED shader
  socket stay the empty closure. (2) *The deferred plan refuses it by
  name.* As first fixed, every such link refused: the frame pass emitted
  one colour chain into the diffuse term and had no slot for a raw
  emission lobe (the Emission node's own refusal). The emission slot
  routing -- the next entry -- has since put the LONE link on the
  driver; what `gpu/shade._probe_material` still refuses is the link
  BESIDE another kind of lobe (the lobe is marked `implicit`), with
  `IMPLICIT_EMISSION_REFUSAL` as it reads now: "a colour, value or
  vector output is wired straight into a shader socket beside another
  kind of lobe: the link is an Emission of it (Blender's implicit
  conversion, the picture the CPU shades), and the frame pass emits ONE
  colour chain, which cannot feed the emission term and the diffuse term
  at once -- this material shades on the CPU exactly (alone, the link
  rides the emission slot on the driver; so does a Halcyon Shader with
  the chain in its Diffuse Color or Self-Illumination)". It refuses on
  the lobe's PRESENCE, never its weight: the probe reads sixteen
  fragments, and a Mix Shader's Fac mask can read 0 on every one of them
  while the emitter still mixes the link into the albedo elsewhere. (3)
  *This changes pictures.* On the CPU device such a material rendered
  its fallback surface and now renders the wired colour as emitted light
  alone, the Emission node's picture (no glossy lobe: specular level 0;
  no diffuse lobe: no diffuse, the no-diffuse entry below); on the demo
  frame every pixel off the material is bitwise what it was. On the GPU
  device it rendered the colour as a lit albedo and now draws the CPU's
  picture: with the reason printed once where the plan refuses, on the
  driver for the lone link (the next entry). A
  colour wired into the WORLD Output's Surface was ignored (the world
  fell through to its image, blend or flat colour) and is the sky now,
  bitwise the Background node at Strength 1; the GPU sky pass already
  refuses, by name, every world graph but a lone Background with
  constant inputs. Graphs without such a link are untouched: the
  conversion happens on that link and nowhere else. (4) *Considered and
  left.* Twinning it instead -- routing the chain to the pass's emission
  term, as the specular slot routing did for a lone GLOSSY lobe -- would
  lift the Emission node's refusal with it and is its own piece of work
  (the corner road lights emission in the corners; the layer and hit
  passes each need the route proven); it was queued here and is the
  next entry. And the frame both devices drew was Halcyon's Emission
  node's, not Cycles': with no DIFFUSE lobe in the closure
  `closure_to_surface` left the material's fallback diffuse LIT under
  the glow, where Blender shades black, and that rule read its batch
  (`np.any(diff_w > 1e-6)`: a pure-emission pixel had diffuse level 0
  when a diffuse pixel shared its batch and level 1 when it was alone).
  Recorded here as an open item, seen at the function alone; hunted in
  a frame and closed by the no-diffuse entry below. Pinned in
  `test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads`
  (48 checks, since the routing took the lone link's plan answer to its
  own test) for a colour, a value and a vector: the CPU frame bitwise
  the Emission node's; the wired colour arriving as exactly itself
  (frame(colour) - frame(black) within 1e-6 on every pixel of the Ball,
  and since the no-diffuse entry frame(black) is black: it sat over the
  lit fallback); and the road the refusal names -- a Halcyon Shader in
  between, either socket -- qualifying with the simulator frame the CPU
  frame at the twin bar (measured 4.8e-6; the bar is under 6e-3 with no
  pixel over 1e-2). Then the Mix and Add Shader inputs beside a Diffuse
  BSDF refusing word for word, at Fac 0 too; a Reroute on the way and a
  Coded Shader's output on the CPU; the unwired socket and a Diffuse
  BSDF in between still twinning; an Emission node beside a Diffuse BSDF
  keeping the raw-lobe refusal; the GPU device road on the fake driver
  returning the refused wiring's CPU frame bit for bit; the World
  Output. Ten mutations of the fix each fail it.
- **The emission slot routing: a surface that is light alone shades on
  the driver (the wired colour and the Emission node twin where the plan
  refused them).** What was asked: "Now do the emission slot routing so
  the GPU twins it" -- the item the entry above left queued. *What the
  pass could not do.* The frame pass emits ONE per-pixel colour chain and
  fed it to the diffuse term, or, for a lone GLOSSY lobe, to the specular
  colour (the specular slot routing). An emission lobe's colour belongs
  to neither, so every graph whose surface is light alone -- a colour,
  value or vector wired straight into Surface, an Emission node, a Mix or
  Add Shader of those -- refused the deferred plan by name and shaded on
  the CPU. *The route.* When EVERY lobe of the closure is an EMISSION,
  no lobe feeds the diffuse or the specular slot: `closure_to_surface`
  hands back constants there -- whatever it decides for a surface with
  no diffuse and no glossy lobe, the probe bakes that and restates
  nothing -- and puts the weighted sum of colour x Strength in
  `surf.emission`. That sum is the one chain the emitter already writes
  for those nodes
  (`e_emission`: colour x Strength; `e_mix_shader` and `e_add_shader`:
  the mix and the sum that `n_mix_shader` and `n_add_shader` hand on; a
  Reroute forwards). So `gpu/shade._probe_material` grants
  `bake['__slot'] = 'emission'`, exempts the emission from the constancy
  rule (its variation is the point) and bakes the flat diffuse, as the
  specular route does; and `gpu/material.assemble_frame` hands the chain
  to the pass's one `total = total + ...` line -- where `light_surface`
  adds `surf.emission`, after the light loop and the clamp, the line a
  master's linked Self-Illumination already rides -- and writes the
  diffuse and the specular colour as constants. No new GLSL. The
  shadeless early return (diffuse x level + emission), the layer passes
  and the ray-hit passes all read that line; a corner-rate pass never
  reaches it, because the CPU lights the corners WITH the emission in
  them and the pass multiplies by the constant diffuse, exactly the
  CPU's own albedo half. *Three rules keep it exact.* (1) STRUCTURAL:
  every lobe, whatever its weight. A Diffuse BSDF that a Mix Shader's
  Fac zeroes at the sixteen probed fragments is still a diffuse lobe
  elsewhere, so a Diffuse mixed with an Emission at Fac 1 stays refused.
  (2) PROVEN NODES ONLY: `gpu/material.emission_route_block` walks the
  Surface link's shader path by the evaluator's own cache (an output
  that evaluated to a closure is a shader node; anything else on a
  shader socket is the implicit link) and lets through Emission, Mix
  Shader, Add Shader and Reroute (`EMISSION_ROUTE_NODES`). The Max
  blends are stopped by name -- their chains drop a bare side OUT of the
  mix, the master lobes' normalisation, where an emission lobe is scaled
  by its weight -- with `emission_route_refusal`: "an emission-only
  closure reaches the Surface through HALCYON_MaxBlendNode, whose chain
  is not the emission the CPU sums (Emission, Mix Shader, Add Shader and
  Reroute are) -- this material shades on the CPU exactly". (3) A BARE
  SOCKET IS NO LIGHT: an unlinked Mix or Add Shader input is the empty
  closure, and under the route the emitter writes `vec4(0.0, 0.0, 0.0,
  1.0)` for it (`Emitter.bare_shader_zero`, armed by the assembler under
  this slot alone: in the diffuse slot a bare side must drop out of the
  mix, which zero is not). *Still refused, by name:* the implicit link
  BESIDE another kind of lobe (`IMPLICIT_EMISSION_REFUSAL`, reworded in
  the entry above) and a raw Emission lobe beside a diffuse or a glossy
  one (the raw-lobe text) -- one chain cannot be two slots. *Found on
  the way: both routes read the wrong alpha.* The decal combines
  (DS_DECAL, DECAL_ALPHA, N64_BLENDRGBA) take the TEXEL's alpha from the
  chain. A routed chain is not the albedo: the CPU's albedo pass hands
  those combines the constant diffuse and the surface's own opacity, 1.
  A lone Glossy BSDF whose colour carried alpha 0.3 -- the specular
  route, since it was ported -- qualified and drew 0.48 off the CPU over
  593 or more of the Ball's 598 pixels at the corner rates under those
  three models; the emission route would have been 0.37 off over all
  598. A routed chain's alpha is pinned at 1.0 on both routes now, and
  both read 0.0. *Measured* (the demo Ball at 128x96, `R.render` against
  `gpu/shade.simulate`): the lone colour was 1.69 off before the entry
  above, refused after it, and is 4.8e-7 off on the Ball now. The twin
  is not two roads agreeing on a missing emission: the SIMULATOR's
  frame(colour) - frame(black) is the colour to 1e-6 on every pixel of
  the Ball. A traced reflection of a routed floor moves the mirror Box's
  pixels by the same 0.45 on both roads (0.5 x 0.9; the frames 2e-5
  apart). Every engine model forced over a routed vector is on the twin
  bar, but for the Super FX plot (refused by its own palette rule) and
  WIREFRAME, whose wires are carved after the stage the simulator covers
  -- the same 6,172 pixels with and without the route. *A one-off sweep*
  put a routed material into all 313 feature-matrix rows on the fake
  driver (the real draw functions over `tests/fakedevice`), three times:
  a vector on material 1, a colour on the last material, a Coded
  Shader's stripes on the floor. The frame shaded on the driver in at
  least 307 rows of each sweep -- the rest fall back for the row's own
  reason (in the floor sweep: the three ray-traced rows, whose sweeps
  want a real driver, the Env hole and the light-split passes) -- and no
  row's frame was over the deferred bar where the unmodified row was
  under it, nor more than twice as far from the CPU device's. An Image
  Texture wired into Surface across the 54 textured rows: all on the
  driver, none worse. *The strict gate* (`tests/glsl_strict.py`, glslang
  15.1.0) gained `check_emission_route`: ten graphs at three rates,
  shadows off and on -- 60 planned frames, 39 distinct texts, none
  rejected, none refused (the whole gate on this tree: 559 distinct
  texts, none rejected).
  *This changes where these pictures are made, not the pictures:* on the
  GPU device such a material fell back to the CPU's frame and now shades
  on the driver, within the deferred bar of that frame; the CPU device
  is untouched. *Whichever way the CPU fills that surface.* The route
  restates nothing about a surface with no diffuse lobe: it bakes what
  `closure_to_surface` hands the probe. Under the batch rule recorded in
  the entry above that is the material's fallback diffuse, lit; under
  the per-pixel rule that replaces it (white at level 0, no ambient
  share -- applied over this tree from its branch before it landed) it
  is no diffuse at all, and the route's test and the entry above's pass
  unchanged on both, 50 and 47 checks. Pinned in
  `test_emission_slot_routing_twins_an_emission_only_closure` (50
  checks): the four kinds of lone link; the non-vacuity and the pass's
  own two lines (the chain added to the total, the diffuse a constant);
  the Emission node with constant and with linked sockets, the Mix (a
  constant and a per-pixel Fac) and the Add Shader over emissions, an
  explicit lobe beside the implicit one, a bare side, a Reroute; a
  material with its own fallback diffuse, level and model (the
  simulator moving with it exactly as the CPU does); the VERTEX
  and FACE rates, the three decal combines on both routes, the CONSTANT
  model; a frame with an Ambient Only lamp (the light row's ambient
  tail); an Image Texture, nearest and through the mip footprint; the
  traced reflection; two routed materials sharing ONE source text (a
  colour change is a texel, not a recompile); what stays refused -- a
  Diffuse mixed with an Emission, at Fac 1 too, and the Max blend by its
  name word for word, behind a Mix Shader and behind a Reroute included;
  the walk and the bare-socket rule; and the GPU device road on the fake
  driver drawing it within the bar with no fallback line. Eighteen
  mutations of the route each fail it.
- **A closure with no diffuse lobe has no diffuse, at every pixel (the
  rule read its batch: a hard edge through the glow).** The open item of
  the shader-socket entry (two above), hunted in a frame and closed. In
  `core/render.closure_to_surface` the branch for a closure with no
  master lobe decided its slots with `np.any(...)` over the BATCH it was
  handed: `if np.any(diff_w > 1e-6)` took the diffuse colour and level
  from the lobes and otherwise left the material's fallback diffuse (the
  viewport colour, level 1) fully lit, and the specular colour and the
  roughness had the same shape. One pixel's surface hung on which other
  pixels were shaded with it -- and a batch is a chunk of the shading
  loop, a band of the worker pool, a render region, the sixteen
  fragments of the deferred probe. *It reaches the picture.* Reproduced
  headless before the fix on the demo scene, the Ball wearing a Mix
  Shader (a Diffuse BSDF, an Emission) whose Fac is a mask (the
  interpolated normal's z > 0: the upper side glows). (a) *One
  in-process frame.* At 1280x960 the Ball's 60,697 pixels shade in four
  16,384-pixel chunks (`MIN_CHUNK`); the chunks lying wholly on the
  emission side lit the fallback under the glow, and the frame carries a
  horizontal edge through it -- 27,929 pixels up to 1.93 off the
  one-chunk frame, the two rows at the chunk boundary reading (0.9, 0.3,
  0.1) and (2.71, 2.13, 2.02). No pool and no region is needed. (b) *The
  worker pool.* 640x480 with three workers (bands cut at rows 160 and
  320): 3,093 pixels of the Ball's cap up to 1.79 off the in-process
  frame, rows 319 and 320 reading (0.9, 0.3, 0.1) and (2.60, 2.00,
  1.86). Two and four workers happen to cut where every band holds both
  sides, and drew the in-process frame. (c) *A render region* drawn
  wholly inside the glow: every one of its 1,200 pixels off the full
  frame's, up to 1.73. (d) *The corner road.* A pure-emission pixel
  sharing its batch with a diffuse one got a BLACK albedo (0 / 1e-6),
  and at the vertex and face rates `light x albedo` wiped the glow the
  corners had just lit: the emission side of the masked Ball rendered
  black (Ball mean (0.04, 0.05, 0.16) at the vertex rate, (0.67, 0.26,
  0.23) now). (e) *The probe is a batch too.* On a Ball that is Glossy
  but for a small Diffuse island (79 of its 598 pixels) the sixteen
  probe fragments were all glossy and baked the lit fallback for the
  whole material, while the frame's own batch held the island and shaded
  the glossy side with no diffuse at all: the plan QUALIFIED and the
  simulator frame was 1.93 off the CPU's on all 598 pixels. (1) *The
  rule, per pixel.* Every slot of that branch is decided from the
  pixel's own lobe weights. A pixel WITH a diffuse weight keeps what it
  had, bit for bit: the lobes' weighted colour, the clipped weight as
  the level, the material's ambient level (which carries the diffuse
  share since the entry below). A pixel with NONE has no diffuse
  response -- Blender's answer for an Emission, a Glossy, a Glass or a
  Transparent alone, and what every such pixel already was the moment
  one diffuse pixel shared its batch: diffuse level 0 and NO ambient
  share, over a WHITE diffuse colour. The ambient share goes because the
  ambient term is the diffuse colour times the ambient level and never
  reads the diffuse level. The colour is white rather than the mixed
  batch's black because nothing at the pixel rate reads it once the
  level and the ambient are 0, and the two roads that read the colour
  alone want the identity: the corner road's `light x albedo` (finding
  d) and a see-through pixel's refraction tint. And "no diffuse" cannot
  hang on the colour: the deferred pass fills its diffuse slot from the
  graph's own colour chain, so only the level and the ambient can say it
  to both devices. The specular level is the clipped glossy weight (0
  without one, as it always was) and the specular colour stays the
  material's own there; a pixel no rough-carrying lobe reaches keeps the
  material's roughness and exponent (it read roughness 0 and the clamped
  exponent 8192 the moment a rough lobe shared its batch). The
  graph-less material and the EMPTY closure (an unwired Surface socket)
  never reach the branch: the early return keeps the fallback surface,
  bit for bit. (2) *This changes pictures: every closure with no diffuse
  lobe, on the CPU device.* Surveyed old against new on 216 frames of
  the demo scene at 128x96 -- twenty closures at nine settings (the
  three rates, ray tracing, Transparency None, Screen Door and A-Buffer,
  a forced Constant model, shadows and the environment off) and six
  see-through closures on the layer road at six. An **Emission node**
  alone, and a colour wired straight into Surface: the Ball is exactly
  the colour at the pixel, vertex and face rates; it was the colour over
  the lit fallback (Ball mean (2.04, 1.44, 1.27)), and at the corner
  rates the fallback colour multiplied into both. Shader to RGB of an
  Emission reads the colour itself (through a second Emission the Ball
  was (3.19, 2.58, 2.44): the read lit a default surface under the
  glow). A **Glossy BSDF**, a **Metallic BSDF** and a **Principled BSDF
  at Metallic 1** (the node always hands on a DIFFUSE lobe, whose weight
  is 0 at Metallic 1 or Transmission 1): the highlight alone, Ball mean
  (1.15, 1.15, 1.17) -> (0.009, 0.006, 0.002). A GLOSSY lobe carries no
  reflection in this engine, so a metal is its highlight on black where
  it was its highlight on the viewport colour lit as plastic -- the
  largest change in this entry for an imported scene. A **Glass BSDF**,
  a **Refraction BSDF** and a Principled at Transmission 1: the
  reflection and the highlight alone ((1.21, 1.24, 1.31) -> (0.07, 0.10,
  0.14); (0.16, 0.22, 0.25) ray traced). A **Transparent BSDF** and a
  **Holdout** alone: bitwise unchanged wherever alpha composites
  (Sorted, A-Buffer, ray traced, the corner rates: the layer is alpha
  0); under Transparency None, where alpha is forced to 1, the opaque
  surface is black where it was the lit fallback, and under Screen Door
  the dropped pixel carries no colour. **Mixes by a mask**, one batch,
  pixel rate: Diffuse with Emission, with Glossy, and with Transparent
  on the layer road are bitwise what they were (their no-diffuse pixels
  were already unlit); Diffuse with Glass differs on the glass side's
  highlight alone (139 pixels, up to 0.87): a GLASS lobe adds a specular
  weight and no roughness, so its exponent was 8192 in a mixed batch and
  the material's 25 alone, and is the material's now. At the corner
  rates the emission, glossy and glass sides appear where they were
  black. Unchanged, bit for bit, at all nine settings: a Diffuse BSDF, a
  Translucent BSDF (a diffuse lobe at half its weight), a Principled
  BSDF at Metallic 0 and 0.5 and at Alpha 0, the unwired Surface socket
  and the graph-less material. (3) *Both roads.* What the plan qualifies
  it twins by construction, with no GLSL changed: the probe bakes
  through `closure_to_surface`, and the pass's ambient term is
  `s.diffuse * (ambient * the baked level)`. An emission-only closure
  rides the emission slot routing of the entry above, which restates
  nothing about the surface under the light: the probe bakes diffuse
  level 0, ambient 0 and a white diffuse, and the Emission Ball is its
  colour in the simulator as on the CPU (the frames 7.2e-7, 4.8e-7 and
  1.2e-7 apart at the pixel, vertex and face rates). A lone Glossy or
  Metallic (the specular slot routing) bakes the same three and twins at
  5.1e-5, 4.8e-7 and 1.2e-7. A raw EMISSION lobe beside a diffuse one
  (the masked mix) still refuses the plan by name, so both devices draw
  the CPU frame there. Two disagreements the batch rule held go with it.
  Finding (e): the glossy pixels twin now (the island's 79 remain, item
  4 (iv) below). And a Transparent BSDF alone under Transparency None or
  Screen Door, where the plan qualified and the pass lit the node's own
  colour as an albedo while the CPU lit the fallback: 0.48 off on all
  598 pixels of the Ball for a white one and 1.44 for a tinted one,
  7.2e-7 now (level 0 and ambient 0 leave the colour unread on both).
  (4) *Considered and left* (items i and iii are closed by the entry
  below). (i) *The ambient share of a PARTIAL diffuse weight.* A diffuse
  lobe at any weight above zero kept the whole ambient term, so a Fac
  running up to 1 stepped down by the lobe's colour times the ambient
  pool where it arrived: Fac 0.999 against Fac 1 was (0.021, 0.036,
  0.145) on the demo Ball. (Under the batch rule the step at exactly 1
  was the whole fallback popping IN, 1.12 the other way.) (ii) *The
  Glass and Refraction lobes' Roughness.* The nodes evaluate it and
  `closure_to_surface` never accumulates it, which is why a glass pixel
  has a specular weight and no roughness of its own (the 139 pixels
  above). (iii) *The master branch read its batch for the MODEL.* Two
  Halcyon Shaders of different models mixed by a mask shaded each batch
  with its heaviest lobe's model (the mean weight over the batch): Phong
  against Toon on the Ball, three bands at 640x480 against the
  in-process frame, 3,089 pixels up to 1.51. (iv) *The probe's sampled
  constancy cannot see an island.* The 79 island pixels of finding (e)
  still draw as the glossy surface in the simulator: sixteen fragments
  missed a lobe living on 13% of the material, `PROBE_FRAGMENTS`'
  standing risk for every baked field. (v) *A Mix Shader (Diffuse,
  Transparent) at a constant Fac 0.5 under Transparency None or Screen
  Door* qualifies and draws 0.48 off the CPU on all 598 pixels, before
  and after this fix (it has a diffuse lobe; the pass's colour chain is
  the mix of both nodes' colours, the CPU's diffuse the Diffuse BSDF's
  alone). Recorded, not touched. Pinned in
  `test_a_closure_with_no_diffuse_lobe_has_no_diffuse_at_every_pixel`
  (90 checks). At the function: the report's own two lines; every slot
  of every pixel bitwise the same alone and in a mixed batch, for a
  Diffuse mixed with an Emission, a Glossy, a Glass, a Refraction, a
  Transparent, a Holdout and a Translucent, with and without a material;
  the rule slot by slot; the graph-less and the empty closure keeping
  the material's own fields; a glossy probe of the masked material
  baking the frame's own rule. In the frame: the Emission Ball exactly
  its colour at the three rates, and through Shader to RGB; each side of
  the masked Ball bitwise its own node alone; a band, three rejoined
  bands, a render region and 96-pixel chunks each bitwise the in-process
  frame, the band at the corner rates too; the three-worker pool at
  320x240 bitwise the in-process frame (its top band is pure glow;
  skipped by name without a worker interpreter); a Glossy, a Metallic, a
  Principled-metal and a Glass Ball black with every lamp removed while
  a Diffuse Ball is not; the Transparent Ball under Transparency None;
  the Glossy and Metallic twins at three rates with the probe's bake
  read back; the tinted Transparent twin under None and Screen Door; the
  Emission node's twin at three rates with its Ball the colour in the
  simulator and the emission slot's bake read back; the masked mix's
  refusal. Twelve mutations each fail it: the batch rule restored for
  the diffuse, the specular or the roughness slot; the fallback kept per
  pixel; the ambient share kept, taken from every pixel, or decided by
  the batch; a black albedo, or the fallback's; the level or the
  specular level left the material's; the rule reaching the empty
  closure.
  `test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads`
  was re-run: the wired colour is still bitwise the Emission node's
  frame and frame(colour) - frame(black) still the colour, and it now
  also holds that frame(black) is black (48 checks).
- **A mix carries its ambient by share, the master branch is per pixel,
  and what cannot blend votes once per material (what the entry above
  left in the same function).** Three more reads of
  `closure_to_surface`, each seen in a frame before it was touched. (1)
  *The ambient share.* A diffuse lobe at any weight above zero kept the
  WHOLE ambient term -- the ambient is the diffuse colour times the
  ambient level and reads no weight -- so a Fac running up to 1 dropped
  the lobe's colour times the ambient pool where it arrived: (0.021,
  0.036, 0.145) between Fac 0.999 and Fac 1 on the demo Ball, for a
  Diffuse BSDF and for a Halcyon Shader mixed with an Emission alike.
  The ambient level carries the diffuse SHARE of the pixel's covered
  surface now. Plain branch: the material's level times (the Diffuse
  weights + the Translucent weights) / (1 - the Transparent share) -- a
  Translucent counted whole (the diffuse slot takes half of it, but both
  halves answer the pool), and a share within 1e-6 of the whole taken as
  the whole. Master branch: `1 + (ambient - 1) x the master's share`, as
  it was, less the opaque plain share that is not diffuse (an Emission,
  a Glossy, a Glass: it counted as a diffuse at ambient 1.0). The step
  is gone: Fac 0.999 is within 0.01 of Fac 1 on every pixel of the Ball,
  the lobe's own last thousandth. *What it changes:* a diffuse side (a
  Diffuse BSDF, a Translucent, a Halcyon Shader) mixed with an Emission,
  a Glossy or a Glass at a PARTIAL Fac, and a Principled BSDF at a
  partial Metallic or Transmission, lose the ambient of the share that
  is not diffuse -- 0.072 on the Ball at Fac 0.5 (half the lobe's colour
  times the pool), 0.036 at Fac 0.25, 0.044 for a Principled at Metallic
  0.5. Blender's reading (the diffuse BSDF's share of the world light),
  and the same on both devices: the probe bakes the level through
  `closure_to_surface` (0.5 where it baked 1.0). *What it does not:*
  every whole share is bitwise -- a Diffuse, a Translucent, a Principled
  at Metallic 0, a Specular BSDF, a Subsurface, an Add Shader with an
  Emission, and any of them, or a Halcyon Shader at any Ambient, mixed
  with a Transparent alone (the share is of what is covered). (2) *The
  master branch, per pixel.* The entry above first said this branch's
  slots other than the model were per pixel. They were not: the check
  behind that used levels under 1, where a batch-wide clip does not
  bite. The block that mixes master lobes with plain ones ran for the
  WHOLE batch when any pixel carried a plain weight, and its inner tests
  were `np.any` too. A pure-master pixel beside a plain one had its
  Reflection clipped at 1 (1.4 -> 1.0) and its opacity re-derived
  through 1 - (1 - o) (an ulp at 0.1); beside a plain glossy or diffuse
  pixel its Specular or Diffuse Level was clipped at 1 (2.0 -> 1.0, 1.5
  -> 1.0). Every test is the pixel's own weight now: a pixel with no
  plain weight is the master alone, bit for bit, and a mixed pixel keeps
  the arithmetic it had. Where neither a master nor a diffuse lobe
  weighs (the Emission side of a mask) the pixel takes the plain
  branch's rule -- level 0, no ambient, a white albedo. The weighted
  colour is black there, and at the vertex and face rates `light x
  albedo` wiped the glow: a Halcyon Shader mixed with an Emission by a
  mask rendered all 421 emission pixels of the Ball (0, 0, 0), and
  renders them (0.9, 0.3, 0.1). (3) *The vote.* What cannot blend -- the
  model, the normal's carrier, the Faceted flag, the python extras (the
  BI kit, the shadow ramp) -- is named by the heaviest master lobe, and
  a plain glossy side heavier than every master outvotes the model.
  "Heaviest" was the mean weight over the BATCH. Two Halcyon Shaders
  (Phong, Toon) mixed by a mask shaded each band with whichever side
  filled it: three bands at 640x480 against the in-process frame, 3,089
  pixels up to 1.51. A master at Specular Level 2 mixed with a Glossy
  BSDF by a mask: the band under the glossy cap at 320x240, 250 pixels
  up to 1.63 (the outvote and the clip of (2) together). The vote is
  cast ONCE PER MATERIAL: `ShadeJob.lobe_rank` evaluates the graph at
  the centroids of up to 64 of the material's own triangles (evenly
  spaced through its triangle list), `core/render.closure_rank` takes
  the mean weights there, each centroid counted by its triangle's area,
  and `closure_to_surface(..., rank)` reads them in place of the batch's
  -- from `shade_batch`, and from the deferred probe and the normal
  pre-passes of `gpu/shade.py`, which are batches too (sixteen probe
  fragments on the Toon cap baked Toon for the whole material; they bake
  the material's Phong, and the simulator frame is the CPU frame). It is
  a function of the scene alone: the same in every chunk, band, region
  and worker, and the same as the camera moves. A graph that will not
  evaluate there says so once and each batch votes for itself, as
  before; so does a bare call with no vote handed in. *What it changes:*
  a mix whose weights VARY over the surface is voted by its surface, not
  by the pixels a batch holds. On the survey's two masked Balls the
  visible majority had named the model (Toon; the Glossy side's
  Cook-Torrance) and the surface names the other (1.89 and 1.62 on the
  Ball). A mix at constant weights votes as it did: two masters at Fac
  0.3, 0.5 and 0.7 are bitwise. It is still ONE model per material; each
  side of a mask keeping its own is a per-pixel light loop, which this
  does not attempt. *Surveyed* old against new on 144 frames (48
  closures at the pixel rate, the vertex rate and Transparency None): 95
  bitwise, 49 changed, each of them named above. *Left.* (i)
  `light_surface` reads three blended flags for the batch and not the
  pixel -- the cel key (`cel_light`, the batch's first pixel), the
  console light limit (the batch's maximum) and the axis viewer (any
  pixel) -- so two masters that DIFFER in one of them and are mixed by a
  mask still read their batch there. Read in the code, not run. (ii) A
  mixed pixel still sums its levels toward 1, so a master at a level
  over 1 steps down where a plain weight begins. (iii) Bump Strength's
  bend is gated by `np.any` (a pixel at strength 1 is re-derived beside
  one that is not: an ulp). (iv) Shader to RGB collapses its closure
  with a bare call, so a mix read through it has no material to vote.
  (v) The deferred pass's ONE colour chain for a Mix Shader of two plain
  lobes is the mix of both nodes' colours, where the CPU's diffuse is
  the diffuse lobe's own: Mix (Diffuse, Glossy, 0.5) qualifies 0.45 off
  the CPU on all 598 pixels of the Ball (0.42 after this entry), the
  Diffuse / Transparent pair of the entry above 0.48. Seen, not touched:
  its own fix. Pinned in
  `test_a_mix_carries_its_ambient_by_share_and_votes_once_per_material`
  (61 checks). At the function: the share for a Diffuse mixed with an
  Emission, a Glossy, a Glass and a Refraction at three Facs; the
  bitwise set (a Transparent, a Holdout, a Translucent, an Add, a nested
  Transparent mix) and the covered-surface share of a three-way mix;
  every slot of every pixel of a hot master (levels 1.5 and 2,
  Reflection 1.4, opacity 0.1) mixed with each of seven plain lobes
  bitwise the same alone and in a batch, and beside a third lobe it does
  not carry; the master's ambient against a diffuse, a transparent and a
  non-diffuse plain side; the bare pixel's rule; the vote for the model,
  the normal and the extras in four batches, a vote of another shape, an
  unreadable vote, the glossy outvote both ways. In the frame: a band
  wholly on the Toon side and 96-pixel chunks bitwise the in-process
  frame; the frame bitwise the one whose minority master wears the
  majority's model, with the mask either way round; `lobe_rank` read
  back (three quarters against one quarter of the sphere, cached, None
  without a graph, and None said once by name when the graph will not
  evaluate there); the probe's baked model and the simulator twin; the
  hot master's band; the emission side at the three rates; Fac 0.999
  against the Emission alone for both branches; the probe's baked
  ambient for both. Twenty-two mutations each fail it: for the share,
  none applied, of the whole pixel, a Translucent at half, no
  float-noise snap, a Diffuse uncounted; for the master branch, the
  block or either inner test run for the batch, the non-diffuse share
  kept, unsnapped, or counting a diffuse or the transparent side, no
  white albedo or a kept ambient on a bare pixel; for the vote, ignored,
  taken at another shape, the outvote uncast or cast by the batch as
  well, `lobe_rank` empty or over one triangle, and `shade_batch` or the
  probe not handing it on.
- **A moved object under a still camera rendered from stale caches.**
  The CPU's G-buffer cache (R170) keys on a mesh fingerprint and the
  camera, and the fingerprint read the vertex positions as a strided
  sum. A translation whose components cancel -- the demo ball moved by
  (0.3, -0.3, 0), a grid-snapped (1, -1, 0) -- kept that sum, so with the
  camera unmoved the CPU restored the previous frame's raster (`gbuf
  HIT`) and shaded the old positions: 1.80 against the GPU's freshly
  rastered twin, 7.6e-6 in a fresh process. A viewport with a still
  camera and every motion-blur slice reach that road. A strided sum has
  a second blind spot, and three more caches shared it: an object lying
  wholly between the stride's samples (a 24-vertex box in a
  40,000-vertex scene: stride 79, not one of its vertices read) kept the
  mesh key, the BVH cache's key, the BVH upload key and the shadow maps'
  base signature for ANY move -- the box drawn, shadowed and ray-shadowed
  where it used to stand (1.62 / 1.72 / 1.72 against a fresh render).
  `core/mathx.content_digest` -- the CRC-32 of every position byte, one
  pass measured at ~2 us for the demo scene, ~60 us for 20,000 vertices
  and ~0.6 ms for 200,000 -- now sits beside the sums in
  `gpu/shade._mesh_key` (the one function behind the G-buffer cache, the
  plan and the triaux / sres / attribute uploads),
  `core/render._cached_bvh`, `gpu/rtrace.bvh_fingerprint` and
  `core/lights._shadow_base_signature`. Per-axis weights on the strided
  sum were weighed and left: beside the plain sum they still pass a
  (2, -3, 1) move, and no strided statistic sees between the samples.
  A re-export of an unmoved scene keeps every key and still hits.
  Pinned in `tests/test_r253_generated_coords.py`
  (`test_d_moved_objects_miss_the_frame_caches`: the frame after each
  move is bitwise a fresh render with no cache cleared, the GPU twin
  7.6e-6) and in `test_r253_review`'s signature test, which asserted the
  translation KEPT the mesh key and now asserts it changes while the
  plan signature still fingerprints the baked frame on its own. This
  fix left the normal, UV, material-index and triangle-index parts of
  the mesh key as strided sums; the R254 viewport round digests those
  arrays whole (`gpu/shade._attr_digests`, docs-dev/r254/viewport.md,
  section 9), and `test_d_unmoved_edits_move_the_mesh_key` holds the
  cases this entry named: an edit that cancels in the sum (every UV
  shifted by (0.1, -0.1); two rows exchanged), an edit between the
  samples, and two visible triangles' rows exchanged on the CPU road
  (bitwise a fresh render with no cache cleared; 1.40 on the stale
  raster).
- **A name the Coded Shader never declared is a compile error.** The
  report: `shaders/compiler.try_compile` on a six-line source whose
  `main` reads `nothing` returned a program with error `None` and no
  warnings, and `prog.run` raised `ShaderError: unknown identifier
  'nothing'` -- so `HALCYON_CodeNode.compile_source` left the error box
  empty and rebuilt the sockets, `export._node_props` filed the program
  and warned of nothing, and at shading time
  `GraphEvaluator.eval_output` caught the exception and the node read
  zeros. A missing parenthesis, a missing brace and an unknown function
  were compile errors already. The cause is the interpreter itself: it
  resolves a name when a lane reaches it. The source generator it
  replaced resolved names as it emitted code (`undeclared identifier`,
  `has no field`, `bad swizzle`, `swizzle out of range`), and
  `Interpreter.validate` had brought back the unknown-function check
  alone. *What the read found first*, each reproduced headless before
  anything changed. (a) *The same slip wherever a name stands*: a
  function's or a builtin's argument, a helper's body, an assignment
  target, an array index, a ternary arm; a local read above its
  declaration, outside its block, after its loop, or from inside the
  function that was called; a misspelled struct field, read or written
  (`S has no field 'aa'` when it ran); a swizzle that is none (`bad
  swizzle 'xq'` when it ran) and a swizzle past the vector's width,
  which was not even a ShaderError -- a bare numpy `IndexError`. (b)
  *Names never reported at all*: one in a branch no lane takes, or in a
  function nothing calls, ran clean frame after frame; one in a global's
  initialiser or a uniform's default (`const float TAU = 2.0 * PIE;`)
  was evaluated in an empty scope, the failure swallowed, and the global
  read zero. (c) *Declared names the run could not find*: `const float
  TAU = 2.0 * PI;` read zero by that same swallow; `uniform float a, b;`
  and the members of an HLSL `cbuffer` were dropped whole (the parser
  hands those back as a block and the interpreter walked single
  declarations), so no socket was made and the names were unknown
  identifiers when the shader ran; a free `gl_FragCoord`, bound in the
  root since 1.25.16, stopped there -- `call_user` copied the declared
  names into a function's scope one by one, so the builtin was in scope
  everywhere except inside a function. (d) *The GPU road.*
  `gpu/emit._port_code_node` only parses, so it accepts every one of
  these. With the name on a path a probed lane takes (the report), the
  plan's probe refuses the frame -- `'Coded': nodes the evaluator itself
  does not know: HALCYON_CodeNode: ShaderError: unknown identifier
  'nothing'` -- and the CPU draws it, the node at zeros. With the name
  behind a branch none of the sixteen probe points takes, or in a
  function nothing calls, the plan QUALIFIED, `device.preflight` (the
  same front-end) passed the pass, and its text carried the name to the
  driver as written (`_cncode_Color = vec4(nothing);`): the simulator
  drew the CPU's picture, and GLSL obliges a driver to refuse the text.
  And where the name is one the frame shader itself declares (`vUV`,
  read without its `in`, behind `vScreenUV.x` between 0.50 and 0.53 at
  128x96), the probes missed the strip, the frame qualified, and the
  simulated GPU frame differed from the CPU frame by up to 2.17 over all
  598 pixels of the material: nine of them reached the name on the CPU
  and the evaluator zeroed the node for the whole batch, while the pass
  drew every pixel, those nine in the full-screen quad's UV. *What
  changed.* (1) `shaders/names.py`: `NameCheck`, run by
  `Interpreter.validate` and so by every `try_compile`, walks the tree
  with the scopes `run`, `call_user`, `block` and `loop` build, holding
  a type where they hold a value, and raises with the line of the name
  (the parser's `var`, `call` and `member` nodes carry it): `line 6:
  undeclared identifier 'nothing'`, `line 6: S has no field 'aa'`,
  `line 4: bad swizzle 'xq'`, `line 4: swizzle 'xyz' out of range for
  vec2`, and `line 3: unknown function 'vec9'`, which had no line.
  Untaken branches, uncalled functions, uniform defaults and global
  initialisers are walked like everything else. (2) *Nothing declared
  is refused.* Where the check and the run disagreed about a declared
  name, the run changed: the globals are bound in source order
  (`Interpreter.order`) and an initialiser is evaluated in the root as
  it stands, so it sees what is declared above it (`TAU` is 6.28); a
  function's scope starts from the root's own slots, `gl_FragCoord`
  among them; `Parser.parse` flattens a multi-declarator global into its
  declarations, and a cbuffer's members are uniforms, with sockets. A
  swizzle is refused as out of range only past the widest the value can
  be when it runs -- `pow(2.0, v).xyz` is typed by its first argument
  and computed across the vector, and still compiles. (3) *A varying
  still needs its declaration.* `codegen.VARYINGS` is what `in vec3
  vNormal;` binds to, not a list of names in scope without one: the
  generator bound them free, the interpreter never has (an unknown
  identifier when the shader ran), and the port binds declared `in`
  names only. Read free, each of the 28 is refused with the line to
  write -- `undeclared identifier 'vNormal' (declare it: in vec3
  vNormal;)` -- and so are the generator's five implicit outputs
  (`gl_FragColor`: `declare it: out vec4 gl_FragColor;`). (4) *The
  callers follow with no change of their own*: Compile is cancelled
  with the error in the node's box and the sockets left alone; the
  export warns `<node>: line 6: undeclared identifier 'nothing'` and
  files no program; with no program both roads take the never-compiled
  answer (zeros -- the evaluator records no error, the frame qualifies,
  no pass carries the text); and `device.preflight` refuses an emitter's
  own pass that names something undeclared before the driver sees it.
  *Left as it is*: a global or a function written BELOW the function
  that uses it still compiles (the run binds every global before `main`;
  a driver refuses that order) -- tightening it would refuse sources
  that run today. *Seen in the same read, not name errors, not
  touched*: a struct constructor `S(a, b)`
  returns its first argument (a field read then raises `IndexError`);
  the comma operator parses and is `unsupported expression 'seq'` when
  it runs; a user function called with the wrong number of arguments
  compiles and runs (the R235 guard lives in the generator nothing
  calls); `vec2 + vec3` is a `ShaderError` when it runs;
  `mainImage(out vec4 fragColor, in vec2 fragCoord)` yields no output
  socket. Pinned in `tests/test_shaders.py` (3 tests, 50 checks: 28
  sources that must be refused, each with its whole message; 16 that
  must compile AND run to a stated value; all 29 varyings declared, and
  the 28 read free; both starter shaders) and in
  `test_code_node_name_errors_are_compile_errors` in
  `tests/test_r253_review.py` (15 checks: the real node class through
  the Compile operator and its drawn error box, the real
  `serialize_tree`, the evaluator, the plan and the simulator,
  `device.preflight`); against the previous front-end 36 and 9 of them
  fail, and eleven mutations, each taking one piece of the fix out alone
  (the check, the line, the branch walk, the initialiser walk, the field
  check, the width rule, the root scope, the source order, the builtin's
  reach, the flattening, the cbuffer's qualifier), each fail at least
  one. One-off scans of the new front-end, each beside the same scan of
  the old one: all 121 shipped presets through the GPU road on the fake
  device (the demo frame, the render and the post chain) compile the
  same 165 distinct pass sources, none refused; the enumeration of
  `tests/glsl_strict.py` with the front-end standing where glslang
  stands (every engine model at every rate, every console type and
  option, the cel masters, the post stages, the sky and the raster
  fragments -- what `device.preflight` will say to each) compiles the
  same 615 and leaves the same 15 refused by name, 245 distinct texts,
  none rejected; and a run of the whole suite as it stood, with every
  front-end compile error logged, raised three (each a test's own negative
  case), failed no check the old front-end passes, and met no global
  initialiser reading a name in any engine source -- the initialiser
  change moves no engine value. The check costs about 4 ms on a
  1,562-line material pass, once per distinct text.
- **The shadow atlas sampler is declared once per pass (the redefinition
  the node-name entry above left for its own fix).** The report: the raw
  GLSL of a deferred material pass declared `uniform sampler2D
  hal_shadowpack;` once per shadowed lamp. Every mapped lamp reads the ONE
  combined atlas, and `gpu/material._shadow_function` opened each lamp's
  `hal_shadow_vis<i>` with that line, so a pass under two mapped lamps --
  the demo scene's SUN and POINT, with shadows on -- held it twice.
  glslang 15.1.0 (`-S frag` under a `#version 430 core` line, what
  `tests/glsl_strict.compile_source` does) refuses the text,
  `'hal_shadowpack' : redefinition`, and compiles it with the line kept
  once. *Who was handed it.* On the CreateInfo road nobody:
  `device.strip_declarations` takes every such line off and the interface
  declares the sampler once. `device._compile_dynamic_miss` hands a driver
  the raw text on one road only, the legacy `GPUShader(VERTEX, fragment)`
  constructor, which it reaches when `gpu.types` has no
  `GPUShaderCreateInfo`. *Why no check saw it.* Halcyon's own front-end
  takes a redeclared uniform, so the simulator and the fake device drew
  the pass; the pre-flight's `duplicate_definitions` counts function
  bodies; and the strict gate planned every scene with shadows off
  (`check_models`, `check_masters`, the console scene builder,
  `check_coded` on purpose), so it had never compiled a pass that carries
  a shadow term. *What changed.* (1) *The emitter.*
  `gpu/material.SHADOWPACK_DECL` is the line; `assemble_frame` appends it
  once, ahead of the loop that writes the shadow functions, when at least
  one lamp of the pass is mapped, and `_shadow_function` writes no
  declaration. The sampler is listed once in the pass's
  `binds['samplers']`, where it was; `strip_declarations` and the
  CreateInfo sampler list are untouched. Measured on the tree before and
  after, over 366 plans (every engine model at the three rates with
  shadows on and off, and nine shadow variants at two sizes; 1098 passes):
  the binds are identical in every pass, the stripped text is line for
  line the same in every pass (blank lines aside), and the raw text is
  byte for byte the same in 966 and, in the other 132, exactly the old
  text with its declarations gathered into one (two into one in 102, three
  in 18, eight in 6; in 6 passes with a single mapped lamp the one line
  moved from the head of its function to the head of the block). No value
  moves: the simulator's frame of the once-declared passes is bit for bit
  its frame of the 1.92.0 text. (2) *The strict gate*
  (`tests/glsl_strict.py`) covers shadows, and its docstring says so. The
  material sweeps run twice, shadows off and then on with the demo scene's
  two mapped lamps: the engine models at every rate (174 compiled, 3
  refused by name, 90 passes carrying a shadow term), the console types
  and options (396, 12 refused, 228), the cel masters (4; 6) and the coded
  shaders by node name (8; 8). `check_shadow_kinds` plans the demo scene
  under thirteen more kinds -- one, nine and sixteen PCF taps, softness 0,
  the MIDPOINT map, a SPOT's single map beside the POINT's cube, eight
  mapped lamps on the one atlas, two traced lamps hard and soft (a SUN's
  disc, a POINT's sphere), a traced AREA lamp over its rectangle and over
  its disk, and a mapped lamp beside a traced one under Per Light, hard
  and soft -- and holds every pass to the shadow functions its kind names.
  A traced lamp plans NO shadow term without a BVH on the job, so
  `_pass_sources` puts render()'s own on it (`_frame_bvh`;
  `tests/test_r251_material._job_for` takes a `bvh`), and a shadowed sweep
  in which no pass carries a shadow function fails by name: that is the
  state the gate was in. A text glslang has judged is not handed over
  again, so the doubled sweeps ask it for 406 distinct texts where the
  single ones made 1779 calls. With glslang 15.1.0 the gate passes;
  against the 1.92.0 emitter it fails 246 labels (30 engine models, 198
  console, 2 masters, 8 kinds, 8 coded), each one this redefinition and
  nothing else. *What else shadows on turned up*: nothing. One-off, each
  pass both as raw text and as the unit CreateInfo declares (the interface
  `gpu/shade` builds over the stripped text), under glslang: the gate's
  sweeps with shadows on and thirty-odd variants of the demo scene (the
  kinds above, image cookies on a SPOT and a SUN, an Only Shadow and a
  negative lamp, ambient occlusion and radiosity beside the shadows, up to
  eight lamps) -- 694 frames, 15 refused by name, 2037 passes, every one
  compiled on both roads; no uniform declared twice, every sampler of a
  spec declared once in its raw text and every declared name in its spec.
  *Left as it is*: the pre-flight is not widened to repeated declarations
  -- on the CreateInfo road they are stripped and harmless, and a refusal
  there would turn a pass that draws into a CPU frame; the text-level
  check lives in the test. *Not in the gate*: the caustics cookie and the
  World Light's lamp, which render() adds to a frame and the gate's job
  recipe does not (in a pass they are a cookie function and one more
  mapped SUN). Pinned in
  `test_shadow_atlas_sampler_is_declared_once_per_pass` in
  `tests/test_r253_review.py` (52 checks, 54 where `HALCYON_GLSLANG`
  points at glslang, which then compiles each shadowed pass and refuses
  the 1.92.0 text of each): the demo scene's three passes declaring the
  sampler once, on its own line, ahead of the first fetch, and listing it
  once; three and eight mapped lamps the same; one declaration beside a
  traced lamp, whichever comes first, and none (in the text or in the
  binds) for traced lamps alone, at the vertex rate or with shadows off;
  no uniform of any name declared twice; the 1.92.0 text put back -- named
  by the text-level check, let through by the pre-flight, its stripped
  text and its simulated frame the fixed pass's own -- so the checks can
  tell; what each road of `_compile_dynamic_miss` hands a stand-in driver
  (the raw text with one declaration; the stripped text with none and the
  sampler once, in its slot); and the rule read from the code. Against the
  1.92.0 emitter 13 of them fail, and fourteen mutations each fail at
  least two, none by raising: the fix taken out whole; a declaration left
  in each function beside the assembler's; the assembler's written after
  the functions, once per mapped lamp, twice, or not at all; the sampler
  unlisted or listed twice; the decision made for a traced lamp too, from
  the second mapped lamp on, by the first lamp alone or by the last; the
  line without its newline or under another spelling.
- **The sized-lamp lookup leaves the atlas declaration to the assembler
  (the two fixes met).** The entry above declares `hal_shadowpack` once
  per pass; the sized map lookup of the light round (`gpu/softmap.py`,
  the function a mapped lamp with a size takes from Samples 2 up) landed
  beside it still opening its own text with a copy, as every mapped
  function did when it was written. Each alone was whole; merged, the raw
  text of a pass declared the sampler twice with one sized lamp and three
  times with two -- the same redefinition, back on the legacy-constructor
  road, for exactly the lamps the light round made soft. The copy is
  gone, and `test_shadow_atlas_sampler_is_declared_once_per_pass` plans
  the demo scene with one and with both lamps sized (one declaration, no
  uniform twice) and reads every string constant under `gpu/` for a
  second writer of the line (there is one: `SHADOWPACK_DECL`). Putting the
  copy back fails all three.
- **What the Coded Shader compiles, it runs: struct constructors, the
  comma, argument counts, vector widths, entry-point parameters, input
  warnings.** Six kinds of source went through
  `shaders/compiler.try_compile` with error `None` and then failed or
  went wrong in `prog.run` -- none of them a name error, so the name
  check (`shaders/names.py`, run by `Interpreter.validate`) passed every
  one. Each was reproduced headless first (`try_compile(src)`, then
  `prog.run({}, {}, 4)`). (1) *Struct constructors.* `S s = S(0.5,
  vec3(0.1, 0.2, 0.3)); Color = vec4(s.a, s.b);` raised `IndexError` on
  the field read. `Parser.parse_struct` wrote every struct into
  `gtypes.BY_NAME`, the module's table of builtin types;
  `Interpreter.constructor` looked there first, took the scalar-cast path
  and answered its first argument typed `S` -- the struct branch below it
  could not be reached. The table is module-global, so the name also
  outlived its source: a later shader's own `float S(float a, float b)`
  was read as that constructor, and `S(0.25, 0.5)` answered 0.25. The
  registration is gone (the parser, the name check and the interpreter
  read the source's own `structs`), the constructor looks a struct up
  before the builtin types and builds one field per argument, each
  converted to its field's type, and the struct's type is one `GType`
  wherever it is made (the unreachable branch and the name check built it
  with `rows=1`, a matrix by `is_matrix`). One argument per field is the
  rule: `S(0.5)`, and the one-argument `S(s)` the scalar path answered as
  a copy by accident, are `S() takes 2 arguments (one per field), got 1`
  -- no driver compiles either. (2) *The comma operator.* `for (int i =
  0; i < 3; i++, b += 1.0)` parsed to `('seq', l, r)` and the interpreter
  had no case: `unsupported expression 'seq'` when it ran. It has one
  (the left side for what it writes, the right for the value). The same
  gap was looked for in everything the parser emits: of its 19 binary and
  11 assignment operators `^^` alone had no case either (`unsupported
  operator` when a lane got there), and has one. (3) *Argument counts.*
  With `float f(float a)`, both `f(1.0, 2.0)` and `f()` compiled and ran
  -- the extra argument dropped, the missing one zero. The guard that
  refuses this (R235) sits in `codegen.Generator.user_call`, and nothing
  has called the generator since the interpreter replaced it.
  `Interpreter.wrong_count` is the one place that knows the counts -- a
  shader's own functions (every overload), a struct's fields, a builtin's
  range -- and the name check asks it for every call before the shader
  runs: `line 3: f() takes 1 argument, got 2`, `f() takes 1 or 2
  arguments, got 3`, `sin() takes 1 argument, got 2` (a builtin raised
  only when a lane reached the call, and never in a branch none took).
  `Interpreter.pick` no longer falls back to the first function of the
  name, and a shader's own function hides the builtin of its name
  whatever the count, as on a driver. *An HLSL default argument is kept.*
  The parser parsed `float f(float a, float b = 0.5)` and dropped the
  default, so `f(a)` read `b` as zero; the parameter tuple carries it
  now, the count is 1 to 2, and the argument left out reads the default
  (evaluated among the globals). Without it the count check would have
  refused a call the dialect allows. (4) *Vector widths.* `vec2 a; vec3
  b; a + b` raised `cannot combine vec2 with vec3` from `gtypes.promote`
  when the operator ran. The name check already computes every
  expression's type by the interpreter's own rules, so it refuses the
  pair where the run would: every operator that goes through `promote`, a
  comparison, the two arms of a ternary, a compound assignment, and the
  builtins whose rule promotes its arguments (`min(a, b)`, where NumPy
  raised on the two shapes first). The message is `promote`'s and the
  line is the operator's: the parser's `bin`, `assign` and `cond` nodes
  end with it. *A text a driver compiles is not refused on the
  front-end's own shortcut* (every pass the engine emits is pre-flighted
  here). Where the interpreter's type is not the language's -- a matrix
  times a scalar is typed as a vector, and of two functions with one name
  and argument count the first is taken whatever the argument types --
  the type is marked loose and no width is judged on it or on anything
  computed from it; the swizzle check reads it as before. (5) *Entry
  points.* `void mainImage(out vec4 fragColor, in vec2 fragCoord)`
  compiled into a program with no outputs: `Interpreter.run` called the
  entry with no arguments, so every parameter read zero and nothing an
  `out` parameter was given left the call. The entry's parameters are
  bound as the generator bound them (`codegen.param_binding`, its
  `bind_param` moved out of the class): an `in` reads the renderer's
  input of its name -- `fragCoord` the two-wide fragment coordinate,
  `uv`, `normal`, `vColor`, cut or filled to the parameter's own width --
  and an `out` / `inout` starts at zero and is an output of the program,
  a socket on the node. The value the entry returns is delivered too: the
  `Result` socket of a `vec4 main()` was declared by the schema and read
  `None`. The entry is looked for under the generator's names again
  (`main`, `mainImage`, then `PSMain`, `psmain`, `MainPS`, `frag`,
  `fragment`, `pixel`, `PS`; the first function when none of them is
  there, as before), so an HLSL `float4 PSMain(float2 uv : TEXCOORD0) :
  SV_Target` written below its helpers is the function that runs. On the
  GPU road a `mainImage` was already refused for the deferred pass by
  name; a `main` that takes parameters is too (the frame's `main()` calls
  it with none, which no driver compiles and the count check now stops at
  the pre-flight). (6) *Warnings.* `Interpreter.warnings` was never
  written to, so the generator's `unknown input 'vNormals': bound to
  zero` was gone and the node's warning box (`HALCYON_CodeNode.warn`) had
  nothing to draw. A declared `in` / `varying` / `attribute`, or an entry
  parameter, that the renderer binds nothing to is warned of by name. One
  that is in `codegen.VARYINGS`, or is itself a key the renderer supplies
  (`in vec3 position;` reads `position`: the run looks a declared input
  up under its own name first), is not -- `codegen.INPUT_KEYS` is that
  list, pinned against `core/nodeeval._shader_inputs`. `compile_source`
  clears the warning with the program it belonged to (a source that no
  longer compiles, or is empty, kept the last good program's line).
  *Nothing that compiled and ran is refused.* Held three ways. Every
  source the whole suite hands the compiler was recorded (1739 distinct
  texts: the engine's passes through the simulators and the pre-flight,
  the twins, the tests' own) and compiled by the front-end before and
  after: 1675 compile in both with the same outputs, uniforms and inputs
  and no warning, 64 are refused in both with the same words. The
  enumeration of `tests/glsl_strict.py` with a recorder standing where
  glslang stands (every engine model at every rate, every console type
  and option, the cel masters, the post stages, the sky and raster
  fragments, the coded shaders by node name: 1771 fragment texts, 253
  distinct) and the planner's pass texts for all 313 feature-matrix rows
  and for 11760 Console Emulation Shader configurations (every machine by
  its panel's options by the node's rate, at three scene rates, with the
  light limit, a painted mesh and the luma / toon dials: 408 distinct)
  compile the same in both, with the same declarations. And 12000 seeded
  random shaders (operators, calls, constructors, swizzles, ternaries
  over every type; half with wrong counts mixed in) were compiled and run
  by both: of the 5431 that ran before, 5274 run to the same values, 104
  are refused for the wrong count of a function or a struct constructor,
  none is refused for its widths, and 53 raise where they "ran" -- each
  uses a struct constructor's value as a number, which ran only because
  the constructor answered its first argument; of the 625 that raised
  `cannot combine` when run, 624 are compile errors (the last combines a
  struct, below). What IS newly refused is, by intent: a wrong count
  wherever it is written (it ran, on other arguments than the ones
  written), and two widths in code no lane reached -- a branch never
  taken, a function never called -- as the name check refuses an
  undeclared name there. *Seen and left* (each reproduced; none is one of
  the six): of two functions with one name and argument count the first
  is called whatever the argument types (`float f(float)` then `vec3
  f(vec3)`: `f(v)` runs the float one); a matrix times a scalar raises in
  `einsum`; a struct used as a number, or compared with another, raises
  when run; a struct field declared as an array is stored and typed as
  one element (`l.dir[1] = v` reads back wrong, `l.dir[1].xy` is refused
  as a swizzle of a float); a `vec3` initialised from a `vec2` keeps two
  components, and `vec3(a, b)` / `vec2(a, b, c)` pad and cut in silence;
  a source with no function at all compiles and raises `no main() in
  shader` when run; `in vec3 vBitangent;` reads zero unwarned (the table
  binds it, the renderer supplies nothing); and `in vec3 position;` -- a
  renderer key used as a name -- reads the position on the CPU and zero
  in the deferred pass (`gpu/emit` binds table names alone). Pinned in
  `tests/test_shaders.py` (six tests, 67 checks: each reported shader,
  its neighbours, per-lane runs, every operator the parser knows, every
  builtin at each count of its range and one either side, 121 type pairs
  by 7 operators against `promote` itself, the entry bindings, the
  warning list) and in `test_code_node_compiles_what_it_runs` in
  `tests/test_r253_review.py` (18 checks: the real node class through
  Compile, the warning and error boxes, the export, the evaluator at
  shading time, the GPU port and `device.preflight`). 32 mutations of the
  fix each fail them.
- **The sky twins hold on any BLAS kernel, and the jitter pair on NumPy
  2.4 and later (the cloud-container run).** In a cloud container (Linux,
  Python 3.13, NumPy 2.5.3 on OpenBLAS 0.3.34, an AVX-512 Xeon) `run_all`
  failed 20 checks that demand `d == 0.0` of a GPU sky twin and raised in
  one test, beside the 1.89.0 zip pins that fail wherever the zip is
  absent (ten names, twenty checks in the fifteen R251 modules): eight in
  `test_gpu_sky_pass` (NODES `sky_blend` 10 px, 40 at ss 2 on the block
  road; GRADIENT 11; HDRI and the env image 13; `shade.simulate`'s
  mirror), three under Ground Fog (5.96e-07), eight in
  `test_r251_sky_camera` (the sheared vp 4.47e-08, LightWave's backdrop
  2.4e-07 to 8.3e-07, the fake device's LW frame), the Super FX plot
  through the fake device (5.96e-08), and `test_w2_c127_jitter`
  (`TypeError: only 0-dimensional arrays can be converted to Python
  scalars`). (1) *One operation, and it is not in the sky.* All twenty are
  the ray of `render._background_image`: `world = pts @ inv.T`, an
  (N,4)x(4,4) float32 matmul that NumPy hands to BLAS `sgemm`, against the
  four statements of `hal_sky_ray`, `((nx*i.x + ny*i.y) + i.z) + i.w`.
  Nothing in `core/sky.py` and nothing in the simulator's NumPy backend
  moved -- no `pow`, no `atan`, no `exp`, no reduction. An unfused `sgemm`
  sums in the shader's order and rounds after every operation; OpenBLAS's
  Haswell-and-later cores compute the second product fused into the first,
  `fma(ny, i.y, nx*i.x)` -- one rounding where the chain has two -- and
  land an ulp away in about three rows of ten on random input. Emulated in
  float64, that fused chain is bitwise those kernels on 100,000 rows and
  gives the container's own counts on the suite's frame (10, 40, 11 and 11
  px; 7.15e-07 on the (7, 3) squeeze). (2) *The variable is the kernel,
  not the NumPy version.* Measured on that one CPU with the version and
  the core varied separately (`OPENBLAS_CORETYPE`), twelve configurations
  of the unfixed tree: NumPy 1.26.4 passes the twenty there, but only
  because its OpenBLAS 0.3.23 does not know the CPU and falls back to
  Prescott -- forced to Haswell the same 1.26.4 fails the same twenty;
  2.0.2, 2.1.3, 2.2.6, 2.3.5, 2.4.6 and 2.5.3 (OpenBLAS 0.3.27 to 0.3.34,
  which run it as Cooperlake or SkylakeX) fail them; and 2.3.5 and 2.5.3
  forced to Nehalem or Sandybridge pass them. The verdict follows fused /
  unfused on every version. So the twenty pass wherever the float32 matmul
  is unfused -- those cores, or a NumPy without BLAS (its own loop is the
  chain: measured through the strided path of 1.26.4 to 2.2.6) -- and fail
  on the default wheel of any CPU OpenBLAS runs as Haswell or later, 1.26
  to 2.5 alike. Which one the release machine has could not be measured
  from the container; the new module prints it, and (5) shows the R253
  round itself ran on a fused one. (3) *The fix is the discipline, not a
  tolerance.* `core/mathx.unproject_far(nx, ny, inv)` is the chain, one
  ufunc call per rounding -- `a = nx*i.x; b = ny*i.y; a = a + b; a = a +
  i.z; a = a + i.w` -- on NumPy's own float32 multiply and add, which are
  separate correctly rounded operations on every build and CPU (the
  `view_depth` / `object_space_points` precedent), and `_background_image`
  and `sky.mode7_rows` (whose docstring promises the frame's own
  unprojection) call it. At 720p the chain takes 9 ms where the stack and
  the four-thread `sgemm` took 6, inside the noise of the 165 ms the whole
  CPU sky of that frame costs. The twins keep `d == 0.0`: a stated
  tolerance would have had to cover a ray an ulp off, and that is not
  bounded at the pixel (it flips a band edge or a nearest texel whole,
  which is why the bar is 0.0). After the change the twenty pass on ten
  configurations: NumPy 1.26.4, 2.0.2, 2.3.5, 2.4.6 and 2.5.3 on Prescott,
  Haswell, Cooperlake, SkylakeX and Nehalem, and 2.5.3 with NumPy's own
  AVX-512 and AVX2 dispatch switched off (`NPY_DISABLE_CPU_FEATURES`: the
  `pow`, `atan2` and `exp` both roads share return other bits there -- a
  fifth of `pow`'s results and two fifths of `atan2`'s move without
  AVX-512 -- and the roads still agree, each calling the same ufunc on the
  same operands). Bit-neutral on an unfused kernel: the old tree and the
  new return the same bytes from all 685 `_background_image` calls, all 48
  `mode7_rows` calls and all 659 `render()` frames of the sky, fog,
  cube-map and region modules, so every pin of a CPU frame is what it was
  there. On a fused kernel the sky pixels move by that ulp ONTO those
  bytes: the 130 skies of the nine GPU-twinned sky tests, 57 of which
  differed between SkylakeX and Nehalem, are one set of bytes on SkylakeX,
  Haswell and Nehalem now. Not claimed: a whole CPU frame identical across
  kernels -- the vertex transforms, the camera matrices' own products and
  the CPU-only skies (`dirs @ sun`) are still matmuls, read by both roads
  alike. Closed on the way: a ONE-row product (a frame with one uncovered
  pixel) took another BLAS routine and another order, off the chain in
  nearly four rows of five on fused and unfused cores alike; the chain has
  no batch in it. Left as they were: the three other `@ inv.T`
  unprojections (the light cones in `core/render.py`, the volume
  containers in `core/volume.py`) are CPU-only, with no twin to hold. (4)
  *`raster.jitter_at` answers in the shape it was asked.* The function
  takes pixel indices of "any shape", runs its hash on `np.atleast_1d` of
  them (a wrapping uint32 multiply warns on NumPy scalars and wraps
  silently on arrays) and never gave the shape back: a pair of scalars
  came out as two (1,) arrays, and the test's `float(jx1)` is a
  DeprecationWarning from NumPy 1.25 and a TypeError from 2.4. That is the
  function's leak, not the test's: the offsets now come back in the
  indices' broadcast shape -- planes for planes (`jitter_offsets`), a list
  for a list (the compute raster's replay, which indexes `jx[:, None]`),
  0-d float32 for a scalar pair -- and the R251 test stands as written.
  This one IS the NumPy version: 1.26.4 to 2.3.5 warn, 2.4.1 to 2.5.3
  raise. A scan of the whole suite under `sys.monitoring` (every raise of
  that TypeError recorded, caught or not) found this one site: nothing the
  suite reaches in the engine converts a (1,) array to a scalar. (5) *Two
  kinds of pin hold a CPU frame's bytes, and on a fused kernel both held
  its ray.* There the fix moves ten sky pixels of a 96x72 frame by one ulp
  (5.96e-08), onto the chain; a machine with a fused kernel could not hold
  `d == 0.0` against the shader and the bytes of a matmul sky at once, and
  never could. (a) *The mapping round's three frame digests.*
  `test_r253_mapping`'s `OLD_FRAME_DIGESTS` ("the pre-edit bytes",
  computed on `claude-cloud` b0cbe1c) matched in the container before the
  fix and not after: they are a fused kernel's frames, with its ray in
  them. So on the machine that computed them the twenty sky twins were red
  already, and have been since the mapping round at least. The test now
  renders each scene twice: with the ray put back as the matmul it renders
  the pre-edit digests, untouched (the ray is the whole difference, 10
  px), and as it is `FRAME_DIGESTS`. Both rows are the bytes of ONE
  machine class, as the pre-edit row always was -- a fused BLAS under
  NumPy's AVX-512 loops (1.26.4, 2.0.2, 2.3.5 and 2.5.3 agree there on
  Haswell, Cooperlake and SkylakeX); an unfused kernel renders 1539176161
  / 928542048 / 2436496908 before and after the fix, and a NumPy without
  the AVX-512 loops other bytes again, because the vertex transforms are
  matmuls too and `pow` / `atan2` are SIMD loops. Off that class the pin
  fails exactly as it did; widening it (a digest per class, or a
  comparison in place of a checksum) is the next entry's. (b) *The
  previous releases in their zips.* `test_render._prev_engine` renders
  "bitwise the 1.89.0 engine", and every release computed the ray as the
  matmul (the line stands exactly once in all 29 revisions of
  `core/render.py` in the repository, 1.9.0 to 1.92.0). On a fused kernel
  with the zip present the fix alone would have turned the identity pins
  red: measured with a stand-in (the pre-fix tree zipped as
  `halcyon-1.89.0.zip`), 161 failing lines in 13 of the 16 modules that
  load it, against 27 before (the twelve sky twins those modules hold and
  two artefacts of the stand-in). The loader now replaces that ONE line in
  the extracted copy (`world = pts @ inv.T`, and `sky.mode7_rows`' own
  from 1.90 on) with a call to this tree's `mathx.unproject_far`, appended
  to the file as its source so the copy stays whole for a worker that
  imports it afresh: line for line, exactly once or the file is left as
  shipped and named in a `note` (a release cut after this fix states the
  chain already and is left alone silently: the 1.92.0 zip the R254
  ray-road pin asks for), and a failure there never stops the engine
  importing. On an unfused kernel that is the shipped arithmetic bit for
  bit, so the zip pins are what they were; on a fused one both engines
  state the same ray and the pins compare everything else, as they were
  written to. With it the stand-in run is 4 failing lines (the two
  artefacts, twice) and no new one, the same against a second stand-in
  zipped from the 1.90.0 tag, and on Nehalem, over three of the modules,
  the fixed tree and the pre-fix tree fail the same one check. Pinned in
  `tests/test_r253_kernels.py` (6 tests, 33 checks, registered in
  `run_all`), on every machine whatever its BLAS: the chain against a
  float64-then-round reference (IEEE's own answer for each single
  operation, with no float32 arithmetic of NumPy's in it) on the demo
  frame and on 100,000 random rows; four twins of the report at `d ==
  0.0`, and the SAME four with the emulated fused ray standing in for the
  chain, which move by the container's counts -- the failure reproduced
  wherever the suite runs; a source scan (no matmul in the two functions,
  multiply and add alone in the chain); `mode7_rows` reading the same ray;
  the shape law, with `float()` of the scalar pair held silent under
  warnings-as-errors so it bites on NumPy 1.26 too; the loader's
  replacement on the two lines as the releases wrote them (the chain's
  bytes after, the machine's matmul before, the line numbers kept, a line
  that stands twice left alone and named once, a release already on the
  chain left alone and not named). Six mutations each fail it on fused and
  unfused kernels alike (the matmul back in `_background_image`, in
  `mode7_rows`, inside `unproject_far`; the chain reassociated; the
  reshape dropped; the loader's replacement dropped). The module prints
  one `note` line, never asserted: `NumPy 2.5.3: this machine's float32
  matmul is the fused chain, bitwise (an FMA kernel) -- 32375 of 100000
  rows off the chain` in the container, `... is the chain, bitwise (an
  unfused kernel)` where the twins were green. With this the container's
  `run_all` fails the zip pins and nothing else: on the tree this entry
  landed on (`claude-cloud` 1e6239c), 12,939 checks pass where 12,863 did,
  and the failing lines that remain are the baseline's zip lines, byte for
  byte.
- **The three frame digests hold in every machine class that was measured
  (the decision the entry above left open).** `test_mapping_frame_twin`
  ends on the bitwise-neutral rule for whole frames: three pre-existing
  featurematrix scenes (`textured`, `demo`, `mirror`; `R.render`, 96x72,
  transparency NONE) held to frozen `adler32` constants -- the pre-edit
  bytes with the sky ray as the matmul it was, and the frame as it renders
  now. A whole-frame checksum is the bytes of ONE machine class, so the
  three checks passed on a fused BLAS under NumPy's AVX-512 loops and
  failed on every other machine, whatever the code. (1) *Two functions,
  isolated.* The three frames were traced call by call. The kernel half is
  the float32 matmul of the vertex transforms (BLAS `sgemm`, fused or not
  by the core OpenBLAS picks, as in the entry above). The other half is
  float32 `np.power` and nothing else: with NumPy's AVX-512 loops left on
  and `mathx.safe_pow` alone handed the C library's `powf` (twelve calls
  over the three frames, some 1,600 results changed), the frames ARE the
  bytes of the class without those loops, digest for digest on both
  kernels. NumPy runs its AVX-512 `pow` loop where it dispatches one and
  calls `powf` where it does not (bitwise glibc 2.39's on 8,192 lanes; a
  fifth of them differ between the two). Not `arctan2`: the three frames
  never call it (one import-time table does, in float64). Not the camera:
  its float64 `tan` / `arctan` take the AVX-512 loops too, and build the
  same matrices in all four classes. Not float32 `sin`, which differs
  between the AVX2 and the SSE loops and reaches no pixel of these frames
  -- so with AVX-512 off, AVX2 on or off is one class. Two functions with
  two answers each: four classes. (2) *The pin is a table, and its key is
  measured.* `machine_class()` computes the key where the test runs:
  `adler32` of two float32 matmuls in the shapes the vertex and normal
  transforms take ((N,4)x(4,4) and (N,3)x(3,3), every column general) and
  `adler32` of `np.power` on 8,192 lanes (a base in [0.25, 1) to an
  exponent in [0, 32), a base in [0.25, 4) to one in [-4, 4): every result
  a normal float32, so the key cannot turn on how a process treats
  denormals), the operands built from an integer hash and one exact
  division so that they are the same floats on every NumPy -- no RNG
  stream to trust. Never a CPU name, a NumPy version or a feature flag:
  the bytes are the key, and the kernel module's reading of the same
  matmul on the sky's rows ("the chain" / "the fused chain") is printed
  beside it in a `note`, never keyed on. `FRAME_DIGESTS` holds one row per
  key and, for each scene, the pair (pre-edit, now). As textured / demo /
  mirror: a fused matmul under the AVX-512 loop (`0xc1c32195`,
  `0x7f01125b`: the pair the test held before) 1656485521 / 1045851408 /
  1324349618 and 3858757274 / 3248123161 / 3526621371; the unfused chain
  under that loop (`0xb34ecf20`) 1539176161 / 928542048 / 2436496908,
  twice; a fused matmul under `powf` (`0x04fa130f`) 3073177232 /
  2462543119 / 2741041329 and 981464729 / 370830616 / 649328826; the chain
  under `powf` 2955867872 / 2345233759 / 3853188619, twice. Where the
  matmul is the chain the sky ray never moved and the pair is one number.
  (3) *Every row is the pre-edit tree's own.* The pre-edit tree
  (`claude-cloud` b0cbe1c) was exported and run in each class, not
  inferred from this one. Over 168 configurations -- NumPy 2.5.3 under
  eighteen settings of `OPENBLAS_CORETYPE` and four levels of
  `NPY_DISABLE_CPU_FEATURES` (everything on; without the Ice Lake and
  Sapphire Rapids groups; without AVX-512; without AVX2 too), and 1.26.4,
  2.0.2, 2.2.6 and 2.3.5 under six cores and four levels each, with
  Katmai, Prescott, Nehalem, Sandybridge, Haswell, SkylakeX and Cooperlake
  the kernels that ran (OpenBLAS 0.3.23 to 0.3.34) -- that tree renders
  the first column of the row this tree's test picks, bit for bit, and the
  test finds its key and passes its three checks, in every one. The 1.91.0
  commit (116863b) renders the same first column in the four classes: the
  pre-edit bytes are the previous release's too. A CPU without FMA takes
  other variants of glibc's libm; forced with `GLIBC_TUNABLES`, float64
  `pow`, `exp`, `sin` and `cos` move, and `powf`, the key and the frames
  do not. (4) *A class with no row fails, by name.* A key in no row is one
  FAIL -- "this machine's class has a row in FRAME_DIGESTS" -- beside a
  `note` that says which half of the key the table knows, and the
  machine's own row printed ready to paste; it is never skipped, because
  there the test cannot say whether a picture moved. A key WITH a row
  whose frames are not the row fails its scene's check and prints the row
  as rendered. Not measured, and each expected to be a class of its own:
  another C library's `powf` (Windows, macOS, an older glibc), ARM, a
  NumPy on another BLAS or on none. A row is to be added only from a tree
  that passes in a class already listed: that is what makes the bytes it
  prints a machine's and not a moved picture's. (5) *Not a tolerance, and
  not the zip.* The pin stays bitwise in every class; a moved pixel fails
  as it did. Comparing against the previous release in its zip
  (`test_render._prev_engine`) would be machine-independent, but it skips
  wherever the zip is absent -- the container among them -- and the table
  needs nothing beside the tree. (6) *It still bites.* Two mutations, each
  run in the four classes and failing in all four: the Image Texture's
  flat U 1e-5 high (`core/nodeeval.n_tex_image`: ONE pixel of `textured`
  crosses a checker edge and that scene's check fails, while `demo` and
  `mirror`, which hold no texture, stay green), and `safe_pow` one ulp
  high (eleven pixels of each frame, 2.4e-07 to 4.8e-07: all three fail).
  The path of a class with no row was run by taking this machine's row
  away, and with `np.power` itself an ulp high (the kernel half named from
  the table, the `pow` half "in no row"). *Seen and left as found:*
  `test_projector_old_modes_frozen` in the same module holds the
  Projector's CYLINDER and SPHERE lanes to digests of float32 `arctan2` /
  `arccos`, which have the same two answers, so its six checks (two modes
  by three axes) fail without the AVX-512 loops, before and after this
  entry. The pre-edit tree's own digests there are 3794526883 / 2315255300
  / 2522480949 (CYLINDER X / Y / Z) and 2796990144 / 4009201160 /
  2923579709 (SPHERE), the same on NumPy 1.26.4 to 2.5.3. The terrain
  tool's "is the 1.91 mesh bitwise" hashes (`test_r253_terrain_tool`) are
  a third pin of the kind: Dunes and Volcano fail without the AVX-512
  loops, Crater too once AVX2 is off, before and after this entry. Those
  pins are a lane's and a mesh's, not a frame's, and how to widen each is
  its own decision. In the container `run_all` is otherwise what it was.
  On the tree the pin was measured on (`claude-cloud` 1e6239c with the
  entry above), as the machine runs it (a fused kernel under the AVX-512
  loops), 12,939 checks pass before and after and the failing lines are
  the 32 zip lines. Forced into the other three classes, the module's
  three checks turn from FAIL to ok and no other line of the suite
  changes: on the unfused kernel 12,939 pass where 12,936 did; on the
  fused one without the AVX-512 loops 12,931 where 12,928 did; on the
  unfused one without AVX2 either 12,930 where 12,927 did -- and what
  still fails there beside the zip lines is the six Projector lanes and
  the two or three terrain hashes named above, as before. Merged onto
  `claude-cloud` e228c85, where the suite has grown, the same four classes
  pass 13,106, 13,106, 13,098 and 13,097 checks and fail those names and
  no other.
- **A master shader inside a node group bumps from its Bump Height (the
  socket was read by nothing there, and the surface drew flat).** The
  master's Bump Height is sugar: nothing shades from the socket.
  `core/nodeeval.desugar_master_bump` rewrites a linked one, at the top
  of every render, into a real Bump node -- Height takes the link,
  Strength the master's Bump Strength, and the master's Normal then
  reads the new node -- and it walked `graph['nodes']`, the material's
  OWN tree. A group node carries its tree under `'group'`, which the
  rewrite never entered, so a master inside a group kept a linked socket
  that nothing read. The Bump-in-a-group entry above saw it from its
  probes and left it for its own fix. *Wrong picture or fallback: the
  wrong picture*, and a quiet one: no warning and no refusal, a height
  map wired in and a flat surface drawn. The probe (the demo floor
  through the real `serialize_tree`, `R.render` on the CPU) put ONE
  master with one height -- Gradient Texture (Quadratic Sphere) -> Math
  Multiply 6.0 -> the master's Bump Height -- once in the material's own
  tree and once inside a group whose Surface output feeds the Material
  Output. In the material's tree, linked and unlinked are 0.89 apart
  over 4435 pixels of the floor at 128x96. Inside the group they were
  ONE frame, bit for bit, and the group frame was 0.89 off the top-level
  one: 0.60 over 27678 pixels at 320x240, 0.37 over 110679 at 640x480,
  0.36 see-through. No pre-pass runs at 128x96; the height was simply
  dropped. (1) *Every tree.* The rewrite enters a group node's tree by
  calling itself, to any depth, and rewrites each tree where it stands:
  the synthesized Bump goes into the tree its master sits in. That is
  the tree holding the nodes its links name -- a height chain inside the
  group, or the Group Input when the height, the strength or the normal
  underneath is handed in through the group's interface -- and the tree
  the group's evaluator shades (`group_evaluator`), so the master's
  Normal finds the node; the material's own tree gains nothing. The
  exporter serializes a group's tree once per group NODE, so two
  instances of one group are two trees and each gets its own Bump. (2)
  *An id of that tree.* The node is filed under
  `nodeeval.synth_bump_id(master id, tree path)`: `__bump_` ahead of the
  master's own step, behind the path of its tree. `__bump_Halcyon` in
  the material's tree, the id it always had; `Group/__bump_Halcyon`
  inside the group node 'Group', `Group.001/__bump_Halcyon` in a second
  instance of that group, `Outer/Inner/__bump_Halcyon` two levels down.
  It is the id `export.node_id` gives a node NAMED `__bump_Halcyon` of
  that same tree, escapes included, so it can only ever meet a node of
  its own tree -- which the rewrite looks for before it writes, the rule
  the material's tree always had (a node already under that id is not
  written over). `__bump_` ahead of the WHOLE id, the old line applied
  to a path, spells a node of another tree (`__bump_Group/Halcyon` is a
  'Halcyon' inside a group node named '__bump_Group'), where no look at
  this tree could see the clash. A tree built by hand whose ids carry no
  path gets its group's path in the synthesized id all the same, so the
  group's Bump cannot take the field of a like-named master of the
  material's own tree. The core may not import the exporter (it imports
  `bpy`), so the separator moved house: `ID_SEP` is defined in
  `core/nodeeval.py` and `export.py` imports it -- one constant, as
  before. (3) *The any-depth roads were already there.* That entry built
  them and the synthesized node rides them like any Bump of a group:
  `has_bump` finds it, `bump_heights` evaluates its height in the
  group's context with the group's inputs bound and hands it out under
  its id, `_bump_field_for` asks by that id. On a frame large enough to
  pre-pass (320x240 on the demo floor) the field `Group/__bump_Halcyon`
  is built and the frame is the top-level frame bit for bit -- itself
  the frame of the material shaded in ONE batch. With that field
  withheld the frame is cut along the batch boundary (0.31 over 230
  pixels on rows 53-54; 0.15 over 1472 pixels on six rows at 640x480),
  so the frames can tell. The layered road's per-rank fields, the band
  scissor's context row and the banded `job.bump_field_source` take the
  node the same way. (4) *Idempotent as it was.* The rewrite consumes
  the Bump Height link, in a group's tree as in the material's, so the
  pass at the top of the next render finds nothing to do: three more
  passes leave a rewritten graph exactly as the first left it, and one
  scene rendered three times builds the one field each time and draws
  one frame. *The GPU device.* Nothing emits a group's tree, so such a
  material shaded on the CPU before and does now; the plan's line for it
  follows the picture. A master in a group with nothing bending its
  normal is refused as "no GLSL emitter for ShaderNodeGroup". One whose
  normal IS bent is refused a step earlier, as "the graph bends the
  shading normal outside the master shader's Normal socket" -- the line
  a Bump wired by hand to that master inside the group always printed,
  and the line the Bump Height prints now that it bends anything.
  *Pictures that change*: a material whose master -- Halcyon, BI
  Material, 3ds Max Standard or Raytrace, Console -- sits inside a node
  group with its Bump Height linked. It is bumped now: the frame the
  same nodes draw outside the group, and the frame of the same height
  wired by hand into a Bump node inside the group. Nothing else moves:
  21 frames whose masters all sit in the material's own tree (no height,
  the sugar, the sugar over a Normal chain with its Scale and Strength
  fed, that entry's Bump inside a group; 128x96, 320x240 and 640x480;
  see-through; banded; an unlinked Bump Strength of 0.4) are bit for bit
  the frames the tree drew before. Pinned in
  `test_master_bump_height_inside_a_group_is_a_bump_of_that_group` (34
  checks), through the real `serialize_tree`, with a spy on the pre-pass
  so every frame says which fields it built: the pair at 128x96 (linked
  and unlinked two pictures in the material's tree and, now, inside the
  group; the group frame the top-level frame); the rewritten graph read
  off the scene (one Bump, in the group's tree, under its path, its
  Height the chain's Math node, the master's Normal on it, the link
  consumed, every link naming a node of its own tree); 320x240 with the
  field built under 'Group/__bump_Halcyon', the top-level frame and the
  one-batch frame bit for bit, and the field withheld cutting two rows;
  640x480; the same height wired by hand into a Bump node of the group;
  a group that takes the height's Scale, the Bump Strength or the Normal
  underneath through its interface, one at a time and all three on a
  pre-passed frame, and the height itself handed in; an unlinked Bump
  Strength; ONE group instanced twice at two Scales (two fields, one
  three times the other, each instance drawing its own height); a master
  named 'Halcyon' in the material's tree and another inside a group (two
  Bumps, two fields); a group in a group ('Outer/Inner/__bump_Halcyon');
  idempotence on the graph and on the picture; the see-through floor on
  the layered road; three bands joined against the whole frame; the GPU
  device on the stand-in driver (refused by the hand-wired Bump's own
  line, the CPU's frame bit for bit); each of the five masters rewritten
  inside a group, the two Max nodes with their Bump Amount for a
  strength; the id against `export.node_id` for 40 names and paths
  holding the separator and the escape; a node already under the id left
  alone; a graph spelt by hand with like-named masters in two trees; and
  the rule read from the code (both render entries call the rewrite, one
  place writes the id, one separator). The Bump-in-a-group test's
  `_bump_chain` is `_height_chain` plus its Bump now, and its 20 checks
  pass as they were. Twenty-five mutations each fail the new test: the
  rewrite not entering groups, or one level only; the path not handed
  down, handed down without its separator, doubled, or the parent's
  handed on; the id without its path, with the path twice, or with
  `__bump_` behind the step; the node filed under one id and carrying
  another; the link not consumed; a node already under the id written
  over; the master's Normal not pointed at the node; the Height, the
  Strength link, the Strength default or the Normal underneath not
  taken; the Console master left out; the Max masters reading the
  socket; either render entry skipping the rewrite; `has_bump`, the
  pre-pass walk or the band scissor looking at the material's own tree
  only; and the exporter defining a separator of its own. *Seen from the
  same probes and left for its own fix*: a master inside a group loses
  its shading RATE. `core/render.material_model` reads the model off the
  material's own tree, so for a Gouraud or Flat master inside a group it
  answers the scene's default model (Phong in the probe) and the floor
  shades per pixel: 0.64 and 1.09 off the same master at the top level
  over 4848 pixels at 128x96, nothing linked. With `material_model` made
  to answer the master's own model the group frame is the top-level
  frame bit for bit, and the per-pixel models (Lambert, Phong, Toon)
  agree bit for bit as they stand. NOT verified, here or by the report:
  a real Blender, a real GPU, and the world graph (the same call
  rewrites it; nothing drew one).
- **A Coded Shader in a world's node tree draws its program (it drew a
  black sky: the world export compiled the program and kept no reference
  to it).** The node is offered in world trees (`HalcyonNodeBase.poll`
  takes any `ShaderNodeTree`), and `export.serialize_tree` compiles its
  source there as it does in a material -- into the `programs` dict its
  caller hands it. The world export handed it a dict literal,
  `world.graph = serialize_tree(bw.node_tree, images, {}, warnings)`:
  `_node_props` filed the program in it and the dict was gone on the
  next line. `core/scene.World` had a field for the ground material's
  programs (`ground_programs`, kept a few lines above) and none for its
  own tree's; `core/render.world_color`, the one place a world graph is
  evaluated, built `GraphEvaluator(world.graph, ctx, textures, {})`; and
  `n_halcyon_code` answers zeros for a node with no program. *Wrong
  picture or fallback: the wrong picture, and nothing said.* The probe
  put the real node class (stand-in sockets) through the real
  `export_scene` under the fake Blender and rendered `R.render` at 96x72
  over the demo scene: with Coded Shader (Color) -> Background (Color)
  -> World Output (Surface) and a shader that writes `vec4(0.15, 0.35,
  0.8, 1.0)`, all 2649 uncovered pixels of the 6912 read (0, 0, 0),
  where an RGB node of that colour in the node's place reads (0.15,
  0.35, 0.8). No warning was due: the compile had succeeded. The GPU
  device drew the same black -- its sky pass refuses every world graph
  but a lone Background by name (`gpu/sky.refusal`), so the sky in its
  frame is the CPU evaluator's -- and so did a mirror, whose environment
  term is `world_color` along the reflected ray (339 pixels of the demo
  Ball at `reflect_level` 0.6, up to 0.48 off). (1) *The programs ride
  the World.* `World.programs`, beside `ground_programs`: what
  `Material.programs` is to a material's graph. The world export hands
  `serialize_tree` a dict it keeps and files it there. The generic World
  loop skips the field (it is no setting of the panel's), and
  `presets/skies.EXCLUDED` names it beside `graph`: a sky preset resets
  every field it owns, and must not take the programs from under a tree
  it leaves alone. A scene pickled for the worker pool carries the field
  with the rest of the World, each program recompiling on arrival as a
  material's does (`Program.__setstate__`). (2) *The world evaluator is
  handed them.* `world_color` passes `world.programs` to its evaluator
  (None on a World built by hand: the evaluator takes that as none, and
  such a world reads what it read). Every road onto the sky calls that
  one function -- the background pass and the environment term of a
  reflective material on both devices (both rendered here), a traced ray
  that hits nothing and the fog that fades to the backdrop (read in the
  code, not rendered) -- so there is one line to hand them to. (3) *Node
  groups.* Nothing more was needed: a program is filed under its node's
  graph id (`export.node_id`, the path), `serialize_tree` hands the one
  dict down its group recursion, and `group_evaluator` hands its
  sub-evaluator the same `programs`. A Coded Shader inside a node group
  of the world tree is 'Group/Coded Shader', two levels down
  'Outer/Group/Coded Shader', and each runs its own; the world's own
  "Coded Shader" beside a group's "Coded Shader" are two programs. A
  source that does not compile files none and the export warns by the
  same id ("Group/Coded Shader: line 6: undeclared identifier
  'nothing'"), as it already did. (4) *What the inputs read: the world
  evaluator's context, as every other node of a world tree reads it.*
  Nothing was added to that context for this node. The ray is all a sky
  has: `I` is its direction (the Camera Data node's View Vector) and so
  is `vObject` (the Texture Coordinate node's Generated); `vNormal` /
  `N` and `vView` / `V` are its reverse (Texture Coordinate's Normal,
  Geometry's Incoming); `vPosition` / `P` / `worldPos` is a point 1e6
  along it (Geometry's Position); `vTangent` is a tangent of that
  reversed normal (Geometry's Tangent). What a sky has none of reads the
  context's default, and the default is zero: `vUV` (under each of its
  names) and `vUV2`, `vGeoNormal`, `vBitangent`, `vDepth`,
  `vBackfacing`, `vRandom`, `vScreenUV`, `vCamera`, and `gl_FragCoord`,
  which is (0, 0, 0, 1). Two defaults are not zero: `vColor` is white,
  (1, 1, 1, 1), the context's default where there is no colour layer,
  and `iResolution` is (0, 0, 1). That context carries no clock: `iTime`
  reads 0 and `iFrame` reads 1 on every frame, as the Screen Info node's
  Time and Frame do there (below). `dFdx` / `dFdy` / `fwidth` read zero
  and `dither4x4` hands its value on, for want of a pixel; `discard` is
  not read (a sky has no alpha to cut). None of it raises: the test runs
  all 29 names with the evaluator's STRICT switch on, which lets no node
  error be swallowed. (5) *The GPU device* needed no change and is right
  now. The sky pass refuses the graph by name -- `[Halcyon GPU] sky on
  the CPU: the world node graph is evaluated on the CPU` -- and the CPU
  draws the sky; the deferred plan gives a world that has a graph the
  CPU's environment term (`('CPU',)`), so a mirror on that device
  reflects the coded sky as well. No cache keys on the world graph, so
  none had to learn the source: the plan's signature holds WHETHER the
  world has a graph, not the graph, because no pass bakes the sky or the
  environment term of such a world. An edit of the source re-exports,
  and the next frame's sky and reflections are the new shader's on the
  plan the last frame left. *Pictures that change*: a world whose node
  tree holds a Coded Shader was black wherever the world is evaluated --
  the sky and its reflections -- and is the shader now, on both devices.
  A node nobody has typed in draws the starter shader it opens with: its
  Base Color in bands across the dome (the normal it shades by is the
  reversed ray). Nothing else moves: a world without the node is
  evaluated over the same empty programs as before, and a World built by
  hand has none. Pinned in `test_code_node_in_the_world_tree` (37
  checks), with the real node class through the real `export_scene`: the
  program kept under the node's id, compiled from the source its graph
  entry carries; every uncovered pixel exactly the shader's colour and
  the whole frame the RGB node's, bit for bit; the 1.92.0 export of the
  same tree (the graph, no program) black, so the frames can tell; the
  starter node; a linked uniform under a Background Strength of 1.3
  against the RGB node; the Color wired straight into the World Output
  and the node's own Output as Surface, each the same sky bit for bit;
  the group, the same-named pair summed (the frame of the inner node
  renamed, and of two RGB nodes through the same Mix) and a group in a
  group; the two compile warnings; the 29 inputs, each held bit for bit
  to the stock node that reads the same field in a world tree or to the
  context's default, with those stock outputs checked to exist; a
  mirror, three bands against the whole frame, and the pickled world;
  and on the stand-in driver the refusal, its console line, the device's
  sky bit for bit the CPU's, the mirror within the shading twin's 6e-6,
  and the edited source followed on one plan with every cache left warm.
  Nineteen mutations each fail it: the dict literal back in the export;
  the dict kept but not filed, filed under `ground_programs`, or an
  empty one filed; the evaluator handed `{}` or the ground material's
  programs; `World` without the field; the field left to the sky
  presets; a group's evaluator or a group's export handed a dict of its
  own; the program filed by bare name; the compile warning dropped; the
  context's normal turned along the ray; the coded inputs reading a UV
  or a Generated coordinate the context does not hold; the fragment
  coordinate or a derivative requiring a pixel; the sky pass not
  refusing the graph; the plan not giving a graph world the CPU's
  environment term. *Seen from the same probes and left for their own
  fixes*: three more evaluators are built without the material's
  programs, and a Coded Shader that feeds them reads zero there in the
  same silence -- `core/volume._sample_volume` (a Value node of 0.5 into
  a Halcyon Volume's Density gives extinction 0.5, the coded shader that
  writes 0.5 gives 0: the volume is not there) and the Anime Shader's
  two bakes, `bake_anime_ramp` and `bake_face_sdf` (the Shadow Ramp and
  Face Shadow (SDF) tables come out all zero). A `uniform sampler2D` has
  no road to an image -- nothing in the add-on writes the socket's
  `halcyon_image_key` -- and `texture()` on the unbound sampler raises
  inside `prog.run` (`'numpy.ndarray' object has no attribute
  'sample'`); the evaluator swallows that, the node reads zero, and
  where a material's evaluator errors are printed (`node evaluator:
  ...`) the world road reads neither `ev.unsupported` nor `ev.errors`,
  so a node that raises in a world tree says nothing. And the world's
  node context has no clock, no camera position and no pixel although
  `world_color` is handed the scene's time (for the sky models) and the
  eye: with the scene at 2.5 seconds, frame 61, `iTime` read 0 and
  `iFrame` 1 -- an animated sky cannot be written in any node of a world
  tree until that context carries them.
- **A Bump whose height is fed by another Bump gathers from fields built
  in dependency order (it was cut at every pre-pass chunk, band edge,
  region edge and layer).** The height pre-pass -- `n_bump` differences
  its height toward the +x / +y neighbours shaded in the SAME batch, so
  a material that covers more than one shading batch has its height
  chains rendered to whole-material gradient fields first -- evaluated
  every Bump's Height chain in ONE pass over the material, chunk by
  chunk, and the material's fields exist only once that pass returns
  (`job.bump_fields` is updated after `_bump_height_fields`). A Bump B
  whose Height chain holds a Bump A therefore had A evaluated with no
  field: A fell back to differencing the batch in front of it
  (`_screen_grad`), which there is the PRE-PASS chunk, and A's normal --
  B's height -- was cut at every pre-pass chunk boundary. "Heights are
  per-pixel independent, so chunking here cannot cut anything" holds
  for a height that reads no Bump, and for no other. The Bump-in-a-group
  entry above saw it and left it for its own fix. *Wrong picture or
  fallback: the wrong picture.* The probe (the demo floor through the
  real `serialize_tree`, `R.render` on the CPU): Gradient Texture
  (Quadratic Sphere) -> Math Multiply 6.0 -> Bump A; A's Normal ->
  Separate XYZ (X) -> Math Multiply 6.0 -> Bump B's Height; B -> the
  master's Normal. Against the frame of the material shaded in ONE batch
  (the batch size raised past the floor: no pre-pass at all) it is
  identical at 128x96, where no pre-pass runs; 0.84 off over 461 pixels
  on rows 52-54 at 320x240; 0.98 off over 2946 pixels on nine rows at
  640x480; and with a Bump C on B's normal, 1.35 off over 692 pixels on
  rows 51-54. Every road that entry named was then probed, and each
  had the hole. *Inside groups*: the whole probe in one group, A in a
  group under B, B in a group over A (its height the group's input), a
  group in a group -- 0.84 off on the same 461 pixels, each of them.
  *See-through*: the layered road evaluated its heights `MAX_CHUNK`
  fragments at a time across ALL ranks. One layer of the probe is right
  at the stock chunk size (one chunk holds the floor) and 0.65 off over
  378 pixels on rows 70-72 with its heights in two chunks; TWO layers (a
  second sheet of the material over the floor) are 0.69 off over 8806
  pixels at the stock size, 0.79 over 9135 with three levels -- A
  differenced both ranks' fragments in one batch, two heights colliding
  at every pixel both layers cover: the collision the per-rank fields
  were built to remove, one level down. *Bands*: a band's scissor keeps
  ONE context row past the band, the row a Bump's gradient reads. B's
  gradient on the band's last row reads B's height on that context row,
  which reads A's gradient there, which reads A's height one row further
  -- never rasterised. On a floor of small triangles (the scissor keeps
  whole triangles, so the demo floor's two cover every row whatever it
  asks for and hide this) three bands of the 320x240 frame are 1.01 off
  along rows 79 and 126, the last row of each band that has another
  above it; sixteen-row bands 1.21 off over 337 pixels, the last row of
  every such band; and a see-through floor in three bands 0.60. *The
  render region*: the same reading, in the ring
  `_region_reach` keeps past the rect -- 1.05 off along the rect's edge.
  *The GPU device*: the deferred pass refuses a Bump inside another
  Bump's height chain by name where it could emit the chain, and the
  frame falls to the same CPU road, cut included. A chain it cannot emit
  -- a node group above all -- is evaluated on the CPU into the Bump's
  height image (`gpu/shade._cpu_height_image`), the upstream Bump with
  it: over the whole material in one batch, so the whole frame was right
  (inside the deferred bar on the simulator: one pixel 0.0022 from the
  CPU frame, every other under 1e-5), but a refine pass (adaptive
  anti-aliasing) evaluates the flagged pixels grown by ONE ring, and
  its flagged pixels were 1.27 off at 446 of 3840 on a 160x120 probe.
  (1) *The order.*
  `core/nodeeval.bump_levels(graph)` sorts a material's Bumps into
  levels: the first holds every Bump whose height chain reads no other
  Bump, each later one the Bumps whose height reads only Bumps of the
  levels before it. The walk is `bump_heights`' own (a tree's nodes,
  then its groups', an id counted once), so the two cannot disagree
  about which node an id names. What feeds a height is read from the
  links (`_bump_feeders`), never less than the evaluator reads: every
  input of every node on the way, a group node's output as its inner
  Group Output's socket of that name, a Group Input's output as the
  holding group node's input of that name, else of that position (the
  readings of `n_group` and `n_group_input`). A Bump met on the way is a
  feeder whose HEIGHT is not followed -- by the time it is asked it
  gathers from its field -- and whose Strength, Distance and Normal are:
  a height reads what the Bump it takes reads. The group boundary is
  taken socket by socket because the usual stack of bump groups hands
  the NORMAL down, group after group: read by node, every group of such
  a stack would seem to feed the next one's height and each would cost a
  pass. (2) *The opaque frame.* `_bump_height_fields` builds its fields
  a level at a time, each level's chunks evaluated with every field
  built before it in reach (`ctx.bump_fields`), so the upstream Bump
  gathers from its whole-material field there as it does in the shading
  pass; `bump_heights(ev, only)` evaluates one level's Bumps and passes
  the others over. (3) *The layers.* "Per-fragment pure" is the first
  level's alone, so the layered road evaluates the first level across
  the ranks, as it did, and hands each rank's fragments to that same
  builder for the levels above it (`first=1`), the rank's first-level
  fields filed where the builder looks. (4) *A pixel of context per
  level.* A level-0 gradient reads the neighbour's height; each level
  above it reads one neighbour further. The band scissor keeps
  `len(bump_levels)` rows past the band where it kept one,
  `_region_reach` that many rings, and `_cpu_height_image` grows a
  refine mask by one ring more per level of the node it evaluates. (5)
  *One difference.* The forward differences behind a field were written
  out at both sites; they are `_bump_gradients`, the one place.
  *Pictures that change*: a material in which one Bump's Height chain
  reads another Bump -- the master's own Bump Height fed by a Bump's
  normal included,
  since it desugars to a Bump -- where that material covers more than
  16384 pixels of the frame, on a banded frame, in a render region,
  see-through, or in a refine pass on the GPU device. It draws the
  one-batch frame now, on every one of those roads. *Nothing else
  moves, and nothing else pays*: a material whose Bumps do not feed each
  other's height is ONE level -- the pass it always was, the same
  evaluator calls in the same order. Rendered by the tree before this
  entry and by this one, the frames of five such materials (one Bump;
  two chained through the Normal socket; one group instanced twice and
  four times, each bending the last one's normal; a group in a group)
  are identical bit for bit, opaque, see-through, in three bands and in
  a region, and their pre-passes build the same number of contexts.
  Pinned in
  `test_bump_fed_by_a_bump_builds_its_fields_in_dependency_order` (43
  checks), through the real `serialize_tree`, with a spy on the pre-pass
  that says for every frame whether it ran, for which ids, how many
  times it evaluated the material, and whether any Bump differenced a
  chunk on the way: the levels read from the graphs (the probe at two
  and three levels, the four group arrangements under their path ids, a
  Normal-chained pair and a stack of three groups as ONE level, a group
  with a second output no Bump feeds, a Bump read THROUGH another Bump's
  Normal socket and through its Strength or Distance, a link that passes
  a Reroute, a muted node and a group handing its input straight out, a
  diamond, the master's desugared Bump Height in the material's tree
  and inside a group, a loop of heights cut rather than walked for
  ever); the probe at 320x240 in two passes with
  no chunk differenced, bit for bit the one-batch frame; the old
  pre-pass put back (every Bump in one level) cutting a few rows by more
  than 0.1, so the frames can tell; a finer chunking (`MIN_CHUNK`
  lowered); the link through the Reroute, the muted node and the
  pass-through group, bit for bit the plain probe's frame; 128x96 and
  640x480; the four group arrangements against the top-level one-batch
  frame; three levels; four Bumps on three; the stack of groups in one
  pass; the three-level probe see-through in two layers, its heights in
  more than one chunk, against each rank shaded in ONE batch with no
  field at all, the same inside a group, and two see-through MATERIALS
  wearing it (the floor and the ball: three ranks of mixed materials)
  with a stale field planted under the upstream Bump's key; three bands
  and a region of a floor of small triangles against the whole frame;
  a master INSIDE a group whose Bump Height is fed by a Bump -- the
  group-master entry's rewrite and this order together -- against its
  one-batch frame, and in three bands on that floor; the GPU device on
  the stand-in driver (the probe refused by name and
  the CPU's pre-passed frame bit for bit; a group's Bump drawn by the
  deferred pass and a refine pass's flagged pixels bit for bit the whole
  frame's); and the rule read from the code (one order behind every
  road, one difference). Each road has
  its own "can tell" check with the levels flattened. Fourteen mutations
  each fail it: the builder back to one pass; a level evaluated without
  the earlier fields in reach; every level evaluating every Bump; the
  levels in reverse; the band's context back to one row; the region's
  ring back to one pixel; the height image's mask back to one ring; the
  layered road evaluating every level across the ranks, or starting its
  per-rank builder a level late; the walk never crossing a group
  boundary; a met Bump ending the walk, or passing on its Normal alone;
  a Group Input, or a group node's output, read by node instead of by
  socket. *Seen from the same probes
  and left*: `core/render._promote_clip` evaluates a punch-through
  material's alpha chain `MAX_CHUNK` fragments at a time with no field
  in reach, so an alpha that reads a Bump's normal differences that
  chunk -- read in the code, not probed; and the deferred pass still
  refuses a Bump inside a height chain it could otherwise emit (the
  frame is right, on the CPU).
- **The Coded Shader's matrices, writes through a variable index, array
  fields and integer division.** Four more kinds of source went through
  `shaders/compiler.try_compile` with error `None` and then failed, or
  answered another number, in `prog.run` -- what was left after the name
  check and the six above. Each was reproduced headless first
  (`try_compile(src)`, then `prog.run({}, {}, 4)`) on `claude-cloud` at
  `1e6239c`. (1) *A matrix is one array.* `mat3 m = mat3(1.0, 0.0, 0.0,
  0.0, 2.0, 0.0, 0.0, 0.0, 3.0); Color = vec4(m[1], 1.0);` raised
  `ValueError`. The interpreter held a matrix two ways. `rt.mat` (a
  constructor) builds one (lanes, columns, rows) array -- the shape
  `shaders/builtins.py` documents and every function in it takes -- while
  `Interpreter.zero` (a declared `mat3 m;`) made a Python list of
  columns, and `subscript` and `store` indexed whatever they were handed
  as that list. A column read of a constructed matrix therefore indexed
  the LANE axis: an error at four lanes, and at three, where the shapes
  fit, lane 1's rows without a word (`[1, 0, 0]` for the `[0, 2, 0]`
  asked for). `m[1][1]`, `m[1].xyz` and `m[0] = vec3(0.5)` failed the
  same way, and a declared matrix with its columns written handed
  `mat_mul_vec` a list (`AttributeError: 'list' object has no attribute
  'shape'`). `mat3(...) * v` alone worked, the one road the tests walked.
  The retired generator had it right (`np.zeros((1, C, R))`, `(m)[:, c]`,
  `rt.col_set`); the interpreter that replaced it did not carry that
  over. A matrix is the one array wherever it comes from: `zero` builds
  it, `subscript` reads a column of it (`rt.col`; through a variable
  index a column per lane, where it read column 0 for every lane) and
  `store` writes a column into it (`rt.col_set`, the value converted as a
  vector's is). With that a matrix also survives what it could not: being
  returned from a function, written by an `out` parameter, held in a
  struct or an array, or declared and then assigned whole -- each
  selected the array against the list (`ValueError`). *`m.xyz` is a
  compile error*, `line 4: 'xyz' of mat3: a matrix is indexed first
  (m[0].xyz)`, where it raised `ValueError` when run (and `m.x` answered
  column 0, typed float). `shaders/names.py` stepped around its swizzle
  check for a matrix -- and for an array in the same condition, so `a.x`
  of a `float a[3]` and `a.k` of an array of structs are refused with it
  (a NumPy error and a `TypeError` when run). A loosely typed value is
  left to the run, as everywhere. (2) *A write through a variable index.*
  `float a[3]; for (int i = 0; i < 3; i++) { a[i] = float(i) * 0.25; }`
  raised `ShaderError: dynamic array writes are not supported`; `vec3 v;
  int i = 1; v[i] = 0.5;` and `mat3 m; int i = 1; m[i] = vec3(0.5);` the
  matching messages. `store` read the index off the tree (`_const_int`: a
  number written in the brackets and nothing else) where `subscript`
  evaluates it and chooses per lane. `store` evaluates it too: the lanes
  whose index is element j are the lanes element j is written for
  (`Interpreter.indexed`), under the mask of the branch the write stands
  in -- for an array's element, a vector's component and a matrix's
  column, through any nesting (`a[i].y`, `a[i][j]`, `s.w[i]`, `m[c][r]`).
  The index is clamped into range as the read's is, so what is written
  through an index is what a read through it finds. *One read changed
  with it.* A vector's per-lane read clamped an index below zero and not
  one past the last component: there the lanes that DIFFERED read
  component 0 while lanes that agreed read the last, so one index read
  two components depending on its neighbours in the batch. It is clamped
  at both ends, as an array's always was. (An index out of range is
  undefined in the language, and no run in the suite moves by it: below.)
  *An index is evaluated once.* A write through `a[i].y` reads `a[i]`,
  changes it and writes it back, and `a[i] += x` reads before it writes:
  each step names the element again, which cost nothing while the index
  could only be a number. `Interpreter.hold` evaluates the indices a
  target is reached through once and keeps the values in the node, so
  `a[i++].y = x`, `a[i++].x += x`, `++a[i++].x` and an `out` argument
  `f(i, a[i])` whose call moves `i` each name one element. *Two writes
  stay refused, as they were*: a NUMBER written past the size (`a[3] = x`
  of a `float a[3]` raised `IndexError`, and no driver compiles it:
  `index 3 is out of range for float[3]`), and a variable index into what
  holds no elements (`float f; f[i] = x`) -- which is what keeps an array
  sized by a named constant from going wrong in silence, below. (3) *A
  struct's array field.* `struct S2 { float w[3]; }; S2 s; s.w[1] = 0.5;
  Color = vec4(s.w[1]);` raised `IndexError`. `Parser.parse_struct`
  records each field as (type, name, array size) and nothing read the
  third: `Interpreter.zero` built ONE float for `w`, and the name check
  typed `s.w` float -- so `s.p[0].y` of a `vec3 p[2]` was refused at
  compile time as `swizzle 'y' out of range for float`, a text every
  driver compiles. `interp.field_type` is the field's type with its size,
  and the five readers ask it: the value a declared struct starts with, a
  field read, a field write, the constructor (`S(float[3](...), 2.0)`)
  and the name check. (4) *Signed integer division.* `int a = 7; int b =
  2; int c = a / b;` gave 3.5 and `-a / b` -3.5 with no error; `ivec2(7,
  9) / 2` gave (3.5, 4.5) and `7 / 2 * 2` gave 7. `Interpreter.binary`
  cut the quotient for `uint` alone. On the GPU road the same text is
  inlined in the deferred pass and a driver answers 3, -3, (3, 4) and 6,
  so the two roads disagreed on any Coded Shader that divides two ints.
  The quotient of two ints is cut toward zero (`np.trunc`), a vector's
  component by component and `/=` with it; a float on either side is a
  float quotient as it was, and `uint` is as it was. *The cut is all that
  changes.* The quotient is held as before -- NumPy's float64, exact for
  any two 32-bit operands -- so one that had no fraction is the same
  number in the same dtype and nothing computed from it moves by a bit.
  (Held as an int32 it would have: a float vector built from ints alone
  was kept an int array then, a looseness of the constructor's, and
  `vec2(n / 2, n / 2) * 0.1 * 0.7` would have run its two products in
  float64 and rounded the other way. That shader is pinned, and the
  entry below converts the vector.) A zero divisor answers what it did;
  the language leaves it undefined. *What moves.* Asked before anything
  was changed: the unchanged tree ran the whole suite with a recorder on
  that one line of `binary` (on `1e6239c`, and again on `9fc4d43`, whose
  numbers these are). 2912 division sites in 1360 distinct sources divide
  two signed ints (87569 executions, 260 million live lanes, 224 million
  of them with a fraction in the quotient, no negative operand, no zero
  divisor in a live lane and 1.96 million in lanes outside the mask).
  2910 of the sites are the engine's own and they are ONE idiom in 12
  spellings -- the texel address of a data texture,
  `texelFetch(hal_gb_attrs, ivec2(index % side, index / side), 0)`, over
  `hal_gb_attrs`, `hal_gb_tris`, `hal_ink_fn`, `hal_sres`,
  `hal_sss_tree`, `hal_triaux`, `hal_vlight` and `hal_vscreen`
  (`gpu/gbuffer.py`, `gpu/ink.py`, `gpu/glsl_shading.py`,
  `gpu/combine.py`, `gpu/material.py`); the other two are the shader
  tests' own, with exact quotients. In every one the fractional quotient
  went straight into an `ivec2`, whose constructor cut it, so the address
  was already the right texel and is the same texel now: no engine pass
  moves, held bit for bit below. (The cut now comes before the float32
  that constructor passes through, where it came after. Measured on the
  idiom alone, the old address is right for every index of a square
  texture up to 4096 texels a side -- all 16.7 million of them -- and
  puts the last texel of the last row in the next row at 4097; the new
  one is exact. No run in the suite is that large.) What does move is a
  user's Coded Shader that divides two ints with a remainder, on the CPU
  road: `1 / 2` is 0 and `x / 255` of an `int x` is 0 below 255 -- the
  value the deferred pass always gave it. Where the float quotient was
  meant, a float is written: `1.0 / 2.0`, `float(x) / 255.0`. *Nothing
  that compiled and ran is refused, and nothing else moved.* Held four
  ways. The suite, `python3 -m halcyon.tests.run_all` on the unchanged
  head and on the changed tree, the checks compared line by line: on
  `claude-cloud` at `9fc4d43`, 13221 checks pass and 32 FAIL lines are
  printed before (each the `halcyon-1.89.0.zip` no fresh container has
  beside the package), 13287 pass and the same 32 lines are printed
  after, character for character and in the same order; no check is lost
  and the 66 more are the new ones. On `1e6239c`, where the four were
  reproduced, 12863 and 64 before (32 the zip, 32 the bitwise twins a few
  1e-7 off that claude-cloud has since put right), 12929 and the same 64
  lines after. Every compile and every run the suite makes, repeated by
  the unchanged front-end imported beside the changed one, on the same
  uniforms and inputs: 3734 distinct sources through `try_compile` (3217
  compile in both to the same outputs, uniforms, inputs and warnings, 505
  are refused in both in the same words, and the 12 that differ are the
  new tests' own: 7 members of a matrix or an array, 5 array fields) and
  10074 runs of 2418 distinct programs. Outside the four new tests every
  run returns the same arrays -- values, shapes and dtypes -- and the
  same discard mask (one raises, the same error in both): 9547 runs. The
  other 527 are the new tests' shaders, which raised or answered
  otherwise before. 6000 seeded random shaders of what ran before (every
  type, operator, constructor, swizzle, array, struct, function and
  branch, 5760 of them dividing ints) were compiled and run by the
  unchanged front-end and the changed one, and where the two differed by
  the changed one with the cut switched off: all compile and run in both,
  1790 to the same bits and dtypes, and the other 4210 differ by the cut
  alone -- with it off they are the old values bit for bit (1550 of the
  dividing shaders do not move at all: their quotients are exact). 6000
  more with what could not run mixed in (columns and components of
  matrices, declared matrices, functions returning one, writes through a
  variable index, array fields, indices that leave the range): 4294
  raised before and run, 251 were refused before and compile (each the
  array field's `swizzle ... out of range for float`), 446 are the same
  bits, 964 differ by the cut alone, 25 by the clamp of a vector index
  past the end (with or without the cut: the old read restored, they are
  the old values), 20 raise in both -- a number written past the size,
  now by name -- and none is newly refused, newly raises or moves for
  another reason. And 40 of 41 mutations of the fix each fail the tests;
  the one that does not (the struct constructor converting to the bare
  field type) changes nothing a shader can observe, since a conversion
  passes an array of its own base through. What IS newly refused is, by
  intent: a member of a matrix or an array wherever it is written (it
  raised when a lane got there, or swizzled the lanes). What changes
  value is, by intent: a quotient of two ints that has a fraction, and a
  per-lane read of a vector past its last component. *Seen and left*
  (each reproduced; none is one of the four): an array sized by a named
  constant is one element -- `const int N = 3; float a[N];` declares a
  float, because `Parser.parse_array_suffix` reads a number and nothing
  else (`#define N 3` works); reads through any index answer that one
  float, as they did, and a write past index 0 is refused by name, as it
  raised. A compound assignment evaluates its right side twice
  (`Interpreter.assign`): `x += float(i++)` leaves `i` at 2 and adds 1.
  The value of an assignment through a swizzle or a vector component is
  the vector as it was (`a.x = b.x = 1.0` gives `a.x` the old `b.x`);
  through an element, a column or a field it is the value written. A
  matrix times or divided by a scalar still raises (the six above left
  it), and so does `m == m`. A `uniform mat3` on the node is a float
  socket (`KIND_TO_SOCKET`), so the uniform arrives as a float and every
  use of it fails -- unbound it is a zero matrix and works. A number
  written past the size in a READ (`a[5]` of a `float a[3]`) is clamped
  in silence where a driver refuses the text. `uint` division by zero
  answers 0 with a NumPy cast warning, `int` division by zero the
  dividend times 1e30. A float vector built from ints alone was held as
  an int array and arithmetic on it ran in float64 -- why the quotient
  keeps its dtype; it is float32 since, the entry below. Pinned in
  `tests/test_shaders.py` (four tests, 66 checks: each reported shader
  at 1, 2, 3, 4 and 7 lanes; a matrix made 18 ways by used 12 ways; the
  one shape of a declared and a constructed matrix; per-lane writes into
  an array, a vector and a matrix under a branch; 120 seeded programs of
  indexed reads and writes held lane by lane against a plain Python run
  of the same program; the array field built and typed; every pair of
  -12 .. 12 divided; the ends of the 32-bit range; the exact quotient
  that moves nothing).
- **A float vector the Coded Shader builds from ints is float32 (it was
  held as an int array, and arithmetic on it ran in float64).** `int q =
  3; vec2 g = vec2(q, q) * 0.1 * 0.7;` answered 0.21 -- both products
  taken in float64 and rounded once, where the value was stored -- where
  two float32 products answer 0.21000001. The entry above lists it under
  *Seen and left*; reproduced headless (`try_compile(src)`, then
  `prog.run({}, {}, 4)`) on `claude-cloud` at `31aa3f1` with that
  entry's branch merged in (`2dcc4aa`: `claude-cloud` did not hold it
  yet, and this stands on its integer division and its pin). It is the
  drift R243 removed for `float(u)` and for a vector with a Python
  literal in it (`Interpreter.constructor`, `builtins._f32_only`), left
  standing for the vector that has no float part at all. *The root.*
  `Interpreter.constructor` builds a vector with `rt.vec(t.n, *parts)`
  and converted the int, uint and bool vectors alone. `rt.vec` joins the
  parts and `_f32_only` converts a FLOAT join: ints alone stayed the
  int32 array NumPy made of them (uints an int64 one, bools a bool one),
  typed `vec2`. `Interpreter.coerce` converted where the source's BASE
  differs from the destination's, and both said float; so the int32
  array met a float32 scalar, NumPy answered float64, and the chain ran
  on at that precision until something stored it. A vector with one
  float part was float32 already. A declaration hid most of it -- `vec2
  g = vec2(q, q);` converts, because there the two types are the same
  OBJECT and `coerce` held its rule for that case alone -- so the ints
  survived only in a vector USED where it is built, or reached through a
  swizzle, a ternary or a binary result, whose types are equal and not
  the same object. *It was not only a last bit.* The same array gave
  other NUMBERS wherever it was used as the float it was typed: a
  component written into one was cut to an integer (`vec2 g = vec2(1,
  2).yx; g.x = 0.5;` read 0, and so in a function's parameter and after
  a ternary); two of them added or multiplied as int32 and wrapped
  (`vec2(2000000000, 1) + vec2(2000000000, 1)` gave -294967296,
  `vec2(100000, 3) * vec2(100000, 3)` 1410065408); an int no float32
  holds was never rounded (`vec2(16777217, 3).x - 16777216.0` gave 1);
  two built from bools added as bools (`vec2(true, true) + vec2(true,
  true)` gave 1) and one negated raised `TypeError` out of NumPy; and a
  vector written whole to an `out` left the program as a float64 array.
  *Measured before anything was decided.* Converting changes the last
  bit of any shader that builds a float vector from ints and computes on
  it, and the suite compares the GLSL simulators with the CPU bitwise:
  so, first, does any engine pass do that? The whole suite (`python3 -m
  halcyon.tests.run_all`) was run with the unchanged front-end imported
  beside the changed one. Every `Program.run` -- the engine's texts
  reach it from `gpu/shade`, `gpu/sky`, `gpu/ink`, `gpu/craster`,
  `gpu/rtrace`, `core/nodeeval` and `tests/fakedevice`, and from the
  tests that run an engine text by hand; `device.preflight` compiles and
  runs nothing -- was repeated by the unchanged front-end on the very
  same uniforms and inputs, the two results compared as bytes (values,
  shapes, dtypes, the discard mask), and in both every expression's
  value was checked against its type where it was evaluated: outside the
  shader tests 9651 runs of 1977 distinct sources, 47.4 million lanes.
  Those sources were also typed by the front-end's own name check for
  where such a vector is WRITTEN, a lane reaching it or not. The answer
  is yes: the engine's GLSL builds one and computes on it in eight
  statements, three idioms. (a) *A texel address taken from a uv*, four
  times: `ivec2 gbsz = textureSize(hal_gb_ids, 0); ivec2 gbpx =
  ivec2(clamp(screen_uv * vec2(gbsz), vec2(0.0), vec2(gbsz) -
  vec2(1.0)));` in the G-buffer read (`gpu/gbuffer.py`) and the same in
  `gpu/material.py` over `hal_uvgrad` (twice) and `hal_gb_idslin` --
  built by a lane in 1447 of the 1977 sources, 9136 times. `vec2(gbsz)`
  was an int array, and the product, the difference and the clamp ran in
  float64. Nothing computed from it moves: the result goes straight into
  an `ivec2`, whose constructor cuts it, and at a texel centre -- the
  only place these reads are asked, `(x + 0.5) / W` -- the float32
  product is `x + 0.5` to a few parts in ten million, nowhere near the
  half texel that would cut it to another `x`. (On a texel EDGE, `uv = k
  / W`, which the reads never ask for, the float64 product fell a texel
  short of a driver's float32 one at 3364 of the 7199 edges of a
  7200-pixel row; the simulator is the driver's there now.) (b) *An int
  vector over 255*, `Color = vec4(vec3(o) / 255.0, t.a);`, three times
  in the two N64 Video Interface stages (`gpu/stages_vi.py`), written in
  three of the sources and built by a lane in one. It does not move
  either: it is ONE division, and a float64 quotient rounded once to
  float32 is the float32 quotient. (c) *The Blender Internal Voronoi's
  feature point*, `vec3 pt = bi_hpntv(hi) + vec3(c);` of an `ivec3 c`
  (`gpu/procedural.py`, `bi_voronoi`): a hash point plus its cell.
  `vec3(c)` was an int array, the sum was float64, and `pt` stayed
  float64 through the distance taken from it and into the `pa[]` it is
  stored in -- its declaration converted nothing, the sum's type being
  equal to `vec3` and not the same object. **This one moves**, and it is
  all that does. Seven of the sources carry the statement and a lane
  reaches it in one, which returns other bytes in 2 of its 13 runs:
  `tests/test_legacy.py`'s twin of the BI textures against
  `core/bitex.py` (thirteen modes at 256 points), in the two modes that
  reach `bi_voronoi`. Both move TOWARD the CPU's NumPy float32, the
  reference the twin is held to. Voronoi (Minkowski): 215 lanes move,
  every one of them nearer; 41 lanes were the CPU's bits before and all
  256 are now; the largest distance goes from 4.17e-07 to 0. Crackle: 70
  lanes move, 59 nearer and 11 farther; 186 lanes were the CPU's bits
  and 231 are; the largest distance goes from 1.85e-05 to 1.43e-05 (what
  is left is its near-ties, as the test says). The other eleven modes do
  not move by a bit. Both checks pass at their 1e-4 bar before and
  after; what changes is the distance they print (`GPU twin: voronoi
  minkowski  max diff 4.17e-07` prints `0.00e+00`, `GPU twin: crackle
  max diff 1.85e-05` prints `1.43e-05`), and the constructor alone,
  switched on over the unchanged code, gives exactly these values. One
  more engine text held a float as float64 without building a vector:
  the node emitters write a whole-number socket value as an int
  (`Emitter._m` formats 1.0 as `1`), so the Hue/Saturation node's
  `mix(c.rgb, rgb, clamp(1, 0.0, 1.0))` and the Map Range node's
  `max(abs((1) - (0)), 1e-8)` handed a builtin an int beside floats, in
  the two sources that two tests assemble by hand (the deferred pass
  lifts such a constant into a texel, `hal_MV`, and no pass it runs
  holds one). Those are (5) below, and their outputs do not move. (The
  emitters write a whole-number colour the same way -- `vec4 _v18 =
  vec4(1, 1, 1, 1);` in four sources a third test assembles by hand, and
  a palette's unrolled entry `e = vec3(0, 0, 0);` (`gpu/emit.py`) in one
  deferred pass. Each is declared or assigned where it is built, which
  converted it then as now, and nothing was computed from the ints.)
  Every other run outside the shader tests -- all but those 2 of the
  9651 -- returns the same bytes and dtypes from both front-ends, and
  nothing else the suite runs there held a float as anything else,
  before or after. `gpu/stages.py` knew: its gamma-2 resolve writes
  `vec3(float(ii.r), float(ii.g), float(ii.b))` with a comment naming
  the simulator's `vec3(ivec3)` as "a front-end blind spot this pack's
  twin test found" -- the one place an engine pass had met it and
  written around it. The text stays as it is pinned; the comment says
  the blind spot is gone. *The change.* A value typed float is held as
  float32, and five places make it so. (1) `Interpreter.constructor`
  converts a float vector where it is built, whatever its parts -- the
  root, and by itself the reported shader and the Voronoi above. (2)
  `Interpreter.coerce` converts for a float destination whichever type
  object names it: every operand of an operator, every declaration,
  assignment, argument, returned value, field and ternary arm goes
  through it, so nothing typed float reaches arithmetic as another dtype
  whatever built it (its comment already said "a float destination is
  float32, full stop", and the rule held for one object). The other
  three are where an int array, rightly typed int, met a float without
  being converted, and NumPy answered float64 in the same way: (3) a
  write to an array's element was merged in as it came, where a variable
  and a struct's field are converted (`float a[3]; a[0] = q;` left the
  element float64) -- `store` converts it to the element's type; (4) a
  matrix times a vector ran its sum in float64 for an `ivec3`
  (`mat2(0.1, 0.1, 0.7, 0.7) * ivec2(3, 7)` gave 5.1999998 where the
  same matrix times `vec2(3.0, 7.0)` gives 5.2000003) -- `binary` hands
  the vector over as floats; (5) a builtin was handed its arguments as
  they were evaluated, so `clamp(x, 0, 5)`, `max(x, 1)`, `max(q, 0.5)`,
  `mix(x, 3, t)` and `smoothstep(0, 1, x)` came back float64 and
  `outerProduct` of two `ivec2` was a matrix of ints -- `call` converts
  the numbers of a builtin that answers a float, as a driver converts an
  int argument on the way in. A bool is left a bool (`mix` selects by
  one), a builtin that answers an int keeps its ints (`max(a, b)` of two
  ints), and nothing is converted where a sampler is read (`texelFetch`
  addresses a texel with its ints: 16777217 is not a float32). *The
  quotient of two ints is held as it was.* The entry above kept it
  NumPy's float64 so that `vec2(n / 2, n / 2) * 0.1 * 0.7` would not
  move, because from an int32 quotient the constructor would have built
  an int array. That reason is gone -- the vector is float32 from either
  -- and the pin holds bit for bit as it stands; its comment says so
  now, and a second check beside it runs the same chain from the
  quotient through an `int` variable and from the plain `int` the
  float64 stood in for. How the quotient is held is no longer tied to
  this, and is left (below). *What moves, and which way.* No check
  changes state. Two print another number, the Voronoi twin's two
  distances above, both smaller: toward the float32 reference. Beyond
  the suite, what moves is a Coded Shader on the CPU road, and always
  toward float32: 240 seeded chains (a vector of two ints, then two to
  five of `+ - * /` against written floats and int-built vectors, in one
  expression) are NumPy's float32 chain bit for bit, where the unchanged
  front-end answered 107 of them and rounded the other 133 from float64.
  The moves are a last bit wherever the old value was a float64 chain
  rounded once (the vector, a component or swizzle of it, a ternary of
  it, a matrix times it or times an `ivec`, an int stored in a float
  array, a builtin given an int beside a float); the number itself in
  the cases under *It was not only a last bit*; and a float32 array
  where an `out` was float64. The deferred pass, which inlines the same
  text for a driver, always computed the new value: the two roads
  disagreed and now do not. Three side effects of (3), each what a
  variable of the type already did: an `int` element given a float is
  cut (`int a[2]; a[1] = 2.75;` held 2.75), a `uint` element given a
  negative int wraps, and a scalar or a wider vector written to a vector
  element fills or is cut where it raised `ValueError`. The cost is a
  dtype test where a conversion is asked: on 120 of the seeded shaders
  below, 5 % at 64 lanes and nothing measurable at 20000. *Nothing that
  compiled and ran is refused, and nothing else moved.* The change is in
  five methods the run calls; nothing a compile reads is touched. Held
  four ways. The suite on the unchanged head and on the changed tree,
  the checks compared name by name and in order: 13737 pass and 32 FAIL
  lines are printed before (each the `halcyon-1.89.0.zip` no fresh
  container has beside the package), 13785 pass and the same 32 lines
  after, character for character and in the same order; no check is lost
  and the 48 more are the new ones. (What a check prints beside its name
  differs in 39 of them between the two runs: timings, thread ids,
  temporary paths, the order of a printed set -- and the two distances.)
  Every run the suite makes, as above; in the shader tests themselves,
  1588 runs of 792 sources, the 338 that differ are the new checks' own
  shaders. In the changed front-end no value typed float is held as
  anything else at any node of any source the suite runs, the engine's
  included -- but for the one check that hands a float array uniform
  over as ints, to show its uses convert. 6000 seeded random shaders
  written the way a driver accepts and the old constructor mishandled --
  ints, uints and bools as the parts of float vectors, written ints in
  `min`, `max`, `clamp`, `mix`, `step` and `smoothstep`, an `ivec` times
  a matrix, ints stored in a float array, through swizzles, ternaries,
  functions, branches and loops -- were compiled and run by both: 2049
  return the same bits and dtypes, 1367 the same values with a float64
  output now float32, 2567 other values, and 17 raised `TypeError`
  before and run (a bool-built vector negated or subtracted). With the
  five conversions switched off in the changed front-end all 3951 are
  the old result bit for bit, so the five are all that changed (each
  alone, switched on over the old code, moves 1985, 3726, 6, 782 and
  2806 of them in the order above; `coerce` and the array element can be
  switched off alone without one result moving, because what they leave
  is converted at its next use -- they are what keeps the value float32
  where it is HELD, which the check on every node pins). The unchanged
  front-end held some float as another dtype in 5996 of the 6000 and the
  changed one in none. And 6000 more with an int nowhere it went wrong
  (a float part in every float vector, floats to the builtins, float
  vectors to the matrices) -- all 6000 the same bits and dtypes from
  both, and every float float32 in both. 17 of 17 mutations of the
  change each fail the tests: every conversion removed in turn, the
  constructor leaving ints, uints or bools, the argument rule leaving a
  float, an int or a uint, converting a bool, converting where a sampler
  is read, skipping a matrix result, or converting for an int result.
  What changes value is, by intent: a float vector with no float part
  and what is computed from it; an int beside a float in an array
  element, a matrix product or a builtin; and what is stored into an
  array's element. *Seen and left* (each reproduced; all but the last
  three are a value typed INT held as something else, the opposite
  looseness, and none is asked for here): a declared `ivec2 p;` is a
  float32 array (`Interpreter.zero` makes every vector one), so `p.x =
  16777217;` reads back 16777216; `uvec2(4000000000u, 1u)` goes through
  int32 and `u.x == 4000000000u` is false, where a `uint` scalar holds
  it; a builtin typed by its first argument answers a float typed int
  when handed an int beside floats (`int z = clamp(q, 0.5, 2.5);` holds
  2.5, `mix(1, 2, 0.3)` is 1.3 typed int -- a driver takes the float
  overload); the quotient of two ints is still held as float64, exact
  and typed int, so `int h = 6 / 2; h * 1000000000` is 3e9 where an int
  wraps to -1294967296; a bit operator with a float operand (`3 & 1.0`,
  which no driver compiles) is typed float and held as ints until the
  next operator converts it; and a float array uniform is bound as the
  caller hands it -- its uses convert, the binding does not, and on the
  node an array uniform is one socket, as a `uniform mat3` is; and the
  node emitters' `1` for 1.0, which a driver converts and the simulator
  now does too. Pinned in `tests/test_shaders.py`
  (`test_a_float_vector_is_float32_whatever_it_is_built_from`, 47
  checks, and one more beside the exact-quotient pin: the reported
  shader and 23 other ways to 3 * 0.1 * 0.7, each the float32 chain bit
  for bit; the numbers that were wrong; 511 ints across the 32-bit range
  lane by lane; the 240 seeded chains; every value typed float at every
  node of 548 shaders; a float array uniform handed over as ints; what
  stays -- the int, uint and bool constructors, an int-answering
  builtin, `mix` by a bool, a texel address past 2^24 -- and the
  engine's own two: the Voronoi feature point as the float32 sum at 256
  points, and the texel address at every centre of five frame sizes up
  to 16384 across). 42 of the 48 fail on the unchanged front-end; the
  other 6 pin what did not change, and that the two references differ.
- **Findings judged and left.** The review also named: the cel converters
  landing an Emission shader's colour twice (paint plus self-illumination)
  and the Cartoon era presets overwriting the flat-paint extras; the Toon
  Size to Shadow Threshold domain; the BI source table's missing Emit /
  Intensity / Mirror sockets and its Alpha carried with the transparency
  panel off; the console source's pinned specular carried as a value;
  `CONSOLE_GLOSS_MAX`'s comment; the hair part's identity matrix under
  Generated; Generated measured on the evaluated mesh rather than
  Blender's undeformed orco; the cube map's zero-direction sample, its
  atlas memory and the Texture Size Limit halving a cross; the DS
  shininess table cache under a linked Glossiness; the N64 decal's
  one-step rounding, the DS decal's 6-bit expansion, the Saturn half
  order, the Jaguar's /255, the 3DO's 5-bit grid, RenderWare's dual pass
  before lighting, Model 3 fixed shading's dropped sun. Those are
  recorded here as open accuracy items and converter refinements for the
  next round, not fixed in this one: each changes a pinned picture or a
  documented stand-in and needs its own source read.

## README

(no README paragraph: the fixes restore what the README already claims)

## Blender-only

- A real zero-scale object in a live file: the once-printed singular
  warning in Blender's console, the frame drawing, and the viewport
  worker's plan re-use after the object's scale returns.
- The Saturn mesh material reaching the transparent layer plan on a real
  driver (the deferred target still refuses Saturn MESH opacity by name;
  the CPU road draws the half-transparent polygon).
- The Texture Op on the three plain PC shading types compiling and
  matching on a driver (the simulator twin is 1.8e-7).
- Model 2 / Model 3 masters from a real 1.90 .blend migrating with their
  snapped exponent, and the PC node's Colour Vertex toggle hiding the
  Vertex Color Mix socket at once.
- A moved object's procedural texture following it in the live viewport
  without a mesh edit (the plan signature carrying the frame).
- The field pair itself on the reporter's machine: with 1.92.0 on, the
  Debug panel's red "Blender job running" line while the F9 panels are
  grey names the job (RENDER means a render that never ended -- Esc in
  the render window), and `bpy.app.is_job_running('RENDER')` in the
  Python console says the same.
- The Coded Shader's undo road in a live node editor (this fix was
  proven against the fake bpy alone): a Language or Text edit, then
  Ctrl+Z taking the whole edit back in one press and Ctrl+Shift+Z
  bringing the new sockets and their links back after the main loop's
  next pass; an undo from a later step onto the flagged one leaving the
  redo stack intact; a file saved flagged opening whole, in the UI and
  under `-b`.
- The Coded Shader without its Auto Compile box in a live node editor:
  the sidebar (N) showing the Available inputs list alone, and a 1.92.0
  file with the box unticked opened in the UI. (The load was run in a
  real Blender 5.2.2 as the `bpy` module, background, which is also the
  first real-Blender sight of the entry above's background road: the
  node saved flagged came back rebuilt by `load_post`.)
- The Coded Shader's one source on a real driver. The fix is proven on
  the simulator and, through the registered node and `export_material`,
  in a real Blender 5.2.2 as the `bpy` module in background (the
  Text-sourced node 1.46 off before, 1e-5 after) -- never on a GPU and
  never in the UI. Open there: a Text-sourced coded shader's pass
  compiling and matching on the driver; an edit in the Text Editor
  followed by a render drawing the new shader (one re-plan per edit);
  the emptied Text's line in the node's error box at the node's width,
  and the same line in the F12 report; and a 1.92.0 file whose Coded
  Shader holds an empty Text opening to zeros with that warning.
- Same-named Coded Shaders across trees, in the UI and on a driver.
  The fix is proven headless and in a real Blender 5.2.2 as the `bpy`
  module in background (a real node group holding a default-named Coded
  Shader beside the material's own: 1.52 off before, bit for bit after)
  -- never in the UI and never on a GPU. Open there: the viewport and
  an F12 of such a file drawing each node's own shader; two coded
  shaders in one tree named 'Coded Shader' and 'Coded_Shader' compiling
  and matching on the driver under the ordinal tag
  (`_cn2_Coded_Shader_main`); and the F12 report naming a group's
  failing Coded Shader by its path ("Group/Coded Shader: ...").
- A Coded Shader named in another script, on a real driver. The fix is
  proven on the simulator, on the device's pre-flight, on glslang 15.1
  (the pass as GLSL 4.30) and, through the registered node and
  `export_material`, in a real Blender 5.2.2 as the `bpy` module in
  background (refused before, 2.0e-5 after) -- never on a GPU and never
  in the UI. Open there: a node renamed 'Señal' (and one in Cyrillic or
  CJK) in a live file compiling and matching on the OpenGL, Vulkan and
  Metal backends under its code-point tag (`_cnSeu00F1al_main`), with no
  "shading on the CPU: the driver rejected ..." line in the console; and
  the double underscores that names with runs of punctuation make as
  they always did (`_cnGroup__Coded_Shader_main`), which glslang takes
  with a warning under desktop and Vulkan rules and which no driver and
  no Metal toolchain has been asked about.
- A Bump inside a node group in a real file. The fix is proven
  headless alone (stand-in trees through the real `serialize_tree`, CPU
  frames; bands rendered one after another through `render(band=...)`
  in one process; the GPU device on the stand-in driver) -- never in
  Blender and never on a GPU. Open there: an F12 at a real frame size
  of a material whose Bump sits in a real node group, against the same
  nodes ungrouped; the same frame through the worker pool, where each
  band is its own process; the viewport of such a material on the GPU
  device printing its refusal ("no GLSL emitter for ShaderNodeGroup")
  and drawing the CPU's frame; and a group whose interface holds two
  sockets of one name feeding the height.
- The llvmpipe session itself (Blender 5.2.2, OpenGL backend, Mesa
  llvmpipe): the three lines gone, `raster split` and `shade split`
  printed in their place, and the Timing Breakdown naming the stripped
  uniforms. NOT run in this round -- the stripped-uniform fix is proven
  on the stand-in driver (`fakedevice.installed_gl`) only. Open on the
  real one: what a linker strips beyond the names a text never reads (a
  uniform whose every use is dead code -- the door takes those the same
  way), whether `uniform_sampler` raises for a stripped sampler on that
  backend (it is skipped if it does), and a Vulkan session printing no
  stripped line at all.
- A colour wired straight into a shader socket in a live node editor
  (proven against hand-built graphs alone): Blender drawing the link
  valid and `serialize_tree` exporting it with the source socket's own
  type, for the Material Output's Surface, a Mix Shader input and the
  World Output; the F12 console's `shading on the CPU:` line naming the
  wiring on the GPU device; the same file in EEVEE beside it, where the
  surface under the glow is black, as Halcyon's is since the
  no-diffuse entry above.
- The emission slot routing on a real driver (proven in the simulator,
  on the fake driver and under glslang alone): an Emission node and a
  texture wired straight into Surface shading on the GPU device with no
  `shading on the CPU:` line and matching the CPU device's frame; a
  Mix Shader with one input left empty over an Emission doing the same;
  the routed surface seen in a ray-traced mirror (the sweeps never ran
  on the fake driver).
- The Coded Shader's name error in a live node editor (proven against
  the fake bpy alone): the box under the Compile button reading `line 6:
  undeclared identifier 'nothing'`, the operator's red report in the
  status bar, the export's warning in the render's console; how the
  longer varying message (`... (declare it: in vec3 vNormal;)`) reads at
  the node's width, where a label is clipped rather than wrapped; and a
  cbuffer's members arriving as sockets on an HLSL node. NOT run on a
  driver: what one does with the two old texts -- the pass that carried
  `vec4(nothing)` behind an untaken branch (a compile error by GLSL's own
  rule) and the pass that read the frame shader's `vUV` (which compiles,
  and draws what the CPU does not). `tests/glsl_strict.py` was not run in
  this round: the machine had no glslang.
- The once-declared shadow atlas on a real driver. The fix is proven on
  glslang 15.1.0 (every shadowed pass `tests/glsl_strict.py` plans, as raw
  GLSL 4.30), on the simulator and on a stand-in `gpu` module for the
  hand-over -- never on a GPU. The raw text reaches a driver through the
  legacy `GPUShader(VERTEX, fragment)` constructor alone, which
  `_compile_dynamic_miss` calls only when `gpu.types` has no
  `GPUShaderCreateInfo`; no Blender session ran in this round. Open there:
  a build that still takes that road compiling a material pass under two
  or more shadowed lamps, and what its driver says to the 1.92.0 text (a
  redefinition by GLSL's rule; no driver was asked).
- The Coded Shader's warning box and a Shadertoy entry point in a live
  node editor (proven against the fake bpy and the headless evaluator
  alone): `in vec3 vNormals;` drawing the INFO line under the Compile
  button and the corrected name clearing it; a `void mainImage(out vec4
  fragColor, in vec2 fragCoord)` source growing a Frag Color socket on
  Compile, keeping its links through the rebuild, and shading the pixel
  coordinate in an F12 frame on the CPU road (the deferred pass refuses
  it by name). A wrong argument count or two vector widths in a coded
  shader of a real file: the red line in the node, and the frame shading
  without that node's program rather than failing halfway.
- The Coded Shader's matrices, indexed writes and integer division in a
  live file (proven against the headless evaluator alone): a node whose
  source declares a `mat3`, writes its columns in a loop and multiplies a
  normal by it, shading an F12 frame on the CPU road; `m.xyz` drawing
  `line N: 'xyz' of mat3: a matrix is indexed first (m[0].xyz)` in the
  box under the Compile button. NOT run on a driver, and the claim that
  matters most: a Coded Shader that divides two ints (`int cells = 7; ...
  cells / 2`), inlined in the deferred pass on a real GPU, against the
  CPU frame of the same file -- the cut toward zero is C's rule and the
  one this front-end now follows, but no driver was asked here, and what
  one answers for a NEGATIVE dividend is the case to look at. A matrix in
  a coded shader never reached a driver either: a matrix uniform is
  refused for the deferred pass by name, a local matrix is inlined as
  written.
- The Coded Shader's float vectors on a real driver (proven against the
  headless evaluator alone, with NumPy's float32 as the reference): a
  node whose source is the reported `int q = 3; vec2 g = vec2(q, q) *
  0.1 * 0.7;`, inlined in the deferred pass on a real GPU, against the
  CPU frame of the same file. No driver was asked here. The claim is
  that a driver's two float32 products are the CPU road's now, where the
  CPU road rounded once from float64; what a driver's own compiler does
  with a chain of written constants (folding it, or contracting a
  multiply and an add) is its business, so the case to look at is a
  chain fed by a varying -- `vec2(int(vPosition.x), 3) * vUV.x * 0.7` --
  on both devices. Also open, and the engine's own: a Blender Internal
  Voronoi texture (any distance metric, and Crackle) on the GPU device
  against the CPU frame -- the simulator's feature point is the float32
  sum a driver takes now, and its twin of `core/bitex.py` is exact for
  Voronoi where it sat 4e-7 off, but no driver drew it here.
- The release machine itself, where the zips are (the cloud-container
  fix was proven on the container's one Xeon, with the kernel varied by
  `OPENBLAS_CORETYPE` and the previous engine stood in for by this
  repository's own trees): the `note` line of
  `tests/test_r253_kernels.py` there, and under Blender's own Python,
  naming which matmul each NumPy has; and every real zip from 1.70.0 to
  1.89.0 loading through `_prev_engine` with its sky ray chained -- no
  `note` naming a zip whose `core/render.py` does not hold the line
  exactly once -- with the identity pins as they were. NOT run in this
  round: the container has no zip. If that machine's line says "an FMA
  kernel", its sky twins were red before this round and are green now;
  the three mapping frame digests pass there if its class is one of the
  table's four (the next item).
- Any machine but the container, for the frame-digest table. The four rows
  of `FRAME_DIGESTS` were each produced on the container's one Xeon
  (Linux, glibc 2.39), by forcing the BLAS kernel and narrowing NumPy's
  dispatch; no second machine was run. NOT run in this round: a real
  AVX2-only CPU and a real pre-Haswell one (under this glibc, expected in
  the table: the forced classes are their arithmetic, and glibc's variants
  for a CPU without FMA were forced too), Blender's own Python and NumPy,
  Windows, macOS and ARM (each expected to be a class of its own). There
  `test_mapping_frame_twin` either finds its key and passes, or fails
  "this machine's class has a row in FRAME_DIGESTS" and prints the row, to
  be pasted only from a tree that passes in the container. A machine
  without the AVX-512 loops also fails the six Projector lane checks of
  `test_projector_old_modes_frozen` and two or three of the terrain tool's
  mesh hashes, as it did before this round's table: those pins were not
  widened.
- A master inside a node group, bumped, in a real file. The fix is
  proven headless alone (stand-in trees through the real
  `serialize_tree`, CPU frames; bands rendered one after another through
  `render(band=...)` in one process; the GPU device on the stand-in
  driver) -- never in Blender and never on a GPU. Open there: an F12 of
  a material whose Halcyon Shader sits in a real node group with a
  texture on its Bump Height, against the same nodes ungrouped, at a
  small frame and at a real frame size; the same for the BI Material,
  the two 3ds Max nodes and the Console node, which were rewritten here
  on the graph alone (no frame of them was drawn inside a group); the
  same frame through the worker pool, where each band is its own
  process; the viewport of such a material on the GPU device printing
  "the graph bends the shading normal outside the master shader's Normal
  socket" and drawing the CPU's frame; a World whose node tree holds
  such a group (the same call rewrites the world graph; nothing here
  drew one); and a lightmap bake of such a material (`bake_lightmap`
  makes the same call; no bake was run).
- A Coded Shader in a world's node tree, in Blender and on a driver. The
  fix is proven headless alone (the real node class on stand-in sockets
  through the real `export_scene` under the fake Blender, CPU frames,
  the GPU device on the stand-in driver) -- never in a Blender session
  and never on a GPU (no Blender and no `bpy` module in the session this
  was written in). Open there: a Coded Shader added to a real world tree
  and an F12 of it (the starter shader's bands on the sky, then an
  edited source after Compile); a real `bpy.types.World` whose tree
  holds the node inside a real node group; the viewport following an
  edit of the Text after a re-export; the GPU device's console line
  (`[Halcyon GPU] sky on the CPU: the world node graph is evaluated on
  the CPU`) beside a sky drawn by the CPU; and the F12 report carrying a
  world Coded Shader's compile error by its node's name.
- A Bump fed by another Bump in a real file. The fix is proven
  headless alone (stand-in trees through the real `serialize_tree`, CPU
  frames; bands rendered one after another through `render(band=...)`
  in one process; the GPU device on the stand-in driver) -- never in
  Blender and never on a GPU. Open there: an F12 at a real frame size
  of a material whose Bump takes its Height from another Bump's Normal
  (through a Separate XYZ, a Dot Product, anything), against the same
  frame at a size small enough to shade in one batch; the same frame
  through the worker pool, where each band is its own process, on a
  mesh of small triangles; Ctrl+B over it; such a material see-through
  over itself; and on the GPU device, the console's refusal line ("a
  Bump node inside another Bump's height chain is not in the deferred
  pass yet") with the CPU's frame, and an Adaptive anti-aliased frame of
  a material whose upstream Bump sits in a node group.
- A closure with no diffuse lobe in a live file (proven against
  hand-built graphs and a headless worker pool alone): a masked Mix
  Shader (Diffuse, Emission) at 1280x960 or larger in a real F12, in
  process, across the worker pool and under Ctrl+B, with no edge through
  the glow; a Principled BSDF at Metallic 1 from a real export (the
  weight-0 DIFFUSE lobe the node always hands on) drawing its highlight
  on black beside EEVEE's reflecting metal; a lone Glossy BSDF on a real
  driver matching the CPU device (the simulator twin is 5.1e-5); a
  Transparent BSDF alone under Transparency None on the driver.
- A mix in a live file (proven against hand-built graphs alone): two
  Halcyon Shaders of different models mixed by a painted mask drawing
  ONE model in every band and region and holding it as the camera moves;
  the console line when a graph will not evaluate at the triangle
  centroids (each batch then votes for itself); a Principled BSDF at
  Metallic 0.5 beside EEVEE.
