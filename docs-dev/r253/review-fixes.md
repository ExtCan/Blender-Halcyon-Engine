# R253 review fixes: what the adversarial re-read found and what changed

The round's review (three lenses per feature, three refuters per finding)
and the console-shader deep re-read the field log asked for. Each item
below was reproduced headless before it was fixed and is pinned in
`tests/test_r253_review.py` (registered in `run_all`); the Coded
Shader's name resolution is pinned in `tests/test_shaders.py` as well.

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
  *This changes pictures.* On the CPU device such a
  material rendered its fallback surface and now adds the wired colour
  as emitted light, without the fallback's highlight (a plain closure
  with no glossy lobe has specular level 0 -- the Emission node's
  picture); on the demo frame every pixel off the material is bitwise
  what it was. On the GPU device it rendered the colour as a lit albedo
  and now draws the CPU's picture: with the reason printed once where
  the plan refuses, on the driver for the lone link (the next entry). A
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
  next entry. And the frame both devices draw is Halcyon's Emission
  node's, not Cycles': with no DIFFUSE lobe in the closure
  `closure_to_surface` leaves the material's fallback diffuse LIT under
  the glow, where Blender shades black. That rule also reads its batch
  (`np.any(diff_w > 1e-6)`: a pure-emission pixel has diffuse level 0
  when a diffuse pixel shares its batch and level 1 when it is alone --
  seen at the function, not hunted in a frame). Recorded, not changed:
  every Emission-node material's picture hangs on it. Pinned in
  `test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads`
  (47 checks, since the routing took the lone link's plan answer to its
  own test) for a colour, a value and a vector: the CPU frame bitwise
  the Emission node's; the wired colour arriving as exactly itself over
  the fallback (frame(colour) - frame(black) within 1e-6 on every pixel
  of the Ball); and the road the refusal names -- a Halcyon Shader in
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
  comparison in place of a checksum) is a decision left open. (b) *The
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
  surface under the glow is black and Halcyon's is its lit fallback
  (the open item above).
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
  the three mapping frame digests pass there only if its NumPy also runs
  the AVX-512 loops.
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
