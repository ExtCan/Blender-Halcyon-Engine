# R253 review fixes: what the adversarial re-read found and what changed

The round's review (three lenses per feature, three refuters per finding)
and the console-shader deep re-read the field log asked for. Each item
below was reproduced headless before it was fixed and is pinned in
`tests/test_r253_review.py` (registered in `run_all`).

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
  silent; the old tooltip. *Seen from the same probe and left for their
  own fixes*: a colour wired STRAIGHT into the Material Output's Surface
  (a plain RGB node does it, no coded shader needed) qualifies for the
  deferred pass and the simulator's frame is 1.69 off the CPU's over the
  material -- identical on the tree before this fix; and
  `serialize_tree` hands one `programs` dict down its node-group
  recursion keyed by `node.name`, which is unique per tree only, so a
  Coded Shader in the material and one inside a group that share a name
  (the default one) leave ONE program, the last compiled (seen at the
  export).
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
  name.* The frame pass emits one colour chain into the diffuse term and
  has no slot for a raw emission lobe (the Emission node's own standing
  refusal), so `gpu/shade._probe_material` refuses the lobe -- it is
  marked `implicit` -- with `IMPLICIT_EMISSION_REFUSAL`: "a colour, value
  or vector output is wired straight into a shader socket: that is an
  Emission of it (Blender's implicit conversion, the picture the CPU
  shades), and the frame pass emits only the diffuse chain -- this
  material shades on the CPU exactly (a Halcyon Shader in between, the
  chain in its Diffuse Color or Self-Illumination, puts it on the
  driver)". It refuses on the lobe's PRESENCE, never its weight: the
  probe reads sixteen fragments, and a Mix Shader's Fac mask can read 0
  on every one of them while the emitter still mixes the link into the
  albedo elsewhere. (3) *This changes pictures.* On the CPU device such a
  material rendered its fallback surface and now adds the wired colour
  as emitted light, without the fallback's highlight (a plain closure
  with no glossy lobe has specular level 0 -- the Emission node's
  picture); on the demo frame every pixel off the material is bitwise
  what it was. On the GPU device it rendered the colour as a lit albedo
  and now renders the CPU's frame, with the reason printed once. A
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
  passes each need the route proven); it stays queued, and until then
  the two devices draw the same frame. And that frame is Halcyon's
  Emission node's, not Cycles': with no DIFFUSE lobe in the closure
  `closure_to_surface` leaves the material's fallback diffuse LIT under
  the glow, where Blender shades black. That rule also reads its batch
  (`np.any(diff_w > 1e-6)`: a pure-emission pixel has diffuse level 0
  when a diffuse pixel shares its batch and level 1 when it is alone --
  seen at the function, not hunted in a frame). Recorded, not changed:
  every Emission-node material's picture hangs on it. Pinned in
  `test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads`
  (53 checks) for a colour, a value and a vector: the CPU frame bitwise
  the Emission node's; the wired colour arriving as exactly itself over
  the fallback (frame(colour) - frame(black) within 1e-6 on every pixel
  of the Ball); the refusal word for word; and the road it names -- a
  Halcyon Shader in between, either socket -- qualifying with the
  simulator frame the CPU frame at the twin bar (measured 4.8e-6; the
  bar is under 6e-3 with no pixel over 1e-2). Then the Mix and Add
  Shader inputs, Fac 0, a Reroute on the way and a Coded Shader's
  output; the unwired socket and a Diffuse BSDF in between still
  twinning; the Emission node keeping its own refusal; the GPU device
  road on the fake driver returning the CPU frame bit for bit; the World
  Output. Ten mutations of the fix each fail it.
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
  plan signature still fingerprints the baked frame on its own. Left as
  they were: the normal, UV, material-index and triangle-index parts of
  the mesh key are still strided sums (a UV edit confined to unsampled
  rows, or one that cancels, does not re-upload the attribute texture).
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
