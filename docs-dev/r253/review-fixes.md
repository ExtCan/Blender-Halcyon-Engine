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
  material rendered its fallback surface and now renders the wired
  colour as emitted light alone, the Emission node's picture (no glossy
  lobe: specular level 0; no diffuse lobe: no diffuse, the next entry);
  on the demo frame every pixel off the material is bitwise what it
  was. On the GPU device it rendered the colour as a lit albedo
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
  the two devices draw the same frame. And that frame was Halcyon's
  Emission node's, not Cycles': with no DIFFUSE lobe in the closure
  `closure_to_surface` left the material's fallback diffuse LIT under
  the glow, where Blender shades black, and that rule read its batch
  (`np.any(diff_w > 1e-6)`: a pure-emission pixel had diffuse level 0
  when a diffuse pixel shared its batch and level 1 when it was alone).
  Recorded here as an open item, seen at the function alone; hunted in
  a frame and closed by the next entry. Pinned in
  `test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads`
  (54 checks) for a colour, a value and a vector: the CPU frame bitwise
  the Emission node's; the wired colour arriving as exactly itself
  (frame(colour) - frame(black) within 1e-6 on every pixel of the Ball,
  and since the next entry frame(black) is black: it sat over the lit
  fallback); the refusal word for word; and the road it names -- a
  Halcyon Shader in between, either socket -- qualifying with the
  simulator frame the CPU frame at the twin bar (measured 4.8e-6; the
  bar is under 6e-3 with no pixel over 1e-2). Then the Mix and Add
  Shader inputs, Fac 0, a Reroute on the way and a Coded Shader's
  output; the unwired socket and a Diffuse BSDF in between still
  twinning; the Emission node keeping its own refusal; the GPU device
  road on the fake driver returning the CPU frame bit for bit; the World
  Output. Ten mutations of the fix each fail it.
- **A closure with no diffuse lobe has no diffuse, at every pixel (the
  rule read its batch: a hard edge through the glow).** The open item of
  the entry above, hunted in a frame and closed. In
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
  the level, the material's ambient level. A pixel with NONE has no
  diffuse response -- Blender's answer for an Emission, a Glossy, a
  Glass or a Transparent alone, and what every such pixel already was
  the moment one diffuse pixel shared its batch: diffuse level 0 and NO
  ambient share, over a WHITE diffuse colour. The ambient share goes
  because the ambient term is the diffuse colour times the ambient level
  and never reads the diffuse level. The colour is white rather than the
  mixed batch's black because nothing at the pixel rate reads it once
  the level and the ambient are 0, and the two roads that read the
  colour alone want the identity: the corner road's `light x albedo`
  (finding d) and a see-through pixel's refraction tint. And "no
  diffuse" cannot hang on the colour: the deferred pass fills its
  diffuse slot from the graph's own colour chain, so only the level and
  the ambient can say it to both devices. The specular level is the
  clipped glossy weight (0 without one, as it always was) and the
  specular colour stays the material's own there; a pixel no
  rough-carrying lobe reaches keeps the material's roughness and
  exponent (it read roughness 0 and the clamped exponent 8192 the moment
  a rough lobe shared its batch). The graph-less material and the EMPTY
  closure (an unwired Surface socket) never reach the branch: the early
  return keeps the fallback surface, bit for bit. (2) *This changes
  pictures: every closure with no diffuse lobe, on the CPU device.*
  Surveyed old against new on 216 frames of the demo scene at 128x96 --
  twenty closures at nine settings (the three rates, ray tracing,
  Transparency None, Screen Door and A-Buffer, a forced Constant model,
  shadows and the environment off) and six see-through closures on the
  layer road at six. An **Emission node** alone, and a colour wired
  straight into Surface: the Ball is exactly the colour at the pixel,
  vertex and face rates; it was the colour over the lit fallback (Ball
  mean (2.04, 1.44, 1.27)), and at the corner rates the fallback colour
  multiplied into both. Shader to RGB of an Emission reads the colour
  itself (through a second Emission the Ball was (3.19, 2.58, 2.44): the
  read lit a default surface under the glow). A **Glossy BSDF**, a
  **Metallic BSDF** and a **Principled BSDF at Metallic 1** (the node
  always hands on a DIFFUSE lobe, whose weight is 0 at Metallic 1 or
  Transmission 1): the highlight alone, Ball mean (1.15, 1.15, 1.17) ->
  (0.009, 0.006, 0.002). A GLOSSY lobe carries no reflection in this
  engine, so a metal is its highlight on black where it was its
  highlight on the viewport colour lit as plastic -- the largest change
  in this entry for an imported scene. A **Glass BSDF**, a **Refraction
  BSDF** and a Principled at Transmission 1: the reflection and the
  highlight alone ((1.21, 1.24, 1.31) -> (0.07, 0.10, 0.14); (0.16,
  0.22, 0.25) ray traced). A **Transparent BSDF** and a **Holdout**
  alone: bitwise unchanged wherever alpha composites (Sorted, A-Buffer,
  ray traced, the corner rates: the layer is alpha 0); under
  Transparency None, where alpha is forced to 1, the opaque surface is
  black where it was the lit fallback, and under Screen Door the dropped
  pixel carries no colour. **Mixes by a mask**, one batch, pixel rate:
  Diffuse with Emission, with Glossy, and with Transparent on the layer
  road are bitwise what they were (their no-diffuse pixels were already
  unlit); Diffuse with Glass differs on the glass side's highlight alone
  (139 pixels, up to 0.87): a GLASS lobe adds a specular weight and no
  roughness, so its exponent was 8192 in a mixed batch and the
  material's 25 alone, and is the material's now. At the corner rates
  the emission, glossy and glass sides appear where they were black.
  Unchanged, bit for bit, at all nine settings: a Diffuse BSDF, a
  Translucent BSDF (a diffuse lobe at half its weight), a Principled
  BSDF at Metallic 0 and 0.5 and at Alpha 0, the unwired Surface socket
  and the graph-less material. (3) *Both roads.* The raw EMISSION lobe
  and the implicit one still refuse the deferred plan by name, so both
  devices draw the CPU frame there. What the plan qualifies it twins by
  construction, with no GLSL changed: the probe bakes through
  `closure_to_surface`, and the pass's ambient term is `s.diffuse *
  (ambient * the baked level)`. A lone Glossy or Metallic (the specular
  slot routing) bakes diffuse level 0, ambient 0 and a white diffuse,
  and the simulator frame is the CPU frame at the pixel, vertex and face
  rates (5.1e-5, 4.8e-7, 1.2e-7). Two disagreements the batch rule held
  go with it. Finding (e): the glossy pixels twin now (the island's 79
  remain, item 4 below). And a Transparent BSDF alone under Transparency
  None or Screen Door, where the plan qualified and the pass lit the
  node's own colour as an albedo while the CPU lit the fallback: 0.48
  off on all 598 pixels of the Ball for a white one and 1.44 for a
  tinted one, 7.2e-7 now (level 0 and ambient 0 leave the colour unread
  on both). (4) *Considered and left.* (i) *The ambient share of a
  PARTIAL diffuse weight.* A diffuse lobe at any weight above zero keeps
  the whole ambient term, so a Fac running up to 1 steps down by the
  lobe's colour times the ambient pool where it arrives: Fac 0.999
  against Fac 1 is (0.021, 0.036, 0.145) on the demo Ball. (Under the
  batch rule the step at exactly 1 was the whole fallback popping IN,
  1.12 the other way.) Scaling the ambient level by the diffuse weight
  would close it and is Blender's reading too, but it changes every
  plain mix with a partial weight on both devices (a Principled at
  Metallic 0.5 would lose half its ambient); left for its own decision.
  (ii) *The Glass and Refraction lobes' Roughness.* The nodes evaluate
  it and `closure_to_surface` never accumulates it, which is why a glass
  pixel has a specular weight and no roughness of its own (the 139
  pixels above). (iii) *The master branch reads its batch for the
  MODEL.* Two Halcyon Shaders of different models mixed by a mask shade
  each batch with its heaviest lobe's model (the mean weight over the
  batch): Phong against Toon on the Ball, three bands at 640x480 against
  the in-process frame, 3,089 pixels up to 1.51. Seen in a frame, not
  changed: the light loop takes one model per call, and a per-pixel
  model is its own piece of work. That branch's other slots were checked
  and are per pixel (a pure-master and a pure-plain pixel are bitwise
  the same alone and in a mixed batch, against an Emission, a Glossy, a
  Diffuse and a Transparent). (iv) *The probe's sampled constancy cannot
  see an island.* The 79 island pixels of finding (e) still draw as the
  glossy surface in the simulator: sixteen fragments missed a lobe
  living on 13% of the material, `PROBE_FRAGMENTS`' standing risk for
  every baked field. (v) *A Mix Shader (Diffuse, Transparent) at a
  constant Fac 0.5 under Transparency None or Screen Door* qualifies and
  draws 0.48 off the CPU on all 598 pixels, before and after this fix
  (it has a diffuse lobe; the pass's colour chain is the mix of both
  nodes' colours, the CPU's diffuse the Diffuse BSDF's alone). Recorded,
  not touched. Pinned in
  `test_a_closure_with_no_diffuse_lobe_has_no_diffuse_at_every_pixel`
  (84 checks). At the function: the report's own two lines; every slot
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
  two EMISSION refusals. Twelve mutations each fail it: the batch rule
  restored for the diffuse, the specular or the roughness slot; the
  fallback kept per pixel; the ambient share kept, taken from every
  pixel, or decided by the batch; a black albedo, or the fallback's; the
  level or the specular level left the material's; the rule reaching the
  empty closure.
  `test_colour_wired_into_a_shader_socket_is_an_emission_on_both_roads`
  was re-run: the wired colour is still bitwise the Emission node's
  frame and frame(colour) - frame(black) still the colour, and it now
  also holds that frame(black) is black (54 checks).
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
- A closure with no diffuse lobe in a live file (proven against
  hand-built graphs and a headless worker pool alone): a masked Mix
  Shader (Diffuse, Emission) at 1280x960 or larger in a real F12, in
  process, across the worker pool and under Ctrl+B, with no edge through
  the glow; a Principled BSDF at Metallic 1 from a real export (the
  weight-0 DIFFUSE lobe the node always hands on) drawing its highlight
  on black beside EEVEE's reflecting metal; a lone Glossy BSDF on a real
  driver matching the CPU device (the simulator twin is 5.1e-5); a
  Transparent BSDF alone under Transparency None on the driver.
