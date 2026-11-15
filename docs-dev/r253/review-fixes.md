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
  source the whole suite hands the compiler was recorded (1690 distinct
  texts: the engine's passes through the simulators and the pre-flight,
  the twins, the tests' own) and compiled by the front-end before and
  after: 1628 compile in both with the same outputs, uniforms and inputs
  and no warning, 62 are refused in both with the same words. The
  planner's pass texts for all 312 feature-matrix rows and for 11760
  Console Emulation Shader configurations (every machine by its panel's
  options by the node's rate, at three scene rates, with the light limit,
  a painted mesh and the luma / toon dials) came to 405 more distinct
  texts: all compile in both, same declarations. And 12000 seeded random
  shaders (operators, calls, constructors, swizzles, ternaries over every
  type; half with wrong counts mixed in) were compiled and run by both:
  of the 5431 that ran before, 5274 run to the same values, 104 are
  refused for the wrong count of a function or a struct constructor, none
  is refused for its widths, and 53 raise where they "ran" -- each uses a
  struct constructor's value as a number, which ran only because the
  constructor answered its first argument; of the 625 that raised `cannot
  combine` when run, 624 are compile errors (the last combines a struct,
  below). What IS newly refused is, by intent: a wrong count wherever it
  is written (it ran, on other arguments than the ones written), and two
  widths in code no lane reached -- a branch never taken, a function
  never called -- as the name check refuses an undeclared name there.
  *Seen and left* (each reproduced; none is one of the six): of two
  functions with one name and argument count the first is called whatever
  the argument types (`float f(float)` then `vec3 f(vec3)`: `f(v)` runs
  the float one); a matrix times a scalar raises in `einsum`; a struct
  used as a number, or compared with another, raises when run; a struct
  field declared as an array is stored and typed as one element
  (`l.dir[1] = v` reads back wrong, `l.dir[1].xy` is refused as a swizzle
  of a float); a `vec3` initialised from a `vec2` keeps two components,
  and `vec3(a, b)` / `vec2(a, b, c)` pad and cut in silence; a source
  with no function at all compiles and raises `no main() in shader` when
  run; `in vec3 vBitangent;` reads zero unwarned (the table binds it, the
  renderer supplies nothing); and `in vec3 position;` -- a renderer key
  used as a name -- reads the position on the CPU and zero in the
  deferred pass (`gpu/emit` binds table names alone). Pinned in
  `tests/test_shaders.py` (six tests, 67 checks: each reported shader,
  its neighbours, per-lane runs, every operator the parser knows, every
  builtin at each count of its range and one either side, 121 type pairs
  by 7 operators against `promote` itself, the entry bindings, the
  warning list) and in `test_code_node_compiles_what_it_runs` in
  `tests/test_r253_review.py` (18 checks: the real node class through
  Compile, the warning and error boxes, the export, the evaluator at
  shading time, the GPU port and `device.preflight`). 33 mutations of the
  fix each fail them.
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
