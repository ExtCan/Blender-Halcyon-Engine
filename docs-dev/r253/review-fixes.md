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
  a named IOError and the in-process road. (2) *The GPU marshal's pickup
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
