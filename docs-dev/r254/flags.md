# R254 -- flags (the per-lamp contribution flags and light linking)

Work package `flags` of the light audit. The integrator owns CHANGELOG.md,
README.md, version.py, `__init__.py`, blender_manifest.toml and
tests/run_all.py; nothing here touches them. What they need from this
package is listed under "For the integrator" at the end.

The user's report: "The lights don't get softer/harder with size. Also, I
think some of the general options just don't work properly." This package
is the second half for the lamp's contribution flags (Ambient Only,
Negative, Only Shadow, Diffuse Only, Specular Only) and Light Linking: the
`hal-flags` audit cluster less its Light Limit, ray-BVH, bake and clock
findings (other packages), the verifier's new finding
flags-volume-beam-01, `gpu-ambient-only-plan-stale` of the viewport-update
cluster (the same defect as ambient-gpu-stale-01), and four defects found
while testing the corners nobody had rendered.

Roads checked for every fix: the CPU road (core/render.py, headless and
real Blender 5.2.2 F12), the GPU road through its simulator twin
(gpu/shade.plan_frame + simulate) and through the fake device
(tests/fakedevice: the whole shade_frame road under the driver's rules;
the machine has no real GPU driver), and the rendered viewport under xvfb
for the two edits a user makes first. "Fork point" below is commit 8de170d
(branch claude-cloud when this package started), a frozen copy of which
every before-number was measured on.

## 1. Ambient Only did not remove the lamp

- **What the user saw.** Ticking Ambient Only on a Sun left the sun: its
  directional light, its highlight and its shadow stayed, with a flat
  ambient added on top. On a Point lamp at a working wattage the frame went
  white. The tooltip promised "flat ambient everywhere instead of
  directional light".
- **What was wrong.** One reader knew the flag: `core/lights.
  ambient_light_split` added colour x Energy to the ambient pool. Every
  other reader of the lamp list tested the lamp's TYPE for `'AMBIENT'`
  only, so the lamp stayed in the light loop (`select_lights`), in the
  volume march (`core/volume.march`), the beam (`core/render.
  _light_volumes`), the shafts and the flare (`shaft_sources`,
  `_flare_sources`), the key lamp of the line (`core/lines.key_light`),
  the BVH decision (`_per_light_ray_shadows`) and under the Light Limit,
  where it took a slot.
- **What it does now.** `LI.is_ambient_lamp(light)` -- the AMBIENT type or
  a lamp with Ambient Only -- is the one predicate, asked by
  `select_lights` (and through `LI.frame_lights` by the CPU loop, the GPU
  plan and its texel repack, the planar shadows, the fog lobe, the ground
  plane and the light meter), by the volume march, the beam, the shafts,
  the flare, the shaded halo, `key_light` and `_per_light_ray_shadows`. An
  ambient lamp is no member of the frame's lamps: no diffuse, no
  highlight, no shadow, no beam, shaft or flare, no slot under the limit,
  no line in the limit's report. What is left is the pool, whose formula
  is the one it was.
- **What Energy means there (the decision).** The pool reads an ambient
  lamp's Energy as a FLAT level, colour x Energy x style gain, 1.0 = white
  ambient -- whatever the lamp's kind, where every other use of a Point
  lamp's Energy divides its watts by 4 pi d^2. That is kept: a file that
  used the option at a level that rendered (a fraction) keeps its ambient
  to the bit, and no distance-free conversion from watts exists that
  would not move those files. So an 800 W Point lamp with the box ticked
  is still 800 times white. It is now SAID: under the checkbox the panel
  draws the lamp's own Energy again as "Ambient Level" with the note "A
  flat level, not watts: 1.0 = white ambient", the row turns red and the
  note becomes "Level 800 whites out the frame: 1.0 = white ambient" above
  2.0 (`LI.AMBIENT_BRIGHT`), the tooltip gives the unit, and an F12 prints
  and reports "Ambient Only: 'Key' at level 800. An ambient lamp's Energy
  is a flat level (1.0 = white ambient), not watts, so the frame whites
  out. Lower it in the Light panel, under Ambient Only"
  (`LI.ambient_report`). If the owner would rather give watts a meaning on
  an ambient lamp, the conversion is one line in `ambient_light_split` and
  needs a migration of stored lamps.
- **Pictures that change.** Only scenes holding a lamp with Ambient Only:
  its direct term, its shadow, its Shadow-pass count, its beam / shaft /
  flare and its scatter in a volume disappear; with such a lamp first in
  the scene the cel line's key lamp is the next lamp; over the Light Limit
  the slot it held goes to a real lamp. The ambient it adds does not move.
- **Proof.** Headless, fill + key at 800 W with the flag (the verifier's
  rig, Shadow Map / Ray Traced / Per Light + Ray Traced alike): Diffuse
  pass on the lit floor +0.3083 -> 0.0000, highlight Spec 0.1419 -> 0.0000,
  Ambient pass 560.0 -> 560.0. An Ambient Only Sun of strength 0.5 over a
  0.7 floor: floor min 0.3500 / max 0.4513 -> 0.3500 / 0.3500, Shadow pass
  min 0.00 -> 1.00. Moving the lamp (max change of any pixel): Point
  0.00223, Sun 0.12632, Spot 0.00205 -> 0 exactly, and in a scene with a
  volume container 0.03862 / 0.06288 / 0.00365 -> 0; on the twin 0. Light
  Limit 2 over [ambient, A, B] kept ['Amb', 'A'] -> ['A', 'B']. Real
  Blender F12: Sun 0.5, lit ground 0.4392 / inside the slab's shadow 0.3490
  -> 0.3490 / 0.3490; Point at 0.25, moving the lamp 0.0039 -> 0.0000;
  Point at 800 W, frame mean 0.9517 before and after (the unit), with the
  report line above. Rendered viewport (xvfb): rotating an Ambient Only
  Sun -- classified 'lights', the lights-only export -- moved the picture
  by 0.1098 -> 0.0000.

## 2. The GPU plan drew a stale ambient

- **What the user saw.** GPU device: editing an Ambient Only lamp's Power
  or colour, or letting its Light Style run, left the ambient where it was
  until something else forced a new plan. A styled ambient lamp never
  flickered.
- **What was wrong.** `gpu/shade.plan_frame` baked the pool into every
  pass as three `vec3` literals (`gpu/material.assemble_frame`) while
  `_plan_sig` held the settings' Global Ambient and the world's ambient
  and nothing of the lamps that feed the pool: their energy and colour
  left the signature in R169, when lamp values became texels.
- **What it does now.** The pool is a VALUE. A scene that carries an
  ambient lamp appends an ambient tail to the `hal_lights` row -- three
  texels after the last light's block: the whole pool, its engine part,
  its world part, the CPU's own float32 triples
  (`pack_light_texels`, `GM.AMBIENT_TAIL_TEXELS`) -- and its passes read
  `hal_ltex(tail)` in place of the literal (both halves for a Blender
  Internal material's flat world ambient). The row is repacked on every
  plan-cache hit already, so a Power, colour or style edit is a hit plus
  one tiny upload. The signature gains one entry, `_ambient_sig`: None
  without an ambient lamp, the flag with one. The radiosity gather still
  bakes the pool (its sky term, in two functions of the pre-pass and the
  material pass); under Radiosity the entry carries the pool's value, so
  an edit there is a new plan and the right picture. A pass with no lamp
  left in its loop (a scene lit by an ambient lamp alone) declares and
  binds the row for the tail.
- **Why a value and not the signature.** The verifier measured the
  signature fix on a pulsing ambient lamp: 20 new plans in 48 frames, each
  a shader compile on a device. As a value the same 48 frames are 1 plan
  and 47 hits.
- **Pictures that change.** GPU road only: frames after an edit of an
  ambient lamp on a warm plan cache (they now equal the CPU frame). A scene
  with an AMBIENT-type light and no edit draws the same bits as before
  through different GLSL (the literal was the float32 the texel is).
- **Proof.** One scene, the cache kept: Energy 0.1 -> 0.4, hit, max
  |CPU - GPU| 2.10e-01 -> 1.19e-06; colour, hit, 2.10e-01 -> 1.19e-06; a
  FAST_STROBE step, hit, 2.10e-01 -> 1.19e-06; Ambient Only off, a stale
  HIT at 7.0e-02 -> a miss at 1.19e-06. The fake device, two frames with a
  Power edit between: 2.10e-01 -> 1.31e-06, plans 1 -> 1. A Blender
  Internal material after a Power edit on the warm cache: 2.40e-01 ->
  1.19e-06. 48 frames of SLOW_STRONG_PULSE: 1 plan / 47 hits with a worst
  error of 0.3325 -> 1 plan / 47 hits at 0.0000. Not run on a real GPU
  (llvmpipe falls back to the CPU here); the fake device applies the
  driver's rule that a declared sampler is bound.

## 3. Light linking reached one instance of three, and no instancer

- **What the user saw.** A lamp linked to a collection excluded only the
  LAST of several instances of an object in it. Linked to the collection
  that holds the instancer empties, it excluded nothing. The hair of a
  linked object stayed lit. And with a library object and a local object
  of one name in the scene, excluding the local one darkened the library
  one instead.
- **What was wrong.** Both export sites (`export.export_scene` and the
  viewport's `export_lights_into`) resolved names through `{o.name: i}`:
  one index per name, the last one. An instancer is no exported object, so
  its name resolved to nothing. The resolve ran before the hair rows were
  appended. Names were bare.
- **What it does now.** `export._resolve_links(lights, objects)` serves
  both sites: every exported row of a name answers, and every row
  INSTANCED BY a named object. `compat.instancer_name(inst)` is the
  outermost object that generated an instance -- the empty of a
  collection instance, the object carrying a geometry-nodes or particle
  instancer, the parent of a vertex instance -- and `ObjectInfo.instancer`
  carries it (hair rows too). `compat.evaluated_meshes` leaves it in
  `compat.LAST_INSTANCER[0]` before each yield and `export_scene` reads it
  while it stands on the item; the generator's four-element yield and its
  loop head in `export_scene`, which the viewport package rewrote while
  this one was written, are untouched (the audit's verifier asked for a
  fifth tuple element: that is the tidier form once both are in, a
  three-line change). Names are library-qualified (`compat.link_name`:
  Blender's `name_full`, `ObjectInfo.link_name`). The resolver decides
  every picked link, runs after the hair rows exist, and a volume
  container follows its object (section 5).
- **On the GPU.** With instances resolved a link can hold hundreds of
  indices, and the plan refused above 64. The ladder is one term per RUN
  of consecutive indices now (`GM.link_ladder`: a lone index is the
  comparison it always was, a run is one range test) and the refusal
  counts runs: 300 instances side by side plan as one term.
- **Pictures that change.** Scenes whose linked collection holds an
  instanced object, an instancer, an emitter with hair, or an object that
  shares its name with a library object. A link to plain, separately named
  objects resolves to the indices it did.
- **Proof (real Blender F12, top faces of three instances, lamp on
  Exclude).** Collection instances, linked to the object's collection:
  [0.2896, 0.3764, 0.0] with indices (3,) -> [0.0, 0.0, 0.0] with
  (1, 2, 3). Linked to the collection of the three empties: all lit, ()
  -> all dark, (1, 2, 3). Geometry-nodes instances, linked to the host:
  () -> (1, 2, 3); to the instanced object: (3,) -> (1, 2, 3). Vertex
  instances by parent and by child, particle instances by emitter and by
  object: the same four numbers each. Hair: indices (1,) -> (1, 2), the
  hair window 0.2066 -> 0.1055 (the ground between strands). Library
  'Rock [lib.blend]' beside local 'Rock', Exclude the local: tops
  library / local [0.0, 0.3117] -> [0.3155, 0.0]. The lights-only export
  over the real depsgraph resolves the same three links: (4,) / () / ()
  -> (2, 3, 4) / (2, 3, 4) / (-1,). Headless: a 300-index run plans and
  matches (it refused by name).

## 4. 'Only' with an empty collection lit everything

- **What the user saw.** Linking 'Only' with a collection that holds no
  rendered object lit the whole scene; the item reads "Nothing but the
  collection is lit by this lamp".
- **What was wrong.** The resolver skipped an empty name set and left
  `exclude_objects = ()`, which the CPU loop and the GLSL ladder both read
  as no link.
- **What it does now.** A picked collection that resolves to nothing
  writes `(-1,)` under 'Only': no object carries that index, so the lamp
  lights nothing on either road, in the viewport too. Under 'Exclude' an
  unresolved link leaves the lamp unlinked, as before. No renderer change.
- **Proof.** Real Blender F12: frame mean 0.2799 -> 0.0000. Headless:
  'Only' (-1,) is 0 on the CPU and the twin; 'Exclude' (-1,) is bitwise
  the unlinked frame. Rendered viewport: the step did not change the
  picture -> the lamp's light leaves (frame mean 0.104 -> 0.059).

## 5. Volumes, beams, shafts, flares and halos ignored the flags

- **What the user saw.** In fog (a volume container) a Negative lamp
  brightened the fog it darkened on the floor, an Only Shadow lamp lit
  it, and a lamp the Light Limit had dropped still lit it. The lamp's own
  visible beam did the same. A shaded halo took light from Only Shadow
  and Specular Only lamps and from lamps the limit had dropped, and a
  Negative lamp brightened it.
- **What was wrong.** `core/volume.march` built its lamp list from
  `scene.lights` less the AMBIENT type and added `rad * vis * phase` with
  no sign; `core/render._light_volumes` took every volumetric lamp of a
  marching kind; `shaft_sources` and `_flare_sources` walked
  `scene.lights`; the shaded halo (`_draw_halos`) walked `scene.lights`
  reading Ambient Only alone. None asked the Light Limit.
- **What it does now.** `LI.lights_volume(light)` is the sign a lamp's
  scatter takes in a medium: +1, -1 for Negative, 0 for an ambient or an
  Only Shadow lamp. `VOL.scatter_lights(scene, st)` -- the frame's lamps
  (`LI.frame_lights`) with a sign, less the lamps with no diffuse lobe
  (Specular Only, Diffuse off) -- is the march's list, and the march
  applies the lamp's light link to the container's own object. The beam,
  the shafts and the flare read `_medium_lamps` (the frame's lamps less
  sign 0); a Negative beam subtracts. The shaded halo is a diffuse
  receiver of the frame's lamps with their sign. Diffuse Only and
  Specular Only leave the lamp's own beam alone: it is neither lobe (the
  Specular Only tooltip says so). All of these are CPU passes shared by
  both devices.
- **In `_light_volumes` (shared with the beam's unit and cone law).** Two
  edits only: the list (`lights = _beam_lamps(scene, st)`, in place of
  the comprehension over `scene.lights`) and, between the marcher call
  and `if scatter.any():`, three lines that negate `scatter` for a
  Negative lamp. The composite line is untouched.
- **Pictures that change.** Volumes, beams, shafts, flares and shaded
  halos lit by a Negative, Only Shadow, Specular Only or ambient lamp, or
  by a lamp over the Light Limit; a volume whose container is in a lamp's
  link. A scene with none of those marches the same lamps in the same
  order with the same signs.
- **Proof.** Headless, what the lamp adds to the fog (the plain lamp
  +0.4562): Negative +0.4562 -> -0.4562; Only Shadow, Specular Only, both
  Only boxes, dropped by Light Limit 1: +0.4562 -> 0.0000 each; a dropped
  Point with a map shadow changed a limit-1 frame by up to 1.2694 and its
  map was read (0.1727) -> 0 and 0. The beam (plain +0.5393): Negative
  +0.5393 -> -0.5393; Only Shadow, Ambient Only, dropped: +0.5393 -> 0;
  Specular Only +0.5393 (kept). The shaded halo (plain +0.4426): Only
  Shadow, Specular Only, dropped +0.4481 -> 0; a plain lamp and its
  Negative twin +0.8962 -> 0. Shaft and flare sources of an Only Shadow or
  dropped lamp: (1, 1) -> (0, 0). Real Blender F12, a fog box with a fill
  lamp: a Negative 350 W lamp adds +0.1867 -> -0.0971 to the fog; an Only
  Shadow lamp +0.2486 -> 0.000 (its shadow still falls on the floor);
  Light Limit 1, the dropped lamp +0.2486 -> 0.000. Device road, a
  Negative lamp in fog: 7.3e-07 from the CPU frame.

## 6. The Light panel

Three notes, no pixel changed.

- **Light Style on a still (style-still-ui-01).** At frame 1 six of the
  eleven styles render like Normal and three switch the lamp off. Under
  Style Rate a line now says what the style does on the frame the editor
  stands on: "Animated: Power x2.08 on frame 24 (step 11 of 16)", or
  "Animated: the lamp is DARK on frame 1 (step 1 of 16)" with the error
  icon (`ui.light_style_status`). The gain is `LI.style_gain` at
  `ui.scene_clock`, which reads the exporter's clock -- `export.scene_time`
  where the exporter carries it (the frame-rate package's helper), else
  the expression `export_scene` evaluates -- and a test compares the two
  through a real export at five frame / rate / base / subframe settings,
  so the line cannot disagree with the render.
- **Diffuse Only + Specular Only (both-only-ui-01).** With both ticked a
  row says "Diffuse Only + Specular Only: this lamp lights no surface";
  the tooltips say the same. Stored values are untouched.
- **Ambient Only.** Section 1: the Ambient Level row, its unit, the red
  warning, "No direct light, shadow, beam or flare".

Tooltips rewritten: Diffuse Only, Specular Only, Ambient Only, Linking
(what an empty 'Only' does), Light Linking (instances, instancers, hair,
fog), Only Shadow (no fog, no beam; what Negative does to it).

## 7. Found on the way

- **A screen spot outside the frame's lamps broke the GPU fog pass.** With
  Spotlight Fog on, a Model 3 screen spot the Light Limit dropped left the
  plan reading `hal_spotfog`, an accumulator no pass declared: "unknown
  identifier 'hal_spotfog'", on the fork point (with this package an
  Ambient Only screen spot would have hit it too). `gpu/shade.
  _fog_structure` gated the lobe on any screen spot of `scene.lights`; it
  reads the frame's lamps now, as the pass's own declaration and the CPU's
  `spot_fog_of` do. Both cases plan and match at 2.3e-06 / 1.5e-05.
- **The Light Meter node** read the raw Power of every lamp: a Negative
  lamp read as light, an Only Shadow or Specular Only lamp too, and a
  flickering lamp did not flicker. It reads what the lamp gives a diffuse
  surface now (sign, lobe, style gain). CPU only on both devices (the
  plan refuses the node by name).
- **A lamp nothing shades no longer asks for a BVH**
  (`_per_light_ray_shadows` reads the frame's lamps: an ambient lamp or
  one over the limit set to Ray Traced under Per Light built one for
  nobody).

## The corners nobody had rendered

Each on both roads; "twin" is max |CPU - GPU| over covered pixels.

- **Shading rate VERTEX and FACE** (a 30 x 30 floor so corners fall in
  the shadow): Negative is minus the plain lamp (1.0e-07 / 4.5e-08), both
  Only boxes are the frame without the lamp (bitwise), Only Shadow takes
  -0.1852 / -0.1809 in the shadow core where the lamp gives +0.1858 /
  +0.1809 and 0 on the lit floor, Ambient Only is flat 0.1750, an excluded
  floor is unlit; twin 1.2e-07 / 6.0e-08 at worst for Negative, Only
  Shadow, Ambient Only and a link. One thing that is NOT a flag defect:
  under VERTEX a mesh whose triangles are flagged flat while its vertices
  carry smooth normals (which a hand-built test mesh has and an export
  does not write) shades 0.13 apart on the two roads, with or without any
  flag; with the smooth flag set the twin is at 1e-07. Reported, not
  chased.
- **Console models** GX_LIGHT, DS_FIXED, PS1_MODULATE, N64_COMBINE,
  SEGA_MODEL2, SATURN_ADD, each lit by a Negative, an Ambient Only, an Only
  Shadow and a linked lamp together: twin 0.0 on all six.
- **A material's light group under the Light Limit.** Lamps [A, B, C],
  limit 2 (First keeps A, B), a floor whose group is [B, C]: the floor is
  lit by B alone -- its group's lamps that survive the limit -- bitwise the
  B-only frame; twin 3.7e-06.
- **A console material's own light limit under the scene's.** System 22
  (one lamp) under Light Limit 2 with Brightest keeping B, C of [Weak, B,
  C]: the machine's lamp is B, the first SURVIVOR in scene order, bitwise
  the B-only frame; twin 6.0e-08.
- **Only Shadow with Negative.** The shadow core goes from -0.1892 (Only
  Shadow) to +0.1892: the lamp gives its light back only where its shadow
  falls. This is Blender Internal's arithmetic -- the lamp's energy is
  negated at conversion and LA_ONLYSHADOW subtracts energy x (1 - shadow)
  -- and the two roads agree at 1.3e-06. Kept, and now in the Only Shadow
  tooltip.
- **Only Shadow on a Model 3 screen spot** is ignored on both roads (the
  lamp keeps its lobe; a screen spot casts no shadow to give). The screen
  spot belongs to the spot package; reported.
- **The viewport's lights-only export**: the link resolver (section 3,
  one function for both sites, proven over the real depsgraph) and the
  rotation of an ambient Sun (section 1, classified 'lights'). The flags
  themselves ride `export_light`, the same function on both paths.
- **Hair and particles**: strands are surface rows of the light loop (the
  flags act as on any surface; their link is section 3); particle halos
  are the shaded halo of section 5; instanced particles are instances.
- **Not run: a real GPU device.** Every GPU claim here is the simulator
  twin or the fake device.

## Default neutrality

- The feature matrix (tests/featurematrix.py), all 313 rows at 96 x 72,
  fork point against this tree: the CPU frame, the planned GLSL (every
  pass source, or the refusal text), the twin's frame and the uploaded
  `hal_lights` bytes are identical on 313 of 313 rows (291 planned, 22
  refused by name on both). Four of those rows use a flag, a link or a
  beam; none carries an ambient lamp.
- A scene with an AMBIENT-type light (no flag): CPU frame and twin frame
  bitwise the fork point's on two scenes; its GLSL reads the tail.
- Without an ambient lamp the light row has no tail (n x 16 texels, the
  bytes it had), `_ambient_sig` is None and the passes bake the ambient
  literal they did; a link to scattered objects emits the comparisons it
  did; a scene with no flagged lamp and none over the limit hands the
  volume march, the beam, the shafts and the flare the lists they read.
- No property was added or removed; no enum moved. `ObjectInfo` gained two
  fields with defaults (`instancer`, `link_name`).

## Tests

tests/test_r254_flags.py, 137 checks, `python3 -m
halcyon.tests.test_r254_flags` (about 55 s). On the fork point 126 of them
run and 68 fail, at least one per fix. With HALCYON_GLSLANG set its last
test adds one: 18 planned passes of the ambient tail and the link ladder
compile under strict GLSL 4.30. The strict gate (tests/glsl_strict.py) is green on
this tree: 174 + 396 + 4 + 38 + 3 + 8 + 8 shaders, 0 bad.

Existing tests run green on this tree: test_r254_falloff (152),
test_r253_light_options (195), test_r254_world_light (188),
test_r253_halo_node (128), test_r253_console_presets (113),
test_r252_console (154), test_r253_region (130), test_legacy (345), and 36
functions of tests/test_render.py that touch the changed functions (1096
checks). test_r251_lighting_lamps, _lighting_models, _lighting_fog,
_material and _shadow fail only on the names of the baseline list (the
1.89.0 zip pins, the sky ulp pins, the Super FX device pin).

One pin changed:

- tests/test_render.py `test_the_shading_routes_fall`: 'a 70-object
  linking list refuses by name' with `exclude_objects = tuple(range(70))`
  -> 'a linking list of 70 scattered objects refuses by name' with
  `tuple(range(0, 140, 2))` (70 objects in a row are one range test and
  plan; the refusal counts runs, its text says "64 separate runs").

## Not done here, on purpose

- **The shadow bake still bakes a map for an ambient lamp.**
  `core/lights._build_shadow_maps` is the bake package's: its skip
  (`light.type == 'AMBIENT'`) should become `is_ambient_lamp(light)`.
  Until then such a lamp costs a bake nobody reads; no picture depends on
  it.
- **An instancer that sits INSIDE an instanced collection.** Blender's
  depsgraph names the outermost generator of an instance only. A nested
  instance (OuterEmpty -> a collection holding InnerEmpty -> the object)
  exports with instancer 'OuterEmpty': a lamp linked to the outer empty or
  to the object reaches it ((1,), top face 0.0000 in real Blender), a lamp
  linked to the inner empty alone resolves nothing ((), top 0.3423).
  Reaching it through the inner instancer's own collection would also
  catch that collection's other instances; left as it is, and said here.
- Light Limit ranking, the ray BVH, the scene clock (frame-rate base and
  subframe), the Shadows rows and the falloff rows of the panel: other
  packages.

## For the integrator

- Merges into claude-cloud at 3e30799 (the ray-road, spot-beam, map-soft
  and viewport packages in) without a conflict, checked with
  `git merge-tree`; on that merged tree tests/test_r254_flags.py and the
  viewport, spot-beam, falloff, world-light and light-options modules
  pass (ray-road and map-soft passed on the merge with 4e655f6).
- `export.export_lights_into` (the viewport package's): this package only
  ADDS a line, `_resolve_links(lights, parked.objects or [])`, after the
  single-index statements that function had. They are redundant now (the
  resolver decides every picked link) and can be deleted: seven lines
  from `name_to_index = {o.name: i` to the end of its loop.
- `compat.evaluated_meshes` keeps its four-element yield; the instancer
  rides `compat.LAST_INSTANCER` (section 3 says why, and what the tidier
  form is).
- `core/render.shaft_sources`: the filter sits at the `out.append` (a
  lamp must be one of `_medium_lamps`) so it merges under the spot
  package's rewrite of the loop head; it may move to the loop head later.
- `ui.scene_clock` looks for `export.scene_time` (the name the frame-rate
  package's helper carries in its worktree). If that helper lands under
  another name, `test_f_the_panel_says_what_the_flags_do` fails on the
  frame-rate-base rows and names the two clocks: point `scene_clock` at
  the helper.
- tests/run_all.py: add `test_r254_flags`.
- A scene with an ambient lamp uploads a `hal_lights` row three texels
  longer; texels 12-15 of each light's block are still reserved zeros.

### CHANGELOG fragment

    - Ambient Only is ambient only. A lamp with the box ticked kept its
      directional light, its highlight and its shadow, with the ambient
      added on top; it took a slot under the Light Limit and still lit
      fog. It is no lamp of the frame now: flat ambient and nothing else.
      Its Energy is a flat level there (1.0 = white ambient), which the
      panel now draws as Ambient Level, in red when it whites the frame
      out, and an F12 reports.
    - GPU: an edit of an ambient lamp (Power, colour, a light style) no
      longer leaves the old ambient on a cached plan. The pool rides the
      light row as values: no new plan, no recompile per flicker step.
    - Light Linking reaches every instance of a linked object, everything
      instanced by a linked object (collection instances, geometry nodes,
      particles, vertex instances), the hair of a linked object and the
      fog of a linked container; a library object no longer answers for
      a local one of the same name. 'Only' with an empty collection
      lights nothing. On the GPU a run of linked objects is one test (an
      instanced forest no longer falls back to the CPU).
    - Fog, the visible beam, light shafts, lens flares and shaded halos
      honour Negative, Only Shadow, Specular Only and the Light Limit: a
      Negative lamp darkens the fog, an Only Shadow lamp does not light
      it, a lamp over the limit shows nowhere.
    - Light panel: what a Light Style does on the current frame, a warning
      when Diffuse Only and Specular Only are both ticked.
    - Fixed: a Model 3 screen spot dropped by the Light Limit with
      Spotlight Fog on failed to compile on the GPU; the Light Meter node
      read Negative, Only Shadow and Specular Only lamps as light.

## Probes

Under the session scratch, `wt/flags/probes/` (each takes `VC_PKG=base`
for the fork point or `VC_PKG=wt` for this tree): `vc.py` (the verifier's
rig), `v01_flags.py`, `v03_volume.py`, `v04_link_ambient.py`,
`v06_other_receivers.py`, `v08_ambient_plan_cost.py` (the verifier's
probes), `q01_screenspot.py`, `q03_vertex.py`, `q04_ambient_type.py`,
`n01_neutral.py` + `n01_compare.py` (the feature-matrix sweep),
`bl_flags_ba.py` (real Blender F12, 27 cases) + `make_sheet.py` (the
sheet `wt/flags/out/flags_before_after.png`), `bl_viewport_flags.py` (the
rendered viewport under xvfb).
