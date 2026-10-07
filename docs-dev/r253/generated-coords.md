# R253 generated-coords — Generated coordinates in the object's own box

Branch `r253/generated-coords`. Ships in 1.92.0. The integrator owns
CHANGELOG.md / README.md / the version stamps; the fragments below are
written for them.

## CHANGELOG

- **Generated coordinates follow the object (R253).** What was asked: a
  procedural texture (Marble, Noise, Bozo, any pattern node with an
  unlinked Vector, Texture Coordinate ▸ Generated) slid across a mesh
  whenever the object was moved, turned, scaled or animated. Blender's
  Generated output is the mesh's *texture space* — the object-space
  bounding box (`BKE_mesh_texspace_calc`, auto texspace) or the mesh's
  manual Texture Space (Cycles `mesh_texture_space`: `loc - size ..
  loc + size`; 2.79 Internal's TEXCO_ORCO) — so it never changes with the
  object transform. Halcyon measured it over the WORLD box of the exported
  (already transformed) vertices: a translation kept the box glued, but a
  rotation or a scale reshaped the world AABB, and that was the scroll.
  Now the exporter measures the box on the UNTRANSFORMED vertex array
  (`export._mesh_arrays` → `gen_bounds`, honouring `use_auto_texspace` /
  `texspace_location` / `texspace_size` behind a getattr guard, carried on
  `ObjectInfo.gen_bounds` through `_info_object`'s new optional third
  argument and the mesh cache's `data`), and the renderer takes every
  surface point back into its object's frame through the per-object
  inverse matrix R243 already ships to both devices before normalising:
  `core/mathx.object_space_points` is the ONE chain (the sequential float32
  sum the GLSL simulator's `dot` evaluates — bitwise the simulator and
  bitwise the einsum the Texture Coordinate Object output ran before, 200k
  points, 0 ULP — so the Object output shares it and no picture moves),
  `ShadeJob.object_generated_frame` the per-object (lo, span) table (the
  exported box; at an identity matrix the world rows verbatim; else the
  object's vertices taken back through its inverse and boxed — hand-built
  scenes), `ShadeJob._object_space` the per-object transform without an
  (n,4,4) temporary, and `context()` the switch. On the GPU
  `gpu/material._generated_frame` bakes the object box as `hal_lgen_lo` /
  `hal_lgen_span` (the same `_bounds_fns` if-chain the world table uses)
  and emits `hal_generated = (vec3(dot(hal_obj_r0(td.y), vec4(P, 1.0)),
  …) - hal_lgen_lo(td.y)) / hal_lgen_span(td.y)` in both `assemble_frame`
  and the bump height pre-pass; `_object_frame(need_rows=True)` emits the
  three `hal_obj_rN` rows exactly ONCE per pass whether Generated, the
  Object output or both read them (`gpu/device.duplicate_definitions`
  stays silent). The twin is BITWISE in the simulator at the function level
  (the generated line compiled standalone on the context's own P and
  object ids: 0.0 over every triangle) and 1.8e-5 at the frame level on a
  moved + turned + non-uniformly scaled ball (the R243 bar of 6e-3 stands
  for a driver's FMA fusion of the three dots). The WORLD table
  `hal_gen_lo` / `hal_gen_span` stays for the cartoon centre (`hal_ccen`)
  and the hair-shine azimuth (`hal_hcen`): they measure a direction from
  the world point to the world centre and are right as they are, exactly
  as the CPU keeps `ctx.obj_bounds` world; only the shine's height term
  (Generated.z) follows the object now, on both devices. **Default
  neutrality:** untransformed objects are bitwise the 1.91 numbers (the
  identity inverse returns P bit for bit, the local box IS the world rows)
  — proved on contexts and on whole frames (a Generated marble, the
  textured demo card, a bump pre-pass); transformed objects change BY
  DESIGN. `generated_space` (Render ▸ Textures, "Generated Space") is the
  escape hatch: OBJECT (default, Blender's rule) or WORLD, the pre-1.92
  text verbatim on both devices, in `_plan_sig` because the bake reads it.
  Stand-ins / divergences kept on purpose: Halcyon keeps its `max(hi - lo,
  1e-6)` clamp for a flat axis (Blender's texspace gives such an axis size
  1, so a plane's Generated.z would be 0.5) so a flat ground plane with a
  3-D Noise renders as before; the auto path measures the EVALUATED mesh,
  so an armature or shape-key deformation still breathes the box slightly
  where Blender keeps the undeformed texspace; volumes (`core/volume.py`)
  still measure Generated over the container's world AABB; the Blur node's
  `_shifted_ctx` still adds a world-space delta (CPU-only node). Refuses by
  name: `'generated coordinates need the per-object frame the caller did
  not supply'` (no `obj_gen`), the legacy road's bounds refusal and
  `_object_frame`'s matrices refusal unchanged.
- **Tests.** `tests/test_r253_generated_coords.py` (88 checks: the one
  chain bitwise the simulator dot and the einsum; Generated follows a
  moved + turned + scaled ball within 2.4e-7, the exported box road, a
  pure translation, the legacy formula bitwise and its 0.448 scroll;
  identity contexts and frames bitwise under OBJECT and WORLD; the GPU twin
  on moved objects — the pass, Object + Generated in one pass with the rows
  once, the standalone generated line bitwise, the legacy text verbatim,
  the plan-signature flip; the bump pre-pass twin 4.5e-4; the manual
  texture space 0.25..0.75 and its literals; the cartoon centre bitwise
  unchanged with `hal_ccen` from `hal_gen_lo(`; the export road on
  bpy-free fakes — `_mesh_arrays`, texspace on/off, `_info_object`
  two- and three-argument, `export_scene` of a moved cube; the setting's
  surface, the refusals by name, prewarm and an object-less scene),
  registered in `run_all` after the R252 block; `test_render.py` gains the
  `generated_space` A/B row (a DIFF on a moved marble ball, since identity
  scenes are EQUAL by design) in `test_every_setting_does_what_it_says`.
  The constraining tests hold: `test_generated_coords_are_per_object`,
  `test_period_patterns_shade_on_the_gpu` (the 26 pattern twins),
  `test_cartoon_shader_gpu_parity` / `test_anime_80s_gpu_parity` (the
  pinned names), `test_max_shelf_completion` (the R243 Object-output twin
  on moved objects), `test_bump_glass_layers_shade_on_the_gpu`,
  `test_blender5_socket_names_and_volume_chains`,
  `test_light_edit_fast_paths`, `test_glsl_node_emitters_match_cpu`,
  `test_generated_glsl_is_driver_strict`, `test_no_duplicate_definitions`,
  the two tooltip sweeps, `test_r252_console` (146).

## README

**Generated coordinates are measured in the object's own box.** Every
pattern node with an unlinked Vector, and Texture Coordinate ▸ Generated,
reads the mesh's *texture space* as Blender defines it: the object-space
bounding box, or the mesh's manual Texture Space (Object Data ▸ Texture
Space, with Auto Texture Space off) as `location ± size`. The box is
measured on the untransformed mesh at export and the shading point is
taken back into the object's frame on both devices, so a marble ball keeps
its veins when you move, turn or scale it and an animated object no longer
scrolls through its procedural textures — on the GPU the same three dot
products the Object output uses, bit for bit in the simulator. The
Cartoon Shader's shape smoothing and the Anime Shader's hair-shine
azimuth still measure from the object's world-space centre, as they should;
only the shine's height band rides the object's own Z. Two things stay
Halcyon's: a flat object keeps a thin box on its flat axis (Blender gives
it a size of 1), so a ground plane with a 3-D Noise renders exactly as it
did, and a mesh deformed by an armature or shape keys measures its
evaluated box. A file that relied on the old world-space box can set
**Generated Space** (Render ▸ Textures) to *World box (legacy, pre-1.92)*;
untransformed objects render identically either way.

## Blender-only

- That `Mesh.use_auto_texspace` / `texspace_location` / `texspace_size`
  are readable on the evaluated mesh `compat.evaluated_meshes` hands out in
  Blender 4.x / 5.x (the getattr guard falls back to the vertex box if not).
- That the mesh cache (`export._cache_key`, keyed on the object token and
  the matrix) serves a fresh `gen_bounds` after a manual Texture Space
  edit with no geometry change (if the data token does not bump on that
  edit, fold the texspace triple into the token).
- That a procedural texture stays glued while dragging / animating an
  object in the live viewport preview, and how visible the breathing of
  the evaluated box is under an armature or shape-key deformation.
- The real-driver tolerance of the three `dot()` products (FMA fusion)
  versus the simulator's bitwise chain on NVIDIA / AMD / Intel; the
  integer-hash patterns can amplify a 1-ULP seam at a cell boundary
  (the same exposure R243's `hal_object` already has).
- Loading a pre-1.92 .blend: `generated_space` is absent from the
  PropertyGroup and resolves to the OBJECT default (enum items only added,
  no migration needed).
- The Textures panel row renders and the tooltip reads in the UI.
