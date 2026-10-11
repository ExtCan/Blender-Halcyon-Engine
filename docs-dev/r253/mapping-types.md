# R253 mapping-types — More mapping types (Tri-Planar, etc)

Branch `r253/mapping-types-v2`. Ships in 1.92.0. The integrator owns
CHANGELOG.md / README.md / the version stamps; the fragments below are
written for them.

## CHANGELOG

- **More mapping types: Tri-Planar, cube atlas, the sphere conventions,
  screen / reflection / normal / tangent texco, object projection, hex
  grid (R253).** What was asked: the era's full menu of texture-coordinate
  generators, not one tri-planar trick — Blender 2.79 Internal's texco
  (`render_texture.c` `texco_mapping`: Object with an object picked, Refl,
  Nor, Window), the 2.79 Mapping panel's Flat / Cube / Tube / Sphere, 3D
  Studio R4's Mapping and Camera Map, Imagine 3.0's Screen brush mapping,
  the spot-lamp slide projector, Debevec's light-probe conventions
  (SIGGRAPH 98), Geiss's tri-planar weights (GPU Gems 3 ch. 1), Red Blob
  Games' hexagonal grids. What was built:
  - **The Projector node grows nine modes**, APPENDED after the four
    shipped ones (saved files keep their index): *Cube Atlas* (six faces
    by the normal's dominant axis and sign into a 3 × 2 atlas, z the face
    0..5), *Equirectangular* (`texture.env_equirect_uv`'s text on the
    axis-reordered triple, no normalize — atan2 is scale-invariant),
    *Mirror Ball* (`env_sphere_uv`, the GL_SPHERE_MAP formula), *Angular
    Map* (Debevec: radius = angle from the axis / π), *Capped Cylinder*
    (the Cylindrical text round the side, a flat lid where |N·axis| >
    0.70710678, z = 1 there), *Screen* (the Tex Coord Window number: the
    pixel's own `(px + 0.5) / w`; zeros on ray hits and secondary passes on
    both devices), *Reflection* (`n_tex_coord`'s Reflection through the
    equirect — BI's Refl texco), *World Normal* (N × 0.5 + 0.5; the
    view-space Nor is the Matcap Coordinates node) and *Tangent Space*
    (the vector in the Normal Map node's tangent / bitangent / normal
    frame, `mathx.orthonormal_basis`). A new **Keep Aspect** bool (Screen
    only, default off) multiplies x by width / height. The four shipped
    modes are byte-for-byte 1.91 — `core/nodeeval.n_v16_projector` and
    `gpu/emit.e_v16_projector` keep their lines and dispatch the new modes
    to `_projector_r253` / `_e_projector_r253` before them; a test pins the
    CPU digests on 4096 lanes × 3 axes and the emitted GLSL text. **The
    shipped Box mode picks its face by the COORDINATE's largest axis, not
    the normal** (`nodeeval.py` BOX branch, as it always did); it is kept
    for picture stability and documented in its tooltip: *Box picks by the
    coordinate, Cube Atlas and the Tri-Planar nodes pick by the normal.*
  - **Tri-Planar Coordinates** (`HALCYON_TriPlanarNode`, Vector family,
    `MATCUBE`): three planar frames (the Image Texture's Box pairs: X →
    (y, z), Y → (x, z), Z → (x, y)) over Scale and the normal's blend
    weights `w_i = max(|n_i| − t, 0) / max((wx + wy) + wz, 1e-9)`, `t =
    (1 − Blend) × 0.5` (0.5 < 1/√3, so a unit normal's dominant axis always
    survives; Blend 0 is the hardest pick — a face fades in only past |n_i|
    = 0.5 — Blend 1 weights by |n|). **Tri-Planar Mix**
    (`HALCYON_TriPlanarMixNode`, Utilities, `SNAP_FACE`): `(cx·wx + cy·wy)
    + cz·wz` on all four channels, Alpha the blended alpha. Both bitwise
    twins (`nodeeval._tri_planar_frames` ↔ `emit._tri_planar_frames_glsl`,
    one shared helper per road so the two tri-planar nodes cannot drift).
  - **Tri-Planar Texture** (`HALCYON_TriPlanarTextureNode`, hand-written,
    Vector family, `IMAGE_PLANE`): owns an Image (`template_ID`),
    Interpolation and Extension, samples it through the three frames and
    blends by the weights — what an artist drops on a rock. The sampler
    contract is `n_tex_image`'s (filter / wrap maps, the renderer's Texture
    Filter overrides the node, no image = opaque black and NO sampler on
    the GPU); the three taps are plain top-level samples (a non-flat
    projection keeps the top level, the R251 rule), ONE `em.samplers`
    entry with `footprint: False`. Bitwise under NEAREST, ≤ 2e-3 under
    Linear (the simulator's bilinear against the CPU's).
  - **Object Projection** (`HALCYON_ObjectMapNode`, hand-written, Vector
    family, `OBJECT_DATA`): an Object picker and a Mode (*Auto* / *Local
    Frame* / *Perspective* / *Parallel*), inputs Vector (unlinked = the
    WORLD position: `_tex_vector` / `tex_vector` gain the `'world'`
    default → `ctx.P` / `hal_P`), Ortho Size, Angle; outputs Vector,
    Inside (the decal mask: in front of the lens and inside the 0..1
    window), Depth (−local.z). `export.object_map_props` bakes rows 0..2 of
    the picked object's inverse `matrix_world` (the `object_matrices()`
    convention, scale included) as twelve floats, resolves Auto from the
    object (spot lamp / perspective camera → PERSPECTIVE with tan(spot_size
    / 2) or sensor / (2 · lens), sensor_fit VERTICAL reading sensor_height;
    sun lamp / orthographic camera → ORTHO with ortho_scale / 2; anything
    else → FRAME), writes `proj_tan` / `proj_half` whenever one number can
    be resolved (the object's lens first, else the unlinked socket) and
    warns *"no object picked; world space"* on an empty picker. The
    matrix product is the explicit `((m0·x + m1·y) + m2·z) + m3` on BOTH
    roads (not einsum, not GLSL `dot()`), the twelve floats travel through
    `em.const` so they lift into material texels under `mark_values`
    (moving the projector re-uploads, never recompiles — the R174
    contract), and every baked scalar is rounded through the same
    8-significant-digit decimal the GLSL literal carries
    (`nodeeval._lit8` / `export._lit8` ↔ `Emitter._m`'s `{:.8g}`) so the
    twin is bitwise on all three outputs in all three modes.
  - **Hex Grid** (`HALCYON_HexGridNode`, Vector family, `MESH_ICOSPHERE`):
    pixel-to-hex by axial coordinates and cube rounding (Red Blob Games),
    Pointy / Flat Top (the x/y swap); outputs the local 0..1 frame (z the
    cell id), Cell ID (the TileRandom integer hash `hash3(rx, rz, 53)` ↔
    `hal_pt_hash3` through `_need_prims`) and Edge (1 at the centre, 0 on
    the edge — the pointy-top hex's inradius distance, clamped at 0).
    Rounds with `floor(x + 0.5)` on both roads with the add in its own
    statement (np.round's half-even would split from the GPU: the
    VectorSnap ROUND seam, not repeated). Bitwise twin.
  - **Fix: the Image Texture's Sphere / Tube / Box projections reach the
    GPU.** `gpu/emit.e_tex_image` sampled every projection FLAT (a silent
    CPU/GPU split, never refused); it now carries the equirect text
    (Sphere), `atan / 2π` + `z × 0.5 + 0.5` (Tube) and the `hal_N`
    dominant-axis pick through `_box_project`'s pairs (Box). CPU untouched
    (bitwise); the FLAT text is byte-identical (pinned); Box bitwise,
    Sphere / Tube the same texel on 4096 lanes. `projection_blend` stays
    unread on both roads, as it always was — the Tri-Planar nodes are the
    blended answer.
  - Stand-ins / notes: a spiral MAPPING is Polar + one Math add (angle + k
    · radius), so no new node; Screen mapping of a PICKED camera is Object
    Projection in Perspective mode; the hex tie-break at an exact cell
    boundary is `floor(x + 0.5)`, not Blender's roundEven (measure-zero).
    **Refuses by name** (CPU-only, printed once by the plan):
    `HALCYON_ProjectorNode (Screen mode with Keep Aspect needs the frame
    resolution)` when an Emitter has no resolution, and
    `HALCYON_ObjectMapNode (a linked Angle needs tan() at shading time; set
    the Angle on the node)`. No new REFUSED entries: every new id is in
    DISPATCH and EMITTERS. **Default neutrality:** every new setting is off
    or absent in an old file (Keep Aspect False, the enum appended), the
    four shipped Projector modes are the 1.91 bytes, and the `textured`,
    `demo` and `mirror` featurematrix pictures render the pre-edit bytes
    (adler32-pinned).
- **Tests.** `tests/test_r253_mapping.py` (340 checks: the shipped
  Projector modes' CPU digests and GLSL texts pinned, the Box-by-vector
  quirk documented; the nine new modes vs `node_twin` on 4096 lanes × 3
  axes — Cube / Screen / Normal bitwise, the rest ≤ 2e-6 (measured ≤
  3.6e-7); Screen is the pixel, zeros on hits and secondary passes, Keep
  Aspect's refusal by name and its bitwise twin; the tri-planar weight
  laws, the one-hot axis case, Blend 1 ∝ |n|, the Box pairs, all six
  outputs bitwise; Tri-Planar Mix bitwise; Tri-Planar Texture = Image
  Texture Box bitwise on axis normals, the exact three-tap mean, no image →
  black and no sampler, one sampler with footprint False, NEAREST bitwise,
  Linear ≤ 2e-3; Object Projection's local box bitwise vs the explicit
  chain, the perspective axis / cone edge / behind / depth laws, Ortho
  Size 2, the export resolution on six fake object kinds + the unpicked
  warning, all three outputs bitwise in all three modes, the linked-Angle
  refusal by name, the `hal_MV` lift; the hex centre / neighbours / edge
  midpoints / vertex / inside laws, Pointy ↔ Flat swap, bitwise twin; the
  Image Texture projections on an index image; the tables census; two
  frame twins on the demo scene at 6.2e-6 and the three pinned pictures),
  registered in `run_all` after the R253 compositing block. The
  constraining tests hold: `test_the_new_halcyon_nodes_travel`,
  `test_every_node_prop_reaches_the_renderer`, `test_tooltips_everywhere`,
  `test_shader_add_menu_census`, `test_icon_census`,
  `test_node_callback_signatures`,
  `test_texture_coordinates_and_two_uv_maps_on_the_gpu`,
  `test_matcap_coordinates`, `test_the_twelve_new_nodes`,
  `test_every_version_stamp_agrees`, `test_gpu_shaders_compile_and_agree`,
  `test_r252_console::test_glsl_texts_define_each_function_once`.

## README

**More mapping types.** The Projector node now carries the whole period
menu: besides Planar, Cylindrical, Spherical and Box it offers a six-face
**Cube Atlas**, the three sphere conventions (**Equirectangular**, the
World's HDRI layout; **Mirror Ball**, the photographed chrome sphere;
**Angular Map**, Debevec's light probe), a **Capped Cylinder**, **Screen**
mapping locked to the view (Imagine's and 3D Studio's, with *Keep Aspect*
for a square image on a wide frame), Blender Internal's **Reflection** and
**World Normal** texco and **Tangent Space**. Box keeps its shipped
behaviour — it picks the face by the coordinate; Cube Atlas and the
tri-planar nodes pick by the normal. **Tri-planar texturing** comes two
ways: *Tri-Planar Texture* owns an image and projects it from X, Y and Z,
cross-faded by the surface normal (rock, bark, terrain — anything without
a UV map), while *Tri-Planar Coordinates* + *Tri-Planar Mix* are the
same thing as a node graph, for three different images or any three
textures. **Object Projection** measures coordinates in another object's
frame: an empty for Blender Internal's Object coordinates with an object
picked, a spot lamp for the slide-projector decal (its cone becomes the
0..1 window, *Inside* the mask, *Depth* the throw), a camera for 3D
Studio's Camera Map, or any object flat along its −Z with *Parallel*.
**Hex Grid** tiles hexagonal cells with a per-cell random, an edge
falloff and a local frame, pointy- or flat-topped. Every mode and node
shades on both devices — bitwise in the simulator where the arithmetic
is plain, within 2e-6 for the trigonometric ones — and the Image
Texture's own Sphere / Tube / Box projections now reach the GPU too. A
spiral mapping is Polar plus one Math node (angle + k × radius).

## Blender-only

- The Image picker (`template_ID`) and the Object picker draw and poll
  in the real node editor; the stub never calls `draw_buttons`.
- Camera sensor fit (AUTO / HORIZONTAL / VERTICAL), pixel aspect and the
  render border on `proj_tan` and Keep Aspect (the exporter reads
  `sensor_fit`, `sensor_width`, `sensor_height`, `lens`, `ortho_scale`,
  `spot_size` behind getattr guards).
- Depsgraph re-export when the picked projector object moves (the
  viewport preview following a moved lamp / camera: the matrix is baked at
  export, lifted into material texels, so it should follow without a
  recompile).
- Real-driver precision of `atan` / `acos` / `sqrt` for the trigonometric
  modes (the simulator twins at ≤ 3.6e-7; the driver's bar is the frame's
  6e-3) and of `texture()` filtering for Tri-Planar Texture under Linear.
- glslang strictness of the new GLSL texts (`tests/glsl_strict.py` needs
  `HALCYON_GLSLANG`); the nested ternaries and `int()` casts follow the
  Region / TileRandom precedents.
- Saved .blend load: a 1.91 Projector node reopened in 1.92 keeps `mode`
  and shows the thirteen-entry menu (enum appended, no migration).
- That Tri-Planar Texture and Object Projection read right to an artist
  under Add ▸ Halcyon ▸ Vector.
