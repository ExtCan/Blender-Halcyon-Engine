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
