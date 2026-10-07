# R253 -- halo-node

## CHANGELOG

- **What was asked.** Move the Material tab's halo kit into a proper
  shader node, so the glow's colour, size and ramp can be driven by
  sockets and the kit can keep growing; add a halo shape (falloff)
  menu, blend modes, a depth-sorting choice and per-halo size.
- **What was built.** `nodes/shader_nodes.HALCYON_HaloNode` (Shading
  family, icon PARTICLES): 17 input sockets (Color, Size, Alpha, Add,
  Edge Color, Ramp, Ring/Line/Ray/Bolt Color, Fade Near/Far, Stretch,
  Stretch Angle, Glow Size/Strength/Color) with `HALO_SOCKET_DOCS`
  tooltips applied at creation and at load (`ensure_sockets` also grows
  any socket a saved node lacks), and the ~50 kit properties the panel
  carried -- hardness, seed, rings/lines/star and their counts and
  widths, the 12 silhouettes, the image, noise, bolts, rays, gradient,
  aspect/rotation, the random HSV scatter and the HSV shift, pulse /
  flicker / spin and the per-effect clocks, the four 2.79 mode bits --
  every default equal to the panel's so a migrated material is value
  identical. `draw_buttons` pages them (`ui_page`: Core / Trim / Energy
  / Colour / Motion / Advanced). New with the node: `falloff` (BI
  hardness ladder, Gaussian, Linear, Quadratic, Hard disc, Ring Only
  with `ring_inner`), `blend` (Add Slider, Alpha Over, Additive,
  Screen), `depth_mode` (Z-Buffer, Over Everything), `animate_seed`,
  camera-distance Fade Near/Far, an anamorphic Stretch along a screen
  angle, a second Gaussian Glow layer, and per-halo Size from exactly
  three link sources (Attribute = a POINT float attribute, Color
  Attribute = a POINT layer's red channel, Particle Info > Size); any
  other link on Size, or on a colour/value socket other than an
  RGB/Value node, warns by name and the socket value is used.
  `export.py`: `halo_node_of`, `material_is_halo`, `_halo_socket_value`,
  `_halo_size_source_of_node`, `_halo_sizes_from`, `_halo_spec_from_node`
  (the panel road's exact dict keys and arithmetic plus the new keys)
  and `_halo_spec_from_panel` (the old block, factored unchanged);
  `export_material` reads the node first and the panel only as the
  deprecated fallback; `_collect_halo_points` / `_collect_extra_halos`
  gate on `material_is_halo` and fill the group's `'sizes'` array
  (CORNER-domain attributes refused by name). `core/render._draw_halos`:
  the seed walks with the frame, the camera distance rides each halo
  for the fade, the pixel window and the halo frame became closures
  (`_window` / `_frame`) so Stretch and the Glow can widen the rect,
  `_halo_falloff`, `_halo_stretch` and `_halo_blend` are the new
  module helpers, Over Everything skips the z-test and the old
  near-geometry softening, and the glow composites before the core
  with the same blend and depth rule. Every new key is read with
  `spec.get(key, default)` and gated at its default, so an old spec
  dict takes the pre-R253 path bit for bit. Migration:
  `shader_nodes.migrate_halo_material` runs at load_post (before the
  socket pass) and from `legacy_import` and the new
  `HALCYON_OT_halo_node` operator: it copies `HALO_PANEL_MAP` value for
  value, turns the R200 'unset follows Line Colour' contract into the
  Ray/Bolt Own Colour toggles via `is_property_set`, relinks a
  `__halo_ramp` widget into Ramp, lands the output on an unlinked
  Surface only, and clears `hs.halo` so the node is the one source of
  truth. The Material tab shows the node by name, or the deprecated
  legacy kit with a Convert button, or the Add Halo Node button; the
  `halo` checkbox is no longer drawn (the panel path is
  creation-locked; the 52 `halo_*` properties stay registered as the
  deprecated carrier so old files load).
- **Sources.** Blender 2.79 `convertblender.c make_render_halos`,
  `renderdatabase.c RE_inithalo`, `shadeoutput.c shadeHaloFloat`,
  `rendercore.c addalphaAddfacFloat`; the 2.79 and 2.4x manuals' Halo
  pages; Reeves 1983 (additive Gaussian sprites); Porter & Duff 1984
  and the PDF Reference 1.7 section 11.3.5 (Screen); GL_ARB_point_sprite
  / D3D point sprites (per-point size, the depth-test-off overlay).
- **Stand-ins / refusals by name.** Halos remain a CPU splat over the
  finished frame on BOTH devices (no halo code in `gpu/`): the frame
  leaves the GPU at `'halos'` as before, `gpu/capability.FEATURES`
  gains the non-blocking `halo_splat` (NOT_YET) row that says so,
  `gpu/emit.REFUSED['HALCYON_HaloNode']` refuses the node by name if it
  is ever wired as a surface, and `core/nodeeval.DISPATCH` names it the
  same way (`_n_named`, an empty `Halo` closure). Socket links resolve
  to constants only (RGB / Value nodes) -- a texture chain cannot be
  evaluated per halo at export and says so. The glow layer under a Soft
  halo is z-tested, not soft-faded. Glow Size clamps at 8, Stretch at
  20 (window cost).
- **Tests.** `tests/test_r253_halo_node.py` (8 tests, 128 checks):
  registration / family / icon / tooltips / NODE_PROPS / panel-map
  defaults; the node spec equals the panel spec on all 46 shared keys,
  Own Colour gating, the 32x4 ramp LUT, RGB/Value constants, named
  warnings, `size_source` from Attribute / Particle Info; node wins
  over panel and the panel dict is the pre-R253 dict key for key;
  migration on a fake tree (52 values, toggles, ramp, Surface,
  idempotence, linked Surface kept, muted-node pick); headless splat:
  new keys at defaults `array_equal` with the old road (also around
  live soft/aspect/rotation/trim), every option speaks, Add 0 / Add 1
  bitwise ALPHA / ADDITIVE, OVER draws the buried point, SCREEN never
  passes white where ADDITIVE does, Stretch widens along its angle,
  per-halo sizes, determinism; attribute sizes from a stub mesh (POINT
  float, POINT colour red channel, CORNER refused, missing warned);
  refusal by name on both roads; the Material tab / operator sources
  and the operator on fake materials. `tests/test_render.py`:
  NOT_SHADING gains `ui_page`, the socket-doc table and the
  `ensure_sockets` count gain `HALO_SOCKET_DOCS`, and
  `test_halo_materials`' neutral kit pins the twelve new keys at their
  defaults bitwise on the old road.

## README

**Halo materials are a node now.** Add *Halcyon > Shading > Halo* to a
material and its vertices glow as Blender Internal halos -- faces do not
draw -- with the whole kit on one node: the colour, size, alpha and Add
slider as sockets (an RGB or Value node can drive them), the rings,
lines, star, the twelve silhouettes and the image, noise, bolts and
rays, the gradient and a Color Ramp linked straight into the Ramp
socket, random HSV scatter, the HSV shift, pulse, flicker, spin and the
per-effect clocks, and the four 2.79 mode bits. New with the node:
*Falloff* (the 2.79 hardness ladder, or Gaussian, linear, quadratic, a
hard disc, or a hollow ring), *Blend* (the Add slider, alpha over,
additive, screen), *Depth* (z-buffered, or over everything for the
point-sprite overlay), *Animate Seed*, a camera-distance *Fade
Near/Far*, an anamorphic *Stretch* along any screen angle, an outer
*Glow* layer, and a per-halo *Size*: link an Attribute node naming a
float point attribute, a Color Attribute (its red channel) or Particle
Info > Size and every vertex, point or particle carries its own radius.
A file saved before 1.92 grows the node at load with every value
carried over and the old Material-tab toggle cleared; a material that
still shows the legacy block has a *Convert to Halo Node* button. Halos
splat on the CPU over the finished frame on both devices, as they
always did.

## Blender-only

- The node's draw_buttons paging, the template_ID image picker on a
  node, and the socket tooltips as Blender renders them (fakebpy
  validates arities and icons only).
- `migrate_halo_material` inside a real load_post: the use_nodes flip,
  node creation, `__halo_ramp` relinking, output Surface linking, and
  the file's dirty state afterwards; a PointerProperty image assigned
  on a node at load.
- `HALCYON_OT_halo_node` from the Material tab (operator context, undo)
  and the panel's three states (node / legacy kit / button).
- Live re-render on node edits through the depsgraph (engine.py covers
  ShaderNodeTree ids; an image swap on the node is the untested case).
- Per-halo Size from a real mesh float attribute, a POINT-domain colour
  attribute, and Particle Info > Size on a Blender 5.x particle system
  (the legacy particle API may be absent).
- The exported ramp LUT from a real ColorRamp linked into Ramp
  (`cr.evaluate` on a live node).
- The GPU road in the field: the deferred frame leaving the GPU at
  'halos' and the post warning line once per frame (headless uses the
  fake device).
- Blender's own validation of a Node class with ~52 annotated
  properties and a 12-item EnumProperty at register_class.
