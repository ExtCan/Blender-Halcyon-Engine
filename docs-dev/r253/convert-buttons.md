# R253 -- convert-buttons (Convert to Anime / Cartoon / Game)

Branch `r253/convert-buttons`, written against the `claude-cloud` tip
(1.91.0 + the field fix). The version stays 1.91.0 in the worktree; the
integrator bumps it once for the round.

## CHANGELOG

### Convert to Anime / Cartoon / Game (R253)

- **What was asked.** "Add Convert to Anime, Convert to Cartoon and
  Convert to Game buttons next to the Halcyon Shader and Blender Internal
  ones, with the same relink-not-reset behaviour, and a way to choose the
  anime style, the cartoon era and the console; converting an existing
  Halcyon Shader material should work too." 'Game' is the Console
  Emulation Shader, the only game-hardware node; the menu says "Game
  (Console Emulation)" and the enum id is `CONSOLE`.
- **Nine buttons, one operator.** The Material panel's converter box
  (`ui.HALCYON_PT_material.draw`) grows three blocks after the BI one,
  drawn by `ui._draw_convert_target`: *To Anime Shader* (with the scene's
  Style and Compatibility menus), *To Cartoon Shader* (its Era) and *To
  Game (Console)* (its machine), each with This Material / Selected /
  Scene on `halcyon.convert_to_node` (`convert.HALCYON_OT_convert_to_node`,
  a `target` enum and the three scopes; the scene choices ride on its
  string properties the way `model` rides on the master conversion, so
  no enum is duplicated). This Material reconverts (force) when the tree
  already holds the target node, as the master's own button does. The two
  existing operators are untouched; the three master buttons and the
  three BI buttons keep their pinned counts.
- **The scene's choices** are four new `HalcyonSettings` properties
  (`convert_anime_style`, `convert_anime_compat`, `convert_cartoon_era`,
  `convert_console`) -- UI state beside `convert_detection`, never a
  render setting (`to_settings` iterates `RenderSettings`' fields, so a
  saved scene renders byte-identically; `test_old_scenes_render_identically`
  proves it through the engine stub). Each is a static tuple with a
  default, reading the node's own menu: `ANIME_STYLE_ITEMS`,
  `CARTOON_ERA_ITEMS`, `core/console.CONSOLE_ITEMS` and the Anime
  Shader's compat menu, which is **hoisted** to
  `core/shading.ANIME_COMPAT_ITEMS` (seven entries, same identifiers,
  same order, texts verbatim) so the node and the setting read one tuple
  and a saved file's compat value keeps its meaning.
- **The decision is bpy-free** (`core/convert.plan_for`), in the master
  shader's socket vocabulary: `plan()` turns any source into master-named
  pairs and extras, `TARGET_MAP` turns those into the target's sockets
  (the Toon BSDF's Size / Smooth become the first shadow band's threshold
  / softness, the master's Rim Light colour the Rim Color, the console the
  identity over the 21 names it shares), and what a target lacks is
  dropped and the first drops reported by name ("IOR has no anime
  equivalent"). **The emission is the one place the targets differ**: the
  cel nodes own an Emission Strength socket that multiplies
  Self-Illumination in the evaluator (`core/nodeeval.py`), so colour AND
  strength travel straight -- Blender 4's white emission at strength 0
  lands as colour x 0 = black, no multiply node, no burn; the console has
  no strength socket, so the master's fold and its multiply-node markers
  apply (`convert._carry` is the master's pairs / extras / scale_links
  surgery parameterised on the node). Per target: **Anime** -- Style and
  Compatibility set (compat first, then style, whose preset is applied
  explicitly), a Principled roughness under *Custom* becomes a Specular
  Size through `anime_specular_size` (a STAND-IN curve, 0.9 / sqrt(gloss)
  clamped 0.02..0.5: the cel highlight is a painted shape, not a lobe),
  metal lifts the Specular Level and says "feed a metal matcap for the
  HoYo look"; **Cartoon** -- the Era set, an Emission shader becomes flat
  paint (Shadow Amount 0, Lamp Influence 0); **Game** -- the machine set
  (its type and options at the machine's defaults, editable on the node),
  Glossiness clamped to `CONSOLE_GLOSS_MAX` 128 (the DS shininess table
  and the GX / Model 3 exponents top out there -- GBATEK, the GX API
  reference), the `bi_plan` metal move (highlight tinted with the base
  colour, level >= 0.8), and a Vertex Color / Attribute node feeding Base
  Color becomes the machine's OWN material source instead of a texture
  chain: `Vertex Color` linked at Mix 1.0 with GX_SRC_VTX on the GameCube
  (`gc_material_src 'VTX'`, Nintendo GX API reference GX_SetChanCtrl),
  D3DRS_COLORVERTEX / GL_COLOR_MATERIAL on PC fixed-function
  (`pc_color_vertex`, Direct3D 7-9 docs), G_LIGHTING off on the N64
  (`n64_type 'VERTEX'`, gbi.h). **A chosen Style / Era owns its sockets**
  (`STYLE_OWNED` / `ERA_OWNED`, derived at import from the presets): a
  CONSTANT the conversion would write there yields to the preset (noted
  by name), a LINKED chain always carries -- it is the artist's network.
- **The engine's own nodes are sources** (`core/convert.HALCYON_SOURCES`:
  the master, the Console, Anime, Cartoon and BI nodes in the master
  vocabulary, modelled on `migrate_master_node`'s travel by socket name;
  BI's Specular Hardness 1..511 is a Glossiness, per the 2.79 manual).
  `plan()` falls back to that table, so *Convert This Material* on an
  Anime / Cartoon / BI / Console material -- to any target, the master
  included -- now carries by name where it used to carry only the generic
  colour-and-normal pair; a cel node's Emission Strength folds on the way
  back to the master (the fold now runs for every source in
  `_EMIT_COLOR`, which gained the two cel nodes). A behaviour change for
  that one button, called out here; nothing pinned it. A source naming
  its own Glossiness no longer has its Roughness socket re-derive one.
- **The slot list and the header** count a converted cel material: the
  UIList used to label an Anime / Cartoon material "not converted"
  (`ui.material_state` now returns "Anime <style>" / "Cartoon <era>"; the
  panel header's `has_master` tuple names both nodes).
- **The alpha predicate sees the cel nodes.** `export._alpha_reason`
  reads Opacity on the Anime, Cartoon and Console nodes (and the
  console's Edge Opacity) exactly as it reads the master's -- linked, or
  below 1.0, is see-through by name. Neutral at the nodes' default 1.0
  unlinked; a converted semi-transparent Principled is the first thing
  that reaches it. A user file that already set a cel Opacity below 1 was
  classified opaque before and takes the see-through layer plan now
  (correctly), on both devices; a per-pixel alpha there may now be named
  by the GPU layer refusals as a master material's would.
- **GPU story.** Nothing in `gpu/` changes and no renderer code is added:
  a conversion edits a node tree, and the converted material renders
  through the target node's existing two-road implementation (the Anime
  twin at < 6e-3, the Cartoon twin with its refusals by name, the Console
  twin per machine / type) -- `test_anime_shader_gpu_parity`,
  `test_cartoon_shader_gpu_parity` and `test_r252_console.test_gpu_parity`
  re-run unchanged.
- **Tests.** `tests/test_r253_convert_buttons.py` (123 checks): the
  tables against the real SOCKETS tuples, the three plans, the engine
  nodes as sources, the surgery and the operator on a fake tree built
  from the real socket tables with the real `apply_style` / `apply_era`
  bound in, the scene props and the panel (its drawing recorded on a fake
  layout), the alpha predicate for the three nodes, and default
  neutrality (a byte-identical frame before and after, the default
  `RenderSettings` unchanged, the master converter's pinned fold cases
  through the new lookup). Constraining tests re-run green:
  `test_material_conversion_plan`,
  `test_the_converter_stops_burning_materials_white`,
  `test_convert_to_blender_internal`,
  `test_convert_shader_detection_option`,
  `test_enum_callback_default_rule`,
  `test_every_setting_has_a_detailed_tooltip`, `test_tooltips_everywhere`,
  `test_material_panel_visibility`,
  `test_a_blend_mode_alone_does_not_make_a_material_see_through`,
  `test_every_version_stamp_agrees`, `test_max_shelf_completion`, the two
  cel parity tests and `test_r252_console.test_node_and_migration` /
  `test_gpu_parity`.

## README

**Convert to Anime, Cartoon or Game (1.92).** The Material panel's
converter has three more one-click targets beside the Halcyon Shader
and Blender Internal ones: **To Anime Shader**, **To Cartoon Shader**
and **To Game (Console)**, each for this material, the selected objects
or the whole scene. The same promise holds -- textures are relinked, not
discarded: whatever fed Base Color arrives in Diffuse Color or Paint
Color, the normal chain in Normal, the alpha in Opacity, the emission in
Self-Illumination (with its strength on the cel nodes' own Emission
Strength socket, so a Blender 4 material's white-at-zero emission stays
dark). Above each block the scene chooses what the new node starts as: an
anime **Style** (the decades) and **Compatibility** (Generic Cel or a game
pipeline's texture decode), a cartoon **Era**, or the **Console** the
Game conversion builds -- the machine's shader type and options stay
editable on the node. A chosen style or era writes its sockets and wins
over the constants a conversion would have put there; a linked chain
always carries. The source may be any Blender shader or one of Halcyon's
own nodes -- a Halcyon Shader material converts to Anime by socket name,
an Anime material to Cartoon, a Blender Internal one to a console (its
Hardness becoming the Glossiness). On the Game side a Vertex Color node
feeding Base Color becomes the machine's own vertex-colour material
source (GX_SRC_VTX, D3DRS_COLORVERTEX, the N64's lighting-off shade)
rather than a texture chain, and glossiness is clamped to the period
light units' 128. The slot list now labels converted Anime / Cartoon
materials by their style / era, and a cel or console material with
Opacity below 1 takes the see-through pass like a Halcyon Shader one.

## Blender-only

- EnumProperty update callbacks firing on setattr inside the operator
  (`style` -> `apply_style`, `era` -> `apply_era`, `console` ->
  `refresh_sockets`) and the socket hide flags they set; headless, the
  explicit `apply_*` calls and a stubbed `refresh_sockets` stand in.
- `tree.links.new` between real socket types (an Image Texture RGBA into
  NodeSocketColor, a Normal Map VECTOR into NodeSocketVector, a Vertex
  Color RGBA into 'Vertex Color') and Blender's refusal of invalid pairs;
  the real `ShaderNodeMix` multiply node's socket identifiers.
- The panel layout: three aligned buttons per row at the default
  properties-editor width (labels kept to This Material / Selected /
  Scene), the choice rows' dropdown widths, the UIList's new
  "Anime <style>" / "Cartoon <era>" labels and the header switching to
  "Halcyon Shader" with the CHECKMARK icon for a cel material.
- Undo of a conversion (REGISTER / UNDO) restoring the muted source and
  the removed links.
- The `depsgraph_update_post` pristine-material watcher
  (`convert._halcyon_default_material`) not re-firing on a freshly
  converted cel material.
- A converted semi-transparent Anime / Cartoon / Console material taking
  the see-through layer plan on a real render, CPU and GPU, including
  the GPU layer refusal text when its alpha is per-pixel.
- Viewport preview redraw after the conversion.

## Deviations from the design

- Test module is `tests/test_r253_convert_buttons.py` (the protocol's
  `test_r253_<feature-key>.py` with the hyphen made importable), not the
  design's `test_r253_convert.py`.
- `HALCYON_SOURCES` / `TARGET_MAP` carry `Rim Amount` (Anime, Cartoon)
  and `Matcap Blend` (Anime) in addition to the design's lists; the BI
  source table also carries `Bump Strength` / `Bump Height`. All exist on
  the nodes (pinned by `test_target_tables`).
- Style / era ownership applies to carried CONSTANT pairs as well as to
  the derived extras (a linked pair always carries); the design's text
  covered the derived extras only. This is what makes "Toon Size 0.7
  under UPA_50S is absent" true.
- `plan()`'s emission fold runs for every source in `_EMIT_COLOR`
  (`if idname in _EMIT_COLOR`) rather than the literal two-name tuple, so
  a cel node converted back to the master folds its Emission Strength;
  identical for Principled and the Emission shader.
- `plan()` no longer derives Glossiness from a Roughness constant when
  the source already carries a Glossiness pair (only the engine's own
  nodes do), so a master source's explicit exponent is not replaced and
  then clamped.
- `_carry`'s extras loop copies colours element-wise (a 3-tuple onto an
  RGBA socket) instead of assigning the tuple, so the metal
  `Specular Color` extra lands in Blender; `convert_material`'s own loop
  is untouched.
- Dropped-socket notes skip `Metalness` / `Roughness` / `Glossiness`
  (consumed by the highlight derivations) so every Principled conversion
  does not report them.

## Limitations

- `anime_specular_size` is a stand-in curve (documented as such).
- The vertex-colour detection reads the Base Color link's `from_node`
  only; a vertex colour mixed through a MixRGB lands in Diffuse Color as
  a texture chain (correct, just not the machine's own switch).
- A carried constant on a console socket the machine hides (e.g. a
  Glossiness on a type that pins it) is simply unused; the node's
  `refresh_sockets` keeps linked sockets visible.
- `test_material_conversion_plan` and `test_max_shelf_completion` in
  `tests/test_render.py` import `nodes/shader_nodes` without installing
  `fakebpy` themselves and raise `ModuleNotFoundError: bpy` when run
  FIRST in a fresh process (pre-existing suite-order dependency; green
  after any fakebpy-installing test, as in `run_all`).
