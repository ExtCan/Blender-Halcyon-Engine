# Halcyon

> **This Addon is and always will be free. If you paid for this, you were
> scammed. Please demand your money back and report the seller.**

A from-scratch render engine for Blender, built for the looks Cycles and
EEVEE cannot give you: the 3D of every era before physically-based
rendering, and 2D — hand-drawn, painted, filmed — down to the wire.

Not a filter over a modern render. Halcyon is its own scanline z-buffer
rasteriser with optional ray tracing, the reflectance models the old
packages actually shipped, real framebuffer quantisation, a genuine
GLSL/HLSL compiler for coded-shader nodes, and a complete GPU port of all
three stages that is held to the CPU picture pixel for pixel — same frame,
same numbers, on either device, in any band, on any worker. Everything is
Python and NumPy, nothing compiled, and the whole engine runs headless
without Blender for its 9503-check test suite.

![one toy, sixteen looks](docs/halcyon_contact_sheet.png)

**Three roads, one engine.**

- **The eras.** 112 render presets, each what its target actually did —
  Infini-D, Ray Dream, 3D Studio R4 and MAX, trueSpace, LightWave, Imagine,
  POV-Ray, Bryce, Softimage, Alias, Maya 4, RenderMan; VGA, EGA, CGA, the
  Mac 1-bit and 8-bit palettes, Amiga OCS/EHB/AGA/HAM, Atari ST, MSX,
  PC-98, X68000; PlayStation, Saturn, N64, Voodoo, Dreamcast, Nintendo DS,
  Mega Drive; Sega's Model boards, Namco System 21 and the Atari vector
  monitors; NTSC, VHS, Video CD, the early web.
  Shading rates (Gouraud and Flat as *rates*, not models), affine texture
  warp, vertex snapping, three-point filtering, error-diffused and ordered
  dither, real palettes, HAM fringing, interlace, JPEG's DCT — the
  mechanism, never a blur.
- **The libraries.** Your **Blender 2.79 files open with their Blender
  Internal materials intact** — a plain File ▸ Append brings a scene saved
  twenty years ago in looking as it did: shader pairs, ramps, all eighteen
  texture slots on a ported BI texture engine, lamps at their true 2.79
  energies, the old world as sky and fog. The **Bryce 1995 material
  library**, decoded from the original `.mat` files byte by byte, sits on
  the Pre-Made shelf with the numbers Bryce saved in December 1995. And
  **3ds Max's material and map library** — the Standard material's eight
  shaders, the Raytrace material, Blend / Double Sided / Top-Bottom /
  Shellac / Composite, the maps, utilities and coordinate rollouts, a
  shelf of thirty-five nodes — runs on Max's own algorithms, reimplemented
  on Halcyon's own tables (no Autodesk code or data ships), with a 3ds Max
  2012 render preset.
- **The 2D road.** A **Cartoon Shader** (paint, not light: the Golden Age,
  UPA, Xerox-line, Saturday-morning, 90s feature, wartime noir and modern
  flat eras) and an **Anime Shader** (kage tones, the hair shine, the
  airbrush, a six-decade Style menu, and compatibility modes that decode
  real game texture conventions — ArcSys, Genshin, ZZZ); an **ink pass**
  drawn from the shared G-buffer — clean, brush and pencil lines that
  taper, roughen, drift, skip and boil, the inker's weight by the light,
  Arc System Works' vertex-colour line control; **2D media** converters
  that turn the lit cel into hatching, pencil scribble, stipple, charcoal,
  ink wash, paint strokes and paper; **painted backdrops** and painted
  grounds laid in brush strokes; and **the film the cel went through** —
  the three-strip camera's records and filters, the timed print curve,
  impure dyes, dye-transfer registration, the silver key, then grain by
  size and clump, dust on the cel, the negative and the print, hairs in
  the gate, scratches, cue marks, gate weave, flicker, frames held on
  twos and threes.

![the eras](docs/halcyon_eras.png)

![the 2D road](docs/halcyon_2d.png)

![the libraries](docs/halcyon_libraries.png)

**At a glance:** 49 shading models · 249 node types evaluated (every shader
node Blender 5.x offers, plus Halcyon's own 143) · 112 render presets in
eight categories · 143 resolution presets · 114 Pre-Made material templates
in fourteen families · eleven sky modes, 303 skies, 48 waters, ten infinite
grounds · real volumes (eight scattering models, mesh containers, voxels) ·
396 render settings, every one proven to change what it claims · a
9503-check suite that runs without Blender · about 128,000 lines
of Python and NumPy (201,000 with the tests), no compiled dependencies.

---

## Install

Download the latest `halcyon-*.zip` from
[Releases](../../releases) — take the zip, not the source archive.

**Blender 5.1+ (extension):** Edit ▸ Preferences ▸ Get Extensions ▸ ▾ ▸
*Install from Disk…* and pick the zip.

**Any version (legacy add-on):** Edit ▸ Preferences ▸ Add-ons ▸ *Install…*,
pick the zip, then tick **Halcyon Render Engine**.

Then set **Render Properties ▸ Render Engine ▸ Halcyon**. If Blender's view
transform is not **Standard**, the Display panel will say so and offer a button
to fix it — Halcyon outputs display-referred pixels, so AgX or Filmic on top
double-transforms them. Open the *Halcyon Presets* panel and load one —
`VGA Mode 13h` or `PlayStation` show the character of the engine fastest.

Requires Blender 5.1 or newer, and only NumPy, which Blender already ships. No
compiled dependencies, nothing to build.

### Running the tests

The renderer and the shader compiler import nothing from `bpy`, so the whole
suite runs on a plain Python with NumPy — no Blender, no display:

```
git clone <this repo> halcyon
python -m halcyon.tests.run_all
```

That is what CI runs, on Linux, Windows and macOS.

### Reporting a problem

Turn on **Developer Options** in Preferences ▸ Add-ons ▸ Halcyon, then use
**Run Self Test** in the Debug panel. It copies a report to your clipboard
covering your GPU, the shaders compiled on your own driver, a 299-row
feature-by-feature comparison of the GPU picture against the CPU's, per-stage
frame timings and thread scaling. Nearly every bug in this engine's history was
diagnosed from that output; almost none from a description alone.

---

## Blender 2.79 appending — the headline feature

Blender removed the Internal engine in 2.80, and every .blend saved before
then carries materials modern Blender silently drops. Halcyon brings them
back, two ways:

(A second importer, **File ▸ Import ▸ Sparking! ZERO Material (FModel
.json)**, is described with the Anime Shader below: it builds a
character's materials from the game's own export.)

**File ▸ Import ▸ Legacy Scene (.blend)** appends a 2.79-or-earlier file
through Blender's own loader — constraints, custom normals, modifiers, vertex
groups, parenting, animation and every object type arrive exactly as
File ▸ Append would bring them — while Halcyon reads the same file directly
for everything that loader drops:

- **Blender Internal materials, rebuilt one to one.** Diffuse and specular
  shader pairs (all the Internal menu's models), hardness, ramps with their
  inputs, blends and factors, mirror with Fresnel, transparency in Z and
  raytrace modes, emit, ambient, translucency, Shadeless, Shadows Only,
  ray-bias and shadow-bias terminator fixes, object colour, light groups —
  every panel field accounted for, transcribed against the 2.79 source code
  rather than approximated from memory.
- **All eighteen texture slots per material**, with their mappings,
  projections, influences and blend modes — and the procedural textures
  arrive on the **BI Texture node**, a port of the original texture engine:
  Clouds, Wood, Marble, Magic, Blend, Stucci, Noise, Musgrave, Voronoi and
  Distorted Noise over the original noise bases and lookup tables,
  colorbands included, verified CPU-against-GPU in the suite. Packed images
  unpack and travel; image sequences keep their settings.
- **Lamps with their real energies.** 2.79's lamp math — distance falloffs,
  sphere clipping, spot blends in cosine space, the quadratic sliders —
  is applied from the file's own DNA, so a scene lit for Internal is not a
  washed-out or pitch-black surprise. Ray and buffer shadows both convert,
  bias and softness included; pre-2.70 spot sizes convert their degrees.
- **The world comes along**: horizon and zenith as a Halcyon gradient sky,
  ambient into the lighting, mist as fog with its falloff, exposure and
  range, and the file's own colour-management settings mapped onto the
  engine's pipeline (a 2.4x-era file renders bytes-in bytes-out, exactly
  as it did).
- **The file's own framing**: render size, percentage, transparent film,
  and the saved "Selected Objects Only" state are honoured; hidden objects
  can come along or stay home, by checkbox.

**Plain File ▸ Append works too.** A watcher recognises a pre-2.80 file the
moment Blender finishes appending from it and fixes the lamps automatically —
no special import path required, and a **Fix Appended Lamps** button in the
Lighting panel covers anything appended before Halcyon was enabled. Every
import writes a full log to a text datablock, so what was converted — and
anything that could not be — is named, not guessed at.

---

## What it does

**49 shading models**, each implemented from its published formulation rather
than approximated:

Lambert · Gouraud · Flat · Phong · Blinn-Phong · Blinn · Cook-Torrance ·
Oren-Nayar · Minnaert · Ward · Anisotropic · Metal · Strauss · Multi-Layer ·
Toon · Translucent · Constant · Wireframe · Anime/Cel · Cartoon ·
Oren-Nayar-Blinn · the Blender Internal matrix (every 2.79 diffuse/specular
pair) · and 3ds Max's own eight — Phong, Blinn, Metal, Anisotropic,
Multi-Layer, Oren-Nayar-Blinn, Strauss, Translucent — on Max's light loops ·
and (1.90) seventeen period light units and combiners, described with the
period features below: GX (GameCube), Sega Model 2 and Model 3, the DS light
unit, the two provoking-vertex flats (OpenGL last, Direct3D first), the
PlayStation, Saturn, N64 and System 22 modulates, the PS2's HIGHLIGHT,
Direct3D's separate specular, the PowerVR PCX intensity, the DS toon and
highlight tables, the Mega Drive's shadow / highlight ramps and the Super FX
plot

Gouraud and Flat are treated as **shading rates**, not reflectance models,
because that is what they are. Selecting Gouraud on a material evaluates its
lighting once per vertex and interpolates the colour across the triangle;
selecting Flat evaluates once per face. That is where the banding and the
faceting genuinely come from, and it is why they look right instead of merely
blurry.

**249 node types** are evaluated — 102 of Blender's own, audited against its
full surface-node registry, and the audit came back clean: every shader node Blender 5.x offers
has an evaluator except Freestyle's stroke UV, which has no meaning outside
Freestyle. That includes the full Principled BSDF, node groups (recursively),
muted nodes, reroutes, all the texture and colour nodes, and every Math and
Vector Math operation. Nodes the engine doesn't know pass their first matching
input through and are reported as a warning rather than failing the render.
A second audit stands behind the first: every property a Halcyon node
declares must reach the renderer — a node whose dial changes nothing fails
the build with the dial named.

**The master shader** carries the era's whole bag of tricks on one node —
Fresnel, rim light, sheen, matcap, reflection tint, edge opacity, backface
override, vertex colour mix — all applied outside the reflectance model so they
behave the same on every one. **Bump Height** takes any greyscale texture and
bumps it straight into the shading normal: behind the scenes it becomes a real
Bump node between the texture and the Normal chain, so it renders identically
on both devices by construction. The Material Properties tab drives the same
material with plain sliders when **Override** is on — override withholds the
node tree at export, so what the panel shows is exactly what renders.

New materials created while Halcyon is the active engine are **born as master
shader materials** — the panel's New button builds one directly, and a
strictly-guarded watcher converts factory-fresh default materials (exactly the
two untouched default nodes; anything a person has edited is never touched).

**The Anime Shader (1.62)** is a second master node, built for the
2000s-2020s 3D-anime look rather than the 1990s one: N·L wrapped and cut
into two or three tone bands whose shadow COLOURS multiply the base — a
shadow is a colour, never a darkness — with a stepped cel highlight, rim,
matcap, line art, per-material Light Response, and an emission road. Its
**Compatibility mode decodes real game texture conventions**: the ArcSys
ILM/SSS maps of the Guilty Gear Xrd lineage and Dragon Ball FighterZ
(channel semantics from the published shader recreations of the GDC 2015
pipeline), the HoYo character lightmaps of Genshin Impact (decode taken
from the PrimoToon shader source) and Zenless Zone Zero (per its modding
guides), Kakarot on the ArcSys-lineage decode its look descends from
(its tooltip says exactly that rather than inventing channel dumps nobody
has published), and **Sparking! Zero decoded from the game's own material
export** (1.87), read against the field's reconstruction of its master
material: the flat `Color1` as exported (Unreal's linear colour — the
swatch the artist chose), the `Mask1` line-art sheet as a linear
multiply, the 16×256 tone strip read down its height with white at the
top by the half-Lambert cosine, the tone lifted from `GradientAdjust1`'s
floor to white by the strip, the game's outline shell made invisible
(Halcyon inks its own) — with **File ▸ Import ▸ Sparking! ZERO Material**
building every part from its FModel .json, textures found beside it, the
shared outline/mouth/sweat instances found under the export's Common
folder for the slots the pick did not cover. Plug a game's own ILM, lightmap or tone strip into the node and
the channels mean what they meant at home; every mode shades identically
on CPU and GPU, per-pixel texture decode included. Vertex-colour AO
(the Xrd painters' convention) and alpha-is-emission (the HoYo one) ride
checkboxes. **The ramp road (1.65)**: link a texture — the game's own
ramp, a ColorRamp, any chain — into **Shadow Ramp** and the tone bands
come from the ramp itself, baked once into a LUT the lamp loop samples
by the light term; shadow side, transition and lit side are all painted,
exactly the games' convention. **Ramp Row** picks the row of a multi-row
ramp, and under GENSHIN the Game Texture's alpha — the material id,
exactly where the game keeps it — picks the row automatically: the
per-row character ramps travel at last. Both devices sample the same
baked texels.

**The period features (1.90)** — the field's ask: *"Add some more 3D
software/console period features, as many as possible that works with
both GPU and CPU modes"*. 109 mechanisms in nine packs, each taken from a
named machine or package and a named source, each bitwise-neutral at its
default (the 1.89.0 engine is rendered from its own zip and compared),
and each either drawn on both devices or routed to the CPU **by name**.
What "both devices" means this round, said once: every twin is held to
the CPU in the GLSL simulator and through the fake device, headless. Both
waves then ran on the RTX 5060 Ti, on the final tree: 206 variants at
1280×720, CPU device against GPU device, post chain included, the GPU
rasteriser engaged. What you will see: with an error-diffusion dither the
two devices draw different dither patterns (67,372 to 354,017 of 921,600
pixels in the Atari ST, Amiga OCS and Studio R4 rows); three signal rows
— RF, MPEG-1, Video CD — differ by more than 1e-3 at 23,542, 4,774 and
9,266 pixels; every other row is bitwise identical or differs at 1 to 42
pixels (446 and 440 in two rows under an HDRI world, 128 under the VHS
preset), among them a few pixels that differ by large amounts, read as
rasteriser edges (*Known limitations*). Of the second wave's 88 rows, 16
are identical, 66 differ at 1 to 42 pixels, one at 128, one at 440 and
four are among the diverged.
The classes, the stages and the refusals are under *GPU support*; the
per-row table is `docs-dev/r251_field4_verdicts.txt`. No 1280×720 row
covers the six new nodes, the PS2's two-pass alpha, Blender 2.4x's Z
Offset, Invert Z and Env, Thin Wall Refraction or Imagine's fog object: on
the driver those ran only in the 96×72 self-test matrix, where their rows
matched. Where a source did not publish a number, the stand-in is
Halcyon's and says so; the CHANGELOG carries the homework line by line.

*Raster.* **Integer pixel centres** (Direct3D 3–9; the XBOX preset): the
raster's sample grid and the wire lines move half a pixel — the sky,
halos, flares, shafts and the cel field stay on the +0.5 grid, as a D3D9
title's screen-space effects sat half a pixel off its geometry.
**Near-plane rejection** (the PS1's GTE, the PS2's VU1): a polygon with
a vertex past the near plane or the fixed 6.4-half-screen guard band is
dropped whole, so the PlayStation presets pop geometry at the near wall
instead of cutting it. **The ordering table** (PlayStation): whole
polygons bucketed on their average depth, far buckets first, first-added
on top, anything past the table or past 1023×511 pixels not drawn — and
because that tie rule is order-free, **Painter's algorithm now runs on
the GPU rasteriser** (a banded frame and the overdraw census still
rasterise on the CPU, now printed by name). **Depth encodings**: the N64
RDP's 18-bit floating z, the GameCube's compressed 14e2 / 13e3 / 12e4
(the three layouts are reconstructed from the manual's stated
resolutions, not transcribed), the Xbox's fixed-point W-buffer and the
Voodoo's 16-bit floating W, so z-fighting lands where each machine put
it. On a driver the winner at a marked pixel is the CPU's own replay and
a lone boundary pixel may store a code one step off; a frame that marks
more than a tenth of its pixels rasterises on the CPU whole, with the
count printed; the W encodings refuse an orthographic camera by name,
and every encoding falls back to linear under Painter's by name.
**Vertex format quantisation** (PS1 16-bit positions and integer texel
coordinates, N64 8-bit normals): the mesh is rounded once before
anything reads it — on a WORLD grid, where the consoles quantised in
model space, so an animated object crunches against a fixed lattice.
**Jittered sample positions** (REYES / PRMan PixelSamples). **The
rear-plane depth bitmap** (Nintendo DS CLEAR_DEPTH): a float Z pass as
the buffer both rasterisers start from. **N64 coverage anti-aliasing and
the VI filter**: the RDP's 3-bit coverage blended at scan-out toward
the fully covered neighbours, then the divot median (transcribed from
angrylion's source, including the correction that the blend never
passes its neighbours). It reads the frame as five bits per channel
whatever Colour Depth says, so the whole frame bands like a 15-bit
buffer, not only its edges; it needs one sample per pixel; the RDP's
CLAMP coverage accumulate and the VI's 2× bilinear scale are not
modelled, so interior polygon seams soften too and half of the "N64
blur" stays open. **Limit Dynamic Range** (LightWave 5.6–7.5: samples
clipped at 1.0 before the anti-aliasing filter) and the **gamma-2 OSA
blend** (Blender 2.2x–2.41). **The Elite wireframe rule** (BBC Micro: an
edge draws when either of its faces looks at the viewer, no depth test)
and **the vector monitor beam** (Atari DVG / AVG / Star Wars: a Gaussian
phosphor spot at quantised intensity with dwell dots) are drawn on the
CPU over the wire road's readback on both devices, by name; the dwell
constants are chosen, not measured, and the DVG's length dimming is not
modelled.

*Textures.* **Texel formats** (Glide, Direct3D, PowerVR PCX, Sega Model
2's 4-bit luma, 3dfx NCC, the DS's A3I5 / A5I3 — sixteen items; the NCC
fit is Halcyon's own, TexUS's was never published), **the N64's TMEM
budget** (4 KB, nine formats; the CI palettes are Halcyon's median cut)
and **block compression** (DXT1, the NV2A's 16-bit decode — its rounding
is a reading, not a document —, the GameCube's CMPR, the Dreamcast's VQ
on Halcyon's own seeded codebook, whose fit is not pinned across NumPy
versions) convert once at upload, so both devices sample one prepared
array. **Coarse bilinear weights** (Voodoo Graphics' 4-bit and Voodoo2's
8-bit texel fraction) and **POV-Ray's normalised-distance filter**
(interpolate 4). **GL_CLAMP border seams** (OpenGL 1.0/1.1's dark
half-texel seam on conformant drivers). **Chroma key after the
filter** (Voodoo / Direct3D: the key tested on the filtered result, so
a dark fringe survives) — Halcyon blends linear texels where the card
blended bytes, so the fringe is slightly wider unless colour management
is off. Under the VOODOO and VOODOO2 presets (5:6:5 texels, no alpha) a
texture's cut-outs are stored as the key colour and cut after the
filter: wire the image's Alpha to see through them, or the holes show
black. **Mip level select**
(the N64's blend of 3-point levels, OpenGL's nearest level, the Voodoo's
dithered pick — the matrix is applied vertically flipped relative to
the card), **per-polygon mip level** (Riva 128 / Verite: one level per
triangle, so a floor's two triangles show the seam), **LOD from Q**
(the PS2 GS's TEX1 formula; K = −2 in the PS2 preset is Halcyon's
tuning for scene-unit depths), **Sharpen** (N64 G_TD_SHARPEN; Halcyon
lerps float texels where the RDP lerped 8-bit ones) and the
**summed-area filter** (3D Studio R2–R4 and MAX; footprints capped at
255 texels, the box clamps at the texture edge, and at 1:1 one pixel of
a 4,263-pixel test floor differs by a whole texel between the roads,
headless).

*Lighting and fog.* **Fog runs inside the material pass on the GPU** —
see GPU support below. The curves the machines had: **the GTE depth
cue** (PlayStation: affine in 1/z, floored to 12 bits), **the Voodoo's
64-entry w-table** with the Voodoo2's fog dither, **the PowerVR2's
128-entry table** (Dreamcast), **the DS's 32-entry density table** (on
eye depth; the DS indexed its own z- or w-buffer) and **Direct3D's
z-fog**. One caveat for all the tables: the VOODOO, VOODOO2 and
DREAMCAST presets shade at the vertex rate, so their table is read at
the lit corners, where there is no pixel — the hardware fogged per
pixel, and the Voodoo2's dither adds nothing there. **GameCube fog
range adjust** (no preset: at the GAMECUBE preset's vertex rate it
would be inert), **per-polygon fog** (Namco System 21: one bank per
polygon — per triangle here, so a split quad can pop as two halves),
**material fog control** (Sega Model 3's burn-through, Namco System
22's bias and bank; two banks where System 22 had four), **backdrop
fog** (LightWave's Use Backdrop Color; the CPU's backdrop is uploaded
once per plan, 14.7 MB at 720p), **ground fog** (POV-Ray fog_type 2,
the atan integral; a level or falling sky ray is the fog colour
exactly, so a camera that looks down sees a uniformly fogged sky),
**fog turbulence** (POV-Ray; Halcyon's integer-hash turbulence, not
POV's Perlin vector) and **the Model 3 spotlight's fog lobe**. The
lamps: a **camera-axis viewer** for highlights (OpenGL 1.1's default,
the Sega boards, the DS), **Only Shadow lamps** (Blender Internal — and
old files that used them now import them), **cone laws** (OpenGL's
exponent, POV-Ray's hotspot and falloff, the GameCube's angular
functions), **decay laws** (OpenGL's three terms, POV-Ray's fade, GX's
distance tables — the POV-Ray presets' lamps no longer fall off, as in
POV) and **the screen-space spotlight** (Sega Model 3, after
Supermodel's reconstruction; the lobe lives on the screen, so ray hits,
transparent layers and vertex corners get nothing). The light units:
**GX** (GameCube: Lambert plus the rational quadratic "shininess",
8-bit vertex output), **Sega Model 2 and Model 3** (computed on the
CPU's corner road on both devices — the boards lit per polygon and per
vertex, the models are their rates, and a pixel-rate request refuses by
name) and **the DS light unit** (the 128-entry shininess table, 5-bit
vertex colours; the table's shape is Halcyon's, games authored their
own). And POV-Ray's finish on the master shader: **Brilliance**,
**Crand** (grain in the direct diffuse, from Halcyon's hash rather than
POV's sequential generator, so a frame renders the same bits twice) and
**Metallic (POV)**.

*Shadows.* **The midpoint shadow map** (Woo 1992; Maya's Use Mid Dist;
Blender's Classic-Halfway — two caster rasters per map, no bias),
**planar projected shadows** (Blinn 1988; the Sega Model 1 attribution
is the catalogue's, not a primary source), **Dreamcast modifier
volumes** (the PowerVR2's stencil parity and its shadow scale, applied
to the shaded linear colour rather than the vertex colours before
texturing) and **DS shadow polygons** (POLYGON_ATTR mode 3, the
polygon-ID self-exclusion, the 6-bit blend). The planar and volume
shadows are one mask laid over the finished opaque frame on both
devices — after the fog, where the machines shadowed before it — and
the frame leaves the GPU for it by name.

*Materials.* The period combiners are shading *rates* plus one
machine's integer arithmetic: the corners are lit and quantised to the
machine's depth, then interpolated in float32 (not the machine's fixed
point). **Provoking-vertex flat** (OpenGL copies the last corner,
Direct3D the first), the **PlayStation** (texel × vertex / 128),
**Saturn** (the VDP1's signed 5-bit add), **N64** (the combiner's
multiply) and **System 22** modulates (the last inferred from MAME, not
a Namco document), **the PS2's HIGHLIGHT**, **separate specular**
(Direct3D 5–7, the Dreamcast's offset colour), **PowerVR PCX** (one
colour per triangle — the mean of its corners, Halcyon's choice — times
an interpolated intensity), **the 64-step luma ramp** of the Sega
boards (a linear stand-in for ramps the games wrote), **the Mega Drive's
shadow / highlight** DAC levels (the lamp-to-operator mapping is
Halcyon's; the VDP had no lighting), and **Super FX plot** (Star Fox:
one pair of palette entries per face as a checkerboard; it needs a fixed
palette of at most sixteen entries and says so, shading flat,
otherwise). **The DS toon and highlight tables**: the lit red indexes
a 32-entry table built from Toon Size and Toon Steps (the table's
content was game data; this is a reconstruction). **Open: DS Highlight
whitens every lit surface at the default table** — 61% of the test
frame pure white; a narrower band or more steps do not help, because
the generated table always tops out at full and the mode adds the entry
after modulating by it. The hardware rule itself is undecided: GBATEK
gives the form as built (modulate by the table entry, add the entry),
melonDS modulates by the vertex red as grey and adds the entry. Six
new nodes: **Combiner Stage** (one GameCube
TEV or Xbox NV2A register-combiner stage in fixed point; the NV2A's mux,
dot-product and final combiner and TEV's konst selection are not built),
**SR Bump** (the PowerVR2's two-angle bump texels; the azimuth origin is
Halcyon's and has not been checked against KallistiOS), **Emboss Shift
and Emboss Bump** (DirectX 6's two-stage embossing), **Roughness
(Imagine)** (a per-pixel random turn of the normal; the 0–255
calibration is Halcyon's) and **Env Chrome** (Alias / Maya's showroom
reflection: Maya's parameters at Maya's defaults, Halcyon's own blend
curves — a showroom of the same description, not a pixel match). And
**the REYES shading rate** (RenderMan's ShadingRate with constant
interpolation, diced per triangle rather than per primitive; refuses
the GPU by name under affine texturing and for a Bump-node material).

*Transparency.* **Fixed-function blend equations**: the PlayStation's,
the Saturn's VDP1 and the 3DO's pixel processor in 5-bit integers, **SNES and GBA colour math**, **the Mega Drive's column
mesh**, **DS translucency** ((A+1)/32 on 6-bit channels, one blend per
polygon ID, the DS's own order), **framebuffer formats at the blend**
(the PS2's 16-bit buffer with its DIMX dither, the GameCube's RGBA6,
the Voodoo's 565 with dither subtraction — under supersampling the
resolve averages the dither away) and **Doom's fuzz** (the Spectre; a
per-column hash replaces the global counter). All are in Halcyon's own
depth order, not the machines' submission order. Every equation that
reads the frame beneath refuses the GPU's *layer* passes by name — a
5-bit lattice would turn a one-ulp device difference into a whole level
— while the opaque frame under it stays on the GPU. **The N64's random
alpha compare** (Screen Door, re-rolled per frame), **the PS2's
two-pass alpha** (the solid half in the z-buffer, the soft remainder
blended), **Blender 2.4x's Z Offset, Invert Z and Env** (an Env
material shades the frame on the CPU by name), **3ds Max's Thin Wall
Refraction** (a screen-space jog, never a ray; the jog scale is
Halcyon's calibration) and **Imagine's fog object** (opacity by
thickness through a closed mesh, unlit).

*Sky and camera.* **Y-shear pitch** (Heretic, Hexen, Build, Marathon;
the caps are ZDoom's), **the cylinder sky** (Doom: the image reads
right-to-left around the turn, as Doom's did, and mirrors do not see
it), **the gradient backdrop with squeeze** (LightWave: the squeeze
curve was never documented — Halcyon's is uncalibrated against a real
LightWave render), **the Mode 7 floor** (SNES / GBA: per-row integer
registers over a 256-colour map; unlit, and reflections see no floor),
**sliced panoramas** (Blender 2.4's Pano + Xparts, kinks kept),
**layer-parallax stereo** (Virtual Boy: one integer shift per object,
not two cameras), **lens-sampled depth of field** (REYES, the SGI
accumulation buffer, 3ds Max's multi-pass: whole renders averaged — an
F12 road, the viewport shows none) and **dithered time-slice motion
blur** (3D Studio / Max's Object Motion Blur, LightWave's Dithered).
One fix rides along: the stereo convergence shift had its sign inverted
— the eyes diverged — so every stereo pair with a non-zero eye distance
changes.

*Colour and palette.* **The palette snap and the ordered dither now
draw on the GPU** — see GPU support. **Palette register depth** (an
ST's 512 colours, the Amiga's 4,096, VGA's 6-bit DAC; the SNES, Neo Geo
and 32X presets' "5-bit" depth quantised nothing before this round),
**Extra Half-Brite** (Amiga: 32 registers and their halves), **CRY**
(Atari Jaguar: the 16×16 chroma square is Halcyon's own derivation, not
the manual's tables — how many cells differ is unmeasured) and **YJK**
(MSX2+ Screen 12). **Attribute cells** (ZX Spectrum, MSX1, C64 hires
and multicolour: two or four colours per cell, fitted by least squared
error — the ZX preset's clash is real now), **per-scanline palettes**
(Spectrum 512, Dynamic HiRes, Sliced HAM: a median cut per line, on the
CPU on both devices by name; Spectrum 512's staggered reloads are three
fixed thirds here), **Super Black** (3D Studio / Max; the mask is the
opaque G-buffer's, and worker processes carry no plane yet) and **Video
Color Check** (Max's broadcast legaliser, NTSC and PAL envelopes).

*Video signal.* **The N64 VI's dither filter and gamma**, **the
GameCube's copy-filter deflicker**, **the PS2's CRTC blend** (against
the previous frame or the background register), **the 3dfx "22-bit"
scan-out filter** (Voodoo Graphics and Voodoo2; the taps are 86Box's
reconstruction from captures, 3dfx never published them), **the 3DO's
2× interpolation** (at one fixed cornerweight) and **the GBA's Mode 5
stretch** (the last two after the final readback, on both devices).
**Chroma subsampling and siting** (D1 4:2:2, DV 4:1:1, MPEG 4:2:0, the
GameCube XFB), **the cable** (S-Video for NTSC and PAL; an RF
modulator with its sound beat, snow and ghost — bandwidth, beat and
snow are dials set by eye), **the PAL receiver** (delay line, Hanover
bars, PAL crawl; the bleed under it is still the NTSC-shaped chain),
**the tape** (nine formats from VHS to Type C by their published
lines of resolution, generation loss, head switching, dropouts),
**MPEG-1 intra blocks** (Video CD; no inter-frame road), **Smacker
blocks** (the thresholds are Halcyon's) and **the backlit matte
glow** (Tron, 1982: mattes exposed through diffusion in the gel's
colour; an F12 stage).

*Presets.* Twenty-two new — Maya 4, Blender 2.41, Sega Model 1, Model 2
and Model 3, Namco System 21, Atari vector arcade, Elite, Voodoo2,
Direct3D 5 retail, PowerVR PCX2, Nintendo DS, Mega Drive, GBA Mode 5,
Amiga Extra Half-Brite, MSX1, MSX2+, Spectrum 512, RF modulator, Video
CD, Smacker and Tron — and an **Arcade Boards** category. **Existing
presets whose pictures move**, because they now run the machine's own
mechanism where a stand-in ran before: PSX and PSX Hi-Res, Saturn, N64,
PS2, GameCube, Xbox, Dreamcast, Voodoo, 3DO, SNES, Neo Geo, 32X, Jaguar,
Virtual Boy, ZX Spectrum, C64, Atari ST, Amiga OCS, MSX2, PC-98, Turbo
Silver, Doom (see-through surfaces, and pitched cameras), LightWave
5.6, RenderMan, Blender Internal (imported buffer shadows), POV-Ray 2
and 3.1, SGI Indy, SGI Broadcast, PAL TV, S-Video, the Toaster, CD-ROM
FMV and the VHS presets — the CHANGELOG names each change. That list was
written from what each mechanism does. Rendered afterwards against 1.89.0
on the demo scene (`docs-dev/r251_presets.log`), 33 of the 39 edited
presets move and six move no pixel there — Saturn, POV-Ray 2 and 3.1, 3ds
Max R2 and 2012, Studio R4 — because their new settings need something
that scene lacks (see *Known limitations*).

*Still open, disclosed.* DS Highlight whitens lit surfaces at the
default table, and its hardware rule is undecided between GBATEK and
melonDS (above). With an error-diffusion dither the two devices draw
different dither patterns: on the driver the Atari ST, Amiga OCS and
Studio R4 presets differ at 351,922, 67,372 and 354,017 of 921,600
pixels with the display transform already on the CPU by name, and at 0,
1 and 4 pixels with the dither off. The console says so since fix pass
4; whether such frames should render on the CPU by name is undecided.
With the GPU rasteriser engaged a few pixels of a 1280×720 frame differ
between the devices by large amounts (38 above 1e-2 on the field's own
frame at 2× supersampling); the cause is not diagnosed. Two rows of the
self-test matrix fail on the release hardware, and three signal rows
(RF, MPEG-1, Video CD) differ past the bar their test states; *GPU
support* has the numbers.
The texture cache is keyed by an array's memory address (so it has been
since before this round): a stale prepared texture was measured once,
headless; whether it can happen inside Blender is not known. During the
field test Blender died silently, twice, after a few dozen 720p variants
in one process (each variant passes alone); the cause is not found, and in
the full run of seventeen fresh processes none died. Two rows of the round's preview collage (the
Mega Drive column mesh and the N64 noise compare) move no colour pixels
— they change the alpha plane only — and the TEV combiner row shows a
plain white ball because its test registers saturate. Known
limitations, below, says which of these a user can hit.

**The Max study, second reading (1.86)** — the field supplied Max
2012's own map and shader algorithms (the MetaSL/HLSL ports that ship
with Max), and the 3DS Max shelf now runs them step for step on
**Halcyon's own hash-derived tables — no Autodesk code or data ships**.
Every 3D map rides Max's 512-lattice gradient noise (Perlin's 1989
noise as max_texutil carries it, 3D and 4D with Phase as time);
**Cellular** is Max's Poisson Worley (point counts per cell, squared
distances, the fractal sum by lacunarity, a Variation control and a
Cell ID output); **Dent**, **Planet** and **Wood** ride Max's
20-periodic linear noise; **Smoke**, **Speckle**, **Splat**,
**Stucco**, **Swirl**, **Checker** (Max's two checks per tile, the
integrated Soften), **Tiles** (Max's mortar geometry, Line Shift,
Random Shift, Color Variance) and **Waves** (wave sets seeded by the
C runtime's rand(), so the same seed places the same sets) follow
their originals; five maps are new — **Marble**, **Perlin Marble**
(the thirteen-knot spline), **Wood**, **Dent** and **Gradient Ramp**
(Max's eleven gradient shapes on a three-flag ramp). **Falloff**
gains Max's exact Fresnel equation, Max's Distance Blend order (Side
near, Front far) with Extrapolate, and Shadow / Light (the Light
Meter's road, CPU-only by name); **Output** gains Output Amount and
Alpha From RGB in Max's order, **Mask** scales the alpha too. Sizes
are Max's Size (coordinate over Size) at Max's own defaults over 100.
The master shader gains **the eight Standard shaders as new models**
— Phong, Blinn, Metal, Anisotropic, Multi-Layer (with its second
highlight's own sockets), Oren-Nayar-Blinn (Max's Oren-Nayar with the
interreflection term), Strauss and Translucent (with a Translucent
Color), each labelled "(3ds Max)" — on Max's own light loops, on both
devices. Every twin is held to its NumPy original at the bit where
the maths allows, and the GLSL simulator itself gained a float32
discipline on the way (a Python literal in a vector constructor used
to promote a lane to float64). Every road that did not change holds
bitwise against 1.85.0. Then, on the field's second ask, **Max's own
material panels as nodes**: **Standard (3ds Max)** — the Standard
material 1:1 with its rollouts (Wire, Faceted, the eight-shader
menu, Ambient/Diffuse/Specular with the lock, Self-Illumination as
Max's own lerp or as a colour, Opacity, Specular Level, Glossiness,
Soften, each shader's extras, Opacity Falloff, IOR, the Diffuse and
Bump map amounts, Reflection) — and **Raytrace (3ds Max)** (Reflect,
Luminosity and Transparency as colours, the way Max reads them),
both showing Max's labels and units over the master's identifiers so
the bake machinery treats them exactly as the master (a linked
percentage refuses the GPU by name rather than shading raw); the
node at Blinn renders bitwise the master at MAX_BLINN. The shelf's
last maps on Max's own algorithms: **Gradient** (with its noise and
`sramp` threshold), the **Composite map** (Max's twenty-five blend
modes and its "over"), **Color Correction** (Max's HSL, Standard and
Advanced lightness), **Vertex Color**, and the **XYZ Coordinates**
rollout in Max's matrix order — and, so Object XYZ could shade on
both devices, **the object frame now reaches the GPU** (the Texture
Coordinate node's Object output had been answering world position
there, a silent split on any moved object). A **3ds Max 2012, Default
Scanline** render preset at 2012's own defaults, gamma off as it
shipped. 35 nodes on the shelf.

**The Max study (1.85)** — 3ds Max's material and map library, from
the 2012 reference rollout by rollout, under a **3DS Max submenu** of
the Halcyon Add menu (which now leads the Add menu instead of
trailing it). *Materials*: **Blend** (two materials by a Mix Amount
or a mask, with the mixing curve's Lower/Upper zone), **Double
Sided** (a Facing and a Back material, Translucency letting each show
through), **Top/Bottom** (by the normal's up, with Position and
Blend), **Shellac** (the second material's colour added by Color
Blend) and **Composite** (a base and four layers, each Additive or
Mix by its Amount). *Textures*: **Noise** (Regular/Fractal/Turbulence
with fractional Levels, Low/High thresholds, Phase), **Cellular**
(Circular or Chips, Spread, Fractal iterations, Roughness, the
Low/Mid/High cut into Cell and two Division colours), **Smoke**,
**Speckle**, **Splat**, **Stucco**, **Swirl** (Twist, Intensity,
Amount, Constant Detail, Center, Contrast, seed), **Planet** (three
water and five land colours by depth and height, Ocean %, Island
Factor, Blend), **Waves** (wave sets on a sphere or circle,
wavelength range, Amplitude, Phase), **Checker** (with Soften) and
**Tiles** (Stack, Running, English and Flemish bonds, counts, gaps,
Holes, Fade Variance) and **Falloff** (Perpendicular/Parallel,
Towards/Away, Fresnel, Distance Blend, against the view or a world
axis). *Utilities*: **Mix** with the mixing curve, **RGB Tint**,
**Output** (Level, Offset, Invert, Clamp), **Mask** and **RGB
Multiply**. *Vectors*: the **Coordinates** rollout (map channel,
object or world XYZ, screen; Offset, Tiling, Mirror, W Angle). Every
map's GLSL twin is held to its NumPy original point by point and
every node to the CPU frame on the deferred pass. The master shader
gains Max's **Oren-Nayar-Blinn** model and a **Faceted** flag, and
the Cartoon Era menu gains **Ink 'n Paint** at Max's own defaults
(with a shelf recipe). Housekeeping the field asked for: the
texture shelf is categorised (Noise & Fractal; Stone, Wood & Organic;
Tiles & Fabric; Sky, Water & Effects; 2D Media), Matcap Coordinates
lives in the Vector family and the BI Texture in the Blender
Internal family. One engine change came out of the study: an Add
Shader of two Halcyon masters now SUMS their colours on the CPU as
the GPU always did (it averaged before — a silent divergence the
Shellac road exposed); every mix holds bitwise.

**The hair pass, the film scans and the tooltips (1.84)** — three
finishing fronts. The **hair highlight** grows up: the angel ring's
edge takes a **Shape** menu (the smooth wave it always had, plus the
zigzag "W" highlight of 90s TV, the scalloped run of petals, and the
hard stepped notch of the digital era), a **Hair Shine Angle** that
turns the wave's phase around the head (stagger the notches on a
second character with nothing else changed), a **Hair Shine Follow**
that lets the band ride up and down with the key light's height —
0 is the painted band that ignores lighting, exactly as before; it
listens to the scene key's elevation and stays put under a purely
positional lamp — and a **Hair Shine Second Color** so the echo band
below can carry its own tint rather than a copy. The **Cartoon
master takes the whole hair bag** — all eleven sockets, the same
code — because the American tradition airbrushed hair too. Both
devices render every new road identically; every knob at its default
is bitwise the 1.83.0 engine. The resolution shelf gains **24
formats**, including a whole **Film Scans** group (35 mm Full
Aperture and Academy in 2K and 4K, Super 16, and the CinemaScope 2K
scan whose 2:1 anamorphic pixels un-squeeze on the way out — render
spherical, deliver scope), the missing home computers (Amstrad CPC,
VIC-20, the Atari ST's three modes, Sharp X68000, NEC PC-88), the
handheld and console gaps (TurboGrafx-16, 3DO, Game Gear, Lynx,
WonderSwan), the laptop panels every deck actually shipped on
(1366×768, 1600×900, XGA+, WQXGA, QSXGA) and the modern ultrawides
(DQHD, 5K2K, the 4:5 social portrait). And **every option in the
add-on now explains itself**: every socket on the three master
shaders, every node property, every enum entry, every render, world,
material and light setting carries a written tooltip — applied to
old files' nodes the moment they load, and enforced by the suite so
a thin tooltip is a failing test from here on.

**The preset expansion (1.83)** — more of both traditions, on one
menu each. The Anime Shader gains a **Style menu** (the Cartoon
Era menu's idea for the anime decades): *80s Film Feature* (one
restrained warm kage under the optical printer's softness, matte
paint, a wide gentle hair sheen, a breath of airbrush), *80s TV* (the
hard cool-violet tone and the waving angel-ring), *80s OVA* (the
video market's glamour: two kage tones, a Fresnel rim, a double hair
shine, airbrush both sides), *90s TV* (cooler, more saturated, a
bolder simpler hair band), *2000s Digital* (dead-hard bands, the
early RGB palettes' desaturated kage, a crisp waveless shine), and
*Modern* (a gently softened tone, the depth rim, the marched contact
shadows and the camera key — the current pipeline in one click).
Every value is a starting point; the menu never touches your paints,
textures or the key's angle, and the shadow colours are multiply
tints, so one style serves any palette. The Cartoon Era menu grows
**Wartime Noir** (the dark 40s theatrical short: a deep hard
transparent shadow pushed onto the lit side) and **Modern Flat** (one
barely-darker cool tone over heavily simplified forms — the
thin-uniform-line era). Five more **Cel & Film** render presets
bracket the century: the worn silent nitrate with the projectionist's
cue discs, the 35 mm anime feature, the 2000s digital transition (no
film in the chain at all), the modern 1080p master with its one fine
grain sheet, and the modern flat TV cartoon. Three new shelf recipes
tie in, generated from the same tables so the shelf and the menus can
never drift (112 templates).

**The Xrd study (1.82)** — the Guilty Gear techniques, from the
sources themselves: Motomura's GDC talk and Arc System Works' own
line-control deck. **Line Control** (the material's Ink row): the
mesh's vertex colours steer the line exactly as ASW paints them —
ALPHA multiplies the width (0.5 the width as set, 1 double, 0 erases
the line at that vertex: a glove's rim, a sleeve's opening), BLUE
holds interior and marked lines back until the surface turns toward
its silhouette (their hull's depth push — the nose line that only
draws in profile); silhouettes always keep their line, and Halcyon's
screen-space ink never needs their distance compensation. **Vertex
Colour AO now works under every Compatibility mode** — the GDC talk's
"offset on the Threshold" for hand-painted models, no game texture
required. **Face Shadow (SDF)** on the Anime Shader: the modern anime
face's terminator is DRAWN in a map, not found on the normals — link
the game's own face map (or any gradient chain) and it sweeps with
the key's angle about the head's frame (Face Forward / Face Up),
mirrored across the face's centre line, exactly the convention Guilty
Gear and the HoYo titles share; cast and screen shadows still land on
it. And the study's foundation was already here: **hand-edited custom
split normals travel** — Blender's Data Transfer from a sphere, the
Normal Edit modifier, any sculpted normals reach both devices, and
because Halcyon's line is a post-process, edited shading normals
never bend the outline (Xrd needed a second normal set for that; a
distance-field line gets it by construction).

**The cel's light (1.81)** — better shadows and better lighting for
the 2D shaders, the way the drawings and the 3D-anime pipelines
actually do them. A hand-drawn cel is not lit by lamps in a room: the
key is decided per shot, its cast shadows are drawn shapes, and the
silhouette carries a rim. The Cartoon and Anime Shaders now name their
**Shading Light**: *Scene Lamps* (as before), *Camera Key* — one key
fixed to the camera by **Light Azimuth** and **Light Elevation** about
the screen, so the character is lit the same way on screen wherever
the camera goes — or *World Key*, fixed in the world. Under a fixed
key the lit tone is the paint as painted, and the scene's lamps only
CAST: their shadow maps still fall across the cel. **Screen Shadow**
is the drawn contact shadow — the hair's across the brow, the chin's
on the neck — marched from the frame's own depth toward the key over
**Screen Shadow Length** pixels: what is nearer along the way shadows
the pixel, crisp and close, on both devices from the same field. **Rim
Mode ▸ Depth Rim** is the anime rim: a band **Rim Width** pixels wide
just inside the silhouette, wherever the surface a few pixels along
the screen is farther — a line of light that keeps its width whatever
the form, on the *Lit Side*, the *Shadow Side* or *Both*. **Smoothing
Shape** gives Shadow Smoothing a *Sphere* (as before), an upright
*Cylinder* (the terminator runs straight down a limb) or *Facing the
Camera* — the anime face rule, the whole face one lit plane with only
the drawn shadows on it. The Cartoon Shader gains the Anime Shader's
**Airbrush** (colour, width and side) against its own shadow edge.
Three templates carry the recipes: Cel Camera Key, Cel Anime Face,
Cartoon Rounded Feature. Every existing file renders bitwise as
before.

**The print's wear (1.80)** — the 40s arc's last front and the
field's ask (*"the grain and hairs need to be much more customizable
and interesting"*). Display ▸ Cel Film ▸ **Print Wear**, each thing
where it happened to the film: **Grain** is the negative's silver
printed through — a sheet of grains of Grain Size, a Grain Clump share
in clumps, each record's own sheet by Grain Chroma (coloured grain, the
way three negatives print), density noise strongest where half the
grains developed and nothing at the clear gate or at D-max, so a
print's whites and blacks are clean. **Dust** in three populations: Cel
Dust sat on the cel under the rostrum camera and stays for every frame
of a hold; Negative Dust blocked the printer light and prints clear —
one record's colour missing under the three-strip process; the rest is
dirt on the print, fresh per frame. **Hairs** are caught in the
projector gate: anchored at the aperture's edge, dancing for their Hair
Hold, about Hairs at a time. **Scratches** run the frame's height at
one place for their Scratch Hold, on the emulsion side taking the dye
away (neutral on a dye-transfer print, one colour on a two-colour one,
blue on a chromogenic one) or on the base side scattering the lamp.
**Reel Length** puts the projectionist's cue marks top right — four
frames, eight seconds and one second before every reel's end. Sizes are
given at 1080 lines and scale with the frame's height. The chain now
runs in the order the light met it: cel dust, optics, process, grade,
the print's wear, dirt, weave, hairs, flicker.

**The colour process (1.79)** — the film look rebuilt around what
actually happened to the cel, the study's fourth rank and the field's
own complaint (*"the coloured ones don't really work well, it doesn't
look like the cartoons do"*). Display ▸ Cel Film ▸ **Colour Process**:
not a grade but the chain — the camera's black-and-white **records**
through their red, green and blue filters (a successive-exposure or
three-strip camera; the filters' cross-talk included, and **Filter
Sharpness** for the taking filters' cut: sharp-cutting filters separate
the records further than the eye separates the colours — the three-
strip's more-than-life saturation, from the filters, not a saturation
dial), **halation** in the negative's base (each record's bright areas
veil their surroundings in their own colour), the timed print's
**curve** to a dye amount (the Hurter–Driffield straight line of Print
Gamma in log exposure, pinned so a white cel prints clear and no light
prints Dye Density, the H&D toe between; a mid grey prints at
0.18^gamma — 1.5 is the theatre's dense print, 1.3 that print as a
telecine shows it), the dye-transfer print's
three dyes with their documented **unwanted absorptions** (a cyan that
also eats some green and blue, a magenta that eats red and blue, a
yellow that eats green — balanced so a grey scale prints neutral and
only colours shift: reds deep, greens darker and toward cyan, blues
toward purple; Dye Purity walks toward the process's ideal dyes), the
**silver key**
printed from the green record under the dyes, the dye layers'
**registration** error (every edge, the ink line included, fringed in
colour — the honest misregistration of a printed cartoon), and the
**projector** (the screen's white is the clear gate of a normally timed
print). **Two-colour** is the same chain with two records and two dyes
— Cinecolor and the 1930s two-strip: no true green, skies cyan, foliage
olive, skin salmon; its dyes' green absorptions are the process, not an
impurity. Exposure is the timer's printer light in stops: down prints
the whole frame denser and darker, up thinner and paler. Three new Cel &
Film presets — *1941 Fleischer (Superman)*, *1940s Cinecolor short*,
*early-30s two-strip Technicolor* — and the 1940s Technicolor feature
preset takes the process instead of the grade. The 1.73 stock grades
are untouched and still there for the video era.

**Media 2 (1.78)** — the study's second media front. **Direction** on
Hatching and Pencil Scribble: *Form* runs the strokes along the
silhouette-parallel tangent (the normal crossed with the view — the
isophote direction of the classic pen drawing, which wraps a sphere like
latitude and follows a cylinder's length), *Slope* runs them down the
form; a turning direction cannot be one lane field, so the node keeps
twelve coherent fields at 15-degree steps and cross-fades the two a pixel
sits between, and a stroke never breaks. **Indication** on every
converter — Winkenbach and Salesin's rule that a drawing shows detail
only where the artist indicates it: link any mask (an inverted Facing
draws near the silhouettes and leaves the middle bare; a vertex colour
paints where). **Placement: Count** on Stipple is Secord's stippling —
dots of one size whose number carries the tone, from three nested
lattices in a fixed order so a darkening tone only ever adds dots and the
crowd is evenly spread at every density. **Blend** on Pencil Scribble and
Charcoal is Sousa and Buchanan's tortillon: graphite taken off the paper's
peaks and pushed into its valleys — the stroke keeps half its contrast
and a soft halo appears around it; the charcoal's tooth flattens to an
even tone. Every twin held to media.py; nodes saved before the sockets
read Indication 1 and Blend 0 and grow the sockets at load; four shelf
recipes (Form Hatching, Indicated Hatching, Counted Stipple, Smudged
Pencil). And the GLSL front-end refuses a call with the wrong number of
arguments now, as every driver does — it used to zero-fill them.

**The inker's line (1.77)** — the Imitation Study's line arc 2, chosen
by the field as the front after the backdrop. Three roads to a line
that is *"dynamic and occasionally rough"* instead of traced. **Light
Weight** is Goodwin's isophote distance (NPAR 2007): every silhouette
point walks inward along the surface's own normal until the key lamp's
light returns (Shadow Level), and the line is as heavy as that walk is
long — heavy on a broad shadowed flank, a quarter width on the lit side
and on thin features, heavier near than far, all from one measurement;
the walk stops at the object's edge and at any depth step. Three more
line sources on the Outline panel, each an interior class at the
Interior Scale on both roads: **Form Lines** (the valleys of the facing
|n·v| — DeCarlo's image-space suggestive contours, found with Steger's
sub-pixel valley detector so the line is one pixel wide and the
silhouette's own fall-off never fires), **Shadow Lines** (the terminator
inked, as the cel inker traced the painter's shadow boundary) and **Tone
Lines** (a flow-guided difference of Gaussians on the shaded frame —
Kang's FDoG, Winnemöller's XDoG — one pixel wide on the dark side of
every tone step, a line only where the step is within one surface). And
**the stroke road**: the contour is read as a graph (every pixel and the
runs of its eight neighbours — two runs a path, one an end, three a
junction) and everything after is a local operation on it, so a pooled
band draws what the frame draws: **Smooth** (Taubin's shrink-free
smoothing of the pixel staircase, corners pinned), **Pressure** (the
net turning of the chain — a staircase's jogs cancel, a curve keeps its
rate, a corner its whole angle — bears the pen down through curves and
leaves a blob at every corner), **Overshoot** (true ends, chains
arriving at junctions and both edges of every corner run past it along
their tangents, tapering, about half of them, each by an amount hashed
from the point on the surface, so animation holds), and the seeds are
drawn back carrying their **origin** pixel so every width, colour and
depth lookup still reads a surface. **Anchor: Surface** samples the
line's noises (weight, roughness, drift, gaps) at the surface point
under the line instead of the screen — Kalnins' coherence without a
previous frame. Also: the shadow-side weight and the Light gradient
read the smooth normal and a point lamp's true direction now (a point
lamp's rotation was being read as its direction). Every dial band-
invariant bitwise, deterministic, off by default: the plain and styled
lines are the previous release bitwise.

**The painted backdrop, the painted background road, the setback (1.76)**
— the pagoda's first tell was its sky, a flat tint in a frame that was
otherwise drawn. World ▸ **Painted Backdrop** is a background painting on
a flat panel in front of the camera (aim it with Panel Direction): the
gradient brushed in gouache — the brush's streaks along a stroke angle,
optional impasto dabs — painted clouds from a stretched three-octave
field, dry-brushed or airbrushed at the edge, lit on top and shadowed
beneath by the field's own vertical difference, the board's tooth, and a
watercolour granulation applied as pigment density in Bousseau's law
(C′ = C − (C − C²)(d − 1)). A pan crosses the painting and a tilt climbs
it, the way the animation stand's background did; behind the panel the
plain gradient continues, the back of the stage. Four looks on a menu:
Gouache Day, Watercolour Dusk, Fleischer Night, Storyboard. Evaluated per
direction on the CPU on both device roads, like the Bryce sky (the simple
skies and HDRI draw on the GPU since 1.89.0) — exact by construction;
mirrors reflect it by the same road. **Paint Mode** on
every material: *Cel* (everything as before) or *Background*, which
takes the **painted background road** (Display ▸ Painted Backgrounds):
the surface's lit colour laid down as brush strokes Meier's way
(SIGGRAPH 1996) — particles fixed on the surface, one hashed sequence per
triangle cut to the screen density the stroke size asks, so a camera
that comes closer adds strokes at the end of every sequence and never
moves one already there; each drawn far to near at a screen-constant
size over an abstracted base, its direction from the colour's contours,
the surface's silhouette or a fixed angle, bristle-streaked, drying
toward its end, lightened or darkened by its own hash — with the ink
suppressed on the painting unless the material says Always. The
**setback** is the Fleischer lens on the miniature: Background materials
and the sky soften by eye distance while the cels stay sharp, a masked
Gaussian so nothing bleeds across the cel's edge. One CPU pass from the
shared G-buffer before the ink on either device road; band-invariant
bitwise; a cel pixel is never touched; the default paint mode is the
previous release bitwise. And the media nodes gained a third **Space**,
*View* — the camera's sphere, the eye-to-point direction octahedrally
unfolded — Lucas Pope's Obra Dinn answer to the shower door: turn the
camera and the marks stay on the scene.

**The 2D media (1.75)** — the verdict's second sentence: *"there should
be textures/converters specifically for 2D-looking things — paint
strokes, pencil scribbling, charcoals, inking"*. Seven texture nodes on
the Halcyon Textures shelf, five of them **converters** with a per-pixel
**Tone** input (lightness, 1 = bare paper — a Shader to RGB luminance, a
Screen Info Facing, a dot with a light direction): **Hatching** (pen
cross-hatching that fills in layer by layer as the tone darkens — the
tonal art map, continuous — each stroke wobbling, varying in pressure,
drawn in lifted segments, breaking dry, and swelling to solid ink over
the last fifth of the range), **Pencil Scribble** (hatch lanes bent by a
curling domain warp that folds into loops past Curl 1, fanned in shallow
layers, the graphite catching the paper's tooth so light pressure
speckles and heavy pressure fills), **Stipple** (dots on a jittered
lattice whose *density* carries the tone, a finer lattice closing the
darks, the dots merging into solid at the very end the way a stippler
reaches black), **Charcoal** (the stick depositing on a four-part paper
tooth, peaks first, streaked along the stroke, smudged by a thumb,
banded by its own pressure), **Ink Wash** (flat washes by level, pigment
pooling inside every edge, settling into the granulation, the edges
wandering with Bleed); and two textures: **Paint Strokes** (brush dabs
on a jittered lattice, later over earlier through a two-deep painter's
composite, turned off the angle by a hashed *slope*, bristle-streaked,
drying toward their ends — link a lit colour into Color 2 and the
strokes carry it, the gouache road) and **Paper** (tooth, fibres, mottle
as a height). Every one has a **Space**: the Vector input (the marks on
the surface) or **Screen**, the camera's own paper — Scale cells across
the frame's height whatever the resolution — and a **Boil** that redraws
the marks every N frames through the salt, so a drawing boils on its own
clock like the line does; a **Seed** keeps two materials' hatching from
being the same hatching. Every field is the integer hash the pattern
library rides, so every one travels to the GPU exactly — the seven GLSL
twins are held against the NumPy originals on random points and whole
materials against `render()`; a linked dial refuses by name, the boil
rides the frame uniform (no recompile per frame), Screen space is
band-invariant bitwise. Seven **2D Media** shelf recipes: five shadeless
drawings on a Facing tone (hatched, pencil, charcoal, stippled, wash),
a Gouache Background, and Drawing Paper with its own relief.

**The drawn line (1.74)** — the verdict on the ink pack was "decent, but
they look artificial; old cartoon outlines are dynamic and occasionally
rough", and this round answers it on the distance-field road. **End
Taper / End Length**: the brush lifts and lands — every stroke end and
junction is found on the contour itself (a segment has two, a line that
vanishes behind an object ends where it meets that object's contour, a
closed silhouette has none) and the line thins toward them.
**Roughness / Roughness Scale**: three octaves of edge irregularity, in
*patches* — a slow field decides where the line gets rough, so it is
occasionally rough rather than uniformly hairy. **Drift**: the hand
wanders — the line's centre moves in and out of the true contour, the
same on both sides of the line. **Gaps**: dry-brush skips, short breaks
where a fine noise runs high, more readily where the stroke is thin.
**Line Texture**: solid ink, *Brush Streaks* (lighter streaks running
along the stroke — the along coordinate is the source pixel's, the
across coordinate the signed distance, so the streaks run parallel and
unbroken through the line's centre) or *Charcoal* (a soft bloom around
a dark core, the paper's tooth breaking the body). Every dial is a hash
of pixel, phase and seed: band-invariant bitwise, the same frame the
same line; every default is the previous release's line bitwise. The
Cel & Film presets draw with them.

**The era looks (1.73)** — the cel arc's last front: the cel
*photographed and printed*. Render Properties ▸ Display ▸ **Cel Film**.
**Film Stock** is the stock's colour response in linear light —
Technicolor's purified primaries and cyan shadows, a faded 70s
Eastmancolor print (magenta cast, lifted blacks), the 80s telecine, a
VHS dub, or black-and-white panchromatic negative — with a Stock Amount
mix. **Softness** is the rostrum camera's optics (a Gaussian, sigma in
pixels); **Gate Weave** shifts the whole frame by a per-frame
registration error; **Dust & Hairs** puts dirt, clear scratches and the
odd hair on the print by density (per film area: about forty specks on
a 1080p frame at 1); **Grain** is the emulsion, seven parts luminance to
three parts colour, mean-preserving; **Flicker** is the projector lamp.
**Shoot On** 2 or 3 is the animator's economy made literal: a held frame
is *not rendered* — the key frame's cel is photographed again (the film
stages still run per frame, so grain, dust and weave keep moving) and an
animation on twos renders in half the time; render sequences from their
first frame, and a frame whose key frame is missing says so and renders
fresh. The **Paint** box works before the ink: **Paint Misregistration**
slides the painted colour by a per-frame error under lines that stay
where the drawing put them, **Paint Bleed** softens the paint under and
past the line. The **Print** box is the newsstand: **Halftone** lays four
ink screens at the classic angles (cyan 15°, magenta 75°, yellow 0°,
black 45°), each dot's area its ink's coverage with the classic inversion
above 50%, full under-colour removal, reproducing every grey within a
percent; **Halftone Pitch** is the cell in pixels. Every stage is a pure
function of the frame number and the seed — the same frame renders the
same bits on either device, in any pooled band — and every default is
off: the frame is bitwise the previous release's. Six **Cel & Film**
render presets set a period at once: 1930s black-and-white, 1940s
Technicolor feature, 1970s Saturday morning (on threes, through the
composite chain), 1980s OVA on LaserDisc, 1980s TV anime on VHS, and
the comic print. The full Technicolor chain costs about 0.35 s at 1080p;
the dot screen about 0.5 s.

**The 80s additions to the Anime Shader (1.72)** — the cel arc's third
front. **Hair Shine** paints the angel ring: a band of the shine colour
across the object at a fraction of its own height (Height), Width wide,
its edge waving around the object (Wave, Waves per turn), on the
camera-facing surface, fading only on the underside — light-independent,
because it was painted on the cel at a position, not lit — with a
thinner **Second** band that far below. **Airbrush** lays the cel
painter's soft gradation against the first shadow edge: on the lit side
the tone multiplies toward the airbrush colour just above the edge and
fades out toward full light, on the shadow side the shadow tone blends
toward it approaching the edge, or both (Airbrush Side), Width in light
units. **Shadow Smoothing** is the Cartoon Shader's inker's road, shared.
And the **Line Colour** menu: *Ink Settings* (as before), *Iro-Trace* —
this material's ink line in its own shaded colour per pixel, darkened
by Line Darken (hair lines in the hair's tone, skin lines in the skin's,
the 80s coloured trace line; it runs the distance-field ink road), or
the *Line Color* socket as a constant of the material's own. Every
addition is neutral at its default: a node saved before them shades
bitwise as it did, and old files grow the sockets at load. GPU twins
for all of it, per-pixel amounts and colours included. Two shelf
recipes, **Cel 80s Hair** and **Cel 80s Skin**, arm them. On the way, a
latent fix to the per-material ink road: a silhouette line now belongs
to its nearer surface on the mask road too, so a material's coloured or
widened line is whole on its sky side (it was half the global line).

**The Cartoon Shader (1.71)** is the third master node and the OTHER
tradition — "a 40s Looney Tunes cartoon is vastly different to an 80s
anime". Its colour is **paint, not light**: flat, the same under every
lamp whatever its energy (a lamp at 40 W paints the same cel a lamp at
2 W does). Where the key lamps do not reach, one painted **shadow
tone** lands, by mode: **Transparent Cel** (paint × Shadow Color by
Shadow Amount — the Golden Age shadow cel, a second exposure through a
tinted overlay), **Painted** (a second flat paint replacing the first —
UPA and the television decades), or **None** (flat paint everywhere —
limited animation at its cheapest). Shadow Threshold is where the edge
falls on the wrapped light term (0.5 the terminator, higher pushes the
shadow onto the lit side), Shadow Softness airbrushes it, and **Shadow
Smoothing** bends the shading normal toward a sphere around the object
so the terminator sweeps a form as one clean shape the way an inker
simplifies it (a sphere is its own smoothed self, bitwise; a cube gets
a curved terminator across a flat face). The **highlight is a painted
dot**, not a reflection — Highlight Size is its radius (0.3 ≈ a
17-degree dot, 1.0 a 60-degree cap), Highlight Softness feathers it, and it only lands
inside the lit region of the lamp that makes it. Across lamps the paint
keeps the **strongest lamp's verdict** (a region is lit if any lamp
reaches it — the paint never brightens past paint), and **Lamp
Influence** is the one road back toward lit shading: 0 is pure paint,
1 lets the lamps' energy and colour modulate the LIT paint — the
shadow tone is a colour and stays one. Rim, emission and
opacity ride the master's shared tail. An **Era** menu writes the
sockets to a named starting point — Golden Age (1930s-40s), UPA Modern
(1950s), Xerox Era (1960s-70s), Saturday Morning (1970s-80s), 90s
Feature, 90s TV (Dark Deco) — and the sockets stay yours afterwards.
Shades identically on CPU and GPU, per-pixel chains into Shadow Color,
Shadow Amount, Shadow Threshold, Shadow Smoothing, Highlight Color,
Highlight Size and Lamp Influence included; a chain into a softness
socket refuses the driver by name. The master shader's model menu
carries **Cartoon (Paint)** too, running the node at its defaults.

**Convert to Halcyon chooses its shader your way (1.62).** The Material
panel's converter grew a Detection switch: **Automatic** keeps reading
the source shader (Principled roughness becomes Glossiness, Glass
becomes Blinn, Toon becomes Toon), **Set Shader** forces one chosen
model — any of the 19, the Anime/Cel bands included — onto everything it
converts.

**143 resolution presets in nine categories (1.62, 1.84, 1.90)** — the
Super FX's three heights (256×192 for Star Fox, 256×160, 256×128, at the
SNES's 8:7) joined in 1.90; the shelf grew from 75: the handhelds and arcade boards (NES at its true 8:7 PPU
pixel, Game Boy, GBA, DS, PSP, Neo Geo, CPS-2, Virtual Boy), the 8-bit
micros (ZX Spectrum, C64, MSX, Apple II, Atari 8-bit, BBC Micro Mode 0
with its double-tall pixels, NEC PC-9801), DVD and the anamorphic tape
formats (HDV 1440×1080 at 4:3 pixels, DVCPRO HD), bake textures to 4K,
DCI cinema, 5K/8K, ultrawide 21:9, vertical 9:16 and A3/US-Letter print.

**Cartoon outlines.** Ink drawn from the renderer's own G-buffer: object
silhouettes, material borders, depth breaks and normal creases, each its own
toggle, with width, colour, opacity and over-sky control. **Marked Edges**
draws the edges you marked — Freestyle edge marks, Sharp, creases — as
interior ink, hidden-line-removed by the z-buffer's own verdict (each pixel
measures its exact distance to its own triangle's marked edges; a marked edge
behind a wall simply is not there). And **every material has a say**: Ink
mode (follow / always / never — one inked character in a plain scene, or
clean glass in an inked one), its own ink colour, its own width. The ink
lands at the *internal* resolution, so supersampling anti-aliases the line on
the way down — clean cel edges with no line renderer. Computed from the same
buffers on either device, so the picture cannot differ between them. Render
Properties ▸ Shading ▸ Cartoon Outlines, and the material panel's Ink row.

**114 material templates on the Pre-Made shelf** — Add ▸ Pre-Made in the
shader editor, sorted into fourteen families (Metal, Mineral, Glass,
Water, Liquid, Wood, Cloud, **Cel & Anime**, **Cartoon**, **2D Media**,
**Volumes**, Terrain, Surfaces, Effects). Picking one drops its nodes beside your
existing graph rather than replacing it. The **Cel & Anime** shelf
(1.68) is the Anime Shader one click deep — a skin/cloth/metal cel trio
plus ArcSys and HoYo starters with their compat modes armed, and (1.72)
the 80s hair and skin with the shine band, the airbrush and iro-trace
lines armed; the
**Cartoon** shelf (1.71) is the Cartoon Shader one click deep, one
recipe per era — Golden Age, UPA Modern, Xerox Era, Saturday Morning,
90s Feature, 90s TV — generated from the node's own era table so the
shelf and the Era menu can never disagree; the **Volumes** shelf drops a
Halcyon Volume wired to the output's Volume socket — Fog Bank, God Ray
Chamber, Cel Fog, Ember Glow, and (1.69) Fire on the Combustion model,
Murky Air, and a Voxel Cloud on a 12-cube lattice — so any mesh becomes
a marched container.
The engine's own 28 — Chrome through Wireframe, Water, Lava, Tile Floor,
Brick Wall, Hammered Metal, Leopard, Cloth, Dead Channel — plus:

**The Bryce 1995 library, decoded — 46 presets.** The original preset
collection ships as MetaTools CCmF containers, and Halcyon's parser
(`core/brycemat.py`) cracks them completely: the RLE preview thumbnails,
the 976-byte material records (diffuse/ambient/specular colours, the
per-channel specular coefficients, transparency, reflection, bump, the
metallic flag, the refractive index — pinned by diffing the teaching
presets against their own 1995 manual text), and the texture records
with their exact colour tables, frequencies and octave counts. Polished
Gold through Kryptonite carry the stored numbers: the glass ladder
refracts at 1.12/1.52/1.68, Water at 1.33, Diamond at 2.55, and
Kryptonite is the anisotropic reptilian-scale crackle its own colour
table says it is. Preset notes quote the original library descriptions.
Built at runtime as recipes rather than saved node trees, so they always
match the current nodes, and every one is rendered by the test suite to
prove it changes the frame.

**34 procedural texture nodes**, 27 of the kind these packages shipped —
Marble, Wood, Granite, Dents, Crackle, Plasma, Ripples, Caustics, Starfield,
Weave, Scratches, Tiles, Spiral, Cells, TV Static, Fur Tufts (the shell-fur
field: round strand cross-sections that taper per layer) and the POV-Ray
family (Bozo, Agate, Leopard, Onion, Bumps, Wrinkles, Brick), plus:

- **Fractal Noise** — the integer-hash fractal in three profiles, on a
  **1D, 2D, 3D or 4D lattice** with a W socket. The hash travels to the GPU
  bit-for-bit, which Blender's own sin-fract noise cannot.
- **Water Noise** — 1 to 12 drifting layers folded toward crests by
  Choppiness, with animation Speed and a **Loop** that closes the cycle
  *exactly* over Loop Frames by moving time onto a circle through the
  lattice's fourth dimension. A looping deck breathes and shimmers in place
  rather than drifting; that trade is stated on the node.
- **Gradient (Shaped)** — linear, reflected, spherical, quadratic, square,
  diamond, conical and spiral falloffs with Centre, Rotation, Scale,
  clip/repeat/ping-pong and easing.
- **Matcap Coordinates** — sphere-map UVs from the view-space normal, now
  with an Offset socket and a **Centered** switch, so a Spherical gradient
  fed the vector lands centred instead of cornered.
- **The 2D media (1.75, 1.78)** — Hatching, Pencil Scribble, Stipple,
  Charcoal, Ink Wash (converters on a per-pixel Tone and an Indication),
  Paint Strokes and Paper, each with a Space (surface, the camera's paper
  or the camera's sphere), a Seed and a Boil; hatching and scribble with a
  Direction (angle, form, slope), stipple with a Placement (size, count),
  pencil and charcoal with a Blend.

Each pattern is written from its published definition rather than from
something that looks similar. Agate's 0.77 exponent is the whole character
of that pattern. And where the published definition looked nothing like the
real thing, the real thing won: **Leopard draws true rosettes** — curved
dark arcs broken around a warmer interior patch, the odd solid spot between,
a gentle domain warp bending every ring organic — with ground, ring and
Interior colours on separate sockets and the ring mask on Fac, instead of
POV-Ray's summed-sines polka dots.

**30 more nodes in two families (1.60).** Fifteen **Utilities** — among
them a **Light Meter** that reads how lit the surface actually is at shade
time (Fac and per-lamp Color out, optional shadow test), which is the whole
trick behind glow-in-the-dark and light-reactive materials; a Timer,
Oscillator, Counter, Pulse, Gate, Selector, Color Key, Measure, Step Ramp,
Wobble, Frame Blend, Blackbody, Compare and On Frame. Light Meter reads the
lamp list at shading time, so the GPU plan refuses it by name and that
material shades on the CPU. And fifteen **Vector** nodes — among them
**Array**, which lays a decal's UVs out along a line, grid, circle, square,
polygon or star with per-copy jitter and rotation (made for stamping image
decals; following seams, creases or sharp edges needs mesh edge flags the
exporter does not carry yet, and is stated on the node rather than faked);
plus Mirror Tile, Kaleidoscope, Polar, Twirl, Lens, Ripple Warp, Wave Warp,
Tile Random, Snap, Shear, Orbit, Region, Projector and Spin.

**The Add menu is families now.** Add ▸ Halcyon opens Pre-Made, then
Shading, Blender Internal, Utilities, Vector, Retro Screen and Textures as
submenus — 92 nodes (64 shader, 27 texture, BI Texture) sorted where a
person would look for them, and a census test proves every registered node
is reachable from exactly one family.

**Colour tools with taste:** a **Color Ramp (Spaces)** node blends up to six
stops in RGB, **OKLab**, **OKLCh** (hue taking the short way round) or HSV —
stop colours are sockets, so they can be driven — and a **Blur** node that
truly re-evaluates whatever is plugged into it at shifted points in the
surface plane, procedural or image alike, at three tap qualities. Blur's
re-run is CPU work and says so: the GPU plan refuses it by name and that
material shades on the CPU.

**Coded shader nodes.** A real compiler — preprocessor, recursive-descent
parser, type inference, and NumPy code generation with SIMT execution masks.
Declare a uniform and an input socket appears, named and defaulted from the
declaration:

```glsl
uniform vec3 tint = vec3(1.0, 0.6, 0.2);
uniform float rimPower = 2.5;

in vec3 vNormal;
in vec3 vView;

out vec4 Color;

void main() {
    float rim = pow(1.0 - abs(dot(normalize(vNormal), normalize(vView))), rimPower);
    Color = vec4(tint * rim, 1.0);
}
```

That gives you a node with **Tint** and **Rim Power** sockets and a **Color**
output. Edit the source, press Compile, and the sockets rebuild while keeping
every link that still has somewhere to go. Coded shaders compile natively into
the GPU's deferred pass too, mangled and inlined, so the same source runs on
both devices.

**Eleven sky modes** — node tree, solid, gradient, **banded gradient** (the same
blend cut into the handful of flat steps a 256-colour palette could actually
spare for a sky), **starfield**, Preetham physical sky, HDRI with rotation and
tint, and a full **Bryce Sky Lab**: sky dome, sun corona, haze that warms
toward the sun, separate ground fog, a wind-streaked stratus deck, a
self-shadowed cumulus deck built from turbulence rather than fBm (which is
where the cauliflower edges come from), a rainbow at the correct 42 degrees,
stars, comets that travel their own great circles off the scene's clock — and
now a **nebula wash** under the Bryce dome as well as the starfield, because a
night preset that sets nebula settings should get a nebula. The ninth,
the **Painted Backdrop** (1.76), is described with the 40s arc above; the
tenth and eleventh (1.90) are Doom's **Cylinder Sky** and LightWave's
**Gradient Backdrop** with its Sky and Ground Squeeze, described with the
period features.

**303 sky presets, every one with a rendered thumbnail.** The original 43 plus
**260 new skies in 26 authored families** — Golden Hour, Blue Hour, High Noon,
Storm Front, Ember, Midnight Clear, Vapor Dream, Winter Overcast, Desert Dusk,
Alien Twilight, Monsoon, Candy, Nebula Night, Fog Bank, Thunderhead, Moonrise,
Cinema Grade, Arctic Night, Sea Dawn, Toxic, Haze Valley, Comet Watch, Cirrus
Day, After the Rain, Noir and Spring Front — each variant carrying its own
tooltip. The preset picker is a browsable gallery: the engine's own sky module
drew all 303 thumbnails, and a test holds the library to it — real fields only,
real notes, a thumbnail per key, and every preset must actually change the sky
it is applied to. *Save As...* writes a `.halsky` anywhere, *Add to Library*
puts one beside the built-ins. Applying a preset refreshes the rendered view
immediately — it tags the world the way a slider does, which for a long time it
did not.

**48 water presets** in the same shape, under the water plane rather than the
Sky Lab because that is where Bryce kept them — including a Bryce-1995-shaped
ocean set. Skies and waters own disjoint halves of the world, so applying one
never disturbs the other.

**Ten infinite grounds, and nine of them answer the scene's lighting** —
the tenth (1.90) is the SNES's **Mode 7** floor, a backdrop drawn from
per-row integer registers: unlit, never faded, and absent from
reflections. The nine: solid,
checker, fractal, **tiles** with grout width, grout glow and per-tile
shading (thin bright grout past glow 1 *is* the synthwave neon floor),
**dunes** with a ridge-strength dial, **snowfield** with sun-glints in
their own colour and sparkle amount, **lava** with pulsing cracks, ember
colour and crack-width dials, a **Material** ground that evaluates any
picked material's node graph to the horizon, and the full **animated
ocean**: a directional wave spectrum fanned off a wind direction, deep
and shallow colours with the path length between them, and the sun's
glitter found in the distribution of wave normals rather than painted
on. All intersected analytically in the background pass, exactly as
POV-Ray and Bryce provided one — and, unlike theirs, lit by the scene:
every lamp's falloff, the hemi wrap, the area-lamp form factor, and
**cast shadows from geometry**, ray-traced or shadow-mapped per lamp,
with a Scene Lighting dial whose zero restores the self-lit flat look
bit for bit. The self-luminous parts — lava heat, neon grout — keep
burning inside shadows.

**Weather** — rain, snow, embers and ash as a screen-space particle
field composited in front of the finished frame, layered for parallax,
resolution-true, and a pure function of (seed, layer, index, time): the
same frame renders identically across runs, devices and supersample
factors, and scrubbing the timeline is stable because nothing
integrates.

**A terrain generator** — Add ▸ Halcyon ▸ Terrain drops a fractal
heightfield in seven landforms (mountain, hills, canyon, dunes with a
wind direction, crater, volcano, plateau), deterministic per seed and
ready-wired to the **Altitude & Slope** node, which grades a material by
height and steepness — with an optional noise socket to roughen the band
edges, and a ColorRamp downstream is the classic snow-line in two nodes.

**Halo materials** — the era's glow sprites, splatted against the
frame's own depth: images, noise breakup, colour ramps over radius,
rays, star spikes, lightning bolts, shockwave rings, halos along curve
objects, and lens flares that sample their lamp's visibility off the
z-buffer. A sky-only scene can flare; a halo behind a wall cannot.

**Animation, told honestly** — sequence renders print per-frame time and
a completion estimate that knows the first frame pays for the caches the
rest reuse (shadow maps, BVHs, compiled shaders), so the ETA converges
on the truth instead of the warm-up cost.

**The image-as-palette road** — point the quantiser at any image and the
whole render is forced through that image's colours; an image-editor
operator turns any picture into a one-pixel-per-colour palette table to
feed it. Both dither roads clamp before they quantise and before they
diffuse, so an unrepresentable hue becomes the nearest palette colour
instead of a smear.

**112 render presets** across eight categories — 3D software (Infini-D, Ray
Dream, 3D Studio, MAX R2 and 3ds Max 2012, trueSpace, LightWave, POV-Ray,
Bryce, Softimage|3D, Maya 4, Blender 2.41 and the rest), home computers
(VGA Mode 13h through PC-98 and X68000, the Amiga's Extra Half-Brite, MSX1
and MSX2+, Spectrum 512, Elite on the BBC Micro), consoles and 3D cards
(PlayStation, Saturn, N64, Voodoo and Voodoo2, Dreamcast, Nintendo DS, Mega
Drive, PowerVR PCX2, Direct3D 5 retail…), arcade boards (Sega Model 1, 2
and 3, Namco System 21, the Atari vector monitors), broadcast (Video
Toaster, PAL, VHS, the RF modulator, Video CD), handhelds, the early web
(GIF, JPEG, CD-ROM FMV, Smacker), and fifteen Cel & Film presets from worn
silent nitrate to the modern digital master, Tron's backlit mattes among
them. Applying one resets everything first, so presets
never accumulate.

**Render passes** — Depth, Normal, Position, UV, Object Index and Material
Index, written under Blender's own names and channel layouts so a Halcyon Z
pass drops into a comp built for Cycles without rewiring.

**396 settings, all exposed, all proven, all explained.** Two tests stand
behind that sentence: one holds every setting to a proof that it changes what
it claims to change (a matrix row, an A/B render, a behavioural check, or a
declared reason — nothing silently exempt), and one fails the build the moment
any property ships without a detailed tooltip.

**Legacy import: any 2.79-or-earlier .blend, materials included** — the
headline feature, described in full above: File ▸ Import ▸ Legacy Scene, the
automatic fix-up on plain File ▸ Append, Blender Internal materials and all
eighteen texture slots rebuilt one to one, lamps with their real energies,
and the old world as sky, ambient and fog.

---

## Design notes

### The pipeline

```
supersample → rasterise z-buffer → reconstruct fragment attributes
→ evaluate node graph per material → collapse closure to a reflectance model
→ light → ray-traced reflection/refraction → A-buffer transparency → fog
→ cartoon outline ink                  [from the G-buffer, both devices]
→ filtered downsample
→ glow / star / flare (linear light)
→ display transform
→ colour depth + dither + palette      [the framebuffer]
→ composite NTSC encode/decode         [the cable]
→ interlace                            [the signal]
→ CRT mask, scanlines, curvature       [the glass]
→ JPEG artefacts                       [the file]
→ pixel aspect and nearest upscale
```

Glow happens before quantisation and scanlines after it. That is not a
stylistic preference — a 1996 machine glowed in its framebuffer and scanned on
its tube, and doing it in the other order looks wrong in a way that is hard to
name but easy to see.

### Determinism

The same frame renders the same pixels — across runs, across thread counts,
and across devices. That is a doctrine with machinery behind it, not a hope:

- Every stochastic effect (soft shadows, AO, dither jitter) draws from an
  integer hash that is a pure function of (pixel, sample, stream, seed), and
  angles come from a shared 256-entry table rather than anyone's `sin`.
- Ties are **named rules, not races**. The rasteriser resolves coverage ties
  to the lowest triangle id; the ray tracer resolves equal-distance hits the
  same way — the answer is a function of the candidate set, never of
  traversal order or scheduling.
- Where a driver's last-bit arithmetic genuinely cannot decide reproducibly —
  a reflection ray grazing two coincident surfaces — the GPU **routes the tie
  to the CPU** and returns the reference's own answer by construction.
  Route, never guess.
- Every internal data texture on the GPU is read by `texelFetch` with integer
  coordinates, never through a sampler whose filter state the Python API
  cannot control. That one is written in scar tissue: a filtered read of the
  triangle-id buffer once drew a faint wireframe over every edge of a scene,
  visible only at resolutions the test suites never rendered.

### Closures to reflectance models

Blender's node graphs produce Cycles-style closures: additive, weighted lobes.
A 1990s renderer has one shader with a diffuse term and a specular term. The
translation sums the weighted lobes into those two slots and infers the model
from what the tree contains. It's in `core/render.py:closure_to_surface` and
documented there. It is an honest lossy mapping, not a pretence that the two
systems are the same.

Roughness maps to a Phong/Blinn exponent by the classic `2/r⁴ − 2` relation,
clamped.

### Transparency

A true A-buffer (Carpenter 1984). Every transparent fragment is shaded and
kept; fragments are then sorted per pixel and composited by layer rank. It is
correct through any depth of overlapping surfaces, unlike the per-object
sorting most period renderers used — which is also available, as `Painter's
Algorithm`, because its failures are part of the look. Screen Door
transparency punches dither-pattern holes instead: no sorting, no blending,
pure period.

**Alpha Clip (punch-through)** is the material panel's other alpha mode, and
the era's cheap one: the alpha chain is tested against a threshold and the
surface is either fully there or fully absent. Its pixels resolve in the
z-buffer *before* shading and shade once — no layers, no sorting, no
per-layer GPU passes — which is exactly why fur, foliage, fences and cut-out
sprites cost almost nothing on 1990s hardware. Fur Shells materials use it
automatically; the suite holds the clip road pixel-identical to the
A-buffer's answer on hard-alpha scenes.

### Shadows

Shadow maps are compared in **linear light-space distance**, not NDC z, so the
bias is a world-space number that means something. Normal-offset biasing —
stepping off the surface by a texel or so, scaled by how obliquely the light
hits — removes acne without the detached shadows a large depth bias causes.
Shadow maps and shadow rays agree to within 0.0015 mean difference on the test
scene, which is the real evidence that both are right. Point lights get proper
six-face cube maps, and the same depth images travel to the GPU as atlases.

### Rays

The BVH builds by **binned surface-area heuristic** — a median split over a
scene with a huge ground plane produces sibling boxes that overlap almost
entirely, and the profiler measured shadow rays paying for both subtrees all
the way down. The SAH tree made the field scene's shadow rays 12× faster and
its reflection rays 6×, on both devices at once, because the GPU kernels
traverse the same packed tree. Traversal on the CPU runs level-synchronous
waves of (node, ray) pairs — a few dozen large array operations per query
instead of thousands of small ones.

### The bpy boundary

Everything under `core/`, `shaders/` and the numerical half of `gpu/` imports
NumPy and nothing else. No bpy. The exporter flattens node trees into plain
dicts, bakes colour ramps and curve mappings into 256-entry LUTs, and hands
over dataclasses. That boundary is why the whole renderer can be tested
headlessly, and it is the reason the test suite below exists at all.

`properties.py` is **generated from the `RenderSettings` dataclass**, so the
UI cannot drift from the renderer. A test asserts every field has a matching
property; another asserts every property has a reader inside the engine, which
is how seven corpse settings were found and removed.

---

## Running the tests without Blender

```bash
python3 -m halcyon.tests.run_all              # shader + renderer tests
python3 -m halcyon.tests.run_all --images out # ...and write demo PNGs
```

The suite covers the shader compiler (divergent control flow, loops with
per-lane trip counts, out-parameters, early return, structs, matrices, swizzle
assignment, discard, the preprocessor, both dialects, error reporting) and the
renderer (geometry landing where independently projected, shadows by both
methods agreeing, every shading model distinct, all debug passes, affine
texture warp, vertex snapping, A-buffer transparency, ray-traced reflection to
any depth, node-graph evaluation including group recursion and unknown-node
fallback, palette colour counts, the full post chain, the generated period
objects, all 114 templates rendering, all 303 skies applying and differing,
all 112 presets, every setting's proof, and every tooltip's existence).
Fifteen further modules (`tests/test_r251_*.py`, 1.90) hold the period
features: each pins its pack's defaults bitwise against the 1.89.0 zip,
then its laws, its GLSL twins and its refusals by name.

The GPU pipeline is tested headlessly too: the same GLSL the driver compiles
is executed by the compiler's own NumPy backend against the same packed
textures — raster kernel, deferred shading, ray kernels, the ink, the sky,
the resident frame's resolve and the post chain, each a twin held to the CPU
function it replaces — so a change that would move a GPU pixel fails the
suite on a machine with no GPU at all. The driver's plumbing is covered too:
a fake device (`tests/fakedevice.py`) applies the driver's own rules — every
uniform in the CreateInfo spec, every sampler bound, ints as ints, clear,
blend, discard, scissor, a target pool that hands the same object back, no
draw sampling its own or a freed target — so `render()` and `post.process`
run the whole GPU road headless (shading, sky, ink, resolve, resident post)
and are held within 6e-6 of the CPU render (the shading twin's rounding:
one 8-bit level at 1–3 pixels of 76,800 after the depth), the resident post
chain bitwise the CPU chain over the same frame. The final word still belongs to
hardware: **Run Self Test** renders the
299-row feature matrix on your actual driver and reports any row where the
two devices disagree.

Set `HALCYON_DEBUG=1` to make node evaluation raise instead of falling back
silently — useful when a material isn't doing what you expect.

---

## The GLSL / HLSL subset

Verified working:

| | |
|---|---|
| Preprocessor | `#define` (object and function-like), `#ifdef`, `#ifndef`, `#if`, `#elif`, `#else`, `#endif`, `#undef` |
| Types | `float` `int` `uint` `bool`, `vec2/3/4`, `ivec`, `bvec`, `mat2/3/4`, `sampler2D`, user `struct` |
| HLSL aliases | `float2/3/4`, `float4x4`, `cbuffer`, semantics (`SV_TARGET`, `TEXCOORD0`, …) |
| Control flow | `if`/`else`, `for`, `while`, `do`, `break`, `continue`, `return`, `discard`, `switch` |
| Functions | user functions, `in`/`out`/`inout` parameters, overloading by arity |
| Arrays | declaration, indexing, assignment, **per-lane divergent dynamic indices** |
| Swizzles | read and write, `xyzw` / `rgba` / `stpq` |
| Derivatives | `dFdx`, `dFdy`, `fwidth` — real screen-space differences, not stubs |
| Textures | `texture`, `texelFetch`, `textureSize`, `textureLod`, `textureGrad` |
| Extras | `noise3`, `fbm`, `quantize`, `posterize`, `dither4x4`, `hsv2rgb` |

Divergent control flow is handled with execution masks, so different pixels
genuinely take different branches and run loops for different numbers of
iterations.

**Not supported:**

- **Recursion.** Rejected at compile time with the call cycle named. Under SIMT
  masking every lane walks both sides of a branch, so a recursive call never
  reaches its base case.
- **Array constructor syntax** — `float[3](1.0, 2.0, 3.0)`. Declare and assign
  instead.
- Geometry, tessellation and compute *user* stages. User shaders are
  fragment-language only (the engine's own kernels use compute internally).
- `sampler3D`, `samplerCube`, integer textures.
- Uniform block layout rules — `cbuffer` members are flattened to plain
  uniforms.

---

## GPU support

**The port is complete: rasterisation, shading and post all run on the GPU**,
through Blender's own `gpu` module — the layer EEVEE is built on, and the only
route an add-on has. Set **Device: GPU** in Render Properties and the three
stage toggles come on together.

- **Rasterisation** runs as a compute kernel built to produce the same
  G-buffer the CPU produces — triangle ids, perspective-correct barycentrics,
  depth at the chosen precision, the affine-warp interpolants, snap and
  16-bit modes included. On the release hardware it does so at 96×72 (the
  self test's table) and at 1280×720 except for a few edge pixels per frame
  (see *Known limitations*). Coverage at shared edges is governed by the
  same watertight window
  and canonical clip rules on all four engines (CPU loop, CPU batch, GLSL
  kernel, NumPy replay), because a half-ulp disagreement near the near plane
  once opened pixel-wide cracks. Since 1.90.0 **Painter's algorithm rides
  the kernel** (the polygon's one depth and a named tie rule made it
  order-free; a banded frame and the overdraw census rasterise on the CPU
  and print why), and the period depth encodings compare on their own keys
  there: the winner at a marked pixel is the CPU's replay, a frame that
  marks more than a tenth of its pixels rasterises on the CPU whole with
  the count printed, the W-buffers refuse an orthographic camera by name.
  Pixel centres, near rejection, the ordering table and the vertex formats
  act before either fill, on shared data.
- **Deferred shading** packs the G-buffer into textures and shades one
  full-screen pass per material, every frame constant baked into the shader
  source. Lights, shadow atlases (cube faces included, every Vogel PCF tap
  reproduced), image textures with the CPU's own filter arithmetic, vertex
  colours, per-pixel surface parameters, the master shader's whole colour
  chain, coded GLSL nodes inlined natively, and the ray sweeps — reflections
  and refractions to any depth, traced by compute kernels against the same
  SAH tree, with each level's secondary passes scissored to the pixels its
  rays actually hit.
- **Fog** (1.90.0) runs inside the material pass — the four curves, the
  per-vertex rounding, the cel bands, the height layer, and the period
  tables and cues (the GTE's, the Voodoo's, the PowerVR2's, the DS's,
  Direct3D's z-fog; and the GameCube range adjust, per-polygon fog, the
  material fog dials, backdrop fog, POV-Ray's ground fog and turbulence) —
  the CPU's own arithmetic in the simulator, and the frame stays on the GPU
  under fog, where 1.89.0 read it back and paid about 550 ms of CPU fog on
  a 720p frame. **What still takes the CPU's fog over the readback, by
  name:** a material with traced reflections, refractions or a
  CPU-evaluated environment term (their composites land after the readback;
  the console prints `fog on the CPU for '<material>'`). Vertex-rate
  materials keep their fog in the CPU-lit corners, so a table or a dither
  that needs a pixel (the Voodoo2's fog dither, the GameCube's column
  adjust) does nothing at that rate. Reflection hits fog with the scene's
  curve and ignore a material's fog dials. A fog socket driven per pixel
  refuses by name. Headless, the fog stage over the same simulated shading
  is bitwise for every quantised fog mode and 2.4e-7 on the smooth curves
  (the G-buffer packs its third barycentric as 1 − x − y, one ULP off the
  rasteriser's own on most pixels — still open); the whole fogged frame is
  2.0e-6 to 5.0e-6 from the CPU frame on the seven basic fog rows,
  quantised ones included (8.3e-7 to 1.1e-5 across the fog rows the suite
  checks), against the deferred pass's pinned 1e-3 (the suite log of
  2026-10-04, `docs-dev/suite_1.90.0_fix2.log:12823-12878`). On a driver
  the exponential curves are driver-rounded and one 8-bit level may move
  at a step edge.
- **The period light units and combiners** (1.90.0): GX and the DS unit
  shade in GLSL; the thirteen combiners interpolate CPU-lit, quantised
  corners and apply the machine's rule in the pass. **Refusing by name,
  shading on the CPU:** Sega Model 2 and Model 3 (no GLSL — the CPU's
  corner road on both devices; a pixel-rate request refuses), a DS toon
  table with a linked Toon Size, Super FX plot without a fixed palette,
  the REYES shading rate under affine texturing, for a Bump-node material
  and on hit and layer passes, a per-pixel chain into Brilliance, into the
  DS unit's Glossiness or into SR Bump's Light, and an Env-hole
  material (the whole frame, until the sky pass can draw through holes).
- **Shadows and transparency** (1.90.0): the midpoint map is the same atlas
  and compare as the classic one. Planar shadows, Dreamcast modifier
  volumes and DS shadow polygons are a mask applied over the read-back
  frame on both devices, and the framebuffer formats (PS2, GameCube,
  Voodoo) a pack over it — the frame leaves the GPU for them by name.
  Every blend equation that reads the frame beneath (the PlayStation,
  Saturn, 3DO, SNES, GBA and DS items, Thin Wall, the Imagine fog
  object) refuses the GPU's layer passes by name and composites on the
  CPU; Doom's fuzz and the N64's noise compare keep the layers (the
  noise compare reads the CPU's own map, uploaded per frame); the PS2's
  two-pass alpha keeps a constant alpha's layer and refuses a per-pixel
  one. The Elite and vector-beam wires are drawn on the CPU over the wire
  road's readback.
- **The sky's new modes** (1.90.0): the cylinder sky, the gradient backdrop
  and the Mode 7 floor draw in the sky pass from the CPU's own integer
  tables; a reflective material under a Mode 7 floor still shades on the
  CPU. Y-shear is one matrix both devices read. Sliced panoramas, the
  parallax stereo, the lens passes and the time-slice blur are whole
  renders combined on the CPU — the same bytes on either device by
  construction, the resident frame read back per pass.
- **The ink** (1.88.0) draws its line as fragment passes on the same
  G-buffer: the seed compares, the mask roads as one diamond-search pass,
  the style road's chamfer as an iterated relaxation exact within the band,
  the style maths dial for dial, boil and pencil, the composite — bitwise the
  CPU's line on the field's 720p frame, at the cost of a few passes and a
  readback where it cost half a second of NumPy; since 1.89.0 it reads the
  frame the shading left on the GPU in place, no upload. The stroke road,
  the isophote weight, the Surface anchor, Form / Shadow / Tone lines and
  the vertex-colour line control stay on the CPU and say so by name.
- **The sky** (1.89.0) is the last draw of the shading's own burst: covered
  pixels discard, so the material texels stand bit for bit and every
  uncovered texel is the world's colour — the solid sky, the node world's
  flat colour, sky blend and environment image, a plain Background node,
  the gradient, the bands and HDRI, the supersampled fast-background road
  reproduced exactly — and the readback is the whole frame. The Bryce,
  painted, starfield and physical skies, richer world graphs and the ground
  plane stay on the CPU and say so by name.
- **The frame stays on the GPU** (1.89.0). The shading's target is kept
  when nothing on the CPU needs to edit it; the ink reads it in place; the
  supersample resolve draws it at output size (the readback a quarter of
  the bytes at 2×, the taps summed in the CPU filter's own order); the post
  chain inherits it. Any CPU stage that edits the frame releases it by name,
  and the console says which.
- **Post** runs as a resident chain (1.89.0): texture in, texture out, no
  upload when the render's frame is inherited (otherwise one, plus one more
  after each active CPU-only stage that hands the frame back), the stages
  drawn in ping-pong targets and read back once at the end. The film grain
  (to 1.2 px) with the flicker, the display transform, the bit depth without
  a dither (on the CPU's own level table, bitwise), NTSC and CRT draw on the
  GPU, each measured against its CPU function before it was allowed to
  default on. **Since 1.90.0 the ordered dither and the palette snap draw
  there too**: the ordered dither to a bit depth on the CPU's own Bayer
  matrix and level table, the palette snap as the CPU's own inverse
  colormap read as a texture (a fixed palette costs nothing; an adaptive
  one under Lock Palette costs one named readback on its first frame, then
  none; headless, a fixed-palette or ordered-dither chain records no
  readback and no upload). **The new resident stages:** Extra Half-Brite, CRY, YJK, the
  N64 VI's dither filter and gamma, the GameCube copy filter, the PS2 CRTC
  blend (the previous-frame mode reads its output back by name), the two
  3dfx scan-out filters; and attribute cells, Super Black, the video
  legaliser (within 1e-5: a square root and a division), the N64 coverage
  blend and divot, chroma siting, S-Video and RF, the PAL receiver, the
  tape, MPEG-1 (five draws), Smacker and the matte glow. **What refuses by
  name and runs on the CPU:** every error-diffusion dither (Floyd-Steinberg
  and its relatives are sequential by definition) — and the display
  transform ahead of one (that routing is active on the driver, and the two
  devices still diffuse different pictures: see Known limitations) —
  the Noise and Blue Noise dithers, an adaptive palette with Lock Palette
  off, a Custom palette without an image, per-scanline palettes, C64
  multicolour cells with Lock Palette off, NTSC dot crawl (a composite
  frame using it falls back whole), halftone, interlace, lens, JPEG, the
  optical stages, the film beyond grain, the 3DO's 2× and the GBA's Mode 5
  stretch (after the final readback), and by size: a matte-glow diffusion
  past 160 taps, the NTSC S-Video cable on frames wider than 1,685
  pixels, a tape low-pass wider than 96 pixels. An active CPU-only stage
  reads the frame back by name and the chain carries on.

**What a driver has run, and what it has not.** Both waves of 1.90's
features ran on the release hardware (RTX 5060 Ti, Vulkan, Blender 5.2.0)
on the final tree, 2026-10-05: 206 variants, four error-diffusion controls
and two frames of the field's own export at 1280×720, CPU device against
GPU device with the post chain included, in seventeen fresh Blender
processes, none of which died. In all 212 rows the GPU device's first frame
rasterised on the GPU; the earlier run (2026-10-04) is superseded, because
the G-buffer cache served its GPU frames the CPU's raster in 188 of its 208
rows. Of the 212 rows, 43 are bitwise identical; 34 differ by one 8-bit
level and 13 by at most two, at 1 to 28 pixels of 921,600; in 56 the
largest difference is one shadow-map compare at pixel (76, 688) of the demo
scene, which has differed between the devices since an earlier release and
is parked, and nothing else is above 1e-2; in 60 a pixel other than that
one is above 1e-2 (1 to 8 pixels above 1e-2 in 58 of them, 38 and 42 in the
other two; 1 to 42 above 1e-3, 120 in one row); six diverge. Outside those
six no row differs at more than 28 pixels, except five: two under an HDRI
world (446 and 440, of which 431 are one 8-bit level in the sky pass alone
— read as 1.89.0's HDRI class), the VHS preset (128), the S-Video preset
(42) and the field's own frame at 2× supersampling (39). The field's own
frame: 1 pixel by one 8-bit level at one sample (CPU 1,652 ms; GPU 105 and
100 ms warm), and 39 pixels (38 above 1e-2), max 9.80e-1, at its own 2×
supersampling (6,036 ms; 306 and 302). Those 38, and the pixels above 1e-2
that other rows gained in this run, are read as edges the two rasterisers
decide differently (Known limitations).

Three of the six are presets with a Floyd-Steinberg dither, where the
two devices draw different dither patterns: Atari ST 351,922 pixels above
1e-2 (max 1.00), Amiga OCS 67,372 (max 0.533), Studio R4 354,017 (max
0.484). On the CPU's raster the earlier run counted 289,708, 364,620 and
312,123: the count moves with the frame beneath the dither. In all three
the report prints the display transform on the CPU by name and GPU stages
`[]` (and no post-chain-alone line), so that routing is active on the
driver and did not make the frames agree. The controls rule out the palette
and the summed-area footprint: Atari ST with the dither off 0 pixels and
with Bayer 4×4 0 pixels, Amiga OCS with the dither off 1 pixel (2.00e-1;
its post chain alone 0), Studio R4 with the dither off and Summed Area ×4
kept 4 pixels (max 3.23e-2). The reading: error diffusion carries the small
float difference between the GPU's shading and the CPU's into a different,
equally dithered picture — for any error-diffusion dither, since before
1.90, never measured until this round; the console says so since fix pass
4, and what to do about it is open (Known limitations).

The other three are signal and codec stages that drew on the GPU. The RF
modulator over composite (NTSC, CABLE_RF): 23,542 pixels above 1e-3, 795
above 1e-2, max 3.53e-2; its post chain alone 23,507 and 787, max
1.96e-2, over the 4/255 its variant states. MPEG-1 intra blocks (MPEG1):
4,774 and 63, max 1.41e-1 at the parked pixel; chain alone 4,712 and 3,
max 1.18e-2, over its stated 1/255. The Video CD preset (MPEG1, NTSC,
CRT): 921,589 pixels above 1e-6, 9,266 above 1e-3, none above 1e-2, max
8.32e-3, over its stated 1/255. The second wave's other post stages show
in their rows' post lines as GPU stages — CELLS, SUPERBLACK, LEGALISE,
N64VI_AA, N64VI_DIVOT, CHROMA_SITE, CABLE_CHROMA, PAL_DECODE, TAPE,
SMACKER, MATTE_GLOW — and those rows sit in the classes above. Handed to
the CPU by name in second-wave rows: interlace (four preset rows), the
Spectrum 512 per-scanline palette, NTSC under dot crawl (the VHS preset),
the display ahead of Floyd-Steinberg (Studio R4), and the whole shading
of the two RenderMan-preset rows (the console names the REYES shading
rate on a hit or layer pass): those two take the CPU's time — 6,024 ms
against 5,696 and 5,760 at rate 16, 28,210 against 26,831 and 26,979 for
the preset — and differ at 1 and 4 pixels, all above 1e-2 (max 2.26e-1
and 1.67e-1), where the earlier run on the CPU's raster had 0. Of the 88
second-wave rows the frame reaches the post chain resident in 72, leaves
the GPU by name at the framebuffer format in ten (the PS2, GameCube,
Voodoo and Voodoo2 rows), and in six the report prints `frame kept
through nothing` (Sega Model 2, Mega Drive, the two N64 noise-compare
rows, the two RenderMan rows).

The 299-row matrix ran at 96×72 in every one of those processes with one
result: 293 rows drawn on the driver and matched, 4 partially routed to
the CPU by name and matched, 2 FAILED. The earlier run printed 4 FAILED;
the matrix clears the G-buffer cache itself, so it used the GPU
rasteriser in that run too. Two of those four were a defect and
are fixed: under the GPU rasteriser (on by default) N64 coverage
anti-aliasing lost its coverage plane, so the N64 filter was skipped on
the GPU device (5,512 and 5,452 of 6,912 pixels wrong); in this run
those rows read 0.000000. The two that remain: error diffusion FLOYD
(724 pixels above 1e-2, max 0.032258 against a bar of 0.006 — the
finding above) and Summed Area (1 pixel, 0.278431 against 0.05; its
footprint ×4 row matches). Not covered
at 1280×720: the
six new nodes, the PS2's two-pass alpha, Blender 2.4x's Z Offset, Invert
Z and Env, Thin Wall Refraction and Imagine's fog object have matrix rows
only (matched); no row triggers the refusals that hang on a setting or a
size (a DS toon table with a linked Toon Size, Super FX plot without a
fixed palette, the REYES rate under affine texturing or a Bump node, C64
multicolour cells with Lock Palette off, the size limits of the matte
glow, the S-Video cable and the tape); and the Model 3 fog-lobe row has,
by its own label, no screen spot in its scene. A row's agreement shows
that the devices agree, not that the labelled feature changed the
picture.

The whole matrix — 299 feature rows — is compared against the CPU picture by
**Run Self Test** on your own driver. A row that cannot yet run on the GPU is
*routed*: the frame says why on the console and shades that part on the CPU.
That is the rule, and on the release hardware two rows still break it: under
an error-diffusion dither the routed frame differs from the CPU's (the
finding above), and Summed Area differs at one pixel. The table prints them
as FAILED; it does not hide them. The same
honesty applies at runtime — if the GPU path cannot deliver a frame (a driver
hiccup, a timeout under a heavy foreign add-on), the frame renders on the CPU
with the reason printed, never half-drawn.

Frame-to-frame, unchanged uploads — shadow atlases, mesh attributes, texture
pixels, the BVH — are cached behind content fingerprints, so an animation
re-uploads what moved and nothing else. The console prints a per-stage split
(raster clip/pack/dispatch/read, shade plan/upload/draw/reflect/composite, and
inside reflect: trace, secondary draws, sky-along-misses, levels run and
skipped), so when a frame is slow, the stage that owns the time names itself.

---

## Known limitations

These are real, and I would rather write them down than have you find them.

**The Blender layer is only partly validated here.** Two levels of stand-in
exist: one imports and registers every module against a `bpy` stub, and one
runs a whole frame through `HalcyonRenderEngine.render()` — property group,
exporter, renderer, post chain, delivery and passes — against a fake
depsgraph. Neither can catch a segfault, a driver quirk or an RNA lifetime
bug. If something misbehaves, the traceback in Window ▸ Toggle System Console
is the fastest way to tell me what happened.

**The 1.90 period features carry open items, and here they are.**

- **DS Highlight whitens lit surfaces.** At the default toon table (Toon
  Size 0.5, Toon Steps 2) the Highlight table (Nintendo DS) model turns
  every lit surface white — 61% of the test frame. A narrower band or more
  steps do not help; half-energy lamps with eight steps give a banded,
  coloured picture. The hardware rule is
  itself undecided between two sources: GBATEK (as built: the texel
  modulated by the table entry, the entry added again) and melonDS (the
  texel modulated by the vertex red as grey, the entry added). The
  arithmetic is unchanged until that is settled; the model's tooltip says
  that the default table saturates and what it takes to keep the texel.
- **The chroma key: what it does and where it does not reach.** Under a
  texel format with no alpha plane (the Voodoo presets' RGB565, also
  RGB332, RGB5550, NCC, I4) a texture's cut-outs are stored as the key
  colour and key as on the card. So under the two Voodoo presets a
  texture with alpha below one half shows those texels black unless the
  image's Alpha is wired — also a texture that only carries a mask
  there. Not reached, in no preset's combination: the formats whose
  alpha is not the source's (I8 and the N64's I4 read it from intensity,
  Model 2's I4 from its white marker); block compression after an
  alpha-less format moves neighbouring texels and VQ can lose holes; NCC
  fits its table before the key.
- **An error-diffusion dither gives each device its own pattern.** With
  a Floyd-Steinberg dither the GPU device and the CPU device render
  different dither patterns; the same presets without it agree at 0 to 4
  pixels. Measured on the driver at 1280×720: Atari ST 351,922 of 921,600
  pixels above 1e-2 (max 1.00), Amiga OCS 67,372 (max 0.533), Studio R4
  354,017 (max 0.484); the count moves with the frame beneath the dither
  (289,708, 364,620 and 312,123 in the earlier run, on the CPU's raster).
  Controls: Atari ST with the dither off 0 pixels, with Bayer 4×4 0
  pixels; Amiga OCS with the dither off 1 pixel (2.00e-1: the next
  bullet); Studio R4 with the dither off and Summed Area ×4 kept 4
  pixels (max 3.23e-2). So neither the palette nor the
  summed-area filter is the cause. The display transform already ran on
  the CPU by name in the three diverged frames: that routing is active on
  the driver and does not cure it. The reading: the GPU's shading differs
  from the CPU's by small float amounts (0.000023 and 0.000052 at most on
  the two frames of the self test's shading check), and a sequential
  dither carries them across the frame into a different, equally dithered
  picture — so it holds for any error-diffusion dither, predates 1.90 and
  was never measured before. It breaks the rule of identical pixels
  across devices, and the console now says so. Undecided: render such
  frames on the CPU by name, or keep the GPU's speed (Atari ST: 4,583 ms
  on the CPU device, 1,792 and 1,782 ms warm on the GPU device) and print
  the difference. Until then, render a diffused frame on the CPU device
  when it has to match.
- **A few pixels differ between the devices, read as rasteriser edges.**
  With the GPU rasteriser on (the default on the GPU device) a handful of
  pixels of a 1280×720 frame come out different from the CPU device's, by
  large amounts at isolated places, and more of them under supersampling
  on a detailed model. Measured on the driver: ten rows that were bitwise
  identical on the CPU's raster in the earlier run now differ, nine at 1
  to 6 pixels and the VHS preset at 128; in the Direct3D 5 retail row it
  is one pixel, 1.90e-1 at (414, 573), with the sky alone and the post
  chain alone at 0; the field's own frame, 1 pixel by one 8-bit level at
  one sample in both runs, went from 3 pixels to 39, 38 above 1e-2, max
  9.80e-1, at 2× supersampling. The same places recur across rows:
  (37, 789) holds the largest difference in eight. The reading: the GPU
  rasteriser decides a few triangle edges differently from the CPU's. It
  predates 1.90, and no earlier driver run measured it at this size
  because the G-buffer cache served the CPU's raster to the GPU frames.
  It breaks the rule of identical pixels across devices, and the cause
  is not diagnosed. The CPU device is the reference: render there when a
  frame has to match.
- **What the driver run leaves open.** Both waves have now run on the RTX
  5060 Ti at 1280×720 (*What a driver has run*, under GPU support). Still
  without a row at that size: the six new nodes, the PS2's two-pass
  alpha, Blender 2.4x's Z Offset, Invert Z and Env, Thin Wall Refraction
  and Imagine's fog object — on the driver they ran only in the 96×72
  matrix, and matched. Two of the 299 matrix rows FAIL on that hardware:
  error diffusion FLOYD (724 pixels above 1e-2) and Summed Area (1 pixel,
  0.278431). Two more failed in the first run and were a defect, fixed
  and rerun at 0 pixels: N64 coverage anti-aliasing was skipped on the GPU
  device under the GPU rasteriser. Three signal rows differ past the
  bar their test states: the RF modulator (23,542 pixels above 1e-3, 795
  above 1e-2), MPEG-1 intra blocks (4,774 and 63) and the Video CD preset
  (9,266 and none). **Run Self Test** renders all 299 matrix rows on your
  driver and reports any that disagree.
- **The texture cache is keyed by an array's memory address** (it has been
  since before this round). Headless, one process rendering many scenes
  once read a stale prepared texture after an address was reused. Whether
  that can happen inside Blender is not known. Textured variants of the
  driver test may therefore have tested another texture than their label
  says — on both devices alike, so their agreement stands.
- **Edited presets are compared with 1.89.0 by a sheet, not by the
  suite.** 39 of the 90 shipped presets have edited settings, 22 presets
  are new, 51 are unchanged (`docs-dev/r251_presets.log`, line 1). The
  suite's identity pin covers only the 51; nothing in the suite compares
  an edited preset with its 1.89.0 self. `tools/r251_preset_sheet.py`
  renders that comparison on the demo scene at 320×240 on the CPU road: a
  look, not a test, and not a driver measurement. On that scene six edited
  presets move no pixel — 3ds Max R2 and 2012, Studio R4, POV-Ray 2 and
  3.1, Saturn — because their new settings need what the scene lacks:
  motion blur or depth of field switched on, a lamp whose Decay is left at
  Default, a material with no model of its own.
- **A preset's Default Model reaches only materials that name no model.**
  `material_model` (`core/render.py:6337`) gives Force Model, then the
  material's own model, then a Halcyon node's model, precedence over the
  preset's Default Model. Seven presets changed their Default Model in
  1.90 (Dreamcast, GameCube, N64, PS2, PSX, PSX Hi-Res, Saturn). Measured:
  Saturn with Default Model Flat against Saturn Add moves 0 pixels on the
  demo scene and on the feature matrix's textured scene, whose materials
  all name a model; with the materials' models removed it moves 12,136 of
  76,800 (Dreamcast 31,456, GameCube 70,272, N64 76,798, PS2 73,130, PSX
  and PSX Hi-Res 47,280). How Blender materials reach this rule in a real
  scene was not read. Force Model applies the board's model to everything.
- **Two presets now change what you get, not only how it looks.** Doom
  sets Y-shear, so choosing it changes the projection of a pitched camera
  (31,022 of 76,800 pixels against 1.89.0 on the demo scene). Virtual Boy
  renders a side-by-side stereo pair with parallax layers (48,393 pixels
  against 1.89.0) — and renders grey, as it did in 1.89.0, not red.
- **Colour depths that were no-ops in 1.89.0 quantise now.** SNES, 32X
  and Neo Geo ('5'), Turbo Silver ('6') and ZX Spectrum ('3') did nothing
  in 1.89.0. Distinct colours on the demo frame, before → after: SNES
  92 → 16, 32X 6,607 → 5,114, Neo Geo 1,935 → 102, Turbo Silver 614 → 54,
  ZX Spectrum 201 → 5.
- **Register-depth palettes lose entries.** The palette is built at full
  precision and then snapped to the machine's register depth
  (`core/post.py:323-325`), so entries collide: Amiga OCS shows 24
  distinct colours of 32, Atari ST 11 of 16. A design point, undecided.
- **Three preset keys of the plan left out on purpose.** GameCube carries
  no fog range adjust: at the preset's per-vertex rate it does nothing
  (pinned bitwise), and GXInit leaves it off. Direct3D 5 retail does not
  set Fog Depth to Z: with it, switching Fog on at ordinary Start and
  End fogs nothing (set the dial yourself and give Start/End as 0..1
  depths; the console names the frame's span on a final render, not in
  the viewport). PowerVR PCX2 does not set a 5:5:5 texel format: one
  global alpha-less format turns every cut-out texture opaque, where the
  card kept 4:4:4:4 for those.
- **CPU cost of the palette and cell roads, measured small only.** Three
  consecutive frames at 320×240 on the CPU road, on a busy machine. A
  first-frame cost that did not repeat on the next two frames (whether it
  returns when a cache is evicted or a scene changes was not measured): CRY (Jaguar) about 0.67 s; VGA256 with Bayer
  4×4 about 0.5 s; the Dreamcast VQ rows about 0.9 s in the render; SNES,
  Neo Geo and 32X about 0.3 s in post. Every frame: C64 multicolour cells
  about 570 ms, MSX1 cells about 190, MSX2 about 127, Turbo Silver about
  124, ZX Spectrum about 95, VGA256 with Bayer 4×4 about 30. Not measured:
  larger frames, the GPU device.
- **The adaptive palette lock is keyed without the picture** — by palette
  size, method and seed only (`core/post.py:317`), on purpose, so an
  animation's colours do not crawl; it has been since before this round.
  `clear_caches()` in `core/palette.py` clears it; the renderer's own
  `clear_caches()` (`core/render.py:257-259`) does not. Headless, one
  process rendering many scenes gave a later frame an earlier scene's
  palette. Whether a second scene rendered in one Blender session inherits
  the first scene's palette was not read.
- **Blender died during the driver test**, silently and twice, after a few
  dozen 720p variants in one process; each variant passes alone, and the
  test now runs in fresh processes of thirteen: in the full run all
  seventeen (the last of four variants) finished. The cause is not found.
- **N64 coverage AA makes the whole frame 15-bit.** The stage reads the
  frame at five bits per channel whatever Colour Depth says, so smooth
  shading bands across faces, not only at edges. It also softens interior
  polygon seams (the RDP's coverage accumulate is not modelled) and needs
  one sample per pixel.
- **Table fog at the vertex rate is per corner.** The VOODOO, VOODOO2 and
  DREAMCAST presets shade at the vertex rate, so their fog tables are read
  at the lit corners and the Voodoo2's fog dither adds nothing; the
  GameCube fog range adjust is inert at that rate too. The hardware fogged
  per pixel.
- **Worker processes and the viewport.** With Use Worker Processes, Super
  Black has no coverage plane (as depth of field has none); a frame with
  fuzz, Thin Wall, or the matte glow skips the pool by name. Lens-pass
  depth of field and the matte glow are F12 roads: the viewport shows
  neither. Planar and volume shadows bake on every viewport draft.
- **Backdrops are not environments.** Mirrors and ray misses do not see the
  cylinder sky's image or the Mode 7 floor; the Mode 7 floor is unlit.
- **Stand-ins, named.** Where a machine's data was game-authored or never
  published, Halcyon's substitute is its own and is not a measurement: the
  DS toon and shininess tables, the Sega boards' luma ramps, the Jaguar's
  CRY chroma square (unmeasured against the manual's tables), LightWave's
  squeeze curve (uncalibrated against a real render), the 3dfx scan-out
  taps, the Dreamcast VQ and NCC fits, the Smacker thresholds, the tape's
  noise, head-switch and dropout figures, the Env Chrome blend curves. The
  period combiners interpolate in float32, not the machines' fixed point.
- **Presets that need the scene's help.** The Tron preset glows only where
  a material carries a Glow Gel colour; the VHS presets record at one
  generation (raise Generations for a dub); the Super FX plot needs a fixed
  palette of at most sixteen entries with gamma 1 and no display transform.
- **Stereo pairs changed.** The convergence shift's sign was inverted
  before 1.90 (the eyes diverged); every stereo pair with a non-zero eye
  distance renders differently now.

**Blender's procedural textures differ slightly from Cycles.** Noise, Voronoi,
Musgrave and Magic are independently implemented from their published
definitions — the right kind of pattern with the right statistics, not
bit-identical to Blender's, so a material tuned against Cycles may need a
nudge. They also cannot travel to the GPU (their sin-fract hash decorrelates
on a driver's float32), so they route those materials to the CPU by name.
Halcyon's own pattern nodes ride an integer hash that is bit-exact on both
devices — the Fractal Noise node exists precisely to be the portable
replacement.

**The Sky Texture is Preetham, not Nishita.** In period, driven by the same
sun inputs, and close enough in shape for output about to be quantised to 256
colours. `altitude`, `air_density`, `dust_density` and `ozone_density` are
exported but unused.

**Volumes are real now — and stylized (1.66).** A mesh whose material
output links a **Volume** chain (Halcyon Volume, Principled Volume,
Volume Scatter/Absorption) becomes a container: every camera ray marches
its bounding box — the era's volume gizmos were boxes and spheres —
accumulating per-channel Beer-Lambert absorption and single scatter from
the scene's own lamps, falloffs, gobos, light linking and per-lamp
shadows included; a shadowed lamp carves **god-rays** through the fog.
Whatever chain feeds Density is evaluated at every march sample with
Generated coordinates spanning the box, so a Bozo or Cells chain shapes
clouds exactly as POV-Ray's media and 3D Studio's Volume Fog did. The
**Halcyon Volume** node adds the cel half: Edge Threshold/Softness cut
hard anime cloud edges from soft density, Bands posterizes the scattered
light into cel steps, Shadow Tint makes a shadowed region take a COLOUR
rather than darkness. Fixed-count midpoint marching, deterministic per
pixel; low step counts band, and the banding is the era's own slicing
artefact. A baked
fluid-sim DOMAIN feeds its smoke voxels straight into the march
(trilinear, defensively read — an unbaked cache says so and falls back
to the node chain), containers split per object, and traced
reflections pass through containers rather than reflecting a black
box. **Fog Bands** quantizes the depth fog itself into hard cel steps
— distance as flat painted planes. The screen-space kinds remain
alongside: light shafts smeared from bright pixels, the visible beam
cones, Height Fog.

**Volume models, container shapes, voxels (1.69).** The Halcyon Volume
node carries a **Model** menu like the master shader's: Henyey-
Greenstein (the default), Uniform, a dual-lobe cloud phase, POV-Ray's
Rayleigh / Mie haze / Mie murky media types, and two unlit atmospherics
from 3D Studio — **Volume Fog** (the fog colour composited by density,
exactly Color × (1 − T), no lamps) and **Combustion** (the Fire Effect:
Color is the sparse outer flame, Emission Color the dense inner one,
brightness follows density, Absorption sets how much the flames hide).
Every lit law is normalised to unit integral, so Color stays the albedo
whichever you pick. **Shape** decides what the ray marches: **Mesh** (the
default now — the container's own faces, depth-peeled by the engine's
rasteriser, so a sphere is a sphere and a torus keeps its hole; a mesh
that IS its bounding box takes the box road bitwise), or the Box /
Sphere / Cylinder gizmos of the era's atmospherics. **Voxels** reads the
density and colour chains — and a smoke grid — at the centres of an N³
lattice over the bound: the blocky voxel volume of the era's grid
volumetrics, 0 for continuous. Mesh containers use their face winding to
tell inside from outside (recalculate normals outward if a shape
inverts) and keep up to four solid spans per ray.

**The ink style pack (1.70).** The Cartoon Outlines panel's Line Style
box draws the line from a distance field the moment any dial leaves its
default: **Clean** (the trace machine's crisp even line — the 80s cel),
**Brush** (a soft bleeding edge) or **Pencil** (offset strokes with
grain); **Depth Taper** (thick near, thin far), **Interior Scale**
(thin creases under a thick silhouette), **Weight Noise** (the hand's
pressure along the line), **Shadow Side** (thicker where the surface
turns from the key lamp), **Boil** on a clock (on twos, on threes, or
one static wobble), **Grain**, a line colour **From Fill** (the
self-coloured line of hand-inked features and iro-trace anime) or a
**Gradient** by depth, height or light, and **True To Height** so a 4K
frame keeps its line. Defaults are the plain mask line, bitwise; pooled
bands draw the whole frame's line (a seam under thick outlines, there
since the outline pass shipped, is gone).

**Textures drive volume density (1.69.1).** Feed Density any chain and
it is evaluated at every march sample: Halcyon's own patterns (Bozo,
Cells, Fractal Noise) and Blender's procedurals (Noise, Voronoi, Wave,
Magic, Musgrave types) shape the fog in 3D. A volume sample carries the
pixel, the eye, the clock and its object, so **Texture Coordinate ▸
Window** projects an image from the camera through the fog (a drawn
cloud, lit and shadowed as a volume), **Object** follows the container's
matrix, and scrolling or clocked chains animate. A 2D image with no
Vector extrudes along the box's Z (Flat reads the Generated XY); set its
projection to **Box** and the three axis projections average into one
solid texture from a single tile. Blender 5.x renamed the texture nodes'
"Fac" outputs to "Factor" behind an unchanged identifier — Halcyon reads
the identifier now, so those outputs no longer resolve to zero (the
"textures make the volume vanish" report), and a chain that fails for
any reason is printed by name instead of shading a silent zero.

**Area lamps light by their shape (1.69).** A Disk or Ellipse area lamp
runs BI's Stokes contour over an equal-area 16-gon instead of its
bounding rectangle, normalised by the emitting area (π/4 of the
rectangle) so a disc and a square of the same size throw the same energy
and differ only in shape — as the soft shadows and beam cones already
did. The rectangle shapes are the compiled-C road bitwise; the GPU twin
bakes the polygon's corners.

**Every lamp projects a gobo.** Load an image into a lamp's Projected
Texture box: a Spot throws it through its cone like a slide projector, a
Sun/Hemi tiles it along its rays as a cloud shadow, a Point wraps it
around itself lat-long like a pierced lantern, an Area carries it on its
face like a printed gel. The lamp's Angle/Radius softens the projection
exactly as a bigger source would (baked into the image once, so both
devices read identical texels); extension (Era Default / Repeat / Extend
/ Clip — Clip is the projector's gate) and interpolation (Bilinear /
Closest / Cubic) are per-lamp choices, mirrored texel-for-texel in GLSL.

**Depth of field is layered, not sampled.** Depth slabs, each blurred by its
circle of confusion — a handful of blurs instead of hundreds of rays, as
compositors of the era did.

**Displacement drives a bump, not geometry.** The height becomes a normal
perturbation from its screen-space gradient. Nothing is tessellated, which is
also what 1990s scanline renderers did.

**Ambient occlusion and radiosity are not period-universal** and are off by
default. They're there because they're occasionally useful and, in
radiosity's case, because the era's boxes did ship it.

**Hair and particles render.** Particle-system hair and Curves-object hair
export as camera-facing tapered ribbons — the primitive Blender Internal
rendered strands as — children and emitter root UVs included, with the
**Hair Info** node live on both devices (Intercept, Random, Length,
Thickness off a documented colour-layer convention any custom mesh can
adopt via the material panel's Hair Geometry switch). **Add ▸ Halcyon ▸
Fur Shells** grows the era's other hair on any mesh: shell texturing with
an auto-wired tuft-threshold material, root-to-tip ColorRamp included —
and its dials (shells, length, packing, comb, density, colours) stay live
in **Object Properties ▸ Halcyon Fur**, re-growing the pelt as you drag,
with a Rebuild From Source button for when the source mesh itself changed.
The pelt wears its source: vertex groups, shape keys (as per-shell
deltas), deforming modifiers (Armature above all — rigged fur follows the
rig), every UV layer copied loop-exactly, the source's own colour layers,
and parenting. Topology-changing modifiers are skipped by name.
The fur material rides Alpha Clip, so a pelt renders at punch-through
speed instead of drowning the frame in blend layers — and it casts no
shadow, because the shadow roads see full quads, never the alpha holes:
a pelt that cast would blanket its own lower layers and the body in
black (the era's shell fur was never in a shadow map either). Both
flags travel with the material and can be flipped in the material
panel; the export health check names a casting tuft material with its
cure.
Emitter particles render as halos when their material is a halo, and as
instances when they instance objects or collections. Baking to texture is
not implemented. Motion blur *is* — as averaged time-offset frames across
a shutter, which is exactly how the era faked it, at the cost of Steps
extra renders.

**Mesh, curve, surface, text and metaball objects render**; Blender converts
each to triangles and Halcyon takes it from there. Hair curves render as
strand ribbons, and point clouds as halos when their material is one.
Grease pencil and volumes do not, and are named in the info bar rather
than quietly left out. An object that will not convert is skipped with its name
reported — one bad object costs you that object, not the frame.

---

## Performance

Profile first. **Developer Options ▸ Debug ▸ Timing Breakdown** prints a
per-stage table for every frame, and the GPU path prints its own splits down
to the reflection sweep's internals. Three rounds of optimising this engine
were once aimed at whichever stage happened to have been profiled, and twice
that was not the stage the frame was spending its time in. The engine now
optimises the way it renders: measured, on the scene that hurt, with losing
experiments written down so nobody re-fights them.

Recent measured wins on real field scenes, every one pixel-identical by
test: a 25-material, 171-object character file that first rendered in
**8:48** now renders warm in **0.7 s** — the arc that got there: SAH BVH
(shadow rays **12×**, reflection rays **6×**, both devices), light and
material values moved into textures so lamp and slider edits re-upload a
texel instead of recompiling shaders (and same-structure materials now
share one compiled shader), a G-buffer cache so an unchanged camera never
re-rasterises, a per-object mesh-export cache driven by Blender's own
update reports so an unchanged scene exports in a millisecond, the BVH
persisted to disk across sessions under a content digest, the frame's
draw calls batched into one main-thread crossing inside one render pass,
and shadow rays skipped exactly where a lamp contributes nothing. Losing
experiments are written down next to the wins so nobody re-fights them.

The knobs that matter, in order:

1. **`aa_samples` is quadratic — under Supersample.** Supersample 24 renders a
   5× frame each way — 25× the pixels. Drop to 1 while you light the scene, or
   switch Anti-Aliasing to **Adaptive (Edge Pass)**: it renders the frame once,
   then re-samples only the edge pixels (Bryce's anti-aliasing sweep, POV-Ray's
   `+A`), which buys Supersample-grade silhouettes for a few percent of the
   cost.
2. **`Pixel Scale` is free performance.** Renders at 1/N of the output size
   and nearest-upscales — 16× cheaper at 4×, and more authentic than shrinking
   a large render.
3. **Resolution.** Also quadratic.
4. **Ray tracing.** Reflective materials cost rays; depth multiplies them.
   The reflect console split will tell you which part owns the time.
5. **`shadow_map_size`.** Each map is a full rasterisation pass; a point light
   needs six. 512 is usually plenty at these resolutions.
6. **`preview_scale`** controls viewport resolution; the viewport also caches
   the exported scene and only rebuilds what changed.

### Threading, and why processes beat threads

Shading runs across a thread pool and is **bit-identical** at any thread
count — a test asserts it — but NumPy releases the interpreter lock only for
large array work, so threads mostly contend. **Use Worker Processes** splits
the frame across separate interpreters instead (possible only because `core/`
is bpy-free), bit-identical to in-process rendering. For sequences, **Lock
Palette** builds the adaptive palette once, and turning serpentine off lets
the error diffusion run a diagonal at a time — bit-identical and two to three
times faster.

---

## Layout

```
halcyon/
  core/          bpy-free renderer
    mathx.py       vector maths on (N,3) arrays
    scene.py       dataclasses the renderer consumes
    settings.py    RenderSettings — the 396 knobs
    raster.py      clipping, z-buffer, watertight edge rules, A-buffer
    bvh.py         binned-SAH BVH, wave traversal, order-free ties
    texture.py     sampling, mips, N64 three-point filter; the period
                   texel formats, compression, level roads and the
                   summed-area table (1.90.0)
    shading.py     the 49 models (Max's eight and the period light units
                   included)
    combine.py     the period combiners: the corner quantiser and each
                   machine's integer rule (1.90.0)
    fog.py         the fog curves, tables and cues both devices share
                   (1.90.0)
    shadowmask.py  planar shadows, Dreamcast modifier volumes, DS shadow
                   polygons: the mask over the opaque frame (1.90.0)
    reyes.py       the REYES micropolygon grid and its snap (1.90.0)
    srbump_tables.py  the PowerVR2 bump angle tables (1.90.0)
    lights.py      attenuation, shadow maps, PCF, ray shadows, gobos
    nodeeval.py    249 node types, the bump desugar, the space ramps
    patterns.py    the integer-hash pattern library, 1D–4D noise
    maxmaps.py     3ds Max's map algorithms on Halcyon's own tables
    bitex.py       the ported Blender Internal texture engine (+ tables)
    blend279.py    the 2.79 DNA reader and the BI material mapping
    brycemat.py    the Bryce 1995 .mat decoder — container, previews,
                   material records, texture colour tables
    sky.py         the sky modes, the Bryce dome, the painted backdrop,
                   the lit grounds, the weather overlay
    volume.py      real volumes: containers, eight scattering models
    sss.py         subsurface scattering, the BI way
    celfield.py    the cel's light: keys, screen shadows, the depth rim
    ink.py         the ink style pack: the distance-field line road
    lines.py       the inker's line: isophote weight, form / shadow /
                   tone lines, the stroke road
    media.py       the 2D media converters and paint strokes
    gouache.py     the painted background road, Meier's particles
    film.py        the colour process and the era looks
    wear.py        the print's wear: grain, dust, hairs, scratches
    render.py      the orchestrator, outlines, closure translation
    post.py        glow, palettes, dither, NTSC, CRT, JPEG
    palette.py     median cut, octree, k-means, VGA/Mac/EGA/HAM
    palette_era.py the era colour roads: register depth, Extra Half-Brite,
                   CRY, YJK, attribute cells, per-scanline palettes, Super
                   Black, the video legaliser (1.90.0)
    n64vi.py       the N64 VI's coverage blend and divot (1.90.0)
    signal_era.py  scan-out: the VI dither filter and gamma, the GameCube
                   copy filter, the PS2 CRTC blend, the 3dfx filters, the
                   3DO and GBA stretches (1.90.0)
    signal_tape.py the tape, the cables, the PAL receiver, chroma siting
                   (1.90.0)
    signal_codec.py  MPEG-1 intra blocks, Smacker, the backlit matte glow
                   (1.90.0)
    dither.py      Bayer, Floyd-Steinberg, Stucki, Atkinson, …
    geometry.py    the Add-menu objects, the seven terrains
    cones.py parallel.py worker.py stats.py convert.py
  gpu/           the GPU port
    craster.py     the compute rasteriser and its NumPy twin
    shade.py       frame planning, deferred passes, the ray sweeps
    material.py    GLSL assembly per material
    emit.py        186 node emitters
    procedural.py  the pattern, media and Max map libraries as GLSL,
                   twin by twin
    glsl_shading.py  the reflectance models in GLSL
    combine.py     the period combiners as GLSL text (1.90.0)
    rtrace.py      BVH kernels, the tie referral
    gbuffer.py     G-buffer packing and exact reconstruction
    ink.py         the ink pass (1.88.0): seeds, the mask roads, the
                   chamfer as a relaxation, the style maths, the composite
    sky.py         the sky / background drawn last in the shading burst
                   (1.89.0)
    frame.py       the frame kept on the GPU between stages, the
                   supersample resolve (1.89.0)
    stages_palette.py chain_palette.py   the palette snap, the ordered
                   dither and the era colour stages (1.90.0)
    stages_signal.py chain_signal.py     the scan-out stages (1.90.0)
    stages_tape.py chain_tape.py         tape, cable, PAL, chroma siting
                   (1.90.0)
    stages_codec.py chain_codec.py       MPEG-1, Smacker, the matte glow
                   (1.90.0)
    stages_vi.py chain_vi.py             the N64 coverage blend and divot
                   (1.90.0)
    device.py marshal.py stages.py chain.py capability.py
  shaders/       bpy-free GLSL/HLSL compiler and its NumPy interpreter
    lexer.py parser.py gtypes.py builtins.py codegen.py compiler.py
    interp.py
  nodes/         Blender node classes: the master, Anime and Cartoon
                 shaders and the BI Material node (shader_nodes.py), the
                 pattern and media shelf (pattern_nodes.py), the 3DS Max
                 shelf (max_nodes.py), the BI Texture node
  presets/       112 render presets, 303 skies (+ thumbs/), 48 waters
  tests/         headless suite, bpy stub, fake Blender, the fake device
                 (the driver's rules, headless), feature matrix, Blender
                 5.2's icon enum; and 1.90.0's fifteen modules, one or
                 more per pack, with their shared helper r251_common.py:
    test_r251_shadow.py
    test_r251_lighting_fog.py  test_r251_lighting_lamps.py
    test_r251_lighting_models.py
    test_r251_material.py  test_r251_material_nodes.py
    test_r251_raster.py  test_r251_raster_wire.py
    test_r251_texture.py
    test_r251_transparency.py
    test_r251_sky_camera.py
    test_r251_post_palette.py
    test_r251_post_signal.py  test_r251_post_tape.py
    test_r251_post_codec.py
  docs/          the preview sheets
  legacy_import.py   the 2.79 importer
  sparking.py        the Sparking! ZERO material importer (FModel .json)
  append_watch.py    the automatic lamp fix on plain File ▸ Append
  compat.py properties.py export.py engine.py preview.py ui.py
  objects.py convert.py templates.py selftest.py
```

---

## Licence

GPL-3.0-or-later, matching Blender.

---

## Credits

Built by Claude with help from Mr. Emotiman.

This Addon is and always will be free. If you paid for this, you were scammed.
Please demand your money back and report the seller.
