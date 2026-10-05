# Halcyon Render Engine

**A from-scratch render engine for the looks Cycles can't do — every 3D era
before physically-based rendering, and 2D down to the wire**

---

> **This Addon is and always will be free. If you paid for this, you were
> scammed. Please demand your money back and report the seller.**

Halcyon is GPL-3.0-or-later, the same licence as Blender. You are free to use,
modify and share it. Nobody is entitled to charge you for it.

---

Halcyon is a complete render engine, not a post-processing filter over a
modern render: its own scanline z-buffer rasteriser with optional ray tracing,
the reflectance models the old packages actually shipped, real framebuffer
quantisation, a genuine GLSL/HLSL compiler for coded-shader nodes, and a
complete GPU port held to the CPU picture pixel for pixel. Python and NumPy
only — nothing compiled — and the whole engine runs headless without Blender
for its 9503-check test suite.

**Three roads, one engine.**

**The eras.** 112 render presets, each what its target actually did:
Infini-D, Ray Dream, 3D Studio R4 and MAX, trueSpace, LightWave, Imagine,
POV-Ray, Bryce, Softimage, Alias, RenderMan; VGA, EGA, CGA, the Mac 1-bit and
8-bit palettes, Amiga OCS/AGA/HAM, Atari ST, PC-98, X68000; PlayStation,
Saturn, N64, Voodoo, Dreamcast; NTSC, VHS, the early web. Gouraud and Flat are
treated as *shading rates*, not reflectance models, because that is what they
were — selecting Gouraud evaluates lighting once per vertex and interpolates,
which is where the banding genuinely comes from. PlayStation mode snaps
vertices and drops perspective correction, so textures swim across polygons
the way they did. The Amiga HAM modes reproduce hold-and-modify properly,
fringing and all. Applying a preset resets everything first, so presets never
accumulate.

**The libraries.** Open your Blender 2.79 scenes: File ▸ Import ▸ Legacy Scene
(.blend) — or a plain File ▸ Append — brings a 2.79-or-earlier file in whole,
its Blender Internal materials rebuilt one to one (shader pairs, ramps,
mirror, transparency, all eighteen texture slots, the original procedural
textures on a ported BI Texture engine), lamps at their true 2.79 energies
and falloffs, shadows in both ray and buffer forms, the old world as sky,
ambient and fog — transcribed against the 2.79 source and proven by the test
suite. The Bryce 1995 material library's `.mat` files were decoded byte by
byte, and 46 presets sit on the Pre-Made shelf carrying the numbers Bryce
saved in December 1995. And 3ds Max's material and map library — the Standard
material's eight shaders, the Raytrace material, Blend / Double Sided /
Top-Bottom / Shellac / Composite, the maps, utilities and coordinate
rollouts, a shelf of thirty-five nodes — runs on Max's own algorithms, reimplemented on Halcyon's own tables
(no Autodesk code or data ships), with a 3ds Max 2012 render preset.

**The 2D road.** A Cartoon Shader — paint, not light, with the Golden Age,
UPA, Xerox-line, Saturday-morning, 90s feature, wartime noir and modern flat
eras on a menu — and an Anime Shader with kage tones, the hair shine, the
airbrush, a six-decade Style menu, and compatibility modes that decode real
game texture conventions (ArcSys, Genshin, ZZZ, and Sparking! ZERO from the
game's own material export, with an importer that builds a character's
materials from its FModel .json). An ink pass drawn from the
shared G-buffer: clean, brush and pencil lines that taper, roughen, drift,
skip and boil, the inker's weight by the light, Arc System Works' vertex-colour
line control. 2D media converters that turn the lit cel into hatching, pencil
scribble, stipple, charcoal, ink wash, paint strokes and paper. Painted
backdrops on a world-fixed panel and painted grounds laid in brush strokes.
And the film the cel went through: the three-strip camera's records and
filters, the timed print curve, impure dyes, dye-transfer registration, the
silver key, then grain by size and clump, dust on the cel, the negative and
the print, hairs in the gate, scratches, cue marks, gate weave, flicker,
frames held on twos and threes.

## What you get

**49 shading models**, each implemented from its published formulation —
Lambert, Gouraud, Flat, Phong, Blinn-Phong, Blinn, Cook-Torrance, Oren-Nayar,
Minnaert, Ward, Anisotropic, Metal, Strauss, Multi-Layer, Toon, Translucent,
Constant, Wireframe, Anime, Cartoon, Oren-Nayar-Blinn, the whole Blender
Internal diffuse/specular matrix, 3ds Max's own eight, and seventeen period
light units and combiners (GameCube, Sega Model 2 and 3, Nintendo DS,
PlayStation, Saturn, N64, PS2, Direct3D, PowerVR, Mega Drive, Super FX).

**249 node types** are evaluated — every shader node Blender 5.x offers (the
full Principled BSDF, recursive node groups, every Math and Vector Math
operation) plus Halcyon's own 143: the master shader with the era's whole bag
of tricks on one node, 34 procedural textures, the 2D media, thirty utilities
and vector warps, the BI Texture and BI Material nodes, the 3DS Max shelf.
Nodes the engine does not recognise pass through and report a warning rather
than failing the render.

**Coded shader nodes.** A real GLSL and HLSL compiler — preprocessor,
recursive-descent parser, type inference, and code generation with SIMT
execution masks, so different pixels genuinely take different branches.
Declare `uniform float rimPower = 2.5;` and a Rim Power socket appears on the
node. Coded shaders compile natively into the GPU's deferred pass too.

**114 material templates** on the Pre-Made shelf in fourteen families — the
engine's own recipes, the 46 decoded Bryce presets, the cel, cartoon, media
and volume shelves. Picking one adds its nodes beside your graph.

**Real volumes** — a mesh whose material links a Volume chain marches as a
container: per-channel Beer-Lambert, single scatter from the scene's own
lamps and shadows (god rays included), eight scattering models, mesh-shaped
containers, voxels, smoke grids, and the cel dials for stylized fog.

**Eleven sky modes** — node tree, solid, gradient, banded, starfield, Preetham,
HDRI, a full Bryce Sky Lab, the painted backdrop, Doom's Cylinder Sky and
LightWave's Gradient Backdrop — with **303 sky
presets** in a thumbnail gallery, **48 water presets**, and **ten infinite
grounds**: nine that answer the scene's lighting, and the SNES's Mode 7
floor, which is unlit. Weather, a seven-landform terrain
generator, halo materials and lens flares against the frame's own depth.

**Output that lands in the right decade.** Colour depth from 32-bit down to
1-bit, real VGA, Macintosh, EGA, CGA and web-safe palettes, four adaptive
quantisers, seven error-diffusion kernels plus ordered dither, any image as a
palette, composite NTSC encoding with chroma bleed and dot crawl, CRT masks,
interlacing, and a genuine 8×8 DCT round-trip for JPEG artefacts. **143
resolution presets** from the NES's 8:7 pixel to 8K and the 35 mm scan
formats.

**396 settings, all proven.** One test holds every setting to evidence that
it changes what it claims to change; another fails the build if any property
ships without a real tooltip.

## GPU support

**The GPU port is complete** — rasterisation, deferred shading and post all
run through Blender's own `gpu` module, and since 1.89.0 the frame stays
there between the stages: the sky is drawn in the shading's own burst, the
ink and the supersample resolve read the frame in place, and the post chain
runs resident — one upload at most, one readback at the end — with the film
grain and the bit depth as stages of their own. Set Device: GPU in Render
Properties.
A 299-row feature matrix in the built-in Self Test compares the GPU picture
against the CPU's on your own driver; anything that cannot run on the GPU is
routed to the CPU **by name**, with the reason printed, and the picture stays
right. Same frame, same numbers, on either device, in any band, on any worker.

## Getting started

1. Set **Render Properties ▸ Render Engine ▸ Halcyon**
2. Open the **Halcyon Presets** panel and load one — *VGA Mode 13h* or
   *PlayStation* show the character of the engine fastest; *Cel: 1940s
   Technicolor feature* with a Cartoon template shows the other road
3. If the Display panel warns that Blender's view transform is not Standard,
   press the button it offers. Halcyon outputs display-referred pixels, so AgX
   on top double-transforms them
4. On an existing scene, use **Convert Whole Scene** in the Material panel;
   for a 2.79 file, File ▸ Import ▸ Legacy Scene or a plain Append

## Performance

**Pixel Scale** is the cheapest large win: it renders at 1/N of your output
resolution and scales back up with nearest-neighbour — set 1920×1080 with
Pixel Scale 4× and the engine renders 480×270, which is more authentic than
rendering at 1080p anyway. After that, `aa_samples` costs the most (it is
quadratic) — or switch to Adaptive anti-aliasing, which re-samples only the
edges. **Use Worker Processes** splits a frame across interpreters,
bit-identical to in-process. The Timing Breakdown prints a per-stage table
when you want to know where a slow frame spends its time.

## Requirements

Blender 5.1 or newer. No compiled dependencies — it uses only NumPy, which
Blender already ships. Developed and tested against Blender 5.2.

## Honest limitations

- Depth of field is layered (depth slabs by circle of confusion), not sampled
- Motion blur averages time-offset frames across a shutter — the era's way
- Displacement drives a bump, not tessellation
- Hair renders as tapered strand ribbons (particle and Curves hair) or as
  shell-textured fur; emitter particles render as halos or as instances
- Blender's own procedural noises are re-implemented from their published
  definitions — the right pattern, not bit-identical to Cycles, and they
  shade on the CPU by name; Halcyon's own pattern nodes are the portable,
  bit-exact replacements
- The Sky Texture node is Preetham, not Nishita
- Most film stages run on the CPU on both device roads; the grain (to
  1.2 px) and the flicker draw on the GPU, within a measured tolerance
- The 3ds Max shelf reproduces Max's algorithms and controls, not a Max
  scene's exact random tables — the character, not the pattern

## Reporting problems

Turn on **Developer Options**, then **Run Self Test** in the Debug panel: it
copies a report covering your GPU, the shaders compiled on your driver, the
299-row CPU/GPU feature comparison and per-stage timings. Nearly every bug in
this engine's history was diagnosed from that output.

## Credits

Built by Claude with help from Mr. Emotiman.

## Licence

GPL-3.0-or-later, matching Blender.

This Addon is and always will be free. If you paid for this, you were scammed.
Please demand your money back and report the seller.
