# Halcyon Render Engine

**Period-accurate mid-to-late 1990s CGI, not a filter**

---

> **This Addon is and always will be free. If you paid for this, you were
> scammed. Please demand your money back and report the seller.**

Halcyon is GPL-3.0-or-later, the same licence as Blender. You are free to use,
modify and share it. Nobody is entitled to charge you for it.

---

Halcyon is a complete render engine that reproduces what 3D software looked like
when it ran on a home computer. Not a post-processing filter over a modern
render — a scanline z-buffer rasteriser with optional ray tracing, the
reflectance models those packages actually shipped, and real framebuffer
quantisation at the end of it.

The difference shows. Gouraud and Flat are treated as *shading rates*, not
reflectance models, because that is what they were: selecting Gouraud evaluates
lighting once per vertex and interpolates the colour across the triangle. That
is where the banding genuinely comes from, and it is why it looks right rather
than merely blurred. Transparency uses a true A-buffer. PlayStation mode snaps
vertices to integer screen coordinates and drops perspective correction, so
textures swim across polygons the way they did. The Amiga HAM modes reproduce
hold-and-modify properly, fringing and all.

**Open your Blender 2.79 scenes.** File ▸ Import ▸ Legacy Scene (.blend) — or
a plain File ▸ Append — brings a 2.79-or-earlier file in whole: the Blender
Internal materials rebuild one to one (shader pairs, ramps, mirror,
transparency, all eighteen texture slots, the original procedural textures on
a ported BI Texture engine), lamps arrive with their true 2.79 energies and
falloffs, shadows convert in both ray and buffer forms, and the old world
comes along as sky, ambient and fog. Transcribed against the 2.79 source code
and proven by the test suite, not approximated.

**The Bryce 1995 material library, decoded.** The original preset collection's
`.mat` files were cracked byte by byte — material channel records, colour
tables, refractive indices — and 46 presets sit on the Pre-Made shelf carrying
the exact numbers Bryce saved in December 1995, notes quoting the original
library text.

## What you get

**18 shading models**, each implemented from its published formulation —
Lambert, Gouraud, Flat, Phong, Blinn-Phong, Blinn, Cook-Torrance, Oren-Nayar,
Minnaert, Ward, Anisotropic, Metal, Strauss, Multi-Layer, Toon, Translucent,
Constant and Wireframe.

**75 render presets** across six categories. Infini-D, Ray Dream, StudioPro,
3D Studio R4 and MAX, trueSpace, LightWave, Imagine, POV-Ray, Bryce,
ElectricImage, Softimage|3D, Alias PowerAnimator, Wavefront, CINEMA 4D, Real
3D, Vistapro, Animation:Master, Vue. VGA Mode 13h, EGA, CGA, Hercules,
Macintosh 8-bit and 1-bit, Windows 3.1 and 95, Amiga OCS and AGA, Atari ST,
PC-98, X68000, SVGA, Quake software. PlayStation, Saturn, N64, Voodoo,
Dreamcast, 3DO, Jaguar. Video Toaster, PAL, VHS, S-Video. Web GIF, JPEG,
PNG-8, CD-ROM FMV. Applying one resets everything first, so presets never
accumulate.

**167 Blender node types** are evaluated, including the full Principled BSDF
and recursive node groups. Nodes the engine does not recognise pass through
and report a warning rather than failing the render.

**74 material templates** in the shader editor's Pre-Made menu, sorted into
ten families — the engine's own 28 plus the 46 decoded Bryce 1995 presets.
Picking one adds its nodes beside your graph rather than replacing it.

**26 procedural texture nodes** of the kind these packages shipped — Marble,
Wood, Granite, Dents, Crackle, Plasma, Ripples, Caustics, Starfield, Weave,
Scratches, Tiles, Spiral, Cells, TV Static, the POV-Ray family (Bozo, Agate,
Leopard, Onion, Bumps, Wrinkles, Brick), a 1D–4D Fractal Noise whose integer
hash is bit-exact on CPU and GPU, an animated Water Noise with an exact loop,
a shaped Gradient, and Matcap Coordinates. Solid textures, evaluated in 3D,
so a shape carved out of marble has veins running through it.

**Coded shader nodes.** A real GLSL and HLSL compiler — preprocessor,
recursive-descent parser, type inference, and code generation with SIMT
execution masks, so different pixels genuinely take different branches.
Declare `uniform float rimPower = 2.5;` and a Rim Power socket appears on the
node, defaulted to 2.5. Coded shaders compile natively into the GPU's
deferred pass too, so the same source runs on both devices.

**Eight sky modes** — node tree, solid, gradient, banded gradient, starfield,
Preetham physical sky, HDRI, and a full Bryce Sky Lab: sky dome, sun corona,
haze, ground fog, a wind-streaked stratus deck, a self-shadowed cumulus deck
built from turbulence, a rainbow at the correct 42 degrees, stars, comets and
a nebula wash. **303 sky presets** with rendered thumbnails in a browsable
gallery, and **48 water presets** under the water plane, where Bryce kept
them.

**Nine infinite grounds that answer the scene's lighting** — checker, tiles
(thin glowing grout is the synthwave floor), dunes, snowfield, lava, an
animated ocean with a directional wave spectrum, and a Material mode that
paints any node graph to the horizon. Sun angle, lamp falloff and cast
shadows from geometry land on the plane; a Scene Lighting dial returns the
old self-lit look exactly.

**Weather** — rain, snow, embers and ash over any finished frame,
deterministic and stable under timeline scrubbing.

**A terrain generator** — seven landforms from mountain to volcano, wired for
the Altitude & Slope node, with a ColorRamp downstream making the classic
snow-line in two nodes.

**Halo materials** — the era's glow sprites against the frame's own depth:
images, noise, rays, star spikes, lightning bolts, shockwaves, halos along
curves, and lens flares that check their lamp's visibility.

**Output that lands in the right decade.** Colour depth from 32-bit down to
1-bit, real VGA, Macintosh, EGA, CGA and web-safe palettes, four adaptive
quantisers, seven error-diffusion kernels plus ordered dither — or force the
whole frame through the colours of any image you pick — composite NTSC
encoding with chroma bleed and dot crawl, CRT aperture grille and shadow
masks, interlacing, and a genuine 8×8 DCT round-trip for JPEG artefacts.

**198 settings, all proven.** One test holds every setting to evidence that
it changes what it claims to change; another fails the build if any property
ships without a real tooltip.

## GPU support

**The GPU port is complete** — rasterisation, deferred shading and post all
run through Blender's own `gpu` module. Set Device: GPU in Render Properties.
A 116-row feature matrix in the built-in Self Test compares the GPU picture
against the CPU's on your own driver; anything that cannot run on the GPU is
routed to the CPU **by name**, with the reason printed, and the picture stays
right. Frame-to-frame caches mean an animation re-uploads only what moved.

## Getting started

1. Set **Render Properties ▸ Render Engine ▸ Halcyon**
2. Open the **Halcyon Presets** panel and load one — *VGA Mode 13h* or
   *PlayStation* show the character of the engine fastest
3. If the Display panel warns that Blender's view transform is not Standard,
   press the button it offers. Halcyon outputs display-referred pixels, so AgX
   on top double-transforms them
4. On an existing scene, use **Convert Whole Scene** in the Material panel

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

- Volumetrics are screen-space light shafts; no volume is integrated
- Depth of field is layered (depth slabs by circle of confusion), not sampled
- Motion blur averages time-offset frames across a shutter — the era's way
- Displacement drives a bump, not tessellation
- Particles and hair are not supported; halo materials cover the classic
  glow-sprite look
- Blender's own procedural noises are re-implemented from their published
  definitions — the right pattern, not bit-identical to Cycles, and they
  shade on the CPU by name; Halcyon's own pattern nodes are the portable,
  bit-exact replacements
- The Sky Texture node is Preetham, not Nishita

## Reporting problems

Turn on **Developer Options**, then **Run Self Test** in the Debug panel: it
copies a report covering your GPU, the shaders compiled on your driver, the
116-row CPU/GPU feature comparison and per-stage timings. Nearly every bug in
this engine's history was diagnosed from that output.

## Credits

Built by Claude with help from Mr. Emotiman.

## Licence

GPL-3.0-or-later, matching Blender.

This Addon is and always will be free. If you paid for this, you were scammed.
Please demand your money back and report the seller.
