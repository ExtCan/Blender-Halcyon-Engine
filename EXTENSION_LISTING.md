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

## What you get

**18 shading models**, each implemented from its published formulation —
Lambert, Gouraud, Flat, Phong, Blinn-Phong, Blinn, Cook-Torrance, Oren-Nayar,
Minnaert, Ward, Anisotropic, Metal, Strauss, Multi-Layer, Toon, Translucent,
Constant and Wireframe.

**52 presets** across six categories. Infini-D, Ray Dream, StudioPro, 3D Studio
R4 and MAX, trueSpace, LightWave, Imagine, POV-Ray, Bryce, ElectricImage,
Softimage|3D, Alias PowerAnimator, Wavefront, CINEMA 4D, Real 3D, Vistapro,
Animation:Master, Vue. VGA Mode 13h, EGA, CGA, Hercules, Macintosh 8-bit and
1-bit, Windows 3.1 and 95, Amiga OCS and AGA, Atari ST, PC-98, X68000, SVGA,
Quake software. PlayStation, Saturn, N64, Voodoo, Dreamcast, 3DO, Jaguar. Video
Toaster, PAL, VHS, S-Video. Web GIF, JPEG, PNG-8, CD-ROM FMV.

**105 Blender node types** are evaluated, including the full Principled BSDF and
recursive node groups. Nodes the engine does not recognise pass through and
report a warning rather than failing the render.

**Coded shader nodes.** A real GLSL and HLSL compiler — preprocessor,
recursive-descent parser, type inference, and code generation with SIMT
execution masks, so different pixels genuinely take different branches. Declare
`uniform float rimPower = 2.5;` and a Rim Power socket appears on the node,
defaulted to 2.5. Declare an output and an output socket appears.

**12 procedural textures** of the kind these packages shipped with — Marble,
Wood, Granite, Dents, Crackle, Plasma, Ripples, Starfield, Weave, Scratches,
Tiles and Spiral. Solid textures, evaluated in 3D, so a shape carved out of
marble has veins running through it. Plasma and Ripples animate.

**Six sky modes** — node tree, solid, gradient, Preetham physical sky, HDRI, and
a full Bryce atmosphere with sky dome, sun corona, haze, a wind-streaked stratus
deck, a self-shadowed cumulus deck built from turbulence, a rainbow at the
correct 42 degrees, and stars.

**Material conversion.** Three buttons convert the active material, everything on
the selected objects, or the whole scene onto the Halcyon shader — relinking
existing textures rather than discarding them, and choosing a reflectance model
from what the source shader actually was.

**Output that lands in the right decade.** Colour depth from 32-bit down to
1-bit, real VGA, Macintosh, EGA, CGA and web-safe palettes, four adaptive
quantisers, seven error-diffusion kernels plus ordered dither, composite NTSC
encoding with chroma bleed and dot crawl, CRT aperture grille and shadow masks,
interlacing, and a genuine 8×8 DCT round-trip for JPEG artefacts.

132 settings in total, and 17 period resolutions with their correct pixel
aspects.

## Getting started

1. Set **Render Properties ▸ Render Engine ▸ Halcyon**
2. Open the **Halcyon Presets** panel and load one — *VGA Mode 13h* or
   *PlayStation* show the character of the engine fastest
3. If the Display panel warns that Blender's view transform is not Standard,
   press the button it offers. Halcyon outputs display-referred pixels, so AgX
   on top double-transforms them
4. On an existing scene, use **Convert Whole Scene** in the Material panel

Applying a preset resets everything first, so presets never accumulate. Machine
settings like thread count are preserved.

## Performance

Shading runs across threads, and **Pixel Scale** is the cheapest large win: it
renders at 1/N of your output resolution and scales back up with
nearest-neighbour, so the output stays the size you set while the render costs
N² times less. Set the output to 1920×1080 with Pixel Scale at 4× and the engine
renders 480×270 — which is more authentic than rendering at 1080p anyway.

After that, `aa_samples` is the setting that costs the most: it is quadratic, and
most presets ship at 4.

## Requirements

Blender 5.1 or newer. No compiled dependencies — it uses only NumPy, which
Blender already ships.

## Please read before installing

This is a CPU renderer written in Python. It is not fast, and there is **no GPU
support**. A 640×480 frame takes seconds to low minutes depending on the scene.
Use the period resolutions and Pixel Scale and it is perfectly usable; expect
1080p at high sample counts to be slow.

Other things it does not do:

- No volumetrics, depth of field, motion blur, particles or hair
- Displacement is evaluated but does not tessellate geometry
- Procedural noise is independently implemented from published definitions, so
  it is the right kind of pattern but not bit-identical to Cycles — a material
  tuned against Cycles will need a nudge
- The Sky Texture node uses the Preetham analytic model rather than Blender's
  Nishita atmospheric simulation
- Ambient occlusion is available but off by default, because it is not period
  correct

Developed and tested against Blender 5.2. Blender 5.1 is supported but has not
been tested directly; earlier versions are not supported.

## Reporting problems

There is a **Halcyon Diagnostics** button at the bottom of the Performance panel.
It prints the scene as the engine actually receives it — every node with its
properties and links, the world mode, materials, lights and resolved settings —
to the system console (**Window ▸ Toggle System Console**). Including that output
with a bug report makes almost anything diagnosable in one pass.

## Credits

Built by Claude with help from Mr. Emotiman.

## Licence

GPL-3.0-or-later, matching Blender.

This Addon is and always will be free. If you paid for this, you were scammed.
Please demand your money back and report the seller.
