# Halcyon

> **This Addon is and always will be free. If you paid for this, you were
> scammed. Please demand your money back and report the seller.**

A from-scratch render engine for Blender that reproduces the output of
mid-to-late 1990s home-computer 3D software.

Not a filter over a modern render. A scanline z-buffer rasteriser with optional
ray tracing, the reflectance models those packages actually shipped, real
framebuffer quantisation, a genuine GLSL/HLSL compiler for the coded-shader
nodes — and a complete GPU port of all three stages, proven against the CPU
picture feature by feature on real hardware. 48,000 lines of Python and NumPy
(68,000 with the test suite), no compiled dependencies.

![contact sheet](docs/halcyon_contact_sheet.png)

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
covering your GPU, the shaders compiled on your own driver, a 116-row
feature-by-feature comparison of the GPU picture against the CPU's, per-stage
frame timings and thread scaling. Nearly every bug in this engine's history was
diagnosed from that output; almost none from a description alone.

---

## What it does

**18 shading models**, each implemented from its published formulation rather
than approximated:

Lambert · Gouraud · Flat · Phong · Blinn-Phong · Blinn · Cook-Torrance ·
Oren-Nayar · Minnaert · Ward · Anisotropic · Metal · Strauss · Multi-Layer ·
Toon · Translucent · Constant · Wireframe

Gouraud and Flat are treated as **shading rates**, not reflectance models,
because that is what they are. Selecting Gouraud on a material evaluates its
lighting once per vertex and interpolates the colour across the triangle;
selecting Flat evaluates once per face. That is where the banding and the
faceting genuinely come from, and it is why they look right instead of merely
blurry.

**145 node types** are evaluated — audited against Blender's full surface-node
registry, and the audit came back clean: every shader node Blender 5.x offers
has an evaluator except Freestyle's stroke UV, which has no meaning outside
Freestyle. That includes the full Principled BSDF, node groups (recursively),
muted nodes, reroutes, all the texture and colour nodes, and every Math and
Vector Math operation. Nodes the engine doesn't know pass their first matching
input through and are reported as a warning rather than failing the render.

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

**Cartoon outlines.** Ink drawn from the renderer's own G-buffer: object
silhouettes, material borders, depth breaks and normal creases, each its own
toggle, with width, colour, opacity and over-sky control. The ink lands at the
*internal* resolution, so supersampling anti-aliases the line on the way down —
clean cel edges with no line renderer. Computed from the same buffers on either
device, so the picture cannot differ between them. Render Properties ▸ Shading
▸ Cartoon Outlines.

**28 material templates** — Chrome through Wireframe on the Simple shelf;
Water, Lava, Tile Floor, Brick Wall, Hammered Metal, Leopard, Cloth and Dead
Channel among the Advanced. Built at runtime as recipes rather than saved node
trees, so they always match the current nodes, and every one is rendered by the
test suite to prove it changes the frame.

**25 procedural texture nodes** of the kind these packages shipped —
Marble, Wood, Granite, Dents, Crackle, Plasma, Ripples, Starfield, Weave,
Scratches, Tiles, Spiral, Cells, TV Static and the POV-Ray family (Bozo, Agate,
Leopard, Onion, Bumps, Wrinkles, Brick), plus:

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

Each pattern is written from its published definition rather than from
something that looks similar. Agate's 0.77 exponent is the whole character of
that pattern; leopard really is three sines summed and squared.

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

**Eight sky modes** — node tree, solid, gradient, **banded gradient** (the same
blend cut into the handful of flat steps a 256-colour palette could actually
spare for a sky), **starfield**, Preetham physical sky, HDRI with rotation and
tint, and a full **Bryce Sky Lab**: sky dome, sun corona, haze that warms
toward the sun, separate ground fog, a wind-streaked stratus deck, a
self-shadowed cumulus deck built from turbulence rather than fBm (which is
where the cauliflower edges come from), a rainbow at the correct 42 degrees,
stars, comets that travel their own great circles off the scene's clock — and
now a **nebula wash** under the Bryce dome as well as the starfield, because a
night preset that sets nebula settings should get a nebula.

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

**20 water presets** in the same shape, under the water plane rather than the
Sky Lab because that is where Bryce kept them. Skies and waters own disjoint
halves of the world, so applying one never disturbs the other.

**Nine infinite grounds** — solid, checker, fractal, **neon grid** (the
synthwave floor, lines widening with distance so they survive minification),
**tiles** with grout and per-tile shading, **dunes**, **snowfield** with
sun-glints, **lava** with pulsing cracks, and the full **animated ocean**: a
directional wave spectrum fanned off a wind direction, deep and shallow colours
with the path length between them, and the sun's glitter found in the
distribution of wave normals rather than painted on. All intersected
analytically in the background pass, exactly as POV-Ray and Bryce provided one.

**72 render presets** across six categories — 3D software (Infini-D, Ray
Dream, 3D Studio, trueSpace, LightWave, POV-Ray, Bryce, Softimage|3D and the
rest), home computers (VGA Mode 13h through PC-98 and X68000), consoles
(PlayStation, Saturn, N64, Voodoo, Dreamcast…), broadcast (Video Toaster, PAL,
VHS), handhelds, and the early web (GIF, JPEG, CD-ROM FMV). Applying one
resets everything first, so presets never accumulate.

**Render passes** — Depth, Normal, Position, UV, Object Index and Material
Index, written under Blender's own names and channel layouts so a Halcyon Z
pass drops into a comp built for Cycles without rewiring.

**189 settings, all exposed, all proven, all explained.** Two tests stand
behind that sentence: one holds every setting to a proof that it changes what
it claims to change (a matrix row, an A/B render, a behavioural check, or a
declared reason — nothing silently exempt), and one fails the build the moment
any property ships without a detailed tooltip.

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
objects, all 28 templates rendering, all 303 skies applying and differing, all
72 presets, every setting's proof, and every tooltip's existence).

The GPU pipeline is tested headlessly too: the same GLSL the driver compiles
is executed by the compiler's own NumPy backend against the same packed
textures — raster kernel, deferred shading, ray kernels, post stages — so a
change that would move a GPU pixel fails the suite on a machine with no GPU at
all. The final word still belongs to hardware: **Run Self Test** renders the
116-row feature matrix on your actual driver and reports any row where the
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

- **Rasterisation** runs as a compute kernel producing the same G-buffer the
  CPU produces — triangle ids, perspective-correct barycentrics, depth at the
  chosen precision, the affine-warp interpolants, snap and 16-bit modes
  included. Coverage at shared edges is governed by the same watertight window
  and canonical clip rules on all four engines (CPU loop, CPU batch, GLSL
  kernel, NumPy replay), because a half-ulp disagreement near the near plane
  once opened pixel-wide cracks.
- **Deferred shading** packs the G-buffer into textures and shades one
  full-screen pass per material, every frame constant baked into the shader
  source. Lights, shadow atlases (cube faces included, every Vogel PCF tap
  reproduced), image textures with the CPU's own filter arithmetic, vertex
  colours, per-pixel surface parameters, the master shader's whole colour
  chain, coded GLSL nodes inlined natively, and the ray sweeps — reflections
  and refractions to any depth, traced by compute kernels against the same
  SAH tree, with each level's secondary passes scissored to the pixels its
  rays actually hit.
- **Post** runs the parallel stages as GLSL, measured stage by stage on real
  hardware before each was allowed to default on. Error-diffusion dither is
  inherently serial and stays on the CPU, honestly.

The whole matrix — 116 feature rows — is compared against the CPU picture by
**Run Self Test** on your own driver. A row that cannot yet run on the GPU is
*routed*: the frame says why on the console and shades that part on the CPU,
so the picture is always right and the reason is always named. The same
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

**Volumetrics are screen-space.** A light's shafts are smeared from bright
pixels, which is how it was done then. No volume is integrated.

**Depth of field is layered, not sampled.** Depth slabs, each blurred by its
circle of confusion — a handful of blurs instead of hundreds of rays, as
compositors of the era did.

**Displacement drives a bump, not geometry.** The height becomes a normal
perturbation from its screen-space gradient. Nothing is tessellated, which is
also what 1990s scanline renderers did.

**Ambient occlusion and radiosity are not period-universal** and are off by
default. They're there because they're occasionally useful and, in
radiosity's case, because the era's boxes did ship it.

**Particles and hair are not supported.** Baking to texture is not implemented
either. Motion blur *is* — as averaged time-offset frames across a shutter,
which is exactly how the era faked it, at the cost of Steps extra renders.

**Mesh, curve, surface, text and metaball objects render**; Blender converts
each to triangles and Halcyon takes it from there. Grease pencil, hair curves,
point clouds and volumes do not, and are named in the info bar rather than
quietly left out. An object that will not convert is skipped with its name
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

Recent measured wins on a real half-million-triangle field scene: SAH BVH —
shadow rays **12×**, reflection rays **6×**, both devices; sky evaluation
**1.9×** (a float64 leak and a corner-hash reuse, bit-for-bit where it
counts); the GPU frame's composite bucket **3.7×**; all-miss reflection
levels skipped outright.

The knobs that matter, in order:

1. **`aa_samples` is quadratic.** Supersample 24 renders a 5× frame each way —
   25× the pixels. Drop to 1 while you light the scene.
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
    settings.py    RenderSettings — the 189 knobs
    raster.py      clipping, z-buffer, watertight edge rules, A-buffer
    bvh.py         binned-SAH BVH, wave traversal, order-free ties
    texture.py     sampling, mips, N64 three-point filter
    shading.py     the 18 reflectance models
    lights.py      attenuation, shadow maps, PCF, ray shadows
    nodeeval.py    145 node types, the bump desugar, the space ramps
    patterns.py    the integer-hash pattern library, 1D–4D noise
    sky.py         eight sky modes, the Bryce dome, nine grounds
    render.py      the orchestrator, outlines, closure translation
    post.py        glow, palettes, dither, NTSC, CRT, JPEG
    palette.py     median cut, octree, k-means, VGA/Mac/EGA/HAM
    dither.py      Bayer, Floyd-Steinberg, Stucki, Atkinson, …
    geometry.py    the Add-menu objects, generated
  gpu/           the GPU port
    craster.py     the compute rasteriser and its NumPy twin
    shade.py       frame planning, deferred passes, the ray sweeps
    material.py    GLSL assembly per material
    emit.py        82 node emitters
    procedural.py  the pattern library as GLSL, twin by twin
    rtrace.py      BVH kernels, the tie referral
    gbuffer.py     G-buffer packing and exact reconstruction
    device.py marshal.py stages.py chain.py capability.py
  shaders/       bpy-free GLSL/HLSL compiler
    lexer.py parser.py gtypes.py builtins.py codegen.py compiler.py
  nodes/         Blender node classes (master shader, patterns, ramps)
  presets/       72 render presets, 303 skies (+ thumbs/), 20 waters
  tests/         headless suite, bpy stub, fake Blender, feature matrix
  compat.py properties.py export.py engine.py ui.py objects.py convert.py
```

---

## Licence

GPL-3.0-or-later, matching Blender.

---

## Credits

Built by Claude with help from Mr. Emotiman.

This Addon is and always will be free. If you paid for this, you were scammed.
Please demand your money back and report the seller.
