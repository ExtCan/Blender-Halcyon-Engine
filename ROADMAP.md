# Halcyon — the road ends: 1.30.0

This file used to be called "the road to 1.26.0", and 1.26.0 was "the
big one": the GPU port complete on all fronts. The bar was met and the
field renamed the milestone — **1.30.0 is that release.** Every feature
either runs on the GPU (proven on real hardware against the CPU frame),
refuses by name (the frame still renders correctly on the CPU and the
console says why), or is impossible by construction and documented as
such.

The proof is the FEATURE × DEVICE MATRIX in the self test — 116 rows,
every feature the engine has, rendered by both devices and diffed. The
field verdict on the release hardware (RTX 5060 Ti, Vulkan):

    116 rows: 114 raster+shade on the driver and matched,
    2 partially routed to the CPU by name and matched, 0 FAILED
    every feature works with the GPU device: the driver reproduces it,
    or the switch routes it honestly

The two routed rows are Painter's algorithm — ordered submission fill,
kept on the CPU by design and named in the console every time.

Numbers throughout are from the field hardware unless marked headless.

## The port, complete (what the matrix stands on)

- **Post stages** — display transform, lens, CRT, dither, NTSC
  composite (dot crawl stays CPU by construction — frame-dependent
  state — and routes the frame honestly when used)
- **Compute rasteriser** — IS `fill()`: 0 differing pixels; the named
  tie rule (equal depth → lowest triangle id) holds in all four
  engines, and 16-bit z / PS1 vertex snap / fixed-point subpixel run
  through the tie referral: the kernel marks genuinely fragile
  decisions and the CPU replays them with its own arithmetic
- **Deferred shading** — 0.000048-class whole frames: all 18
  reflectance models (CONSTANT and WIREFRAME emit the shadeless early
  return; Gouraud/flat interpolate CPU-lit corners), SUN/POINT/SPOT/
  AREA, shadow-map atlases, image textures, converted master
  materials, per-pixel surface parameters, vertex colours,
  matcap/backface, coded shader nodes (native GLSL), all pattern
  textures, generated coordinates, light linking, screen-door stipple
  (bit-equal alpha), FACE/True-Normal/Random-Per-Island, affine
  mapping with subdivision, specular slot routing, the Wireframe node
- **Normal chains** — Normal Map bends the shading normal; Bump
  renders a height pre-pass and differences it the CPU's own way;
  sin-fract Noise heights evaluate on the CPU into the pre-pass,
  exactly
- **Ray tracing, ANY depth** — hard and SOFT shadows, ambient
  occlusion, reflections, refraction with TIR, blurry (cone)
  reflections, bent rays through the chains, secondary passes for
  every material, deterministic hash sampling identical on both
  devices, the full recursion tree (mirror-in-mirror 0.000024, 0 px)
- **Radiosity** — the one-bounce gather in-shader, full-rate and
  interpolated grid
- **Every world reflects** — simple modes as baked GLSL; STARFIELD,
  BRYCE, PHYSICAL, HDRI, world graphs and the ground plane evaluated
  by the renderer itself along the reflected rays
- **Transparent layers** — sorted blend and A-buffer shading, under
  rays, with bump pre-passes, hybrid GPU/CPU routing invisible in the
  picture
- **Texture filters** — TRILINEAR, mip bias, anisotropy, N64 3-point,
  from the CPU's own mip atlases with its own footprint field
- **The viewport** — drafts and refines honor the device switch
  through the F12 marshal
- **The engine's honesty** — the picture does not depend on chunk
  size, thread count, band count, batch order, or device; every
  refusal is by name with the reason in the console; the settings
  audit holds every slider to a proof (`test_every_setting_does_what_
  it_says`)

## Impossible by construction (documented, not planned)

- Error diffusion dithers on the GPU — sequential by definition; the
  wavefront helps the CPU (and since 1.25.106 High Color diffuses per
  channel), but no GPU formulation keeps the result
- A-buffer fragment COLLECTION — an unbounded per-pixel fragment list
  is a different algorithm, not a port; the shading of those fragments
  is on the GPU, the list itself stays
- Dot crawl (NTSC) — frame-dependent state
- Painter's algorithm on the GPU — ordered submission fill; routed by
  name

## Beyond 1.30.0 (the next roads)

Named refusals that could someday lift (each prints and shades on the
CPU exactly today):

- Vertex-rate materials behind glass
- TRILINEAR footprints in LAYER passes
- Per-pixel opacity under stipple
- Ortho Backfacing (needs a mixed-winding ortho fixture first)
- Light linking past 64 objects
- Rich worlds behind NON-ray layers

New machinery:

- Region/border render with exact pixel-identity
- Hybrid opaque frame (split one frame's materials across devices)
- QTVR strip panorama camera
- DepthCue node emitter
- In-graph Bevel
- Gabor noise and Sky Texture GLSL emitters (CPU evaluators exist;
  they refuse per material today)
- Cast-filtered shadows
- Baking

Housekeeping:

- A tagged GitHub release (the repo has none; the README install link
  points at an empty page)
