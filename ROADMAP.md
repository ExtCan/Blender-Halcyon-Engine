# Halcyon — the road to 1.26.0

1.26.0 is "the big one": the GPU port complete on all fronts. This file
is the living map of the distance left. Every item below either runs on
the GPU (proven on real hardware against the CPU frame), refuses by name
(the frame still renders correctly on the CPU and the console says why),
or is impossible by construction and documented as such. The engine is
coherent at every point in between — completeness is scope, not repair.

Updated with each release. Numbers are from the field hardware
(RTX 5060 Ti, Vulkan) unless marked headless.

## Proven on hardware (done)

- **Post stages** — display transform, lens, CRT, dither, NTSC composite
- **Compute rasteriser** — IS `fill()`: 0 differing pixels, both sizes
- **Deferred shading** — 0.000048-class whole frames: all shipped
  reflectance models, SUN/POINT/SPOT/AREA, shadow-map atlases, image
  textures, converted master materials, per-pixel surface parameters,
  vertex colours, matcap/backface, coded shader nodes (native GLSL),
  all 19 integer-hash pattern textures, generated coordinates
- **Normal chains** — Normal Map bends the shading normal; the Bump node
  renders a height pre-pass and differences it the CPU's own way;
  sin-fract Noise heights evaluate on the CPU into the pre-pass, exactly
- **Ray tracing, ANY depth — the arc is COMPLETE on hardware** — hard
  ray shadows, SOFT ray shadows, ambient occlusion, reflections,
  refraction with TIR, bent rays through the chains, secondary passes
  for every material (visible or not), deterministic hash sampling
  identical on both devices, and the full recursion tree at ray
  depth > 1: mirror-in-mirror measured at 0.000024, 0 px of 172800 on
  the field driver
- **Sky/env in reflections** — SOLID, BLEND, GRADIENT, BANDS, EQUIRECT,
  MIRRORBALL
- **The engine's own honesty** — the picture no longer depends on chunk
  size, thread count, or batch order, on either device

## Remaining for 1.26.0

Ray arc: **DONE** — confirmed on hardware in the 1.25.45 report.

Skies and worlds (env term in reflections): **DONE — confirmed on
hardware** (the Bryce sky lab in a mirror: 0.000048, 0 px of 172800 on
the field driver). One mechanism took the whole column: rich worlds
(STARFIELD, BRYCE, PHYSICAL, HDRI, world graphs, the ground plane) are
evaluated by the renderer itself along the reflected rays and
composited after readback — exact for any world by construction.

Texture filters and transparency:

- [x] Transparent-layer shading (Sorted Blend / A-Buffer) — shipped in
      1.25.50-52: each depth layer's fragments shade as a full-screen
      deferred pass with the REAL alpha chain emitted, merged under
      the proven blend state, gathered per fragment. Field-verified at
      the fragment level in the 1.25.51 report: 0 of 76573 flips, max
      0.000048 on the driver. The whole-frame residual traced to the
      A-buffer depth tie at modeled contacts and fixed in 1.25.52
      (raster.abuf_depth_limit — the picture no longer depends on
      which rasteriser rounded the opaque depth's last ULP).
- [x] Transparent layers under ray tracing — DONE in 1.25.54: each
      rank's fragments are a virtual PRIMARY surface and `_run_sweeps`
      walks the full recursion from it — refraction through the glass,
      reflections, any depth, soft shadows and AO sampled from the
      fragment's own pixel identity, rich worlds at the final depth via
      the '__env' composite. Headless proof per fragment against the
      compositor's own CPU call, all exact: 0 of 7303 on the field's
      all-transparent + ray shape. CONFIRMED on hardware in the
      1.25.54 report: 0.000012, 0 px of 172800.
- [x] Bump pre-passes on transparent layers — DONE in 1.25.55: height
      pre-passes draw per rank over each layer's own ids (sin-fract
      chains CPU-evaluated over the rank's virtual surface), and the
      CPU now shades the A-buffer rank by rank with per-rank gradient
      fields — fixing a pre-existing chunk-dependence in transparent
      bump shading that the port surfaced. The Water anatomy as
      glass: 0 of 678 headless, field confirmation pending.
      Remaining named refusal: rich worlds behind NON-ray layers.
- [ ] TRILINEAR / N64 three-point (need a mip footprint)
- [ ] STIPPLE alpha
- [ ] Affine barycentric carry (`tex_perspective` off)

Textures:

- [ ] Gabor noise
- [ ] TexSky node

Shadows:

- [ ] Cast-filtered shadows

CPU-side polish (not GPU, still 1.26-adjacent):

- [x] Worker-band bump/gradient seams — DONE in 1.25.48: bands
      rasterise one context row and build gradient fields from full
      coverage; two bands stitch to the exact whole frame (0.0). The
      picture no longer depends on ANY internal scheduling — chunks,
      threads, or bands.
- [ ] Viewport GPU mode (design question — the preview is CPU by design)
- [x] F12 UI freeze — DONE in 1.25.53: `bl_use_gpu_context` is off by
      default; the render thread runs with no GPU context and every
      driver burst (compile, upload, draw, read back) marshals to the
      main thread through gpu/marshal.py, so the interface breathes
      between bursts. Background renders hold the context automatically
      (nothing to freeze, and `-b` may pump no timers); the Debug
      panel's "Hold GPU Context" restores the old mode; every
      marshalling failure falls back to the CPU frame with the reason
      printed. Field confirmation pending.

## Impossible by construction (documented, not planned)

- Error diffusion dithers (Floyd–Steinberg family) — sequential by
  definition; the wavefront helps the CPU, no GPU formulation keeps
  the result
- A-buffer fragment COLLECTION — an unbounded per-pixel fragment list
  is a different algorithm, not a port; the rasterise, depth sort,
  rank and farthest-first composite stay CPU. (The SHADING of those
  fragments deferred to the GPU in 1.25.50 — the list itself is what
  stays.)
- Dot crawl (NTSC) — frame-dependent state

## Beyond the GPU port

- Baking, motion blur, subpixel precision / affine subdivision options
- A tagged GitHub release
