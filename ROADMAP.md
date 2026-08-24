# Halcyon — roadmap

## Where the road has been

**1.30.0 — the GPU port, complete.** Every feature either runs on the GPU
(proven on real hardware against the CPU frame), refuses by name (the frame
still renders correctly on the CPU and the console says why), or is impossible
by construction and documented as such. The proof is the FEATURE × DEVICE
MATRIX in the self test — 116 rows, rendered by both devices and diffed. The
field verdict on the release hardware (RTX 5060 Ti, Vulkan):

    116 rows: 114 raster+shade on the driver and matched,
    2 partially routed to the CPU by name and matched, 0 FAILED

The two routed rows are Painter's algorithm — ordered submission fill, kept on
the CPU by design and named in the console every time.

**1.31–1.44 — the look rounds.** The halo energy kit (images, noise, bolts,
rays, shockwaves, colour ramps, halos on curves, lens flares off the real
z-buffer), the gradient family, live-updating ramps, and a long run of field
fixes.

**1.45–1.47 — weather, terrain, palettes.** Rain/snow/embers/ash as a
deterministic screen-space particle field; the terrain generator's seven
landforms; the Altitude & Slope noise socket; the Iridescent node; the
image-as-palette road and the palette-table operator; the master shader's
socket regroup; animation ETA that understands warm-up; 20 Bryce-shaped ocean
presets.

**1.48–1.49 — the Bryce decode.** The MetaTools `.mat` container cracked end
to end — RLE preview thumbnails, the 976-byte material channel records
(layout pinned against the 1995 manual's own text), texture colour tables,
per-record endianness for the 68k-saved files — and all 46 library presets
rebuilt from the stored numbers. The Pre-Made shelf (74 templates, ten
families) moved into the shader editor's Add menu. The infinite grounds
gained real scene lighting with cast shadows, per-tone dials, and a Material
mode. The hard alpha test no longer deletes sub-half-opacity surfaces in
blended transparency.

**1.50–1.51 — the metric-honesty rounds.** Camera object scale (the .3DS/FBX
`$$$DUMMY` rigs) is stripped exactly as Blender's own renderers do, so fog,
clip planes and depth of field read true distances; the console names a
fog-swallowed frame with numbers instead of leaving a beige rectangle; the
ColorRamp/Float Curve/RGB Curves family emits GLSL from its baked LUTs, so
graded terrains stay on the GPU.

**1.52.0 — the public release.** Documentation rewritten against live counts;
the tagged release the repo needed.

## Named refusals that could someday lift

Each prints its reason and shades on the CPU correctly today:

- Vertex-rate materials behind glass
- TRILINEAR footprints in LAYER passes
- Per-pixel opacity under stipple
- Ortho Backfacing (needs a mixed-winding ortho fixture first)
- Light linking past 64 objects
- Rich worlds behind NON-ray layers
- The Blur node's re-evaluation (CPU by construction — it re-runs the
  upstream chain at shifted points)
- DepthCue per-material fog (wants the frame's depth at material rate)
- Gabor noise and Sky Texture GLSL emitters (CPU evaluators exist)

## Impossible by construction (documented, not planned)

- Error diffusion on the GPU — sequential by definition
- A-buffer fragment *collection* on the GPU — the shading of those fragments
  is already there; the unbounded per-pixel list is a different algorithm
- Dot crawl (NTSC) — frame-dependent state
- Painter's algorithm on the GPU — ordered submission fill; routed by name

## New machinery under consideration

- Region/border render with exact pixel-identity
- Hybrid opaque frame (split one frame's materials across devices)
- QTVR strip panorama camera
- In-graph Bevel
- Cast-filtered shadows
- Baking to texture
- More of the Bryce library's texture components mapped one-to-one as the
  decoder's understanding of the component blocks deepens
