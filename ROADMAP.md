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

**1.31–1.52 — the look rounds and the public release.** The halo energy
kit, the gradient family, weather, the terrain generator, the
image-as-palette road, the Bryce 1995 `.mat` decode (all 46 library presets
from the stored numbers), the lit infinite grounds, the metric-honesty
rounds, and the first tagged release with its documentation rewritten
against live counts.

**1.53–1.62 — the Blender Internal arc and the Anime Shader.** Any
2.79-or-earlier .blend appends with its Blender Internal materials rebuilt
one to one (the BI Material node, the ported BI texture engine, lamps at
their 2.79 energies, the old world), a plain File ▸ Append fixing its lamps
automatically; the Anime Shader with its game-texture compatibility modes;
116 resolution presets.

**1.63–1.69 — the stylized road.** Per-material ink and marked edges, the
Shadow Ramp road, real volumes (eight scattering models, mesh containers,
voxels, smoke grids), round area lamps, the Add-menu shelves.

**1.70–1.73 — the cel arc.** The ink style pack, the Cartoon Shader (paint,
not light, six eras), the 80s additions to the Anime Shader, and the era
looks: film stocks, weave, dust, grain, flicker, holds on twos and threes.

**1.74–1.80 — the 40s arc,** in the order chosen from the Imitation Study:
the drawn line, the 2D media, the painted backdrop and the painted
background road, the inker's line, media 2, the colour process (the
three-strip camera, the timed print, impure dyes, registration, the silver
key), and the print's wear.

**1.81–1.84 — the cel's light, the Xrd study, the preset expansion, the
hair pass.** Camera and world keys, screen shadows, the depth rim; Arc
System Works' line control and the SDF face shadow; the Anime Style menu
and fourteen Cel & Film presets; the hair shine's shapes; 140 resolution
presets; a written tooltip on every option, suite-enforced.

**1.85–1.86 — the Max study.** 3ds Max's material and map library under a
3DS Max submenu; then, from the field's own Max 2012 shader library, every
map rebuilt on Max's own algorithm on Halcyon's own tables (no Autodesk code
or data ships), the eight Standard shaders as master models, the Standard
and Raytrace materials as nodes 1:1 with their rollouts, the last maps, the
object frame on the GPU, and a 3ds Max 2012 render preset. 1.86.1 fixed the
icon that stopped 1.86.0 enabling, and taught the suite Blender's icon enum.

## Named refusals that could someday lift

Each prints its reason and shades on the CPU correctly today:

- Vertex-rate materials behind glass
- TRILINEAR footprints in LAYER passes
- Per-pixel opacity under stipple
- Ortho Backfacing (needs a mixed-winding ortho fixture first)
- Light linking past 64 objects
- Rich worlds behind NON-ray layers
- The Light Meter, Falloff's Shadow/Light and Shader to RGB (they read the
  lamp loop at shade time; the GPU's tone is a Facing or a Layer Weight)
- The Blur node's re-evaluation (it re-runs the upstream chain at shifted
  points)
- DepthCue per-material fog and Vector Transform's camera half (want
  per-frame uniforms at material rate)
- Gabor noise and Sky Texture GLSL emitters (CPU evaluators exist)
- Named colour layers (the G-buffer carries the active one)
- A linked percentage on the 3ds Max material nodes (the evaluator converts
  the unit; the raw chain earns no per-pixel grant)

## Impossible by construction (documented, not planned)

- Error diffusion on the GPU — sequential by definition
- A-buffer fragment *collection* on the GPU — the shading of those fragments
  is already there; the unbounded per-pixel list is a different algorithm
- Dot crawl (NTSC) — frame-dependent state
- Painter's algorithm on the GPU — ordered submission fill; routed by name
- The film stages on the GPU — they run once on the CPU on both device
  roads, so the two roads see the same numbers by construction

## New machinery under consideration

- Generalising the Shadow Ramp road past the Anime Shader
- A GPU march for the cel field, if the viewport's refine asks for it
- The Imitation Study's unbuilt ranks: stroke coherence across frames, tonal
  art map scale nesting, the watercolour law as its own media node, the
  successive-exposure false-colour paint palette
- 3ds Max leftovers: Matte/Shadow, Raytrace's extended parameters, the maps
  still without a node (Bitmap, Combustion, Particle Age, Flat Mirror,
  Reflect/Refract, Thin Wall Refraction, Camera Map Per Pixel)
- Region/border render with exact pixel-identity
- Hybrid opaque frame (split one frame's materials across devices)
- QTVR strip panorama camera
- In-graph Bevel
- Cast-filtered shadows
- Baking to texture
- More of the Bryce library's texture components mapped one-to-one as the
  decoder's understanding of the component blocks deepens
