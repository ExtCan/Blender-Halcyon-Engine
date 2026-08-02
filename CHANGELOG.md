# Changelog

All notable changes to Halcyon are recorded here. Dates are ISO 8601.

---

## [1.25.66] — 2026-08-02

You're right, and I should have shipped this last round instead of
describing it. Two fixes, both measured.

### Fixed — a shading RATE interpolates the light, not the texture

Halcyon evaluated the **whole material** at each vertex — texture
sampling included — and interpolated the result across the triangle.
On a 1209-triangle character that means three texture samples spread
over a face, which is the smear the field photographed. Measured on
the demo scene's textured material: mean horizontal contrast **0.0092
against per-pixel shading's 0.0692 — a 7.5x blur**.

Hardware of the period did not do that. It interpolated the vertex
**colour** and sampled the **texture** at every pixel, then multiplied:
MODULATE, the fixed-function combiner. (Before OpenGL 1.2 added
`GL_SEPARATE_SPECULAR_COLOR`, that combined vertex colour was the
entire lighting result — which is why this multiplies all of it by the
texel rather than holding specular out. That is the 1990s default, not
a simplification.)

So the vertex and face rates now shade the lighting over a **white**
surface, interpolate that, and multiply by albedo and alpha sampled at
the **pixel** rate. Alpha comes from the pixel pass on purpose: a
cut-out texture's edge is the one thing that must never be
interpolated between vertices.

Detail restored to **0.0587** — 85% of per-pixel sharpness, with the
banding intact. And the change is a fix rather than a new look: an
untextured diffuse material renders **bit-identically** to before
(max difference 0.000000), because factoring one flat colour out and
multiplying it back is an identity.

### Fixed — the depth line understands Painter's algorithm

The field's console read:

    ndc z 4.467430..5.482941; every covered pixel sits at this
    projection's depth asymptote, where NO distance can be recovered

Those are not ndc values — they are **view distances**, 4.5 to 5.5
units. Painter's algorithm gives every fragment of a polygon that
polygon's single depth, so the buffer holds distances, and reading
them as ndc produced an alarming diagnosis of a scene that was sitting
normally in front of the camera. The line now detects the mode and
says what it actually means: one depth per polygon, no per-pixel
z-buffer, the bit-depth setting does not apply, and interpenetrating
polygons meeting along an edge is the algorithm rather than a fault —
with Depth Method → Z-Buffer named as the per-pixel alternative.

That also re-frames the whole thread honestly: with Painter's
selected, *some* of what looked like broken depth was the period
algorithm doing exactly what it does. What was genuinely broken —
every material mis-classified see-through (1.25.64) — is fixed, and
now the console distinguishes the two.

### Verification

- The blur is measured, and the OLD path is reproduced inline as a
  negative control so the test can tell a fix from a coincidence.
- Untextured diffuse: bit-identical to the previous behaviour.
- The rates still differ from each other and from per-pixel shading —
  the banding a shading rate exists for is intact.
- Painter's mode is reported as Painter's, with no asymptote claim.
- Full renderer and shader suites green.

---

## [1.25.65] — 2026-08-02

The 1.25.64 console is a clean bill of health on the thing that was
actually broken: **the transparency lines are gone entirely.** No
mis-flagged materials, nothing pushed out of the depth-buffered pass,
no A-buffer stage in the breakdown. Sonic is z-buffered, with culling
and depth writes, for the first time in this whole thread.

What the field saw instead is a *different* pass taking over, and one
line in that console names it:

    [Halcyon GPU] shading on the CPU: the scene shading rate is VERTEX

### Why the picture changed character

`shading_rate` is a **rate**, not a model: the same shading maths
evaluated at vertices and interpolated across the triangle — Gouraud,
which most of the console and home-computer presets select on purpose.
The frames before 1.25.64 never applied it, because every triangle was
in the transparent pass, and the A-buffer path shades per fragment
without consulting the rate at all. So the mis-classification had been
silently overriding the preset's own shading rate, and fixing it
handed the frame back to Gouraud.

That is the setting doing its job. The immediate dial is **Shading
Rate → Pixel** (Render properties), which gives per-pixel shading with
the depth now correct — the sharpness of the old frames, without the
tearing.

It also exposes a genuine fidelity gap worth naming plainly: Halcyon
evaluates the *whole* material per vertex, textures included, so a
textured model smears. Period hardware interpolated the **lighting**
per vertex and still sampled the **texture** per pixel — `texel ×
gouraud colour` — which is why a Gouraud-shaded PlayStation model has
soft banded light over a sharp texture. Making that split properly is
its own piece of work (the albedo has to come out of the shading
chain, and specular does not multiply by it); it is not shipped here
rather than shipped as a guess.

### Fixed — the depth line no longer invents a number

1.25.63's instrument reported:

    the subject sits at 2e+14..2e+14, where the buffer resolves
    9.32e+19..9.32e+19 world units

which is nonsense, and mine. ndc z reaches 1 exactly *at* the far
plane and approaches `(f+n)/(f-n)` — an asymptote — as distance runs
to infinity. The first version divided by that vanishing denominator
with a `1e-12` clamp and printed **the clamp** as a measurement. The
line now:

- prints the raw `ndc z` range every time, so the input is visible;
- names pixels **past the far clip** as their own condition (drawn
  anyway — period renderers did — but outside the range the buffer was
  set up for), while still measuring the rest;
- refuses outright when depths sit at the asymptote, and says the
  geometry is effectively infinitely far for this clip range rather
  than making a number up.

An instrument that lies is worse than no instrument. This one now says
"I cannot measure this, and here is why".

### Verification

- New checks: asymptote depths are refused and never printed as a
  figure; past-the-far-clip pixels are named while the rest are still
  measured; the raw ndc range appears in every variant.
- Full renderer and shader suites green.

---

## [1.25.64] — 2026-08-02

Caught, by the line 1.25.63 added:

    [Halcyon] transparency: 25 of 25 materials are see-through
    (1209 of 1209 triangles) -- s_kihon10_sonic.nja.sa1mdl_0: flagged
    see-through on export (blend mode or an alpha socket) ...
    [Halcyon] transparency: NOTHING is in the depth-buffered pass

Not a depth bug. A **classification** bug, and mine.

### Fixed — a blend mode is not evidence of alpha

`_tree_has_alpha` returned True for any material whose Blender
`blend_method` was anything other than OPAQUE. Importers routinely set
a non-opaque blend mode on every material they create, and this model
arrived from a game format with all 25 of its carrying one. So all
1209 triangles were classified see-through, the depth-buffered pass
received **nothing**, and a solid character was rasterised with
**culling off and no depth writes** — hidden-surface removal never
happened at all. Its fragments were then composited as stacked layers,
and under Sorted Blend that is polygon-centroid ordering, whose
signature failure is exactly what the field kept photographing: wedges
of one surface punching through another, back faces showing through
the front, holes where the layers ran out.

Three rounds went looking for that in the z-buffer. It was never in
the z-buffer.

Alpha must now be **used**, not merely permitted. A material is
see-through when it has an opacity below one, a
transparent/glass/refraction/holdout node, or an Alpha socket that is
linked or set below one — and it is opaque whatever its blend mode
says. Nothing is lost: a material with no alpha anywhere would have
blended with alpha 1.0, which is what the z-buffer does for free and
correctly. `blend_method` is now deliberately not consulted in either
direction, because Halcyon's own alpha lives on the master shader's
sockets independently of EEVEE's blend mode — reading the mode would
misclassify the add-on's own Glass preset, which is the bug this
check was added to fix in 1.25.50.

The console line also names the specific evidence now — `Opacity
0.120`, `its Opacity socket is linked`, `a Transparent BSDF node in
its tree` — instead of the vague "flagged see-through on export" that
let a whole mis-flagged character hide behind it.

### What this should do to the picture

Sonic's 25 materials go back through the z-buffer: culling on, depth
writes on, per-pixel hidden-surface removal. The tearing through his
face should be gone, not reduced. As a bonus the transparency stage
disappears from that frame entirely — it was doing A-buffer work for a
model that never needed a single layer.

If some materials are still listed as see-through after this, the line
now says which evidence found them, and that evidence is real alpha.

### Verification

- New test: every blend mode — CLIP, HASHED, BLEND, DITHERED, BLENDED
  — leaves a solid material in the depth-buffered pass, while every
  genuine kind of alpha (opacity, master Opacity, Edge Opacity, a
  linked Opacity socket, a Transparent BSDF anywhere in the tree)
  still routes it to the A-buffer with its reason named. End to end,
  a fully solid model puts every triangle in the depth-buffered pass.
- The 1.25.50 regression this check originally fixed stays fixed: the
  master shader's Opacity and Edge Opacity sockets are still read, and
  the existing glass tests still pass.
- Full renderer and shader suites green.

---

## [1.25.63] — 2026-08-02

The close-up says 1.25.62 did not fix the picture, and that is
consistent with what 1.25.62 actually was: at 24 bits its grid step is
6e-8, so an invisible-by-arithmetic fix cannot be the one the field is
looking at. It was a real bug and it stays fixed; it was not **this**
bug. Rather than guess a fourth time from a JPEG, this release makes
the frame state its own depth situation, in numbers, on every render.

### Added — the frame reports what its z-buffer can resolve

    [Halcyon] depth: 16-bit z-buffer, clip 0.1..200; the subject sits
    at 5.83..7.11, where the buffer resolves 0.00519..0.00772 world
    units -- surfaces closer together than that cannot be told apart

Near and far come back out of the projection matrix, the frame's own
covered depths say where the subject sits, and the N-bit grid step
converts to the smallest world separation two surfaces can have and
still be told apart *there*. Perspective depth is hyperbolic —
resolution falls with the SQUARE of distance — so a near plane set
very close spends the whole buffer on empty air in front of the
subject. That is the classic cause of close-fitting surfaces tearing
through each other at a bit depth that sounds generous, and it is now
a printed number instead of a suspicion. The claim is falsifiable and
the test falsifies it: surfaces a quarter of the reported figure apart
quantize to the same value, surfaces four times it apart do not.

### Added — the frame names which surfaces left the depth-buffered pass

    [Halcyon] transparency: 3 of 5 materials are see-through (1184 of
    1620 triangles) -- Skin: flagged see-through on export (blend mode
    or an alpha socket); Eyes: Opacity 0.980

A material in the transparent pass is rasterised with **culling off
and no depth write**, and its fragments stack as A-buffer layers. For
glass that is the whole point. For a solid character whose materials
were merely *flagged* — an imported model often arrives with every
material set to Alpha Clip or Alpha Blend, or with an alpha socket
linked — it means the surface no longer participates in
hidden-surface removal at all, so it can show its own back faces and
its interior geometry through itself. That reads exactly like broken
depth, and until now nothing said it was happening. If every material
is flagged, the line says so outright and names the setting that puts
the frame back through the z-buffer.

### Added — the layer cap says when it drops fragments

    [Halcyon] transparency: the 16-layer cap dropped 41822 fragments
    at 9310 pixels -- those layers are not drawn, and where nothing
    opaque sits behind them the background shows through. Raise Max
    Transparent Layers to draw them

`max_transparent_layers` truncated silently. Where the dropped layers
were the only geometry at a pixel — which is precisely the case when
a solid object went entirely into the transparent pass — the
**background** showed through instead: black patches in the middle of
solid-looking geometry. The field frame printing exactly "16 layers"
was the cap being hit, not a coincidence of scene depth.

### Verification

- New test: the reported resolution is checked against the
  quantization it predicts (a quarter of it indistinguishable, four
  times it distinguishable); near/far recovered from the projection
  matrix match the camera's own; the classification record names
  reasons for opacity-driven and export-flagged materials alike, and
  records nothing for an all-opaque scene.
- Full renderer and shader suites green.

### What to try first, before any more code

One render settles it. If the character's materials are meant to be
solid, set **Transparency → None** (Render properties) and render the
same frame: the whole model goes back through the z-buffer, culling
on, depth writes on. If the tearing vanishes, the mechanism was the
transparent pass, not depth precision — and the fix is the materials'
blend mode, not the renderer. If it survives, the depth line above
tells us what the buffer could resolve, and that is the next round's
starting number.

---

## [1.25.62] — 2026-08-01

Two field pictures named the same bug: solid wedges punching through
Sonic's face, and bad depth where the grass meets the stone. Both are
what a low-bit z-buffer looks like when the quantization is applied in
the WRONG PLACE.

### Fixed — depth quantizes per PIXEL, not per vertex

`build_screen_tris` rounded the **vertex** z to the N-bit grid and the
fillers then interpolated between rounded corners. That tilts every
triangle's whole depth plane by up to half a step — and between two
independently-triangulated close surfaces (a muzzle over a head, eye
sockets around eyes, grass against stone) the tilted planes CROSS, so
one surface stomps through the other in big solid wedges. Measured on
a slanted quad at 16 bits: stored depths landed up to a **full step
off the grid** (1.51e-5, one whole 16-bit step).

Real N-bit hardware interpolated depth at full precision and rounded
when the value met the buffer. Both rasterisers now do exactly that:

- `fill` and `fill_batched` take `depth_bits` and round the
  interpolated `zz` per pixel (`quantize_depth`, float32 throughout);
  `build_screen_tris` no longer touches z.
- The compute kernel applies the identical formula before its depth
  compare — `roundEven`, which is NumPy's own half-to-even, so both
  rasterisers round the same half-cases the same way. The front-end
  mirror (`simulate_raster`) carries the same `depth_bits`.

What changes on screen: at the default 24 bits, nothing visible (the
grid step is 6e-8). At low bit depths — the period look the setting
exists for — every stored depth sits ON the grid, so two surfaces
fight only where they are genuinely within one step of each other:
thin dithered bands at the true crossing, the authentic artifact,
instead of view-dependent wedges. Exactly coplanar contacts quantize
to the SAME value on both rasterisers and resolve stably by
submission order, and the A-buffer's tolerant keep still holds
(equal quantized depths are within the limit by definition).

### Verification

- New test: on a slanted 16-bit quad, every stored depth lies exactly
  on the N-bit grid (max off-grid 0.0); the OLD vertex-quantized
  behaviour is reproduced as a negative control and is provably off
  the grid (1.51e-5); `fill` and `fill_batched` quantize identically;
  the kernel's front-end mirror picks the same winners and stores the
  same quantized depths at 16 bits, bit for bit; quantized coplanar
  contacts stay within the A-buffer keep limit.
- Full renderer and shader suites green — including the raster-pair,
  ULP-invariance, painter's and A-buffer seams, all of which ride the
  changed fillers at their scenes' own bit depths.

If the face still shows artifacts after this on YOUR scene, check the
Z-Buffer Bits value (Debug panel): at 16 bits a close-up face with a
distant far plane is genuinely at the edge of what period hardware
could hold apart — that part is authentic. This fix removes the part
that wasn't.

---

## [1.25.61] — 2026-08-01

The 1.25.60 field frame did what it was built to do: **it printed the
frame's own depth distribution** — `per-layer frags 2.7M 1.6M 1.2M …
15.8k 12.5k 274` — and the honest accounting put a name on the missing
seconds: `reads+sync 4.9s` of an 11.0 s stage, with `draws 0.6s`. Two
verdicts fall straight out of those numbers, and one apology rides
along.

First: the 0.25 % routing default was set ~20× below THIS frame's
break-even, so routing moved exactly one layer of 274 fragments — the
machinery proven, the dial wrong. Measured properly from the split:
~0.4 s of fixed driver cost per layer against ~3.7 µs per fragment on
the CPU (34.8 s for 9.4 M fragments on the 1.25.56 frame) puts the
break-even near **3 %** of the frame, not 0.25 %.

Second: `reads 4.9s` is not mostly copying. A draw call *submits*;
the readback is where the driver **synchronises** — the queued passes
execute inside that wait. So the old `draws 2.0s` was submission plus
sync lumped together, and the new split separates them. The bucket is
now printed as **`reads+sync`** so nobody (the author included) reads
it as pure transfer again.

The apology: the new HYBRID LAYER ROUTING self-test section reported
`NOT A MIX` — vacuous — because the bumpy-glass frame's two layers are
the SAME glass ball's front and back faces, **exactly equal in size**
(8542 and 8542), and no threshold fits between equals. The section
said so honestly instead of passing silently, which is the one thing
it did right. Fixed below.

### Changed — the threshold is now the measured one

`layer_gpu_min_frac` default 0.0025 → **0.02**: just under the ~3 %
measured break-even, margin included. On the 1.25.60 field
distribution this routes the 50.2k / 15.8k / 12.5k / 274 layers — four
of sixteen — each a near-certain win.

### Added — targets live for the loop

The field frame allocated and freed a ~50 MB `GPUOffScreen` **per
layer**, plus one per sweep level and one per height pre-pass — driver
allocations at that size cost milliseconds each, sixteen layers deep.
One layer target, one sweep target, and one pre-pass target per
(material, height-node) now live for the whole loop. Semantically
identical by construction: every pass full-clears before its first
draw, and every rank draws at least one pass (coverage guaranteed it),
so a reused target can never show a stale layer. The pre-pass texture
handles are fetched once, so the marshal crosses less too.

### Added — scissored layer passes

Each depth layer's passes and readbacks are clipped to the layer's own
bounding box (`gpu.state.scissor_set` on the draw, `read_color` on the
region). The clear stays full-frame — every texel defined, so a
neighbour fetch outside the scissor reads a clean zero and is masked
by the ids test exactly as the CPU masks it. A layer whose fragments
sit in a corner stops paying for the whole frame's rasterisation, sync
and transfer. The split prints the aggregate — `scissor 41% of full
frames` — so the field says how much area the boxes actually saved.

Because scissored *reads* are a newer driver path than the proven
full-frame texture read, two guards ship with it: a Debug toggle
(**Scissor Layer Passes**, on by default) that reverts to the proven
path, and a new self-test section — SCISSORED LAYERS — that renders
the same frame both ways and demands **bit-identical** output on your
driver, printing the verdict either way.

### Changed — the per-rank NumPy pays less

The ids scatter is two fused writes instead of four, the reset one
write instead of two — the `other` bucket on a 9.4 M-fragment frame is
substantially these fancy-index passes.

### Fixed — the hybrid self test can actually mix

The HYBRID LAYER ROUTING section now makes the mirror see-through
too, so the frame holds two transparent materials with different
footprints — layers of *distinct* sizes — and picks its threshold
between the two largest distinct sizes. It renders its own CPU and
all-GPU references on that scene and proves the mixed picture against
both, as designed. (The exact-merge and maths-level hybrid tests in
the headless suite already split correctly — their splitter
deduplicated sizes; the section's didn't. It does now.)

### On the +5.5 s

Total 15.3 s → 20.9 s between 1.25.59 and 1.25.60 on essentially
identical submitted GPU work (the .60 code changes were CPU-side and
strictly smaller: same draws, same reads, one routed layer fewer) —
and uploads dropped 1611 → 1510 MB, which says the scene itself was
not byte-identical between the two pastes either. Run-to-run driver
variance and scene drift both live in that delta; the split's
*internal shares* are the numbers to steer by, not one run's total.
If 1.25.61 lands and the total still sits high while `scissor %` is
large and `reads+sync` barely moves, the next lever is architectural
(compacting the readbacks to fragment lists instead of frames) and
the split will have said so.

### Verification

- Both suites green from the shipped zip (routing exact-merge and
  hybrid tests unchanged and passing — they pin their own thresholds).
- SCISSORED LAYERS: bit-identity demanded on the driver, verdict
  printed.
- HYBRID LAYER ROUTING: forced mix on the new two-glass frame, proven
  against both pure runs, bucket-sum soundness line kept.
- The three GPU layer sections still pin routing off and now run WITH
  the scissor — a region bug cannot hide from their 0 px demands.

---

## [1.25.60] — 2026-08-01

The 1.25.59 field split falsified 1.25.59's own bet, and it deserves
saying plainly: the absent-material skip and the buffer reuse **bought
nothing** — `draws 2.0s` (was 1.9), stage 7.386s (was 7.076), total
15.299s (was 15.395). The crossings dropped 741 → 662, so the skip
*fired*; it just skipped passes that were never the cost. The scene's
see-through materials appear in nearly every rank, and the real bill
is structural: the driver pays **full-frame fixed costs per depth
layer** — a draw and a readback cover every pixel whether the layer
holds three fragments or a million — sixteen times, plus ~1.8 s of
worker-side NumPy the split didn't even have a bucket for. Wrong
theory, correctly measured; this round changes the structure instead.

### Added — rank routing: each layer shades where it is cheap

The CPU pays **per fragment**; the driver pays **per layer**. So the
compositor now routes each depth layer to whichever path is cheaper
for it: layers whose fragment count falls below
`layer_gpu_min_frac` of the frame's pixels (default **0.25 %** —
deliberately far below the ~1 % break-even the field numbers suggest,
so every routed layer is a near-certain win and a mistaken route
costs milliseconds) shade on the proven per-rank CPU path; the dense
layers stay on the driver. Three properties hold by construction:

- **Whole layers only.** A rank is never split across paths, so both
  sides build their per-rank Bump gradient fields from complete
  layers — each fragment's colour is exactly what the pure run would
  have given it. The new exact-merge test stubs the driver with the
  CPU's own colours and demands the routed picture equal the pure CPU
  picture **bit for bit**; it does.
- **A route is not a refusal.** A frame whose every layer sits below
  the break-even prints `layer routing: all N layers below the GPU
  break-even (…); shaded on the CPU (routed, not refused)` — calmly,
  and only when a driver was actually present to be skipped.
- **The field sees the decision.** Every GPU layer frame now prints a
  routing line with per-layer fragment counts:
  `layer routing: GPU 6 layers (4.8M), CPU 10 layers (81k, 0.4s);
  per-layer frags 1.9M 1.2M …` — the frame's own depth distribution,
  so the next threshold argument is measured, not guessed.

Coverage even *relaxes* correctly: a material that refuses GLSL but
appears only in routed layers no longer blocks the dense layers from
the driver, because coverage is checked against the fragments the
driver is actually asked to shade.

### Added — the split accounts for every millisecond

The 1.25.59 split summed to ~5.6 s of a 7.386 s stage; the missing
1.8 s had no name, and a remainder nobody prints is a cost nobody
attacks. The buckets are now **disjoint** and the line adds three:

    transparent split: plan 0.1s, compile 0.1s, uploads 0.5s (1611 MB),
    draws 1.2s, reads 0.8s, sweeps 1.6s (of which CPU ray build 0.9s),
    other 1.8s of 7.4s; 6 of 16 layers on the GPU; marshal …

`reads` pulls the full-frame readbacks out of `draws` (they were
hiding there), `compile` names the cold-frame shader builds, sweep
time no longer double-counts the uploads and reads inside it, and
`other` is the printed remainder — the worker-side scatters, gathers
and copies — measured against a true stage total.

### Changed — one sort finds every layer

Both per-rank loops (driver and CPU) found each layer with a full
`rank == r` scan over every fragment in the frame — sixteen scans of
millions. One stable argsort plus `searchsorted` bounds now yields
each layer as a slice, bit-identical to the scans (stable sort keeps
equal ranks in original index order), the same pattern the compositor
itself has always used. The per-rank evaluator caches are also cleared
when the driver loop ends, so nothing dangles on the job.

### Verification

- New: the exact-merge routing test (stub driver answers with the
  CPU's own colours; routed picture must be bit-identical; frac 0 and
  frac 1 negative controls; partition asserted layer by layer against
  the routing record — a hybrid that quietly ran pure fails by name).
- New: the maths-level hybrid test through `simulate_fragments` — the
  mixed frame sits within the proven layer tolerance of BOTH pure
  runs, with the mix asserted non-vacuous.
- Self Test: the three GPU layer sections pin `layer_gpu_min_frac 0`
  so they keep proving the full driver machinery; a new HYBRID LAYER
  ROUTING section forces a split on the bumpy-glass frame, checks the
  mixed picture against both pure runs above it, and checks the split
  buckets sum to the stage total (`sound` / `OVERLAPPING` printed).
- Field-size slicing benchmark: slices bit-identical to the scans;
  routing partition ~15 ms on 3.2 M fragments.

What this should do to the field frame depends on its depth
distribution, which the routing line will now print. If the deep
tail is sparse (typical), several of the 16 layers leave the driver
and their full-frame fixed costs leave with them; if the tail is
dense, the counts will say so and the threshold argument moves to
measured ground either way. The `other`/`reads` buckets name the
remaining ~1.8 s regardless.

---

## [1.25.59] — 2026-08-01

The first field split answered the question precisely: `uploads 0.5s
(1611 MB)` — the bus is fine — `marshal 0.3s waiting` over 741
crossings — the marshal is fine — and **16 layers**, each paying
full-frame fixed costs however few fragments it holds. The fat is
per-rank overhead: every depth layer drew every material's pass over
the whole screen and allocated fresh 50 MB buffers, even when a deep,
sparse layer held a few thousand fragments of one material.

### Changed — the per-rank loop pays for the rank, not the frame

Three cuts, each bit-identical by construction:

- **Absent materials skip.** A layer pass writes only where its keep
  is one; a rank that contains no fragments of a material gets
  nothing from its pass, so it no longer runs — nor do its height
  pre-passes. A deep rank usually holds one material, not the scene's
  whole list; on the field frame that alone should fold most of the
  1.9 s of draws.
- **One ids buffer, one virtual surface, for the whole loop.** Each
  rank scatters its fragments in, and afterwards scatters them back
  out — a reset proportional to the rank's own fragment count instead
  of a fresh 50 MB clear per layer. The ray machinery's evaluator
  caches key on surface identity, so the reused surface gets fresh
  cache dictionaries per rank.
- The simulate mirror skips identically, so the headless seams keep
  proving the exact frame the driver draws.

Every per-fragment seam holds unchanged: plain glass, ray-traced
glass, the Water anatomy, the last-ULP invariance — all exact.

---

## [1.25.58] — 2026-08-01

The 1.25.57 field frame is the milestone this whole arc pointed at:
**no fallback line** — the scene's transparent layers shaded on the
driver, on the real frame, and 50.7 seconds became **16.0** with the
interface live throughout. Against the 33.7 s where the transparency
arc began, the frame has halved, and every named refusal along the way
was lifted rather than worked around.

### Added — the transparent split names the next second

The summary line still points at `transparency shading (GPU) 7.9s` —
on the GPU, not yet fast on it. Before optimizing blind, the stage
now prints where its seconds went, one line after every successful
GPU layer frame:

    [Halcyon GPU] transparent split: plan 0.2s, uploads 3.1s (610 MB),
    draws 0.8s, sweeps 3.2s (of which CPU ray build 1.7s), 4 layers;
    marshal 214 crossings, 0.4s waiting

The suspects it separates: per-layer ids uploads (a 1024×768 frame at
SuperSample ×4 uploads ~50 MB per depth layer, plus a ray buffer per
sweep level), the full-screen draws, the sweeps' CPU ray building,
and the marshal's own crossing latency (wall time waiting minus time
the bursts actually ran). The marshal carries the accounting
(`marshal.acct()`), so the field's next paste names the perf round
instead of guessing at it.

---

## [1.25.57] — 2026-08-01

The 1.25.56 field frame got past the Mapping gap — the layer plan
qualified for the first time on the real scene — and then the marshal
failed it: `transparency shading (GPU) 8.011s` in the breakdown is
exactly the 8-second timeout, expiring on a burst that was busy
succeeding. 43 seconds became 50: the work was discarded and the CPU
path paid on top. Two design flaws, both fixed.

### Fixed — the timeout bounds pickup, never execution

The old cap timed start-to-FINISH: a real frame's layer burst
(shader compiles, 50 MB uploads per depth layer, the ray sweeps)
legitimately needs longer than any fixed number, so the worker gave
up on work that was completing. The timeout now answers exactly one
question — is the main loop alive? — by bounding only the PICKUP. A
burst that has started runs to completion, however long it takes.
And a burst the worker abandoned is skipped if the timer fires late,
so no main loop ever blocks on work whose result was given up on.

### Changed — the marshal moved to the device boundary

The deeper flaw: the bursts were too big. A whole `shade_frame` or
layer pass crossed as ONE unit, carrying its CPU halves with it —
ray building, composites, height chains — which meant minutes-class
work could sit on the main thread, freezing the interface the
marshal exists to free. The crossings now live inside
`gpu/device.py` itself: compile, upload, draw, dispatch, read-back —
each a genuine millisecond burst — while every CPU stretch stays on
the worker with the interface untouched. Cache hits (shaders,
uploaded atlases) return on the calling thread without crossing at
all, and texture packing happens worker-side before the upload
crosses. The pump also lingers two milliseconds after each burst, so
a streaming call sequence (upload, draw, draw, read...) crosses in
one timer slice instead of paying a tick per call.

Nothing above the device layer carries marshal code any more — the
call sites that wrapped whole stages went back to plain calls, and
any new device use is automatically marshalled right by
construction. The self test is untouched (operators run on the main
thread, where every crossing short-circuits in place).

Proof: the marshal's new semantics test one by one (a picked-up
burst outliving its pickup timeout, an abandoned burst never
running), and the threaded whole-engine render still completes with
no deadlock and nothing left registered.

---

## [1.25.56] — 2026-08-01

The 1.25.55 self test was all zeros — BUMPY GLASS LAYERS at
**0.000012, 0 px**: the entire transparent-layer arc is proven on
hardware. The field frame moved to a new KIND of wall — not an
architecture refusal but an emitter gap, named by material:
`'Material.002' (as a transparent layer): no GLSL emitter for
ShaderNodeMapping`. And the frame time rose 33→43 s: the per-rank
correctness fix was re-running the waves' noise chain once per depth
layer. Three fixes.

### Added — ShaderNodeMapping emits GLSL

The 58th node type. All four vector types — POINT, TEXTURE, VECTOR,
NORMAL — with `n_mapping`'s exact arithmetic: multiply by Scale,
rotate X-then-Y-then-Z in the evaluator's own hand-rolled sequence,
then add Location or normalize; TEXTURE subtracts Location, rotates
by the NEGATED angles through the SAME sequence (the evaluator's
quirk, reproduced rather than corrected), and divides by the Scale
floored at 1e-8. When the Rotation socket is unlinked — practically
always — its cos/sin are computed at plan time with NumPy's own
float32 trig and baked as literals, so the driver does no
trigonometry a CPU frame did not do (a driver's sin rounds
differently, and a texture lookup at a texel boundary is a cliff).
Per-pixel-driven Rotation falls back to in-shader trig.

Verified in the emitter parity harness (all four modes, a varying
vector, a linked rotation — 59 cases all matching the evaluator) and
end to end as the field's own shape: a Texture-Coordinate → Mapping →
Image-Texture chain driving glass, per-fragment against the
compositor's CPU call — 0 of 678, max 0.000000.

### Fixed — the per-rank correctness fix no longer re-runs the chains

1.25.55's rank-by-rank CPU shading rebuilt each Bump material's
height field per rank — re-evaluating the noise chain once per depth
layer, which took the field frame from 33 to 43 seconds while the
Mapping gap kept it off the GPU. Heights are per-fragment pure, so
each material's chain now evaluates ONCE over all its fragments;
each rank scatters its own subset and differences on the frame grid.
Same fields to the bit, a fraction of the evaluator cost. (Once the
GPU takes the frame this path only matters for CPU renders — but a
correctness fix should not cost 30% either way.)

### Fixed — the console hint names a real toggle

The summary line said "Show Statistics in the Debug panel"; the
toggle is labelled **Timing Breakdown**. The field looked for a
button that did not exist. The hint now names the label on the
actual switch.

---

## [1.25.55] — 2026-08-01

The 1.25.54 report proved the ray-traced layers on hardware
(**0.000012, 0 px**) and the field named its last wall by material:
`'Material.008': a Bump pre-pass on a transparent layer is not ported
yet`. The field's glass is bump water — the Water anatomy itself.
This round ports it, and fixes the CPU bug the port dug up.

### Fixed — transparent bump gradients were a function of the batch

Older than the port: a transparent fragment's Bump gradients came
from `_screen_grad` over whatever shading batch it landed in. All
ranks shade in ONE mixed array there, so front and back faces of the
same glass COLLIDED in the height scatter (last write winning by
sort order), and every chunk boundary cut the waves: in the repro
scene, 539 of 1914 fragments moved with the chunk size — by up to
2.98 — in rasterisation order, dozens even in the compositor's
pixel-sorted order. The picture depended on the batch layout, on the
field's own 25-second path, and nothing had ever diffed it.

The layer is the surface: one fragment per pixel per rank. The CPU
now shades the A-buffer RANK BY RANK when any fragment's material
carries a Bump node, with whole-material gradient fields built from
each rank's own fragments — `_bump_height_fields`, the same pre-pass
the opaque frame runs, applied per layer. Frames without a Bump
material keep the old single call untouched. The bumpy-glass frame
is now bit-identical across thread counts and chunk caps, and the
bump still moves the fragments by 3.38 — invariant, not inert.

### Added — Bump pre-passes ride the transparent layers

The GPU side is the same definition: each material's height pre-pass
draws PER RANK over that layer's own ids texture (the same shader
names the opaque frame compiles — cache hits), and chains the GLSL
emitter refuses — Blender's sin-fract Noise above all, the exact
Water — are CPU-evaluated over the rank's virtual surface into the
pre-pass image, exactly as the opaque path does. The main pass takes
its neighbour differences from the rank's own coverage, which IS the
per-rank field the CPU now gathers from. The refusal is gone in both
plan paths (plain and ray-traced layers).

Proof per FRAGMENT against the compositor's own CPU call, alpha
included: bump glass **0 of 678, max 0.000022**; the full Water
anatomy — sin-fract Noise into Bump on 0.5-opacity glass, ray
tracing ON, refracting among mirrors — **0 of 678, max 0.000003**.
The self test gains a BUMPY GLASS LAYERS whole-render section:
NOISE-into-BUMP glass with a mirror, ray on, CPU vs driver.

With this, every named transparent-layer refusal that fit the
architecture is lifted: layers shade, recurse, and bend on the GPU.
(Rich worlds behind NON-ray layers remain the one named holdout.)

---

## [1.25.54] — 2026-08-01

The 1.25.53 report confirmed the freeze fix in five words better than
the last five ("It no longer stops responding") and left exactly one
named enemy: `transparent layers under ray tracing recurse on the
CPU` — 25.1 of the 33.3 seconds. This round lifts it.

### Added — transparent layers ray-trace on the GPU

Under ray tracing, a transparent fragment spawns the same recursion an
opaque pixel does: reflection rays scaled by the material's constants,
refraction rays lerped through the glass, children at every depth. The
executors now run exactly that machinery per layer: each rank's
fragments become a virtual PRIMARY surface — triangle ids and true
barycentrics are all the ray machinery reads from a surface — and
`_run_sweeps` walks `_add_raytraced`'s tree from it, with the layer
pass's real alpha riding through untouched (the CPU's own order: rays
modify rgb, then alpha computes independently). The secondary passes
compile under the same names the opaque frame uses, so a frame that
already traced its opaque half pays no new compiles. Sampling identity
is the fragment's own pixel — the same streams the CPU threads through
`trace()` — so soft shadows and ambient occlusion jitter identically
from a layer fragment on either device.

The plan side grew three legs to stand on:

- see-through materials pass the same ray gate opaque ones do, and
  their reflect scales and refraction lerps join the ray plan — which
  the LAYER materials can now open by themselves, so an all-glass
  frame (the field's shape) traces with no opaque material anywhere;
- secondary (hit) probes treat the alpha fields as inert — every
  consumer of a hit colour reads rgb only — closing a latent refusal
  where any see-through material in the mesh pushed a whole
  Sorted-Blend ray frame off the GPU with 'visible only in
  reflections' plus the alpha message;
- rich worlds behind RAY-TRACED glass qualify outright: under ray
  tracing the environment term lives at the recursion's final depth,
  where the CPU-composite `__env` machinery already covers any world.
  (The non-ray rich-world refusal keeps its name, and a Bump pre-pass
  on a layer still refuses by name.)

Proof per FRAGMENT against `_shade_chunked` — the compositor's own
call — alpha included, all exact on the first full run: one bounce
**0 of 678, max 0.000003** (and ray tracing moves those fragments by
1.59, so the rays are load-bearing); ray depth 2 **0 of 678**; the
field's own shape, all-transparent AND ray-traced, **0 of 7303**;
Bryce behind reflective glass **0 of 678**; soft shadows + AO sampled
from layer fragments **0 of 304**. The self test gains a RAY-TRACED
GLASS LAYERS whole-render section (glass among mirrors, CPU vs your
driver).

---

## [1.25.53] — 2026-08-01

The 1.25.52 report closed the transparent-layer arc on hardware —
**0.000024, 0 px** — and the field named the remaining wound in five
words: "still does the not responding thing". This is the freeze
round, planned since 1.25.49 and deliberately shipped alone.

### Fixed — the interface stays alive during a render

`bl_use_gpu_context` gave the render thread a GPU context by freezing
the interface: Blender cannot draw a window while another thread holds
the context, so a 33-second frame was 33 seconds of "Not Responding".

The flag is now off by default. The render thread runs with NO
context — export, rasterising, sorting, compositing, every CPU
fallback — and the moments that genuinely need the driver (compile,
upload, draw, read back) are marshalled to the main thread through the
new `gpu/marshal.py`: the worker queues the burst and waits, a
`bpy.app.timers` timer (Blender's own documented cross-thread
pattern) runs it on the main thread, and the result or the exception
crosses back. Each burst is milliseconds; the interface breathes
between bursts. The four crossings are the compute rasteriser, the
deferred frame, the transparent layers, and each post stage.

Every failure mode stays honest and lands on the CPU with the reason
printed, never a broken picture:

- background renders (`blender -b`) hold the context automatically —
  there is no interface to freeze, and a windowless main loop may
  pump no timers;
- a main thread that never runs the burst is a bounded timeout, not a
  hang, and the frame shades on the CPU with that reason printed;
- the Debug panel gains **Hold GPU Context (freezes UI)** to restore
  the exact pre-1.25.53 behaviour with one click, from the next
  render;
- the self test is untouched by construction: operators already run
  on the main thread, so its bursts run in place.

Proven headless with the fake Blender grown a pumpable main loop: the
marshal's mechanics (crossing, result, exception, timeout, timer
retirement) test one by one, and a whole engine render runs from a
worker thread while the test pumps the timer — completes with no
deadlock, delivers a real frame, five bursts crossed, nothing left
registered. The real proof is the field's own F12: the render should
keep painting tiles and the interface should keep drawing.

Expect the frame a shade slower than 1.25.52 in exchange: each burst
now waits for a main-loop tick (~10 ms each, a handful per frame).
The 25.2 s that remains in the field frame is the ray-traced layer
recursion — top of the ROADMAP, unchanged by this round.

---

## [1.25.52] — 2026-08-01

The 1.25.51 report was the diagnosis working as designed. The new
fragment-level instrument reported **0 of 76573 flips, max 0.000048**
— the driver's layer shading is exact — while the whole-render diff
sat at *bit-identical* numbers to the round before (0.125066,
1036 px) across two different blend modes. Identical numbers under a
changed suspect acquit the suspect: the divergence was never in the
layer shading.

### Fixed — the picture no longer depends on the rasteriser's last ULP

The real mechanism, confirmed numerically: the demo box's bottom face
is **coplanar with the floor** — a modeled contact — so its
transparent fragments interpolate to exactly the opaque depth, give
or take a few float32 ULPs. The CPU rasteriser and the compute kernel
round those ULPs differently (zndc measured ~9e-7 apart, within the
rasteriser section's stated bounds), and the A-buffer's bare `<`
depth test let that rounding decide, pixel by pixel, whether the
contact layer exists: 2877 tie pixels in the scene, ~1000 flipped
per device — the 1036.

Collection and compositor now share one tolerant limit
(`raster.abuf_depth_limit`, ~30 ULPs): a fragment on or immediately
behind the opaque surface is kept the same way whichever rasteriser
wrote the depth. Opaque hidden-surface removal keeps its exact `<` —
the tolerance exists only to decide whether a see-through fragment
sits ON a surface, where modeled geometry makes exact ties common.
This extends the scheduling-invariance doctrine to the rasteriser
pair: chunks, threads, bands — and now devices — cannot move the
picture. Proof: 3606 keep flips under last-ULP rounding with the bare
comparison, **0** with the limit; the end-to-end surviving fragment
sets are identical (81544 = 81544); the contact layer is kept whole.

The next self test's TRANSPARENT LAYERS section should read
0.0001-class. Nothing about this was specific to transparency ports —
the tie has been latent since the compute rasteriser landed; the new
section was simply the first to diff a transparent frame across
devices.

### The field frame's named refusal stands — and names the next round

`transparent layers under ray tracing recurse on the CPU` is the
correct, intended refusal for the field scene (all-transparent AND
ray-traced): a traced frame's transparent fragments spawn their own
reflection and refraction rays, and that recursion is not ported yet.
It is now the top of the ROADMAP — the sweeps machinery (secondary
passes, hit shading, the recursion tree) already runs per-frame on
the driver; the round to come runs it per LAYER, over the same ids
frames the layer passes already draw.

---

## [1.25.51] — 2026-08-01

The 1.25.50 field report caught the layer port twice — once refusing
where it should have engaged, once engaging wrongly — and both
mechanisms are now closed. The field frame printed `transparent layers
on the CPU: no transparent-layer plan for this frame` and kept its
26.5 s; the new TRANSPARENT LAYERS self-test section measured
**0.125066, 1036 px** against the CPU.

### Fixed — an all-transparency frame never built its layer plan

The field frame's `transparency shading 26.5s` with no `shade` stage
anywhere in the top list was the tell: every visible material in that
scene is see-through, so the **whole picture** goes through the
A-buffer and the opaque G-buffer is empty. `plan_frame`'s
"nothing to shade is a success" early return — written when the plan
only served the opaque pass — returned before the layer planning ever
ran, with empty atlases and the unnamed default message. An empty
opaque frame now falls through: the opaque loop sees no materials (an
empty pass list is still a success), and the layer plan builds from
each see-through material's own triangles, no G-buffer pixels
required. Proven headless on the exact shape: all three demo
materials at opacity 0.5, opaque G-buffer empty, layer plan holds
every material, per-fragment seam 0 of 7303 at max 0.000006.

And the default message is gone as a category: when the layer plan is
empty but fragments arrive anyway, the refusal now names the first
fragment's material and the exact fields the predicate read
(`'Water' (opacity 1.000, has_alpha False read as opaque...)`) — a
disagreement between the transparent subset and the layer predicate
can no longer hide behind a generic line.

### Fixed — the layer merge stands on the proven blend state

The self-test's fragment shading mismatched 1036 px of 172800 (max
0.125) on hardware while the headless seam was exact at the same
480×360 — a driver-only divergence, in the one place this engine used
a blend state no field frame had ever proven: `ADDITIVE_PREMULT`. The
proven paths — the opaque frame's material merge and every secondary
sweep, same scattered ids content, same texture sizes — all composite
disjoint per-pixel writes under `ALPHA_PREMULT`, and for disjoint
writes the two are the same arithmetic (`out = src + dst·(1−src.a)`:
at a material's own pixels dst is 0; everywhere else src is vec4(0)
and dst passes through). The layer merge now uses `ALPHA_PREMULT`
like everything else. Lesson recorded in the code: never stand new
driver state under a new feature when a proven state computes the
same thing.

### Added — the layers section diagnoses itself

If the TRANSPARENT LAYERS section still disagrees, it now corners the
mechanism at the FRAGMENT: the compositor's own `_shade_chunked` call
against the driver's layer draws on identical sorted ranks, with flip
counts, channel structure (rgb-only / alpha-only / both), histograms
by rank, material, and facing, a driver-vs-itself determinism check,
and the three worst fragments in numbers. Whatever survives the blend
fix gets named, not guessed at.

---

## [1.25.50] — 2026-08-01

The 1.25.49 summary line did its job on its first field frame: of the
33.7 seconds, **transparency 28.4s — transparency shading 25.7s**. The
A-buffer's fragments (two faces of every glass surface, times four at
SuperSample ×4) were shaded entirely on the CPU: the deferred pass
only ever drew the opaque G-buffer. This release puts the transparent
layers on the driver too.

### Added — transparent layers shade on the GPU

Under Sorted Blend and A-Buffer transparency, the compositor's
fragments now shade through the same deferred machinery as the opaque
frame. Per depth layer: the layer's fragments become an ids texture
(real triangle, REAL barycentrics, everything else uncovered), every
see-through material draws a full-screen LAYER pass over it, and the
per-pixel-disjoint materials merge additively into one target — one
readback per layer, gathered back per fragment. The depth sort, the
per-pixel ranking, and the farthest-first alpha blend stay exactly
where they were, on the CPU; only the shading crossed.

The layer pass emits the material's REAL alpha, the same chain
`shade_batch` runs: opacity clamped, the hard Alpha Threshold cutoff,
then the edge-opacity silhouette blend — measured against the bent
normal, exactly as the CPU tests `ctx.N` (a normal-mapped glass
silhouette fades identically on both devices). The driver merge uses
ADDITIVE_PREMULT — the plain (ONE, ONE) sum on both channels. Plain
ADDITIVE would have premultiplied rgb by alpha and accumulated NO
alpha at all: every layer would have read back invisible.

Refusals stay narrow and named, and the transparent shading falls back
to the CPU with the reason printed — the opaque frame keeps its GPU
pass either way:

- `transparent layers under ray tracing recurse on the CPU` (a traced
  frame's layers spawn their own rays — unchanged this round)
- `a rich world behind transparent layers stays on the CPU` (the
  CPU-composite env term adds at G-buffer pixels; a layer's fragments
  are not those)
- `a Bump pre-pass on a transparent layer is not ported yet`
- materials that never rasterise a transparent fragment are not
  probed for layers at all — an opaque Bump material next to glass no
  longer costs the frame its layer pass, and a run-time coverage
  check refuses by name if a fragment ever arrives with no pass

Proof, headless: the exact per-fragment compare — `_shade_chunked`
(the compositor's own CPU call) against the layer passes through the
GLSL front-end on identical sorted ranks, **alpha included**. Two
glass materials at once (the smooth ball and the anisotropic box),
1631 backfaces of 3049 fragments, layers 4 deep, the env term on a
layer: **0 fragments off by >0.01, max 0.000006**. The edge-opacity
trial (threshold 0.6 zeroing the facing area, rim term only):
**max 0.000000**. The self test gains a TRANSPARENT LAYERS section
that runs the whole two-glass render on the driver.

### Fixed — the add-on's own glass templates exported as opaque

Surfaced by the layer port: `_tree_has_alpha` never looked at the
master shader node, so a converted or template material carrying its
alpha on the node's sockets — the Glass template is Opacity **0.12**,
with the override off — exported `has_alpha=False`, landed in the
OPAQUE pass, and rendered solid under Sorted/A-Buffer. The exporter
now reads the master node's Opacity and Edge Opacity (linked, or
below 1.0) exactly as it already read Principled's Alpha. Ray-traced
refraction had hidden this: `surf.opacity` drives refraction rays
regardless of the pass split, so glass LOOKED right under Ray Trace
and quietly went solid without it.

### Fixed — a stale Alpha Threshold could outlive its plan

The plan cache's signature did not include `alpha_threshold`, which
the layer alpha chain bakes as a constant; editing the threshold on
an otherwise-unchanged scene would have re-served the old cutoff.
It signs the plan now.

---

## [1.25.49] — 2026-07-31

The field reported a 33-second 1024×768 render (SuperSample ×4 —
2048×1536, 3.1 million pixels) with Task Manager reading 0% GPU and
the interface frozen until the frame finished. The self-test was all
zeros. Three findings, two fixed here:

### Fixed — the worker pool no longer throws the GPU away

With Use Processes on, F12 rendered through the worker pool — and the
pool splits the frame into bands, where the deferred pass (whole-frame
by design) silently never engages. No fallback line printed, because a
banded worker never even attempts the driver. A 3.1-million-pixel
frame CPU-shaded across bands: 33 seconds, and Task Manager's 0% GPU
was the literal truth. The pool made sense in the CPU-only era; with
GPU Shading on it is strictly slower.

Now, when the device is GPU with GPU Shading enabled, the pool is
skipped with a printed line — whole frames, in-process, on the driver.
Turning GPU Shading off restores the pool for CPU rendering.

### Fixed — every render prints where its seconds went

The frame breakdown was collected for every render all along, but only
printed with Show Statistics ticked — the 33-second mystery had its
answer gated behind a panel. Every F12 now ends with one line:
`[Halcyon] 12.3s -- top stages: shade 8.1s, post 2.2s, rasterise 1.1s`
— and points at the Debug panel for the full table.

### Known, planned — the UI freeze during long renders

`bl_use_gpu_context` holds the GPU context for the whole render, so
Blender's interface cannot draw until the frame ends. With the pool
fix the freeze shrinks with the render time, but the real cure is the
viewport's own pattern — CPU work with the context released, GPU
bursts marshalled to the main thread — which lands as its own
carefully-tested round (ROADMAP has the entry). Task Manager note for
the meantime: the deferred pass's driver work is milliseconds per
frame, so 0-1% GPU utilisation is what a HEALTHY GPU render reads —
the number to watch is the frame time, not the gauge.

---

## [1.25.48] — 2026-07-31

The 1.25.47 report confirmed the sky column on hardware — the Bryce
sky lab in a mirror at **0.000048, 0 px of 172800** — thirteen
sections, all zero. This release closes the last scheduling-dependent
picture artifact anywhere in the engine.

### Fixed — worker bands no longer seam the waves

A pooled CPU render splits the frame into bands, each worker shading
only its rows — and `n_bump`'s gradients difference toward the +y
neighbour, so a band's top row flattened its waves against missing
coverage. The chunk seam's sibling, at every band edge, in every
pooled render of a bump material.

Two small moves kill it exactly: the band's scissor rasterises ONE
context row past its edge (the scissor culls whole triangles, so that
row's coverage is complete), and the bump gradient fields build from
the G-buffer's full coverage instead of the band's shading rows. A
band's gradients are now the whole frame's, bit for bit: two bands
stitch to the exact whole frame (0.000000000), with the negative
control proving the seam was real (0.28 on exactly the edge row with
the mechanism disabled).

With chunks (1.25.40), threads (1.25.43's sampling), and now bands,
the invariant is complete: **the picture never depends on internal
scheduling.** Not chunk size, not thread count, not how the pool
splits the frame.

### ROADMAP

Skies marked confirmed-on-hardware; the band item checked off. What
remains for 1.26.0: texture filters (TRILINEAR/N64, STIPPLE, affine
carry), Gabor/TexSky, cast-filtered shadows, the viewport question.

---

## [1.25.47] — 2026-07-31

The ray arc confirmed complete, the roadmap said skies — and the whole
sky column falls to one mechanism instead of five ports.

### Added — every world reflects, evaluated by the renderer itself

The insight: the environment-reflection term is the LAST rgb term the
CPU adds to a pixel (fog frames already refuse), and every pixel it
applies to is CPU-known — reflective primaries by mask, depth-exhausted
hits by readback. So a world too rich to bake into GLSL does not need
porting at all: the composite asks `world_color` — the renderer's own
world, the full Bryce sky lab included — along the reflected rays, and
adds the term after readback. Exact for ANY world, by construction:
same normals (the proven bent, unflipped ray normals), same camera V,
same add, in the same place.

Six refusals deleted at once: **STARFIELD, BRYCE, PHYSICAL, HDRI,
world node graphs, and the ground plane** all reflect now — in primary
env (ray tracing off), at depth-exhausted hits (ray tracing on, any
depth), and under the recursion (Bryce at ray depth 2, mirror-in-
mirror under the sky lab, proven). The simple modes (SOLID, GRADIENT,
BANDS, BLEND, EQUIRECT, MIRRORBALL) keep their baked GLSL fast paths.

Headless seams: all twelve world/site combinations at **0 px**, max
0.000006, each with the env term proven load-bearing in the mirror.
The honest cost: one CPU world evaluation per frame over the env
pixels — proportional to reflective coverage, like the ray slices.

The self-test gains a RICH WORLDS IN REFLECTIONS section: the Bryce
sky lab in a mirror, on your driver.

### ROADMAP

The skies-and-worlds column is DONE. Remaining for 1.26.0: texture
filters (TRILINEAR/N64, STIPPLE, affine carry), Gabor/TexSky,
cast-filtered shadows, the worker-band seams, the viewport question.

---

## [1.25.46] — 2026-07-31

The 1.25.45 report, RAY DEPTH 2 section, first working driver run:

```
max difference : 0.000024
px off by >0.01: 0 of 172800
mirror-in-mirror on your driver -- the ray arc is COMPLETE
```

### Changed — `raytrace` flips to BOTH

The capability table's bar for BOTH is "ported, and measured against
the CPU path on real hardware." Every piece of the ray arc has now met
it, on the field machine, at zero: hard ray shadows (0.000048, 0 px),
soft ray shadows and ambient occlusion (0.000047, 0 px), one traced
bounce (0.000048, 0 px), refraction through bent noise-and-bump
normals (0.000028, 0 px), and the full recursion tree at depth beyond
one (0.000024, 0 px). The flag's text now records those numbers, and
the ROADMAP marks the ray column DONE.

"Ray tracing and all that" — the scope 1.26.0 was named for — is on
the GPU, in full, exactly.

Next per the ROADMAP: the remaining sky modes in reflections
(STARFIELD, BRYCE, PHYSICAL, HDRI, the ground plane).

---

## [1.25.45] — 2026-07-31

The 1.25.44 report: every proven section still zero — soft+AO at
0.000047/0 px again, 53.9 ms — and the new RAY DEPTH 2 section crashed
on its first driver run:

```
[gpu compute capability failed: ValueError: operands could not be
 broadcast together with shapes (34940,4) (34940,3)]
```

### Fixed — the level-image contract, and shade_frame's promise

A backend split the headless path could never reach: the front-end
adapter returns level images with THREE channels, the driver's
`read_target` returns FOUR — readable by every composite that only
looked, but depth > 1's child composites are the first code to WRITE
into a level image, and `(N,4) + (N,3)` is a broadcast error. The
shared recursion now normalises every level image to (H, W, 3) at the
single point both backends meet, and the depth test exercises a
driver-shaped four-channel adapter through the real recursion and
trace: both shapes must produce the identical picture.

Second fix, the deeper one: that ValueError ESCAPED `shade_frame`,
whose contract says every failure is a reason and the caller shades on
the CPU — instead it killed the whole self-test section. The sweep
loop now converts any unexpected exception into a named reason
(`the ray sweeps failed: ...`), so a driver-path surprise costs a CPU
fallback with its cause printed, never a crashed render.

Run Self Test again: the RAY DEPTH 2 line should read its numbers this
time, and if the max is zero-class, the ray arc's last driver
confirmation is in.

---

## [1.25.44] — 2026-07-31

The 1.25.43 report proved the sampling arc on hardware first try —
SOFT SHADOWS + AMBIENT OCCLUSION at 0.000047, **0 px of 172800**, 69.6
ms against the CPU's 2270 — alongside the first field picture rendered
entirely on the GPU. One ray item remained, and this release takes it:
**ray depth > 1**. The ray arc is now ported whole.

### Added — the recursion tree, walked branch by branch

The CPU's `_add_raytraced` recurses: at depth d < D a hit's own
reflective and refractive materials spawn the next rays — and its
shading carries NO environment term, because the traced child replaces
it — while at d == D the hit shades with the environment, the
depth-exhausted branch. The deferred pass now walks the same tree:

- per level, hits shade through the secondary passes — a new
  `secondary_mid` variant (no env) above the final depth, the existing
  env-bearing variant at it;
- hits on ray materials spawn the next level's rays FROM the hit
  surfaces: camera V (ctx.I = P − eye, always — the CPU's own rule),
  the Normal Map chain bending, the Bump node a wire (ctx.px is None
  at a hit), reflection stepping off the surface and refraction into
  it by the raw bias;
- children composite BACKWARD with the HIT material's constants — the
  reflect scale, the refraction lerp's k and tint — and `world_color`
  along child rays that missed;
- a pixel whose reflection hit both reflects and refracts BRANCHES,
  exactly as the CPU branches: the tree is per-branch state, not one
  collapsed image;
- hidden materials' constants join the plan (a level-2 ray can hit a
  material no camera ray sees, and ITS hits spawn level 3);
- the deterministic sampling identity follows every chain (1.25.43's
  threading was already recursive), so soft shadows and AO at any
  depth's hits draw the same streams on both devices.

Depth 1 reduces to the exact flat sweep it generalises — same draws,
same trace, same composite. Headless seams, two mirrors + branching
glass (second bounce load-bearing at 0.615): **depth 1: 0 px. Depth 2:
0 px, max 0.000003. Depth 3: 0 px, max 0.000003.**

The self-test gains a RAY DEPTH 2 section — mirror-in-mirror on the
driver — and the capability table now says it plainly: the ray arc is
ported whole, flipping to BOTH when the depth-2 frame earns its driver
zero. The old refusal ('ray depth N recurses on the CPU') is deleted.

### ROADMAP

Ray depth > 1 is checked off. The ray column of the 1.26.0 map is
DONE, pending its last driver confirmation.

---

## [1.25.43] — 2026-07-31

The ray arc resumes, by request: of "ray depth beyond 1, ambient
occlusion, and soft ray shadows", the two sampling features fall
together this release — because they shared one disease, and it was
the same disease the bump seam had.

### Fixed, then ported — sampling is a pure function of the pixel

Soft ray shadows drew their jitter from a sequential random stream —
two scalars per sample **shared by every pixel in a batch**, the
sequence owned by chunk order. Ambient occlusion drew per-pixel numbers
from the same kind of stream. Both meant the PICTURE depended on
internal scheduling (penumbras could seam at chunk boundaries exactly
like the bump waves did), and neither could ever be reproduced by a
full-screen pass. The old refusal named it precisely.

Now every sample is a pure function of (pixel, sample index, stream,
seed): draws through the same integer hash the pattern textures proved
bit-exact on real drivers, and angles from a shared 256-entry
unit-circle table — a texture both devices read, because a driver's own
sin/cos round differently and an occlusion ray is a cliff. All float32,
mirrored operation for operation. Consequences, in order:

- the CPU picture is **chunking- and thread-invariant** (proven
  0.000000000 across thread counts), and soft penumbras are per-pixel
  jittered now — strictly better-looking than the old batch-uniform
  offsets;
- traced HITS sample the same streams: the CPU threads the spawning
  pixel's identity through `trace()`, and the GPU's secondary-pass
  fragment simply IS that pixel;
- both features shade in the deferred pass: the soft loop in the ray
  shadow function, `hal_ao` scaling the ambient term exactly where
  `light_surface` scales it, both walking the shared BVH traversal.

Headless seam, the full stack at once — soft fill light + AO + traced
reflections with hits shading both: **0 px off, max 0.000013**. Two
refusals gone: 'a random stream whose sequence the CPU batch order
owns' and 'ambient occlusion is not ported'. The self-test gains a
SOFT SHADOWS + AMBIENT OCCLUSION section to prove it on your driver.

The interface declarations moved to the head of the assembled source
(the sampling helpers read `vUV` inside function bodies, and GLSL wants
declarations first) — the driver-strict lint holds as before.

### Added — ROADMAP.md

The living map of the distance to 1.26.0: what is proven on hardware,
what remains (ray depth > 1 is the ray arc's last item), what refuses
by name, and what is impossible by construction. Updated each release.

---

## [1.25.42] — 2026-07-31

The 1.25.41 field console:

```
[Halcyon GPU] shading on the CPU: 'Water': its Bump gradients chunk
on the CPU at 347106 covered pixels (one batch holds 262144)
```

The 1.25.40 refusal — added for what looked like an edge case — fired
on the field's own Water and threw the frame back to an 11-second CPU
shade. That refusal was sized to a demo scene, and it was treating a
symptom. This release removes the disease instead, and the refusal
with it.

### Fixed — bump gradients are whole at ANY size, so nothing refuses

The cap existed because the CPU could only make `n_bump`'s gradients
whole by shading a material in ONE batch, and one batch is memory-
bounded. But the gradients never needed the *shading* to be one batch —
they need the *heights* to be one picture. So the CPU now does exactly
what the deferred pass already proved: a material too covered for one
batch gets a whole-material height PRE-PASS first — its height chains
evaluated over the material's full frame pixels (chunked internally;
heights are per-pixel independent, so chunking there cannot cut
anything), scattered to a frame grid, differenced ONCE — and every
shading chunk gathers its gradients from that field instead of from
its own batch. Bitwise the same arithmetic as a whole-material batch:
same float32 grid, same forward differences, same validity, same
gather. Materials that fit one batch never build a field and take the
exact path they always took.

Proven at the field's own scale: a 307200-pixel bump material (over
the 262144 cap) renders IDENTICALLY at threads 1 and threads 8
(0.000000000), QUALIFIES for the deferred pass, and the GPU frame
matches the whole-gradient CPU frame at 0 px off, max 0.000009. The
memory bound chunking exists for is untouched.

What remains named: the WORKER-POOL bands still cut gradients at band
edges (each band shades its rows independently) — visible only in
pooled CPU renders of bump materials, never in the deferred frame.

### For the field scene

F12. The console line above is gone — 'Water' at 347106 pixels takes
the height pre-pass on the CPU and the deferred pass on your driver,
at any resolution, permanently. The 2-second frame is not coming back;
it was never supposed to leave.

---

## [1.25.41] — 2026-07-31

The 1.25.40 report came back all zeros — the Water anatomy proven end
to end on hardware, the chunk seam confirmed dead — which promotes the
next number in line: the warm frame's ~2.1 s of `shade (GPU)` at
640×480, which is not GPU time at all but the per-frame CPU slices
(ray construction and the noise heights). This release makes them
share their work.

### Changed — the frame's CPU slices compute once, not three times

Profiled at 640×480: the height image cost 54 ms, reflection rays
164 ms, refraction rays 85 ms — and they were all rebuilding the same
things. Both sweeps ran `job.context` AND the full evaluator over the
water's pixels (it reflects AND refracts), and the height pre-pass
built its own context and evaluated the noise chain a third time.

Now one per-material, per-frame cache (`_mat_eval`, living on the job —
one job is one frame, keyed by the G-buffer's identity) holds each
material's pixels, shading context and GraphEvaluator. The height
image, the reflection rays and the refraction rays all pull from it,
and sharing the EVALUATOR is the point: its per-node cache means the
sin-fract noise evaluates once and every later ask — the bend, the
pre-pass — is a lookup. Ray building itself became per-material blocks
(`_ray_blocks`), so a material in both sweeps pays its bend once.

Exactness is untouched by construction — the blocks run the same
per-pixel arithmetic the mixed selections ran, every consumer scatters
by pixel coordinates, and the shared evaluator returns the identical
cached arrays: the 640×480 Water frame still reads **0 px off, max
0.000018** against `render()`. Measured: heights + both sweeps went
**305 ms → 140 ms** at 640×480 in the build sandbox (~2.2×; a
water-dominated frame gains more, since the water is exactly the
material both sweeps share).

### Added — the report shows where the sweep milliseconds go

`LAST_TIMINGS` gains `ray_build_ms`, and the self-test's shade split
now reads `sweeps X (of which CPU ray build Y)` — so the next
conversation about the warm frame starts from a number instead of a
guess.

### For the field scene

F12 twice and compare the warm frame: the shade (GPU) stage should
drop by roughly the shared work — the water pays its context, its
noise and its bend once per frame now. The remaining CPU cost is the
one honest slice left (one evaluator run per ray material per frame);
if the report shows it still dominating, a bend pre-pass on the GPU is
the next candidate — but only if it can earn a zero of its own.

---

## [1.25.40] — 2026-07-31

The A/B quartet convicted the bump bend — and then the hunt turned the
verdict inside out: **the driver was innocent all along. The GPU
picture was right; the CPU picture had the bug.**

### Fixed — the CPU's bump gradients no longer depend on the chunking

The trail, step by step: the 38 bad pixels reproduced HEADLESSLY, byte
for byte — same count, same max (0.192843), same worst pixel (166,196)
— so this was never driver arithmetic. The front-end's bent normal,
`P`, `V` and the two-sided flip all matched the CPU's own evaluator
exactly at the bad pixels; the sky was already ruled out by magnitude;
shadows survived their own A/B. What remained was the reference itself:
the CPU shades PIXEL-rate fragments in mixed-material chunks, `n_bump`'s
neighbour validity means "shaded in the same batch", and chunk boundary
79917 of the 480×360 frame landed mid-glass on row 166 — every pixel on
that row whose +y neighbour fell in the NEXT chunk shaded with its
waves flattened (dhdy forced to 0). One row of subtly wrong water, on
any machine whose settings chunked there, in every CPU render since the
Bump node existed. The GPU's whole-material height pre-pass computed
those gradients CORRECTLY — the 38 px were the fix disagreeing with the
bug.

The cure is CPU-side and principled: `_shade_all` now shades PIXEL-rate
fragments ONE MATERIAL PER CHUNKED CALL. Gradients are whole for any
material that fits the existing memory cap (262144 fragments — the cap
itself is untouched, chunks just never cross a material's interior
below it), and the picture is now a function of the scene, not of the
thread count. Proven by invariant: threads 1 and threads 8 render the
IDENTICAL frame (0.000000000), and the field-size Water frame reads
**0 px off, max 0.000012** against the deferred pass — the very check
that read 38.

Honesty at the boundary: a Bump material covering MORE than one batch's
cap would still cut on the CPU, so the plan now refuses it by name
("its Bump gradients chunk on the CPU at N covered pixels") rather than
render a different, seamless picture. And a named limitation that
remains: the WORKER-POOL bands still cut bump gradients at band edges
(each band shades its rows independently), so a pooled CPU render of a
bump material can still seam at band boundaries — the deferred GPU
path, which shades the frame whole, does not.

One side effect worth naming: CPU frames rendered with soft-shadow or
other per-chunk sampling may show a different (equally valid) noise
pattern, because the chunk seeding now walks materials in order.

### For the field scene

Run Self Test: the refracted section should finally read near-zero with
no DIAGNOSIS block at all. And your renders gain something real — every
CPU frame of the water was carrying a one-row flattened-wave seam
wherever the chunking landed; it is gone from CPU and GPU alike, and
the two now agree because both are right.

---

## [1.25.39] — 2026-07-30

The 1.25.38 report is the one this arc was for: the field F12 went
**12.7 s → 3.9 s** with a single `shade (GPU)` line — 'Water' compiles,
the fallback is gone — and the refracted section reads `shade_frame
directly: OK` at 111 ms against 390 CPU. And then the honest number
underneath: **max 0.192843, 38 px of 172800**. This frame's GLSL — the
bump main pass, the secondary passes of a two-sweep frame — had never
actually executed on a driver until the glue fix landed; every earlier
0.000012 in this section was CPU against CPU. The first real run
disagrees somewhere, and this release makes the section corner it.

### Added — the refracted frame diagnoses itself

Headless analysis already ruled the BANDS sky OUT for this scene: its
bands are two near-identical darks, a band flip moves a pixel by about
0.01, and the ground branch is off — nothing in that chain can make
0.19. What remains needs the driver to separate, so on disagreement the
section now prints:

- **bad px by surface** (floor / glass / mirror / background), their
  **ray membership** (reflection px, refraction px), and **what their
  own rays hit** — rays rebuilt by the frame's own builders and traced
  by the proven CPU BVH, so the classification is exact;
- the **three worst pixels** with both devices' RGB — channel structure
  tells a shadow flip (large, light-coloured, all channels) from a tint
  or band error (small, colour-shaped);
- the **A/B quartet**: the same frame re-rendered four times, one
  suspect disabled per run — shadows OFF, sky strength 0, bump
  strength 0, ray refraction OFF — with the differing-pixel count for
  each. Whichever variant reads zero names the mechanism, and the next
  round aims instead of guessing.

State is restored after every variant; the quartet costs a few seconds
of self-test time and only runs when the frame disagrees.

### For the field scene

Press F12 **twice** and compare: the 3.29 s shade (GPU) in the report
was a first-frame number — it pays every material's shader compiles
(primary + secondary + pre-passes). The second, warm frame shows what
an animation would pay, which is the per-frame CPU slices (ray-context
bends and the noise heights) plus the true GPU time. And Run Self Test
once more: if the 38 px reproduce, the DIAGNOSIS block underneath is
the whole next round.

---

## [1.25.38] — 2026-07-30

The 1.25.37 report's new line said everything: **`shade_frame directly:
FELL BACK: the driver rejected 'Bumpy': HAL_MAT_1: CreateInfo failed`**
— and the field's F12 console showed the corpse itself:

```
757 | uniform sampler2D hal_bump0;uniform sampler2D hal_shadow0;
      python_shader.glsl:418: Error: 'hal_bump0' : redefinition
```

Two declarations **glued on one line**. Every bump-node material has
been rejected by the driver since 1.25.34 and silently shaded on the
CPU — 'Bumpy' in the self-test, 'Water' in the field, 11.4 s a frame.

### Fixed — the splice

`assemble_frame` joins its source blocks with `''.join`, and each block
was expected to end with its own newline. The texture samplers did; the
Bump emitter's `uniform sampler2D hal_bump0;` did not — so whenever a
bump material had NO image texture in its main pass (the empty texture
block between them), it glued straight onto the shadow block's
`uniform sampler2D hal_shadow0;`. One line, two declarations. The
whole-line stripper couldn't match it, both names reached CreateInfo
already declared, the driver refused the shader, and `render()` fell
back — while every headless seam read zero, because Halcyon's own
front-end happily parses the glued line. The normal-mapped 'Bumpy' in
the first deferred section compiled fine all along (its texture block
is non-empty and newline-terminated), which is why the failure hid in
the refracted section alone.

Three walls now stand where none did:

1. **The joins are line-safe.** All multi-line blocks go through
   `_block()` — newline-separated, newline-TERMINATED when non-empty —
   in the frame assembler and the height-pass assembler both. No
   emitter has to remember its own trailing newline again.
2. **The stripper is immune anyway.** `strip_declarations` now strips a
   line made of ANY number of whole declarations, so even a future
   splice of two declarations cannot reach CreateInfo. (Verified: with
   the old join deliberately restored in memory, the hardened stripper
   alone saves the frame.)
3. **The lint hunts the whole class.** `surviving()` no longer shares
   the stripper's blind spot: it flags glued declaration lines AND any
   surviving `uniform` token anywhere — a stripped source has no
   legitimate use for the word. And the lint finally assembles the
   field's exact combo (Bump material under SHADOW MAPS with an empty
   texture block — `shadows=False` had kept the shadow block out of
   every linted source): primary passes, the height PRE-PASS and the
   secondary passes all strip clean, by name.

The negative control ran before the fix shipped: the old join
reproduces the field's glued line byte for byte, and the old stripper
misses it. Both new walls kill it independently.

### For the field scene

'Water' should compile on your driver now — this was the wall between
your scene and the GPU since the Bump round. F12: the console's
`[Halcyon GPU] shading on the CPU` line should be GONE, and the 11.4 s
shade stage should become GPU milliseconds. Run Self Test too: the
refracted section should read `shade_frame directly: OK` with the
stage table showing ONE shade line, and the section's GPU total should
finally sit far under the CPU's.

---

## [1.25.37] — 2026-07-30

The 1.25.36 report cracked both mysteries open. The new stage table
showed **`shade (GPU)` AND `shade` both running** in the warm frame —
shade_frame has been failing on the driver and falling back to CPU
shading in that section, its reason printed only to the Blender console
— and the field's F12 crashed the off-screen material probe:
**'Material.003' (visible only in reflections): probing it off-screen
failed (operands could not be broadcast together with shapes (16,3,3)
(16,2,1))**. This release fixes the crash outright and drags the hidden
fallback reason into the report where it can be aimed at.

### Fixed — a material visible only in reflections no longer crashes the plan

The off-screen probe — how a material with no pixel on screen still
gets its secondary pass, sixteen samples spread over its own triangles
— built 2-wide synthetic barycentrics. `gbuf.bary` is (H,W,3) and
`raster.fetch` requires (N,3); the shapes collided on the first scene
that actually exercised the path, which was the field's. The probe now
builds full three-component barycentrics (1/3, 1/3, 1/3: the triangle
centroid), and the path finally has the test that should have shipped
with it: a glass floor (IOR 1.0, rays pass straight through) over a
self-lit plane the z-buffer occludes everywhere, plus the mirror. The
plan qualifies, the hidden material gets its probed pass, the frame's
own refraction rays land on it ([3] confirmed by the BVH), the frame
matches `render()` at **0.00002**, and recolouring the hidden plane
moves the picture by 0.25 — through the glass alone — with the GPU
tracking the recolour exactly.

### Added — the silent CPU fallback now names itself in the report

`render()` prints shade_frame's rejection to the console and shades on
the CPU; the report's match then reads 0.000012 — CPU against CPU — and
looks like victory. The RAY-REFRACTED section now calls `shade_frame`
DIRECTLY after the warm render and prints the verdict into the report:
`shade_frame directly: OK` or `shade_frame directly: FELL BACK:
<reason>`. If the stage table shows both shade lines, the line below it
now says why — that reason is the whole next round.

### For the field scene

Material.003 — whatever surface hides where only the Water's rays can
see it — now probes instead of crashing. F12 again: either the frame
qualifies, or the console names the next layer. And Run Self Test once
more: the FELL BACK line will name what the driver rejects.

---

## [1.25.36] — 2026-07-30

The first picture of the field scene arrived — checkerboard plane, palm,
grand piano, noise-rippled water, banded sky; the exact mid-90s image
this engine exists to make — and its console named the next wall:
**'Water' reflects the world: the BANDS sky mode is not in the deferred
pass yet**. So the water reflects too, and the sky it reflects is the
quantised gradient. This release ports it.

### Added — the simple sky modes join the environment term

`sky.solid`, `sky.gradient` and `sky.bands`, term for term, with every
world constant baked into the reflective materials' passes: the
horizon/zenith/ground colours, horizon height, falloff, the blend curve
(LINEAR/SMOOTH/SHARP/EASE), the band count and softness with the same
quantise-in-the-blend-parameter arithmetic, the below-horizon ground
branch, and the strength multiplier — exactly as `evaluate()` applies
them. Rotation is correctly absent: it spins x and y, these formulas
read only the ray's z. STARFIELD, BRYCE, PHYSICAL and HDRI still refuse
by name; the ground PLANE (a traced surface, not a sky) still refuses.

Seam numbers, headless: the complete field frame — noise-into-Bump
REFLECTIVE wavy glass under a BANDS sky, with the mirror — matches
`render()` at **0.00001**, with the banding load-bearing against a
smooth gradient (0.026). GRADIENT qualifies and matches identically.

### Added — the self-test corners the missing milliseconds

The 1.25.35 report proved the shade split honest (28 ms accounted) and
the section still 383 ms — so the hole is OUTSIDE the shade stage. The
RAY-REFRACTED section now prints the warm GPU render's full stage-by-
stage frame breakdown, the same table the field console shows. Whatever
stage holds the missing ~350 ms has nowhere left to hide. (The section
also gains the field's own sky: BANDS, reflective water included.)

### For the field scene

Water's console trail so far: refraction → ported; the Normal chain →
rays bend; the Bump node → height pre-pass; the noise height → CPU
image; the BANDS sky → this release. Each layer was named, each is
lifted. F12 again — and if a next layer exists, the console will name
it, as it has named every one before.

---

## [1.25.35] — 2026-07-30

The 1.25.34 report held two teeth. The Bump machinery proved exact on
hardware (0.000012, 0 px) but its section ran SLOWER than the CPU — 355
ms against 325 — and the field scene peeled to Water's actual core:
**ShaderNodeTexNoise feeding the Bump height**, the sin-fract wall this
project built on purpose. Both fall this release.

### Fixed — the 355 ms: a texture-cache stampede, not the new code

The upload cache held 24 entries and its eviction policy was *clear
everything*. The moment a frame's working set crossed the cap — and the
Bump height map was the texture that tipped it — every warm frame
re-packed and re-uploaded everything it had just evicted, the 33 MB
shadow atlases included. A 60 ms section became 355 over ONE texture of
growth. The cache now holds 64 entries, moves hits to the end, and
evicts the least-recently-used half — a working set that fits never
loses a member. The section also prints its full shade split now
(plan / pack+upload / draw+read / sweeps / other), so a cost like this
can never hide in a catch-all again.

### Added — sin-fract heights evaluate on the CPU, into the pre-pass

The Noise-family refusal exists because a driver's float32 sin
decorrelates the sin-fract hash — the GPU would render a DIFFERENT
pattern. But the Bump pre-pass is only an IMAGE. So a height chain the
emitter refuses no longer refuses the material: the renderer's own
evaluator produces the height image on the CPU — float64 sin and all,
exactly — and the GPU takes its neighbour differences from that image
by texelFetch, precisely as it would from a GPU-rendered pre-pass. The
honest cost: one height-chain evaluation over the material's pixels per
frame. The honest boundary: noise driving a COLOUR still refuses by
name — that would need per-pixel colour from the CPU, which is the
whole frame's job, not an image's.

Seam numbers, headless: **NOISE-into-Bump wavy glass under ray tracing
matches `render()` at 0.00001**, the pre-pass confirmed CPU-evaluated,
the noise load-bearing (strength moves the glass by 0.72).

### Self-test

The RAY-REFRACTED section now renders the exact Water anatomy — Blender
noise through Bump into the master Normal socket, refraction, mirror —
and prints the full shade split.

### For the field scene

Noise-into-Bump was the last named layer of Water's console trail. If
nothing else hides behind it, this is the release where the
eleven-second shade moves to the GPU.

---

## [1.25.34] — 2026-07-30

The 1.25.33 report proved the bent-ray machinery on hardware (wavy glass
at 0.000012, 0 px) — and the field scene answered with what its Water is
actually made of: **ShaderNodeBump**, height-driven, not a Normal Map.
The last refusal standing in that scene's console was an emitter gap.
This release closes it.

### Added — the Bump node shades on the GPU

`n_bump`'s exact algorithm, split the only way a fragment shader can run
it:

- **A height pre-pass per Bump node**: the height chain — textures,
  patterns, generated coordinates, whatever feeds it — renders to its
  own full-screen target over the same ids texture, writing
  `(height, 0, 0, keep)`. Alpha is the material's own coverage, which is
  exactly `n_bump`'s validity mask.
- **The main pass takes the CPU's own differences**: the +x and +y
  neighbour texels fetched by `texelFetch` with INTEGER coordinates
  (the sampler lottery stays closed; explicit edge guards because
  out-of-bounds texelFetch is undefined on a driver), gated on the
  neighbour's coverage, then
  `normalize(n − invert·strength·distance·(t·dhdx + b·dhdy)·20)` with
  the renderer's own tangent basis.
- **Hit shading treats the node as a wire** — `trace()` shades hits with
  `ctx.px` None and `n_bump` passes its Normal input through there — and
  **ray construction still bends**, through the CPU-evaluated chain the
  last release added. A Bump inside another Bump's height chain refuses
  by name.

Seam numbers, headless: a Bump-driven master material against `render()`
at **0.00003**, with the bump load-bearing (strength moves the ball by
2.19) — and the complete field shape, **Bump-driven wavy glass under ray
tracing**, at **0.00001**.

### Self-test

The RAY-REFRACTED DEFERRED FRAME section now builds its wavy glass with
the Bump chain — height pre-pass, texelFetch differences, bent rays,
refraction and the mirror, all in one frame on your driver. This is the
exact shape the field scene's Water takes.

### Noted

One honest asterisk carried from the CPU: `n_bump`'s validity mask is
the shading batch, so on the CPU a frame large enough to split into
several chunks (over ~262k fragments per material) can see gradient
seams at chunk boundaries that the GPU — which shades the whole material
in one pass — does not reproduce. At working resolutions a material is
one chunk and the two agree to the numbers above.

---

## [1.25.33] — 2026-07-30

The 1.25.32 report: refraction proven on hardware first try (0.000017,
0 px, glass and mirror in one frame) — and the field scene answered with
its next wall, the most water-shaped refusal possible: **'Water' bends
its refraction rays with a Normal chain, which the rays are built
before**. Of course it does. Water is normal-mapped waves. This release
builds the rays after the bend.

### Fixed — secondary rays bend with the Normal chain

Ray construction now evaluates the master Normal chain for exactly the
ray pixels, with the CPU's own closure code — `GraphEvaluator` through
`closure_to_surface`, the same calls `shade_batch` makes — so a
normal-mapped material builds the same reflection and refraction rays
on either device, **by construction**, not by approximation. The two
Normal-chain refusals are gone.

Honest cost note: this is the one slice of a ray-traced frame that
still runs the node evaluator on the CPU, proportional to the ray
count, not the frame. It is the price of an exact zero today; a
GPU-side bend pre-pass can replace it later if it earns a zero of its
own.

Seam numbers, headless: normal-mapped wavy glass (the Water shape —
master graph, Normal chain, flat tint, Opacity 0.4, IOR 1.33) plus the
mirror, against `render()` with ray tracing ON: **max 0.00001** — with
proof the chain is load-bearing (flattening Bump Strength moves the
glass by 0.70, so the match is not vacuous).

### Self-test

The RAY-REFRACTED DEFERRED FRAME section now renders **wavy** glass —
the normal-mapped master graph — so the next report proves the exact
field shape on the driver, bent rays and all.

### For the field scene

If Water's base colour is flat, this is the release that moves the
eleven-second shade to the GPU. The remaining named gates a Water could
still hit: a base colour that varies per pixel (the refraction tint
must be constant), a per-pixel IOR, per-pixel Specular Color on a
reflective material. The console names whichever applies, per material,
as always.

---

## [1.25.32] — 2026-07-30

The 1.25.31 report read zero — the reflected frame at 0.000048 with 0 px
off, the texelFetch fix confirmed on hardware, the one-bounce arc proven
end to end. So this release takes the next wall, and it is the one the
field scene names: **refraction**.

### Added — ray refraction shades on the GPU

The refraction half of `_add_raytraced`, term for term, on the secondary
machinery reflections already proved:

- **Rays**: per-fragment eta chosen by which side the camera sees
  (`dot(N,V) < 0` → exiting), GLSL `refract` with the total-internal-
  reflection fallback to a mirror bounce, origin stepped INTO the
  surface by the raw ray bias.
- **Hits** shade through the same secondary passes (env on, backface
  off, camera V), misses fall to `world_color` along the transmitted
  ray.
- **The blend is the CPU's lerp**: `rgb*(1−k) + hit*k*diffuse` with
  `k = (1−opacity)·refraction` — applied AFTER the reflection add,
  in `_add_raytraced`'s order. Glass and mirrors share one tree in one
  frame, and a material may be both.
- The tint needs the primary pixel's base colour as a constant, so the
  probe now remembers a FLAT base colour even for graph materials —
  which is what qualifies a real converted Water material — and a base
  colour that varies per pixel refuses by name. Per-pixel IOR refuses;
  a Normal chain on a refracting material refuses (the rays are built
  before the bend); a varying Refraction Amount refuses through the
  standard constancy probe.

Seam numbers, headless: the glass frame against `render()` at
**0.00001**; refraction changes the glass by 2.3 (no vacuous matches);
refraction OFF under ray tracing still qualifies and matches the CPU
without the lerp; texture-tinted glass refuses with the varying-colour
message.

### Self-test

A **RAY-REFRACTED DEFERRED FRAME** section: glass Ball AND mirror Box in
one frame, both sweeps, whole render both ways — max difference, >0.01
pixel count, timings, and the combined traced-sweep slice.

### For the field scene

'Water' was the one named refusal left. If its base colour is flat and
its IOR unlinked, this release moves it — and the ten-second shade —
to the GPU. If anything still refuses, the console names it, per
material, as always.

---

## [1.25.31] — 2026-07-30

**Found it.** The 1.25.30 probes ended the hunt: fresh textures flip the
same 95 (texture-state theory dead), the path is deterministic, and the
flip dumps were the confession — `cpu id 0, t 0.196` → `dev id −1,
t 1e+30`, and one `t 0`, which is a tmax read back as zero. The flipped
rays sit at indices 122, 245, 367: **once per row of the 245-wide rays
texture, every one a row-boundary-adjacent texel** (an org at x=244
whose dir wraps to the next row; an org at x=0 of a row). The `t 0` case
means the fetch landed on a neighbouring DIR texel, whose `.w` is 0 by
layout.

The mechanism: `texture()` with computed normalized coordinates can
resolve to the adjacent texel at specific (position, texture-side)
combinations — side 283 reads clean on this driver, side 245 does not.
Data-texture exactness through the sampler is a texture-size lottery,
and every measured zero this project has earned through `texture()` was
earned at sides that happened to win it.

### Fixed — compute kernels fetch data by texelFetch

Integer texel addressing, exact by specification, no sampler in the
chain. Applied to every driver-only compute source:

- the occlusion and closest-hit wrappers' RAY fetches (where the flips
  were caught), with integer index arithmetic throughout;
- the compute rasteriser's corner, bin and tile fetches — its zeros were
  real but earned at lucky sides, so it exits the lottery too, and the
  next report's DIFFERING PIXELS integer re-proves it.

The FRAGMENT variants keep `texture()`: they are what the NumPy
front-end runs, every headless proof is built on them, and the shared
BVH texel helpers stay with them — fragment-measured at zero across the
shadow and kernel sections. A new test locks the split: compute sources
must texelFetch their data and strip clean, fragment sources must stay
front-end shaped.

Expected next report: RAY-REFLECTED DEFERRED FRAME at ~0.000048 with
0 px off, the DIAGNOSIS block silent, and every existing zero still a
zero.

### Verified

Full suite green from the packaged zip — kernels, rasteriser, seams,
and the new exact-fetch guard.

---

## [1.25.30] — 2026-07-30

The 1.25.29 diagnosis came back perfectly sharp: **95 = 95 = 95**. Every
bad pixel is on the mirror, every one is a trace flip, and the trace
flips account for all of them — the secondary shading and the composite
are fully exonerated. This release acts on that number.

### The contradiction, and the one suspect left standing

The same box-pixel rays — bit-identical values — sit inside the 40,000
kernel-section rays that read **zero** mismatches on the same driver. So
the flips cannot be a function of the ray or of the tree content (the
ray-shadow frame reads the same cached tree textures across 172,800
pixels at zero). Ruled out this round, by computation: texel-centre
rounding for both ray-texture sides (worst deviation 7e-6 texels against
a half-subtexel budget of 2e-3), and ulp sensitivity (previously: zero
flips at 1e-6 jitter). What remains is the one thing the failing path
does that no proven path does: it samples texture objects in a COMPUTE
dispatch that fragment passes had previously sampled — the kernel
sections upload fresh textures that only ever see compute, and the
flips are deterministic, which fits per-object image state and not a
race.

### Fixed (most probably) — the compute trace owns its textures

`intersect_frame` no longer reuses the texture objects the ray-shadow
fragment passes bind. It keeps its own cached copies of the tree under
compute-only keys, so no texture object ever crosses the
fragment/compute boundary in either direction. Two copies of a small
tree in VRAM is the price. If the theory is right, the reflected
frame's next run reads 0 px off and the DIAGNOSIS block never prints.

### Added — and if it still disagrees, the section digs deeper

When trace flips remain, the DIAGNOSIS now also prints: the same frame
rays through the fresh-texture kernel path (separating texture state
from arithmetic conclusively), the cached path against itself
(nondeterminism check), and the first three flips as `cpu id/t → dev
id/t` — near-t neighbours would mean arithmetic; wild values would mean
misread data.

### Verified

Full suite green from the packaged zip; the reflection seams
(0.00002), the full ray stack, and the headless no-driver degradation
of the reworked `intersect_frame` all hold.

---

## [1.25.29] — 2026-07-30

The reflected frame's first hardware run disagreed: 95 pixels of 172,800
off by more than 0.01, max 0.459 — where the headless seam holds at
0.00002, and every other section (the closest-hit kernel itself included)
still reads zero. This release does not guess at the cause. It makes the
self-test find it.

### Added — the reflected-frame section diagnoses itself

When the RAY-REFLECTED DEFERRED FRAME disagrees, the section now rebuilds
the frame's exact reflection rays and prints the numbers that decide the
next round:

- **whose pixels the bad ones are** — on the mirror vs elsewhere;
- **frame-ray trace agreement** — the same rays through `bvh.intersect()`
  and through the device kernel, hit id against hit id (the 40,000-ray
  kernel sample reads zero; this is the exact frame population instead);
- if the trace disagrees: how many bad pixels are trace-flipped;
- if the trace agrees: how many bad pixels are ray MISSES (the
  `world_color` composite path) versus hits (the secondary-pass shading)
  — which splits the remaining suspects cleanly.

Ruled out already, headlessly: ulp-scale ray sensitivity (perturbing the
frame's 29,967 rays by up to 1e-6 flips ZERO hits, so this is not
silhouette float-dust), and driver-strictness of the secondary sources
(the strip-declarations lint now runs over the reflection passes too —
they come back clean, and that check stays in the suite).

### Fixed

- `intersect_frame` now degrades with a reason on a machine with no GPU
  module instead of raising — found by dry-running the new diagnosis
  path headlessly.

### Verified

Full suite green from the packaged zip, including the extended
driver-strictness test over the secondary passes.

---

## [1.25.28] — 2026-07-30

**Ray-traced reflections shade on the GPU.** The refusal this whole arc
existed to lift — "ray tracing shades recursively on the CPU" — lifts:
a frame with ray tracing ON at depth 1 now rasterises, shades, traces
its reflections and shades the hit points entirely on the driver, and
matches `render()` to 0.00002 headlessly. The self-test's closest-hit
gate passed first (0 mismatched hits in 40,000, ties decided by the
baked visit order, 80.5 ms → 1.9 ms), and this release stands on it.

### Added — one traced bounce, the whole formula

- **Rays** are built exactly as `_add_raytraced` builds them: the
  unflipped interpolated normal, the camera V, the RAW ray bias (no
  floor — the shadow branch's floor is the shadow branch's), directions
  mirroring V about N, tmax 1e30.
- **Hits** come from the closest-hit kernel already proven tie-for-tie
  against `bvh.intersect()`, through a frame-rate dispatcher that rides
  the same content-fingerprint upload cache as every atlas.
- **Hit points shade as a second G-buffer**: the hits become an ids
  texture aligned with the primary pixels, and every material in the
  MESH — including materials with no pixel on screen, probed over their
  own triangles, because a mirror can see what the camera cannot — gets
  a secondary pass. Two deliberate differences from the primary pass,
  both because they are what the CPU does: the environment term is ON
  (the depth-exhausted branch shades hits with it) and the backface
  override is OFF (`trace()` shades hits with `front=None`). The
  secondary V is the camera's — `ctx.I` is always `P − eye` on the CPU,
  hits included — so the same `hal_eye` uniform serves both passes.
- **The composite is `_add_raytraced`'s own blend**: hit colour (or
  `world_color` along the ray for misses — computed by the renderer's
  own function, so ANY world is exact there, however rich the sky) times
  reflect × specular × reflect colour, a per-material constant.
- **Ray shadows and reflections run together**: one frame can walk the
  same two BVH textures for both, and the full stack matches the
  renderer at 0.00002.

### Honesty at the edges — refusals, named, per material

Ray depth beyond 1 ("depth 1 travels: one traced bounce, then the
environment"); a refracting material under ray refraction; a reflective
material whose Normal chain would bend the rays; a reflective material
scaling its reflection by per-pixel Specular Color; screen-space shader
inputs anywhere in a reflective frame (a hit point has no screen
position — `ctx.px` is None on the CPU too). Ray tracing ON with
reflections OFF qualifies and mirrors the CPU exactly: no traced bounce
and no env term either.

### Self-test

A **RAY-REFLECTED DEFERRED FRAME** section renders the whole pipeline
both ways — ray tracing on, mirror Box, one bounce — and prints the max
difference, the >0.01 pixel count, CPU vs GPU whole-render timings, and
the trace + secondary-pass slice. The CPU side of that frame measures
~1.1 s in development; the GPU side is expected in the tens of
milliseconds. This is the section aimed at the field scene's
eleven-second shade.

### Verified

Full suite green from the packaged zip, including: the reflection seam
test against `render()` with ray tracing ON (max 0.00002, with proof the
mirror really reflects — removing ray tracing moves it by 0.53), the
full-stack RAY-shadows-plus-reflections test, every named refusal above,
and the no-reflection no-env mirror case.

---

## [1.25.27] — 2026-07-30

The reflections arc opens, on the same rhythm that carried ray shadows:
kernel first, proven headlessly to the last bit, then measured on real
hardware, then — next round — integrated into the deferred pass.

### Added — the closest-hit kernel

`hal_bvh_intersect` in `gpu/rtrace.py`: the stackless threaded walk now
answers *which* triangle, not just *whether*. It is `bvh.intersect()`
operation for operation:

- The slab test prunes against the **live best t** — as hits shrink it,
  whole subtrees stop qualifying, exactly as the CPU's `best_t[rays]`
  bound does.
- Möller–Trumbore with the identical epsilon set, strict `t < best_t`,
  and the **original** triangle id riding the packed texture's spare
  channel (float32 is id-exact to 2²⁴ triangles).
- **Ties are the point.** This is what the threaded links were built for
  back in 1.25.22: the CPU pops LIFO (right subtree first) and keeps a
  hit only when strictly closer; within a leaf, argmin keeps the first
  of equal minima. The links bake that exact visit order, so the
  sequential walk agrees on every tie by construction. The test forces
  the case outright with byte-identical duplicated triangles — several
  ids hitting at the *same* t — and the kernel picks the CPU's winner
  every time, including rays whose winner IS a duplicate.
- A miss leaves t at tmax and returns id −1, exactly as the CPU leaves
  `best_t` — the integration will read misses straight into the
  world-colour fallback `trace()` uses.

Verified headlessly at **zero mismatched hit ids and 0.0 difference in
t, u, v** — box-random rays, forced ties, and the exact reflection shape
`trace()` will cast (surface origins along `reflect(−V, N)`, tmax 1e30).

### Added — the self-test measures it on your driver

A **CLOSEST-HIT KERNEL** section: 40,000 reflection rays off real
G-buffer surfaces, `MISMATCHED HITS` as the integer that matters, t and
barycentric agreement, CPU vs compute timings. For scale: the CPU side
of that query measures ~500 ms here — closest-hit cannot early-out the
way any-hit does, which is exactly why this kernel is the one that
matters for the field scene's eleven-second shade.

### Next

Reflections into the deferred pass: trace reflect rays for the pixels
whose material reflects, shade the hit points through the same
per-material passes (the secondary "eye" is the primary surface point),
composite with `trace()`'s own blend — misses to the world colour, hits
scaled by reflect × specular × reflect colour. Single bounce first,
depth after, and the plan-level "ray tracing shades recursively on the
CPU" refusal lifts only when the whole formula travels.

---

## [1.25.26] — 2026-07-30

The viewport field report round. 1.25.25's rendered mode works — and the
report named its two flaws precisely: poor fps, and updates only when the
camera stops moving. Both trace to one design error, fixed here. The same
report's self-test also delivered the ray arc's integration proof, now
recorded where the UI reads it.

### Fixed — the viewport updates only at rest

The bug was the coalescing being too eager: every camera movement aborted
the in-flight render. And since aborts could only land between render
stages, each render burned most of its work before dying — so during an
orbit, *nothing ever completed*. The preview only updated when the camera
finally rested long enough for one render to survive. Three changes:

- **Drafts always finish.** While the view is in motion (within 0.35 s of
  its last change) the worker renders at `preview_scale × 2` — a quarter
  of the pixels — and those frames are never aborted. An orbit now
  streams coarse frames continuously instead of showing a frozen stale
  one.
- **Rest refines.** The moment the view rests, the parked draft re-kicks
  at full preview quality; a resting viewport with its full frame kicks
  nothing and costs zero. Entering rendered mode also counts as motion,
  so the *first* picture arrives at draft speed and then sharpens.
- **Aborts are back where they belong**: a full-quality refine overtaken
  by new motion (its draft is already on screen — nothing is lost), and
  anything belonging to a re-exported scene. And they land fast now:
  `render()`'s progress callback ticks **between shading chunks**, not
  just between stages, so an abort interrupts the expensive part instead
  of politely waiting for it to finish. (F12 gets the same courtesy: a
  cancelled render stops mid-shade, and the progress bar moves through
  the shade stage instead of jumping over it.)

### Fixed — viewport fps: the BVH was rebuilt every frame

A ray-shadowed or ray-traced scene rebuilt its BVH on *every* `render()`
call — ~0.2 s per orbit step on the field scene, for a tree that depends
only on the mesh. `render()` now caches the BVH on the scene object
behind a strided content fingerprint (the same idiom as every GPU upload
cache): a viewport orbit builds it once, an animation of a still mesh
builds it once, an edited mesh rebuilds it. Locked by test, including
that two renders of one scene share one tree and produce identical
frames.

Expectations, honestly stated: the preview renders on the CPU (worker
threads have no GPU context), so its ceiling is the CPU frame at draft
resolution. The real fps answer is the ray arc finishing — reflections
are the next stage — after which the heaviest scenes stop refusing the
GPU path on F12, and a viewport GPU mode becomes worth designing.

### Changed — the capability table records the driver proof

The self-test on real hardware (RTX 5060 Ti, Vulkan) measured the
ray-shadowed deferred frame at **0.000048 whole-frame, zero shadow-edge
pixels flipped in 172,800, 538.7 ms → 21.6 ms**. That was also the first
time the stackless BVH traversal ran as a *fragment* shader on a real
driver — the while-loop walk holds there exactly as it did as compute.
The raytrace capability row now carries those numbers; reflections and
AO are what keep the flag NOT_YET.

### Verified

- The viewport test now drives the draft/refine/abort state machine with
  an injected clock: first-entry drafts, rest refines to a frame that
  matches `render()` + post **exactly**, motion never aborts a draft,
  motion does abort a refine, scene exports abort everything of the old
  scene.
- Full suite green from the packaged zip.

---

## [1.25.25] — 2026-07-30

Two fronts. Ray-traced shadows move into the deferred GPU pass — the first
stage of the ray-tracing arc to ship, standing on the occlusion kernel the
last report proved at 0 mismatched rays in 40,000 on real hardware. And the
viewport's rendered mode, reported dead in the field, is rebuilt from the
ground up.

### Added — hard ray-traced shadows shade on the GPU

Shadow Method RAY no longer refuses the deferred pass. The port is
`visibility()`'s own RAY branch, term for term:

- The BVH travels as **two textures shared by every ray light** — nodes
  (bounds + the stackless traversal links) and leaf-ordered triangles —
  in the same cached-upload idiom as the shadow-map atlases, keyed on a
  content fingerprint so an unchanged mesh never re-packs or re-uploads.
- Each ray light's visibility function is emitted with the CPU's exact
  arithmetic: origin biased along the shading normal AND the light
  direction by `max(ray_bias, 1e-4)`, the ray clipped at
  `dist * (1 − 1e-3)` (1e9 for a SUN), one `hal_bvh_occluded()` call —
  the traversal already proven against `bvh.occluded()` — and **no
  density term**, because the CPU's RAY branch applies none.
- The texture sides bake into the traversal as literals; the plan
  signature fingerprints the mesh, so a changed BVH re-plans and re-bakes.
  No new uniform plumbing anywhere.
- Honesty at the edges: **soft ray shadows** (a light with radius under
  more than one shadow sample) average a random stream whose sequence the
  CPU batch layout owns — refused by name, per light. A RAY frame whose
  job carries **no BVH** mirrors the CPU exactly (fully lit, qualifies).
  Radius with `shadow_samples = 1` is the hard path on both devices.

Seam test: the demo scene under Shadow Method RAY, GPU frame vs
`render()` itself — **max difference 0.00001** headlessly, with proof the
shadows are really in the picture (removing them moves the frame by 1.6).
Run Self Test gains a **RAY-SHADOWED DEFERRED FRAME** section that renders
the whole pipeline both ways on your driver and prints the max difference,
a count of pixels off by more than 0.01 (where a flipped shadow-edge ray
would land), and the CPU/GPU timings.

The capability table now tells the ray story as it stands: shadows in,
reflections next, AO after. The blanket "ray tracing shades recursively on
the CPU" refusal still guards scenes with ray *reflections* on — that is
the next stage of the arc, not an oversight.

### Fixed — viewport rendered mode rebuilt

The field report: rendered mode draws nothing at all. The old path did all
of its work — export, render, post — **synchronously inside Blender's draw
callback**, freezing the whole UI for seconds per redraw on a real scene,
and it ran the GPU stages inside the viewport's own draw under Vulkan,
which is the prime suspect for the permanent blank: any failure fell into
a silent `return`, leaving the viewport empty forever with nothing on
screen and nothing in the console.

Rebuilt on the shape Blender's own engines use:

- `view_update` exports on the main thread (where bpy access is legal) and
  hands the bpy-free scene to a **background worker**; `view_draw` only
  blits the newest finished frame. The UI never waits on a render again.
- An orbit **coalesces**: a newer view aborts the in-flight render at its
  next progress tick and re-kicks, so the preview catches up to where you
  are instead of queueing every intermediate step.
- The worker renders **CPU-only** (device, shading, raster, post all
  forced to the CPU path for the preview): worker threads have no GPU
  context, and nothing GPU-shaped ever runs inside a draw callback again.
  The look is still the F12 look — dither, CRT, the whole post chain, at
  1/N resolution per the existing Preview Scale setting.
- **No silent blanks.** Every failure prints one `[Halcyon viewport]`
  line with the traceback, once per distinct reason. While the first
  frame renders, the region fills with the world colour instead of
  nothing, so entering rendered mode visibly did something.
- The blit is TRI_STRIP (the fan `draw_texture_2d` still uses leaves in
  Blender 6.0 and was flooding consoles with one deprecation warning per
  redraw), uploads through the buffer protocol instead of `.tolist()`,
  composites premultiplied so Film > Transparent works, and the viewport
  camera now carries the region's real PERSP/ORTHO type instead of
  assuming perspective.

The working half lives in a new bpy-free module, `preview.py`, and the
suite drives the exact loop Blender will: export in, want a view, kick,
park a frame, re-kick only when the camera or scene moved. The parked
frame is checked against `render()` + the post chain **exactly** (max
difference 0.0). What remains Blender-side — the draw idiom, the redraw
tags — is deliberately thin, and every piece of it reports rather than
blanks.

### Verified

- Full suite green from the packaged zip, 113 s: the new RAY-shadow seam
  test, the headless viewport-machinery test, the flipped RAY-mode
  negative (no BVH → fully lit → qualifies), and every existing test.

---

## [1.25.24] — 2026-07-30

The clean console from 1.25.23 named both real problems in one screen.
This is the F12 release: the difference between the GPU port working in a
report and working on the render button.

### Fixed — "No active GPU context found" on F12

`bl_use_gpu_context = True` on the render engine. The final render runs on
a render thread where Blender's `gpu` module has no context unless the
engine asks for one — so every GPU stage fell back to the CPU the moment a
real F12 ran, while twenty rounds of self-tests (operators, main thread,
context present) measured perfectly. One class attribute; the whole
distinction.

### Fixed — a GPU switch over a CPU render

The 10.7-second frame was `gpu_shading` stored OFF in the scene from an
older version, silently, under a switch that said GPU. Two changes so that
cannot recur: **flipping the device switch to GPU now turns the proven
stages on** (the Debug toggles still opt out individually, and flipping to
CPU changes nothing), and when a stage IS toggled off under the GPU
device, the console says so in one plain line each instead of printing a
six-month-old note about narrow scope. The device tooltip now describes
what GPU actually does in 2026 rather than what it did in the post-only
era.

### Noted

The field scene ray-traces — the deferred pass refuses it correctly and
says so. That is not a bug; that is the ray-tracing arc's job, already
under way: the BVH occlusion kernel is verified headless and awaiting its
first driver numbers in the next Run Self Test.

---

## [1.25.23] — 2026-07-30

A field report from a real render on 1.25.21: a console full of errors and
no picture. Triage first, honestly.

### Not ours — the KeyError that looks fatal

`Error in bpy.app.handlers.depsgraph_update_pre[6]: KeyError: 'bpy_prop_
collection[key]: key "Camera" not found'` — **Halcyon registers no
application handlers at all.** That is another add-on looking a camera up
by the literal name "Camera" and dying when a scene's camera is named
anything else. Disable other add-ons one at a time to find it (or rename
the camera to "Camera" as a stopgap). It fires on every scene change,
which is what made the console look catastrophic.

### Ours — the deprecation flood, silenced properly

- **`use_nodes`**: Blender 5.2 warns on *every read* ahead of the 6.0
  removal — ten materials times every export made real errors unfindable.
  `compat.uses_nodes()` / `compat.enable_nodes()` keep the 5.x semantics
  (a legacy material with the toggle off still renders from its flat
  settings) while suppressing only the deprecation warning, and become
  correct no-ops when the attribute disappears.
- **`TRI_FAN`**: the fullscreen quad now draws as a `TRI_STRIP`. The fan
  is deprecated and *leaves* in 6.0 — this one was a warning today and a
  broken GPU pipeline next year. The strip cuts the quad along the other
  diagonal, which changes nothing: a fullscreen quad's varyings are
  affine, so interpolation is identical however the quad is split.

### Still open — "no renders"

Nothing in the pasted console shows a Halcyon traceback, and the engine
prints one and reports to the UI on any render failure — so the actual
blocker is not yet in evidence. The next report needs: the FULL console
from one F12 (any `Traceback (most recent call last)` lines especially),
what the Render Result window showed (black, empty, never opened), and
whether the top-of-panel switch was on CPU or GPU for that render.

---

## [1.25.22] — 2026-07-30

The ray-tracing arc opens — the piece 1.26.0 waits for. First kernel:
any-hit BVH traversal, the foundation under ray shadows, reflections and
ambient occlusion alike.

### Added — the BVH occlusion kernel, and it IS bvh.occluded()

The traversal is **stackless**: the verification front-end supports
divergent while-loops but not dynamic array writes, so instead of a
traversal stack, every node carries two links precomputed at pack time —
`first` (the child visited first) and `miss` (where to go when a subtree
is done or skipped). One scalar walks the whole tree. The links bake the
CPU's *exact* LIFO order — right subtree first — which any-hit never
needs but closest-hit ties will, so `intersect` inherits the right
tie-breaks for free the day it ports.

Everything numerical mirrors `bvh.py` to the operation: the slab test
with its sign-losing 1e12 fallback, Möller–Trumbore with the exact
epsilon set. The bar stayed an integer and was met: **zero mismatched
rays** against `bvh.occluded()` on box-random rays and on the real
shadow-ray shape (surface origins toward a light, tmax at the light) —
plus structural checks that the threading links are the pop order.

### Added — the driver measures it

The self-test's compute section now ends with the arc's first hardware
number: 40,000 shadow rays from real G-buffer surface points toward the
demo scene's point light, mismatch count (an integer) and CPU-vs-compute
milliseconds. When your count reads zero, ray shadows move into the
deferred pass next — then reflections, then AO, and 1.26.0's "complete"
gets close.

---

## [1.25.21] — 2026-07-30

Round twenty-two delivered the number the whole port aimed at: **a full
`render()` call with both stages on the GPU at 19.0 ms against the CPU's
218.6 — 11.5× — at max difference 0.000048**, byte-for-byte the same
agreement the deferred pass alone has carried for ten rounds. The
rasteriser added zero error to the frame.

The 1.26.0 decision came back: *it needs to be complete — ray tracing and
all that.* So 1.25.X continues, the ray-tracing arc opens next, and this
release ships the other half of the answer.

### Added — the CPU/GPU switch, first thing in render properties

The device choice now sits at the top of the render properties panel as a
two-button switch, where a device choice belongs. **Choosing GPU now means
the GPU**: the three proven stages — rasteriser, shading, post — default
ON. Each remains a Debug-panel toggle for opting back out per stage, the
capability table still lives there, and every qualification rule is
unchanged: anything a frame uses that the GPU does not reproduce falls
back per stage with the reason printed. The full suite passes with the
new defaults — headless, every GPU path degrades to the identical CPU
frame, which is the fallback design doing exactly what it was built for.

### Next — the ray-tracing arc

The BVH is already texture-shaped: flat node arrays, contiguous leaf
ranges, precomputed Möller–Trumbore edges, and two traversal entries —
`occluded` (any-hit, shadow rays) and `intersect` (closest-hit,
reflections). The port follows the established doctrine: pack, traverse
in a kernel verified headlessly against `bvh.py` itself, then measure on
hardware. Ray shadows first, then reflections, then AO.

---

## [1.25.20] — 2026-07-30

Round twenty-one closed the race: **0 differing pixels and 3.2 ms against
the CPU's 23** at working size — the read-back fix was indeed 54 of the 56
milliseconds. The compute rasteriser has earned its place, and this
release wires it into `render()`.

### Added — GPU Rasteriser, opt-in in the Debug panel

With Device: GPU, **GPU Rasteriser** routes the opaque pass through the
compute kernel: pack, dispatch, read back, and reconstruct the G-buffer
with `fill()`'s exact conventions — including the CPU's *own* third
barycentric (the kernel's aux image now carries `b2` rather than leaving
it to `1−b0−b1`), and `+inf` depth at empty pixels, which the transparent
pass depth-tests against. Subsets (the opaque/transparent split), vertex
snap, depth precision and culling all pass through.

Strictly qualified, as everything here is: Painter's depth sort, affine
texture mode (screen-linear barycentrics not carried yet), overdraw
debugging and banded worker renders rasterise on the CPU with the reason
printed. With no driver at all, the frame is untouched — the fallback
test holds that.

**Both GPU stages can now run together**: raster on the GPU, shading on
the GPU, with the CPU left holding shadow maps, sky and resolve. The
self-test's compute section now ends with exactly that race — a full
`render()` call, both stages on, max difference and milliseconds against
the all-CPU frame.

### Housekeeping

The GPU Shading tooltip had gone stale ("normal and bump chains" listed
as not reproduced — normal chains shipped six releases ago); it now lists
the true scope and the true refusals.

### Verification

- The reconstruction locked field by field: identical tri ids, bary
  carrying the CPU's own b2, +inf empties, identical front flags.
- The no-driver fallback renders a byte-identical frame.
- Kernel exactness re-held across all six scenes after the aux change.

---

## [1.25.19] — 2026-07-30

Round twenty delivered the number the whole rasteriser arc aimed at:
**DIFFERING PIXELS: 0 of 76800** on real hardware. The kernel is fill() on
the driver — every pixel, every tie, every clipped edge — and `rasterise`
flips to BOTH in the capability table with that integer as its measurement.

The same report showed the honest cost: compute warm 56 ms against the
CPU's 12.1. This release is the diagnosis and the first two fixes.

### Fixed — the read-back preferred to_list()

`dispatch_compute` read images back through `.to_list()` — the exact
316 ms-class mistake this project documented in round two and then made
again in fresh code. It now uses the buffer protocol, exactly as
`read_target` has all along, with `to_list()` only as a last-resort
fallback.

### Fixed — the binning was a Python triple loop

The per-tile triangle expansion in `pack_raster_inputs` looped in Python
per triangle per tile. It is now fully vectorised (repeat/cumsum block
expansion, stable sort preserving submission order per bin — the order the
tie semantics depend on). Pack+bin at 480×360 measures **0.65 ms**
headless.

### Added — the self-test splits the milliseconds and races working size

The compute-rasteriser section now prints the warm split (clip+project /
pack+bin / upload / dispatch+read) and adds the decision race at 480×360 —
differing pixels and CPU-vs-compute milliseconds at the size where the CPU
spends 66 ms. The compute path earns its way into `render()` when it wins
that race; the next report says where the remaining milliseconds live.

---

## [1.25.18] — 2026-07-30

The probe came back all green on the first guess — `gpu.compute.dispatch`,
compute CreateInfo, image load/store, a correct read-back — and taught one
constraint: no storage buffers, so everything rides textures, which is the
shape the G-buffer already had. **The compute rasteriser is built.**

### Added — the kernel, and it IS fill()

`gpu/craster.py`: the CPU rasteriser's exact rules as one thread per
pixel. The design that makes exactness possible is **per-pixel sequential
resolve**: the screen is cut into 16×16 tiles, triangles are binned per
tile on the CPU (vectorised, submission order preserved), and every pixel
walks its tile's bin in order with the strict `<` depth test — first
triangle wins ties because it is literally tested first, no atomics, no
resolve pass, nothing order-dependent. Both-inclusive edge tests, the
CPU's clamped-bounding-box guard, perspective correction after the depth
decision, `l2` recomputed the CPU's way rather than as `1−l0−l1`.

One shared GLSL core, two thin wrappers: a fragment-style one the NumPy
front-end runs headlessly (per tile, so the loop bound is lane-uniform)
and the compute one the driver runs. Headless verification, the bar being
an integer: **zero differing triangle ids** across the demo scene, both
cull modes, mixed winding, exact depth ties and near-plane-clipped
geometry — with barycentrics to the ulp (1.6×10⁻⁷) and depth exactly 0.

One expectation died honestly on the way: merely *coplanar* geometry is
not a depth tie at the ulp — different corners round differently, and
both implementations agree about that. The tie test now submits
byte-identical triangles twice, and the first submission wins every
covered pixel.

### Added — the driver path and its proof harness

`device.compile_compute` / `device.dispatch_compute` wrap the exact API
shape the probe proved, and the self-test's compute section now runs the
real kernel on the demo scene at 320×240 and **counts differing pixels**
against the CPU G-buffer — an integer, not a tolerance — plus CPU
rasterise vs compute cold/warm timings. Nothing renders through the
compute raster yet: it earns its way into `render()` only after your
driver's count reads zero.

---

## [1.25.17] — 2026-07-30

The rasteriser round opens — with a study and a probe rather than code,
because the study found a decision and the decision was made deliberately.

### The finding — hardware rasterisation can never match this renderer

The CPU rasteriser's rules are now written down as the port's
specification: **both-inclusive** edge tests (`e ≥ 0` all, or `e ≤ 0` all
— shared edges are covered by both triangles and the depth test decides),
a strict `<` depth test (first triangle wins ties, in submission order),
perspective correction from per-corner 1/w after the depth decision, and
the front flag from the sign of the screen area. Hardware rasterisation
uses the top-left fill convention, its own tie-breaks and its own
interpolation — both are correct, and they disagree at triangle edges.
A hardware draw could therefore never reproduce the CPU frame, and "the
GPU frame is the CPU frame" is the doctrine every round of this port has
been measured against.

### The decision — a compute rasteriser

The port will be a **compute-shader port of the CPU's own algorithm**:
the same edge functions, the same depth rule (a two-pass atomic min —
depth first, then lowest-triangle-id at the winning depth, which
reproduces first-triangle-wins exactly), barycentrics recomputed with the
CPU's formula rather than read from hardware interpolators. Same
formulas, float32 both sides; residual disagreement shrinks to sub-ulp
edge ties.

### Added — the self-test probes your Blender's compute support

Whether the `gpu` module exposes compute shaders, storage buffers and
image load/store under your Vulkan backend cannot be checked away from a
driver — so **Run Self Test now has a GPU COMPUTE CAPABILITY section**.
It reports what the module surface offers, attempts the most likely API
shape step by step (CreateInfo compute setup → compile → dispatch and
read back a known pattern), and prints failures verbatim: a miss names
the real signature, so the next build reads it. Nothing in this section
changes how anything renders.

The capability table's `rasterise` row now states the plan and why.

---

## [1.25.16] — 2026-07-30

Round seventeen put the environment term on hardware — the reflective box
at the same 0.000020, five stacked features now measured in one small
frame. The round completed the coded-shader contract: the three refusals
left from 1.25.11 all travel now.

### Added — coded shaders sample images

A sampler uniform's socket names the image, and its prepared pixels ride
the same manual-sampler machinery as every other texture — filtered with
the scene's own filter and REPEAT wrap, which is exactly the context the
evaluator hands the program on the CPU. The texture-call rewrite runs over
the shader's own inlined functions, mangled sampler names included, so two
coded shaders with an `img` each cannot collide. A missing image breaks
the node on the CPU too (the evaluator errors), so the probe refuses those
frames itself rather than inventing a fallback.

### Added — vScreenUV and iResolution

Both bake from the frame size, which joins the plan signature so a
resolution change re-bakes. Screen coordinates derive from **vUV**, whose
orientation every agreement test since the blended-target readback has
proven — not from `gl_FragCoord`, whose y origin nothing headless can
check. `gl_FragCoord` itself refuses by name: its z is the view-space
depth on the CPU, which the fullscreen pass does not carry — the message
points at `vScreenUV` for the xy.

### Added — the front-end knows gl_FragCoord is a builtin

It parsed but could not resolve a free `gl_FragCoord` (unknown
identifier at run time). It is now bound in scope without a declaration,
exactly as GLSL defines it — zeros when the caller supplies no fragcoord,
the supplied values otherwise. Coded-shader previews benefit immediately.

### Verification

- Image-sampling coded shader under NEAREST and BILINEAR against
  `render()` (max 0.00000 / 0.00083); the mangled sampler binding
  asserted; the missing-image probe refusal; baked screen inputs differing
  across resolutions; the gl_FragCoord refusal naming vScreenUV.
- One outdated negative flipped: "a sampler uniform is refused" became
  "a sampler uniform is carried, mangled".
- The self-test scene is unchanged — five features already prove
  themselves there, and the timing history stays comparable.

---

## [1.25.15] — 2026-07-30

The INERT list — the set of surface fields that forced a frame off the GPU
just by being non-zero — is now empty in its honest cases. Studying what
the CPU actually does with each field found two refusals that were always
broader than the truth.

### Added — environment reflections travel

The `reflect` term needed the world evaluated along the reflected ray, and
for the plain NODES path that is three small formulas: a **solid colour**,
the **two-colour blend**, or an **environment texture** — equirect and
mirror-ball mappings both, sampled through the same manual bilinear
arithmetic every other image uses, so the same prepared pixels travel.
`R = reflect(-V, Nsurf)` — the bent, unflipped normal, exactly `ctx.N` at
that point — and the term lands after matcap and backface, exactly where
`shade_batch` adds it. The world spec joins the plan signature, so a sky
edit re-plans the reflective materials.

An active sky mode (BRYCE, PHYSICAL, and the rest), a world node graph and
the ground plane refuse by name. And a reflective material with
**Environment Reflection off is inert on the CPU** — it now shades instead
of refusing, which the old blanket INERT rule got wrong.

### Fixed — opacity and edge opacity under Transparency NONE

Transparency NONE forces alpha to 1.0 *after everything* — the era's
no-alpha-unit behaviour, already in the renderer — so under NONE neither
opacity nor edge opacity can reach the picture, and both now shade freely.
Under SORTED/ABUFFER the refusal stands and names what is actually
missing: alpha compositing in the deferred target. (Materials whose
*material-level* opacity is below 0.999 leave the opaque pass entirely
under SORTED, so the refusal only ever concerned graph-driven opacity.)

### Verification

- Reflection over blend/solid/equirect/mirror-ball worlds, each against
  `render()`; vacuity (the term changes the frame by 0.08); the
  setting-off inert case; Bryce-sky and ground-plane refusals by name.
- Half-opacity plus edge opacity under NONE agreeing exactly; the same
  material under SORTED refusing by name.
- One outdated negative flipped: "a reflective material is refused"
  became "…under a Bryce sky is refused" — the honesty moved, not
  vanished.
- **The self-test box now reflects the blend sky** (reflect 0.3) — the
  sphere-map term proves itself on your driver next run, alongside the
  marble, the normal map and the coded floor. Four rounds of scope in one
  small frame.

---

## [1.25.14] — 2026-07-30

Round fifteen was the cleanest report yet — cold at 22.8 ms with the
driver's disk cache finally holding every source, everything agreeing at
its established numbers. The round went to the two INERT-field refusals
that have been on the remaining-work list since round seven: **matcap**
and the **backface override**.

### Added — matcap and backface ride the deferred pass

**Matcap** is the simple one: the whole lit result lerps toward one
constant colour, after fresnel and rim and before the backface, exactly as
`apply_surface_effects` orders it. Constancy-checked and baked like the
other extras.

**The backface override** needed something the G-buffer never carried:
the rasteriser's front flag. The answer is geometry, not plumbing — for a
perspective camera, projected-winding front is *exactly* the plane-side
test against the eye, computed from the three corner positions the
attribute texture has always held. The sign convention was **measured
against the rasteriser, not derived**: `is_front` is screen area < 0,
which equals plane < 0 (and the stock demo scene turns out to be entirely
back-facing by winding, which is why two-sided lighting always mattered
there). The seam test renders a floor with alternating winding — front and
back sharing the frame — and pins the emitted expression textually so the
sign cannot quietly flip back. Orthographic cameras refuse by name (the
plane test is the perspective answer), and the camera **type** joins the
plan signature while its position stays out, so orbits still re-plan
nothing.

### Added — the MatcapUV node

Sphere-map coordinates from the view-space normal — the era's entire
environment-reflection trick, one image carrying a whole material. Both
outputs travel (Vector and Facing), and the degenerate guard tests the raw
cross product instead of normalising a near-zero vector, so the straight-
down case cannot produce NaN on a driver. The full period chain — MatcapUV
→ Image Texture → master shader — now shades on the GPU.

### Verification

- Matcap at blend 0.55, exact; mixed-winding backface at max 0.00002 with
  both facings present; the MatcapUV→texture chain exact; the ortho
  refusal by name; 56 node types emitting.
- What remains of the INERT list: `reflect` (needs the sky on the GPU),
  `edge_opacity` and `opacity` (need alpha compositing the deferred target
  does not do yet). Each still refuses by name.

---

## [1.25.13] — 2026-07-30

Round fourteen put the integer-hash claim on hardware: red marble in
generated coordinates under a normal map, max 0.000020 on the driver — the
uint32-wrap-equals-int64-mask argument is now a measurement, not a proof
sketch. Nothing needed fixing, so this round finishes what 1.25.12 started.

### Added — the pattern library completes on the GPU

The remaining **fourteen** period patterns shade in the deferred pass:
**Plasma, Ripples, Starfield, Weave, Scratches, Tiles, Brick, Spiral,
Bozo, Agate, Leopard, Onion, Bumps, Wrinkles** — every one against
`render()` itself, worst case 4×10⁻⁵. With 1.25.12's five, all nineteen of
Halcyon's procedural textures now run on the GPU, wired as users actually
wire them (generated coordinates by default).

Two seams were specific to this batch:

- **Seeded generators bake.** Ripples' source positions and Scratches'
  angles come from a seeded RNG the evaluator re-runs every frame — being
  seeded, they are per-scene constants, so the emitter runs the same
  generator once at assembly and bakes the numbers as literals. The GLSL
  never needs the generator, and both sides read identical values down to
  the evaluator's own `cos`/`sin` of them.
- **The animated ones ride `hal_time`.** Plasma (palette cycling included
  — the phase rides the raw clock, not the speed-scaled one, exactly as
  the evaluator), Ripples and Starfield's twinkle all animate through the
  per-frame uniform: a time change leaves every source byte-identical, so
  an animation never recompiles a plasma. The demoscene effect, at last,
  actually running on video hardware.

Multi-output nodes carry their extras: Weave's **Thread**, Tiles' **Tile
ID**, Brick's **Brick ID**, with the tile/brick colour composites
(variation shading included, per-pixel) mirroring the evaluator's exactly.
Halcyon's own Brick node travels — its per-brick id is the integer hash —
while Blender's `ShaderNodeTexBrick` stays refused with its sin-fract
tint named.

### Verification

- 55 node types now emit GLSL (was 41).
- Fourteen materials against `render()`, plus the plasma byte-identity
  check across a time change and its `hal_time` binding.
- The self-test scene is unchanged this round — the marble ball already
  proves the integer-hash stack on your driver, and the timing scenes
  stay comparable.

---

## [1.25.12] — 2026-07-30

Round thirteen proved the coded shader node on hardware — `Coded, Bumpy,
Box` at max 0.000021 — and `code_node` flipped to BOTH in the capability
table, its bar finally met. The round itself went to the piece every
remaining-work list has called "the bulk": procedural textures. What
decided the roster was the hash.

### Added — the period pattern textures shade on the GPU

Halcyon's own procedurals — **Marble, Wood, Granite, Dents, Crackle** —
ride the integer hash in `patterns.py`: multiply-accumulate, xorshift,
mask. That is what makes them portable at all — masking a uint32 product
to 31 bits equals masking the CPU's exact int64 product to 31 bits,
because 2³¹ divides 2³². The GLSL library (`gpu/procedural.py`) mirrors
`patterns.py` line for line and is verified primitive by primitive: the
hash is **bit-exact** over thousands of random cells; value noise, fBm,
turbulence and Worley F1/F2 agree to ≤ 2×10⁻⁵; whole marble/wood/granite/
dents/crackle materials match `render()` at ≤ 7×10⁻⁵.

The evaluator collapses every scalar pattern parameter to its batch mean,
so a *linked* scalar socket refuses by name — a varying chain would render
differently on the two paths. Colours, Vector and Scale stay per-pixel on
both sides, exactly as the evaluator keeps them.

### Added — generated coordinates travel in the deferred pass

Blender normalises Generated coordinates over each object's own bounding
box, and those bounds are per-scene constants — so they bake as two lookup
functions over the object index the tri_data texture has carried since the
G-buffer existed: `hal_generated = (P − lo[obj]) / span[obj]`, exactly
`ctx.generated`. This lifts the oldest refusal in the frame assembler: the
**default wiring of every texture node** (unlinked Vector means generated,
not UV) now qualifies instead of refusing, and coded shaders may read
`vObject` too.

### Added — the front-end learned real uint semantics

The verification backbone could not check any of the above: hex literals
with uint suffixes misparsed (`0x7fffffffu` read as 7.0 — stripping float
suffixes from a hex literal eats its `f` digits), `uint(-7)` did not wrap,
and 32-bit wrap arithmetic did not exist. The interpreter now carries uint
values in int64 and wraps at 2³² after every arithmetic op, shift and
conversion, exactly as a driver does — verified bit-for-bit against
`patterns.hash3`. This is the enabling fix for the whole procedural tier,
and it makes user-written hash shaders in coded nodes verifiable too.

### Added — Gradient, Magic and Wave

Pure arithmetic, ported whole: **Gradient** in all seven types, **Magic**
with its baked turbulence-depth loop, **Wave** in bands and rings with
sine/saw/triangle profiles. Wave refuses only when *distorted* — its
distortion runs Blender-style Perlin.

### Refused by name — the sin-fract family

Blender's **Noise, White Noise, Voronoi, Musgrave and Brick** stay on the
CPU, and the reason is printed rather than papered over: their hash is
`fract(sin(x·127.1+…)·43758.5453)`, evaluated in float64 on the CPU. A
driver's float32 `sin` decorrelates completely after that amplification —
the GPU would render a *different pattern*, which is the worst outcome
there is. The refusals point at the pattern textures, which do travel.

### Verification

- 41 node types now emit GLSL (was 32).
- Primitives against `patterns.py` through the uint-correct front-end;
  whole materials against `render()`; every refusal asserted by name,
  including the linked-scalar batch-mean rule.
- **The self-test ball is now red marble** — integer-hash pattern in the
  ball's own generated coordinates — under its tangent-space normal map,
  next to the coded floor: three rounds of scope proving themselves in one
  small frame on your driver. The 480×360 timing scene stays untouched.

---

## [1.25.11] — 2026-07-30

The capability table has called the coded shader node "the easiest piece of
the port rather than the hardest" since the table was written — its source
is already GLSL; only the CPU needs to compile it to NumPy. This round
collects that debt. Round twelve's report (the first with the normal-mapped
ball) confirmed the bend on hardware at max 0.000021 and needed no fixes,
so the whole round went to new ground.

### Added — coded shader nodes run their GLSL natively in the deferred pass

The user's source is inlined into the frame shader under per-node mangled
names — functions, structs, globals, uniforms, ins and outs alike — so two
coded shaders sharing a uniform name cannot collide. The contract around
the source is reproduced exactly as `n_halcyon_code` honours it:

- **Uniforms become sockets.** A socket's value wins over the declared
  default (`uniform vec3 c = vec3(…)` is stripped — Vulkan refuses
  initialised uniforms anyway — and the socket's constant is baked, or its
  linked chain feeds the value **per pixel** through the same emitter as
  everything else). A uniform with no socket at all falls back to the
  declared default, exactly as the interpreter does.
- **Declared `in` names bind to the renderer's varyings** through the same
  table the CPU uses: position, normal (the unflipped `ctx.N`, per 1.25.10's
  ordering), uv, vertex colour, view, incident, camera, tangent (the
  renderer's own orthonormal frame), and the clock. An `in` name the CPU
  cannot bind reads zeros there and here both.
- **`time` and `frame` ride as per-frame uniforms** like `hal_eye`, so an
  animated coded shader re-renders without recompiling: a time change
  leaves every source byte-identical, and the plan cache keeps its meaning.
- **A node whose program never compiled reads zeros** — the CPU's own
  answer — rather than refusing the frame.
- Everything outside the contract refuses by name: image inputs, matrix and
  exotic-type uniforms, `discard` (sixteen probe fragments cannot rule out
  a conditional one), `mainImage` entry points, returning a value from
  `main`, and the varyings the G-buffer honestly lacks (fragcoord,
  view-space depth, backfacing, geonormal, uv2, generated, random).
- **HLSL-flavoured GLSL refuses by name too**: Halcyon's own front-end
  forgives `saturate`/`lerp`/`frac` and friends in GLSL mode, and a real
  driver will not — "compiles in the preview, dies on the driver" is the
  worst possible seam, so it is closed here rather than discovered there.
  HLSL as a declared language keeps its existing refusal.

`code_node` left the BLOCKING list — a coded-shader scene is no longer
forced onto the CPU — and its capability row says precisely where it
stands: verified against the renderer's own compiler at 0.000004, NOT_YET
only because this table's bar for BOTH is a measurement from your driver.

### Verification

- The full contract in one material — helper function, initialised global,
  socket values beating declared defaults, a MapRange chain feeding a
  uniform per pixel, and the clock — against `render()` itself: max
  0.00000.
- Two coded shaders sharing a uniform name in one graph: mangled apart,
  both agreeing with the CPU.
- The missing-program zeros case, agreeing with the CPU's zeros.
- Every refusal above asserted by name, including the three that render
  happily on the forgiving CPU front-end and must still refuse the GPU.
- Sources strip clean for CreateInfo; the full suite passes.
- **The self-test's small deferred frame now has a coded-shader floor**
  (checker tiles from user GLSL) alongside the normal-mapped ball — the
  next Run Self Test measures both on your driver. The 480×360 timing
  scene stays untouched, so its numbers remain comparable across rounds.

---

## [1.25.10] — 2026-07-30

The wall named at the start of the round: **normal and bump chains**. The
key that unlocked it was a fact about the CPU, not the GPU — `ctx.T` is
never set anywhere in the renderer, so the Normal Map node always builds its
tangent frame from the geometric normal with `orthonormal_basis`. A
deterministic construction from data the G-buffer already carries is exactly
reproducible in GLSL, so the whole chain travels.

### Added — Normal Map chains bend the deferred normal

The **Normal Map node** has a GLSL emitter (32 of 106 node types now):
tangent space builds the frame the evaluator builds — same up-vector
selection, same cross products — and OBJECT/WORLD read the colour as a
world normal, both ending on the node's own Strength lerp. A chain linked
to the **master shader's Normal socket** is emitted by the frame assembler
through the same emitter as the colour (a texture feeding both is sampled
once, exactly as the evaluator's node cache), then bent exactly as
`closure_to_surface` does it: `normalize(N0 + (normalize(chain) − N0) ×
Bump Strength)`, with a linked or constant **Bump Strength** and the full-
strength default when the socket is absent. The probe's "the graph bends
the shading normal" refusal lifts for precisely this case and no other — a
normal bent on a BSDF lobe of a non-master graph still shades on the CPU
and says why.

The **Bump node** refuses with its reason instead of just its name: its
height input needs screen-space gradients read from neighbouring pixels
(`_screen_grad`), which a fragment shading one G-buffer texel does not
compute yet. A new `REFUSED` table carries such reasons into both the
fallback message and the UI's missing-node list.

### Fixed — the order the normal actually flows on the CPU

Porting the bend exposed a latent ordering bug in the frame shader. The CPU
evaluates node graphs against the **unflipped** interpolated normal
(`ctx.N`); two-sided lighting flips *inside* `light_surface`, testing the
**bent** normal against the view; and the tangent frame is built from the
bent normal **before** that flip. The frame shader flipped first and fed
every chain — Fresnel, Layer Weight, Geometry, and now Normal Map — a
normal the CPU never showed them, wrong on every back face, and built the
anisotropy frame from the flipped normal (a negated tangent on back faces).
`main()` now runs in the CPU's order: chains against the geometric normal,
bend, flip the bent normal, tangent frame from the unflipped bent normal.
The seam test's ordering proof is a back-facing, two-sided, normal-mapped
floor — the exact pixels where flip-first disagrees — at max 0.00002.

Also fixed while in the neighbourhood: **Normal Source FACE** now refuses
the deferred pass by name. The CPU replaces the shading normal with the
face normal *after* the graph runs — the graph still sees the smooth one —
so the frame would need two normals per fragment where the G-buffer carries
one. Before this it would have shaded smooth meshes wrong silently;
`normal_source` joins the plan signature so toggling it re-plans.

### Verification

- Tangent-space and world-space normal maps, Strength 0.8, Bump Strength
  0.65, against `render()` itself: max 0.00001 / 0.00000.
- The back-face ordering proof above, plus text-level locks that the flip
  tests the bent normal and the tangent frame reads the unflipped one.
- Both refusals (Bump, FACE) and the non-master negative, each by name.
- The new sources strip clean for CreateInfo under the driver-strict test's
  own regex, and the full suite passes.
- **The self-test's small deferred frame now wears a tangent-space normal
  map on the ball** — the next Run Self Test proves the bend on your
  driver, not just in the headless compiler. The 480×360 timing scene is
  deliberately untouched so its numbers stay comparable across rounds.

---

## [1.25.9] — 2026-07-30

Round nine was the best warm frame yet — 1.7 ms small, 6.5 ms at 480×360 —
and it also closed the case on the wandering CPU numbers: they wandered back,
so rounds seven and eight were the machine's mood, exactly as suspected and
exactly why they went unclaimed.

### Added — vertex colours travel in the G-buffer

The attribute texture's fourth slot was reserved for a tangent nothing ever
wrote. The mesh's painted corners live there now, all four components, and
with them the master shader's one remaining refusal is lifted: **Vertex Color
Mix** blends the paint over the diffuse exactly as the evaluator does — an
unlinked socket reading the mesh's own colours, an unpainted mesh reading
white — and the **Color Attribute node** reads the same slot. A node naming
some *other* colour layer than the active one still refuses, by name, rather
than quietly reading the wrong paint.

The seam test paints every corner from its position — a gradient no constant
could fake — blends it at 0.75 through the master node, and matches
`render()` at max 0.00000, with the unpainted-mesh case proving the white
fallback agrees too.

### Changed — the composite slice was mostly a redundant mask

Round nine's split showed composite as the biggest warm cost at 480×360
(3.5 ms of 6.5). Most of it was a `where()` masking the blended readback
against coverage — but the blend target is cleared to zero and untouched
pixels stay zero, so the mask reproduced what the buffer already contained.
The readback's colour planes are now taken as they are. Materials that do not
read the paint pay nothing for the new slot either: the fetch is emitted only
into shaders that reference it, so still-scene sources without paint are
unchanged and every cache keeps its meaning.

---

## [1.25.8] — 2026-07-30

Round eight held every number steady through the scope release — warm 3.4 ms
small, 8.5 ms at 480×360, agreement to the last digit — which is what a
regression-free release looks like from the outside. So this one takes the
next scope wall down.

### Added — surface parameters vary per pixel

Until now only the base colour could vary across a surface: a texture on
Roughness refused the whole material, which is a strange thing to explain to
anyone who has ever textured a material. Linked sockets on the master shader
node — Roughness, Glossiness, Specular Level, Specular Color, Metalness,
Soften, IOR, Translucency, Anisotropy and its Rotation, Toon Size and Smooth,
and Self-Illumination — now emit their node chains into the frame shader,
through the same emitter as the colour so shared subexpressions are computed
once, and the probe exempts exactly those fields from its constancy rule. One
function decides which fields are per-pixel and both the probe and the
assembler read it, so they cannot disagree.

The seam test drives roughness, specular level, specular colour and emission
from UV-derived chains under a Cook-Torrance model and matches `render()` at
max 0.00000 — below display precision — with a vacuity check proving the
variation is really in the frame (pinning the chains changes it by 0.21).

Unlinked sockets bake as constants exactly as before, so still-scene shader
sources are unchanged and every cache keeps its meaning.

---

## [1.25.7] — 2026-07-30

Round seven closed the performance chapter: plan 0.2 ms, warm shade 7.6 ms at
480×360 against a 195.6 ms CPU frame, agreement unchanged — which is also the
blended compositing path proving itself on real hardware. With speed settled,
this release is all scope, and it opens the door that mattered most.

### Added — converted materials qualify for the GPU

Every material the conversion buttons produce is built around the master
shader node, and the master shader node had no GLSL emitter — so the probe
would harvest its constants happily and the emitter would then refuse the
node, and no converted scene ever shaded on the GPU. The emitter now knows
it: the Diffuse Color chain is emitted per pixel, and every other socket
rides as a probed constant, exactly as the CPU resolves them. Vertex-colour
blending is the one thing it refuses, by name, until vertex colours travel in
the G-buffer.

### Added — rim, fresnel and sheen in the deferred pass

The era's silhouette cheats and the velvet lobe are on nearly every converted
material, and refusing them refused real scenes. When their coefficients are
constant — which on real materials they are — they now bake and run:
fresnel and rim as `apply_surface_effects` computes them, after emission,
outside the reflectance model; sheen as `light_surface` computes it, per
light, needing a light. A test proves the frame being matched visibly
contains them (removing the three changes the frame by 0.92) and still
matches `render()` at 0.00001.

### Added — area lights in the light loop

The direct math was already right: the CPU samples an area light without an
explicit surface point exactly as a point at its centre, and the softness
lives in the shadow term — where the map builder has always given area
lights a six-face cube. So an area light now qualifies instead of refusing,
and the seam test lights a converted material with sun, point, spot and a
cube-shadowed area panel at once.

Still outside, and said so by name: ray-traced shadows, fog, ray tracing,
matcap, reflection, backface override, edge opacity, per-pixel surface
parameters beyond the base colour, vertex-colour blending, and the trilinear
and N64 filters.

---

## [1.25.6] — 2026-07-29

Round six accounted for every warm millisecond — plan 4.4, pack+upload 1.0,
draw+read 2.9, composite 5.9, of 14.3 at 480×360 — and the two big slices
were both work being repeated on data that had not changed. Both are gone.

### Changed — an unchanged scene plans once, not once per frame

Planning a frame probes every material through the CPU's own closure path and
assembles the shader sources — necessary work, but scene-dependent, and it
was running per frame. Plans are now cached on a content signature of
everything the plan reads: the mesh fingerprint, every material's fields and
graph, every light's fields, the shadow maps' own fingerprints, the relevant
settings, and which textures are prepared. A camera move hits the cache — the
eye left the sources last release, which is what makes this possible — and a
material edit misses it, both held by tests. The one trade is stated in the
docstring: the constancy probes run on the first frame of a scene state
rather than every frame, and anything that would slip past that would also
have slipped past frame one's sixteen probe points.

### Changed — one readback per frame, not one per material

Each material pass read its pixels back so NumPy could mask them into the
frame — 5.9 ms of masking and copying at 480×360. The passes now blend into a
single target on the GPU: premultiplied alpha, each material writing only
where its coverage is one, cleared once, read back once, however many
materials the frame holds. If a driver objects to the blend path, the
per-pass readback path is still there and the frame completes — slower is
better than absent, and the agreement number in the next self-test would say
so.

Correctness is unchanged by construction where it could be checked headlessly
— the NumPy simulation path keeps its own compositing and still matches
`render()` on every seam test — and the blended path's agreement gets its
verdict from the next report's numbers, like everything before it.

---

## [1.25.5] — 2026-07-29

Round five confirmed the upload cache (warm frame 20.9 ms → 4.3 ms; 15.8 ms at
480×360 against a 199.7 ms CPU frame) — and its timing split exposed two
things, one of which was about to hurt.

### Fixed — a moving camera would have recompiled every shader, every frame

The eye was baked into the shader source like every other frame constant.
Correct for a still; catastrophic for an animation, because the first orbit
frame would have changed every source, and every frame after it would have
paid the driver's shader compile — the very cost the source-hash cache exists
to pay once. The eye is now the deferred pass's one per-frame uniform (twelve
bytes of push constant), and a test holds the line: two camera positions,
byte-identical shader sources, both frames matching `render()`. Nobody had
rendered an orbit yet; the timing split is what made it visible before
somebody did.

### Changed — the warm split now covers everything

Round five's split accounted for 4 ms of a 15.8 ms warm frame, which meant
the report was measuring the pipeline and not the whole function. The
self-test now prints plan / pack+upload / draw+read / composite, where plan is
the per-frame qualification pass — probing materials through the CPU's own
closure path and assembling sources — and composite is the NumPy masking that
folds each material's pixels into the frame. Whichever of those dominates in
round six is what gets attention next; the plan pass is scene-dependent and a
candidate for caching, but that is a decision the numbers should make.

---

## [1.25.4] — 2026-07-29

Round four measured the shadowed deferred pass at working size: agreement
0.000048 at 480×360, CPU full frame 194.4 ms, GPU warm shade 30.2 ms. The
stage that eats 72% of a CPU frame is beaten several times over — but 30 ms
for three full-screen passes is not shading cost, it is upload cost, and most
of it was avoidable.

### Fixed — 33 MB of unchanged shadow atlas re-uploaded every frame

The CPU caches its shadow maps across frames; the GPU path re-packed and
re-uploaded them anyway — a point light's six-face cube atlas is 33 MB at the
default size, every frame, unchanged. Uploads are now cached behind content
fingerprints (a strided sum, the same trick the shadow cache itself uses): the
shadow atlases, the mesh attribute and triangle-data textures, and image
texture pixels all skip both the packing and the upload when nothing changed.
The ids texture still uploads per frame, because the camera is in it. An
animation now re-uploads what moved and nothing else.

### Added — image textures in the deferred pass

The next scope gate after shadows. The CPU samples the *prepared* pixels —
resized, quantised, colourspace-converted at texture-prep time — so those
exact pixels travel to the GPU, and the filter arithmetic is reproduced in the
shader rather than left to the driver's undocumented sampler state:
floor-based nearest, half-texel bilinear, and all four wrap modes (Repeat,
Extend, Clip, Mirror), each matching `Texture.sample` to the texel on uvs well
outside the unit square. The filter the CPU would resolve — the node asks, the
period Texture Filter setting overrides — is baked per sampler. Trilinear and
the N64 three-point filter still refuse by name (they need a mip footprint the
deferred pass does not have), as do non-flat projections.

With that, the stock demo scene — image texture, sun and cube shadows, three
materials — shades on the GPU end to end, matching `render()` at 0.00002
under Nearest and 0.00004 under Bilinear.

### Added — the warm frame explains itself

`shade_frame` now times its two halves, and the self-test prints them: pack +
upload against draw + read back. The next report shows where the remaining
warm milliseconds live instead of leaving it to inference.

---

## [1.25.3] — 2026-07-29

Round three of the self-test confirmed where the time went: the 316 ms cold
frame collapsed to 8.3 ms once the upload stopped detouring through a Python
list, and the warm frame runs at 3.6 ms. With correctness and cost both
measured, this release goes after the biggest thing still keeping real scenes
off the GPU.

### Added — shadow maps travel with the deferred frame

Until now any frame with shadows fell back to the CPU whole, which in practice
meant nearly every real frame. The maps the CPU has already baked — it
rasterises them either way — now ride along to the GPU: each shadowed light's
depth map is packed into an atlas (one cell for a sun or spot map, six for a
point light's cube), and the shader reproduces `ShadowMap.lookup` exactly —
the light-space matrix as baked row vectors, the same linearisation of depth,
the slope bias, the normal offset scaled by texel size, every Vogel-disc PCF
tap unrolled with the softness multiplied through, the same outside-the-map
answer, the same density mix. Cube faces pick by major axis, as CubeShadow
does.

The proof is the same standard as before: a frame with sun, cube and spot
shadows in it matches `render()` — shadows and all — to a max difference of
0.00002 through Halcyon's own compiler, and the test also proves the shadows
are really in the picture being matched (removing them changes the frame by
1.69, so the agreement is not vacuous). Ray-traced shadows remain CPU-only
and say so.

### Changed — the self-test now measures the question that matters

The deferred section's small frame now carries all three shadow map kinds,
and a second measurement renders the demo scene at 480×360 — the size the
frame-breakdown section has always used, where the CPU spends 72% of its time
shading — and reports CPU full frame against GPU warm shade, plus agreement
at that size. The next report answers, with numbers, what the 72% becomes.

---

## [1.25.2] — 2026-07-29

Self-test round two came back from the RTX 5060 Ti, and both waiting stages
measured clean. This release ships them enabled, with the numbers that earned
it.

### Enabled — the NTSC stage, at 0.00037

The hardware measured the rebuilt three-draws-and-a-combine pipeline at a max
difference of 0.00037 against the CPU path — in agreement with the 0.0004 the
NumPy backend predicted, which is worth a sentence of its own: the headless
verification and the driver now vouch for each other. After a year of
UNPROVEN, composite video runs on the GPU when GPU post is on. Dot crawl is
frame-dependent and keeps any frame using it on the CPU, whole.

### Enabled — deferred GPU shading, at 0.000051

The driver reproduced the renderer's own frame to a max difference of
0.000051 over every covered pixel — sun, point and spot lights, two-sided
lighting, flat and smooth normals, all three materials. The capability table
now says BOTH with the number attached, and the GPU Shading switch is a
measured choice rather than an experiment. It stays opt-in: its scope is
still narrow (no shadows, fog, image textures, area lights, or per-pixel
surface parameters beyond the base colour), and every frame outside that
scope shades on the CPU and says why.

### Fixed — the first GPU frame paid for a Python list

The measured cold frame took 316 ms against the CPU's 31. Most of that is the
driver compiling three 500-line shaders, which is per scene and cached — but
a real slice was `upload()` converting every G-buffer texel into a Python
float object via `.tolist()` before handing it to the driver. Blender's
Buffer accepts anything with the buffer protocol, and a NumPy array is one;
the upload is now a straight copy, with the list path kept as a fallback for
older builds. The per-material offscreen target is also created once per
frame rather than once per pass.

### Changed — the self-test reports cold and warm frames separately

A first frame compiles shaders; every frame after it does not. The deferred
section now renders twice and prints both times, because the warm number is
what an animation pays and the cold number is what a still pays once. Next
report will show where the remaining time actually lives.

---

## [1.25.1] — 2026-07-29

The first self-test report from real hardware came back for 1.25.0, and it was
a good one: all four enabled post stages reproduced their claimed numbers to
the last digit on an RTX 5060 Ti under Vulkan, and the frame breakdown
measured shading at 72.2% and rasterising at 9.6% — the 71/9 split the whole
deferred-shading plan rests on, confirmed. Two things failed, and both are
fixed here. (This line also begins the policy that the project stays in
1.25.x until the GPU port is complete on all fronts; 1.26.0 is reserved for
that.)

### Fixed — the driver refused every deferred shader, over a comment

`strip_declarations` removes the global `uniform`/`in`/`out` lines from a
generated shader before handing it to Blender's CreateInfo, which declares
those resources itself. It required the semicolon to *end* the line — so the
four G-buffer declarations that carry trailing comments slipped through, got
declared a second time by CreateInfo, and the driver rejected the whole
shader with an unhelpfully generic error. Halcyon's own front-end tolerates
redeclaration, which is precisely why every headless check passed while every
frame on real hardware failed.

The strip now parses past comments, the post stages' own stripper delegates
to it rather than agreeing to disagree, and a new test holds the strictness
the driver holds: no stage body and no generated material pass may leave a
single declaration standing, every baked literal must carry its decimal
point, and no generated identifier may be a GLSL reserved word. That test is the stand-in for a real
driver on this machine — every class of rejection hardware has actually
produced is now checked on every run.

### Fixed — the NTSC measurement skipped itself

The self-test lifts the enabled-stages gate to measure the unproven NTSC
pipeline, but the orchestrator also checks the user's own GPU Post switch —
which the self-test never turned on. It is on for the duration of the
measurement now. Round one reported SKIPPED because of exactly this.

### Added — the self-test bisects a refused shader

A driver's "Shader Compile Error" names nothing. When the deferred frame
fails to compile, the report now compiles the same source in four layers —
reflectance models, the dispatch, the G-buffer reader, the full material
pass — and prints which layer the driver refused, so the next report
localises the fault even without the console log.

---

## [1.25.0] — 2026-07-29

### Added — deferred GPU shading, the second stage of the port

The capability table has said it since it was written: shading is about 71% of
a frame, rasterising about 9%, and the CPU's G-buffer already holds everything
shading needs. The upload and the draw were the missing pieces, and they now
exist. With **GPU Shading** on (Debug panel, Device set to GPU), the CPU
rasterises as always, the G-buffer is packed into three textures, and the
frame is shaded in one full-screen pass per material on Blender's own `gpu`
module — the same mechanism the measured post stages run on.

Some structure worth naming:

- **Every frame constant is baked into the shader source** — surface values,
  lights, the eye, the ambient — so the only uniforms are the three G-buffer
  samplers and their sizes. Vulkan's 128-byte push-constant budget never
  enters into it, and an unchanged scene compiles its shaders once, however
  many frames it renders.
- **The surface constants are not re-derived.** Each material is probed
  through `closure_to_surface` — the renderer's own code — on fragments taken
  from the actual frame, so what the GPU bakes is what the CPU would have
  used, by construction.
- **Qualification is explicit and it talks.** A frame using anything the GLSL
  does not reproduce yet — shadows, fog, image textures, ray tracing, area
  lights, light linking, a surface parameter that varies per pixel beyond the
  base colour — shades on the CPU exactly as before, and the console says
  which material or feature disqualified it, by name.
- **The whole pipeline is verified without a GPU**, by running the same
  shader sources through Halcyon's own GLSL front-end against the renderer's
  actual output: max difference 0.00002 across the test frame, through
  packing, reconstruction, probing, three light types, two-sided lighting and
  flat-versus-smooth normals. The one thing this machine cannot execute is
  the driver, so the stage defaults off until **Run Self Test** — which
  gained a *GPU Deferred Shading* section that renders a real frame both ways
  on your hardware — reports the same numbers back.

### Fixed — the specular colour was applied twice, sometimes

The CPU folds the specular *colour* into the reflectance model's own term —
which is what lets Metal tint its highlight by the diffuse, as 3D Studio's
Metal did — and the GLSL shading path multiplied by the specular colour again
in its light loop. Invisible for a white specular, which is what every test
used; wrong for a coloured one, and doubly wrong for Metal and Strauss. The
dispatch now folds the tint exactly as the CPU does, and the test that should
have caught it now uses a coloured specular.

### Fixed — the GPU NTSC stage was the wrong shape entirely

It blurred I and Q with one 13-tap triangle at one radius. The CPU path — the
one every composite picture actually went through — blurs I at one radius and
Q at twice that, because chroma bandwidth on a composite cable was not one
number, and it blurs with a box run three times, re-padding the frame edge
before every pass. No single pass can reproduce that; the stage is now three
blur draws and a combine, orchestrated like the CPU's own passes, and agrees
with the CPU path to 0.0004 through the NumPy backend. It stays disabled until
a driver measures the same — but for the first time it has the right shape to
be measured against.

### Changed

- The Debug panel's capability list now tells the truth about the deferred
  stage: written and verified, off until measured on hardware.
- The self-test's GPU stage table runs the NTSC measurement through the real
  multi-pass orchestrator, and the deferred-shading section reports covered
  pixels, max/mean difference and CPU-versus-GPU frame times.

---

## [1.24.0] — 2026-07-29

### Added — the comets move

Bryce put comets in the Celestial tab and Halcyon drew them, but they were
nailed to the sky: the same streak in the same place on every frame of an
animation. Each one now runs around a great circle of its own at **Comet
Speed**, at its own pace, off the scene's own clock — so the same frame number
gives the same sky on any machine that renders it. They all start where a still
frame put them, so turning the speed up never empties frame one.

Four more controls came with it, because everything about them had been a
hard-coded random number: **Tail Length**, **Tail Width**, **Comet Colour**,
and **Tail Direction**. That last one is the interesting one. A comet's ion
tail is blown straight away from the sun and its dust tail trails its own path,
so the two disagree and the honest answer is a mix; the slider is that mix.
Which matters more than it sounds, because the sun vector had been an argument
of the comet function since the day it was written and was **never once read** —
the tails pointed wherever the random number generator sent them, including at
the sun.

### Fixed — comets were drawn as lines, not comets

The streak was bounded at the far end only, so it ran on *in front of* the head
as far as a cosine falloff allowed — around forty degrees of sky. Two of them
crossing drew a plus sign. A comet is now a compact coma at the head with a
tail behind it and nothing in front, and the tail flares and dims as it goes,
which is the shape a dust tail actually has.

### Added — 20 water presets, and a library to keep them in

Bryce kept its waters in the Materials Library under **Waters & Liquids**, and
dropping one onto the water plane was how a Bryce picture got its sea. The
infinite ocean now has the same thing, built the same way as the sky library:
a dropdown, an explicit **Apply Preset**, *Save As...*, *Add to Library* and
*Import Preset...*, in a `.halwater` file that is a flat dict of field names
and so still loads when a later version adds fields to it.

The twenty: **Bryce Water**, then the tropics — **Caribbean Resort**,
**Tropical Shallows**, **Calm Lagoon**, **Mediterranean**; the open water —
**Open Ocean**, **Rolling Swell**, **Choppy Bay**, **Deep Atlantic**,
**Storm Swell**, **Arctic Sea**; inland — **Millpond**, **Mountain Lake**,
**Swimming Pool**, **Black Lagoon**, **Glacial Melt**; and light on water —
**Moonlit Water**, **Sunset Ocean**, **Liquid Mercury**, **Alien Sea**.

Same honesty as the skies. Two things here are Bryce's and are used
deliberately: the category name *Waters & Liquids*, and **Caribbean Resort**,
which a period tutorial places second along the top row of it. Every other name
is the obvious name for the thing rather than a claim about what Bryce shipped,
and all twenty are waters built out of Halcyon's own controls. No test passes
unless every one of them applies, renders, survives a save and a load exactly,
and lands on a different picture from all the others.

### Fixed — picking a sky threw your water away

The sky library reset *every* World field it did not explicitly exclude, and
the infinite plane was not excluded. Applying a sky therefore silently returned
the water to defaults — colour, waves, glitter, all of it — which is a hard
thing to notice and a harder one to explain. The two libraries now own disjoint
halves of the World and a test fails if that ever stops being true, so a sky
cannot touch the water and a water cannot touch the sky.

### Changed — skylight on water multiplies instead of adding

Water is lit by the whole sky and not only by what it mirrors, or the troughs
go black. That contribution was being *added* as a flat fraction of the zenith
colour, which made it a floor: the darkest waters could not be dark, and a
black lagoon under a blue sky came out the same mid blue as everything else. It
now multiplies the body colour, because what comes back up is skylight the
water scattered and water that scatters nothing returns nothing. Mid-tone water
lands within a few percent of where it was; dark water is finally dark.

---

## [1.23.1] — 2026-07-29

### Fixed — the waves stopped before the horizon

Reported against 1.23.0, and correctly: the water still lost its waves with
distance, and the ocean's own Distance Fade would not bring them back because
that is haze and this was not.

It was the Nyquist fade. A wave narrower than the pixel it lands in cannot be
drawn cleanly, only aliased, so it was being faded out — which is what a modern
renderer does and is why distant water goes smooth. Bryce did none of it. Its
ocean was a procedural water material on an infinite plane, evaluated per pixel
with nothing filtering it, so the waves ran all the way to the horizon and
compressed into a band of shimmer instead of flattening into glass. That
shimmer is not an artefact of the reproduction, it is what the pictures look
like, and smoothing it away is the less accurate of the two options.

**So the fade is off.** It is now a control — *Horizon Smoothing*, defaulting
to zero — for anyone who wants the smooth version, and everything it removes
still goes into widening the glitter path and roughening the reflection as
before. At zero, which is the default, every wave train is drawn at every
distance.

### Added — Horizon Shimmer

Keeping the waves to the horizon exposed the reason the fade was there. Where
one pixel covers many wavelengths, sampling the middle of it makes the wave
trains beat against the pixel grid, and the far water fills with moiré fringes:
regular, diagonal, and unmistakably a rendering artefact rather than a sea.
Bryce did not have them, because a noise field undersampled gives speckle and
a sine wave undersampled gives fringes.

*Horizon Shimmer* takes the sample from a fixed random point inside the pixel
rather than its centre. The contrast is untouched — the water is neither
brighter nor darker, and a still frame renders identically twice — the detail
is only decorrelated between neighbours, so the same waves read as the fine
shimmer a Bryce ocean has. On by default; turn it down to see the fringes it
is replacing.

---

## [1.23.0] — 2026-07-29

### Fixed — text objects took the whole frame down

Adding a text object to a scene killed the render. Blender hands a *mesh*
object geometry the depsgraph already owns, so freeing the conversion
afterwards frees nothing and the old export ordering — convert every object,
then read them all — got away with it for the entire project. Text, curves,
surfaces and metaballs are different: they *build* a temporary mesh, and the
next conversion of the same object destroys the last one. The first object
that is not a mesh is therefore the first object that could ever hit it, and
reading freed geometry is a crash rather than an exception.

The conversion now happens one object at a time and each temporary is released
on the way to the next, which is the only ordering that is safe for all five
types. It also stops the exporter holding every converted mesh in memory at
once, which a heavy scene was paying for.

Three things around it, so that this class of fault cannot cost a frame again:

- **One object that will not convert is now skipped, not fatal.** It is named
  in the Blender info bar and its traceback goes to the console, and the rest
  of the scene renders. Losing an object is a problem; losing the frame and
  being told nothing about which object did it is a worse one.
- **A text object with no faces says so.** The most common reason text is
  invisible is that it converted to outlines — no Fill Mode, no extrude — and
  Halcyon now names the object and says which setting to reach for instead of
  rendering an empty picture.
- **Grease pencil, hair curves, point clouds and volumes are named as
  unsupported** rather than silently missing from the render.

### Fixed — the ocean's waves could not be made smaller

Two faults, and between them they cut most of the wave trains out of every
picture: what was left read as a few big smooth swells near the camera and
glass everywhere beyond them, and turning Wave Scale down pushed the rest
under the same cut and flattened the sea entirely.

**The pixel footprint was a guess.** Waves smaller than the pixel they land in
can only alias, so they are faded out — the right thing to do, but it was being
measured against a hard-coded 0.002 radians per pixel that nothing ever wrote.
A 1080-row frame at a 50 mm lens is 0.0006, and half that again at 4x
supersampling. The number is now taken from the actual projection and the
actual rendered height, so a frame rendered larger resolves finer water, which
is what it should always have done.

**And it was measured along the wrong axis.** A ray that grazes the water is
stretched a long way *along* the view but stays narrow *across* it, and a wave
train running across the view is perfectly resolvable at that distance. Taking
the long axis over-blurred by a further ten times near the horizon. It now uses
the area-equivalent square, which is what a pixel actually covers.

Together, at sixty metres out: one wave train survived before, four of five do
now.

**What still cannot be drawn now roughens the water instead of vanishing.**
Slope too fine to resolve is still slope — it scatters what the water mirrors
rather than reflecting it cleanly. That already widened the sun's glitter path;
it now also softens the reflection toward the colour of the sky at the horizon,
so distant water reads as water seen from far away rather than as a sheet of
glass.

### Changed — Wave Scale is now Wave Size, and it is a length

It used to be a multiplier on the ground plane's **Scale** — the size of the
chequerboard squares, which is not drawn under water and has nothing to do with
it — so the same value meant a different sea in every scene. It is now the
length of the longest wave train, crest to crest, in metres, and nothing else
feeds into it. The range runs from 2 cm to 500 m.

Existing scenes will find their water about half the size it was at the default
ground Scale of 2; set Wave Size to twice its current value to get exactly what
you had. The foam speckle follows the wave size now too, rather than the
chequerboard.

### Changed — wave trains no longer start in step

Every train began at phase zero, so they all peaked together at the world
origin and the sea read as corrugated iron. Each train now starts wherever the
sky's seed puts it. It was invisible while the waves were too big to see and
obvious the moment they were not.

---

## [1.22.1] — 2026-07-29

### Changed — the sky library asks before it acts

Three things about the preset system, all of them about not doing anything you
did not ask for.

**The library only appears under the Bryce sky.** It was always Bryce's
library — every field it writes is a Sky Lab field — but it sat in the panel
under Solid and Gradient and Physical too, offering to load skies that those
modes have no way to show. It is now drawn only when the sky mode is Bryce, and
it no longer switches the mode out from under you when you apply something.

**Picking and applying are two separate acts.** The presets are now a dropdown
that only holds a selection, plus an **Apply Preset** button that writes it into
the world. Scrolling through the list to read the names no longer rewrites your
sky forty-three times on the way past. The dropdown lists the built-in skies
first and any you have saved to the library underneath, under a *Saved Skies*
heading.

**Import Preset adds a file to the library and stops there.** The old Load File
button read a `.halsky` off disk and immediately applied it. It now copies the
file into your sky library, selects it in the dropdown, and leaves the world
alone — press Apply Preset when you actually want it. If a file of that name is
already in the library the copy is given a numbered name rather than replacing
what is there, and the file is checked for being a Halcyon sky *before* anything
is written to disk, so a wrong pick leaves no trace.

---
## [1.22.0] — 2026-07-29

### Added — 19 more skies, 43 in total

Six asked for by name in pink and purple, thirteen more built from names alone.

**Rose Tinted Glasses** — everything a shade warmer and kinder than it was,
with the haze in on it. **Valentines** — red at the horizon into rose into deep
pink, the clouds catching all of it. **In The Land of Vapor** — pink one way
and cyan the other, pastel and soft-edged. **Vaportrails** — high thin streaks
pulled right across a lilac sky and nothing below them, which is the stratus
deck squashed to five and sharpened. **Synthetic Wonderworld** — hot magenta at
the deck, indigo overhead, one enormous low sun; the airbrush school, rendered.
**See It To Believe It** — a sky nobody would accept as a photograph: magenta
banding, a corona too wide for its sun, and a bow over the top.

Then: **Afterlife Day** and **Afterlife Night** as one place twice, the day
blown out and lilac-shadowed, the night a violet dark that glows rather than
falls. **Alien Planet** — violet air over a teal horizon and a small hard white
sun, kept distinct from the existing green Alien Sky. **Abstract Movie** — flat
saturated bands and clouds with no soft edge left in them; a title sequence
rather than a place. **Old Age Photograph** and **Stone Age Photograph** — sepia
and low contrast, then older and colder, a plate rather than a print. **Amber
Lamps** — sodium light on the underside of a low overcast and no sky past it.
**Sapphire Imagination** — jewel blue all the way down, lit from inside. **Beyond
The Rainbow** — past where the colours stop making sense, a full bow and a
secondary and a sky already doing the same thing. **Halloween** — a big low
orange moon with ragged cloud crossing it. **Christmas** — cold blue dusk over
snow, the light going and the first stars up. **Easter** — pastel and evenly
lit, the sky as a sugared egg. **Crystal Stars** — air with nothing in it, every
star out and a violet wash where the sun went.

### Note on the tuning

Five of them came out of the first pass as white paper. The cause was the same
each time and it is worth writing down: **haze density above about 0.6 with a
blend toward the sky will eat any gradient underneath it.** A pastel sky and a
heavy haze are the same instruction twice, and the haze wins. They were retuned
by pulling the haze back to where the dome could be seen through it, not by
darkening the colours.

The test that no two skies land on the same picture now runs over 43 of them,
which is 903 pairs and the point at which that test starts being worth having.

---

## [1.21.0] — 2026-07-29

### Added — a sky library, and a file format for skies

Bryce's Sky & Fog palette kept a library of skies and a row of memory dots to
drop them onto, and half of using Bryce was starting from one of those and
pushing it somewhere. A Sky Lab with sixty controls and no library is a Sky Lab
nobody opens twice.

**24 skies**, in the World panel under *Sky Presets*: Bryce Default, Dawn,
Sunrise, Morning Haze, High Noon, Desert Noon, Tropical Afternoon, Golden Hour,
Sunset, Dusk, Moonlit Night, Deep Night, Clear Blue, Mackerel Sky, Overcast,
Storm Front, Fog Bank, After the Rain, Sun Through Cloud, Mars, Alien Sky, Ice
World, Ashfall and Deep Space.

Each is built out of the Sky Lab's own controls, so every one of them is also a
worked example of what those controls do together — Deep Night is the Celestial
tab with everything on, Sun Through Cloud is Volumetric World, Fog Bank is the
fog base height doing the thing base heights are for.

**Save Sky** writes a `.halsky` file, **Add to Library** drops one into
Halcyon's own preset folder so it appears in the list beside the built-in ones,
and **Load File** reads one back. The format is a flat JSON dict of field names
to values, which is not laziness: a sky saved by a later version still loads
here, minus whatever fields this version has never heard of, and it says how
many it had to leave out rather than failing or lying.

A sky file carries the sky and nothing else. Node trees, image datablocks, the
world's Strength and the render-side mist settings are excluded by name, and a
test asserts none of them ever appears in a saved file.

### What these presets are, and are not

They are skies built to Bryce's controls and tuned to the conditions they are
named for. **They are not Bryce's own preset files.** Those shipped inside the
application, there is no published list of them, and I could not verify one —
so presenting these as Bryce's would be a claim I cannot back. Where a name is
one Bryce itself used for a category — dawn, storm, alien — it is used here
because it is the obvious name for the thing, not because the numbers came from
Bryce.

### The tests

Applying every sky and looking at the contact sheet is how they were tuned, but
it is not what is asserted. Three things are: every preset applies and renders
finite; **no two of them land on the same picture**, which is how a copy-paste
in a library of twenty-four would otherwise go unnoticed; and every one
survives a save and a load with every field unchanged. Plus the awkward cases —
a file from a later version, a file that is not a sky, a file that is not JSON —
each refused or handled rather than half-applied.

Presets also reset before they apply, for the same reason the render presets
do: without it they accumulate, and the tenth sky you try is nine skies deep. A
test applies Ashfall, then Clear Blue, and checks the result is field-for-field
identical to Clear Blue on a fresh world.

---

## [1.20.1] — 2026-07-29

### Fixed — the clouds raced when the camera moved

Reported straight after 1.20.0. Two things were wrong and they compounded.

**Link Clouds to View shipped defaulting to off.** That control exists to stop
the cloud pattern changing as the camera moves, and Halcyon's behaviour before
1.20.0 was equivalent to having it on. Shipping it off silently switched every
scene to world-locked clouds. It is on by default now, which restores exactly
what the sky did before — a camera at any position gets a **bit-identical**
sky, and that is what the test asserts, because a sky that does not depend on
the camera cannot depend on it a little.

Worth naming why "the camera rotates" was enough to trigger it: orbiting a
viewport *translates* the eye around a pivot. It reads as a rotation and it is
a move.

**And the parallax was unweighted.** The camera offset was added flat to the
dome projection, so the whole sky slid — the horizon along with everything
else. A cloud overhead is at the deck's height and genuinely swings past you as
you move; a cloud on the horizon is effectively at infinity and does not move
at all. The offset is now weighted by how steeply the ray looks, so parallax is
greatest overhead and vanishes at the horizon. Measured on the same camera
move: 0.121 overhead against 0.024 at the horizon, where before it was uniform.

There is a second-order reason the old behaviour was so violent. Cloud Height
is a *dome parameter*, not a distance — the default is 1.0 — so world-locking
against it means a one-unit camera move is a whole deck-height of parallax.
That is geometrically right and practically useless, and it is now said in the
control's own tooltip.

### Fixed — the water reflected a sky the camera was not under

The ocean evaluated the sky for its reflection from the origin, while the sky
above the horizon was evaluated from the camera. With camera-dependent clouds
switched on, the two disagreed: the water mirrored a sky that was not there.
The camera position is carried through to the reflection now.

---

## [1.20.0] — 2026-07-29

### Added — the Sky Lab's own controls, under the Sky Lab's own names

Thirty-two more world settings, taken from what Bryce's Sky & Fog palette and
Sky Lab actually shipped rather than from what seemed useful. Where Bryce had a
name for something, that is the name it has here.

**Sky Mode.** Bryce's palette offered Soft Sky and Custom Sky. Custom is the
three gradient stops as set. Soft derives the horizon from the sun's own glow
colour and leaves only the dome to the user, which is why every default Bryce
sky warmed toward the sun without anybody choosing to make it.

**Sun Glow Colour** as its own swatch, separate from the sun's light colour —
Bryce kept them apart and the corona takes the glow one. **Shadow Colour** and
**Shadow Intensity**, which tint the shaded side of a cloud. **Softness** on the
moon's terminator, next to the phase, because a hard one reads as a cut-out and
a soft one as a sphere.

**Comets**, with an intensity and a count. They were in the Celestial tab and
they are the reason a Bryce night sky was never just a starfield: each is a
great-circle streak with a bright head and a tail that thins along it.

**Frequency, Amplitude and Turbulence** on both cloud decks, which are Bryce's
three cloud controls under Bryce's three names. Amplitude is the interesting
one — it swings the noise about the *cover threshold* rather than about zero,
so raising it separates the billows without changing how much sky is covered,
which is exactly what the control did.

**Spherical Clouds**, **Link Clouds to View** and **Fixed Cloud Plane**. The
first rolls the dome projection off toward a sphere so clouds stay puffy at the
horizon instead of smearing into streaks. The second keeps the pattern still as
the camera moves. The third measures the deck's height from the camera rather
than the ground, so climbing never puts you inside it.

**Base Height** for fog and for haze, and **Blend With Sun** and **Blend With
Sky** for fog as well as haze — the Sky Lab's Atmosphere tab has all four, and
the base height is what lets a fog bank sit above the camera instead of always
hugging zero.

**Colour Perspective**, the rate at which distance takes the haze colour, and
**Volumetric World**, which lights the haze along the rays that reach the sun
through a break in the deck. The shafts are proportional to the haze that is
actually there — a shaft needs something to scatter off, and without that the
control just floods the frame, which is what the first attempt did.

### Fixed — the sky was stacked in the wrong order

Bryce's Sky Lab layers in a fixed order and the order is half of why a Bryce sky
reads as one. Halcyon had stars composited *over* the cloud decks, and haze
applied *before* them — so stars shone through solid cloud, and clouds at the
horizon stayed crisp against a sky that had hazed over behind them.

It now goes: sky dome, then everything beyond the atmosphere (stars, comets,
the sun or moon), then the cloud decks in front of those, and the atmosphere
last, because haze and fog are between the viewer and all of it. Two tests
check it rather than a picture: a solid deck must reduce the variance of the
pixels that had stars in them, and haze must dim the clouds and not only the
sky behind them.

### Added — the ocean, rebuilt

The old water crossed four sine trains at fixed angles, which reads as
corrugation. Real waves run mostly *with* the wind, with the shorter ones fanned
out either side of it, so there is now a **Wind Direction** and a **Spread**:
zero gives a regular swell, one gives confused chop. **Wave Detail** sets how
many trains are summed and **Wave Scale** their base size.

**Sun Glitter** is the piece that was missing, and it is what makes a Bryce
water picture. It is not a highlight on a plane — it is the sun found in the
*distribution* of wave normals, so it widens with the chop and tightens as the
water calms.

Which leads to the part worth writing down. Waves smaller than the pixel they
land in cannot be drawn, only aliased, so they are faded out with distance —
and fading them to nothing is what turns distant water into glass. **A wave too
small to draw still tilts the water inside its pixel.** So the slope that gets
faded out is measured and folded into the glitter's width instead: the
highlight broadens toward the horizon rather than disappearing, which is the
glitter path spreading away from you across the water.

Also **Deep** and **Shallow** colours with the path length between them,
**Transparency**, and an optional **Foam** on the crests. Foam is off by
default and says so in its tooltip: Bryce had no foam control, and adding one
unasked would be inventing a feature it did not have.

### On "1 to 1"

It is worth being straight about this. Without Bryce's source there is no way
to be bit-identical to it, and anything claiming otherwise would be guessing
with confidence. What this is: every control the Sky Lab and the Sky & Fog
palette are documented as having, under the same names, grouped into the same
tabs, layered in the same order, each one doing what the documentation says it
does. Where a control's *internal* formulation is not documented — cloud
amplitude, the glitter's width, the shaft falloff — the behaviour is
reconstructed from what the control is described as doing and tuned by eye
against Bryce output, and those are the places this is a faithful
reconstruction rather than a copy.

A test perturbs all thirty-two new controls one at a time and fails if any of
them leaves the picture unchanged. Four of them did, on the first pass.

---

## [1.19.0] — 2026-07-29

Reported: the Wireframe model and the render passes still do not work. Both
were fixed in 1.18.0 by reasoning rather than by reproducing, and reasoning
lost. This release is mostly about not doing that again.

### Added — a fake Blender the add-on can actually be run against

`tests/fakebpy.py` proved the modules import and register. It caught typos and
bad enum defaults, and it caught nothing whatever about whether the engine
works, because the engine was never executed: **six bugs shipped through it in
a row**, every one a control wired up at one end and not the other.

`tests/fakeblender.py` goes one level further. It builds a scene — an object
with a real mesh, a material with a node tree, a light and a camera — hands it
to `HalcyonRenderEngine.render()` through a fake depsgraph, and captures what
comes back out of `begin_result`. That is the whole path: the property group,
`to_settings`, the exporter, the renderer, the post chain, the delivery, the
passes.

It is not Blender and never will be; it cannot catch a segfault, a driver quirk
or an RNA lifetime bug. What it catches is the thing that kept getting through:
a setting that never arrives.

The first thing it did was render every pass in the dropdown through the real
engine and show all six of them arriving correctly — which is a finding, even
though it is not the one that was wanted. See the note at the end.

### Fixed — the Wireframe fill, which was arithmetic rather than a bug

Reproduced at last, and it is worth stating plainly because it is not what it
looked like. Measured at 120x90:

| triangles | pixels per triangle | fraction of the object drawn as wire |
|---|---|---|
| 32 | 169 | 20% |
| 288 | 19 | 43% |
| 1,152 | 4.7 | 52% |
| 3,200 | 1.7 | **53% — the whole object** |

Once triangles are a couple of pixels across, **every** pixel is within a wire
width of an edge, so a wireframe of every triangle edge *is* a solid fill. No
width setting escapes that; it is the same arithmetic that made the wire look
broken on a dense mesh and fine on a cube.

Three things came out of it:

**Creases & Silhouette.** A new wire mode that keeps only the outline and the
edges where the surface actually turns by more than a set angle. It is one
pixel wide however many triangles are behind it — measured at 1.9% of the frame
on the same object at 32, 288, 3,200 and 12,800 triangles, a spread of nothing.
All Edges stays the default, because that is what these renderers did.

**Wire Size, reachable.** It was exported only inside the material-override
branch, and a material shaded as Wireframe by a Halcyon Shader *node* never
goes through that branch — so its wire width was stuck at the default with no
control anywhere in the interface that could change it. It is now on the shader
node, shown when the model is Wireframe, and read from there.

**A Wireframe panel.** `render_wire`, `wire_width` and `wire_color` had no
panel at all: three settings the renderer read and nothing could set. Render
Properties ▸ Shading ▸ Wireframe.

### Changed — the edge distance is computed, not estimated

The screen-space distance to a triangle's edge came from a finite difference of
the barycentrics. It is now solved directly from the triangle's own projected
vertices: for a point with barycentrics l0,l1,l2 in a triangle of screen area
A, the distance to the edge opposite corner i is `|l_i| * 2A / |e_i|`. Exact,
at any triangle size, at any resolution, with no neighbouring pixels involved
at all — which removes the entire class of failure the previous two attempts
were chasing.

### Note — the render passes

Every render pass in the Debug panel's dropdown is produced correctly through
the full engine path in the new harness: with each of four presets applied,
with the GPU device selected, with worker processes on, at 4x supersampling and
at 3x pixel scale. I could not reproduce the failure, and I would rather say
that than ship a third fix I cannot verify.

So this release makes the next report conclusive instead. Every render now
prints its version, the requested pass and the wire mode to the system console:

    [Halcyon] 1.19.0 rendering 1920x1080  pass=DEPTH  wire=ALL

and **Print Halcyon Diagnostics** now reports the version and package name, the
pass the renderer received, which materials resolve to Wireframe with their
wire sizes, and how many pixels a triangle covers — the number that decides
whether All Edges can draw a wire at all.

If that banner says a version older than this one, Blender is running a
different build from the one installed, and that alone would explain two fixes
appearing to do nothing.

---

## [1.18.0] — 2026-07-28

Four bugs reported from real use in Blender, and the two more that turned up
while chasing them. Every one of them is the same shape: a control that was
wired up at one end and not the other.

### Fixed — the render passes were one pass

Halcyon force-enabled Blender's own **View Layer ▸ Passes** panel and then
wrote exactly one pass into the render result. Every checkbox in it was a
control that did nothing, and the compositor read black for all of them.

Now there is a **Halcyon Passes** panel in View Layer Properties offering the
six this engine can genuinely fill — **Depth, Normal, Position, UV, Object
Index, Material Index** — declared to Blender through `update_render_passes`
and written alongside Combined. They use Blender's own names and channel
layouts, so a Halcyon Z pass drops into a comp built for Cycles without
rewiring.

Blender's panel is no longer forced on. It lists mist, vectors, denoising data
and the light-component passes, none of which this engine produces, and showing
controls that do nothing is the exact thing `EXCLUDED_PANELS` exists to
prevent.

These are data, so they skip the display chain, the palette and the dither
entirely, and at 4x supersampling they take one sample per output pixel rather
than an average — averaging a normal or an object index produces a value that
was never on any surface.

The worker pool sends pixels back through its pipes, not buffers, so a frame
with passes enabled renders in-process and says so on the console rather than
quietly dropping them.

### Fixed — the depth buffer was never a distance

Found while writing the Z pass. `gbuf.depth` is normalised device depth: it
runs 0 to 1 and crowds an entire scene into the last few thousandths. A Z pass
in NDC is useless in a comp — and **depth of field was comparing a Focus
Distance in metres against it**, so every value the slider allows sat far
behind the whole scene and the frame blurred uniformly.

Both now use a real distance from the camera, reconstructed from the G-buffer.
Focus Distance is the distance it says it is.

### Fixed — selecting HLSL crashed Blender

`sock.default_value` on a colour or vector socket is a **live view into the
socket's own memory**, not a copy. The socket rebuild captured those views,
called `inputs.clear()`, and then read them back to restore the values — a
use-after-free, which takes the process down rather than raising, so no amount
of exception handling was ever going to help. Every captured value is
materialised now, before anything is cleared.

The re-entrancy guard next to it was `self._busy`, an ordinary Python
attribute. Blender hands out a fresh wrapper object on every access to a node,
so that attribute was written to a temporary and read back as the class
default: **the guard had never once closed**. It lives in a module-level set
keyed by the node's own pointer, which is the only identity that survives the
wrapper being rebuilt.

### Fixed — the Wireframe model drew dots, not wires

The screen-space edge distance came from a central difference that required the
pixels on *both* sides to belong to the same triangle. On anything denser than
a cube that is rarely true: the derivative collapsed to zero, the distance went
to infinity, and the wireframe came out as a scatter of specks — or, with the
interior knocked through to the background, as very little at all.

One-sided differences are used instead, so a pixel needs only one neighbour
along each axis, in either direction. A pixel with no same-triangle neighbour
at all is a triangle about one pixel across; there is no derivative to measure
and no interior to speak of, so it counts as being *on* the edge rather than
infinitely far from it.

The test asks the question that matters rather than counting pixels: a wire is
connected, and specks are not. Over 90% of lit pixels must have a lit
neighbour, on a sphere and on teapots of 2,600 and 13,000 triangles.

### Fixed — Toon Steps never did anything

It was offered by the node, copied by the exporter, carried into the shading
function's signature — and then not read. `diffuse_toon` took a `steps`
argument and produced one hard step regardless. Four releases of a control that
did nothing.

Cel shading now cuts the light ramp into that many flat tones, with the edges
spread evenly across the lit range. **Two is bit-identical** to the single-step
version it replaces — the loop runs once and reproduces the old expression
exactly — which matters because two is the default and every toon render made
before this used it. The GLSL emitter got the same loop, so the GPU path agrees.

### Fixed — the GPU device plan had never run

`_cap.plan(scene, settings)` was called with `scene` several lines before
`scene` was assigned. The `UnboundLocalError` went into a bare `except` and was
swallowed, so the device plan and every note it prints have never once
executed. Moved to after the export.

---

## [1.17.1] — 2026-07-28

### Fixed — the generated objects were wound inside out

Reported after 1.17.0 went out: under the PlayStation preset the teapot's own
base was visible through its side.

Every surface of revolution in `core/geometry.py` was wound the wrong way
round. Walking *down* the profile while going *anticlockwise* around the axis
winds each quad clockwise as seen from outside, which points every normal into
the solid. Nothing says so — the z-buffer does not care which way a face
points, and the shading takes an absolute value — right up until backface
culling is on, when the entire outer surface is culled and what you are looking
at is the far interior. That reads as an object with depth problems rather than
as an object inside out, which is why 1.17.0's own culling test did not catch
it: that test was written against the Cornell box and the checkerboard, which
were both correct.

Three things changed:

**The sweeps are wound outward.** One rule now governs every profile in the
file and it is written down at the top of it: a profile is walked with the
solid on its right. The teacup, saucer and spoon read more naturally written
bottom-up, so they are written that way and reversed before sweeping.

**The handle and spout are capped.** They were open tubes ending inside the
body, which is invisible until culling removes the far interior as well and the
pot goes see-through. Their cap winding is decided by *measuring* the fan's
normal against the next ring in rather than by reasoning about which way the
ring turns — that is exactly the kind of sign that was already wrong once here.

**The lid meets the rim.** It stopped 0.1 short of it, leaving an annular slot
straight into the shell through which the culled and unculled renders
legitimately disagreed. The original teapot is famously not watertight — it had
no bottom either until somebody added one — but a tenth of a unit of lid radius
buys a solid whose culled render is provably the same picture, and that is
worth more here than the tenth.

### The test that would have caught it

Two, and the cheap one first. **Signed volume**: the divergence theorem gives
the enclosed volume straight from the winding, and it is exact for these shapes
even though the band seams are topologically open, because the two rings at a
seam sit at identical positions. Every lathed object must enclose a positive
volume. The Cornell box must enclose a negative one — its walls face inward on
purpose, and stating that as the rule rather than as an exception is the point.

**And the invariant the user actually sees**: each object is rendered from
seven directions with culling on and off, and the two must agree. For a closed
solid viewed from outside they must — a back face is always behind a front face
along the same ray. A handful of pixels may still flip exactly at the
silhouette, where a front and a back face meet inside one pixel and the tie is
broken differently; the threshold allows one pixel in two thousand, because a
winding error is not a handful of pixels, it is the whole object.

---

## [1.17.0] — 2026-07-28

### Added — period objects, in the Add menu

**Add ▸ Halcyon**, in the 3D view. Four things to point a camera at, generated
rather than shipped: they have a resolution knob, they weigh nothing in the
zip, and they cannot drift out of step with the code that builds them.

**Utah teapot.** Newell's, at whatever resolution you ask for. The rotational
parts — rim, body, lid, foot — are swept from the teapot's own profile using
the constant Newell used for a quarter circle: **0.56**, not the 0.5523 that
actually approximates one. The teapot has been very slightly out of round in
every render of it since 1975, and rounding that off would be tidying up
somebody else's decision.

Said plainly, because it matters: the handle and the spout are swept here to
the published silhouette rather than lifted from the 1975 control-point file.
This is the Utah teapot's shape and proportions. It is not a copy of that file.

**Cornell box.** The published measurements, in millimetres, converted and
nothing else. The walls really are not square to each other and the blocks
really are rotated by those odd angles; that is what makes it a *measurement*
rather than a model. Six materials come with it and an area lamp is placed at
the ceiling panel, so it is lit the moment it is added.

The Y-up to Z-up conversion is the part worth writing down. The obvious map,
`(x, z, y)`, has determinant **−1**: it mirrors the box, and the red and green
walls swap sides. `(x, −z, y)` keeps the handedness and leaves the open face
toward +Y where a camera naturally sits. A test asserts the red wall is where
the published scene puts it, and that every wall normal points into the room.

**Newell teaset.** The rest of the service — cup, saucer and spoon — lathed
from their profiles, which is how they were made: a surface of revolution is
one curve and a sweep, and every modeller of the period had that tool before it
had anything else.

**Checker ground plane.** A checkerboard as real alternating faces with two
materials, not a texture. Every ray tracer's first picture stood its spheres on
one of these, and it was a genuine chequered surface — which is why the squares
stayed square out to the horizon instead of swimming.

`core/geometry.py` is bpy-free like the rest of `core/`, so the shapes are
tested headlessly: every edge used once or twice, no degenerate quads, no index
past the end, and the teapot 3.15 units tall with a handle on one side and a
spout on the other.

### Added — seven more procedural textures, 20 in total

The POV-Ray pattern family, each from its published definition:

**Bozo** — plain noise under a colour map, optionally turbulence-displaced.
Half the materials of the era started here. **Agate** — a band thrown about by
a large turbulence and raised to **0.77**, and that exponent is the whole
character of the pattern: it pushes the midtones up so the bands read as
layered stone rather than as a sine wave. **Leopard** — three sines summed and
squared, which lands a rounded spot in every unit cell. **Onion** — concentric
spherical shells. **Bumps** — smooth noise read as a height field, one octave
by default because that is what makes a bump rather than a crumple. **Wrinkles**
— folded noise summed at halving amplitude, creasing wherever an octave crosses
zero. **Brick** — running bond with mortar courses, bevelled edges and a
per-brick id, because a wall where every brick is exactly the same colour is
the one thing that never looked right.

Wrinkles is lifted by a fixed gain rather than normalised across the batch. The
renderer shades in chunks; anything normalised per chunk would tile differently
in every one of them.

### Added — two more sky modes, eight in total

**Banded Gradient.** The gradient cut into flat steps. Not a stylised gradient
— it is what a gradient *was* on a machine with 256 colours and most of them
already spent on the scene. Doing it here rather than leaving it to the palette
stage matters: the palette stage quantises the whole frame at once, and a sky
that was already stepped keeps its steps whatever the rest of the image spends
its colours on.

`Bands` is the number of colours from horizon to zenith **inclusive**, so the
divisor is one less than the count. Dividing by the count instead leaves a band
at the zenith that is infinitesimally thin — only ever hit exactly at the top —
and every palette then carries an entry it never spends. A test counts the
levels.

**Starfield.** Space: a flat backdrop, stars all the way round, and an optional
turbulent nebula. There is no dome here, so a star at −60° is as valid as one
at +60° — which is the one thing the Bryce star layer deliberately does not do,
because it sits under a sky. A test checks both halves of the sphere have the
same star density, since the difference is invisible until someone points a
camera up from below.

Stars are not all white. A single hash per cell decides whether one reads hot
and blue or cool and amber.

### Added — three more master shader inputs

**Sheen**, with its own colour and roughness. A velvet lobe: light scattered
back toward the viewer at grazing angles, which is what makes velvet, suede and
dusty cloth bright at their edges and dark face-on. It is added in the light
loop rather than inside a reflectance model, so it behaves the same on Lambert
as on Cook-Torrance — and unlike the rim term, **it needs a light**. A test
turns the lights off and asserts the sheen goes with them.

**Bump Strength.** Scales how far the Normal input is allowed to bend the
shading normal away from the surface it sits on. Measured on the normals rather
than on pixels, where a saturating highlight could hide it: 0 bends by exactly
zero radians, and the angle grows with the knob.

**Refraction Amount.** How much of the ray traced *through* a transparent
surface is used. 1 is glass; lower keeps what is behind the surface where it
is, which is how a scanline renderer's alpha blend looked.

All three ship at defaults that are inert, and a test asserts a render with
them set to those defaults is **bit-identical** to one from a graph that has
never heard of them. They were added to a shader that already had 33 inputs and
anything that shifted an existing render by a hair would have broken every
scene made before them.

### Fixed — backface culling was removing the wrong side

`backface_cull` kept the faces pointing **away** from the camera and threw away
the ones pointing at it. Five presets set it -- PlayStation among them -- and on
those a solid object showed its own dark interior, which reads as the object not
being there rather than as being inside out. That is why it survived: nothing
looked wrong in a way that named itself.

The test that catches it needs no reference image. Culling the back of a
**closed convex** solid can only ever be invisible, because the back was behind
the front anyway -- so a cube and a sphere are now asserted to render
**bit-identically** with culling on and off, and a single plane is asserted to
survive facing the camera and to vanish when its winding is reversed.

Found while checking that the new checker ground plane renders under the
PlayStation preset. It did not.

### Note

The sky-mode test used to compare the strip of background one camera happened
to see, and it started failing when Banded Gradient was added — correctly. Down
at the demo scene's horizon a banded gradient and a smooth one agree *exactly*,
because the first band is the horizon colour. Distinctness is now asked of the
whole sphere, which is what the claim was always about.

---

## [1.16.0] — 2026-07-28

### Added — 18 more presets, 69 in total

**Handhelds and later consoles.** Game Boy (160x144, four greys, everything
reading as silhouette), Virtual Boy (four levels of red — the only console that
shipped a palette with one hue in it), Game Gear (4096-colour master palette on
a backlit screen that smeared what it showed), Super Nintendo (polygons came
from a chip on the cartridge, so few, flat and sorted per polygon), Neo Geo,
Sega 32X.

**Home computers.** Commodore 64 (160x200 with pixels twice as wide as they are
tall, because the mode was), ZX Spectrum, Apple IIGS, MSX2 (Japan's home
standard, better at this than anything sold in the West that year), NeXTSTEP
(two-bit greyscale on a large sharp display, Atkinson dithered), SGI Indy (the
workstation everything else was compared against — no palette at all).

**Software renderers.** Doom, where light levels quantised into bands *are* the
shading model rather than an artefact of one. RenderMan, for what the film
houses used while everyone else argued about palettes. Turbo Silver, the Amiga
raytracer that became Imagine. Lightscape, back when radiosity meant hours of
solving before anything appeared. AutoShade, from when rendering meant filling
each face with one colour and calling it a day.

Every one renders in the suite, which is what the preset test is for.

### Note

The ZX Spectrum preset approximates rather than reproduces. The real machine
allowed two colours per 8x8 character cell, which is why everything on it looked
coloured in afterwards, and Halcyon has no attribute-clash mode to drive. The
palette and resolution are right; the constraint that gave the machine its
signature is not modelled, and the preset's note says so.

---

## [1.15.0] — 2026-07-28

### Added — visible spot light cones

The beam you can see in the air. Every package of the era had these and none of
them integrated a volume properly: LightWave called them volumetric lights, 3D
Studio called them Volume Lights, and both marched a handful of samples down the
view ray and added up whatever fell inside the cone. That is what this does,
because that is what it looked like.

The cone is intersected **analytically** rather than built from geometry. A ray
against an infinite double cone is a quadratic, and its two roots bracket the
part of the view ray inside the beam — so the cost is a few samples across that
segment rather than a march down the whole ray hoping to find it.

Verified against a brute-force march that walks the entire ray in 4,096 steps
and tests containment at each one, sharing no code with the fast path:

- **100%** agreement on which rays are inside the beam
- **0.00000** median relative error on how much each one scatters

Depth cuts the beam short, so it stops at whatever it crosses rather than
shining through — a test asserts a surface can never make a beam *brighter*, and
that it does dim one somewhere. Rays facing away get exactly zero, and point,
sun and area lights have no cone at all.

**Render Properties ▸ Lighting ▸ Spot Cones.** Each spot light's own Volumetric
value decides whether it has a beam and how strong; the panel scales all of them
together.

`Samples` is exposed because low counts band, and **the banding is the period
artefact rather than a defect** — it is the same slicing those renderers showed,
and burying it under a large default would be the wrong kind of accurate. 8 to
16 is where they sat.

`Falloff` is the exponent on distance from the lamp — 2 is inverse-square, lower
carries the beam further. `Reach` bounds a beam that nothing stops; a cone is
infinite and its contribution converges, but the integration needs a finite
bound and an endless beam looks wrong regardless.

---

## [1.14.0] — 2026-07-28

### Changed course — GPU shading does not need a GPU rasteriser first

The plan had been rasteriser, then shading. The frame breakdown says that is
the wrong order:

| stage | share of frame |
|---|---|
| **shade** | **71%** |
| rasterise | 9% |

A GPU rasteriser is the harder piece, the less verifiable piece, and the one
worth 9%. Sequencing it first was habit rather than evidence.

The CPU already produces everything shading needs — triangle IDs, barycentric
coordinates, depth — and that is a fixed-size buffer per pixel, which is exactly
what a full-screen fragment pass consumes. Upload it, shade, read back. **The
same mechanism as the post stages**, which are measured working on this
hardware four times over.

### Added — the G-buffer packs and rebuilds exactly

`gpu/gbuffer.py`. The buffer becomes two textures: barycentrics plus triangle ID
per pixel, and per-corner vertex attributes indexed by that ID. GLSL rebuilds
position, normal and UV by interpolating the three corners — the same arithmetic
the CPU does, moved.

Verified against the CPU's own reconstruction: **0.000000 maximum difference**
on positions, with the coverage mask agreeing pixel for pixel.

Triangle IDs ride in an alpha channel as floats. A float32 carries integers
exactly to 16.7 million, far past anything this renderer will rasterise, and it
avoids an integer texture format — one fewer thing to get wrong at bind time.

### What remains

The upload and the draw. Those cannot be checked without a GPU, so they will
want a self test on real hardware the moment they exist. Everything either side
of them — the G-buffer packing, the attribute reconstruction, the 17 reflectance
models, the 29 node emitters, whole-material assembly — is written and matches
the CPU exactly.

---

## [1.13.2] — 2026-07-28

The worker pool is **2.97x faster** than rendering in process on a 20-core
machine. Six releases to get there; here is where the rest of the gap is.

### Measured

| | |
|---|---|
| total duplicated work at one band per worker | 1.61x a single frame |
| ideal on 20 cores | 12.4x |
| **measured** | **2.97x** (24% of ideal) |

The remaining gap is not duplicated rendering — that is down to 1.61x — it is
the parent. Each frame moves about 5 MB of band results back through pipes, and
unpickling them holds the interpreter lock, so the parent cannot overlap that
work with the workers still rendering.

### Changed — band results travel as raw bytes

The worker sends its array's shape and `tobytes()`; the parent rebuilds it with
`np.frombuffer`. Pickling a float array per band cost the parent lock time it
had no way to overlap. Output is still **bit-identical** to rendering in
process, on the first frame and on repeats.

### Note on threads

With the pool measured at 2.97x and threads measured at 0.94x, worker processes
are now the only route to CPU parallelism that does anything. **Use Worker
Processes** in the Performance panel is worth turning on for final renders; it
helps most on 24-bit presets and least on heavily quantised ones, where the
palette stage still runs in Blender's own process.

---

## [1.13.1] — 2026-07-28

The worker pool runs, and the first measurement of it was **0.44x** — two and a
half times *slower* than not using it. This release is why, and the fix.

### Fixed — every band was rasterising the whole mesh

A pooled frame was split into three bands per worker, and each band called
`render()` with a row range. That shaded only its own rows, but it **projected,
clipped and rasterised every triangle in the scene** first. Sixty bands meant
sixty rasterisations, and the duplicated work dwarfed the parallelism.

Two changes:

**The rasteriser takes a scissor.** Triangles wholly outside the band are
dropped, and dropped *before clipping* rather than after — clipping every
triangle in every band was the larger cost. Only triangles entirely in front of
the near plane can be judged that cheaply; any straddling it go to the clipper
as usual.

**One band per worker, not three.** Every band repeats the fixed cost of a
render call, and with the rasterisation now proportional there is little left to
load-balance.

Measured on the demo scene at 640x480, as total work relative to a single frame:

| | before | after |
|---|---|---|
| 60 bands | 5.01x | — |
| 20 bands | 1.65x | **1.61x** |
| 8 bands | 1.21x | 1.01x |

At one band per worker on 20 cores the ceiling moves from 0.25x to **0.081x**,
which is a 12x speedup rather than a 2.3x loss. What the pool actually delivers
against that ceiling still needs measuring on hardware with more than one core.

Bands still rejoin **bit-identically** to a whole-frame render at 1x and 2x
supersampling, and a test asserts the scissored raster produces the same
triangle IDs and the same depths inside the band, while touching fewer
triangles outside it.

### Note

Two of the three checks in the new test were wrong on first writing, and both
were the test rather than the code: one expected a scissored raster to write
nothing outside the band, when a triangle overlapping the band legitimately
covers rows beyond it; the other compared depths with `inf - inf` and got NaN.
The code was right both times.

---

## [1.13.0] — 2026-07-28

### Fixed — the worker pool, after five releases: it was a pickle name

The log finally said it outright:

    got message 'ping' / ping answered
    read RAISED ModuleNotFoundError("No module named 'bl_ext'")
      in pickle.loads(bytes(buf))

The handshake worked. The **scene** could not be unpickled.

Installed as an extension, Blender imports this add-on as
`bl_ext.user_default.halcyon_render`, so every dataclass pickled inside Blender
carries that module path. The worker imports the same files from disk as plain
`halcyon_render`, so those names resolve to nothing and the unpickle fails.

The worker now installs an import hook that makes the package answer to the name
it had inside Blender. A finder rather than a fixed list, because there is no
knowing in advance which submodules a pickle will reach for — and it also
supplies the parents of the alias, since pickle imports the top of a dotted path
first and would otherwise fail on `bl_ext` before ever reaching the package.

The parent passes its own import name across, so nothing is hard-coded.

A test pickles objects under a Blender-style name and unpickles them in a fresh
interpreter that has only the plain package on its path.

### Why it took five attempts

Each round fixed a real bug that was not the bug: a pipe closed by garbage
collection, a Windows command line with quoting in it, a lost lock, a missing
import. All genuine, none causal.

The thing that actually found it was a log line, added in the fifth round, after
noticing that the worker's read loop caught every exception and `break` — which
made a failed unpickle indistinguishable from a clean end of stream. The
ambiguity was the bug behind the bug.

The same lesson as the timing breakdown, which found the ray-tracing explosion
after three wrong guesses. It was slower to apply the second time.

---

## [1.12.3] — 2026-07-28

### Changed — Threads now defaults to 1, because it was measured

Two independent runs on a 20-core machine, at two frame sizes:

| | one thread | auto |
|---|---|---|
| 640x480, no supersampling | 0.279 s | 0.288 s |
| 480x360, 4x supersampling | 1.176 s | 1.205 s |

Auto is consistently about **3% slower** and never faster. NumPy releases the
interpreter lock only for large array operations, and the node evaluator is
dominated by Python dispatch between small ones, so the pool contends rather
than divides.

The default is 1. The setting is still there, and a very large frame may behave
differently — but nothing measured so far has, and a default should follow the
evidence rather than the intention.

### Fixed — a read error in the worker looked exactly like a clean exit

The worker's loop caught every read exception and `break`, which is
indistinguishable from end of stream. Four rounds of diagnosis were spent on an
ambiguity that a single log line removes. It now records what it read, what
raised, and the traceback.

The last report narrowed it a long way: bootstrap, NumPy, package import and
`entering main` all succeeded, and `main returned normally` — so the loop ended
on its own rather than crashing.

### Changed — the protocol writes to a raw descriptor

Every previous attempt went through a `BufferedWriter` borrowed from
`sys.stdout`, and every failure looked like a clean end of stream. The protocol
now writes with `os.write` on a duplicated descriptor: no buffering, and no
wrapper whose lifetime can close the stream. That removes the variable rather
than reasoning about it again.

The worker also logs each message it receives and each reply it sends, so the
next report will say whether the ping arrived, whether the answer was written,
and what happened if it was not.

---

## [1.12.2] — 2026-07-28

### Corrected — threading does not help, and has been claimed to for months

Measured on a 20-core machine at 640x480: 1.00x at one thread, 0.96x at four,
0.93x at sixteen, **0.87x at thirty-two**. More threads are slightly *slower*.

NumPy releases the interpreter lock only for large array operations, and the
node evaluator is dominated by Python dispatch between small ones — so the
threads contend rather than divide the work.

This was claimed to work for several releases on reasoning rather than
measurement, on a machine with one core where it could not have been observed
either way. The README now carries the table and says so plainly, because it is
the sort of claim that quietly wastes an afternoon.

The output is still bit-identical at 1, 4 and 20 threads. It was always correct;
it was never fast.

**Worker processes are the answer**, not threads. Separate interpreters have no
shared lock at all, which makes finishing that path the priority rather than a
curiosity.

### Reverted — the chunking change from 1.12.1

Giving every worker its own chunk was sound reasoning and measured worse:
1.00x became 0.87x, because the shading path is not limited by chunk count. The
floor is back, and the code says why so nobody re-derives it.

### Fixed — two real bugs in the worker pool

- **`self.lock` had been lost** from the Worker constructor in an earlier edit,
  so `call` had no lock at all. Restored.
- **`threading` was never imported.** The pool failed with a NameError the
  moment the lock came back, which is the sort of thing that only surfaces once
  the first bug is fixed.

### Added — the worker logs its own startup

Four attempts have now ended in `exited cleanly having said nothing`, which is
the least informative failure there is. The bootstrap now writes a log before
each step it attempts — interpreter, version, whether stdin has a descriptor,
whether NumPy imports, whether the package imports, whether `main` was entered
and how it left — and the parent reads that file when a worker will not answer.

Guessing has now cost four releases. Logging costs twenty lines.

---

## [1.12.1] — 2026-07-28

Three findings from a self test on an RTX 5060 Ti with 20 cores.

### Fixed — thread scaling had collapsed to 1.00x

Every thread count, 1 through 32, rendered in the same time. The cause is a
structural limit rather than a broken pool: shading chunks had a 16,384-fragment
floor, so a 320x240 frame is **three chunks** and cannot occupy more than three
threads however many are asked for.

The floor now yields when it would starve the pool — an idle core costs far more
than a chunk's overhead. Measured by counting chunks, which is the mechanism
rather than the symptom:

| fragments | 4 threads | 16 threads | 20 threads |
|---|---|---|---|
| 47k (320x240) | 3 to 5 | 3 to 17 | **3 to 21** |
| 307k (640x480) | 4 to 16 | 4 to 19 | 4 to 20 |

Output is still identical at 1 and 16 threads.

**This machine has one core, so the speedup is unmeasured.** What is measured is
that the pool now has enough work to divide. That distinction has mattered
before and it matters here.

### Fixed — the self test measured a frame too small to scale

At 320x240 the whole frame is a few milliseconds on a fast machine, and the
thread pool costs more than it saves — so the test was reporting an artefact of
its own scene size. It now uses 640x480, takes the best of three runs rather
than one, and says outright that a small frame cannot show scaling.

### Fixed — the worker pool, third attempt

Still `worker exited with code 0` after the pipe fix, which means the child
started and did nothing. The remaining suspect is the other thing that produces
exactly that signature: a `-c` command line carrying a quoted Windows path with
spaces, inside a string that itself contains quotes.

The bootstrap is now a **file** written to the temp directory, invoked as
`python <file>`, with the package root passed through the environment. There is
no quoting left to get wrong. The failure message also names the bootstrap path
so the next report can say whether it was even created.

### Changed — Lens distortion promoted, on your measurement

0.318 before the half-texel fix, **0.00426 after**. Promoted from UNPROVEN to
CLOSE and enabled, on the hardware number rather than the harness one. Four of
five GPU post stages now run.

Only composite NTSC remains disabled, at 0.287, because the CPU path blurs I and
Q with different radii and the shader uses one for both.

---

## [1.12.0] — 2026-07-28

**Everything below the rasteriser now exists and is proven correct.** A whole
material assembles into a single GLSL fragment shader that shades identically to
the CPU path. What is still missing is the draw itself.

### Added — complete material shader assembly

`gpu/material.py` takes a material's node graph and produces one compilable
fragment shader: the node emitter fills the surface parameters, the GLSL
reflectance models shade it, and an unrolled light loop sums the lights.

Verified by running the assembled shader through NumPy and comparing against the
CPU shading path: **all 15 applicable models agree to float32 rounding** — about
1e-6 relative — on a material with real node work in it (two colours mixed, then
hue-shifted) lit by a sun and an attenuating point light.

This is the test that matters most, because it covers the *seams*. The emitter,
the models and the light loop were each already checked alone; this checks them
together, and the two bugs it found were both in the joins rather than the
parts.

### Changed — the light loop is unrolled

Not indexed into a `uniform vec3 lights[N]`. A driver prefers the unrolled form
— no dynamic indexing, and the count folds at compile time — and it removes a
genuine ambiguity: a `vec3` read out of a uniform array is indistinguishable
from three lanes of a scalar, which is exactly the shape of bug that produced a
40% error across every model until it was found.

### Fixed — the image texture emitter is registered

It reads `props['image']`, not `image_name`, which was a fault in the check
rather than the emitter. Verified to **0.000000** against the CPU under
nearest-neighbour sampling, which is what proves the coordinate handling: an
image texture defaults to **UV**, not generated coordinates, and getting that
wrong samples the wrong place everywhere. Filtering is a binding-time property
of the sampler and is carried in the emitter's output for the assembler.

The demo scene's materials are now all emittable.

### Fixed — uniform vectors in the shader interpreter

A 1-D array of two to four elements was ambiguous between that many lanes of a
scalar and one uniform vector, and `select` left it unbroadcast. The other
operand settles it, so it does now. This is a fix to Halcyon's own GLSL backend
rather than to the port, and it is what let the assembled shader run at all.

### Still missing

The rasteriser. Drawing the mesh into a G-buffer is the last piece, and the one
part of this that cannot be checked without a GPU — so it will want a self-test
run on real hardware the moment it exists.

---

## [1.11.0] — 2026-07-28

Stage three of the GPU port, or most of the part that can be built without a
GPU. Nothing here renders on a GPU yet — the rasteriser is still missing — but
the two hardest pieces are now written **and proven correct**.

### How anything was proven at all

Halcyon has its own GLSL front-end with a NumPy backend. That means GLSL written
for a GPU can be *executed here* and compared against the CPU function it
replaces, exactly. Every number below is a measured maximum absolute difference,
not an estimate.

### Added — all 17 reflectance models in GLSL, matching the CPU exactly

`gpu/glsl_shading.py`. Every model agrees with `core/shading.py` to **0.000000**.

Nine were wrong on the first run, and each would have looked plausible:

- **Blinn-Phong** takes four times the stated exponent
- **Blinn** uses a Beckmann distribution with Schlick's Fresnel, not GGX with a
  dielectric one
- **Minnaert** takes `1 + 2·roughness` as its darkness, and multiplies by N·V
- **Ward** is driven by roughness directly, not by the Phong exponent
- **Anisotropic** normalises by `sqrt((nu+1)(nv+1))/8π`, then multiplies by 8
- **Metal** is Cook-Torrance driven by `1/gloss`, not Blinn-Phong
- **Toon** steps on the *angle*, not the cosine, and halves its specular size
- **Strauss** scales diffuse by `rn`, tints with a plain `1 - |N·L|`, and skips
  the soften pass entirely
- **Constant** returns zero diffuse, not one

### Added — a node graph to GLSL emitter, 28 node types verified

`gpu/emit.py` walks the same exported graph dicts the NumPy evaluator consumes
and emits GLSL. **53 cases, all matching to 0.000000**: RGB, Value, MixRGB in
nine blend modes, Math in 21 operations, Vector Math in 11, Invert, Gamma,
Bright/Contrast, Combine and Separate XYZ and RGB, Clamp, Map Range, Hue/
Saturation, Layer Weight, Checker, Fresnel, Tex Coord, UV Map, Geometry, the
shader mixes, and reroutes.

Five more real errors, all of which would have rendered something believable:

- alpha was blended in MixRGB where Blender keeps it from the first input
- RGB and Value read a property, not a socket
- Logarithm takes a base
- an unlinked texture Vector means **generated coordinates**, not the socket
  default — reading the default gives a texture that never varies
- Layer Weight's Facing output is driven by an exponent, not a plain
  one-minus-cosine

**An unknown node is declined, never approximated.** `can_emit()` reports which
types are missing and the whole material stays on the CPU. An emitter that
merely looks right still produces a picture, and a wrong picture is harder to
notice than a missing feature.

The image texture emitter is written, compiles, and is **deliberately not
registered**, because it has not been shown to agree with the CPU. It is the
single node blocking the demo scene, and finishing it is the obvious next step.

### On the coded shader node

Its emitter is four lines: inline the user's source and call it. There is no
translation, because it is already GLSL. On the CPU that same source must first
be compiled to NumPy — work that a GPU makes unnecessary. It remains restricted,
but as the easiest piece of the port rather than the hardest, and the capability
table now says so in terms a test enforces.

### Changed — capability reports per material

`emittable_materials()` reports how many of a scene's materials could be emitted
today and which node types block the rest. On the demo scene: two of three,
blocked by the image texture.

---

## [1.10.0] — 2026-07-28

### Added — a Device switch, and an honest capability table behind it

**Device: CPU or GPU**, in the Debug panel. Cycles has the same problem and
solves it the same way — Open Shading Language was CPU-only in Cycles for years
for exactly this shape of reason.

New `gpu/capability.py` holds one table of what runs where, and it distinguishes
two things that are easy to conflate:

| | meaning |
|---|---|
| **BOTH** | ported, and measured against the CPU path on real hardware |
| **NOT_YET** | could run on a GPU, nobody has written it |
| **NEVER** | the algorithm cannot run on a GPU at all |

That distinction is the whole point. "Error diffusion is CPU-only" and "the node
evaluator is CPU-only" are true for completely different reasons, and only one
of them will ever change. One flag would have hidden the roadmap.

- **BOTH** — display transform, ordered dither, CRT mask
- **NEVER** — error diffusion (sequential by construction) and the A-buffer
  (an unbounded per-pixel fragment list is a different algorithm, not a port)
- **NOT_YET** — everything else, each with the reason attached

Choosing GPU never refuses to render. Unsupported work moves to the CPU and the
console names each feature and why. The panel shows the whole table with a tick,
a clock or a cross against every entry, so the cost of the choice is visible
before the render rather than after it.

A test asserts nothing may claim `BOTH` without a stage measured on hardware,
that the two impossible features stay impossible, and that both devices produce
an identical image when everything falls back.

### Note on the coded shader node

The suggestion that prompted this was to restrict the GLSL/HLSL node to CPU. It
is restricted — but it is worth recording that it is the **easiest** piece of
the GPU port, not the hardest. It is already GLSL; on a GPU the NumPy
compilation it does today simply stops being necessary. The hard one is the node
evaluator, where 106 types would each need a GLSL emitter.

The table records that difference in `code_node` (NOT_YET, "the easiest piece")
against `node_graph` (NOT_YET, "this is the hard one"), and a test asserts the
node evaluator is the one described as hard — so the next person to read it,
including me, does not get the two the wrong way round again.

---

## [1.9.1] — 2026-07-27

### Fixed — selecting HLSL crashed Blender

Not an error, a segfault, and the reason is that the Coded Shader node rebuilt
its sockets from inside an RNA property update callback. Blender is mid-update
there, and adding or removing sockets at that moment crashes rather than
raising.

Selecting a language hit the worst version of it. `_lang_changed` set
`source_text`, whose own update callback then rebuilt the sockets -- so the
topology change happened inside a **nested** callback, two levels deep in RNA
that was already being modified.

Both callbacks now only set a flag and ask for a rebuild. The rebuild runs on a
one-shot `bpy.app.timers` callback, outside any update, and finds the nodes that
asked for it by scanning for the flag rather than by holding a node pointer
across the timer -- a pointer that the tree may well have invalidated in the
meantime. A re-entrancy guard stops the nested callback firing at all.

Both default templates were checked through the interpreter: GLSL and HLSL each
compile and run, so the language itself was never the problem.

### Note on the test

This cannot be exercised without Blender, so the rule is checked statically
instead: no function used as an `update=` handler may reach socket
add/remove/clear. That test immediately found a second callback calling
`refresh_sockets` on the master shader -- which turned out to be **safe**, since
it only sets `hide` and a description on sockets that already exist, exactly
what Blender's own nodes do. The rule was narrowed to topology changes, which is
the real hazard, and a second check now asserts that function stays free of
them so the exemption remains true.

---

## [1.9.0] — 2026-07-27

### Fixed — no exec anywhere (review item 1, complete)

The shader backend generated Python source and executed it. It now walks the
parsed tree directly, in a new `shaders/interp.py`.

**Nothing about the execution model changed**, which was the whole aim. Every
value is still a NumPy array with one entry per lane, and divergent control flow
is still handled with masks rather than branches: an `if` evaluates both sides
and selects, a loop runs until every lane has left it. That is what lets one
shader take different paths for different pixels, and it survives intact.

The runtime library did not change either. `rt.sel`, `rt.sw_set`,
`rt.mat_mul_vec` and every builtin are the same functions the generated code
used to call -- only the thing calling them is different.

**All 16 shader tests pass on the interpreter**, which was the acceptance bar:
divergent control flow, per-lane loop trip counts, `break` and `continue`,
user functions with `out` parameters, early return, structs and arrays, `mat3`,
swizzle reads and writes, `discard`, the preprocessor, HLSL, ternaries,
`sampler2D`, and both error cases.

Bugs found and fixed while getting there, each a real mismatch with the parser
or runtime rather than a design problem:

- `('post', '++', target)` puts the operator first; reading it as the target
  made `expr('++')` parse `'+'` as a node kind
- pre-increment arrives as a unary, not as its own node
- `BUILTINS` maps a name to a *runtime function name*, not to a callable
- return-type rules take the argument-type **list**, not varargs
- varyings can be supplied under their declared name or their binding name, and
  the generator accepted either

An interpreter also would not notice an unknown function until a pixel reached
that call, turning a compile error into a render that fails halfway. A
validation pass now walks the tree at compile time and rejects them, which is
what makes the last of the 16 tests pass.

`Program` pickles again without exec: workers recompile from the source on
arrival, verified to produce identical output.

### Note

`codegen.py` still contains the old generator. It is no longer reached --
`compiler.py` never calls it and there is no `exec` on any live path -- but
removing the class took shared helpers with it and broke eighteen tests, so it
stays until those can be separated properly. It emits text and nothing runs it.

---

## [1.8.2] — 2026-07-27

Extension review compliance. Five of the six items are done; the sixth is a
rewrite and is described below.

### Fixed — no threading or queues (review item 3)

Every use is gone. Shading and the background pass still work in bounded chunks,
because that is what keeps peak memory a function of chunk size rather than of
resolution, but the chunks now run in sequence.

The worker pool no longer manages its children with threads either. A round of
requests is written to every worker first and the replies collected afterwards,
so the workers still run at the same time -- which was always the point -- while
this process only ever does one thing at once.

Real parallelism now comes from the worker processes alone, as the review
suggests. The **Threads** control has been removed from the UI, since a control
labelled Threads that no longer threads is precisely the kind of thing that
should not ship.

### Fixed — no sys.path manipulation (review item 2)

The worker bootstrap edited `sys.path` so the child could find the add-on. It
sets `PYTHONPATH` in the child's environment instead.

### Fixed — no eval in the preprocessor (review item 1, part one)

`#if` conditions were handed to `eval` with an empty builtins dict. A
preprocessor conditional is integer arithmetic and comparisons and nothing
else, so it now has a small precedence-climbing parser of its own -- about
seventy lines, covering the operators the C preprocessor defines, and verified
against a table of expected values.

### Fixed — packaging (review item 4)

`blender_manifest.toml` gained a `[build]` section excluding `tests/`, `docs/`,
the changelog, the store listing, `__pycache__` and stray zips. Built with
`blender --command extension build` from now on rather than by hand.

### Fixed — installation guide removed (review item 6)

The listing no longer explains how to install. The platform does that.

### Outstanding — exec in the shader compiler (review item 1, part two)

The GLSL/HLSL compiler generates Python source and executes it. That is the
whole architecture of its backend and cannot be patched out; it needs replacing
with a tree of closures that walks the parsed AST directly. The parser, type
checker and builtin library are unaffected -- only the code generator changes,
from emitting text to building callables.

Until that lands, this build still contains two `exec` calls and is **not**
ready for resubmission.

---

## [1.8.1] — 2026-07-27

### Fixed — the infinite ground was unfindable

Two reasons. It is a **world property, not an object**, so there is nothing in
the Add menu to look for -- and the panel was gated on the sky mode not being
"Use Node Tree", which is the default, so for most scenes it did not appear at
all.

The panel is now shown whatever the sky mode is, and the ground is applied after
whichever mode ran, node trees included: an endless floor has nothing to do with
how the sky above it is coloured. When it is switched off the panel says plainly
that it is a world property and where the checkbox is.

**To find it: World Properties > Halcyon World > Infinite Ground.**

### Added — the rest of Bryce's Sky Lab

- **Sun or Moon.** One body, swapped. The moon is a disc with a real
  terminator, so it shows a phase from new through full and back, with an
  earthshine term lighting the unlit side.
- **Three-stop dome gradient.** Horizon, mid and zenith with a movable mid
  height, as Bryce's gradient editor allowed rather than the two stops it had
  before.
- **Atmosphere**, separate from the horizon haze band: exponential depth haze
  across the whole dome, with its own colour and falloff.
- **Blend With Sky** on the haze, so it takes the sky's own colour instead of
  its swatch -- Bryce's own control, and the reason its horizons never look
  pasted on.
- **Wind.** Speed and direction drift both decks over time, stratus slower than
  cumulus because height says it should.
- **Ambience** on the cumulus, filling the shadowed side with sky light.
- **Cloud shadows** cast onto the infinite ground, sampled from the same noise
  the deck is drawn from -- so a shadow always lands under a cloud rather than
  near one.

That is 88 world settings, all exposed, each with a test that it changes the
sky on its own. Wind is additionally checked to move only with time, and the
moon to show genuinely different phases.

---

## [1.8.0] — 2026-07-27

Two additions while 1.7.5 is in review. Not for pushing to the store until that
clears.

### Added — infinite ground plane

An endless floor cannot be geometry, so it is intersected analytically in the
background pass — which is exactly how POV-Ray and Bryce provided one. It costs
nothing per frame however far it reaches, and it never needs subdividing.

Four surfaces:

- **Solid** — one flat colour
- **Checker** — the infinite chequerboard of a thousand ray tracing demos
- **Fractal** — two colours mixed by fbm, for terrain seen from height
- **Ocean** — crossed sine trains perturbing the normal, reflecting the sky
  through a Fresnel term so it mirrors at glancing angles and shows its own
  depth colour overhead. Driven by scene time, so it animates.

**Distance Fade** blends the plane into the horizon. Without it the thing reads
as a flat sheet rather than as ground going away, which is the whole trick.

Tests assert each surface renders below the horizon, that none of them touches a
ray pointing up, that the ocean animates with scene time and the others do not.

### Added — 13 material templates

Ready-made setups on the Halcyon shader, in the Material panel: Chrome, Gold,
Brushed Metal, Glass, Shiny Plastic, Rubber, Polished Marble, Varnished Wood,
Terrain, Cel Shaded, Velvet, Hologram and Wireframe.

They are recipes built at runtime rather than saved node trees, so a template
always matches the current version of the nodes instead of rotting into sockets
that no longer exist. Several exist partly to show what the less obvious inputs
do — Fresnel on the chrome, edge opacity on the glass, the rim term on the cel
shading, a Scratches texture driving anisotropic rotation on the brushed metal.

A test walks every recipe and fails if it names a model, socket or texture node
that does not exist, which is the failure mode a runtime-built tree invites: a
renamed socket makes a template a silent no-op rather than an error.

### Note

The marble template shipped with `'label": "Polished Marble',` — the same
quoting slip that once made the PAL broadcast preset unparseable. It was a
syntax error this time, so it failed loudly on import rather than quietly at
runtime.

---

## [1.7.5] — 2026-07-27

Third report from the RTX 5060 Ti. Three answers, all acted on.

### Fixed — the worker pool closed its own pipe

The failure finally identified itself: `worker exited with code 0`. A *clean*
exit meant the loop ended normally, which meant stdin hit EOF the moment it was
read.

The cause is one line in the worker. It captured `sys.stdout.buffer` and then
reassigned `sys.stdout` to stderr, so that renderer output could not corrupt
the protocol stream. That reassignment dropped the last reference to the
original wrapper, and when Python collected it, it **closed the pipe
underneath**. The parent then saw EOF, the worker's reply failed, and it exited
tidily having said nothing at all.

The protocol stream is now a duplicated file descriptor, independent of
`sys.stdout`'s lifetime. Verified: the pool renders again and its output is
identical to in-process on the first frame and the second.

Three releases of "worker closed the connection" were all this. The clue that
solved it was the exit code, which only existed because the previous release
started reporting it.

### Fixed — DISPLAY is exact after all

`RGBA32F` brought it from 0.0079 to **0.00001** on hardware — the error really
was half-float storage amplified by the gamma curve. Graded EXACT again, on the
hardware measurement this time rather than on the NumPy harness.

### Fixed — LENS half-texel convention

Now graded for the first time, at 0.318. The CPU path normalises over
`width - 1`, working in pixel indices; the shader normalised over the 0..1
texture range. That is half a texel, and the gap grows with the distortion. The
shader works in pixel indices now, like the CPU one.

### Known — NTSC still diverges

Graded for the first time at 0.287. The CPU path blurs I and Q with different
radii — chroma bandwidth is not symmetric in a composite signal — and the
shader uses one filter for both. That is a real difference, not a convention,
and it stays UNPROVEN and disabled until the shader carries two radii.

---

## [1.7.4] — 2026-07-27

Second report from an RTX 5060 Ti under Vulkan. **All five GPU stages now
compile and run on the driver.** Three corrections from what they measured.

### Fixed — half-float storage, not half-right shaders

DISPLAY was graded EXACT because it is bit-identical in the NumPy harness. On
hardware it came back at 0.0079 maximum difference. That is not a logic error:
the textures and offscreen targets were `RGBA16F`, which carries about eleven
bits of mantissa, and the gamma curve at the end of the display transform
amplifies error near black. Both are `RGBA32F` now.

### Changed — grades come from hardware

The validation table used to record what the NumPy harness measured. Where the
two disagree the hardware number wins, because that is the one that reaches the
screen. DISPLAY is now CLOSE rather than EXACT, and the tolerances match what
the driver actually produced: CRT 0.0113, DITHER 0.0327.

CRT and DITHER both came out *better* on hardware than in the harness, which is
worth knowing: the NumPy backend is the pessimistic estimate, not the optimistic
one.

### Added — LENS and NTSC now have CPU references

Both ran cleanly but had nothing to compare against, so they stayed UNPROVEN by
default rather than by evidence. The self test now computes a CPU reference for
each, so the next run grades all five.

### Fixed — the worker pool still explained nothing

It reported `worker closed the connection` with no detail, because the child was
still alive and its pipes therefore unread. A worker that starts but never
answers is now killed so its output can be collected, and the failure carries
the interpreter path and the package it tried to import.

The self test reports the interpreter and bootstrap whether the pool works or
not, since on the machine where it fails that is the missing information.

---

## [1.7.3] — 2026-07-27

First report back from real hardware — an RTX 5060 Ti on Windows, 20 cores.
Three findings, all acted on.

### Fixed — every GPU shader failed, because the backend is Vulkan

All five stages returned `cannot create 'GPUShader' instances`. That is not a
rejected shader: on the **Vulkan** backend, which Blender now defaults to on
Windows, the legacy `GPUShader(vertex, fragment)` constructor does not exist at
all. Shaders must be built from a `GPUShaderCreateInfo`, which carries the
interface itself and wants the GLSL *without* its declarations.

Every stage now has an interface spec, the source has its declarations stripped
for the CreateInfo path, and the old constructor is kept only as a fallback for
OpenGL builds. Tests assert every stage has a spec, that stripping leaves
`main()` and the helper functions intact, that every declared uniform is used by
the body, and that every declaration in the source appears in the spec.

The validation harness could never have caught this. It proved the GLSL logic
was right — and the logic was never the problem.

### Explained — thread scaling plateaus at 1.84x, and that is correct

1, 2, 4, 8, 16, 32 threads gave 1.00, 1.31, 1.84, 1.80, 1.79, 1.79. That looked
like a bug and is not one. Shading was 53% of that frame and was the only
threaded stage, so the ceiling is 1 / (0.47 + 0.53/N) — **2.13x at infinite
threads**. Reaching 1.84x means threading works and there is nothing left to win
without threading something else.

So the background pass, the third largest stage on that frame, is now threaded
too. Its output is identical at 1, 4 and 16 threads to within one ULP of
float32 -- summing a slice and summing the whole array can differ in the last
bit, and on a 320x240 frame that showed up on 11 pixels out of 76,800 at a
magnitude of 6e-08. The test asserts that bound rather than exact equality,
because exact equality would be a claim the arithmetic cannot support.

The remaining serial stages are rasterisation and the shadow maps.

### Fixed — the worker pool failed with nothing to go on

`OSError: worker closed the connection` and no more. A worker that dies during
startup explains itself on stderr, and that was being discarded. The child's
stderr, or its exit code, is now attached to the error.

---

## [1.7.2] — 2026-07-27

### Fixed — the self test could not run as an installed extension

Two faults, reported on the very first run.

**`bl_info` does not exist in an installed extension.** Blender 4.2+ uses
`blender_manifest.toml` and does not expose `bl_info` at all, so
`from . import bl_info` raised ImportError on the install path most people take.
The preferences panel had the same call, wrapped in a `try` that quietly
reported version 1.0.0 — wrong rather than broken, which is worse.

There is now one source of truth, `version.py`, which reads the manifest so an
installed extension reports what is actually installed. A test walks every
module and fails if anything reads `bl_info` again, and checks the version
matches the manifest.

**The report aborted at the first failure**, which lost the GPU and CPU
measurements that were the entire reason to run it. A diagnostic that stops on
its first problem is close to useless — the problems are what it is for.

Each section is now isolated. A failure names itself, says the rest still ran,
and the remaining sections carry on. The operator reports how many sections
failed. A test injects a simulated driver failure and asserts the later sections
still produce output.

---

## [1.7.1] — 2026-07-27

### Added — Run Self Test

There is no way to give me access to a machine, so this goes the other way: a
probe that runs the checks I would run, and prints one report that can be pasted
straight back.

**Debug panel > Measure this machine > Run Self Test.** Output goes to the
system console and to the clipboard.

It covers exactly the three things this add-on has had to ship unmeasured:

- **GPU stages.** Compiles every GLSL shader on the real driver and prints the
  driver's own error verbatim if one is rejected. Runs each on a fixed test
  image and compares against the CPU function it replaces, reporting max and
  mean difference beside the grade claimed here. That turns the two stages
  currently marked unproven into measured ones, or shows them wrong.
- **CPU scaling.** The same scene at 1, 2, 4, 8, 16 and 32 threads with speedup
  against one thread, then the worker pool against in-process. The pool has
  shipped since 1.2.0 with its speedup explicitly unverified; this measures it.
- **Frame breakdown.** The per-stage timing table, so a report always says where
  the time went.

Every section degrades: no GPU means that section states why and the rest still
runs. The report opens with add-on and Blender versions, platform, Python,
NumPy, core count and the GPU vendor, backend and driver version, so a pasted
report is self-contained.

Tested headlessly on a machine with no GPU and one core, which is the awkward
case.

---

## [1.7.0] — 2026-07-27

### Added — the GPU port, stage one

Blender's `gpu` module is the layer EEVEE is built on, and the only GPU route
open to an add-on: Cycles' device abstraction is C++ with precompiled kernels
and is not exposed to Python at all. This release puts the parallel post stages
on it.

**GPU Post Processing** in the Debug panel. New `halcyon/gpu/`:

- `device.py` — probes for the module and a backend, compiles and caches
  shaders, manages offscreen targets, and turns every failure into a reason
  rather than an exception
- `stages.py` — the post chain as GLSL
- `chain.py` — runs a stage on the GPU, falls back to the CPU function per
  stage, and prints why once

### How shaders written without a GPU were validated

Nothing here has been executed by a driver. This machine has none. What it does
have is Halcyon's own GLSL front-end and NumPy backend — so each stage is
compiled with that and run, and the result compared against the CPU function it
replaces. That proves the *logic* even though it cannot prove the *execution*.

| stage | grade | max difference |
|---|---|---|
| Display transform | **exact** | 0.000000 |
| CRT mask, scanlines, vignette | close | 0.025 |
| Ordered dither and bit depth | close | 0.032 |
| Lens distortion | **unproven** | 0.99 |
| Composite NTSC | **unproven** | never compared |

**Only the first three are enabled.** A stage is allowed to run because it was
measured, not because it was written, and a test fails if anything unproven
appears in the enabled list.

The exercise earned its keep: it found that the lens shader let the sampler
**wrap**, so distortion pushing coordinates past the edge brought the far side
of the picture round into the corners. That is a bug that would have shipped and
shown only on real hardware. It is clamped now, though a half-pixel convention
difference against the CPU path remains, which is why the stage stays disabled.

### What this does not do

This is one of three stages. It does not touch rasterisation or shading, so it
will not move a frame whose time is in ray tracing — which, on the scene you
profiled, is where it was. The two remaining stages are:

- **Rasterisation to a G-buffer** — moderate, and there is already a
  bit-identical CPU reference to diff against
- **Shading** — hard: the node evaluator's 105 types would each need a GLSL
  emitter rather than a NumPy one. The coded-shader node gets *easier*, since
  it is already GLSL and currently compiles to NumPy for no reason

The infrastructure added here — context probing, shader caching, offscreen
targets, per-stage fallback and the validation harness — is what those two need.

---

## [1.6.0] — 2026-07-27

Six features from the review list.

### Added — Displacement as bump

A material's Displacement output was computed on every fragment and thrown
away. Actually displacing geometry means tessellating it, which no 1990s
scanline renderer did either — they turned the height into a normal
perturbation and called it bump mapping. That is what happens now, from the
screen-space gradient of the height, with one-sided differences at silhouettes
so an edge does not invent a gradient out of empty space. **Displacement** in
the Shading panel scales it.

### Added — Light linking

Every package of the era let you say which objects a light touched, and it is
still the quickest way to take control of a render. Each lamp gets a collection
and a mode: **Exclude** leaves that collection unlit by the lamp, **Only** lights
nothing else. Authored by collection so Blender's own outliner does the work.

### Added — Lens distortion and chromatic aberration

Barrel below zero, pincushion above, with the three colour channels displaced
by slightly different amounts. A real lens does not focus red and blue in the
same place, and that fringing is the clearest sign an image went through glass
rather than straight to a framebuffer.

### Added — Volumetric light shafts

`volumetric` had been on every lamp since the first release and did nothing.
Setting it above zero now throws shafts: what is bright gets smeared outward
from the light's position on screen in halving steps and added back, which is
exactly how it was done then — no volume is integrated, the illusion is entirely
in the streaks.

A directional light is projected as a **vanishing point** rather than a distant
position, and a light behind the camera is skipped, both with tests.

### Added — Depth of field

Sampling a lens properly needs many rays per pixel. Splitting the frame into
depth slabs and blurring each by its circle of confusion is what compositors of
the era did, at a handful of blurs instead of hundreds of samples. Focus
distance, falloff, slab count and maximum radius are all exposed.

### Not done, and why

**Baking to texture** needs a UV-space rasteriser — a real piece of work rather
than a setting, and better done deliberately than squeezed in here.

**Motion blur** by frame accumulation needs the render engine to step the frame
and re-evaluate the depsgraph mid-render, which is not obviously safe inside
Blender's render loop. It wants investigating before it is written, not after.

---

## [1.5.0] — 2026-07-27

### Added — Painter's Algorithm

`depth_sort` had been in the settings since the first release, named in five
console presets, and did nothing. It works now.

Painter's does not compare fragments, it compares whole polygons: the one whose
depth is nearest wins the pixel outright. Rather than re-sorting geometry and
drawing back to front, every fragment of a triangle is given that triangle's
single depth — which turns the existing z-buffer into exactly that comparison,
works on both the batched and sequential rasterisers, and needs no ordering
guarantees from the clipper.

It reproduces the real failures, which is the point:

- surfaces that interpenetrate meet along a **polygon edge** instead of their
  true intersection
- a large polygon can be wrongly occluded by a small nearer one it actually
  passes in front of
- the errors move as the camera does, which is what makes them read as an
  artifact of the era rather than as a mistake

**Sort By** chooses which point on a polygon decides the order — Centroid,
Nearest Vertex or Farthest Vertex. It moves where the algorithm goes wrong:
nearest is better on surfaces facing the camera and worse on long polygons
running away from it. All three give visibly different results.

PlayStation, Saturn, 3DO, Jaguar and PlayStation High-Res set it, so those
presets now behave as their notes always claimed.

Tests assert every fragment carries its polygon's depth, that all three sort
keys differ, that interpenetrating surfaces render differently from the
z-buffer, and that the batched and sequential rasterisers agree.

---

## [1.4.1] — 2026-07-27

A code-wide audit rather than a feature release. Two classes of problem found by
scanning for them, and both now have tests that stop them coming back.

### Fixed — 292 lines of shadowed dead code in ui.py

Eight top-level definitions were declared **twice**, including four whole panels
and the preferences accessor. Python takes the later definition, so behaviour
was correct, but the file carried nearly three hundred lines that never ran —
and if the two copies had drifted, the wrong one could have won silently.

Caused by patches applying the same replacement at two sites, the same mistake
that duplicated the diagnostics operator in 1.3.0. A test now walks every module
and fails on any repeated top-level definition.

### Fixed — controls that did nothing

An audit of all 138 settings found **20 that no code reads**. Eleven of them
were drawn in the UI, so they were sliders and dropdowns that silently did
nothing — worse than absent, because they cost time and undermine trust in the
controls that do work.

Three were worth implementing and now are:

- **Alpha Threshold** — a hard alpha cutoff rather than a blend, which is what
  hardware without an alpha unit did with cut-out textures.
- **Vertex Fog** — evaluates fog per vertex and interpolates, so the fog band
  follows the tessellation rather than the surface, as fixed-function hardware
  did.
- **Default Falloff** — lights now default to Scene Default and take their
  falloff from Render Properties, which is what that setting was always meant
  to do.

The rest were removed from the UI: edge-AA threshold, subpixel precision, depth
sort, polygon offset, reflection blur samples, texture anisotropy, affine
subdivision, jitter, tile size, bucket order and progressive. They remain in the
settings dataclass as reserved names, and none of them is exposed any more.

A test asserts no unimplemented setting is drawn in the UI, so this cannot
regress and nothing new can be added and left half-finished.

---

## [1.4.0] — 2026-07-27

### Added — twelve more inputs on the Halcyon Shader

All of them apply **after** the reflectance model, so they behave identically
whether the surface is Lambert, Cook-Torrance or Toon. That is deliberate: these
are the artistic cheats every package of the era layered on top of whichever
shader you picked.

- **Fresnel**, **Fresnel Power**, **Fresnel Color** — brightens the highlight
  toward the silhouette, the way a real surface reflects more at grazing angles.
  The cheapest way to stop plastic looking flat.
- **Rim Light**, **Rim Amount**, **Rim Power** — an additive rim that needs no
  light source. The backlight cheat used to lift a subject off its background.
- **Matcap**, **Matcap Blend** — a sphere-mapped image sampled by the view-space
  normal, carrying a whole material in one picture. At blend 1 it ignores the
  scene lights entirely.
- **Reflection Color** — what the Reflection amount is multiplied by. Feed an
  environment image in and you have a reflection map that costs nothing, with no
  ray tracing, exactly as it was done before ray tracing was affordable.
- **Edge Opacity** — opacity at the silhouette, blended by the Fresnel curve.
  Below the centre opacity it thins edges for holograms; above it thickens them,
  which is how glass reads.
- **Backface Color**, **Backface Mix** — different shading where a surface faces
  away. For single-sided leaves, cloth and cards.

New **Matcap Coordinates** node under Add > Halcyon > Halcyon Textures produces
the sphere-map vector to drive an Image Texture into the Matcap input.

Every one is documented with a tooltip and covered by a test that asserts it
changes the render, stays finite, and works across four different reflectance
models.

### Fixed — found while adding the above

- **`Surface.backfacing` was never populated.** The context knew which fragments
  faced away and the surface did not, so the backface override had nothing to
  key off. It silently did nothing.
- **New inputs broke older materials.** A material saved before a socket existed
  has no such socket, and reading a missing one yields zero — which for Edge
  Opacity turned every existing material's silhouette invisible. Newer inputs
  now fall back to values that do nothing, so adding one can never change an
  existing material.
- Two tests depended on another test having installed the bpy stub first. The
  runner orders tests alphabetically and that ordering shifted, which broke
  them; they set up their own stub now.

---

## [1.3.2] — 2026-07-27

### Fixed — no way to create a material from the Halcyon panel

Two faults combined into a dead end:

- The slot list added in 1.0.8 was drawn only when an object had **more than
  one** material slot, so with none or one there was no list and therefore no
  Add button.
- The panel's `poll` required `context.material` to already exist, so an object
  with no material at all had no Halcyon Material panel whatsoever.

Between them, the first material on an object could not be created from this
panel — which is precisely when you most need to.

Now:

- The slot list is **always** drawn, with Add and Remove beside it. Select By
  Material only appears when there is more than one slot, since it is
  meaningless otherwise.
- The panel appears for any object that can hold materials, whether it has one
  or not.
- Beneath the list is Blender's standard material selector with its **New**
  button, so a material can be created and assigned here.
- With an empty slot the panel says so and points at New, instead of showing
  conversion buttons with nothing to convert.

A test checks the panel appears in every combination of object, slot count and
engine, and that the list is not conditional on the count.

---

## [1.3.1] — 2026-07-27

### Added — tooltips on the Halcyon Shader

Every one of the node's nineteen inputs now has a tooltip saying what it does
and which reflectance models it affects — for example:

> **Glossiness** — Tightness of the highlight, as a cosine exponent. Low values
> give a broad sheen, high values a small hard glint. This is the period
> control; the microfacet models use Roughness instead. *(affects Gouraud,
> Flat, Phong, Blinn Phong, Blinn, Anisotropic, Metal, Strauss, Multi Layer)*

The model applicability is not a guess. Each parameter was perturbed and the
models whose output changed were recorded, and **a test re-runs that measurement
against the documented table on every run** — so a claim like "Glossiness does
nothing on Cook-Torrance" stays true if the shading code changes.

That test earned its place immediately: it caught the table claiming Specular
Colour affects Strauss when it does not (Strauss derives its highlight colour
from metalness and the base colour) and misses Toon, which does read it.

### Added — the model dropdown explains itself

All eighteen model descriptions were a single terse line. They now say what the
model looks like, what it is for, and where it came from:

> **Oren-Nayar** — Rough diffuse (1994). Surfaces stay bright toward their edges
> instead of falling off, which is what makes clay, plaster, dust and unglazed
> ceramic look right. Driven by Roughness.

Gouraud and Flat say plainly that they are shading *rates* rather than
reflectance models, since that surprises people.

### Added — the sidebar spells it out

The node's N-panel now shows the current model's description, the list of inputs
it actually uses, and how many it ignores. The node header carries a compact
"12 of 19 inputs used" line.

---

## [1.3.0] — 2026-07-27

### Changed — the diagnostic tools now live behind one switch

They had accumulated across several releases and were scattered through panels
that ordinary work uses. **Preferences > Add-ons > Halcyon > Developer Options**
turns them all on and off together, and they are hidden by default.

With it on, a **Debug** panel appears in Render Properties holding:

- **Render Pass** — depth, normal, UV, material ID, overdraw, wireframe
- **Timing Breakdown** — the per-frame, per-stage table
- **Halcyon Diagnostics** — dumps the exported scene to the console
- **Worker Processes**, in a box marked Experimental, with its speedup honestly
  described as unverified

The Performance panel is back to the settings that affect a normal render:
threads, shadow caching, fast background, tiles and preview scale.

### Added — Strict Node Evaluation

A preference, alongside Developer Options. A node that raises is normally caught
and replaced by passing its first input through, which still produces
plausible-looking variation — that is precisely how a crash inside the Noise
texture survived three rounds of testing that all reported it working. Strict
mode raises instead.

`HALCYON_DEBUG=1` still works for headless runs.

### Fixed

- The diagnostics and palette-cache operators were **defined twice** in `ui.py`,
  a leftover from a patch applying the same replacement to two sites. Python
  took the later definition so both worked, but three kilobytes of dead code
  were shipping. Removed.
- Developer settings are now preserved when a preset is applied, alongside the
  machine settings. Loading a preset should not silently switch off the timing
  breakdown you were reading.

---

## [1.2.4] — 2026-07-27

### Fixed — one transparent material cost a whole frame of ray tracing

`_add_raytraced` decided whether to trace secondary rays with `np.any(...)` and
then traced them for **the entire batch**. A single transparent fragment
anywhere in a chunk of up to 262,144 meant a refraction ray for every fragment
in that chunk, recursively to the ray depth. A scene with one sheet of glass paid
to refract the whole frame, which is why turning ray tracing off took sixteen
seconds down to two.

Rays are now traced only for the fragments that want them — reflection where
`reflect > 0`, refraction where `opacity < 1`. `trace()` also stopped shading the
world for every ray before overwriting most of them; it does the background only
for the rays that actually miss.

### Fixed — Opaque and Screen Door rendered nothing

`_split_by_alpha` ignored the transparency mode, so transparent geometry was
always pulled into a separate pass. For **Opaque** and **Screen Door** that pass
was never shaded, so the geometry simply vanished.

Neither mode wants a separate pass at all — their geometry belongs in the
depth-buffered pass with everything else:

- **Opaque** now ignores alpha entirely, which is what it should always have
  meant.
- **Screen Door** is implemented properly: an ordered threshold decides whether
  each pixel is drawn or dropped, with no blending, exactly as hardware without
  an alpha unit managed it. The pattern follows the Stipple Pattern setting.

### Fixed — Sorted Blend and A-Buffer were the same thing

Both ran the identical per-fragment path, so the setting did nothing. They are
different techniques and now behave like it:

- **Sorted Blend** orders whole polygons by centroid depth — what a renderer
  without per-fragment lists could manage — and shows the classic sorting errors
  where surfaces interpenetrate.
- **A-Buffer** sorts every fragment on its own depth and is correct through any
  arrangement.

On convex geometry they agree, which is expected. On two interpenetrating
transparent sheets they visibly part company, and a test checks exactly that.

---

## [1.2.3] — 2026-07-27

### Fixed — transparency was 92% of the frame

A timing breakdown came back with **transparency at 15.3 seconds of a 16.6
second frame**. That stage had never been profiled once. Two faults in it, and
both explain a CPU meter sitting at one core's worth of load:

**The A-buffer shading was single-threaded.** `_composite_abuffer` called
`job.shade()` directly instead of going through the chunked thread pool, so
every transparent fragment in the frame was shaded on one thread regardless of
the thread count. It now uses the pool, like the opaque pass, and the result is
identical at 1, 4 and 16 threads.

**Compositing was quadratic in depth.** It walked the depth layers with
`rank == r`, which is a full boolean scan and a full gather across *every*
fragment in the frame, once per layer. A pixel a hundred layers deep in glass
therefore cost a hundred passes over millions of fragments. Sorting by layer
first makes each one a contiguous slice, so the whole composite is a single pass.

### Added — Maximum Transparent Layers

Beyond a handful of overlapping transparent surfaces the furthest ones
contribute almost nothing while costing exactly as much as the first. The limit
defaults to 16 and keeps the *nearest* fragments, which are the visible ones; 0
restores unlimited.

On a test scene of fourteen stacked glass spheres, capping at 16 differed from
unlimited by 0.00001 mean — invisible — and the three changes together took the
frame from 4.48 s to 2.21 s **on a single core**. The threading half of that has
not been measured on a machine with more.

### Note

`transparency` is in the breakdown because the previous release added the
unaccounted-for row and the missing stage timers. Three earlier attempts at this
problem optimised palettes, rasterising and shading — none of which was ever the
bottleneck on this scene.

---

## [1.2.2] — 2026-07-27

### Fixed — the timing breakdown was lying

A report came back showing stages summing to 0.081 s against a total of 19.491 s.
Over nineteen seconds were unattributed, and `export scene` and `shade` were not
listed at all. That is worse than having no report, because it points at the
wrong stage with apparent authority — it named "deliver to Blender" as the
slowest thing at 0.3% of the frame.

Three faults:

- **The export timing was recorded and then erased.** `ST.reset()` ran a second
  time after the export, because `started = time.time()` sits after it in
  `render()` and the reset had been attached to that line. Now reset once,
  before the export, and the total is measured from there.
- **Whole regions were never instrumented.** The background and sky, the BVH
  build, transparency compositing, wireframe, the supersample resolve and the
  worker pool all ran untimed. On a frame looking mostly at sky, the entire cost
  was in one of them.
- **The report did not check its own arithmetic.** It now prints an
  **unaccounted for** row whenever the parts do not sum to the total, and
  refuses to name a slowest stage at all when most of the frame is untracked.
  A test asserts both.

### Added — Fast Background

The background was evaluated once per *supersample*. At 4x that is sixteen
procedural sky evaluations or HDRI lookups for every output pixel, and a sky is
smooth almost everywhere. It is now evaluated at output resolution and expanded,
which on a test frame with a Bryce sky cut the render from 2.50 s to 1.40 s with
**no measurable difference in the image**.

On by default; turn it off if a sharp sun disc or a detailed HDRI shows aliasing.

### Note

`background / sky` appearing in the breakdown for the first time is the point of
this release. If a frame is mostly sky, that line is where it went, and nothing
before now would have shown it.

---

## [1.2.1] — 2026-07-27

### Added — a timing breakdown, because the guessing has to stop

**Timing Breakdown** in the Performance panel prints where each frame actually
went, to the system console:

    ----------------------------------------------------
    Halcyon frame breakdown              time     share
    ----------------------------------------------------
      export scene (Blender side)      0.412s     31.4%
      prepare textures                 0.008s      0.6%
      shadow maps                      0.107s      8.2%
      rasterise                        0.061s      4.7%
      shade                            0.418s     31.9%
      post processing                  0.290s     22.1%
      deliver to Blender               0.014s      1.1%
    ----------------------------------------------------
      slowest stage: shade (32% of the frame)

Three rounds of optimisation were aimed at the bpy-free core, because that is
what could be profiled here. The export layer runs inside Blender and had never
been timed at all. **`export scene` is in that list for the first time**, and on
a scene with large textures it may well be the answer.

### Fixed — Gouraud and flat shading were never threaded

`_shade_interpolated` called `job.shade()` directly, bypassing the chunked
thread pool entirely. Every preset with a vertex or face shading rate — which is
most of the console and home-computer ones, including PlayStation, N64, Saturn,
VGA Mode 13h, EGA and Voodoo — rendered **single-threaded regardless of the
thread count**. That matches a CPU meter sitting at one core's worth of load.

Both paths now go through the pool, with a test that the result is identical at
one thread and four for all three shading rates.

### Added — caching of work that repeats every frame

- **Image pixels.** Reading a texture out of Blender copies the whole buffer
  through the Python API — tens of megabytes for a large one — and it was done
  for every image on every frame. Now cached, keyed on name, size, source,
  filepath, dirty flag and colour space, and never cached for image sequences or
  movies, whose pixels genuinely change.
- **Prepared textures.** Mip building and colour quantisation depend on nothing
  that changes between frames.
- **Shadow maps** (Cache Shadow Maps, on by default). Each map is a full
  rasterisation pass and a point light needs six. They are reused while the
  lights and geometry hold still, keyed on a fingerprint of both. Roughly halved
  a 320x240 test frame; moving a light correctly rebuilds them, and a test
  checks the output is unchanged either way.

---

## [1.2.0] — 2026-07-27

### Added — worker processes

**Use Worker Processes** in the Performance panel splits each frame across
separate Python interpreters. Threads only run in parallel where NumPy releases
the interpreter lock, which covers the array work but not the Python around it —
node evaluation, per-material dispatch, building the shading context. Separate
processes have no shared lock at all.

This is only possible because `core/` and `shaders/` import nothing from bpy: a
worker is a plain Python interpreter with NumPy, and Blender bundles one next to
`sys.prefix`. That boundary was built for testability and turns out to have paid
for itself twice.

How it works:

- `render()` gained a `band` argument for a range of output rows. A test asserts
  bands rejoin bit-identically to a whole-frame render, at 1x and 2x
  supersampling.
- The scene is pickled to each worker **once** and reused for every band and
  every frame after, so shipping it is not paid per tile.
- Compiled GLSL/HLSL shaders could not be pickled — the generated function is
  not a picklable object. `Program` now serialises the source it generated and
  re-compiles on arrival, with a test that a pickled shader still runs
  identically.
- Workers persist between frames. Starting an interpreter and importing NumPy
  costs a good fraction of a second, which would swamp the saving on one frame.
- Bands are cut three per worker so a slow one cannot leave a core idle.

**Every refusal is a reason, not a crash.** No interpreter found, a worker that
will not start, a frame too small to be worth splitting, a scene that will not
pickle — each returns an explanation, prints it to the console and renders in
Blender as before. A slow render is a much better failure than none.

### Measured, and not measured

Output is **bit-identical** to rendering in-process, verified at 320x240 across
three workers and again on a second frame through the same pool.

The speedup is **not measured**. This machine has one core, so running the pool
here is slower than not — it pays the pipe traffic and gets no parallelism back.
The mechanism is verified; the benefit is not. It is off by default for that
reason. Turn it on, render a frame each way, and the console will say if the
pool declined.

The split covers the renderer. Post-processing still runs in Blender's process,
so on a heavily quantised preset — where the palette stage dominates — expect
less from this than on a 24-bit one.

---

## [1.1.1] — 2026-07-27

### Changed — error diffusion is not actually serial

Floyd-Steinberg and its relatives are usually treated as strictly sequential,
because each pixel needs the error from the pixel to its left. That is true
pixel to pixel, but it does not make the *image* serial: a pixel only ever
depends on neighbours up and to the left, so every pixel lying on the skewed
diagonal `x + b*y = t` is independent of every other pixel on it.

For Floyd-Steinberg the four sources sit at offsets whose `x + 2y` are all
negative, so `b = 2` is a valid schedule and a whole diagonal can be quantised
in one go. The skew is derived per kernel from its own offsets, so Stucki, JJN,
Burkes and Sierra get `b = 3` and Atkinson and Sierra Lite get `b = 2`, and a
kernel that admits no schedule is refused rather than silently mishandled.

Two details make it exact rather than approximate:

- Within one diagonal and one kernel offset the targets are provably distinct —
  two different source pixels cannot collide under a fixed translation — so
  plain fancy indexing accumulates correctly and the much slower unbuffered
  `np.add.at` is not needed.
- The buffer is float64, matching the sequential path, which accumulates in
  Python floats. Mixing float32 and float64 made the two disagree on borderline
  pixels.

**2 to 3 times faster, and bit-identical** — a test compares the two paths
exactly, index map and pixels, for all seven kernels.

This only works in one scan direction. Serpentine traversal alternates the
direction per row, which breaks the schedule, so that path stays sequential. The
Colour Depth panel now says so next to the checkbox, since turning serpentine
off roughly halves the frame time.

### Measured

640x480 at 4x supersampling, VGA 256-colour preset, on one core:

| | first frame | steady state |
|---|---|---|
| before 1.1.0 | — | ~6.4 s |
| 1.1.0 | 1.46 s | 0.99 s |
| 1.1.1, serpentine off | 1.37 s | **0.51 s** |

About twelve times faster than where this started, without changing a pixel of
the output.

---

## [1.1.0] — 2026-07-27

### Fixed — render time

Profiling a 640x480 frame with a 256-colour preset found the cost was not in the
renderer at all. **The palette stage took 6.2 seconds, more than the entire
render.** Three separate faults, all in the same place:

**The nearest-colour cube was computed as one enormous temporary.** Building it
compares every cell of a 6-bit RGB cube against every palette entry — 67 million
distances — and the chunking arithmetic collapsed to a single block, allocating
roughly **800 MB** at once. The machine spent its time in the memory system
rather than doing arithmetic.

Rewritten using the expansion `|c-p|² = |c|² - 2c·p + |p|²`. The `|c|²` term is
constant across the palette for any given cell, so it cannot change which entry
is nearest and simply drops out of the argmin — which leaves a single matrix
product that BLAS handles properly.

**6106 ms to 610 ms, exactly ten times faster**, and a test asserts it still
picks the same palette entry as brute force on every one of 2000 random cells at
three palette sizes.

**The cube was rebuilt on every frame**, and error diffusion built a second copy
internally. It depends only on the palette, so it is now cached.

**Error diffusion indexed NumPy scalars per pixel.** The dependency on the pixel
to the right makes it genuinely serial — it cannot be vectorised like the rest of
the engine — but element-by-element ndarray access costs far more than plain
Python floats, and there are three reads and several writes per pixel. The
buffer is unpacked to flat Python floats for the loop and packed back after:
**1409 ms to 698 ms**.

Measured on a five-frame animation at 640x480 with 4x supersampling and a VGA
256-colour preset: **6.4 s per frame down to 0.83 s**, and the first frame is
the only expensive one.

### Added — Lock Palette

An adaptive palette recomputed every frame is not only slow, it makes the
colours crawl: median cut lands somewhere slightly different each time, so flat
areas shimmer between frames. **Lock Palette** builds it once and reuses it, and
is on by default because for animation it is both faster and correct. A
**Rebuild Palette** button next to it releases the lock when the scene has
changed enough to want a new one.

### Note on what was not the problem

The renderer itself was 0.20 s of that frame. Threading, the batched rasteriser
and chunked shading were all working as intended; the time was going somewhere
nobody had measured. Worth remembering the next time something feels slow.

---

## [1.0.11] — 2026-07-27

### Added — free-software notice

> This Addon is and always will be free. If you paid for this, you were
> scammed. Please demand your money back and report the seller.

Shown in four places, chosen so that someone who was sold a copy actually
encounters it:

- **Preferences > Add-ons > Halcyon**, as an alert box alongside a note that the
  add-on is GPL-3.0-or-later and nobody is entitled to charge for it. This is
  the one that matters — a buyer of a resold copy will never see the store page,
  but they will open the add-on's own preferences.
- The **Halcyon Presets** panel, under the resolution menu.
- The top of the store listing, immediately under the tagline.
- The top of the README, and again under Licence in both documents.

The in-Blender text wraps to the panel it is drawn in rather than being cut off
at a fixed width.

---

## [1.0.10] — 2026-07-27

### Changed

- Minimum Blender version raised to **5.1**, in the manifest and in `bl_info`.
  The listing now says 5.2 is what it was tested against and 5.1 is supported
  but untested, rather than claiming a 4.2 floor nobody has tried.
- Credited to **Built by Claude with help from Mr. Emotiman** in the manifest
  maintainer field, `bl_info`, the README and the store listing.

### Added

- `EXTENSION_LISTING.md`: the store description for extensions.blender.org.
- Manifest brought up to submission standard — `tags` was missing entirely and
  is required, and the version string had been left at 1.0.0 since the first
  release.

### Note

The manifest maintainer still ends in a placeholder email address
(`<your@email.address>`), which must be replaced before the extension is
submitted.

---

## [1.0.9] — 2026-07-27

### Fixed — presets overlapped each other

`apply_preset` only wrote the keys a preset mentions, so everything it stayed
silent about kept whatever the previous preset had left. Going from **EGA** to
**Infini-D 4** carried EGA's 16-colour palette, its 2-light limit, its 1.2 pixel
aspect and its 3x output scale straight into a preset that is supposed to be
clean 24-bit — because Infini-D's entry does not mention any of them.

Applying a preset now resets every setting to its default first. Each preset is
a complete description of a look rather than a patch on whatever came before.

Machine and pipeline settings are deliberately preserved, since they describe the
computer rather than the look: **Threads**, tile size, bucket order, preview
scale, progressive, stats, render pass, seeds, texture memory and **Transparent
Film**. A test asserts these survive a preset change and that everything else
does not.

On the Blender side the reset uses `property_unset`, which restores each
property's registered default — and those defaults are generated from the same
dataclass the bpy-free path resets to, so the two cannot disagree.

Three buttons in the Presets panel:

- **Apply Preset** — reset, then apply
- **Add On Top** — apply without resetting, for deliberately layering one preset
  over another
- **Reset All** — back to defaults

### Added — Halcyon Default preset

A **Default** entry under a new *General* category. Selecting it returns every
setting to its default, which is also exactly what the reset does before any
other preset is applied.

### Added — 27 more presets, bringing the total to 52

**3D software** (now 20): ElectricImage 2.9, Softimage|3D, Alias PowerAnimator,
Wavefront Advanced Visualizer, CINEMA 4D v4, Real 3D 2, Vistapro, Hash
Animation:Master, POV-Ray 2.2, Vue d'Esprit 2.

**Home computers** (now 15): Atari ST, Amiga AGA 256, CGA 4-colour, Hercules
mono, Macintosh 1-bit, NEC PC-98, Sharp X68000, Windows 3.1, SVGA High Colour.

**Consoles** (now 8): Sega Dreamcast, 3DO Interactive, Atari Jaguar, PlayStation
high-res.

**Broadcast** (now 4): VHS tape (third-generation dub, with the chroma smearing
and ringing that implies) and S-Video (luma and chroma kept apart, so no dot
crawl).

**Early web** (now 4): CD-ROM full-motion video and PNG-8 sprite.

All 52 render without failure, and a test checks every preset is complete, sits
in a real category, names only settings that exist, and has a unique label.

---

## [1.0.8] — 2026-07-27

### Fixed — the material slot list was missing, and much else with it

A custom render engine only sees the stock property panels that name it in their
`COMPAT_ENGINES`. Halcyon carried a hand-written list of 42 panel names, and
`MATERIAL_PT_context_material` — the slot list at the top of the Material tab —
was not on it. With Halcyon selected there was no way to reach any material but
the active one, which makes a model with several materials effectively
uneditable.

Auditing the rest of the list found the same problem elsewhere. Also missing:

- **UV Maps** (`DATA_PT_uv_texture`) — no way to manage UVs, on a renderer whose
  textures depend on them
- **Colour Attributes** and **Vertex Groups**
- **Shape Keys**, **Normals**, **Custom Data**
- **Material viewport display**
- **Colour Management** (see below)
- **World viewport display**

The hand-written list is gone. Halcyon now discovers panels: any stock property
panel that marks itself engine-agnostic by listing `BLENDER_RENDER` in
`COMPAT_ENGINES` is adopted automatically, plus a short forced list for panels
Blender does not mark that way but which are needed here. A small exclusion list
covers features this engine genuinely does not implement — Freestyle and grease
pencil — because a control that silently does nothing is worse than an absent
one.

This fixes the whole class of "panel X is missing" rather than one instance of
it, and it will not rot with the next Blender release. A test asserts the
essential panels are adopted, that foreign and unsupported ones are not, and
that unregistering removes the engine from every panel it touched.

### Added — material list in the Halcyon panel

For objects with more than one material slot, the Halcyon Material panel now
shows its own slot list with add, remove and select-by-material buttons. Each row
reports the model that slot resolves to and whether it has been converted, with
a running "N of M slots converted" count underneath — so on a multi-material
model it is obvious at a glance which materials still need converting, which the
stock list cannot tell you.

### Added — view transform warning

Halcyon hands Blender pixels that are already display-referred: gamma applied,
palette quantised, dithered. Blender then applies the scene's view transform on
top, and since 4.0 that defaults to **AgX**, which washes the result out and
undoes a good deal of what the engine just did.

The Display panel now detects this and says so, with a one-click button to set
the view transform to Standard. `RENDER_PT_color_management` is also reachable
now, so the setting can be found at all.

---

## [1.0.7] — 2026-07-27

### Added — material conversion

Three buttons at the top of the Halcyon Material panel:

- **Convert This Material**
- **Convert Selected Objects**
- **Convert Whole Scene**

They rebuild each material around the Halcyon Shader. This is deliberately not a
reset button: whatever was feeding the source shader — image textures, ramps,
whole node networks — is **relinked** onto the equivalent input. A material with
a texture in Base Colour comes out with that same texture in Diffuse Colour.

What carries over:

| source | becomes |
|---|---|
| Base Colour, Colour | Diffuse Colour |
| Metallic | Metalness |
| Roughness | Roughness, and the specular exponent derived from it |
| Specular / Specular IOR Level | Specular Level |
| Emission Colour | Self-Illumination |
| Alpha | Opacity |
| IOR, Anisotropic, Anisotropic Rotation, Normal | the matching inputs |
| Transmission | opacity and reflection |

The model is chosen from what the source actually was: Diffuse becomes Lambert
(or Oren-Nayar when rough), Glossy becomes Phong or Cook-Torrance by roughness,
Emission becomes Constant, Glass becomes Blinn, Toon becomes Toon, a metallic
Principled becomes Metal, an anisotropic one becomes Anisotropic. A **linked**
Metallic or Anisotropic counts even when its constant reads zero, because a
texture driving it means the parameter matters.

Thirteen source shader types are mapped, and anything unrecognised still
converts — colour and normal are carried and the model falls back to Phong
rather than the conversion failing.

Details:

- Socket names are matched through alias lists, so Blender 3.x's `Specular`,
  `Emission` and `Transmission` work alongside 4.x's `Specular IOR Level`,
  `Emission Color` and `Transmission Weight`.
- Mix Shader and Add Shader collapse to their dominant branch — a single master
  shader cannot carry two — and the operator says so in the console.
- Reroutes are followed through.
- **Keep Original Shader** is on by default: the old node is muted and moved
  aside rather than deleted, so a conversion can be inspected and undone by eye.
- Per-material results are printed to the console; the header reports how many
  were converted, skipped and failed.
- Materials already using the Halcyon Shader are skipped unless Reconvert is set.
  The single-material button sets that automatically when the material has one.

The decision logic is in `core/convert.py`, which imports nothing from bpy, so
the mapping tables and model choice are covered by tests — including one that
asserts **every socket named in the mapping actually exists on the master
shader**, which is the failure mode a table like this invites.

---

## [1.0.6] — 2026-07-27

### Fixed — the background was always transparent

`_background_image` allocated a zeroed RGBA buffer and only ever wrote the
colour channels, so background alpha stayed at 0 on every render. There was no
setting to change it because film transparency had never been implemented at all.

There is now a **Transparent Film** toggle in the Display panel, defaulting to
off, and Blender's own **Film > Transparent** switches it on as well — that is
where people look for it, so it wins when set. Wireframe's see-through pixels
follow the same rule instead of always punching a hole.

### Added — 12 procedural texture nodes

Solid textures, evaluated in 3D, so a shape carved out of marble has veins
running through it rather than a picture wrapped around it. These are the
patterns that came in the box with POV-Ray, 3D Studio, Infini-D and Bryce's Deep
Texture Editor:

| node | what it is |
|---|---|
| **Marble** | Sine banding displaced by turbulence, with vein axis, count and sharpness |
| **Wood** | Concentric rings around an axis, warped by turbulence, plus fine grain |
| **Granite** | Stacked high-frequency octaves, contrast-stretched into speckled stone |
| **Dents** | Sparse pits from a Worley field — 3D Studio's material of the same name |
| **Crackle** | The boundary network between cells: crazed glaze, dried mud, veins |
| **Plasma** | Interfering sine fields with optional palette cycling. The demoscene one, and it animates |
| **Ripples** | Concentric waves from several point sources, interfering, animated |
| **Starfield** | Placed stars with size, density and twinkle |
| **Weave** | Over-under fabric with separate warp and weft colours and a Thread output |
| **Scratches** | Fine anisotropic scuffs for brushed and worn metal |
| **Tiles** | Rectangular tiles with grout, row offset, bevel shading and per-tile colour variation, with a Tile ID output |
| **Spiral** | Archimedean spiral banding around an axis |

They live under **Add > Halcyon > Halcyon Textures** in the shader editor. Each
one has Colour and Fac outputs; Weave adds Thread and Tiles adds Tile ID.
Plasma and Ripples read scene time, so they animate across a frame range.

The node classes and their evaluators are generated from and checked against one
shared spec table, so a socket cannot drift away from the code that reads it —
the same arrangement as the render settings and their dataclass.

New `core/patterns.py` holds the pattern maths and the noise primitives, which
`core/sky.py` now shares rather than keeping a second copy of.

### Added — tests

- Every pattern node is driven through the evaluator with graphs built from the
  spec table, asserting each one produces every declared output, **raises
  nothing** (the silent-fallback trap that hid the Noise bug), and varies.
- Every pattern node is rendered through the full material pipeline.
- Film transparency is asserted in both states, including that geometry stays
  opaque either way.

---

## [1.0.5] — 2026-07-27

### Changed — the Bryce atmosphere, rebuilt properly

The previous version was a gradient with a noise layer on top. This is the Sky &
Fog stack Bryce actually had, layer for layer, in the order it composited them:

    sky dome gradient
      + sun corona (tight core plus a wide outer halo)
      + haze, thickening toward the horizon and taking the sun's colour near it
      + stratus deck (high, wind-streaked, wispy)
      + cumulus deck (low, billowy, self-shadowed, sun-rimmed)
      + ground-hugging fog
      + rainbow
      + stars
      + sun disc

**Clouds are built from turbulence, not fBm.** Bryce's cumulus come from summed
`|signed noise|`, and the cusps where that crosses zero are the cauliflower
edges on every Bryce cloud. Plain fBm cannot produce them — it gives smooth
blobs. Stratus uses stretched fBm instead, because that layer is wispy rather
than billowy.

**The decks are projected onto planes at altitude**, so they compress toward the
horizon the way a real cloud layer does, rather than being painted on the dome.

**Cumulus are shaded, not stencilled.** A second noise sample taken further along
the ray gives the deck bulk: where it reads thicker, the underside goes into
shadow. Combined with a sun-side rim colour, the deck has form instead of looking
like flat cut-outs.

**Haze and fog are separate**, as they were in Bryce. Haze is atmospheric
perspective — it thickens with distance toward the horizon and warms toward the
sun. Fog hugs the ground with its own colour and height.

New controls, grouped into Bryce Sky Lab style sub-panels (Sun & Corona,
Atmosphere, Cumulus, Stratus, Rainbow & Stars):

- **Sun**: altitude, azimuth, colour, intensity, glow, corona width, disc, size
- **Haze**: amount, colour, height, sun tint
- **Ground Fog**: amount, colour, height
- **Cumulus**: cover, opacity, frequency, altitude, thickness, fuzziness,
  detail, seed, top colour, base colour, sun rim
- **Stratus**: cover, opacity, frequency, altitude, streak, fuzziness, detail,
  colour
- **Rainbow**: intensity, radius, width, secondary bow
- **Stars**: density, brightness

**The rainbow uses the real geometry** — a band at 42° from the antisolar point
with red outermost, and an optional dimmer secondary at ~51° with the spectrum
reversed. A test asserts the bow lands at 42° and that red is on the outside.

Sun elevation and rotation are relabelled **Sun Altitude** and **Sun Azimuth**,
which is what Bryce called them.

### Added

- Tests covering all 16 Bryce controls individually, and the rainbow's angular
  geometry.
- Sky sampling in the tests now uses a proper hemisphere grid. The previous
  spiral of sample directions could miss a thin band like a rainbow completely,
  and did.

---

## [1.0.4] — 2026-07-27

### Fixed — the Noise texture, and why my own tests missed it

`fractal_noise` evaluated `if distortion:` on a NumPy array, which raises
`ValueError`. The node evaluator caught it and fell back to pass-through, which
hands the node's first input straight to its output.

That fallback still produces plausible spatial variation, so every test I had
written — all of which asked "does this vary across the surface?" — passed while
the node was completely broken. **Noise was never running.** The same silent
fallback is the likeliest reason Factor outputs looked dead: pass-through
resolves a Colour output to the incoming vector, which varies, while a Factor
output resolves to something constant.

Two changes so this cannot hide again:

- Node exceptions now record the node id, type and error, and surface as render
  warnings instead of vanishing into a set of bare type names.
- A new test drives every texture node and **fails if any of them raises at
  all**, rather than checking the output merely looks non-uniform.

### Added — a real sky system

The Halcyon world settings could never take effect, because Blender worlds
always carry a node tree and the node tree was checked first. There is now an
explicit **Sky** mode that wins over the node graph:

- **Use Node Tree** — previous behaviour, still the default
- **Solid Colour**
- **Gradient** — horizon/zenith with falloff, blend curve, horizon height and an
  optional ground colour
- **Bryce Atmosphere** — the layered stack Bryce actually used: sky gradient,
  sun glow, a haze band with its own colour, density and height, and a fractal
  cloud deck with cover, density, scale, height, detail, softness, seed, and
  separate lit and shadow colours
- **Physical Sky** — Preetham analytic daylight with turbidity and ground albedo
- **Image / HDRI** — equirectangular or mirror-ball, with rotation, tint,
  strength and filter

Sun elevation, rotation, colour, size, intensity and disc are shared by the
Bryce and Physical modes. The whole sky rotates on one control.

### Added

- **Gabor noise** (`ShaderNodeTexGabor`), bringing the evaluator to 93 node
  types.
- **Noise `noise_type`** now works. Blender folded the Musgrave node into Noise
  in 4.1; Multifractal, Ridged Multifractal, Hybrid Multifractal and Hetero
  Terrain each build their octaves differently and were all rendering as plain
  fBm.
- **Halcyon Diagnostics** operator in the Performance panel. It prints the
  actual exported scene — every node, its properties and its links, the world
  mode, materials, lights and resolved settings — to the system console. If
  something still looks wrong, that output says exactly what the engine received.

### Fixed — render passes

Data passes were being run through the period display chain, so a depth ramp got
quantised to the palette and an overdraw heat map snapped to a single blue. Any
pass other than Beauty now bypasses colour depth, dither, palettes, CRT
simulation, composite and JPEG entirely; only the display gamma is applied.

Overdraw is additionally normalised across its real range rather than against
its maximum, so uniform coverage no longer reads as one flat colour, and it is
forced onto the sequential rasteriser, which is the path that counts fragments.

### Fixed — Wireframe as a material override

`material_model` returned nothing for a material that had a node tree but no
Halcyon Shader node — and Blender materials always have a node tree. A material
set to Wireframe in the Halcyon panel therefore shaded as flat unlit colour and
never had its edges carved out, which is why it looked like plain emission. The
material-level override is now authoritative.

---

## [1.0.3] — 2026-07-27

Sky, textures and wireframe, all reported from real use.

### Fixed — the sky was black

Two independent faults, both of which produced an unlit background.

- **The background pass was handed an empty texture dictionary.**
  `_background_image` called `world_color(..., {}, ...)`, so any world driven by
  an **Environment Texture** — an HDRI, the usual way to make a sky — resolved
  its image to nothing and rendered black. The scene's prepared textures are now
  passed through.
- **`ShaderNodeTexSky` was not implemented at all.** It fell through to the
  unknown-node handler, which passes the first matching input through; with only
  a Vector input to offer, that is black. Implemented as the **Preetham**
  analytic daylight model (Preetham, Shirley & Smits, SIGGRAPH 1999) with sun
  elevation, rotation, turbidity, ground albedo and a sun disc.

Also fixed: `world_color` looked the environment image up by `id(...)` while
`prepare_textures` keys by name, so the non-node environment path could never
find its image either.

Ambient always worked because it comes from the light accumulator, not the world
graph — which is why the symptom was "ambient works, sky doesn't".

### Fixed — procedural textures came out flat

**Generated coordinates were normalised over the whole scene instead of per
object.** Blender normalises them over each object's own bounding box. With a
large ground plane in the scene, a normal-sized object's Generated coordinates
spanned about **1% of the range they should**, so every procedural texture
sampled a tiny patch of its own pattern and resolved to flat colour.

Checker appeared to survive only by accident: it sums the floors of all three
axes, and the one axis that still varied was enough to keep a visible pattern.
That is why Checker looked fine while Noise, Voronoi, Magic, Gradient, Wave and
Brick looked broken.

Per-object bounds are now computed once and indexed per fragment. A test asserts
a small object in a large scene still gets a full 0..1 span, and another asserts
every procedural texture produces real spatial variation.

### Fixed — the Wireframe model rendered on black

A regression from 1.0.1. The background pass was optimised to shade only the
pixels geometry does not cover, but the wireframe path used that same buffer to
fill its see-through pixels — and it is empty exactly where geometry is. Every
non-edge pixel of a wireframe material was therefore set to black instead of the
world behind it, so the model read as "not working" even though the edges were
being drawn correctly.

Wireframe now shades the world along the camera rays of the pixels it clears.

### Added

- `ShaderNodeTexSky` (Preetham), bringing the evaluator to **92 node types**,
  with its properties added to the exporter.
- Regression tests for all five faults above: the three world background types,
  per-object Generated span, procedural texture variation, and wireframe showing
  the world behind it.

### Changed

- README documents the Preetham-versus-Nishita difference, and replaces the GPU
  section with a concrete account of why Cycles' device layer is unreachable
  from Python, why EEVEE's `gpu` module is not, and what a staged port would
  involve.

---

## [1.0.2] — 2026-07-27

Fixes the material-preview crash, and makes the engine use the whole machine.

### Fixed — the preview crash

Two independent faults, either of which takes Blender down rather than raising
a catchable error.

- **Null depsgraph passed into `calc_matrix_camera`.** `_projection_from_camera`
  fished a depsgraph out of `scene.view_layers[0].depsgraph`, which is NULL
  during a material preview render. `calc_matrix_camera` is C code that
  dereferences what it is handed, so this is a segfault — and a segfault cannot
  be caught by the `try/except` that was wrapped around it. The real depsgraph
  is now passed in from `export_scene`, and the guard happens *before* the call.

  This is the fault that fired every time a preview thumbnail was generated.

- **Render-result buffer overrun.** `_deliver` sized the render result from the
  *post-processed* image and then wrote every pixel of it into `rect`. The post
  chain legitimately resizes — `Pixel Scale` multiplies by up to 4, pixel aspect
  stretches — so with any preset that sets those, the engine wrote several times
  more floats than Blender had allocated, corrupting the heap.

  The buffer size is now dictated by Blender (`size_x`/`size_y`, which is the
  only correct answer during a preview), the image is fitted to it before a
  single value is written, and a final length check guards `foreach_set`.
  A test asserts `post.process` honours a requested size across every
  combination of Pixel Scale, pixel aspect, CRT curvature, composite and JPEG.

### Fixed — other crash risks found while hardening

- `to_mesh()` / `to_mesh_clear()` were called while the `object_instances`
  iterator was still live, which can invalidate it. The instance list is now
  snapshotted first.
- `image.pixels.foreach_get` was called on images with no pixel buffer, reading
  from a null pointer. `has_data` is checked, with an `update()` retry.
- `gpu.types.Buffer` could be handed a buffer of the wrong length in the
  viewport path. Length is now checked and non-finite values scrubbed.
- Warnings and errors are no longer reported from preview renders, which pushes
  UI work onto a background thread for a thumbnail nobody reads.

### Changed — `Pixel Scale` now means what it should

Previously it multiplied the output size, which is what overran the buffer.
It now renders at **1/N of the output resolution** and scales back up with
nearest-neighbour. The output stays exactly the size you set, and the render
costs N² times less — 1920×1080 at 4× Pixel Scale renders 480×270 internally.
That is both the useful behaviour and the safe one.

Blender applies its own pixel aspect at display time, so the post chain no
longer applies it again in the engine path.

### Added — threaded shading

Shading now runs across a thread pool. Deferred shading makes every fragment
independent, so the frame is split into bounded chunks and distributed; the
workers only touch bpy-free code and read-only shared state, and NumPy releases
the GIL for array work, so the threads do real parallel work.

- **Threads** is exposed in the Performance panel, with the detected core count
  shown beside it. 0 follows Blender's own setting.
- Per-worker RNGs, and the lazily-cached tables are pre-warmed before any worker
  starts, so the workers are pure readers.
- A test asserts the result is **bit-identical** at 1, 4 and 20 threads.

Speedup could not be measured here: the development sandbox has one core. The
threading path is verified correct, not verified fast.

### Added — bounded memory

Shading is chunked whether or not threads are in use. A 3440×1440 frame at 4×
supersampling is 79 million fragments, and building the shading context for all
of them at once would want tens of gigabytes; peak memory is now a function of
chunk size and thread count rather than resolution.

### Changed — viewport

The exported scene is cached and rebuilt only when `view_update` reports a
change. Orbiting previously re-converted every mesh in the scene on every
redraw.

### Added

- Descriptions for `threads`, `preview_scale` and `output_scale` explaining what
  each actually costs.
- README gains a *Threading* section and reorders the performance advice to put
  `Pixel Scale` near the top, since it is the cheapest large win available.

---

## [1.0.1] — 2026-07-27

Fixes for three problems found on first install in real Blender.

### Fixed

- **The render was upside down.** `_deliver` flipped the image vertically before
  handing it to Blender. It should not have: the rasteriser maps NDC y = −1 to
  row 0, so row 0 is already the bottom of the picture, which is exactly the
  order `RenderResult` expects. The flip was introduced by a docstring on
  `render()` that wrongly claimed "top row first"; the docstring has been
  corrected and now explains the convention and who needs to flip (PIL and most
  image libraries do — the demo-image writer was silently producing upside-down
  PNGs for the same reason, and now flips explicitly).

  The viewport preview had the identical bug and is fixed too. A test now
  asserts row 0 is the bottom of the frame.

- **Every material rendered white.** The diffuse term was missing its
  Lambertian **1/π** normalisation. Light energy is taken from Blender in
  watt-based units, and Cycles divides reflected radiance by π; without it every
  surface came out π times (3.14×) too bright and clipped to pure white, hiding
  the material colour completely.

  On a default Blender scene — 0.8 grey Principled BSDF, one 1000 W point
  light — this took **28.8% of the frame to pure white**, with lit surfaces
  reaching 2.05 against a display maximum of 1.0. After the fix the same scene
  peaks at 0.68 with nothing clipped, and red, green and blue materials are
  cleanly distinguishable.

  The factor is applied to the diffuse and specular lobes equally, so the
  diffuse/specular balance and the period-correct unnormalised highlight shape
  are unchanged. This is a units conversion, not a change to the reflectance
  models.

  Two tests now guard it: a Blender-default light rig must not blow the frame
  out, and material colour must survive to the framebuffer.

- The demo scene's light energies were hand-tuned around the old, wrong
  exposure. They now use Blender-typical values (sun 6, point 600 W), which
  makes the test scene a fairer proxy for a real one.

### Changed — performance

- **New batched rasteriser.** The per-triangle Python loop is gone from the hot
  path. Triangles are bucketed by bounding-box size class — independently in
  width and height, so a long thin triangle is not padded out to a square — and
  every candidate pixel in a bucket is tested in a single vectorised sweep. The
  depth resolve sorts fragments per pixel and takes the nearest, which selects
  the same winner a sequential depth test would.

  Triangles with large bounding boxes are routed to the sequential path, where
  the per-triangle cost is already amortised by pixel count and bucket padding
  would be pure waste. Below a threshold of small triangles the whole batch is
  delegated to the sequential path, so the common high-supersampling case (where
  every triangle is large in pixels) does not pay setup cost for nothing.

  Measured on one core, complete renders including post:

  | scene | before | after | |
  |---|---|---|---|
  | 782 tris, 320×240 | 0.29 s | 0.22 s | 1.3× |
  | 782 tris, 640×480 AA4 | 2.06 s | 2.09 s | 1.0× |
  | 18.7k tris, 640×480 | 2.43 s | 0.59 s | 4.1× |
  | 18.7k tris, 640×480 AA4 | 5.69 s | 2.20 s | 2.6× |

  Rasterisation alone is up to 15× faster; the gain scales with triangle count,
  because the old path cost roughly 20 µs per triangle in Python overhead
  regardless of coverage.

  The reference implementation is retained and two tests assert the fast path is
  **bit-identical** to it — triangle buffer, depths, perspective-correct and
  screen-linear barycentrics, and the full A-buffer fragment set.

- **Per-fragment object matrices are now built lazily.** Gathering a 4×4 matrix
  for every fragment cost 63 ms on a 300k-fragment frame and almost no node
  graph asks for object space.

- **The background is only shaded where geometry does not cover it.** Previously
  every pixel got a world-colour lookup that was then overwritten.

- **Mesh bounds are cached** instead of being recomputed over every vertex once
  per material batch.

### Added

- `README.md` gains a *Performance* section with the measured figures, the
  settings that actually cost time in priority order, and a straight account of
  what GPU support would require and why it is not in this release.

---

## [1.0.0] — 2026-07-27

First release. Complete engine, built from nothing in a single session.

### Added — renderer core (`halcyon/core/`, no bpy imports)

**`mathx.py`** — vector maths over `(N,3)` / `(N,)` arrays: `dot`, `length`,
`normalize`, `cross`, `reflect`, `refract`, `faceforward`, `saturate`, `mix`,
`smoothstep`, `luminance` (709 and 601), point/direction/normal transforms,
`orthonormal_basis`, sRGB ↔ linear, gamma encode/decode, HSV ↔ RGB, `safe_pow`,
`look_at`, `perspective`, `orthographic`.

**`scene.py`** — the dataclasses the renderer consumes: `ImageBuffer`,
`Material`, `MeshData`, `ObjectInfo`, `Light`, `Camera`, `World`, `Scene`.
Lights carry type, decay mode, spot cone with hotspot, area shape and axes,
shadow controls, and the period flags — negative, diffuse-only, specular-only,
ambient-only, volumetric.

**`settings.py`** — `RenderSettings` dataclass, 131 fields, plus
`RESOLUTION_PRESETS` with 17 period formats (CGA, VGA mode 13h, QVGA, VGA,
Mac Classic, SVGA, XGA, Amiga PAL and hires, NTSC D1, PAL D1, Toaster, PSX,
PSX hi-res, N64, Quake).

**`raster.py`** — projection, Sutherland-Hodgman near-plane clipping, edge-function
triangle fill supporting both windings, perspective-correct barycentrics,
configurable z-buffer bit depth, polygon offset, overdraw counting, front/back
face tracking, and `FragmentList` for unbounded A-buffer capture.

**`texture.py`** — `Texture` with nearest, bilinear, trilinear and the N64
three-point filter; repeat/extend/clip/mirror wrapping; mip chain construction;
size clamping and colour quantisation to emulate texture memory limits;
spherical and equirectangular environment lookups.

**`palette.py`** — fixed palettes (EGA 16, CGA 4, Windows 20, web-safe 216,
6:6:6 and 3:3:2 cubes, greyscale, the real VGA BIOS palette with its 6-bit DAC
structure, the Macintosh System table, Amiga OCS); quantisers (Heckbert median
cut, popularity, Gervautz-Purgathofer octree, k-means); `InverseColormap` with a
6-bit lookup cube; bit snapping; and honest scanline-sequential HAM6/HAM8
encoding.

**`dither.py`** — recursive Bayer matrix generation (2×2 through 16×16),
clustered-dot halftone, and error-diffusion kernels: Floyd-Steinberg,
Jarvis-Judice-Ninke, Stucki, Atkinson, Burkes, Sierra, Sierra Lite. Serpentine
traversal, LUT-driven for speed.

**`shading.py`** — 18 reflectance models, each from its published formulation:
Lambert, Gouraud, Flat, Phong, Blinn-Phong, Blinn, Cook-Torrance, Oren-Nayar,
Minnaert, Ward, Anisotropic, Metal, Strauss, Multi-Layer, Toon, Translucent,
Constant, Wireframe. Schlick and dielectric Fresnel, and 3D Studio's *Soften*
parameter.

**`bvh.py`** — median-split BVH with vectorised node traversal, any-hit
`occluded` and closest-hit `intersect` via Möller-Trumbore.

**`nodeeval.py`** — evaluator for **91 node types**: input nodes, all texture
nodes, colour and vector operations, 35 Math operations, converters, every BSDF
including a full Principled → closure translation, plus reroutes, muted nodes
and recursive node groups. Unknown nodes pass their first matching input through
and are recorded for reporting.

**`lights.py`** — attenuation modes (none, inverse, inverse square, custom
range), spot falloff with blend, orthographic shadow maps for suns, perspective
for spots, six-face cube maps for points, Vogel-disc PCF, area-light sampling,
ray-traced shadows, hardware light-limit emulation, and ambient accumulation.

**`render.py`** — the orchestrator. Camera matrices, texture preparation,
closure-to-surface resolution, direct lighting, ambient occlusion, Shader-to-RGB,
world and environment lookup, fog, `ShadeJob` for shading arbitrary point sets,
per-material shading-rate dispatch, ray-traced reflection and refraction, A-buffer
compositing, six debug passes, five reconstruction filters, and supersample
resolve.

**`post.py`** — separable triple-box Gaussian, glow, cross-screen star filter,
lens flare with ghosts and streaks, display transform (exposure, gamma,
brightness, contrast, saturation, optional tone maps), colour-depth reduction
across ten formats including HAM6/HAM8 and 1-bit, composite NTSC encode/decode
with bandwidth-limited chroma, ringing overshoot and dot crawl, interlacing,
CRT simulation (aperture grille, slot and shadow masks, scanlines, barrel
curvature, vignette, bloom), a real 8×8 DCT JPEG round-trip using the standard
luminance and chrominance quantisation tables, pixel aspect correction and
nearest-neighbour upscaling with optional pixel grid.

### Added — shader compiler (`halcyon/shaders/`, no bpy imports)

- **`lexer.py`** — comment stripping, full preprocessor (object- and
  function-like `#define`, the complete conditional family, `#undef`),
  tokeniser, HLSL type aliases.
- **`gtypes.py`** — the type lattice, swizzle resolution across `xyzw`/`rgba`/
  `stpq`, and promotion rules.
- **`parser.py`** — recursive-descent parser producing a tuple AST. Structs,
  `cbuffer`, HLSL semantics, `in`/`out`/`inout` parameters, arrays, complete
  statement set with `switch` lowered to if-chains, and full expression
  precedence.
- **`builtins.py`** — the GLSL builtin library, **real screen-space
  derivatives** (`dFdx`/`dFdy`/`fwidth` computed by scattering to a 2D grid and
  differencing), texture sampling functions, HLSL aliases, and Halcyon
  extensions (`noise3`, `fbm`, `quantize`, `posterize`, `dither4x4`,
  `hsv2rgb`).
- **`codegen.py`** — AST to NumPy with **SIMT execution masks**, so divergent
  branches and loops with per-lane trip counts execute correctly.
- **`compiler.py`** — `Program` with uniform and output schemas, SHA-1 keyed
  cache, and starter shaders for both dialects.

### Added — Blender integration

- **`compat.py`** — shims across recent API drift: loop triangles, corner
  normals (handling the 4.1 removal of `calc_normals_split`), UV and colour
  layers, evaluated meshes, image pixel access, node menu registration
  (4.0 removed `nodeitems_utils`), and curve/ramp baking.
- **`properties.py`** — **generated from the `RenderSettings` dataclass**, so the
  UI cannot drift from the renderer. 20+ enum tables with real descriptions,
  plus material, light and world property groups.
- **`export.py`** — depsgraph to dataclasses. Per-corner split vertex buffers via
  `foreach_get`, recursive node-tree flattening, colour ramps and curve mappings
  baked to 256-entry LUTs at export time, image collection, coded-shader
  compilation, and camera projection taken from `calc_matrix_camera` so framing
  matches the viewport exactly.
- **`engine.py`** — `HalcyonRenderEngine` with progress reporting, cancellation,
  warning surfacing, a GPU-texture viewport preview, and a compatible-panel set
  of roughly 40 stock panels.
- **`ui.py`** — 21 panels plus preset and resolution operators.
- **`nodes/shader_nodes.py`** — the classic shader node with 19 sockets and
  model-aware socket hiding that never drops a link; the coded-shader node that
  builds its sockets from the compiled shader's own uniform and output
  declarations; and four retro utility nodes (Posterize, Ordered Dither, Depth
  Cue, Screen Info).
- **`presets/library.py`** — 24 presets across software, home computers,
  consoles, broadcast and early web.

### Added — testing

- Headless test suite runnable with no Blender: `python3 -m halcyon.tests.run_all`.
- 16 shader-compiler tests and ~30 renderer assertions.
- `tests/fakebpy.py`, a minimal `bpy` stub that validates registration and
  property declarations, including that all 131 settings fields map across.
- `HALCYON_DEBUG=1` makes node evaluation raise rather than fall back silently.

### Fixed — bugs found during the build

- **Colour socket defaults were misread as per-point data.** `to_value`,
  `to_color` and `to_vector` treated a 1-D array as one value per shading point,
  so an unlinked socket default like `[0.25, 0.5, 0.75, 1.0]` became four
  greyscale points instead of one RGBA constant. This would have broken
  essentially every real Blender material. Constants are now distinguished from
  per-point data by type (`ndarray` vs Python list/scalar) rather than by shape.
- **A scalar socket default crashed the evaluator.** `to_value` indexed
  `a.shape[1]` on a zero-dimensional array.
- **`build_screen_tris` returned five arrays instead of six** on the
  everything-clipped path, breaking `rasterize` whenever a shadow map saw no
  geometry.
- **Cube shadow-map lookups crashed** when given a per-point bias array: the
  positions were subset per cube face but the bias was not.
- **`force_model` was ignored for materials without a node graph**, because
  `closure_to_surface` returned early before the override was applied.
- **Shadow maps had depth acne on curved surfaces**, over-darkening renders.
  Fixed with normal-offset biasing scaled by texel size and light obliquity.
  Map and ray shadows now agree to within 0.0015 mean difference.
- **Ward and Anisotropic shared a code path** and produced identical output at
  zero anisotropy. Anisotropic is now 3D Studio's elliptical Blinn highlight
  with two exponents; Ward remains the Ward Gaussian.
- **The Wireframe model rendered as flat colour.** It now draws actual triangle
  edges, using barycentrics divided by their screen-space derivative so the line
  width stays constant with distance. A global wire overlay was added alongside.
- **Gouraud and Flat did nothing when selected as material models.** They are
  shading *rates*, and now switch that material to vertex- or face-frequency
  shading — per material, not globally.
- **Recursive shaders hung the interpreter.** Recursion is now rejected at
  compile time with the call cycle named, because SIMT masking means a recursive
  call never reaches its base case.
- **`break` and `continue` were broken in the code generator.** The old
  `dir()`-based sentinel idiom didn't work and `continue` failed to skip the rest
  of the body. Replaced with a proper loop stack and per-loop mask variables.
- **Uniforms fell back to zero rather than their declared initialiser.**
- **`narrow()` failed to remove returned, broken and continued lanes**, so
  inactive lanes kept executing.
- **`ham_encode`'s `(image, palette)` tuple return was unpacked as an image.**
- A quoting typo in the PAL broadcast preset (`'label": "PAL broadcast'`) made
  the module unparseable.

### Notes

- Everything under `core/` and `shaders/` is free of any `bpy` import by design.
- Post-chain ordering is physically motivated and documented in `post.py`:
  optical effects in linear light before quantisation, display simulation after.
- Ambient occlusion is off by default; it is not period correct.
- The Blender integration layer is import- and registration-checked against a
  stub but has not been run inside Blender. See *Known limitations* in the
  README.
