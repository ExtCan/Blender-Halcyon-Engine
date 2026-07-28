# Changelog

All notable changes to Halcyon are recorded here. Dates are ISO 8601.

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
