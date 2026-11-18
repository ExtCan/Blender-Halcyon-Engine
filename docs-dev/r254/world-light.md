# R254 -- world-light (World Light: the sky's own sun or moon as a scene light)

Branch `r254/world-lighting`. The integrator owns CHANGELOG.md, README.md,
EXTENSION_LISTING.md, version.py, `__init__.py` and blender_manifest.toml;
the fragments below are written in their voice. Version stays 1.92.0 in
this branch: whether this ships as its own round is the user's call.

## CHANGELOG

- **What was asked.** "Add a feature that will use the world for lighting
  (the Bryce sky's sun/moon will be a light, a cubemap sun will be a light,
  etc)." The world was a backdrop plus an ambient colour: a Bryce sky could
  show a blazing low sun and the scene under it was lit only by whatever
  lamps existed.
- **World Light** (World properties > Halcyon World > World Light, off by
  default). With it on, the luminary the sky DRAWS becomes one ordinary Sun
  lamp at render time -- it lights, casts shadows, ranks under the light
  limit, throws light shafts and reaches the GPU's `hal_lights` texels as
  any lamp does, because past `core/worldlight.sync` it IS a lamp (the test
  pins the frame bitwise against the same Sun lamp made by hand, on every
  world type).
- **What each world type gives** (`core/worldlight.luminary`):
  - *Bryce*: the sun, or the moon when Body is Moon -- the same direction
    (Bryce's one body, swapped), the Moon Colour, softness 2 x Moon Size,
    and `MOON_LIGHT` = 0.25 of the sun's strength scaled by the lit fraction
    of the phase the disc shows with the Earthshine floor (`sky._moon`'s own
    `lit * (1 - shine) + shine`): new moon 0, quarter 1/2, full 1.
  - *Physical*: the sun (the fields Bryce shares).
  - *Nodes*: the first Sky Texture node's sun, in the colour the node draws
    its disc; refused by name when its Vector input is linked (a mapped sky
    has no direction this can know). The node graph does not read the World
    Rotation and neither does its sun.
  - *HDRI and Cube Map*: the brightest region of the image, found once per
    image (`find_image_sun`), or -- Find: Manual -- an Elevation, Azimuth
    and Colour typed into the panel.
  - *Solid, Gradient, Bands, Starfield, Painted, Cylinder, LightWave
    Backdrop*: no luminary. No light is made, the panel says so in place of
    the controls and the console says so once.
- **Direction.** Where the picture shows the body: `sky.evaluate` turns
  view directions by -Rotation before sampling, so the light's direction is
  the sky-space direction turned by +Rotation. The tests sample the sky
  itself, find its brightest point and require the luminary inside the
  disc, under three World rotations. The ocean glitter and the cloud
  shadows on the ground plane already work in sky space with the same
  `_sun_vector`, so they agree with the light without a change.
- **Quantities.** Energy = pi x Sun Intensity x World Strength x World
  Light Strength (pi because the Lambert term divides by it: at 1 a surface
  facing the sun squarely reads exactly its own colour). Colour = Sun
  Colour as set -- no reddening is added, the sky adds none to its disc.
  Shadow softness = the disc's angular diameter (2 x Sun Size) through the
  Sun lamp's own Angle, up to `MAX_ANGLE` = 0.06 rad (the default Bryce
  disc) and no further: measured here, the engine's map shadows blur away
  to nothing as ANY Sun lamp's Angle passes about 0.1 (884 shadowed pixels
  at 0.06, 270 at 0.10, none at 0.15 on the demo scene), and a sky that
  showed a cinema sun and cast no shadow would be the dead control this
  feature exists to remove. The light fades linearly as the disc crosses the
  horizon (none below, half with its centre on it) so a sunset dims out
  instead of popping; image and Manual suns are not cut at the horizon (a
  space skybox has no ground). An image sun's energy is pi x World Strength
  x World Light Strength whatever the texel values: an 8-bit sky clips its
  sun, so the picture's own number means nothing there.
- **The image search** reads the SKY, not the file: the sky's own sampler
  is evaluated over a latitude-longitude grid about as fine as the image
  (512 to 2048 columns), so equirectangular, mirror ball, every cube layout
  and convention and the per-face turns are all covered by the one rule the
  frame draws with. The grid is pooled into ~1.4 degree cells (a hot texel
  alone does not win), the brightest cell is the candidate, and the answer
  is the luminance-weighted centre of the bright texels round it, in their
  colour, with the region's own angular size as the shadow softness
  (capped as above). A wrong
  sun is worse than none, so two cases are refused by name and point at
  Manual: nothing standing out from the sky's median, and a second region
  as bright as the first somewhere else (an 8-bit sky clips its sun and its
  white clouds to the same 1.0). Cached four deep by image identity and the
  sampling dials.
- **Where it is made.** At render entry, beside `collect_exclusive_lights`
  (both shading entries of `core/render.py`), not at export: the lamp-drag
  fast path `export_lights_into` rebuilds `scene.lights` from the lamps
  alone, hand-built scenes never see the exporter, and the image search
  wants the prepared textures. `sync` is idempotent (it drops the light it
  made last time and appends a fresh one, LAST, so no lamp changes its
  index) and with the toggle off and no synthetic light present it does not
  touch the scene at all. A world-only edit re-lights at the next render;
  the GPU plan signature follows through the existing per-light signatures.
- **Beside real lamps.** The world light is added to them: a scene that
  already has a Sun lamp is lit by both, and the panel says so. The Light
  Limit never drops the world light, in all three limit modes: it ranks
  the LAMPS for the slots left beside it, by the limit as the falloff
  package of this round reconciled it (`lights.frame_lights` -- Brightest
  on what a lamp delivers, Nearest from the camera), and a scene at or
  under the limit is not touched.
- **Shadow** (None / Shadow Map / Ray Traced) is the lamp's own Shadow
  choice and obeys the render settings' Shadow Method the same way; under
  Per Light, a Ray Traced world light gets its BVH as any Ray Traced lamp
  does (`render._per_light_ray_shadows`).
  **Light Shafts** is the lamp's Volumetric amount: the shaft pass streaks
  from the sun's own vanishing point on screen.
- **Presets.** The eight `world_light*` fields are excluded from sky
  presets and sky files: the toggle is a decision about the scene's
  lighting, and picking a sky neither switches it off nor ships it.
- **Neutral.** Off by default; with it off every World Light dial is inert
  on every world type (bitwise frames), the scene's light list is the same
  object holding the same lamps, and no existing render moves.
- **Files.** `core/worldlight.py` (new), `core/scene.py` (eight World
  fields), `core/render.py` (two sync calls),
  `core/lights.py` (`select_lights`), `properties.py`, `ui.py`
  (`HALCYON_PT_world_light`), `presets/skies.py` (EXCLUDED),
  `tests/test_r254_world_light.py`, `tests/featurematrix.py` (the row
  "world light (HDRI sun, no lamps)"), `tests/run_all.py`.

## README

**World Light** -- the sky's own sun or moon lights the scene. Switch it on
in the World panel and the luminary the sky draws becomes a real Sun lamp
at render time, aimed from where the picture shows it and casting shadows:
the Bryce sun, or the Bryce moon at a quarter of the strength and following
its phase; the Physical sky's sun; a Sky Texture node's sun; the brightest
region of an HDRI or a cube map, found once per image -- or a direction and
colour set by hand when an 8-bit sky cannot be searched. It follows the
World Rotation and Strength, fades as the sun sets, keeps its place under
the light limit, can throw light shafts, and renders the same on both
devices. World types with no sun or moon say so. Off by default: no
existing scene changes.

## Blender-only

Proven headless (CPU frames, the GLSL simulator and the fake device);
nothing below has been seen in a real Blender.

- The World Light sub-panel drawing under Halcyon World: the header
  checkbox, the "no sun or moon" lines on a luminary-less world type, the
  Find / Elevation / Azimuth / Colour rows on HDRI and Cube Map, the last
  report line and the Sun-lamp warning (fakebpy has no layout).
- The eight properties reaching the exporter through the generic World
  loop from a real `bpy.types.World` (the test checks the names and
  defaults against the dataclass, not a real RNA read).
- The viewport: toggling World Light, dragging Sun Elevation / Azimuth or
  the World Rotation and seeing the light follow at interactive rate; a
  lamp drag (the `export_lights_into` fast path) keeping the world light.
- The report line refreshing: it is written at render time and the panel
  redraws on its own schedule, so it can lag one redraw behind.
- The image search on real photographs: a 4k-16k HDRI with a true sun, an
  overcast HDRI (should refuse), an 8-bit skybox with sun and white clouds
  (should refuse and point at Manual), and the time the first search takes
  on a large image (once per image; ~2 million sky samples at most).
- A real Sky Texture node (Nishita / Hosek types export the same sun
  fields; only the Preetham drawing was sampled here).
- The real driver: the world light is an ordinary entry of `hal_lights`,
  so no new shader text is involved, but only a device shows it.
- Animation: a keyed sunset crossing the horizon fading the light, and the
  moon phase keyed across a month.
