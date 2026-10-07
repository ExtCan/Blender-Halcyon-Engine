# R253 camera-box: the render region (Ctrl+B border) and Camera Frame Only

Feature key `camera-box`, branch `r253/camera-box`. The integrator owns
CHANGELOG.md / README.md / the version stamps; the fragments below are in
their house voice.

## CHANGELOG

- **What was asked.** "Box rendering (Ctrl+B functionality in Halcyon)"
  and "when in camera view, render ONLY inside of camera box (option)".
  Blender already owns the user-facing half -- Ctrl+B (`view3d.render_border`)
  writes `space_data.use_render_border` / `render_border_*` in a free view
  and `scene.render.use_border` / `border_min_x..max_y` in camera view,
  Ctrl+Alt+B clears them, Output > Format shows Render Region and Crop to
  Render Region (RENDER_PT_format is marked BLENDER_RENDER, so
  `enable_compatible_panels` already adopts it) -- and Halcyon neither read
  nor honoured any of it: an F12 with Render Region on handed Blender's
  border-sized result a full-frame buffer through `np.resize` (the latent
  scramble is gone with this round). No second operator: Halcyon obeys
  those exact properties.
- **One mechanism for both.** `core/render.region_pixels(st, W, H)` turns
  Blender's fractions (five new `RenderSettings` fields `use_border`,
  `border_min_x/y`, `border_max_x/y`, derived from `scene.render` at F12 by
  `engine._settings_from_scene`, never a Halcyon property --
  `properties._DERIVED_FIELDS`, the `palette_colors` precedent) into an
  output rect with the pipeline's own truncation on the min edge and a ceil
  on the max edge (a Pixel Scale render rect is a superset of the output
  crop); `render()` rasterises the FULL frame on both devices (the G-buffer
  cache stays keyed without the rect: a cached full frame serves a region
  frame, a region frame can never poison a full one), masks the shading to
  the rect plus a context ring (`_region_keep` / `_region_reach`: the band
  scissor's own rules -- the ink reach, one row for a bump chain, 2N for an
  interpolated radiosity grid -- plus one ring for the edge tent and two for
  the adaptive contrast window), marches the volume containers over the
  same mask, takes the rect's rows as a band takes its own for the
  see-through raster and composite, and zeroes RGBA outside the rect at the
  frame's exit (`_apply_region`), returning the FULL-FRAME shape. The post
  chain therefore runs over a full-frame canvas and every stage that
  addresses the pixel grid from the origin -- the dither tiles, the CRT
  mask and scanlines, interlace rows, the N64 / noise hashes, film halftone
  lattices, codec blocks, every GPU twin's `resolution` uniform -- keeps
  the whole frame's anchoring by construction, with zero per-stage changes;
  the crop happens LAST. Bands and the region compose: a pooled band of a
  region frame is the full frame's rows x rect, `core/parallel.render_parallel`
  bands only the rect's rows (the size gates judge the rect's pixel count)
  and places them in a zero canvas, so the pool's pixels stay the
  in-process pixels.
- **The GPU road is pure transport, no new GLSL.** `gpu/shade.shade_frame
  (job, gbuf, region=box)` scissors every material pass and the sky pass to
  the rect's box (the per-rank scissor road `gpu_scissor` already proved on
  the driver) and reads back the box alone, scattered into zeros exactly as
  the sparse ray sweep scatters its hit box; the unshaded-pixel refusal
  looks inside the box only; the transparent layer passes
  (`shade_fragments_frame`) take each depth rank's bbox AND the box from
  `job.region`. The scissored target is released BY NAME right after the
  readback (`frame.edited(st, 'render region', 'the context ring is cut on
  the CPU')`): it holds the rect and its shaded context ring, the ring is
  cut on the CPU side only, and a resident frame would have fed the GPU
  ink, resolve and post chain pixels the CPU road zeroes -- a device-parity
  hole in every non-local stage. The post chain uploads the zero-padded
  frame once and says why. `gpu/shade.simulate(job, gbuf, region=box)` is
  the headless twin -- the uv
  grid covers the box's lanes, every lane scatters into a zero frame -- and
  the self-test's new RENDER REGION section A/Bs the scissored burst
  against the full frame's box on the driver (bar: bitwise) and the whole
  render + post chain under the border, CPU device against GPU device.
- **Two viewport modes, one key.** `preview.region_rect(w, h, frame,
  border)` (pure, pixel-quantised with Blender's truncation so a zooming
  camera view does not re-kick drafts) joins `Viewport.want(camera, w, h,
  rect=)` and the worker's key; the per-frame settings copy carries the
  rect as the five border fields (`shape_settings` forces the scene's own
  F12 border OFF for the viewport: the rect comes from the draw side).
  Border mode (Ctrl+B in a free view: `space_data.render_border_*`; in
  camera view the scene border mapped INTO the camera frame, Cycles'
  `get_buffer_params` rule, `preview.border_in_frame`) keeps the post chain
  over the whole region and parks a region-sized frame with alpha 0
  outside, so the existing ALPHA_PREMULT blit leaves Blender's background
  showing through. Camera Frame Only (`viewport_camera_frame`, Performance
  panel, default off) projects the camera's `view_frame` corners through
  `view3d_utils.location_3d_to_region_2d` (`engine._view_rect`), crops
  BEFORE the post chain so the frame's own corner is the pattern origin
  exactly as an F12 of the frame, parks `Viewport.frame_rect`, and the
  draw side clears the region to the placeholder colour and blits at the
  rect (`_draw_texture(tex, w, h, x, y)`, batches keyed on all four). The
  black-frame guard measures the rect only, so a border change never
  reads as a blackout.
- **F12 delivery.** `engine._border_rect` (both edges truncated, the
  pipeline's own `disprect`) names the region in the header line, the
  finished picture and every extra pass are cropped after `post.process`,
  and `_deliver(..., size=(bw, bh))` calls `begin_result(0, 0, bw, bh)` --
  Blender's own contract for an external engine under `use_border`
  (`render_result_uncrop` places it in the full frame when Crop to Render
  Region is off, so no `crop_to_border` field exists in Halcyon) -- with a
  `len(layer.rect)` guard that prints both sizes and fits to Blender's
  allocation rather than scramble. The Output tab shows a passive
  "Render Region: x0..x1 x y0..y1 (cropped | placed in the full frame)"
  row beside Blender's own checkboxes; the hold cache's fingerprint carries
  the border automatically; `presets.PRESERVED` keeps the six keys (output
  plumbing, never a look).
- **Refuses by name.** A PANO camera and Pano Parts render whole and are
  cropped after the stitch (`[Halcyon] render region: the panorama camera
  renders whole and is cropped after the stitch`, once per frame): a strip
  of the drum has no rect meaning. The Fuzz / Thin Wall blends read the
  finished frame's neighbouring rows layer upon layer, so no finite ring
  holds: the frame renders whole and is cropped, printed once (`the Fuzz /
  Thin Wall blend reads neighbouring pixels of the finished frame`), the
  worker-pool gate's own precedent.
- **Stated tolerance.** Every per-pixel stage is bitwise the full frame's
  rect on both devices. The non-local post stages (glow, star, lens / lamp
  flares, DOF, shafts, the NTSC / tape filters, the codec blocks) see zeros
  past the rect edge and are region-local within their own kernel radius
  of the edge -- exactly as Cycles' compositor sees only the border -- and
  deeper inside agree to within the running-sum box blur's float rounding
  (at most one 1/255 step after the 8-bit quantisation; pinned at 1/255 for
  glow radius 4 beyond 14 px, 1e-4 for the NTSC chroma blur beyond 12 px).
  The GPU reflection sweeps and the environment composite still spawn
  rays for every reflective pixel of the frame (cost, not pixels: the box
  is cut after).
- **Tests.** `tests/test_r253_region.py` (15 headings, 127 checks):
  `region_pixels` rounding / clamps / None cases and the keep mask + box;
  the bitwise pin on the demo at 160x120 for plain, 2x2 supersampled (box
  and triangle filters), ink width 3, a ShaderNodeBump chain, EDGE and
  ADAPTIVE AA, radiosity spacing 2, Thin Wall, Fuzz (whole-then-crop, the
  line counted), A-buffer and Sorted see-through, transparent film, a
  corner rect, the full-frame border identity and the six defaults;
  `last_depth` / `last_passes` / `last_coverage` / `last_cvg` /
  `last_shafts` / `last_flares` full-frame; bands and the pool (640x480,
  bitwise; a too-short rect declines with a reason); the post chain over
  the zero-padded canvas (CRT aperture / slot / shadow, 16-bit Bayer, noise
  and N64 noise dither, interlace, the display transform, Super Black, an
  8-bit palette) bitwise over the rect and the glow / NTSC tolerance both
  held and shown real; the GLSL simulator region twin (plain and bump)
  bitwise inside the box, zero outside; the GPU device's headless fallback
  equal with the verdict naming the driver; PANO and Pano Parts by name;
  SBS stereo (zero and parted eyes, the squeezed rect) and ACCUMULATE 4;
  `region_rect` / `border_in_frame`; the viewport's border mode (draft and
  refine sizes, the parked frame = render()+post of the bordered settings,
  alpha 0 outside, re-kick on a changed rect only, rect None = the pre-R253
  frames and key byte for byte) and camera-frame mode (rect-sized frame,
  `frame_rect`, post over the cropped render, the guard over the rect
  only); the engine's F12 delivery through fakeblender (header line,
  `begin_result` at the border size, the buffer and the Depth pass = the
  full delivery's rect, `_settings_from_scene`, preview thumbnails ignore
  the border, Pixel Scale superset, the `len(rect)` guard); the settings
  plumbing (derived fields, no generated property, the tooltip, PRESERVED,
  `apply_preset`, the hold fingerprint, the panels, copy / as_dict /
  apply). `tests/featurematrix.py` gains two rows (`render region quarter`,
  `render region + ink + bump` on the new `region_bump` scene) so both
  devices prove fallback equality and the five fields have a matrix home;
  `INFRA` names `viewport_camera_frame`. Constraining tests re-run green:
  bands rejoin, the worker pool, the headless viewport, the performance
  shape, the black guard and the device switch, the caches, every setting
  does what it says (235 checks), the tooltips, no setting lies, the
  presets, the version stamps, stereo, the panorama, pano parts, the whole
  feature matrix fallback, the R252 console module (146 checks).

## README

**Render Region (Ctrl+B).** Blender's own render border works in Halcyon
now, for F12 and for the rendered viewport. Press Ctrl+B in camera view to
set the scene's Render Region (Output > Format shows it, with Crop to
Render Region beside it), Ctrl+B in a free view for the viewport's own box,
Ctrl+Alt+B to clear either. The engine rasterises the whole frame on
either device, shades only the box and a thin context ring, and runs the
post chain over the full-frame canvas with zeros outside the box -- so a
dither tile, a CRT mask, an interlace field or a codec block inside the
box is exactly the pixel the full render would have put there, and the
crop happens last. The box is cropped or placed in the full frame as
Blender's checkbox says. The Output tab names the rect in pixels. Pooled
bands, stereo pairs, accumulate AA and Pixel Scale all compose with it.

**Camera Frame Only** (Performance panel, off by default). In camera view
the rendered viewport shades only the pixels inside the camera frame; the
rest of the region keeps the viewport background (Blender's passepartout
darkens it as usual). Fewer pixels per draft and refine, and the picture
you see is the picture F12 frames -- the frame's own corner is the pattern
origin, as it is for an F12. A Ctrl+B box in camera view is honoured on
top of it.

**Known limitations (append).** Under a render border the non-local post
stages -- glow, star, lens / lamp flares, depth of field, light shafts,
the NTSC / tape filters, the codec blocks -- see nothing outside the box,
so within their own kernel radius of the box edge they differ from the
full render (as Cycles' compositor does with a blur over a border);
deeper inside they agree to within the running-sum blur's float rounding,
at most one 8-bit step. A PANO camera or Pano Parts renders whole and is
cropped after the stitch (printed once). The Fuzz / Thin Wall blends
render whole and are cropped (printed once): they read the finished
frame's neighbouring rows layer upon layer. The GPU reflection sweeps
still trace every reflective pixel of the frame under a border (the box is
cut afterwards): a tiny box on a mirror-heavy frame saves the material
passes, not the rays. Post over the full-frame canvas costs the full
frame's post time for any box size.

## Blender-only

- `begin_result(0, 0, bw, bh)` under `scene.render.use_border` with and
  without `use_crop_to_border`: the picture landing in the right place, no
  scramble, the extra passes aligned, and the `len(RenderPass.rect)`
  semantics (floats vs pixels) of the delivery guard.
- The camera-frame rectangle from `view_frame` corners through
  `view3d_utils.location_3d_to_region_2d` matching Blender's drawn
  passepartout at every `view_camera_zoom` / `view_camera_offset` / Lock
  Camera to View state (a 1-px halo at the frame edge is possible until
  tuned), and the clear + offset blit leaving the viewport background
  outside the frame.
- Ctrl+B / Ctrl+Alt+B in a free view (`space_data.use_render_border`) and
  in camera view (`scene.render.use_border` relative to the frame)
  re-kicking the worker and drawing only the box, with Blender's own
  background showing through the alpha-0 outside; Blender's own dashed
  border overlay over the blit; Film > Transparent against the alpha-0
  outside.
- The scissored GPU readback on a real driver: the self-test's RENDER
  REGION section (shade_frame(region) bitwise against the full frame's
  box, the scissor + `read_color` region path on Vulkan and OpenGL
  backends, the named release of the scissored target and the post
  chain's one upload under the border).
- The Output tab showing Blender's Render Region / Crop to Render Region
  checkboxes next to the new status row, and the Performance panel's
  Camera Frame Only toggle.
- Pixel Scale (2X..4X / GBA Mode 5 / 3DO 2X) with a border: the render
  rect is computed at render resolution and the crop at output resolution
  (ceil on the max edge keeps the render a superset; a 1-px drift at the
  right / top edge is possible).
