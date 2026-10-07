"""The Blender-side settings, generated from core.settings.RenderSettings.

Generating rather than hand-writing guarantees the UI and the renderer can never
drift apart: every dataclass field becomes exactly one property, and
`to_settings()` copies them straight back.
"""

import bpy
from bpy.props import (BoolProperty, EnumProperty, FloatProperty,
                       FloatVectorProperty, IntProperty, PointerProperty,
                       StringProperty)
from bpy.types import PropertyGroup

from .core.settings import (RESOLUTION_GROUPS, RenderSettings,
                            resolution_description, resolution_label)
from .core import shading as _shading
from .core.ink import (INK_COLOR_ITEMS, INK_GRADIENT_ITEMS, INK_STYLE_ITEMS,
                       INK_TEXTURE_ITEMS)
from .core.lines import INK_ANCHOR_ITEMS
from .core.film import FILM_GRADE_ITEMS, FILM_PROCESS_ITEMS
from .core.wear import FILM_SCRATCH_SIDE_ITEMS
from .core.sky import PAINTED_LOOK_ITEMS, PAINTED_LOOKS
from .core.gouache import BG_DIRECTION_ITEMS
from .core.shading import MODEL_ITEMS
from .core import console as _console
from .presets.library import preset_items


def _items(*pairs):
    return [(a, b, c) for a, b, c in pairs]


AA_MODE = _items(
    ('NONE', "None", "One sample per pixel -- hard, aliased edges"),
    ('SUPERSAMPLE', "Supersample", "Render larger and filter down"),
    ('EDGE', "Edge Only", "Extra samples only where geometry IDs differ"),
    ('ADAPTIVE', "Adaptive (Edge Pass)",
     "Render once, then re-sample only the edge pixels -- the Bryce / "
     "POV-Ray anti-aliasing pass, at a fraction of Supersample's cost"),
    ('ACCUMULATE', "Accumulation Buffer", "Jittered passes averaged together"),
)
STEREO_MODE = _items(
    ('NONE', "None", "One camera, one picture"),
    ('ANAGLYPH', "Anaglyph (Red/Cyan)",
     "Left eye in red, right eye in green and blue -- the classic "
     "cardboard-glasses stereo of every 90s magazine cover"),
    ('SBS', "Side by Side",
     "Squeezed left|right half-frames in one picture"),
    ('CROSS', "Cross-Eyed",
     "Right|left half-frames, for free-viewing by crossing your eyes"),
)
# R251 C098: depth of field by lens sampling (REYES / SGI accumulation
# buffer / 3ds Max multi-pass) beside the layered post blur
DOF_METHOD = _items(
    ('POST', "Layered Post Blur",
     "The existing depth-of-field: the finished frame blurred in depth "
     "slabs (dof_layers, dof_max_radius)"),
    ('LENS_ACCUMULATE', "Lens Passes (Accumulation Buffer)",
     "Render the frame once per lens point with the eye moved across the "
     "aperture and the window sheared so the focus plane stays put, then "
     "average -- REYES lens sampling by the SGI accumulation buffer's "
     "method (Haeberli & Akeley 1990) and 3ds Max's multi-pass DOF. Exact "
     "occlusion, the lens pattern as the bokeh, N full renders. An F12 "
     "road: the viewport shows no depth of field under Lens Passes"),
)
DOF_LENS_PATTERN = _items(
    ('HALTON_DISC', "Halton disc (SGI 1990)",
     "Lens points from the same Halton sequence the Accumulation Buffer AA "
     "uses, mapped to the unit disc (Shirley-Chiu concentric map); the "
     "paper's own point table was never published, this stands in"),
    ('MAX_SPIRAL', "Max multi-pass spiral (3ds Max 3+)",
     "Pass 0 at the lens centre (Use Original Location), then Halcyon's "
     "golden-angle spiral over 3ds Max's multi-pass dials (Sample Radius "
     "1.0, Sample Bias 0.5); Max documents the dials, not its point "
     "pattern"),
)
# R251 C090: the era's time-slice motion blur beyond the plain average
MOTION_BLUR_MODE = _items(
    ('MEAN', "Average every step",
     "Every time step averaged into the frame -- the accumulation-buffer "
     "trails Halcyon has always drawn"),
    ('MAX_SLICES', "Max Object Motion Blur (random slices)",
     "Each pixel averages only Samples of the Blur Steps slices, the subset "
     "picked per pixel by hash -- 3D Studio / 3ds Max Object Motion Blur's "
     "grainy blur when Samples < Duration Subdivisions"),
    ('LW_FIELD', "LightWave Dithered (row parity)",
     "Twice the steps, even scanlines averaging the even-numbered slices "
     "and odd scanlines the odd -- LightWave 5.6-7's Dithered Motion Blur, "
     "the field-rendering split"),
)
AA_FILTER = _items(
    ('BOX', "Box", "Flat average -- what most 1990s renderers used"),
    ('TRIANGLE', "Triangle", "Linear falloff"),
    ('GAUSS', "Gaussian", "Soft"),
    ('CATROM', "Catmull-Rom", "Sharpening"),
    ('MITCHELL', "Mitchell-Netravali", "Balanced"),
)
# R251 raster pack (RAST-A2, C127): where the supersampler tests inside
# each subpixel
AA_PATTERN = _items(
    ('GRID', "Regular grid", "Every sample at its subpixel centre"),
    ('JITTER', "Jittered (REYES / PRMan)",
     "Each subpixel's sample moves to a hashed position inside its cell, "
     "fresh per pixel and per frame"),
)
SUBPIXEL = _items(
    ('FLOAT', "Floating Point", "Modern sub-pixel accuracy"),
    ('FIXED_4', "Fixed 1/4 pixel", "Quarter-pixel snapping"),
    ('FIXED_1', "Fixed 1 pixel", "Whole-pixel vertices"),
    ('INTEGER', "Integer",
     "Integer raster coordinates -- vertices land on the whole-pixel grid, "
     "the PS1 wobble at its coarsest. Snapping rounds rather than "
     "truncates, so this shares Fixed 1's grid"),
)
# R251 raster pack (RAST-A1): the pixel centre, the near-plane rule and
# the z-buffer encoding family -- each item names its machine
PIXEL_CENTER = _items(
    ('HALF', "Centre +0.5 (OpenGL / Direct3D 10)",
     "The sample sits in the middle of the pixel"),
    ('INTEGER_D3D', "Integer corner (Direct3D 3-9)",
     "Direct3D 3 to 9 sampled at the integer coordinate"),
)
VERTEX_QUANTIZE = _items(
    ('NONE', "Float (software renderers)", "Vertices as exported"),
    ('PS1', "PlayStation (16-bit, 1.3.12 normals, integer-texel UVs)",
     "16-bit positions on the world grid, 1.3.12 normals, UVs on the "
     "1/256 texel page, 8-bit colours"),
    ('N64', "Nintendo 64 (16-bit, 8-bit normals)",
     "16-bit positions on the world grid, signed 8-bit normals, 8-bit "
     "colours; UVs untouched"),
)
NEAR_CLIP = _items(
    ('CLIP', "Clip (software renderers)",
     "Triangles are cut at the near plane"),
    ('REJECT', "Reject whole triangle (PS2 VU1 / PS1)",
     "A triangle with any vertex past the near plane, the far plane or "
     "the 6.4-half-screen guard band is dropped whole"),
)
DEPTH_ENCODING = _items(
    ('LINEAR', "Fixed point (Z-Buffer Bits)",
     "Uniform steps in normalised device z"),
    ('N64_FLOAT18', "N64 18-bit floating (RDP)",
     "64-unit steps near, 1-unit steps far: the RDP's piecewise-floating z"),
    ('GC_14E2', "GameCube 16-bit 14e2 (near ratio)",
     "Reverse-float compressed z: 15 bits near, 17 far"),
    ('GC_13E3', "GameCube 16-bit 13e3 (mid ratio)",
     "Reverse-float compressed z: 14 bits near, 20 far"),
    ('GC_12E4', "GameCube 16-bit 12e4 (far ratio)",
     "Reverse-float compressed z: 13 bits near, the full 24 far"),
    ('W_FIXED', "Fixed-point W (Xbox NV2A, Direct3D W-buffer)",
     "Eye depth w in Z-Buffer Bits, uniform steps in distance"),
    ('VOODOO_W16', "16-bit floating W (3dfx Voodoo / Glide)",
     "1/w as a 4-bit octave and 12 inverted mantissa bits: constant "
     "relative precision per octave"),
)
DEPTH_SORT = _items(
    ('ZBUFFER', "Z-Buffer", "Per-pixel depth test. Always correct"),
    ('PAINTERS', "Painter's Algorithm",
     "Compare whole polygons instead of fragments, as hardware without a depth "
     "buffer had to. Interpenetrating surfaces meet along a polygon edge rather "
     "than their true intersection, and a large polygon can be wrongly occluded "
     "by a small nearer one"),
)
PAINTERS_KEY = _items(
    ('CENTROID', "Centroid", "Sort on the middle of each polygon. The usual choice"),
    ('NEAREST', "Nearest Vertex",
     "Sort on the closest corner. Fewer errors on surfaces that face the "
     "camera, more on long polygons running away from it"),
    ('FARTHEST', "Farthest Vertex", "Sort on the furthest corner"),
    ('ORDERING_TABLE', "Ordering Table (PlayStation)",
     "The PS1's integer depth buckets: mean depth into L entries, far "
     "buckets first, first-added on top; polygons past the far wall or "
     "1023x511 px (a near-clipped polygon measured after the cut) are "
     "dropped whole"),
)
SHADING_RATE = _items(
    ('PIXEL', "Per Pixel (Phong)", "Shade every fragment"),
    ('VERTEX', "Per Vertex (Gouraud)", "Shade at vertices and interpolate colour"),
    ('FACE', "Per Face (Flat)", "One colour per polygon"),
)
FALLOFF = _items(
    ('NONE', "None", "No attenuation"),
    ('INVERSE', "Inverse", "1/d"),
    ('INVERSE_SQUARE', "Inverse Square", "1/d^2, physically correct"),
    ('CUSTOM', "Custom Range", "Linear ramp between start and end"),
    ('BI_LINEAR', "Inverse Linear (Blender Internal)",
     "Blender Internal's D/(D+d): bounded at 1, half strength at the "
     "Falloff End distance. The default curve classic .blend files "
     "were lit against; legacy imports use it"),
    ('BI_SQUARE', "Inverse Square (Blender Internal)",
     "Blender Internal's D/(D+d*d), exactly as 2.79 shipped it (the "
     "source's own comment calls it a hack, and keeps it). For "
     "classic files that chose Inverse Square"),
    ('BI_SLIDERS', "Lin/Quad Sliders (Blender Internal)",
     "Blender Internal's slider falloff: D/(D+lin*d) times "
     "D^2/(D^2+quad*d^2), each factor only while its slider is "
     "above zero -- the 2.4x Quad lamp's att1/att2 pair"),
    # R251 F013 (LIGHT-B1): appended at the END -- _items numbers
    # positionally and saved files hold the numbers
    ('GL_3TERM', "OpenGL 1.1 (1 + lin d + quad d^2)",
     "1/(1 + Linear Slider x d + Quad Slider x d^2): OpenGL's three-term "
     "attenuation, both sliders 0 = no falloff, GL's default"),
    ('POV_FADE_LINEAR', "POV-Ray fade_power 1",
     "2/(1 + d/D): twice the strength at the lamp, exactly 1 at Falloff "
     "End, POV-Ray's linear fade"),
    ('POV_FADE_SQUARE', "POV-Ray fade_power 2",
     "2/(1 + (d/D)^2): POV-Ray's inverse-square fade, 1 at Falloff End; "
     "other fade powers are not modelled"),
    ('GX_GENTLE', "GX Gentle (GameCube)",
     "GX_DA_GENTLE: Ref Brightness at Falloff End, linear term only"),
    ('GX_MEDIUM', "GX Medium (GameCube)",
     "GX_DA_MEDIUM: half linear, half quadratic"),
    ('GX_STEEP', "GX Steep (GameCube)",
     "GX_DA_STEEP: quadratic term only"),
)
# R251 F012 (LIGHT-B1): the spot lamp's cone law
SPOT_LAW = _items(
    ('BLENDER', "Blender Internal",
     "BI's smoothstep blend band times the raw cosine, the Spot Blend "
     "slider's own law"),
    ('GL11', "OpenGL 1.1",
     "A hard cutoff at the cone edge, the cosine to the Cone Exponent "
     "inside"),
    ('POV', "POV-Ray",
     "Flat inside the Hotspot, a Hermite step down to the cone edge, "
     "times the cosine to the Cone Exponent"),
    ('GX_FLAT', "GX Flat (GameCube)",
     "GX_SP_FLAT: full inside the cutoff, a saturating step at the edge"),
    ('GX_COS', "GX Cosine (GameCube)",
     "GX_SP_COS: linear in the cosine from the edge to the axis"),
    ('GX_COS2', "GX Cosine Squared (GameCube)",
     "GX_SP_COS2: the cosine squared from the edge, a softer centre"),
    ('GX_SHARP', "GX Sharp (GameCube)",
     "GX_SP_SHARP: a peaked lobe, brightest on the axis and falling fast"),
    ('GX_RING1', "GX Ring 1 (GameCube)",
     "GX_SP_RING1: a ring around the axis, dark on the axis, peaking "
     "about two thirds of the way out"),
    ('GX_RING2', "GX Ring 2 (GameCube)",
     "GX_SP_RING2: a narrower ring, dark on the axis, brightest at the "
     "cone edge and fading outside it"),
)
SHADOW_MODE = _items(
    ('NONE', "None", "No shadows"),
    ('MAP', "Shadow Maps", "Depth maps rendered from each light"),
    ('RAY', "Ray Traced", "Hard or area-sampled shadow rays"),
    ('PER_LIGHT', "Per Light", "Use each light's own setting"),
    # R251 C052: appended LAST (item order is menu order; positional numbering)
    ('PLANAR', "Planar Polygons (Model 1 / Blinn)",
     "Casters projected onto the floor plane and drawn as flat polygons, "
     "Blinn 1988 / Sega Model 1"),
)
# R251 C117: what a shadow-map texel stores
SHADOW_MAP_DEPTH = _items(
    ('CLASSIC', "Classic (nearest + bias)",
     "The nearest caster depth with the bias and normal offset, as every "
     "map until now"),
    ('MIDPOINT', "Midpoint (Maya Use Mid Dist / Blender Classic-Halfway)",
     "Woo's midpoint between the two nearest casters; bias and offset "
     "become zero"),
)
LIGHT_MAP_DEPTH = _items(
    ('INHERIT', "Use Render Setting",
     "This lamp's map stores what the render's Map Depth says"),
    ('CLASSIC', "Classic (nearest + bias)",
     "The nearest caster depth with the bias and normal offset, as every "
     "map until now"),
    ('MIDPOINT', "Midpoint (Maya Use Mid Dist / Blender Classic-Halfway)",
     "Woo's midpoint between the two nearest casters; bias and offset "
     "become zero"),
)
# R251 C020/C036: a material's triangles as an authored shadow volume
VOLUME_ROLE = _items(
    ('NONE', "Surface", "Drawn as geometry, as every material until now"),
    ('DC_INCLUDE', "Dreamcast Modifier (inside)",
     "A closed modifier volume: pixels INSIDE it by crossing parity take "
     "the Dreamcast shadow scale"),
    ('DC_EXCLUDE', "Dreamcast Modifier (outside)",
     "A closed modifier volume: EVERYTHING in the frame outside it takes "
     "the Dreamcast shadow scale -- the whole picture darkens except the "
     "inside, not only a rim around it (PowerVR EXCLUDE_LAST_POLY)"),
    ('DS_SHADOW', "DS Shadow Polygon",
     "A closed volume drawn as a Nintendo DS mode-3 shadow polygon: "
     "depth-fail mask, then a 5-bit blend of this material's colour"),
)
TEX_FILTER = _items(
    ('NEAREST', "Nearest", "Point sampling -- chunky texels"),
    ('BILINEAR', "Bilinear", "Smooth"),
    ('TRILINEAR', "Trilinear", "Bilinear plus mip blending"),
    ('N64_3POINT', "3-Point (N64)", "The Reality Coprocessor's triangular filter"),
    # R251 (TEX-1 C111; TEX-2 appends SUMMED_AREA after it, at the END)
    ('POV_NORMDIST', "Normalised Distance (POV-Ray 4)",
     "POV-Ray image_map interpolate 4: inverse-square weights, cusps at texel centres"),
    # R251 (TEX-2 C088), at the END
    ('SUMMED_AREA', "Summed Area (3D Studio / Max)",
     "Crow's summed-area table: an exact box average over the footprint's rectangle, no pyramid"),
)
WRAP = _items(('REPEAT', "Repeat", ""), ('EXTEND', "Extend", ""),
              ('CLIP', "Clip", ""), ('MIRROR', "Mirror", ""))
# R251 texture pack (TEX-1): the once-per-upload texel formats (C074)
TEX_FORMAT = _items(
    ('NONE', "Off", "The image's own precision"),
    ('RGB565', "RGB 5:6:5", "3dfx Glide GR_TEXFMT_RGB_565 / Direct3D 5-7"),
    ('ARGB1555', "ARGB 1:5:5:5", "Glide / D3D: 1-bit alpha cut-outs"),
    ('ARGB4444', "ARGB 4:4:4:4", "Glide / D3D / PowerVR PCX fallback, 16 levels per channel"),
    ('RGB332', "RGB 3:3:2", "Glide GR_TEXFMT_RGB_332 / PCX 8-bit"),
    ('RGB5550', "RGB 5:5:5", "PowerVR PCX1/PCX2, green at five bits"),
    ('I8', "Intensity 8", "Glide INTENSITY_8: grey that also feeds alpha"),
    ('A8', "Alpha 8", "Glide ALPHA_8: white with the image's alpha"),
    ('AI44', "Alpha 4 + Intensity 4", "Glide ALPHA_INTENSITY_44 fonts and lightmaps"),
    ('AI88', "Alpha 8 + Intensity 8", "Glide ALPHA_INTENSITY_88"),
    ('I4', "Luma 4-bit", "Sega Model 2 / Model 3 luminance texture RAM"),
    ('I4_MODEL2', "Luma 4-bit, 0xF transparent", "Sega Model 2: level 15 is the hole marker"),
    ('A3I5', "Alpha 3 + Intensity 5 (DS)", "Nintendo DS translucent texture: 5-bit grey, 8 alpha levels"),
    ('A5I3', "Alpha 5 + Intensity 3 (DS)", "Nintendo DS translucent texture: 3-bit grey, 32 alpha levels"),
    ('YIQ422', "NCC YIQ 4:2:2", "3dfx TexUS narrow-channel compression, 8 bits per texel"),
    ('AYIQ8422', "NCC AYIQ 8:4:2:2", "3dfx NCC with an 8-bit alpha byte"),
)
# the Nintendo 64's 4 KB TMEM budget by format (C013)
TEX_TMEM = _items(
    ('OFF', "Off", "No texture-memory budget"),
    ('RGBA16', "RGBA16 (N64 TMEM)", "5551 texels, 2048 of them fit"),
    ('RGBA32', "RGBA32 (N64 TMEM)", "8888 split across both halves, 1024 texels"),
    ('CI8', "CI8 (N64 TMEM)", "256-colour index in the lower 2 KB, 15-bit entries"),
    ('CI4', "CI4 (N64 TMEM)", "16-colour index, 4096 texels (64x64)"),
    ('IA16', "IA16 (N64 TMEM)", "8-bit intensity + 8-bit alpha"),
    ('IA8', "IA8 (N64 TMEM)", "4-bit intensity + 4-bit alpha"),
    ('IA4', "IA4 (N64 TMEM)", "3-bit intensity + 1-bit alpha, 8192 texels"),
    ('I8', "I8 (N64 TMEM)", "8-bit intensity, alpha = intensity"),
    ('I4', "I4 (N64 TMEM)", "4-bit intensity, alpha = intensity, 8192 texels"),
)
# baked block compression (C024)
TEX_COMPRESS = _items(
    ('NONE', "Off", "Uncompressed texels"),
    ('DXT1', "DXT1 (S3TC, PC 1998+)", "4x4 blocks, two 5:6:5 endpoints, reference 8-bit decode"),
    ('DXT1_NV2A', "DXT1, 16-bit decode (Xbox)", "NV2A decodes the interpolants at 5:6:5, banding gradients"),
    ('CMPR_GC', "CMPR (GameCube / Wii)", "DXT1 blocks with the TEV's 3/8-5/8 interpolants"),
    ('VQ_DC', "VQ codebook (Dreamcast / Naomi)", "256 entries of 2x2 texels, 2 bits per texel"),
)
# the Voodoo's coarse bilinear fraction (C080)
TEX_FRAC_BITS = _items(
    ('FLOAT', "Float", "Full-precision blend weights"),
    ('BITS_4', "4-bit (Voodoo1, Verite V1000)", "Sixteen blend steps between texels"),
    ('BITS_8', "8-bit (Voodoo2)", "256 blend steps between texels"),
)
# R251 texture pack (TEX-2, wave 2): how Extend nodes clamp (C077)
TEX_CLAMP_MODE = _items(
    ('EDGE', "Clamp to Edge (OpenGL 1.2+, consoles)", "Extend repeats the last texel"),
    ('GL_CLAMP', "GL_CLAMP border (OpenGL 1.0/1.1, 3dfx MiniGL)",
     "Linear taps past the last texel read the (0,0,0,0) border: the half-texel seam"),
)
# how a mip level is picked (C072)
TEX_MIP_SELECT = _items(
    ('FILTER', "As the filter", "Trilinear blends two levels; the other filters read level 0"),
    ('BLEND', "Blend two levels (N64 RDP)",
     "3-Point, Bilinear or Nearest taps at each level, lerped by the LOD fraction; Bilinear + Blend is Trilinear"),
    ('NEAREST_LEVEL', "One level, nearest (OpenGL 1.1, Direct3D point mip)",
     "level = ceil(lod + 1/2) - 1, no blend: the popping of 1996 cards"),
    ('DITHER_VOODOO', "One level, 4x4 dithered (3dfx Voodoo1)",
     "level = floor(lod + bayer/16) on 8.8 fixed LOD: stippled mip seams"),
)
# where the mip LOD comes from (C022 / C079)
TEX_LOD_SOURCE = _items(
    ('DERIVATIVE', "Screen derivatives (PC cards, N64)", "Per-pixel texture footprint, as today"),
    ('GS_Q', "log2 of Q, 7.4 fixed (PlayStation 2 GS)",
     "LOD from depth alone: (log2(1/|Q|) << L) + K with four fractional bits"),
    ('TRIANGLE', "One level per polygon (Riva 128, Verite V1000)",
     "The triangle's texel-area / screen-area ratio picks its ONE nearest level, no blend: mip seams at polygon edges"),
)
TRANSPARENCY = _items(
    ('NONE', "Opaque", "Ignore alpha"),
    ('STIPPLE', "Screen Door", "Dithered stipple, as hardware without blending did"),
    ('SORTED', "Sorted Blend", "Depth-sorted alpha blending"),
    ('ABUFFER', "A-Buffer", "Per-pixel fragment lists, correct through any depth"),
)
FOG_MODE = _items(
    ('LINEAR', "Linear", ""), ('EXP', "Exponential", ""),
    ('EXP2', "Exponential Squared", ""),
    ('TABLE16', "16-Step Table", "Banded fixed-function fog table"),
    # R251 F001 (PlayStation GTE); F009's GROUND item lands in wave 2
    ('GTE_1Z', "Hyperbolic 1/z (PlayStation GTE)",
     "The GTE's depth cue: a factor affine in 1/z that is 0 at Fog "
     "Start and 1 at Fog End, floored to 4.12 fixed point (IR0) before "
     "the far colour blends in -- the PlayStation's and Namco System "
     "11/12's fog. At the Gouraud (vertex) rate this is the GTE's own "
     "per-vertex cue interpolated across the polygon; per pixel the "
     "same curve is evaluated everywhere, which no console did. "
     "Per-Vertex Fog's 1/8 rounding is skipped: the mode has its own "
     "12-bit floor"),
    # R251 F009 (LIGHT-A2): POV-Ray's fog_type 2
    ('GROUND', "Ground Fog, atan integral (POV-Ray)",
     "POV-Ray's fog_type 2: density 1/(1+Y^2) above Fog Ground Offset, "
     "integrated along the eye ray, exp(-distance x mean density x Fog "
     "Density); the sky is fogged by elevation, as POV fogs a ray that "
     "hits nothing. The hardware fog tables and Height Fog are inert "
     "here: the integral is its own height law"),
)
# R251 F008 (LIGHT-A2): what the fog fades toward
FOG_SOURCE = _items(
    ('FIXED', "Fog Colour", "The Fog Colour swatch"),
    ('BACKDROP', "Backdrop (LightWave)",
     "The sky or backdrop behind each pixel"),
)
# R251 F002 / F003 / F022: the accelerators' own fog tables
FOG_TABLE = _items(
    ('NONE', "None", "The fog curve as computed, no hardware table"),
    ('VOODOO64', "64 Entries on 1/w (3dfx Voodoo)",
     "Glide's 64-entry table at 2^(3+i/4)/(8-i%4) scene units, "
     "interpolated by the w mantissa's next 8 bits"),
    ('PVR128', "128 Log Entries (PowerVR2 / Dreamcast)",
     "The CLX2's 128 entries over 8 octaves of 1/w below Fog End, two "
     "8-bit densities blended per entry"),
    ('DS32', "32 Entries (Nintendo DS)",
     "The DS's 32-entry table of 7-bit densities at equal steps from "
     "Fog Start to Fog End, blended in 1/128 units"),
)
# R251 F007: where the fog reads its distance
FOG_DEPTH = _items(
    ('W', "Eye Depth (w)", "Eye-space distance, as W-fog and every console did"),
    ('Z', "Z-Buffer Depth (Direct3D z-fog)",
     "Post-projection z in 0..1 -- hyperbolic, so the fog piles into the "
     "far stretch"),
)
COLOR_DEPTH = _items(
    ('32', "32-bit (8:8:8:8)", "True colour with alpha"),
    ('24', "24-bit (8:8:8)", "True colour"),
    ('16', "16-bit (5:6:5)", "High colour"),
    ('15', "15-bit (5:5:5)", "High colour, PlayStation and Voodoo"),
    ('12', "12-bit (4:4:4)", "Amiga AGA register depth"),
    ('8', "8-bit indexed", "256 colours from a palette"),
    ('4', "4-bit indexed", "16 colours"),
    ('1', "1-bit", "Monochrome"),
    ('HAM8', "Amiga HAM8", "Hold-and-modify, 6 bits per channel"),
    ('HAM6', "Amiga HAM6", "Hold-and-modify, 4 bits per channel"),
    # R251 (post-palette): appended at the END -- _items numbers positionally
    ('CRY16', "16-bit CRY (Atari Jaguar)",
     "8-bit intensity and 256 chroma cells: smooth luminance, stepped hue "
     "and saturation"),
    ('YJK', "YJK 19,268 colours (MSX2+ Screen 12)",
     "5-bit luma per pixel, chroma shared by aligned groups of four output "
     "pixels (before Pixel Scale)"),
)
PALETTE_MODE = _items(
    ('ADAPTIVE', "Adaptive", "Built from this image's own colours"),
    ('VGA256', "VGA Default 256", "The IBM VGA BIOS palette"),
    ('MAC256', "Macintosh System", "The System 7 256-colour table"),
    ('WEB216', "Web Safe 216", "The browser-safe cube"),
    ('FIXED_666', "6:6:6 Cube", "216-entry RGB cube"),
    ('WIN20', "Windows 20", "The reserved system colours"),
    ('EGA16', "EGA 16", "IBM Enhanced Graphics Adapter"),
    ('CGA4', "CGA 4", "Cyan/magenta/white"),
    ('GRAY', "Greyscale", ""),
    ('CUSTOM', "Custom", "From a palette image"),
    # R251 (post-palette): appended at the END -- _items numbers positionally
    ('EHB', "Extra Half-Brite (Amiga OCS/ECS)",
     "32 adaptive 12-bit registers plus their exact halves: 64 colours; Lock "
     "Palette shares its cache with a 32-colour adaptive palette"),
)
PALETTE_METHOD = _items(
    ('MEDIAN_CUT', "Median Cut", "Heckbert's algorithm, the period standard"),
    ('OCTREE', "Octree", "Gervautz-Purgathofer"),
    ('POPULARITY', "Popularity", "Most frequent colours"),
    ('KMEANS', "K-Means", "Slowest, best quality"),
)
# R251 (post-palette C061): the palette RAM / DAC precision the registers
# are snapped to before the per-pixel search; the level count is the item
PALETTE_BITS = _items(
    ('NONE', "Full (AGA / Mac / VGA at 8 bits)",
     "Registers keep their fitted 8-bit values"),
    ('BITS_1', "1-bit -- PC-88 digital RGB (8)",
     "Each channel on or off (the ZX Spectrum's fifteen are an attribute "
     "set: Attribute Cells)"),
    ('BITS_2', "2-bit -- EGA (64)", "Four levels per channel"),
    ('CPC_27', "3 levels -- Amstrad CPC (27)",
     "Three voltages per channel, not a bit depth"),
    ('BITS_3', "3-bit -- Atari ST / MSX2 (512)",
     "Eight levels per channel (the Mega Drive's 9-bit CRAM lattice too, "
     "but its DAC ramp is non-linear and not modelled here)"),
    ('BITS_4', "4-bit -- Amiga OCS-ECS / STE / PC-98 (4096)",
     "Sixteen levels per channel"),
    ('BITS_5', "5-bit -- SNES CGRAM / Sega 32X / Neo Geo (32768)",
     "Thirty-two levels per channel (the Neo Geo's one shared dark bit is "
     "not modelled; the PS1 has no frame palette registers -- its 15-bit "
     "framebuffer is Colour Depth 15)"),
    ('BITS_6', "6-bit -- VGA DAC (262144)", "Sixty-four levels per channel"),
)
# R251 (post-palette C050): colour stored per character cell; the cell
# size, the colour set and the shared-bright rule are the item's
ATTRIBUTE_CELLS = _items(
    ('NONE', "Off", "Colour per pixel"),
    ('ZX_SPECTRUM', "ZX Spectrum (8x8: ink + paper, shared BRIGHT)",
     "Two of fifteen per cell, one brightness bit for both"),
    ('MSX1', "MSX1 Screen 2 (8x1: two of fifteen)",
     "The TMS9918's two colours per eight-pixel run"),
    ('C64_HIRES', "C64 hires bitmap (8x8: two of sixteen)",
     "Two VIC-II colours per cell"),
    ('C64_MULTI', "C64 multicolour (4x8: three + shared background)",
     "Three cell colours plus the most common colour of the frame"),
)
# R251 (post-palette C060): the registers rewritten every scanline; the
# cost is in the item, so it is read before the render
SCANLINE_PALETTE = _items(
    ('NONE', "Off", "One palette for the frame"),
    ('SPECTRUM_512', "Spectrum 512 (Atari ST): 3 x 16 per line from 512",
     "The 68000 reloads the ST's registers three times a line (slow: a "
     "median cut per third of a line, on the CPU)"),
    ('DYNAMIC_HIRES', "Dynamic HiRes (Amiga): 16 per line from 4096",
     "Sixteen Copper MOVEs in every horizontal blank (slow: a median cut "
     "per line, on the CPU)"),
    ('SHAM', "Sliced HAM (Amiga): HAM6 with 16 base colours per line",
     "Hold-and-modify with the fringes reset every line (slowest: a "
     "median cut and a HAM encode per line, on the CPU)"),
)
# R251 (post-palette C093): 3D Studio's / 3ds Max's Video Color Check
VIDEO_COLOR_CHECK = _items(
    ('NONE', "Off", "No legality test"),
    ('FLAG_BLACK', "Flag with Black (3D Studio / Max)",
     "Illegal pixels turn black for inspection"),
    ('SCALE_LUMA', "Scale Luma (Max)",
     "Darken the pixel into range, hue and saturation kept"),
    ('SCALE_SAT', "Scale Saturation (Max)",
     "Desaturate toward its own luma, brightness kept"),
)
VIDEO_SYSTEM = _items(
    ('NTSC', "NTSC-M (7.5 IRE setup, gain 92.5)",
     "Black at 7.5 IRE, the American envelope"),
    ('PAL', "PAL (0 IRE setup, gain 100)",
     "Black at 0 IRE, the European envelope"),
)
VIDEO_IRE_LIMIT = _items(
    ('IRE_120', "120 IRE (the NTSC envelope, Max's test)",
     "The composite maximum"),
    ('IRE_110', "110 IRE (broadcast house rule, After Effects)",
     "The conservative limit stations asked for"),
)
DITHER = _items(
    ('NONE', "None", ""),
    ('BAYER2', "Bayer 2x2", ""), ('BAYER4', "Bayer 4x4", ""),
    ('BAYER8', "Bayer 8x8", ""), ('BAYER16', "Bayer 16x16", ""),
    ('HALFTONE', "Halftone", "Clustered dot"),
    ('FLOYD', "Floyd-Steinberg", ""), ('JJN', "Jarvis-Judice-Ninke", ""),
    ('STUCKI', "Stucki", ""), ('ATKINSON', "Atkinson", "The Macintosh kernel"),
    ('BURKES', "Burkes", ""), ('SIERRA', "Sierra", ""),
    ('SIERRA_LITE', "Sierra Lite", ""), ('NOISE', "Blue Noise", ""),
)
# R251 (transparency pack): the blend unit's fixed equations. FIVE-tuples
# with EXPLICIT numbers: `_items` builds three-tuples that Blender numbers
# by POSITION and stores as integers in the .blend, so a separator or a
# machine group inserted later would renumber every saved scene. The
# numbers are core/render.py's MODE_INDEX (the composite's one table of
# ids); separators take 900+ (never stored); INHERIT 100 / ENV_HOLE 101
# exist only in the Material list. Grouped by machine with the empty-
# identifier separator idiom `res_preset` already uses.
BLEND_EQUATION = [
    ('ALPHA', "Alpha Over (Halcyon)",
     "Halcyon's float blend: F*a + B*(1-a) per layer, in depth order", '', 0),
    ('', "PlayStation", '', '', 900),
    ('PS1_AVG', "(B+F)/2 (PlayStation average)",
     "The PS1 GPU's 0.5B+0.5F on 5-bit channels: a fixed 50% glass whatever "
     "the opacity; MascotCapsule PATTR_BLEND_HALF", '', 1),
    ('PS1_ADD', "B+F (PlayStation add)",
     "The PS1 GPU's 1.0B+1.0F on 5-bit channels, saturating at 31: additive "
     "glows white out; MascotCapsule PATTR_BLEND_ADD", '', 2),
    ('PS1_SUB', "B-F (PlayStation subtract)",
     "The PS1 GPU's 1.0B-1.0F on 5-bit channels, clamped at 0: subtractive "
     "shadows and smoke; MascotCapsule PATTR_BLEND_SUB", '', 3),
    ('PS1_QUARTER', "B+F/4 (PlayStation quarter-add)",
     "The PS1 GPU's 1.0B+0.25F on 5-bit channels, saturating at 31: the "
     "faint additive for particles and lens glints", '', 4),
    ('', "Sega Saturn", '', '', 901),
    ('SATURN_HALF', "B/2+F/2 (Saturn VDP1 half-transparency)",
     "VDP1's colour-calculation half-transparency: each 5-bit operand halved "
     "before the add; also the 3DO PIXC's default P/2+S/2", '', 5),
    ('SATURN_SHADOW', "B/2 (Saturn VDP1 shadow)",
     "VDP1's shadow calculation: the frame pixel halved, the source colour "
     "ignored (a drop shadow sprite)", '', 6),
    ('SATURN_HALF_LUM', "F/2 (Saturn VDP1 half-luminance)",
     "VDP1's half-luminance replace: the source at half brightness, the "
     "frame pixel ignored -- a per-sprite bit, so a Material Blend Mode "
     "only", '', 7),
    ('', "3DO", '', '', 902),
    ('THREEDO_SUB', "F-B (3DO PIXC subtract)",
     "The PIXC's cel minus the frame pixel on 5-bit channels, clamped at 0 "
     "(S = the frame pixel; the CCB constant source is not offered)", '', 8),
    ('THREEDO_XOR', "F xor B (3DO PIXC)",
     "The PIXC's xor of the cel against the frame pixel (S = the frame "
     "pixel; the CCB constant source is not offered)", '', 9),
    ('', "SNES", '', '', 903),
    ('SNES_ADD', "B+F clamp 31 (SNES colour math add)",
     "The PPU's colour-math add of the main screen and ONE sub screen, "
     "clamped at 31 -- the PS1's arithmetic with bit-replicated read-back "
     "and one sub screen", '', 10),
    ('SNES_SUB', "B-F clamp 0 (SNES colour math subtract)",
     "The PPU's colour-math subtract of one sub screen, clamped at 0 -- the "
     "PS1's arithmetic with bit-replicated read-back and one sub screen",
     '', 11),
    ('SNES_ADD_HALF', "(B+F)/2 (SNES colour math add-half)",
     "The PPU's colour-math add with the halve bit: the average of the main "
     "screen and one sub screen -- the PS1's arithmetic with bit-replicated "
     "read-back and one sub screen", '', 12),
    ('SNES_SUB_HALF', "max(0,B-F)/2 (SNES colour math subtract-half)",
     "The PPU's colour-math subtract with the halve bit, clamped at 0 then "
     "halved -- the PS1's arithmetic with bit-replicated read-back and one "
     "sub screen", '', 13),
    ('', "Game Boy Advance", '', '', 904),
    ('GBA', "(F*EVA+B*EVB)/16 (GBA BLDALPHA)",
     "The LCD's BLDALPHA blend of the top two layers: EVA = the alpha in "
     "sixteenths, EVB = 16-EVA, clamped at 31 on 5-bit channels, bit-"
     "replicated read-back", '', 14),
    ('', "Nintendo DS", '', '', 905),
    ('DS', "(F*(A+1)+B*(31-A))/32, alpha max, same-ID once (Nintendo DS)",
     "The DS 3D engine's blend on 6-bit channels with a 5-bit polygon alpha: "
     "destination alpha = max, and a pixel already holding the same "
     "translucent polygon ID (Halcyon's object index) is never blended "
     "twice", '', 15),
    ('', "Software", '', '', 906),
    ('FUZZ', "Fuzz (Doom Spectre)",
     "The background one row up or down, at 26/32 brightness: id's "
     "r_draw.c fuzz columns", '', 16),
    ('THIN_WALL', "Thin Wall (3ds Max refraction)",
     "The background jogged along the projected normal by Thickness Offset "
     "x (IOR - 1), the 3ds Max Thin Wall Refraction map; no rays", '', 17),
    ('IMAGINE_FOG', "Imagine Fog (Impulse Imagine)",
     "Opacity = the object's thickness along the view / Fog Length, capped "
     "at 1, unlit -- Imagine's fog object; needs A-Buffer", '', 18),
]
#: the global menu: every equation but the VDP1's per-sprite half-luminance
#: replace (a frame-wide F/2 makes every see-through surface an opaque
#: half-bright replace, which no Saturn frame was)
BLEND_EQUATION_GLOBAL = [i for i in BLEND_EQUATION if i[0] != 'SATURN_HALF_LUM']
BLEND_MODE_MAT = [('INHERIT', "Inherit",
                   "The render setting's Blend Equation", '', 100)] \
    + BLEND_EQUATION \
    + [('ENV_HOLE', "Env Hole (Blender 2.4x)",
        "The world along the view ray, alpha 0, nothing behind it", '',
        101)]
# R251: Screen Door's own pattern list -- the ORDERED kinds only (an
# error-diffusion kind has no threshold map, and R:2717 used to index the
# None), as five-tuples carrying the numbers DITHER gave the five ordered
# kinds (BAYER2 1 .. HALFTONE 5) so a scene saved with Screen Door at
# Bayer 4x4 (stored 2) reloads as Bayer 4x4; COLUMNS 14 and N64_NOISE 15
# sit past DITHER's fourteen so a later `dither` item never collides. A
# stored number no item carries (the old NONE = 0, an error-diffusion kind
# 6..13) reads back from Blender as '' and the renderer falls back to
# Bayer 4x4 by name.
STIPPLE_PATTERN = [
    ('BAYER2', "Bayer 2x2", "", '', 1), ('BAYER4', "Bayer 4x4", "", '', 2),
    ('BAYER8', "Bayer 8x8", "", '', 3), ('BAYER16', "Bayer 16x16", "", '', 4),
    ('HALFTONE', "Halftone", "Clustered dot", '', 5),
    ('COLUMNS', "Columns 1x2 (Mega Drive / SNES 512)",
     "Alternate pixel columns: the stripe mesh a composite TV blended", '',
     14),
    ('N64_NOISE', "Random compare (N64 RDP)",
     "A fresh 8-bit hash per pixel and frame against the alpha, "
     "dither_alpha_en", '', 15),
]
TRANSLUCENT_ORDER = _items(
    ('DEPTH', "Depth (Halcyon)", "Per-fragment or per-polygon depth"),
    ('Y_SORT', "Bottom Row (DS auto-sort)",
     "Bottom screen row, then top row, then submission -- the DS's order, "
     "never depth"),
    ('SUBMISSION', "Submission (DS manual / GL)", "Draw order as exported"))
FRAMEBUFFER = _items(
    ('NONE', "24-bit float (Halcyon)", "No in-pipeline truncation"),
    ('PS2_CT16', "PSMCT16 (PlayStation 2)",
     "5-5-5 per write with the GS's DIMX dither, read back per blend"),
    ('GC_RGBA6', "RGBA6 (GameCube EFB)",
     "6 bits per channel, Flipper's 2x2 dither at every write"),
    ('VOODOO_565_4X4', "RGB565 4x4 (3dfx Voodoo)",
     "MAME's hardware-verified pack, dithered read-back"),
    ('VOODOO_565_2X2', "RGB565 '2x2' (3dfx Voodoo)",
     "grDitherMode 2x2: the real {8,10;11,9} matrix"))
COLOR_MGMT = _items(
    ('NONE', "None (Period Correct)", "No transform -- 1990s renderers had none"),
    ('SRGB', "sRGB", "Modern display transform"),
    ('FILMIC', "Filmic", "Highlight rolloff"),
    ('REINHARD', "Reinhard", "Simple tone map"),
)
CRT_MASK = _items(
    ('NONE', "None", ""),
    ('APERTURE', "Aperture Grille", "Trinitron vertical stripes"),
    ('SLOT', "Slot Mask", "Staggered slots"),
    ('SHADOW', "Shadow Mask", "Triad dots"),
)
INTERLACE = _items(
    ('NONE', "None", ""),
    ('FIELDS', "Fields", "Alternate scan lines per frame"),
    ('BLEND', "Blended", "Darken alternate lines"),
)
OUTPUT_SCALE = _items(('NONE', "1x", ""), ('2X', "2x", ""), ('3X', "3x", ""),
                      ('4X', "4x", ""),
                      # R251 post-signal: the two machine resamplers (appended at
                      # the END: the enum is numbered positionally)
                      ('THREEDO_2X', "2x interpolated (3DO, fixed cornerweight)",
                       "The Opera display generator's 320x240 to 640x480 "
                       "interpolation with every sample at the same subpixel "
                       "(plain 2x bilinear on the cornerweight lattice) and blue "
                       "at 4 bits; the projector's per-cel subpixel bits wait on "
                       "the coverage road"),
                      ('GBA_MODE5', "Mode 5 affine stretch (GBA)",
                       "The 160x128 bitmap stretched to the 240x160 LCD through "
                       "BG2's 8.8 registers PA = 0xAB, PD = 0xCD: uneven columns, "
                       "no filtering"))
# R251 post-signal: the machine's scan-out stages (core/signal_era.py)
VI_GAMMA = _items(
    ('NONE', "Off", "No VI gamma: the framebuffer reaches the DAC as it is"),
    ('GAMMA', "Gamma (N64 GAMMA_ENABLE)",
     "2 * isqrt(64 * v): the Video Interface's integer square-root curve"),
    ('GAMMA_DITHER', "Gamma + dither (N64 GAMMA_DITHER_ENABLE)",
     "Six hashed bits folded under the root: the N64's fine gradient noise"),
    ('DITHER_ONLY', "Dither only (N64, no gamma)",
     "One hashed bit added to each channel's LSB"),
)
COPY_FILTER = _items(
    ('NONE', "Off",
     "The EFB copies unfiltered (the progressive set {0,0,21,22,21,0,0}, "
     "an identity)"),
    ('DEFLICKER', "Deflicker (GameCube / Wii default)",
     "libogc's {8,8,10,12,10,8,8}: lines above and below at 16/64, the "
     "line at 32/64"),
    ('DEFLICKER_AA', "Deflicker, AA modes (GameCube / Wii)",
     "The *Aa render modes' {4,8,12,16,12,8,4}: 12/64, 40/64, 12/64"),
)
CRTC_BLEND = _items(
    ('NONE', "Off", "One read circuit: the framebuffer as it is"),
    ('PREVIOUS_FRAME', "Previous frame (PS2 RC2)",
     "RC2 scans the previous frame's CRTC output: a recursive trail"),
    ('BG_COLOR', "Background colour (PS2 SLBG)",
     "RC1 mixed with the BGCOLOR register: the CRTC fade"),
)
VIDEO_FILTER = _items(
    ('NONE', "Off", "The 16-bit framebuffer reaches the DAC as it is"),
    ('VOODOO1', "4x1 scan-out filter (Voodoo Graphics)",
     "86Box's reconstruction of the '22-bit' filter: four 2-tap passes "
     "along the line, each halving a capped difference"),
    ('VOODOO2', "Look-ahead scan-out filter (Voodoo2)",
     "86Box's Voodoo2 rule: each pixel lifted toward its brighter "
     "neighbours three back and one ahead by three fifths of the "
     "difference, at most 32 levels, never past the threshold"),
)
# R251 SIG-2: the digital formats' chroma, the cable, the PAL receiver, the tape
CHROMA_FORMAT = _items(
    ('NONE', "Off", "RGB straight through: no chroma sampling"),
    ('Y422', "4:2:2 co-sited (D1 / Digital Betacam, 1986)",
     "One Cb/Cr per two luma samples, sited on the even one"),
    ('Y411', "4:1:1 co-sited (NTSC DV, 1995)",
     "One Cb/Cr per four luma samples: colour edges in four-pixel steps"),
    ('Y420_MPEG1', "4:2:0 centred (MPEG-1 / Video CD / PS1 MDEC)",
     "One Cb/Cr per 2x2 block, sited at its centre"),
    ('Y420_MPEG2', "4:2:0 MPEG-2 (DVD)",
     "One per 2x2 block, co-sited horizontally, centred vertically"),
    ('Y420_DVPAL', "4:2:0 alternate lines (PAL DV)",
     "Cb on even lines, Cr on odd lines, each two pixels wide"),
    ('XFB_422', "XFB 4:2:2 (GameCube / Wii, 2001)",
     "The pair averaged in RGB, then the BT.601 integers Dolphin measured "
     "on the hardware and the legal clamps; runs after the copy filter, "
     "in the EFB-to-XFB copy's order"),
)
CHROMA_UPSAMPLE = _items(
    ('HOLD', "Sample and hold",
     "Each pixel takes its site's chroma: the blocky DV / VCD colour"),
    ('LINEAR', "Linear",
     "Chroma interpolated between the two nearest sites along the "
     "subsampled axis"),
)
SIGNAL = _items(
    ('RGB', "RGB / SCART / component",
     "No cable artefact: the frame as the machine made it"),
    ('SVIDEO', "S-Video, NTSC (1987)",
     "Luma whole; the encoder's I at 1.3 MHz and Q at 0.5 MHz as a "
     "wideband decoder passes them: chroma blur, no dot crawl (Composite "
     "Video is greyed: the cable is one or the other)"),
    ('SVIDEO_PAL', "S-Video, PAL",
     "Luma whole; U and V at 1.3 MHz over a 52 us line"),
    ('RF', "RF modulator (channel 3/4)",
     "The composite chain through a modulator: luma low-pass, the 920 kHz "
     "beat, snow and a ghost (turn Composite Video on for the full chain)"),
)
PAL_DECODER = _items(
    ('NONE', "Off", "No PAL decoder stage"),
    ('DELAY_LINE', "PAL-D delay line (1967)",
     "Each line's chroma averaged with the line above in the same field: "
     "vertical colour resolution halved, no hue error"),
    ('SIMPLE', "Simple PAL, no delay line",
     "The cheap decoder: a phase error shows as Hanover bars, alternating "
     "hue two lines tall"),
)
TAPE = _items(
    ('NONE', "Off", "No tape: the frame goes straight down the cable"),
    ('VHS', "VHS (1976)",
     "240 TVL luma, 40 TVL colour-under chroma, 0.6 us Y/C delay"),
    ('SVHS', "S-VHS (1987)",
     "400 TVL luma on the raised carrier, colour-under chroma as VHS"),
    ('BETAMAX', "Betamax (1975)",
     "250 TVL luma, 40 TVL colour-under chroma"),
    ('UMATIC', "U-matic (1971)",
     "260 TVL luma, 45 TVL chroma on the 688 kHz carrier, 0.5 us delay"),
    ('VIDEO8', "Video8 (1985)",
     "240 TVL luma, 40 TVL chroma on the 743 kHz carrier"),
    ('HI8', "Hi8 (1989)",
     "400 TVL luma, colour-under chroma as Video8"),
    ('BETACAM', "Betacam (1982)",
     "Component CTDM: 300 TVL luma, 120 TVL chroma, no Y/C delay"),
    ('BETACAM_SP', "Betacam SP (1986)",
     "Component: 340 TVL luma, 120 TVL chroma"),
    ('TYPE_C', "1-inch Type C (1978)",
     "The broadcast master: 400 TVL luma, 120 TVL chroma, no delay"),
)
DEBUG_PASS = _items(
    ('BEAUTY', "Beauty", ""), ('DEPTH', "Depth", ""), ('NORMAL', "Normal", ""),
    ('UV', "UV", ""), ('MATID', "Material ID", ""), ('OVERDRAW', "Overdraw", ""),
    ('WIREFRAME', "Wireframe", ""),
)
WIRE_MODE = _items(
    ('ALL', "All Edges", "Every triangle edge. What a wireframe renderer drew "
                         "-- and on a dense mesh at a period resolution every "
                         "pixel is within a pixel of an edge, so the surface "
                         "fills in solid"),
    ('CREASE', "Creases & Silhouette",
     "Only the outline and the edges where the surface turns by more than the "
     "angle below. Stays a wireframe however many triangles are behind it"),
    # R251 (RAST-B): the two period wire roads, both without a depth test
    ('ELITE', "Elite rule (BBC Micro 1984)",
     "An edge draws when either of its faces faces the camera; no depth "
     "test, so concave hulls show through-lines and nothing hides behind "
     "anything"),
    ('BEAM', "Vector beam (Atari DVG / AVG, Vectrex)",
     "Each feature edge is a phosphor stroke: Gaussian spot, quantised "
     "intensity, end-dwell dots, additive where strokes cross; no depth "
     "test"),
)
LIGHT_LIMIT = _items(
    ('BRIGHTEST', "Brightest", "Keep the strongest lights"),
    ('NEAREST', "Nearest", "Keep the closest lights"),
    ('FIRST', "First", "Keep them in scene order"),
)
SPEC_VIEWER = _items(
    ('PIXEL', "Per Pixel", "The true eye direction at every shaded point"),
    ('AXIS', "Camera Axis (OpenGL 1.1 / Sega Model / DS)",
     "One view vector for the whole frame: the camera's own axis"),
)
NORMAL_SOURCE = _items(
    ('AUTO', "Auto", "Use the mesh's own smooth/flat flags"),
    ('SMOOTH', "Force Smooth", ""), ('FACE', "Force Faceted", ""),
)
FORCE_MODEL = [('NONE', "Don't Override", "Let each material choose")] + \
    [(a, b, c) for a, b, c in MODEL_ITEMS]

RENDER_DEVICE = _items(
    ('CPU', "CPU", "Everything on the CPU. Every feature works"),
    ('GPU', "GPU", "Move the proven post stages to the GPU. Features that "
                   "cannot run there fall back to the CPU automatically"),
)

# R251 C055: which vector generator quantises the beam's strokes
BEAM_MACHINE = _items(
    ('DVG', "DVG: Asteroids / Battlezone (4-bit)",
     "16 intensity levels and a long end dwell; monochrome behind the "
     "overlay colour"),
    ('AVG', "AVG: Tempest / Major Havoc (3-bit doubled, colour RAM)",
     "Intensities 4..14 of 15, 1 bit per colour channel, a short dwell"),
    ('STARWARS', "AVG + STATZ: Star Wars 1983 (8-bit)",
     "256 intensity levels, 3 direct RGB bits, a short dwell"),
)

ENUMS = {
    # R233: the painted background road
    'bg_direction': _items(*BG_DIRECTION_ITEMS),
    'ink_style': _items(*INK_STYLE_ITEMS),
    'ink_texture': _items(*INK_TEXTURE_ITEMS),
    # R234: the inker's line
    'ink_anchor': _items(*INK_ANCHOR_ITEMS),
    'film_grade': _items(*FILM_GRADE_ITEMS),
    # R236: the colour process
    'film_process': _items(*FILM_PROCESS_ITEMS),
    'film_scratch_side': _items(*FILM_SCRATCH_SIDE_ITEMS),
    'ink_color_mode': _items(*INK_COLOR_ITEMS),
    'ink_gradient': _items(*INK_GRADIENT_ITEMS),
    'render_device': RENDER_DEVICE,
    'stereo_mode': STEREO_MODE,
    'dof_method': DOF_METHOD, 'dof_lens_pattern': DOF_LENS_PATTERN,
    'motion_blur_mode': MOTION_BLUR_MODE,
    'aa_mode': AA_MODE, 'aa_filter': AA_FILTER, 'subpixel_precision': SUBPIXEL,
    'aa_sample_pattern': AA_PATTERN,
    'depth_sort': DEPTH_SORT, 'painters_key': PAINTERS_KEY, 'shading_rate': SHADING_RATE,
    'pixel_center': PIXEL_CENTER, 'near_clip_mode': NEAR_CLIP,
    'depth_encoding': DEPTH_ENCODING,
    'vertex_quantize': VERTEX_QUANTIZE,
    'default_model': [(a, b, c) for a, b, c in MODEL_ITEMS],
    'force_model': FORCE_MODEL, 'normal_source': NORMAL_SOURCE,
    'specular_viewer': SPEC_VIEWER,
    'light_falloff_default': FALLOFF, 'shadow_default': SHADOW_MODE,
    'shadow_map_depth': SHADOW_MAP_DEPTH,
    'light_limit_mode': LIGHT_LIMIT, 'tex_filter': TEX_FILTER,
    'tex_wrap_default': WRAP, 'transparency': TRANSPARENCY,
    # R251 texture pack (TEX-1 + TEX-2)
    'tex_format': TEX_FORMAT, 'tex_tmem_format': TEX_TMEM,
    'tex_compress': TEX_COMPRESS, 'tex_frac_bits': TEX_FRAC_BITS,
    'tex_clamp_mode': TEX_CLAMP_MODE, 'tex_mip_select': TEX_MIP_SELECT,
    'tex_lod_source': TEX_LOD_SOURCE,
    'stipple_pattern': STIPPLE_PATTERN, 'fog_mode': FOG_MODE,
    'fog_table': FOG_TABLE, 'fog_depth': FOG_DEPTH,
    'fog_color_source': FOG_SOURCE, 'glow_quality':
    _items(('BOX', "Box", ""), ('GAUSS', "Gaussian", "")),
    'color_depth': COLOR_DEPTH, 'palette_mode': PALETTE_MODE,
    'palette_method': PALETTE_METHOD, 'dither': DITHER,
    'palette_bits': PALETTE_BITS,
    # R251 (post-palette, wave 2)
    'attribute_cells': ATTRIBUTE_CELLS,
    'scanline_palette': SCANLINE_PALETTE,
    'video_color_check': VIDEO_COLOR_CHECK,
    'video_system': VIDEO_SYSTEM, 'video_ire_limit': VIDEO_IRE_LIMIT,
    'color_management': COLOR_MGMT, 'crt_mask': CRT_MASK,
    'interlace': INTERLACE, 'output_scale': OUTPUT_SCALE,
    # R251 post-signal
    'vi_gamma': VI_GAMMA, 'copy_filter': COPY_FILTER,
    'crtc_blend': CRTC_BLEND, 'video_filter': VIDEO_FILTER,
    'chroma_format': CHROMA_FORMAT, 'chroma_upsample': CHROMA_UPSAMPLE,
    'signal': SIGNAL, 'pal_decoder': PAL_DECODER, 'tape': TAPE,
    'debug_pass': DEBUG_PASS,
    'wire_mode': WIRE_MODE,
    # R251 (transparency pack): the blend unit, the composite order,
    # the framebuffer format
    'blend_equation': BLEND_EQUATION_GLOBAL,
    'translucent_order': TRANSLUCENT_ORDER,
    'framebuffer': FRAMEBUFFER,
    'beam_machine': BEAM_MACHINE,     # R251 C055
    'material_override': _items(
        ('NONE', "Off", "Materials render as authored"),
        ('CLAY', "Clay", "Every material becomes one plain matte "
         "surface -- lights, shadows and geometry stay real")),
    # grouped: an item with an empty identifier is a category separator
    'res_preset': [('CUSTOM', "Custom", "")] + [
        item
        for label, keys in RESOLUTION_GROUPS
        for item in [('', label, '')] + [
            (k, resolution_label(k), resolution_description(k)) for k in keys]
    ],
}

# field -> (min, max, soft_min, soft_max, step/precision hints)
RANGES = {
    'outline_width': (1, 8),
    'ink_reference_height': (0, 8640), 'ink_taper': (0.0, 1.0),
    'ink_interior_scale': (0.0, 2.0), 'ink_weight_noise': (0.0, 1.0),
    'ink_weight_scale': (2.0, 400.0), 'ink_shadow_side': (0.0, 1.0),
    'ink_boil': (0.0, 16.0), 'ink_boil_fps': (0, 60),
    'ink_boil_scale': (2.0, 400.0), 'ink_pencil_strokes': (1, 6),
    'ink_pencil_spread': (0.0, 12.0), 'ink_grain': (0.0, 1.0),
    'ink_fill_darken': (0.0, 1.0),
    # R231: the drawn line
    'ink_end_taper': (0.0, 1.0), 'ink_end_length': (1.0, 200.0),
    'ink_roughness': (0.0, 1.0), 'ink_roughness_scale': (1.0, 64.0),
    'ink_drift': (0.0, 8.0), 'ink_gaps': (0.0, 1.0),
    'ink_texture_amount': (0.0, 1.0),
    # R234: the inker's line
    'outline_form_threshold': (0.02, 5.0), 'outline_shadow_level': (-0.5, 0.9),
    'outline_tone_threshold': (0.02, 1.0),
    'ink_isophote': (0.0, 1.0), 'ink_isophote_range': (2.0, 200.0),
    'ink_smooth': (0.0, 6.0), 'ink_pressure': (0.0, 1.0),
    'ink_overshoot': (0.0, 40.0),
    # R230: the era looks
    'film_grade_amount': (0.0, 1.0), 'film_softness': (0.0, 6.0),
    # R236: the colour process
    'film_process_amount': (0.0, 1.0), 'film_exposure': (-3.0, 3.0),
    'film_gamma': (0.8, 3.0), 'film_density': (1.0, 3.5),
    'film_filters': (-0.3, 0.6),
    'film_dye_purity': (0.0, 1.0), 'film_key': (0.0, 1.0),
    'film_halation': (0.0, 1.0), 'film_halation_radius': (2.0, 64.0),
    'film_register': (0.0, 4.0),
    'film_grain_size': (0.3, 8.0), 'film_grain_clump': (0.0, 1.0),
    'film_grain_chroma': (0.0, 1.0), 'film_dust_size': (0.3, 8.0),
    'film_dust_negative': (0.0, 1.0), 'film_dust_cel': (0.0, 1.0),
    'film_hairs': (0.0, 1.0), 'film_hair_length': (10.0, 600.0),
    'film_hair_width': (0.5, 6.0), 'film_hair_hold': (1, 240),
    'film_scratches': (0.0, 1.0), 'film_scratch_width': (0.5, 8.0),
    'film_scratch_hold': (1, 2000), 'film_reel': (0.0, 22.0),
    'film_weave': (0.0, 8.0), 'film_dust': (0.0, 1.0),
    'film_grain': (0.0, 1.0), 'film_flicker': (0.0, 1.0),
    'film_hold': (1, 4), 'film_halftone': (0.0, 1.0),
    'film_halftone_pitch': (2.0, 40.0), 'film_misregister': (0.0, 6.0),
    'film_bleed': (0.0, 4.0),
    # R233: the painted background road
    'bg_paint': (0.0, 1.0), 'bg_stroke_size': (2.0, 200.0),
    'bg_stroke_length': (1.0, 8.0), 'bg_angle': (-180.0, 180.0),
    'bg_spread': (0.0, 1.0), 'bg_bristles': (0.0, 1.0),
    'bg_variation': (0.0, 1.0), 'bg_smooth': (0.0, 12.0),
    'bg_paper': (0.0, 1.0), 'setback': (0.0, 16.0),
    'setback_start': (0.0, 10000.0), 'setback_range': (0.01, 10000.0),
    'volume_steps': (4, 256),
    'fog_bands': (0, 16),
    # R251 F015 (LIGHT-B1)
    'fog_spot': (0.0, 4.0),
    # R251 LIGHT-A2 (F006 / F009 / F010)
    'fog_ambient': (0.0, 4.0),
    'fog_bank1_start': (0.0, 100000.0), 'fog_bank1_end': (0.0, 100000.0),
    'fog_ground_offset': (-100000.0, 100000.0),
    'fog_ground_alt': (0.001, 100000.0),
    'fog_turbulence': (0.0, 10.0), 'fog_turb_depth': (0.0, 1.0),
    'outline_opacity': (0.0, 1.0),
    'outline_depth_threshold': (0.0005, 1.0),
    'outline_normal_angle': (5.0, 179.0),
    'stereo_eye_distance': (0.001, 2.0), 'stereo_convergence': (0.05, 500.0),
    'stereo_parallax_max': (1, 64),
    'aa_samples': (1, 64), 'aa_filter_width': (0.1, 4.0),
    'aa_edge_threshold': (0.0, 100.0), 'vertex_snap_grid': (0.05, 16.0),
    'depth_precision': (4, 32), 'ot_length': (16, 65536), 'ot_far': (0.1, 100000.0), 
    'vertex_units': (1.0, 4096.0),
    'light_clamp': (0.0, 1000.0), 'ao_distance': (0.001, 1000.0),
    'ao_samples': (1, 256), 'ao_intensity': (0.0, 1.0),
    'radiosity_samples': (1, 64), 'radiosity_distance': (0.01, 1000.0),
    'radiosity_spacing': (1, 8),
    'radiosity_intensity': (0.0, 4.0), 'reflection_blur': (0.0, 45.0),
    'global_ambient_level': (0.0, 10.0), 'max_lights': (0, 64),
    'process_count': (0, 64), 'shadow_map_size': (32, 4096), 'shadow_bias': (0.0, 10.0),
    'shadow_softness': (0.0, 32.0), 'shadow_samples': (1, 64),
    'planar_plane_z': (-10000.0, 10000.0), 'modvol_scale': (0, 255),
    'ray_depth': (0, 16), 'ray_bias': (0.0, 1.0),
    'reflection_blur_samples': (1, 64), 'tex_mip_bias': (-4.0, 4.0),
    'tex_aniso': (1, 16), 'tex_max_size': (0, 4096), 'tex_quantize': (0, 256),
    # R251 texture pack (wave-2 fields, TEX-2: the register's own ranges)
    'tex_colorkey_range': (0, 255), 'tex_lod_k': (-64.0, 63.9375), 'tex_lod_l': (0, 3),
    'tex_affine_subdiv': (0, 64), 'alpha_bits': (1, 8),
    'alpha_threshold': (0.0, 1.0), 'fog_start': (0.0, 100000.0),
    'fog_end': (0.0, 100000.0), 'fog_density': (0.0, 10.0),
    'fog_height_top': (-100000.0, 100000.0),
    'fog_height_falloff': (0.0, 100.0),
    'motion_shutter': (0.0, 4.0), 'motion_steps': (2, 64),
    'motion_samples': (1, 32), 'motion_dither': (0.0, 1.0),
    'motion_dither_tile': (8, 64),
    'pano_parts': (1, 16),
    'lens_distortion': (-1.0, 1.0), 'chromatic_aberration': (0.0, 8.0),
    'shaft_threshold': (0.0, 4.0), 'shaft_length': (0.0, 1.0),
    'shaft_decay': (0.5, 1.0), 'shaft_samples': (2, 128),
    'dof_focus': (0.01, 10000.0), 'dof_amount': (0.0, 8.0),
    'dof_layers': (2, 16), 'dof_max_radius': (1.0, 128.0),
    'dof_lens_samples': (2, 64),
    'displacement_scale': (0.0, 8.0),
    'shading_rate_area': (0.0, 64.0),          # R251 C119 (MAT-B)
    'glow_threshold': (0.0, 4.0), 'glow_radius': (1.0, 128.0),
    'glow_intensity': (0.0, 4.0), 'star_points': (2, 12),
    'star_length': (1.0, 256.0), 'star_rotation': (0.0, 6.2832),
    'star_intensity': (0.0, 4.0), 'flare_intensity': (0.0, 4.0),
    'flare_ghosts': (1, 12), 'flare_streak': (0.0, 2.0),
    'palette_size': (2, 256), 'dither_strength': (0.0, 2.0),
    'exposure': (0.0, 64.0), 'gamma': (0.1, 5.0), 'contrast': (-1.0, 4.0),
    'saturation': (0.0, 4.0), 'brightness': (-1.0, 1.0),
    'super_black_threshold': (0, 255),      # R251 C092: an 8-bit value
    'crt_scanlines': (0.0, 1.0), 'crt_mask_strength': (0.0, 1.0),
    'crt_bloom': (0.0, 2.0), 'crt_curvature': (0.0, 2.0),
    'crt_vignette': (0.0, 2.0), 'composite_bleed': (0.0, 2.0),
    'composite_ringing': (0.0, 2.0), 'composite_dot_crawl': (0.0, 2.0),
    'jpeg_quality': (1, 100), 'jpeg_passes': (1, 8), 'block_size': (4, 16),
    # R251 post-signal
    'crtc_alpha': (0, 255), 'video_filter_threshold': (0, 255),
    'rf_bandwidth': (2.0, 4.2), 'rf_beat': (0.0, 2.0), 'rf_snow': (0.0, 0.5),
    'rf_ghost': (0.0, 1.0), 'rf_ghost_delay': (0.1, 5.0),
    'pal_phase_error': (0.0, 45.0), 'pal_crawl': (0.0, 2.0),
    'tape_generations': (1, 8), 'tape_noise': (0.0, 0.5),
    'tape_dropouts': (0.0, 50.0),
    # R251 SIG-3: the codecs and the optical printer
    'mpeg1_qscale': (1, 31), 'mpeg1_gop': (0, 30),
    'smacker_quality': (0.0, 1.0),
    'matte_glow_radius': (1.0, 64.0), 'matte_glow_passes': (1, 5),
    'matte_glow_exposure': (0.0, 4.0),
    'threads': (0, 64), 'preview_scale': (1, 16), 'orbit_scale': (0, 16),
    'seed': (0, 2 ** 30),
    'wire_width': (0.1, 8.0), 'wire_angle': (0.0, 180.0), 'resolution_x': (1, 16384),
    'resolution_y': (1, 16384), 'pixel_aspect_x': (0.01, 100.0),
    'pixel_aspect_y': (0.01, 100.0), 
    'clip_near_epsilon': (1e-6, 1.0), 'decay_start': (0.0, 10000.0),
    # R251 (RAST-B): Elite's dot distance, the beam's spot sigma
    'wire_dot_distance': (0.0, 100000.0), 'beam_sigma': (0.2, 4.0),
}

LABELS = {
    # R251 (transparency pack)
    'blend_equation': "Blend Equation", 'translucent_order': "Composite Order",
    'translucent_depth_write': "Translucent Depth Write",
    'framebuffer': "Framebuffer Format", 'fb_dither': "Buffer Dither",
    'fb_dither_subtract': "Dither Subtraction",
    'stereo_mode': "Stereo", 'stereo_eye_distance': "Eye Distance",
    'stereo_convergence': "Convergence",
    'stereo_parallax_layers': "Parallax Layers (Virtual Boy)",
    'stereo_parallax_max': "Parallax Cap (px)",
    'camera_yshear': "Y-Shear Pitch",
    'pano_parts': "Pano Parts (2.4)",
    'aa_mode': "Anti-Aliasing", 'aa_samples': "Samples",
    'aa_filter': "Filter", 'aa_filter_width': "Filter Width",
    'aa_edge_threshold': "Edge Depth Threshold",
    'motion_blur': "Motion Blur", 'motion_shutter': "Shutter (frames)",
    'motion_steps': "Blur Steps",
    'motion_blur_mode': "Blur Mode", 'motion_samples': "Samples (Max)",
    'motion_dither': "Pass Dither", 'motion_dither_tile': "Dither Tile (px)",
    'dof_method': "DOF Method", 'dof_lens_pattern': "Lens Pattern",
    'dof_lens_samples': "Lens Passes",
    'fog_height': "Height Fog", 'fog_height_top': "Fog Top",
    'fog_height_falloff': "Height Falloff",
    'vertex_snap': "Vertex Snapping",
    'vertex_snap_grid': "Snap Grid (px)",
    'depth_precision': "Z-Buffer Bits", 'depth_sort': "Depth Method",
    'pixel_center': "Pixel Centre", 'near_clip_mode': "Near Plane",
    'depth_encoding': "Z-Buffer Encoding",
    'ot_length': "Table Entries", 'ot_far': "Table Far Distance",
    'vertex_quantize': "Vertex Format", 'vertex_units': "Vertex Units",
    'aa_sample_pattern': "Sample Pattern",
    'n64_coverage_aa': "N64 Coverage AA", 'n64_divot': "N64 Divot Filter",
    'shading_rate': "Shading Rate", 'default_model': "Default Shader",
    'force_model': "Override Shader",
    'specular_in_gamma': "Specular in Gamma Space",
    'crand_per_frame': "Crand Flickers per Frame",
    'specular_viewer': "Specular Viewer",
    'max_lights': "Light Limit", 'shadow_default': "Shadow Method",
    'shadow_map_depth': "Map Depth", 'planar_plane_z': "Floor Plane Z",
    'modvol_scale': "DC Shadow Scale",
    'tex_filter': "Texture Filter",
    'tex_perspective': "Perspective Correction",
    'tex_max_size': "Texture Size Limit", 'tex_quantize': "Texture Colours",
    # R251 texture pack
    'tex_format': "Texel Format", 'tex_tmem_format': "TMEM Budget (N64)",
    'tex_compress': "Block Compression", 'tex_frac_bits': "Bilinear Fraction Bits",
    'tex_clamp_mode': "Extend Clamp Mode", 'tex_colorkey': "Chroma Key (after filter)",
    'tex_colorkey_range': "Chroma Key Range", 'tex_mip_select': "Mip Level Select",
    'tex_lod_source': "Mip LOD Source", 'tex_lod_k': "GS LOD K", 'tex_lod_l': "GS LOD L",
    'tex_lod_sharpen': "Sharpen (N64)",
    'threads': "Threads", 'film_transparent': "Transparent Film",
    'cache_shadows': "Cache Shadow Maps", 'show_stats': "Timing Breakdown",
    'fast_background': "Fast Background",
    'orbit_scale': "Orbit Pixel Size",
    'material_override': "Material Override",
    'override_color': "Override Colour",
    'use_processes': "Use Worker Processes", 'process_count': "Processes",
    'gpu_post': "GPU Post Processing", 'render_device': "Device",
    'gpu_shading': "GPU Shading",
    'gpu_raster': "GPU Rasteriser",
    'gpu_hold_context': "Hold GPU Context (freezes UI)",
    'gpu_scissor': "Scissor Layer Passes",
    'viewport_gpu': "Viewport GPU",
    'radiosity': "Radiosity",
    'radiosity_samples': "Gather Samples",
    'radiosity_distance': "Gather Distance",
    'radiosity_intensity': "Bleed Intensity",
    'radiosity_spacing': "Gather Spacing",
    'reflection_blur': "Reflection Blur",
    'reflection_blur_samples': "Blur Samples",
    'watermark': "Burn-In Text",
    'palette_lock': "Lock Palette", 'displacement_scale': "Displacement",
    'shading_rate_area': "Shading Rate Area (REYES)",     # R251 C119 (MAT-B)
    'color_depth': "Colour Depth", 'palette_mode': "Palette",
    'palette_method': "Quantiser", 'dither': "Dither",
    'palette_bits': "Register Depth",
    # R251 (post-palette, wave 2)
    'attribute_cells': "Attribute Cells",
    'scanline_palette': "Scanline Palette",
    'super_black': "Super Black",
    'super_black_threshold': "Super Black Threshold",
    'video_color_check': "Video Color Check",
    'video_system': "Video System", 'video_ire_limit': "IRE Limit",
    'color_management': "View Transform", 'crt': "CRT Simulation",
    'composite': "Composite Video", 'jpeg_artifacts': "JPEG Artefacts",
    'output_scale': "Pixel Scale", 'debug_pass': "Render Pass",
    # R251 post-signal
    'vi_dither_filter': "VI De-dither (N64)", 'vi_gamma': "VI Gamma (N64)",
    'copy_filter': "Copy Filter (GameCube)",
    'crtc_blend': "CRTC Blend (PS2)", 'crtc_alpha': "CRTC Alpha",
    'crtc_bg_color': "CRTC Background",
    'video_filter': "Scan-out Filter (3dfx)",
    'video_filter_threshold': "Filter Threshold",
    'chroma_format': "Chroma Sampling", 'chroma_upsample': "Chroma Upsample",
    'signal': "Cable", 'rf_bandwidth': "RF Bandwidth (MHz)",
    'rf_beat': "RF 920 kHz Beat", 'rf_snow': "RF Snow",
    'rf_ghost': "RF Ghost", 'rf_ghost_delay': "Ghost Delay (us)",
    'pal_decoder': "PAL Decoder", 'pal_phase_error': "PAL Phase Error",
    'pal_crawl': "PAL Crawl",
    'tape': "Tape Format", 'tape_generations': "Generations",
    'tape_noise': "Tape Noise", 'tape_head_switch': "Head Switch",
    'tape_dropouts': "Dropouts / Frame",
    # R251 SIG-3: the codecs and the optical printer
    'mpeg1': "MPEG-1 (Video CD)", 'mpeg1_qscale': "MPEG-1 Scale",
    'mpeg1_gop': "MPEG-1 GOP",
    'smacker': "Smacker Blocks", 'smacker_quality': "Smacker Quality",
    'matte_glow': "Matte Glow (Tron)", 'matte_glow_radius': "Matte Radius",
    'matte_glow_passes': "Matte Passes",
    'matte_glow_exposure': "Matte Exposure",
    'render_wire': "Wireframe Overlay", 'wire_width': "Wire Width",
    'outline': "Cartoon Outlines", 'outline_color': "Ink Colour",
    'outline_width': "Ink Width", 'outline_opacity': "Ink Opacity",
    'outline_objects': "Object Edges",
    'outline_materials': "Material Edges",
    'outline_depth': "Depth Breaks",
    'outline_depth_threshold': "Depth Threshold",
    'outline_normals': "Creases",
    'outline_normal_angle': "Crease Angle",
    'outline_over_sky': "Ink Silhouettes",
    'outline_marked': "Marked Edges",
    'ink_style': "Line Style", 'ink_reference_height': "True To Height",
    'ink_taper': "Depth Taper", 'ink_interior_scale': "Interior Scale",
    'ink_weight_noise': "Weight Noise", 'ink_weight_scale': "Weight Scale",
    'ink_shadow_side': "Shadow Side", 'ink_boil': "Boil",
    'ink_boil_fps': "Boil Rate", 'ink_boil_scale': "Boil Scale",
    'ink_pencil_strokes': "Pencil Strokes", 'ink_pencil_spread': "Pencil Spread",
    'ink_grain': "Grain", 'ink_color_mode': "Line Colour",
    'ink_fill_darken': "Fill Darken", 'ink_color2': "Ink Colour 2",
    'ink_end_taper': "End Taper", 'ink_end_length': "End Length",
    'ink_roughness': "Roughness", 'ink_roughness_scale': "Roughness Scale",
    'ink_drift': "Drift", 'ink_gaps': "Gaps",
    'ink_texture': "Line Texture", 'ink_texture_amount': "Texture Amount",
    'outline_form': "Form Lines", 'outline_form_threshold': "Form Threshold",
    'outline_shadow': "Shadow Lines", 'outline_shadow_level': "Shadow Level",
    'outline_tone': "Tone Lines", 'outline_tone_threshold': "Tone Threshold",
    'ink_isophote': "Light Weight", 'ink_isophote_range': "Light Range",
    'ink_smooth': "Smooth", 'ink_pressure': "Pressure",
    'ink_overshoot': "Overshoot", 'ink_anchor': "Anchor",
    'film_grade': "Film Stock", 'film_grade_amount': "Stock Amount",
    'film_process': "Colour Process", 'film_process_amount': "Process Amount",
    'film_exposure': "Exposure", 'film_gamma': "Print Gamma",
    'film_density': "Dye Density", 'film_filters': "Filter Sharpness",
    'film_dye_purity': "Dye Purity",
    'film_key': "Silver Key", 'film_halation': "Halation",
    'film_halation_radius': "Halation Radius", 'film_register': "Register",
    'film_grain_size': "Grain Size", 'film_grain_clump': "Grain Clump",
    'film_grain_chroma': "Grain Chroma", 'film_dust_size': "Dust Size",
    'film_dust_negative': "Negative Dust", 'film_dust_cel': "Cel Dust",
    'film_hairs': "Hairs", 'film_hair_length': "Hair Length",
    'film_hair_width': "Hair Width", 'film_hair_hold': "Hair Hold",
    'film_scratches': "Scratches", 'film_scratch_width': "Scratch Width",
    'film_scratch_hold': "Scratch Hold", 'film_scratch_side': "Scratch Side",
    'film_reel': "Reel Length",
    'film_softness': "Softness", 'film_weave': "Gate Weave",
    'film_dust': "Dust & Hairs", 'film_grain': "Grain",
    'film_flicker': "Flicker", 'film_hold': "Shoot On",
    'film_halftone': "Halftone", 'film_halftone_pitch': "Halftone Pitch",
    'film_misregister': "Paint Misregistration", 'film_bleed': "Paint Bleed",
    # R233: the painted background road
    'bg_paint': "Brush Strokes", 'bg_stroke_size': "Stroke Size",
    'bg_stroke_length': "Stroke Length", 'bg_direction': "Stroke Direction",
    'bg_angle': "Stroke Angle", 'bg_spread': "Spread",
    'bg_bristles': "Bristles", 'bg_variation': "Variation",
    'bg_smooth': "Base Smoothing", 'bg_paper': "Board",
    'setback': "Setback", 'setback_start': "Setback Start",
    'setback_range': "Setback Range", 'setback_sky': "Setback the Sky",
    'ink_gradient': "Gradient",
    'volume_steps': "Volume Steps",
    'volume_shadows': "Volume Shadows",
    'fog_bands': "Fog Bands",
    'fog_spot': "Spotlight Fog (Model 3)",
    'fog_table': "Hardware Fog Table", 'fog_dither': "Fog Dither (Voodoo2)",
    'fog_face': "Per-Polygon Fog (Namco System 21)",
    'fog_depth': "Fog Depth",
    # R251 LIGHT-A2
    'fog_range_adjust': "Fog Range Adjust (GameCube)",
    'fog_ambient': "Fog Ambient (Model 3)",
    'fog_bank1_start': "Bank 1 Start", 'fog_bank1_end': "Bank 1 End",
    'fog_color_source': "Fog Target",
    'fog_ground_offset': "Fog Ground Offset",
    'fog_ground_alt': "Fog Ground Altitude",
    'fog_turbulence': "Fog Turbulence",
    'fog_turb_depth': "Fog Turbulence Depth",
    'wire_color': "Wire Colour", 'wire_mode': "Edges",
    'wire_angle': "Crease Angle",
    # R251 (RAST-B)
    'aa_clamp_samples': "Limit Dynamic Range",
    'aa_gamma_blend': "Gamma-2 Sample Blend",
    'wire_dot_distance': "Dot Distance",
    'beam_machine': "Vector Generator", 'beam_sigma': "Beam Spot",
    'pass_depth': "Depth", 'pass_normal': "Normal",
    'pass_position': "Position", 'pass_uv': "UV",
    'pass_object_index': "Object Index", 'pass_material_index': "Material Index",
}

DESCRIPTIONS = {
    # ---------------------------------------------------------- output
    'res_preset': "Jump the output straight to a period format -- VGA, "
                  "SVGA, broadcast -- with the pixel aspect that format "
                  "actually used, instead of dialling four numbers",
    'resolution_x': "Width of the rendered frame in pixels. Period software "
                    "lived at 320-800; the engine goes as high as you like",
    'resolution_y': "Height of the rendered frame in pixels. With the "
                    "pixel aspect pair this sets the picture's true shape",
    'pixel_aspect_x': "Horizontal pixel stretch. Broadcast and mode-13h "
                      "formats used non-square pixels; 1.0 is square",
    'pixel_aspect_y': "Vertical pixel stretch. Set with X to reproduce a "
                      "format's true pixel shape; 1.0 is square",
    # ---------------------------------------------------- anti-aliasing
    'aa_mode': "How edges are smoothed. None is the raw hard-edged raster, "
               "Supersample renders larger and filters down (the era's "
               "quality switch), Edge Only spends samples where geometry "
               "ids change, Adaptive renders once then re-samples just "
               "the edge pixels (Bryce's second pass), Accumulation "
               "averages jittered whole frames",
    'aa_samples': "Samples per pixel for the chosen mode. Supersample "
                  "rounds it to a square (24 renders at 5x5); more is "
                  "smoother and proportionally slower. Adaptive pays "
                  "this price only at edge pixels",
    'aa_filter': "The downfilter shape used to combine samples. Box is the "
                 "period default; Tent and Gauss trade a little sharpness "
                 "for calmer edges",
    'aa_filter_width': "Radius of the downfilter in output pixels. Wider "
                       "is softer; 1.0 keeps each pixel to its own samples",
    'stereo_mode': "Render two eyes and combine them: red/cyan anaglyph "
                   "or a side-by-side pair. Parallel cameras with an "
                   "off-axis frustum, so vertical parallax never appears",
    'stereo_eye_distance': "Distance between the two cameras, in scene "
                           "units. Human-scale scenes want ~0.065",
    'stereo_convergence': "Distance at which the two views line up "
                          "exactly (zero parallax). Nearer objects pop "
                          "out of the screen, farther ones sink in",
    'stereo_parallax_layers': "Virtual Boy stereo: render ONCE and shift every "
                              "object as a flat card by one whole-pixel parallax "
                              "per object (its centroid depth through Eye "
                              "Distance and Convergence), the sky by the cap -- "
                              "the VIP's per-world parallax, not two cameras. "
                              "Pack the pair with the Stereo mode chosen above",
    'stereo_parallax_max': "Largest whole-pixel shift a layer may take either "
                           "way; the sky always takes it. Virtual Boy games "
                           "kept their parallax within a few pixels of the "
                           "384-wide frame for comfort",
    'camera_yshear': "Render the camera LEVEL and slide the projection centre by "
                     "focal x tan(pitch) instead of tilting it -- the Y-shear of "
                     "Heretic, Hexen, Marathon and the Build engine (Duke Nukem "
                     "3D). Verticals stay vertical, the horizon slides; the shear "
                     "is capped at 32 degrees up and 56 down, ZDoom's limits, and "
                     "roll is dropped as those engines had none. Perspective "
                     "cameras only",
    'pano_parts': "Blender 2.4's Pano + Xparts: the frame width is divided "
                  "into N planar strips (the last one cropped when N does not "
                  "divide it), the camera yawed by one strip's horizontal "
                  "field between them and the strips butted together "
                  "unblended -- straight lines kink at every seam, the tell "
                  "of the era's QuickTime VR cylinders. With a 32 mm sensor "
                  "four 16 mm parts close the circle. Stereo and Y-shear are "
                  "off inside the strips, as under the Panoramic camera; "
                  "accumulation AA, adaptive AA and lens-pass depth of field "
                  "run inside each strip; the worker pool stands aside (the "
                  "strips render in-process). 1 = off; the PANO camera type "
                  "is the true cylinder",
    'motion_blur': "Blur moving objects across the shutter interval by "
                   "averaging time-offset frames, as the era's renderers "
                   "faked it -- expect the cost of Steps extra renders",
    'motion_shutter': "How long the virtual shutter stays open, in frames. "
                      "0.5 is a 180-degree film shutter; longer smears more",
    'motion_steps': "Time samples across the shutter. Few steps show the "
                    "classic stepped ghosting; many approach smooth blur "
                    "at a render each",
    'motion_blur_mode': "How the time slices combine: every step averaged "
                        "(the accumulation buffer), 3ds Max Object Motion "
                        "Blur's per-pixel random subset of the slices, or "
                        "LightWave's Dithered blur splitting the slices by "
                        "scanline parity",
    'motion_samples': "Max's Samples: how many of the Blur Steps slices each "
                      "pixel averages. Equal to Blur Steps the blur is smooth "
                      "(the plain average); fewer is grainier. 3ds Max's "
                      "default 10 of 10, cap 32",
    'motion_dither': "3ds Max 3+ multi-pass Dither Strength: each slice's "
                     "weight is 1 +/- this by an 8x8 ordered pattern phased "
                     "per slice, weights normalised -- 0 is the plain "
                     "average, Max's default 0.4",
    'motion_dither_tile': "3ds Max multi-pass Tile Size: how many pixels the "
                          "8x8 dither pattern spans (Max's default 32)",
    # ------------------------------------------------------- rasteriser
    'backface_cull': "Skip triangles facing away from the camera at the "
                     "raster, as period hardware did. Closed meshes look "
                     "identical and draw faster; open shells lose their "
                     "insides",
    'two_sided_lighting': "Light both faces of every surface: a normal "
                          "facing away from the light flips before shading. "
                          "How the era rendered single-sided geometry "
                          "without black backs",
    'subpixel_precision': "How many fractional bits vertex positions keep "
                          "at the raster. Full precision is modern and "
                          "steady; fewer bits snap vertices to a coarser "
                          "grid and edges wobble as things move -- the "
                          "PS1-era jitter, by its real mechanism",
    'vertex_snap_grid': "Snap screen-space vertices to this fraction of a "
                        "pixel before rasterising. 0 is off; larger steps "
                        "give the polygon shimmer of fixed-point hardware",
    'depth_precision': "Bits of depth buffer. 32 is exact for any scene; "
                       "24 is the period standard; 16 brings the z-fighting "
                       "and poke-through of budget hardware, faithfully",
    'pixel_center': "Where the rasteriser samples each pixel. Direct3D 3 "
                    "to 9 sampled at the integer corner while texels sat "
                    "at +0.5, so 1:1 textures blurred by half a texel and "
                    "the whole frame sat half a pixel up-left; OpenGL and "
                    "Direct3D 10 sample at +0.5",
    'near_clip_mode': "What happens to a triangle that crosses the near "
                      "plane. Clip cuts it as software renderers did; "
                      "Reject drops it whole, as the PS2's VU1 fast path "
                      "and the PS1 did, so geometry pops instead of being "
                      "cut",
    'depth_encoding': "How the z-buffer stores depth. Fixed point rounds "
                      "normalised z to Z-Buffer Bits; the N64 RDP kept "
                      "18-bit z in a piecewise-floating code (coarse near, "
                      "fine far), the GameCube compressed 24-bit z by "
                      "octave, Xbox and Voodoo buffered w instead of z -- "
                      "each moves the z-fighting bands to where that "
                      "machine had them",
    'ot_length': "Entries in the PlayStation ordering table. Every "
                 "polygon's mean depth is floored into one of these "
                 "buckets; fewer buckets merge more polygons into one draw "
                 "order and mis-sort more",
    'ot_far': "Scene distance that lands in the last ordering-table "
              "bucket. Polygons past it fall outside the table and are "
              "not drawn at all, the PS1's far pop-in",
    'vertex_quantize': "Round mesh data to the console's vertex format "
                       "before projection: 16-bit integer positions on a "
                       "world grid of Vertex Units per unit, PS1 1.3.12 "
                       "or N64 8-bit normals, PS1 integer-texel UVs and "
                       "8-bit vertex colours -- the crunch and texture "
                       "jitter of integer geometry. A world grid, "
                       "disclosed: the consoles quantised in model space",
    'vertex_units': "Integer vertex units per world unit for the "
                    "vertex-format quantisation. 64 puts vertices on a "
                    "1/64-unit lattice; fewer units crunch harder",
    'aa_sample_pattern': "Where the supersampler tests inside each subpixel. "
                         "Grid samples the centre (moire and stair-steps); "
                         "Jitter hashes a position inside every subpixel per "
                         "pixel and frame, as REYES did, so aliasing turns "
                         "into fine noise. At 1 sample it is a per-pixel "
                         "offset",
    'n64_coverage_aa': "The N64's anti-aliasing: the rasteriser stores 3 "
                       "bits of coverage per pixel from 16 subsamples of "
                       "the winning polygon, and the Video Interface pulls "
                       "every partially covered pixel toward the "
                       "penultimate extremes of its six fully covered "
                       "neighbours by (7 - coverage)/8, softening every "
                       "polygon edge, interior seams included. Needs 1 "
                       "sample per pixel; reads the frame as the RDP's "
                       "16-bit (5551) framebuffer, so smooth shading "
                       "bands at 5 bits per channel across the whole "
                       "frame, whatever Colour Depth is set to",
    'n64_divot': "The VI's divot filter after the coverage blend: any "
                 "pixel whose horizontal triple holds a partial-coverage "
                 "pixel becomes the per-channel median of the three. The "
                 "N64's default with anti-aliasing on",
    'clip_near_epsilon': "Safety margin for the near-plane clip, in NDC "
                         "units. Raise it only if geometry grazing the "
                         "camera shows cracks; lowering it buys nothing",
    # ---------------------------------------------------------- shading
    'default_model': "The reflectance model a material gets when its node "
                     "tree does not choose one -- the engine-wide house "
                     "style. Each entry's tooltip describes its era",
    'force_model': "Override EVERY material's model with one choice for "
                   "this render -- the whole scene as wireframe, flat, or "
                   "chalk in one switch. A diagnostic and a look in itself",
    'shading_rate': "Where lighting is computed: every pixel (Phong-era "
                    "per-pixel), once per vertex and interpolated "
                    "(Gouraud, the hardware look), or once per face "
                    "(flat). The single biggest period-look switch",
    'normal_source': "Which normals shading uses: the mesh's smoothed "
                     "normals, or faceted face normals everywhere -- the "
                     "un-smoothed look of early scanline output",
    'clamp_specular': "Cap each highlight's brightness at this value "
                      "before it saturates the frame. 0 is uncapped; the "
                      "era clamped to keep 8-bit channels from blowing out",
    'light_clamp': "Cap the summed light on any surface point. Keeps "
                   "stacked lights inside the period's dynamic range "
                   "instead of washing to white; 0 is uncapped",
    'ambient_occlusion': "Darken creases and contact areas with hemisphere "
                         "rays -- not period-authentic, but period-adjacent "
                         "grime that reads well on chunky geometry",
    'ao_distance': "How far an occlusion ray looks for nearby geometry. "
                   "Short keeps the effect to contact shadows; long "
                   "darkens whole rooms",
    'ao_samples': "Occlusion rays per pixel. More is smoother and slower; "
                  "8 reads clean at period resolutions",
    'ao_intensity': "Strength of the occlusion darkening. 1 is full "
                    "effect; fractions fade it toward none",
    'global_ambient': "Colour of the light that reaches everything from "
                      "nowhere -- the flat ambient term every 1990s "
                      "renderer had. Tints every unlit area",
    'global_ambient_level': "Multiplier on the global ambient colour. The "
                            "fastest single knob for overall scene "
                            "brightness outside the lights themselves",
    'light_falloff_default': "How light dims with distance when a light "
                             "does not choose: physically correct inverse "
                             "square, the gentler inverse, the era's "
                             "none-at-all, or a custom start/end ramp",
    'light_limit_mode': "Which lights survive when the scene has more than "
                        "the Light Limit: the brightest, the nearest, or "
                        "simply the first -- emulating fixed-function "
                        "hardware's light slots",
    # ---------------------------------------------------------- shadows
    'shadows': "Master switch for all shadowing. Off is the flat, "
               "floating look of the earliest real-time output",
    'sss': "Master switch for Blender Internal subsurface scattering "
           "(2.79's R_SSS): materials with SSS enabled pre-render a "
           "point cloud and gather the dipole, on either device",
    'auto_fix_appended_lamps':
        "When File > Append (or a drag-and-drop) brings in lights from "
        "a classic .blend (2.79 and earlier), immediately stamp the "
        "file's own Blender Internal values onto them -- energy, "
        "colour, falloff, shadow rule -- instead of Blender's watt "
        "conversion (which lands every classic sun at 1.0). Receipts "
        "go to a '<file> lamp fix log' text datablock. The Fix "
        "Appended Lamps button below does the same by hand",
    'shadow_default': "How shadows are computed when a light does not "
                      "choose: depth maps rendered from each light (soft, "
                      "fast, the period standard), traced rays (hard and "
                      "exact), or per-light choice",
    'shadow_map_size': "Resolution of each light's depth map. Small maps "
                       "give blocky period shadows; large maps sharpen "
                       "contact edges. Lights can override individually",
    'shadow_bias': "World-space offset that keeps a surface from shadowing "
                   "itself. Too low shows acne stippling; too high "
                   "detaches shadows from their objects (Peter Panning)",
    'shadow_softness': "Blur radius of mapped shadows, in shadow-map "
                       "texels. 0 is hard-edged; more taps a wider "
                       "neighbourhood for softer penumbras",
    'shadow_samples': "Taps per pixel for soft mapped shadows, or rays per "
                      "pixel for soft traced shadows from area-sized "
                      "lights. More is smoother and slower",
    # R251 C117
    'shadow_map_depth': "What each shadow-map texel stores. CLASSIC keeps "
                        "the nearest caster and fights acne with Bias and "
                        "a normal offset. MIDPOINT stores the halfway "
                        "point between the two nearest casters along the "
                        "texel (Woo 1992, Maya's Use Mid Dist, Blender's "
                        "Classic-Halfway): the compare lands inside the "
                        "caster's thickness, so Bias and the offset are "
                        "zero and thin closed objects cast without a gap. "
                        "A texel that sees a lone surface stores the far "
                        "plane: a single open face casts nothing (mental "
                        "ray's rule). A MIDPOINT map costs two rasters of "
                        "the casters instead of one",
    # R251 C052
    'planar_plane_z': "World height of the receiver plane (normal +Z) that "
                      "planar shadows are drawn on: casters are projected "
                      "along each lamp onto Z = this value, the way Model 1 "
                      "flattened fighters onto the ring floor. Only lamps "
                      "within the Light Limit cast; every frame (viewport "
                      "drafts included) pays one extra raster of the "
                      "casters per casting lamp",
    # R251 C020
    'modvol_scale': "The Dreamcast's FPU_SHAD_SCALE register: inside a "
                    "modifier volume the 8-bit colour becomes colour x "
                    "scale / 256, integer -- 128 is the half-brightness "
                    "shadow Sonic Adventure used, 0 is black, 255 barely "
                    "darkens. Read only while a Dreamcast Modifier volume "
                    "exists in the scene; each such volume costs one "
                    "fragment capture per frame",
    # ------------------------------------------------------ ray tracing
    'raytrace': "Master switch for traced reflections and refractions -- "
                "the checkbox that separated the raytracers from the "
                "scanliners. Materials still choose their own amounts",
    'ray_depth': "How many times a ray may bounce between mirrors or "
                 "through glass before giving up and taking the sky. "
                 "Two facing mirrors show this number directly",
    'ray_reflection': "Allow traced reflections for materials that ask "
                      "for them. Off, reflective surfaces fall back to "
                      "the environment term",
    'ray_refraction': "Allow traced refraction through transparent "
                      "materials with an IOR. Off, glass becomes simple "
                      "alpha transparency",
    'ray_bias': "How far a spawned ray steps off its surface before "
                "testing the world, in world units. Too low and surfaces "
                "shadow and reflect themselves as speckle; too high and "
                "contact detail goes missing",
    'env_reflection': "Let materials reflect the world background where "
                      "rays are off or exhausted -- the era's spherical "
                      "environment map trick",
    # --------------------------------------------------------- textures
    'tex_filter': "How textures are sampled: Nearest is the chunky texel "
                  "look, Bilinear smooths, Trilinear adds mip blending, "
                  "3-Point is the N64's, Normalised Distance is POV-Ray's "
                  "interpolate 4, Summed Area is 3D Studio's box integral. "
                  "The single most visible texture-era switch",
    'tex_mipmap': "Build and use mip pyramids so distant textures calm "
                  "down instead of sparkling. Off reproduces the shimmer "
                  "of software that never mipped",
    'tex_mip_bias': "Shift mip selection sharper (negative) or blurrier "
                    "(positive), in levels. Period hardware often biased "
                    "sharp and lived with the noise; under Summed Area it "
                    "scales the footprint rectangle by 2^bias, with or "
                    "without Mipmaps; inert under the PlayStation 2 LOD (K "
                    "replaces it)",
    'tex_aniso': "Maximum anisotropy for Trilinear: up to this many trilinear "
                 "taps along the footprint's long axis, the N-tap box of "
                 "late-90s cards (not EWA). Higher keeps floors readable at "
                 "grazing angles; 1 is isotropic. Off under the PlayStation 2 "
                 "LOD, a Mip Level Select or Sharpen",
    'tex_max_size': "Downsample any texture larger than this before use, "
                    "as period VRAM budgets forced. 0 keeps full size",
    'tex_quantize': "Reduce every texture to this many colours before "
                    "sampling -- palettised texture memory, with its "
                    "banding, exactly as shipped games had it",
    'tex_affine_subdiv': "For affine (PS1-style) texture mapping: cut each "
                         "triangle into this many pieces so the warp stays "
                         "bounded. 1 is the full classic wobble",
    'tex_wrap_default': "What textures do past their edges when the node "
                        "does not say: repeat, mirror, clamp, or clip to "
                        "nothing",
    # ------------------------------------------ R251 texture pack (TEX-1)
    'tex_format': "Store every texture at the card's texel format before "
                  "sampling: rounded once like the Glide and Direct3D drivers "
                  "did, expanded by bit replication on fetch. 4:4:4:4 bands "
                  "skies, 1:5:5:5 cuts alpha hard, Model 2's 4-bit luma tints "
                  "through the material colour, the DS's A3I5 keeps 8 alpha "
                  "levels, 3dfx NCC keeps 16 luma steps and a 4x4 chroma table. "
                  "Ignored while a TMEM Budget is set: the N64 format owns the "
                  "texel",
    'tex_tmem_format': "Fit every texture into the Nintendo 64's 4 KB TMEM by "
                       "the chosen format: rows padded to 64-bit words, "
                       "colour-index formats confined to the lower 2 KB, mip "
                       "levels counted when Mipmaps is on, then the largest "
                       "halving that fits and the format's own texel depth "
                       "(RGBA16 = 5551, CI4 = 16 colours, I4 = 16 greys feeding "
                       "alpha). A TMEM format owns the texel format: Texel "
                       "Format is ignored while this is set",
    'tex_compress': "Bake the era's block compression into every texture "
                    "before sampling: DXT1's two 5:6:5 endpoints and four "
                    "colours per 4x4 block (S3TC), the Xbox NV2A's 16-bit decode "
                    "banding, the GameCube's CMPR 3/8-5/8 interpolants, or the "
                    "Dreamcast's 256-entry 2x2 vector-quantised codebook. Mip "
                    "levels are box-halved from the compressed level",
    'tex_frac_bits': "Keep only the top bits of the texel fraction before "
                     "blending, as the Voodoo1's bilinear unit did (4 bits: "
                     "sixteen steps between texels, faint terraces on magnified "
                     "textures) or the Voodoo2's (8 bits). Applies to Bilinear "
                     "and to each Trilinear level; 3-Point keeps the RDP's own "
                     "arithmetic",
    # ------------------------------------------ R251 texture pack (TEX-2, wave 2)
    'tex_clamp_mode': "How Extend-mode textures clamp: Clamp to Edge repeats the "
                      "last texel; GL_CLAMP clamps the coordinate but lets the "
                      "bilinear taps past the last texel read OpenGL 1.1's "
                      "transparent-black border, the half-texel dark seam of "
                      "GLQuake-era sky boxes on conformant drivers",
    'tex_colorkey': "Test the texture's colour against black AFTER filtering, as "
                    "the Voodoo's chroma key and Direct3D's colour key did; alpha "
                    "cut-outs are stored as opaque black first, so the filter "
                    "bleeds a dark fringe into every keyed edge that then "
                    "survives the test. A texture's own alpha above one half is "
                    "kept",
    'tex_colorkey_range': "Voodoo2 chromaRange: a texel within this many 8-bit "
                          "levels of black on every channel is keyed out; 0 is "
                          "the Voodoo1 and Direct3D exact match",
    'tex_mip_select': "How a mip level is picked when Mipmaps is on: as the "
                      "filter does today (Trilinear blends, the rest read level "
                      "0), Blend two levels with the filter's own taps (the N64 "
                      "RDP's 3-point mip blend; Bilinear + Blend is exactly "
                      "Trilinear), one Nearest level as OpenGL 1.1's "
                      "LINEAR_MIPMAP_NEAREST and Direct3D's point mip did, or the "
                      "Voodoo1's 4x4-dithered level pick that stipples every mip "
                      "seam. Anisotropy is off under a level select: no card had "
                      "both",
    'tex_lod_source': "Where the mip level comes from: the screen-space texture "
                      "derivatives every PC card used, the PlayStation 2 GS's "
                      "rule -- log2 of the interpolated Q (1/w) alone, shifted by "
                      "L and offset by K in 7.4 fixed point -- so mip bands sit "
                      "at fixed distances whatever the surface angle, or one "
                      "level per polygon from the triangle's texel-to-pixel area "
                      "ratio (Riva 128, Verite) -- one NEAREST level, no blend: "
                      "under it As the Filter and Blend read as Nearest Level, "
                      "the Voodoo dither still applies. Needs Mipmaps (the GS "
                      "rule also needs Trilinear or a Mip Level Select); "
                      "anisotropy and Mip Bias are off under the GS rule (K "
                      "replaces the bias)",
    'tex_lod_k': "The GS TEX1 K offset in levels, signed 7.4 fixed point "
                 "(quantised to 1/16): negative sharpens, positive blurs, tuned "
                 "per game on the real machine",
    'tex_lod_l': "The GS TEX1 L shift: the LOD from Q is multiplied by 2^L "
                 "before K is added",
    'tex_lod_sharpen': "Nintendo 64 sharpen mode (gDPSetTextureDetail "
                       "G_TD_SHARPEN): under magnification the RDP extrapolates "
                       "level 0 away from level 1 with a negative LOD fraction -- "
                       "most at full magnification, none at 1:1 -- and its 9-bit "
                       "clamp saturates the overshoot into crisp, ringing texel "
                       "edges. Needs Mipmaps on; anisotropy is off under it",
    # ----------------------------------------------------- transparency
    'transparency': "How see-through surfaces composite: sorted alpha "
                    "blending, the A-buffer's exact per-pixel lists, or "
                    "Screen Door's dither-pattern holes (no sorting, no "
                    "blending, pure period)",
    'stipple_pattern': "The hole pattern Screen Door transparency punches, "
                       "per opacity level: the Bayer and clustered-dot "
                       "ordered maps, the Mega Drive / SNES pseudo-hires "
                       "1x2 column mesh, or the N64 RDP's per-pixel random "
                       "alpha compare. Each is a real pattern family from "
                       "shipped hardware",
    # R251 (transparency pack)
    'blend_equation': "The blend unit's equation for every see-through "
                      "surface under Sorted Blend or A-Buffer. Alpha Over "
                      "is Halcyon's float blend; the PlayStation, Saturn "
                      "VDP1, 3DO PIXC, SNES, GBA and DS items are those "
                      "chips' fixed integer formulas on 5- or 6-bit "
                      "channels (saturating adds and subtracts, halves, "
                      "sixteenths), applied per layer in Halcyon's own "
                      "depth order. Fuzz, Thin Wall and Imagine Fog are "
                      "the software looks of the same stage. A "
                      "material's Blend Mode overrides this for that "
                      "material",
    'translucent_order': "The order see-through layers composite in under "
                         "Sorted Blend or A-Buffer. Depth is Halcyon's "
                         "per-fragment (A-Buffer) or per-polygon (Sorted) "
                         "order; the Nintendo DS auto-sort ordered "
                         "translucent polygons by their bottom screen "
                         "row, then top row, then submission, never by "
                         "depth -- the source of its mis-sorted water and "
                         "glass -- and its manual sort drew them in "
                         "submission order",
    'translucent_depth_write': "Translucent fragments write the depth "
                               "buffer as they composite, so a later "
                               "fragment in the composite order that lies "
                               "behind one already drawn at that pixel is "
                               "dropped -- the DS POLYGON_ATTR bit 11 "
                               "behaviour (and fixed-function GL with depth "
                               "writes left on). Off, layers never occlude "
                               "each other",
    'framebuffer': "The framebuffer format the frame is written INTO, not "
                   "the output depth: the PlayStation 2's 16-bit buffer, "
                   "the GameCube EFB's 6-bit channels and the 3dfx "
                   "Voodoo's RGB565 truncate at every write and read the "
                   "truncated value back for every blend, so stacked "
                   "see-through layers, particles and fog volumes "
                   "accumulate their dither into grain where they "
                   "overlap. Colour Depth at post quantises the finished "
                   "frame once; this quantises inside the pipeline. "
                   "Under supersampling the resolve averages the dither "
                   "away: the presets that ship it use no or edge AA",
    'fb_dither': "The buffer's hardware dither at each write: the GS "
                 "DTHE/DIMX matrix (PlayStation 2), the Voodoo's "
                 "grDitherMode matrix. Off is the GS with DTHE clear or "
                 "Glide's GR_DITHER_DISABLE -- plain truncation. The "
                 "GameCube RGBA6 format always dithers and ignores this",
    'fb_dither_subtract': "Voodoo 'alpha dither subtraction' (fbzMode bit "
                          "19): a blend reads the dithered 565 destination "
                          "back with the matrix value added back before "
                          "blending, undoing the dither for that read. Off "
                          "leaves the read raw, so the grain compounds "
                          "harder. With Buffer Dither off nothing was "
                          "added at the write, so nothing is subtracted "
                          "at the read. Only the Voodoo formats read this",
    'alpha_bits': "Bits of opacity resolution for the alpha channel. "
                  "Fewer bits step smooth fades into visible bands",
    'alpha_threshold': "Hard opacity cutoff: below this a pixel is "
                       "dropped outright instead of blended -- the "
                       "classic cut-out alpha test for foliage cards "
                       "and fences. 0 disables it so thin glass and "
                       "smoke blend properly",
    # --------------------------------------------------------------- fog
    'fog': "Master switch for distance fog, the era's draw-distance "
           "disguise and mood in one",
    'fog_mode': "The fog curve: Linear between Start and End, or the two "
                "exponential falls hardware offered. Linear is the most "
                "controllable; Exp2 is the deepest soup",
    'fog_color': "The colour distant geometry fades toward. Matching the "
                 "sky hides the far clip; contrasting it makes fog a look",
    'fog_start': "Distance where linear fog begins, in world units. "
                 "Nearer than this is fully clear",
    'fog_end': "Distance where linear fog saturates; beyond this "
               "everything is fog colour",
    'fog_density': "Steepness of the exponential fog modes. Small numbers "
                   "haze the horizon; large ones swallow the middle ground",
    'fog_table': "Quantise the fog through the accelerator's own fog "
                 "table: 3dfx Voodoo's 64 entries indexed by the pixel's "
                 "1/w and interpolated by its next 8 mantissa bits "
                 "(Glide), the PowerVR2's 128 log-spaced entries below "
                 "Fog End (Dreamcast), or the Nintendo DS's 32 equal "
                 "steps between Fog Start and Fog End. The Fog Mode "
                 "curve fills the table at each entry's own depth; a "
                 "16-Step Table mode is read as Linear here. For the "
                 "Voodoo table w is eye depth in scene units (one scene "
                 "unit is one Glide w unit, so a metre-scale scene uses "
                 "the near octaves) and Fog Start does not move the "
                 "table; every table replaces Per-Vertex Fog's rounding "
                 "with its own steps",
    'fog_dither': "Voodoo2's fog dither: the 4x4 ordered matrix is added "
                  "to the fog interpolation fraction before its final "
                  "shift (fogMode bit 6), the fine checker inside the "
                  "Voodoo2's fog bands. Only the 64-entry table reads it",
    'fog_face': "One fog step per polygon: the fog curve is evaluated once at "
                "the mean depth of the triangle's three corners and every pixel "
                "of the polygon takes that value, so whole polygons pop between "
                "haze bands as on Namco's System 21 (the board averaged its "
                "quads' four corners; a triangle pair straddling a bank pops as "
                "two halves). Fog Bands sets the bank count (16 or 32 on the "
                "hardware); it replaces Per-Vertex Fog's rounding with the "
                "polygon's own step",
    # R251 LIGHT-A2: F004 / F006 / F008 / F009 / F010
    'fog_range_adjust': "GameCube's horizontal fog range adjustment: the "
                        "planar fog depth is scaled by the secant of each "
                        "pixel column's view angle through a 10-knot table "
                        "at 32-pixel steps of the output frame "
                        "(GX_InitFogAdjTable), so the fog stops breathing "
                        "at the frame edges when the camera turns. Off in "
                        "GXInit; inert under an orthographic camera",
    'fog_ambient': "Model 3's fogAmbient: scales the fog colour every "
                   "material blends toward, the 8-bit viewport register "
                   "that dimmed the whole haze. A Backdrop fog target is "
                   "not scaled (LightWave had no such register)",
    'fog_bank1_start': "Fog Start for materials on Fog Bank 1 -- System "
                       "22's second cz table; inert while Bank 1 End <= "
                       "Bank 1 Start. The board had four banks; Halcyon "
                       "ships this one beside the scene's",
    'fog_bank1_end': "Fog End for materials on Fog Bank 1 -- System 22's "
                     "second cz table; inert while Bank 1 End <= Bank 1 "
                     "Start. The board had four banks; Halcyon ships this "
                     "one beside the scene's",
    'fog_color_source': "What the fog fades toward: the Fog Colour swatch, "
                        "or LightWave's Use Backdrop Color -- the world "
                        "colour the same pixel would show if nothing were "
                        "drawn there (gradient, bands, HDRI, painted "
                        "backdrop), so distant objects dissolve into the "
                        "sky behind them while the backdrop itself "
                        "receives no fog. Reflections and vertex-lit "
                        "corners take the world colour along their own "
                        "direction instead",
    'fog_ground_offset': "World height of POV-Ray's constant-density "
                         "layer: below it the ground fog is full density, "
                         "above it thins as 1/(1+Y^2) over Fog Ground "
                         "Altitude",
    'fog_ground_alt': "POV-Ray's fog_alt: the height above the offset at "
                      "which the ground fog's density has halved (the "
                      "shipped code's law, not the manual's quarter)",
    'fog_turbulence': "POV-Ray's fog turbulence: one value-noise "
                      "turbulence read at the middle of each pixel's "
                      "fogged path scales the fog distance, faded by "
                      "exp(-distance x Fog Density) so near fog goes wispy "
                      "and far fog stays smooth. Six octaves, lambda 2, "
                      "omega 0.5 (POV's defaults); 0 leaves the fog "
                      "untouched",
    'fog_turb_depth': "POV-Ray's turb_depth: how much of the fog distance "
                      "the turbulence may remove (min(1, turbulence x "
                      "depth)); 0.5 is POV's default",
    'fog_depth': "Where the fog reads its distance: eye depth (w), or "
                 "Direct3D's post-projection z in 0..1 as pixel fog did "
                 "on devices without W-fog -- Fog Start and End are then "
                 "0..1 depths and almost the whole scene sits above "
                 "0.99, so the fog arrives as a wall. Applies to the "
                 "Linear/Exp/Exp2/16-Step curves; the console modes and "
                 "hardware tables keep eye depth",
    'fog_vertex': "Compute fog per vertex and interpolate, as fixed-"
                  "function pipelines did -- visible banding on long "
                  "triangles, which is the point",
    'fog_spot': "How much a Model 3 screen spotlight brightens the fog "
                "inside its ellipse (Supermodel's spotFogColor x "
                "fogAttenuation): the lobe's colour is added to the fog "
                "target where the lamp shines. 0 leaves the fog untouched",
    'fog_bands': "Quantize the fog into this many hard depth steps -- "
                 "distance reads as flat painted planes, the anime "
                 "background trick (and the PS1's own coarse fog "
                 "tables). 0 keeps the fog smooth",
    'fog_height': "Limit fog to below a world height, thinning with "
                  "altitude -- valley mist instead of uniform soup. "
                  "Ignored under Ground Fog, which integrates its own "
                  "height law",
    'fog_height_top': "World height where height fog has fully thinned "
                      "to nothing",
    'fog_height_falloff': "How sharply height fog thins between its base "
                          "and top. Higher hugs the ground",
    # -------------------------------------------------------------- glow
    'glow': "Bloom around bright areas -- the soft television halo every "
            "period FMV had. Threshold picks what counts as bright",
    'glow_threshold': "Brightness above which a pixel feeds the glow. "
                      "Lower pulls midtones into the halo; higher keeps "
                      "glow to highlights",
    'glow_radius': "Size of the halo in pixels at output resolution",
    'glow_intensity': "Strength of the glow added back over the frame",
    'glow_quality': "Blur passes for the halo. More is smoother and "
                    "slower; low counts show the period's boxy bloom",
    'star_filter': "Cross-screen star spikes on bright points, the "
                   "camera-filter look pasted over renders of the era",
    'star_points': "How many spikes each star throws. 4 is the classic "
                   "cross-screen filter; 6 and 8 the denser gauzes",
    'star_length': "Length of the spikes in pixels at output resolution "
                   "-- longer reads as a heavier filter on the lens",
    'star_rotation': "Rotation of the whole star pattern, in degrees",
    'star_intensity': "Brightness of the star spikes relative to the "
                      "highlight that threw them",
    'lens_flare': "Draw a lens flare from the brightest light in frame -- "
                  "ghosts, streak and all. The single most period effect "
                  "there is",
    'flare_intensity': "Overall strength of the flare elements -- glow, "
                       "ghosts and streak all scale together from here",
    'flare_ghosts': "How many aperture ghosts march across the frame "
                    "opposite the light",
    'flare_streak': "Strength of the horizontal anamorphic streak",
    # ------------------------------------------------------------ colour
    'color_depth': "Bits per pixel of the delivered frame. 24 is "
                   "truecolor; 16 and below quantise with the exact "
                   "banding of the era's framebuffers; HAM modes "
                   "reproduce the Amiga's tricks precisely; CRY16 is the "
                   "Jaguar's intensity-plus-chroma-cell pixel, YJK the "
                   "MSX2+'s shared-chroma pixel",
    'palette_mode': "For palettised depths: use a fixed period palette, "
                    "or fit one to this frame the way the era's "
                    "converters did",
    'palette_size': "Number of palette entries when fitting: 256 is VGA, "
                    "16 is EGA territory",
    'palette_method': "How the fitted palette is chosen -- median cut is "
                      "the classic, octree the smoother",
    'palette_bits': "Precision of the palette registers or DAC the chosen "
                    "colours are snapped to before display: an Atari ST held "
                    "16 colours from 512 (3 bits per channel), an Amiga 32 "
                    "from 4096, the VGA DAC 6 bits. Applies to every palette "
                    "mode -- fixed, adaptive or custom -- so the per-pixel "
                    "search lands on colours the machine could hold; ignored "
                    "under Extra Half-Brite (12-bit by definition) and "
                    "attribute cells (fixed hardware colours)",
    'attribute_cells': "Store colour per character cell, not per pixel, "
                       "the way tile-based displays did: each cell may "
                       "hold only its machine's two colours (ZX "
                       "Spectrum, MSX1, C64 hires) or three plus one "
                       "screen-wide background (C64 multicolour), chosen "
                       "by least squared error over the cell's pixels "
                       "-- where two colours meet inside a cell the "
                       "attribute clash shows. Owns the colour stage: "
                       "Colour Depth and Palette are not consulted; an "
                       "ordered Dither perturbs the pixels before the "
                       "fit, error diffusion and Blue Noise are ignored "
                       "(printed once)",
    'scanline_palette': "Rewrite the palette registers every scanline "
                        "the way raster-timed software did: Spectrum 512 "
                        "reloads the Atari ST's 16 registers three times "
                        "per line (48 colours a line, 3 bits per "
                        "channel), Dynamic HiRes loads 16 Amiga colours "
                        "per line, Sliced HAM gives HAM6 a fresh "
                        "16-colour base each line. A median cut per "
                        "line, so it runs on the CPU and refuses the GPU "
                        "road by name",
    'dither': "The error-spreading pattern that sells low colour depths: "
              "ordered Bayer matrices in period sizes, or Floyd-"
              "Steinberg diffusion",
    'dither_strength': "How much of the quantisation error the dither "
                       "spreads. 1 is full correction; less shows more "
                       "banding on purpose",
    'exposure': "Linear brightness multiplier applied before the display "
                "curve. 1 is neutral",
    'gamma': "The display curve's exponent. 1 is neutral; below brightens "
             "midtones, above deepens them. Period CRTs lived near 2.2",
    'contrast': "S-curve strength around middle grey. 0 is neutral",
    'saturation': "Colour intensity. 1 is neutral, 0 is greyscale, above "
                  "1 pushes toward the era's oversaturated promos",
    'brightness': "Flat offset added to the frame after exposure. 0 is "
                  "neutral",
    'input_gamma_naive': "Treat texture pixels as already-linear the way "
                         "naive period software did, instead of "
                         "converting from sRGB. Changes every texture's "
                         "midtones; authentic, not correct",
    # R251 (post-palette C092 / C093): the 3D Studio / Max video-out pair
    'super_black': "Keep every pixel covered by geometry at or above the "
                   "threshold while uncovered background pixels stay at "
                   "0, so a luminance keyer can tell them apart -- 3D "
                   "Studio's and 3ds Max's Super Black for video keying. "
                   "Applied per pixel after the blend, so a high "
                   "threshold lifts anti-aliased edges (Max's own "
                   "caveat); not under the process pool this round (the "
                   "workers return pixels only)",
    'super_black_threshold': "The lowest 8-bit value (0-255) a "
                             "geometry-covered pixel may show under "
                             "Super Black; Max's default is 15, 0 "
                             "changes nothing",
    'video_color_check': "Test every display-referred pixel against the "
                         "composite video envelope in IRE (luma plus the "
                         "chroma subcarrier) and correct the illegal ones "
                         "the way 3D Studio and 3ds Max did: flag them "
                         "black, scale their luminance into range, or "
                         "scale their saturation toward grey. Runs after "
                         "the display transform, before any dither or "
                         "palette",
    'video_system': "Which composite envelope the legaliser tests "
                    "against: NTSC-M with its 7.5 IRE black setup and "
                    "92.5 IRE of picture, or PAL with black at 0 IRE and "
                    "100 of picture (100 IRE = 700 mV: PAL is specified "
                    "in millivolts, the IRE figure is a borrowing)",
    'video_ire_limit': "The peak the composite envelope may reach: 120 "
                       "IRE is the NTSC maximum 3ds Max tested against, "
                       "110 IRE the broadcast house rule After Effects "
                       "defaults to; the trough limit is -20 IRE either "
                       "way",
    # ---------------------------------------------------------- CRT & TV
    'crt': "Draw the frame as a CRT would show it: scanlines, phosphor "
           "mask, bloom and curvature, each with its own control",
    'crt_scanlines': "Darkened lines between the picture's rows. The "
                     "strength of the CRT's most recognisable artefact",
    'crt_mask': "The phosphor arrangement simulated: aperture grille, "
                "slot mask or shadow mask",
    'crt_mask_strength': "How visibly the phosphor mask dims the picture",
    'crt_bloom': "How much bright areas bleed into neighbouring "
                 "phosphors",
    'crt_curvature': "Bulge of the simulated glass. 0 is flat; more bows "
                     "the picture and its scanlines outward like a tube",
    'crt_vignette': "Darkening toward the tube's corners, the way a real "
                    "shadow-mask CRT fell off at its edges",
    'composite': "Push the frame through a simulated composite video "
                 "cable: colour bleed, ringing and dot crawl, the way "
                 "most people actually saw this era",
    'composite_bleed': "How far chroma smears horizontally -- the "
                       "rainbow fringing on sharp colour edges",
    'composite_ringing': "Ghost echoes after hard luma edges, from the "
                         "cable's bandwidth limit",
    'composite_dot_crawl': "The crawling checkerboard on coloured edges "
                           "where chroma and luma interfere",
    'interlace': "Render alternating fields with a one-frame comb offset "
                 "on motion -- broadcast video's signature",
    'jpeg_artifacts': "Recompress the frame as a period JPEG, 8x8 blocks "
                      "and all -- the look of every mid-90s CD-ROM still",
    'jpeg_quality': "The simulated JPEG quality. Lower is blockier",
    'jpeg_passes': "Recompression generations. Each pass compounds the "
                   "damage, as re-saved files did",
    'block_size': "Pixelate the frame into blocks of this size after "
                  "rendering. 1 is off; larger is chunkier",
    'pixel_grid': "Darken the seams between output pixels, as LCD "
                  "previews and some grabbers showed them",
    # ------------------------------------------------------------- misc
    'progressive': "Deliver the frame in coarse-to-fine passes while it "
                   "renders instead of row by row -- the era's preview "
                   "refinement, and a quicker first look now",
    'seed': "Random seed for every stochastic effect: soft shadows, AO, "
            "dither jitter. Same seed, same picture, every time",
    'render_wire': "Draw the mesh's edges over the finished shading -- "
                   "the hidden-line overlay of period modellers, at "
                   "render quality",
    'wire_mode': "Which edges the overlay inks: every triangle edge, "
                 "only marked/creased ones, or silhouette and folds",
    'wire_angle': "For angle-based wire modes: the crease angle in "
                  "degrees above which an edge is inked",
    'wire_color': "Colour the wireframe's edges are inked in, drawn over "
                  "whatever the surface renders beneath them",
    'wire_width': "Width of the inked edges, in output pixels",
    # R251 (RAST-B): C094, C122, C063, C055
    'aa_clamp_samples': "LightWave's Limit Dynamic Range: clip every "
                        "sample's colour at 1.0 before the anti-aliasing "
                        "filter, so an overbright edge blends between "
                        "displayable values instead of bleeding a halo. "
                        "Glow and flare then see a clipped frame",
    'aa_gamma_blend': "Blender 2.2x-2.41's OSA gamma: each anti-aliasing "
                      "sample is squared through the 400-entry table "
                      "before the filter and the sum square-rooted after, "
                      "so bright thin lines stay wide and dark ones pinch. "
                      "Only the sample blend changes; flat areas keep "
                      "their value to a table step",
    'wire_dot_distance': "Beyond this camera distance an object under the "
                         "Elite rule collapses to a single dot at its "
                         "centre, as Elite's ships did. 0 never",
    'beam_machine': "Which vector generator quantises the strokes: the "
                    "DVG's 4-bit intensity with bright end dots, the AVG's "
                    "3-bit doubled levels, or Star Wars' 8-bit STATZ. AVG "
                    "and Star Wars colour is 1 bit per channel, so a wire "
                    "colour below 0.5 on every channel draws nothing and "
                    "says so",
    'beam_sigma': "The phosphor spot's Gaussian sigma in pixels at 480 "
                  "lines (scaled with the frame); larger is a fatter, "
                  "softer beam",
    'outline': "Ink cartoon outlines over the frame, drawn from the "
               "renderer's own buffers -- silhouettes, material borders, "
               "depth breaks and creases, each its own toggle below. At "
               "high anti-aliasing the line smooths beautifully",
    'outline_color': "Colour of the ink. Black for classic cel work; try "
                     "a dark tone of the scene's palette for softer looks",
    'outline_width': "Thickness of the ink in INTERNAL pixels -- under "
                     "Supersample the delivered line is this divided by "
                     "the sample grid, so raise it for heavy AA",
    'outline_opacity': "How solidly the ink covers what is under it. 1 is "
                       "full ink; fractions tint instead of cover",
    'outline_objects': "Ink where one object ends and another begins -- "
                       "the silhouette lines",
    'outline_materials': "Ink where the material changes on a surface, "
                         "even inside one object",
    'outline_depth': "Ink where depth jumps -- edges a silhouette test "
                     "misses when an object overlaps itself",
    'outline_depth_threshold': "How large a depth jump counts, as a "
                               "fraction of the nearer distance. Smaller "
                               "inks more overlaps; larger keeps ink to "
                               "big steps",
    'outline_normals': "Ink creases: edges where the surface turns harder "
                       "than the angle below -- the cube's edges, a "
                       "cylinder's rim",
    'outline_normal_angle': "The crease angle in degrees. Smaller inks "
                            "gentler turns; 90 keeps ink to hard corners",
    'outline_over_sky': "Let the ink land on the background at object "
                        "silhouettes, not only on other surfaces",
    'outline_marked': "Draw the edges YOU marked as interior ink lines: "
                      "Freestyle edge marks, edges marked Sharp, and "
                      "creased edges all draw, hidden-line-removed "
                      "against the z-buffer -- the hand-inked interior "
                      "detail a screen-space edge detector cannot find",
    'ink_style': "How the line is drawn. Clean: a crisp even line, the "
                 "trace machine and the xerox -- the 80s cel. Brush: a "
                 "soft bleeding edge, ink from a brush on cel. Pencil: "
                 "several offset strokes with grain, a sketch. Any style "
                 "dial off its default moves the ink to the distance-field "
                 "road, which anti-aliases the line",
    'ink_reference_height': "0: line widths are internal pixels. N: widths "
                            "are true to an N-line frame -- a width-2 line "
                            "at 480 is a width-4 line at 960, so a 4K frame "
                            "keeps the same line, not a hairline",
    'ink_taper': "Thick near, thin far: the line's width follows distance "
                 "from the camera -- 1 doubles the nearest line and thins "
                 "the farthest to nothing, the brush inker's perspective. "
                 "0 keeps every line its set width",
    'ink_interior_scale': "Width of the interior lines -- creases, material "
                          "breaks, marked edges -- relative to the "
                          "silhouette. 0.5 is the classic thick-outer, "
                          "thin-inner drawing; 1 is even",
    'ink_weight_noise': "Thick-and-thin along the line, as the hand's "
                        "pressure on a brush: the width wanders by up to "
                        "this fraction of itself over Weight Scale pixels",
    'ink_weight_scale': "The screen distance, in pixels, over which the "
                        "Weight Noise wanders",
    'ink_shadow_side': "Thicker on the shadow side: the line grows by up "
                       "to this fraction where the surface turns away from "
                       "the key lamp -- the underlit line of comics and "
                       "some 80s anime",
    'ink_boil': "The hand-traced wobble: the finished line is displaced by "
                "a noise field up to this many pixels, and the field steps "
                "at Boil Rate -- so the line boils frame to frame the way "
                "cels traced by different hands did. 0 is still",
    'ink_boil_fps': "How many times a second the boil field changes: 12 is "
                    "on twos at 24 fps (the classic), 8 is on threes, 24 "
                    "boils every frame, 0 holds one static wobble",
    'ink_boil_scale': "The screen size, in pixels, of the boil's wobbles -- "
                      "small for a nervous line, large for a slow drift",
    'ink_pencil_strokes': "Pencil style: how many offset strokes make each "
                          "line -- 2 is a doubled line, 4 a scribble",
    'ink_pencil_spread': "Pencil style: how far, in pixels, the strokes "
                         "scatter from the true line",
    'ink_grain': "Per-pixel grain in the line's coverage -- the tooth of "
                 "paper under pencil, the dry brush. 0 is solid ink",
    'ink_color_mode': "Where the line's colour comes from: the ink colour, "
                      "the fill it outlines darkened (the self-coloured "
                      "line of hand-inked features and iro-trace anime), "
                      "or a gradient to Ink Colour 2",
    'ink_fill_darken': "From Fill: how much darker than the surface the "
                       "line is -- 0 is the fill colour itself, 1 is black",
    'ink_color2': "The Gradient line colour's far end -- lines blend "
                  "from Line Colour to this along the gradient's axis",
    # R231: the drawn line
    'ink_end_taper': "The brush lifts and lands: the line thins toward every "
                     "stroke end and junction -- where a contour stops, "
                     "meets another or vanishes behind an object -- to 15 "
                     "percent of its width at 1. Each end is found on the "
                     "line itself, so a closed silhouette keeps its weight",
    'ink_end_length': "How far from a stroke end the thinning reaches, in "
                      "pixels (True To Height scales it)",
    'ink_roughness': "Edge irregularity: three octaves of noise on the "
                     "line's width, in PATCHES -- a slow field decides "
                     "where the line gets rough, so it is occasionally "
                     "rough rather than uniformly hairy",
    'ink_roughness_scale': "The coarsest roughness scale in pixels; the two "
                           "finer octaves halve it twice",
    'ink_drift': "The hand drifts: the line's centre wanders in and out of "
                 "the true contour by up to this many pixels, on the Weight "
                 "Scale noise, the same on both sides of the line",
    'ink_gaps': "Dry-brush skips: the line breaks where a slow noise runs "
                "high, more readily where the stroke is already thin",
    'ink_texture': "What the body of the line is made of: solid ink, the "
                   "brush's streaks running along the stroke, or the "
                   "charcoal stick's soft bloom and paper tooth",
    'ink_texture_amount': "How strongly the texture works the lines over "
                          "-- 0 leaves solid ink, 1 is the full breakup",
    # R234: the inker's line (core/lines.py)
    'outline_form': "Form lines: ink the valleys of the surface's facing -- "
                    "the fold of a cheek, the crease of a concave form, the "
                    "places an artist draws a line where no edge is (image-"
                    "space suggestive contours). Interior lines, at the "
                    "Interior Scale",
    'outline_form_threshold': "How sharp a fold must be to earn a form line: "
                              "lower finds gentler folds (and more of them), "
                              "higher keeps only the deep creases",
    'outline_shadow': "Shadow lines: ink the terminator -- the line where the "
                      "key lamp's light stops on each object, at the Shadow "
                      "Level -- as a cel inker traced the painter's shadow "
                      "boundary. Interior lines, at the Interior Scale",
    'outline_shadow_level': "Where the light stops, as the surface's cosine to "
                            "the key lamp (0 is the geometric terminator, 0.1 "
                            "a little into the light). Shared by Shadow "
                            "Lines and the Light Weight",
    'outline_tone': "Tone lines: ink wherever the shaded frame's tone steps "
                    "-- the shadow's edge, a cast shadow's outline, a "
                    "painted highlight's rim, a texture's edge -- found by a "
                    "flow-guided difference of Gaussians along the local "
                    "edge direction, one pixel wide on the dark side. "
                    "Interior lines, at the Interior Scale",
    'outline_tone_threshold': "The smallest tone step (as a fraction of the "
                              "display range) that draws a tone line: lower "
                              "finds softer gradations, higher keeps only "
                              "the hard steps",
    'ink_isophote': "The inker's thick-and-thin by the light (the isophote "
                    "distance): each silhouette point walks inward along the "
                    "surface toward where the light returns, and the line is "
                    "as heavy as that walk is long -- heavy on a broad "
                    "shadowed flank, a quarter width on the lit side and on "
                    "thin features, heavier near than far. 0 is off, 1 the "
                    "full effect",
    'ink_isophote_range': "The walk, in pixels, that earns the full line "
                          "width (True To Height scales it): shorter makes "
                          "every shadowed edge heavy, longer reserves the "
                          "heavy line for the broadest forms",
    'ink_smooth': "Draw the line along a smoothed contour instead of the "
                  "pixel staircase: the width, in pixels, of the smoothing "
                  "-- a pen's curve on a low-polygon silhouette. Corners "
                  "are kept sharp. 0 follows the pixels exactly",
    'ink_pressure': "The pen bears down through a curve and pauses at a "
                    "corner: the line grows by up to one and a half widths "
                    "where the contour turns, a blob at every corner. 0 is "
                    "an even hand",
    'ink_overshoot': "Lines run past their ends, past the corner where two "
                     "meet and across the junction where a chain arrives, "
                     "by up to this many pixels, tapering -- the crossed "
                     "corners of a 1940s drawing. About half the ends "
                     "overshoot, each by its own hashed amount, fixed to "
                     "the point on the surface so animation holds",
    'ink_anchor': "Where the line's noises (weight, roughness, drift, gaps) "
                  "live: on the screen, as a cel traced over a still "
                  "camera, or on the surface under the line, so they "
                  "travel with the object and a moving camera does not "
                  "make them swim",
    # R236: the colour process
    'film_process': "The colour process the cel went through -- the camera's "
                    "black-and-white records through their filters, the "
                    "print's dyes with their own impurities, the silver key "
                    "and the registration of the dye layers. Three-strip / "
                    "successive exposure is the 1935-55 Technicolor cartoon; "
                    "Two-colour is Cinecolor and the 1930s two-strip. Runs "
                    "before the Film Stock grade, on the linear frame",
    'film_process_amount': "How much of the printed frame shows over the "
                           "render's own colour",
    'film_exposure': "The timer's printer light, in stops: 0 prints a white "
                     "cel clear; down prints the whole frame denser and "
                     "darker (the rich print), up thinner and paler, the "
                     "highlights running into the matrix's knee at clear",
    'film_gamma': "The contrast of the negative-and-print chain: the slope "
                  "of the straight line in log exposure. 1.5 is the "
                  "theatre's dense print in a dark room; 1.2-1.3 is that "
                  "print as a telecine shows it on a screen",
    'film_density': "The dyes' maximum density (a black's D-max): higher "
                    "prints deeper blacks and richer saturated colours; the "
                    "silver key adds to it",
    'film_filters': "The taking filters' sharpness: sharp-cutting filters "
                    "separate the three records further than the eye "
                    "separates the colours -- the three-strip's more-than-"
                    "life saturation -- while overlapping filters (below 0) "
                    "muddy them. A white cel exposes every record alike "
                    "whatever the setting",
    'film_dye_purity': "0 is the imbibition dyes as documented -- a cyan that "
                       "also eats some green and blue, a magenta that eats "
                       "red and blue, a yellow that eats green -- balanced "
                       "so greys stay neutral and only colours shift (reds "
                       "deep, greens toward cyan, blues toward purple); 1 is "
                       "the process's ideal dyes with no cross-talk. The "
                       "two-colour dyes' green absorptions are the process, "
                       "not an impurity, and stay",
    'film_key': "The silver key image printed from the green record under "
                "the dyes (the later three-strip prints): extra density in "
                "the shadows, deeper blacks, a harder look",
    'film_halation': "Light scattered in the negative's base: each record's "
                     "bright areas veil their surroundings in that record's "
                     "own colour, over Halation Radius pixels",
    'film_halation_radius': "How far, in pixels, the halation's veil reaches",
    'film_register': "The dye-transfer registration error, in pixels: every "
                     "dye layer but the first lands its own hair off, "
                     "independently per frame, so every edge -- the ink "
                     "line included -- carries a coloured fringe. The honest "
                     "misregistration of a printed cartoon",
    # R237: the print's wear
    'film_grain_size': "The emulsion's grains' width in pixels at 1080 lines "
                       "(scaled with the frame's height): grains finer than "
                       "a pixel average down and show less, a pixel or wider "
                       "each show whole",
    'film_grain_clump': "The share of the grain that lies in clumps three "
                        "grains wide -- a coarse, pushed stock; 0 is an even "
                        "fine grain",
    'film_grain_chroma': "How much each record's grain is its own: 0 is one "
                         "sheet of grain for every record (a black-and-white "
                         "negative's), 1 gives each record its own sheet -- "
                         "coloured grain, the way three separate negatives "
                         "print",
    'film_dust_size': "The specks' mean radius in pixels at 1080 lines, "
                      "scaled with the frame's height; each speck its own "
                      "size and its own ragged edge",
    'film_dust_negative': "The share of the dust that sat on the negative "
                          "at printing: those specks blocked the printer "
                          "light and print clear -- and on a three-strip "
                          "print in one record's missing colour",
    'film_dust_cel': "The share of the dust that sat on the cel or the "
                     "platen glass under the rostrum camera: dark grey "
                     "under the lights, and the same specks on every frame "
                     "photographed from the same cel (a hold)",
    'film_hairs': "Hairs caught in the projector gate: about this many in "
                  "the gate at any time (up to four), each anchored at the "
                  "aperture's edge and dancing for its run of frames",
    'film_hair_length': "A hair's length in pixels at 1080 lines, scaled "
                        "with the frame's height; each hair its own length "
                        "around it",
    'film_hair_width': "A hair's width in pixels at 1080 lines, scaled with "
                       "the frame's height",
    'film_hair_hold': "About how many frames a hair stays in the gate before "
                      "it is gone; each hair its own run around it",
    'film_scratches': "Scratches the transport left on the print: about this "
                      "many running at any time (up to three), each the "
                      "length of the frame at one place with a slow wander",
    'film_scratch_width': "A scratch's width in pixels at 1080 lines, scaled "
                          "with the frame's height",
    'film_scratch_hold': "About how many frames a scratch runs before it "
                         "ends; each scratch its own run around it",
    'film_scratch_side': "Which side of the film the scratches are on: the "
                         "emulsion side takes the dye away (a bright line -- "
                         "neutral on a dye-transfer print, one dye's colour "
                         "on a two-colour print, blue on a chromogenic one), "
                         "the base side scatters the lamp (a dark line)",
    'film_reel': "The reel's length in minutes, for the projectionist's cue "
                 "marks: a scraped circle top right, four frames long, eight "
                 "seconds before every reel's end (the motor cue) and again "
                 "one second before it (the changeover). 0 is no cue marks",
    # R230: the era looks
    'film_grade': "The film stock the cel was photographed on, as a colour "
                  "response in linear light: Technicolor's purified "
                  "primaries and cyan shadows, a faded 70s Eastmancolor "
                  "print, the 80s telecine, a VHS dub, or black-and-white "
                  "panchromatic negative. Each entry's tooltip says what "
                  "it does",
    'film_grade_amount': "How much of the stock's response is applied: 1 "
                         "the full grade, 0 none",
    'film_softness': "The rostrum camera and optical printer's softness: "
                     "a Gaussian of this many pixels (sigma) over the "
                     "whole frame, before the print's dust and grain",
    'film_weave': "Gate weave: the whole frame shifts by up to this many "
                  "pixels per frame -- a slow wander plus a per-frame "
                  "jitter, decided by the frame number and the seed, so "
                  "a frame always weaves the same way",
    'film_dust': "Dust, dirt and the odd hair on the print: a density -- "
                 "1 is about forty specks on a 1080p frame, scaling with "
                 "the frame's area -- placed by the frame's own hash so "
                 "the sequence twinkles as a projected print does",
    'film_grain': "Emulsion grain: a per-pixel, per-frame noise, seven "
                  "parts luminance to three parts colour, multiplying the "
                  "linear frame. The same frame is always the same sheet",
    'film_flicker': "Projector flicker: the frame's exposure wanders by up "
                    "to 12 percent at 1, per frame, by the frame's hash",
    'film_hold': "Shoot on ones, twos or threes: 2 photographs every "
                 "drawing for two frames, 3 for three. A held frame is "
                 "not rendered again -- the key frame's picture is "
                 "photographed again, film stages fresh (grain, dust, "
                 "weave still move) -- so an animation on twos renders in "
                 "half the time. Render the sequence from its start: a "
                 "frame whose key frame was not rendered in this session "
                 "renders fresh and says so",
    'film_halftone': "Ben-Day / newsprint dots on the finished frame: four "
                     "ink screens at the classic angles (cyan 15, magenta "
                     "75, yellow 0, black 45 degrees), dots sized by each "
                     "ink's coverage at the cell centre, laid "
                     "subtractively. 1 is the full print, 0 none",
    'film_halftone_pitch': "The dot screen's cell size in output pixels -- "
                           "the distance between dot centres",
    'film_misregister': "Paint misregistration: the painted colour slides "
                        "by up to this many pixels per frame (by the "
                        "frame's hash) BEFORE the ink lines are drawn, so "
                        "the lines land where the drawing put them over "
                        "paint that missed. A stylisation -- a real cel's "
                        "paint sat behind its own ink; the printed "
                        "cartoon's honest misregistration is the Colour "
                        "Process's Register",
    'film_bleed': "Paint bleed: the painted colour softens by this many "
                  "pixels (sigma) before the ink lines are drawn, the paint "
                  "soaking under and past the line; the lines stay crisp",
    # R233: the painted background road
    'bg_paint': "The painted background: how much of a Background "
                "material's lit colour is laid down as brush strokes -- "
                "particles fixed on the surface, drawn far to near as "
                "strokes of a screen-constant size over an abstracted "
                "base (Meier 1996). 0 leaves the surface as rendered",
    'bg_stroke_size': "The brush's width in output pixels; strokes keep "
                      "this size at any distance, the way a painter's "
                      "brush does, and there are about as many as the "
                      "frame needs at this spacing",
    'bg_stroke_length': "How long a stroke is, as a multiple of its width",
    'bg_direction': "Which way the strokes run: along the colour's "
                    "contours, along the surface's silhouette, or at a "
                    "fixed angle",
    'bg_angle': "The stroke angle, degrees off horizontal -- the Fixed "
                "Angle direction, and the fallback where a contour or a "
                "silhouette has no direction",
    'bg_spread': "How far each stroke may turn from its direction, up to "
                 "forty-five degrees, by its own hash",
    'bg_bristles': "Bristle streaks across each stroke, lighter and darker "
                   "lines along its length",
    'bg_variation': "How much each stroke lightens or darkens its colour "
                    "by its own hash -- the impasto's value jitter",
    'bg_smooth': "The abstracted base the strokes are painted over: a "
                 "Gaussian of this many pixels on the Background pixels "
                 "only (a cel beside them never bleeds in). It shows "
                 "between strokes",
    'bg_paper': "The board's tooth over the painting, in screen space",
    'setback': "The Fleischer setback: the lens on the miniature set behind "
               "the cel. The Background materials (and the sky) soften "
               "by eye distance up to this many pixels (sigma) while the "
               "cels stay sharp. 0 is off",
    'setback_start': "The eye distance at which the setback's softness "
                     "begins, in scene units",
    'setback_range': "The distance over which the softness climbs from "
                     "nothing to Setback",
    'setback_sky': "Soften the sky at the setback's far end too -- the "
                   "painted backdrop behind the miniature",
    'ink_gradient': "What the Gradient runs along: distance from the "
                    "camera, the height of the frame, or the key lamp's "
                    "lit-to-shadow turn",
    'pass_depth': "Also deliver a Depth pass: each pixel's distance from "
                  "the camera, for compositing",
    'pass_normal': "Also deliver a Normal pass: the shading normal per "
                   "pixel, for relighting tricks",
    'pass_position': "Also deliver a world Position pass per pixel",
    'pass_uv': "Also deliver the UV coordinates per pixel",
    'pass_object_index': "Also deliver each pixel's object index, for "
                         "per-object masks downstream",
    'pass_material_index': "Also deliver each pixel's material index, "
                           "for per-material masks downstream",
    'debug_pass': "Replace the delivered image with one internal buffer "
                  "-- depth, normals, UVs, overdraw, wireframe -- to see "
                  "what the renderer sees",
    # ------------------------------------------------- layered rendering
    'layer_gpu_min_frac': "Depth layers whose pixel share is below this "
                          "fraction shade on the CPU rather than paying a "
                          "GPU pass's fixed costs. 0 sends everything to "
                          "the GPU, 1 keeps every layer on the CPU",
    # ------------------------------------------------------ lens effects
    'lens_vignette_edges': "Darkening toward the frame's corners from the "
                           "simulated lens itself, independent of the CRT "
                           "vignette",
    'shaft_decay': "How quickly volumetric light shafts fade along their "
                   "length. Higher dies faster",
    'shaft_samples': "March steps per pixel for light shafts. More is "
                     "smoother and slower",
    'dof_max_radius': "Cap on the depth-of-field blur circle, in pixels. "
                      "Keeps extreme defocus affordable",
    'radiosity': "One bounce of gathered colour bleed -- the era's "
                 "Radiosity checkbox. Rays that see sky return the ambient "
                 "colour; rays that land on a surface return its flat "
                 "diffuse. Supersedes plain Ambient Occlusion while on",
    'radiosity_samples': "Hemisphere rays gathered per pixel. 8 is the "
                         "period look; more is smoother and slower",
    'radiosity_distance': "How far a gather ray reaches before it counts "
                          "as seeing sky",
    'radiosity_intensity': "Strength of the colour a surface lends its "
                           "neighbours",
    'radiosity_spacing': "Gather every Nth pixel and blend between the "
                         "points -- the interpolated mode the era "
                         "actually shipped. 1 gathers every pixel; 2 "
                         "casts a quarter of the rays, 4 a sixteenth",
    'reflection_blur': "Cone angle in degrees for blurry (glossy) ray "
                       "reflections. 0 keeps mirrors sharp. Blurry frames "
                       "shade on the CPU -- the deferred pass traces one "
                       "ray per fragment",
    'reflection_blur_samples': "Jittered rays averaged per reflective "
                               "fragment when Reflection Blur is above 0",
    'watermark': "Burn a line of text into the corner of the final frame, "
                 "the VTR way. Tokens: %F frame number, %R resolution, "
                 "%V engine version, %D date, %T time of day, %S render "
                 "time, %B Blender version. Ink colours: &%r red, "
                 "&%g green, &%b blue, &%y yellow, &%c cyan, &%m magenta, "
                 "&%w white again -- each colours everything after it. "
                 "Empty means no burn-in",
    'viewport_gpu': "Let the viewport preview use the GPU device (when the "
                    "top switch is GPU). Turn OFF to force every viewport "
                    "frame onto the CPU while F12 keeps the driver -- the "
                    "bisect switch for viewport-only driver problems: if a "
                    "glitch follows this toggle, it lives in the viewport's "
                    "GPU path; if it stays, it never did",

    'spot_cones': "Draw the visible beam of every spot light whose Volumetric "
                  "value is above zero. The view ray is intersected with the "
                  "cone and a few samples are summed along whatever falls "
                  "inside it -- which is what LightWave and 3D Studio did, "
                  "rather than integrating a volume",
    'spot_cone_samples': "Samples along each view ray. Low counts band, and "
                         "the banding is the period artefact rather than a "
                         "defect -- 8 to 16 is where those renderers sat",
    'spot_cone_density': "Overall strength of every cone. Each light's own "
                         "Volumetric value scales it further",
    'spot_cone_falloff': "How fast scattering fades with distance from the "
                         "lamp. 2 is inverse-square; lower carries further",
    'spot_cone_reach': "How far a beam is drawn when nothing stops it",
    'volume_steps': "March samples through every VOLUME CONTAINER -- a "
                    "mesh whose material output links a Volume chain "
                    "(Halcyon Volume, Principled Volume, Volume "
                    "Scatter/Absorption). More steps resolve finer "
                    "density detail; low counts band, and the banding "
                    "is the era's own slicing artefact",
    'volume_shadows': "Let lamps be shadowed INSIDE volume containers, "
                      "through each lamp's own shadow road -- what "
                      "carves god-rays and cloud self-shading out of "
                      "the scattered light. Off is faster and evenly "
                      "lit",
    'threads': "How many threads share the shading work. Measured on a 20-core "
               "machine this is neutral at best and about 3% slower at worst, "
               "because NumPy releases the interpreter lock only for large "
               "array operations and the node evaluator is dominated by Python "
               "dispatch between small ones. Defaults to 1 for that reason. "
               "Worker Processes is the route that actually parallelises",
    'render_device': "Where the frame is computed. GPU rasterises the frame, "
                     "shades it, draws the sky in the same burst for the "
                     "simple and HDRI worlds, keeps the frame on the GPU "
                     "for the ink, the supersample resolve and the post "
                     "chain, and runs that chain resident -- each stage "
                     "measured against the CPU frame on real hardware, and "
                     "each falling back per frame with the reason on the "
                     "console when a scene uses something still CPU-only: "
                     "a caster mask on ray shadows, the Bryce and painted "
                     "skies, world node graphs beyond a plain Background "
                     "node, the ground plane, the film beyond grain, the "
                     "dithers and palettes. Flipping to GPU turns the "
                     "proven stages on; the Debug panel can switch the "
                     "rasteriser and the shading off individually (GPU "
                     "Post follows the device switch)",
    'gpu_post': "Run the post chain on the GPU through Blender's own gpu "
                "module, the layer EEVEE is built on -- resident since 1.89: "
                "the frame the render left on the GPU is inherited (no "
                "upload; otherwise one, plus one after each CPU-only stage "
                "that hands it back), the film grain and flicker, the display "
                "transform, the bit depth, NTSC and CRT draw stage to stage "
                "in place, and the picture reads back once at the end. Each "
                "stage is measured against the CPU stage on real hardware; "
                "a CPU-only stage that is on (a dither or palette, halftone, "
                "interlace, lens, JPEG, the optical stages, the film beyond "
                "grain) reads the frame back by name and says so on the "
                "console",
    'gpu_hold_context': "Give the render thread the GPU context for the "
                        "whole frame, as before 1.25.53. Bursts start "
                        "instantly but Blender's interface cannot redraw "
                        "until the frame ends (Not Responding on long "
                        "renders). Off, the interface stays live and each "
                        "GPU burst runs on the main thread instead. Takes "
                        "effect from the next render",
    'gpu_scissor': "Limit each transparent depth layer's GPU passes and "
                   "readbacks to the layer's own bounding box. The same "
                   "pixels shade either way -- the self test proves the "
                   "two paths identical on your driver -- but a sparse "
                   "layer stops paying for the whole frame. Turn off only "
                   "if a driver disagrees with scissored reads; the "
                   "picture is then the proven full-frame path",
    'gpu_shading': "Shade the frame on the GPU: the G-buffer is shaded in "
                   "one full-screen pass per material -- shadow maps, image "
                   "textures, converted master-shader materials, normal-map "
                   "chains, coded shader nodes, the period pattern "
                   "textures, matcap, backface and environment reflections "
                   "all included, with sun, point, spot and area lights -- "
                   "and the sky or background for the simple and HDRI "
                   "worlds is drawn in the same burst, so the readback is "
                   "the whole frame, which then stays on the GPU for the "
                   "ink, the supersample resolve and the post chain. "
                   "Measured against the CPU frame at 0.00002 max "
                   "difference on real hardware. Frames using what the GLSL "
                   "does not reproduce yet -- a caster mask on ray shadows, "
                   "a Bump node inside a transparency layer, per-pixel "
                   "opacity under Screen Door, the refusals the console "
                   "names -- shade on the CPU with the reason on the "
                   "console; a fog frame shades on the GPU and takes the "
                   "CPU's own fog over the readback; a Bryce or painted "
                   "sky, a starfield or physical sky, a world node graph "
                   "beyond a plain Background node, or the ground plane "
                   "keeps the sky on the CPU and says so. Shader compiles, "
                   "plans and "
                   "unchanged uploads are all cached across frames",
    'gpu_raster': "Rasterise the frame on the GPU too: a compute-shader "
                  "port of the CPU rasteriser's own fill rules, one thread "
                  "per pixel, measured at ZERO differing pixels against the "
                  "CPU on real hardware and several times faster at "
                  "working sizes. Frames it cannot reproduce yet -- "
                  "affine texture mode, overdraw "
                  "debugging, banded worker renders -- rasterise on the "
                  "CPU with the reason on the console",
    'lens_distortion': "Barrel distortion below zero, pincushion above. What a "
                       "cheap lens does to straight lines",
    'chromatic_aberration': "Splits the colour channels radially, because a "
                            "real lens does not focus red and blue in the same "
                            "place. The clearest tell that an image went "
                            "through glass",
    'dof': "Defocus by splitting the frame into depth slabs and blurring each. "
           "What compositors of the era did, at a handful of blurs rather than "
           "hundreds of samples",
    'dof_focus': "Distance from the camera that stays sharp",
    'dof_method': "How depth of field is made: the finished frame blurred "
                  "in depth slabs (Layered Post Blur), or the frame rendered "
                  "once per lens point and averaged (Lens Passes -- REYES / "
                  "the SGI accumulation buffer / 3ds Max multi-pass DOF)",
    'dof_lens_pattern': "Where the lens points sit across the aperture: a "
                        "Halton disc (the accumulation buffer paper's "
                        "sequence) or 3ds Max's multi-pass spiral with pass 0 "
                        "at the lens centre",
    'dof_lens_samples': "Lens points, one full render each: the accumulation "
                        "buffer paper's example used 23, 3ds Max's Total Passes "
                        "default is 12. Few passes show as rings, which is the "
                        "era's own look",
    'dof_amount': "How quickly things go soft either side of the focus distance",
    'dof_layers': "Number of depth slabs. More is smoother and slower",
    'shaft_threshold': "How bright a pixel must be to throw a shaft",
    'shaft_length': "How far the streaks reach toward the light",
    # R251 C119 (MAT-B)
    'shading_rate_area': "REYES ShadingRate: shade on a surface-aligned grid "
                         "whose cells cover about this many pixels of the "
                         "OUTPUT frame's area, one shade per cell (constant "
                         "interpolation) -- the micropolygon facet of Pixar's "
                         "renderer, PRMan and BMRT. 0 shades every pixel; 1 is "
                         "PRMan's default; 4-16 give the preview-render look. "
                         "Inert at the VERTEX / FACE rates (they shade coarser "
                         "than any grid); both roads snap the same cells",
    'displacement_scale': "Strength of the bump derived from a material's "
                          "Displacement output. Geometry is not tessellated -- "
                          "the height becomes a normal perturbation, which is "
                          "what scanline renderers of the era did",
    'painters_key': "Which point on a polygon decides its sort order. Changing "
                    "it moves where Painter's algorithm goes wrong",
    'aa_edge_threshold': "Depth break, in scene units, that counts as a "
                         "crease worth extra samples. Both edge modes always "
                         "treat silhouettes; this adds interior breaks. Edge "
                         "Only tests the step between neighbours; Adaptive "
                         "tests the curvature, so a smooth floor running "
                         "away from the camera is never mistaken for one "
                         "long crease. 0 treats silhouettes only",
    'ray_shadows': "Master switch for ray-traced shadows. Off, lights set to "
                   "trace (and lights with no shadow map to fall back on) "
                   "cast no shadow at all",
    'max_transparent_layers': "How many overlapping transparent surfaces a "
                              "pixel may accumulate. Beyond a handful the "
                              "furthest ones contribute almost nothing but cost "
                              "as much as the first. 0 means no limit",
    'fast_background': "Evaluate the world once per output pixel instead of "
                       "once per supersample. A sky is smooth almost "
                       "everywhere, and at 4x this is sixteen times less work "
                       "for the background. Turn it off if a sharp sun disc or "
                       "a detailed HDRI shows aliasing",
    'show_stats': "Print a per-stage timing breakdown to the system console "
                  "after each frame, so it is clear where the time actually "
                  "went rather than where it is assumed to go",
    'cache_shadows': "Reuse shadow maps while the lights and geometry hold "
                     "still. Each map is a full rasterisation pass and a point "
                     "light needs six, so this is most of the saving on an "
                     "animation with static lighting",
    'use_processes': "Split each frame across separate Python processes. "
                     "Threads only run in parallel where NumPy releases the "
                     "interpreter lock; processes have no shared lock at all. "
                     "Falls back to rendering in Blender if workers cannot "
                     "start, and the reason is printed to the console",
    'process_count': "How many worker processes to start. 0 uses one per core",
    'dither_serpentine': "Alternate the scan direction each row, which hides the "
                         "directional worming error diffusion can produce. "
                         "Turning it off is about twice as fast, because the "
                         "diffusion can then be processed a diagonal at a time "
                         "instead of a pixel at a time",
    'palette_lock': "Build the adaptive palette once and reuse it. Stops the "
                    "colours crawling between frames of an animation, and skips "
                    "rebuilding the palette on every frame",
    'film_transparent': "Render the background with zero alpha so it can be "
                        "composited. Off means an opaque background. Blender's "
                        "own Film > Transparent also switches this on",
    'preview_scale': "Viewport preview is rendered at 1/N resolution and scaled "
                     "up. Raise it for a faster, chunkier preview",
    'orbit_scale': "Pixel size for MOTION frames only -- orbiting, panning, "
                   "edits mid-stream. Auto (0) keeps motion frames under a "
                   "fixed pixel budget however large the region; a number "
                   "renders them at exactly 1/N. The resting refine always "
                   "uses Pixel Size",
    'material_override': "Render every material as a plain matte surface for "
                         "test renders -- geometry, lights and shadows stay "
                         "real, textures and shading graphs are ignored",
    'override_color': "The matte colour Material Override renders with",
    'output_scale': "Render at 1/N of the output resolution and scale back up "
                    "with nearest-neighbour. The output stays the size you set, "
                    "and the render costs N squared times less, or resample "
                    "the way the 3DO's display generator and the GBA's affine "
                    "background did (the 3DO and GBA items)",
    # R251 post-signal: the machine's scan-out stages
    'vi_dither_filter': "The N64 Video Interface's DITHER_FILTER_ENABLE: at "
                        "scan-out each 5-bit channel steps toward each of its "
                        "eight neighbours by one 8-bit level, so the RDP's "
                        "ordered dither melts into 8-bit tone. Reads the N64's "
                        "15-bit (5551) frame; on a 5:6:5 frame the 6-bit green "
                        "steps on its own 8-bit expansion, an extension of the "
                        "VI's rule",
    'vi_gamma': "The N64 Video Interface's gamma stage: an integer square-root "
                "curve (2*isqrt(64*v)) with or without six random bits under "
                "the root (GAMMA_DITHER_ENABLE), or the single-bit dither "
                "alone. Replaces the display Gamma dial for an N64 look",
    'copy_filter': "The GameCube and Wii EFB-to-XFB copy filter: a three-line "
                   "vertical blur in 1/64 units run on every frame before the "
                   "video encoder, the SDK's deflicker set or the antialiased "
                   "modes' set; the 1/64 divide floors, a choice (Dolphin "
                   "computes in float and lets the store round; the hardware's "
                   "rounding is undocumented)",
    'crtc_blend': "The PlayStation 2 CRTC's PMODE mix: the frame blended per "
                  "pixel with the previous frame's output (a persistence "
                  "trail) or with the BGCOLOR register, as "
                  "(cur*A + other*(255-A))/255 in integers",
    'crtc_alpha': "The PMODE ALP register, 0..255: how much of the current "
                  "frame reaches the screen; 255 shows only the current frame, "
                  "0 only RC2",
    'crtc_bg_color': "The GS BGCOLOR register: what RC2 shows in Background "
                     "mode, and what a frame with no rendered predecessor "
                     "blends against",
    'video_filter': "The 3dfx Voodoo's filter between the 5:6:5 framebuffer "
                    "and the RAMDAC that dissolved its ordered dither into "
                    "gradients (the '22-bit colour' of 1997); 86Box's "
                    "reconstructions of the Voodoo Graphics four-pass line and "
                    "the Voodoo2 look-ahead. At the frame's border columns a "
                    "missing neighbour counts as the pixel itself (the pack's "
                    "rule; the emulator leaves its edge columns partly "
                    "filtered). Reads a 16-bit frame",
    'video_filter_threshold': "The driver's SST_VIDEO_FILTER_THRESHOLD (86Box "
                              "scrfilterThreshold), one cap for all three "
                              "channels (86Box carries a byte per channel out "
                              "of one 24-bit value): on the Voodoo Graphics a "
                              "difference is halved as if it were at most the "
                              "threshold, so an edge moves by at most half the "
                              "threshold per pass; on the Voodoo2 a difference "
                              "beyond the threshold moves nothing. 0 disables "
                              "the passes. Calibrate against a capture",
    # R251 SIG-2
    'chroma_format': "How the digital tape, disc or console output stored "
                     "colour: Y'CbCr at 8-bit legal levels with chroma "
                     "sampled at the format's rate and SITE, the way D1, DV, "
                     "Video CD, DVD and the GameCube XFB did. The file "
                     "formats run before the display's own quantiser; the "
                     "XFB runs after the copy filter",
    'chroma_upsample': "How the decoder rebuilt full-resolution chroma from "
                       "the sites: held (cheap decoders, visible chroma "
                       "blocking) or linearly interpolated (better ones)",
    'signal': "The cable between the machine and the television: S-Video "
              "band-limits the encoded chroma only (no crawl; the NTSC item "
              "models a wideband I/Q decoder on the encoder's own bands); "
              "RF adds the modulator's luma low-pass, the 920 kHz "
              "sound-chroma beat, snow and a multipath ghost after the "
              "composite cable. Composite itself is the Composite Video "
              "switch; RGB and component carry nothing per pixel",
    'rf_bandwidth': "The RF modulator's video bandwidth in MHz, a horizontal "
                    "luma low-pass; cheap modulators sat near 3 MHz, the NTSC "
                    "limit is 4.2",
    'rf_beat': "The 920 kHz herringbone: the 4.5 MHz sound carrier beating "
               "with the 3.58 MHz colour subcarrier, added to luma in "
               "proportion to chroma amplitude",
    'rf_snow': "Thermal noise on the whole RF signal as a fraction of white: "
               "luma grain and colour speckle, hashed per pixel and frame",
    'rf_ghost': "A multipath ghost: the picture added back at this strength, "
                "displaced to the right by the ghost delay",
    'rf_ghost_delay': "The ghost's delay in microseconds along the 52.66 us "
                      "NTSC active line (1 us is about 12 pixels at 640)",
    'pal_decoder': "The PAL television's chroma decoder: the delay-line set "
                   "that averages two lines of colour, or the simple set that "
                   "shows phase error as Hanover bars. The NTSC composite "
                   "stage stays the cable; this is the receiver",
    'pal_phase_error': "Differential phase error in degrees; a simple PAL "
                       "decoder turns it into Hanover bars (+phi on even "
                       "lines, -phi on odd), a delay-line decoder cancels it. "
                       "Read under the Simple decoder only",
    'pal_crawl': "PAL dot crawl: the 4.43 MHz subcarrier left in luma, "
                 "advancing three quarters of a cycle per line and repeating "
                 "over eight fields, in proportion to chroma amplitude",
    'tape': "The analogue tape format the frame was recorded on: its luma "
            "FM bandwidth (TVL) as a horizontal low-pass, colour-under "
            "chroma band-limited and delayed to the right, the head-switch "
            "tear, dropouts, and every dub compounding all of it. Runs "
            "before the cable",
    'tape_generations': "How many dubs deep: the tape's low-pass is applied "
                        "once per generation (sigma times the square root of "
                        "N) and its noise grows with the square root of N; 1 "
                        "is a first-generation recording",
    'tape_noise': "The FM demodulation noise on the luma as a fraction of "
                  "white (chroma at half), hashed per pixel and frame; grows "
                  "with the square root of the generation count",
    'tape_head_switch': "The head-switch tear: the bottom seven lines (at "
                        "480) of the overscanned picture torn sideways by a "
                        "few pixels, re-hashed per frame, with four times the "
                        "luma noise",
    'tape_dropouts': "Expected tape dropouts per frame: bright horizontal "
                     "dashes 8 to 48 pixels long at hashed rows and columns, "
                     "the oxide the heads could not read",
    # R251 SIG-3: the codecs and the optical printer
    'mpeg1': "The JPEG stage's cousin with MPEG-1's own rules: recode the "
             "frame as Video CD intra blocks -- 8x8 DCT in 4:2:0 with the "
             "standard's intra matrix times a linear quantiser scale, an "
             "8-bit DC in steps of 8, the mismatch-control oddification "
             "and the GOP pumping the scale per frame. One codec at a "
             "time: JPEG Artefacts are greyed while this is on",
    'mpeg1_qscale': "quantizer_scale for I frames, 1..31: the linear "
                    "multiplier on MPEG-1's intra matrix; Video CD "
                    "encoders sat around 8-12 at 1.15 Mbit/s",
    'mpeg1_gop': "Frames per group of pictures (Video CD: 15, with every "
                 "third a P frame); B frames take 1.4 times the I scale "
                 "(MPEG Test Model 5), so quality pumps through the "
                 "group. 0 codes every frame as an I frame",
    'smacker': "Recode the frame as Smacker blocks (RAD Game Tools, 1994): "
               "a 256-colour palette fitted to the frame, then every 4x4 "
               "block filled with one colour, split into two (the mono "
               "block with its bit mask) or kept at sixteen, by the "
               "encoder's error budget",
    'smacker_quality': "The Smacker encoder's budget: low values fill and "
                       "mono-code more blocks (the chunky cutscene), 1 "
                       "keeps nearly every block at full sixteen colours. "
                       "Halcyon's own thresholds, the codec's block grammar",
    'matte_glow': "The optical printer's backlit matte (Tron, 1982): every "
                  "material with a Glow Gel colour is a clear-or-opaque "
                  "Kodalith matte exposed in its gel's colour, once crisp "
                  "and once per diffusion pass at doubling radius, summed "
                  "on the negative before the film stages. An F12 stage: "
                  "the viewport draws the frame without it",
    'matte_glow_radius': "The first diffusion pass's blur in pixels at 1080 "
                         "lines of the Tron printer stage; each further "
                         "pass doubles it and halves its exposure",
    'matte_glow_passes': "How many diffusion passes each Kodalith matte is "
                         "exposed through, at radius x1, x2, x4... and "
                         "exposure 1/2, 1/4, 1/8...",
    'matte_glow_exposure': "The gel's exposure in the Tron printer stage: "
                           "what one crisp pass adds; the diffusion "
                           "passes add half, a quarter... of it",
    'vertex_snap': "Round transformed vertices to a pixel grid, as the "
                   "PlayStation's integer GTE did",
    'tex_perspective': "Turn off for the affine texture warp of hardware with "
                       "no perspective divide per texel",
    'specular_viewer': "The view vector the reflectance models use: the true "
                       "per-pixel eye direction, or one fixed camera axis for "
                       "the whole frame -- OpenGL 1.1's default infinite "
                       "viewer, the Sega Model boards' R.z and the DS's "
                       "(0,0,-1) line of sight -- so highlights sit still "
                       "under perspective. Every model's own view term takes "
                       "the axis (a Cook-Torrance or Strauss Fresnel "
                       "included: that is the infinite viewer); the Master "
                       "shader's rim, Fresnel and sheen cheats and the "
                       "two-sided flip keep the true eye",
    'specular_in_gamma': "Compute highlights in display space. 1990s renderers "
                         "had no linear workflow, and the blown-out speculars "
                         "are a large part of the look",
    'crand_per_frame': "Fold the frame number into the crand hash so the grain "
                       "re-rolls every frame, reproducing POV-Ray's "
                       "sequential-random flicker on purpose; off, the same "
                       "frame always renders the same grain (the Master "
                       "shader's Crand socket draws the grain)",
    'color_management': "Leave at None for period-correct output. Blender's own "
                        "view transform is bypassed by this engine",
    'depth_sort': "Painter's algorithm reproduces the sorting errors of "
                  "hardware without a depth buffer",
    'max_lights': "Hardware light limit. Lights beyond this are dropped, as on "
                  "fixed-function pipelines. Zero means unlimited",
}

COLOR_FIELDS = {'global_ambient', 'fog_color', 'wire_color',
                'override_color', 'ink_color2'}


#: dataclass fields that are DERIVED at export rather than edited as
#: properties -- each one's UI lives elsewhere (palette_colors is read
#: out of the picked Palette Image; a generated FloatVector of size 0
#: would not even register)
_DERIVED_FIELDS = {'palette_colors'}


def _build():
    """Turn the dataclass into a dict of bpy properties."""
    import dataclasses
    props = {}
    for f in dataclasses.fields(RenderSettings):
        name = f.name
        if name in _DERIVED_FIELDS:
            continue
        default = f.default
        label = LABELS.get(name, name.replace('_', ' ').title())
        desc = DESCRIPTIONS.get(name, '')
        if name in ENUMS:
            extra = {}
            if name == 'render_device':
                # the top-of-panel switch MEANS it: flipping to GPU turns
                # the proven stages on, so a scene saved back when they
                # defaulted off cannot silently render a 10-second CPU
                # frame under a switch that says GPU. Flipping to CPU
                # leaves them alone; the Debug toggles still opt out
                def _device_flip(self, _context):
                    if str(self.render_device).upper() == 'GPU':
                        self.gpu_raster = True
                        self.gpu_shading = True
                        self.gpu_post = True
                extra['update'] = _device_flip
            props[name] = EnumProperty(name=label, description=desc,
                                       items=ENUMS[name],
                                       default=default if any(
                                           i[0] == default for i in ENUMS[name])
                                       else ENUMS[name][0][0], **extra)
        elif isinstance(default, bool):
            props[name] = BoolProperty(name=label, description=desc,
                                       default=default)
        elif isinstance(default, int):
            lo, hi = RANGES.get(name, (0, 2 ** 31 - 1))
            props[name] = IntProperty(name=label, description=desc,
                                      default=default, min=int(lo), max=int(hi),
                                      soft_min=int(lo), soft_max=int(hi))
        elif isinstance(default, float):
            lo, hi = RANGES.get(name, (-1e6, 1e6))
            props[name] = FloatProperty(name=label, description=desc,
                                        default=default, min=lo, max=hi,
                                        soft_min=lo, soft_max=hi)
        elif isinstance(default, tuple) or name in COLOR_FIELDS:
            props[name] = FloatVectorProperty(
                name=label, description=desc, subtype='COLOR', size=3,
                default=tuple(default), min=0.0, max=1.0)
        else:
            props[name] = StringProperty(name=label, description=desc,
                                         default=str(default or ''))
    return props


# The preset list is import-time static, so the enum is too. It used to be
# a dynamic callback, and its first entry is a category HEADER ('', ...):
# a dynamic enum's unset value is index 0, which resolved to the header's
# empty identifier and made Blender log "current value '0' matches no enum"
# on EVERY redraw of the presets panel -- the field's console flood. A
# static list with an explicit default names a real entry from the start.
# (Kept at module level: Blender requires static enum item strings to
# outlive the property.)
_PRESET_ITEMS = preset_items()


class HalcyonSettings(PropertyGroup):
    """All Halcyon render settings, mirroring core.settings.RenderSettings."""

    __annotations__ = _build()

    preset: EnumProperty(
        name="Preset",
        description="Load the settings of a specific 1990s renderer or machine",
        items=_PRESET_ITEMS,
        default='DEFAULT',
    )
    convert_detection: EnumProperty(
        name="Shader Detection", default='AUTO',
        items=(('AUTO', "Automatic",
                "Pick the Halcyon model from each material's source "
                "shader (Principled roughness becomes Glossiness, "
                "Glass becomes Blinn, Toon becomes Toon...)"),
               ('SET', "Set Shader",
                "Force one chosen Halcyon model onto every converted "
                "material, whatever it was before")),
        description="How Convert to Halcyon chooses the shading model")
    # a STATIC items list, built once at import from the same table the
    # master node's own menu reads. Blender refuses an EnumProperty
    # that has both an items CALLBACK and a default ("'default' cannot
    # be set when 'items' is a function" -- the 1.62.0 enable failure),
    # and a callback that imported the add-on by the literal name
    # 'halcyon' would have broken anyway under an extension's mangled
    # package name. Static items have neither problem, and the enum
    # identifiers are stable strings safe to save in a .blend.
    # R252: the list is the MASTER node's menu (core/shading.MASTER_MODELS
    # with each item's engine index as its number): a conversion builds a
    # Halcyon Shader, which no longer offers the anime / cartoon / Max /
    # console models -- those have their own nodes
    convert_model: EnumProperty(
        name="Model", default='PHONG',
        items=_shading.master_model_items(),
        description="The model every conversion gets when Shader "
                    "Detection is Set Shader")
    # R253: the scene's choices for the Convert to Anime / Cartoon / Game
    # buttons -- UI state like convert_detection, never a render setting
    # (to_settings iterates RenderSettings' fields, so these are ignored
    # by the renderer and a saved scene renders exactly as before). All
    # four are STATIC tuples with a default, the rule fakebpy enforces;
    # each tuple is the one the target node's own menu reads, so the
    # identifiers a .blend carries keep their meaning on the node.
    convert_anime_style: EnumProperty(
        name="Anime Style", default='CUSTOM',
        items=_shading.ANIME_STYLE_ITEMS,
        description="The Style the Anime Shader starts from when Convert "
                    "to Anime builds it -- a named decade's tone bands, "
                    "hair shine and rim; Custom leaves the node at its "
                    "defaults")
    convert_anime_compat: EnumProperty(
        name="Anime Compatibility", default='GENERIC',
        items=_shading.ANIME_COMPAT_ITEMS,
        description="How Convert to Anime sets the new node's texture "
                    "decode: Generic Cel, or one of the game pipelines "
                    "whose maps you will plug into Game Texture / Detail "
                    "Texture")
    convert_cartoon_era: EnumProperty(
        name="Cartoon Era", default='CUSTOM',
        items=_shading.CARTOON_ERA_ITEMS,
        description="The Era the Cartoon Shader starts from when Convert "
                    "to Cartoon builds it -- shadow mode, tone and "
                    "highlight written to the node; Custom leaves the "
                    "defaults")
    convert_console: EnumProperty(
        name="Console", default='PS1',
        items=_console.CONSOLE_ITEMS,
        description="The machine Convert to Game builds the Console "
                    "Emulation Shader for; its shader type and options "
                    "stay at that machine's defaults and are editable on "
                    "the node")
    ui_tab: EnumProperty(
        name="Tab", items=_items(
            ('SAMPLING', "Sampling", ""), ('SHADING', "Shading", ""),
            ('OUTPUT', "Output", ""), ('DISPLAY', "Display", "")),
        default='SAMPLING')
    palette_image: PointerProperty(
        name="Palette Image", type=bpy.types.Image,
        description="R202: with Palette set to Custom, the whole "
                    "render is forced through THIS image's colours -- "
                    "point it at a palette table (Image editor > Image "
                    "> Make Palette Table builds one from any picture) "
                    "or at any small image whose colours you want the "
                    "frame to live in")

    def to_settings(self):
        """Copy into a plain RenderSettings for the bpy-free renderer."""
        import dataclasses
        st = RenderSettings()
        for f in dataclasses.fields(RenderSettings):
            if not hasattr(self, f.name):
                continue
            v = getattr(self, f.name)
            if isinstance(f.default, tuple):
                v = tuple(v)
            setattr(st, f.name, v)
        # R202: the image-as-palette road. Custom palette mode reads
        # the picked image's own colours into the quantiser's palette
        # -- the whole render is then FORCED through those colours,
        # which is what the enum item always promised
        if str(st.palette_mode) == 'CUSTOM':
            img = getattr(self, 'palette_image', None)
            if img is not None:
                try:
                    from . import compat
                    from .core.palette import palette_from_pixels
                    px = compat.image_pixels(img)
                    if px is not None:
                        st.palette_colors = tuple(
                            map(tuple,
                                palette_from_pixels(px,
                                                    int(st.palette_size))))
                except Exception:                               # noqa: BLE001
                    pass
        return st


class HalcyonMaterialSettings(PropertyGroup):
    """Per-material overrides, for materials that don't use a node tree."""

    use_override: BoolProperty(
        name="Halcyon Shader", default=False,
        description="Shade this material with a fixed model instead of its node tree")
    strand: BoolProperty(
        name="Hair Geometry", default=False,
        description="This material dresses hair: the mesh's colour "
                    "layer carries strand data (red = root-to-tip "
                    "intercept, green = per-strand random, blue = "
                    "length, alpha = thickness) and the Hair Info "
                    "node reads it. Set automatically on fur-shell "
                    "and exported-hair materials; set it by hand on "
                    "any custom hair mesh built to the same layout")
    alpha_mode: EnumProperty(
        name="Alpha Mode",
        items=[('BLEND', "Blend",
                "Sorted, composited transparency layers -- for glass "
                "and anything genuinely translucent. Every covered "
                "pixel shades once per layer, so deep stacks cost "
                "real time"),
               ('CLIP', "Clip (Punch-Through)",
                "The era's cut-out alpha test: the alpha is compared "
                "against the threshold and the surface is either "
                "fully there or fully absent. Its pixels resolve in "
                "the z-buffer and shade ONCE -- no layers, no "
                "sorting, no per-layer GPU passes -- which is why "
                "fur, foliage, fences and cut-out sprites were cheap "
                "on 1990s hardware. Fur Shells materials use this "
                "automatically"),
               # R251 C031 (transparency pack)
               ('CLIP_BLEND', "Clip + Blend (PS2 two-pass)",
                "The PlayStation 2's two-pass cut-out: alpha at or above "
                "the Clip Threshold resolves in the z-buffer and shades "
                "once, like Clip; everything below it is drawn a second "
                "time as a blended layer with no depth write (GS AFAIL "
                "'framebuffer only'), so hair and foliage keep a soft "
                "edge without sorting")],
        default='BLEND',
        description="How this material's alpha reaches the frame")
    alpha_clip: FloatProperty(
        name="Clip Threshold", default=0.5, min=0.0, max=1.0,
        description="Alpha at or above this renders solid; below it, "
                    "nothing renders at all")
    # R251 (transparency pack): the per-material blend equation
    blend_mode: EnumProperty(
        name="Blend Mode", items=BLEND_MODE_MAT, default='INHERIT',
        description="How this material's fragments blend into the frame: "
                    "the render setting's Blend Equation, or one chip's "
                    "fixed formula for this material alone -- a "
                    "PlayStation scene mixes additive glows and averaged "
                    "glass per primitive, and this is that switch. Env "
                    "Hole shows the world through the surface and writes "
                    "alpha 0 (Blender 2.4x 'Env')")
    # R251 (transparency pack, TRANS-2): Blender 2.4x's Zoffs / ZInvert
    # (C126), Max's Thin Wall Refraction Thickness Offset (C095), Imagine's
    # Fog Length (C101) -- the per-material dials of the blend items
    z_offset: FloatProperty(
        name="Z Offset", default=0.0, min=-1000.0, max=1000.0,
        description="Blender 2.4x Z Offset: this material's see-through "
                    "fragments sort and depth-test as if this much nearer "
                    "the camera, so a decal or label laid on a surface "
                    "never fights it. Shading and fog use the true depth. "
                    "The raster tests and collects this material's "
                    "see-through fragments at the offset depth, so a decal "
                    "a little behind its surface is captured and drawn; an "
                    "opaque material ignores it")
    z_invert: BoolProperty(
        name="Invert Z Depth", default=False,
        description="Blender 2.4x Invert Z Depth: this material's own "
                    "see-through fragments sort far-first, so a closed "
                    "shell shows its far inner wall in front of its near "
                    "wall -- the hollow-object trick. Read under Depth "
                    "order only")
    thin_wall_offset: FloatProperty(
        name="Thickness Offset", default=0.5, min=0.0, max=10.0,
        description="3ds Max Thin Wall Refraction's Thickness Offset: how "
                    "far the picture behind this material is jogged along "
                    "its projected normal, scaled by the material's IOR (0 "
                    "= an invisible pane, 10 = the largest jog). No ray "
                    "leaves the surface; the object's own thickness never "
                    "distorts anything, exactly the map")
    fog_length: FloatProperty(
        name="Fog Length", default=1.0, min=0.0, max=100000.0,
        description="Imagine's Fog Length: the path length through this "
                    "closed object at which its fog becomes fully opaque "
                    "-- opacity is the ray's thickness through the object "
                    "divided by this, capped at 1, with no lamps, shadows "
                    "or scattering (Imagine 3.0 'Fog Attribute'); nested "
                    "shells of one fog material count once. Use a "
                    "Constant (shadeless) model for Imagine's unlit colour")
    model: EnumProperty(name="Model", items=[(a, b, c) for a, b, c in MODEL_ITEMS],
                        default='PHONG')
    diffuse: FloatVectorProperty(name="Diffuse", subtype='COLOR', size=3,
                                 default=(0.8, 0.8, 0.8), min=0.0, max=1.0)
    diffuse_level: FloatProperty(name="Diffuse Level", default=1.0, min=0.0, max=2.0)
    specular: FloatVectorProperty(name="Specular", subtype='COLOR', size=3,
                                  default=(1.0, 1.0, 1.0), min=0.0, max=1.0)
    specular_level: FloatProperty(name="Specular Level", default=0.5, min=0.0, max=4.0)
    glossiness: FloatProperty(name="Glossiness", default=25.0, min=0.5, max=8192.0)
    ambient_level: FloatProperty(name="Ambient", default=1.0, min=0.0, max=4.0)
    emission: FloatVectorProperty(name="Self-Illumination", subtype='COLOR', size=3,
                                  default=(0.0, 0.0, 0.0), min=0.0, max=1.0)
    emission_level: FloatProperty(name="Self-Illum Level", default=0.0, min=0.0,
                                  max=64.0)
    opacity: FloatProperty(name="Opacity", default=1.0, min=0.0, max=1.0)
    ior: FloatProperty(name="IOR", default=1.45, min=1.0, max=4.0)
    roughness: FloatProperty(name="Roughness", default=0.3, min=0.0, max=1.0)
    anisotropy: FloatProperty(name="Anisotropy", default=0.0, min=-1.0, max=1.0)
    aniso_rotation: FloatProperty(name="Aniso Rotation", default=0.0, min=0.0,
                                  max=6.2832)
    metallic: FloatProperty(name="Metalness", default=0.0, min=0.0, max=1.0)
    reflect_level: FloatProperty(name="Reflection", default=0.0, min=0.0, max=1.0)
    soften: FloatProperty(name="Soften", default=0.0, min=0.0, max=1.0,
                          description="3D Studio's specular softening at grazing angles")
    two_sided: BoolProperty(name="Two Sided", default=True)
    shadeless: BoolProperty(name="Shadeless", default=False)
    cast_shadow: BoolProperty(name="Cast Shadows", default=True)
    receive_shadow: BoolProperty(name="Receive Shadows", default=True)
    # R251 C134: the gel of the Tron printer's backlit matte
    glow_gel: FloatVectorProperty(
        name="Glow Gel", subtype='COLOR', size=3, min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0),
        description="The gel this material's Kodalith matte is backlit "
                    "through in the Tron printer stage; black means no "
                    "matte for this material (Optical Effects > Matte "
                    "Glow must be on)")
    # R251 C020/C036: the material's triangles as an authored volume
    volume_role: EnumProperty(
        name="Volume Role", default='NONE', items=VOLUME_ROLE,
        description="What this material's triangles are: a drawn surface, "
                    "or a closed authored volume that is never drawn as "
                    "geometry and instead darkens the pixels it encloses "
                    "-- the Dreamcast's modifier volumes (stencil parity "
                    "times the shadow scale; the outside kind darkens the "
                    "whole frame except the inside) or the Nintendo DS's "
                    "shadow polygons (depth-fail mask, then a 5-bit alpha "
                    "blend of this colour). Every frame, viewport drafts "
                    "included, pays one fragment capture per volume while "
                    "the render's Shadows are on (off draws no volume)")
    polygon_id: IntProperty(
        name="Polygon ID", default=0, min=0, max=63,
        description="The Nintendo DS polygon ID (0-63) written to the "
                    "attribute buffer with this material's pixels; a DS "
                    "shadow polygon darkens only pixels whose ID differs "
                    "from its own, which is how a character is kept out "
                    "of its own shadow -- give the caster and its shadow "
                    "volume the same ID")
    shadow_alpha: IntProperty(
        name="DS Shadow Alpha", default=16, min=1, max=30,
        description="Alpha of this material as a DS shadow polygon, in "
                    "the DS's 5-bit units 1-30: the pixel becomes "
                    "(colour x (alpha+1) + pixel x (31-alpha)) / 32 on "
                    "6-bit channels -- 16 is the half-dark shadow most "
                    "games used, 30 nearly solid")
    # R233: the cel or the painting
    paint_mode: EnumProperty(
        name="Paint Mode", default='CEL', items=_items(
            ('CEL', "Cel",
             "Painted on celluloid: flat paint under the ink line, as "
             "every material was until now"),
            ('BACKGROUND', "Background",
             "The background painting: the lit colour laid down as "
             "brush strokes fixed on the surface (Render Properties > "
             "Display > Painted Backgrounds), no ink unless Ink says "
             "Always, softened by the setback")),
        description="Cel or background painting -- which department "
                    "painted this surface")
    # R220: per-material ink -- this material's say over the cartoon
    # outline pass, which until now was one global render setting
    ink_mode: EnumProperty(
        name="Ink", default='INHERIT', items=_items(
            ('INHERIT', "Follow Render Setting",
             "Ink exactly as the render's Cartoon Outlines switch says "
             "-- the pre-R220 behaviour"),
            ('ON', "Always Ink",
             "This material's edges draw ink even when the render "
             "switch is off -- one inked character in a plain scene"),
            ('OFF', "Never Ink",
             "No ink ever lands on this material's pixels -- glass, "
             "effects and skies stay clean in an inked scene")),
        description="Whether the cartoon outline pass inks this "
                    "material's pixels")
    ink_use_color: BoolProperty(
        name="Own Ink Colour", default=False,
        description="Ink this material's edges with its own colour "
                    "instead of the render's global Ink Colour -- "
                    "coloured line art, per material")
    ink_color: FloatVectorProperty(
        name="Ink Colour", subtype='COLOR', size=3,
        default=(0.0, 0.0, 0.0), min=0.0, max=1.0,
        description="This material's line colour, used when Own Ink "
                    "Colour is on")
    ink_width: IntProperty(
        name="Ink Width", default=0, min=0, max=8,
        description="This material's line width in pixels; 0 inherits "
                    "the render's global Ink Width")
    # R239: the Guilty Gear line control -- the mesh's vertex colours
    # steer the line per vertex, exactly the convention Arc System
    # Works teaches for their inverted-hull outlines
    ink_vc: EnumProperty(
        name="Line Control", default='OFF', items=_items(
            ('OFF', "Off",
             "The line ignores the mesh's vertex colours"),
            ('ARCSYS', "Vertex Colour (ArcSys)",
             "The Guilty Gear convention on this material's vertex "
             "colours: ALPHA multiplies the line's width -- 0.5 is the "
             "width as set, 1 doubles it, 0 erases the line at that "
             "vertex (a glove's rim, a sleeve's opening); BLUE holds "
             "interior and marked lines back until the surface turns "
             "toward its silhouette -- the nose line that only draws "
             "in profile (their hull's depth push, read by facing). "
             "Silhouettes always keep their line. Paint on the "
             "mesh's first colour layer")),
        description="Per-vertex say over this material's ink, painted "
                    "in the mesh's vertex colours (Arc System Works' "
                    "own outline convention)")
    wire: BoolProperty(name="Wireframe", default=False)
    wire_size: FloatProperty(name="Wire Size", default=1.0, min=0.1, max=16.0)
    # ---- Halo material (R191): Blender Internal's MA_TYPE_HALO ----
    halo: BoolProperty(
        name="Halo", default=False,
        description="Render this material as Blender Internal halos: "
                    "the mesh's vertices become depth-tested billboard "
                    "glows instead of surfaces -- the 90s way to do "
                    "sparks, fairy dust, star fields and energy effects")
    halo_size: FloatProperty(
        name="Halo Size", default=0.5, min=0.0, max=100.0,
        description="World-space radius of each glow (2.79's HaloSize)")
    halo_hardness: IntProperty(
        name="Hardness", default=50, min=0, max=127,
        description="Falloff shape, 2.79's exact ladder: below 20 "
                    "squares the falloff, 30/40/50 each soften it a "
                    "step further")
    halo_add: FloatProperty(
        name="Add", default=0.0, min=0.0, max=1.0,
        description="Slides the blend from alpha-over (0) to pure "
                    "additive glow (1) -- 2.79's Add slider")
    halo_alpha: FloatProperty(
        name="Alpha", default=1.0, min=0.0, max=1.0,
        description="The halo's own opacity (2.79's material Alpha)")
    halo_color: FloatVectorProperty(
        name="Halo Colour", subtype='COLOR', size=3,
        default=(0.8, 0.8, 0.8), min=0.0, max=1.0,
        description="The glow's own colour -- 2.79 read the material's "
                    "base colour for this")
    halo_seed: IntProperty(
        name="Seed", default=0, min=0, max=255,
        description="Starting seed for the rings and lines hash -- "
                    "each vertex walks on from it, exactly 2.79")
    halo_rings: BoolProperty(
        name="Rings", default=False,
        description="Concentric circles around each halo, placed by the "
                    "seed hash -- BI's Rings flag")
    halo_ring_count: IntProperty(
        name="Ring Count", default=4, min=1, max=24,
        description="How many concentric circles each halo draws, "
                    "placed by the seed hash")
    halo_ring_color: FloatVectorProperty(
        name="Ring Colour", subtype='COLOR', size=3,
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        description="2.79 took this from the material's Mirror colour")
    halo_lines: BoolProperty(
        name="Lines", default=False,
        description="Random radial streaks through each halo, directions "
                    "drawn from the seed hash -- BI's Lines flag")
    halo_line_count: IntProperty(
        name="Line Count", default=12, min=1, max=250,
        description="How many radial streaks cross each halo, "
                    "directions drawn from the seed hash")
    halo_line_color: FloatVectorProperty(
        name="Line Colour", subtype='COLOR', size=3,
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        description="2.79 took this from the material's Specular colour")
    halo_star: BoolProperty(
        name="Star", default=False,
        description="Pinch each halo into a star with the Star Tips "
                    "point count -- BI's Star flag")
    halo_star_tips: IntProperty(
        name="Star Tips", default=4, min=3, max=50,
        description="Points on the star the halo is pinched into "
                    "(4 was the classic lens sparkle)")
    halo_shape: EnumProperty(
        name="Shape", default='DISC',
        description="The glow's core silhouette; every shape still "
                    "runs the hardness ladder and the effects",
        items=_items(
            ('DISC', "Disc", "The classic round glow, 2.79's own core"),
            ('RING', "Ring", "A hollow ring -- the centre pushed to the "
                             "rim by 2.79's own flare-circle formula"),
            ('HEX', "Hexagon", "A soft six-sided glow, the lens-iris "
                               "look of the era's flare kits"),
            ('DIAMOND', "Diamond", "A four-pointed soft diamond sparkle"),
            ('TRIANGLE', "Triangle", "A soft three-sided glow, vertex "
                                     "up under zero rotation"),
            ('PENTAGON', "Pentagon", "A soft five-sided iris glow"),
            ('OCTAGON', "Octagon", "A soft eight-sided iris glow"),
            ('CROSS', "Cross", "A plus-sign glow, bright along both "
                               "axes and tapering at the rim"),
            ('SQUARE', "Square", "A soft axis-aligned square glow"),
            ('STAR', "Star", "A solid five-point star, points on the "
                             "halo circle"),
            ('HEART', "Heart", "The classic heart, radial glow inside"),
            ('IMAGE', "Image", "An image IS the halo: its alpha the "
                               "shape, its colours the glow -- the "
                               "HaloTex idea, native")))
    halo_image: PointerProperty(
        name="Halo Image", type=bpy.types.Image,
        description="The picture drawn as the halo when Shape is "
                    "Image; alpha carves the silhouette")
    halo_noise: FloatProperty(
        name="Noise", default=0.0, min=0.0, max=1.0,
        description="Animated value noise carving and boosting the "
                    "core -- the energy-blast writhe; composes with "
                    "every shape and the image")
    halo_noise_scale: FloatProperty(
        name="Noise Scale", default=4.0, min=0.2, max=32.0,
        description="Cells of noise across the halo; higher is "
                    "finer boiling")
    halo_bolts: IntProperty(
        name="Bolts", default=0, min=0, max=24,
        description="Electric arcs radiating from the centre, each "
                    "wiggling with radius and re-striking eight times "
                    "per animation second")
    halo_bolt_width: FloatProperty(
        name="Bolt Width", default=1.0, min=0.05, max=8.0,
        description="Thickness of the electric arcs, in the same "
                    "resolution-true units as Line Width")
    halo_bolt_color: FloatVectorProperty(
        name="Bolt Colour", subtype='COLOR', size=3,
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        description="The electric arcs' own colour; until you set it, "
                    "bolts follow the Line Colour as they always did")
    halo_rays: IntProperty(
        name="Rays", default=0, min=0, max=64,
        description="EVENLY spaced rays -- the symmetric starburst "
                    "the hashed Lines cannot make; they turn with "
                    "Rotation and Spin")
    halo_ray_sharp: FloatProperty(
        name="Ray Sharpness", default=8.0, min=0.5, max=64.0,
        description="How needle-thin the even rays are; higher is "
                    "sharper spikes")
    halo_ray_color: FloatVectorProperty(
        name="Ray Colour", subtype='COLOR', size=3,
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        description="The even rays' own colour; until you set it, "
                    "rays follow the Line Colour as they always did")
    halo_rings_even: BoolProperty(
        name="Even Rings", default=False,
        description="Space the rings evenly out from the centre -- "
                    "shockwaves -- instead of hashing their radii the "
                    "2.79 way")
    halo_gradient_type: EnumProperty(
        name="Gradient Type", default='RADIAL',
        description="How the gradient (or colour ramp) sweeps the "
                    "halo; linear sweeps turn with Rotation and Spin",
        items=_items(
            ('RADIAL', "Centre Out (Radial)",
             "Centre to rim along the radius"),
            ('ANGULAR', "Angular", "A full turn around the centre -- "
                                   "the conic sweep"),
            ('HORIZONTAL', "Horizontal",
             "Left to right across the halo"),
            ('VERTICAL', "Vertical", "Bottom to top across the halo"),
            ('DIAGONAL', "Diagonal", "Corner to corner at 45 degrees")))
    halo_gradient_noise: FloatProperty(
        name="Gradient Noise", default=0.0, min=0.0, max=1.0,
        description="Wobbles the gradient coordinate with animated "
                    "noise -- turbulent colour bands; Noise Scale sets "
                    "the cell size, Anim Speed drives it, and the "
                    "angular sweep wraps seamlessly")
    halo_aspect: FloatProperty(
        name="Aspect", default=1.0, min=0.05, max=20.0,
        description="Stretches the halo horizontally (above 1) or "
                    "vertically (below 1) -- with Rotation, a turned "
                    "anamorphic streak; applies to every shape, the "
                    "lines and the star")
    halo_rotation: FloatProperty(
        name="Rotation", default=0.0, min=-6.2832, max=6.2832,
        subtype='ANGLE',
        description="Static turn of the shape, lines and star about "
                    "the centre; Spin animates on top of it")
    halo_line_width: FloatProperty(
        name="Line Width", default=1.0, min=0.05, max=8.0,
        description="Thickness of the radial streaks; 1.0 is the "
                    "classic hairline")
    halo_ring_width: FloatProperty(
        name="Ring Width", default=1.0, min=0.05, max=8.0,
        description="Thickness of the concentric rings; 1.0 is the "
                    "classic thin circle")
    halo_gradient: BoolProperty(
        name="Gradient", default=False,
        description="Blend from the halo colour at the centre to the "
                    "Edge Colour at the rim")
    halo_color2: FloatVectorProperty(
        name="Edge Colour", subtype='COLOR', size=3,
        default=(0.0, 0.0, 0.0), min=0.0, max=1.0,
        description="The rim end of the gradient; the halo colour "
                    "holds the centre")
    halo_rand_hue: FloatProperty(
        name="Random Hue", default=0.0, min=0.0, max=1.0,
        description="Per-halo hue scatter off the seed -- confetti "
                    "clouds from one material, deterministic per "
                    "vertex; it scatters every coloured option, trim "
                    "included")
    halo_rand_sat: FloatProperty(
        name="Random Saturation", default=0.0, min=0.0, max=1.0,
        description="Per-halo saturation scatter off the seed, over "
                    "every coloured option")
    halo_rand_val: FloatProperty(
        name="Random Value", default=0.0, min=0.0, max=1.0,
        description="Per-halo brightness scatter off the seed, over "
                    "every coloured option")
    halo_hue_shift: FloatProperty(
        name="Hue Shift", default=0.0, min=-1.0, max=1.0,
        description="Turns the hue of EVERY coloured option together "
                    "-- body, gradient end, ramp, image, rings, "
                    "lines, rays and bolts; a full turn is 1.0, and "
                    "keyframing it cycles the whole halo through the "
                    "wheel")
    halo_sat_shift: FloatProperty(
        name="Saturation Shift", default=1.0, min=0.0, max=2.0,
        description="Scales the saturation of every coloured option "
                    "together; 0 drains the halo to grey, above 1 "
                    "over-saturates")
    halo_val_shift: FloatProperty(
        name="Value Shift", default=1.0, min=0.0, max=4.0,
        description="Scales the brightness of every coloured option "
                    "together; keyframable for fades that keep the "
                    "alpha shape")
    halo_pulse: FloatProperty(
        name="Pulse", default=0.0, min=0.0, max=1.0,
        description="Each halo's size breathes over time on its own "
                    "hashed phase, so a cloud shimmers instead of "
                    "throbbing in sync")
    halo_flicker: FloatProperty(
        name="Flicker", default=0.0, min=0.0, max=1.0,
        description="Per-frame per-halo brightness jitter -- the 90s "
                    "sparkle; deterministic in (seed, frame)")
    halo_spin: FloatProperty(
        name="Spin", default=0.0, min=-16.0, max=16.0,
        description="Turns the lines, star and shaped cores about the "
                    "centre, in radians per second of scene time")
    halo_anim_speed: FloatProperty(
        name="Anim Speed", default=1.0, min=0.0, max=8.0,
        description="The master clock every animated halo effect "
                    "rides -- Pulse, Noise, Bolts and Gradient Noise "
                    "all scale by it; 0 freezes them all in place")
    halo_pulse_speed: FloatProperty(
        name="Pulse Speed", default=1.0, min=0.0, max=20.0,
        description="Scales the Pulse breathing alone, on top of the "
                    "master Anim Speed; 0 freezes just the pulse")
    halo_flicker_speed: FloatProperty(
        name="Flicker Speed", default=1.0, min=0.0, max=20.0,
        description="How often the Flicker re-rolls: 1 is every "
                    "frame (the classic sparkle), 0.5 every other "
                    "frame, 0 holds one roll forever")
    halo_noise_speed: FloatProperty(
        name="Noise Speed", default=1.0, min=0.0, max=20.0,
        description="Scales the core Noise writhe alone, on top of "
                    "the master Anim Speed; 0 freezes the boil")
    halo_bolt_speed: FloatProperty(
        name="Bolt Speed", default=1.0, min=0.0, max=20.0,
        description="Scales the electric arcs alone -- strike rate "
                    "and writhe together -- on top of the master "
                    "Anim Speed; 0 freezes the strike")
    halo_grad_noise_speed: FloatProperty(
        name="Gradient Noise Speed", default=1.0, min=0.0, max=20.0,
        description="Scales the Gradient Noise wobble alone, on top "
                    "of the master Anim Speed; 0 freezes the bands")
    halo_xalpha: BoolProperty(
        name="Extreme Alpha", default=False,
        description="Square the alpha for a hotter core (MA_HALO_XALPHA)")
    halo_soft: BoolProperty(
        name="Soft", default=False,
        description="Soften halos where they intersect geometry by how "
                    "much of their depth is visible (MA_HALO_SOFT)")
    halo_shaded: BoolProperty(
        name="Shaded", default=False,
        description="Tint each halo by the scene's lamps at its centre "
                    "(MA_HALO_SHADE)")
    halo_puno: BoolProperty(
        name="Vertex Normal", default=False,
        description="Scale each halo by its vertex normal's facing -- "
                    "rear-facing verts glow, camera-facing ones vanish "
                    "(MA_HALOPUNO)")


class HalcyonLightSettings(PropertyGroup):
    decay: EnumProperty(
        name="Falloff",
        items=[('DEFAULT', "Scene Default",
                "Use the falloff set in Render Properties > Lighting")] + FALLOFF,
        default='DEFAULT')
    decay_start: FloatProperty(name="Falloff Start", default=0.0, min=0.0)
    decay_end: FloatProperty(name="Falloff End", default=25.0, min=0.0)
    decay_ld1: FloatProperty(
        name="Linear Slider", default=0.0, min=0.0, max=1.0,
        description="BI's Lin slider (att1): active under the "
                    "Lin/Quad Sliders falloff while above zero")
    decay_ld2: FloatProperty(
        name="Quad Slider", default=0.0, min=0.0, max=1.0,
        description="BI's Quad slider (att2): active under the "
                    "Lin/Quad Sliders falloff while above zero")
    bi_sphere: BoolProperty(
        name="Sphere", default=False,
        description="BI's Sphere clamp: the light fades linearly to "
                    "zero at its Falloff End and never reaches past "
                    "it, whatever the falloff curve")
    gx_ref_brite: FloatProperty(
        name="Ref Brightness (GX)", default=0.5, min=0.05, max=0.95,
        description="GX_InitLightDistAttn's ref_brite: the fraction of the "
                    "lamp's energy left at Falloff End (its ref_dist) under "
                    "the three GameCube distance laws")
    spot_law: EnumProperty(
        name="Cone Law", items=SPOT_LAW, default='BLENDER',
        description="How the spot's cone fades: Blender Internal's smoothstep "
                    "band times the cosine, OpenGL 1.1's hard cutoff with a "
                    "cosine power, POV-Ray's flat hotspot with a Hermite "
                    "edge, or the GameCube's six GX angular functions built "
                    "from the cone angle (the ring functions light a ring "
                    "and leave the axis dark). Spot Blend is Blender "
                    "Internal's only: the other laws never read it")
    spot_exponent: FloatProperty(
        name="Cone Exponent", default=0.0, min=0.0, max=128.0,
        description="OpenGL's GL_SPOT_EXPONENT / POV-Ray's tightness: the "
                    "cosine of the angle to the axis raised to this power "
                    "inside the cone; 0 (both defaults) is flat")
    screen_spot: BoolProperty(
        name="Screen Spotlight (Model 3)", default=False,
        description="Sega Model 3's viewport spotlight: an ellipse pinned to "
                    "the screen at this lamp's projected position, half the "
                    "Spot Size wide as a fraction of the frame height, adding "
                    "the lamp's colour at its Energy over 4 pi (the same "
                    "radiance scale every Halcyon lamp uses) to the diffuse "
                    "inside a depth window from the near clip over the lamp's "
                    "Falloff End -- the headlight pool that slides with the "
                    "camera. No shadows, no cone; it still counts as one of "
                    "Max Lights")
    shadow: EnumProperty(name="Shadows", items=_items(
        ('NONE', "None", ""), ('MAP', "Shadow Map", ""),
        ('RAY', "Ray Traced", ""),
        # R251 C052: appended last
        ('PLANAR', "Planar Polygons (Model 1 / Blinn)",
         "This lamp projects the casters onto the Floor Plane Z and "
         "draws them as flat polygons, Blinn 1988 / Sega Model 1")),
        default='MAP')
    shadow_map_size: IntProperty(
        name="Map Size", default=0, min=0, max=4096,
        description="Shadow map resolution for this light. 0 uses the "
                    "render setting's Shadow Map Size")
    shadow_bias: FloatProperty(
        name="Bias", default=0.0, min=0.0, max=10.0,
        description="Depth offset that stops a surface shadowing itself. "
                    "0 uses the render setting's Shadow Bias")
    # R251 C117: the per-lamp map depth (Maya's Use Mid Dist and
    # Blender's buffer type were per lamp)
    shadow_map_depth: EnumProperty(
        name="Map Depth", default='INHERIT', items=LIGHT_MAP_DEPTH,
        description="This lamp's own map depth: Inherit follows the "
                    "render's Map Depth; Classic and Midpoint override it "
                    "for this lamp alone -- Maya's Use Mid Dist and "
                    "Blender's buffer type were per lamp, and a scene can "
                    "mix a tight midpoint spot with a classic sun")
    shadow_softness: FloatProperty(name="Softness", default=1.0, min=0.0, max=32.0)
    shadow_samples: IntProperty(name="Samples", default=4, min=1, max=64)
    shadow_density: FloatProperty(name="Density", default=1.0, min=0.0, max=1.0)
    shadow_color: FloatVectorProperty(name="Shadow Colour", subtype='COLOR', size=3,
                                      default=(0.0, 0.0, 0.0), min=0.0, max=1.0)
    negative: BoolProperty(name="Negative", default=False,
                           description="Subtract light instead of adding it, as "
                                       "3D Studio and LightWave allowed")
    only_shadow: BoolProperty(
        name="Only Shadow", default=False,
        description="Blender Internal's Only Shadow lamp: the lamp lights "
                    "nothing and only darkens the diffuse where its shadow "
                    "falls, by the plain diffuse it would have given (a "
                    "Blender Internal material's diffuse ramp and "
                    "terminator bias are not part of the subtraction) -- "
                    "a shadow without a light. With shadows off, or on a "
                    "lamp that affects no diffuse, it does nothing")
    hemi: BoolProperty(
        name="Hemisphere (BI Hemi)", default=False,
        description="Shade this Sun as Blender Internal's Hemi lamp: "
                    "the 0.5+0.5*N.L wrap on the diffuse, a wrapped "
                    "half-vector highlight, no shadows -- the dome "
                    "light 2.79 had and Blender since 2.8 does not")
    diffuse_only: BoolProperty(name="Diffuse Only", default=False)
    specular_only: BoolProperty(name="Specular Only", default=False)
    ambient_only: BoolProperty(name="Ambient Only", default=False)
    hotspot: FloatProperty(name="Hotspot", default=0.0, min=0.0, max=3.1416,
                           description="Inner cone angle, the 3D Studio "
                                       "hotspot/falloff pair; POV-Ray's "
                                       "radius under the POV cone law, "
                                       "never wider than the Spot Size")
    area_gamma: FloatProperty(
        name="Gamma", default=1.0, min=0.01, max=2.0,
        description="Blender Internal's area lamp Gamma (la->k): shapes "
                    "the form-factor energy as pow(intensity, gamma). "
                    "1.0 is linear; below softens the falloff across "
                    "the lit field, above sharpens it")
    volumetric: FloatProperty(
        name="Volumetric", default=0.0, min=0.0, max=4.0,
        description="Scatters light along the view ray toward this lamp. "
                    "Spot lamps cast their visible cone, point lamps a "
                    "bounded glow, area lamps a soft-edged slab beam -- all "
                    "layered against the scene's depth; a Sun (no apex to "
                    "march from) drives the screen-space light shafts "
                    "instead")
    volumetric_occlusion: BoolProperty(
        name="Beam Occlusion", default=False,
        description="Trace each beam sample back to the lamp, so the "
                    "visible beam stops at a mesh in its way instead of "
                    "shining through it. Costs a shadow ray per sample "
                    "per pixel inside the beam")
    flare: FloatProperty(
        name="Lens Flare", default=0.0, min=0.0, max=4.0,
        description="Draw this lamp's lens flare: hot core, chromatic "
                    "halo rings, a star of streaks and aperture ghosts "
                    "marching through frame centre -- the Video Post kit. "
                    "Anchored on the lamp and faded by its visibility, so "
                    "the flare dies as the lamp slips behind geometry")
    flare_scale: FloatProperty(
        name="Flare Scale", default=1.0, min=0.05, max=4.0,
        description="Size of every flare element, relative to the frame")
    flare_streaks: IntProperty(
        name="Streaks", default=6, min=0, max=16,
        description="Star spokes radiating from the source. 0 for none; "
                    "6 and 8 were the era's favourites")
    flare_rings: IntProperty(
        name="Rings", default=1, min=0, max=4,
        description="How many chromatic halo rings circle the source, "
                    "each fringed a different hue")
    flare_ghosts: IntProperty(
        name="Ghosts", default=6, min=0, max=12,
        description="Aperture ghosts along the line through frame centre")
    caustics: FloatProperty(
        name="Caustics", default=0.0, min=0.0, max=4.0,
        description="Project the animated pool-light web through this "
                    "lamp -- the writhing bright cell edges every 1990s "
                    "pool floor and water cave was lit with. A spot casts "
                    "it through its cone; a sun tiles it seamlessly "
                    "across the world. A real cookie image on the lamp "
                    "takes priority")
    caustics_scale: FloatProperty(
        name="Caustic Scale", default=4.0, min=0.05, max=64.0,
        description="Spot: how many cells across the cone. Sun: world "
                    "units covered by one seamless tile")
    caustics_speed: FloatProperty(
        name="Caustic Speed", default=1.0, min=0.0, max=8.0,
        description="How fast the caustic web writhes over the frames")
    exclude_collection: PointerProperty(
        name="Light Linking", type=bpy.types.Collection,
        description="A collection this lamp treats specially. Every 1990s "
                    "package let you say which objects a light touched, and it "
                    "is still the quickest way to control a render")
    exclude_mode: EnumProperty(name="Linking", default='EXCLUDE', items=_items(
        ('EXCLUDE', "Exclude", "The collection is not lit by this lamp"),
        ('ONLY', "Only", "Nothing but the collection is lit by this lamp")))
    cookie: PointerProperty(
        name="Projected Texture", type=bpy.types.Image,
        description="An image this lamp projects -- the gobo/cookie of the "
                    "sixth-generation consoles. A Spot throws it through its "
                    "cone like a slide projector (Splinter Cell's window "
                    "patterns); a Sun tiles it across the world as a cloud "
                    "shadow; a Point wraps it around itself like a pierced "
                    "lantern; an Area carries it on its face like a printed "
                    "gel. The lamp's Angle/Radius slider softens the "
                    "projection, exactly as a bigger source would")
    cookie_strength: FloatProperty(
        name="Projection Strength", default=1.0, min=0.0, max=1.0,
        description="Blend between plain light (0) and the fully projected "
                    "image (1)")
    cookie_scale: FloatProperty(
        name="Projection Scale", default=10.0, min=0.01,
        description="Sun only: world size of one tile of the projected "
                    "image, in scene units")
    cookie_extend: EnumProperty(
        name="Extension", default='AUTO', items=_items(
            ('AUTO', "Era Default",
             "What each projection always did: a Sun tiles its cloud "
             "shadow, a Spot and an Area clamp at the image edge, a "
             "Point wraps around its own seam"),
            ('REPEAT', "Repeat", "Tile the image endlessly"),
            ('EXTEND', "Extend", "The edge pixels continue forever"),
            ('CLIP', "Clip",
             "Outside the image there is nothing -- the projector's "
             "gate: at full strength, no light past the slide")),
        description="How the projected image continues past its edges")
    cookie_filter: EnumProperty(
        name="Interpolation", default='BILINEAR', items=_items(
            ('BILINEAR', "Bilinear",
             "The era's four-texel blend -- the default, and what every "
             "projection used before this was a choice"),
            ('CLOSEST', "Closest",
             "One texel, hard edges -- the pixelated slide"),
            ('CUBIC', "Cubic",
             "A smooth sixteen-texel B-spline -- never overshoots, so "
             "the projection cannot ring")),
        description="How texels of the projected image are blended, "
                    "identically on both devices")


def _col(name, default, desc=''):
    return FloatVectorProperty(name=name, description=desc, subtype='COLOR',
                               size=3, default=default, min=0.0, max=1.0)


# --------------------------------------------------------------- sky library
#
# The bpy-touching half of the sky preset system. `presets/skies.py` stays
# bpy-free because the renderer reads it; finding the folder Blender keeps
# user presets in does not belong there.


def sky_library_dir():
    """Where saved skies live, alongside Blender's own presets.

    Never fatal: a build that cannot hand out a scripts folder should cost the
    saved-sky list, not the whole World panel.
    """
    import os
    try:
        path = bpy.utils.user_resource('SCRIPTS', path="presets/halcyon_skies",
                                       create=True)
    except Exception:                                           # noqa: BLE001
        return None
    return path if path and os.path.isdir(path) else None


def user_skies():
    """(path, label) for every sky saved into the library, sorted."""
    import os
    out = []
    folder = sky_library_dir()
    if not folder:
        return out
    try:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith('.halsky'):
                out.append((os.path.join(folder, name),
                            os.path.splitext(name)[0]))
    except OSError:
        pass
    return out


#: Blender does not keep a reference to the strings an items callback returns,
#: and a garbage-collected enum item is a corrupted menu. The list is held here
#: for as long as the property might read it.
_SKY_ITEMS = []


_SKY_PREVIEWS = [None]


def sky_previews():
    """The thumbnail collection, loaded lazily from presets/thumbs/.

    Every built-in sky ships a rendered thumbnail (the engine's own sky
    module drew them). Never fatal: without bpy.utils.previews the enum
    simply has no pictures.
    """
    if _SKY_PREVIEWS[0] is not None:
        return _SKY_PREVIEWS[0]
    import os
    try:
        import bpy.utils.previews
        pcoll = bpy.utils.previews.new()
        base = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), 'halcyon', 'presets', 'thumbs')
        if not os.path.isdir(base):
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'presets', 'thumbs')
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.endswith('.png'):
                    key = name[:-4]
                    pcoll.load(key, os.path.join(base, name), 'IMAGE')
        _SKY_PREVIEWS[0] = pcoll
    except Exception:                                           # noqa: BLE001
        _SKY_PREVIEWS[0] = {}
    return _SKY_PREVIEWS[0]


def sky_preset_items(self=None, context=None):
    from .presets.skies import sky_items
    pcoll = sky_previews()
    items = []
    for i, (k, label, note) in enumerate(sky_items()):
        icon = 0
        try:
            if pcoll and k in pcoll:
                icon = pcoll[k].icon_id
        except Exception:                                       # noqa: BLE001
            icon = 0
        items.append((k, label, note, icon, i) if icon else (k, label, note))
    saved = user_skies()
    if saved:
        items.append(('', "Saved Skies", ''))
        for path, label in saved:
            items.append(('FILE:' + path, label, "A sky you saved"))
    # a mixed 3/5-tuple list confuses Blender: normalise to 5-tuples
    norm = []
    for j, it in enumerate(items):
        if len(it) == 3:
            norm.append((it[0], it[1], it[2], 0, len(sky_items()) + j))
        else:
            norm.append(it)
    _SKY_ITEMS[:] = norm
    return _SKY_ITEMS


def water_library_dir():
    """Where saved waters live. Never fatal, for the same reason as skies."""
    import os
    try:
        path = bpy.utils.user_resource('SCRIPTS', path="presets/halcyon_waters",
                                       create=True)
    except Exception:                                           # noqa: BLE001
        return None
    return path if path and os.path.isdir(path) else None


def user_waters():
    """(path, label) for every water saved into the library, sorted."""
    import os
    out = []
    folder = water_library_dir()
    if not folder:
        return out
    try:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith('.halwater'):
                out.append((os.path.join(folder, name),
                            os.path.splitext(name)[0]))
    except OSError:
        pass
    return out


_WATER_ITEMS = []


def water_preset_items(self=None, context=None):
    from .presets.waters import water_items
    items = [(k, label, note) for k, label, note in water_items()]
    saved = user_waters()
    if saved:
        items.append(('', "Saved Waters", ''))
        for path, label in saved:
            items.append(('FILE:' + path, label, "A water you saved"))
    _WATER_ITEMS[:] = items
    return _WATER_ITEMS


def _painted_look_changed(self, context):
    """R233: the Look menu writes its dials onto the world (Custom writes
    nothing), exactly the Cartoon node's Era road."""
    look = PAINTED_LOOKS.get(str(self.paint_look))
    if not look:
        return
    for k, v in look.items():
        try:
            setattr(self, k, v)
        except (TypeError, ValueError, AttributeError):
            pass


class HalcyonWorldSettings(PropertyGroup):
    sky_preset: EnumProperty(
        name="Sky Preset", items=sky_preset_items,
        description="Pick a sky, then press Apply Preset. Choosing one here "
                    "does not change anything on its own")
    water_preset: EnumProperty(
        name="Water Preset", items=water_preset_items,
        description="Pick a water, then press Apply Preset. Choosing one here "
                    "does not change anything on its own. Waters and skies "
                    "are separate libraries and neither overwrites the other")

    mode: EnumProperty(name="Sky", default='NODES', items=_items(
        ('NODES', "Use Node Tree", "Evaluate the world's own shader nodes"),
        ('SOLID', "Solid Colour", "A single flat background colour"),
        ('GRADIENT', "Gradient", "Horizon to zenith blend, with an optional ground"),
        ('BANDS', "Banded Gradient",
         "The same blend cut into flat steps, the way a 256-colour palette "
         "could only ever render one"),
        ('STARFIELD', "Starfield",
         "Space: a flat backdrop, stars all the way round, optional nebula"),
        ('BRYCE', "Bryce Atmosphere",
         "Layered sky: gradient, sun glow, haze band and a fractal cloud deck"),
        ('PHYSICAL', "Physical Sky", "Preetham analytic daylight"),
        ('HDRI', "Image / HDRI", "Wrap an image around the scene"),
        ('PAINTED', "Painted Backdrop",
         "A background painting on a flat panel in front of the camera: "
         "the gradient brushed in gouache, painted clouds, the board's "
         "tooth, a watercolour granulation -- a pan crosses it, a tilt "
         "climbs it, the way the animation stand's background did"),
        # R251 sky-camera: appended at the END (positional numbering)
        ('CYLINDER', "Cylinder Sky (Doom)",
         "Doom's angle-mapped sky: the image wrapped as a cylinder by "
         "camera yaw alone, rows 1:1 with a 200-line screen, unlit, never "
         "tilting with pitch. The image runs right-to-left around the "
         "turn as Doom's own did, and reflections see the flat World "
         "colour -- a backdrop, not an environment. In a stereo pair the "
         "sky sits at infinity (each eye's window shift moves it), and "
         "accumulation AA jitters it with the rest of the frame"),
        ('LW_GRADIENT', "Gradient Backdrop (LightWave)",
         "LightWave's four-colour backdrop: Sky to Zenith above the "
         "horizon, Ground to Nadir below, a hard horizon step, each blend "
         "squeezed toward the horizon by a whole-number power. The "
         "defaults are the LightWave 5-7 manual's example values, to be "
         "calibrated against a real render; Rotation does not turn it (a "
         "backdrop symmetric about the vertical)")))
    # ---- R251 C056 Cylinder Sky (Doom): the image rides env_image
    sky_cylinder_repeats: IntProperty(
        name="Repeats", default=4, min=1, max=16,
        description="How many times the sky image goes round a full turn. "
                    "Doom's 256-wide sky repeated four times (1024 columns "
                    "per turn, ANGLETOSKYSHIFT 22); Build skies use their "
                    "own count")
    sky_cylinder_mid: FloatProperty(
        name="Centre Row", default=0.78125, min=0.0, max=1.0,
        description="Fraction of the image's height drawn ABOVE the "
                    "screen's centre line; the rest hangs below. Doom put "
                    "texture row 100 of 128 on the centre line "
                    "(skytexturemid), 0.78125")
    # ---- R251 C100 LightWave Gradient Backdrop: four colours, two squeezes
    lw_zenith: _col("Zenith Color", (0.0, 0.156862745, 0.31372549),
                    "Zenith Color: the colour straight up; LightWave's "
                    "Backdrop blends Sky Color into it above the horizon")
    lw_sky: _col("Sky Color", (0.470588235, 0.705882353, 0.941176471),
                 "Sky Color: the colour all around the horizon's upper edge "
                 "(LightWave Backdrop)")
    lw_ground: _col("Ground Color", (0.196078431, 0.156862745, 0.117647059),
                    "Ground Color: the colour all around the horizon's lower "
                    "edge; there is no blend between it and Sky Color "
                    "(LightWave)")
    lw_nadir: _col("Nadir Color", (0.392156863, 0.31372549, 0.235294118),
                   "Nadir Color: the colour straight down; Ground Color "
                   "blends into it (LightWave Backdrop)")
    lw_sky_squeeze: IntProperty(
        name="Sky Squeeze", default=2, min=1, max=20,
        description="Sky Squeeze: compresses the Sky-to-Zenith blend toward "
                    "the horizon as a whole-number power (LightWave's "
                    "default 2; 20 leaves a thin band at the horizon)")
    lw_ground_squeeze: IntProperty(
        name="Ground Squeeze", default=2, min=1, max=20,
        description="Ground Squeeze: compresses the Ground-to-Nadir blend "
                    "toward the horizon as a whole-number power "
                    "(LightWave's default 2)")
    strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=64.0)
    rotation: FloatProperty(name="Rotation", default=0.0, min=-6.2832, max=6.2832,
                            subtype='ANGLE',
                            description="Spin the whole sky around the "
                                        "vertical axis -- line the sun, "
                                        "clouds and stars up with the shot")
    ambient: _col("Ambient", (0.0, 0.0, 0.0))
    ambient_level: FloatProperty(name="Ambient Level", default=1.0, min=0.0, max=8.0)
    exposure: FloatProperty(
        name="Exposure", default=0.0, min=0.0, max=1.0,
        description="Blender Internal's world Exposure: a soft "
                    "1-exp curve on the lit result (0 with Range 1 "
                    "is off) -- wrld_exposure_correct, verbatim")
    exposure_range: FloatProperty(
        name="Range", default=1.0, min=0.2, max=5.0,
        description="The input value that maps to white under the "
                    "exposure curve, exactly 2.79's Range slider")

    color: _col("Colour", (0.05, 0.05, 0.06))
    horizon: _col("Horizon", (0.55, 0.65, 0.80))
    zenith: _col("Zenith", (0.10, 0.25, 0.65))
    ground_color: _col("Ground", (0.18, 0.15, 0.12))
    show_ground: BoolProperty(name="Ground Plane", default=False,
                              description="Draw the ground half of the dome: "
                                          "everything below the horizon takes "
                                          "the ground colouring instead of sky")
    horizon_height: FloatProperty(name="Horizon Height", default=0.0,
                                  min=-1.0, max=1.0)
    gradient_falloff: FloatProperty(name="Falloff", default=1.0, min=0.01, max=8.0,
                                    description="Higher values keep the horizon "
                                                "colour further up the sky")
    blend_mode: EnumProperty(name="Blend", default='LINEAR', items=_items(
        ('LINEAR', "Linear", ""), ('SMOOTH', "Smooth", ""),
        ('SHARP', "Sharp", ""), ('EASE', "Ease", "")))

    # ------------------------------------------------ R233: the painted sky
    paint_look: EnumProperty(
        name="Look", default='CUSTOM', items=_items(*PAINTED_LOOK_ITEMS),
        update=_painted_look_changed,
        description="A background department's sky, written onto the dials "
                    "below when chosen (Custom writes nothing)")
    paint_angle: FloatProperty(
        name="Panel Direction", default=90.0, min=-360.0, max=360.0,
        description="Where the painting stands, in degrees round from +X "
                    "(90 = +Y, in front of Blender's default camera). Aim "
                    "it at the camera; directions behind it get the plain "
                    "gradient, the back of the stage")
    paint_seed: IntProperty(name="Seed", default=0, min=0, max=9999,
                            description="Random seed for the brushwork -- "
                                        "another painting of the same sky, "
                                        "every stroke falling differently")
    paint_streaks: FloatProperty(
        name="Streaks", default=0.35, min=0.0, max=1.0,
        description="The brush's streaks across the gradient: value noise "
                    "stretched along the stroke direction")
    paint_streak_scale: FloatProperty(
        name="Streak Scale", default=12.0, min=0.5, max=80.0,
        description="How many streaks fit across 45 degrees of view -- "
                    "higher is finer, tighter brushwork")
    paint_streak_angle: FloatProperty(
        name="Stroke Angle", default=0.0, min=-90.0, max=90.0,
        description="The brush direction, degrees off horizontal")
    paint_dabs: FloatProperty(
        name="Dabs", default=0.0, min=0.0, max=1.0,
        description="Impasto dabs over the streaks -- the Paint Strokes "
                    "field on the panel. The costly part of the sky: about "
                    "five times the rest")
    paint_dab_scale: FloatProperty(name="Dab Scale", default=10.0, min=0.5,
                                   max=60.0,
                                   description="How many dabs fit across 45 "
                                               "degrees of view -- higher is "
                                               "smaller, busier impasto")
    paint_clouds: FloatProperty(
        name="Clouds", default=0.35, min=0.0, max=1.0,
        description="Cloud coverage: how much of the deck the shapes fill")
    paint_cloud_scale: FloatProperty(name="Cloud Scale", default=2.2,
                                     min=0.2, max=12.0,
                                     description="How many cloud shapes fit "
                                                 "across 45 degrees of view -- "
                                                 "lower is fewer, grander forms")
    paint_cloud_softness: FloatProperty(
        name="Cloud Softness", default=0.3, min=0.0, max=1.0,
        description="The edge: 0 a dry brush's ragged edge, 1 an airbrush")
    paint_cloud_color: _col("Cloud Colour", (0.97, 0.96, 0.93),
                            "The lit top of a painted cloud, the colour "
                            "the brush laid where the light falls")
    paint_cloud_shadow: _col("Cloud Shadow", (0.58, 0.60, 0.70),
                             "The shadowed underside of a painted cloud, "
                             "mixed in by the cloud's own vertical shape")
    paint_cloud_height: FloatProperty(
        name="Cloud Height", default=0.05, min=-1.0, max=2.0,
        description="Where the deck begins above the horizon, in tangent "
                    "units (0 the horizon, 1 forty-five degrees up)")
    paint_paper: FloatProperty(
        name="Board", default=0.3, min=0.0, max=1.0,
        description="The board's tooth showing through the paint")
    paint_paper_scale: FloatProperty(
        name="Board Scale", default=8.0, min=0.5, max=60.0,
        description="The board's tooth and the granulation: cells across "
                    "45 degrees of view")
    paint_wash: FloatProperty(
        name="Granulation", default=0.25, min=0.0, max=1.0,
        description="Watercolour granulation: pigment settling into the "
                    "tooth, as a pigment density on the colour")

    # ------------------------------------------------------ Bryce's Sky Lab
    sky_mode: EnumProperty(name="Sky Mode", default='CUSTOM', items=_items(
        ('SOFT', "Soft Sky",
         "Bryce's default: the horizon is derived from the sun's glow colour "
         "and only the dome is set by hand, which is why every Bryce sky "
         "warmed toward the sun without anybody choosing to"),
        ('CUSTOM', "Custom Sky", "All three gradient stops set directly")))
    sun_glow_color: _col("Sun Glow Colour", (1.0, 0.86, 0.62),
                         "The corona's own colour, which Bryce kept separate "
                         "from the sun's light colour")
    shadow_color: _col("Shadow Colour", (0.30, 0.34, 0.45),
                       "What the shaded side of a cloud is tinted toward")
    shadow_intensity: FloatProperty(name="Shadow Intensity", default=1.0,
                                    min=0.0, max=1.0)
    fog_base_height: FloatProperty(
        name="Fog Base Height", default=0.0, min=-1.0, max=1.0,
        description="Where the fog bank starts. Below this it is solid, above "
                    "it falls away over the fog height")
    haze_base_height: FloatProperty(
        name="Haze Base Height", default=0.0, min=-1.0, max=1.0,
        description="Where the haze starts. The Sky Lab has this and the Sky "
                    "& Fog palette does not")
    fog_blend_sky: FloatProperty(name="Fog Blend With Sky", default=0.0,
                                 min=0.0, max=1.0)
    fog_sun_tint: FloatProperty(name="Fog Blend With Sun", default=0.0,
                                min=0.0, max=1.0)
    color_perspective: FloatProperty(
        name="Colour Perspective", default=0.0, min=0.0, max=4.0,
        description="How fast distance takes the haze colour. Bryce applied "
                    "this to everything in the scene, not only to the sky")
    volumetric_world: FloatProperty(
        name="Volumetric World", default=0.0, min=0.0, max=4.0,
        description="Shafts of sunlight through the atmosphere, which in Bryce "
                    "was the render-time setting nobody left on by accident")
    cloud_frequency: FloatProperty(
        name="Frequency", default=1.0, min=0.05, max=16.0,
        description="How tight the cloud pattern is. Bryce's own control name")
    cloud_amplitude: FloatProperty(
        name="Amplitude", default=1.0, min=0.0, max=4.0,
        description="How far the pattern swings either side of the cover "
                    "threshold -- which separates billows without changing how "
                    "much sky is covered")
    cloud_turbulence: FloatProperty(name="Turbulence", default=1.0, min=0.05,
                                    max=2.0)
    spherical_clouds: BoolProperty(
        name="Spherical Clouds", default=True,
        description="On, clouds stay puffy out to the horizon; off, they "
                    "stretch into streaks, which is what the switch did")
    link_clouds_to_view: BoolProperty(
        name="Link Clouds to View", default=True,
        description="Keep the cloud pattern fixed relative to the camera, so "
                    "moving does not change which clouds are where. Turn it "
                    "off for real parallax -- but note that Cloud Height is a "
                    "dome parameter rather than a distance, so a low deck "
                    "slides a long way for a small move")
    fixed_cloud_plane: BoolProperty(name="Fixed Cloud Plane", default=True)
    stratus_frequency: FloatProperty(name="Stratus Frequency", default=1.0,
                                     min=0.05, max=16.0)
    stratus_amplitude: FloatProperty(name="Stratus Amplitude", default=1.0,
                                     min=0.0, max=4.0)
    moon_softness: FloatProperty(
        name="Softness", default=0.05, min=0.0, max=1.0,
        description="How hard the terminator is across the moon's disc")
    comets: FloatProperty(
        name="Comet Intensity", default=0.0, min=0.0, max=4.0,
        description="Bryce put comets in the Celestial tab, and they are the "
                    "reason its night skies were never just a starfield")
    comet_count: IntProperty(name="Comets", default=3, min=1, max=32)
    comet_speed: FloatProperty(
        name="Comet Speed", default=0.05, min=0.0, max=4.0,
        description="How fast each comet runs around its own great circle, in "
                    "radians of sky per second. At zero they stand still, as "
                    "they used to. They all start where a still frame puts "
                    "them, so turning this up never empties the first frame")
    comet_length: FloatProperty(
        name="Tail Length", default=0.10, min=0.005, max=1.5,
        description="How far the tail reaches behind the head, as a fraction "
                    "of the sky. Each comet varies either side of it")
    comet_width: FloatProperty(
        name="Tail Width", default=0.006, min=0.0005, max=0.2,
        description="How wide the tail is at the head. It flares out from "
                    "there and dims as it goes, which is the shape a comet's "
                    "dust tail actually has")
    comet_tail_sun: FloatProperty(
        name="Tail Direction", default=0.6, min=0.0, max=1.0,
        description="0 trails the comet's own path, the way a dust tail does; "
                    "1 points straight away from the sun, the way an ion tail "
                    "does. Real comets show both at once, so the truth is in "
                    "between")
    comet_color: _col("Comet Colour", (1.0, 0.96, 0.88))

    sun_elevation: FloatProperty(name="Sun Altitude", default=0.35,
                                 min=-1.5708, max=1.5708, subtype='ANGLE')
    sun_rotation: FloatProperty(name="Sun Azimuth", default=0.6,
                                min=-6.2832, max=6.2832, subtype='ANGLE')
    sun_color: _col("Sun Colour", (1.0, 0.94, 0.82))
    sun_size: FloatProperty(name="Sun Size", default=0.03, min=0.0, max=1.5,
                            subtype='ANGLE')
    sun_intensity: FloatProperty(name="Sun Intensity", default=1.0, min=0.0, max=64.0)
    sun_glow: FloatProperty(name="Sun Glow", default=0.35, min=0.0, max=1.0,
                            description="Width of the halo hugging the sun "
                                        "disc -- the bright inner glow before "
                                        "the corona takes over")
    sun_disc: BoolProperty(name="Sun Disc", default=True)

    celestial: EnumProperty(name="Body", default='SUN', items=_items(
        ('SUN', "Sun", "A bright disc with a corona"),
        ('MOON', "Moon", "A disc with a terminator, so it shows a phase")))
    moon_phase: FloatProperty(
        name="Phase", default=0.25, min=0.0, max=1.0,
        description="0 and 1 are new, 0.5 is full. The terminator sweeps "
                    "across the disc between them")
    moon_color: _col("Moon Colour", (0.86, 0.88, 0.95))
    moon_size: FloatProperty(name="Moon Size", default=0.045, min=0.0, max=1.0,
                             subtype='ANGLE')
    moon_earthshine: FloatProperty(
        name="Earthshine", default=0.06, min=0.0, max=1.0,
        description="Faint light on the unlit part of the disc, reflected back "
                    "off the planet")
    sky_mid: _col("Mid Sky", (0.35, 0.50, 0.78))
    sky_mid_height: FloatProperty(name="Mid Height", default=0.35, min=0.01,
                                  max=0.99)
    use_sky_mid: BoolProperty(
        name="Three-Stop Gradient", default=True,
        description="A third colour between horizon and zenith, as Bryce's "
                    "dome gradient allowed")
    atmosphere_density: FloatProperty(
        name="Atmosphere", default=0.0, min=0.0, max=4.0,
        description="Exponential depth haze over the whole dome, on top of the "
                    "horizon band")
    atmosphere_falloff: FloatProperty(name="Atmosphere Falloff", default=1.0,
                                      min=0.01, max=8.0)
    atmosphere_color: _col("Atmosphere Colour", (0.70, 0.78, 0.90))
    haze_blend_sky: FloatProperty(
        name="Blend With Sky", default=0.5, min=0.0, max=1.0,
        description="How much the haze takes the sky's own colour instead of "
                    "its swatch")
    cloud_wind: FloatProperty(
        name="Wind Speed", default=0.0, min=0.0, max=64.0,
        description="Drifts both cloud decks across the sky over time. Stratus "
                    "moves slower, as height dictates")
    cloud_wind_angle: FloatProperty(name="Wind Direction", default=0.0,
                                    min=-6.2832, max=6.2832, subtype='ANGLE')
    cloud_ambience: FloatProperty(
        name="Ambience", default=0.35, min=0.0, max=1.0,
        description="How much sky light fills the shadowed side of a cloud")
    cloud_shadows: FloatProperty(
        name="Cloud Shadows", default=0.0, min=0.0, max=1.0,
        description="Casts the cumulus deck onto the infinite ground below it, "
                    "sampled from the same noise so a shadow always lands "
                    "under a cloud")
    sun_corona: FloatProperty(name="Corona", default=1.0, min=0.0, max=4.0,
                              description="Strength of the wide outer halo "
                                          "far beyond the glow -- hazy "
                                          "daylight skies wear more of it")

    haze_color: _col("Haze Colour", (0.82, 0.86, 0.92))
    haze_density: FloatProperty(name="Haze", default=0.45, min=0.0, max=1.0,
                                description="Atmospheric perspective. Thickens "
                                            "toward the horizon")
    haze_height: FloatProperty(name="Haze Height", default=0.22, min=0.01, max=2.0)
    haze_sun_tint: FloatProperty(name="Sun Tint", default=0.5, min=0.0, max=1.0,
                                 description="How much the haze takes the sun's "
                                             "colour when looking toward it")
    fog_color: _col("Fog Colour", (0.90, 0.90, 0.88))
    fog_density: FloatProperty(name="Fog", default=0.0, min=0.0, max=1.0,
                               description="Ground-hugging fog, separate from "
                                           "haze as it was in Bryce")
    fog_height: FloatProperty(name="Fog Height", default=0.05, min=0.005, max=1.0)

    clouds: BoolProperty(name="Cumulus", default=True)
    cloud_color: _col("Cloud Colour", (1.0, 1.0, 1.0))
    cloud_shadow: _col("Cloud Base", (0.42, 0.45, 0.55))
    cloud_cover: FloatProperty(name="Cover", default=0.5, min=0.0, max=1.0)
    cloud_density: FloatProperty(name="Opacity", default=0.95, min=0.0, max=1.0)
    cloud_height: FloatProperty(name="Altitude", default=1.0, min=0.05, max=20.0)
    cloud_scale: FloatProperty(name="Frequency", default=1.4, min=0.01, max=64.0,
                               description="Bryce's Cloud Frequency: larger "
                                           "values make bigger, fewer clouds")
    cloud_detail: IntProperty(name="Detail", default=5, min=1, max=10)
    cloud_softness: FloatProperty(name="Fuzziness", default=1.0, min=0.01, max=8.0)
    cloud_thickness: FloatProperty(name="Thickness", default=0.35, min=0.0, max=2.0,
                                   description="Depth of the deck, which drives "
                                               "the self-shadowing on its base")
    cloud_rim: FloatProperty(name="Sun Rim", default=0.4, min=0.0, max=4.0,
                             description="Bright edge where the sun catches the "
                                         "sunward side of a cloud")
    cloud_seed: IntProperty(name="Seed", default=0, min=0, max=9999)

    stratus: BoolProperty(name="Stratus", default=False)
    stratus_color: _col("Stratus Colour", (0.95, 0.95, 0.98))
    stratus_amount: FloatProperty(name="Cover", default=0.45, min=0.0, max=1.0)
    stratus_density: FloatProperty(name="Opacity", default=0.6, min=0.0, max=1.0)
    stratus_altitude: FloatProperty(name="Altitude", default=3.0, min=0.1, max=40.0)
    stratus_scale: FloatProperty(name="Frequency", default=3.0, min=0.01, max=64.0)
    stratus_detail: IntProperty(name="Detail", default=4, min=1, max=10)
    stratus_sharpness: FloatProperty(name="Fuzziness", default=1.4, min=0.01, max=8.0)
    stratus_squash: FloatProperty(name="Streak", default=1.0, min=0.05, max=8.0,
                                  description="Stretches the layer into wind-blown "
                                              "streaks")

    rainbow: BoolProperty(name="Rainbow", default=False)
    rainbow_intensity: FloatProperty(name="Intensity", default=0.35, min=0.0, max=4.0)
    rainbow_radius: FloatProperty(name="Radius", default=42.0, min=5.0, max=90.0,
                                  description="Degrees from the antisolar point. "
                                              "42 is where a real bow sits")
    rainbow_width: FloatProperty(name="Width", default=3.0, min=0.2, max=20.0)
    rainbow_secondary: FloatProperty(name="Secondary Bow", default=0.5,
                                     min=0.0, max=2.0)
    stars: BoolProperty(name="Stars", default=False)
    star_density: FloatProperty(name="Density", default=0.5, min=0.0, max=1.0)
    star_brightness: FloatProperty(name="Brightness", default=0.8, min=0.0, max=4.0)
    star_size: FloatProperty(
        name="Star Size", default=0.35, min=0.01, max=1.0,
        description="Diameter of a star within its cell. Small values give "
                    "single-pixel points, which is what these looked like")
    star_twinkle: FloatProperty(name="Twinkle", default=0.0, min=0.0, max=1.0,
                                description="Animated flicker, hashed per "
                                            "star so each twinkles on its own "
                                            "rhythm; 0 holds them all steady")
    old_stars: BoolProperty(
        name="Old Stars", default=False,
        description="Draw stars the pre-1.38 way: each star fills its whole "
                    "grid cell, so its pixel size follows the render "
                    "resolution (blocky squares at high resolutions, and "
                    "arbitrary sizes in Starfield mode). Off, every star is "
                    "a round point of fixed angular size. On for scenes "
                    "tuned to the old look")
    nebula: FloatProperty(
        name="Nebula", default=0.0, min=0.0, max=4.0,
        description="Turbulent cloud behind the stars. Zero leaves plain space")
    nebula_color: _col("Nebula Colour", (0.35, 0.15, 0.55))
    nebula_scale: FloatProperty(name="Nebula Scale", default=2.0, min=0.05,
                                max=32.0)
    nebula_detail: IntProperty(name="Nebula Detail", default=5, min=1, max=10)

    band_count: IntProperty(
        name="Bands", default=8, min=1, max=64,
        description="How many flat steps the gradient is cut into. A 256-colour "
                    "machine could spare about this many for the sky")
    band_softness: FloatProperty(
        name="Softness", default=0.0, min=0.0, max=1.0,
        description="Rounds the step edges. 0 is the hard band the hardware "
                    "actually gave you")

    turbidity: FloatProperty(name="Turbidity", default=2.5, min=1.0, max=10.0,
                             description="Atmospheric haziness. 2 is a clear "
                                         "day, 6 is city smog")
    ground_albedo: FloatProperty(name="Ground Albedo", default=0.3, min=0.0, max=1.0)

    ground_plane: BoolProperty(
        name="Infinite Ground", default=False,
        description="An endless plane, intersected analytically rather than "
                    "built from geometry. POV-Ray and Bryce both offered one, "
                    "and it costs nothing per frame however far it reaches")
    ground_mode: EnumProperty(name="Surface", default='SOLID', items=_items(
        ('SOLID', "Solid", "One flat colour"),
        ('CHECKER', "Checker", "The infinite chequerboard of a thousand ray "
                               "tracing demos"),
        ('NOISE', "Fractal", "Two colours mixed by fractal noise, for terrain "
                             "seen from height"),
        ('TILES', "Tiles", "Square tiles with grout width, glow and "
                           "per-tile shading dials. Thin glowing grout IS "
                           "the synthwave neon floor (the old Neon Grid "
                           "entry retired into this)"),
        ('DESERT', "Dunes", "Wind-ribbed sand ridges warped by noise, the "
                            "second colour on the crests"),
        ('SNOW', "Snowfield", "A bright field with blue-shadowed hollows "
                              "and sparse sun glints"),
        ('LAVA', "Lava", "Plates of darkened crust split by ridged "
                         "fissures, heat bleeding out of every crack -- "
                         "Crack Width, Glow and Pulse dials"),
        ('OCEAN', "Ocean", "Animated waves reflecting the sky, with a Fresnel "
                           "term so it mirrors at glancing angles"),
        ('MATERIAL', "Material", "The plane wears a material you pick: its "
                                 "node graph is evaluated across the "
                                 "infinite ground, tiled by Scale"),
        # R251 C048, appended at the END (positional numbering)
        ('MODE7', "Mode 7 (SNES / GBA)",
         "A 1024x1024 map of 256 fifteen-bit colours read through per-row "
         "8.8 fixed-point steps from a whole-texel origin, point-sampled, "
         "unlit -- the SNES PPU's BG mode 7 driven per scanline as F-Zero, "
         "Pilotwings and Mario Kart did. Not the Material floor "
         "(perspective-correct per pixel, filtered, lit): per-row fixed "
         "point, 256 colours, unlit, and mirrors do not see it. Needs a "
         "Map image")))
    # ---- R251 C048 Mode 7 floor
    ground_image: PointerProperty(
        name="Mode 7 Map", type=bpy.types.Image,
        description="The map the Mode 7 floor reads: resampled to 1024x1024 "
                    "texels by nearest sampling and cut to 256 colours of "
                    "15-bit RGB once at load, as the SNES's tile map and "
                    "palette held it")
    mode7_texel_size: FloatProperty(
        name="Texel Size", default=1.0, min=0.001, max=1000.0,
        description="Scene units per map texel: how far along the floor one "
                    "Mode 7 texel reaches (the whole 1024-texel map spans "
                    "1024 x this)")
    mode7_over: EnumProperty(
        name="Outside the Map", default='WRAP',
        description="What the SNES PPU drew past the edge of the 1024x1024 "
                    "Mode 7 map (the M7SEL register): the map wrapped, "
                    "nothing (the sky shows), or tile 0 repeated",
        items=_items(
        ('WRAP', "Wrap (M7SEL 0, GBA wrap)",
         "Outside the map the map repeats: the address is taken modulo "
         "1024 in both axes"),
        ('TRANSPARENT', "Transparent (M7SEL 2)",
         "Outside the 1024x1024 map nothing is drawn and the sky shows; "
         "the origin is sign-clipped to the SNES's 13 bits (the GBA's "
         "28-bit origin differs only past 4096 texels, not modelled)"),
        ('TILE0', "Tile 0 (M7SEL 3)",
         "Outside the map the SNES fills with tile 0: the map's top-left "
         "8x8 texels, repeated")))
    ground_material: PointerProperty(
        name="Ground Material", type=bpy.types.Material,
        description="R203: the material the infinite plane wears in "
                    "Material mode -- its node graph paints the ground "
                    "to the horizon")
    ground_grout: FloatProperty(
        name="Grout Width", default=0.04, min=0.0, max=0.45,
        description="Width of the grout line as a fraction of one "
                    "tile; near zero with a glowing colour is the neon "
                    "grid floor")
    ground_tile_shade: FloatProperty(
        name="Tile Variance", default=0.25, min=0.0, max=1.0,
        description="Per-tile brightness scatter so the floor is not "
                    "one flat repeat; 0 makes every tile identical")
    ground_grout_glow: FloatProperty(
        name="Grout Glow", default=1.0, min=0.0, max=8.0,
        description="Multiplies the grout colour; above 1 the lines "
                    "GLOW -- the synthwave floor lives around 1.6")
    ground_crack_width: FloatProperty(
        name="Crack Width", default=0.35, min=0.02, max=2.0,
        description="How wide the lava fissures split, with the heat "
                    "bleed scaling along")
    ground_glow: FloatProperty(
        name="Lava Glow", default=1.0, min=0.0, max=8.0,
        description="Strength of the heat pouring out of the lava "
                    "cracks and embers")
    ground_pulse: FloatProperty(
        name="Heat Pulse", default=0.15, min=0.0, max=1.0,
        description="How much the lava's glow breathes over time; 0 "
                    "holds it steady")
    ground_color3: _col("Third Colour", (1.0, 1.0, 1.0),
                        "The ground's third tone where the surface has "
                        "one: the colour of snow glints and of the "
                        "lava's drifting embers")
    ground_sparkle: FloatProperty(
        name="Sparkle", default=1.0, min=0.0, max=4.0,
        description="Strength of the snowfield's sun glints; 0 turns "
                    "the glitter off entirely")
    ground_ridge: FloatProperty(
        name="Ridge Strength", default=0.6, min=0.0, max=1.0,
        description="How strongly the desert's dune crests pull toward "
                    "the second colour; 0 flattens the ripple away")
    ground_lighting: FloatProperty(
        name="Scene Lighting", default=1.0, min=0.0, max=1.0,
        description="How much the infinite ground answers the scene's "
                    "sun and lamps, including shadows cast by objects; "
                    "0 is the old self-lit flat look")
    ground_height: FloatProperty(name="Height", default=0.0, min=-1e4, max=1e4)
    ground_scale: FloatProperty(name="Scale", default=2.0, min=0.001, max=1e4)
    ground_color2: _col("Second Colour", (0.55, 0.52, 0.48))
    ground_fade: FloatProperty(
        name="Distance Fade", default=60.0, min=0.0, max=1e5,
        description="How far the plane reaches before it has faded into the "
                    "horizon. Without this it reads as a flat sheet rather "
                    "than as ground going away")
    ocean_wind_angle: FloatProperty(name="Wind Direction", default=0.6,
                                    min=-6.2832, max=6.2832, subtype='ANGLE',
                                    description="Compass direction the wind "
                                                "blows from; the wave trains "
                                                "run mostly with it, fanned "
                                                "out by Spread")
    ocean_spread: FloatProperty(
        name="Spread", default=0.6, min=0.0, max=1.0,
        description="How far the shorter waves fan off the wind. 0 is a "
                    "regular swell, 1 is confused chop")
    ocean_wave_scale: FloatProperty(
        name="Wave Size", default=1.0, min=0.02, max=500.0, soft_min=0.05,
        soft_max=40.0, subtype='DISTANCE', unit='LENGTH',
        description="The length of the longest wave train, crest to crest. "
                    "The shorter trains are fractions of it, so this sets the "
                    "size of the whole sea at once. It no longer multiplies "
                    "the ground Scale, which is the chequerboard's and has "
                    "nothing to do with water")
    ocean_detail: IntProperty(name="Wave Detail", default=5, min=1, max=10,
                              description="How many wave trains are summed. "
                                          "More trains break the swell's "
                                          "repetition; each costs a little")
    ocean_sparkle: FloatProperty(
        name="Horizon Shimmer", default=1.0, min=0.0, max=1.0,
        description="Where a pixel covers many waves, take the sample from a "
                    "random point inside it rather than its centre. Sampling "
                    "the centre makes the wave trains beat against the pixel "
                    "grid and fills the distance with moire fringes; this "
                    "turns the same detail into the fine shimmer a Bryce "
                    "ocean has. Turn it down to see the fringes")
    ocean_horizon_smooth: FloatProperty(
        name="Horizon Smoothing", default=0.0, min=0.0, max=1.0,
        description="Fades out waves too small for a pixel to draw cleanly. "
                    "Bryce did none of this -- its water kept every wave to "
                    "the horizon and compressed them into a band of shimmer, "
                    "which is what its pictures look like. Turn this up for "
                    "smoother distant water, at the cost of it going to glass")
    ocean_deep: _col("Deep Colour", (0.03, 0.09, 0.13))
    ocean_shallow: _col("Shallow Colour", (0.06, 0.22, 0.26))
    ocean_glitter: FloatProperty(
        name="Sun Glitter", default=1.0, min=0.0, max=16.0,
        description="The sun's reflection smeared down the wave slopes. Waves "
                    "too small to draw widen the path rather than vanishing, "
                    "which is what makes it spread toward the horizon")
    ocean_glitter_size: FloatProperty(name="Glitter Width", default=0.45,
                                      min=0.01, max=4.0)
    ocean_foam: FloatProperty(
        name="Foam", default=0.0, min=0.0, max=1.0,
        description="Off by default: Bryce had no foam control, and adding one "
                    "unasked would be inventing a feature it did not have")
    ocean_foam_color: _col("Foam Colour", (0.92, 0.95, 0.96))
    ocean_transparency: FloatProperty(name="Transparency", default=0.25,
                                      min=0.0, max=1.0)

    ocean_choppiness: FloatProperty(name="Choppiness", default=0.35, min=0.0,
                                    max=4.0)
    ocean_speed: FloatProperty(name="Wave Speed", default=1.0, min=0.0, max=16.0)
    env_image: PointerProperty(name="Image", type=bpy.types.Image)
    env_mapping: EnumProperty(name="Projection", items=_items(
        ('EQUIRECT', "Equirectangular", "Latitude/longitude panorama"),
        ('MIRRORBALL', "Mirror Ball", "The sphere map of the era"),
        ('SCREEN', "Screen", "")), default='EQUIRECT')
    env_filter: EnumProperty(name="Filter", items=_items(
        ('BILINEAR', "Bilinear", ""), ('NEAREST', "Nearest", "")),
        default='BILINEAR')
    env_tint: _col("Tint", (1.0, 1.0, 1.0))
    # R251 C038: the DS rear-plane depth bitmap
    backdrop_depth_image: PointerProperty(
        name="Backdrop Depth", type=bpy.types.Image,
        description="An image whose red channel is eye-space depth in scene "
                    "units -- the Z pass of a pre-rendered background -- that "
                    "the z-buffer honours: geometry passes behind the parts of "
                    "the background that are nearer, the DS's rear-plane depth "
                    "bitmap and the pre-rendered-background trick. Tiled 1:1. "
                    "Must be a float image (a Z pass) in Non-Color space")
    backdrop_offset_x: IntProperty(
        name="Offset X", default=0, min=-4096, max=4096,
        description="Pixel offset of the tiled backdrop depth bitmap (the DS's "
                    "CLRIMAGE_OFFSET), horizontal")
    backdrop_offset_y: IntProperty(
        name="Offset Y", default=0, min=-4096, max=4096,
        description="Pixel offset of the tiled backdrop depth bitmap (the DS's "
                    "CLRIMAGE_OFFSET), vertical")
    sky_blend: BoolProperty(name="Sky Gradient", default=False,
                            options={'HIDDEN'})
    # ------------------------------------------------ R200: Weather
    weather: EnumProperty(
        name="Weather", default='NONE',
        description="A particle overlay falling in front of the whole "
                    "picture -- it never replaces the sky, it weathers "
                    "it. Deterministic: the same frame is the same "
                    "storm on every render",
        items=_items(
            ('NONE', "None", "No weather overlay; the picture exactly "
                             "as it was"),
            ('RAIN', "Rain", "Streaked drops added as light -- the "
                             "era's sprite rain; pick a green colour "
                             "for acid rain"),
            ('SNOW', "Snow", "Soft flakes composited over the picture; "
                             "raise Drift for the tumble"),
            ('EMBERS', "Embers", "Glowing motes added as light -- set "
                                 "Angle to 180 degrees so they rise, "
                                 "and try Glow and Flicker"),
            ('ASH', "Ash", "Grey flakes composited over the picture, "
                           "the fallout drift; slow Speed suits it")))
    weather_density: FloatProperty(
        name="Density", default=1.0, min=0.0, max=10.0,
        description="How many particles fill the frame; the count "
                    "follows the output size, so density reads the "
                    "same at every resolution")
    weather_size: FloatProperty(
        name="Size", default=1.0, min=0.05, max=8.0,
        description="Particle size, resolution-true against the "
                    "480-line reference like the halo widths")
    weather_speed: FloatProperty(
        name="Speed", default=1.0, min=0.0, max=10.0,
        description="How fast the particles travel across the screen; "
                    "0 hangs them in the air")
    weather_angle: FloatProperty(
        name="Angle", default=0.0, min=-6.2832, max=6.2832,
        subtype='ANGLE',
        description="Direction of travel: 0 falls straight down, 180 "
                    "degrees rises (embers), anything between is the "
                    "diagonal sweep")
    weather_drift: FloatProperty(
        name="Drift", default=0.2, min=0.0, max=4.0,
        description="Sideways wobble across the travel line, each "
                    "particle on its own hashed phase -- snow tumbles, "
                    "rain streaks straight at 0")
    weather_color: _col("Colour", (0.85, 0.90, 1.0),
                        "The particles' colour: cold white for rain "
                        "and snow, sickly green for acid rain, hot "
                        "orange for embers, grey for ash")
    weather_opacity: FloatProperty(
        name="Opacity", default=0.8, min=0.0, max=1.0,
        description="How strongly the weather covers (or lights up) "
                    "what is behind it; 0 switches it off entirely")
    weather_layers: IntProperty(
        name="Layers", default=3, min=1, max=4,
        description="Parallax depth: layer 1 is nearest -- largest, "
                    "fastest, brightest -- and each deeper layer "
                    "recedes")
    weather_streak: FloatProperty(
        name="Streak", default=1.0, min=0.0, max=6.0,
        description="Rain only: how long each drop's stroke trails "
                    "behind it; 0 draws round drops")
    weather_glow: FloatProperty(
        name="Glow", default=0.0, min=0.0, max=1.0,
        description="A soft halo of light around each particle -- "
                    "embers and fireflies want it, snow does not")
    weather_flicker: FloatProperty(
        name="Flicker", default=0.0, min=0.0, max=1.0,
        description="Each particle's brightness pulses on its own "
                    "hashed phase -- the ember twinkle")
    weather_seed: IntProperty(
        name="Seed", default=0, min=0, max=10000,
        description="Re-deals every particle's path; the same seed is "
                    "the same storm, always")


#: tooltips for group properties declared inline above -- patched into the
#: deferred property definitions below, so every control in every panel
#: carries a real explanation without repeating boilerplate at each site
GROUP_DOCS = {
    'HalcyonSettings': {
        'ui_tab': "Which page of Halcyon's settings this panel shows. "
                  "Purely a UI switch; it changes nothing in the render",
    },
    'HalcyonMaterialSettings': {
        'model': "The reflectance model this override shades with -- each "
                 "entry's tooltip names its era and character",
        'diffuse': "The surface's base colour under white light",
        'diffuse_level': "How strongly the diffuse term contributes. 0 "
                         "kills the base colour entirely",
        'specular': "Colour of the highlight. Plastics keep it white; "
                    "period metals tinted it to fake conductor response",
        'specular_level': "Brightness of the highlight. 0 is fully matte",
        'glossiness': "Tightness of the highlight: low is a broad sheen, "
                      "high is a small hard sparkle. The classic Phong "
                      "exponent, under its 3D Studio name",
        'ambient_level': "How much of the scene's ambient light this "
                         "surface accepts. The era's per-material shadow "
                         "filler",
        'emission': "Light the surface gives off by itself, unaffected by "
                    "any lamp -- the Self-Illumination of the period",
        'emission_level': "Multiplier on the emission colour -- how far "
                          "past its base shade the surface glows",
        'opacity': "How solid the surface is. Below 1 it composites "
                   "through the transparency mode the render settings "
                   "chose",
        'ior': "Index of refraction for traced glass. 1.0 bends nothing; "
               "1.45 is glass; 1.33 water",
        'roughness': "Micro-surface roughness for the models that read it "
                     "(Oren-Nayar, Cook-Torrance, Minnaert)",
        'anisotropy': "Stretches the highlight along one direction, for "
                      "brushed metal. 0 is round",
        'aniso_rotation': "Rotates the stretched highlight's direction",
        'metallic': "Blends the surface toward conductor behaviour: "
                    "diffuse falls away and reflections take the base "
                    "colour",
        'reflect_level': "How much traced mirror reflection the surface "
                         "adds. 0 is none; it costs rays above that",
        'two_sided': "Shade back faces as if they were front faces, so "
                     "open geometry has no black inside",
        'shadeless': "Skip lighting entirely and show the diffuse colour "
                     "flat -- the CONSTANT model by another switch",
        'cast_shadow': "Whether this surface blocks light on its way to "
                       "other surfaces",
        'receive_shadow': "Whether other objects' shadows darken this "
                          "surface",
        'wire': "Draw this material's triangle edges over its shading, "
                "per-material rather than scene-wide",
        'wire_size': "Width of this material's inked edges, in rendered "
                     "pixels",
    },
    'HalcyonLightSettings': {
        'decay': "How this light dims with distance: physically correct "
                 "inverse square, the gentler inverse, the era's none, or "
                 "a custom start/end ramp",
        'decay_start': "Distance where the custom ramp starts dimming. "
                       "Closer than this is full brightness",
        'decay_end': "Distance where the custom ramp reaches zero",
        'shadow': "This light's own shadow method: depth map, traced "
                  "rays, or none -- overriding the scene default when "
                  "the scene is set to Per Light",
        'shadow_softness': "Blur of this light's mapped shadow, in "
                           "shadow-map texels. 0 is hard",
        'shadow_samples': "Taps or rays per pixel for this light's soft "
                          "shadows",
        'shadow_density': "How dark this light's shadows get. 1 is full "
                          "occlusion; less lets colour bleed through, as "
                          "the era's fake fill lights did",
        'shadow_color': "Colour inside this light's shadows instead of "
                        "black -- the tinted-shadow trick of period art",
        'diffuse_only': "This light affects only the diffuse term, "
                        "leaving highlights untouched",
        'specular_only': "This light affects only highlights, adding "
                         "sparkle without lifting the surface",
        'ambient_only': "This light adds flat ambient everywhere instead "
                        "of directional light",
        'exclude_mode': "Whether the collection below is kept OUT of this "
                        "light, or is the only thing kept IN it",
    },
    'HalcyonWorldSettings': {
        'mode': "What the sky IS: the material node tree, a flat colour, "
                "a gradient, banded steps, a starfield, the Bryce sky "
                "lab, a physical atmosphere, or an HDRI image",
        'strength': "Multiplier on everything the sky contributes -- "
                    "background, ambient and reflections together",
        'ambient': "Colour of the light the world adds from every "
                   "direction, independent of any lamp",
        'ambient_level': "Multiplier on the world's ambient colour",
        'color': "The flat background colour in Solid mode",
        'horizon': "Sky colour at the horizon in Gradient and Bands "
                   "modes",
        'zenith': "Sky colour straight up in Gradient and Bands modes",
        'ground_color': "Colour below the horizon when Show Ground is on",
        'horizon_height': "Vertical position of the horizon line, as the "
                          "ray direction's Z. 0 is level with the camera",
        'blend_mode': "The shape of the horizon-to-zenith blend: linear, "
                      "smooth, sharp or eased",
        'sky_mode': "Bryce's Sky Mode: Soft Sky derives the dome from the "
                    "sun's own colour; Custom Sky exposes the three "
                    "colour stops directly",
        'shadow_intensity': "Bryce's shadow-strength slider: how dark the "
                            "sky-lab lighting draws its shadows",
        'fog_blend_sky': "How much the fog band takes its colour from the "
                         "sky rather than its own",
        'fog_sun_tint': "How much the fog band warms toward the sun's "
                        "colour near the sun",
        'cloud_turbulence': "How hard the cloud noise is folded. Bryce's "
                            "third cloud control: higher is stormier",
        'fixed_cloud_plane': "Anchor the cloud pattern to the world "
                             "rather than the view, so orbiting the "
                             "camera does not slide the deck",
        'stratus_frequency': "Tightness of the stratus layer's pattern",
        'stratus_amplitude': "Contrast swing of the stratus layer about "
                             "its cover threshold",
        'comet_count': "How many comets streak the celestial sphere",
        'comet_color': "Colour of the comet heads and the tails they "
                       "drag across the celestial sphere",
        'sun_elevation': "Height of the sun above the horizon, in "
                         "radians. Near 0 is sunset; 1.57 is overhead",
        'sun_rotation': "Compass direction of the sun, in radians",
        'sun_color': "Colour of the sun disc and the light it throws "
                     "into the sky model",
        'sun_size': "Angular size of the visible sun disc. The real sun "
                    "spans about half a degree; bigger reads as cinema",
        'sun_intensity': "Brightness of the sun's contribution to the "
                         "dome",
        'sun_disc': "Whether the sun itself is drawn, or only its light",
        'celestial': "Master switch for the celestial layer: moon, "
                     "stars, comets",
        'moon_color': "Colour of the moon disc drawn on the night dome",
        'moon_size': "Angular size of the moon disc on the night dome; "
                     "the real moon spans about half a degree",
        'sky_mid': "Custom Sky's middle colour stop, between horizon "
                   "and zenith",
        'sky_mid_height': "Where the middle stop sits between horizon "
                          "(0) and zenith (1)",
        'atmosphere_falloff': "How quickly the sky colour transitions "
                              "happen with altitude. Higher hugs the "
                              "horizon",
        'atmosphere_color': "Overall atmospheric tint layered over the "
                            "dome",
        'cloud_wind_angle': "Compass direction the cloud decks drift, "
                            "in radians",
        'haze_color': "Colour of the haze band hugging the horizon, "
                      "lightening the dome where it meets the ground",
        'haze_height': "Vertical thickness of the haze band -- how far "
                       "up the dome the horizon's milkiness climbs",
        'fog_color': "Colour of the Bryce fog band at the horizon",
        'fog_height': "Vertical thickness of the Bryce fog band",
        'clouds': "Master switch for the cumulus cloud deck",
        'cloud_color': "Colour of the cumulus deck's sunlit faces",
        'cloud_shadow': "How darkly the deck shades its own undersides",
        'cloud_cover': "How much of the sky the cumulus deck covers. "
                       "Bryce's cover slider",
        'cloud_density': "How solid each cloud reads against the sky "
                         "behind it",
        'cloud_height': "Altitude of the cumulus deck on the dome. "
                        "Lower decks race past; higher ones sit still",
        'cloud_detail': "Noise octaves in the cloud pattern. More is "
                        "crinklier and slower",
        'cloud_softness': "Softness of the cloud edges against the sky",
        'cloud_seed': "Random seed for the cloud pattern. Change it for "
                      "a different sky with the same settings",
        'stratus': "Master switch for the high thin stratus layer",
        'stratus_color': "Colour of the high thin stratus wisps drawn "
                         "over the dome",
        'stratus_amount': "Coverage of the stratus layer: how much of "
                          "the sky its thin wisps veil",
        'stratus_density': "How solid the stratus wisps read against "
                           "the sky -- low is a breath, high a sheet",
        'stratus_altitude': "Altitude of the stratus layer on the dome",
        'stratus_scale': "Size of the stratus features across the dome "
                         "-- lower stretches them into long banners",
        'stratus_detail': "Noise octaves in the stratus pattern. More "
                          "is stringier and a little slower",
        'stratus_sharpness': "Edge hardness of the stratus wisps: soft "
                             "veils at 0, torn-paper streaks at 1",
        'rainbow': "Draw a rainbow opposite the sun, as Bryce could",
        'rainbow_intensity': "Brightness of the rainbow arc against the "
                             "sky -- keep it faint to sit in the light",
        'rainbow_width': "Angular width of the rainbow band -- the real "
                         "bow spans about two degrees of sky",
        'rainbow_secondary': "Strength of the fainter, colour-reversed "
                             "outer bow",
        'stars': "Master switch for the star layer of night skies",
        'star_density': "How many stars fill the celestial sphere -- "
                        "from a handful of bright ones to a deep field",
        'star_brightness': "Brightness of the star points against the "
                           "night dome, before any twinkle plays on them",
        'nebula_color': "Colour of the faint nebula wash behind the "
                        "stars",
        'nebula_scale': "Size of the nebula wash's billows behind the "
                        "stars -- lower is one broad band of colour",
        'nebula_detail': "Noise octaves in the nebula wash. More folds "
                         "finer filaments into the colour",
        'ground_albedo': "How much light the physical atmosphere's "
                         "ground bounces back into the sky",
        'ground_mode': "What the infinite ground plane is made of -- "
                       "each entry is its own material with its own "
                       "controls",
        'ground_height': "World height of the infinite ground plane",
        'ground_scale': "Feature size of the ground material's pattern",
        'ground_color2': "The ground material's secondary colour, where "
                         "its pattern uses one",
        'ocean_deep': "Water colour looking straight into deep water, "
                      "where the surface stops reflecting sky",
        'ocean_shallow': "Water colour near the surface and crests",
        'ocean_glitter_size': "Size of the sun-glitter sparkles on the "
                              "water",
        'ocean_foam_color': "Colour of the foam along the wave crests, "
                            "brightening where the water breaks",
        'ocean_transparency': "How much the water lets the sky's "
                              "reflection give way to its own colour",
        'ocean_choppiness': "How steep and broken the waves are. Calm "
                            "rollers at 0, whitecap-ready peaks at full",
        'ocean_speed': "How fast the waves animate over frames -- a "
                       "multiplier on the sea's clock; 0 freezes it",
        'env_image': "The image used as the world in HDRI mode",
        'env_mapping': "How the image wraps the sphere: equirectangular, "
                       "mirror ball, or screen-locked",
        'env_filter': "Filtering used when sampling the environment "
                      "image",
        'env_tint': "Colour multiplied over the environment image",
        'sky_blend': "Blend the lower sky toward the horizon colour for "
                     "a softer meeting with the ground",
    },
}


def _apply_group_docs():
    """Patch GROUP_DOCS into the deferred property definitions.

    Works on both the real bpy (property functions return a deferred with
    `.keywords`) and the test fake (`_Prop.kw`), and only fills holes --
    an inline description written at the definition site always wins.
    """
    for cls in (HalcyonSettings, HalcyonMaterialSettings,
                HalcyonLightSettings, HalcyonWorldSettings):
        docs = GROUP_DOCS.get(cls.__name__) or {}
        for pname, desc in docs.items():
            ann = cls.__annotations__.get(pname)
            if ann is None:
                continue
            for attr in ('keywords', 'kw'):
                kw = getattr(ann, attr, None)
                if isinstance(kw, dict) and not kw.get('description'):
                    kw['description'] = desc


_apply_group_docs()


CLASSES = (HalcyonSettings, HalcyonMaterialSettings, HalcyonLightSettings,
           HalcyonWorldSettings)


def register():
    for c in CLASSES:
        bpy.utils.register_class(c)
    bpy.types.Scene.halcyon = bpy.props.PointerProperty(type=HalcyonSettings)
    bpy.types.Material.halcyon = bpy.props.PointerProperty(
        type=HalcyonMaterialSettings)
    bpy.types.Light.halcyon = bpy.props.PointerProperty(type=HalcyonLightSettings)
    bpy.types.World.halcyon = bpy.props.PointerProperty(type=HalcyonWorldSettings)


def unregister():
    for attr, owner in (('halcyon', bpy.types.World), ('halcyon', bpy.types.Light),
                        ('halcyon', bpy.types.Material), ('halcyon', bpy.types.Scene)):
        if hasattr(owner, attr):
            try:
                delattr(owner, attr)
            except Exception:                                   # noqa: BLE001
                pass
    for c in reversed(CLASSES):
        try:
            bpy.utils.unregister_class(c)
        except Exception:                                       # noqa: BLE001
            pass
