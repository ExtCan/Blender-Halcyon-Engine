"""Reflectance models.

Every model here is the actual published formulation, evaluated per fragment on
flat numpy arrays. Gouraud and Flat are *interpolation* rates rather than
reflectance models -- they are handled by the renderer's shading-rate machinery
(vertex-rate and face-rate shading respectively) and reuse whichever reflectance
model the material asks for, which is historically correct.

Each model returns (diffuse_weight, specular_weight):
  diffuse_weight  (N,)  or (N,3)
  specular_weight (N,)  or (N,3)
both already including the N.L cosine term where the model calls for it.
"""

import numpy as np

from . import mathx as M

EPS = 1e-6

DIFFUSE_MODELS = ('LAMBERT', 'OREN_NAYAR', 'MINNAERT', 'TOON_DIFFUSE', 'FUJII')
SPECULAR_MODELS = ('PHONG', 'BLINN_PHONG', 'BLINN', 'COOK_TORRANCE', 'WARD',
                   'ANISOTROPIC', 'TOON_SPEC', 'METAL', 'STRAUSS', 'NONE')

MODEL_ITEMS = (
    ('LAMBERT', "Lambert",
     "Pure diffuse, no highlight at all. Brightness depends only on the angle "
     "to the light, so surfaces read as chalk or matte paint. The oldest model "
     "there is (1760) and the default for anything that should not shine"),
    ('GOURAUD', "Gouraud",
     "A shading RATE, not a reflectance model: lighting is evaluated once per "
     "vertex and the colour interpolated across the triangle (1971). Gives the "
     "banding and the sliding highlights of console-era hardware. Coarse "
     "geometry shows it most"),
    ('FLAT', "Flat / Faceted",
     "A shading RATE: one lighting evaluation for the whole polygon, so every "
     "face is a single flat colour. The look of very early hardware and of "
     "un-smoothed low-polygon models"),
    ('PHONG', "Phong",
     "Interpolated normals with a (R.V)^n highlight (1975). The workhorse of "
     "1990s software -- Infini-D, 3D Studio, LightWave. A tight round "
     "highlight that blows out to white readily"),
    ('BLINN_PHONG', "Blinn-Phong",
     "Phong's highlight computed from the half-vector instead (1977). Broader "
     "and softer than Phong at the same Glossiness, and better behaved at "
     "grazing angles. What most hardware actually implemented"),
    ('BLINN', "Blinn (microfacet)",
     "Torrance-Sparrow microfacets with Fresnel, as 3D Studio MAX shipped it. "
     "The highlight brightens toward the edges of a surface, which reads as "
     "polished plastic or coated metal"),
    ('COOK_TORRANCE', "Cook-Torrance",
     "Beckmann microfacet distribution with geometric masking (1982). The "
     "physically-grounded option: Roughness drives it rather than Glossiness, "
     "and metals keep their colour in the highlight"),
    ('OREN_NAYAR', "Oren-Nayar",
     "Rough diffuse (1994). Surfaces stay bright toward their edges instead of "
     "falling off, which is what makes clay, plaster, dust and unglazed "
     "ceramic look right. Driven by Roughness"),
    ('MINNAERT', "Minnaert",
     "Diffuse with darkening or limb-brightening controlled by Roughness "
     "(1941). Originally for the Moon; useful for velvet, dusty surfaces and "
     "anything with a soft rim"),
    ('WARD', "Ward",
     "Anisotropic Gaussian on the slope distribution (1992). A physically "
     "grounded stretched highlight -- the model to reach for on hair, satin "
     "and machined metal when the shape of the streak matters"),
    ('ANISOTROPIC', "Anisotropic",
     "3D Studio's elliptical highlight: a Blinn lobe given two exponents so it "
     "stretches. Brushed metal and vinyl records. With Anisotropy at 0 it is "
     "Blinn-Phong, which is exactly what the original did"),
    ('METAL', "Metal",
     "The period metal shader: the highlight takes the diffuse colour instead "
     "of the light's, so gold stays gold in its reflection. Diffuse is "
     "suppressed. Gives chrome and brass without needing reflections"),
    ('STRAUSS', "Strauss",
     "Parameterised by metalness and glossiness rather than by lobes (1990). "
     "An early attempt at the controls PBR later settled on, and the easiest "
     "of the metal models to dial in"),
    ('MULTI_LAYER', "Multi-Layer",
     "Two independent specular lobes: a tight one over a broad one. Car paint, "
     "lacquer, and anything with a clear coat over a base"),
    ('TOON', "Toon",
     "Diffuse quantised into bands with a hard-edged highlight. Toon Size sets "
     "where the terminator falls and Toon Smooth how sharp it is. Cel "
     "animation looks"),
    ('TRANSLUCENT', "Translucent",
     "Lambert plus a back-side lobe, so light coming from behind shows "
     "through. Paper, leaves, lampshades, thin fabric. Driven by Translucency, "
     "and identical to Lambert while that is 0"),
    ('CONSTANT', "Constant / Shadeless",
     "Unlit: emits its Diffuse Colour flat, ignoring every light in the scene. "
     "For skyboxes, UI elements, self-lit panels and anything that must not "
     "receive shading"),
    ('WIREFRAME', "Wireframe",
     "Draws the triangle edges only and leaves the rest of the surface "
     "see-through. Width comes from the material's Wire Size"),
    ('BI_COOKTORR', "CookTorr (Blender Internal)",
     "Blender Internal's default highlight, transcribed from 2.79: a "
     "half-vector lobe raised to Glossiness (Hardness), divided by "
     "(0.1 + N.V) so it brightens toward grazing view. The renderer "
     "classic .blend files were lit for; legacy imports use it"),
    ('BI_PHONG', "Phong (Blender Internal)",
     "Blender Internal's Phong, transcribed from 2.79 -- which was "
     "always the HALF-VECTOR lobe pow(N.H, Hardness), not the "
     "reflection-vector Phong of the textbooks. Legacy imports use it "
     "for materials that chose Phong"),
    ('BI_BLINN', "Blinn (Blender Internal)",
     "Blender Internal's Blinn, transcribed from 2.79: Torrance-"
     "Sparrow geometry with BI's own refraction-index Fresnel and a "
     "Gaussian half-angle lobe whose width comes from Hardness. IOR "
     "is BI's Refr slider. Legacy imports use it for Blinn materials"),
    ('ANIME', "Anime / Cel",
     "The 2000s-2020s cel look: N.L wrapped to 0..1, cut into two or "
     "three hard-edged tone bands whose SHADOW COLOURS multiply the "
     "base (a shadow is a colour, not a darkness), with a stepped "
     "highlight. The dedicated Anime Shader node carries the full "
     "control set and the game compatibility modes; picking this on "
     "the master shader runs the same bands at their defaults"),
    ('CARTOON', "Cartoon (Paint)",
     "The Western cel: the colour is PAINT, not light -- flat, the "
     "same under every lamp -- with one painted shadow tone where the "
     "key lamps do not reach (a transparent shadow cel over the paint, "
     "or a second flat paint), a painted highlight dot, and a shadow "
     "shape smoothed toward a sphere the way an inker simplifies. The "
     "Cartoon Shader node carries the controls and the era presets; "
     "picking this on the master shader runs it at its defaults"),
    ('OREN_NAYAR_BLINN', "Oren-Nayar-Blinn",
     "3ds Max's matte shader: Oren-Nayar's rough diffuse (Roughness) "
     "under 3D Studio MAX's Blinn highlight (Glossiness, IOR) -- fabric, "
     "terra cotta, clay with a sheen. The one Max shader Halcyon lacked "
     "until R242; with Roughness at 0 it is plain Blinn"),
    # ---- R243: 3ds Max's eight Standard shaders on Max's own light
    # loops (the field's ports of shaders/stdmtl2), appended so no
    # code moves. Glossiness and Specular Level read as Max's percent
    # over 100; Soften is Max's; the frame is world Z projected onto
    # the surface (Max's object Z), turned by Anisotropic Rotation
    ('MAX_PHONG', "Phong (3ds Max)",
     "3ds Max's Phong exactly: the reflected eye ray against the light "
     "raised to 2^(10 Glossiness), Soften folding the cosine before the "
     "power, Lambert beneath. The Standard material's plastic"),
    ('MAX_BLINN', "Blinn (3ds Max)",
     "3ds Max's Blinn exactly: the half-vector cosine raised to "
     "4 x 2^(10 Glossiness), Soften folding it first, Lambert beneath. "
     "The Standard material's default shader"),
    ('MAX_METAL', "Metal (3ds Max)",
     "3ds Max's Metal exactly: Cook-Torrance with the Beckmann slope "
     "1 - Glossiness, the geometric term, and a Fresnel from the "
     "diffuse colour's own intensity, so the highlight takes the "
     "metal's colour; the diffuse dims by Specular Level"),
    ('MAX_ANISOTROPIC', "Anisotropic (3ds Max)",
     "3ds Max's Anisotropic exactly: the Gaussian highlight stretched "
     "by Anisotropy along a frame turned by Anisotropic Rotation, "
     "Glossiness its width, Diffuse Level the Lambert beneath"),
    ('MAX_MULTI_LAYER', "Multi-Layer (3ds Max)",
     "3ds Max's Multi-Layer exactly: Max's Oren-Nayar diffuse "
     "(Roughness) under two anisotropic highlights -- the first from "
     "Specular Color / Level / Glossiness / Anisotropy / Rotation, the "
     "second from their '2' sockets, the second showing through what "
     "the first leaves"),
    ('MAX_OREN_NAYAR_BLINN', "Oren-Nayar-Blinn (3ds Max)",
     "3ds Max's Oren-Nayar-Blinn exactly: Max's Oren-Nayar diffuse "
     "with its interreflection term (Roughness, Diffuse Level) under "
     "Max's Blinn highlight (Glossiness, Soften)"),
    ('MAX_STRAUSS', "Strauss (3ds Max)",
     "3ds Max's Strauss exactly: Glossiness and Metalness alone -- the "
     "highlight 3/(1 - Glossiness) sharp, its colour sliding from the "
     "light's to the diffuse by Metalness and Strauss's Fresnel, the "
     "diffuse dimmed by both; Opacity is Strauss's transparency. "
     "Specular Level does not apply (Max has none here)"),
    ('MAX_TRANSLUCENT', "Translucent (3ds Max)",
     "3ds Max's Translucent exactly: Blinn under Lambert, and the "
     "Translucent Color lit by every lamp's light from either side, "
     "darkening where the diffuse is already lit -- Max's leaf and "
     "lampshade shader (the hemisphere sums are taken per lamp here)"),
    # R251 (LIGHT-B2): the console light units, items 32..35 -- appended
    # so no saved model index moves (gpu/shade._model_index is positional)
    ('GX_LIGHT', "GameCube GX (2001)",
     "The GameCube's fixed-function light unit exactly: Lambert diffuse "
     "plus GX's rational 'shininess' highlight (N.H)^2 / (s/2 + (1 - s/2)"
     "(N.H)^2) with s = Glossiness, the half vector against the camera "
     "axis, specular from Sun lamps only, and the lit colour saturated "
     "and written as 8 bits per channel as the vertex unit did. Meant "
     "for the Gouraud rate: GX never lit per pixel"),
    ('SEGA_MODEL2', "Sega Model 2 (1993)",
     "Sega's Model 2 board exactly: Lambert diffuse and a highlight on "
     "the reflection's component along the camera axis (2(N.L)N.z - L.z, "
     "no per-pixel eye vector), raised to 1, 2, 4 or 8 by repeated "
     "squaring -- Glossiness snaps to the nearest of the four. Shaded "
     "once per polygon, as the board did, then the 64-step luma ramp: "
     "the lit term collapses to one luminance quantised to 64 steps "
     "BEFORE it meets colour, and each 5-bit channel of the polygon "
     "colour is looked up through a linear 64-entry ramp (the games "
     "wrote their own; disclosed) -- so Daytona's polygons band at the "
     "same 64 rungs whatever their hue"),
    ('SEGA_MODEL3', "Sega Model 3 (1996)",
     "Sega's Model 3 board's smooth polygons exactly: Lambert diffuse "
     "and a highlight keyed to N.L raised to 8, 16, 32 or 64 by "
     "squaring, times the board's 1.6/1.6/2.4/3.2 gain for each step -- "
     "Glossiness snaps to the nearest exponent. Lit per vertex, one "
     "sun, no per-pixel eye vector, then the 64-step luma ramp: the "
     "lit term collapses to one luminance quantised to 64 steps BEFORE "
     "it meets colour, and each 5-bit channel of the polygon colour is "
     "looked up through a linear 64-entry ramp (the games wrote their "
     "own; disclosed), the 64 rungs interpolated across the polygon"),
    ('DS_FIXED', "Nintendo DS (2004)",
     "The DS light unit exactly: Lambert diffuse and a highlight from "
     "the unnormalised half vector of the light and the fixed line of "
     "sight (0,0,-1), squared, then read through a 128-entry 8-bit "
     "shininess table shaped by Glossiness (the games authored theirs; "
     "the entry curve here is (i/128)^(Glossiness/8), so Glossiness 8 "
     "is a linear table and 64 a tight pin), the lit colour saturated "
     "to 5 bits per channel. Lit per vertex, four directional lights; "
     "at the Vertex and Face shading rates the lit colour then meets "
     "the texel through the DS's +1 modulate on 6-bit channels, "
     "GBATEK's texture blend (a per-pixel scene shades the light unit "
     "alone)"),
    # ---- R251 material pack: period combiners (each a shading RATE plus
    # the machine's combine arithmetic), items 36..48 -- appended after the
    # lighting pack's four so no saved model index moves (MAT-A)
    ('FLAT_GL_LAST', "Flat, last vertex (OpenGL 1.x / PSP)",
     "A shading RATE like Flat, but the machine's: the whole triangle takes "
     "the LAST corner's Gouraud lighting with that corner's smooth normal "
     "-- OpenGL 1.0-1.5 GL_FLAT and the PSP's GU_FLAT. A quad splits along "
     "its diagonal and a lit grid shows the diamond pattern; the centroid "
     "Flat above cannot"),
    ('FLAT_D3D_FIRST', "Flat, first vertex (Direct3D 3-9)",
     "A shading RATE like Flat, but Direct3D's D3DSHADE_FLAT: the whole "
     "triangle takes the FIRST corner's Gouraud lighting with that corner's "
     "smooth normal, so the two halves of a quad take different corners and "
     "a lit surface shows the 1996-2005 PC diagonal split"),
    ('PS1_MODULATE', "Gouraud x/128 (PlayStation / PS2 GS / PSP, 1994)",
     "A shading RATE plus the PlayStation GPU's texture blend: the lit "
     "colour becomes an 8-bit vertex colour with 80h meaning 1.0, the texel "
     "is multiplied by it and shifted right 7, saturating at FFh -- so a "
     "lit vertex up to 2.0 brightens the texture twofold and clips flat. "
     "The PS2 GS (>>7, 0x80 = 1.0) and the PSP's TFX modulate are the same "
     "arithmetic"),
    ('PS2_HIGHLIGHT', "Highlight (PlayStation 2 GS, 2000)",
     "A shading RATE plus the Graphics Synthesizer's HIGHLIGHT texture "
     "function: the diffuse light becomes an 8-bit vertex colour with 0x80 "
     "= 1.0, multiplied into the texel and shifted right 7, then the vertex "
     "ALPHA -- the specular sum, always white on a PS2 -- is added to all "
     "three channels and truncated at 0xFF, so highlights float white over "
     "any texture"),
    ('SATURN_ADD', "Gouraud add (Sega Saturn VDP1, 1994)",
     "A shading RATE plus the VDP1's Gouraud table: the lit colour becomes "
     "a 5-bit value with 10h neutral, and (value - 16) is ADDED to each "
     "5-bit channel of the texel, clamped 0..31 -- 00h darkens by 16 steps, "
     "1Fh brightens by 15, so shading pushes a texture past its own "
     "brightness instead of scaling it"),
    ('N64_COMBINE', "Colour combiner (Nintendo 64 RDP, 1996)",
     "A shading RATE plus the RDP colour combiner's first cycle (TEXEL0 - "
     "0) * SHADE + 0 on 8-bit values: ((t * s) + 0x80) >> 8, no headroom "
     "above 1.0 and a rounding term the PS1 lacks, so full light leaves "
     "bright texels one step short of themselves -- the N64's own quiet "
     "darkening"),
    ('S22_MODULATE', "Gouraud x/64 (Namco System 22, 1993)",
     "A shading RATE plus the System 22's vertex brightness with unity at "
     "0x40: the texel is multiplied by an 8-bit shade and shifted right 6, "
     "saturating -- near-4x headroom, so Ridge Racer's lit polygons burn to "
     "white long before a PlayStation's would (read from MAME's arithmetic, "
     "not a Namco document)"),
    ('D3D_SEPARATE_SPEC',
     "Gouraud, separate specular (Direct3D 5-7 / OpenGL 1.2 / Dreamcast, 1996)",
     "A shading RATE plus the fixed-function OFFSET colour: the texture "
     "modulates the interpolated diffuse light only, and the interpolated "
     "specular sum is added AFTER the texture, clamped at 1 -- so highlights "
     "float white over dark textures the way every 1996-2001 consumer card "
     "and the Dreamcast's offset colour drew them. Halcyon's plain Gouraud "
     "multiplies the highlight into the texel instead"),
    ('PCX_INTENSITY', "Intensity Gouraud (PowerVR PCX1 / PCX2, 1996)",
     "A shading RATE plus the PCX's monochrome interpolation: each triangle "
     "takes one base colour (the mean of its three lit corners) and "
     "interpolates only a scalar intensity -- corner luminance over the "
     "base's luminance -- so two coloured lamps meeting on one polygon "
     "become a grey ramp over an average tint. Textures multiply as usual. "
     "Rectified in PowerVR Series 2"),
    ('DS_TOON', "Toon table (Nintendo DS, 2004)",
     "A shading RATE plus the DS's mode-2 toon: the lit colour's RED "
     "channel (ambient, specular and emission included) becomes a 5-bit "
     "index into a 32-entry table that REPLACES the vertex colour before "
     "the texel modulates it; green and blue of the lighting are "
     "discarded. The table is this material's Toon Size and Toon Steps "
     "sampled at 32 stops with hard edges (the DS had no smooth band) and "
     "quantised to 15 bits"),
    ('DS_HIGHLIGHT', "Highlight table (Nintendo DS, 2004)",
     "A shading RATE plus the DS's mode-2 highlight: the same 32-entry "
     "table indexed by the lit RED channel, but the entry is modulated with "
     "the texel AND added again, truncated at 63 -- GBATEK's highlight "
     "mode, the glossy toon of DS racers. With the default two-step toon "
     "table every lit entry is 63 and the add saturates: lit surfaces go "
     "pure white whatever the texture. The texel shows only at table "
     "entries strictly between 0 and 63 (entry 0 is black), which takes "
     "more Toon Steps together with lamps dimmed off the top entry -- "
     "dimmed too far, the surface falls to entry 0"),
    ('MEGA_DRIVE_SH', "Shadow/Highlight (Sega Mega Drive VDP, 1988)",
     "A shading RATE (one class per polygon) plus the VDP's shadow/highlight "
     "mode: the albedo is crushed to 3 bits per channel and read through "
     "one of three measured DAC ramps -- NORMAL, SHADOW (the key lamp's lit "
     "fraction below 1/2; the key is the scene's first lamp, Halcyon's own "
     "rule, and a light limit that drops the first lamp shadows everything) "
     "or HIGHLIGHT (a specular sum of 1/2 or more) -- so darkening is a "
     "table, not a multiply, the way the Genesis 3D racers and Vectorman "
     "shaded"),
    ('SUPERFX_PLOT', "Dither plot (Super FX / Star Fox, 1993)",
     "A shading RATE (one colour per polygon) plus the Super FX PLOT "
     "dither: the face's lit colour is matched by the closest PAIR of "
     "entries of the frame's fixed palette (16 colours or fewer, the chip's "
     "4-bpp mode) and the face is filled with a 50% checkerboard of the two "
     "at output-pixel pitch, before any post. Needs a fixed palette of at "
     "most 16 entries, gamma 1 and no display transform, or it shades flat "
     "and says so"),
)

#: R251 material pack (MAT-A C034): the DS toon / highlight items light
#: with the DS_FIXED lobe (the mode-2 polygon's light unit is the DS's
#: one light unit) -- consulted as the first line of `evaluate` and by
#: light_surface, so the fixed viewer, the table highlight and the 5-bit
#: step all apply under the alias
LOBE_ALIAS = {'DS_TOON': 'DS_FIXED', 'DS_HIGHLIGHT': 'DS_FIXED'}

#: R251 (LIGHT-B2): models that force the camera-axis viewer inside
#: themselves (GL 1.1's infinite viewer, the Sega boards' R.z, the DS's
#: (0,0,-1) line of sight) -- light_surface hands them the axis whatever
#: `specular_viewer` says, and the plan registers the axis texel for them
AXIS_MODELS = frozenset({'GX_LIGHT', 'SEGA_MODEL2', 'SEGA_MODEL3',
                         'DS_FIXED'})
# (R251 material pack, MAT-A C034: DS_TOON / DS_HIGHLIGHT reach this set
# through LOBE_ALIAS -- light_surface and evaluate alias them to DS_FIXED
# before reading it -- so the set itself stays the lighting pack's four)
#: R251 (LIGHT-B2): models that ARE their shading rate, as GOURAUD / FLAT
#: are (render.RATE_FOR_MODEL merges this table): Model 2 shaded one luma
#: per polygon, Model 3 per vertex. Neither reaches GLSL: the corner road
#: (the CPU lights the corners, the pass interpolates) is the machine on
#: both devices, so a PIXEL-rate request refuses by name through
#: gpu/shade.UNSUPPORTED_MODELS, which merges the set below.
RATE_FOR_MODEL = {'SEGA_MODEL2': 'FACE', 'SEGA_MODEL3': 'VERTEX'}
UNSUPPORTED_MODELS = frozenset({'SEGA_MODEL2', 'SEGA_MODEL3'})


def axis_viewer(view):
    """R251 (LIGHT-B2): the fixed viewer of the console light units --
    the camera's own +Z row of the CPU's view matrix (`view = inv(mw)`),
    normalised: the world-space direction toward a viewer at infinity.
    OpenGL 1.1's infinite viewer, the Sega boards' R.z, the DS's (0,0,-1)
    line of sight. One float32 (3,) vector per frame; the GPU road reads
    the SAME bits from a texel, so both roads shade the same highlight."""
    row = np.asarray(view, np.float32)[2:3, :3]
    return M.normalize(row)[0].astype(np.float32)

#: models whose highlight the light loop must NOT scale by Specular
#: Level: Strauss has no such dial in Max, and Multi-Layer applies its two
#: levels lobe by lobe inside the model
LEVEL_FREE_MODELS = frozenset({'MAX_STRAUSS', 'MAX_MULTI_LAYER'})
MAX_MODELS = frozenset({'MAX_PHONG', 'MAX_BLINN', 'MAX_METAL', 'MAX_ANISOTROPIC',
                        'MAX_MULTI_LAYER', 'MAX_OREN_NAYAR_BLINN', 'MAX_STRAUSS',
                        'MAX_TRANSLUCENT'})

#: R228: the Cartoon Shader's shadow modes -- how its one painted
#: shadow tone lands on the paint. The closure carries the code as a
#: float (cartoon_mode: 0, 1, 2) so both devices read one number.
CARTOON_SHADOW_MODE_ITEMS = (
    ('TRANSPARENT', "Transparent Cel",
     "The shadow is a tinted cel laid OVER the paint: paint x Shadow "
     "Color, by Shadow Amount. The Golden Age shadow cel (a second "
     "exposure through a tinted overlay) and every airbrushed-tone look"),
    ('PAINTED', "Painted",
     "The shadow is a second flat paint: Shadow Color replaces the paint "
     "by Shadow Amount, whatever the paint was. UPA, the TV decades, "
     "the 90s features"),
    ('NONE', "None",
     "No shadow tone at all: the paint is flat everywhere, lit or not. "
     "Limited animation's cheapest look; the highlight dot still lands"),
)

#: R228: the era presets. Each writes the node's sockets and its shadow
#: mode (the node's Era menu applies one; the Cartoon templates drop
#: them ready-made). The values are STARTING POINTS that name what they
#: imitate -- the sockets stay editable afterwards. Socket names are
#: the Cartoon node's own; 'shadow_mode' is the node prop.
CARTOON_ERA_ITEMS = (
    ('CUSTOM', "Custom", "No preset: the sockets are yours"),
    ('GOLDEN_40S', "Golden Age (1930s-40s)",
     "Feature-era cel paint under a transparent shadow cel: the shadow "
     "darkens the paint part-way (a second exposure through a tinted "
     "overlay), edges slightly airbrushed, forms rounded, no highlight"),
    ('UPA_50S', "UPA Modern (1950s)",
     "Graphic flat colour: the shadow is a second paint in a bold "
     "contrasting hue, hard-edged, shapes simplified toward the form"),
    ('XEROX_60S', "Xerox Era (1960s-70s)",
     "Muted flat painted tones under scratchy xeroxed lines: the "
     "shadow is a desaturated second paint, hard-edged, lightly "
     "rounded. Pair with the Pencil ink style"),
    ('SATURDAY_70S', "Saturday Morning (1970s-80s)",
     "Limited television animation: flat paint and nothing else -- no "
     "shadow tone, no highlight"),
    ('FEATURE_90S', "90s Feature",
     "Digital ink-and-paint features: a painted shadow tone with a "
     "soft edge, forms rounded, and a small painted highlight dot"),
    ('TV_90S', "90s TV (Dark Deco)",
     "The high-contrast television look: a deep painted shadow, "
     "hard-edged, pushed well onto the lit side, no highlight"),
    # R240: two more eras, from the ends of the century
    ('NOIR_40S', "Wartime Noir (1940s)",
     "The dark theatrical short: the shadow cel exposed deep and "
     "pushed well onto the lit side for drama, its edge dead hard, "
     "forms rounded, no highlight -- the look of the 40s action and "
     "superhero shorts, made for a hard key and dark paints"),
    ('FLAT_10S', "Modern Flat (2010s)",
     "The modern digital TV cartoon: flat fills with one barely-darker "
     "cool tone in a small shadow region, hard-edged, forms heavily "
     "simplified, no highlight -- the thin-uniform-line era; pair "
     "with a clean, even ink"),
    ('INK_N_PAINT', "Ink 'n Paint (3ds Max)",
     "3ds Max's Ink 'n Paint material at its defaults: two Paint Levels "
     "(lit and a Shaded tone at 70 percent of the paint -- a transparent "
     "shadow cel, not a second paint), a hard split, no highlight, no "
     "smoothing -- the look of every Max cartoon test render since 2002; "
     "pair with the material's ink at 2 to 4 pixels"),
)

CARTOON_ERA_PRESETS = {
    'GOLDEN_40S': {
        'shadow_mode': 'TRANSPARENT',
        'Shadow Color': (0.62, 0.56, 0.58, 1.0), 'Shadow Amount': 0.55,
        'Shadow Threshold': 0.48, 'Shadow Softness': 0.06,
        'Shadow Smoothing': 0.4, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    'UPA_50S': {
        'shadow_mode': 'PAINTED',
        'Shadow Color': (0.32, 0.22, 0.48, 1.0), 'Shadow Amount': 1.0,
        'Shadow Threshold': 0.5, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.6, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    'XEROX_60S': {
        'shadow_mode': 'PAINTED',
        'Shadow Color': (0.50, 0.44, 0.44, 1.0), 'Shadow Amount': 0.8,
        'Shadow Threshold': 0.5, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.2, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    'SATURDAY_70S': {
        'shadow_mode': 'NONE',
        'Shadow Color': (0.55, 0.45, 0.62, 1.0), 'Shadow Amount': 1.0,
        'Shadow Threshold': 0.5, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.0, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    'FEATURE_90S': {
        'shadow_mode': 'PAINTED',
        'Shadow Color': (0.42, 0.34, 0.50, 1.0), 'Shadow Amount': 0.9,
        'Shadow Threshold': 0.5, 'Shadow Softness': 0.05,
        'Shadow Smoothing': 0.25, 'Highlight Size': 0.3,
        'Highlight Softness': 0.01, 'Lamp Influence': 0.0},
    'TV_90S': {
        'shadow_mode': 'PAINTED',
        'Shadow Color': (0.18, 0.16, 0.26, 1.0), 'Shadow Amount': 1.0,
        'Shadow Threshold': 0.56, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.3, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    # R240: the wartime noir short -- a deep transparent shadow pass
    # pushed onto the lit side, dead hard; and the modern flat TV
    # cartoon -- one barely-darker cool tone in a small region over
    # heavily simplified forms
    'NOIR_40S': {
        'shadow_mode': 'TRANSPARENT',
        'Shadow Color': (0.40, 0.38, 0.52, 1.0), 'Shadow Amount': 0.85,
        'Shadow Threshold': 0.60, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.35, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    'FLAT_10S': {
        'shadow_mode': 'TRANSPARENT',
        'Shadow Color': (0.82, 0.79, 0.90, 1.0), 'Shadow Amount': 1.0,
        'Shadow Threshold': 0.42, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.55, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
    # R242: Max's Ink 'n Paint defaults -- Shaded = 70% of Lighted (a
    # transparent 0.7 grey over the paint), Paint Levels 2 (one hard
    # split at the lambert's half), Highlight off, no smoothing
    'INK_N_PAINT': {
        'shadow_mode': 'TRANSPARENT',
        'Shadow Color': (0.70, 0.70, 0.70, 1.0), 'Shadow Amount': 1.0,
        'Shadow Threshold': 0.5, 'Shadow Softness': 0.0,
        'Shadow Smoothing': 0.0, 'Highlight Size': 0.0,
        'Highlight Softness': 0.02, 'Lamp Influence': 0.0},
}


#: R241: the hair shine wave's shape -- how the band's edge travels
#: around the head. Each is a period-matched wave in -1..1, so a shape
#: swap keeps the band's height and reach.
HAIR_SHINE_SHAPE_ITEMS = (
    ('SMOOTH', "Smooth Wave",
     "The classic angel ring: the band's edge swings around the head "
     "as a smooth sine -- the 80s TV convention, drawn with a soft "
     "hand"),
    ('ZIGZAG', "Zigzag",
     "The serrated highlight: straight teeth instead of a swing -- "
     "the late-90s TV and game convention, drawn with a ruler; pair "
     "with a low Softness for the crisp look"),
    ('SCALLOP', "Scallop",
     "Round arcs bulging downward with sharp cusps between them -- "
     "the scalloped ring of the shoujo tradition, each tooth a little "
     "drawn arc"),
    ('STEP', "Step",
     "The band jumps between two levels with no slope at all -- the "
     "blocky digital highlight of the 2000s, cut in straight "
     "segments"),
)

#: R240: the Anime Shader's Style menu -- the same applicator idea as
#: the Cartoon Shader's Era menu, for the anime tradition's own
#: decades. Every socket value is a starting point the artist edits
#: freely afterwards; the shadow colours are multiply TINTS on the
#: material's own paint (the cel painter's kage colour picked one step
#: down from the local colour), so one style serves every palette.
#: Each style writes the SAME full set of sockets and menus, so
#: switching styles never leaves the last one's leftovers behind.
ANIME_STYLE_ITEMS = (
    ('CUSTOM', "Custom", "No preset: the sockets are yours"),
    ('MOVIE_80S', "80s Film Feature",
     "The 35 mm theatrical feature: one restrained warm-grey shadow "
     "tone with the optical printer's slight softness, matte paint "
     "(no specular), a wide gentle sheen across the hair, a breath of "
     "airbrush inside the shadow edge -- the painterly end of the cel "
     "era"),
    ('TV_80S', "80s TV",
     "The broadcast cel: one hard cool-violet shadow tone cut at the "
     "terminator, and the classic angel-ring hair band waving around "
     "the head -- no airbrush, no rim; the schedule allowed neither"),
    ('OVA_80S', "80s OVA",
     "The video market's glamour pass: TWO shadow tones (a deeper "
     "second kage), a Fresnel rim as the painted edge-light, a double "
     "hair shine and airbrushed gradations against both sides of the "
     "shadow edge -- the richest cel dressing of the era"),
    ('TV_90S', "90s TV",
     "The late-cel broadcast look: one shadow tone pushed cooler and "
     "more saturated, its edge a shade harder than the 80s, the hair "
     "band bolder and simpler (fewer, deeper waves), faces held a "
     "touch brighter"),
    ('DIGITAL_00S', "2000s Digital",
     "Digital ink and paint: dead-hard tone bands (no gate or optics "
     "to soften them), the kage slightly desaturated the way the "
     "early RGB palettes ran, a crisp thin hair band with no wave, "
     "a small tight specular"),
    ('MODERN_20S', "Modern (2010s-20s)",
     "The current digital pipeline: a gently softened single tone, "
     "warm-shifted kage, a depth rim on the lit edge, the drawn "
     "contact shadows marched in the frame, and the key fixed to the "
     "camera -- the character lit the same way in every shot"),
)

ANIME_STYLE_PRESETS = {
    'MOVIE_80S': {
        '__props': {'tones': 'TWO', 'airbrush_side': 'SHADOW',
                    'light_source': 'SCENE', 'rim_mode': 'FRESNEL',
                    'rim_side': 'LIT'},
        'Shadow 1 Color': (0.74, 0.64, 0.66, 1.0),
        'Shadow 1 Threshold': 0.5, 'Shadow 1 Softness': 0.03,
        'Shadow 2 Color': (0.46, 0.36, 0.44, 1.0),
        'Shadow 2 Threshold': 0.24, 'Shadow 2 Softness': 0.03,
        'Shadow Bias': 0.0, 'Specular Level': 0.0,
        'Specular Size': 0.12, 'Specular Sharpness': 0.05,
        'Rim Amount': 0.0, 'Rim Power': 2.5,
        'Hair Shine': 0.3, 'Hair Shine Height': 0.75,
        'Hair Shine Width': 0.10, 'Hair Shine Wave': 0.02,
        'Hair Shine Waves': 4.0, 'Hair Shine Softness': 0.02,
        'Hair Shine Second': 0.0,
        'Airbrush': 0.2, 'Airbrush Width': 0.35,
        'Shadow Smoothing': 0.15, 'Screen Shadow': 0.0,
        'Screen Shadow Length': 24.0, 'Rim Width': 4.0,
        'Ambient': 0.4, 'Light Response': 1.0},
    'TV_80S': {
        '__props': {'tones': 'TWO', 'airbrush_side': 'LIT',
                    'light_source': 'SCENE', 'rim_mode': 'FRESNEL',
                    'rim_side': 'LIT'},
        'Shadow 1 Color': (0.64, 0.52, 0.62, 1.0),
        'Shadow 1 Threshold': 0.5, 'Shadow 1 Softness': 0.008,
        'Shadow 2 Color': (0.42, 0.30, 0.44, 1.0),
        'Shadow 2 Threshold': 0.24, 'Shadow 2 Softness': 0.008,
        'Shadow Bias': 0.0, 'Specular Level': 0.0,
        'Specular Size': 0.12, 'Specular Sharpness': 0.05,
        'Rim Amount': 0.0, 'Rim Power': 2.5,
        'Hair Shine': 0.6, 'Hair Shine Height': 0.78,
        'Hair Shine Width': 0.07, 'Hair Shine Wave': 0.035,
        'Hair Shine Waves': 6.0, 'Hair Shine Softness': 0.008,
        'Hair Shine Second': 0.0,
        'Airbrush': 0.0, 'Airbrush Width': 0.35,
        'Shadow Smoothing': 0.1, 'Screen Shadow': 0.0,
        'Screen Shadow Length': 24.0, 'Rim Width': 4.0,
        'Ambient': 0.35, 'Light Response': 1.0},
    'OVA_80S': {
        '__props': {'tones': 'THREE', 'airbrush_side': 'BOTH',
                    'light_source': 'SCENE', 'rim_mode': 'FRESNEL',
                    'rim_side': 'LIT'},
        'Shadow 1 Color': (0.68, 0.55, 0.66, 1.0),
        'Shadow 1 Threshold': 0.52, 'Shadow 1 Softness': 0.01,
        'Shadow 2 Color': (0.42, 0.32, 0.50, 1.0),
        'Shadow 2 Threshold': 0.26, 'Shadow 2 Softness': 0.01,
        'Shadow Bias': 0.0, 'Specular Level': 0.3,
        'Specular Size': 0.10, 'Specular Sharpness': 0.03,
        'Rim Amount': 0.35, 'Rim Power': 2.0,
        'Hair Shine': 0.8, 'Hair Shine Height': 0.78,
        'Hair Shine Width': 0.07, 'Hair Shine Wave': 0.03,
        'Hair Shine Waves': 6.0, 'Hair Shine Softness': 0.01,
        'Hair Shine Second': 0.5,
        'Airbrush': 0.35, 'Airbrush Width': 0.4,
        'Shadow Smoothing': 0.15, 'Screen Shadow': 0.0,
        'Screen Shadow Length': 24.0, 'Rim Width': 4.0,
        'Ambient': 0.35, 'Light Response': 1.0},
    'TV_90S': {
        '__props': {'tones': 'TWO', 'airbrush_side': 'LIT',
                    'light_source': 'SCENE', 'rim_mode': 'FRESNEL',
                    'rim_side': 'LIT'},
        'Shadow 1 Color': (0.58, 0.46, 0.66, 1.0),
        'Shadow 1 Threshold': 0.5, 'Shadow 1 Softness': 0.006,
        'Shadow 2 Color': (0.38, 0.28, 0.46, 1.0),
        'Shadow 2 Threshold': 0.24, 'Shadow 2 Softness': 0.006,
        'Shadow Bias': 0.03, 'Specular Level': 0.0,
        'Specular Size': 0.12, 'Specular Sharpness': 0.05,
        'Rim Amount': 0.0, 'Rim Power': 2.5,
        'Hair Shine': 0.5, 'Hair Shine Height': 0.78,
        'Hair Shine Width': 0.055, 'Hair Shine Wave': 0.05,
        'Hair Shine Waves': 3.0, 'Hair Shine Softness': 0.006,
        'Hair Shine Second': 0.0,
        'Airbrush': 0.0, 'Airbrush Width': 0.35,
        'Shadow Smoothing': 0.1, 'Screen Shadow': 0.0,
        'Screen Shadow Length': 24.0, 'Rim Width': 4.0,
        'Ambient': 0.35, 'Light Response': 1.0},
    'DIGITAL_00S': {
        '__props': {'tones': 'TWO', 'airbrush_side': 'LIT',
                    'light_source': 'SCENE', 'rim_mode': 'FRESNEL',
                    'rim_side': 'LIT'},
        'Shadow 1 Color': (0.66, 0.58, 0.70, 1.0),
        'Shadow 1 Threshold': 0.5, 'Shadow 1 Softness': 0.004,
        'Shadow 2 Color': (0.46, 0.38, 0.54, 1.0),
        'Shadow 2 Threshold': 0.24, 'Shadow 2 Softness': 0.004,
        'Shadow Bias': 0.0, 'Specular Level': 0.25,
        'Specular Size': 0.08, 'Specular Sharpness': 0.02,
        'Rim Amount': 0.0, 'Rim Power': 2.5,
        'Hair Shine': 0.5, 'Hair Shine Height': 0.78,
        'Hair Shine Width': 0.05, 'Hair Shine Wave': 0.01,
        'Hair Shine Waves': 2.0, 'Hair Shine Softness': 0.004,
        'Hair Shine Second': 0.0,
        'Airbrush': 0.0, 'Airbrush Width': 0.35,
        'Shadow Smoothing': 0.1, 'Screen Shadow': 0.0,
        'Screen Shadow Length': 24.0, 'Rim Width': 4.0,
        'Ambient': 0.35, 'Light Response': 1.0},
    'MODERN_20S': {
        '__props': {'tones': 'TWO', 'airbrush_side': 'LIT',
                    'light_source': 'CAMERA', 'rim_mode': 'SCREEN',
                    'rim_side': 'LIT'},
        'Shadow 1 Color': (0.76, 0.66, 0.72, 1.0),
        'Shadow 1 Threshold': 0.48, 'Shadow 1 Softness': 0.02,
        'Shadow 2 Color': (0.52, 0.42, 0.52, 1.0),
        'Shadow 2 Threshold': 0.24, 'Shadow 2 Softness': 0.02,
        'Shadow Bias': 0.0, 'Specular Level': 0.0,
        'Specular Size': 0.12, 'Specular Sharpness': 0.05,
        'Rim Amount': 0.45, 'Rim Power': 2.5,
        'Hair Shine': 0.35, 'Hair Shine Height': 0.76,
        'Hair Shine Width': 0.08, 'Hair Shine Wave': 0.02,
        'Hair Shine Waves': 4.0, 'Hair Shine Softness': 0.02,
        'Hair Shine Second': 0.0,
        'Airbrush': 0.0, 'Airbrush Width': 0.35,
        'Shadow Smoothing': 0.25, 'Screen Shadow': 0.5,
        'Screen Shadow Length': 20.0, 'Rim Width': 3.0,
        'Ambient': 0.4, 'Light Response': 1.0},
}


class Surface:
    """Per-fragment material parameters, all broadcast to (N,) or (N,3)."""

    __slots__ = ('n', 'diffuse', 'specular', 'glossiness', 'roughness', 'metallic',
                 'ior', 'anisotropy', 'aniso_rot', 'soften', 'ambient', 'emission',
                 'opacity', 'diffuse_level', 'specular_level', 'translucency',
                 'toon_size', 'toon_smooth', 'toon_steps', 'reflect', 'model',
                 'tangent', 'bitangent', 'backfacing',
                 'fresnel', 'fresnel_power', 'fresnel_color', 'fresnel_blend',
                 'rim', 'rim_power', 'rim_color', 'rim_blend',
                 'matcap', 'matcap_blend', 'matcap_mode', 'reflect_color',
                 'edge_opacity', 'backface_color', 'backface_mix', 'alpha_clip',
                 # R251 C031: 1.0 under Alpha Mode Clip+Blend (PS2 two-pass)
                 'alpha_soft',
                 'sheen', 'sheen_color', 'sheen_roughness', 'refraction',
                 # R251 lighting: the period finish dials (F006, F019-F021)
                 'fog_burn', 'fog_bias', 'fog_bank', 'brilliance', 'crand',
                 'pov_metallic',
                 # the anime/cel master (R218): tone bands, their
                 # colours, the decoded game-texture drivers
                 'anime_shadow1', 'anime_shadow2', 'anime_th1',
                 'anime_soft1', 'anime_th2', 'anime_soft2',
                 'anime_bias', 'anime_tones', 'anime_spec_size',
                 'anime_sharp', 'anime_mask', 'anime_gain',
                 # R221: the baked Shadow Ramp spec (a python object,
                 # riding like `bi`) and its per-pixel row pick
                 'anime_ramp', 'anime_ramp_row',
                 # R239: the material index the SDF face road resolves
                 # its frame from (-1 = none; the face spec itself
                 # rides anime_ramp's extras object)
                 'material_index',
                 # R229: the 80s additions -- the hair shine band, the
                 # airbrush gradation (the shadow smoothing rides
                 # cartoon_smooth, shared with the paint master)
                 'anime_shine', 'anime_shine_color', 'anime_shine_h',
                 'anime_shine_w', 'anime_shine_wave', 'anime_shine_waves',
                 'anime_shine_soft', 'anime_shine_second',
                 # R241: the hair pass -- the wave's shape, the phase
                 # around the head, the band riding the key's height,
                 # the second band's own tint
                 'anime_shine_shape', 'anime_shine_angle',
                 'anime_shine_follow', 'anime_shine_color2',
                 'anime_air', 'anime_air_color', 'anime_air_width',
                 'anime_air_side',
                 # the cartoon/paint master (R228)
                 'cartoon_shadow', 'cartoon_amount', 'cartoon_th',
                 'cartoon_soft', 'cartoon_smooth', 'cartoon_hl_color',
                 'cartoon_hl_size', 'cartoon_hl_soft', 'cartoon_mode',
                 'cartoon_lamp',
                 # the BI material node's own controls: the specular
                 # Toon pair, the diffuse-Fresnel pair, WardIso's Slope
                 'toon_size2', 'toon_smooth2', 'bi_fresnel',
                 'bi_fresnel_fac', 'bi_slope',
                 # the BI panel round: transparency Fresnel/Blend and
                 # spectra, mirror Fresnel/Blend, the ray-transparency
                 # IOR and Filter, Cubic and Tangent shading, the
                 # shadow flags, the mist gate -- and `bi`, one python
                 # object per batch carrying the non-numeric material
                 # extras (ramp specs, light group)
                 'bi_transp_fresnel', 'bi_transp_blend', 'bi_spectra',
                 'bi_mir_fresnel', 'bi_mir_blend', 'ray_ior',
                 'bi_ray_filter', 'bi_cubic', 'bi_tangent',
                 'shadow_receive', 'cast_only', 'shadows_only',
                 'use_mist', 'bi',
                 # R238: the cel's light -- the material's key (scene
                 # lamps / camera / world, and the key's own-frame
                 # vector), the screen shadow, the depth rim, the
                 # smoothing shape
                 'cel_light', 'cel_dir', 'cel_ss', 'cel_ss_len',
                 'cel_rim_mode', 'cel_rim_width', 'cel_rim_side',
                 'cel_shape',
                 # R243: Multi-Layer's second highlight and the
                 # Translucent shader's colour
                 'specular2', 'specular_level2', 'glossiness2',
                 'anisotropy2', 'aniso_rot2', 'translucent_color')

    def __init__(self, n):
        self.n = n
        f32 = np.float32
        one = np.ones(n, f32)
        self.diffuse = np.full((n, 3), 0.8, f32)
        self.specular = np.ones((n, 3), f32)
        self.glossiness = one * 25.0
        self.roughness = one * 0.3
        self.metallic = np.zeros(n, f32)
        self.ior = one * 1.45
        self.anisotropy = np.zeros(n, f32)
        self.aniso_rot = np.zeros(n, f32)
        self.soften = np.zeros(n, f32)
        self.ambient = one.copy()
        self.emission = np.zeros((n, 3), f32)
        self.opacity = one.copy()
        self.diffuse_level = one.copy()
        self.specular_level = one * 0.5
        self.translucency = np.zeros(n, f32)
        self.toon_size = one * 0.5
        self.toon_smooth = one * 0.05
        self.toon_steps = one * 2.0
        self.toon_size2 = one * 0.5      # the BI node's spec Toon pair
        self.toon_smooth2 = one * 0.1
        self.bi_fresnel = one * 0.1      # BI diffuse-Fresnel grad/fac
        self.bi_fresnel_fac = one * 0.5
        self.bi_slope = one * 0.1        # BI WardIso Slope (rms)
        # ---- the BI panel round's fields; every default reproduces the
        # engine's behaviour before the field existed
        self.bi_transp_fresnel = np.zeros(n, f32)   # 0 = plain alpha
        self.bi_transp_blend = one * 1.25
        self.bi_spectra = np.zeros(n, f32)
        self.bi_mir_fresnel = np.zeros(n, f32)      # 0 = flat mirror
        self.bi_mir_blend = one * 1.25
        self.ray_ior = one * 1.45        # refraction bend (master: = ior)
        self.bi_ray_filter = one.copy()  # 1 = full diffuse tint (master)
        self.bi_cubic = np.zeros(n, f32)
        self.bi_tangent = np.zeros(n, f32)
        self.shadow_receive = one.copy()
        self.cast_only = np.zeros(n, f32)
        self.shadows_only = np.zeros(n, f32)
        self.use_mist = one.copy()
        self.bi = None                   # per-batch material extras
        self.reflect = np.zeros(n, f32)
        self.tangent = None
        self.bitangent = None
        self.backfacing = np.zeros(n, bool)
        # artistic terms applied after the reflectance model, so they behave the
        # same whichever one is chosen
        self.fresnel = np.zeros(n, np.float32)
        self.fresnel_power = np.full(n, 3.0, np.float32)
        self.fresnel_color = np.ones((n, 3), np.float32)
        # anime/cel fields -- inert on every other model
        self.anime_shadow1 = np.full((n, 3), (0.62, 0.44, 0.48), f32)
        self.anime_shadow2 = np.full((n, 3), (0.38, 0.26, 0.38), f32)
        self.anime_th1 = one * 0.5
        self.anime_soft1 = one * 0.04
        self.anime_th2 = one * 0.22
        self.anime_soft2 = one * 0.04
        self.anime_bias = np.zeros(n, f32)
        self.anime_tones = one * 2.0
        self.anime_spec_size = one * 0.12
        self.anime_sharp = one * 0.05
        self.anime_mask = one.copy()
        self.anime_gain = one.copy()
        # R238: the cel's light, inert at the defaults (scene lamps, no
        # screen shadow, the Fresnel rim, the sphere)
        self.cel_light = np.zeros(n, f32)
        self.cel_dir = np.full((n, 3), (0.0, 0.0, 1.0), f32)
        self.cel_ss = np.zeros(n, f32)
        self.cel_ss_len = one * 24.0
        self.cel_rim_mode = np.zeros(n, f32)
        self.cel_rim_width = one * 4.0
        self.cel_rim_side = np.zeros(n, f32)
        self.cel_shape = np.zeros(n, f32)
        self.anime_ramp = None
        self.anime_ramp_row = np.zeros(n, f32)
        self.material_index = -1
        # R229: the 80s additions, all off by default
        self.anime_shine = np.zeros(n, f32)
        self.anime_shine_color = np.ones((n, 3), f32)
        self.anime_shine_h = one * 0.78
        self.anime_shine_w = one * 0.06
        self.anime_shine_wave = one * 0.03
        self.anime_shine_waves = one * 6.0
        self.anime_shine_soft = one * 0.01
        self.anime_shine_second = np.zeros(n, f32)
        # R241: the hair pass, all neutral at the 1.83 behaviour
        self.anime_shine_shape = np.zeros(n, f32)
        self.anime_shine_angle = np.zeros(n, f32)
        self.anime_shine_follow = np.zeros(n, f32)
        self.anime_shine_color2 = np.ones((n, 3), f32)
        self.anime_air = np.zeros(n, f32)
        self.anime_air_color = np.full((n, 3), (0.82, 0.62, 0.62), f32)
        self.anime_air_width = one * 0.35
        self.anime_air_side = np.zeros(n, f32)
        # cartoon/paint fields -- inert on every other model (R228)
        self.cartoon_shadow = np.full((n, 3), (0.55, 0.45, 0.62), f32)
        self.cartoon_amount = one.copy()
        self.cartoon_th = one * 0.5
        self.cartoon_soft = one * 0.02
        self.cartoon_smooth = np.zeros(n, f32)
        self.cartoon_hl_color = np.ones((n, 3), f32)
        self.cartoon_hl_size = np.zeros(n, f32)
        self.cartoon_hl_soft = one * 0.02
        self.cartoon_mode = np.zeros(n, f32)
        self.cartoon_lamp = np.zeros(n, f32)
        self.rim = np.zeros(n, np.float32)
        self.rim_power = np.full(n, 3.0, np.float32)
        self.rim_color = np.ones((n, 3), np.float32)
        self.matcap = np.zeros((n, 3), np.float32)
        self.matcap_blend = np.zeros(n, np.float32)
        # how each silhouette cheat lands on the lit result. 0 keeps the
        # historical behaviour (Add for fresnel/rim, Mix for matcap);
        # 1/2/3 are Mix-or-Add / Multiply / Screen, decoded in
        # apply_surface_effects and mirrored by the GPU pass
        self.fresnel_blend = np.zeros(n, np.float32)
        self.rim_blend = np.zeros(n, np.float32)
        self.matcap_mode = np.zeros(n, np.float32)
        self.reflect_color = np.ones((n, 3), np.float32)
        self.edge_opacity = np.ones(n, np.float32)
        # R211 punch-through: >= 0 is the material's CLIP threshold; the
        # shading law then forces alpha to exactly 0 or 1. -1 = BLEND
        self.alpha_clip = np.full(n, -1.0, np.float32)
        # R251 C031: > 0.5 keeps the sub-threshold alpha (the blend half)
        self.alpha_soft = np.zeros(n, np.float32)
        self.backface_color = np.zeros((n, 3), np.float32)
        self.backface_mix = np.zeros(n, np.float32)
        # a velvet lobe, added in the light loop rather than inside a model:
        # the packages that had this offered it on top of whichever shader was
        # picked, exactly like the terms above
        self.sheen = np.zeros(n, np.float32)
        self.sheen_color = np.ones((n, 3), np.float32)
        self.sheen_roughness = np.full(n, 0.3, np.float32)
        # R251 lighting: the period finish dials (F006, F019-F021)
        self.fog_burn = np.zeros(n, np.float32)
        self.fog_bias = np.zeros(n, np.float32)
        self.fog_bank = np.zeros(n, np.float32)
        self.brilliance = np.ones(n, np.float32)
        self.crand = np.zeros(n, np.float32)
        self.pov_metallic = np.zeros(n, np.float32)
        # how much of the ray traced through a transparent surface is kept
        self.refraction = np.ones(n, np.float32)
        # R243: the Max Multi-Layer's second highlight, Max's defaults
        # (a second, wider lobe off until its level is raised), and the
        # Translucent shader's colour (black = no translucency)
        self.specular2 = np.full((n, 3), 0.9, np.float32)
        self.specular_level2 = np.zeros(n, np.float32)
        self.glossiness2 = np.full(n, 25.0, np.float32)
        self.anisotropy2 = np.zeros(n, np.float32)
        self.aniso_rot2 = np.zeros(n, np.float32)
        self.translucent_color = np.zeros((n, 3), np.float32)
        self.model = 'PHONG'


def fresnel_schlick(cos_t, f0):
    c = np.clip(1.0 - cos_t, 0.0, 1.0)
    c5 = c * c * c * c * c
    if np.ndim(f0) == 2:
        return f0 + (1.0 - f0) * c5[:, None]
    return f0 + (1.0 - f0) * c5


def fresnel_dielectric(cos_t, ior):
    """Exact unpolarised dielectric Fresnel -- what the raytracers of the era used."""
    ci = np.clip(cos_t, 0.0, 1.0)
    eta = np.where(ci > 0, ior, 1.0 / np.maximum(ior, EPS))
    st2 = (1.0 - ci * ci) / np.maximum(eta * eta, EPS)
    tir = st2 > 1.0
    ct = np.sqrt(np.maximum(1.0 - st2, 0.0))
    rs = (eta * ci - ct) / np.maximum(eta * ci + ct, EPS)
    rp = (ci - eta * ct) / np.maximum(ci + eta * ct, EPS)
    f = 0.5 * (rs * rs + rp * rp)
    return np.where(tir, 1.0, np.clip(f, 0.0, 1.0))


def _gloss_to_alpha(gloss):
    """Phong exponent -> microfacet roughness (Walter et al. mapping)."""
    return np.sqrt(2.0 / np.maximum(gloss + 2.0, EPS))


def _soften(spec, ndl, amount):
    """3D Studio's 'Soften': fades the highlight where N.L approaches zero."""
    if not np.any(amount > 0):
        return spec
    s = np.clip(ndl * 3.0, 0.0, 1.0)
    fade = 1.0 - amount + amount * s
    return spec * fade


# ----------------------------------------------------------------- diffuse


def diffuse_lambert(ndl, **_):
    return np.maximum(ndl, 0.0)


def diffuse_oren_nayar(ndl, ndv, l, v, n, roughness, realnl=None, **_):
    """2.79's OrenNayar_Diff, verbatim (R155): nv clamps at 0 (View_A
    caps at pi/2), the projected-vector cosine floors at 0, and the
    smaller angle is scaled by 0.95 before tan -- the C's own guard
    against the tangent shooting to infinity.

    `realnl` is the C's own split (R190): OrenNayar_Diff takes an area
    lamp's `inp` as its first argument but recomputes `realnl = n.l`
    inside, gates on BOTH (realnl <= 0 and nl < 0 each return 0), and
    builds every angle and projection from realnl -- inp survives only
    as the outer factor `i = nl * (...)`. Without it (everywhere but
    an area lamp) the two are the same number and the arithmetic is
    bit-for-bit the old road."""
    s2 = roughness * roughness
    a = 1.0 - 0.5 * s2 / (s2 + 0.33)
    b = 0.45 * s2 / (s2 + 0.09)
    nl = np.clip(ndl if realnl is None else realnl, -1.0, 1.0)
    nv = np.maximum(np.clip(ndv, -1.0, 1.0), 0.0)
    ti = np.arccos(np.clip(nl, -1.0, 1.0))
    tr = np.arccos(np.clip(nv, -1.0, 1.0))
    alpha = np.maximum(ti, tr)
    beta = np.minimum(ti, tr) * 0.95
    lp = l - n * nl[:, None]
    vp = v - n * nv[:, None]
    cos_dphi = np.clip(M.dot(M.normalize(lp), M.normalize(vp)), -1.0, 1.0)
    if realnl is None:
        outer = np.maximum(nl, 0.0)
    else:
        # if (realnl <= 0.0f) return 0.0f; if (nl < 0.0f) return 0.0f;
        outer = np.where((nl > 0.0) & (np.asarray(ndl) >= 0.0),
                         ndl, 0.0).astype(np.float32)
    return (outer * (a + b * np.maximum(cos_dphi, 0.0) *
                     np.sin(alpha) * np.tan(beta))
            ).astype(np.float32)


def diffuse_minnaert(ndl, ndv, darkness=1.0, **_):
    nl = np.maximum(ndl, 0.0)
    nv = np.maximum(ndv, EPS)
    k = np.maximum(darkness, 0.0)
    return nl * np.power(np.maximum(nl * nv, EPS), k - 1.0) * nv


def diffuse_toon(ndl, size, smooth, steps=2.0, **_):
    """Cel shading: the light ramp cut into `steps` flat tones.

    `steps` counts tones, not edges, so two is the familiar lit/unlit cel and
    the edges for more than that are spread evenly across the lit range. At two
    this is bit-identical to the single-step version it replaces -- the loop
    runs once and reproduces the old expression exactly -- which matters
    because two is the default and every toon render made before this used it.

    The parameter was accepted and ignored for four releases: it reached the
    node, the exporter and this signature, and then nothing read it.
    """
    nl = np.clip(ndl, 0.0, 1.0)
    ang = np.arccos(np.clip(nl, 0.0, 1.0)) / (np.pi * 0.5)
    lim = np.clip(1.0 - size, 0.0, 1.0)
    sm = np.maximum(smooth, 1e-4)
    edges = np.maximum(np.round(np.asarray(steps, np.float32)) - 1.0, 1.0)
    top = int(np.max(edges)) if np.size(edges) else 1
    if top <= 1:
        return np.clip((lim + sm - ang) / sm, 0.0, 1.0)
    acc = np.zeros_like(nl, np.float32)
    for i in range(1, top + 1):
        band = np.clip((lim * (np.float32(i) / edges) + sm - ang) / sm, 0.0, 1.0)
        acc = acc + np.where(i <= edges, band, 0.0)
    return (acc / edges).astype(np.float32)


def diffuse_fujii(ndl, ndv, roughness, **_):
    """Energy-conserving 'Fujii' Oren-Nayar variant (cheap, well behaved)."""
    s = roughness
    fl = 1.0 / (np.pi * (1.0 + 0.5 * s))
    return np.maximum(ndl, 0.0) * (1.0 + s * (1.0 - 0.5 * np.maximum(ndl, 0)) *
                                   (1.0 - 0.5 * np.maximum(ndv, 0))) * fl * np.pi


# ---------------------------------------------------------------- specular


def spec_phong(ndl, rdv, gloss, **_):
    return np.where(ndl > 0, M.safe_pow(np.maximum(rdv, 0.0), gloss), 0.0)


def spec_blinn_phong(ndl, ndh, gloss, **_):
    return np.where(ndl > 0, M.safe_pow(np.maximum(ndh, 0.0), gloss * 4.0), 0.0)


def spec_blinn(ndl, ndv, ndh, vdh, gloss, ior, **_):
    """Blinn 1977 / Torrance-Sparrow: D * G * F / (4 N.V)."""
    a = _gloss_to_alpha(gloss)
    a2 = np.maximum(a * a, 1e-6)
    nh = np.maximum(ndh, 0.0)
    d = np.exp((nh * nh - 1.0) / np.maximum(a2 * nh * nh, EPS)) / \
        (np.pi * a2 * np.maximum(nh ** 4, EPS))
    g = np.minimum(1.0, np.minimum(2.0 * nh * np.maximum(ndv, 0) / np.maximum(vdh, EPS),
                                   2.0 * nh * np.maximum(ndl, 0) / np.maximum(vdh, EPS)))
    f0 = ((ior - 1.0) / np.maximum(ior + 1.0, EPS)) ** 2
    f = fresnel_schlick(np.maximum(vdh, 0.0), f0)
    out = d * g * f / (4.0 * np.maximum(ndv, EPS))
    return np.where(ndl > 0, np.maximum(out, 0.0), 0.0)


def spec_cook_torrance(ndl, ndv, ndh, vdh, roughness, ior, **_):
    m = np.maximum(roughness, 0.02)
    m2 = m * m
    nh = np.maximum(ndh, EPS)
    nh2 = nh * nh
    d = np.exp((nh2 - 1.0) / np.maximum(m2 * nh2, EPS)) / \
        (np.pi * m2 * np.maximum(nh2 * nh2, EPS))
    g = np.minimum(1.0, np.minimum(2.0 * nh * np.maximum(ndv, 0) / np.maximum(vdh, EPS),
                                   2.0 * nh * np.maximum(ndl, 0) / np.maximum(vdh, EPS)))
    f = fresnel_dielectric(np.maximum(vdh, 0.0), ior)
    out = d * g * f / (np.pi * np.maximum(ndv, EPS) * np.maximum(ndl, EPS))
    return np.where(ndl > 0, np.clip(out * np.maximum(ndl, 0), 0.0, 64.0), 0.0)


def spec_ward(ndl, ndv, h, n, t, b, ax, ay, **_):
    hdn = np.maximum(M.dot(h, n), EPS)
    hdt = M.dot(h, t)
    hdb = M.dot(h, b)
    ax = np.maximum(ax, 0.005)
    ay = np.maximum(ay, 0.005)
    ex = -((hdt / ax) ** 2 + (hdb / ay) ** 2) / np.maximum(hdn * hdn, EPS)
    denom = 4.0 * np.pi * ax * ay * np.sqrt(np.maximum(np.maximum(ndl, EPS) *
                                                       np.maximum(ndv, EPS), EPS))
    out = np.exp(ex) / np.maximum(denom, EPS)
    return np.where(ndl > 0, np.clip(out * np.maximum(ndl, 0), 0.0, 64.0), 0.0)


def spec_aniso_blinn(ndl, ndh, h, n, t, b, gloss, aniso, **_):
    """Elliptical Blinn highlight, as 3D Studio's Anisotropic shader.

    Distinct from Ward: this stretches a Blinn cosine lobe by giving it two
    exponents, rather than evaluating a Gaussian on the slope distribution.
    At anisotropy 0 the two exponents coincide and it reduces to Blinn-Phong,
    which is exactly what the original did.
    """
    ht = M.dot(h, t)
    hb = M.dot(h, b)
    hn = np.clip(ndh, -1.0, 1.0)
    denom = np.maximum(1.0 - hn * hn, 1e-6)
    a = np.clip(aniso, -0.95, 0.95)
    nu = np.maximum(gloss * (1.0 + a), 0.5)
    nv = np.maximum(gloss * (1.0 - a), 0.5)
    e = (nu * ht * ht + nv * hb * hb) / denom
    lobe = np.power(np.clip(hn, 0.0, 1.0), np.clip(e, 0.0, 8192.0))
    norm = np.sqrt((nu + 1.0) * (nv + 1.0)) / (8.0 * np.pi)
    return np.where(ndl > 0.0, lobe * norm * 8.0, 0.0).astype(np.float32)


def bi_spec_pow(inp, gloss):
    """2.79's spec(): the integer-bit square-multiply power, verbatim.

    Not pow(x, n): b1 = x*x floors at 0.01 before the bit ladder, b1
    zeroes below 0.001 twice on the way up, an EVEN hardness drops the
    x^1 factor, and bit 256 squares once more. Hardness itself is
    shi->har -- a SHORT -- so the per-pixel float chain truncates to an
    integer here, exactly where the C's assignment did. The 0.01 floor
    is visible: it brightens the dim tail of low-hardness highlights
    (the porcelain range) over what a plain power gives."""
    inp = np.asarray(inp, np.float32)
    hard = np.asarray(np.floor(np.asarray(gloss, np.float32)),
                      np.int32)
    x = np.clip(inp, 0.0, 1.0)
    out = np.where((hard & 1) == 0, np.float32(1.0), x)
    b1 = np.maximum(x * x, np.float32(0.01))
    out = np.where((hard & 2) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 4) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 8) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 16) != 0, out * b1, out)
    b1 = b1 * b1
    b1 = np.where(b1 < 0.001, np.float32(0.0), b1)
    out = np.where((hard & 32) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 64) != 0, out * b1, out)
    b1 = b1 * b1
    out = np.where((hard & 128) != 0, out * b1, out)
    b1 = np.where(b1 < 0.001, np.float32(0.0), b1)
    out = np.where((hard & 256) != 0, out * (b1 * b1), out)
    return np.where(inp >= 1.0, np.float32(1.0),
                    np.where(inp <= 0.0, np.float32(0.0),
                             out)).astype(np.float32)


def spec_bi_cooktorr(ndl, ndv, ndh, gloss, **_):
    """Blender Internal's CookTorr_Spec, verbatim from 2.79 (R155).

    spec(N.H, hardness) / (0.1 + N.V) -- the divisor is the whole
    character: the same highlight brightens up to 10x toward grazing
    view. The C has NO N.L gate (spec can sit past the terminator,
    a quirk BI shipped for its whole life) and no upper clamp; nh < 0
    returns 0, nv < 0 clamps to 0."""
    nv = np.maximum(ndv, 0.0)
    out = bi_spec_pow(ndh, gloss) / (0.1 + nv)
    return np.where(ndh < 0.0, np.float32(0.0), out).astype(np.float32)


def spec_bi_phong(ndl, ndh, gloss, **_):
    """Blender Internal's Phong_Spec, verbatim from 2.79 (R155): the
    HALF-vector lobe through spec(); rslt <= 0 returns 0, and there is
    no N.L gate in the C."""
    return np.where(ndh > 0.0, bi_spec_pow(ndh, gloss),
                    np.float32(0.0)).astype(np.float32)


def spec_bi_blinn(ndl, ndv, ndh, vdh, gloss, ior, **_):
    """Blender Internal's Blinn_Spec, transcribed term for term.

    Hardness maps onto a Gaussian half-angle width exactly as BI did
    (sqrt(1/hard) under 100, 10/hard above), the geometry term is
    Torrance-Sparrow's, and the Fresnel is BI's own refraction-index
    form. Returns 0 below refrac 1, as BI did."""
    refrac = np.maximum(np.asarray(ior, np.float32), 0.0)
    # spec_power arrives as (float)shi->har -- an INT-truncated short
    sp = np.maximum(np.floor(np.asarray(gloss, np.float32)), 1.0)
    spow = np.where(sp < 100.0, np.sqrt(1.0 / sp), 10.0 / sp)
    nh = np.maximum(ndh, 0.0)
    nv = np.maximum(ndv, 0.01)
    nl = np.maximum(ndl, 0.0)
    vh = np.maximum(vdh, 0.01)
    # the C's geometry pick is a STRICT-compare chain: g stays 0.0 on
    # ties (verbatim R155) -- a<b&&a<c, elif b<a&&b<c, elif c<a&&c<b
    a = np.ones_like(nh)
    b = 2.0 * nh * nv / vh
    c = 2.0 * nh * nl / vh
    g = np.where((a < b) & (a < c), a,
                 np.where((b < a) & (b < c), b,
                          np.where((c < a) & (c < b), c,
                                   np.float32(0.0))))
    p = np.sqrt(np.maximum(refrac * refrac + vh * vh - 1.0, 0.0))
    f = ((p - vh) ** 2 / (p + vh) ** 2) * \
        (1.0 + ((vh * (p + vh) - 1.0) ** 2
                / (vh * (p - vh) + 1.0) ** 2))
    ang = np.arccos(np.clip(nh, -1.0, 1.0))
    out = f * g * np.exp(-(ang * ang)
                         / np.maximum(2.0 * spow * spow, 1e-8))
    # verbatim gates: refrac < 1 -> 0, nl <= 0.01 -> 0, nh < 0 -> 0,
    # negative result -> 0; NO upper clamp in the C
    out = np.where((refrac >= 1.0) & (ndl > 0.01) & (ndh >= 0.0),
                   out, 0.0)
    return np.maximum(out, 0.0).astype(np.float32)


def bi_fresnel_fac(t1, grad, fac):
    """Blender Internal's fresnel_fac(), transcribed: t1 is view.vn."""
    fac = np.asarray(fac, np.float32)
    t2 = np.where(t1 > 0.0, 1.0 + t1, 1.0 - t1)
    t2 = grad + (1.0 - grad) * M.safe_pow(t2, fac)
    out = np.clip(t2, 0.0, 1.0)
    return np.where(fac == 0.0, 1.0, out).astype(np.float32)


def diffuse_bi_fresnel(ndl, grad, fac, **_):
    """BI's Fresnel diffuse, transcribed: fresnel_fac(lv, vn, ...) with
    lv pointing LAMP to surface, so view.vn = -N.L. It REPLACES the
    cosine outright -- the classic BI look where grazing lights glow."""
    return bi_fresnel_fac(-np.asarray(ndl, np.float32), grad, fac)


def diffuse_bi_toon(ndl, size, smooth, **_):
    """BI's Toon_Diff, transcribed: a hard angular band on acos(N.L)."""
    ang = np.arccos(np.clip(ndl, -1.0, 1.0))
    sm = np.asarray(smooth, np.float32)
    ramp = np.where(sm <= 0.0, 0.0,
                    1.0 - (ang - size) / np.maximum(sm, 1e-9))
    out = np.where(ang < size, 1.0,
                   np.where(ang >= size + sm, 0.0, ramp))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def diffuse_bi_minnaert(ndl, ndv, darkness, **_):
    """BI's Minnaert_Diff, verbatim (R155), BOTH branches: darkness <=
    1 darkens toward the rim via pow(max(nv*nl, 0.1), dark-1); above 1
    it brightens the rim via pow(1.001 - nv, dark-1) -- 1.001, the C's
    own constant."""
    nl = np.maximum(ndl, 0.0)
    nv = np.maximum(ndv, 0.0)
    dk = np.asarray(darkness, np.float32)
    low = nl * M.safe_pow(np.maximum(nv * nl, 0.1), dk - 1.0)
    high = nl * M.safe_pow(np.maximum(1.001 - nv, 1e-6), dk - 1.0)
    return np.where(dk <= 1.0, low, high).astype(np.float32)


def spec_bi_toon(ndl, ndh, size, smooth, **_):
    """BI's Toon_Spec, verbatim: the angular band on acos(N.H). The C
    has no N.L gate -- the band shows wherever the half-vector allows,
    exactly like the other BI speculars."""
    ang = np.arccos(np.clip(ndh, -1.0, 1.0))
    sm = np.asarray(smooth, np.float32)
    ramp = np.where(sm <= 0.0, 0.0,
                    1.0 - (ang - size) / np.maximum(sm, 1e-9))
    out = np.where(ang < size, 1.0,
                   np.where(ang >= size + sm, 0.0, ramp))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def spec_bi_wardiso(ndl, ndv, ndh, rms, **_):
    """BI's WardIso_Spec, verbatim from 2.79 (R155): an isotropic
    Gaussian on tan(acos(N.H)) with the Slope (rms) width. The C
    CLAMPS nl/nv/nh to 0.001 -- it never gates, so even a backfacing
    light keeps a (vanishing) term -- and applies no upper clamp."""
    nh = np.maximum(ndh, 0.001)
    nv = np.maximum(ndv, 0.001)
    nl = np.maximum(ndl, 0.001)
    alpha = np.maximum(np.asarray(rms, np.float32), 0.001)
    angle = np.tan(np.arccos(np.clip(nh, -1.0, 1.0)))
    out = nl * (1.0 / (4.0 * np.pi * alpha * alpha)) * \
        (np.exp(-(angle * angle) / (alpha * alpha))
         / np.sqrt(np.maximum(nv * nl, 1e-8)))
    return out.astype(np.float32)


#: the BI material node's shader menus, in DNA order: the model string
#: 'BI_MATRIX_{d}_{s}' carries one digit from each
BI_DIFF_ORDER = ('LAMBERT', 'OREN_NAYAR', 'TOON', 'MINNAERT', 'FRESNEL')
BI_SPEC_ORDER = ('COOKTORR', 'PHONG', 'BLINN', 'TOON', 'WARDISO')


def bi_matrix_terms(model, surf, n, l, v, ndl, ndv, ndh, vdh,
                    area_ndl=None):
    """(diffuse, specular scalar) for a BI material node's shader pair.

    The node keeps Blender Internal's diffuse and specular menus
    INDEPENDENT -- the 5x5 matrix one collapsed 'model' never could --
    and every branch is the transcribed 2.79 formula. Field packing:
    roughness carries Oren-Nayar roughness or Minnaert darkness (the
    diffuse menu chooses one), bi_slope carries WardIso's Slope,
    toon_size2/toon_smooth2 the specular Toon pair, glossiness the
    Hardness, ior the Refr slider.

    `area_ndl`, when given, is an AREA lamp's form-factor energy (the
    `inp` shade_one_light computes before the diffuse dispatch). It
    replaces the dot for exactly the shaders whose C took `inp` as an
    argument -- Lambert, Oren-Nayar, Minnaert. Toon_Diff and
    Fresnel_Diff took only the raw vectors, so under an area lamp they
    never saw the form factor: the quirk is kept. Specular gates keep
    the true dots too (the C's spec functions compute their own);
    the caller multiplies the finished specular by `inp`."""
    di = int(model[10])
    si = int(model[12])
    nd = ndl if area_ndl is None else area_ndl
    if di == 1:
        dif = diffuse_oren_nayar(
            nd, ndv, l, v, n, surf.roughness,
            realnl=(ndl if area_ndl is not None else None))
    elif di == 2:
        dif = diffuse_bi_toon(ndl, surf.toon_size, surf.toon_smooth)
    elif di == 3:
        dif = diffuse_bi_minnaert(nd, ndv, surf.roughness)
    elif di == 4:
        dif = diffuse_bi_fresnel(ndl, surf.bi_fresnel,
                                 surf.bi_fresnel_fac)
    else:
        dif = np.maximum(nd, 0.0)
    if si == 1:
        spec = spec_bi_phong(ndl, ndh, surf.glossiness)
    elif si == 2:
        spec = spec_bi_blinn(ndl, ndv, ndh, vdh, surf.glossiness,
                             surf.ior)
    elif si == 3:
        spec = spec_bi_toon(ndl, ndh, surf.toon_size2, surf.toon_smooth2)
    elif si == 4:
        spec = spec_bi_wardiso(ndl, ndv, ndh, surf.bi_slope)
    else:
        spec = spec_bi_cooktorr(ndl, ndv, ndh, surf.glossiness)
    return dif, spec


def bi_cubic(dif):
    """BI's Cubic Interpolation: smoothstep on the diffuse term.

    Transcribed with 2.79's own guard -- only values strictly inside
    (0, 1) are reshaped, so a Fresnel diffuse sitting at exactly 1.0
    or a negative pre-clamp term passes through untouched."""
    dif = np.asarray(dif, np.float32)
    inside = (dif > 0.0) & (dif < 1.0)
    return np.where(inside, 3.0 * dif * dif - 2.0 * dif * dif * dif,
                    dif).astype(np.float32)


def bi_tangent_normal(t, l):
    """BI's Tangent Shading: the per-light fake normal.

    2.79 builds cross(lv, tang) then cross(tang, that) -- algebraically
    L - T*(T.L), the light direction stripped of its along-strand
    component -- and shades with it in place of the surface normal.
    Returns the normalized fake normal (falls back to L where the
    light runs exactly along the tangent)."""
    t = np.asarray(t, np.float32)
    l = np.asarray(l, np.float32)
    n_eff = l - t * M.dot(t, l)[:, None]
    length = np.sqrt(np.maximum((n_eff * n_eff).sum(1), 1e-18))
    return (n_eff / length[:, None]).astype(np.float32)


#: BI ramp_blend() mode names, in MA_RAMP_* DNA order (material.c)
BI_RAMP_BLEND_ORDER = ('MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV',
                       'DIFF', 'DARK', 'LIGHT', 'OVERLAY', 'DODGE',
                       'BURN', 'HUE', 'SAT', 'VAL', 'COLOR', 'SOFT',
                       'LINEAR')

#: BI ramp input names, in MA_RAMP_IN_* DNA order
BI_RAMP_INPUT_ORDER = ('SHADER', 'ENERGY', 'NORMAL', 'RESULT')


def bi_ramp_blend(mode, col, fac, rampcol):
    """2.79's ramp_blend() (blenkernel material.c), vectorized.

    col (N,3) is blended toward rampcol (N,3) by fac (N,); every mode
    is the C transcribed, including the per-channel conditionals and
    the achromatic guards on HUE/COLOR (a grey ramp colour leaves the
    base untouched) and SAT (a grey base keeps its grey)."""
    col = np.asarray(col, np.float32).copy()
    rc = np.asarray(rampcol, np.float32)
    fac = np.asarray(fac, np.float32)
    if fac.ndim == 1:
        fac = fac[:, None]
    facm = 1.0 - fac
    if mode == 'MIX':
        return (facm * col + fac * rc).astype(np.float32)
    if mode == 'ADD':
        return (col + fac * rc).astype(np.float32)
    if mode == 'MULT':
        return (col * (facm + fac * rc)).astype(np.float32)
    if mode == 'SCREEN':
        return (1.0 - (facm + fac * (1.0 - rc)) *
                (1.0 - col)).astype(np.float32)
    if mode == 'OVERLAY':
        low = col * (facm + 2.0 * fac * rc)
        high = 1.0 - (facm + 2.0 * fac * (1.0 - rc)) * (1.0 - col)
        return np.where(col < 0.5, low, high).astype(np.float32)
    if mode == 'SUB':
        return (col - fac * rc).astype(np.float32)
    if mode == 'DIV':
        # per channel: only where the ramp colour is nonzero
        out = facm * col + fac * col / np.where(rc != 0.0, rc, 1.0)
        return np.where(rc != 0.0, out, col).astype(np.float32)
    if mode == 'DIFF':
        return (facm * col + fac * np.abs(col - rc)).astype(np.float32)
    if mode == 'DARK':
        tmp = rc + (1.0 - rc) * facm
        return np.minimum(col, tmp).astype(np.float32)
    if mode == 'LIGHT':
        return np.maximum(col, fac * rc).astype(np.float32)
    if mode == 'DODGE':
        tmp = 1.0 - fac * rc
        lifted = np.where(tmp <= 0.0, 1.0,
                          np.minimum(col / np.where(tmp <= 0.0, 1.0, tmp),
                                     1.0))
        return np.where(col != 0.0, lifted, col).astype(np.float32)
    if mode == 'BURN':
        tmp = facm + fac * rc
        burned = np.where(tmp <= 0.0, 0.0,
                          np.clip(1.0 - (1.0 - col) /
                                  np.where(tmp <= 0.0, 1.0, tmp), 0.0, 1.0))
        return burned.astype(np.float32)
    if mode in ('HUE', 'SAT', 'VAL', 'COLOR'):
        rh, rs, rv = M.rgb_to_hsv(col[:, 0], col[:, 1], col[:, 2])
        ch, cs, cv = M.rgb_to_hsv(rc[:, 0], rc[:, 1], rc[:, 2])
        f1 = fac[:, 0]
        fm1 = 1.0 - f1
        if mode == 'HUE':
            tr, tg, tb = M.hsv_to_rgb(ch, rs, rv)
            mixed = (facm * col +
                     fac * np.stack([tr, tg, tb], 1)).astype(np.float32)
            return np.where((cs != 0.0)[:, None], mixed,
                            col).astype(np.float32)
        if mode == 'SAT':
            tr, tg, tb = M.hsv_to_rgb(rh, fm1 * rs + f1 * cs, rv)
            out = np.stack([tr, tg, tb], 1)
            return np.where((rs != 0.0)[:, None], out,
                            col).astype(np.float32)
        if mode == 'VAL':
            tr, tg, tb = M.hsv_to_rgb(rh, rs, fm1 * rv + f1 * cv)
            return np.stack([tr, tg, tb], 1).astype(np.float32)
        # COLOR: ramp hue+sat over base value, achromatic-guarded
        tr, tg, tb = M.hsv_to_rgb(ch, cs, rv)
        mixed = (facm * col +
                 fac * np.stack([tr, tg, tb], 1)).astype(np.float32)
        return np.where((cs != 0.0)[:, None], mixed, col).astype(np.float32)
    if mode == 'SOFT':
        scr = 1.0 - (1.0 - rc) * (1.0 - col)
        return (facm * col +
                fac * ((1.0 - col) * rc * col +
                       col * scr)).astype(np.float32)
    if mode == 'LINEAR':
        return (col + fac * np.where(rc > 0.5, 2.0 * (rc - 0.5),
                                     2.0 * rc - 1.0)).astype(np.float32)
    return col.astype(np.float32)


def spec_toon(ndl, rdv, size, smooth, **_):
    ang = np.arccos(np.clip(rdv, -1.0, 1.0)) / (np.pi * 0.5)
    lim = np.clip(1.0 - size, 0.0, 1.0)
    sm = np.maximum(smooth, 1e-4)
    return np.where(ndl > 0, np.clip((lim + sm - ang) / sm, 0.0, 1.0), 0.0)


def spec_strauss(ndl, ndv, rdv, h, n, l, v, smoothness, metalness, transparency, **_):
    """Strauss 1990, as shipped in 3D Studio MAX."""
    s = np.clip(smoothness, 0.0, 1.0)
    m = np.clip(metalness, 0.0, 1.0)
    t = np.clip(transparency, 0.0, 1.0)
    h_ = 3.0 / np.maximum(1.0 - s, 1e-3)
    rn = (1.0 - t) - (1.0 - s) ** 3 * (1.0 - t)
    kf, kg = 1.12, 1.01
    fnl = _strauss_f(np.arccos(np.clip(ndl, -1, 1)) / (np.pi * 0.5), kf)
    gnl = _strauss_g(np.arccos(np.clip(ndl, -1, 1)) / (np.pi * 0.5), kg)
    gnv = _strauss_g(np.arccos(np.clip(ndv, -1, 1)) / (np.pi * 0.5), kg)
    j = fnl * gnl * gnv
    rj = np.minimum(1.0, rn + (rn + 0.1) * j)
    rs = M.safe_pow(np.maximum(-rdv, 0.0), h_) * rj
    return np.where(ndl > 0, rs, 0.0), rn, m


def _strauss_f(x, k):
    return (1.0 / (x - k) ** 2 - 1.0 / (k * k)) / (1.0 / (1.0 - k) ** 2 - 1.0 / (k * k))


def _strauss_g(x, k):
    return (1.0 / (k - 1.0) ** 2 - 1.0 / (x - k) ** 2) / \
           (1.0 / (k - 1.0) ** 2 - 1.0 / (k * k))


# ------------------------------------------------------------------ driver


# ------------------------------------------- R243: 3ds Max's own shaders
# The field's own ports of Max's Standard shaders (read for the
# algorithms; nothing of them ships), followed per light: Max's
# `lcolor` is the lamp's incident light, which
# the loop multiplies in afterwards; `vdir` is the ray from the eye, so
# Max's -dot(N, vdir) is our N.V and Max's L - vdir our L + V. Every
# function takes the caller's unit vectors: n, l (to the light), v (to
# the eye).

def max_soften(c, ndl, soft):
    """maxSoften: the cosine scaled by r (2 - r), r = N.L / Soften, where
    N.L falls under Soften -- BEFORE the power, as Max applies it."""
    soft = np.asarray(soft, np.float32)
    r = ndl / np.maximum(soft, np.float32(1e-6))
    fold = np.where((soft > 0.0) & (ndl < soft), r * (np.float32(2.0) - r), 1.0)
    return (c * fold).astype(np.float32)


def max_gloss(gloss):
    """Max's Glossiness in 0..1: the master's percent over 100."""
    return np.clip(np.asarray(gloss, np.float32) / np.float32(100.0), 0.0, 1.0)


def max_spec_phong(ndl, rdv, gloss, soft):
    """maxphong2: pow(R.L, 2^(10 g)) after Soften, where R.L > 0."""
    e = np.power(np.float32(2.0), max_gloss(gloss) * np.float32(10.0))
    c = max_soften(np.maximum(rdv, 0.0), ndl, soft)
    return np.where(rdv > 0.0, np.power(c, e), 0.0).astype(np.float32)


def max_spec_blinn(ndl, ndh, gloss, soft):
    """maxBlinn2: pow(N.H, 4 x 2^(10 g)) after Soften, where N.H > 0."""
    e = np.power(np.float32(2.0), max_gloss(gloss) * np.float32(10.0)) * np.float32(4.0)
    c = max_soften(np.maximum(ndh, 0.0), ndl, soft)
    return np.where(ndh > 0.0, np.power(c, e), 0.0).astype(np.float32)


def max_oren_nayar(ndl, ndv, l, v, n, rough, rho):
    """max_OrenNayarIllum exactly: the full Oren-Nayar with its
    interreflection term. `rough` is Max's Diffuse Roughness in 0..1
    (times pi/2 inside), `rho` the diffuse colour (N, 3). Returns the
    coloured diffuse (N, 3), clamped to 0..1 as Max clamps it."""
    f32 = np.float32
    rough = np.asarray(rough, f32) * f32(np.pi * 0.5)
    NL = np.asarray(ndl, f32)
    a = np.where(NL < 0.9999, np.arccos(np.clip(NL, -1.0, 1.0)), 0.0).astype(f32)
    a = np.clip(a, -np.pi * 0.49, np.pi * 0.49).astype(f32)
    NV = np.asarray(ndv, f32)
    vv = np.where((NV < 0.0)[:, None], -v, v)
    NV = np.abs(NV)
    b = np.where(NV < 0.9999, np.arccos(np.clip(NV, -1.0, 1.0)), 0.0).astype(f32)
    swap = b > a
    a2 = np.where(swap, b, a)
    b2 = np.where(swap, a, b)
    a, b = a2.astype(f32), b2.astype(f32)
    tanV = vv - n * NV[:, None]
    tanL = l - n * NL[:, None]
    w = np.sqrt((tanV * tanV).sum(1)) * np.sqrt((tanL * tanL).sum(1))
    cosDPhi = np.where(np.abs(w) >= 0.0004,
                       (tanV * tanL).sum(1) / np.where(np.abs(w) >= 0.0004, w, 1.0), 1.0)
    cosDPhi = np.clip(cosDPhi, -1.0, 1.0).astype(f32)
    bc = np.where(cosDPhi >= 0.0, -b, b) * f32(2.0 / np.pi)
    bCube = (bc * bc * bc).astype(f32)
    sigma2 = np.sqrt(rough).astype(f32)
    sigma3 = sigma2 / (sigma2 + f32(0.09))
    c1 = f32(1.0) - f32(0.5) * (sigma2 / (sigma2 + f32(0.33)))
    c2 = f32(0.45) * sigma3 * (np.sin(a) - bCube)
    c3 = f32(0.125) * sigma3 * np.sqrt(np.maximum(f32(4.0) * a * b / f32(2.0 * np.pi), 0.0))
    tanB = np.clip(np.tan(b), -100.0, 100.0)
    tanAB = np.clip(np.tan((a + b) * f32(0.5)), -100.0, 100.0)
    l1 = (c1 + c2 * cosDPhi * tanB + c3 * (f32(1.0) - np.abs(cosDPhi)) * tanAB).astype(f32)
    l2 = (f32(0.17) * (sigma2 / (sigma2 + f32(0.13)))
          * (f32(1.0) - cosDPhi * np.sqrt(np.maximum(f32(2.0) * b / f32(np.pi), 0.0)))).astype(f32)
    rho = np.clip(np.asarray(rho, f32)[:, :3], 0.0, 1.0)
    out = l1[:, None] * rho + l2[:, None] * np.sqrt(rho)
    return np.clip(out, 0.0, 1.0).astype(f32)


def max_tangent(n):
    """max_get_tangent: world Z (Max's object Z) projected onto the
    surface, world Y where the surface faces Z."""
    U = np.zeros_like(n)
    U[:, 2] = 1.0
    UN = M.dot(U, n)
    flip = UN > 0.9999
    U[flip] = (0.0, 1.0, 0.0)
    UN = M.dot(U, n)
    return M.normalize(U - n * UN[:, None])


def max_rotate_about(t, n, ang):
    """rotate_vector: t turned about n by ang radians (Rodrigues)."""
    ca = np.cos(ang)[:, None]
    sa = np.sin(ang)[:, None]
    return (t * ca + M.cross(n, t) * sa + n * (M.dot(n, t) * (1.0 - ca[:, 0]))[:, None]).astype(np.float32)


def max_gauss_highlight(n, l, v, ndl, gloss, aniso, orient, t):
    """max_gauss_high_light exactly: the anisotropic Gaussian lobe of
    the Anisotropic and Multi-Layer shaders. gloss in 0..1, aniso in
    0..1, orient in turns (Halcyon's Anisotropic Rotation)."""
    f32 = np.float32
    asz = (f32(1.0) - gloss) * f32(0.5 - 0.015)
    ax = np.maximum(f32(0.015) + asz, 0.0)
    ay = np.maximum(f32(0.015) + asz * (f32(1.0) - aniso), 0.0)
    h = M.normalize(l + v)
    NH = M.dot(n, h)
    NV = np.maximum(M.dot(n, v), f32(0.001))
    NL = np.asarray(ndl, f32)
    g = np.minimum(f32(1.0) / np.sqrt(np.maximum(NL * NV, f32(1e-12))), f32(3.0))
    ang = np.asarray(orient, f32) * f32(2.0 * np.pi)
    t1 = np.where((np.abs(ang) > 1e-9)[:, None], max_rotate_about(t, n, ang), t)
    b = M.cross(t1, n)
    x = M.dot(h, t1) / np.maximum(ax, f32(1e-6))
    y = M.dot(h, b) / np.maximum(ay, f32(1e-6))
    e = np.exp(f32(-2.0) * (x * x + y * y) / (f32(1.0) + NH))
    norm = f32(1.0 / (4.0 * np.pi * 0.03))
    out = norm * g * e * f32(0.5)
    return np.where(NH > 0.0, out, 0.0).astype(f32)


def max_fres_metal(c, k):
    b = k * k + np.float32(1.0)
    c2 = c * c
    rpl = (b * c2 - np.float32(2.0) * c + 1.0) / (b * c2 + np.float32(2.0) * c + 1.0)
    rpp = (b - np.float32(2.0) * c + c2) / (b + np.float32(2.0) * c + c2)
    return (np.float32(0.5) * (rpl + rpp)).astype(np.float32)


def max_spec_metal(n, l, v, ndl, ndv, gloss, diffuse):
    """maxMetal2: Cook-Torrance with Beckmann at m = 1 - g, the G term,
    and the Fresnel of the diffuse colour's intensity; returns the
    COLOURED lobe (N, 3) before Specular Level."""
    f32 = np.float32
    r = np.clip(f32(1.0) - gloss, 0.00001, 0.99999)
    m2inv = f32(1.0) / (r * r)
    h = M.normalize(l + v)
    LH = M.dot(l, h)
    NH = M.dot(n, h)
    VH = M.dot(v, h)
    NV = np.asarray(ndv, f32)
    NL = np.asarray(ndl, f32)
    G = np.where(NV < NL, f32(2.0) * NV * NH, f32(2.0) * NL * NH) / np.where(VH != 0.0, VH, 1.0)
    fav0 = np.minimum((diffuse[:, 0] + diffuse[:, 1] + diffuse[:, 2]) * f32(1.0 / 3.0), f32(0.9999))
    kav = f32(2.0) * np.sqrt(np.maximum(fav0, 0.0)) / np.sqrt(np.maximum(f32(1.0) - fav0, 1e-6))
    fav = max_fres_metal(LH, kav)
    t = (fav - fav0) / np.maximum(f32(1.0) - fav0, 1e-6)
    fcol = (f32(1.0) - t)[:, None] * diffuse[:, :3] + t[:, None]
    sec2 = f32(1.0) / np.maximum(NH * NH, f32(1e-12))
    D = f32(0.5 / np.pi) * sec2 * sec2 * m2inv * np.exp((f32(1.0) - sec2) * m2inv)
    G = np.minimum(G, f32(1.0))
    Rs = D * G / (NV + f32(0.05))
    ok = (NV >= 0.0) & (NH > 0.0) & (G > 0.0)
    return np.where(ok[:, None], fcol * Rs[:, None], 0.0).astype(f32)


def _max_strauss_F(x):
    KF = 1.12
    xb = np.clip(x, 0.0, 1.0)
    xkf = 1.0 / ((xb - KF) * (xb - KF))
    return ((xkf - 1.0 / (KF * KF)) / (1.0 / ((1.0 - KF) ** 2) - 1.0 / (KF * KF))).astype(np.float32)


def _max_strauss_G(x):
    KG = 1.01
    xb = np.clip(x, 0.0, 1.0)
    xkg = 1.0 / ((xb - KG) * (xb - KG))
    return ((1.0 / ((1.0 - KG) ** 2) - xkg) / (1.0 / ((1.0 - KG) ** 2) - 1.0 / (KG * KG))).astype(np.float32)


def max_strauss(n, l, v, ndl, ndv, gloss, metal, opacity, diffuse):
    """maxStrauss2 per light: returns (diffuse factor, coloured highlight)
    -- the highlight's light-colour mix taken for a white lamp (the
    loop multiplies the lamp's colour in afterwards)."""
    f32 = np.float32
    g3 = gloss * gloss * gloss
    d = f32(1.0) - metal * gloss
    rd = (f32(1.0) - metal * g3) * opacity
    rn = opacity - (f32(1.0) - g3) * opacity
    h_e = np.where(gloss >= 1.0, f32(600.0), f32(3.0) / np.maximum(f32(1.0) - gloss, 1e-6))
    NL = np.asarray(ndl, f32)
    NV = np.asarray(ndv, f32)
    dif = np.maximum(NL, 0.0) * d * rd
    R = l - n * (f32(2.0) * NL)[:, None]
    RV = M.dot(M.normalize(R), v)
    RV = np.where(NL < f32(0.15), RV * max_soften(np.ones_like(NL), NL, f32(0.15)), RV)
    s = f32(1.3) * np.power(np.maximum(-RV, 0.0), h_e)
    a = np.arccos(np.clip(NL, -1.0, 1.0)) / f32(0.5 * np.pi)
    b = np.arccos(np.clip(NV, -1.0, 1.0)) / f32(0.5 * np.pi)
    fa = _max_strauss_F(a)
    j = fa * _max_strauss_G(a) * _max_strauss_G(b)
    rj = np.where(rn > 0.0, np.clip(rn + (rn + f32(0.1)) * j, 0.0, 1.0), rn)
    white = np.ones_like(diffuse[:, :3])
    Cs = white + (metal * (f32(1.0) - fa))[:, None] * (diffuse[:, :3] - white)
    spec = np.where(((RV < 0.0) & (NL >= 0.0))[:, None], (s * rj)[:, None] * Cs, 0.0)
    return dif.astype(f32), spec.astype(f32)


def evaluate_max(model, surf, n, l, v, ndl, ndv, ndh, rdv):
    """The eight Max shaders: (diffuse, specular) per light -- diffuse a
    scalar for the Lambert ones and a COLOUR (N, 3) where Max's
    diffuse carries its own colour (the Oren-Nayar pair, Translucent);
    specular the coloured lobe before Specular Level."""
    f32 = np.float32
    g = max_gloss(surf.glossiness)
    NL = np.maximum(ndl, 0.0).astype(f32)
    lit = ndl >= 0.0
    if model == 'MAX_PHONG':
        sp = max_spec_phong(ndl, rdv, surf.glossiness, surf.soften)
        return NL, (np.where(lit, sp, 0.0)[:, None] * surf.specular).astype(f32)
    if model == 'MAX_BLINN':
        sp = max_spec_blinn(ndl, ndh, surf.glossiness, surf.soften)
        return NL, (np.where(lit, sp, 0.0)[:, None] * surf.specular).astype(f32)
    if model == 'MAX_METAL':
        spec = max_spec_metal(n, l, v, ndl, ndv, g, surf.diffuse)
        omabs = np.maximum(f32(1.0) - np.abs(np.minimum(surf.specular_level, f32(9.99))), 0.0)
        return (NL * omabs).astype(f32), np.where(lit[:, None], spec, 0.0).astype(f32)
    if model == 'MAX_ANISOTROPIC':
        t = max_tangent(n)
        gs = max_gauss_highlight(n, l, v, NL, g, np.clip(surf.anisotropy, 0.0, 1.0),
                                 surf.aniso_rot, t)
        return NL, (np.where(lit, NL * gs, 0.0)[:, None] * surf.specular).astype(f32)
    if model == 'MAX_MULTI_LAYER':
        dif = max_oren_nayar(ndl, ndv, l, v, n, np.clip(surf.roughness, 0.0, 1.0), surf.diffuse)
        dif = dif * NL[:, None]
        t = max_tangent(n)
        g1 = max_gauss_highlight(n, l, v, NL, g, np.clip(surf.anisotropy, 0.0, 1.0),
                                 surf.aniso_rot, t)
        g2 = max_gauss_highlight(n, l, v, NL, max_gloss(surf.glossiness2),
                                 np.clip(surf.anisotropy2, 0.0, 1.0), surf.aniso_rot2, t)
        # Max: spec1 = clamp(NL g1 level1 spec1 light), spec2 = NL g2
        # level2 spec2 light; the sum spec1 + (1 - spec1) spec2. Both
        # levels apply HERE (the model is level-free in the loop, so a
        # first lobe at level 0 still lets the second show); the first
        # lobe's clamp is taken before the lamp's colour
        s1 = np.clip((NL * g1 * np.minimum(surf.specular_level, f32(9.99)))[:, None]
                     * surf.specular, 0.0, 1.0)
        s2 = (NL * g2 * np.minimum(surf.specular_level2, f32(9.99)))[:, None] * surf.specular2
        spec = s1 + (f32(1.0) - s1) * s2
        return np.where(lit[:, None], dif, 0.0).astype(f32), np.where(lit[:, None], spec, 0.0).astype(f32)
    if model == 'MAX_OREN_NAYAR_BLINN':
        dif = max_oren_nayar(ndl, ndv, l, v, n, np.clip(surf.roughness, 0.0, 1.0), surf.diffuse)
        dif = dif * NL[:, None]
        sp = max_spec_blinn(ndl, ndh, surf.glossiness, surf.soften)
        return np.where(lit[:, None], dif, 0.0).astype(f32), \
            (np.where(lit, sp, 0.0)[:, None] * surf.specular).astype(f32)
    if model == 'MAX_STRAUSS':
        dif, spec = max_strauss(n, l, v, ndl, ndv, g, np.clip(surf.metallic, 0.0, 1.0),
                                np.clip(surf.opacity, 0.0, 1.0), surf.diffuse)
        return dif, spec
    if model == 'MAX_TRANSLUCENT':
        # Blinn without Soften; the translucent colour lit from either
        # side, per lamp: T (front + back - N.L)(1 - N.L) with unit lamp
        # energy inside the darkening term (Max sums all lamps first)
        sp = max_spec_blinn(ndl, ndh, surf.glossiness, np.zeros_like(ndl))
        back = (ndl < 0.0).astype(f32)
        trans = surf.translucent_color * ((f32(1.0) + back - NL) * (f32(1.0) - NL))[:, None]
        dif = surf.diffuse * NL[:, None] + np.maximum(trans, 0.0)
        return dif.astype(f32), (np.where(lit, sp, 0.0)[:, None] * surf.specular).astype(f32)
    raise ValueError(model)


def evaluate(model, surf, n, l, v, ndl_raw=None, area_ndl=None,
             area_ndl_back=None):
    """Evaluate one light for `model`.

    n, l, v: (N,3) unit vectors. l points from surface *toward* the light,
    v points from surface toward the eye.
    Returns (diffuse (N,), specular (N,3)).

    `area_ndl` is an AREA lamp's form-factor energy (lights.area_inp):
    it stands in for the diffuse cosine the way shade_one_light's
    `inp` did, while specular keeps its own true-dot gates -- the
    caller multiplies the returned specular by the same energy
    (specfac *= inp, the C's area lamp correction). `area_ndl_back`
    is the flipped-normal twin, consumed by BI translucency's
    negated-normal rerun.
    """
    # R251 (MAT-A C034): the DS toon pair takes the DS_FIXED lobe
    model = LOBE_ALIAS.get(model, model)
    ndl = M.dot(n, l) if ndl_raw is None else ndl_raw
    ndv = M.dot(n, v)
    h = M.normalize(l + v)
    ndh = M.dot(n, h)
    vdh = M.dot(v, h)
    r = M.reflect(-l, n)
    rdv = M.dot(r, v)
    gloss = surf.glossiness
    zero = np.zeros_like(ndl)

    if model in ('CONSTANT', 'WIREFRAME'):
        return zero, np.zeros((surf.n, 3), np.float32)

    if model in MAX_MODELS:
        # R243: Max's own light loops, their own Soften inside
        ndl_m = ndl if area_ndl is None else area_ndl
        return evaluate_max(model, surf, n, l, v, ndl_m, ndv, ndh, rdv)

    if model in ('ANIME', 'CARTOON'):
        # the cel bands need the SHADOW term inside their step input,
        # so the banding happens in the lamp loop (render._anime_lamp,
        # and the cartoon's lit accumulation) where visibility exists.
        # evaluate hands back the wrapped cosine 0..1 -- the games'
        # half-Lambert -- and no specular (the stepped highlight is
        # the loop's too, it needs N.H).
        ndl_a = ndl if area_ndl is None else area_ndl
        wrap = np.clip(ndl_a * np.float32(0.5) + np.float32(0.5),
                       0.0, 1.0).astype(np.float32)
        return wrap, np.zeros((surf.n, 3), np.float32)

    if isinstance(model, str) and model.startswith('BI_MATRIX_'):
        # the BI material node: independent diffuse and specular menus,
        # every branch a transcribed 2.79 formula
        if np.any(surf.bi_tangent > 0.5) and surf.tangent is not None:
            # Tangent Shading: 2.79 swaps the surface normal for
            # cross(tang, cross(lv, tang)) -- the light direction
            # stripped of its along-tangent component -- per light,
            # for BOTH lobes
            n_t = bi_tangent_normal(surf.tangent, l)
            on = (surf.bi_tangent > 0.5)[:, None]
            n_eff = np.where(on, n_t, n)
            ndl = M.dot(n_eff, l)
            ndv = M.dot(n_eff, v)
            ndh = M.dot(n_eff, h)
            n_use = n_eff
        else:
            n_use = n
        dif, spec = bi_matrix_terms(model, surf, n_use, l, v,
                                    ndl, ndv, ndh, vdh,
                                    area_ndl=area_ndl)
        if np.any(surf.translucency > 0.0):
            # BI's translucency: the SAME diffuse shader, evaluated
            # through the flipped normal, scaled by the slider -- for
            # every shader, not just a dedicated model
            dif_back, _sb = bi_matrix_terms(model, surf, -n_use, l, v,
                                            -ndl, -ndv, -ndh, vdh,
                                            area_ndl=area_ndl_back)
            dif = dif + np.clip(surf.translucency, 0.0, 1.0) * dif_back
        if np.any(surf.bi_cubic > 0.5):
            dif = np.where(surf.bi_cubic > 0.5, bi_cubic(dif), dif)
        spec = _soften(spec, ndl, surf.soften)
        return dif, spec[:, None] * surf.specular

    # ---- diffuse term (an AREA lamp's form factor stands in for the
    # cosine, exactly as shade_one_light's `inp` reassignment did)
    ndl_d = ndl if area_ndl is None else area_ndl
    if model in AXIS_MODELS:
        # R251 (LIGHT-B2): the console light units -- their own diffuse
        # and highlight, `v` being the camera axis (light_surface's Vs)
        return evaluate_console(model, surf, n, l, v, ndl, ndl_d, rdv)
    if model in ('OREN_NAYAR', 'OREN_NAYAR_BLINN'):
        dif = diffuse_oren_nayar(
            ndl_d, ndv, l, v, n, surf.roughness,
            realnl=(ndl if area_ndl is not None else None))
    elif model == 'MINNAERT':
        dif = diffuse_minnaert(ndl_d, ndv, 1.0 + surf.roughness * 2.0)
    elif model == 'TOON':
        dif = diffuse_toon(ndl_d, surf.toon_size, surf.toon_smooth, surf.toon_steps)
    elif model == 'TRANSLUCENT':
        back_d = (-ndl) if area_ndl_back is None else area_ndl_back
        dif = np.maximum(ndl_d, 0.0) + np.maximum(back_d, 0.0) * surf.translucency
    elif model == 'STRAUSS':
        dif = np.maximum(ndl_d, 0.0)
    else:
        dif = np.maximum(ndl_d, 0.0)

    # ---- specular term
    if model in ('LAMBERT', 'OREN_NAYAR', 'MINNAERT', 'TRANSLUCENT'):
        spec = zero
        spec_col = surf.specular
    elif model == 'PHONG':
        spec = spec_phong(ndl, rdv, gloss)
        spec_col = surf.specular
    elif model in ('BLINN_PHONG', 'GOURAUD', 'FLAT'):
        spec = spec_blinn_phong(ndl, ndh, gloss)
        spec_col = surf.specular
    elif model in ('BLINN', 'OREN_NAYAR_BLINN'):
        spec = spec_blinn(ndl, ndv, ndh, vdh, gloss, surf.ior)
        spec_col = surf.specular
    elif model == 'COOK_TORRANCE':
        spec = spec_cook_torrance(ndl, ndv, ndh, vdh, surf.roughness, surf.ior)
        spec_col = surf.specular
    elif model == 'WARD':
        t, b = _aniso_frame(surf, n)
        rough = np.maximum(surf.roughness, 0.02)
        aniso = np.clip(surf.anisotropy, -0.99, 0.99)
        ax = rough * (1.0 + aniso)
        ay = rough * (1.0 - aniso)
        spec = spec_ward(ndl, ndv, h, n, t, b, ax, ay)
        spec_col = surf.specular
    elif model == 'ANISOTROPIC':
        t, b = _aniso_frame(surf, n)
        spec = spec_aniso_blinn(ndl, ndh, h, n, t, b, gloss, surf.anisotropy)
        spec_col = surf.specular
    elif model == 'METAL':
        # 3D Studio's Metal: highlight takes the diffuse colour, no white sheen
        spec = spec_cook_torrance(ndl, ndv, ndh, vdh,
                                  np.maximum(1.0 / np.maximum(gloss, 1.0), 0.02),
                                  surf.ior)
        spec_col = surf.diffuse
    elif model == 'TOON':
        spec = spec_toon(ndl, rdv, surf.toon_size * 0.5, surf.toon_smooth)
        spec_col = surf.specular
    elif model == 'STRAUSS':
        smooth = np.clip(gloss / 100.0, 0.0, 1.0)
        spec, rn, met = spec_strauss(ndl, ndv, rdv, h, n, l, v, smooth,
                                     surf.metallic, 1.0 - surf.opacity)
        white = np.ones((surf.n, 3), np.float32)
        cs = white + met[:, None] * (1.0 - _strauss_fresnel(ndl)[:, None]) * \
            (surf.diffuse - white)
        return dif * rn, spec[:, None] * cs
    elif model == 'BI_COOKTORR':
        spec = spec_bi_cooktorr(ndl, ndv, ndh, gloss)
        spec_col = surf.specular
    elif model == 'BI_PHONG':
        spec = spec_bi_phong(ndl, ndh, gloss)
        spec_col = surf.specular
    elif model == 'BI_BLINN':
        spec = spec_bi_blinn(ndl, ndv, ndh, vdh, gloss, surf.ior)
        spec_col = surf.specular
    elif model == 'MULTI_LAYER':
        s1 = spec_blinn_phong(ndl, ndh, gloss)
        s2 = spec_blinn_phong(ndl, ndh, np.maximum(gloss * 0.15, 1.0))
        spec = s1 + s2 * 0.35
        spec_col = surf.specular
    else:
        spec = spec_blinn_phong(ndl, ndh, gloss)
        spec_col = surf.specular

    spec = _soften(spec, ndl, surf.soften)
    if np.ndim(spec) == 1:
        spec = spec[:, None] * spec_col
    return dif, spec


def _strauss_fresnel(ndl):
    return np.clip(1.0 - np.abs(ndl), 0.0, 1.0)


def _aniso_frame(surf, n):
    if surf.tangent is not None:
        t = surf.tangent
        b = surf.bitangent if surf.bitangent is not None else M.cross(n, t)
    else:
        t, b = M.orthonormal_basis(n)
    rot = surf.aniso_rot
    if np.any(np.abs(rot) > 1e-6):
        a = rot * 2.0 * np.pi
        ca, sa = np.cos(a)[:, None], np.sin(a)[:, None]
        t2 = t * ca + b * sa
        b2 = -t * sa + b * ca
        t, b = t2, b2
    return M.normalize(t), M.normalize(b)


def ambient_term(surf, model):
    if model in ('CONSTANT', 'WIREFRAME'):
        return np.ones((surf.n, 3), np.float32)
    return surf.ambient[:, None] * np.ones((1, 3), np.float32)


# ------------------------------------------------ R251 (LIGHT-B2): the
# console light units (GX_LIGHT, SEGA_MODEL2, SEGA_MODEL3, DS_FIXED) and
# the POV-Ray finish dials (brilliance, crand, metallic). Every line is
# float32, one operation per statement, in the order the GLSL twin
# (gpu/glsl_shading.py hal_gx_spec, gpu/material.py's DS lines) runs it.

def gx_spec(ndl, n, l, v, gloss):
    """GX_InitLightShininess's rational highlight: (N.H)^2 / (s/2 +
    (1 - s/2)(N.H)^2), the attenuation unit fed a = (0,0,1), k = (s/2, 0,
    1 - s/2) on N.H -- "only a ratio of quadratics, a true exponential
    function is not possible" (libogc gx.h). `v` is the camera axis."""
    hs = l + v
    hs = M.normalize(hs)
    h = M.dot(n, hs)
    h = np.maximum(h, np.float32(0.0))
    h2 = h * h
    sh = gloss * np.float32(0.5)
    om = np.float32(1.0) - sh
    den = om * h2
    den = sh + den
    # den = sh (1 - h2) + h2 > 0 for any Glossiness > 0; the guard is for
    # Glossiness 0 at h = 0 (0/0), where the GLSL twin takes the same 0
    sp = np.where(den > 0.0, h2 / np.where(den > 0.0, den, 1.0), 0.0)
    return sp.astype(np.float32)


_DS_TABLES = {}


def ds_shininess_table(gloss):
    """The DS's 128-entry 8-bit shininess table for one Glossiness value:
    T[i] = round(255 * ((i + 0.5)/128) ^ (Glossiness/8)), float64 fill
    (the games authored theirs; GBATEK documents no default), cached per
    value. Returned as float32 integers 0..255 (the texture's texels)."""
    g = float(gloss)
    t = _DS_TABLES.get(g)
    if t is None:
        p = g / 8.0
        i = np.arange(128, dtype=np.float64)
        vals = 255.0 * ((i + 0.5) / 128.0) ** p
        t = np.array([int(round(float(x))) for x in vals],
                     np.float32)
        _DS_TABLES[g] = t
    return t


def ds_spec(n, l, v, gloss):
    """GBATEK's ShininessLevel = max(0, -H.N)^2 with H the UNNORMALISED
    (LightVector + LineOfSight)/2 -- `v` the fixed line of sight -- then
    the 128-entry table indexed by truncation, the 8-bit entry as a 0.8
    fixed value (1/256)."""
    hs = l + v
    hs = hs * np.float32(0.5)
    s = M.dot(n, hs)
    s = np.maximum(s, np.float32(0.0))
    s = s * s
    si = s * np.float32(128.0)
    idx = np.minimum(si.astype(np.int32), 127)
    gloss = np.asarray(gloss, np.float32)
    if gloss.ndim == 0 or gloss.size == 1:
        t = ds_shininess_table(float(gloss.reshape(-1)[0]))
        val = t[idx]
    else:
        val = np.zeros(idx.shape, np.float32)
        for g in np.unique(gloss):
            m = gloss == g
            val[m] = ds_shininess_table(float(g))[idx[m]]
    sp = val * np.float32(0.00390625)
    return sp.astype(np.float32)


_MODEL3_MULT = np.array([1.6, 1.6, 2.4, 3.2], np.float32)


def sega_model2_spec(rdv, gloss):
    """Model 2's geometry engine: spec = 2(N.L)N.z - L.z (the reflection's
    component along the camera axis -- `rdv` with v the axis), clamped at
    0 and squared 0-3 times per the specular_control bits (MAME
    model2_v.cpp); Glossiness snaps to the nearest of 1, 2, 4, 8."""
    c = np.maximum(rdv, np.float32(0.0)).astype(np.float32)
    k = np.where(gloss < 1.5, 0, np.where(gloss < 3.0, 1,
                                          np.where(gloss < 6.0, 2, 3)))
    c2 = c * c
    c4 = c2 * c2
    c8 = c4 * c4
    sp = np.choose(k, [c, c2, c4, c8])
    return sp.astype(np.float32)


def sega_model3_spec(ndl, gloss):
    """Model 3's smooth polygons: pow(max(0, N.L), {8,16,32,64}[k]) *
    {1.6,1.6,2.4,3.2}[k] (Supermodel R3DShaderTriangles.h), the power by
    repeated squaring; Glossiness snaps to the nearest exponent."""
    c = np.maximum(ndl, np.float32(0.0)).astype(np.float32)
    k = np.where(gloss < 12.0, 0, np.where(gloss < 24.0, 1,
                                           np.where(gloss < 48.0, 2, 3)))
    c2 = c * c
    c4 = c2 * c2
    c8 = c4 * c4
    c16 = c8 * c8
    c32 = c16 * c16
    c64 = c32 * c32
    e = np.choose(k, [c8, c16, c32, c64])
    mult = _MODEL3_MULT[k]
    sp = e * mult
    return sp.astype(np.float32)


def evaluate_console(model, surf, n, l, v, ndl, ndl_d, rdv):
    """`evaluate`'s branch for the AXIS_MODELS: (diffuse (N,), specular
    (N,3)). `v` is the camera axis light_surface selected. GX_LIGHT's
    highlight passes through Soften exactly as the GLSL dispatch's
    hal_soften does. DS_FIXED returns the Lambert diffuse ONLY here --
    its table highlight (`ds_spec`) is added by light_surface after this
    call, exactly where the GPU's per-light block reads the `hal_dstab`
    texture (the shared GLSL dispatch carries no sampler), so the two
    dispatches agree term for term. The Sega models take no Soften (the
    boards had none) and never reach GLSL (RATE_FOR_MODEL)."""
    dif = np.maximum(ndl_d, np.float32(0.0)).astype(np.float32)
    gloss = surf.glossiness
    if model == 'GX_LIGHT':
        sp = gx_spec(ndl, n, l, v, gloss)
        sp = _soften(sp, ndl, surf.soften)
    elif model == 'DS_FIXED':
        sp = np.zeros_like(dif)
    elif model == 'SEGA_MODEL2':
        sp = sega_model2_spec(rdv, gloss)
    else:                                   # SEGA_MODEL3
        sp = sega_model3_spec(ndl, gloss)
    spec = sp[:, None] * surf.specular
    return dif, spec.astype(np.float32)


def quantize_lit(out, levels):
    """The vertex unit's saturation and integer output: clip to 0..1,
    floor(x * levels + 0.5) / levels, a DIVISION by `levels` so full
    white is exactly 1.0 (255/255, 31/31) -- the GX's 8-bit vertex colour
    (levels 255) and the DS's 5-bit one (31). Half-up tie, the same
    expression on both roads."""
    lv = np.float32(levels)
    o = np.clip(out, np.float32(0.0), np.float32(1.0))
    o = o * lv
    o = o + np.float32(0.5)
    o = np.floor(o)
    o = o / lv
    return o.astype(np.float32)


def apply_brilliance(dif, brilliance):
    """POV-Ray's finish brilliance (trace.cpp ComputeDiffuseColour):
    `if (Brilliance != 1.0) intensity = pow(fabs(cos), Brilliance)` --
    the diffuse cosine raised to the power, skipped exactly at 1.0."""
    b = np.asarray(brilliance, np.float32)
    if not np.any(b != 1.0):
        return dif
    d = np.asarray(dif, np.float32)
    p = np.power(np.maximum(d, np.float32(0.0)), b)
    return np.where(b == 1.0, d, p).astype(np.float32)


_POV_TWO_OVER_PI = np.float32(2.0 / np.pi)
_POV_MET_A = np.float32(0.014567225)
_POV_MET_B = np.float32(0.011612903)
_POV_MET_X0 = np.float32(1.12)


def pov_metallic_fresnel(ndl):
    """POV-Ray's ComputeMetallic Fresnel: x = acos(cos)/(pi/2), F =
    0.014567225/(x - 1.12)^2 - 0.011612903 clamped to 0..1. float32, one
    operation per statement (the GLSL twin carries the same float32
    literals through _f)."""
    c = np.clip(np.asarray(ndl, np.float32), np.float32(0.0),
                np.float32(1.0)).astype(np.float32)
    x = np.arccos(c)
    x = x * _POV_TWO_OVER_PI
    xm = x - _POV_MET_X0
    xm = xm * xm
    f = _POV_MET_A / xm
    f = f - _POV_MET_B
    f = np.clip(f, np.float32(0.0), np.float32(1.0))
    return f.astype(np.float32)


def apply_pov_metallic(spec, ndl, metallic, diffuse):
    """`colour *= 1 + M (1 - F)(pigment - 1)` on the highlight colour:
    pigment-coloured facing the light, the light's own colour at
    grazing. `spec` is the model's (N,3) specular (its colour folded in)."""
    f = pov_metallic_fresnel(ndl)
    m = np.float32(1.0) - f
    m = m * np.asarray(metallic, np.float32)
    tint = np.asarray(diffuse, np.float32) - np.float32(1.0)
    tint = tint * m[:, None]
    tint = tint + np.float32(1.0)
    return (np.asarray(spec, np.float32) * tint).astype(np.float32)
