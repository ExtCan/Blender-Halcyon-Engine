"""R252: the Console Emulation Shader's tables and its resolver.

The master shader used to carry the period machines' light units and
combiners beside Phong and Blinn, with one Glossiness slider standing in
for a specular-control bit field and no way to say "decal", "fixed
shading" or "the GTE's three lights". This module is the bpy-free half
of the node that replaced that: the machine menu, each machine's SHADER
TYPES (what the hardware's polygon attribute word could actually
select), the options that word carried, and `resolve`, which turns one
node's properties into what the renderer reads -- an engine model name
(core/shading.MODEL_ITEMS), the Surface fields the light loop consults
(fixed shading, the light limit, the viewer axis, GX's channel
functions, Model 3's sun clamp), the socket overrides a type implies
(an explicit exponent, a specular bit, the vertex colour as material)
and the combine-stage options core/combine.py reads per material.

Every entry names its machine and its source. Where a source published no
number the value is a stand-in and the text says so.
"""

#: engine model -> the machine's own shading rate when the node's Rate
#: menu says Machine (None: the scene's rate). The rate-fixed models
#: (core/combine.RATE_FIXED, shading.RATE_FOR_MODEL) pin theirs anyway.
MACHINE_RATE = {
    'GAMECUBE': 'VERTEX', 'MODEL2': 'FACE', 'MODEL3': 'VERTEX',
    'DS': 'VERTEX', 'PS1': 'VERTEX', 'PS2': 'VERTEX', 'PSP': 'VERTEX',
    'SATURN': 'VERTEX', 'N64': 'VERTEX', 'SYSTEM22': 'VERTEX',
    'DREAMCAST': 'VERTEX', 'PC_FIXED': 'VERTEX', 'PCX': 'VERTEX',
    'MEGADRIVE': 'FACE', 'SUPERFX': 'FACE', 'JAGUAR': 'VERTEX',
    'THREEDO': 'FACE', 'RENDERWARE': 'VERTEX',
}

#: the hardware light counts the per-console Light Limit toggles apply
LIGHT_LIMIT = {'GAMECUBE': 8, 'DS': 4, 'PS1': 3, 'PS2': 3, 'PSP': 4,
               'N64': 7, 'SYSTEM22': 1, 'RENDERWARE': 0}

CONSOLE_ITEMS = (
    ('GAMECUBE', "Nintendo GameCube / Wii (GX, 2001)",
     "Flipper's fixed-function light unit: Lambert diffuse under GX's "
     "rational highlight, lit per vertex, saturated to 8 bits -- with the "
     "channel control word's diffuse and attenuation functions"),
    ('MODEL2', "Sega Model 2 (1993)",
     "One luma per polygon, the specular-control bits' 1/2/4/8 power and "
     "the 64-step luma ramp each 5-bit channel is looked up through "
     "(MAME's model2 geometry engine)"),
    ('MODEL3', "Sega Model 3 (1996)",
     "Smooth polygons lit per vertex by one sun, the polygon header's "
     "8/16/32/64 specular exponent and gain, fixed shading, the sun "
     "clamp bit and the 64-step luma ramp (Supermodel's R3DShader)"),
    ('DS', "Nintendo DS (2004)",
     "The four polygon modes of the DS geometry engine -- modulation, "
     "decal, toon and highlight -- on its four-light unit with the "
     "128-entry shininess table and 5-bit vertex colours (GBATEK)"),
    ('PS1', "Sony PlayStation (1994)",
     "The GTE's three-light colour matrix feeding the GPU's x/128 texture "
     "blend: Gouraud or flat primitives, the raw-texture bit, 8-bit "
     "vertex colours with 80h as 1.0 (nocash's GTE and GPU notes)"),
    ('PS2', "Sony PlayStation 2 (2000)",
     "The Graphics Synthesizer's four texture functions -- modulate, "
     "decal, highlight, highlight 2 -- over VU1 per-vertex lighting with "
     "the common microcode's three directional lights (GS User's Manual)"),
    ('PSP', "Sony PSP (2004)",
     "The GU's texture functions (modulate, decal, replace, add), smooth "
     "or flat shading, and its four-light unit (sceGu reference)"),
    ('SATURN', "Sega Saturn (1994)",
     "VDP1 colour calculation: the Gouraud table ADDED to 5-bit texels, "
     "replace, half luminance, and the mesh (checkerboard) transparency "
     "bit (VDP1 User's Manual, CMDPMOD)"),
    ('N64', "Nintendo 64 (1996)",
     "The RDP colour combiner's presets -- modulate, decal, shade only, "
     "blend by texel alpha -- over the RSP's seven directional lights or "
     "the vertex colours with lighting off (the gbi.h G_CC_ presets)"),
    ('SYSTEM22', "Namco System 22 (1993)",
     "Vertex brightness with unity at 0x40 and near-4x headroom, one "
     "parallel light, fixed polygons (read from MAME's arithmetic, not a "
     "Namco document)"),
    ('DREAMCAST', "Sega Dreamcast / Naomi (PowerVR2, 1998)",
     "The CLX2's packed-colour vertices with the offset (specular) "
     "colour, its intensity vertex formats (one base colour times a "
     "per-vertex intensity), decal and flat shading"),
    ('PC_FIXED', "PC fixed function (Direct3D 3-9 / OpenGL 1.x)",
     "Gouraud with the separate specular colour, Direct3D's and OpenGL's "
     "provoking-vertex flat shading, the texture-stage ops (modulate, "
     "2x, 4x, add, add signed), the local viewer and colour vertex "
     "render states"),
    ('PCX', "PowerVR PCX1 / PCX2 (1996)",
     "The first PowerVR's monochrome Gouraud: one base colour per "
     "triangle and an interpolated intensity, from the mean of the "
     "corners or the first corner"),
    ('MEGADRIVE', "Sega Mega Drive / Genesis (1988)",
     "The VDP's shadow / highlight mode: albedo crushed to 3 bits a "
     "channel through the measured NORMAL, SHADOW or HIGHLIGHT DAC ramp, "
     "the class chosen by the key lamp or forced"),
    ('SUPERFX', "Super FX (SNES, 1993)",
     "Star Fox's PLOT: one colour per polygon, matched by the closest "
     "pair of entries of a fixed palette and filled as a checkerboard of "
     "the two -- or the nearest single entry"),
    ('JAGUAR', "Atari Jaguar (1993)",
     "The blitter's Gouraud in CRY: the polygon keeps one chroma and only "
     "the 8-bit intensity byte interpolates across it, so a lit "
     "polygon never changes hue -- the Jaguar's own shaded look"),
    ('THREEDO', "3DO Interactive Multiplayer (1993)",
     "The cel engine's PIXC multiplier: a cel is lit as a whole, its "
     "colour scaled by n/8 for n in 1..8 -- the only shading the "
     "machine's sprites-as-polygons could take"),
    ('RENDERWARE', "RenderWare (Criterion, 1999-2007)",
     "Criterion's RpMaterial on the PS2 / GameCube / Xbox / PC default "
     "pipelines: RwSurfaceProperties ambient and diffuse coefficients, "
     "prelit vertex colours added to the lighting, per-vertex on every "
     "platform, the RpMatFX environment / bump / dual-texture effects "
     "and the GTA-era specular and night-colour plugins"),
)

# --------------------------------------------------------- the shader types
# One tuple per machine: (identifier, label, description). `resolve` maps
# each to an engine model plus the flags below.

GC_TYPE_ITEMS = (
    ('LIT', "Lit polygon (GX light unit)",
     "GX_SetChanCtrl with lights on: Lambert under the rational shininess "
     "highlight, per vertex, saturated to 8 bits per channel"),
    ('VERTEX', "Vertex colour, lights off (GX_SRC_VTX)",
     "The channel's material source is the vertex colour and no light is "
     "enabled: the painted colour reaches the TEV untouched"),
)
GC_DIFFUSE_FN_ITEMS = (
    ('CLAMP', "Clamp (GX_DF_CLAMP)",
     "max(0, N.L): the ordinary Lambert cosine -- the default diffuse "
     "function of every GX title"),
    ('SIGN', "Signed (GX_DF_SIGN)",
     "N.L with its sign kept: a light behind the surface SUBTRACTS, and "
     "the channel saturates at 0 afterwards -- the GX manual's signed "
     "diffuse, used for two-sided tricks"),
    ('NONE', "None (GX_DF_NONE)",
     "No cosine at all: each light adds its colour flat, whatever the "
     "normal -- the light becomes a tint, as GX allowed"),
)
GC_ATTN_FN_ITEMS = (
    ('SPEC', "Specular (GX_AF_SPEC)",
     "The attenuation unit runs the rational highlight: Sun lamps add the "
     "shininess term, point lamps do not (GX_InitSpecularDir)"),
    ('SPOT', "Spot (GX_AF_SPOT)",
     "The attenuation unit does the spot cone and distance falloff instead "
     "of a highlight: no specular at all on this channel"),
    ('NONE', "None (GX_AF_NONE)",
     "No attenuation and no highlight: every light reaches the channel at "
     "full strength, diffuse only"),
)
GC_MATERIAL_SRC_ITEMS = (
    ('REG', "Register (GX_SRC_REG)",
     "The material colour comes from the channel's material register -- "
     "this node's Diffuse Color"),
    ('VTX', "Vertex colour (GX_SRC_VTX)",
     "The material colour comes from the vertex colour attribute -- the "
     "mesh's painted colour replaces Diffuse Color before lighting"),
)

M2_TYPE_ITEMS = (
    ('LIT', "Lit polygon (luma ramp)",
     "The geometry engine's one luma per polygon through the 64-step ramp: "
     "ambient plus the sun's cosine plus the specular bits' power"),
    ('FIXED', "Fixed luma (unlit polygon)",
     "The polygon's luma pinned at the ramp's top: the colour is shown as "
     "it is, the board's unlit polygon flag"),
)
M2_SPECULAR_ITEMS = (
    ('OFF', "Off", "No specular term: the specular-control bits cleared"),
    ('P1', "Power 1", "The reflection's camera-axis component as it is "
                      "(no squaring): the broadest highlight the bits allow"),
    ('P2', "Power 2", "Squared once: the second of the four specular settings"),
    ('P4', "Power 4", "Squared twice: the third of the four specular settings"),
    ('P8', "Power 8", "Squared three times: the tightest highlight the "
                      "specular-control bits can ask for"),
)

M3_TYPE_ITEMS = (
    ('SMOOTH', "Smooth polygon (per-vertex lit)",
     "The polygon header's smooth-shading bit: the sun's cosine, the "
     "ambient, the specular gain per vertex, then the 64-step luma ramp"),
    ('FIXED', "Fixed shading (vertex colours)",
     "The header's fixed-shading bit: the vertex colours are shown as they "
     "are, no light evaluated -- Supermodel's fixedShading path"),
)
M3_SPECULAR_ITEMS = (
    ('OFF', "Off", "The header's specular-enable bit cleared: no highlight"),
    ('P8', "Exponent 8, gain 1.6",
     "The first of the four polygon-header specular settings"),
    ('P16', "Exponent 16, gain 1.6",
     "The second of the four polygon-header specular settings"),
    ('P32', "Exponent 32, gain 2.4",
     "The third of the four polygon-header specular settings"),
    ('P64', "Exponent 64, gain 3.2",
     "The fourth and tightest of the polygon-header specular settings"),
)

DS_TYPE_ITEMS = (
    ('MODULATE', "Modulation (polygon mode 0)",
     "GBATEK's mode 0: the lit vertex colour and the texel multiply through "
     "the DS's +1 modulate on 6-bit channels"),
    ('DECAL', "Decal (polygon mode 1)",
     "GBATEK's mode 1: the texel REPLACES the lit colour by its own alpha "
     "-- (texel x alpha + lit x (63 - alpha)) / 64 on 6-bit channels"),
    ('TOON', "Toon table (polygon mode 2)",
     "The lit RED channel indexes a 32-entry table that replaces the "
     "vertex colour before the texel modulates it -- the DS's cel look"),
    ('HIGHLIGHT', "Highlight table (polygon mode 2, highlight)",
     "The same 32-entry table, but the entry is modulated with the texel "
     "AND added again, truncated at 63 -- the glossy toon of DS racers"),
)
DS_TABLE_ITEMS = (
    ('LINEAR', "Linear (table disabled)",
     "The shininess table left at its identity: the squared half-vector "
     "term is used as it is (Glossiness 8, the linear table)"),
    ('SOFT', "Soft (Glossiness 16)",
     "A gently pinned table: the highlight tightens a little past the "
     "linear ramp"),
    ('PIN', "Pin (Glossiness 64)",
     "A tightly pinned table: a small bright pin of a highlight, the look "
     "most DS titles authored"),
    ('CUSTOM', "Custom (Glossiness socket)",
     "The Glossiness socket shapes the 128-entry table: the entry curve is "
     "(i/128)^(Glossiness/8)"),
)

PS1_TYPE_ITEMS = (
    ('GOURAUD', "Gouraud, textured (GTE lit, GPU x/128)",
     "A Gouraud-shaded textured primitive: the GTE's colour per vertex, "
     "the GPU's texel x colour / 128 saturating blend"),
    ('FLAT', "Flat, textured (one colour per polygon)",
     "A flat-shaded textured primitive: one GTE colour for the whole "
     "polygon (the Rate menu's Face), the same x/128 blend"),
    ('RAW', "Raw texture (unlit)",
     "The GPU's raw-texture bit: the texel is drawn as it is, the vertex "
     "colour and the lighting ignored"),
)

PS2_TYPE_ITEMS = (
    ('MODULATE', "Modulate (TFX 0)",
     "The GS's MODULATE texture function: texel x vertex colour >> 7, "
     "0x80 as 1.0, saturating at 0xFF"),
    ('DECAL', "Decal (TFX 1)",
     "The GS's DECAL texture function: the texel replaces the vertex "
     "colour outright -- an unlit polygon"),
    ('HIGHLIGHT', "Highlight (TFX 2)",
     "The GS's HIGHLIGHT function: the texel modulates the diffuse and the "
     "vertex alpha -- the specular sum -- is added to all three channels"),
    ('HIGHLIGHT2', "Highlight 2 (TFX 3)",
     "The GS's HIGHLIGHT2 function: the same colour arithmetic as "
     "HIGHLIGHT; the difference lives in the output alpha, which keeps "
     "the texel's -- Halcyon's alpha is the texel's on both, so the two "
     "draw alike here"),
)

PSP_TYPE_ITEMS = (
    ('MODULATE', "Modulate (GU_TFX_MODULATE)",
     "Texel x vertex colour with 0x80 as 1.0, saturating -- the PSP's "
     "default, the PlayStation GPU's own arithmetic"),
    ('DECAL', "Decal (GU_TFX_DECAL)",
     "The texel blends over the vertex colour by its own alpha: "
     "texel x alpha + colour x (1 - alpha), 8-bit"),
    ('REPLACE', "Replace (GU_TFX_REPLACE)",
     "The texel replaces the vertex colour: an unlit polygon"),
    ('ADD', "Add (GU_TFX_ADD)",
     "The texel and the vertex colour ADD, saturating at 255 -- the "
     "glow and lightmap trick"),
    ('FLAT', "Flat (GU_FLAT, last vertex)",
     "GU_FLAT: the whole triangle takes its last vertex's colour, the "
     "OpenGL provoking vertex (plain modulate on this road)"),
)

SAT_TYPE_ITEMS = (
    ('GOURAUD', "Gouraud table (colour calculation 4)",
     "CMDPMOD colour calculation 4: each 5-bit texel channel plus the "
     "corner's table value minus 16, clamped 0..31"),
    ('REPLACE', "Replace (colour calculation 0)",
     "CMDPMOD colour calculation 0: the texel drawn as it is, no Gouraud "
     "table -- VDP1's plain sprite"),
    ('HALF_LUM', "Gouraud + half luminance (calculation 6)",
     "CMDPMOD colour calculation 6: the texel halved before the Gouraud "
     "table adds -- the dark-sprite trick"),
    ('MESH', "Gouraud + mesh (checkerboard transparency)",
     "The MESH bit: every other pixel drawn -- Opacity 0.5 under the "
     "scene's Screen Door transparency reproduces it; under Sorted it "
     "blends"),
)

N64_TYPE_ITEMS = (
    ('MODULATE', "Modulate (G_CC_MODULATERGB)",
     "The combiner's (TEXEL0 - 0) * SHADE + 0 on 8-bit values with the "
     "RDP's rounding term -- the ordinary lit texture"),
    ('DECAL', "Decal (G_CC_DECALRGB)",
     "(TEXEL0 - 0) * 1 + 0: the texel alone, no shade -- an unlit polygon"),
    ('SHADE', "Shade only (G_CC_SHADE)",
     "(0 - 0) * 0 + SHADE: the lit vertex colour alone, the texel ignored "
     "-- untextured N64 geometry"),
    ('BLENDRGBA', "Blend by texel alpha (G_CC_BLENDRGBA)",
     "(TEXEL0 - SHADE) * TEXEL0_ALPHA + SHADE: the texel fades over the "
     "lit colour by its alpha -- the decal-over-shade preset"),
    ('VERTEX', "Vertex colours, lighting off (G_LIGHTING clear)",
     "With G_LIGHTING off the vertex colours ARE the shade: the painted "
     "colour modulates the texel with no light evaluated"),
)

S22_TYPE_ITEMS = (
    ('LIT', "Lit polygon (x/64 brightness)",
     "The vertex brightness with unity at 0x40: the texel times the 8-bit "
     "shade, shifted right 6, saturating -- near-4x headroom"),
    ('FIXED', "Fixed polygon (unlit)",
     "The brightness pinned at unity: the texel drawn as it is"),
)

DC_TYPE_ITEMS = (
    ('PACKED', "Packed colour + offset colour",
     "The CLX2's packed 32-bit vertex colour with the offset (specular) "
     "colour added after the texture -- the Dreamcast's lit texture"),
    ('INTENSITY', "Intensity colour (base x intensity)",
     "The intensity vertex formats: one base colour per polygon and a "
     "per-vertex intensity, PowerVR's monochrome Gouraud carried over"),
    ('DECAL', "Decal (texel only)",
     "The TSP's DECAL texture / shading instruction: the texel alone, "
     "no vertex colour -- an unlit polygon"),
    ('FLAT', "Flat (Gouraud bit clear)",
     "The TSP's Gouraud bit clear: one colour for the whole polygon, "
     "taken from the last vertex (the OpenGL rule; the CLX2's own "
     "corner choice is a stand-in here)"),
)

PC_TYPE_ITEMS = (
    ('GOURAUD_SEP', "Gouraud, separate specular (D3D5-7 / OpenGL 1.2)",
     "D3DRS_SPECULARENABLE / GL_SEPARATE_SPECULAR_COLOR: the texture "
     "modulates the diffuse and the specular sum is added after it"),
    ('GOURAUD', "Gouraud, modulated specular (D3D3 / OpenGL 1.0)",
     "The highlight folded into the vertex colour BEFORE the texture, so "
     "a dark texel darkens the highlight -- the earliest fixed function"),
    ('FLAT_GL', "Flat, last vertex (OpenGL GL_FLAT)",
     "OpenGL 1.0-1.5's provoking vertex: the triangle takes its LAST "
     "corner's Gouraud lighting"),
    ('FLAT_D3D', "Flat, first vertex (D3DSHADE_FLAT)",
     "Direct3D 3-9's provoking vertex: the triangle takes its FIRST "
     "corner's Gouraud lighting"),
)
PC_TEXTURE_OP_ITEMS = (
    ('MODULATE', "Modulate (D3DTOP_MODULATE)",
     "texel x diffuse, 8-bit, the default texture stage op"),
    ('MODULATE2X', "Modulate 2x (D3DTOP_MODULATE2X)",
     "texel x diffuse shifted left one: twice the product, saturating -- "
     "the lightmap-era brightening"),
    ('MODULATE4X', "Modulate 4x (D3DTOP_MODULATE4X)",
     "texel x diffuse shifted left two: four times the product, saturating"),
    ('ADD', "Add (D3DTOP_ADD)",
     "texel + diffuse, saturating at 1 -- the additive glow stage"),
    ('ADDSIGNED', "Add signed (D3DTOP_ADDSIGNED)",
     "texel + diffuse - 0.5, clamped 0..1 -- the detail-texture stage "
     "(DirectX 6 emboss uses it)"),
)

PCX_BASE_ITEMS = (
    ('MEAN', "Mean of the three corners",
     "The triangle's base colour is the average of its three lit corners "
     "(Halcyon's stand-in for the driver's choice)"),
    ('FIRST', "First corner",
     "The triangle's base colour is its first corner's lit colour; the "
     "other corners contribute intensity only"),
)

MD_TYPE_ITEMS = (
    ('AUTO', "Shadow / highlight by class",
     "The VDP's S/H mode with the class chosen per polygon: SHADOW where "
     "the key lamp's lit fraction is below a half, HIGHLIGHT where the "
     "specular sum reaches a half, NORMAL otherwise"),
    ('NORMAL', "Normal ramp only (S/H off)",
     "Every polygon through the NORMAL DAC ramp: the plain 3-bit palette "
     "with no shadow or highlight class"),
    ('SHADOW', "Shadow class forced",
     "Every polygon through the SHADOW ramp -- the half-brightness DAC "
     "levels"),
    ('HIGHLIGHT', "Highlight class forced",
     "Every polygon through the HIGHLIGHT ramp -- the lifted DAC levels"),
)

SFX_TYPE_ITEMS = (
    ('DITHER', "Dither pair (PLOT checkerboard)",
     "The face's colour matched by the closest PAIR of palette entries "
     "and filled as a 50% checkerboard at output-pixel pitch"),
    ('SOLID', "Nearest entry (no dither)",
     "The face filled with the single closest palette entry -- the flat "
     "fills Star Fox used on its brightest polygons"),
)

JAG_TYPE_ITEMS = (
    ('GOURAUD', "CRY intensity Gouraud (blitter GOURD)",
     "The blitter's Gouraud mode: the polygon's chroma fixed, its 8-bit "
     "intensity interpolated per pixel from the corners"),
    ('FLAT', "Flat fill (one intensity per polygon)",
     "The blitter's plain fill: one chroma and one intensity for the "
     "whole polygon (the Rate menu's Face)"),
)

TDO_TYPE_ITEMS = (
    ('PIXC', "Cel shading (PIXC multiplier n/8)",
     "The cel engine's pixel processor: the cel's colour times n/8, n in "
     "1..8, chosen once per cel from its lighting"),
)

RW_TYPE_ITEMS = (
    ('DEFAULT', "Default pipeline (lit, no prelight)",
     "rpGEOMETRYLIGHT without rpGEOMETRYPRELIT: ambient x the ambient "
     "coefficient plus each light's cosine x the diffuse coefficient, per "
     "vertex, clamped, then the material colour and the texture"),
    ('PRELIT', "Prelit + lights (rpGEOMETRYPRELIT | rpGEOMETRYLIGHT)",
     "The prelight vertex colour is ADDED to the computed light before "
     "the clamp -- the baked-lighting road of every RW world mesh"),
    ('UNLIT', "Prelit only, lights off (rpGEOMETRYPRELIT)",
     "No light evaluated: the prelight colour alone, times the material "
     "colour and the texture -- RW's unlit geometry"),
)
RW_PLATFORM_ITEMS = (
    ('PS2', "PlayStation 2 (GS, 0x80 = 1.0)",
     "The lit colour on the GS's 8-bit grid with 0x80 as 1.0, so a vertex "
     "may reach 0xFF, twice the texture's brightness -- the PS2's "
     "overbright headroom; texel x colour >> 7, saturating"),
    ('GC', "GameCube (TEV, 8-bit)",
     "The lit colour saturated to 8 bits, the TEV's modulate with its "
     "255 -> 256 trick: (texel x (c + c >> 7) + 128) >> 8"),
    ('PC', "Xbox / PC Direct3D 8 (8-bit)",
     "The lit colour saturated to 8 bits, an exact 8-bit modulate "
     "(texel x colour / 255, rounded) -- the register combiner's and "
     "the reference rasteriser's product"),
)
RW_SPECULAR_ITEMS = (
    ('NONE', "None (default pipelines)",
     "RwSurfaceProperties.specular is not used by RenderWare's default "
     "pipelines (the SDK says so): no highlight at all"),
    ('PLUGIN', "Specular plugin (GTA San Andreas vehicles)",
     "Rockstar's specular material plugin: a Blinn-Phong highlight from "
     "the directional lights, Specular Level its level and Glossiness its "
     "power, added after the texture"),
)
RW_MATFX_ITEMS = (
    ('NONE', "None (rpMATFXEFFECTNULL)",
     "No material effect: the lit, textured material alone"),
    ('ENVMAP', "Environment map (rpMATFXEFFECTENVMAP)",
     "A sphere-mapped environment texture added over the material, scaled "
     "by the env-map coefficient -- link the map into Env Map"),
    ('BUMPMAP', "Bump map (rpMATFXEFFECTBUMPMAP)",
     "A height map bent into the shading normal, scaled by the bumpiness "
     "coefficient -- link the map into Bump Map"),
    ('BUMPENVMAP', "Bump + environment (rpMATFXEFFECTBUMPENVMAP)",
     "Both effects at once: the bumped normal lights the surface and "
     "reflects the environment map"),
    ('DUAL', "Dual texture (rpMATFXEFFECTDUAL)",
     "A second texture pass blended over the base texture by the dual "
     "blend mode -- link the second map into Dual Texture"),
)
RW_DUAL_BLEND_ITEMS = (
    ('MODULATE', "Modulate (rwBLENDZERO, rwBLENDSRCCOLOR)",
     "The second texture multiplies the base: the lightmap pass"),
    ('ADD', "Add (rwBLENDONE, rwBLENDONE)",
     "The second texture adds over the base, saturating: the glow pass"),
    ('ALPHA', "Alpha blend (rwBLENDSRCALPHA, rwBLENDINVSRCALPHA)",
     "The second texture blends over the base by its own alpha: the "
     "decal pass"),
)

RATE_ITEMS = (
    ('MACHINE', "Machine's own rate",
     "Shade at the rate the chosen machine lit at (per vertex for the "
     "consoles' light units, per polygon for Model 2, the Mega Drive, "
     "the Super FX and the 3DO)"),
    ('SCENE', "Scene's rate",
     "Follow the scene's Shading Rate setting instead (a combiner needs "
     "a vertex or face rate; a pixel-rate scene shades it per vertex)"),
    ('VERTEX', "Per vertex (Gouraud)",
     "Light at the vertices and interpolate -- the banded console look"),
    ('FACE', "Per face (flat)",
     "Light once per polygon -- the faceted look of the earliest machines"),
)

#: the node's enum properties, in panel order: (prop, items, default)
ENUM_PROPS = (
    ('console', CONSOLE_ITEMS, 'PS1'),
    ('gc_type', GC_TYPE_ITEMS, 'LIT'),
    ('gc_diffuse_fn', GC_DIFFUSE_FN_ITEMS, 'CLAMP'),
    ('gc_attn_fn', GC_ATTN_FN_ITEMS, 'SPEC'),
    ('gc_material_src', GC_MATERIAL_SRC_ITEMS, 'REG'),
    ('m2_type', M2_TYPE_ITEMS, 'LIT'),
    ('m2_specular', M2_SPECULAR_ITEMS, 'P4'),
    ('m3_type', M3_TYPE_ITEMS, 'SMOOTH'),
    ('m3_specular', M3_SPECULAR_ITEMS, 'P16'),
    ('ds_type', DS_TYPE_ITEMS, 'MODULATE'),
    ('ds_table', DS_TABLE_ITEMS, 'PIN'),
    ('ps1_type', PS1_TYPE_ITEMS, 'GOURAUD'),
    ('ps2_type', PS2_TYPE_ITEMS, 'MODULATE'),
    ('psp_type', PSP_TYPE_ITEMS, 'MODULATE'),
    ('sat_type', SAT_TYPE_ITEMS, 'GOURAUD'),
    ('n64_type', N64_TYPE_ITEMS, 'MODULATE'),
    ('s22_type', S22_TYPE_ITEMS, 'LIT'),
    ('dc_type', DC_TYPE_ITEMS, 'PACKED'),
    ('pc_type', PC_TYPE_ITEMS, 'GOURAUD_SEP'),
    ('pc_texture_op', PC_TEXTURE_OP_ITEMS, 'MODULATE'),
    ('pcx_base', PCX_BASE_ITEMS, 'MEAN'),
    ('md_type', MD_TYPE_ITEMS, 'AUTO'),
    ('sfx_type', SFX_TYPE_ITEMS, 'DITHER'),
    ('jag_type', JAG_TYPE_ITEMS, 'GOURAUD'),
    ('tdo_type', TDO_TYPE_ITEMS, 'PIXC'),
    ('rw_type', RW_TYPE_ITEMS, 'PRELIT'),
    ('rw_platform', RW_PLATFORM_ITEMS, 'PS2'),
    ('rw_specular', RW_SPECULAR_ITEMS, 'NONE'),
    ('rw_matfx', RW_MATFX_ITEMS, 'NONE'),
    ('rw_dual_blend', RW_DUAL_BLEND_ITEMS, 'MODULATE'),
    ('rate', RATE_ITEMS, 'MACHINE'),
)

#: the node's bool / int / float properties: (prop, kind, default)
SCALAR_PROPS = (
    ('light_limit', 'BOOL', False),
    ('m3_sun_clamp', 'BOOL', True),
    ('m3_alpha_steps', 'BOOL', False),
    ('pc_local_viewer', 'BOOL', True),
    ('pc_color_vertex', 'BOOL', False),
    ('luma_gamma', 'FLOAT', 1.0),
    ('toon_steps', 'INT', 2),
)

#: which properties each machine's panel draws, in order (the shared
#: Rate menu and Light Limit toggle are drawn by the node itself)
PANEL = {
    'GAMECUBE': ('gc_type', 'gc_diffuse_fn', 'gc_attn_fn', 'gc_material_src'),
    'MODEL2': ('m2_type', 'm2_specular', 'luma_gamma'),
    'MODEL3': ('m3_type', 'm3_specular', 'm3_sun_clamp', 'm3_alpha_steps',
               'luma_gamma'),
    'DS': ('ds_type', 'ds_table', 'toon_steps'),
    'PS1': ('ps1_type',),
    'PS2': ('ps2_type',),
    'PSP': ('psp_type',),
    'SATURN': ('sat_type',),
    'N64': ('n64_type',),
    'SYSTEM22': ('s22_type',),
    'DREAMCAST': ('dc_type',),
    'PC_FIXED': ('pc_type', 'pc_texture_op', 'pc_local_viewer',
                 'pc_color_vertex'),
    'PCX': ('pcx_base',),
    'MEGADRIVE': ('md_type',),
    'SUPERFX': ('sfx_type',),
    'JAGUAR': ('jag_type',),
    'THREEDO': ('tdo_type',),
    'RENDERWARE': ('rw_type', 'rw_platform', 'rw_specular', 'rw_matfx',
                   'rw_dual_blend'),
}

#: the sockets each machine's panel shows beside the shared ones
MACHINE_SOCKETS = {
    'GAMECUBE': ('Specular Color', 'Specular Level', 'Glossiness', 'Soften'),
    'MODEL2': ('Specular Color', 'Specular Level', 'Fog Burn-Through'),
    'MODEL3': ('Specular Color', 'Specular Level', 'Fog Burn-Through',
               'Fog Bias', 'Fog Bank'),
    'DS': ('Specular Color', 'Specular Level', 'Glossiness', 'Toon Size'),
    'PS1': ('Specular Color', 'Specular Level', 'Glossiness'),
    'PS2': ('Specular Color', 'Specular Level', 'Glossiness'),
    'PSP': ('Specular Color', 'Specular Level', 'Glossiness'),
    'SATURN': ('Specular Color', 'Specular Level', 'Glossiness'),
    'N64': ('Specular Color', 'Specular Level', 'Glossiness'),
    'SYSTEM22': ('Specular Color', 'Specular Level', 'Glossiness',
                 'Fog Burn-Through'),
    'DREAMCAST': ('Specular Color', 'Specular Level', 'Glossiness'),
    'PC_FIXED': ('Specular Color', 'Specular Level', 'Glossiness'),
    'PCX': ('Specular Color', 'Specular Level', 'Glossiness'),
    'MEGADRIVE': ('Specular Color', 'Specular Level', 'Glossiness'),
    'SUPERFX': (),
    'JAGUAR': ('Specular Color', 'Specular Level', 'Glossiness'),
    'THREEDO': (),
    'RENDERWARE': ('Specular Color', 'Specular Level', 'Glossiness',
                   'Prelit Color', 'Night Color', 'Night Blend',
                   'Env Map', 'Env Map Coefficient', 'Dual Texture'),
}

#: the engine models this node can name (every one in MODEL_ITEMS)
CONSOLE_MODELS = frozenset({
    'GX_LIGHT', 'SEGA_MODEL2', 'SEGA_MODEL3', 'DS_FIXED', 'DS_TOON',
    'DS_HIGHLIGHT', 'DS_DECAL', 'PS1_MODULATE', 'PS2_HIGHLIGHT',
    'DECAL_ALPHA', 'ADD8_COMBINE', 'FLAT_GL_LAST', 'FLAT_D3D_FIRST',
    'SATURN_ADD', 'N64_COMBINE', 'N64_SHADE', 'N64_BLENDRGBA',
    'S22_MODULATE', 'D3D_SEPARATE_SPEC', 'GOURAUD', 'PCX_INTENSITY',
    'MEGA_DRIVE_SH', 'SUPERFX_PLOT', 'JAGUAR_CRY', 'THREEDO_PIXC',
    'RENDERWARE_PS2', 'RENDERWARE_GC', 'RENDERWARE_PC'})

#: the engine models the master shader hands to this node at file load
#: (R252: every console / period model a saved master node may carry),
#: each with the node properties that reproduce it exactly
MIGRATE = {
    'GX_LIGHT': {'console': 'GAMECUBE', 'rate': 'SCENE'},
    'SEGA_MODEL2': {'console': 'MODEL2', 'rate': 'SCENE'},
    'SEGA_MODEL3': {'console': 'MODEL3', 'rate': 'SCENE'},
    'DS_FIXED': {'console': 'DS', 'ds_type': 'MODULATE', 'ds_table': 'CUSTOM',
                 'rate': 'SCENE'},
    'DS_TOON': {'console': 'DS', 'ds_type': 'TOON', 'ds_table': 'CUSTOM',
                'rate': 'SCENE'},
    'DS_HIGHLIGHT': {'console': 'DS', 'ds_type': 'HIGHLIGHT',
                     'ds_table': 'CUSTOM', 'rate': 'SCENE'},
    'FLAT_GL_LAST': {'console': 'PC_FIXED', 'pc_type': 'FLAT_GL',
                     'rate': 'SCENE'},
    'FLAT_D3D_FIRST': {'console': 'PC_FIXED', 'pc_type': 'FLAT_D3D',
                       'rate': 'SCENE'},
    'PS1_MODULATE': {'console': 'PS1', 'ps1_type': 'GOURAUD', 'rate': 'SCENE'},
    'PS2_HIGHLIGHT': {'console': 'PS2', 'ps2_type': 'HIGHLIGHT',
                      'rate': 'SCENE'},
    'SATURN_ADD': {'console': 'SATURN', 'sat_type': 'GOURAUD', 'rate': 'SCENE'},
    'N64_COMBINE': {'console': 'N64', 'n64_type': 'MODULATE', 'rate': 'SCENE'},
    'S22_MODULATE': {'console': 'SYSTEM22', 's22_type': 'LIT', 'rate': 'SCENE'},
    'D3D_SEPARATE_SPEC': {'console': 'PC_FIXED', 'pc_type': 'GOURAUD_SEP',
                          'rate': 'SCENE'},
    'PCX_INTENSITY': {'console': 'PCX', 'rate': 'SCENE'},
    'MEGA_DRIVE_SH': {'console': 'MEGADRIVE', 'md_type': 'AUTO',
                      'rate': 'SCENE'},
    'SUPERFX_PLOT': {'console': 'SUPERFX', 'sfx_type': 'DITHER',
                     'rate': 'SCENE'},
}

#: the combine-stage options' defaults (core/combine.py and gpu/combine.py
#: read the dict with exactly these fallbacks)
COMBINE_DEFAULTS = {'luma_gamma': 1.0, 'saturn_half': False, 'texop': 'MOD',
                    'pcx_first': False, 'md_class': -1.0, 'sfx_dither': True,
                    'rw_matfx': 'NONE', 'rw_dual': 'MODULATE'}

_M2_GLOSS = {'P1': 1.0, 'P2': 2.0, 'P4': 4.0, 'P8': 8.0}
_M3_GLOSS = {'P8': 8.0, 'P16': 16.0, 'P32': 32.0, 'P64': 64.0}
_DS_GLOSS = {'LINEAR': 8.0, 'SOFT': 16.0, 'PIN': 64.0}
_GC_DIFF = {'CLAMP': 0.0, 'SIGN': 1.0, 'NONE': 2.0}
_GC_ATTN = {'SPEC': 0.0, 'SPOT': 1.0, 'NONE': 2.0}
_MD_CLASS = {'AUTO': -1.0, 'NORMAL': 0.0, 'SHADOW': 1.0, 'HIGHLIGHT': 2.0}
_PC_TEXOP = {'MODULATE': 'MOD', 'MODULATE2X': 'MOD2X', 'MODULATE4X': 'MOD4X',
             'ADD': 'ADD', 'ADDSIGNED': 'ADDSIGNED'}
_RW_MODEL = {'PS2': 'RENDERWARE_PS2', 'GC': 'RENDERWARE_GC',
             'PC': 'RENDERWARE_PC'}


def _p(props, key):
    """A property with its declared default when the node predates it."""
    v = (props or {}).get(key)
    if v is None:
        for name, _items, default in ENUM_PROPS:
            if name == key:
                return default
        for name, _kind, default in SCALAR_PROPS:
            if name == key:
                return default
        return None
    return v


def resolve(props):
    """One node's properties -> what the renderer reads.

    Returns a dict:
      model        the engine model name (MODEL_ITEMS)
      fixed_shade  1.0 when no light is evaluated (the colour shown as is)
      light_limit  0, or the machine's light count when the toggle is on
      axis_viewer  1.0 when the material's highlights take the camera axis
      vmix         None, or 1.0 when the vertex colour IS the material
      gx_diff_fn / gx_attn_fn / sun_clamp   the light-unit fields
      gloss        None, or the Glossiness the type pins
      spec_level   None, or the Specular Level the type pins (0 = off)
      opacity      None, or the Opacity the type pins
      alpha_steps  0, or the step count the polygon alpha quantises to
      rate         None (the scene's), 'VERTEX' or 'FACE'
      combine      the combine-stage options core/combine.py reads
      prelit       'ADD' (added to the lighting), 'ONLY', or None
      untextured   True when the Diffuse Color's link is ignored (the
                   N64's G_CC_SHADE: the flat colour, no texel)
    """
    p = props or {}
    con = str(_p(p, 'console'))
    out = {'model': 'PS1_MODULATE', 'fixed_shade': 0.0, 'light_limit': 0,
           'axis_viewer': 0.0, 'vmix': None, 'gx_diff_fn': 0.0,
           'gx_attn_fn': 0.0, 'sun_clamp': 1.0, 'gloss': None,
           'spec_level': None, 'opacity': None, 'alpha_steps': 0,
           'rate': None, 'combine': {}, 'prelit': None,
           'untextured': False}
    cmb = out['combine']
    if con == 'GAMECUBE':
        out['model'] = 'GX_LIGHT'
        t = str(_p(p, 'gc_type'))
        out['gx_diff_fn'] = _GC_DIFF.get(str(_p(p, 'gc_diffuse_fn')), 0.0)
        out['gx_attn_fn'] = _GC_ATTN.get(str(_p(p, 'gc_attn_fn')), 0.0)
        if str(_p(p, 'gc_material_src')) == 'VTX' or t == 'VERTEX':
            out['vmix'] = 1.0
        if t == 'VERTEX':
            out['fixed_shade'] = 1.0
    elif con == 'MODEL2':
        out['model'] = 'SEGA_MODEL2'
        sp = str(_p(p, 'm2_specular'))
        if sp == 'OFF':
            out['spec_level'] = 0.0
        else:
            out['gloss'] = _M2_GLOSS.get(sp, 4.0)
        if str(_p(p, 'm2_type')) == 'FIXED':
            out['fixed_shade'] = 1.0
        cmb['luma_gamma'] = float(_p(p, 'luma_gamma') or 1.0)
    elif con == 'MODEL3':
        out['model'] = 'SEGA_MODEL3'
        sp = str(_p(p, 'm3_specular'))
        if sp == 'OFF':
            out['spec_level'] = 0.0
        else:
            out['gloss'] = _M3_GLOSS.get(sp, 16.0)
        if str(_p(p, 'm3_type')) == 'FIXED':
            out['fixed_shade'] = 1.0
            out['vmix'] = 1.0
        out['sun_clamp'] = 1.0 if bool(_p(p, 'm3_sun_clamp')) else 0.0
        if bool(_p(p, 'm3_alpha_steps')):
            out['alpha_steps'] = 32
        cmb['luma_gamma'] = float(_p(p, 'luma_gamma') or 1.0)
    elif con == 'DS':
        t = str(_p(p, 'ds_type'))
        out['model'] = {'MODULATE': 'DS_FIXED', 'DECAL': 'DS_DECAL',
                        'TOON': 'DS_TOON', 'HIGHLIGHT': 'DS_HIGHLIGHT'
                        }.get(t, 'DS_FIXED')
        tb = str(_p(p, 'ds_table'))
        if tb in _DS_GLOSS:
            out['gloss'] = _DS_GLOSS[tb]
    elif con == 'PS1':
        out['model'] = 'PS1_MODULATE'
        t = str(_p(p, 'ps1_type'))
        if t == 'RAW':
            out['fixed_shade'] = 1.0
        elif t == 'FLAT':
            out['rate'] = 'FACE'
    elif con == 'PS2':
        t = str(_p(p, 'ps2_type'))
        if t in ('HIGHLIGHT', 'HIGHLIGHT2'):
            out['model'] = 'PS2_HIGHLIGHT'
        else:
            out['model'] = 'PS1_MODULATE'
            if t == 'DECAL':
                out['fixed_shade'] = 1.0
    elif con == 'PSP':
        t = str(_p(p, 'psp_type'))
        out['model'] = {'MODULATE': 'PS1_MODULATE', 'DECAL': 'DECAL_ALPHA',
                        'REPLACE': 'PS1_MODULATE', 'ADD': 'ADD8_COMBINE',
                        'FLAT': 'FLAT_GL_LAST'}.get(t, 'PS1_MODULATE')
        if t == 'REPLACE':
            out['fixed_shade'] = 1.0
    elif con == 'SATURN':
        out['model'] = 'SATURN_ADD'
        t = str(_p(p, 'sat_type'))
        if t == 'REPLACE':
            out['fixed_shade'] = 1.0
        elif t == 'HALF_LUM':
            cmb['saturn_half'] = True
        elif t == 'MESH':
            out['opacity'] = 0.5
    elif con == 'N64':
        t = str(_p(p, 'n64_type'))
        out['model'] = {'MODULATE': 'N64_COMBINE', 'DECAL': 'N64_COMBINE',
                        'SHADE': 'N64_SHADE', 'BLENDRGBA': 'N64_BLENDRGBA',
                        'VERTEX': 'N64_COMBINE'}.get(t, 'N64_COMBINE')
        if t == 'DECAL':
            out['fixed_shade'] = 1.0
        elif t == 'VERTEX':
            out['fixed_shade'] = 1.0
            out['vmix'] = 1.0
        elif t == 'SHADE':
            # G_CC_SHADE: the texture ignored -- the Diffuse Color's
            # flat value (its link dropped) stands in for the lights'
            # colours
            out['untextured'] = True
    elif con == 'SYSTEM22':
        out['model'] = 'S22_MODULATE'
        if str(_p(p, 's22_type')) == 'FIXED':
            out['fixed_shade'] = 1.0
    elif con == 'DREAMCAST':
        t = str(_p(p, 'dc_type'))
        out['model'] = {'PACKED': 'D3D_SEPARATE_SPEC',
                        'INTENSITY': 'PCX_INTENSITY',
                        'DECAL': 'D3D_SEPARATE_SPEC',
                        'FLAT': 'FLAT_GL_LAST'}.get(t, 'D3D_SEPARATE_SPEC')
        if t == 'DECAL':
            out['fixed_shade'] = 1.0
            out['spec_level'] = 0.0
    elif con == 'PC_FIXED':
        t = str(_p(p, 'pc_type'))
        out['model'] = {'GOURAUD_SEP': 'D3D_SEPARATE_SPEC',
                        'GOURAUD': 'GOURAUD', 'FLAT_GL': 'FLAT_GL_LAST',
                        'FLAT_D3D': 'FLAT_D3D_FIRST'}.get(t, 'D3D_SEPARATE_SPEC')
        cmb['texop'] = _PC_TEXOP.get(str(_p(p, 'pc_texture_op')), 'MOD')
        if not bool(_p(p, 'pc_local_viewer')):
            out['axis_viewer'] = 1.0
        if bool(_p(p, 'pc_color_vertex')):
            out['vmix'] = 1.0
    elif con == 'PCX':
        out['model'] = 'PCX_INTENSITY'
        cmb['pcx_first'] = str(_p(p, 'pcx_base')) == 'FIRST'
    elif con == 'MEGADRIVE':
        out['model'] = 'MEGA_DRIVE_SH'
        cmb['md_class'] = _MD_CLASS.get(str(_p(p, 'md_type')), -1.0)
    elif con == 'SUPERFX':
        out['model'] = 'SUPERFX_PLOT'
        cmb['sfx_dither'] = str(_p(p, 'sfx_type')) != 'SOLID'
    elif con == 'JAGUAR':
        out['model'] = 'JAGUAR_CRY'
        if str(_p(p, 'jag_type')) == 'FLAT':
            out['rate'] = 'FACE'
    elif con == 'THREEDO':
        out['model'] = 'THREEDO_PIXC'
    elif con == 'RENDERWARE':
        out['model'] = _RW_MODEL.get(str(_p(p, 'rw_platform')), 'RENDERWARE_PS2')
        t = str(_p(p, 'rw_type'))
        if t == 'PRELIT':
            out['prelit'] = 'ADD'
        elif t == 'UNLIT':
            out['prelit'] = 'ONLY'
            out['fixed_shade'] = 1.0
        if str(_p(p, 'rw_specular')) != 'PLUGIN':
            out['spec_level'] = 0.0
        fx = str(_p(p, 'rw_matfx'))
        cmb['rw_matfx'] = fx
        cmb['rw_dual'] = str(_p(p, 'rw_dual_blend'))
    # the shared options
    if bool(_p(p, 'light_limit')):
        out['light_limit'] = int(LIGHT_LIMIT.get(con, 0))
    rate = str(_p(p, 'rate'))
    if out['rate'] is None:
        # a type that IS a rate (a flat primitive) keeps it whatever the
        # menu says; the others take the menu's choice
        if rate == 'MACHINE':
            out['rate'] = MACHINE_RATE.get(con)
        elif rate in ('VERTEX', 'FACE'):
            out['rate'] = rate
    # the combine-stage dict carries what differs from the machine's
    # default only (the hook and the bake read it with these defaults)
    for k, v in list(cmb.items()):
        if COMBINE_DEFAULTS.get(k, object()) == v:
            del cmb[k]
    return out


def model_of(props):
    """The engine model a node's properties name (render.material_model)."""
    return resolve(props)['model']


def rate_of(props):
    """The per-material rate a node asks for ('VERTEX' / 'FACE'), or None
    when the scene's rate decides (combine.rate_for_model)."""
    return resolve(props)['rate']


def combine_opts(props):
    """The combine-stage options for one node (combine.recombine and the
    GPU pass's bake read the same dict)."""
    return dict(resolve(props)['combine'])


def console_node(graph):
    """The Console Emulation Shader node of an exported graph, or None."""
    if not graph:
        return None
    for node in (graph.get('nodes') or {}).values():
        if node.get('bl_idname') == 'HALCYON_ConsoleShaderNode':
            return node
    return None


def label_of(props):
    """'<machine> / <type>' for the UI's material list."""
    p = props or {}
    con = str(_p(p, 'console'))
    lab = next((b for a, b, _c in CONSOLE_ITEMS if a == con), con)
    lab = lab.split(' (')[0]
    panel = PANEL.get(con, ())
    if panel:
        prop = panel[0]
        items = next((it for nm, it, _d in ENUM_PROPS if nm == prop), ())
        val = str(_p(p, prop))
        tlab = next((b for a, b, _c in items if a == val), val)
        return f'{lab} / {tlab.split(" (")[0]}'
    return lab
