"""Halcyon's own shader nodes.

The coded-shader node is the interesting one: it compiles real GLSL or HLSL and
*reads the uniform declarations back out* to build its input sockets. Declare
`uniform float rimPower = 2.5;` and a Rim Power socket appears, defaulted to
2.5. Declare `out vec4 Color;` and an output socket appears. The node is driven
by the shader, not by a fixed set of slots the user has to map onto.
"""

import bpy
from bpy.props import (BoolProperty, CollectionProperty, EnumProperty,
                       FloatProperty, FloatVectorProperty, IntProperty,
                       PointerProperty, StringProperty)
from bpy.types import Node, NodeSocket, PropertyGroup

from ..core.shading import (MODEL_ITEMS, MASTER_MODELS, MOVED_MODELS,
                            master_model_items)
from ..core import console as CON
from ..core.volume import VOLUME_MODEL_ITEMS, VOLUME_SHAPE_ITEMS
from ..core.celfield import (CEL_LIGHT_ITEMS, CEL_RIM_MODE_ITEMS,
                             CEL_RIM_SIDE_ITEMS, CEL_SHAPE_ITEMS)
from ..core.nodeeval import FACE_AXIS_ITEMS
from ..core.shading import (CARTOON_SHADOW_MODE_ITEMS,
                            CARTOON_ERA_ITEMS, CARTOON_ERA_PRESETS,
                            ANIME_STYLE_ITEMS, ANIME_STYLE_PRESETS,
                            HAIR_SHINE_SHAPE_ITEMS)
from ..shaders.compiler import DEFAULT_GLSL, DEFAULT_HLSL, try_compile

ENGINE = 'HALCYON_RENDER'


class HalcyonNodeBase:
    """Shared behaviour: only show up in shader trees."""

    @classmethod
    def poll(cls, tree):
        return tree.bl_idname in ('ShaderNodeTree',)


# =========================================================== classic shader


# R241: every socket of the cel masters and the volume master carries a
# tooltip too. The blocks the two cel masters share are written once.
_CEL_LIGHT_SOCKET_DOCS = {
    'Light Azimuth':
        "Where the fixed key sits, in degrees. Under Camera Key it is "
        "measured about the screen (0 from the viewer, positive from "
        "the screen's left, the drawing's classic key at 35); under "
        "World Key, from the world +X axis counter-clockwise. Ignored "
        "under Scene Lamps",
    'Light Elevation':
        "How high the fixed key sits, in degrees above the screen's "
        "centre (Camera Key) or the horizon (World Key). The classic "
        "drawing key sits around 30. Ignored under Scene Lamps",
    'Screen Shadow':
        "The drawn contact shadow's strength: the hair's shadow across "
        "the brow, the chin's on the neck, marched from the frame's own "
        "depth toward the key. 0 is off; 1 pushes the shadowed pixels "
        "fully into the shadow tone. Under Scene Lamps it rides the "
        "first non-ambient lamp",
    'Screen Shadow Length':
        "How far the contact-shadow march reaches, in pixels at 1080 "
        "lines (it scales with the frame's height). Short lengths keep "
        "the shadows tight and drawn-looking; long ones let a body "
        "shadow the ground",
    'Rim Width':
        "The depth rim's band width in pixels at 1080 lines (it scales "
        "with the frame's height) -- how far inside the silhouette the "
        "line of light reaches when Rim Mode is Depth Rim",
}

_CEL_HAIR_SOCKET_DOCS = {
    'Hair Shine':
        "The painted hair band's strength -- the 80s 'angel ring': a "
        "band of the shine colour painted across the object, "
        "light-independent (it was painted on the cel at a position, "
        "not lit). 0 is off, 1 paints the band at full",
    'Hair Shine Color':
        "The band's paint. Classic rings are white or a pale tint of "
        "the hair's own colour; a linked texture varies it per pixel",
    'Hair Shine Height':
        "Where the band sits, as a fraction of the object's own height "
        "(0 the bottom, 1 the top). The classic ring sits just above "
        "three quarters",
    'Hair Shine Width':
        "The band's thickness, as a fraction of the object's height. "
        "Television rings ran thin; the OVA glamour pass wore them "
        "wider",
    'Hair Shine Wave':
        "How far the band's edge swings up and down as it travels "
        "around the head, as a fraction of the object's height. 0 is a "
        "dead-level band",
    'Hair Shine Waves':
        "How many swings the edge makes in one full turn around the "
        "head. The 80s ring waved five or six times; the 90s made "
        "three bolder ones",
    'Hair Shine Softness':
        "The band edge's blur, as a fraction of the object's height. "
        "Keep it tiny for the crisp cel look; raise it for an "
        "airbrushed sheen",
    'Hair Shine Second':
        "Puts a thinner echo band this far BELOW the main one (a "
        "fraction of the object's height; 0 is off) -- the double ring "
        "of the richer productions",
    'Hair Shine Angle':
        "Turns the wave pattern around the head, in degrees -- where "
        "the teeth and notches sit. Use it to put the pattern's notch "
        "at the front of the hair, or to animate a slow shimmer",
    'Hair Shine Follow':
        "How much the band rides the key light's height: 0 leaves it "
        "painted where it is (the classic per-cut drawing); 1 lifts "
        "and lowers it with the key, as if redrawn for every lighting "
        "change. Follows a directional key (a Sun or Hemi scene key, "
        "or the material's Camera / World key); a point-family key "
        "leaves the band painted",
    'Hair Shine Second Color':
        "The echo band's own tint, multiplied over the Hair Shine "
        "Color where the second band owns the pixel. White keeps both "
        "bands the same paint",
}

_CEL_AIR_SOCKET_DOCS = {
    'Airbrush':
        "The cel painter's soft gradation against the shadow edge: the "
        "lit tone tinted toward the airbrush colour just above the "
        "edge, or the shadow tone blended toward it below (the side is "
        "the Airbrush Side menu). 0 is off",
    'Airbrush Color':
        "The gradation's tint. Warm rose against a cool shadow is the "
        "classic feature-cel choice; a linked texture varies it per "
        "pixel",
    'Airbrush Width':
        "How far the gradation reaches from the shadow edge, as a "
        "share of the light term's range -- wider washes further into "
        "the lit or shadow side",
}

ANIME_SOCKET_DOCS = dict({
    'Diffuse Color':
        "The material's own paint under the lit bands. Plug the "
        "character's base (diffuse) texture in here; the shadow "
        "colours below multiply it, the games' own arrangement",
    'Line Art':
        "A drawn-line texture multiplied over the base -- the inner "
        "lines drawn in the texture itself (an ILM alpha, a hand-drawn "
        "line map). White is no line",
    'Shadow 1 Color':
        "The first shadow band's tint, multiplied over the paint -- "
        "the cel painter's kage colour, picked a step darker and "
        "usually cooler than the local colour. A shadow is a colour "
        "here, never a darkness",
    'Shadow 1 Threshold':
        "Where the first band's edge falls on the wrapped light term "
        "(0.5 is the terminator; higher pushes the shadow onto the lit "
        "side)",
    'Shadow 1 Softness':
        "The first band edge's blur. The cel tradition cuts hard "
        "(0.01 and under); features and the modern digital look "
        "soften a little",
    'Shadow 2 Color':
        "The second, deeper shadow band's tint (Tones: Three) -- the "
        "core shadow inside the first",
    'Shadow 2 Threshold':
        "Where the second band's edge falls on the light term, deeper "
        "than the first (a smaller number)",
    'Shadow 2 Softness':
        "The second band edge's blur, usually matched to the first",
    'Shadow Ramp':
        "Link a texture -- the game's own ramp, a ColorRamp, any chain "
        "-- and the tone bands come from the RAMP instead: shadow "
        "side, transition and lit side all painted, sampled by the "
        "light term. The tone sliders above stand down while linked. "
        "Under SPARKING the strip is read DOWN its height (white at "
        "the top = lit) by the half-Lambert cosine, Sparking! ZERO's "
        "GradientTexture as exported",
    'Ramp Row':
        "Which row of a multi-row ramp texture to sample (0 the "
        "bottom). Under GENSHIN with a Game Texture linked, the "
        "texture's alpha (the material id) picks the row by itself",
    'Shadow Bias':
        "Shifts the whole light term before the bands read it: "
        "positive pushes pixels toward the lit side, negative into "
        "shadow. The game modes add their own texture-driven bias on "
        "top",
    'Game Texture':
        "The game's own packed control map, decoded by the "
        "Compatibility mode: the ArcSys lineage's ILM (specular "
        "intensity / shadow bias / highlight size / drawn lines), the "
        "HoYo lightmap, the ZZZ map, Sparking! ZERO's Mask1 (the "
        "greyscale line-art sheet that multiplies the flat colour). "
        "Leave unlinked outside the game modes",
    'Detail Texture':
        "The companion map the mode expects: the ArcSys lineage's SSS "
        "map (its colour multiplies the first shadow tint); ZZZ reads "
        "its blue as extra specular mask. SPARKING has no companion "
        "map and does not read this socket",
    'Specular Color':
        "The stepped cel highlight's paint. Keep it near white for "
        "hair and metal glints",
    'Specular Level':
        "The stepped highlight's strength. Cel paint itself is matte "
        "-- keep it 0 except on hair, metal and eyes",
    'Specular Size':
        "The highlight's angular size on the wrapped half-vector: "
        "bigger paints a larger glint",
    'Specular Sharpness':
        "The highlight edge's blur -- tiny for the hard cel glint, "
        "larger for a soft sheen",
    'Rim Color':
        "The rim light's paint -- the painted edge-light of the OVA "
        "look, or the modern depth rim's line of light",
    'Rim Amount':
        "The rim's strength. 0 is off; with Rim Mode Depth Rim it "
        "scales the constant-width band inside the silhouette",
    'Rim Power':
        "The Fresnel rim's falloff exponent: higher keeps the rim "
        "tighter to the silhouette (Depth Rim ignores it -- its width "
        "is Rim Width, in pixels)",
    'Matcap':
        "A material-capture image applied by the view normal (the "
        "sphere convention). Use for painterly sheens the lamps "
        "cannot give; blended in by Matcap Blend",
    'Matcap Blend':
        "How much of the matcap lands on the surface, by the Matcap "
        "Mode menu's blend",
    'Self-Illumination':
        "Light the surface emits on its own, added after the bands -- "
        "glowing eyes, runes, screens. Scaled by Emission Strength",
    'Emission Strength':
        "Multiplies Self-Illumination (and, with Alpha Is Emission "
        "on, the base texture's alpha-masked glow -- the HoYo "
        "convention)",
    'Light Response':
        "Per-material gain on every lamp's contribution: the anime "
        "compositor's per-character light dial. 1 is physical; below "
        "flattens the character against the scene's lighting",
    'Ambient':
        "How much of the world's ambient light reaches the material. "
        "The cel look usually keeps a healthy floor so shadows stay "
        "coloured, never black",
    'Opacity':
        "The material's coverage: 1 solid, toward 0 see-through "
        "(needs a transparency mode in the render settings)",
    'Normal':
        "A replacement shading normal (a Bump or Normal Map chain). "
        "Unlinked, the mesh's own -- including any custom split "
        "normals you authored",
    'Bump Strength':
        "How far the linked Normal chain may bend the surface's "
        "shading normal (0 ignores it, 1 takes it fully)",
    'Shadow Smoothing':
        "The inker's simplification: bends the shading normal toward "
        "the Smoothing Shape (sphere, cylinder or the camera) so the "
        "terminator sweeps as one clean drawn shape instead of "
        "following every bump",
    'Line Color':
        "This material's own ink colour, used when the Line Colour "
        "menu says Line Color Socket (read per material, not per "
        "pixel)",
    'Line Darken':
        "Under Iro-Trace, how far the surface's own colour is darkened "
        "to make its line -- the 80s coloured trace: hair lines in the "
        "hair's tone, skin lines in the skin's",
    'Face Shadow (SDF)':
        "Link the face's shadow-sweep map (the game's own, or any "
        "gradient chain) and it REPLACES the light term on this "
        "material: lit where the map's field outweighs the key's "
        "horizontal angle about the head's frame, mirrored across the "
        "face's centre line -- the anime face's drawn terminator. "
        "Unlinked, nothing changes",
}, **_CEL_LIGHT_SOCKET_DOCS, **_CEL_HAIR_SOCKET_DOCS,
   **_CEL_AIR_SOCKET_DOCS)

CARTOON_SOCKET_DOCS = dict({
    'Paint Color':
        "The cel's flat paint -- the same under every lamp, whatever "
        "its energy. Colour is PAINT here, not light; plug the "
        "painted texture in for patterned surfaces",
    'Shadow Color':
        "The one shadow tone's colour. Under the Transparent mode it "
        "multiplies the paint (the shadow cel's double exposure); "
        "under Painted it REPLACES the paint outright (a second flat "
        "paint, the UPA way)",
    'Shadow Amount':
        "How fully the shadow tone lands: 1 the full cel, less a "
        "lighter exposure of it (Transparent mode mixes part-way)",
    'Shadow Threshold':
        "Where the shadow's edge falls on the wrapped light term (0.5 "
        "the terminator; higher pushes the shadow well onto the lit "
        "side -- the dramatic 40s and 90s look)",
    'Shadow Softness':
        "The shadow edge's blur. 0 is the hard inked edge; a little "
        "softness reads as the feature cel's airbrushed edge",
    'Highlight Color':
        "The painted highlight dot's colour (the brightest cel, "
        "usually just off-white)",
    'Highlight Size':
        "The painted highlight dot's size on the lit side. 0 is none "
        "-- most television eras painted none",
    'Highlight Softness':
        "The highlight dot edge's blur -- keep it tiny for a painted "
        "dot, larger for a sheen",
    'Lamp Influence':
        "How much the lamps' actual energy and colour reach the paint. "
        "0 is pure paint (the classic cel); 1 lets a red lamp redden "
        "the paint like a surface",
}, **{k: v for k, v in ANIME_SOCKET_DOCS.items()
      if k in ('Rim Color', 'Rim Amount', 'Rim Power',
               'Self-Illumination', 'Emission Strength', 'Opacity',
               'Normal', 'Shadow Smoothing')},
   **_CEL_LIGHT_SOCKET_DOCS, **_CEL_HAIR_SOCKET_DOCS,
   **_CEL_AIR_SOCKET_DOCS)

VOLUME_SOCKET_DOCS = {
    'Density':
        "How much stuff fills the container per unit of distance. "
        "Whatever chain feeds it is evaluated INSIDE the volume at "
        "every march step -- textures make smoke",
    'Color':
        "The scattering tint: the colour the fog throws back at the "
        "lamps that reach it",
    'Absorption':
        "How strongly the volume eats the light passing through, on "
        "top of what it scatters -- higher reads darker and smokier",
    'Anisotropy':
        "The scatter direction: 0 even in all directions, positive "
        "forward (halos around backlights, the anime god-ray look), "
        "negative back toward the lamp",
    'Emission Color':
        "Light the volume gives off on its own -- fire, ember glow, "
        "the neon interior. Scaled by Emission Strength",
    'Emission Strength':
        "Multiplies the Emission Color: 0 is an unlit volume, higher "
        "glows through the fog around it",
    'Edge Threshold':
        "The stylized volume models' cut: where the accumulated "
        "density starts to count as a drawn edge (the cel-fog and "
        "banded looks read it)",
    'Edge Softness':
        "The blur of that stylized cut -- 0 the hard drawn edge, "
        "larger an airbrushed one",
    'Bands':
        "How many flat tone bands the banded volume models quantize "
        "into -- the drawn-smoke look's steps",
    'Shadow Tint':
        "The colour the volume's shadowed side leans toward (a cel "
        "shadow for fog: a colour, not a darkness)",
    'Tint Amount':
        "How strongly the Shadow Tint takes hold in the volume's "
        "unlit parts. 0 leaves the plain scattering colour",
}


def _apply_socket_tips(node, docs):
    """R241: put the table's tooltip on every input that has one --
    run at creation AND at the load-post migration, so a node saved
    before the tips grows them the moment its file opens."""
    for sock in node.inputs:
        tip = docs.get(getattr(sock, 'name', None))
        if tip:
            try:
                sock.description = tip
            except (AttributeError, TypeError):
                pass


# What each input does. The list of models that use it is derived from RELEVANT
# below rather than written out again, so the two cannot disagree.
SOCKET_DOCS = {
    'Diffuse Color': "Base colour of the surface under direct light. Plug a "
                     "texture in here for anything patterned",
    'Diffuse Level': "How much of the diffuse term reaches the image. 0 leaves "
                     "only the highlight, which is how chrome is made",
    'Specular Color': "Colour of the highlight. White for plastic and painted "
                      "surfaces; tint it toward the base colour for metal",
    'Specular Level': "Strength of the highlight. 0 gives a completely matte "
                      "surface whatever the model",
    'Glossiness': "Tightness of the highlight, as a cosine exponent. Low values "
                  "give a broad sheen, high values a small hard glint. This is "
                  "the period control -- the microfacet models use Roughness "
                  "instead",
    'Roughness': "Surface microstructure, 0 polished to 1 completely rough. "
                 "Drives the microfacet and rough-diffuse models; the "
                 "cosine-lobe models use Glossiness",
    'Metalness': "Blends the highlight toward the diffuse colour and suppresses "
                 "the diffuse term, so the surface reads as metal rather than "
                 "as a painted object",
    'Anisotropy': "Stretches the highlight along the surface. 0 is round; "
                  "higher values give the streak of brushed metal, hair or "
                  "satin. Negative stretches the other way",
    'Anisotropic Rotation': "Turns the direction the highlight stretches in, in "
                            "radians around the surface normal",
    'Soften': "3D Studio's Soften: rolls the highlight off at grazing angles so "
              "it does not terminate in a hard edge at the silhouette",
    'Ambient': "How strongly the surface picks up the scene's ambient light. "
               "1990s renderers had no bounce light, so this is what keeps "
               "shadowed areas from going black",
    'Self-Illumination': "Colour the surface emits on its own, added after "
                         "lighting. It does not light anything else -- this "
                         "engine has no bounce",
    'Opacity': "1 is solid, 0 is invisible. Anything below 1 sends the surface "
               "through the transparency mode set in Render Properties",
    'IOR': "Index of refraction. Bends rays passing through a transparent "
           "surface, and feeds the Fresnel term of the microfacet models. "
           "Glass is about 1.5, water 1.33",
    'Reflection': "How much of the ray-traced reflection is mixed in. Needs Ray "
                  "Tracing enabled in Render Properties; without it this falls "
                  "back to the environment colour",
    'Translucency': "How much light passes through from behind. Paper, leaves "
                    "and lampshades",
    'Toon Size': "Where the light-to-dark step falls, as a fraction of the "
                 "diffuse range; under the DS Toon / DS Highlight models it "
                 "and Toon Steps also shape the material's 32-entry DS table "
                 "(hard-edged: the DS had no smooth band)",
    'Toon Smooth': "How soft that step is. 0 is a hard cel edge",
    'Normal': "Replaces the shading normal, for normal and bump mapping",
    'Fresnel': "Brightens the highlight toward the silhouette, the way a real "
               "surface reflects more at grazing angles. The cheapest way to "
               "stop a plastic surface looking flat",
    'Fresnel Power': "How tightly the Fresnel effect hugs the silhouette. "
                     "Higher values confine it to a thinner edge",
    'Fresnel Color': "Tint of the Fresnel boost. White for a clear coat, or "
                     "tint it for anodised metal and soap-film effects",
    'Rim Light': "Colour of the rim term added at the silhouette",
    'Rim Amount': "Strength of an additive rim light. Unlike Fresnel this does "
                  "not need a light source -- it is the backlight cheat every "
                  "1990s demo used to lift a subject off its background",
    'Rim Power': "How tight the rim band is. Higher confines it to the edge",
    'Matcap': "A sphere-mapped image sampled by the view-space normal, giving "
              "a whole material's worth of lighting from one picture. Feed it "
              "an Image Texture through a Matcap Coordinates node",
    'Matcap Blend': "How much the Matcap replaces the lit result. 1 is pure "
                    "matcap and ignores the scene lights entirely",
    'Reflection Color': "Colour the Reflection amount is multiplied by. Plug an "
                        "environment image in here for a reflection map that "
                        "costs nothing, with no ray tracing needed",
    'Edge Opacity': "Opacity at the silhouette, blended toward by the Fresnel "
                    "curve. Below 1 it thins the edges for holograms; above the "
                    "centre opacity it thickens them, which is how glass reads",
    'Backface Color': "Colour used where a surface faces away from the camera",
    'Vertex Color': "Colour carried on the mesh's own vertices. Leave it "
                    "unlinked and set Vertex Color Mix above zero to use the "
                    "active colour attribute directly -- which is how the "
                    "packages that had this worked, since a vertex colour was "
                    "a property of the model rather than something you routed",
    'Sheen': "A velvet lobe added on top of whichever model is chosen: light "
             "scattered back toward the viewer at grazing angles, which is "
             "what makes velvet, suede and dusty cloth bright at their edges "
             "and dark face-on",
    'Sheen Color': "Colour of the sheen lobe. Real velvet's sheen is close to "
                   "white however deeply the pile is dyed",
    'Sheen Roughness': "Width of the sheen band. 0 confines it to the "
                       "silhouette; 1 spreads it across the whole surface",
    # R251 lighting F006: Model 3 / System 22 per-material fog control
    'Fog Burn-Through': "Model 3's polygon light modifier: the share of the "
                        "fog this material shines through (0 = fogged like "
                        "everything, 1 = never fogged -- the same as the "
                        "Blender Internal node's Mist toggle off), the 5-bit "
                        "header field Daytona 2's neon used",
    'Fog Bias': "System 22's per-polygon cz delta: added to the fog amount "
                "before the clamp, so this material fogs earlier (positive) "
                "or later (negative) than the scene's curve; -1..1",
    'Fog Bank': "System 22's cz bank: 0 reads the scene's Fog Start/End, 1 "
                "reads the Fog Bank 1 Start/End pair in Render Properties > "
                "Fog (inert while that End is not past its Start). System 22 "
                "had four banks; Halcyon ships two. Exponential fog reads "
                "density only and ignores the bank",
    # R251 lighting F019 / F020 / F021: POV-Ray finish
    'Brilliance': "POV-Ray's brilliance: the diffuse cosine raised to this "
                  "power before the model's own diffuse law -- above 1 the lit "
                  "region hugs the light and darkens on the flanks (POV's "
                  "metallic diffuse), below 1 it flattens toward a wrapped "
                  "look. 1.0 is exact Lambert and costs nothing. Inert on the "
                  "3ds Max shaders, whose diffuse laws carry their own colour",
    'Crand': "POV-Ray's crand: a random darkening of the direct diffuse term "
             "per lamp and per pixel -- sandpaper on the lit side only, "
             "highlights and shadows untouched. Halcyon draws it from its "
             "integer hash so a frame renders the same bits every time; a "
             "reflection grains with its parent pixel's number; 0 is off",
    'Metallic (POV)': "POV-Ray's finish metallic: the highlight's colour slides "
                      "from the light's colour toward the diffuse pigment by "
                      "POV's rational Fresnel of N.L -- pigment-coloured facing "
                      "the light, the light's own colour at grazing. 0 is off; "
                      "POV's Metal model and Reflection Tint tint uniformly, "
                      "this does not. Inert on the Anime and Cartoon models and "
                      "the Hemi lamp, which paint their highlights in the lamp "
                      "loop",
    'Bump Strength': "Scales how far the Normal input is allowed to bend the "
                     "shading normal away from the surface. 0 ignores the bump "
                     "entirely, 1 uses it as given, above 1 exaggerates it",
    'Bump Height': "A greyscale height field bumped straight into the shading "
                   "normal -- plug any texture here and its bright parts rise. "
                   "Behind the scenes this is exactly a Bump node between the "
                   "texture and Normal, scaled by Bump Strength, so it renders "
                   "identically on both devices. Unlinked, it does nothing",
    'Refraction Amount': "How much of the ray traced *through* a transparent "
                         "surface is used. 1 is glass; lower values keep what "
                         "is behind the surface where it is, which is how a "
                         "scanline renderer's alpha blend looked",
    'Vertex Color Mix': "How much the vertex colour replaces Diffuse Color. At "
                        "1 the mesh's colours are the surface colour outright, "
                        "which is what the flat-shaded era used them for",
    'Backface Mix': "How strongly Backface Color replaces the normal shading on "
                    "back faces. Useful on single-sided leaves, cloth and cards",
    'Specular Color 2': "Multi-Layer (3ds Max)'s second highlight colour -- the "
                        "wider, softer lobe that shows through what the first "
                        "leaves. Max's Second Specular Layer",
    'Specular Level 2': "Strength of the second highlight, Max's percent over "
                        "100 (0 turns the layer off, as Max's default does)",
    'Glossiness 2': "Sharpness of the second highlight, Max's percent -- "
                    "lower than the first for the classic hot-dot-in-a-glow",
    'Anisotropy 2': "How far the second highlight stretches, 0 round to 1 a "
                    "streak -- Max's Anisotropy on the second layer",
    'Anisotropic Rotation 2': "Turns the second highlight's stretch about the "
                              "normal, in turns -- Max's Orientation on the "
                              "second layer",
    'Translucent Color': "Translucent (3ds Max)'s colour: light from either "
                         "side of the surface scatters through as this tint, "
                         "strongest where the diffuse is unlit. Black is opaque",
}


# Which models each input genuinely affects. ALL means the parameter is applied
# outside the reflectance function -- in the lighting loop -- so it works the
# same whichever model is chosen. The rest were measured by perturbing each
# parameter and seeing which models changed their output, and a test re-runs
# that measurement against this table so it cannot rot.
ALL = '*'

SOCKET_MODELS = {
    'Diffuse Color': ALL,
    'Diffuse Level': ALL,
    # Strauss derives its highlight colour from metalness and the base colour
    # rather than reading this socket; Toon does read it.
    'Specular Color': ('GOURAUD', 'FLAT', 'PHONG', 'BLINN_PHONG', 'BLINN',
                       'COOK_TORRANCE', 'WARD', 'ANISOTROPIC', 'MULTI_LAYER',
                       'TOON', 'BI_COOKTORR', 'BI_PHONG', 'BI_BLINN',
                       'OREN_NAYAR_BLINN',
                       # R243: Max's own shaders (Metal and Strauss colour
                       # their highlight from the diffuse)
                       'MAX_PHONG', 'MAX_BLINN', 'MAX_ANISOTROPIC',
                       'MAX_MULTI_LAYER', 'MAX_OREN_NAYAR_BLINN',
                       'MAX_TRANSLUCENT',
                       # R251 (LIGHT-B2): the console light units (the DS
                       # lobe reads it beside the evaluate call)
                       'GX_LIGHT', 'SEGA_MODEL2', 'SEGA_MODEL3',
                       'DS_FIXED',
                       # R251 material pack (MAT-A): the period combiners
                       # light their corners with Blinn-Phong (the CPU's
                       # own fallback); the DS pair under the DS_FIXED lobe
                       'FLAT_GL_LAST', 'FLAT_D3D_FIRST', 'PS1_MODULATE',
                       'PS2_HIGHLIGHT', 'SATURN_ADD', 'N64_COMBINE',
                       'S22_MODULATE', 'D3D_SEPARATE_SPEC', 'PCX_INTENSITY',
                       'DS_TOON', 'DS_HIGHLIGHT', 'MEGA_DRIVE_SH',
                       'SUPERFX_PLOT',
                       # R252: the Console Emulation Shader's items light
                       # their corners with the same Blinn-Phong fallback
                       'RENDERWARE_PS2', 'RENDERWARE_GC', 'RENDERWARE_PC',
                       'DS_DECAL', 'DECAL_ALPHA', 'ADD8_COMBINE',
                       'N64_SHADE', 'N64_BLENDRGBA', 'JAGUAR_CRY',
                       'THREEDO_PIXC'),
    'Specular Level': ALL,
    'Glossiness': ('GOURAUD', 'FLAT', 'PHONG', 'BLINN_PHONG', 'BLINN',
                   'ANISOTROPIC', 'METAL', 'STRAUSS', 'MULTI_LAYER',
                   'BI_COOKTORR', 'BI_PHONG', 'BI_BLINN', 'OREN_NAYAR_BLINN',
                   'MAX_PHONG', 'MAX_BLINN', 'MAX_METAL', 'MAX_ANISOTROPIC',
                   'MAX_MULTI_LAYER', 'MAX_OREN_NAYAR_BLINN', 'MAX_STRAUSS',
                   'MAX_TRANSLUCENT',
                   # R251 (LIGHT-B2): GX's s/2 shape, Model 2's 1/2/4/8
                   # snap, Model 3's 8..64 snap, the DS table's exponent
                   'GX_LIGHT', 'SEGA_MODEL2', 'SEGA_MODEL3', 'DS_FIXED',
                   # R251 material pack (MAT-A): Blinn-Phong's exponent at
                   # the corners; the DS pair's table exponent
                   'FLAT_GL_LAST', 'FLAT_D3D_FIRST', 'PS1_MODULATE',
                   'PS2_HIGHLIGHT', 'SATURN_ADD', 'N64_COMBINE',
                   'S22_MODULATE', 'D3D_SEPARATE_SPEC', 'PCX_INTENSITY',
                   'DS_TOON', 'DS_HIGHLIGHT', 'MEGA_DRIVE_SH',
                   'SUPERFX_PLOT',
                   # R252: the Console Emulation Shader's items
                   'RENDERWARE_PS2', 'RENDERWARE_GC', 'RENDERWARE_PC',
                   'DS_DECAL', 'DECAL_ALPHA', 'ADD8_COMBINE',
                   'N64_SHADE', 'N64_BLENDRGBA', 'JAGUAR_CRY',
                   'THREEDO_PIXC'),
    'Roughness': ('COOK_TORRANCE', 'OREN_NAYAR', 'MINNAERT', 'WARD',
                  'OREN_NAYAR_BLINN', 'MAX_MULTI_LAYER', 'MAX_OREN_NAYAR_BLINN'),
    'Metalness': ALL,
    'Anisotropy': ('WARD', 'ANISOTROPIC', 'MAX_ANISOTROPIC', 'MAX_MULTI_LAYER'),
    'Anisotropic Rotation': ('WARD', 'ANISOTROPIC', 'MAX_ANISOTROPIC',
                             'MAX_MULTI_LAYER'),
    'Specular Color 2': ('MAX_MULTI_LAYER',),
    'Specular Level 2': ('MAX_MULTI_LAYER',),
    'Glossiness 2': ('MAX_MULTI_LAYER',),
    'Anisotropy 2': ('MAX_MULTI_LAYER',),
    'Anisotropic Rotation 2': ('MAX_MULTI_LAYER',),
    'Translucent Color': ('MAX_TRANSLUCENT',),
    'Soften': ALL,
    'Ambient': ALL,
    'Self-Illumination': ALL,
    'Opacity': ALL,
    'IOR': ('BLINN', 'COOK_TORRANCE', 'BI_BLINN', 'OREN_NAYAR_BLINN'),
    'Reflection': ALL,
    'Translucency': ('TRANSLUCENT',),
    'Toon Size': ('TOON',),
    'Toon Smooth': ('TOON',),
    'Normal': ALL,
    'Fresnel': ALL, 'Fresnel Power': ALL, 'Fresnel Color': ALL,
    'Rim Light': ALL, 'Rim Amount': ALL, 'Rim Power': ALL,
    'Matcap': ALL, 'Matcap Blend': ALL, 'Reflection Color': ALL,
    'Edge Opacity': ALL, 'Backface Color': ALL, 'Backface Mix': ALL,
    'Vertex Color': ALL, 'Vertex Color Mix': ALL,
    'Sheen': ALL, 'Sheen Color': ALL, 'Sheen Roughness': ALL,
    'Fog Burn-Through': ALL, 'Fog Bias': ALL, 'Fog Bank': ALL,
    'Brilliance': ALL, 'Crand': ALL, 'Metallic (POV)': ALL,
    'Bump Strength': ALL, 'Bump Height': ALL, 'Refraction Amount': ALL,
}

# measured parameters -- the ones a test verifies against the shading code
MEASURED = {'Specular Color', 'Glossiness', 'Roughness', 'Anisotropy',
            'Anisotropic Rotation', 'IOR', 'Toon Size', 'Toon Smooth'}


def _models_using(socket_name, relevant=None, all_models=None):
    """Human-readable note on which models an input affects."""
    who = SOCKET_MODELS.get(socket_name)
    if who is None:
        return ""
    if who == ALL:
        return "affects every model"
    pretty = ', '.join(m.replace('_', ' ').title() for m in who)
    if len(who) == 1:
        return f"only affects {pretty}"
    return f"affects {pretty}"


class HALCYON_BlendValueSocket(NodeSocket):
    """R202: a float amount whose blend-mode menu rides DIRECTLY under
    it -- the Fresnel/Rim/Matcap menus used to sit at the top of the
    node, a screen away from the sliders they belong to. The socket
    exports as a plain VALUE (the kind map's fallback), so the render
    roads never see the difference."""

    bl_idname = 'HALCYON_BlendValueSocket'
    bl_label = "Amount + Blend"

    default_value: FloatProperty(
        name="Amount", default=0.0, min=0.0, soft_max=1.0,
        description="The effect's strength; its blend menu sits "
                    "directly below while the effect is live")
    blend_prop: StringProperty(
        default='',
        description="Name of the node enum drawn under the amount")

    def draw(self, context, layout, node, text):
        col = layout.column(align=True)
        if self.is_output or self.is_linked:
            col.label(text=text)
        else:
            col.prop(self, 'default_value', text=text)
        bp = str(getattr(self, 'blend_prop', '') or '')
        if bp and hasattr(node, bp):
            try:
                live = node._effect_active(self.name)
            except (AttributeError, TypeError):
                live = True
            if live:
                col.prop(node, bp, text="")

    def draw_color(self, context, node):
        # the float socket's grey, because that is what this is
        return (0.63, 0.63, 0.63, 1.0)


#: which node enum each blend-carrying amount socket draws
_BLEND_SOCKET_PROPS = {'Fresnel': 'fresnel_blend',
                       'Rim Amount': 'rim_blend',
                       'Matcap Blend': 'matcap_mode'}


class HALCYON_ShaderNode(Node, HalcyonNodeBase):
    """A 1990s reflectance model with its own controls"""

    bl_idname = 'HALCYON_ShaderNode'
    bl_label = "Halcyon Shader"
    bl_icon = 'SHADING_RENDERED'
    bl_width_default = 200

    def _update(self, context):
        self.refresh_sockets()

    # R252: the menu is the master's own list (core/shading.MASTER_MODELS)
    # -- the anime / cartoon masters, the eight 3ds Max shaders and the
    # period machines each have their own node now. Each item carries its
    # MODEL_ITEMS index as its number, so a file saved with Phong at 3
    # still reads Phong; a file saved with a moved model is rebuilt as the
    # right node at load (_migrate_master_sockets)
    model: EnumProperty(name="Model", items=master_model_items(),
                        default='PHONG', update=_update)
    toon_steps: IntProperty(name="Toon Steps", default=2, min=1, max=16)

    # how each silhouette cheat lands on the lit result. The defaults are
    # the behaviour every scene already has: fresnel and rim ADD, matcap
    # MIXES -- so old files render identically
    _LAYER_BLENDS = (
        ('ADD', "Add", "Added on top of the lit result"),
        ('MIX', "Mix", "Blends the result toward the effect colour"),
        ('MULTIPLY', "Multiply", "Darkens the result by the effect"),
        ('SCREEN', "Screen", "Brightens like projected light, never clips"),
    )
    _MATCAP_BLENDS = (
        ('MIX', "Mix", "Blends the result toward the matcap"),
        ('ADD', "Add", "Added on top of the lit result"),
        ('MULTIPLY', "Multiply", "Darkens the result by the matcap"),
        ('SCREEN', "Screen", "Brightens like projected light, never clips"),
    )
    fresnel_blend: EnumProperty(
        name="Fresnel Blend", items=_LAYER_BLENDS, default='ADD',
        description="How the Fresnel layer combines with the shaded surface")
    rim_blend: EnumProperty(
        name="Rim Blend", items=_LAYER_BLENDS, default='ADD',
        description="How the rim light combines with the shaded surface")
    matcap_mode: EnumProperty(
        name="Matcap Blend", items=_MATCAP_BLENDS, default='MIX',
        description="How the matcap combines with the shaded surface")
    faceted: BoolProperty(
        name="Faceted", default=False,
        description="3ds Max's Faceted: shade every face flat by its own "
                    "face normal, whatever the model -- the un-smoothed "
                    "look of early hardware on a mesh that keeps its "
                    "smooth normals for everything else. Ignores the "
                    "Normal input while on")
    wire_size: FloatProperty(
        name="Wire Size", default=1.0, min=0.05, max=16.0,
        description="Width of the drawn edge, in rendered pixels. A material "
                    "shaded as Wireframe had no reachable width at all before "
                    "-- on a dense mesh at a period resolution every pixel is "
                    "within a pixel of an edge, and the surface fills in")

    # which sockets each model actually uses -- the rest are hidden, not removed,
    # so switching models never loses a connection
    MODEL_ORDER = tuple(MASTER_MODELS)

    RELEVANT = {
        'LAMBERT': {'Diffuse Color', 'Diffuse Level', 'Ambient', 'Opacity',
                    'Self-Illumination', 'Normal', 'Bump Strength', 'Bump Height'},
        'GOURAUD': None, 'FLAT': None, 'PHONG': None, 'BLINN_PHONG': None,
        'BLINN': None, 'COOK_TORRANCE': None,
        'BI_COOKTORR': None, 'BI_PHONG': None,
        'BI_BLINN': None,
        'OREN_NAYAR': {'Diffuse Color', 'Diffuse Level', 'Roughness', 'Ambient',
                       'Opacity', 'Self-Illumination', 'Normal',
                       'Bump Strength', 'Bump Height'},
        'MINNAERT': {'Diffuse Color', 'Diffuse Level', 'Roughness', 'Ambient',
                     'Opacity', 'Self-Illumination', 'Normal',
                     'Bump Strength', 'Bump Height'},
        'WARD': None, 'ANISOTROPIC': None, 'METAL': None, 'STRAUSS': None,
        'MULTI_LAYER': None,
        # R243: Max's own shaders derive their panels from the measured
        # table, like the models they mirror
        'MAX_PHONG': None, 'MAX_BLINN': None, 'MAX_METAL': None,
        'MAX_ANISOTROPIC': None, 'MAX_MULTI_LAYER': None,
        'MAX_OREN_NAYAR_BLINN': None, 'MAX_STRAUSS': None,
        'MAX_TRANSLUCENT': None,
        # R242: Max's matte shader -- the rough diffuse with the Blinn
        # highlight's dials
        'OREN_NAYAR_BLINN': {'Diffuse Color', 'Diffuse Level', 'Roughness',
                             'Specular Color', 'Specular Level',
                             'Glossiness', 'Soften', 'IOR', 'Ambient',
                             'Opacity', 'Self-Illumination', 'Normal',
                             'Bump Strength', 'Bump Height'},
        'TOON': {'Diffuse Color', 'Diffuse Level', 'Specular Color',
                 'Specular Level', 'Toon Size', 'Toon Smooth', 'Ambient',
                 'Opacity', 'Self-Illumination', 'Normal', 'Bump Strength', 'Bump Height'},
        'TRANSLUCENT': {'Diffuse Color', 'Diffuse Level', 'Translucency',
                        'Ambient', 'Opacity', 'Self-Illumination', 'Normal',
                        'Bump Strength', 'Bump Height'},
        'CONSTANT': {'Diffuse Color', 'Opacity', 'Self-Illumination'},
        'WIREFRAME': {'Diffuse Color', 'Opacity'},
        # (R252: the CARTOON row and the period machines' rows left with
        # their models -- the Cartoon Shader and the Console Emulation
        # Shader carry them; the Max rows above stay for the measured
        # SOCKET_MODELS table, which still names every engine model)
    }

    # R202: the order is the panel. Reflection's inputs sit together
    # with Refraction Amount right beside them; Normal leads straight
    # into the two Bump inputs (the bump IS a normal edit); the three
    # blend-carrying amounts are HALCYON_BlendValueSocket so their
    # blend menu draws directly below the slider. `sort_sockets` walks
    # saved nodes into this order at file load.
    SOCKETS = (
        ('NodeSocketColor', 'Diffuse Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketFloat', 'Diffuse Level', 1.0),
        ('NodeSocketColor', 'Specular Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Specular Level', 0.5),
        ('NodeSocketFloat', 'Glossiness', 25.0),
        ('NodeSocketFloat', 'Roughness', 0.3),
        ('NodeSocketFloat', 'Metalness', 0.0),
        ('NodeSocketFloat', 'Anisotropy', 0.0),
        ('NodeSocketFloat', 'Anisotropic Rotation', 0.0),
        ('NodeSocketFloat', 'Soften', 0.0),
        ('NodeSocketFloat', 'Ambient', 1.0),
        ('NodeSocketColor', 'Self-Illumination', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Opacity', 1.0),
        ('NodeSocketFloat', 'IOR', 1.45),
        ('NodeSocketFloat', 'Reflection', 0.0),
        ('NodeSocketColor', 'Reflection Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Refraction Amount', 1.0),
        ('NodeSocketFloat', 'Translucency', 0.0),
        ('NodeSocketFloat', 'Toon Size', 0.5),
        ('NodeSocketFloat', 'Toon Smooth', 0.05),
        ('NodeSocketVector', 'Normal', None),
        ('NodeSocketFloat', 'Bump Strength', 1.0),
        ('NodeSocketFloat', 'Bump Height', 0.5),
        ('HALCYON_BlendValueSocket', 'Fresnel', 0.0),
        ('NodeSocketFloat', 'Fresnel Power', 3.0),
        ('NodeSocketColor', 'Fresnel Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketColor', 'Rim Light', (1.0, 1.0, 1.0, 1.0)),
        ('HALCYON_BlendValueSocket', 'Rim Amount', 0.0),
        ('NodeSocketFloat', 'Rim Power', 3.0),
        ('NodeSocketColor', 'Matcap', (0.0, 0.0, 0.0, 1.0)),
        ('HALCYON_BlendValueSocket', 'Matcap Blend', 0.0),
        ('NodeSocketFloat', 'Edge Opacity', 1.0),
        ('NodeSocketColor', 'Backface Color', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Backface Mix', 0.0),
        ('NodeSocketColor', 'Vertex Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Vertex Color Mix', 0.0),
        ('NodeSocketFloat', 'Sheen', 0.0),
        ('NodeSocketColor', 'Sheen Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Sheen Roughness', 0.3),
        # R251 lighting: the period finish dials (F006 Model 3 / System
        # 22 fog control; F019 / F020 / F021 POV-Ray finish)
        ('NodeSocketFloat', 'Fog Burn-Through', 0.0),
        ('NodeSocketFloat', 'Fog Bias', 0.0),
        ('NodeSocketFloat', 'Fog Bank', 0.0),
        ('NodeSocketFloat', 'Brilliance', 1.0),
        ('NodeSocketFloat', 'Crand', 0.0),
        ('NodeSocketFloat', 'Metallic (POV)', 0.0),
        # R243: the Max Multi-Layer's second highlight and the Max
        # Translucent's colour
        ('NodeSocketColor', 'Specular Color 2', (0.9, 0.9, 0.9, 1.0)),
        ('NodeSocketFloat', 'Specular Level 2', 0.0),
        ('NodeSocketFloat', 'Glossiness 2', 25.0),
        ('NodeSocketFloat', 'Anisotropy 2', 0.0),
        ('NodeSocketFloat', 'Anisotropic Rotation 2', 0.0),
        ('NodeSocketColor', 'Translucent Color', (0.0, 0.0, 0.0, 1.0)),
    )

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            sock = self.inputs.new(kind, name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
            if name in ('Glossiness',):
                sock.default_value = 25.0
            if name in _BLEND_SOCKET_PROPS:
                try:
                    sock.blend_prop = _BLEND_SOCKET_PROPS[name]
                except (AttributeError, TypeError):
                    pass
            self._document(sock, name)
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Collapses to the chosen reflectance "
            "model when Halcyon renders it")
        self.refresh_sockets()

    def _document(self, sock, name):
        """Attach the tooltip. Older builds have no settable socket
        description, so this must never be fatal."""
        doc = SOCKET_DOCS.get(name)
        if not doc:
            return
        models = _models_using(name)
        try:
            sock.description = f"{doc}. ({models})" if models else doc
        except (AttributeError, TypeError):
            pass

    #: applied after the reflectance model, so never hidden
    ALWAYS = ('Fresnel', 'Fresnel Power', 'Fresnel Color', 'Rim Light',
              'Rim Amount', 'Rim Power', 'Matcap', 'Matcap Blend',
              'Reflection Color', 'Edge Opacity', 'Backface Color',
              'Backface Mix', 'Sheen', 'Sheen Color', 'Sheen Roughness',
              'Refraction Amount')

    def ensure_sockets(self):
        """Create any spec socket this saved instance predates.

        NOT called from update callbacks (socket topology changes there
        are forbidden by the guard test, for good reason): the load-post
        migration below runs it at file load, where mutation is safe --
        so an old file gains Bump Height the moment it opens.
        """
        have = {s.name for s in self.inputs}
        for kind, name, default in self.SOCKETS:
            if name in have:
                continue
            try:
                sock = self.inputs.new(kind, name)
                if default is not None:
                    sock.default_value = default
                if name in _BLEND_SOCKET_PROPS:
                    sock.blend_prop = _BLEND_SOCKET_PROPS[name]
            except Exception:                                   # noqa: BLE001
                pass

    def upgrade_blend_sockets(self):
        """R202, load-time only: swap the three blend-carrying amounts
        to HALCYON_BlendValueSocket, preserving value and links.

        Per-socket and best-effort: a socket that cannot be swapped is
        LEFT ALONE and keeps the old top-of-node menu (draw_buttons
        falls back for exactly that case), so a failure here can never
        cost anyone a connection."""
        tree = self.id_data
        for nm, bp in _BLEND_SOCKET_PROPS.items():
            s = self.inputs.get(nm)
            if s is None or \
                    getattr(s, 'bl_idname', '') == 'HALCYON_BlendValueSocket':
                continue
            try:
                val = float(getattr(s, 'default_value', 0.0))
                froms = [ln.from_socket for ln in getattr(s, 'links', ())]
                self.inputs.remove(s)
                ns = self.inputs.new('HALCYON_BlendValueSocket', nm)
                ns.default_value = val
                ns.blend_prop = bp
                self._document(ns, nm)
                for f in froms:
                    tree.links.new(f, ns)
            except Exception:                                   # noqa: BLE001
                pass

    def sort_sockets(self):
        """Walk this saved node's inputs into the SOCKETS order --
        R202 regrouped Reflection/Refraction and Normal/Bump, and an
        old file should see the same panel a new one does. Load-time
        only, like every other topology change."""
        order = [name for _k, name, _d in self.SOCKETS]
        for target, name in enumerate(order):
            if target >= len(self.inputs):
                break
            idx = next((i for i, s in enumerate(self.inputs)
                        if s.name == name), None)
            if idx is not None and idx != target:
                try:
                    self.inputs.move(idx, target)
                except Exception:                               # noqa: BLE001
                    pass

    def refresh_sockets(self):
        keep = self.RELEVANT.get(self.model)
        if keep is None:
            # models the RELEVANT table left open used to show every
            # socket -- a PHONG node offered Roughness, Toon Size and
            # Translucency, none of which its model reads. The measured
            # SOCKET_MODELS table (a test re-derives it by perturbing
            # each input) already knows exactly which sockets each model
            # shades with, so the panel now shows those and only those
            keep = {name for _k, name, _d in self.SOCKETS
                    if SOCKET_MODELS.get(name) == ALL
                    or (SOCKET_MODELS.get(name)
                        and self.model in SOCKET_MODELS[name])}
        keep = set(keep) | set(self.ALWAYS)
        for sock in self.inputs:
            sock.hide = bool(sock.name not in keep and not sock.is_linked)
            self._document(sock, sock.name)

    def model_description(self):
        for ident, _label, desc in MODEL_ITEMS:
            if ident == self.model:
                return desc
        return ""

    def _effect_active(self, name):
        try:
            s = self.inputs.get(name)
            if s is None:
                return False
            if s.is_linked:
                return True
            v = getattr(s, 'default_value', 0.0)
            return float(v) > 1e-4
        except (TypeError, ValueError):
            return False

    def draw_buttons(self, context, layout):
        layout.prop(self, 'model', text="")
        if self.model == 'TOON':
            layout.prop(self, 'toon_steps')
        if self.model == 'WIREFRAME':
            layout.prop(self, 'wire_size')
        if self.model not in ('FLAT', 'WIREFRAME', 'CONSTANT'):
            layout.prop(self, 'faceted')
        # R202: each blend menu draws directly under its own amount
        # slider (the HALCYON_BlendValueSocket does it); a socket the
        # load migration could not swap keeps the old top-of-node menu
        def _legacy(nm):
            s = self.inputs.get(nm)
            return s is not None and \
                getattr(s, 'bl_idname', '') != 'HALCYON_BlendValueSocket'
        if self._effect_active('Fresnel') and _legacy('Fresnel'):
            layout.prop(self, 'fresnel_blend')
        if self._effect_active('Rim Amount') and _legacy('Rim Amount'):
            layout.prop(self, 'rim_blend')
        if self._effect_active('Matcap Blend') and _legacy('Matcap Blend'):
            layout.prop(self, 'matcap_mode')
        used = sum(1 for _k, n, _d in self.SOCKETS
                   if SOCKET_MODELS.get(n) == ALL
                   or (SOCKET_MODELS.get(n) and self.model in SOCKET_MODELS[n]))
        row = layout.row()
        row.active = False
        row.label(text=f"{used} of {len(self.SOCKETS)} inputs used",
                  icon='HIDE_OFF')

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, 'model', text="")
        col = layout.column(align=True)
        col.prop(self, 'fresnel_blend')
        col.prop(self, 'rim_blend')
        col.prop(self, 'matcap_mode')
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8
        for line in _wrap_text(self.model_description(), 40):
            col.label(text=line)
        layout.separator()
        layout.label(text="Inputs this model uses:", icon='HIDE_OFF')
        col = layout.column(align=True)
        col.scale_y = 0.8
        shown = 0
        for _kind, name, _default in self.SOCKETS:
            who = SOCKET_MODELS.get(name)
            if who == ALL or (who and self.model in who):
                col.label(text="  " + name)
                shown += 1
        if shown < len(self.SOCKETS):
            layout.label(text=f"{len(self.SOCKETS) - shown} inputs are ignored "
                              f"by this model", icon='INFO')

    def draw_label(self):
        for ident, label, _ in MODEL_ITEMS:
            if ident == self.model:
                return label
        return self.bl_label


# =========================================== R252: the console emulation node

#: the sockets every machine shares (the master's names, so the roads
#: downstream read one vocabulary); each machine's own extras come from
#: core/console.MACHINE_SOCKETS
_CONSOLE_SHARED = ('Diffuse Color', 'Diffuse Level', 'Ambient',
                   'Self-Illumination', 'Opacity', 'Normal',
                   'Bump Strength', 'Bump Height', 'Vertex Color',
                   'Vertex Color Mix', 'Reflection', 'Reflection Color',
                   'Edge Opacity')

CONSOLE_SOCKET_DOCS = {
    'Diffuse Color': "The polygon's colour (RenderWare's material colour, "
                     "GX's material register, the GTE's material RGB): a "
                     "linked texture is the texel the machine's combine "
                     "multiplies per pixel",
    'Diffuse Level': "Scales the lit diffuse term; for RenderWare it is "
                     "RwSurfaceProperties.diffuse, the coefficient every "
                     "light's cosine is multiplied by",
    'Specular Color': "The highlight's tint (every machine's highlight was "
                      "the light's colour, so leave it white for the real "
                      "look)",
    'Specular Level': "The highlight's strength; a shader type that pins "
                      "or disables the specular (Model 2 / 3 Off, "
                      "RenderWare's default pipelines) ignores it",
    'Glossiness': "The highlight's exponent where the machine took one: "
                  "GX's shininess, the DS's custom table, the GTA-style "
                  "specular power; Model 2 / 3 and the DS presets pin "
                  "their own",
    'Soften': "GX only: softens the highlight at grazing light, the master "
              "shader's own Soften",
    'Ambient': "Scales the scene ambient on this material; for RenderWare "
               "it is RwSurfaceProperties.ambient",
    'Self-Illumination': "Added after the lighting, untouched by the lamps "
                         "-- the emissive colour",
    'Opacity': "The polygon's alpha (Model 3 steps it to its 32 levels "
               "when Alpha Steps is on; the Saturn's mesh type pins 0.5)",
    'Toon Size': "The DS toon / highlight table's step position, sampled "
                 "at 32 stops with hard edges (Toon Steps sets how many)",
    'Normal': "A bent shading normal; unlinked takes the mesh's own",
    'Bump Strength': "How far the Bump Height map bends the normal (for "
                     "RenderWare's RpMatFX bump it is the bumpiness "
                     "coefficient, rpMatFXMaterialSetBumpMapCoefficient)",
    'Bump Height': "A greyscale height map bumped into the shading normal "
                   "(RenderWare's rpMATFXEFFECTBUMPMAP map goes here)",
    'Vertex Color': "The mesh's painted colour; unlinked reads the mesh's "
                    "own colour attribute. The types that make the vertex "
                    "colour the material (GX_SRC_VTX, Model 3 fixed "
                    "shading, the N64 with lighting off, D3D's colour "
                    "vertex) read it whole",
    'Vertex Color Mix': "How far the vertex colour replaces the Diffuse "
                        "Color (0 none, 1 whole); the vertex-colour types "
                        "force 1",
    'Reflection': "Ray-traced reflection strength -- none of these "
                  "machines traced rays, so 0 is the period look",
    'Reflection Color': "The ray-traced reflection's tint, white for the "
                        "mirror's own colour (period machines had none)",
    'Edge Opacity': "Opacity at the silhouette, 1 for the period look",
    'Fog Burn-Through': "Model 3 / System 22's polygon light modifier: the "
                        "share of this material that burns through the "
                        "scene fog",
    'Fog Bias': "A per-material offset on the fog's start (the Model 3's "
                "polygon fog bias)",
    'Fog Bank': "A per-material fog density scale (the Model 3's fog "
                "bank)",
    'Prelit Color': "RenderWare's prelight (rpGEOMETRYPRELIT): the baked "
                    "vertex colour ADDED to the computed light before the "
                    "clamp; unlinked reads the mesh's own colour attribute",
    'Night Color': "GTA San Andreas's extra vertex colours: the night "
                   "prelight the Prelit Color blends toward by Night Blend",
    'Night Blend': "How far the prelight is blended toward the Night "
                   "Color (0 day, 1 night) -- the game's time-of-day lerp",
    'Env Map': "RenderWare's rpMATFXEFFECTENVMAP texture, sphere-mapped "
               "(link an Image Texture through Matcap Coordinates), ADDED "
               "over the material by the coefficient",
    'Env Map Coefficient': "rpMatFXMaterialSetEnvMapCoefficient: how much "
                           "of the environment map is added (the PS2 and "
                           "D3D passes blend ONE / ONE)",
    'Dual Texture': "RenderWare's rpMATFXEFFECTDUAL second texture, "
                    "blended over the base texture by the Dual Blend mode",
}


class HALCYON_ConsoleShaderNode(Node, HalcyonNodeBase):
    """R252: the Console Emulation Shader -- a period machine's light
    unit and combiner, with the SHADER TYPES its polygon attribute word
    could select (GX's channel control, the Model 2 specular bits, the
    Model 3 header's fixed shading and sun clamp, the DS's four polygon
    modes, the PS1's raw-texture bit, the GS's four texture functions,
    the PSP's GU_TFX set, VDP1's colour calculations, the N64's G_CC
    presets, the Dreamcast's vertex formats, Direct3D's texture ops and
    render states, the Mega Drive's S/H classes, the Super FX plot, the
    Jaguar's CRY Gouraud, the 3DO's PIXC, RenderWare's pipelines) and
    the options each carried. The sockets are the master shader's own
    names, so every road downstream -- the corner road, the combines,
    the GPU bake -- reads one vocabulary."""

    bl_idname = 'HALCYON_ConsoleShaderNode'
    bl_label = "Console Emulation Shader"
    bl_icon = 'SYSTEM'
    bl_width_default = 230

    def _update(self, context):
        self.refresh_sockets()

    console: EnumProperty(
        name="Console", items=CON.CONSOLE_ITEMS, default='PS1',
        update=_update,
        description="The machine whose light unit and combiner this "
                    "material shades with; the panel below shows that "
                    "machine's shader types and the options its polygon "
                    "attribute word carried")
    gc_type: EnumProperty(
        name="Shader Type", items=CON.GC_TYPE_ITEMS, default='LIT',
        update=_update,
        description="The GameCube channel's shader type: the lit light "
                    "unit or the vertex colour with the lights off")
    gc_diffuse_fn: EnumProperty(
        name="Diffuse Function", items=CON.GC_DIFFUSE_FN_ITEMS, default='CLAMP',
        description="GX_SetChanCtrl's diffuse function: the cosine clamped "
                    "(GX_DF_CLAMP), signed (GX_DF_SIGN) or none "
                    "(GX_DF_NONE, the light added flat)")
    gc_attn_fn: EnumProperty(
        name="Attenuation Function", items=CON.GC_ATTN_FN_ITEMS, default='SPEC',
        description="GX_SetChanCtrl's attenuation function: the rational "
                    "highlight (GX_AF_SPEC), the spot cone (GX_AF_SPOT) or "
                    "none (GX_AF_NONE) -- only SPEC adds a highlight")
    gc_material_src: EnumProperty(
        name="Material Source", items=CON.GC_MATERIAL_SRC_ITEMS, default='REG',
        update=_update,
        description="GX_SetChanCtrl's material source: the material "
                    "register (this node's Diffuse Color) or the vertex "
                    "colour attribute")
    m2_type: EnumProperty(
        name="Shader Type", items=CON.M2_TYPE_ITEMS, default='LIT',
        update=_update,
        description="Model 2's polygon shading: lit through the luma ramp, "
                    "or the fixed-luma unlit polygon")
    m2_specular: EnumProperty(
        name="Specular Control", items=CON.M2_SPECULAR_ITEMS, default='P4',
        update=_update,
        description="Model 2's specular-control bits: off, or the "
                    "reflection term raised to 1, 2, 4 or 8 by repeated "
                    "squaring (MAME model2_v.cpp)")
    m3_type: EnumProperty(
        name="Shader Type", items=CON.M3_TYPE_ITEMS, default='SMOOTH',
        update=_update,
        description="Model 3's polygon header: smooth per-vertex lighting "
                    "or the fixed-shading bit (the vertex colours shown as "
                    "they are)")
    m3_specular: EnumProperty(
        name="Specular", items=CON.M3_SPECULAR_ITEMS, default='P16',
        update=_update,
        description="Model 3's polygon-header specular: off, or the four "
                    "exponent / gain pairs 8 x 1.6, 16 x 1.6, 32 x 2.4, "
                    "64 x 3.2 (Supermodel R3DShaderTriangles)")
    m3_sun_clamp: BoolProperty(
        name="Sun Clamp", default=True,
        description="The polygon header's sun-clamp bit (Supermodel's "
                    "sunClamp): on, the sun's cosine is clamped at 0; off, "
                    "a face turned away goes darker than ambient -- the "
                    "Model 3's negative lighting")
    m3_alpha_steps: BoolProperty(
        name="Alpha Steps (32)", default=False,
        description="Quantise this material's Opacity to Model 3's 32 "
                    "polygon translucency levels (the header's 5-bit alpha)")
    luma_gamma: FloatProperty(
        name="Luma Ramp Gamma", default=1.0, min=0.1, max=4.0,
        description="The 64-entry luma ramp the game wrote, as a gamma on "
                    "the 64-step index (1 is the linear ramp; below 1 "
                    "lifts the mid-tones the way Daytona's and Virtua "
                    "Fighter's ramps did; the ramps themselves are "
                    "unpublished -- a stand-in)")
    ds_type: EnumProperty(
        name="Polygon Mode", items=CON.DS_TYPE_ITEMS, default='MODULATE',
        update=_update,
        description="The DS polygon attribute's mode: modulation (0), "
                    "decal (1), toon or highlight (2) -- GBATEK's texture "
                    "blend modes")
    ds_table: EnumProperty(
        name="Shininess Table", items=CON.DS_TABLE_ITEMS, default='PIN',
        update=_update,
        description="The DS's 128-entry shininess table: left linear "
                    "(disabled), soft, pinned, or shaped by the Glossiness "
                    "socket")
    ps1_type: EnumProperty(
        name="Primitive", items=CON.PS1_TYPE_ITEMS, default='GOURAUD',
        update=_update,
        description="The PlayStation GPU primitive: Gouraud textured, flat "
                    "textured, or the raw (unlit) texture bit")
    ps2_type: EnumProperty(
        name="Texture Function", items=CON.PS2_TYPE_ITEMS, default='MODULATE',
        update=_update,
        description="The Graphics Synthesizer's TEX0 TFX: modulate, decal, "
                    "highlight or highlight 2")
    psp_type: EnumProperty(
        name="Texture Function", items=CON.PSP_TYPE_ITEMS, default='MODULATE',
        update=_update,
        description="The PSP GU's texture function (sceGuTexFunc): "
                    "modulate, decal, replace or add -- or the GU_FLAT "
                    "shading model")
    sat_type: EnumProperty(
        name="Colour Calculation", items=CON.SAT_TYPE_ITEMS, default='GOURAUD',
        update=_update,
        description="VDP1's CMDPMOD colour calculation: the Gouraud table, "
                    "replace, Gouraud with half luminance, or the mesh "
                    "(checkerboard transparency) bit")
    n64_type: EnumProperty(
        name="Combiner Preset", items=CON.N64_TYPE_ITEMS, default='MODULATE',
        update=_update,
        description="The RDP colour combiner's gbi.h preset: modulate, "
                    "decal, shade only, blend by texel alpha -- or the "
                    "vertex colours with G_LIGHTING off")
    s22_type: EnumProperty(
        name="Shader Type", items=CON.S22_TYPE_ITEMS, default='LIT',
        update=_update,
        description="System 22's polygon: lit through the x/64 brightness, "
                    "or fixed (unlit)")
    dc_type: EnumProperty(
        name="Vertex Format", items=CON.DC_TYPE_ITEMS, default='PACKED',
        update=_update,
        description="The CLX2's vertex / shading instruction: packed "
                    "colour with the offset colour, the intensity formats, "
                    "decal, or flat")
    pc_type: EnumProperty(
        name="Shading", items=CON.PC_TYPE_ITEMS, default='GOURAUD_SEP',
        update=_update,
        description="The fixed-function shading model: Gouraud with the "
                    "separate specular, the early modulated specular, or "
                    "the two provoking-vertex flats")
    pc_texture_op: EnumProperty(
        name="Texture Op", items=CON.PC_TEXTURE_OP_ITEMS, default='MODULATE',
        description="The texture stage's colour op (D3DTOP_*): modulate, "
                    "modulate 2x / 4x, add, add signed -- applied under "
                    "the separate specular")
    pc_local_viewer: BoolProperty(
        name="Local Viewer", default=True,
        description="D3DRS_LOCALVIEWER / GL_LIGHT_MODEL_LOCAL_VIEWER: on, "
                    "the highlight takes the true eye vector per pixel; "
                    "off, one camera axis for the whole frame (OpenGL's "
                    "default)")
    pc_color_vertex: BoolProperty(
        name="Colour Vertex", default=False,
        description="D3DRS_COLORVERTEX with D3DMCS_COLOR1: the vertex "
                    "colour is the material's diffuse (GL_COLOR_MATERIAL)")
    pcx_base: EnumProperty(
        name="Base Colour", items=CON.PCX_BASE_ITEMS, default='MEAN',
        description="Which lit colour the PCX's one base colour per "
                    "triangle comes from: the mean of the corners or the "
                    "first corner")
    md_type: EnumProperty(
        name="S/H Mode", items=CON.MD_TYPE_ITEMS, default='AUTO',
        update=_update,
        description="The VDP's shadow / highlight mode: the class per "
                    "polygon by the lighting, or every polygon forced "
                    "NORMAL, SHADOW or HIGHLIGHT")
    sfx_type: EnumProperty(
        name="Fill", items=CON.SFX_TYPE_ITEMS, default='DITHER',
        update=_update,
        description="The Super FX PLOT fill: the dithered pair of palette "
                    "entries, or the nearest single entry")
    jag_type: EnumProperty(
        name="Blitter Mode", items=CON.JAG_TYPE_ITEMS, default='GOURAUD',
        update=_update,
        description="The Jaguar blitter's fill: CRY intensity Gouraud or "
                    "a flat fill")
    tdo_type: EnumProperty(
        name="Cel Shading", items=CON.TDO_TYPE_ITEMS, default='PIXC',
        description="The 3DO cel engine's shading: the PIXC multiplier, "
                    "one eighth-step per cel")
    rw_type: EnumProperty(
        name="Geometry Flags", items=CON.RW_TYPE_ITEMS, default='PRELIT',
        update=_update,
        description="RenderWare's RpGeometry flags: lit, prelit and lit, "
                    "or prelit only (rpGEOMETRYLIGHT / rpGEOMETRYPRELIT)")
    rw_platform: EnumProperty(
        name="Platform", items=CON.RW_PLATFORM_ITEMS, default='PS2',
        update=_update,
        description="The RenderWare platform pipeline: the PS2's GS grid "
                    "with its overbright headroom, the GameCube's TEV "
                    "modulate, or the Xbox / Direct3D 8 8-bit modulate")
    rw_specular: EnumProperty(
        name="Specular", items=CON.RW_SPECULAR_ITEMS, default='NONE',
        update=_update,
        description="None (the default pipelines do not use the specular "
                    "coefficient) or the GTA San Andreas specular plugin's "
                    "highlight")
    rw_matfx: EnumProperty(
        name="Material Effect", items=CON.RW_MATFX_ITEMS, default='NONE',
        update=_update,
        description="RpMatFX's effect on this material: none, environment "
                    "map, bump map, both, or the dual-texture pass")
    rw_dual_blend: EnumProperty(
        name="Dual Blend", items=CON.RW_DUAL_BLEND_ITEMS, default='MODULATE',
        description="The dual-texture pass's blend pair "
                    "(rpMatFXMaterialSetDualBlendModes): modulate, add or "
                    "alpha blend")
    rate: EnumProperty(
        name="Rate", items=CON.RATE_ITEMS, default='MACHINE',
        description="The shading rate this material lights at: the "
                    "machine's own, the scene's, per vertex or per face "
                    "(the hardware-fixed items keep theirs)")
    light_limit: BoolProperty(
        name="Machine's Light Limit", default=False,
        description="Light this material with the machine's own lamp "
                    "count only (GX 8, DS 4, PS1 3, PS2 3, PSP 4, N64 7, "
                    "System 22 1), the first lamps in scene order -- the "
                    "games' register order")
    toon_steps: IntProperty(
        name="Toon Steps", default=2, min=1, max=16,
        description="How many flat bands the DS toon / highlight table "
                    "cuts the light into (2 is the classic two-tone)")

    SOCKETS = (
        ('NodeSocketColor', 'Diffuse Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketFloat', 'Diffuse Level', 1.0),
        ('NodeSocketColor', 'Specular Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Specular Level', 0.5),
        ('NodeSocketFloat', 'Glossiness', 25.0),
        ('NodeSocketFloat', 'Soften', 0.0),
        ('NodeSocketFloat', 'Ambient', 1.0),
        ('NodeSocketColor', 'Self-Illumination', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Opacity', 1.0),
        ('NodeSocketFloat', 'Toon Size', 0.5),
        ('NodeSocketVector', 'Normal', None),
        ('NodeSocketFloat', 'Bump Strength', 1.0),
        ('NodeSocketFloat', 'Bump Height', 0.5),
        ('NodeSocketColor', 'Vertex Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Vertex Color Mix', 0.0),
        ('NodeSocketFloat', 'Reflection', 0.0),
        ('NodeSocketColor', 'Reflection Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Edge Opacity', 1.0),
        ('NodeSocketFloat', 'Fog Burn-Through', 0.0),
        ('NodeSocketFloat', 'Fog Bias', 0.0),
        ('NodeSocketFloat', 'Fog Bank', 0.0),
        ('NodeSocketColor', 'Prelit Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketColor', 'Night Color', (0.2, 0.2, 0.3, 1.0)),
        ('NodeSocketFloat', 'Night Blend', 0.0),
        ('NodeSocketColor', 'Env Map', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Env Map Coefficient', 0.0),
        ('NodeSocketColor', 'Dual Texture', (1.0, 1.0, 1.0, 1.0)),
    )

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            sock = self.inputs.new(kind, name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
            if name == 'Normal':
                try:
                    sock.hide_value = True
                except (AttributeError, TypeError):
                    pass
        _apply_socket_tips(self, CONSOLE_SOCKET_DOCS)
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Shades as the chosen machine's "
            "light unit and combiner when Halcyon renders it")
        self.refresh_sockets()

    def ensure_sockets(self):
        have = {s.name for s in self.inputs}
        for kind, name, default in self.SOCKETS:
            if name in have:
                continue
            try:
                sock = self.inputs.new(kind, name)
                if default is not None:
                    sock.default_value = default
            except Exception:                                   # noqa: BLE001
                pass
        _apply_socket_tips(self, CONSOLE_SOCKET_DOCS)

    def _props(self):
        out = {}
        for name, _items, _d in CON.ENUM_PROPS:
            out[name] = getattr(self, name, _d)
        for name, _kind, _d in CON.SCALAR_PROPS:
            out[name] = getattr(self, name, _d)
        return out

    def refresh_sockets(self):
        con = str(self.console)
        keep = set(_CONSOLE_SHARED) | set(CON.MACHINE_SOCKETS.get(con, ()))
        res = CON.resolve(self._props())
        if res['gloss'] is not None:
            keep.discard('Glossiness')
        if res['spec_level'] is not None and res['spec_level'] == 0.0:
            keep.discard('Specular Color')
            keep.discard('Specular Level')
            keep.discard('Glossiness')
        if res['opacity'] is not None:
            keep.discard('Opacity')
        if res['vmix'] is not None:
            keep.discard('Vertex Color Mix')
        if con == 'RENDERWARE':
            fx = str(self.rw_matfx)
            if fx not in ('ENVMAP', 'BUMPENVMAP'):
                keep.discard('Env Map')
                keep.discard('Env Map Coefficient')
            if fx not in ('BUMPMAP', 'BUMPENVMAP'):
                keep.discard('Bump Strength')
                keep.discard('Bump Height')
            if fx != 'DUAL':
                keep.discard('Dual Texture')
            if res['prelit'] is None:
                keep.discard('Prelit Color')
                keep.discard('Night Color')
                keep.discard('Night Blend')
        if con == 'DS' and str(self.ds_type) not in ('TOON', 'HIGHLIGHT'):
            keep.discard('Toon Size')
        if res['fixed_shade'] > 0.5:
            for nm in ('Specular Color', 'Specular Level', 'Glossiness',
                       'Soften', 'Ambient', 'Diffuse Level'):
                keep.discard(nm)
        for sock in self.inputs:
            sock.hide = bool(sock.name not in keep and not sock.is_linked)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'console', text="")
        con = str(self.console)
        col = layout.column(align=True)
        for name in CON.PANEL.get(con, ()):
            if name == 'toon_steps' and \
                    str(self.ds_type) not in ('TOON', 'HIGHLIGHT'):
                continue
            col.prop(self, name)
        row = layout.row(align=True)
        row.prop(self, 'rate', text="")
        if CON.LIGHT_LIMIT.get(con, 0) > 0:
            layout.prop(self, 'light_limit')

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        box = layout.box()
        col = box.column(align=True)
        col.scale_y = 0.8
        desc = next((c for a, _b, c in CON.CONSOLE_ITEMS
                     if a == str(self.console)), '')
        for line in _wrap_text(desc, 40):
            col.label(text=line)
        res = CON.resolve(self._props())
        layout.label(text=f"Engine model: {res['model']}", icon='INFO')

    def draw_label(self):
        try:
            return CON.label_of(self._props())
        except Exception:                                       # noqa: BLE001
            return self.bl_label


# ======================================================== coded shader node

LANGUAGES = (
    ('GLSL', "GLSL", "OpenGL Shading Language"),
    ('HLSL', "HLSL", "High Level Shading Language (Direct3D)"),
)

KIND_TO_SOCKET = {
    'VALUE': 'NodeSocketFloat',
    'INT': 'NodeSocketInt',
    'BOOL': 'NodeSocketBool',
    'VECTOR': 'NodeSocketVector',
    'VECTOR2': 'NodeSocketVector',
    'RGBA': 'NodeSocketColor',
    'MATRIX': 'NodeSocketFloat',
    'IMAGE': 'NodeSocketColor',
}


def _wrap_text(text, width):
    words = text.split()
    lines, cur = [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    return lines


def _pretty(name):
    """rimPower -> Rim Power, base_color -> Base Color."""
    out = []
    prev_lower = False
    for ch in name.replace('_', ' '):
        if ch.isupper() and prev_lower:
            out.append(' ')
        out.append(ch)
        prev_lower = ch.islower() or ch.isdigit()
    return ''.join(out).strip().title()


_REBUILD_QUEUED = [False]


def _iter_trees():
    for coll in (getattr(bpy.data, 'materials', ()),
                 getattr(bpy.data, 'worlds', ()),
                 getattr(bpy.data, 'node_groups', ())):
        for block in coll:
            tree = getattr(block, 'node_tree', None) or (
                block if hasattr(block, 'nodes') else None)
            if tree is not None:
                yield tree


def _run_rebuilds():
    """Recompile every node that asked for it, outside any update callback.

    Nothing is carried across the timer except a flag stored on the nodes
    themselves -- holding a node pointer across a callback is its own way to
    crash, because the tree may have been rebuilt in between.
    """
    _REBUILD_QUEUED[0] = False
    for tree in _iter_trees():
        for node in list(getattr(tree, 'nodes', ())):
            if node.bl_idname == 'HALCYON_CodeNode' and \
                    getattr(node, 'needs_rebuild', False):
                node.needs_rebuild = False
                try:
                    node.compile_source()
                except Exception as exc:                        # noqa: BLE001
                    node.error = f'{type(exc).__name__}: {exc}'
    return None                       # one-shot


#: nodes currently inside their own update callback, by pointer
_UPDATING = set()


def _node_key(node):
    try:
        return node.as_pointer()
    except Exception:                                           # noqa: BLE001
        return id(node)


def _schedule_rebuild():
    if _REBUILD_QUEUED[0]:
        return
    _REBUILD_QUEUED[0] = True
    try:
        bpy.app.timers.register(_run_rebuilds, first_interval=0.0)
    except Exception:                                           # noqa: BLE001
        # no timer available (headless, or during registration): do it now,
        # which is safe precisely because we are not inside a callback
        _run_rebuilds()


class HALCYON_CodeNode(Node, HalcyonNodeBase):
    """Write a shader in GLSL or HLSL; its uniforms become input sockets"""

    bl_idname = 'HALCYON_CodeNode'
    bl_label = "Coded Shader"
    bl_icon = 'TEXT'
    bl_width_default = 260

    def _lang_changed(self, context):
        # Sockets must not be added or removed from inside an RNA update
        # callback -- Blender is mid-update and it segfaults -- so the rebuild
        # is only ever flagged here and done from a timer.
        #
        # The re-entrancy guard used to be `self._busy`, an ordinary Python
        # attribute. Blender hands out a fresh wrapper object on every access
        # to a node, so that attribute was written to a temporary and read back
        # as the class default: the guard was never once closed. It lives in a
        # module-level set now, keyed by the node's own pointer, which is the
        # only identity that survives the wrapper being rebuilt.
        key = _node_key(self)
        if key in _UPDATING:
            return
        _UPDATING.add(key)
        try:
            if not self.source and not self.source_text.strip():
                self.source_text = DEFAULT_GLSL if self.language == 'GLSL' \
                    else DEFAULT_HLSL
        finally:
            _UPDATING.discard(key)
        self.needs_rebuild = True
        _schedule_rebuild()

    def _source_changed(self, context):
        if _node_key(self) in _UPDATING:
            return
        self.needs_rebuild = True
        _schedule_rebuild()

    language: EnumProperty(name="Language", items=LANGUAGES, default='GLSL',
                           update=_lang_changed)
    needs_rebuild: BoolProperty(default=False, options={'HIDDEN', 'SKIP_SAVE'})
    source: PointerProperty(name="Text", type=bpy.types.Text,
                            description="A text datablock holding the shader",
                            update=_source_changed)
    source_text: StringProperty(name="Inline Source", default='',
                                options={'HIDDEN'})
    as_surface: BoolProperty(
        name="Output as Surface", default=False,
        description="Emit the first output as a shader closure instead of colour")
    error: StringProperty(name="Error", default='', options={'HIDDEN'})
    warn: StringProperty(name="Warning", default='', options={'HIDDEN'})
    auto_compile: BoolProperty(name="Auto Compile", default=True)

    def init(self, context):
        self.source_text = DEFAULT_GLSL
        self.outputs.new('NodeSocketColor', 'Color')
        self.compile_source()

    # ....................................................... compilation
    def get_source(self):
        if self.source is not None:
            try:
                return self.source.as_string()
            except Exception:                                   # noqa: BLE001
                return ''
        return self.source_text or ''

    def compile_source(self):
        src = self.get_source()
        if not src.strip():
            self.error = ''
            return None
        prog, err = try_compile(src, self.language)
        if prog is None:
            self.error = err or 'compile failed'
            return None
        self.error = ''
        self.warn = '; '.join(prog.warnings[:3]) if prog.warnings else ''
        self.rebuild_sockets(prog)
        return prog

    def rebuild_sockets(self, prog):
        """Sockets follow the shader's declarations, keeping existing links."""
        wanted_in = prog.uniform_schema()
        wanted_out = prog.output_schema()

        # `sock.default_value` on a colour or vector socket is a live view into
        # the socket's own memory, not a copy. Holding one across
        # `inputs.clear()` leaves a pointer into freed memory, and reading it
        # back afterwards to restore the value is a use-after-free -- which
        # takes Blender down rather than raising. Every value is materialised
        # here, before anything is cleared.
        keep = {}
        for sock in self.inputs:
            value = None
            if hasattr(sock, 'default_value'):
                dv = sock.default_value
                value = tuple(dv) if hasattr(dv, '__len__') else float(dv)
            keep[sock.name] = (value,
                               [(l.from_node.name, l.from_socket.name)
                                for l in sock.links])
        links_out = []
        for sock in self.outputs:
            for l in sock.links:
                links_out.append((sock.name, l.to_node.name, l.to_socket.name))

        tree = self.id_data
        self.inputs.clear()
        for u in wanted_in:
            kind = u.get('kind', 'VALUE')
            stype = KIND_TO_SOCKET.get(kind, 'NodeSocketFloat')
            label = _pretty(u['name'])
            sock = self.inputs.new(stype, label)
            sock.halcyon_uniform = u['name']
            sock.halcyon_is_image = (kind == 'IMAGE')
            dv = u.get('default')
            if dv is not None and hasattr(sock, 'default_value'):
                try:
                    if kind == 'RGBA':
                        v = list(dv) + [1.0] * (4 - len(dv)) if hasattr(dv, '__len__') \
                            else [float(dv)] * 3 + [1.0]
                        sock.default_value = v[:4]
                    elif kind in ('VECTOR', 'VECTOR2'):
                        v = list(dv) if hasattr(dv, '__len__') else [float(dv)] * 3
                        sock.default_value = (v + [0.0, 0.0, 0.0])[:3]
                    else:
                        sock.default_value = float(dv if not hasattr(dv, '__len__')
                                                   else dv[0])
                except (TypeError, ValueError):
                    pass
            prev = keep.get(label)
            if prev and prev[0] is not None and hasattr(sock, 'default_value'):
                try:
                    sock.default_value = prev[0]
                except (TypeError, ValueError):
                    pass

        self.outputs.clear()
        if self.as_surface:
            self.outputs.new('NodeSocketShader', 'Surface')
        for o in wanted_out:
            stype = KIND_TO_SOCKET.get(o.get('kind', 'RGBA'), 'NodeSocketColor')
            sock = self.outputs.new(stype, _pretty(o['name']))
            sock.halcyon_key = o['name']

        # restore links that still have somewhere to go
        for name, (dv, srcs) in keep.items():
            sock = self.inputs.get(name)
            if sock is None:
                continue
            for from_name, from_sock in srcs:
                node = tree.nodes.get(from_name)
                if node and node.outputs.get(from_sock):
                    try:
                        tree.links.new(node.outputs[from_sock], sock)
                    except Exception:                           # noqa: BLE001
                        pass
        for out_name, to_name, to_sock in links_out:
            sock = self.outputs.get(out_name)
            node = tree.nodes.get(to_name)
            if sock is not None and node is not None and node.inputs.get(to_sock):
                try:
                    tree.links.new(sock, node.inputs[to_sock])
                except Exception:                               # noqa: BLE001
                    pass

    # ............................................................. drawing
    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, 'language', expand=True)
        layout.prop(self, 'source', text="")
        row = layout.row(align=True)
        row.operator('halcyon.compile_shader', icon='FILE_REFRESH').node = self.name
        row.operator('halcyon.new_shader_text', icon='ADD', text="")
        layout.prop(self, 'as_surface')
        if self.error:
            box = layout.box()
            box.alert = True
            for line in self.error.split('\n')[:4]:
                box.label(text=line, icon='ERROR')
        elif self.warn:
            box = layout.box()
            for line in self.warn.split(';')[:3]:
                box.label(text=line.strip(), icon='INFO')

    def draw_buttons_ext(self, context, layout):
        layout.prop(self, 'auto_compile')
        layout.separator()
        layout.label(text="Available inputs:")
        col = layout.column(align=True)
        for name in ('vPosition', 'vNormal', 'vUV', 'vColor', 'vView',
                     'gl_FragCoord', 'iTime', 'iResolution'):
            col.label(text="  " + name)

    def draw_label(self):
        return f"{self.language} Shader"


class HALCYON_OT_compile_shader(bpy.types.Operator):
    bl_idname = 'halcyon.compile_shader'
    bl_label = "Compile"
    bl_description = "Compile the shader and rebuild this node's sockets"
    bl_options = {'REGISTER', 'UNDO'}

    node: StringProperty()

    def execute(self, context):
        space = context.space_data
        tree = getattr(space, 'edit_tree', None) or getattr(space, 'node_tree', None)
        if tree is None:
            return {'CANCELLED'}
        node = tree.nodes.get(self.node) or context.active_node
        if node is None or node.bl_idname != 'HALCYON_CodeNode':
            return {'CANCELLED'}
        prog = node.compile_source()
        if prog is None:
            self.report({'ERROR'}, node.error or "Compile failed")
            return {'CANCELLED'}
        self.report({'INFO'}, f"Compiled: {len(node.inputs)} uniforms, "
                              f"{len(node.outputs)} outputs")
        return {'FINISHED'}


class HALCYON_OT_new_shader_text(bpy.types.Operator):
    bl_idname = 'halcyon.new_shader_text'
    bl_label = "New Shader Text"
    bl_description = "Create a text datablock with a starter shader"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        node = context.active_node
        lang = getattr(node, 'language', 'GLSL')
        text = bpy.data.texts.new(f"halcyon_{lang.lower()}")
        text.write(DEFAULT_GLSL if lang == 'GLSL' else DEFAULT_HLSL)
        if node is not None and node.bl_idname == 'HALCYON_CodeNode':
            node.source = text
        return {'FINISHED'}


# ========================================================== retro utilities


class HALCYON_BIInfluenceNode(Node, HalcyonNodeBase):
    """Blender Internal's value-channel influence, as one node.

    2.79's texture_value_blend(): the texture intensity does not become
    the value -- it blends the Base toward DVar by Intensity x Factor,
    in the slot's blend mode. Every imported hardness/emit/alpha slot
    rides one of these, and it is available for hand wiring too.
    """

    bl_idname = 'HALCYON_BIInfluenceNode'
    bl_label = "BI Influence"
    bl_icon = 'MOD_HUE_SATURATION'

    blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f'texture_value_blend {m}') for m in (
                'MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                'LIGHT', 'SCREEN', 'OVERLAY', 'SOFT', 'LINEAR')])
    # ---- do_material_tex's slot flags, set by the importer from the
    # texture kind and the MTex texflag. With tex_rgb on, the value
    # channel takes the COLOUR's Rec.709 luminance (or its alpha under
    # AlphaMix) as the intensity, exactly as an RGB texture on a value
    # channel did in BI; Intensity is then only the no-colour fallback.
    tex_rgb: BoolProperty(
        name="Texture Yields RGB", default=False,
        description="The linked texture produces colour (a colorband, "
                    "an image, Magic): the intensity is taken from the "
                    "Color input per BI's rules")
    rgbtoint: BoolProperty(
        name="RGB to Intensity", default=False,
        description="MTex RGBToIntensity: collapse the colour to its "
                    "luminance before anything else")
    negative: BoolProperty(
        name="Negative", default=False,
        description="MTex Negative: invert the texture output")
    alphamix: BoolProperty(
        name="Alpha Mix", default=False,
        description="MTex Calculate Alpha mix: the texture's alpha is "
                    "the intensity")
    calc_alpha: BoolProperty(
        name="Calculate Alpha", default=False,
        description="TEX_CALCALPHA: the alpha is max(r,g,b), exactly "
                    "imagewrap")
    neg_alpha: BoolProperty(
        name="Negate Alpha", default=False,
        description="TEX_NEGALPHA: the alpha inverts after it is "
                    "decided")

    def init(self, context):
        self.inputs.new('NodeSocketFloat', 'Base').default_value = 0.5
        self.inputs.new('NodeSocketFloat', 'Intensity').default_value = 0.0
        self.inputs.new('NodeSocketFloat', 'Factor').default_value = 1.0
        self.inputs.new('NodeSocketFloat', 'DVar').default_value = 1.0
        self.inputs.new('NodeSocketColor', 'Color').default_value = \
            (1.0, 1.0, 1.0, 1.0)
        self.inputs.new('NodeSocketFloat', 'Alpha').default_value = 1.0
        self.outputs.new('NodeSocketFloat', 'Value')
        docs = {
            'Base': "The channel's own value before the texture",
            'Intensity': "The texture intensity (a Fac output)",
            'Factor': "The influence slider; negative flips the blend, "
                      "exactly BI's slider",
            'DVar': "The blend TARGET -- BI's DVar slider, not the "
                    "texture value",
            'Color': "The texture's colour output, read when the "
                     "texture yields RGB",
            'Alpha': "The texture's alpha, read under Alpha Mix",
        }
        for s in self.inputs:
            d = docs.get(s.name)
            if d:
                try:
                    s.description = d
                except (AttributeError, TypeError):
                    pass

    def draw_buttons(self, context, layout):
        layout.prop(self, 'blend', text="")
        if self.tex_rgb or self.rgbtoint or self.negative or self.alphamix:
            row = layout.row(align=True)
            row.label(text=('RGB ' if self.tex_rgb else '')
                      + ('toInt ' if self.rgbtoint else '')
                      + ('Neg ' if self.negative else '')
                      + ('AlphaMix' if self.alphamix else ''))


class HALCYON_BIRGBBlendNode(Node, HalcyonNodeBase):
    """Blender Internal's colour-channel influence, as one node.

    2.79's texture_rgb_blend(): the base colour blends toward `tcol` by
    the texture's per-pixel factor times the influence slider. `tcol`
    is the texture's own colour when it yields one -- its ALPHA is then
    the factor -- and the SLOT's colour swatch when it does not, with
    the intensity as the factor. Every imported Color/Specular
    Color/Mirror Color slot rides one of these.
    """

    bl_idname = 'HALCYON_BIRGBBlendNode'
    bl_label = "BI Color Influence"
    bl_icon = 'MOD_HUE_SATURATION'

    blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f'texture_rgb_blend {m}') for m in (
                'MIX', 'MUL', 'ADD', 'SUB', 'DIV', 'DARK', 'DIFF',
                'LIGHT', 'SCREEN', 'OVERLAY', 'HUE', 'SAT', 'VAL',
                'COLOR', 'SOFT', 'LINEAR')])
    tex_rgb: BoolProperty(
        name="Texture Yields RGB", default=False,
        description="The linked texture produces colour: it supplies "
                    "tcol, and its alpha is the per-pixel factor")
    rgbtoint: BoolProperty(
        name="RGB to Intensity", default=False,
        description="MTex RGBToIntensity: collapse the colour to its "
                    "luminance first; the slot colour becomes tcol")
    negative: BoolProperty(
        name="Negative", default=False,
        description="MTex Negative: invert the texture output")
    alphamix: BoolProperty(
        name="Alpha Mix", default=False,
        description="MTex Calculate Alpha mix")
    calc_alpha: BoolProperty(
        name="Calculate Alpha", default=False,
        description="TEX_CALCALPHA: the alpha is max(r,g,b), exactly "
                    "imagewrap")
    neg_alpha: BoolProperty(
        name="Negate Alpha", default=False,
        description="TEX_NEGALPHA: the alpha inverts after it is "
                    "decided")
    map_alpha: BoolProperty(
        name="Slot Also Maps Alpha", default=False,
        description="The same slot drives Alpha: BI then keeps the "
                    "intensity as the factor unless Alpha Mix is on")

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Base').default_value = \
            (0.8, 0.8, 0.8, 1.0)
        self.inputs.new('NodeSocketColor', 'Color').default_value = \
            (1.0, 1.0, 1.0, 1.0)
        self.inputs.new('NodeSocketFloat', 'Intensity').default_value = 0.0
        self.inputs.new('NodeSocketFloat', 'Alpha').default_value = 1.0
        self.inputs.new('NodeSocketFloat', 'Factor').default_value = 1.0
        self.inputs.new('NodeSocketColor', 'Slot Color').default_value = \
            (1.0, 0.0, 1.0, 1.0)
        self.outputs.new('NodeSocketColor', 'Color')
        docs = {
            'Base': "The channel's own colour before this slot",
            'Color': "The texture's colour output, used as tcol when "
                     "the texture yields RGB",
            'Intensity': "The texture intensity (a Fac output): the "
                         "per-pixel factor for intensity textures",
            'Alpha': "The texture's alpha: the per-pixel factor for "
                     "RGB textures",
            'Factor': "The influence slider (0..1 in BI for colours)",
            'Slot Color': "The MTex colour swatch: tcol whenever the "
                          "texture yields no RGB (BI's pink default)",
        }
        for s in self.inputs:
            d = docs.get(s.name)
            if d:
                try:
                    s.description = d
                except (AttributeError, TypeError):
                    pass

    def draw_buttons(self, context, layout):
        layout.prop(self, 'blend', text="")


class HALCYON_PosterizeNode(Node, HalcyonNodeBase):
    """Quantise a colour to a fixed number of levels per channel"""

    bl_idname = 'HALCYON_PosterizeNode'
    bl_label = "Posterize"
    bl_icon = 'IMAGE_ZDEPTH'

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Color').default_value = (.8, .8, .8, 1)
        self.inputs.new('NodeSocketFloat', 'Levels').default_value = 8.0
        self.outputs.new('NodeSocketColor', 'Color')


class HALCYON_DitherNode(Node, HalcyonNodeBase):
    """Ordered dither against the screen position, at shading time"""

    bl_idname = 'HALCYON_DitherNode'
    bl_label = "Ordered Dither"
    bl_icon = 'TEXTURE'

    pattern: EnumProperty(name="Pattern", items=(
        ('BAYER2', "Bayer 2x2", ""), ('BAYER4', "Bayer 4x4", ""),
        ('BAYER8', "Bayer 8x8", ""), ('HALFTONE', "Halftone", "")),
        default='BAYER4')

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Color').default_value = (.8, .8, .8, 1)
        self.inputs.new('NodeSocketFloat', 'Levels').default_value = 4.0
        self.inputs.new('NodeSocketFloat', 'Strength').default_value = 1.0
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'pattern', text="")


class HALCYON_DepthCueNode(Node, HalcyonNodeBase):
    """Blend toward a fog colour by distance, the way depth cueing worked"""

    bl_idname = 'HALCYON_DepthCueNode'
    bl_label = "Depth Cue"
    bl_icon = 'MOD_FLUIDSIM'

    mode: EnumProperty(name="Falloff", items=(
        ('LINEAR', "Linear", ""), ('EXP', "Exponential", ""),
        ('EXP2', "Exponential Squared", ""),
        ('TABLE16', "16-Step Table", "")), default='LINEAR')

    def init(self, context):
        self.inputs.new('NodeSocketColor', 'Color').default_value = (.8, .8, .8, 1)
        self.inputs.new('NodeSocketColor', 'Fog Color').default_value = (.5, .55, .65, 1)
        self.inputs.new('NodeSocketFloat', 'Start').default_value = 5.0
        self.inputs.new('NodeSocketFloat', 'End').default_value = 40.0
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode', text="")


class HALCYON_ScreenInfoNode(Node, HalcyonNodeBase):
    """Screen-space information: pixel coordinates, depth, facing"""

    bl_idname = 'HALCYON_ScreenInfoNode'
    bl_label = "Screen Info"
    bl_icon = 'VIEW_CAMERA'

    def init(self, context):
        self.outputs.new('NodeSocketVector', 'Screen UV')
        self.outputs.new('NodeSocketVector', 'Pixel')
        self.outputs.new('NodeSocketFloat', 'Depth')
        self.outputs.new('NodeSocketFloat', 'Facing')
        self.outputs.new('NodeSocketFloat', 'Frame')
        self.outputs.new('NodeSocketFloat', 'Time')


class HALCYON_PixelateNode(Node, HalcyonNodeBase):
    """Snap a coordinate to a coarse texel grid, for chunky low-res texturing"""

    bl_idname = 'HALCYON_PixelateNode'
    bl_label = "Pixelate"
    bl_icon = 'MOD_REMESH'

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = ("The coordinate to snap. Unlinked means the UV map, "
                         "since fat texels live in texture space")
        self.inputs.new('NodeSocketFloat', 'Pixels X').default_value = 64.0
        self.inputs.new('NodeSocketFloat', 'Pixels Y').default_value = 64.0
        z = self.inputs.new('NodeSocketFloat', 'Pixels Z')
        z.default_value = 0.0
        z.description = "0 leaves the third axis untouched (2D pixelation)"
        self.outputs.new('NodeSocketVector', 'Vector')
        self.outputs[0].description = (
            "Each axis snapped to the centre of its cell. Feed any texture's "
            "Vector input for instant fat texels")


class HALCYON_ScrollNode(Node, HalcyonNodeBase):
    """Animated UV transform: the scrolling water, lava and conveyor trick"""

    bl_idname = 'HALCYON_ScrollNode'
    bl_label = "UV Scroll"
    bl_icon = 'ANIM'

    animate: BoolProperty(name="Animate", default=True)
    fps: IntProperty(
        name="Steps Per Second", default=0, min=0, max=60,
        description="0 scrolls smoothly. Above 0 the clock advances in "
                    "steps, the way texture animation looked at the era's "
                    "frame rates -- 15 is the classic choppy water")

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "The coordinate to move. Unlinked means the UV map"
        self.inputs.new('NodeSocketFloat', 'Scroll X').default_value = 0.1
        self.inputs.new('NodeSocketFloat', 'Scroll Y').default_value = 0.0
        s = self.inputs.new('NodeSocketFloat', 'Spin')
        s.default_value = 0.0
        s.description = ("Rotation about the (0.5, 0.5) UV centre, in turns "
                         "per second")
        self.outputs.new('NodeSocketVector', 'Vector')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')
        layout.prop(self, 'fps')


class HALCYON_ScanlinesNode(Node, HalcyonNodeBase):
    """Darken alternate lines across a surface -- an in-scene CRT screen"""

    bl_idname = 'HALCYON_ScanlinesNode'
    bl_label = "Scanlines"
    bl_icon = 'ALIGN_JUSTIFY'

    animate: BoolProperty(
        name="Roll", default=False,
        description="Drift the lines upward over time, the way a set rolls "
                    "when the vertical hold is off")

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.8, 0.8, 0.8, 1.0)
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = ("Where the lines live. Unlinked means the UV map -- "
                         "a television's scanlines belong to ITS screen, "
                         "not the camera's")
        self.inputs.new('NodeSocketFloat', 'Lines').default_value = 240.0
        self.inputs.new('NodeSocketFloat', 'Darkness').default_value = 0.4
        self.inputs.new('NodeSocketFloat', 'Thickness').default_value = 0.5
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_PaletteNode(Node, HalcyonNodeBase):
    """Snap a colour to the nearest entry of a period hardware palette"""

    bl_idname = 'HALCYON_PaletteNode'
    bl_label = "Hardware Palette"
    bl_icon = 'COLOR'

    palette: EnumProperty(name="Palette", items=(
        ('EGA', "EGA (16)", "The 16 colours of the EGA default palette"),
        ('C64', "C64 (16)", "The Commodore 64's 16 colours"),
        ('CGA', "CGA (4)", "CGA palette 1, high intensity: black, cyan, "
                           "magenta, white"),
        ('GAMEBOY', "Game Boy (4)", "The DMG's four shades of green"),
        ('GRAY4', "Grayscale (4)", "Four grey levels"),
        ('GRAY16', "Grayscale (16)", "Sixteen grey levels"),
        ('RGB332', "RGB 3-3-2 (256)", "Each channel crushed to 3-3-2 bits "
                                      "-- the byte-per-pixel truecolour "
                                      "compromise"),
    ), default='EGA')

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.8, 0.8, 0.8, 1.0)
        self.inputs.new('NodeSocketFloat', 'Mix').default_value = 1.0
        self.outputs.new('NodeSocketColor', 'Color')
        idx = self.outputs.new('NodeSocketFloat', 'Index')
        idx.description = ("The chosen entry's position, 0 to 1 -- drive a "
                           "Color Ramp with it for palette remapping")

    def draw_buttons(self, context, layout):
        layout.prop(self, 'palette', text="")


class HALCYON_ColorCycleNode(Node, HalcyonNodeBase):
    """Rotate a ramp phase over time -- Mark Ferrari's colour cycling"""

    bl_idname = 'HALCYON_ColorCycleNode'
    bl_label = "Color Cycle"
    bl_icon = 'FILE_REFRESH'

    animate: BoolProperty(name="Animate", default=True)

    def init(self, context):
        f = self.inputs.new('NodeSocketFloat', 'Fac')
        f.default_value = 0.0
        f.description = ("The phase to rotate -- typically a texture's Fac, "
                         "with a Color Ramp after this node")
        self.inputs.new('NodeSocketFloat', 'Speed').default_value = 0.5
        s = self.inputs.new('NodeSocketFloat', 'Steps')
        s.default_value = 0.0
        s.description = ("0 cycles smoothly. Above 0 the phase advances in "
                         "that many discrete steps per revolution -- the "
                         "palette-register waterfall")
        self.outputs.new('NodeSocketFloat', 'Fac')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_FlipbookNode(Node, HalcyonNodeBase):
    """Play an N-by-M sprite sheet as an animated texture"""

    bl_idname = 'HALCYON_FlipbookNode'
    bl_label = "Flipbook"
    bl_icon = 'RENDER_ANIMATION'

    animate: BoolProperty(name="Animate", default=True)

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "The coordinate to map into one cell. Unlinked " \
                        "means the UV map"
        self.inputs.new('NodeSocketFloat', 'Columns').default_value = 4.0
        self.inputs.new('NodeSocketFloat', 'Rows').default_value = 4.0
        r = self.inputs.new('NodeSocketFloat', 'Rate')
        r.default_value = 8.0
        r.description = "Cells per second. Fire and explosion sheets of " \
                        "the era ran 8 to 15"
        o = self.inputs.new('NodeSocketFloat', 'Cell Offset')
        o.default_value = 0.0
        o.description = "Which cell to start from -- or, with Animate " \
                        "off, which cell to hold"
        out = self.outputs.new('NodeSocketVector', 'Vector')
        out.description = ("Feed an Image Texture's Vector. Cells read "
                           "left to right, top row first, wrapping at "
                           "the end")

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_UVWaveNode(Node, HalcyonNodeBase):
    """Sine-warp a coordinate -- the underwater and heat-haze wobble"""

    bl_idname = 'HALCYON_UVWaveNode'
    bl_label = "UV Wave"
    bl_icon = 'MOD_WAVE'

    animate: BoolProperty(name="Animate", default=True)

    def init(self, context):
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "The coordinate to wobble. Unlinked means the UV map"
        self.inputs.new('NodeSocketFloat', 'Amplitude X').default_value = 0.02
        self.inputs.new('NodeSocketFloat', 'Amplitude Y').default_value = 0.02
        self.inputs.new('NodeSocketFloat', 'Frequency').default_value = 8.0
        self.inputs.new('NodeSocketFloat', 'Speed').default_value = 1.0
        self.outputs.new('NodeSocketVector', 'Vector')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


class HALCYON_HalftoneNode(Node, HalcyonNodeBase):
    """A rotated dot screen whose dots grow where the input darkens"""

    bl_idname = 'HALCYON_HalftoneNode'
    bl_label = "Halftone"
    bl_icon = 'LIGHTPROBE_SPHERE'

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.5, 0.5, 0.5, 1.0)
        c.description = "The shade the dots reproduce (Rec.601 luma -- " \
                        "the NTSC weights)"
        v = self.inputs.new('NodeSocketVector', 'Vector')
        v.description = "Where the screen lives. Unlinked means the UV map"
        self.inputs.new('NodeSocketFloat', 'Dots').default_value = 24.0
        a = self.inputs.new('NodeSocketFloat', 'Angle')
        a.default_value = 45.0
        a.description = "Screen angle in degrees. Newsprint runs its " \
                        "black plate at 45"
        self.inputs.new('NodeSocketColor', 'Ink Color').default_value = \
            (0.05, 0.05, 0.05, 1.0)
        self.inputs.new('NodeSocketColor', 'Paper Color').default_value = \
            (0.95, 0.93, 0.88, 1.0)
        self.outputs.new('NodeSocketColor', 'Color')
        self.outputs.new('NodeSocketFloat', 'Fac')


class HALCYON_ThresholdNode(Node, HalcyonNodeBase):
    """Cut a value into 0 or 1 at a level, with an optional soft edge"""

    bl_idname = 'HALCYON_ThresholdNode'
    bl_label = "Threshold"
    bl_icon = 'IPO_CONSTANT'

    def init(self, context):
        self.inputs.new('NodeSocketFloat', 'Fac').default_value = 0.5
        self.inputs.new('NodeSocketFloat', 'Level').default_value = 0.5
        s = self.inputs.new('NodeSocketFloat', 'Smooth')
        s.default_value = 0.0
        s.description = "Width of the soft edge around the level. " \
                        "0 is a hard cut"
        self.outputs.new('NodeSocketFloat', 'Fac')


class HALCYON_QuantizeNode(Node, HalcyonNodeBase):
    """Posterize for a single value: snap a Fac to discrete steps"""

    bl_idname = 'HALCYON_QuantizeNode'
    bl_label = "Quantize"
    bl_icon = 'SEQ_HISTOGRAM'

    def init(self, context):
        self.inputs.new('NodeSocketFloat', 'Fac').default_value = 0.5
        self.inputs.new('NodeSocketFloat', 'Steps').default_value = 4.0
        out = self.outputs.new('NodeSocketFloat', 'Fac')
        out.description = ("The cel-band helper: quantize a lighting or "
                           "texture Fac before it drives a Color Ramp")


RAMP_SPACES = (
    ('RGB', "RGB", "Straight-line blend in linear RGB -- the classic, "
     "with its muddy middles between saturated complements"),
    ('OKLAB', "OKLab", "Blend in the OKLab perceptual space: even "
     "lightness, no muddy middles, hues that pass where you expect"),
    ('OKLCH', "OKLCh", "OKLab in polar form: lightness and chroma blend "
     "straight while HUE rotates the short way round -- rainbow ramps "
     "without grey valleys"),
    ('HSV', "HSV", "Blend hue, saturation and value separately -- the "
     "paint-program ramp, vivid and slightly lawless"),
)


class HALCYON_RampNode(Node, HalcyonNodeBase):
    """A multi-stop colour ramp that blends in a chosen colour SPACE.

    Blender's own Color Ramp mixes in RGB and nothing else; this one adds
    OKLab, OKLCh and HSV, with up to six stops whose positions live on the
    node and whose colours are sockets -- so a stop's colour can be driven
    by another node.
    """

    bl_idname = 'HALCYON_RampNode'
    bl_label = "Color Ramp (Spaces)"
    bl_icon = 'COLOR'
    bl_width_default = 200

    def _update(self, context):
        self.refresh_stops()

    space: EnumProperty(
        name="Space", items=RAMP_SPACES, default='OKLAB', update=_update,
        description="The colour space the blend walks through. The stops "
                    "themselves are always plain colours; only the path "
                    "between them changes")
    stops: IntProperty(
        name="Stops", default=2, min=2, max=6, update=_update,
        description="How many colour stops the ramp uses. Each stop is a "
                    "socket below, with its position alongside")
    positions: FloatVectorProperty(
        name="Positions", size=6, min=0.0, max=1.0,
        default=(0.0, 1.0, 0.5, 0.5, 0.5, 0.5),
        description="Where each stop sits along the ramp, 0 to 1. Stops "
                    "are blended in position order")
    easing: EnumProperty(
        name="Easing", default='LINEAR', update=_update,
        items=(('LINEAR', "Linear", "Even blend between stops"),
               ('SMOOTH', "Smooth", "Ease in and out of every stop"),
               ('CONSTANT', "Constant", "Hold each stop until the next -- "
                "hard bands, the palette look")),
        description="The blend profile between neighbouring stops")

    _STOP_DEFAULTS = ((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0),
                      (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0),
                      (0.5, 0.5, 0.5, 1.0), (0.5, 0.5, 0.5, 1.0))

    def init(self, context):
        f = self.inputs.new('NodeSocketFloat', 'Fac')
        f.default_value = 0.5
        f.description = "The position sampled along the ramp"
        for i in range(6):
            s = self.inputs.new('NodeSocketColor', f'Color {i + 1}')
            s.default_value = self._STOP_DEFAULTS[i]
            s.description = (f"Stop {i + 1}'s colour. Its position is on "
                             "the node body")
        out = self.outputs.new('NodeSocketColor', 'Color')
        out.description = "The blended colour at Fac, in the chosen space"
        self.outputs.new('NodeSocketFloat', 'Alpha')
        self.refresh_stops()

    def refresh_stops(self):
        for i in range(6):
            name = f'Color {i + 1}'
            for s in self.inputs:
                if s.name == name:
                    s.hide = bool(i >= self.stops and not s.is_linked)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'space', text="")
        layout.prop(self, 'easing', text="")
        layout.prop(self, 'stops')
        col = layout.column(align=True)
        for i in range(int(self.stops)):
            col.prop(self, 'positions', index=i, text=f"Pos {i + 1}")


class HALCYON_BlurNode(Node, HalcyonNodeBase):
    """Blur whatever is plugged in, by re-sampling it at shifted points.

    The input chain is evaluated several times at offsets in the surface
    plane and averaged -- true blur of any texture, procedural or image.
    That re-run is CPU work: a material using Blur shades on the CPU and
    says so in the console.
    """

    bl_idname = 'HALCYON_BlurNode'
    bl_label = "Blur"
    bl_icon = 'PROP_CON'

    taps: EnumProperty(
        name="Quality", default='MEDIUM',
        items=(('FAST', "Fast (5)", "Five taps -- soft, slightly boxy"),
               ('MEDIUM', "Medium (9)", "Nine taps -- clean for most "
                "sizes"),
               ('FINE', "Fine (17)", "Seventeen taps -- smooth at large "
                "sizes, at proportional cost")),
        description="How many shifted evaluations the blur averages. More "
                    "taps stay smooth at larger sizes and cost more")

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.5, 0.5, 0.5, 1.0)
        c.description = "The chain to blur -- any texture or pattern"
        s = self.inputs.new('NodeSocketFloat', 'Size')
        s.default_value = 0.05
        s.description = ("Blur radius, in the texture's own coordinate "
                         "units (a fraction of the 0-1 span)")
        out = self.outputs.new('NodeSocketColor', 'Color')
        out.description = "The average of the shifted evaluations"


class HalcyonBIRampStop(PropertyGroup):
    """One colorband stop of a BI material ramp."""
    position: FloatProperty(
        name="Position", default=0.0, min=0.0, max=1.0,
        description="Where along the band this stop sits")
    color: FloatVectorProperty(
        name="Color", subtype='COLOR', size=4, min=0.0, max=1.0,
        default=(0.0, 0.0, 0.0, 1.0),
        description="The stop's colour; alpha is the blend factor at "
                    "this stop")


class HALCYON_OT_bi_ramp_gradient(bpy.types.Operator):
    """Create the gradient widget for a BI material ramp, seeded from
    the fallback stop rows."""
    bl_idname = 'halcyon.bi_ramp_gradient'
    bl_label = "BI Ramp Gradient"
    bl_options = {'INTERNAL', 'UNDO'}

    node_name: StringProperty()
    ramp: EnumProperty(items=[('DIF', "Diffuse", ""),
                              ('SPEC', "Specular", "")], default='DIF')

    def execute(self, context):
        tree = getattr(context.space_data, 'edit_tree', None)
        node = tree.nodes.get(self.node_name) if tree else None
        if node is None or node.bl_idname != 'HALCYON_BIMaterialNode':
            return {'CANCELLED'}
        which = 'dif' if self.ramp == 'DIF' else 'spec'
        stops = node.ramp_stops(which)
        if not stops:
            stops = [(0.0, 0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0, 1.0)]
        node.set_ramp_stops(which, stops, node.ramp_ipo(which))
        return {'FINISHED'}


class HALCYON_OT_bi_ramp_stop(bpy.types.Operator):
    """Add or remove a stop on a BI material ramp."""
    bl_idname = 'halcyon.bi_ramp_stop'
    bl_label = "BI Ramp Stop"
    bl_options = {'INTERNAL', 'UNDO'}

    node_name: StringProperty()
    ramp: EnumProperty(items=[('DIF', "Diffuse", ""),
                              ('SPEC', "Specular", "")], default='DIF')
    action: EnumProperty(items=[('ADD', "Add", ""),
                                ('REMOVE', "Remove", "")], default='ADD')
    index: IntProperty(default=-1)

    def execute(self, context):
        tree = getattr(context.space_data, 'edit_tree', None)
        node = tree.nodes.get(self.node_name) if tree else None
        if node is None or node.bl_idname != 'HALCYON_BIMaterialNode':
            return {'CANCELLED'}
        stops = node.dif_stops if self.ramp == 'DIF' else node.spec_stops
        if self.action == 'ADD':
            s = stops.add()
            s.position = 1.0 if len(stops) > 1 else 0.0
            s.color = (1.0, 1.0, 1.0, 1.0) if len(stops) > 1 \
                else (0.0, 0.0, 0.0, 1.0)
        elif 0 <= self.index < len(stops):
            stops.remove(self.index)
        return {'FINISHED'}


class HALCYON_AnimeShaderNode(Node, HalcyonNodeBase):
    """The anime/cel master shader (R218; ramp road R221).

    N.L wrapped to 0..1 and cut into two or three tone bands whose
    shadow COLOURS multiply the base -- a shadow is a colour, never a
    darkness -- with a stepped highlight, rim, matcap and line art.
    Link a texture into Shadow Ramp and the bands come from the RAMP
    instead: it is baked once into a LUT the lamp loop samples by the
    light term (shadow side, transition and lit side all painted, the
    games' own convention; the tone sliders stand down while linked;
    a clock-driven ramp chain bakes its first frame). Ramp Row picks
    the row of a multi-row ramp -- and under GENSHIN, the Game
    Texture's alpha (the material id) picks it automatically.
    The Compatibility mode decodes the texture conventions of the 3D
    anime pipelines: the ArcSys ILM/SSS maps (Guilty Gear Xrd lineage,
    Dragon Ball FighterZ), the HoYo lightmaps (Genshin Impact, ZZZ),
    Kakarot on the ArcSys lineage it descends from, and Sparking! ZERO
    from the game's own material export (R246: the flat Color1, the
    Mask1 line-art multiply, the T_Tone strip read down its height by
    the half-Lambert cosine). Plug the game's own textures into
    Game Texture and Detail Texture and the channels mean what they
    meant at home.

    R229, the 80s additions: Hair Shine paints the angel ring -- a
    band of the shine colour across the object at a fraction of its
    height, its edge waving around it, light-independent, with a
    thinner second band below on request; Airbrush lays the cel
    painter's soft gradation against the first shadow edge (lit side,
    shadow side or both); Shadow Smoothing bends the terminator toward
    a sphere, shared with the Cartoon Shader; and the Line Colour menu
    inks this material's lines in its own shaded tone (iro-trace) or a
    colour of its own."""

    bl_idname = 'HALCYON_AnimeShaderNode'
    bl_label = "Anime Shader"
    bl_icon = 'IPO_CONSTANT'
    bl_width_default = 220

    def _style_changed(self, context):
        self.apply_style(self.style)

    # R240: the Style menu -- the Cartoon Shader's Era idea for the
    # anime tradition's own decades
    style: EnumProperty(
        name="Style", default='CUSTOM', items=ANIME_STYLE_ITEMS,
        update=_style_changed,
        description="Writes the tone sockets and menus below to a "
                    "named decade's starting point -- the kage tints, "
                    "the band edges, the hair shine, the airbrush, the "
                    "rim, the modern pipeline's key and depth rim; "
                    "edit them freely afterwards (the menu does not "
                    "follow your edits, and never touches your paints, "
                    "textures or the key's angle)")

    compat: EnumProperty(
        name="Compatibility", default='GENERIC',
        items=(
            ('GENERIC', "Generic Cel",
             "No texture decode: the sockets are the controls. The "
             "clean slate for original anime materials"),
            ('ARCSYS', "ArcSys (Guilty Gear Xrd / Strive)",
             "The Xrd-lineage ILM map: R specular intensity, G shadow "
             "bias, B highlight size, A drawn line art; the Detail "
             "Texture is the SSS map whose rgb tints the first "
             "shadow. Channel semantics from the published shader "
             "recreations of the GDC 2015 pipeline"),
            ('DBFZ', "Dragon Ball FighterZ",
             "The same ArcSys ILM decode -- FighterZ ships the Xrd "
             "pipeline. Pair with hard softness values (0) for the "
             "flat two-tone the game reads as"),
            ('KAKAROT', "DBZ: Kakarot",
             "The ArcSys-lineage decode with the CC2 game's slightly "
             "lifted, softer tone placement. Kakarot's exact channel "
             "dumps are not publicly documented; this mode applies "
             "the lineage its look descends from, and says so"),
            ('SPARKING', "DB: Sparking! Zero",
             "Decoded from the game's own material export (the FModel "
             "MI parameter set, the 16x256 T_Tone strips, the "
             "character sheets) against the field's reconstruction of "
             "its master material: Diffuse Color is Color1 as exported "
             "(Unreal's linear colour, the swatch the artist chose), "
             "the Game Texture is Mask1 (greyscale line art, a linear "
             "multiply), the Shadow Ramp is GradientTexture read down "
             "its height with white at the top by the half-Lambert "
             "cosine, the tone lifted from GradientAdjust1's floor to "
             "white by the strip; the outline shell the export carries "
             "is made invisible (Halcyon inks). File > Import > "
             "Sparking! ZERO Material builds all of it from the .json"),
            ('GENSHIN', "Genshin Impact",
             "The HoYo character lightmap: R specular/metal mask "
             "(0.9+ reads as metal), G occlusion into the shadow "
             "decision, B inverted highlight threshold, A material "
             "id. Per-id ramp rows do not travel to a tone node: "
             "split materials as the game does and set the tone "
             "colours per part. Decode taken from the PrimoToon "
             "shader source"),
            ('ZZZ', "Zenless Zone Zero",
             "The ZZZ maps per the modding guides: lightmap R is the "
             "shadow/outline configuration, G metallic, B gloss; the "
             "Detail Texture is the material map whose B carries "
             "specular"),
        ),
        description="How Game Texture and Detail Texture channels are "
                    "decoded. Each mode's tooltip names its source")
    tones: EnumProperty(
        name="Tones", default='TWO',
        items=(('TWO', "Two Tone", "Lit and one shadow band"),
               ('THREE', "Three Tone",
                "Lit, first shadow, and a deeper second band")),
        description="How many bands the light is cut into")
    use_vertex_ao: BoolProperty(
        name="Vertex Colour AO", default=False,
        description="ArcSys convention: painted-down vertex RED "
                    "forces a region into shadow -- baked occlusion "
                    "the way Xrd artists painted it (an offset on the "
                    "shadow threshold, per the GDC talk). Works under "
                    "every Compatibility mode, hand-painted GENERIC "
                    "models included; white is neutral")
    emission_alpha: BoolProperty(
        name="Alpha Is Emission", default=False,
        description="HoYo convention: the base texture's alpha "
                    "channel is the emission mask")
    rim_blend: EnumProperty(
        name="Rim Blend", default='ADD',
        items=(('ADD', "Add", "Added on top of the lit result"),
               ('MIX', "Mix", "Blends toward the rim colour"),
               ('MULTIPLY', "Multiply", "Darkens by the rim"),
               ('SCREEN', "Screen", "Brightens without clipping")),
        description="How the rim light lands on the shaded surface")
    matcap_mode: EnumProperty(
        name="Matcap Blend", default='MIX',
        items=(('MIX', "Mix", "Blends toward the matcap"),
               ('ADD', "Add", "Added on top"),
               ('MULTIPLY', "Multiply", "Darkens by the matcap"),
               ('SCREEN', "Screen", "Brightens without clipping")),
        description="How the matcap lands -- feed a metal matcap "
                    "here for the HoYo metal look")
    # R229: the 80s additions' two menus
    airbrush_side: EnumProperty(
        name="Airbrush Side", default='LIT',
        items=(('LIT', "Lit Side",
                "The airbrush colour multiplies the lit tone just above "
                "the first shadow edge, fading out toward full light"),
               ('SHADOW', "Shadow Side",
                "The shadow tone blends toward the airbrush colour "
                "approaching the edge from inside the shadow"),
               ('BOTH', "Both Sides", "Both gradations at once")),
        description="Which side of the first shadow edge the airbrush "
                    "gradation sits on")
    # R238: the cel's light -- the material's key, the depth rim's
    # mode and side, the smoothing shape
    light_source: EnumProperty(
        name="Shading Light", default='SCENE', items=CEL_LIGHT_ITEMS,
        description="What lights this cel: the scene's lamps, or one "
                    "key fixed to the camera or the world (Light "
                    "Azimuth / Elevation), with the scene's cast "
                    "shadows folded in")
    rim_mode: EnumProperty(
        name="Rim Mode", default='FRESNEL', items=CEL_RIM_MODE_ITEMS,
        description="The Fresnel rim by view angle, or the anime depth "
                    "rim: a band Rim Width pixels wide inside the "
                    "silhouette, read from the frame's depth")
    rim_side: EnumProperty(
        name="Rim Side", default='LIT', items=CEL_RIM_SIDE_ITEMS,
        description="Which side of the key the depth rim sits on")
    smooth_shape: EnumProperty(
        name="Smoothing Shape", default='SPHERE', items=CEL_SHAPE_ITEMS,
        description="The shape Shadow Smoothing bends the normal toward")
    # R239: the SDF face shadow's frame -- which object axes the face
    # looks along (the map is read against the key's angle about them)
    face_forward: EnumProperty(
        name="Face Forward", default='NEG_Y', items=FACE_AXIS_ITEMS,
        description="The axis the face looks along, on the object "
                    "wearing this material (a Blender character built "
                    "facing the front view looks along -Y). The SDF "
                    "face shadow reads the key's angle about this "
                    "frame; the frame follows the FIRST object wearing "
                    "the material -- give each head its own face "
                    "material, as the games do")
    face_up: EnumProperty(
        name="Face Up", default='POS_Z', items=FACE_AXIS_ITEMS,
        description="The axis out of the top of the head, on the "
                    "object wearing this material")
    # R241: the hair pass -- the shine wave's shape
    hair_shine_shape: EnumProperty(
        name="Hair Shine Shape", default='SMOOTH',
        items=HAIR_SHINE_SHAPE_ITEMS,
        description="How the hair shine band's edge travels around "
                    "the head: the classic smooth swing, the ruler-"
                    "drawn zigzag, the shoujo scallop of little arcs, "
                    "or the blocky digital step -- each period-"
                    "matched, so a swap keeps the band's height and "
                    "reach")
    line_source: EnumProperty(
        name="Line Colour", default='INK',
        items=(('INK', "Ink Settings",
                "The Cartoon Outlines panel and the material's Ink row "
                "decide this material's line colour, as before"),
               ('IRO', "Iro-Trace (Own Colour)",
                "The 80s coloured trace line: this surface's own shaded "
                "colour, darkened by Line Darken, per pixel -- hair "
                "lines in the hair's tone, skin lines in the skin's. "
                "Runs the line on the distance-field road"),
               ('CUSTOM', "Line Color Socket",
                "The Line Color socket's value, per material (a linked "
                "chain is not read -- lines are inked per material)")),
        description="Where this material's ink line takes its colour")

    SOCKETS = (
        ('NodeSocketColor', 'Diffuse Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketColor', 'Line Art', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketColor', 'Shadow 1 Color', (0.62, 0.44, 0.48, 1.0)),
        ('NodeSocketFloat', 'Shadow 1 Threshold', 0.5),
        ('NodeSocketFloat', 'Shadow 1 Softness', 0.04),
        ('NodeSocketColor', 'Shadow 2 Color', (0.38, 0.26, 0.38, 1.0)),
        ('NodeSocketFloat', 'Shadow 2 Threshold', 0.22),
        ('NodeSocketFloat', 'Shadow 2 Softness', 0.04),
        # R221: the ramp shading road. Link a texture (the game's own
        # ramp, a ColorRamp, any chain) into Shadow Ramp and it is
        # baked to a LUT sampled by the light term inside the lamp
        # loop -- shadow side, transition and lit side all painted in
        # the ramp; the tone sliders above stand down while linked.
        # Ramp Row picks the row (v) of a multi-row ramp; under
        # GENSHIN with a Game Texture linked and no explicit row, the
        # texture's alpha (the material id) picks the row instead.
        ('NodeSocketColor', 'Shadow Ramp', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Ramp Row', 0.0),
        ('NodeSocketFloat', 'Shadow Bias', 0.0),
        ('NodeSocketColor', 'Game Texture', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketColor', 'Detail Texture', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketColor', 'Specular Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Specular Level', 0.55),
        ('NodeSocketFloat', 'Specular Size', 0.12),
        ('NodeSocketFloat', 'Specular Sharpness', 0.05),
        ('NodeSocketColor', 'Rim Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Rim Amount', 0.0),
        ('NodeSocketFloat', 'Rim Power', 2.5),
        ('NodeSocketColor', 'Matcap', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Matcap Blend', 0.0),
        ('NodeSocketColor', 'Self-Illumination', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Emission Strength', 1.0),
        ('NodeSocketFloat', 'Light Response', 1.0),
        ('NodeSocketFloat', 'Ambient', 0.35),
        ('NodeSocketFloat', 'Opacity', 1.0),
        ('NodeSocketVector', 'Normal', None),
        ('NodeSocketFloat', 'Bump Strength', 1.0),
        # R229: the 80s additions. Hair Shine is the "angel ring": a
        # band of the shine colour across the object at Height (a
        # fraction of its own height), Width wide, its edge waving
        # around the object (Wave amplitude, Waves per turn), on the
        # camera-facing surface; Second puts a thinner band that far
        # below. Airbrush is the soft gradation against the first
        # shadow edge (side by the menu). Shadow Smoothing is the
        # inker's simplification shared with the Cartoon Shader. Line
        # Color / Line Darken feed the Line Colour menu.
        ('NodeSocketFloat', 'Hair Shine', 0.0),
        ('NodeSocketColor', 'Hair Shine Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Hair Shine Height', 0.78),
        ('NodeSocketFloat', 'Hair Shine Width', 0.06),
        ('NodeSocketFloat', 'Hair Shine Wave', 0.03),
        ('NodeSocketFloat', 'Hair Shine Waves', 6.0),
        ('NodeSocketFloat', 'Hair Shine Softness', 0.01),
        ('NodeSocketFloat', 'Hair Shine Second', 0.0),
        ('NodeSocketFloat', 'Airbrush', 0.0),
        ('NodeSocketColor', 'Airbrush Color', (0.82, 0.62, 0.62, 1.0)),
        ('NodeSocketFloat', 'Airbrush Width', 0.35),
        ('NodeSocketFloat', 'Shadow Smoothing', 0.0),
        ('NodeSocketColor', 'Line Color', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Line Darken', 0.55),
        # R238: the cel's light. Light Azimuth / Elevation place the
        # key under Shading Light Camera or World (degrees); Screen
        # Shadow is the depth-marched cast shadow toward the key
        # (amount) over Screen Shadow Length pixels at 1080 lines; Rim
        # Width is the depth rim's band (pixels at 1080 lines)
        ('NodeSocketFloat', 'Light Azimuth', 35.0),
        ('NodeSocketFloat', 'Light Elevation', 30.0),
        ('NodeSocketFloat', 'Screen Shadow', 0.0),
        ('NodeSocketFloat', 'Screen Shadow Length', 24.0),
        ('NodeSocketFloat', 'Rim Width', 4.0),
        # R239: the SDF face shadow. Link the face's shadow map (the
        # game's own, or any gradient chain) and it replaces the
        # lambert term on this material: the map's red field against
        # the key's horizontal angle about the face's frame (Face
        # Forward / Face Up), mirrored across the face's centre line
        # for the other side -- the anime face's DRAWN terminator,
        # sweeping as the light or the head turns. Unlinked, nothing
        # changes.
        ('NodeSocketColor', 'Face Shadow (SDF)', (0.0, 0.0, 0.0, 1.0)),
        # R241: the hair pass -- the wave's phase around the head, the
        # band riding the key's height, the second band's own tint
        ('NodeSocketFloat', 'Hair Shine Angle', 0.0),
        ('NodeSocketFloat', 'Hair Shine Follow', 0.0),
        ('NodeSocketColor', 'Hair Shine Second Color', (1.0, 1.0, 1.0, 1.0)),
    )

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            sock = self.inputs.new(kind, name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
        _apply_socket_tips(self, ANIME_SOCKET_DOCS)
        self.outputs.new('NodeSocketShader', 'Surface')

    def ensure_sockets(self):
        """R229: a node saved before the 80s additions grows their
        sockets at their neutral defaults (the evaluator reads a
        missing socket as its neutral value too, so old files shade
        the same before and after). R238: the cel's light sockets the
        same way. R241: every socket gains its tooltip here too, so
        an old file gets them the moment it opens."""
        have = {s.name for s in self.inputs}
        for kind, name, default in self.SOCKETS:
            if name in have:
                continue
            try:
                sock = self.inputs.new(kind, name)
                if default is not None:
                    sock.default_value = default
            except (TypeError, ValueError, RuntimeError):
                pass
        _apply_socket_tips(self, ANIME_SOCKET_DOCS)

    def apply_style(self, style):
        """R240: write one style's starting point into the sockets and
        the menus it names. CUSTOM (and an unknown key) writes
        nothing. The paints, textures and the key's angle are the
        artist's -- a style never touches them."""
        preset = ANIME_STYLE_PRESETS.get(str(style))
        if not preset:
            return
        for key, value in preset.items():
            if key == '__props':
                for pname, pval in value.items():
                    try:
                        setattr(self, pname, pval)
                    except (TypeError, ValueError):
                        pass
                continue
            sock = self.inputs.get(key) if hasattr(self.inputs, 'get') \
                else None
            if sock is None:
                for cand in self.inputs:
                    if getattr(cand, 'name', None) == key:
                        sock = cand
                        break
            if sock is None:
                continue
            try:
                sock.default_value = value
            except (TypeError, ValueError):
                pass

    def draw_buttons(self, context, layout):
        layout.prop(self, 'style', text="")
        layout.prop(self, 'compat', text="")
        row = layout.row()
        row.prop(self, 'tones', expand=True)
        layout.prop(self, 'use_vertex_ao')
        layout.prop(self, 'emission_alpha')
        layout.prop(self, 'line_source', text="")
        layout.prop(self, 'light_source', text="")

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        layout.prop(self, 'rim_blend')
        layout.prop(self, 'rim_mode')
        layout.prop(self, 'rim_side')
        layout.prop(self, 'smooth_shape')
        layout.prop(self, 'matcap_mode')
        layout.prop(self, 'airbrush_side')
        # R239: the SDF face shadow's frame
        layout.prop(self, 'face_forward')
        layout.prop(self, 'face_up')
        # R241: the hair pass
        layout.prop(self, 'hair_shine_shape')


class HALCYON_CartoonNode(Node, HalcyonNodeBase):
    """The cartoon/paint master (R228) -- the Western cel.

    The colour is PAINT, not light: flat, the same under every lamp,
    whatever its energy. One painted shadow tone lands where the key
    lamps do not reach -- as a transparent shadow cel over the paint
    (the Golden Age double exposure: paint x Shadow Color), as a
    second flat paint (UPA and the television decades), or not at all
    (the cheapest limited animation). Shadow Threshold is where the
    edge falls on the wrapped light term (0.5 = the terminator; higher
    pushes the shadow onto the lit side); Softness airbrushes it;
    Shadow Smoothing bends the shading normal toward a sphere around
    the object so the terminator sweeps as one clean shape the way an
    inker simplifies a form. The highlight is a painted dot, not a
    reflection: Highlight Size opens it, Softness feathers it. Lamp
    Influence lets the lamps' energy and colour modulate the paint,
    from none (pure paint) to full (a lit surface).

    The Era menu writes the sockets and the shadow mode to a named
    starting point -- Golden Age, UPA, Xerox, Saturday morning, 90s
    feature, 90s TV -- and the sockets stay yours afterwards. Anime
    is the other tradition: use the Anime Shader for tone bands, game
    texture decodes and ramps."""

    bl_idname = 'HALCYON_CartoonNode'
    bl_label = "Cartoon Shader"
    bl_icon = 'IPO_CONSTANT'
    bl_width_default = 220

    def _era_changed(self, context):
        self.apply_era(self.era)

    era: EnumProperty(
        name="Era", default='CUSTOM', items=CARTOON_ERA_ITEMS,
        update=_era_changed,
        description="Writes the sockets below and the shadow mode to a "
                    "named era's starting point; edit them freely "
                    "afterwards (the menu does not follow your edits)")
    shadow_mode: EnumProperty(
        name="Shadow", default='TRANSPARENT',
        items=CARTOON_SHADOW_MODE_ITEMS,
        description="How the one shadow tone lands on the paint")
    rim_blend: EnumProperty(
        name="Rim Blend", default='ADD',
        items=(('ADD', "Add", "Added on top of the paint"),
               ('MIX', "Mix", "Blends toward the rim colour"),
               ('MULTIPLY', "Multiply", "Darkens by the rim"),
               ('SCREEN', "Screen", "Brightens without clipping")),
        description="How the rim light lands on the painted surface")
    # R238: the cel's light, shared with the Anime Shader
    light_source: EnumProperty(
        name="Shading Light", default='SCENE', items=CEL_LIGHT_ITEMS,
        description="What lights this cel: the scene's lamps, or one "
                    "key fixed to the camera or the world (Light "
                    "Azimuth / Elevation), with the scene's cast "
                    "shadows folded in")
    rim_mode: EnumProperty(
        name="Rim Mode", default='FRESNEL', items=CEL_RIM_MODE_ITEMS,
        description="The Fresnel rim by view angle, or the depth rim: "
                    "a band Rim Width pixels wide inside the "
                    "silhouette, read from the frame's depth")
    rim_side: EnumProperty(
        name="Rim Side", default='LIT', items=CEL_RIM_SIDE_ITEMS,
        description="Which side of the key the depth rim sits on")
    smooth_shape: EnumProperty(
        name="Smoothing Shape", default='SPHERE', items=CEL_SHAPE_ITEMS,
        description="The shape Shadow Smoothing bends the normal toward")
    airbrush_side: EnumProperty(
        name="Airbrush Side", default='LIT',
        items=(('LIT', "Lit Side",
                "The airbrush colour multiplies the lit paint just above "
                "the shadow edge, fading out toward full light"),
               ('SHADOW', "Shadow Side",
                "The shadow tone blends toward the airbrush colour "
                "approaching the edge from inside the shadow"),
               ('BOTH', "Both Sides", "Both gradations at once")),
        description="Which side of the shadow edge the airbrush "
                    "gradation sits on -- the rounded rendering of the "
                    "feature cel")
    # R241: the highlight cel's wave shape (shared with the Anime
    # Shader's hair pass)
    hair_shine_shape: EnumProperty(
        name="Hair Shine Shape", default='SMOOTH',
        items=HAIR_SHINE_SHAPE_ITEMS,
        description="How the hair shine band's edge travels around "
                    "the head: the classic smooth swing, the ruler-"
                    "drawn zigzag, the shoujo scallop of little arcs, "
                    "or the blocky digital step -- each period-"
                    "matched, so a swap keeps the band's height and "
                    "reach")

    SOCKETS = (
        ('NodeSocketColor', 'Paint Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketColor', 'Shadow Color', (0.55, 0.45, 0.62, 1.0)),
        ('NodeSocketFloat', 'Shadow Amount', 1.0),
        ('NodeSocketFloat', 'Shadow Threshold', 0.5),
        ('NodeSocketFloat', 'Shadow Softness', 0.02),
        ('NodeSocketFloat', 'Shadow Smoothing', 0.0),
        ('NodeSocketColor', 'Highlight Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Highlight Size', 0.0),
        ('NodeSocketFloat', 'Highlight Softness', 0.02),
        ('NodeSocketFloat', 'Lamp Influence', 0.0),
        ('NodeSocketColor', 'Rim Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Rim Amount', 0.0),
        ('NodeSocketFloat', 'Rim Power', 2.5),
        ('NodeSocketColor', 'Self-Illumination', (0.0, 0.0, 0.0, 1.0)),
        ('NodeSocketFloat', 'Emission Strength', 1.0),
        ('NodeSocketFloat', 'Opacity', 1.0),
        ('NodeSocketVector', 'Normal', None),
        # R238: the cel's light (as on the Anime Shader) and the
        # airbrush gradation against the paint's shadow edge
        ('NodeSocketFloat', 'Light Azimuth', 35.0),
        ('NodeSocketFloat', 'Light Elevation', 30.0),
        ('NodeSocketFloat', 'Screen Shadow', 0.0),
        ('NodeSocketFloat', 'Screen Shadow Length', 24.0),
        ('NodeSocketFloat', 'Rim Width', 4.0),
        ('NodeSocketFloat', 'Airbrush', 0.0),
        ('NodeSocketColor', 'Airbrush Color', (0.82, 0.62, 0.62, 1.0)),
        ('NodeSocketFloat', 'Airbrush Width', 0.35),
        # R241: the highlight cel over the paint -- the Anime Shader's
        # whole hair bag on the paint master, inert at zero
        ('NodeSocketFloat', 'Hair Shine', 0.0),
        ('NodeSocketColor', 'Hair Shine Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Hair Shine Height', 0.78),
        ('NodeSocketFloat', 'Hair Shine Width', 0.06),
        ('NodeSocketFloat', 'Hair Shine Wave', 0.03),
        ('NodeSocketFloat', 'Hair Shine Waves', 6.0),
        ('NodeSocketFloat', 'Hair Shine Softness', 0.01),
        ('NodeSocketFloat', 'Hair Shine Second', 0.0),
        ('NodeSocketFloat', 'Hair Shine Angle', 0.0),
        ('NodeSocketFloat', 'Hair Shine Follow', 0.0),
        ('NodeSocketColor', 'Hair Shine Second Color', (1.0, 1.0, 1.0, 1.0)),
    )

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            sock = self.inputs.new(kind, name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
        _apply_socket_tips(self, CARTOON_SOCKET_DOCS)
        self.outputs.new('NodeSocketShader', 'Surface')

    def ensure_sockets(self):
        """R238: a node saved before the cel's light grows its sockets
        at their neutral defaults (the evaluator reads a missing socket
        as its neutral value too, so old files shade the same). R241:
        every socket gains its tooltip here too, so an old file gets
        them the moment it opens."""
        have = {s.name for s in self.inputs}
        for kind, name, default in self.SOCKETS:
            if name in have:
                continue
            try:
                sock = self.inputs.new(kind, name)
                if default is not None:
                    sock.default_value = default
            except (TypeError, ValueError, RuntimeError):
                pass
        _apply_socket_tips(self, CARTOON_SOCKET_DOCS)

    def apply_era(self, era):
        """Write one era's starting point into the sockets and the
        shadow mode. CUSTOM (and an unknown key) writes nothing."""
        preset = CARTOON_ERA_PRESETS.get(str(era))
        if not preset:
            return
        for key, value in preset.items():
            if key == 'shadow_mode':
                try:
                    self.shadow_mode = value
                except (TypeError, ValueError):
                    pass
                continue
            sock = self.inputs.get(key) if hasattr(self.inputs, 'get') \
                else None
            if sock is None:
                for cand in self.inputs:
                    if getattr(cand, 'name', None) == key:
                        sock = cand
                        break
            if sock is None:
                continue
            try:
                sock.default_value = value
            except (TypeError, ValueError):
                pass

    def draw_buttons(self, context, layout):
        layout.prop(self, 'era', text="")
        layout.prop(self, 'shadow_mode', text="")
        layout.prop(self, 'light_source', text="")

    def draw_buttons_ext(self, context, layout):
        self.draw_buttons(context, layout)
        layout.prop(self, 'rim_blend')
        layout.prop(self, 'rim_mode')
        layout.prop(self, 'rim_side')
        layout.prop(self, 'smooth_shape')
        layout.prop(self, 'airbrush_side')
        # R241: the hair pass
        layout.prop(self, 'hair_shine_shape')


class HALCYON_VolumeNode(Node, HalcyonNodeBase):
    """The volume master (R222) -- real marched volumes, cel dials on.

    Link its Volume output to the Material Output's Volume socket and
    the mesh becomes a VOLUME CONTAINER: every camera ray marches the
    container's bounding box (the era's volume gizmos were boxes and
    spheres), accumulating absorption and single scatter from the
    scene's own lamps -- falloffs, gobos, light linking and per-lamp
    shadows all included. Whatever chain feeds Density is evaluated at
    every march sample with Generated coordinates spanning the box, so
    a Bozo or Cells chain shapes clouds exactly as POV-Ray's media and
    3D Studio's Volume Fog did.

    The stylized half: Edge Threshold/Softness cut HARD anime cloud
    edges out of a soft density field; Bands posterizes the scattered
    light into cel steps; Shadow Tint applies the anime rule to
    volumes -- a shadowed region takes a COLOUR, never just darkness.
    All neutral at their defaults.

    R225: Model picks the scattering law (eight of them, from
    Henyey-Greenstein through POV-Ray's media types to 3D Studio's
    unlit Volume Fog and Combustion), Shape decides what the ray
    marches (the mesh itself by default; box, sphere and cylinder
    gizmos on request) and Voxels reads the chains on an N^3 lattice
    for the blocky voxel look.

    Density takes any chain: a 3D pattern (Bozo, Cells, Blender's
    Noise or Voronoi) shapes the fog in three dimensions; a 2D image
    extrudes along the box's Z (Flat), averages its three axis
    projections into a solid (Box), or projects from the camera
    through the fog when driven by Texture Coordinate > Window -- a
    drawn cloud, lit and shadowed as a volume (R226)."""

    bl_idname = 'HALCYON_VolumeNode'
    bl_label = "Halcyon Volume"
    bl_icon = 'MOD_FLUIDSIM'
    bl_width_default = 200

    model: EnumProperty(
        name="Model", default='HG', items=VOLUME_MODEL_ITEMS,
        description="The scattering law the lamps light this volume "
                    "by -- or, for the two 3D Studio atmospherics, the "
                    "self-lit rule that replaces the lamps. Every lit "
                    "law is normalised to the same total energy, so "
                    "switching changes the shape of the glow, not its "
                    "brightness")
    shape: EnumProperty(
        name="Shape", default='MESH', items=VOLUME_SHAPE_ITEMS,
        description="What a camera ray marches: the container's own "
                    "faces, or a box, sphere or cylinder gizmo fitted "
                    "to its bound")
    voxels: IntProperty(
        name="Voxels", default=0, min=0, max=512,
        description="0: the density and colour chains are read at each "
                    "march sample. N: read at the centres of an N x N x "
                    "N lattice over the bound instead -- the blocky "
                    "voxel volume of the era's grid volumetrics (a smoke "
                    "grid snaps to the same lattice). Set Volume Steps "
                    "above 2N so the march resolves the cells")

    SOCKETS = (
        ('NodeSocketFloat', 'Density', 1.0),
        ('NodeSocketColor', 'Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketFloat', 'Absorption', 0.0),
        ('NodeSocketFloat', 'Anisotropy', 0.0),
        ('NodeSocketColor', 'Emission Color', (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Emission Strength', 0.0),
        ('NodeSocketFloat', 'Edge Threshold', 0.0),
        ('NodeSocketFloat', 'Edge Softness', 0.25),
        ('NodeSocketFloat', 'Bands', 0.0),
        ('NodeSocketColor', 'Shadow Tint', (0.35, 0.3, 0.5, 1.0)),
        ('NodeSocketFloat', 'Tint Amount', 0.0),
    )

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            sock = self.inputs.new(kind, name)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
        _apply_socket_tips(self, VOLUME_SOCKET_DOCS)
        self.outputs.new('NodeSocketShader', 'Volume')

    def ensure_sockets(self):
        """R241: the tooltips land on an old file's node at load (the
        socket set itself has not changed since the node shipped)."""
        _apply_socket_tips(self, VOLUME_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'model', text="")
        row = layout.row(align=True)
        row.prop(self, 'shape', text="")
        row.prop(self, 'voxels')
        note = layout.column(align=True)
        note.scale_y = 0.8
        if self.model == 'COMBUSTION':
            note.label(text="Fire: Color = outer, Emission Color = inner")
            note.label(text="Absorption = how much the flames hide")
        elif self.model == 'FOG':
            note.label(text="Unlit: Color composited by Density")
        note.label(text="Link to Material Output > Volume")


class HALCYON_BIMaterialNode(Node, HalcyonNodeBase):
    """Blender Internal's material panel as one node.

    The second master shader: where the Halcyon Shader packages one
    reflectance model, this keeps Blender Internal's DIFFUSE and
    SPECULAR shader menus independent -- the full 5x5 matrix a single
    model enum can never carry (Oren-Nayar diffuse under a CookTorr
    highlight, Fresnel diffuse under WardIso, all of it). Every branch
    is the transcribed 2.79 formula; the sockets show BI's own labels
    (Hardness, Refr, Slope, Alpha) while carrying master-shader
    identifiers underneath, so the whole bake/frame machinery treats
    it exactly like the master.
    """

    bl_idname = 'HALCYON_BIMaterialNode'
    bl_label = "BI Material"
    bl_icon = 'MATERIAL'
    bl_width_default = 200

    def _update(self, context):
        self.refresh_sockets()

    diff_shader: EnumProperty(
        name="Diffuse", update=_update, default='LAMBERT',
        items=[
            ('LAMBERT', "Lambert", "Plain N.L -- BI's default diffuse"),
            ('OREN_NAYAR', "Oren-Nayar",
             "Rough diffuse, driven by Roughness"),
            ('TOON', "Toon",
             "A hard angular band with its own Size and Smooth"),
            ('MINNAERT', "Minnaert",
             "Rim darkening below Darkness 1, rim brightening above"),
            ('FRESNEL', "Fresnel",
             "Replaces the cosine with BI's fresnel_fac of the light "
             "angle -- grazing lights glow"),
        ])
    spec_shader: EnumProperty(
        name="Specular", update=_update, default='COOKTORR',
        items=[
            ('COOKTORR', "CookTorr",
             "pow(N.H, Hardness) / (0.1 + N.V) -- BI's default, with "
             "its 11x grazing brightening"),
            ('PHONG', "Phong",
             "pow(N.H, Hardness) -- BI's Phong was the half-vector "
             "lobe"),
            ('BLINN', "Blinn",
             "Torrance-Sparrow with BI's refraction-index Fresnel; "
             "Refr drives it"),
            ('TOON', "Toon",
             "A hard angular highlight band with its own Size/Smooth"),
            ('WARDISO', "WardIso",
             "Isotropic Gaussian on the microfacet slope; Slope (rms) "
             "drives the width"),
        ])
    shadeless: BoolProperty(
        name="Shadeless", default=False, update=_update,
        description="Emit the diffuse colour flat, ignoring every "
                    "light -- BI's Shadeless toggle")

    # ---- Shading panel extras
    use_cubic: BoolProperty(
        name="Cubic Interpolation", default=False,
        description="Smoothstep the diffuse term, exactly BI's Cubic "
                    "Interpolation -- softer terminators")
    use_tangent_v: BoolProperty(
        name="Tangent Shading", default=False,
        description="Shade with a per-light fake normal built from the "
                    "surface tangent -- BI's anisotropic strand trick")

    # ---- Transparency panel
    use_transparency: BoolProperty(
        name="Transparency", default=False, update=_update,
        description="Enable the transparency panel; off, the material "
                    "is opaque and Alpha is inert, exactly the greyed "
                    "2.79 panel")
    transp_mode: EnumProperty(
        name="Type", default='Z_TRANSPARENCY', update=_update,
        items=[
            ('Z_TRANSPARENCY', "Z Transparency",
             "Plain alpha blending in depth order"),
            ('RAYTRACE', "Raytrace",
             "Refract what lies behind through Ray IOR, tinted by "
             "Filter (needs Raytracing on in the render settings)"),
        ],
        description="How transparency composites (BI's Mask mode is "
                    "not carried: it masked the sky in a scanline "
                    "world this engine does not reproduce)")

    # ---- Mirror panel
    use_mirror: BoolProperty(
        name="Mirror", default=False, update=_update,
        description="Enable ray-mirror reflection; off, Mirror "
                    "sliders are inert, exactly the greyed 2.79 panel")

    # ---- ramps
    use_ramp_dif: BoolProperty(
        name="Diffuse Ramp", default=False,
        description="Recolour the diffuse by a colorband, exactly "
                    "BI's diffuse ramp")
    ramp_dif_input: EnumProperty(
        name="Input", default='SHADER', items=[
            ('SHADER', "Shader", "The diffuse shader's own value"),
            ('ENERGY', "Energy",
             "The lit energy: shader times lamp times shadow"),
            ('NORMAL', "Normal", "The view angle against the normal"),
            ('RESULT', "Result",
             "The final accumulated diffuse, ramped once after all "
             "lamps"),
        ])
    ramp_dif_blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f"ramp_blend {m}") for m in (
                'MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV', 'DIFF',
                'DARK', 'LIGHT', 'OVERLAY', 'DODGE', 'BURN', 'HUE',
                'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR')])
    ramp_dif_factor: FloatProperty(
        name="Factor", default=1.0, min=0.0, max=1.0,
        description="How strongly the ramp recolours")
    ramp_dif_ipo: EnumProperty(
        name="Interpolation", default='LINEAR', items=[
            ('LINEAR', "Linear", "Straight blends between stops"),
            ('EASE', "Ease", "Smoothstep blends between stops"),
            ('B_SPLINE', "B-Spline", "Smooth curve approaching stops"),
            ('CARDINAL', "Cardinal", "Smooth curve through stops"),
            ('CONSTANT', "Constant", "Hard steps at each stop"),
        ])
    dif_stops: CollectionProperty(type=HalcyonBIRampStop)
    dif_ramp_tex: StringProperty(default='')
    use_ramp_spec: BoolProperty(
        name="Specular Ramp", default=False,
        description="Recolour the specular by a colorband, exactly "
                    "BI's specular ramp")
    ramp_spec_input: EnumProperty(
        name="Input", default='SHADER', items=[
            ('SHADER', "Shader", "The specular shader's own value"),
            ('ENERGY', "Energy",
             "The lit energy: shader times lamp times shadow"),
            ('NORMAL', "Normal", "The view angle against the normal"),
            ('RESULT', "Result",
             "The final accumulated specular, ramped once after all "
             "lamps"),
        ])
    ramp_spec_blend: EnumProperty(
        name="Blend", default='MIX', items=[
            (m, m.title(), f"ramp_blend {m}") for m in (
                'MIX', 'ADD', 'MULT', 'SUB', 'SCREEN', 'DIV', 'DIFF',
                'DARK', 'LIGHT', 'OVERLAY', 'DODGE', 'BURN', 'HUE',
                'SAT', 'VAL', 'COLOR', 'SOFT', 'LINEAR')])
    ramp_spec_factor: FloatProperty(
        name="Factor", default=1.0, min=0.0, max=1.0,
        description="How strongly the ramp recolours")
    ramp_spec_ipo: EnumProperty(
        name="Interpolation", default='LINEAR', items=[
            ('LINEAR', "Linear", "Straight blends between stops"),
            ('EASE', "Ease", "Smoothstep blends between stops"),
            ('B_SPLINE', "B-Spline", "Smooth curve approaching stops"),
            ('CARDINAL', "Cardinal", "Smooth curve through stops"),
            ('CONSTANT', "Constant", "Hard steps at each stop"),
        ])
    spec_stops: CollectionProperty(type=HalcyonBIRampStop)
    spec_ramp_tex: StringProperty(default='')

    # ---- Options panel
    use_mist: BoolProperty(
        name="Use Mist", default=True,
        description="Off, the material ignores the scene fog entirely "
                    "-- BI's Use Mist")
    vcol_paint: BoolProperty(
        name="Vertex Color Paint", default=False,
        description="Vertex colours replace the base colour (a linked "
                    "Color chain wins)")
    vcol_light: BoolProperty(
        name="Vertex Color Light", default=False,
        description="Vertex colours (times their alpha) add to the "
                    "emit term -- BI's extra lighting")
    light_group: StringProperty(
        name="Light Group", default='',
        description="Name of a collection: only its lamps light this "
                    "material")
    light_group_exclusive: BoolProperty(
        name="Exclusive", default=False,
        description="Lamps of this group light ONLY materials naming "
                    "the group")

    # ---- Shadow panel
    shadow_receive: BoolProperty(
        name="Receive", default=True,
        description="Off, shadows never darken this material")
    shadow_cast: BoolProperty(
        name="Cast", default=True,
        description="Off, this material's faces are pulled out of "
                    "every shadow map")
    shadow_cast_only: BoolProperty(
        name="Cast Only", default=False,
        description="Invisible to the camera while still casting "
                    "shadows (and appearing in reflections)")
    shadow_only: BoolProperty(
        name="Shadows Only", default=False,
        description="The shadow catcher: renders black at the mean of "
                    "its lamps' shadow, transparent elsewhere")
    sbias: FloatProperty(
        name="Shadow Bias", default=0.0, min=0.0, max=0.25,
        description="2.79's terminator fix: diffuse from shadowed "
                    "lamps fades out below this N.L threshold "
                    "(phongcorr), hiding the jagged shadow terminator")
    raybias: BoolProperty(
        name="Ray Bias", default=False,
        description="Use the object's Auto Smooth angle as the "
                    "terminator threshold for ray-shadowed lamps on "
                    "smooth faces (MA_RAYBIAS)")
    use_obcolor: BoolProperty(
        name="Object Color", default=False,
        description="Multiply the final shaded colour by each "
                    "object's own Color (and opacity by its alpha "
                    "when transparency is on) -- per-object tinting "
                    "with one shared material")

    # ---- Subsurface Scattering panel (2.79's point-cloud dipole,
    # transcribed from sss.c; CPU only -- the GPU plan refuses by name)
    sss_enable: BoolProperty(
        name="Subsurface Scattering", default=False,
        description="BI's two-pass SSS: a pre-pass renders this "
                    "material's lit surface into a point cloud, and "
                    "the dipole gather replaces the diffuse term. "
                    "Runs on both devices -- the GPU walks the same "
                    "octree from a data texture")
    sss_scale: FloatProperty(
        name="Scale", default=0.1, min=0.0001, max=1000.0,
        description="Object scale in Blender units per 1 real-world "
                    "unit of the radius")
    sss_radius: FloatVectorProperty(
        name="Radius", size=3, default=(1.0, 1.0, 1.0), min=0.0001,
        description="Mean free path per channel: how far red, green "
                    "and blue light travel below the surface")
    sss_color: FloatVectorProperty(
        name="Scattering Color", subtype='COLOR', size=3,
        default=(1.0, 1.0, 1.0), min=0.0, max=1.0,
        description="The reflectance the dipole solves for, per "
                    "channel")
    sss_ior: FloatProperty(
        name="IOR", default=1.3, min=0.1, max=2.0,
        description="Index of refraction (skin is about 1.3-1.4)")
    sss_error: FloatProperty(
        name="Error", default=0.05, min=0.0001, max=10.0,
        description="Hierarchy acceptance threshold: lower is more "
                    "exact and slower, exactly 2.79's Error slider")
    sss_colfac: FloatProperty(
        name="Color Factor", default=1.0, min=0.0, max=1.0,
        description="Blend between the scattering colour and white "
                    "in the dipole's reflectance")
    sss_texfac: FloatProperty(
        name="Texture Factor", default=0.0, min=0.0, max=1.0,
        description="How much surface texture survives the scatter: "
                    "0 keeps the full texture, 1 dissolves it")
    sss_front: FloatProperty(
        name="Front", default=1.0, min=0.0, max=2.0,
        description="Front-scattering weight")
    sss_back: FloatProperty(
        name="Back", default=1.0, min=0.0, max=10.0,
        description="Back-scattering weight (light through thin "
                    "parts)")

    #: (socket kind, BI display label, master-compatible identifier,
    #: default). The identifier is what the evaluator, the bake probe
    #: and the frame assembler match on.
    BI_SOCKETS = (
        ('NodeSocketColor', 'Color', 'Diffuse Color', (0.8, 0.8, 0.8, 1.0)),
        ('NodeSocketFloat', 'Intensity', 'Diffuse Level', 0.8),
        ('NodeSocketFloat', 'Roughness', 'Roughness', 0.5),
        ('NodeSocketFloat', 'Darkness', 'Darkness', 1.0),
        ('NodeSocketFloat', 'Toon Size', 'Toon Size', 0.5),
        ('NodeSocketFloat', 'Toon Smooth', 'Toon Smooth', 0.1),
        ('NodeSocketFloat', 'Fresnel', 'BI Fresnel', 0.1),
        ('NodeSocketFloat', 'Fresnel Factor', 'BI Fresnel Factor', 0.5),
        ('NodeSocketColor', 'Specular Color', 'Specular Color',
         (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Specular Intensity', 'Specular Level', 0.5),
        ('NodeSocketFloat', 'Hardness', 'Glossiness', 50.0),
        ('NodeSocketFloat', 'Refr', 'IOR', 4.0),
        ('NodeSocketFloat', 'Slope', 'Slope', 0.1),
        ('NodeSocketFloat', 'Spec Toon Size', 'Spec Toon Size', 0.5),
        ('NodeSocketFloat', 'Spec Toon Smooth', 'Spec Toon Smooth', 0.1),
        ('NodeSocketFloat', 'Emit', 'Emit', 0.0),
        ('NodeSocketFloat', 'Ambient', 'Ambient', 1.0),
        ('NodeSocketFloat', 'Translucency', 'Translucency', 0.0),
        ('NodeSocketFloat', 'Alpha', 'Opacity', 1.0),
        ('NodeSocketFloat', 'Transp Fresnel', 'Transp Fresnel', 0.0),
        ('NodeSocketFloat', 'Transp Blend', 'Transp Blend', 1.25),
        ('NodeSocketFloat', 'Transp Specular', 'Transp Specular', 1.0),
        ('NodeSocketFloat', 'Ray IOR', 'Ray IOR', 1.3),
        ('NodeSocketFloat', 'Filter', 'Filter', 0.0),
        ('NodeSocketFloat', 'Mirror', 'Reflection', 0.0),
        ('NodeSocketColor', 'Mirror Color', 'Reflection Color',
         (1.0, 1.0, 1.0, 1.0)),
        ('NodeSocketFloat', 'Mirror Fresnel', 'Mirror Fresnel', 0.0),
        ('NodeSocketFloat', 'Mirror Blend', 'Mirror Blend', 1.25),
        ('NodeSocketVector', 'Normal', 'Normal', None),
        ('NodeSocketFloat', 'Bump Strength', 'Bump Strength', 1.0),
        ('NodeSocketFloat', 'Bump Height', 'Bump Height', 0.5),
    )

    BI_DOCS = {
        'Color': "Diffuse colour, BI's base colour",
        'Intensity': "Diffuse intensity (Ref): how much of the colour "
                     "reflects",
        'Roughness': "Oren-Nayar surface roughness",
        'Darkness': "Minnaert darkness: below 1 darkens the rim, above "
                    "1 brightens it",
        'Toon Size': "Angular size of the lit toon band (radians)",
        'Toon Smooth': "Softness of the toon band's edge",
        'Fresnel': "Fresnel diffuse: the gradient term of BI's "
                   "fresnel_fac",
        'Fresnel Factor': "Fresnel diffuse: the power term; 0 disables "
                          "and shades flat",
        'Specular Color': "Highlight colour",
        'Specular Intensity': "Specular intensity (Spec): highlight "
                              "strength",
        'Hardness': "Specular hardness 1-511, exactly BI's slider: the "
                    "cosine exponent of the highlight",
        'Refr': "Blinn's refraction index, driving its Fresnel",
        'Slope': "WardIso's standard deviation of the microfacet "
                 "slope (rms)",
        'Spec Toon Size': "Angular size of the toon highlight",
        'Spec Toon Smooth': "Softness of the toon highlight's edge",
        'Emit': "Emit: the diffuse colour glows by this amount, "
                "textures included, exactly as BI multiplied it",
        'Ambient': "How much of the world's ambient colour this "
                   "material receives (Amb)",
        'Translucency': "Light from behind shows through by this much",
        'Alpha': "Opacity, BI's Alpha slider (Transparency Fresnel "
                 "above 0 REPLACES it, as BI did)",
        'Transp Fresnel': "View-angle transparency: 0 keeps Alpha; "
                          "higher fades the facing surface out",
        'Transp Blend': "Blending of the transparency Fresnel; above "
                        "1 turns the gradient the classic way",
        'Transp Specular': "How much highlights stay opaque on "
                           "transparent areas (BI's spectra)",
        'Ray IOR': "Refraction index for Raytrace transparency (the "
                   "Blinn Refr slider is spectral, not refractive)",
        'Filter': "0 refracts untinted; 1 tints the refraction by the "
                  "base colour",
        'Mirror': "Ray-mirror reflectivity",
        'Mirror Color': "Tint of the mirrored reflection",
        'Mirror Fresnel': "View-angle mirror: 0 reflects flat; higher "
                          "keeps reflection at grazing angles only",
        'Mirror Blend': "Blending of the mirror Fresnel",
        'Normal': "Replacement shading normal",
        'Bump Strength': "How far Bump Height may bend the normal",
        'Bump Height': "Height field to bump the surface with",
    }

    #: sockets every shader pair shows (the Transparency and Mirror
    #: panels add theirs behind their toggles)
    BI_BASE = {'Color', 'Intensity', 'Specular Color',
               'Specular Intensity', 'Emit', 'Ambient', 'Translucency',
               'Normal', 'Bump Strength', 'Bump Height'}
    #: extra sockets per DIFFUSE choice (display names)
    BI_DIFF_EXTRA = {'OREN_NAYAR': {'Roughness'},
                     'MINNAERT': {'Darkness'},
                     'TOON': {'Toon Size', 'Toon Smooth'},
                     'FRESNEL': {'Fresnel', 'Fresnel Factor'}}
    #: extra sockets per SPECULAR choice
    BI_SPEC_EXTRA = {'COOKTORR': {'Hardness'}, 'PHONG': {'Hardness'},
                     'BLINN': {'Hardness', 'Refr'},
                     'TOON': {'Spec Toon Size', 'Spec Toon Smooth'},
                     'WARDISO': {'Slope'}}

    def init(self, context):
        for kind, name, ident, default in self.BI_SOCKETS:
            try:
                sock = self.inputs.new(kind, name, identifier=ident)
            except TypeError:
                # an API without the identifier kwarg: fall back to the
                # identifier AS the name, keeping the machinery correct
                # at the cost of the BI label
                sock = self.inputs.new(kind, ident)
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass
            doc = self.BI_DOCS.get(name)
            if doc:
                try:
                    sock.description = doc
                except (AttributeError, TypeError):
                    pass
        self.outputs.new('NodeSocketShader', 'Surface')
        self.outputs[0].description = (
            "Connect to Material Output. Shades with the chosen "
            "Blender Internal diffuse and specular pair")
        self.refresh_sockets()

    def ensure_sockets(self):
        """Create any socket this saved instance predates (file load)."""
        have = {s.name for s in self.inputs}
        have |= {getattr(s, 'identifier', s.name) for s in self.inputs}
        for kind, name, ident, default in self.BI_SOCKETS:
            if name in have or ident in have:
                continue
            try:
                sock = self.inputs.new(kind, name, identifier=ident)
            except TypeError:
                try:
                    sock = self.inputs.new(kind, ident)
                except Exception:                               # noqa: BLE001
                    continue
            if default is not None:
                try:
                    sock.default_value = default
                except (TypeError, ValueError):
                    pass

    def refresh_sockets(self):
        if self.shadeless:
            keep = {'Color'}
            if self.use_transparency:
                keep.add('Alpha')
        else:
            keep = set(self.BI_BASE)
            keep |= self.BI_DIFF_EXTRA.get(self.diff_shader, set())
            keep |= self.BI_SPEC_EXTRA.get(self.spec_shader, set())
            if self.use_transparency:
                keep |= {'Alpha', 'Transp Fresnel', 'Transp Blend',
                         'Transp Specular'}
                if self.transp_mode == 'RAYTRACE':
                    keep |= {'Ray IOR', 'Filter'}
            if self.use_mirror:
                keep |= {'Mirror', 'Mirror Color', 'Mirror Fresnel',
                         'Mirror Blend'}
        for sock in self.inputs:
            sock.hide = bool(sock.name not in keep
                             and not sock.is_linked)

    # ---- the gradient widget: a hidden Blend texture datablock owns
    # a real ColorRamp per ramp, so the band edits as a band (the 2.79
    # panel's own control) instead of position/colour rows. The stops
    # collection stays the engine's fallback: export prefers the ramp.
    def ramp_texture(self, which, create=False):
        """The hidden texture whose color_ramp IS this ramp's band."""
        attr = f'{which}_ramp_tex'
        name = getattr(self, attr, '')
        try:
            texes = bpy.data.textures
        except AttributeError:
            return None
        tex = texes.get(name) if name else None
        if tex is None and create:
            import uuid
            name = f'.hal_biramp_{uuid.uuid4().hex[:12]}'
            try:
                tex = texes.new(name, type='BLEND')
                tex.use_color_ramp = True
                tex.use_fake_user = True
                setattr(self, attr, name)
            except Exception:                                   # noqa: BLE001
                return None
        return tex

    def ramp_stops(self, which):
        """[(pos, r, g, b, a), ...] -- the GRADIENT when it exists,
        else the fallback collection. This is what export reads."""
        tex = self.ramp_texture(which)
        ramp = getattr(tex, 'color_ramp', None) if tex is not None \
            else None
        if ramp is not None and len(ramp.elements):
            return sorted(
                (float(e.position), float(e.color[0]), float(e.color[1]),
                 float(e.color[2]), float(e.color[3]))
                for e in ramp.elements)
        stops = getattr(self, 'dif_stops' if which == 'dif'
                        else 'spec_stops')
        return sorted((float(s.position), float(s.color[0]),
                       float(s.color[1]), float(s.color[2]),
                       float(s.color[3])) for s in stops)

    def ramp_ipo(self, which):
        """do_colorband's integer, from the gradient's own dropdown
        when it exists, else the enum prop."""
        ipo_map = {'LINEAR': 0, 'EASE': 1, 'B_SPLINE': 2,
                   'CARDINAL': 3, 'CONSTANT': 4}
        tex = self.ramp_texture(which)
        ramp = getattr(tex, 'color_ramp', None) if tex is not None \
            else None
        if ramp is not None:
            return ipo_map.get(str(getattr(ramp, 'interpolation',
                                           'LINEAR')), 0)
        return ipo_map.get(str(getattr(self, f'ramp_{which}_ipo',
                                       'LINEAR')), 0)

    def set_ramp_stops(self, which, stops, ipotype=0):
        """Write stops into BOTH the gradient and the fallback
        collection -- the importer's entry point."""
        coll = getattr(self, 'dif_stops' if which == 'dif'
                       else 'spec_stops')
        try:
            coll.clear()
            for stop in stops:
                s = coll.add()
                s.position = float(stop[0])
                s.color = (float(stop[1]), float(stop[2]),
                           float(stop[3]), float(stop[4]))
        except Exception:                                       # noqa: BLE001
            pass
        ipo_names = ('LINEAR', 'EASE', 'B_SPLINE', 'CARDINAL',
                     'CONSTANT')
        try:
            setattr(self, f'ramp_{which}_ipo',
                    ipo_names[int(ipotype)]
                    if 0 <= int(ipotype) < 5 else 'LINEAR')
        except (TypeError, ValueError):
            pass
        tex = self.ramp_texture(which, create=True)
        ramp = getattr(tex, 'color_ramp', None) if tex is not None \
            else None
        if ramp is None or not stops:
            return
        try:
            ramp.interpolation = ipo_names[int(ipotype)] \
                if 0 <= int(ipotype) < 5 else 'LINEAR'
        except (TypeError, ValueError):
            pass
        try:
            # a ColorRamp always keeps at least one element: position
            # the first, add the rest, then trim extras
            while len(ramp.elements) > 1:
                ramp.elements.remove(ramp.elements[-1])
            ramp.elements[0].position = float(stops[0][0])
            ramp.elements[0].color = tuple(float(x) for x in stops[0][1:5])
            for stop in stops[1:]:
                e = ramp.elements.new(float(stop[0]))
                e.color = tuple(float(x) for x in stop[1:5])
        except Exception:                                       # noqa: BLE001
            pass

    def _draw_ramp(self, layout, prefix, stops_attr):
        box = layout.box()
        row = box.row(align=True)
        row.prop(self, f'ramp_{prefix}_input', text="")
        row.prop(self, f'ramp_{prefix}_blend', text="")
        box.prop(self, f'ramp_{prefix}_factor')
        tex = self.ramp_texture(prefix)
        if tex is None:
            # first touch: offer the gradient; rows stay as fallback
            op = box.operator('halcyon.bi_ramp_gradient',
                              text="Show Gradient", icon='COLOR')
            op.node_name, op.ramp = self.name, \
                'DIF' if prefix == 'dif' else 'SPEC'
        if tex is not None and hasattr(tex, 'color_ramp'):
            box.template_color_ramp(tex, 'color_ramp', expand=True)
            return
        row = box.row(align=True)
        row.prop(self, f'ramp_{prefix}_ipo', text="")
        stops = getattr(self, stops_attr)
        which = 'DIF' if prefix == 'dif' else 'SPEC'
        for i, s in enumerate(stops):
            row = box.row(align=True)
            row.prop(s, 'position', text="Pos")
            row.prop(s, 'color', text="")
            op = row.operator('halcyon.bi_ramp_stop', text="",
                              icon='REMOVE')
            op.node_name, op.ramp, op.action, op.index = \
                self.name, which, 'REMOVE', i
        op = box.operator('halcyon.bi_ramp_stop', text="Add Stop",
                          icon='ADD')
        op.node_name, op.ramp, op.action = self.name, which, 'ADD'
        if len(stops) == 0:
            box.label(text="No stops: the ramp is inert", icon='INFO')

    def draw_buttons(self, context, layout):
        col = layout.column()
        col.prop(self, 'diff_shader', text="Diffuse")
        col.prop(self, 'use_ramp_dif', toggle=False)
        if self.use_ramp_dif:
            self._draw_ramp(col, 'dif', 'dif_stops')
        col.prop(self, 'spec_shader', text="Specular")
        col.prop(self, 'use_ramp_spec', toggle=False)
        if self.use_ramp_spec:
            self._draw_ramp(col, 'spec', 'spec_stops')
        col.separator()
        row = col.row(align=True)
        row.prop(self, 'shadeless', toggle=True)
        row = col.row(align=True)
        row.prop(self, 'use_cubic', text="Cubic", toggle=True)
        row.prop(self, 'use_tangent_v', text="Tangent", toggle=True)
        col.separator()
        col.prop(self, 'use_transparency')
        if self.use_transparency:
            col.row().prop(self, 'transp_mode', expand=True)
        col.prop(self, 'use_mirror')
        col.separator()
        col.label(text="Options:")
        row = col.row(align=True)
        row.prop(self, 'use_mist', text="Mist", toggle=True)
        row = col.row(align=True)
        row.prop(self, 'vcol_paint', text="VCol Paint", toggle=True)
        row.prop(self, 'vcol_light', text="VCol Light", toggle=True)
        row = col.row(align=True)
        row.prop_search(self, 'light_group', bpy.data, 'collections',
                        text="Light Group")
        if self.light_group:
            col.prop(self, 'light_group_exclusive')
        col.separator()
        col.label(text="Shadow:")
        row = col.row(align=True)
        row.prop(self, 'shadow_receive', text="Receive", toggle=True)
        row.prop(self, 'shadow_cast', text="Cast", toggle=True)
        row = col.row(align=True)
        row.prop(self, 'shadow_cast_only', text="Cast Only", toggle=True)
        row.prop(self, 'shadow_only', text="Shadows Only", toggle=True)
        row = col.row(align=True)
        row.prop(self, 'sbias', text="Bias")
        row.prop(self, 'raybias', text="Ray Bias", toggle=True)
        col.prop(self, 'use_obcolor')
        col.separator()
        col.prop(self, 'sss_enable')
        if self.sss_enable:
            box = col.box()
            box.prop(self, 'sss_ior')
            box.prop(self, 'sss_scale')
            box.prop(self, 'sss_color')
            box.prop(self, 'sss_radius')
            row = box.row(align=True)
            row.prop(self, 'sss_colfac', text="Color")
            row.prop(self, 'sss_texfac', text="Texture")
            row = box.row(align=True)
            row.prop(self, 'sss_front', text="Front")
            row.prop(self, 'sss_back', text="Back")
            box.prop(self, 'sss_error')


# ===================================================== normal-map workflow


class HALCYON_NormalMapNode(Node, HalcyonNodeBase):
    """Decode a normal map of any common flavour into a shading normal"""

    bl_idname = 'HALCYON_NormalMapNode'
    bl_label = "Normal Map+"
    bl_icon = 'NORMALS_FACE'

    space: EnumProperty(name="Space", default='TANGENT', items=(
        ('TANGENT', "Tangent Space",
         "The blue-ish maps baked against the surface -- the usual kind"),
        ('OBJECT', "Object Space",
         "Rainbow maps whose colours are directions in the model itself"),
        ('WORLD', "World Space", "Directions in world axes")))
    map_type: EnumProperty(name="Type", default='OPENGL', items=(
        ('OPENGL', "OpenGL (Y+)",
         "Green points up -- Blender, Maya, and most bakers"),
        ('DIRECTX', "DirectX (Y-)",
         "Green points down -- 3ds Max, Unreal, many game rips. If your "
         "bumps look inverted, the map is the other type")))

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.5, 0.5, 1.0, 1.0)
        s = self.inputs.new('NodeSocketFloat', 'Strength')
        s.default_value = 1.0
        self.outputs.new('NodeSocketVector', 'Normal')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'space', text="")
        layout.prop(self, 'map_type', text="")


class HALCYON_NormalMixNode(Node, HalcyonNodeBase):
    """Combine two normals -- the Mix node's job, for normal chains"""

    bl_idname = 'HALCYON_NormalMixNode'
    bl_label = "Normal Mix"
    bl_icon = 'ORIENTATION_NORMAL'

    mode: EnumProperty(name="Mode", default='DETAIL', items=(
        ('DETAIL', "Detail",
         "Reorient the second normal onto the first -- base map plus "
         "detail map, the way engines layer them"),
        ('ADD', "Add",
         "Sum the two tilts away from the surface. Cheaper, cruder, and "
         "what most 1990s tools did when they did anything"),
        ('MIX', "Mix", "Fade between the two normals by the factor")))

    def init(self, context):
        self.inputs.new('NodeSocketVector', 'Base')
        self.inputs.new('NodeSocketVector', 'Detail')
        f = self.inputs.new('NodeSocketFloat', 'Factor')
        f.default_value = 1.0
        self.outputs.new('NodeSocketVector', 'Normal')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode', text="")


# ========================================================== utility nodes


class HALCYON_AltitudeSlopeNode(Node, HalcyonNodeBase):
    """World height, steepness and facing -- Bryce's terrain-material trio"""

    bl_idname = 'HALCYON_AltitudeSlopeNode'
    bl_label = "Altitude & Slope"
    bl_icon = 'RNDCURVE'

    #: R202: the optional noise -- spatial value noise wobbling the
    #: three masks, so altitude bands get the ragged natural edge a
    #: hard height line never has. 0 is exactly the old node.
    SOCKETS = (('NodeSocketFloat', 'Minimum', 0.0),
               ('NodeSocketFloat', 'Maximum', 10.0),
               ('NodeSocketFloat', 'Noise', 0.0),
               ('NodeSocketFloat', 'Noise Scale', 4.0))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            s.default_value = default
        try:
            self.inputs['Noise'].description = (
                "Wobbles Altitude, Factor and Slope with spatial value "
                "noise -- ragged snowlines and rock bands instead of "
                "hard height contours; 0 is off")
            self.inputs['Noise Scale'].description = (
                "World-units-per-cell frequency of the band noise; "
                "higher is finer raggedness")
        except (AttributeError, TypeError):
            pass
        self.outputs.new('NodeSocketFloat', 'Altitude')
        self.outputs.new('NodeSocketFloat', 'Factor')
        self.outputs.new('NodeSocketFloat', 'Slope')
        self.outputs.new('NodeSocketFloat', 'Orientation')

    def ensure_sockets(self):
        """Old saved nodes gain the R202 noise inputs at file load."""
        have = {s.name for s in self.inputs}
        for kind, name, default in self.SOCKETS:
            if name in have:
                continue
            try:
                s = self.inputs.new(kind, name)
                s.default_value = default
            except Exception:                                   # noqa: BLE001
                pass


class HALCYON_FacingNode(Node, HalcyonNodeBase):
    """View-angle masks: facing ratio, incidence and a Fresnel curve"""

    bl_idname = 'HALCYON_FacingNode'
    bl_label = "Facing"
    bl_icon = 'MATSPHERE'

    def init(self, context):
        p = self.inputs.new('NodeSocketFloat', 'Power')
        p.default_value = 1.0
        i = self.inputs.new('NodeSocketFloat', 'IOR')
        i.default_value = 1.45
        self.outputs.new('NodeSocketFloat', 'Facing')
        self.outputs.new('NodeSocketFloat', 'Incidence')
        self.outputs.new('NodeSocketFloat', 'Fresnel')


class HALCYON_IridescentNode(Node, HalcyonNodeBase):
    """R202: view-angle iridescence -- the colour that will not sit
    still. Four types: a pure rainbow sweep, physical-ish thin-film
    interference (the soap bubble), pearl nacre, and an oil slick
    whose film thickness swirls with surface noise. Deterministic on
    both devices; plug Color into a diffuse or specular input and
    Factor wherever the rim mask helps."""

    bl_idname = 'HALCYON_IridescentNode'
    bl_label = "Iridescent"
    bl_icon = 'NODE_MATERIAL'

    mode: EnumProperty(
        name="Type", default='SPECTRUM',
        description="Which iridescence this node makes; each entry "
                    "names its optical character",
        items=[
            ('SPECTRUM', "Rainbow Sweep",
             "The pure hue wheel swept across the facing angle -- the "
             "90s logo chrome"),
            ('THIN_FILM', "Thin Film",
             "Soap-bubble interference: per-channel cosines at real "
             "wavelength ratios, so the fringes order themselves the "
             "way a real film's do"),
            ('PEARL', "Pearl",
             "Nacre: white face rolling to the Tint at the rim, with "
             "a soft spectral kiss riding the turn"),
            ('OIL', "Oil Slick",
             "Thin film whose thickness swirls with surface noise -- "
             "the parking-lot rainbow"),
        ])

    SOCKETS = (('NodeSocketFloat', 'Shift', 0.0),
               ('NodeSocketFloat', 'Scale', 1.0),
               ('NodeSocketFloat', 'Saturation', 1.0),
               ('NodeSocketColor', 'Tint', (0.72, 0.48, 0.85, 1.0)),
               ('NodeSocketFloat', 'Noise Scale', 6.0))

    _DOCS = {
        'Shift': "Slides the whole colour cycle around; keyframe it "
                 "and the rainbow crawls across the surface",
        'Scale': "How many colour cycles fit across the facing angle "
                 "(the film's thickness, for the physical types)",
        'Saturation': "1 is the full colour; toward 0 the fringes "
                      "fade into the surface neutrally",
        'Tint': "The Pearl type's rim colour; the face stays white",
        'Noise Scale': "The Oil Slick's swirl frequency in world "
                       "units; higher is a tighter marbling",
    }

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            if default is not None:
                try:
                    s.default_value = default
                except (TypeError, ValueError):
                    pass
            try:
                s.description = self._DOCS.get(name, '')
            except (AttributeError, TypeError):
                pass
        self.outputs.new('NodeSocketColor', 'Color')
        self.outputs.new('NodeSocketFloat', 'Factor')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'mode', text="")


class HALCYON_SwitchNode(Node, HalcyonNodeBase):
    """Hard A/B selector -- anything above one half picks B"""

    bl_idname = 'HALCYON_SwitchNode'
    bl_label = "Switch"
    bl_icon = 'ARROW_LEFTRIGHT'

    def init(self, context):
        s = self.inputs.new('NodeSocketFloat', 'Switch')
        s.default_value = 0.0
        a = self.inputs.new('NodeSocketColor', 'A')
        a.default_value = (0.0, 0.0, 0.0, 1.0)
        b = self.inputs.new('NodeSocketColor', 'B')
        b.default_value = (1.0, 1.0, 1.0, 1.0)
        self.outputs.new('NodeSocketColor', 'Color')


class HALCYON_RandomPerObjectNode(Node, HalcyonNodeBase):
    """A stable random value and colour per object, for cheap variation"""

    bl_idname = 'HALCYON_RandomPerObjectNode'
    bl_label = "Random Per Object"
    bl_icon = 'FORCE_TURBULENCE'

    def init(self, context):
        s = self.inputs.new('NodeSocketFloat', 'Seed')
        s.default_value = 0.0
        self.outputs.new('NodeSocketFloat', 'Value')
        self.outputs.new('NodeSocketColor', 'Color')


class HALCYON_LevelsNode(Node, HalcyonNodeBase):
    """Black point, white point, gamma -- the Levels dialog as a node"""

    bl_idname = 'HALCYON_LevelsNode'
    bl_label = "Levels"
    bl_icon = 'SEQ_HISTOGRAM'

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.8, 0.8, 0.8, 1.0)
        b = self.inputs.new('NodeSocketFloat', 'Black')
        b.default_value = 0.0
        w = self.inputs.new('NodeSocketFloat', 'White')
        w.default_value = 1.0
        g = self.inputs.new('NodeSocketFloat', 'Gamma')
        g.default_value = 1.0
        om = self.inputs.new('NodeSocketFloat', 'Out Min')
        om.default_value = 0.0
        ox = self.inputs.new('NodeSocketFloat', 'Out Max')
        ox.default_value = 1.0
        self.outputs.new('NodeSocketColor', 'Color')


class HALCYON_SmoothStepNode(Node, HalcyonNodeBase):
    """Remap a value's range to 0..1, with a choice of easing"""

    bl_idname = 'HALCYON_SmoothStepNode'
    bl_label = "Smooth Step"
    bl_icon = 'IPO_EASE_IN_OUT'

    interp: EnumProperty(name="Easing", default='SMOOTH', items=(
        ('SMOOTH', "Smooth", "Hermite smoothstep"),
        ('SMOOTHER', "Smoother", "Perlin's quintic -- flat at both ends"),
        ('LINEAR', "Linear", "A straight clamped ramp")))

    def init(self, context):
        v = self.inputs.new('NodeSocketFloat', 'Value')
        v.default_value = 0.5
        a = self.inputs.new('NodeSocketFloat', 'From Min')
        a.default_value = 0.0
        b = self.inputs.new('NodeSocketFloat', 'From Max')
        b.default_value = 1.0
        self.outputs.new('NodeSocketFloat', 'Value')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'interp', text="")


class HALCYON_ChannelShuffleNode(Node, HalcyonNodeBase):
    """Reorder a colour's channels -- the compositor's Shuffle, at shade time"""

    bl_idname = 'HALCYON_ChannelShuffleNode'
    bl_label = "Channel Shuffle"
    bl_icon = 'COLOR'

    _CH = (('R', "R", ""), ('G', "G", ""), ('B', "B", ""), ('A', "A", ""),
           ('ZERO', "0", ""), ('ONE', "1", ""))
    out_r: EnumProperty(name="R", items=_CH, default='R')
    out_g: EnumProperty(name="G", items=_CH, default='G')
    out_b: EnumProperty(name="B", items=_CH, default='B')
    out_a: EnumProperty(name="A", items=_CH, default='A')

    def init(self, context):
        c = self.inputs.new('NodeSocketColor', 'Color')
        c.default_value = (0.8, 0.8, 0.8, 1.0)
        self.outputs.new('NodeSocketColor', 'Color')

    def draw_buttons(self, context, layout):
        row = layout.row(align=True)
        row.prop(self, 'out_r', text="")
        row.prop(self, 'out_g', text="")
        row.prop(self, 'out_b', text="")
        row.prop(self, 'out_a', text="")


class HALCYON_DistanceMaskNode(Node, HalcyonNodeBase):
    """0 to 1 by camera distance -- Depth Cue's factor, free for anything"""

    bl_idname = 'HALCYON_DistanceMaskNode'
    bl_label = "Distance Mask"
    bl_icon = 'DRIVER_DISTANCE'

    def init(self, context):
        s = self.inputs.new('NodeSocketFloat', 'Start')
        s.default_value = 5.0
        e = self.inputs.new('NodeSocketFloat', 'End')
        e.default_value = 50.0
        self.outputs.new('NodeSocketFloat', 'Factor')
        self.outputs.new('NodeSocketFloat', 'Distance')


class HALCYON_StepTimeNode(Node, HalcyonNodeBase):
    """Hold the clock every N frames -- animating on twos, as a node"""

    bl_idname = 'HALCYON_StepTimeNode'
    bl_label = "Stepped Time"
    bl_icon = 'KEYFRAME_HLT'

    def init(self, context):
        s = self.inputs.new('NodeSocketFloat', 'Step Frames')
        s.default_value = 2.0
        self.outputs.new('NodeSocketFloat', 'Frame')
        self.outputs.new('NodeSocketFloat', 'Phase')


class HALCYON_WaveNode(Node, HalcyonNodeBase):
    """A waveform oscillator: sine, square, triangle or saw of any input"""

    bl_idname = 'HALCYON_WaveNode'
    bl_label = "Waveform"
    bl_icon = 'FORCE_HARMONIC'

    wave: EnumProperty(name="Wave", default='SINE', items=(
        ('SINE', "Sine", "Smooth oscillation"),
        ('SQUARE', "Square", "Hard on/off, the blink"),
        ('TRIANGLE', "Triangle", "Linear rise and fall"),
        ('SAW', "Saw", "Linear rise, instant drop")))

    def init(self, context):
        v = self.inputs.new('NodeSocketFloat', 'Value')
        v.default_value = 0.0
        f = self.inputs.new('NodeSocketFloat', 'Frequency')
        f.default_value = 1.0
        p = self.inputs.new('NodeSocketFloat', 'Phase')
        p.default_value = 0.0
        mn = self.inputs.new('NodeSocketFloat', 'Minimum')
        mn.default_value = 0.0
        mx = self.inputs.new('NodeSocketFloat', 'Maximum')
        mx.default_value = 1.0
        self.outputs.new('NodeSocketFloat', 'Value')

    def draw_buttons(self, context, layout):
        layout.prop(self, 'wave', text="")


# --------------------------------------- the R216 families, spec-generated
#
# Fifteen utilities and fifteen vector nodes, from one spec table each --
# the same generation the pattern nodes use, so a socket cannot drift
# from its evaluator. All are pure functions of their inputs and the
# frame clock except Light Meter, which reads the LAMPS at shading time
# (and says so on the GPU, by name).

_GF = 'NodeSocketFloat'
_GC = 'NodeSocketColor'
_GV = 'NodeSocketVector'

#: name, label, icon, doc, inputs (kind, name, default), props, outputs
UTIL_SPECS = [
    ('LightMeter', "Light Meter", 'LIGHT_SUN',
     "How much lamp light lands on this point: Color is the summed "
     "lamp light (Lambert), Fac its brightness clamped 0..1. Invert "
     "Fac to drive Self-Illumination and a material GLOWS IN THE DARK "
     "-- charge-by-light, the era's phosphor trick. Reads the lamps at "
     "shading time; Shadows adds one occlusion ray per lamp",
     [(_GF, 'Exposure', 1.0)],
     {'shadows': ('bool', False, "Shadows")},
     [(_GC, 'Color'), (_GF, 'Fac')]),
    ('Timer', "Timer", 'TIME',
     "The scene clock as sockets: Frame, Seconds, and Loop -- the "
     "frame wrapped 0..1 over Loop Frames, ready for any cyclic "
     "animation",
     [(_GF, 'Loop Frames', 48.0)], {},
     [(_GF, 'Frame'), (_GF, 'Seconds'), (_GF, 'Loop')]),
    ('Oscillator', "Oscillator", 'IPO_SINE',
     "A waveform on the clock: sine, square, triangle or saw between "
     "Min and Max. Pulsing glows, blinking beacons, breathing scale",
     [(_GF, 'Speed', 1.0), (_GF, 'Phase', 0.0), (_GF, 'Min', 0.0),
      (_GF, 'Max', 1.0)],
     {'wave': ('enum', 'SINE',
               (('SINE', "Sine", ""), ('SQUARE', "Square", ""),
                ('TRIANGLE', "Triangle", ""), ('SAW', "Saw", "")),
               "Wave")},
     [(_GF, 'Value')]),
    ('Counter', "Counter", 'LINENUMBERS_ON',
     "An integer that steps up every N frames and wraps at Modulo -- "
     "the flipbook and palette driver",
     [(_GF, 'Frames Per Step', 4.0), (_GF, 'Modulo', 8.0),
      (_GF, 'Offset', 0.0)], {},
     [(_GF, 'Value')]),
    ('Pulse', "Pulse", 'PMARKER_ACT',
     "1 for Width frames out of every Period frames, else 0 -- strobe "
     "lights, camera flashes, warning beacons",
     [(_GF, 'Period', 24.0), (_GF, 'Width', 4.0), (_GF, 'Phase', 0.0)],
     {},
     [(_GF, 'Fac')]),
    ('Gate', "Gate", 'CHECKBOX_HLT',
     "A threshold with an optional soft edge: 0 below, 1 above, "
     "smooth across Softness. The building block of masks",
     [(_GF, 'Value', 0.5), (_GF, 'Threshold', 0.5),
      (_GF, 'Softness', 0.0)],
     {'invert': ('bool', False, "Invert")},
     [(_GF, 'Fac')]),
    ('Selector', "Selector", 'PRESET',
     "Index picks one of four colours -- palette animation's other "
     "half, fed from a Counter",
     [(_GF, 'Index', 0.0),
      (_GC, 'Color 1', (0.9, 0.2, 0.2, 1.0)),
      (_GC, 'Color 2', (0.9, 0.8, 0.2, 1.0)),
      (_GC, 'Color 3', (0.2, 0.8, 0.3, 1.0)),
      (_GC, 'Color 4', (0.2, 0.4, 0.9, 1.0))],
     {'wrap': ('bool', True, "Wrap")},
     [(_GC, 'Color')]),
    ('ColorKey', "Color Key", 'EYEDROPPER',
     "Chroma key: Fac is 1 where Color sits within Tolerance of the "
     "Key, softened over Softness. Multiply a decal's alpha by Matte "
     "to knock its backing colour out",
     [(_GC, 'Color', (0.8, 0.8, 0.8, 1.0)),
      (_GC, 'Key', (0.0, 1.0, 0.0, 1.0)),
      (_GF, 'Tolerance', 0.15), (_GF, 'Softness', 0.1)], {},
     [(_GF, 'Fac'), (_GF, 'Matte')]),
    ('Measure', "Measure", 'DRIVER_DISTANCE',
     "A distance as a factor: from the object origin, a point, the "
     "camera, or along an axis, divided by Scale. Gradients that "
     "follow space instead of UVs",
     [(_GV, 'Point', None), (_GF, 'Scale', 1.0)],
     {'mode': ('enum', 'ORIGIN',
               (('ORIGIN', "From Object Origin", ""),
                ('POINT', "From Point", ""),
                ('CAMERA', "From Camera", ""),
                ('AXIS_X', "Along X", ""), ('AXIS_Y', "Along Y", ""),
                ('AXIS_Z', "Along Z", "")), "Measure")},
     [(_GF, 'Fac')]),
    ('StepRamp', "Step Ramp", 'SEQ_HISTOGRAM',
     "The ramp quantised to N flat bands -- toon shading's colour "
     "side, no stops to edit. Band is the integer band index",
     [(_GF, 'Fac', 0.5), (_GF, 'Steps', 4.0),
      (_GC, 'Color 1', (0.05, 0.05, 0.08, 1.0)),
      (_GC, 'Color 2', (0.95, 0.92, 0.85, 1.0))], {},
     [(_GC, 'Color'), (_GF, 'Band')]),
    ('Wobble', "Wobble", 'FORCE_TURBULENCE',
     "A value with smooth animated jitter on it: the idle hover, the "
     "nervous needle, the unsteady candle. Integer-hash noise, so both "
     "devices wobble identically",
     [(_GF, 'Value', 0.0), (_GF, 'Amount', 0.1), (_GF, 'Speed', 1.0)],
     {'seed': ('int', 0, 0, 9999, "Seed")},
     [(_GF, 'Value')]),
    ('FrameBlend', "Frame Blend", 'PREVIEW_RANGE',
     "Crossfade from A to B between two frames -- reveals, day-to-"
     "night sweeps, damage states",
     [(_GC, 'A', (0.8, 0.8, 0.8, 1.0)), (_GC, 'B', (0.1, 0.1, 0.1, 1.0)),
      (_GF, 'Start Frame', 1.0), (_GF, 'End Frame', 24.0)], {},
     [(_GC, 'Color'), (_GF, 'Fac')]),
    ('Blackbody', "Blackbody", 'LIGHT',
     "Temperature to colour, 1000K to 12000K: candle orange through "
     "daylight to sky blue -- lava, filaments, star fields",
     [(_GF, 'Kelvin', 3000.0)], {},
     [(_GC, 'Color')]),
    ('Compare', "Compare", 'ARROW_LEFTRIGHT',
     "A against B: three 0/1 sockets for equal (within Epsilon), "
     "greater and less. Logic without nests of Math nodes",
     [(_GF, 'A', 0.0), (_GF, 'B', 0.0), (_GF, 'Epsilon', 0.0001)], {},
     [(_GF, 'Equal'), (_GF, 'Greater'), (_GF, 'Less')]),
    ('OnFrame', "On Frame", 'MARKER_HLT',
     "1 while the frame sits inside Start..End, else 0 -- shot "
     "switches and timed reveals",
     [(_GF, 'Start Frame', 1.0), (_GF, 'End Frame', 24.0)], {},
     [(_GF, 'Fac')]),
]

VEC_SPECS = [
    ('ArrayVec', "Array", 'MOD_ARRAY',
     "Decal placement: copies of a local frame arranged on a line, "
     "grid, circle, square, polygon or star. Feed the Vector into an "
     "Image Texture (CLIP extension) and the image lands once per "
     "copy; Index and Random tell the copies apart. Orient turns "
     "copies to follow the shape",
     [(_GV, 'Vector', None), (_GV, 'Center', None),
      (_GF, 'Size', 0.18), (_GF, 'Spacing', 0.25),
      (_GF, 'Radius', 0.35), (_GF, 'Rotation', 0.0),
      (_GF, 'Jitter', 0.0), (_GF, 'Inner', 0.5)],
     {'mode': ('enum', 'CIRCLE',
               (('LINE', "Line", "Count copies along a line"),
                ('GRID', "Grid", "Count x Rows copies"),
                ('CIRCLE', "Circle", "Count copies on a ring"),
                ('SQUARE', "Square", "Count copies round a square"),
                ('POLYGON', "Polygon", "A copy at each of Sides "
                 "corners"),
                ('STAR', "Star", "Sides points, Inner sets the "
                 "notch")), "Arrange"),
      'count': ('int', 8, 1, 64, "Count"),
      'sides': ('int', 5, 2, 24, "Sides / Rows"),
      'orient': ('bool', True, "Orient To Shape")},
     [(_GV, 'Vector'), (_GF, 'Index'), (_GF, 'Random')]),
    ('MirrorTile', "Mirror Tile", 'MOD_MIRROR',
     "Mirror-repeat tiling: every second tile flips, so any image "
     "becomes seamless -- the era's bathroom-floor trick",
     [(_GV, 'Vector', None), (_GF, 'Scale', 2.0)],
     {'axis_x': ('bool', True, "Mirror X"),
      'axis_y': ('bool', True, "Mirror Y")},
     [(_GV, 'Vector')]),
    ('Kaleidoscope', "Kaleidoscope", 'SEQ_CHROMA_SCOPE',
     "The angle folded into N mirrored sectors about the centre -- "
     "mandalas from anything",
     [(_GV, 'Vector', None), (_GV, 'Center', None),
      (_GF, 'Sectors', 6.0), (_GF, 'Angle', 0.0)], {},
     [(_GV, 'Vector')]),
    ('Polar', "Polar Coordinates", 'CURVE_NCIRCLE',
     "Rectangular to polar and back: Vector out carries (angle 0..1, "
     "radius, z), with Radius and Angle as their own sockets -- ring "
     "gradients, radar sweeps, clock faces",
     [(_GV, 'Vector', None), (_GV, 'Center', None)],
     {'direction': ('enum', 'TO_POLAR',
                    (('TO_POLAR', "To Polar", ""),
                     ('FROM_POLAR', "From Polar", "")), "Direction")},
     [(_GV, 'Vector'), (_GF, 'Radius'), (_GF, 'Angle')]),
    ('Twirl', "Twirl", 'FORCE_VORTEX',
     "A rotation that falls off with distance from the centre -- the "
     "classic 90s image-editor twirl, in UV space",
     [(_GV, 'Vector', None), (_GV, 'Center', None),
      (_GF, 'Angle', 3.14159), (_GF, 'Radius', 0.5)], {},
     [(_GV, 'Vector')]),
    ('Lens', "Lens Distort", 'PROP_PROJECTED',
     "Barrel (positive) or pincushion (negative) distortion about the "
     "centre -- the CRT bulge, the fisheye lens",
     [(_GV, 'Vector', None), (_GV, 'Center', None),
      (_GF, 'Amount', 0.2)], {},
     [(_GV, 'Vector')]),
    ('RippleWarp', "Ripple Warp", 'MOD_WAVE',
     "Concentric sine displacement from the centre, animated -- the "
     "stone in the pond, warping whatever samples through it",
     [(_GV, 'Vector', None), (_GV, 'Center', None),
      (_GF, 'Amplitude', 0.02), (_GF, 'Frequency', 10.0),
      (_GF, 'Speed', 1.0)],
     {'animate': ('bool', True, "Animate")},
     [(_GV, 'Vector')]),
    ('WaveWarp', "Wave Warp", 'MOD_NOISE',
     "A directional sine offset, animated -- flags, heat shimmer, "
     "seaweed sway",
     [(_GV, 'Vector', None), (_GF, 'Amplitude', 0.03),
      (_GF, 'Wavelength', 0.25), (_GF, 'Speed', 1.0)],
     {'axis': ('enum', 'X', (('X', "Along X", ""), ('Y', "Along Y", "")),
               "Axis"),
      'animate': ('bool', True, "Animate")},
     [(_GV, 'Vector')]),
    ('TileRandom', "Tile Random", 'MESH_GRID',
     "Tiles whose contents each get a hashed 90-degree turn and flip "
     "-- one floor texture stops repeating. Tile ID drives per-tile "
     "variation downstream",
     [(_GV, 'Vector', None), (_GF, 'Scale', 4.0)],
     {'rotate': ('bool', True, "Random Turn"),
      'flip': ('bool', True, "Random Flip")},
     [(_GV, 'Vector'), (_GF, 'Tile ID')]),
    ('VectorSnap', "Vector Snap", 'SNAP_GRID',
     "The vector quantised to a grid step -- chunky UVs, mosaic "
     "sampling, deliberate pixelation in texture space",
     [(_GV, 'Vector', None), (_GF, 'Step', 0.1)],
     {'mode': ('enum', 'FLOOR', (('FLOOR', "Floor", ""),
                                 ('ROUND', "Round", "")), "Snap")},
     [(_GV, 'Vector')]),
    ('Shear', "Shear", 'MOD_SIMPLEDEFORM',
     "X pushed by Y and Y pushed by X -- italic decals, raked "
     "checkerboards",
     [(_GV, 'Vector', None), (_GF, 'X By Y', 0.0),
      (_GF, 'Y By X', 0.0)], {},
     [(_GV, 'Vector')]),
    ('Orbit', "Orbit", 'ORIENTATION_GIMBAL',
     "The vector translated round a small circle over time -- drifting "
     "highlights, floating dust decals, restless goo",
     [(_GV, 'Vector', None), (_GF, 'Radius', 0.05),
      (_GF, 'Speed', 1.0), (_GF, 'Phase', 0.0)],
     {'animate': ('bool', True, "Animate")},
     [(_GV, 'Vector')]),
    ('Region', "Region", 'SELECT_SET',
     "A box in texture space with a chosen outside: clip, wrap, "
     "mirror or extend -- and Inside as a mask. Place ONE decal: clip "
     "outside, multiply your alpha by Inside",
     [(_GV, 'Vector', None), (_GV, 'Min', None), (_GV, 'Max', None)],
     {'outside': ('enum', 'CLIP',
                  (('CLIP', "Clip", ""), ('WRAP', "Wrap", ""),
                   ('MIRROR', "Mirror", ""), ('EXTEND', "Extend", "")),
                  "Outside")},
     [(_GV, 'Vector'), (_GF, 'Inside')]),
    ('Projector', "Projector", 'MOD_UVPROJECT',
     "Planar, cylindrical, spherical or box projection of the input "
     "vector -- the era's mapping modes as a node, for any coordinate "
     "you feed it",
     [(_GV, 'Vector', None), (_GF, 'Scale', 1.0)],
     {'mode': ('enum', 'PLANAR',
               (('PLANAR', "Planar", ""), ('CYLINDER', "Cylindrical", ""),
                ('SPHERE', "Spherical", ""), ('BOX', "Box", "")),
               "Projection"),
      'axis': ('enum', 'Z', (('X', "X", ""), ('Y', "Y", ""),
                             ('Z', "Z", "")), "Axis")},
     [(_GV, 'Vector')]),
    ('Spin', "Spin", 'FILE_REFRESH',
     "A steady rotation about the centre, animated -- fans, wheels, "
     "record decals, hypno-spirals",
     [(_GV, 'Vector', None), (_GV, 'Center', None),
      (_GF, 'Speed', 1.0), (_GF, 'Angle', 0.0)],
     {'animate': ('bool', True, "Animate")},
     [(_GV, 'Vector')]),
]


def _gen_family(specs):
    made = []
    for name, label, icon, doc, ins, props, outs in specs:
        ann = {}
        for key, spec in props.items():
            if spec[0] == 'int':
                _k, default, lo, hi, plabel = spec
                ann[key] = IntProperty(name=plabel, default=default,
                                       min=lo, max=hi)
            elif spec[0] == 'bool':
                _k, default, plabel = spec
                ann[key] = BoolProperty(name=plabel, default=default)
            else:
                _k, default, items, plabel = spec
                ann[key] = EnumProperty(name=plabel, items=list(items),
                                       default=default)

        # Blender's RNA validation counts EVERY named parameter --
        # init must be exactly (self, context) and draw_buttons exactly
        # (self, context, layout). The default-argument capture idiom
        # (init(self, context, _ins=ins)) registers fine in a stub and
        # refuses to register in Blender ("expected ... 2 args, found
        # 4", the field's paste). Real closures keep the signature.
        def _make_init(ins_, outs_):
            def init(self, context):
                for kind, sock_name, default in ins_:
                    sock = self.inputs.new(kind, sock_name)
                    if default is not None:
                        try:
                            sock.default_value = default
                        except (TypeError, ValueError):
                            pass
                for kind, out_name in outs_:
                    self.outputs.new(kind, out_name)
            return init

        def _make_draw(props_):
            def draw_buttons(self, context, layout):
                for key in props_:
                    layout.prop(self, key, text="")
            return draw_buttons

        init = _make_init(ins, outs)
        draw_buttons = _make_draw(props)

        cls = type(f'HALCYON_{name}Node', (Node, HalcyonNodeBase), {
            '__doc__': doc,
            'bl_idname': f'HALCYON_{name}Node',
            'bl_label': label,
            'bl_icon': icon,
            'bl_width_default': 160,
            '__annotations__': ann,
            'init': init,
            'draw_buttons': draw_buttons,
        })
        made.append(cls)
    return tuple(made)


UTIL_NODES = _gen_family(UTIL_SPECS)
VEC_NODES = _gen_family(VEC_SPECS)

#: what the exporter copies for the generated families
FAMILY_NODE_PROPS = {f'HALCYON_{s[0]}Node': tuple(s[5].keys())
                     for s in UTIL_SPECS + VEC_SPECS}


NODES = (HALCYON_RampNode, HALCYON_BlurNode,
         HALCYON_ShaderNode, HALCYON_AnimeShaderNode,
         HALCYON_CartoonNode, HALCYON_ConsoleShaderNode,
         HALCYON_VolumeNode,
         HALCYON_BIMaterialNode,
         HALCYON_BIInfluenceNode, HALCYON_BIRGBBlendNode,
         HALCYON_CodeNode, HALCYON_PosterizeNode,
         HALCYON_DitherNode, HALCYON_DepthCueNode, HALCYON_ScreenInfoNode,
         HALCYON_PixelateNode, HALCYON_ScrollNode, HALCYON_ScanlinesNode,
         HALCYON_PaletteNode, HALCYON_ColorCycleNode, HALCYON_FlipbookNode,
         HALCYON_UVWaveNode, HALCYON_HalftoneNode, HALCYON_ThresholdNode,
         HALCYON_QuantizeNode,
         HALCYON_NormalMapNode, HALCYON_NormalMixNode,
         HALCYON_AltitudeSlopeNode, HALCYON_FacingNode,
         HALCYON_IridescentNode, HALCYON_SwitchNode,
         HALCYON_RandomPerObjectNode, HALCYON_LevelsNode,
         HALCYON_SmoothStepNode, HALCYON_ChannelShuffleNode,
         HALCYON_DistanceMaskNode, HALCYON_StepTimeNode,
         HALCYON_WaveNode) + UTIL_NODES + VEC_NODES
#: HalcyonBIRampStop is a PropertyGroup, not a node: it registers here,
#: BEFORE the node whose CollectionProperty points at it, and stays out
#: of the Add menu (which lists NODES only)
OPERATORS = (HALCYON_BlendValueSocket,
             HalcyonBIRampStop, HALCYON_OT_bi_ramp_stop,
             HALCYON_OT_bi_ramp_gradient,
             HALCYON_OT_compile_shader, HALCYON_OT_new_shader_text)


# ------------------------------------------------------------- the Add menu

def draw_add_menu(self, context):
    """4.0 removed nodeitems_utils, so append to the Add menu directly."""
    tree = getattr(context.space_data, 'edit_tree', None)
    if tree is None or tree.bl_idname != 'ShaderNodeTree':
        return
    if context.engine != ENGINE:
        return
    layout = self.layout
    # R242: the Halcyon menu is PREPENDED to the Add menu -- first entry,
    # not last -- so the separator follows it
    layout.menu('NODE_MT_halcyon_add', icon='SHADING_RENDERED')
    layout.separator()


# ---- R216: the Halcyon menu, by family. Every NODES member belongs to
# exactly one family below; a census test holds the union to NODES, so a
# new node cannot silently fall out of the menu.
# =============================================================== R241 tips
# Every property of every node carries a real tooltip. Most are set at
# their declarations; the table below back-fills the ones declared
# before the rule (mutating a property's keywords before registration
# is the documented road -- the annotation is a deferred declaration
# until register_class reads it). ITEM_TIPS does the same for enum
# items the declarations left empty.
PROP_TIPS = {
    ('HALCYON_ShaderNode', 'model'):
        "The reflectance model this material shades with -- each "
        "implemented from its published formulation. The menu's own "
        "entries describe every model; the sockets grey out to what "
        "the chosen model actually reads. The anime and cartoon "
        "masters, the 3ds Max shaders and the period machines live on "
        "their own nodes (Anime Shader, Cartoon Shader, the 3DS Max "
        "shelf, Console Emulation Shader)",
    ('HALCYON_ShaderNode', 'toon_steps'):
        "How many flat bands the Toon model quantizes its light into "
        "(2 is the classic cel two-tone; more approaches a smooth "
        "ramp in steps)",
    ('HALCYON_AnimeShaderNode', 'tones'):
        "How many shadow bands the cel cuts into: Two (lit plus one "
        "kage -- the TV standard) or Three (a second, deeper kage "
        "inside the first -- the OVA and feature dressing)",
    ('HALCYON_BIMaterialNode', 'diff_shader'):
        "Blender Internal's diffuse shader menu, verbatim: Lambert, "
        "Oren-Nayar, Toon, Minnaert or Fresnel, each transcribed from "
        "2.79's own code so appended materials match",
    ('HALCYON_BIMaterialNode', 'spec_shader'):
        "Blender Internal's specular shader menu, verbatim: CookTorr, "
        "Phong, Blinn, Toon or WardIso, transcribed from 2.79",
    ('HALCYON_BIMaterialNode', 'ramp_dif_input'):
        "What drives the diffuse colour ramp, as BI had it: the "
        "shader's own result, its energy, the normal, or the light "
        "term",
    ('HALCYON_BIMaterialNode', 'ramp_dif_blend'):
        "How the diffuse ramp's colour lands on the base -- BI's full "
        "blend-mode list, transcribed",
    ('HALCYON_BIMaterialNode', 'dif_stops'):
        "The diffuse colorband's stops (position, colour, alpha), "
        "edited through the ramp rows drawn in the panel",
    ('HALCYON_BIMaterialNode', 'spec_stops'):
        "The specular colorband's stops (position, colour, alpha), "
        "edited through the ramp rows drawn in the panel",
    ('HALCYON_CodeNode', 'source'):
        "The Text datablock holding the shader source -- edit it in "
        "the Text Editor and the node recompiles on every edit",
    ('HALCYON_BIMaterialNode', 'ramp_dif_factor'):
        "How strongly the diffuse ramp's colour takes over its input "
        "(BI's Factor slider, 0 off to 1 full)",
    ('HALCYON_BIMaterialNode', 'ramp_dif_ipo'):
        "The diffuse ramp's interpolation between stops: linear, "
        "ease, B-spline, cardinal or constant -- BI's own menu",
    ('HALCYON_BIMaterialNode', 'dif_ramp_tex'):
        "Bookkeeping for the gradient widget backing the diffuse "
        "ramp; the stops themselves serialize -- not a look control",
    ('HALCYON_BIMaterialNode', 'ramp_spec_input'):
        "What drives the specular colour ramp, as BI had it",
    ('HALCYON_BIMaterialNode', 'ramp_spec_blend'):
        "How the specular ramp's colour lands on the highlight -- "
        "BI's blend-mode list, transcribed",
    ('HALCYON_BIMaterialNode', 'ramp_spec_factor'):
        "How strongly the specular ramp's colour takes over its "
        "input (BI's Factor slider, 0 off to 1 full)",
    ('HALCYON_BIMaterialNode', 'ramp_spec_ipo'):
        "The specular ramp's interpolation between stops -- BI's own "
        "menu",
    ('HALCYON_BIMaterialNode', 'spec_ramp_tex'):
        "Bookkeeping for the gradient widget backing the specular "
        "ramp; the stops themselves serialize -- not a look control",
    ('HALCYON_BIMaterialNode', 'shadow_receive'):
        "Whether other objects' cast shadows land on this material "
        "(BI's Receive toggle); its own casting is the Cast toggle "
        "on the material panel",
    ('HALCYON_BIMaterialNode', 'sss_front'):
        "BI's subsurface Front weight: how much of the scatter "
        "gathered in FRONT of the surface reaches the image (Back is "
        "its through-the-surface partner)",
    ('HALCYON_BIInfluenceNode', 'blend'):
        "How this texture channel lands on what it influences -- "
        "BI's texture blend menu (Mix, Multiply, Add, ...), "
        "transcribed from 2.79",
    ('HALCYON_BIRGBBlendNode', 'blend'):
        "The blend mode this node applies between its two colours -- "
        "BI's texture blend list, so appended trees keep their look",
    ('HALCYON_BIRGBBlendNode', 'alphamix'):
        "Uses the incoming alpha as the blend factor, the way BI's "
        "texture stack did when a texture carried its own alpha",
    ('HALCYON_CodeNode', 'language'):
        "Which shading language the source pane holds: the GLSL "
        "subset or the HLSL one. Both compile to the same portable "
        "program and render identically on CPU and GPU",
    ('HALCYON_CodeNode', 'needs_rebuild'):
        "Set while the source is newer than the compiled program -- "
        "compile bookkeeping the Build button clears, not a look "
        "control",
    ('HALCYON_CodeNode', 'source_text'):
        "The name of the Text datablock holding this shader's "
        "source, when it is edited in the Text Editor instead of the "
        "node's own pane",
    ('HALCYON_CodeNode', 'error'):
        "The compiler's last error for this node, shown in the "
        "editor -- diagnostics only, cleared by a clean build",
    ('HALCYON_CodeNode', 'warn'):
        "The compiler's last warning for this node -- diagnostics "
        "only; the program still runs",
    ('HALCYON_CodeNode', 'auto_compile'):
        "Rebuilds the program automatically whenever the source "
        "changes; off waits for the Build button (steadier while "
        "typing long shaders)",
    ('HALCYON_DitherNode', 'pattern'):
        "The ordered-dither matrix: Bayer 2x2 (coarsest, boldest "
        "crosshatch), 4x4 (the VGA-era standard), 8x8 (finest), or a "
        "45-degree clustered halftone dot",
    ('HALCYON_DepthCueNode', 'mode'):
        "The fog's falloff curve between Start and End: linear, "
        "exponential, exponential squared, or a 16-step hardware "
        "fog table -- the classic depth-cue voices",
    ('HALCYON_ScrollNode', 'animate'):
        "Scrolls with the scene clock (UV units per second) instead "
        "of holding the Scroll X / Scroll Y offsets still",
    ('HALCYON_PaletteNode', 'palette'):
        "The classic hardware palette to snap colours into: EGA 16, "
        "the VGA 256 default, Game Boy greens, CGA modes, "
        "greyscale, or a custom count",
    ('HALCYON_ColorCycleNode', 'animate'):
        "Cycles the palette with the scene clock -- the demoscene "
        "waterfall trick -- instead of holding the Phase still",
    ('HALCYON_FlipbookNode', 'animate'):
        "Steps through the sprite sheet's frames with the scene "
        "clock at the given rate; off shows the Frame input's cell",
    ('HALCYON_UVWaveNode', 'animate'):
        "Runs the wave's phase with the scene clock, so the warp "
        "rolls on its own; off holds the Phase input's moment",
    ('HALCYON_NormalMapNode', 'space'):
        "Which space the map's vectors live in: Tangent (the "
        "ordinary baked map, needs UVs), Object, or World",
    ('HALCYON_NormalMapNode', 'map_type'):
        "The map's convention: OpenGL (green up, Blender's own) or "
        "DirectX (green down -- flip for maps baked elsewhere)",
    ('HALCYON_NormalMixNode', 'mode'):
        "How the two normal maps combine: a simple lerp, a whiteout "
        "blend (keeps both sets of detail), or reoriented (the "
        "detail map ridden on the base's frame -- the correct one)",
    ('HALCYON_SmoothStepNode', 'interp'):
        "The step's easing: hard threshold, smoothstep, or the "
        "flatter smootherstep",
    ('HALCYON_ChannelShuffleNode', 'out_r'):
        "Which input channel lands in the output's RED -- rewire "
        "packed textures without a chain of separates",
    ('HALCYON_ChannelShuffleNode', 'out_g'):
        "Which input channel lands in the output's GREEN",
    ('HALCYON_ChannelShuffleNode', 'out_b'):
        "Which input channel lands in the output's BLUE",
    ('HALCYON_ChannelShuffleNode', 'out_a'):
        "Which input channel lands in the output's ALPHA",
    ('HALCYON_WaveNode', 'wave'):
        "The wave's profile: sine, triangle, square or sawtooth -- "
        "the classic procedural stripes' four voices",
    ('HALCYON_LightMeterNode', 'shadows'):
        "Whether the meter reads the lamps' cast shadows into its "
        "value, or the raw unshadowed light term",
    ('HALCYON_OscillatorNode', 'wave'):
        "The oscillator's profile over time: sine, triangle, square "
        "or sawtooth -- drive anything that should pulse or blink",
    ('HALCYON_GateNode', 'invert'):
        "Flips the gate: passes when the control is BELOW the "
        "threshold instead of above it",
    ('HALCYON_SelectorNode', 'wrap'):
        "Wraps the index around the input count instead of clamping "
        "at the ends -- a counter cycles through the inputs forever",
    ('HALCYON_MeasureNode', 'mode'):
        "Which distance the node measures: from the object's "
        "origin, from the Point input, from the camera, or the "
        "coordinate along one world axis",
    ('HALCYON_WobbleNode', 'seed'):
        "Picks a different random wobble path; the same seed always "
        "replays the same wander",
    ('HALCYON_ArrayVecNode', 'mode'):
        "How the copies are laid out: a line, a grid, a ring, or a "
        "regular polygon's corners",
    ('HALCYON_ArrayVecNode', 'count'):
        "How many copies the layout places along its line, grid row, "
        "ring or polygon",
    ('HALCYON_ArrayVecNode', 'sides'):
        "The polygon layout's corner count (3 a triangle, 6 a "
        "hexagon ring)",
    ('HALCYON_ArrayVecNode', 'orient'):
        "Turns each copy to face along the layout (around the ring, "
        "along the line) instead of keeping them all upright",
    ('HALCYON_MirrorTileNode', 'axis_x'):
        "Mirrors every second tile horizontally, so patterns meet "
        "their reflections seamlessly across tile edges",
    ('HALCYON_MirrorTileNode', 'axis_y'):
        "Mirrors every second tile vertically, so patterns meet "
        "their reflections seamlessly across tile edges",
    ('HALCYON_PolarNode', 'direction'):
        "Which way the node converts: rectangular coordinates to "
        "polar (rings and sweeps), or polar back to rectangular",
    ('HALCYON_RippleWarpNode', 'animate'):
        "Rolls the ripples outward with the scene clock; off holds "
        "the Phase input's instant",
    ('HALCYON_WaveWarpNode', 'axis'):
        "Which direction the warp displaces: along X (shearing the "
        "rows sideways) or along Y (shearing the columns)",
    ('HALCYON_WaveWarpNode', 'animate'):
        "Runs the warp's phase with the scene clock; off holds the "
        "Phase input's moment",
    ('HALCYON_TileRandomNode', 'rotate'):
        "Gives each tile a random quarter-turn, hiding the repeat in "
        "tiled textures",
    ('HALCYON_TileRandomNode', 'flip'):
        "Gives each tile a random mirror flip, hiding the repeat in "
        "tiled textures",
    ('HALCYON_VectorSnapNode', 'mode'):
        "Snap down to the cell's corner (floor) or to the nearest "
        "grid point (round) -- pixel-grid versus centred quantizing",
    ('HALCYON_OrbitNode', 'animate'):
        "Circles the point with the scene clock at the given speed; "
        "off holds the Angle input's position",
    ('HALCYON_RegionNode', 'outside'):
        "What happens beyond the box: clip it away (Inside goes "
        "0), wrap around, mirror back, or extend the edge",
    ('HALCYON_ProjectorNode', 'mode'):
        "The projection that makes the coordinates: flat planar, "
        "cylinder, sphere, or box (the three planes picked by the "
        "normal)",
    ('HALCYON_ProjectorNode', 'axis'):
        "The projection's axis: which way the plane faces, the "
        "cylinder stands, or the box's dominant plane is chosen",
    ('HALCYON_SpinNode', 'animate'):
        "Spins with the scene clock at the given rate; off holds the "
        "Angle input's turn",
}

ITEM_TIPS = {
    ('HALCYON_AnimeShaderNode', 'rim_blend'): {
        'ADD': "Added on top of the banded result -- the light "
               "wrapping past the silhouette"},
    ('HALCYON_AnimeShaderNode', 'matcap_mode'): {
        'MIX': "Blends the capture toward the surface by the Matcap "
               "Blend amount"},
    ('HALCYON_CartoonNode', 'rim_blend'): {
        'ADD': "Added on top of the paint -- the light wrapping past "
               "the silhouette"},
    ('HALCYON_DitherNode', 'pattern'): {
        'BAYER2': "The 2x2 ordered matrix: four levels, the boldest "
                  "crosshatch texture",
        'BAYER4': "The 4x4 ordered matrix: sixteen levels, the "
                  "VGA-era standard look",
        'BAYER8': "The 8x8 ordered matrix: the finest ordered grain",
        'HALFTONE': "A clustered dot at the classic 45 degrees -- "
                    "the newspaper's screen rather than a matrix"},
    ('HALCYON_DepthCueNode', 'mode'): {
        'LINEAR': "Fog thickens evenly from Start to End -- the "
                  "SGI-era hardware default",
        'EXP': "Exponential falloff: fast at first, easing with "
               "distance -- natural atmospheric haze",
        'EXP2': "Exponential squared: clear near the camera, then "
                "closing in hard -- the heaviest classic curve",
        'TABLE16': "The blend quantised to 16 hardware fog bands "
                   "-- visible stepping, as console fog tables did"},
    ('HALCYON_PaletteNode', 'palette'): {
        'CUSTOM': "Quantize each channel into the given number of "
                  "levels instead of a named palette",
        'GREY4': "Four greys -- the original Game Boy class of "
                 "display, without the green cast"},
    ('HALCYON_SmoothStepNode', 'interp'): {
        'HARD': "A hard threshold at the edge -- no blend at all"},
    ('HALCYON_ChannelShuffleNode', 'out_r'): {
        'R': "Take the input's red channel", 'G': "Take the input's "
        "green channel", 'B': "Take the input's blue channel",
        'A': "Take the input's alpha channel", 'ONE': "A constant "
        "1.0 in this channel", 'ZERO': "A constant 0.0 in this "
        "channel"},
    ('HALCYON_WaveNode', 'wave'): {
        'SAW': "The sawtooth: a linear ramp that snaps back -- "
               "conveyor stripes and scan ramps"},
    ('HALCYON_OscillatorNode', 'wave'): {
        'SINE': "The smooth sine pulse -- breathing glows",
        'TRIANGLE': "A linear rise and fall -- even ramps both ways",
        'SQUARE': "On-off switching at the rate -- blinkers and "
                  "beacons",
        'SAW': "A linear ramp that snaps back each cycle -- "
               "counters and sweeps"},
    ('HALCYON_MeasureNode', 'mode'): {
        'ORIGIN': "Distance from the object's origin, over Scale "
                  "-- radial gradients growing from the pivot",
        'POINT': "Distance from the Point input, over Scale -- "
                 "blast rings and glow falloffs placed anywhere",
        'CAMERA': "Distance from the camera, over Scale -- fades "
                  "and cue masks that follow the view",
        'AXIS_X': "The world X coordinate over Scale -- a flat "
                  "gradient along X",
        'AXIS_Y': "The world Y coordinate over Scale -- a flat "
                  "gradient along Y",
        'AXIS_Z': "The world height over Scale -- altitude bands "
                  "and waterlines"},
    ('HALCYON_ArrayVecNode', 'mode'): {
        'LINE': "Copies along a straight line at even steps"},
    ('HALCYON_PolarNode', 'direction'): {
        'TO_POLAR': "UV becomes (angle around the centre, distance "
                    "from it) -- rings, sweeps, radar",
        'FROM_POLAR': "(angle, distance) becomes UV again -- "
                      "unwrap a ring back to a strip"},
    ('HALCYON_WaveWarpNode', 'axis'): {
        'X': "Displace along X, shearing the rows sideways",
        'Y': "Displace along Y, shearing the columns up and down"},
    ('HALCYON_VectorSnapNode', 'mode'): {
        'FLOOR': "Snap down to the cell's corner -- the pixel-grid "
                 "convention",
        'ROUND': "Round to the nearest step -- cells centre on the "
                 "grid lines rather than between them"},
    ('HALCYON_RegionNode', 'outside'): {
        'CLIP': "Outside the box the Inside mask reads 0 -- "
                "multiply your alpha by it to place a decal once",
        'WRAP': "Repeat the region endlessly beyond its edges",
        'MIRROR': "Reflect the region back and forth seamlessly",
        'EXTEND': "Hold the edge value beyond the box -- the "
                  "border pixels stretch outward"},
    ('HALCYON_ProjectorNode', 'mode'): {
        'PLANAR': "Project flat from one direction -- decals and "
                  "screens",
        'CYLINDER': "Wrap around the axis -- cans, columns, tree "
                    "trunks",
        'SPHERE': "Wrap around a sphere -- planets and domes",
        'BOX': "Three planar projections picked by the surface "
               "normal -- quick clean mapping without UVs"},
    ('HALCYON_ProjectorNode', 'axis'): {
        'X': "The projection faces / stands along X",
        'Y': "The projection faces / stands along Y",
        'Z': "The projection faces / stands along Z"},
}

# The shuffle's four channels share one item set; so do their docs.
for _ch in ('out_g', 'out_b', 'out_a'):
    ITEM_TIPS[('HALCYON_ChannelShuffleNode', _ch)] = \
        ITEM_TIPS[('HALCYON_ChannelShuffleNode', 'out_r')]
del _ch


def _apply_prop_tips():
    """R241: back-fill the tables above into the deferred property
    declarations, before registration reads them. A property whose
    declaration already carries a real description (40 characters or
    more) keeps its own words; empty or stub descriptions take the
    table's. Enum items with thin docs take ITEM_TIPS' by
    identifier."""
    for cls in NODES:
        idn = getattr(cls, 'bl_idname', '')
        for pname, prop in getattr(cls, '__annotations__', {}).items():
            kw = getattr(prop, 'keywords', None)
            if not isinstance(kw, dict):
                kw = getattr(prop, 'kw', None)
            if not isinstance(kw, dict):
                continue
            tip = PROP_TIPS.get((idn, pname))
            if tip and len(str(kw.get('description') or '')) < 40:
                kw['description'] = tip
            fills = ITEM_TIPS.get((idn, pname))
            items = kw.get('items')
            if fills and items and not callable(items):
                kw['items'] = tuple(
                    (i[0], i[1],
                     (i[2] if len(i) > 2 and len(str(i[2])) >= 20
                      else fills.get(i[0], str(i[2]) if len(i) > 2
                                     else '')))
                    for i in items)


_apply_prop_tips()


MENU_FAMILIES = (
    ('Shading', 'MATERIAL',
     (HALCYON_ShaderNode, HALCYON_AnimeShaderNode, HALCYON_CartoonNode,
      HALCYON_ConsoleShaderNode,
      HALCYON_VolumeNode,
      HALCYON_RampNode,
      HALCYON_CodeNode, HALCYON_FacingNode, HALCYON_IridescentNode)),
    ('Blender Internal', 'NODE_MATERIAL',
     (HALCYON_BIMaterialNode, HALCYON_BIInfluenceNode,
      HALCYON_BIRGBBlendNode,
      # R242: the BI texture lives with the BI shading nodes
      ('HALCYON_BITextureNode', "BI Texture", 'TEXTURE'))),
    ('Utilities', 'TOOL_SETTINGS',
     (HALCYON_SwitchNode, HALCYON_RandomPerObjectNode,
      HALCYON_LevelsNode, HALCYON_SmoothStepNode,
      HALCYON_ChannelShuffleNode, HALCYON_DistanceMaskNode,
      HALCYON_StepTimeNode, HALCYON_WaveNode, HALCYON_ThresholdNode,
      HALCYON_FlipbookNode, HALCYON_AltitudeSlopeNode,
      HALCYON_BlurNode) + UTIL_NODES),
    ('Vector', 'ORIENTATION_NORMAL',
     (HALCYON_NormalMapNode, HALCYON_NormalMixNode,
      HALCYON_UVWaveNode, HALCYON_ScrollNode) + VEC_NODES
     # R242: the matcap coordinates are a vector, not a texture
     + (('HALCYON_MatcapUVNode', "Matcap Coordinates", 'MATSPHERE'),)),
    ('Retro Screen', 'RENDER_STILL',
     (HALCYON_PosterizeNode, HALCYON_DitherNode, HALCYON_DepthCueNode,
      HALCYON_ScreenInfoNode, HALCYON_PixelateNode,
      HALCYON_ScanlinesNode, HALCYON_PaletteNode,
      HALCYON_ColorCycleNode, HALCYON_HalftoneNode,
      HALCYON_QuantizeNode)),
)


#: R242: a family may list a node another module owns, by
#: (bl_idname, label, icon) -- the census test resolves each reference
CROSS_FAMILY = {'HALCYON_MatcapUVNode': 'Vector',
                'HALCYON_BITextureNode': 'Blender Internal'}


def _family_menu(title, members):
    def draw(self, context):
        layout = self.layout
        for cls in members:
            if isinstance(cls, tuple):
                idname, label, icon = cls
            else:
                idname, label = cls.bl_idname, cls.bl_label
                icon = getattr(cls, 'bl_icon', 'NONE')
            op = layout.operator('node.add_node', text=label, icon=icon)
            op.type = idname
            op.use_transform = True
    ident = 'NODE_MT_halcyon_' + title.lower().replace(' ', '_')
    return type(ident, (bpy.types.Menu,), {
        'bl_idname': ident, 'bl_label': title, 'draw': draw})


MENU_SUBMENUS = tuple(_family_menu(t, m) for t, _i, m in MENU_FAMILIES)


class NODE_MT_halcyon_add(bpy.types.Menu):
    bl_idname = 'NODE_MT_halcyon_add'
    bl_label = "Halcyon"

    def draw(self, context):
        layout = self.layout
        # R203: the Pre-Made shelf lives at the top of the Halcyon
        # menu, where the field asked for it
        try:
            layout.menu('NODE_MT_halcyon_premade', icon='PRESET')
            layout.separator()
        except Exception:                                       # noqa: BLE001
            pass
        for (title, icon, _members), sub in zip(MENU_FAMILIES,
                                                MENU_SUBMENUS):
            layout.menu(sub.bl_idname, icon=icon)
        layout.separator()
        # R242: the 3DS Max shelf -- its materials, textures, utilities
        # and vectors under one entry
        layout.menu('NODE_MT_halcyon_max', icon='MESH_CUBE')
        layout.separator()
        layout.menu('NODE_MT_halcyon_textures', icon='TEXTURE')


_menu_owner = None


#: R252: the Max shader type the Max Standard node takes for each moved
#: Max model (nodes/max_nodes.MAX_MODEL_FOR, inverted)
_MAX_TYPE_FOR = {'MAX_ANISOTROPIC': 'ANISOTROPIC', 'MAX_BLINN': 'BLINN',
                 'MAX_METAL': 'METAL', 'MAX_MULTI_LAYER': 'MULTI_LAYER',
                 'MAX_OREN_NAYAR_BLINN': 'OREN_NAYAR_BLINN',
                 'MAX_PHONG': 'PHONG', 'MAX_STRAUSS': 'STRAUSS',
                 'MAX_TRANSLUCENT': 'TRANSLUCENT'}


def saved_master_model(node):
    """R252: the model a saved master node carries, by name -- the live
    property when its value is still on the menu, else the raw enum
    integer read back through the engine table (an EnumProperty without
    explicit numbers stored the item's INDEX, and MODEL_ITEMS is that
    order, never reordered). None when nothing is stored."""
    try:
        live = str(getattr(node, 'model', '') or '')
    except Exception:                                           # noqa: BLE001
        live = ''
    if live and live in MASTER_MODELS:
        return live
    if live and live in MOVED_MODELS:
        return live
    raw = None
    try:
        raw = node.get('model')
    except Exception:                                           # noqa: BLE001
        raw = None
    if isinstance(raw, int) and 0 <= raw < len(MODEL_ITEMS):
        return MODEL_ITEMS[raw][0]
    if isinstance(raw, str) and raw:
        return raw
    return None


def migrate_master_node(tree, node):
    """R252: rebuild one master node saved with a model the master no
    longer offers as the node that carries it now -- the Anime Shader,
    the Cartoon Shader, the Max Standard material (its shader type set)
    or the Console Emulation Shader (its machine, type and options set
    from core/console.MIGRATE). Links and socket values travel by socket
    name; the output relinks to whatever the old Surface fed. Returns
    the new node, or None when the node needs no migration. Best-effort
    and load-time only, like every other topology change."""
    model = saved_master_model(node)
    if model is None or model in MASTER_MODELS:
        return None
    target = MOVED_MODELS.get(model)
    if target is None:
        return None
    new = tree.nodes.new(target)
    try:
        new.location = node.location
        new.label = node.label
        new.hide = node.hide
    except Exception:                                           # noqa: BLE001
        pass
    if target == 'HALCYON_MaxStandardNode':
        try:
            new.shader_type = _MAX_TYPE_FOR.get(model, 'BLINN')
        except Exception:                                       # noqa: BLE001
            pass
    elif target == 'HALCYON_ConsoleShaderNode':
        for k, v in CON.MIGRATE.get(model, {}).items():
            try:
                setattr(new, k, v)
            except Exception:                                   # noqa: BLE001
                pass
        try:
            new.toon_steps = int(getattr(node, 'toon_steps', 2) or 2)
        except Exception:                                       # noqa: BLE001
            pass
    # the sockets: values and links by name
    for old_s in list(node.inputs):
        ns = new.inputs.get(old_s.name)
        if ns is None:
            continue
        try:
            if not old_s.is_linked and hasattr(old_s, 'default_value'):
                ns.default_value = old_s.default_value
        except Exception:                                       # noqa: BLE001
            pass
        for ln in list(getattr(old_s, 'links', ())):
            try:
                tree.links.new(ln.from_socket, ns)
            except Exception:                                   # noqa: BLE001
                pass
    for old_o in list(node.outputs):
        for ln in list(getattr(old_o, 'links', ())):
            try:
                tree.links.new(new.outputs[0], ln.to_socket)
            except Exception:                                   # noqa: BLE001
                pass
    try:
        if callable(getattr(new, 'refresh_sockets', None)):
            new.refresh_sockets()
    except Exception:                                           # noqa: BLE001
        pass
    try:
        tree.nodes.remove(node)
    except Exception:                                           # noqa: BLE001
        pass
    return new


def _migrate_master_sockets(_arg=None):
    """load_post: saved master nodes gain any socket their file predates.

    Socket creation is forbidden inside update callbacks (the guard test
    holds refresh_sockets to toggles only); file load is the one place
    topology change is safe, so it happens here -- Bump Height appears on
    old files the moment they open. R252: a master node saved with a
    model the master no longer offers is rebuilt here as the node that
    carries it (migrate_master_node), before the socket pass.
    """
    try:
        trees = []
        for mat in bpy.data.materials:
            if getattr(mat, 'use_nodes', False) and mat.node_tree:
                trees.append(mat.node_tree)
        for grp in getattr(bpy.data, 'node_groups', []):
            trees.append(grp)
        for tree in trees:
            for node in list(getattr(tree, 'nodes', [])):
                if getattr(node, 'bl_idname', '') == 'HALCYON_ShaderNode':
                    try:
                        migrate_master_node(tree, node)
                    except Exception:                           # noqa: BLE001
                        pass
            for node in getattr(tree, 'nodes', []):
                idn = getattr(node, 'bl_idname', '')
                if idn in ('HALCYON_ShaderNode', 'HALCYON_BIMaterialNode',
                           # R229: the anime master grew sockets
                           'HALCYON_AnimeShaderNode',
                           'HALCYON_AltitudeSlopeNode') or (
                        # R235: the media nodes grew Indication / Blend
                        idn.startswith('HALCYON_')
                        and callable(getattr(node, 'ensure_sockets', None))):
                    try:
                        node.ensure_sockets()
                    except Exception:                           # noqa: BLE001
                        pass
                if idn == 'HALCYON_ShaderNode':
                    # R202: the three blend menus move down beside
                    # their sliders, and the panel regroups -- both
                    # only ever at load, where topology is safe
                    try:
                        node.upgrade_blend_sockets()
                        node.sort_sockets()
                    except Exception:                           # noqa: BLE001
                        pass
    except Exception:                                           # noqa: BLE001
        pass


def register():
    global _menu_owner
    from . import pattern_nodes
    pattern_nodes.register()
    from . import max_nodes
    max_nodes.register()
    for sock_cls in (NodeSocket,):
        if not hasattr(sock_cls, 'halcyon_uniform'):
            sock_cls.halcyon_uniform = StringProperty(default='')
            sock_cls.halcyon_key = StringProperty(default='')
            sock_cls.halcyon_image_key = StringProperty(default='')
            sock_cls.halcyon_is_image = BoolProperty(default=False)
    for cls in OPERATORS + NODES:
        bpy.utils.register_class(cls)
    for _sub in MENU_SUBMENUS:
        bpy.utils.register_class(_sub)
    bpy.utils.register_class(NODE_MT_halcyon_add)
    from .. import compat
    _menu_owner = compat.register_node_menu(draw_add_menu)
    try:
        hs = bpy.app.handlers.load_post
        if _migrate_master_sockets not in hs:
            hs.append(_migrate_master_sockets)
        _migrate_master_sockets()          # the already-open file too
    except Exception:                                           # noqa: BLE001
        pass


def unregister():
    try:
        hs = bpy.app.handlers.load_post
        while _migrate_master_sockets in hs:
            hs.remove(_migrate_master_sockets)
    except Exception:                                           # noqa: BLE001
        pass
    from . import max_nodes
    max_nodes.unregister()
    from . import pattern_nodes
    pattern_nodes.unregister()
    from .. import compat
    compat.unregister_node_menu(draw_add_menu, _menu_owner)
    for cls in (NODE_MT_halcyon_add,) + tuple(MENU_SUBMENUS) \
            + tuple(reversed(NODES + OPERATORS)):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:                                       # noqa: BLE001
            pass


# ---- R251 material pack, wave 2 (MAT-B): period nodes ----
# Five nodes of the period-combiner pack: the GameCube / Xbox fixed-point
# combiner stage (C028), the PowerVR2 (S,R) bump (C023), the DirectX 6
# emboss (C135), Imagine's Roughness (C099) and Alias / Maya's Env Chrome
# (C123). Each is one class + a socket-doc table; the section's tail
# registers them in NODES, MENU_FAMILIES (rebuilt, MENU_SUBMENUS with it)
# and PERIOD_NODE_PROPS -> FAMILY_NODE_PROPS (the exporter's road).

_CB_HARDWARE = [
    ('TEV', "TEV (GameCube / Wii, 2001)",
     "8-bit inputs, S10 intermediates, GX's bias / scale / clamp"),
    ('NV2A', "Register combiner (Xbox NV2A, 2001)",
     "9-bit signed A*B + C*D with the NV input mappings"),
]
_CB_TEV_OPS = [
    ('ADD', "Add", "GX_TEV_ADD: d + lerp(a, b, c) + bias, then scale"),
    ('SUB', "Subtract", "GX_TEV_SUB: d - lerp(a, b, c) + bias, then scale"),
    ('COMP_R8_GT', "Compare R8 >", "GX_TEV_COMP_R8_GT: d + (a.r > b.r ? c : 0)"),
    ('COMP_R8_EQ', "Compare R8 =", "GX_TEV_COMP_R8_EQ: d + (a.r == b.r ? c : 0)"),
    ('COMP_GR16_GT', "Compare GR16 >",
     "GX_TEV_COMP_GR16_GT: d + (a.gr > b.gr ? c : 0) on the 16-bit packed pair"),
    ('COMP_GR16_EQ', "Compare GR16 =",
     "GX_TEV_COMP_GR16_EQ: d + (a.gr == b.gr ? c : 0) on the 16-bit packed pair"),
    ('COMP_BGR24_GT', "Compare BGR24 >",
     "GX_TEV_COMP_BGR24_GT: d + (a.bgr > b.bgr ? c : 0) on the 24-bit packed triple"),
    ('COMP_BGR24_EQ', "Compare BGR24 =",
     "GX_TEV_COMP_BGR24_EQ: d + (a.bgr == b.bgr ? c : 0) on the 24-bit packed triple"),
    ('COMP_RGB8_GT', "Compare RGB8 >",
     "GX_TEV_COMP_RGB8_GT: per channel a > b ? c : 0, plus d"),
    ('COMP_RGB8_EQ', "Compare RGB8 =",
     "GX_TEV_COMP_RGB8_EQ: per channel a == b ? c : 0, plus d"),
]
_CB_TEV_BIAS = [
    ('ZERO', "Zero", "GX_TB_ZERO: no bias on the register"),
    ('ADD_HALF', "+0.5", "GX_TB_ADDHALF: +128 on the S10 register"),
    ('SUB_HALF', "-0.5", "GX_TB_SUBHALF: -128 on the S10 register"),
]
_CB_TEV_SCALE = [
    ('X1', "x1", "GX_CS_SCALE_1: the register as it is"),
    ('X2', "x2", "GX_CS_SCALE_2: the register doubled"),
    ('X4', "x4", "GX_CS_SCALE_4: the register times four"),
    ('HALF', "/2", "GX_CS_DIVIDE_2: an arithmetic shift right (floor on the signed S10)"),
]
_CB_NV_MAPS = [
    ('UNSIGNED_IDENTITY', "Unsigned identity", "max(0, e): the input as it is, negatives clamped"),
    ('UNSIGNED_INVERT', "Unsigned invert", "1 - min(max(e, 0), 1): the input inverted"),
    ('EXPAND_NORMAL', "Expand normal", "2 * max(0, e) - 1: 0..1 stretched to -1..1"),
    ('EXPAND_NEGATE', "Expand negate", "-(2 * max(0, e) - 1): the expansion negated"),
    ('HALF_BIAS_NORMAL', "Half-bias normal", "max(0, e) - 0.5: the input biased down"),
    ('HALF_BIAS_NEGATE', "Half-bias negate", "-(max(0, e) - 0.5): the biased input negated"),
    ('SIGNED_IDENTITY', "Signed identity", "e: the signed input as it is"),
    ('SIGNED_NEGATE', "Signed negate", "-e: the signed input negated"),
]
_CB_NV_SCALE = [
    ('X1', "x1", "NV_NONE: the sum as it is"),
    ('X2', "x2", "NV_SCALE_BY_TWO_NV: the sum doubled"),
    ('X4', "x4", "NV_SCALE_BY_FOUR_NV: the sum times four"),
    ('HALF', "/2", "NV_SCALE_BY_ONE_HALF_NV: floor on the 9-bit signed sum"),
]
_CB_NV_BIAS = [
    ('NONE', "None", "NV_NONE: no bias on the sum"),
    ('MINUS_HALF', "-0.5", "NV_BIAS_BY_NEGATIVE_ONE_HALF_NV: -128 on the 9-bit sum"),
]

COMBINER_SOCKET_DOCS = {
    'A': "First input: 8-bit (TEV) or 9-bit signed (NV2A) fixed point; a "
         "texture, the rasterised colour, a register or a constant",
    'B': "Second input on the same fixed-point grid; TEV's lerp end, the "
         "NV2A's right factor of the first product A*B",
    'C': "The lerp weight in TEV: 255 maps to 256 exactly, the hardware's "
         "c9 = c8 + (c8 >> 7); the NV2A's left factor of C*D",
    'D': "TEV: the signed 10-bit register the lerp adds to; NV2A: the second "
         "product's right factor. Its signed range (-4.02..4.01 unclamped) "
         "is reachable only through a LINK from a previous stage: a colour "
         "socket's own default is clamped to 0..1",
}


class HALCYON_CombinerStageNode(Node, HalcyonNodeBase):
    """R251 C028: one fixed-point combiner stage -- the GameCube / Wii
    TEV (8-bit inputs, signed 10-bit register, bias / scale / clamp and
    the compare ops) or the Xbox NV2A register combiner (9-bit signed
    A*B + C*D with the eight input mappings). Integer arithmetic on both
    devices, so the posterisation and sign-clamping of 2001 multitexture
    appear by construction. Chain stages by linking Color into D."""

    bl_idname = 'HALCYON_CombinerStageNode'
    bl_label = "Combiner Stage"
    bl_icon = 'NODE_COMPOSITING'

    hardware: EnumProperty(
        name="Hardware", items=_CB_HARDWARE, default='TEV',
        description="Which machine's combiner arithmetic this stage runs: "
                    "GX's TEV (8-bit, S10 register) or the NV2A register "
                    "combiner (9-bit signed products)")
    op: EnumProperty(
        name="Op", items=_CB_TEV_OPS, default='ADD',
        description="TEV: the stage operation -- add or subtract the lerp, "
                    "or one of GX's compare ops on packed 8 / 16 / 24-bit "
                    "values selecting C or zero")
    bias: EnumProperty(
        name="Bias", items=_CB_TEV_BIAS, default='ZERO',
        description="TEV: GX's bias on the S10 register for ADD / SUB "
                    "(the compare ops ignore bias and scale, as GX requires)")
    scale: EnumProperty(
        name="Scale", items=_CB_TEV_SCALE, default='X1',
        description="TEV: GX's output scale on the S10 register -- x1, x2, "
                    "x4 or an arithmetic shift right (floor division by 2)")
    clamp: BoolProperty(
        name="Clamp", default=True,
        description="GX_TRUE: clamp the register to 0..255; off keeps the "
                    "signed 10-bit range (-1024..1023) for the next stage")
    map_a: EnumProperty(
        name="Map A", items=_CB_NV_MAPS, default='UNSIGNED_IDENTITY',
        description="NV2A: the NV_register_combiners input mapping applied "
                    "to A before the 9-bit quantisation")
    map_b: EnumProperty(
        name="Map B", items=_CB_NV_MAPS, default='UNSIGNED_IDENTITY',
        description="NV2A: the NV_register_combiners input mapping applied "
                    "to B before the 9-bit quantisation")
    map_c: EnumProperty(
        name="Map C", items=_CB_NV_MAPS, default='UNSIGNED_IDENTITY',
        description="NV2A: the NV_register_combiners input mapping applied "
                    "to C before the 9-bit quantisation")
    map_d: EnumProperty(
        name="Map D", items=_CB_NV_MAPS, default='UNSIGNED_IDENTITY',
        description="NV2A: the NV_register_combiners input mapping applied "
                    "to D before the 9-bit quantisation")
    nv_scale: EnumProperty(
        name="NV Scale", items=_CB_NV_SCALE, default='X1',
        description="NV2A: the combiner output scale on the 9-bit signed "
                    "sum A*B + C*D (x1, x2, x4, or floor /2)")
    nv_bias: EnumProperty(
        name="NV Bias", items=_CB_NV_BIAS, default='NONE',
        description="NV2A: the combiner output bias on the 9-bit signed "
                    "sum -- none, or -0.5 (-128) before the clamp")

    SOCKETS = (('NodeSocketColor', 'A', (0.0, 0.0, 0.0, 1.0)),
               ('NodeSocketColor', 'B', (1.0, 1.0, 1.0, 1.0)),
               ('NodeSocketColor', 'C', (0.0, 0.0, 0.0, 1.0)),
               ('NodeSocketColor', 'D', (0.0, 0.0, 0.0, 1.0)))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            try:
                s.default_value = default
            except (TypeError, ValueError):
                pass
        _apply_socket_tips(self, COMBINER_SOCKET_DOCS)
        self.outputs.new('NodeSocketColor', 'Color')

    def ensure_sockets(self):
        _apply_socket_tips(self, COMBINER_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'hardware', text="")
        if self.hardware == 'NV2A':
            col = layout.column(align=True)
            col.prop(self, 'map_a')
            col.prop(self, 'map_b')
            col.prop(self, 'map_c')
            col.prop(self, 'map_d')
            row = layout.row(align=True)
            row.prop(self, 'nv_scale', text="")
            row.prop(self, 'nv_bias', text="")
        else:
            layout.prop(self, 'op', text="")
            row = layout.row(align=True)
            row.prop(self, 'bias', text="")
            row.prop(self, 'scale', text="")
            layout.prop(self, 'clamp')


_SR_BLEND = [
    ('MULTIPLY', "Multiply (DECAL x intensity)",
     "The DC's second pass: the base texture times the bump intensity"),
    ('ADD', "Add",
     "The intensity added over the base -- the era's other second pass"),
]

SRBUMP_SOCKET_DOCS = {
    'Color': "The (S,R) source: a tangent-space normal map texel, quantised "
             "to the PVR2's two 8-bit angles (elevation S, azimuth R) through "
             "baked tables",
    'Light': "The lamp's direction in the texture's tangent frame (t, b, n); "
             "one direction per polygon on the PVR2, so keep it unlinked for "
             "the GPU",
    'Strength': "H, the bump strength: K1 = 1 - H is the ambient floor, K2 = "
                "sin T * H and K3 = cos T * H the light's split by elevation T",
    'Base': "The DECAL texture the intensity multiplies or adds over",
}


class HALCYON_SRBumpNode(Node, HalcyonNodeBase):
    """R251 C023: PowerVR2 (S,R) bump mapping (Dreamcast CLX2, Naomi). A
    bump texel is two 8-bit angles; one directional light per polygon,
    no view vector, 256 azimuth steps: `I = clamp(K1 + K2 sin S + K3 cos
    S cos(R - Q), 0, 1)`, then a multiply or add pass over the base
    texture. Bitwise on both devices under NEAREST (baked tables)."""

    bl_idname = 'HALCYON_SRBumpNode'
    bl_label = "SR Bump (Dreamcast)"
    bl_icon = 'MOD_DISPLACE'

    blend: EnumProperty(
        name="Blend", items=_SR_BLEND, default='MULTIPLY',
        description="The PVR2's second pass over the base texture: the "
                    "intensity multiplies the DECAL, or is added over it")

    SOCKETS = (('NodeSocketColor', 'Color', (0.5, 0.5, 1.0, 1.0)),
               ('NodeSocketVector', 'Light', (0.0, 0.0, 1.0)),
               ('NodeSocketFloat', 'Strength', 0.0),
               ('NodeSocketColor', 'Base', (0.8, 0.8, 0.8, 1.0)))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            try:
                s.default_value = default
            except (TypeError, ValueError):
                pass
            if name == 'Strength':
                try:
                    s.min_value, s.max_value = 0.0, 1.0
                except (AttributeError, TypeError):
                    pass
        _apply_socket_tips(self, SRBUMP_SOCKET_DOCS)
        self.outputs.new('NodeSocketFloat', 'Intensity')
        self.outputs.new('NodeSocketColor', 'Color')

    def ensure_sockets(self):
        _apply_socket_tips(self, SRBUMP_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'blend', text="")


EMBOSS_SHIFT_SOCKET_DOCS = {
    'Vector': "The height map's UV (unlinked: the UV map); the second stage "
              "samples the SAME map at this UV shifted toward the light",
    'Light': "The lamp direction in the tangent frame; only its (t, b) part "
             "shifts the second sample -- unnormalised, as the DX6 pipe "
             "interpolated it",
    'Offset': "The shift toward the light in texels (about one texel; larger "
              "shifts invert the relief, the era's over-shift artefact)",
    'Texture Size': "The height map's width in texels: the shift is Offset / "
                    "Texture Size in UV, so this must match the image",
}

EMBOSS_SOCKET_DOCS = {
    'Height': "The height map sampled at the UV (scale the map into 0..0.5 "
              "as the era did, or accept the clamp)",
    'Height Shifted': "The SAME height map sampled at the Shifted UV of the "
                      "Emboss Shift node -- the second texture stage",
    'Base': "The base texture, applied modulate-2x over the emboss",
}


class HALCYON_EmbossShiftNode(Node, HalcyonNodeBase):
    """R251 C135, stage one of DirectX 6 texture embossing: the second
    texture stage's coordinate set -- the UV shifted toward the light by
    about one texel in the tangent plane (`uv + Offset * Light.xy /
    Texture Size`). Wire it into a second Image Texture of the SAME
    height map and hand both samples to the Emboss Bump node; a node
    cannot feed its own input (a cycle), which is why the two stages
    are two nodes, exactly the DX6 multitexture setup."""

    bl_idname = 'HALCYON_EmbossShiftNode'
    bl_label = "Emboss Shift (DirectX 6)"
    bl_icon = 'SHADING_TEXTURE'

    SOCKETS = (('NodeSocketVector', 'Vector', (0.0, 0.0, 0.0)),
               ('NodeSocketVector', 'Light', (0.7071, 0.7071, 0.0)),
               ('NodeSocketFloat', 'Offset', 1.0),
               ('NodeSocketFloat', 'Texture Size', 256.0))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            try:
                s.default_value = default
            except (TypeError, ValueError):
                pass
            if name == 'Vector':
                try:
                    s.hide_value = True
                except (AttributeError, TypeError):
                    pass
        _apply_socket_tips(self, EMBOSS_SHIFT_SOCKET_DOCS)
        self.outputs.new('NodeSocketVector', 'Shifted UV')

    def ensure_sockets(self):
        _apply_socket_tips(self, EMBOSS_SHIFT_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        pass


class HALCYON_EmbossBumpNode(Node, HalcyonNodeBase):
    """R251 C135, stage two of DirectX 6 texture embossing (RIVA TNT /
    TNT2, GeForce 256, Voodoo3, 1998-2001): no per-pixel normal -- the
    inverted second sample (0.5 - h') is added to the first
    (D3DTOP_ADDSIGNED), 0.5 on flat ground and a signed bias on slopes,
    and the base texture is applied modulate-2x (D3DTOP_MODULATE2X), so
    relief reads as a one-directional emboss that slides and inverts
    as the light moves. Bitwise on both devices at NEAREST."""

    bl_idname = 'HALCYON_EmbossBumpNode'
    bl_label = "Emboss Bump (DirectX 6)"
    bl_icon = 'SHADING_TEXTURE'

    SOCKETS = (('NodeSocketFloat', 'Height', 0.5),
               ('NodeSocketFloat', 'Height Shifted', 0.5),
               ('NodeSocketColor', 'Base', (0.8, 0.8, 0.8, 1.0)))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            try:
                s.default_value = default
            except (TypeError, ValueError):
                pass
        _apply_socket_tips(self, EMBOSS_SOCKET_DOCS)
        self.outputs.new('NodeSocketFloat', 'Factor')
        self.outputs.new('NodeSocketColor', 'Color')

    def ensure_sockets(self):
        _apply_socket_tips(self, EMBOSS_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        pass


ROUGHNESS_SOCKET_DOCS = {
    'Normal': "The normal to roughen; unlinked takes the surface's shading "
              "normal (the master shader's own)",
    'Roughness': "Imagine's 0..255 attribute: 255 turns the normal by about "
                 "25 degrees rms, per PIXEL, from a stable hash of the pixel "
                 "position and the render seed",
}


class HALCYON_ImagineRoughnessNode(Node, HalcyonNodeBase):
    """R251 C099: Imagine's Roughness (Impulse Imagine 1.1-4.0, Turbo
    Silver): a per-pixel random turn of the shading normal before the
    lighting -- the sandpaper of Amiga ray-traced pottery. A stable hash
    of (pixel, seed) on both devices; Shimmer re-randomises per frame,
    Imagine's own behaviour, off by default (the manual warned against
    it in animation)."""

    bl_idname = 'HALCYON_ImagineRoughnessNode'
    bl_label = "Roughness (Imagine)"
    bl_icon = 'MOD_NOISE'

    animate: BoolProperty(
        name="Shimmer", default=False,
        description="Re-randomise every frame, Imagine's own behaviour -- the "
                    "manual warned against it in animation, so it is off; on, "
                    "the material becomes time-dependent")

    SOCKETS = (('NodeSocketVector', 'Normal', (0.0, 0.0, 0.0)),
               ('NodeSocketFloat', 'Roughness', 0.0))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            try:
                s.default_value = default
            except (TypeError, ValueError):
                pass
            if name == 'Normal':
                try:
                    s.hide_value = True
                except (AttributeError, TypeError):
                    pass
            if name == 'Roughness':
                try:
                    s.min_value, s.max_value = 0.0, 255.0
                except (AttributeError, TypeError):
                    pass
        _apply_socket_tips(self, ROUGHNESS_SOCKET_DOCS)
        self.outputs.new('NodeSocketVector', 'Normal')

    def ensure_sockets(self):
        _apply_socket_tips(self, ROUGHNESS_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        layout.prop(self, 'animate')


ENVCHROME_SOCKET_DOCS = {
    'Normal': "The normal the view ray reflects about; unlinked takes the "
              "surface's shading normal",
    'Sky Color': "Maya's envChrome skyColor: the sky plane at the horizon, "
                 "blending to the Zenith Color straight up",
    'Zenith Color': "Maya's envChrome zenithColor: the sky straight up "
                    "(reflection z = 1)",
    'Light Color': "Maya's envChrome lightColor: the rectangular fluorescent "
                   "tubes laid out on the sky plane; they never illuminate",
    'Floor Color': "Maya's envChrome floorColor: the floor plane straight "
                   "down, blending to the Horizon Color at grazing",
    'Horizon Color': "Maya's envChrome horizonColor: the floor at the horizon "
                     "(reflection z -> 0 from below)",
    'Grid Color': "Maya's envChrome gridColor: the ruled lines on the floor "
                  "plane",
}

#: (prop, Maya attribute, default, description)
_EC_PROPS = (
    ('light_width', 'lightWidth', 0.5,
     "Maya's envChrome lightWidth: the fraction of each lattice cell a "
     "fluorescent tube covers across the sky plane"),
    ('light_depth', 'lightDepth', 0.1,
     "Maya's envChrome lightDepth: the fraction of each lattice cell a "
     "tube covers along the sky plane's depth"),
    ('light_width_gain', 'lightWidthGain', 1.0,
     "Maya's envChrome lightWidthGain: tubes per unit of the sky plane "
     "across -- doubling it doubles the tube count"),
    ('light_width_offset', 'lightWidthOffset', 0.0,
     "Maya's envChrome lightWidthOffset: slides the tube lattice across "
     "the sky plane (in cells)"),
    ('light_depth_gain', 'lightDepthGain', 1.0,
     "Maya's envChrome lightDepthGain: tubes per unit of the sky plane "
     "along its depth"),
    ('light_depth_offset', 'lightDepthOffset', 0.0,
     "Maya's envChrome lightDepthOffset: slides the tube lattice along "
     "the sky plane's depth (in cells)"),
    ('grid_width', 'gridWidth', 0.1,
     "Maya's envChrome gridWidth: the fraction of each floor cell a grid "
     "line covers across the floor"),
    ('grid_depth', 'gridDepth', 0.1,
     "Maya's envChrome gridDepth: the fraction of each floor cell a grid "
     "line covers along the floor's depth"),
    ('grid_width_gain', 'gridWidthGain', 1.0,
     "Maya's envChrome gridWidthGain: grid lines per unit of the floor "
     "across"),
    ('grid_width_offset', 'gridWidthOffset', 0.0,
     "Maya's envChrome gridWidthOffset: slides the grid across the floor "
     "(in cells)"),
    ('grid_depth_gain', 'gridDepthGain', 1.0,
     "Maya's envChrome gridDepthGain: grid lines per unit of the floor "
     "along its depth"),
    ('grid_depth_offset', 'gridDepthOffset', 0.0,
     "Maya's envChrome gridDepthOffset: slides the grid along the floor's "
     "depth (in cells)"),
    ('floor_altitude', 'floorAltitude', -1.0,
     "Maya's envChrome floorAltitude: the height (Blender Z) of the floor "
     "plane the downward reflections hit"),
)


class HALCYON_EnvChromeNode(Node, HalcyonNodeBase):
    """R251 C123: Alias PowerAnimator / Maya 1-4 envChrome -- a procedural
    showroom seen only in reflections: a sky plane with rectangular
    fluorescent tubes laid out by width / depth gains and offsets, a
    floor at an altitude carrying a ruled grid; the chrome of every
    1990s Alias logo. Maya's PARAMETERS at Maya's defaults; the plane
    at unit height and the linear sky / floor blends are Halcyon's
    (Maya's are unpublished). Plug Color into the master shader's
    Matcap (Add) -- Maya's Reflected Color slot. Maya's Y-up is
    Blender's Z-up here. The tubes never illuminate (per Maya)."""

    bl_idname = 'HALCYON_EnvChromeNode'
    bl_label = "Env Chrome (Alias / Maya)"
    bl_icon = 'MATSPHERE'

    light_width: FloatProperty(name="Light Width", default=_EC_PROPS[0][2], min=0.0, max=1.0,
                               description=_EC_PROPS[0][3])
    light_depth: FloatProperty(name="Light Depth", default=_EC_PROPS[1][2], min=0.0, max=1.0,
                               description=_EC_PROPS[1][3])
    light_width_gain: FloatProperty(name="Light Width Gain", default=_EC_PROPS[2][2], min=0.0, soft_max=16.0,
                                    description=_EC_PROPS[2][3])
    light_width_offset: FloatProperty(name="Light Width Offset", default=_EC_PROPS[3][2], soft_min=-4.0, soft_max=4.0,
                                      description=_EC_PROPS[3][3])
    light_depth_gain: FloatProperty(name="Light Depth Gain", default=_EC_PROPS[4][2], min=0.0, soft_max=16.0,
                                    description=_EC_PROPS[4][3])
    light_depth_offset: FloatProperty(name="Light Depth Offset", default=_EC_PROPS[5][2], soft_min=-4.0, soft_max=4.0,
                                      description=_EC_PROPS[5][3])
    grid_width: FloatProperty(name="Grid Width", default=_EC_PROPS[6][2], min=0.0, max=1.0,
                              description=_EC_PROPS[6][3])
    grid_depth: FloatProperty(name="Grid Depth", default=_EC_PROPS[7][2], min=0.0, max=1.0,
                              description=_EC_PROPS[7][3])
    grid_width_gain: FloatProperty(name="Grid Width Gain", default=_EC_PROPS[8][2], min=0.0, soft_max=16.0,
                                   description=_EC_PROPS[8][3])
    grid_width_offset: FloatProperty(name="Grid Width Offset", default=_EC_PROPS[9][2], soft_min=-4.0, soft_max=4.0,
                                     description=_EC_PROPS[9][3])
    grid_depth_gain: FloatProperty(name="Grid Depth Gain", default=_EC_PROPS[10][2], min=0.0, soft_max=16.0,
                                   description=_EC_PROPS[10][3])
    grid_depth_offset: FloatProperty(name="Grid Depth Offset", default=_EC_PROPS[11][2], soft_min=-4.0, soft_max=4.0,
                                     description=_EC_PROPS[11][3])
    floor_altitude: FloatProperty(name="Floor Altitude", default=_EC_PROPS[12][2], soft_min=-20.0, soft_max=20.0,
                                  description=_EC_PROPS[12][3])
    real_floor: BoolProperty(
        name="Real Floor", default=True,
        description="Maya's envChrome realFloor: the floor is a true plane at "
                    "Floor Altitude the reflected ray intersects from the "
                    "surface point (parallax); off, the grid depends on the "
                    "reflection direction alone")

    SOCKETS = (('NodeSocketVector', 'Normal', (0.0, 0.0, 0.0)),
               ('NodeSocketColor', 'Sky Color', (0.55, 0.62, 0.78, 1.0)),
               ('NodeSocketColor', 'Zenith Color', (0.15, 0.22, 0.48, 1.0)),
               ('NodeSocketColor', 'Light Color', (1.0, 1.0, 1.0, 1.0)),
               ('NodeSocketColor', 'Floor Color', (0.18, 0.18, 0.18, 1.0)),
               ('NodeSocketColor', 'Horizon Color', (0.5, 0.5, 0.5, 1.0)),
               ('NodeSocketColor', 'Grid Color', (0.04, 0.04, 0.04, 1.0)))

    def init(self, context):
        for kind, name, default in self.SOCKETS:
            s = self.inputs.new(kind, name)
            try:
                s.default_value = default
            except (TypeError, ValueError):
                pass
            if name == 'Normal':
                try:
                    s.hide_value = True
                except (AttributeError, TypeError):
                    pass
        _apply_socket_tips(self, ENVCHROME_SOCKET_DOCS)
        self.outputs.new('NodeSocketColor', 'Color')

    def ensure_sockets(self):
        _apply_socket_tips(self, ENVCHROME_SOCKET_DOCS)

    def draw_buttons(self, context, layout):
        col = layout.column(align=True)
        col.label(text="Lights (sky plane)")
        col.prop(self, 'light_width')
        col.prop(self, 'light_depth')
        col.prop(self, 'light_width_gain')
        col.prop(self, 'light_width_offset')
        col.prop(self, 'light_depth_gain')
        col.prop(self, 'light_depth_offset')
        col = layout.column(align=True)
        col.label(text="Floor grid")
        col.prop(self, 'grid_width')
        col.prop(self, 'grid_depth')
        col.prop(self, 'grid_width_gain')
        col.prop(self, 'grid_width_offset')
        col.prop(self, 'grid_depth_gain')
        col.prop(self, 'grid_depth_offset')
        col.prop(self, 'floor_altitude')
        col.prop(self, 'real_floor')


# ---- MAT-B: period node registration ----
PERIOD_NODES = (HALCYON_CombinerStageNode, HALCYON_SRBumpNode, HALCYON_EmbossShiftNode, HALCYON_EmbossBumpNode, HALCYON_ImagineRoughnessNode, HALCYON_EnvChromeNode,)
#: what the exporter copies for the period nodes (export.py folds
#: FAMILY_NODE_PROPS into NODE_PROPS; the Emboss node has no props)
PERIOD_NODE_PROPS = {
    'HALCYON_CombinerStageNode': ('hardware', 'op', 'bias', 'scale', 'clamp',
                                  'map_a', 'map_b', 'map_c', 'map_d',
                                  'nv_scale', 'nv_bias'),
    'HALCYON_SRBumpNode': ('blend',),
    'HALCYON_ImagineRoughnessNode': ('animate',),
    'HALCYON_EnvChromeNode': ('light_width', 'light_depth', 'light_width_gain', 'light_width_offset', 'light_depth_gain', 'light_depth_offset', 'grid_width', 'grid_depth', 'grid_width_gain', 'grid_width_offset', 'grid_depth_gain', 'grid_depth_offset', 'floor_altitude', 'real_floor',),
}
FAMILY_NODE_PROPS.update(PERIOD_NODE_PROPS)
NODES = NODES + PERIOD_NODES
_PERIOD_SHADING = tuple(c for c in PERIOD_NODES
                        if c.bl_idname != 'HALCYON_ImagineRoughnessNode')
_PERIOD_VECTOR = tuple(c for c in PERIOD_NODES
                       if c.bl_idname == 'HALCYON_ImagineRoughnessNode')
MENU_FAMILIES = tuple(
    (t, i, (m + _PERIOD_SHADING if t == 'Shading'
            else (m + _PERIOD_VECTOR if t == 'Vector' else m)))
    for t, i, m in MENU_FAMILIES)
MENU_SUBMENUS = tuple(_family_menu(t, m) for t, _i, m in MENU_FAMILIES)
