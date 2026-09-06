"""Shipped presets: named after the machines and packages they emulate.

Each entry is a partial RenderSettings override. The values are chosen from what
each target actually did -- Infini-D's Phong with no ambient occlusion and a
sharp 24-bit output, the PlayStation's affine texture warp and vertex snapping,
Imagine's HAM8 framebuffer, and so on. Nothing here is a generic "retro" blur.
"""

CATEGORIES = (
    ('GENERAL', "General"),
    ('SOFTWARE', "3D Software"),
    ('PLATFORM', "Home Computers"),
    ('CONSOLE', "Game Consoles"),
    ('BROADCAST', "Video & Broadcast"),
    ('WEB', "Early Web"),
    ('CEL', "Cel & Film"),
)

PRESETS = {

    'DEFAULT': {
        'label': "Halcyon Default",
        'category': 'GENERAL',
        'note': "Every setting back to its default. Applying any preset resets "
                "first, so this is also what you get by clearing one.",
        'settings': {},
    },

    # ------------------------------------------------------------- software
    'BLENDER_INTERNAL': {
        'label': "Blender Internal 2.79 (2017)",
        'category': 'SOFTWARE',
        'note': "The renderer the classic .blend files were made for. "
                "BI's own Lambert/CookTorr default, per-lamp shadows "
                "exactly as each lamp authored them (Ray, a spot's "
                "Buffer, or none at all), ray depth 2, Mitchell "
                "anti-aliasing and Blender's soft dither. 2.79 "
                "rendered scene-linear and showed it through the "
                "scene's 'Default' view -- the sRGB display encode -- "
                "so this preset runs the same two pipeline ends: "
                "sRGB textures linearize on load, and the sRGB curve "
                "grades the final frame. Pair it with a legacy import "
                "for the closest match to the original render. BI "
                "defaulted to 8 AA samples; 4 here renders 4x the "
                "pixels instead of 9x -- raise it if the edges matter "
                "more than the wait.",
        'settings': {
            'default_model': 'BI_COOKTORR', 'shading_rate': 'PIXEL',
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'aa_filter': 'MITCHELL',
            # PER_LIGHT: the import stamps each lamp with 2.79's own
            # rule (spot+buffer -> map, the Ray bit -> traced, neither
            # -> NO shadow). Forcing RAY here shadowed lamps BI never
            # shadowed AND built a BVH into every viewport update --
            # the field's 'preset cripples rendering time'.
            'shadows': True, 'shadow_default': 'PER_LIGHT',
            'raytrace': True, 'ray_depth': 2,
            'ray_reflection': True, 'ray_refraction': True,
            'transparency': 'SORTED',
            'tex_filter': 'TRILINEAR', 'tex_perspective': True,
            'color_depth': '24', 'dither': 'NOISE',
            # 2.79's REAL pipeline (the file DNA says so: view
            # transform 'Default' on an sRGB display): linear render,
            # sRGB display encode. SRGB here is Halcyon's OWN curve --
            # Blender stays pinned to Raw, so nothing double-grades
            # (the R153 grey wash was Blender's 'Standard' stacked on
            # engine output). input_gamma_naive False is the matching
            # input end: sRGB textures linearize on load, exactly as
            # BI sampled them. DNA colours are already linear.
            # specular_in_gamma True: BI added specular in linear
            # light like everything else; False's pow-2.2 crush is a
            # 2.4x-era emulation, not 2.79.
            'color_management': 'SRGB', 'input_gamma_naive': False,
            'gamma': 1.0, 'specular_in_gamma': True,
            'global_ambient': (0.0, 0.0, 0.0),
            'fog': False, 'glow': False,
        },
    },
    'INFINID_4': {
        'label': "Specular Infini-D 4 (1995)",
        'category': 'SOFTWARE',
        'note': "Mac Phong renderer, 1995. Clean 24-bit output, soft shadow maps, "
                "no ambient occlusion and a slightly hot specular.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'BOX',
            'default_model': 'PHONG', 'shading_rate': 'PIXEL',
            'specular_in_gamma': True, 'clamp_specular': True,
            'shadows': True, 'shadow_default': 'MAP', 'shadow_map_size': 512,
            'shadow_softness': 1.5, 'shadow_samples': 8,
            'tex_filter': 'BILINEAR', 'tex_perspective': True,
            'transparency': 'SORTED', 'color_depth': '24', 'dither': 'NONE',
            'global_ambient': (0.08, 0.08, 0.09), 'gamma': 1.8,
            'raytrace': True, 'ray_depth': 2, 'ray_reflection': True,
        },
    },
    'RAY_DREAM_5': {
        'label': "Ray Dream Studio 5 (1996)",
        'category': 'SOFTWARE',
        'note': "Ray-traced reflections and a glossy plastic default. "
                "Gaussian AA, slight glow on highlights.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'GAUSS',
            'default_model': 'BLINN', 'raytrace': True, 'ray_depth': 3,
            'ray_reflection': True, 'ray_refraction': True,
            'shadows': True, 'shadow_default': 'RAY', 'shadow_samples': 1,
            'glow': True, 'glow_threshold': 0.9, 'glow_intensity': 0.35,
            'glow_radius': 8.0, 'color_depth': '24', 'gamma': 2.2,
            'tex_filter': 'BILINEAR',
        },
    },
    'STRATA_PRO': {
        'label': "Strata StudioPro 1.75 (1995)",
        'category': 'SOFTWARE',
        'note': "The chrome-and-marble Mac look: hard ray-traced reflections, "
                "sharp shadows, a cross-screen star filter on the highlights.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'BLINN', 'raytrace': True, 'ray_depth': 4,
            'shadows': True, 'shadow_default': 'RAY',
            'star_filter': True, 'star_points': 4, 'star_length': 40.0,
            'star_intensity': 0.55, 'glow': True, 'glow_intensity': 0.3,
            'color_depth': '24', 'gamma': 1.8, 'env_reflection': True,
        },
    },
    'MAX_R2': {
        'label': "3D Studio MAX R2 (1997)",
        'category': 'SOFTWARE',
        'note': "Blinn default with the Soften parameter, shadow maps, and the "
                "characteristic slightly grey ambient.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'BLINN', 'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 512, 'shadow_bias': 0.02, 'shadow_softness': 2.0,
            'shadow_samples': 8, 'global_ambient': (0.12, 0.12, 0.12),
            'specular_in_gamma': True, 'color_depth': '24',
            'tex_filter': 'BILINEAR', 'gamma': 2.2,
        },
    },
    'MAX_2012': {
        'label': "3ds Max 2012, Default Scanline (2011)",
        'category': 'SOFTWARE',
        'note': "The Default Scanline Renderer as 3ds Max 2012 shipped it, "
                "at its own defaults: 640x480 square pixels, the Standard "
                "material's Blinn shader for anything without a material "
                "(the '(3ds Max)' models and the Standard / Raytrace nodes "
                "carry the rest), anti-aliasing on with the Area filter "
                "(a pixel-wide box over 4x4 samples is the nearest this "
                "engine has to Max's 1.5-pixel area), Filter Maps on "
                "(pyramidal texture filtering), shadow maps at Max's 512 "
                "with its Sample Range of 4, no ambient light (Max 2012's "
                "environment ambient is black), the raytracer's global "
                "depth of 9 for Raytrace materials, and NO gamma/LUT "
                "correction -- Max 2012 shipped with it off, so the frame "
                "is the linear light itself on a 24-bit output with no "
                "dither. No light limit, no fog, no glow.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'BOX',
            'aa_filter_width': 1.0,
            'default_model': 'MAX_BLINN', 'shading_rate': 'PIXEL',
            'specular_in_gamma': True,
            'shadows': True, 'shadow_default': 'MAP', 'shadow_map_size': 512,
            'shadow_bias': 0.02, 'shadow_softness': 4.0, 'shadow_samples': 8,
            'global_ambient': (0.0, 0.0, 0.0), 'max_lights': 0,
            'raytrace': True, 'ray_depth': 9,
            'ray_reflection': True, 'ray_refraction': True,
            'transparency': 'SORTED',
            'tex_filter': 'TRILINEAR', 'tex_mipmap': True, 'tex_perspective': True,
            'color_depth': '24', 'dither': 'NONE',
            'gamma': 1.0, 'color_management': 'NONE', 'input_gamma_naive': True,
            'fog': False, 'glow': False,
        },
    },
    'STUDIO_R4': {
        'label': "3D Studio R4, DOS (1994)",
        'category': 'SOFTWARE',
        'note': "The 320x200 VGA workhorse. Phong, 8 lights maximum, "
                "256 colours with an adaptive palette.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'max_lights': 8,
            'shadows': True, 'shadow_default': 'MAP', 'shadow_map_size': 256,
            'tex_filter': 'NEAREST', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'MEDIAN_CUT', 'dither': 'FLOYD',
            'gamma': 2.2, 'output_scale': '2X',
        },
    },
    'TRUESPACE_2': {
        'label': "trueSpace 2 (1995)",
        'category': 'SOFTWARE',
        'note': "Fast Phong scanline with hard shadow maps and 16-bit colour.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 2,
            'default_model': 'PHONG', 'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 256, 'shadow_samples': 1, 'shadow_softness': 0.0,
            'color_depth': '16', 'dither': 'BAYER4', 'tex_filter': 'BILINEAR',
            'gamma': 2.2,
        },
    },
    'LIGHTWAVE_56': {
        'label': "LightWave 5.6 (1998)",
        'category': 'SOFTWARE',
        'note': "Broadcast-quality scanline: sharp AA, ray-traced shadows, "
                "the classic Toaster-era specular.",
        'settings': {
            'resolution_x': 752, 'resolution_y': 480,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9, 'aa_filter': 'MITCHELL',
            'default_model': 'PHONG', 'raytrace': True, 'ray_depth': 2,
            'shadows': True, 'shadow_default': 'RAY',
            'color_depth': '24', 'gamma': 2.2, 'glow': True,
            'glow_intensity': 0.25, 'glow_threshold': 0.92,
        },
    },
    'IMAGINE_3': {
        'label': "Imagine 3.0, Amiga (1994)",
        'category': 'SOFTWARE',
        'note': "HAM8 framebuffer, Phong shading, the fringing on colour "
                "transitions that hold-and-modify always produced.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 256,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'max_lights': 8,
            'shadows': True, 'shadow_default': 'MAP',
            'color_depth': 'HAM8', 'dither': 'FLOYD', 'dither_strength': 0.7,
            'tex_filter': 'NEAREST', 'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'POVRAY_31': {
        'label': "POV-Ray 3.1 (1998)",
        'category': 'SOFTWARE',
        'note': "Pure ray tracer: hard shadows, mirror reflections, "
                "no ambient occlusion and a flat ambient term.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9, 'aa_filter': 'BOX',
            'default_model': 'PHONG', 'raytrace': True, 'ray_depth': 5,
            'ray_reflection': True, 'ray_refraction': True,
            'shadows': True, 'shadow_default': 'RAY', 'shadow_samples': 1,
            'global_ambient': (0.1, 0.1, 0.1), 'color_depth': '24',
            'gamma': 1.0, 'color_management': 'NONE',
        },
    },
    'BRYCE_2': {
        'label': "Bryce 2 (1996)",
        'category': 'SOFTWARE',
        'note': "Hazy terrain look: heavy linear fog, soft key light, "
                "gentle bloom.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'LAMBERT', 'fog': True, 'fog_mode': 'EXP',
            'fog_density': 0.02, 'fog_color': (0.62, 0.70, 0.82),
            'shadows': True, 'shadow_default': 'MAP', 'shadow_softness': 3.0,
            'shadow_samples': 12, 'glow': True, 'glow_intensity': 0.3,
            'glow_threshold': 0.75, 'color_depth': '24', 'gamma': 2.2,
            'saturation': 1.1,
        },
    },

    'BRYCE_STILL': {
        'label': "Bryce still (overnight render)",
        'category': 'SOFTWARE',
        'note': "The postcard: the render you queued at midnight and "
                "collected at breakfast. High supersampling, soft ray "
                "shadows, a breath of haze and bloom, colour turned up "
                "a notch. Pair it with the Bryce sky and a Terrain.",
        'settings': {
            'resolution_x': 800, 'resolution_y': 600,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9,
            'aa_filter': 'CATROM',
            'default_model': 'LAMBERT',
            'shadows': True, 'shadow_default': 'RAY',
            'ray_shadows': True, 'shadow_softness': 2.0,
            'shadow_samples': 16,
            'fog': True, 'fog_mode': 'EXP', 'fog_density': 0.008,
            'fog_color': (0.66, 0.74, 0.85),
            'glow': True, 'glow_intensity': 0.22,
            'glow_threshold': 0.8,
            'color_depth': '24', 'dither': 'NONE',
            'gamma': 2.2, 'saturation': 1.15,
        },
    },

    # ---------------------------------------------------- more software
    'ELECTRIC_IMAGE': {
        'label': "ElectricImage 2.9 (1996)",
        'category': 'SOFTWARE',
        'note': "The high-end Mac scanline renderer. Very clean edges, tight "
                "speculars, 24-bit output and no visible dither.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 16, 'aa_filter': 'MITCHELL',
            'default_model': 'PHONG', 'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 1024, 'shadow_samples': 12,
            'specular_in_gamma': True, 'color_depth': '24', 'dither': 'NONE',
            'gamma': 1.8, 'tex_filter': 'BILINEAR', 'glow': True,
            'glow_intensity': 0.2, 'glow_threshold': 0.95,
        },
    },
    'SOFTIMAGE_3D': {
        'label': "Softimage|3D (1994)",
        'category': 'SOFTWARE',
        'note': "Film-house scanline: heavy anti-aliasing, ray-traced shadows, "
                "restrained specular. The look of mid-90s effects work.",
        'settings': {
            'resolution_x': 720, 'resolution_y': 486,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 16, 'aa_filter': 'GAUSS',
            'default_model': 'BLINN', 'raytrace': True, 'ray_depth': 3,
            'shadows': True, 'shadow_default': 'RAY', 'shadow_samples': 4,
            'color_depth': '24', 'gamma': 2.2, 'clamp_specular': True,
        },
    },
    'ALIAS_POWER': {
        'label': "Alias PowerAnimator (1993)",
        'category': 'SOFTWARE',
        'note': "SGI workstation output: Blinn surfaces, clean ray tracing, "
                "and the slightly cool cast of an Indigo monitor.",
        'settings': {
            'resolution_x': 646, 'resolution_y': 485,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9, 'aa_filter': 'CATROM',
            'default_model': 'BLINN', 'raytrace': True, 'ray_depth': 4,
            'ray_reflection': True, 'shadows': True, 'shadow_default': 'RAY',
            'color_depth': '24', 'gamma': 1.7, 'saturation': 0.95,
        },
    },
    'WAVEFRONT': {
        'label': "Wavefront Advanced Visualizer (1988)",
        'category': 'SOFTWARE',
        'note': "Early-90s SGI scanline. Hard shadow maps, Phong highlights, "
                "no ambient occlusion of any kind.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'BOX',
            'default_model': 'PHONG', 'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 512, 'shadow_softness': 0.0, 'shadow_samples': 1,
            'global_ambient': (0.12, 0.12, 0.14), 'color_depth': '24',
            'gamma': 2.2, 'specular_in_gamma': True,
        },
    },
    'CINEMA4D_4': {
        'label': "CINEMA 4D v4 (1996)",
        'category': 'SOFTWARE',
        'note': "The Amiga-descended PC release. Fast scanline, soft shadow "
                "maps, 24-bit, slightly warm.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'TRIANGLE',
            'default_model': 'PHONG', 'shadows': True, 'shadow_default': 'MAP',
            'shadow_softness': 2.5, 'shadow_samples': 8,
            'color_depth': '24', 'gamma': 2.2, 'saturation': 1.05,
        },
    },
    'REAL3D': {
        'label': "Real 3D 2, Amiga (1993)",
        'category': 'SOFTWARE',
        'note': "Amiga ray tracer with hard shadows and mirror reflections, "
                "written to a 24-bit framebuffer.",
        'settings': {
            'resolution_x': 384, 'resolution_y': 288,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'raytrace': True, 'ray_depth': 6,
            'ray_reflection': True, 'ray_refraction': True,
            'shadows': True, 'shadow_default': 'RAY', 'shadow_samples': 1,
            'color_depth': '24', 'gamma': 2.2, 'output_scale': '2X',
        },
    },
    'VISTAPRO': {
        'label': "Vistapro (1991)",
        'category': 'SOFTWARE',
        'note': "Fractal landscape generator: flat Lambert terrain, heavy "
                "distance haze, 256 colours.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'NONE', 'default_model': 'LAMBERT',
            'shading_rate': 'FACE', 'shadows': False,
            'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 3.0,
            'fog_end': 45.0, 'fog_color': (0.68, 0.76, 0.88),
            'color_depth': '8', 'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'dither': 'NONE', 'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'ANIMATION_MASTER': {
        'label': "Hash Animation:Master (1996)",
        'category': 'SOFTWARE',
        'note': "Spline modeller with a soft, plasticky shader and gentle "
                "toon-adjacent falloff.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9, 'aa_filter': 'GAUSS',
            'default_model': 'BLINN_PHONG', 'shadows': True,
            'shadow_default': 'MAP', 'shadow_softness': 3.0,
            'shadow_samples': 12, 'color_depth': '24', 'gamma': 2.2,
            'saturation': 1.15, 'glow': True, 'glow_intensity': 0.25,
        },
    },
    'POVRAY_2': {
        'label': "POV-Ray 2.2 (1993)",
        'category': 'SOFTWARE',
        'note': "The earlier ray tracer: no area lights, hard shadows, and a "
                "completely flat ambient term.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'BOX',
            'default_model': 'PHONG', 'raytrace': True, 'ray_depth': 5,
            'ray_reflection': True, 'shadows': True, 'shadow_default': 'RAY',
            'shadow_samples': 1, 'global_ambient': (0.15, 0.15, 0.15),
            'color_depth': '24', 'gamma': 1.0, 'color_management': 'NONE',
            'output_scale': '2X',
        },
    },
    'VUE_DESPRIT': {
        'label': "Vue d'Esprit 2 (1997)",
        'category': 'SOFTWARE',
        'note': "Bryce's rival: atmospheric outdoor scenes, soft light, "
                "strong haze and a warm cast.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'LAMBERT', 'shadows': True, 'shadow_default': 'MAP',
            'shadow_softness': 4.0, 'shadow_samples': 12,
            'fog': True, 'fog_mode': 'EXP', 'fog_density': 0.015,
            'fog_color': (0.72, 0.76, 0.82), 'glow': True,
            'glow_intensity': 0.35, 'glow_threshold': 0.8,
            'color_depth': '24', 'gamma': 2.2, 'saturation': 1.1,
        },
    },

    # ---------------------------------------------------- more platforms
    'ATARI_ST': {
        'label': "Atari ST (1985)",
        'category': 'PLATFORM',
        'note': "320x200 in 16 colours chosen from 512. Chunky, and dithered "
                "to death.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'ADAPTIVE', 'palette_size': 16,
            'palette_method': 'MEDIAN_CUT', 'dither': 'FLOYD',
            'shadows': False, 'max_lights': 2, 'gamma': 2.2,
            'output_scale': '3X',
        },
    },
    # ---- handhelds and later consoles -------------------------------------
    'GAME_BOY': {
        'label': "Game Boy (1989)",
        'category': 'CONSOLE',
        'note': "160x144 in four shades. No colour, no shadows, and a screen "
                "small enough that everything had to read as silhouette.",
        'settings': {
            'resolution_x': 160, 'resolution_y': 144,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '2',
            'palette_mode': 'GRAY', 'palette_size': 4,
            'dither': 'BAYER2', 'shadows': False, 'max_lights': 1,
            'output_scale': '4X',
        },
    },
    'VIRTUAL_BOY': {
        'label': "Virtual Boy (1995)",
        'category': 'CONSOLE',
        'note': "384x224 in four levels of red on black. The only console that "
                "shipped a palette with one hue in it.",
        'settings': {
            'resolution_x': 384, 'resolution_y': 224,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '2',
            'palette_mode': 'GRAY', 'palette_size': 4,
            'dither': 'BAYER4', 'shadows': False, 'max_lights': 1,
            'crt': True, 'crt_scanlines': 0.0, 'crt_bloom': 0.4,
            'output_scale': '2X',
        },
    },
    'GAME_GEAR': {
        'label': "Game Gear (1990)",
        'category': 'CONSOLE',
        'note': "160x144 from a 4096-colour master palette, on a backlit "
                "screen that smeared everything it showed.",
        'settings': {
            'resolution_x': 160, 'resolution_y': 144,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'ADAPTIVE', 'palette_size': 32,
            'palette_method': 'MEDIAN_CUT', 'dither': 'BAYER4',
            'shadows': False, 'max_lights': 2, 'glow': True,
            'glow_threshold': 0.6, 'glow_intensity': 0.25,
            'output_scale': '4X',
        },
    },
    'SNES': {
        'label': "Super Nintendo (1990)",
        'category': 'CONSOLE',
        'note': "256x224. Polygons on this hardware came from a chip on the "
                "cartridge, so they were few, flat and unfiltered.",
        'settings': {
            'resolution_x': 256, 'resolution_y': 224,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.14,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '5',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'MEDIAN_CUT', 'dither': 'NONE',
            'depth_sort': 'PAINTERS', 'shadows': False, 'max_lights': 1,
            'output_scale': '3X',
        },
    },
    'NEO_GEO': {
        'label': "Neo Geo (1990)",
        'category': 'CONSOLE',
        'note': "320x224 from 65,536 colours. The most expensive way to see a "
                "sprite in 1990.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 224,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 2,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'color_depth': '5',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'OCTREE', 'dither': 'NONE',
            'shadows': True, 'max_lights': 3, 'output_scale': '3X',
        },
    },
    'SEGA_32X': {
        'label': "Sega 32X (1994)",
        'category': 'CONSOLE',
        'note': "Two extra processors bolted on top of a Mega Drive, and "
                "32,768 colours to show for it.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 224,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '5',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'MEDIAN_CUT', 'dither': 'BAYER4',
            'depth_sort': 'PAINTERS', 'shadows': False, 'max_lights': 2,
            'composite': True, 'output_scale': '2X',
        },
    },

    # ---- home computers ----------------------------------------------------
    'C64': {
        'label': "Commodore 64 (1982)",
        'category': 'PLATFORM',
        'note': "160x200 in sixteen fixed colours, half of them barely "
                "distinguishable. Wide pixels, because the mode was.",
        'settings': {
            'resolution_x': 160, 'resolution_y': 200,
            'pixel_aspect_x': 2.0, 'pixel_aspect_y': 1.0,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'ADAPTIVE', 'palette_size': 16,
            'palette_method': 'MEDIAN_CUT', 'dither': 'BAYER4',
            'shadows': False, 'max_lights': 1, 'output_scale': '3X',
        },
    },
    'ZX_SPECTRUM': {
        'label': "ZX Spectrum (1982)",
        'category': 'PLATFORM',
        'note': "256x192 from fifteen colours. The real machine allowed two "
                "per character cell, which is why everything on it looked "
                "like it had been coloured in afterwards.",
        'settings': {
            'resolution_x': 256, 'resolution_y': 192,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '3',
            'palette_mode': 'ADAPTIVE', 'palette_size': 15,
            'palette_method': 'MEDIAN_CUT', 'dither': 'BAYER8',
            'shadows': False, 'max_lights': 1, 'output_scale': '3X',
        },
    },
    'APPLE_IIGS': {
        'label': "Apple IIGS (1986)",
        'category': 'PLATFORM',
        'note': "320x200, sixteen colours a line chosen from 4096. Gentler "
                "than the PC palettes of the same year.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 2,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'ADAPTIVE', 'palette_size': 16,
            'palette_method': 'MEDIAN_CUT', 'dither': 'FLOYD',
            'shadows': False, 'max_lights': 2, 'output_scale': '3X',
        },
    },
    'MSX2': {
        'label': "MSX2 (1985)",
        'category': 'PLATFORM',
        'note': "256x212 in 256 colours. Japan's home standard, and better at "
                "this than anything sold in the West that year.",
        'settings': {
            'resolution_x': 256, 'resolution_y': 212,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'MEDIAN_CUT', 'dither': 'FLOYD',
            'shadows': False, 'max_lights': 2, 'output_scale': '3X',
        },
    },
    'NEXTSTEP': {
        'label': "NeXTSTEP (1989)",
        'category': 'PLATFORM',
        'note': "Two-bit greyscale on a large, sharp display. Everything the "
                "MegaPixel monitor showed was four shades and no apology.",
        'settings': {
            'resolution_x': 1120, 'resolution_y': 832,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'shading_rate': 'PIXEL',
            'tex_filter': 'BILINEAR', 'color_depth': '2',
            'palette_mode': 'GRAY', 'palette_size': 4,
            'dither': 'ATKINSON', 'shadows': True, 'max_lights': 4,
        },
    },
    'SGI_INDY': {
        'label': "SGI Indy (1993)",
        'category': 'PLATFORM',
        'note': "The workstation everything else was compared against: full "
                "24-bit colour, smooth shading and no palette at all.",
        'settings': {
            'resolution_x': 1024, 'resolution_y': 768,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'shading_rate': 'PIXEL',
            'tex_filter': 'TRILINEAR', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'dither': 'NONE', 'shadows': True, 'max_lights': 8,
        },
    },

    # ---- software renderers ------------------------------------------------
    'DOOM': {
        'label': "Doom software (1993)",
        'category': 'PLATFORM',
        'note': "320x200 in 256 colours with light levels quantised into "
                "bands. The bands are the shading model, not an artefact.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'MEDIAN_CUT', 'dither': 'NONE',
            'depth_sort': 'PAINTERS', 'shadows': False, 'max_lights': 2,
            'fog': True, 'output_scale': '3X',
        },
    },
    'RENDERMAN': {
        'label': "RenderMan (1988)",
        'category': 'SOFTWARE',
        'note': "What the film houses used while everyone else argued about "
                "palettes. Full colour, clean edges, and time to spare.",
        'settings': {
            'resolution_x': 1024, 'resolution_y': 778,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'BLINN', 'shading_rate': 'PIXEL',
            'tex_filter': 'TRILINEAR', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'dither': 'NONE', 'shadows': True, 'max_lights': 8,
            'raytrace': True, 'ambient_occlusion': True,
        },
    },
    'TURBO_SILVER': {
        'label': "Turbo Silver (1987)",
        'category': 'SOFTWARE',
        'note': "The Amiga raytracer that became Imagine. Hard shadows, hard "
                "reflections, and a HAM palette doing its best underneath.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 256,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.1,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 2,
            'default_model': 'PHONG', 'shading_rate': 'PIXEL',
            'tex_filter': 'NEAREST', 'color_depth': '6',
            'palette_mode': 'ADAPTIVE', 'palette_size': 64,
            'palette_method': 'MEDIAN_CUT', 'dither': 'FLOYD',
            'shadows': True, 'shadow_softness': 0.0, 'max_lights': 3,
            'raytrace': True, 'output_scale': '2X',
        },
    },
    'LIGHTSCAPE': {
        'label': "Lightscape (1994)",
        'category': 'SOFTWARE',
        'note': "Radiosity, when that meant hours of solving before anything "
                "appeared. Soft, indirect, and short of hard shadows.",
        'settings': {
            'resolution_x': 800, 'resolution_y': 600,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'LAMBERT', 'shading_rate': 'PIXEL',
            'tex_filter': 'BILINEAR', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'dither': 'NONE', 'shadows': True, 'shadow_softness': 4.0,
            'max_lights': 6, 'ambient_occlusion': True, 'ao_intensity': 0.8,
            'global_ambient_level': 0.35,
        },
    },
    'AUTOSHADE': {
        'label': "AutoShade (1987)",
        'category': 'SOFTWARE',
        'note': "AutoCAD's renderer, when rendering meant filling each face "
                "with one colour and calling it a day.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'EGA16', 'dither': 'NONE',
            'shadows': False, 'max_lights': 1,
        },
    },
    'AMIGA_AGA': {
        'label': "Amiga AGA 256 (1992)",
        'category': 'PLATFORM',
        'note': "The AGA chipset: 256 colours from a 24-bit master palette at "
                "320x256 PAL.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 256,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'tex_filter': 'NEAREST',
            'color_depth': '8', 'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'OCTREE', 'dither': 'FLOYD',
            'shadows': True, 'shadow_default': 'MAP', 'gamma': 2.2,
            'output_scale': '3X',
        },
    },
    'CGA': {
        'label': "CGA 4 colour (1981)",
        'category': 'PLATFORM',
        'note': "Cyan, magenta, white and black. The most punishing palette "
                "the PC ever shipped.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'CGA4', 'dither': 'BAYER4',
            'shadows': False, 'max_lights': 1, 'gamma': 2.2,
            'output_scale': '3X',
        },
    },
    'HERCULES': {
        'label': "Hercules mono (1982)",
        'category': 'PLATFORM',
        'note': "720x348 in one bit. Everything is dither pattern.",
        'settings': {
            'resolution_x': 720, 'resolution_y': 348,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.55,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'LAMBERT', 'tex_filter': 'NEAREST',
            'color_depth': '1', 'dither': 'BAYER8', 'shadows': False,
            'gamma': 2.2,
        },
    },
    'MAC_1BIT': {
        'label': "Macintosh 1-bit (1984)",
        'category': 'PLATFORM',
        'note': "512x342 black and white with Atkinson dither -- the kernel "
                "Bill Atkinson wrote for exactly this screen.",
        'settings': {
            'resolution_x': 512, 'resolution_y': 342,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'LAMBERT', 'tex_filter': 'NEAREST',
            'color_depth': '1', 'dither': 'ATKINSON',
            'shadows': True, 'shadow_default': 'MAP', 'gamma': 1.8,
        },
    },
    'PC98': {
        'label': "NEC PC-98 (1982)",
        'category': 'PLATFORM',
        'note': "640x400 in 16 colours from 4096. The Japanese business PC "
                "that ran a surprising number of 3D demos.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 400,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'ADAPTIVE', 'palette_size': 16,
            'dither': 'BAYER4', 'shadows': False, 'gamma': 2.2,
        },
    },
    'X68000': {
        'label': "Sharp X68000 (1987)",
        'category': 'PLATFORM',
        'note': "512x512 in 65536 colours. The best-looking 16-bit home "
                "computer there was.",
        'settings': {
            'resolution_x': 512, 'resolution_y': 512,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'tex_filter': 'NEAREST',
            'color_depth': '16', 'dither': 'BAYER2',
            'shadows': True, 'shadow_default': 'MAP', 'gamma': 2.2,
        },
    },
    'WIN31': {
        'label': "Windows 3.1, 16 colour (1992)",
        'category': 'PLATFORM',
        'note': "640x480 on the VGA system palette. Every 1992 screenshot.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'NONE', 'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'EGA16', 'dither': 'BAYER8',
            'shadows': False, 'gamma': 2.2,
        },
    },
    'SVGA_HICOLOR': {
        'label': "SVGA High Colour (1995)",
        'category': 'PLATFORM',
        'note': "800x600 in 16-bit. The 1995 upgrade everyone saved up for.",
        'settings': {
            'resolution_x': 800, 'resolution_y': 600,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'tex_filter': 'BILINEAR',
            'color_depth': '16', 'dither': 'BAYER4', 'dither_strength': 0.5,
            'shadows': True, 'shadow_default': 'MAP', 'gamma': 2.2,
        },
    },

    # ----------------------------------------------------- more consoles
    'DREAMCAST': {
        'label': "Sega Dreamcast (1998)",
        'category': 'CONSOLE',
        'note': "640x480 with proper perspective correction, bilinear filtering "
                "and per-pixel fog. The end of the era.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'TRIANGLE',
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'BILINEAR', 'tex_mipmap': True, 'tex_max_size': 256,
            'tex_perspective': True, 'color_depth': '16', 'dither': 'BAYER2',
            'fog': True, 'fog_mode': 'EXP', 'fog_density': 0.02,
            'fog_color': (0.4, 0.45, 0.55), 'shadows': False,
            'max_lights': 8, 'transparency': 'SORTED', 'gamma': 2.2,
        },
    },
    'PS2': {
        'label': "PlayStation 2 (2000)",
        'category': 'CONSOLE',
        'note': "640x448 field-rendered with the GS's famous ordered dither, "
                "bilinear mipmaps that pop, edge antialias (the flicker "
                "filter), accumulation trails available, and bloom bleeding "
                "off the brights. The machine that made 'PS2 haze' a look.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 448,
            'aa_mode': 'EDGE', 'aa_edge_threshold': 0.08,
            'default_model': 'PHONG', 'shading_rate': 'PIXEL',
            'tex_filter': 'BILINEAR', 'tex_mipmap': True, 'tex_mip_bias': 0.5,
            'tex_max_size': 256, 'tex_perspective': True,
            'color_depth': '16', 'dither': 'BAYER4', 'dither_strength': 0.8,
            'interlace': 'FIELDS',
            'glow': True, 'glow_threshold': 0.8, 'glow_radius': 10.0,
            'glow_intensity': 0.5, 'glow_quality': 'BOX',
            'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 18.0,
            'fog_end': 90.0, 'fog_color': (0.45, 0.5, 0.58),
            'shadows': True, 'shadow_default': 'MAP', 'shadow_map_size': 512,
            'max_lights': 8, 'transparency': 'SORTED', 'gamma': 2.2,
        },
    },
    'GAMECUBE': {
        'label': "Nintendo GameCube (2001)",
        'category': 'CONSOLE',
        'note': "640x480 with clean trilinear mipmaps, per-pixel table fog "
                "with a height layer (the Flipper's fog unit), soft shadow "
                "maps and a gentle deflicker. The tidy one of the three.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'EDGE', 'aa_edge_threshold': 0.1,
            'default_model': 'PHONG', 'shading_rate': 'PIXEL',
            'tex_filter': 'TRILINEAR', 'tex_mipmap': True, 'tex_aniso': 2,
            'tex_max_size': 512, 'tex_perspective': True,
            'color_depth': '24', 'dither': 'NONE',
            'fog': True, 'fog_mode': 'TABLE16', 'fog_start': 12.0,
            'fog_end': 80.0, 'fog_color': (0.5, 0.55, 0.62),
            'fog_height': True, 'fog_height_top': 3.0,
            'fog_height_falloff': 0.4,
            'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 1024, 'shadow_softness': 2.0,
            'shadow_samples': 8,
            'max_lights': 8, 'transparency': 'SORTED', 'gamma': 2.2,
        },
    },
    'XBOX': {
        'label': "Microsoft Xbox (2001)",
        'category': 'CONSOLE',
        'note': "640x480 with trilinear plus anisotropy, per-pixel specular "
                "everywhere, big soft shadow maps, projected light textures "
                "on the spots (set an image on a lamp) and a hot bloom. The "
                "pixel-shader flex of the generation.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'BOX',
            'default_model': 'BLINN', 'shading_rate': 'PIXEL',
            'tex_filter': 'TRILINEAR', 'tex_mipmap': True, 'tex_aniso': 4,
            'tex_max_size': 1024, 'tex_perspective': True,
            'color_depth': '24', 'dither': 'NONE',
            'glow': True, 'glow_threshold': 0.9, 'glow_radius': 14.0,
            'glow_intensity': 0.6, 'glow_quality': 'GAUSS',
            'specular_in_gamma': True, 'clamp_specular': False,
            'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 1024, 'shadow_softness': 1.5,
            'shadow_samples': 8,
            'max_lights': 8, 'transparency': 'SORTED', 'gamma': 2.2,
        },
    },
    'THREEDO': {
        'label': "3DO Interactive (1993)",
        'category': 'CONSOLE',
        'note': "Cel-based hardware: warped textures, no z-buffer, 320x240 "
                "with visible seams between quads.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'NONE', 'default_model': 'FLAT', 'shading_rate': 'FACE',
            'vertex_snap': True, 'vertex_snap_grid': 1.0,
            'tex_filter': 'NEAREST', 'tex_perspective': False,
            'depth_sort': 'PAINTERS', 'color_depth': '15', 'dither': 'BAYER2',
            'shadows': False, 'max_lights': 2, 'backface_cull': True,
            'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'JAGUAR': {
        'label': "Atari Jaguar (1993)",
        'category': 'CONSOLE',
        'note': "Gouraud-shaded flat-lit polygons at 320x240, 16-bit, no "
                "texture filtering.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'NONE', 'default_model': 'GOURAUD',
            'shading_rate': 'VERTEX', 'tex_filter': 'NEAREST',
            'tex_perspective': False, 'color_depth': '16', 'dither': 'NONE',
            'shadows': False, 'max_lights': 2, 'backface_cull': True,
            'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'PSX_HIRES': {
        'label': "PlayStation high-res (1994)",
        'category': 'CONSOLE',
        'note': "512x240 mode: the same warping and snapping, twice the "
                "horizontal detail. Used for menus and FMV overlays.",
        'settings': {
            'resolution_x': 512, 'resolution_y': 240,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 2.0,
            'aa_mode': 'NONE', 'default_model': 'GOURAUD',
            'shading_rate': 'VERTEX', 'vertex_snap': True,
            'vertex_snap_grid': 1.0, 'subpixel_precision': 'INTEGER',
            'tex_filter': 'NEAREST', 'tex_perspective': False,
            'depth_sort': 'PAINTERS', 'painters_key': 'CENTROID',
            'transparency': 'SORTED',
            'color_depth': '15', 'dither': 'BAYER4', 'shadows': False,
            'max_lights': 4, 'backface_cull': True, 'gamma': 2.2,
        },
    },

    # --------------------------------------------------- more broadcast
    'VHS': {
        'label': "VHS tape (1976)",
        'category': 'BROADCAST',
        'note': "Third-generation dub: chroma smeared into next week, ringing, "
                "dot crawl and interlace.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'shadows': True,
            'color_depth': '24', 'composite': True, 'composite_bleed': 2.0,
            'composite_ringing': 1.0, 'composite_dot_crawl': 1.2,
            'interlace': 'BLEND', 'crt': True, 'crt_scanlines': 0.2,
            'crt_vignette': 0.4, 'crt_bloom': 0.3,
            'jpeg_artifacts': True, 'jpeg_quality': 45, 'jpeg_passes': 2,
            'saturation': 0.85, 'contrast': -0.1, 'gamma': 2.2,
        },
    },
    'SVIDEO': {
        'label': "S-Video (1987)",
        'category': 'BROADCAST',
        'note': "Luma and chroma kept apart: no dot crawl, only a little "
                "chroma softening. The good cable.",
        'settings': {
            'resolution_x': 720, 'resolution_y': 486,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'MITCHELL',
            'default_model': 'PHONG', 'shadows': True, 'shadow_default': 'MAP',
            'color_depth': '24', 'composite': True, 'composite_bleed': 0.4,
            'composite_ringing': 0.15, 'composite_dot_crawl': 0.0,
            'interlace': 'BLEND', 'crt': True, 'crt_scanlines': 0.1,
            'crt_mask': 'APERTURE', 'crt_mask_strength': 0.15, 'gamma': 2.2,
        },
    },

    # --------------------------------------------------------- more web
    'CD_ROM_FMV': {
        'label': "CD-ROM full-motion video (1992)",
        'category': 'WEB',
        'note': "Cinepak-era video: tiny, blocky, quantised and doubled up to "
                "fill the window.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256,
            'palette_method': 'OCTREE', 'dither': 'NONE',
            'jpeg_artifacts': True, 'jpeg_quality': 30, 'jpeg_passes': 2,
            'block_size': 4, 'shadows': True, 'gamma': 2.2,
            'output_scale': '2X', 'saturation': 0.9,
        },
    },
    'WEB_PNG8': {
        'label': "PNG-8 sprite (1996)",
        'category': 'WEB',
        'note': "Small adaptive-palette PNG with a hard alpha edge, as used "
                "for every rendered button on the early web.",
        'settings': {
            'resolution_x': 256, 'resolution_y': 256,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9,
            'default_model': 'BLINN', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 128,
            'palette_method': 'MEDIAN_CUT', 'dither': 'NONE',
            'film_transparent': True, 'alpha_bits': 1,
            'shadows': True, 'shadow_default': 'MAP', 'gamma': 2.2,
        },
    },

    # ------------------------------------------------------------- platforms
    'VGA_13H': {
        'label': "VGA Mode 13h (1987)",
        'category': 'PLATFORM',
        'note': "320x200 in 256 colours on a 1.2:1 pixel. The DOS demo look.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'NONE', 'aa_samples': 1,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'tex_perspective': False,
            'color_depth': '8', 'palette_mode': 'VGA256', 'dither': 'FLOYD',
            'max_lights': 4, 'shadows': False, 'gamma': 2.2,
            'output_scale': '3X',
        },
    },
    'MAC_8BIT': {
        'label': "Macintosh 8-bit (1987)",
        'category': 'PLATFORM',
        'note': "512x342 on the System palette, ordered dither, 1:1 pixels.",
        'settings': {
            'resolution_x': 512, 'resolution_y': 342,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'tex_filter': 'BILINEAR',
            'color_depth': '8', 'palette_mode': 'MAC256', 'dither': 'BAYER4',
            'gamma': 1.8, 'shadows': True, 'shadow_default': 'MAP',
            'output_scale': '2X',
        },
    },
    'WIN95': {
        'label': "Windows 95, 8-bit (1995)",
        'category': 'PLATFORM',
        'note': "640x480 with the 20 reserved system colours plus a halftone "
                "palette -- the look of a screenshot from 1996.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'NONE', 'default_model': 'GOURAUD',
            'shading_rate': 'VERTEX', 'tex_filter': 'NEAREST',
            'color_depth': '8', 'palette_mode': 'WEB216', 'dither': 'BAYER8',
            'gamma': 2.2, 'shadows': False,
        },
    },
    'EGA': {
        'label': "EGA 16 colour (1984)",
        'category': 'PLATFORM',
        'note': "The 16-colour IBM palette with heavy error diffusion. "
                "Almost all of the image is dither pattern.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'FLAT', 'shading_rate': 'FACE',
            'tex_filter': 'NEAREST', 'color_depth': '4',
            'palette_mode': 'EGA16', 'dither': 'STUCKI',
            'shadows': False, 'max_lights': 2, 'gamma': 2.2,
            'output_scale': '3X',
        },
    },
    'AMIGA_OCS': {
        'label': "Amiga OCS 32 colour (1985)",
        'category': 'PLATFORM',
        'note': "320x256 PAL, 32 colours from a 12-bit master palette.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 256,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'NEAREST', 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 32,
            'palette_method': 'MEDIAN_CUT', 'dither': 'FLOYD',
            'shadows': False, 'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'QUAKE_SW': {
        'label': "Quake software renderer (1996)",
        'category': 'PLATFORM',
        'note': "Affine-ish texture mapping on a 256-colour palette with "
                "no filtering and heavy light-map style falloff.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 200,
            'pixel_aspect_x': 1.0, 'pixel_aspect_y': 1.2,
            'aa_mode': 'NONE', 'default_model': 'LAMBERT',
            'tex_filter': 'NEAREST', 'tex_perspective': True,
            'tex_affine_subdiv': 16, 'color_depth': '8',
            'palette_mode': 'ADAPTIVE', 'palette_size': 256, 'dither': 'NONE',
            'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 4.0,
            'fog_end': 30.0, 'fog_color': (0.05, 0.05, 0.06),
            'shadows': False, 'max_lights': 4, 'gamma': 2.2,
            'output_scale': '3X',
        },
    },

    # -------------------------------------------------------------- consoles
    'PSX': {
        'label': "PlayStation (1994)",
        'category': 'CONSOLE',
        'note': "Integer vertex snapping, affine texture warp, no z-buffer "
                "sorting, 15-bit colour with ordered dither.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'NONE', 'default_model': 'GOURAUD',
            'shading_rate': 'VERTEX', 'vertex_snap': True,
            'vertex_snap_grid': 1.0, 'subpixel_precision': 'INTEGER',
            'tex_filter': 'NEAREST', 'tex_perspective': False,
            'depth_sort': 'PAINTERS', 'painters_key': 'CENTROID',
            'transparency': 'SORTED',
            'color_depth': '15', 'dither': 'BAYER4', 'dither_strength': 1.0,
            'shadows': False, 'max_lights': 4, 'backface_cull': True,
            'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'SATURN': {
        'label': "Sega Saturn (1994)",
        'category': 'CONSOLE',
        'note': "Quad-based renderer: flat-ish shading, no perspective "
                "correction, 15-bit output, visible seams.",
        'settings': {
            'resolution_x': 352, 'resolution_y': 240,
            'aa_mode': 'NONE', 'default_model': 'FLAT', 'shading_rate': 'FACE',
            'vertex_snap': True, 'vertex_snap_grid': 1.0,
            'tex_filter': 'NEAREST', 'tex_perspective': False,
            'depth_sort': 'PAINTERS', 'color_depth': '15', 'dither': 'NONE',
            'shadows': False, 'max_lights': 2, 'backface_cull': True,
            'gamma': 2.2, 'output_scale': '3X',
        },
    },
    'N64': {
        'label': "Nintendo 64 (1996)",
        'category': 'CONSOLE',
        'note': "Three-point filtered textures at 64x64, aggressive fog, "
                "16-bit framebuffer with the RDP's dither.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'TRIANGLE',
            'default_model': 'GOURAUD', 'shading_rate': 'VERTEX',
            'tex_filter': 'N64_3POINT', 'tex_max_size': 64,
            'tex_perspective': True, 'tex_mipmap': True,
            'color_depth': '16', 'dither': 'BAYER2',
            'fog': True, 'fog_mode': 'LINEAR', 'fog_start': 6.0,
            'fog_end': 26.0, 'fog_color': (0.35, 0.42, 0.55),
            'shadows': False, 'max_lights': 4, 'gamma': 2.2,
            'output_scale': '3X',
        },
    },
    'VOODOO': {
        'label': "3dfx Voodoo Graphics (1996)",
        'category': 'CONSOLE',
        'note': "Bilinear filtering, 16-bit colour with the 22-bit "
                "post-filter, table fog. The 1997 accelerated look.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'aa_mode': 'NONE', 'default_model': 'GOURAUD',
            'shading_rate': 'VERTEX', 'tex_filter': 'BILINEAR',
            'tex_mipmap': True, 'tex_max_size': 256,
            'color_depth': '16', 'dither': 'BAYER4', 'dither_strength': 0.6,
            'fog': True, 'fog_mode': 'TABLE16', 'fog_start': 8.0,
            'fog_end': 40.0, 'shadows': False, 'max_lights': 8,
            'transparency': 'SORTED', 'gamma': 2.2,
        },
    },

    # ------------------------------------------------------------- broadcast
    'TOASTER': {
        'label': "Video Toaster / NTSC (1990)",
        'category': 'BROADCAST',
        'note': "D1 NTSC with non-square pixels, composite chroma bleed, "
                "interlace and a CRT.",
        'settings': {
            'resolution_x': 720, 'resolution_y': 486,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4, 'aa_filter': 'MITCHELL',
            'default_model': 'PHONG', 'shadows': True, 'shadow_default': 'MAP',
            'color_depth': '24', 'composite': True, 'composite_bleed': 1.0,
            'composite_ringing': 0.45, 'composite_dot_crawl': 0.4,
            'interlace': 'BLEND', 'crt': True, 'crt_scanlines': 0.15,
            'crt_mask': 'APERTURE', 'crt_mask_strength': 0.2,
            'crt_vignette': 0.25, 'gamma': 2.2, 'glow': True,
            'glow_intensity': 0.3,
        },
    },
    'SGI_BROADCAST': {
        'label': "SGI broadcast CGI (1994)",
        'category': 'BROADCAST',
        'note': "The Saturday-morning television pipeline: big-iron SGI "
                "frames at D1 NTSC, plastic Phong characters, hard map "
                "shadows, broadcast-legal colour and a light interlace "
                "blend. ReBoot on your engine.",
        'settings': {
            'resolution_x': 720, 'resolution_y': 486,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 9,
            'aa_filter': 'CATROM',
            'default_model': 'PHONG', 'specular_in_gamma': True,
            'shadows': True, 'shadow_default': 'MAP',
            'shadow_map_size': 1024, 'shadow_softness': 1.0,
            'shadow_samples': 4,
            'global_ambient': (0.10, 0.10, 0.12),
            'color_depth': '24', 'interlace': 'BLEND',
            'gamma': 2.2, 'saturation': 0.9, 'dither': 'NONE',
        },
    },
    'PAL_TV': {
        'label': "PAL broadcast (1967)",
        'category': 'BROADCAST',
        'note': "720x576 with PAL pixel aspect and a softer composite.",
        'settings': {
            'resolution_x': 720, 'resolution_y': 576,
            'pixel_aspect_x': 59.0, 'pixel_aspect_y': 54.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'shadows': True,
            'color_depth': '24', 'composite': True, 'composite_bleed': 0.7,
            'composite_ringing': 0.3, 'interlace': 'BLEND',
            'crt': True, 'crt_scanlines': 0.12, 'crt_vignette': 0.2,
            'gamma': 2.2,
        },
    },

    # ------------------------------------------------------------------ web
    'WEB_GIF': {
        'label': "Web-safe GIF, 216 colours (1996)",
        'category': 'WEB',
        'note': "The 216-colour browser palette with ordered dither. "
                "Every 1997 splash page.",
        'settings': {
            'resolution_x': 400, 'resolution_y': 300,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'tex_filter': 'BILINEAR',
            'color_depth': '8', 'palette_mode': 'WEB216', 'dither': 'BAYER4',
            'shadows': True, 'shadow_default': 'MAP', 'gamma': 2.2,
        },
    },
    'WEB_JPEG': {
        'label': "Early web JPEG (1995)",
        'category': 'WEB',
        'note': "Small, over-compressed, heavily blocked -- 28.8k modem era.",
        'settings': {
            'resolution_x': 320, 'resolution_y': 240,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'PHONG', 'color_depth': '24',
            'jpeg_artifacts': True, 'jpeg_quality': 22, 'jpeg_passes': 2,
            'shadows': True, 'gamma': 2.2, 'output_scale': '2X',
        },
    },

    # ------------------------------------------------------- cel & film
    # R230: the era looks -- the cel photographed and printed. Each entry
    # sets the ink, the film stages and the shooting rhythm of a period;
    # the materials (Cartoon Shader eras, the Anime Shader's 80s dials)
    # come from the Pre-Made shelf. Pixel-rate cel shading throughout.
    'CEL_MONO_30S': {
        'label': "Cel: 1930s black-and-white",
        'category': 'CEL',
        'note': "The panchromatic short: b/w stock, a soft optical print "
                "that weaves in the gate, dust and grain on every frame, "
                "a flickering lamp, shot on twos. Brush ink with a boil.",
        'settings': {
            'resolution_x': 960, 'resolution_y': 720,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 3, 'ink_style': 'BRUSH',
            'ink_taper': 0.4, 'ink_weight_noise': 0.35, 'ink_boil': 1.2,
            'ink_boil_fps': 12,
            # R231: the drawn line -- the 30s brush lifts and lands, gets
            # rough in patches, skips on the dry bits
            'ink_end_taper': 0.55, 'ink_end_length': 14.0,
            'ink_roughness': 0.5, 'ink_roughness_scale': 6.0,
            'ink_drift': 0.8, 'ink_gaps': 0.15,
            'ink_texture': 'STREAKS', 'ink_texture_amount': 0.35,
            # R234: the inker's line -- heavy on the shadowed flank, the
            # pen pausing at corners and running past them, the hand's
            # noises fixed to the drawing
            'ink_isophote': 0.55, 'ink_isophote_range': 24.0,
            'ink_smooth': 1.0, 'ink_pressure': 0.35, 'ink_overshoot': 5.0,
            'ink_anchor': 'SURFACE',
            'film_grade': 'MONO', 'film_grade_amount': 1.0,
            'film_softness': 0.9, 'film_weave': 0.6, 'film_dust': 0.45,
            'film_grain': 0.35, 'film_flicker': 0.12, 'film_hold': 2,
            # R237: one black-and-white negative's grain, coarse and
            # clumped; a print that has been through many gates
            'film_grain_size': 1.6, 'film_grain_clump': 0.4,
            'film_grain_chroma': 0.0, 'film_dust_size': 1.8,
            'film_dust_negative': 0.3, 'film_dust_cel': 0.3,
            'film_hairs': 0.3, 'film_hair_length': 110.0,
            'film_hair_width': 1.6, 'film_hair_hold': 14,
            'film_scratches': 0.4, 'film_scratch_width': 1.4,
            'film_scratch_hold': 72, 'film_scratch_side': 'BASE',
            'film_misregister': 0.6, 'film_bleed': 0.3,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_TECHNICOLOR_40S': {
        'label': "Cel: 1940s Technicolor feature",
        'category': 'CEL',
        'note': "The three-strip print: purified saturated primaries, a "
                "gentle optical softness, light gate weave and grain, a "
                "little dust, shot on twos. Brush ink with taper.",
        'settings': {
            'resolution_x': 1440, 'resolution_y': 1080,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 3, 'ink_style': 'BRUSH',
            'ink_taper': 0.45, 'ink_weight_noise': 0.3,
            # R231: the drawn line
            'ink_end_taper': 0.6, 'ink_end_length': 14.0,
            'ink_roughness': 0.4, 'ink_roughness_scale': 6.0,
            'ink_drift': 0.7, 'ink_gaps': 0.1,
            'ink_texture': 'STREAKS', 'ink_texture_amount': 0.4,
            # R234: the inker's line
            'ink_isophote': 0.6, 'ink_isophote_range': 28.0,
            'ink_smooth': 1.2, 'ink_pressure': 0.4, 'ink_overshoot': 6.0,
            'ink_anchor': 'SURFACE',
            # R236: the process itself, not a grade -- three records, the
            # dye-transfer print with its impurities and key, the dye
            # layers a hair off register, halation in the negative
            'film_process': 'THREE_STRIP', 'film_process_amount': 1.0,
            'film_exposure': -0.1, 'film_gamma': 1.35, 'film_density': 2.4,
            'film_filters': 0.3, 'film_dye_purity': 0.5, 'film_key': 0.3,
            'film_halation': 0.12, 'film_halation_radius': 14.0,
            'film_register': 0.5,
            'film_grade': 'NONE', 'film_grade_amount': 1.0,
            'film_softness': 0.6, 'film_weave': 0.3, 'film_dust': 0.2,
            'film_grain': 0.2, 'film_flicker': 0.04, 'film_hold': 2,
            # R237: three records' own grain, fine; dust on the cel
            # under the rostrum camera; a hair now and then
            'film_grain_size': 1.1, 'film_grain_clump': 0.15,
            'film_grain_chroma': 0.7, 'film_dust_size': 1.4,
            'film_dust_negative': 0.3, 'film_dust_cel': 0.35,
            'film_hairs': 0.12, 'film_hair_length': 90.0,
            'film_hair_width': 1.4, 'film_hair_hold': 12,
            'film_scratches': 0.15, 'film_scratch_width': 1.1,
            'film_scratch_hold': 60, 'film_scratch_side': 'EMULSION',
            'film_misregister': 0.0, 'film_bleed': 0.25,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_FLEISCHER_41': {
        'label': "Cel: 1941 Fleischer (Superman)",
        'category': 'CEL',
        'note': "The Famous / Fleischer three-strip short: a hard, deep "
                "print -- heavy silver key, high dye density, the records "
                "held a third of a stop under -- halation glowing off the "
                "lit deco planes, the dye layers a hair off, shot on twos. "
                "Pair with the Cartoon Shader's Golden Age era and dark "
                "paints.",
        'settings': {
            'resolution_x': 1440, 'resolution_y': 1080,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 4, 'ink_style': 'BRUSH',
            'ink_taper': 0.4, 'ink_weight_noise': 0.25,
            'ink_end_taper': 0.5, 'ink_end_length': 16.0,
            'ink_roughness': 0.3, 'ink_roughness_scale': 6.0,
            'ink_drift': 0.5, 'ink_gaps': 0.05,
            'ink_texture': 'STREAKS', 'ink_texture_amount': 0.3,
            'ink_isophote': 0.75, 'ink_isophote_range': 32.0,
            'ink_smooth': 1.2, 'ink_pressure': 0.45, 'ink_overshoot': 5.0,
            'ink_anchor': 'SURFACE',
            'film_process': 'THREE_STRIP', 'film_process_amount': 1.0,
            'film_exposure': -0.3, 'film_gamma': 1.5, 'film_density': 2.8,
            'film_filters': 0.35, 'film_dye_purity': 0.5, 'film_key': 0.5,
            'film_halation': 0.25, 'film_halation_radius': 18.0,
            'film_register': 0.6,
            'film_grade': 'NONE',
            'film_softness': 0.5, 'film_weave': 0.35, 'film_dust': 0.25,
            'film_grain': 0.25, 'film_flicker': 0.05, 'film_hold': 2,
            # R237: the dense print's grain a touch coarser, the key's
            # silver in it; dust on the cel and the setback glass
            'film_grain_size': 1.3, 'film_grain_clump': 0.25,
            'film_grain_chroma': 0.6, 'film_dust_size': 1.5,
            'film_dust_negative': 0.25, 'film_dust_cel': 0.4,
            'film_hairs': 0.15, 'film_hair_length': 100.0,
            'film_hair_width': 1.5, 'film_hair_hold': 12,
            'film_scratches': 0.2, 'film_scratch_width': 1.2,
            'film_scratch_hold': 72, 'film_scratch_side': 'EMULSION',
            'film_bleed': 0.2,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_CINECOLOR_40S': {
        'label': "Cel: 1940s Cinecolor short",
        'category': 'CEL',
        'note': "The two-colour bipack process on a duplitized print: "
                "red-orange and blue-green records, no true green or "
                "violet -- skies cyan, foliage olive, skin salmon -- the "
                "two sides of the print a hair off register, a softer "
                "grainier print than Technicolor, shot on twos.",
        'settings': {
            'resolution_x': 1440, 'resolution_y': 1080,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 3, 'ink_style': 'BRUSH',
            'ink_taper': 0.35, 'ink_weight_noise': 0.25,
            'ink_end_taper': 0.5, 'ink_end_length': 14.0,
            'ink_roughness': 0.35, 'ink_drift': 0.6, 'ink_gaps': 0.1,
            'ink_texture': 'STREAKS', 'ink_texture_amount': 0.35,
            'ink_isophote': 0.5, 'ink_smooth': 1.0, 'ink_pressure': 0.3,
            'ink_overshoot': 4.0, 'ink_anchor': 'SURFACE',
            'film_process': 'TWO_COLOUR', 'film_process_amount': 1.0,
            'film_exposure': -0.1, 'film_gamma': 1.3, 'film_density': 2.2,
            'film_filters': 0.3, 'film_dye_purity': 0.0, 'film_key': 0.0,
            'film_halation': 0.15, 'film_halation_radius': 14.0,
            'film_register': 0.8,
            'film_grade': 'NONE',
            'film_softness': 0.8, 'film_weave': 0.4, 'film_dust': 0.3,
            'film_grain': 0.3, 'film_flicker': 0.06, 'film_hold': 2,
            # R237: two records' own grain on the duplitized print, its
            # emulsion scratches one side's colour
            'film_grain_size': 1.4, 'film_grain_clump': 0.3,
            'film_grain_chroma': 0.8, 'film_dust_size': 1.6,
            'film_dust_negative': 0.35, 'film_dust_cel': 0.3,
            'film_hairs': 0.2, 'film_hair_length': 100.0,
            'film_hair_width': 1.5, 'film_hair_hold': 12,
            'film_scratches': 0.3, 'film_scratch_width': 1.3,
            'film_scratch_hold': 72, 'film_scratch_side': 'EMULSION',
            'film_bleed': 0.25,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_TWO_STRIP_30S': {
        'label': "Cel: early-30s two-strip Technicolor",
        'category': 'CEL',
        'note': "The 1930-32 two-colour cartoon (the ComiColor and the "
                "first Silly Symphonies before three-strip): a bright, "
                "thin two-dye print, the whites open, everything else "
                "red-orange and blue-green, a soft weaving print with "
                "dust and grain, shot on twos.",
        'settings': {
            'resolution_x': 960, 'resolution_y': 720,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 3, 'ink_style': 'BRUSH',
            'ink_taper': 0.4, 'ink_weight_noise': 0.3, 'ink_boil': 1.0,
            'ink_boil_fps': 12,
            'ink_end_taper': 0.55, 'ink_end_length': 14.0,
            'ink_roughness': 0.45, 'ink_drift': 0.8, 'ink_gaps': 0.15,
            'ink_texture': 'STREAKS', 'ink_texture_amount': 0.35,
            'ink_isophote': 0.5, 'ink_smooth': 1.0, 'ink_pressure': 0.35,
            'ink_overshoot': 5.0, 'ink_anchor': 'SURFACE',
            'film_process': 'TWO_COLOUR', 'film_process_amount': 1.0,
            'film_exposure': 0.0, 'film_gamma': 1.2, 'film_density': 2.0,
            'film_filters': 0.2, 'film_dye_purity': 0.0, 'film_key': 0.0,
            'film_halation': 0.2, 'film_halation_radius': 16.0,
            'film_register': 0.9,
            'film_grade': 'NONE',
            'film_softness': 1.0, 'film_weave': 0.6, 'film_dust': 0.5,
            'film_grain': 0.4, 'film_flicker': 0.1, 'film_hold': 2,
            # R237: a surviving print -- coarse clumped grain, dirt,
            # hairs, base scratches the length of the reel
            'film_grain_size': 1.8, 'film_grain_clump': 0.45,
            'film_grain_chroma': 0.7, 'film_dust_size': 2.0,
            'film_dust_negative': 0.3, 'film_dust_cel': 0.25,
            'film_hairs': 0.35, 'film_hair_length': 120.0,
            'film_hair_width': 1.7, 'film_hair_hold': 16,
            'film_scratches': 0.5, 'film_scratch_width': 1.5,
            'film_scratch_hold': 96, 'film_scratch_side': 'BASE',
            'film_bleed': 0.3,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_TV_70S': {
        'label': "Cel: 1970s Saturday morning",
        'category': 'CEL',
        'note': "Limited television animation on a faded syndication "
                "print: the Eastmancolor cast, a soft print, shot on "
                "threes, then the composite broadcast chain -- chroma "
                "bleed, a CRT. Clean xeroxed ink.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 2, 'ink_style': 'CLEAN',
            'ink_grain': 0.25,
            # R231: the xeroxed line drops out in patches
            'ink_roughness': 0.3, 'ink_roughness_scale': 4.0,
            'ink_gaps': 0.12,
            'film_grade': 'EASTMAN_70S', 'film_grade_amount': 1.0,
            'film_softness': 0.7, 'film_weave': 0.25, 'film_dust': 0.15,
            'film_grain': 0.2, 'film_flicker': 0.03, 'film_hold': 3,
            # R237: a 16 mm print on the telecine -- coarser coloured
            # grain, the odd emulsion scratch
            'film_grain_size': 1.8, 'film_grain_clump': 0.3,
            'film_grain_chroma': 0.8, 'film_dust_size': 1.6,
            'film_dust_negative': 0.2, 'film_dust_cel': 0.2,
            'film_hairs': 0.08, 'film_hair_length': 80.0,
            'film_hair_width': 1.4, 'film_hair_hold': 10,
            'film_scratches': 0.12, 'film_scratch_width': 1.1,
            'film_scratch_hold': 48, 'film_scratch_side': 'EMULSION',
            'film_misregister': 0.5, 'film_bleed': 0.2,
            'composite': True, 'composite_bleed': 1.2,
            'composite_ringing': 0.4, 'crt': True, 'crt_scanlines': 0.15,
            'crt_vignette': 0.25, 'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_OVA_80S': {
        'label': "Cel: 1980s OVA (LaserDisc)",
        'category': 'CEL',
        'note': "The straight-to-video anime on a clean disc: the telecine "
                "grade, a touch of optical softness and grain, the gate "
                "barely moving, shot on twos. A thin clean trace line -- "
                "pair with the Anime Shader's 80s dials.",
        'settings': {
            'resolution_x': 960, 'resolution_y': 720,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'ANIME', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 2, 'ink_style': 'CLEAN',
            'ink_reference_height': 720,
            'film_grade': 'TV_80S', 'film_grade_amount': 1.0,
            'film_softness': 0.45, 'film_weave': 0.15, 'film_dust': 0.06,
            'film_grain': 0.14, 'film_flicker': 0.0, 'film_hold': 2,
            # R237: a clean 35 mm negative on the telecine -- fine grain
            'film_grain_size': 1.0, 'film_grain_clump': 0.1,
            'film_grain_chroma': 0.6, 'film_dust_size': 1.2,
            'film_dust_negative': 0.5, 'film_dust_cel': 0.15,
            'film_hairs': 0.0, 'film_scratches': 0.0,
            'film_misregister': 0.3, 'film_bleed': 0.15,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_VHS_80S': {
        'label': "Cel: 1980s TV anime (VHS)",
        'category': 'CEL',
        'note': "The broadcast episode taped off air: the VHS grade under "
                "the composite chain, soft, grainy, shot on threes with "
                "the dust of a well-played tape's print.",
        'settings': {
            'resolution_x': 640, 'resolution_y': 480,
            'pixel_aspect_x': 10.0, 'pixel_aspect_y': 11.0,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'ANIME', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 2, 'ink_style': 'CLEAN',
            'film_grade': 'VHS', 'film_grade_amount': 1.0,
            'film_softness': 1.0, 'film_weave': 0.2, 'film_dust': 0.1,
            'film_grain': 0.25, 'film_flicker': 0.05, 'film_hold': 3,
            # R237: the film's grain under the tape's own noise
            'film_grain_size': 1.4, 'film_grain_clump': 0.2,
            'film_grain_chroma': 0.5, 'film_dust_size': 1.4,
            'film_dust_negative': 0.3, 'film_dust_cel': 0.1,
            'film_hairs': 0.05, 'film_hair_length': 80.0,
            'film_hair_width': 1.4, 'film_hair_hold': 10,
            'film_scratches': 0.1, 'film_scratch_width': 1.1,
            'film_scratch_hold': 48, 'film_scratch_side': 'EMULSION',
            'film_misregister': 0.4, 'film_bleed': 0.2,
            'composite': True, 'composite_bleed': 2.0,
            'composite_ringing': 0.8, 'composite_dot_crawl': 0.8,
            'interlace': 'BLEND', 'crt': True, 'crt_scanlines': 0.2,
            'crt_vignette': 0.35, 'color_depth': '24', 'gamma': 2.2,
        },
    },
    # R240: five more decades, both traditions
    'CEL_SILENT_20S': {
        'label': "Cel: 1920s silent (worn nitrate)",
        'category': 'CEL',
        'note': "The rubber-hose short on a print that has run a "
                "thousand times: black-and-white, heavy weave and "
                "flicker, coarse clumped grain, dirt and hairs and "
                "base scratches everywhere, the projectionist's cue "
                "discs at every reel's end. A heavy boiling brush "
                "line. Pair with flat grey paints and the Saturday "
                "Morning era (no shadow tone -- the 20s painted "
                "none).",
        'settings': {
            'resolution_x': 768, 'resolution_y': 576,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': False,
            'outline': True, 'outline_width': 4, 'ink_style': 'BRUSH',
            'ink_taper': 0.35, 'ink_weight_noise': 0.45,
            'ink_boil': 1.6, 'ink_boil_fps': 12,
            'ink_end_taper': 0.5, 'ink_end_length': 12.0,
            'ink_roughness': 0.6, 'ink_roughness_scale': 5.0,
            'ink_drift': 1.0, 'ink_gaps': 0.2,
            'ink_texture': 'STREAKS', 'ink_texture_amount': 0.4,
            'ink_smooth': 0.8, 'ink_pressure': 0.3,
            'ink_overshoot': 4.0, 'ink_anchor': 'SURFACE',
            'film_grade': 'MONO', 'film_grade_amount': 1.0,
            'film_softness': 1.1, 'film_weave': 0.9, 'film_dust': 0.7,
            'film_grain': 0.45, 'film_flicker': 0.22, 'film_hold': 2,
            'film_grain_size': 2.0, 'film_grain_clump': 0.5,
            'film_grain_chroma': 0.0, 'film_dust_size': 2.0,
            'film_dust_negative': 0.35, 'film_dust_cel': 0.4,
            'film_hairs': 0.5, 'film_hair_length': 130.0,
            'film_hair_width': 1.8, 'film_hair_hold': 10,
            'film_scratches': 0.6, 'film_scratch_width': 1.6,
            'film_scratch_hold': 36, 'film_scratch_side': 'BASE',
            'film_reel': 15.0,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_ANIME_MOVIE_80S': {
        'label': "Cel: 1980s anime feature (35mm)",
        'category': 'CEL',
        'note': "The theatrical anime feature on a well-kept 35 mm "
                "print: fine coloured grain, a gentle optical "
                "softness, the gate barely moving, the telecine's "
                "slight warmth, shot on twos. A thin steady trace "
                "line. Pair with the Anime Shader's 80s Film Feature "
                "style.",
        'settings': {
            'resolution_x': 1440, 'resolution_y': 1080,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'ANIME', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 2, 'ink_style': 'CLEAN',
            'ink_reference_height': 1080, 'ink_taper': 0.2,
            'film_grade': 'TV_80S', 'film_grade_amount': 0.4,
            'film_softness': 0.5, 'film_weave': 0.12, 'film_dust': 0.08,
            'film_grain': 0.16, 'film_flicker': 0.02, 'film_hold': 2,
            'film_grain_size': 1.0, 'film_grain_clump': 0.1,
            'film_grain_chroma': 0.6, 'film_dust_size': 1.2,
            'film_dust_negative': 0.4, 'film_dust_cel': 0.2,
            'film_hairs': 0.04, 'film_hair_length': 80.0,
            'film_hair_width': 1.3, 'film_hair_hold': 12,
            'film_scratches': 0.06, 'film_scratch_width': 1.0,
            'film_scratch_hold': 96, 'film_scratch_side': 'EMULSION',
            'film_misregister': 0.25, 'film_bleed': 0.2,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_DIGITAL_00S': {
        'label': "Cel: 2000s digital ink and paint",
        'category': 'CEL',
        'note': "The digital transition: no film in the chain at all "
                "-- no grain, no weave, no dust, dead-hard paint "
                "edges under a clean thin line, held on threes for "
                "television, only a slight master softness from the "
                "SD finish. Pair with the Anime Shader's 2000s "
                "Digital style.",
        'settings': {
            'resolution_x': 960, 'resolution_y': 720,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'ANIME', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 2, 'ink_style': 'CLEAN',
            'ink_reference_height': 720,
            'film_softness': 0.25, 'film_hold': 3,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_ANIME_MODERN': {
        'label': "Cel: modern digital anime (1080p)",
        'category': 'CEL',
        'note': "The current pipeline: a clean 1080p master with the "
                "compositor's faint film-emulation grain laid over "
                "it (one fine monochrome sheet), shot on twos, a "
                "thin resolution-true line. Pair with the Anime "
                "Shader's Modern style -- the camera key, the depth "
                "rim and the marched contact shadows live there.",
        'settings': {
            'resolution_x': 1920, 'resolution_y': 1080,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'ANIME', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 2, 'ink_style': 'CLEAN',
            'ink_reference_height': 1080,
            'film_grain': 0.06, 'film_grain_size': 0.8,
            'film_grain_clump': 0.0, 'film_grain_chroma': 0.0,
            'film_hold': 2,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'CEL_FLAT_10S': {
        'label': "Cartoon: modern flat TV (1080p)",
        'category': 'CEL',
        'note': "The modern flat TV cartoon: a crisp digital 1080p "
                "frame, a thin dead-even vector line (no taper, no "
                "noise, no boil), nothing filmed anywhere in the "
                "chain, shot on twos. Pair with the Cartoon Shader's "
                "Modern Flat era and bright paints.",
        'settings': {
            'resolution_x': 1920, 'resolution_y': 1080,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 3, 'ink_style': 'CLEAN',
            'ink_reference_height': 1080,
            'film_hold': 2,
            'color_depth': '24', 'gamma': 2.2,
        },
    },
    'COMIC_PRINT': {
        'label': "Comic print (Ben-Day dots)",
        'category': 'CEL',
        'note': "The newsstand page: four-colour dot screens at the classic "
                "angles, the paint slipping off register under a heavy "
                "clean line, a little bleed. Not film: shot on ones, no "
                "grain, no weave.",
        'settings': {
            'resolution_x': 1200, 'resolution_y': 900,
            'aa_mode': 'SUPERSAMPLE', 'aa_samples': 4,
            'default_model': 'CARTOON', 'shading_rate': 'PIXEL',
            'shadows': True, 'shadow_default': 'MAP',
            'outline': True, 'outline_width': 4, 'ink_style': 'CLEAN',
            # R231: the comic inker's pen -- thick-and-thin, ends lifted
            'ink_end_taper': 0.7, 'ink_end_length': 18.0,
            'ink_weight_noise': 0.3, 'ink_drift': 0.5,
            'film_halftone': 1.0, 'film_halftone_pitch': 5.0,
            'film_misregister': 1.5, 'film_bleed': 0.4,
            'film_hold': 1, 'color_depth': '24', 'gamma': 2.2,
        },
    },
}


# The device family: where the frame computes is a property of the person's
# machine, never of a look. A preset describes what a 1996 renderer drew, and
# that picture is identical on either device -- so selecting one must not
# flip the CPU/GPU switch or any of its Debug toggles. Kept as its own set
# because apply_preset also refuses these keys from preset dicts by name:
# preserving them from the reset is not enough if a future preset entry
# were to list one.
DEVICE_KEYS = frozenset({
    'render_device', 'gpu_post', 'gpu_shading', 'gpu_raster',
    'gpu_hold_context', 'gpu_scissor', 'layer_gpu_min_frac',
})

# Machine and pipeline settings, which describe the computer or the output
# plumbing rather than the look. A preset has no business resetting these.
PRESERVED = frozenset({
    'threads', 'preview_scale', 'progressive',
    'show_stats', 'debug_pass', 'seed',
    'film_transparent', 'use_processes', 'process_count',
}) | DEVICE_KEYS


def reset_settings(settings, preserve=PRESERVED):
    """Return every field to its dataclass default, bar the preserved ones."""
    import dataclasses
    from ..core.settings import RenderSettings
    fresh = RenderSettings()
    for f in dataclasses.fields(RenderSettings):
        if f.name in preserve:
            continue
        setattr(settings, f.name, getattr(fresh, f.name))
    return settings


def apply_preset(settings, key, reset=True, preserve=PRESERVED):
    """Apply a preset, resetting to defaults first.

    Without the reset, presets accumulate: going from EGA to Infini-D used to
    leave EGA's 16-colour palette, its 2-light limit, its 1.2 pixel aspect and
    its 3x scale behind, because Infini-D's entry does not mention any of them.
    Every preset is now a complete description of a look rather than a patch on
    whatever came before.
    """
    p = PRESETS.get(key)
    if not p:
        return settings
    if reset:
        reset_settings(settings, preserve)
    for k, v in p['settings'].items():
        if k in DEVICE_KEYS:
            continue                    # a look never chooses the device
        if hasattr(settings, k):
            setattr(settings, k, v)
    return settings


def preset_items():
    """(identifier, label, description) tuples grouped for a Blender EnumProperty."""
    out = []
    for cat, cat_label in CATEGORIES:
        members = [(k, v) for k, v in PRESETS.items() if v['category'] == cat]
        if not members:
            continue
        out.append(None)
        out.append(('', cat_label, ''))
        for k, v in sorted(members, key=lambda kv: kv[1]['label']):
            out.append((k, v['label'], v['note']))
    return [o for o in out if o is not None]


def list_presets():
    return sorted(PRESETS.keys())
