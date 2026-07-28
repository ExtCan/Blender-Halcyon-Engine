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
    'INFINID_4': {
        'label': "Specular Infini-D 4",
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
        'label': "Ray Dream Studio 5",
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
        'label': "Strata StudioPro 1.75",
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
        'label': "3D Studio MAX R2",
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
    'STUDIO_R4': {
        'label': "3D Studio R4 (DOS)",
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
        'label': "trueSpace 2",
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
        'label': "LightWave 5.6",
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
        'label': "Imagine 3.0 (Amiga)",
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
        'label': "POV-Ray 3.1",
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
        'label': "Bryce 2",
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

    # ---------------------------------------------------- more software
    'ELECTRIC_IMAGE': {
        'label': "ElectricImage 2.9",
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
        'label': "Softimage|3D",
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
        'label': "Alias PowerAnimator",
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
        'label': "Wavefront Advanced Visualizer",
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
        'label': "CINEMA 4D v4",
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
        'label': "Real 3D 2 (Amiga)",
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
        'label': "Vistapro",
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
        'label': "Hash Animation:Master",
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
        'label': "POV-Ray 2.2",
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
        'label': "Vue d'Esprit 2",
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
        'label': "Atari ST",
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
    'AMIGA_AGA': {
        'label': "Amiga AGA 256",
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
        'label': "CGA 4 colour",
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
        'label': "Hercules mono",
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
        'label': "Macintosh 1-bit",
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
        'label': "NEC PC-98",
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
        'label': "Sharp X68000",
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
        'label': "Windows 3.1 (16 colour)",
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
        'label': "SVGA High Colour",
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
        'label': "Sega Dreamcast",
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
    'THREEDO': {
        'label': "3DO Interactive",
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
        'label': "Atari Jaguar",
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
        'label': "PlayStation (high-res)",
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
        'label': "VHS tape",
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
        'label': "S-Video",
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
        'label': "CD-ROM full-motion video",
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
        'label': "PNG-8 sprite",
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
        'label': "VGA Mode 13h",
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
        'label': "Macintosh 8-bit",
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
        'label': "Windows 95 (8-bit)",
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
        'label': "EGA 16 colour",
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
        'label': "Amiga OCS 32 colour",
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
        'label': "Quake software renderer",
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
        'label': "PlayStation",
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
        'label': "Sega Saturn",
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
        'label': "Nintendo 64",
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
        'label': "3dfx Voodoo Graphics",
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
        'label': "Video Toaster / NTSC",
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
    'PAL_TV': {
        'label': "PAL broadcast",
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
        'label': "Web-safe GIF (216)",
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
        'label': "Early web JPEG",
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
}


# Machine and pipeline settings, which describe the computer or the output
# plumbing rather than the look. A preset has no business resetting these.
PRESERVED = frozenset({
    'threads', 'tile_size', 'bucket_order', 'preview_scale', 'progressive',
    'show_stats', 'debug_pass', 'seed', 'jitter_seed', 'max_texture_memory',
    'film_transparent', 'use_processes', 'process_count',
})


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
