"""Sky presets for the Bryce atmosphere, and the format they save in.

Bryce's Sky & Fog palette kept a library of skies and a row of memory dots to
drop them onto, and half of using Bryce was starting from one of those and
pushing it somewhere. A Sky Lab with sixty controls and no library is a Sky Lab
nobody opens twice.

**What these are.** Skies built out of Bryce's own controls, tuned to the
conditions they are named for. **What they are not:** Bryce's own preset files.
Those shipped inside the application and there is no published list of them, so
claiming these are those would be a claim I cannot back. Where a name is one
Bryce itself used for a category -- dawn, storm, alien -- it is used here
because it is the obvious name for the thing, not because the numbers came from
Bryce.

bpy-free, like everything else the renderer reads. A preset is a plain dict of
World field names to values, so saving one is `json.dump` and loading one is
`setattr` in a loop -- and a sky saved by a future version with fields this one
has never heard of still loads, minus those fields.
"""

import json

FORMAT = 'halcyon-sky'
FORMAT_VERSION = 1

#: fields that are not part of a sky and must never be written into one:
#: node trees, image datablocks and the render-side mist settings -- plus the
#: infinite plane, which belongs to the water library. A sky preset resets
#: every field it owns, so a field in both libraries would mean picking a sky
#: silently threw your water away. They own disjoint halves and neither can
#: reach the other's.
EXCLUDED = frozenset({
    'graph', 'env_image', 'mist', 'mist_start', 'mist_depth', 'mist_color',
    'mist_falloff', 'mist_intensity', 'mode', 'strength',
})


def sky_fields():
    """Every World field a sky preset may carry."""
    import dataclasses

    from ..core.scene import World
    from .waters import WATER_FIELDS
    blocked = EXCLUDED | set(WATER_FIELDS)
    return tuple(f.name for f in dataclasses.fields(World)
                 if f.name not in blocked)


# ------------------------------------------------------------------ library
#
# Each entry is only the settings that differ from a World's defaults, which
# keeps them readable and means a preset never silently pins a control it has
# no opinion about. `apply_sky` resets first, so that is safe.

SKIES = {
    'BRYCE_DEFAULT': {
        'label': "Bryce Default",
        'note': "The sky Bryce opened on: a blue dome, a scatter of cumulus "
                "and enough haze to put a horizon under them",
        'settings': {},
    },

    # ------------------------------------------------------- times of day
    'DAWN': {
        'label': "Dawn",
        'note': "Sun still under the horizon, the whole dome lit from below",
        'settings': {
            'sun_elevation': -0.04, 'sun_rotation': 1.55,
            'sky_mode': 'SOFT',
            'horizon': (0.86, 0.52, 0.38), 'sky_mid': (0.46, 0.40, 0.55),
            'zenith': (0.10, 0.14, 0.36),
            'sun_color': (1.0, 0.72, 0.48), 'sun_glow_color': (1.0, 0.55, 0.32),
            'sun_glow': 0.7, 'sun_corona': 1.3, 'sun_intensity': 1.2,
            'sun_size': 0.02, 'sun_disc': False,
            'haze_density': 0.85, 'haze_height': 0.13,
            'haze_color': (0.92, 0.70, 0.58), 'haze_sun_tint': 0.9,
            'cloud_cover': 0.35, 'cloud_color': (0.96, 0.78, 0.72),
            'cloud_shadow': (0.35, 0.30, 0.42), 'cloud_rim': 1.0,
            'stratus': True, 'stratus_amount': 0.45,
            'stratus_color': (1.0, 0.80, 0.72),
        },
    },
    'SUNRISE': {
        'label': "Sunrise",
        'note': "The disc just clear of the horizon, everything still orange",
        'settings': {
            'sun_elevation': 0.05, 'sun_rotation': 1.55, 'sky_mode': 'SOFT',
            'horizon': (0.98, 0.74, 0.46), 'sky_mid': (0.62, 0.60, 0.66),
            'zenith': (0.14, 0.28, 0.60),
            'sun_color': (1.0, 0.82, 0.55), 'sun_glow_color': (1.0, 0.66, 0.34),
            'sun_glow': 0.62, 'sun_intensity': 1.5, 'sun_size': 0.024,
            'haze_density': 0.8, 'haze_height': 0.15, 'haze_sun_tint': 0.85,
            'haze_color': (0.96, 0.82, 0.70),
            'cloud_cover': 0.45, 'cloud_rim': 0.9,
            'cloud_color': (1.0, 0.92, 0.86),
        },
    },
    'MORNING_HAZE': {
        'label': "Morning Haze",
        'note': "Low sun through thick air, the horizon barely there",
        'settings': {
            'sun_elevation': 0.22, 'sun_rotation': 1.2,
            'horizon': (0.86, 0.86, 0.82), 'sky_mid': (0.62, 0.72, 0.84),
            'zenith': (0.24, 0.42, 0.72),
            'haze_density': 0.95, 'haze_height': 0.30, 'haze_sun_tint': 0.5,
            'haze_color': (0.92, 0.92, 0.90), 'haze_blend_sky': 0.35,
            'fog_density': 0.35, 'fog_height': 0.05,
            'cloud_cover': 0.3, 'cloud_density': 0.7,
            'sun_glow': 0.45, 'sun_disc': False,
        },
    },
    'HIGH_NOON': {
        'label': "High Noon",
        'note': "Sun overhead, thin haze, small hard cumulus",
        'settings': {
            'sun_elevation': 1.35, 'sun_rotation': 0.5,
            'horizon': (0.72, 0.82, 0.92), 'sky_mid': (0.36, 0.58, 0.86),
            'zenith': (0.06, 0.28, 0.72),
            'sun_glow': 0.22, 'sun_size': 0.016, 'sun_intensity': 1.6,
            'haze_density': 0.35, 'haze_height': 0.10,
            'cloud_cover': 0.4, 'cloud_scale': 0.9, 'cloud_amplitude': 1.4,
            'cloud_thickness': 0.5, 'cloud_softness': 1.4,
        },
    },
    'GOLDEN_HOUR': {
        'label': "Golden Hour",
        'note': "The hour every Bryce postcard was rendered in",
        'settings': {
            'sun_elevation': 0.16, 'sun_rotation': 1.45, 'sky_mode': 'SOFT',
            'horizon': (0.96, 0.82, 0.58), 'sky_mid': (0.52, 0.62, 0.80),
            'zenith': (0.08, 0.24, 0.62),
            'sun_color': (1.0, 0.88, 0.62), 'sun_glow_color': (1.0, 0.72, 0.38),
            'sun_glow': 0.58, 'sun_corona': 1.2, 'sun_intensity': 1.5,
            'haze_density': 0.72, 'haze_height': 0.16, 'haze_sun_tint': 0.8,
            'cloud_cover': 0.5, 'cloud_rim': 0.85, 'cloud_thickness': 0.45,
            'stratus': True, 'stratus_amount': 0.3,
        },
    },
    'SUNSET': {
        'label': "Sunset",
        'note': "Deep orange at the horizon, the dome already going navy",
        'settings': {
            'sun_elevation': 0.03, 'sun_rotation': 4.7, 'sky_mode': 'SOFT',
            'horizon': (1.0, 0.56, 0.26), 'sky_mid': (0.60, 0.36, 0.44),
            'zenith': (0.08, 0.12, 0.36),
            'sun_color': (1.0, 0.64, 0.34), 'sun_glow_color': (1.0, 0.42, 0.18),
            'sun_glow': 0.72, 'sun_corona': 1.4, 'sun_intensity': 1.6,
            'sun_size': 0.028,
            'haze_density': 0.9, 'haze_height': 0.14, 'haze_sun_tint': 0.95,
            'haze_color': (0.96, 0.66, 0.46),
            'cloud_cover': 0.55, 'cloud_rim': 1.2, 'cloud_ambience': 0.2,
            'cloud_color': (1.0, 0.86, 0.76),
            'cloud_shadow': (0.32, 0.22, 0.30),
            'stratus': True, 'stratus_amount': 0.5,
            'stratus_color': (1.0, 0.72, 0.58),
        },
    },
    'DUSK': {
        'label': "Dusk",
        'note': "Sun gone, the last of it under a cold sky, first stars out",
        'settings': {
            'sun_elevation': -0.10, 'sun_rotation': 4.7,
            'horizon': (0.62, 0.42, 0.42), 'sky_mid': (0.24, 0.24, 0.42),
            'zenith': (0.03, 0.05, 0.18),
            'sun_color': (0.9, 0.55, 0.42), 'sun_glow_color': (0.9, 0.42, 0.30),
            'sun_glow': 0.5, 'sun_intensity': 0.7, 'sun_disc': False,
            'haze_density': 0.6, 'haze_height': 0.10, 'haze_sun_tint': 0.8,
            'haze_color': (0.55, 0.38, 0.40),
            'cloud_cover': 0.4, 'cloud_ambience': 0.15,
            'cloud_color': (0.66, 0.58, 0.62),
            'cloud_shadow': (0.16, 0.14, 0.22),
            'stars': True, 'star_brightness': 0.5, 'star_density': 0.4,
        },
    },
    'MOONLIT': {
        'label': "Moonlit Night",
        'note': "The Sun & Moon toggle thrown, with a phase and earthshine",
        'settings': {
            'celestial': 'MOON', 'sun_elevation': 0.55, 'sun_rotation': 2.2,
            'horizon': (0.12, 0.14, 0.24), 'sky_mid': (0.05, 0.07, 0.17),
            'zenith': (0.01, 0.02, 0.07),
            'sun_color': (0.78, 0.83, 0.97), 'sun_glow_color': (0.66, 0.74, 0.95),
            'sun_glow': 0.35, 'sun_intensity': 1.0,
            'moon_phase': 0.38, 'moon_size': 0.05, 'moon_earthshine': 0.10,
            'moon_softness': 0.18,
            'haze_density': 0.35, 'haze_color': (0.30, 0.36, 0.50),
            'cloud_cover': 0.35, 'cloud_color': (0.50, 0.56, 0.70),
            'cloud_shadow': (0.08, 0.10, 0.18), 'cloud_ambience': 0.2,
            'stars': True, 'star_brightness': 1.4, 'star_density': 0.7,
        },
    },
    'STARFIELD_NIGHT': {
        'label': "Deep Night",
        'note': "No moon, full stars, comets. The Celestial tab with "
                "everything on",
        'settings': {
            'sun_elevation': -0.5, 'sun_disc': False, 'sun_glow': 0.0,
            'sun_intensity': 0.2,
            'horizon': (0.05, 0.06, 0.12), 'sky_mid': (0.02, 0.03, 0.08),
            'zenith': (0.005, 0.008, 0.03),
            'haze_density': 0.25, 'haze_color': (0.14, 0.17, 0.28),
            'clouds': False,
            'stars': True, 'star_brightness': 1.8, 'star_density': 0.9,
            'comets': 1.2, 'comet_count': 4,
        },
    },

    # ------------------------------------------------------------ weather
    'CLEAR_BLUE': {
        'label': "Clear Blue",
        'note': "Nothing in the sky at all, which is harder to light than it "
                "sounds",
        'settings': {
            'sun_elevation': 0.85, 'clouds': False, 'stratus': False,
            'horizon': (0.66, 0.80, 0.94), 'sky_mid': (0.30, 0.54, 0.86),
            'zenith': (0.04, 0.24, 0.70),
            'haze_density': 0.30, 'haze_height': 0.09,
            'sun_glow': 0.25, 'sun_size': 0.018,
        },
    },
    'OVERCAST': {
        'label': "Overcast",
        'note': "Total cover, no sun, everything flat -- the Cloud Cover tab "
                "pushed to one",
        'settings': {
            'sun_elevation': 0.7, 'sun_disc': False, 'sun_glow': 0.1,
            'sun_intensity': 0.5,
            'horizon': (0.74, 0.76, 0.78), 'sky_mid': (0.66, 0.68, 0.72),
            'zenith': (0.52, 0.56, 0.62),
            'cloud_cover': 1.0, 'cloud_density': 1.0, 'cloud_softness': 2.2,
            'cloud_amplitude': 0.5, 'cloud_thickness': 0.15,
            'cloud_color': (0.80, 0.81, 0.84),
            'cloud_shadow': (0.55, 0.57, 0.62), 'cloud_ambience': 0.7,
            'cloud_rim': 0.0,
            'haze_density': 0.55, 'haze_color': (0.78, 0.79, 0.82),
            'haze_blend_sky': 0.6,
        },
    },
    'STORM': {
        'label': "Storm Front",
        'note': "Heavy dark cumulus with the light coming under them",
        'settings': {
            'sun_elevation': 0.12, 'sun_rotation': 1.0,
            'horizon': (0.72, 0.66, 0.58), 'sky_mid': (0.34, 0.36, 0.42),
            'zenith': (0.10, 0.12, 0.20),
            'sun_color': (1.0, 0.88, 0.70), 'sun_glow': 0.45,
            'sun_intensity': 1.4, 'sun_disc': False,
            'cloud_cover': 0.88, 'cloud_density': 1.0, 'cloud_thickness': 0.75,
            'cloud_detail': 7, 'cloud_amplitude': 1.8,
            'cloud_turbulence': 1.3, 'cloud_softness': 0.7,
            'cloud_color': (0.72, 0.70, 0.72),
            'cloud_shadow': (0.12, 0.12, 0.16), 'cloud_ambience': 0.12,
            'cloud_rim': 1.4,
            'shadow_color': (0.10, 0.11, 0.16), 'shadow_intensity': 1.0,
            'haze_density': 0.6, 'haze_color': (0.66, 0.62, 0.58),
            'haze_sun_tint': 0.6,
        },
    },
    'MACKEREL': {
        'label': "Mackerel Sky",
        'note': "High rippled cirrus and nothing below it -- the stratus deck "
                "on its own",
        'settings': {
            'sun_elevation': 0.5, 'clouds': False,
            'horizon': (0.74, 0.82, 0.90), 'sky_mid': (0.38, 0.58, 0.84),
            'zenith': (0.08, 0.28, 0.70),
            'stratus': True, 'stratus_amount': 0.72, 'stratus_density': 0.85,
            'stratus_scale': 2.2, 'stratus_frequency': 2.4,
            'stratus_amplitude': 1.6, 'stratus_sharpness': 2.0,
            'stratus_detail': 6, 'stratus_squash': 1.6,
            'haze_density': 0.35,
        },
    },
    'FOG_BANK': {
        'label': "Fog Bank",
        'note': "Ground fog with a base height, so it sits as a layer rather "
                "than filling the frame",
        'settings': {
            'sun_elevation': 0.30, 'sun_glow': 0.4, 'sun_disc': False,
            'horizon': (0.84, 0.84, 0.86), 'sky_mid': (0.60, 0.68, 0.80),
            'zenith': (0.22, 0.38, 0.68),
            'fog_density': 0.95, 'fog_height': 0.10, 'fog_base_height': 0.06,
            'fog_color': (0.90, 0.91, 0.92), 'fog_blend_sky': 0.25,
            'haze_density': 0.45,
            'cloud_cover': 0.25, 'cloud_density': 0.6,
        },
    },
    'RAINBOW': {
        'label': "After the Rain",
        'note': "Sun low behind you, the bow at its 42 degrees, secondary "
                "faint above it",
        'settings': {
            'sun_elevation': 0.14, 'sun_rotation': 4.7,
            'horizon': (0.82, 0.82, 0.78), 'sky_mid': (0.48, 0.62, 0.80),
            'zenith': (0.12, 0.32, 0.68),
            'haze_density': 0.6, 'haze_sun_tint': 0.5,
            'cloud_cover': 0.6, 'cloud_thickness': 0.6,
            'cloud_color': (0.92, 0.92, 0.94),
            'cloud_shadow': (0.34, 0.36, 0.44),
            'rainbow': True, 'rainbow_intensity': 0.75,
            'rainbow_secondary': 0.6, 'rainbow_width': 3.5,
        },
    },
    'CREPUSCULAR': {
        'label': "Sun Through Cloud",
        'note': "Volumetric World, which is the setting that cost Bryce users "
                "their evenings",
        'settings': {
            'sun_elevation': 0.28, 'sun_rotation': 1.5,
            'horizon': (0.88, 0.82, 0.70), 'sky_mid': (0.50, 0.60, 0.78),
            'zenith': (0.10, 0.26, 0.64),
            'sun_glow': 0.5, 'sun_intensity': 1.6, 'sun_disc': False,
            'volumetric_world': 2.5,
            'haze_density': 0.8, 'haze_height': 0.22, 'haze_sun_tint': 0.7,
            'cloud_cover': 0.7, 'cloud_thickness': 0.6, 'cloud_rim': 1.1,
            'cloud_shadow': (0.30, 0.30, 0.38),
        },
    },

    # ------------------------------------------------------------- exotic
    'MARS': {
        'label': "Mars",
        'note': "A butterscotch sky, because the dust is in the air rather "
                "than the air being blue",
        'settings': {
            'sun_elevation': 0.55, 'sun_rotation': 0.9,
            'horizon': (0.82, 0.62, 0.44), 'sky_mid': (0.74, 0.54, 0.38),
            'zenith': (0.54, 0.36, 0.28),
            'sun_color': (1.0, 0.94, 0.84), 'sun_glow_color': (0.9, 0.78, 0.62),
            'sun_glow': 0.4, 'sun_size': 0.014, 'sun_intensity': 1.1,
            'haze_density': 0.7, 'haze_height': 0.4,
            'haze_color': (0.80, 0.58, 0.42), 'haze_blend_sky': 0.3,
            'clouds': False,
            'stratus': True, 'stratus_amount': 0.18,
            'stratus_color': (0.92, 0.80, 0.70), 'stratus_density': 0.4,
        },
    },
    'ALIEN_GREEN': {
        'label': "Alien Sky",
        'note': "Green dome, twin coronas, wrong clouds. Every 1990s render "
                "pack had one",
        'settings': {
            'sun_elevation': 0.35, 'sun_rotation': 2.4,
            'horizon': (0.72, 0.86, 0.52), 'sky_mid': (0.36, 0.64, 0.40),
            'zenith': (0.08, 0.24, 0.22),
            'sun_color': (0.86, 1.0, 0.72), 'sun_glow_color': (0.70, 1.0, 0.55),
            'sun_glow': 0.65, 'sun_corona': 1.5, 'sun_intensity': 1.4,
            'sun_size': 0.035,
            'haze_density': 0.55, 'haze_color': (0.60, 0.86, 0.52),
            'haze_sun_tint': 0.6,
            'cloud_cover': 0.45, 'cloud_color': (0.86, 0.94, 0.70),
            'cloud_shadow': (0.20, 0.34, 0.24), 'cloud_amplitude': 1.6,
            'cloud_turbulence': 1.4,
            'stars': True, 'star_brightness': 0.6,
        },
    },
    'ICE_WORLD': {
        'label': "Ice World",
        'note': "Cold and pale, a low sun and no warmth in the haze at all",
        'settings': {
            'sun_elevation': 0.10, 'sun_rotation': 3.6,
            'horizon': (0.88, 0.92, 0.96), 'sky_mid': (0.58, 0.74, 0.88),
            'zenith': (0.16, 0.36, 0.66),
            'sun_color': (0.92, 0.96, 1.0), 'sun_glow_color': (0.86, 0.94, 1.0),
            'sun_glow': 0.5, 'sun_intensity': 1.2, 'sun_size': 0.02,
            'haze_density': 0.7, 'haze_color': (0.90, 0.94, 0.98),
            'haze_sun_tint': 0.1, 'haze_blend_sky': 0.4,
            'cloud_cover': 0.3, 'cloud_color': (0.96, 0.98, 1.0),
            'cloud_shadow': (0.62, 0.70, 0.80), 'cloud_ambience': 0.5,
            'stratus': True, 'stratus_amount': 0.4,
        },
    },
    'VOLCANIC': {
        'label': "Ashfall",
        'note': "Sun strangled by ash: a dark dome with one hot spot in it",
        'settings': {
            'sun_elevation': 0.20, 'sun_rotation': 0.4,
            'horizon': (0.40, 0.24, 0.18), 'sky_mid': (0.24, 0.16, 0.15),
            'zenith': (0.10, 0.07, 0.08),
            'sun_color': (1.0, 0.42, 0.18), 'sun_glow_color': (1.0, 0.32, 0.10),
            'sun_glow': 0.8, 'sun_corona': 1.6, 'sun_intensity': 2.0,
            'sun_size': 0.03,
            'haze_density': 0.85, 'haze_height': 0.5,
            'haze_color': (0.42, 0.26, 0.20), 'haze_sun_tint': 0.7,
            'cloud_cover': 0.75, 'cloud_color': (0.42, 0.32, 0.30),
            'cloud_shadow': (0.10, 0.07, 0.07), 'cloud_ambience': 0.15,
            'cloud_rim': 1.5, 'cloud_turbulence': 1.5, 'cloud_detail': 7,
            'shadow_color': (0.10, 0.06, 0.06),
        },
    },
    'DEEP_SPACE': {
        'label': "Deep Space",
        'note': "No atmosphere to speak of. The Starfield mode is the better "
                "tool, but Bryce did it from the Sky Lab",
        'settings': {
            'sun_elevation': 0.15, 'sun_rotation': 0.8,
            'horizon': (0.02, 0.02, 0.05), 'sky_mid': (0.01, 0.01, 0.03),
            'zenith': (0.0, 0.0, 0.01),
            'use_sky_mid': True,
            'sun_color': (1.0, 0.98, 0.94), 'sun_glow_color': (0.9, 0.94, 1.0),
            'sun_glow': 0.25, 'sun_size': 0.012, 'sun_intensity': 3.0,
            'sun_corona': 0.6,
            'haze_density': 0.0, 'clouds': False, 'stratus': False,
            'stars': True, 'star_brightness': 2.0, 'star_density': 1.0,
            'comets': 0.8, 'comet_count': 2,
        },
    },
    'DESERT_NOON': {
        'label': "Desert Noon",
        'note': "Bleached out, almost no cloud, haze doing all the work",
        'settings': {
            'sun_elevation': 1.15, 'sun_rotation': 0.2,
            'horizon': (0.92, 0.88, 0.78), 'sky_mid': (0.56, 0.68, 0.84),
            'zenith': (0.14, 0.34, 0.74),
            'sun_glow': 0.3, 'sun_intensity': 1.8, 'sun_size': 0.015,
            'haze_density': 0.62, 'haze_height': 0.20,
            'haze_color': (0.94, 0.90, 0.80), 'haze_sun_tint': 0.4,
            'cloud_cover': 0.12, 'cloud_density': 0.5,
            'cloud_scale': 0.8, 'cloud_thickness': 0.35,
        },
    },
    'TROPICAL': {
        'label': "Tropical Afternoon",
        'note': "Tall bright cumulus over a deep blue dome, the postcard sky",
        'settings': {
            'sun_elevation': 0.75, 'sun_rotation': 1.1,
            'horizon': (0.78, 0.88, 0.94), 'sky_mid': (0.30, 0.58, 0.88),
            'zenith': (0.03, 0.22, 0.70),
            'sun_glow': 0.28, 'sun_intensity': 1.5,
            'haze_density': 0.4, 'haze_height': 0.11,
            'cloud_cover': 0.5, 'cloud_scale': 1.0, 'cloud_thickness': 0.8,
            'cloud_detail': 7, 'cloud_amplitude': 1.7, 'cloud_softness': 0.9,
            'cloud_rim': 0.6, 'cloud_ambience': 0.35,
            'spherical_clouds': True,
        },
    },

    # ------------------------------------------------- pink and purple
    #
    # Bryce's own palette ran to the naturalistic, but the machines it ran on
    # were the same ones the airbrush-and-chrome school was working with, and
    # a Bryce sky pushed into magenta is as period as one pushed into blue.

    'ROSE_TINTED': {
        'label': "Rose Tinted Glasses",
        'note': "Everything a shade warmer and kinder than it was. Pink from "
                "the horizon all the way up, with the haze in on it",
        'settings': {
            'sun_elevation': 0.42, 'sun_rotation': 1.3, 'sky_mode': 'SOFT',
            'horizon': (1.0, 0.82, 0.84), 'sky_mid': (0.94, 0.66, 0.76),
            'zenith': (0.58, 0.34, 0.62),
            'sun_color': (1.0, 0.88, 0.88), 'sun_glow_color': (1.0, 0.74, 0.80),
            'sun_glow': 0.55, 'sun_corona': 1.2, 'sun_intensity': 1.3,
            'sun_size': 0.024,
            'haze_density': 0.68, 'haze_height': 0.24,
            'haze_color': (1.0, 0.80, 0.84), 'haze_sun_tint': 0.6,
            'haze_blend_sky': 0.25,
            'cloud_cover': 0.42, 'cloud_softness': 1.5, 'cloud_thickness': 0.4,
            'cloud_color': (1.0, 0.92, 0.94),
            'cloud_shadow': (0.72, 0.48, 0.64), 'cloud_ambience': 0.5,
            'cloud_rim': 0.6,
            'shadow_color': (0.66, 0.42, 0.58),
        },
    },
    'AFTERLIFE_NIGHT': {
        'label': "Afterlife Night",
        'note': "A violet dark that glows rather than falls. Stars through it, "
                "and no sun to explain the light",
        'settings': {
            'sun_elevation': -0.12, 'sun_rotation': 3.0, 'sun_disc': False,
            'horizon': (0.44, 0.22, 0.50), 'sky_mid': (0.24, 0.10, 0.36),
            'zenith': (0.06, 0.02, 0.16),
            'sun_color': (0.86, 0.60, 1.0), 'sun_glow_color': (0.80, 0.40, 0.95),
            'sun_glow': 0.75, 'sun_corona': 1.6, 'sun_intensity': 1.1,
            'haze_density': 0.6, 'haze_height': 0.22,
            'haze_color': (0.52, 0.26, 0.62), 'haze_sun_tint': 0.7,
            'cloud_cover': 0.4, 'cloud_density': 0.75,
            'cloud_color': (0.62, 0.42, 0.74),
            'cloud_shadow': (0.14, 0.05, 0.22), 'cloud_ambience': 0.3,
            'cloud_rim': 1.2,
            'stars': True, 'star_brightness': 1.3, 'star_density': 0.75,
            'shadow_color': (0.16, 0.06, 0.24),
        },
    },
    'SEE_IT_TO_BELIEVE_IT': {
        'label': "See It To Believe It",
        'note': "A sky nobody would accept as a photograph: magenta banding, "
                "a corona too wide for its sun, and a bow over the top",
        'settings': {
            'sun_elevation': 0.10, 'sun_rotation': 4.6,
            'horizon': (1.0, 0.52, 0.72), 'sky_mid': (0.72, 0.30, 0.86),
            'zenith': (0.16, 0.10, 0.58), 'sky_mid_height': 0.42,
            'gradient_falloff': 0.7,
            'sun_color': (1.0, 0.80, 1.0), 'sun_glow_color': (1.0, 0.30, 0.72),
            'sun_glow': 0.85, 'sun_corona': 2.0, 'sun_intensity': 1.8,
            'sun_size': 0.04,
            'haze_density': 0.7, 'haze_color': (1.0, 0.56, 0.82),
            'haze_sun_tint': 0.9,
            'cloud_cover': 0.5, 'cloud_amplitude': 2.2,
            'cloud_color': (1.0, 0.90, 1.0),
            'cloud_shadow': (0.46, 0.16, 0.60), 'cloud_rim': 1.6,
            'rainbow': True, 'rainbow_intensity': 0.9,
            'rainbow_secondary': 0.7, 'rainbow_width': 4.0,
        },
    },
    'SYNTHETIC_WONDERWORLD': {
        'label': "Synthetic Wonderworld",
        'note': "Hot magenta at the deck, indigo overhead, one enormous low "
                "sun. The airbrush school, rendered",
        'settings': {
            'sun_elevation': 0.02, 'sun_rotation': 1.5708,
            'horizon': (1.0, 0.24, 0.52), 'sky_mid': (0.62, 0.14, 0.62),
            'zenith': (0.06, 0.04, 0.34), 'sky_mid_height': 0.30,
            'gradient_falloff': 0.6,
            'sun_color': (1.0, 0.52, 0.78), 'sun_glow_color': (1.0, 0.22, 0.56),
            'sun_glow': 0.6, 'sun_corona': 1.1, 'sun_intensity': 2.2,
            'sun_size': 0.075,
            'haze_density': 0.55, 'haze_height': 0.10,
            'haze_color': (1.0, 0.34, 0.62), 'haze_sun_tint': 0.85,
            'clouds': False,
            'stratus': True, 'stratus_amount': 0.5, 'stratus_density': 0.7,
            'stratus_color': (1.0, 0.60, 0.86), 'stratus_altitude': 4.0,
            'stratus_squash': 2.2, 'stratus_sharpness': 2.2,
            'stars': True, 'star_brightness': 0.7, 'star_density': 0.5,
        },
    },
    'LAND_OF_VAPOR': {
        'label': "In The Land of Vapor",
        'note': "Pastel and washed out, pink one way and cyan the other, with "
                "enough haze that nothing has an edge",
        'settings': {
            'sun_elevation': 0.30, 'sun_rotation': 2.0,
            'horizon': (1.0, 0.72, 0.84), 'sky_mid': (0.72, 0.72, 0.96),
            'zenith': (0.30, 0.72, 0.86), 'sky_mid_height': 0.40,
            'sun_color': (1.0, 0.92, 0.96), 'sun_glow_color': (1.0, 0.80, 0.92),
            'sun_glow': 0.45, 'sun_corona': 1.0, 'sun_intensity': 0.8,
            'sun_disc': False,
            'haze_density': 0.5, 'haze_height': 0.14,
            'haze_color': (1.0, 0.74, 0.86), 'haze_blend_sky': 0.2,
            'haze_sun_tint': 0.35,
            'fog_density': 0.18, 'fog_height': 0.07,
            'fog_color': (0.96, 0.84, 0.94), 'fog_blend_sky': 0.3,
            'cloud_cover': 0.3, 'cloud_softness': 2.4, 'cloud_density': 0.6,
            'cloud_color': (1.0, 0.94, 0.98),
            'cloud_shadow': (0.76, 0.72, 0.90), 'cloud_ambience': 0.7,
        },
    },
    'VAPORTRAILS': {
        'label': "Vaportrails",
        'note': "High thin streaks pulled right across a lilac sky, and "
                "nothing below them",
        'settings': {
            'sun_elevation': 0.24, 'sun_rotation': 0.9,
            'horizon': (1.0, 0.86, 0.86), 'sky_mid': (0.82, 0.62, 0.86),
            'zenith': (0.34, 0.26, 0.70),
            'sun_color': (1.0, 0.90, 0.92), 'sun_glow_color': (1.0, 0.72, 0.84),
            'sun_glow': 0.5, 'sun_intensity': 1.3,
            'haze_density': 0.5, 'haze_height': 0.16,
            'haze_color': (1.0, 0.84, 0.88), 'haze_sun_tint': 0.6,
            'clouds': False,
            'stratus': True, 'stratus_amount': 0.62, 'stratus_density': 0.9,
            'stratus_altitude': 5.0, 'stratus_scale': 3.4,
            'stratus_frequency': 0.7, 'stratus_amplitude': 2.4,
            'stratus_sharpness': 2.6, 'stratus_squash': 5.0,
            'stratus_detail': 5, 'stratus_color': (1.0, 0.94, 0.96),
            'cloud_wind': 0.4,
        },
    },

    # ------------------------------------------------------------ themed
    'AFTERLIFE_DAY': {
        'label': "Afterlife Day",
        'note': "The same place in daylight: blown out and lilac-shadowed, "
                "with the sun somewhere behind all of it",
        'settings': {
            'sun_elevation': 0.85, 'sun_rotation': 1.2, 'sun_disc': False,
            'horizon': (1.0, 0.94, 0.97), 'sky_mid': (0.80, 0.72, 0.94),
            'zenith': (0.42, 0.40, 0.82), 'sky_mid_height': 0.36,
            'sun_color': (1.0, 0.98, 0.96), 'sun_glow_color': (1.0, 0.94, 0.98),
            'sun_glow': 0.7, 'sun_corona': 1.6, 'sun_intensity': 1.4,
            'haze_density': 0.55, 'haze_height': 0.20,
            'haze_color': (1.0, 0.92, 0.98), 'haze_blend_sky': 0.25,
            'cloud_cover': 0.35, 'cloud_softness': 2.0,
            'cloud_color': (1.0, 1.0, 1.0),
            'cloud_shadow': (0.82, 0.78, 0.94), 'cloud_ambience': 0.75,
            'shadow_color': (0.72, 0.68, 0.90),
        },
    },
    'ALIEN_PLANET': {
        'label': "Alien Planet",
        'note': "Violet air over a teal horizon, a small hard white sun and a "
                "sky that never sees blue",
        'settings': {
            'sun_elevation': 0.45, 'sun_rotation': 3.4,
            'horizon': (0.34, 0.72, 0.68), 'sky_mid': (0.40, 0.30, 0.72),
            'zenith': (0.20, 0.06, 0.34), 'sky_mid_height': 0.38,
            'sun_color': (1.0, 0.98, 1.0), 'sun_glow_color': (0.86, 0.78, 1.0),
            'sun_glow': 0.4, 'sun_size': 0.012, 'sun_intensity': 2.0,
            'sun_corona': 0.8,
            'haze_density': 0.6, 'haze_height': 0.20,
            'haze_color': (0.48, 0.62, 0.74), 'haze_sun_tint': 0.3,
            'cloud_cover': 0.35, 'cloud_amplitude': 2.0,
            'cloud_turbulence': 1.5, 'cloud_detail': 7,
            'cloud_color': (0.80, 0.74, 0.92),
            'cloud_shadow': (0.24, 0.14, 0.40),
            'stars': True, 'star_brightness': 0.9, 'star_density': 0.6,
            'comets': 0.6, 'comet_count': 2,
        },
    },
    'ABSTRACT_MOVIE': {
        'label': "Abstract Movie",
        'note': "Flat saturated bands and clouds with no soft edge left in "
                "them. A title sequence rather than a place",
        'settings': {
            'sun_elevation': 0.20, 'sun_rotation': 2.6,
            'horizon': (1.0, 0.34, 0.30), 'sky_mid': (0.92, 0.24, 0.70),
            'zenith': (0.14, 0.06, 0.46), 'sky_mid_height': 0.5,
            'gradient_falloff': 0.5, 'use_sky_mid': True,
            'sun_color': (1.0, 0.94, 0.40), 'sun_glow_color': (1.0, 0.70, 0.20),
            'sun_glow': 0.3, 'sun_size': 0.055, 'sun_intensity': 2.4,
            'sun_corona': 0.4,
            'haze_density': 0.2,
            'cloud_cover': 0.45, 'cloud_amplitude': 3.0,
            'cloud_softness': 0.25, 'cloud_turbulence': 1.6,
            'cloud_color': (1.0, 0.86, 0.20),
            'cloud_shadow': (0.30, 0.06, 0.34), 'cloud_ambience': 0.05,
            'cloud_rim': 0.0, 'cloud_thickness': 0.2,
        },
    },
    'OLD_PHOTOGRAPH': {
        'label': "Old Age Photograph",
        'note': "Sepia and low contrast, the way a print looks after eighty "
                "years in a drawer",
        'settings': {
            'sun_elevation': 0.55, 'sun_rotation': 1.0, 'sun_disc': False,
            'horizon': (0.88, 0.80, 0.62), 'sky_mid': (0.72, 0.62, 0.44),
            'zenith': (0.44, 0.36, 0.24),
            'sun_color': (0.92, 0.86, 0.70), 'sun_glow_color': (0.94, 0.86, 0.66),
            'sun_glow': 0.4, 'sun_intensity': 0.9,
            'haze_density': 0.5, 'haze_height': 0.18,
            'haze_color': (0.82, 0.74, 0.58), 'haze_blend_sky': 0.35,
            'cloud_cover': 0.55, 'cloud_softness': 1.3, 'cloud_thickness': 0.4,
            'cloud_color': (0.90, 0.85, 0.72),
            'cloud_shadow': (0.60, 0.54, 0.44), 'cloud_ambience': 0.6,
            'cloud_rim': 0.2,
            'shadow_color': (0.52, 0.46, 0.36),
        },
    },
    'STONE_AGE_PHOTOGRAPH': {
        'label': "Stone Age Photograph",
        'note': "Older still: a cold plate, almost no colour, and a sky burnt "
                "to paper white behind the clouds",
        'settings': {
            'sun_elevation': 0.65, 'sun_rotation': 0.3, 'sun_disc': False,
            'horizon': (0.74, 0.73, 0.70), 'sky_mid': (0.52, 0.53, 0.54),
            'zenith': (0.22, 0.24, 0.27),
            'sun_color': (0.88, 0.87, 0.84), 'sun_glow_color': (0.92, 0.90, 0.86),
            'sun_glow': 0.5, 'sun_intensity': 1.2, 'sun_corona': 1.2,
            'haze_density': 0.5, 'haze_height': 0.16,
            'haze_color': (0.78, 0.77, 0.74), 'haze_blend_sky': 0.35,
            'cloud_cover': 0.7, 'cloud_density': 1.0, 'cloud_softness': 0.9,
            'cloud_detail': 6, 'cloud_amplitude': 1.6,
            'cloud_color': (0.84, 0.83, 0.80),
            'cloud_shadow': (0.26, 0.27, 0.29), 'cloud_ambience': 0.25,
            'cloud_rim': 0.1,
            'shadow_color': (0.36, 0.37, 0.39),
        },
    },
    'AMBER_LAMPS': {
        'label': "Amber Lamps",
        'note': "Sodium light on the underside of a low overcast, and no sky "
                "past it at all",
        'settings': {
            'sun_elevation': -0.06, 'sun_rotation': 1.0, 'sun_disc': False,
            'horizon': (0.62, 0.34, 0.10), 'sky_mid': (0.34, 0.16, 0.06),
            'zenith': (0.08, 0.04, 0.04),
            'sun_color': (1.0, 0.62, 0.16), 'sun_glow_color': (1.0, 0.52, 0.10),
            'sun_glow': 0.9, 'sun_corona': 2.0, 'sun_intensity': 1.4,
            'haze_density': 0.8, 'haze_height': 0.16,
            'haze_color': (0.86, 0.46, 0.14), 'haze_sun_tint': 0.9,
            'cloud_cover': 0.85, 'cloud_density': 1.0, 'cloud_thickness': 0.3,
            'cloud_softness': 1.8,
            'cloud_color': (0.72, 0.40, 0.14),
            'cloud_shadow': (0.16, 0.07, 0.03), 'cloud_ambience': 0.25,
            'cloud_rim': 1.0,
            'shadow_color': (0.14, 0.06, 0.03),
        },
    },
    'SAPPHIRE_IMAGINATION': {
        'label': "Sapphire Imagination",
        'note': "Jewel blue all the way down, lit from inside rather than "
                "from a sun",
        'settings': {
            'sun_elevation': 0.30, 'sun_rotation': 2.8,
            'horizon': (0.16, 0.48, 0.92), 'sky_mid': (0.08, 0.22, 0.78),
            'zenith': (0.02, 0.04, 0.30),
            'sun_color': (0.86, 0.94, 1.0), 'sun_glow_color': (0.50, 0.78, 1.0),
            'sun_glow': 0.7, 'sun_corona': 1.5, 'sun_intensity': 2.0,
            'sun_size': 0.018,
            'haze_density': 0.5, 'haze_height': 0.18,
            'haze_color': (0.24, 0.54, 0.96), 'haze_sun_tint': 0.5,
            'cloud_cover': 0.3, 'cloud_amplitude': 1.8,
            'cloud_color': (0.72, 0.88, 1.0),
            'cloud_shadow': (0.06, 0.16, 0.52), 'cloud_ambience': 0.25,
            'cloud_rim': 1.3,
            'stars': True, 'star_brightness': 0.8, 'star_density': 0.5,
        },
    },
    'BEYOND_THE_RAINBOW': {
        'label': "Beyond The Rainbow",
        'note': "Past where the colours stop making sense: a full bow, a "
                "secondary, and a sky already doing the same thing",
        'settings': {
            'sun_elevation': 0.12, 'sun_rotation': 4.7,
            'horizon': (0.94, 0.74, 0.86), 'sky_mid': (0.56, 0.52, 0.92),
            'zenith': (0.14, 0.10, 0.46), 'sky_mid_height': 0.40,
            'gradient_falloff': 0.8,
            'sun_color': (1.0, 0.94, 0.98), 'sun_glow_color': (1.0, 0.78, 0.92),
            'sun_glow': 0.4, 'sun_intensity': 1.3, 'sun_disc': False,
            'haze_density': 0.32, 'haze_color': (0.92, 0.78, 0.92),
            'haze_blend_sky': 0.25,
            'cloud_cover': 0.22, 'cloud_density': 0.7,
            'cloud_color': (1.0, 0.96, 1.0),
            'cloud_shadow': (0.40, 0.34, 0.62),
            'rainbow': True, 'rainbow_intensity': 1.0,
            'rainbow_secondary': 0.85, 'rainbow_width': 5.0,
            'rainbow_radius': 42.0,
        },
    },
    'HALLOWEEN': {
        'label': "Halloween",
        'note': "A big low orange moon, ragged cloud crossing it, and a sky "
                "that has gone the colour of a pumpkin lid",
        'settings': {
            'celestial': 'MOON', 'sun_elevation': 0.10, 'sun_rotation': 1.5708,
            'horizon': (0.52, 0.22, 0.05), 'sky_mid': (0.22, 0.10, 0.10),
            'zenith': (0.04, 0.02, 0.06),
            'sun_color': (1.0, 0.62, 0.20), 'sun_glow_color': (1.0, 0.44, 0.08),
            'sun_glow': 0.7, 'sun_corona': 1.5, 'sun_intensity': 1.4,
            'moon_color': (1.0, 0.68, 0.26), 'moon_size': 0.085,
            'moon_phase': 0.5, 'moon_earthshine': 0.04, 'moon_softness': 0.10,
            'haze_density': 0.6, 'haze_height': 0.14,
            'haze_color': (0.64, 0.28, 0.08), 'haze_sun_tint': 0.85,
            'cloud_cover': 0.55, 'cloud_density': 0.9,
            'cloud_amplitude': 2.2, 'cloud_turbulence': 1.5,
            'cloud_softness': 0.6, 'cloud_detail': 7,
            'cloud_color': (0.46, 0.28, 0.16),
            'cloud_shadow': (0.06, 0.03, 0.05), 'cloud_ambience': 0.1,
            'cloud_rim': 1.4,
            'stars': True, 'star_brightness': 0.9, 'star_density': 0.55,
        },
    },
    'CHRISTMAS': {
        'label': "Christmas",
        'note': "Cold blue dusk over snow, the light going and the first "
                "stars already up",
        'settings': {
            'sun_elevation': -0.05, 'sun_rotation': 3.9, 'sun_disc': False,
            'horizon': (0.90, 0.80, 0.72), 'sky_mid': (0.34, 0.46, 0.72),
            'zenith': (0.05, 0.10, 0.30),
            'sun_color': (1.0, 0.86, 0.72), 'sun_glow_color': (1.0, 0.74, 0.58),
            'sun_glow': 0.45, 'sun_intensity': 0.9,
            'haze_density': 0.65, 'haze_height': 0.13,
            'haze_color': (0.82, 0.84, 0.92), 'haze_sun_tint': 0.45,
            'cloud_cover': 0.45, 'cloud_softness': 1.6,
            'cloud_color': (0.86, 0.90, 0.98),
            'cloud_shadow': (0.28, 0.36, 0.56), 'cloud_ambience': 0.4,
            'stars': True, 'star_brightness': 1.0, 'star_density': 0.6,
            'shadow_color': (0.26, 0.34, 0.54),
        },
    },
    'VALENTINES': {
        'label': "Valentines",
        'note': "Red at the horizon into rose into deep pink, and the clouds "
                "catching all of it",
        'settings': {
            'sun_elevation': 0.08, 'sun_rotation': 1.2, 'sky_mode': 'SOFT',
            'horizon': (1.0, 0.30, 0.36), 'sky_mid': (1.0, 0.58, 0.68),
            'zenith': (0.52, 0.14, 0.36), 'sky_mid_height': 0.38,
            'sun_color': (1.0, 0.66, 0.68), 'sun_glow_color': (1.0, 0.36, 0.42),
            'sun_glow': 0.65, 'sun_corona': 1.3, 'sun_intensity': 1.6,
            'sun_size': 0.03,
            'haze_density': 0.72, 'haze_height': 0.18,
            'haze_color': (1.0, 0.52, 0.58), 'haze_sun_tint': 0.8,
            'cloud_cover': 0.5, 'cloud_thickness': 0.55,
            'cloud_color': (1.0, 0.84, 0.88),
            'cloud_shadow': (0.62, 0.18, 0.34), 'cloud_ambience': 0.3,
            'cloud_rim': 1.3,
            'shadow_color': (0.56, 0.16, 0.30),
        },
    },
    'EASTER': {
        'label': "Easter",
        'note': "Pastel and evenly lit, the sky as a sugared egg",
        'settings': {
            'sun_elevation': 0.60, 'sun_rotation': 1.6,
            'horizon': (1.0, 0.88, 0.60), 'sky_mid': (0.62, 0.92, 0.74),
            'zenith': (0.52, 0.70, 0.98), 'sky_mid_height': 0.40,
            'gradient_falloff': 0.75,
            'sun_color': (1.0, 0.98, 0.86), 'sun_glow_color': (1.0, 0.94, 0.76),
            'sun_glow': 0.35, 'sun_intensity': 1.0, 'sun_size': 0.02,
            'haze_density': 0.28, 'haze_height': 0.10,
            'haze_color': (1.0, 0.92, 0.80), 'haze_blend_sky': 0.3,
            'cloud_cover': 0.4, 'cloud_softness': 1.8, 'cloud_thickness': 0.35,
            'cloud_color': (1.0, 0.98, 1.0),
            'cloud_shadow': (0.82, 0.78, 0.90), 'cloud_ambience': 0.65,
            'cloud_rim': 0.4,
        },
    },
    'CRYSTAL_STARS': {
        'label': "Crystal Stars",
        'note': "Air with nothing in it. Every star out, a violet wash where "
                "the sun went, and no cloud to soften any of it",
        'settings': {
            'sun_elevation': -0.22, 'sun_rotation': 4.2, 'sun_disc': False,
            'horizon': (0.20, 0.16, 0.36), 'sky_mid': (0.08, 0.07, 0.24),
            'zenith': (0.01, 0.01, 0.06),
            'sun_color': (0.70, 0.56, 1.0), 'sun_glow_color': (0.58, 0.36, 0.92),
            'sun_glow': 0.55, 'sun_corona': 1.2, 'sun_intensity': 0.9,
            'haze_density': 0.28, 'haze_height': 0.10,
            'haze_color': (0.28, 0.22, 0.46), 'haze_sun_tint': 0.7,
            'clouds': False, 'stratus': False,
            'stars': True, 'star_brightness': 2.0, 'star_density': 1.0,
            'comets': 0.9, 'comet_count': 3,
        },
    },
}

#: display order, because a dict is not an ordering anyone chose
ORDER = (
    'BRYCE_DEFAULT',
    'DAWN', 'SUNRISE', 'MORNING_HAZE', 'HIGH_NOON', 'DESERT_NOON',
    'TROPICAL', 'GOLDEN_HOUR', 'SUNSET', 'DUSK', 'MOONLIT', 'STARFIELD_NIGHT',
    'CLEAR_BLUE', 'MACKEREL', 'OVERCAST', 'STORM', 'FOG_BANK', 'RAINBOW',
    'CREPUSCULAR',
    'MARS', 'ALIEN_GREEN', 'ALIEN_PLANET', 'ICE_WORLD', 'VOLCANIC',
    'DEEP_SPACE',
    # pink and purple
    'ROSE_TINTED', 'VALENTINES', 'LAND_OF_VAPOR', 'VAPORTRAILS',
    'SYNTHETIC_WONDERWORLD', 'SEE_IT_TO_BELIEVE_IT', 'BEYOND_THE_RAINBOW',
    'AFTERLIFE_DAY', 'AFTERLIFE_NIGHT', 'CRYSTAL_STARS',
    'SAPPHIRE_IMAGINATION', 'AMBER_LAMPS',
    # themed
    'ABSTRACT_MOVIE', 'OLD_PHOTOGRAPH', 'STONE_AGE_PHOTOGRAPH',
    'HALLOWEEN', 'CHRISTMAS', 'EASTER',
)


# ------------------------------------------------ the extended library
try:
    from .skies_extra import ORDER_EXTRA, SKIES_EXTRA
    SKIES.update(SKIES_EXTRA)
    ORDER = tuple(ORDER) + tuple(k for k in ORDER_EXTRA if k in SKIES)
except Exception:                                               # noqa: BLE001
    pass


def sky_items():
    """(key, label, note) in display order, for an enum or a menu."""
    return tuple((k, SKIES[k]['label'], SKIES[k]['note'])
                 for k in ORDER if k in SKIES)


# ------------------------------------------------------------ apply / save


def reset_sky(world):
    """Every sky field back to its default, leaving mode and strength alone.

    Presets reset before they apply, for the same reason the render presets do:
    without it they accumulate, and the tenth sky you try is nine skies deep.
    """
    from ..core.scene import World
    fresh = World()
    for name in sky_fields():
        setattr(world, name, getattr(fresh, name))
    return world


def apply_sky(world, key, reset=True):
    """Load a named sky onto a World. Returns (ok, message)."""
    entry = SKIES.get(key)
    if entry is None:
        return False, f'unknown sky {key!r}'
    if reset:
        reset_sky(world)
    allowed = set(sky_fields())
    for name, value in entry['settings'].items():
        if name in allowed:
            setattr(world, name, value)
    return True, entry['label']


def sky_to_dict(world, name=''):
    """A saveable dict of everything that makes this sky what it is."""
    out = {}
    for field in sky_fields():
        value = getattr(world, field, None)
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            out[field] = value
        elif hasattr(value, '__len__'):
            try:
                out[field] = [float(v) for v in value]
            except (TypeError, ValueError):
                continue
    return {'format': FORMAT, 'version': FORMAT_VERSION,
            'name': name or 'Sky', 'settings': out}


def sky_from_dict(world, data, reset=True):
    """Load a saved sky onto a World. Returns (ok, message).

    Unknown fields are skipped rather than refused: a sky saved by a later
    version should still load here, minus whatever this version has never
    heard of. That is the whole reason the format is a flat dict of names.
    """
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        return False, 'not a Halcyon sky file'
    version = int(data.get('version', 0))
    if version > FORMAT_VERSION:
        # forward compatible on purpose, but say so
        pass
    settings = data.get('settings')
    if not isinstance(settings, dict):
        return False, 'the file has no settings in it'
    if reset:
        reset_sky(world)
    allowed = set(sky_fields())
    skipped = 0
    from ..core.scene import World
    defaults = World()
    for name, value in settings.items():
        if name not in allowed:
            skipped += 1
            continue
        want = getattr(defaults, name)
        try:
            if isinstance(want, tuple):
                setattr(world, name, tuple(float(v) for v in value))
            elif isinstance(want, bool):
                setattr(world, name, bool(value))
            elif isinstance(want, int):
                setattr(world, name, int(value))
            elif isinstance(want, float):
                setattr(world, name, float(value))
            else:
                setattr(world, name, value)
        except (TypeError, ValueError):
            skipped += 1
    note = data.get('name', 'Sky')
    if skipped:
        note += f' ({skipped} setting(s) this version does not have)'
    return True, note


def dumps(world, name=''):
    return json.dumps(sky_to_dict(world, name), indent=1, sort_keys=True)


def loads(world, text, reset=True):
    try:
        data = json.loads(text)
    except ValueError as exc:
        return False, f'not valid JSON: {exc}'
    return sky_from_dict(world, data, reset=reset)
