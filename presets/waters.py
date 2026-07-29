"""Water presets for the infinite plane, and the format they save in.

Bryce kept its waters in the Materials Library under a category called
**Waters & Liquids**, and dropping one onto the water plane was how a Bryce
picture got its sea. A plane with twenty controls and no library is the same
problem the Sky Lab had.

**What these are.** Waters built out of Halcyon's own ocean controls, tuned to
the conditions they are named for. **What they are not:** Bryce's preset files.
Those shipped inside the application and no published list of them survives.
Two things here *are* Bryce's and are used deliberately: the category name
Waters & Liquids, and **Caribbean Resort**, which a period tutorial places
second along the top row of it. Every other name is the obvious name for the
thing, not a claim about what Bryce shipped.

bpy-free, like everything else the renderer reads. A preset is a plain dict of
World field names to values, so saving one is `json.dump` and loading one is
`setattr` in a loop -- and a water saved by a future version with fields this
one has never heard of still loads, minus those fields.

The split from the sky library matters and is deliberate: a sky preset resets
every field it owns, so if the two libraries shared fields, picking a sky would
silently throw your water away. They own disjoint halves of the World and
neither can touch the other's.
"""

import json

FORMAT = 'halcyon-water'
FORMAT_VERSION = 1

#: Every World field a water preset owns -- the whole infinite plane, because
#: the water is one of its modes rather than a thing beside it. `ground_color`
#: is deliberately NOT here: the sky modes use it to fill below the horizon
#: when Show Ground is on, so it belongs to the sky and moving it would make
#: loading a water repaint the sky.
WATER_FIELDS = (
    'ground_plane', 'ground_mode', 'ground_height', 'ground_scale',
    'ground_color2', 'ground_fade',
    'ocean_choppiness', 'ocean_speed', 'ocean_wind_angle', 'ocean_spread',
    'ocean_wave_scale', 'ocean_detail', 'ocean_sparkle',
    'ocean_horizon_smooth', 'ocean_deep', 'ocean_shallow', 'ocean_glitter',
    'ocean_glitter_size', 'ocean_foam', 'ocean_foam_color',
    'ocean_transparency',
)


def water_fields():
    """Every World field a water preset may carry, as it exists in this build.

    Filtered against the dataclass rather than trusted, so a field renamed out
    from under this list is dropped instead of writing an attribute nothing
    reads.
    """
    import dataclasses

    from ..core.scene import World
    known = {f.name for f in dataclasses.fields(World)}
    return tuple(n for n in WATER_FIELDS if n in known)


# ------------------------------------------------------------------ library
#
# Each entry is only what differs from a World's defaults, which keeps them
# readable and means a preset never silently pins a control it has no opinion
# about. `apply_water` resets first, so that is safe. Every one of them turns
# the plane on and puts it in Water: picking a water and getting no water
# would be a strange thing to ship.

_ON = {'ground_plane': True, 'ground_mode': 'OCEAN'}


def _w(**kw):
    d = dict(_ON)
    d.update(kw)
    return d


WATERS = {
    'BRYCE_WATER': {
        'label': "Bryce Water",
        'note': "The water the plane starts as: mid blue, moderate chop, the "
                "sun's path straight down the middle of it",
        'settings': _w(),
    },

    # --------------------------------------------------------- the tropics
    'CARIBBEAN_RESORT': {
        'label': "Caribbean Resort",
        'note': "Bryce's own name, from the top row of Waters & Liquids: "
                "turquoise, clear enough to see through, and a fine bright "
                "chop",
        'settings': _w(ocean_deep=(0.02, 0.17, 0.21),
                       ocean_shallow=(0.11, 0.47, 0.46),
                       ocean_transparency=0.22, ocean_wave_scale=0.55,
                       ocean_choppiness=0.22, ocean_spread=0.5,
                       ocean_glitter=1.7, ocean_glitter_size=0.30,
                       ground_fade=140.0),
    },
    'TROPICAL_SHALLOWS': {
        'label': "Tropical Shallows",
        'note': "Pale green over sand, barely any water above it, ripples "
                "rather than waves",
        'settings': _w(ocean_deep=(0.10, 0.34, 0.33),
                       ocean_shallow=(0.30, 0.60, 0.53),
                       ocean_transparency=0.12, ocean_wave_scale=0.30,
                       ocean_choppiness=0.14, ocean_detail=6,
                       ocean_glitter=1.3, ocean_glitter_size=0.24,
                       ground_fade=90.0),
    },
    'CALM_LAGOON': {
        'label': "Calm Lagoon",
        'note': "Sheltered water: glassy, with just enough ripple to break "
                "the reflection",
        'settings': _w(ocean_deep=(0.02, 0.14, 0.16),
                       ocean_shallow=(0.09, 0.38, 0.38),
                       ocean_transparency=0.20, ocean_wave_scale=0.9,
                       ocean_choppiness=0.09, ocean_spread=0.25,
                       ocean_glitter=0.8, ocean_glitter_size=0.18,
                       ground_fade=110.0),
    },
    'MEDITERRANEAN': {
        'label': "Mediterranean",
        'note': "Clear blue-green with a steady breeze across it and a bright "
                "high sun on the crests",
        'settings': _w(ocean_deep=(0.01, 0.10, 0.18),
                       ocean_shallow=(0.06, 0.32, 0.38),
                       ocean_transparency=0.24, ocean_wave_scale=1.4,
                       ocean_choppiness=0.30, ocean_spread=0.55,
                       ocean_glitter=1.3, ground_fade=200.0),
    },

    # ------------------------------------------------------- the open water
    'OPEN_OCEAN': {
        'label': "Open Ocean",
        'note': "Deep blue with a long swell running under it and nothing in "
                "any direction",
        'settings': _w(ocean_deep=(0.008, 0.045, 0.105),
                       ocean_shallow=(0.03, 0.15, 0.24),
                       ocean_wave_scale=6.0, ocean_choppiness=0.40,
                       ocean_spread=0.45, ocean_detail=6,
                       ocean_transparency=0.18, ground_fade=400.0),
    },
    'ROLLING_SWELL': {
        'label': "Rolling Swell",
        'note': "One long period and almost no chop on top of it, which is "
                "swell that has travelled a long way from its weather",
        'settings': _w(ocean_deep=(0.01, 0.05, 0.11),
                       ocean_shallow=(0.04, 0.17, 0.25),
                       ocean_wave_scale=16.0, ocean_choppiness=0.45,
                       ocean_spread=0.05, ocean_detail=3,
                       ocean_speed=0.55, ocean_transparency=0.15,
                       ground_fade=500.0),
    },
    'CHOPPY_BAY': {
        'label': "Choppy Bay",
        'note': "Short steep waves running every which way, the sea a wind "
                "makes in shallow water",
        'settings': _w(ocean_deep=(0.02, 0.07, 0.10),
                       ocean_shallow=(0.07, 0.22, 0.25),
                       ocean_wave_scale=1.1, ocean_choppiness=0.60,
                       ocean_spread=1.0, ocean_detail=7, ocean_speed=1.4,
                       ocean_foam=0.15, ground_fade=160.0),
    },
    'DEEP_ATLANTIC': {
        'label': "Deep Atlantic",
        'note': "Cold, dark and big: a heavy swell with the tops going over",
        'settings': _w(ocean_deep=(0.005, 0.025, 0.055),
                       ocean_shallow=(0.02, 0.09, 0.14),
                       ocean_wave_scale=9.0, ocean_choppiness=0.55,
                       ocean_spread=0.7, ocean_detail=7, ocean_foam=0.28,
                       ocean_transparency=0.10, ocean_glitter=0.7,
                       ground_fade=350.0),
    },
    'STORM_SWELL': {
        'label': "Storm Swell",
        'note': "Grey, steep and breaking, with the sun buried far enough "
                "that there is no path left on it",
        'settings': _w(ocean_deep=(0.018, 0.035, 0.045),
                       ocean_shallow=(0.07, 0.11, 0.125),
                       ocean_wave_scale=7.0, ocean_choppiness=0.85,
                       ocean_spread=0.95, ocean_detail=8, ocean_speed=1.6,
                       ocean_foam=0.62, ocean_foam_color=(0.86, 0.88, 0.88),
                       ocean_glitter=0.25, ocean_transparency=0.06,
                       ground_fade=120.0),
    },
    'ARCTIC_SEA': {
        'label': "Arctic Sea",
        'note': "Near-black water under a pale sky, moving slowly because "
                "cold water is heavy",
        'settings': _w(ocean_deep=(0.004, 0.014, 0.024),
                       ocean_shallow=(0.02, 0.07, 0.09),
                       ocean_wave_scale=11.0, ocean_choppiness=0.32,
                       ocean_spread=0.35, ocean_speed=0.5, ocean_foam=0.18,
                       ocean_transparency=0.05, ocean_glitter=0.55,
                       ocean_glitter_size=0.6, ground_fade=280.0),
    },

    # ------------------------------------------------------- inland waters
    'MILLPOND': {
        'label': "Millpond",
        'note': "Dead calm. Whatever is in the sky is in the water, the right "
                "way up and barely bent",
        'settings': _w(ocean_deep=(0.012, 0.04, 0.05),
                       ocean_shallow=(0.05, 0.14, 0.15),
                       ocean_wave_scale=3.5, ocean_choppiness=0.045,
                       ocean_spread=0.15, ocean_speed=0.35,
                       ocean_glitter=0.45, ocean_glitter_size=0.11,
                       ocean_transparency=0.15, ground_fade=220.0),
    },
    'MOUNTAIN_LAKE': {
        'label': "Mountain Lake",
        'note': "Dark green and very still, with a fine ripple the wind off "
                "the slope keeps putting back",
        'settings': _w(ocean_deep=(0.008, 0.035, 0.028),
                       ocean_shallow=(0.04, 0.15, 0.12),
                       ocean_wave_scale=0.8, ocean_choppiness=0.10,
                       ocean_spread=0.4, ocean_speed=0.6,
                       ocean_glitter=0.7, ocean_glitter_size=0.2,
                       ocean_transparency=0.20, ground_fade=180.0),
    },
    'SWIMMING_POOL': {
        'label': "Swimming Pool",
        'note': "Chlorine blue, shallow, and rippled far finer than any sea",
        'settings': _w(ocean_deep=(0.06, 0.34, 0.42),
                       ocean_shallow=(0.20, 0.55, 0.60),
                       ocean_wave_scale=0.22, ocean_choppiness=0.13,
                       ocean_detail=6, ocean_speed=1.3,
                       ocean_transparency=0.08, ocean_glitter=1.5,
                       ocean_glitter_size=0.2, ground_fade=60.0),
    },
    'BLACK_LAGOON': {
        'label': "Black Lagoon",
        'note': "Water with nothing under it and nothing through it, which is "
                "how you get a sea that reads as depth rather than as colour",
        'settings': _w(ocean_deep=(0.002, 0.006, 0.010),
                       ocean_shallow=(0.008, 0.022, 0.030),
                       ocean_wave_scale=1.6, ocean_choppiness=0.28,
                       ocean_transparency=0.0, ocean_glitter=1.1,
                       ocean_glitter_size=0.35, ground_fade=140.0),
    },
    'GLACIAL_MELT': {
        'label': "Glacial Melt",
        'note': "Rock flour in the water turns it opaque and milky, a pale "
                "blue that light does not get into",
        'settings': _w(ocean_deep=(0.10, 0.24, 0.30),
                       ocean_shallow=(0.28, 0.46, 0.52),
                       ocean_wave_scale=0.7, ocean_choppiness=0.18,
                       ocean_spread=0.5, ocean_speed=0.8,
                       ocean_transparency=0.0, ocean_glitter=0.8,
                       ocean_glitter_size=0.5, ground_fade=150.0),
    },

    # ----------------------------------------------------- light on water
    'MOONLIT_WATER': {
        'label': "Moonlit Water",
        'note': "Almost black, with one narrow silver road running out to the "
                "moon and nothing else lit at all",
        'settings': _w(ocean_deep=(0.002, 0.005, 0.012),
                       ocean_shallow=(0.010, 0.022, 0.045),
                       ocean_wave_scale=1.8, ocean_choppiness=0.30,
                       ocean_spread=0.5, ocean_glitter=2.6,
                       ocean_glitter_size=0.17, ocean_transparency=0.0,
                       ground_fade=260.0),
    },
    'SUNSET_OCEAN': {
        'label': "Sunset Ocean",
        'note': "A low sun makes the widest path it will make all day, and it "
                "reaches from the horizon to your feet",
        'settings': _w(ocean_deep=(0.020, 0.055, 0.085),
                       ocean_shallow=(0.09, 0.17, 0.20),
                       ocean_wave_scale=3.5, ocean_choppiness=0.42,
                       ocean_spread=0.6, ocean_glitter=2.1,
                       ocean_glitter_size=0.95, ocean_transparency=0.12,
                       ground_fade=300.0),
    },
    'LIQUID_MERCURY': {
        'label': "Liquid Mercury",
        'note': "The other half of Waters & Liquids: no colour of its own and "
                "nothing through it, so all it can do is mirror",
        'settings': _w(ocean_deep=(0.020, 0.021, 0.023),
                       ocean_shallow=(0.055, 0.056, 0.060),
                       ocean_wave_scale=0.45, ocean_choppiness=0.16,
                       ocean_spread=0.3, ocean_speed=0.7,
                       ocean_transparency=0.0, ocean_glitter=3.2,
                       ocean_glitter_size=0.075, ground_fade=200.0),
    },
    'ALIEN_SEA': {
        'label': "Alien Sea",
        'note': "Violet under, green on top, and a chop with no weather "
                "behind it that makes any sense",
        'settings': _w(ocean_deep=(0.055, 0.010, 0.090),
                       ocean_shallow=(0.09, 0.30, 0.16),
                       ocean_wave_scale=2.2, ocean_choppiness=0.50,
                       ocean_spread=0.85, ocean_detail=7, ocean_speed=1.5,
                       ocean_transparency=0.30, ocean_glitter=1.4,
                       ocean_glitter_size=0.5, ground_fade=220.0),
    },
}


ORDER = (
    'BRYCE_WATER',
    'CARIBBEAN_RESORT', 'TROPICAL_SHALLOWS', 'CALM_LAGOON', 'MEDITERRANEAN',
    'OPEN_OCEAN', 'ROLLING_SWELL', 'CHOPPY_BAY', 'DEEP_ATLANTIC',
    'STORM_SWELL', 'ARCTIC_SEA',
    'MILLPOND', 'MOUNTAIN_LAKE', 'SWIMMING_POOL', 'BLACK_LAGOON',
    'GLACIAL_MELT',
    'MOONLIT_WATER', 'SUNSET_OCEAN', 'LIQUID_MERCURY', 'ALIEN_SEA',
)


def water_items():
    """(key, label, note) in display order, for an enum or a menu."""
    return tuple((k, WATERS[k]['label'], WATERS[k]['note'])
                 for k in ORDER if k in WATERS)


# ------------------------------------------------------------ apply / save


def reset_water(world):
    """Every water field back to its default, touching nothing else."""
    from ..core.scene import World
    fresh = World()
    for name in water_fields():
        setattr(world, name, getattr(fresh, name))
    return world


def apply_water(world, key, reset=True):
    """Load a named water onto a World. Returns (ok, message)."""
    entry = WATERS.get(key)
    if entry is None:
        return False, f'unknown water {key!r}'
    if reset:
        reset_water(world)
    allowed = set(water_fields())
    for name, value in entry['settings'].items():
        if name in allowed:
            setattr(world, name, value)
    return True, entry['label']


def water_to_dict(world, name=''):
    """A saveable dict of everything that makes this water what it is."""
    out = {}
    for field in water_fields():
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
            'name': name or 'Water', 'settings': out}


def water_from_dict(world, data, reset=True):
    """Load a saved water onto a World. Returns (ok, message).

    Unknown fields are skipped rather than refused, so a water saved by a
    later version still loads here minus whatever this one has never heard of.
    """
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        return False, 'not a Halcyon water file'
    settings = data.get('settings')
    if not isinstance(settings, dict):
        return False, 'the file has no settings in it'
    if reset:
        reset_water(world)
    allowed = set(water_fields())
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
    note = data.get('name', 'Water')
    if skipped:
        note += f' ({skipped} setting(s) this version does not have)'
    return True, note


def dumps(world, name=''):
    return json.dumps(water_to_dict(world, name), indent=1, sort_keys=True)


def loads(world, text, reset=True):
    try:
        data = json.loads(text)
    except ValueError as exc:
        return False, f'not valid JSON: {exc}'
    return water_from_dict(world, data, reset=reset)
