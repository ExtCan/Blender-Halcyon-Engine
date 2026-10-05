"""R251 helpers shared by the round's test modules.

The preset identity law (R251 integration pass 1): a pack's neutrality
pin over the shipped presets compares THIS tree's presets against the
1.89.0 library imported from the zip beside the package (through the
``_prev_engine`` package, so the two libraries are the two releases'
own text), and the presets that must render bitwise the 1.89.0 engine
are exactly those whose ``'settings'`` dict is EQUAL to the old
library's.  A preset the round edited (any pack, any key) or added is
excluded by construction, never by a hand-kept list that a later merge
silently invalidates.  The round's exact preset count is pinned ONCE,
in test_render.py, by the integrator; a pack asserts only that ITS
keys exist with the right shelf and a note, and that the count is at
least the round's.

The palette lock (pass 1b): ``post._palette_for`` keeps an ADAPTIVE
palette under ``('ADAPTIVE', size, method, seed)`` -- no image content
in the key, on purpose: ``palette_lock`` lets the first frame of an
animation decide the palette (1.89.0 behaviour, kept).  Each engine
(this tree, the zip's) holds its own ``_PALETTE_CACHE``, so two engines
that post-process the same frame agree bitwise ONLY when both build the
palette from that frame: a comparison that clears one side (or neither)
inherits whatever frame each side locked earlier in the suite and
diverges by run order (APPLE_IIGS, C64, GAME_GEAR ... moved by 0.13-0.87
in pass 1b).  ``clear_palette_locks(RP)`` clears both before every
cross-engine post comparison.
"""

import collections
import importlib

PresetDelta = collections.namedtuple(
    'PresetDelta', 'unchanged changed added removed')


def prev_library(RP):
    """The previous release's ``presets.library`` module, imported
    through the package ``tests._prev_engine`` unpacked (``RP`` is its
    ``core.render``: two levels up is the package)."""
    return importlib.import_module(
        RP.__name__.rsplit('.', 2)[0] + '.presets.library')


def prev_palette(RP):
    """The previous release's ``core.palette`` module (``RP`` is its
    ``core.render``)."""
    return importlib.import_module(RP.__name__.rsplit('.', 1)[0] + '.palette')


def clear_palette_locks(RP=None):
    """Empty the adaptive palette lock of THIS tree and (when ``RP`` is
    given) of the previous engine, so that a frame post-processed by both
    builds its palette from itself on both sides.  Call it before every
    cross-engine ``post.process`` comparison (see the module docstring)."""
    from ..core import palette as PA
    PA.clear_caches()
    if RP is not None:
        prev_palette(RP).clear_caches()


def preset_delta(RP, PRESETS=None):
    """Sort this tree's presets against the previous release's library.

    Returns ``PresetDelta(unchanged, changed, added, removed)``, four
    sorted key lists: ``unchanged`` are the keys present in both whose
    ``'settings'`` dict compares EQUAL (``==``) to the old one, ``changed``
    the keys present in both whose dict differs, ``added`` the keys only
    this tree has, ``removed`` the keys only the old library had."""
    if PRESETS is None:
        from ..presets.library import PRESETS
    old = prev_library(RP).PRESETS
    unchanged, changed = [], []
    for key in sorted(PRESETS):
        if key not in old:
            continue
        if PRESETS[key]['settings'] == old[key]['settings']:
            unchanged.append(key)
        else:
            changed.append(key)
    added = sorted(k for k in PRESETS if k not in old)
    removed = sorted(k for k in old if k not in PRESETS)
    return PresetDelta(unchanged, changed, added, removed)


def unchanged_presets(RP, PRESETS=None):
    """The keys whose ``'settings'`` dict is EQUAL to the 1.89.0 library's:
    those, and only those, must render bitwise the 1.89.0 engine (render
    and post).  Sorted."""
    return preset_delta(RP, PRESETS).unchanged


def delta_extra(delta):
    """The check's extra: how many presets were compared and how many the
    round's edits excluded, by name."""
    return (f'{len(delta.unchanged)} compared; excluded {len(delta.changed)} '
            f'edited {delta.changed} and {len(delta.added)} new {delta.added}'
            + (f'; removed {delta.removed}' if delta.removed else ''))
