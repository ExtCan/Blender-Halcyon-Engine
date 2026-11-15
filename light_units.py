"""Area lamp power units: which lamps keep Blender Internal's law.

An Area lamp's Power used to mean one thing under Halcyon: Blender
Internal's energy, normalised at the lamp's Distance dial, D^2 / 2
times Blender's watts (312 at the factory Distance 25). A lamp added in
Blender 5 arrives in watts, so a new Area lamp whited out the frame.
Since 1.93 every Area lamp carries `halcyon.area_power`: 'WATTS' (the
RNA default, what a NEW lamp gets) or 'BI' (the old law, bitwise).

A lamp saved before the option existed has no value stored, and would
read the default and render 312 times darker. This module is what keeps
those lamps as they were: wherever an Area lamp ARRIVES without a stored
value, it is judged by the file it was last saved in, and 'BI' is
written onto it when that file is an older Halcyon file.

    the file it was saved in ...                    the lamp gets
    carries the stamp (saved by 1.93 or later)      nothing: watts
    no stamp, a scene renders with Halcyon          'BI'
    no stamp, the lamp's own Halcyon dials moved    'BI'
    no stamp, no sign of Halcyon (EEVEE, Cycles)    nothing: watts

The stamp is `scene.halcyon.light_units_version`, written onto every
scene when a file is loaded, when the add-on is enabled over an open
file, and before every save. The routes a lamp can arrive by:

* File > Open, Revert, Recover, the startup file: `load_post`. The
  loaded scenes say what the file is. The undo stack is initialised
  after load_post, so the written values are part of its first state.
* Linked lamps (`lamp.library` set): judged at every load and at link
  time by THEIR library file, whose scenes were not loaded: its bytes
  are searched for the stamp's name and for the engine's name (through
  zstd or gzip when the file is compressed). A linked datablock cannot
  be saved from here, so the value is written again at each load.
* File > Append and Link, drag and drop, the asset browser,
  `bpy.data.libraries.load`: `blend_import_post`, the same search of
  the source file. A classic (2.79) source is the append watch's: its
  lamps get 'BI' with the rest of their Blender Internal values.
* The add-on enabled, or updated, over an open file: `register` sorts
  the file at once -- every Area lamp that exists at that moment was lit
  under the old law, saved or not. Blender swaps `bpy.data` for a stub
  while an add-on registers; the real one is read through the restricted
  context (`bpy.context._real_data`, what its own `window_manager`
  property reads). Where a build offers no such road the sort is left
  pending for a zero-delay timer or `render_init`, and then only touches
  the lamps the file on disk holds (a lamp a script made after enabling
  the add-on is a new lamp). Either way the writes happen outside the
  undo stack, so undo could take them back: every lamp this session set
  to 'BI' is remembered by session_uid, and `undo_post` / `redo_post`
  write the value again where it went missing.
* A lamp created by a script or an operator has no stored value: watts,
  the RNA default. `save_pre` stores the value each local Area lamp is
  using, so a lamp appended later from that file says what it is by
  itself.

Not covered: a compressed library when Blender's zstandard module is
missing (the lamp's own dials decide alone); a lamp made while the
add-on was disabled in an unstamped Halcyon file (it is sorted with the
old ones); "Reset to Default Value" on a migrated lamp's Power Units
followed by an undo in the same session (the 'BI' comes back; choosing
an item from the menu is never overridden); a library override of a
linked lamp follows its own stored value like any local lamp.
"""

import os

import bpy

def persistent(fn):
    """bpy.app.handlers.persistent where it exists (a handler without
    the mark is dropped by the very file load it answers), identity
    elsewhere."""
    try:
        return bpy.app.handlers.persistent(fn)
    except Exception:                                           # noqa: BLE001
        return fn

ENGINE = 'HALCYON_RENDER'

#: the value `scene.halcyon.light_units_version` carries once a file's
#: Area lamps have been sorted
UNITS_VERSION = 1

#: True: only a file that shows Halcyon use keeps its unsorted Area
#: lamps on Blender Internal units; a file authored for EEVEE or Cycles
#: opens in watts, as its lamps were made. False stamps 'BI' onto every
#: unsorted Area lamp of every older file, whatever it was rendered with
HALCYON_FILES_ONLY = True

#: bytes whose presence in a .blend says what it is: the engine's name
#: sits in a scene's RenderData, the stamp's name in the property itself
_MARK_ENGINE = ENGINE.encode('ascii')
_MARK_STAMP = b'light_units_version'

#: session_uids of the lamps this session set to 'BI' (undo, redo)
_KEPT_BI = set()

#: (path, mtime, size) -> (halcyon, sorted) of library files searched
_MARKS = {}

#: True while the open file still waits for its sort (the add-on was
#: enabled over it and bpy.data was not readable during register)
_PENDING = [False]


# ------------------------------------------------------------ inspection


def is_stored(hs, name='area_power'):
    """True when the property has a value of its own in the file (the
    user chose it, or a sort wrote it); None when that cannot be asked
    (a stub without is_property_set)."""
    try:
        return bool(hs.is_property_set(name))
    except Exception:                                           # noqa: BLE001
        return None


def group_touched(group, skip=()):
    """True when any property of a Halcyon settings group has a stored
    value -- someone moved a Halcyon dial on that datablock."""
    try:
        props = group.bl_rna.properties
    except Exception:                                           # noqa: BLE001
        return False
    for p in props:
        name = p.identifier
        if name == 'rna_type' or name in skip:
            continue
        try:
            if group.is_property_set(name):
                return True
        except Exception:                                       # noqa: BLE001
            continue
    return False


def _local(idb):
    return getattr(idb, 'library', None) is None


def file_is_sorted(data):
    """True when a local scene carries the stamp."""
    for sc in getattr(data, 'scenes', ()):
        if not _local(sc):
            continue
        hs = getattr(sc, 'halcyon', None)
        if int(getattr(hs, 'light_units_version', 0) or 0) >= UNITS_VERSION:
            return True
    return False


def file_shows_halcyon(data):
    """True when the open file was worked on with Halcyon: a local scene
    renders with it, or carries a moved Halcyon render setting."""
    for sc in getattr(data, 'scenes', ()):
        if not _local(sc):
            continue
        if getattr(getattr(sc, 'render', None), 'engine', '') == ENGINE:
            return True
        hs = getattr(sc, 'halcyon', None)
        if hs is not None and group_touched(
                hs, skip=('light_units_version',)):
            return True
    return False


def _search(chunks, marks):
    """Which of `marks` occur in a stream of byte chunks (an overlap is
    carried across chunk borders)."""
    found = [False] * len(marks)
    keep = max(len(m) for m in marks) - 1
    tail = b''
    for chunk in chunks:
        buf = tail + chunk
        for i, m in enumerate(marks):
            if not found[i] and m in buf:
                found[i] = True
        if all(found):
            break
        tail = buf[-keep:] if keep > 0 else b''
    return found


def _chunks(fh, size=1 << 22):
    while True:
        b = fh.read(size)
        if not b:
            return
        yield b


def file_has(path, marks):
    """Which of the byte strings `marks` a .blend on disk contains, as a
    tuple of bools, or None when the file cannot be read. Plain, zstd
    (Blender 3.0+) and gzip (older) files; cached per (path, mtime,
    size, marks)."""
    marks = tuple(marks)
    try:
        st = os.stat(path)
        key = (path, st.st_mtime, st.st_size, marks)
    except OSError:
        return None
    hit = _MARKS.get(key)
    if hit is not None:
        return hit
    try:
        with open(path, 'rb') as fh:
            head = fh.read(4)
            fh.seek(0)
            if head[:2] == b'\x1f\x8b':
                import gzip
                with gzip.open(fh, 'rb') as gz:
                    found = _search(_chunks(gz), marks)
            elif head == b'\x28\xb5\x2f\xfd':
                import zstandard
                with zstandard.ZstdDecompressor().stream_reader(
                        fh, read_across_frames=True) as zr:
                    found = _search(_chunks(zr), marks)
            else:
                found = _search(_chunks(fh), marks)
    except Exception:                                           # noqa: BLE001
        return None
    out = tuple(bool(f) for f in found)
    if len(_MARKS) > 64:
        _MARKS.clear()
    _MARKS[key] = out
    return out


def file_marks(path):
    """(shows Halcyon, is sorted) for a .blend on disk, read from its
    bytes, or None when the file cannot be read."""
    return file_has(path, (_MARK_ENGINE, _MARK_STAMP))


def file_holds_lamp(path, name):
    """True when the .blend on disk holds a Light datablock of this name
    (its ID name as the file stores it: the two-letter code, the name,
    a terminator); None when the file cannot be read."""
    try:
        mark = b'LA' + str(name).encode('utf-8') + b'\x00'
    except Exception:                                           # noqa: BLE001
        return None
    hit = file_has(path, (mark,))
    return None if hit is None else hit[0]


def _abspath(fp, library=None):
    try:
        return bpy.path.abspath(fp, library=library)
    except Exception:                                           # noqa: BLE001
        try:
            return bpy.path.abspath(fp)
        except Exception:                                       # noqa: BLE001
            return fp


def keeps_bi(light, shows_halcyon, is_sorted):
    """The rule: does an Area lamp WITHOUT a stored value, last saved in
    a file that (shows Halcyon, is sorted), keep Blender Internal units?
    `is_sorted` None means the file could not be read: the lamp's own
    dials decide alone."""
    if is_sorted:
        return False
    if not HALCYON_FILES_ONLY and is_sorted is not None:
        return True
    if shows_halcyon:
        return True
    return group_touched(getattr(light, 'halcyon', None),
                         skip=('area_power',))


def _keep(light):
    """Write 'BI' onto one lamp and remember it for undo."""
    try:
        light.halcyon.area_power = 'BI'
    except Exception:                                           # noqa: BLE001
        return False
    uid = getattr(light, 'session_uid', None)
    if uid is not None:
        _KEPT_BI.add(int(uid))
    return True


def _unsorted_area(light):
    """An Area lamp whose units nobody has stored yet."""
    if getattr(light, 'type', '') != 'AREA':
        return False
    hs = getattr(light, 'halcyon', None)
    if hs is None or not hasattr(hs, 'area_power'):
        return False
    return is_stored(hs) is False


# ------------------------------------------------------------- the sorts


def stamp_scenes(data):
    for sc in getattr(data, 'scenes', ()):
        if not _local(sc):
            continue
        hs = getattr(sc, 'halcyon', None)
        try:
            if hs is not None and int(hs.light_units_version) \
                    < UNITS_VERSION:
                hs.light_units_version = UNITS_VERSION
        except Exception:                                       # noqa: BLE001
            pass


def sort_file(data=None, on_disk_only=False):
    """The load-time sort of the open file. Returns the names of the
    lamps kept on Blender Internal units.

    `on_disk_only` is the pending form (the add-on enabled over an open
    file on a build whose data could not be read at register): time has
    passed since, so a local lamp is sorted only when the file on disk
    holds it -- anything else was made after the add-on came up and is a
    new lamp."""
    data = bpy.data if data is None else data
    kept = []
    lights = [la for la in getattr(data, 'lights', ())
              if _unsorted_area(la)]
    if lights:
        own_sorted = file_is_sorted(data)
        own_halcyon = file_shows_halcyon(data)
        own_path = str(getattr(data, 'filepath', '') or '')
        for la in lights:
            lib = getattr(la, 'library', None)
            if lib is None:
                if on_disk_only and not (own_path and file_holds_lamp(
                        own_path, la.name)):
                    continue
                halcyon, is_sorted = own_halcyon, own_sorted
            else:
                marks = file_marks(_abspath(
                    getattr(lib, 'filepath', '') or '',
                    getattr(lib, 'library', None)))
                halcyon, is_sorted = marks if marks is not None \
                    else (False, None)
            if keeps_bi(la, halcyon, is_sorted) and _keep(la):
                kept.append(la.name)
    stamp_scenes(data)
    return kept


def _item_path(item):
    libs = getattr(item, 'source_libraries', None)
    if libs is not None:
        try:
            for lib in libs:
                fp = getattr(lib, 'filepath', '') or ''
                if fp:
                    return fp
        except TypeError:
            pass
    lib = getattr(item, 'source_library', None)
    if lib is not None:
        return getattr(lib, 'filepath', '') or ''
    return ''


def sort_import(items):
    """The append / link sort: the Area lamps among the import items,
    each judged by its source file. Returns the names kept on 'BI'."""
    kept = []
    seen = set()
    for item in items or ():
        idb = getattr(item, 'id', None)
        if idb is None:
            continue
        la = idb
        if getattr(item, 'id_type', '') == 'OBJECT' or (
                getattr(idb, 'type', '') == 'LIGHT'
                and hasattr(idb, 'data')):
            la = getattr(idb, 'data', None)
        if la is None or id(la) in seen or not _unsorted_area(la):
            continue
        seen.add(id(la))
        fp = _item_path(item)
        marks = file_marks(_abspath(fp)) if fp else None
        halcyon, is_sorted = marks if marks is not None else (False, None)
        if keeps_bi(la, halcyon, is_sorted) and _keep(la):
            kept.append(la.name)
    return kept


# -------------------------------------------------------------- handlers


def _readable(data):
    return hasattr(data, 'lights') and hasattr(data, 'scenes')


def _data_ready():
    """False while Blender restricts bpy.data (an add-on registering)."""
    return _readable(getattr(bpy, 'data', None))


def _data_now():
    """The open file's data, also while an add-on registers: bpy.data,
    or the real one behind Blender's restricted context. None when
    neither can be read."""
    data = getattr(bpy, 'data', None)
    if _readable(data):
        return data
    data = getattr(getattr(bpy, 'context', None), '_real_data', None)
    return data if _readable(data) else None


def _report(kept):
    if kept:
        print(f'[Halcyon] {len(kept)} Area lamp(s) from before 1.93 '
              'keep their Blender Internal power units: '
              + ', '.join(sorted(kept)[:12])
              + ('...' if len(kept) > 12 else ''))


@persistent
def _on_load(*_args):
    """load_post: sort the file that just opened."""
    try:
        _PENDING[0] = False
        _KEPT_BI.clear()
        _report(sort_file())
    except Exception:                                           # noqa: BLE001
        import traceback
        traceback.print_exc()


@persistent
def _sort_pending(*_args):
    """The sort of a file that was already open when the add-on was
    enabled: from the timer register leaves, or from render_init. Runs
    once; returns None, which also ends the timer."""
    try:
        if _PENDING[0] and _data_ready():
            _PENDING[0] = False
            _report(sort_file(on_disk_only=True))
    except Exception:                                           # noqa: BLE001
        import traceback
        traceback.print_exc()
    return None


@persistent
def _on_import(*args):
    """blend_import_post: sort the lamps an append or a link brought."""
    try:
        ctx = args[0] if args else None
        if ctx is None:
            return
        stage = getattr(ctx, 'process_stage', 'DONE')
        if stage not in (None, '', 'DONE'):
            return
        sort_import(getattr(ctx, 'import_items', None))
    except Exception:                                           # noqa: BLE001
        import traceback
        traceback.print_exc()


@persistent
def _on_undo(*_args):
    """undo_post / redo_post: a 'BI' this session wrote outside the undo
    stack is written again where a restored state dropped it."""
    try:
        if not _KEPT_BI:
            return
        for la in bpy.data.lights:
            uid = getattr(la, 'session_uid', None)
            if uid is not None and int(uid) in _KEPT_BI \
                    and _unsorted_area(la):
                la.halcyon.area_power = 'BI'
    except Exception:                                           # noqa: BLE001
        pass


@persistent
def _on_save(*_args):
    """save_pre: the file being written says what its lamps are -- each
    local Area lamp stores the units it is using, every scene the
    stamp."""
    try:
        for la in bpy.data.lights:
            try:
                if _local(la) and _unsorted_area(la):
                    la.halcyon.area_power = la.halcyon.area_power
            except Exception:                                   # noqa: BLE001
                continue
        stamp_scenes(bpy.data)
    except Exception:                                           # noqa: BLE001
        pass


_HOOKS = (('load_post', _on_load), ('blend_import_post', _on_import),
          ('undo_post', _on_undo), ('redo_post', _on_undo),
          ('save_pre', _on_save), ('render_init', _sort_pending))


def register():
    h = getattr(getattr(bpy, 'app', None), 'handlers', None)
    for name, fn in _HOOKS:
        lst = getattr(h, name, None)
        if isinstance(lst, list) and fn not in lst:
            lst.append(fn)
    # the already-open file too: the lamps that exist now were lit
    # under the old law (the add-on enabled or updated mid-session; at
    # start-up the file is not loaded yet and its load_post follows)
    data = _data_now()
    if data is not None:
        try:
            _report(sort_file(data))
        except Exception:                                       # noqa: BLE001
            pass
        return
    _PENDING[0] = True
    try:
        bpy.app.timers.register(_sort_pending, first_interval=0.0)
    except Exception:                                           # noqa: BLE001
        pass


def unregister():
    h = getattr(getattr(bpy, 'app', None), 'handlers', None)
    for name, fn in _HOOKS:
        lst = getattr(h, name, None)
        if isinstance(lst, list):
            while fn in lst:
                lst.remove(fn)
    try:
        if bpy.app.timers.is_registered(_sort_pending):
            bpy.app.timers.unregister(_sort_pending)
    except Exception:                                           # noqa: BLE001
        pass
    _PENDING[0] = False
    _KEPT_BI.clear()
    _MARKS.clear()
