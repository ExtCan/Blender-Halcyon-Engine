"""The append watch: classic lamps fixed the moment they arrive.

The field's workflow is plain File > Append -- a route on which no
Halcyon operator ever runs. Blender's own append machinery converts
2.79 lamp strengths to its modern units (every appended sun arrives at
exactly 1.0 W), and the manual "Fix Appended Lamps" repair, however
correct, is a second step the user has to remember. Convenience is the
requirement; this module removes the step.

Blender 4.1+ fires `bpy.app.handlers.blend_import_post` after every
link or append -- File > Append, drag-and-drop, the asset browser, and
`bpy.data.libraries.load` alike -- handing the callback ONE argument, a
`BlendImportContext` (confirmed against the current Blender source:
rna_blendfile_import.cc and bpy_app_handlers.cc). Its `import_items`
carry the actual imported IDs (`item.id`) and, decisively, the SOURCE
file each came from (`item.source_libraries[...].filepath`), so the
watch knows which .blend to consult without guessing at operator state.

The watch stays out of the way unless EVERY gate passes:

* the scene's render engine is Halcyon (appending a classic prop into
  someone's Eevee project must not stamp Blender-Internal energies);
* the scene toggle (Render Properties > Lighting > Auto-Fix Appended
  Lamps) has not been switched off;
* the import was an APPEND -- linked IDs are library-owned and
  read-only, so a link passes through untouched;
* the source file is CLASSIC: a 12-byte header sniff (through gzip,
  as old saves often are) reads the version digits, and only < 2.80
  proceeds. Zstandard-compressed files are modern by definition;
* lights actually arrived. Per item, `KEEP_LINKED` and `REUSE_LOCAL`
  are skipped -- a reused local light already lives in the scene, and
  whatever the user has done to it since is theirs, not this watch's
  to overwrite.

What passes is handed to the SAME core as the manual fixer
(`legacy_import.fix_lights_from_parsed` -> `_apply_lamp_bi`: one
function for every route, so they cannot disagree), matching by object
name first and lamp-data name second, exact before rename-suffix
strip. Receipts land in the console and a version-stamped
'<file> lamp fix log' text datablock -- only when there was something
to say; a lightless append leaves no trace.

The Halcyon importer's own append fires this handler too, before its
own enrichment runs. That is deliberate double coverage, not an
accident: the two apply identical values, and the watch keeps working
even if the importer's receipt stage misbehaves (a field
log). The cost is one extra geometry-free parse per import.
"""

import os

import bpy

from .core import blend279
from . import legacy_import as _li

try:                                        # fakebpy has no handlers
    from bpy.app.handlers import persistent
except Exception:                           # noqa: BLE001  # pragma: no cover
    def persistent(fn):
        return fn


_GZIP = b'\x1f\x8b'
_ZSTD = b'\x28\xb5\x2f\xfd'


def blend_header_version(path):
    """The 3-digit version from a .blend header, or None.

    Reads 12 bytes -- through gzip when the file is gzip-wrapped, as
    2.4x-era saves often are. Zstandard-compressed files (Blender 3.0+)
    are modern by definition and return None without decompressing.
    Anything unreadable or unrecognisable is None: the watch treats
    'not provably classic' as 'leave it alone'.
    """
    try:
        with open(path, 'rb') as fh:
            head = fh.read(12)
        if head[:2] == _GZIP:
            import gzip
            with gzip.open(path, 'rb') as fh:
                head = fh.read(12)
        if head[:4] == _ZSTD:
            return None
        if len(head) < 12 or head[:7] != b'BLENDER':
            return None
        v = head[9:12].decode('ascii', 'replace')
        return int(v) if v.isdigit() else None
    except Exception:                                           # noqa: BLE001
        return None


def _item_filepath(item):
    """The source .blend an imported item came from, or ''.

    `source_libraries` is the import context's own record of the path
    (it survives the append discarding the library datablock);
    `source_library` is the fallback for shapes that only carry the
    pointer."""
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


def _abspath(fp):
    try:
        return bpy.path.abspath(fp)
    except Exception:                                           # noqa: BLE001
        return fp


def collect_light_jobs(items):
    """The import items that are appended lights, grouped by source
    file: {filepath: {'objects': [light objects], 'lights': [bare
    Light datablocks]}}. Pure inspection -> testable."""
    jobs = {}
    for item in items or ():
        if getattr(item, 'append_action', '') in ('KEEP_LINKED',
                                                  'REUSE_LOCAL'):
            continue
        idb = getattr(item, 'id', None)
        if idb is None or getattr(idb, 'library', None) is not None:
            continue
        itype = getattr(item, 'id_type', '')
        kind = None
        if itype == 'OBJECT' and getattr(idb, 'type', '') == 'LIGHT' \
                and getattr(idb, 'data', None) is not None:
            kind = 'objects'
        elif itype == 'LIGHT':
            kind = 'lights'
        if kind is None:
            continue
        fp = _item_filepath(item)
        if not fp:
            continue
        jobs.setdefault(fp, {'objects': [], 'lights': []})[kind].append(idb)
    return jobs


def _fix_from_file(fp, light_obs, bare_lights):
    """One source file's arrived lights through the shared fixer core,
    with the same receipts discipline as the manual operator."""
    file_ver = blend_header_version(fp)
    if file_ver is None or file_ver >= 280:
        return
    sc = blend279.read_legacy_scene(fp, geometry=False)
    warnings = []
    receipts, unmatched, by_obj, by_data = _li.fix_lights_from_parsed(
        sc, light_obs, warnings, bare_lights=bare_lights)
    if not receipts and not unmatched and not warnings:
        return
    if unmatched:
        warnings.append(
            f'{len(unmatched)} appended light(s) matched no lamp in '
            'the file and were left as Blender converted them: '
            + ', '.join(sorted(unmatched)[:12])
            + ('...' if len(unmatched) > 12 else ''))
        warnings.append(
            '  the file offers these lamp names: '
            + ', '.join(sorted(by_obj or by_data)[:12])
            + ('...' if len(by_obj or by_data) > 12 else ''))
    try:
        from .version import VERSION
        ver = '.'.join(str(v) for v in VERSION)
    except Exception:                                           # noqa: BLE001
        ver = '?'
    base = os.path.splitext(os.path.basename(fp))[0] or 'Legacy'
    # distinct lights: a Light datablock item whose object also arrived
    # is the same lamp, not a second one
    covered = {id(getattr(ob, 'data', None)) for ob in light_obs}
    n_lights = len(light_obs) + sum(
        1 for lt in bare_lights if id(lt) not in covered)
    stage = (f'Halcyon {ver}: append watch fixed {len(receipts)} of '
             f'{n_lights} appended light(s) from {os.path.basename(fp)} '
             "with the file's own Blender Internal values"
             + (f'; {len(unmatched)} unmatched' if unmatched else ''))
    for line in [stage] + receipts + warnings:
        _li.safe_print(f'[halcyon append watch] {line}')
    _li.write_log_text(f'{base} lamp fix log',
                       [stage] + receipts + warnings)
    try:
        from . import fault_note
        fault_note(f'append watch completed ({stage})', key='appendwatch')
    except Exception:                                           # noqa: BLE001
        pass


@persistent
def on_blend_import_post(*args):
    """The handler. Never raises into Blender's import machinery: a
    watch that could abort the user's append would be worse than no
    watch at all."""
    try:
        ctx = args[0] if args else None
        if ctx is None:
            return
        stage = getattr(ctx, 'process_stage', 'DONE')
        if stage not in (None, '', 'DONE'):
            return                  # the pre stage of a future contract
        opts = getattr(ctx, 'options', None) or set()
        if 'LINK' in opts:
            return                  # linked IDs are read-only
        scene = getattr(getattr(bpy, 'context', None), 'scene', None)
        if scene is None:
            return
        if getattr(getattr(scene, 'render', None), 'engine', '') \
                != 'HALCYON_RENDER':
            return                  # never touch another engine's append
        hs = getattr(scene, 'halcyon', None)
        if hs is not None and \
                getattr(hs, 'auto_fix_appended_lamps', True) is False:
            return
        jobs = collect_light_jobs(getattr(ctx, 'import_items', None))
        for fp, group in jobs.items():
            try:
                _fix_from_file(_abspath(fp), group['objects'],
                               group['lights'])
            except blend279.BlendError as e:
                _li.safe_print(f'[halcyon append watch] {fp}: {e}')
            except OSError as e:
                _li.safe_print(
                    f'[halcyon append watch] cannot read {fp}: {e}')
    except Exception:                                           # noqa: BLE001
        import traceback
        traceback.print_exc()


def register():
    h = getattr(getattr(bpy, 'app', None), 'handlers', None)
    lst = getattr(h, 'blend_import_post', None)
    if isinstance(lst, list) and on_blend_import_post not in lst:
        lst.append(on_blend_import_post)


def unregister():
    h = getattr(getattr(bpy, 'app', None), 'handlers', None)
    lst = getattr(h, 'blend_import_post', None)
    if isinstance(lst, list) and on_blend_import_post in lst:
        lst.remove(on_blend_import_post)
