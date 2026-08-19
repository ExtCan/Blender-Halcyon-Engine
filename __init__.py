"""Halcyon -- a from-scratch render engine for Blender 5.2 that targets the
output of mid-to-late 1990s home-computer 3D software.

Everything under halcyon/core/ and halcyon/shaders/ is free of any bpy import,
so the renderer, the shader compiler and the post chain can all be run and
tested without Blender at all:

    python3 -m halcyon.tests.run_all
"""

bl_info = {
    "name": "Halcyon Render Engine",
    "author": "Built by Claude with help from Mr. Emotiman",
    "version": (1, 36, 0),
    "blender": (5, 1, 0),
    "location": "Render Properties > Render Engine > Halcyon",
    "description": "Scanline/raytrace hybrid engine reproducing mid-to-late "
                   "1990s CGI, with classic shading models and coded GLSL/HLSL "
                   "shader nodes",
    "warning": "",
    "doc_url": "",
    "category": "Render",
}

_MODULES = ()
_FAULT_LOG = None
_FAULT_NOTES = {}


def _enable_faulthandler():
    """Native-crash tracebacks with PYTHON line numbers, to a file.

    The field's crash reports carry driver stacks but no Python frames;
    with this on, %TEMP%/halcyon_faulthandler.log names the exact
    Halcyon line active on every thread at the moment of a native
    crash. Costs nothing until a crash happens.

    CAVEAT, learned from the first field log: on Windows, faulthandler
    prints every FIRST-CHANCE access violation -- including ones a
    driver or the OS raises and HANDLES as part of normal operation
    (guarded probes, memory-mapped I/O, antivirus filters), after which
    the application carries on. An entry proves a crash only when the
    log goes silent after it. The milestone notes fault_note() writes
    between exceptions are what make the log readable as a timeline.
    """
    global _FAULT_LOG
    try:
        import faulthandler
        import os
        import tempfile
        from . import version as _v
        path = os.path.join(tempfile.gettempdir(),
                            'halcyon_faulthandler.log')
        _FAULT_LOG = open(path, 'a', buffering=1)
        _FAULT_LOG.write(
            f'\n--- Halcyon {_v.version_string()} session start ---\n'
            '(note: Windows logs some access violations that the\n'
            ' application SURVIVES; an entry is fatal only when no\n'
            ' milestone line ever follows it)\n')
        faulthandler.enable(_FAULT_LOG, all_threads=True)
    except Exception:                                           # noqa: BLE001
        _FAULT_LOG = None


def fault_note(msg, key=None, limit=6):
    """A timestamped milestone into the faulthandler log.

    These lines turn the log into a timeline: exceptions FOLLOWED by a
    milestone were survived; the fatal one is the entry nothing follows.
    Per-key limited so a per-frame call site cannot flood the file.
    """
    try:
        if _FAULT_LOG is None:
            return
        k = key or msg
        n = _FAULT_NOTES.get(k, 0)
        if n >= limit:
            return
        _FAULT_NOTES[k] = n + 1
        import time as _t
        _FAULT_LOG.write(f'[{_t.strftime("%H:%M:%S")}] OK: {msg}\n')
    except Exception:                                           # noqa: BLE001
        pass


def _import_modules():
    from . import (append_watch, compat, convert, engine, export,
                   legacy_import, objects, properties, selftest,
                   templates, ui)
    from .nodes import shader_nodes
    return (properties, shader_nodes, convert, templates, legacy_import,
            append_watch, export, objects, engine, selftest, ui)


def register():
    global _MODULES
    _enable_faulthandler()
    _MODULES = _import_modules()
    for mod in _MODULES:
        if hasattr(mod, 'register'):
            mod.register()


def unregister():
    global _FAULT_LOG
    # the CLEAN-SHUTDOWN marker, written before anything is torn down.
    # A field log that simply STOPPED after a parked frame could not be
    # told apart from a crash: an orderly Blender quit (or add-on
    # disable) runs unregister and leaves this line; a crash, a driver
    # device-loss kill or a frozen process never does. 'log ends
    # without the shutdown line' now MEANS abnormal end.
    try:
        fault_note('addon unregistered (clean shutdown)', key='bye')
    except Exception:                                           # noqa: BLE001
        pass
    for mod in reversed(_MODULES):
        if hasattr(mod, 'unregister'):
            try:
                mod.unregister()
            except Exception:                                   # noqa: BLE001
                import traceback
                traceback.print_exc()
    try:
        import faulthandler
        faulthandler.disable()
        if _FAULT_LOG is not None:
            _FAULT_LOG.close()
    except Exception:                                           # noqa: BLE001
        pass
    _FAULT_LOG = None


if __name__ == "__main__":
    register()
