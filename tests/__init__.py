"""The headless suite (bpy-free). `python -m halcyon.tests.run_all`."""


def utf8_console():
    """Make the suite's own output survive a Windows console.

    R250: the first run of the suite on the user's machine (Windows 11,
    Blender 5.2's Python) aborted a renderer test in the middle of a
    passing check -- `print` raised UnicodeEncodeError on the '▸' in a
    warning's text, because a redirected stdout there is cp1252, not UTF-8
    (the Linux container that ran R1-R249 never saw it). A check that
    crashes while printing 'ok' is a false failure and hides every check
    after it; the console is reconfigured once, replacing what it cannot
    encode, and the engine's own output is never touched.
    """
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            enc = str(getattr(stream, 'encoding', '') or '')
            if enc.lower().replace('-', '') != 'utf8' and                     hasattr(stream, 'reconfigure'):
                stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:                                       # noqa: BLE001
            pass
