"""One place the version lives.

Installed as an extension, Blender does not expose `bl_info` at all -- it uses
blender_manifest.toml -- so importing it raises. Anything that wants the version
asks here instead, and this tries the manifest first so an installed extension
reports what is actually installed.
"""

import os

VERSION = (1, 24, 0)


def _from_manifest():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'blender_manifest.toml')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line.startswith('version') and '=' in line:
                    raw = line.split('=', 1)[1].strip().strip('"\'')
                    return tuple(int(p) for p in raw.split('.'))
    except Exception:                                           # noqa: BLE001
        pass
    return None


def version():
    return _from_manifest() or VERSION


def version_string():
    return '.'.join(str(v) for v in version())
