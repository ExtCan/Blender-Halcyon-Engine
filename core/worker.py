"""Worker process for parallel rendering.

Runs in a plain Python interpreter with NumPy — no Blender. That is only
possible because everything under core/ and shaders/ is free of bpy imports,
which is the whole reason that boundary exists.

Protocol on stdin/stdout: a 4-byte big-endian length followed by a pickle.

    ('scene', scene, settings)  -> ('ok',)        stores the scene
    ('band', y0, y1)            -> ('band', arr)  renders those output rows
    ('quit',)                   -> exits

The scene is sent once and reused for every band and every frame that follows,
so the cost of shipping it is paid once rather than per tile.
"""

import pickle
import struct
import sys

_STATE = {'scene': None, 'settings': None}


def _read(stream):
    head = stream.read(4)
    if not head or len(head) < 4:
        return None
    (size,) = struct.unpack('>I', head)
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return pickle.loads(bytes(buf))


def _write(stream, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack('>I', len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    import os

    stdin = sys.stdin.buffer
    # Duplicate the descriptor rather than borrowing sys.stdout's buffer.
    # Reassigning sys.stdout drops the last reference to the original wrapper,
    # and when it is collected it closes the pipe underneath -- the parent then
    # sees EOF and the worker exits cleanly having said nothing at all.
    try:
        stdout = os.fdopen(os.dup(sys.stdout.fileno()), 'wb')
    except Exception:
        stdout = sys.stdout.buffer
    # anything printed by the renderer must not land in the protocol stream
    sys.stdout = sys.stderr
    while True:
        try:
            msg = _read(stdin)
        except Exception:                                       # noqa: BLE001
            break
        if msg is None:
            break
        kind = msg[0]
        try:
            if kind == 'quit':
                break
            if kind == 'scene':
                _STATE['scene'] = msg[1]
                _STATE['settings'] = msg[2]
                _write(stdout, ('ok',))
            elif kind == 'band':
                from . import render as core_render
                arr = core_render.render(_STATE['scene'], _STATE['settings'],
                                         band=(msg[1], msg[2]))
                _write(stdout, ('band', arr))
            elif kind == 'ping':
                _write(stdout, ('ok',))
            else:
                _write(stdout, ('error', f'unknown message {kind!r}'))
        except Exception as exc:                                # noqa: BLE001
            import traceback
            traceback.print_exc(file=sys.stderr)
            try:
                _write(stdout, ('error', f'{type(exc).__name__}: {exc}'))
            except Exception:                                   # noqa: BLE001
                break


if __name__ == '__main__':
    main()
