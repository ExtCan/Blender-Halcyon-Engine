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

import os
import pickle
import struct
import sys

_STATE = {'scene': None, 'settings': None}


def _log(msg):
    """Append to the bootstrap's log, if one was asked for."""
    path = os.environ.get('HALCYON_WORKER_LOG')
    if not path:
        return
    try:
        with open(path, 'a', encoding='utf-8') as fh:
            fh.write(str(msg) + '\n')
    except Exception:                                           # noqa: BLE001
        pass


def _read(stream):
    head = stream.read(4)
    if not head or len(head) < 4:
        _log('read header got %d bytes: %r' % (len(head or b''), head))
        return None
    (size,) = struct.unpack('>I', head)
    buf = bytearray()
    while len(buf) < size:
        chunk = stream.read(size - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return pickle.loads(bytes(buf))


class _RawOut:
    """Write straight to a file descriptor.

    No buffering, no wrapper whose lifetime can end the stream. Every previous
    attempt at this protocol went through a BufferedWriter borrowed from
    sys.stdout, and each time the failure looked like a clean end of stream.
    A descriptor has none of that behaviour.
    """

    __slots__ = ('fd',)

    def __init__(self, fd):
        self.fd = fd

    def write(self, data):
        view = memoryview(data)
        while view:
            written = os.write(self.fd, view)
            view = view[written:]

    def flush(self):
        pass


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
    # keep the original wrapper alive as well as duplicating its descriptor:
    # collecting it closes the pipe underneath, which is the bug this replaced
    sys.__halcyon_stdout__ = sys.stdout
    try:
        fd = os.dup(sys.stdout.fileno())
        if hasattr(os, 'set_inheritable'):
            os.set_inheritable(fd, False)
        stdout = _RawOut(fd)
        _log('protocol stream is raw fd %d' % fd)
    except Exception as exc:                                    # noqa: BLE001
        _log('os.dup failed (%r), falling back to sys.stdout.buffer' % (exc,))
        stdout = sys.stdout.buffer
    # anything printed by the renderer must not land in the protocol stream
    sys.stdout = sys.stderr
    _log('loop start, stdin=%r' % (stdin,))
    while True:
        try:
            msg = _read(stdin)
        except Exception as exc:                                # noqa: BLE001
            # a read error used to break silently, which is indistinguishable
            # from a clean end of stream -- and that ambiguity cost four rounds
            import traceback
            _log('read RAISED %r' % (exc,))
            _log(traceback.format_exc())
            break
        if msg is None:
            _log('end of stream, leaving')
            break
        _log('got message %r' % (msg[0],))
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
                # the array travels as raw bytes with its shape alongside.
                # Pickling it costs the parent interpreter-lock time that
                # cannot overlap with the other workers still running.
                import numpy as _np
                arr = _np.ascontiguousarray(arr, dtype=_np.float32)
                _write(stdout, ('band_raw', arr.shape, arr.tobytes()))
            elif kind == 'ping':
                _write(stdout, ('ok',))
                _log('ping answered')
            else:
                _write(stdout, ('error', f'unknown message {kind!r}'))
        except Exception as exc:                                # noqa: BLE001
            import traceback
            _log('handling %r RAISED %r' % (kind, exc))
            _log(traceback.format_exc())
            traceback.print_exc(file=sys.stderr)
            try:
                _write(stdout, ('error', f'{type(exc).__name__}: {exc}'))
            except Exception:                                   # noqa: BLE001
                break


if __name__ == '__main__':
    main()
