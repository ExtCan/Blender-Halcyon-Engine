"""A pool of real worker processes, for splitting a frame across cores.

Threads help where NumPy releases the GIL, which is most of the array work, but
not where the time goes into Python itself — node evaluation, the per-material
dispatch, the shading context. Separate processes have no shared interpreter
lock at all.

This is only possible because core/ and shaders/ import nothing from bpy: a
worker is a plain Python interpreter with NumPy, and Blender ships one. The pool
is deliberately conservative — if anything at all goes wrong finding an
interpreter, starting a worker or talking to one, it reports why and the caller
renders in-process instead. A slow render is a far better failure than none.
"""

import os
import pickle
import struct
import subprocess
import sys

# The child is told where to import from with PYTHONPATH rather than by
# editing sys.path, which an extension may not do.
BOOTSTRAP = "from {pkg}.core import worker; worker.main()"


def find_interpreter():
    """A Python executable with NumPy. Blender bundles one next to sys.prefix."""
    cands = []
    ver = f'python{sys.version_info.major}.{sys.version_info.minor}'
    for sub in (('bin', 'python.exe'), ('bin', f'{ver}.exe'), ('python.exe',),
                ('bin', ver), ('bin', 'python3'), ('bin', 'python')):
        cands.append(os.path.join(sys.prefix, *sub))
    base = os.path.basename(sys.executable).lower()
    if base.startswith('python'):
        cands.append(sys.executable)
    for c in cands:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def package_location():
    """(parent directory, package name) for importing this add-on in a worker."""
    here = os.path.dirname(os.path.abspath(__file__))        # .../<pkg>/core
    pkg_dir = os.path.dirname(here)                           # .../<pkg>
    return os.path.dirname(pkg_dir), os.path.basename(pkg_dir)


class Worker:
    def __init__(self, exe, parent, pkg):
        code = BOOTSTRAP.format(pkg=pkg)
        env = dict(os.environ)
        env.pop('PYTHONHOME', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = (parent + os.pathsep + existing) if existing else parent
        self.proc = subprocess.Popen(
            [exe, '-c', code], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, env=env, cwd=parent)
        self.alive = True

    def send(self, msg):
        payload = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
        self.proc.stdin.write(struct.pack('>I', len(payload)))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    def recv(self):
        head = self.proc.stdout.read(4)
        if not head or len(head) < 4:
            raise IOError('worker closed the connection')
        (size,) = struct.unpack('>I', head)
        buf = bytearray()
        while len(buf) < size:
            chunk = self.proc.stdout.read(size - len(buf))
            if not chunk:
                raise IOError('worker closed mid-message')
            buf.extend(chunk)
        return pickle.loads(bytes(buf))

    def call(self, msg):
            try:
                self.send(msg)
                return self.recv()
            except Exception as exc:
                # a worker that dies on startup says why on stderr, and that
                # is the only thing that makes the failure diagnosable
                raise IOError(str(exc) + self.stderr_tail()) from exc

    def stderr_tail(self, limit=400):
        try:
            if self.proc.poll() is None:
                # alive but not answering: kill it so its pipes close and
                # whatever it managed to say can be read
                try:
                    self.proc.kill()
                    self.proc.wait(timeout=2)
                except Exception:
                    return ' | worker is alive but did not answer'
            data = self.proc.stderr.read() or b''
            text = data.decode('utf-8', 'replace').strip()
            if text:
                return ' | worker said: ' + text[-limit:]
            return f' | worker exited with code {self.proc.returncode}'
        except Exception:
            return ''

    def close(self):
        self.alive = False
        try:
            self.send(('quit',))
        except Exception:                                       # noqa: BLE001
            pass
        try:
            self.proc.wait(timeout=2)
        except Exception:                                       # noqa: BLE001
            try:
                self.proc.kill()
            except Exception:                                   # noqa: BLE001
                pass


class Pool:
    """Persistent workers. Keeping them alive across frames matters: starting an
    interpreter and importing NumPy costs a good fraction of a second, which
    would swamp the saving on a single frame."""

    def __init__(self, count):
        self.workers = []
        self.error = None
        self._scene_key = None
        exe = find_interpreter()
        if exe is None:
            self.error = ("no Python interpreter found to run workers "
                          f"(looked under {sys.prefix})")
            return
        parent, pkg = package_location()
        self.interpreter = exe
        self.bootstrap = (parent, pkg)
        for _ in range(count):
            try:
                w = Worker(exe, parent, pkg)
                reply = w.call(('ping',))
                if reply[0] != 'ok':
                    raise IOError(f'worker refused handshake: {reply!r}')
                self.workers.append(w)
            except Exception as exc:                            # noqa: BLE001
                self.error = (f'{type(exc).__name__}: {exc}'
                              f' | interpreter {exe}'
                              f' | importing {pkg} from {parent}')
                self.close()
                return

    def ok(self):
        return bool(self.workers) and self.error is None

    def upload(self, scene, settings, key):
        if key is not None and key == self._scene_key:
            return True
        for w in self.workers:
            reply = w.call(('scene', scene, settings))
            if reply[0] != 'ok':
                self.error = f'scene upload failed: {reply!r}'
                return False
        self._scene_key = key
        return True

    def render_bands(self, bands):
        """Farm out (y0, y1) ranges and return the arrays in the same order."""
        # No threads: requests for a round go out to every worker first, then
        # the replies are collected. The workers still run at the same time --
        # that is the entire point -- but this process only ever does one thing
        # at once, which is what an extension is allowed to do.
        out = [None] * len(bands)
        errors = []
        n_workers = len(self.workers)
        for start in range(0, len(bands), n_workers):
            batch = list(enumerate(bands))[start:start + n_workers]
            issued = []
            for (i, (y0, y1)), worker in zip(batch, self.workers):
                try:
                    worker.send(('band', y0, y1))
                    issued.append((i, worker))
                except Exception as exc:                        # noqa: BLE001
                    errors.append(f'{type(exc).__name__}: {exc}')
            for i, worker in issued:
                try:
                    reply = worker.recv()
                except Exception as exc:                        # noqa: BLE001
                    errors.append(f'{type(exc).__name__}: {exc}'
                                  + worker.stderr_tail())
                    continue
                if reply[0] != 'band':
                    errors.append(str(reply))
                    continue
                out[i] = reply[1]
            if errors:
                break
        if errors or any(a is None for a in out):
            self.error = errors[0] if errors else 'a band came back empty'
            return None
        return out

    def close(self):
        for w in self.workers:
            w.close()
        self.workers = []


def _stderr_tail(proc, limit=300):
    if proc is None:
        return ''
    try:
        proc.stdout.close()
        data = proc.stderr.read() or b''
        return data.decode('utf-8', 'replace').strip()[-limit:]
    except Exception:                                           # noqa: BLE001
        return ''


_POOL = None
_POOL_SIZE = 0


def get_pool(count):
    global _POOL, _POOL_SIZE
    if _POOL is not None and _POOL_SIZE == count and _POOL.ok():
        return _POOL
    shutdown()
    _POOL = Pool(count)
    _POOL_SIZE = count
    return _POOL


def shutdown():
    global _POOL, _POOL_SIZE
    if _POOL is not None:
        _POOL.close()
    _POOL = None
    _POOL_SIZE = 0


def render_parallel(scene, settings, workers, scene_key=None, progress=None):
    """Render a frame across worker processes.

    Always returns (image, error). A None image means "render this in-process
    instead" and is a normal outcome, not a failure: a single-core machine, a
    locked-down Python, or a scene that will not pickle should all still render.

    """
    import numpy as np

    H = max(int(settings.resolution_y), 1)
    W = max(int(settings.resolution_x), 1)
    count = max(1, min(int(workers), 64))
    if count < 2:
        return None, 'only one worker requested'
    if H < 4 * count:
        return None, 'frame too short to split usefully'
    if W * H < 64 * 1024:
        # below this the pickling and pipe traffic cost more than the split saves
        return None, 'frame too small to be worth splitting'

    pool = get_pool(count)
    if not pool.ok():
        return None, pool.error

    # more bands than workers, so a slow band cannot leave a core idle
    n_bands = min(count * 3, H)
    edges = [round(i * H / n_bands) for i in range(n_bands + 1)]
    bands = [(edges[i], edges[i + 1]) for i in range(n_bands)
             if edges[i + 1] > edges[i]]

    try:
        if not pool.upload(scene, settings, scene_key):
            return None, pool.error
        if progress:
            progress(0.15, f'Rendering across {len(pool.workers)} processes')
        parts = pool.render_bands(bands)
    except Exception as exc:                                    # noqa: BLE001
        return None, f'{type(exc).__name__}: {exc}'
    if parts is None:
        return None, pool.error
    return np.concatenate(parts, axis=0), None
