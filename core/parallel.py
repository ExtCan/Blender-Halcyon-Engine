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
import threading

import numpy as np
import sys

# The child is told where to import from with PYTHONPATH rather than by
# editing sys.path, which an extension may not do.
# Written to a file rather than passed with -c. On Windows a -c string reaches
# the child through the shell's quoting rules, and when that goes wrong the
# child exits cleanly having done nothing -- which is indistinguishable from a
# closed pipe, and cost three releases to tell apart. A file has no quoting.
BOOTSTRAP_SOURCE = """import os
import sys
import traceback

# The worker has failed four times with 'exited cleanly having said nothing',
# which is the least informative failure there is. So it keeps a log: every
# step is recorded before it is attempted, and the parent reads the file when a
# worker will not answer. Guessing has been more expensive than logging.
LOG = os.environ.get('HALCYON_WORKER_LOG')


def note(msg):
    if not LOG:
        return
    try:
        with open(LOG, 'a', encoding='utf-8') as fh:
            fh.write(str(msg) + chr(10))
    except Exception:
        pass


note('bootstrap start ' + sys.executable)
note('version ' + sys.version.replace(chr(10), ' '))
note('stdin  ' + repr(sys.stdin))
note('stdout ' + repr(sys.stdout))
try:
    note('stdin fileno ' + repr(sys.stdin.fileno()))
except Exception as exc:
    note('stdin fileno FAILED ' + repr(exc))
try:
    import numpy
    note('numpy ' + numpy.__version__)
except Exception:
    note('numpy import FAILED')
    note(traceback.format_exc())
    raise
root = os.environ.get('HALCYON_WORKER_ROOT')
if root:
    sys.path.insert(0, root)
pkg = os.environ.get('HALCYON_WORKER_PKG')
alias = os.environ.get('HALCYON_WORKER_ALIAS')
note('root  ' + repr(root))
note('pkg   ' + repr(pkg))
note('alias ' + repr(alias))
if alias and alias != pkg:
    # Make the package answer to the name it had inside Blender, so pickles
    # made there resolve here. A finder rather than a fixed list, because we
    # cannot know in advance which submodules a pickle will reach for.
    import importlib
    import types
    from importlib.machinery import ModuleSpec

    class _AliasLoader(object):
        def __init__(self, module):
            self.module = module

        def create_module(self, spec):
            return self.module

        def exec_module(self, module):
            pass

    class _AliasFinder(object):
        def __init__(self, alias_name, real_name):
            self.alias = alias_name
            self.real = real_name

        def find_spec(self, fullname, path=None, target=None):
            # pickle imports the top of the dotted path first, so the parents
            # of the alias have to exist as well: 'bl_ext' and
            # 'bl_ext.user_default' before 'bl_ext.user_default.halcyon'.
            if self.alias.startswith(fullname + '.'):
                stub = types.ModuleType(fullname)
                stub.__path__ = []
                return ModuleSpec(fullname, _AliasLoader(stub),
                                  is_package=True)
            if fullname != self.alias and not fullname.startswith(self.alias + '.'):
                return None
            suffix = fullname[len(self.alias):]
            try:
                module = importlib.import_module(self.real + suffix)
            except ImportError:
                return None
            return ModuleSpec(fullname, _AliasLoader(module),
                              is_package=hasattr(module, '__path__'))

    sys.meta_path.insert(0, _AliasFinder(alias, pkg))
    note('alias finder installed: ' + alias + ' -> ' + pkg)

try:
    mod = __import__(pkg + '.core.worker', fromlist=['main'])
except Exception:
    note('import FAILED')
    note(traceback.format_exc())
    raise
note('entering main')
try:
    mod.main()
    note('main returned normally')
except Exception:
    note('main RAISED')
    note(traceback.format_exc())
    raise
"""


def _bootstrap_path():
    import hashlib
    import tempfile
    tag = hashlib.md5(BOOTSTRAP_SOURCE.encode('utf-8')).hexdigest()[:10]
    path = os.path.join(tempfile.gettempdir(), 'halcyon_worker_%s.py' % tag)
    if not os.path.exists(path):
        tmp = path + '.%d' % os.getpid()
        with open(tmp, 'w', encoding='utf-8') as fh:
            fh.write(BOOTSTRAP_SOURCE)
        try:
            os.replace(tmp, path)
        except OSError:
            pass
    return path


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


def import_name():
    """What this package is called *inside Blender*.

    Installed as an extension it is `bl_ext.user_default.halcyon_render`, and
    every dataclass pickled from it carries that module path. A worker that
    imports the same code as plain `halcyon_render` cannot resolve those names,
    and unpickling the scene fails with `No module named 'bl_ext'` -- which for
    five releases looked like the worker exiting for no reason.
    """
    return __name__.rsplit('.core.parallel', 1)[0]


class Worker:
    def __init__(self, exe, parent, pkg):
        env = dict(os.environ)
        env.pop('PYTHONHOME', None)
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = (parent + os.pathsep + existing) if existing else parent
        env['HALCYON_WORKER_ROOT'] = parent
        env['HALCYON_WORKER_PKG'] = pkg
        env['HALCYON_WORKER_ALIAS'] = import_name()
        import tempfile
        self.log = os.path.join(tempfile.gettempdir(),
                                'halcyon_worker_%d_%d.log'
                                % (os.getpid(), id(self) & 0xffff))
        try:
            if os.path.exists(self.log):
                os.remove(self.log)
        except OSError:
            pass
        env['HALCYON_WORKER_LOG'] = self.log
        self.bootstrap = _bootstrap_path()
        self.proc = subprocess.Popen(
            [exe, self.bootstrap], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            cwd=parent)
        self.lock = threading.Lock()
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
        with self.lock:
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
            tail = f' | worker exited with code {self.proc.returncode}'
            return tail + self.log_tail()
        except Exception:
            return ''

    def log_tail(self, limit=600):
        """What the child recorded about its own startup."""
        try:
            with open(self.log, encoding='utf-8') as fh:
                text = fh.read().strip()
            if not text:
                return ' | worker log is empty (it never started)'
            return ' | worker log: ' + text.replace(chr(10), ' / ')[-limit:]
        except FileNotFoundError:
            return ' | no worker log was written (the bootstrap never ran)'
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
                              f' | importing {pkg} from {parent}'
                              f' | bootstrap {_bootstrap_path()}')
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
                if reply[0] == 'band_raw':
                    # raw bytes plus shape: pickling the array costs the parent
                    # interpreter-lock time it cannot overlap with the other
                    # workers still rendering
                    shape, blob = reply[1], reply[2]
                    out[i] = np.frombuffer(blob, dtype=np.float32).reshape(shape)
                elif reply[0] == 'band':
                    out[i] = reply[1]
                else:
                    errors.append(str(reply)[:200])
                    continue
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

    # One band per worker, not three. Every band repeats the fixed cost of a
    # render call -- projecting the vertices, preparing the light lookups --
    # and with a scissor keeping the rasterisation proportional there is little
    # left to load-balance. Three bands each made the pool slower than not
    # using it at all.
    n_bands = min(count, H)
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
