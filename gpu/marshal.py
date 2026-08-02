"""Run GPU bursts on Blender's main thread while the render thread stays free.

`bl_use_gpu_context` hands the render thread a GPU context by FREEZING the
interface: Blender cannot draw a window while another thread holds the
context, so a 33-second frame was 33 seconds of "Not Responding" -- the
field named it twice. The viewport preview's architecture is the answer,
applied to F12: the heavy work (export, rasterising, sorting, compositing,
every CPU fallback) runs on the worker with NO context, and the moments
that genuinely need the driver -- compile, upload, draw, read back -- are
marshalled here, to the main thread, which always has one. Each crossing
is a burst of milliseconds; the interface breathes between bursts.

Mechanics: the worker queues a callable and blocks on an event; a
`bpy.app.timers` timer -- registering one from a worker thread is
Blender's own documented cross-thread pattern -- drains the queue on the
main thread and sets the event. Every exit stays honest:

- marshalling disabled, or already on the main thread (the self-test's
  operator, a background render), or no bpy at all (the headless test
  suite): the callable simply runs where it was called;
- the timer never fires (a main thread that is not pumping events):
  `MarshalTimeout` after a bounded wait, which the call sites turn into
  their usual (None, why) CPU fallback with the reason printed;
- the callable raises: the exception crosses back to the worker intact.

The timer outlives individual bursts and unregisters itself once
marshalling is disabled and the queue is dry, so a finished render leaves
nothing running.
"""

import queue
import threading

_JOBS = queue.Queue()
_STATE = {'enabled': False, 'timer': False}

#: how long a worker waits for the main thread to PICK a burst UP before
#: falling back to the CPU. This bounds "is the main loop alive?" and
#: nothing else -- a burst that has STARTED may run as long as it needs.
#: The first field frame of the layer port taught the difference: an
#: 8-second cap on start-to-FINISH timed out a burst that was busy
#: succeeding, discarded its work, and paid the CPU path on top.
TIMEOUT = 8.0

#: how often the drain timer fires while marshalling is on.
TICK = 0.008

#: after running a burst, the pump lingers this long for the next one --
#: a worker streaming device calls (upload, draw, read, upload, ...)
#: lands each next call inside the window, so a whole call sequence
#: crosses in ONE timer slice instead of paying a tick per call.
LINGER = 0.002


class MarshalTimeout(RuntimeError):
    """The main thread did not pick the burst up in time."""


#: crossing accounting since the last reset: how many bursts crossed,
#: how long they RAN on the main thread, and how long the worker spent
#: WAITING wall-clock -- wait minus exec is pure marshal latency, the
#: number that decides whether the crossings themselves need work
ACCT = {'crossings': 0, 'exec_ms': 0.0, 'wall_ms': 0.0}


def acct_reset():
    ACCT.update(crossings=0, exec_ms=0.0, wall_ms=0.0)


def acct():
    return dict(ACCT)


class _Job:
    __slots__ = ('fn', 'box', 'started', 'done', 'abandoned')

    def __init__(self, fn):
        self.fn = fn
        self.box = {}
        self.started = threading.Event()
        self.done = threading.Event()
        self.abandoned = False

    def run(self):
        if self.abandoned:
            # the worker gave up waiting and fell back to the CPU --
            # running the work now would block the interface for a
            # result nobody will read
            self.done.set()
            return
        import time as _time
        self.started.set()
        t0 = _time.perf_counter()
        try:
            self.box['r'] = self.fn()
        except BaseException as exc:                            # noqa: BLE001
            self.box['e'] = exc
        finally:
            ACCT['exec_ms'] += (_time.perf_counter() - t0) * 1000.0
            self.done.set()


def _pump():
    """Main-thread timer body: run queued bursts, linger, breathe."""
    did = False
    while True:
        try:
            job = _JOBS.get_nowait()
        except queue.Empty:
            if not did or not _STATE['enabled']:
                break
            try:
                job = _JOBS.get(timeout=LINGER)
            except queue.Empty:
                break
        did = True
        job.run()
    if not _STATE['enabled'] and _JOBS.empty():
        _STATE['timer'] = False
        return None                     # unregister: the render is over
    return TICK


def _ensure_timer():
    if _STATE['timer']:
        return
    try:
        import bpy
        bpy.app.timers.register(_pump, first_interval=0.0)
        _STATE['timer'] = True
    except Exception:                                           # noqa: BLE001
        _STATE['timer'] = False


def enable():
    """Marshalling on for the duration of a render."""
    _STATE['enabled'] = True
    _ensure_timer()


def disable():
    """Marshalling off; the timer retires itself on its next tick."""
    _STATE['enabled'] = False


def enabled():
    return bool(_STATE['enabled'])


def on_main():
    return threading.current_thread() is threading.main_thread()


def run_on_main(fn, timeout=None, what='GPU work'):
    """fn() on the main thread when marshalling is on; in place otherwise.

    Returns fn's result; re-raises fn's exception on the calling thread.
    The timeout bounds only the PICKUP -- how long the main loop may take
    to start the burst. A burst that has started runs to completion,
    however long it takes: the timeout answers "is the main loop alive?",
    never "is this frame cheap?". Raises MarshalTimeout only when the
    main thread never picked the job up -- callers treat that exactly
    like any other GPU failure: a reason, and the CPU path. An abandoned
    job is skipped if the timer fires later, so no interface ever blocks
    on work whose result was already given up on.
    """
    if not _STATE['enabled'] or on_main():
        return fn()
    try:
        import bpy                                              # noqa: F401
    except Exception:                                           # noqa: BLE001
        return fn()                     # headless: no main loop to borrow
    _ensure_timer()
    if not _STATE['timer']:
        return fn()                     # no timers either: run in place
    import time as _time
    job = _Job(fn)
    t0 = _time.perf_counter()
    _JOBS.put(job)
    if not job.started.wait(TIMEOUT if timeout is None else timeout):
        job.abandoned = True
        if not job.started.is_set():
            raise MarshalTimeout(
                f'{what}: the main thread did not pick the GPU burst up '
                'in time (is it blocked?); this frame shades on the CPU '
                'instead')
        # the pickup raced the deadline: the burst is running -- take it
    while not job.done.wait(1.0):
        pass
    ACCT['crossings'] += 1
    ACCT['wall_ms'] += (_time.perf_counter() - t0) * 1000.0
    if 'e' in job.box:
        raise job.box['e']
    return job.box.get('r')
