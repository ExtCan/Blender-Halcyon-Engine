"""A very small timing collector.

Three rounds of optimising this engine were aimed at whatever I happened to have
profiled, which was the bpy-free core -- while the export layer, which runs
inside Blender, was never timed at all. This exists so the answer to "where did
the frame go" is measured rather than guessed.
"""

import time

_STAGES = {}
_ORDER = []
_ENABLED = False


def enable(on=True):
    global _ENABLED
    _ENABLED = bool(on)


def reset():
    _STAGES.clear()
    _ORDER.clear()


def add(name, seconds):
    if name not in _STAGES:
        _ORDER.append(name)
        _STAGES[name] = 0.0
    _STAGES[name] += float(seconds)


class track:
    """Context manager: `with stats.track('export'): ...`"""

    __slots__ = ('name', 't0')

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        add(self.name, time.perf_counter() - self.t0)
        return False


def report(total=None, printer=print):
    if not _ORDER:
        return
    total = total or sum(_STAGES.values())
    total = max(total, 1e-9)
    printer("-" * 52)
    printer(f"{'Halcyon frame breakdown':<32}{'time':>9}{'share':>10}")
    printer("-" * 52)
    for name in _ORDER:
        t = _STAGES[name]
        printer(f"  {name:<30}{t:8.3f}s{100 * t / total:9.1f}%")
    # A stage nobody instrumented is exactly where a mystery hides, so the
    # remainder is always shown rather than left to be worked out by subtraction.
    rest = total - sum(_STAGES.values())
    if rest > max(0.005, total * 0.01):
        printer(f"  {'unaccounted for':<30}{rest:8.3f}s{100 * rest / total:9.1f}%")
    printer("-" * 52)
    printer(f"  {'TOTAL':<30}{total:8.3f}s")
    rest = total - sum(_STAGES.values())
    if rest > sum(_STAGES.values()):
        printer(f"  {'TOTAL':<30}{total:8.3f}s")
        printer("  most of this frame is in a stage that is not instrumented")
        printer("-" * 52)
        return
    slowest = max(_ORDER, key=lambda k: _STAGES[k])
    printer(f"  slowest stage: {slowest} "
            f"({100 * _STAGES[slowest] / total:.0f}% of the frame)")
    printer("-" * 52)
