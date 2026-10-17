import os
import sys
import threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions import DEFAULT_BINDINGS, Scroll
from dsp import Frame
from gesture_app import TAPE_LEN, Controller
from gestures import Dropout, Live, RecognizerParams, ThresholdRecognizer
from levels import Levels, Status
from pipeline import Tick

HOP_S = 0.04
LOUD = 1e-3  # well above motion_on


class FakePipeline:
    """Yields scripted observations as Ticks, then raises TimeoutError like a stalled stream."""

    def __init__(self, observations):
        self.queue = list(observations)

    def next_tick(self):
        if not self.queue:
            raise TimeoutError("no audio")
        obs = self.queue.pop(0)
        frame = Frame(np.zeros(1), np.zeros(1), getattr(obs, "shift_hz", 0.0), 19000.0, getattr(obs, "motion", 0.0))
        return Tick(obs.t, Levels(Status.OK, -20.0, -60.0), frame, obs)


def idle(n, t0=0.0):
    return [Live(t0 + i * HOP_S, 0.0, 1e-5) for i in range(n)]


def burst(shift_hz, n=8, t0=0.0):
    return [Live(t0 + i * HOP_S, shift_hz, LOUD) for i in range(n)]


def run_script(observations, stop=None):
    actions = []
    pipe = FakePipeline(observations)
    ctl = Controller(pipe, ThresholdRecognizer(RecognizerParams()), DEFAULT_BINDINGS, actions.append)
    ctl.run(stop or threading.Event())
    return ctl, actions


def test_push_burst_scrolls_up_once():
    _, actions = run_script(idle(5) + burst(60.0, t0=0.2) + idle(5, t0=0.52))
    assert actions == [Scroll(+3)], actions


def test_pull_burst_scrolls_down_once():
    _, actions = run_script(idle(5) + burst(-60.0, t0=0.2) + idle(5, t0=0.52))
    assert actions == [Scroll(-3)], actions


def test_idle_and_dropouts_fire_nothing():
    obs = idle(10) + [Dropout(0.4 + i * HOP_S, Status.NO_CARRIER) for i in range(5)] + idle(5, t0=0.6)
    _, actions = run_script(obs)
    assert actions == []


def test_timeout_ends_run_without_hanging():
    result = {}

    def target():
        result["ctl"], result["actions"] = run_script(idle(3))

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(5)
    assert not t.is_alive(), "run hung on a stalled pipeline"
    assert result["ctl"].stalled


def test_preset_stop_consumes_nothing():
    stop = threading.Event()
    stop.set()
    ctl, actions = run_script(burst(60.0), stop)
    assert actions == [] and ctl.drain() == []
    assert not ctl.stalled


def test_drain_returns_ticks_with_their_events_once():
    ctl, _ = run_script(burst(60.0, n=8))
    tape = ctl.drain()
    assert len(tape) == 8
    fired = [e for _, events in tape for e in events]
    assert len(fired) == 1
    assert ctl.drain() == []


def test_tape_is_bounded():
    ctl, _ = run_script(idle(TAPE_LEN + 50))
    assert len(ctl.drain()) == TAPE_LEN


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
