import os
import sys
import threading

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from actions import DEFAULT_BINDINGS
from doppler_viz import Viz
from dsp import Frame
from gesture_app import Controller
from gestures import Dropout, Gesture, GestureEvent, Live, RecognizerParams, ThresholdRecognizer
from levels import Levels, Status
from pipeline import Tick

HOP_S = 0.04
HISTORY = 20
OFFSETS = np.linspace(-400, 400, 31)


def make_viz():
    return Viz(OFFSETS, 19000.0, 400.0, HISTORY)


def live_tick(i, shift, motion=1e-5):
    obs = Live(i * HOP_S, shift, motion)
    frame = Frame(OFFSETS, np.full(OFFSETS.size, -50.0), shift, 19000.0, motion)
    return Tick(obs.t, Levels(Status.OK, -20.0, -60.0), frame, obs)


def dropout_tick(i, status=Status.NO_CARRIER):
    obs = Dropout(i * HOP_S, status)
    frame = Frame(OFFSETS, np.zeros(OFFSETS.size), 0.0, 19000.0, 0.0)
    return Tick(obs.t, Levels(status, -90.0, -85.0), frame, obs)


def event(gesture, i=0):
    return GestureEvent(gesture, i * HOP_S, 100.0, 0.1)


def test_update_draws_trace_spectrum_and_readout():
    viz = make_viz()
    viz.update([(live_tick(0, 110.0), []), (live_tick(1, 55.0), [])])
    assert viz.shift_line.get_ydata()[-2:].tolist() == [110.0, 55.0]
    assert viz.line.get_ydata().tolist() == [-50.0] * OFFSETS.size
    assert viz.img.get_array()[-1, 0] == -50.0
    assert "shift   +55.0 Hz" in viz.readout.get_text()
    assert "m/s" in viz.readout.get_text() and "snr" in viz.readout.get_text()
    assert viz.alarm.get_text() == ""


def test_dropout_shows_alarm_and_zero_shift():
    viz = make_viz()
    viz.update([(live_tick(0, 80.0), []), (dropout_tick(1), [])])
    assert viz.shift_line.get_ydata()[-1] == 0.0
    assert viz.alarm.get_text() == "NO CARRIER"
    assert "---" in viz.readout.get_text() and "NO CARRIER" in viz.readout.get_text()
    viz.update([(live_tick(2, 10.0), [])])
    assert viz.alarm.get_text() == ""


def test_empty_update_changes_nothing():
    viz = make_viz()
    viz.update([(live_tick(0, 90.0), [])])
    before = viz.shift_line.get_ydata().copy()
    viz.update([])
    assert viz.shift_line.get_ydata().tolist() == before.tolist()


def marker_points(viz, gesture):
    x, y = viz.markers[gesture].get_data()
    return list(zip(x, y))


def test_event_marker_sits_on_trace_and_scrolls_with_it():
    viz = make_viz()
    viz.update([(live_tick(0, 0.0), []), (live_tick(1, 120.0), [event(Gesture.PUSH, 1)])])
    assert marker_points(viz, Gesture.PUSH) == [(-1, 120.0)]
    assert marker_points(viz, Gesture.PULL) == []
    viz.update([(live_tick(2, 0.0), []), (live_tick(3, 0.0), [])])
    assert marker_points(viz, Gesture.PUSH) == [(-3, 120.0)]
    assert viz.shift_line.get_ydata()[-3] == 120.0  # same position on the trace


def test_push_and_pull_use_separate_markers_and_expire_off_screen():
    viz = make_viz()
    viz.update([(live_tick(0, 100.0), [event(Gesture.PUSH)]), (live_tick(1, -100.0), [event(Gesture.PULL, 1)])])
    assert marker_points(viz, Gesture.PUSH) == [(-2, 100.0)]
    assert marker_points(viz, Gesture.PULL) == [(-1, -100.0)]
    viz.update([(live_tick(i, 0.0), []) for i in range(2, 2 + HISTORY - 1)])
    assert marker_points(viz, Gesture.PUSH) == []
    assert marker_points(viz, Gesture.PULL) == [(-HISTORY, -100.0)]
    viz.update([(live_tick(99, 0.0), [])])
    assert marker_points(viz, Gesture.PULL) == []
    assert len(viz.marks) == 0


class ScriptedPipeline:
    def __init__(self, ticks):
        self.queue = list(ticks)

    def next_tick(self):
        if not self.queue:
            raise TimeoutError("done")
        return self.queue.pop(0)


def test_viz_shows_events_fired_by_a_controller():
    ticks = [live_tick(i, 0.0, 1e-5) for i in range(5)]
    ticks += [live_tick(5 + i, 120.0, 1e-3) for i in range(8)]
    ctl = Controller(ScriptedPipeline(ticks), ThresholdRecognizer(RecognizerParams()), DEFAULT_BINDINGS, lambda a: None)
    ctl.run(threading.Event())
    viz = Viz(OFFSETS, 19000.0, 400.0, 200)
    viz.update(ctl.drain())
    points = marker_points(viz, Gesture.PUSH)
    assert len(points) == 1 and points[0][1] == 120.0
    assert marker_points(viz, Gesture.PULL) == []


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
