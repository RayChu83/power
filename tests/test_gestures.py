import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures import (
    Dropout,
    Gesture,
    GestureEvent,
    Idle,
    Live,
    Lockout,
    RecognizerParams,
    ThresholdRecognizer,
    Tracking,
)
from levels import Status

# Dyadic values so time comparisons at thresholds are exact in floating point.
HOP = 0.0625
P = RecognizerParams(
    motion_on=1e-4,
    motion_off=3e-5,
    shift_on_hz=20.0,
    min_duration_s=0.125,
    gap_s=0.125,
    lockout_s=0.5,
)
LOUD = 1e-3
QUIET = 1e-5
MID = 5e-5  # between motion_off and motion_on


def at(k):
    return k * HOP


def live(k, shift, motion=LOUD):
    return Live(at(k), shift, motion)


def run(obs, params=P):
    rec = ThresholdRecognizer(params)
    events = []
    for o in obs:
        events += rec.feed(o)
    return events, rec


def test_positive_burst_fires_one_push():
    events, _ = run([live(k, 80.0) for k in range(10)])
    assert [e.gesture for e in events] == [Gesture.PUSH], events


def test_negative_burst_fires_one_pull():
    events, _ = run([live(k, -80.0) for k in range(10)])
    assert [e.gesture for e in events] == [Gesture.PULL], events


def test_push_then_retract_inside_lockout_fires_only_push():
    # Push ends by k=5; the retract runs k=6..9, all inside lockout_s of the emit at k=2.
    obs = [live(k, 80.0) for k in range(6)] + [live(k, -80.0) for k in range(6, 12)]
    events, _ = run(obs)
    assert [e.gesture for e in events] == [Gesture.PUSH], events


def test_second_gesture_fires_after_lockout():
    first = [live(k, 80.0) for k in range(4)]
    quiet = [live(k, 0.0, QUIET) for k in range(4, 20)]
    second = [live(k, -80.0) for k in range(20, 24)]
    events, _ = run(first + quiet + second)
    assert [e.gesture for e in events] == [Gesture.PUSH, Gesture.PULL], events


def test_frame_that_ends_lockout_is_processed_as_idle():
    # Emit at k=2, lockout_s=0.5 -> until = at(2) + 0.5 = at(10). A burst starting at k=10 tracks.
    obs = [live(k, 80.0) for k in range(3)] + [live(k, 0.0, QUIET) for k in range(3, 10)]
    rec = ThresholdRecognizer(P)
    for o in obs:
        rec.feed(o)
    assert isinstance(rec.state, Lockout), rec.state
    rec.feed(live(9, -80.0))
    assert isinstance(rec.state, Lockout), rec.state
    rec.feed(live(10, -80.0))
    assert isinstance(rec.state, Tracking), rec.state


def test_motion_below_motion_on_fires_nothing():
    events, rec = run([live(k, 80.0, MID) for k in range(20)])
    assert events == [] and isinstance(rec.state, Idle), (events, rec.state)


def test_shift_below_shift_on_fires_nothing():
    events, rec = run([live(k, 10.0) for k in range(20)])
    assert events == [] and isinstance(rec.state, Idle), (events, rec.state)


def test_burst_shorter_than_min_duration_fires_nothing():
    # Two active frames span 0.0625 s, under min_duration_s = 0.125.
    obs = [live(0, 80.0), live(1, 80.0)] + [live(k, 0.0, QUIET) for k in range(2, 12)]
    events, rec = run(obs)
    assert events == [] and isinstance(rec.state, Idle), (events, rec.state)


def test_short_blip_does_not_fire_on_a_dip_frame():
    # gap_s == min_duration_s: the dip frame at k=2 is 0.125 s after the start but must not emit.
    obs = [live(0, 80.0), live(1, 0.0, QUIET), live(2, 0.0, QUIET), live(3, 0.0, QUIET)]
    events, _ = run(obs)
    assert events == [], events


def test_mid_burst_sign_flip_fires_nothing():
    events, rec = run([live(0, 80.0), live(1, -80.0)])
    assert events == [] and isinstance(rec.state, Idle), (events, rec.state)


def test_dip_does_not_flip_sign():
    # push, quiet dip within gap_s, push again: the dip must not read as a sign change, one PUSH.
    obs = [live(0, 80.0), live(1, 0.0, QUIET), live(2, 80.0), live(3, 80.0)]
    events, _ = run(obs)
    assert [e.gesture for e in events] == [Gesture.PUSH], events


def test_dip_with_opposite_sign_shift_below_thresholds_does_not_flip():
    # Low motion with a negative shift is a dip, not a pull.
    obs = [live(0, 80.0), live(1, -80.0, MID * 0.1), live(2, 80.0), live(3, 80.0)]
    events, _ = run(obs)
    assert [e.gesture for e in events] == [Gesture.PUSH], events


def test_dip_longer_than_gap_ends_the_burst():
    obs = [live(0, 80.0), live(1, 0.0, QUIET), live(2, 0.0, QUIET), live(3, 0.0, QUIET), live(4, 80.0)]
    events, rec = run(obs)
    assert events == [], events
    assert isinstance(rec.state, Tracking) and rec.state.t0 == at(4), rec.state


def test_dropout_mid_burst_fires_nothing_and_clears_state():
    obs = [live(0, 80.0), live(1, 80.0), Dropout(at(2), Status.NO_CARRIER), live(3, 80.0)]
    events, rec = run(obs[:3])
    assert events == [] and isinstance(rec.state, Idle), (events, rec.state)
    # The burst does not resume across the dropout: it needs a fresh min_duration_s.
    events, rec = run(obs)
    assert events == [] and isinstance(rec.state, Tracking) and rec.state.t0 == at(3), (events, rec.state)


def test_dropout_during_lockout_does_not_cancel_it():
    obs = [live(k, 80.0) for k in range(3)]
    obs += [Dropout(at(3), Status.MIC_SILENT), live(4, -80.0), live(5, -80.0), live(6, -80.0)]
    events, rec = run(obs)
    assert [e.gesture for e in events] == [Gesture.PUSH], events
    assert isinstance(rec.state, Lockout), rec.state


def test_dropout_after_lockout_ends_returns_to_idle():
    obs = [live(k, 80.0) for k in range(3)] + [Dropout(at(11), Status.NO_CARRIER)]
    _, rec = run(obs)
    assert isinstance(rec.state, Idle), rec.state


def test_event_fields():
    # Start at k=2 (t=0.125); active frames to k=4 emit at t=0.25 with peak 140 Hz.
    obs = [live(0, 0.0, QUIET), live(1, 0.0, QUIET), live(2, 60.0), live(3, 140.0), live(4, 90.0), live(5, 200.0)]
    events, _ = run(obs)
    assert events == [GestureEvent(Gesture.PUSH, at(4), 140.0, at(4) - at(2))], events


def test_pull_event_strength_is_absolute():
    events, _ = run([live(k, -50.0 - 10 * k) for k in range(4)])
    assert len(events) == 1 and events[0].gesture is Gesture.PULL, events
    assert events[0].strength_hz == 70.0 and events[0].t == at(2), events[0]


def test_event_is_emitted_before_burst_ends():
    rec = ThresholdRecognizer(P)
    emitted = []
    for k in range(3):
        emitted += rec.feed(live(k, 80.0))
    assert len(emitted) == 1 and emitted[0].t == at(2), emitted


def test_bad_hysteresis_params_rejected():
    try:
        RecognizerParams(motion_on=1e-4, motion_off=1e-4)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_default_params_recognize_a_typical_push():
    p = RecognizerParams()
    events, _ = run([Live(0.04 * k, 80.0, 1e-3) for k in range(8)], p)
    assert [e.gesture for e in events] == [Gesture.PUSH], events


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
