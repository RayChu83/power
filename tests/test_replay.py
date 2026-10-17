import contextlib
import io
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures import Dropout, Gesture, Live, RecognizerParams
from levels import Status
from replay import load_params, main, match_labels, run_recognizer
from trace import Label, load_trace, observations, save_trace

DT = 0.04


def stroke(t0, shift, dur=0.4):
    n = int(dur / DT)
    return [Live(round(t0 + i * DT, 6), shift, 1e-3) for i in range(n)]


def idle(t0, t1):
    out, t = [], t0
    while t < t1:
        out.append(Live(round(t, 6), 0.0, 1e-5))
        t += DT
    return out


def labeled_trace():
    # push at 1.0, pull at 3.0, push at 5.0 (never labeled gesture: nothing fires), extra pull at 7.0
    obs = (idle(0, 1.0) + stroke(1.0, 40.0) + idle(1.4, 3.0) + stroke(3.0, -40.0)
           + idle(3.4, 5.0) + idle(5.0, 7.0) + stroke(7.0, -40.0) + idle(7.4, 8.0))
    labels = [Label(1.0, Gesture.PUSH), Label(3.0, Gesture.PULL), Label(5.0, Gesture.PUSH)]
    return obs, labels


def tmp(name):
    return os.path.join(tempfile.mkdtemp(), name)


def run_cli(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


def test_round_trip_is_exact_including_dropout_status():
    recs = [Live(0.1 + 1e-9, 12.345678901234, 3.3e-5), Dropout(0.2, Status.NO_CARRIER),
            Label(0.15, Gesture.PULL), Dropout(0.3, Status.MIC_SILENT), Live(0.4, -1.0, 0.0)]
    p = tmp("t.jsonl")
    save_trace(p, recs)
    assert load_trace(p) == recs
    assert observations(load_trace(p)) == [r for r in recs if not isinstance(r, Label)]


def test_replay_labeled_trace_counts_confusion_and_latency():
    obs, labels = labeled_trace()
    events = run_recognizer(obs, RecognizerParams())
    pairs, extra = match_labels(labels, events)
    assert [e is not None for _, e in pairs] == [True, True, False], pairs
    assert [e.gesture for _, e in pairs[:2]] == [Gesture.PUSH, Gesture.PULL]
    assert len(extra) == 1 and extra[0].gesture is Gesture.PULL
    lat = pairs[0][1].t - 1.0
    assert 0.09 < lat < 0.2, lat

    p = tmp("t.jsonl")
    save_trace(p, obs + labels)
    code, out = run_cli(p)
    assert code == 0
    assert "push accuracy 1/2 (50%)" in out, out
    assert "pull accuracy 1/1 (100%)" in out, out
    assert "latency s: mean" in out, out
    rows = {l.split()[0]: l.split()[1:] for l in out.splitlines() if l.split()[:1] in (["push"], ["pull"], ["extra"])
            and len(l.split()) > 2 and l.split()[1].isdigit()}
    assert rows["push"] == ["1", "0", "1"], rows
    assert rows["pull"] == ["0", "1", "0"], rows
    assert out.splitlines()[-2].split() == ["extra", "1"], out


def test_wrong_gesture_is_a_confusion_not_a_miss():
    obs = idle(0, 1.0) + stroke(1.0, -40.0)
    events = run_recognizer(obs, RecognizerParams())
    pairs, extra = match_labels([Label(1.0, Gesture.PUSH)], events)
    assert pairs[0][1].gesture is Gesture.PULL and not extra


def test_no_gesture_trace_false_trigger_rate():
    obs = idle(0, 30.0) + stroke(30.0, 40.0) + idle(30.4, 60.0)
    p = tmp("t.jsonl")
    save_trace(p, obs)
    code, out = run_cli(p, "--no-gesture")
    assert code == 0
    assert "false triggers 1 (1.00/min)" in out, out


def test_params_override_changes_result():
    obs, labels = labeled_trace()
    p, pj = tmp("t.jsonl"), tmp("p.json")
    save_trace(p, obs + labels)
    with open(pj, "w") as f:
        json.dump({"shift_on_hz": 100.0}, f)
    assert load_params(pj).shift_on_hz == 100.0
    _, out = run_cli(p, "--params", pj)
    assert "0 events" in out, out


def test_unknown_params_key_errors_clearly():
    pj = tmp("p.json")
    with open(pj, "w") as f:
        json.dump({"shift_on": 1.0}, f)
    try:
        load_params(pj)
    except ValueError as e:
        assert "shift_on" in str(e) and "valid" in str(e), e
    else:
        raise AssertionError("expected ValueError")
    p = tmp("t.jsonl")
    save_trace(p, idle(0, 1.0))
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            main([p, "--params", pj])
        except SystemExit as e:
            assert e.code == 2
        else:
            raise AssertionError("expected SystemExit")


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
