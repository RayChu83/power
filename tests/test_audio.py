import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import audio
from audio import AudioEngine

RATE = 48000
FREQ = 19000.0


class FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def make_engine(devices=(None, None)):
    audio.sd.Stream = FakeStream  # no audio device is ever opened
    return AudioEngine(RATE, FREQ, 0.5, devices)


def feed(engine, samples):
    indata = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
    outdata = np.zeros_like(indata)
    engine._callback(indata, outdata, len(indata), None, None)
    return outdata


def test_constructor_passes_devices_to_stream():
    e = make_engine((3, 7))
    assert e.stream.kwargs["device"] == (3, 7)
    assert e.stream.kwargs["samplerate"] == RATE


def test_counter_advances_by_frames():
    e = make_engine()
    assert e.total_written == 0
    feed(e, np.zeros(480))
    feed(e, np.zeros(256))
    assert e.total_written == 736


def test_read_window_returns_latest_and_end_index():
    e = make_engine()
    feed(e, np.arange(100))
    feed(e, np.arange(100, 150))
    samples, end = e.read_window(30)
    assert end == 150
    assert np.array_equal(samples, np.arange(120, 150, dtype=np.float32))
    assert np.array_equal(e.latest(30), samples)


def test_output_tone_is_phase_continuous():
    e = make_engine()
    out = np.concatenate([feed(e, np.zeros(300))[:, 0], feed(e, np.zeros(212))[:, 0]])
    expected = 0.5 * np.sin(2 * np.pi * FREQ * np.arange(512) / RATE)
    assert np.allclose(out, expected, atol=1e-4)


def test_wait_until_true_when_already_satisfied():
    e = make_engine()
    feed(e, np.zeros(100))
    assert e.wait_until(100, 0.0) is True
    assert e.wait_until(50, 0.0) is True


def test_wait_until_false_on_timeout():
    e = make_engine()
    feed(e, np.zeros(100))
    t0 = time.monotonic()
    assert e.wait_until(101, 0.05) is False
    assert time.monotonic() - t0 >= 0.04


def test_wait_until_wakes_when_callback_delivers():
    e = make_engine()
    threading.Timer(0.05, feed, (e, np.zeros(200))).start()
    t0 = time.monotonic()
    assert e.wait_until(200, 2.0) is True
    assert time.monotonic() - t0 < 1.0


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
