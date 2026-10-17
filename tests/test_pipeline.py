import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gestures import Dropout, Live
from levels import Status
from pipeline import Pipeline

RATE = 48000
FREQ = 19000.0
SPAN = 400.0
NFFT = 4096
GUARD = 20.0
HOP = 1920  # 0.04 s at 48 kHz
TICKS = 12
TOL_HZ = 15.0


class FakeEngine:
    """Serves a deterministic signal indexed by absolute sample; arrival is instant."""

    def __init__(self, source, start=NFFT, stall=False):
        self.rate = RATE
        self.source = source  # source(indices) -> float samples
        self.pos = start
        self.stall = stall
        self.waits = []

    def read_window(self, n):
        idx = np.arange(self.pos - n, self.pos)
        return self.source(idx).astype(np.float32), self.pos

    def wait_until(self, sample_index, timeout):
        self.waits.append(sample_index)
        if self.stall:
            return False
        self.pos = max(self.pos, sample_index)
        return True


def tone(echo_hz=None, echo_db=None, carrier=True, seed=0):
    rng = np.random.default_rng(seed)

    def source(idx):
        t = idx / RATE
        x = rng.normal(0.0, 1e-4, idx.size)
        if carrier:
            x += 0.5 * np.sin(2 * np.pi * FREQ * t + 0.3)
        if echo_hz is not None:
            x += 0.5 * 10 ** (echo_db / 20) * np.sin(2 * np.pi * (FREQ + echo_hz) * t + 1.1)
        return x

    return source


def silence(idx):
    return np.zeros(idx.size)


def pipe_for(engine):
    return Pipeline(engine, FREQ, SPAN, NFFT, GUARD, hop_s=0.04)


def ticks(pipe, n=TICKS):
    return [pipe.next_tick() for _ in range(n)]


def test_one_tick_per_hop_and_times_advance_by_exactly_hop():
    engine = FakeEngine(tone())
    out = ticks(pipe_for(engine))
    assert engine.waits == [NFFT + HOP * (i + 1) for i in range(TICKS)], engine.waits
    assert [round(t.t * RATE) for t in out] == [NFFT + HOP * (i + 1) for i in range(TICKS)]
    assert np.allclose(np.diff([t.t for t in out]), HOP / RATE, rtol=0, atol=1e-12)
    assert all(t.obs.t == t.t for t in out)


def test_tick_time_is_sample_count_not_wall_clock():
    out = ticks(pipe_for(FakeEngine(tone(), start=10 * RATE)), 2)
    assert out[0].t == (10 * RATE + HOP) / RATE


def test_silence_gives_mic_silent_dropout():
    out = ticks(pipe_for(FakeEngine(silence)))
    for tick in out:
        assert tick.levels.status is Status.MIC_SILENT
        assert tick.obs == Dropout(tick.t, Status.MIC_SILENT), tick.obs


def test_noise_without_carrier_gives_no_carrier_dropout():
    rng = np.random.default_rng(1)
    out = ticks(pipe_for(FakeEngine(lambda idx: rng.normal(0.0, 1e-2, idx.size))), 3)
    assert all(t.obs == Dropout(t.t, Status.NO_CARRIER) for t in out)


def test_carrier_only_gives_live_with_small_shift_and_motion():
    last = ticks(pipe_for(FakeEngine(tone())))[-1]
    assert isinstance(last.obs, Live), last.obs
    assert abs(last.obs.shift_hz) < TOL_HZ, last.obs
    assert last.obs.motion < 1e-4, last.obs


def test_carrier_plus_echo_gives_live_with_shift_and_larger_motion():
    idle = ticks(pipe_for(FakeEngine(tone())))[-1].obs
    moving = ticks(pipe_for(FakeEngine(tone(echo_hz=110.0, echo_db=-40.0))))[-1].obs
    assert isinstance(moving, Live), moving
    assert abs(moving.shift_hz - 110.0) < TOL_HZ, moving
    assert moving.motion > 20 * idle.motion, (moving, idle)


def test_dropout_resets_analyzer_so_no_echo_survives_the_gap():
    engine = FakeEngine(tone(echo_hz=110.0, echo_db=-40.0))
    pipe = pipe_for(engine)
    echoing = ticks(pipe, 8)[-1]
    assert isinstance(echoing.obs, Live) and abs(echoing.obs.shift_hz - 110.0) < TOL_HZ, echoing.obs

    # Swap the whole source between ticks so each window is homogeneous.
    engine.source = silence
    gap = pipe.next_tick()
    assert isinstance(gap.obs, Dropout), gap.obs

    engine.source = tone()
    back = pipe.next_tick()
    assert isinstance(back.obs, Live), back.obs
    assert abs(back.obs.shift_hz) < TOL_HZ, back.obs
    assert back.obs.motion < 1e-4, back.obs


def test_without_the_gap_the_echo_would_still_be_visible():
    engine = FakeEngine(tone(echo_hz=110.0, echo_db=-40.0))
    pipe = pipe_for(engine)
    ticks(pipe, 8)
    engine.source = tone()
    carried = pipe.next_tick()
    assert carried.obs.motion > 1e-4, carried.obs


def test_analyzer_is_zeroed_on_the_tick_that_drops_out():
    pipe = pipe_for(FakeEngine(silence))
    pipe.next_tick()
    assert not pipe.analyzer.smoothed.any()


def test_stalled_stream_raises_instead_of_blocking_forever():
    pipe = pipe_for(FakeEngine(tone(), stall=True))
    try:
        pipe.next_tick()
    except TimeoutError:
        return
    raise AssertionError("expected TimeoutError")


def test_consumer_that_falls_behind_resyncs_without_repeating_a_window():
    engine = FakeEngine(tone())
    pipe = pipe_for(engine)
    first = pipe.next_tick()
    engine.pos += 5 * HOP + 7  # stream ran ahead while the consumer was busy
    second = pipe.next_tick()
    third = pipe.next_tick()
    assert second.t > first.t + 5 * HOP / RATE - 1e-9
    assert abs(third.t - second.t - HOP / RATE) < 1e-12, (second.t, third.t)


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
