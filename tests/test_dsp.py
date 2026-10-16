import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dsp import SpectrumAnalyzer

RATE = 48000
FREQ = 19000.0
SPAN = 400.0
NFFT = 4096
GUARD = 20.0
FRAMES = 12
TOL_HZ = 15.0  # "near the true value": within ~14% of a 110 Hz (1 m/s) shift


def run(carrier_offset=0.0, echo_hz=None, echo_db=None, seed=0):
    """Feed FRAMES consecutive windows of a continuous signal; return every Frame."""
    rng = np.random.default_rng(seed)
    analyzer = SpectrumAnalyzer(RATE, FREQ, SPAN, NFFT, GUARD)
    carrier = FREQ + carrier_offset
    frames = []
    for i in range(FRAMES):
        t = np.arange(i * NFFT, (i + 1) * NFFT) / RATE
        x = 0.5 * np.sin(2 * np.pi * carrier * t + 0.3)
        if echo_hz is not None:
            # Echo shift is relative to the emitted tone, i.e. the real carrier.
            x += 0.5 * 10 ** (echo_db / 20) * np.sin(2 * np.pi * (carrier + echo_hz) * t + 1.1)
        x += rng.normal(0.0, 1e-4, NFFT)
        frames.append(analyzer.process(x.astype(np.float32)))
    return frames


def settled_shift(**kw):
    return run(**kw)[-1].shift_hz


def test_no_echo_reads_near_zero():
    for seed in range(3):
        s = settled_shift(seed=seed)
        assert abs(s) < 3.0, (seed, s)


def test_echo_plus_110_at_minus_45_db_reads_near_plus_110():
    s = settled_shift(echo_hz=110.0, echo_db=-45.0)
    assert abs(s - 110.0) < TOL_HZ, s


def test_echo_minus_110_at_minus_45_db_reads_near_minus_110():
    s = settled_shift(echo_hz=-110.0, echo_db=-45.0)
    assert abs(s + 110.0) < TOL_HZ, s


def test_sign_is_right_down_to_minus_55_db():
    bad = []
    for hz in (60.0, 110.0, 200.0):
        for sign in (1, -1):
            for db in (-30.0, -35.0, -40.0, -45.0, -50.0, -55.0):
                s = settled_shift(echo_hz=sign * hz, echo_db=db)
                if not (np.sign(s) == sign and abs(s) > 3.0):
                    bad.append((sign * hz, db, round(s, 1)))
    assert not bad, bad


def test_small_40_hz_shift_at_minus_40_db_has_right_sign():
    up = settled_shift(echo_hz=40.0, echo_db=-40.0)
    down = settled_shift(echo_hz=-40.0, echo_db=-40.0)
    assert up > 3.0, up
    assert down < -3.0, down


def test_carrier_off_nominal_without_echo_reads_near_zero():
    for off in (3.0, -3.0):
        s = settled_shift(carrier_offset=off)
        assert abs(s) < 3.0, (off, s)


def test_carrier_off_nominal_with_echo_reads_true_shift():
    for off in (3.0, -3.0):
        s = settled_shift(carrier_offset=off, echo_hz=110.0, echo_db=-45.0)
        assert abs(s - 110.0) < TOL_HZ, (off, s)


def test_observed_carrier_is_reported():
    for off in (0.0, 3.0, -3.0):
        f = run(carrier_offset=off, echo_hz=110.0, echo_db=-45.0)[-1]
        assert abs(f.carrier_hz - (FREQ + off)) < 0.2, (off, f.carrier_hz)


def test_silent_startup_frames_do_not_poison_later_output():
    # doppler_viz's ring buffer starts as zeros, so the first windows are pure silence.
    analyzer = SpectrumAnalyzer(RATE, FREQ, SPAN, NFFT, GUARD)
    frames = [analyzer.process(np.zeros(NFFT, dtype=np.float32)) for _ in range(3)]
    for i in range(FRAMES):
        t = np.arange(i * NFFT, (i + 1) * NFFT) / RATE
        x = 0.5 * np.sin(2 * np.pi * FREQ * t) + 0.5 * 10 ** (-45 / 20) * np.sin(2 * np.pi * (FREQ + 110) * t)
        frames.append(analyzer.process(x.astype(np.float32)))
    for f in frames:
        assert np.isfinite(f.shift_hz) and np.isfinite(f.spectrum_db).all() and np.isfinite(f.carrier_hz), f.shift_hz
    assert abs(frames[-1].shift_hz - 110.0) < TOL_HZ, frames[-1].shift_hz


def test_output_axis_has_fixed_length():
    frames = run(carrier_offset=3.0, echo_hz=110.0, echo_db=-45.0) + run(carrier_offset=-3.0)
    sizes = {(f.offsets.size, f.spectrum_db.size) for f in frames}
    assert len(sizes) == 1, sizes
    (n, m), = sizes
    assert n == m, sizes
    assert frames[0].offsets[0] <= -SPAN + 5 and frames[0].offsets[-1] >= SPAN - 5, frames[0].offsets[[0, -1]]


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
