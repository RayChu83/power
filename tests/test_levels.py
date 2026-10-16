import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from levels import Status, measure

RATE = 48000
FREQ = 19000.0  # not bin-centred at N=4096 (bin 1621.33), so scalloping would show up
N = 4096
t = np.arange(N) / RATE


def noise(std, seed=0):
    return np.random.default_rng(seed).normal(0.0, std, N)


def tone(amp):
    return amp * np.sin(2 * np.pi * FREQ * t + 0.3)


def test_zeros_is_mic_silent():
    assert measure(np.zeros(N, dtype=np.float32), RATE, FREQ).status is Status.MIC_SILENT


def test_near_zero_dither_is_mic_silent():
    assert measure(noise(1e-9), RATE, FREQ).status is Status.MIC_SILENT


def test_white_noise_only_is_no_carrier():
    for std in (1e-4, 0.1):
        assert measure(noise(std), RATE, FREQ).status is Status.NO_CARRIER, std


def test_strong_carrier_is_ok_at_expected_level():
    lv = measure(tone(0.5) + noise(1e-3), RATE, FREQ)
    assert lv.status is Status.OK
    # A sine of amplitude 0.5 is 20*log10(0.5) = -6.02 dBFS. Band power is
    # independent of where the tone falls between bins, so only leakage
    # outside the band and noise remain; both are far below 0.5 dB.
    assert abs(lv.carrier_dbfs - (-6.02)) < 0.5, lv.carrier_dbfs
    assert lv.noise_dbfs < lv.carrier_dbfs - 40, lv


def test_weak_real_carrier_is_ok():
    lv = measure(tone(0.01) + noise(1e-4), RATE, FREQ)
    assert lv.status is Status.OK
    assert abs(lv.carrier_dbfs - (-40.0)) < 0.5, lv.carrier_dbfs


def test_carrier_equal_to_noise_is_not_ok():
    # Broadband noise RMS says nothing about detectability of a narrow tone,
    # so "equal" means: tone power equals the noise power inside a 100 Hz
    # band around the carrier, i.e. the tone does not stand out locally.
    std = 0.05
    amp = std * np.sqrt(2 * 100.0 / (RATE / 2))
    for seed in range(5):
        lv = measure(tone(amp) + noise(std, seed), RATE, FREQ)
        assert lv.status is not Status.OK, (seed, lv)


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
