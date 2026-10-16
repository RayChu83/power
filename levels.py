"""Absolute carrier and noise levels from a raw mic window, plus a health status."""

from dataclasses import dataclass
from enum import Enum

import numpy as np

CARRIER_HALF_BAND_HZ = 30.0  # Hann main lobe is +/-2 bins (~23 Hz at 48k/4096)
NOISE_INNER_HZ = 500.0  # beyond any hand-motion Doppler sideband
NOISE_OUTER_HZ = 2000.0
SILENT_PEAK = 1e-6  # -120 dBFS; any live mic's self-noise is far above this
MIN_SNR_DB = 15.0
FLOOR_DBFS = -200.0


class Status(Enum):
    OK = "ok"
    MIC_SILENT = "MIC SILENT"
    NO_CARRIER = "NO CARRIER"


@dataclass(frozen=True)
class Levels:
    status: Status
    carrier_dbfs: float  # power in the carrier band, 0 dBFS = full-scale sine
    noise_dbfs: float  # off-carrier noise scaled to the carrier band's width

    @property
    def snr_db(self):
        return self.carrier_dbfs - self.noise_dbfs


def _dbfs(mean_square):
    return max(10 * np.log10(mean_square / 0.5 + 1e-30), FLOOR_DBFS)


def measure(x, rate, carrier_hz):
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    w = np.hanning(n)
    # One-sided bin powers scaled so their sum is the window's mean square.
    power = 2 * np.abs(np.fft.rfft(x * w)) ** 2 / (n * np.sum(w**2))
    dist = np.abs(np.fft.rfftfreq(n, 1 / rate) - carrier_hz)

    carrier_bins = dist <= CARRIER_HALF_BAND_HZ
    noise_bins = (dist >= NOISE_INNER_HZ) & (dist <= NOISE_OUTER_HZ)
    carrier_ms = power[carrier_bins].sum()
    # Median ignores stray tones/echo peaks that would inflate a mean.
    noise_ms = np.median(power[noise_bins]) * carrier_bins.sum()

    carrier_dbfs = _dbfs(carrier_ms)
    noise_dbfs = _dbfs(noise_ms)
    if np.max(np.abs(x), initial=0.0) < SILENT_PEAK:
        status = Status.MIC_SILENT
    elif carrier_dbfs - noise_dbfs < MIN_SNR_DB:
        status = Status.NO_CARRIER
    else:
        status = Status.OK
    return Levels(status, carrier_dbfs, noise_dbfs)
