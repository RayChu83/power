from dataclasses import dataclass

import numpy as np

# Blackman: sidelobes below -58 dB, main lobe +-3 bins. Hann's -31 dB sidelobes let
# carrier leakage outweigh a -40 dB echo out to ~100 Hz.
LOBE_BINS = 3.0


@dataclass(frozen=True)
class Frame:
    offsets: np.ndarray  # Hz relative to the observed carrier, fixed axis
    spectrum_db: np.ndarray  # smoothed level per offset, 0 dB = carrier
    shift_hz: float  # net sideband shift, + = toward mic
    carrier_hz: float  # observed carrier this frame


class SpectrumAnalyzer:
    """Turns raw mic windows into a carrier-relative spectrum and a net Doppler shift."""

    def __init__(self, rate, freq, span, nfft, guard):
        self.rate = rate
        self.pad = nfft * 4  # zero-pad for smoother peak interpolation
        self.window = np.blackman(nfft)
        self.phase_step = -2j * np.pi * np.arange(nfft) / rate
        step = rate / self.pad
        k = int(span // step)
        self.bins = np.arange(-k, k + 1)
        self.offsets = self.bins * step
        lobe = LOBE_BINS * rate / nfft
        self.guard = max(guard, lobe)
        self.sideband = np.abs(self.offsets) > self.guard
        freqs = np.fft.rfftfreq(self.pad, 1 / rate)
        self.search = np.flatnonzero(np.abs(freqs - freq) <= lobe)
        self.smoothed = np.zeros(self.offsets.size)

    def reset(self):
        self.smoothed[:] = 0.0

    def process(self, x):
        xw = x * self.window
        carrier = self._locate_carrier(xw)
        # Shift the observed carrier to exactly 0 Hz so the offset axis, guard, and
        # shift are symmetric about it whatever the DAC/ADC clock offset.
        base = np.fft.fft(xw * np.exp(self.phase_step * carrier), n=self.pad)[self.bins]
        power = np.abs(base) ** 2
        ref = power[self.bins.size // 2]
        # Silent windows (the viz ring buffer starts as zeros) carry nothing to normalise.
        if ref > 0:
            self.smoothed = 0.6 * self.smoothed + 0.4 * power / ref

        side = np.where(self.sideband, self.smoothed, 0.0)
        total = side.sum()
        shift = float((self.offsets * side).sum() / total) if total > 0 else 0.0
        return Frame(self.offsets, 10 * np.log10(self.smoothed + 1e-12), shift, carrier)

    def _locate_carrier(self, xw):
        mag = np.abs(np.fft.rfft(xw, n=self.pad))
        i = self.search[np.argmax(mag[self.search])]
        a, b, c = np.log(mag[i - 1 : i + 2] + 1e-12)
        curvature = a - 2 * b + c
        delta = 0.5 * (a - c) / curvature if curvature < 0 else 0.0
        return (i + delta) * self.rate / self.pad
