"""Turns an audio engine's sample stream into a fixed-rate series of Ticks."""

from dataclasses import dataclass

from dsp import Frame, SpectrumAnalyzer
from gestures import Dropout, Live, Observation
from levels import Levels, Status, measure

STALL_TIMEOUT_S = 2.0  # a healthy stream delivers a hop every ~40 ms


@dataclass(frozen=True)
class Tick:
    t: float  # window end, seconds = end_sample / rate (never wall-clock)
    levels: Levels
    frame: Frame  # for the viz; spectrum and shift are meaningless unless obs is Live
    obs: Observation  # for the recognizer


class Pipeline:
    """Single consumer of an engine: any object with `rate`, `read_window`, `wait_until`."""

    def __init__(self, engine, freq, span, nfft, guard, hop_s=0.04):
        self.engine = engine
        self.freq = freq
        self.rate = engine.rate
        self.nfft = nfft  # analysis window length in samples
        self.hop = round(hop_s * self.rate)
        self.analyzer = SpectrumAnalyzer(self.rate, freq, span, nfft, guard)
        self._target = None  # sample index the next window must end at or after

    def next_tick(self) -> Tick:
        """Block until `hop` new samples exist, then run measure -> analyze.

        Raises TimeoutError if the stream stops delivering samples.
        """
        if self._target is None:
            _, end = self.engine.read_window(self.nfft)
            self._target = end + self.hop
        if not self.engine.wait_until(self._target, STALL_TIMEOUT_S):
            raise TimeoutError(f"no audio for {STALL_TIMEOUT_S} s; is the stream running?")
        window, end = self.engine.read_window(self.nfft)
        # On-time ticks stay on the hop grid. If the consumer fell a whole hop behind,
        # resync to the stream instead of replaying windows that end at the same sample.
        self._target = self._target + self.hop if end < self._target + self.hop else end + self.hop

        t = end / self.rate
        levels = measure(window, self.rate, self.freq)
        frame = self.analyzer.process(window)
        if levels.status is Status.OK:
            obs = Live(t, frame.shift_hz, frame.motion)
        else:
            # Frames without a carrier normalise noise to 0 dB and would swamp a real echo once the tone returns.
            self.analyzer.reset()
            # Without a carrier the normalised spectrum is just rescaled noise, so its shift is meaningless.
            obs = Dropout(t, levels.status)
        return Tick(t, levels, frame, obs)
