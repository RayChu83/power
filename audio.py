"""Full-duplex audio: plays a pilot tone and buffers the microphone."""

import threading

import numpy as np
import sounddevice as sd


class AudioEngine:
    """Full-duplex stream: emits a phase-continuous sine, buffers mic input."""

    def __init__(self, rate, freq, volume, devices):
        """`devices` is an (input, output) pair of sounddevice device specs."""
        self.freq = freq
        self.rate = rate
        self.volume = volume
        self.phase = 0.0
        self.buf = np.zeros(rate * 2, dtype=np.float32)  # 2 s ring buffer
        self.total_written = 0  # samples received since start; never wraps
        self.lock = threading.Lock()
        self._written = threading.Condition(self.lock)
        self.stream = sd.Stream(
            samplerate=rate,
            channels=(1, 1),
            dtype="float32",
            device=devices,
            callback=self._callback,
        )

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print("audio status:", status)
        step = 2 * np.pi * self.freq / self.rate
        ph = self.phase + step * np.arange(frames)
        outdata[:, 0] = self.volume * np.sin(ph)
        self.phase = (self.phase + step * frames) % (2 * np.pi)
        with self._written:
            self.buf = np.roll(self.buf, -frames)
            self.buf[-frames:] = indata[:, 0]
            self.total_written += frames
            self._written.notify_all()

    def latest(self, n):
        with self.lock:
            return self.buf[-n:].copy()

    def read_window(self, n):
        """Latest n samples and the sample index of the window's end."""
        with self.lock:
            return self.buf[-n:].copy(), self.total_written

    def wait_until(self, sample_index, timeout):
        """Block until total_written >= sample_index. False on timeout."""
        with self._written:
            return self._written.wait_for(lambda: self.total_written >= sample_index, timeout)

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()
