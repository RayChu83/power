"""Live Doppler visualizer: plays a pilot tone and plots the echo spectrum."""

import argparse
import threading

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

from dsp import SpectrumAnalyzer
from levels import Status, measure

SPEED_OF_SOUND = 343.0  # m/s


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--freq", type=float, default=19000.0, help="pilot tone in Hz (default 19000)")
    p.add_argument("--rate", type=int, default=48000, help="sample rate (default 48000)")
    p.add_argument("--volume", type=float, default=0.5, help="tone amplitude 0-1 (default 0.5)")
    p.add_argument("--span", type=float, default=400.0, help="plot +/- Hz around carrier (default 400)")
    p.add_argument("--nfft", type=int, default=4096, help="analysis window in samples (default 4096)")
    p.add_argument("--history", type=int, default=200, help="spectrogram frames to keep (default 200)")
    p.add_argument("--guard", type=float, default=20.0, help="Hz around carrier ignored by shift estimate")
    p.add_argument("--input-device", default=None, help="input device index or name")
    p.add_argument("--output-device", default=None, help="output device index or name")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    return p.parse_args()


class AudioEngine:
    """Full-duplex stream: emits a phase-continuous sine, buffers mic input."""

    def __init__(self, args):
        self.freq = args.freq
        self.rate = args.rate
        self.volume = args.volume
        self.phase = 0.0
        self.buf = np.zeros(args.rate * 2, dtype=np.float32)  # 2 s ring buffer
        self.lock = threading.Lock()
        self.stream = sd.Stream(
            samplerate=args.rate,
            channels=(1, 1),
            dtype="float32",
            device=(args.input_device, args.output_device),
            callback=self._callback,
        )

    def _callback(self, indata, outdata, frames, time, status):
        if status:
            print("audio status:", status)
        step = 2 * np.pi * self.freq / self.rate
        ph = self.phase + step * np.arange(frames)
        outdata[:, 0] = self.volume * np.sin(ph)
        self.phase = (self.phase + step * frames) % (2 * np.pi)
        with self.lock:
            self.buf = np.roll(self.buf, -frames)
            self.buf[-frames:] = indata[:, 0]

    def latest(self, n):
        with self.lock:
            return self.buf[-n:].copy()

    def __enter__(self):
        self.stream.start()
        return self

    def __exit__(self, *exc):
        self.stream.stop()
        self.stream.close()


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    analyzer = SpectrumAnalyzer(args.rate, args.freq, args.span, args.nfft, args.guard)
    offsets = analyzer.offsets
    history = np.full((args.history, offsets.size), -120.0)
    shift_trace = np.zeros(args.history)

    fig, (ax_spec, ax_wf, ax_shift) = plt.subplots(
        3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 3, 1.5]}
    )
    fig.suptitle(f"Live Doppler visualizer - carrier {args.freq:.0f} Hz")

    (line,) = ax_spec.plot(offsets, history[-1], lw=1.2)
    ax_spec.axvline(0, color="r", ls="--", lw=0.8, alpha=0.6)
    ax_spec.set_xlim(offsets[0], offsets[-1])
    ax_spec.set_ylim(-110, 0)
    ax_spec.set_ylabel("Level (dB)")
    ax_spec.set_xlabel("Offset from carrier (Hz)")
    ax_spec.grid(alpha=0.3)

    img = ax_wf.imshow(
        history,
        aspect="auto",
        origin="lower",
        extent=[offsets[0], offsets[-1], -args.history, 0],
        vmin=-100,
        vmax=-20,
        cmap="magma",
    )
    ax_wf.set_ylabel("Frames ago")
    ax_wf.set_xlabel("Offset from carrier (Hz)")

    (shift_line,) = ax_shift.plot(np.arange(-args.history, 0), shift_trace, lw=1.2)
    ax_shift.axhline(0, color="gray", lw=0.8)
    ax_shift.set_xlim(-args.history, 0)
    ax_shift.set_ylim(-args.span / 2, args.span / 2)
    ax_shift.set_ylabel("Net shift (Hz)")
    ax_shift.set_xlabel("Frames ago   (+ = toward mic, - = away)")
    ax_shift.grid(alpha=0.3)
    readout = ax_shift.text(0.01, 0.9, "", transform=ax_shift.transAxes, va="top", family="monospace")
    alarm = ax_spec.text(
        0.5, 0.5, "", transform=ax_spec.transAxes, ha="center", va="center",
        fontsize=28, weight="bold", color="red",
    )

    fig.tight_layout()

    def update(_):
        nonlocal history, shift_trace
        window = engine.latest(args.nfft)
        levels = measure(window, args.rate, args.freq)
        ok = levels.status is Status.OK
        frame = analyzer.process(window)
        if not ok:
            # Frames without a carrier normalise noise to 0 dB and would swamp a real echo once the tone returns.
            analyzer.reset()
        # Without a carrier the normalised spectrum is just rescaled noise, so its shift is meaningless.
        shift = frame.shift_hz if ok else 0.0

        history = np.roll(history, -1, axis=0)
        history[-1] = frame.spectrum_db
        shift_trace = np.roll(shift_trace, -1)
        shift_trace[-1] = shift

        line.set_ydata(frame.spectrum_db)
        img.set_data(history)
        shift_line.set_ydata(shift_trace)
        level = f"carrier {levels.carrier_dbfs:6.1f} dBFS  snr {levels.snr_db:5.1f} dB"
        if ok:
            velocity = shift * SPEED_OF_SOUND / (2 * args.freq)
            readout.set_text(f"shift {shift:+7.1f} Hz  ~ {velocity:+.2f} m/s   {level}")
        else:
            readout.set_text(f"shift    --- Hz  ~   --- m/s   {level}   {levels.status.value}")
        alarm.set_text("" if ok else levels.status.value)
        return line, img, shift_line, readout, alarm

    with AudioEngine(args) as engine:
        anim = FuncAnimation(fig, update, interval=40, blit=False, cache_frame_data=False)
        print("Running. Wave a hand near the mic/trackpad. Close the window to quit.")
        plt.show()


if __name__ == "__main__":
    main()
