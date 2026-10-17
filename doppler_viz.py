"""Live Doppler visualizer: plays a pilot tone and plots the echo spectrum."""

import argparse
import threading
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.animation import FuncAnimation

from audio import AudioEngine
from cli import add_audio_args
from gestures import Gesture, Live, RecognizerParams, ThresholdRecognizer
from levels import Status
from pipeline import Pipeline

SPEED_OF_SOUND = 343.0  # m/s
DEFAULT_HISTORY = 200  # frames kept in the waterfall and shift trace
STOP_JOIN_S = 3.0  # worker is at most one hop from noticing stop; the margin covers a stalled stream's timeout


class Viz:
    """Plots Ticks. Never touches the audio engine or the analyzer."""

    def __init__(self, offsets, freq, span, history=DEFAULT_HISTORY):
        self.freq = freq
        self.history_len = history
        self.history = np.full((history, offsets.size), -120.0)
        self.shift_trace = np.zeros(history)
        self.count = 0  # ticks drawn so far; positions event markers in time
        self.marks = deque()  # (tick number, shift at that tick, Gesture) for events still on screen

        self.fig, (ax_spec, ax_wf, ax_shift) = plt.subplots(
            3, 1, figsize=(10, 9), gridspec_kw={"height_ratios": [2, 3, 1.5]}
        )
        self.fig.suptitle(f"Live Doppler visualizer - carrier {freq:.0f} Hz")

        (self.line,) = ax_spec.plot(offsets, self.history[-1], lw=1.2)
        ax_spec.axvline(0, color="r", ls="--", lw=0.8, alpha=0.6)
        ax_spec.set_xlim(offsets[0], offsets[-1])
        ax_spec.set_ylim(-110, 0)
        ax_spec.set_ylabel("Level (dB)")
        ax_spec.set_xlabel("Offset from carrier (Hz)")
        ax_spec.grid(alpha=0.3)

        self.img = ax_wf.imshow(
            self.history,
            aspect="auto",
            origin="lower",
            extent=[offsets[0], offsets[-1], -history, 0],
            vmin=-100,
            vmax=-20,
            cmap="magma",
        )
        ax_wf.set_ylabel("Frames ago")
        ax_wf.set_xlabel("Offset from carrier (Hz)")

        (self.shift_line,) = ax_shift.plot(np.arange(-history, 0), self.shift_trace, lw=1.2)
        ax_shift.axhline(0, color="gray", lw=0.8)
        ax_shift.set_xlim(-history, 0)
        ax_shift.set_ylim(-span / 2, span / 2)
        ax_shift.set_ylabel("Net shift (Hz)")
        ax_shift.set_xlabel("Frames ago   (+ = toward mic, - = away)")
        ax_shift.grid(alpha=0.3)
        self.readout = ax_shift.text(0.01, 0.9, "", transform=ax_shift.transAxes, va="top", family="monospace")
        self.alarm = ax_spec.text(
            0.5, 0.5, "", transform=ax_spec.transAxes, ha="center", va="center",
            fontsize=28, weight="bold", color="red",
        )
        self.markers = {
            Gesture.PUSH: ax_shift.plot([], [], "^", color="tab:green", ms=10, mec="k", label="push")[0],
            Gesture.PULL: ax_shift.plot([], [], "v", color="tab:orange", ms=10, mec="k", label="pull")[0],
        }
        ax_shift.legend(loc="lower right", fontsize=8, framealpha=0.6)

        self.fig.tight_layout()

    def update(self, ticks):
        """Draw `ticks`, a list of (Tick, events) as returned by Controller.drain()."""
        for tick, events in ticks:
            self._advance(tick, events)
        if ticks:
            self._draw(ticks[-1][0])
        return self.line, self.img, self.shift_line, self.readout, self.alarm, *self.markers.values()

    def _advance(self, tick, events):
        # Without a carrier the shift is not representable; the trace shows 0, as before.
        shift = tick.obs.shift_hz if isinstance(tick.obs, Live) else 0.0
        self.history = np.roll(self.history, -1, axis=0)
        self.history[-1] = tick.frame.spectrum_db
        self.shift_trace = np.roll(self.shift_trace, -1)
        self.shift_trace[-1] = shift
        self.count += 1
        for e in events:
            self.marks.append((self.count, shift, e.gesture))
        while self.marks and self.count - self.marks[0][0] >= self.history_len:
            self.marks.popleft()

    def _draw(self, tick):
        levels = tick.levels
        ok = levels.status is Status.OK
        self.line.set_ydata(tick.frame.spectrum_db)
        self.img.set_data(self.history)
        self.shift_line.set_ydata(self.shift_trace)
        for gesture, artist in self.markers.items():
            hits = [(n - self.count - 1, y) for n, y, g in self.marks if g is gesture]
            artist.set_data([x for x, _ in hits], [y for _, y in hits])

        shift = self.shift_trace[-1]
        level = f"carrier {levels.carrier_dbfs:6.1f} dBFS  snr {levels.snr_db:5.1f} dB"
        if ok:
            velocity = shift * SPEED_OF_SOUND / (2 * self.freq)
            self.readout.set_text(f"shift {shift:+7.1f} Hz  ~ {velocity:+.2f} m/s   {level}")
        else:
            self.readout.set_text(f"shift    --- Hz  ~   --- m/s   {level}   {levels.status.value}")
        self.alarm.set_text("" if ok else levels.status.value)


def run_viz(ctl, span, history=DEFAULT_HISTORY):
    """Run `ctl` on a worker thread and plot its ticks on this one until the window closes."""
    viz = Viz(ctl.pipe.analyzer.offsets, ctl.pipe.freq, span, history)
    stop = threading.Event()
    worker = threading.Thread(target=ctl.run, args=(stop,), daemon=True)

    def frame(_):
        artists = viz.update(ctl.drain())
        if ctl.stalled:
            viz.alarm.set_text("AUDIO STALLED")
        return artists

    anim = FuncAnimation(viz.fig, frame, interval=40, blit=False, cache_frame_data=False)  # noqa: F841 (must stay referenced)
    worker.start()
    try:
        plt.show()
    finally:
        stop.set()
        worker.join(STOP_JOIN_S)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_audio_args(p)
    p.add_argument("--history", type=int, default=DEFAULT_HISTORY, help="spectrogram frames to keep (default 200)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return

    from gesture_app import Controller  # lazy: when gesture_app is __main__ it would otherwise load twice

    with AudioEngine(args.rate, args.freq, args.volume, (args.input_device, args.output_device)) as engine:
        pipe = Pipeline(engine, args.freq, args.span, args.nfft, args.guard)
        # No bindings and a no-op actuator: this command never posts input events.
        ctl = Controller(pipe, ThresholdRecognizer(RecognizerParams()), {}, lambda action: None)
        print("Running. Wave a hand near the mic/trackpad. Close the window to quit.")
        run_viz(ctl, args.span, args.history)


if __name__ == "__main__":
    main()
