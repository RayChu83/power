"""Gesture scroll: plays a pilot tone, recognizes push/pull hand motion, scrolls the page."""

import argparse
import sys
import threading
from collections import deque

import sounddevice as sd

from actions import DEFAULT_BINDINGS, Action, Actuator, make_actuator
from audio import AudioEngine
from gestures import Gesture, GestureEvent, RecognizerParams, ThresholdRecognizer
from pipeline import Pipeline, Tick

TAPE_LEN = 500  # ~20 s of ticks at the 40 ms hop; bounds memory if nothing drains it


class Controller:
    """Single consumer of the pipeline: tick -> recognizer -> binding -> actuator."""

    def __init__(self, pipe: Pipeline, rec: ThresholdRecognizer, bindings: dict[Gesture, Action], act: Actuator):
        self.pipe = pipe
        self.rec = rec
        self.bindings = bindings
        self.act = act
        self.stalled = False  # run ended because the audio stream stopped delivering
        self._lock = threading.Lock()
        self._tape: deque[tuple[Tick, list[GestureEvent]]] = deque(maxlen=TAPE_LEN)

    def drain(self) -> list[tuple[Tick, list[GestureEvent]]]:
        """Ticks (with the events each produced) recorded since the last drain; for the viz."""
        with self._lock:
            out = list(self._tape)
            self._tape.clear()
            return out

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set():
            try:
                tick = self.pipe.next_tick()
            except TimeoutError as e:
                self.stalled = True
                print(f"audio stalled, stopping: {e}", file=sys.stderr)
                return
            events = self.rec.feed(tick.obs)
            for e in events:
                print(f"[{e.t:7.2f}s] {e.gesture.value} peak {e.strength_hz:.0f} Hz over {e.duration_s * 1000:.0f} ms")
                action = self.bindings.get(e.gesture)
                if action is not None:
                    self.act(action)
            with self._lock:
                self._tape.append((tick, events))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--freq", type=float, default=19000.0, help="pilot tone in Hz (default 19000)")
    p.add_argument("--rate", type=int, default=48000, help="sample rate (default 48000)")
    p.add_argument("--volume", type=float, default=0.5, help="tone amplitude 0-1 (default 0.5)")
    p.add_argument("--span", type=float, default=400.0, help="analyze +/- Hz around carrier (default 400)")
    p.add_argument("--nfft", type=int, default=4096, help="analysis window in samples (default 4096)")
    p.add_argument("--guard", type=float, default=20.0, help="Hz around carrier ignored by shift estimate")
    p.add_argument("--input-device", default=None, help="input device index or name")
    p.add_argument("--output-device", default=None, help="output device index or name")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    p.add_argument("--live", action="store_true", help="post real scroll events (default: print only)")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return 0

    try:
        act = make_actuator(live=args.live)
    except PermissionError as e:
        print(e, file=sys.stderr)
        return 1

    with AudioEngine(args.rate, args.freq, args.volume, (args.input_device, args.output_device)) as engine:
        pipe = Pipeline(engine, args.freq, args.span, args.nfft, args.guard, hop_s=0.04)
        ctl = Controller(pipe, ThresholdRecognizer(RecognizerParams()), DEFAULT_BINDINGS, act)
        mode = "LIVE: posting real scrolls" if args.live else "dry-run: printing only"
        print(f"{mode}. Ctrl-C to stop.")
        stop = threading.Event()
        try:
            ctl.run(stop)
        except KeyboardInterrupt:
            stop.set()
            print("stopped")
    return 1 if ctl.stalled else 0


if __name__ == "__main__":
    sys.exit(main())
