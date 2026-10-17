"""Gesture scroll: plays a pilot tone, recognizes push/pull hand motion, scrolls the page."""

import argparse
import sys
import threading
from collections import deque

import sounddevice as sd

from actions import DEFAULT_BINDINGS, Action, Actuator, make_actuator
from audio import AudioEngine
from cli import add_audio_args
from gestures import Gesture, GestureEvent, RecognizerParams, ThresholdRecognizer
from pipeline import Pipeline, Tick
from trace import Label, Record, save_trace

TAPE_LEN = 500  # ~20 s of ticks at the 40 ms hop; bounds memory if nothing drains it


class Controller:
    """Single consumer of the pipeline: tick -> recognizer -> binding -> actuator."""

    def __init__(self, pipe: Pipeline, rec: ThresholdRecognizer, bindings: dict[Gesture, Action], act: Actuator,
                 record: list[Record] | None = None):
        self.pipe = pipe
        self.rec = rec
        self.bindings = bindings
        self.act = act
        self.record = record  # every observation (and label), for --record
        self.last_t = 0.0  # sample-time of the newest tick; labels are stamped with it
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
            self.last_t = tick.t
            if self.record is not None:
                self.record.append(tick.obs)
            events = self.rec.feed(tick.obs)
            for e in events:
                print(f"[{e.t:7.2f}s] {e.gesture.value} peak {e.strength_hz:.0f} Hz over {e.duration_s * 1000:.0f} ms")
                action = self.bindings.get(e.gesture)
                if action is not None:
                    self.act(action)
            with self._lock:
                self._tape.append((tick, events))


LABEL_KEYS = {"p": Gesture.PUSH, "l": Gesture.PULL}


def label_from_stdin(ctl: Controller) -> None:
    """Type p (push) or l (pull) plus Enter as you start a gesture; stamps it with the newest tick time."""
    for line in sys.stdin:
        gesture = LABEL_KEYS.get(line.strip().lower())
        if gesture is not None and ctl.record is not None:
            ctl.record.append(Label(ctl.last_t, gesture))
            print(f"labeled {gesture.value} at {ctl.last_t:.2f}s")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_audio_args(p)
    p.add_argument("--live", action="store_true", help="post real scroll events (default: print only)")
    p.add_argument("--record", metavar="PATH", help="save observations to a trace file; type p or l + Enter to label")
    p.add_argument("--viz", action="store_true", help="show live plots with fired gestures marked")
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
        records: list[Record] | None = [] if args.record else None
        ctl = Controller(pipe, ThresholdRecognizer(RecognizerParams()), DEFAULT_BINDINGS, act, record=records)
        if records is not None:
            threading.Thread(target=label_from_stdin, args=(ctl,), daemon=True).start()
        mode = "LIVE: posting real scrolls" if args.live else "dry-run: printing only"
        print(f"{mode}. Ctrl-C to stop.")
        if args.viz:
            from doppler_viz import run_viz  # lazy: pulls in matplotlib

            run_viz(ctl, args.span)
        else:
            stop = threading.Event()
            try:
                ctl.run(stop)
            except KeyboardInterrupt:
                stop.set()
                print("stopped")
        if records is not None:
            save_trace(args.record, records)
            print(f"saved {len(records)} records to {args.record}")
    return 1 if ctl.stalled else 0


if __name__ == "__main__":
    sys.exit(main())
