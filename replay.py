"""Replay a trace through a fresh ThresholdRecognizer. No audio device is opened.

    python replay.py trace.jsonl [--params params.json] [--no-gesture]

params.json is an object of RecognizerParams field names to values; unknown keys are an error.
--no-gesture marks the trace as containing no gestures, so every event is a false trigger.
"""

import argparse
import dataclasses
import json
import sys

from gestures import Gesture, GestureEvent, RecognizerParams, ThresholdRecognizer
from trace import Label, load_trace, observations

MATCH_WINDOW_S = 1.0


def load_params(path: str | None) -> RecognizerParams:
    if path is None:
        return RecognizerParams()
    with open(path) as f:
        overrides = json.load(f)
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: expected a JSON object of parameter overrides")
    known = {f.name for f in dataclasses.fields(RecognizerParams)}
    unknown = sorted(set(overrides) - known)
    if unknown:
        raise ValueError(f"{path}: unknown params {unknown}; valid: {sorted(known)}")
    return RecognizerParams(**overrides)


def run_recognizer(obs, params: RecognizerParams) -> list[GestureEvent]:
    rec = ThresholdRecognizer(params)
    return [e for o in obs for e in rec.feed(o)]


def match_labels(labels: list[Label], events: list[GestureEvent]):
    """Pair each label with an event.

    Rule: in label order, a label takes the earliest unclaimed event of any gesture with
    onset <= event.t <= onset + MATCH_WINDOW_S. Unclaimed labels are missed; unclaimed
    events are extra. A wrong-gesture match counts as a confusion, not a miss.
    """
    claimed: set[int] = set()
    pairs: list[tuple[Label, GestureEvent | None]] = []
    for lab in sorted(labels, key=lambda l: l.t):
        hit = next(
            (i for i, e in enumerate(events)
             if i not in claimed and lab.t <= e.t <= lab.t + MATCH_WINDOW_S),
            None,
        )
        if hit is not None:
            claimed.add(hit)
        pairs.append((lab, events[hit] if hit is not None else None))
    extra = [e for i, e in enumerate(events) if i not in claimed]
    return pairs, extra


def report_no_gesture(obs, events) -> list[str]:
    minutes = (obs[-1].t - obs[0].t) / 60 if len(obs) > 1 else 0.0
    rate = len(events) / minutes if minutes > 0 else float("nan")
    return [f"duration {minutes * 60:.1f} s", f"false triggers {len(events)} ({rate:.2f}/min)"]


def report_labeled(labels, events) -> list[str]:
    pairs, extra = match_labels(labels, events)
    cols = [Gesture.PUSH.value, Gesture.PULL.value, "missed"]
    conf = {g: dict.fromkeys(cols, 0) for g in (Gesture.PUSH, Gesture.PULL)}
    latencies = []
    for lab, ev in pairs:
        if ev is None:
            conf[lab.gesture]["missed"] += 1
        else:
            conf[lab.gesture][ev.gesture.value] += 1
            latencies.append(ev.t - lab.t)
    lines = []
    for g, row in conf.items():
        total = sum(row.values())
        acc = f"{row[g.value] / total:.0%}" if total else "n/a"
        lines.append(f"{g.value} accuracy {row[g.value]}/{total} ({acc})")
    lines.append("confusion (rows: labeled, cols: detected)")
    lines.append(f"{'':>6}" + "".join(f"{c:>8}" for c in cols) + f"{'extra':>8}")
    for g, row in conf.items():
        lines.append(f"{g.value:>6}" + "".join(f"{row[c]:>8}" for c in cols) + f"{'':>8}")
    lines.append(f"{'extra':>6}{'':>24}{len(extra):>8}")
    if latencies:
        lines.append(
            f"latency s: mean {sum(latencies) / len(latencies):.3f} "
            f"min {min(latencies):.3f} max {max(latencies):.3f}"
        )
    else:
        lines.append("latency s: n/a")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path")
    ap.add_argument("--params")
    ap.add_argument("--no-gesture", action="store_true")
    args = ap.parse_args(argv)
    try:
        params = load_params(args.params)
    except (ValueError, OSError) as e:
        ap.error(str(e))
    records = load_trace(args.path)
    obs = observations(records)
    labels = [r for r in records if isinstance(r, Label)]
    events = run_recognizer(obs, params)
    print(f"{len(obs)} observations, {len(labels)} labels, {len(events)} events")
    for e in events:
        print(f"  {e.t:8.3f}s {e.gesture.value:<4} strength {e.strength_hz:.1f} Hz dur {e.duration_s:.3f} s")
    if args.no_gesture:
        if labels:
            ap.error("--no-gesture trace must not contain labels")
        lines = report_no_gesture(obs, events)
    else:
        lines = report_labeled(labels, events)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
