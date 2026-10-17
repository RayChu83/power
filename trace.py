"""Trace files: JSON lines, one record per line.

    {"kind": "live", "t": 1.2, "shift_hz": 35.0, "motion": 0.002}
    {"kind": "dropout", "t": 1.3, "status": "NO_CARRIER"}   # Status member name
    {"kind": "label", "t": 1.2, "gesture": "push"}          # onset time of a labeled gesture

Recording (for the --record flag): collect every Observation the pipeline yields, plus a
Label when the user marks a gesture, then call save_trace(path, records) once at exit.
Records may be in any order; save_trace writes them as given. Floats round-trip exactly.
"""

import json
from dataclasses import dataclass

from gestures import Dropout, Gesture, Live, Observation
from levels import Status


@dataclass(frozen=True)
class Label:
    t: float  # onset time of the gesture
    gesture: Gesture


Record = Observation | Label


def _to_dict(r: Record) -> dict:
    match r:
        case Live():
            return {"kind": "live", "t": r.t, "shift_hz": r.shift_hz, "motion": r.motion}
        case Dropout():
            return {"kind": "dropout", "t": r.t, "status": r.status.name}
        case Label():
            return {"kind": "label", "t": r.t, "gesture": r.gesture.value}
    raise TypeError(f"not a trace record: {r!r}")


def _from_dict(d: dict) -> Record:
    match d["kind"]:
        case "live":
            return Live(d["t"], d["shift_hz"], d["motion"])
        case "dropout":
            return Dropout(d["t"], Status[d["status"]])
        case "label":
            return Label(d["t"], Gesture(d["gesture"]))
    raise ValueError(f"unknown record kind {d['kind']!r}")


def save_trace(path: str, records: list[Record]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(_to_dict(r)) + "\n")


def load_trace(path: str) -> list[Record]:
    out = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                out.append(_from_dict(json.loads(line)))
            except (ValueError, KeyError) as e:
                raise ValueError(f"{path}:{n}: bad trace record: {e!r}") from e
    return out


def observations(records: list[Record]) -> list[Observation]:
    return [r for r in records if not isinstance(r, Label)]
