from dataclasses import dataclass, replace
from enum import Enum

from levels import Status


class Gesture(Enum):
    PUSH = "push"  # hand toward mic/screen, shift > 0
    PULL = "pull"  # away, shift < 0


@dataclass(frozen=True)
class Live:
    t: float
    shift_hz: float
    motion: float


@dataclass(frozen=True)
class Dropout:
    """No valid carrier: a shift is not representable."""

    t: float
    status: Status


Observation = Live | Dropout


@dataclass(frozen=True)
class GestureEvent:
    gesture: Gesture
    t: float  # emit time
    strength_hz: float  # peak |shift| so far
    duration_s: float  # emit time minus burst start


@dataclass(frozen=True)
class RecognizerParams:
    """Milestone 3 onboarding will fit these per user.

    The defaults are untuned guesses. S9 sets them from real recordings.
    """

    motion_on: float = 1e-4  # idle is ~1e-5, real motion 1e-4 to 1e-2
    motion_off: float = 3e-5  # < motion_on: hysteresis
    shift_on_hz: float = 20.0
    min_duration_s: float = 0.10
    gap_s: float = 0.08  # tolerated dip below motion_off before the burst ends
    lockout_s: float = 0.6  # swallows the return stroke

    def __post_init__(self):
        if not self.motion_off < self.motion_on:
            raise ValueError("motion_off must be below motion_on")


@dataclass(frozen=True)
class Idle:
    pass


@dataclass(frozen=True)
class Tracking:
    gesture: Gesture
    t0: float
    peak_hz: float
    last_active: float


@dataclass(frozen=True)
class Lockout:
    until: float


State = Idle | Tracking | Lockout


def _gesture_of(shift_hz: float) -> Gesture:
    return Gesture.PUSH if shift_hz > 0 else Gesture.PULL


class ThresholdRecognizer:
    def __init__(self, params: RecognizerParams):
        self.params = params
        self.state: State = Idle()

    def feed(self, obs: Observation) -> list[GestureEvent]:
        """Advance the state machine by one observation; return the events it emits.

        Dropout in Tracking returns to Idle (a gesture never spans a dropout). Dropout in
        Lockout leaves it running, so the return stroke cannot fire after a dropout.
        The frame that ends Lockout is processed as Idle.
        Emits at min_duration_s, not at burst end, so latency excludes the full stroke.
        """
        if isinstance(self.state, Lockout):
            if obs.t < self.state.until:
                return []
            self.state = Idle()

        if isinstance(obs, Dropout):
            self.state = Idle()
            return []

        match self.state:
            case Idle():
                self._feed_idle(obs)
                return []
            case Tracking():
                return self._feed_tracking(self.state, obs)
        raise AssertionError(f"unreachable state {self.state!r}")

    def _feed_idle(self, obs: Live) -> None:
        p = self.params
        if obs.motion >= p.motion_on and abs(obs.shift_hz) >= p.shift_on_hz:
            self.state = Tracking(_gesture_of(obs.shift_hz), obs.t, abs(obs.shift_hz), obs.t)

    def _feed_tracking(self, tr: Tracking, obs: Live) -> list[GestureEvent]:
        p = self.params
        active = obs.motion >= p.motion_off and abs(obs.shift_hz) >= p.shift_on_hz
        if not active:
            # A dip never flips the sign; it only counts toward gap_s.
            if obs.t - tr.last_active > p.gap_s:
                self.state = Idle()
            return []
        if _gesture_of(obs.shift_hz) is not tr.gesture:
            self.state = Idle()
            return []
        tr = replace(tr, peak_hz=max(tr.peak_hz, abs(obs.shift_hz)), last_active=obs.t)
        if obs.t - tr.t0 < p.min_duration_s:
            self.state = tr
            return []
        self.state = Lockout(obs.t + p.lockout_s)
        return [GestureEvent(tr.gesture, obs.t, tr.peak_hz, obs.t - tr.t0)]
