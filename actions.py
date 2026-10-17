import ctypes
import ctypes.util
import os
import sys
from dataclasses import dataclass
from typing import Callable

from gestures import Gesture


@dataclass(frozen=True)
class Scroll:
    lines: int  # + = up


Action = Scroll  # becomes a union when swipes add KeyChord (back/forward)

DEFAULT_BINDINGS: dict[Gesture, Action] = {
    Gesture.PUSH: Scroll(+3),  # PRD: push -> scroll up
    Gesture.PULL: Scroll(-3),  # PRD: pull -> scroll down
}

Actuator = Callable[[Action], None]


def _accessibility_trusted() -> bool:
    """macOS silently drops posted events without Accessibility permission, so ask directly."""
    if sys.platform != "darwin":
        return True
    lib = ctypes.util.find_library("ApplicationServices")
    if lib is None:
        return True
    return bool(ctypes.cdll.LoadLibrary(lib).AXIsProcessTrusted())


def _permission_message(detail: str = "") -> str:
    terminal = os.environ.get("TERM_PROGRAM", "your terminal app")
    msg = (
        "Cannot post scroll events. Grant Accessibility permission in "
        f"System Settings > Privacy & Security > Accessibility to {terminal} "
        "(the terminal app running this), then restart it."
    )
    return f"{msg} ({detail})" if detail else msg


def _log_actuator(action: Action) -> None:
    print(f"[dry-run] {action}")


def make_actuator(live: bool) -> Actuator:
    """live: pynput.mouse.Controller().scroll(0, lines); raises PermissionError with a clear
    message if Accessibility permission is missing. Otherwise: log only."""
    if not live:
        return _log_actuator

    if not _accessibility_trusted():
        raise PermissionError(_permission_message())
    from pynput import mouse  # lazy: dry-run and tests never need pynput

    try:
        controller = mouse.Controller()
    except Exception as e:
        raise PermissionError(_permission_message(repr(e))) from e

    def actuate(action: Action) -> None:
        controller.scroll(0, action.lines)

    return actuate
