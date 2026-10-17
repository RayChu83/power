import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import actions
from actions import DEFAULT_BINDINGS, Scroll, make_actuator
from gestures import Gesture


class FakeController:
    scrolls: list = []
    fail = False

    def __init__(self):
        if FakeController.fail:
            raise RuntimeError("not trusted")

    def scroll(self, dx, dy):
        FakeController.scrolls.append((dx, dy))


def install_fake_pynput(monkey):
    FakeController.scrolls = []
    FakeController.fail = False
    pkg = types.ModuleType("pynput")
    mouse = types.ModuleType("pynput.mouse")
    mouse.Controller = FakeController
    pkg.mouse = mouse
    monkey["pynput"] = pkg
    monkey["pynput.mouse"] = mouse


class FakePynput:
    def __enter__(self):
        self.saved = {k: sys.modules.get(k) for k in ("pynput", "pynput.mouse")}
        self.trusted = actions._accessibility_trusted
        actions._accessibility_trusted = lambda: True
        install_fake_pynput(sys.modules)
        return self

    def __exit__(self, *exc):
        actions._accessibility_trusted = self.trusted
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def test_bindings_cover_both_gestures():
    assert DEFAULT_BINDINGS[Gesture.PUSH] == Scroll(+3)
    assert DEFAULT_BINDINGS[Gesture.PULL] == Scroll(-3)


def test_dry_run_logs_bound_actions(capsys=None):
    import contextlib
    import io

    buf = io.StringIO()
    act = make_actuator(live=False)
    with contextlib.redirect_stdout(buf):
        for g in (Gesture.PUSH, Gesture.PULL):
            act(DEFAULT_BINDINGS[g])
    out = buf.getvalue()
    assert "Scroll(lines=3)" in out and "Scroll(lines=-3)" in out, out


def test_dry_run_imports_nothing_from_pynput():
    for k in [k for k in sys.modules if k == "pynput" or k.startswith("pynput.")]:
        del sys.modules[k]
    make_actuator(live=False)(Scroll(+3))
    leaked = [k for k in sys.modules if k == "pynput" or k.startswith("pynput.")]
    assert not leaked, leaked


def test_live_scrolls_through_pynput():
    with FakePynput():
        act = make_actuator(live=True)
        act(DEFAULT_BINDINGS[Gesture.PUSH])
        act(DEFAULT_BINDINGS[Gesture.PULL])
        assert FakeController.scrolls == [(0, 3), (0, -3)], FakeController.scrolls


def test_live_without_accessibility_names_the_setting_and_terminal():
    with FakePynput():
        actions._accessibility_trusted = lambda: False
        old = os.environ.get("TERM_PROGRAM")
        os.environ["TERM_PROGRAM"] = "FakeTerm"
        try:
            make_actuator(live=True)
        except PermissionError as e:
            msg = str(e)
        else:
            raise AssertionError("expected PermissionError")
        finally:
            if old is None:
                del os.environ["TERM_PROGRAM"]
            else:
                os.environ["TERM_PROGRAM"] = old
        assert "System Settings > Privacy & Security > Accessibility" in msg, msg
        assert "FakeTerm" in msg, msg
        assert FakeController.scrolls == []


def test_live_controller_failure_becomes_permission_error():
    with FakePynput():
        FakeController.fail = True
        try:
            make_actuator(live=True)
        except PermissionError as e:
            assert "Accessibility" in str(e), e
        else:
            raise AssertionError("expected PermissionError")


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {e!r}")
    print(f"{len(tests) - failed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
