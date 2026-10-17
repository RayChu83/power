# Acoustic Gesture Controller

## Milestone 1: live Doppler visualizer

Plays a 19 kHz tone, records the echo, and plots the spectrum around the tone live.

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python doppler_viz.py
```

Wave a hand near the trackpad or mic. The three panels show:

- **Spectrum**: level vs. offset from the carrier (0 Hz = the tone itself).
- **Waterfall**: the same spectrum over time. A moving hand draws streaks off the center line.
- **Net shift**: power-weighted sideband offset in Hz. Positive = toward the mic, negative = away.

Tips:

- Grant microphone access to your terminal app (System Settings > Privacy & Security > Microphone).
- Use the built-in speakers and mic, not Bluetooth headphones.
- Use `--list-devices` to see devices, `--freq` to try 18000-20000 Hz, `--volume` to adjust loudness.
- A hand at about 1 m/s shifts the echo by roughly 110 Hz at 19 kHz.

## Milestone 2: gesture scroll

`gesture_app.py` plays the same tone, recognizes a push (hand toward the mic, scroll up) or pull (away, scroll down), and prints each gesture. It posts real scroll events only with `--live`.

```
.venv/bin/python gesture_app.py          # dry-run: prints fired gestures, posts nothing
.venv/bin/python gesture_app.py --viz    # same, plus the live plots with fired gestures marked
.venv/bin/python gesture_app.py --live   # posts real scrolls (needs Accessibility permission)
```

With `--viz`, each fired gesture appears as a marker on the net-shift trace (green up-triangle = push, orange down-triangle = pull). The recognizer thresholds are untuned guesses, so use `--viz` to watch the trace against the markers before trusting `--live`.

`doppler_viz.py` is the Milestone 1 view. It runs the same pipeline with no scroll bindings, so it never posts input events, and its markers show what the recognizer would have fired.
