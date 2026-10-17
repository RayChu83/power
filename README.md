# Acoustic Gesture Controller

Control your Mac with hand gestures in the air, using only its built-in speakers and microphone. No camera, no extra hardware.

## The idea

The app plays a tone too high for people to hear (about 19 kHz). When your hand moves near the laptop, the sound bouncing off it comes back at a slightly different pitch. This is the Doppler effect, the same thing that makes a passing siren change pitch.

- Hand moving **toward** the mic: the echo's pitch goes up.
- Hand moving **away**: the echo's pitch goes down.

The app listens for those pitch changes and works out which gesture you made. It is inspired by Microsoft Research's SoundWave.

Why no camera? Privacy, and because almost nobody does gesture control with sound alone.

## The end goal

A menu-bar app, open source, that turns gestures into everyday actions in any app:

| Gesture | Example action |
|---|---|
| Swipe left / right | Browser back / forward |
| Push (toward the screen) | Play / pause, or scroll up |
| Pull (away from the screen) | Scroll down |
| Hover / hold | To be decided |

Users can remap gestures. Under the hood, a detected gesture sends a fake keystroke or scroll event, so it works everywhere.

Every MacBook and every room sounds different, so a short setup step will have each user perform the gestures a few times, plus a few seconds of doing nothing. The app learns from that, and all data stays on the device.

Success means reliable recognition of the core gestures with few false triggers, reported as per-gesture accuracy, false-trigger rate, and detection latency.

## Where the project is

| Milestone | What | Status |
|---|---|---|
| 1 | Live visualizer that shows your hand's movement as a frequency shift | Done |
| 2 | Push and pull mapped to scroll up and down | Built, thresholds still need tuning on a real MacBook |
| 3 | Onboarding plus a small machine-learning classifier for the full gesture set (including swipes) | Not started |
| 4 | Menu-bar packaging and publishing with accuracy numbers | Not started |

Milestone 2 uses push and pull instead of the swipes originally planned. The pipeline reduces the echo to one signed number (positive means toward the mic), which can tell toward from away but not left from right. Swipes need richer signals and come with Milestone 3.

The detector's thresholds are untuned guesses. Whether push and pull can be told apart reliably on a real MacBook is the main open question, and the recorded-trace replay (below) is how it gets answered.

## Setup

```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Grant microphone access to your terminal app (System Settings > Privacy & Security > Microphone). Use the built-in speakers and mic, not Bluetooth headphones.

## Milestone 1: live Doppler visualizer

```
.venv/bin/python doppler_viz.py
```

Wave a hand near the trackpad or mic. The three panels show:

- **Spectrum**: level vs. offset from the carrier (0 Hz = the tone itself).
- **Waterfall**: the same spectrum over time. A moving hand draws streaks off the center line.
- **Net shift**: power-weighted sideband offset in Hz. Positive = toward the mic, negative = away.

Tips:

- Use `--list-devices` to see devices, `--freq` to try 18000-20000 Hz, `--volume` to adjust loudness.
- A hand at about 1 m/s shifts the echo by roughly 110 Hz at 19 kHz.

## Milestone 2: gesture scroll

`gesture_app.py` plays the same tone, recognizes a push (hand toward the mic, scroll up) or pull (away, scroll down), and prints each gesture. It posts real scroll events only with `--live`.

```
.venv/bin/python gesture_app.py          # dry-run: prints fired gestures, posts nothing
.venv/bin/python gesture_app.py --viz    # same, plus the live plots with fired gestures marked
.venv/bin/python gesture_app.py --live   # posts real scrolls (needs Accessibility permission)
```

With `--viz`, each fired gesture appears as a marker on the net-shift trace (green up-triangle = push, orange down-triangle = pull). Use `--viz` to watch the trace against the markers before trusting `--live`.

`doppler_viz.py` is the Milestone 1 view. It runs the same pipeline with no scroll bindings, so it never posts input events, and its markers show what the recognizer would have fired.

### Tuning with recorded traces

`--record PATH` saves what the app observed (type `p` or `l` + Enter to label a push or pull as it happens). `replay.py` runs the recognizer over a saved trace with no audio device and reports false triggers per minute, per-gesture accuracy, and detection latency.

```
.venv/bin/python gesture_app.py --record gestures.jsonl
.venv/bin/python replay.py gestures.jsonl --params params.json
.venv/bin/python replay.py idle.jsonl --no-gesture   # false-trigger rate on a no-gesture recording
```

## How the code is laid out

Audio flows one way: sound in, then numbers, then a gesture, then an action.

| File | Job |
|---|---|
| `audio.py` | Plays the tone and records the echo |
| `dsp.py`, `levels.py` | Turn audio into a frequency shift and a motion level; detect a silent mic or missing tone |
| `pipeline.py` | Steps through the audio in fixed slices and hands out one reading per slice |
| `gestures.py` | The recognizer: decides when a reading stream is a push or pull |
| `actions.py` | Maps gestures to scroll events (dry-run by default) |
| `gesture_app.py` | The command-line app that wires it all together |
| `doppler_viz.py` | Live plots |
| `trace.py`, `replay.py` | Record and replay readings for tuning |

The recognizer takes plain readings and knows nothing about audio, so it can be tested and replayed without a device. Milestone 3 swaps it for a per-user trained classifier.

Tests (from this folder, one file at a time):

```
.venv/bin/python tests/test_gestures.py
```

## More detail

Design documents are in `../prd/`:

- `requirements.md`: what the project is and the milestones.
- `phase2-architecture.md`: how gesture scroll is designed and why.
