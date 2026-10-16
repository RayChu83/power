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
