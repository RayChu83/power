"""Audio flags shared by the entry points; kept free of matplotlib so importing it stays cheap."""


def add_audio_args(p):
    p.add_argument("--freq", type=float, default=19000.0, help="pilot tone in Hz (default 19000)")
    p.add_argument("--rate", type=int, default=48000, help="sample rate (default 48000)")
    p.add_argument("--volume", type=float, default=0.5, help="tone amplitude 0-1 (default 0.5)")
    p.add_argument("--span", type=float, default=400.0, help="analyze/plot +/- Hz around carrier (default 400)")
    p.add_argument("--nfft", type=int, default=4096, help="analysis window in samples (default 4096)")
    p.add_argument("--guard", type=float, default=20.0, help="Hz around carrier ignored by shift estimate")
    p.add_argument("--input-device", default=None, help="input device index or name")
    p.add_argument("--output-device", default=None, help="output device index or name")
    p.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
