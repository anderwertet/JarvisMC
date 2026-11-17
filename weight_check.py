"""Ear-check the weight chain without reloading the 5.6 GB voice model.

    python weight_check.py

Reads any `audio/jarvis_*.wav` already on disk, runs the same `_weight()` the
server uses, and writes `audio/heavy_*.wav` beside them. Prints F0 before and
after plus the processing cost, so both halves of "heavier" are measurable.
"""

from __future__ import annotations

import glob
import os
import sys
import time
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402
from app.tts import _weight  # noqa: E402


def read(path: str) -> tuple[np.ndarray, int, bytes]:
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, rate, raw


def f0(x: np.ndarray, rate: int) -> float:
    """Median F0 over voiced frames, via librosa's YIN.

    Autocorrelation on short frames kept locking onto harmonics and reported
    swings that did not exist; YIN is the right tool for a pitch comparison.
    """
    import librosa

    track = librosa.yin(x, fmin=60, fmax=400, sr=rate, frame_length=2048, hop_length=256)
    rms = librosa.feature.rms(y=x, frame_length=2048, hop_length=256)[0]
    keep = rms > (0.06 * rms.max() if rms.size else 0.0)
    if keep.sum() == 0:
        return 0.0
    n = min(len(track), len(keep))
    return float(np.median(track[:n][keep[:n]]))


def low_share(x: np.ndarray, rate: int) -> float:
    """Fraction of spectral energy below 250 Hz — the part that reads as weight."""
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    freqs = np.fft.rfftfreq(len(x), 1 / rate)
    total = spectrum.sum()
    if total <= 0:
        return 0.0
    return float(spectrum[freqs < 250].sum() / total)


def main() -> None:
    sources = sorted(glob.glob(str(config.AUDIO_DIR / "jarvis_*.wav")))
    if not sources:
        raise SystemExit("no audio/jarvis_*.wav to work from; run jarvis_check.py first")

    print(f"pitch {config.XTTS_PITCH} st, shelf {config.XTTS_BODY} dB at "
          f"{config.XTTS_BODY_HZ} Hz, drive x{config.XTTS_DRIVE}\n")

    for path in sources:
        samples, rate, pcm = read(path)

        started = time.perf_counter()
        heavy, out_rate = _weight(
            pcm,
            rate,
            semitones=config.XTTS_PITCH,
            shelf_db=config.XTTS_BODY,
            shelf_hz=config.XTTS_BODY_HZ,
            drive=config.XTTS_DRIVE,
        )
        elapsed = time.perf_counter() - started

        out_path = config.AUDIO_DIR / os.path.basename(path).replace("jarvis_", "heavy_")
        with wave.open(str(out_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(out_rate)
            wf.writeframes(heavy)

        after = np.frombuffer(heavy, dtype="<i2").astype(np.float32) / 32768.0
        before_f0, after_f0 = f0(samples, rate), f0(after, out_rate)
        print(
            f"{os.path.basename(path):18s} F0 {before_f0:6.1f} -> {after_f0:6.1f} Hz "
            f"({after_f0 - before_f0:+.1f})  "
            f"low {low_share(samples, rate) * 100:4.1f}% -> {low_share(after, out_rate) * 100:4.1f}%  "
            f"fx {elapsed * 1000:.0f} ms"
        )

    print(f"\nwritten to {config.AUDIO_DIR}")


if __name__ == "__main__":
    main()
