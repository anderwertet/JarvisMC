"""Prove the spoken line is still intelligible: transcribe it back.

    python asr_roundtrip.py audio\\file1.wav ...

A synthesised voice that wins on timbre but loses words is a regression, so this
runs the same Whisper build the assistant listens with and prints what it heard.
"""

from __future__ import annotations

import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402
from app.stt import _register_cuda_dlls  # noqa: E402

_register_cuda_dlls()


def read(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if rate != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=rate, target_sr=16000)
    return x


def main() -> int:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        str(config.WHISPER_MODELS["turbo"]),
        device="cuda",
        compute_type=config.STT_COMPUTE,
    )

    for path in sys.argv[1:]:
        if not os.path.exists(path):
            print(f"{os.path.basename(path):26s} missing")
            continue
        segments, info = model.transcribe(
            read(path), beam_size=1, vad_filter=True, without_timestamps=True
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        print(f"{os.path.basename(path):26s} {info.language} -> {text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
