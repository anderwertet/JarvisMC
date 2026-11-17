"""Time speech recognition on real audio of several lengths.

    python stt_speed.py

Prints wall time per file and the per-second-of-audio cost, which is what the
player feels: a long utterance must not turn into a ten-second wait.
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


def read(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if rate != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=rate, target_sr=16000)
    return x, 16000


def main() -> int:
    import time

    from faster_whisper import WhisperModel

    files = sys.argv[1:]
    if not files:
        files = [
            "heavy_01_ru.wav",
            "heavy_03_ru.wav",
            "heavy_05_en.wav",
        ]

    model = WhisperModel(
        str(config.WHISPER_MODELS[config.STT_MODEL]),
        device=config.STT_DEVICE,
        compute_type=config.STT_COMPUTE,
    )
    list(model.transcribe(read(str(config.AUDIO_DIR / files[0]))[0], beam_size=1)[0])

    for name in files:
        path = config.AUDIO_DIR / name
        if not path.exists():
            print(f"{name:24s} missing")
            continue
        audio, _ = read(str(path))
        started = time.perf_counter()
        segments, info = model.transcribe(
            audio,
            beam_size=config.STT_BEAM_SIZE,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        elapsed = time.perf_counter() - started
        seconds = len(audio) / 16000
        print(
            f"{name:24s} {seconds:5.2f}s audio -> {elapsed:5.2f}s "
            f"(x{seconds / elapsed:4.1f} realtime) | {text[:52]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
