"""Render the Jarvis clone on real phrases, for ear-checking and speed-chasing.

    python jarvis_check.py

Writes E:\\JarvisMC\\audio\\jarvis_*.wav and prints synth latency against audio
length, so the realtime factor is visible rather than guessed.
"""

from __future__ import annotations

import os
import sys
import time
import wave

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402
from app.diction import address  # noqa: E402
from app.tts import Speaker  # noqa: E402

LINES = [
    ("ru", "Открываю инвентарь."),
    ("ru", "Готово. Время установлено на день."),
    ("ru", "Крипер — это враждебный моб, он взрывается, если подойти слишком близко."),
    ("en", "Opening the inventory now."),
    ("en", "All systems are nominal. The reactor is holding at eighty percent."),
]


def main() -> None:
    speaker = Speaker()
    started = time.perf_counter()
    speaker.load()
    if not speaker.engines:
        raise SystemExit("no voice engine loaded")
    print(
        f"load {time.perf_counter() - started:.1f}s engines {speaker.engines} "
        f"(x{config.XTTS_PITCH} st, +{config.XTTS_BODY} dB, drive x{config.XTTS_DRIVE})"
    )

    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    total = 0.0

    for index, (lang, line) in enumerate(LINES, start=1):
        text = address(line, lang)
        started = time.perf_counter()
        pcm, rate = speaker.synthesize(text, lang)
        elapsed = time.perf_counter() - started
        total += elapsed

        seconds = len(pcm) / 2 / rate
        path = config.AUDIO_DIR / f"jarvis_{index:02d}_{lang}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(pcm)

        print(
            f"{index}: {lang} synth {elapsed:.2f}s  audio {seconds:.2f}s  "
            f"rtf x{seconds / elapsed:.2f} | {text}"
        )

    print(f"\navg synth {total / len(LINES):.2f}s -> {config.AUDIO_DIR}")


if __name__ == "__main__":
    main()
