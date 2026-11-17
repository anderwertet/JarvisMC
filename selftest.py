"""Offline self-check for every stage. Run: venv\\Scripts\\python.exe selftest.py"""

from __future__ import annotations

import io
import sys
import time
import wave

import numpy as np

sys.path.insert(0, r"E:\JarvisMC")

from app import config  # noqa: E402

RESULTS: list[tuple[str, bool, str]] = []


def step(name: str, fn) -> None:
    started = time.perf_counter()
    try:
        detail = fn()
        RESULTS.append((name, True, f"{detail} [{time.perf_counter() - started:.2f}s]"))
    except Exception as exc:
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))


def check_whisper_load():
    from app.stt import Transcriber

    t = Transcriber()
    t.load()
    return f"device={t.device}"


def check_voice():
    """Load whichever engines win, then speak one phrase per language."""
    from app.tts import Speaker

    s = Speaker()
    s.load()
    if not s.engines:
        raise RuntimeError("no voices loaded")
    detail = []
    for lang in sorted(s.engines):
        pcm, rate = s.synthesize("Проверка связи, сэр." if lang == "ru" else "Radio check, sir.", lang)
        if not pcm:
            raise RuntimeError(f"{lang}: no audio")
        detail.append(f"{lang}={s.engines[lang]} {len(pcm) / 2 / rate:.2f}s")
    return ", ".join(detail)


def check_roundtrip():
    """Speak through the live voice chain, resample to 16k, feed it back to Whisper."""
    from faster_whisper.audio import decode_audio

    from app.stt import Transcriber
    from app.tts import Speaker

    phrase = "The redstone torch powers the piston."
    s = Speaker()
    s.load()
    pcm, rate = s.synthesize(phrase, "en")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)

    audio = decode_audio(io.BytesIO(buf.getvalue()), sampling_rate=16000).astype(np.float32)
    t = Transcriber()
    t.load()
    text, elapsed = t.transcribe(audio)
    if not text:
        raise RuntimeError("whisper returned empty text")
    return f'heard "{text}" in {elapsed:.2f}s'


def check_ollama():
    from app.llm import Brain

    b = Brain()
    if not b.healthy():
        raise RuntimeError("ollama not reachable")
    models = b.models()
    if config.OLLAMA_MODEL not in models:
        raise RuntimeError(f"{config.OLLAMA_MODEL} not pulled; have {models}")
    answer, elapsed = b.ask("Скажи одно короткое приветствие.")
    return f"{elapsed:.2f}s -> {answer[:60]}"


def check_mic():
    import sounddevice as sd

    from app.audio import list_devices

    devices = list_devices()
    if not devices:
        raise RuntimeError("no input devices")
    default = sd.query_devices(kind="input")
    return f"{len(devices)} inputs, default={default['name']}"


def main() -> int:
    config.ensure_dirs()
    step("mic devices", check_mic)
    step("voice chain", check_voice)
    step("whisper load", check_whisper_load)
    step("stt round-trip", check_roundtrip)
    step("ollama chat", check_ollama)

    width = max(len(n) for n, _, _ in RESULTS)
    failed = 0
    print()
    for name, ok, detail in RESULTS:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  {mark}  {name.ljust(width)}  {detail}")
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
