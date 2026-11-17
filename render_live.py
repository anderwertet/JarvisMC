"""Render the live voice through the running bridge, so the wavs are exactly what
Minecraft will play — clone, weight chain and the ", сэр" injection included.

    python render_live.py

Writes audio/heavy_01_ru.wav ... and prints synthesis time per phrase.
"""

from __future__ import annotations

import os
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402
from app.diction import address  # noqa: E402

LINES = [
    ("ru", "Открываю инвентарь.", 1),
    ("ru", "Готово. Время установлено на день.", 2),
    ("ru", "Крипер — это враждебный моб, он взрывается, если подойти слишком близко.", 3),
    ("en", "Opening the inventory now.", 4),
    ("en", "All systems are nominal. The reactor is holding at eighty percent.", 5),
]

BASE = f"http://127.0.0.1:{config.PORT}"


def main() -> int:
    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=BASE, timeout=120) as client:
        try:
            client.get("/health").raise_for_status()
        except httpx.HTTPError:
            print(f"bridge is not answering on {BASE}; start run_server.bat first")
            return 1

        for lang, line, index in LINES:
            text = address(line, lang)
            started = time.perf_counter()
            resp = client.post("/tts.wav", json={"text": text, "lang": lang})
            elapsed = time.perf_counter() - started
            if resp.status_code != 200:
                print(f"{index}: http {resp.status_code} {resp.text[:200]}")
                return 1

            path = config.AUDIO_DIR / f"heavy_{index:02d}_{lang}.wav"
            path.write_bytes(resp.content)
            print(f"{index}: {lang} {elapsed:.2f}s {len(resp.content) / 2 / 24000:.2f}s audio -> {path.name}")

    print(f"\nwritten to {config.AUDIO_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
