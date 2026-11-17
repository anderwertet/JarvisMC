"""End-to-end check of /agent on the live bridge: latency, say-text, actions.

Usage:
    python check_agent.py
"""

from __future__ import annotations

import base64
import json
import time

import httpx

BASE = "http://127.0.0.1:8765"

CASES = [
    "что такое редстоун",
    "сколько алмазов на каком уровне копать",
    "как сделать кирку",
    "кто такой крипер",
    "зачем нужен верстак",
    "открой инвентарь",
    "напиши в чат я тут",
    "закрой окно",
    "поставь время день",
    "как приручить волка",
]


def decode(value: str | None) -> str:
    if not value:
        return ""
    return base64.b64decode(value).decode("utf-8", "replace")


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=180)
    total = 0.0

    for text in CASES:
        started = time.perf_counter()
        resp = client.post("/agent", json={"text": text})
        elapsed = time.perf_counter() - started
        total += elapsed

        if resp.status_code != 200:
            print(f"{elapsed:5.2f}s | {text:38s} | HTTP {resp.status_code} {resp.text[:80]}")
            continue

        say = decode(resp.headers.get("X-Say-B64"))
        actions = decode(resp.headers.get("X-Actions-B64"))
        timings = resp.headers.get("X-Timings", "")
        print(f"{elapsed:5.2f}s | {text:38s} | {say:60s} | {actions:46s} | {timings}")

    print(f"\navg {total / len(CASES):.2f}s over {len(CASES)} turns")
    client.close()


if __name__ == "__main__":
    main()
