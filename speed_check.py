"""Time the brain on the live bridge: five phrases, no speech, LLM only."""

from __future__ import annotations

import time

import httpx

PHRASES = [
    "что такое редстоун",
    "как сделать каменную кирку",
    "сколько камня нужно для печки",
    "кто такой крипер",
    "поставь время день",
]


def main() -> int:
    times = []
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=120) as client:
        health = client.get("/health").json()
        print("llm:", health["llm"])

        for phrase in PHRASES:
            started = time.perf_counter()
            resp = client.post("/chat", json={"text": phrase})
            elapsed = time.perf_counter() - started
            times.append(elapsed)
            reply = resp.json().get("reply", "")
            print(f"{elapsed:5.2f}s  {phrase:34s} -> {reply[:70]}")

        print(f"\navg {sum(times) / len(times):.2f}s  max {max(times):.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
