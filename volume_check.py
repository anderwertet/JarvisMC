"""Check that the volume intent turns into a set_volume action."""

from __future__ import annotations

import base64
import json

import httpx

PHRASES = [
    "громкость 50 процентов",
    "сделай потише",
    "сделай громче",
    "выключи звук",
    "открой инвентарь",
]


def main() -> int:
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=90) as client:
        client.get("/health").raise_for_status()
        for phrase in PHRASES:
            resp = client.post("/agent", json={"text": phrase})
            say_raw = resp.headers.get("X-Say-B64", "")
            act_raw = resp.headers.get("X-Actions-B64", "")
            say = base64.b64decode(say_raw).decode() if say_raw else ""
            actions = json.loads(base64.b64decode(act_raw)) if act_raw else []
            kinds = [f"{a.get('type')}:{a.get('value', a.get('command', ''))}" for a in actions]
            joined = ",".join(kinds) if kinds else "[]"
            print(f"{phrase:26s} -> {joined:32s} | {say}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
