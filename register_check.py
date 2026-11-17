"""Check the butler register end to end, with UTF-8 that survives the shell."""

from __future__ import annotations

import base64
import json

import httpx

QUESTIONS = ["привет", "здравствуй", "спасибо", "ты кто", "как дела", "что такое редстоун"]


def main() -> int:
    with httpx.Client(base_url="http://127.0.0.1:8765", timeout=90) as client:
        for question in QUESTIONS:
            resp = client.post("/chat", json={"text": question})
            reply = json.loads(resp.content).get("reply", "")
            print(f"{question:22s} -> {reply}")

        resp = client.post("/agent", json={"text": "привет"})
        raw = resp.headers.get("X-Say-B64")
        say = base64.b64decode(raw).decode() if raw else "<no reply>"
        print(f"{'agent: привет':22s} -> {say}")

        print(f"{'volume':22s} -> {client.get('/volume').json()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
