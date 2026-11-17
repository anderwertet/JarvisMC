"""Prove the cloud transport without a cloud key.

    python cloud_probe.py

Starts a throwaway OpenAI-compatible server on 127.0.0.1:8899 that records what
it receives, then points the real `Cloud` backend and the real `Router` at it.
Covers the four things that break in production: the happy path, JSON mode, a
429 that must be retried, and a dead provider that must fall back to local.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402
from app.backends import Cloud, Ollama, Router  # noqa: E402

PORT = 8899
SEEN: list[dict] = []
MODE = {"value": "ok", "hits": 0}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep the probe output clean
        pass

    def do_GET(self) -> None:
        if self.path.endswith("/models"):
            self._json({"data": [{"id": "deepseek-chat"}, {"id": "deepseek-reasoner"}]})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length) or b"{}")
        SEEN.append({"path": self.path, "auth": self.headers.get("Authorization"), "payload": payload})

        if MODE["value"] == "429":
            MODE["hits"] += 1
            if MODE["hits"] == 1:
                self._json({"error": "rate limited"}, 429)
                return
        if MODE["value"] == "down":
            self._json({"error": "upstream exploded"}, 503)
            return

        if payload.get("response_format"):
            content = json.dumps({"say": "Открываю, сэр.", "actions": [{"type": "open_inventory"}]}, ensure_ascii=False)
        else:
            content = "Это блок, передающий сигнал, сэр."

        self._json(
            {
                "choices": [{"message": {"role": "assistant", "content": content}}],
                "usage": {"total_tokens": 42},
            }
        )

    def _json(self, body: dict, status: int = 200) -> None:
        raw = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.3)

    cloud = Cloud("probe", f"http://127.0.0.1:{PORT}", "deepseek-chat", "test-key", timeout=10)
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
        if not ok:
            failures += 1

    print("cloud backend")
    listing = cloud.models()
    check("models list", "deepseek-chat" in listing, str(listing))

    text, elapsed = cloud.chat([{"role": "user", "content": "привет"}])
    check("plain chat", text.startswith("Это блок"), f"{elapsed * 1000:.0f} ms -> {text}")
    check("bearer header", SEEN[-1]["auth"] == "Bearer test-key", str(SEEN[-1]["auth"]))
    check("model forwarded", SEEN[-1]["payload"]["model"] == "deepseek-chat")

    text, _ = cloud.chat([{"role": "user", "content": "открой инвентарь"}], json_mode=True)
    check(
        "json mode",
        SEEN[-1]["payload"].get("response_format") == {"type": "json_object"} and '"say"' in text,
        text,
    )

    MODE["value"], MODE["hits"] = "429", 0
    text, _ = cloud.chat([{"role": "user", "content": "привет"}])
    check("429 retried", MODE["hits"] == 2 and bool(text), f"hits={MODE['hits']}")

    MODE["value"] = "down"
    try:
        cloud.chat([{"role": "user", "content": "привет"}])
        check("503 surfaces as error", False, "no exception")
    except Exception as exc:  # noqa: BLE001 - the probe wants the failure
        check("503 surfaces as error", "503" in str(exc), str(exc)[:70])

    print("\nrouter")
    local = Ollama(config.OLLAMA_URL, config.OLLAMA_MODEL, timeout=30)
    router = Router(cloud, local)

    MODE["value"] = "ok"
    router.reset_cooldown()
    text, _ = router.chat([{"role": "user", "content": "привет"}])
    check("primary used", router.name == "probe" and text.startswith("Это блок"), text)

    MODE["value"] = "down"
    router.reset_cooldown()
    text, elapsed = router.chat([{"role": "user", "content": "что такое редстоун"}])
    check("falls back to ollama", bool(text) and router.last_error != "", f"{elapsed:.2f}s -> {text[:50]}")

    router.close()
    cloud.close()
    server.shutdown()

    print(f"\n{'all checks passed' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
