"""Check the brain: which provider is active, is the key good, how fast is a turn.

    python llm_check.py                # what config resolves to
    python llm_check.py --models       # list model ids the provider offers
    python llm_check.py "что такое редстоун"

Prints the resolved provider, a live chat turn, and the same turn through the
agent path so both transports are proven on the real endpoint.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import config  # noqa: E402
from app.agent import Agent  # noqa: E402
from app.backends import build, describe  # noqa: E402
from app.llm import Brain  # noqa: E402


def main() -> int:
    args = [a for a in sys.argv[1:] if a.strip()]
    list_models = "--models" in args
    question = next((a for a in args if not a.startswith("--")), "Скажи одно короткое приветствие.")

    described = describe()
    print(f"config: provider={described['provider']} model={described['model']} local={described['local']}")
    print(f"key file: {config.KEY_FILE} exists={config.KEY_FILE.exists()}")

    if described["local"]:
        print("\nno cloud key found, so the local Ollama model answers.")
        print("paste a key into keys.env (see keys.env.example) to switch to DeepSeek.")

    router = build()
    print(f"active: {router.name}/{router.model}")

    if list_models:
        started = time.perf_counter()
        models = router.models()
        print(f"models ({time.perf_counter() - started:.2f}s, {len(models)} ids):")
        for model in models:
            mark = "  <- active" if model == router.model else ""
            print(f"  {model}{mark}")
        return 0 if models else 1

    brain = Brain()
    started = time.perf_counter()
    try:
        answer, elapsed = brain.ask(question)
        print(f"\nchat  {elapsed:.2f}s ({time.perf_counter() - started:.2f}s wall)\n  {answer}")
    except Exception as exc:  # noqa: BLE001 - this script exists to show the error
        print(f"\nchat FAILED: {type(exc).__name__}: {exc}")
        print(f"provider said: {brain.info().get('last_error')}")
        return 1

    agent = Agent(brain.router)
    plan, elapsed = agent.plan("открой инвентарь")
    print(f"\nagent {elapsed:.2f}s\n  say: {plan['say']}\n  actions: {plan['actions']}")

    brain.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
