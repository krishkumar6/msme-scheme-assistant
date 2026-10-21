"""Time-to-first-token for a provider, streaming vs not, small vs large prompt.

    python scripts/probe_provider.py gemini
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request

from nabard_qa.config import Settings


def call(base: str, key: str, model: str, prompt: str, stream: bool, extra: dict) -> None:
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 1024,
            "stream": stream, **extra}
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(), headers={
        "Content-Type": "application/json", "Authorization": f"Bearer {key}", "User-Agent": "nabard-qa/0.1"})
    t0 = time.perf_counter()
    first = None
    n = 0
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line:
                    continue
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
    except Exception as e:  # noqa: BLE001
        print(f"  stream={stream} {extra} -> ERROR after {time.perf_counter() - t0:.1f}s: {e}")
        return
    print(f"  stream={stream} {extra} -> first bytes {first:.1f}s, total {time.perf_counter() - t0:.1f}s, {n} lines")


def main(provider: str) -> None:
    s = Settings()
    base, key, model = {
        "gemini": (s.gemini_base_url, s.gemini_api_key, s.gemini_model),
        "groq": (s.groq_base_url, s.groq_api_key, s.groq_model),
    }[provider]
    small = "In one sentence, what is NABARD?"
    big = ("Summarise the following text in 5 bullets.\n\n" + ("NABARD supports farmer producer organisations "
           "through grants, credit guarantees and capacity building across states. ") * 400)  # ~8k tokens
    for label, prompt in (("small", small), ("large ~8k tokens", big)):
        print(f"{model} / {label}")
        for extra in ({}, {"reasoning_effort": "low"}):
            call(base, key, model, prompt, True, extra)
        call(base, key, model, prompt, False, {"reasoning_effort": "low"})


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gemini")
