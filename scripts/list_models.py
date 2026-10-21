"""List the chat models your Groq / Gemini key can use (keys are never printed).

    python scripts/list_models.py groq
    python scripts/list_models.py gemini
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

from nabard_qa.config import Settings

SKIP = ("whisper", "tts", "guard", "embed", "orpheus", "imagen", "veo", "aqa", "image", "audio", "live")


def main(provider: str) -> None:
    s = Settings()
    key, base, current = {
        "groq": (s.groq_api_key, s.groq_base_url, s.groq_model),
        "gemini": (s.gemini_api_key, s.gemini_base_url, s.gemini_model),
    }[provider]
    if not key:
        raise SystemExit(f"{provider} API key is not set in .env")
    req = urllib.request.Request(
        f"{base.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {key}", "User-Agent": "nabard-qa/0.1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            models = json.load(resp)["data"]
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{provider} returned HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    for m in sorted(models, key=lambda m: m["id"]):
        mid = m["id"].removeprefix("models/")
        if any(x in mid.lower() for x in SKIP) or not m.get("active", True):
            continue
        ctx = m.get("context_window")
        print(f"{mid:<50}{f' context={ctx}' if ctx else ''}")
    print(f"\ncurrently configured: {current}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "groq")
