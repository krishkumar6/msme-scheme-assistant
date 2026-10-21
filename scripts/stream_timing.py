"""Measure when each SSE event of /api/ask/stream reaches the client.

    python scripts/stream_timing.py "your question" [fast|deep]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request


def main(question: str, mode: str = "fast", url: str = "http://127.0.0.1:8000/api/ask/stream") -> None:
    body = json.dumps({"question": question, "mode": mode}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    deltas, first_delta, last_delta, chars = 0, None, None, 0
    with urllib.request.urlopen(req, timeout=300) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            ev = json.loads(line[5:])
            t = time.perf_counter() - t0
            if ev["type"] == "delta":
                deltas += 1
                chars += len(ev["text"])
                first_delta = first_delta or t
                last_delta = t
                if deltas <= 5 or deltas % 25 == 0:
                    print(f"{t:7.2f}s  delta #{deltas:<4} {ev['text'][:40]!r}")
            else:
                extra = ev.get("result", {}).get("provider", "") if ev["type"] == "done" else ""
                print(f"{t:7.2f}s  {ev['type']} {extra}")
    if deltas:
        span = last_delta - first_delta
        print(f"\n{deltas} deltas, {chars} chars; first text at {first_delta:.2f}s, "
              f"last at {last_delta:.2f}s (text streamed over {span:.2f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "fast")
