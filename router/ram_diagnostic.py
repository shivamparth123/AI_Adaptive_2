"""
RAM Diagnostic
Runs one tiny query with keep_alive="0s", then samples free RAM and the RAM held
by Ollama-loaded models every 2 seconds for 40 seconds.

Purpose: find out WHY available RAM drops right after a query.
  - Does free RAM recover, and how fast? (unload lag)
  - Does it stay low even when Ollama holds nothing? (something else is using RAM)

Usage:
    python ram_diagnostic.py
"""

import time

import psutil
from ollama import chat, ps

MODEL = "llama3.2:3b"


def free_gb() -> float:
    return round(psutil.virtual_memory().available / (1024 ** 3), 2)


def ollama_loaded_gb() -> float:
    try:
        return round(sum((getattr(m, "size", 0) or 0) for m in ps().models) / (1024 ** 3), 2)
    except Exception:
        return -1.0


print(f"BEFORE query : free={free_gb()} GB | held by Ollama={ollama_loaded_gb()} GB")

chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Say hi in one short sentence."}],
    options={"num_predict": 30},
    keep_alive="0s",
)

print("\nQuery finished. Sampling every 2 seconds for 40 seconds:")
for t in range(0, 42, 2):
    print(f"  t+{t:2d}s : free={free_gb()} GB | held by Ollama={ollama_loaded_gb()} GB")
    time.sleep(2)