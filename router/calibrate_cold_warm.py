"""
Cold/Warm Calibration (Phase B) - Multi-sample version

Measures each model's cold-start overhead with REPEATS cold/warm pairs per model,
not just one. This gives a real mean, standard deviation, and sample count instead
of a single noisy point estimate.

For each of the 4 live-router models, repeated REPEATS times:
    1. COLD call - model not loaded (previous rep's WARM call unloaded it), kept
       loaded afterward (keep_alive="5m")
    2. WARM call - same model, immediately after, already loaded (keep_alive="0s",
       unloads it again so the NEXT repetition's cold call is genuinely cold)

Per-pair overhead = max(0, cold_sec - warm_sec) - floored at 0 because a warm call
taking longer than a cold call by chance is timing noise, not negative load time.

Still NOT a full re-benchmark: this measures load-time overhead with ONE fixed short
prompt, not per-category. Load time is primarily a function of model file size and
disk speed, which doesn't vary by task category - that is a stated simplification,
not a hidden one.

RESUMABLE: safe to stop (Ctrl+C) and rerun - it continues from however many
repetitions are already recorded per model, same pattern as benchmark_text.py.

Usage:
    cd router
    python calibrate_cold_warm.py
"""

import csv
import json
import os
import statistics
import sys
import time

from ollama import chat

sys.path.append(os.path.dirname(__file__))
import router as R

REPEATS = 10  # per model - 10 x 4 models = 80 total calls. Raise toward 12-15 (up to
              # ~100-120 calls) for even tighter statistics if you have the time; lower
              # only if you're short on time, since fewer samples means a noisier mean.

SAMPLES_FILE = os.path.join("..", "results", "cold_warm_calibration_samples.csv")
SUMMARY_FILE = os.path.join("..", "results", "cold_start_calibration.json")
CALIBRATION_PROMPT = "Say OK."

FIELDNAMES = ["model", "rep", "cold_sec", "warm_sec", "overhead_sec"]


def load_completed_reps() -> dict:
    """Returns {model: count_of_completed_reps} by reading any existing samples file."""
    counts = {model: 0 for model in R.LIVE_ROUTER_MODELS}
    if os.path.exists(SAMPLES_FILE):
        with open(SAMPLES_FILE, "r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["model"] in counts:
                    counts[row["model"]] += 1
    return counts


def ensure_samples_file():
    os.makedirs(os.path.dirname(SAMPLES_FILE), exist_ok=True)
    if not os.path.exists(SAMPLES_FILE):
        with open(SAMPLES_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_sample(row: dict):
    with open(SAMPLES_FILE, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)


def measure_one_pair(model: str) -> dict:
    # COLD call - keep_alive="5m" so it stays loaded for the warm call right after
    start = time.time()
    chat(
        model=model,
        messages=[{"role": "user", "content": CALIBRATION_PROMPT}],
        options={"num_predict": 20},
        think=False,
        keep_alive="5m",
    )
    cold_sec = round(time.time() - start, 2)

    # WARM call - same model, should still be loaded from the call above.
    # keep_alive="0s" unloads it after, so the NEXT repetition's cold call is genuine.
    start = time.time()
    chat(
        model=model,
        messages=[{"role": "user", "content": CALIBRATION_PROMPT}],
        options={"num_predict": 20},
        think=False,
        keep_alive="0s",
    )
    warm_sec = round(time.time() - start, 2)

    overhead_sec = round(max(0.0, cold_sec - warm_sec), 2)
    return {"cold_sec": cold_sec, "warm_sec": warm_sec, "overhead_sec": overhead_sec}


def summarize(samples_by_model: dict) -> dict:
    summary = {}
    for model, rows in samples_by_model.items():
        if not rows:
            continue
        overheads = [r["overhead_sec"] for r in rows]
        colds = [r["cold_sec"] for r in rows]
        warms = [r["warm_sec"] for r in rows]
        summary[model] = {
            "overhead_sec": round(statistics.mean(overheads), 2),  # mean - read by router.py
            "mean_overhead_sec": round(statistics.mean(overheads), 2),
            "std_overhead_sec": round(statistics.stdev(overheads), 2) if len(overheads) > 1 else 0.0,
            "median_overhead_sec": round(statistics.median(overheads), 2),
            "min_overhead_sec": round(min(overheads), 2),
            "max_overhead_sec": round(max(overheads), 2),
            "mean_cold_sec": round(statistics.mean(colds), 2),
            "mean_warm_sec": round(statistics.mean(warms), 2),
            "sample_count": len(rows),
        }
    return summary


def load_all_samples() -> dict:
    samples_by_model = {model: [] for model in R.LIVE_ROUTER_MODELS}
    with open(SAMPLES_FILE, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["model"] in samples_by_model:
                samples_by_model[row["model"]].append({
                    "cold_sec": float(row["cold_sec"]),
                    "warm_sec": float(row["warm_sec"]),
                    "overhead_sec": float(row["overhead_sec"]),
                })
    return samples_by_model


def main():
    ensure_samples_file()
    completed = load_completed_reps()

    total_pairs_needed = REPEATS * len(R.LIVE_ROUTER_MODELS)
    already_done = sum(completed.values())
    print(f"Target: {REPEATS} reps x {len(R.LIVE_ROUTER_MODELS)} models = {total_pairs_needed} pairs "
          f"({total_pairs_needed * 2} total calls). Already done: {already_done}.")

    pair_num = already_done
    for model in R.LIVE_ROUTER_MODELS:
        remaining = REPEATS - completed[model]
        if remaining <= 0:
            print(f"\n{model}: already have {completed[model]}/{REPEATS} reps, skipping.")
            continue

        print(f"\n=== Calibrating {model} ({remaining} reps remaining) ===")
        for rep in range(completed[model], REPEATS):
            pair_num += 1
            print(f"[{pair_num}/{total_pairs_needed}] {model} rep {rep + 1}/{REPEATS}...",
                  end=" ", flush=True)
            result = measure_one_pair(model)
            append_sample({"model": model, "rep": rep, **result})
            print(f"cold={result['cold_sec']}s warm={result['warm_sec']}s "
                  f"overhead={result['overhead_sec']}s")

    print("\nAll calibration pairs complete. Computing summary statistics...")
    samples_by_model = load_all_samples()
    summary = summarize(samples_by_model)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved per-sample data to {SAMPLES_FILE}")
    print(f"Saved summary statistics to {SUMMARY_FILE}")
    print("\n=== SUMMARY ===")
    for model, s in summary.items():
        print(f"  {model:20s} mean_overhead={s['mean_overhead_sec']}s "
              f"(std={s['std_overhead_sec']}s, n={s['sample_count']}) "
              f"[cold~{s['mean_cold_sec']}s, warm~{s['mean_warm_sec']}s]")


if __name__ == "__main__":
    main()