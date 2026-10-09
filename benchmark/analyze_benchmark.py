"""
Performance Analysis Script
Reads results/benchmark_results.csv and produces summary tables:
- per-model overall averages
- per-model per-category averages (this is the important one)
- error counts

Run this anytime after benchmark_text.py has produced data - it's instant,
no models are called, just CSV analysis with pandas.

Usage:
    python analyze_benchmark.py
"""

import os
import pandas as pd

RESULTS_FILE = os.path.join("..", "results", "benchmark_results.csv")
OUTPUT_DIR = os.path.join("..", "results")


def load_data():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Could not find {RESULTS_FILE}. Run benchmark_text.py first.")
    df = pd.read_csv(RESULTS_FILE)
    return df


def main():
    df = load_data()

    print(f"Total rows loaded: {len(df)}")

    # Separate successful runs from errors
    errors = df[df["error"].notna() & (df["error"] != "")]
    ok = df[~(df["error"].notna() & (df["error"] != ""))]

    if len(errors) > 0:
        print(f"\n WARNING: {len(errors)} rows had errors. Error breakdown by model:")
        print(errors.groupby("model").size())
        errors.to_csv(os.path.join(OUTPUT_DIR, "benchmark_errors.csv"), index=False)
        print(f"Full error details saved to results/benchmark_errors.csv")

    # Make sure numeric columns are actually numeric (blank/error rows can break this)
    numeric_cols = ["response_time_sec", "ram_delta_mb", "cpu_percent", "tokens_per_sec", "eval_count"]
    for col in numeric_cols:
        ok[col] = pd.to_numeric(ok[col], errors="coerce")

    # ---------------- 1. Overall per-model summary ----------------
    print("\n=== OVERALL PER-MODEL SUMMARY ===")
    overall = ok.groupby("model").agg(
        avg_response_time_sec=("response_time_sec", "mean"),
        avg_tokens_per_sec=("tokens_per_sec", "mean"),
        avg_ram_delta_mb=("ram_delta_mb", "mean"),
        avg_cpu_percent=("cpu_percent", "mean"),
        avg_tokens_generated=("eval_count", "mean"),
        num_prompts=("prompt_id", "count"),
    ).round(2)
    print(overall)
    overall.to_csv(os.path.join(OUTPUT_DIR, "model_overall_summary.csv"))

    # ---------------- 2. Per-model per-category summary (IMPORTANT) ----------------
    print("\n=== PER-MODEL PER-CATEGORY SUMMARY (response time) ===")
    cat_time = ok.groupby(["model", "category"])["response_time_sec"].mean().unstack().round(2)
    print(cat_time)
    cat_time.to_csv(os.path.join(OUTPUT_DIR, "model_category_response_time.csv"))

    print("\n=== PER-MODEL PER-CATEGORY SUMMARY (tokens/sec) ===")
    cat_tps = ok.groupby(["model", "category"])["tokens_per_sec"].mean().unstack().round(2)
    print(cat_tps)
    cat_tps.to_csv(os.path.join(OUTPUT_DIR, "model_category_tokens_per_sec.csv"))

    print("\n=== PER-MODEL PER-CATEGORY SUMMARY (RAM delta MB) ===")
    cat_ram = ok.groupby(["model", "category"])["ram_delta_mb"].mean().unstack().round(2)
    print(cat_ram)
    cat_ram.to_csv(os.path.join(OUTPUT_DIR, "model_category_ram.csv"))

    print("\nAll summary tables saved to the results/ folder.")
    print("Next step: run quality_evaluation.py to get quality scores, "
          "then these tables combine into task-specific model profiles.")


if __name__ == "__main__":
    main()