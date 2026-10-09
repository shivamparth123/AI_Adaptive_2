"""
Build Model Profiles
Merges results/benchmark_results.csv (speed/RAM/CPU) with results/quality_results.csv
(quality scores) into task-specific model profiles: for each (model, category) pair,
a single row with average quality, latency, tokens/sec, RAM, and CPU.

This is the foundation the router will use to score candidate models per query.

Outputs:
    results/model_profiles.csv   - human-readable table
    results/model_profiles.json  - structured format for the router to load directly

Usage:
    python build_model_profiles.py
"""

import json
import os
import pandas as pd

BENCHMARK_FILE = os.path.join("..", "results", "benchmark_results.csv")
QUALITY_FILE = os.path.join("..", "results", "quality_results.csv")
PROFILES_CSV = os.path.join("..", "results", "model_profiles.csv")
PROFILES_JSON = os.path.join("..", "results", "model_profiles.json")

# Which models are allowed in the LIVE router pool vs benchmark-only reference points.
# This mirrors the locked project scope - change here if that scope ever changes.
LIVE_ROUTER_MODELS = {"llama3.2:3b", "phi4-mini:latest", "qwen3:4b", "gemma3:4b"}
BENCHMARK_ONLY_MODELS = {"deepseek-r1:8b", "qwen3-vl:4b"}


def load_and_clean():
    bench = pd.read_csv(BENCHMARK_FILE)
    quality = pd.read_csv(QUALITY_FILE)

    # Keep only successful benchmark rows with real response text
    bench = bench[bench["error"].isna() | (bench["error"] == "")]
    bench = bench[bench["response_text"].notna() & (bench["response_text"] != "")]

    # Keep only successfully-scored quality rows
    quality = quality[quality["error"].isna() | (quality["error"] == "")]
    quality = quality[quality["overall_quality"].notna()]

    # Make sure numeric columns are actually numeric
    for col in ["response_time_sec", "ram_delta_mb", "cpu_percent", "tokens_per_sec"]:
        bench[col] = pd.to_numeric(bench[col], errors="coerce")
    for col in ["correctness", "relevance", "completeness", "clarity",
                "instruction_following", "overall_quality"]:
        quality[col] = pd.to_numeric(quality[col], errors="coerce")

    return bench, quality


def merge_and_profile(bench: pd.DataFrame, quality: pd.DataFrame) -> pd.DataFrame:
    # Merge on the (model, prompt_id) pair - this links each response's
    # performance metrics to its quality score
    merged = pd.merge(
        bench[["model", "prompt_id", "category", "response_time_sec",
               "ram_delta_mb", "cpu_percent", "tokens_per_sec"]],
        quality[["model", "prompt_id", "correctness", "relevance",
                 "completeness", "clarity", "instruction_following", "overall_quality"]],
        on=["model", "prompt_id"],
        how="inner"  # only keep rows present in BOTH files
    )

    print(f"Merged {len(merged)} rows (present in both benchmark and quality data).")

    # Group by (model, category) and average everything
    profile = merged.groupby(["model", "category"]).agg(
        avg_quality=("overall_quality", "mean"),
        avg_correctness=("correctness", "mean"),
        avg_relevance=("relevance", "mean"),
        avg_completeness=("completeness", "mean"),
        avg_clarity=("clarity", "mean"),
        avg_instruction_following=("instruction_following", "mean"),
        avg_latency_sec=("response_time_sec", "mean"),
        avg_tokens_per_sec=("tokens_per_sec", "mean"),
        avg_ram_delta_mb=("ram_delta_mb", "mean"),
        avg_cpu_percent=("cpu_percent", "mean"),
        sample_count=("prompt_id", "count"),
    ).round(2).reset_index()

    profile["in_live_router"] = profile["model"].apply(lambda m: m in LIVE_ROUTER_MODELS)

    return profile


def to_nested_json(profile: pd.DataFrame) -> dict:
    """
    Convert the flat table into a nested dict structure that's easy for the
    router to load and query directly:
    {
        "llama3.2:3b": {
            "in_live_router": true,
            "categories": {
                "coding": {"avg_quality": 8.5, "avg_latency_sec": 12.3, ...},
                ...
            }
        },
        ...
    }
    """
    nested = {}
    for model, group in profile.groupby("model"):
        nested[model] = {
            "in_live_router": bool(group["in_live_router"].iloc[0]),
            "categories": {}
        }
        for _, row in group.iterrows():
            nested[model]["categories"][row["category"]] = {
                "avg_quality": row["avg_quality"],
                "avg_correctness": row["avg_correctness"],
                "avg_relevance": row["avg_relevance"],
                "avg_completeness": row["avg_completeness"],
                "avg_clarity": row["avg_clarity"],
                "avg_instruction_following": row["avg_instruction_following"],
                "avg_latency_sec": row["avg_latency_sec"],
                "avg_tokens_per_sec": row["avg_tokens_per_sec"],
                "avg_ram_delta_mb": row["avg_ram_delta_mb"],
                "avg_cpu_percent": row["avg_cpu_percent"],
                "sample_count": int(row["sample_count"]),
            }
    return nested


def main():
    bench, quality = load_and_clean()
    profile = merge_and_profile(bench, quality)

    profile.to_csv(PROFILES_CSV, index=False)
    print(f"\nSaved flat profile table to {PROFILES_CSV}")

    nested = to_nested_json(profile)
    with open(PROFILES_JSON, "w", encoding="utf-8") as f:
        json.dump(nested, f, indent=2)
    print(f"Saved structured profile JSON to {PROFILES_JSON}")

    # Quick sanity printout - this is the table that answers "which model
    # actually wins per category" instead of relying on a single averaged score
    print("\n=== QUALITY BY MODEL x CATEGORY ===")
    pivot_quality = profile.pivot(index="model", columns="category", values="avg_quality")
    print(pivot_quality)

    print("\n=== LATENCY (sec) BY MODEL x CATEGORY ===")
    pivot_latency = profile.pivot(index="model", columns="category", values="avg_latency_sec")
    print(pivot_latency)

    print("\n=== BEST MODEL PER CATEGORY (by quality, live-router models only) ===")
    live_only = profile[profile["in_live_router"]]
    best_per_cat = live_only.loc[live_only.groupby("category")["avg_quality"].idxmax()]
    print(best_per_cat[["category", "model", "avg_quality", "avg_latency_sec"]].to_string(index=False))


if __name__ == "__main__":
    main()