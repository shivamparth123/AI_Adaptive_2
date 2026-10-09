"""
Baseline Comparison Experiment (trace-driven replay)

Compares the adaptive router against fixed-model and single-objective baselines
over the benchmark prompt set WITHOUT re-running any model. For each prompt:
    1. classify the prompt text with the rule-based classifier
    2. let each policy choose a model under a simulated free-RAM condition
    3. look up that model's MEASURED latency and quality for that prompt

Methodology notes (state these in the report):
  - Leave-one-out profiles: the router's per-category profiles are rebuilt WITHOUT the
    prompt being evaluated, so the router is not graded on data it learned from.
  - Complete cases only: prompts where all 4 live models have a valid measurement.
  - RAM scenarios are SIMULATED thresholds fed to the hard-constraint check, not real
    memory pressure. A fixed model that does not fit is reported as "does not fit".
  - Quality is the LLM-judge score (Gemma 3 4B, near-ceiling, self-judging bias) and
    latency is the benchmark's end-to-end time (includes answer verbosity).

Usage:
    cd router
    python experiment_baselines.py

Outputs:
    results/baseline_decisions.csv  - every (scenario, policy, prompt) decision
    results/baseline_summary.csv    - aggregated comparison table
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "dataset"))

import router as R                       # reuse the exact routing logic
from task_classifier import classify_query
from prompts import PROMPTS

RESULTS_DIR = os.path.join(HERE, "..", "results")
BENCHMARK_FILE = os.path.join(RESULTS_DIR, "benchmark_results.csv")
QUALITY_FILE = os.path.join(RESULTS_DIR, "quality_results.csv")
DECISIONS_FILE = os.path.join(RESULTS_DIR, "baseline_decisions.csv")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "baseline_summary.csv")

LIVE = R.LIVE_ROUTER_MODELS
FALLBACK_MODEL = "llama3.2:3b"           # mirrors the router's low-memory fallback

RAM_SCENARIOS = {
    "tight (3.7 GB free)": 3.7,
    "moderate (4.5 GB free)": 4.5,
    "ample (6.0 GB free)": 6.0,
}

# Score-based policies share the router's scoring code and differ only in weights.
# "adaptive router" is the project's chosen default (latency weighted heaviest, since
# benchmark quality was near-ceiling and low-discriminative - see model_profiles
# analysis). "balanced" flips the emphasis (quality weighted more than latency) to
# test how much the routing decision actually depends on that choice.
SCORING_POLICIES = {
    "adaptive router (latency>quality)": {"quality": 0.3, "latency": 0.5, "resource": 0.2},
    "balanced (quality>latency)": {"quality": 0.5, "latency": 0.3, "resource": 0.2},
    "speed-only": {"quality": 0.0, "latency": 1.0, "resource": 0.0},
    "quality-only": {"quality": 1.0, "latency": 0.0, "resource": 0.0},
}

DEFAULT_POLICY = "adaptive router (latency>quality)"  # the project's chosen default


def load_measurements() -> pd.DataFrame:
    bench = pd.read_csv(BENCHMARK_FILE)
    qual = pd.read_csv(QUALITY_FILE)

    bench = bench[bench["error"].isna() | (bench["error"] == "")]
    bench = bench[bench["response_text"].notna() & (bench["response_text"] != "")]
    qual = qual[qual["error"].isna() | (qual["error"] == "")]
    qual = qual[qual["overall_quality"].notna()]

    df = bench.merge(
        qual[["model", "prompt_id", "overall_quality"]],
        on=["model", "prompt_id"], how="inner",
    )
    df["response_time_sec"] = pd.to_numeric(df["response_time_sec"], errors="coerce")
    df["overall_quality"] = pd.to_numeric(df["overall_quality"], errors="coerce")
    df = df.dropna(subset=["response_time_sec", "overall_quality"])
    df = df[df["model"].isin(LIVE)]
    df = df.drop_duplicates(["model", "prompt_id"], keep="last")

    # Keep only prompts that all live models answered validly (fair comparison)
    per_prompt = df.groupby("prompt_id")["model"].nunique()
    complete_ids = per_prompt[per_prompt == len(LIVE)].index
    return df[df["prompt_id"].isin(complete_ids)]


def build_loo_profiles(df: pd.DataFrame, held_out_prompt_id: str) -> dict:
    """Category profiles built from every prompt EXCEPT the held-out one."""
    train = df[df["prompt_id"] != held_out_prompt_id]
    grouped = train.groupby(["model", "category"]).agg(
        q=("overall_quality", "mean"),
        l=("response_time_sec", "mean"),
    )
    profiles = {}
    for (model, category), row in grouped.iterrows():
        profiles.setdefault(model, {"categories": {}})["categories"][category] = {
            "avg_quality": row["q"],
            "avg_latency_sec": row["l"],
        }
    return profiles


def run_experiment(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {(r.model, r.prompt_id): (r.response_time_sec, r.overall_quality)
              for r in df.itertuples()}
    prompts = {p["id"]: p for p in PROMPTS if p["id"] in set(df["prompt_id"])}

    rows = []

    def add_row(scenario, policy, pid, true_cat, pred_cat, model, fits):
        latency, quality = lookup[(model, pid)]
        rows.append({
            "scenario": scenario, "policy": policy, "prompt_id": pid,
            "true_category": true_cat, "predicted_category": pred_cat,
            "chosen_model": model, "fits_in_ram": fits,
            "latency_sec": latency, "quality": quality,
        })

    for pid, p in prompts.items():
        profiles = build_loo_profiles(df, pid)
        predicted = classify_query(p["prompt"])

        for scenario, ram_gb in RAM_SCENARIOS.items():
            feasible = R.apply_hard_constraints(LIVE, ram_gb)

            # Fixed-model baselines (a model that doesn't fit cannot actually run)
            for model in LIVE:
                add_row(scenario, f"fixed: {model}", pid, p["category"], predicted,
                        model, model in feasible)

            # Score-based policies only ever choose among models that fit
            candidates = feasible if feasible else [FALLBACK_MODEL]
            for policy, weights in SCORING_POLICIES.items():
                scores, _ = R.score_candidates(candidates, predicted, profiles, weights=weights)
                chosen = max(scores, key=scores.get)
                add_row(scenario, policy, pid, p["category"], predicted, chosen, True)

            # Oracle: same router, but told the TRUE category. The gap between this and
            # DEFAULT_POLICY is the cost of classifier mistakes.
            scores, _ = R.score_candidates(candidates, p["category"], profiles,
                                           weights=SCORING_POLICIES[DEFAULT_POLICY])
            chosen = max(scores, key=scores.get)
            add_row(scenario, f"{DEFAULT_POLICY} (true category)", pid,
                    p["category"], predicted, chosen, True)

    return pd.DataFrame(rows)


def paired_ci(dec: pd.DataFrame, scenario: str, baseline: str, metric: str,
              n_boot: int = 2000, seed: int = 0):
    """
    Paired bootstrap over prompts: mean of (router - baseline) per prompt with a 95% CI.
    If the interval excludes 0, the difference is unlikely to be chance.
    """
    sub = dec[dec["scenario"] == scenario]
    a = sub[sub["policy"] == DEFAULT_POLICY].set_index("prompt_id")[metric]
    b = sub[sub["policy"] == baseline].set_index("prompt_id")[metric]
    diff = (a - b.reindex(a.index)).to_numpy()
    rng = np.random.default_rng(seed)
    boot_means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return diff.mean(), lo, hi


def print_report(dec: pd.DataFrame):
    n_prompts = dec["prompt_id"].nunique()
    print(f"\nPrompts evaluated (complete cases): {n_prompts}")

    summary_parts = []
    for scenario in RAM_SCENARIOS:
        sub = dec[dec["scenario"] == scenario]
        agg = sub.groupby("policy").agg(
            fits=("fits_in_ram", "all"),
            mean_latency_sec=("latency_sec", "mean"),
            median_latency_sec=("latency_sec", "median"),
            mean_quality=("quality", "mean"),
        ).round(2)

        agg_out = agg.reset_index()
        agg_out.insert(0, "scenario", scenario)
        summary_parts.append(agg_out)

        print("\n" + "=" * 72)
        print(f"SCENARIO: {scenario}")
        print("=" * 72)
        print(f"{'policy':32s} {'mean lat (s)':>12s} {'median (s)':>11s} {'quality':>8s}")
        for policy, r in agg.sort_values("mean_latency_sec").iterrows():
            if not r["fits"]:
                print(f"{policy:32s} {'DOES NOT FIT IN RAM':>32s}")
            else:
                print(f"{policy:32s} {r['mean_latency_sec']:12.2f} "
                      f"{r['median_latency_sec']:11.2f} {r['mean_quality']:8.2f}")

        router_row = agg.loc[DEFAULT_POLICY]
        print(f"\n{DEFAULT_POLICY} vs other policies that fit:")
        for policy, r in agg.iterrows():
            if policy == DEFAULT_POLICY or not r["fits"]:
                continue
            lat_change = (router_row["mean_latency_sec"] - r["mean_latency_sec"]) / r["mean_latency_sec"] * 100
            q_change = router_row["mean_quality"] - r["mean_quality"]
            d_mean, d_lo, d_hi = paired_ci(dec, scenario, policy, "latency_sec")
            print(f"  vs {policy:34s} latency {lat_change:+6.1f}%  "
                  f"(paired {d_mean:+.2f}s, 95% CI [{d_lo:+.2f}, {d_hi:+.2f}])  "
                  f"quality {q_change:+.2f}")

    print("\n" + "=" * 72)
    print("MODEL SELECTION COUNTS BY POLICY (weight-sensitivity comparison)")
    print("=" * 72)
    print("Shows how much the routing decision depends on the weight choice itself.")
    for policy_name in SCORING_POLICIES:
        print(f"\n--- {policy_name} ---")
        counts = (dec[dec["policy"] == policy_name]
                  .groupby(["scenario", "chosen_model"]).size().unstack(fill_value=0))
        print(counts)

    uniq = dec.drop_duplicates("prompt_id")
    acc = (uniq["true_category"] == uniq["predicted_category"]).mean() * 100
    print("\n" + "=" * 72)
    print(f"RULE-BASED CLASSIFIER ACCURACY on these prompts: {acc:.1f}%")
    print("(optimistic: the keyword lists were written knowing these categories)")
    print("=" * 72)
    uniq = uniq.assign(correct=(uniq["true_category"] == uniq["predicted_category"]))
    per_cat = (uniq.groupby("true_category")["correct"].mean() * 100).round(1)
    print(per_cat.to_string())
    print("\nConfusion (rows = true category, columns = predicted category):")
    print(pd.crosstab(uniq["true_category"], uniq["predicted_category"]).to_string())

    pd.concat(summary_parts).to_csv(SUMMARY_FILE, index=False)


# --------------------- Phase C: Pareto dominance analysis ---------------------
PROFILES_FILE = os.path.join(HERE, "..", "results", "model_profiles.json")
PARETO_FILE = os.path.join(RESULTS_DIR, "pareto_analysis.csv")
SWITCH_FREQ_FILE = os.path.join(RESULTS_DIR, "switch_frequency.csv")


def pareto_analysis() -> pd.DataFrame:
    """
    For each category, checks whether each live-router model is Pareto-dominated:
    another model has quality >= its quality AND latency <= its latency, with at
    least one strictly better. A dominated model can NEVER be the rational choice
    under ANY quality/latency weighting - this explains precisely why a model like
    Qwen3:4b never wins, rather than just observing that its weighted score was lower.
    """
    import json
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    categories = set()
    for model in LIVE:
        categories.update(profiles.get(model, {}).get("categories", {}).keys())

    rows = []
    for category in sorted(categories):
        cat_data = {
            m: profiles[m]["categories"][category]
            for m in LIVE if category in profiles.get(m, {}).get("categories", {})
        }
        for model, data in cat_data.items():
            dominators = []
            for other, odata in cat_data.items():
                if other == model:
                    continue
                better_or_equal = (odata["avg_quality"] >= data["avg_quality"]
                                   and odata["avg_latency_sec"] <= data["avg_latency_sec"])
                strictly_better = (odata["avg_quality"] > data["avg_quality"]
                                   or odata["avg_latency_sec"] < data["avg_latency_sec"])
                if better_or_equal and strictly_better:
                    dominators.append(other)
            rows.append({
                "category": category, "model": model,
                "quality": data["avg_quality"], "latency_sec": data["avg_latency_sec"],
                "is_dominated": len(dominators) > 0,
                "dominated_by": ", ".join(dominators),
            })
    return pd.DataFrame(rows)


def print_pareto_summary(pareto_df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("PARETO DOMINANCE ANALYSIS")
    print("=" * 72)
    print("A dominated model is beaten or matched on BOTH quality and latency by some")
    print("other model, with at least one strictly better - it cannot rationally win")
    print("under any weighting, no matter how weights are tuned.\n")

    summary = pareto_df.groupby("model").agg(
        categories_total=("category", "count"),
        categories_dominated=("is_dominated", "sum"),
    )
    summary["pct_dominated"] = (summary["categories_dominated"] / summary["categories_total"] * 100).round(1)
    print(summary.to_string())

    print("\nDetail (only dominated rows):")
    dominated_only = pareto_df[pareto_df["is_dominated"]]
    if len(dominated_only) == 0:
        print("  (no model is dominated in any category)")
    else:
        print(dominated_only[["category", "model", "quality", "latency_sec", "dominated_by"]]
              .to_string(index=False))

    pareto_df.to_csv(PARETO_FILE, index=False)


# --------------------- Phase C: switch-frequency analysis ---------------------
def switch_frequency_analysis(dec: pd.DataFrame) -> pd.DataFrame:
    """
    Counts how often the selected model changes from one prompt to the next, per
    (scenario, policy), using the dataset's prompt_id order as an ILLUSTRATIVE
    sequence. IMPORTANT CAVEAT: this is NOT a real user session - the trace-replay
    evaluates each prompt independently, so this sequencing is a proxy for "if a
    user asked these prompts in this order", not a measured real-usage switch rate.
    State this plainly if cited in the report.
    """
    rows = []
    for (scenario, policy), group in dec.groupby(["scenario", "policy"]):
        g = group.sort_values("prompt_id")
        seq = g["chosen_model"].tolist()
        switches = sum(1 for i in range(1, len(seq)) if seq[i] != seq[i - 1])
        rate = round(100 * switches / (len(seq) - 1), 1) if len(seq) > 1 else 0.0
        rows.append({
            "scenario": scenario, "policy": policy,
            "total_prompts": len(seq), "switches": switches, "switch_rate_pct": rate,
        })
    return pd.DataFrame(rows)


def print_switch_frequency(switch_df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("SWITCH FREQUENCY (illustrative - dataset order, NOT a real usage session)")
    print("=" * 72)
    print(switch_df.to_string(index=False))
    switch_df.to_csv(SWITCH_FREQ_FILE, index=False)


def main():
    df = load_measurements()
    print(f"Measurements loaded: {len(df)} rows across {df['prompt_id'].nunique()} prompts.")
    dec = run_experiment(df)
    dec.to_csv(DECISIONS_FILE, index=False)
    print_report(dec)

    pareto_df = pareto_analysis()
    print_pareto_summary(pareto_df)

    switch_df = switch_frequency_analysis(dec)
    print_switch_frequency(switch_df)

    print(f"\nSaved: {DECISIONS_FILE}\nSaved: {SUMMARY_FILE}"
          f"\nSaved: {PARETO_FILE}\nSaved: {SWITCH_FREQ_FILE}")


if __name__ == "__main__":
    main()