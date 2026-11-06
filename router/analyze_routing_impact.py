"""
Routing-Impact Analysis: does a classifier mistake actually change which model is picked?

Classifier accuracy (77.8% on the held-out set) answers "how often is the category
wrong?". This script answers the question that matters for the router: "when the
category is wrong, how often does the router pick a DIFFERENT model than it would
have with the correct category - and what does that cost in predicted latency/quality?"

Method (read-only, no model calls):
  For each held-out prompt, run the router's selection logic twice - once with the
  classifier's PREDICTED category, once with the TRUE category (oracle) - under each
  mode and RAM scenario, using the real model_profiles.json. The scoring below mirrors
  router.score_candidates / apply_hard_constraints (ratio scoring, cold-start-adjusted
  latency, RAM hard constraint + 0.5 GB buffer). All models are treated as COLD, which
  is how the live CLI/web app run (keep_alive="0s").

Usage (from anywhere):
    python analyze_routing_impact.py
Output: printed tables + results/routing_impact.csv

IMPORTANT: uses the held-out set for MEASUREMENT only. Do not tune keywords against it.
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "dataset"))

from task_classifier import classify_query
from test_prompts import TEST_PROMPTS

PROFILES_FILE = os.path.join(HERE, "..", "results", "model_profiles.json")
CALIBRATION_FILE = os.path.join(HERE, "..", "results", "cold_start_calibration.json")
OUT_FILE = os.path.join(HERE, "..", "results", "routing_impact.csv")

LIVE_MODELS = ["llama3.2:3b", "phi4-mini:latest", "qwen3:4b", "gemma3:4b"]
RAM_GB = {"llama3.2:3b": 2.6, "phi4-mini:latest": 3.1, "qwen3:4b": 3.2, "gemma3:4b": 2.9}
BUFFER_GB = 0.5
MODES = {
    "fast": {"quality": 0.10, "latency": 0.70, "resource": 0.20},
    "balanced": {"quality": 0.30, "latency": 0.50, "resource": 0.20},
    "quality": {"quality": 1.00, "latency": 0.00, "resource": 0.00},
}
# Effective-RAM scenarios (GB). 3.2 -> only Llama fits; 3.5 -> Llama+Gemma;
# 4.5 -> all four fit (your typical operating range); 7.0 -> plenty of headroom.
RAM_SCENARIOS = [3.2, 3.5, 4.5, 7.0]
PRIMARY_SCENARIO = 4.5
FALLBACK_OVERHEAD = {"llama3.2:3b": 4.8, "phi4-mini:latest": 4.76, "qwen3:4b": 4.41, "gemma3:4b": 7.4}


def load_overhead():
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            return {m: v["overhead_sec"] for m, v in json.load(f).items()}, "calibration file"
    except FileNotFoundError:
        return FALLBACK_OVERHEAD, "built-in fallback values (calibration file not found)"


def category_stats(profiles, model, category):
    cat = profiles.get(model, {}).get("categories", {}).get(category)
    if cat is None:
        return 8.0, 30.0  # same neutral fallback the router uses
    return cat["avg_quality"], cat["avg_latency_sec"]


def ratio(values, higher_is_better):
    if higher_is_better:
        best = max(values.values())
        return {k: (v / best if best > 0 else 1.0) for k, v in values.items()}
    best = min(values.values())
    return {k: (best / v if v > 0 else 1.0) for k, v in values.items()}


def select(category, ram_gb, weights, profiles, overhead):
    survivors = [m for m in LIVE_MODELS if ram_gb >= RAM_GB[m] + BUFFER_GB] or ["llama3.2:3b"]
    q, lat, res = {}, {}, {}
    for m in survivors:
        q[m], raw = category_stats(profiles, m, category)
        lat[m] = raw + overhead.get(m, 0.0)  # cold start for every candidate
        res[m] = RAM_GB[m] * 1024
    qn, ln, rn = ratio(q, True), ratio(lat, False), ratio(res, False)
    scores = {m: weights["quality"] * qn[m] + weights["latency"] * ln[m] + weights["resource"] * rn[m]
              for m in survivors}
    return max(scores, key=scores.get)


def main():
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    overhead, overhead_src = load_overhead()

    preds = []
    for item in TEST_PROMPTS:
        preds.append((item, item["category"], classify_query(item["prompt"])))
    n = len(preds)
    wrong = [p for p in preds if p[1] != p[2]]
    print("=" * 78)
    print(f"ROUTING-IMPACT ANALYSIS - {n} held-out prompts, rule-based classifier")
    print(f"Classifier accuracy: {(n - len(wrong)) / n * 100:.1f}%  ({len(wrong)} misclassified)")
    print(f"Cold-start overhead source: {overhead_src}")
    print("=" * 78)

    rows, summary = [], []
    for ram in RAM_SCENARIOS:
        for mode, w in MODES.items():
            changed = harmless_wrong = 0
            dlat, dq = [], []
            for item, true_c, pred_c in preds:
                sel_pred = select(pred_c, ram, w, profiles, overhead)
                sel_true = select(true_c, ram, w, profiles, overhead)
                q_p, l_p = category_stats(profiles, sel_pred, true_c)
                q_t, l_t = category_stats(profiles, sel_true, true_c)
                differs = sel_pred != sel_true
                changed += differs
                if true_c != pred_c and not differs:
                    harmless_wrong += 1
                dlat.append(l_p - l_t)
                dq.append(q_p - q_t)
                rows.append({"effective_ram_gb": ram, "mode": mode, "id": item["id"],
                             "prompt": item["prompt"], "true_category": true_c,
                             "predicted_category": pred_c, "model_with_true_category": sel_true,
                             "model_with_predicted_category": sel_pred, "routing_changed": differs,
                             "latency_penalty_sec": round(l_p - l_t, 2),
                             "quality_delta": round(q_p - q_t, 2)})
            summary.append((ram, mode, changed, harmless_wrong,
                            sum(dlat) / n, sum(dq) / n))

    print(f"\n{'RAM GB':>7} {'mode':>9} | {'routing changed':>16} | {'wrong-but-harmless':>19} | "
          f"{'avg latency penalty':>20} | {'avg quality delta':>17}")
    print("-" * 100)
    for ram, mode, changed, harmless, dl, dq in summary:
        h = f"{harmless}/{len(wrong)}" if wrong else "-"
        print(f"{ram:>7} {mode:>9} | {changed:>3}/{n} ({changed / n * 100:4.1f}%)  | {h:>19} | "
              f"{dl:>+17.2f}s    | {dq:>+17.3f}")

    print(f"\nMisclassified prompts at {PRIMARY_SCENARIO} GB, balanced mode "
          f"(did the mistake change the selected model?):")
    for r in rows:
        if (r["effective_ram_gb"] == PRIMARY_SCENARIO and r["mode"] == "balanced"
                and r["true_category"] != r["predicted_category"]):
            tag = "CHANGED " if r["routing_changed"] else "same    "
            print(f"  [{tag}] {r['true_category']:>13} -> {r['predicted_category']:<13} "
                  f"({r['model_with_true_category']} vs {r['model_with_predicted_category']})  "
                  f"\"{r['prompt'][:46]}\"")

    with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nSaved per-prompt detail to {OUT_FILE}")
    print("\nNOTE FOR REPORT: n=36, so each prompt is ~2.8 points. Latency penalty uses benchmark")
    print("(warm-measured) category latency of the chosen model on the TRUE category - an")
    print("estimate of routing cost, not a live measurement.")


if __name__ == "__main__":
    main()