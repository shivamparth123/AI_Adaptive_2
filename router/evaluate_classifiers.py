"""
Evaluates rule-based vs ML (SVM) vs HYBRID on both held-out sets, with two safeguards
that make the numbers honest:

  1. NEAR-DUPLICATE FILTER. Each test prompt is compared (TF-IDF cosine) to the ORIGINAL
     training prompts. Prompts with similarity >= 0.5 are near-copies of training data
     (e.g. "how do you say thank you very much in japanese" vs the benchmark's
     "How do you say 'Thank you very much' in Japanese?") and would inflate ML scores.
     The headline numbers use only the CLEAN prompts.
  2. BASE vs AUGMENTED ML. "base" trains on the original data only; "aug" also adds
     the terse real-style examples (dataset/classifier_extra_examples_v2.py), so you can
     see what the extra data actually buys.

Test sets:
  v1 - dataset/test_prompts.py     (36; its failures were seen during development)
  v2 - dataset/test_prompts_v2.py  (45; written by the assistant, so labels are one
                                    person's judgement, and 9 prompts turned out to be
                                    near-duplicates of training data)

Usage:  python evaluate_classifiers.py
Output: printed tables + results/classifier_evaluation_v2.csv
"""
import os
import sys
import warnings

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "dataset"))

from task_classifier import classify_query
from hybrid_classifier import HybridClassifier
from ml_classifier import MLClassifier, PROMPTS, EXTRA_TRAINING_EXAMPLES, EXTRA_TRAINING_EXAMPLES_V2
from test_prompts import TEST_PROMPTS
from test_prompts_v2 import TEST_PROMPTS_V2

OUT = os.path.join(HERE, "..", "results", "classifier_evaluation_v2.csv")
NEAR_DUPLICATE_SIMILARITY = 0.5
METHODS = ["rule", "svm_base", "svm_aug", "hybrid_base", "hybrid_aug"]


def report(title, d):
    n = len(d)
    if n == 0:
        return
    print(f"\n{title}  (n={n}, one prompt = {100 / n:.1f} points)")
    for m in METHODS:
        correct = int((d[m] == d["true"]).sum())
        print(f"  {m:<12} {correct / n * 100:5.1f}%  ({correct}/{n})")


def main():
    base_train = PROMPTS + EXTRA_TRAINING_EXAMPLES
    aug_train = base_train + EXTRA_TRAINING_EXAMPLES_V2
    if not EXTRA_TRAINING_EXAMPLES_V2:
        print("NOTE: dataset/classifier_extra_examples_v2.py not found - 'aug' equals 'base'.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        svm_base, svm_aug = MLClassifier("svm"), MLClassifier("svm")
        svm_base.train(base_train)
        svm_aug.train(aug_train)
    hyb_base, hyb_aug = HybridClassifier(ml=svm_base), HybridClassifier(ml=svm_aug)

    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit([p["prompt"] for p in base_train])
    train_matrix = vec.transform([p["prompt"] for p in base_train])

    rows = []
    for set_name, items in (("v1", TEST_PROMPTS), ("v2", TEST_PROMPTS_V2)):
        sims = cosine_similarity(vec.transform([i["prompt"] for i in items]), train_matrix).max(axis=1)
        for item, sim in zip(items, sims):
            q = item["prompt"]
            rows.append({
                "set": set_name, "id": item["id"], "prompt": q, "true": item["category"],
                "rule": classify_query(q),
                "svm_base": svm_base.predict(q), "svm_aug": svm_aug.predict(q),
                "hybrid_base": hyb_base.classify(q), "hybrid_aug": hyb_aug.classify(q),
                "hybrid_aug_source": hyb_aug.last_source,
                "max_similarity_to_training": round(float(sim), 2),
                "near_duplicate": bool(sim >= NEAR_DUPLICATE_SIMILARITY),
            })
    df = pd.DataFrame(rows)
    clean = df[~df["near_duplicate"]]

    print("=" * 76)
    print("CLASSIFIER EVALUATION  (rule vs SVM vs hybrid, base vs augmented training data)")
    print(f"Near-duplicates of training prompts excluded from 'CLEAN': "
          f"{int(df['near_duplicate'].sum())}/{len(df)}")
    print("=" * 76)
    report("v1 - all", df[df["set"] == "v1"])
    report("v2 - all (includes near-duplicates, inflated)", df[df["set"] == "v2"])
    report("v2 - CLEAN", clean[clean["set"] == "v2"])
    report(">>> v1 + v2 CLEAN combined (headline number)", clean)

    per = pd.DataFrame({m: (clean[m] == clean["true"]).groupby(clean["true"]).mean() * 100
                        for m in ("rule", "svm_aug", "hybrid_aug")}).round(0)
    print("\nPer-category accuracy on CLEAN combined (%):")
    print(per.to_string())

    bad = clean[clean["hybrid_aug"] != clean["true"]]
    print(f"\nhybrid_aug errors on CLEAN prompts ({len(bad)}):")
    for _, r in bad.iterrows():
        print(f"  [{r['true']:>13}] -> {r['hybrid_aug']:<13} via {r['hybrid_aug_source']:<9} \"{r['prompt'][:52]}\"")

    df.to_csv(OUT, index=False)
    print(f"\nSaved to {OUT}")
    print("\nREPORT NOTES: gaps of 1-2 prompts are noise at these sample sizes. Test labels were")
    print("written by one person; v2 was written by the assistant. Say so in the report.")


if __name__ == "__main__":
    main()