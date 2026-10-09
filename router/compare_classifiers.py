"""
Classifier Comparison: Rule-based (Version 1) vs ML/TF-IDF+LogReg (Version 2)

Evaluated on dataset/test_prompts.py - 36 prompts written AFTER both classifiers
were built, never used to write keywords or train the model. This is the only
fair accuracy number to report; the 108 training prompts and the earlier 105
benchmark prompts are NOT held-out for either classifier.

Usage:
    python compare_classifiers.py
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)
sys.path.append(os.path.join(HERE, "..", "dataset"))

from task_classifier import classify_query
from ml_classifier import MLClassifier
from test_prompts import TEST_PROMPTS


def main():
    clf_logreg = MLClassifier(model_type="logreg")
    clf_logreg.train()

    clf_svm = MLClassifier(model_type="svm")
    clf_svm.train()

    rows = []
    for item in TEST_PROMPTS:
        true_cat = item["category"]
        rule_pred = classify_query(item["prompt"])
        logreg_pred = clf_logreg.predict(item["prompt"])
        svm_pred = clf_svm.predict(item["prompt"])
        rows.append({
            "id": item["id"],
            "prompt": item["prompt"],
            "true_category": true_cat,
            "rule_prediction": rule_pred,
            "rule_correct": rule_pred == true_cat,
            "logreg_prediction": logreg_pred,
            "logreg_correct": logreg_pred == true_cat,
            "svm_prediction": svm_pred,
            "svm_correct": svm_pred == true_cat,
        })

    df = pd.DataFrame(rows)

    rule_acc = df["rule_correct"].mean() * 100
    logreg_acc = df["logreg_correct"].mean() * 100
    svm_acc = df["svm_correct"].mean() * 100

    print("=" * 70)
    print(f"HELD-OUT TEST SET: {len(df)} prompts (never used to build any classifier)")
    print("=" * 70)
    print(f"Rule-based classifier accuracy      : {rule_acc:.1f}%")
    print(f"ML classifier (Logistic Regression) : {logreg_acc:.1f}%")
    print(f"ML classifier (SVM, linear kernel)  : {svm_acc:.1f}%")

    print("\nPer-category accuracy:")
    per_cat = df.groupby("true_category")[["rule_correct", "logreg_correct", "svm_correct"]].mean() * 100
    print(per_cat.round(1).to_string())

    print("\nWhere all three disagree:")
    disagreements = df[~((df["rule_prediction"] == df["logreg_prediction"]) &
                          (df["logreg_prediction"] == df["svm_prediction"]))]
    if len(disagreements) == 0:
        print("  (none - all three classifiers agreed on every prompt)")
    else:
        for _, row in disagreements.iterrows():
            def mark(correct):
                return "correct" if correct else "WRONG"
            print(f"  [{row['true_category']:12s}] \"{row['prompt'][:50]}\"")
            print(f"      rule -> {row['rule_prediction']:12s} ({mark(row['rule_correct'])})   "
                  f"logreg -> {row['logreg_prediction']:12s} ({mark(row['logreg_correct'])})   "
                  f"svm -> {row['svm_prediction']:12s} ({mark(row['svm_correct'])})")

    out_path = os.path.join(HERE, "..", "results", "classifier_comparison.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to {out_path}")

    print("\n" + "=" * 70)
    print("NOTE FOR REPORT: with only 36 test prompts, each one is worth ~2.8")
    print("percentage points. A gap smaller than ~1-2 prompts (about 3-6%) should")
    print("be described as 'comparable', not as one classifier being definitively")
    print("better - the sample is too small to support a strong claim either way.")
    print("=" * 70)


if __name__ == "__main__":
    main()