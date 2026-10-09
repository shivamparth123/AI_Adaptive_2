"""
ML Task Classifier (Version 2 - TF-IDF + Logistic Regression)

Trained on the 108 labeled prompts in dataset/prompts.py. This is a small training
set (~12 examples per category), so:
  - 5-fold cross-validation is used to get an honest accuracy estimate on the
    training data itself (never trust plain train-then-test-on-same-data accuracy).
  - The real comparison against the rule-based classifier happens on
    dataset/test_prompts.py, a separate held-out set (see compare_classifiers.py).

Usage (standalone - trains and reports cross-validated accuracy):
    python ml_classifier.py

Usage (from other code):
    from ml_classifier import MLClassifier
    clf = MLClassifier()
    clf.train()
    category = clf.predict("Write a function to sort a list")
"""

import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "dataset"))
from prompts import PROMPTS
from classifier_extra_examples import EXTRA_TRAINING_EXAMPLES

# Combined training set: the 108 benchmark prompts + 24 classifier-only extras.
# The extras are NOT part of the LLM benchmark, only used to give the classifiers
# more phrasing variety to learn from.
ALL_TRAINING_PROMPTS = PROMPTS + EXTRA_TRAINING_EXAMPLES


def _build_pipeline(model_type: str) -> Pipeline:
    if model_type == "svm":
        # kernel='linear' with probability=True so predict_proba_top still works
        clf = SVC(kernel="linear", probability=True, C=1.0, random_state=42)
    elif model_type == "logreg":
        clf = LogisticRegression(max_iter=1000, C=1.0)
    else:
        raise ValueError(f"Unknown model_type: {model_type} (use 'logreg' or 'svm')")

    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, lowercase=True)),
        ("clf", clf),
    ])


class MLClassifier:
    def __init__(self, model_type: str = "logreg"):
        self.model_type = model_type
        self.pipeline = _build_pipeline(model_type)
        self._trained = False

    def train(self, prompts: list = None):
        prompts = prompts if prompts is not None else ALL_TRAINING_PROMPTS
        X = [p["prompt"] for p in prompts]
        y = [p["category"] for p in prompts]
        self.pipeline.fit(X, y)
        self._trained = True
        return X, y

    def predict(self, query: str) -> str:
        if not self._trained:
            raise RuntimeError("Call .train() before .predict()")
        return self.pipeline.predict([query])[0]

    def predict_proba_top(self, query: str, top_n: int = 3) -> list:
        """Returns the top N (category, probability) pairs - useful for debugging
        ambiguous queries, e.g. checking how close 'knowledge' and 'other' were."""
        if not self._trained:
            raise RuntimeError("Call .train() before .predict()")
        probs = self.pipeline.predict_proba([query])[0]
        classes = self.pipeline.classes_
        ranked = sorted(zip(classes, probs), key=lambda x: -x[1])
        return ranked[:top_n]


def cross_validated_accuracy(prompts: list = None, k: int = 5, model_type: str = "logreg"):
    """
    Honest accuracy estimate: 5-fold cross-validation on the training data.
    This is NOT the number to compare against the rule-based classifier - use
    compare_classifiers.py for that. This number only tells you how well the
    ML approach fits the training prompts internally.
    """
    prompts = prompts if prompts is not None else ALL_TRAINING_PROMPTS
    X = [p["prompt"] for p in prompts]
    y = [p["category"] for p in prompts]

    pipeline = _build_pipeline(model_type)

    # StratifiedKFold keeps each category represented proportionally in every fold -
    # important here since some categories have as few as 15 examples total.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf)
    return scores.mean(), scores.std()


if __name__ == "__main__":
    print(f"Training set size: {len(ALL_TRAINING_PROMPTS)} "
          f"({len(PROMPTS)} benchmark prompts + {len(EXTRA_TRAINING_EXAMPLES)} extra examples)\n")

    for model_type in ["logreg", "svm"]:
        mean_acc, std_acc = cross_validated_accuracy(model_type=model_type)
        print(f"[{model_type}] 5-fold cross-validated accuracy: "
              f"{mean_acc*100:.1f}% (+/- {std_acc*100:.1f}%)")
    print("(Internal training-set estimates only - see compare_classifiers.py for "
          "the real comparison on held-out test_prompts.py)")

    clf = MLClassifier(model_type="logreg")
    clf.train()

    print("\nQuick sanity check on a few new phrasings (logreg):")
    samples = [
        "hey what's up",
        "my code keeps crashing with an index error",
        "how do you say good morning in italian",
        "what is the boiling point of water",
    ]
    for s in samples:
        pred = clf.predict(s)
        top3 = clf.predict_proba_top(s)
        top3_str = ", ".join(f"{c}={p:.2f}" for c, p in top3)
        print(f"  [{pred:15s}] {s}   (top3: {top3_str})")