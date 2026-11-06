"""
Hybrid Classifier: rules first, ML as the fallback for what the rules cannot place.

Why hybrid: the rule-based classifier is fast and fully explainable, but its failures
are almost all "no keyword matched -> 'other'" (e.g. "Write a short poem...",
"Is this message spam?"). The ML classifier (TF-IDF + SVM) generalizes from wording,
but on its own was no better than the rules overall (75.0% vs 77.8% held-out).
Combining them uses each where it is strongest.

Decision order for one query:
  1. Rules match a category                     -> use it            (source: "rule")
  2. No rule matched AND it is a follow-up
     ("continue", "yes and numerical") with
     earlier conversation history               -> inherit earlier   (source: "inherited")
  3. No rule matched, 3+ words                  -> ML prediction     (source: "ml")
  4. Otherwise                                  -> "other"           (source: "default")

The ML model is trained lazily on first use (about a second on this dataset) from the
existing training data via ml_classifier.MLClassifier. If scikit-learn or the training
data is unavailable, this degrades to the rule-based behaviour with a one-time warning.
"""
import os
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(HERE)

from task_classifier import classify_query, is_followup

ML_MIN_WORDS = 3  # 1-2 word inputs carry too little signal for the ML model to be trusted


class HybridClassifier:
    def __init__(self, ml=None, model_type: str = "svm", ml_min_words: int = ML_MIN_WORDS):
        self._ml = ml                    # inject a trained model (used by tests/evaluation)
        self._model_type = model_type
        self._ml_min_words = ml_min_words
        self._load_attempted = ml is not None
        self.last_source = "rule"        # how the most recent classification was decided

    def _get_ml(self):
        if self._load_attempted:
            return self._ml
        self._load_attempted = True
        try:
            from ml_classifier import MLClassifier
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # sklearn FutureWarning noise
                clf = MLClassifier(model_type=self._model_type)
                clf.train()
            self._ml = clf
        except Exception as e:
            print(f"\nNOTE: ML fallback unavailable ({e}); using rule-based classification only.\n")
        return self._ml

    def classify(self, query: str, history: list = None) -> str:
        category = classify_query(query)
        if category != "other":
            self.last_source = "rule"
            return category

        if history and is_followup(query):
            for msg in reversed(history):
                if msg.get("role") != "user":
                    continue
                earlier = classify_query(msg.get("content", ""))
                if earlier != "other":
                    self.last_source = "inherited"
                    return earlier

        if len(query.split()) >= self._ml_min_words:
            ml = self._get_ml()
            if ml is not None:
                try:
                    predicted = ml.predict(query)
                    self.last_source = "ml"
                    return predicted
                except Exception as e:
                    print(f"\nNOTE: ML prediction failed ({e}); falling back to 'other'.\n")

        self.last_source = "default"
        return "other"