"""
Task Classifier (Version 1 - Rule-based)
Classifies a user's query into one of the 9 categories used in benchmarking,
so the router can look up task-specific model profiles for that category.

This is deliberately simple (keyword matching) for Version 1. It mirrors the
categories in dataset/prompts.py exactly, so the classifier's output can be
used directly as a lookup key into model_profiles.json.

Usage (standalone test):
    python task_classifier.py

Usage (from other code):
    from task_classifier import classify_query
    category = classify_query("Write a Python function to reverse a string")
"""

import re

# Keyword sets per category. Order matters - checked top to bottom, first match wins.
# Keep "other" as the final fallback.
CATEGORY_KEYWORDS = {
    "programming": [
        "code", "python", "function", "script", "program", "debug", "bug", "error in this",
        "c++", "java", "sql", "algorithm", "loop", "variable", "class ", "def ", "compile",
        "syntax", "regex", "api", "implement a", "write a program", "sort", "array", "linked list",
        "reverse a string", "reverse this", "merge two", "convert this list", "parse", "iterate",
        "for loop", "while loop", "index error", "crashing", "bash", "javascript", "typescript",
    ],
    "reasoning": [
        "solve", "prove", "derivative", "equation", "calculate", "math", "logic", "if.*then",
        "compare and", "trade-off", "profit", "percentage", "geometry", "algebra", "word problem",
        "workers", "days", "km/h", "area of", "complexity", "logically valid", "grows faster",
        "asymptotically", "average speed", "dimensions", "perimeter", "which is bigger",
    ],
    "language": [
        "translate", "translation", "in hindi", "in french", "in spanish", "in japanese",
        "rewrite", "grammar", "formal", "email to", "write an email", "simplify this sentence",
        "means in english", "how do you say", "how would i say", "sound more professional",
        "sound more formal", "politely decline", "formal way to say", "decline this",
    ],
    "content": [
        "summarize", "summary", "write a story", "write a poem", "creative", "brainstorm",
        "suggest.*names", "suggest.*ideas", "tagline", "motivational message", "title for",
        "write a caption", "name ideas", "write a haiku", "come up with", "one-paragraph summary",
    ],
    "data": [
        "extract", "classify this", "csv", "dataset", "outlier", "correlation", "entities",
        "spam or not", "positive, negative", "urgency", "keywords from", "pull out", "pull the",
        "sentiment", "sarcastic", "expense or income", "is spam", "standard deviation",
    ],
    "task_planning": [
        "step by step", "step-by-step", "plan a", "schedule", "instructions for", "how to set up",
        "how to install", "agenda", "based on this text", "based on this policy",
        "walk me through", "checklist", "steps to", "process to", "publish a", "revision schedule",
    ],
    "conversation": [
        "how are you", "how was your day", "recommend a movie", "feeling stressed",
        "icebreaker", "opinion on", "weekend activity", "good friend", "motivated",
        "hey", "what's up", "how's it going", "bored", "any suggestions", "just finished",
        "feeling great", "tired of", "can't decide",
    ],
    "knowledge": [
        "what is", "explain", "difference between", "how does", "why does", "why is",
        "what causes", "what does.*do", "can you tell me how", "how do.*work",
    ],
}


def _build_pattern(keyword: str) -> re.Pattern:
    """
    Turns a keyword string into a safe compiled regex pattern.
    - If it's an intentional regex pattern (contains '.*'), compile as-is.
    - Otherwise, escape all regex special characters (fixes the 'c++' bug,
      where an unescaped '+' meant "one or more of the previous char" instead
      of a literal plus sign).
    - If it's a single plain word (letters/digits only, no spaces), wrap it
      with word boundaries (\\b) so it can't match as a substring inside an
      unrelated word (fixes 'api' matching inside 'capital').
    """
    if ".*" in keyword:
        return re.compile(keyword)

    escaped = re.escape(keyword)

    if re.fullmatch(r"[a-z0-9]+", keyword):
        return re.compile(r"\b" + escaped + r"\b")

    return re.compile(escaped)


# Pre-compile all patterns once at import time (faster, and fails loudly here
# instead of silently misbehaving at classification time)
_COMPILED_KEYWORDS = {
    category: [_build_pattern(kw) for kw in keywords]
    for category, keywords in CATEGORY_KEYWORDS.items()
}


def classify_query(query: str) -> str:
    """
    Returns the best-matching category for a query, or 'other' if nothing matches.
    Case-insensitive keyword/pattern matching.
    """
    query_lower = query.lower()

    for category, patterns in _COMPILED_KEYWORDS.items():
        for pattern in patterns:
            if pattern.search(query_lower):
                return category

    return "other"


if __name__ == "__main__":
    test_queries = [
        "Write a Python function to check if a number is prime",
        "What is RAM and why does a computer need it?",
        "Solve for x: 3x + 7 = 22",
        "Translate this sentence to Hindi: The weather is nice",
        "Summarize this paragraph in 50 words",
        "Extract the email addresses from this text",
        "Give me step by step instructions to install Ollama",
        "How are you doing today?",
        "What's the capital of Australia?",
    ]

    print("=== TASK CLASSIFIER TEST ===")
    for q in test_queries:
        category = classify_query(q)
        print(f"[{category:15s}] {q}")