"""
Task Classifier (Version 1 - Rule-based)
Classifies a user's query into one of the 9 categories used in benchmarking,
so the router can look up task-specific model profiles for that category.

Mirrors the categories in dataset/prompts.py exactly, so the output can be used
directly as a lookup key into model_profiles.json.

Revision notes (from real session-log failures, Oct 2026):
  1. Single-word keywords now also match simple inflections (s/es/d/ed/ing), so
     "equations", "sorting", "debugged" match "equation", "sort", "debug".
     Previously the strict word boundary missed every plural/inflected form.
  2. Added keywords: programming ("programming", "coding"), reasoning
     ("calculation"), knowledge ("define", "definition of", "describe",
     "tell me about", "what are", "meaning of").
  3. classify_with_context(): a FOLLOW-UP message that matches nothing ("continue",
     "yes and numerical", "what about X") inherits the category of the most recent
     earlier user message that had a real category. Short standalone questions
     ("who wrote hamlet") do not inherit - see is_followup().

Usage:
    from task_classifier import classify_query, classify_with_context
"""
import re
CATEGORY_KEYWORDS = {
    "programming": [
        "code", "python", "function", "script", "program", "debug", "bug", "error in this",
        "c++", "java", "sql", "algorithm", "loop", "variable", "class ", "def ", "compile",
        "syntax", "regex", "api", "programming", "coding", "implement a", "write a program", "sort", "array", "linked list",
        "reverse a string", "reverse this", "merge two", "convert this list", "parse", "iterate",
        "for loop", "while loop", "index error", "crashing", "bash", "javascript", "typescript",
    ],
    "reasoning": [
        "solve", "prove", "derivative", "equation", "calculate", "calculation", "math", "logic", "if.*then",
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
        "define", "definition of", "describe", "tell me about", "what are", "meaning of",
    ],
}

def _build_pattern(keyword: str) -> re.Pattern:
    if ".*" in keyword:
        return re.compile(keyword)
    escaped = re.escape(keyword)
    if re.fullmatch(r"[a-z0-9]+", keyword):
        return re.compile(r"\b" + escaped + r"(?:s|es|d|ed|ing)?\b")
    return re.compile(escaped)

_COMPILED_KEYWORDS = {
    category: [_build_pattern(kw) for kw in keywords]
    for category, keywords in CATEGORY_KEYWORDS.items()
}

def classify_query(query: str) -> str:
    query_lower = query.lower()
    for category, patterns in _COMPILED_KEYWORDS.items():
        for pattern in patterns:
            if pattern.search(query_lower):
                return category
    return "other"


FOLLOWUP_MAX_WORDS = 8  # a follow-up cue only counts in a short message
_FOLLOWUP_CUES = re.compile(
    r"^(continue|go on|more|yes|yeah|yep|ok|okay|and|also|what about|how about|then|next|elaborate|expand)\b"
)


def is_followup(query: str) -> bool:
    """
    True if the message looks like a continuation of the previous turn rather than a
    new standalone question: very short (1-2 words, e.g. "continue", "more"), or a
    short message that OPENS with a follow-up cue ("yes and numerical",
    "what about critical point"). A short standalone question such as
    "who wrote hamlet" is NOT a follow-up, so it never inherits the previous topic.
    """
    q = query.lower().strip()
    n = len(q.split())
    if n <= 2:
        return True
    return n <= FOLLOWUP_MAX_WORDS and bool(_FOLLOWUP_CUES.match(q))


def classify_with_context(query: str, history: list = None) -> str:
    """
    Like classify_query, but a FOLLOW-UP message (see is_followup) that matches no
    category inherits the category of the most recent earlier USER message that had a
    real one. Anything that already matches a category keeps it.
    """
    category = classify_query(query)
    if category != "other" or not history or not is_followup(query):
        return category
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        earlier = classify_query(msg.get("content", ""))
        if earlier != "other":
            return earlier
    return category