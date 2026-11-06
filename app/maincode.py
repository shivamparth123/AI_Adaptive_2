"""
Interactive Adaptive AI Application (CLI)

Type queries one after another and watch the router pick a model for each one,
live - based on the query's category and current system resources at that moment.

This is the first real "application" milestone: everything built so far
(classifier, resource monitor, model profiles, router) wired into something you
can actually sit down and use, and demo.

Usage:
    python main.py

Commands while running:
    type any query          -> routes it, shows the decision, shows the response
    "image: <path> :: <question>"  -> sends an image + question to a vision model.
                                       Auto-picks between Gemma 3 4B (general images)
                                       and Qwen3-VL 4B (OCR/document/text-reading
                                       questions) using a rule-based heuristic - see
                                       choose_vision_model() in router.py. Example:
                                       image: C:\photos\diagram.png :: What does this show?
    "image: <question>"      -> (no path, no "::") asks a FOLLOW-UP question about the
                                MOST RECENTLY shared image, without retyping its path.
    "imagecompare: <path> :: <question>"  -> runs BOTH Gemma and Qwen3-VL on the same
                                image+question and shows both answers side by side,
                                with timing for each. Use this to build real comparison
                                data between the two vision models.
    "clear image"             -> forgets the remembered image, so "image: <question>"
                                will require a path again
    (automatic) text follow-up questions -> the last few exchanges are kept as
                                conversation context, so "explain the second point
                                further" works even if a DIFFERENT model answers
                                this turn than answered the previous one
    "clear context"            -> forgets the conversation context used for follow-ups
                                (image memory is separate and unaffected by this; this
                                is NOT the same as "history", which shows session stats)
    "mode: fast"             -> switch to speed-weighted routing
    "mode: balanced"         -> switch to the default weighting (quality 30/latency 50/resource 20)
    "mode: quality"          -> switch to quality-only routing (the only mode where
                                Gemma 3 4B was ever selected in testing, and only
                                when enough RAM is free - see experiment_baselines.py)
    "mode: explore"          -> OPT-IN ONLY. Scores exactly like balanced mode, but
                                ~10% of queries deliberately select a non-top model
                                instead, to collect real data on models (e.g. Qwen3:4b)
                                that normally never win - the honest way to test for
                                an undiscovered niche without hardcoding a winner.
    "history"                -> shows a summary of this session's model selections
    "quit" or "exit"         -> ends the session, saves the session log
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "router"))
from router import (load_profiles, route_and_generate, route_and_generate_stream,
                     print_routing_explanation, print_routing_decision, MODES,
                     route_and_generate_image, print_vision_explanation,
                     generate_vision_comparison, print_vision_comparison, EXPLORATION_RATE)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))  # for session_logger at project root
from session_logger import LOG_FILE, ensure_log_file, log_interaction as _shared_log_interaction


def log_interaction(result: dict, previous_model: str = None):
    """CLI wrapper around the shared logger - always tags rows with source='cli'."""
    _shared_log_interaction(result, previous_model=previous_model, source="cli")


def print_session_history():
    if not os.path.exists(LOG_FILE):
        print("No queries logged yet this session or in previous sessions.")
        return

    import pandas as pd
    df = pd.read_csv(LOG_FILE)
    if len(df) == 0:
        print("No queries logged yet.")
        return

    print(f"\n=== SESSION HISTORY ({len(df)} total queries logged) ===")
    print("\nModel selection frequency:")
    print(df["selected_model"].value_counts().to_string())
    print("\nCategory frequency:")
    print(df["category"].value_counts().to_string())
    print(f"\nAverage response time: {df['actual_response_time_sec'].mean():.2f}s")
    print(f"Last 5 queries:")
    print(df[["timestamp", "query", "category", "selected_model"]].tail(5).to_string(index=False))


def main():
    print("=" * 60)
    print("  ADAPTIVE LOCAL AI - Interactive Session")
    print("=" * 60)
    print("Type a query and press Enter.")
    print(f"Modes available: {', '.join(MODES.keys())}, explore (opt-in) - switch with 'mode: <name>'")
    print("Type 'history' to see this session's model selections, or 'quit' / 'exit' to stop.\n")

    ensure_log_file()
    profiles = load_profiles()  # loaded once - the model profiles don't change during a session
    current_mode = "balanced"
    last_image_path = None  # remembers the most recently shared image for follow-up questions
    conversation_history = []  # recent {"role", "content"} pairs, for text follow-up questions
    MAX_HISTORY_TURNS = 3  # keep last 3 exchanges (6 messages) - bounds context size on CPU
    last_model_used = None  # tracks the most recently selected model (text OR vision), for
                            # model-switch logging - a switch from a text model to a vision
                            # model is still a real reload event, so this is tracked uniformly

    while True:
        try:
            query = input(f"\n[{current_mode}] You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended.")
            break

        if not query:
            continue

        if query.lower() in ("quit", "exit"):
            print("Session ended.")
            break

        if query.lower() == "history":
            print_session_history()
            continue

        if query.lower() == "clear context":
            turns = len(conversation_history) // 2
            conversation_history = []
            print(f"Cleared conversation context ({turns} exchange(s) forgotten).")
            continue

        if query.lower() == "clear image":
            if last_image_path is None:
                print("No image was being remembered.")
            else:
                print(f"Cleared remembered image: {last_image_path}")
            last_image_path = None
            continue

        if query.lower().startswith("mode:"):
            requested = query.split(":", 1)[1].strip().lower()
            # "explore" is deliberately NOT in MODES (see router.py) so it can never be
            # picked by accident elsewhere in the code - it's allowed here ONLY because
            # the user explicitly typed it.
            valid_modes = list(MODES.keys()) + ["explore"]
            if requested in valid_modes:
                current_mode = requested
                if requested == "explore":
                    print(f"Mode switched to 'explore' - {int(EXPLORATION_RATE*100)}% of queries will "
                          f"deliberately try a non-optimal model to collect real performance data.")
                else:
                    print(f"Mode switched to '{current_mode}'.")
            else:
                print(f"Unknown mode '{requested}'. Available: {', '.join(valid_modes)}")
            continue

        if query.lower().startswith("imagecompare:"):
            body = query.split(":", 1)[1].strip()
            if "::" not in body:
                print("Format: imagecompare: <path> :: <question>")
                continue
            image_path, question = (part.strip() for part in body.split("::", 1))
            last_image_path = image_path
            result = generate_vision_comparison(image_path, question)
            if result.get("error"):
                print(result["error"])
                continue
            print_vision_comparison(result)
            # log one row per model so the session CSV captures both for later comparison
            for model, r in result["results_by_model"].items():
                log_interaction({
                    "query": question, "mode": "vision_comparison", "category": "image_understanding",
                    "resource_state": {"ram_available_gb": r["ram_available_gb"],
                                        "effective_ram_gb": r["effective_ram_gb"]},
                    "selected_model": model,
                    "actual_response_time_sec": r["actual_response_time_sec"],
                    "ran_under_ram_constraint": r["ran_under_ram_constraint"],
                }, previous_model=last_model_used)
                last_model_used = model
            continue

        if query.lower().startswith("image:"):
            body = query.split(":", 1)[1].strip()

            if "::" in body:
                # New image explicitly given - path :: question
                image_path, question = (part.strip() for part in body.split("::", 1))
                last_image_path = image_path
            else:
                # No path given - treat as a follow-up about the last remembered image
                if last_image_path is None:
                    print("No image is currently remembered. Use: image: <path> :: <question>")
                    continue
                image_path = last_image_path
                question = body
                print(f"(Reusing last image: {image_path})")

            result = route_and_generate_image(image_path, question)
            print_vision_explanation(result)
            if result.get("response_text") is not None:
                log_interaction(result, previous_model=last_model_used)
                last_model_used = result.get("selected_model")
            continue

        result = None
        for item in route_and_generate_stream(query, profiles, mode=current_mode,
                                               history=conversation_history):
            if item["event"] == "routing":
                print_routing_decision(item["data"])
                print("\nResponse:")
            elif item["event"] == "token":
                print(item["text"], end="", flush=True)
            elif item["event"] == "done":
                result = item["data"]
                print(f"\n\nActual response time: {result['actual_response_time_sec']}s")
                print("=" * 60)

        log_interaction(result, previous_model=last_model_used)
        last_model_used = result.get("selected_model")

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": result["response_text"]})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(conversation_history) > max_messages:
            conversation_history = conversation_history[-max_messages:]


if __name__ == "__main__":
    main()