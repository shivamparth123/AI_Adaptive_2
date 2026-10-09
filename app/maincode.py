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

import csv
import os
import sys
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "router"))
from router import (load_profiles, route_and_generate, print_routing_explanation, MODES,
                     route_and_generate_image, print_vision_explanation,
                     generate_vision_comparison, print_vision_comparison, EXPLORATION_RATE)

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "logs", "interactive_session_log.csv")
LOG_FIELDNAMES = [
    "timestamp", "query", "mode", "category",
    "ram_available_gb", "effective_ram_gb", "stress_adjusted_ram_gb", "stress_reasons",
    "cpu_percent", "battery_percent", "battery_charging",
    "previous_model", "selected_model", "model_switch", "model_was_warm",
    "greedy_choice", "exploration_occurred",
    "predicted_quality", "predicted_latency_sec", "selected_score", "all_candidate_scores",
    "ram_settle_wait_sec", "routing_overhead_sec",
    "actual_response_time_sec", "ran_under_ram_constraint",
]


def ensure_log_file():
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=LOG_FIELDNAMES).writeheader()


def log_interaction(result: dict, previous_model: str = None):
    """
    Logs one interaction. Works for both text-routing results (which carry full
    scores_breakdown, stress info, timing) and vision results (which carry neither) -
    every field beyond the basics uses .get() with a safe default so this never
    crashes on whichever kind of result it's given.
    """
    rs = result.get("resource_state", {})
    selected = result.get("selected_model")

    breakdown = result.get("scores_breakdown", {})
    selected_breakdown = breakdown.get(selected, {})

    all_scores_str = ",".join(
        f"{m}:{b['final_score']:.3f}" for m, b in breakdown.items()
    ) if breakdown else ""

    stress_reasons = rs.get("stress_reasons", [])

    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "query": result.get("query", ""),
            "mode": result.get("mode", ""),
            "category": result.get("category", ""),
            "ram_available_gb": rs.get("ram_available_gb", ""),
            "effective_ram_gb": rs.get("effective_ram_gb", ""),
            "stress_adjusted_ram_gb": rs.get("stress_adjusted_ram_gb", ""),
            "stress_reasons": "; ".join(stress_reasons) if stress_reasons else "",
            "cpu_percent": rs.get("cpu_percent", ""),
            "battery_percent": rs.get("battery_percent", ""),
            "battery_charging": rs.get("battery_charging", ""),
            "previous_model": previous_model or "",
            "selected_model": selected,
            "model_switch": (previous_model is not None and previous_model != selected),
            "model_was_warm": result.get("selected_is_warm", ""),
            "greedy_choice": result.get("greedy_choice", ""),
            "exploration_occurred": result.get("exploration_occurred", False),
            "predicted_quality": selected_breakdown.get("quality_raw", ""),
            "predicted_latency_sec": selected_breakdown.get("latency_raw_sec", ""),
            "selected_score": selected_breakdown.get("final_score", ""),
            "all_candidate_scores": all_scores_str,
            "ram_settle_wait_sec": result.get("ram_settle_wait_sec", ""),
            "routing_overhead_sec": result.get("routing_overhead_sec", ""),
            "actual_response_time_sec": result.get("actual_response_time_sec", ""),
            "ran_under_ram_constraint": result.get("ran_under_ram_constraint", False),
        })


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

        result = route_and_generate(query, profiles, mode=current_mode, history=conversation_history)
        print_routing_explanation(result)
        log_interaction(result, previous_model=last_model_used)
        last_model_used = result.get("selected_model")

        conversation_history.append({"role": "user", "content": query})
        conversation_history.append({"role": "assistant", "content": result["response_text"]})
        max_messages = MAX_HISTORY_TURNS * 2
        if len(conversation_history) > max_messages:
            conversation_history = conversation_history[-max_messages:]


if __name__ == "__main__":
    main()