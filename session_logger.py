"""
Shared Session Logger

Extracted from main.py so that BOTH the CLI (main.py) and the web backend
(webapp/backend/app.py) log interactions to the SAME file with the SAME schema.
Without this, usage through the web app would be invisible in session history -
a real gap, since the two interfaces would otherwise silently drift apart.

Path is anchored to this file's own location (not the importing process's cwd),
same fix applied to router.py - necessary because this module is imported from
multiple folder depths: app/ (1 level under AdaptiveLocalAI2) and
webapp/backend/ (2 levels under).
"""

import csv
import os
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(_THIS_DIR, "logs", "interactive_session_log.csv")

LOG_FIELDNAMES = [
    "timestamp", "source", "query", "mode", "category",
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


def log_interaction(result: dict, previous_model: str = None, source: str = "cli"):
    """
    Logs one interaction. Works for both text-routing results (full scores_breakdown,
    stress info, timing) and vision results (neither) - every field beyond the basics
    uses .get() with a safe default so this never crashes on whichever kind of result
    it's given.

    source: "cli" or "web" - lets you later filter/compare usage by interface.
    """
    ensure_log_file()

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
            "source": source,
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