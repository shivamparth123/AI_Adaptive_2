"""
Adaptive Router (Version 1 - Rule-based classifier + weighted scoring)

This is the core of the BTP. Given a user query, it:
    1. Classifies the query into a task category (task_classifier.py)
    2. Reads current system resource state (resource_monitor.py)
    3. Applies HARD CONSTRAINTS - removes any model that genuinely can't run
       given current available RAM
    4. Scores the surviving candidates using a weighted formula combining
       quality, latency, and resource cost (model_profiles.json)
    5. Selects the highest-scoring model and builds a human-readable explanation
    6. Calls Ollama with the selected model and returns the response

Usage (standalone test):
    python router.py

Usage (from other code):
    from router import route_and_generate
    result = route_and_generate("Write a Python function to reverse a string")
"""

import json
import os
import random
import sys
import time

import psutil
from ollama import chat, ps as ollama_ps

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "resource_monitor"))
sys.path.append(os.path.dirname(__file__))  # for task_classifier in same folder

from resource_monitor import get_system_state
from task_classifier import classify_query
from hybrid_classifier import HybridClassifier

_HYBRID_CLASSIFIER = HybridClassifier()  # rules first, ML fallback (trained lazily)

# Anchored to THIS FILE's location (not the importing process's current working
# directory), so these paths resolve correctly regardless of which folder depth
# a caller imports router.py from - e.g. app/main.py (1 level under AdaptiveLocalAI2)
# and webapp/backend/app.py (2 levels under) both need this to work identically.
_ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
PROFILES_FILE = os.path.join(_ROUTER_DIR, "..", "results", "model_profiles.json")
CALIBRATION_FILE = os.path.join(_ROUTER_DIR, "..", "results", "cold_start_calibration.json")

# Live router pool - the ONLY models this router is allowed to select.
# DeepSeek-R1 8B and Qwen3-VL 4B are deliberately excluded (benchmark-only,
# per project scope - too slow / unreliable for repeated live use on this hardware).
LIVE_ROUTER_MODELS = ["llama3.2:3b", "phi4-mini:latest", "qwen3:4b", "gemma3:4b"]

# Approximate RAM footprint per model in GB (based on Ollama disk sizes with a
# buffer for runtime overhead). Used for hard constraint filtering - a model
# is excluded if there isn't enough available RAM to safely run it right now.
MODEL_RAM_REQUIREMENT_GB = {
    # Measured via `ollama ps` SIZE column (actual loaded-memory footprint), not disk
    # size estimates. This matters: the previous disk-based estimate for gemma3:4b
    # (4.3 GB) was 1.4 GB too high, which had been excluding it from the candidate
    # pool far more often than its real memory footprint warranted.
    "llama3.2:3b": 2.6,       # measured - matched the old estimate exactly
    "phi4-mini:latest": 3.1,  # measured - was 3.2, close
    "qwen3:4b": 3.2,          # measured - matched the old estimate exactly
    "gemma3:4b": 2.9,         # measured - was 4.3, SIGNIFICANT correction (-1.4 GB)
    "qwen3-vl:4b": 3.5,       # measured - was 4.3 (disk-size estimate), corrected by -0.8 GB
}

# UNITS FIX: the footprints above were read from the `ollama ps` command, which prints DECIMAL GB
# (1 GB = 1e9 bytes), but free RAM from psutil / Task Manager is in BINARY GiB (2**30 bytes). Convert once
# so both sides use the same unit (e.g. phi4-mini: 3.1 -> 2.89, matching the 2.88 GiB measured via the API).
MODEL_RAM_REQUIREMENT_GB = {m: round(v * 1e9 / 1024 ** 3, 2) for m, v in MODEL_RAM_REQUIREMENT_GB.items()}

# ---------------------------------------------------------------------------------------------
# OUTPUT BUDGETS - the maximum number of tokens a model may generate per answer.
#
# Why the CONTEXT WINDOW (num_ctx) matters as much as the output cap (num_predict): the window must hold
# the prompt AND the answer. Ollama's default window is 4096 tokens. A screenshot costs ~1200 prompt tokens, so
# with the default window a 4096-token output can never happen - the answer is cut at ~2900 tokens. And Qwen3-VL
# (the "thinking" edition) spends output tokens on hidden reasoning BEFORE the visible answer, so a small budget
# can end with reasoning only and an EMPTY answer (seen: 347 s, empty). So Qwen3-VL gets a bigger window.
#
# Cost of a bigger window: more RAM per loaded model (the KV cache grows with the window - roughly
# +0.15 MB per token for these 3-4B models, i.e. ~+0.6 GB for 4096 extra tokens).
# ---------------------------------------------------------------------------------------------
TEXT_NUM_PREDICT = 3600    # text models (blocking AND streaming - the web app and CLI use streaming)
TEXT_NUM_CTX = None        # None = Ollama's default (4096). Set e.g. 6144 if long conversations + long answers get cut
                           # (+~0.25 GB RAM per loaded text model; footprints above were measured at the default).
VISION_NUM_PREDICT = 4096  # vision models
QWEN_VL_NUM_CTX = 8192     # Qwen3-VL: prompt (~1200) + reasoning + answer (4096) must fit in the window
QWEN_VL_EXTRA_KV_GB = 0.5  # ESTIMATE of the extra RAM that window needs; measure with `ollama ps` while Qwen3-VL
                           # is loaded and replace. Set to 0 to disable. Only affects RAM warnings for vision.
MODEL_RAM_REQUIREMENT_GB["qwen3-vl:4b"] = round(MODEL_RAM_REQUIREMENT_GB["qwen3-vl:4b"] + QWEN_VL_EXTRA_KV_GB, 2)

# Scoring weights - how much each factor matters when picking a model.
# Latency is weighted more heavily than quality because benchmark data showed
# quality scores were narrowly clustered (9.2-9.97/10) across all models with
# limited discriminative power, while latency showed dramatic (3-4x) differences.
# This is a deliberate, data-informed design choice - see report methodology.
WEIGHTS = {
    "quality": 0.30,
    "latency": 0.50,
    "resource": 0.20,
}

# User-selectable modes, confirmed against experiment_baselines.py results:
#   - "fast" and "balanced" were tested there (as "speed-only" / the default weights)
#     and were found to make nearly identical decisions on this dataset (Llama/Phi
#     only - Gemma and Qwen3 were never selected under either).
#   - "quality" (100% quality) was the ONLY tested weighting where Gemma 3 4B ever
#     got selected, and only when RAM was ample (>=6GB free). This is the mode to
#     use in a demo to actually show Gemma being chosen.
MODES = {
    "fast": {"quality": 0.10, "latency": 0.70, "resource": 0.20},
    "balanced": {"quality": 0.30, "latency": 0.50, "resource": 0.20},  # = default WEIGHTS
    "quality": {"quality": 1.00, "latency": 0.00, "resource": 0.00},
}

# --------------------- Phase C: exploration mode (STRICTLY OPT-IN) ---------------------
# "explore" is NOT in MODES above - it is never selected by accident, never the default,
# and never silently changes fast/balanced/quality behavior. It must be explicitly
# requested (mode: explore). Scoring during exploration uses the SAME weights as
# "balanced" - only the final SELECTION step differs: with EXPLORATION_RATE probability,
# a non-top candidate is picked instead of the highest scorer, purely to collect real
# data on how lower-scoring models (e.g. Qwen3:4b, which never wins under greedy
# selection - see experiment_baselines.py) actually perform when they DO get used.
# This is the only legitimate way to test "does Qwen have an undiscovered niche"
# without hardcoding it to win, which earlier analysis correctly identified as
# scientifically dishonest.
EXPLORATION_RATE = 0.10
EXPLORE_SCORING_WEIGHTS = MODES["balanced"]

# Safety margin - require this much RAM headroom ABOVE the model's own
# footprint before considering it safe to run (avoids starving the OS/other apps)
# How long Ollama keeps a model loaded after it answers. This was "0s" (unload at once), which made
# EVERY query a cold start and meant the warm/cold logic could never trigger. A non-zero value lets the
# next query reuse the loaded model; get_warm_models() sees it via `ollama ps` and the router drops the
# cold-start overhead for it. Run Ollama with OLLAMA_MAX_LOADED_MODELS=1 so only one model stays resident.
# Docstrings elsewhere in this file that mention keep_alive="0s" describe the OLD behaviour.
# Override without editing code:  set ADAPTIVE_KEEP_ALIVE=0s  (always cold)  or  2m / 5m
KEEP_ALIVE = os.environ.get("ADAPTIVE_KEEP_ALIVE", "3m")
# Vision models are big and slow to load (~7 s or more), so keeping them loaded speeds up follow-up questions
# about the same image. The trade-off: ~3-3.5 GB of RAM stay in use for that long. Same default as above.
VISION_KEEP_ALIVE = os.environ.get("ADAPTIVE_VISION_KEEP_ALIVE", KEEP_ALIVE)
RAM_SAFETY_BUFFER_GB = 0.5  # lowered from 1.0 (unmeasured guess) - your sessions consistently
                            # show 3.5-4.7GB free, and the old buffer was excluding Phi-4-mini
                            # even then. Replace MODEL_RAM_REQUIREMENT_GB with real `ollama ps`
                            # SIZE values once measured, and this buffer can be revisited too.


# --------------------- Phase A: CPU/battery-aware constraints ---------------------
# Until now, hard constraints only considered RAM, even though resource_monitor.py
# already captures CPU load and battery state. These thresholds bias the EFFECTIVE
# RAM BUDGET downward under system stress, reusing the existing RAM hard-constraint
# mechanism rather than adding a second, separate decision path. This is a stated
# design heuristic, not a measured calibration - say so plainly in the report.
BATTERY_CRITICAL_PERCENT = 15      # below this, and discharging, counts as "stressed"
CPU_HIGH_PERCENT = 85              # above this counts as "stressed"
STRESS_BUDGET_REDUCTION_FACTOR = 0.7  # effective RAM budget is multiplied by this under stress


def compute_stress_adjustment(resource_state: dict) -> tuple:
    """
    Returns (adjusted_effective_ram_gb, stress_reasons: list[str]).
    Looks at battery and CPU (not RAM - RAM is already handled separately) and
    reduces the effective RAM budget used for hard constraints when the system
    is under stress, biasing selection toward lighter/faster models.
    """
    reasons = []

    battery_pct = resource_state.get("battery_percent")
    battery_charging = resource_state.get("battery_charging")
    if battery_pct is not None and battery_pct < BATTERY_CRITICAL_PERCENT and not battery_charging:
        reasons.append(f"battery critical ({battery_pct}%, not charging)")

    cpu_pct = resource_state.get("cpu_percent", 0)
    if cpu_pct > CPU_HIGH_PERCENT:
        reasons.append(f"CPU load high ({cpu_pct}%)")

    original = resource_state["effective_ram_gb"]
    if reasons:
        adjusted = round(original * STRESS_BUDGET_REDUCTION_FACTOR, 2)
    else:
        adjusted = original

    return adjusted, reasons


def load_profiles() -> dict:
    with open(PROFILES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


_calibration_warned = False  # module-level flag so the missing-calibration warning prints once, not every query


def load_calibration() -> dict:
    """
    Loads cold-start overhead per model from calibrate_cold_warm.py's output.
    Falls back to zero overhead for every model (with a one-time warning) if the
    calibration file doesn't exist yet - the router still works, just without the
    Phase B cold-start correction, equivalent to Phase A behavior.
    """
    global _calibration_warned
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {model: v["overhead_sec"] for model, v in data.items()}
    except FileNotFoundError:
        if not _calibration_warned:
            print(f"\nNOTE: No cold-start calibration found at {CALIBRATION_FILE}. "
                  f"Run calibrate_cold_warm.py once to enable cold-start-aware scoring. "
                  f"Falling back to zero overhead for now.\n")
            _calibration_warned = True
        return {model: 0.0 for model in LIVE_ROUTER_MODELS}


def wait_for_ram_to_settle(max_wait_sec: float = 6.0, interval_sec: float = 0.5,
                           tolerance_gb: float = 0.1) -> None:
    """
    After a model is unloaded (keep_alive="0s"), Windows takes ~1-2 seconds to return
    its memory. Measured with ram_diagnostic.py: 1.76 GB free at t+0s, ~4.1 GB from
    t+2s onward. Reading RAM immediately after a query therefore under-reports what
    is really free. Poll until two consecutive readings agree (or give up after
    max_wait_sec) so the hard-constraint check sees the true state.
    """
    deadline = time.time() + max_wait_sec
    previous = psutil.virtual_memory().available / (1024 ** 3)
    while time.time() < deadline:
        time.sleep(interval_sec)
        current = psutil.virtual_memory().available / (1024 ** 3)
        if abs(current - previous) <= tolerance_gb:
            return
        previous = current


def get_ollama_loaded_gb() -> float:
    """
    RAM currently held by models Ollama has loaded. This memory is RECLAIMABLE:
    when the router picks a different model, Ollama evicts the loaded one to make
    room. Without adding it back, the router would count its own previously-used
    model against every alternative and wrongly exclude them.
    """
    try:
        loaded = ollama_ps()
        total_bytes = sum((getattr(m, "size", 0) or 0) for m in loaded.models)
        return round(total_bytes / (1024 ** 3), 2)
    except Exception:
        return 0.0


def get_warm_models() -> set:
    """
    Returns the set of model names Ollama currently has loaded in memory ("warm").
    IMPORTANT CONTEXT: the original benchmark (benchmark_text.py) used Ollama's
    default keep_alive (5 min), and ran all 12 prompts per model back-to-back - so
    only the FIRST prompt per model was a true cold start; the other 11 were warm.
    The live router uses keep_alive="0s" (forces a cold start every call, so RAM
    readings stay accurate between different queries). This means the category
    latency numbers used for scoring are mostly WARM-measured, while live usage is
    always COLD - live latency can legitimately run higher than the predicted
    figure for this reason alone, independent of query difficulty.
    """
    try:
        return {m.model for m in ollama_ps().models}
    except Exception:
        return set()


def apply_hard_constraints(candidates: list, ram_available_gb: float, warm_models: set = None) -> list:
    """Remove any model that doesn't fit in currently available RAM (+ safety buffer)."""
    warm_models = warm_models or set()
    survivors = []
    for model in candidates:
        if model in warm_models:  # already in memory: needs no NEW memory, so it always "fits"
            survivors.append(model)
            continue
        required = MODEL_RAM_REQUIREMENT_GB.get(model, 999)  # unknown model = treat as unsafe
        if ram_available_gb >= required + RAM_SAFETY_BUFFER_GB:
            survivors.append(model)
    return survivors


def _normalize(values: dict, higher_is_better: bool) -> dict:
    """
    Ratio-to-best scoring: each model is scored as a fraction of the best candidate
    (best = 1.0). This replaces min-max normalization, which was distorted by outliers
    (Qwen3's 50-67s latency stretched the scale so that Llama 9.6s vs Phi 5.6s looked
    negligible) and blew up tiny differences in other factors (a 0.6 GB footprint gap
    got the full 0-1 range). Ratio scoring keeps differences proportional to their
    real size and is not affected by one bad outlier.
    """
    if higher_is_better:
        best = max(values.values())
        return {k: (v / best if best > 0 else 1.0) for k, v in values.items()}
    best = min(values.values())
    return {k: (best / v if v > 0 else 1.0) for k, v in values.items()}


def score_candidates(candidates: list, category: str, profiles: dict, weights: dict = None,
                     warm_models: set = None, calibration: dict = None) -> dict:
    """
    Scores each candidate model for the given category using the weighted formula:
        score = w_quality * quality_norm + w_latency * latency_norm + w_resource * resource_norm
    All three components are normalized 0-1 across the candidate set before weighting,
    so the raw scales (10-point quality vs seconds vs MB) don't distort the result.

    Phase B: latency used for SCORING is cold-start-adjusted, not the raw benchmark
    figure. The benchmark latency is mostly warm-measured (see earlier methodology
    note), so if a candidate is not currently warm, its calibrated cold-start overhead
    (from calibrate_cold_warm.py) is added before computing the latency ratio. This is
    also, in effect, the "model switching cost" - selecting a non-warm model always
    means a fresh load in this single-model-loaded setup.
    """
    warm_models = warm_models or set()
    calibration = calibration if calibration is not None else {}

    quality_raw, latency_raw, latency_adjusted, ram_raw = {}, {}, {}, {}

    for model in candidates:
        model_data = profiles.get(model, {})
        cat_data = model_data.get("categories", {}).get(category)

        # Resource cost uses the model's fixed footprint, NOT the benchmark RAM delta:
        # system-wide RAM deltas were noisy (some negative, caused by other processes).
        ram_raw[model] = MODEL_RAM_REQUIREMENT_GB.get(model, 5.0) * 1024

        if cat_data is None:
            # Model has no benchmark data for this specific category - fall back
            # to a neutral middling estimate rather than crashing or excluding it
            quality_raw[model] = 8.0
            latency_raw[model] = 30.0
        else:
            quality_raw[model] = cat_data["avg_quality"]
            latency_raw[model] = cat_data["avg_latency_sec"]

        cold_overhead = 0.0 if model in warm_models else calibration.get(model, 0.0)
        latency_adjusted[model] = latency_raw[model] + cold_overhead

    quality_norm = _normalize(quality_raw, higher_is_better=True)
    latency_norm = _normalize(latency_adjusted, higher_is_better=False)  # cold-adjusted, lower = better
    ram_norm = _normalize(ram_raw, higher_is_better=False)               # lower RAM = better

    final_scores = {}
    breakdown = {}
    for model in candidates:
        w = weights if weights is not None else WEIGHTS
        score = (
            w["quality"] * quality_norm[model]
            + w["latency"] * latency_norm[model]
            + w["resource"] * ram_norm[model]
        )
        final_scores[model] = round(score, 4)
        breakdown[model] = {
            "quality_raw": quality_raw[model],
            "latency_raw_sec": latency_raw[model],
            "latency_adjusted_sec": round(latency_adjusted[model], 2),
            "cold_start_overhead_applied": round(latency_adjusted[model] - latency_raw[model], 2),
            "ram_raw_mb": ram_raw[model],
            "quality_norm": round(quality_norm[model], 3),
            "latency_norm": round(latency_norm[model], 3),
            "ram_norm": round(ram_norm[model], 3),
            "final_score": final_scores[model],
        }

    return final_scores, breakdown


def select_model(query: str, profiles: dict, mode: str = "balanced",
                 history: list = None, forced_model: str = None) -> dict:
    """
    Runs the full routing decision pipeline and returns a dict with the
    selected model plus a full explanation of why - this is used both to
    actually pick the model AND to display transparency in the demo/UI.

    mode: one of "fast", "balanced", "quality" (see MODES), or the special opt-in
    "explore" mode (Phase C - see EXPLORATION_RATE above). Confirmed against
    experiment_baselines.py results - "quality" is the only mode where Gemma 3 4B
    ever gets selected, and only when enough RAM is free.
    """
    if mode == "explore":
        weights = EXPLORE_SCORING_WEIGHTS
    elif mode not in MODES:
        raise ValueError(f"Unknown mode '{mode}', choose from {list(MODES.keys()) + ['explore']}")
    else:
        weights = MODES[mode]

    settle_start = time.time()
    wait_for_ram_to_settle()  # avoid reading RAM while an unloaded model's memory is still being reclaimed
    ram_settle_wait_sec = round(time.time() - settle_start, 2)

    # Everything from here on is the actual routing DECISION - timed separately from
    # the settle wait above, since that wait is a measurement-accuracy mechanism
    # (compensating for Windows memory-reclaim lag), not routing logic itself.
    overhead_start = time.time()

    category = _HYBRID_CLASSIFIER.classify(query, history)
    resource_state = get_system_state()

    # Effective RAM = truly free RAM + RAM held by models Ollama would evict
    reclaimable_gb = get_ollama_loaded_gb()
    effective_ram_gb = round(resource_state["ram_available_gb"] + reclaimable_gb, 2)
    resource_state["ollama_loaded_gb"] = reclaimable_gb
    resource_state["effective_ram_gb"] = effective_ram_gb

    # Phase A: CPU/battery-aware budget adjustment, applied BEFORE hard constraints
    stress_adjusted_ram_gb, stress_reasons = compute_stress_adjustment(resource_state)
    resource_state["stress_reasons"] = stress_reasons
    resource_state["stress_adjusted_ram_gb"] = stress_adjusted_ram_gb

    candidates_before = LIVE_ROUTER_MODELS.copy()
    warm_models = get_warm_models()  # read first: a loaded model needs no new memory
    candidates_after = apply_hard_constraints(candidates_before, stress_adjusted_ram_gb, warm_models)

    excluded_by_constraint = [m for m in candidates_before if m not in candidates_after]

    if not candidates_after:
        # Extreme low-memory fallback - pick the smallest model regardless of
        # the constraint, better to attempt with a warning than refuse entirely
        candidates_after = ["llama3.2:3b"]
        excluded_by_constraint = [m for m in candidates_before if m != "llama3.2:3b"]
        fallback_triggered = True
    else:
        fallback_triggered = False

    # Phase B: warm detection and calibration are gathered BEFORE scoring now, since
    # the cold-start-adjusted latency needs them during scoring, not just for the
    # post-decision explanation.
    calibration = load_calibration()

    scores, breakdown = score_candidates(
        candidates_after, category, profiles, weights=weights,
        warm_models=warm_models, calibration=calibration,
    )
    greedy_choice = max(scores, key=scores.get)
    selected_model = greedy_choice
    exploration_occurred = False

    if mode == "explore" and len(candidates_after) > 1:
        if random.random() < EXPLORATION_RATE:
            other_candidates = [m for m in candidates_after if m != greedy_choice]
            selected_model = random.choice(other_candidates)
            exploration_occurred = True

    # Manual override: the user picked this model explicitly (frontend "Model" selector).
    # The router's own decision above stays in `greedy_choice` for transparency and logging.
    manual_override = bool(forced_model)
    forced_fits_ram = None
    if manual_override:
        if forced_model not in LIVE_ROUTER_MODELS:
            raise ValueError(f"Unknown model '{forced_model}', choose from {LIVE_ROUTER_MODELS}")
        forced_fits_ram = (forced_model in warm_models) or (stress_adjusted_ram_gb
                           >= MODEL_RAM_REQUIREMENT_GB.get(forced_model, 999) + RAM_SAFETY_BUFFER_GB)
        if forced_model not in breakdown:
            # score it alongside the survivors so the routing panel can still show its numbers
            scores, breakdown = score_candidates(
                candidates_after + [forced_model], category, profiles, weights=weights,
                warm_models=warm_models, calibration=calibration,
            )
        selected_model = forced_model
        exploration_occurred = False

    for model in breakdown:
        breakdown[model]["warm"] = model in warm_models
    selected_is_warm = selected_model in warm_models

    routing_overhead_sec = round(time.time() - overhead_start, 4)

    explanation = {
        "query": query,
        "ram_settle_wait_sec": ram_settle_wait_sec,
        "routing_overhead_sec": routing_overhead_sec,
        "mode": mode,
        "category": category,
        "resource_state": resource_state,
        "candidates_considered": candidates_before,
        "excluded_by_hard_constraint": excluded_by_constraint,
        "fallback_triggered": fallback_triggered,
        "scores_breakdown": breakdown,
        "selected_model": selected_model,
        "selected_is_warm": selected_is_warm,
        "greedy_choice": greedy_choice,
        "exploration_occurred": exploration_occurred,
        "selection_reason": (
            f"[mode: {mode}] Classified as '{category}'. "
            + (f"STRESS DETECTED ({', '.join(stress_reasons)}) - effective RAM budget reduced from "
               f"{resource_state['effective_ram_gb']} GB to {stress_adjusted_ram_gb} GB before "
               f"applying constraints, biasing selection toward lighter models. "
               if stress_reasons else "")
            + f"Among {len(candidates_after)} models that fit the "
            f"{'stress-adjusted' if stress_reasons else 'effective'} RAM budget "
            f"({stress_adjusted_ram_gb} GB), "
            + (f"EXPLORATION TRIGGERED ({EXPLORATION_RATE*100:.0f}% chance) - deliberately selected "
               f"'{selected_model}' instead of the highest scorer ('{greedy_choice}'), to collect real "
               f"data on a normally-passed-over model. "
               if exploration_occurred else
               f"'{selected_model}' scored highest ")
            + f"(quality={breakdown[selected_model]['quality_raw']}/10, "
            f"benchmark latency={breakdown[selected_model]['latency_raw_sec']}s, "
            + (f"cold-start-adjusted to {breakdown[selected_model]['latency_adjusted_sec']}s "
               f"[+{breakdown[selected_model]['cold_start_overhead_applied']}s calibrated overhead, "
               f"model is COLD] "
               if not selected_is_warm and breakdown[selected_model]['cold_start_overhead_applied'] > 0
               else "model is currently WARM, no cold-start overhead applied " if selected_is_warm
               else "no calibration data available yet, run calibrate_cold_warm.py ")
            + f"used for scoring) "
            f"using weights quality={weights['quality']}, latency={weights['latency']}, "
            f"resource={weights['resource']}."
        ),
    }

    explanation["manual_override"] = manual_override
    explanation["router_choice"] = greedy_choice
    if manual_override:
        need_gb = MODEL_RAM_REQUIREMENT_GB.get(forced_model, 0) + RAM_SAFETY_BUFFER_GB
        explanation["mode"] = "manual"
        explanation["scoring_mode"] = mode
        explanation["forced_model_fits_ram"] = forced_fits_ram
        explanation["selection_reason"] = (
            f"[MANUAL] You selected '{forced_model}'. "
            + (f"It fits the current RAM budget (needs ~{need_gb:.1f} GB, {stress_adjusted_ram_gb} GB available). "
               if forced_fits_ram else
               f"WARNING: it normally needs ~{need_gb:.1f} GB but only {stress_adjusted_ram_gb} GB is "
               f"available - it will still run but may be slow. ")
            + f"For reference, the router's own pick for this '{category}' query "
            f"(mode: {mode}) would have been '{greedy_choice}'."
        )

    return explanation


def _text_options() -> dict:
    """Ollama options for the TEXT models (see the OUTPUT BUDGETS note near the top)."""
    opts = {
        "num_predict": TEXT_NUM_PREDICT,  # raised over time (600 -> 3000 -> 3600): short caps truncated harder
                                          # coding answers mid-function, which on a small model can also trigger
                                          # repetition loops instead of a clean cutoff
        "temperature": 0.3,               # lower temperature reduces repetition/looping - standard
                                          # practice for code generation specifically
    }
    if TEXT_NUM_CTX:
        opts["num_ctx"] = TEXT_NUM_CTX
    return opts


def _vision_options(model: str) -> dict:
    """Ollama options for the VISION models (see the OUTPUT BUDGETS note near the top)."""
    opts = {
        "num_predict": VISION_NUM_PREDICT,  # raised from 800 - comparison testing showed Qwen3-VL's longer,
                                            # more verbose answers getting truncated
        "temperature": 0.3,
    }
    if "qwen3-vl" in model:
        # thinking edition: hidden reasoning + answer must fit, and the prompt (image ~1200 tokens) is in the window too
        opts["num_ctx"] = QWEN_VL_NUM_CTX
    return opts


def route_and_generate(query: str, profiles: dict = None, mode: str = "balanced",
                       history: list = None, forced_model: str = None) -> dict:
    """
    Full pipeline: route the query, call the selected model, return response + explanation.

    history: optional list of prior {"role": "user"/"assistant", "content": ...} messages,
    for follow-up questions. Works even when the router picks a DIFFERENT model than the
    previous turn - chat history is just data passed in the request, not something tied to
    a specific loaded model, so any model can "continue" a conversation it did not start.
    Routing/classification still only looks at the CURRENT query's text, not the history.
    """
    if profiles is None:
        profiles = load_profiles()

    routing_info = select_model(query, profiles, mode=mode, history=history, forced_model=forced_model)
    selected_model = routing_info["selected_model"]

    messages = (history or []) + [{"role": "user", "content": query}]

    start = time.time()
    response = chat(
        model=selected_model,
        messages=messages,
        options=_text_options(),
        think=False,
        keep_alive=KEEP_ALIVE,  # see KEEP_ALIVE at the top (the model stays loaded so the next query can be WARM)
    )
    elapsed = time.time() - start

    routing_info["response_text"] = response["message"]["content"]
    routing_info["actual_response_time_sec"] = round(elapsed, 2)
    routing_info["context_messages_used"] = len(history or [])

    return routing_info


def route_and_generate_stream(query: str, profiles: dict = None, mode: str = "balanced",
                              history: list = None, forced_model: str = None):
    """
    Streaming generator version of route_and_generate. Yields dicts of three kinds,
    in order, so a caller (CLI or a future web backend) can render progressively
    instead of waiting for the full response:

        {"event": "routing", "data": routing_info}
            - everything known BEFORE generation starts (classification, resource
              state, candidate scores, selection reason). Yielded once, immediately.

        {"event": "token", "text": "..."}
            - one piece of generated text. Yielded repeatedly as Ollama streams
              tokens back. Concatenating all "text" values in order reconstructs
              the full response.

        {"event": "done", "data": result}
            - yielded once, after generation finishes. `result` has the same shape
              as route_and_generate's return value (routing_info + response_text +
              actual_response_time_sec + context_messages_used), suitable for
              logging exactly like the blocking path.

    route_and_generate (blocking) is untouched - this is a separate function so
    nothing that already depends on the blocking version needs to change.
    """
    if profiles is None:
        profiles = load_profiles()

    routing_info = select_model(query, profiles, mode=mode, history=history, forced_model=forced_model)
    selected_model = routing_info["selected_model"]
    yield {"event": "routing", "data": routing_info}

    messages = (history or []) + [{"role": "user", "content": query}]

    full_text_parts = []
    start = time.time()
    stream = chat(
        model=selected_model,
        messages=messages,
        options=_text_options(),
        think=False,
        keep_alive=KEEP_ALIVE,
        stream=True,  # the only difference from route_and_generate's call - returns
                      # an iterator of partial chunks instead of one blocking response
    )
    for chunk in stream:
        piece = chunk.get("message", {}).get("content", "")
        if piece:
            full_text_parts.append(piece)
            yield {"event": "token", "text": piece}
    elapsed = time.time() - start

    result = dict(routing_info)
    result["response_text"] = "".join(full_text_parts)
    result["actual_response_time_sec"] = round(elapsed, 2)
    result["context_messages_used"] = len(history or [])
    yield {"event": "done", "data": result}


def print_routing_decision(result: dict):
    """
    Prints everything known BEFORE generation starts: classification, resource
    state, candidate scores, selection reason. Does NOT print the response or
    timing footer - those come after generation (see print_routing_explanation
    for the blocking path, or the streaming loop in main.py for the live path).
    """
    print("=" * 60)
    print(f"QUERY: {result['query']}")
    print(f"Mode: {result.get('mode', 'balanced')}")
    print(f"Category classified as: {result['category']}")
    rs = result["resource_state"]
    battery_str = ""
    if rs['battery_percent'] is not None:
        charging = "charging" if rs['battery_charging'] else "on battery"
        battery_str = f", Battery {rs['battery_percent']}% ({charging})"
    print(f"System state: RAM free {rs['ram_available_gb']} GB "
          f"+ held by loaded Ollama models {rs['ollama_loaded_gb']} GB "
          f"= effective {rs['effective_ram_gb']} GB, "
          f"CPU {rs['cpu_percent']}%{battery_str}")
    print(f"GPU: {rs.get('gpu_note', 'not tracked')}")

    if rs.get("stress_reasons"):
        print(f"STRESS DETECTED: {', '.join(rs['stress_reasons'])} "
              f"-> RAM budget reduced to {rs['stress_adjusted_ram_gb']} GB for this decision")

    if result["excluded_by_hard_constraint"]:
        print(f"Excluded by hard constraint (insufficient RAM): {result['excluded_by_hard_constraint']}")
    if result["fallback_triggered"]:
        print("WARNING: Extreme low-memory fallback triggered - forced smallest model.")

    print(f"Timing: RAM-settle wait {result.get('ram_settle_wait_sec', 0)}s "
          f"+ routing decision overhead {result.get('routing_overhead_sec', 0)}s")

    if result.get("exploration_occurred"):
        print(f"\n*** EXPLORATION MODE TRIGGERED *** "
              f"(greedy choice was '{result['greedy_choice']}', "
              f"exploring '{result['selected_model']}' instead)")

    print("\nCandidate scores:")
    for model, b in result["scores_breakdown"].items():
        marker = ""
        if model == result["selected_model"] and model == result.get("greedy_choice"):
            marker = " <== SELECTED"
        elif model == result["selected_model"]:
            marker = " <== SELECTED (EXPLORATION)"
        elif model == result.get("greedy_choice"):
            marker = " (greedy choice, not used this turn)"
        warm_tag = "[WARM]" if b.get("warm") else "[cold]"
        overhead = b.get("cold_start_overhead_applied", 0)
        latency_str = (f"latency={b['latency_raw_sec']}s"
                       + (f" -> {b['latency_adjusted_sec']}s (+{overhead}s cold-start)"
                          if overhead > 0 else ""))
        print(f"  {model:20s} {warm_tag} score={b['final_score']:.3f} "
              f"(quality={b['quality_raw']}, {latency_str}){marker}")

    print(f"\nReason: {result['selection_reason']}")

    ctx = result.get("context_messages_used", 0)
    if ctx:
        print(f"(Included {ctx} prior message(s) as conversation context)")


def print_routing_explanation(result: dict):
    """Blocking-path printer: decision, then the complete response, all at once."""
    print_routing_decision(result)
    if "response_text" in result:
        print(f"\nActual response time: {result['actual_response_time_sec']}s")
        print(f"Response:\n{result['response_text']}")
    print("=" * 60)


VISION_MODELS = ["gemma3:4b", "qwen3-vl:4b"]
DEFAULT_VISION_MODEL = "gemma3:4b"

# Version 1 vision routing is RULE-BASED, same starting point as the original text
# router (Version 1 rule-based classifier, before anything data-driven). This is
# necessary here because, unlike the text models, NEITHER vision model has been
# benchmarked yet - there is no quality/latency profile to score against. Qwen3-VL
# is documented (Ollama's model card) as strong on OCR, long documents, and
# text-in-image reading; that is the only basis for this rule, not measured data.
# Use "imagecompare:" (see generate_vision_comparison below) to build real
# comparative data over actual usage, which can replace this heuristic later.
VISION_OCR_KEYWORDS = [
    "text in", "read the", "extract text", "extract the text", "ocr",
    "document", "transcribe", "what does it say", "what's written",
    "words in", "written on", "read this", "what text",
]


def choose_vision_model(question: str) -> tuple:
    """
    Returns (model, reason). Rule-based Version 1: OCR/document-reading phrasing
    routes to Qwen3-VL (per its documented OCR strength), everything else defaults
    to Gemma 3 4B (the model this project has already tested most for general
    image description).
    """
    q_lower = question.lower()
    for kw in VISION_OCR_KEYWORDS:
        if kw in q_lower:
            return "qwen3-vl:4b", (
                f"Question phrasing matched OCR/document-reading pattern ('{kw}') - "
                f"routed to qwen3-vl:4b, documented as stronger for text-in-image/OCR tasks. "
                f"This is a Version 1 RULE, not based on measured comparison yet."
            )
    return DEFAULT_VISION_MODEL, (
        "General image question (no OCR/document keywords matched) - defaulted to "
        "gemma3:4b. This is a Version 1 RULE, not based on measured comparison yet."
    )


def _check_vision_resources(model: str) -> dict:
    wait_for_ram_to_settle()
    resource_state = get_system_state()
    reclaimable_gb = get_ollama_loaded_gb()
    effective_ram_gb = round(resource_state["ram_available_gb"] + reclaimable_gb, 2)
    resource_state["ollama_loaded_gb"] = reclaimable_gb
    resource_state["effective_ram_gb"] = effective_ram_gb
    required_gb = MODEL_RAM_REQUIREMENT_GB[model] + RAM_SAFETY_BUFFER_GB
    resource_state["fits"] = (effective_ram_gb >= required_gb) or (model in get_warm_models())
    resource_state["required_gb"] = required_gb
    return resource_state


def _vision_messages(image_path: str, question: str, history: list = None) -> list:
    """
    Chat messages for a vision call. With earlier Q&A about the SAME image, the image is attached
    to the FIRST user message and the new question is appended, so the model sees the whole
    exchange. Routing (which vision model) still looks only at the NEW question.
    """
    if history and history[0].get("role") == "user":
        messages = [dict(m) for m in history]
        messages[0]["images"] = [image_path]
        messages.append({"role": "user", "content": question})
        return messages
    return [{"role": "user", "content": question, "images": [image_path]}]


def _call_vision_model(model: str, image_path: str, question: str, history: list = None) -> dict:
    """Low-level single call - used by both route_and_generate_image and the comparison mode."""
    start = time.time()
    response = chat(
        model=model,
        messages=_vision_messages(image_path, question, history),
        options=_vision_options(model),
        think=False,
        keep_alive=VISION_KEEP_ALIVE,
    )
    elapsed = round(time.time() - start, 2)
    msg = response["message"]
    try:
        thinking = msg.get("thinking") or ""  # Qwen-family models put their hidden reasoning here
    except Exception:
        thinking = ""
    try:
        done_reason, eval_count = response.get("done_reason"), response.get("eval_count")
    except Exception:
        done_reason, eval_count = None, None
    return {"response_text": msg["content"], "actual_response_time_sec": elapsed,
            "done_reason": done_reason, "eval_count": eval_count, "thinking_chars": len(thinking)}


def route_and_generate_image(image_path: str, question: str, profiles: dict = None,
                             history: list = None) -> dict:
    """
    Handles a query that includes an image, choosing between Gemma 3 4B and
    Qwen3-VL 4B using the Version 1 rule-based heuristic (choose_vision_model).

    Unlike the text router, there is no benchmark-derived quality/latency profile
    to score against yet, so this is rule-based, not scored. Resource handling
    mirrors the single-model version: insufficient RAM produces a WARNING and the
    query proceeds anyway (no smaller vision fallback exists), with the outcome
    recorded via ran_under_ram_constraint.
    """
    if not os.path.exists(image_path):
        return {
            "query": question, "image_path": image_path, "mode": "vision",
            "category": "image_understanding", "selected_model": None,
            "selection_reason": f"Image file not found at: {image_path}",
            "response_text": None,
        }

    chosen_model, routing_reason = choose_vision_model(question)
    rs = _check_vision_resources(chosen_model)
    was_warm = chosen_model in get_warm_models()  # already loaded? (needs VISION_KEEP_ALIVE > 0)
    fits = rs["fits"]

    result = {
        "query": question,
        "image_path": image_path,
        "mode": "vision",
        "category": "image_understanding",
        "resource_state": rs,
        "selected_model": chosen_model,
        "fits_in_ram": fits,
        "ran_under_ram_constraint": not fits,
    }

    if not fits:
        print(
            f"\nWARNING: {chosen_model} normally requires ~{rs['required_gb']:.1f} GB effective "
            f"RAM; only {rs['effective_ram_gb']} GB is currently available. There is no smaller "
            f"vision model to fall back to, so this will proceed anyway - it may be noticeably "
            f"slower than usual while the system compensates."
        )

    gen = _call_vision_model(chosen_model, image_path, question, history)
    fallback_used = False

    # Empty-response fallback: real comparison testing found Qwen3-VL occasionally returns
    # NOTHING on dense-text/OCR images - the exact use case the OCR rule routes it to -
    # likely the same unsuppressable internal-reasoning behavior observed earlier when it
    # was tried as the quality judge (see quality_evaluation.py). Rather than show the user
    # an empty answer, fall back to the other vision model once.
    if (not gen["response_text"] or not gen["response_text"].strip()) and chosen_model != DEFAULT_VISION_MODEL:
        print(f"\nNOTE: {chosen_model} returned an empty response - falling back to {DEFAULT_VISION_MODEL}.")
        fallback_used = True
        chosen_model = DEFAULT_VISION_MODEL
        rs = _check_vision_resources(chosen_model)
        was_warm = chosen_model in get_warm_models()
        fits = rs["fits"]
        result["resource_state"] = rs
        result["fits_in_ram"] = fits
        result["ran_under_ram_constraint"] = not fits
        gen = _call_vision_model(chosen_model, image_path, question, history)

    result["selected_model"] = chosen_model
    result["selected_is_warm"] = was_warm
    result["vision_fallback_used"] = fallback_used
    result.update(gen)

    fallback_note = (
        f" NOTE: the originally chosen model returned an empty response and this fell back "
        f"to {chosen_model}." if fallback_used else ""
    )
    if fits:
        ram_note = (
            f"Effective RAM ({rs['effective_ram_gb']} GB) was sufficient "
            f"(required ~{rs['required_gb']:.1f} GB)."
        )
    else:
        ram_note = (
            f"Ran despite insufficient effective RAM ({rs['effective_ram_gb']} GB vs "
            f"~{rs['required_gb']:.1f} GB required) - no smaller vision model was available as fallback."
        )
    ram_note += fallback_note
    warm_text = "WARM (already loaded, no load time)" if was_warm else "COLD (had to be loaded first)"
    result["selection_reason"] = f"{routing_reason} {ram_note} Model was {warm_text}."
    return result


def generate_vision_comparison(image_path: str, question: str) -> dict:
    """
    Runs BOTH vision models on the same image+question and returns both results
    side by side. Use this deliberately (via the 'imagecompare:' command) to build
    real comparative data over actual usage - the benchmark comparison that was
    never collected for vision models in the original benchmark run.
    """
    if not os.path.exists(image_path):
        return {"query": question, "image_path": image_path, "error": f"Image file not found at: {image_path}"}

    results = {}
    for model in VISION_MODELS:
        rs = _check_vision_resources(model)
        if not rs["fits"]:
            print(f"\nWARNING: {model} normally requires ~{rs['required_gb']:.1f} GB effective RAM; "
                  f"only {rs['effective_ram_gb']} GB available. Running anyway for comparison.")
        gen = _call_vision_model(model, image_path, question)
        results[model] = {
            "response_text": gen["response_text"],
            "actual_response_time_sec": gen["actual_response_time_sec"],
            "done_reason": gen.get("done_reason"),
            "thinking_chars": gen.get("thinking_chars", 0),
            "ram_available_gb": rs["ram_available_gb"],
            "effective_ram_gb": rs["effective_ram_gb"],
            "ran_under_ram_constraint": not rs["fits"],
        }

    return {
        "query": question,
        "image_path": image_path,
        "mode": "vision_comparison",
        "category": "image_understanding",
        "results_by_model": results,
    }


def print_vision_explanation(result: dict):
    print("=" * 60)
    print(f"IMAGE: {result['image_path']}")
    print(f"QUESTION: {result['query']}")
    if result.get("selected_model") is None:
        print(f"\n{result['selection_reason']}")
        print("=" * 60)
        return

    rs = result["resource_state"]
    print(f"Selected vision model: {result['selected_model']}")
    print(f"System state: RAM free {rs['ram_available_gb']} GB "
          f"+ held by loaded Ollama models {rs['ollama_loaded_gb']} GB "
          f"= effective {rs['effective_ram_gb']} GB")
    print(f"Fits in RAM: {result['fits_in_ram']}"
          + (" (ran anyway - no fallback vision model exists)" if result.get("ran_under_ram_constraint") else ""))
    print(f"\nReason: {result['selection_reason']}")

    if result.get("response_text"):
        print(f"\nActual response time: {result['actual_response_time_sec']}s")
        print(f"Response:\n{result['response_text']}")
    print("=" * 60)


def print_vision_comparison(result: dict):
    print("=" * 60)
    print(f"IMAGE: {result['image_path']}")
    print(f"QUESTION: {result['query']}")
    print("=" * 60)
    for model, r in result["results_by_model"].items():
        constraint_note = " (ran under RAM constraint)" if r["ran_under_ram_constraint"] else ""
        print(f"\n--- {model} --- ({r['actual_response_time_sec']}s{constraint_note}, "
              f"effective RAM at call time: {r['effective_ram_gb']} GB)")
        print(r["response_text"])
    print("\n" + "=" * 60)


if __name__ == "__main__":
    profiles = load_profiles()

    test_queries = [
        "Write a Python function to check if a number is prime",
        "What is RAM and why does a computer need it?",
        "Translate this sentence to Hindi: The weather is nice today",
        "How are you doing today?",
        "Extract the email addresses from this text: contact us at a@x.com or b@y.org",
        "Summarize this paragraph in one sentence: Local AI runs models on your own device, which protects privacy and removes cloud costs.",
        "Solve for x: 3x + 7 = 22",
        "Give me step by step instructions to create a Python virtual environment",
    ]

    for q in test_queries:
        result = route_and_generate(q, profiles)
        print_routing_explanation(result)
        print()