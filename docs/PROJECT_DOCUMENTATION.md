# Adaptive Local AI - Complete Project Documentation

**Project:** Hardware-aware adaptive routing of local LLMs (BTech final-year project, SK2 "Local AI")
**Location:** `D:\AdaptiveLocalAI2\`  |  **Documented:** October 2026
**Platform:** Windows 11, PowerShell, Python virtual environment, Ollama (CPU-only inference)

> This document describes everything that was built, how it works, how to run it, what data and results it produced,
> where every file the system generates is stored (and how to clean it up), and what its limits are.
> To install the project on a new PC, see `README.md`.

---

## Table of contents
1. [What the project is](#1-what-the-project-is)
2. [Environment and reproducibility](#2-environment-and-reproducibility)
3. [Models](#3-models)
4. [System architecture](#4-system-architecture)
5. [Workflows step by step](#5-workflows-step-by-step)
6. [Folder and file reference](#6-folder-and-file-reference)
7. [Datasets](#7-datasets)
8. [Experiments, data and results](#8-experiments-data-and-results)
9. [Configuration reference](#9-configuration-reference)
10. [How to install and run](#10-how-to-install-and-run)
11. [Web application guide](#11-web-application-guide)
12. [Command-line (CLI) guide](#12-command-line-cli-guide)
13. [Logging: the session CSV](#13-logging-the-session-csv)
14. [Where data is stored and how to clean it up](#14-where-data-is-stored-and-how-to-clean-it-up)
15. [Testing and verification](#15-testing-and-verification)
16. [Troubleshooting](#16-troubleshooting)
17. [Limitations and future work](#17-limitations-and-future-work)
18. [Development history and patch order](#18-development-history-and-patch-order)
19. [Using this document for the report](#19-using-this-document-for-the-report)
20. [Glossary](#20-glossary)

---

## 1. What the project is

### 1.1 Problem
Running large language models locally is attractive (privacy, no cloud cost) but a laptop has limited, shared resources.
One fixed model is a poor fit: a small model is fast but weaker on some tasks, a bigger one is slower and may not even
fit in the memory left over by other programs. Which model is best also depends on the *kind* of question.

### 1.2 Solution
An **adaptive router** that, for every query, chooses one of several local models by combining:

| Signal | How it is used |
|---|---|
| **Task category** (9 classes) | Looks up each model's measured quality and latency *for that category* |
| **Model quality** | LLM-as-judge scores from a 108-prompt benchmark |
| **Model latency** | Measured benchmark latency, plus a calibrated cold-start overhead when the model is not loaded |
| **Free RAM** | A hard constraint: a model that would have to be loaded must fit in available memory |
| **CPU load / battery** | Under stress the usable RAM budget is cut to 70% so lighter models are preferred |
| **Warm / cold state** | A model already loaded in memory needs no load time and no new RAM |
| **User mode** | `fast`, `balanced`, `quality` weightings (plus opt-in `explore` and manual model choice) |

It is delivered as (a) a command-line application and (b) a browser-based web application, both using the same engine
and writing to the same log.

### 1.3 Scope decisions (state these in the report)
- **Image *understanding*, not image generation.** Generating images is not feasible on this CPU-only 16 GB laptop with
  the available model pool, so the image feature answers questions about an uploaded picture. The supervisor's wording
  ("coherent text/image output automation") was never explicitly confirmed - tell the supervisor of this decision.
- **Quantization was not varied.** Ollama's default quantized builds are used throughout; routing, not quantization, is the contribution.
- **Single user, single machine, CPU-only inference** (the integrated Radeon GPU is not used by Ollama on this hardware).

### 1.4 Headline results (details in section 8)
- No single model is best for every category; Qwen3:4b is dominated in 9 of 9 categories.
- The router's decisions are stable under reasonable weight changes (identical selections for 0.3/0.5 vs 0.5/0.3).
- Hybrid classifier: **79.7%** on 64 clean held-out prompts vs 56.2% for keyword rules alone.
- A classifier mistake changes the chosen model but costs only about **+1.1 s** of predicted latency on average.
- Warm-model awareness changes the choice in 7 of 36 (category x loaded-model) cases in balanced/fast mode.
- Real-machine diagnosis: the free-RAM measurement is exact; the laptop is genuinely memory-tight (typically ~2-3 GB available at idle).

---

## 2. Environment and reproducibility

### 2.1 Hardware
| Item | Value |
|---|---|
| CPU | AMD Ryzen 7 5800H |
| RAM | 16 GB DDR4-3200 (15.34 GiB usable; 674 MB hardware-reserved) |
| GPU | AMD Radeon integrated - **not used** by Ollama (its startup log: "dropping integrated GPU") |
| Storage | NVMe SSD (C: and D:) |
| OS | Windows 11 |

### 2.2 Software (as reported by the machine)
| Component | Version / note |
|---|---|
| Ollama | **0.33.2** (`ollama --version`) |
| Models in Ollama | see the table in section 3.1 (`ollama list`) |
| Python | **3.11** (the compiled cache files are `cpython-311`; confirm the exact patch version with `python --version`) |
| Python packages | `requirements.txt` |
| Key packages | `fastapi`, `uvicorn[standard]`, `python-multipart`, `pandas`, `scikit-learn`, `psutil`, `ollama` (Python client) |
| Browser front end | plain HTML/CSS/JavaScript; `marked` 12.0.0 (Markdown) and `KaTeX` 0.16.9 (math) loaded from the jsDelivr CDN |

### 2.3 Settings that change behaviour (record the values you used)
| Setting | Value used | Effect |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | `1` (User environment variable, confirmed live) | Only one model stays in RAM at a time |
| `OLLAMA_MODELS` | `D:\Ollama\models` | Where Ollama stores downloaded models |
| Router `KEEP_ALIVE` | `3m` | Text models stay loaded 3 minutes after answering |
| Router `VISION_KEEP_ALIVE` | `3m` (same as text) | Vision models stay loaded 3 minutes |
| Token budgets | text 3600, vision 4096, Qwen3-VL context 8192 | See section 9 |

---

## 3. Models

### 3.1 Installed models (`ollama list`, 19 Oct 2026)
| Name | ID | Disk size | Role in the project |
|---|---|---|---|
| `llama3.2:3b` | a80c4f17acd5 | 2.0 GB | **Live text router pool** |
| `llama3.2:latest` | a80c4f17acd5 | 2.0 GB | Same model as `llama3.2:3b` (identical ID - an extra tag, no extra disk) |
| `phi4-mini:latest` | 78fad5d182a7 | 2.5 GB | **Live text router pool** |
| `qwen3:4b` | 359d7dd4bcda | 2.5 GB | **Live text router pool** (never wins under normal routing; used by `explore`/manual) |
| `gemma3:4b` | a2af6cc3eb7f | 3.3 GB | **Live text pool + vision** (default vision model) |
| `qwen3-vl:4b` | 1343d82ebee3 | 3.3 GB | **Vision only** (OCR/document questions) |
| `deepseek-r1:8b` | 6995872bfe4c | 5.2 GB | **Benchmark only** - too slow, returns empty output on some prompts |

### 3.2 Roles
- **Text router pool (4 models):** llama3.2:3b, phi4-mini, qwen3:4b, gemma3:4b.
- **Vision pool (2 models):** gemma3:4b (general images) and qwen3-vl:4b (chosen when the question contains OCR-style words).
- **Judge:** Gemma 3 4B scored answer quality in the benchmark (Qwen3-VL failed as judge - empty output).

### 3.3 Memory footprint (important detail: units)
Footprints were first read from the `ollama ps` **command**, which prints *decimal* gigabytes (1 GB = 10^9 bytes). Free RAM
from `psutil`/Task Manager is in *binary* gibibytes (1 GiB = 2^30 bytes). Comparing them overstated every model by ~7.4%.
Evidence: the CLI printed phi4-mini as "3.1 GB" while the Python API reported 2.88 for the same loaded model.
`router.py` now converts once at import.

| Model | `ollama ps` (decimal GB) | In GiB (used by the router) | RAM budget needed to *load* it (+0.5 GB safety buffer) |
|---|---|---|---|
| llama3.2:3b | 2.6 | 2.42 | 2.92 |
| gemma3:4b | 2.9 | 2.70 | 3.20 |
| phi4-mini | 3.1 | 2.89 | 3.39 |
| qwen3:4b | 3.2 | 2.98 | 3.48 |
| qwen3-vl:4b | 3.5 | 3.26 (+0.5 estimated KV cache for its larger context window = 3.76) | 4.26 |

All footprints were measured at Ollama's default context window of 4096 tokens.

### 3.4 Cold-start calibration (10 repetitions per model)
Time to load a model, measured as cold-run time minus warm-run time.

| Model | Mean overhead (used) | Std dev | Median | Min | Max | Mean cold | Mean warm |
|---|---|---|---|---|---|---|---|
| llama3.2:3b | **4.80 s** | 1.18 | 4.36 | 4.24 | 8.09 | 5.26 | 0.46 |
| phi4-mini | **4.76 s** | 0.79 | 4.44 | 3.90 | 5.90 | 5.53 | 0.78 |
| qwen3:4b | **4.41 s** | 0.55 | 4.35 | 3.65 | 5.70 | 5.79 | 1.38 |
| gemma3:4b | **7.40 s** | 0.43 | 7.39 | 6.80 | 8.16 | 7.75 | 0.35 |

The first repetition of each model is a slow outlier (llama 8.09 s vs median 4.36 s), which inflates the mean. The mean
is what the router uses; state this in the report. Saved in `results/cold_start_calibration.json` and
`results/cold_warm_calibration_samples.csv`.

---

## 4. System architecture

### 4.1 Component diagram
```
                         +---------------------------+
   Browser (web UI)      |  webapp/frontend           |   Command line
   index.html            |  index.html style.css      |   app/maincode.py
   style.css  app.js ----+  app.js  (localStorage)    |        |
        |                +-------------+-------------+        |
        | HTTP (JSON, SSE stream)      |                      | direct Python import
        v                              |                      v
+--------------------------------------+----------------------------------+
|  webapp/backend/app.py  (FastAPI, served on 127.0.0.1:8000)              |
|  /api/status  /api/chat/stream  /api/image/*  /api/history   + static    |
+-----------------------------------+--------------------------------------+
                                    |  imports (same code as the CLI)
                                    v
+----------------------------------------------------------------------------+
|  router/router.py  - THE ENGINE                                             |
|   select_model()  -> classify -> read resources -> hard constraints         |
|                      -> score -> choose -> explain                          |
|   route_and_generate_stream()   route_and_generate_image()                  |
|   generate_vision_comparison()                                              |
+---+-----------------+-------------------+------------------+---------------+
    |                 |                   |                  |
    v                 v                   v                  v
 task_classifier   hybrid_classifier   resource_monitor    results/*.json
 (keyword rules)   (rules + SVM)       (psutil: RAM/CPU/   (model_profiles,
                                        battery)            cold_start_calibration)
                                    |
                                    v
                         Ollama server (localhost:11434)  <- models in D:\Ollama\models
                                    |
                         session_logger.py --> logs/interactive_session_log.csv
```

### 4.2 Design principles
1. **One engine, two front ends.** The CLI and the web app call the same `router.py` and write the same log, so results are comparable.
2. **Hard constraints first, soft preferences second.** A model that cannot run is excluded before any score is computed.
3. **Everything is explainable.** Every answer carries a routing explanation (scores, RAM budget, exclusions, reason) shown in the UI and logged.
4. **Honest fallbacks.** When nothing fits RAM the router still answers (smallest model) but says so; a manual choice runs even if it does not fit, with a warning.
5. **Measured, not assumed.** Footprints, cold-start overhead, quality and latency come from measurements on this machine.

### 4.3 Technology
| Layer | Technology |
|---|---|
| Inference | Ollama (CPU) via the `ollama` Python client |
| Engine | Python (psutil, scikit-learn, pandas) |
| Web backend | FastAPI + uvicorn; streaming via Server-Sent-Events-style `data:` lines |
| Web frontend | Vanilla JavaScript, no framework; Markdown (`marked`) and math (`KaTeX`) rendering |
| Persistence | CSV log on disk; browser `localStorage` for chats |

---

## 5. Workflows step by step

### 5.1 A text question, end to end (web app)
1. You type a question and press Enter. The browser sends `POST /api/chat/stream` with the query, the mode, the last ≤6 messages of that conversation as context, and an optional manually chosen model.
2. `app.py` validates the request and starts `route_and_generate_stream()` (an async generator that pulls the router generator in a worker thread).
3. **Routing** (`select_model`, section 5.2) picks a model and builds an explanation. The first event, `routing`, is sent immediately so the routing panel appears at once.
4. The router calls Ollama with the chosen model (`think=False`, token budget from section 9, `keep_alive=3m`, `stream=True`). Each piece of text becomes a `token` event, shown live.
5. When generation ends, a `done` event carries the full result. The backend writes one row to the session CSV (`source = web`).
6. The browser renders the final Markdown/math, adds the routing summary and a Copy button, and saves the exchange (question, answer, routing record) in `localStorage`.
7. If you press **Stop** the browser aborts the request; the backend's `finally` closes the router generator, which closes the HTTP stream to Ollama, and Ollama stops generating (verified within about one second).

### 5.2 The routing decision (`select_model`)
| Step | What happens |
|---|---|
| 1 | Pick weights from the mode: `fast` 0.1/0.7/0.2, `balanced` 0.3/0.5/0.2, `quality` 1.0/0/0 (quality/latency/resource); `explore` uses the balanced weights. |
| 2 | `wait_for_ram_to_settle()`: poll free RAM every 0.5 s (up to 6 s) until two readings agree within 0.1 GB, because Windows takes 1-2 s to reclaim memory after a model unloads. |
| 3 | **Classify** the query into one of 9 categories (section 5.3). |
| 4 | **Read resources** with `psutil`: available RAM, CPU % (0.5 s sample), battery % and charging state. |
| 5 | **Effective RAM** = free RAM + RAM held by models Ollama has loaded (that memory is reclaimable: Ollama evicts it when another model is needed). |
| 6 | **Stress adjustment:** if CPU > 85 % or battery < 15 % and discharging, the budget is multiplied by 0.7. |
| 7 | Read which models are **warm** (`ollama ps`). |
| 8 | **Hard constraint:** a warm model always passes (it needs no new memory); any other model needs `footprint + 0.5 GB <= budget`. |
| 9 | If nothing passes, fall back to the smallest model (`llama3.2:3b`) and flag `fallback_triggered`. |
| 10 | **Score** the survivors (formula below). |
| 11 | Greedy choice = highest score. In `explore` mode, with probability 10 % pick a random *other* candidate instead. If the user forced a model, use it and keep the router's own pick in `greedy_choice`. |
| 12 | Build the explanation (scores, warm flags, exclusions, RAM figures, a plain-language reason). |

**Scoring formula**
```
score(model) = w_quality  * quality_ratio
             + w_latency  * latency_ratio
             + w_resource * resource_ratio
quality_ratio  = model_quality / best_quality_among_candidates          (higher is better)
latency_ratio  = best_latency  / model_latency_adjusted                 (lower is better)
resource_ratio = smallest_footprint / model_footprint                   (lower is better)
latency_adjusted = benchmark latency for the category + cold-start overhead (0 if the model is warm)
```
Ratio scoring replaced min-max normalisation because one outlier (Qwen3's 50-67 s latency) distorted the scale.

**Worked example (a real routing panel, balanced mode, category `knowledge`, 3.64 GB budget, nothing loaded)**

| Model | Quality | Latency | + cold start | Adjusted | quality_ratio | latency_ratio | resource_ratio | Score |
|---|---|---|---|---|---|---|---|---|
| llama3.2:3b | 9.50 | 31.72 s | +4.80 | 36.52 s | 0.992 | 0.663 | 1.000 | 0.829 |
| **phi4-mini** | 9.57 | 19.44 s | +4.76 | 24.20 s | 0.999 | 1.000 | 0.837 | **0.967** |
| gemma3:4b | 9.58 | 59.74 s | +7.40 | 67.14 s | 1.000 | 0.360 | 0.896 | 0.659 |

(phi: 0.3x0.999 + 0.5x1.000 + 0.2x0.837 = 0.967. The panel showed exactly these numbers.
This panel was captured **before** the footprint unit fix, when qwen3:4b still needed 3.7 GB and was excluded at 3.64 GB.
With the corrected footprints it needs 3.48 GB and would now be scored too - its quality/latency are worse than the best
values, so it would score lower and the three scores above would not change.)

### 5.3 Classifying a query (hybrid classifier)
1. **Rules first.** Keyword/regex lists are checked in this order: programming, reasoning, language, content, data, task_planning, conversation, knowledge; no match gives `other`. Single words match with simple inflections (`equation` matches `equations`, `sort` matches `sorting`).
2. **Follow-up inheritance.** If no rule matched and the message looks like a continuation (1-2 words, or up to 8 words starting with *continue, go on, more, yes, yeah, ok, and, also, what about, how about, then, next, elaborate, expand*) and there is earlier conversation, it inherits the category of the most recent earlier question that had a real category. A short standalone question such as "who wrote hamlet" does **not** inherit.
3. **ML fallback.** If still unmatched and the message has at least 3 words, a TF-IDF (1-2-grams) + linear SVM trained on 204 examples decides. It is trained lazily on first use (about a second).
4. Otherwise `other`.

### 5.4 An image question (web app)
1. **Attach** (+ button, paste, or drag-and-drop): the image is uploaded **once** (`POST /api/image/upload`) and gets an id; a 160-pixel thumbnail is made in the browser for display.
2. **Ask**: `POST /api/image/ask` sends the image id, the question and up to the last 2 earlier Q&A pairs about this same image (each answer cut to 1200 characters) so follow-ups such as "explain the second point" work.
3. **Routing is rule-based:** if the question contains an OCR-style phrase (*text in, read the, extract text, ocr, document, transcribe, what does it say, what's written, words in, written on, read this, what text*) the model is Qwen3-VL, otherwise Gemma 3 4B. No vision benchmark exists, so this is a documented heuristic.
4. The RAM check only warns (there is no smaller vision model); a loaded vision model counts as fitting.
5. **Empty-answer fallback:** if Qwen3-VL returns nothing, the router retries once with Gemma.
6. The answer says whether the model was **WARM** (already loaded) or **COLD**.
7. **Same question twice** on the same image shows the earlier answer instead of running again; **Re-run** forces a new answer.
8. **Compare** is a toggle: turn it on, type the question, press Enter or the arrow. Both vision models run; the toggle then switches itself off (a comparison takes about twice as long). Empty answers explain why (token limit, hidden "thinking").

### 5.5 Manual model choice
The Model selector offers *Auto (router)* or one specific live model. The router still computes its own decision; the answer shows `[MANUAL]`, highlights your model and labels the router's pick. If the chosen model is not loaded and does not fit the RAM budget it still runs, with a warning. The scoring mode is disabled while a model is forced.

### 5.6 Warm/cold lifecycle
```
query 1 (llama chosen, cold: +4.8 s load) -> answers -> stays loaded 3 min (keep_alive=3m)
query 2 (within 3 min): llama is WARM -> no cold overhead in its score, no new RAM needed
switch to another model: Ollama evicts the loaded one (OLLAMA_MAX_LOADED_MODELS=1), new model is cold
after 3 idle minutes: model unloads, memory returns to Windows
```
Before this change the router used `keep_alive="0s"`, so every query was cold and the warm logic never triggered.

### 5.7 Saving chats and the activity log
- **Browser:** each completed exchange is written to `localStorage`. Failed or stopped-before-text requests are not saved. Image chats store their messages and thumbnails; the image itself stays on the server.
- **Server:** every answered request appends one row to `logs/interactive_session_log.csv` (queries and routing data, **not** answer text).

### 5.8 Backend start-up
1. `app.py` imports `router` and `session_logger`, loads `model_profiles.json`.
2. Creates `webapp/backend/uploads/` and deletes uploads older than 72 hours.
3. Detects whether the router supports vision history and manual model choice (so an un-patched router degrades gracefully).
4. Serves the frontend folder at `/`.

---

## 6. Folder and file reference

```
D:\AdaptiveLocalAI2\
|-- README.md  LICENSE  .gitignore       GitHub front page, license, upload rules
|-- requirements.txt                     Python packages (minimum versions)
|-- setup.ps1  run_web.bat  run_cli.bat  one-time setup; start the web app / command line
|-- session_logger.py                    shared CSV logger (must be in this top-level folder)
|-- app\         maincode.py             command-line application
|-- benchmark\                           benchmark and profile-building scripts
|-- dataset\                             prompts and test sets
|-- logs\                                session log CSV (private; only .gitkeep is uploaded)
|-- resource_monitor\                    system-state reader
|-- results\                             all measured data (CSV/JSON)
|-- router\                              the engine, classifiers, experiments
|-- webapp\backend\                      FastAPI server (+ uploads\ folder, private)
|-- webapp\frontend\                     index.html, style.css, app.js
|-- tools\                               check_setup, check_requirements, cleanup, organize_project, ram_check, qwen_vl_debug
|-- docs\                                PROJECT_DOCUMENTATION.md
|-- archive\                             local only: old patch scripts, router backups, old logs, private notes
`-- venv\                                Python virtual environment (not project code, not uploaded)
```

### 6.1 Top level, `tools/`, `docs/` and `archive/`
| File | Purpose |
|---|---|
| `README.md`, `LICENSE`, `.gitignore` | GitHub front page (install and run guide), MIT license, rules that keep private data off GitHub |
| `requirements.txt` | Packages needed |
| `setup.ps1` | One-time Windows setup: virtual environment, packages, optional model download (`-PullModels`), Ollama setting (`-SetOllamaEnv`), final check |
| `run_web.bat`, `run_cli.bat` | Start the web app / the command-line app |
| `session_logger.py` | `ensure_log_file()` and `log_interaction()`; defines the 26-column log schema; used by both CLI and web. **Must sit directly in the top-level project folder, not inside a folder of the same name.** |
| `tools/check_setup.py` | Read-only readiness check: Python, packages, project files, Ollama server and models, `OLLAMA_MAX_LOADED_MODELS`, RAM/disk, and a dry run of the routing engine (`--generate` adds a real tiny generation) |
| `tools/check_requirements.py` | Compares the imports in the code with `requirements.txt` |
| `tools/cleanup.py` | Dry-run-first removal of uploaded images, `__pycache__` folders and (with backup) the activity log |
| `tools/organize_project.py` | Moves one-time patch scripts, backups, old logs and private notes into `archive/` (never deletes) |
| `tools/ram_check.py` | Read-only RAM diagnostic: app figure vs OS, router budget, who uses memory, footprint units, relevant settings (`--watch 12` samples over time) |
| `tools/qwen_vl_debug.py` | One capped vision call that reports `done_reason`, tokens, hidden-thinking length, window use (`--model --question --tokens --think --ctx --list`) |
| `docs/PROJECT_DOCUMENTATION.md` | This document |
| `archive/patches/patch_router*.py` | One-time router patches, all applied: `patch_router` (follow-up context), `_hybrid` (hybrid classifier), `_vision` (vision follow-ups), `_manual` (manual model), `_keepalive` (3-minute keep-alive, WARM/COLD), `_ramfix` (GiB units, loaded models eligible), `_visiondiag` (vision diagnostics) |
| `archive/router_backups/` | `router_BEFORE_*.py` safety copies and `task_classifier_OLD.py` |

### 6.2 `app/`
| File | Purpose |
|---|---|
| `maincode.py` | Interactive CLI: type queries, switch modes, ask about images, compare vision models, view session statistics. Uses streaming and prints the routing explanation live. See section 12. |

### 6.3 `benchmark/`
| File | Purpose |
|---|---|
| `benchmark_text.py` | Runs the 108 benchmark prompts on 5 models; records latency, tokens/s, RAM and CPU; resumable (writes `results/benchmark_results.csv`). |
| `analyze_benchmark.py` | Quick pandas summary of benchmark results (no model calls). |
| `quality_evaluation.py` | LLM-as-judge: Gemma scores each answer (correctness, relevance, completeness, clarity, instruction-following) -> `results/quality_results.csv`. |
| `build_model_profiles.py` | Merges benchmark and quality into per-model, per-category averages -> `results/model_profiles.json/.csv` (what the router reads). |

### 6.4 `dataset/`
| File | Purpose |
|---|---|
| `prompts.py` | 108 benchmark prompts: 9 categories x 12, with subcategories. |
| `classifier_extra_examples.py` | 24 extra classifier-only training examples. |
| `classifier_extra_examples_v2.py` | 72 terse, lowercase, real-style training examples (8 per category). Training only. |
| `test_prompts.py` | Held-out test set v1: 36 prompts (4 per category). |
| `test_prompts_v2.py` | Held-out test set v2: 45 prompts (5 per category), written after the hybrid classifier was designed. |

### 6.5 `resource_monitor/`
| File | Purpose |
|---|---|
| `resource_monitor.py` | `get_system_state()`: total/available RAM, RAM used %, CPU % (0.5 s sample), battery % and charging, plus a documented GPU note (not a live reading). |

### 6.6 `results/` (measured data - keep for the report)
| File | Content |
|---|---|
| `benchmark_results.csv` | Raw per-prompt benchmark runs |
| `quality_results.csv` | Judge scores per answer |
| `model_profiles.json` / `.csv` | Per-model, per-category quality, latency, tokens/s (router input) |
| `cold_start_calibration.json` | Cold-start overhead per model (router input) |
| `cold_warm_calibration_samples.csv` | 40 raw cold/warm samples (10 per model) |
| `baseline_decisions.csv`, `baseline_summary.csv` | Router vs fixed-model baselines |
| `pareto_analysis.csv` | Dominance analysis |
| `switch_frequency.csv` | Switch rate: predicted vs true categories |
| `classifier_comparison.csv` | Original rule vs LogReg vs SVM on test set v1 |
| `classifier_evaluation_v2.csv` | Final rule/SVM/hybrid evaluation, both test sets, with near-duplicate flags |
| `routing_impact.csv` | Per-prompt effect of a classifier error on the chosen model |

### 6.7 `router/`
| File | Purpose |
|---|---|
| `router.py` | **The engine** (section 5). Also holds all tunable constants (section 9). |
| `task_classifier.py` | Keyword/regex classifier, follow-up detection (`is_followup`, `classify_with_context`). |
| `hybrid_classifier.py` | Rules -> follow-up inheritance -> SVM fallback -> `other`. Records which path decided (`last_source`). |
| `ml_classifier.py` | TF-IDF + LogReg/SVM; trains on prompts + extras. |
| `compare_classifiers.py` | Original rule vs LogReg vs SVM comparison on test set v1. |
| `evaluate_classifiers.py` | Final evaluation: rule / SVM base / SVM augmented / hybrid base / hybrid augmented on both test sets, with near-duplicate filtering. |
| `analyze_routing_impact.py` | Measures how often (and how costly) a classifier error changes the chosen model. |
| `experiment_baselines.py` | Baselines (always-Llama/Phi/Qwen3, speed-only), Pareto analysis, switch frequency, bootstrap CIs. |
| `calibrate_cold_warm.py` | The 10-repetition cold/warm calibration. |
| `ram_diagnostic.py` | Early one-off script that discovered the 1-2 s memory-reclaim delay. |
| `router_BEFORE_*.py`, `task_classifier_OLD.py` | Backups made by the patch scripts - safe to archive. |

### 6.8 `webapp/`
| File | Purpose |
|---|---|
| `backend/app.py` | FastAPI server and static file host (section 11.5 lists the endpoints). |
| `backend/uploads/` | Stored uploaded images (section 14). |
| `frontend/index.html` | Page structure: sidebar, three tabs, composer, undo toast. |
| `frontend/style.css` | Dark theme, layout, chips, routing panel styles. |
| `frontend/app.js` | All client logic: chat streaming, image tab, saved conversations, status chips, routing panel, manual model, Stop, Compare toggle, history table. |

---

## 7. Datasets

### 7.1 Benchmark prompts (`dataset/prompts.py`)
108 prompts = 9 categories x 12. Categories and typical subcategories:

| Category | Examples of content |
|---|---|
| knowledge | Explain how a transistor works; TCP vs UDP; why ice floats |
| reasoning | Algebra, derivative, profit %, logic, "prove that...", trade-offs |
| programming | Write/debug Python, C++, SQL, regex; fix an IndexError |
| language | Translate to Hindi/French/Spanish, rewrite formally, fix grammar, write an email |
| content | Summaries, short stories, poems, name ideas, taglines |
| data | Extract entities/emails, classify reviews/spam/urgency, outliers, correlation |
| task_planning | Study plans, step-by-step instructions, document Q&A, agendas |
| conversation | Greetings, tips, opinions, icebreakers |
| other | Quick facts and conversions (capital of Australia, 25 °C in °F) |

These are well-formed, formal, capitalised prompts. Real users type tersely, which is why the classifier needed extra training data (section 8.8).

### 7.2 Classifier data
| Set | Size | Use |
|---|---|---|
| Benchmark prompts | 108 | Training |
| `classifier_extra_examples.py` | 24 | Training |
| `classifier_extra_examples_v2.py` | 72 (8/category) | Training; terse real-style phrasing |
| **Total training** | **204** | |
| `test_prompts.py` (v1) | 36 (4/category) | Held-out; its failures were seen during development, so it is a *development* check |
| `test_prompts_v2.py` (v2) | 45 (5/category) | Held-out; written by the assistant after the hybrid design |

**Near-duplicate filter:** every test prompt was compared with the training prompts (TF-IDF cosine similarity). Those at or above 0.5 are near-copies (for example "how do you say thank you very much in japanese" vs the benchmark's "How do you say 'Thank you very much' in Japanese?"). **17 of 81** test prompts were flagged and excluded from the headline number (n = 64). The 72 terse training examples were checked against both test sets: maximum similarity 0.36, so no leakage.

### 7.3 Quality of the data - limits to state
- Labels were written by one person (and v2 by the assistant); some are genuinely ambiguous (e.g. "difference between == and === in js").
- One prompt is worth 1.6-2.8 accuracy points, so gaps of 1-2 prompts are noise.
- The benchmark has 9-12 successful samples per model per category (a few empty outputs were dropped).

---

## 8. Experiments, data and results

### 8.1 Benchmark and model profiles
**Method.** 108 prompts (section 7.1) were run on 5 models (the 4 live text models plus `deepseek-r1:8b`). For each answer the
project recorded latency, tokens/second, memory change and CPU %, and Gemma 3 4B acted as an LLM judge (correctness, relevance,
completeness, clarity, instruction-following). Results were merged into per-model, per-category averages
(`results/model_profiles.json`), which is the router's data source.

**Average quality (0-10) / average latency (seconds) per category** - benchmark data, mostly warm-measured:

| Category | llama3.2:3b | phi4-mini | qwen3:4b | gemma3:4b |
|---|---|---|---|---|
| content | 9.55 / 11.71 | 9.35 / **7.43** | 9.23 / 49.24 | **9.60** / 23.20 |
| conversation | **9.65** / **21.14** | 9.60 / 22.78 | 8.73 / 67.10 | 9.45 / 52.41 |
| data | 9.78 / 9.61 | **9.85** / **5.61** | 9.15 / 59.31 | 9.78 / 19.81 |
| knowledge | 9.50 / 31.72 | 9.57 / **19.44** | 8.95 / 66.47 | **9.58** / 59.74 |
| language | 9.70 / 6.91 | 9.76 / **4.36** | 8.65 / 63.53 | **9.78** / 23.56 |
| other | **9.97** / **3.36** | 9.83 / 5.13 | 9.27 / 53.08 | 9.88 / 5.83 |
| programming | 9.48 / **18.36** | 9.62 / 24.05 | 8.05 / 62.79 | **9.68** / 67.60 |
| reasoning | 9.63 / **17.29** | 9.77 / 17.84 | 9.45 / 63.58 | **9.78** / 44.40 |
| task_planning | 9.17 / 29.40 | 9.25 / **19.31** | 9.00 / 64.50 | **9.42** / 54.00 |

`deepseek-r1:8b` (benchmark only): quality 9.31-9.84, latency 71-128 s per category - too slow to route to.
Generation speed (tokens/s): llama 16.8-18.6, phi 13.2-18.4, qwen3 12.7-13.8, gemma 11.1-14.2, deepseek 6.7-7.3.

**What the table shows**
- Quality is tightly clustered (8.05-9.97, most cells 9.2-9.9) with little discriminating power, while latency varies 3-16x (e.g. `other`: llama 3.36 s vs qwen3 53.08 s). That is why the default weights favour latency (0.5) over quality (0.3).
- **No model wins everywhere:** llama is best for `other` and conversation; phi for data, language and most latency; gemma has the highest quality in several categories but the highest latency.
- **qwen3:4b is never the best option** in any category.

### 8.2 Findings from the router experiments
| # | Finding | Evidence |
|---|---|---|
| 1 | **Qwen3:4b is Pareto-dominated in 9 of 9 categories (100%)**: another model is at least as good on quality and latency. It was never selected under any tested policy (105 prompts x 4 policies x 3 RAM scenarios). | `results/pareto_analysis.csv` |
| 2 | **Qwen-family models cannot reliably disable their reasoning trace** (`think=False` is not enough). Confirmed three times independently: Qwen3-VL as judge -> empty output; DeepSeek-R1 -> empty output on some prompts; Qwen3:4b in explore mode leaked raw `<think>` text into the answer (350 s). | benchmark/judge runs, session log 8 Oct |
| 3 | **Weight sensitivity:** swapping quality/latency weights 0.3/0.5 <-> 0.5/0.3 (resource 0.2) gave *identical* selections, so the router is stable in that range. | `experiment_baselines.py` |
| 4 | **CPU/battery awareness is decisive, not cosmetic:** with latency weight 0 (pure quality mode), simulated stress excluded every model except Llama, overriding the stated objective. | stress test |
| 5 | **Cold vs warm:** benchmark latencies are mostly warm-measured; live usage was always cold (`keep_alive="0s"`), so the cold-start overhead is added inside the scoring. | section 3.4 |
| 6 | **Switching cost of classifier errors:** predicted-category routing switches models 25.0% of the time vs 3.8% with true categories. (Dataset-order proxy, not a real session.) | `results/switch_frequency.csv` |
| 7 | **Baselines (95% bootstrap CIs):** adaptive routing beats always-Llama by **17.9%** in latency (CI excludes 0) and always-Qwen3 by **77.5%**; it does **not** beat always-Phi or speed-only with statistical significance (CIs include 0). Report this honestly. | `results/baseline_summary.csv` |

### 8.3 Classifier results

**Original comparison (test set v1, 36 prompts, before the hybrid):** rule-based **77.8%**, SVM **75.0%**, Logistic Regression **69.4%** (rule vs SVM statistically comparable).

| Category | Rule % | LogReg % | SVM % |
|---|---|---|---|
| content | 50 | 75 | 75 |
| conversation | 100 | 75 | 75 |
| data | 75 | 75 | 75 |
| knowledge | 75 | 50 | 50 |
| language | 75 | 50 | 75 |
| other | 50 | 100 | 100 |
| programming | 100 | 75 | 75 |
| reasoning | 100 | 50 | 75 |
| task_planning | 75 | 75 | 75 |

**Final evaluation (`evaluate_classifiers.py`, your machine).** "base" = trained on the original 132 examples; "aug" = plus the 72 terse examples. The hybrid = rules, then follow-up inheritance, then the SVM when no rule matches.

| Test subset | n | Rules | SVM base | SVM aug | Hybrid base | **Hybrid aug** |
|---|---|---|---|---|---|---|
| v1 (all) | 36 | 77.8% | 75.0% | 86.1% | 88.9% | 88.9% |
| v2 (all, includes near-duplicates - inflated) | 45 | 37.8% | 73.3% | 77.8% | 73.3% | 75.6% |
| v2 clean | 30 | 30.0% | 60.0% | 70.0% | 63.3% | 70.0% |
| **v1 + v2 clean (headline)** | **64** | **56.2%** | **67.2%** | **78.1%** | **76.6%** | **79.7%** |

Per-category accuracy on the 64 clean prompts (%):

| Category | Rules | SVM aug | Hybrid aug |
|---|---|---|---|
| content | 40 | 100 | 100 |
| conversation | 50 | 62 | 75 |
| data | 62 | 88 | 100 |
| knowledge | 43 | 57 | 57 |
| language | 62 | 75 | 75 |
| other | 40 | 100 | 40 |
| programming | 75 | 75 | 88 |
| reasoning | 67 | 83 | 100 |
| task_planning | 56 | 78 | 78 |

**How to read it**
- The keyword rules fit the *style of the benchmark prompts* (77.8% there) but generalise poorly to terse, real-style phrasing (30% on clean v2); 24 of their 28 errors on v2 were "no keyword matched".
- Adding 72 realistic training examples lifted the SVM by about 11 points (67.2% -> 78.1%) - more than any design change.
- **Hybrid vs rules: +23.5 points (15 prompts) - a clear win. Hybrid vs SVM alone: 79.7% vs 78.1% = one prompt, a tie.** The hybrid is kept because rules are exact and explainable for clear cases and it degrades to rules if scikit-learn is unavailable.
- Known weakness: quick-fact `other` queries ("largest ocean") are caught by the "what is" rule (hybrid 40% on `other`, SVM alone 100%).
- The 13 hybrid errors are mostly terse knowledge questions ("whats the point of virtual memory"), "what is ..." facts, and casual conversation lines.

**Caveats:** labels from one person; v2 and the extra training data written by the assistant (checked for leakage, but a shared style makes scores optimistic); one prompt = 1.6-2.8 points.

### 8.4 Does a classifier mistake matter? (routing-impact study)
`analyze_routing_impact.py` runs the router's selection twice for each of the 36 held-out prompts - once with the predicted category, once with the true one - for several RAM levels and modes (all models cold; rule classifier: 8 mistakes).

| Effective RAM | Mode | Selection changed | Wrong-but-harmless | Avg latency penalty | Avg quality change |
|---|---|---|---|---|---|
| 3.2 GB | all | 0/36 | 8/8 | 0 s | 0 |
| 3.5 GB | fast, balanced | 0/36 | 8/8 | 0 s | 0 |
| 3.5 GB | quality | 6/36 (16.7%) | 2/8 | -1.74 s | -0.012 |
| 4.5 GB | fast, balanced | 8/36 (22.2%) | 0/8 | **+1.14 s** | **-0.004** |
| 4.5 GB | quality | 7/36 (19.4%) | 1/8 | -1.63 s | -0.014 |
| 7.0 GB | fast, balanced | 8/36 (22.2%) | 0/8 | +1.14 s | -0.004 |
| 7.0 GB | quality | 7/36 (19.4%) | 1/8 | -1.63 s | -0.014 |

**Interpretation.** At low RAM the hard constraint overrides the classifier (nothing to choose between). At normal RAM every mistake flips the chosen model (mostly Phi vs Llama), but because their profiles are close the cost is small: about **+1.1 s of predicted latency per query (about 5 s per misclassified query)** and about -0.004 quality points. The latency is the benchmark (warm) figure of the chosen model on the true category - an estimate, not a live measurement.

### 8.5 Effect of keeping models loaded (warm awareness)
Using the real profiles and calibration with all four models eligible, 7 of the 36 (category x already-loaded-model) cases (19%) change the router's choice in `balanced` and `fast` mode, and none in `quality` mode (latency weight 0). Live confirmation: the second query of the same type shows the model as `[WARM]` with no cold-start overhead, and `ollama ps` shows one model at a time with about 2-3 minutes remaining.

### 8.6 RAM diagnostics on the real machine (`ram_check.py`)
| Measurement | Value |
|---|---|
| Total / usable memory | 15.34 GiB |
| Available (idle, no model loaded) | 2.06 GB, later 1.97 GB (used 13.28 GB = 86.6%) |
| App's free-RAM figure vs OS | **identical (difference 0.00 GB)**; Task Manager "Available 2.2 GB" matched too |
| Task Manager | In use 13.1 GB (1.2 GB compressed), Available 2.2, Committed **30.5 / 31.7 GB**, Cached 2.2, hardware reserved 674 MB, paged pool 938 MB, non-paged 1013 MB |
| Largest users at idle | Chrome 3.76 GB (27 processes), Windows services 1.35, Antigravity 0.91, Edge WebView2 0.86, ChatGPT app 0.75, VS Code 0.71, memory compression 0.66, Edge 0.47 |
| With phi4-mini loaded | `llama-server.exe` 2.88 GB (equals `ollama ps`), memory compression grew to 2.42 GB, available 3.03 GB, effective 5.91 GB |
| While generating (watch run) | free RAM fell 2.66 -> 1.95 GB over 33 s with 2.88 GB loaded (budget 5.54 -> 4.83 GB) |

**Conclusions**
1. **The calculation is correct;** the laptop is genuinely memory-tight (typically ~2-3 GB available when idle). At ~2 GB nothing passes the check (llama needs 2.92 GB), so the red RAM chip and the `[LOW RAM]` fallback are accurate.
2. **Bug found and fixed - units:** footprints came from the CLI (decimal GB) but were compared with GiB, overstating every model ~7.4%.
3. **Bug found and fixed - loaded models:** a model already in memory needs no new RAM but still had to pass "footprint + 0.5 GB <= budget", causing false `[LOW RAM]` while it was generating. Loaded models are now always eligible; only models that must be *loaded* need to fit.
4. Ollama's model-runner process is named `llama-server.exe` (not "ollama").

### 8.7 Qwen3-VL empty answers - diagnosis
An image Compare run returned **an empty answer after 347 s** for `qwen3-vl:4b`. `qwen_vl_debug.py` (token cap 600, `think=False`, same screenshot) showed:

| Run | Question | Wall time | Generated | Speed | Hidden thinking | Visible answer | Stop reason |
|---|---|---|---|---|---|---|---|
| 1 | "What is in this image?" | 94.8 s (load 8.5 s) | 600 tokens | 11.6 tok/s | 654 chars | 1,503 chars (cut at the cap) | `length` |
| 2 | "Describe this image in detail." | 55.7 s | 600 tokens | 10.9 tok/s | **2,061 chars** | **0 chars** | `length` |

- Ollama reports the model's capabilities as `completion, vision, tools, **thinking**`: the default `qwen3-vl:4b` tag is the *thinking* edition, and **`think=False` is ignored** (thinking appeared in both runs).
- Open-ended questions make it reason longer; if the reasoning uses the whole token budget the visible answer is empty (Ollama returns the reasoning in a separate field).
- The prompt (image + question) used 1,177 tokens of the default 4,096-token window, so a 4,096-token output could never have fitted - hence the larger window for Qwen3-VL in `router.py`.
- At ~11 tokens/s, 2,500 tokens takes about 230 s; the observed 347 s was probably slowed further by very low free RAM at that moment.
- **Mitigations in the system:** single-question mode falls back to Gemma when Qwen returns nothing; Compare reports the reason; the token budget and window were enlarged. A non-thinking edition (`qwen3-vl:4b-instruct`) was not downloaded and is future work.

### 8.8 Observed live behaviour (from sessions and logs)
- Typical CPU-only answer times: **25-120 s** for text (examples: llama 27-58 s, phi 25-68 s); a stopped or very long answer can reach several minutes.
- Exploration run: Qwen3:4b took **350 s** and leaked its reasoning.
- Vision: Gemma about 61-64 s per image answer; Qwen3-VL one successful run at 160 s, one empty run at 347 s.
- Routing overhead is about 0.5 s plus up to 6 s of RAM-settle wait.
- Manual selection of `gemma3:4b` and `qwen3:4b`, Stop, greyed-out models in the routing panel, the 2-minute countdown (`ollama ps`) and the Compare toggle were all confirmed working by use.

---

## 9. Configuration reference

### 9.1 Router (`router/router.py`)
| Constant | Value | Meaning |
|---|---|---|
| `LIVE_ROUTER_MODELS` | llama3.2:3b, phi4-mini:latest, qwen3:4b, gemma3:4b | The only models the router may select |
| `MODEL_RAM_REQUIREMENT_GB` | 2.42 / 2.89 / 2.98 / 2.70 / 3.76 (qwen3-vl) | Footprints in GiB (converted from the CLI's decimal GB) |
| `RAM_SAFETY_BUFFER_GB` | 0.5 | Headroom required above a model's footprint before loading it |
| `WEIGHTS` / `MODES` | fast 0.1/0.7/0.2, balanced 0.3/0.5/0.2, quality 1/0/0 | quality / latency / resource weights |
| `EXPLORATION_RATE` | 0.10 | Chance of picking a non-top model in `explore` mode (opt-in only) |
| `BATTERY_CRITICAL_PERCENT` | 15 | Below this and discharging = stressed |
| `CPU_HIGH_PERCENT` | 85 | Above this = stressed |
| `STRESS_BUDGET_REDUCTION_FACTOR` | 0.7 | Budget multiplier under stress |
| `KEEP_ALIVE` | `"3m"` | How long a text model stays loaded (env `ADAPTIVE_KEEP_ALIVE`) |
| `VISION_KEEP_ALIVE` | same as `KEEP_ALIVE` | Vision models (env `ADAPTIVE_VISION_KEEP_ALIVE`) |
| `TEXT_NUM_PREDICT` | 3600 | Max tokens per text answer (blocking and streaming) |
| `TEXT_NUM_CTX` | `None` | Context window for text models (default 4096). Set e.g. 6144 if long chats get cut (+~0.25 GB/model) |
| `VISION_NUM_PREDICT` | 4096 | Max tokens per vision answer |
| `QWEN_VL_NUM_CTX` | 8192 | Bigger window for Qwen3-VL (prompt + reasoning + answer) |
| `QWEN_VL_EXTRA_KV_GB` | 0.5 | **Estimate** of the extra RAM that window needs; set 0 to disable; refine with `ollama ps` |
| `VISION_OCR_KEYWORDS` | 13 phrases | Questions containing these go to Qwen3-VL |

### 9.2 Classifier
| Item | Value |
|---|---|
| Rule categories (checked in order) | programming, reasoning, language, content, data, task_planning, conversation, knowledge, else `other` |
| SVM | TF-IDF (1-2-grams), linear `SVC`, `C=1.0`, `random_state=42` |
| `ML_MIN_WORDS` | 3 (shorter inputs do not use the SVM) |
| `FOLLOWUP_MAX_WORDS` | 8 |
| Training data | 108 + 24 + 72 = 204 examples |

### 9.3 Backend (`webapp/backend/app.py`)
| Item | Value |
|---|---|
| Port | 8000 (`uvicorn app:app --reload --port 8000`) |
| `UPLOAD_RETENTION_HOURS` | 72 - unused uploads older than this are deleted at start-up (using an image refreshes its timer) |
| `MAX_UPLOAD_BYTES` | 20 MB |
| Allowed image types | .png .jpg .jpeg .gif .webp .bmp |
| Image id | `<32 hex characters>.<extension>` (validated; no path tricks) |
| Status polling | every 10 s from the browser |

### 9.4 Frontend (`webapp/frontend/app.js`)
| Item | Value |
|---|---|
| `localStorage` key | `adaptiveAI.v2` |
| Conversations kept | 40 (oldest dropped first) |
| Entries per conversation | 120 (chat), 80 (image) |
| Context sent with a chat message | last 6 messages (3 exchanges) |
| Image follow-up context | last 2 Q&A pairs, each answer cut to 1200 characters |
| Thumbnail size | 160 px, JPEG quality 0.7 |
| Undo window after Delete chat | 10 s |

### 9.5 Environment variables
| Variable | Where | Value used |
|---|---|---|
| `OLLAMA_MAX_LOADED_MODELS` | Windows User variable | 1 |
| `OLLAMA_MODELS` | Windows User variable | `D:\Ollama\models` |
| `ADAPTIVE_KEEP_ALIVE`, `ADAPTIVE_VISION_KEEP_ALIVE` | optional, before starting uvicorn/CLI | not set (defaults 3m) |

---

## 10. How to install and run

> **On a new PC the quickest way is `README.md`:** install Python 3.11, Git and Ollama, then run `setup.ps1 -PullModels -SetOllamaEnv`, restart Ollama, run `python tools\check_setup.py`, then `run_web.bat`. This section is the detailed manual version.

### 10.1 One-time setup
1. **Install Ollama** for Windows. Set `OLLAMA_MODELS=D:\Ollama\models` and `OLLAMA_MAX_LOADED_MODELS=1` (`setx NAME value`), then **Quit Ollama from the tray and start it again** - the tray app restarts the server itself and a stale environment is inherited otherwise. Open a *new* terminal so it sees the variables.
2. **Pull the models:**
   ```powershell
   ollama pull llama3.2:3b
   ollama pull phi4-mini
   ollama pull qwen3:4b
   ollama pull gemma3:4b
   ollama pull qwen3-vl:4b
   ollama pull deepseek-r1:8b      # only needed to re-run the benchmark
   ```
3. **Python environment** (from `D:\AdaptiveLocalAI2`):
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   pip install fastapi "uvicorn[standard]" python-multipart pandas scikit-learn psutil ollama
   pip freeze > requirements.txt
   ```
4. **Check the layout:** `session_logger.py` directly in the project root; `results\model_profiles.json` and `results\cold_start_calibration.json` present; the router patches applied (section 18).

### 10.2 Run the web application
```powershell
# Terminal 1 (only if Ollama is not already running as a tray app)
ollama serve

# Terminal 2
cd D:\AdaptiveLocalAI2
.\venv\Scripts\activate
cd webapp\backend
uvicorn app:app --reload --port 8000
```
Open **http://127.0.0.1:8000** (use this address consistently - `localhost:8000` has its own separate saved chats).
`--reload` only watches `webapp\backend`: after **any change under `router\`**, stop uvicorn (Ctrl+C) and start it again. After changing `app.js`/`style.css`, hard-refresh with Ctrl+Shift+R.

### 10.3 Run the command-line application
```powershell
cd D:\AdaptiveLocalAI2\app
python maincode.py
```
(Ollama must be running; the venv must be active.) See section 12 for commands.

### 10.4 Reproduce the experiments (in this order)
Run each from its own folder with the venv active; each script's docstring lists options. Long runs are resumable where noted.
| Step | Command | Produces |
|---|---|---|
| 1 | `python benchmark\benchmark_text.py` (hours; resumable) | `results\benchmark_results.csv` |
| 2 | `python benchmark\quality_evaluation.py` (Gemma as judge) | `results\quality_results.csv` |
| 3 | `python benchmark\build_model_profiles.py` | `results\model_profiles.json/.csv` |
| 4 | `python router\calibrate_cold_warm.py` | `results\cold_start_calibration.json`, samples CSV |
| 5 | `python router\experiment_baselines.py` | baselines, Pareto, switch-frequency CSVs |
| 6 | `python router\compare_classifiers.py` | `results\classifier_comparison.csv` |
| 7 | `python router\evaluate_classifiers.py` | `results\classifier_evaluation_v2.csv` |
| 8 | `python router\analyze_routing_impact.py` | `results\routing_impact.csv` |

### 10.5 Diagnostics
| Command | Use |
|---|---|
| `python tools\ram_check.py` | Is free RAM measured correctly? Which models fit? Who uses the memory? |
| `python tools\ram_check.py --watch 12` | Watch RAM change while an answer generates |
| `python tools\qwen_vl_debug.py "D:\path\image.png" --question "..."` | Why did a vision model return nothing? |
| `ollama ps` | Which model is loaded and for how long |
| `Select-String -Path "$env:LOCALAPPDATA\Ollama\server.log" -Pattern "OLLAMA_MAX_LOADED_MODELS" \| Select-Object -Last 1` | What the running Ollama server really uses |

### 10.6 Suggested demonstration script (5-7 minutes)
1. Open the web app; point at the status chips (RAM, CPU, battery).
2. Ask "what is TCP?" in **Auto**; expand the routing panel; explain scores, warm/cold and exclusions.
3. Ask a second question of the same type; show `[WARM]` and the shorter predicted latency; run `ollama ps`.
4. Switch **Model** to a specific one; send; show `[MANUAL]` and the router's own pick.
5. Press **Stop** during a long answer; show the kept partial answer.
6. Image tab: attach a screenshot, ask a question, then a follow-up; show WARM/COLD in the reason line.
7. Turn on **Compare**, ask a question, compare both vision models' answers and times.
8. Open **History** to show the unified CLI + web log.

---

## 11. Web application guide

### 11.1 Layout
- **Left sidebar:** *New chat*, the three sections (Chat, Image, History), and **Recents** (saved conversations, newest first; hover a row for ✕ to delete it). Collapse it with « (it becomes an overlay on narrow screens).
- **Top bar:** conversation title, **status chips**, and *Delete chat*.
- **Message area:** the only scrolling region; a ↓ button jumps to the newest message when you scroll up.
- **Composer:** text box, the **Model** and **Mode** selectors, and a round send/stop button.

### 11.2 Status chips
| Chip | Colour rule |
|---|---|
| **RAM x GB free** | green = all 4 models fit the router's budget; amber = some fit; red = none fit. A model that is already loaded counts as fitting. Hover for: free RAM + RAM held by loaded models = budget, and how many models fit. |
| **CPU x%** | amber above 85% (router becomes conservative) |
| **Battery** | ⚡ when charging; red below 15% on battery; hidden on a desktop |
| **CPU-only** | tooltip: the GPU is not used for inference |
| (backend down) | one red chip: "Backend unreachable - is uvicorn running?" |

### 11.3 Chat tab
- **Enter** sends, **Shift+Enter** new line. While an answer generates the send arrow becomes a red **■ Stop**.
- **Routing panel** (collapsed line above each answer: `category -> model [tags] (time)`). Expand to see every model: scores, quality, latency (with the cold-start overhead), **[WARM]/[cold]**, models that did not fit shown greyed as `[excluded]` with the reason (`needs ~3.4 GB, only 2.9 GB available`), and an amber warning when no model fit and the smallest was used. Tags: `[MANUAL]`, `[EXPLORATION]`, `[LOW RAM]`.
- **Copy** button under each answer; Markdown and LaTeX math are rendered.
- **Model selector:** Auto or one model. Below the box a note shows "Needs ~X GB, Y GB available" (green) or a warning (amber), or "Already loaded in memory".
- **Mode selector** (`balanced`, `fast`, `quality`, `explore`) - disabled when a model is forced.
- **Stopped answers** keep their partial text, are labelled, and are used as context for the next question.
- **Saved conversations:** every completed exchange is stored in your browser; they survive refreshes. **Delete chat** removes the conversation and shows an **Undo** bar for 10 seconds.

### 11.4 Image tab
1. Attach with **+**, paste (Ctrl+V into the box) or drag-and-drop. A chip shows the image; **Clear image** detaches it but keeps the chat.
2. Ask a question; follow-ups reuse the image and send the last two Q&A pairs as context.
3. **Re-run** under an answer forces a fresh answer; asking the identical question again shows the earlier answer instead of re-running.
4. **Compare models** is a toggle (highlighted when on). Turn it on, type the question, press Enter or the arrow; both models run, then the toggle switches off. Click again before sending to turn it off.
5. Attaching a different image mid-chat starts a clean context for that image. **Delete chat** removes the whole image conversation (Undo available).
6. **Stop** abandons the wait; the server may still finish the image in the background for a moment.

### 11.5 API reference
| Method and path | Request | Response |
|---|---|---|
| `GET /api/status` | - | `resource_state` (RAM total/available, CPU, battery, GPU note), `ollama_loaded_gb`, `loaded_models`, `effective_ram_gb`, `ram_budget_gb`, `stress_reasons`, `modes`, `live_models`, `manual_model_supported`, `model_requirements_gb` |
| `POST /api/chat/stream` | `{query, mode, history[], model?}` | Server-sent lines `data: {...}` with events `routing`, `token`, `done`, `error` |
| `POST /api/chat` | same | Full result JSON (non-streaming) |
| `POST /api/image/upload` | multipart field `image` | `{image_id, filename, size_bytes}`; errors 415 (type), 413 (too large), 400 (empty) |
| `POST /api/image/ask` | `{image_id, question, history[]}` | Result with `response_text`, `selected_model`, `selection_reason`, `selected_is_warm`, `done_reason`, `thinking_chars`; 404 if the image is gone |
| `POST /api/image/compare_saved` | `{image_id, question}` | `results_by_model` with each model's text, time, `done_reason`, `thinking_chars` |
| `POST /api/image`, `POST /api/image/compare` | multipart `question` + `image` | Older upload-every-time versions (kept for compatibility) |
| `GET /api/history?limit=N` | - | `{rows[], total_logged}` from the session CSV |
| `GET /` | - | The web app (static files) |

---

## 12. Command-line (CLI) guide

Start: `cd D:\AdaptiveLocalAI2\app` then `python maincode.py`.

| Type this | What it does |
|---|---|
| *any text* | Routes the query, prints the routing explanation, streams the answer, prints the time. The last 3 exchanges are kept as context for follow-ups. |
| `image: D:\Screenshots\Screenshot 2025-06-30 183453.png :: What is in this image?` | Asks one question about an image. The path may contain spaces; **no quotes**; `::` separates path and question. |
| `image: explain the second part` | Follow-up about the most recent image (no path, no `::`). |
| `imagecompare: D:\path\pic.png :: Describe this image` | Runs both vision models and prints both answers with timings; logs one row per model. |
| `clear image` | Forget the remembered image. |
| `clear context` | Forget the text conversation context (not the image). |
| `mode: fast` / `balanced` / `quality` / `explore` | Switch the scoring weights (`explore` is opt-in: ~10% of queries try a non-top model). |
| `history` | Print summary statistics of the session log (selection frequency, categories, average time, last 5 queries). |
| `quit` or `exit` | End the session. |

**Image routing in the CLI:** Qwen3-VL is used only if the question contains OCR-style words (*read the, text in, document, transcribe, what does it say, extract text, ocr, ...*); everything else goes to Gemma 3 4B. The output shows the chosen model, RAM state, whether it fits, the reason (including **WARM/COLD**), the response and its time.

**Differences from the web app:** image answers are not streamed; there is no Stop button (Ctrl+C during generation closes the CLI); follow-ups reuse the image but do not send earlier answers as context; Compare runs immediately; there is no manual model selector.

**Example (older run, before the RAM fixes - shows the low-RAM fallback):**
```
[balanced] You: what is conditional probability
QUERY: what is conditional probability
Mode: balanced
Category classified as: knowledge
System state: RAM free 2.23 GB + held by loaded Ollama models 0.0 GB = effective 2.23 GB, CPU 2.6%, Battery 52% (on battery)
Excluded by hard constraint (insufficient RAM): ['phi4-mini:latest', 'qwen3:4b', 'gemma3:4b']
WARNING: Extreme low-memory fallback triggered - forced smallest model.
Candidate scores:
  llama3.2:3b [cold] score=1.000 (quality=9.5, latency=31.72s -> 36.52s (+4.8s cold-start)) <== SELECTED
Actual response time: 32.65s
```

---

## 13. Logging: the session CSV

File: `logs\interactive_session_log.csv`, created automatically with this header. One row per answered request (web and CLI). **Answer text is not stored here - only the query and routing data.** Vision comparison writes one row per model.

| # | Column | Meaning |
|---|---|---|
| 1 | `timestamp` | ISO time, seconds |
| 2 | `source` | `cli` or `web` |
| 3 | `query` | The question text |
| 4 | `mode` | `balanced`, `fast`, `quality`, `explore`, `manual`, `vision`, `vision_comparison` |
| 5 | `category` | Classifier category (or `image_understanding`) |
| 6-8 | `ram_available_gb`, `effective_ram_gb`, `stress_adjusted_ram_gb` | Free RAM; free + loaded-model RAM; budget after the stress cut |
| 9 | `stress_reasons` | e.g. "CPU load high (92%)" |
| 10-12 | `cpu_percent`, `battery_percent`, `battery_charging` | System state |
| 13-14 | `previous_model`, `selected_model` | Last model used (this session/process), model chosen |
| 15 | `model_switch` | True if the model changed |
| 16 | `model_was_warm` | Whether the chosen model was already loaded |
| 17-18 | `greedy_choice`, `exploration_occurred` | Router's own best pick (also the pick when you forced a model); whether explore overrode it |
| 19-22 | `predicted_quality`, `predicted_latency_sec`, `selected_score`, `all_candidate_scores` | Router's predictions and all scores |
| 23-24 | `ram_settle_wait_sec`, `routing_overhead_sec` | Routing timing |
| 25 | `actual_response_time_sec` | Measured generation time |
| 26 | `ran_under_ram_constraint` | True if it ran although RAM was below the requirement |

Example row (a real row; the `source` column added):
```
2026-10-16T15:37:55,cli,"explain gpu and kernel ,thread in OS",balanced,knowledge,6.87,6.87,6.87,,3.0,53,True,,phi4-mini:latest,False,False,phi4-mini:latest,False,9.57,19.44,0.9674,"llama3.2:3b:0.829,phi4-mini:latest:0.967,qwen3:4b:0.614,gemma3:4b:0.659",0.5,0.5202,48.84,False
```
Never append to a CSV with a different header; older files with other layouts were archived (`_OLD`, `_MIXED`).

---

## 14. Where data is stored and how to clean it up

Nothing is sent to an online service: models, prompts and answers stay on your machine. The only internet use is the browser
loading the Markdown (`marked`) and math (`KaTeX`) libraries from the jsDelivr CDN; without internet the answers still show
but as plain text.

### 14.1 Everything the system generates

| # | What | Exact location | What it contains | Cleaned automatically? | Safe to delete? |
|---|---|---|---|---|---|
| 1 | **Uploaded images** (web app) | `D:\AdaptiveLocalAI2\webapp\backend\uploads\` | One file per upload, named `<32 hex characters>.png/.jpg/...`. Older endpoints leave randomly named temp files here too. | **Yes** - files unused for **72 hours** are removed each time the backend starts (using an image resets its timer) | **Yes.** Affected image chats then say "image no longer available - attach it again"; their text and answers stay |
| 2 | **Chats** (text and image conversations) | **Browser storage** (`localStorage`), key `adaptiveAI.v2`, for the address `http://127.0.0.1:8000` (and separately `http://localhost:8000` if you ever used it; and separately per browser) | Questions, answers, routing records, image-chat messages with 160-px thumbnails (base64), the attached-image id, last Mode/Model, sidebar state, last tab | No (capped: 40 conversations, oldest dropped) | **Yes** - see 14.2 |
| 3 | **Query log / History tab** | `D:\AdaptiveLocalAI2\logs\interactive_session_log.csv` | One row per answered request (web + CLI): query text, category, model, RAM/CPU, scores, time. **No answer text.** | No | Yes, but it is **evidence for your report** - back it up first. It is re-created automatically with the right header on the next run |
| 4 | Old log archives | `logs\interactive_session_log_OLD.csv`, `_MIXED.csv` | Earlier test runs in older column layouts | No | Yes (reference only) |
| 5 | **CLI conversation and remembered image path** | Memory only | - | Lost when you quit | Nothing to delete. The CLI reads images from the path you give and never copies them |
| 6 | **Experiment results** | `D:\AdaptiveLocalAI2\results\` | Benchmark, quality scores, model profiles, calibration, baselines, classifier evaluations, routing impact | No | **Keep** - the router reads `model_profiles.json` and `cold_start_calibration.json`; the report needs the rest |
| 7 | Python caches | every `__pycache__` folder (`dataset`, `resource_monitor`, `router`, `webapp\backend`) | Compiled bytecode | No | Yes - regenerated automatically |
| 8 | Router backups / patch scripts | after `tools\organize_project.py`: `archive\router_backups\`, `archive\patches\` (before: `router\router_BEFORE_*.py`, `router\task_classifier_OLD.py`, `patch_router_*.py` in the project root) | Copies made before each patch | No | Yes (move to an `archive` folder rather than deleting, if you want a record) |
| 9 | **Ollama models** | `D:\Ollama\models` | Downloaded model files (about 19 GB in total) | No | Remove with `ollama rm <name>`. `llama3.2:latest` is a duplicate tag of `llama3.2:3b` (same ID) - removing the tag frees no space but is safe. `deepseek-r1:8b` (5.2 GB) is only needed to re-run the benchmark |
| 10 | Ollama logs | `%LOCALAPPDATA%\Ollama\` (`server.log`, `app.log`, rotated copies) | Server settings, request metadata, errors (not your prompt text, as far as is known) | Rotated by Ollama | Yes, with Ollama quit |
| 11 | Models loaded in RAM | not on disk | - | Unloads after 3 idle minutes | Free it now with `ollama stop <model>` |
| 12 | Virtual environment | `D:\AdaptiveLocalAI2\venv\` | Python packages | No | Only if you will rebuild it (`pip install -r requirements.txt`) |

**Never delete:** `session_logger.py`, everything in `router\` except the `router_BEFORE_*.py` backups, `resource_monitor\`, `dataset\`,
`results\model_profiles.json`, `results\cold_start_calibration.json`, and `webapp\` (except the contents of `uploads\`).

### 14.2 Clean-up recipes (PowerShell)

**Shortcut:** `python tools\cleanup.py` previews everything; `python tools\cleanup.py --uploads --apply`, `--pycache --apply` or `--log --apply` (backs the log up first) delete only what you name.

```powershell
# --- 1. Uploaded images: look first (nothing is deleted by this command)
Get-ChildItem D:\AdaptiveLocalAI2\webapp\backend\uploads | Measure-Object -Property Length -Sum

# --- 2. Delete all uploaded images (the folder is re-created at the next backend start)
Remove-Item D:\AdaptiveLocalAI2\webapp\backend\uploads\* -Force

# --- 3. Delete Python caches (skips the virtual environment)
Get-ChildItem D:\AdaptiveLocalAI2 -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\venv\*" } | Remove-Item -Recurse -Force

# --- 4. Reset the activity log (back it up first!)
$stamp = Get-Date -Format yyyyMMdd
Copy-Item D:\AdaptiveLocalAI2\logs\interactive_session_log.csv "D:\AdaptiveLocalAI2\logs\interactive_session_log_BACKUP_$stamp.csv"
Remove-Item D:\AdaptiveLocalAI2\logs\interactive_session_log.csv

# --- 5. Remove the duplicate model tag (keeps llama3.2:3b)
ollama rm llama3.2:latest

# --- 6. Unload models from RAM right now
ollama ps
ollama stop <name shown above>
```

**Browser chats (item 2)** - choose one:
- In the app: press **Delete chat** (or the ✕ in Recents) for each conversation.
- Open the app tab, press **F12** -> **Console**, and run: `localStorage.removeItem("adaptiveAI.v2")`, then refresh.
- Or F12 -> **Application** -> **Local Storage** -> `http://127.0.0.1:8000` -> right-click `adaptiveAI.v2` -> Delete.
- Or Chrome -> Settings -> Privacy -> Site settings -> *View permissions and data stored* -> remove `127.0.0.1`.
Do **not** delete files inside the browser's own profile folder by hand; use the browser tools above. If you also opened the app as `localhost:8000` or in another browser, repeat there.

**Important relationships**
- Deleting a chat in the browser does **not** delete its uploaded images, and deleting images does not delete chats.
- The History tab and the CLI `history` command read only the CSV; they are unaffected by browser clean-ups.
- If you delete the CSV, the History tab shows "No interactions logged yet" until the next question.

---

## 15. Testing and verification

### 15.1 Verified on your machine (by use)
| Item | Evidence |
|---|---|
| Free-RAM measurement | `ram_check.py`: app vs OS difference 0.00 GB; Task Manager matched |
| Warm/cold behaviour | Second routing panel shows `[WARM]`; `ollama ps` shows the model with ~2 minutes left |
| One model in RAM at a time | Four consecutive `ollama ps` outputs each listing one model while switching llama -> qwen3 -> gemma |
| Stop button, greyed-out models, manual model choice, Compare toggle, vision options, new `router.py`, CLI image commands | Confirmed working by you |
| Patch scripts | Run on your files; outputs showed "Patched" and the expected `Select-String` lines |

### 15.2 Verified during development (automated, in a simulated environment)
About 170 automated checks were run while building: the web frontend in a simulated browser with a fake backend (persistence across refreshes, switching/deleting/undoing chats, a late answer arriving after you left a conversation, Stop, Compare toggle, routing panel rows, manual model, status chips, quota/failure handling); the backend with a stand-in router (upload validation, path-traversal rejection, async streaming closing the Ollama stream on disconnect, status fields); and the router logic with a stand-in Ollama (units, loaded-model eligibility, manual override, low-RAM fallback, token options, vision fallback), including each patch script (applies, is idempotent, refuses if the file differs). **These test scripts live outside the project folder and are not part of it**; they are available if you want them packaged.

### 15.3 Not verified / caveats
- The simulated tests cannot see real rendering or real Ollama timing - the visual design was judged from screenshots of the running app.
- The extra 0.5 GB RAM figure for Qwen3-VL's larger window (`QWEN_VL_EXTRA_KV_GB`) is an estimate; measure with `ollama ps` while it is loaded.
- Python and package versions are not recorded here (section 2.2).

---

## 16. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: session_logger` | The file ended up inside a *folder* named `session_logger.py` | Move the real file to `D:\AdaptiveLocalAI2\session_logger.py`; verify with `python -c "import session_logger"` from the project root |
| History tab empty or not updating | Log file has an old/mixed column layout | Archive the old CSV; the new one is created with the correct header |
| Browser shows "Cannot reach the backend server" | uvicorn is not running or crashed | Start it again (section 10.2) |
| Red message `[WinError 10061] ... actively refused it` | Ollama is not running | Start Ollama (tray app or `ollama serve`) |
| `ollama serve` says the port is already in use | The Ollama tray app restarted the server by itself | Quit Ollama from the tray icon; don't run a second server |
| Changes to `router.py` have no effect | `uvicorn --reload` only watches `webapp\backend` | Stop and start uvicorn (and restart the CLI) |
| Patch script prints "NOT patched ... differs" | `router.py` differs from what the patch expects (or an earlier patch is missing) | Apply patches in order (section 18) or send the lines it names |
| `OLLAMA_MAX_LOADED_MODELS` shows "(not set)" | The terminal was opened before `setx` | Open a **new** terminal; to check the running server read `server.log` (section 10.5) |
| Two models listed in `ollama ps` | The setting has not reached the server | Quit Ollama from the tray, start it again from a new session |
| Red RAM chip / `[LOW RAM]` almost always | The laptop really has ~2 GB available (browser tabs, editors, apps) | Close Chrome tabs/apps; run `python tools\ram_check.py` to see who uses memory |
| Qwen3-VL answer is empty | The default tag is a "thinking" model that ignores `think=False` and can use its whole token budget on hidden reasoning | Use single-question mode (falls back to Gemma), use Gemma, or try a non-thinking tag later; see section 8.7 |
| Chats missing after opening the app another way | `localhost:8000` and `127.0.0.1:8000` store separately; so do different browsers | Use one address and one browser |
| Dropdown lists hard to read | Browser draws native pop-ups | Styled in `style.css`; hard-refresh (Ctrl+Shift+R) |
| "Image no longer available" | The upload was cleaned up (72 h) or deleted | Attach the image again |
| Math/Markdown not formatted | No internet (CDN libraries not loaded) | Reconnect, or accept plain text |
| First ML-fallback query is slower | The SVM trains on first use (about 1 s) | Normal; later queries are instant |

---

## 17. Limitations and future work

### 17.1 Limitations (state these in the report)
- **Hardware-bound:** CPU-only inference on a memory-tight 16 GB laptop; answers take 25-120 s; RAM (~2-3 GB available at idle) often limits which models can be loaded.
- **Small evaluation sets:** classifier tests have 64 clean prompts; labels from one person; v2 and the extra training data were written by the assistant; one prompt = 1.6-2.8 points.
- **Router vs baselines:** it beats always-Llama and always-Qwen3 but not always-Phi or speed-only with statistical significance.
- **Benchmark mostly warm-measured**, while live use mixes cold and warm; cold-start overhead is a calibrated constant.
- **Vision routing is a keyword rule** (no vision benchmark); Qwen3-VL (thinking edition) can return nothing.
- **Earlier experiments assumed all-cold and 7%-conservative footprints** - note which setup each result used.
- **Single-user, single-machine, Windows-specific**; chats live in one browser per address.
- Image generation is out of scope (section 1.3).
- The "stress" thresholds (85% CPU, 15% battery, x0.7) are stated heuristics, not measured calibrations.

### 17.2 Future work
1. A non-thinking Qwen3-VL edition, or streaming vision answers so they can be stopped for real.
2. A sentence-embedding classifier (e.g. a small local embedding model) for better generalisation on terse phrasing.
3. Fix the "what is ..." rule vs `other` (quick facts) - evaluate on a *new* test set.
4. A vision benchmark so image routing can be data-driven instead of keyword-based.
5. A "free RAM" button that unloads models; per-conversation model choice.
6. Run-on-all-models side-by-side for text; a CLI `model:` command and Ctrl+C-to-stop.
7. Re-measure footprints with `ollama ps` at the context sizes actually used; long-term real-usage logging for learning routing weights.

---

## 18. Development history and patch order

### 18.1 Timeline
| Phase | Work |
|---|---|
| Scoping | Pivot from "just use Ollama" to an adaptive router; model pool locked; supervisor ask interpreted as image understanding |
| Measurement | 108-prompt benchmark on 5 models; Gemma judge; model profiles; real RAM footprints (`ollama ps`) |
| Router v1 | Rule classifier; hard RAM constraint; ratio scoring; modes fast/balanced/quality |
| Phase A | CPU/battery-aware budget (x0.7 under stress) |
| Phase B | Cold/warm calibration; cold-start overhead inside the scoring |
| Phase C | Opt-in exploration mode |
| Classifier study | Rule vs LogReg vs SVM on a held-out set |
| Applications | CLI; vision (single, follow-up, compare); streaming; shared logger |
| Web app | FastAPI + frontend; History tab; KaTeX; image tab |
| Redesign | ChatGPT/DeepSeek-style layout; saved conversations; Delete + Undo; Compare toggle; routing panel with all models; status chips |
| Hybrid classifier | Follow-up inheritance + SVM fallback; terse training data; near-duplicate filtering; new test set; routing-impact study |
| Controls | Manual model choice; Stop button (server closes the Ollama stream) |
| Warm/cold in practice | `keep_alive=3m`, one loaded model, WARM/COLD labels |
| RAM diagnosis | `ram_check.py`; unit fix; loaded-model eligibility |
| Vision diagnosis | `qwen_vl_debug.py`; reasons for empty answers; larger Qwen budgets |

### 18.2 Router patch order (apply to a fresh `router.py`)
After tidying, the scripts live in `archive\patches\` and the backups in `archive\router_backups\`.
| # | Script | Backup it creates | Status |
|---|---|---|---|
| 1 | `patch_router.py` | `router_BEFORE_PATCH.py` | applied |
| 2 | `patch_router_hybrid.py` | `router_BEFORE_HYBRID.py` | applied |
| 3 | `patch_router_vision.py` | `router_BEFORE_VISION.py` | applied |
| 4 | `patch_router_manual.py` | `router_BEFORE_MANUAL.py` | applied |
| 5 | `patch_router_keepalive.py` | `router_BEFORE_KEEPALIVE.py` | applied |
| 6 | `patch_router_ramfix.py` | `router_BEFORE_RAMFIX.py` | applied |
| 7 | `patch_router_visiondiag.py` | `router_BEFORE_VISIONDIAG.py` | applied |
| 8 | (final) larger token budgets + Qwen3-VL context window | `router_BEFORE_TOKENS.py` | delivered as a complete `router.py` that already contains steps 1-7 |

Every script is idempotent (running twice does nothing) and refuses to edit if the file differs from what it expects.

---

## 19. Using this document for the report

### 19.1 Suggested chapter outline
1. **Introduction** - problem, motivation, objectives (section 1).
2. **Background and related work** - local LLMs, quantization, routing/ensembles (add your literature).
3. **System design** - architecture and design principles (4), routing pipeline and scoring (5.2), classifier (5.3), vision (5.4).
4. **Implementation** - environment (2), files (6), web app (11), CLI (12), logging (13), configuration (9).
5. **Experimental methodology** - benchmark, judge, calibration, datasets, evaluation design (7, 8.1, 3.4).
6. **Results** - model profiles (8.1), findings (8.2), classifier (8.3), routing impact (8.4), warm/cold (8.5), RAM (8.6), vision (8.7).
7. **Discussion** - what the numbers mean, honest comparison with baselines, what surprised you (qwen3 dominated; units bug; thinking model).
8. **Limitations and future work** (17).
9. **Conclusion.**
10. **Appendices** - configuration table, log schema, API, run instructions, demo script.

### 19.2 Figures and tables worth producing
- Architecture diagram (4.1) and the routing flowchart (5.2).
- Quality/latency per category per model (8.1) as grouped bar charts, from `results\model_profiles.csv`.
- The worked scoring example (5.2) next to a screenshot of the routing panel.
- Classifier accuracy bars: rules / SVM / hybrid on the clean set (8.3).
- Routing-impact table (8.4) and the cold-start calibration table (3.4).
- RAM timeline from `ram_check.py --watch`; Task Manager screenshot.
- Screenshots: chat with routing panel, manual model, low-RAM greyed-out models, image tab with Compare, History tab.

### 19.3 Statements to include verbatim in spirit
- Scope: image *understanding*, not generation, and why.
- Classifier: "the hybrid is clearly better than rules (+23.5 points) and ties the SVM alone".
- Router: "beats always-Llama and always-Qwen3 significantly, not always-Phi".
- Corrections found during validation (units, loaded-model rule) - these show rigour.
- Qwen3-VL: the declared capability and real behaviour differ (`think=False` ignored).

---

## 20. Glossary

| Term | Meaning |
|---|---|
| **Router** | The program that decides which model answers each query |
| **Hard constraint** | A rule that excludes a model outright (it does not fit RAM) |
| **RAM budget** | Free RAM + RAM held by loaded models, reduced under stress; what a model to be loaded must fit into |
| **Warm / cold** | A model already in memory (warm) vs one that must be loaded first (cold) |
| **Keep-alive** | How long Ollama keeps a model in RAM after answering (3 minutes here) |
| **Footprint** | Memory a loaded model occupies |
| **GB vs GiB** | Decimal gigabyte (10^9 bytes) vs binary gibibyte (2^30 bytes); Ollama's CLI prints the former, Windows/psutil the latter |
| **Context window (`num_ctx`)** | Maximum tokens (prompt + answer) a model can hold at once |
| **`num_predict`** | Maximum tokens a model may generate for one answer |
| **KV cache** | Memory that grows with the context window during generation |
| **Thinking model** | A model that produces hidden reasoning before the visible answer |
| **LLM-as-judge** | Using a model (Gemma) to score other models' answers |
| **Pareto dominated** | Another option is at least as good on every measure and better on one |
| **Hybrid classifier** | Keyword rules first, then follow-up inheritance, then an SVM |
| **SVM / TF-IDF** | A text classifier and the word-weighting scheme it uses |
| **Near-duplicate** | A test prompt (cosine similarity >= 0.5) too similar to a training prompt |
| **SSE (streaming)** | Server sends the answer piece by piece as `data:` lines |
| **`localStorage`** | Small storage inside the browser, per address |
| **Explore mode** | Opt-in mode where ~10% of queries deliberately try a non-top model to collect data |

---

## Appendix A - Command cheat sheet
```powershell
# start everything
cd D:\AdaptiveLocalAI2 ; .\venv\Scripts\activate
cd webapp\backend ; uvicorn app:app --reload --port 8000      # then open http://127.0.0.1:8000
cd D:\AdaptiveLocalAI2\app ; python maincode.py                # CLI

# CLI image examples
image: D:\Screenshots\shot.png :: What is in this image?
image: explain the second point
imagecompare: D:\Screenshots\shot.png :: Describe this image
clear image

# checks
python tools\ram_check.py
python tools\ram_check.py --watch 12
python tools\qwen_vl_debug.py "D:\Screenshots\shot.png" --question "Describe this image."
ollama ps
ollama list

# reproduce evaluations (from D:\AdaptiveLocalAI2\router)
python evaluate_classifiers.py
python analyze_routing_impact.py
```

## Appendix B - Before you submit
- [x] Environment recorded (Python 3.11; `requirements.txt`).
- [ ] Run `python tools\organize_project.py --apply` (moves patch scripts and backups into `archive\`).
- [x] Published on GitHub: https://github.com/shivamparth123/AI_Adaptive_2
- [ ] Back up `logs\interactive_session_log.csv` and keep all of `results\`.
- [ ] Optional clean-up (section 14.2): uploaded images, `__pycache__`, browser chats.
- [ ] Tell the supervisor the image-generation scope decision.
- [ ] Take the screenshots listed in 19.2 and rehearse the demo (10.6).
- [ ] Zip the whole project folder (without `venv`, which can be rebuilt from `requirements.txt`).
