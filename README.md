# Adaptive Local AI

**A hardware-aware router that picks the best local language model for every question - using your task, your RAM, your CPU and battery, and which model is already loaded.**

![Python](https://img.shields.io/badge/python-3.11-blue) ![Platform](https://img.shields.io/badge/platform-Windows%2011-lightgrey) ![Ollama](https://img.shields.io/badge/runs%20on-Ollama-black) ![License](https://img.shields.io/badge/license-MIT-green)

BTech final-year project, School of Entrepreneurship and Management (SME), IIT Jodhpur. Everything runs **on your own computer** (CPU only) - no cloud, no API keys, no data leaves the machine.

---

## Contents
1. [What it does](#1-what-it-does)
2. [Use case: private AI for organizations](#2-use-case-private-ai-for-organizations)
3. [How it works](#3-how-it-works)
4. [What you need](#4-what-you-need)
5. [Quick start on a new Windows PC](#5-quick-start-on-a-new-windows-pc)
6. [Manual installation (without the scripts)](#6-manual-installation-without-the-scripts)
7. [Using the web app](#7-using-the-web-app)
8. [Using the command line](#8-using-the-command-line)
9. [Models and modes](#9-models-and-modes)
10. [Configuration](#10-configuration)
11. [Project structure](#11-project-structure)
12. [Results at a glance](#12-results-at-a-glance)
13. [Reproducing the experiments](#13-reproducing-the-experiments)
14. [Where your data is stored and how to clean up](#14-where-your-data-is-stored-and-how-to-clean-up)
15. [Troubleshooting](#15-troubleshooting)
16. [Limitations](#16-limitations)
17. [Linux and macOS](#17-linux-and-macos)
18. [Documentation, license, credits](#18-documentation-license-credits)

---

## 1. What it does
Running LLMs locally is attractive, but a laptop has limited, shared memory, and no single model is best at everything. This project **routes each question to one of several local models** (Llama 3.2 3B, Phi-4-mini, Qwen3 4B, Gemma 3 4B) by weighing:

| Signal | Why it matters |
|---|---|
| **Task category** (9 kinds: programming, reasoning, knowledge, ...) | Each model was benchmarked per category on quality and speed |
| **Free RAM** | A model that cannot fit in memory is excluded (hard constraint) |
| **CPU load / battery** | Under stress the usable memory budget is cut so lighter models win |
| **Warm / cold state** | A model already loaded needs no load time and no extra RAM |
| **Your mode** | `fast`, `balanced`, `quality` weightings; or pick a model yourself |

**Features**
- Web app (chat with live streaming, Markdown and math), command-line app, shared activity log.
- **Explainable routing:** every answer shows all models' scores, which ones were excluded and why, warm/cold, and a plain-language reason.
- **Manual model choice**, a **Stop** button, saved conversations with Undo.
- **Image questions** (Gemma 3 / Qwen3-VL): attach once, ask follow-ups, **Compare** both vision models.
- Live status chips: RAM, CPU, battery.
- Hybrid task classifier (keyword rules + small SVM) and a complete experimental record (benchmark, calibration, evaluations).


## 2. Use case: private AI for organizations
Employees need AI tools, but cloud AI sends prompts, code and documents to third-party servers. Many organizations therefore restrict AI use, and running one large model locally is too heavy for an ordinary laptop. Adaptive Local AI runs entirely on the user's own computer. It identifies the task type, checks the free RAM, and automatically selects the best local model, so even a normal laptop stays responsive.

**Example:** when a new intern joins a firm, IT installs the application on the intern's system. The intern gets AI assistance from the first day, and all prompts, files and answers stay on that computer. Nothing is shared with other users or sent to the cloud, and removing the project folder removes the data.

**Benefits**
- **Data privacy:** works offline after setup; no data leaves the device.
- **No running cost:** no subscription, no API fees, free and open source (MIT).
- **Works on ordinary hardware:** the router avoids models that do not fit in free memory.
- **No cross-user leakage:** every installation is separate.

**Who it suits:** engineering and manufacturing firms, consultancies, legal and finance teams, startups and small businesses, colleges and labs.

**Current limits:** the models are small (3 to 8 billion parameters), results come from one 16 GB Windows laptop, and the tool is single-user (see [Limitations](#16-limitations)).


## 3. How it works
```
 Browser (web app)      Command line
        \                  /
         v                v
     webapp/backend/app.py  (FastAPI)         app/maincode.py
                  \              /
                   v            v
              router/router.py   <-- the engine
   classify query -> read RAM/CPU/battery -> hard constraints -> score -> pick -> explain
         |                |                         |
   task_classifier   resource_monitor        results/model_profiles.json
   hybrid_classifier                         results/cold_start_calibration.json
                          |
                   Ollama (localhost:11434)  -> models on disk
```
Score for each eligible model (ratios, best = 1.0):
```
score = w_quality * quality + w_latency * latency + w_resource * resource
        latency uses benchmark latency + a calibrated cold-start overhead (0 if the model is already loaded)
        balanced mode: 0.3 / 0.5 / 0.2     fast: 0.1 / 0.7 / 0.2     quality: 1 / 0 / 0
```
Full details, worked example and every design decision: [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).

## 4. What you need

| | Minimum | Recommended (what it was built on) |
|---|---|---|
| **OS** | Windows 10/11 | Windows 11 |
| **RAM** | 8 GB (expect many "low RAM" fallbacks) | **16 GB** (AMD Ryzen 7 5800H, CPU-only) |
| **Disk** | 15 GB free | 25 GB free |
| **Python** | 3.10 | **3.11** |
| **Ollama** | recent version | 0.33.2 |
| **Internet** | for the downloads below (the app itself works offline; only Markdown/math rendering loads two small libraries from a CDN) | |

**Downloads for a brand-new PC**

| What | Where | Size |
|---|---|---|
| Python 3.11 | https://www.python.org/downloads/ (tick **Add python.exe to PATH**) | ~30 MB |
| Git (to clone the repository; optional - you can download the ZIP instead) | https://git-scm.com/downloads | ~60 MB |
| Ollama | https://ollama.com/download | ~1 GB |
| Python packages | automatic (`pip install -r requirements.txt`) | ~0.5 GB |
| **5 required models** | `ollama pull ...` (below) | **~14 GB** |
| `deepseek-r1:8b` (optional, only to re-run the benchmark) | `ollama pull deepseek-r1:8b` | 5.2 GB |

| Model | Command | Disk | Role |
|---|---|---|---|
| Llama 3.2 3B | `ollama pull llama3.2:3b` | 2.0 GB | text pool |
| Phi-4-mini | `ollama pull phi4-mini` | 2.5 GB | text pool |
| Qwen3 4B | `ollama pull qwen3:4b` | 2.5 GB | text pool |
| Gemma 3 4B | `ollama pull gemma3:4b` | 3.3 GB | text pool + vision |
| Qwen3-VL 4B | `ollama pull qwen3-vl:4b` | 3.3 GB | vision (OCR questions) |

## 5. Quick start on a new Windows PC

Open **PowerShell** (Start menu -> "PowerShell"). Each step says what you should see.

**Step 1 - Install Python 3.11.** Run the installer, tick *Add python.exe to PATH*. Open a **new** PowerShell and check:
```powershell
python --version        # Python 3.11.x
```

**Step 2 - Install Ollama** from https://ollama.com/download and start it (a llama icon appears in the system tray). Check:
```powershell
ollama --version
```

**Step 3 - Get the code.** Either with Git:
```powershell
git clone https://github.com/shivamparth123/AI_Adaptive_2.git
cd AI_Adaptive_2
```
or click **Code -> Download ZIP** on the GitHub page, unzip it, and open PowerShell in that folder.

**Step 4 - Run the setup script** (creates the virtual environment, installs the Python packages, downloads the models, sets the recommended Ollama option, and checks everything):
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1 -PullModels -SetOllamaEnv
```
The model download (~14 GB) is the slow part. You can run `.\setup.ps1` again at any time; it only does what is missing.

**Step 5 - Restart Ollama once** (only because of `-SetOllamaEnv`): right-click the Ollama tray icon -> **Quit Ollama**, start Ollama again from the Start menu, then open a **new** PowerShell in the project folder.

**Step 6 - Verify the installation:**
```powershell
.\venv\Scripts\python.exe tools\check_setup.py
```
Every line should say `[ OK ]`. A `[FAIL]` line tells you exactly what to fix. Add `--generate` to also ask `llama3.2:3b` for a tiny reply.

**Step 7 - Start the web app:**
```powershell
.\run_web.bat
```
Open **http://127.0.0.1:8000** in your browser. Ask: *"what is a semaphore in an operating system?"* and click the small routing line above the answer to see how the model was chosen. The first answer is slower (the model must load); CPU-only answers take 25-120 seconds.

**Step 8 - Or start the command-line app:**
```powershell
.\run_cli.bat
```

To stop the web app press **Ctrl+C** in its window.

## 6. Manual installation (without the scripts)
```powershell
# in the project folder
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull phi4-mini
ollama pull qwen3:4b
ollama pull gemma3:4b
ollama pull qwen3-vl:4b

# recommended: keep only ONE model in RAM at a time, then quit/restart Ollama and open a NEW terminal
setx OLLAMA_MAX_LOADED_MODELS 1

python tools\check_setup.py

# web app
cd webapp\backend
uvicorn app:app --port 8000          # then open http://127.0.0.1:8000
# command line (new terminal, venv active, from the project folder)
cd app
python maincode.py
```
Add `--reload` to the `uvicorn` command while developing the web backend. Note that it only watches `webapp\backend`: after changing anything in `router\`, stop and start `uvicorn` again.

## 7. Using the web app
- **Left sidebar:** *New chat*, the sections *Chat / Image / History*, and **Recents** (your saved conversations; hover a row and click ✕ to delete one).
- **Chat:** Enter sends, Shift+Enter inserts a new line. The send arrow becomes a red **Stop** square while an answer is being written. Choose **Model** (*Auto (router)* or one specific model) and **Mode** under the text box. Click the routing line above an answer for the full explanation: scores per model, `[WARM]`/`[cold]`, models greyed out with the reason they did not fit, and a warning if the smallest model had to be used.
- **Status chips** (top right): RAM (green = every model fits, amber = some, red = none), CPU, battery. Hover for details.
- **Image tab:** attach an image with **+** (or paste / drag and drop), ask a question, then keep asking - the image stays attached and earlier answers are used as context. **Compare models** is a toggle: turn it on, type your question, press Enter, and both vision models answer. **Re-run** under an answer asks again.
- **Delete chat** removes a conversation; an **Undo** bar appears for 10 seconds.
- **History:** every question answered by the web app or the command line (the text of your questions and routing details; not the answers).
- Conversations are saved **in your browser** for the address you opened. Always use `http://127.0.0.1:8000` (not `localhost`) so they stay together.

## 8. Using the command line
Start with `run_cli.bat` (or `cd app` then `python maincode.py`).

| Type this | What it does |
|---|---|
| *any question* | Routes it, prints the routing explanation, streams the answer |
| `image: C:\path\to\picture.png :: What is in this image?` | Asks about an image. **No quotes** around the path (spaces are fine); `::` separates path and question |
| `image: explain the second part` | Follow-up about the last image (no path, no `::`) |
| `imagecompare: C:\path\to\picture.png :: Describe this image` | Runs both vision models and prints both answers |
| `clear image` / `clear context` | Forget the remembered image / the conversation |
| `mode: fast` / `balanced` / `quality` / `explore` | Change the scoring weights (`explore` is opt-in: ~10% of questions try a non-top model) |
| `history` | Statistics of the activity log |
| `quit` | Leave |

Vision routing: a question containing OCR-style words (*read the, text in, document, transcribe, what does it say, ...*) goes to **Qwen3-VL**; everything else goes to **Gemma 3**. In the CLI image answers are not streamed and there is no Stop button (Ctrl+C closes the program).

## 9. Models and modes
| Mode | Weights (quality / speed / memory) | Use it for |
|---|---|---|
| `balanced` (default) | 0.3 / 0.5 / 0.2 | Everyday use |
| `fast` | 0.1 / 0.7 / 0.2 | Quick answers |
| `quality` | 1.0 / 0 / 0 | Best answer regardless of speed (the only mode where Gemma wins) |
| `explore` | like balanced, 10% random | Collecting data on rarely-chosen models |
| **Manual model** | - | You choose; the router still shows what it would have chosen |

## 10. Configuration
Everything tunable is a constant at the top of `router/router.py`:

| Constant | Default | Meaning |
|---|---|---|
| `KEEP_ALIVE` / `VISION_KEEP_ALIVE` | `3m` | How long a model stays loaded after answering (env `ADAPTIVE_KEEP_ALIVE`, `ADAPTIVE_VISION_KEEP_ALIVE`) |
| `TEXT_NUM_PREDICT` | 3600 | Max tokens in a text answer |
| `VISION_NUM_PREDICT` / `QWEN_VL_NUM_CTX` | 4096 / 8192 | Vision answer limit and Qwen3-VL context window |
| `RAM_SAFETY_BUFFER_GB` | 0.5 | Headroom required above a model's size |
| `CPU_HIGH_PERCENT`, `BATTERY_CRITICAL_PERCENT`, `STRESS_BUDGET_REDUCTION_FACTOR` | 85, 15, 0.7 | When the system counts as stressed, and how much the memory budget shrinks |

> `results/model_profiles.json` and `results/cold_start_calibration.json` were measured on the author's laptop (Ryzen 7 5800H, 16 GB, CPU-only). On very different hardware, re-measure: `python router\calibrate_cold_warm.py` (and, optionally, the benchmark in section 13).

## 11. Project structure
```
AI_Adaptive_2/
|-- README.md  LICENSE  .gitignore
|-- requirements.txt            Python packages (minimum versions)
|-- setup.ps1                   one-time setup (venv, packages, models, checks)
|-- run_web.bat  run_cli.bat    start the web app / command line
|-- session_logger.py           shared activity log (must stay in this folder)
|-- app/maincode.py             command-line application
|-- router/                     THE ENGINE: router.py, classifiers, evaluation scripts
|-- resource_monitor/           reads RAM, CPU, battery
|-- webapp/backend/app.py       FastAPI server   (uploads/ = stored images, not uploaded to GitHub)
|-- webapp/frontend/            index.html, style.css, app.js
|-- benchmark/                  benchmark, LLM-as-judge, model-profile builder
|-- dataset/                    prompts, training examples, held-out test sets
|-- results/                    all measured data (CSV/JSON)
|-- logs/                       activity log (private, not uploaded)
|-- tools/                      check_setup.py, check_requirements.py, cleanup.py, ram_check.py, qwen_vl_debug.py, organize_project.py
`-- docs/                       PROJECT_DOCUMENTATION.md
```

## 12. Results at a glance
Measured on a Ryzen 7 5800H, 16 GB RAM, CPU-only (full tables and method in [the documentation](docs/PROJECT_DOCUMENTATION.md)).

| Finding | Result |
|---|---|
| Benchmark | 108 prompts x 5 models, 9 categories, Gemma as judge |
| Best model per category | **No single winner** (Llama best for quick facts and conversation, Phi for data/language and speed, Gemma for top quality at high latency); **Qwen3 4B is dominated in 9 of 9 categories** |
| Stability | Swapping quality/latency weights (0.3/0.5 <-> 0.5/0.3) gives **identical** selections |
| Cold-start overhead | llama 4.8 s, phi 4.76 s, qwen3 4.41 s, gemma 7.4 s |
| Router vs fixed model | **17.9% faster** than always-Llama, 77.5% than always-Qwen3; not significantly different from always-Phi |
| Task classifier (64 clean held-out prompts) | keyword rules **56.2%**, SVM 78.1%, **hybrid 79.7%** |
| Cost of a classifier mistake | changes the chosen model but only **~+1.1 s** predicted latency on average |
| Warm-model awareness | changes the choice in 7 of 36 cases (balanced/fast) |
| Real RAM check | the app's free-RAM figure matched the OS exactly; two router bugs (unit mismatch, loaded-model rule) found and fixed |

## 13. Reproducing the experiments
Run from the project folder with the virtual environment active. The benchmark takes hours on a CPU; it is resumable.
```powershell
python benchmark\benchmark_text.py            # needs all 6 models incl. deepseek-r1:8b
python benchmark\quality_evaluation.py        # Gemma as judge
python benchmark\build_model_profiles.py      # -> results\model_profiles.json
python router\calibrate_cold_warm.py          # -> results\cold_start_calibration.json
python router\experiment_baselines.py         # baselines, Pareto, switch frequency
python router\compare_classifiers.py          # rules vs LogReg vs SVM
python router\evaluate_classifiers.py         # final: rules / SVM / hybrid on both test sets
python router\analyze_routing_impact.py       # effect of classifier errors on routing
python tools\ram_check.py                     # is free RAM measured correctly? who uses it?
```
Each script's first lines explain its options and outputs.

## 14. Where your data is stored and how to clean up
Nothing is sent online. The program creates these (all local):

| What | Where | Notes |
|---|---|---|
| Uploaded images | `webapp\backend\uploads\` | Files unused for 72 hours are deleted each time the web app starts |
| Chats | Your **browser** (`localStorage`, key `adaptiveAI.v2`, per address and browser) | Delete in the app, or open the app, press F12 -> Console and run `localStorage.removeItem("adaptiveAI.v2")` |
| Question log / History tab | `logs\interactive_session_log.csv` | Text of your questions + routing details, no answers |
| Models | Ollama's folder (`%USERPROFILE%\.ollama\models` unless `OLLAMA_MODELS` is set) | `ollama rm <name>` |

Safe clean-up (shows what it would do first; nothing is deleted without `--apply`):
```powershell
python tools\cleanup.py                       # preview
python tools\cleanup.py --uploads --apply     # delete uploaded images
python tools\cleanup.py --pycache --apply     # delete Python caches
python tools\cleanup.py --log --apply         # back up, then reset the activity log
```

## 15. Troubleshooting
| Problem | Fix |
|---|---|
| `setup.ps1` is blocked | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then run it again |
| `python` is not recognised | Reinstall Python and tick *Add python.exe to PATH*; open a new PowerShell |
| Browser: "Cannot reach the backend server" | The web app is not running - start `run_web.bat` |
| Red message `... actively refused it` | Ollama is not running - start it (tray icon) |
| `ollama serve`: port already in use | Ollama already runs as a tray app; do not start a second server |
| Models do not seem to switch / two models stay in RAM | Set `OLLAMA_MAX_LOADED_MODELS=1`, quit Ollama from the tray, start it again, open a **new** terminal |
| RAM chip is red / "LOW RAM" warnings | The PC really is short of memory (browser tabs, apps). Close some; run `python tools\ram_check.py` to see who uses it |
| A change to `router\` has no effect | Stop and start the web app / CLI - it does not reload that folder |
| Chats disappeared | Another address or browser was used; use `http://127.0.0.1:8000` |
| Qwen3-VL answer is empty | The default tag is a "thinking" model that can use its whole token budget on hidden reasoning. Single questions fall back to Gemma; Compare says why |
| `python tools\check_setup.py` shows `[FAIL]` | The line tells you the command that fixes it |

## 16. Limitations
- CPU-only inference on a memory-tight laptop: answers take 25-120 s, and free RAM (often only 2-3 GB at idle) limits which models can be loaded.
- Classifier tests are small (64 clean prompts, labels by one person); the hybrid ties the SVM alone.
- The router beats always-Llama and always-Qwen3 but not always-Phi with statistical significance.
- Vision routing is a keyword rule (no vision benchmark exists); Qwen3-VL can return an empty answer.
- Developed and tested on Windows 11 only; single user; image *generation* is out of scope (the project does image *understanding*).

## 17. Linux and macOS
The Python code uses `psutil`, `ollama` and standard paths, so it should work, but it was **only tested on Windows**. Use the manual steps in section 6 with these changes: `source venv/bin/activate` instead of `.\venv\Scripts\activate`; set `export OLLAMA_MAX_LOADED_MODELS=1` before starting `ollama serve`; the `.ps1`/`.bat` scripts are Windows-only (run `python tools/check_setup.py`, then `cd webapp/backend && uvicorn app:app --port 8000` and `cd app && python maincode.py`).

## 18. Documentation, license, credits
- **Full documentation** (architecture, data, results, every file, API, log format, troubleshooting): [docs/PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md)
- **License:** MIT - see [LICENSE](LICENSE). The language models are **not** included and each has its own license (see its page on ollama.com).
- **Built with:** [Ollama](https://ollama.com), FastAPI, scikit-learn, psutil, pandas, marked, KaTeX.
- **Authors:** Shivam Kumar - <https://github.com/shivamparth123>, and Yash Singholiya.
- **Supervisor:** Dr. Venkat Ram Reddy Ganuthula
- **Institution:** School of Entrepreneurship and Management (SME), IIT Jodhpur