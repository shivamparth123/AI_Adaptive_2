"""
check_setup.py - is this machine ready to run Adaptive Local AI?   READ-ONLY: changes nothing.

Checks, in order: Python version, Python packages, project files, Ollama (command + server + models),
the OLLAMA_MAX_LOADED_MODELS setting, memory, and a dry-run of the routing engine (no model is loaded).

Usage (project folder, virtual environment active):
    python tools/check_setup.py               all checks (no model is run)
    python tools/check_setup.py --generate    also asks llama3.2:3b for a 5-token reply (loads it, ~10-30 s)

Exit code 0 = everything essential passed; 1 = at least one [FAIL].
"""
import argparse
import importlib.util
import os
import platform
import shutil
import sys
from importlib import metadata

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GB = 1024 ** 3

PACKAGES = [  # (import name, pip name)
    ("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("multipart", "python-multipart"),
    ("pandas", "pandas"), ("numpy", "numpy"), ("sklearn", "scikit-learn"),
    ("psutil", "psutil"), ("ollama", "ollama"),
]
TEXT_MODELS = ["llama3.2:3b", "phi4-mini:latest", "qwen3:4b", "gemma3:4b"]
VISION_MODELS = ["qwen3-vl:4b"]           # gemma3:4b (above) is the other vision model
OPTIONAL_MODELS = ["deepseek-r1:8b"]      # only needed to re-run the benchmark
REQUIRED_FILES = [
    "session_logger.py", "requirements.txt", "app/maincode.py", "router/router.py", "router/task_classifier.py",
    "router/hybrid_classifier.py", "router/ml_classifier.py", "resource_monitor/resource_monitor.py",
    "dataset/prompts.py", "dataset/classifier_extra_examples.py", "results/model_profiles.json",
    "results/cold_start_calibration.json", "webapp/backend/app.py", "webapp/frontend/index.html",
    "webapp/frontend/app.js", "webapp/frontend/style.css",
]

counts = {"ok": 0, "warn": 0, "fail": 0}


def report(level, text, hint=None):
    counts[level] += 1
    tag = {"ok": "[ OK ]", "warn": "[WARN]", "fail": "[FAIL]"}[level]
    print(f"{tag} {text}")
    if hint:
        print(f"       -> {hint}")


def check_python():
    v = sys.version_info
    text = f"Python {v.major}.{v.minor}.{v.micro} ({platform.system()} {platform.release()})"
    if (v.major, v.minor) < (3, 10):
        report("fail", text, "Python 3.10 or newer is required (3.11 recommended): https://www.python.org/downloads/")
    elif (v.major, v.minor) != (3, 11):
        report("warn", text + " - developed and tested on 3.11", None)
    else:
        report("ok", text)
    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        report("warn", "Not running inside a virtual environment", "Activate it first:  .\\venv\\Scripts\\activate")


def check_packages():
    for imp, pip_name in PACKAGES:
        if importlib.util.find_spec(imp) is None and not (imp == "multipart" and importlib.util.find_spec("python_multipart")):
            report("fail", f"Python package '{pip_name}' is missing", "pip install -r requirements.txt")
            continue
        try:
            ver = metadata.version(pip_name)
        except Exception:
            ver = "installed"
        report("ok", f"{pip_name} {ver}")


def check_files():
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(ROOT, f))]
    if missing:
        report("fail", "Project files missing: " + ", ".join(missing), "Re-download / re-clone the project, or restore these files")
    else:
        report("ok", f"All {len(REQUIRED_FILES)} essential project files are present")
    for folder in ("logs", os.path.join("webapp", "backend", "uploads")):
        if not os.path.isdir(os.path.join(ROOT, folder)):
            report("warn", f"Folder '{folder}' does not exist yet", "It is created automatically on first use (or run setup.ps1)")


def ollama_names():
    import ollama
    names = []
    for m in ollama.list().models:
        n = getattr(m, "model", None) or getattr(m, "name", None)
        if n is None and hasattr(m, "get"):
            n = m.get("model") or m.get("name")
        names.append(n)
    return names


def check_ollama():
    exe = shutil.which("ollama")
    if exe:
        report("ok", f"Ollama command found ({exe})")
    else:
        report("warn", "The 'ollama' command is not on PATH", "Install Ollama from https://ollama.com/download (then open a NEW terminal)")
    try:
        names = ollama_names()
    except Exception as e:
        report("fail", "Cannot reach the Ollama server at localhost:11434", f"Start Ollama (tray app or 'ollama serve'). Detail: {e!r}")
        return None
    report("ok", f"Ollama server reachable - {len(names)} model(s) installed")
    for m in TEXT_MODELS + VISION_MODELS:
        if m in names:
            report("ok", f"model {m}")
        else:
            report("fail", f"model {m} is not installed", f"ollama pull {m.replace(':latest', '')}")
    for m in OPTIONAL_MODELS:
        report("ok" if m in names else "warn", f"optional model {m}" + ("" if m in names else " not installed (only needed to re-run the benchmark)"))
    return names


def check_env():
    v = os.environ.get("OLLAMA_MAX_LOADED_MODELS")
    if v == "1":
        report("ok", "OLLAMA_MAX_LOADED_MODELS = 1 (this terminal)")
    else:
        report("warn", f"OLLAMA_MAX_LOADED_MODELS is {v or 'not set'} in this terminal",
               "Recommended 1 so only one model stays in RAM: setx OLLAMA_MAX_LOADED_MODELS 1, quit Ollama from the tray, start it again, open a NEW terminal")


def check_memory():
    import psutil
    vm = psutil.virtual_memory()
    total, avail = vm.total / GB, vm.available / GB
    if total < 12:
        report("warn", f"RAM: {total:.1f} GiB total, {avail:.1f} GiB available", "16 GB is recommended; with less, expect many 'low RAM' fallbacks")
    else:
        report("ok", f"RAM: {total:.1f} GiB total, {avail:.1f} GiB available right now")
    free_disk = shutil.disk_usage(ROOT).free / GB
    report("ok" if free_disk >= 20 else "warn", f"Disk: {free_disk:.0f} GiB free where the project lives", None if free_disk >= 20 else "The models need about 14 GB (19 GB with the benchmark model)")


def check_engine():
    sys.path.insert(0, os.path.join(ROOT, "router"))
    try:
        import router as R
        profiles = R.load_profiles()
        missing = [m for m in R.LIVE_ROUTER_MODELS if m not in profiles]
        if missing:
            report("fail", "model_profiles.json has no data for: " + ", ".join(missing))
            return
        d = R.select_model("what is a semaphore in an operating system", profiles)
        warm = "warm" if d["selected_is_warm"] else "cold"
        report("ok", f"Routing engine dry-run: category '{d['category']}' -> {d['selected_model']} ({warm}); "
                     f"RAM budget {d['resource_state']['stress_adjusted_ram_gb']} GB; excluded for RAM: {d['excluded_by_hard_constraint'] or 'none'}")
        if d["fallback_triggered"]:
            report("warn", "Low-memory fallback: no model fits the current free RAM", "Close browser tabs/apps and run again")
    except Exception as e:
        report("fail", f"Routing engine failed to run: {e!r}", "Run  python tools/ram_check.py  and check the traceback")


def check_generate():
    try:
        import ollama
        r = ollama.chat(model="llama3.2:3b", messages=[{"role": "user", "content": "Say OK."}],
                        options={"num_predict": 5}, keep_alive="1m")
        text = r["message"]["content"].strip()
        report("ok", f"llama3.2:3b generated a reply: {text!r}")
    except Exception as e:
        report("fail", f"Generation test failed: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true", help="also run a tiny real generation")
    a = ap.parse_args()
    print(f"Adaptive Local AI - setup check  (project: {ROOT})\n")
    check_python()
    check_packages()
    check_files()
    names = check_ollama()
    check_env()
    check_memory()
    check_engine()
    if a.generate and names is not None:
        check_generate()
    print(f"\nSummary: {counts['ok']} ok, {counts['warn']} warnings, {counts['fail']} failures")
    if counts["fail"]:
        print("Fix the [FAIL] items above, then run this check again.")
    else:
        print("Ready. Start the web app with run_web.bat (or the CLI with run_cli.bat).")
    sys.exit(1 if counts["fail"] else 0)


if __name__ == "__main__":
    main()
