"""
RAM check - READ-ONLY diagnostic for the router's memory calculation. It loads nothing, changes nothing.

It answers three questions:
  1. Is the free-RAM number the app uses (resource_monitor.py) the same as the one Windows/psutil reports?
  2. What does the router do with that number (loaded-model memory added back, stress cut, which models fit)?
  3. Who is actually using the memory (so you can tell "bad calculation" from "RAM really is full")?

Usage (from the project folder, venv active):
    python tools\\ram_check.py               one snapshot
    python tools\\ram_check.py --watch 12    12 samples, 2.5 s apart - start it, then send a chat query and watch free RAM

Run it TWICE and paste both outputs:  (a) with nothing generating and `ollama ps` empty,
                                      (b) while an answer is being generated (or right after one).
"""
import argparse
import collections
import os
import sys
import time

import psutil

GB = 1024 ** 3
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE if os.path.isdir(os.path.join(HERE, "router")) else os.path.dirname(HERE)  # works from tools/ or the project root
for sub in ("router", "resource_monitor"):
    sys.path.append(os.path.join(ROOT, sub))

# ---- reuse the real code when it can be imported, otherwise fall back to the documented constants ----
R, monitor_state, notes = None, None, []
try:
    import router as R  # router.py: footprints, buffer, stress rule, loaded-model memory
except Exception as e:
    notes.append(f"could not import router.py ({e!r}) - using built-in constants instead")
try:
    from resource_monitor import get_system_state as monitor_state  # what the web app reads
except Exception as e:
    notes.append(f"could not import resource_monitor.py ({e!r}) - using psutil directly")

LIVE = getattr(R, "LIVE_ROUTER_MODELS", ["llama3.2:3b", "phi4-mini:latest", "qwen3:4b", "gemma3:4b"])
FOOT = getattr(R, "MODEL_RAM_REQUIREMENT_GB", {"llama3.2:3b": 2.6, "phi4-mini:latest": 3.1, "qwen3:4b": 3.2, "gemma3:4b": 2.9})
BUFFER = getattr(R, "RAM_SAFETY_BUFFER_GB", 0.5)


OLLAMA_NAMES = ("ollama", "llama-server", "llama_server")  # Ollama's runner process is llama-server.exe


def is_ollama(process_name):
    return any(t in process_name for t in OLLAMA_NAMES)


def loaded_models():
    """[(name, size_gb, until)] of models Ollama currently holds in memory."""
    try:
        import ollama
        out = []
        for m in ollama.ps().models:
            out.append((getattr(m, "model", "?"), round((getattr(m, "size", 0) or 0) / GB, 2), str(getattr(m, "expires_at", "") or "")[:19]))
        return out, None
    except Exception as e:
        return [], repr(e)


def stress(state, effective):
    if R is not None and hasattr(R, "compute_stress_adjustment"):
        try:
            return R.compute_stress_adjustment(dict(state, effective_ram_gb=effective))
        except Exception:
            pass
    reasons = []
    if state.get("battery_percent") is not None and state["battery_percent"] < 15 and not state.get("battery_charging"):
        reasons.append("battery critical")
    if state.get("cpu_percent", 0) > 85:
        reasons.append("CPU load high")
    return (round(effective * 0.7, 2) if reasons else effective), reasons


def snapshot():
    vm = psutil.virtual_memory()
    raw_free = vm.available / GB
    app_state, app_err = None, None
    if monitor_state is not None:
        try:
            app_state = monitor_state()
        except Exception as e:
            app_err = repr(e)
    state = dict(app_state) if app_state else {"ram_available_gb": round(raw_free, 2), "cpu_percent": psutil.cpu_percent(interval=0.3)}
    if "battery_percent" not in state:
        b = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
        state["battery_percent"], state["battery_charging"] = (b.percent, b.power_plugged) if b else (None, None)
    loaded, ps_err = loaded_models()
    loaded_gb = round(sum(s for _, s, _ in loaded), 2)
    effective = round(state["ram_available_gb"] + loaded_gb, 2)
    budget, reasons = stress(state, effective)
    return dict(vm=vm, raw_free=raw_free, state=state, app_err=app_err, loaded=loaded, ps_err=ps_err,
                loaded_gb=loaded_gb, effective=effective, budget=budget, reasons=reasons)


def fits(budget):
    return {m: (budget >= FOOT.get(m, 999) + BUFFER) for m in LIVE}


def report():
    s = snapshot()
    vm = s["vm"]
    print("=" * 74)
    print("1) MEMORY AS THE OPERATING SYSTEM SEES IT (psutil)")
    print(f"   total {vm.total / GB:5.2f} GB | available {vm.available / GB:5.2f} GB | used {vm.used / GB:5.2f} GB ({vm.percent}%)")
    sw = psutil.swap_memory()
    print(f"   page file/swap in use: {sw.used / GB:.2f} GB of {sw.total / GB:.2f} GB")
    print("\n2) THE NUMBER THE APP USES (resource_monitor.get_system_state)")
    if s["app_err"]:
        print("   FAILED:", s["app_err"])
    elif monitor_state is None:
        print("   (resource_monitor.py not importable - the figure above is used directly)")
    else:
        diff = s["state"]["ram_available_gb"] - s["raw_free"]
        print(f"   app free RAM {s['state']['ram_available_gb']} GB vs psutil available {s['raw_free']:.2f} GB  (difference {diff:+.2f} GB)")
        print("   -> " + ("MATCH: the monitor reads the same figure as the OS." if abs(diff) < 0.3
                          else "MISMATCH: resource_monitor.py reports a different number than psutil - check how it computes free RAM."))
    print("\n3) WHAT THE ROUTER DOES WITH IT")
    if s["ps_err"]:
        print("   `ollama ps` could not be read:", s["ps_err"])
    print(f"   models Ollama holds in memory now: " + (", ".join(f"{n} {sz} GB (unloads {u or '?'})" for n, sz, u in s["loaded"]) or "none"))
    unit_hits = []
    for n, sz, _ in s["loaded"]:
        const = FOOT.get(n)
        if const and sz:
            pct = (const - sz) / sz * 100
            print(f"   footprint check: {n} measured {sz} GiB (same units as free RAM); router.py constant {const} ({pct:+.1f}%)")
            if 5.0 <= pct <= 10.0:
                unit_hits.append(n)
    print(f"   free {s['state']['ram_available_gb']} + held by loaded models {s['loaded_gb']} = effective {s['effective']} GB")
    cut = f" -> after stress cut ({', '.join(s['reasons'])}): {s['budget']} GB" if s["reasons"] else ""
    print(f"   RAM budget used for the hard constraint: {s['budget']} GB{cut}")
    for m, ok in fits(s["budget"]).items():
        need = FOOT.get(m, 999) + BUFFER
        print(f"     {m:<18} needs {need:.1f} GB ({FOOT.get(m)} model + {BUFFER} buffer)  ->  {'fits' if ok else 'EXCLUDED'}")
    n_ok = sum(fits(s["budget"]).values())
    print(f"   {n_ok} of {len(LIVE)} models eligible" + ("  (router falls back to the smallest model)" if n_ok == 0 else ""))

    print("\n4) WHO IS USING THE MEMORY (sum of each program's working set; Chrome etc. are many processes)")
    groups = collections.defaultdict(lambda: [0, 0])
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            mi = p.info["memory_info"]
            if mi:
                g = groups[(p.info["name"] or "?").lower()]
                g[0] += mi.rss; g[1] += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    top = sorted(groups.items(), key=lambda kv: -kv[1][0])[:12]
    for name, (rss, n) in top:
        print(f"   {rss / GB:5.2f} GB  {name} ({n} process{'es' if n != 1 else ''})")
    total_rss = sum(v[0] for v in groups.values()) / GB
    print(f"   all processes together: {total_rss:.2f} GB (can double-count shared memory)")
    ollama_rss = sum(v[0] for k, v in groups.items() if is_ollama(k)) / GB
    print(f"   Ollama processes (ollama.exe + llama-server.exe): {ollama_rss:.2f} GB vs {s['loaded_gb']} GB reported by `ollama ps`")

    print("\n5) SETTINGS THAT AFFECT MEMORY")
    for var in ("OLLAMA_MAX_LOADED_MODELS", "OLLAMA_KEEP_ALIVE", "ADAPTIVE_KEEP_ALIVE", "ADAPTIVE_VISION_KEEP_ALIVE", "OLLAMA_NUM_PARALLEL"):
        print(f"   {var} = {os.environ.get(var, '(not set)')}")
    print(f"   router KEEP_ALIVE constant: {getattr(R, 'KEEP_ALIVE', '(not patched - models unload after every answer)') if R else '(router not imported)'}")

    print("\nWHAT THIS SUGGESTS")
    hints = []
    if monitor_state is not None and not s["app_err"] and abs(s["state"]["ram_available_gb"] - s["raw_free"]) >= 0.3:
        hints.append("The app's free-RAM figure disagrees with the OS - that is a real bug in resource_monitor.py.")
    if len(s["loaded"]) > 1 and not os.environ.get("OLLAMA_MAX_LOADED_MODELS"):
        hints.append("Several models are loaded at once and OLLAMA_MAX_LOADED_MODELS is not set - old models are piling up in RAM. Set it to 1.")
    if s["loaded_gb"] and ollama_rss - s["loaded_gb"] > 1.0:
        hints.append("Ollama's processes use well over what `ollama ps` reports - the footprints in router.py may be too small for the context size in use.")
    if not s["loaded_gb"] and ollama_rss > 1.0:
        hints.append("Ollama processes hold memory although `ollama ps` shows nothing loaded.")
    big = [(k, v[0] / GB) for k, v in groups.items() if v[0] / GB > 1.5 and not is_ollama(k)]
    if big:
        hints.append("Large non-Ollama users: " + ", ".join(f"{k} {g:.1f} GB" for k, g in sorted(big, key=lambda x: -x[1])[:4]) + " - closing these frees memory directly.")
    if unit_hits:
        hints.append("router.py's footprint constants are ~7% too high: they were read from the `ollama ps` command (decimal GB) "
                     "but are compared with RAM in binary GiB. Convert them (patch_router_ramfix.py does).")
    if vm.percent > 90:
        hints.append("Windows reports >90% memory in use: the RAM really is nearly full, so low-RAM warnings are correct.")
    if not hints:
        hints.append("Nothing unusual spotted automatically - paste this output and the Task Manager screenshot.")
    for h in hints:
        print("   -", h)
    for n in notes:
        print("   NOTE:", n)


def watch(samples):
    print(f"{'time':>8} | {'OS free':>8} | {'app free':>8} | {'loaded':>7} | {'budget':>7} | eligible | CPU%")
    t0 = time.time()
    for _ in range(samples):
        s = snapshot()
        print(f"{time.time() - t0:7.1f}s | {s['raw_free']:6.2f}GB | {s['state']['ram_available_gb']:6.2f}GB | {s['loaded_gb']:5.2f}GB | "
              f"{s['budget']:5.2f}GB | {sum(fits(s['budget']).values())} of {len(LIVE)}   | {s['state'].get('cpu_percent', '?')}")
        time.sleep(2.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", type=int, default=0, help="take N samples, 2.5 s apart")
    a = ap.parse_args()
    watch(a.watch) if a.watch else report()
