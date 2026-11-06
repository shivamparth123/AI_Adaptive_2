"""
cleanup.py - remove generated files. DRY-RUN by default: it only SHOWS what it would delete.

Usage (project folder):
    python tools/cleanup.py                       show everything that could be cleaned (deletes nothing)
    python tools/cleanup.py --uploads --apply     delete the uploaded images
    python tools/cleanup.py --pycache --apply     delete __pycache__ folders (never inside venv)
    python tools/cleanup.py --log --apply         back up logs/interactive_session_log.csv, then delete it
    python tools/cleanup.py --all --apply         all three

Never touches: venv, results, dataset, router, webapp source code.
Browser chats are NOT handled here - they live in the browser (see README, "Where your data is stored").
"""
import argparse
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS = os.path.join(ROOT, "webapp", "backend", "uploads")
LOG = os.path.join(ROOT, "logs", "interactive_session_log.csv")
NEVER_ENTER = {"venv", ".venv", ".git", "node_modules"}


def size_of(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(d, f))
            except OSError:
                pass
    return total


def human(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def find_uploads():
    if not os.path.isdir(UPLOADS):
        return []
    return [os.path.join(UPLOADS, f) for f in os.listdir(UPLOADS)
            if f != ".gitkeep" and os.path.isfile(os.path.join(UPLOADS, f))]


def find_pycache():
    found = []
    for d, dirnames, _ in os.walk(ROOT):
        dirnames[:] = [x for x in dirnames if x not in NEVER_ENTER]
        if os.path.basename(d) == "__pycache__":
            found.append(d)
            dirnames[:] = []
    return found


def find_log():
    return [LOG] if os.path.isfile(LOG) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uploads", action="store_true")
    ap.add_argument("--pycache", action="store_true")
    ap.add_argument("--log", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--apply", action="store_true", help="really delete (without it nothing is changed)")
    a = ap.parse_args()
    chosen = {"uploads": a.uploads or a.all, "pycache": a.pycache or a.all, "log": a.log or a.all}
    if a.apply and not any(chosen.values()):
        sys.exit("Say WHAT to clean, e.g.  python tools/cleanup.py --uploads --apply   (or --all --apply)")
    show_all = not a.apply and not any(chosen.values())

    groups = [
        ("uploads", "Uploaded images (webapp/backend/uploads)", find_uploads()),
        ("pycache", "Python caches (__pycache__)", find_pycache()),
        ("log", "Activity log (logs/interactive_session_log.csv) - backed up first", find_log()),
    ]
    mode = "APPLYING" if a.apply else "DRY RUN - nothing is deleted"
    print(f"cleanup.py ({mode})  project: {ROOT}\n")
    for key, title, items in groups:
        if not (show_all or chosen[key]):
            continue
        total = sum(size_of(p) for p in items)
        print(f"{title}: {len(items)} item(s), {human(total)}")
        for p in items[:8]:
            print("    ", os.path.relpath(p, ROOT))
        if len(items) > 8:
            print(f"     ... and {len(items) - 8} more")
        if not a.apply:
            continue
        if key == "log" and items:
            backup = LOG.replace(".csv", f"_BACKUP_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            shutil.copy2(LOG, backup)
            print("     backup saved:", os.path.relpath(backup, ROOT))
        for p in items:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            else:
                os.remove(p)
        print(f"     deleted {len(items)} item(s)")
    if not a.apply:
        print("\nNothing was deleted. Add --apply together with --uploads / --pycache / --log / --all to delete.")


if __name__ == "__main__":
    main()
