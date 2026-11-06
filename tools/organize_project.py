"""
organize_project.py - tidy the project folder for GitHub. It only MOVES files (never deletes). DRY-RUN by default.

What it does (see docs/REPO_GUIDE.md for the reasons):
  - patch_router*.py                                  -> archive/patches/        (one-time scripts, already applied)
  - router/router_BEFORE_*.py, task_classifier_OLD.py -> archive/router_backups/ (safety copies)
  - logs/*_OLD.csv, logs/*_MIXED.csv                  -> archive/old_logs/       (old column layouts)
  - ram_check.py, qwen_vl_debug.py                    -> tools/                  (helper scripts)
  - PROJECT_DOCUMENTATION.md                          -> docs/
  - PROJECT_STATE.md                                  -> archive/notes/          (private working notes, not for GitHub)
  - copies requirements.txt to requirements-lock.txt if that file does not exist yet (your exact versions)
  - creates tools/, docs/images/, archive/ and .gitkeep files so empty folders are kept by git
It also lists files you must look at yourself ("REVIEW").

Usage (project folder):
    python tools/organize_project.py            show the plan only
    python tools/organize_project.py --apply    do it
"""
import argparse
import glob
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDERS = ["tools", "docs/images", "archive/patches", "archive/router_backups", "archive/old_logs", "archive/notes", "archive/old_versions"]
GITKEEP = ["logs", "webapp/backend/uploads", "docs/images"]
MOVES = [  # (glob relative to the project, destination folder)
    ("patch_router*.py", "archive/patches"),
    ("router/router_BEFORE_*.py", "archive/router_backups"),
    ("router/task_classifier_OLD.py", "archive/router_backups"),
    ("logs/interactive_session_log_OLD.csv", "archive/old_logs"),
    ("logs/interactive_session_log_MIXED.csv", "archive/old_logs"),
    ("ram_check.py", "tools"),
    ("qwen_vl_debug.py", "tools"),
    ("PROJECT_DOCUMENTATION.md", "docs"),
    ("PROJECT_STATE.md", "archive/notes"),
]
REVIEW = [
    ("app/test_models.py", "Open it. If it is only an old Ollama smoke test, move it to archive/; tools/check_setup.py replaces it."),
    ("results/interactive_session_log.csv", "A second copy of the activity log inside results/. If logs/interactive_session_log.csv is the real one, archive this copy."),
]


def unique_target(dest_dir, name):
    target = os.path.join(dest_dir, name)
    stem, ext = os.path.splitext(name)
    n = 1
    while os.path.exists(target):
        target = os.path.join(dest_dir, f"{stem}_{n}{ext}")
        n += 1
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    print(("APPLYING" if a.apply else "DRY RUN - nothing is changed") + f"   project: {ROOT}\n")

    for f in FOLDERS:
        path = os.path.join(ROOT, f)
        if not os.path.isdir(path):
            print(f"create folder  {f}/")
            if a.apply:
                os.makedirs(path, exist_ok=True)

    moved = 0
    for pattern, dest in MOVES:
        for src in sorted(glob.glob(os.path.join(ROOT, pattern))):
            if not os.path.isfile(src):
                continue
            dest_dir = os.path.join(ROOT, dest)
            if os.path.dirname(os.path.abspath(src)) == os.path.abspath(dest_dir):
                continue
            if os.path.exists(os.path.join(dest_dir, os.path.basename(src))):
                # the project package already put a (newer) copy there: keep the old one in archive, do not duplicate
                dest_dir = os.path.join(ROOT, "archive/old_versions")
            target = unique_target(dest_dir, os.path.basename(src))
            print(f"move  {os.path.relpath(src, ROOT)}  ->  {os.path.relpath(target, ROOT)}")
            moved += 1
            if a.apply:
                os.makedirs(dest_dir, exist_ok=True)
                shutil.move(src, target)

    req, lock = os.path.join(ROOT, "requirements.txt"), os.path.join(ROOT, "requirements-lock.txt")
    if os.path.isfile(req) and not os.path.exists(lock):
        pinned = sum(1 for line in open(req, encoding="utf-8", errors="ignore") if "==" in line)
        if pinned >= 8:
            print(f"copy  requirements.txt  ->  requirements-lock.txt   ({pinned} pinned versions = your exact environment)")
            if a.apply:
                shutil.copy2(req, lock)

    for folder in GITKEEP:
        keep = os.path.join(ROOT, folder, ".gitkeep")
        if not os.path.exists(keep):
            print(f"create  {folder}/.gitkeep   (so git keeps the empty folder)")
            if a.apply:
                os.makedirs(os.path.dirname(keep), exist_ok=True)
                open(keep, "a").close()

    print("\nREVIEW these yourself (not moved automatically):")
    for rel, why in REVIEW:
        if os.path.exists(os.path.join(ROOT, rel)):
            print(f"  {rel}\n      {why}")
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        print("\nNote: this folder is not a git repository yet (see docs/REPO_GUIDE.md, step 'git init').")
    print(f"\n{moved} file(s) {'moved' if a.apply else 'would be moved'}." + ("" if a.apply else "  Run again with --apply to do it."))


if __name__ == "__main__":
    main()
