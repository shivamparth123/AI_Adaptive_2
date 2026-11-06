"""
check_requirements.py - does requirements.txt cover every third-party package the project's code imports?
READ-ONLY. Scans all .py files (skipping venv, archive, caches) and compares with requirements.txt.

Usage (project folder):    python tools/check_requirements.py
Exit code 1 if an imported third-party package is missing from requirements.txt.
"""
import ast
import os
import re
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"venv", ".venv", "env", "archive", "__pycache__", ".git", "node_modules"}
# import name -> pip distribution name (only where they differ)
ALIASES = {"sklearn": "scikit-learn", "multipart": "python-multipart", "python_multipart": "python-multipart",
           "cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "bs4": "beautifulsoup4", "dotenv": "python-dotenv"}


def normalise(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def project_modules():
    """Names of the project's own modules/packages (stems of .py files and folders), so they are not mistaken for pip packages."""
    local = set()
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for d in dirnames:
            local.add(d)
        for f in filenames:
            if f.endswith(".py"):
                local.add(f[:-3])
    return local


def imports_in(path):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # e.g. a docstring containing a Windows path such as C:\photos
            tree = ast.parse(open(path, encoding="utf-8").read())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module.split(".")[0])
    return found


def read_requirements():
    path = os.path.join(ROOT, "requirements.txt")
    if not os.path.exists(path):
        sys.exit("requirements.txt not found in the project folder")
    names = set()
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        names.add(normalise(re.split(r"[<>=!~\[; ]", line, 1)[0]))
    return names


def main():
    stdlib = set(sys.stdlib_module_names)
    local = project_modules()
    where = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.endswith(".py"):
                full = os.path.join(dirpath, f)
                for mod in imports_in(full):
                    if mod not in stdlib and mod not in local:
                        where.setdefault(mod, set()).add(os.path.relpath(full, ROOT))
    required = read_requirements()
    missing = {}
    for mod, files in sorted(where.items()):
        dist = normalise(ALIASES.get(mod, mod))
        if dist not in required:
            missing[mod] = (dist, sorted(files))
    print(f"Third-party imports found in the code: {', '.join(sorted(where)) or 'none'}")
    if missing:
        print("\nMISSING from requirements.txt:")
        for mod, (dist, files) in missing.items():
            print(f"  {dist}   (imported as '{mod}' in: {', '.join(files[:4])}{' ...' if len(files) > 4 else ''})")
        print("\nAdd the lines above to requirements.txt (check the pip name on pypi.org).")
        sys.exit(1)
    used = {normalise(ALIASES.get(m, m)) for m in where}
    unused = sorted(r for r in required if r not in used)
    print("\nrequirements.txt covers every imported package.")
    if unused:
        print("Listed but not imported directly (fine for runtime dependencies such as uvicorn/numpy): " + ", ".join(unused))


if __name__ == "__main__":
    main()
