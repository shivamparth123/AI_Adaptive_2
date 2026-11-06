# Repository guide - tidy the folder and publish on GitHub

This guide has three parts:
- **Part A** - exactly what to do in your `D:\AdaptiveLocalAI2` folder (about 15 minutes).
- **Part B** - what to keep, upload, move, merge or delete, file by file.
- **Part C** - step-by-step GitHub publishing for someone who has never used it.

All commands are for **PowerShell**. Replace `D:\AdaptiveLocalAI2` if your folder is elsewhere.

---

## Part A - Prepare the folder

### A0. Make a safety backup (do this first)
```powershell
Get-ChildItem D:\AdaptiveLocalAI2 -Exclude venv | Compress-Archive -DestinationPath D:\AdaptiveLocalAI2_backup_before_github.zip
```
(`venv` is skipped on purpose - it is large and can be rebuilt.)

### A1. Stop everything that is running
Press **Ctrl+C** in the terminal running `uvicorn`, and `quit` the CLI if it is open. Leave Ollama running.

### A2. Keep your exact package versions
You already have a `requirements.txt` (a `pip freeze`). Keep it under a new name; the package brings a cleaner one.
```powershell
cd D:\AdaptiveLocalAI2
Rename-Item requirements.txt requirements-lock.txt
```

### A3. Copy the new files into the project
Unzip `github_package.zip` **into the project folder** (it only adds files; it does not touch your code):
```powershell
Expand-Archive -Path "$HOME\Downloads\github_package.zip" -DestinationPath D:\AdaptiveLocalAI2 -Force
```
New in your folder now: `README.md`, `LICENSE`, `.gitignore`, `requirements.txt`, `setup.ps1`, `run_web.bat`, `run_cli.bat`, `tools\`, `docs\`, `.gitkeep` files.

### A4. Let the organizer move the clutter (it never deletes)
```powershell
.\venv\Scripts\activate
python tools\organize_project.py            # preview - read it
python tools\organize_project.py --apply    # do it
```
It moves the 7 `patch_router*.py` scripts, the `router_BEFORE_*.py` backups, `task_classifier_OLD.py`, the old log files, `PROJECT_STATE.md` and your old root copies of `ram_check.py`, `qwen_vl_debug.py`, `PROJECT_DOCUMENTATION.md` into `archive\`, so the project root is clean. `archive\` stays on your PC and is **not** uploaded (it is in `.gitignore`).

### A5. Delete generated files
```powershell
python tools\cleanup.py                                  # preview
python tools\cleanup.py --pycache --uploads --apply      # __pycache__ folders + the uploaded images
```
(Your 6 uploaded screenshots are private - they must not go to GitHub. `.gitignore` also blocks them.)

### A6. Review the two files the organizer left for you
| File | What to do |
|---|---|
| `app\test_models.py` | Open it. If it is just an old Ollama smoke test, move it into `archive\` (`tools\check_setup.py` replaces it). If you still use it, keep it. |
| `results\interactive_session_log.csv` | A second copy of the activity log. Compare: `fc results\interactive_session_log.csv logs\interactive_session_log.csv`. If it is an old copy, move it to `archive\old_logs\`. It is blocked from upload anyway because it holds your questions. |

### A7. Look for personal paths and secrets in the code
```powershell
Get-ChildItem -Recurse -Filter *.py | Where-Object { $_.FullName -notmatch '\\(venv|archive)\\' } |
    Select-String -Pattern 'D:\\\\','C:\\\\Users','password','api_key','token'
```
Examples in comments or docstrings (such as `C:\photos\diagram.png` in `maincode.py`) are fine. A path used **in code** (for example `open("D:\\...")`) must be changed to a path relative to the file, like the rest of the project does.

### A8. Look for very large files
```powershell
Get-ChildItem -Recurse -File | Where-Object { $_.FullName -notmatch '\\(venv|archive|\.git)\\' } |
    Sort-Object Length -Descending | Select-Object -First 8 FullName, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}
```
GitHub **rejects files over 100 MB**. Your results files should be far smaller. If one is large, tell me before uploading.

### A9. Check that `requirements.txt` is complete
```powershell
python tools\check_requirements.py
```
It scans every `.py` file. If it prints **MISSING**, add those lines to `requirements.txt`. (If `benchmark\` or another script imports something extra, such as `matplotlib`, this is where it shows up.)

### A10. Test that everything still works
```powershell
python tools\check_setup.py          # every line [ OK ] (Ollama must be running)
.\run_web.bat                        # open http://127.0.0.1:8000, ask a question, then Ctrl+C
.\run_cli.bat                        # ask a question, then: image: <path> :: <question>, then quit
```

### A11. Add screenshots for the README
Take 6 screenshots and save them in `docs\images\` with **these exact names** (the README shows them):
| File name | What to show |
|---|---|
| `chat-routing-panel.png` | An answer with the routing panel expanded |
| `manual-model.png` | The Model selector open or the manual note under the box |
| `low-ram-excluded.png` | A routing panel with greyed-out `[excluded]` models |
| `image-tab.png` | The image tab with an attached image and an answer |
| `compare.png` | A Compare result with both models |
| `history.png` | The History tab |
If you skip them, delete the "Screenshots" section from `README.md`, otherwise GitHub shows broken images.

### A12. Fill in the placeholders
- `README.md`: search for `<your-username>`, `<Your Name>`, `<name>`, `<college/university>`.
- `LICENSE`: replace `<Your Name>`.

### A13. The final layout
```
D:\AdaptiveLocalAI2\
|-- README.md  LICENSE  .gitignore  requirements.txt  requirements-lock.txt
|-- setup.ps1  run_web.bat  run_cli.bat  session_logger.py
|-- app\  benchmark\  dataset\  resource_monitor\  results\  router\  webapp\
|-- logs\          (only .gitkeep is uploaded)
|-- tools\         check_setup.py  check_requirements.py  cleanup.py  organize_project.py  ram_check.py  qwen_vl_debug.py
|-- docs\          PROJECT_DOCUMENTATION.md  REPO_GUIDE.md  images\
|-- archive\       (local only: old patches, backups, old logs, private notes)
`-- venv\          (local only)
```

---

## Part B - Every file: keep, upload, move, merge or delete

**Legend:** UPLOAD = goes to GitHub · LOCAL = keep on your PC but never upload · MOVE = organizer moves it · MERGE = combine · DELETE = remove

### B1. Project root
| Item | Decision | Why |
|---|---|---|
| `session_logger.py` | **UPLOAD** (stay in root) | Imported by the CLI and web app; must remain in the top folder |
| `requirements.txt` (new, curated) | **UPLOAD** | What a new machine installs |
| your old `requirements.txt` -> `requirements-lock.txt` | **UPLOAD** (MERGE idea: two files, two purposes) | Exact versions for reproducing your environment |
| `setup.ps1`, `run_web.bat`, `run_cli.bat` | **UPLOAD** | One-click setup and launch |
| `README.md`, `LICENSE`, `.gitignore` | **UPLOAD** | Project front page, license, upload rules |
| `patch_router*.py` (7 files) | **MOVE** to `archive\patches` - **LOCAL** | One-time scripts, already applied; the history is documented |
| `ram_check.py`, `qwen_vl_debug.py` | **MOVE** to `tools\` - **UPLOAD** | Useful diagnostics |
| `PROJECT_DOCUMENTATION.md` | **MOVE** to `docs\` - **UPLOAD** | The full documentation |
| `PROJECT_STATE.md` | **MOVE** to `archive\notes` - **LOCAL** | Private working notes written for continuing the project in a chat; not for visitors |
| `__pycache__\` | **DELETE** | Regenerated automatically |
| `venv\` | **LOCAL** (never upload) | Huge; rebuilt with `setup.ps1` |

### B2. Code folders
| Item | Decision | Why |
|---|---|---|
| `app\maincode.py` | **UPLOAD** | The command-line app |
| `app\test_models.py` | **REVIEW** -> likely archive | Old smoke test; replaced by `tools\check_setup.py` |
| `benchmark\*.py` (4 files) | **UPLOAD** | Benchmark, judge, profile builder |
| `dataset\*.py` | **UPLOAD** | Prompts and train/test sets |
| `resource_monitor\resource_monitor.py` | **UPLOAD** | RAM/CPU/battery reader |
| `router\router.py` | **UPLOAD** | The engine |
| `router\task_classifier.py`, `hybrid_classifier.py`, `ml_classifier.py` | **UPLOAD** | Classification |
| `router\compare_classifiers.py` + `evaluate_classifiers.py` | **UPLOAD both** (do not merge) | Different outputs: the first is the original rule/LogReg/SVM comparison, the second the final hybrid evaluation |
| `router\analyze_routing_impact.py`, `experiment_baselines.py`, `calibrate_cold_warm.py` | **UPLOAD** | Experiments |
| `router\ram_diagnostic.py` | **UPLOAD** (historical) | The early script that found the memory-reclaim delay; `tools\ram_check.py` is its modern successor. Merging is not worth the risk |
| `router\router_BEFORE_*.py` (8 files) | **MOVE** to `archive\router_backups` - **LOCAL** | Safety copies made by patch scripts |
| `router\task_classifier_OLD.py` | **MOVE** to archive - **LOCAL** | Old version |
| `router\__pycache__\` and every other `__pycache__` | **DELETE** | Regenerated |
| `webapp\backend\app.py` | **UPLOAD** | The server |
| `webapp\backend\uploads\*` (6 images) | **DELETE** / **LOCAL** | Private screenshots; only `.gitkeep` is kept |
| `webapp\frontend\index.html`, `style.css`, `app.js` | **UPLOAD** | The web interface |

### B3. Data folders
| Item | Decision | Why |
|---|---|---|
| `results\model_profiles.json`, `results\cold_start_calibration.json` | **UPLOAD - required** | The router reads them; without them it cannot start |
| other `results\*.csv` (benchmark, quality, baselines, classifier, routing impact, ...) | **UPLOAD** | The evidence behind the report; small |
| `results\interactive_session_log.csv` | **REVIEW** -> **LOCAL** | A copy of your questions (private) |
| `logs\interactive_session_log.csv` | **LOCAL** | Holds the text of your questions; `.gitignore` blocks it |
| `logs\interactive_session_log_OLD.csv`, `_MIXED.csv` | **MOVE** to `archive\old_logs` - **LOCAL** | Old column layouts |
| `logs\.gitkeep`, `webapp\backend\uploads\.gitkeep` | **UPLOAD** | Make git keep the empty folders |

### B4. Which files get merged
| Merge | Result |
|---|---|
| `requirements.txt` (freeze) | -> `requirements.txt` (what you need) **+** `requirements-lock.txt` (exact versions) |
| `PROJECT_STATE.md` + `PROJECT_DOCUMENTATION.md` | -> one public document (`docs\PROJECT_DOCUMENTATION.md`); the state file stays private |
| old root `ram_check.py` / `qwen_vl_debug.py` / documentation | -> replaced by the versions in `tools\` and `docs\` (old copies go to `archive\old_versions`) |
| `test_models.py` | -> replaced by `tools\check_setup.py` (if it is only a smoke test) |

### B5. Quick answers
- **Upload:** all source code, `results\` data, `dataset\`, `docs\`, `tools\`, the new top-level files.
- **Keep but never upload:** `venv\`, `archive\`, `logs\*.csv`, `uploads\*`, `PROJECT_STATE.md`.
- **Delete:** `__pycache__\` folders, the uploaded screenshots (when you are done with them).
- **You do not need to delete `archive\`** - `.gitignore` keeps it off GitHub.

---

## Part C - Publish on GitHub, step by step

### C1. Install Git
Download **Git for Windows** from https://git-scm.com/downloads and install it with the default options. Open a **new** PowerShell:
```powershell
git --version
```

### C2. Tell Git who you are (once per computer)
```powershell
git config --global user.name "Your Name"
git config --global user.email "your-github-email@example.com"
```

### C3. Create a GitHub account (skip if you have one)
Go to https://github.com/signup and follow the steps (verify your email).

### C4. Create an empty repository on GitHub
1. Click the **+** at the top right -> **New repository**.
2. **Repository name:** `adaptive-local-ai` (any name works).
3. **Description:** `Hardware-aware router that picks the best local LLM for every question (Ollama, FastAPI, CPU-only).`
4. Choose **Public** (anyone can see it) or **Private**.
5. **Do NOT tick** "Add a README file", ".gitignore" or "license" - your folder already has them.
6. Click **Create repository**. Leave the page open; it shows your repository address, like `https://github.com/<your-username>/adaptive-local-ai.git`.

### C5. Turn the folder into a Git repository
```powershell
cd D:\AdaptiveLocalAI2
git init -b main
```
(If you see an error about `-b`, use `git init` then `git branch -M main`.)

### C6. See exactly what would be uploaded - this is the important check
```powershell
git add .
git status --short
```
**The list must contain:** `README.md`, `LICENSE`, `.gitignore`, `requirements*.txt`, `setup.ps1`, the `.bat` files, `session_logger.py`, and files under `app/`, `router/`, `webapp/`, `benchmark/`, `dataset/`, `resource_monitor/`, `results/`, `tools/`, `docs/`.
**It must NOT contain:** anything under `venv/`, `archive/`, any `.png` in `uploads/`, any `.csv` in `logs/`, `PROJECT_STATE.md`, or `__pycache__`.
If something wrong is listed: `git rm -r --cached <path>`, add it to `.gitignore`, then run `git add .` again. (It is much easier to fix now than after publishing.)

### C7. Make the first commit
```powershell
git commit -m "Initial commit: Adaptive Local AI - hardware-aware local LLM router"
```

### C8. Connect to GitHub and upload
```powershell
git remote add origin https://github.com/<your-username>/adaptive-local-ai.git
git push -u origin main
```
A browser window or Git Credential Manager box asks you to **sign in to GitHub** - do that. If it asks for a password in the terminal, GitHub no longer accepts account passwords; create a **Personal Access Token**: GitHub -> your profile picture -> **Settings** -> **Developer settings** -> **Personal access tokens** -> **Tokens (classic)** -> *Generate new token* -> tick **repo** -> copy it and paste it as the password.

### C9. Check the result
Open `https://github.com/<your-username>/adaptive-local-ai`. You should see the README rendered with the screenshots, the file list, and "MIT license" in the sidebar.

### C10. Make the page look good
- Click the **gear** next to **About**: add the description and **Topics**, for example: `llm` `ollama` `local-llm` `model-routing` `fastapi` `python` `scikit-learn` `adaptive-systems` `btech-project`.
- **Releases -> Create a new release**: tag `v1.0`, title *"Adaptive Local AI v1.0"*, a short description. GitHub attaches the source ZIP automatically.
- Optional: pin the repository on your profile.

### C11. Test it like a new user (recommended)
In another folder:
```powershell
cd $HOME
git clone https://github.com/<your-username>/adaptive-local-ai.git test-clone
cd test-clone
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
.\venv\Scripts\python.exe tools\check_setup.py
```
Follow your own README from the top. Anything confusing is worth fixing in the README.

### C12. Publishing changes later
```powershell
cd D:\AdaptiveLocalAI2
git add .
git status --short          # check again
git commit -m "Describe what you changed"
git push
```

### C13. If something goes wrong
| Message or problem | What to do |
|---|---|
| `File ... is 123 MB; this exceeds GitHub's file size limit of 100 MB` | Remove it from the commit: `git rm --cached <file>`, add it to `.gitignore`, `git commit --amend`, push again |
| `failed to push some refs ... fetch first` / non-fast-forward | The GitHub repository was created with a README. Run `git pull origin main --allow-unrelated-histories`, fix any conflict (keep your README), then `git push` |
| `Support for password authentication was removed` | Use the browser sign-in or a Personal Access Token (step C8) |
| `Permission denied (publickey)` | You used an SSH address; use the HTTPS one shown above |
| `.gitignore` is ignored for a file already added | `git rm -r --cached .` then `git add .` then commit |
| You committed a private file | `git rm --cached <file>`, add it to `.gitignore`, commit, push. If it contained a **password or key**, change that secret - removing it from later commits does not remove it from history |
| `fatal: not a git repository` | You are in the wrong folder, or step C5 was not run |

### C14. No Git? Two alternatives
- **GitHub Desktop** (https://desktop.github.com): *File -> Add local repository* -> choose `D:\AdaptiveLocalAI2` -> *Publish repository*. It shows the list of files to upload and respects `.gitignore`.
- **Browser upload:** on the empty repository page click *uploading an existing file* and drag in the files. Limits: 100 files per upload and no folders of hidden files (`.gitignore`), so it is only suitable for a quick start. Prefer Git.

---

## Before you make it public - checklist
- [ ] `git status --short` shows no `venv`, `archive`, uploads, logs, `PROJECT_STATE.md`.
- [ ] No personal data in text files (names/emails you do not want public): `Get-ChildItem -Recurse -File -Include *.py,*.md,*.json,*.csv | Where-Object { $_.FullName -notmatch '\\(venv|archive)\\' } | Select-String -Pattern 'your-name','your-email'`.
- [ ] README placeholders filled; LICENSE has your name.
- [ ] Screenshots added (or the section removed).
- [ ] `python tools\check_requirements.py` says requirements are complete.
- [ ] A fresh clone passes `tools\check_setup.py` (step C11).
- [ ] You told your supervisor that the project does image **understanding**, not generation.
