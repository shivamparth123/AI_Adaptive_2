"""
FastAPI backend - full version (streaming + image support)

Thin HTTP wrapper around router.py. No routing, scoring, or resource logic is
duplicated here - all of that stays in router.py and is reused directly.

Prerequisites (run once):
    pip install fastapi "uvicorn[standard]" python-multipart --break-system-packages

Two things must be running at the same time:
    1. Ollama:   ollama serve                              (separate terminal)
    2. Backend:  uvicorn app:app --reload --port 8000       (from this folder)

Then open http://127.0.0.1:8000 in a browser.

Endpoints:
    GET  /api/status          - resource state, available modes, live model list
    POST /api/chat/stream     - SSE-style streaming chat (preferred); optional "model" = manual choice
    POST /api/chat            - non-streaming chat (simple fallback / debugging)
    POST /api/image           - single-image vision query
    POST /api/image/compare   - runs both vision models, returns both results
    POST /api/image/upload    - stores an image ONCE, returns an image_id
    POST /api/image/ask       - question about a stored image (+ earlier Q&A as context)
    POST /api/image/compare_saved - both vision models on a stored image
    GET  /api/history         - recent logged interactions (CLI + web, unified)
"""

import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "router"))
import router as R

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))  # project root, for session_logger
from session_logger import LOG_FILE, log_interaction

app = FastAPI(title="Adaptive Local AI - Web Backend")

# Tracks the most recently selected model ACROSS ALL WEB REQUESTS, for model-switch
# logging - mirrors main.py's last_model_used, but as a global since this is a
# single-process, effectively-single-user local app (not meant for multiple
# simultaneous users with separate conversation state).
_last_model_used = None

# Same-origin setup: the frontend is served BY this same FastAPI app (see the
# StaticFiles mount at the bottom), so CORS is not required for normal use.
# Left permissive here only to make local development easier.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8000", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_profiles = None  # loaded once at startup, reused across every request
ALL_MODES = list(R.MODES.keys()) + ["explore"]  # "explore" is deliberately not in
                                                 # R.MODES (see router.py) - mirrors
                                                 # exactly how main.py validates it

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")

# --- stored images (upload once, ask many times) ---
# image_id is "<32 hex>.<ext>" and IS the file name, so ids keep working after a backend
# restart (as long as the file still exists) and can be validated strictly (no path tricks).
IMAGE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}\.(png|jpg|jpeg|gif|webp|bmp)$")
ALLOWED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
CONTENT_TYPE_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/gif": ".gif",
                    "image/webp": ".webp", "image/bmp": ".bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UPLOAD_RETENTION_HOURS = 72  # unused files older than this are deleted at startup

# The vision router only accepts conversation history once router.py has been patched
# (patch_router_vision.py). Detect that instead of crashing on an unexpected argument.
_VISION_HISTORY_SUPPORTED = "history" in inspect.signature(R.route_and_generate_image).parameters

# Manual model choice needs the router patched too (patch_router_manual.py).
_MANUAL_MODEL_SUPPORTED = (
    "forced_model" in inspect.signature(R.route_and_generate_stream).parameters
    and "forced_model" in inspect.signature(R.route_and_generate).parameters
)


def _manual_kwargs(req) -> dict:
    """{"forced_model": name} when the user picked a model manually, else {} (router decides)."""
    if req.model and req.model != "auto":
        return {"forced_model": req.model}
    return {}


def _cleanup_old_uploads():
    cutoff = time.time() - UPLOAD_RETENTION_HOURS * 3600
    try:
        names = os.listdir(UPLOAD_DIR)
    except OSError:
        return
    for name in names:
        path = os.path.join(UPLOAD_DIR, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _resolve_image(image_id: str) -> str:
    if not IMAGE_ID_PATTERN.match(image_id or ""):
        raise HTTPException(status_code=400, detail="Invalid image id")
    path = os.path.join(UPLOAD_DIR, image_id)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Image no longer available on the server - please attach it again")
    try:
        os.utime(path, None)  # using an image keeps it alive
    except OSError:
        pass
    return path


@app.on_event("startup")
def startup():
    global _profiles
    _profiles = R.load_profiles()
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    _cleanup_old_uploads()
    print("Model profiles loaded. Upload directory ready. Backend ready.")


class ChatRequest(BaseModel):
    query: str
    mode: str = "balanced"
    history: list = []  # [{"role": "user"/"assistant", "content": "..."}]
    model: Optional[str] = None  # manual override: run THIS model instead of the router's pick.
                                 # None / "auto" = let the router decide (the default).


def _validate_chat_request(req: ChatRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    if req.mode not in ALL_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown mode '{req.mode}'. Valid: {ALL_MODES}")
    if req.model and req.model != "auto":
        if req.model not in R.LIVE_ROUTER_MODELS:
            raise HTTPException(status_code=400,
                                detail=f"Unknown model '{req.model}'. Valid: {R.LIVE_ROUTER_MODELS}")
        if not _MANUAL_MODEL_SUPPORTED:
            raise HTTPException(status_code=501,
                                detail="Manual model choice needs the router patch - run patch_router_manual.py "
                                       "and restart the backend.")


def _ram_budget(state: dict) -> dict:
    """
    The RAM figure the ROUTER checks models against: free RAM + RAM held by models Ollama would
    unload, then reduced under CPU/battery stress. The UI uses this for its warnings, so a
    manual-model warning matches what the router would really do (raw free RAM alone is
    misleading while a model is still loaded). Falls back to plain free RAM if unavailable.
    """
    free = state.get("ram_available_gb", 0)
    try:
        loaded = R.get_ollama_loaded_gb()
        effective = round(free + loaded, 2)
        budget, reasons = R.compute_stress_adjustment(dict(state, effective_ram_gb=effective))
    except Exception:
        loaded, effective, budget, reasons = 0.0, free, free, []
    try:
        loaded_models = sorted(R.get_warm_models())  # already in memory: they need no NEW RAM
    except Exception:
        loaded_models = []
    return {"ollama_loaded_gb": loaded, "loaded_models": loaded_models, "effective_ram_gb": effective,
            "ram_budget_gb": budget, "stress_reasons": reasons}


@app.get("/api/status")
def get_status():
    """Health check + live resource state, polled periodically by the frontend."""
    try:
        state = R.get_system_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resource monitor failed: {e}")
    return {
        "resource_state": state,
        **_ram_budget(state),
        "modes": ALL_MODES,
        "live_models": R.LIVE_ROUTER_MODELS,
        "manual_model_supported": _MANUAL_MODEL_SUPPORTED,
        # RAM each model needs (incl. the safety buffer), so the UI can warn before a manual pick
        "model_requirements_gb": {
            m: round(R.MODEL_RAM_REQUIREMENT_GB.get(m, 0) + R.RAM_SAFETY_BUFFER_GB, 1)
            for m in R.LIVE_ROUTER_MODELS
        },
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Streaming chat endpoint. Emits Server-Sent-Events-style lines:
        data: {"event": "routing", "data": {...}}\\n\\n
        data: {"event": "token", "text": "..."}\\n\\n           (repeated)
        data: {"event": "done", "data": {...}}\\n\\n
        data: {"event": "error", "message": "..."}\\n\\n        (only on failure)

    Consumed on the frontend via fetch() + a manual ReadableStream reader, not the
    browser's native EventSource object - EventSource only supports GET requests,
    and history/mode need to travel as a JSON body (POST), so manual SSE parsing
    on the client is the simpler, more flexible choice here.
    """
    _validate_chat_request(req)

    # An async generator that pulls the (blocking) router generator one item at a time in a worker
    # thread. This matters for the frontend's STOP button: when the browser aborts the request,
    # Starlette cancels this generator and the `finally` below closes the router generator, which
    # closes the HTTP stream to Ollama - and Ollama then stops generating. (A plain sync generator
    # was only abandoned, never closed, so the Ollama connection stayed open.)
    async def event_generator():
        global _last_model_used
        gen = R.route_and_generate_stream(req.query, _profiles, mode=req.mode,
                                          history=req.history, **_manual_kwargs(req))
        sentinel = object()

        def pull():
            return next(gen, sentinel)

        try:
            while True:
                item = await run_in_threadpool(pull)
                if item is sentinel:
                    break
                if item["event"] == "done":
                    log_interaction(item["data"], previous_model=_last_model_used, source="web")
                    _last_model_used = item["data"].get("selected_model")
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        finally:
            gen.close()  # normal end, error, AND client disconnect (Stop button)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Non-streaming chat - simple fallback, returns the full result in one response."""
    global _last_model_used
    _validate_chat_request(req)
    try:
        result = R.route_and_generate(req.query, _profiles, mode=req.mode, history=req.history,
                                      **_manual_kwargs(req))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")
    log_interaction(result, previous_model=_last_model_used, source="web")
    _last_model_used = result.get("selected_model")
    return result


def _save_upload(image: UploadFile) -> str:
    suffix = os.path.splitext(image.filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=UPLOAD_DIR) as tmp:
        shutil.copyfileobj(image.file, tmp)
        return tmp.name


@app.post("/api/image")
async def image_chat(question: str = Form(...), image: UploadFile = File(...)):
    """
    Single-image vision query. Not streaming - route_and_generate_image is
    blocking (vision models aren't wired for streaming in router.py yet).
    """
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    global _last_model_used
    tmp_path = _save_upload(image)
    try:
        result = R.route_and_generate_image(tmp_path, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    if result.get("response_text") is None:
        raise HTTPException(status_code=422, detail=result.get("selection_reason", "Image query failed"))

    log_interaction(result, previous_model=_last_model_used, source="web")
    _last_model_used = result.get("selected_model")
    return result


@app.post("/api/image/compare")
async def image_compare(question: str = Form(...), image: UploadFile = File(...)):
    """Runs BOTH vision models on the same image+question, returns both results."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    global _last_model_used
    tmp_path = _save_upload(image)
    try:
        result = R.generate_vision_comparison(tmp_path, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image comparison failed: {e}")

    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])

    _log_comparison(question, result)

    return result


def _log_comparison(question: str, result: dict):
    """One log row per vision model, like the CLI's imagecompare."""
    global _last_model_used
    for model, r in result["results_by_model"].items():
        log_interaction({
            "query": question, "mode": "vision_comparison", "category": "image_understanding",
            "resource_state": {"ram_available_gb": r["ram_available_gb"],
                                "effective_ram_gb": r["effective_ram_gb"]},
            "selected_model": model,
            "actual_response_time_sec": r["actual_response_time_sec"],
            "ran_under_ram_constraint": r["ran_under_ram_constraint"],
        }, previous_model=_last_model_used, source="web")
        _last_model_used = model


class ImageAskRequest(BaseModel):
    image_id: str
    question: str
    history: list = []  # earlier [{"role", "content"}] turns about THIS image


class ImageCompareSavedRequest(BaseModel):
    image_id: str
    question: str


# NOTE: the three endpoints below are plain `def` (not `async def`) on purpose: FastAPI then runs
# them in a worker thread, so a long vision generation does not freeze /api/status polling.
@app.post("/api/image/upload")
async def image_upload(image: UploadFile = File(...)):
    """Stores an image once and returns its id; later questions reference the id, not the bytes."""
    ext = os.path.splitext(image.filename or "")[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        ext = CONTENT_TYPE_EXT.get((image.content_type or "").lower(), "")  # e.g. pasted images
    if not ext:
        raise HTTPException(status_code=415, detail="Unsupported file type - use PNG, JPG, GIF, WEBP or BMP")
    data = await image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large (limit 20 MB)")
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    image_id = uuid.uuid4().hex + ext
    with open(os.path.join(UPLOAD_DIR, image_id), "wb") as f:
        f.write(data)
    return {"image_id": image_id, "filename": image.filename, "size_bytes": len(data)}


@app.post("/api/image/ask")
def image_ask(req: ImageAskRequest):
    """Question about a stored image. Earlier Q&A is passed as context when the router supports it."""
    global _last_model_used
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    path = _resolve_image(req.image_id)
    kwargs = {"history": req.history} if (_VISION_HISTORY_SUPPORTED and req.history) else {}
    try:
        result = R.route_and_generate_image(path, req.question, **kwargs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")

    if result.get("response_text") is None:
        raise HTTPException(status_code=422, detail=result.get("selection_reason", "Image query failed"))

    result["image_id"] = req.image_id
    result["context_supported"] = _VISION_HISTORY_SUPPORTED
    log_interaction(result, previous_model=_last_model_used, source="web")
    _last_model_used = result.get("selected_model")
    return result


@app.post("/api/image/compare_saved")
def image_compare_saved(req: ImageCompareSavedRequest):
    """Runs BOTH vision models on a stored image."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    path = _resolve_image(req.image_id)
    try:
        result = R.generate_vision_comparison(path, req.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image comparison failed: {e}")
    if result.get("error"):
        raise HTTPException(status_code=422, detail=result["error"])
    _log_comparison(req.question, result)
    return result


@app.get("/api/history")
def get_history(limit: int = 30):
    """
    Returns recent logged interactions, UNIFIED across both the CLI (main.py) and
    this web backend - both write to the same file via session_logger.py, tagged
    with a "source" column ("cli" or "web") so the frontend can distinguish them.
    """
    if not os.path.exists(LOG_FILE):
        return {"rows": [], "total_logged": 0}
    try:
        # on_bad_lines="skip": the log file may contain rows from an older schema
        # (different column count); skip those instead of failing the whole endpoint.
        df = pd.read_csv(LOG_FILE, on_bad_lines="skip", engine="python")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read log file: {e}")

    df = df.fillna("")
    rows = df.tail(limit).to_dict(orient="records")
    rows.reverse()  # most recent first
    return {"rows": rows, "total_logged": len(df)}


@app.get("/favicon.ico")
def favicon():
    """
    Minimal response so browsers requesting /favicon.ico don't generate a 404 on
    every page load. Not a real icon file - just a quick 204 (No Content), which
    browsers accept without complaint and don't retry endlessly.
    """
    return StreamingResponse(iter([b""]), status_code=204)


# Serves the frontend from this same server, at "/". Must be registered AFTER all
# /api/* routes above, so those take priority over this catch-all static handler.
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")