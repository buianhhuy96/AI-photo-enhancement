"""FastAPI endpoints — bridges React frontend to backend services.

Images are NEVER copied. The server stores original file paths and reads
from disk on demand (thumbnails, previews, AI processing).
"""
import io
import base64
import uuid
import json
import queue
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from backend.config import SUPPORTED_EXTENSIONS
from backend.services import (
    process_single_image,
    blend_preview,
    resize_image,
)

THUMBNAIL_SIZE = (120, 80)

app = FastAPI(title="AI Photo Enhancer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory session storage ──
_sessions: dict = {}

# ── Async job storage ──
_jobs: dict = {}  # job_id → {"status": "processing"|"done"|"error", "result": ..., "error": ...}

# ── Persistent import cache ──
CACHE_FILE = Path(".cache/imported_files.json")


def _load_cache() -> dict:
    """Load cached session from disk."""
    if CACHE_FILE.exists():
        try:
            data = json.loads(CACHE_FILE.read_text())
            # Filter out files that no longer exist
            data["files"] = [f for f in data.get("files", []) if Path(f).exists()]
            return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def _save_cache(session_id: str, files: list[str]):
    """Persist imported file list to disk."""
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps({
        "session_id": session_id,
        "files": files,
    }, indent=2))

# Preview cap — scale large images down for the viewer
PREVIEW_MAX = 1600

# Server-side preview cache: (session_id, index, file_path) → data URL
_preview_cache: dict[tuple, str] = {}


# ── Request models ──

class ImportRequest(BaseModel):
    path: str
    session_id: Optional[str] = None


class ImportFilesRequest(BaseModel):
    files: list[str]
    session_id: Optional[str] = None


class IndicesRequest(BaseModel):
    indices: list[int]


class PipelineStep(BaseModel):
    name: str  # "resize", "reflection", "restore"
    params: dict = {}


class PipelineRequest(BaseModel):
    steps: list[PipelineStep]


# ── Helpers ──

def _img_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _img_to_data_url(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}.get(fmt, "image/jpeg")
    return f"data:{mime};base64,{_img_to_base64(img, fmt, quality)}"


def _collect_images(path_str: str) -> list[str]:
    """Collect image file paths from a file or directory."""
    p = Path(path_str).expanduser().resolve()
    if not p.exists():
        return []
    if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS:
        return [str(p)]
    if p.is_dir():
        return sorted(
            str(f.resolve())
            for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    return []


def _make_thumbnail(file_path: str) -> dict:
    """Generate a thumbnail data URL for one image (no copy)."""
    img = Image.open(file_path).convert("RGB")
    w, h = img.size
    img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
    return {
        "name": Path(file_path).stem,
        "url": _img_to_data_url(img, "JPEG", 70),
        "aspect": round(w / h, 3),
    }


def _session_thumbnails(session) -> list[dict]:
    return [_make_thumbnail(f) for f in session["files"]]


# ── Endpoints ──

@app.get("/api/restore")
async def restore_session():
    """Restore previously imported session from .cache."""
    cached = _load_cache()
    if not cached or not cached["files"]:
        return {"restored": False}

    session_id = cached["session_id"]
    if session_id not in _sessions:
        _sessions[session_id] = {
            "files": cached["files"],
            "processed": {},
            "original": {},
        }

    session = _sessions[session_id]
    return {
        "restored": True,
        "session_id": session_id,
        "count": len(session["files"]),
        "thumbnails": _session_thumbnails(session),
    }


@app.get("/api/listdir")
async def list_directory(path: str = "~"):
    """List subdirectories for the folder tree."""
    p = Path(path).expanduser().resolve()
    if not p.exists() or not p.is_dir():
        raise HTTPException(400, f"Not a valid directory: {path}")

    dirs = []
    try:
        for item in sorted(p.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                # Count images inside
                img_count = sum(1 for f in item.iterdir()
                                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
                dirs.append({
                    "name": item.name,
                    "path": str(item),
                    "imageCount": img_count,
                })
    except PermissionError:
        pass

    return {
        "current": str(p),
        "parent": str(p.parent) if p != p.parent else None,
        "dirs": dirs,
    }


@app.get("/api/browse")
async def browse_folder(path: str):
    """List images in a folder with thumbnails for the import picker."""
    files = _collect_images(path)
    if not files:
        raise HTTPException(400, f"No supported images found at: {path}")
    items = []
    for f in files:
        img = Image.open(f).convert("RGB")
        img.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)
        items.append({
            "path": f,
            "name": Path(f).name,
            "url": _img_to_data_url(img, "JPEG", 60),
        })
    return {"path": path, "count": len(items), "images": items}

@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...), session_id: Optional[str] = Form(None)):
    """Upload images via drag-and-drop. Saves to a temp dir, stores paths."""
    if session_id and session_id in _sessions:
        upload_dir = Path("uploads") / session_id
    else:
        session_id = str(uuid.uuid4())
        upload_dir = Path("uploads") / session_id
        _sessions[session_id] = {
            "files": [],
            "processed": {},
            "original": {},
        }

    upload_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for f in files:
        dest = upload_dir / f.filename
        content = await f.read()
        dest.write_bytes(content)
        new_paths.append(str(dest.resolve()))

    existing = set(_sessions[session_id]["files"])
    new_paths = [p for p in new_paths if p not in existing]
    _sessions[session_id]["files"].extend(new_paths)

    session = _sessions[session_id]
    return {
        "session_id": session_id,
        "count": len(session["files"]),
        "added": len(new_paths),
        "thumbnails": _session_thumbnails(session),
    }


@app.post("/api/import")
async def import_images(req: ImportRequest):
    """Import images by path. No copying — stores original paths only."""
    new_files = _collect_images(req.path)
    if not new_files:
        raise HTTPException(400, f"No supported images found at: {req.path}")

    session_id = req.session_id
    if session_id and session_id in _sessions:
        existing = set(_sessions[session_id]["files"])
        new_files = [f for f in new_files if f not in existing]
        if not new_files:
            session = _sessions[session_id]
            return {
                "session_id": session_id,
                "count": len(session["files"]),
                "added": 0,
                "thumbnails": _session_thumbnails(session),
            }
        _sessions[session_id]["files"].extend(new_files)
    else:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "files": new_files,
            "processed": {},
            "original": {},
        }

    session = _sessions[session_id]
    return {
        "session_id": session_id,
        "count": len(session["files"]),
        "added": len(new_files),
        "thumbnails": _session_thumbnails(session),
    }


@app.post("/api/import-files")
async def import_selected_files(req: ImportFilesRequest):
    """Import specific files by path. Zero-copy — stores paths only."""
    # Validate all paths exist and are images
    valid = [f for f in req.files if Path(f).exists() and Path(f).suffix.lower() in SUPPORTED_EXTENSIONS]
    if not valid:
        raise HTTPException(400, "No valid image files in selection")

    session_id = req.session_id
    if session_id and session_id in _sessions:
        existing = set(_sessions[session_id]["files"])
        valid = [f for f in valid if f not in existing]
        _sessions[session_id]["files"].extend(valid)
    else:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {
            "files": valid,
            "processed": {},
            "original": {},
        }

    session = _sessions[session_id]
    _save_cache(session_id, session["files"])
    return {
        "session_id": session_id,
        "count": len(session["files"]),
        "added": len(valid),
        "thumbnails": _session_thumbnails(session),
    }


@app.get("/api/image/{session_id}/{index}")
async def get_image(session_id: str, index: int):
    """Get a preview-sized image (reads from original path, no copy)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")
    files = session["files"]
    if index < 0 or index >= len(files):
        raise HTTPException(404, "Image index out of range")

    file_path = files[index]
    if not Path(file_path).exists():
        raise HTTPException(404, f"File no longer exists: {file_path}")

    cache_key = (session_id, index, file_path)
    if cache_key in _preview_cache:
        return _preview_cache[cache_key]

    img = Image.open(file_path).convert("RGB")
    orig_w, orig_h = img.size
    if max(img.size) > PREVIEW_MAX:
        img.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    url = _img_to_data_url(img, "JPEG", 90)
    result = {"url": url, "width": orig_w, "height": orig_h}
    _preview_cache[cache_key] = result
    return result


# ── Pipeline cache: (session_id, index) → { steps_hash → PIL.Image }
_pipeline_cache: dict[tuple, dict] = {}


def _pipeline_hash(steps: list[PipelineStep], up_to: int) -> str:
    """Hash the first N steps + params for cache lookup."""
    import hashlib
    key = json.dumps([{"name": s.name, "params": s.params} for s in steps[:up_to]])
    return hashlib.md5(key.encode()).hexdigest()


@app.post("/api/pipeline/{session_id}/{index}")
async def run_pipeline(session_id: str, index: int, req: PipelineRequest):
    """Start a pipeline job. Returns job_id immediately for polling."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    files = session["files"]
    if index < 0 or index >= len(files):
        raise HTTPException(404, "Image index out of range")

    file_path = files[index]
    if not Path(file_path).exists():
        raise HTTPException(404, f"File no longer exists: {file_path}")

    # Check cache — if all steps are cached, return instantly
    cache_key = (session_id, index)
    if cache_key in _pipeline_cache:
        final_hash = _pipeline_hash(req.steps, len(req.steps))
        if final_hash in _pipeline_cache[cache_key]:
            cached_img = _pipeline_cache[cache_key][final_hash]
            url = _img_to_data_url(cached_img)
            return {"status": "done", "url": url}

    # Create job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "processing", "result": None, "error": None}

    # Run in background thread
    def run_job():
        try:
            result = _execute_pipeline(session_id, index, file_path, req.steps)
            _jobs[job_id]["result"] = result
            _jobs[job_id]["status"] = "done"
        except Exception as e:
            _jobs[job_id]["error"] = str(e)
            _jobs[job_id]["status"] = "error"

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/pipeline/status/{job_id}")
async def get_pipeline_status(job_id: str):
    """Poll for pipeline job result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    if job["status"] == "done":
        result = job["result"]
        # Clean up job
        del _jobs[job_id]
        return {"status": "done", **result}
    elif job["status"] == "error":
        error = job["error"]
        del _jobs[job_id]
        raise HTTPException(500, error)
    else:
        return {"status": "processing"}


def _execute_pipeline(session_id: str, index: int, file_path: str, steps):
    """Execute pipeline steps synchronously (called from background thread)."""
    session = _sessions[session_id]
    mock_mode = _sessions.get("__mock_mode__", True)
    engine = _sessions.get("__engine__")
    restore_engine = _sessions.get("__restore_engine__")
    upscale_engine = _sessions.get("__upscale_engine__")

    # Get or create cache for this image
    cache_key = (session_id, index)
    if cache_key not in _pipeline_cache:
        _pipeline_cache[cache_key] = {}
    cache = _pipeline_cache[cache_key]

    # Load original
    img = Image.open(file_path).convert("RGB")

    # Walk through steps, using cache where possible
    for i, step in enumerate(steps):
        step_hash = _pipeline_hash(steps, i + 1)

        if step_hash in cache:
            img = cache[step_hash].copy()
            continue

        # Apply step
        if step.name == "resize":
            w = int(step.params.get("width", img.size[0]))
            h = int(step.params.get("height", img.size[1]))
            denoise_str = float(step.params.get("denoise_strength", 0.5))
            if mock_mode:
                img = img.resize((w, h), Image.LANCZOS)
            else:
                img = resize_image(img, w, h, upscale_engine, denoise_str)

        elif step.name == "reflection":
            quality = int(step.params.get("quality", 0))
            strength = float(step.params.get("strength", 0.5))
            use_4bit = bool(step.params.get("use_4bit", True))
            if mock_mode:
                import time
                time.sleep(0.3)
            else:
                print(f"[pipeline] Reflection: quality={quality}, strength={strength}, use_4bit={use_4bit}, img_size={img.size}")
                _, original, processed, status_msg = process_single_image(
                    engine, img, quality, 1.0, use_4bit, "png", 95,
                    mock_mode=False,
                )
                print(f"[pipeline] Result: processed={'OK' if processed is not None else 'NONE'}, status={status_msg}")
                if processed is not None:
                    # Check if processed differs from original
                    import numpy as np
                    orig_arr = np.array(original)
                    proc_arr = np.array(processed)
                    diff = np.abs(orig_arr.astype(float) - proc_arr.astype(float)).mean()
                    print(f"[pipeline] Mean pixel diff: {diff:.2f} (0 = identical)")
                    img = blend_preview(strength, original, processed)
                else:
                    print(f"[pipeline] WARNING: processed is None, keeping original")
                session["original"][index] = original
                session["processed"][index] = processed

        elif step.name == "restore":
            denoise_level = float(step.params.get("denoise", 0))
            deblur_level = float(step.params.get("deblur", 0))
            if mock_mode:
                import time
                time.sleep(0.3)
            else:
                if restore_engine is None:
                    raise Exception("Restore engine not initialized")
                if denoise_level > 0:
                    img = restore_engine.denoise(img, denoise_level)
                if deblur_level > 0:
                    img = restore_engine.deblur(img, deblur_level)

        # Cache intermediate
        cache[step_hash] = img.copy()

    # Generate preview
    preview = img.copy()
    if max(preview.size) > PREVIEW_MAX:
        preview.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)

    url = _img_to_data_url(preview, "JPEG", 90)
    print(f"[pipeline] Returning preview: size={preview.size}, url_len={len(url)}")
    return {"url": url}


@app.post("/api/resize/{session_id}/{index}")
async def resize_preview(session_id: str, index: int, width: int, height: int, denoise_strength: float = 0.5):
    """Resize image and return a preview. Uses AI upscaling for enlargements."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    files = session["files"]
    if index < 0 or index >= len(files):
        raise HTTPException(404, "Image index out of range")

    if width < 1 or height < 1 or width > 20000 or height > 20000:
        raise HTTPException(400, "Invalid dimensions")

    file_path = files[index]
    if not Path(file_path).exists():
        raise HTTPException(404, f"File no longer exists: {file_path}")

    img = Image.open(file_path).convert("RGB")
    mock_mode = _sessions.get("__mock_mode__", True)

    if mock_mode:
        # In mock mode, just use LANCZOS for both up and down
        resized = img.resize((width, height), Image.LANCZOS)
    else:
        upscale_engine = _sessions.get("__upscale_engine__")
        resized = resize_image(img, width, height, upscale_engine, denoise_strength)

    # Store in session for export
    session.setdefault("resized", {})[index] = (width, height, denoise_strength)

    # Generate preview
    if max(resized.size) > PREVIEW_MAX:
        preview = resized.copy()
        preview.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    else:
        preview = resized
    return {"url": _img_to_data_url(preview, "JPEG", 90), "width": width, "height": height}


@app.post("/api/remove/{session_id}")
async def remove_images(session_id: str, req: IndicesRequest):
    """Remove images from session (library only, files stay on disk)."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    for idx in sorted(req.indices, reverse=True):
        if 0 <= idx < len(session["files"]):
            session["files"].pop(idx)
    session["processed"] = {}
    session["original"] = {}
    # Invalidate preview cache for this session
    _preview_cache.clear()
    _save_cache(session_id, session["files"])

    return {
        "count": len(session["files"]),
        "thumbnails": _session_thumbnails(session),
    }


@app.post("/api/delete/{session_id}")
async def delete_images(session_id: str, req: IndicesRequest):
    """Remove from session AND delete files from disk."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    paths_to_delete = []
    for idx in sorted(req.indices, reverse=True):
        if 0 <= idx < len(session["files"]):
            paths_to_delete.append(session["files"].pop(idx))

    for p in paths_to_delete:
        path = Path(p)
        if path.exists():
            path.unlink()

    session["processed"] = {}
    session["original"] = {}
    _preview_cache.clear()
    _save_cache(session_id, session["files"])

    return {
        "count": len(session["files"]),
        "thumbnails": _session_thumbnails(session),
    }


@app.post("/api/process/{session_id}/{index}")
async def process_image(
    session_id: str,
    index: int,
    quality: int = 0,
    strength: float = 0.5,
    use_4bit: bool = True,
    output_format: str = "png",
    jpg_quality: int = 95,
):
    """Process one image. Uses original path on disk for AI input."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    files = session["files"]
    if index < 0 or index >= len(files):
        raise HTTPException(404, "Image index out of range")

    img = Image.open(files[index]).convert("RGB")

    engine = _sessions.get("__engine__")
    mock_mode = _sessions.get("__mock_mode__", True)

    preview, original, processed, status_msg = process_single_image(
        engine, img, quality, strength, use_4bit, output_format, jpg_quality,
        mock_mode=mock_mode,
    )

    if processed is None:
        return {"status": status_msg, "result": None}

    session["original"][index] = original
    session["processed"][index] = processed

    return {
        "status": status_msg,
        "result": _img_to_data_url(preview, "JPEG", 90),
    }


@app.post("/api/blend/{session_id}/{index}")
async def blend(session_id: str, index: int, strength: float = 1.0):
    """Live strength blending between original and processed."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    original = session["original"].get(index)
    processed = session["processed"].get(index)
    if original is None or processed is None:
        raise HTTPException(400, "Image not processed yet")

    result = blend_preview(strength, original, processed)
    return {"result": _img_to_data_url(result, "JPEG", 90)}


@app.post("/api/export/{session_id}")
async def export_all(
    session_id: str,
    quality: int = 0,
    strength: float = 0.5,
    use_4bit: bool = True,
    output_format: str = "png",
    jpg_quality: int = 95,
):
    """Export all images with SSE progress streaming. Uses pipeline cache when available."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    engine = _sessions.get("__engine__")
    mock_mode = _sessions.get("__mock_mode__", True)
    q = queue.Queue()

    def progress_cb(fraction, description):
        q.put(json.dumps({"progress": fraction, "message": description}))

    def run_export():
        file_list = session["files"]
        if not file_list:
            q.put(json.dumps({"progress": 1.0, "message": "No images imported.", "done": True}))
            q.put(None)
            return

        export_dir = Path("exports")
        export_dir.mkdir(exist_ok=True)
        total = len(file_list)
        lines = []

        for idx, img_path in enumerate(file_list):
            name = Path(img_path).stem
            progress_cb((idx) / total, f"[{idx+1}/{total}] {name}...")

            # Check if we have a cached pipeline result for this image
            cache_key = (session_id, idx)
            cached_img = None
            if cache_key in _pipeline_cache and _pipeline_cache[cache_key]:
                # Get the most complete cached result (longest step hash)
                best = max(_pipeline_cache[cache_key].items(), key=lambda x: len(x[0]))
                cached_img = best[1]

            if cached_img is not None:
                # Use cached result — no need to re-run AI
                out_path = export_dir / f"{name}_clean.{output_format}"
                if output_format == "jpg":
                    cached_img.save(str(out_path), "JPEG", quality=jpg_quality)
                elif output_format == "webp":
                    cached_img.save(str(out_path), "WEBP", quality=jpg_quality)
                else:
                    cached_img.save(str(out_path), "PNG")
                lines.append(f"[{idx+1}/{total}] Done: {name} (cached)")
            elif mock_mode:
                import time
                time.sleep(0.3)
                out_path = export_dir / f"{name}_clean.{output_format}"
                Image.open(img_path).convert("RGB").save(str(out_path))
                lines.append(f"[{idx+1}/{total}] {name} (mock)")
            else:
                # No cache — export the original (don't re-run GPU processing)
                out_path = export_dir / f"{name}_clean.{output_format}"
                img = Image.open(img_path).convert("RGB")
                if output_format == "jpg":
                    img.save(str(out_path), "JPEG", quality=jpg_quality)
                elif output_format == "webp":
                    img.save(str(out_path), "WEBP", quality=jpg_quality)
                else:
                    img.save(str(out_path), "PNG")
                lines.append(f"[{idx+1}/{total}] {name} (original — not yet processed)")

            progress_cb((idx + 1) / total, f"[{idx+1}/{total}] {name} done")

        status = f"Exported {total} images \u2192 exports/\n" + "\n".join(lines)
        q.put(json.dumps({"progress": 1.0, "message": status, "done": True}))
        q.put(None)

    def event_stream():
        thread = threading.Thread(target=run_export, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/api/export/download/{session_id}")
async def download_exports(session_id: str):
    """Download all exported files as a zip."""
    import zipfile
    export_dir = Path("exports")
    if not export_dir.exists() or not list(export_dir.iterdir()):
        raise HTTPException(404, "No exported files found")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(export_dir.iterdir()):
            if f.is_file():
                zf.write(f, f.name)
    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=exports.zip"}
    )


@app.post("/api/denoise/{session_id}/{index}")
async def denoise_image(session_id: str, index: int, strength: float = 1.0):
    """Denoise image using NAFNet and return preview."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    files = session["files"]
    if index < 0 or index >= len(files):
        raise HTTPException(404, "Image index out of range")

    file_path = files[index]
    if not Path(file_path).exists():
        raise HTTPException(404, f"File no longer exists: {file_path}")

    restore_engine = _sessions.get("__restore_engine__")
    mock_mode = _sessions.get("__mock_mode__", True)

    img = Image.open(file_path).convert("RGB")

    if mock_mode:
        import time
        time.sleep(0.5)
        result = img.copy()
    else:
        if restore_engine is None:
            raise HTTPException(500, "Restore engine not initialized")
        result = restore_engine.denoise(img, strength)

    # Generate preview
    if max(result.size) > PREVIEW_MAX:
        result.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return {"url": _img_to_data_url(result, "JPEG", 90)}


@app.post("/api/deblur/{session_id}/{index}")
async def deblur_image(session_id: str, index: int, strength: float = 1.0):
    """Deblur image using NAFNet and return preview."""
    session = _sessions.get(session_id)
    if not session:
        raise HTTPException(404, "Session not found")

    files = session["files"]
    if index < 0 or index >= len(files):
        raise HTTPException(404, "Image index out of range")

    file_path = files[index]
    if not Path(file_path).exists():
        raise HTTPException(404, f"File no longer exists: {file_path}")

    restore_engine = _sessions.get("__restore_engine__")
    mock_mode = _sessions.get("__mock_mode__", True)

    img = Image.open(file_path).convert("RGB")

    if mock_mode:
        import time
        time.sleep(0.5)
        result = img.copy()
    else:
        if restore_engine is None:
            raise HTTPException(500, "Restore engine not initialized")
        result = restore_engine.deblur(img, strength)

    # Generate preview
    if max(result.size) > PREVIEW_MAX:
        result.thumbnail((PREVIEW_MAX, PREVIEW_MAX), Image.LANCZOS)
    return {"url": _img_to_data_url(result, "JPEG", 90)}


@app.get("/api/settings/status")
async def get_settings_status():
    """Return environment, package, and model weight status."""
    import sys
    import shutil
    import subprocess

    # Check PyTorch
    torch_version = None
    cuda_available = False
    gpu_name = None
    gpu_vram = None
    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram = f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    except ImportError:
        pass

    # Check required packages
    packages = {}
    required = {
        "torch": "PyTorch (GPU inference)",
        "torchvision": "TorchVision (image transforms)",
        "transformers": "Transformers (model loading)",
        "diffusers": "Diffusers (pipeline)",
        "accelerate": "Accelerate (device mgmt)",
        "peft": "PEFT (LoRA adapter)",
        "safetensors": "SafeTensors (weight loading)",
        "basicsr": "BasicSR (NAFNet restore)",
        "realesrgan": "Real-ESRGAN (upscaling)",
        "huggingface_hub": "HuggingFace Hub (downloads)",
        "bitsandbytes": "BitsAndBytes (4-bit quant)",
        "PIL": "Pillow (image processing)",
        "numpy": "NumPy (array processing)",
        "imageio": "ImageIO (image I/O)",
        "tqdm": "tqdm (progress bars)",
        "fastapi": "FastAPI (server)",
        "uvicorn": "Uvicorn (ASGI server)",
    }
    for pkg, desc in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "installed")
            packages[pkg] = {"name": desc, "installed": True, "version": ver}
        except ImportError:
            packages[pkg] = {"name": desc, "installed": False, "version": None}

    # Check HuggingFace token
    import os
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    hf_token_set = bool(hf_token)
    # Also check huggingface-cli login
    if not hf_token_set:
        hf_token_file = Path.home() / ".cache" / "huggingface" / "token"
        hf_token_set = hf_token_file.exists()

    # Check which models are available
    models = {}

    # Reflection removal LoRA
    try:
        from huggingface_hub import try_to_load_from_cache
        lora_path = try_to_load_from_cache(
            "huawei-bayerlab/windowseat-reflection-removal-v1-0",
            "pytorch_lora_weights.safetensors",
            subfolder="transformer_lora",
        )
        models["reflection_lora"] = {"name": "Reflection Removal (LoRA)", "size": "~500MB", "downloaded": lora_path is not None and lora_path != "NOT_FOUND"}
    except Exception:
        models["reflection_lora"] = {"name": "Reflection Removal (LoRA)", "size": "~500MB", "downloaded": False}

    # Base model - Transformer
    try:
        from huggingface_hub import try_to_load_from_cache
        tf_path = try_to_load_from_cache(
            "Qwen/Qwen-Image-Edit-2509",
            "config.json",
            subfolder="transformer",
        )
        models["qwen_transformer"] = {"name": "Qwen Image Edit Transformer", "size": "~8GB", "downloaded": tf_path is not None and tf_path != "NOT_FOUND"}
    except Exception:
        models["qwen_transformer"] = {"name": "Qwen Image Edit Transformer", "size": "~8GB", "downloaded": False}

    # Base model - VAE
    try:
        from huggingface_hub import try_to_load_from_cache
        vae_path = try_to_load_from_cache(
            "Qwen/Qwen-Image-Edit-2509",
            "config.json",
            subfolder="vae",
        )
        models["qwen_vae"] = {"name": "Qwen Image Edit VAE", "size": "~300MB", "downloaded": vae_path is not None and vae_path != "NOT_FOUND"}
    except Exception:
        models["qwen_vae"] = {"name": "Qwen Image Edit VAE", "size": "~300MB", "downloaded": False}

    # NAFNet denoise/deblur
    try:
        from huggingface_hub import try_to_load_from_cache
        nafnet_path = try_to_load_from_cache(
            "piddnad/nafnet-denoise-deblur-weights",
            "NAFNet-SIDD-width64.pth",
        )
        models["nafnet"] = {"name": "NAFNet (Denoise/Deblur)", "size": "~260MB", "downloaded": nafnet_path is not None and nafnet_path != "NOT_FOUND"}
    except Exception:
        models["nafnet"] = {"name": "NAFNet (Denoise/Deblur)", "size": "~260MB", "downloaded": False}

    # Real-ESRGAN upscale
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        models["realesrgan"] = {"name": "Real-ESRGAN (Upscaling)", "size": "~60MB", "downloaded": True}
    except Exception:
        models["realesrgan"] = {"name": "Real-ESRGAN (Upscaling)", "size": "~60MB", "downloaded": False}

    # Environment info
    env = {
        "python_version": sys.version.split()[0],
        "torch_version": torch_version or "Not installed",
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "gpu_vram": gpu_vram,
        "mock_mode": _sessions.get("__mock_mode__", True),
        "disk_free": f"{shutil.disk_usage('.').free / 1e9:.1f} GB",
        "hf_token_set": hf_token_set,
    }

    return {"env": env, "models": models, "packages": packages}


@app.post("/api/settings/install-packages")
async def install_packages():
    """Install all required Python packages. Streams progress via SSE."""
    q = queue.Queue()

    def do_install():
        import subprocess, sys
        pip = [sys.executable, "-m", "pip", "install", "--quiet"]

        steps = [
            ("PyTorch (CUDA 12.4)", [*pip, "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124"]),
            ("Transformers + Diffusers", [*pip, "transformers", "diffusers", "accelerate", "peft", "safetensors"]),
            ("HuggingFace Hub", [*pip, "huggingface_hub"]),
            ("BasicSR + Real-ESRGAN", [*pip, "basicsr", "realesrgan"]),
            ("BitsAndBytes", [*pip, "bitsandbytes"]),
            ("Core libraries", [*pip, "Pillow", "numpy", "imageio", "tqdm"]),
        ]

        for i, (label, cmd) in enumerate(steps):
            progress = i / len(steps)
            q.put(json.dumps({"progress": progress, "message": f"Installing {label}..."}))
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                q.put(json.dumps({"progress": progress, "message": f"Error installing {label}: {e.stderr[:200]}"}))

        q.put(json.dumps({"progress": 1.0, "message": "All packages installed", "done": True}))
        q.put(None)

    def event_stream():
        thread = threading.Thread(target=do_install, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/settings/set-hf-token")
async def set_hf_token(token: str):
    """Save HuggingFace token for model downloads."""
    import os
    os.environ["HF_TOKEN"] = token
    # Also persist to huggingface cache
    try:
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(token)
        return {"status": "ok", "message": "Token saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/api/settings/download")
async def download_model(model_id: str):
    """Download a specific model weight. Streams progress via SSE."""
    q = queue.Queue()

    def do_download():
        try:
            from huggingface_hub import snapshot_download
            if model_id == "reflection_lora":
                q.put(json.dumps({"progress": 0.1, "message": "Downloading Reflection Removal LoRA..."}))
                snapshot_download(
                    "huawei-bayerlab/windowseat-reflection-removal-v1-0",
                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.md"],
                )
                q.put(json.dumps({"progress": 1.0, "message": "Reflection Removal LoRA downloaded", "done": True}))
            elif model_id == "qwen_transformer":
                q.put(json.dumps({"progress": 0.1, "message": "Downloading Transformer (~8GB)..."}))
                snapshot_download(
                    "Qwen/Qwen-Image-Edit-2509",
                    allow_patterns=["transformer/**"],
                )
                q.put(json.dumps({"progress": 1.0, "message": "Transformer downloaded", "done": True}))
            elif model_id == "qwen_vae":
                q.put(json.dumps({"progress": 0.1, "message": "Downloading VAE..."}))
                snapshot_download(
                    "Qwen/Qwen-Image-Edit-2509",
                    allow_patterns=["vae/**"],
                )
                q.put(json.dumps({"progress": 1.0, "message": "VAE downloaded", "done": True}))
            elif model_id == "realesrgan":
                q.put(json.dumps({"progress": 0.5, "message": "Real-ESRGAN downloads on first use"}))
                q.put(json.dumps({"progress": 1.0, "message": "Real-ESRGAN ready", "done": True}))
            elif model_id == "nafnet":
                q.put(json.dumps({"progress": 0.1, "message": "Downloading NAFNet weights..."}))
                snapshot_download(
                    "piddnad/nafnet-denoise-deblur-weights",
                    allow_patterns=["*.pth"],
                )
                q.put(json.dumps({"progress": 1.0, "message": "NAFNet downloaded", "done": True}))
            else:
                q.put(json.dumps({"progress": 1.0, "message": f"Unknown model: {model_id}", "done": True}))
        except Exception as e:
            q.put(json.dumps({"progress": 1.0, "message": f"Error: {e}", "done": True}))
        q.put(None)

    def event_stream():
        thread = threading.Thread(target=do_download, daemon=True)
        thread.start()
        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {item}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def configure(engine=None, mock_mode=False, upscale_engine=None, restore_engine=None):
    """Set the engine and mock mode for API endpoints."""
    _sessions["__engine__"] = engine
    _sessions["__mock_mode__"] = mock_mode
    _sessions["__upscale_engine__"] = upscale_engine
    _sessions["__restore_engine__"] = restore_engine
