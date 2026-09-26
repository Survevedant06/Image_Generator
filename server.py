#!/usr/bin/env python3
"""
server.py - Production-Ready FastAPI Backend for E-Commerce Product Image Processing

Features:
  - Drag-and-drop multi-file upload for sellers (supporting .jpg, .png, .webp, .heic, etc.)
  - Real-time queue state auditing (Unprocessed vs. Processed)
  - Non-blocking background AI background removal (BiRefNet)
  - Interactive background preview endpoints and asset downloads
  - Static file hosting for dashboard UI
"""

import os
import sys
import gc
import re
import uuid
import time
import json
import queue
import asyncio
import shutil
import threading
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form, Query, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from pydantic import BaseModel

# Suppress Windows symlinks warning
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import db
import image_search
import beautify_service
import category_classifier

from batch_remove_bg import (
    load_birefnet_model,
    remove_background,
    load_and_standardize_image,
    is_valid_input_image,
    SUPPORTED_EXTENSIONS
)
from generate_scene import (
    STUDIO_PRESETS,
    ASPECT_RATIOS,
    generate_product_scene,
    generate_multi_channel_pack,
    GENERATED_SCENES_DIR
)
from angles_service import (
    ANGLE_METADATA,
    get_available_angle_sets,
    generate_product_angles,
    create_angles_zip,
    create_angles_gif,
    OUTPUT_ANGLES_DIR
)

INPUTS_DIR = os.path.join(BASE_DIR, "inputs")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_scenes")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(INPUTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_ANGLES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

class GenerateSceneRequest(BaseModel):
    cutout_filename: str
    preset_id: str = "white"
    aspect_ratio: str = "1:1"  # "1:1", "4:5", "9:16", "16:9", or "all"
    custom_prompt: Optional[str] = None
    shadow_intensity: Optional[float] = None

class GenerateAnglesRequest(BaseModel):
    cutout_filename: Optional[str] = None
    base_name: Optional[str] = None

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = ""
    icon: Optional[str] = "tag"

class CategoryUpdate(BaseModel):
    name: str
    description: Optional[str] = ""
    icon: Optional[str] = "tag"

class ProductCreate(BaseModel):
    name: str
    sku: Optional[str] = None
    category_id: Optional[int] = None
    description: Optional[str] = ""
    tags: Optional[str] = ""
    raw_image: Optional[str] = None
    cutout_image: Optional[str] = None

class ProductUpdate(BaseModel):
    name: Optional[str] = None
    sku: Optional[str] = None
    category_id: Optional[int] = None
    description: Optional[str] = None
    tags: Optional[str] = None
    status: Optional[str] = None

class ExportImageRequest(BaseModel):
    source_image: str
    output_format: str = "PNG"
    aspect_ratio: str = "1:1"
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    dpi: int = 300
    background_color: str = "transparent"
    auto_enhance: bool = True
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    sharpness: float = 1.0

class DetectCategoryRequest(BaseModel):
    name: str
    filename: Optional[str] = None
    tags: Optional[str] = ""
    description: Optional[str] = ""

class CleanupDuplicatesRequest(BaseModel):
    product_ids: Optional[List[int]] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    db.sync_existing_files_into_db()
    yield

app = FastAPI(title="E-Commerce AI Studio", version="1.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Tracker for Background Processing
processing_lock = threading.Lock()
is_processing = False
processing_status = {
    "is_running": False,
    "current_item": "",
    "total_queued": 0,
    "completed": 0,
    "errors": 0,
    "last_run_time": ""
}

# Global State Tracker for 3D Multi-Angle Generation
angles_processing_lock = threading.Lock()
angles_status = {
    "is_running": False,
    "current_product": "",
    "step": "idle",  # "queued", "loading_model", "synthesizing", "done", "failed"
    "message": "",
    "error": None,
    "started_at": 0,
    "completed_at": 0,
    "render_time_seconds": 0
}


def angles_background_worker(cutout_path: str, base_name: str):
    global angles_status
    with angles_processing_lock:
        angles_status["is_running"] = True
        angles_status["current_product"] = base_name
        angles_status["step"] = "synthesizing"
        angles_status["message"] = f"Synthesizing 6 canonical perspectives for '{base_name}'..."
        angles_status["error"] = None
        angles_status["started_at"] = time.time()
        print(f"[Angles Worker] Starting 3D angle generation for '{base_name}'...")

        try:
            res = generate_product_angles(cutout_path, base_name=base_name)

            # Link with product in SQLite DB if matching
            try:
                products = db.list_products()
                for p in products:
                    p_raw = p.get("raw_image") or ""
                    p_base = os.path.splitext(os.path.basename(p_raw))[0]
                    p_norm = p["name"].lower().replace(" ", "_")
                    if base_name in (p_base, p_norm) or p_norm in base_name or p_base in base_name:
                        for v in res.get("views", []):
                            db.record_product_asset(p["id"], "angle", v["path"], metadata={"azimuth": v.get("azimuth"), "elevation": v.get("elevation")})
                        break
            except Exception as db_err:
                print(f"[Angles Worker DB Warning] {db_err}")

            angles_status["step"] = "done"
            angles_status["message"] = f"Successfully generated 6 perspectives for '{base_name}'."
            angles_status["render_time_seconds"] = res.get("render_time_seconds", 0)
            angles_status["result"] = res
            print(f"[Angles Worker] Completed '{base_name}' in {angles_status['render_time_seconds']}s.")
        except Exception as e:
            print(f"[Angles Worker Error] {e}")
            angles_status["step"] = "failed"
            angles_status["error"] = str(e)
            angles_status["message"] = f"Generation failed: {str(e)}"
        finally:
            angles_status["is_running"] = False
            angles_status["completed_at"] = time.time()

# Cached AI Model instance
cached_model = None
cached_device = None
_model_lock = threading.Lock()


def get_model():
    global cached_model, cached_device
    import torch
    if cached_model is None:
        with _model_lock:
            if cached_model is None:
                cached_device = "cuda" if torch.cuda.is_available() else "cpu"
                cached_model = load_birefnet_model(cached_device)
    return cached_model, cached_device


def _warmup_birefnet():
    """Warms up BiRefNet segmentation model in background at startup so first request has 0ms load delay."""
    try:
        get_model()
        print("[Startup] BiRefNet model pre-cached in memory & ready.")
    except Exception as e:
        print(f"[Startup Notice] BiRefNet model warmup: {e}")

threading.Thread(target=_warmup_birefnet, daemon=True).start()


# ---------------------------------------------------------------------------
# Real-Time Event Streaming (Server-Sent Events) & Caching
# ---------------------------------------------------------------------------
_event_subscribers: List[queue.Queue] = []
_event_lock = threading.Lock()
_skipped_duplicate_cache: Dict[str, Tuple[float, int, int]] = {}  # {filename: (mtime, size, product_id)}
_sync_lock = threading.Lock()
_last_sync_time = 0.0
_active_upload_paths: set = set()
_active_upload_lock = threading.Lock()


def broadcast_event(event_type: str, data: dict):
    """Pushes a real-time event to all active dashboard EventSource connections."""
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _event_lock:
        dead = []
        for q in _event_subscribers:
            try:
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
        for d in dead:
            try:
                _event_subscribers.remove(d)
            except ValueError:
                pass


def sync_project_folders() -> Dict[str, Any]:
    """
    Fast, optimized folder synchronizer.
    Scans inputs/ and the project root directory for image files.
    Automatically detects any newly downloaded or added images:
      - Copies them into inputs/ if placed in the project root.
      - Uses caching on skipped duplicates and valid images for sub-millisecond execution.
      - Auto-generates clean product titles & unique SKUs.
      - Auto-resolves/creates categories via category_classifier.
      - Registers new products into SQLite DB.
      - Links any existing cutouts in processed_images/ to products.
      - Broadcasts real-time events to connected clients.
    """
    global _last_sync_time
    with _sync_lock:
        _last_sync_time = time.time()
        new_products = []
        updated_cutouts = []

        # 1. Fast check for standalone images in project root and copy to inputs/
        EXCLUDED_ROOT_NAMES = {
            "test_seller_item.jpg", "image.jpg", "image_1.jpg", "backpack_white.png", 
            "image_cutout.png", "image_1_cutout.png", "test.jpg", "test_seller_item_cutout.png"
        }
        try:
            for item in os.listdir(BASE_DIR):
                lower = item.lower()
                if (lower.startswith("test") or lower.startswith("sample") or 
                    lower.startswith("mock") or lower.startswith("temp") or lower.startswith("tmp")):
                    continue
                if item in EXCLUDED_ROOT_NAMES or item.startswith("."):
                    continue
                if not lower.endswith(SUPPORTED_EXTENSIONS):
                    continue
                full_p = os.path.join(BASE_DIR, item)
                if not os.path.isfile(full_p):
                    continue
                if is_valid_input_image(full_p):
                    dest_p = os.path.join(INPUTS_DIR, item)
                    if not os.path.exists(dest_p):
                        try:
                            shutil.copy2(full_p, dest_p)
                        except Exception:
                            pass
        except Exception as e:
            print(f"[Sync Root Error] {e}")

        # 2. Inspect valid input files in inputs/ (excluding test/mock patterns)
        try:
            input_files = [
                f for f in os.listdir(INPUTS_DIR)
                if f.lower().endswith(SUPPORTED_EXTENSIONS) and 
                not f.lower().startswith(("test", "sample", "mock", "temp", "tmp")) and
                is_valid_input_image(os.path.join(INPUTS_DIR, f))
            ]
        except Exception:
            input_files = []

        all_prods = db.list_products(limit=2000).get("products", [])
        raw_map = {
            os.path.basename(p["raw_image"]).lower(): p
            for p in all_prods if p.get("raw_image")
        }
        existing_raw_norm = {
            os.path.normpath(os.path.abspath(p["raw_image"])).lower()
            for p in all_prods if p.get("raw_image")
        }

        # 3. Link existing cutouts on disk for products in DB missing cutout_image
        for p in all_prods:
            if not p.get("cutout_image") and p.get("raw_image"):
                base = os.path.splitext(os.path.basename(p["raw_image"]))[0]
                cand = os.path.join(PROCESSED_DIR, f"{base}_cutout.png")
                if os.path.exists(cand):
                    db.record_product_asset(p["id"], "cutout", cand)
                    db.update_product(p["id"], status="cutout_ready")
                    updated_cutouts.append(p["id"])

        # 4. Ingest newly added files in inputs/ that are not yet in DB
        with _active_upload_lock:
            active_uploads = set(_active_upload_paths)

        for f in input_files:
            dest_path = os.path.join(INPUTS_DIR, f)
            norm_dest = os.path.normpath(os.path.abspath(dest_path)).lower()

            # Guard: Skip if file is actively being uploaded by API
            if norm_dest in active_uploads:
                continue

            # Guard: Skip if file is already registered in DB
            if f.lower() in raw_map or norm_dest in existing_raw_norm:
                continue

            try:
                st = os.stat(dest_path)
                mtime = st.st_mtime
                size = st.st_size
                if size == 0:
                    continue
            except OSError:
                continue

            # Check cache for known duplicate / skipped files
            if f in _skipped_duplicate_cache:
                c_mtime, c_size, c_pid = _skipped_duplicate_cache[f]
                if c_mtime == mtime and c_size == size:
                    continue

            base = os.path.splitext(f)[0]
            clean_title = re.sub(r"[_\-\+]+", " ", base).strip().title()
            if not clean_title or len(clean_title) < 2:
                clean_title = f"Product {base.title()}"

            clean_title = category_classifier.refine_product_title(
                clean_title,
                raw_image_path=dest_path,
                filename=f
            )

            # Perceptual cross-format duplicate check
            dup_match = db.find_duplicate_or_similar_image(dest_path, max_hamming_distance=6)
            if dup_match:
                _skipped_duplicate_cache[f] = (mtime, size, dup_match["product_id"])
                print(f"[Folder Watcher] Skipping '{f}' - visual duplicate of Product #{dup_match['product_id']} '{dup_match['product_name']}'.")
                continue

            cat_id, cat_name, is_new = category_classifier.resolve_product_category(
                product_name=clean_title,
                raw_image_path=dest_path,
                filename=f
            )

            sku = f"SKU-{uuid.uuid4().hex[:8].upper()}"
            new_prod = db.create_product(
                name=clean_title,
                sku=sku,
                category_id=cat_id,
                raw_image=dest_path,
                status="raw"
            )

            # Check if cutout already exists in processed_images
            cand = os.path.join(PROCESSED_DIR, f"{base}_cutout.png")
            if os.path.exists(cand):
                db.record_product_asset(new_prod["id"], "cutout", cand)
                db.update_product(new_prod["id"], status="cutout_ready")

            prod_payload = {
                "id": new_prod["id"],
                "name": clean_title,
                "category_name": cat_name,
                "category_id": cat_id,
                "filename": f,
                "sku": sku,
                "raw_url": f"/inputs/{f}",
                "cutout_url": f"/processed_images/{base}_cutout.png" if os.path.exists(cand) else None,
                "status": "cutout_ready" if os.path.exists(cand) else "raw"
            }
            new_products.append(prod_payload)
            print(f"[Folder Watcher] Real-time detected & registered '{f}' -> Product #{new_prod['id']} '{clean_title}' in category '{cat_name}'.")
            broadcast_event("product_added", {
                "product": prod_payload,
                "total_products": len(all_prods) + len(new_products)
            })

        if new_products or updated_cutouts:
            broadcast_event("sync_update", {
                "new_count": len(new_products),
                "updated_cutouts": len(updated_cutouts),
                "total_products": len(all_prods) + len(new_products)
            })

        return {
            "new_products": new_products,
            "updated_cutouts": updated_cutouts,
            "total_new": len(new_products)
        }


# ---------------------------------------------------------------------------
# High-Performance Watchdog Real-Time Filesystem Observer
# ---------------------------------------------------------------------------
class ImageFolderEventHandler(FileSystemEventHandler):
    """Native OS filesystem event listener detecting additions and downloads in milliseconds."""
    def __init__(self):
        super().__init__()
        self._debounce_timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _trigger_sync(self):
        with self._lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(0.25, self._run_sync)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def _run_sync(self):
        try:
            sync_project_folders()
        except Exception as e:
            print(f"[Watchdog Event Sync Error] {e}")

    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            self._trigger_sync()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith(SUPPORTED_EXTENSIONS):
            self._trigger_sync()

    def on_moved(self, event):
        dest = getattr(event, "dest_path", "")
        if dest and dest.lower().endswith(SUPPORTED_EXTENSIONS):
            self._trigger_sync()


# Initialize Watchdog Observer immediately
watchdog_observer = Observer()
_watch_handler = ImageFolderEventHandler()
watchdog_observer.schedule(_watch_handler, path=INPUTS_DIR, recursive=False)
watchdog_observer.schedule(_watch_handler, path=BASE_DIR, recursive=False)
try:
    watchdog_observer.start()
    print("[Watchdog] Real-time directory monitoring active on inputs/ and project root.")
except Exception as e:
    print(f"[Watchdog Warning] Could not start native observer: {e}")


def folder_watcher_daemon():
    """Fallback background watcher ensuring sync even if filesystem events are suppressed."""
    while True:
        try:
            time.sleep(2)
            sync_project_folders()
        except Exception:
            time.sleep(3)


# Launch fallback folder watcher thread
watcher_thread = threading.Thread(target=folder_watcher_daemon, daemon=True)
watcher_thread.start()


def scan_queue_state() -> Dict[str, Any]:
    """Scans inputs/ and processed_images/ to categorize images."""
    inputs_files = sorted(os.listdir(INPUTS_DIR))
    processed_files = sorted(os.listdir(PROCESSED_DIR))

    processed_bases = {
        f[:-11]: f for f in processed_files if f.endswith("_cutout.png")
    }

    unprocessed = []
    processed = []

    for f in inputs_files:
        full_p = os.path.join(INPUTS_DIR, f)
        if not is_valid_input_image(full_p):
            continue

        base_name = os.path.splitext(f)[0]
        size_bytes = os.path.getsize(full_p)
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(full_p)))

        if base_name in processed_bases:
            cutout_name = processed_bases[base_name]
            cutout_p = os.path.join(PROCESSED_DIR, cutout_name)
            processed.append({
                "id": f"proc_{base_name}",
                "original_filename": f,
                "cutout_filename": cutout_name,
                "base_name": base_name,
                "original_url": f"/inputs/{f}",
                "cutout_url": f"/processed_images/{cutout_name}",
                "size_kb": round(size_bytes / 1024, 1),
                "cutout_size_kb": round(os.path.getsize(cutout_p) / 1024, 1) if os.path.exists(cutout_p) else 0,
                "modified": mod_time
            })
        else:
            unprocessed.append({
                "id": f"unproc_{base_name}",
                "filename": f,
                "base_name": base_name,
                "url": f"/inputs/{f}",
                "size_kb": round(size_bytes / 1024, 1),
                "modified": mod_time
            })

    return {
        "unprocessed": unprocessed,
        "processed": processed,
        "unprocessed_count": len(unprocessed),
        "processed_count": len(processed),
        "total_count": len(unprocessed) + len(processed)
    }


def background_worker(target_files: Optional[List[str]] = None):
    """Background task that runs BiRefNet sequentially over pending files."""
    global is_processing, processing_status
    with processing_lock:
        is_processing = True
        processing_status["is_running"] = True
        processing_status["completed"] = 0
        processing_status["errors"] = 0

        # Sync folders first so all files are registered in DB
        sync_project_folders()

        queue_info = scan_queue_state()
        if target_files:
            to_process = [
                item for item in queue_info["unprocessed"]
                if item["filename"] in target_files
            ]
        else:
            to_process = queue_info["unprocessed"]

        processing_status["total_queued"] = len(to_process)

        if not to_process:
            is_processing = False
            processing_status["is_running"] = False
            return

        try:
            model, device = get_model()

            for i, item in enumerate(to_process, 1):
                filename = item["filename"]
                processing_status["current_item"] = filename
                in_path = os.path.join(INPUTS_DIR, filename)
                out_path = os.path.join(PROCESSED_DIR, f"{item['base_name']}_cutout.png")

                try:
                    std_rgb, orig_sz = load_and_standardize_image(in_path)
                    cutout = remove_background(std_rgb, orig_sz, model, device)
                    cutout.save(out_path, format="PNG")
                    processing_status["completed"] += 1

                    # CRITICAL: Link cutout in DB for this product!
                    all_prods = db.list_products(limit=1000).get("products", [])
                    matching_prod = next(
                        (p for p in all_prods if os.path.basename(p.get("raw_image") or "") == filename),
                        None
                    )
                    if matching_prod:
                        db.record_product_asset(matching_prod["id"], "cutout", out_path)
                        db.update_product(matching_prod["id"], status="cutout_ready")

                except Exception as err:
                    print(f"[Worker Error] Failed on {filename}: {err}")
                    processing_status["errors"] += 1

        except Exception as e:
            print(f"[Worker Fatal] {e}")
        finally:
            is_processing = False
            processing_status["is_running"] = False
            processing_status["current_item"] = ""
            processing_status["last_run_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
            # Sync once more to ensure everything is matched up
            sync_project_folders()


# ---------------------------------------------------------------------------
# API Routes & Real-Time SSE Stream
# ---------------------------------------------------------------------------
@app.get("/api/events")
async def sse_events(request: Request):
    """
    Real-time Server-Sent Events stream.
    Instantly pushes folder additions, background processing progress, and catalog sync events
    to connected browser dashboards with zero polling delay.
    """
    q: queue.Queue = queue.Queue(maxsize=100)
    with _event_lock:
        _event_subscribers.append(q)

    async def event_generator():
        try:
            # Initial connection confirmation
            yield 'event: connected\ndata: {"status": "connected"}\n\n'
            while True:
                if await request.is_disconnected():
                    break
                try:
                    loop = asyncio.get_event_loop()
                    # Wait up to 1.5s for queued event or send ping keepalive
                    msg = await loop.run_in_executor(None, lambda: q.get(timeout=1.5))
                    yield msg
                except Exception:
                    yield ": ping\n\n"
        finally:
            with _event_lock:
                if q in _event_subscribers:
                    _event_subscribers.remove(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/status")
def get_status():
    """
    Returns current counts and image lists for Unprocessed and Processed.
    Optimized for high-frequency polling with non-blocking cached state.
    """
    global _last_sync_time
    now = time.time()
    # If more than 1.5s passed since last sync, trigger lightweight check
    if now - _last_sync_time > 1.5:
        sync_project_folders()

    queue_data = scan_queue_state()
    prods = db.list_products(limit=1000).get("products", [])
    pending_prods = [p for p in prods if not p.get("cutout_image") or p.get("status") == "raw"]
    return {
        "status": "ok",
        "queue": queue_data,
        "processing": processing_status,
        "pending_count": len(queue_data["unprocessed"]),
        "total_products": len(prods),
        "pending_products_count": len(pending_prods)
    }


@app.post("/api/sync")
def sync_folder_endpoint():
    """Manually triggers folder detection and synchronization with database."""
    res = sync_project_folders()
    broadcast_event("sync_update", {
        "new_count": res.get("total_new", 0),
        "updated_cutouts": len(res.get("updated_cutouts", [])),
        "total_products": len(db.list_products(limit=2000).get("products", []))
    })
    return {"status": "ok", "result": res}


@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Uploads one or multiple product photos into inputs/."""
    uploaded = []
    rejected = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            rejected.append({"filename": file.filename, "reason": "Unsupported format"})
            continue

        safe_filename = os.path.basename(file.filename)
        dest_path = os.path.join(INPUTS_DIR, safe_filename)

        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        uploaded.append(safe_filename)

    return {
        "uploaded": uploaded,
        "rejected": rejected,
        "message": f"Successfully uploaded {len(uploaded)} image(s)."
    }


@app.post("/api/upload/batch")
async def upload_batch_products(files: List[UploadFile] = File(...)):
    """
    Ingests and deploys multiple product photos simultaneously into the system.
    For every uploaded image:
      1. Validates format against SUPPORTED_EXTENSIONS.
      2. Saves into inputs/ with conflict-free naming.
      3. Performs perceptual visual hash matching (check_duplicate).
      4. Auto-resolves/creates appropriate category via category_classifier.
      5. Auto-generates clean product name & unique SKU.
      6. Registers product in SQLite DB and links raw/cutout assets.
      7. Returns batch deployment summary and any detected duplicates.
    """
    imported = []
    rejected = []
    duplicates = []

    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            rejected.append({"filename": file.filename, "reason": "Unsupported format"})
            continue

        raw_name = os.path.basename(file.filename)
        base, file_ext = os.path.splitext(raw_name)

        # Conflict-free safe filename in inputs/
        safe_filename = raw_name
        dest_path = os.path.join(INPUTS_DIR, safe_filename)
        counter = 1
        while os.path.exists(dest_path):
            safe_filename = f"{base}_{int(time.time())}_{counter}{file_ext}"
            dest_path = os.path.join(INPUTS_DIR, safe_filename)
            counter += 1

        norm_dest = os.path.normpath(os.path.abspath(dest_path)).lower()
        with _active_upload_lock:
            _active_upload_paths.add(norm_dest)

        try:
            with open(dest_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Generate human-friendly title & refine with AI vision if generic
            clean_title = re.sub(r"[_\-\+]+", " ", base).strip().title()
            if not clean_title or len(clean_title) < 2:
                clean_title = f"Product {base.title()}"

            clean_title = category_classifier.refine_product_title(
                clean_title,
                raw_image_path=dest_path,
                filename=safe_filename
            )

            # Perceptual duplicate check
            dup_info = image_search.check_duplicate(dest_path)
            is_dup = dup_info.get("exists", False)
            dup_match = dup_info.get("match")
            if is_dup and dup_match:
                duplicates.append({
                    "filename": safe_filename,
                    "product_name": clean_title,
                    "matched_product": dup_match
                })

            # Auto-resolve or create category
            cat_id, cat_name, is_new = category_classifier.resolve_product_category(
                product_name=clean_title,
                raw_image_path=dest_path,
                filename=safe_filename
            )

            # Unique SKU
            sku = f"SKU-{uuid.uuid4().hex[:8].upper()}"

            # Create product in DB
            prod = db.create_product(
                name=clean_title,
                sku=sku,
                category_id=cat_id,
                raw_image=dest_path,
                status="raw"
            )

            # If cutout already exists in processed_images, link it immediately
            cutout_candidate = os.path.join(PROCESSED_DIR, f"{base}_cutout.png")
            if os.path.exists(cutout_candidate):
                db.record_product_asset(prod["id"], "cutout", cutout_candidate)
                db.update_product(prod["id"], status="cutout_ready")
                prod["status"] = "cutout_ready"
                prod["cutout_image"] = cutout_candidate

            imported.append({
                "id": prod["id"],
                "product_id": prod["id"],
                "name": clean_title,
                "sku": sku,
                "category_id": cat_id,
                "category_name": cat_name,
                "category_is_new": is_new,
                "filename": safe_filename,
                "raw_url": f"/inputs/{safe_filename}",
                "status": prod["status"],
                "is_duplicate": is_dup,
                "duplicate_match": dup_match
            })
        finally:
            with _active_upload_lock:
                _active_upload_paths.discard(norm_dest)

    return {
        "status": "success",
        "total_received": len(files),
        "imported_count": len(imported),
        "imported": imported,
        "duplicates": duplicates,
        "rejected": rejected,
        "message": f"Successfully ingested and deployed {len(imported)} product(s) into catalog."
    }


@app.post("/api/process")
def trigger_process(background_tasks: BackgroundTasks):
    """Triggers background processing for all pending unprocessed images."""
    global is_processing
    if is_processing:
        raise HTTPException(status_code=409, detail="A background removal batch is already running. Please wait for it to complete.")

    # 1. Sync project folders first so any newly downloaded files are registered
    sync_project_folders()

    # 2. Check pending queue
    queue_info = scan_queue_state()
    unprocessed = queue_info["unprocessed"]

    if not unprocessed:
        return {
            "status": "idle",
            "message": "All images are already processed! No pending images found.",
            "total_queued": 0
        }

    background_tasks.add_task(background_worker)
    return {
        "status": "started",
        "message": f"Started background removal for {len(unprocessed)} pending image(s).",
        "total_queued": len(unprocessed)
    }


@app.post("/api/process-single/{filename}")
def trigger_process_single(filename: str, background_tasks: BackgroundTasks):
    """Triggers background removal for a single specific image."""
    global is_processing
    if is_processing:
        raise HTTPException(status_code=409, detail="A processing batch is already running.")

    in_path = os.path.join(INPUTS_DIR, filename)
    if not os.path.exists(in_path):
        raise HTTPException(status_code=404, detail="File not found in inputs directory.")

    background_tasks.add_task(background_worker, target_files=[filename])
    return {"message": f"Processing queued for '{filename}'."}


@app.delete("/api/delete/{category}/{filename}")
def delete_file(category: str, filename: str):
    """Deletes an image from inputs or processed_images."""
    safe_name = os.path.basename(filename)
    if category == "unprocessed":
        path = os.path.join(INPUTS_DIR, safe_name)
    elif category == "processed":
        path = os.path.join(PROCESSED_DIR, safe_name)
    else:
        raise HTTPException(status_code=400, detail="Invalid category.")

    if os.path.exists(path):
        os.remove(path)
        return {"message": f"Deleted {safe_name}."}
    else:
        raise HTTPException(status_code=404, detail="File not found.")


@app.get("/api/download/{filename}")
def download_cutout(filename: str):
    """Downloads the cutout file."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(PROCESSED_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed cutout not found.")
    return FileResponse(file_path, media_type="image/png", filename=safe_name)


# ---------------------------------------------------------------------------
# Studio Scene Presets & Generation Routes (V1-04 & V1-05)
# ---------------------------------------------------------------------------
@app.get("/api/presets")
def get_presets():
    """Returns the list of available studio scene presets and metadata."""
    preset_list = []
    for pid, pdata in STUDIO_PRESETS.items():
        preset_list.append({
            "id": pid,
            "name": pdata["name"],
            "badge": pdata["badge"],
            "description": pdata["description"],
            "icon": pdata["icon"],
            "prompt_template": pdata["prompt_template"],
            "has_reflection": pdata.get("has_reflection", False)
        })
    return {"presets": preset_list}


@app.get("/api/aspect-ratios")
def get_aspect_ratios():
    """Returns all supported commercial multi-channel aspect ratios (V1-07)."""
    return {"aspect_ratios": list(ASPECT_RATIOS.values())}


@app.post("/api/generate-scene")
def generate_scene_endpoint(req: GenerateSceneRequest):
    """
    Renders a product cutout onto a commercial studio scene preset or custom prompt.
    Supports individual aspect ratios (1:1, 4:5, 9:16, 16:9) or 'all' for the full marketing pack.
    """
    safe_cutout = os.path.basename(req.cutout_filename)
    cutout_path = os.path.join(PROCESSED_DIR, safe_cutout)

    # Fallback to root or inputs if not in processed_images
    if not os.path.exists(cutout_path):
        root_path = os.path.join(BASE_DIR, safe_cutout)
        if os.path.exists(root_path):
            cutout_path = root_path
        else:
            raise HTTPException(status_code=404, detail=f"Cutout '{safe_cutout}' not found.")

    try:
        if req.aspect_ratio == "all":
            # Generate all 4 marketing aspect ratios in one shot
            pack = generate_multi_channel_pack(
                cutout_path=cutout_path,
                preset_id=req.preset_id,
                custom_prompt=req.custom_prompt,
                shadow_intensity=req.shadow_intensity
            )
            return {
                "status": "success",
                "mode": "multi_channel",
                "preset_id": req.preset_id,
                "items": pack,
                "total": len(pack)
            }
        else:
            # Generate single specified aspect ratio
            res = generate_product_scene(
                cutout_path=cutout_path,
                preset_id=req.preset_id,
                aspect_ratio=req.aspect_ratio,
                custom_prompt=req.custom_prompt,
                shadow_intensity=req.shadow_intensity
            )
            return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scene generation failed: {str(e)}")


@app.get("/api/scenes")
def list_generated_scenes():
    """Returns a list of all rendered commercial scenes in generated_scenes/."""
    files = sorted(os.listdir(GENERATED_DIR), reverse=True)
    scenes = []
    for f in files:
        if not f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        full_p = os.path.join(GENERATED_DIR, f)
        size_kb = round(os.path.getsize(full_p) / 1024, 1)
        mod_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(full_p)))
        
        # Detect preset name
        preset_found = "Custom Scene"
        for pid, pdata in STUDIO_PRESETS.items():
            if f"_{pid}" in f:
                preset_found = pdata["name"]
                break

        # Detect aspect ratio
        ar_found = "1:1 Square"
        ar_channel = "Amazon / Shopify"
        for arid, ardata in ASPECT_RATIOS.items():
            if f"_{ardata['tag']}.png" in f:
                ar_found = ardata["name"]
                ar_channel = ardata["channel"]
                break

        scenes.append({
            "id": f"scene_{f}",
            "filename": f,
            "url": f"/generated_scenes/{f}",
            "preset": preset_found,
            "aspect_ratio": ar_found,
            "channel": ar_channel,
            "size_kb": size_kb,
            "modified": mod_time
        })
    return {"scenes": scenes, "total": len(scenes)}


@app.get("/api/download-scene/{filename}")
def download_scene(filename: str):
    """Downloads a generated commercial render."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(GENERATED_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Rendered scene not found.")
    return FileResponse(file_path, media_type="image/png", filename=safe_name)


@app.delete("/api/delete-scene/{filename}")
def delete_scene(filename: str):
    """Deletes a generated commercial scene."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(GENERATED_DIR, safe_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        return {"message": f"Deleted scene {safe_name}."}
    raise HTTPException(status_code=404, detail="Render not found.")


# ---------------------------------------------------------------------------
# 3D Multi-Angle Generation & Interactive Turntable Routes
# ---------------------------------------------------------------------------
@app.get("/api/angles")
def list_available_angles():
    """Returns all products with generated 6-angle camera perspectives."""
    sets = get_available_angle_sets()
    return {
        "angle_sets": sets,
        "total_products": len(sets),
        "angles_metadata": ANGLE_METADATA
    }


@app.get("/api/angles-status")
def get_angles_status():
    """Returns current progress status of background 3D angle generation."""
    return {
        "status": "ok",
        "angles_processing": angles_status
    }


@app.post("/api/generate-angles")
def generate_angles_endpoint(req: GenerateAnglesRequest):
    """
    Synthesizes 6 canonical perspectives for product cutout.
    Updates SQLite DB product asset tracking and returns generated views.
    """
    raw_cutout = req.cutout_filename or ""
    raw_base = req.base_name or ""

    if not raw_cutout and not raw_base:
        raise HTTPException(
            status_code=400,
            detail="Missing required parameter: please provide either 'cutout_filename' or 'base_name'."
        )

    # Determine base name and candidate cutout filename
    if raw_cutout:
        safe_cutout = os.path.basename(raw_cutout)
        base_name = raw_base if raw_base else os.path.splitext(safe_cutout)[0].replace("_cutout", "").replace("_std", "")
    else:
        base_name = os.path.basename(raw_base).replace("_cutout", "").replace("_std", "")
        safe_cutout = f"{base_name}_cutout.png"

    # Search for cutout file in common locations
    candidate_paths = [
        os.path.join(PROCESSED_DIR, safe_cutout),
        os.path.join(PROCESSED_DIR, f"{base_name}_cutout.png"),
        os.path.join(INPUTS_DIR, f"{base_name}.png"),
        os.path.join(INPUTS_DIR, f"{base_name}.jpg"),
        os.path.join(INPUTS_DIR, f"{base_name}.jpeg"),
        os.path.join(INPUTS_DIR, f"{base_name}.webp"),
    ]

    cutout_path = next((p for p in candidate_paths if os.path.exists(p)), None)

    if not cutout_path:
        raise HTTPException(
            status_code=404,
            detail=f"Cutout image for '{base_name}' not found. Please remove the background first."
        )

    try:
        res = generate_product_angles(cutout_path, base_name=base_name)
        # Update SQLite DB asset tracking
        try:
            products = db.list_products()
            for p in products:
                p_raw = p.get("raw_image") or ""
                p_base = os.path.splitext(os.path.basename(p_raw))[0]
                p_norm = p["name"].lower().replace(" ", "_")
                if base_name in (p_base, p_norm) or p_norm in base_name or p_base in base_name:
                    for v in res.get("views", []):
                        db.record_product_asset(p["id"], "angle", v["path"], metadata={"azimuth": v.get("azimuth"), "elevation": v.get("elevation")})
                    break
        except Exception as db_err:
            print(f"[Generate Angles DB Warning] {db_err}")

        res["message"] = f"Successfully generated 6 perspectives for '{base_name}'."
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D angle synthesis failed: {str(e)}")


@app.get("/api/download-angles-zip/{base_name}")
def download_angles_zip(base_name: str):
    """Downloads all 6 camera angles bundled in an in-memory ZIP archive."""
    safe_base = os.path.basename(base_name).replace(".zip", "")
    candidate_bases = [
        safe_base,
        safe_base.lower(),
        safe_base.lower().replace(" ", "_"),
        safe_base.replace("_cutout", "")
    ]
    target_base = next((b for b in candidate_bases if os.path.exists(os.path.join(OUTPUT_ANGLES_DIR, b))), safe_base)

    try:
        zip_buf = create_angles_zip(target_base)
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={target_base}_6_angles.zip"}
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/download-angles-gif/{base_name}")
def download_angles_gif(base_name: str, duration: int = 65):
    """Downloads an animated 360-degree rotating turntable GIF from the 6 camera perspectives."""
    safe_base = os.path.basename(base_name).replace(".gif", "")
    candidate_bases = [
        safe_base,
        safe_base.lower(),
        safe_base.lower().replace(" ", "_"),
        safe_base.replace("_cutout", "")
    ]
    target_base = next((b for b in candidate_bases if os.path.exists(os.path.join(OUTPUT_ANGLES_DIR, b))), safe_base)

    try:
        gif_buf = create_angles_gif(target_base, duration_ms=duration)
        return StreamingResponse(
            gif_buf,
            media_type="image/gif",
            headers={"Content-Disposition": f"attachment; filename={target_base}_360_turntable.gif"}
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/angles-gif/{base_name}")
def view_angles_gif(base_name: str, duration: int = 65):
    """Streams an animated 360-degree rotating turntable GIF inline for interactive browser preview."""
    safe_base = os.path.basename(base_name).replace(".gif", "")
    candidate_bases = [
        safe_base,
        safe_base.lower(),
        safe_base.lower().replace(" ", "_"),
        safe_base.replace("_cutout", "")
    ]
    target_base = next((b for b in candidate_bases if os.path.exists(os.path.join(OUTPUT_ANGLES_DIR, b))), safe_base)

    try:
        gif_buf = create_angles_gif(target_base, duration_ms=duration)
        return StreamingResponse(
            gif_buf,
            media_type="image/gif"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))



# ---------------------------------------------------------------------------
# Category Management CRUD Routes
# ---------------------------------------------------------------------------
@app.get("/api/categories")
def get_categories():
    """Returns all categories with their active product counts."""
    return {"categories": db.list_categories()}


@app.post("/api/categories")
def create_category_endpoint(req: CategoryCreate):
    """Creates a new product category."""
    try:
        cat = db.create_category(req.name, req.description, req.icon)
        return {"status": "success", "category": cat}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/categories/{category_id}")
def update_category_endpoint(category_id: int, req: CategoryUpdate):
    """Updates an existing category."""
    cat = db.update_category(category_id, req.name, req.description, req.icon)
    if not cat:
        raise HTTPException(status_code=404, detail="Category not found.")
    return {"status": "success", "category": cat}


@app.delete("/api/categories/{category_id}")
def delete_category_endpoint(category_id: int, reassign_to_id: Optional[int] = None):
    """Deletes a category and reassigns products."""
    count = db.delete_category(category_id, reassign_to_id=reassign_to_id)
    return {"status": "success", "reassigned_products": count, "message": f"Category {category_id} deleted."}


# ---------------------------------------------------------------------------
# Product Management CRUD Routes
# ---------------------------------------------------------------------------
@app.get("/api/products")
def list_products_endpoint(
    category_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    sort_by: str = Query("newest"),
    limit: int = Query(100),
    offset: int = Query(0)
):
    """Lists products with optional category, search, status filters, and sorting."""
    return db.list_products(
        category_id=category_id,
        search=search,
        status=status,
        sort_by=sort_by,
        limit=limit,
        offset=offset
    )


@app.get("/api/products/{product_id}")
def get_product_endpoint(product_id: int):
    """Retrieves full details, status, and image assets for a single product."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")
    return {"status": "ok", "product": prod}


@app.post("/api/products")
def create_product_endpoint(req: ProductCreate):
    """Creates a new product record, auto-categorizing if category_id is omitted."""
    try:
        cat_id = req.category_id
        if not cat_id:
            c_id, _, _ = category_classifier.resolve_product_category(
                product_name=req.name,
                raw_image_path=req.raw_image,
                tags=req.tags,
                description=req.description
            )
            cat_id = c_id

        prod = db.create_product(
            name=req.name,
            sku=req.sku,
            category_id=cat_id,
            description=req.description,
            tags=req.tags,
            raw_image=req.raw_image,
            cutout_image=req.cutout_image
        )
        return {"status": "success", "product": prod}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.put("/api/products/{product_id}")
def update_product_endpoint(product_id: int, req: ProductUpdate):
    """Updates product details, metadata, and status."""
    prod = db.update_product(
        product_id=product_id,
        name=req.name,
        sku=req.sku,
        category_id=req.category_id,
        description=req.description,
        tags=req.tags,
        status=req.status
    )
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")
    return {"status": "success", "product": prod}


@app.delete("/api/products/{product_id}")
def delete_product_endpoint(product_id: int, remove_files: bool = True):
    """Deletes a product, its DB records, and all associated image assets from disk."""
    ok = db.delete_product(product_id, remove_files=remove_files)
    if not ok:
        raise HTTPException(status_code=404, detail="Product not found.")
    return {"status": "success", "message": f"Product {product_id} deleted."}


# ---------------------------------------------------------------------------
# Upload & Duplicate / Visual Similarity Checking
# ---------------------------------------------------------------------------
TEMP_CHECK_DIR = os.path.join(BASE_DIR, "data", "temp_check")
os.makedirs(TEMP_CHECK_DIR, exist_ok=True)


@app.post("/api/upload/check-duplicate")
async def check_duplicate_endpoint(file: UploadFile = File(...)):
    """
    Fast pre-flight duplicate check on an image before or during upload.
    Returns whether this image matches an existing product in the catalog.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(TEMP_CHECK_DIR, f"temp_{int(time.time()*1000)}{ext}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        dup_info = image_search.check_duplicate(temp_path)
        return dup_info
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/upload/product")
async def upload_product_with_image(
    file: UploadFile = File(...),
    name: str = Form(...),
    category_id: Optional[int] = Form(None),
    sku: Optional[str] = Form(None),
    description: Optional[str] = Form(""),
    tags: Optional[str] = Form("")
):
    """Uploads an image, creates a new product in the catalog, and indexes hashes."""
    safe_filename = os.path.basename(file.filename)
    dest_path = os.path.join(INPUTS_DIR, safe_filename)

    # If filename exists, add timestamp prefix to avoid accidental overwrite
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(safe_filename)
        safe_filename = f"{base}_{int(time.time())}{ext}"
        dest_path = os.path.join(INPUTS_DIR, safe_filename)

    norm_dest = os.path.normpath(os.path.abspath(dest_path)).lower()
    with _active_upload_lock:
        _active_upload_paths.add(norm_dest)

    try:
        with open(dest_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Refine title if cryptic
        refined_name = category_classifier.refine_product_title(
            name,
            raw_image_path=dest_path,
            filename=safe_filename
        )

        # Auto-detect category if not specified
        if not category_id:
            c_id, _, _ = category_classifier.resolve_product_category(
                product_name=refined_name,
                raw_image_path=dest_path,
                tags=tags,
                description=description,
                filename=safe_filename
            )
            category_id = c_id

        # Create product in DB (with deduplication safeguard)
        prod = db.create_product(
            name=refined_name,
            sku=sku,
            category_id=category_id,
            description=description,
            tags=tags,
            raw_image=dest_path,
            status="raw"
        )
    finally:
        with _active_upload_lock:
            _active_upload_paths.discard(norm_dest)

    return {
        "status": "success",
        "product": prod,
        "filename": safe_filename,
        "message": f"Product '{name}' created successfully."
    }


@app.post("/api/detect-category")
def detect_category_endpoint(req: DetectCategoryRequest):
    """
    Auto-detects the appropriate category for a product.
    Reuses existing category if matching, or creates a new category automatically.
    """
    cat_id, cat_name, is_new = category_classifier.resolve_product_category(
        product_name=req.name,
        filename=req.filename,
        tags=req.tags,
        description=req.description
    )
    return {
        "category_id": cat_id,
        "category_name": cat_name,
        "is_new": is_new
    }


# ---------------------------------------------------------------------------
# Global Search & Reverse Image Search
# ---------------------------------------------------------------------------
@app.get("/api/search")
def search_catalog_endpoint(
    q: str = Query("", description="Query string for name, sku, tags, category"),
    category_id: Optional[int] = Query(None),
    status: Optional[str] = Query(None)
):
    """Full-text / metadata search across the catalog."""
    return db.list_products(category_id=category_id, search=q, status=status)


@app.post("/api/search/visual")
async def visual_search_endpoint(
    file: UploadFile = File(...),
    max_distance: int = 12
):
    """
    Perceptual visual reverse image search.
    Computes visual hashes of uploaded image, matches against DB using Hamming distance,
    and returns matching products sorted by similarity percentage.
    """
    dist_val = 12
    if not hasattr(max_distance, "default"):
        try:
            dist_val = int(max_distance)
        except Exception:
            dist_val = 12

    ext = os.path.splitext(file.filename)[1].lower()
    temp_path = os.path.join(TEMP_CHECK_DIR, f"vis_{int(time.time()*1000)}{ext}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        matches = image_search.reverse_image_search(temp_path, top_k=8, max_distance=dist_val)
        return {
            "status": "success",
            "matches": matches,
            "total": len(matches)
        }
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.post("/api/search/by-image")
async def reverse_image_search_endpoint(file: UploadFile = File(...)):
    """Reverse visual search: uploads a query image to locate matching/similar products."""
    return await visual_search_endpoint(file, max_distance=12)


# ---------------------------------------------------------------------------
# Duplicate Detection & 1-Click Batch Cleanup Routes
# ---------------------------------------------------------------------------
@app.get("/api/duplicates")
def get_duplicates_endpoint(max_distance: int = Query(6, description="Max hamming distance threshold for duplicates")):
    """
    Scans the entire product catalog for duplicate items using exact raw paths and perceptual visual dHash.
    Clusters duplicates, automatically selects the best primary item to KEEP, and returns candidates to delete.
    """
    return db.get_catalog_duplicates(max_hamming_distance=max_distance)


@app.post("/api/duplicates/cleanup")
def cleanup_duplicates_endpoint(req: CleanupDuplicatesRequest):
    """
    Deletes specified duplicate product IDs or all catalog duplicates in 1 click.
    Safely purges redundant product records from the database while keeping primary original files intact.
    """
    return db.cleanup_catalog_duplicates(product_ids=req.product_ids)


# ---------------------------------------------------------------------------
# Product-Linked AI Pipelines (Matting, Multi-Angles, Stylize, Beautify, Export)
# ---------------------------------------------------------------------------
@app.post("/api/products/{product_id}/remove-bg")
def product_remove_background_endpoint(product_id: int):
    """Runs BiRefNet on the product's raw image and updates cutout asset."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    raw_path = prod.get("raw_image")
    if not raw_path or not os.path.exists(raw_path):
        raise HTTPException(status_code=400, detail="Product has no valid source image in inputs/.")

    base_name = os.path.splitext(os.path.basename(raw_path))[0]
    out_path = os.path.join(PROCESSED_DIR, f"{base_name}_cutout.png")

    try:
        model, device = get_model()
        std_rgb, orig_sz = load_and_standardize_image(raw_path)
        cutout = remove_background(std_rgb, orig_sz, model, device)
        cutout.save(out_path, format="PNG")

        db.record_product_asset(product_id, "cutout", out_path)
        cutout_url = f"/processed_images/{os.path.basename(out_path)}"
        broadcast_event("product_updated", {
            "id": product_id,
            "status": "cutout_ready",
            "cutout_url": cutout_url
        })
        updated = db.get_product(product_id)
        return {
            "status": "success",
            "cutout_url": cutout_url,
            "product": updated
        }
    except Exception as e:
        print(f"[Error remove-bg product {product_id}]: {e}")
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")


@app.post("/api/products/{product_id}/generate-angles")
def product_generate_angles_endpoint(product_id: int):
    """
    Synthesizes 6 canonical camera perspectives for the specified product.
    Updates SQLite DB product asset tracking and sets product.status = 'angles_ready'.
    """
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    cutout_path = prod.get("cutout_image")
    if not cutout_path or not os.path.exists(cutout_path):
        # Fallback to check candidate path in processed_images
        raw_path = prod.get("raw_image") or ""
        base_cand = os.path.splitext(os.path.basename(raw_path))[0]
        cand_path = os.path.join(PROCESSED_DIR, f"{base_cand}_cutout.png")
        if os.path.exists(cand_path):
            cutout_path = cand_path
        else:
            raise HTTPException(status_code=400, detail="Cutout image not available. Please remove background on Tab 1 first.")

    raw_path = prod.get("raw_image") or ""
    base_name = os.path.splitext(os.path.basename(raw_path))[0] if raw_path else prod["name"].lower().replace(" ", "_")

    try:
        res = generate_product_angles(cutout_path, base_name=base_name)
        # Record assets in DB
        for v in res.get("views", []):
            db.record_product_asset(product_id, "angle", v["path"], metadata={"azimuth": v.get("azimuth"), "elevation": v.get("elevation")})

        updated = db.get_product(product_id)
        res["product"] = updated
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"3D angle synthesis failed: {str(e)}")


def resolve_product_angles_base(prod: dict) -> Tuple[str, bool]:
    """Resolves the output_angles directory base name for a product and checks if angles exist."""
    raw_path = prod.get("raw_image") or ""
    raw_base = os.path.splitext(os.path.basename(raw_path))[0] if raw_path else ""
    name_base = re.sub(r"[^a-z0-9_]+", "", prod.get("name", "").lower().replace(" ", "_"))

    candidates = [
        raw_base,
        raw_base.replace("_cutout", ""),
        name_base,
        prod.get("name", "").lower()
    ]
    candidates = [c for c in candidates if c]

    for cand in candidates:
        cand_dir = os.path.join(OUTPUT_ANGLES_DIR, cand)
        if os.path.isdir(cand_dir):
            files = [f for f in os.listdir(cand_dir) if f.endswith(".png") and f != "angles_grid.png"]
            if files:
                return cand, True

    # Check database product_images
    p_images = prod.get("images") or []
    for img in p_images:
        if img.get("image_type") == "angle":
            f_path = img.get("file_path") or ""
            if f_path and os.path.exists(f_path):
                parent_dir = os.path.basename(os.path.dirname(f_path))
                return parent_dir, True

    fallback = raw_base or name_base or "product"
    return fallback, False


@app.get("/api/products/{product_id}/angles")
def product_get_angles_endpoint(product_id: int):
    """Retrieves 3D camera angles specifically for this product, or indicates if not yet synthesized."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    base_name, has_angles = resolve_product_angles_base(prod)
    if not has_angles:
        return {
            "status": "not_synthesized",
            "has_angles": False,
            "product_id": product_id,
            "base_name": base_name,
            "views": [],
            "message": f"Perspectives have not been synthesized for {prod['name']} yet."
        }

    views = []
    prod_dir = os.path.join(OUTPUT_ANGLES_DIR, base_name)
    if os.path.isdir(prod_dir):
        for meta in ANGLE_METADATA:
            f_name = f"{meta['id']}.png"
            f_path = os.path.join(prod_dir, f_name)
            if os.path.exists(f_path):
                views.append({
                    **meta,
                    "label": meta.get("name", meta.get("id")),
                    "filename": f_name,
                    "path": f_path,
                    "url": f"/output_angles/{base_name}/{f_name}",
                    "size_kb": round(os.path.getsize(f_path) / 1024, 1)
                })

    return {
        "status": "success",
        "has_angles": len(views) > 0,
        "product_id": product_id,
        "base_name": base_name,
        "total_angles": len(views),
        "views": views,
        "angles": views,
        "grid_url": f"/output_angles/{base_name}/angles_grid.png",
        "gif_url": f"/api/products/{product_id}/angles-gif"
    }


@app.get("/api/products/{product_id}/angles-gif")
def product_view_angles_gif_endpoint(product_id: int, duration: int = 65):
    """Streams the 360-degree rotating GIF specifically for this product."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    base_name, has_angles = resolve_product_angles_base(prod)
    if not has_angles:
        raise HTTPException(status_code=404, detail=f"360° spin has not been synthesized for {prod['name']} yet.")

    try:
        gif_buf = create_angles_gif(base_name, duration_ms=duration)
        return StreamingResponse(gif_buf, media_type="image/gif")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}/export-angles-zip")
def product_export_angles_zip_endpoint(product_id: int):
    """Downloads all 6 camera angles bundled in a ZIP archive for this specific product."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    base_name, has_angles = resolve_product_angles_base(prod)
    if not has_angles:
        raise HTTPException(status_code=400, detail=f"3D camera perspectives have not been synthesized for {prod['name']} yet. Please click 'Synthesize 6 Perspectives' first.")

    try:
        zip_buf = create_angles_zip(base_name)
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prod["name"]).lower()
        return StreamingResponse(
            zip_buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}_perspectives.zip"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/products/{product_id}/export-angles-gif")
def product_export_angles_gif_endpoint(product_id: int, duration: int = 65):
    """Downloads the 360-degree rotating GIF as an attachment for this specific product."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    base_name, has_angles = resolve_product_angles_base(prod)
    if not has_angles:
        raise HTTPException(status_code=400, detail=f"360° spin has not been synthesized for {prod['name']} yet. Please click 'Synthesize 6 Perspectives' first.")

    try:
        gif_buf = create_angles_gif(base_name, duration_ms=duration)
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]+", "_", prod["name"]).lower()
        return StreamingResponse(
            gif_buf,
            media_type="image/gif",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}_360_turntable.gif"'}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/products/{product_id}/stylize")
def product_stylize_endpoint(product_id: int, req: GenerateSceneRequest):
    """Renders a product cutout onto a commercial scene preset or custom prompt."""
    prod = db.get_product(product_id)
    if not prod:
        raise HTTPException(status_code=404, detail="Product not found.")

    cutout_path = prod.get("cutout_image")
    if not cutout_path or not os.path.exists(cutout_path):
        safe_cutout = os.path.basename(req.cutout_filename) if req.cutout_filename else ""
        candidate = os.path.join(PROCESSED_DIR, safe_cutout)
        if os.path.exists(candidate):
            cutout_path = candidate
        else:
            raise HTTPException(status_code=400, detail="Cutout image not available. Please remove background first.")

    try:
        res = generate_product_scene(
            cutout_path=cutout_path,
            preset_id=req.preset_id,
            aspect_ratio=req.aspect_ratio,
            custom_prompt=req.custom_prompt,
            shadow_intensity=req.shadow_intensity
        )
        scene_path = res.get("output_path") or res.get("file_path")
        if scene_path and os.path.exists(scene_path):
            db.record_product_asset(product_id, "scene", scene_path, metadata={"preset": req.preset_id, "ar": req.aspect_ratio})

        res["url"] = res.get("output_url")
        res["file_url"] = res.get("output_url")
        res["file_path"] = scene_path
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stylization failed: {str(e)}")


@app.get("/api/export-dimensions")
def get_export_dimensions():
    """Returns standard commercial e-commerce export dimensions."""
    return {"dimensions": beautify_service.EXPORT_DIMENSIONS}


@app.post("/api/export")
def export_image_endpoint(req: ExportImageRequest):
    """Exports and beautifies an image in requested format (PNG, JPG, WebP, TIFF) & size."""
    safe_name = os.path.basename(req.source_image)
    candidates = [
        os.path.join(PROCESSED_DIR, safe_name),
        os.path.join(GENERATED_DIR, safe_name),
        os.path.join(INPUTS_DIR, safe_name),
        os.path.join(BASE_DIR, safe_name),
    ]
    source_path = next((p for p in candidates if os.path.exists(p)), None)
    if not source_path:
        raise HTTPException(status_code=404, detail=f"Source image '{safe_name}' not found.")

    try:
        res = beautify_service.render_and_export_image(
            source_image_path=source_path,
            output_format=req.output_format,
            aspect_ratio=req.aspect_ratio,
            target_width=req.target_width,
            target_height=req.target_height,
            dpi=req.dpi,
            background_color=req.background_color,
            auto_enhance=req.auto_enhance,
            brightness=req.brightness,
            contrast=req.contrast,
            saturation=req.saturation,
            sharpness=req.sharpness
        )
        res["download_url"] = res.get("file_url")
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# Mount static assets & storage directories for direct image viewing in the browser
app.mount("/inputs", StaticFiles(directory=INPUTS_DIR), name="inputs")
app.mount("/processed_images", StaticFiles(directory=PROCESSED_DIR), name="processed_images")
app.mount("/generated_scenes", StaticFiles(directory=GENERATED_DIR), name="generated_scenes")
app.mount("/output_angles", StaticFiles(directory=OUTPUT_ANGLES_DIR), name="output_angles")
app.mount("/exports", StaticFiles(directory=EXPORTS_DIR), name="exports")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_dashboard():
    """Serves the main HTML dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse({"status": "E-Commerce Studio Running", "docs": "/docs"})


if __name__ == "__main__":
    print("=" * 70)
    print("STARTING E-COMMERCE SELLER STUDIO SERVER")
    print("=" * 70)
    print("Local Dashboard URL: http://localhost:8000")
    print("REST API Docs      : http://localhost:8000/docs")
    print("Inputs Directory   : " + INPUTS_DIR)
    print("Processed Directory: " + PROCESSED_DIR)
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
