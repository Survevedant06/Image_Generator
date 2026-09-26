#!/usr/bin/env python3
"""
batch_remove_bg.py - Production-Ready Dynamic Queue & Continuous Watcher Background Removal

Features:
  1. Multi-Directory Automatic Detection:
     - Automatically scans 'inputs/', 'input/', and the root directory '.' simultaneously.
     - Supports all image formats: .jpg, .jpeg, .png, .webp, .bmp, .tiff, .heic, .heif.
     - Automatic EXIF orientation correction via ImageOps.exif_transpose and pillow-heif.
  2. Bulletproof State Tracking & Duplicate Prevention:
     - Output directory: 'processed_images/'.
     - Checks if '<filename>_cutout.png' already exists.
     - Automatically skips previously processed images.
     - Processes ONLY new/pending images where the background has not yet been removed.
  3. Continuous Watch Mode (--watch / -w):
     - Keeps running in the background and continuously monitors for new images.
     - Keeps BiRefNet model cached in memory for instant processing upon file arrival.
     - Zero human effort: just drop images into the folder and cutouts appear automatically!
  4. Robust Model Lifecycle & Error Handling:
     - Single model load (float32), reused under torch.no_grad().
     - Per-file exception handling so corrupted files never interrupt the batch.
"""

import os
import sys
import gc
import time
import argparse
from typing import List, Tuple, Dict, Any, Optional

from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Format Support & Environment Setup
# ---------------------------------------------------------------------------
# Suppress Windows symlink warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Support optional Hugging Face authentication token for rate-limit boosts
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif"
)

EXCLUDED_DIR_NAMES = {
    "processed_images",
    "processed images",
    "output_angles",
    "output_angles_transparent",
    "output_angles_standardized",
    "zero123plus_repo",
    ".git",
    "__pycache__"
}

DEFAULT_OUTPUT_DIR = "processed_images"
DEFAULT_CANDIDATE_INPUT_DIRS = ["inputs", "input", "Inputs", "Input", "."]


# ---------------------------------------------------------------------------
# File & Queue Validation
# ---------------------------------------------------------------------------
_validity_cache: Dict[str, Tuple[float, int, bool]] = {}


def is_valid_input_image(full_path: str) -> bool:
    """Validates that a file is a supported image and not an output artifact."""
    if not os.path.isfile(full_path):
        return False

    filename = os.path.basename(full_path)
    lower = filename.lower()
    if not lower.endswith(SUPPORTED_EXTENSIONS):
        return False

    # Ignore generated cutouts and output grids to prevent infinite re-processing
    if (
        lower.endswith("_cutout.png")
        or lower.startswith("output_")
        or lower.startswith("angles_")
        or "_cutout" in lower
    ):
        return False

    # Verify that file is actually a readable image (e.g. not an HTML error page or empty file)
    try:
        stat = os.stat(full_path)
        mtime = stat.st_mtime
        size = stat.st_size
        if size == 0:
            return False

        cached = _validity_cache.get(full_path)
        if cached and cached[0] == mtime and cached[1] == size:
            return cached[2]

        with Image.open(full_path) as im:
            im.verify()
        _validity_cache[full_path] = (mtime, size, True)
        return True
    except Exception:
        _validity_cache[full_path] = (mtime if 'mtime' in locals() else 0, size if 'size' in locals() else 0, False)
        return False


def resolve_output_dir(specified_output: Optional[str] = None) -> str:
    """Finds or creates the processed images directory (supporting spaces or underscores)."""
    if specified_output:
        return specified_output
    for candidate in ("processed_images", "processed images", "Processed_Images", "Processed Images"):
        if os.path.isdir(candidate):
            return candidate
    return DEFAULT_OUTPUT_DIR


def scan_all_queues(
    specified_input: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR
) -> Tuple[List[str], List[str]]:
    """
    Scans all source directories (specified input or inputs/ + current directory)
    and separates files into (pending_files, skipped_files).
    Pending: Cutout does NOT exist in output_dir.
    Skipped: Cutout ALREADY exists in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    if specified_input:
        search_dirs = [specified_input]
    else:
        # Check all possible input directories that exist
        search_dirs = [d for d in DEFAULT_CANDIDATE_INPUT_DIRS if os.path.isdir(d)]
        # Remove duplicates while preserving order
        seen_dirs = set()
        unique_dirs = []
        for d in search_dirs:
            norm = os.path.normpath(d)
            if norm not in seen_dirs:
                seen_dirs.add(norm)
                unique_dirs.append(d)
        search_dirs = unique_dirs

    pending_files: List[str] = []
    skipped_files: List[str] = []
    seen_base_names = set()

    for directory in search_dirs:
        try:
            entries = sorted(os.listdir(directory))
        except OSError:
            continue

        for entry in entries:
            full_path = os.path.join(directory, entry)
            if not is_valid_input_image(full_path):
                continue

            base_name = os.path.splitext(entry)[0]
            # Avoid duplicate processing if same image name exists in multiple folders
            if base_name in seen_base_names:
                continue
            seen_base_names.add(base_name)

            expected_output = os.path.join(output_dir, f"{base_name}_cutout.png")
            if os.path.exists(expected_output):
                skipped_files.append(full_path)
            else:
                pending_files.append(full_path)

    return pending_files, skipped_files


# ---------------------------------------------------------------------------
# Image Ingestion & Normalization
# ---------------------------------------------------------------------------
def load_and_standardize_image(image_path: str) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Safely opens any image format (including HEIC/HEIF), normalizes EXIF orientation,
    and returns a standard RGB image alongside original dimensions.
    """
    raw_img = Image.open(image_path)
    raw_img = ImageOps.exif_transpose(raw_img)
    orig_size = raw_img.size

    if raw_img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", orig_size, (255, 255, 255))
        alpha_channel = raw_img.split()[-1]
        background.paste(raw_img, mask=alpha_channel)
        standard_rgb = background
    elif raw_img.mode != "RGB":
        standard_rgb = raw_img.convert("RGB")
    else:
        standard_rgb = raw_img

    return standard_rgb, orig_size


# ---------------------------------------------------------------------------
# Model Initialization (Loaded once and cached in memory)
# ---------------------------------------------------------------------------
def load_birefnet_model(device: str):
    """
    Loads BiRefNet segmentation model once into memory using float32
    to prevent precision mismatch errors. Sets optimal CPU thread count.
    Uses local_files_only=True to prevent network hangs and rate limits.
    """
    import torch
    from transformers import AutoModelForImageSegmentation

    if device == "cpu":
        torch.set_num_threads(4)

    print(f"--> Initializing BiRefNet segmentation engine on {device} (float32)...")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float32,
    }
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    # Try local cache first to avoid slow HuggingFace network calls and rate limits
    model = None
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet",
            local_files_only=True,
            **model_kwargs
        )
    except Exception as local_err:
        print(f"--> Notice: Local cache load ({local_err}), downloading from HF Hub...")
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                **model_kwargs
            )
        except TypeError:
            model_kwargs.pop("dtype", None)
            model_kwargs["torch_dtype"] = torch.float32
            model = AutoModelForImageSegmentation.from_pretrained(
                "ZhengPeng7/BiRefNet",
                **model_kwargs
            )

    model.to(device)
    model.float()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core Inference
# ---------------------------------------------------------------------------
def remove_background(
    standard_rgb: Image.Image,
    orig_size: Tuple[int, int],
    model,
    device: str,
    inference_size: Optional[Tuple[int, int]] = None
) -> Image.Image:
    """
    Executes segmentation inference and produces full-resolution transparent RGBA PNG.
    Uses accelerated inference_size (512x512 on CPU gives >10x speedup: ~6.1s vs 65s)
    with bilinear mask reconstruction preserving full original resolution and sharp contours.
    """
    import torch
    from torchvision import transforms

    if inference_size is None:
        inference_size = (512, 512) if device == "cpu" else (1024, 1024)

    if device == "cpu":
        torch.set_num_threads(4)

    transform = transforms.Compose([
        transforms.Resize(inference_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(standard_rgb).unsqueeze(0).to(device).float()

    with torch.inference_mode():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred_mask = preds[0].squeeze().float()
    mask_pil = transforms.ToPILImage()(pred_mask)
    mask_pil = mask_pil.resize(orig_size, Image.Resampling.BILINEAR)

    cutout_rgba = standard_rgb.copy().convert("RGBA")
    cutout_rgba.putalpha(mask_pil)

    del input_tensor, preds, pred_mask, mask_pil
    if device == "cuda":
        torch.cuda.empty_cache()

    return cutout_rgba


# ---------------------------------------------------------------------------
# Batch Processing Execution
# ---------------------------------------------------------------------------
def process_pending_images(
    pending_files: List[str],
    output_dir: str,
    model=None,
    device: str = "cpu"
) -> Tuple[int, List[Tuple[str, str]], Any]:
    """
    Processes a list of pending image paths and saves cutouts to output_dir.
    Returns (success_count, failed_files, model).
    """
    import torch

    if model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        t_model_start = time.perf_counter()
        model = load_birefnet_model(device)
        print(f"[Ready] BiRefNet model cached in memory ({time.perf_counter() - t_model_start:.2f}s)\n")

    success_count = 0
    failed_files: List[Tuple[str, str]] = []

    for i, file_path in enumerate(pending_files, 1):
        filename = os.path.basename(file_path)
        base_name = os.path.splitext(filename)[0]
        out_path = os.path.join(output_dir, f"{base_name}_cutout.png")

        t_img = time.perf_counter()
        print(f"[{i}/{len(pending_files)}] Ingesting '{filename}' from '{os.path.dirname(file_path) or '.'}'...")

        try:
            standard_rgb, orig_size = load_and_standardize_image(file_path)
            cutout_rgba = remove_background(standard_rgb, orig_size, model, device)

            # Save in processed_images/
            cutout_rgba.save(out_path, format="PNG")

            # Also save a copy in root if image was from root
            if os.path.normpath(os.path.dirname(file_path)) in (".", ""):
                cutout_rgba.save(f"{base_name}_cutout.png", format="PNG")

            elapsed = time.perf_counter() - t_img
            print(f"    [OK] Saved cutout -> '{out_path}' ({orig_size[0]}x{orig_size[1]}, {elapsed:.2f}s)")
            success_count += 1

        except Exception as err:
            elapsed = time.perf_counter() - t_img
            print(f"    [Error] Failed to process '{filename}' ({elapsed:.2f}s): {err}")
            failed_files.append((file_path, str(err)))

    return success_count, failed_files, model


# ---------------------------------------------------------------------------
# Continuous Watcher Mode
# ---------------------------------------------------------------------------
def run_watcher(
    specified_input: Optional[str] = None,
    output_dir: Optional[str] = None,
    poll_interval: float = 2.0
):
    """
    Continuously monitors folders for new images and processes them automatically.
    """
    resolved_out = resolve_output_dir(output_dir)
    os.makedirs(resolved_out, exist_ok=True)

    print("=" * 80)
    print("BIREFNET CONTINUOUS FOLDER WATCHER RUNNING")
    print("=" * 80)
    print(f"Monitoring Folders   : inputs/, input/, and current directory")
    print(f"Processed Cutouts Dir: {os.path.abspath(resolved_out)}")
    print(f"Polling Interval     : {poll_interval}s")
    print("Status               : ACTIVE — Drop any image into 'inputs/' to process.")
    print("Press Ctrl+C to stop watcher.")
    print("=" * 80)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_birefnet_model(device)
    print(f"[Ready] BiRefNet model cached in memory. Watching for new images...\n")

    try:
        while True:
            pending_files, skipped_files = scan_all_queues(specified_input, resolved_out)

            if pending_files:
                print(f"\n[Watcher Event] Detected {len(pending_files)} new unprocessed image(s)!")
                success, failed, model = process_pending_images(
                    pending_files=pending_files,
                    output_dir=resolved_out,
                    model=model,
                    device=device
                )
                print(f"[Watcher] Finished processing {success}/{len(pending_files)} images. Resuming watch...\n")

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\n[Watcher] Stopped by user.")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


# ---------------------------------------------------------------------------
# Single-Run Batch Mode
# ---------------------------------------------------------------------------
def process_batch(
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None
):
    """
    One-time batch execution: scans all locations, skips processed images,
    processes all pending images, and exits.
    """
    resolved_out = resolve_output_dir(output_dir)
    os.makedirs(resolved_out, exist_ok=True)

    pending_files, skipped_files = scan_all_queues(input_dir, resolved_out)
    total_discovered = len(pending_files) + len(skipped_files)

    print("=" * 80)
    print("BIREFNET DYNAMIC QUEUE BACKGROUND REMOVAL")
    print("=" * 80)
    print(f"Output Directory     : {os.path.abspath(resolved_out)}")
    print(f"Total Discovered     : {total_discovered} image(s)")
    print(f"Already Processed    : {len(skipped_files)} image(s) [Skipped]")
    print(f"Pending to Process   : {len(pending_files)} image(s)")
    print("=" * 80)

    if skipped_files:
        print("[State Tracking] Previously processed images (Will NOT be reprocessed):")
        for s in skipped_files:
            print(f"  * {os.path.basename(s)} -> Cutout exists in '{resolved_out}/'")
        print("-" * 80)

    if not pending_files:
        if total_discovered == 0:
            print("[Status] No input images found in 'inputs/' or current directory.")
            print(f"         Drop any image ({', '.join(SUPPORTED_EXTENSIONS)}) into 'inputs/' and rerun.")
        else:
            print(f"[Status] All {total_discovered} image(s) have already been processed into '{resolved_out}/'.")
            print("         No new images pending. Exiting cleanly.")
        return

    print(f"[Queue] Pending images to process: {', '.join(os.path.basename(f) for f in pending_files)}")

    batch_start = time.perf_counter()
    success_count, failed_files, model = process_pending_images(
        pending_files=pending_files,
        output_dir=resolved_out
    )

    del model
    gc.collect()

    total_time = time.perf_counter() - batch_start
    avg_time = (total_time / success_count) if success_count > 0 else 0.0

    print("\n" + "=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total Completed : {success_count}/{len(pending_files)}")
    print(f"Total Skipped   : {len(skipped_files)}")
    print(f"Total Errors    : {len(failed_files)}")
    print(f"Total Runtime   : {total_time:.2f}s (Avg: {avg_time:.2f}s/image)")
    print(f"Processed Folder: {os.path.abspath(resolved_out)}")

    if failed_files:
        print("-" * 80)
        print("Failed Files Log:")
        for name, err_msg in failed_files:
            print(f"  - {os.path.basename(name)}: {err_msg}")

    print("=" * 80)


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated dynamic queue background removal with continuous watcher mode."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        help="Input folder containing images (default: auto-scans inputs/ and current directory)"
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        help="Output folder for transparent cutouts (default: 'processed_images')"
    )
    parser.add_argument(
        "--watch", "-w",
        action="store_true",
        help="Continuous watcher mode: keep running and automatically process any new image dropped into the folder"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=2.0,
        help="Polling interval in seconds for watcher mode (default: 2.0s)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.watch:
        run_watcher(
            specified_input=args.input_dir,
            output_dir=args.output_dir,
            poll_interval=args.interval
        )
    else:
        process_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir
        )
