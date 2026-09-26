#!/usr/bin/env python3
"""
run_pipeline.py - Unified In-Memory Pipeline for Background Removal, Novel-View Synthesis,
                  and Automated Secondary Background Stripping.

Workflow:
  1. Input Ingestion: Load any format (HEIC/PNG/JPG/etc.) & normalize EXIF rotation.
  2. Primary Matting: BiRefNet generates transparent cutout of the input image.
  3. Cutout Export: Saves primary cutout BOTH separately in root and inside output folders.
  4. Conditioning Canvas: In-memory white background composition for Zero123++.
  5. Novel-View Synthesis: Zero123++ generates 6 novel camera angles in a 3x2 grid.
  6. Slicing: Extract the 6 individual camera views into 'output_angles/'.
  7. Secondary Matting: Reuse in-memory BiRefNet to strip diffusion studio background.
  8. Alpha Cleanup & De-fringing: Eliminate edge halos and smooth boundary contours.
  9. Export & Verification: Save transparent views to 'output_angles_transparent/'
     and display a side-by-side verification audit table.
"""

import os
import sys
import gc
import time
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps

# ---------------------------------------------------------------------------
# Format Support & Environment Setup
# ---------------------------------------------------------------------------
# Suppress Windows symlinks warning from huggingface_hub
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Support Hugging Face authentication token from environment
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

# Dynamically locate and add zero123plus repository to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(BASE_DIR, "zero123plus_repo")

if os.path.exists(REPO_ROOT):
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    for root, dirs, files in os.walk(REPO_ROOT):
        if "pipeline.py" in files or any(f.endswith(".py") for f in files):
            if root not in sys.path:
                sys.path.insert(0, root)

try:
    try:
        from zero123plus.pipeline import Zero123PlusPipeline
    except ImportError:
        from pipeline import Zero123PlusPipeline

    from diffusers import EulerAncestralDiscreteScheduler
except ImportError as e:
    print(f"[Error] Failed to import Zero123PlusPipeline: {e}")
    sys.exit(1)

from transformers import AutoModelForImageSegmentation

SUPPORTED_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".heic", ".heif"
)

VIEW_NAMES = [
    "angle_front_right_low",
    "angle_front_right_high",
    "angle_rear_right_low",
    "angle_rear_right_high",
    "angle_rear_left_low",
    "angle_rear_left_high"
]


# ---------------------------------------------------------------------------
# Image Ingestion & Standardization
# ---------------------------------------------------------------------------
def load_and_standardize_image(image_path: str) -> Image.Image:
    """
    Safely opens any image format (including HEIC/HEIF), normalizes EXIF orientation,
    and returns a standard RGB image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    raw_img = Image.open(image_path)
    raw_img = ImageOps.exif_transpose(raw_img)

    if raw_img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", raw_img.size, (255, 255, 255))
        alpha_channel = raw_img.split()[-1]
        background.paste(raw_img, mask=alpha_channel)
        return background
    elif raw_img.mode != "RGB":
        return raw_img.convert("RGB")

    return raw_img


# ---------------------------------------------------------------------------
# Alpha Cleanup & Halo Removal
# ---------------------------------------------------------------------------
def clean_alpha_mask(
    mask: Image.Image,
    low_thresh: float = 0.06,
    high_thresh: float = 0.92
) -> Image.Image:
    """
    Eliminates low-alpha background haze via smoothstep thresholding and clamps
    confident foreground to 255, avoiding jagged staircase boundaries.
    """
    mask_np = np.array(mask).astype(np.float32) / 255.0

    # Smoothstep curve: 3t^2 - 2t^3
    t = np.clip((mask_np - low_thresh) / (high_thresh - low_thresh + 1e-6), 0.0, 1.0)
    smooth_mask = t * t * (3.0 - 2.0 * t)

    cleaned_pil = Image.fromarray((smooth_mask * 255.0).astype(np.uint8), mode="L")
    return cleaned_pil


def defringe_cutout(
    image_rgba: Image.Image,
    bg_color: Tuple[int, int, int] = (250, 250, 250)
) -> Image.Image:
    """
    Unblends the studio background color from semi-transparent edge pixels
    to completely eliminate off-white / light-gray boundary fringes.
    """
    img_np = np.array(image_rgba).astype(np.float32)
    rgb = img_np[:, :, :3]
    alpha = img_np[:, :, 3:4] / 255.0

    bg = np.array(bg_color, dtype=np.float32).reshape(1, 1, 3)
    # Only target semi-transparent boundary pixels
    edge_mask = (alpha > 0.01) & (alpha < 0.98)

    unmixed_rgb = np.where(
        edge_mask,
        np.clip((rgb - (1.0 - alpha) * bg) / np.maximum(alpha, 0.05), 0.0, 255.0),
        rgb
    )

    result_np = np.concatenate([unmixed_rgb, img_np[:, :, 3:4]], axis=2)
    return Image.fromarray(result_np.astype(np.uint8), mode="RGBA")


# ---------------------------------------------------------------------------
# Model Management: BiRefNet
# ---------------------------------------------------------------------------
def load_birefnet_model(device: str):
    """
    Loads BiRefNet segmentation model once into memory using dtype=torch.float32
    to prevent precision mismatch and deprecation errors.
    """
    print(f"--> Loading BiRefNet segmentation model on {device} (float32)...")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.float32,
    }
    if HF_TOKEN:
        model_kwargs["token"] = HF_TOKEN

    model = AutoModelForImageSegmentation.from_pretrained(
        "ZhengPeng7/BiRefNet",
        **model_kwargs
    )
    model.to(device)
    model.float()
    model.eval()
    return model


def run_birefnet_inference(
    image: Image.Image,
    model,
    device: str,
    apply_cleanup: bool = False
) -> Image.Image:
    """
    Executes BiRefNet segmentation on a single image entirely in memory.
    Optionally applies smooth alpha thresholding and edge de-fringing.
    """
    original_size = image.size
    standard_rgb = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(standard_rgb).unsqueeze(0).to(device).float()

    with torch.no_grad():
        preds = model(input_tensor)[-1].sigmoid().cpu()

    pred_mask = preds[0].squeeze().float()
    mask_pil = transforms.ToPILImage()(pred_mask)
    mask_pil = mask_pil.resize(original_size, Image.Resampling.BILINEAR)

    if apply_cleanup:
        mask_pil = clean_alpha_mask(mask_pil)

    cutout_rgba = standard_rgb.copy().convert("RGBA")
    cutout_rgba.putalpha(mask_pil)

    if apply_cleanup:
        cutout_rgba = defringe_cutout(cutout_rgba)

    return cutout_rgba


# ---------------------------------------------------------------------------
# Conditioning for Multiview
# ---------------------------------------------------------------------------
def prepare_conditioning_canvas(cutout_rgba: Image.Image) -> Image.Image:
    """
    Composites the transparent RGBA cutout over a pure white canvas
    to prepare the conditioning input expected by Zero123++.
    """
    print("--> [Conditioning] Compositing cutout over white canvas in memory...")
    white_bg = Image.new("RGBA", cutout_rgba.size, (255, 255, 255, 255))
    conditioned_img = Image.alpha_composite(white_bg, cutout_rgba).convert("RGB")
    return conditioned_img


# ---------------------------------------------------------------------------
# Novel-View Synthesis (Zero123++)
# ---------------------------------------------------------------------------
def synthesize_multiview(conditioned_img: Image.Image, device: str) -> Image.Image:
    """
    Loads Zero123PlusPipeline, configures EulerAncestralDiscreteScheduler with
    trailing timestep spacing, and executes 28 denoising steps.
    """
    print(f"--> [Diffusion] Loading Zero123++ diffusion model on {device}...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe_kwargs = {"torch_dtype": dtype}
    if HF_TOKEN:
        pipe_kwargs["token"] = HF_TOKEN

    pipeline = Zero123PlusPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        **pipe_kwargs
    )

    pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipeline.scheduler.config,
        timestep_spacing="trailing"
    )
    pipeline.to(device)

    print("--> [Diffusion] Running multi-view diffusion synthesis (28 steps)...")
    with torch.no_grad():
        output_grid = pipeline(
            conditioned_img,
            num_inference_steps=28,
            guidance_scale=4.0
        ).images[0]

    del pipeline
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return output_grid


# ---------------------------------------------------------------------------
# Slicing & Grid Operations
# ---------------------------------------------------------------------------
def slice_grid_views(grid_image: Image.Image) -> List[Tuple[str, Image.Image]]:
    """
    Slices the 3x2 composite grid into 6 named PIL view images.
    """
    width, height = grid_image.size
    cols, rows = 2, 3
    tile_w = width // cols
    tile_h = height // rows

    sliced_views = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            box = (c * tile_w, r * tile_h, (c + 1) * tile_w, (r + 1) * tile_h)
            tile = grid_image.crop(box)
            name = VIEW_NAMES[idx]
            sliced_views.append((name, tile))
            idx += 1

    return sliced_views


def stitch_transparent_grid(transparent_views: List[Tuple[str, Image.Image]]) -> Image.Image:
    """
    Stitches the 6 transparent angle views back into a clean 3x2 transparent RGBA composite grid.
    """
    if not transparent_views:
        return None

    tile_w, tile_h = transparent_views[0][1].size
    grid_w = tile_w * 2
    grid_h = tile_h * 3

    grid_rgba = Image.new("RGBA", (grid_w, grid_h), (0, 0, 0, 0))
    idx = 0
    for r in range(3):
        for c in range(2):
            if idx < len(transparent_views):
                tile = transparent_views[idx][1]
                grid_rgba.paste(tile, (c * tile_w, r * tile_h))
                idx += 1

    return grid_rgba


# ---------------------------------------------------------------------------
# Secondary Background Stripping Pass
# ---------------------------------------------------------------------------
def secondary_background_removal(
    sliced_views: List[Tuple[str, Image.Image]],
    birefnet_model,
    device: str
) -> List[Tuple[str, Image.Image]]:
    """
    Passes all sliced camera views through the in-memory BiRefNet model instance
    to strip off-white/gray studio backgrounds and apply anti-halo cleanup.
    """
    print("\n" + "=" * 60)
    print("SECONDARY PASS: STRIPPING STUDIO BACKGROUND FROM NOVEL VIEWS")
    print("=" * 60)
    print(f"Reusing loaded BiRefNet instance in memory across {len(sliced_views)} views...")

    transparent_views = []
    for i, (name, img) in enumerate(sliced_views, 1):
        t_view = time.perf_counter()
        print(f"--> [Matting {i}/{len(sliced_views)}] Processing '{name}'...")
        transp_cutout = run_birefnet_inference(
            image=img,
            model=birefnet_model,
            device=device,
            apply_cleanup=True
        )
        elapsed = time.perf_counter() - t_view
        print(f"    Cleaned alpha and de-fringed in {elapsed:.2f}s")
        transparent_views.append((name, transp_cutout))

    return transparent_views


# ---------------------------------------------------------------------------
# Export & Verification Audit
# ---------------------------------------------------------------------------
def export_and_audit(
    input_image_path: str,
    cutout_rgba: Image.Image,
    grid_image: Image.Image,
    output_grid_path: str,
    studio_views: List[Tuple[str, Image.Image]],
    transparent_views: List[Tuple[str, Image.Image]],
    studio_dir: str = "output_angles",
    transparent_dir: str = "output_angles_transparent"
) -> List[Dict[str, Any]]:
    """
    Saves:
      1. Primary cutout:
         - Separately in root directory (<base>_cutout.png)
         - Inside transparent output folder (output_angles_transparent/<base>_cutout.png)
         - Inside studio folder (output_angles/<base>_cutout.png)
      2. Multi-view composite grids (both studio & transparent):
         - Separately in root directory
         - Inside corresponding output folders
      3. Sliced 6 camera views:
         - Studio views in output_angles/
         - Transparent assets in output_angles_transparent/
    Outputs a complete side-by-side verification audit table.
    """
    os.makedirs(studio_dir, exist_ok=True)
    os.makedirs(transparent_dir, exist_ok=True)

    input_base = os.path.splitext(os.path.basename(input_image_path))[0]
    cutout_filename = f"{input_base}_cutout.png"

    # 1. Save Primary Cutout in multiple accessible locations
    # (a) Separately in the project root directory
    root_cutout_path = cutout_filename
    cutout_rgba.save(root_cutout_path, format="PNG")
    print(f"\n[Saved] Primary Cutout (Root Standalone)    -> {root_cutout_path}")

    # (b) Inside output_angles_transparent/
    folder_transp_cutout_path = os.path.join(transparent_dir, cutout_filename)
    cutout_rgba.save(folder_transp_cutout_path, format="PNG")
    # Also save as alias 'input_cutout.png' for standardized viewer pipelines
    cutout_rgba.save(os.path.join(transparent_dir, "input_cutout.png"), format="PNG")
    print(f"[Saved] Primary Cutout (In Transparent Dir) -> {folder_transp_cutout_path}")

    # (c) Inside output_angles/ for completeness
    folder_studio_cutout_path = os.path.join(studio_dir, cutout_filename)
    cutout_rgba.save(folder_studio_cutout_path, format="PNG")

    # 2. Save Composite Grids (Studio and Transparent)
    grid_image.save(output_grid_path)
    grid_image.save(os.path.join(studio_dir, "output_grid.png"))
    print(f"[Saved] Composite Grid (Studio)             -> {output_grid_path}")

    grid_transparent = stitch_transparent_grid(transparent_views)
    transp_grid_path = "output_grid_transparent.png"
    grid_transparent.save(transp_grid_path, format="PNG")
    grid_transparent.save(os.path.join(transparent_dir, "output_grid_transparent.png"), format="PNG")
    print(f"[Saved] Composite Grid (Transparent)        -> {transp_grid_path}")

    # 3. Save Studio Views
    for name, img in studio_views:
        path = os.path.join(studio_dir, f"{name}.png")
        img.save(path)

    # 4. Save Transparent Views & Compute Audit Metrics
    audit_data = []

    # Include Primary Cutout in the audit
    alpha_primary = np.array(cutout_rgba.split()[-1])
    transp_px_pri = int(np.count_nonzero(alpha_primary == 0))
    transp_pct_pri = (transp_px_pri / alpha_primary.size) * 100.0

    audit_data.append({
        "name": f"[Input Cutout] {cutout_filename}",
        "studio_path": root_cutout_path,
        "transp_path": folder_transp_cutout_path,
        "resolution": f"{cutout_rgba.width}x{cutout_rgba.height}",
        "mode": cutout_rgba.mode,
        "transp_pct": transp_pct_pri
    })

    # Include all 6 angle views
    for name, img in transparent_views:
        path = os.path.join(transparent_dir, f"{name}.png")
        img.save(path)

        # Audit transparency
        alpha_arr = np.array(img.split()[-1])
        transparent_px = int(np.count_nonzero(alpha_arr == 0))
        total_px = alpha_arr.size
        transp_pct = (transparent_px / total_px) * 100.0

        audit_data.append({
            "name": name,
            "studio_path": os.path.join(studio_dir, f"{name}.png"),
            "transp_path": path,
            "resolution": f"{img.width}x{img.height}",
            "mode": img.mode,
            "transp_pct": transp_pct
        })

    # 5. Print Side-by-Side Verification Table
    print("\n" + "=" * 98)
    print("MULTI-VIEW ASSET VERIFICATION AUDIT")
    print("=" * 98)
    print(f"{'Asset / Camera Angle':<32} | {'Resolution':<10} | {'Studio / Root':<20} | {'Transparent Folder':<22} | {'Transparency'}")
    print("-" * 98)
    for row in audit_data:
        studio_file = os.path.basename(row["studio_path"])
        transp_file = os.path.basename(row["transp_path"])
        pct_str = f"{row['transp_pct']:>5.1f}% [RGBA Clean]"
        print(f"{row['name']:<32} | {row['resolution']:<10} | {studio_file:<20} | {transp_file:<22} | {pct_str}")
    print("=" * 98)
    print("[OK] All cutout and multi-angle assets saved separately in root and in dedicated folders.")

    return audit_data


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------
def run_pipeline(
    input_image_path: str,
    output_grid_path: str = "output_grid.png",
    studio_angles_dir: str = "output_angles",
    transparent_angles_dir: str = "output_angles_transparent"
):
    total_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("UNIFIED IMAGE-TO-3D MULTI-VIEW GENERATION PIPELINE")
    print("=" * 60)
    print(f"Input file : {input_image_path}")
    print(f"Device     : {device.upper()}")
    print("=" * 60)

    # 1. Ingestion & Normalization
    t0 = time.perf_counter()
    standard_rgb = load_and_standardize_image(input_image_path)
    print(f"[Done] Input standardized: {standard_rgb.size[0]}x{standard_rgb.size[1]} ({time.perf_counter() - t0:.2f}s)")

    # 2. Initialize BiRefNet (Loaded once and kept in memory)
    t_model = time.perf_counter()
    birefnet_model = load_birefnet_model(device)
    print(f"[Done] BiRefNet loaded in {time.perf_counter() - t_model:.2f}s")

    # 3. Step 1: Primary Matting (In-Memory Cutout)
    t1 = time.perf_counter()
    print("--> [Step 1/4] Running primary background matting on input image...")
    cutout_rgba = run_birefnet_inference(
        image=standard_rgb,
        model=birefnet_model,
        device=device,
        apply_cleanup=True
    )
    print(f"[Done] Primary cutout created in {time.perf_counter() - t1:.2f}s")

    # 4. Step 2: Prepare Conditioning Canvas
    t2 = time.perf_counter()
    conditioned_img = prepare_conditioning_canvas(cutout_rgba)
    print(f"[Done] Conditioning canvas prepared ({time.perf_counter() - t2:.2f}s)")

    # 5. Step 3: Zero123++ Novel-View Synthesis
    # Offload BiRefNet to CPU if on CUDA to maximize VRAM headroom for diffusion
    if device == "cuda":
        birefnet_model.to("cpu")
        torch.cuda.empty_cache()

    t3 = time.perf_counter()
    print("--> [Step 2/4] Generating novel multi-angle views via Zero123++...")
    grid_image = synthesize_multiview(conditioned_img, device)
    print(f"[Done] Novel-view synthesis complete ({time.perf_counter() - t3:.2f}s)")

    # Re-engage BiRefNet on target device without disk reload
    if device == "cuda":
        birefnet_model.to(device)

    # 6. Slicing Composite Grid
    print("--> [Step 3/4] Slicing composite grid into 6 camera angles...")
    studio_views = slice_grid_views(grid_image)

    # 7. Step 4: Secondary Background Removal Pass on all 6 views
    t4 = time.perf_counter()
    transparent_views = secondary_background_removal(studio_views, birefnet_model, device)
    print(f"[Done] Secondary matting pass finished ({time.perf_counter() - t4:.2f}s)")

    # 8. Cleanup BiRefNet from memory
    del birefnet_model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    # 9. Export & Verification Audit
    t5 = time.perf_counter()
    export_and_audit(
        input_image_path=input_image_path,
        cutout_rgba=cutout_rgba,
        grid_image=grid_image,
        output_grid_path=output_grid_path,
        studio_views=studio_views,
        transparent_views=transparent_views,
        studio_dir=studio_angles_dir,
        transparent_dir=transparent_angles_dir
    )
    print(f"[Done] File export & audit completed in {time.perf_counter() - t5:.2f}s")

    total_elapsed = time.perf_counter() - total_start
    input_base = os.path.splitext(os.path.basename(input_image_path))[0]
    print("=" * 60)
    print(f"PIPELINE COMPLETED IN {total_elapsed:.2f} SECONDS")
    print(f"Primary Cutout (Separate Root): {os.path.abspath(f'{input_base}_cutout.png')}")
    print(f"Primary Cutout (In Folder)    : {os.path.abspath(os.path.join(transparent_angles_dir, f'{input_base}_cutout.png'))}")
    print(f"Composite Grid (Studio)       : {os.path.abspath(output_grid_path)}")
    print(f"Composite Grid (Transparent)  : {os.path.abspath('output_grid_transparent.png')}")
    print(f"Studio Views (6)              : {os.path.abspath(studio_angles_dir)}")
    print(f"Transparent Assets (6)        : {os.path.abspath(transparent_angles_dir)}")
    print("=" * 60)


def auto_detect_input_image() -> str:
    """Finds the first supported input image in current working directory."""
    for file in os.listdir("."):
        if (
            file.lower().endswith(SUPPORTED_EXTENSIONS)
            and not file.endswith("_cutout.png")
            and not file.startswith("output_")
            and not file.startswith("angles_")
        ):
            return file
    return ""


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_image = sys.argv[1]
    else:
        target_image = auto_detect_input_image()

    if not target_image or not os.path.exists(target_image):
        print("[Error] No valid input image specified or found.")
        print(f"Usage: python run_pipeline.py <path_to_image>")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    run_pipeline(target_image)
