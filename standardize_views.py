#!/usr/bin/env python3
"""
standardize_views.py - Standalone Multi-View Normalization & Centering Script

Standardizes a folder of transparent product angle images for catalog uniformity:
  1. Reads the alpha channel of each RGBA image to find the exact non-zero bounding box.
  2. Crops the subject cleanly to its bounding box.
  3. Resizes the subject proportionally onto a square canvas (default 1024x1024)
     such that its longest dimension occupies a fixed target fill percentage (default 80%,
     leaving a consistent 10% safety margin on all sides).
  4. Centers the subject using high-quality Lanczos resampling.
  5. Exports standardized transparent PNGs with identical filenames and displays
     a structured verification audit table.
"""

import os
import sys
import argparse
import time
from typing import List, Tuple, Optional, Dict, Any

from PIL import Image, ImageOps

DEFAULT_INPUT_DIR = "output_angles_transparent"
DEFAULT_OUTPUT_DIR = "output_angles_standardized"
DEFAULT_CANVAS_SIZE = 1024
DEFAULT_SCALE_RATIO = 0.80  # 80% of canvas (10% margin on all sides)


def standardize_single_image(
    image_path: str,
    output_path: str,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    scale_ratio: float = DEFAULT_SCALE_RATIO
) -> Optional[Dict[str, Any]]:
    """
    Crops the subject to its alpha bounding box, resizes proportionally,
    and centers it on a square transparent canvas.
    """
    raw_img = Image.open(image_path)
    raw_img = ImageOps.exif_transpose(raw_img)

    if raw_img.mode != "RGBA":
        raw_img = raw_img.convert("RGBA")

    orig_w, orig_h = raw_img.size

    # 1. Read alpha channel to calculate exact bounding box
    alpha = raw_img.split()[-1]
    bbox = alpha.getbbox()

    if bbox is None:
        print(f"  [Warning] No non-zero alpha pixels found in '{os.path.basename(image_path)}'. Skipping.")
        return None

    crop_left, crop_top, crop_right, crop_bottom = bbox
    crop_w = crop_right - crop_left
    crop_h = crop_bottom - crop_top

    # 2. Crop cleanly to bounding box
    cropped = raw_img.crop(bbox)

    # 3. Compute proportional scale preserving aspect ratio
    target_dim = int(round(canvas_size * scale_ratio))
    max_subject_dim = max(crop_w, crop_h)

    if max_subject_dim <= 0:
        return None

    scale_factor = target_dim / max_subject_dim
    new_w = max(1, int(round(crop_w * scale_factor)))
    new_h = max(1, int(round(crop_h * scale_factor)))

    # 4. Resize with high-quality Lanczos filter
    resized_subject = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # 5. Create uniform square transparent canvas and center subject
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    offset_x = (canvas_size - new_w) // 2
    offset_y = (canvas_size - new_h) // 2
    canvas.paste(resized_subject, (offset_x, offset_y), mask=resized_subject)

    # 6. Save output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, format="PNG")

    return {
        "filename": os.path.basename(image_path),
        "input_dim": f"{orig_w}x{orig_h}",
        "bbox": f"({crop_left},{crop_top},{crop_right},{crop_bottom})",
        "crop_dim": f"{crop_w}x{crop_h}",
        "scaled_dim": f"{new_w}x{new_h}",
        "output_dim": f"{canvas_size}x{canvas_size}",
        "fill_pct": f"{(max(new_w, new_h) / canvas_size) * 100:.1f}%",
        "output_path": output_path
    }


def process_directory(
    input_dir: str = DEFAULT_INPUT_DIR,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    scale_ratio: float = DEFAULT_SCALE_RATIO
) -> List[Dict[str, Any]]:
    """
    Standardizes all transparent PNG images in input_dir into output_dir.
    """
    if not os.path.isdir(input_dir):
        print(f"[Error] Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    image_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(".png")
    ])

    if not image_files:
        print(f"[Warning] No PNG images found in directory: {input_dir}")
        return []

    print("=" * 100)
    print("PRODUCT VIEW STANDARDIZATION & CENTERING")
    print("=" * 100)
    print(f"Input Directory  : {os.path.abspath(input_dir)}")
    print(f"Output Directory : {os.path.abspath(output_dir)}")
    print(f"Target Canvas    : {canvas_size}x{canvas_size} px (Square)")
    print(f"Target Subject Fill: {scale_ratio * 100:.0f}% (Safety Margin: {(1.0 - scale_ratio) / 2 * 100:.0f}%)")
    print(f"Images Found     : {len(image_files)}")
    print("=" * 100)

    start_time = time.perf_counter()
    audit_results: List[Dict[str, Any]] = []

    for i, filename in enumerate(image_files, 1):
        in_path = os.path.join(input_dir, filename)
        out_path = os.path.join(output_dir, filename)
        t0 = time.perf_counter()

        result = standardize_single_image(
            image_path=in_path,
            output_path=out_path,
            canvas_size=canvas_size,
            scale_ratio=scale_ratio
        )

        if result:
            result["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0
            audit_results.append(result)
            print(f"  [{i}/{len(image_files)}] Standardized '{filename}' -> {result['scaled_dim']} on {result['output_dim']} ({result['elapsed_ms']:.1f}ms)")

    total_elapsed = time.perf_counter() - start_time

    # Verification Audit Table
    print("\n" + "=" * 104)
    print("STANDARDIZATION VERIFICATION AUDIT")
    print("=" * 104)
    print(f"{'Filename':<28} | {'Input':<10} | {'Bounding Box':<22} | {'Cropped':<10} | {'Scaled':<10} | {'Canvas':<10} | {'Fill'}")
    print("-" * 104)
    for row in audit_results:
        print(
            f"{row['filename']:<28} | "
            f"{row['input_dim']:<10} | "
            f"{row['bbox']:<22} | "
            f"{row['crop_dim']:<10} | "
            f"{row['scaled_dim']:<10} | "
            f"{row['output_dim']:<10} | "
            f"{row['fill_pct']}"
        )
    print("=" * 104)
    print(f"[Done] Successfully standardized {len(audit_results)}/{len(image_files)} views in {total_elapsed:.2f}s.")
    print(f"Destination: {os.path.abspath(output_dir)}\n")

    return audit_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Standardize transparent product angle images for catalog uniformity."
    )
    parser.add_argument(
        "input_folder",
        nargs="?",
        default=DEFAULT_INPUT_DIR,
        help=f"Path to input folder containing transparent PNGs (default: {DEFAULT_INPUT_DIR})"
    )
    parser.add_argument(
        "output_folder",
        nargs="?",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Path to output folder for standardized PNGs (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--size", "-s",
        type=int,
        default=DEFAULT_CANVAS_SIZE,
        help=f"Output canvas square size in pixels (default: {DEFAULT_CANVAS_SIZE})"
    )
    parser.add_argument(
        "--scale", "-p",
        type=float,
        default=DEFAULT_SCALE_RATIO,
        help=f"Proportional fill ratio of the subject relative to canvas (default: {DEFAULT_SCALE_RATIO} = 80%%)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(
        input_dir=args.input_folder,
        output_dir=args.output_folder,
        canvas_size=args.size,
        scale_ratio=args.scale
    )
