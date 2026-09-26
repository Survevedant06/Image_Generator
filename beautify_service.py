#!/usr/bin/env python3
"""
beautify_service.py - AI Beautification, Stylization & Multi-Format Exporter
Provides:
  - Automatic & manual photo beautification (clarity, contrast, vibrancy, sharpness)
  - Color grading and lighting enhancements for commercial product photos
  - Multi-format conversion (PNG, JPG, WebP, TIFF) with custom DPI
  - Standard e-commerce dimension scaling & aspect ratio padding
"""

import os
import io
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Standard Commercial E-Commerce Aspect Ratios & Dimensions
EXPORT_DIMENSIONS: Dict[str, Dict[str, Any]] = {
    "1:1": {
        "name": "1:1 Square (Amazon / Shopify)",
        "width": 2000,
        "height": 2000,
        "description": "Standard high-resolution e-commerce hero shot"
    },
    "4:5": {
        "name": "4:5 Portrait (Instagram Feed / Mobile)",
        "width": 1080,
        "height": 1350,
        "description": "Optimized for mobile vertical feed engagement"
    },
    "9:16": {
        "name": "9:16 Vertical (TikTok / Reels / Stories)",
        "width": 1080,
        "height": 1920,
        "description": "Full-screen vertical social commerce format"
    },
    "16:9": {
        "name": "16:9 Landscape (Web Banners / Headers)",
        "width": 1920,
        "height": 1080,
        "description": "Wide hero banners and desktop displays"
    }
}


def apply_beautification(
    image: Image.Image,
    auto_enhance: bool = True,
    brightness: float = 1.0,  # 0.5 - 1.5
    contrast: float = 1.0,    # 0.5 - 1.8
    saturation: float = 1.0,  # 0.5 - 2.0
    sharpness: float = 1.0,   # 0.0 - 3.0
    vibrance_boost: bool = True
) -> Image.Image:
    """
    Applies commercial photo beautification:
    - Auto white-point & shadow contrast stretching
    - Unsharp mask for high-frequency edge crispness
    - Color tone & saturation optimization
    - User fine-tuning sliders
    """
    has_alpha = (image.mode == "RGBA")
    if has_alpha:
        alpha = image.split()[3]
        rgb_img = image.convert("RGB")
    else:
        alpha = None
        rgb_img = image.convert("RGB")

    # 1. Automatic Dynamic Range & Tone Enhancement
    if auto_enhance:
        # Subtle auto-contrast (cut 0.5% extremes to avoid clipping)
        rgb_img = ImageOps.autocontrast(rgb_img, cutoff=0.5)

        # Commercial Unsharp Mask for micro-contrast & crisp edges
        rgb_img = rgb_img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=120, threshold=3))

    # 2. Brightness Adjustment
    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(rgb_img)
        rgb_img = enhancer.enhance(brightness)

    # 3. Contrast Adjustment
    if contrast != 1.0:
        enhancer = ImageEnhance.Contrast(rgb_img)
        rgb_img = enhancer.enhance(contrast)

    # 4. Color Saturation & Vibrance
    sat_factor = saturation
    if vibrance_boost:
        sat_factor *= 1.10  # gentle 10% pop for commercial appeal
    if sat_factor != 1.0:
        enhancer = ImageEnhance.Color(rgb_img)
        rgb_img = enhancer.enhance(sat_factor)

    # 5. User Sharpness
    if sharpness != 1.0:
        enhancer = ImageEnhance.Sharpness(rgb_img)
        rgb_img = enhancer.enhance(sharpness)

    # Recombine Alpha if present
    if has_alpha and alpha:
        rgb_img.putalpha(alpha)
        return rgb_img

    return rgb_img


def render_and_export_image(
    source_image_path: str,
    output_format: str = "PNG",       # "PNG", "JPG", "WEBP", "TIFF"
    aspect_ratio: str = "1:1",        # "1:1", "4:5", "9:16", "16:9", "original", or "custom"
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    dpi: int = 300,
    background_color: str = "transparent", # "transparent", "white", "#hex"
    auto_enhance: bool = True,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
    output_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Renders, beautifies, resizes, and exports an image to the requested format and dimensions.
    """
    if not os.path.exists(source_image_path):
        raise FileNotFoundError(f"Source image not found: {source_image_path}")

    img = Image.open(source_image_path)

    # Apply Beautification
    img = apply_beautification(
        img,
        auto_enhance=auto_enhance,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        sharpness=sharpness
    )

    # Determine Dimensions
    if aspect_ratio in EXPORT_DIMENSIONS:
        dims = EXPORT_DIMENSIONS[aspect_ratio]
        out_w = dims["width"]
        out_h = dims["height"]
    elif aspect_ratio == "custom" and target_width and target_height:
        out_w = max(100, min(8000, target_width))
        out_h = max(100, min(8000, target_height))
    else:
        out_w, out_h = img.size

    # Fit image onto canvas with desired background
    fmt = output_format.upper()
    if fmt in ["JPG", "JPEG"]:
        bg_mode = "RGB"
        default_bg = (255, 255, 255)
    else:
        bg_mode = "RGBA"
        default_bg = (0, 0, 0, 0) if background_color == "transparent" else (255, 255, 255, 255)

    if background_color == "white":
        fill_color = (255, 255, 255, 255) if bg_mode == "RGBA" else (255, 255, 255)
    elif background_color.startswith("#"):
        hex_c = background_color.lstrip("#")
        if len(hex_c) == 6:
            r, g, b = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            fill_color = (r, g, b, 255) if bg_mode == "RGBA" else (r, g, b)
        else:
            fill_color = default_bg
    else:
        fill_color = default_bg

    canvas = Image.new(bg_mode, (out_w, out_h), fill_color)

    # Preserve subject aspect ratio and center on canvas
    src_w, src_h = img.size
    scale = min((out_w * 0.85) / src_w, (out_h * 0.85) / src_h)
    new_w = int(src_w * scale)
    new_h = int(src_h * scale)
    resized_subject = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    offset_x = (out_w - new_w) // 2
    offset_y = (out_h - new_h) // 2

    if resized_subject.mode == "RGBA":
        canvas.paste(resized_subject, (offset_x, offset_y), mask=resized_subject.split()[3])
    else:
        canvas.paste(resized_subject, (offset_x, offset_y))

    # Normalize format for Pillow registry and file extension
    fmt_upper = (fmt or "PNG").upper()
    if fmt_upper in ["JPG", "JPEG"]:
        pil_format = "JPEG"
        ext = "jpg"
    elif fmt_upper == "WEBP":
        pil_format = "WEBP"
        ext = "webp"
    elif fmt_upper in ["TIF", "TIFF"]:
        pil_format = "TIFF"
        ext = "tiff"
    else:
        pil_format = "PNG"
        ext = "png"

    base_src = os.path.splitext(os.path.basename(source_image_path))[0]
    timestamp = int(time.time())

    if not output_filename:
        output_filename = f"{base_src}_{out_w}x{out_h}_{timestamp}.{ext}"

    dest_path = os.path.join(EXPORTS_DIR, output_filename)

    save_kwargs = {}
    if pil_format == "JPEG":
        if canvas.mode != "RGB":
            canvas = canvas.convert("RGB")
        save_kwargs["quality"] = 95
        save_kwargs["optimize"] = True
    elif pil_format == "WEBP":
        save_kwargs["quality"] = 95
        save_kwargs["method"] = 6
    elif pil_format == "PNG":
        save_kwargs["optimize"] = True
    elif pil_format == "TIFF":
        save_kwargs["compression"] = "tiff_deflate"

    # Set DPI metadata
    dpi_tuple = (dpi, dpi)
    canvas.save(dest_path, format=pil_format, dpi=dpi_tuple, **save_kwargs)

    file_size_kb = round(os.path.getsize(dest_path) / 1024, 1)

    return {
        "status": "success",
        "filename": output_filename,
        "file_url": f"/exports/{output_filename}",
        "file_path": dest_path,
        "format": fmt,
        "width": out_w,
        "height": out_h,
        "dpi": dpi,
        "size_kb": file_size_kb
    }
