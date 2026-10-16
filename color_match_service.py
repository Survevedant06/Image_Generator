#!/usr/bin/env python3
"""
color_match_service.py - Commercial Color Tone & Atmosphere Harmonization Engine
Auto-adjusts product white balance, lighting temperature, and color distribution
to realistically match the target studio scene or background environment.

Features:
  - Alpha-Masked Reinhard Statistical Color Transfer in CIE-LAB space
  - Illuminant White Balance / Color Temperature Chromatic Adaptation
  - Optical Ambient Rim Light Bleed (Fresnel edge light wrap from background)
  - Controllable match strength / blend factor (0.0 = preserve 100% original, 1.0 = full adaptation)
  - Luminance protection to retain product contrast, label legibility, and material texture
"""

import os
import math
from typing import Tuple, Optional, Dict, Any, Union
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_scenes")


def calculate_image_stats_lab(lab_array: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes mean and standard deviation for L, A, B channels, optionally masked.
    """
    if mask is not None and np.any(mask > 0):
        # Extract pixels where mask > 0
        valid_pixels = lab_array[mask > 20]
        if len(valid_pixels) == 0:
            valid_pixels = lab_array.reshape(-1, 3)
    else:
        valid_pixels = lab_array.reshape(-1, 3)

    mean = np.mean(valid_pixels, axis=0)
    std = np.std(valid_pixels, axis=0)
    # Prevent divide by zero
    std = np.where(std < 1e-4, 1e-4, std)
    return mean, std


def apply_rim_light_bleed(
    fg_rgba: Image.Image,
    bg_rgb: Image.Image,
    target_pos: Tuple[int, int],
    rim_width: int = 4,
    rim_strength: float = 0.35
) -> Image.Image:
    """
    Adds a subtle atmospheric Fresnel rim light bleed from the background onto
    the product's outer perimeter, making it organically seated in the scene lighting.
    """
    pw, ph = fg_rgba.size
    px, py = target_pos
    bw, bh = bg_rgb.size

    # Crop the corresponding background area
    crop_x1 = max(0, px)
    crop_y1 = max(0, py)
    crop_x2 = min(bw, px + pw)
    crop_y2 = min(bh, py + ph)

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return fg_rgba

    bg_crop = bg_rgb.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    # Resize to exact product size if offset
    bg_crop = bg_crop.resize((pw, ph), resample=Image.Resampling.BILINEAR)

    # Blur background to get diffuse ambient light color
    bg_ambient = bg_crop.filter(ImageFilter.GaussianBlur(radius=8))

    # Calculate rim mask: alpha eroded subtracted from alpha
    alpha = fg_rgba.split()[3]
    alpha_np = np.array(alpha)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rim_width * 2 + 1, rim_width * 2 + 1))
    eroded = cv2.erode(alpha_np, kernel)
    rim_mask_np = cv2.subtract(alpha_np, eroded)
    rim_mask = Image.fromarray((rim_mask_np * rim_strength).astype(np.uint8))

    # Composite ambient light onto product along rim
    rim_layer = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    rim_layer.paste(bg_ambient, (0, 0), rim_mask)

    res = fg_rgba.copy()
    res.alpha_composite(rim_layer)
    return res


def match_color_tone_to_scene(
    product_img: Image.Image,
    scene_img: Image.Image,
    target_pos: Optional[Tuple[int, int]] = None,
    blend_intensity: float = 0.70,
    preserve_luminance: bool = True,
    edge_light_bleed: bool = True
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Harmonizes product white balance and color temperature with the target scene atmosphere.

    Parameters:
      product_img: PIL Image of product (with alpha channel if cutout)
      scene_img: PIL Image of target background or studio scene
      target_pos: Position (x, y) where product is placed on the scene
      blend_intensity: 0.0 (keep original) to 1.0 (full atmospheric color adaptation)
      preserve_luminance: Whether to keep product's core lightness/contrast intact
      edge_light_bleed: Whether to apply subtle ambient lighting wrap around edges

    Returns:
      (adjusted_product_img, metadata_dict)
    """
    has_alpha = (product_img.mode in ("RGBA", "LA") or "transparency" in product_img.info)
    if has_alpha:
        rgba = product_img.convert("RGBA")
        r, g, b, alpha = rgba.split()
        alpha_np = np.array(alpha)
        fg_rgb = Image.merge("RGB", (r, g, b))
    else:
        rgba = product_img.convert("RGB")
        alpha_np = np.ones((rgba.size[1], rgba.size[0]), dtype=np.uint8) * 255
        alpha = None
        fg_rgb = rgba

    bg_rgb = scene_img.convert("RGB")
    pw, ph = fg_rgb.size
    bw, bh = bg_rgb.size

    if target_pos is None:
        target_pos = ((bw - pw) // 2, (bh - ph) // 2)

    # 1. Convert FG and BG to LAB
    fg_np = np.array(fg_rgb)
    bg_np = np.array(bg_rgb)

    fg_lab = cv2.cvtColor(fg_np, cv2.COLOR_RGB2LAB).astype(np.float32)
    bg_lab = cv2.cvtColor(bg_np, cv2.COLOR_RGB2LAB).astype(np.float32)

    # 2. Compute statistics
    fg_mean, fg_std = calculate_image_stats_lab(fg_lab, alpha_np)
    
    # For background, sample the region near the product or entire background
    px, py = target_pos
    crop_x1 = max(0, px - 50)
    crop_y1 = max(0, py - 50)
    crop_x2 = min(bw, px + pw + 50)
    crop_y2 = min(bh, py + ph + 50)
    
    if crop_x2 > crop_x1 and crop_y2 > crop_y1:
        bg_sample = bg_lab[crop_y1:crop_y2, crop_x1:crop_x2]
        bg_mean, bg_std = calculate_image_stats_lab(bg_sample)
    else:
        bg_mean, bg_std = calculate_image_stats_lab(bg_lab)

    # 3. Reinhard Color Transfer
    # Shift and scale each channel
    adjusted_lab = np.zeros_like(fg_lab)

    # For L channel (lightness):
    if preserve_luminance:
        # Keep product contrast, only subtle 20% exposure normalization
        scale_L = 1.0 + (bg_std[0] / fg_std[0] - 1.0) * (blend_intensity * 0.25)
        scale_L = max(0.85, min(1.15, scale_L))
        shift_L = (bg_mean[0] - fg_mean[0]) * (blend_intensity * 0.20)
        adjusted_lab[:, :, 0] = (fg_lab[:, :, 0] - fg_mean[0]) * scale_L + fg_mean[0] + shift_L
    else:
        scale_L = bg_std[0] / fg_std[0]
        adjusted_lab[:, :, 0] = (fg_lab[:, :, 0] - fg_mean[0]) * scale_L + bg_mean[0]

    # For A & B channels (green-red and blue-yellow chromaticity):
    for i in (1, 2):
        scale_chroma = bg_std[i] / fg_std[i]
        # Restrict extreme scaling to prevent oversaturating or discoloration
        scale_chroma = max(0.5, min(1.8, scale_chroma))
        target_channel = (fg_lab[:, :, i] - fg_mean[i]) * scale_chroma + bg_mean[i]
        # Blend with original
        adjusted_lab[:, :, i] = (1.0 - blend_intensity) * fg_lab[:, :, i] + blend_intensity * target_channel

    adjusted_lab = np.clip(adjusted_lab, 0, 255).astype(np.uint8)
    adjusted_rgb_np = cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)
    adjusted_pil = Image.fromarray(adjusted_rgb_np)

    if has_alpha and alpha is not None:
        adjusted_rgba = adjusted_pil.convert("RGBA")
        adjusted_rgba.putalpha(alpha)
    else:
        adjusted_rgba = adjusted_pil

    # 4. Optional Optical Rim Light Bleed
    if edge_light_bleed and has_alpha:
        adjusted_rgba = apply_rim_light_bleed(
            adjusted_rgba, bg_rgb, target_pos, rim_width=3, rim_strength=0.30 * blend_intensity
        )

    metadata = {
        "blend_intensity": blend_intensity,
        "fg_mean_lab": [round(float(x), 2) for x in fg_mean],
        "bg_mean_lab": [round(float(x), 2) for x in bg_mean],
        "preserve_luminance": preserve_luminance,
        "edge_light_bleed": edge_light_bleed
    }

    return adjusted_rgba, metadata
