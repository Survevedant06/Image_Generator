#!/usr/bin/env python3
"""
shadow_service.py - Professional Artificial Shadow Generation Engine
Provides:
  - Physics-grounded Contact Shadow (Ambient Occlusion at base contact points)
  - Directional Drop / Cast Shadow with 360-degree angle control, distance offset & blur
  - Ground projection foreshortening or floating perspective drop shadow
  - Alpha channel compositing with customizable opacity and background surfaces
"""

import os
import math
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")
GENERATED_DIR = os.path.join(BASE_DIR, "generated_scenes")
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)


def create_contact_shadow(
    alpha_mask: Image.Image,
    target_size: Tuple[int, int],
    product_pos: Tuple[int, int],
    product_size: Tuple[int, int],
    contact_strength: float = 0.85,
    contact_blur: float = 4.0,
    vertical_ratio: float = 0.08
) -> Image.Image:
    """
    Creates an Ambient Occlusion contact shadow beneath the lowest contact points of the subject.
    """
    cw, ch = target_size
    pw, ph = product_size
    px, py = product_pos

    contact_canvas = Image.new("L", (cw, ch), 0)

    # Base height: tight footprint from lower portion of object
    contact_h = max(6, int(ph * vertical_ratio))
    contact_w = max(6, int(pw * 0.94))

    # Scale alpha vertically into a tight elliptical ground contact patch
    contact_patch = alpha_mask.resize((contact_w, contact_h), resample=Image.Resampling.BILINEAR)
    
    # Gaussian blur for soft ambient occlusion falloff
    if contact_blur > 0:
        contact_patch = contact_patch.filter(ImageFilter.GaussianBlur(radius=contact_blur))

    # Place tightly at product base
    base_x = px + (pw - contact_w) // 2
    base_y = py + ph - int(contact_h * 0.6)

    contact_canvas.paste(contact_patch, (base_x, base_y), contact_patch)

    # Multiply by strength
    contact_np = np.array(contact_canvas, dtype=np.float32) * float(contact_strength)
    contact_np = np.clip(contact_np, 0, 255).astype(np.uint8)
    return Image.fromarray(contact_np)


def create_directional_drop_shadow(
    alpha_mask: Image.Image,
    target_size: Tuple[int, int],
    product_pos: Tuple[int, int],
    product_size: Tuple[int, int],
    light_angle_deg: float = -60.0,
    shadow_distance: float = 25.0,
    shadow_blur: float = 16.0,
    shadow_intensity: float = 0.55,
    mode: str = "ground"  # "ground" (perspective cast) or "floating" (straight offset drop shadow)
) -> Image.Image:
    """
    Creates a directional cast/drop shadow with 360-degree angle control and distance offset.
    - If mode == "ground": applies physical ground projection foreshortening
    - If mode == "floating": applies uniform offset drop shadow
    """
    cw, ch = target_size
    pw, ph = product_size
    px, py = product_pos

    rad = math.radians(light_angle_deg)
    # Direction vectors
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    cast_canvas = Image.new("L", (cw, ch), 0)

    if mode == "ground":
        # Realistic ground cast shadow:
        # dx driven by horizontal angle component
        # dy foreshortened by vertical angle component
        ground_foreshorten = max(0.15, min(0.65, abs(sin_a) * 0.5 + 0.15))
        shadow_h = max(10, int(ph * ground_foreshorten))
        shadow_w = int(pw * 1.04)

        cast_base = alpha_mask.resize((shadow_w, shadow_h), resample=Image.Resampling.BILINEAR)
        # Flip vertically to project on ground floor
        cast_base = ImageOps.flip(cast_base)

        # Calculate position
        offset_x = int(cos_a * shadow_distance * 1.8)
        offset_y = int(abs(sin_a) * shadow_distance * 0.6)

        cast_x = px + (pw - shadow_w) // 2 + offset_x
        cast_y = py + ph - int(shadow_h * 0.3) + offset_y
    else:
        # Floating drop shadow (e.g. wall or soft backdrop offset)
        shadow_w, shadow_h = pw, ph
        cast_base = alpha_mask.copy()
        offset_x = int(cos_a * shadow_distance)
        offset_y = int(sin_a * shadow_distance)
        cast_x = px + offset_x
        cast_y = py + offset_y

    cast_canvas.paste(cast_base, (cast_x, cast_y), cast_base)

    if shadow_blur > 0:
        cast_canvas = cast_canvas.filter(ImageFilter.GaussianBlur(radius=shadow_blur))

    cast_np = np.array(cast_canvas, dtype=np.float32) * float(shadow_intensity)
    cast_np = np.clip(cast_np, 0, 255).astype(np.uint8)
    return Image.fromarray(cast_np)


def synthesize_artificial_shadows(
    cutout_img: Image.Image,
    canvas_size: Optional[Tuple[int, int]] = None,
    position: Optional[Tuple[int, int]] = None,
    contact_shadow: bool = True,
    drop_shadow: bool = True,
    light_angle_deg: float = -60.0,
    shadow_distance: float = 25.0,
    shadow_blur: float = 16.0,
    contact_intensity: float = 0.85,
    drop_intensity: float = 0.55,
    shadow_color: Tuple[int, int, int] = (20, 25, 35),
    shadow_mode: str = "ground",
    background_type: str = "transparent",
    custom_bg_color: Optional[Tuple[int, int, int]] = None,
    auto_padding: bool = True
) -> Image.Image:
    """
    Synthesizes artificial contact and drop shadows for a product cutout.
    Returns:
      A composited Image (RGBA or RGB) with the realistic shadow and product.
    """
    if cutout_img.mode != "RGBA":
        cutout_img = cutout_img.convert("RGBA")

    # Crop to subject bounding box if auto_padding
    alpha = cutout_img.split()[3]
    bbox = alpha.getbbox()
    if bbox:
        cropped = cutout_img.crop(bbox)
    else:
        cropped = cutout_img

    pw, ph = cropped.size

    # Determine canvas size
    if canvas_size is None:
        if auto_padding:
            # Add generous padding for shadow spread and offset
            pad_x = max(80, int(pw * 0.25) + int(shadow_distance * 1.5))
            pad_y = max(80, int(ph * 0.25) + int(shadow_distance * 1.5))
            cw = pw + pad_x * 2
            ch = ph + pad_y * 2
            px = pad_x
            py = pad_y
        else:
            cw, ch = pw, ph
            px, py = 0, 0
    else:
        cw, ch = canvas_size
        if position is None:
            px = (cw - pw) // 2
            py = (ch - ph) // 2
        else:
            px, py = position

    alpha_cropped = cropped.split()[3]

    # 1. Contact Shadow Mask
    if contact_shadow:
        contact_mask = create_contact_shadow(
            alpha_cropped, (cw, ch), (px, py), (pw, ph),
            contact_strength=contact_intensity,
            contact_blur=max(2.0, shadow_blur * 0.25),
            vertical_ratio=0.08
        )
    else:
        contact_mask = Image.new("L", (cw, ch), 0)

    # 2. Directional Drop Shadow Mask
    if drop_shadow:
        drop_mask = create_directional_drop_shadow(
            alpha_cropped, (cw, ch), (px, py), (pw, ph),
            light_angle_deg=light_angle_deg,
            shadow_distance=shadow_distance,
            shadow_blur=shadow_blur,
            shadow_intensity=drop_intensity,
            mode=shadow_mode
        )
    else:
        drop_mask = Image.new("L", (cw, ch), 0)

    # 3. Combine Shadow Masks (maximum of both)
    combined_mask = np.maximum(np.array(contact_mask), np.array(drop_mask))
    combined_mask_img = Image.fromarray(combined_mask)

    # 4. Create Shadow Color Layer
    r, g, b = shadow_color
    shadow_layer = Image.new("RGBA", (cw, ch), (r, g, b, 0))
    shadow_layer.putalpha(combined_mask_img)

    # 5. Build Background
    if background_type == "transparent":
        canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    elif background_type == "white":
        canvas = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))
    elif background_type == "custom" and custom_bg_color:
        canvas = Image.new("RGBA", (cw, ch), (*custom_bg_color, 255))
    else:
        canvas = Image.new("RGBA", (cw, ch), (255, 255, 255, 255))

    # 6. Alpha Composite Shadow then Subject
    canvas.alpha_composite(shadow_layer)
    canvas.paste(cropped, (px, py), cropped)

    if background_type in ("white", "custom"):
        return canvas.convert("RGB")

    return canvas
