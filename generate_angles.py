#!/usr/bin/env python3
"""
generate_angles.py - CLI utility for 3D Multi-Angle Canonical Camera Perspective Synthesis.
Uses Zero123++ diffusion model via angles_service to output 6 angles per product.
"""

import os
import sys
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from angles_service import generate_product_angles, OUTPUT_ANGLES_DIR

PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")


def resolve_cutout_path(arg: Optional[str] = None) -> tuple[str, str]:
    """
    Resolves a target argument to an absolute cutout path and base_name.
    """
    if arg:
        # Check if direct file path
        if os.path.exists(arg):
            base_name = os.path.splitext(os.path.basename(arg))[0].replace("_cutout", "")
            return os.path.abspath(arg), base_name
        
        # Check in processed_images
        candidate = os.path.join(PROCESSED_DIR, arg)
        if os.path.exists(candidate):
            base_name = os.path.splitext(arg)[0].replace("_cutout", "")
            return candidate, base_name
            
        candidate_cutout = os.path.join(PROCESSED_DIR, f"{arg}_cutout.png")
        if os.path.exists(candidate_cutout):
            return candidate_cutout, arg
            
        raise FileNotFoundError(f"Cutout not found for argument '{arg}'. Checked direct path and processed_images/.")

    # Auto-detect from processed_images
    if os.path.exists(PROCESSED_DIR):
        cutouts = [f for f in sorted(os.listdir(PROCESSED_DIR)) if f.endswith("_cutout.png")]
        if cutouts:
            selected = cutouts[0]
            base_name = selected.replace("_cutout.png", "")
            print(f"[Auto-Detect] Found {len(cutouts)} processed cutouts. Using first: '{base_name}'")
            return os.path.join(PROCESSED_DIR, selected), base_name

    raise FileNotFoundError("No cutouts found in processed_images/. Please remove backgrounds from products first.")


if __name__ == "__main__":
    target_arg = sys.argv[1] if len(sys.argv) > 1 else None
    
    try:
        cutout_path, base_name = resolve_cutout_path(target_arg)
        print("=" * 60)
        print("3D MULTI-ANGLE CAMERA RIG GENERATOR")
        print("=" * 60)
        print(f"Product Base Name : {base_name}")
        print(f"Input Cutout Path : {cutout_path}")
        print(f"Destination Folder: {os.path.join(OUTPUT_ANGLES_DIR, base_name)}")
        print("=" * 60)
        
        result = generate_product_angles(cutout_path=cutout_path, base_name=base_name)
        
        print("\n" + "=" * 60)
        print(f"[SUCCESS] 6 Perspectives Generated in {result['render_time_seconds']}s")
        print(f"Device Used       : {result['device'].upper()}")
        print(f"Angle Assets Dir  : {os.path.join(OUTPUT_ANGLES_DIR, base_name)}")
        print("Views:")
        for v in result['views']:
            print(f"  - {v['name']:<28} : {v['filename']} ({v['size_kb']} KB)")
        print("=" * 60)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)