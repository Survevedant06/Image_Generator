#!/usr/bin/env python3
"""
image_search.py - Search & Visual Similarity Engine for E-Commerce AI Studio
Provides:
  - Text & metadata multi-field query matching
  - Cryptographic (MD5) & Perceptual (dHash) image duplicate detection
  - Reverse visual image search by photo input
"""

import os
from typing import Dict, Any, List, Optional
import db


def check_duplicate(temp_image_path: str, max_hamming_distance: int = 6) -> Dict[str, Any]:
    """
    Checks if an uploaded image already exists in the system.
    Returns:
      {
        "exists": bool,
        "match": Optional[Dict],
        "message": str
      }
    """
    match = db.find_duplicate_or_similar_image(temp_image_path, max_hamming_distance=max_hamming_distance)
    if match:
        if match["match_type"] == "exact":
            msg = f"Exact duplicate of product '{match['product_name']}' (SKU: {match['product_sku']}) found."
        else:
            pct = int(match["similarity_score"] * 100)
            msg = f"Visually similar image ({pct}% match) found in product '{match['product_name']}'."
        return {
            "exists": True,
            "match": match,
            "message": msg
        }

    return {
        "exists": False,
        "match": None,
        "message": "No duplicates found. New unique product image."
    }


def reverse_image_search(query_image_path: str, top_k: int = 6, max_distance: int = 16) -> List[Dict[str, Any]]:
    """
    Performs visual similarity search across all stored catalog product images.
    Returns the top matching products ranked by visual similarity score.
    """
    if not os.path.exists(query_image_path):
        return []

    query_dhash = db.compute_dhash(query_image_path)
    if not query_dhash:
        return []

    conn = db.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT pi.*, p.id as product_id, p.name as product_name, p.sku as product_sku,
           p.raw_image, p.cutout_image, p.status as product_status,
           c.name as category_name
    FROM product_images pi
    JOIN products p ON pi.product_id = p.id
    LEFT JOIN categories c ON p.category_id = c.id
    WHERE pi.dhash IS NOT NULL AND pi.dhash != '' AND pi.image_type IN ('raw', 'cutout');
    """)
    candidates = cursor.fetchall()
    conn.close()

    results = []
    seen_products = set()

    for c_row in candidates:
        c = dict(c_row)
        prod_id = c["product_id"]
        dist = db.hamming_distance(query_dhash, c["dhash"])
        if dist <= max_distance:
            if prod_id in seen_products:
                continue
            seen_products.add(prod_id)
            sim_score = round(1.0 - (dist / 64.0), 3)
            results.append({
                "product_id": prod_id,
                "product_name": c["product_name"],
                "product_sku": c["product_sku"],
                "category_name": c["category_name"] or "General",
                "similarity_score": sim_score,
                "hamming_distance": dist,
                "thumbnail_url": c["file_url"],
                "raw_url": f"/inputs/{os.path.basename(c['raw_image'])}" if c.get("raw_image") else None,
                "cutout_url": f"/processed_images/{os.path.basename(c['cutout_image'])}" if c.get("cutout_image") else None,
                "status": c["product_status"]
            })

    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]
