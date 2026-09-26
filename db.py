#!/usr/bin/env python3
"""
db.py - SQLite Database Storage & Management for E-Commerce AI Studio
Provides relational persistence and CRUD operations for:
  - Categories (create, read, update, delete, counts)
  - Products (create, read, update, delete, status tracking)
  - Product Images & Visual Hashes (raw, cutouts, angles, stylized scenes, md5, dhash)
  - Automated migration / indexing of existing assets on startup
"""

import os
import sqlite3
import hashlib
import json
import time
import math
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "studio.db")
os.makedirs(DATA_DIR, exist_ok=True)


def get_db_connection() -> sqlite3.Connection:
    """Returns a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH, timeout=15.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    """Initializes the database schema if tables do not already exist."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Categories Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        slug TEXT UNIQUE NOT NULL,
        description TEXT DEFAULT '',
        icon TEXT DEFAULT 'tag',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # Products Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        sku TEXT UNIQUE,
        category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
        description TEXT DEFAULT '',
        tags TEXT DEFAULT '',
        raw_image TEXT,
        cutout_image TEXT,
        status TEXT DEFAULT 'raw', -- raw, cutout_ready, angles_ready, completed
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)

    # Product Images & Assets Table (tracks raw, cutouts, angles, scenes + hashes)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS product_images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id INTEGER NOT NULL REFERENCES products(id) ON DELETE CASCADE,
        image_type TEXT NOT NULL, -- raw, cutout, angle, scene
        file_path TEXT NOT NULL,
        file_url TEXT NOT NULL,
        file_size INTEGER DEFAULT 0,
        width INTEGER DEFAULT 0,
        height INTEGER DEFAULT 0,
        md5_hash TEXT,
        dhash TEXT,
        metadata TEXT DEFAULT '{}',
        created_at TEXT NOT NULL
    );
    """)

    # Create Indexes for fast searches and duplicate lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_product ON product_images(product_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_md5 ON product_images(md5_hash);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_dhash ON product_images(dhash);")

    # Insert default Category if empty
    cursor.execute("SELECT COUNT(*) as cnt FROM categories;")
    if cursor.fetchone()["cnt"] == 0:
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO categories (name, slug, description, icon, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?);
        """, ("General Catalog", "general-catalog", "Default catalog for e-commerce products", "package", now, now))

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Perceptual & Cryptographic Hash Utilities
# ---------------------------------------------------------------------------
def compute_md5(file_path: str) -> str:
    """Computes exact MD5 hash of a file."""
    if not os.path.exists(file_path):
        return ""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(img_or_path) -> str:
    """
    Computes a 64-bit Difference Hash (dHash) for fast visual similarity matching.
    Works directly with file paths or PIL Image instances.
    Crucially normalizes across image formats (PNG, JPG, WebP, BMP), transparent
    alpha channels (compositing onto white background), and EXIF orientations so
    that the same subject in different formats or encodings yields consistent hashes.
    """
    if not img_or_path:
        return ""
    try:
        if isinstance(img_or_path, str):
            if not os.path.exists(img_or_path):
                return ""
            img = Image.open(img_or_path)
        else:
            img = img_or_path

        # Normalize EXIF orientation
        img = ImageOps.exif_transpose(img)

        # Normalize transparency: composite RGBA / LA onto white background
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            alpha_channel = img.split()[-1]
            background.paste(img, mask=alpha_channel)
            img_gray = background.convert("L")
        elif img.mode != "L":
            img_gray = img.convert("L")
        else:
            img_gray = img

        # Resize to 9x8 (width=9, height=8) for horizontal gradient
        res = img_gray.resize((9, 8), Image.Resampling.BILINEAR)
        pixels = list(res.tobytes())

        # Compare adjacent pixels
        diff = []
        for row in range(8):
            row_start = row * 9
            for col in range(8):
                pixel_left = pixels[row_start + col]
                pixel_right = pixels[row_start + col + 1]
                diff.append(1 if pixel_left > pixel_right else 0)

        # Convert binary array to hex string (16 hex chars = 64 bits)
        decimal_val = 0
        for bit in diff:
            decimal_val = (decimal_val << 1) | bit
        return f"{decimal_val:016x}"
    except Exception as e:
        print(f"[dHash Error] {img_or_path}: {e}")
        return ""


def compute_phash(img_or_path, hash_size: int = 8, highfreq_factor: int = 4) -> str:
    """
    Computes a 64-bit Perceptual Hash (pHash) using discrete cosine transform (DCT).
    Highly robust across image formats (JPEG, PNG, WebP, BMP), compression artifacts,
    and resizing. Distinguishes different objects while matching identical items across formats.
    """
    if not img_or_path:
        return ""
    try:
        if isinstance(img_or_path, str):
            if not os.path.exists(img_or_path):
                return ""
            img = Image.open(img_or_path)
        else:
            img = img_or_path

        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            alpha_channel = img.split()[-1]
            background.paste(img, mask=alpha_channel)
            img = background
        img_gray = img.convert("L")

        N = hash_size * highfreq_factor  # 32
        img_res = img_gray.resize((N, N), Image.Resampling.BILINEAR)
        pixels = list(img_res.tobytes())
        matrix = [pixels[i * N:(i + 1) * N] for i in range(N)]

        # 2D DCT computation for top 8x8 low-frequency components
        dct = [[0.0] * hash_size for _ in range(hash_size)]
        for u in range(hash_size):
            for v in range(hash_size):
                sum_val = 0.0
                for x in range(N):
                    for y in range(N):
                        sum_val += matrix[x][y] * math.cos(((2 * x + 1) * u * math.pi) / (2.0 * N)) * math.cos(((2 * y + 1) * v * math.pi) / (2.0 * N))
                dct[u][v] = sum_val

        # Exclude DC coefficient (0,0) from median calculation
        vals = [dct[u][v] for u in range(hash_size) for v in range(hash_size) if not (u == 0 and v == 0)]
        median = sorted(vals)[len(vals) // 2]
        bits = [1 if dct[u][v] > median else 0 for u in range(hash_size) for v in range(hash_size)]

        val = 0
        for b in bits:
            val = (val << 1) | b
        return f"{val:016x}"
    except Exception as e:
        print(f"[pHash Error] {img_or_path}: {e}")
        return ""


def hamming_distance(hex1: str, hex2: str) -> int:
    """Computes bitwise Hamming distance between two 64-bit hex hashes."""
    if not hex1 or not hex2 or len(hex1) != 16 or len(hex2) != 16:
        return 64
    val1 = int(hex1, 16)
    val2 = int(hex2, 16)
    xor_val = val1 ^ val2
    return bin(xor_val).count("1")


def images_are_visually_identical(img_path1: str, img_path2: str) -> bool:
    """
    Advanced multi-hash detector that determines if two images represent the same visual product,
    even across different file formats (PNG vs JPG vs WebP), compression, or resolution.
    Returns True if dHash <= 4 or pHash <= 6.
    """
    if not img_path1 or not img_path2:
        return False
    if not os.path.exists(img_path1) or not os.path.exists(img_path2):
        return False

    # Check dHash
    d1 = compute_dhash(img_path1)
    d2 = compute_dhash(img_path2)
    if d1 and d2:
        if hamming_distance(d1, d2) <= 4:
            return True

    # Check pHash
    p1 = compute_phash(img_path1)
    p2 = compute_phash(img_path2)
    if p1 and p2:
        if hamming_distance(p1, p2) <= 6:
            return True

    return False


# ---------------------------------------------------------------------------
# Category CRUD Operations
# ---------------------------------------------------------------------------
def list_categories() -> List[Dict[str, Any]]:
    """Lists all categories along with their current product count."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT c.id, c.name, c.slug, c.description, c.icon, c.created_at, c.updated_at,
           COUNT(p.id) as product_count
    FROM categories c
    LEFT JOIN products p ON c.id = p.category_id
    GROUP BY c.id
    ORDER BY c.name ASC;
    """)
    rows = cursor.fetchall()
    categories = [dict(row) for row in rows]
    conn.close()
    return categories


def get_category(category_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a single category by ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM categories WHERE id = ?;", (category_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def create_category(name: str, description: str = "", icon: str = "tag") -> Dict[str, Any]:
    """Creates a new category."""
    name = name.strip()
    slug = name.lower().replace(" ", "-").replace("/", "-")
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO categories (name, slug, description, icon, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (name, slug, description.strip(), icon.strip(), now, now))
    cat_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_category(cat_id)


def update_category(cat_id: int = None, name: str = "", icon: Optional[str] = None, description: Optional[str] = None, category_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Updates an existing category metadata."""
    cid = cat_id if cat_id is not None else category_id
    if cid is None:
        raise ValueError("Category ID must be provided.")

    current = get_category(cid)
    if not current:
        return None

    new_name = name.strip() if name else current["name"]
    slug = new_name.lower().replace(" ", "-").replace("/", "-")
    new_icon = icon.strip() if icon is not None else current.get("icon", "tag")
    new_desc = description.strip() if description is not None else current.get("description", "")
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE categories
    SET name = ?, slug = ?, description = ?, icon = ?, updated_at = ?
    WHERE id = ?;
    """, (new_name, slug, new_desc, new_icon, now, cid))
    conn.commit()
    conn.close()
    return get_category(cid)


def delete_category(cat_id: int = None, category_id: Optional[int] = None, reassign_to_id: Optional[int] = None) -> int:
    """
    Safely deletes a category. Ensures child products are reassigned to
    'General Catalog' (or first available category, or NULL).
    Returns the count of reassigned products.
    """
    cid = cat_id if cat_id is not None else category_id
    if cid is None:
        return 0

    conn = get_db_connection()
    cursor = conn.cursor()

    # Count how many products are in this category
    cursor.execute("SELECT COUNT(*) as cnt FROM products WHERE category_id = ?;", (cid,))
    row = cursor.fetchone()
    reassigned_count = row["cnt"] if row else 0

    if reassign_to_id:
        fallback_id = reassign_to_id
    else:
        # Look for a 'General' or 'General Catalog' category that isn't the one being deleted
        cursor.execute("SELECT id FROM categories WHERE (LOWER(name) LIKE '%general%' OR LOWER(slug) LIKE '%general%') AND id != ? LIMIT 1;", (cid,))
        gen_cat = cursor.fetchone()
        if gen_cat:
            fallback_id = gen_cat["id"]
        else:
            # Fallback to the first available category, or None (NULL)
            cursor.execute("SELECT id FROM categories WHERE id != ? ORDER BY id ASC LIMIT 1;", (cid,))
            default_cat = cursor.fetchone()
            fallback_id = default_cat["id"] if default_cat else None

    if reassigned_count > 0:
        cursor.execute("UPDATE products SET category_id = ? WHERE category_id = ?;", (fallback_id, cid))

    cursor.execute("DELETE FROM categories WHERE id = ?;", (cid,))
    conn.commit()
    conn.close()
    return reassigned_count


# ---------------------------------------------------------------------------
# Product CRUD Operations
# ---------------------------------------------------------------------------
def generate_unique_sku(cursor: sqlite3.Cursor, base_text: str) -> str:
    """Generates a guaranteed-unique SKU for the database."""
    clean = "".join(c for c in base_text if c.isalnum()).upper()[:8]
    if not clean:
        clean = "PROD"
    candidate = f"SKU-{clean}"
    cursor.execute("SELECT id FROM products WHERE sku = ?;", (candidate,))
    if not cursor.fetchone():
        return candidate
    counter = 1
    while True:
        candidate = f"SKU-{clean}-{counter:03d}"
        cursor.execute("SELECT id FROM products WHERE sku = ?;", (candidate,))
        if not cursor.fetchone():
            return candidate
        counter += 1


def create_product(
    name: str,
    sku: Optional[str] = None,
    category_id: Optional[int] = None,
    description: str = "",
    tags: str = "",
    raw_image: Optional[str] = None,
    cutout_image: Optional[str] = None,
    status: str = "raw"
) -> Dict[str, Any]:
    """Creates a new product entry in the database."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()

    if not sku:
        sku = generate_unique_sku(cursor, name)

    # Deduplication Guard: Check if a product with this raw_image already exists
    if raw_image:
        norm_raw = os.path.normpath(os.path.abspath(raw_image)).lower()
        cursor.execute("SELECT id, raw_image, cutout_image FROM products WHERE raw_image IS NOT NULL;")
        for row in cursor.fetchall():
            if row[1] and os.path.normpath(os.path.abspath(row[1])).lower() == norm_raw:
                existing_id = row[0]
                if cutout_image and not row[2] and os.path.exists(cutout_image):
                    cursor.execute(
                        "UPDATE products SET cutout_image = ?, status = 'cutout_ready', updated_at = ? WHERE id = ?",
                        (cutout_image, now, existing_id)
                    )
                    _add_product_image(cursor, existing_id, "cutout", cutout_image)
                    conn.commit()
                conn.close()
                return get_product(existing_id)

    cursor.execute("""
    INSERT INTO products (name, sku, category_id, description, tags, raw_image, cutout_image, status, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (name.strip(), sku.strip(), category_id, description.strip(), tags.strip(), raw_image, cutout_image, status, now, now))
    prod_id = cursor.lastrowid

    # If raw image provided, index it into product_images
    if raw_image and os.path.exists(raw_image):
        _add_product_image(cursor, prod_id, "raw", raw_image)

    if cutout_image and os.path.exists(cutout_image):
        _add_product_image(cursor, prod_id, "cutout", cutout_image)

    conn.commit()
    conn.close()
    return get_product(prod_id)


def _add_product_image(cursor: sqlite3.Cursor, product_id: int, image_type: str, file_path: str, metadata: dict = None):
    """Internal helper to insert a product image with computed hashes."""
    if not os.path.exists(file_path):
        return
    file_size = os.path.getsize(file_path)
    width, height = 0, 0
    try:
        with Image.open(file_path) as im:
            width, height = im.size
    except Exception:
        pass

    md5_h = compute_md5(file_path)
    d_h = compute_dhash(file_path)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    meta_json = json.dumps(metadata or {})

    # Compute URL path relative to workspace
    filename = os.path.basename(file_path)
    if "processed_images" in file_path:
        file_url = f"/processed_images/{filename}"
    elif "generated_scenes" in file_path:
        file_url = f"/generated_scenes/{filename}"
    elif "output_angles" in file_path:
        file_url = f"/output_angles/{filename}"
    else:
        file_url = f"/inputs/{filename}"

    cursor.execute("""
    INSERT INTO product_images (product_id, image_type, file_path, file_url, file_size, width, height, md5_hash, dhash, metadata, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (product_id, image_type, file_path, file_url, file_size, width, height, md5_h, d_h, meta_json, now))


def get_product(product_id: int) -> Optional[Dict[str, Any]]:
    """Fetches a product by ID along with its category name and associated images."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT p.*, c.name as category_name, c.icon as category_icon
    FROM products p
    LEFT JOIN categories c ON p.category_id = c.id
    WHERE p.id = ?;
    """, (product_id,))
    row = cursor.fetchone()
    if not row:
        conn.close()
        return None

    product = dict(row)

    # Fetch associated images
    cursor.execute("""
    SELECT * FROM product_images
    WHERE product_id = ?
    ORDER BY id ASC;
    """, (product_id,))
    img_rows = cursor.fetchall()
    product["images"] = [dict(img) for img in img_rows]

    # Quick access helpers
    product["raw_url"] = f"/inputs/{os.path.basename(product['raw_image'])}" if product.get("raw_image") else None
    product["cutout_url"] = f"/processed_images/{os.path.basename(product['cutout_image'])}" if product.get("cutout_image") else None

    conn.close()
    return product


def list_products(
    category_id: Optional[int] = None,
    search: Optional[str] = None,
    status: Optional[str] = None,
    sort_by: str = "newest",
    limit: int = 100,
    offset: int = 0
) -> Dict[str, Any]:
    """Lists products with optional filtering by category, text search, or status."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
    SELECT p.*, c.name as category_name, c.icon as category_icon,
           (SELECT COUNT(*) FROM product_images pi WHERE pi.product_id = p.id AND pi.image_type = 'angle') as angle_count,
           (SELECT COUNT(*) FROM product_images pi WHERE pi.product_id = p.id AND pi.image_type = 'scene') as scene_count
    FROM products p
    LEFT JOIN categories c ON p.category_id = c.id
    WHERE 1=1
    """
    params = []

    if category_id is not None:
        query += " AND p.category_id = ?"
        params.append(category_id)

    if status:
        query += " AND p.status = ?"
        params.append(status)

    if search:
        term = f"%{search.strip()}%"
        query += " AND (p.name LIKE ? OR p.sku LIKE ? OR p.tags LIKE ? OR p.description LIKE ? OR c.name LIKE ?)"
        params.extend([term, term, term, term, term])

    if sort_by == "oldest":
        query += " ORDER BY p.id ASC"
    elif sort_by == "name_asc":
        query += " ORDER BY p.name ASC"
    elif sort_by == "name_desc":
        query += " ORDER BY p.name DESC"
    else:
        query += " ORDER BY p.id DESC"

    # Count total matching
    count_query = f"SELECT COUNT(*) as total FROM ({query})"
    cursor.execute(count_query, params)
    total = cursor.fetchone()["total"]

    # Pagination
    query += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    cursor.execute(query, params)
    rows = cursor.fetchall()
    products = []
    for r in rows:
        p = dict(r)
        p["raw_url"] = f"/inputs/{os.path.basename(p['raw_image'])}" if p.get("raw_image") else None
        p["cutout_url"] = f"/processed_images/{os.path.basename(p['cutout_image'])}" if p.get("cutout_image") else None
        products.append(p)

    conn.close()
    return {
        "products": products,
        "total": total,
        "limit": limit,
        "offset": offset
    }


def update_product(
    product_id: int,
    name: Optional[str] = None,
    sku: Optional[str] = None,
    category_id: Optional[int] = None,
    description: Optional[str] = None,
    tags: Optional[str] = None,
    status: Optional[str] = None,
    raw_image: Optional[str] = None,
    cutout_image: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Updates product attributes and status."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()

    updates = ["updated_at = ?"]
    params = [now]

    if name is not None:
        updates.append("name = ?")
        params.append(name.strip())
    if sku is not None:
        updates.append("sku = ?")
        params.append(sku.strip())
    if category_id is not None:
        updates.append("category_id = ?")
        params.append(category_id)
    if description is not None:
        updates.append("description = ?")
        params.append(description.strip())
    if tags is not None:
        updates.append("tags = ?")
        params.append(tags.strip())
    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if raw_image is not None:
        updates.append("raw_image = ?")
        params.append(raw_image)
        _add_product_image(cursor, product_id, "raw", raw_image)
    if cutout_image is not None:
        updates.append("cutout_image = ?")
        params.append(cutout_image)
        _add_product_image(cursor, product_id, "cutout", cutout_image)

    params.append(product_id)
    cursor.execute(f"UPDATE products SET {', '.join(updates)} WHERE id = ?;", params)
    conn.commit()
    conn.close()
    return get_product(product_id)


def record_product_asset(product_id: int, image_type: str, file_path: str, metadata: dict = None):
    """Records an angle view or generated scene asset for a product."""
    conn = get_db_connection()
    cursor = conn.cursor()
    _add_product_image(cursor, product_id, image_type, file_path, metadata)

    # Update product status if relevant
    if image_type == "cutout":
        cursor.execute("UPDATE products SET cutout_image = ?, status = 'cutout_ready' WHERE id = ?;", (file_path, product_id))
    elif image_type == "angle":
        cursor.execute("UPDATE products SET status = 'angles_ready' WHERE id = ? AND status != 'completed';", (product_id,))
    elif image_type == "scene":
        cursor.execute("UPDATE products SET status = 'completed' WHERE id = ?;", (product_id,))

    conn.commit()
    conn.close()


def delete_product(product_id: int, remove_files: bool = True) -> bool:
    """
    Deletes the product from SQLite DB, deletes child records from product_images,
    and removes associated files from disk (inputs/, processed_images/, generated_scenes/, output_angles/).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Fetch product info
    cursor.execute("SELECT name, raw_image, cutout_image FROM products WHERE id = ?;", (product_id,))
    prod_row = cursor.fetchone()
    if not prod_row:
        conn.close()
        return False

    files_to_remove = set()

    # 2. Add product images from DB
    cursor.execute("SELECT file_path FROM product_images WHERE product_id = ?;", (product_id,))
    for row in cursor.fetchall():
        if row["file_path"]:
            files_to_remove.add(row["file_path"])

    if prod_row["raw_image"]:
        files_to_remove.add(prod_row["raw_image"])
    if prod_row["cutout_image"]:
        files_to_remove.add(prod_row["cutout_image"])

    # 3. Clean up potential angle / scene files matching base_name
    prod_name = prod_row["name"] or ""
    raw_path = prod_row["raw_image"] or ""
    base_candidates = set()
    if raw_path:
        base_candidates.add(os.path.splitext(os.path.basename(raw_path))[0].lower())
    if prod_name:
        clean_name = "".join(c if c.isalnum() else "_" for c in prod_name.lower()).strip("_")
        base_candidates.add(clean_name)

    angles_dir = os.path.join(BASE_DIR, "output_angles")
    scenes_dir = os.path.join(BASE_DIR, "generated_scenes")

    if remove_files:
        for b_name in base_candidates:
            if not b_name:
                continue
            if os.path.exists(angles_dir):
                for f in os.listdir(angles_dir):
                    if f.lower().startswith(b_name):
                        files_to_remove.add(os.path.join(angles_dir, f))
            if os.path.exists(scenes_dir):
                for f in os.listdir(scenes_dir):
                    if f.lower().startswith(b_name):
                        files_to_remove.add(os.path.join(scenes_dir, f))

        # Safely delete collected files only if not shared by any other product
        for p in files_to_remove:
            if p and os.path.exists(p):
                cursor.execute(
                    "SELECT COUNT(*) as cnt FROM products WHERE id != ? AND (LOWER(raw_image) = LOWER(?) OR LOWER(cutout_image) = LOWER(?));",
                    (product_id, p, p)
                )
                shared = cursor.fetchone()
                if shared and shared["cnt"] > 0:
                    continue  # File is still in use by another product, do not delete from disk

                cursor.execute(
                    "SELECT COUNT(*) as cnt FROM product_images WHERE product_id != ? AND LOWER(file_path) = LOWER(?);",
                    (product_id, p)
                )
                shared_img = cursor.fetchone()
                if shared_img and shared_img["cnt"] > 0:
                    continue  # File is still in use by another product image

                try:
                    os.remove(p)
                except Exception as e:
                    print(f"[Delete Product File Error] {p}: {e}")

    # 4. Cascade delete DB records
    cursor.execute("DELETE FROM product_images WHERE product_id = ?;", (product_id,))
    cursor.execute("DELETE FROM products WHERE id = ?;", (product_id,))
    conn.commit()
    conn.close()
    return True


# ---------------------------------------------------------------------------
# Visual Similarity & Duplicate Checking
# ---------------------------------------------------------------------------
def find_duplicate_or_similar_image(file_path: str, max_hamming_distance: int = 6) -> Optional[Dict[str, Any]]:
    """
    Checks whether the given image matches any existing product image via:
      1. Exact file path (case-normalized)
      2. Exact MD5 hash
      3. Dual-layer perceptual hashing (dHash + pHash) invariant to formats (PNG, JPG, WebP, BMP)
    Returns details of the matching product and image if found, else None.
    """
    if not os.path.exists(file_path):
        return None

    target_norm = os.path.normcase(os.path.normpath(file_path))
    target_md5 = compute_md5(file_path)
    target_dhash = compute_dhash(file_path)
    target_phash = compute_phash(file_path)

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Exact File Path or Exact MD5 Match
    cursor.execute("""
    SELECT pi.*, p.id as product_id, p.name as product_name, p.sku as product_sku,
           p.raw_image, p.cutout_image, p.status as product_status,
           c.name as category_name
    FROM product_images pi
    JOIN products p ON pi.product_id = p.id
    LEFT JOIN categories c ON p.category_id = c.id
    WHERE pi.image_type IN ('raw', 'cutout');
    """)
    all_images = cursor.fetchall()

    for row in all_images:
        row_dict = dict(row)
        cand_path = row_dict.get("file_path") or ""
        cand_norm = os.path.normcase(os.path.normpath(cand_path)) if cand_path else ""

        # Exact path check
        if cand_norm and cand_norm == target_norm:
            conn.close()
            return {
                "match_type": "exact_file",
                "similarity_score": 1.0,
                "hamming_distance": 0,
                "product_id": row_dict["product_id"],
                "product_name": row_dict["product_name"],
                "product_sku": row_dict["product_sku"],
                "category_name": row_dict["category_name"] or "Uncategorized",
                "matched_image_type": row_dict["image_type"],
                "matched_image_url": row_dict["file_url"],
                "product_raw_url": f"/inputs/{os.path.basename(row_dict['raw_image'])}" if row_dict.get("raw_image") else None,
                "product_cutout_url": f"/processed_images/{os.path.basename(row_dict['cutout_image'])}" if row_dict.get("cutout_image") else None,
                "product_status": row_dict["product_status"]
            }

        # Exact MD5 check
        if target_md5 and row_dict.get("md5_hash") == target_md5:
            conn.close()
            return {
                "match_type": "exact_md5",
                "similarity_score": 1.0,
                "hamming_distance": 0,
                "product_id": row_dict["product_id"],
                "product_name": row_dict["product_name"],
                "product_sku": row_dict["product_sku"],
                "category_name": row_dict["category_name"] or "Uncategorized",
                "matched_image_type": row_dict["image_type"],
                "matched_image_url": row_dict["file_url"],
                "product_raw_url": f"/inputs/{os.path.basename(row_dict['raw_image'])}" if row_dict.get("raw_image") else None,
                "product_cutout_url": f"/processed_images/{os.path.basename(row_dict['cutout_image'])}" if row_dict.get("cutout_image") else None,
                "product_status": row_dict["product_status"]
            }

    # 2. Perceptual Similarity Match via dHash & pHash
    if target_dhash or target_phash:
        best_match = None
        lowest_dist = 999

        for row in all_images:
            cand = dict(row)
            cand_dhash = cand.get("dhash") or ""
            cand_path = cand.get("file_path") or ""

            is_match = False
            match_dist = 64

            # Cutout images: require BOTH tight dHash and pHash to prevent silhouette collisions
            if cand["image_type"] == "cutout":
                if target_dhash and cand_dhash:
                    d_dist = hamming_distance(target_dhash, cand_dhash)
                    if d_dist <= 4:
                        if target_phash and cand_path and os.path.exists(cand_path):
                            cand_phash = compute_phash(cand_path)
                            if cand_phash:
                                p_dist = hamming_distance(target_phash, cand_phash)
                                if p_dist <= 6:
                                    is_match = True
                                    match_dist = min(d_dist, p_dist)
                        elif d_dist <= 2:
                            is_match = True
                            match_dist = d_dist
            else:
                # Raw images: natural backgrounds
                if target_dhash and cand_dhash:
                    d_dist = hamming_distance(target_dhash, cand_dhash)
                    if d_dist <= max_hamming_distance:
                        is_match = True
                        match_dist = d_dist
                if not is_match and target_phash and cand_path and os.path.exists(cand_path):
                    cand_phash = compute_phash(cand_path)
                    if cand_phash:
                        p_dist = hamming_distance(target_phash, cand_phash)
                        if p_dist <= max_hamming_distance:
                            is_match = True
                            match_dist = min(match_dist, p_dist)

            if is_match and match_dist < lowest_dist:
                lowest_dist = match_dist
                best_match = cand

        if best_match:
            conn.close()
            similarity = round(1.0 - (lowest_dist / 64.0), 3)
            return {
                "match_type": "visual_similarity",
                "similarity_score": similarity,
                "hamming_distance": lowest_dist,
                "product_id": best_match["product_id"],
                "product_name": best_match["product_name"],
                "product_sku": best_match["product_sku"],
                "category_name": best_match["category_name"] or "Uncategorized",
                "matched_image_type": best_match["image_type"],
                "matched_image_url": best_match["file_url"],
                "product_raw_url": f"/inputs/{os.path.basename(best_match['raw_image'])}" if best_match.get("raw_image") else None,
                "product_cutout_url": f"/processed_images/{os.path.basename(best_match['cutout_image'])}" if best_match.get("cutout_image") else None,
                "product_status": best_match["product_status"]
            }

    conn.close()
    return None


def search_products_by_dhash(target_dhash: str, max_distance: int = 10) -> List[Dict[str, Any]]:
    """
    Compares a 64-bit dHash against stored product hashes using Hamming distance,
    returning matching products sorted by similarity score (1 - Hamming Distance / 64).
    """
    if not target_dhash or len(target_dhash) != 16:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT pi.id as image_id, pi.product_id, pi.file_url, pi.image_type, pi.dhash,
           p.id, p.name as product_name, p.sku as product_sku, p.status as product_status,
           p.raw_image, p.cutout_image, p.tags as product_tags, p.category_id,
           c.name as category_name
    FROM product_images pi
    JOIN products p ON pi.product_id = p.id
    LEFT JOIN categories c ON p.category_id = c.id
    WHERE pi.dhash IS NOT NULL AND pi.dhash != '';
    """)
    candidates = cursor.fetchall()
    conn.close()

    matched_dict = {}
    for c_row in candidates:
        r = dict(c_row)
        cand_dhash = r["dhash"]
        dist = hamming_distance(target_dhash, cand_dhash)
        if dist <= max_distance:
            pid = r["product_id"]
            score = round(1.0 - (dist / 64.0), 3)
            pct = round((1.0 - (dist / 64.0)) * 100, 1)

            # Build image URLs
            raw_url = f"/inputs/{os.path.basename(r['raw_image'])}" if r.get("raw_image") else None
            cutout_url = f"/processed_images/{os.path.basename(r['cutout_image'])}" if r.get("cutout_image") else None
            thumb_url = cutout_url or raw_url or r["file_url"]

            # Retain the highest scoring match for each product
            if pid not in matched_dict or score > matched_dict[pid]["similarity_score"]:
                matched_dict[pid] = {
                    "product_id": pid,
                    "id": pid,
                    "name": r["product_name"],
                    "product_name": r["product_name"],
                    "sku": r["product_sku"] or "NO-SKU",
                    "product_sku": r["product_sku"] or "NO-SKU",
                    "category_id": r["category_id"],
                    "category_name": r["category_name"] or "General",
                    "similarity_score": score,
                    "similarity_percentage": pct,
                    "hamming_distance": dist,
                    "matched_image_url": r["file_url"],
                    "matched_image_type": r["image_type"],
                    "thumbnail_url": thumb_url,
                    "raw_url": raw_url,
                    "cutout_url": cutout_url,
                    "status": r["product_status"],
                    "tags": r["product_tags"] or ""
                }

    matches = list(matched_dict.values())
    matches.sort(key=lambda x: x["similarity_score"], reverse=True)
    return matches


# ---------------------------------------------------------------------------
# Startup Migration & Automated Asset Indexing
# ---------------------------------------------------------------------------
def sync_existing_files_into_db():
    """
    Inspects inputs/, processed_images/, generated_scenes/, output_angles/
    and creates database entries for any unindexed files so that existing user
    assets appear seamlessly in the catalog.
    """
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get default category
    cursor.execute("SELECT id FROM categories ORDER BY id ASC LIMIT 1;")
    default_cat = cursor.fetchone()
    default_cat_id = default_cat["id"] if default_cat else 1

    inputs_dir = os.path.join(BASE_DIR, "inputs")
    processed_dir = os.path.join(BASE_DIR, "processed_images")

    if not os.path.exists(inputs_dir):
        conn.close()
        return

    # Check all images in inputs/
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".heic"}
    for filename in sorted(os.listdir(inputs_dir)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in valid_exts:
            continue

        raw_path = os.path.join(inputs_dir, filename)
        base_name = os.path.splitext(filename)[0]
        title = base_name.replace("_", " ").replace("-", " ").title()

        # Check if already indexed as a product by path, filename, or name
        cursor.execute(
            "SELECT id FROM products WHERE raw_image = ? OR raw_image LIKE ? OR name = ? OR name = ?;",
            (raw_path, f"%{filename}", base_name, title)
        )
        existing_p = cursor.fetchone()

        cutout_name = f"{base_name}_cutout.png"
        cutout_path = os.path.join(processed_dir, cutout_name)
        has_cutout = os.path.exists(cutout_path)

        status = "cutout_ready" if has_cutout else "raw"

        if not existing_p:
            sku = generate_unique_sku(cursor, base_name)
            cat_id = default_cat_id
            try:
                import category_classifier
                c_id, _, _ = category_classifier.resolve_product_category(title, raw_image_path=raw_path, filename=filename)
                cat_id = c_id
            except Exception as ce:
                print(f"[Auto-Categorize Sync Notice] {ce}")

            cursor.execute("""
            INSERT INTO products (name, sku, category_id, description, tags, raw_image, cutout_image, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                title,
                sku,
                cat_id,
                f"Imported from {filename}",
                "imported",
                raw_path,
                cutout_path if has_cutout else None,
                status,
                time.strftime("%Y-%m-%d %H:%M:%S"),
                time.strftime("%Y-%m-%d %H:%M:%S")
            ))
            prod_id = cursor.lastrowid
            _add_product_image(cursor, prod_id, "raw", raw_path)
            if has_cutout:
                _add_product_image(cursor, prod_id, "cutout", cutout_path)
        else:
            prod_id = existing_p["id"]
            if has_cutout:
                cursor.execute("SELECT id FROM product_images WHERE product_id = ? AND image_type = 'cutout';", (prod_id,))
                if not cursor.fetchone():
                    _add_product_image(cursor, prod_id, "cutout", cutout_path)
                    cursor.execute("UPDATE products SET cutout_image = ?, status = 'cutout_ready' WHERE id = ?;", (cutout_path, prod_id))

    conn.commit()
    conn.close()


def get_catalog_duplicates(max_hamming_distance: int = 6) -> Dict[str, Any]:
    """
    Identifies all duplicate clusters in the product catalog using perceptual visual hashing (dHash)
    and exact image path matching.
    For each cluster:
      - Designates the optimal primary product to KEEP (evaluating custom naming, metadata, and assets).
      - Designates all other products as DUPLICATES recommended for deletion.
    """
    prods = list_products(limit=2000).get("products", [])
    n = len(prods)
    if n < 2:
        return {"duplicate_groups_count": 0, "total_duplicates": 0, "groups": []}

    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    def norm_p(p):
        return os.path.normcase(os.path.normpath(p)) if p else ""

    # Precompute hashes for raw and cutout images
    prod_meta = []
    for p in prods:
        raw_p = p.get("raw_image")
        cut_p = p.get("cutout_image")
        raw_n = norm_p(raw_p)
        cut_n = norm_p(cut_p)
        dh_r = compute_dhash(raw_p) if raw_p and os.path.exists(raw_p) else ""
        ph_r = compute_phash(raw_p) if raw_p and os.path.exists(raw_p) else ""
        dh_c = compute_dhash(cut_p) if cut_p and os.path.exists(cut_p) else ""
        ph_c = compute_phash(cut_p) if cut_p and os.path.exists(cut_p) else ""
        prod_meta.append({
            "raw_path": raw_n,
            "cutout_path": cut_n,
            "dh_r": dh_r,
            "ph_r": ph_r,
            "dh_c": dh_c,
            "ph_c": ph_c,
        })

    # Compare all pairs
    for i in range(n):
        for j in range(i + 1, n):
            m1 = prod_meta[i]
            m2 = prod_meta[j]

            # 1. Exact raw_image match (case-normalized)
            if m1["raw_path"] and m2["raw_path"] and m1["raw_path"] == m2["raw_path"]:
                union(i, j)
                continue

            # 2. Exact cutout_image match (case-normalized, shared or duplicate cutout asset)
            if m1["cutout_path"] and m2["cutout_path"] and m1["cutout_path"] == m2["cutout_path"]:
                union(i, j)
                continue

            # 3. Cross-format visual hash match on raw photos
            if m1["dh_r"] and m2["dh_r"] and m1["ph_r"] and m2["ph_r"]:
                d_dist = hamming_distance(m1["dh_r"], m2["dh_r"])
                p_dist = hamming_distance(m1["ph_r"], m2["ph_r"])
                if d_dist <= max_hamming_distance or p_dist <= max_hamming_distance:
                    union(i, j)
                    continue

            # 4. Cross-format visual hash match on cutout images
            # Cutouts have clean backgrounds: require BOTH dHash <= 4 and pHash <= 6 to prevent silhouette false positives
            if m1["dh_c"] and m2["dh_c"] and m1["ph_c"] and m2["ph_c"]:
                d_dist = hamming_distance(m1["dh_c"], m2["dh_c"])
                p_dist = hamming_distance(m1["ph_c"], m2["ph_c"])
                if d_dist <= min(4, max_hamming_distance) and p_dist <= min(6, max_hamming_distance):
                    union(i, j)
                    continue

    # Group into clusters
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append(prods[i])

    result_groups = []
    group_idx = 1
    for root, members in clusters.items():
        if len(members) < 2:
            continue

        # Score members to pick primary to KEEP
        def score(p):
            s = 0
            name = (p.get("name") or "").strip()
            # Non-generic name
            if not name.lower().startswith("product ") and name.lower() != "product":
                s += 10
            # Custom SKU
            sku = (p.get("sku") or "").strip()
            if sku and not sku.startswith("SKU-"):
                s += 15
            # Rich metadata
            if p.get("description"): s += 3
            if p.get("tags"): s += 2
            # Assets created
            s += (p.get("angle_count", 0) + p.get("scene_count", 0)) * 5
            # Earlier ID tiebreaker
            s -= p["id"] * 0.001
            return s

        members.sort(key=score, reverse=True)
        keep = members[0]
        duplicates = members[1:]

        # Add visual match percentage relative to keep item
        keep_cut = norm_p(keep.get("cutout_image"))
        keep_raw = norm_p(keep.get("raw_image"))
        keep_ph = compute_phash(keep.get("cutout_image") or keep.get("raw_image"))
        keep_dh = compute_dhash(keep.get("cutout_image") or keep.get("raw_image"))

        enhanced_duplicates = []
        for d in duplicates:
            d_cut = norm_p(d.get("cutout_image"))
            d_raw = norm_p(d.get("raw_image"))

            match_pct = 100
            if keep_cut and d_cut and keep_cut == d_cut:
                match_pct = 100
            elif keep_raw and d_raw and keep_raw == d_raw:
                match_pct = 100
            else:
                d_ph = compute_phash(d.get("cutout_image") or d.get("raw_image"))
                d_dh = compute_dhash(d.get("cutout_image") or d.get("raw_image"))
                dists = []
                if keep_ph and d_ph:
                    dists.append(hamming_distance(keep_ph, d_ph))
                if keep_dh and d_dh:
                    dists.append(hamming_distance(keep_dh, d_dh))
                if dists:
                    lowest_dist = min(dists)
                    match_pct = max(0, round((1.0 - (lowest_dist / 64.0)) * 100))

            enhanced_d = dict(d)
            enhanced_d["similarity_percentage"] = match_pct
            enhanced_duplicates.append(enhanced_d)

        result_groups.append({
            "group_id": f"group_{group_idx}",
            "category_name": keep.get("category_name") or "General",
            "keep": keep,
            "duplicates": enhanced_duplicates,
            "count": len(members)
        })
        group_idx += 1

    total_dups = sum(len(g["duplicates"]) for g in result_groups)
    return {
        "duplicate_groups_count": len(result_groups),
        "total_duplicates": total_dups,
        "groups": result_groups
    }


def cleanup_catalog_duplicates(product_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Deletes specified duplicate product IDs or all identified duplicate products across the catalog in 1 click.
    Safely purges the duplicate database records while keeping the primary products intact.
    """
    if product_ids is None:
        # Collect all duplicates from all groups
        dup_data = get_catalog_duplicates()
        to_delete_ids = []
        for g in dup_data.get("groups", []):
            for d in g.get("duplicates", []):
                to_delete_ids.append(d["id"])
    else:
        to_delete_ids = list(product_ids)

    deleted_count = 0
    deleted_ids = []
    for pid in to_delete_ids:
        if delete_product(pid, remove_files=True):
            deleted_count += 1
            deleted_ids.append(pid)

    return {
        "status": "success",
        "deleted_count": deleted_count,
        "deleted_ids": deleted_ids,
        "message": f"Successfully deleted {deleted_count} duplicate product(s) in 1-click!"
    }


if __name__ == "__main__":
    init_db()
    sync_existing_files_into_db()
    print("Database initialized & existing assets synchronized.")
