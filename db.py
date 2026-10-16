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
from PIL import Image, ImageOps, ImageStat
import numpy as np
try:
    from scipy.fftpack import dct as scipy_dct
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

_PHASH_CACHE = {}
_DHASH_CACHE = {}
_CROP_HASH_CACHE = {}


def get_cached_crop_hashes(p_path: str) -> Tuple[str, str]:
    if not p_path or not os.path.exists(p_path):
        return "", ""
    try:
        mtime = os.path.getmtime(p_path)
        ck = (p_path, mtime)
        if ck in _CROP_HASH_CACHE:
            return _CROP_HASH_CACHE[ck]
        crop = get_foreground_crop(p_path)
        dh = compute_dhash(crop) if crop else ""
        ph = compute_phash(crop) if crop else ""
        _CROP_HASH_CACHE[ck] = (dh, ph)
        return dh, ph
    except Exception:
        return "", ""



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

    # Users Table (authentication)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")

    # Categories Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS categories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        slug TEXT NOT NULL,
        description TEXT DEFAULT '',
        icon TEXT DEFAULT 'tag',
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        UNIQUE(slug, user_id)
    );
    """)

    # Products Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        sku TEXT UNIQUE,
        category_id INTEGER REFERENCES categories(id) ON DELETE SET NULL,
        user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
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

    # --- Migrations: add user_id to existing tables if missing (MUST run before indexes) ---
    try:
        cursor.execute("ALTER TABLE products ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;")
    except Exception:
        pass  # Column already exists
    try:
        cursor.execute("ALTER TABLE categories ADD COLUMN user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;")
    except Exception:
        pass  # Column already exists

    # Create Indexes for fast searches and duplicate lookups
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_category ON products(category_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_user ON products(user_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_products_sku ON products(sku);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_product ON product_images(product_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_md5 ON product_images(md5_hash);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_images_dhash ON product_images(dhash);")

    # Seed standard default categories if table is empty or missing Footwear
    cursor.execute("SELECT id FROM categories WHERE LOWER(name) = 'footwear' LIMIT 1;")
    if not cursor.fetchone():
        now_str = time.strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
        INSERT INTO categories (name, slug, description, icon, created_at, updated_at)
        VALUES ('Footwear', 'footwear', 'Footwear, sneakers, athletic shoes, loafers, boots, heels, clogs, and sandals', 'footprints', ?, ?);
        """, (now_str, now_str))

    conn.commit()
    conn.close()
    ensure_default_users()


def ensure_default_users():
    """Ensures exactly the 3 clean accounts (alex, morgan, vedant) exist in the database with password: password123."""
    conn = get_db_connection()
    cursor = conn.cursor()

    demo_users = [
        ("alex", "alex@studio.com", "Alex Studio"),
        ("morgan", "morgan@catalog.com", "Morgan Catalog"),
        ("vedant", "vedant@studio.com", "Vedant Surve"),
    ]

    now_str = time.strftime("%Y-%m-%d %H:%M:%S")
    demo_hash = None
    for username, email, full_name in demo_users:
        cursor.execute("SELECT id, password_hash FROM users WHERE LOWER(username) = ?;", (username.lower(),))
        existing = cursor.fetchone()
        if not existing:
            if not demo_hash:
                import bcrypt
                demo_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt(10)).decode("utf-8")
            cursor.execute("""
            INSERT INTO users (username, email, password_hash, full_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?);
            """, (username.lower(), email.lower(), demo_hash, full_name, now_str, now_str))
        else:
            # Guarantee password123 works for demo accounts
            curr_hash = existing["password_hash"] if existing else ""
            import bcrypt
            try:
                if not bcrypt.checkpw(b"password123", curr_hash.encode("utf-8")):
                    if not demo_hash:
                        demo_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt(10)).decode("utf-8")
                    cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?;", (demo_hash, existing["id"]))
            except Exception:
                if not demo_hash:
                    demo_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt(10)).decode("utf-8")
                cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?;", (demo_hash, existing["id"]))

    # Prune any other user accounts so only alex, morgan, vedant remain
    cursor.execute("DELETE FROM users WHERE LOWER(username) NOT IN ('alex', 'morgan', 'vedant');")

    # Also assign any unpartitioned products (user_id IS NULL) to demo users
    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'alex';")
    alex_row = cursor.fetchone()
    cursor.execute("SELECT id FROM users WHERE LOWER(username) = 'morgan';")
    morgan_row = cursor.fetchone()

    if alex_row and morgan_row:
        alex_id = alex_row["id"]
        morgan_id = morgan_row["id"]
        cursor.execute("UPDATE products SET user_id = ? WHERE user_id IS NULL AND category_id IN (25, 30, 28, 29, 33);", (alex_id,))
        cursor.execute("UPDATE products SET user_id = ? WHERE user_id IS NULL AND category_id IN (26, 18, 31, 32, 27);", (morgan_id,))
        cursor.execute("UPDATE products SET user_id = ? WHERE user_id IS NULL;", (alex_id,))

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# User CRUD Operations
# ---------------------------------------------------------------------------
def create_user(username: str, email: str, password_hash: str, full_name: str = "") -> Dict[str, Any]:
    """Creates a new user account. Returns the created user dict."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO users (username, email, password_hash, full_name, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?);
    """, (username.strip().lower(), email.strip().lower(), password_hash, full_name.strip(), now, now))
    user_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return get_user_by_id(user_id)


def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Retrieves a user by their primary key."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, username, email, full_name, created_at FROM users WHERE id = ?;", (user_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """Retrieves a full user row (including password_hash) by username, with alias mapping."""
    if not username:
        return None
    uname = username.strip().lower()
    alias_map = {
        "alex_studio": "alex",
        "morgan_catalog": "morgan",
        "vedant_surve": "vedant",
    }
    uname = alias_map.get(uname, uname)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(username) = ?;", (uname,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Retrieves a full user row by email address."""
    if not email:
        return None
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?;", (email.strip().lower(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_suggested_users() -> List[Dict[str, Any]]:
    """
    Returns exactly the 3 users (alex, morgan, vedant) for login suggestions and selection.
    Includes live product counts for each user.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT u.id, u.username, u.email, u.full_name, u.created_at,
           COUNT(p.id) as product_count
    FROM users u
    LEFT JOIN products p ON u.id = p.user_id
    GROUP BY u.id
    ORDER BY 
        CASE 
            WHEN u.username = 'alex' THEN 1
            WHEN u.username = 'morgan' THEN 2
            WHEN u.username = 'vedant' THEN 3
            ELSE 4
        END,
        u.id ASC;
    """)
    rows = cursor.fetchall()
    conn.close()

    users = []
    for r in rows:
        d = dict(r)
        uname = d["username"]
        if uname == "alex":
            field = "Fashion, Beauty & Apparel"
        elif uname == "morgan":
            field = "Tech, Tools & Electronics"
        elif uname == "vedant":
            field = "Studio Director & Admin"
        else:
            field = d.get("full_name") or "Catalog Account"
        d["field"] = field
        d["is_demo"] = uname in ("alex", "morgan", "vedant")
        users.append(d)
    return users



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
    cache_key = None
    if isinstance(img_or_path, str):
        if not os.path.exists(img_or_path):
            return ""
        try:
            mtime = os.path.getmtime(img_or_path)
            cache_key = (img_or_path, mtime, hash_size, highfreq_factor)
            if cache_key in _PHASH_CACHE:
                return _PHASH_CACHE[cache_key]
        except Exception:
            cache_key = None

    try:
        if isinstance(img_or_path, str):
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

        if _HAS_SCIPY:
            arr = np.frombuffer(img_res.tobytes(), dtype=np.uint8).reshape((N, N)).astype(np.float64)
            dct_2d = scipy_dct(scipy_dct(arr, axis=0, type=2), axis=1, type=2)
            dct_low = dct_2d[:hash_size, :hash_size]
            vals = [dct_low[u, v] for u in range(hash_size) for v in range(hash_size) if not (u == 0 and v == 0)]
            median = sorted(vals)[len(vals) // 2]
            bits = [1 if dct_low[u, v] > median else 0 for u in range(hash_size) for v in range(hash_size)]
        else:
            pixels = list(img_res.tobytes())
            matrix = [pixels[i * N:(i + 1) * N] for i in range(N)]
            dct = [[0.0] * hash_size for _ in range(hash_size)]
            for u in range(hash_size):
                for v in range(hash_size):
                    sum_val = 0.0
                    for x in range(N):
                        for y in range(N):
                            sum_val += matrix[x][y] * math.cos(((2 * x + 1) * u * math.pi) / (2.0 * N)) * math.cos(((2 * y + 1) * v * math.pi) / (2.0 * N))
                    dct[u][v] = sum_val
            vals = [dct[u][v] for u in range(hash_size) for v in range(hash_size) if not (u == 0 and v == 0)]
            median = sorted(vals)[len(vals) // 2]
            bits = [1 if dct[u][v] > median else 0 for u in range(hash_size) for v in range(hash_size)]

        val = 0
        for b in bits:
            val = (val << 1) | b
        res_val = f"{val:016x}"
        if cache_key:
            _PHASH_CACHE[cache_key] = res_val
        return res_val
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


def compute_avg_rgb(img_or_path) -> Tuple[float, float, float]:
    """Computes average RGB color across resized 16x16 thumbnail for color verification."""
    if not img_or_path:
        return (0.0, 0.0, 0.0)
    try:
        if isinstance(img_or_path, str):
            if not os.path.exists(img_or_path):
                return (0.0, 0.0, 0.0)
            img = Image.open(img_or_path)
        else:
            img = img_or_path
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            alpha_channel = img.split()[-1]
            background.paste(img, mask=alpha_channel)
            img = background
        else:
            img = img.convert("RGB")
        img_thumb = img.resize((16, 16), Image.Resampling.BILINEAR)
        stat = ImageStat.Stat(img_thumb)
        return (float(stat.mean[0]), float(stat.mean[1]), float(stat.mean[2]))
    except Exception:
        return (0.0, 0.0, 0.0)


def rgb_distance(rgb1: Tuple[float, float, float], rgb2: Tuple[float, float, float]) -> float:
    """Euclidean distance in RGB color space."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(rgb1, rgb2)))


def get_foreground_crop(img_or_path):
    """
    Crops an image tightly to its primary foreground object by eliminating:
      1. Alpha/transparent padding (for cutouts).
      2. Solid white/light neutral studio borders (for staged or cropped plates).
    This allows perceptual visual hashing to match objects regardless of whether
    the image is in a wide canvas or a cropped bounding box.
    """
    if not img_or_path:
        return None
    try:
        if isinstance(img_or_path, str):
            if not os.path.exists(img_or_path):
                return None
            img = Image.open(img_or_path)
        else:
            img = img_or_path

        img = ImageOps.exif_transpose(img)

        # 1. Check alpha channel for transparent PNGs
        if img.mode in ("RGBA", "LA"):
            alpha = img.split()[-1]
            bbox = alpha.getbbox()
            if bbox and (bbox[2] - bbox[0] > 15) and (bbox[3] - bbox[1] > 15):
                return img.crop(bbox)

        # 2. Check solid/light background for RGB images (e.g. white background studio exports)
        corners = [
            img.getpixel((0, 0)),
            img.getpixel((img.width - 1, 0)),
            img.getpixel((0, img.height - 1)),
            img.getpixel((img.width - 1, img.height - 1))
        ]
        avg_corner = sum(sum(c[:3]) if isinstance(c, tuple) else c for c in corners) / (len(corners) * (3 if isinstance(corners[0], tuple) else 1))
        if avg_corner > 220:
            gray = img.convert("L")
            inv = ImageOps.invert(gray)
            mask = inv.point(lambda p: 255 if p > 30 else 0)
            bbox = mask.getbbox()
            if bbox and (bbox[2] - bbox[0] > 20) and (bbox[3] - bbox[1] > 20):
                return img.crop(bbox)

        return img
    except Exception as e:
        return None


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

    # Check direct dHash
    d1 = compute_dhash(img_path1)
    d2 = compute_dhash(img_path2)
    if d1 and d2 and hamming_distance(d1, d2) <= 4:
        return True

    # Check direct pHash
    p1 = compute_phash(img_path1)
    p2 = compute_phash(img_path2)
    if p1 and p2 and hamming_distance(p1, p2) <= 6:
        return True

    # Check foreground cropped hashes
    c1 = get_foreground_crop(img_path1)
    c2 = get_foreground_crop(img_path2)
    if c1 and c2:
        cd1 = compute_dhash(c1)
        cd2 = compute_dhash(c2)
        cp1 = compute_phash(c1)
        cp2 = compute_phash(c2)
        if cd1 and cd2 and cp1 and cp2:
            if hamming_distance(cd1, cd2) <= 4 and hamming_distance(cp1, cp2) <= 6:
                return True

    return False


# ---------------------------------------------------------------------------
# Category CRUD Operations
# ---------------------------------------------------------------------------
def list_categories(user_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """Lists all categories along with their current product count (scoped by user_id if provided)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    if user_id is not None:
        cursor.execute("""
        SELECT c.id, c.name, c.slug, c.description, c.icon, c.created_at, c.updated_at,
               COUNT(p.id) as product_count
        FROM categories c
        LEFT JOIN products p ON c.id = p.category_id AND p.user_id = ?
        GROUP BY c.id
        ORDER BY c.name ASC;
        """, (user_id,))
    else:
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
    status: str = "raw",
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Creates a new product entry in the database."""
    now = time.strftime("%Y-%m-%d %H:%M:%S")

    conn = get_db_connection()
    cursor = conn.cursor()

    if not sku:
        sku = generate_unique_sku(cursor, name)

    # Deduplication Guard: Check if a product with this raw_image already exists (scoped to user)
    if raw_image:
        norm_raw = os.path.normpath(os.path.abspath(raw_image)).lower()
        base_query = "SELECT id, raw_image, cutout_image FROM products WHERE raw_image IS NOT NULL"
        q_params = []
        if user_id is not None:
            base_query += " AND user_id = ?"
            q_params.append(user_id)
        cursor.execute(base_query, q_params)
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
    INSERT INTO products (name, sku, category_id, user_id, description, tags, raw_image, cutout_image, status, created_at, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (name.strip(), sku.strip(), category_id, user_id, description.strip(), tags.strip(), raw_image, cutout_image, status, now, now))
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
    offset: int = 0,
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """Lists products with optional filtering by category, text search, status, or user."""
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

    if user_id is not None:
        query += " AND p.user_id = ?"
        params.append(user_id)

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
def find_duplicate_or_similar_image(file_path: str, max_hamming_distance: int = 3, min_similarity: float = 0.94) -> Optional[Dict[str, Any]]:
    """
    Checks whether the given image matches any existing product image via:
      1. Exact file path (case-normalized)
      2. Exact MD5 hash
      3. Dual-layer perceptual hashing (dHash + pHash) with color verification,
         invariant to image formats (PNG, JPG, WebP, BMP) and compression,
         strictly requiring dual-hash convergence (min_similarity >= 0.94).
    Returns details of the matching product and image if found, else None.
    """
    if not os.path.exists(file_path):
        return None

    target_norm = os.path.normcase(os.path.normpath(file_path))
    target_md5 = compute_md5(file_path)
    target_dhash = compute_dhash(file_path)
    target_phash = compute_phash(file_path)
    target_rgb = compute_avg_rgb(file_path)

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
        cand_md5 = row_dict.get("md5_hash")
        if target_md5 and (cand_md5 == target_md5 or (cand_path and os.path.exists(cand_path) and compute_md5(cand_path) == target_md5)):
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

    # 2. Perceptual Similarity Match via Dual-Hash Convergence (dHash + pHash) & Color Validation
    if target_dhash and target_phash:
        best_match = None
        highest_sim = 0.0
        best_dist = 64

        for row in all_images:
            cand = dict(row)
            cand_path = cand.get("file_path") or ""
            if not cand_path or not os.path.exists(cand_path):
                continue

            cand_dhash = cand.get("dhash") or ""
            if not cand_dhash:
                cand_dhash = compute_dhash(cand_path)
            cand_phash = compute_phash(cand_path)

            if not cand_dhash or not cand_phash:
                continue

            d_dist = hamming_distance(target_dhash, cand_dhash)
            p_dist = hamming_distance(target_phash, cand_phash)

            # Perceptual duplicate criteria: require dual-hash convergence
            is_clone = (
                (d_dist <= 2 and p_dist <= 3) or
                (d_dist == 0 and p_dist <= 4) or
                (p_dist <= 1 and d_dist <= 3) or
                (d_dist <= 3 and p_dist <= 4)
            )

            if not is_clone:
                continue

            # Color verification: prevent silhouette false-positives between different products
            cand_rgb = compute_avg_rgb(cand_path)
            c_dist = rgb_distance(target_rgb, cand_rgb)
            if c_dist > 35.0:
                continue

            sim_d = 1.0 - (d_dist / 64.0)
            sim_p = 1.0 - (p_dist / 64.0)
            combined_sim = round(0.5 * (sim_d + sim_p), 3)

            if combined_sim >= min_similarity and combined_sim > highest_sim:
                highest_sim = combined_sim
                best_dist = max(d_dist, p_dist)
                best_match = cand

        if best_match:
            conn.close()
            return {
                "match_type": "visual_similarity",
                "similarity_score": highest_sim,
                "hamming_distance": best_dist,
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


def get_image_info_quick(path: Optional[str]) -> Dict[str, Any]:
    """Fast metadata extraction: file size and resolution for UI diff display."""
    if not path or not os.path.exists(path):
        return {"size_kb": 0, "width": 0, "height": 0, "resolution": "N/A"}
    try:
        sz_kb = round(os.path.getsize(path) / 1024, 1)
        with Image.open(path) as img:
            w, h = img.size
        return {"size_kb": sz_kb, "width": w, "height": h, "resolution": f"{w}x{h}"}
    except Exception:
        return {"size_kb": 0, "width": 0, "height": 0, "resolution": "N/A"}


def get_catalog_duplicates(
    max_hamming_distance: int = 6,
    user_id: Optional[int] = None,
    sensitivity: Optional[str] = None
) -> Dict[str, Any]:
    """
    Identifies all duplicate clusters in the product catalog using:
      1. Exact cryptographic MD5 file content hash (identical files with different names).
      2. Normalized filesystem paths.
      3. Perceptual visual hashing (dHash + pHash) invariant to format/compression.
      4. Tight foreground-cropped object bounding box matching.

    Supports sensitivity levels:
      - 'strict': Hamming distance <= 3 (exact clones & 95%+ match).
      - 'standard': Hamming distance <= 6 (standard 88%+ match).
      - 'relaxed': Hamming distance <= 10 (similar angles / crops 75%+).

    For each cluster:
      - Automatically designates the highest-quality primary product to KEEP.
      - Provides full side-by-side comparison metrics (resolution, file size, similarity %, match reason).
    """
    if sensitivity == "strict":
        max_hamming_distance = 3
    elif sensitivity == "relaxed":
        max_hamming_distance = 10
    elif sensitivity == "standard":
        max_hamming_distance = 6

    prods = list_products(limit=2000, user_id=user_id).get("products", [])
    n = len(prods)
    if n < 2:
        return {"duplicate_groups_count": 0, "total_duplicates": 0, "groups": [], "sensitivity": sensitivity or "standard"}

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

    # Precompute hashes and metadata for raw and cutout images
    prod_meta = []
    for p in prods:
        raw_p = p.get("raw_image")
        cut_p = p.get("cutout_image")
        raw_n = norm_p(raw_p)
        cut_n = norm_p(cut_p)

        md5_r = compute_md5(raw_p) if raw_p and os.path.exists(raw_p) else ""
        md5_c = compute_md5(cut_p) if cut_p and os.path.exists(cut_p) else ""

        dh_r = compute_dhash(raw_p) if raw_p and os.path.exists(raw_p) else ""
        ph_r = compute_phash(raw_p) if raw_p and os.path.exists(raw_p) else ""
        dh_c = compute_dhash(cut_p) if cut_p and os.path.exists(cut_p) else ""
        ph_c = compute_phash(cut_p) if cut_p and os.path.exists(cut_p) else ""

        cdh_r, cph_r = get_cached_crop_hashes(raw_p) if raw_p and os.path.exists(raw_p) else ("", "")
        cdh_c, cph_c = get_cached_crop_hashes(cut_p) if cut_p and os.path.exists(cut_p) else ("", "")

        prod_meta.append({
            "raw_path": raw_n,
            "cutout_path": cut_n,
            "md5_r": md5_r,
            "md5_c": md5_c,
            "dh_r": dh_r,
            "ph_r": ph_r,
            "dh_c": dh_c,
            "ph_c": ph_c,
            "cdh_r": cdh_r,
            "cph_r": cph_r,
            "cdh_c": cdh_c,
            "cph_c": cph_c,
        })

    # Compare all pairs
    pair_match_details = {}  # (min(i,j), max(i,j)) -> (match_pct, match_reason)
    for i in range(n):
        for j in range(i + 1, n):
            m1 = prod_meta[i]
            m2 = prod_meta[j]
            pair_key = (i, j)

            # 1. Exact MD5 binary file content match
            if m1["md5_r"] and m2["md5_r"] and m1["md5_r"] == m2["md5_r"]:
                union(i, j)
                pair_match_details[pair_key] = (100, "Identical File Content (MD5 Match)")
                continue

            if m1["md5_c"] and m2["md5_c"] and m1["md5_c"] == m2["md5_c"]:
                union(i, j)
                pair_match_details[pair_key] = (100, "Identical Cutout File (MD5 Match)")
                continue

            # 2. Exact filesystem path match
            if m1["raw_path"] and m2["raw_path"] and m1["raw_path"] == m2["raw_path"]:
                union(i, j)
                pair_match_details[pair_key] = (100, "Identical Source Path (100%)")
                continue

            if m1["cutout_path"] and m2["cutout_path"] and m1["cutout_path"] == m2["cutout_path"]:
                union(i, j)
                pair_match_details[pair_key] = (100, "Shared Cutout Mask (100%)")
                continue

            # 3. Cross-format visual hash match on raw photos (strict dual-hash convergence)
            if m1["dh_r"] and m2["dh_r"] and m1["ph_r"] and m2["ph_r"]:
                d_dist = hamming_distance(m1["dh_r"], m2["dh_r"])
                p_dist = hamming_distance(m1["ph_r"], m2["ph_r"])
                is_raw_clone = (
                    (d_dist <= 2 and p_dist <= 3) or
                    (d_dist == 0 and p_dist <= 4) or
                    (p_dist <= 1 and d_dist <= 3) or
                    (d_dist <= 3 and p_dist <= 4)
                )
                if is_raw_clone:
                    worst_dist = max(d_dist, p_dist)
                    union(i, j)
                    pct = max(0, round((1.0 - (worst_dist / 64.0)) * 100))
                    pair_match_details[pair_key] = (pct, f"Visual Clone ({pct}%)")
                    continue

            # 4. Cross-format visual hash match on cutout images
            if m1["dh_c"] and m2["dh_c"] and m1["ph_c"] and m2["ph_c"]:
                d_dist = hamming_distance(m1["dh_c"], m2["dh_c"])
                p_dist = hamming_distance(m1["ph_c"], m2["ph_c"])
                is_cutout_clone = (
                    (d_dist <= 2 and p_dist <= 3) or
                    (d_dist <= 1 and p_dist <= 4) or
                    (d_dist <= 3 and p_dist <= 4)
                )
                if is_cutout_clone:
                    worst_dist = max(d_dist, p_dist)
                    union(i, j)
                    pct = max(0, round((1.0 - (worst_dist / 64.0)) * 100))
                    pair_match_details[pair_key] = (pct, f"Cutout Visual Clone ({pct}%)")
                    continue

            # 5. Cross-format raw vs cutout visual hash match
            cross_pairs = [
                (m1["dh_r"], m1["ph_r"], m2["dh_c"], m2["ph_c"]),
                (m1["dh_c"], m1["ph_c"], m2["dh_r"], m2["ph_r"]),
            ]
            cross_matched = False
            for dh1, ph1, dh2, ph2 in cross_pairs:
                if dh1 and dh2 and ph1 and ph2:
                    d_dist = hamming_distance(dh1, dh2)
                    p_dist = hamming_distance(ph1, ph2)
                    if (d_dist <= 2 and p_dist <= 3) or (d_dist <= 1 and p_dist <= 4):
                        worst_dist = max(d_dist, p_dist)
                        union(i, j)
                        pct = max(0, round((1.0 - (worst_dist / 64.0)) * 100))
                        pair_match_details[pair_key] = (pct, f"Raw/Cutout Hybrid Match ({pct}%)")
                        cross_matched = True
                        break
            if cross_matched:
                continue

            # 6. Foreground-cropped visual hash match (tight object bounds)
            crop_pairs = [
                (m1["cdh_r"], m1["cph_r"], m2["cdh_r"], m2["cph_r"]),
                (m1["cdh_c"], m1["cph_c"], m2["cdh_c"], m2["cph_c"]),
                (m1["cdh_r"], m1["cph_r"], m2["cdh_c"], m2["cph_c"]),
                (m1["cdh_c"], m1["cph_c"], m2["cdh_r"], m2["cph_r"]),
            ]
            for cdh1, cph1, cdh2, cph2 in crop_pairs:
                if cdh1 and cdh2 and cph1 and cph2:
                    d_dist = hamming_distance(cdh1, cdh2)
                    p_dist = hamming_distance(cph1, cph2)
                    lowest_dist = min(d_dist, p_dist)
                    if d_dist <= min(4, max_hamming_distance) and p_dist <= min(6, max_hamming_distance):
                        union(i, j)
                        pct = max(0, round((1.0 - (lowest_dist / 64.0)) * 100))
                        pair_match_details[pair_key] = (pct, f"Foreground Object Match ({pct}%)")
                        break

    # Group into clusters
    clusters = {}
    for i in range(n):
        r = find(i)
        clusters.setdefault(r, []).append((i, prods[i]))

    result_groups = []
    group_idx = 1
    for root, member_pairs in clusters.items():
        if len(member_pairs) < 2:
            continue

        members = [p for _, p in member_pairs]
        idx_map = {p["id"]: idx for idx, p in member_pairs}

        # Scoring function: selects optimal product to KEEP
        def score(p):
            s = 0
            name = (p.get("name") or "").strip()
            if not name.lower().startswith("product ") and name.lower() != "product":
                s += 12
            sku = (p.get("sku") or "").strip()
            if sku and not sku.startswith("SKU-"):
                s += 15
            if p.get("description"): s += 5
            if p.get("tags"): s += 4
            cut_img = p.get("cutout_image")
            if cut_img and os.path.exists(cut_img):
                s += 25
            if p.get("status") == "completed":
                s += 20
            elif p.get("status") == "angles_ready":
                s += 15
            elif p.get("status") == "cutout_ready":
                s += 10
            s += (p.get("angle_count", 0) + p.get("scene_count", 0)) * 8
            s -= p["id"] * 0.001
            return s

        members.sort(key=score, reverse=True)
        keep = dict(members[0])
        duplicates = members[1:]

        # Attach image info to keep item
        k_raw_info = get_image_info_quick(keep.get("raw_image"))
        k_cut_info = get_image_info_quick(keep.get("cutout_image"))
        keep["raw_size_kb"] = k_raw_info["size_kb"]
        keep["cutout_size_kb"] = k_cut_info["size_kb"]
        keep["resolution"] = k_cut_info["resolution"] if k_cut_info["resolution"] != "N/A" else k_raw_info["resolution"]

        keep_idx = idx_map.get(keep["id"])
        enhanced_duplicates = []
        for d in duplicates:
            d_idx = idx_map.get(d["id"])
            pair_k = (min(keep_idx, d_idx), max(keep_idx, d_idx)) if keep_idx is not None and d_idx is not None else None
            
            match_pct, match_reason = pair_match_details.get(pair_k, (100, "Visual Similarity (100%)")) if pair_k else (100, "Visual Similarity (100%)")

            enhanced_d = dict(d)
            d_raw_info = get_image_info_quick(d.get("raw_image"))
            d_cut_info = get_image_info_quick(d.get("cutout_image"))
            enhanced_d["raw_size_kb"] = d_raw_info["size_kb"]
            enhanced_d["cutout_size_kb"] = d_cut_info["size_kb"]
            enhanced_d["resolution"] = d_cut_info["resolution"] if d_cut_info["resolution"] != "N/A" else d_raw_info["resolution"]
            enhanced_d["similarity_percentage"] = match_pct
            enhanced_d["match_reason"] = match_reason
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
        "sensitivity": sensitivity or "standard",
        "threshold_max_distance": max_hamming_distance,
        "groups": result_groups
    }


def smart_merge_duplicates(
    primary_id: int,
    duplicate_ids: List[int],
    user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Consolidates duplicate product assets and metadata into the primary product,
    then safely removes redundant duplicate records.
    Transfers:
      - Cutout masks (if primary is missing one)
      - Generated 3D camera angles
      - Generated commercial scenes
      - Descriptions and unique search tags
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM products WHERE id = ?;", (primary_id,))
    primary_row = cursor.fetchone()
    if not primary_row:
        conn.close()
        return {
            "success": False,
            "status": "error",
            "error": f"Primary product ID {primary_id} not found."
        }

    primary = dict(primary_row)
    inherited_assets = 0
    inherited_features = []

    for dup_id in duplicate_ids:
        if dup_id == primary_id:
            continue
        cursor.execute("SELECT * FROM products WHERE id = ?;", (dup_id,))
        dup_row = cursor.fetchone()
        if not dup_row:
            continue
        dup = dict(dup_row)

        # 1. Inherit cutout mask if primary lacks one
        if not primary.get("cutout_image") and dup.get("cutout_image"):
            if os.path.exists(dup["cutout_image"]):
                cursor.execute(
                    "UPDATE products SET cutout_image = ?, status = 'cutout_ready' WHERE id = ?;",
                    (dup["cutout_image"], primary_id)
                )
                primary["cutout_image"] = dup["cutout_image"]
                primary["status"] = "cutout_ready"
                inherited_assets += 1
                inherited_features.append("Cutout Mask")

        # 2. Inherit description if primary is empty
        if not primary.get("description") and dup.get("description"):
            cursor.execute("UPDATE products SET description = ? WHERE id = ?;", (dup["description"], primary_id))
            primary["description"] = dup["description"]
            inherited_features.append("Product Description")

        # 3. Inherit unique tags
        if dup.get("tags"):
            existing_tags = set(t.strip().lower() for t in (primary.get("tags") or "").split(",") if t.strip())
            new_tags = [t.strip() for t in dup["tags"].split(",") if t.strip() and t.strip().lower() not in existing_tags]
            if new_tags:
                merged_tags = ((primary.get("tags") or "") + (", " if primary.get("tags") else "")) + ", ".join(new_tags)
                cursor.execute("UPDATE products SET tags = ? WHERE id = ?;", (merged_tags, primary_id))
                primary["tags"] = merged_tags
                inherited_features.append(f"{len(new_tags)} Search Tags")

        # 4. Re-parent product images (3D angles & commercial scenes) to primary product
        cursor.execute("SELECT * FROM product_images WHERE product_id = ?;", (dup_id,))
        dup_images = cursor.fetchall()
        for img in dup_images:
            img_dict = dict(img)
            itype = img_dict.get("image_type")
            fpath = img_dict.get("file_path")
            if itype in ("angle", "scene") and fpath and os.path.exists(fpath):
                cursor.execute("SELECT id FROM product_images WHERE product_id = ? AND file_path = ?;", (primary_id, fpath))
                if not cursor.fetchone():
                    cursor.execute("UPDATE product_images SET product_id = ? WHERE id = ?;", (primary_id, img_dict["id"]))
                    inherited_assets += 1

    conn.commit()
    conn.close()

    # 5. Delete duplicate product records safely
    deleted_count = 0
    reclaimed_bytes = 0
    for dup_id in duplicate_ids:
        if dup_id != primary_id:
            if delete_product(dup_id, remove_files=True):
                deleted_count += 1

    updated_primary = get_product(primary_id)
    return {
        "success": True,
        "status": "success",
        "primary_product": updated_primary,
        "primary_id": primary_id,
        "merged_count": deleted_count,
        "merged_ids": [d for d in duplicate_ids if d != primary_id],
        "inherited_assets": inherited_assets,
        "inherited_features": inherited_features,
        "message": f"Successfully merged {deleted_count} duplicate(s) into '{primary['name']}' (inherited {inherited_assets} creative assets)!"
    }


def cleanup_catalog_duplicates(product_ids: Optional[List[int]] = None, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Deletes specified duplicate product IDs or all identified duplicate products across the catalog in 1 click.
    Calculates exact disk storage reclaimed and safely purges redundant records while keeping primary originals intact.
    """
    if product_ids is None:
        dup_data = get_catalog_duplicates(user_id=user_id)
        to_delete_ids = []
        for g in dup_data.get("groups", []):
            for d in g.get("duplicates", []):
                to_delete_ids.append(d["id"])
    else:
        to_delete_ids = list(product_ids)

    reclaimed_bytes = 0
    conn = get_db_connection()
    cursor = conn.cursor()
    for pid in to_delete_ids:
        cursor.execute("SELECT raw_image, cutout_image FROM products WHERE id = ?;", (pid,))
        p_row = cursor.fetchone()
        if p_row:
            for fld in ["raw_image", "cutout_image"]:
                fp = p_row[fld]
                if fp and os.path.exists(fp):
                    cursor.execute(
                        "SELECT COUNT(*) as cnt FROM products WHERE id != ? AND (LOWER(raw_image) = LOWER(?) OR LOWER(cutout_image) = LOWER(?));",
                        (pid, fp, fp)
                    )
                    sh = cursor.fetchone()
                    if not sh or sh["cnt"] == 0:
                        try:
                            reclaimed_bytes += os.path.getsize(fp)
                        except Exception:
                            pass
    conn.close()

    deleted_count = 0
    deleted_ids = []
    for pid in to_delete_ids:
        if delete_product(pid, remove_files=True):
            deleted_count += 1
            deleted_ids.append(pid)

    reclaimed_mb = round(reclaimed_bytes / (1024 * 1024), 2)
    return {
        "status": "success",
        "deleted_count": deleted_count,
        "deleted_ids": deleted_ids,
        "reclaimed_bytes": reclaimed_bytes,
        "reclaimed_mb": reclaimed_mb,
        "message": f"Successfully purged {deleted_count} duplicate product(s) in 1-click! Reclaimed {reclaimed_mb} MB of storage."
    }


if __name__ == "__main__":
    init_db()
    sync_existing_files_into_db()
    print("Database initialized & existing assets synchronized.")
